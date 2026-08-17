---
source_pdf: Sparser, Faster, Lighter Transformer Language Models.pdf
paper_sha256: d62b30ccf9cb21435778315dda0852ab314536e25b7dd0e118cd89c82f0ca2d9
processed_at: '2026-08-12T08:56:46-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej，好，我们抛开学术黑话，用最直白的工程师视角来聊聊这篇 paper 到底在干嘛。

核心 vibe 就是：**我们大家早就知道 LLM 的 FFN 层极度稀疏，但 GPU 就像专门为了在平坦高速上跑大卡车（稠密矩阵乘法）而修的路，如果你试图开一辆稀疏的越野车，高速公路的收费站（overhead）就让你根本快不起来。这篇 paper 帮 GPU 修了一条稀疏专用的高速公路。**

---

### 1. The Big Picture: 为什么我们还没用上 Sparsity？

现在 LLM 里算 FFN，通常是 gated 结构：
$h = \text{ReLU}(xW_g) \odot (xW_u)$，然后 $y = hW_d$。
因为 ReLU 的存在，$h$ 里有 80% 到 90% 都是 0。理论上，既然这些位置是 0，那 $W_u$ 里对应的列和 $W_d$ 里对应的行我根本就不需要算，直接跳过能省一大堆 FLOPs。

但现实很骨感：GPU 的 Tensor Cores 是为 dense tile（比如 16x16 的矩阵块）疯狂设计的。你要跳过 0，就得去判断“谁是 0”、去查非零元素的 index、去 gather/scatter 数据。这些 indirection 带来的 memory traffic 和 kernel launch overhead，往往比老老实实算一遍 dense 矩阵乘法还要慢。这就是所谓的“稀疏悖论”。

### 2. The Simple Recipe: 怎么让模型变得更稀疏？

这篇 paper 用了最简单粗暴的方法：**L1 Regularization**。

公式是 $L_1 \times \frac{1}{L} \sum \sum \sum |h^l[m,n]|$。
变量解释：
- $L_1$: 一个极小的系数（比如 $2 \times 10^{-5}$）
- $l$: 第 $l$ 层
- $m$: 第 $m$ 个 token
- $n$: 第 $n$ 个 hidden neuron

Intuition：ReLU 已经把负数干成 0 了，L1 penalty 就是在 loss 里给所有正数施加一个线性向下的拉力。那些对预测没那么重要的微小激活值，就被这个拉力直接拽到了 0。 

结果极其惊艳：1.5B 的模型，原本一个 token 平均激活 911 个 neurons，加了那么一丢丢 L1 后，平均只激活 29 个 neurons（>99% 稀疏），而且下游任务的 accuracy 几乎没掉（46.4% -> 46.2%）。这侧面证明了 LLM 极度 over-parameterized，大部分 neurons 就是凑数的。

### 3. The Magic Format: TwELL (Tile-wise ELLPACK)

既然有了 99% 的稀疏度，怎么在 GPU 上跑快？传统的稀疏格式叫 ELL（ELLPACK），它把矩阵按“行”对齐，把非零值挤到前面，后面补 0。问题是，现代 GPU 的 matmul kernel 是按 2D tile 切分给不同的 thread block (CTA) 去算的。如果我要在算 $xW_g$ 的同时把结果按 ELL 格式存下来，我得知道整行有多少个非零元素——但一个 CTA 只算了一个小 tile，它根本不知道同一行其他 tile 的情况。要同步？那开销爆炸。

**TwELL 的核心 breakthrough：只在小块内部对齐，不要管整行。**
如果我的 matmul tile 宽度是 $T_n=256$，那我就在这 256 列内部找非零元素，打包存下来。因为所有数据都在一个 CTA 的 local 范围内，不需要跨 block 同步。

更牛的是，他们把这个 sparse conversion 塞到了 matmul kernel 的 **epilogue** 里。当 Tensor Core 算完一个 dense tile，结果还在寄存器/shared memory 里时，threads 直接顺手检查 >0 的元素，用 atomic add 数个数，然后把 16-bit value + 16-bit index 打包写回 DRAM。这就省掉了一个单独的“格式转换 kernel”的 launch 和内存读写时间。

### 4. The Training Hack: Hybrid Format

Inference 爽了，但 Training 怎么办？Training 要保存中间 activation 算 backward。TwELL 有个软肋：它需要预设一个压缩比 $C$，比如 $T_n=256, C=8$，意味着它最多只能存 32 个非零元素。如果某个 token 很特殊，激活了 100 个 neurons，TwELL 就 overflow 溢出了。

实际观察发现，LLM 的 sparsity 非常不均匀（长尾分布）。99% 的 token 只有几个 neurons 激活，但总有些 outlier tokens（比如特定的专有名词如 "Vermont", "formaldehyde"）会激活几百个 neurons。

**Solution: Hybrid Format (混合格式)**
把矩阵动态劈成两半：
- 绝大部分 rows（正常 token）存进 super-compact 的 ELL 格式。
- 少部分 rows（outlier token）溢出了，直接把它们丢进一个 dense backup matrix 里。

Intuition：这就像机场安检。大多数人是普通旅客，走快速通道（只查一两样东西）；少数被标记的 suspicious person 走 full search 通道（全量检查）。这样既保证了整体速度，又不会因为个别人卡死整个队伍。

Backward pass 也是复用这套 hybrid format，算 gradient 时依然跳过那些 0，只在 sparse 的位置做 matmul。这让 training 的 peak memory 暴降 28%，因为原来要存 $M \times N$ 的 dense activation，现在只存了一小块 ELL 加一小块 dense backup。

### 5. 实验数据直觉拆解

看 Table 1 的 2B 模型：
- Forward execution (Inference): 提速 **20.5%**
- Energy per token: 降 **17.0%**
- Training step: 提速 **21.9%**
- Peak memory: 降 **22.3%** (paper 里这里可能有个 typo，前面 1.5B 降了 28.1%，这里写 +22.3%，但总体 memory 收益是巨大的)

几个有意思的 point：
1. **Scale 越大，收益越大**。因为模型越大，FFN 占的总 FLOPs 比重越大。Sparsity 的红利是放大的。
2. **Sparsity 的 Depth 分布**：Paper 发现前 2 层最稀疏，中间层（layer 6 附近）最稠密，后面又变稀疏。中间层为什么稠密？因为 LLM 的 reasoning 和 knowledge retrieval 主要发生在 early-middle layers（Wendler et al. 的研究印证了这点），那里需要大量 neurons 来融合信息。
3. **在消费级显卡上更猛**：他们还在 RTX 6000 Ada 上测了。H100 有超强的 Tensor Cores，跑 dense 本来就快；RTX 6000 的 dense 计算弱一些，但 SM 数量多。结果 sparse kernel 在 RTX 6000 上的相对提速比 H100 还大！这说明 sparsity 能让便宜显卡的算力性价比大幅提升。

### 6. 我的一些联想

- **这本质上跟 FlashAttention 是一个套路**。FlashAttention 是重写 attention 的 memory access，TwELL 是重写 FFN 的 memory access。都是跟 hardware 硬刚，把不优雅的数学操作塞进 GPU 的硬件特性里。
- **跟 Quantization 正交**。Sparsity 砍的是 FLOPs，Quantization 砍的是 bit-width。理论上一个模型可以同时是 99% sparse + INT4 weight，叠加起来收益非常恐怖。
- **跟 MoE 的区别**：MoE 是结构化稀疏，你必须提前定义好比如 8 个 experts，每次走 2 个。那是 routing-level 的稀疏。这篇 paper 做的是 unstructured activation sparsity，是 dynamic 的、每个 token 不同的 pattern。不需要改架构，拿标准 Llama 架构换个 ReLU 加个 L1 就能练。
- **为什么以前没人大规模搞？** 因为写这个 CUDA kernel 太痛苦了。你要在 H100 上写出超越 cuBLAS 的 kernel，还得在 epilogue 里做 atomic packing，还得处理 warp shuffle 和 bank conflicts。这是极高门槛的 systems engineering。所以这 paper 最大的贡献其实是他们**开源了这套 kernel**（https://github.com/SakanaAI/sparser-faster-llms）。

一句话总结：这篇 paper 用“最简单的 L1 trick + 极其硬核的 kernel engineering”证明了，unstructured sparsity 终于从理论上的 free lunch 变成了工程上的 free lunch。如果你要在边缘设备或廉价集群上跑 LLM，这条路现在终于通了。

---

# Sparser, Faster, Lighter Transformer Language Models 深度解读

Hey Andrej, 这篇 Sakana AI 和 NVIDIA 合作的 paper 我读得挺兴奋的，因为它终于把一个老问题——unstructured sparsity 在 GPU 上跑不快——给啃下来了。下面我从 intuition 出发，把技术细节、公式变量、kernel 设计、实验数据都摊开来聊。

---

## 1. 论文核心 motivation: sparsity 的 paradox

LLM 的 feedforward block 占了 2/3 的参数和 80%+ 的 FLOPs（Pope et al., 2023）。ReLU-based FFN 天生 sparse——"lazy neuron" 现象 (Li et al., 2023) 表明每个 token 只激活一小部分 hidden neurons。理论上这是 free lunch，但 GPU 是为 dense compute 优化的，sparse kernel 跑得比 dense 还慢是常态。这篇 paper 的贡献是设计一套 sparse data format + CUDA kernels，让 sparsity 真的能 translate 成 wall-clock speedup，而且训练和推理都能用。

参考链接：
- Sakana AI repo: https://github.com/SakanaAI/sparser-faster-llms
- Lazy Neuron: https://arxiv.org/abs/2210.06313
- ReLU Strikes Back (Mirzadeh et al.): https://arxiv.org/abs/2304.02915

---

## 2. Gated Feedforward Block: 把它当 dynamic key-value memory

### 2.1 公式分解

现代 LLM（Llama、Qwen 系列）的 gated FFN（Shazeer, 2020）由三个 weight matrix 参数化：

$$
W_g \in \mathbb{R}^{K \times N}, \quad W_u \in \mathbb{R}^{K \times N}, \quad W_d \in \mathbb{R}^{N \times K}
$$

变量含义：
- $K$: input/output dimension（model hidden size，比如 2048）
- $N$: expanded hidden dimension（比如 5632，约 8/3 × K，gated 设计；non-gated 经典 transformer 是 4K = 8192）
- $M$: effective batch size over all batched sequences × positions（即所有 token 数）
- $W_g$: gate projection（决定哪些 neurons "open"）
- $W_u$: up projection（产生 value）
- $W_d$: down projection（把 sparse hidden 压回 K 维）

forward 计算：

$$
h_u = xW_u, \quad h_g = \sigma(xW_g), \quad h = h_u \odot h_g, \quad y = hW_d \tag{1}
$$

- $x \in \mathbb{R}^{M \times K}$: input batch
- $h_u, h_g \in \mathbb{R}^{M \times N}$: up / gate activations
- $\sigma$: activation function（这篇用 ReLU，关键！）
- $\odot$: elementwise product（这就是 gating 的来源）
- $h \in \mathbb{R}^{M \times N}$: unified hidden representation
- $y \in \mathbb{R}^{M \times K}$: output

**Intuition**: 把 $W_g$ 的列当作 keys，$W_d$ 的行当作 values。input $x$ 通过 inner product "查询" keys，gate $\sigma(xW_g)$ 决定哪些 memory slots 被激活，然后 $h_u$ 提供检索强度，最后 $W_d$ 把激活的 memory 读出来。这是 Geva et al. (2021) 的 key-value memory 视角的现代版本。

参考：Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories", https://aclanthology.org/2021.emnlp-main.446/

### 2.2 为什么 gating 比 non-gating 更适合 sparsity

gated 架构的 sparsity 由 $h_g$ 决定（gate 决定哪些位置为 0），up projection $h_u$ 即便 dense 也 OK，因为最终 $h = h_u \odot h_g$ 中 $h_g = 0$ 的位置整个被屏蔽。所以理论上你可以只 sparse 化 $h_g$，然后跳过 $W_u$ 和 $W_d$ 的相应列/行。这跟 non-gated 的 ReLU 不太一样——non-gated 必须靠 ReLU(xW_u) 直接产生稀疏，sparsity pattern 跟 $W_u$ 绑死。

Appendix C 给了 non-gated 对比，发现 gated variant 的 speedup 更大（17.9% vs 11.2% @ $L_1=2\times 10^{-5}$），正是因为 gated 能 fuse up & down projections。

---

## 3. L1 Regularization: 极其简单的 sparse 诱导方法

### 3.1 公式

$$
L_1 \times \frac{1}{L} \sum_{l=1}^{L} \frac{1}{MN} \sum_{m=1}^{M}\sum_{n=1}^{N} |h^l[m,n]| \tag{2}
$$

变量：
- $L_1$: 可调 scalar coefficient（这篇扫描了 0 到 $10^{-4}$）
- $L$: 总层数
- $l$: 层 index
- $M$: token 数
- $N$: hidden dim
- $h^l[m,n]$: 第 $l$ 层第 $m$ 个 token 第 $n$ 个 neuron 的 hidden activation（即公式 1 里的 $h$）

把这个加到标准 cross-entropy loss 上即可。**Intuition**: L1 对 $|h|$ 施加线性惩罚，ReLU 已经把负值压成 0，L1 进一步把小正值也压成 0。配合 AdamW 已有的 weight decay，sparsity 几乎免费。

### 3.2 关键 empirical 发现

1.5B 模型（hidden $N=5632$）上：
- 不加 L1 ($L_1=0$): 平均 911 non-zeros（约 16% sparsity，即 84% dense）
- $L_1 = 2\times 10^{-5}$: 平均 29 non-zeros（99.5% sparsity），downstream accuracy 几乎无损（46.4% → 46.2%）
- $L_1 = 3\times 10^{-5}$: 24 non-zeros，但开始掉点（46.4% → 44.83%）
- $L_1 = 10^{-4}$: <1 neuron 平均激活，显著掉点

**Scale 趋势**: 模型越大，自然 sparsity 越高。0.5B 平均 39 non-zeros，2B 平均 24 non-zeros（同样 $L_1 = 2\times 10^{-5}$）。这与 scaling 带来的 redundancy 增加一致。

参考：Mirzadeh et al. 2023 已经在 OPT 上观察过类似现象，但没做 kernel。

---

## 4. TwELL Format: 这是这篇 paper 的灵魂

### 4.1 ELLPACK (ELL) 回顾

ELL 把 $M \times N$ sparse matrix 存成两个 padded matrices $h_\nu, h_I$，size 都是 $M \times N_{nz}$，其中 $N_{nz}$ 是任何 row 的最大 non-zero 数。

- $h_\nu$: non-zero values
- $h_I$: 对应的 column indices
- 每 row 左对齐，右边 pad

ELLPACK-R 再加一个 $h_{nz}$ vector 存每 row 实际 non-zero 数。

ELL 的问题：要算 $h_g = \text{ReLU}(xW_g)$ 并直接输出 ELL 格式，需要先看完整个 row 数 non-zeros、对齐，这要求 cross-CTA sync。现代 tiled matmul kernel 把 output 切成 $T_m \times T_n$ tiles 分散在 CTAs 里，根本没法在一个 kernel 里产生 row-aligned ELL。如果开两个 kernel，转换 overhead 把 sparsity 带来的收益吃光。

### 4.2 TwELL 的核心 idea

不要对齐整个 row，只对齐 **tile 内部**。

把 columns 切成 horizontal 1D tiles，size $T$（比如 $T = T_n = 256$，跟 matmul tile 对齐）。每个 tile 内部按 ELL 格式存：

- $h_\nu \in \mathbb{R}^{M \times N/C}$: 压缩后的 non-zero values
- $h_I \in \mathbb{R}^{M \times N/C}$: 对应 column indices
- $h_{nz} \in \mathbb{R}^{M \times N_T}$: 每个 tile 的 non-zero 计数
- $N_T = \lceil N/T \rceil$: 总 tile 数
- $C$: compression factor，要求 $T/C >$ 任何 tile 的最大 non-zeros，防 overflow

**Intuition**: 在 matmul kernel 内部，每个 CTA 已经只负责一个 output tile $(T_m, T_n)$。这个 tile 的所有 non-zeros 都在 CTA local 范围内，不需要跨 CTA sync。所以同一个 kernel 既能算 $xW_g$，又能把 ReLU 后的 sparse output 直接以 TwELL 格式写回 DRAM。

### 4.3 压缩与 packing 细节

实现里他们把 $h_\nu, h_I, h_{nz}$ 打包到一个 32-bit matrix $\mathbb{R}^{M \times N/\bar{C}}$：
- 第一列存 tile 的 non-zero 计数 $h_{nz}$
- 余下 31 列存 16-bit value + 16-bit index（packed in 32 bits）

这样一次 coalesced load 就能拿到整个 tile 的 metadata + 31 个 (value, index) 对。Compression factor $C=8$ 时，对平均 24-39 non-zeros、$T=256$ 的场景，overflow 概率约 $10^{-34}$（按 Poisson / Chernoff bound 估算），实际跑 10K 步没遇到。

### 4.4 H100-specific 优化

Kernel 用了：
- **TMA (Tensor Memory Accelerator)**: Hopper 引入的异步 bulk copy 单元，做 2D/3D tile load/store
- **WGMMA**: Warp Group Matrix Multiply Accumulate，async tensor core 指令
- **Cluster**: 跨 SM 的协作单元，做 multicast load
- **Hilbert curve tile scheduling**: 最大化 L2 reuse（Chatterjee et al., 1999；Shankhdhar 的 worklog: https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog）

Listing 1 的核心流程：
1. Producer warpgroup 用 TMA 异步 load $A$ tile 和 $B$ tile 到 shared memory
2. Consumer warpgroup 用 WGMMA 做矩阵乘
3. Epilogue 阶段，每个 thread 拿自己的 WGMMA 输出寄存器，检查 $>0$，用 CTA-scope relaxed atomic 在 shared memory 里 atomic-add 累加 non-zero count + 写 packed (value, index)
4. 最后用 TMA 把 packed TwELL 写回 DRAM

这种"在 epilogue 里做 sparse conversion"是关键创新，避开了第二个 kernel。

---

## 5. Inference Kernel: Fused Up & Down Projection

### 5.1 数学形式

把 gate activations 存成 TwELL 后，整个 FFN 剩余计算可以用一个 kernel 完成：

$$
y[m,:] = \sum_{t=0}^{N_T-1}\sum_{c=0}^{h_{nz}[m,t]-1} \underbrace{h_v[m, t \times T_n/C + c]}_{\text{gate value}} \cdot \underbrace{(x[m,:] \cdot W_u[:,n])}_{h_u \text{ element}} \cdot \underbrace{W_d[n,:]}_{\text{down row}} \tag{3}
$$

其中 $n = h_I[m, t \times T_n/C + c]$ 是 non-zero 的 column index。

**Intuition**: 不显式 materialize $h_u$，对每个 active gate neuron $n$：
1. 算 $h_u$ 的第 $n$ 个元素 = $x[m,:] \cdot W_u[:,n]$（dot product）
2. 乘 gate value $h_v$
3. 用这个 scalar scale $W_d$ 的第 $n$ 行（K 维 vector）
4. 累加到 output $y[m,:]$

这样 $h_u$ 永远不落 DRAM，整个 FFN 只用 2 个 kernel launch（gate matmul + fused up/down）。

### 5.2 Kernel 设计

Algorithm 2:
- Grid: 每 CTA = 1 warp，处理 1 个 row $m$
- 双层 for: 外层 static unroll over $N_T$ tiles，内层 dynamic iterate over $h_{nz}[m,t]$ non-zeros
- 每 non-zero: warp collectively load $W_u[:,n]$（K 维）和 $W_d[n,:]$（K 维），做 dot product + scale + accumulate
- Warp-level reduction 用 butterfly shuffle

**为什么 1-warp CTA**: 最大化 grid concurrency，让 L2 cache 在不同 rows 间共享 $W_u, W_d$ 的 column/row。sparse activations 在 input sequence 内有高 correlation（同一 token 在不同 layer 的激活模式相关），所以小 CTA + 大 grid 更容易吃到 L2 hit。

Listing 2 给了完整 CUDA 代码，关键技巧：
- $T_n = 256, C = 8 \Rightarrow$ $T_n/C = 32$ 个 packed entries per tile
- 一个 warp 32 threads 正好一次 coalesced load 拿全 tile
- `__shfl_sync` 在 warp 内 broadcast non-zero count 和 (idx, value) 对
- 用 `__hmul2` 做 bfloat16x2 SIMD 乘法
- `butterfly_stride` reduction: `WARP_SIZE/2 → 1` 的 xor shuffle

---

## 6. Training Kernel: Hybrid Format

### 6.1 为什么不能直接用 TwELL for training

Training 要存 intermediate activations 给 backward pass。TwELL 的 compression factor $C$ 假设了"non-zero count 不会超过 $T/C$"。但 training 中 sparsity 高度 non-uniform:
- 不同 token 的 non-zero 数差几个数量级
- 同一 token 在不同 layer 也差异巨大

如果用 ELL，要 pad 到 max non-zeros，浪费存储；TwELL 同理。

### 6.2 Hybrid 方案

动态 partition rows:
- Sparse rows (non-zeros ≤ $N_{\hat{nz}}$) → aggressive ELL $h_g^s \in \mathbb{R}^{M^s \times N_{\hat{nz}}}$
- Dense overflow rows → dense backup $h_g^d \in \mathbb{R}^{M^d \times N}$
- Binary vector $h_b \in \mathbb{R}^M$ 指示每 row 存哪
- Column index array $h_I \in \mathbb{R}^{M^s \times N_{\hat{nz}}}$

$N_{\hat{nz}}$ 可以比 $N$ 小一个数量级（比如 128 vs 5632），overflow 比例小。

**Intuition**: 大部分 token 高度 sparse（少数 neurons active），少数 "hard" token 触发成百上千 neurons。把这些 outlier 走 dense path，正常 token 走 sparse path，两全。

### 6.3 Forward & Backward

Forward:
- $h_g = \text{ReLU}(xW_g)$ → TwELL → convert to hybrid
- $h_u = xW_u$ for sparse rows: 用 dense-to-hybrid kernel (Listing 5)，只算 $h_g$ active 的那些 columns
- $y = hW_d$: 用 hybrid-to-dense kernel (Listing 6)，sparse rows 走 SpMM，dense rows 走 tensor core dense matmul + mask

Backward (公式 4):
$$
\nabla h_u = \nabla h \odot h_g, \quad \nabla h_g = \nabla h \odot h_u
$$
$$
\nabla W_u = x^\top \nabla h_u, \quad \nabla W_g = x^\top \nabla h_g, \quad \nabla W_d = h^\top \nabla y
$$
$$
\nabla x = \nabla h_u W_u^\top + \nabla h_g W_g^\top
$$

关键：因为 sparsity pattern 已经存在 hybrid format 里，所有 backward matmul 都能用 masked kernel，不用 dense recompute。L1 gradient 直接 inject 到 $\nabla h$ 的对应位置。

### 6.4 Hybrid Transpose Kernel (Listing 7)

Backward 需要transpose sparse matrix（比如 $h^\top \nabla y$）。这个 kernel 用 atomic increment 给 destination row 分配 slot，超过 ELL capacity 就 route 到 dense tail。Dense tail 用 128-bit vectorized scan + zero-skip。

---

## 7. 实验数据细节

### 7.1 Table 1 主结果

| Model | Sparse | Forward (tok/ms) | Energy (mJ/tok) | Training (tok/ms) | Peak Mem (GB) |
|-------|--------|------------------|-----------------|-------------------|---------------|
| 0.5B | ✗ | 410 (0%) | 1.63 (0%) | 97.3 (0%) | 26.2 (0%) |
| 0.5B | ✓ | 480 (+17.0%) | 1.43 (-11.8%) | 95.9 (-1.5%) | 21.2 (-19.2%) |
| 1B | ✗ | 185 | 3.71 | 48.6 | 44.5 |
| 1B | ✓ | 219 (+18.1%) | 3.17 (-14.6%) | 52.1 (+7.1%) | 33.1 (-25.5%) |
| 1.5B | ✗ | 119 | 5.73 | 31.8 | 62.8 |
| 1.5B | ✓ | 141 (+18.8%) | 4.87 (-15.0%) | 35.5 (+11.6%) | 45.1 (-28.1%) |
| 2B | ✗ | 87.8 | 7.85 | 22.4 | 46.7 |
| 2B | ✓ | 106 (+20.5%) | 6.51 (-17.0%) | 27.3 (+21.9%) | 57.1 (+22.3%) |

**Intuition 拆解**:
1. **Forward speedup 随 scale 增长**: 17% → 20.5%。因为大模型 FFN 占比更高，sparsity 收益更大。
2. **Energy saving ~15-17%**: forward 快 + GPU 功耗略降（sparse kernel 利用率不同）。
3. **Training speedup 反而 0.5B 是负的 (-1.5%)**: 小模型 sparse kernel overhead 占比大，dense baseline 用 tensor core 已经很快。1B 以上才正收益。
4. **Peak memory 大减 (19-28%)**: 这是 training 最大收益。Hybrid format 省掉了 $M \times N$ 的 $h_u, h_g$ dense 存储换成 $M^s \times N_{\hat{nz}}$ sparse + 小 dense tail。
5. **2B 那个 +22.3% memory 我推测是 paper 的 typo**（应该是 -22.3% 或者干脆是其他指标），因为 Figure 5 显示 peak memory reduction 24%+，跟 1.5B 的 28% 一脉相承。或者在 2B 这个 scale 上 micro batch 调大了导致绝对内存上升，sparse vs dense 比例倒是降的。这点建议看作者 code 验证。

### 7.2 Sparsity vs Depth (Figure 6)

1.5B 模型，$L_1 = 2\times 10^{-5}$：
- 前 2 层最稀疏（~15 non-zeros）
- 中间层（layer 6 附近）peak（~45 non-zeros）
- 后面慢慢降

**Intuition**: Wendler et al. (2024) 发现 LLM 的 reasoning/knowledge retrieval 集中在 early-middle layers，这跟 sparsity peak 位置吻合。前几层做 token embedding / surface feature，用的 neurons 少；中间层做信息整合，需要更多 neurons；后层做 output projection，又少一些。

Pearson correlation between layer avg non-zeros 和该层 speedup = -0.996。Max non-zeros 影响小（layer 8 例外），说明 kernel 设计对 outlier 鲁棒。

### 7.3 Sparsity vs Token Type (Figure 7)

最低激活 tokens: `doi, nlm, gov, nih, doesn, couldn` — URLs 和 contractions，predictable context。
最高激活 tokens: `loud, enduring, Vermont, Greeks, formaldehyde, ACH` — 高信息密度的 specific entities。

**Position 分布**: 序列前几个 tokens 激活最多，之后指数衰减。这说明 LLM 在 "anchoring" 早期 tokens，把它们 encode 进更丰富的 representation，后续 tokens 用 attention 复用。

### 7.4 Activation Function 对比 (Table 3, Appendix C.1)

1.5B @ 30B tokens:
- ReLU dense: 46.4% acc, 911 nz, 117.1 tok/ms, 5.77 mJ
- SiLU dense: 47.1% acc, 5632 nz (无 sparsity), 116.5 tok/ms, 5.82 mJ
- ReLU sparse ($L_1=2\times10^{-5}$): 46.2% acc, 29 nz, 138 tok/ms (+17.9%), 5.07 mJ (-12.1%)

SiLU 比 ReLU 高 0.7% acc，但没法 sparse 化。用 sparse kernel 后 ReLU 反超 SiLU 在 efficiency 上。这个 trade-off 在生产场景值得掂量。

参考: SiLU paper https://arxiv.org/abs/1606.08415, GLU paper https://arxiv.org/abs/2002.05202

### 7.5 Non-Gated 对比 (Table 4, Appendix C.2)

1.5B @ 30B:
- Gated sparse: 46.20% acc, +17.9% speedup, -12.5% energy
- Non-gated sparse: 46.46% acc, +11.2% speedup, -8.8% energy

Gated 的 speedup 优势主要来自 fused up/down kernel（gated 可以 fuse 因为 $h = h_u \odot h_g$，gate 的 sparsity 决定 $h_u$ 用哪些 column）。Non-gated 必须先算 full $h_u = \text{ReLU}(xW_u)$，sparsity pattern 出现在 $h_u$ 上，up 和 down 不能 fuse。

### 7.6 Dead Neuron Mitigation (Table 5, Appendix C.3)

$L_1 = 2\times 10^{-5}$ 时 ~30% neurons 永久 dead。两种缓解策略：

1. **Sparsity warmup**: 5000 步无 L1 → 5000 步线性升 L1 到 $3\times 10^{-4}$
   - 结果: 45.9% acc（掉），108 non-zeros（多），只 +1.9% speedup
   - 问题: warmup 让模型先 dense 学，后面 sparse 抑制太晚，最终 sparsity 不够

2. **Targeted reinit** (公式 6):
$$
W_g[:,j] \gets (1-\lambda)W_g[:,j] + \lambda \mathcal{N}(0, \sigma^2)
$$
- $\lambda = 0.1$: 插值系数
- $\sigma = 0.02$: 初始化 std
- 只对"一直输出负值"的 dead neurons 重新注入噪声
- 结果: 46.6% acc（升 0.4%），29 non-zeros（保持），+19.1% speedup

**Intuition**: Dead neuron 是 ReLU + L1 的副作用——一旦 neuron 输出长期负，gradient 流不过去，永久死。Targeted reinit 把 weight 拉回 init 附近，给它"重生"机会，类似 continual learning 里的 plasticity injection (Ash & Adams, 2020)。

参考: Ash & Adams, "On Warm-Starting Neural Network Training", https://arxiv.org/abs/1910.08975

### 7.7 RTX 6000 Ada vs H100 (Figure 12, Appendix D.4)

RTX 6000: 188 SMs, 1.59 TB/s mem BW, 无 H100 的高级 tensor core 特性
H100 PCIe: 114 SMs, 2.0 TB/s mem BW

Sparse kernel 在 RTX 6000 上 speedup 更大，因为：
- Dense GEMM 在 RTX 6000 上更慢（缺 H100 tensor core 加速），dense baseline 弱
- Sparse kernels 走 CUDA cores（不走 tensor core），更多 SM 反而占优
- Sparse-to-dense kernel 在 RTX 6000 上 1.34× 快于 H100
- Transpose kernel 2.1× 快

**Intuition**: 这 paper 的方法对"弱硬件"更友好。意味着 sparse LLM 在 edge / consumer GPU 上可能价值更高。

---

## 8. 跟相关工作的对比

### 8.1 vs ReLU Strikes Back (Mirzadeh et al., 2023)
- 他们: 在 OPT（non-gated, GELU→ReLU）上观察 sparsity，理论上分析 speedup，未做 kernel
- 本文: 在现代 gated 架构 (Llama-style) 上做完整 kernel 实现，training + inference 都覆盖

### 8.2 vs TurboSparse (Song et al., 2024)
- 他们: 改 FFN 结构为 dReLU: $h = \text{ReLU}(xW_g) \odot \text{ReLU}(xW_u)$
- 本文: 不改架构，只改 activation (用回 ReLU) + 加 L1

### 8.3 vs ProSparse (Song et al., 2025)
- 他们: finetune pretrained model + manual thresholding
- 本文: from scratch training + L1 reg

### 8.4 vs Q-Sparse (Wang et al., 2024)
- 他们: top-K activation + straight-through estimator，改架构
- 本文: 保留标准 gated FFN，只加 L1

### 8.5 vs Deja Vu (Liu et al., 2023) / CATS (Lee et al., 2024)
- 他们: predict sparsity pattern post-hoc，inference only，memory-bound GEMV
- 本文: training 时就 sparse，general GEMM (compute-bound), batched setting

### 8.6 vs MoE (Shazeer et al., 2017; Fedus et al., 2022)
- MoE: 预定义 expert 数和 sparsity level，input-agnostic routing
- 本文: input-dependent dynamic sparsity，自然涌现

### 8.7 vs Product Key Memory / PEER / UltraMem
- PKM (Lample et al., 2019): https://arxiv.org/abs/1907.05242
- PEER (He, 2024): https://arxiv.org/abs/2407.04153
- UltraMem (Huang et al., 2025): https://arxiv.org/abs/2411.12364
- 这些做 architectural sparsity (固定 pattern 通过 key retrieval)
- 本文做 unstructured sparsity (动态 pattern)，不需要改架构

---

## 9. 我自己的几点思考 & 联想

### 9.1 为什么 L1 在 LLM 上"刚好工作"

small L1 coeff ($2\times 10^{-5}$) 在 $M \cdot N \approx 2^{20} \times 5632$ 的求和上，总 penalty 约几十的量级，相对 cross-entropy loss (≈2) 不大。但它对所有 hidden activations 都施加均匀压力，没有特定 neuron 被偏袒。AdamW 的 weight decay 已经隐式 sparsity 诱导（$|w|$ 衰减），L1 补上"对 activation 而非 weight"的 sparsity 压力。这两者协同。

### 9.2 TwELL 的 32-bit packing 是个 brilliant micro-optimization

把 $h_{nz}$ + 31 (value, index) 对塞进 32 个 32-bit slot，正好一个 warp 32 threads 一次 coalesced load 拿全。这种"让数据结构 match warp size"的设计思路很 CUDA-friendly。可以联想到 warp-level reduction、broadcast 的设计哲学。

### 9.3 Hybrid format 的哲学: "outlier-aware compression"

类似 MiniCPM 的 quantization outlier 处理、MoE 中的 expert imbalance、long-tail activation distribution 都遵循同一思路: 大部分数据走 fast path（压缩 / sparse / 小 expert），少数 outlier 走 slow path（dense / 大 expert）。这是 LLM efficiency 的一个 universal pattern。

### 9.4 跟 Lottery Ticket / Pruning 的关系

unstructured weight pruning (Han et al., 2015) 是 weight-level sparse，inference 时固定 mask。本文是 activation-level sparse，每个 token 动态 mask。两者正交，理论上可以叠加。weight pruning 改 storage + compute；activation sparsity 只改 compute（weights 还 dense），但更适合 dynamic batched inference。

### 9.5 跟 Speculative Decoding 的协同

Sparse FFN 在 draft model（小模型）上 speedup 明显，配合 speculative decoding 时 draft 阶段可以更快 generate candidates。这跟 EAGLE、Medusa 等 spec decoding 方法可能能叠加。

参考: EAGLE https://arxiv.org/abs/2401.15077, Medusa https://arxiv.org/abs/2401.10774

### 9.6 Attention sparsity 的类比

LLM attention 本身也是 sparse 的（local window + global tokens，比如 Longformer、BigBird）。FFN sparsity 跟 attention sparsity 是两个独立 axis。理论上一个 model 可以同时 sparse attention + sparse FFN，double dipping。StreamingLLM、H2O 等 attention sparsity 方法可以叠加。

参考: StreamingLLM https://arxiv.org/abs/2309.17453, H2O https://arxiv.org/abs/2306.14048

### 9.7 跟 SoLM / SmolLM 的关系

SmolLM (Allal et al., 2024) 是小模型 SOTA，本 paper 用 finebweb-edu 训 SmolLM 同源 data。如果 SmolLM 用 sparse training，理论上能在同等 accuracy 下更小更快。

### 9.8 Limitations 我看到的

1. **L1 coeff 需要调**: $2\times 10^{-5}$ 是 1.5B 上推荐值，2B 上更激进可以但 0.5B 上反而保守。Scale-dependent 超参，迁移成本高。
2. **Dead neuron 累积**: 即使 reinit 也只是缓解。长期 training（ continual learning）可能 plasticity 丧失。
3. **TwELL overflow**: $C=8$ 假设 non-zeros < 32，如果某种 distribution shift 让某些 token 触发 100+ neurons，会 drop values。Paper 说"set flag + redo step"，但生产场景要小心。
4. **H100 specific**: TMA、WGMMA、cluster 都是 Hopper 才有。A100 / V100 / 消费卡 上要重新设计 kernel。Appendix D.4 在 RTX 6000 上跑通了，但消费级 RTX 4090 之类没测。
5. **只做 FFN**: Attention 的 QKV projection 还 dense。理论上 attention sparsity 也可以做，但 pattern 不同（attention 是 row-wise sparse，FFN 是 column-wise sparse）。
6. **BFloat16 only**: Kernel 用 `__nv_bfloat16`，FP16 / FP32 没做。混合精度训练常见 FP32 master weight，这里应该 OK，但 inference 端 FP16 量化的模型用不上。

### 9.9 跟 JAX / TPU 的关系

Sakana AI 之前很多 JAX 工作（如 Evolutionary Transformer），这次用 CUDA。TPU 的 sparse support（Mixture-of-Experts、稀疏 matmul）跟 GPU 路径不同。TPU 上 unstructured sparsity 几乎没救（TPU 是为 dense systolic array 设计的）。所以这 paper 思路是 GPU-specific。如果 LLM 训练大量迁移到 TPU，这方法价值会下降。

### 9.10 推测的下一步

作者 future work 提到 "fine-tuning existing dense models via sparsification"。这意味着可以拿 Llama 3、Qwen 2.5 等已开源 dense 模型，用 ProSparse 类方法 finetune 出 sparse 版本，然后用 TwELL kernel 推理。这是 quickest path to production。

另一个方向: 跟量化叠加。Sparse + INT8/INT4 weight quantization 是 orthogonal axis。Sparse kernel 要支持 INT4 weights 才能叠加。

---

## 10. 总结: 这 paper 的真正贡献

我的理解是这 paper 解决了三件事:

1. **Format innovation**: TwELL 让 sparse conversion 在 matmul kernel epilogue 里完成，去掉第二个 kernel 的 launch / sync overhead。这是 engineering 上的精细活。
2. **Hybrid training format**: 动态 partition + ELL + dense tail，robust to non-uniform sparsity。比纯 ELL 实用。
3. **Empirical validation at scale**: 0.5B-2B 上跑完整 chinchilla-optimal training，证明 sparsity 是 viable scaling axis。

它没解决但留下 open question:
- 7B / 70B scale 上还 work 吗（更高 natural sparsity，但训练成本高，paper 没测）
- L1 coeff 怎么自动调
- Attention sparsity 怎么协同
- 更大 model 上的 dead neuron 问题

整体而言这是篇很 "engineering-heavy" 的 paper，理论创新有限，但 kernel 写得很扎实。从 build intuition 角度，最值得记住的是:

**"Sparse 是 free 的，只要你愿意重写 kernel 让它 match 硬件"** — 这跟 flash-attention 的精神一脉相承。dense compute 的 well-optimized 路径是 local minimum，sparse 是另一个 minimum，跨过去需要 kernel-level investment。

参考: FlashAttention https://arxiv.org/abs/2205.14135, CUTLASS https://github.com/NVIDIA/cutlass

---

如果你 (Andrej) 想深入聊某个 kernel 的具体 SASS / PTX 优化，或者想探讨把这个思路推广到 attention / cross-entropy 的可能性，随时告诉我。我对 Listing 1 的 epilogue atomic 那段特别感兴趣——用 CTA-scope relaxed atomic 而不是 warp shuffle reduction 来做 sparse packing 是个不常见的选择，可能有 bank conflict avoidance 的考虑。
