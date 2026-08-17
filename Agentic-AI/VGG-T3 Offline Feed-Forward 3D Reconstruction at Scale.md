---
source_pdf: VGG-T3 Offline Feed-Forward 3D Reconstruction at Scale.pdf
paper_sha256: 957444fcd232731d557ab7795b5fa8ded67b1b672e8671758cba26f9ccf22dd8
processed_at: '2026-08-13T00:23:04-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VGG-T³ 人话版

## 一句话概括

VGG-T³干的事情就是：**把VGGT里那个"装所有图像信息的内存"从"一个会变大的柜子"换成了"一个固定大小的压缩包"**，代价是每次用的时候要花点时间"解压"，但好处是处理1k张图像从11分钟降到58秒。

---

## 1. 问题出在哪？

VGGT这类model处理多张图像重建3D场景时，核心步骤是**global attention**——让每张图像的每个patch token都能"看到"所有其他图像的token，从而融合多视角信息。

这个操作的计算量是 $O(n^2)$，n是图像数量。原因很直观：n张图像里的每个token都要和其他所有token算一次相似度。

**类比**：想象你有一个会议室，n个人要互相交换名片。每个人都要和其他所有人交换，所以总交换次数是 $n \times (n-1) \approx n^2$。人越多，效率越低。

之前有人试图缓解这个问题：
- **SparseVGGT**: 只跟"附近"的人交换名片（block-sparse attention）
- **FastVGGT**: 把相似的人合并成一组，组内只交换一次（token merging）

但这些方法本质上还是 $O(n^2)$，只是把常数变小了。100人变200人，还是4倍工作量，只是基数小了一点。

---

## 2. 核心Insight：KV就是scene的memory

Paper第一个关键观察是：**global attention的KV space其实就是scene的"记忆库"**。

每张图像经过encoder变成一串token，project成key-value pairs存在那里。当model要预测某张图的深度时，就用那张图的query去这个KV库里"检索"相关信息。

这个KV库的size随n线性增长，而检索一次要扫遍整个库，所以是 $O(n^2)$。

**那能不能把这个"会变大的记忆库"换成一个"固定大小的东西"？**

---

## 3. TTT：把memory塞进MLP weights里

这里引入Sun et al.的Test-Time Training idea。

**原始attention的逻辑**：
- 存一个大的(K,V)对照表
- 给一个query q，通过softmax相似度加权检索出对应的value

**TTT的逻辑**：
- 训练一个小MLP $\mathrm{T}_\theta$，让它学会"看到key k就输出value v"
- 给一个query q，直接让MLP输出 $o = \mathrm{T}_\theta(q)$
- MLP的weights θ就是压缩后的"记忆"

公式说就是：
$$\arg\min_\theta \sum_i \|\mathrm{T}_\theta(k_i) - v_i\|^2$$

**类比**：原来你有一本1000页的phonebook，查名字找电话。现在你把整本phonebook背下来（塞进大脑神经元连接里），虽然背的过程费劲，但以后查询时不用翻书，直接想。

MLP的参数量是固定的（比如1024维的hidden state，参数量几十万），不随n增长。这就是从 $O(n^2)$ 到 $O(n)$ 的根本来源。

---

## 4. 为什么不能naive地替换？

Paper发现直接把attention换成TTT，从pre-trained VGGT初始化后**收敛极慢**。两个原因：

### 4.1 LayerNorm搞乱了输入空间

VGGT原本用LayerNorm做QK normalization稳定attention训练。但LN有可学习的affine参数 $\gamma, \beta$：

$$\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

这些参数会扭曲key的分布，让MLP在test-time训练时面对的K空间跟pre-trained时不一样。

**解决**：删掉LN，用L2 normalization代替。把K和Q都normalize到unit sphere上。这样MLP面对的输入分布是稳定的。

### 4.2 K和V的线性依赖问题（最clever的部分）

这个很数学但很关键。看QKV projection：
$$K = W_k \cdot x, \quad V = W_v \cdot x$$

K和V都是从同一个token x线性变换来的。所以理论上 $V = W_v W_k^{-1} \cdot K$（假设 $W_k$ 可逆）。

这意味着K和V之间的关系是**纯线性的**！MLP只要学一个线性映射就行，这是个trivial solution，根本没有压缩任何scene-specific信息。

**解决：ShortConv2D**

这是paper最聪明的设计。具体操作：
1. 把1D的value sequence reshape回2D image grid $(N, H/p, W/p, d)$
2. 在2D space上做一次3×3 convolution，让每个token的value包含其spatial neighborhood的信息，得到 $V'$
3. flatten回1D

现在MLP要学的mapping变成 $K \to V'$，即**从一个token的local feature预测它周围neighborhood的aggregated feature**。

这个objective不再trivial，因为V'里有了K里没有的spatial context信息。MLP被迫学习真正的geometric reasoning。

**直觉类比**：原来让你学"看到'苹果'这个词就输出'苹果'这个词"——这是identity mapping，没意思。改成"看到'苹果'就输出'红色的水果，长在树上'"——这需要你真正理解"苹果"是什么。ShortConv2D就是把V变成那个"更丰富的描述"。

---

## 5. Test-Time Scaling：训练时1步，推理时2步

VGGT训练时最多24张图像，TTT只做1步optimizer update就够了。但推理时如果要处理1k张图像，信息量是50倍，1步根本不够把这些信息压进MLP。

Paper的实验（Fig. 3a）显示：
- 20张图像（训练分布内）：1步最优
- 1k张图像：需要更多步

实际推理时用2步optimizer update，就能在任意长度序列上达到接近恒定的error。这是个免费的test-time scaling——多花一点点compute换质量。

---

## 6. Distributed Inference：梯度天然可分

TTT的loss是所有token loss的求和：
$$L_{\text{total}} = \sum_i L(k_i, v_i)$$

梯度也是求和：
$$\frac{dL_{\text{total}}}{d\theta} = \sum_i \frac{dL(k_i, v_i)}{d\theta}$$

这意味着可以把图像分成minibatch，分别在不同GPU上算梯度，然后同步MLP weights θ就行。

MLP weights很小（几十万参数），all-to-all通信开销极低。对比VGGT需要ring attention这种复杂的context-parallel实现，TTT的distributed version简单得多。

**单GPU也能跑arbitrary大的collection**：每次只load一个minibatch到GPU算梯度，算完off-load回CPU。VGGT做不到这个，因为它forward pass需要所有QKV同时在GPU memory里。

---

## 7. Visual Localization：白送的feature

重建完成后，MLP weights θ里存的是压缩后的scene。给定一张新query图像：
- 冻结θ
- 跑forward pass
- 在global attention层，只apply frozen MLP到query，不更新θ

这样model就变成单图像Transformer，直接输出query image的pose和geometry。不需要单独的localization pipeline。

实验显示在Wayspots上比TTT3R好很多（translation error 74.45m → 32.04m）。

---

## 8. 实验数据里的关键数字

### Pointmap（Table 1）
- vs TTT3R（同样O(n)）：DTU上error低3.4×，ETH3D低1.8×，NRGBD-D低2.4×
- vs VGGT（O(n²)）：DTU上甚至更好（1.654 vs 1.537），ETH3D和NRGBD有小gap

### Speed（Table 4, 2k images）
- VGG-T³单GPU：230.7s
- VGGT 4 GPU：1590.2s（33×慢）
- TTT3R单GPU：126.2s（但不支持multi-GPU，且精度差）

### Camera Pose（Table 3）
- **这是明确失败的地方**：VGG-T³在pose estimation上比TTT3R差
- 原因：VGGT有专门的camera token，形成heterogeneous modality，MLP难以memorize

### Ablation（Table 6）
- From scratch训练TTT：CD=0.262（很烂）
- Linearization from pre-trained：CD=0.074（好很多）
- + ShortConv2D：CD=0.066（接近softmax attention的0.061）

---

## 9. 一张表总结Trade-off

| 维度 | VGGT | VGG-T³ | TTT3R |
|------|------|--------|------|
| Complexity | O(n²) | O(n) | O(n) |
| 点云精度 | 最好 | 接近VGGT | 差 |
| 速度(1k图) | 11min | 58s | 61s |
| 多GPU | 需ring attention | DDP直接用 | 不支持 |
| 单GPU跑2k图 | OOM | 可以 | 可以 |
| Camera pose | 最好 | 差 | 中等 |
| 序列顺序 | 无序 | 无序 | 必须有序 |
| Visual localization | 不支持 | 支持 | 支持 |

---

## 10. 我的理解

这篇paper最核心的贡献是**重新定义了scene representation**：

- VGGT：scene = variable-size KV pairs（explicit memory，精确但贵）
- VGG-T³：scene = fixed-size MLP weights（compressed memory，便宜但lossy）
- NeRF：scene = MLP weights（但训练慢，per-scene optimization）
- 3DGS：scene = explicit gaussians（快但memory大）

VGG-T³介于VGGT和NeRF之间：保留VGGT的pre-trained prior，用TTT做"轻量级per-scene optimization"，但不像NeRF那样从头训练。

**Limitation很诚实**：
- Camera pose失败说明fixed-size MLP对heterogeneous tokens不够expressive
- Wide-baseline场景不如softmax attention，说明MLP的expressivity ceiling存在

**Future work联想**：
- 给camera token单独设计TTT mechanism
- Adaptive steps based on scene complexity
- Hybrid架构：部分layer用TTT，部分保留attention

---

参考链接：
- VGGT: https://vgg-t.github.io/
- TTT (Sun et al.): https://arxiv.org/abs/2407.04620
- TTT3R (concurrent): https://arxiv.org/abs/2509.26645
- DeepSDF: https://arxiv.org/abs/1901.05103
- Mamba2: https://arxiv.org/abs/2405.21060
- Muon optimizer: https://github.com/KellerJordan/Muon
- LaCT: https://arxiv.org/abs/2505.23884
- Hyena (short conv inspiration): https://arxiv.org/abs/2302.10866
- Linear attention: https://arxiv.org/abs/2006.16236
- LoLCats (LLM linearization): https://arxiv.org/abs/2410.10254
- FlashAttention2: https://arxiv.org/abs/2307.08691
- Ring Attention: https://arxiv.org/abs/2310.01889

---

# VGG-T³: 深度技术解析

这篇paper解决了一个非常实际的问题：feed-forward 3D reconstruction model（如VGGT）的quadratic scaling bottleneck。让我从底层原理开始build你的intuition。

## 1. 核心问题的本质

VGGT这类model的quadratic complexity来源很微妙。表面看是softmax attention的O(n²)，但paper指出**根本原因是variable-length KV space representation**。

为什么这个观察重要？因为它说明：即使你用sparse attention或token merging（如FastVGGT、SparseVGGT），本质上还是O(n²)——只是把常数因子从O(n²)降到O(n/r²)，asymptotic complexity没变。

这个insight让我想到DeepSDF的哲学：用fixed-size decoder + small latent code表示shape。VGG-T³把这个principle搬到了feed-forward multi-view reconstruction里。

参考：
- VGGT paper: https://vgg-t.github.io/
- DeepSDF: https://arxiv.org/abs/1901.05103

## 2. VGGT Architecture 回顾

VGGT的结构是alternating的：
- **Image-wise self-attention**: 每张图像内部token之间attention（complexity与n无关）
- **Global self-attention**: 跨所有图像的token做attention（这是O(n²)的来源）

QKV projection with QK normalization (LayerNorm):
$$q_i = \mathrm{LN}_q(W_q x_i), \quad k_i = \mathrm{LN}_k(W_k x_i), \quad v_i = W_v x_i$$

其中：
- $x_i \in \mathbb{R}^d$: 第i个input token，d是hidden dimension
- $W_q, W_k, W_v \in \mathbb{R}^{d \times d}$: learned projection matrices
- $\mathrm{LN}_q, \mathrm{LN}_k$: LayerNorm with learnable affine params

Softmax attention:
$$o_i = \sum_j \mathrm{softmax}_j\left(\frac{q_i^T k_j}{\sqrt{d}}\right) v_j$$

其中 $\sqrt{d}$ 是scaling factor防止dot product过大导致softmax饱和。

**关键观察**: KV space $\{k_j, v_j\}_{j=1}^{N \times H/p \times W/p}$ 就是scene的implicit representation。N张图像，每张有$H/p \times W/p$个patch token。这个representation的length随N线性增长，而query它需要O(n²)compute。

## 3. TTT: 把Attention重新解释为Learning

Sun et al. [88]的核心insight（这是整个VGG-T³的理论基础）：

softmax attention本质是在学习一个mapping $K \to V$，然后通过query retrieval。那么为什么不显式地学这个mapping？

TTT formulation:
$$\arg\min_\theta \sum_i L_t(\mathrm{T}_\theta(k_i) - v_i)$$
$$o_i = \mathrm{T}_\theta(q_i)$$

变量解释：
- $\theta$: fast weights (MLP的参数)，在train和test time都更新
- $\mathrm{T}_\theta$: 一个small MLP，输入dimension = output dimension = d
- $k_i, v_i, q_i$: 第i个token的key, value, query
- $L_t$: self-supervised loss（这里用dot product loss）

**Intuition**: 
- Softmax attention: 把(K,V)存在一个"dictionary"里，query时做weighted average retrieval
- TTT: 把(K→V)这个mapping压缩到MLP的weights θ里，query时MLP直接输出对应value
- 两者都在做"associative memory"，但TTT用fixed-size memory (θ)，attention用variable-size memory (KV pairs)

这个insight非常深刻，因为它把attention的"软检索"重新frame为"参数化学习"，从而memory cost从O(n)降到O(|θ|)（fixed）。

参考：
- TTT (Learning to Learn at Test Time): https://arxiv.org/abs/2407.04620
- Titans (类似idea): https://arxiv.org/abs/2501.00663

## 4. Linearization的关键挑战

naive地把softmax attention替换为TTT，从pre-trained VGGT初始化，**收敛非常慢**。Paper发现两个关键问题：

### 4.1 LayerNorm的问题

LayerNorm有可学习的affine parameters $\gamma, \beta$:
$$\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

这些参数distort了MLP试图学习的输入空间。**为什么？** 因为MLP要学的是 $K \to V$ mapping，但LN的affine参数会改变K的分布，使得MLP在train time学到的mapping在test time（更新θ时）面对的是distorted的K space。

**解决方案**: 移除LN，用L2 normalization代替：
$$k_i = \frac{W_k x_i}{\|W_k x_i\|_2}, \quad q_i = \frac{W_q x_i}{\|W_q x_i\|_2}$$

这样K和Q都在unit sphere上，分布稳定，TTT能快速收敛。

### 4.2 Trivial Solution问题

这是个非常subtle的数学问题。看QKV projection：
$$K = W_k x, \quad V = W_v x$$

所以 $V = W_v W_k^{-1} K$（如果$W_k$可逆）。这意味着K和V之间的关系是**线性的**！

如果直接优化 $\mathrm{T}_\theta(K) \approx V$，MLP只需要学一个线性映射 $W_v W_k^{-1}$，这是个trivial solution，没有压缩任何scene-specific信息。

**解决方案**: ShortConv2D on V space。这是paper最clever的设计之一。

## 5. ShortConv2D: Non-linear Spatial Mixing

受linear language models里short convolutions启发（Hyena, Mamba2等），但adapt到2D image structure。

**3步实现**：
1. **Reshape**: 把1D token sequence $V \in \mathbb{R}^{(N \cdot H/p \cdot W/p) \times d}$ 重塑成2D grid $(N, H/p, W/p, d)$
2. **Convolve**: 用3×3 2D convolution聚合local neighborhood，得到 $V'$
3. **Flatten**: $V'$ 重塑回1D sequence

**为什么这work**？关键在于打破K和V的线性依赖：
- $K$ 来自单token $x_i$（context-limited）
- $V'$ 来自token neighborhood（context-aware）

现在MLP要学的mapping是 $K \to V'$，即从单token的feature预测其neighborhood的aggregated feature。这强制MLP学习**spatial context aggregation**，而trivial线性映射不再可行。

**与ViT³的关联**: Concurrent work ViT³ [36]也用convolution in inner model，但用于classification。这里的核心都是用spatial structure来增强TTT objective。

参考：
- Hyena: https://arxiv.org/abs/2302.10866
- Gated Delta Networks: https://arxiv.org/abs/2412.06464

## 6. Test-Time Scaling: 序列长度泛化

VGGT训练时最多24张图像，但test时可能要处理1k+图像。paper发现这个gap很大：从N=100到N=1k，reconstruction error增加5×。

**假设**: 训练时只用1个optimizer step（因为24张图像信息量小），但1k图像的信息量是50×大，1个step不足以把这些信息压缩进fixed-size MLP。

**Solution**: test-time增加optimizer steps。Fig. 3a显示：
- 20张图像（in-distribution）: 1 step最优
- 1k图像（out-of-distribution）: 需要更多steps

实际用2个steps，达到almost constant scaling。这有点像test-time scaling in LLMs（DeepSeek-R1的inference-time reasoning），但这里是test-time optimization。

参考：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

## 7. Distributed Inference: 梯度可分解性

这是TTT formulation的另一个beautiful property。看total loss：
$$L_{\mathrm{total}} = \sum_i L(k_i, v_i)$$

它的gradient：
$$\frac{dL_{\mathrm{total}}}{d\theta} = \sum_i \frac{d}{d\theta} L(k_i, v_i) = \sum_s \left(\sum_{i \in s} \frac{d}{d\theta} L(k_i, v_i)\right)$$

其中$s$是minibatch。这意味着：
1. **DDP直接可用**: 把图像shard到不同GPU，各自计算local gradient，all-to-all同步θ
2. **CPU off-loading**: 单GPU也能处理arbitrary大的collection，每次只load一个minibatch到GPU

对比VGGT需要ring attention这种复杂的context-parallel实现，TTT的distributed version简单得多，因为MLP weights很小，通信开销低。

参考：
- LaCT (Test-Time Training Done Right): https://arxiv.org/abs/2505.23884
- Ring Attention: https://arxiv.org/abs/2310.01889

## 8. Visual Localization: 副产品变成feature

重建完成后，θ存的是compressed scene representation。给定新query image：
1. 冻结MLP weights θ
2. 跑standard forward pass
3. 在global attention layer，**只apply frozen MLP**到query features，不更新θ

这把model变成single-image Transformer for query，实现feed-forward visual localization。**不需要explicit mapping step**，因为mapping = TTT optimization。

对比ACEZero [12]需要iterative optimization收敛，VGG-T³只需几次token-space optimization。这是一个unified的mapping + localization framework。

参考：
- ACE (Accelerated Coordinate Encoding): https://arxiv.org/abs/2305.12059

## 9. 实验数据深度分析

### 9.1 Pointmap Estimation (Tab. 1)

| Method | DTU CD↓ | ETH3D CD↓ | NRGBD-D CD↓ |
|--------|---------|-----------|-------------|
| VGGT (O(n²)) | 1.537 | 0.279 | 0.014 |
| SparseVGGT (O(n²)) | 1.541 | 0.327 | 0.018 |
| TTT3R (O(n)) | 5.708 | 0.885 | 0.071 |
| **VGG-T³ (O(n))** | **1.654** | **0.480** | **0.029** |

VGG-T³ vs TTT3R的improvement：
- DTU: 5.708/1.654 ≈ **3.4× lower error**
- ETH3D: 0.885/0.480 ≈ **1.8× lower error**
- NRGBD-D: 0.071/0.029 ≈ **2.4× lower error**

vs VGGT的gap很小（DTU上甚至更好），但复杂度从O(n²)降到O(n)。

### 9.2 大规模scaling (Tab. 4)

| Method | 2k images, 1 GPU | 2k images, 4 GPUs |
|--------|------------------|-------------------|
| TTT3R | 126.2s | N/A (不支持multi-GPU) |
| VGGT | OOM | 1590.2s |
| **VGG-T³** | **230.7s** | **48.5s** |

VGG-T³ 2k images: 230.7s vs VGGT 2k on 2 GPUs: 2827.1s = **12.2× speedup单GPU**
48.5s vs 1590.2s = **33× speedup 4-GPU**

注意VGGT在1 GPU上OOM，必须distributed。VGG-T³单GPU就能跑。

### 9.3 Camera Pose Estimation的失败 (Tab. 3)

这是个honest的limitation。VGG-T³在camera pose上表现不好：
- TUM ATE: VGG-T³ 0.037 vs TTT3R 0.025 vs VGGT 0.012

**原因**: VGGT有专门的camera token，append在image tokens之前进入attention layer，形成heterogeneous "modalities"。MLP很难memorize这种异构结构。这暗示future work需要为不同type的token设计不同的TTT mechanism。

### 9.4 Ablation (Tab. 6) 关键发现

| Variant | CD↓ | NC↑ |
|---------|-----|-----|
| Softmax Attention (upper bound) | 0.061 | 0.844 |
| (i) Scratch (TTT from scratch) | 0.262 | 0.727 |
| (ii) T2R | 0.137 | 0.804 |
| (iii) LoLCats | 0.097 | 0.804 |
| (iv) Ours (linearization from pre-trained) | 0.074 | 0.833 |
| (v) Ours + ShortConv2D | **0.066** | **0.838** |

Insights：
1. **从scratch训练TTT会stuck在local optimum** (i) — linearization from pre-trained model至关重要
2. **VGG-T³的linearization方法 (iv) 显著优于T2R (ii) 和LoLCats (iii)** — 这些是LLM领域的linearization方法
3. **ShortConv2D (v) 进一步缩小gap到softmax attention** — 验证non-linear spatial mixing的必要性

## 10. 训练细节

- **数据集**: Table 7列了 indoor/outdoor/object-centric/mixed datasets，类似VGGT的训练数据
- **Optimizer**: AdamW, lr=1e-4, weight decay=0.05, $\beta_1=0.9, \beta_2=0.95$
- **Inner TTT optimization**: Muon optimizer (5 Newton-Schulz iterations), lr=0.1
- **Cost**: 100k steps on 8×A100-80GB，约VGGT从scratch训练的12%
- **Frozen**: encoder, per-image attention, prediction heads
- **Trainable**: QKV projection, output projection, TTT module parameters

**Muon optimizer很值得注意** — 它是Jordan等人的recent work，专门为hidden layer optimization设计，用Newton-Schulz迭代做orthogonalization。对TTT这种inner-loop optimization特别合适，因为fast weights需要稳定更新。

参考：
- Muon: https://github.com/KellerJordan/Muon
- SwiGLU: https://arxiv.org/abs/2002.05202

## 11. VGGT Baseline的Enhancement (Appendix B)

为了让比较公平，paper还enhance了VGGT baseline：

1. **Memory optimization**: discard unused activations (来自FastVGGT)，单GPU能跑1k images
2. **Context parallel**: 用Ulysses [41]做context parallel，仍用FlashAttention2
3. **Entropy scaling** for long sequences:
$$\lambda' = \lambda \cdot \max(1.0, \log_{N_T} N)$$

其中：
- $\lambda = 1/\sqrt{d}$: 原始scaling
- $N_T = 24 \times (518/14)^2 = 32856$: 训练时最大token数
- $N$: 当前序列的token数

这保证训练时的scaling不变，但对更长序列sharpen attention distribution。Tab. 8显示这个trick显著improve VGGT在large collection上的性能（1000 images CD从0.041降到0.029）。

这是个细节但很重要的baseline engineering，说明paper的comparison是fair的。

## 12. 与相关工作的更广联系

### 12.1 Linear Attention家族

Linear attention (Katharopoulos et al. [49]):
$$o_i = \sum_j \frac{\phi(q_i)^T \phi(k_j)}{\sum_j \phi(q_i)^T \phi(k_j)} v_j$$

通过feature map $\phi$ 让attention可分解为linear recurrence。Mamba [34]是structured的linear attention，gated SSM。

TTT是更general的framework：
- Linear attention: $\mathrm{T}_\theta$ 是linear, $\theta$ 通过closed-form update
- Mamba: $\mathrm{T}_\theta$ 是gated, structured transition
- TTT: $\mathrm{T}_\theta$ 是任意network, $\theta$ 通过gradient descent update

VGG-T³选择TTT因为MLP有更强expressivity，且在3D重建这种需要geometric precision的任务上更合适。

参考：
- Linear Transformers are RNNs: https://arxiv.org/abs/2006.16236
- Mamba2: https://arxiv.org/abs/2405.21060

### 12.2 Online vs Offline Methods

| 类型 | 代表方法 | Complexity | 特点 |
|------|----------|------------|------|
| Offline, quadratic | VGGT, Fast3R, π³ | O(n²) | 全局信息aggregation，精度高 |
| Offline, linear (this) | VGG-T³ | O(n) | global aggregation + linear scaling |
| Online, quadratic | StreamVGGT | O(n²) with KV cache | causal, ordered |
| Online, linear | CUT3R, TTT3R, Point3R | O(n) | autoregressive, fixed-size memory |

VGG-T³是offline method（bidirectional，无序输入）但linear complexity。这是个独特的position：既保留offline的全局aggregation advantage（精度高），又有online方法的scaling property。

### 12.3 Post-training Linearization in LLMs

LLM领域有T2R [48], LoLCats [112], Mamba in Llama [94]等方法把pre-trained transformer转成linear complexity。VGG-T³把这种post-training linearization extend到multi-view 3D reconstruction。

但3D task有独特挑战：
1. **2D spatial structure** → ShortConv2D
2. **Heterogeneous tokens** (image + camera) → camera pose estimation失败
3. **Geometric precision要求** → Muon optimizer + 多个TTT steps

参考：
- LoLCats: https://arxiv.org/abs/2410.10254
- T2R: https://arxiv.org/abs/2103.13076

## 13. 我的Intuition总结

如果让我提炼这篇paper的核心insight：

1. **Attention是parameter-efficient的associative memory，但memory cost随sequence length线性增长**。TTT把memory从"data space"（KV pairs）转移到"parameter space"（MLP weights），用optimization代替检索。

2. **Pre-trained model的linearization比from-scratch training好得多**。这呼应了LLM领域的发现：pre-trained softmax attention学到了rich的feature space，linearization只是改变retrieval mechanism，不需要重新learn features。

3. **打破K-V线性依赖是关键**。ShortConv2D的clever之处在于它利用了2D image的spatial locality，让V'包含context，而K不包含。这强制TTT learn geometric reasoning而不是trivial identity mapping。

4. **梯度可分解性是TTT的副产品**。Attention的梯度也sum over tokens，但attention forward需要所有tokens同时present。TTT的forward不需要，所以梯度可以分布式计算。这是个architectural advantage beyond just complexity。

5. **Test-time scaling是free lunch**。当test sequence比训练时长很多时，增加optimizer steps几乎免费地提升质量。这是TTT相比linear attention的另一个优势——linear attention的update rule是closed-form的，没有"多走几步"的option。

**Limitations值得思考**：
- Camera pose estimation失败说明fixed-size MLP对heterogeneous tokens不够expressive
- Wide-baseline场景表现不如softmax attention
- MLP的expressivity ceiling vs attention的unbounded KV space是fundamental trade-off

**Future direction联想**：
- Adaptive TTT steps based on scene complexity?
- Hybrid: 用TTT for大多数layers，保留少量softmax attention layers for critical global reasoning?
- Per-token-type TTT modules（image tokens vs camera tokens）?
- Hierarchical TTT（local MLP + global MLP）?

这篇paper的contribution不只是一个engineering trick，而是提出了一种**新的scene representation paradigm**：把implicit KV scene memory压缩成explicit parameter space。这跟NeRF（pixel → density）、DeepSDF（latent → SDF）、3DGS（explicit gaussian）是同一个narrative的不同instantiation，都在探索"如何最efficient地represent 3D scene"这个根本问题。

参考paper链接汇总：
- Main paper (VGG-T³): NVIDIA发布，arxiv待更新
- VGGT: https://vgg-t.github.io/
- TTT: https://arxiv.org/abs/2407.04620
- TTT3R (concurrent): https://arxiv.org/abs/2509.26645
- CUT3R: https://arxiv.org/abs/2412.08627
- Mamba2: https://arxiv.org/abs/2405.21060
- Muon: https://github.com/KellerJordan/Muon
- DeepSDF: https://arxiv.org/abs/1901.05103
- Hyena: https://arxiv.org/abs/2302.10866
- Linear Attention: https://arxiv.org/abs/2006.16236
