---
source_pdf: Relaxed Recursive Transformers.pdf
paper_sha256: 680a597088a3642bdd1c4eef51f687556c0e10d5ebc0b9f44736241a51cfbfb3
processed_at: '2026-08-11T22:38:53-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，那我们就抛开那些学术黑话，直接从最底层的直觉开始，把这个 paper 的核心 idea 用人话掰碎了揉烂了讲清楚。为了 build your intuition，我尽量用一些物理世界的类比，同时保留必要的公式细节来锚定背后的数学逻辑。

---

## 1. 最直觉的比喻：流水线与“回炉重造”

假设你有一个 18 层楼的超级工厂（比如 Gemma 2B 的 18 个 Transformer layers），原材料进去，一层层加工，最后出成品。
现在你觉得 18 层太贵了，占地太大。怎么办？

**Recursive Transformer 的朴素想法**：我只盖 9 层楼的工厂，但是让原材料**跑两圈**。
第一圈：原材料过 1 到 9 层。
第二圈：把第一圈的输出，**再重新喂回** 1 到 9 层。
这就叫参数共享。因为 1-9 层的机器（weights）和 10-18 层的机器干的活一模一样，那我干嘛还要买 10-18 层的机器？直接让 1-9 层的机器循环用两次不就好了？

**这在数学上就是 Eq. 2**：
$$ \mathbf{h}_t^\ell = f\Big(\mathbf{h}_t^{\ell-1}; \Phi_{((\ell-1) \bmod L/B)+1}'\Big) $$
这里的 $\ell$ 是当前的物理层数（从 1 到 $L=18$），$B$ 是循环次数（这里 $B=2$）。下标 $((\ell-1) \bmod 9)+1$ 的意思就是：当你跑到第 10 层（$\ell=10$）时，算出来的下标是 $((10-1) \bmod 9)+1 = 1$。也就是说，第 10 层直接去拿第 1 层的参数 $\Phi'$ 来用。这就是“循环”。

**但这有个致命问题**：1-9 层的机器在第一圈和第二圈干的活完全一样。可是原本第 1 层和第 10 层虽然相似，但好歹有点区别（比如第 1 层可能做粗加工，第 10 层做精加工）。你现在让同一批机器用完全相同的参数干两遍，模型的表达能力肯定要掉。

---

## 2. Relaxed 的破局之法：给工人配“便宜的工具箱”

为了弥补上面说的能力损失，作者提出了 Relaxed Recursive Transformer。

**人话类比**：我还是只有 1-9 层的工人（共享的基础 weights $\Phi'$ 不变），但是我给每个工人配两个**非常便宜的工具箱**（LoRA 模块 $\Delta\Phi'$）。
第一圈的时候，工人挂上 1 号工具箱干活。
第二圈的时候，工人挂上 2 号工具箱干活。
这样，虽然主体机器是一样的，但凭借不同的小工具，他们在第一圈和第二圈就能干出稍微不一样的活。

**这对应的数学是 Eq. 3**：
$$ \mathbf{h}_t^\ell = f\Big(\mathbf{h}_t^{\ell-1}; \Phi_{((\ell-1) \bmod L/B)+1}', \Delta\Phi_\ell'\Big) $$
这里的 $\Delta\Phi_\ell'$ 就是那层专属的小工具。具体到每个线性层，原本的 $\mathbf{h} = \mathbf{W}'\mathbf{x}$ 变成了 $\mathbf{h} = \mathbf{W}'\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$。
其中 $\mathbf{A} \in \mathbb{R}^{r \times k}$ 是把输入压扁，$\mathbf{B} \in \mathbb{R}^{d \times r}$ 是把压扁的还原回来。$r$ 就是这个工具箱的大小（Rank）。

如果 $r=0$，工具箱是空的，就是纯 Recursive 模型；如果 $r$ 极大，就是完全恢复 18 层的原版模型。所以 Rank 这个旋钮，让你可以在“省内存”和“高精度”之间做平滑的插值。

---

## 3. SVD 初始化的“白话”精髓：抄近道

接下来的核心难点是：我这 9 层的共享机器 $\mathbf{W}'$ 和那两个工具箱 $\mathbf{B}, \mathbf{A}$，一开始该设成什么样？

**朴素的方法**：机器 $\mathbf{W}'$ 用原版 18 层里前 9 层的参数。那对于第 10 层来说，原本的第 10 层参数 $\mathbf{W}_{10}$ 和共享的 $\mathbf{W}'$（其实就是 $\mathbf{W}_1$）之间，差距太大了。你用一个极小 Rank 的 LoRA（比如 $r=64$），根本无法弥补 $\mathbf{W}_{10}$ 和 $\mathbf{W}_1$ 之间巨大的 representation gap。这就是为什么 Stepwise 初始化在 Relaxed 模型里表现很差的原因。

**Average 初始化的直觉**：把原版 18 层里的第 1 层和第 10 层的参数直接平均一下，作为共享底座 $\mathbf{W}'$。
几何直觉上，平均值就是重心。原版的 $\mathbf{W}_1$ 和 $\mathbf{W}_{10}$ 距离这个重心的残差都不大。你用一个小 Rank 的 LoRA 去近似一个“小残差”，自然容易得多。

**SVD 初始化（这步太聪明了）**：既然我知道了共享底座 $\mathbf{W}'$，也知道原版第 $\ell$ 层的真实参数 $\mathbf{W}_\ell$，我要让 LoRA 满足：
$\mathbf{W}_\ell \approx \mathbf{W}' + \mathbf{B}\mathbf{A}$
也就是让 LoRA 去近似残差矩阵 $\Delta \mathbf{W} = \mathbf{W}_\ell - \mathbf{W}'$。
怎么做最省力？对这个残差矩阵做 Truncated SVD（截断奇异值分解）：
$$ \mathbf{U}_r^\ell, \Sigma_r^\ell, \mathbf{V}_r^\ell = \text{Truncated SVD}(\mathbf{W}_\ell - \mathbf{W}'_{(\cdot)}; r) $$
然后把 $\mathbf{B}$ 初始化为 $\mathbf{U}_r \Sigma_r$，把 $\mathbf{A}$ 初始化为 $\mathbf{V}_r^\top$。

**SVD 在干什么？** 残差矩阵里包含了很多信息，SVD 就是帮你把这个残差里“最重要、能量最大”的 $r$ 个方向挑出来。你一开始就把这些最重要的方向塞进 LoRA 里，模型一开始跑起来，输出就已经和原版 18 层模型非常接近了！
根据 Eckart-Young 定理，这是在 Frobenius 范数下对残差最好的低秩近似。所以你不用从头慢慢学，而是直接站在了离原版模型最近的那个低秩子空间上。这就是为什么作者在 Table 3 里展示，仅用 15B tokens 训练，性能就能恢复得这么好。

---

## 4. 部署时的降维打击：Continuous Depth-wise Batching

这篇 paper 最大的贡献其实在于 serving 时的吞吐量。这就要讲到一个非常绝妙的推理调度方法：Continuous Depth-wise Batching (CDB)。

**原本的痛点**：在传统的 18 层模型里，如果你一个 batch 里有 32 个请求，必须所有请求都跑完第 1 层，才能一起进入第 2 层。如果有个请求很简单，跑到第 5 层就已经知道答案了（Early Exit），它能在第 5 层就退出吗？不能。因为它得在第 5 层干等着，或者占着坑不拉屎，等剩下的 31 个请求都跑完第 18 层，这个 batch 才能腾出空位给新来的请求。

**Sequence-wise Batching (CSB, 比如用 vLLM)** 的做法是：在 token 维度上动态插队。某一个请求生成完了最后一个 token，它的坑位立刻被下一个新的请求填补。

**Depth-wise Batching (CDB) 的降维打击**：
在 Recursive Transformer 里，第 1 圈的第 9 层和第 2 圈的第 9 层（即第 18 层），用的是同一套 weights！
这意味着什么？这意味着我不需要强迫整个 batch 同步走完 18 层。
如果 batch 里有 16 个请求正在跑第 1 圈的第 9 层，同时另外 16 个新来的请求，可以直接插进同一个 GPU Kernel，跟它们一起跑第 9 层！只不过前 16 个是在“跑第一圈”，后 16 个是在“跑第二圈”。
如果前 16 个里有 8 个在第 1 圈跑完就 Early Exit 了，那立刻又有 8 个新请求可以插进来，填满 GPU 的 batch。

这就把原本在 sequence 维度的插队，拓展到了 depth 维度的插队。GPU 的计算单元永远处于满载状态，没有任何气泡。这是这篇文章在工程上最震撼的 insight。具体吞吐量数据看 Figure 8，理论上能比原版快 2.66 倍，这非常恐怖。

---

## 5. 脑洞与延伸联想

当你把 Recursive 和 Low-rank 这两件事结合在一起时，可以产生很多极其有意思的联想。

### (1) Latent Reasoning（隐式推理）与 Test-time Compute
现在大家都在讲 test-time compute，比如 OpenAI 的 o1。最直观的做法是生成一长串 "chain of thought" tokens。
但是 Recursive Transformer 提供了另一种思路：在固定的 latent space 里“多想几遍”。你把输入 embedding 喂进去，循环跑 $B$ 次。如果遇到难题，你把 $B$ 从 2 调到 4，它就在 latent space 里多 refine 两次，不需要生成任何中间文本。
这和最近 Geiping 等人做的 "recurrent depth for test-time compute"（https://arxiv.org/abs/2502.05171）以及 Coconut（https://arxiv.org/abs/2412.06769）的思想是高度共鸣的。本质上就是把 transformer 变成了一个可以在隐空间里做不动点迭代的 RNN。因为有了 LoRA 这个 "工具箱"，你每次迭代的微调是不一样的，这就避免了单纯循环导致的表示坍缩。

### (2) 与 MoE (Mixture of Experts) 的几何对偶
传统的 MoE 是在 **宽度** 上做文章：每一层有多个 FFN 专家，根据 token 路由给不同的专家。
Relaxed Recursive Transformer 其实相当于在 **深度** 上做 MoE：同一个底座参数，根据当前所处的循环深度（第 1 圈还是第 2 圈），挂载不同的 LoRA 专家。
如果把每个 loop iteration 的 LoRA 看成一个 expert，那这就是 Depth-wise MoE。未来如果把 MoEUT（https://arxiv.org/abs/2405.16039）和这个结合，在循环体内部用 MoE，外部用 Recursive + LoRA，参数效率可能会指数级提升。

### (3) Model Soup 的几何直觉
为什么 Average initialization 效果最好？这让我想起 model soup（https://arxiv.org/abs/2203.05482）。把多个 fine-tuned 模型的权重平均起来，往往泛化性更好。在这里，对原版 18 层做平均，得到的重心具有最小的平均方差。低秩矩阵去拟合这种小幅度的波动是最容易的。这其实暗示了一种普适的模型压缩范式：**寻找一个 Centroid Model，然后用低秩补丁去逼近各个特定场景的变体。** 这和分布式训练里的 FedAvg 也有异曲同工之妙。

### (4) DEQ (Deep Equilibrium Models) 的联系
DEQ（https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html）是在找一个隐层的不动点 $h^* = f(h^*)$。Recursive Transformer 跑 $B$ 次，其实就是在做不动点迭代的有限步截断。
如果训练的时候用 $B=2$，推理的时候遇到复杂问题能不能用 $B=4$？这就是所谓的 depth generalization。因为底座参数 $\Phi'$ 是共享的，LoRA 也是每层固定的，只要它收敛，增加循环次数就只是增加了不动点迭代的精度。这在 algorithmic reasoning 任务上（比如大数相加、排序）可能有奇效，因为算法本质上就是迭代的。

---

总结一句人话：
这篇 paper 告诉我们，Transformer 太深了本来就是个浪费。你把 18 层砍成 9 层循环跑两圈，只要给这 9 层配上几个用 SVD 精心调教的“小工具箱”，性能几乎不掉。最爽的是，这种“循环”结构让 GPU 在推理时可以在“深度”维度上随意插队填空，吞吐量直接翻倍。以后大模型部署可能真的不需要堆那么多无谓的层了，循环才是王道。

Reference links for intuition building:
- SVD initialization (PiSSA): https://arxiv.org/abs/2404.02948
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- Continuous Depth-wise Compute (MoD): https://arxiv.org/abs/2404.02258
- Latent Reasoning (Coconut): https://arxiv.org/abs/2412.06769

---

# Relaxed Recursive Transformers: 深度技术解析

## 1. 论文核心 idea 与 motivation

这篇 paper 的核心观察非常优雅：**预训练好的 LLM 中，相邻层的权重矩阵往往是高度相似的**，而且很多层在 inference 时是可以 skip 的（Fan et al., 2020; Elhoushi et al., 2024; Zeng et al., 2023）。如果我们把 18 层的 Gemma 2B 压缩成 9 层、循环 2 次（B=2），只需要少量 "uptraining"（15B tokens）就能恢复大部分性能。这与 lottery ticket hypothesis (Frankle & Carbin, 2019) 和 layer skipping 文献的精神相通——大模型其实有大量的冗余 depth。

进一步，作者提出 **Relaxed Recursive Transformer**：把 strict parameter sharing 用 layer-wise LoRA 模块稍加 "放松"。最巧妙的 insight 在于——LoRA 的 rank $r$ 成为一个 hyperparameter，可以在 vanilla（完全独立参数）和 recursive（完全共享参数）两个极端之间做 smooth interpolation。当 $r=0$ 时退化为 Recursive Transformer，当 $r=\text{full}$ 时近似恢复 vanilla model。这是一种 "structural interpolation"。

参考链接：
- LoRA paper: https://openreview.net/forum?id=nZeVKeeFYf9
- Universal Transformer: https://openreview.net/forum?id=HyzdRiR9Y7
- PiSSA (相关的 SVD 初始化工作): https://arxiv.org/abs/2404.02948
- LayerSkip: https://aclanthology.org/2024.acl-long.681

---

## 2. Recursive Transformer 的数学 formulation

### 2.1 Vanilla Transformer 的 forward pass

$$\mathbf{h}_t^\ell = f(\mathbf{h}_t^{\ell-1}; \Phi_\ell), \quad \ell \in [1, L]$$

变量含义：
- $\mathbf{h}_t^\ell \in \mathbb{R}^{d_{\text{model}}}$：第 $\ell$ 层、第 $t$ 个时间步（token position）的 hidden state。上标 $\ell$ 标识 layer，下标 $t$ 标识 sequence 维度。
- $f(\cdot; \Phi_\ell)$：第 $\ell$ 层的变换函数（MHA + FFN + residual + RMSNorm）。Gemma 用的是 GeGLU FFN（Shazeer, 2020）：$\text{FFN}(\mathbf{x}) = \mathbf{W}_\ell^{\text{down}} (\text{GELU}(\mathbf{x}\mathbf{W}_\ell^{\text{gate}}) \odot \mathbf{x}\mathbf{W}_\ell^{\text{up}})$。
- $\Phi_\ell$：第 $\ell$ 层所有 trainable parameter 的集合，包括 attention 的 $W^Q, W^K, W^V, W^{\text{out}}$、FFN 的三个矩阵、RMSNorm 的 scale。
- $L$：总层数。
- $\mathbf{h}_t^0$：token $y_{t-1}$ 的 embedding。

### 2.2 Recursive Transformer (Eq. 2)

$$\mathbf{h}_t^\ell = f\Big(\mathbf{h}_t^{\ell-1}; \Phi_{((\ell-1) \bmod L/B)+1}'\Big), \quad \ell \in [1, L]$$

变量含义：
- $\Phi'$：共享后的参数集合，只有 $K = L/B$ 个 unique blocks。
- $B$：looping block 数量（必须是 $L$ 的因数）。例如 Gemma 2B 有 $L=18$ 层，设 $B=2$，则 $K=9$，前 9 层 unique，后 9 层复用。
- $((\ell-1) \bmod L/B)+1$：循环索引。当 $\ell=1$ 时映射到 1，当 $\ell=10$ 时映射到 $((10-1) \bmod 9)+1 = 1$，于是第 10 层与第 1 层共享权重。
- $\bmod$：取模运算。

这里采用的是 Takase & Kiyono (2023) 提出的 **CYCLE** 策略（与 SEQUENCE、CYCLE(REV) 并列，详见附录 B）。CYCLE 的好处是天然 compatible with early exiting——每个 loop 完成后都可以产出 prediction。

### 2.3 三种初始化策略（这是 paper 的关键 ablation）

如何把一个 unshared 的 pretrained model "投影" 到 shared 的 subspace？作者提出三种方法（Figure 2）：

**(a) Stepwise**：从 pretrained model 中等间隔采样 $K$ 层，固定 first 和 last 层。例如 $L=18, B=2, K=9$，可能选择 $\{1, 3, 5, 7, 9, 11, 13, 15, 18\}$。  
**Insight**：这对应 layer skipping 文献的发现——LLM 对 skip 中间层有 robustness。

**(b) Average**：对所有 $B$ 个应该共享的层取算术平均。即 $\Phi'_k = \frac{1}{B}\sum_{b=1}^B \Phi_{k + (b-1)K}$。  
**Insight**：让 shared weights 成为各 layer 的 "centroid"。

**(c) Lower**：直接用 pretrained model 的前 $K$ 层。  
**Insight**：最 naive 的 baseline。

实验发现（Figure 5）：
- 对于纯 Recursive Transformer，**Stepwise 最好**（loss 最低、few-shot 准确率最高）。这与 layer skipping 的 robustness 直觉一致。
- 对于 Relaxed Recursive Transformer，**Average 最好**。这非常 reasonable——如果我们要用低秩 LoRA 去 approximate 每层相对于 shared weights 的 delta，那么 shared weights 应该是各层的 "中心点"，这样 delta 的 norm 最小，低秩 approximation 最有效。

这个 ablation 给我的 intuition 是：**Recursive 和 Relaxed 是两种不同的 compression paradigm，需要不同的 initialization**。前者是 "select & uptrain"，后者是 "average & refine with low-rank"。

---

## 3. Relaxed Recursive Transformer 与 SVD 初始化

### 3.1 Forward pass (Eq. 3)

$$\mathbf{h}_t^\ell = f\Big(\mathbf{h}_t^{\ell-1}; \Phi_{((\ell-1) \bmod L/B)+1}', \Delta\Phi_\ell'\Big), \quad \ell \in [1, L]$$

- $\Delta\Phi_\ell'$：第 $\ell$ 层的 LoRA 模块参数（每个 loop iteration 有独立的 LoRA）。

对于任意 linear layer $\mathbf{h} = \mathbf{W}'\mathbf{x}$，relaxed 版本变成：

$$\mathbf{h} = \mathbf{W}'\mathbf{x} + \Delta\mathbf{W}'\mathbf{x} = \mathbf{W}'\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$$

- $\mathbf{A} \in \mathbb{R}^{r \times k}$：down-projection，把 $k$ 维输入压到 $r$ 维。
- $\mathbf{B} \in \mathbb{R}^{d \times r}$：up-projection，把 $r$ 维 expand 回 $d$ 维。
- $r$：LoRA rank。$r \ll \min(d, k)$。
- $d$：output 维度，$k$：input 维度。

### 3.2 SVD 初始化的关键公式 (Eq. 4)

$$\mathbf{U}_r^\ell, \Sigma_r^\ell, \mathbf{V}_r^\ell = \text{Truncated SVD}\Big(\mathbf{W}_\ell - \mathbf{W}'_{((\ell-1) \bmod L/B)+1}; r\Big), \quad \ell \in [1, L]$$

变量含义：
- $\mathbf{W}_\ell \in \mathbb{R}^{d \times k}$：原始 pretrained full-size 模型第 $\ell$ 层的权重。
- $\mathbf{W}'_{(\cdot)} \in \mathbb{R}^{d \times k}$：共享（tied）权重，对应第 $\ell$ 层所在的 block。
- $\mathbf{W}_\ell - \mathbf{W}'_{(\cdot)}$：**残差矩阵**——我们要让 LoRA approximate 的目标。
- $\mathbf{U}_r^\ell \in \mathbb{R}^{d \times r}$：top-$r$ 左奇异向量。
- $\Sigma_r^\ell \in \mathbb{R}^{r \times r}$：top-$r$ 奇异值组成的对角矩阵。
- $\mathbf{V}_r^\ell \in \mathbb{R}^{k \times r}$：top-$r$ 右奇异向量。
- $r$：截断 rank，保留前 $r$ 个最大奇异值。

初始化方式：
$$\mathbf{B} = \mathbf{U}_r \Sigma_r, \quad \mathbf{A} = \mathbf{V}_r^\top$$

于是 LoRA 的 output 为 $\mathbf{U}_r \Sigma_r \mathbf{V}_r^\top \mathbf{x}$，根据 Eckart-Young theorem，这是残差矩阵 $\mathbf{W}_\ell - \mathbf{W}'$ 在 Frobenius norm 下的最优 rank-$r$ 近似。

### 3.3 为什么 SVD 初始化重要（Eq. 5）

$$\mathbf{W}\mathbf{x} \approx \mathbf{W}'\mathbf{x} + (\mathbf{U}_r \Sigma_r)(\mathbf{V}_r^\top)\mathbf{x} = \mathbf{W}'\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$$

这个公式揭示了一个非常重要的 property：当 $r$ 足够大时，relaxed 模型可以 **近似完美地 recover vanilla model 的权重**。所以 rank 是一个真实的 interpolation parameter：
- $r=0$：纯 Recursive Transformer，LoRA 不贡献参数。
- $r=\min(d,k)$：完整 SVD，恢复 vanilla model（参数等于 vanilla 减去 shared 部分）。
- 中间 $r$：intermediate compression ratio。

这个 framing 非常优雅，把 LoRA 的角色从 "finetuning 工具" 提升到了 "structural interpolation mechanism"。

### 3.4 与 PiSSA 的对比

PiSSA (Meng et al., 2024) 也用 SVD 初始化 LoRA，但策略相反：PiSSA 对 $\mathbf{W}$ 本身做 SVD，把 principal components（大奇异值）作为 trainable LoRA，其余作为 frozen residual。这样 trainable 部分能 capture 大部分信息。

本文的方法是对**残差** $\mathbf{W} - \mathbf{W}'$ 做 SVD，目标是 recover original $\mathbf{W}$。两者都是 "low-rank 信息集中在 principal components" 这一思想的应用，但应用场景不同：
- PiSSA：参数高效 finetuning，frozen base + trainable principal LoRA。
- 本文：参数压缩，shared base + depth-specific LoRA 做 relaxation。

### 3.5 Stepwise + SVD 的失败模式

一个很有趣的 ablation finding（Figure 6, Figure G.2）：用 Stepwise 初始化 shared weights，再加 SVD 初始化 LoRA，**有时比不加 LoRA 还差**。原因：Stepwise 让 shared weights 等于 pretrained 模型的某些具体层（如 layer 1, 3, 5...），那么 layer 2 相对 shared weights 的 residual 是一个 "完全不同的层"，低秩近似无法 capture 这种完全不同的 representation。这正是 Average 初始化的优势——它是 "centroid"，所有层相对它的 residual 都比较小，且谱集中度高。

这个 finding 给我的 intuition 是：**LoRA 的 effectiveness 依赖于 base model 与 target 之间的距离小且低秩**。这也是为什么 LoRA 适合 finetuning（base 和 target 差异小），但不适合 from-scratch training（差异大）。

---

## 4. Continuous Depth-wise Batching (CDB)

### 4.1 与 Continuous Sequence-wise Batching (CSB) 的对比

CSB（vLLM, Orca, Kwon et al., 2023; Yu et al., 2022）的核心 insight 是：不同 sequence 的不同 token 在 decode 时是独立的，所以一个 sequence 跑完可以立即被另一个 sequence 替换。这是沿 **sequence 维度** 的 dynamic batching。

CDB 的核心 insight 是：在 Recursive Transformer 中，不同 sample 可以同时处于 **不同的 loop iteration**。这是沿 **depth 维度** 的 dynamic batching。

### 4.2 为什么这只有 Recursive Transformer 能做

vanilla Transformer 中，第 5 层和第 10 层的 weights 不同，所以一个 batch 里所有 sample 必须同时处于同一层。但 Recursive Transformer 中，第 5 层和第 14 层（如果 $K=9$）的 weights 相同，所以 batch 里可以有 sample A 在第 1 个 loop 的第 5 层，同时 sample B 在第 2 个 loop 的第 5 层——它们共用一个 GPU kernel。

### 4.3 Early Exiting 的配合

结合 early exiting（Bae et al., 2023; Schuster et al., 2022; Elbayad et al., 2020），CDB 的威力完全释放。Figure 3(c) 的例子：batch 中有些 sample 在 loop 1 后就 exit，腾出的 slot 立即被新到的 sample 填充。这避免了 vanilla Transformer 中 early exit 的 synchronization 问题——通常 vanilla 中即使一个 token 提前 exit，它也得等 batch 中其他 token 跑完所有层。

---

## 5. Early-Exit Training 的策略

### 5.1 Loss function

朴素方法（Schuster et al., 2022; Bae et al., 2023）用 weighted CE：

$$\mathcal{L} = \sum_{i=1}^B \alpha_i \mathcal{L}_i, \quad \alpha_i = \frac{i}{\sum_j j}$$

但作者发现这 **overemphasize** 中间 outputs，损害 final performance。他们提出 **aggressive coefficient**：

$$\alpha_i = 0.1 \text{ for } i < B, \quad \alpha_B = 1$$

加上 self-distillation：intermediate output 从 detached final output 做 KD loss。

### 5.2 关键 ablation（Table 3）

- Weighted CE 会让 final accuracy 下降 1.2%（51.7 → 50.5）。
- Aggressive(0.1) + KD 让 final accuracy 提升 0.9%（51.7 → 52.6），同时中间 loop 的 accuracy 达到 48.0%。
- 这意味着仅用 9 层（half depth）就能达到接近 18 层 90%+ 的准确率。

---

## 6. 主要实验结果解读

### 6.1 Figure 4 的核心 message

三个 model（Gemma, TinyLlama, Pythia）的 plot 展示：
- **Reduced-size baseline**（同等参数、from scratch + KD）：最差。
- **Recursive model**（Stepwise init）：显著好于 reduced-size baseline。
- **Relaxed model**（Average init + SVD LoRA）：随 rank 增加单调提升，rank=512 时接近 full-size。

对 Gemma 2B（uptrained on 60B tokens + KD）：
- Full-size: 58.6%
- Relaxed (r=512): 58.4%（差距仅 0.2%！）
- Recursive (r=0): 54.0%
- Reduced-size: ~44%

### 6.2 Recursive Gemma 1B 超越 vanilla TinyLlama 和 Pythia 1B

这可能是 paper 最 striking 的结果。Recursive Gemma 1B（用 Stepwise init，uptrain 15B tokens）的 average accuracy 是 51.7%，而：
- TinyLlama 1.1B（pretrained on 105B tokens）：43.3%
- Pythia 1B（pretrained on 300B tokens）：48.8%

Insight：**leverage 优秀的 pretrained weights 比增加 training tokens 更重要**。Gemma 2B 是 trained on 3T tokens，其 9 层的 representation 比 TinyLlama 22 层的 representation 信息密度更高。

### 6.3 Distribution shift 的 challenge（Table 2）

uptraining 用 SlimPajama，但 Gemma 的 pretraining data 是 unreleased（可能是更 curated 的 data）。结果是：uptraining 反而让 Gemma 的 few-shot 性能下降（61.7 → 60.1 → 58.6，随 tokens 增加）。这说明 paper 的方法在 distribution shift 场景下有 "ceiling"——能接近 "uptrained full-size model"，但未必能达到 "original pretrained model" 的水平。

### 6.4 Throughput 数据（Figure 8, Table K.2）

vanilla Gemma 2B（CSB）：1528 tokens/sec（baseline）  
Recursive Gemma 1B + CDB + early-exit：2877 tokens/sec（**2.66× vanilla，1.88× CSB**）  
Relaxed (r=512)：1719 tokens/sec（1.59× vanilla）

更 impressive 的是：recursive Gemma 在 throughput 上比 vanilla Pythia 1B 快接近 4×（2877 vs 822），同时性能更好（54.0 vs 49.3）。

---

## 7. 我对这篇 paper 的 critical 思考与延伸联想

### 7.1 与 Deep Equilibrium Models (DEQ) 的联系

DEQ (Bai et al., 2019) 把 deep network 看作 implicit fixed-point iteration：找到 $\mathbf{h}^*$ 使得 $\mathbf{h}^* = f(\mathbf{h}^*; \Phi)$。Recursive Transformer 是 explicit iteration（固定 $B$ 次），DEQ 是 implicit iteration（用 root finding 求解）。如果把 $B$ 趋向无穷，Recursive Transformer 在某种意义上是 DEQ 的离散化。这引出一个问题：能不能用 DEQ 的 Anderson acceleration 来加速 convergence？

### 7.2 与 latent reasoning / pause tokens 的联系

paper 在 future work 提到 "Latent Reasoning via Recurrent Depth"——这正好对应最近的工作如 Coconut (Hao et al., 2024)、pause tokens (Goyal et al., 2024)、 recurrent depth for test-time compute (Geiping et al., 2025)。Recursive Transformer 把 depth 当作 "thinking steps"，这可能比 chain-of-thought 更高效——后者要 expand 到 discrete tokens，前者在 continuous latent space 做 iterative refinement。

### 7.3 与 Mixture of Experts (MoE) 的结合

paper future work 提到 "incorporating MoE within looped blocks"。一个有趣的设计：让每个 loop iteration 的 LoRA 选择不同的 expert，这相当于 "depth-wise MoE"。这比 standard MoE（每层多个 expert）多了一个 dimension——**同一个 expert 可以被 reuse 在不同 depth**。这与 MoEUT (Csordás et al., 2024) 的方向一致。

### 7.4 与 SSIE / YOCO 的联系

YOCO (Sun et al., 2024) 提出 "You Only Cache Once"——只在 decoder 的前半部分计算 KV cache，后半部分共享。这与 Recursive Transformer 有精神上的相似——都是通过 architecture 让某些计算 "shared"。两者的结合很值得探索：recursive + cross-layer attention，可能让 KV cache 进一步压缩。

### 7.5 LoRA rank 的 "Pareto frontier" 与 PiSSA 的最优 rank

Figure 7(a) 显示 rank 越大性能越好，但有 diminishing returns。这与 PiSSA 的发现一致——principal singular values 集中了大部分信息。一个自然的问题：**不同 layer 的最优 rank 是否相同？** Table G.1 的 ablation 给出答案：FFN 的 rank 影响最大（因为 hidden dim 大），attention 的 Q/K/V rank 影响小（因为 KV cache 本来就 shared）。这暗示一个 "structured rank allocation" 策略——FFN 用大 rank，attention 用小 rank。

### 7.6 关于 "weight averaging" 的几何 intuition

为什么 Average init 在 relaxed model 中最好？这让我想起 model soups (Wortsman et al., 2022)——多个 fine-tuned model 的 weight average 往往 outperform 单个 model。这里 Average init 相当于 "depth-wise model soup"——把不同 depth 的 weights 平均后，每个 depth 用 LoRA recover 自己的 specificity。这与 multi-task learning 中 "shared trunk + task-specific head" 的设计哲学一致。

### 7.7 与 SSM (Mamba, Zamba) 的结合

paper future work 提到 "incorporating SSM components within looped blocks"。Zamba (Glorioso et al., 2024) 已经做了 hybrid SSM-Transformer。一个有趣的 question：如果 looped block 是 SSM，那 recursive 就变成 "RNN with shared transition matrix"。这可能是 parameter-efficient RNN 的一个新方向——把 SSM 看作 "infinite-depth recursive" 的特例。

### 7.8 关于 "depth generalization" 的思考

Looped transformer 在算法任务上有 length generalization 的能力（Fan et al., 2024; Schwarzschild et al., 2021）——train on short sequences, test on long。Recursive Transformer 天然支持 "depth generalization"：在训练时用 $B=2$，inference 时用 $B=4$（多 loop 两次）。这能否用于 test-time compute scaling？这与 Geiping et al. (2025) 的 "recurrent depth for test-time compute" 直接呼应。

### 7.9 一个 "naive but interesting" 的 extension

如果每个 loop iteration 用不同的 LoRA，那 $B$ 个 LoRA 模块可以看作 $B$ 个 "agent" 协作完成任务。能不能让它们 specialize 成不同的 capability（如一个做 reasoning，一个做 fact recall）？这相当于 "depth-wise mixture of experts"，但 expert 数量等于 loop 数。

---

## 8. Limitations 与 open questions

paper 自己坦诚地讨论了几个 limitations：

1. **Oracle exiting**：throughput 数据是基于 oracle（假设我们能完美知道何时 exit），实际 confidence-based early exit 会有 performance loss。
2. **KV cache for exited tokens**：early exit 后，后续 layers 仍需要这些 token 的 KV cache 用于后续 sequence generation。这需要 "backfill" 计算，paper 声称这部分 overhead 在 memory-bound 场景下可忽略，但在 compute-bound 场景会显著。
3. **LoRA serving efficiency**：多个 LoRA 模块的 batched inference 比较麻烦，paper 用了 "concatenate into single big LoRA" 的 hack，但引入冗余计算。需要 CUDA kernel 优化（参考 Punica, S-LoRA, Sheng et al., 2023）。
4. **Distribution shift**：uptraining data 和 pretraining data 不同会引入性能 ceiling。

我会再补充几个：
5. **Layer-wise 学习率 / gradient flow**：同一组 weights 被 loop B 次，gradient 是 B 个 loop 的 sum。这可能引起 gradient explosion/vanishing，paper 没讨论 training stability。
6. **Expressivity upper bound**：固定 $K$ 层 + $B$ 次 loop，模型 expressivity 由 $K$ 决定，$B$ 只增加 "refinement steps"。这对需要 $L$ 个 distinct computation 的任务（如 algorithmic reasoning 中每层做一步）可能 fundamentally 不足。
7. **Memory bandwidth bottleneck**：recursive 减少了 parameter memory，但 activation memory 不变。对长 context generation，KV cache memory 仍是 bottleneck——除非结合 YOCO 或 cross-layer attention。

---

## 9. 总结：这篇 paper 给我的核心 takeaways

1. **Architecture constraint 可以变成 advantage**：参数共享本是 compression 的 "loss"，但 recursive structure 反而 enable 了 CDB 这个新的 serving paradigm。
2. **"Interpolation via rank" 是个 powerful framing**：把 LoRA 从 "finetuning tool" 提升为 "interpolation mechanism between two architectures"，这个 abstraction 很有启发性。
3. **Initialization > Training**：从 pretrained model 出发，15B tokens 就能 recover 大部分性能。这印证了 "pretrained weights 是金子" 的直觉。
4. **Centroid initialization 是低秩近似的 key**：Average init 在 relaxed model 中最好，因为 centroid 让 residual 的 norm 最小。这个 insight 可以推广到其他 "shared base + low-rank delta" 的场景。
5. **Serving 与 architecture 是 co-design**：这篇 paper 的最终 contribution 不只是 accuracy，而是 "architecture enables new serving pattern"——这种 co-design 思维是 LLM 时代的新范式。

参考链接：
- vLLM / PagedAttention: https://dl.acm.org/doi/10.1145/3600006.3613165
- Orca: https://www.usenix.org/conference/osdi22/presentation/yu
- Confident Adaptive Language Modeling: https://papers.nips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html
- Deep Equilibrium Models: https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html
- Coconut (latent reasoning): https://arxiv.org/abs/2412.06769
- Recurrent depth for test-time compute: https://arxiv.org/abs/2502.05171
- PiSSA: https://arxiv.org/abs/2404.02948
- MoEUT: https://arxiv.org/abs/2405.16039
- YOCO: https://arxiv.org/abs/2405.05254
- S-LoRA: https://arxiv.org/abs/2311.03285
- Punica: https://arxiv.org/abs/2311.03285

如果你想 dive deeper 某个具体方面（比如 SVD 初始化的数值稳定性、CDB 调度算法的实现细节、或者与某种 specific architecture 如 MoE 的结合），我很乐意展开。
