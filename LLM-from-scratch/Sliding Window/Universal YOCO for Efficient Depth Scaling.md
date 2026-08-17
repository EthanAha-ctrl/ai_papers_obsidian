---
source_pdf: Universal YOCO for Efficient Depth Scaling.pdf
paper_sha256: ce487edcfa35819a18b0edd086bd981faea185c32f933960a35cd07e58637f92
processed_at: '2026-08-12T20:20:19-07:00'
target_folder: LLM-from-scratch/Sliding Window
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 YOCO-U

## 一句话版本

YOCO-U 就是在 YOCO 这个"只缓存一次"的架构上，把前半部分（self-decoder）**拿同样的参数来回跑好几遍**，模型变"深"了但参数没变多、KV cache 没变大，白嫖了表达能力。

---

## 1. 故事从哪开始？

现在所有大模型都在玩 **test-time scaling**，就是让模型推理时多算几步（比如 o1、DeepSeek-R1 那些 thinking tokens）。但有个尴尬的事实：**pre-training 阶段没法高效地"多算几步"**。

为啥不能？因为标准 Transformer 你想让它"多算几步"，就只能加层。加层就等于：
- 更多参数
- 更多 KV cache（每层都要存）
- 更多内存

Universal Transformer（UT）2018 年就提过一个 idea：**别加参数，把同一组参数来回跑 T 次**。听起来很美，但实际上 UT 很难训，而且每次 loop 都要把整个网络重跑一遍，KV cache 会变成 $\mathcal{O}(LTND)$——depth 和 loop 次数相乘，内存直接爆炸。

所以 recursion 这个 idea 一直没在 LLM scale 上真正 work 起来。

---

## 2. YOCO 是个啥？先补个课

YOCO（You Only Cache Once）是同一个作者团队去年的 NeurIPS 论文。核心 idea 是把 Transformer 拆成两半：

```
输入 → [Self-Decoder 一半层] → 生成一份全局 KV cache (K̂, V̂)
                                    ↓（所有层共用这一份）
        [Cross-Decoder 一半层] → 输出
```

**关键 insight**：Cross-Decoder 每层都做 cross-attention，query 是自己算的，但 Key 和 Value 共用同一份全局缓存。所以不管模型有多少层，KV cache 只有 $\mathcal{O}(N)$ 这一份，跟层数 $L$ 无关。

Self-Decoder 那一半用 sliding-window attention（SWA），只缓存局部窗口（比如 512 tokens），开销很小。

结果就是：**KV cache 不随层数增长，prefilling 是 linear 而非 quadratic**。长序列推理特别省。

参考：[YOCO 原论文](https://arxiv.org/abs/2405.05254) | [YOCO 官方页面](https://aka.ms/GeneralAI)

---

## 3. YOCO-U 的核心 idea：在前半段加循环

YOCO 省内存是省内存，但它就是一个普通的固定深度网络，表达能力被层数卡死了。

YOCO-U 的 idea 很直接：**Self-Decoder 那一半，跑完一遍别扔，拿同样的参数再跑一遍，再跑一遍……跑 $T$ 次。**

```
输入 → [Self-Decoder (L/2层, SWA)] → 同样的层再跑一次 → 再跑一次 → ... (T次)
                ↓
         生成 K̂, V̂ (只生成这一次！)
                ↓
        [Cross-Decoder (L/2层)] → 输出
```

就这么简单。论文默认 $T=3$，也就是 self-decoder 跑 3 遍。

**为什么这个能 work，而 Universal Transformer 不行？**

关键在"在哪 loop"。YOCO-U 只 loop 那些用 efficient attention（SWA）的浅层，**不 loop 全局 attention**。

- UT loop 全网络 → 每次都要重新跑 full attention → KV cache 变 $\mathcal{O}(LTND)$
- YOCO-U 只 loop SWA 层 → 全局 KV cache 还是只生成一次 → KV cache 还是 $\mathcal{O}(N)$

SWA 层的 local window cache 虽然随 $T$ 增长，但 window size $W=512$，序列长 $N$ 可以是 256K，$W \ll N$，这部分开销几乎可以忽略。

所以 YOCO-U **既拿到了 recursive depth 的表达力，又保留了 YOCO 的 inference efficiency**。两个 benefit 同时拿到，不是 trade-off。

---

## 4. 公式讲一遍，但用人话翻译

### Universal Self-Decoder 的定义

$$
\text{USD}(X) = \underbrace{\text{Self-Decoder}^{L/2} \circ \cdots \circ \text{Self-Decoder}^{L/2}}_{T \text{ 次}}(X)
$$

翻译成人话：把 $L/2$ 层的 Self-Decoder 作为一个函数 $F$，对输入 $X$ 连续套 $T$ 次：$F(F(F(X)))$。每次套用的是**同一组参数**。

- $X \in \mathbb{R}^{|x| \times d_{\text{model}}}$：输入 token embeddings，$|x|$ 是序列长度，$d_{\text{model}}$ 是 hidden dimension
- 上标 $L/2$：这个 block 有 $L/2$ 层
- $T$：loop 次数，论文默认 $T=3$
- $\circ$：函数复合，前一个的输出是后一个的输入

### 单层 Self-Decoder 怎么算

$$
Y^l = \text{ESA}(\text{LN}(X^l)) + X^l
$$
$$
X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l
$$

翻译：
- 上标 $l$：第 $l$ 层
- $\text{ESA}(\cdot)$：efficient self-attention，这里就是 sliding-window attention，window=512
- $\text{LN}(\cdot)$：RMSNorm，归一化
- $\text{SwiGLU}(X) = (\text{swish}(XW_G) \odot XW_1) W_2$：SwiGLU 激活，就是个带 gate 的 FFN
  - $W_G, W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$：gate 和 value 的投影矩阵
  - $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$：输出投影
  - $\odot$：逐元素相乘
  - $\text{swish}(x) = x \cdot \sigma(x)$，$\sigma$ 是 sigmoid
- $+ X^l$：residual connection

跟标准 Transformer layer 几乎一样，差别在 attention 换成了 sliding-window。

### 全局 KV cache 只生成一次

$$
\hat{K} = \text{LN}(\text{USD}(X)) W_K, \quad \hat{V} = \text{LN}(\text{USD}(X)) W_V
$$

翻译：USD 跑完 $T$ 次之后，把输出做一次投影，得到 $\hat{K}, \hat{V}$。这就是给 Cross-Decoder 用的全局 KV cache。**注意它跟 $T$ 无关——不管你 loop 几次，最终只投影出一份 cache。**

- $\hat{K}, \hat{V} \in \mathbb{R}^{|x| \times d}$：全局共享的 Key 和 Value
- $W_K, W_V$：投影矩阵

### Cross-Decoder 每层做啥

$$
\hat{Q}^l = \text{LN}(X^l) W_Q^l
$$
$$
Y^l = \text{Attention}(\hat{Q}^l, \hat{K}, \hat{V}) + X^l
$$
$$
X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l
$$

翻译：
- $\hat{Q}^l$：第 $l$ 层自己算的 query，每层有自己的 $W_Q^l$
- $\hat{K}, \hat{V}$：共用 USD 生成的全局 cache，**所有层都看这一份**
- $\text{Attention}(\cdot)$：标准 multi-head attention
- Cross-Decoder 用 NoPE（不加位置编码），让全局检索更自由

---

## 5. 为什么 loop 放在前半段（Self-Decoder）？

论文做了 ablation（Table 5）：

| 变体 | 平均准确率 |
|------|----------|
| YOCO（baseline） | 46.95 |
| YOCO-U（loop Self-Decoder） | **48.25** |
| Upper Loop（loop Cross-Decoder） | 46.41 |
| Upper Loop w/o Shared KV | 46.41 |

**在 Cross-Decoder 上 loop 反而变差**。

为什么？参考 ETD（[Encode-Think-Decode, KLC25](https://arxiv.org/abs/2510.07358)）的发现：**Transformer 后半段的层行为更像"final decoder"，负责输出**。对负责输出的层做 recursion，等于让输出层反复 refine，意义不大。

前半段负责 feature extraction 和 abstraction，对它做 recursion 就像让模型"多想几遍提取特征"，收益更明显。

---

## 6. 为什么用 Efficient Attention 做 loop？

论文做了 architecture comparison（Table 3）：

| Model | Wiki ppl↓ | LMB ppl↓ | Avg acc↑ | KV Cache (256K) |
|-------|----------|---------|---------|-----------------|
| Transformer | 22.52 | 22.26 | 47.1 | 10240 MB |
| YOCO | 22.25 | 18.30 | 47.0 | 522 MB |
| Loop/UT | 21.56 | 22.56 | 47.8 | - |
| ParScale | 23.13 | 24.06 | 46.8 | - |
| RINS | 20.98 | 20.06 | 48.3 | 20480 MB |
| YOCO-U | 21.01 | 18.32 | 48.3 | **542 MB** |

**YOCO-U 和 RINS 性能相当，但 KV cache 差 38 倍**（542 MB vs 20480 MB）。

RINS 是什么？就是在标准 Transformer 前半段做 recursion（跟 YOCO-U loop 位置一样），但因为它用 full attention，每次 loop 都要存新 KV cache。YOCO-U 用 SWA，local window cache 只占 20 MB 量级，几乎白送。

---

## 7. 递归收敛了吗？Representation Analysis

Figure 8 画了相邻层之间的 angular distance：

$$
d(l, l+1) = \arccos\left(\frac{\langle X^l, X^{l+1} \rangle}{\|X^l\| \cdot \|X^{l+1}\|}\right)
$$

变量：
- $X^l$：第 $l$ 层的输出表示
- $\langle \cdot, \cdot \rangle$：内积
- $\|\cdot\|$：向量范数
- $\arccos$：反余弦，得到夹角

**三个观察**：

1. **Self-Decoder 内部**：不同 loop 之间 angular distance pattern 几乎一致，说明 recursive 结构稳定
2. **随 loop 次数递减**：mean angular distance 越来越小，表示表示在趋近 fixed point。就像方程 $X = F(X)$ 在迭代求解，慢慢收敛
3. **Self-Decoder → Cross-Decoder 边界**：sharp spike！angular distance 突然变大

第 3 点很有意思——暗示两个模块承担**不同功能**：
- Self-Decoder：progressive refinement，迭代精炼表示
- Cross-Decoder：retrieval + final decoding，从全局 cache 里捞信息然后输出

分工明确，不是混在一起乱算。

---

## 8. Token Efficiency：最关键的实验结果

Figure 2 的 token scaling 实验最让我兴奋：

- **Equal FLOPs**：YOCO-U 的 validation loss 比 YOCO 低 0.033
- **Equal Tokens**：YOCO-U 用 **80B tokens** 就达到 YOCO **210B tokens** 的效果

**Token efficiency 提升约 62%**。

这意味着什么？同样算力下，YOCO-U 从数据里"榨取"的信息更多。Recursive depth 让模型有更多"消化时间"，每个 token 被更充分地处理。

这个收益在 pre-training 阶段就生效，跟 test-time scaling（o1 那种推理时多算）是 orthogonal 的。论文还做了 Thinking SFT 实验（Figure 3），在 11 个数学 benchmark 上 YOCO-U 平均比 YOCO 高 **24.4%**，证明 latent recursion 和 explicit CoT 可以叠加。

---

## 9. Inference Efficiency：实际部署长啥样

Table 8 是 prefill throughput（tokens/s）：

| Model | 8K | 64K | 256K |
|-------|----|-----|------|
| Transformer | 85707 | 27276 | 7475 |
| YOCO | 220662 | 219106 | 220407 |
| RINS | 42905 | 13630 | 3739 |
| YOCO-U | 75637 | 76148 | 76301 |

- YOCO-U 在 256K 是 Transformer 的 **10×**，RINS 的 **20×**
- YOCO 本身更快（线性复杂度，所以 256K 跟 8K 几乎一样），但 YOCO-U 用约 1/3 的 YOCO 速度换来了明显更好的表示能力
- YOCO-U 的 throughput 也几乎不随序列长度变化，因为 prefilling 复杂度是 $\mathcal{O}(\frac{L}{2} T N D)$，linear in $N$

Table 9 是 decode throughput：

| Model | 16K | 256K |
|-------|-----|------|
| Transformer | 1795 | 137 |
| YOCO | 2539 | 318 |
| RINS | 580 | 56 |
| YOCO-U | 1966 | 303 |

- YOCO-U 比 YOCO 只慢 ~5%，因为额外开销只在 SWA 层
- YOCO-U 在 256K 是 Transformer 的 **2.21×**，RINS 的 **5.4×**

Table 10 是 KV cache 占用：

| Model | 256K |
|-------|------|
| Transformer | 10240 MB |
| RINS | 20480 MB |
| YOCO | 522 MB |
| YOCO-U | **542 MB** |

YOCO-U 只比 YOCO 多 20 MB，因为多出来的只是 SWA 的 local window cache，$W=512$ 跟 $N=256K$ 比起来太小了。

---

## 10. 我的理解：这篇论文的真正贡献

### 10.1 Decouple depth from memory

传统加深网络 = 加层 = 加 KV cache = 加内存。YOCO-U 把这个链条打断了：

**加深 = 同样的层多跑几遍 = KV cache 不变**

这打开了 "depth scaling" 的新维度，而不用付出 memory cost。

### 10.2 Partial recursion > Full recursion

UT 对全网络做 recursion，代价太高。YOCO-U 只对 shallow efficient-attention 层做 recursion，抓住两个关键点：
1. **Shallow 层适合 refinement**（深层是 decoder，refinement 意义不大）
2. **Efficient attention 避免 cache 增长**（SWA 的 local cache 开销可忽略）

这是一个**局部最优的 architecture choice**，不是拍脑袋。

### 10.3 Pre-training 和 Test-time scaling 正交

YOCO-U 在 pre-training 就能提升 token efficiency（80B tokens ≈ 210B tokens 效果）。这跟 o1 式的 test-time scaling 是 orthogonal 的——Thinking SFT 实验证明两者可以叠加（+24.4%）。

所以未来的 scaling 路线可能是：
- Pre-training：recursive depth（YOCO-U 式）
- Post-training：explicit CoT thinking（o1 式）
- 两者叠加

### 10.4 Fixed point intuition

USD 的 $T$ 次迭代可以看作求解不动点 $X = F(X)$。Figure 8 的 angular distance 递减证实了这一点。这跟 [Coconut](https://arxiv.org/abs/2412.06769) 的 latent reasoning 思路异曲同工——都是把"多想几步"压缩进 pre-training 的 representation 里，而不是推理时 explicit 展开。

---

## 11. 实用的 Takeaway

如果你在做 LLM 架构研究，这篇论文给你几个 actionable insights：

1. **Recursion 放在 shallow layer**，别放在 deep layer
2. **Recursion 用 efficient attention**，别用 full attention
3. **Global cache 只生成一次**，避免随 loop 次数增长
4. **Parameter sharing across loops**，不增加参数量
5. **NoPE in cross-attention**，增强全局检索
6. **RoPE in self-attention**，局部窗口内需要位置信息

这套组合拳让 recursive computation 从"理论上好但实践中太贵"变成"实际可用且高效"。

---

## 12. 我还想吐槽的点

1. **Loop 次数 $T$ 怎么选？** 论文只测到 $T=5$，没给出 scaling law 上的 guidance。实际部署时 $T$ 是个超参，不知道 optimal $T$ 跟 model size / data size 的关系。

2. **Fixed point 一定收敛吗？** 论文只 empirically 观察，没理论证明。某些数据分布或参数初始化下会不会发散？未知。

3. **跟 MoE 的交互**？ 论文用了 fine-grained MoE（64 个专家激活 8 个），但没单独分析 MoE routing 在 recursive 设置下的行为。Router 会不会每次 loop 选不同的专家？这会影响表示稳定性吗？

4. **训练稳定性**。论文说 "highly stable, smooth loss, no spikes"，但没给具体的 loss curve。对于想复现的人，这点信息不够。

5. **更长的 context**。论文测到 256K，但 YOCO 这种架构理论上可以到 millions。YOCO-U 在 1M+ context 下表现如何？未知。

---

## 相关参考链接

**核心论文**：
- [Universal YOCO (本论文)](https://aka.ms/GeneralAI)
- [YOCO 原论文 (NeurIPS 2024)](https://arxiv.org/abs/2405.05254)
- [Universal Transformer](https://arxiv.org/abs/1807.03819)

**对比方法**：
- [RINS (Recursive Inference Scaling)](https://arxiv.org/abs/2502.07503)
- [ParScale (Parallel Scaling)](https://arxiv.org/abs/2505.10475)
- [ETD (Encode-Think-Decode)](https://arxiv.org/abs/2510.07358)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)

**Test-time scaling**：
- [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Coconut (latent reasoning)](https://arxiv.org/abs/2412.06769)

**Efficient Attention 方案**：
- [RetNet](https://arxiv.org/abs/2307.08621)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [Jamba (hybrid Transformer-Mamba)](https://arxiv.org/abs/2403.19887)

**位置编码**：
- [RoPE](https://arxiv.org/abs/2104.09864)
- [NoPE (No Position Embedding)](https://arxiv.org/abs/2501.18795)

**推理优化**：
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [Nano-vLLM (论文用的推理框架)](https://github.com/GeeeekExplorer/nano-vllm)

**Scaling Laws**：
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [RMSNorm](https://arxiv.org/abs/1910.07467)

---

# Universal YOCO (YOCO-U): 递归计算与高效注意力的协同架构

## 1. 论文背景与核心动机

这篇论文来自 Microsoft Research 和 Tsinghua University，作者 Yutao Sun, Li Dong 等。核心动机源于一个矛盾现象：**test-time scaling**（如 OpenAI o1, DeepSeek-R1）显著提升了 LLM 的推理能力，但标准 Transformer 在 pre-training 阶段无法高效地 scale computation depth。

两个关键痛点：
- **Loop Transformer 的代价**：在标准 Transformer 中引入循环机制（如 Universal Transformer）会导致 KV cache 随深度线性增长 $\mathcal{O}(LTND)$，同时 global attention 被反复执行，计算复杂度爆炸。
- **YOCO 的深度不足**：YOCO 虽然通过 decoder-decoder 架构实现了 constant KV cache 和 linear pre-filling，但其表达能力受限于固定层数。

YOCO-U 的核心 insight：**把 recursion 限制在 shallow efficient-attention 层，避免触碰 global attention，从而既获得 recursive depth 的表达能力，又保留 YOCO 的 inference efficiency**。

## 2. YOCO 回顾（前置知识）

YOCO (You Only Cache Once) 将模型拆分为两部分：

```
Input → [Self-Decoder (L/2 layers)] → 生成 global KV cache (K̂, V̂)
                                          ↓
       [Cross-Decoder (L/2 layers)] ← 复用同一份 K̂, V̂
                                          ↓
                                     Output logits
```

- **Self-Decoder**：使用 efficient attention（如 sliding-window attention, SWA），只缓存局部窗口 KV
- **Cross-Decoder**：所有层共享同一份 global KV cache，通过 cross-attention 检索信息

关键性质：global KV cache 只 materialize 一次，复杂度 $\mathcal{O}(N)$ 而非 $\mathcal{O}(NL)$。

参考：[YOCO 论文](https://arxiv.org/abs/2405.05254) | [YOCO 官方页面](https://aka.ms/GeneralAI)

## 3. YOCO-U 架构详解

### 3.1 整体架构图解析

```
┌─────────────────────────────────────────────┐
│         Cross-Decoder (L/2 layers)           │
│   每层: Q^l = LN(X^l) W_Q^l                  │
│         Y^l = Attention(Q^l, K̂, V̂) + X^l    │ ← 复用全局 KV cache
│         X^{l+1} = SwiGLU(LN(Y^l)) + Y^l      │
└─────────────────────────────────────────────┘
                    ↑
              K̂, V̂ (生成一次)
                    ↑
┌─────────────────────────────────────────────┐
│     Universal Self-Decoder (T iterations)   │
│   ┌───────────────────────────────────┐     │
│   │ Self-Decoder (L/2 layers, SWA)    │     │
│   │ Y^l = ESA(LN(X^l)) + X^l          │     │ ← 迭代 T 次
│   │ X^{l+1} = SwiGLU(LN(Y^l)) + Y^l   │     │   共享参数
│   └───────────────────────────────────┘     │
│         ↻ 重复 T 次 (parameter sharing)      │
└─────────────────────────────────────────────┘
                    ↑
              Input Embeddings X^0
```

### 3.2 核心数学公式

**Universal Self-Decoder (USD)** 的递归定义：

$$
\text{USD}(X) = \underbrace{\text{Self-Decoder}^{L/2} \circ \cdots \circ \text{Self-Decoder}^{L/2}}_{T \text{ iterations}}(X)
$$

变量解释：
- $X \in \mathbb{R}^{|x| \times d_{\text{model}}}$：输入 token embeddings，$|x|$ 是序列长度，$d_{\text{model}}$ 是 hidden dimension
- $L$：总层数，Self-Decoder 占 $L/2$ 层
- $T$：loop iterations（论文默认 $T=3$）
- $\circ$：函数复合，表示前一模块输出作为后一模块输入

**Self-Decoder 单层计算**（公式 2）：

$$
Y^l = \text{ESA}(\text{LN}(X^l)) + X^l
$$
$$
X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l
$$

变量与符号：
- 上标 $l$：layer index（第 $l$ 层）
- $\text{ESA}(\cdot)$：Efficient Self-Attention，论文默认用 sliding-window attention (SWA)，window size $W=512$
- $\text{LN}(\cdot)$：RMSNorm（Root Mean Square Layer Normalization）
- $\text{SwiGLU}(X) = (\text{swish}(XW_G) \odot XW_1) W_2$：SwiGLU 激活函数
  - $W_G, W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$：gate 和 value 的投影矩阵
  - $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$：输出投影
  - $\odot$：element-wise 乘法
  - $\text{swish}(x) = x \cdot \sigma(x)$，$\sigma$ 是 sigmoid

**Global KV cache 生成**（公式 3）：

$$
\hat{K} = \text{LN}(\text{USD}(X)) W_K, \quad \hat{V} = \text{LN}(\text{USD}(X)) W_V
$$

- $\hat{K}, \hat{V} \in \mathbb{R}^{|x| \times d}$：全局共享的 Key 和 Value cache，所有 Cross-Decoder 层共用
- $W_K, W_V$：投影矩阵
- **关键**：$\hat{K}, \hat{V}$ 只计算一次，与 $T$ 无关

**Cross-Decoder 单层计算**（公式 4）：

$$
\hat{Q}^l = \text{LN}(X^l) W_Q^l
$$
$$
Y^l = \text{Attention}(\hat{Q}^l, \hat{K}, \hat{V}) + X^l
$$
$$
X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l
$$

- $\hat{Q}^l$：第 $l$ 层专属的 query（每层独立计算）
- $W_Q^l \in \mathbb{R}^{d \times d}$：第 $l$ 层的 query 投影矩阵
- $\text{Attention}(\cdot)$：标准 multi-head attention
- Cross-Decoder 使用 NoPE (No Position Embedding) 增强 global retrieval 能力

### 3.3 为什么这个设计 work？Intuition

**递归的固定点视角**：Universal Self-Decoder 的多次迭代可以看作求解一个不动点方程 $X = F(X)$，其中 $F$ 是 Self-Decoder 的计算。每次迭代都在 refine representation，最终趋近于一个稳定的 fixed point。Figure 8 的 angular distance 分析证实了这一点——随 loop 次数增加，相邻层间的 angular distance 递减，表示表示趋于稳定。

**分工假说**：Figure 8 中 Self-Decoder 和 Cross-Decoder 之间出现 sharp spike（angular distance 突变），暗示两个模块承担不同功能：
- Self-Decoder：progressive refinement，迭代精炼中间表示
- Cross-Decoder：information retrieval + final decoding，从全局 cache 中检索

**为什么 recursion 放在 shallow 层**：深层 Transformer 的后段通常承担"final decoder"角色（参考 ETD, [KLC25]），对这些层做 recursion 收益递减。早期层负责 feature extraction 和 abstraction，更适合迭代精炼。

**为什么用 efficient attention 做 recursion**：如果用 full attention 做 recursion，每次迭代都要重新生成 KV cache，导致 $\mathcal{O}(LTND)$ 内存。用 sliding-window attention，local window cache 虽然随 $T$ 增长 $\mathcal{O}(W T L D)$，但 $W \ll N$（$W=512$ vs $N=256K$），额外开销可忽略。

## 4. 复杂度对比表解析

| Model | KV Cache Memory | Prefilling Time | Decoding Time |
|-------|----------------|----------------|---------------|
| Transformer | $\mathcal{O}(LND)$ | $\mathcal{O}(LN^2D)$ | $\mathcal{O}(LND)$ |
| YOCO | $\mathcal{O}((N+WL)D)$ | $\mathcal{O}(\frac{L}{2}ND)$ | $\mathcal{O}(\frac{L}{2}(N+W)D)$ |
| Loop/UT | $\mathcal{O}(LTND)$ | $\mathcal{O}(LTN^2D)$ | $\mathcal{O}(LTND)$ |
| YOCO-U | $\mathcal{O}((N+WTL)D)$ | $\mathcal{O}(\frac{L}{2}TND)$ | $\mathcal{O}(\frac{L}{2}(N+WT)D)$ |

变量含义：
- $N$：sequence length
- $L$：number of layers
- $D$：hidden dimension
- $T$：loop iterations
- $W$：local window size（SWA）

**关键 insight**：
- Transformer 的 KV cache 是 $\mathcal{O}(LND)$，随层数线性增长
- YOCO 和 YOCO-U 的 global KV cache 都是 $\mathcal{O}(ND)$，与 $L$ 无关
- YOCO-U 比 YOCO 多了 $\mathcal{O}(W T L D)$ 的 local cache，但由于 $W \ll N$（512 vs 256K），这部分几乎可忽略
- Loop/UT 的 prefilling 是 $\mathcal{O}(LTN^2D)$，因为每次 loop 都要跑 full attention
- YOCO-U 的 prefilling 是 $\mathcal{O}(\frac{L}{2} T N D)$，linear in $N$，因为 recursion 只作用于 efficient attention

Table 10 的实测数据印证：
| Model | 8K | 16K | 32K | 64K | 128K | 256K |
|-------|----|-----|-----|-----|------|------|
| Transformer | 320 | 640 | 1280 | 2560 | 5120 | 10240 MB |
| RINS | 640 | 1280 | 2560 | 5120 | 10240 | 20480 MB |
| YOCO | 26 | 42 | 74 | 138 | 266 | 522 MB |
| YOCO-U | 46 | 62 | 94 | 158 | 286 | 542 MB |

YOCO-U 在 256K 序列下只需 542 MB，而 RINS 需要 20480 MB（38× 差距）。

## 5. 实验结果深度分析

### 5.1 Training Recipe

- **模型**：10B total params (1.3B activated)，fine-grained MoE with shared experts
  - 64 experts，激活 8 + 1 shared expert
  - expert dim = 1024
- **架构**：20 layers，hidden dim = 2560，Self-Decoder 和 Cross-Decoder 各 10 层
- **位置编码**：Self-Decoder 用 RoPE，Cross-Decoder 用 NoPE
- **训练**：300B tokens，batch size 4M，AdamW ($\beta = 0.9, 0.95$)，LR = 1e-3
- **硬件**：AMD MI300X GPUs

### 5.2 Token Scaling（Figure 2）

- **Equal FLOPs**：YOCO-U 的 validation loss 比 YOCO 低 $\Delta L = 0.033$
- **Equal Tokens**：YOCO-U 用 80B tokens 达到 YOCO 210B tokens 的效果，**token efficiency 提升 ~62%**

这表明 recursive computation 在 pre-training 阶段就能提升 data efficiency，orthogonal 到 test-time scaling。

### 5.3 End-Task Evaluation（Table 2）

| Model | ARC-C | Winogrande | HellaSwag | MMLU | BBH | GSM8K | Humaneval | DROP | Avg |
|-------|-------|-----------|-----------|------|-----|-------|-----------|------|-----|
| YOCO | 46.50 | 61.72 | 63.44 | 49.59 | 33.13 | 38.06 | 9.15 | 32.62 | 41.78 |
| YOCO-U (Equal FLOPs) | 47.87 | 68.67 | 66.80 | 54.63 | 35.49 | 50.49 | 10.98 | 34.94 | 46.23 |
| YOCO-U (Equal Steps) | 48.72 | 69.85 | 67.12 | 55.63 | 36.31 | 50.57 | 10.37 | 38.07 | 47.08 |

即使控制 FLOPs 相同，YOCO-U 仍然 +4.45 average，说明收益来自更高效的 compute allocation，不仅仅是更多计算。

### 5.4 Thinking SFT（Figure 3）

在 280B checkpoint 基础上继续训练 20B tokens 的 math thinking data，评估 11 个 math benchmark：
- GSM8K, MATH, SVAMP, ASDiv, MAWPS, CARP, TABMWP, Gaokao 2023 En, OlympiadBench, CollegeMath, AMC23

YOCO-U 在所有 11 个 benchmark 上都优于 YOCO，**平均准确率提升 24.4%**。

这个结果很有意思：recursive computation（latent reasoning）和 explicit chain-of-thought（test-time scaling）是 orthogonal 的，可以叠加。

### 5.5 Architecture Comparison（Table 3）

与 Universal Transformer, RINS (early-layer recursion), ParScale (parallel scaling) 对比：

| Model | Wiki ppl↓ | LMB ppl↓ | Avg acc↑ |
|-------|----------|---------|---------|
| Transformer | 22.52 | 22.26 | 47.1 |
| YOCO | 22.25 | 18.30 | 47.0 |
| Loop/UT | 21.56 | 22.56 | 47.8 |
| ParScale | 23.13 | 24.06 | 46.8 |
| RINS | 20.98 | 20.06 | 48.3 |
| YOCO-U | 21.01 | 18.32 | 48.3 |

三个关键结论：
1. **Scaling FLOPs on bottom blocks > all blocks**：RINS（early recursion）优于 vanilla UT（全层 recursion）
2. **Recursive scaling > Parallel scaling**：ParScale 不增加 depth，效果最差
3. **Efficient-attention recursion = Full-attention recursion**：YOCO-U 与 RINS 相当，但 KV cache 开销天差地别

### 5.6 Ablation Studies（Table 5）

| 变体 | Wiki ppl↓ | LMB acc↑ | Avg acc↑ |
|------|----------|----------|---------|
| YOCO (baseline) | 22.25 | 41.16 | 46.95 |
| Deep (Instead of Wide) | 22.04 | 37.67 | 46.87 |
| YOCO-U | 21.01 | 41.18 | 48.25 |
| Deeper (Instead of Wide) | 21.42 | 41.39 | 48.59 |
| Upper Loop (Cross-Decoder) | 22.15 | 38.21 | 46.41 |
| Upper Loop w/o Shared KV | 22.06 | 38.21 | 46.41 |

**Loop Position**：在 Cross-Decoder（深层的 full attention）上做 loop 反而变差（46.41 vs 48.25），证实 deep layer recursion 收益递减。

**Model Layout**：把模型变深（40 layers, 1792 hidden）而非加 loop，效果与 YOCO-U 接近，但失去 inference efficiency 优势。

### 5.7 Scaling Property（Figure 5, 6）

- **Parameter Scaling**：从 300M 到 10.8B，YOCO-U 保持稳定 gain。当 activated params > 10B，YOCO-U 甚至接近 non-recursive 变体，说明 recursion 消除了 parameter redundancy。
- **Loop Scaling**：$T \in \{1, 2, 3, 5\}$，loss 随 $T$ 单调下降。$T=5$ 时 FLOPs 增加 3×，仍然有收益。

### 5.8 Inference Efficiency（Figure 7, Table 8/9/10）

**Prefill Throughput** (tokens/s)：

| Model | 8K | 16K | 64K | 256K |
|-------|----|-----|-----|------|
| Transformer | 85707 | 66342 | 27276 | 7475 |
| YOCO | 220662 | 219734 | 219106 | 220407 |
| RINS | 42905 | 33276 | 13630 | 3739 |
| YOCO-U | 75637 | 75694 | 76148 | 76301 |

YOCO-U 在 256K 下是 Transformer 的 10×，是 RINS 的 20×。YOCO 本身更快，但 YOCO-U 用 ~1/3 的 YOCO 速度换来更好的表示能力。

**Decode Throughput**：

| Model | 16K | 64K | 256K |
|-------|-----|-----|------|
| Transformer | 1795 | 450 | 137 |
| YOCO | 2539 | 975 | 318 |
| RINS | 580 | 118 | 56 |
| YOCO-U | 1966 | 865 | 303 |

YOCO-U 在 256K 下是 Transformer 的 2.21×，是 RINS 的 5.4×。YOCO-U 只比 YOCO 慢 ~5%，因为额外开销只在 efficient attention 层。

## 6. 与相关工作的关系

### 6.1 Universal Transformer [DGV+18]

UT 对整个网络做 recursion，导致：
- 每次迭代重新生成 KV cache
- 全层 full attention 反复执行
- 内存和计算随 $T$ 线性增长

YOCO-U 的改进：partial recursion（只对 shallow efficient-attention 层），global KV cache 只生成一次。

参考：[Universal Transformer](https://arxiv.org/abs/1807.03819)

### 6.2 RINS (Recursive Inference Scaling) [AZ25]

RINS 在标准 decoder-only Transformer 上做 early-layer recursion，效果好但 KV cache 仍随 depth 增长。

YOCO-U 与 RINS 性能相当，但 KV cache 小 38×。

参考：[RINS](https://arxiv.org/abs/2502.07503)

### 6.3 ParScale [CHC+25]

ParScale 通过并行分支增加 compute，不增加 depth。效果不如 recursive scaling，因为缺乏 depth 的表达增强。

参考：[ParScale](https://arxiv.org/abs/2505.10475)

### 6.4 ETD (Encode-Think-Decode) [KLC25]

ETD 发现 Transformer 后段层行为类似 final decoder，对深层做 recursion 收益递减。这解释了 YOCO-U 为何选择 shallow recursion。

参考：[ETD](https://arxiv.org/abs/2510.07358)

### 6.5 Coconut (Continuous Latent Reasoning) [HSS+24]

Coconut 将 explicit chain-of-thought 压缩为 continuous representation。YOCO-U 的 recursive computation 可视为 latent reasoning 的一种形式，与 explicit CoT 正交。

参考：[Coconut](https://arxiv.org/abs/2412.06769)

## 7. Representation Analysis 的 Intuition

Figure 8 展示 angular distance $d(l, l+1)$ across layers：

$$
d(l, l+1) = \arccos\left(\frac{\langle X^l, X^{l+1} \rangle}{\|X^l\| \|X^{l+1}\|}\right)
$$

观察：
1. **Self-Decoder 内部**：不同 loop 之间 pattern 一致，说明 recursive 结构稳定
2. **Loop 间递减**：mean angular distance 随 loop 次数递减，表示表示趋近 fixed point
3. **Block 间 spike**：Self-Decoder → Cross-Decoder 边界出现 sharp discontinuity，暗示功能切换

这个分析为"Self-Decoder 做 refinement, Cross-Decoder 做 retrieval"的分工假说提供了证据。

## 8. 核心设计选择的 Intuition 总结

| 设计选择 | 原因 |
|---------|------|
| Recursion 在 Self-Decoder | Shallow 层适合 feature refinement，deep 层是 final decoder |
| 用 Efficient Attention (SWA) | 避免 recursion 带来 KV cache 增长，$W \ll N$ |
| Global KV cache 只生成一次 | Cross-Decoder 复用，保持 $\mathcal{O}(N)$ memory |
| Parameter sharing across loops | 不增加参数量，纯靠 depth 提升表达力 |
| NoPE in Cross-Decoder | 增强全局检索，不受位置编码限制 |
| RoPE in Self-Decoder | 局部窗口内需要位置信息 |

## 9. 局限性与开放问题

论文未深入讨论的点：
1. **Fixed point 的理论保证**：递归是否一定收敛？什么条件下会发散？论文只 empirically 观察到稳定，缺乏理论分析。
2. **Optimal $T$ 的选择**：实验只测到 $T=5$，更大的 $T$ 是否有 diminishing returns 甚至退化？
3. **与 test-time scaling 的交互**：Thinking SFT 实验显示 orthogonal，但更深的交互机制（如动态调整 $T$）未探索。
4. **MoE 与 recursion 的协同**：论文用了 fine-grained MoE，但 MoE routing 在 recursive 设置下的行为未单独分析。

## 10. 个人 Takeaway

这篇论文的精髓在于 **"decouple depth from memory"** 的设计哲学：

- 传统深度扩展 = 更多层 = 更多 KV cache = 更多内存
- YOCO-U 深度扩展 = 同层重复 = KV cache 不变 = 内存恒定

关键 trick 是把 recursion 限制在 efficient-attention 层，避开 full attention 的 quadratic 代价。这让 recursive computation 从"理论上有吸引力但实践中太贵"变成"实践中可行且高效"。

从 scaling law 角度，YOCO-U 揭示了一个新的维度：在固定参数量下，通过 recursive depth 提升 token utility，这为未来 "compute-optimal" 训练提供了新的 knob。

相关参考链接：
- [YOCO 原论文 (NeurIPS 2024)](https://arxiv.org/abs/2405.05254)
- [Universal Transformer](https://arxiv.org/abs/1807.03819)
- [RINS](https://arxiv.org/abs/2502.07503)
- [ParScale](https://arxiv.org/abs/2505.10475)
- [DeepSeek-R1 (test-time scaling)](https://arxiv.org/abs/2501.12948)
- [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)
- [Coconut (latent reasoning)](https://arxiv.org/abs/2412.06769)
- [RetNet](https://arxiv.org/abs/2307.08621)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [NoPE](https://arxiv.org/abs/2501.18795)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [Nano-vLLM (inference framework)](https://github.com/GeeeekExplorer/nano-vllm)
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Scaling Laws](https://arxiv.org/abs/2001.08361)
