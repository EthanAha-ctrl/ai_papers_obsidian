---
source_pdf: Tapered Language Models.pdf
paper_sha256: f517e613c9e519bdb0040eef286be11a71650cfee132214bcd9ece89fd8fcbbc
processed_at: '2026-08-12T12:51:35-07:00'
target_folder: LLM-from-scratch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Tapered Language Models - 人话版

好，我把论文的"骨架"用人话捋一遍，该有的技术细节都在，但讲得更像聊天。

---

## 这篇 paper 在吐槽什么 default？

你看现在所有 LM——不管 Transformer、Mamba、RWKV、Titans、Llama、GPT——它们都有一个共同的"出厂设置"：

**每一层都长得一模一样，参数一样多。**

这个 default 是 2017 年 Vaswani 那篇 Attention is All You Need 留下来的，之后所有人抄，没人 seriously 质疑。

https://arxiv.org/abs/1706.03762

但你看现在一堆 independent 的 research 都在指向同一件事——**层和层的重要性根本不均匀**：

- Tuned Lens (Belrose 2023) 发现 residual stream 在倒数几层之前就差不多收敛了，后面是在"微调"，不是在"创造" https://arxiv.org/abs/2303.08112
- LayerSkip (Elhoushi 2024) 发现推理时直接跳过后面的层，output 几乎不变 https://arxiv.org/abs/2404.16710
- ShortGPT (Men 2025) 和 Gromov et al. 2024 直接把后面的层删掉，模型也没坏多少 https://arxiv.org/abs/2403.17887
- Geva et al. 2021 的 interpretability 工作指出，前几层的 FFN 在记浅层语法，后几层的 FFN 在记深层语义——所以每层"干的活"就不一样 https://aclanthology.org/2021.emnlp-main.446/

那既然重要性不均匀，**为什么参数还均匀分？**这就是这篇 paper 的 question。

---

## 一个最直接的 blunt test

作者先做了一个特别简单的实验：拿一个 440M 的 Transformer，把 12 层分成三组（early / middle / late），给每组分配不同的 $d_{\text{ff}}$（MLP 中间维度），但**总参数严格相同**。

四种配置（Figure 2）：

| 配置 | PPL | vs Uniform |
|---|---|---|
| Uniform | 16.28 | — |
| Wider-early（前宽后窄）| **15.96** | -0.32 |
| Wider-middle（中宽）| 16.61 | +0.33 |
| Wider-late（前窄后宽）| 17.29 | +1.01 |

**同样的参数预算，方向反过来差 1 个 PPL。**

front-loading helps，back-loading hurts。这告诉我们：参数不是平均摊就好的被动资源，要放到该用的地方——也就是早期层。

---

## Tapered Language Models 的核心 idea

既然 piecewise 的 blunt allocation 都有效，那就用更 smooth 的方式 taper。定义很简单（公式3）：

$$
d_C(l+1) \le d_C(l) \quad \text{for all } l
$$
$$
\frac{1}{L}\sum_{l=0}^{L-1} d_C(l) = d_C^{\text{baseline}}
$$

变量含义：
- $d_C(l)$: 第 $l$ 层某个组件 $C$ 的尺寸（比如 MLP 的 $d_{\text{ff}}$，attention head 数，SSM state size 都行）
- $L$: 总层数
- 第一个条件：随着 $l$ 增大（往深处走），$d_C$ 单调不增
- 第二个条件：所有层的 $d_C$ 平均值 = baseline，所以**总参数和总 FLOPs 严格保持不变**

这个 formulation 是 architecture-agnostic 的——任何"按层分布、控制参数量"的维度都可以 taper。但作者选择 **MLP width $d_{\text{ff}}$** 作为 instantiation，因为：

1. MLP 是 parameter 的"大头"——现代 LM 里 MLP 通常占 2/3 以上参数
2. $d_{\text{ff}}$ 是一个干净的 scalar，可以独立调
3. 不管是 vanilla FFN ($2dd_{\text{ff}}$) 还是 SwiGLU ($3dd_{\text{ff}}$)，参数量都和 $d_{\text{ff}}$ 线性相关，所以 average-preserving 约束同时 preserve 参数和 FLOPs

---

## 三种 taper schedule

参数化方式：选 $d_{\text{start}}$（layer 0 的宽度）和 $d_{\text{end}}$（最后一层的宽度），中间用 schedule 决定形状。

### Linear (公式4)
$$
d_{\text{ff}}(l) = d_{\text{start}} - (d_{\text{start}} - d_{\text{end}}) \cdot \frac{l}{L-1}
$$

变量：
- $l$: 当前层 index，从 $0$ 到 $L-1$
- $\frac{l}{L-1}$: 归一化到 $[0, 1]$ 的深度比例
- $d_{\text{start}} - d_{\text{end}}$: 总衰减幅度
- 等速下降，两端没有 plateau

### Cosine (公式5) — winner
$$
d_{\text{ff}}(l) = d_{\text{end}} + \frac{d_{\text{start}} - d_{\text{end}}}{2} \left(1 + \cos\frac{\pi l}{L-1}\right)
$$

变量：
- $\frac{\pi l}{L-1}$: $l$ 从 $0$ 走到 $L-1$ 时，角度从 $0$ 走到 $\pi$
- $\cos(\cdot)$: 从 $1$ 走到 $-1$
- $(1+\cos(\cdot))$: 从 $2$ 走到 $0$
- 整体形状：两端 soft plateau，中段陡降——像半余弦曲线

### Sigmoid (公式6)
$$
d_{\text{ff}}(l) = d_{\text{end}} + \frac{d_{\text{start}} - d_{\text{end}}}{1 + e^{k\left(\frac{l}{L-1} - 0.5\right)}}
$$

变量：
- $k$: steepness，固定取 10
- $\frac{l}{L-1} - 0.5$: 相对中点的偏移
- 形状：transition 集中在中间窄带，两端几乎是 binary 的（要么接近 $d_{\text{start}}$，要么接近 $d_{\text{end}}$）

### 三者直觉对比

- Linear：所有层都被 equally engage，没有 plateau
- Cosine：两端缓、中间快——既不像 linear 那么"机械"，也不像 sigmoid 那么"二值化"
- Sigmoid：大部分层都"困在两端"，只有中间一小段在做 transition

---

## Sweep 结果（Table 1）—— Cosine 严格 dominate

在 440M 上做 5×3 grid：5 个 width ratio × 3 个 schedule，所有 configuration 严格同参数同 FLOPs。

Uniform baseline PPL = 16.28

关键 observations：

1. **Cosine 严格 dominate**: 所有 5 个 width ratio 下都是 cosine < linear < sigmoid，无例外
2. **Cosine 最差 > Linear 最好**: cosine worst (1.75/0.25 = 15.49) 比 linear best (1.625/0.375 = 15.64) 还好
3. **1.5/0.5 是 sweet spot**: PPL 14.44，比 uniform 好 1.84 个点

为什么 1.5/0.5 而不是更激进？直觉是：
- 太 mild (1.25/0.75)：early layers 没多多少，late layers 也没少多少——没 exploit asymmetry
- 太 aggressive (1.75/0.25)：late layers 只剩 25%，连 refinement 都做不了，被 starve
- 1.5/0.5：front-loading 但 late layers 还有 50% 可用——平衡点

作者最终固定 cosine + 1.5/0.5 不变，直接 carry forward 到所有后续实验。

---

## Main Results（Table 2）—— 4 种 architecture × 2 个 scale

为什么测这么多 architecture？因为作者想证明 principle 是 **architecture-agnostic** 的，不是 attention 的某种 property。4 种 architecture 的 token-mixing module 截然不同：

- Transformer: softmax attention https://arxiv.org/abs/1706.03762
- Gated Attention: attention + output gating https://arxiv.org/abs/2505.06708
- Hope-attention: nested learning + self-modifying memory https://arxiv.org/abs/2512.24695
- Titans: attention + neural long-term memory module https://arxiv.org/abs/2501.00663

结果（8 个 architecture×scale 组合）：

- Commonsense reasoning avg accuracy: **8/8 全部 improve**
- LAMBADA PPL: **8/8 全部 improve**
- WikiText PPL: 7/8 improve（Hope-attention 1.3B 微回退 0.03）

scale 上 gain 也没 diminish——1.3B 上仍然稳定 improvement。

这个结果非常 striking：**四种 token-mixing 完全不同的 architecture，tapering 都 work**。强烈暗示 principle 是 about "MLP stack 在 depth 上的 capacity 分配"，而非 token-mixing 的 property。

---

## 为什么 tapering works——Layer-wise Novelty 分析

前面只证明 helps，Section 4 探索 why。

定义两个 cosine similarity（公式8、9）：

$$
\rho_l^{\text{block}} = \cos(h_{l+1} - h_l, h_l)
$$
$$
\rho_l^{\text{MLP}} = \cos(\mathcal{F}_l(z_l), h_l)
$$

变量：
- $h_l$: 进入第 $l$ 层的 residual stream
- $h_{l+1} - h_l$: 该层整个 block 对 residual 写入的 delta
- $\mathcal{F}_l(z_l)$: 第 $l$ 层 MLP 的输出
- $\cos(\cdot, \cdot)$: cosine similarity

**物理意义**：
- $\rho \approx 0$: 该层写入内容与 residual 正交 → 在写**新信息**
- $\rho > 0$: 该层写入内容与 residual 同向 → 在**强化已有方向**（echo / refine，不是 compute new）
- $\rho$ 随深度上升 = **novelty diminishing**

### GPT-2 family 上的测量

作者用 GPT-2 124M 到 1.5B 的 pretrained checkpoint，在 2048 tokens WikiText-2 上测每层 $\rho$（Figure 4）。

**Pattern 在所有 model size 上一致**：
- 两个 quantity 都在 early-middle layers 降到 ≈0 或略负（写新信息）
- 在 second half 单调上升（强化已有）
- Pearson correlation with layer index 都是正的：
  - $\rho_l^{\text{MLP}}$ correlation: $r = 0.49$ 到 $0.71$（更 tight）
  - $\rho_l^{\text{block}}$ correlation: $r = 0.27$ 到 $0.71$

### 这给了 tapering 一个 mechanistic 解释

- **Late layers**（uniform capacity 下）：MLP 输出与 residual 高度 aligned，额外的 hidden dimension 在做"无用功"——reinforce 而非 innovate
- **Early layers**: MLP 输出 orthogonal 于 residual，capacity 有处可用

Tapering 就是把这个 mismatch 修掉：
1. 在 alignment 最大的 late layers 减 capacity（释放参数）
2. 在 alignment 最小的 early layers 增 capacity（把参数送过去）

这就解释了 Figure 2 的 asymmetry：**front-loading helps 因为 early MLPs 用得上；back-loading hurts 因为 late MLPs 用不上**。

### MLP signal 比 block signal 更 tight 这件事

这点本身也有信息：
- 不是 attention 在干活、MLP 坐吃白食，而是 MLP 自己也在 echo
- Block-level 也 rise → principle 不只是 MLP-specific，可以 generalize 到 attention head count、SSM state size 等

---

## Long-context 不受损（Table 3）

有人可能担心：late layers 是不是负责 long-context 信息整合的？tapering 削弱 late layers 会不会破坏 long context？

作者用 Needle-in-a-Haystack 测了 4K/8K/16K：三种 single-needle variant（passkey / number / UUID）+ multi-query variant。

结果：**tapered models match or improve across the table**，gains 集中在 absolute score 最低的 hard cells（比如 16K UUID multi-query，Titans 从 11.8 → 12.4）。

这是个重要 sanity check：tapering 不是用 long-context 能力换 perplexity 的 trade-off。

---

## 和我（Karpathy）的直觉怎么对齐

### 这事让我想起 ResNet

He et al. 2015 发现 deep ResNet block 学的是 residual——identity 已经是 good default，深层 block 只需要小 perturbation。

TLM 在 LM 里揭示类似 pattern：**late MLP blocks 输出与 residual 高度 aligned，等于在学"小强化"而不是"大变换"**。这与 Geva 把 FFN 视作 key-value memory 的 view 一致——late memory slots 在 amplify 已有 activation，early memory slots 在写入新的 syntactic/semantic pattern。

### TLM 不是 MoE，但和它精神相似

MoE 在 **token 维度**动态分配 capacity（每个 token 路由到不同 expert）。TLM 在 **layer 维度**静态分配 capacity（每层 width 不同）。

二者是 orthogonal 的，可以叠加：early layers 用更多 experts + 更宽 MLP，late layers 用更少 experts + 更窄 MLP。这是 §5 提到的 future work 方向。

### 和 LayerDrop / Mixture-of-Depths 的区别

- LayerDrop (Fan 2019): 训练时随机 drop 整层 https://arxiv.org/abs/1909.11556
- Mixture-of-Depths (Raposo 2024): 推理时 token 动态跳过层 https://arxiv.org/abs/2404.02258
- TLM: pre-training 时静态重新分布 capacity，所有 token 走所有层

TLM 是 design-time 的 architectural prior，MoD 是 inference-time 的 dynamic decision。可以叠加。

### 和 post-training pruning 的本质区别

SliceGPT (Ashkboos 2024) https://arxiv.org/abs/2401.15024 和 Sharma 2023 https://arxiv.org/abs/2312.13558 都是先 train uniform model 再 prune MLP weights，然后 fine-tune recover。

TLM 是**一开始就把 capacity 放对位置**——避免 post-hoc 修复，直接 built-in 正确 prior。

哲学差异：post-hoc pruning 在 fix 一个 "wrong default"；TLM 在一开始就 avoid 这个 default。

---

## 一些 potential concerns

### 实验规模偏小

440M / 760M / 1.3B，最大训练 100B tokens。7B+ / 1T+ tokens 上是否保持 gain 是 open question。

我的直觉：如果 late-layer redundancy 随 scale 增加（Gromov et al. 2024 的 "unreasonable ineffectiveness of deeper layers" 暗示），更大 model 可能 benefit from 更激进 taper ratio，比如 2.0/0.3。这是个值得 scale-up 验证的方向。

### Sweep 范围窄

只测了 3 种 schedule，1 个 model scale 上做 sweep。可能有 learnable schedule、power-law decay、piecewise quadratic 等更好。一个 data-driven 的做法是直接 differentiable architecture search 学习 $d_{\text{ff}}(l)$ 形状。

### 与 emergent capability 的关系

某些 emergent capability（in-context learning、CoT）可能在 late layers 实现。Table 2 测的是 commonsense reasoning，没测复杂 reasoning。如果 tapering 削弱 late layers 损害这些 capability，问题就大了。

Table 3 的 NIAH 是 partial sanity check，但还需要更多——比如 ICL、CoT、long-horizon planning benchmarks。

### Tapering 与 LayerNorm 的 interaction

Tapering 改变 $d_{\text{ff}}$，MLP 输出 magnitude 的 statistics 可能改变。LayerNorm/RMSNorm 在 MLP 之后会 normalize，但是否完全 absorb 这个变化？如果不同层 MLP 输出 variance 不同，norm 之后的有效 contribution 也会不同——可能放大或减弱 tapering 效果。

---

## 一句话总结

这篇 paper 说：**"层间参数均匀分配" 是一个被忽视的 suboptimal default。把 MLP width 用 cosine schedule 从 1.5×baseline 滑到 0.5×baseline，免费提升 PPL 和下游 task，跨 4 种 architecture 都 work，mechanism 上有 layer-wise novelty 的直接证据支撑。**

价值不在于 1.84 PPL 的单次 improvement，而在于打开了一个 design axis：**depth-aware capacity allocation**。我预期未来 12-18 个月会看到一系列 follow-up，把 tapering 推到 attention head count、SSM state size、MoE expert count，以及 ViT/DiT/multimodal models 上。

---

## Key Reference Links

- 原论文：基于附件内容，arXiv 待发布
- Transformer: https://arxiv.org/abs/1706.03762
- Geva et al. 2021 (FFN as memory): https://aclanthology.org/2021.emnlp-main.446/
- Tuned Lens: https://arxiv.org/abs/2303.08112
- LayerSkip: https://arxiv.org/abs/2404.16710
- Gromov (deeper layers ineffectiveness): https://arxiv.org/abs/2403.17887
- Titans: https://arxiv.org/abs/2501.00663
- Hope-attention (Nested Learning): https://arxiv.org/abs/2512.24695
- Gated Attention: https://arxiv.org/abs/2505.06708
- SwiGLU: https://arxiv.org/abs/2002.05202
- SliceGPT: https://arxiv.org/abs/2401.15024
- MoD: https://arxiv.org/abs/2404.02258
- LayerDrop: https://arxiv.org/abs/1909.11556
- Switch Transformer (MoE): https://www.jmlr.org/papers/v23/21-0998.html
- Funnel-Transformer: https://arxiv.org/abs/2006.02881
- GPT-2: https://openai.com/research/better-language-models

---

简单说，这是少数让我看完想立刻去 code 里改几行的 paper——零成本、零 downside、跨 architecture 都 work，有 mechanism 解释。下次 pre-train 任何 LM，cosine taper MLP width 1.5→0.5，no excuses。

---

# Tapered Language Models 深度解读

Andrej，这篇论文触及了一个非常 fundamental 的设计 default——所有 modern LM 几乎都沿用了 original Transformer 的"层间参数均匀分配"约定。让我把这篇 paper 拆开讲透，并 build 你的 intuition。

---

## 1. 论文的核心 question

Modern language models 表面看似多样——Transformer、Gated Attention、SSM/LinRNN、Memory-based architectures (Titans, Atlas)——但它们共享一个共同的 chassis:

```
stack of L identical layers, each = token-mixing module M + FFN F
```

每个 layer 拿到完全相同的 parameter budget。这个 uniform allocation 是从 Vaswani et al. 2017 一路继承下来的 default，几乎从未被 seriously challenged。

但有一堆 independent 的 evidence streams 指向同一个方向——**layer importance 是 non-uniform 的**：

- **Early-exit / tuned lens** (Belrose et al. 2023, Elbayad et al. 2020): residual stream 在最后几层之前已经收敛到最终预测，https://arxiv.org/abs/2303.08112
- **Layer-skipping** (Elhoushi et al. 2024): 推理时绕过后层，output quality 几乎不变, https://arxiv.org/abs/2404.16710
- **Structured redundancy** (Men et al. 2025, Gromov et al. 2024): 后期层被 remove 后 performance loss 令人惊讶地小, https://arxiv.org/abs/2403.17887
- **Interpretability** (Geva et al. 2021): FFN 是 key-value memory，低层 capture shallow syntactic，高层 encode semantic, https://aclanthology.org/2021.emnlp-main.446/

自然 question 浮现：**如果 layer importance 是 non-uniform 的，为什么 layer capacity 是 uniform 的？**

---

## 2. Motivating Experiment——一个 blunt 但 striking 的 test

在 440M Transformer 上做 coarse piecewise allocation：

- 把 layers 分成 early / middle / late 三等份
- 给每组分配不同的 MLP intermediate width $d_{\text{ff}}$
- 关键约束：**所有 configuration 的 total parameter count 严格相同**

四种 configuration 对比（Figure 2）：

| Configuration | Validation PPL | Δ vs Uniform |
|---|---|---|
| Uniform (baseline) | 16.28 | — |
| Wider-early (capacity 集中在前层) | **15.96** | -0.32 |
| Wider-middle | 16.61 | +0.33 |
| Wider-late (集中后层) | 17.29 | +1.01 |

关键 insight：**在完全相同的 parameter budget 下，allocation 方向不同导致 PPL 差异超过 1 个 point。**

front-loading capacity helps，back-loading actively hurts。Capacity 不是 passive resource 平均分布，应该放到最需要的地方——early layers。

---

## 3. Tapered Language Models 的形式化定义

### 3.1 一般性 formulation

考虑任何 architectural component $C$ 在 layer $l$ 处有一个 per-layer dimension $d_C(l)$，$l \in \{0, 1, \ldots, L-1\}$。

Uniform design: $d_C(l) = d_C^{\text{baseline}}$ for all $l$.

Tapered design (公式3):
$$
d_C(l+1) \le d_C(l) \quad \text{for all } l, \qquad \frac{1}{L}\sum_{l=0}^{L-1} d_C(l) = d_C^{\text{baseline}}
$$

变量解释：
- $d_C(l)$: 第 $l$ 层 component $C$ 的 dimension
- $d_C^{\text{baseline}}$: uniform baseline 的固定 dimension
- $L$: 总层数
- 第一个不等式：capacity 随 depth 单调(weakly)递减
- 第二个等式：average per-layer dimension 保持不变 → total parameter budget 保持

### 3.2 为什么这个 formulation 优美

这个 principle 是 **architecture-agnostic** 的，$d_C$ 可以是：
- MLP intermediate dimension $d_{\text{ff}}$
- Attention head count
- Key-value dimension
- Recurrent state size (SSM/LinRNN)
- Memory slot count (Titans/Atlas)
- MoE expert count

### 3.3 为什么选择 MLP width 作为 instantiation

MLP 是最 natural 的 site：
1. **Parameter dominance**: 在 modern LM 中 MLP 占绝大部分 parameter，通常 2/3 以上
2. **Clean axis**: $d_{\text{ff}}$ 是一个 scalar width，独立于 surrounding architecture 可调
3. **Consistent across architectures**: vanilla FFN 是 $2dd_{\text{ff}}$，SwiGLU 是 $3dd_{\text{ff}}$，都 linear in $d_{\text{ff}}$

---

## 4. 三种 Decay Schedule 详细对比

Tapering 用 $d_{\text{start}}$ 和 $d_{\text{end}}$ 参数化（$d_{\text{start}} > d_{\text{end}}$），然后通过 schedule 决定中间形状。

### 4.1 Linear Schedule (公式4)

$$
d_{\text{ff}}(l) = d_{\text{start}} - (d_{\text{start}} - d_{\text{end}}) \cdot \frac{l}{L-1}
$$

变量含义：
- $l$: 当前层 index，$l \in \{0, 1, \ldots, L-1\}$
- $L-1$: 最后一层的 index（用于 normalize $l$ 到 $[0,1]$）
- $d_{\text{start}}$: layer 0 的 width
- $d_{\text{end}}$: layer $L-1$ 的 width
- $(d_{\text{start}} - d_{\text{end}})$: 总衰减幅度
- $\frac{l}{L-1}$: 归一化深度比例

几何：等速衰减，两端无 plateau，所有层被 equally engaged。

### 4.2 Cosine Schedule (公式5) — 论文的 winner

$$
d_{\text{ff}}(l) = d_{\text{end}} + \frac{d_{\text{start}} - d_{\text{end}}}{2} \left(1 + \cos\frac{\pi l}{L-1}\right)
$$

变量含义：
- $\frac{\pi l}{L-1}$: 当 $l$ 从 $0$ 走到 $L-1$，这个量从 $0$ 走到 $\pi$
- $\cos(\cdot)$: 从 $\cos(0)=1$ 走到 $\cos(\pi)=-1$
- $(1+\cos(\cdot))$: 从 $2$ 走到 $0$
- $\frac{d_{\text{start}}-d_{\text{end}}}{2}$: 振幅，将 $[0,2]$ 映射到 $[0, d_{\text{start}}-d_{\text{end}}]$
- 整体：$l=0$ 时 $d_{\text{end}} + (d_{\text{start}}-d_{\text{end}}) = d_{\text{start}}$；$l=L-1$ 时 $d_{\text{end}} + 0 = d_{\text{end}}$ ✓

几何：半余弦曲线，**两端 soft plateaus，中点陡降最明显**。这是 linear 和 sigmoid 的中间地带——既有平滑的 entry/exit，又用 gradual transition 让更多 intermediate widths 都被用到。

### 4.3 Sigmoid Schedule (公式6)

$$
d_{\text{ff}}(l) = d_{\text{end}} + \frac{d_{\text{start}} - d_{\text{end}}}{1 + e^{k\left(\frac{l}{L-1} - 0.5\right)}}
$$

变量含义：
- $k$: steepness 参数，实验中固定 $k=10$
- $\frac{l}{L-1} - 0.5$: 围绕中点 0.5 的偏移
- $e^{k(\cdot)}$: 当 $l < L/2$ 时 $e^{k(\cdot)} \to 0$，width 接近 $d_{\text{start}} + (d_{\text{start}}-d_{\text{end}}) = d_{\text{start}}$ ... wait 让我重新算

更准确：
- $l=0$: $e^{k(0 - 0.5)} = e^{-5} \approx 0.0067$，width ≈ $d_{\text{end}} + \frac{d_{\text{start}}-d_{\text{end}}}{1.0067} \approx d_{\text{start}}$
- $l=L-1$: $e^{k(1-0.5)} = e^5 \approx 148$，width ≈ $d_{\text{end}} + \frac{d_{\text{start}}-d_{\text{end}}}{149} \approx d_{\text{end}}$
- $l=L/2$: $e^0 = 1$，width = $d_{\text{end}} + \frac{d_{\text{start}}-d_{\text{end}}}{2}$ = midpoint

几何：转换集中在 narrow band around midpoint，两端 near-binary。

### 4.4 三者对比的 intuition

| Schedule | 几何特征 | 端点 plateau | 中间过渡带宽 |
|---|---|---|---|
| Linear | 等速下降 | 无 | 全层 |
| Cosine | 半余弦 | 两端软 plateau | 较宽 |
| Sigmoid | 阶跃 | 强 plateau | 窄带 around midpoint |

后续实验表明：**cosine 严格 dominate**。这暗示最优 allocation 既不是 uniform-rate (linear)，也不是 near-binary (sigmoid)，而是 smooth gradient across depth。

---

## 5. Parameter Preservation 的精妙之处

公式(7) 的约束：
$$
\frac{1}{L}\sum_{l=0}^{L-1} d_{\text{ff}}(l) = d_{\text{ff}}^{\text{baseline}}
$$

由于 MLP 参数和 FLOPs 都 linear in $d_{\text{ff}}$（vanilla FFN: $2dd_{\text{ff}}$，SwiGLU: $3dd_{\text{ff}}$），**这个 average 约束同时保持 total 参数、training FLOPs、inference FLOPs**。

Implementation 细节：
- $d_{\text{ff}}(l)$ round 到最近的 16 倍数（hardware-friendly）
- 第一层和最后一层严格等于 $d_{\text{start}}$ 和 $d_{\text{end}}$
- 内部 layers 在 16-unit increment 内调整以严格满足公式(7)
- 单调递减顺序保持

也就是说，整个 tapering 操作是 **free lunch**：不增加任何 cost，只是把 capacity 在 depth 维度上重新分布。

---

## 6. Schedule & Width Sweep（Table 1）

在 440M Transformer 上做 5×3 grid search，固定所有 15 个 configuration 的 total params/FLOPs 与 uniform baseline 严格相同。

**Uniform baseline PPL: 16.28**

| Taper range $(d_{\text{start}} \to d_{\text{end}})$ × baseline | Cosine PPL | Δ | Linear PPL | Δ | Sigmoid PPL | Δ |
|---|---|---|---|---|---|---|
| 1.25 → 0.75 | 15.18 | -1.10 | 15.96 | -0.32 | 16.44 | +0.16 |
| 1.375 → 0.625 | 14.59 | -1.69 | 15.80 | -0.48 | 16.44 | +0.16 |
| **1.50 → 0.50** | **14.44** | **-1.84** | 15.96 | -0.32 | 16.12 | -0.16 |
| 1.625 → 0.375 | 14.59 | -1.69 | 15.64 | -0.64 | 15.96 | -0.32 |
| 1.75 → 0.25 | 15.49 | -0.79 | 16.28 | 0.00 | 17.12 | +0.84 |

关键 observations：

1. **Strict ordering**: 在所有 5 个 width ratio 下，cosine < linear < sigmoid（PPL 越小越好）。这个 ordering 严格 hold。

2. **Cosine 的 worst > Linear 的 best**: cosine worst (1.75/0.25, 15.49) 优于 linear best (1.625/0.375, 15.64)。这说明 schedule 的选择比 width ratio 的微调更重要。

3. **U-shape in width ratio**: 对 cosine，PPL 在 1.5/0.5 达到 minimum (14.44)，两侧 degrade。
   - 过窄 (1.25/0.75): redistribution 太 mild，没充分 exploit asymmetry
   - 过宽 (1.75/0.25): 太多 capacity 推到 early layers，starve 后期 layers
   - 1.5/0.5 是 sweet spot

最终选择：**cosine schedule with $d_{\text{start}}/d_{\text{end}} = 1.5/0.5$**，固定不变 carry forward 到所有后续实验。

---

## 7. Main Results（Table 2）—— 跨架构、跨规模

把 1.5/0.5 cosine 配置 unchanged 应用到 760M 和 1.3B 两个 scale，4 种 architecture。

### 7.1 760M / 50B tokens

| Model | Wiki PPL | LMB PPL | LMB acc | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | SIQA | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer++ | 21.86 | 22.29 | 39.0 | 68.7 | 46.3 | 57.1 | 66.8 | 35.3 | 42.5 | 62.3 | 52.25 |
| + Tapered | 21.42 | 21.25 | 40.1 | 69.3 | 47.0 | 57.3 | 66.7 | 35.9 | 43.0 | 63.4 | **52.84** |
| Gated Attention | 20.74 | 21.85 | 39.7 | 69.2 | 46.3 | 57.9 | 68.4 | 35.5 | 41.8 | 62.1 | 52.61 |
| + Tapered | 19.98 | 21.44 | 40.0 | 69.3 | 46.8 | 57.8 | 69.1 | 35.8 | 41.6 | 62.6 | **52.88** |
| Hope-attention | 20.62 | 21.29 | 40.2 | 70.1 | 50.6 | 56.8 | 69.9 | 37.1 | 41.3 | 63.5 | 53.69 |
| + Tapered | 20.50 | 21.07 | 40.3 | 70.7 | 51.0 | 57.4 | 69.2 | 38.1 | 41.8 | 63.9 | **54.05** |
| Titans | 21.58 | 23.09 | 39.2 | 67.7 | 50.0 | 52.8 | 68.0 | 35.6 | 41.4 | 63.7 | 52.30 |
| + Tapered | 20.77 | 22.92 | 39.9 | 69.0 | 51.6 | 54.5 | 67.9 | 36.1 | 42.5 | 64.8 | **53.29** |

### 7.2 1.3B / 100B tokens

| Model | Wiki PPL | LMB PPL | LMB acc | Avg |
|---|---|---|---|---|
| Transformer++ | 17.39 | 17.62 | 45.1 | 56.05 |
| + Tapered | 17.17 | 16.93 | 45.7 | **56.38** |
| Gated Attention | 16.03 | 14.26 | 46.2 | 56.51 |
| + Tapered | 15.92 | 14.11 | 46.5 | **56.80** |
| Hope-attention | 15.91 | 15.48 | 47.0 | 56.95 |
| + Tapered | 15.94 | 14.92 | 47.1 | **57.05** |
| Titans | 16.05 | 14.19 | 46.9 | 56.73 |
| + Tapered | 15.76 | 14.04 | 46.9 | **57.08** |

### 7.3 关键 observations

1. **Commonsense avg accuracy**: 8 次比较 (4 arch × 2 scale) **全部 improve，无一例外**。
2. **LAMBADA PPL**: 8 次比较全部 improve。
3. **WikiText PPL**: 8 次中 7 次 improve（仅 Hope-attention 1.3B 微回退 0.03）。
4. **Scale consistency**: gain 在 1.3B 上没有 diminish，仍然保持。这表明 depth-aware allocation 不随 scale 饱和。
5. **Architecture transferability**: 4 种 token-mixing module 截然不同——softmax attention、gated attention、recurrent self-modifying memory、neural long-term memory——但 tapering 都 work。这强烈暗示 principle 是 about **MLP stack 在 depth 上的 capacity allocation**，而非 token-mixing 的 property。

值得注意：Titans (Behrouz et al. 2024) 是 test-time memorization 的 architecture, https://arxiv.org/abs/2501.00663；Hope-attention 是 nested learning 的 self-modifying memory, https://arxiv.org/abs/2512.24695。在这样不同的 chassis 上仍然 transfer，是很强的信号。

---

## 8. Layer-wise Novelty 分析（Section 4）——为什么 tapering works

这是论文 mechanistic explanation 的核心。前面只证明 "tapering helps"，这里 probe "why"。

### 8.1 两个 cosine quantity 的定义

回到 §2.1 的 notation: $z_l = h_l + \mathcal{M}_l(h_l)$, $h_{l+1} = z_l + \mathcal{F}_l(z_l)$.

**Block-level update alignment** (公式8):
$$
\rho_l^{\text{block}} = \cos(h_{l+1} - h_l, h_l)
$$

变量含义：
- $h_{l+1} - h_l$: 该 block 完整的 additive contribution（attention + MLP 一起对 residual stream 写入的 delta）
- $h_l$: 进入该 block 的 residual stream
- $\cos(\cdot, \cdot)$: cosine similarity，衡量两个向量的方向对齐程度

**MLP-only alignment** (公式9):
$$
\rho_l^{\text{MLP}} = \cos(\mathcal{F}_l(z_l), h_l)
$$

变量含义：
- $\mathcal{F}_l(z_l)$: 第 $l$ 层 MLP 的输出（被 taper 的 component）
- $h_l$: residual stream
- 这个 quantity 隔离出被 taper 的 MLP 组件

### 8.2 物理意义

- $\rho \approx 0$: 该层写入的内容与 residual **正交**——在写**新信息**
- $\rho > 0$: 该层写入的内容与 residual **同向**——在**强化已有方向**，等于在 echo / refine 而非 compute new features
- $\rho$ 随深度上升 = **novelty diminishing**

### 8.3 GPT-2 family 上的实验

在 GPT-2 family (124M 到 1.5B) 的 pretrained checkpoints 上，用 2048 tokens 的 WikiText-2 测量每层的 $\rho_l^{\text{block}}$ 和 $\rho_l^{\text{MLP}}$。Figure 4 的 pattern 在所有 model size 上 **一致**：

- 两个 quantity 都在 early-middle layers 降到 low value（≈0 或略负，写新信息）
- 在 second half of network 单调上升（强化已有内容）
- Pearson correlation with layer index 都是正的：
  - $\rho_l^{\text{MLP}}$: $r = 0.49$ 到 $r = 0.71$（更 tight 的 monotone trend）
  - $\rho_l^{\text{block}}$: $r = 0.27$ 到 $r = 0.71$

### 8.4 为什么 MLP signal 比 block signal 更 clean

MLP 的 alignment 比 block 的更 tight，这本身有信息量：

1. **MLP 自身 becoming redundant**——不是 attention 在干活、MLP 坐着吃白饭，而是 MLP 自己写的内容越来越 echo residual
2. **Depth-wise pattern 不是 MLP-specific**——block-level 也 rise，说明 principle 应该 generalize 到其他 parameter-bearing axis

### 8.5 与 tapering 的 direct connection

这是 intuition 的关键拼图：

- **Late layers** (uniform capacity 下)：MLP 输出与 residual 高度 aligned，额外的 hidden dimension 在"做无用功"（reinforce 而非 innovate）
- **Early layers**: MLP 输出 orthogonal 于 residual，added capacity 有处可用（writing genuinely new features）

Tapering 做的事情：
1. 在 alignment 最大处（late）reduce hidden dimension → 释放参数
2. 在 alignment 最小处（early）增加 hidden dimension → 把释放的参数送到最有用的地方

这就解释了 Figure 2 的 asymmetry：**front-loading helps 因为 early MLPs 用得上 capacity；back-loading hurts 因为 late MLPs 用不上**。

---

## 9. Long-Context Retrieval（Table 3）——验证 tapering 不损害 long context

Needle-in-a-Haystack (NIAH) 测试，三种 single-needle variant (S-NIAH-1 passkey / S-NIAH-2 number / S-NIAH-3 UUID) 加 multi-query variant，在 4K/8K/16K context length 上：

关键 finding：**Tapered models match or improve over uniform counterparts across the table，gains 集中在 absolute score 最低的 hard cells**。

例如 16K S-NIAH-3 UUID 多查询：
- Transformer++ uniform: 27.2 → tapered: 27.6
- Titans uniform: 11.8 → tapered: 12.4

这意味着 tapering redistribute capacity 没有以牺牲 long-context retrieval 为代价——这是一个非常重要的 sanity check，因为有人可能担心后期 layer capacity 减少会破坏 long-range 信息整合。

---

## 10. 与 Related Work 的对比

### 10.1 Non-uniform allocation across depth 的 prior work

| 工作 | 做法 | 与 TLM 区别 |
|---|---|---|
| Funnel-Transformer (Dai et al. 2020) | pooling sequence length | 不动 MLP width |
| Mixture-of-Depths (Raposo et al. 2024) | 动态 route compute to tokens | token 维度路由，不是 layer 维度 |
| LayerDrop (Fan et al. 2019) | training 时 drop entire layers | 不 redistribute capacity |
| Matformer (Kudugunta et al. 2024) | 层内 nested variable-size blocks | 弹性 inference，不是 monotonic taper |
| Block-wise scaling (Mehta et al. 2020) | per-layer FFN multiplier + head count 联合变 | 多轴联合 scaling |
| Baroian & Notebomer 2025 | 几种 layer-wise shapes ablation | 没找到 clear winner |
| Ikeda et al. 2025 | deactivate FFN entirely in some layers | binary allocation，改变 effective depth |

TLM 的 unique 之处：
- **单轴**: 只动 $d_{\text{ff}}$，其他 hyperparameter 不变
- **Smooth decay**: 替代 sharp block transitions
- **Equal-cost baseline**: 严格 matched params + FLOPs
- **Architecture-agnostic**: 跨 4 种 architecture 验证

### 10.2 Layer importance 的 prior analysis

- Early-exit (Elbayad et al. 2020, Belrose et al. 2023): residual converge before final layer
- Layer-skipping (Elhoushi et al. 2024): bypass later layers at inference
- Redundancy (Men et al. 2025, Gromov et al. 2024): later layers removable, https://arxiv.org/abs/2403.17887
- Interpretability (Geva et al. 2021): FFN as key-value memory, shallow→semantic shift, https://aclanthology.org/2021.emnlp-main.446/
- Activation steering (Bayat et al. 2025): steerability 在 intermediate layers 最强
- Post-training pruning (SliceGPT, Sharma et al. 2023): 对 MLP weights 做 selective rank reduction，但需要 extra pruning + fine-tuning stage

TLM 的 contribution：把这些 "layer importance non-uniform" 的分析证据，转化为 **pre-training 时就内置的 architectural choice**——不需要 post-hoc pruning/fine-tuning，免费搭车。

---

## 11. Intuition Building——给 Karpathy 的深度思考

### 11.1 与 ResNet "identity mapping" insight 的呼应

He et al. 2015 的 identity shortcut insight：深层 ResNet block 学到的是 residual，identity 已经是 good default，所以深层 block 只需要 small perturbation。

TLM 揭示的 LM 中类似 pattern：**late MLP blocks 的输出与 residual 高度 aligned，相当于在学一个"小扰动 + 强化"而不是 "大变换"**。这与 Geva et al. 2021 把 FFN 描述为 key-value memory 的 view 一致——late memory slots 在强化已有 activation pattern，early memory slots 在写入新的 syntactic/semantic pattern。

### 11.2 与 U-Net / Funnel-Transformer 的反向对比

U-Net 在 encoder 端 reduce spatial resolution 但 increase channel capacity，decoder 端反过来。Funnel-Transformer 在深度上 pooling sequence length。

TLM 是不同的轴：**不动 sequence length，不动 hidden dim $d$，只动 MLP intermediate dim $d_{\text{ff}}$**。这是个非常 surgical 的 intervention。

### 11.3 与 Mixture-of-Experts 的关系

MoE 在 **token dimension** 上 allocate capacity (per-token routing to experts)。TLM 在 **layer dimension** 上 allocate capacity (per-layer width)。

二者是 orthogonal 的——你可以想象一个 TLM + MoE 的组合：early layers 用更多 experts + 更宽 MLP，late layers 用更少 experts + 更窄 MLP。论文 §5 提到这个方向但留作 future work。

### 11.4 与 MoD (Mixture-of-Depths) 的对比

MoD 在 token 维度动态路由 compute——某些 token 跳过某些 layer。TLM 在 layer 维度静态分配 capacity——所有 token 经过所有 layer，但每个 layer 的 capacity 不同。

TLM 是 design-time static decision，MoD 是 inference-time dynamic decision。二者可以叠加。

### 11.5 与 SliceGPT / Post-training pruning 的对比

SliceGPT (Ashkboos et al. 2024) 在 post-training 阶段对 MLP weights 做 structured pruning，需要额外 fine-tuning recover performance, https://arxiv.org/abs/2401.15024。Sharma et al. 2023 对 specific layers 做 rank reduction, https://arxiv.org/abs/2312.13558。

这些方法都是 **post-hoc compression**——先 train 一个 uniform model，再 prune。TLM 是 **built-in architectural prior**——pre-training 时就让 capacity 在 depth 上正确分布。

哲学差异：post-hoc pruning 在 fix 一个 "wrong" default；TLM 在一开始就 avoid 这个 default。

### 11.6 为什么 cosine 优于 linear 和 sigmoid

直觉解释：
- **Linear** 假设每层"额外 capacity 的边际效用"等速衰减——太简单
- **Sigmoid** 假设 capacity 集中在两端、中间 sharp transition——太 binary，等于退化成 wider-early vs wider-late 的 piecewise
- **Cosine** 假设两端 transition 缓慢、中段 transition 陡——这对应 **early layers 的 novelty 缓慢 decay、middle layers 是 transition phase、late layers 进入 refinement phase**

cosine 的形状匹配了 GPT-2 family 测量到的 $\rho_l$ 曲线形状：early drop + late rise。

### 11.7 关于 1.5/0.5 的 sweet spot

为什么是 1.5/0.5 而不是 1.75/0.25？我直觉是：

- 1.5/0.5: late layers 还保留 50% capacity——足够做 refinement
- 1.75/0.25: late layers 只剩 25%——starve 了，连 refine 都做不到
- 1.25/0.75: early layers 只多 25%——不够 front-load，没充分利用 early MLPs 的 novelty capacity

1.5/0.5 是 "strong front-loading without starving the back" 的平衡。

### 11.8 一个有趣的 open question

论文 §6 提到 sweep 只在 440M 上做，配置 unchanged transfer 到 760M 和 1.3B。这意味着 **报告的 gain 是 lower bound**——更大的 model 或不同 architecture 可能有不同最优 ratio。

如果 deep model 的 late-layer redundancy 随 scale 增加（这是 Gromov et al. 2024 "unreasonable ineffectiveness of deeper layers" 的暗示），那么 **更大的 model 可能 benefit from 更激进的 taper ratio**，比如 2.0/0.3。这是个值得 scale-up 实验的方向。

---

## 12. 我的延伸思考与 potential critiques

### 12.1 实验规模偏小

440M / 760M / 1.3B 都是小模型，最大训练 100B tokens。在 7B+ / 1T+ tokens 上是否保持 gain 仍是 open question。如果 late-layer redundancy 在 large scale 上更严重，tapering 可能带来更大收益；如果 large model 的 late layers 学到更多 emergent capability，tapering 可能有上限。

### 12.2 Cosine schedule 的"运气成分"

cosine 在三种 schedule 中 win，但只 sweep 了三种。可能有其他 schedule（如 learnable schedule、piecewise quadratic、power law decay）效果更好。一个 data-driven 的 approach 是直接学习 $d_{\text{ff}}(l)$ 的最优形状，例如 differentiable architecture search。

### 12.3 与 Geva et al. 2021 的张力

Geva et al. 说 late-layer FFN encode semantic patterns——这听起来像是 late layers 在做重要工作。TLM 说 late layers redundant。这二者如何 reconcile？

我的解读：late FFN **确实** encode semantic patterns，但它们 encode 的方式是 **refinement / amplification of patterns already present in residual**，而不是写入完全 new 的 semantic content。capacity 需求 less，但 role 仍然存在——只是不需要那么多 parameters 来 fulfill this role。

### 12.4 Vision / Diffusion / Multimodal 的 transfer

论文 §7 提到 ViT、DiT、multimodal models 都继承 uniform capacity default。如果 tapering 在这些 domain 也 transfer，将是一个相当 general 的 architectural principle。

直觉上，ViT 也应该有类似 pattern——early patches 编码 low-level features (edge, texture)，late patches 编码 high-level semantics。但 ViT 的"layer novelty"可能下降得更早，因为 vision 的 semantic hierarchy 比 language 更 compressed。Diffusion Transformer 的 dynamic 可能完全不同——timestep conditioning 让所有 layers 都 important。

### 12.5 Tapering 与 emergent capabilities 的关系

某些 emergent capabilities (in-context learning, chain-of-thought) 可能在 late layers 实现。如果 tapering 削弱 late layers，会不会损害这些 capability？

Table 2 的 commonsense reasoning 没有退化，但更复杂的 reasoning 没测。一个 follow-up 应该测 in-context learning performance、CoT benchmarks、long-horizon planning。Table 3 的 NIAH 是一个 partial sanity check，但还需要更多。

### 12.6 与 LayerNorm / RMSNorm 的 interaction

Tapering 改变 $d_{\text{ff}}$，可能改变 MLP 输出 magnitude 的 statistics。LayerNorm/RMSNorm 在 MLP 之后会重新 normalize，但是否完全 absorb 这个变化？如果不同层的 MLP 输出 variance 不同，norm 之后的 effective contribution 也会不同——这可能放大或减弱 tapering 的 effect。

---

## 13. 总结：TLM 的真正 contribution

TLM 的 contribution 不是发明新 mechanism，而是 **指出一个被忽视的设计 default，并系统地证明它是 suboptimal**。这种"free lunch"式的工作价值很高——它告诉整个 community：

> 在你下一次 pre-train 7B model 之前，把 MLP width cosine taper 一下，1.5/0.5 ratio，固定不动——免费的 perplexity 提升。

这种 architectural prior 的优美之处：
- **零额外 cost**：参数和 FLOPs 严格保持
- **Implementation 极简**：只改 per-layer $d_{\text{ff}}$，几行代码
- **Architecture-agnostic**：跨 4 种 token-mixing family 都 work
- **Mechanistically motivated**：有 layer-wise novelty 的直接证据支撑
- **Scale 不饱和**：gain 在 1.3B 上保持

更深层地，TLM 提示一个 broader principle：**深度方向上 capacity 的非均匀分配是被 under-explored 的 design axis**。未来工作可以探索：
- Tapering attention head count
- Tapering key-value dimension
- Tapering recurrent state size (SSM/LinRNN)
- Tapering memory slot count (Titans)
- Tapering MoE expert count
- Learnable taper schedules
- Scale-dependent optimal taper ratio

## Reference Links

- 论文本身（基于附件内容，arXiv 链接需作者发布后查询）
- Vaswani et al. 2017 (Transformer): https://arxiv.org/abs/1706.03762
- Geva et al. 2021 (FFN as key-value memories): https://aclanthology.org/2021.emnlp-main.446/
- Belrose et al. 2023 (Tuned Lens): https://arxiv.org/abs/2303.08112
- Elhoushi et al. 2024 (LayerSkip): https://arxiv.org/abs/2404.16710
- Gromov et al. 2024 (Deeper layers ineffectiveness): https://arxiv.org/abs/2403.17887
- Men et al. 2025 (ShortGPT): https://aclanthology.org/2025.findings-acl.1133/
- Behrouz et al. 2024 (Titans): https://arxiv.org/abs/2501.00663
- Behrouz et al. 2025b (Hope-attention / Nested Learning): https://arxiv.org/abs/2512.24695
- Qiu et al. 2025 (Gated Attention): https://arxiv.org/abs/2505.06708
- Ashkboos et al. 2024 (SliceGPT): https://arxiv.org/abs/2401.15024
- Sharma et al. 2023 (Layer-selective rank reduction): https://arxiv.org/abs/2312.13558
- Shazeer 2020 (SwiGLU): https://arxiv.org/abs/2002.05202
- Bayat et al. 2025 (Activation steering): https://arxiv.org/abs/2503.00177
- Radford et al. 2019 (GPT-2): https://openai.com/research/better-language-models
- Merity et al. 2016 (WikiText): https://arxiv.org/abs/1609.07843
- Paperno et al. 2016 (LAMBADA): https://aclanthology.org/P16-1144/
- Fedus et al. 2022 (Switch Transformers / MoE): https://www.jmlr.org/papers/v23/21-0998.html
- Raposo et al. 2024 (Mixture-of-Depths): https://arxiv.org/abs/2404.02258
- Dai et al. 2020 (Funnel-Transformer): https://arxiv.org/abs/2006.02881
- Fan et al. 2019 (LayerDrop): https://arxiv.org/abs/1909.11556
- Kudugunta et al. 2024 (Matformer): https://arxiv.org/abs/2310.02788
- Mehta et al. 2020 (DeLighT): https://arxiv.org/abs/2008.00623
- Dosovitskiy et al. 2020 (ViT): https://arxiv.org/abs/2010.11929
- Peebles & Xie 2023 (DiT): https://arxiv.org/abs/2212.09748

---

Andrej，这篇 paper 给我的感觉类似于当年 LayerNorm 替代 BatchNorm、或者 SwiGLU 替代 vanilla FFN 的工作——它指出一个 community-wide 的 default 是 suboptimal 的，并提供一个 simple、free、transferable 的 alternative。真正的 value 不在于 1.84 PPL 的单次 improvement，而在于打开了一个 design axis：**depth-aware capacity allocation**。我预期未来 12-18 个月会看到一系列 follow-up 工作，把 tapering 推到 attention head count、SSM state size、MoE expert count，以及 ViT/DiT/multimodal models 上。这是个值得密切 follow 的方向。
