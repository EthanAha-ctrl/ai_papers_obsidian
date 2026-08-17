---
source_pdf: ViT-5.pdf
paper_sha256: 5d93f54a43b1edf953d263024b47992faaf82da164f8e24e74df9c20fbdc774d
processed_at: '2026-08-13T02:28:59-07:00'
target_folder: LLM-from-scratch/ViT
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ViT-5 用人话说

## 一句话版本

ViT 从 2020 年到现在，骨架基本没变过。LLM 这五年倒是把 norm、激活、位置编码、attention 稳定性这些细节打磨了个遍。这篇 paper 就是把这些 LLM 端的"现代化装修"逐个搬到 ViT 上，做了一轮严格 ablation，发现大部分有用，但有一个（SwiGLU）在 vision 上反而有害，最后拼出一个叫 ViT-5 的东西，各个 benchmark 都涨一点。

---

## 为什么要做这件事

你想想，ViT 这东西 2020 年底出来 ([Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929))，之后 vision 社区主要在干嘛？搞 hierarchical（Swin）、搞 hybrid（CoAtNet）、搞预训练方法（MAE、DINO）。但 plain ViT 的**架构本身**几乎没动过——DeiT-III ([Touvron et al. 2022](https://arxiv.org/abs/2204.07118)) 相对原始 ViT 就加了一个 LayerScale，其他没变。SigLIP-2、Qwen3-VL 这些 2025 年的多模态大模型，vision encoder 还是基本原版 ViT。

同期 LLM 端呢？LLaMA ([Touvron et al. 2023](https://arxiv.org/abs/2302.13971)) 把 LayerNorm 换成 RMSNorm，把 MLP 换成 SwiGLU，加了 RoPE；Qwen3 ([Yang et al. 2025](https://arxiv.org/abs/2505.09388)) 加了 QK-Norm、去掉了 QKV bias；Gemma3 ([Team et al. 2025](https://arxiv.org/abs/2503.19786)) 也加了 QK-Norm。五年下来，LLM 的"默认配方"跟原始 Transformer 已经差很多了。

所以作者问的问题特别简单：**ViT 的架构潜力是不是还没榨干？把 LLM 这五年的小改搬过来能涨多少？**

---

## 七个改动，逐个用大白话讲

### 1. LayerScale —— 相当于给每层输出装个"音量旋钮"

原始残差连接是 $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$，new signal 全量注入。

LayerScale ([CaiT, Touvron et al. 2021](https://arxiv.org/abs/2103.17239)) 改成 $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l) \odot \lambda$，其中 $\lambda$ 是 per-channel 可学习向量，初始化 $10^{-4}$。

**人话**：每层 block 的输出，每个 channel 上乘一个可学习的小系数，一开始很小（$10^{-4}$），让 new signal 一开始几乎不注入 residual stream，训练稳定了再慢慢学大。深网络靠这个才训得动。

**有意思的发现**：作者发现 LayerScale 跟 post-RMSNorm 在数学上几乎等价。post-norm 是 $(\mathbf{x}_l + \mathcal{F}) \odot \lambda_p / \text{RMS}(\mathbf{x}_l + \mathcal{F})$，因为深层网络里 $\|\mathbf{x}_l\| \gg \|\mathcal{F}\|$，RMS 基本由 $\mathbf{x}_l$ 决定，所以 post-norm 对 $\mathcal{F}$ 的有效缩放 ≈ $1/\text{RMS}(\mathbf{x}_l)$，跟 LayerScale 的效果一样——都是限制 new signal 对 residual stream 的相对能量。实验上两者性能几乎完全一致（Table 1）。LayerScale 更灵活、计算更省，所以选它。

### 2. RMSNorm 替代 LayerNorm —— 去掉没用的减均值

LayerNorm = 减均值 + 除标准差 + 学缩放偏移。
RMSNorm ([Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)) = 只除 RMS + 学缩放。

**人话**：研究发现 LayerNorm 里那个"减均值"操作几乎不携带信息，去掉反而少一点噪声。LLM 端 LLaMA、PaLM ([Chowdhery et al. 2023](https://arxiv.org/abs/2204.02311))、Gopher ([Rae et al. 2021](https://arxiv.org/abs/2112.11446)) 早就全换成 RMSNorm 了。ViT 上一样 work，涨 0.2%。

### 3. SwiGLU —— **这个被否决了，是全文最有意思的发现**

SwiGLU ([Shazeer 2020](https://arxiv.org/abs/2002.05202)) 是 LLM 端的现代默认 MLP：`FFN(x) = (Swish(xW1) ⊙ xW3)W2`。Swish 那一支天然做稀疏化（负值压到 0 附近），gate 那一支做动态选择。

**问题来了**：LayerScale 也在做 channel-wise gating（静态的），SwiGLU 也在做 channel-wise gating（动态的），两个 gating 叠一起 = **over-gating**，intermediate activation 过度稀疏。

Table 2 的 2×2 实验特别清楚：

| LayerScale | SwiGLU | ImageNet acc | FID |
|---|---|---|---|
| ✗ | ✗ | 83.86 | 15.80 |
| ✗ | ✓ | 83.94 | 15.48 |
| ✓ | ✓ | 83.70 | 16.22 ← **变差** |
| ✓ | ✗ | **84.16** | **14.57** |

**人话**：LLM 端的"黄金组合"（RMSNorm + RoPE + SwiGLU）里，SwiGLU 跟 ViT 端的 LayerScale 互相打架。两个都在做 channel 稀疏化，叠加过头了。所以 ViT-5 回退到原版 GeLU-MLP。

这是 paper 最反直觉、也最有价值的结论：**LLM 的 best practice 不能整包搬到 vision**。你在 nanoGPT 里默认用 SwiGLU，是因为 nanoGPT 没用 LayerScale；ViT 端因为 LayerScale 已经在，SwiGLU 就成了多余。

### 4. 位置编码：APE + 2D RoPE 一起用

原版 ViT 用 learnable absolute positional embedding (APE)。问题：APE 对 dynamic resolution 鲁棒性差，也缺显式 relative positional modeling。

作者引入 2D RoPE ([Su et al. 2024](https://arxiv.org/abs/2104.09864))，但**不丢 APE**。

**为什么不丢？** Figure 2 的 demo 特别直观：只用 RoPE 时，把图像的 patch 顺序翻转（patch-level flip），RoPE 视角下所有相对位置不变，模型看到两个完全不同的图当一样的。分类上无所谓，但分割、检测、生成都需要绝对位置信息。

**实验证据**：Figure 3，DeiT-III 在 224 训练、>256 测试性能暴跌；ViT-5 在 128→512 范围内稳定甚至单调上升。RoPE 的频率构造 $\theta_i = \text{base}^{-2i/d}$ 对 sequence length 天然鲁棒。

### 5. Register tokens 需要单独的高频 RoPE —— 全文最 elegant 的细节

[Registers (Darcet et al. 2024)](https://arxiv.org/abs/2309.16588) 的原始做法：在 patch tokens 后面 append 几个 learnable tokens，让它们吸收全局信息 artifact（attention map 上的高频噪声斑）。

**问题**：当你给 patch tokens 加了 RoPE，但没给 register tokens 加，会发生什么？

RoPE 把 query/key 在 channel 维度上做旋转，旋转角度正比于绝对位置。Patch 位置 $n$ 大、register 位置视为 0 时，register 与 patch 的内积随 $n$ 振荡——位置越远的 patch 与 register 的相似度被相位打散得越厉害。结果就是 register 被"推到序列尾部 patch 的关注盲区"，attention 分布被 implicit positional bias 扭曲。

Table 3 的数据：
- 不加 register：84.02
- 加 register 但不旋转：83.90 ← **比不加还差**
- 加 register + 同频率 RoPE：84.00
- 加 register + **高频率 RoPE**：**84.16**

**解决方案**：给 register 一套独立的 2D RoPE，频率 base 显著高于 patch 的。直觉是让 register 和 patch 占据不同的"频率带"——register 高频快旋转、patch 低频慢旋转，channel 维度上几乎正交，positional correlation 被 decouple。

这跟 LLM 端 [YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753) 的频域直觉是通的：不同 base 的 RoPE = 不同的位置频率分辨率，分配给不同角色 token 是一种 cheap positional disentangle。

### 6. QK-Norm —— 性能涨不多，但训练不炸了

$$Q' = \text{RMSNorm}(Q), \quad K' = \text{RMSNorm}(K)$$

性能只涨 0.1%，但 Figure 5 显示不加 QK-Norm 时 loss 有尖锐 spike，加了之后平滑收敛。Qwen3、Gemma3 都观察到同样现象。

**人话**：Q、K 是高维线性投影输出，norm 容易随训练 drift，导致 $QK^T$ 的 scale 不稳定，softmax 时进饱和区或欠驱动区。RMSNorm 把 Q、K 各自 norm 固定到单位 RMS，相当于给 attention logit 加了个 implicit temperature clamp。

### 7. 去掉 QKV 的 bias —— 一致性考虑

ViT-5 用 bias-free RMSNorm（pre-norm 和 QK-Norm 都没 bias）。为了保持一致性，QKV projection 也去掉 bias。涨 0.06%，微小但一致。

**人话**：self-attention 本质是 weighted projection，additive bias 在这里作用不大。去掉让整个 attention 路径的语义更一致——全靠 scale 和 projection，不靠 shift。

---

## 实验结果

### ImageNet 分类（Table 5）

| Model | Params | FLOPs | Acc@224 | Acc@384 |
|---|---|---|---|---|
| DeiT-III-S | 22M | 4.6G | 81.4 | - |
| ViT-5-S | 22M | 4.7G | 82.2 | - |
| DeiT-III-B | 87M | 17.6G | 83.8 | 85.0 |
| ViT-5-B | 87M | 17.9G | 84.2 | 85.4 |
| DeiT-III-L | 304M | 61.6G | 84.5 | 85.4 |
| ViT-5-L | 304M | 62.8G | 84.9 | **86.0** |

Gap 随 model size 单调扩大——这是 scaling-friendly 的标志。

### 图像生成（Table 7）

作为 SiT ([Ma et al. 2024](https://arxiv.org/abs/2401.14005)) 的 backbone，ImageNet-256，7M steps：

| Backbone | FID ↓ | IS ↑ |
|---|---|---|
| SiT w. ViT-XL | 2.06 | 277.50 |
| SiT w. ViT-5-XL | **1.84** | 282.73 |

Figure 6 的 scaling curve 在 S/B/L/XL 四个 size 上 ViT-5 都在 ViT 之上，且曲线平滑。

### 语义分割 ADE20K（Table 8）

| Backbone | mIoU |
|---|---|
| DeiT-III-L | 49.3 |
| ViT-5-L | **52.0** |

Gap +2.7，比 ImageNet 上的 +0.6 大得多。Dense prediction 对 spatial reasoning 要求高，RoPE + register 的组合在这里收益最显著。

---

## Ablation 里的两个关键 takeaway

### Table 9：跟现有设计比

把 DeiT-III / DINO v2/v3 / VisionLLaMA / LLaMA / Qwen / Gemma3 / GPT-oss 的配置分别套到 ViT-L @ 384：

- 所有 vision-derived 设计：85.39–85.69
- 所有 LLM-derived 设计：85.64–85.75
- **ViT-5：86.00**

LLM 的 best practice 直接搬到 vision 不 work，需要 vision-specific 的组合选择。

### Table 10：单组件 ablation

平均性能 drop 排序：
- GeLU → SwiGLU: **-0.42**（最大，over-gating）
- remove LayerScale: -0.29
- remove 2D RoPE: -0.24
- remove Registers: -0.17
- RMSNorm → LayerNorm: -0.15
- remove QK-Norm: -0.12
- keep QKV-bias: -0.06

影响还是 **scale-dependent**：SwiGLU 在 Small 上掉得最猛；LayerScale 和 2D RoPE 在 Large 上影响最大。这暗示一个 LLM-style scaling law——不同组件的边际收益随 scale 增长，未来在更大模型上 QK-Norm、register 的相对重要性还会提升。

---

## 我觉得你应该 take 走的几点

1. **Over-gating 是 cross-modal 的非平凡现象**。nanoGPT 里你默认用 SwiGLU 因为没有 LayerScale；ViT 端 LayerScale 已经在，SwiGLU 就多余了。Unified transformer 设计要专门 handle 这种"重复 gating"的叠加效应。

2. **APE + RoPE 联用**呼应了 LLM 端 [YaRN / NTK-aware](https://arxiv.org/abs/2309.00071) 的部分思想：absolute 和 relative position 不应互斥。Figure 2 的 patch-flip invariance 是个很直观的 demo。

3. **Register 的高频 RoPE 设计**是全文最 elegant 的细节。RoPE 在 vision 上的 2D extension 不只是简单拼接——不同 role 的 token 需要不同 frequency band。这跟 [LongRoPE](https://arxiv.org/abs/2402.13753) 的进化 search 有相同的频域直觉。

4. **Drop-in 兼容性**是 ViT-5 的实用价值。没改 patchification、没改 token mixing、没引入 hierarchical 结构，所以能直接替换 SiT / DiT / UperNet 的 backbone。SigLIP-2、Qwen3-VL 这些 2025 年的多模态模型，把 vision encoder 换成 ViT-5 应该是 free lunch。

5. **Plain ViT 还有 headroom**——这是最重要的 meta 信号。Vision 社区这几年忙着搞预训练方法和 hierarchical 架构，plain ViT 的 component-wise 现代化一直没人系统做。这篇 paper 证明这个方向有肉吃。

---

## 参考链接

- 论文代码：[github.com/wangf3014/ViT-5](https://github.com/wangf3014/ViT-5)
- ViT 原始：[Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929)
- DeiT-III：[Touvron et al. 2022](https://arxiv.org/abs/2204.07118)
- CaiT (LayerScale)：[Touvron et al. 2021](https://arxiv.org/abs/2103.17239)
- RoPE：[Su et al. 2024](https://arxiv.org/abs/2104.09864)
- SwiGLU：[Shazeer 2020](https://arxiv.org/abs/2002.05202)
- RMSNorm：[Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)
- Registers：[Darcet et al. 2024](https://arxiv.org/abs/2309.16588)
- DiT：[Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)
- SiT：[Ma et al. 2024](https://arxiv.org/abs/2401.14005)
- LLaMA：[Touvron et al. 2023](https://arxiv.org/abs/2302.13971)
- Qwen3：[Yang et al. 2025](https://arxiv.org/abs/2505.09388)
- Gemma3：[Team et al. 2025](https://arxiv.org/abs/2503.19786)
- GPT-oss：[Agarwal et al. 2025](https://arxiv.org/abs/2508.10925)
- SigLIP-2：[Tschannen et al. 2025](https://arxiv.org/abs/2502.14786)
- DINOv3：[Siméoni et al. 2025](https://arxiv.org/abs/2508.10104)
- YaRN：[Peng et al. 2023](https://arxiv.org/abs/2309.00071)
- LongRoPE：[Xiong et al. 2024](https://arxiv.org/abs/2402.13753)

---

# ViT-5 深度讲解：把 LLM 的五年架构进化搬到 Vision 上

Andrej，这篇 paper 读起来非常对你口味的——它做的事情恰好是 nanoGPT 社区已经习以为常、但 vision 社区一直没系统做的：**把 LLM 过去五年的 component-wise 进化逐个拎出来、在 plain ViT 上做严格 ablation，找出哪些真的有用、哪些在 vision 上反而有害**。下面我按"动机 → 七个组件 → 关键实验 → 直觉"的结构详细拆。

---

## 1. 论文的核心问题

ViT 自 2020 年底 ([Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929)) 提出后，plain ViT 的 macro-architecture 几乎没动过。DeiT-III ([Touvron et al. 2022](https://arxiv.org/abs/2204.07118)) 只加了 LayerScale；SigLIP-2、Qwen3-VL 也基本沿用 vanilla ViT。同期 LLM 端 ([Touvron et al. 2023 LLaMA](https://arxiv.org/abs/2302.13971), [Qwen3 2025](https://arxiv.org/abs/2505.09388), [Gemma3 2025](https://arxiv.org/abs/2503.19786)) 在 norm、激活、位置编码、QK 处理上做了大量细化。

作者的问题：**plain ViT 是否仍 under-optimized？把 LLM 的这些 refinement 系统化搬过来，能拿到多少 headroom？**

结论：能拿到相当多。ViT-5-B 在 ImageNet-1k 上 84.2% vs DeiT-III-B 83.8%；作为 SiT ([Ma et al. 2024](https://arxiv.org/abs/2401.14005)) 的 backbone 在 ImageNet-256 上 FID 1.84 vs 2.06；ADE20K 分割 ViT-5-L 52.0 mIoU vs 49.3。

---

## 2. 七个组件逐一拆解

ViT-5 的设计公式上等价于：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l) \odot \lambda$$

保留 Attention–FFN 主体，只改 7 个组件：LayerScale、RMSNorm、Gated MLP（最终被否决）、Positional Encoding、Register Tokens、QK-Norm、QKV bias。

### 2.1 LayerScale ↔ Post-RMSNorm 的功能等价

公式 (1) LayerScale：
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l) \odot \lambda, \quad \lambda \in \mathbb{R}^d$$

这里 $\lambda$ 是 per-channel 可学习缩放向量，下标表示维度 $d$；初始化 $10^{-4}$，由 [CaiT](https://arxiv.org/abs/2103.17239) 引入。

公式 (2)–(3) Post-RMSNorm 的重写：
$$\mathbf{x}_{l+1} = (\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)) \odot \lambda_p / \text{Norm}$$
$$\text{Norm} = \text{RMS}(\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l))$$

$\lambda_p$ 同样是 per-channel 缩放，$\text{RMS}(\cdot)$ 是 root-mean-square。

**关键观察**：LayerScale 只缩放 block 输出 $\mathcal{F}$；post-norm 把 residual $\mathbf{x}_l$ 和 $\mathcal{F}$ 一起做 RMS 再缩放。两者效果几乎一样（Table 1：S/B/L 上 82.18 vs 82.16、84.15 vs 84.16、84.82 vs 84.86）。

**直觉建立**：在 deep residual stream 里，浅层 $\|\mathbf{x}_l\| \gg \|\mathcal{F}(\mathbf{x}_l)\|$，所以 RMS 主要由 $\mathbf{x}_l$ 主导，post-norm 对 $\mathcal{F}$ 的有效缩放因子近似就是 $1/\text{RMS}(\mathbf{x}_l)$——这跟 LayerScale 用一个学到的常数 $\lambda$ 控制 $\mathcal{F}$ 的注入幅度，本质都是限制"new signal 对 residual stream 的相对能量"。这是 LLM 端 post-norm ([GPT-oss Agarwal et al. 2025](https://arxiv.org/abs/2508.10925)) 与 ViT 端 LayerScale 的隐式统一，但 LayerScale 更灵活、计算更省，所以 ViT-5 选了 LayerScale。

### 2.2 RMSNorm 替代 LayerNorm

LLM 端自 [LLaMA](https://arxiv.org/abs/2302.13971)、[PaLM](https://arxiv.org/abs/2204.02311)、[Gopher](https://arxiv.org/abs/2112.11446) 起几乎都切到 RMSNorm ([Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467))。直觉：LayerNorm 里的 re-centering 几乎不携带信息，去掉 mean-subtraction 后只剩 re-scaling 不变性，反而减少 shifting noise。ViT 上同样有效，ViT-B 涨 0.2% top-1。

### 2.3 Gated MLP / SwiGLU —— **被否决的关键组件**

这是全文最有意思的发现。Table 2 的 2×2 网格：

| LayerScale | SwiGLU | IN-1k acc | IN-256 FID |
|---|---|---|---|
| ✗ | ✗ | 83.86 | 15.80 |
| ✗ | ✓ | 83.94 | 15.48 |
| ✓ | ✓ | 83.70 | 16.22 ← **变差** |
| ✓ | ✗ | 84.16 | 14.57 ← **最好** |

**Over-gating 现象**：LayerScale 做 per-channel 静态 gating，SwiGLU ([Shazeer 2020](https://arxiv.org/abs/2002.05202)) 做 per-token 动态 gating；两者都在 channel 维度做 filtering，叠加使 intermediate activations 过度稀疏。

**直觉建立**：让我类比 nanoGPT 里 SwiGLU 的实现——`FFN(x) = (Swish(xW1) ⊙ xW3)W2`。Swish 那一支已经在做稀疏化（负值压到 0 附近），如果外面再乘一个被 LayerScale 压到 $10^{-4}$ 的向量，等于把信号在两个 bottleneck 处同时收紧，channel utilization 急剧下降。在 ViT-XL (449M, hidden=1152) 规模内这个效应明显；更大模型可能缓解，但作者明确留作 future work。

所以 ViT-5 回退到 GeLU-MLP。这是个反直觉但很干净的结论：**LLM 的现代默认配方 (RMSNorm + RoPE + SwiGLU) 不能整包搬到 ViT**。

### 2.4 Positional Encoding: APE + 2D RoPE 联用

ViT 原版用 learnable APE。问题：APE 缺乏显式 relative positional modeling，对 dynamic resolution 鲁棒性差。作者引入 2D RoPE ([Su et al. 2024 RoPE 原文](https://arxiv.org/abs/2104.09864))，但**保留 APE**。

**为什么不丢 APE？** Figure 2 的 demo：用 RoPE-only 时，对图像做 patch-level flip（局部块顺序打乱），模型看到的 token 序列在 RoPE 视角下完全等价——因为 RoPE 只编码 pair-wise 相对位置，全局翻转后所有相对位置保持不变。这在分类上无伤，但通用 vision backbone（分割、检测、生成）需要绝对空间线索。

**实验证据**：Figure 3 的 resolution sweep。DeiT-III（仅 APE）在 224 训练、>256 测试时性能急剧下滑；ViT-5 在 128→512 范围内性能稳定甚至单调上升。RoPE 的 $\theta_i = \text{base}^{-2i/d}$ 频率构造对不同 sequence length 是 resolution-agnostic 的——只要相对距离不变，attention pattern 不变。

### 2.5 Register Tokens 需要独立的高频 RoPE

这是 paper 最 subtle 的设计，我觉得是最佳 intuition 点。

[Registers (Darcet et al. 2024)](https://arxiv.org/abs/2309.16588) 的原始做法：append $N_r$ 个 learnable tokens 到 patch tokens 后面，让它们吸收全局信息 artifact。Table 3 的关键数据：

| Configuration | S | B | L |
|---|---|---|---|
| no register | 82.04 | 84.02 | 84.61 |
| vanilla registers (不旋转) | 81.95 | 83.90 | 84.37 ← **比无 register 还差** |
| RoPE on registers, same freq base | 82.05 | 84.00 | 84.59 |
| RoPE on registers, **high freq base** | 82.16 | 84.16 | 84.86 |

**为什么不旋转的 register 会让性能下降？**

直觉：RoPE 把 $q_m, k_n$ 在 channel 维度上做旋转 $\text{rotate}(q_m, m\theta_i)$，旋转角度正比于绝对位置 $m$。当 patch token 位置 $m$ 大、register 位置视为 0 时，register 与 patch 的内积

$$\langle R_0(q), R_n(k) \rangle = \text{Re}[\sum_i q_i k_i^* e^{in\theta_i}]$$

随 $n$ 振荡——位置越远的 patch 与 register 的相似度被相位打散得越厉害。这意味着 register 实际上"被推到序列尾部 patch 的关注盲区"，attention 分布被 implicit positional bias 扭曲。

**解决方案**：给 register 单独一套 2D RoPE，但 frequency base 显著高于 patch 的 RoPE base。直觉是让 register 和 patch 占据不同的"频率带"——register 在高频快速旋转、patch 在低频慢旋转，两者的 channel 维度上几乎正交，positional correlation 被 decouple。

这跟 [Multi-scale RoPE / NTK-aware](https://arxiv.org/abs/2309.00071) 的思路有共鸣：不同 base 的 RoPE 等价于不同的"位置频率分辨率"，分配给不同角色 token 是一种 cheap positional disentangle。

### 2.6 QK-Norm

公式 (4)–(5)：
$$Q' = \text{RMSNorm}(Q), \quad K' = \text{RMSNorm}(K)$$
$$\text{Attn}(Q,K,V) = \text{Softmax}(Q'K'^T/\sqrt{d})V$$

QK-Norm 的收益在性能上 modest（Table 10: B 上升约 0.1%），但**训练稳定性显著提升**。Figure 5 显示不加 QK-Norm 时 loss 有尖锐 spike；加了之后平滑收敛。这是 [Qwen3](https://arxiv.org/abs/2505.09388) 和 [Gemma3](https://arxiv.org/abs/2503.19786) 都观察到的现象。

**直觉**：Q、K 是高维线性投影的输出，norm 容易随训练 drift，导致 $QK^T$ 的 scale 不稳定，softmax 时进入饱和或欠驱动区。RMSNorm 把 Q、K 各自 norm 固定到单位 RMS，相当于把 attention logit 的有效 scale 显式 clamp，类似一种 implicit temperature scheduling。在 LLM 端，这跟 [Gemma3 报告](https://arxiv.org/abs/2503.19786) 的 attention logit spike 问题同源。

### 2.7 QKV bias 移除

ViT-5 用 bias-free RMSNorm（包括 pre-norm 和 QK-Norm）。直觉：self-attention 本质是 weighted projection 而非 additive bias；移除 QKV 的 bias 项让 RMSNorm 的"纯 scale"语义保持一致。性能上 +0.06%（Table 10），微小但一致。

---

## 3. Architecture 一览

Table 4 的配置，全 alignment with vanilla ViT：

| Model | #layers | dim | #heads | #registers | #params |
|---|---|---|---|---|---|
| ViT-5-S | 12 | 384 | 6 | 4 | 22M |
| ViT-5-B | 12 | 768 | 12 | 4 | 87M |
| ViT-5-L | 24 | 1024 | 16 | 4 | 304M |
| ViT-5-XL | 28 | 1152 | 16 | 4 | 449M |

register 数固定为 4，比 [Darcet et al. 2024](https://arxiv.org/abs/2309.16588) 原文的 4 个一致。Table 11(c) 显示从 4→16→64 性能变化很小（84.16→84.12→84.12），说明 register 容量需求很低。

设计原则（Appendix A.2）三条很关键：
1. **不做 spatial downsampling**——保持通用性，多模态系统容易集成。
2. **只用 self-attention**，不引入 convolutional inductive bias（放弃 [CoAtNet](https://arxiv.org/abs/2106.13448) / [MaxViT](https://arxiv.org/abs/2204.01697) 类的潜在收益）。
3. **不改 patchification**——因为现代生成模型 ([DiT](https://arxiv.org/abs/2212.09748), [VAR](https://arxiv.org/abs/2404.02905)) 用 VAE tokenizer，patch embedding 优化只对 pixel-space 任务有用，不通用。

---

## 4. 三大实验

### 4.1 ImageNet-1k 分类（Table 5）

最 striking 的几个点：
- ViT-5-S (22M) 82.2% vs DeiT-III-S 81.4%（+0.8）；
- ViT-5-B (87M) 84.2% vs DeiT-III-B 83.8%；
- ViT-5-L @ 384² = **86.0%**，超过 ConvNeXt-L @ 384² 的 85.5% 和 DeiT-III-L 的 85.4%。

性能 gap 随 model size 单调扩大——这是 scaling-friendly architecture 的标志，不是某种 trick 在小模型上的过拟合。

### 4.2 图像生成（Table 6, 7, Figure 6）

作为 SiT backbone 直接 drop-in 替换：

| Backbone | FID ↓ | IS ↑ | Prec ↑ | Recall ↑ |
|---|---|---|---|---|
| SiT w. ViT-XL | 2.06 | 277.50 | 0.83 | 0.59 |
| SiT w. ViT-5-XL | **1.84** | 282.73 | 0.83 | 0.60 |

7M steps 长程训练，所有指标都好。Figure 6 的 scaling curve 在 S/B/L/XL 四个 size 上 ViT-5 都在 ViT 之上，且曲线平滑——这跟 [Chinchilla](https://arxiv.org/abs/2203.15556) 风格的 scaling law 友好性吻合。

直觉：diffusion backbone 对 spatial coherence 极敏感。Figure 4 的 attention map 对比显示 ViT-5 的 class-token attention 比 DeiT-III 干净得多、聚焦在语义区域。生成任务里，这种 spatial precision 直接 translate 到 FID 改进。

### 4.3 语义分割 ADE20K（Table 8）

ViT-5-L 52.0 mIoU vs DeiT-III-L 49.3 mIoU，gap **+2.7**——比 ImageNet 上的 gap (+0.6) 大得多。dense prediction 对 spatial reasoning 要求更高，RoPE + register 的组合在这里收益最显著。

---

## 5. Ablation 的两个关键观察

### 5.1 与既有设计对比（Table 9）

把 DeiT-III / DINO v2/v3 / VisionLLaMA / LLaMA / Qwen / Gemma3 / GPT-oss 的配置分别套到 ViT-L @ 384：

- 所有 vision-derived 设计都低于 ViT-5（85.39–85.69 vs 86.00）；
- 所有 LLM-derived 设计也低于 ViT-5（85.64–85.75 vs 86.00）。

**核心信息**：LLM 的 best practice（SwiGLU + RMSNorm + RoPE）直接搬到 vision 不 work。Vision 需要 vision-specific 的组合选择（去 SwiGLU、APE+RoPE、register with high-freq RoPE）。

### 5.2 单组件 ablation（Table 10）

平均性能 drop 排序：
- GeLU → SwiGLU: -0.42（最大，over-gating 引起）
- remove LayerScale: -0.29
- remove 2D RoPE: -0.24
- remove Registers: -0.17
- RMSNorm → LayerNorm: -0.15
- remove QK-Norm: -0.12
- keep QKV-bias: -0.06

但单组件的影响是 **scale-dependent**：
- SwiGLU 在 Small 上掉得最猛（compact model 对 over-gating 敏感）；
- LayerScale 和 2D RoPE 在 Large 上影响最大（深层网络对 stability 和 relative position 依赖更强）；
- Registers 和 QK-Norm 在 Large 上的 drop 也显著大于 Small。

这暗示一个 LLM-style scaling law：**不同组件的边际收益随 scale 增长**，未来在 ViT-5-Huge 或更大规模上，QK-Norm、register 的相对重要性还会提升。

---

## 6. 我（Karpathy 视角）会强调的几个 takeaways

1. **Over-gating 是 LLM 端也有但 vision 端更脆弱的问题**。Llama-3 训练里大家注意到的 attention sink / activation sparsity 在 vision 上被 LayerScale + SwiGLU 加倍放大。这是一个 cross-modal 的非平凡现象，值得在 unified transformer 设计里专门 handle。

2. **APE + RoPE 的联合使用其实呼应了 LLM 端 [YaRN / NTK-aware](https://arxiv.org/abs/2309.00071) 的部分思想**：absolute position 和 relative position 不应互斥。ViT 历史上有 [APE vs RPE 的争论](https://arxiv.org/abs/2106.09687)，这篇给出"全都要"的实证理由（Figure 2 的 patch-flip invariance）。

3. **Register 的高频 RoPE 设计是全文最 elegant 的细节**。它说明 RoPE 在 vision 上的 2D extension 不是简单地把 1D RoPE 拼两遍——不同 role 的 token 需要不同 frequency band。这跟 [Su et al. 原文](https://arxiv.org/abs/2104.09864) 的 long-base RoPE、以及 [LongRoPE](https://arxiv.org/abs/2402.13753) 的进化 search 有相同的频域直觉。

4. **Drop-in 兼容性是 ViT-5 的实用价值**。它没有改 patchification、没改 token mixing、没引入 hierarchical 结构，所以能直接替换 SiT / DiT / UperNet 的 backbone，这个工程友好性是 DeiT / DINO 之外很多 vision variant 做不到的。

5. **Plain ViT 还有 headroom**——这是最重要的 meta 信号。后续 vision-language model（[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[SigLIP-2](https://arxiv.org/abs/2502.14786)、[GPT-4o](https://arxiv.org/abs/2410.21276)）都还在用 vanilla ViT encoder，把 ViT-5 的七项升级应用过去应该是 free lunch。

---

## 7. 参考

- 论文代码：[github.com/wangf3014/ViT-5](https://github.com/wangf3014/ViT-5)
- ViT 原始：[Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929)
- DeiT-III：[Touvron et al. 2022](https://arxiv.org/abs/2204.07118)
- CaiT (LayerScale)：[Touvron et al. 2021](https://arxiv.org/abs/2103.17239)
- RoPE：[Su et al. 2024](https://arxiv.org/abs/2104.09864)
- SwiGLU：[Shazeer 2020](https://arxiv.org/abs/2002.05202)
- RMSNorm：[Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)
- Registers：[Darcet et al. 2024](https://arxiv.org/abs/2309.16588)
- DiT：[Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)
- SiT：[Ma et al. 2024](https://arxiv.org/abs/2401.14005)
- LLaMA：[Touvron et al. 2023](https://arxiv.org/abs/2302.13971)
- Qwen3：[Yang et al. 2025](https://arxiv.org/abs/2505.09388)
- Gemma3：[Team et al. 2025](https://arxiv.org/abs/2503.19786)
- DINOv3：[Siméoni et al. 2025](https://arxiv.org/abs/2508.10104)
- GPT-oss：[Agarwal et al. 2025](https://arxiv.org/abs/2508.10925)
- SigLIP-2：[Tschannen et al. 2025](https://arxiv.org/abs/2502.14786)
- YaRN (RoPE 长上下文)：[Peng et al. 2023](https://arxiv.org/abs/2309.00071)
- LongRoPE：[Xiong et al. 2024](https://arxiv.org/abs/2402.13753)
- VisionLLaMA：[Chu et al. 2024](https://arxiv.org/abs/2407.13359)

---

简短总结一句话：**ViT-5 = vanilla ViT skeleton + 七项 LLM 端成熟组件 + vision-specific 的"减法"（去掉 SwiGLU、给 register 单独高频 RoPE）**。它最大的贡献不是单点 SOTA，而是用严格的 ablation 把"哪些 modernization 对 vision 真正有效"这件事讲清楚了。对我们这些同时在 nanoGPT 和 nano-ViT 之间游走的人，这篇 paper 的工程价值在于：**它给了 plain ViT 一个跟 LLM-2025 默认配方对齐的、可直接 plug-in 的现代化模板**。
