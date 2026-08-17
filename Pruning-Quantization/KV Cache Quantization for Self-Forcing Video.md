---
source_pdf: KV Cache Quantization for Self-Forcing Video.pdf
paper_sha256: 23fb81ca406e3cd211dd6259422159856083f55336f45a6d818b419929b24a2e
processed_at: '2026-08-05T11:30:40-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

好，我尽量把学术腔去掉，用最直白的方式把这个故事讲清楚，但技术细节该讲的还是讲——因为你 build intuition 需要看到实际的数字和公式结构。

---

## 一句话总结这 paper 在干嘛

**你在用 self-forcing 生成视频的时候，越往后生成，KV cache 越来越大，显存扛不住。那能不能把 KV cache 压缩一下？这篇 paper 试了 33 种压缩方法，发现大部分方法理论上压缩了但实际上显存根本没降，真正能用的只有一种——FlowCache 的 soft-prune。**

就这么个事。但细节里有很多魔鬼。

---

## 先搞清楚问题从哪来

### Self-Forcing 是什么

你自己 2025 年那个 Self-Forcing paper（[Huang et al. 2025](https://arxiv.org/abs/2506.08009)）提出来的。核心 idea 很简单：

传统 autoregressive video diffusion 训练时用 ground-truth frames 当 context（teacher forcing），但 inference 时你得用自己生成的 frames 当 context。两边不匹配，error 沿时间维度 compound，视频越往后越崩。

Self-Forcing 的解法：**训练时就让 model 见到自己生成的 context 和 KV-cache rollout**。让 model 从训练阶段就学会 "在 noisy self-generated state 下也能稳定 rollout"。

这带来一个 side effect：**KV cache 从一个 inference-time 的 systems optimization 变成了 training loop 的一等公民**。Cache 的 representation 直接影响 model 学到的 dynamics。你在 inference 时随便改 cache format，model 行为就变了。

### KV cache 为什么在 video 上爆炸

Video generation 和 LLM 不一样。LLM 一个 token 就是一个 token，video 一帧经过 spatial-temporal tokenization 后是 **成百上千个 tokens**。

生成 165 frames @ 480×832 @ 16fps（约 10 秒视频），KV cache 的 BF16 大小：

$$
M_{\text{KV}} = 2 \times N_{\text{tokens}} \times d_{\text{model}} \times L_{\text{layers}} \times 2 \text{ bytes}
$$

这里：
- $2$：Key 和 Value 两个 tensor
- $N_{\text{tokens}}$：sequence length，随帧数线性涨
- $d_{\text{model}}$：hidden dimension
- $L_{\text{layers}}$：transformer 层数
- $2 \text{ bytes}$：BF16 每元素 2 字节

Paper 实测 BF16 baseline 在 10 秒视频上 peak VRAM **19.28 GB**。你要生成 30 秒、60 秒？线性涨上去，单卡根本扛不住。

所以问题很 concrete：**能不能压 KV cache，让长视频生成在单卡上跑得起来？**

---

## 33 种方法到底在搞什么

Paper 测了 33 种 method variants，我帮你归归类，用人话说每个的核心 idea。

### 第一类：无脑量化（RTN）

最傻的方法——Round-to-Nearest symmetric quantization。把 BF16 的 KV cache 直接量化成 INT4 或 INT2。

公式：

$$
s = \frac{\max(|x_i|)}{2^{b-1} - 1}, \quad x_q = \text{round}(x / s)
$$

- $s$：per-block scale factor
- $b$：bit width（INT4 → $b=4$，分母 $2^3 - 1 = 7$）
- $x_q$：量化后的 integer code

**人话**：找到这个 block 里绝对值最大的数，用它当 scale，然后把所有数按比例缩到 INT4 能表示的范围 $[-7, +7]$ 里。

**为什么效果差**：KV cache 里有 outlier（少数特别大的值），这些 outlier 会把 scale 撑得很大，导致其他正常值的量化粒度极粗。好比一个班里有一个人考了 100 分，其他人都考 30-40 分，你用 100 当 scale，那 30 和 40 的区别在量化后可能都是 0——精度全丢了。

实测 RTN INT4 的 SSIM 只有 0.688（vs BF16 的 1.0），而且 peak VRAM **19.98 GB，跟 BF16 几乎一样**。为什么？后面讲。

### 第二类：统计感知量化（KIVI）

KIVI（[Liu et al. 2024](https://arxiv.org/abs/2402.02750)）的观察：**Key 和 Value 的 outlier 分布模式不同**。

- Key 的 outlier 集中在特定 **channel**（某些维度特别大）→ 用 per-channel quantization
- Value 的 outlier 集中在特定 **token**（某些位置特别大）→ 用 per-token quantization

而且用 asymmetric quantization（带 zero-point offset）：

$$
s = \frac{\max(x) - \min(x)}{2^b - 1}, \quad z = \text{round}\left(-\frac{\min(x)}{s}\right)
$$

- $s$：scale
- $z$：zero-point，让 0 能精确映射到一个 integer code
- 重建：$\hat{x} = (x_q - z) \cdot s$

**人话**：Key 沿 channel 方向分组量化，Value 沿 token 方向分组量化，各自找自己的 scale 和 zero-point，这样 outlier 不会互相干扰。

KIVI 在 LLM 上效果极好（2-bit 就能跑），但在这篇 paper 的 video 场景下 **崩了**——INT2 时 SSIM 只有 0.241。推测 video diffusion model 的 KV 分布和 LLM 差异太大，channel-wise outlier 模式不成立。这是一个好的 reminder：**LLM 上的经验不能 blind transfer 到 video diffusion**。

### 第三类：旋转后量化

QuaRot（[Ashkboos et al. 2024](https://arxiv.org/abs/2404.00456)）的 idea 特别优雅：**与其在原空间跟 outlier 搏斗，不如先把 tensor 旋转到一个没有 outlier 的 basis 里，再量化**。

$$
\mathbf{x}_{\text{rot}} = \mathbf{H} \mathbf{x}, \quad \text{quantize}(\mathbf{x}_{\text{rot}}), \quad \text{read 时 inverse rotate: } \hat{\mathbf{x}} = \mathbf{H}^T \hat{\mathbf{x}}_{\text{rot}}
$$

- $\mathbf{H}$：Hadamard matrix，正交矩阵（$\mathbf{H}\mathbf{H}^T = \mathbf{I}$），元素都是 $\pm 1/\sqrt{d}$，乘法极快
- $\mathbf{x}_{\text{rot}}$：旋转后的 tensor，outlier 被 "摊平" 到所有维度

**人话**：Hadamard rotation 就像一个 "搅拌机"——把少数 channel 上的极端值打散到所有 channel，让整体分布变得均匀，对量化特别友好。

**为什么 fidelity 高但 deployment 差**：每次 attention read 都要做一次 inverse rotation $\mathbf{H}^T$，这是 $O(d^2)$ 的矩阵乘。Paper 实测 QuaRot KV INT4 runtime **236.6 秒**（BF16 是 58.6 秒），慢了 4 倍。而且 peak VRAM 19.98 GB，跟 BF16 一样——因为 inverse rotation 时会 materialize 一个 dense BF16 tensor。

**这就是这篇 paper 最重要的 lesson 之一：fidelity 高的方法不一定是 deployment 好的方法。**

### 第四类：残差量化（PRQ）

两阶段量化，类似 audio coding 里的 multi-stage VQ：

$$
\text{Stage 1: } x_{q1} = Q_1(x), \quad r = x - \hat{x}_{q1}
$$
$$
\text{Stage 2: } x_{q2} = Q_2(r), \quad \hat{x} = \hat{x}_{q1} + \hat{x}_{q2}
$$

- $r$：第一阶段量化后的 residual（误差）
- $Q_1, Q_2$：两个独立 quantizer，64-token block
- $\hat{x}$：最终重建 = 第一阶段重建 + 第二阶段重建

**人话**：第一遍量化抓大结构，第二遍专门修补误差。用相同的总 bits，比单阶段量化精度高很多。

实测 PRQ INT4 SSIM **0.824**（所有 compressed 方法里最高），LPIPS **0.082**（最低）。但是——compression ratio 只有 **1.60×**（因为存了两套 codes + scales），peak VRAM **20.69 GB（比 BF16 还高）**，runtime **160 秒**。

**典型的 "research winner, deployment loser"**。算法很漂亮，实际跑不起来。

### 第五类：时间启发式（Age-Tier, TPTQ）

这类方法用一个 prior：**最近的 token 比老的 token 重要**。

Age-Tier：
```
cache = [recent slice: INT4] + [old slice: INT2]
```

TPTQ 更复杂：recent slice 高精度，old slice 用 PRQ + outlier preservation。

**人话**：近的帧用高精度存，远的帧用低精度存。听起来很合理——video 有物理连续性，最近的帧对生成下一帧最重要。

但实测 Age-Tier INT4 SSIM 0.688，跟 RTN INT4 一样。说明这个 prior 在 video diffusion 上收益不如预期。可能因为 attention pattern 比 "近的更重要" 更复杂——远的 frame 也提供 scene identity、object layout 等关键 context。

### 第六类：FlowCache-inspired（**deployment winner**）

这是 paper 的主角。灵感来自 FlowCache（[Ma et al. 2026](https://arxiv.org/abs/2602.10825)），核心思路：**不要无脑量化所有 cache，而是按 frame-aligned chunk 分块，对每个 chunk 做 retain/compress/summarize/prune 的决策**。

#### Hard Prune
保留 recent chunks + important old chunks，其余 **直接丢弃，重建时填零**：

$$
\hat{\mathbf{K}}_{\text{pruned}} = \mathbf{0}
$$

#### Soft-Prune（**最强 deployment point**）
不丢弃，而是为每个 pruned chunk 存 **一个 pooled summary token**，重建时把这个 summary repeat 到整个 evicted span：

$$
\mathbf{s}_i = \text{pool}(\mathbf{K}_i), \quad \hat{\mathbf{K}}_{\text{span } i} = \text{repeat}(\mathbf{s}_i, \text{span\_len})
$$

- $\mathbf{s}_i \in \mathbb{R}^d$：第 $i$ 个 pruned chunk 的 summary（一个 token 的 hidden dim 大小）
- $\text{pool}$：mean pooling over tokens in chunk
- $\text{repeat}$：把 summary 沿 token 维度复制到原 chunk 的 token 数

**人话**：Hard prune 是 "删了就删了，attention 到那块地方全是零，model 完全瞎了"。Soft-prune 是 "删了但留一个平均值当 placeholder，attention 到那块至少有个大概的 context"。

这个区别极其关键。实测：

| Method | VBench Img. | SSIM | Peak VRAM |
|--------|-------------|------|-----------|
| FlowCache Prune INT4 (hard) | 0.727 | 0.457 | 11.71 GB |
| FlowCache Soft-Prune INT4 | **0.739** | 0.544 | 11.71 GB |

两者 VRAM 一样，但 soft-prune 的 VBench 完全追平 BF16（0.739），hard prune 差一些。**一个 summary token 的区别，VBench 差了 0.012**——在 video quality 评估里这已经是有意义的差距。

Importance scoring 的公式（FlowCache Adaptive）：

$$
\text{score}(\text{chunk}_i) = \alpha \cdot \text{recency}(i) + \beta \cdot \|\Delta \mathbf{K}_i\|_1
$$

- $\text{recency}(i)$：chunk $i$ 离当前生成帧的距离（越近越高）
- $\|\Delta \mathbf{K}_i\|_1$：relative-L1 delta，这个 chunk 在相邻 step 的变化幅度（变化大 = 活跃 = 重要）
- $\alpha, \beta$：权重

**人话**：一个 chunk 重不重要，看两件事——它离当前帧多近（近的重要），它最近变不变（变的重要）。两个 factor 加权打分，高分的保留高精度，低分的 prune。

---

## 最炸裂的发现：压缩了但显存没降

这是 paper 最 systems-level 的重要发现，我详细讲。

看这组数据：

| Method | Comp. Ratio | Peak VRAM | Compressed KV @ peak |
|--------|------------|-----------|----------------------|
| BF16 | 1.00× | 19.28 GB | — |
| RTN INT4 | 3.20× | 19.98 GB | — |
| RTN INT4 Recent2 | 2.43× | **21.37 GB** | 4.62 GB |
| RTN INT4 Refresh | 3.20× | **22.64 GB** | 3.51 GB |

**RTN INT4 Recent2 和 Refresh 的 peak VRAM 比 BF16 还高**，尽管它们确实把 KV cache 压缩了（compressed KV 只有 3-4 GB）。

这不矛盾吗？压缩了 3 倍，显存反而涨了？

原因在 **attention compute path 的内存生命周期**：

1. **RTN Recent2**：保留最近 2 个 frame-blocks 的 BF16 KV，旧 prefix 是量化的。但 attention read 时需要 **把整个旧 prefix dequantize 回 dense BF16**，然后和 recent tail 拼接。这一瞬间同时存在：
   - 量化 cache（3-4 GB）
   - dequantized BF16 prefix（~11 GB）
   - BF16 recent tail
   - 中间拼接 buffer
   
   加起来 21 GB。

2. **RTN Refresh**：denoising 期间保持 BF16 cache，refresh pass 里重新量化。refresh 那一刻同时有 BF16 cache + 正在写入的量化 cache + transform buffers。

3. **QuaRot**：rotated low-bit values 在 attention 前需要 inverse Hadamard transform，产生 dense BF16 tensor。

Paper 的 trace（Figure 6）显示 RTN INT4 Refresh 在 peak 时刻：
- Allocated: 21.41 GB
- BF16-equivalent KV: 11.25 GB
- Compressed KV: 3.51 GB
- **剩下的 ~7 GB 全是 reconstruction buffers**

**人话**：这些方法在 "存" 的时候确实省了内存，但在 "用"（attention compute）的那一瞬间，需要把压缩的 cache 解压回 BF16 才能做 attention。这个解压出来的 dense tensor 把省的内存全吃回去了，甚至还不够。

这是一个 deep systems lesson：**KV cache compression 不能只看存储大小，必须看整个 attention compute path 的内存峰值**。很多 paper 只报告 compression ratio，不报告 peak VRAM，这就掩盖了这个问题。

---

## FlowCache paradox：看起来无损但其实变了

这是 paper 第二个重要发现。

看 FlowCache Soft-Prune INT4 的数据：

| Metric | BF16 | FlowCache Soft-Prune INT4 |
|--------|------|---------------------------|
| VBench imaging | 0.739 | **0.739** |
| SSIM | 1.000 | 0.544 |
| LPIPS | 0.000 | 0.297 |

VBench 完全持平 BF16，但 SSIM 掉到 0.544，LPIPS 0.297。

**这意味着什么？**

VBench 测的是 "这个视频看起来像不像真实视频"——perceptual plausibility。SSIM/LPIPS 测的是 "这个视频和 BF16 参考视频是否像素级一致"——structural fidelity。

FlowCache soft-prune 生成的视频 **看起来很 plausible**（VBench 持平），但 **和 BF16 生成的视频是不同的视频**（SSIM 低）。它保留了 "看起来像视频" 的能力，但生成了不同的内容。

为什么？因为 soft-prune 丢弃了 chunk 的 spatial-temporal detail，只保留一个 summary token。Model 用这个 averaged context 生成视频时，会 **hallucinate detail**——它生成的内容 plausible，但和用完整 context 生成的 BF16 视频在像素层面 diverge。

这是 information-theoretic 的必然：你丢了 detail，model 就编 detail。编出来的 detail 可能很 plausible，但它不是 BF16 会生成的那个 detail。

**这就是为什么 paper 强调 dual-axis evaluation**：只看 VBench 会误以为 FlowCache soft-prune "无损"，但实际上它改变了生成内容。要看 SSIM 才能发现这个 difference。

---

## Spatial mixed precision 的灾难性失败

这是 paper 最强的 negative result，值得单独讲。

方法 idea 听起来很合理：
1. 用 temporal variance 构建 foreground/background mask（高 variance = foreground）
2. Foreground tokens → INT4（高精度）
3. Background tokens → INT2（低精度）

**人话**：前景物体（人、车、动物）用高精度存，背景（天空、墙壁）用低精度存。视觉系统本来就更关心前景嘛。

实测结果：

| Method | VBench Img. | SSIM | Drift |
|--------|-------------|------|-------|
| BF16 | 0.739 | 1.000 | 0.739 |
| Spatial Mixed (QuaRot fg, RTN bg) | **0.399** | 0.433 | 0.394 |

VBench 从 0.739 跌到 0.399——**崩溃式下降**。Drift 0.394，说明长 rollout 末段完全崩了。

**为什么这个 "合理的" prior 会失败？**

我的解读：**model 的 attention pattern 和人类视觉的 foreground bias 不一致**。在 autoregressive video generation 中，background tokens 承担的不是 "背景画面" 的作用，而是 **全局上下文 / 场景一致性 / spatial anchor** 的作用。你把 background 粗暴 INT2 化，model 丢失了 scene coherence 的 anchor，整个场景就散架了。

这类似于 LLM 里发现 stop words 也很重要——你不能用人类 prior 替代 model 的实际 dependency structure。Model 学到的 attention pattern 可能和人类直觉完全不同。

**这个 negative result 对未来的 cache compression 设计是个重要 warning：不要假设你知道 model "应该" 关心哪些 token。**

---

## 最终的 deployment recommendation

Paper 的结论很 nuanced，我总结成一句话：

**FlowCache Soft-Prune INT4 是当前 stack 下唯一真正能 deploy 的方法。**

数据说话：

| | BF16 | FlowCache Soft-Prune INT4 | PRQ INT4 |
|--|------|---------------------------|----------|
| Peak VRAM | 19.28 GB | **11.71 GB** | 20.69 GB |
| Runtime | 58.6s | 75.0s | 160.0s |
| VBench | 0.739 | **0.739** | 0.739 |
| Compression | 1.00× | **5.49×** | 1.60× |

- VRAM 降了 39%（19.28 → 11.71 GB），单卡能跑了
- Runtime 只慢了 28%（58.6 → 75.0s），可接受
- VBench 完全持平 BF16
- Compression 5.49×，realized memory relief 确实兑现了

PRQ INT4 虽然 SSIM 最高（0.824），但 VRAM 比 BF16 还高，runtime 慢 2.7 倍——**research winner, deployment loser**。

---

## 这 paper 真正的 contribution

这 paper 不是一个 "新方法" paper。它是一个 **design space map**。价值在于：

1. **Empirical rigor**：33 methods, 610 observations, 6 axes, 2 benchmarks, cross-benchmark correlation $r > 0.93$。这是 long-horizon video KV compression 领域目前最系统的 empirical study。

2. **Systems insight**：揭示了 "nominal compression ≠ peak VRAM relief" 的 anomaly。纯 ML paper 不会 catch 这个，因为它需要 per-step VRAM trace 才能发现。

3. **Quality decomposition**：VBench vs SSIM 的分离（FlowCache paradox）提供了 video generation evaluation 的方法论启示——**perceptual plausibility 和 structural fidelity 是两个不同的 axis**。

4. **Negative result**：Spatial mixed precision 的崩溃警告了 "人类 prior 替代 model dependency" 的陷阱。

5. **Future roadmap**：明确指出 attention path 重设计（in-kernel dequantization）是最关键的 systems follow-up。当前 stack 的核心瓶颈是 attention read 时需要 materialize dense BF16 tensor。如果能做 fused in-attention dequantization（类似 FlashAttention 的 streaming 思路），那 PRQ、QuaRot 这些 high-fidelity 方法可能就能变成 deployment winner。

---

## 我的一些额外联想

### Training-aware quantization

当前所有方法都是 training-free（post-hoc）。但 Self-Forcing 的训练已经包含 KV-cache rollout，理论上可以让 model 在训练时就见到量化 cache。

用 straight-through estimator（STE）：

$$
\hat{\mathbf{x}} = \mathbf{x} + \text{sg}(\text{quantize}(\mathbf{x}) - \mathbf{x})
$$

- $\text{sg}(\cdot)$：stop-gradient，前向走量化，反向梯度直通
- Model 学会 robust to quantization noise

这可能让 INT4 甚至 INT2 的 fidelity 大幅提升。QVG-Pro（[Xi et al. 2026](https://arxiv.org/abs/2602.02958)）可能走的就是这条路，但 paper 没能 reproduce（无公开 code）。

### 和 LLM KV compression 的根本差异

LLM KV compression（KIVI, KVQuant 等）在 2-bit 上就能 preserve quality。为什么 video diffusion 上 INT4 都这么费劲？

我的猜测：
1. **Video KV cache 的 signal-to-noise ratio 更低**——每帧的 KV 信息密度不如 LLM 一个 token 的语义密度
2. **Video diffusion 的 attention pattern 更 dense**——不像 LLM 有明显的 sparse attention 结构，prune 的容错率更低
3. **Self-forcing training 让 model 对 cache representation 更敏感**——model 在训练时学到的是 BF16 cache 的 dynamics，inference 时换 format 破坏了这个 learned dynamics

### Attention kernel 的机会

当前 stack 的核心瓶颈是 attention read 时 dequantize 成 dense BF16。如果能写一个 custom attention kernel，让 dequantization 在 kernel 内部 streaming 进行（类似 FlashAttention 的 tiling），不 materialize 完整 dense tensor，那 peak VRAM 会大幅降低。

Marlin kernel（[GPTQ-Marlin](https://github.com/IST-DASLab/marlin)）在 LLM weight quantization 上做过类似的事——fused dequantize + matmul。把这个思路 extend 到 KV cache + attention 是一个很有价值的 systems 方向。

如果这个 kernel 做出来了，PRQ INT4 的 peak VRAM 可能从 20.69 GB 降到 ~12 GB（只存量化 cache + streaming dequant），runtime 也会降（因为不需要 materialize dense tensor）。那 PRQ 就从 research winner 变成 deployment winner 了。

---

## 参考链接

- [Self-Forcing (Huang et al. 2025)](https://arxiv.org/abs/2506.08009)
- [Wan2.1 (Team Wan 2025)](https://arxiv.org/abs/2503.20314)
- [QuaRot (Ashkboos et al. 2024)](https://arxiv.org/abs/2404.00456)
- [KIVI (Liu et al. 2024)](https://arxiv.org/abs/2402.02750)
- [FlowCache (Ma et al. 2026)](https://arxiv.org/abs/2602.10825)
- [Quant VideoGen / QVG (Xi et al. 2026)](https://arxiv.org/abs/2602.02958)
- [PagedAttention / vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180)
- [VBench (Huang et al. 2023)](https://arxiv.org/abs/2311.17982)
- [MovieGen (Polyak et al. 2024)](https://arxiv.org/abs/2410.13720)
- [CausVid (Yin et al. 2025)](https://arxiv.org/abs/2412.07772)
- [HiAR (Zou et al. 2026)](https://arxiv.org/abs/2603.08703)
- [Paper GitHub repo](https://github.com/suraj-ranganath/kv-quant-longhorizon/)
- [Marlin GPTQ kernel](https://github.com/IST-DASLab/marlin)
- [FlashAttention (Tri Dao)](https://arxiv.org/abs/2205.14135)

---

如果让我用三句话给你 build intuition：

1. **Self-forcing 把 KV cache 从 systems afterthought 变成 training citizen，cache format 改了 model 行为就变了**——这限制了你能多激进地压缩。
2. **Nominal compression ratio 是骗人的，只有 peak VRAM 是真的**——很多方法压缩了 cache 但 attention read 时 dequantize 回 BF16 把内存吃回去了。
3. **FlowCache soft-prune 之所以 win，是因为它不压缩而是丢弃+留 summary token**——这绕过了 dequantization overhead，代价是 structural fidelity 降低（FlowCache paradox）。

---

# KV Cache Quantization for Self-Forcing Video Generation 深度解析

这篇 paper 是一个非常 thorough 的 empirical study，核心问题其实很简单，但 implications 很深：在 autoregressive video diffusion (具体是 Wan2.1 + Self-Forcing) 这种 long-horizon rollout 场景下，KV cache 会随时间线性增长，而 paper 系统性地检验了 33 种 cache compression 方法在 **真实 deployment** 下到底能不能 deliver 内存收益。下面我把关键 intuition、数学结构、和实验数据都展开讲。

---

## 1. 为什么 self-forcing video 让 KV cache 成为一个 first-order bottleneck

### 1.1 Self-Forcing 的本质

Self-Forcing ([Huang et al. 2025, arXiv:2506.08009](https://arxiv.org/abs/2506.08009)) 解决的是 autoregressive video diffusion 中经典的 **train-test mismatch**。传统做法训练时用 ground-truth frames 作为 context（teacher forcing），但 inference 时必须用自己生成的 frames 当 context。这种 mismatch 在 video 上特别致命，因为 error 会沿时间维度 compound。

Self-Forcing 的做法是在训练时就让 model 见到 **自己生成的 context + KV-cache rollout**，让 model 学会在 self-generated noisy state 下仍然稳定 rollout。这把 KV-cache 从一个 "inference-time systems optimization" 提升为 "training-time first-class citizen"。

**直觉**：一旦 KV-cache 是 training loop 的一部分，cache 的 representation 直接影响 model 学到的 dynamics。这意味着你不能在 inference 时随便换 cache format（比如从 BF16 换成 INT4）而不破坏 model 行为——除非 model 训练时就见过这种压缩 cache。

### 1.2 KV cache 在 video 上的 growth

考虑一个 transformer video generator，单次 rollout 生成 165 frames @ 480 × 832 @ 16fps（约 10.3 秒）。每帧经过 spatial-temporal tokenization 后产生大量 tokens。

KV cache 大小（naive BF16）：

$$
\text{KV size} = 2 \cdot N_{\text{tokens}} \cdot d_{\text{model}} \cdot L_{\text{layers}} \cdot b_{\text{bytes}}
$$

其中：
- $N_{\text{tokens}}$ = sequence length (随帧数线性增长)
- $d_{\text{model}}$ = hidden dimension
- $L_{\text{layers}}$ = transformer 层数
- $b_{\text{bytes}}$ = 每元素字节数（BF16 = 2 bytes, INT4 = 0.5 bytes）
- 前面的 2 是 K 和 V 两个 tensor

Paper 报告 BF16 baseline 在 10s 视频上 peak VRAM **19.28 GB**。如果 rollout 延长到 30s 或 60s，cache 会线性膨胀到不可部署。这就是为什么这是 "scaling path" 上的核心 bottleneck。

---

## 2. Method families 详解（数学 + intuition）

下面我把每个 method family 的核心数学结构讲清楚。这有助于理解为什么不同方法在 deployment 下表现差异巨大。

### 2.1 RTN (Round-to-Nearest) — naive symmetric quantization

最简单的 symmetric blockwise quantization。对一个 block $\mathbf{x} \in \mathbb{R}^B$（block size 通常 16）：

$$
\text{scale} = \frac{\max_i |x_i|}{2^{b-1} - 1}, \quad x_q = \text{round}\left(\frac{x}{\text{scale}}\right), \quad \hat{x} = x_q \cdot \text{scale}
$$

变量含义：
- $b$ = bit width（INT2 = 2, INT4 = 4）
- $2^{b-1} - 1$ = symmetric range 的正向上界（INT4 时为 7）
- $\text{scale}$ = per-block scale factor（FP32 存储）
- $x_q$ = integer code（INT4 用 4 bits 存储）
- $\hat{x}$ = dequantized reconstruction

**Intuition**：RTN 假设 tensor 值大致对称分布。但 KV cache 中（尤其 key tensor）常有 channel-wise outliers，symmetric quantization 会被少数大值 "撑爆" scale，导致大部分正常值的量化粒度极粗。这是 RTN_INT4 SSIM 只有 0.688 (MOVIEGEN) 的根本原因。

### 2.2 KIVI — asymmetric, statistics-aware

KIVI ([Liu et al. 2024, arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) 的关键洞察：**key 和 value 的 outlier 分布不同**，应该用不同的 quantization dimension。

$$
\text{scale}_K = \text{per-channel}(\mathbf{K}), \quad \text{scale}_V = \text{per-token}(\mathbf{V})
$$

具体地：
- **Key**：沿 channel dimension 分组（每个 channel 一组），用 per-channel scale + zero-point。因为 key 的 outlier 集中在特定 channel。
- **Value**：沿 token dimension 分组（每个 token 一组），用 per-token scale + zero-point。因为 value 的 outlier 集中在特定 token。

asymmetric quantization 公式：

$$
\text{scale} = \frac{\max(x) - \min(x)}{2^b - 1}, \quad z = \text{round}\left(-\frac{\min(x)}{\text{scale}}\right), \quad x_q = \text{clip}\left(\text{round}\left(\frac{x}{\text{scale}}\right) + z, 0, 2^b - 1\right)
$$

reconstruction：$\hat{x} = (x_q - z) \cdot \text{scale}$

变量含义：
- $z$ = zero-point（整数 offset，让 0 映射到合适的 code）
- $\text{clip}$ 防止 overflow

**Intuition**：KIVI 在 LLM 上效果很好（2-bit 就能 preserve quality），但在这篇 paper 的 video 场景下表现意外地差——INT2 时 SSIM 只有 0.241 (MOVIEGEN)。推测原因是 video diffusion model 的 KV 分布和 LLM 差异大，channel-wise outlier 模式可能不适用。

### 2.3 QuaRot — rotation-based, outlier-free

QuaRot ([Ashkboos et al. 2024, arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) 的核心思想：**与其在原空间处理 outlier，不如先把 tensor 旋转到一个 outlier-free 的 basis**，再做简单 quantization。

数学上：
$$
\mathbf{x}_{\text{rot}} = \mathbf{H} \mathbf{x}, \quad \mathbf{x}_q = Q_{\text{RTN}}(\mathbf{x}_{\text{rot}}), \quad \hat{\mathbf{x}} = \mathbf{H}^T \mathbf{x}_q
$$

其中 $\mathbf{H}$ 是 **Hadamard matrix**（正交，$\mathbf{H} \mathbf{H}^T = \mathbf{I}$，且元素都是 $\pm 1/\sqrt{d}$，所以乘法极快）。

**Intuition**：Hadamard rotation 是一种 "outlier redistribution"——把少数 channel 上的极端值 "摊平" 到所有 channel。这类似于把一个稀疏信号通过正交变换变成 dense 但幅度均匀的信号，对均匀 quantization 极其友好。

**为什么 deployment 上 QuaRot 慢**：每次 attention read 都需要 inverse rotation $\mathbf{H}^T$，这是 $O(d^2)$ 的矩阵乘法。Paper 报告 QuaRot KV INT4 runtime **236.6s**（vs BF16 58.6s），慢了 4 倍。这就是为什么高 fidelity 不等于好 deployment。

### 2.4 PRQ — Progressive Residual Quantization

两阶段残差量化，思路类似 audio coding 中的 multi-stage vector quantization：

$$
\text{Stage 1:} \quad \mathbf{x}_{q1} = Q_1(\mathbf{x}), \quad \mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}_{q1}
$$
$$
\text{Stage 2:} \quad \mathbf{x}_{q2} = Q_2(\mathbf{r}), \quad \hat{\mathbf{x}} = \hat{\mathbf{x}}_{q1} + \hat{\mathbf{x}}_{q2}
$$

变量含义：
- $\mathbf{r}$ = 第一阶段重建后的 residual（quantization error）
- $Q_1, Q_2$ = 两个独立的 quantizer（都是 symmetric blockwise，64-block chunks）
- base storage 是 4 bits (PRQ_INT4) 或 2 bits (PRQ_INT2)

**Intuition**：第一阶段抓大尺度结构，第二阶段专门精修 residual。这比单阶段 quantization 用相同总 bits 能保留更多信息。Paper 中 PRQ_INT4 达到 SSIM 0.824（MOVIEGEN），是最接近 BF16 的 compressed method，但 compression ratio 只有 1.60×——因为存了两套 codes + scales。

### 2.5 QAQ — Outlier-aware split quantization

QAQ 把每个 block 分成 "bulk" 和 "outlier" 两部分：

1. Clip：$x_{\text{clipped}} = \text{clip}(\mathbf{x}, -\tau, \tau)$，其中 $\tau$ 是 outlier threshold
2. Bulk quantization：$x_{\text{bulk}} = Q_{\text{asym}}(x_{\text{clipped}})$
3. Outlier preservation：把 $|x| > \tau$ 的位置单独存为 (index, value) pairs，用高精度

reconstruction 时把 outlier values 写回 bulk tensor 的对应位置。

**Intuition**：和 KIVI 类似都是处理 outlier，但 QAQ 用 explicit storage 而不是 per-channel scale。Paper 中 QAQ 表现一般——INT4 时 SSIM 0.262，可能是因为 outlier threshold 的选择对 video KV 分布不友好。

### 2.6 Age-Tier & TPTQ — Temporal recency heuristics

这类方法利用一个 strong prior：**recent tokens 比 old tokens 更重要**（因为 attention 在 autoregressive generation 中对 recent context 更敏感）。

Age-Tier：
$$
\text{cache} = [\underbrace{\text{recent slice}}_{\text{high-bit quantizer}} \, || \, \underbrace{\text{old slice}}_{\text{low-bit quantizer}}]
$$

通过 `recent_ratio` mask 划分，recent slice 用 INT4，old slice 用 INT2。

TPTQ 更复杂：recent slice 用 high-precision quantizer，old slice 用 **PRQ** + outlier preservation（`outlier_max_ratio` 控制保留多少 old-key outliers）。

**Intuition**：这和 LLM 里 sliding-window attention 的思路类似——注意力对 recent context 的依赖天然更强。但 paper 显示 Age-Tier INT4 SSIM 0.688，和 RTN INT4 一样，说明这个 prior 在 video diffusion 上的收益不如预期。

### 2.7 FlowCache-inspired — chunkwise retention/pruning（**这是 paper 的 deployment winner**）

这是 paper 最核心的方法家族，灵感来自 FlowCache ([Ma et al. 2026, arXiv:2602.10825](https://arxiv.org/abs/2602.10825))。核心思路是 **chunkwise cache policy**：把 cache 按 frame-aligned chunks 划分，对每个 chunk 决定 retain / compress / summarize / prune。

#### FlowCache Hybrid
```
cache = [recent_chunks (high precision)] + [old_chunks (low precision)]
```
importance 由 `layer_role` modulate（不同 layer 的 cache 重要性不同）。

#### FlowCache Adaptive
在 Hybrid 基础上，对 old chunks 做 **importance scoring**：
$$
\text{score}(\text{chunk}_i) = \alpha \cdot \text{recency}(i) + \beta \cdot \|\Delta \mathbf{K}_i\|_1
$$
其中 $\|\Delta \mathbf{K}_i\|_1$ 是 relative-L1 delta（衡量这个 chunk 相邻 step 的变化幅度，变化大说明活跃）。高分 old chunk 升级到高精度。

#### FlowCache Prune（hard prune）
保留 recent chunks + important old chunks，其余 **直接丢弃**，reconstruction 时填零：
$$
\hat{\mathbf{K}}_{\text{pruned}} = \mathbf{0}
$$

#### FlowCache Soft-Prune（**最强 deployment point**）
不直接丢弃，而是为每个 pruned chunk 存 **一个 pooled BF16 summary token**，reconstruction 时把这个 summary repeat 到整个 evicted span：

$$
\mathbf{s}_i = \text{pool}(\mathbf{K}_i), \quad \hat{\mathbf{K}}_{\text{pruned span } i} = \text{repeat}(\mathbf{s}_i, \text{span length})
$$

变量含义：
- $\mathbf{s}_i \in \mathbb{R}^{d}$ = 第 $i$ 个 pruned chunk 的 summary（一个 token 的 hidden dim）
- $\text{pool}$ = 通常是 mean pooling over tokens in chunk
- $\text{repeat}$ = 把 summary 沿 token dimension 复制到原 chunk 的 token 数

**Intuition**：这其实是一种 **lossy compression with learned placeholder**。Hard prune 填零会让 attention 在那些位置完全无信号，导致 divergence；soft-prune 保留一个 "average" 信号，让 attention 至少有一个 reasonable 的 prior。这类似于 neural network 中的 "padding token" 但内容是数据驱动的。

**为什么 soft-prune 在 VBench 上接近 BF16 但 SSIM 差**：VBench 测的是 "这个视频看起来像不像真实视频"，而 SSIM 测的是 "这个视频和 BF16 参考视频是否像素级一致"。Soft-prune 保留了视觉 plausibility（因为 summary token 提供了足够的 contextual prior 让 model 生成 plausible 内容），但生成的具体像素和 BF16 diverge——这是 **"FlowCache paradox"** 的本质。

---

## 3. 实验数据深度解读

### 3.1 关键 operating points（Table 2 + Appendix C）

| Benchmark | Method | Comp. | Peak VRAM (GB) | Runtime (s) | VBench Img. | SSIM | LPIPS | Drift |
|-----------|--------|-------|----------------|-------------|-------------|------|-------|-------|
| MovieGen | BF16 | 1.00× | 19.28 | 58.6 | 0.739 | 1.000 | 0.000 | 0.739 |
| MovieGen | FlowCache Soft-Prune INT4 | 5.49× | **11.71** | 75.0 | **0.739** | 0.544 | 0.297 | 0.738 |
| MovieGen | FlowCache Prune INT4 | 5.50× | 11.71 | 72.2 | 0.727 | 0.457 | 0.412 | 0.726 |
| MovieGen | PRQ INT4 | 1.60× | 20.69 | 160.0 | 0.739 | **0.824** | **0.082** | 0.739 |
| MovieGen | QuaRot KV INT4 | 3.20× | 19.98 | 236.6 | 0.738 | 0.724 | 0.148 | 0.738 |
| MovieGen | RTN INT4 Recent2 | 2.43× | 21.37 | 68.9 | 0.736 | 0.732 | 0.148 | 0.735 |
| MovieGen | RTN INT4 Refresh | 3.20× | 22.64 | 65.0 | 0.736 | 0.693 | 0.178 | 0.735 |
| MovieGen | Spatial Mixed (QuaRot fg, RTN bg) | 3.46× | 14.38 | 224.8 | **0.399** | 0.433 | 0.570 | 0.394 |

读这张表的关键 observations：

**Observation 1**: FlowCache Soft-Prune INT4 是唯一同时达到 "大压缩 + 真 VRAM 降低 + 可接受 runtime + 接近 BF16 的 VBench" 的点。VRAM 从 19.28 → 11.71 GB（降 39%），compression 5.49×，VBench 完全持平 BF16。

**Observation 2**: PRQ INT4 的 SSIM 0.824 比 FlowCache Soft-Prune 的 0.544 高出很多，LPIPS 0.082 vs 0.297 也是碾压。但 PRQ 的 peak VRAM 20.69 GB **比 BF16 还高**，runtime 160s 是 BF16 的 2.7 倍。这是一个 "research winner but deployment loser"。

**Observation 3**: Spatial Mixed 是 catastrophic failure——VBench 0.399（比 BF16 的 0.739 跌了 46%），drift 0.394。这是 paper 最强的 negative result。

### 3.2 "Peak VRAM anomaly" — 为什么 compression 没换来 VRAM 降低

这是 paper 最 systems-level 的重要发现。看这组数据：

| Method | Comp. | Peak VRAM | BF16-equiv KV at peak | Compressed KV at peak |
|--------|-------|-----------|------------------------|------------------------|
| BF16 | 1.00× | 19.28 GB | — | — |
| RTN INT4 | 3.20× | 19.98 GB | — | — |
| RTN INT4 Recent2 | 2.43× | **21.37 GB** | 11.25 GB | 4.62 GB |
| RTN INT4 Refresh | 3.20× | **22.64 GB** | 11.25 GB | 3.51 GB |
| QuaRot KV INT4 | 3.20× | 19.98 GB | — | — |

**RTN INT4 Recent2 和 Refresh 的 peak VRAM 比 BF16 还高**，尽管它们确实压缩了 KV cache（compressed KV 只有 3.51 GB）！

原因在 implementation：

1. **RTN Recent2**：保留最近 2 个 frame-blocks 的 BF16 KV（`recent_k` / `recent_v`），旧 prefix 是量化的。但 attention read 时需要 **dequantize 整个旧 prefix 回 dense BF16** 再和 recent tail 拼接。这一瞬间同时存在：量化 cache + dequantized BF16 prefix + BF16 recent tail + 中间 buffer。

2. **RTN Refresh**：denoising 期间保持 BF16 cache，然后在 refresh pass 里重新量化写回。refresh 那一刻同时有 BF16 cache + 正在写入的量化 cache + transform buffers。

3. **QuaRot**：rotated low-bit values 在 attention 前需要 inverse Hadamard transform，产生 dense BF16 tensor。

**Intuition**：这些方法在 **steady state** 确实节省内存，但 **transient peak**（attention compute 那一刻）被 dequantization / reconstruction buffer 主导。这揭示了一个 deep issue：**KV cache compression 不能只看存储，必须看整个 attention compute path 的内存生命周期**。

Paper 的 trace（Figure 6）显示 RTN_INT4_REFRESH 在 peak 时刻 allocated 21.41 GB，其中 BF16-equivalent KV 11.25 GB，compressed KV 3.51 GB——剩下的 ~7 GB 都是 reconstruction buffers。这是一个巨大的 systems 教训。

### 3.3 Cross-benchmark 一致性

Paper 报告了 MOVIEGEN 和 STORYEVAL 之间的 correlation：

| Metric | Cross-benchmark r |
|--------|-------------------|
| Compression ratio | 0.9999 |
| Runtime | 0.9996 |
| Peak VRAM | 0.99999 |
| VBench imaging | 0.9318 |
| Drift-last | 0.9374 |

**Intuition**：systems metrics 几乎完全 deterministic（由 method 决定，与 prompt 无关），而 quality metrics 有 prompt-dependent 噪声但仍然高相关。这说明结论 **不是 prompt suite 的 artifact**，是 method 本身的 property。这对 empirical study 的可信度很重要。

### 3.4 Spatial mixed precision 为什么失败

这个 negative result 值得深究。方法逻辑：
1. 用 temporal variance 构建 foreground/background mask（高 variance = foreground）
2. Foreground tokens → RTN INT4（高精度）
3. Background tokens → RTN INT2（低精度）

听起来很合理——vision model 应该更关心 foreground object。但实际 SSIM 0.433, VBench 0.399，完全 collapse。

**我的解读**：在 autoregressive video generation 中，**model 的 attention pattern 和人类视觉的 "foreground bias" 不一致**。Background tokens 在 attention 中承担 **全局上下文 / 场景一致性** 的作用，粗暴 INT2 化会破坏 scene coherence。这类似于 LLM 中发现 stop words 也很重要——你不能用人类 prior 替代 model 的实际 dependency structure。

这个 result 对未来的 cache compression 设计是个重要 warning：**不要假设你知道 model "应该" 关心哪些 token**。

---

## 4. 设计空间的 Pareto 分析

Paper 识别了四个 frontier：

1. **Balanced practical**：FlowCache soft-prune INT4 独占
2. **Quality-preserving compression**：FlowCache prune/soft-prune INT4 + RTN Recent2
3. **Systems efficiency**：很少的方法真正改善 memory 而不付出灾难性 runtime/quality cost
4. **Quality-first**：BF16, PRQ, QuaRot, RTN Recent2

这个 Pareto 结构的核心启示：**deployment winner 和 research winner 是不同的 method**。这和 LLM serving 领域的经验一致——很多看起来漂亮的 quantization paper 在真实 serving 下被 dequantization overhead 吃掉收益。

---

## 5. 与相关工作的定位

### 5.1 vs PagedAttention ([Kwon et al. 2023, arXiv:2309.06180](https://arxiv.org/abs/2309.06180))
PagedAttention 解决的是 LLM serving 中 KV memory **碎片化** 问题（request-level throughput），用的是 OS-style paged allocation。它不压缩 KV，只优化 allocation。这篇 paper 解决的是 **单次 long rollout 的绝对内存容量** 问题，是正交的——可以组合。

### 5.2 vs QVG / QVG-Pro ([Xi et al. 2026, arXiv:2602.02958](https://arxiv.org/abs/2602.02958))
QVG 是 training-free KV-cache quantization for autoregressive video，报告 2-bit 就能 preserve quality，还有更强的 QVG-Pro mode。Paper 没能 reproduce（截至 2026-03-17 无公开 code），这是 future work 的重要方向。如果 QVG 的 memory-quality Pareto 真的更好，可能取代 FlowCache soft-prune 的 deployment winner 地位。

### 5.3 vs CausVid ([Yin et al. 2025, arXiv:2412.07772](https://arxiv.org/abs/2412.07772)) & HiAR ([Zou et al. 2026, arXiv:2603.08703](https://arxiv.org/abs/2603.08703))
这两个不走 cache compression 路线：
- **CausVid**：把 slow bidirectional teacher 蒸馏成 fast causal student，从架构层面加速
- **HiAR**：hierarchical denoising 减少 long rollout 的 error accumulation

它们和 KV compression 正交，但 paper 建议未来在它们上跑同一套 33-method benchmark，看结论是否 transfer。

### 5.4 vs Wan2.1 ([Team Wan 2025, arXiv:2503.20314](https://arxiv.org/abs/2503.20314))
Wan2.1 是 base video generator，large-scale transformer-based。Self-Forcing 在它上面加了 causal KV-cache rollout training。Paper 的所有实验都在这个 stack 上。

### 5.5 vs MovieGen ([Polyak et al. 2024, arXiv:2410.13720](https://arxiv.org/abs/2410.13720))
MovieGen 是 Meta 的 media foundation model，paper 用它衍生出的 prompt suite 作为 MOVIEGEN benchmark 的来源。

### 5.6 vs VBench ([Huang et al. 2023, arXiv:2311.17982](https://arxiv.org/abs/2311.17982))
VBench 是 multi-dimensional video realism benchmark，paper 用它的 imaging quality 分数作为 perceptual realism axis。但 paper 强调 VBench alone 不够——必须配合 BF16-referenced SSIM/LPIPS/PSNR 才能 catch structural hallucination。

---

## 6. 评估方法论的设计哲学

Paper 的 evaluation 有一个 deliberate 的 **dual-axis** 设计：

| Axis | Metric | 回答的问题 |
|------|--------|------------|
| Perceptual realism | VBench imaging | 这个视频看起来像不像真实视频？ |
| Structural fidelity | SSIM, LPIPS, PSNR (vs BF16) | 这个视频和 BF16 参考是否像素级一致？ |
| Temporal stability | Drift-last imaging | 长 rollout 末段是否还稳定？ |
| Systems | Peak VRAM, Runtime, Comp. | 能不能部署？ |

**关键 insight**：VBench 高 + SSIM 低 = "FlowCache paradox"。一个方法可以生成看起来 plausible 但和 BF16 结构不同的视频。这在 video generation 评估中是一个微妙但重要的 distinction——如果只看 VBench，你会误以为 FlowCache soft-prune "完全无损"，但实际上它的输出和 BF16 是不同的视频。

类似地，**Compression ratio 高 + Peak VRAM 不降 = "RTN anomaly"**。一个方法可以报告漂亮的 compression 数字但 deployment 时内存根本没省。这是 systems 社区常被忽略的：**nominal compression ≠ realized memory relief**。

---

## 7. Future work 的方向（我的延伸思考）

Paper 提了 5 个 future direction，我重点展开最有价值的几个：

### 7.1 Attention path 重设计（最关键）
当前 stack 的核心问题是 quantized KV 在 attention 前被 dequantize 成 dense BF16。要真正实现内存收益，需要 **in-attention dequantization**：

$$
\text{Attention}(Q, K_q, V_q) = \text{softmax}\left(\frac{Q \cdot \text{dequant}(K_q)^T}{\sqrt{d_k}}\right) \text{dequant}(V_q)
$$

但能不能 **fused** 实现，让 dequantization 在 attention kernel 内部 streaming 进行，不 materialize 完整的 dense tensor？这需要 custom CUDA kernel（类似 FlashAttention 的思路但带 dequantization）。Marlin kernel ([GPTQ-Marlin](https://github.com/IST-DASLab/marlin)) 在 LLM weight quantization 上做过类似的事，可以借鉴。

### 7.2 Training-aware quantization
当前所有方法都是 training-free（post-hoc）。但 Self-Forcing 的训练已经包含 KV-cache rollout，理论上可以让 model 在训练时就见到量化 cache（straight-through estimator 或 QAT）。这可能让 INT4 甚至 INT2 的 fidelity 大幅提升，因为 model 会学会 robust to quantization noise。这是 QVG-Pro 可能走的路线。

### 7.3 Layer-adaptive precision
不同 layer 的 KV 对 quantization 的敏感度不同。Paper 的 FlowCache Adaptive 已经探索了 chunk-level adaptive，但 layer-level adaptive 还没系统做。一个直觉：early layer 提取 low-level feature，KV 可能对 quantization 更敏感；late layer 做 high-level reasoning，可能更 robust。这需要 per-layer sensitivity profiling。

### 7.4 更长 horizon 的 drift 研究
10s 视频只是 long-horizon 的 proxy。真正的挑战在 30s+——identity breakdown, scene inconsistency, action drift 会更严重。Paper 用 prefix-quality curve 推断 drift，但直接测 60s rollout 会更说服力。

### 7.5 KV cache 不是唯一瓶颈
在 video diffusion 中，**activation memory**（denoising step 中的中间激活）也是大头。Paper 聚焦 KV cache，但实际 deployment 还需要考虑 activation checkpointing / gradient checkpointing 的组合优化。

---

## 8. 一个更深的问题：为什么 FlowCache soft-prune 这么有效？

让我 speculate 一下直觉。Soft-prune 的成功可以分解成几个 factor：

1. **Frame-aligned chunking**：video 的 temporal coherence 让相邻 frame 的 KV 高度冗余。Frame-aligned chunk 自然 capture 这种冗余。
2. **Summary token 作为 "context anchor"**：即使 pruned，summary 提供了一个 "这个区域曾经有什么" 的 anchor，让 attention 不会完全 lose context。这比 hard prune 的填零温和得多。
3. **Importance scoring 选对 chunk**：relative-L1 delta 捕获 "活跃" chunk（近期变化大的），这些 chunk 对 generation 最关键。
4. **Recency prior 在 video 上比 LLM 更强**：video 的 causal generation 强烈依赖 recent frames（因为 physical continuity），所以保留 recent chunks 高精度是正确的 prior。

但为什么 SSIM 仍然差？因为 summary token 是 **lossy average**——它丢失了 chunk 内部的 spatial-temporal detail。Model 用这个 averaged context 生成的视频会 plausible 但和 BF16 的 detailed context 生成的视频在像素层面 diverge。这是 **information-theoretic 必然**：你丢了 detail，model 就 hallucinate detail。

---

## 9. 总结：这篇 paper 的真正贡献

这篇 paper 不是一个 "新方法" paper，而是一个 **design space map**。它的价值在于：

1. **Empirical rigor**：33 methods, 610 observations, 6 axes, 2 benchmarks, cross-benchmark correlation 验证。这是 long-horizon video KV compression 领域目前最系统的 empirical study。
2. **Systems-level insight**：揭示了 nominal compression ≠ peak VRAM relief 的 anomaly，这是纯 ML paper 不会 catch 的。
3. **Quality decomposition**：VBench vs SSIM 的分离（FlowCache paradox）提供了 video generation evaluation 的方法论启示。
4. **Negative result 的价值**：Spatial mixed precision 的 collapse 警告了 "人类 prior 替代 model dependency" 的陷阱。
5. **Future roadmap**：明确指出 attention path 重设计是最关键的 systems follow-up。

代码和数据在 [GitHub: suraj-ranganath/kv-quant-longhorizon](https://github.com/suraj-ranganath/kv-quant-longhorizon/)。

---

## 参考链接

- [Self-Forcing (Huang et al. 2025)](https://arxiv.org/abs/2506.08009)
- [Wan2.1 (Team Wan 2025)](https://arxiv.org/abs/2503.20314)
- [QuaRot (Ashkboos et al. 2024)](https://arxiv.org/abs/2404.00456)
- [KIVI (Liu et al. 2024)](https://arxiv.org/abs/2402.02750)
- [VBench (Huang et al. 2023)](https://arxiv.org/abs/2311.17982)
- [PagedAttention / vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180)
- [MovieGen (Polyak et al. 2024)](https://arxiv.org/abs/2410.13720)
- [FlowCache (Ma et al. 2026)](https://arxiv.org/abs/2602.10825)
- [Quant VideoGen / QVG (Xi et al. 2026)](https://arxiv.org/abs/2602.02958)
- [CausVid (Yin et al. 2025)](https://arxiv.org/abs/2412.07772)
- [HiAR (Zou et al. 2026)](https://arxiv.org/abs/2603.08703)
- [Paper GitHub repo](https://github.com/suraj-ranganath/kv-quant-longhorizon/)

如果你想 build 更深的 intuition，我建议重点看三件事：(1) FlowCache soft-prune 的 summary token 机制，(2) RTN anomaly 的 trace（Figure 6），(3) Spatial mixed 的 failure mode。这三点分别对应 "什么 work"、"什么看起来 work 但其实不 work"、"什么看起来应该 work 但 catastrophic fail"——构成了这个 design space 的核心边界。
