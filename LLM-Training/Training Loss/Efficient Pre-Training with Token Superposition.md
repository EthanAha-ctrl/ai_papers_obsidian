---
source_pdf: Efficient Pre-Training with Token Superposition.pdf
paper_sha256: a573055890077081e32b3e5a9eb7df517bfe540d3a1b506fdc1f442d1d127865
processed_at: '2026-08-04T01:53:15-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TST 用人话讲

## 一句话

把相邻的 s 个 token "捆"成一个超级 token 喂模型，让模型在同样的计算量下"看" s 倍的文字，训完一段再切回逐 token 预测，模型快速恢复还比 baseline 强。

## 核心类比

小孩学说话先学"猫""狗""跑"这些概念，后学"猫追狗"还是"狗追猫"的语法。TST 就是这个思路：**训练前期让模型先抓大意，后期再补顺序**。

具体怎么抓大意？把 "the cat sat on the mat" 这 6 个 token 的 embedding 求平均，变成 1 个 super token。模型看到的是"这 6 个词混在一起的意思"，大概知道这段讲猫和垫子。然后让模型预测下一组 6 个 token 的混合表示。顺序信息丢了，但 co-occurrence 统计保留了。

## 为什么这样能省算力

假设模型本来一次处理 4096 个 token。用 TST 把 s=6 个捆 1 个，同样 4096 个 super token 其实对应 24576 个原始 token。模型参数没变、FLOPs 没变、memory 没变，但"吃掉"的 data 多了 6 倍。

这就是 paper 的核心 framing：**equal-FLOPs per step，但 throughput 涨 s 倍**。每一步模型见到的文字量翻了 s 倍，loss 下降自然更快。

## 两个阶段怎么回事

Phase 1（superposition）：占训练步数的 r（比如 30%），按上面方式捆着训，loss 是 non-sensical 的，模型这时候不能正常 generate 文本（你 sample 出来是 6 个 token 的混合概率）。

Phase 2（recovery）：剩下 1-r 步数，切回标准 next-token prediction。模型很快恢复正常 AR 能力，而且 loss 曲线直接超过同等 FLOPs 的 baseline。

## 关键 trick：两阶段必须共享 embedding 和 head

这是整篇 paper 最 important 的 ablation（Table 2）：

- 正常 TST：loss 2.676，比 baseline 2.808 好 0.13
- 在 phase 2 开始时随机重置 input embedding 和 LM head：loss 2.938，比 baseline 还差 0.13

Phase 1 学到的"这段话讲什么"的 representation 必须能 transfer 到 phase 2。你如果把 embedding 重置，phase 1 学的全白费，还浪费了 30% 的步数。之前类似方法（Bolmo 等）没发现这个效果，因为他们用 adapter 在两阶段间过渡，反而模糊了 representation 对齐这个关键因素。

TST 的聪明之处在于：input embedding 和 output head 在两阶段**完全不变**，phase 1 学到的 geometry 直接复用。

## Loss 怎么算的

模型预测下一组 s 个 token，目标让这 s 个 token 每个拿到 1/s 的概率。数学上等价于对 s 个 token 分别算 cross-entropy 然后取平均：

$$\mathcal{L}_{\mathrm{MCE}} = \frac{1}{s} \sum_{y \in \mathbf{y}} \mathcal{L}_{\mathrm{CE}}(\mathbf{z}, y)$$

- $s$: bag size
- $\mathbf{y}$: 这一组 s 个 target token
- $\mathbf{z}$: logits
- $\mathcal{L}_{\mathrm{CE}}$: 标准 cross-entropy

严格推导有个 $\log s$ 常数项要减（因为 uniform target 有 entropy $\log s$），但这个常数对梯度是 0，drop 掉就行。这样能直接用 PyTorch 编译优化的 CE kernel，for-loop 累加 s 次，工程上零改动，几乎无 overhead。

## 数字感受

10B MoE 模型最 large 的 run：

| 设置 | Tokens | B200-hours | Loss |
|------|--------|------------|------|
| Baseline | 1.05T | 12311 | 2.252 |
| TST (s=16) | 2T | 4768 | 2.236 |

TST 训了 2 倍数据，用了不到 40% 的 compute，loss 还更低。Equal-loss 比较下 TST 快 **2.5x**。

3B dense 模型 equal-FLOPs 比较：

| 设置 | Loss | HellaSwag | MMLU |
|------|------|-----------|------|
| Baseline 42B tokens | 2.808 | 57.6 | 31.2 |
| TST 105B tokens (s=6, r=0.3) | 2.676 | 62.4 | 32.8 |

同样 247 B200-hours，TST 各项指标都赢。

## 超参数怎么调

Bag size s 和 step ratio r 有个 U 型关系（Table 4, 5）：

- s 太小（1-2）：效果弱，捆得不够
- s 太大（>12）：恢复难，phase 2 补不回来
- r 太小（<0.1）：TST 没起效
- r 太大（>0.5）：recovery phase 不够

Sweet spot 大概 **s ∈ [4, 8]，r ∈ [0.2, 0.4]**。

## 大 s 时要换权重

s=4 用 uniform 加权最好。s=16 时远处 token 太难预测，uniform 不行，要用 power-law 权重 $g(i) = 1/i$ 给近处 token 更大权重。这跟 DCLM 数据里 mutual information 随距离衰减遵循 $d^{-1.25}$ 的 power law 吻合（Figure 10）。

直觉：预测下一个 token 容易，预测下第 16 个 token 难，所以 loss 里远处的项应该权重小。

## 两个正交机制

Paper 做了 ablation（Figure 6）：

- 只改 input（捆输入，但还预测单 token）：有改善
- 只改 output（单 token 输入，预测 next bag）：有改善
- 两个都改（完整 TST）：**进一步改善**

Input 和 output superposition 是两个独立机制，收益可叠加。Input 改变 granularity 和 FLOPs-per-info，output 改变 prediction target 和 gradient signal。

## 和 MTP 的区别

MTP ([Multi-Token Prediction](https://arxiv.org/abs/2404.19737), DeepSeek-V3 用过）：

- 用 k 个额外 head 精确预测 next k 个 token，顺序保留
- 不增加 throughput，每步还是看同样多 token
- 多 k 个 head 的参数
- Inference 可做 speculative decoding

TST：

- 用 1 个 head 预测无序 bag，顺序丢掉
- Throughput 涨 s 倍
- 零额外参数
- Inference 完全不动

两者**正交**，可以组合：TST 提 throughput，MTP 做 multi-token supervision。

## 局限

1. **只在 compute-bound 场景有用**。TST 用更多数据换更好 loss。如果未来 data 用完了（[Kim et al. 2025](https://arxiv.org/abs/2509.14786) 说会到这一天），output-only superposition（不增加数据量）更合适。

2. **Phase 1 期间模型不能 generate**。bag 预测无采样能力，必须经过 phase 2 才能用。所以 TST 只适用于 pre-training，post-training 阶段没法直接用。

3. **没做多 seed runs**，scaling laws 没建立，statistical significance 不明确。

4. **Long context 优势没评测**。TST folding 让 effective context 变 s 倍，可能利于 long context 数据。

## 直觉总结

TST 的核心 insight 就一句话：**语言学习的 coarse-to-fine 自然规律**。先学 topic 和 co-occurrence（低频信号），后学精确 ordering（高频信号）。这跟 ViT 先大 patch 后小 patch ([Anagnostidis et al.](https://proceedings.mlr.press/v235/anagnostidis24a.html))、先 formal language 后 natural language ([Hu et al.](https://aclanthology.org/2025.acl-long.478/))、先 subword 后 byte ([Minixhofer et al. Bolmo](https://arxiv.org/abs/2512.15586)) 是同一个 universal principle。

工程价值在于：< 30 行代码，不改任何 infrastructure，10B MoE 规模 2.5x 加速。这种"简单到不像会 work 但确实 work"的方法，通常说明我们之前对 pre-training 的理解漏掉了某个 fundamental 的东西——在这里就是 throughput 本身就是效率的核心 driver，而不是 per-token FLOPs。

参考链接：
- [TST Paper (Nous Research)](https://nousresearch.com)
- [Patch-Level Training (Shao et al. ICLR 2025)](https://openreview.net/forum?id=dDpB23VbVa)
- [MTP (Gloeckle et al. ICML 2024)](https://arxiv.org/abs/2404.19737)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [Bolmo (Minixhofer et al.)](https://arxiv.org/abs/2512.15586)
- [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/)
- [Decoupling Subword Tokenization Benefits](https://arxiv.org/abs/2604.27263)
- [Future Summary Prediction](https://arxiv.org/abs/2510.14751)
- [modded-nanogpt PR #178](https://github.com/KellerJordan/modded-nanogpt/pull/178)
- [SmolLM2](https://arxiv.org/abs/2502.02737)
- [DCLM](https://arxiv.org/abs/2406.19436)
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [TorchTitan](https://openreview.net/forum?id=SFN6Wm7YBI)
- [ViT Coarse-to-Fine (Anagnostidis et al.)](https://proceedings.mlr.press/v235/anagnostidis24a.html)
- [Pre-pre-training on Formal Languages](https://aclanthology.org/2025.acl-long.478/)
- [Scaling Laws under Infinite Compute](https://arxiv.org/abs/2509.14786)
- [Word2Vec](https://arxiv.org/abs/1301.3781)
- [Entropy in Literary English (Ebeling & Pöschel)](https://iopscience.iop.org/article/10.1209/0295-5075/26/4/001)
- [WSD Learning Rate](https://arxiv.org/abs/2410.05192)
- [LM-Eval Harness](https://zenodo.org/records/10256836)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

---

# Token Superposition Training (TST) 深度解析

## 1. 论文核心问题与动机

这篇 paper 来自 Nous Research，解决一个具体的 pre-training efficiency 问题：给定固定 compute budget，如何让模型"吃掉"更多 data tokens？

关键背景观察来自 Gigant et al. 2026 ([Decoupling Subword Tokenization Benefits](https://arxiv.org/abs/2604.27263))：subword tokenizer 相对 byte-level 的优势，主要源自 throughput 提升（coarser tokens → 更少 sequence positions → 同样 FLOPs 处理更多 bytes）。这个 "throughput hypothesis" 是 TST 的理论起点。

TST 的核心提问：**能否在不修改最终模型架构和 inference 动态的前提下，强制提高 training-time 的 token throughput？**

## 2. 方法架构详解

### 2.1 两阶段流程

**Phase 1 — Superposition Phase（占比 r）：**

```
原始序列:  [t1, t2, t3, t4, t5, t6, ...]  (长度 L×s)
分 bag:    [(t1,t2,t3,t4), (t5,t6,t7,t8), ...]  (s=4)
求平均:   [s_token1, s_token2, ...]  (长度 L)
模型处理: hidden states 长度 L
预测:     next bag of s tokens，用 MCE loss
```

**Phase 2 — Recovery Phase（占比 1-r）：**
- 切换回标准 next-token prediction
- 保持相同 model architecture（包括 input embedding 和 output LM head）

关键设计决策：**Equal-FLOPs comparison**。Phase 1 时把序列长度从 L 扩展到 L×s，这样每步 FLOPs 与 baseline 完全相同，但模型"看到" s 倍的 raw data tokens。

### 2.2 Input Superposition 数学形式

给定 tokenized sequence 形状 $B \times L \times V$（B: batch size, L: sequence length, V: vocab size），分割为 non-overlapping bags：

$$\text{shape}: B \times l \times s \times V \quad \text{其中} \quad l = L/s$$

对 embedding 层做 averaging（代码 Listing 2）：

$$\mathbf{h}_j = \frac{1}{s} \sum_{k=0}^{s-1} \mathbf{E}_{t_{j \cdot s + k}}$$

- $\mathbf{h}_j \in \mathbb{R}^d$: 第 j 个 s-token 的 hidden representation
- $\mathbf{E}_t$: token t 的 embedding
- $d$: residual stream dimension

数值上用 float32 累加（代码注释明确提到），防止 bf16 下累加误差。

### 2.3 Output Superposition: Multi-hot Cross-Entropy (MCE) Loss 推导

**标准 CE Loss（Eq. 1）：**
$$\mathcal{L}_{\mathrm{CE}}(\mathbf{z}, y) = -z_y + \log \sum_{i=1}^{V} \exp(z_i)$$

- $\mathbf{z} \in \mathbb{R}^V$: logits
- $y$: single target index
- $z_y$: logit at target position
- $V$: vocabulary size
- $\log \sum_i \exp(z_i)$: log-partition function（normalization term）

**MCE Loss 完整推导（Appendix C.1）：**

Target 是 bag $\mathbf{y}$ 内 s 个 tokens 上的 uniform distribution：
$$t_i = \begin{cases} 1/s & \text{if } i \in \mathbf{y} \\ 0 & \text{otherwise}\end{cases}$$

标准 CE with this target：
$$\mathrm{CE}(\mathbf{t}, P) = -\frac{1}{s} \sum_{y \in \mathbf{y}} \log P(y) = -\frac{1}{s} \sum_{y \in \mathbf{y}} \log \frac{\exp(z_y)}{\sum_{i=1}^V \exp(z_i)}$$

由于 target distribution 有 entropy $H(\mathbf{t}) = \log s \neq 0$，CE 最小值是 $\log s$，不趋于 0。为恢复"optimum 时 loss=0"性质，减去 entropy（即用 KL divergence）：

$$\mathcal{L}_{\mathrm{MCE}}(\mathbf{z}, \mathbf{y}) = \mathrm{KL}(\mathbf{t} \| P) = -\frac{1}{s} \sum_{y \in \mathbf{y}} \log \frac{s \cdot \exp(z_y)}{\sum_{i=1}^V \exp(z_i)}$$

展开整理得 Eq. 4：

$$\mathcal{L}_{\mathrm{MCE}}(\mathbf{z}, \mathbf{y}) = \frac{1}{|\mathbf{y}|} \sum_{y \in \mathbf{y}} \left(-z_y + \log \sum_{i=1}^V \exp(z_i)\right) - \log |\mathbf{y}|$$

- $|\mathbf{y}| = s$: bag size
- $\log |\mathbf{y}| = \log s$: entropy correction，常数项

**工程化简化（Eq. 3）：**
$$\mathcal{L}_{\mathrm{MCE}}(\mathbf{z}, \mathbf{y}) = \frac{1}{|\mathbf{y}|} \sum_{y \in \mathbf{y}} \mathcal{L}_{\mathrm{CE}}(\mathbf{z}, y)$$

因为 $\log |\mathbf{y}|$ 对 $\mathbf{z}$ 的梯度为 0，可以 drop。这个 trick 让 TST 直接复用 PyTorch 编译优化的 `cross_entropy` kernel（Listing 3），用 for-loop 累加 s 次，几乎无 overhead。

### 2.4 Causality 处理

为保持半因果半自回归性质，labels 左移 $s-1$ positions：
- Bag at positions $[t, t+s-1]$ 预测 next bag at $[t+s, t+2s-1]$
- 模型仍从左到右预测，但 bag 内 tokens 顺序信息丢失

代码（Listing 3）中用 causal padding 实现：
```python
labels = F.pad(labels, (0, s-1), value=-100)[..., s-1:].view(bs, seq, s)
```

### 2.5 Alternative Loss: MCE_Alt（Appendix C.2）

论文还探索了 sum-to-one variant：
$$\mathcal{L}_{\mathrm{MCE}_{\mathrm{Alt}}}(\mathbf{z}, \mathbf{y}) = -\log \sum_{y \in \mathbf{y}} \exp(z_y) + \log \sum_{i=1}^V \exp(z_i)$$

这等价于把整个 bag 看作一个 composite label，要求 $\sum_{y \in \mathbf{y}} P(y) = 1$，让 model 自己决定 bag 内 probability 分配。作者报告效果与 MCE 相同，但因不能简化为标准 CE kernel，未采用。

## 3. 实验数据深度解读

### 3.1 主结果表（Table 1）关键数字

| Setting | Tokens | B200-Hours | Final Loss | HellaSwag | MMLU |
|---------|--------|------------|------------|-----------|------|
| 3B Baseline (50k steps) | 105B | 622 | 2.640 | 63.9 | 33.3 |
| 3B TST (20k steps, s=6, r=0.3) | 105B | 247 | 2.676 | 62.4 | 32.8 |
| 10B-A1B MoE Baseline | 1.05T | 12311 | 2.252 | 70.1 | 37.4 |
| 10B-A1B MoE TST (s=16) | 2T | 4768 | 2.236 | 71.2 | 39.0 |

**Equal-FLOPs 比较**（3B, 247 hours）：TST loss 2.676 vs Baseline loss 2.808，TST 完胜。

**Equal-Loss 比较**：TST 247 hours 达到 2.676，baseline 需 443 hours 达到 2.677，**2.5x 加速**。

### 3.2 超参数敏感性（Table 4, 5）

以 270M / 100k steps 为例（Table 5 数据）：

| r\s | 4 | 6 | 8 | 12 |
|-----|---|---|---|----|
| 0.1 | 3.0603 | 3.0512 | 3.0488 | 3.0511 |
| 0.2 | 3.0534 | 3.0486 | 3.0479 | 3.0517 |
| 0.3 | 3.0519 | **3.0480** | 3.0485 | 3.0538 |
| 0.5 | 3.0533 | 3.0513 | 3.0534 | 3.0611 |
| 0.9 | 3.0908 | 3.1009 | 3.1161 | 3.1348 |

**U-shape pattern 明显**：r 太小 TST 效果不足，r 太大 recovery phase 不足。Sweet spot 在 r≈0.2-0.3, s≈6-8。

### 3.3 Power-law Weighting（Appendix D, Table 7）

对于大 s（如 s=16），uniform weighting 不够好。作者引入 position-dependent weight $g(i)$：
$$\mathcal{L}_{\mathrm{MCE}}(\mathbf{z}, \mathbf{y}, g) = \frac{1}{\sum_i g(i)} \sum_{y \in \mathbf{y}} g(i) \mathcal{L}_{\mathrm{CE}}(\mathbf{z}, y)$$

测试了 $g(i) = 1$（uniform）、$g(i) = 1/i$（power law）、$g(i) = \exp(-i)$、$g(i) = \delta_1(i)$。

**关键 insight**：DCLM 数据集中 token 间的 mutual information 随距离衰减遵循 power law（Figure 10）：
$$\mathrm{MI}(d) \approx C_0 + a \cdot d^k, \quad C_0 \approx 3.63, \, a \approx 1.35, \, k \approx -1.25$$

这解释了为何 power-law weighting 在大 s 时更好：距离当前位置越远的 token，预测难度越大，应给予更小权重。这呼应了 Ebeling & Pöschel 1995 的文学英语 mutual information 衰减规律 ([EPL paper](https://iopscience.iop.org/article/10.1209/0295-5075/26/4/001))。

### 3.4 Representation Alignment 消融（Table 2）

| 设置 | Final Loss |
|------|------------|
| Baseline | 2.808 |
| TST | 2.676 |
| TST + 在 recovery phase 开始时随机重置 input embedding 和 LM head | 2.938 |

**这个实验至关重要**：随机化后 TST 比 baseline 还差 0.13，说明 phase 1 学到的 representation 必须能 transfer 到 phase 2。TST 之所以成功，关键在于两个 phase 共享同一个 input embedding 和 output head，避免了之前 compressive 方法（如 Bolmo）需要的 adapter alignment phase。

## 4. Ablation: Input vs Output Superposition（Figure 6）

| Setting | Final Loss (recovery phase) |
|---------|---------------------------|
| Baseline | 基准 |
| Input-only (bag input, single token target) | 改善 |
| Output-only (single input, bag target) | 改善 |
| Full superposition | **进一步改善** |

Input 和 output superposition 是两个**正交**机制：
- **Input superposition** 改变 input granularity 和 FLOPs-per-information
- **Output superposition** 改变 prediction target 和 gradient signal

收益可叠加，没有 interference。这个 ablation 强烈支持"两个独立机制"的解释。

## 5. 为什么 TST 有效？构建 Intuition

### 5.1 类比 Vision Transformer Patch Scheduling

Anagnostidis et al. 2024 ([ICML paper](https://proceedings.mlr.press/v235/anagnostidis24a.html)) 在 ViT 中 schedule patch size 从 coarse 到 fine，证明 iso-FLOPs 下性能更好。TST 是同样原理应用到 token embeddings：先学 coarse-grained 统计结构（topic、co-occurrence），再学 fine-grained ordering。

### 5.2 类似 Pre-pre-training on Formal Languages

Hu et al. 2025 ([ACL paper](https://aclanthology.org/2025.acl-long.478/)) 显示在 formal language 上 pre-pre-training 能 impart linguistic biases。TST 的 phase 1 类似：暴露给模型一个 simpler distribution（bag-of-words），它仍 share coarse statistical structure with natural language（local topic、co-occurrence）。

### 5.3 Embedding Geometry Regularization

第二个非互斥解释：embedding 空间中 averaging 隐式 regularize geometry。要让 random s-grams 在 summed 后仍 linearly separable（供 LM head 分类），embedding 必须保持某种结构。这类似 [Word2Vec](https://arxiv.org/abs/1301.3781) 中的 vector compositionality 假设。

### 5.4 Bag-of-Words as Future Summary

Output superposition 类似 [Mahajan et al. 2025 Future Summary Prediction](https://arxiv.org/abs/2510.14751)，但 architectural 区别显著：
- Mahajan: 附加 auxiliary head + BCE loss，pay 额外 parameters 和 loss term
- TST: 复用主 head，**只替换 target**，无额外 parameters

这与 [modded-nanogpt speedrun entry #178](https://github.com/KellerJordan/modded-nanogpt/pull/178) 的 next-bag-of-tokens loss 概念相似，区别是后者用 exponential weighting 和 smooth interpolation 而非 hard switch。

### 5.5 MTP 对比

[Gloeckle et al. MTP](https://arxiv.org/abs/2404.19737)（用于 [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)）用 k 个独立 head 预测 next k tokens。关键区别：

| 方面 | MTP | TST |
|------|-----|-----|
| Throughput | 与 baseline 相同 | s 倍提升 |
| Parameters | 额外 k 个 head | 0 |
| Inference | 可用于 speculative decoding | 不动 |
| Target | 精确 tokens | bag (无序) |
| 主要目标 | sample efficiency + inference speedup | training FLOPs efficiency |

TST 和 MTP 是**正交**的，可以组合使用。

## 6. 与 Patch-Level Training 的关系

Shao et al. [Patch-Level Training](https://openreview.net/forum?id=dDpB23VbVa) 独立提出几乎相同算法。作者承认 "conceptual convergence"。区别：

- **Shao 视角**: 减少 fixed dataset 上的 total FLOPs（同一数据集训练更快）
- **TST 视角**: constant per-step FLOPs 下提高 throughput（每步看更多 tokens）

实际实现上的差异：
1. TST 用 throughput hypothesis 框架，仔细保证 equal-FLOPs comparison
2. TST 用 MCE loss 并基于 entropy correction 论证
3. TST 扩展到 10B MoE / 2T tokens 规模（Shao 止于 2.7B / 360B）
4. 两者都发现：两阶段间保持相同 architecture（无额外 projection layer）是关键

## 7. Limitations 与 Future Work

1. **Compute-bound assumption**: TST 用更多数据换更好 loss。如果未来 LLM pre-training 进入 data-bound regime（[Kim et al. 2025](https://arxiv.org/abs/2509.14786)），output-only superposition 更有优势（不增加数据消耗）。

2. **Long context**: TST folding 让 effective context 长度变 s 倍，可能利于 long context 数据，未评测。

3. **Statistical significance**: 未做多 seed runs，scaling laws 未建立。

4. **Inference-time uses**: TST 训练的模型可能 informative latents 可用于 multi-token prediction/verification，留作 future work。

## 8. 代码实现要点

**Input folding（Listing 1）**:
```python
bs, seq = inputs.shape
inputs = inputs.reshape(bs, seq // s, s)
```

**Bag-of-Token embeddings（Listing 2）**:
- float32 累加保精度
- 最后除以 s 转回原 dtype

**MCE Loss（Listing 3）**:
- Causal padding with `-100` (ignore_index)
- 用 `torch.nn.functional.cross_entropy` 累加 s 次
- 除以 `w_total` 归一化

整个实现 < 30 行代码，drop-in，无需修改 parallelism、optimizer、tokenizer、data pipeline、model architecture。这是 TST 的工程价值所在。

## 9. 关键 Intuition 总结

构建对 TST 的 intuition，核心几点：

1. **Throughput 是 king**: 在 iso-FLOPs 下，处理更多 tokens → 更好 loss。这是 subword > byte 的根本原因，TST 把这个 principle 推到极致。

2. **Coarse-to-fine granularity schedule**: 先学 coarse 统计（topic, co-occurrence），再学 fine ordering。这类似 ViT patch schedule 和 pre-pre-training on formal language。是**通用 ingredient**，不限于 modality。

3. **两个正交机制叠加**: Input superposition（granularity + FLOPs-per-info）和 output superposition（target + gradient）独立 work，组合收益叠加。

4. **Representation alignment 是关键**: 两阶段必须共享 input embedding 和 output head。Randomization ablation 证明：破坏 alignment → 比不用 TST 还差。这解释了为何之前 compressive 方法（Bolmo 等）需要 adapter alignment phase。

5. **MCE loss 的 entropy correction**: Uniform target 有 entropy $\log s$，需减去才能让 loss 在 optimum 时为 0。但梯度上等价于 average of s 个独立 CE losses，可以复用编译优化 kernel。

6. **Mutual information power-law decay 决定 weighting**: 大 s 时 uniform weighting 不够好，因为远处 tokens 难预测。Power-law weighting $g(i) = 1/i$ 呼应 DCLM 数据中 MI 衰减规律 $d^{-1.25}$。

7. **Equal-FLOPs comparison 是诚实实验的基础**: 通过把序列长度扩 s 倍，确保每步 FLOPs 与 baseline 完全相同，这样 loss 比较直接反映 throughput 收益。Shao et al. 的 framing 混淆了 throughput 和 per-token FLOPs。

8. **Recovery phase 快速恢复**: 因为 representation 对齐，phase 2 只需"补上"精确 ordering 信息（高频信号），phase 1 已学到的 topic/concept（低频信号）直接 transfer。这类似 Fourier 分析中先学低频再学高频的 curriculum。

## 10. 相关工作链接

- [MTP - Gloeckle et al. ICML 2024](https://arxiv.org/abs/2404.19737)
- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)
- [Byte Latent Transformer - Pagnoni et al. ACL 2025](https://aclanthology.org/2025.acl-long.453/)
- [Bolmo - Minixhofer et al. 2025](https://arxiv.org/abs/2512.15586)
- [Patch-Level Training - Shao et al. ICLR 2025](https://openreview.net/forum?id=dDpB23VbVa)
- [SmolLM2 - Allal et al. 2025](https://arxiv.org/abs/2502.02737)
- [DCLM - Li et al. 2024](https://arxiv.org/abs/2406.19436)
- [Future Summary Prediction - Mahajan et al. 2025](https://arxiv.org/abs/2510.14751)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [modded-nanogpt PR #178](https://github.com/KellerJordan/modded-nanogpt/pull/178)
- [TorchTitan](https://openreview.net/forum?id=SFN6Wm7YBI)
- [PyTorch FSDP](https://dl.acm.org/doi/10.14778/3611540.3611569)
- [ViT Coarse-to-Fine Scheduling - Anagnostidis et al. ICML 2024](https://proceedings.mlr.press/v235/anagnostidis24a.html)
- [Pre-pre-training on Formal Languages - Hu et al. ACL 2025](https://aclanthology.org/2025.acl-long.478/)
- [Decoupling Subword Tokenization Benefits - Gigant et al. 2026](https://arxiv.org/abs/2604.27263)
- [Scaling Laws under Infinite Compute - Kim et al. 2025](https://arxiv.org/abs/2509.14786)
- [Predicting Order of Upcoming Tokens - Zuhri et al. 2026](https://arxiv.org/abs/2508.19228)
- [Next Concept Prediction - Liu et al. 2026](https://arxiv.org/abs/2602.08984)
- [Word2Vec - Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)
- [Entropy and Long-Range Correlations in Literary English - Ebeling & Pöschel 1994](https://iopscience.iop.org/article/10.1209/0295-5075/26/4/001)
- [WSD Learning Rate - Wen et al. 2024](https://arxiv.org/abs/2410.05192)
- [LM-Eval Harness](https://zenodo.org/records/10256836)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

TST 的优雅之处在于：用 < 30 行代码实现，不改任何 infrastructure，却能在 10B MoE 规模上 2.5x 训练效率。代价是 phase 1 期间 loss 是 non-sensical 的（bag 预测无 sampling 能力），必须经过 phase 2 才能成为正常 AR LM。这限制它只适用于 pre-training，post-training 阶段无法直接使用。
