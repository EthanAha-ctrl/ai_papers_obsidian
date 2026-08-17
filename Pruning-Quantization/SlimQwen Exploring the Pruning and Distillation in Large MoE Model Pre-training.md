---
source_pdf: SlimQwen Exploring the Pruning and Distillation in Large MoE Model Pre-training.pdf
paper_sha256: af4ec92ad91ed62a7683ffeec5d9687a768e250c97a938d076de261660bf6ad2
processed_at: '2026-08-12T07:50:51-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SlimQwen

兄弟，这篇 paper 其实就干了一件事：**把 80B 的 Qwen3-Next MoE 砍到 23B，还能保住约 90% 能力**。但真正有意思的不是结果，是中间踩出来的一堆"反直觉结论"——很多以前 dense model 上认为"显然对"的做法，在 MoE + pretraining scale 上要么不 work，要么差异小到不值得折腾。

我按"我当时读 paper 脑子里冒出来的问题"顺序讲。

---

## 一、先说架构：Qwen3-Next 到底长啥样

你把它想成一个 48 层的三明治：
- **12 层 full attention**（Gated Attention，精确 retrieval 用）
- **36 层 linear attention**（Gated DeltaNet，长上下文压缩用，类似 Mamba2 那一套）
- 每层后面挂一个 MoE 模块，里面有 **512 个 expert**，但每个 token 只激活 **11 个**（10 routed + 1 shared）

总参数 80B，激活 3B——这是 MoE 的"大规模参数 + 稀疏激活"经典配方。Reference: Qwen3-Next blog https://qwen.ai/blog/qwen3-next

**Shared expert 是 always-on**，gate 用 sigmoid（不是 softmax）：

$$z_s(x) = \sigma(x w_{\text{sh}})$$

这个设计很关键——shared expert 像个"global baseline knowledge"，routed experts 在它基础上做 specialization delta。压缩时 shared expert 默认保留不动（paper 没明说，但 Algorithm 1 只处理 routed experts）。

---

## 二、剪枝三个维度，每个都有"反直觉"

### 1. Depth：直接砍最后 12 层，别整花活

$$\mathcal{L}_{\text{keep}} = \{1, \dots, L-N\}$$

paper 砍掉最后 25%（12 层：3 full + 9 linear）。**没有任何 importance metric**，就是砍尾巴。

反直觉在哪？ShortGPT (https://arxiv.org/abs/2403.03853) 那套用 adjacent-layer cosine similarity 找"最冗余中间层"的方法，在 pretraining scale + KD 设定下被吊打：

| 方法 | one-shot MMLU | 120B KD 后 MMLU |
|---|---|---|
| Activation similarity (ShortGPT 风格) | 41.95 | 69.57 |
| Last-layer pruning | 73.86 | 73.02 |

差了 30 多个点！

**Intuition**：深网络前面几层奠定 "what features are present"，后面几层更多是 amplification / refinement。砍尾巴相当于移除一个 lossy decoding stage，KD 能教 student 用更短网络完成最后一步。砍中间层等于把 hierarchical representation 的关键 transformation 阶段挖掉，破坏严重。

参考 "The Curse of Depth" (https://arxiv.org/abs/2502.05795) 也观察到深网络深层 magnitude 大但相对独立——这跟"剪尾巴 work"是一致的。

### 2. Width：用 RMSNorm 的 activation magnitude 当 proxy

$$I_{\text{norm}}^{(k)} = \left[\frac{\sum_{i=0}^{L} \text{Mean}(|\text{RMSNorm}(X)|)}{L}\right]_k$$

- $k$: hidden dim index（1 到 $d$）
- $i$: layer index
- $\text{Mean}(|\cdot|)$: batch + sequence 维度上求绝对值均值

2048 → 1536，砍 25%。保留 top-1536 个最重要的 hidden dim。

**Intuition**：RMSNorm 的 $\gamma$ 是 model 自己学到的"哪个维度重要"的 implicit indicator，activation magnitude 是 empirical confirmation。两者高度相关，直接挑 magnitude 大的保留就行。SliceGPT (https://arxiv.org/abs/2401.15024) 那套 PCA 更精确但计算贵，这里用轻量 proxy 够用。

### 3. Expert：核心创新 — Partial-Preservation Merging

先说三个 importance metric：

**Frequency**（被激活次数）：
$$I_i^{\text{Freq}} = \mathbb{E}_{x \sim \mathcal{C}}[\mathbb{I}[i \in \mathcal{A}(x)]]$$

**Soft Logits**（激活次数 × routing weight）：
$$I_i^{\text{Soft}} = \mathbb{E}_{x \sim \mathcal{C}}\left[\frac{\mathbb{I}[i \in \mathcal{A}(x)] \cdot z_i(x)}{\sum_{j \in \mathcal{A}(x)} z_j(x)}\right]$$

**REAP** (https://arxiv.org/abs/2510.13999)（激活次数 × routing weight × output magnitude）：
$$I_i^{\text{REAP}} = \frac{1}{|\mathcal{X}_i|} \sum_{x \in \mathcal{X}_i} z_i(x) \|E_i(x)\|_2$$

- $\mathcal{A}(x) = \text{TopK}(z(x), k)$: 被 token $x$ 激活的 expert 集合
- $z_i(x)$: expert $i$ 对 token $x$ 的 routing weight
- $E_i(x)$: expert $i$ 对 token $x$ 的输出
- $\mathcal{X}_i$: expert $i$ 被激活过的所有 token 集合

**核心发现（Table 2）**：400B token continual pretraining 之后，**这三个 metric 差异 marginal**，没有一个 dominate。这是个重要的 negative result——以前大家争论 frequency vs REAP 谁更好，在 pretraining scale 上其实都被 continual training "洗"到相近的 basin 了。

但 paper 提出 partial-preservation merging：

$$\tilde{E}_i = \frac{I_i}{I_i + I_{m(i)}} E_i + \frac{I_{m(i)}}{I_i + I_{m(i)}} E_{m(i)}$$

- $E_i$: 保留的 merge base expert
- $E_{m(i)}$: 被合并进来的 discarded expert，$m(i) = \arg\max_{j \in \mathcal{S}_{\text{merge}}} \text{CosineSim}(i, j)$
- $I_i, I_{m(i)}$: 用作加权系数

**策略**：512 experts → 256 experts
1. 保留 importance top-128 个**完全不动**（preserve）
2. 从剩下 384 里再选 128 个作为 merge base
3. 把剩下 256 个 discarded expert 按余弦相似度分配给最近的 base，按 importance 加权 merge

**Intuition**——这是 specialization preservation vs knowledge consolidation 的 trade-off：
- **纯 prune**：丢掉 discarded expert 的所有 knowledge，剩 256 个 specialization 完好但"knowledge ceiling"降低
- **纯 merge**（全 256 都从 merge 来）：knowledge 保留但 specialization 被 homogenize，每个 expert 都成"平均"，routing 容易 collapse
- **Half preserve + half merge**：top specialists 完整保留作为"anchor knowledge"；merge bases 吸收 discarded experts 的 complementary capability

Table 2 验证：partial preservation 在 MMLU-Pro (+1.43)、GSM8K (+3.10) 上有 consistent gain。

为什么是 1/2？作者承认是 ad-hoc。直觉：太少 preserve 让 specialization 丢失，太多 preserve 让 merge 空间不足。一半是 symmetric 稳健点。可以试 1/3 或 2/3 看曲线，paper 没做（Limitation 里提了）。

---

## 三、Distillation：四个 loss term 的组合拳

最终 loss：

$$\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{KD}} + \beta\left[(1-\lambda)\mathcal{L}_{\text{MTP-LM}} + \lambda\mathcal{L}_{\text{MTP-KD}}\right]$$

- $\mathcal{L}_{\text{LM}}$: 标准 next-token cross-entropy 到 ground truth
- $\mathcal{L}_{\text{KD}}$: KL divergence 到 teacher soft distribution
- $\mathcal{L}_{\text{MTP-LM}}$: MTP module 预测 future token 到 ground truth
- $\mathcal{L}_{\text{MTP-KD}}$: MTP module 预测到 teacher MTP soft distribution
- $\lambda$: linear decay 1.0 → 0.75（前期 KD 主导，后期 LM 增加）
- $\beta$: cosine decay 0.3 → 0.1（前期 MTP 信号强，后期 backbone 主导）

### 关键发现 1：KD + LM > 纯 KD

Table 3：
- NTP KD alone: MMLU 74.16, MMLU-Pro 50.97
- NTP KD + LM Loss: MMLU 74.93 (+0.77), MMLU-Pro 51.44 (+0.47)

**Intuition**：纯 KD 让 student 模仿 teacher 的 distribution，但 teacher 对 long-tail fact 给 low-confidence 时，KD 信号弱。LM loss 提供 ground truth 强信号，相当于"事实锚定"。

但 LM loss 在 GSM8K 上反而降（84.27→82.98）——硬目标让 student over-fit token pattern 而忽略 teacher 的 reasoning distribution。所以 $\lambda$ linear decay 1.0→0.75 是个"先模仿 teacher 整体分布、再巩固 ground truth"的 curriculum。

### 关键发现 2：MTP KD — 这是 paper 最有意思的点

MTP module 架构（Eq. 9）：
$$h_i^{\prime k} = M_k\left[\text{RMSNorm}(h_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))\right]$$

- $h_i^{k-1}$: 第 $i$ 个 token 在 depth $k-1$ 的 hidden（$k=1$ 时是 backbone 输出）
- $\text{Emb}(t_{i+k})$: 未来第 $i+k$ 个 token 的 embedding
- $M_k \in \mathbb{R}^{d \times 2d}$: 把 $2d$ 压回 $d$
- 然后过一个 transformer block 产生 $h^k$，再用共享 output head 预测 $t_{i+k}$

Table 3 加 MTP KD 后 MMLU 74.16 → 75.13, MMLU-Pro 50.97 → 51.94, C-Eval 80.00 → 80.82。consistent gain。

**Intuition 1 — Representation quality**：standard NTP KD 只在 logit 层面模仿 teacher，hidden representation 训练信号"间接"通过 logit 误差传回。MTP module 强制 student backbone hidden $h^0$ 不只能预测 $t_{i+1}$，还要"含足够信息"让浅 MTP module 预测 $t_{i+2}$。这相当于显式监督 student 学习 teacher 的"前瞻性 representation"—— teacher 的 hidden 天然编码 multi-step future（因为 teacher 也是 next-token 训练的，representation 有 redundant capacity）。用 MTP KD 显式监督这部分，让 student 在更小容量下尽量保留这种 "compressed future"。

**Intuition 2 — Speculative decoding 友好**：MTP module 本身就是 speculative decoding 的天然 draft model（共享 embedding + 简单 transformer block，验证用 backbone）。Table 4 是最直接的证据：

GSM8K pretrain 上：
| Loss | acc_0 | acc_1 | acc_2 | acc_3 | acc_4 |
|---|---|---|---|---|---|
| MTP Loss | 95.90 | 57.62 | 23.64 | 8.02 | 2.37 |
| MTP KD | 95.50 | 75.18 | 45.67 | 22.43 | 10.37 |

- acc_0: 单 token 接受率（基本不变，这是 backbone 自身能力）
- acc_1: 双 token +17.56 个百分点
- acc_2: 三 token +22.03 个百分点（接近翻倍！）
- acc_3: +14.41
- acc_4: +8.00

**远 token 增益相对更大**。因为 MTP Loss 只让 draft match ground truth，但 draft 与 verifier (backbone) 之间可能有 distribution drift。MTP KD 直接让 draft 模仿 verifier (teacher) distribution，两者 alignment 大幅提升，多 token 同时接受概率几何级数放大。

Speculative decoding 的 wall-clock speedup 几乎线性正比于平均接受 token 数——acc_2 翻倍意味着长 generation speedup 显著提升。这点对实际部署非常实用。

---

## 四、Progressive Pruning > One-shot

三种 schedule（总 token 都是 400B）：
- **Depth-first**: Stage 1 砍 6 层 + 保持 width，Stage 2 砍剩 6 层 + 全部 width
- **Width-first**: Stage 1 砍 50% width + 保持 depth，Stage 2 砍剩 width + 全部 depth
- **Joint**: Stage 1 同时砍 50% depth + 50% width，Stage 2 砍剩 50% + 50%

Table 5：

| Method | MMLU | MMLU-Redux | BBH | GSM8K |
|---|---|---|---|---|
| One-stage | 75.86 | 75.41 | 73.97 | 85.22 |
| Joint | 76.30 | 76.93 | 71.40 | 86.05 |
| Width-first | 77.14 | 77.07 | 75.22 | 84.00 |
| **Depth-first (SlimQwen)** | **77.39** | **78.01** | 70.70 | 85.82 |

**所有 progressive 都比 one-shot 好**。Depth-first 在 MMLU 类最优，Width-first 在 BBH 最优。

**Intuition**：one-shot 4× 容量跳跃，optimizer landscape 突变巨大，continual pretraining 同时要"修复架构损伤"和"学新数据"。Progressive 把 4× 拆成 2× + 2×，每次更 tractable。类似 CV 里的 progressive resizing。

为什么 Depth-first 最优？直觉：**depth reduction 对 representation 影响更"局部"**（剪末尾层只影响后段 computation），先 depth 砍半让 student 学会用更短网络做相同任务；后 width 砍半是在"已经会用短网络"基础上做 representation compression。反过来 Width-first 先砍 width 让每层 representation 容量降低，再砍 depth 时已经"瘦弱"的 layer 难以维持 hierarchical abstraction。

Appendix Table 9 试了 3-stage (20B+20B+360B)，发现相比 2-stage 没显著增益——2-stage 已经够 capture progressive 好处，再分只增加 schedule 复杂度。

---

## 五、Pruning vs. From Scratch — 核心动机

Table 1：

| Init | Training | Avg. |
|---|---|---|
| Random Init + KD | 120B | 61.66 |
| Pruned + LM Loss | 120B | 69.96 |
| **Pruned + KD** | **120B** | **73.45** |
| Qwen3-Next-80A3B (teacher) | — | 82.68 |

**Pruned 比 random init 高 11.79 个点**，恢复 86.5% teacher 性能。

**Intuition**：pretrained MoE 里有大量"可复用 sub-circuits"。尽管砍 50% experts、25% depth、25% width，剩下的 weights 仍保留 task-critical representation 路径。这跟 lottery ticket hypothesis (https://arxiv.org/abs/1803.03635) 在 pretraining scale 上的延伸一致——pruning 后的 subnetwork 是个"winning ticket"。

更深一点：MoE 天然有 redundancy——512 experts 中很多在 calibration set 上 frequency 极低（甚至接近 0），它们对 task performance 贡献小但占用参数。Pruning 这些 "dead-ish" experts 几乎无损。Random init 在 120B token 内根本"学不动"这么多知识（Figure 2 训练曲线证实）。

---

## 六、Efficiency（Table 11）

- Peak memory: 156.56 GB → 43.30 GB（3.6× 降低，**可单 80GB GPU 部署**）
- HF backend prefill latency: 0.99s → 0.44s (2.25× 加速)
- HF decoding throughput: 4.05 → 6.55 tok/s (1.62×)
- vLLM decoding throughput: 142.58 → 210.87 tok/s (1.48×)

单 GPU 部署这点非常实用——去掉 TP/PP 通信开销，小模型 inference 实际 wall-clock 收益更大。

---

## 七、Paper 没说但值得琢磨的点

1. **Expert importance 的 calibration set bias**：1024 样本从训练数据采样。如果训练数据分布与下游 benchmark 不同，importance 估计有偏。Paper 没做 sensitivity analysis on calibration set size/diversity。用更领域多样的 calibration（mix 代码 + 数学 + 多语言），importance ranking 会不会变？open question。

2. **Shared expert 处理**：Algorithm 1 只针对 routed experts。Shared expert 承载"common knowledge"，把它 merge 进 routed experts 可能损失基础能力。这点值得 ablation。

3. **Full vs Linear attention 的 differential pruning**：Qwen3-Next 是 hybrid 12+36。Pruning 砍 3 full + 9 linear 按原比例砍。如果按"full attention 更重要"砍 0 full + 12 linear，或反之，performance 如何？Full attention 对 retrieval 关键（MMLU、EvalPlus），linear attention 对 long-context 关键——砍不同比例会在不同 benchmark 上 trade。open dimension。

4. **MTP depth $D=1$ 局限**：paper 只用 $D=1$。如果 $D=2, 3$，训练成本和收益曲线如何？MTP module 参数会变多，可能稀释 compression 收益。但 speculative decoding 的 speedup 可能更大（acc_2/acc_3 进一步提升）。值得探索。

5. **Progressive pruning 的 LR schedule**：Appendix A.3 说"second stage starts from the learning rate reached at the final step of the first stage"。如果 Stage 1 已经 decay 到很低，Stage 2 的"余热"够不够 recover 新剪的 capacity？是否需要 Stage 2 重新 warmup？paper 没消融。

6. **Expert merging 后的 router rebalancing**：Table 2 显示 9 种组合差异都在 1-2 个点内。但没分析"merge 后某个 base expert 是否变成新的 dead expert"（被 router 几乎不激活）。这关系到是否需要 post-merge retraining rebalance router。

7. **Knowledge recovery 上下界**：86.5% 性能恢复是"好"，但理论上的上限是多少？如果用更激进 distillation（比如 hidden state KD 不只 logit KD），能不能逼近 95%？paper 只用 logit-level KD，representation-level KD（比如 MiniLM 风格，https://arxiv.org/abs/2006.07747）可能有用。

---

## 八、Paper 在 literature 中的位置

填补几个 gap：

- **Minitron (https://arxiv.org/abs/2407.14679)**：dense MNMG 15B 上做 pruning + KD。SlimQwen 把结论 extend 到 MoE + 80B→23B。
- **ShearedLLaMA (https://arxiv.org/abs/2310.06694)**：dense LLaMA 上 progressive pruning。SlimQwen 借鉴 progressive 但用更简单两阶段。
- **ShortGPT (https://arxiv.org/abs/2403.03853)**：layer similarity 找冗余层。SlimQwen 实测在 pretraining-scale KD 下不如直接剪最后几层。
- **SlimMoE (https://arxiv.org/abs/2506.18349)**：MoE 内部 expert intermediate dim pruning。SlimQwen 砍 expert 数量而非 expert 内部 dim。
- **REAP (https://arxiv.org/abs/2510.13999)**：one-shot MoE compression metric 对比。SlimQwen 拿到 pretraining scale re-evaluate，发现差异不大——一个重要的 negative result 校正。
- **Multi-token prediction (https://arxiv.org/abs/2408.01037)**：原始 MTP 用 ground truth label 训 backbone。SlimQwen 第一次把 MTP 用作 distillation target，并验证 speculative decoding 加速。

---

## 九、一句话总结每个 finding 的 intuition

1. **Pruned > Scratch**: pretrained MoE weights 里 encode 了 task-critical sub-circuits，4× 压缩后仍保留 86% 性能，因为 redundancy 主要在 dead-ish experts 和深层 refinement layers。
2. **Expert metric 不重要**: 400B continual training 是个"等价 basin attractor"——不同 one-shot 压缩起点都被拉到相近 final basin，选简单 Frequency 即可。
3. **Partial preservation**: 一半 pristine specialist + 一半 merge，让 model 既保留 specialization anchor 又吸收 discarded knowledge，避免 homogenize。
4. **LM + KD > 纯 KD**: KD 模仿 teacher distribution 但 long-tail fact 信号弱，LM 锚定 ground truth，两者互补。$\lambda$ linear decay 实现先模仿后巩固。
5. **MTP KD**: 让 backbone hidden 显式编码 "multi-step future"，提升 representation quality，并天然 align speculative decoding 的 draft 与 verifier，远 token 接受率几何级数提升。
6. **Progressive > One-shot**: 4× 跳跃拆成 2× + 2×，优化 landscape 更平滑。Depth-first 最优因为 depth reduction 的"局部性"更强。
7. **Last-layer > Middle-layer pruning**: 深层是 amplification/refinement，剪掉 KD 能教 student 用短网络做最后一步；中间层是 hierarchical representation 关键 transformation，剪掉破坏严重。

---

## 十、最后一句话

这篇 paper 最有价值的地方**不是新方法**——partial-preservation merging 和 MTP KD 都是很简单的 idea。**价值在 pretraining scale 上的一组 robust empirical findings**：哪些 fancy 方法其实在 400B token 训练后差异 marginal，哪些简单做法反而 work。对想压缩自己 MoE 模型的团队，几乎可以照搬这个 recipe：

1. 砍最后 25% 层（别找冗余中间层）
2. 用 RMSNorm activation magnitude 砍 25% width
3. 用 Frequency 排 expert，保留 top-1/2 不动，剩 1/2 用 importance-weighted merging
4. Loss = LM + KD + MTP-LM + MTP-KD，$\lambda$ linear decay 1→0.75，$\beta$ cosine decay 0.3→0.1
5. Progressive depth-first: Stage 1 砍 50% depth + 40B token，Stage 2 砍剩 50% depth + 全部 width + 360B token

总训练成本 400B token，最终得到 23A2B 模型，性能保留 86.5%，单 GPU 部署，decoding throughput 提升 1.48×。Practical + reproducible。

参考链接汇总：
- Qwen3-Next: https://qwen.ai/blog/qwen3-next
- Multi-token prediction: https://arxiv.org/abs/2408.01037
- Minitron: https://arxiv.org/abs/2407.14679
- ShearedLLaMA: https://arxiv.org/abs/2310.06694
- ShortGPT: https://arxiv.org/abs/2403.03853
- The Curse of Depth: https://arxiv.org/abs/2502.05795
- REAP: https://arxiv.org/abs/2510.13999
- SliceGPT: https://arxiv.org/abs/2401.15024
- SlimMoE: https://arxiv.org/abs/2506.18349
- Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
- Speculative decoding survey: https://aclanthology.org/2024.findings-acl.456
- MiniLM (representation KD): https://arxiv.org/abs/2006.07747

---

# SlimQwen 深度解析：MoE 大模型 pretraining-scale 压缩

这篇 paper 是 Qwen 团队的一个非常 systematic 的 empirical study，把 Qwen3-Next-80A3B (80B 总参数，3B 激活) 压缩到 SlimQwen-23A2B (23B 总参数，2B 激活)，约 4× compression，在 MMLU/GSM8K/EvalPlus 等 benchmark 上保留约 90% 性能。这篇工作的重要性在于：**它把以前主要在 dense model / 一次性剪枝 / 小规模实验中验证过的 pruning recipe，第一次在 pretraining scale 上、在 hybrid-attention MoE 架构上系统性地检验**，结论不少反直觉。

参考链接：
- Qwen3-Next tech report: https://qwen.ai/blog/qwen3-next (Team, 2025b)
- Multi-token prediction (Gloeckle et al., 2024): https://arxiv.org/abs/2408.01037
- Minitron (Muralidharan et al., 2024): https://arxiv.org/abs/2407.14679
- Sheared LLaMA (Xia et al., 2024b): https://arxiv.org/abs/2310.06694
- ShortGPT (Men et al., 2024): https://arxiv.org/abs/2403.03853
- The Curse of Depth (Sun et al., 2026): https://arxiv.org/abs/2502.05795
- REAP (Lasby et al., 2025a): https://arxiv.org/abs/2510.13999
- SliceGPT (Ashkboos et al., 2024): https://arxiv.org/abs/2401.15024
- Speculative decoding survey (Xia et al., 2024a): https://aclanthology.org/2024.findings-acl.456

---

## 1. 架构背景：Qwen3-Next 是个 hybrid MoE

要理解这篇 paper 的 pruning 维度，先看 teacher 的架构（Table 6）：

| 组件 | 配置 |
|---|---|
| Total layers $L$ | 48 (12 full attention + 36 linear attention) |
| Hidden size $d_{\text{model}}$ | 2048 |
| Total experts $N$ | 512 (per MoE layer) |
| Routed experts per token $n_{\text{routed}}$ | 10 |
| Shared experts $n_{\text{shared}}$ | 1 |
| FFN intermediate $d_{\text{ff}}$ | 512 (per expert) |
| Full attention | Gated Attention (Qiu et al., 2025b)，16 query head / 2 KV head, head_dim=256 |
| Linear attention | Gated DeltaNet (Yang et al., 2025b)，32 v_head, 16 qk_head, d_vhead=d_qkhead=128 |
| MTP modules | 1 (depth $D=1$) |
| Total / Activated params | 80B / 3.8B |

关键直觉：**Qwen3-Next 用 12 个 Gated Attention 处理 "需要精确长程 retrieval" 的内容，36 个 Gated DeltaNet 处理 "需要 fast recurrence + 长上下文压缩" 的内容**。MoE 模块里 512 个 experts 但每 token 只激活 11 个（10 routed + 1 shared）—— 这就是为什么 80B 总参数但只有 3B 激活的根本原因。

公式 (1)–(3) 是基础定义。值得注意：

**RMSNorm (Eq. 3)**：
$$\text{RMSNorm}(X)_i = \frac{X_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} X_{ij}^2 + \epsilon}} \cdot \gamma_i$$
- $X \in \mathbb{R}^{n \times d}$: $n$ 个 token, $d$ 维 hidden
- $i$: token index（行）
- $j$: hidden dim index（列）
- $\gamma \in \mathbb{R}^{1 \times d}$: per-dim 可学习 scale
- $\epsilon$: 数值稳定项

RMSNorm 没有 mean subtraction，比 LayerNorm 更便宜，是 width pruning 中 importance estimation 的核心载体。

**MoE 路由 (Eq. 2)**：
$$\text{MoE}(x) = \sum_{e=1}^{n_{\text{routed}}} z_e(x) \text{Expert}_e(x) + \sum_{s=1}^{n_{\text{shared}}} z_s(x) \text{Expert}_s(x)$$

- $z_e(x) = \text{softmaxTopK}(xW^G, k)$: routed gate，top-k 后归一化
- $z_s(x) = \sigma(x w_{\text{sh}})$: shared gate，**sigmoid**（不是 softmax），因此 shared expert 是 always-on，强度可独立调节

Shared expert 用 sigmoid gate 是 Qwen3-Next 一个有意思的设计 — 让 shared expert 像一个"global knowledge baseline"，routed experts 在此基础上做 specialization delta。压缩时这一点很重要（后续 expert pruning 时 shared expert 是不动还是动的 trade-off 在 paper 里没深入，但默认保留）。

---

## 2. 三个压缩维度（Section 3.2）

### 2.1 Depth Pruning：直接砍最后 25%

公式 (4)：
$$\mathcal{L}_{\text{keep}} = \{1, \dots, L-N\}, \quad \tilde{L} = L-N$$

paper 里 $L=48, N=12$，砍掉最后 12 层（3 full + 9 linear）。**没有任何 fancy 度量**。这点反直觉 — 之前 ShortGPT (Men et al., 2024) 用 adjacent-layer cosine similarity 找"最冗余"的中间层，但 paper 在 Appendix Table 8 的对比显示：

| 方法 | MMLU (one-shot) | MMLU (120B KD后) |
|---|---|---|
| 15A2B Teacher | 75.62 | — |
| Activation Similarity (ShortGPT-style) | 41.95 | 69.57 |
| Last Layer Pruning | 73.86 | 73.02 |

中间层剪枝 one-shot 性能从 75.62 暴跌到 41.95（GSM8K 从 82.41 跌到 11.22），即便 KD 恢复后也仍落后于直接剪最后几层。

**Intuition**：这与 "The Curse of Depth" (Sun et al., 2026, https://arxiv.org/abs/2502.05795) 的观察一致 — 大模型深层反而输出 magnitude 大、input-output residual block 之间相似度低，但**任务相关知识的"载体"在前面的层已经形成了**，深层更多是 amplification / refinement，剪掉后 KD 容易恢复。而中间层往往是 hierarchical representation 的关键 transformation，剪掉破坏严重。

更深的 intuition：在 residual stream 视角下，深网络是 $h_L = h_0 + \sum_\ell f_\ell(h_{\ell-1})$，前几层奠定 "what features are present"，后几层更多是 "how to combine / route"。剪最后几层相当于 lossy decoding stage 移除，KD 能教 student 用更少的 layers 完成最后一步。

### 2.2 Width Pruning：基于 RMSNorm 激活的 mean-absolute

公式 (5)：
$$I_{\text{norm}}^{(k)} = \left[\frac{\sum_{i=0}^{L} \text{Mean}(|\text{RMSNorm}(X)|)}{L}\right]_k, \quad k = 1, \ldots, d$$

- $k$: hidden dim index
- $i$: layer index
- $\text{Mean}(|\cdot|)$: 对 batch + sequence 维度求绝对值平均（公式形式 $\frac{1}{Bn}\sum_{b,t}|Z_{b,t,:}|$）
- $I_{\text{norm}}^{(k)}$: hidden dim $k$ 的重要性

直觉：RMSNorm 的 $\gamma$ scale 是 model 自己学到的"哪个维度重要"的 implicit indicator，而 Mean(|activation|) 是 empirical confirmation — 两者高度相关。直接保留 top-$d_t$（这里 $d_t = 1536$，从 2048 砍 25%）维度即可。

这跟 SliceGPT (Ashkboos et al., 2024, https://arxiv.org/abs/2401.15024) 的思路一脉相承 — SliceGPT 用 covariance 的 PCA 找主轴，这里用 RMSNorm 的 activation magnitude 当 proxy，更简单。但 SliceGPT 的 PCA 是 numerical（计算密集），SlimQwen 的方法更 lightweight。

### 2.3 Expert Compression：核心创新点 — Partial-Preservation Merging

paper 比较三种 importance metric (Eq. 6–7)：

**Frequency-based**：
$$I_i^{\text{Freq}} = \mathbb{E}_{x \sim \mathcal{C}}[\mathbb{I}[i \in \mathcal{A}(x)]]$$
- $\mathcal{A}(x) = \text{TopK}(z(x), k)$: 被激活的 expert 集合
- $\mathbb{I}[\cdot]$: 指示函数
- 直觉：expert $i$ 被激活的频率（路由次数 / 总 token 数）

**Soft Logits**：
$$I_i^{\text{Soft}} = \mathbb{E}_{x \sim \mathcal{C}}\left[\frac{\mathbb{I}[i \in \mathcal{A}(x)] \cdot z_i(x)}{\sum_{j \in \mathcal{A}(x)} z_j(x)}\right]$$
- 在 frequency 基础上用 routing logit $z_i(x)$ 加权（归一化后）
- 直觉：不仅看激活次数，还看激活时的重要性权重

**REAP** (Lasby et al., 2025a, https://arxiv.org/abs/2510.13999)：
$$I_i^{\text{REAP}} = \frac{1}{|\mathcal{X}_i|} \sum_{x \in \mathcal{X}_i} z_i(x) \|E_i(x)\|_2$$
- $\mathcal{X}_i$: expert $i$ 被激活的 token 集合
- $z_i(x)$: routing weight
- $\|E_i(x)\|_2$: expert output 的 L2 norm
- 直觉：激活时输出 magnitude 大的 expert 更"实质地贡献"了 representation

**核心 empirical 发现（Table 2）**：在 400B-token continual pretraining 之后，**这三种 metric 之间差异 marginal**，没有一个 dominate 所有 benchmark。这是 paper 一个非常重要的"negative result" — 在 pretraining scale 上，不管你 expert 怎么压，最终都被 continual training "洗"到一个相近的 basin。

但 paper 提出 **partial-preservation merging** 作为改进点：

公式 (8)：
$$\tilde{E}_i = \frac{I_i}{I_i + I_{m(i)}} E_i + \frac{I_{m(i)}}{I_i + I_{m(i)}} E_{m(i)}$$

- $E_i$: 保留的 expert (merge base)
- $E_{m(i)}$: 被合并进来的 discarded expert，$m(i) = \arg\max_{j \in \mathcal{S}_{\text{merge}}} \text{CosineSim}(i, j)$ 是与 $i$ 最相似的被丢弃 expert
- $I_i, I_{m(i)}$: importance scores，作为加权系数

**Partial-preservation 策略（Algorithm 1）**：
1. 保留 importance 排名 top-$\lfloor \tilde{N}/2 \rfloor$ 的 experts **完全不动**（preserve）
2. 从剩余 experts 中再选 $\tilde{N}/2$ 个作为 merge base（base）
3. 把剩下的 discarded experts 按余弦相似度分配给最近的 base，按 importance 加权 merge

直觉解释 — 这是 **specialization preservation vs. knowledge consolidation** 的 trade-off：
- 纯 prune：丢掉 discarded experts 的所有 knowledge，剩 $\tilde{N}$ 个 specialization 完好但 "knowledge ceiling" 降低
- 纯 merge（全部 $\tilde{N}$ 都从 merge 来）：knowledge 保留但 specialization 被 homogenize，每个 expert 都成了"平均"，routing collapse 风险高
- Partial preservation（保留一半 + merge 一半）：top specialists 完整保留，构成 model 的 "anchor knowledge"；merge bases 吸收 discarded experts 的 complementary capability

Table 2 验证：partial preservation 在 MMLU (+0.23)、MMLU-Pro (+1.43)、GSM8K (+3.10) 等 benchmark 上**一致**有 gain（虽然幅度小）。

为什么是一半？作者在 Limitation 里承认这是个 ad-hoc 选择。但直觉是：太少 preserve 会让 specialization 丢失，太多 preserve 让 merge 的"吸收空间"不足。一半是个 symmetric 的稳健点。可以试 $1/3$ preserve + $2/3$ merge 或者 $2/3$ preserve + $1/3$ merge 看曲线，paper 没做。

---

## 3. Distillation Pretraining（Section 3.3）

### 3.1 四个 loss term 的组合

最终 loss（Eq. 12）：
$$\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{KD}} + \beta\left[(1-\lambda)\mathcal{L}_{\text{MTP-LM}} + \lambda\mathcal{L}_{\text{MTP-KD}}\right]$$

- $\mathcal{L}_{\text{LM}}$: 标准 next-token cross-entropy 到 ground truth one-hot
- $\mathcal{L}_{\text{KD}}$: KL divergence 到 teacher 的 next-token soft distribution
- $\mathcal{L}_{\text{MTP-LM}}$: MTP module 预测第 $k$ 个 future token 到 ground truth
- $\mathcal{L}_{\text{MTP-KD}}$: MTP module 预测到 teacher 的 MTP soft distribution
- $\lambda$: KD vs LM 的权重，linear decay 1.0 → 0.75（**前期偏重 KD 模仿，后期增加 LM grounding**）
- $\beta$: backbone vs MTP 的权重，cosine decay 0.3 → 0.1（前期 MTP 辅助信号强，后期让 backbone 主导）

### 3.2 MTP module 架构（Eq. 9–11）

公式 (9) — depth $k$ 的输入构造：
$$h_i^{\prime k} = M_k\left[\text{RMSNorm}(h_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))\right]$$

- $h_i^{k-1} \in \mathbb{R}^d$: 第 $i$ 个 token 在第 $k-1$ 个 MTP depth 的 hidden
- $h_i^0$: backbone 输出（$k=1$ 时）
- $\text{Emb}(t_{i+k}) \in \mathbb{R}^d$: 第 $i+k$ 个 token 的 embedding
- $[\cdot; \cdot]$: concatenation，维度 $2d$
- $M_k \in \mathbb{R}^{d \times 2d}$: projection 把 $2d$ 压回 $d$

直觉：MTP module 在每个 depth $k$ 接收 (a) 上一层 MTP 的 representation 和 (b) 未来第 $k$ 个 token 的 embedding（作为 "hint"），通过一个 transformer block $\text{TRM}_k$ 产生 $h^k$，再用共享 output head 预测 $t_{i+k}$。

公式 (10) — MTP LM loss：
$$\mathcal{L}_{\text{MTP-LM}} = \frac{1}{D}\sum_{k=1}^{D}\left(-\frac{1}{T-k}\sum_{i=1}^{T-k} \log p_{i+k}^k[t_{i+k}]\right)$$
- $D$: MTP depth（这里 $D=1$）
- $T$: 序列长度
- $p_{i+k}^k[v]$: MTP module 在 depth $k$ 对位置 $i+k$ 预测的词 $v$ 的概率
- $t_{i+k}$: 真实第 $i+k$ 个 token

公式 (11) — MTP KD loss：
$$\mathcal{L}_{\text{MTP-KD}} = -\frac{1}{D}\sum_{k=1}^{D}\left(\frac{1}{T-k}\sum_{i=1}^{T-k}\sum_{v=1}^{V} q_{i+k}[v] \log p_{i+k}^k[v]\right)$$
- $q_{i+k}$: teacher 在位置 $i+k$ 的 soft target distribution
- $V$: 词表大小
- 这本质是 cross-entropy 到 teacher 的 soft label

### 3.3 为什么 MTP KD 比 standard NTP KD 更好

Table 3 显示加 MTP KD 后 MMLU 74.16 → 75.13 (+0.97)、MMLU-Pro 50.97 → 51.94 (+0.97)、C-Eval 80.00 → 80.82、CMMLU 80.24 → 80.64。一致性增益。

**Intuition 1 — Representation quality**：standard NTP KD 只让 student backbone 在最终 logit 层面模仿 teacher，hidden representation 的训练信号"间接"通过 logit 误差传回。MTP module 强制 student 的 backbone hidden $h^0$ 不只能预测 $t_{i+1}$，还要"含足够信息"让一个浅 MTP module 预测 $t_{i+2}$。这相当于让 student 学习 teacher 的"前瞻性 representation"—— teacher 的 hidden 自然而然编码了 multi-step future (因为 teacher 也是 next-token 训练的，但 representation 有 redundant capacity)；用 MTP KD 显式监督这部分，让 student 在更小容量下尽量保留这种 "compressed future"。

**Intuition 2 — Speculative decoding 友好**：Table 4 是这点最直接的证据。MTP module 本身就是 speculative decoding 的天然 draft model（共享 embedding + 简单 transformer block，验证用 backbone）。看 GSM8K pretrain 数据：

| Loss | acc_0 | acc_1 | acc_2 | acc_3 | acc_4 |
|---|---|---|---|---|---|
| MTP Loss | 95.90 | 57.62 | 23.64 | 8.02 | 2.37 |
| MTP KD | 95.50 | 75.18 | 45.67 | 22.43 | 10.37 |

- acc_0: 单 token 接受率（基本不变，因为这是 backbone 自身能力）
- acc_1: 双 token 接受率 +17.56 个百分点
- acc_2: 三 token 接受率 +22.03 个百分点（接近翻倍！）
- acc_3: +14.41
- acc_4: +8.00

**远 token 增益相对更大**。这是因为 MTP Loss 只是让 draft 模型 match ground truth，但 draft 与 verifier (backbone) 之间可能有"分布漂移"——draft 觉得 OK 的 token，verifier 不一定接受。MTP KD 直接让 draft 模仿 verifier (teacher) 的 distribution，两者 distribution alignment 大幅提升，多 token 同时接受的概率几何级数放大。

这点对 inference 非常实用 — Speculative decoding 的 wall-clock speedup 几乎线性正比于平均接受 token 数。acc_2 翻倍意味着在长 generation 上 speedup 显著提升。

### 3.4 KD + LM > pure KD（knowledge-intensive）

Table 3 关键对比：
- NTP KD alone: MMLU 74.16, MMLU-Pro 50.97
- NTP KD + LM Loss: MMLU 74.93 (+0.77), MMLU-Pro 51.44 (+0.47)

但 NTP KD + LM 在 GSM8K (84.27→82.98)、EvalPlus (67.32→66.07) 上略降。

**Intuition**：纯 KD 的 student 容易"模仿 teacher 的错误"，尤其是 teacher 对某些 long-tail fact 给出 low-confidence distribution 时，KD 的信号弱。LM loss 提供 ground truth 的强信号，相当于"事实锚定"。但 LM loss 的硬目标在 reasoning task（GSM8K）上反而可能让 student over-fit 到 token pattern 而忽略 teacher 的 reasoning distribution。

这就是为什么作者把 $\lambda$ 设计成 linear decay 1.0 → 0.75 — **前期 KD 主导（学习 teacher 的整体 distribution），后期 LM 增加（强化 ground truth 锚定）**。一个完整的"先模仿、再巩固" curriculum。

---

## 4. Progressive Pruning > One-shot（Section 3.3 后半 + Table 5）

三种 progressive schedule：
1. **Depth-first**: Stage 1 砍 50% 的 depth reduction (6 层)，保持 width 不变；Stage 2 砍剩 6 层 + 全部 width reduction
2. **Width-first**: Stage 1 砍 50% 的 width reduction (从 2048→1792)，保持 depth；Stage 2 砍剩 width + 全部 depth reduction
3. **Joint**: Stage 1 同时砍 50% depth + 50% width；Stage 2 砍剩 50%+50%

Table 5 关键对比（所有方法总 token 400B）：

| Method | MMLU | MMLU-Redux | BBH | GSM8K |
|---|---|---|---|---|
| One-stage (400B 直接) | 75.86 | 75.41 | 73.97 | 85.22 |
| Joint (40B+360B) | 76.30 | 76.93 | 71.40 | 86.05 |
| Width-first (40B+360B) | 77.14 | 77.07 | 75.22 | 84.00 |
| **Depth-first (SlimQwen)** | **77.39** | **78.01** | 70.70 | 85.82 |

**所有 progressive 都比 one-shot 好**。Depth-first 在 MMLU/MMLU-Redux 最优，width-first 在 BBH 最优。

**Intuition**：
- One-shot 直接把 80B 砍到 23B 是 4× 容量跳跃，optimizer landscape 突变巨大，continual pretraining 必须同时"修复架构损伤"和"学习新数据"
- Progressive 把这个跳跃分成两次 2× 跳跃，每次"修复 + 学新数据"更 tractable
- 类似 curriculum learning / progressive resizing in CV：让 model 先适应中等压缩，再适应更大压缩

为什么 Depth-first 最优？直觉：**depth reduction 对 representation 的影响更"局部"**（剪掉末尾层只影响后段 computation），先 depth 砍半让 student 学会用更短网络做相同任务；后 width 砍半是在"已经会用短网络"基础上做 representation compression。反过来 Width-first 先砍 width 让每层 representation 容量降低，再砍 depth 时已经"瘦弱"的 layers 难以维持 hierarchical abstraction。

Appendix Table 9 测试了 3-stage（20B+20B+360B），发现相比 2-stage 没有显著增益——说明 2-stage 已经足够 capture progressive 的好处，再多分只是徒增 schedule 复杂度。

---

## 5. Pruning vs. From Scratch（Table 1，核心 motivation）

| Init | Training | Avg. |
|---|---|---|
| Random Init + KD | 120B | 61.66 |
| Pruned + LM Loss | 120B | 69.96 |
| **Pruned + KD** | **120B** | **73.45** |
| Qwen3-Next-80A3B (teacher) | — | 82.68 |

**Pruned 初始化比 random init 高出 11.79 个点**，恢复 86.5% teacher 性能。这是一个非常大的 delta，说明：

**Intuition — Pretrained MoE 里有大量"可复用 sub-circuits"**：尽管砍掉 50% experts、25% depth、25% width，剩下的 weights 仍保留了 task-critical 的 representation 路径。这与 lottery ticket hypothesis (Frankle & Carbin, 2018, https://arxiv.org/abs/1803.03635) 在 pretraining scale 上的延伸 — pruning 后的 subnetwork 是个"winning ticket"。

更深一点：MoE 模型天然有 redundancy — 512 个 experts 中很多在 calibration set 上 frequency 极低（甚至接近 0），它们对 task performance 贡献小但占用参数。Pruning 这些 "dead-ish" experts 几乎无损。

Figure 2 的 training loss 曲线也证实：pruned init 收敛更快、最终 loss 更低，KD 进一步降低 loss。Random init 在 120B token 内根本"学不动"这么多知识。

---

## 6. 最终 SlimQwen 性能 + Efficiency

SlimQwen-23A2B 是 Depth-first progressive + 全套 loss (LM + KD + MTP-LM + MTP-KD) 训练 400B tokens 的产物。

Efficiency (Table 11)：
- Peak memory: 156.56 GB → 43.30 GB（3.6× 降低，可单 80GB GPU 部署）
- HF backend prefill latency: 0.99s → 0.44s (2.25× 加速)
- HF backend decoding throughput: 4.05 → 6.55 tok/s (1.62× 加速)
- vLLM prefill latency: 0.08s → 0.06s (1.33×)
- vLLM decoding throughput: 142.58 → 210.87 tok/s (1.48×)

单 GPU 部署这点非常实用 — 去掉 TP/PP 的通信开销，对小模型 inference 实际 wall-clock 收益更大。

---

## 7. 几个 paper 没说但值得思考的点

1. **Expert importance 的 calibration set bias**：1024 样本从训练数据采样。如果训练数据分布与下游 benchmark 不同，importance 估计有偏。Paper 没做 sensitivity analysis on calibration set size/diversity。如果用更领域多样的 calibration（比如 mix 代码 + 数学 + 多语言），importance ranking 会不会变？这是个 open question。

2. **Shared expert 的处理**：MoE 路由公式 (2) 中 shared expert 是 always-on，但 paper 没讨论 pruning 时 shared expert 是否被合并。从 Algorithm 1 看只针对 routed experts。Shared expert 通常承载 "common knowledge"，把它 merge 进 routed experts 可能损失基础能力。这点值得 ablation。

3. **Linear attention 层 vs Full attention 层的 differential pruning**：Qwen3-Next 是 hybrid 12+36。Pruning 砍 3 full + 9 linear 是按原比例砍。如果按"full attention 更重要"砍 0 full + 12 linear，或反之，performance 如何？这是个未探索的维度。Full attention 对 retrieval 任务关键（MMLU、EvalPlus），linear attention 对 long-context 关键 — 砍不同比例会在不同 benchmark 上 trade。

4. **MTP depth $D=1$ 的局限**：paper 只用 $D=1$，即 MTP module 只预测 next-next token。如果 $D=2,3$ 训练成本和收益曲线如何？MTP module 参数会变多（每个 depth 一个 transformer block + projection），可能稀释 compression 收益。但 Speculative decoding 的 speedup 可能更大（acc_2/acc_3 进一步提升）。值得探索。

5. **Progressive pruning 的 LR schedule**：Appendix A.3 说"second stage starts from the learning rate reached at the final step of the first stage"。如果 Stage 1 已经 decay 到很低，Stage 2 的"余热"够不够 recover 新剪的 capacity？是否需要在 Stage 2 重新 warmup？这点 paper 没消融。

6. **Expert merging 的 fine-grained 影响分析**：Table 2 显示 9 种 (importance metric × group method × preserve/no) 组合，差异都在 1-2 个点内。但没分析"merge 后某个 expert 的 routing frequency 分布变化" — 比如 merge 后某个 base expert 是否变成新的 "dead expert"（被 router 几乎不激活）？这关系到是否需要 post-merge retraining 来 rebalance router。

---

## 8. 这篇 paper 在 literature 中的位置

它填补了几个 gap：

- **Minitron (Muralidharan et al., 2024)**：在 dense MNMG 15B 上做 pruning + KD，证明 pruning 比 scratch 好很多。SlimQwen 把这个结论 extend 到 MoE + pretraining scale (80B→23B)。
- **ShearedLLaMA (Xia et al., 2024b)**：在 dense LLaMA 上做 progressive pruning + batchable progressive schedule。SlimQwen 借鉴 progressive 思想但用更简单的两阶段。
- **ShortGPT (Men et al., 2024)**：用 layer similarity 找冗余层。SlimQwen 实测发现在 pretraining-scale KD 设定下不如直接剪最后几层。
- **SlimMoE (Li et al., 2025)**：MoE 内部 expert 的 intermediate dim pruning。SlimQwen 砍 expert 数量而非 expert 内部 dim。
- **REAP (Lasby et al., 2025a)**：one-shot MoE compression 对比各种 metric。SlimQwen 把 REAP 拿到 pretraining scale 上 re-evaluate，发现差异不大 — 一个重要的 negative result 校正。
- **Multi-token prediction (Gloeckle et al., 2024)**：原始 MTP 用 ground truth label 训练 backbone。SlimQwen 第一次把 MTP 用作 distillation target，并显式验证 speculative decoding 加速。

**这篇 paper 最有价值的不是新方法，而是 pretraining scale 上的一组 robust empirical findings + 两个简单但 work 的改进（partial-preservation merging + MTP KD）**。对于想压缩自己 MoE 模型的团队，几乎可以照搬这个 recipe。

---

## 9. 一句话总结每个 finding 的 intuition

1. **Pruned > Scratch**: pretrained MoE 的 weights 里 encode 了 task-critical sub-circuits，4× 压缩后仍保留 86% 性能，因为 redundancy 主要在 dead-ish experts 和深层 refinement layers。
2. **Expert metric 不重要**: 400B token continual training 是个"等价 basin attractor"——不同 one-shot 压缩起点都被拉到相近的 final basin，所以选简单 metric (Frequency) 即可。
3. **Partial preservation**: 一半 pristine specialist + 一半 merge，让 model 既保留 specialization anchor 又吸收 discarded knowledge，避免 homogenize。
4. **LM + KD > 纯 KD**: KD 模仿 teacher distribution 但 long-tail fact 信号弱，LM 锚定 ground truth，两者互补。$\lambda$ linear decay 实现先模仿后巩固。
5. **MTP KD**: 让 backbone hidden 显式编码 "multi-step future"，提升 representation quality，并天然 align speculative decoding 的 draft 与 verifier，远 token 接受率几何级数提升。
6. **Progressive > One-shot**: 4× 跳跃拆成 2× + 2×，优化 landscape 更平滑。Depth-first 最优因为 depth reduction 的"局部性"更强。
7. **Last-layer > Middle-layer pruning**: 深层是 amplification/refinement，剪掉 KD 能教 student 用短网络做最后一步；中间层是 hierarchical representation 关键 transformation，剪掉破坏严重。

这篇 paper 给出了 pretraining-scale MoE 压缩的一组 practical + reproducible recipe，对实际部署大模型有直接价值。
