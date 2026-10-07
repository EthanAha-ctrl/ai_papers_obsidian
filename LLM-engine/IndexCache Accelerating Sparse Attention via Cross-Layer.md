---
source_pdf: IndexCache Accelerating Sparse Attention via Cross-Layer.pdf
paper_sha256: fc2112da624ec7852e20fc83df36cafe1f32e58b5aa9c44b840f0bf157c9beb2
processed_at: '2026-08-05T09:29:40-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 IndexCache

## 一句话版本

Transformer 每层都在干同一件事: 看看前面哪些 token 重要, 然后只关注这几个. IndexCache 发现**相邻几层选出来的重要 token 几乎一模一样**, 所以不用每层都重新算一遍, 大部分层直接抄邻居的答案就行, 省掉 75% 的计算量, 速度翻倍, 质量不掉.

---

## 为什么会有这个问题

先说背景. 现代 LLM 要处理几十万 token 的超长 context, 传统 attention 是 $O(L^2)$, 序列翻倍计算量翻四倍, 撑不住. 所以有了 sparse attention: 每个 query 只挑最重要的 k=2048 个 token 来 attend, 复杂度降到 $O(Lk)$.

DeepSeek 搞的 DSA 就是 production 版本的 sparse attention. 它的做法是每层加一个 lightweight "indexer" — 一个小神经网络, 先快速扫一遍所有 token 打分, 选出 top-k, 然后主 attention 只在这 k 个 token 上算. indexer 本身很轻量 (few heads, low-rank, FP8), 单 FLOP 成本比主 attention 低一个数量级.

**但问题来了**: indexer 虽然单 FLOP 便宜, 它的复杂度还是 $O(L^2)$, 因为要给每个 token 打分. 一层没事, N 层叠起来就是 $O(NL^2)$. 论文 profile 了 30B DSA 模型, 发现 context 长的时候 indexer 占总 latency 的比例急速上升, 尤其 prefill 阶段. 主 attention 已经被 sparse 化了增长温和, indexer 反而成了新瓶颈.

直观比喻: 你为了省油给车装了个智能导航, 导航自己很省电, 但跑长途时导航自己也要持续耗电, 跑久了导航的耗电反而成了主要开销. DSA 现在就这个处境.

---

## 核心发现: 相邻层选的 token 几乎一样

论文做了个很简单的实验: 拿 768 个 200K context 的样本, 跑 DSA 模型, 对每一层, 记录 indexer 选出的 top-k 索引集合 $\mathcal{T}^{(\ell)}$, 然后算任意两层之间的 overlap ratio:

$$\frac{|\mathcal{T}^{(i)} \cap \mathcal{T}^{(j)}|}{k}$$

就是两层选出来的 token 集合的交集大小除以 k.

结果 heatmap 显示:
- **相邻层 overlap 70-100%**, 几乎选的是同一批 token
- 形成明显的 block structure, 比如 layers 3-5 一组, 6-8 一组, 17-30 一组, 块内 overlap 高
- block 边界处 overlap 突然下降, 说明有少数 transition 层会切换注意力焦点
- 早期层 vs 晚期层 overlap 很低 (≤0.4), 因为浅层和深层关注的东西根本不同

这个现象其实之前在 full attention 模型上就有人观察到 (Kascade, HySparse), 但 DSA 已经把 full attention 干掉了, 只剩 lightweight indexer, 论文是第一个验证 indexer 输出也跨层稳定的工作.

为什么稳定? 从 transformer 架构角度想, layer norm + residual connection 让 hidden state 跨层是平滑演化的, query 的 attention pattern 不会剧烈跳变. 每层 indexer 学的是"当前 query 觉得哪些 token 重要", 这个重要性判断与具体哪一层关系不大 — 你在 layer 10 觉得 token A 重要, 到 layer 11 大概率也觉得它重要.

---

## IndexCache 怎么做

很简单. 把 N 层分成两类:

- **F 层 (Full)**: 保留 indexer, 正常算自己的 top-k
- **S 层 (Shared)**: 删掉 indexer, 直接抄最近一个 F 层的 top-k 索引

用个 binary pattern string $\mathbf{c}$ 表示, 比如 1/4 retention 就是 4 层里 1 个 F, 3 个 S: `FSSS...`. 第 1 层永远是 F (要 seed 初始索引).

推理时改动极小: 加一个 if 分支. F 层算完索引存到一个临时 buffer $\mathcal{T}_{\text{cache}}$, S 层直接读这个 buffer. buffer 是单个 tensor, 每个 F 层覆写它, 不增加显存.

复杂度变化:
- 原来 indexer 成本: $O(NL^2)$
- 现在 indexer 成本: $O(F_{\text{count}} \cdot L^2)$, 其中 $F_{\text{count}}$ 是 F 层数量
- Core attention 成本 $O(NLk)$ 不变 (每层还是要跑 attention, 只是用继承来的索引)

保留 1/4 indexer, indexer 计算量砍 75%.

---

## 怎么选哪些层是 F, 哪些是 S

### 朴素方案: uniform interleaving, 不行

最简单的是均匀交错, 比如每隔 4 层放一个 F: `FSSSFSSS...`. 但论文发现**不同层对 indexer 移除的敏感度差异巨大**. 有些层删了 indexer 质量几乎不受影响, 有些层删了质量暴跌. 均匀交错可能正好砍掉关键层的 indexer.

### 方案一: Training-Free Greedy Search

不碰模型权重, 直接在现成 DSA 模型上搜最佳 pattern. 算法很直接:

```
初始化: 全部 F
重复 K 次 (K 是想要的 S 层数):
    对每个当前还是 F 的层, 试探性翻成 S
    跑一下 calibration set 算 LM loss
    commit loss 最低的那个翻转
```

每步 N 次前向, 总共 K 步, 成本 O(N²) 次前向. 用 pipeline parallelism 分块搜可以加速 P 倍.

为什么用 LM loss 不用其他指标? 论文诚实地记录了一个失败的尝试 (Appendix C): 用 attention output 的 cosine similarity 构造相似度矩阵, 再用动态规划搜最大化总相似度的 pattern. 结果 DP-similarity-optimal pattern 在下游任务上和 uniform interleaving 一样差, 完全没用.

根本原因: per-layer output similarity 是 local metric, 只看单层 output 保存得好不好, 忽略误差如何跨层 cascade. 两个层 attention output 看起来几乎一样, 但 reused index 可能漏掉少数 critical tokens, 这些 token 的重要性要在后面好几层的 reasoning 里才显现, 误差累积起来最终质量明显下降. LM loss 是 global metric, 直接捕捉 end-to-end 效果, 能识别出 similarity 看不出的 critical layers.

这个负结果其实挺有启发性的: **相似不等于可互换**, 在深 cascade 系统里, local proxy 基本都会失效, 只有 end-to-end 的 global metric 才靠谱.

Greedy search 有个很漂亮的副产品: per-step LM loss 曲线呈现明显分段 — 前 20 步几乎不涨 (easy layers, 删了无所谓), 35 步后急升 (critical layers, 删了质量崩). 这给出一个 layer importance 的自然排序, 而且跨不同 calibration set 稳定, 是模型内禀属性. 早期层特别敏感因为扰动会传播最长路径, transition 层敏感因为它负责切换注意力焦点.

### 方案二: Training-Aware Multi-Layer Distillation

如果愿意从 DSA base model 重新训练 (或 continue pre-train), 可以做得更好. 标准 DSA 训练时每个 indexer 只对自己层做 distillation:

$$\mathcal{L}^I = \sum_t D_{KL}(\mathbf{p}_t^{(\ell)} \| \mathbf{q}_t^{(\ell)})$$

$\mathbf{p}_t^{(\ell)}$ 是 layer $\ell$ 的 aggregated attention 分布 (target), $\mathbf{q}_t^{(\ell)}$ 是 indexer 输出分布 (被训练). 让 indexer 拟合本层的 attention pattern.

IndexCache 扩展成 multi-layer: F 层 indexer 同时服务自己 + 后面 m 个 S 层:

$$\mathcal{L}_{\text{multi}}^I = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{KL}(\mathbf{p}_t^{(\ell+j)} \| \mathbf{q}_t^{(\ell)})$$

$j$ 从 0 到 $m$ 遍历 F 层和它服务的 m 个 S 层, $\frac{1}{m+1}$ 是均匀权重. 直觉: 让 indexer 学一个对它服务的所有层都有用的 consensus top-k, 不是 overfitting 到本层.

**漂亮的数学性质 (Proposition 1)**: 这个 multi-layer loss 的梯度, 等价于对 averaged target 的单层 distillation:

$$\bar{\mathbf{p}}_t = \sum_{j=0}^{m} \frac{1}{m+1} \mathbf{p}_t^{(\ell+j)}$$

$$\mathcal{L}_{\text{avg}}^I = \sum_t D_{KL}(\bar{\mathbf{p}}_t \| \mathbf{q}_t^{(\ell)})$$

证明很简单: KL 散度里 target $\mathbf{p}$ 的 entropy 项对 $\theta$ 求导为零, 只剩 cross-entropy 项 $\mathbf{p}(s)\log\mathbf{q}(s)$, 加权求和后 $\mathbf{p}$ 被替换成 $\bar{\mathbf{p}}$, 等价.

**直觉**: multi-layer distillation = 让 indexer 拟合它服务的所有层 attention 分布的 centroid (cluster center). indexer 学的是"这一簇层的共识 top-k", 自然能 generalize 到簇内每个成员. 这把"启发式正则"变成了有清晰几何意义的目标.

---

## 实验结果

### 速度 (30B DSA, H100, Table 1)

**Prefill time**:
| Context | DSA | +IndexCache(1/4) | Speedup |
|---|---|---|---|
| 10K | 0.57s | 0.45s | 1.27× |
| 60K | 3.38s | 2.59s | 1.31× |
| 120K | 8.57s | 5.66s | 1.52× |
| 200K | 19.5s | 10.7s | **1.82×** |

Context 越长 speedup 越大, 因为 indexer 占比随 L² 增长.

**Decode per request**:
| Context | DSA | +IndexCache(1/4) | Speedup |
|---|---|---|---|
| 200K | 58 tok/s | 86 tok/s | **1.48×** |

Decode 也能加速, 因为 DSA decode 时每生成一个 token 都要对全 context 跑一遍 indexer, IndexCache 砍掉 75% 这部分计算.

### 质量 (Table 2, Training-Free)

关键数字: **1/4 retention + searched pattern**, Long Avg 49.9, 原始 DSA 是 50.2, 几乎无损. 而 1/4 uniform interleaving (不搜) 是 43.0, 暴跌 7.2. 证明**哪些层留 indexer 比留多少重要得多**.

G&R (general & reasoning) 几乎不受影响, 1/4 searched 甚至在 AIME 2025 (92.6 vs 91.0) 和 GPQA (78.6 vs 77.6) 上略升, 移除冗余 indexer 可能起 mild regularizer 作用.

### 质量 (Table 3, Training-Aware)

最戏剧性的发现: **训练之后 uniform pattern 也能 work**. 1/2 uniform training-aware Long Avg 51.6, 甚至超过 searched pattern 50.6. 与 training-free 完全相反 (training-free 时 uniform 不行, 必须 searched).

原因: 没有 retrain 时, S 层与本层 indexer 的 top-k 紧耦合, 复用引出 distributional shift. Retrain 后, S 层学会适应继承来的 index, F 层 indexer 也学会输出能 generalize 到多层的选择, 双向适应消除了层间敏感性.

去掉 multi-layer distillation loss, AA-LCR 从 49.8 跌到 44.0, 验证 cross-layer distillation 有实质帮助.

### GLM-5 (744B, 40B active) 初步结果 (Table 4)

Training-free 1/4 + searched pattern, Long Avg 78.0, 原始 78.4, 仅降 0.4. 1.2× E2E speedup, 几乎无损. 在 production-scale 上验证了方法的 scalability.

---

## 几个直觉化 takeaway

**1. 瓶颈的二阶优化**: Sparse attention 是对 attention bottleneck 的一阶优化, 把 attention 从 O(L²) 降到 O(Lk). 但优化完之后, 原来跑龙套的 indexer 反而成了新瓶颈. IndexCache 是对 indexer 的再优化, 去掉 indexer 的跨层冗余. 这种"recursively exploit redundancy"的思路很通用 — 每次优化都会暴露下一个 bottleneck.

**2. 为什么 cross-layer reuse 能 work**: Transformer 的 layer norm + residual 让 hidden state 跨层是渐进精炼, 每层做的是 query-conditioned sparse retrieval, 而跨层 retrieval 目标高度稳定. 相邻层觉得重要的 token 几乎一样, 因为 query 的 attention 锚点不会剧烈跳变. 早期层和晚期层在绝对 token 上分歧 (corner dark), 但相邻层稳定.

**3. Local metric 在 cascade 系统中失效**: Appendix C 的负结果值得记住. Attention output similarity 是 local proxy, 但 LLM 是几十层的深 cascade, 微小误差会累积放大. 两个层看起来 output 几乎一样, 漏掉的少数 critical tokens 可能在后面好几层才显现重要性. 真正能指导决策的只有 end-to-end LM loss. 这是为什么 greedy search 工作而 similarity-based DP 不工作.

**4. Training 消除 pattern sensitivity**: Training-free 必须小心选 pattern 避开 critical layers. Training-aware 让 model 适应共享, uniform 也能 work. 本质是训练让 F 层 indexer 输出与 S 层期望对齐, 分布匹配问题被解决了. 这也是为什么 training-aware 不需要 expensive pattern search.

**5. Multi-layer distillation = 拟合 centroid**: Proposition 1 把多目标 distillation 化简为对 averaged target 的单目标 distillation. indexer 学的是"一簇层 attention 分布的中心点", 自然 generalize 到簇内所有成员. 几何意义干净.

**6. 工程极简**: 改动只有一个 if 分支 + 一个 tensor buffer, 无额外显存, 与 pipeline parallelism / dp attention / SGLang serving 完全兼容. 这是能在 GLM-5 744B production 模型上直接验证的关键. 很多好 idea 死在工程不友好上, IndexCache 反过来.

**7. 可扩展性**: 这个 cross-layer reuse 原则可能扩展到任何有动态 token selection 的 sparse attention (MoBA block selection, NSA block selection, SeerAttention gating), 因为所有 selection signal 跨层都应该呈稳定态. 论文 Section 5.2 明确暗示这个方向.

---

## 最后的一句大白话

Transformer 每层都在问"哪些 token 重要", 相邻层给出的答案几乎一样, 所以不用每层都问一遍, 让大部分层抄前面 Full 层的答案, 省 75% 计算, 速度翻倍, 质量几乎不掉. 选哪些层当 Full 层比选多少重要得多, 直接拿 LM loss greedy 搜就行. 如果愿意重新训练, 让 Full 层 indexer 学一个对它服务的所有层都有用的 consensus 答案, 连 pattern 都不用搜, uniform 就行.

---

# IndexCache: 通过 Cross-Layer Index Reuse 加速 Sparse Attention

## 1. 问题动机:Indexer 本身成为瓶颈

DeepSeek Sparse Attention (DSA) 把 attention 分成两阶段:**selection** (lightning indexer 选 top-k tokens) + **computation** (core attention 仅在 k=2048 个 token 上计算). Core attention 从 O(L²) 降到 O(Lk), 但 **indexer 仍然是 O(L²) per layer**, 跨 N 层累加成 O(NL²). 论文里 profile 30B DSA 模型发现: context 增长时, indexer 占总 latency 的比例急速上升 (尤其 prefill 阶段), 因为 MLA core attention 增长温和, 而 indexer 按 L² 增长.

直观上: indexer 是用来"省 attention"的"小工具", 但当小工具本身也按 L² 增长时, 它自己就成了瓶颈. 这就是论文要解决的根本矛盾.

## 2. 关键经验观察:Cross-Layer Top-k Index Overlap

论文用 30B DSA 在 768 个 200K context 样本上计算 pairwise overlap ratio:

$$\frac{|\mathcal{T}^{(i)} \cap \mathcal{T}^{(j)}|}{k}$$

变量含义:
- $\mathcal{T}^{(i)}, \mathcal{T}^{(j)}$: 第 i 层与第 j 层 indexer 输出的 top-k 索引集合 (k=2048)
- $|\cdot|$: 集合基数
- 整体表示两层选中的 token 重叠比例

Appendix A 的 heatmap 显示:
- **相邻层 overlap 70-100%**, 证明 consecutive layers 几乎选同一批 token
- **Block structure**: 如 layers 3-5, 6-8, 17-30, 31-36 形成功能块, 块内 overlap 高
- **Block 边界处 overlap 快速衰减**, 存在"transition layers"会切换注意力焦点
- **Early-late 角落 overlap ≤ 0.4**, 早期层与晚期层关注完全不同的 token

这一现象与 Deshmukh et al. (2025) Kascade 和 Gao et al. (2026) HySparse 在 full attention 上观察到的 cross-layer token selection stability 一致, 但 DSA 完全没有 full attention oracle, 论文是首次在 sparse attention indexer 上验证这个性质.

参考链接:
- Kascade: https://arxiv.org/abs/2512.16391
- HySparse: https://arxiv.org/abs/2602.03560
- DSA (DeepSeek-V3.2): https://arxiv.org/abs/2512.02556

## 3. IndexCache 核心设计

把 N 层分成两类, 用 binary pattern string $\mathbf{c} = c_1 c_2 \cdots c_N$, $c_\ell \in \{F, S\}$:

- **F (Full)**: 保留 indexer, 计算自己的 top-k $\mathcal{T}_t^{(\ell)} = \text{Top-k}(\mathbf{I}_t^{(\ell)})$
- **S (Shared)**: 没有 indexer, 继承最近的 F 层的索引:
  $$\mathcal{T}_t^{(\ell)} \gets \mathcal{T}_t^{(f(\ell))}, \quad f(\ell) = \max\{j < \ell : c_j = F\}$$
  $f(\ell)$ 是最近的 F 层索引

第一层永远是 F, 用来 seed 初始索引. 推理只需一个 conditional branch: F 层算索引并 cache 到 $\mathcal{T}_{\text{cache}}$, S 层直接 copy. 注意 $\mathcal{T}_{\text{cache}}$ 是单个 tensor buffer, 每个新 F 层覆写它, 不增加额外 GPU 显存.

复杂度对比:
- 标准 DSA: indexer O(NL²) + core attention O(NLk)
- IndexCache: indexer O(F·L²) + core attention O(NLk), 其中 F = 数量少的 Full 层

若保留 1/4 的 indexer, indexer 计算量直接砍掉 75%, core attention 不变.

## 4. Training-Free IndexCache: Greedy Layer Selection

### 4.1 为什么 uniform interleaving 不好

最朴素方案是均匀交错, 如 r=4 → "FSSSFSSS...". 但论文 empirical 发现**不同层对 indexer 移除的敏感度差异巨大**: 早期层和 transition 区域的层特别敏感. 均匀交错可能正好砍掉关键层的 indexer.

### 4.2 Greedy Search 算法

Algorithm 1:
```
输入: DSA 模型 M (N 层), calibration set D, 目标 S 层数 K
输出: 最优 pattern c*
1. c ← F^N (全 F 起始)
2. R ← {2,3,...,N} (候选层, 第 1 层永远是 F)
3. for step = 1 to K:
4.     ℓ* ← argmin_{ℓ∈R} EVAL_LOSS(M, D, c|_{c_ℓ→S})  // 试探性翻转
5.     c_{ℓ*} ← S, R ← R \ {ℓ*}  // commit 最优翻转
6. return c
```

每一步在所有候选 F 层中试探性翻成 S, 评估 calibration set 上 LM loss, commit loss 最低的那一个. 总共 K 步, 每步约 N 次前向, 完整搜索成本 O(N²) 次前向.

**Pipeline parallelism 加速**: 模型用 P 个 pipeline stage, 把层切成 P 块, 每块第 1 层固定 F, 每步每块独立搜索最佳翻转, 一次可 commit P 层, 总前向次数下降约 P×.

### 4.3 Greedy 解的三个性质

1. Searched pattern 在相同 retention ratio 下优于 uniform interleaving (Table 2)
2. Per-step LM loss 曲线呈现明显分段: 前 20 步是"easy layers" (loss 几乎不涨), 35 步后是"critical layers" (loss 急升), 给出 layer 重要性的自然排序
3. 不同 calibration set 上结果稳定, 说明重要性排序是模型内禀属性, 不是 data artifact

### 4.4 为什么需要 LM loss 而非 local similarity (Appendix C 的负结果)

论文诚实记录了一个失败的尝试: 用 attention output 的 cosine similarity 矩阵 + 动态规划搜索最优 pattern. 给定 N×N 下三角相似度矩阵 $S_{i,j}$ (i>j, layer i 用 layer j 的 indexer 时 output 与原 output 的余弦相似度), DP 状态:

$$\text{dp}[i][k] = \max_{j<i, c_j=F} \left\{ \text{dp}[j][k-1] + \sum_{m=j+1}^{i-1} S_{m,j} \right\}$$

变量含义:
- $\text{dp}[i][k]$: 考虑前 i 层用恰好 k 个 F 层, 且第 i 层是 F, 能达到的最大累积相似度
- 内层求和: layer j+1 到 i-1 这些 S 层都复用 layer j 的 index, 贡献相似度 $S_{m,j}$

Table 5 显示: DP-similarity-optimal pattern 在下游任务上与 uniform interleaving 几乎一样差 (MRCR 22.9 vs 22.0, GraphWalks 43.5 vs 46.6, RULER 82.9 vs 83.6). 也就是说**显式最大化 cross-layer similarity 没有带来任何好处**.

**根本原因**: per-layer output similarity 是 local metric, 测量单层 output 在 isolation 下保存得多好, 忽略误差如何跨层 cascade. 两个 layer 看起来 attention output 几乎一样 ($S_{i,j} \approx 1$), 但 reused index 可能漏掉了少量 critical tokens, 这些 token 的重要性只在后续 reasoning 步骤才显现, 误差累积导致最终质量明显下降. LM loss 是 global metric, 直接捕捉 end-to-end 效应.

这个负结果对 intuition 很关键: **相似不等于可互换**, cascade 效应让 local proxy 失效.

## 5. Training-Aware IndexCache: Multi-Layer Distillation

### 5.1 标准 DSA 的 indexer distillation

标准 DSA 训练时, 每个 indexer 通过 KL 散度蒸馏到本层 aggregated attention 分布:

$$\mathcal{L}^I = \sum_t D_{KL}\left(\mathbf{p}_t^{(\ell)} \big\| \mathbf{q}_t^{(\ell)}\right)$$

变量含义:
- $\mathbf{p}_t^{(\ell)}$: layer $\ell$ 在 query position $t$ 上的 aggregated attention 分布 (head 间 softmax 平均)
- $\mathbf{q}_t^{(\ell)} = \text{Softmax}(\mathbf{I}_t^{(\ell)})$: indexer 输出分布
- $D_{KL}(p\|q)$: KL 散度, 衡量 q 拟合 p 的程度

### 5.2 Multi-Layer Distillation Loss

扩展到 F 层服务 m 个后续 S 层:

$$\mathcal{L}_{\text{multi}}^I = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{KL}\left(\mathbf{p}_t^{(\ell+j)} \big\| \mathbf{q}_t^{(\ell)}\right) \tag{1}$$

变量含义:
- $\ell$: 当前 F 层索引
- $j \in \{0, 1, ..., m\}$: F 层及其服务的 m 个 S 层偏移量
- $\frac{1}{m+1}$: 均匀权重
- $\mathbf{p}_t^{(\ell+j)}$: 第 $\ell+j$ 层的 attention 分布 (target)
- $\mathbf{q}_t^{(\ell)}$: F 层 indexer 的输出分布 (被训练)

直觉: 让 F 层 indexer 学一个 jointly useful 的 top-k, 而不是 overfitting 到本层.

### 5.3 梯度等价性 (Proposition 1)

定义 averaged target:
$$\bar{\mathbf{p}}_t = \sum_{j=0}^{m} \frac{1}{m+1} \mathbf{p}_t^{(\ell+j)}$$

对应的 single-target loss:
$$\mathcal{L}_{\text{avg}}^I = \sum_t D_{KL}\left(\bar{\mathbf{p}}_t \big\| \mathbf{q}_t^{(\ell)}\right) \tag{2}$$

**Proposition 1**: $\nabla_\theta \mathcal{L}_{\text{multi}}^I = \nabla_\theta \mathcal{L}_{\text{avg}}^I$

证明 (公式 3):
$$\nabla_\theta \mathcal{L}_{\text{multi}}^I = -\sum_{j=0}^{m} \frac{1}{m+1} \sum_t \nabla_\theta \sum_s \mathbf{p}_t^{(\ell+j)}(s) \log \mathbf{q}_t^{(\ell)}(s)$$
$$= -\sum_t \nabla_\theta \sum_s \underbrace{\left(\sum_{j=0}^{m} \frac{1}{m+1} \mathbf{p}_t^{(\ell+j)}(s)\right)}_{\bar{\mathbf{p}}_t(s)} \log \mathbf{q}_t^{(\ell)}(s) = \nabla_\theta \mathcal{L}_{\text{avg}}^I$$

变量含义:
- $\theta$: indexer 参数
- $s$: 索引 token 位置
- 关键步: KL 散度中 target 分布 $\mathbf{p}$ 的 entropy 项对 $\theta$ 求导为 0, 只剩 cross-entropy 项, 是 $\mathbf{p}(s) \log \mathbf{q}(s)$ 形式, 加权求和后把 $\mathbf{p}$ 替换成 $\bar{\mathbf{p}}$, 等价于对 averaged target 的蒸馏

**Interpretation**: multi-layer distillation 等价于让 indexer 学习所有服务层 attention 分布的 centroid, 预测 consensus top-k. 这把"启发式正则"变成"有清晰几何意义"的目标 — indexer 拟合的是 cluster center.

### 5.4 为什么实际用 $\mathcal{L}_{\text{multi}}^I$ 而非 $\mathcal{L}_{\text{avg}}^I$

虽然梯度等价, 但实现效率不同. $\mathcal{L}_{\text{multi}}^I$ 中 S 层只需要接收 F 层的 $\mathbf{q}^{(\ell)}$; $\mathcal{L}_{\text{avg}}^I$ 需要同时传 $\mathbf{q}^{(\ell)}$ 和 $\mathbf{p}^{(\ell)}$, 引入内存和运行时开销.

### 5.5 两阶段训练

1. **Warm-up**: 用 $\mathcal{L}_{\text{multi}}^I$ 训 indexer, 其他参数冻结
2. **Sparse training**: 继续用 $\mathcal{L}_{\text{multi}}^I$ (仅在 top-k 上算 KL) + LM loss 联合训练所有参数

## 6. 实验结果详解

### 6.1 End-to-End Inference Speedup (Table 1, 30B DSA on H100)

**Prefill time (秒)**:
| Context | DSA | +IndexCache(1/2) | +IndexCache(1/4) | Speedup(1/4) |
|---|---|---|---|---|
| 10K | 0.57 | 0.47 | 0.45 | 1.27× |
| 60K | 3.38 | 2.86 | 2.59 | 1.31× |
| 120K | 8.57 | 6.57 | 5.66 | 1.52× |
| 200K | 19.5 | 13.7 | 10.7 | **1.82×** |

Speedup 随 context 增长而增大, 因为 indexer 占比上升.

**Decode per request (tok/s)**:
| Context | DSA | +IndexCache(1/4) | Speedup |
|---|---|---|---|
| 10K | 73.5 | 91.0 | 1.24× |
| 200K | 58.0 | 86.0 | **1.48×** |

**Decode full (KV cache 满载, 总 tok/s)**:
| Context | DSA | +IndexCache(1/4) | Speedup |
|---|---|---|---|
| 10K | 2700 | 3310 | 1.23× |
| 200K | 197 | 297 | **1.51×** |

Decode 阶段 IndexCache 也加速, 因为 DSA decode 时每生成一个 token 都要对全 context 跑一遍 indexer, IndexCache 直接砍掉 75% 这部分计算.

### 6.2 Training-Free Quality (Table 2)

关键发现: **searched pattern 大幅恢复 long-context 能力**.

| Config | Long Avg | G&R Avg | MRCR | GraphWalks |
|---|---|---|---|---|
| Original DSA | 50.2 | 74.6 | 24.5 | 49.6 |
| 1/2 Unif. | 47.4 | 74.3 | 22.0 | 46.6 |
| 1/2 +Search | 50.3 | 74.4 | 24.7 | 49.5 |
| 1/4 Unif. | 43.0 | 73.8 | 17.7 | 37.2 |
| 1/4 +Search | 49.9 | 74.9 | 25.1 | 47.4 |
| 1/8 Unif. | 35.3 | 70.0 | 12.9 | 33.1 |
| 1/8 +Search | 46.1 | 73.7 | 21.7 | 43.8 |

观察:
- 1/4 ratio searched pattern 的 Long Avg (49.9) 几乎等于原 DSA (50.2), 接近无损
- 1/8 极端稀疏下 searched pattern 仍大幅缓解退化 (35.3→46.1), 但仍有不可忽略下降
- G&R (general & reasoning) 几乎不掉, 1/4 searched 甚至在 AIME 2025 (92.6 vs 91.0) 和 GPQA-Diamond (78.6 vs 77.6) 上略升, 提示移除冗余 indexer 可能起 mild regularizer 作用
- 哪些层保留远比保留多少重要

### 6.3 Training-Aware Quality (Table 3)

| Config | Long Avg | G&R Avg | AA-LCR |
|---|---|---|---|
| Original DSA | 51.0 | 74.2 | 47.0 |
| 1/2 Unif. (training-aware) | 51.6 | 74.5 | 49.8 |
| 1/2 w/ searched pattern | 50.6 | 73.6 | 46.6 |
| 1/2 w/o cross-layer loss | 49.8 | 74.5 | 44.0 |
| 1/4 Unif. (training-aware) | 50.6 | 74.1 | 48.4 |

**三个关键洞察**:

1. **训练让 uniform pattern 也能 work**: 1/2 uniform training-aware Long Avg 51.6 > searched 50.6. 与 training-free 相反, training-free 时 searched 关键, training-aware 时 uniform 反而更好. 原因: 没有 retrain 时, S 层与本层 indexer 的 top-k 紧耦合, 复用引出 distributional shift; retrain 后, S 层适应继承的 index, F 层 indexer 同时学习 generalizable 选择, 联合适应消除层间敏感性.

2. **Cross-layer distillation 提供实质帮助**: 去掉 multi-layer loss, AA-LCR 从 49.8 跌到 44.0, Long Avg 从 51.6 跌到 49.8. 验证 indexer 学 consensus top-k 比学本层 top-k 更能 generalize.

3. **1/4 uniform training-aware 也接近 baseline**: Long Avg 50.6 vs 51.0, G&R 74.1 vs 74.2, 几乎无损.

### 6.4 GLM-5 (744B, 40B active) 初步结果 (Table 4)

| Config | Long Avg | MRCR v2 | GraphWalks | LongBench v2 | RULER | AA-LCR |
|---|---|---|---|---|---|---|
| Original DSA | 78.4 | 71.1 | 92.7 | 64.5 | 97.7 | 66.2 |
| 1/2 Unif. | 78.1 | 72.8 | 90.2 | 65.1 | 97.6 | 64.6 |
| 1/2 +Search | 78.7 | 72.3 | 90.8 | 66.0 | 97.3 | 67.2 |
| 1/4 Unif. | 72.7 | 65.8 | 74.9 | 62.2 | 96.2 | 64.6 |
| 1/4 +Search | 78.0 | 70.8 | 90.3 | 63.7 | 97.6 | 67.6 |

- 1/4 searched pattern 仍保持 Long Avg 78.0 (仅降 0.4)
- 1/4 uniform 明显退化 (78.4→72.7), 主要在 GraphWalks 上 (92.7→74.9), 1/4 uniform 偶然跳过了关键 indexer 层的概率下降
- Figure 1 显示 1/2 retention IndexCache 在 Artificial Analysis Index 上整体性能与原 GLM-5 几乎一致, 提供 ~1.2× E2E speedup

## 7. 与 Related Work 的区别

跨层共享方法 TidalDecode (Yang et al., 2025a), LessIsMore (Yang et al., 2025b), OmniKV (Hao et al., 2025), DELTA (Zarch et al., 2025), Kascade (Deshmukh et al., 2025), HySparse (Gao et al., 2026) 都**依赖 full attention anchor 层**作为 oracle 计算 exact top-k.

IndexCache 的两点关键不同:
1. **Oracle 本质更便宜**: 共享 DSA 的 lightweight indexer 输出, 而非 full O(L²) attention scores
2. **系统化优化共享配置**: training-free greedy search 找最优结构 + training-aware multi-layer distillation 适配参数

参考:
- TidalDecode: https://arxiv.org/abs/2503.13276
- OmniKV: https://openreview.net/forum?id=OmniKV
- DSA (DeepSeek-V3.2): https://arxiv.org/abs/2512.02556
- GLM-5: https://arxiv.org/abs/2602.15763
- SeerAttention: https://arxiv.org/abs/2410.13276
- NSA (Yuan et al., 2025): https://arxiv.org/abs/2502.11089
- MoBA (Lu et al., 2025): https://arxiv.org/abs/2502.13189
- DuoAttention: https://arxiv.org/abs/2410.10819
- Quest: https://arxiv.org/abs/2406.10774
- MInference: https://arxiv.org/abs/2407.02490

## 8. 几点 Intuition 总结

1. **Indexer 的"小"是相对的, 量级仍按 L² 增长**: 这是论文的起点. 论文做的是"瓶颈的第二层优化" — 解决 attention 稀疏化后剩下的 indexer 成本.

2. **Cross-layer stability 是 sparse attention 的可复用结构**: 即使没有 full attention, DSA 的 lightweight indexer 输出也跨层稳定, 这让共享成为可能. overlap 70-100% 意味着大部分 indexer 计算确实是冗余的.

3. **Local metric 在 cascade 系统中失效**: Appendix C 的负结果是最具启发性的部分之一. Similarity 是 local proxy, 但 LLM 是深 cascade 系统, 任何看似无害的微小扰动都可能放大. 真正能指导决策的只有 end-to-end LM loss, 这是为什么 greedy search 工作而 DP-on-similarity 不工作.

4. **Greedy 揭示 layer importance ordering**: 前 20 步 LM loss 几乎不涨 → "expendable layers"; 35 步后急升 → "critical layers". 这个 ordering 与 layer 功能相关: 早期层扰动会传播最长路径, transition 层负责切换注意力焦点. 这与 representation 上的"early-late 角落 dark"完全一致.

5. **Training 消除 pattern sensitivity**: 这是 training-aware 与 training-free 最戏剧性的差别. Training-free 必须小心选 pattern 避开 critical layers; training-aware 让 model 适应共享, 即使 uniform 也能工作. 这是分布匹配的本质: 训练让 F 层 indexer 输出与 S 层期望对齐.

6. **Multi-layer distillation 等价于拟合 centroid**: Proposition 1 把多目标 distillation 化简为单目标 centroid distillation, 提供干净的几何直觉 — indexer 学的是"cluster center"的 top-k, 自然 generalize 到所有成员层.

7. **Mild regularization 效应**: 1/4 searched pattern 在 AIME/GPQA 上略升, 提示移除冗余 indexer 可能减少噪声/overfitting, 副作用是性能微提升.

8. **Pattern 的 generality**: Appendix B 列出 30B 与 GLM-5 的具体 searched pattern. 它们不规则 (如 30B 1/4: FSFSFSSSSFSSSFSSFFSSFSSFSSSSFSSSFSSSSFSSSSSSSSS), 反映层重要性非均匀分布, 但与 overlap cluster 不完全重合, 再次证明 local overlap 不能替代 end-to-end 搜索.

## 9. 个人直觉化思考

从 Karpathy 视角, 这个工作对应"system bottleneck 的二阶导数优化" — 当一阶优化 (sparse attention) 普及后, 真正的 cost center 从 attention 转移到 attention 的 selector, IndexCache 就是把 selector 也稀疏化. 这种"recursively exploit redundancy"思路很自然.

更深层的 intuition: transformer 每层做的是 query-conditioned sparse retrieval, 而跨层的 retrieval 目标高度稳定, 因为 transformer 的 layer norm + residual 让 hidden state 演化是平滑的, query 的 attention 锚点不会剧烈跳变. DSA 的 indexer 学到的就是"哪些 token 对当前 query 重要", 这个重要性分布与 layer depth 弱相关 — 早期层和晚期层在绝对 token 上分歧 (corner dark), 但相邻层稳定. 这与 transformer 是"渐进精炼"表示的视角完全契合.

进一步推论: 这种 cross-layer reuse 原则可能扩展到任何包含动态 token selection 步骤的 sparse attention 方法 (MoBA block selection, NSA block selection, SeerAttention gating), 因为所有这些方法的"selection signal"在跨层上应该都呈稳定态. 论文 Section 5.2 末段也明确暗示这个方向.

从工程视角, IndexCache 的实现极简: 单个 conditional branch + 单个 index tensor buffer, 无额外显存, 无架构改动, 与 dp attention / pipeline parallelism / SGLang serving 完全兼容. 这是它能直接在 GLM-5 production-scale 上验证的关键 — 工程友好度极高, 1.2× E2E speedup 在 production 是相当显著的 serving 成本节省.

潜在风险与未来工作:
- 1/8 retention 开始退化, 说明 reuse 上限存在, 1/4 是 sweet spot
- Training-aware 在 744B GLM-5 上还未完成, 是论文明确计划的下一步
- Agentic workflow 下 indexer 行为是否仍稳定值得验证
- 是否能与 hybrid attention (linear+softmax, mamba+transformer) 进一步组合优化

参考论文主页/作者: Yushi Bai, Qian Dong, Jie Tang (清华 KEG / GLM team), Z.ai (智谱): https://github.com/zai-org 或 https://www.z.ai
