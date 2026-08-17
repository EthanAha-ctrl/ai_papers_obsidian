---
source_pdf: SPICE SUBMODULAR PENALIZED INFORMATION–CONFLICT SELECTION FOR EFFICIENT
  LARGE LANGUAGE MODEL TRAINING.pdf
paper_sha256: 71afb4a049a7eeea0e90acf38479181f0d38991015a300503372a18be8839cda
processed_at: '2026-08-12T09:56:32-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SPICE 用人话讲

## 0. 一句话版本

**选数据的时候，光看"信息量大"不够，还得看这些数据的 gradient 会不会互相打架。SPICE 就是在选数据时同时考虑"信息量"和"冲突度"，用 10% 的数据就能打过用 100% 数据训练的效果。**

---

## 1. 这 paper 在解决什么实际问题

LLM fine-tuning 有个很反直觉的现象：用全部数据 train，反而不如只用 10-20% 精选的数据。LESS（ICML'24, https://arxiv.org/abs/2406.06046）和 SelectIT（https://arxiv.org/abs/2402.16705）都验证了这点。

所以问题变成：**怎么从 N 个样本里挑出最值钱的 k 个？**

这叫 data selection for instruction tuning，是 2024-2025 年的热门方向。Survey 见 Albalak et al. 2024: https://arxiv.org/abs/2402.16827

---

## 2. Fisher Information Selection 是什么，为什么大家觉得它好

### 直觉

你 fine-tune 一个 model，本质是用 gradient 更新参数。每个 training sample 都对应一个 gradient vector $g_i = \nabla_\theta \ell(x_i, y_i)$，告诉你"往哪个方向走能降低这个 sample 的 loss"。

如果你能选出一批 samples，它们的 gradients **张成的方向空间最大**，那这些 samples 就最能"教会"model 不同的东西。这就是 Fisher information 的几何含义。

### 数学定义

Subset $S$ 的 Fisher Information Matrix:
$$\mathbf{F}_S = \sum_{i \in S} g_i g_i^\top$$

Utility function:
$$F(S) = \log \det(\mathbf{I} + \alpha \mathbf{F}_S)$$

变量解释：
- $g_i$：sample $i$ 的 gradient，是个 $d$ 维向量（$d$ = 参数数）
- $\mathbf{F}_S$：把所有 selected gradients 外积加起来，是个 $d \times d$ 矩阵
- $\alpha > 0$：scaling 参数，保证矩阵可逆
- $\log \det$：行列式取对数，几何上等于"椭球体积的对数"

$\log \det$ 大 = gradient 方向覆盖广 = 信息丰富。这跟 DPP（Determinantal Point Process, https://arxiv.org/abs/2303.17358）的思路一脉相承——都追求 diversity。

### 理论上的"保证"

$F(S) = \log \det(\mathbf{I} + \alpha \mathbf{F}_S)$ 是 **monotone submodular**（证明见 Appendix B，用 matrix determinant lemma）。

Submodular 的意思就是 diminishing returns：集越大，加新样本的边际收益越小。就像吃自助餐，第一盘最香，第十盘就腻了。

对 submodular function 做 greedy（每次挑 marginal gain 最大的），Nemhauser 1978（https://link.springer.com/article/10.1007/BF01588971）证明你能拿到至少 $(1 - 1/e) \approx 63.2\%$ 的最优解。理论上很稳。

---

## 3. 但实践上 Fisher Greedy 经常拉胯，为什么？

**这就是 paper 要解决的核心 puzzle。**

理论说 greedy 应该 smooth diminishing returns，但实践中 marginal gain 经常**断崖式下跌**。你选到第 20 个样本，marginal gain 还很高；选到第 30 个，突然接近 0。Greedy 的实际表现比 $(1-1/e)$ 差得多。

Figure 1(b) 展示了这个现象：标准 Fisher greedy 的 $\Delta_t$ 衰减很快，累积信息低。

### 作者的 key insight：Gradient Conflict

Gradient conflict 就是不同样本的 gradient 方向不一致，甚至相反。比如一个 sample 要 model 往东走，另一个要往西走，两个一起 train 就互相抵消。

这个现象在 multi-task learning 里早有研究：
- PCGrad (Yu et al. 2020, https://arxiv.org/abs/2001.06782)：project conflicting gradients
- CAGrad (Liu et al. 2024, https://arxiv.org/abs/2110.14048)：conflict-averse descent
- Recon (Shi et al. 2023, https://arxiv.org/abs/2302.11289)：从根上减冲突

**但这些工作都在 training 阶段处理冲突。SPICE 的新意是在 data selection 阶段就处理，并且把它跟 submodular theory 链起来。**

---

## 4. 为什么 Gradient Conflict 会破坏 Greedy？—— ε-decomposition

### 拆解 Marginal Gain

加 sample $x$ 到 set $S$ 的 marginal gain：
$$\Delta_x(S) = F(S \cup \{x\}) - F(S) = \log(1 + \alpha g_x^\top (\mathbf{I} + \alpha \mathbf{F}_S)^{-1} g_x)$$

作者把它拆成两部分（Definition 3）：

$$\Delta_x(S) = \underbrace{\log(1 + \alpha \|g_x\|^2)}_{\text{base}_x: \text{只跟 } x \text{ 自己有关}} + \underbrace{\varepsilon_x(S)}_{\text{perturbation: 跟 } S \text{ 交互有关}}$$

其中：
$$\varepsilon_x(S) = \log \frac{1 + \alpha g_x^\top (\mathbf{I} + \alpha \mathbf{F}_S)^{-1} g_x}{1 + \alpha \|g_x\|^2}$$

### 关键观察（Theorem 1）

Submodularity 的 diminishing returns **完全来自 $\varepsilon_x(S)$**：
$$\Delta_x(A) - \Delta_x(B) = \varepsilon_x(A) - \varepsilon_x(B) \geq 0 \quad \text{当 } A \subseteq B$$

因为 $\text{base}_x$ 跟 $S$ 无关，相减抵消了。

而且 $\varepsilon_x(S) \leq 0$，随着 $|S|$ 增大越来越负。$|\varepsilon_x(S)|$ 正好是 marginal gain 相对 baseline 的累计衰减量。

**所以问题变成：什么控制了 $|\varepsilon_x(S)|$ 的增长速度？**

### Theorem 4：Gradient Inner Product 控制 Perturbation

在技术假设下（$\alpha \|\mathbf{F}_S\| \leq \rho < 1$，gradients 有界），证明见 Appendix E，核心是用 Neumann series 展开 $(\mathbf{I} + \alpha \mathbf{F}_S)^{-1}$：

$$|\varepsilon_x(S)| \leq C \cdot \frac{\alpha^2 \sum_{y \in S} (g_x^\top g_y)^2}{1 + \alpha \|g_x\|^2}$$

变量解释：
- $C = \frac{1}{1 - \rho - \alpha G_{\max}^2 \rho}$：problem-dependent 常数
- $\rho$：$\alpha \|\mathbf{F}_S\|$ 的上界
- $G_{\max} = \max_x \|g_x\|$：最大 gradient norm
- $(g_x^\top g_y)^2$：**gradient inner product 的平方**，这就是 alignment/conflict 的度量

**Intuition**：$\sum_{y \in S} (g_x^\top g_y)^2$ 大 = $g_x$ 跟 set $S$ 里很多 gradient 都有强相关（正或负都算，因为平方）= perturbation 大 = marginal gain 衰减快。

负相关就是 conflict，正相关就是 redundancy，两者都会让 perturbation 增长。但 conflict 更"毒"——它让 greedy 的 approximation guarantee 直接变差。

### Curvature 链条

**Lemma 1**：Submodular 的 total curvature
$$c = 1 - \min_x \frac{\Delta_x(\mathcal{D} \setminus \{x\})}{\Delta_x(\emptyset)} \leq \max_x \frac{|\varepsilon_x(\mathcal{D} \setminus \{x\})|}{\text{base}_x}$$

**Theorem 3**（Conforti & Cornuéjols 1984, https://www.sciencedirect.com/science/article/pii/0166218X84900039）：greedy 保证
$$F(S_{\text{greedy}}) \geq \frac{1 - e^{-c}}{c} \cdot F(S^*)$$

- $c = 1$：退化为经典 $(1-1/e) \approx 0.632$
- $c \to 0$：逼近 $1$（完美）

**Corollary 1**：组合起来
$$\hat{c} \propto \max_x \sum_{y \neq x} (g_x^\top g_y)^2$$

Gradient inner products 小 → perturbation 小 → curvature 小 → greedy 近似好。

**这条链条就是 paper 的核心理论贡献**：从 empirical gradient statistics 到 approximation guarantee，中间用 ε-decomposition 桥接。

---

## 5. 实验验证链条（Section 3, Appendix F）

### Conflict 度量

Definition 4：
$$\text{Conflict}(g_i) = \max\{0, -\frac{g_i^\top \bar{g}}{\|g_i\| \|\bar{g}\| + \eta}\}$$

$\bar{g}$ 是当前已选 set 的 mean gradient。Hinge form 只惩罚"真正反对"mean 方向的样本，不惩罚 small positive alignment。

### 三个关键实证

**Figure 2(a)**：512 个样本 gradient 可视化
- 大部分 align mean
- 少部分 conflict mean
- **关键**：有些高 conflict 样本同时高 Fisher information！

这很重要——直接丢弃 conflict 样本会损失高信息样本。需要 trade-off，不能一刀切。

**Figure 2(b)**：按 conflict 分 top/bottom 20% 跑 Fisher greedy
- Low-conflict：marginal gain 衰减慢，累积信息高
- High-conflict：marginal gain 衰减快，迅速 diminishing
- Half-life 通常 10-30 步 → motivates early stopping

**Figure 2(c)** Spearman correlation：
- Conflict vs marginal gain $\Delta$：$\rho = -0.792$（强负相关）
- Conflict vs $|\varepsilon_x(S)|$：$\rho = 0.901$（强正相关）

跨 6 个数据集（HumanEval, GSM8K, MMLU, Code Alpaca, Stanford Alpaca, ShareGPT）一致，见 Appendix F.2, F.3。

### ε-decomposition 精度验证（Table 4）

| $|S|$ | Spearman $\rho$ | Bound Violation | Decomposition Error |
|------|----------------|-----------------|---------------------|
| 16 | 0.723 | 6.4% | < 1e-6 |
| 256 | 0.818 | 13.4% | < 1e-6 |

Decomposition 误差 < 1e-6，理论精确。10% bound violation 主要来自 AdaFisher 对角近似的偏差。

### Curvature 实测（Table 5）

| Conflict Level | $c$ | Prediction Error |
|---------------|-----|------------------|
| Low | 0.032 | 1.57% |
| Medium | 0.074 | 3.59% |
| High | 0.092 | 4.46% |

$c$ 随 conflict 单调增，验证 Lemma 1。Spearman $\rho = 0.86$。

---

## 6. SPICE Algorithm（Section 4）

### 核心打分函数

公式 14：
$$\text{score}(x | S_{t-1}) = \underbrace{\Delta_x(S_{t-1})}_{\text{Fisher 信息增益}} - \lambda \cdot \underbrace{\text{conflict}(x | S_{t-1})}_{\text{跟 mean gradient 的冲突}}$$

$\lambda \geq 0$ 是 penalty 权重，default 0.1。

**关键设计哲学**：不硬性丢弃高 conflict 样本。只要 $\Delta_x$ 足够大，即使 conflict 高，score 仍可能 competitive。这是 soft penalty，让 information 和 conflict 自己 trade-off。

对比硬过滤：如果直接丢弃 conflict > threshold 的样本，会损失高信息样本。SPICE 保留它们，只在 score 上打折。

### Early Stopping

Adaptive 版本：
$$t_{\text{stop}} = \min\{t : \Delta_{x_t}(S_{t-1}) \leq \omega \cdot \Delta_{x_1}(S_0)\}$$

$\omega = 0.5$ default，意思是 marginal gain 跌到第一个样本的一半就停。Empirically 在 25-30% data 处达到半衰期（Table 9）。

### Proxy Model + Schedule

用 0.5B 小 model 算 gradient，给 7B target model 选数据。基于观察：gradient selection patterns 跨 scale 迁移好（LESS, SelectIT 同样观察）。

Schedule：每个 120-sample pool 选 12 个，每 $T=10$ 轮做一次 training step，reset buffer。

### AdaFisher 降复杂度（Appendix A）

传统 FIM $O(d^2)$ 存储，$O(d^3)$ inversion。AdaFisher（Gomes et al. 2025, https://arxiv.org/abs/2405.16397）用 diagonal block-Kronecker：
$$\hat{F}_i = H_{i-1} \otimes S_i \approx \text{Diag}(H_{i-1}) \otimes \text{Diag}(S_i)$$

降到 $O(d)$。这让 billion-scale model 的 selection 可行。总复杂度 $O(k|\mathcal{D}|d)$。

---

## 7. 实验结果讲人话

### 主实验（Table 1, Qwen2-7B）

8 个 benchmark，10% data：

- **SPICE 平均 58.0**
- Full data 100%：56.4
- DPP 10%：57.0（第二名）
- Fisher 10%：55.9
- Random 10%：55.5

SPICE 用 1/10 数据，平均比 full data 高 1.6 分。8 个 benchmark 里 7 个最优，IFEval 涨 +5.1（33.5 → 38.6）。

LLaMA2-7B（Table 2）类似：SPICE 31.1 vs Full 30.8。

### 时间成本（Table 15）

| Method | Selection Time |
|--------|----------------|
| SPICE | **2:56** |
| IFD | 4:32 |
| LEAD | 12:02 |
| LESS | 16:22 |
| Fisher | 17:01 |
| DPP | 23:32 |
| SelectIT | 25:19 |

SPICE selection 只要 3 小时，总成本（selection + 10% SFT）20 GPU-hours，**比 full data LoRA 还便宜**，性能还更好。

### Ablation：λ（Figure 3b）

- $\lambda = 0$（纯 Fisher）：性能显著下降
- $\lambda \in [0.1, 0.5]$：稳定好
- $\lambda = 1.0$：过强，略降

Default 0.1 验证 conflict penalty 必要。

### Proxy model（Table 10, 11）

Qwen2-0.5B vs Qwen2-7B 做 proxy：67.0 vs 67.2，几乎一样。用小 proxy 完全够。

Cross-architecture（LLaMA2 proxy 给 Qwen2 target）：65.9-66.1，明显低。**Proxy 要同族**。

70B target 用 0.5B proxy 仍工作（Table 11）。

### Budget 扩展（Table 14）

Qwen2-7B 上：
- 1% SPICE：54.2（已超 Null 53.6，接近 Full 56.4）
- 5%：56.5
- 10%：58.0（超 Full）
- 30%（SPICE+）：58.1（边际收益小）

**10% → 30% 提升只有 0.1，说明信息瓶颈在 10% 就接近饱和**。Early stopping 有道理。

### SPICE+ Early Stopping（Table 9）

| $\omega$ | Data Rate | Avg | Time |
|----------|-----------|-----|------|
| 0.1 | 55.1% | 67.1 | 9:55 |
| 0.5 | 26.1% | 67.2 | 5:49 |
| 0.7 | 9.3% | 67.1 | 2:45 |

$\omega = 0.5$ 选 26% 数据，5:49 时间，性能跟 SPICE 10% 持平。说明后期 marginal gain 确实低，early stop 划算。

### Diversity（Table 3）

SPICE 的 NovelSum/LDD 接近 DPP（diversity 专用方法），domain coverage 与 full corpus 接近。Conflict penalty 只压制"破坏性反对"，不压制正常 diversity。

### Overlap with Fisher（Table 12）

SPICE 与 Fisher Jaccard = 0.47，双向 overlap 0.64。**SPICE 保留了 Fisher 的信息核心，差异在 marginal/boundary 样本上**。Case study（Table 13）显示 SPICE 剔除的是简单 redundant 样本（如 "What is a linked list?"），保留核心算术/代码题。

---

## 8. 跟相关方法对比讲人话

| 方法 | 怎么选 | 问题 |
|------|--------|------|
| Random | 随机 | 浪费，但 Xia et al. 2024b (https://arxiv.org/abs/2410.09335) 说 1-2% random 就够 |
| IFD (Li et al. 2024b, https://aclanthology.org/2024.naacl-long.421) | 按 instruction difficulty | 不考虑 model 状态 |
| Fisher (Deb et al. 2025, https://arxiv.org/abs/2505.14826) | log-det FIM greedy | 忽略 gradient conflict |
| LESS (Xia et al. 2024a) | gradient similarity to validation | 需要_validation set，用 cosine 隐式考虑 alignment |
| SelectIT (Liu et al. 2025) | LLM uncertainty | 要 full dataset inference，25 GPU-hours |
| DPP (Zhang et al. 2023) | diversity kernel | 隐式 diversity，无 conflict 理论 |
| TSDS (Liu et al. 2024d, https://arxiv.org/abs/2410.11303) | task-conditioned | 需要 task definition |
| LEAD (Lin et al. 2025, https://arxiv.org/abs/2505.07437) | iterative | 12 GPU-hours，无 conflict 理论 |
| **SPICE** | **log-det FIM - λ·conflict** | **显式 conflict + 数据依赖 approximation bound** |

SPICE 的 unique selling point：**第一个把 gradient conflict 量化链接到 submodular selection theory**，给出 data-dependent approximation factor $\frac{1-e^{-\hat{c}}}{\hat{c}}$ where $\hat{c} \propto \max_x \sum_y (g_x^\top g_y)^2$。

---

## 9. Build Intuition 的三个 mental model

### 模型 1：吃自助餐

Fisher greedy = 每次拿"最香"的菜。但第 10 盘牛肉再香，你已经吃了 9 盘牛肉，边际满足感低。

SPICE = 拿菜时同时考虑"这菜跟已经拿的菜搭不搭"。牛肉拿过了，下次拿蔬菜，即使蔬菜单独看不那么香。

Conflict penalty 就是"搭配度"评分。

### 模型 2：组装战队

每个 sample = 一个英雄，gradient = 他的技能方向。

Fisher greedy = 每次选战斗力（$\|g\|^2$）最高的。但五个刺客打架都往一个方向冲，实际输出低。

SPICE = 战斗力高 + 跟队友技能方向互补（不直接对冲）。战队整体 DPS 高。

### 模型 3：几何视角

Gradients 是 $d$ 维空间的向量。Fisher log-det = 它们张成的椭球体积。

- Pure Fisher：选最长向量 → 但都在一个方向，体积小
- SPICE：选长向量 + 跟已有方向 orthogonal 的 → 体积大

$(g_x^\top g_y)^2$ 大 = 共线（正或负）= 对体积贡献小。SPICE 鼓励"新方向"。

---

## 10. 这 paper 真正的贡献是什么

### 不是什么

- 不是新发明 data selection（LESS/Fisher/DPP 早有）
- 不是新发明 gradient conflict（PCGrad/CAGrad 早有）
- 不是新发明 submodular greedy（Nemhauser 1978 早有）

### 是什么

**第一个建立 quantitative link**：
$$\text{Gradient conflict} \xrightarrow{\sum (g_x^\top g_y)^2} |\varepsilon_x(S)| \xrightarrow{\text{Lemma 1}} \text{curvature } c \xrightarrow{\text{Theorem 3}} \text{greedy approximation factor}$$

这条链让"为什么 Fisher greedy 实践中拉胯"有了理论解释，并且指明解决方向：控制 conflict → 控制 curvature → 改善 greedy。

**方法上**：SPICE 把这个 insight operationalize 成一个简单 score function + early stopping + proxy model，3 小时选完，10% 数据打过 100%。

**实验上**：8 benchmark × 2 model × 多 baseline，ablation 完整，cost analysis 透明，diversity/overlap/case study 都有。

---

## 11. 局限与未来方向

1. **Conflict 定义粗糙**：只用 cosine vs mean gradient，没考虑 high-order（三样本互冲突）。可扩展到 subspace alignment。
2. **Proxy transfer 限制**：cross-architecture 不行（LLaMA proxy 给 Qwen target 差）。需要 architecture-invariant features。
3. **技术假设 $\alpha \|\mathbf{F}_S\| < 1$**：high-conflict data 上违反，prediction error 增大到 4.46%（Table 5）。
4. **Fixed checkpoint**：gradient 在一个 checkpoint 算，training 过程中 model 变化没考虑。LEAD 的 iterative 思路可结合。
5. **RLHF / Multimodal**：gradient conflict 在 RLHF 更严重（reward model + policy），SPICE framework 可直接迁移。
6. **Active learning**：ε-decomposition 可推广到其他 submodular selection（core-set, batch active learning）。

---

## 12. 实现层面的联想

如果你要复现 SPICE，核心代码逻辑：

```python
def spice_select(dataset, k, lambda_=0.1, proxy_model):
    # Step 1: 单次 forward，缓存所有 gradients
    gradients = {i: compute_grad(proxy_model, dataset[i]) for i in range(len(dataset))}
    
    S = []
    for t in range(k):
        # Step 2: 算当前 mean gradient
        mean_g = mean([gradients[i] for i in S]) if S else None
        
        # Step 3: 给每个候选打分
        scores = {}
        for x in candidates_not_in_S:
            # Fisher marginal gain (用 AdaFisher 对角近似)
            delta = log_fisher_marginal(g_x, S, gradients)
            
            # Conflict (hinge form)
            if mean_g is not None:
                conflict = max(0, -cosine_sim(g_x, mean_g))
            else:
                conflict = 0
            
            scores[x] = delta - lambda_ * conflict
        
        # Step 4: 选 max，检查 early stopping
        x_star = argmax(scores)
        if t > 0 and scores[x_star] <= omega * first_delta:
            break
        S.append(x_star)
    
    return S
```

关键 efficiency trick：AdaFisher 对角近似让 `log_fisher_marginal` 从 $O(d^2)$ 降到 $O(d)$：

```python
def log_fisher_marginal(g_x, S, gradients, alpha=0.1):
    # F_S ≈ diag(sum_i g_i^2)  # 对角近似
    diag_F = sum(g_i ** 2 for i in S)  # element-wise
    inv_term = 1.0 / (1.0 + alpha * diag_F)  # O(d)
    return np.log(1 + alpha * np.sum(g_x ** 2 * inv_term))
```

这就是 K-FAC（Martens & Grosse, https://arxiv.org/abs/1503.05671）的极简版，牺牲精度换速度，selection 场景够用。

---

## 13. 总结一句

**SPICE 告诉我们：选 LLM training data 时，"信息量大"和"方向一致"得一起看。纯贪心选高信息样本会让 gradients 互相打架，实际有效信息被抵消。加个 conflict penalty，让 greedy 在 submodular 意义上更接近最优，10% 数据就能打过 100%。**

理论上是 ε-decomposition 把 gradient conflict 链接到 submodular curvature，实践上是 3 小时 selection + 10% SFT = 20 GPU-hours 超过 full data。Clean theory + strong empirics + 实用 pipeline，是个很 solid 的 work。

参考论文本身：SPICE paper（Bilibili Inc., 2025）
相关理论：
- Submodular: Lovász 1983, https://link.springer.com/chapter/10.1007/978-3-642-68874-4_10
- Greedy bound: Nemhauser et al. 1978, https://link.springer.com/article/10.1007/BF01588971
- Curvature: Conforti & Cornuéjols 1984, https://www.sciencedirect.com/science/article/pii/0166218X84900039
- Multilinear extension: Feige & Vondrák 2010, https://theoryofcomputing.org/articles/v006a011

相关方法：
- LESS: https://arxiv.org/abs/2406.06046
- SelectIT: https://arxiv.org/abs/2402.16705
- FisherSFT: https://arxiv.org/abs/2505.14826
- DPP: https://arxiv.org/abs/2303.17358
- LEAD: https://arxiv.org/abs/2505.07437
- IFD: https://aclanthology.org/2024.naacl-long.421

Gradient conflict：
- PCGrad: https://arxiv.org/abs/2001.06782
- CAGrad: https://arxiv.org/abs/2110.14048
- Recon: https://arxiv.org/abs/2302.11289

Efficient optimization：
- K-FAC: https://arxiv.org/abs/1503.05671
- AdaFisher: https://arxiv.org/abs/2405.16397

---

# SPICE: Submodular Penalized Information–Conflict Selection 深度解析

## 1. Paper 核心思想与 Motivation

这篇 paper 来自 Bilibili Inc.,针对 LLM instruction tuning 中的 data selection 问题。核心 question 是:为什么用 log-det Fisher Information Matrix (FIM) 做 greedy selection,理论保证是 $(1-1/e)$ approximation,但实践中性能下降比理论预测快得多?

作者的核心 insight:**gradient conflicts**(per-sample gradients 之间的 misalignment)是 marginal information gain 加速 decay 的根本原因。log-det FIM 确实是 submodular 的(Lovász, 1983),但 submodularity 只保证 diminishing returns,**不保证 decay rate uniform**。Gradient conflicts 越多,perturbation 越大,curvature 越大,greedy approximation 越弱。

参考链接:
- Submodular functions and convexity (Lovász 1983): https://link.springer.com/chapter/10.1007/978-3-642-68874-4_10
- Nemhauser-Wolsey-Fisher greedy bound: https://link.springer.com/article/10.1007/BF01588971
- Conforti-Cornuéjols curvature: https://www.sciencedirect.com/science/article/pii/0166218X84900039

---

## 2. 理论框架细节

### 2.1 Fisher Information Utility

给定 instruction-response pairs $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$,每个样本的 gradient:
$$g_i = \nabla_\theta \ell((x_i, y_i); \theta)$$

Empirical FIM over subset $S$:
$$\mathbf{F}_S = \sum_{i \in S} g_i g_i^\top$$

Utility function:
$$F(S) = \log \det(\mathbf{I} + \alpha \mathbf{F}_S)$$

变量解释:
- $g_i \in \mathbb{R}^d$:per-sample gradient,$d$ 是 model parameter 数
- $\mathbf{F}_S \in \mathbb{R}^{d \times d}$:PSD 矩阵
- $\alpha > 0$:regularization/scale 参数,确保 $\mathbf{I} + \alpha \mathbf{F}_S$ 正定可逆
- $\mathbf{I}$:identity matrix

**为什么 log-det?** Fisher information 度量 model 对 data 的"敏感度",log-det 是 determinant 的对数,可以理解为"几何平均方差"或"信息体积"。最大化它等价于最大化参数空间中能被 data 约束的方向数量,即"覆盖"最 diverse 的 gradient directions。这与 DPP (Determinantal Point Processes) 的 intuition 相同(参考 https://arxiv.org/abs/2303.17358)。

### 2.2 Submodularity Proof (Appendix B)

用 **matrix determinant lemma**:
$$\det(\mathbf{I} + \alpha \mathbf{F}_{S \cup \{x\}}) = \det(\mathbf{I} + \alpha \mathbf{F}_S) \cdot (1 + \alpha g_x^\top (\mathbf{I} + \alpha \mathbf{F}_S)^{-1} g_x)$$

所以 marginal gain:
$$\Delta_x(S) = \log(1 + \alpha g_x^\top (\mathbf{I} + \alpha \mathbf{F}_S)^{-1} g_x)$$

如果 $A \subseteq B$,则 $\mathbf{F}_A \preceq \mathbf{F}_B$(Loewner order),由 PD matrix inversion 的 order-reversing 性质:
$$(\mathbf{I} + \alpha \mathbf{F}_A)^{-1} \succeq (\mathbf{I} + \alpha \mathbf{F}_B)^{-1}$$

预乘和后乘 $g_x$:
$$g_x^\top (\mathbf{I} + \alpha \mathbf{F}_A)^{-1} g_x \geq g_x^\top (\mathbf{I} + \alpha \mathbf{F}_B)^{-1} g_x$$

由于 $z \mapsto \log(1 + \alpha w_x z)$ 严格递增,得 $\Delta_x(A) \geq \Delta_x(B)$。**Submodular + monotone**。

Intuition:每加入一个样本,它在已有"梯度方向空间"中的"投影贡献"递减,因为已有方向被覆盖。

### 2.3 ε-decomposition (核心创新)

**Definition 3** 把 marginal gain 拆为两部分:

Modular baseline(只依赖单样本):
$$\text{base}_x = \log(1 + \alpha \|g_x\|^2)$$

Perturbation(依赖 sample interactions):
$$\varepsilon_x(S) = \Delta_x(S) - \text{base}_x = \log\frac{1 + \alpha g_x^\top (\mathbf{I} + \alpha \mathbf{F}_S)^{-1} g_x}{1 + \alpha \|g_x\|^2}$$

**Theorem 1 关键观察**:
$$\Delta_x(A) - \Delta_x(B) = \underbrace{[\text{base}_x - \text{base}_x]}_{=0} + [\varepsilon_x(A) - \varepsilon_x(B)] = \varepsilon_x(A) - \varepsilon_x(B) \geq 0$$

Submodularity 完全由 perturbation terms 驱动!Modular baseline 不参与 diminishing returns。

更深刻的结果(公式 6):沿 chain $\emptyset = S_0 \subset \cdots \subset S_T = S$:
$$\Delta_x(\emptyset) - \Delta_x(S) = -\varepsilon_x(S)$$

$|\varepsilon_x(S)|$ 正好是 marginal gain 从 baseline decay 的总量。

### 2.4 Curvature-Dependent Greedy Guarantee

**Theorem 3**:定义 total curvature
$$c = 1 - \min_{x \in \mathcal{D}} \frac{\Delta_x(\mathcal{D} \setminus \{x\})}{\Delta_x(\emptyset)} \in [0, 1]$$

Greedy 保证:
$$F(S_{\text{greedy}}) \geq \frac{1 - e^{-c}}{c} \cdot F(S^*)$$

变量解释:
- $c \in [0,1]$:curvature,衡量 marginal gains 因 element interactions 能减少多少
- $S^*$:optimal k-subset
- $c=1$ 时退化为经典 $(1-1/e) \approx 0.632$
- $c \to 0$ 时逼近 $1$(完美近似)

**Lemma 1**:curvature 被 perturbation 控制:
$$c = -\min_x \frac{\varepsilon_x(\mathcal{D} \setminus \{x\})}{\text{base}_x} \leq \max_x \frac{|\varepsilon_x(\mathcal{D} \setminus \{x\})|}{\text{base}_x}$$

Intuition:worst-case normalized perturbation 决定 curvature,所以控制 $|\varepsilon_x(S)|$ 可改善 approximation。

证明思路(Appendix D)用 **multilinear extension**:
$$G(y) = \mathbb{E}_{R \sim y}[f(R)]$$
其中 $R \sim y$ 是按概率 $y_i$ 独立包含 element $i$ 的 product distribution。然后构造 continuous greedy ODE:
$$\frac{dy}{dt} \in \arg\max_{v \in P_k} v \cdot \nabla G(y(t))$$

通过 Gronwall's inequality 解 ODE 得 $H(1) \geq \frac{F(S^*)}{c}(1 - e^{-c})$,再用 **dependent rounding** (Ageev & Sviridenko 2004, https://link.springer.com/article/10.1023/B:JOCO.0000038913.96607.c2) 转回离散解。

### 2.5 Perturbation Bound via Gradient Alignment (关键定理)

**Theorem 4**(详细证明在 Appendix E):
$$|\varepsilon_x(S)| \leq C \cdot \frac{\alpha^2 \sum_{y \in S} (g_x^\top g_y)^2}{1 + \alpha \|g_x\|^2}, \quad x \notin S$$

其中:
$$C(\rho, G_{\max}) = \frac{1}{1 - \rho - \alpha G_{\max}^2 \rho}$$

变量解释:
- $\rho$: $\alpha \|\mathbf{F}_S\|$ 的上界(typically $\rho \in [0.1, 0.5]$)
- $G_{\max} = \max_x \|g_x\|$:max gradient norm
- $(g_x^\top g_y)^2$:gradient inner product squared,衡量 alignment

**证明关键步骤**:
1. Neumann series 展开:$(\mathbf{I} + A)^{-1} = \sum_{m=0}^\infty (-A)^m$,其中 $A = \alpha \mathbf{F}_S$,要求 $\|A\| \leq \rho < 1$
2. 一阶项:$g_x^\top A g_x = \alpha \Sigma_x(S)$,其中 $\Sigma_x(S) = \sum_{y \in S} (g_x^\top g_y)^2$
3. Tail 用 PSD matrix 谱不等式 $A^m \preceq \|A\|^{m-1} A$ 控制几何级数
4. 用 $|\log(1+u)| \leq |u|/(1-|u|)$ 控制 log 项

**Corollary 1**:greedy approximation factor
$$F(S_{\text{greedy}}) \geq \frac{1 - e^{-\hat{c}}}{\hat{c}} \cdot F(S^*)$$
其中 $\hat{c} \propto \max_x \sum_{y \neq x} (g_x^\top g_y)^2$。

**核心 take-away**:smaller gradient inner products → smaller perturbations → lower curvature → stronger approximation guarantee。Gradient misalignment 直接控制 greedy 质量。

这与 multi-task learning 中的 **gradient conflict** 文献高度相关:
- PCGrad (Yu et al. 2020, https://arxiv.org/abs/2001.06782):projection of conflicting gradients
- CAGrad (Liu et al. 2024, https://arxiv.org/abs/2110.14048):conflict-averse gradient descent
- Recon (Shi et al. 2023, https://arxiv.org/abs/2302.11289):reducing conflicting gradients from root

但 SPICE 把 gradient conflict 链接到 submodular selection theory,新视角。

---

## 3. Empirical Analysis

### 3.1 Gradient Conflict 度量

**Definition 4**:
$$\text{Align}(g_i) = \frac{g_i^\top \bar{g}}{\|g_i\| \|\bar{g}\| + \eta}$$
$$\text{Conflict}(g_i) = \max\{0, -\text{Align}(g_i)\}$$

其中 $\bar{g}_t = \frac{1}{|S_{t-1}|}\sum_{x \in S_{t-1}} g_x$ 是当前已选 set 的 mean gradient,$\eta = 10^{-8}$ 防止数值不稳定。

注意 hinge form $\max\{0, -\cdot\}$:只惩罚 negative alignment(真正反对 mean 方向),不惩罚 small positive alignment。

### 3.2 关键 empirical 发现

**Figure 2(a)**:在 2D/3D gradient 空间可视化 512 个样本,发现:
- 大部分样本 align mean gradient
- 少部分样本 conflict mean gradient
- **关键**:有些高 conflict 样本同时具有高 Fisher information!

Intuition:conflicting gradients 可能把 model 拉出 local optima,代表 highest-quality gradients(参考 Recon, Shi et al. 2023)。简单丢弃冲突样本会损失高信息样本,需要 trade-off。

**Figure 2(b)**:把样本按 conflict level 分 top/bottom 20%,各自跑 Fisher greedy:
- Low-conflict 组:marginal gain decay 慢,累积信息高
- High-conflict 组:marginal gain decay 快,迅速 diminishing
- 半衰期(half-life)通常在 10-30 步内达到 → motivates early stopping

**Figure 2(c)** Spearman correlation:
- Conflict vs marginal gain $\Delta$: $\rho = -0.792$(强负相关)
- Conflict vs perturbation $|\varepsilon_x(S)|$: $\rho = 0.901$(强正相关)

这直接验证 Corollary 1,跨 6 个数据集(HumanEval, GSM8K, MMLU, Code Alpaca, Stanford Alpaca, ShareGPT)一致,见 Appendix F.2, F.3。

### 3.3 ε-decomposition 实证(Appendix F.4)

Table 4 在 Qwen2-7B 上用 512 个样本验证:

| $|S|$ | Spearman $\rho$ | Pearson $r$ | Bound Violation | Decomposition Error |
|------|----------------|-------------|-----------------|---------------------|
| 16 | 0.723 | 0.501 | 6.4% | < 1e-6 |
| 32 | 0.756 | 0.515 | 7.1% | < 1e-6 |
| 64 | 0.789 | 0.528 | 10.3% | < 1e-6 |
| 128 | 0.801 | 0.534 | 12.8% | < 1e-6 |
| 256 | 0.818 | 0.541 | 13.4% | < 1e-6 |
| Overall | 0.777 | 0.524 | 10.0% | < 1e-6 |

观察:
- Decomposition error < 1e-6:理论精确
- Correlation 随 $|S|$ 增长:larger subsets amplify conflict effects
- 10% bound violation:acceptable,主要来自 AdaFisher 对角近似引入的偏差

### 3.4 Curvature 参数分析(Appendix F.4.2)

Table 5 按 conflict level 分组测 curvature $c$:

| Conflict Level | Curvature $c$ | Bound Holds | Actual Ratio | Theoretical Ratio | Prediction Error |
|---------------|---------------|-------------|--------------|-------------------|------------------|
| Low | 0.032 | ✓ | 1.000 | 0.984 | 1.57% |
| Medium | 0.074 | ✓ | 1.000 | 0.964 | 3.59% |
| High | 0.092 | ✓ | 1.000 | 0.955 | 4.46% |

- $c$ 单调递增 0.032 → 0.074 → 0.092:conflict 越高,curvature 越大,验证 Lemma 1
- Spearman $\rho = 0.859$ ($p < 0.001$)
- 技术假设 $\alpha \|\mathbf{F}_S\| < 1$ 在 high-conflict 组被违反,prediction error 增大,显示 theoretical bound 的 practical boundary

---

## 4. SPICE Algorithm 详解

### 4.1 Conflict-Aware Greedy Score

**公式 14**:
$$\text{score}(x | S_{t-1}) = \Delta_x(S_{t-1}) - \lambda \cdot \text{conflict}(x | S_{t-1}), \quad \lambda \geq 0$$

变量解释:
- $\Delta_x(S_{t-1})$:Fisher marginal gain(信息项)
- $\text{conflict}(x|S_{t-1}) = \max\{0, -\cos(g_x, \bar{g}_{S_{t-1}})\}$:冲突项
- $\lambda$:penalty 权重,default 0.1

**关键设计**:不丢弃高信息但 conflict 的样本。只要 $\Delta_x$ 足够大,score 仍 competitive。这与简单丢弃 conflict 样本的方法(如基于 alignment 阈值的过滤)本质不同。

Intuition:第一项 favor 高信息,第二项 discourage 强 directional opposition。当 $\lambda = 0$ 退化为 standard Fisher greedy(已知问题:步梯度方向冲突,训练不稳,见 Figure 6)。

### 4.2 Early Stopping Criteria

**Adaptive (data-driven)**:
$$t_{\text{stop}} = \min\{t : \Delta_{x_t}(S_{t-1}) \leq \omega \cdot \Delta_{x_1}(S_0)\}$$

变量:
- $\Delta_{x_1}(S_0)$:第一个被选样本的 marginal gain(最大)
- $\omega \in (0, 1)$:threshold,default 0.5(半衰期)
- $t_{\text{stop}}$:停止时刻,effective budget $k_{\text{eff}} = t_{\text{stop}} \leq k$

Empirically,大多数数据在 25-30% 处达到半衰期(Appendix H.3)。

**Fixed Budget**:为 fair comparison,主实验用 fixed $k$ 模式。

### 4.3 Proxy Model + Schedule

**Proxy Model**:用 small model(0.5B)做 selection,target model 是 LLaMA2-7B / Qwen2-7B。基于观察:gradient-based selection patterns 跨 scale 迁移良好(LESS, SelectIT 同样观察)。

**Selection Schedule**:周期性选择 + 训练。每个 candidate pool(120 样本)选 $k=12$ 个,每 $T \in \{1, 10, 50\}$ 次迭代做一次 training step,reset selection buffer。Default $T=10$。

这个 schedule 平衡 efficiency 和 performance,类似 LEAD (Lin et al. 2025, https://arxiv.org/abs/2505.07437) 的 iterative 思路。

### 4.4 AdaFisher for Efficiency (Appendix A)

传统 FIM 复杂度 $O(d^2)$ 存储,$O(d^3)$ inversion。AdaFisher 用 **diagonal block-Kronecker approximation**:
$$\hat{F}_i = H_{i-1} \otimes S_i$$

其中:
- $H_{i-1} = \mathbb{E}[\bar{h}_{i-1} \bar{h}_{i-1}^\top]$:activation 二阶统计,$\bar{h}_{i-1} = [h_{i-1}^\top, 1]^\top$ 是 augmented activation
- $S_i = \mathbb{E}[s_i s_i^\top]$:pre-activation gradient 二阶统计
- $\otimes$:Kronecker product

对角近似:
$$\tilde{F}_i^D = \text{Diag}(H_{i-1}) \otimes \text{Diag}(S_i) + \lambda I$$

复杂度:
- Storage: $\sum_i (n_{i-1} + n_i) = O(d)$
- Inversion: $O(d)$(element-wise reciprocal)
- Preconditioning: $O(d)$

总 per-iteration: $O(d)$,从二次降到线性。这是 K-FAC (Martens & Grosse 2020, https://arxiv.org/abs/1503.05671) 的简化版,牺牲精度换效率。

参考 AdaFisher (Gomes et al. 2025, https://arxiv.org/abs/2405.16397)。

---

## 5. 主实验结果

### 5.1 Setup

- Training data:97.5K samples,spanning GSM8K(数学), Alpaca Code(代码), ShareGPT + Alpaca(通用)
- Models:LLaMA2-7B, Qwen2-7B
- Fine-tuning:LoRA(r=16, α=32, dropout=0.05),8×H20 GPU
- Budget:10% data
- Baselines:Full, Random, IFD, Fisher, LESS, SelectIT, DPP, TSDS, LEAD

### 5.2 Qwen2-7B Main Results (Table 1)

| Benchmark | Full 100% | Random 10% | Fisher 10% | LESS 10% | SelectIT 10% | DPP 10% | LEAD 10% | **SPICE 10%** | Δ |
|-----------|-----------|------------|------------|----------|--------------|---------|----------|--------------|-----|
| GSM8K | 84.2 | 84.9 | 86.5 | 82.3 | 83.6 | 86.5 | 84.5 | **86.7** | +2.5 |
| BBH | 61.3 | 60.8 | 60.8 | 61.0 | 60.9 | 61.0 | 61.0 | 61.0 | -0.3 |
| MMLU | 65.7 | 63.4 | 65.2 | 66.1 | 64.7 | 66.0 | 66.0 | **67.1** | +1.4 |
| ARC-C | 50.5 | 49.6 | 50.3 | 49.5 | 50.3 | 51.0 | 50.8 | **51.8** | +1.3 |
| TruthfulQA | 54.8 | 54.8 | 55.0 | 54.3 | 54.4 | 55.0 | 54.9 | **55.5** | +0.7 |
| IFEval | 33.5 | 28.3 | 30.6 | 26.0 | 30.8 | 35.4 | 33.0 | **38.6** | +5.1 |
| HumanEval | 45.7 | 45.7 | 44.5 | 46.3 | 46.3 | 45.0 | 46.1 | **47.1** | +1.4 |
| MBPP | 55.2 | 56.1 | 54.6 | 56.2 | 55.2 | 55.7 | 55.6 | 56.2 | +1.0 |
| **Average** | 56.4 | 55.5 | 55.9 | 55.2 | 55.8 | 57.0 | 56.5 | **58.0** | +1.6 |

亮点:
- SPICE 用 10% 数据,**7/8 benchmark 最优**,IFEval 提升 +5.1
- 平均 58.0 vs Full 56.4(+1.6),vs 第二名 DPP 57.0(+1.0)
- LLaMA2-7B(Table 2)平均 31.1 vs Full 30.8(+1.8)

### 5.3 Cost Analysis (Table 15, Appendix H.8)

| Method | Selection Time | Complexity | Qwen2-7B | LLaMA2-7B |
|--------|----------------|------------|----------|-----------|
| Full | 00:00 | O(0) | 56.4 | 30.8 |
| Random | 00:00 | O(k) | 55.5 | 30.1 |
| **SPICE** | **02:56** | **O(k\|D\|d)** | **58.0** | **31.1** |
| IFD | 04:32 | - | 55.4 | 30.0 |
| LESS | 16:22 | O(Nm\|D\|d) | 55.0 | 30.3 |
| Fisher | 17:01 | - | 55.2 | 30.2 |
| SelectIT | 25:19 | - | 55.8 | 30.1 |
| DPP | 23:32 | O(NMD + ND) | 56.3 | 30.5 |
| TSDS | 00:05 | O(ML log N) | 55.4 | 29.9 |
| LEAD | 12:02 | - | 56.5 | 31.0 |

SPICE 总 selection + training time 20 GPU-hours,**比 Full-data LoRA 还低**。对比 SelectIT 25 小时,Fisher 17 小时,SPICE 仅约 3 小时 selection。

### 5.4 Ablation: λ Sensitivity (Figure 3b)

- $\lambda = 0$(无 penalty):性能显著下降
- $\lambda \in [0.1, 0.5]$:稳定高表现
- $\lambda = 1.0$:略降,penalty 过强

Default $\lambda = 0.1$。

### 5.5 Ablation: Proxy Model & Step Interval (Figure 3c, Table 10, 11)

| Step Interval | Qwen2-0.5B | Qwen2-1.5B | Qwen2-7B | LLaMA2-7B |
|---------------|------------|------------|----------|-----------|
| 1 | 67.0 | 67.1 | 67.2 | 65.9 |
| 5 | 67.0 | 66.9 | 67.2 | 66.1 |
| 50 | 66.6 | 66.9 | 67.1 | 66.1 |
| Average | 66.9 | 66.8 | 67.1 | 66.1 |

观察:
- Qwen2 系列 proxy:0.5B vs 7B 几乎相同(67.0 vs 67.2),用 small proxy 完全够
- LLaMA2-7B proxy for Qwen2 target:65.9-66.1,明显低,说明 **cross-architecture transfer 有限**
- 70B target model 用 0.5B proxy 仍工作(Table 11)

### 5.6 Diversity Analysis (Table 3)

| Method | LDD↑ | NovelSum↑ | Code | Math-R | General |
|--------|------|-----------|------|--------|---------|
| Random | -9.5 | 30.3 | 5% | 10% | 10% |
| Fisher | 19.4 | 40.3 | 12% | 2% | 9% |
| LESS | 4.9 | 38.7 | 1% | 12% | 11% |
| SelectIT | 17.6 | 39.0 | 8% | 13% | 9% |
| TSDS | -1.8 | 34.8 | 4% | 5% | 10% |
| DPP | 31.1 | 42.5 | 7% | 7% | 9% |
| LEAD | 9.8 | 37.5 | 2% | 6% | 11% |
| **SPICE** | 22.0 | 41.3 | 10% | 8% | 9% |

- SPICE 的 NovelSum/LDD 超过 Random 和多数 baselines,接近 DPP
- Domain coverage 与 full corpus 接近(10%/8%/9% vs 实际比例),保持平衡

### 5.7 Budget 扩展实验 (Table 14, Appendix H.6)

Qwen2-7B 上:
| Budget | Method | Avg |
|--------|--------|-----|
| 0% | Null | 53.6 |
| 1% | SPICE | 54.2 |
| 5% | SPICE | 56.5 |
| 10% | SPICE | 58.0 |
| ~30% | SPICE+ | 58.1 |
| 100% | Full | 56.4 |

观察:
- 1% SPICE 已经超过 Null,接近 Full
- 5% → 10% 提升明显(56.5 → 58.0)
- 10% → 30% 边际收益小(58.0 → 58.1),印证 early stopping intuition
- 10% SPICE 已经超过 100% Full,说明 information-based selection 在 small budget 比 budget expansion 更有效

### 5.8 SPICE+ Early Stopping (Table 9)

| ω | Data Rate | Average | Time Cost |
|------|-----------|---------|-----------|
| 0.1 | 55.1% | 67.1 | 9:55 |
| 0.3 | 34.5% | 67.2 | 7:01 |
| 0.5 | 26.1% | 67.2 | 5:49 |
| 0.7 | 9.3% | 67.1 | 2:45 |

$\omega = 0.5$ 最佳,选 26% 数据,5:49 时间,与 SPICE 10% 性能几乎相同(67.2 vs 67.1)。$\omega = 0.1$ 时选 55%,验证 marginal gain 后期衰减快。

---

## 6. Selected Data 分析

### 6.1 Overlap with Baselines (Table 12)

| Baseline | Jaccard | Overlap@Ours | Overlap@Base |
|----------|---------|--------------|--------------|
| Fisher | 0.47 | 0.64 | 0.64 |
| Random | 0.05 | 0.10 | 0.09 |
| IFD | 0.02 | 0.03 | 0.03 |
| SelectIT | 0.08 | 0.15 | 0.15 |
| LESS | 0.01 | 0.01 | 0.02 |

SPICE 与 Fisher overlap 最高(Jaccard 0.47,双向 overlap 0.64),验证 SPICE 保留了 Fisher 的"信息核心",差异在 marginal/boundary 样本上(冲突高的)。

### 6.2 Cluster Coverage (Figure 9)

1000-way clustering,SPICE 覆盖 92.1% clusters,接近 Random 92.6%。即使 aggressive compression,broad semantic dispersion 仍保持。

### 6.3 Case Study (Table 13)

Excluded (E,高 conflict 被剔除):
- "Given list of words, come up with sentence..."(creative,可能与其他样本方向冲突)
- "What is a linked list?"(基础,可能 redundancy)
- "Select rows in pandas..."(简单代码)

Shared (S,与 Fisher 共享):
- 算术题 "(4+7)/2*8-3="
- 词义生成题
- capitalize string

显示 SPICE 在简单 redundant 样本上更挑剔,在核心信息样本上与 Fisher 一致。

---

## 7. 我的 Intuition 总结

### 7.1 为什么 SPICE 工作?三个层次

1. **几何层**:log-det FIM 度量 gradient 张成的"信息体积"。Greedy 选 high-norm samples 会重复覆盖相同方向,造成冗余。Conflict-aware penalty 实质上鼓励选 orthogonal 方向,与 DPP 的 kernel intuition 类似但更精细——DPP 鼓励全局 diversity,SPICE 只惩罚与当前 mean 反向的方向。

2. **优化层**:greedy 在 submodular function 上的质量由 curvature 控制。Curvature 由 perturbation 控制,perturbation 由 gradient inner products 平方控制。所以控制 inner product 直接改善 greedy bound。这给出 principled 的 selection criterion,而非 heuristic。

3. **训练动力学层**:SGD update 是 mini-batch gradients 的累加。如果 batch 内 gradients 互相 conflict,实际 update direction 偏离单样本方向,有效信息被"抵消"。SPICE 在 selection 阶段就避免这种抵消,等效于更稳定的训练。

### 7.2 关键 theoretical contribution

不是简单说"gradient conflict 不好",而是**定量链接**:
$$\text{Conflict} \nearrow \Rightarrow \sum_{y \in S} (g_x^\top g_y)^2 \nearrow \Rightarrow |\varepsilon_x(S)| \nearrow \Rightarrow c \nearrow \Rightarrow \frac{1-e^{-c}}{c} \searrow$$

整条因果链从 empirical gradient statistics 到 approximation factor,中间用 ε-decomposition 作桥梁。这是 paper 真正的 intellectual contribution。

### 7.3 与相关工作的对比

| 方法 | Selection criterion | 考虑 conflict? | 理论保证 |
|------|---------------------|----------------|----------|
| Random | 无 | 否 | 无 |
| IFD | instruction difficulty | 否 | 无 |
| Fisher / FisherSFT | log-det FIM | 否 | $(1-1/e)$ |
| LESS | gradient similarity to val | 隐式(用 cosine) | 无 |
| SelectIT | uncertainty via LLM | 否 | 无 |
| DPP | diversity kernel | 隐式 | DPP sampling |
| TSDS | task-conditioned | 否 | 无 |
| LEAD | iterative | 隐式 | 无 |
| **SPICE** | **log-det FIM - λ·conflict** | **显式** | **$(1-e^{-c})/c$ with data-dependent $c$** |

SPICE 是第一个**显式把 gradient conflict 加入 submodular selection theory** 的方法,并给出 data-dependent approximation factor。

### 7.4 局限性与未来方向

1. **Conflict 定义简化**:只用 cosine vs mean gradient,未考虑 high-order interactions(三样本互冲突等)。可扩展到 subspace alignment。
2. **Proxy transfer 限制**:cross-architecture(LLaMA proxy for Qwen target)有限。需要 architecture-invariant features。
3. **α 调参**:$\alpha$ 与 $\rho$ 假设需 empirically 验证,high-conflict data 上违反 $\alpha \|\mathbf{F}_S\| < 1$。
4. **Scale 到 100B+**:虽然 70B 实验有效,但 proxy assumption 在极大规模可能失效。
5. **Multimodal / RL 扩展**:paper 末尾提到,gradient conflict 在 RLHF 中同样存在(PCGrad 类工作),SPICE framework 可直接迁移。
6. **Online selection**:目前用 fixed checkpoint,可结合 LEAD 的 iterative update,让 selection 随 training 演化。

### 7.5 Code-level 联想

实现 SPICE 的核心代码(基于 Algorithm 1)伪代码:

```python
def SPICE_select(D, k, lambda_, proxy_model):
    S = []
    gradients = {i: compute_gradient(proxy_model, D[i]) for i in range(len(D))}
    for t in range(k):
        mean_g = mean([gradients[i] for i in S]) if S else None
        scores = {}
        for x in D_setminus_S:
            delta = fisher_marginal(g_x, S, gradients)
            if mean_g is not None:
                conflict = max(0, -cosine(g_x, mean_g))
            else:
                conflict = 0
            scores[x] = delta - lambda_ * conflict
        x_star = argmax(scores)
        if delta_x_star <= omega * delta_first:
            break
        S.append(x_star)
    return S
```

AdaFisher 的对角近似让 fisher_marginal 复杂度从 $O(d^2)$ 降到 $O(d)$:
```python
def fisher_marginal(g_x, S, gradients, alpha):
    # 对角近似: F_S ≈ diag(sum_i g_i^2)
    diag_F = sum(g_i ** 2 for i in S)
    inv_term = 1.0 / (1.0 + alpha * diag_F)
    return log(1 + alpha * sum(g_x ** 2 * inv_term))
```

这与 K-FAC 的实现思路一致,但更激进(完全对角化),适合 selection 场景(不需要精确 optimization)。

---

## 8. 参考文献与延伸阅读

### 核心理论
- **Submodular functions**: Lovász 1983, https://link.springer.com/chapter/10.1007/978-3-642-68874-4_10
- **Greedy $(1-1/e)$ bound**: Nemhauser-Wolsey-Fisher 1978, https://link.springer.com/article/10.1007/BF01588971
- **Curvature refinement**: Conforti & Cornuéjols 1984, https://www.sciencedirect.com/science/article/pii/0166218X84900039
- **Multilinear extension**: Feige & Vondrák 2010, https://theoryofcomputing.org/articles/v006a011
- **Dependent rounding**: Ageev & Sviridenko 2004, https://link.springer.com/article/10.1023/B:JOCO.0000038913.96607.c2; Gandhi et al. 2006, https://dl.acm.org/doi/10.1145/1147954.1147956
- **Fisher Information**: R.A. Fisher 1922, http://www.jstor.org/stable/91208

### Data Selection for LLM
- **LESS**: Xia et al. 2024, https://arxiv.org/abs/2406.06046 (ICML'24)
- **SelectIT**: Liu et al. 2025, https://arxiv.org/abs/2402.16705
- **FisherSFT**: Deb et al. 2025, https://arxiv.org/abs/2505.14826
- **DPP for data selection**: Zhang et al. 2023, https://arxiv.org/abs/2303.17358
- **IFD**: Li et al. 2024, https://aclanthology.org/2024.naacl-long.421
- **TSDS**: Liu et al. 2024, https://arxiv.org/abs/2410.11303
- **LEAD**: Lin et al. 2025, https://arxiv.org/abs/2505.07437
- **Survey**: Albalak et al. 2024, https://arxiv.org/abs/2402.16827

### Gradient Conflict in Multi-task Learning
- **PCGrad**: Yu et al. 2020, https://arxiv.org/abs/2001.06782
- **CAGrad**: Liu et al. 2024, https://arxiv.org/abs/2110.14048
- **Recon**: Shi et al. 2023, https://arxiv.org/abs/2302.11289

### Efficient Optimization
- **K-FAC**: Martens & Grosse 2020, https://arxiv.org/abs/1503.05671
- **AdaFisher**: Gomes et al. 2025, https://arxiv.org/abs/2405.16397
- **LoRA**: https://arxiv.org/abs/2106.09685

### Benchmarks
- GSM8K: Cobbe et al. 2021, https://arxiv.org/abs/2110.14168
- BBH: Suzgun et al. 2022, https://arxiv.org/abs/2210.09261
- MMLU: Hendrycks et al. 2021, https://arxiv.org/abs/2009.03300
- ARC: Clark et al. 2018, https://arxiv.org/abs/1803.05457
- TruthfulQA: Lin et al. 2022, https://aclanthology.org/2022.acl-long.229
- IFEval: Zhou et al. 2023, https://arxiv.org/abs/2311.07911
- HumanEval: Chen et al. 2021, https://arxiv.org/abs/2107.03374
- MBPP: Austin et al. 2021, https://arxiv.org/abs/2108.07732

---

## 9. 最终 Intuition:一图概括

```
Gradient Conflict (cosine to mean)
       ↓
(g_x^T g_y)^2 ∑ grows
       ↓
|ε_x(S)| grows (perturbation)
       ↓
Curvature c grows
       ↓
(1-e^{-c})/c shrinks → greedy bound weakens
       ↓
Fisher greedy 选的 subset 质量 ↓
       ↓
LLM training 性能 ↓

SPICE Intervention:
score = Δ_x - λ·conflict
       ↓
Control |ε_x(S)| → Reduce c → Tighten bound
       ↓
Same budget, higher F(S_greedy)/F(S*)
       ↓
Better downstream performance
```

这条因果链是 SPICE 的灵魂:从 empirical gradient statistics 出发,经过 ε-decomposition 桥接,到 submodular curvature 理论,再到 greedy approximation quality,最后到 LLM training 实效。**Theory guides method,method validates theory**,paper 在这点上做得很完整。

值得 follow-up 的方向包括 conflict 的 high-order 度量、architecture-invariant proxy、与 RLHF 的结合、以及把 ε-decomposition 推广到其他 submodular selection 问题(如 active learning、core-set selection in vision)。
