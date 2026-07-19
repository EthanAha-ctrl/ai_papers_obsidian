---
source_pdf: A Mechanistic Analysis of Sim-and-Real Co-Training.pdf
paper_sha256: 161bd8aa9bfecc29d806a776c7852710f5f7ba21d0591475ede7f3e725952405
processed_at: '2026-07-17T20:26:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A Mechanistic Analysis of Sim-and-Real Co-Training — 详细解读

这篇paper来自UT Austin的Yuke Zhu组 (arXiv:2509.18631, 作者Yu Lei, Minghuan Liu等), 试图打开sim-and-real co-training这个"black box", 从mechanistic的角度解释为什么co-training有效, 什么时候有效, 以及如何更好地设计co-training算法。

项目主页: https://science-of-co-training.github.io/
相关参考: 
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- MimicGen: https://mimicgen.github.io/
- Robosuite: https://robosuite.ai/
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598

---

## 1. 核心Insight: 两种内在效应

这篇paper最核心的贡献是识别出co-training背后有两个独立的内在效应:

**Primary Effect: Structured Representation Alignment**
这又包含两个互补的属性:
- **Representation Alignment**: source domain (sim) 和 target domain (real) 的observation representation在某个domain-invariant subspace里对齐, 这样task-relevant knowledge才能transfer
- **Domain Discernibility**: representation同时要保留domain-specific的信息, 让action能够adapt到real world, 而不是直接从sim复制过来

这两个属性构成一个balance, 缺一不可。

**Secondary Effect: Importance Reweighting**
这是由data mixing ratio $w$ 引起的, 通过domain-dependent logit modulation, 控制每个domain的training sample对action decision的贡献。这个effect是在primary effect基础上的微调。

---

## 2. Theoretical Analysis — 理论推导

### 2.1 Learning Objective

给定target domain (real) dataset $\mathcal{D}_T = \{(o_i, a_i)\}_{i=1}^N$ 和source domain (sim) dataset $\mathcal{D}_S = \{(o_j, a_j)\}_{j=1}^M$, 其中 $M \gg N$, 用mixing ratio $w$ co-training一个diffusion policy, learning objective为:

$$\mathcal{L}_w(t; \phi, \theta) := w \cdot \mathcal{L}_{\mathcal{D}_T} + (1-w) \cdot \mathcal{L}_{\mathcal{D}_S}$$

其中 $\mathcal{L}_{\mathcal{D}} = \mathbb{E}_{(o_i, a_i) \sim \mathcal{D}, \epsilon \in \mathcal{N}(\mathbf{0}, \mathbf{I}_d)}[\|\epsilon - \epsilon_\theta(a^t, t, o)\|_2^2]$

**变量解释**:
- $\phi$: feature encoder $f_\phi: \mathcal{O} \to \mathcal{Z}$ 的参数, 把observation映射到latent space
- $\theta$: policy model $\pi_\theta: \mathcal{Z} \to \mathcal{A}$ 的参数
- $w$: mixing ratio, target domain的权重, 取值(0,1)
- $t$: diffusion timestep
- $a^t$: 在timestep $t$ 加噪后的action
- $\epsilon$: 标准Gaussian噪声 $\sim \mathcal{N}(0, \mathbf{I}_d)$, $d$ 是action维度
- $\epsilon_\theta$: 神经网络预测的noise

### 2.2 Score Function的解析解

paper证明这个objective存在analytical optimal solution (采用score parameterization):

$$s_w^*(a^t, t, o) = \hat{w}_t \cdot s_t^*(a^t, t, o) + \hat{w}_s \cdot s_s^*(a^t, t, o)$$

**关键公式 — 动态权重**:

$$\hat{w}_t = \frac{w \cdot p_t(a^t, f_\phi(o))}{w \cdot p_t(a^t, f_\phi(o)) + (1-w) \cdot p_s(a^t, f_\phi(o))}$$

其中:
$$p_k(a^t, z) = \frac{1}{|\mathcal{D}_k|} \sum_{i \sim \mathcal{D}_k} p(a^t | a_i^0) \cdot K(z, z_i), \quad k \in \{t, s\}$$

**变量解释**:
- $s_w^*$: co-trained model的optimal score function
- $s_t^*, s_s^*$: 分别是只用target domain和只用source domain训练的optimal score function
- $\hat{w}_t, \hat{w}_s$: 动态权重, 加起来等于1, 决定每个domain的贡献
- $p_k(a^t, z)$: domain $k$ 中给定observation representation $z = f_\phi(o)$ 时action $a^t$ 的边缘概率
- $K(\cdot, \cdot)$: kernel function, 衡量当前observation和dataset中observations的相似度
- $z_i = f_\phi(o_i)$: 第 $i$ 个样本的observation representation

**Intuition**: 这个公式说明co-trained model的最优解是两个domain各自最优解的**加权平均**, 但是权重 $\hat{w}_t$ 是**动态的**, 依赖于当前的observation representation。如果representation学得好, 同一个task的sim和real observations会靠近, 那么 $p_s(a^t, z)$ 也会对real observation有响应, 从而实现knowledge transfer。

### 2.3 三种Representation Alignment Scenario

基于representation alignment的程度, paper假设三种scenario:

**Scenario 1: Disjoint**
sim和real的representation完全分到不同的cluster。inference时real observation的 $p_s(a^t, z) \approx 0$, 所以 $\hat{w}_t \approx 1$, policy完全忽略source domain → **无positive transfer**

**Scenario 2: Structured Aligned** (sweet spot)
学到了task-relevant, domain-invariant的representation, 同时保留domain-specific信息。sim和real的representation靠近但没collapse。Action prediction被source domain的neighbors引导, 但主要由target domain主导 → **adaptive action transfer**

**Scenario 3: Overlapping**
sim和real的representation完全对齐, 但由于domain gap, 对应的actions不同。Policy不知道实际环境, 在source和target actions上呈bimodal distribution → **negative transfer**

### 2.4 Importance Reweighting Effect

对特定observation $o$, 公式简化为:

$$\hat{w}_t := \frac{w \cdot p_t(a^t)}{w \cdot p_t(a^t) + (1-w) \cdot p_s(a^t)}$$

**不同timestep的行为**:
- **大 $t$ (强噪声)**: $p_t(a^t) \approx p_s(a^t)$, model近似两个domain的global average
- **小 $t$ (弱噪声)**: $p(a^t)$ 集中到一个domain, $\hat{w}_t \approx 1$, model收敛到specific domain

定义normalized distance: $r_k(a^t, t) := \frac{\|a^t - \alpha_t a_k\|}{\sigma_t \sqrt{d}}$

进一步简化后的merged score function:

$$s_w^*(a^t, t) = \sum_{i_t}^{N} g_{i_t} s_{i_t}^* + \sum_{i_s}^{M} g_{i_s} s_{i_s}^*$$

$$g_{i_k} = \text{Softmax}(\ln(w_k) - r_k^2(a^t, t) \cdot d/2), \quad k \in \{t, s\}$$

$$w_t = w/N, \quad w_s = (1-w)/M$$

**变量解释**:
- $g_{i_k}$: 每个training sample的posterior weight, 通过softmax计算
- $w_k$: per-sample mixing weight, target domain每个sample权重 $w/N$, source domain每个sample权重 $(1-w)/M$
- $r_k$: normalized distance, 衡量当前noisy action $a^t$ 到训练样本 $\alpha_t a_k$ 的距离, 除以 $\sigma_t \sqrt{d}$ 做normalization
- $\alpha_t, \sigma_t$: forward diffusion process的参数, $\alpha_t$ 控制signal保留, $\sigma_t$ 控制noise尺度

**关键insight**: 这个softmax形式的weight说明, training samples的contributions由两部分决定: (1) $\ln(w_k)$ 是mixing ratio的对数; (2) $r_k^2 \cdot d/2$ 是距离惩罚。由于高维空间Gaussian concentration, 距离惩罚极陡, softmax权重极度偏向最近的training sample。

**Special case — relative weight ratio**:

$$\frac{g_{i_t}}{g_{i_s}} \propto \mathcal{F}\left(\frac{w}{1-w}, \frac{M}{N}, |a_{i_t} - a_{i_s}|\right)$$

这说明modulation的amplitude同时受mixing ratio $w$, dataset size ratio $M/N$, 和domain gap $|a_{i_t} - a_{i_s}|$ 影响。

---

## 3. Controlled Toy Example — 验证理论

### 3.1 实验设计

- Policy: 4-layer MLP作为diffusion model
- Task: 学习映射 $\pi(y|x): \mathbb{R}^3 \to \mathbb{R}^2$
- 两个人工manifold $\mathcal{M}_S$ (source) 和 $\mathcal{M}_T$ (target)
- $N_S \gg N_T$, target domain部分采样
- 沿两个principal方向对齐两个manifold, 第三个方向改变距离 → 模拟不同alignment scenario

### 3.2 两个Key Findings

**Finding 1**: Toy model行为与理论分析一致
- Disjoint: 预测接近只用target训练的model, 但因数据少而memorize, 无法interpolate/extrapolate
- Structured aligned: sweet spot, 高fidelity重建output distribution
- Overlapping: 预测在source和target间随机分布, 无法有效distinguish domains

**意外发现**: 适当的co-training设置能让model在OOD region做出合理预测 → **OOD generalization**。这个能力不是简单复制source knowledge, 而是来自preserved distribution shift in representations。

**Finding 2**: Structured representation alignment是dominant driver

ANOVA variance decomposition分析:
- Structured representation alignment解释 ~**50%** 的loss variance
- Importance reweighting (mixing ratio)只解释 ~**20%**

**结论**: representation alignment是primary determinant, $w$ 只是modulation factor。

---

## 4. Sim-and-Real Experiments — 真实机器人验证

### 4.1 实验设置

**Tasks** (来自robosuite):
- NutAssembly: 精确控制, dense object interaction
- MugHang: 需要rotation motion, 更具挑战性
- MugCleanup: long-horizon reasoning和execution

**Domain gap分解**:
- Visual appearance gap
- Environment physics gap (mass, friction, size)

**三种sim-and-sim设置**:
- Visual-only
- Physics-only
- Visual-physics (vis-phys)

**Data**:
- Target: 50 human demonstrations per task
- Source: MimicGen生成 ~3000 trajectories
- Natural mixing ratio: $w_n = \frac{|D_r|}{|D_r| + |D_s|} = 0.016$
- Sweep $w \in \{0, 0.005, 0.016, 0.1, 0.3, 0.5, 0.8, 1\}$

### 4.2 Observation 1: Representation Alignment可隐式学习

用UMAP可视化不同layer的latent embeddings:
- **Vision stem后**: local geometry alignment, 相似geometric structures
- **Encoder trunk $f_\phi$ 后**: global representation alignment

**定量测量**:
- **Gromov-Wasserstein distance**: 衡量local geometric similarity
- **Wasserstein distance**: 衡量global distributional distance

**发现**: smaller distances → stronger alignment, 且与mixing ratio强相关。这说明mixing ratio不只是re-weight数据, 而是**隐式reshape representation space**。

### 4.3 Observation 2: Alignment与Performance正相关

计算log-transformed Wasserstein distance与success rate的correlation:
- **Pearson和Spearman系数**: 0.6~0.8, p-value < 0.04
- **例外**: physics-only setting, correlation可能为负!

**physics-only的negative correlation**: 当物体物理参数差异大而视觉相似时, policy难以distinguish两个环境。Blind representation alignment反而有害。

### 4.4 Observation 3: Domain Discernibility不可或缺

**Linear probing实验**: 训练2-layer MLP做binary domain classification, 即使representations看似aligned, MLP仍能达到 ~100% accuracy。

**结论**: representations处于**partially aligned**状态, co-training保留了domain-specific信息。

**Ablation验证**: 在vis-phys setting, 故意remove gradient reversal layer, 促进domain-discriminative features:

| Method | NutAssembly | MugCleanup | MugHang | Avg |
|--------|-------------|------------|---------|-----|
| co-training | 0.85 | 0.44 | 0.27 | 0.52 |
| +discrimination | 0.79 | 0.415 | 0.215 | 0.47 |

**性能下降** → 确认discernibility的causal effect。

---

## 5. Unified View of Co-Training Methods

### 5.1 三种代表性方法的分析

**Optimal Transport (OT) based methods**:
$$\min_{\phi, \theta} \mathcal{L}_w(\phi, \theta) + \lambda \cdot \mathcal{L}_{OT}(\mathcal{D}_r, \mathcal{D}_s)$$

- 用Wasserstein distance显式匹配distribution
- **强调**: representation overlap → 强alignment
- **风险**: domain gap大时可能negative transfer

**Adversarial Domain Adaptation (ADDA)**:
$$\min_{\phi, \theta} \mathcal{L}_w(\phi, \theta) + \lambda \cdot \mathcal{L}_{disc}(\mathcal{D}_r, \mathcal{D}_s)$$

- 训练discriminator区分domains, encoder试图fool discriminator
- **强调**: representation indistinguishability
- **与OT类似**: 优先cross-domain overlap

**Classifier-Free Guidance (CFG)**:
$$\tilde{s}_\theta(a, o, c, t) = (1+\lambda) \cdot s_\theta(a, o, c, t) - \lambda \cdot s_\theta(a, o, \emptyset, t)$$

**变量解释**:
- $c$: environment label (one-hot embedding)
- $\emptyset$: dropped label (unconditioned)
- $\lambda$: guidance scale

- **强调**: domain discernibility, 通过separate conditional pathways保留domain awareness
- **优势**: flexible information sharing

### 5.2 CFG-ADDA: 简单组合方法

**核心思想**: 用CFG保留domain-specific信息, 用ADDA促进domain-invariant部分alignment。

**实现**:
- One-hot embeddings $c$ 作为environment labels加到observation features后
- 剩余representation dimensions通过adversarial discriminator alignment
- 训练时以probability $p=0.2$ 随机drop environment labels

**关于 $\lambda$ 的新视角**:
当 $\lambda < 0$ 时, $s_\theta(a, o, \emptyset, t)$ 代表所有domain的average log-probability gradient direction, 所以负 $\lambda$ 主动transfer "averaged knowledge" from surrogate domains during inference。

### 5.3 实验结果

**Sim-and-Sim结果** (Fig. 9):

将mixing ratios分两组:
- **Balanced mixing** (0.016, 0.1, 0.3): sim和real比例相当
- **Unbalanced mixing**: 一方dominant

| 方法 | Balanced | Unbalanced |
|------|----------|------------|
| Co-Training | baseline | baseline |
| +OT | 提升 | 退化 |
| +ADDA | 提升 | 退化 |
| +CFG | 有限提升 | 鲁棒 |
| **+CFG-ADDA** | **强提升** | **鲁棒** |

**Real-World结果** (Table 2):

| Method | NutAssembly | MugCleanup | MugHang | Avg |
|--------|-------------|------------|---------|-----|
| Real-only | 11/30 | 8/30 | 6/30 | 8.6/30 |
| Co-Training ($w=0.016$) | 17/30 | 16/30 | 8/30 | 15.3/30 |
| +CFG-ADDA ($\lambda=-0.5$) | 23/30 | 11/30 | 18/30 | 21/30 |
| +CFG-ADDA ($\lambda=-0.5$) | | 22/30 | 15/30 | |

CFG-ADDA在real world达到 **~74% success rate**, 相比prior methods有显著提升。

### 5.4 Ablation on Guidance Scale $\lambda$

sweep $\lambda \in (-2, 2)$:
- CFG-ADDA consistently优于CFG
- 两者在 $\lambda = -0.5$ 都有提升
- **建议**: 用 $\lambda < 0$ 主动transfer knowledge from surrogate domains during inference

---

## 6. Mixing Ratio Selection Guideline

基于理论分析, paper提供formal guideline (Algorithm 2):

**输入**: source dataset size $N$, target dataset size $M$ ($M > N$)
**输出**: narrowed search range $(w_n, w_q)$

1. 计算natural mixing ratio: $w_n = \frac{N}{N+M}$
2. 用 $w_n$ 作为lower bound
3. 如果 $M/N > 5$:
   - Upper bound: $w_q = \sqrt{\frac{N}{M}}$
4. 否则:
   - 设定desired target contribution $q$ (e.g., 0.8)
   - Upper bound: $w_q = \frac{N \cdot q}{(1-q) \cdot M + N \cdot q}$
5. Domain gap大时, 适当上调 $w_n, w_q$
6. 在 $(w_n, w_q)$ 内search

**推导intuition**: balanced mixing ratio应让real和sim都有贡献, 但real权重更大。相对domain weight $\frac{g_r}{g_s}$ 应在 $(1, \sqrt{M/N})$ 之间。

---

## 7. Three Regimes Formal Definition

paper给出structured representation alignment的数学定义:

$$\mathcal{SRA}(p_s, p_t) = (\mathcal{M}_{align}, \mathcal{D}_{disc})$$

其中:
- $\mathcal{M}_{align} = \mathcal{W}(p_s(z), p_t(z))$: Wasserstein distance衡量alignment
- $\mathcal{D}_{disc} = \frac{1}{2} \sum_{k \in \{s,t\}} \mathbb{E}_{z \sim p_k(z)}[\max_{a^t} p_k(a^t | z)]$: 衡量discernibility

**三种regimes**:
- **Disjoint**: high WD, high discernibility
- **Structured aligned**: lower WD, high discernibility (sweet spot)
- **Overlapping**: low WD, low discernibility

实验验证(Fig. 16): 用discriminator loss weights $\{0, 0.05, 0.5\}$ 控制discernibility, 观察到:
- Overlapping regime: performance与WD的correlation反转
- Disjoint regime: 标准co-training trend (better alignment → better performance)
- Structured aligned regime: positive但较弱的correlation

三种regimes构成一个**U-shape correlation pattern**, 统一在一个underlying mechanism下。

---

## 8. 总结与启示

### 核心insight for building intuition:

1. **Co-training不是简单的data augmentation**: mixing ratio $w$ 同时影响两个内在效应, 不能只调 $w$ 而忽略representation alignment。

2. **Alignment和Discernibility是tension**: 
   - 只追求alignment (OT, ADDA) → 可能negative transfer
   - 只保留discernibility (CFG) → 有限提升
   - 两者balance (CFG-ADDA) → best performance

3. **Score function的softmax形式**: co-trained model的最优解是两个domain各自最优解的动态加权平均, 权重由representation alignment程度和mixing ratio共同决定。

4. **OOD generalization**: 适当的co-training能让model在OOD region做出合理预测, 这来自preserved distribution shift而非简单复制。

5. **Practical guideline**: 
   - 用 $w_n = N/(N+M)$ 作为lower bound
   - 用 $w_q = \sqrt{N/M}$ 作为upper bound
   - 在 $(w_n, w_q)$ 内search
   - Domain gap大时上调

### 开放问题:
- dynamic learning过程中两个effect的interaction
- batch size等practical factors的影响
- representation的intrinsic structure
- 在RL, world modeling等其他settings的适用性

---

## References

- Paper: https://science-of-co-training.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- MimicGen: https://mimicgen.github.io/
- Robosuite: https://robosuite.ai/
- CFG paper: https://arxiv.org/abs/2207.12598
- ADDA paper: https://arxiv.org/abs/1702.05464
- Sim-and-Real Co-Training (Maddukuri et al.): https://arxiv.org/abs/2503.24361
- Empirical Analysis (Wei et al.): https://arxiv.org/abs/2503.22634
- OT-based co-training (Cheng et al.): https://arxiv.org/abs/2509.18631
- UMAP: https://arxiv.org/abs/1802.03426
- Gromov-Wasserstein distance: https://arxiv.org/abs/1107.3170
- Wasserstein distance: https://link.springer.com/article/10.1007/BF00533440
- Gr00t N1: https://arxiv.org/abs/2503.14734
- π0.5: https://arxiv.org/abs/2504.16054

这篇paper的价值在于把co-training这个empirically successful但理论模糊的技术, 通过score function的解析解、toy example验证、和大规模robot实验, 串联成一个统一的explanatory framework, 并inspire出简单的CFG-ADDA方法。对想做sim-to-real transfer或co-training的研究者来说, 提供了清晰的intuition和practical guideline。
