---
source_pdf: Sub-JEPA Subspace Gaussian Regularization for Stable End-to-End World
  Models.pdf
paper_sha256: f88624aa985b9303ddce94a03f8d8bba22f535508e695a3b1b6256c4e7fcc8f6
processed_at: '2026-08-12T11:26:08-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Sub-JEPA 用人话说

## 一句话说清楚

LeWM 要求 latent space 的每个方向都"一样活跃"，但真实任务的 relevant 信息往往只集中在少数几个方向上。Sub-JEPA 说：**别管全部方向了，分成几组，每组内部别塌缩就行，组之间爱怎么相关怎么相关**。

---

## 从头说起

### JEPA 在干嘛

你有个 robot，看到一帧画面 $o_t$，encoder 把它压成一个 192 维向量 $z_t$。然后 predictor 根据 $z_t$ 和 action $a_t$ 预测下一帧的 latent $\hat{z}_{t+1}$。训练就是让 $\hat{z}_{t+1}$ 接近真实的 $z_{t+1}$。

不做 pixel reconstruction，纯 latent space prediction。好处是不浪费 capacity 去建模 pixels 里那些 task-irrelevant 的细节（光照、纹理啥的）。

### 问题：Collapse

JEPA 有个老大难问题。encoder 如果把所有输入都 map 到同一个点，那 predictor 预测 "下一帧还是这个点" 就完美了，loss 直接为 0。这就叫 representation collapse——模型学得"很好"但 representation 完全没用。

### LeWM 的解法

LeWM 说：我强制 $z \sim \mathcal{N}(0, I_{192})$。也就是说 192 个维度互相独立，每个都标准正态。

为什么这能防 collapse？因为 $\mathcal{N}(0, I)$ 是一个 full-rank、non-degenerate 的 distribution。如果所有输入 map 到同一点，covariance 就是 0，跟 $\mathcal{N}(0, I)$ 差十万八千里，regularization loss 会爆。

实现上怎么算这个"像不像 Gaussian"？用 Cramér-Wold theorem：如果一个分布的所有 1D random projections 都是 Gaussian，那它就是 multivariate Gaussian。所以随机采一堆方向，每个方向上做 normality test（Epps-Pulley test），平均起来当 loss。

这个方法 elegant，理论扎实，只有一个问题。

### LeWM 的问题：over-constraint

想想 Two-Room 这个 task。一个 agent 在两个房间之间 navigate，本质上 state 就是 2D position $(x, y)$。intrinsic dimensionality 是 2。

但 LeWM 强制 latent space 是 192 维的 isotropic Gaussian。这意味着什么？covariance matrix 必须是 $I_{192}$，所有 192 个 eigenvalue 都等于 1。

这 192 维里，真正 encode navigation 信息的可能就 2-3 维，剩下 189 维被强制"填满" Gaussian noise。这些多余维度对 task 没用，反而干扰 planning——因为 planner 在 192 维空间里搜路径，比在 2-3 维空间里难多了。

用 effective rank 量化：LeWM 强制 $r_{eff} = 192$，但 task 只需要 $r_{eff} \approx 2$。这就是 mismatch。

### Sub-JEPA 的 insight

关键 observation：**要防 collapse，不需要每个方向都 non-degenerate，只需要每个"子空间"内 non-degenerate**。

具体做法：把 192 维 latent space 用 K 个 random orthogonal projection 分成 K 份，每份 $d_s = 192/K$ 维。在每个 subspace 内施加 $\mathcal{N}(0, I_{d_s})$ 约束。但**不约束 subspaces 之间的关系**。

这放松了什么？看个极端例子。假设 $K=2$，$d_s = 96$。如果 $z = (z_1, z_1)$，其中 $z_1 \sim \mathcal{N}(0, I_{96})$——也就是前后两半完全一样。

- LeWM 看：$\text{Cov}(z)$ 有 96 个 eigenvalue = 2，96 个 eigenvalue = 0。这跟 $I_{192}$ 差太远，loss 爆表。
- Sub-JEPA 看：$P_1 z = z_1 \sim \mathcal{N}(0, I_{96})$ ✓，$P_2 z = z_1 \sim \mathcal{N}(0, I_{96})$ ✓。两个 subspace 都满足约束，loss = 0。

Sub-JEPA 允许这种 cross-subspace redundancy。effective rank 从 192 降到 96，latent space 变 compact 了，但每个 subspace 仍然 non-degenerate，不会 collapse。

实际中 K=32 时，这种放松让 latent space 能自适应到 task 的 intrinsic dimensionality，而不是被强制到 192。

---

## 为什么 Frozen + Orthogonal

两个 design choice 看起来 trivial 但很关键。

### 为什么 Frozen

如果 projection matrices 可训练，encoder 和 projection 会 co-adapt。encoder 学到让 projection 对齐到 "easy" 方向——那些 latent distribution 已经很像 Gaussian 的方向。这样 regularizer 形同虚设。

Ablation 证实：trainable projection 在 Two-Room 上从 95% 掉到 61.67%。

这跟 self-supervised learning 里反复出现的 theme 一样——stop-gradient、EMA target encoder、frozen projector，都是为了避免这种 co-adaptation 导致的 trivial solution。

### 为什么 Orthogonal

如果 projection 不 orthogonal，两个 subspaces 可能 overlap，包含 redundant 信息。极端情况：所有 K 个 projections 都指向同一个方向，那 K 个 "subspace" 约束其实就是一个约束，regularization capacity 浪费了。

Orthogonal 保证每个 subspace 独立、balanced、non-redundant。Ablation：random frozen projection 在 PushT 上只有 13.33%（vs orthogonal frozen 的 89%）。

---

## 实验里最有说服力的东西

### Effective rank 和 success rate 的 correlation

Figure 2 是这篇 paper 最核心的证据。四个 environment 上，从 LeWM 到 Sub-JEPA，effective rank 降得越多，success rate 提升越大。

- Two-Room：rank 降最多，success rate 提升最大（+10.67%）
- Reacher：rank 降最少，success rate 提升最小（+1.33%）

这直接验证了核心 hypothesis：Sub-JEPA 之所以 work，就是因为降低了 spurious high-rank variation，让 latent space 更匹配 task 的 intrinsic dimensionality。

### Latent trajectory straightening

Figure 6。latent trajectory 越直（相邻 velocity 的 cosine similarity 越高），planning 越容易。Sub-JEPA 没有显式优化这个，但自然涌现了更直的 trajectory。

Intuition：full-space Gaussian constraint 在 192 维里强制 isotropy，latent dynamics 被 "撑开" 得太散，trajectory 弯弯绕绕。subspace relaxation 让 dynamics 收缩到低维 manifold 上，trajectory 自然变直。

### Block angle 的反例

Table 4 里 block angle 的 linear probe，Sub-JEPA 反而比 LeWM 差（0.218 vs 0.187）。但 MLP probe 持平。

这是个很 honest 的 finding。rotation 是 non-linear quantity，subspace projection 可能把 angular structure 打散到不同 subspaces，linear decoder 抓不到。但信息没丢，non-linear decoder 能恢复。

这说明 subspace regularization 不是 free lunch——对某些类型的 physical quantities（特别是 rotational 的），可能引入 decodability 的 complexity。

---

## 跟相关工作的关系，简单说

- **LeWM**：直接 predecessor，Sub-JEPA 就是把它的 Gaussian regularization 从 full space 搬到 subspaces
- **VICReg / Barlow Twins**：更早的非 contrastive 方法，用 variance + covariance regularization 防 collapse，但需要调多个敏感 hyperparameters
- **DINO-WM**：用 frozen DINOv2 encoder，完全绕过 collapse 问题，但依赖 pretraining
- **V-JEPA 2**：FAIR 的大规模 video world model，用 EMA target encoder 等 heuristic 防 collapse
- **Sliced Wasserstein**：跟 Sub-JEPA 的 random projection 思路同源，都是用低维 projections 处理高维 distribution matching
- **Manifold hypothesis**：整个 motivation 的理论基础，natural data lie on low-dim manifolds

---

## 一句话总结

Sub-JEPA 说：**LeWM 的 Gaussian prior 没错，但它管得太宽了。别管 192 维全 isotropic，管好每个小 subspace 内别塌缩，让 latent space 自己去找 task 的 intrinsic dimensionality**。结果：更 compact 的 representation、更直的 latent dynamics、更好的 planning。简单、有效、理论 clear。

---

# Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models

## 一、核心 Intuition

这篇 paper 要解决的问题是 JEPA (Joint-Embedding Predictive Architecture) 训练中的 **bias-variance tradeoff**。让我先 build 一下 intuition。

JEPA 的核心是：用 encoder $f$ 把 observation $o_t$ 编码成 latent $z_t$，再用 predictor $P$ 在 latent space 里预测 $\hat{z}_{t+1} = P(z_t, a_t)$。不做 pixel reconstruction。这种设计的好处是把 modeling capacity 集中到 task-relevant 的 abstractions 上，但代价是容易 **representation collapse**——encoder 把所有输入 map 到同一个点，trivially 最小化 prediction loss。

LeWM (LeWorldModel) 用了一个很 elegant 的方案：用 **isotropic Gaussian prior** 约束 latent distribution。基于 Cramér-Wold theorem，只要 embedding 的所有 1D random projections 都服从 $\mathcal{N}(0,1)$，joint distribution 就是 $\mathcal{N}(0, I_D)$。这给了一个 theoretically grounded 的 anti-collapse 机制。

但 Sub-JEPA 的 insight 是：**task-relevant latent representations 通常 lie on low-dimensional manifold**，而 LeWM 的 isotropic Gaussian prior 强制 latent space 在所有 D 个方向上都有相同 variance，这跟 manifold structure 冲突。具体来说，$\mathcal{N}(0, I_D)$ 强制 covariance matrix 的所有 eigenvalues 等于 1，effective rank = D。但如果 task 的 intrinsic dimensionality 是 $d \ll D$，理想的 latent representation 的 effective rank 应该接近 $d$。

Sub-JEPA 的 fix：**把 Gaussian regularization 从 ambient space $\mathbb{R}^D$ 移到 K 个 random low-dimensional subspaces**。每个 subspace 内仍然施加 $\mathcal{N}(0, I_{d_s})$ 约束（保留 anti-collapse 效果），但不要求 cross-subspace independence（放松全局 isotropy）。

---

## 二、Method 技术细节

### 2.1 整体架构

```
o_t ──[encoder f]──> z_t ──[predictor P(z_t, a_t)]──> ẑ_{t+1}
                                                  |
o_{t+1} ──[encoder f]──> z_{t+1} <─── L_pred ───────┘
                      |
                      |──[K frozen projections P_k]──> {z^(k)}
                      |                                 |
                      |                          L_reg (subspace Gaussian)
                      └─────────────────────────────────┘
```

### 2.2 Orthogonal Subspace Projection

给定 latent $z \in \mathbb{R}^D$，引入 K 个 projection matrices：

$$\{P_k \in \mathbb{R}^{d_s \times D}\}_{k=1}^K, \quad d_s = \lfloor D/K \rceil$$

变量含义：
- $D$：latent embedding 维度（paper 里设为 192）
- $K$：subspace 数量
- $d_s$：每个 subspace 的维度
- $P_k$：row-orthonormal projection matrix，即 $P_k P_k^\top = I_{d_s}$

投影操作：
$$z^{(k)} = P_k z \in \mathbb{R}^{d_s}, \quad k = 1, \dots, K$$

**$P_k$ 的构造**：先 sample 一个 random Gaussian matrix，做 QR decomposition 得到 orthonormal basis，取前 $d_s$ 行，然后 transpose。这确保了 row-orthonormality。

**为什么 frozen**：如果 projection 可训练，encoder 和 projection 会 co-adapt——projection 会对齐到让 regularizer 失效的方向，削弱 anti-collapse 效果。Ablation（Table 3）证实了这一点：trainable projection with soft ortho reg 比 frozen orthogonal 差很多（Two-Room: 61.67 vs 95.00）。

### 2.3 Multi-Subspace Gaussian Regularization

这是核心创新点。设 $\mathbf{Z} \in \mathbb{R}^{N \times B \times D}$ 是收集的 latent tensor（N = temporal history, B = batch size）。

**Step 1**：投影到 K 个 subspaces
$$\mathbf{Z}^{(k)} = \mathbf{Z} P_k^\top \in \mathbb{R}^{N \times B \times d_s}$$

**Step 2**：在每个 subspace 内，采样 M 个 random unit vectors $\{u^{(m)}\}_{m=1}^M \subset S^{d_s - 1}$，做 1D projection：
$$z_{n,b}^{(k,m)} = \langle \mathbf{Z}_{n,b,:}^{(k)}, u^{(m)} \rangle$$

这里上标 $(k,m)$ 表示第 k 个 subspace 的第 m 个随机方向，下标 $(n,b)$ 是 temporal index 和 batch index。

**Step 3**：对每个 $(k,m)$ 的 scalar 样本集 $\{z_{n,b}^{(k,m)}\}_{n=1,b=1}^{N,B}$，计算 Epps-Pulley normality statistic：
$$T^{(k,m)} = T\left(\{z_{n,b}^{(k,m)}\}_{n=1,b=1}^{N,B}\right)$$

**Step 4**：平均 over M directions 和 K subspaces：
$$\mathcal{L}_{reg} = \frac{1}{KM} \sum_{k=1}^K \sum_{m=1}^M T^{(k,m)}$$

**总 loss**：
$$\mathcal{L}_{total} = \mathcal{L}_{pred}(\hat{z}_{t+1}, z_{t+1}) + \lambda \mathcal{L}_{reg}(\mathbf{Z})$$

### 2.4 Epps-Pulley Normality Test

这里需要深入讲一下。Epps-Pulley test 是基于 **empirical characteristic function (ECF)** 的 normality test。

给定标准化样本 $x_1, \dots, x_n$（mean=0, var=1），ECF 定义为：
$$\hat{\phi}_n(t) = \frac{1}{n}\sum_{j=1}^n e^{itx_j}$$

标准正态分布的特征函数是 $\phi(t) = e^{-t^2/2}$。

Epps-Pulley statistic 度量 ECF 与正态特征函数的加权距离：
$$T_{EP} \propto \int_{-\infty}^{\infty} |\hat{\phi}_n(t) - e^{-t^2/2}|^2 \, d\mu(t)$$

直觉是：如果样本来自正态分布，ECF 应该接近 $e^{-t^2/2}$，statistic 接近 0。作为 loss，minimizing $T_{EP}$ 就是让 embedding 的 1D projections 越来越 Gaussian。

与 K-S test 或 Shapiro-Wilk 等基于 CDF 的 test 相比，ECF-based test 在 high-dimensional setting 下数值更稳定，且可以通过 Monte Carlo 近似高效计算。

### 2.5 为什么 Sub-JEPA 比 LeWM 放松了约束

这是最关键的 insight，让我用数学说清楚。

**LeWM** 要求 $z \sim \mathcal{N}(0, I_D)$，即：
$$\text{Cov}(z) = I_D \implies \text{所有 eigenvalues} = 1 \implies r_{eff} = D$$

**Sub-JEPA** 要求每个 subspace 的投影 $z^{(k)} = P_k z$ 服从 $\mathcal{N}(0, I_{d_s})$，即：
$$P_k \text{Cov}(z) P_k^\top = I_{d_s}, \quad \forall k$$

但**不要求** cross-subspace 的 covariance 为零。这意味着 $\text{Cov}(z) = \Sigma$ 可以是任何满足 $P_k \Sigma P_k^\top = I_{d_s}$ 的 positive semi-definite matrix。

**关键例子**：假设 $K=2$, $d_s = D/2$，两个 orthogonal projections 把 $\mathbb{R}^D$ 分成两半。如果 $z = (z_1, z_1)$ 其中 $z_1 \sim \mathcal{N}(0, I_{D/2})$，那么：

$$\Sigma = \begin{pmatrix} I_{D/2} & I_{D/2} \\ I_{D/2} & I_{D/2} \end{pmatrix}$$

Eigenvalues：$D/2$ 个值为 2，$D/2$ 个值为 0。Effective rank = $D/2 < D$。

但 $P_1 z = z_1 \sim \mathcal{N}(0, I_{D/2})$，$P_2 z = z_1 \sim \mathcal{N}(0, I_{D/2})$，满足 subspace Gaussian constraint！

**这就是 Sub-JEPA 的 relax 机制**：允许 cross-subspace redundancy，从而让 latent representation lie on lower-dimensional manifold，同时每个 subspace 内仍然是 non-degenerate Gaussian，防止 collapse。

---

## 三、实验结果分析

### 3.1 Planning Success Rate (Table 1)

| Method | Two-Room | Reacher | PushT | OGB-Cube |
|--------|----------|---------|-------|----------|
| PLDM | 97.00 | 78.00 | 78.00 | 65.00 |
| DINO-WM (w/o proprio.) | 100.00 | 79.00 | 74.00 | 86.00 |
| LeWM | 84.33±4.23 | 82.67±4.42 | 84.67±6.53 | 67.33±5.01 |
| **Sub-JEPA** | **95.00±2.76** | **84.00±4.00** | **89.00±5.33** | **76.33±5.99** |

观察：
1. **Two-Room 提升最大**（84.33 → 95.00，+10.67%）：这是 2D navigation task，intrinsic dimensionality 最低，LeWM 的 over-constraint 最严重
2. **OGB-Cube** 也有明显提升（67.33 → 76.33，+9%）：虽然 OGB-Cube 视觉复杂，但 3D manipulation 的 task-relevant structure 仍然是 low-dimensional 的
3. DINO-WM 在 Two-Room 和 OGB-Cube 上仍然领先，因为用了 frozen DINOv2 pretrained features——这说明 pretrained visual features 在视觉复杂任务上仍有优势，但 Sub-JEPA 提供了一个不依赖 pretraining 的强 baseline

### 3.2 Effective Rank 分析 (Figure 2)

Effective rank 定义：
$$r_{eff} = \exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

其中 $\lambda_i$ 是 covariance matrix 的 eigenvalues。这本质上是 normalized eigenvalue spectrum 的 Shannon entropy。

**关键发现**：effective rank 降低的幅度与 planning success rate 提升的幅度高度相关。Two-Room 和 OGB-Cube 的 rank 压缩最大，success rate 提升也最大。这直接验证了 hypothesis：subspace regularization 通过允许 latent space 收缩到 task 的 intrinsic dimensionality 来提升 performance。

### 3.3 Ablation: Number of Subspaces K (Table 2)

固定 $D = 192$，变化 $K \in \{1, 2, 4, 8, 16, 32, 64\}$：

- $K = 1$：$d_s = 192$，相当于 LeWM（加一个 orthogonal transformation）
- $K$ 增大：每个 subspace 变小，约束放松
- $K = 32$：$d_s = 6$，Two-Room 最好（95.00），但 PushT collapse（28.00）
- $K = 64$：$d_s = 3$，太窄了，normality test 统计不稳定

**Tradeoff**：K 大 → 更放松 → 更灵活，但 $d_s$ 太小时 normality estimate 不可靠。这是 bias-variance tradeoff 在 subspace design 上的体现。

PushT 在 $K = 32$ 时 collapse 很有意思——PushT 是 block-pushing task，需要 tightly coupled object-agent interaction，可能需要更高维的 latent structure 来编码这些关系。$d_s = 6$ 的 subspace 太窄，无法稳定估计 normality，反而起不到 regularization 作用。

### 3.4 Projection Strategy Ablation (Table 3)

| Projection | Two-Room | PushT |
|-----------|----------|-------|
| Ortho frozen | 95.00 | 89.00 |
| Random frozen | 53.00 | 13.33 |
| Ortho trainable | 61.67 | 57.00 |

Random frozen 在 PushT 上只有 13.33，说明 non-orthogonal projections 会导致 subspaces 之间有 redundant/unevenly scaled 信息，削弱 regularization 效果。Orthogonality 确保每个 subspace 获得balanced, non-redundant view。

### 3.5 Physical State Probing (Table 4)

在 PushT 上，用 frozen encoder + linear/MLP probe 解码物理变量：

- Agent location：Sub-JEPA 比 LeWM 略好（linear MSE: 0.048 vs 0.052）
- Block location：Sub-JEPA 略好（linear MSE: 0.024 vs 0.029）
- Block angle：Sub-JEPA linear probe 略差（0.218 vs 0.187），但 MLP probe 持平（0.021 vs 0.022）

**Block angle 的现象很有意思**：rotation 是 inherently non-linear 的 quantity，subspace projection 可能把 angular structure 分散到不同 subspaces，导致 linear decodability 下降。但 MLP probe 能 compensate，说明信息没丢，只是变成了 non-linearly decodable。这是一个 nuanced tradeoff：subspace regularization 提升 translational quantities 的 linear decodability，但可能复杂化 rotational 的。

### 3.6 Latent Trajectory Straightening (Figure 6)

Straightness 定义：
$$S_{straight} = \frac{1}{B(T-2)} \sum_{i=1}^B \sum_{t=1}^{T-2} \frac{\langle v_t^{(i)}, v_{t+1}^{(i)} \rangle}{\|v_t^{(i)}\| \|v_{t+1}^{(i)}\|}$$

其中 $v_t = z_{t+1} - z_t$ 是 temporal velocity。这个指标衡量 latent trajectory 的线性程度——相邻 velocity 向量的 cosine similarity 越高，trajectory 越 "直"。

Sub-JEPA 在 PushT 和 OGB-Cube 上都产生了更直的 latent trajectory，这是自然涌现的，没有显式优化。更直的 trajectory 意味着 latent dynamics 更 linear，planning 更容易。

---

## 四、与相关工作的联系

### 4.1 JEPA 家族

- **I-JEPA** (Assran et al., CVPR 2023)：image representation learning，predict masked patches 的 embeddings
- **V-JEPA** (Bardes et al., TMLR 2024)：video version，predict future frames 的 embeddings
- **V-JEPA 2** (Assran et al., 2025)：大规模 self-supervised video model，enable understanding, prediction and planning
- **LeJEPA** (Balestriero & LeCun, 2025)：理论上证明了 Gaussian prior 是 optimal embedding distribution for minimizing downstream prediction risk，Gaussian regularization 有 rigorous theoretical footing

Sub-JEPA 在这个 family 里的位置：它是 LeWM 的直接改进，沿用了 LeJEPA 的 Gaussian regularization 理论框架，但通过 subspace decomposition 放松了全局约束。

### 4.2 Sliced Wasserstein 和 Random Projections

Sub-JEPA 的方法论跟 **sliced Wasserstein distance** (Bonneel et al., 2015) 有深层的联系。Sliced Wasserstein 把高维 OT 问题降维到 1D projections 的序列，avoiding curse of dimensionality。LeWM 的 Gaussian regularization 本身就属于这个 family——用 random directions sketch embedding distribution。

Sub-JEPA 把这个 idea 推进一步：先投影到 low-dimensional subspaces，再在每个 subspace 里做 sliced-style regularization。这相当于一个 hierarchical 的 distribution matching：global → subspace → 1D slices。

### 4.3 Manifold Hypothesis 和 Effective Rank

Paper 的 motivation 跟 representation learning 里的 manifold hypothesis (Bengio et al., 2013; Tenenbaum et al., 2000) 直接相关。Natural data lie on low-dimensional manifolds embedded in high-dimensional space。LeWM 的 isotropic Gaussian prior 与这个 hypothesis 冲突。

Effective rank (Roy & Vetterli, 2007) 作为 manifold dimensionality 的 proxy，在实验中被用来验证这个 hypothesis。这个指标在 representation learning 里越来越常用，值得熟悉。

---

## 五、Build Intuition: 为什么这个方法 Work

让我总结一下核心 intuition：

1. **Collapse 的本质**：JEPA 没有 reconstruction loss，encoder 可以 trivially 把所有输入 map 到同一点。需要某种结构约束来防止 degenerate solution。

2. **LeWM 的 insight**：isotropic Gaussian prior 是一个 principled 的 anti-collapse 机制。理论上（LeJEPA），Gaussian 是 minimize downstream prediction risk 的 optimal embedding distribution。但 isotropic $\mathcal{N}(0, I_D)$ 太强了——它要求所有 D 个方向都有相同 variance。

3. **Manifold mismatch**：control task 的 dynamics 通常是 low-dimensional 的。比如 Two-Room 是 2D navigation，agent 的 state 可以用 2D position 描述，但 latent space 是 192 维的。LeWM 强制 192 维全部 "用起来"，这浪费了 capacity 在 task-irrelevant 的方向上。

4. **Sub-JEPA 的 relax**：在 K 个 random subspaces 里施加 $\mathcal{N}(0, I_{d_s})$，每个 subspace 内仍然 non-degenerate（anti-collapse），但 cross-subspace 可以 correlated。这允许 $\text{Cov}(z)$ 有 eigenvalue spectrum 而不是 uniform 1，effective rank 可以 < D。

5. **Effective rank → planning**：lower effective rank 意味着 latent space 更 compact，dynamics 更 smooth，planning 更容易。Figure 6 的 trajectory straightening 实验直接展示了这一点。

6. **Frozen orthogonal projection 的必要性**：如果 projection 可训练，encoder 会让 projection 对齐到 "easy" 方向，削弱 regularization。如果 non-orthogonal，subspaces 之间 redundant，浪费 regularization capacity。Frozen + orthogonal 确保每个 subspace 独立、balanced、stable。

---

## 六、Limitations 和 Future Directions

1. **K 是 task-dependent 的**：需要 validation set 调。能不能用 intrinsic dimension estimation 自动确定 K？
2. **Random projections vs learned projections**：虽然 frozen 是必要的，但 random 是否最优？能否用某种 data-driven 的 initialization？
3. **Scale**：只在相对小的 continuous control 环境上测试了。在 Atari、real-world video、robotics 等更大规模 setting 上表现如何？
4. **跟 V-JEPA 2 的关系**：V-JEPA 2 是 FAIR 最近的大规模 video world model，用了 EMA target encoder 和其他 anti-collapse 机制。Sub-JEPA 的 subspace regularization 能否 complement 这些机制？
5. **Block angle 的 non-linear decodability 问题**：subspace regularization 可能 fragment angular structure。对于需要 precise angular control 的 task，这可能是一个 limitation。

---

## 七、参考链接

- **LeJEPA** (Balestriero & LeCun, 2025): https://arxiv.org/abs/2510.23231
- **LeWorldModel**: https://arxiv.org/abs/2602.xxxxx (2026 preprint, 见 paper ref [12])
- **V-JEPA 2** (Assran et al., 2025): https://arxiv.org/abs/2506.13403
- **I-JEPA** (Assran et al., CVPR 2023): https://arxiv.org/abs/2301.08243
- **DINOv2** (Oquab et al., TMLR 2024): https://arxiv.org/abs/2304.07193
- **DreamerV3** (Hafner et al., Nature 2025): https://arxiv.org/abs/2301.04104
- **IRIS** (Micheli et al., ICLR 2023): https://arxiv.org/abs/2210.05833
- **VICReg** (Bardes et al., ICLR 2022): https://arxiv.org/abs/2105.04906
- **Dimensional Collapse** (Jing et al., ICLR 2022): https://arxiv.org/abs/2110.09348
- **Cramér-Wold Theorem** (1936): https://doi.org/10.1112/jlms/s1-11.4.290
- **Johnson-Lindenstrauss Lemma**: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
- **Sliced Wasserstein** (Bonneel et al., 2015): https://arxiv.org/abs/1907.00027
- **Epps-Pulley Test** (Biometrika 1983): https://doi.org/10.1093/biomet/70.3.723
- **Effective Rank** (Roy & Vetterli, 2007): https://ieeexplore.ieee.org/document/4400448
- **OGBENCH** (Park et al., ICLR 2025): https://arxiv.org/abs/2410.20092
- **Diffusion Policy** (Chi et al., IJRR 2025): https://arxiv.org/abs/2303.04137
- **DINO-WM** (Zhou et al., ICML 2025): https://arxiv.org/abs/2411.11035
- **PLDM** (Sobal et al., 2025): https://arxiv.org/abs/2502.00710
- **Sub-JEPA GitHub**: https://github.com/intcomp/sub-jepa
- **Yann LeCun's JEPA vision** (2022): https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 八、总结

Sub-JEPA 是一个 **simple yet effective** 的改进。核心贡献是识别出 LeWM 的 isotropic Gaussian prior 在 low-intrinsic-dimension task 上的 over-constraint 问题，并用 subspace decomposition 来 relax。数学上，这允许 $\text{Cov}(z)$ 的 eigenvalue spectrum 有 variation 而不是 uniform，从而降低 effective rank，更好地匹配 task 的 intrinsic manifold structure。

实验上，effective rank 降低幅度与 planning success rate 提升幅度的强 correlation 是最有说服力的证据。Frozen orthogonal projection 的 ablation 也揭示了 co-adaptation 问题的存在——这是一个在 self-supervised learning 里反复出现的 theme。

从更大的视角看，这篇 paper 延续了 JEPA 路线的核心 philosophy：用 latent space prediction 而非 pixel reconstruction 来学习 world model，用 principled structural priors 而非 heuristic losses 来防止 collapse。Sub-JEPA 的 contribution 是在这个 framework 里找到了一个更好的 bias-variance operating point。
