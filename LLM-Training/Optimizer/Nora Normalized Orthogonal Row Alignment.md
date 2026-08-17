---
source_pdf: Nora Normalized Orthogonal Row Alignment.pdf
paper_sha256: 44cdb779690a8452adc41d25055be61fd747e68137cbce00e95d242313a8de8b
processed_at: '2026-08-05T22:34:19-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Nora 人话版

## 一句话概括

Nora = **先把 momentum 里"没用的方向"砍掉，再做 row normalization**，一步到位同时拿到 Muon 的 preconditioning 效果和 scale-invariance 的稳定性，而且只要两行代码。

---

## 为什么要搞这个东西

训练 LLM 的 optimizer 现在分两派：

**Adam 派**：把 weight 当成一维 vector，每个元素独立 scale gradient。简单，但完全忽略了 weight 是个 matrix 这个事实——loss landscape 里 row 和 row 之间是有耦合结构的，Adam 看不见。

**Muon 派**：把 weight 当 matrix 看，对 momentum 做正交化，相当于用一个 full-spectral preconditioner。效果好，但正交化要用 Newton-Schulz 迭代，cost 是 $\mathcal{O}(m^2 n)$。到了 1B 模型的 MLP 层（比如 $5461 \times 2048$），光这个正交化一步就要 7ms，而整个矩阵的 row normalization只要 0.1ms——**慢 70 倍**。

RMNP 是个折中方案：利用 Transformer Hessian 的 row block diagonal dominance（简单说就是 Hessian 的能量主要在行对角块上），把 Muon 的 full preconditioner 简化成 row-wise L2 normalization。快是快了，cost 降到 $\mathcal{O}(mn)$，但它**忘了一个很重要的事**。

---

## 被忘掉的事：scale invariance

Transformer 里铺天盖地的 RMSNorm/LayerNorm 带来一个性质：对很多参数，$f(w) = f(\lambda w)$ 对任意 $\lambda > 0$ 成立。

这意味着什么？**weight 的"长度"不重要，只有"方向"重要。**

几何上看，weight 空间里每个点其实是个方向（ray），沿着 ray 移动 loss 不变。真正有意义的运动是**角度变化**，也就是 tangent space 里的运动。沿着 weight 方向本身（radial 方向）的 update 纯属做无用功。

更糟的是，由于 RMSNorm 的存在，radial update 不仅没用，还有害——它会扰动 weight norm，而 weight norm 又会影响下一层的 effective scale，导致 effective learning rate 乱跳，训练不稳定。

这就像你想调整一个指南针的指向（angular），但有人在旁边不停地推拉指针的长度（radial），指针指向看起来就一直在抖。

---

## Muon 和 RMNP 各自的问题

**Muon 的问题**：贵。而且即使你先做了 projection 把 radial 去掉，它的正交化操作本身会"扭曲"方向，结果 update 又不严格正交于 weight 了。$\langle d_t^M, w_t \rangle \neq 0$。

**RMNP 的问题**：它直接对 $v_t$ 做 row normalization，**radial component 还在里面**。等于说"无用功"被当成有效信号一起 normalize 了，radial 噪声会污染 update 方向。

---

## Nora 的核心 insight

这里有个看起来平凡但其实很巧的观察：

> **如果先 row-project 把 radial 砍掉，再做 row normalization，正交性居然保持住了。**

为什么"居然"？因为一般来讲，preconditioner 是个矩阵操作，会扭曲方向。比如 Muon 的 full-spectral preconditioner 就会把 update 拐出 tangent space。但 row normalization 只是对每行做一个 scalar 缩放，**不改变每行的方向**，所以投影后的正交性被原封不动保留。

这就让两步可以"叠加"：

1. **Row projection**：$v_{t,i:}^{r\perp} = v_{t,i:} - \frac{\langle v_{t,i:}, w_{t,i:}\rangle}{\|w_{t,i:}\|^2} w_{t,i:}$  
   把每行 momentum 里跟 weight 同向的部分减掉，剩下的严格 perpendicular

2. **Row normalization**：$d_{t,i:} = v_{t,i:}^{r\perp} / \|v_{t,i:}^{r\perp}\|_2$  
   每行 normalize 成单位长度

然后 update 就是 $w_{t+1} = w_t - \eta d_t$。

---

## 为什么 row normalization 等价于 Muon-like preconditioning

这步是 paper 的关键 derivation。Muon 的 preconditioner 是 $H_M = (v_t v_t^T)^{1/2} \otimes I_n$，利用 row block diagonal dominance，近似成只留对角块：

$$H_R = (\text{diag}(v_t v_t^T))^{1/2} \otimes I_n$$

作用到 $v_t$ 上就是每行除以自己的 L2 norm——也就是 row normalization。RMNP 就是这么干的。

Nora 把这个 derivation 搬到 $v_t^{r\perp}$ 上：

$$H_N = (\text{diag}(v_t^{r\perp}(v_t^{r\perp})^T))^{1/2} \otimes I_n$$

作用到 $v_t^{r\perp}$ 上：

$$H_N^{-1}[v_t^{r\perp}]_{i:} = \frac{v_{t,i:}^{r\perp}}{\|v_{t,i:}^{r\perp}\|_2}$$

**还是 row normalization**。但这次是对投影后的 momentum 做。

所以整个故事是：row projection 让我们进入了 tangent space，row normalization 让我们拿到 Muon-like preconditioner，而这两步在几何上不互相破坏。这就是 "Normaliz**ed** Orthogonal Row Align**ment**" 这个名字的由来——projection (orthogonal) + normalization 在 row 维度上 align。

---

## 一个不直观的数值现象

Theorem 4.2 给了个很有意思的高维渐近结果：

在高维极限下，projection 几乎不损失信号强度。具体说，原始 gradient $\delta_i x$ 的 norm 是 $\Theta(\sqrt{n})$，而被减掉的 radial component 是 $\mathcal{O}_p(1)$——**差了 $\sqrt{n}$ 倍**。

直觉上你可能会担心："我把 momentum 的一部分砍掉了，信号不就弱了吗？"但高维下，random vector 和 weight 的 inner product 是 $\mathcal{O}(1)$（CLT），而 gradient 本身是 $\mathcal{O}(\sqrt{n})$，所以 radial 部分相对量极小，projection 不会实质削弱 learning signal。

这解释了为什么 projection 在 stability 上赚到了好处，但在 efficiency 上几乎没代价。

---

## µP scaling：$\eta \propto 1/\sqrt{n}$

标准 µP 要求：activation 的 update $\Delta h = \Theta(1)$，不随 width $n$ 爆炸或消失。

Forward update：$\Delta h_i = -\eta \langle d_{i:}, x\rangle$。

Row normalization 让 $\|d_{i:}\|_2 = 1$，标准 init 让 $\|x\|_2 = \Theta(\sqrt{n})$，所以 $\langle d_{i:}, x\rangle = \Theta(\sqrt{n})$，要 $\Delta h = \Theta(1)$ 必须 $\eta = \Theta(1/\sqrt{n})$。

这个 scaling law 跟 Muon 在 large-scale 实验里用的经验值是一致的。意味着 Nora 可以在小模型调好 LR，直接 scale 到大模型——这是工程上极重要的 property。

---

## 代码核心就两行

Reference code 里关键部分：

```python
theta_hat = F.normalize(param_data, p=2, dim=-1, eps=eps)
dot_product = torch.sum(m_t * theta_hat, dim=-1, keepdim=True)
v = m_t - dot_product * theta_hat          # row projection
v_hat = F.normalize(v, p=2, dim=-1, eps=eps) # row normalization
```

整个 Nora 逻辑就这 4 行（前两行是 projection 的 numerically stable 实现）。Plug-and-play 替换 Adam，对非 matrix 参数（embedding, lm_head, 1D bias 等）还是用 Adam。

一个 subtle 点：paper 的算法描述用 $w_{i:}$ 直接做投影，但 reference code 用 $\hat{w}_{i:} = w_{i:}/\|w_{i:}\|$。数学等价，但 code 版本 numerical stable（避免除以 $\|w\|^2$ 在早期训练时 weight 小导致爆炸）。

---

## 跟 Mano 的区别

Mano 是同期工作，思路类似，来自 Riemannian optimization。它做 row/column 交替投影 + 交替 normalization：

- 奇数步：row projection + row normalization
- 偶数步：column projection + column normalization

Nora 的 thesis 是：**Transformer Hessian 的 row block diagonal dominance 是主要结构 prior，column-wise 操作是不必要的 heuristic，反而拖累训练**。

Ablation 实验把这个点证明得很干净：把 Mano 的 weight decay 也设为 0（消除超参差异），Nora 在所有 LR 下都赢。这告诉我们 column projection 不仅没用，还可能引入 noise。

---

## 实验现象值得品味

训练 dynamics 有个有趣模式：

- **RMNP 早期最快**：因为保留了 radial component，短期内 update magnitude 大，loss 下降快
- **Muon plateau 较高**：preconditioner 强但 update 偏离 tangent，effective 信号其实被 radial drift 抵消了一部分
- **Nora 早期慢，后期持续改善**：因为 tangent update 让 $\|w\|$ 几乎不变（只二阶增长），effective learning rate 稳定，可以 sustain 长期优化

这其实是个挺 general 的现象——**短期看，"无用功"也像是在做事；长期看，"无用功"会扰乱节奏**。Nora 早期显得慢，是因为它不去做那些 short-term 看似有用但长期有害的 radial motion。

最终在 135M 上 Nora val loss 3.079，比 Mano 的 3.097、Muon 的 3.142 都低。虽然差距不大，但考虑 wall-clock 时间（135M 上 Nora 比 Muon 快约 10 倍 per optimizer step），trade-off 非常划算。

---

## 还没解决的问题

1. **只验证到 135M**。Muon 已经在 DeepSeek-V3 的 671B 上验证 scalable，Nora 理论上应该更好（复杂度更低），但需要大规模 empirical 验证。

2. **Muon 对比的 confound**。Muon 用 weight_decay=0.1，Nora 用 0。paper 对 Mano 做了 controlled ablation（设 wd=0），但对 Muon 没有。Muon + wd=0 的 baseline 缺失，这让 "Nora vs Muon" 的结论不够干净。

3. **Column-wise 结构被完全抛弃**。Nora 的论据是 Transformer Hessian 的 row-block dominant，但其他 architecture（比如 CNN、SSM、Mamba？）的 Hessian 结构未必如此，Nora 的 generality 存疑。

4. **Numerical stability in early training**。$\|w_{i:}\| \to 0$ 时 projection 不稳定，code 用 eps=1e-10 兜底，但没系统分析 worst case。

---

## 我觉得最 elegant 的地方

Nora 的 elegance 在于它**不是发明了一个新 trick，而是发现了一个代数兼容性**：

- Row projection 是为了 stability（进 tangent space）
- Row normalization 是为了 efficiency（Muon-like preconditioner 的对角近似）
- 这两个目标在一般 preconditioner 框架下是冲突的（preconditioner 会扭曲方向，破坏 tangent 性质）
- 但**当 preconditioner 退化为 row-wise scalar 时，它恰好不改变方向，所以正交性保持**

这是个"对的位置、对的简化、让冲突自然消失"的故事。不是 brute force 设计出来的，是观察到的。

整个 algorithm 只多了几行 element-wise 操作，代价几乎为零，却同时把 Muon 的昂贵正交化、RMNP 的 radial 噪声问题都解决了。这种"零代价拿到多个 property"的设计，在 optimizer 领域不多见。

---

## 参考

- Nora 论文：https://arxiv.org/abs/2603.XXXXX (具体编号见 paper)
- 代码：https://github.com/yuanjinghui/Nora
- Muon scalability: https://arxiv.org/abs/2502.16982
- µP 原始 paper: https://arxiv.org/abs/2203.03466
- Transformer Hessian structure: https://arxiv.org/abs/2402.05128

---

# Nora: Normalized Orthogonal Row Alignment 深度解读

Andrej, 这篇 paper 是 Yuan, Zou, Wang, Liu, Nie 的工作，核心贡献是在 matrix-based optimizer 设计中同时满足 **efficiency**、**stability**、**speed** 三个原则，下面我从 motivation、几何直觉、算法细节、理论、实验五个层面构建你的 intuition。

---

## 1. 核心问题：现有 matrix-based optimizer 的三角不可能

LLM 训练里，AdamW 把 weight 当 flat vector 处理，忽略了 matrix 的结构信息。Muon 通过 Newton-Schulz iteration 对 momentum 做正交化，capture 了 full-spectral preconditioner 的效果，但 cost 是 $\mathcal{O}(m^2 n)$，对 1B 级 MLP 矩阵（如 $2048\times 5461$）来说相当昂贵。

RMNP 利用 Transformer Hessian 的 **row block diagonal dominance**（即 $v_t v_t^T$ 的对角块占主导），把 Muon 的 full preconditioner 简化为 row-wise $\ell_2$ normalization：

$$H_R^{-1}[v_t]_{i:} = \frac{v_{t,i:}}{\|v_{t,i:}\|_2}$$

complexity 从 $\mathcal{O}(m^2 n)$ 降到 $\mathcal{O}(mn)$。但 RMNP 有个致命缺陷：它**忽视了 neural network 的 scale invariance**。

### Scale invariance 的几何含义

由于 RMSNorm / LayerNorm 的存在，对 scale-invariant 参数有 $f(w) = f(\lambda w)$, $\forall \lambda > 0$。这意味着：

1. Loss 只依赖 weight 的 angular orientation，不依赖 $\|w\|$
2. 真实 gradient 严格正交于 weight：$\langle \nabla f(w), w \rangle = 0$
3. **Radial momentum**（即 $v^{\parallel}$ 方向，与 $w$ 同方向）纯粹是 noise，会：
   - 干扰 angular optimization
   - 扰动 $\|w_t\|$，导致 effective learning rate 混沌震荡
   - 引入 internal covariate shift

RMNP 直接对 $v_t$ 做 row normalization，会保留 radial component，radial jitter 会污染 update 方向。Muon 即使对 $v_t^{\perp}$ 做 orthogonalization，结果 $d_t^M = (v_t^{\perp}(v_t^{\perp})^T)^{-1/2} v_t^{\perp}$ 也会偏离 tangent space，即 $\langle d_t^M, w_t \rangle \neq 0$。

---

## 2. Nora 的几何洞察：Projection + Normalization 的"幸运兼容"

Nora 的关键 observation 是一个看似平凡的代数性质：

> **对已经 row-orthogonal to $w_t$ 的向量做 row-wise normalization，正交性被保持。**

这个性质让 projection 和 preconditioning 两个步骤"兼容"。具体算法两步：

### Step 1: Row-wise orthogonal projection

把 momentum $v_t$ 投影到 $w_t$ 的 row-wise tangent space（去掉 radial component）：

$$v_{t,i:}^{r\perp} = v_{t,i:} - \frac{\langle v_{t,i:}, w_{t,i:}\rangle}{\|w_{t,i:}\|_2^2} w_{t,i:} \quad \text{for all row } i$$

这里：
- $v_{t,i:}$：momentum 第 $i$ 行
- $w_{t,i:}$：weight 第 $i$ 行
- $\frac{\langle v_{t,i:}, w_{t,i:}\rangle}{\|w_{t,i:}\|_2^2} w_{t,i:}$：$v_{t,i:}$ 在 $w_{t,i:}$ 方向的投影分量（radial part）

投影后保证 $\langle v_{t,i:}^{r\perp}, w_{t,i:}\rangle = 0$, $\forall i$。

### Step 2: Row-wise normalization (≈ diagonal preconditioner)

$$d_{t,i:} = \frac{v_{t,i:}^{r\perp}}{\|v_{t,i:}^{r\perp}\|_2}$$

### 为什么这等价于 Muon-like preconditioner？

paper 给出的关键 derivation：

利用 row block diagonal dominance 的 prior，preconditioner 取 $H_N = (\text{diag}(v_t^{r\perp}(v_t^{r\perp})^T))^{1/2} \otimes I_n$，作用到 $v_t^{r\perp}$ 上：

$$H_N^{-1}[v_t^{r\perp}]_{i:} = \left((\text{diag}(v_t^{r\perp}(v_t^{r\perp})^T))^{-1/2} v_t^{r\perp}\right)_{i:} = \frac{v_{t,i:}^{r\perp}}{\|v_{t,i:}^{r\perp}\|_2}$$

也就是说：**对 row-orthogonalized momentum 做对角 Gram preconditioning** $\equiv$ **row-wise normalization**。这是 RMNP 的对角近似思想，但作用在投影后的 $v_t^{r\perp}$ 上。

### 正交性保持的关键证明

论文 Equation (2)：

$$\langle H_N^{-1}[v_t^{r\perp}], w_t\rangle = \sum_{i=1}^m \frac{1}{\|v_{t,i:}^{r\perp}\|_2}\langle v_{t,i:}^{r\perp}, w_{t,i:}\rangle = 0$$

因为 row normalization 只是对每行做 scalar 缩放，每行向量方向不变，所以原本的 row-wise 正交性被严格保持。这就让 Nora 同时满足：

- **Efficiency**: 通过对角 Gram 近似实现 Muon-like preconditioning
- **Stability**: update 严格在 $w_t$ 的 tangent space 里，weight norm 不被 radial 噪声扰动
- **Speed**: 只多了一个 row-wise projection（element-wise operations），相对 RMNP 几乎零额外开销

### 与 Muon 的结构性区别

Muon 的 update $d_t^M = (v_t v_t^T)^{-1/2} v_t$ 需要 Newton-Schulz 迭代 $X_{k+1} = \frac{1}{2}X_k(3I - X_k^T X_k)$，且只能处理 $m \leq n$ 的方阵正交化。Nora 因为是基于 row-wise 操作，无论 $m \leq n$ 还是 $m \geq n$ 都统一处理，这点在 Transformer 不同 layer shape 上很方便（attention 的 hidden×hidden vs MLP 的 intermediate×hidden）。

---

## 3. 算法与 Reference Code 解析

Algorithm 1 核心：

```
g_t = ∇f(w_t; ξ_t)
v_t = β v_{t-1} + (1-β) g_t
v_t^{r⊥}_{i:} = v_{t,i:} - (<v_{t,i:}, w_{t,i:}> / ||w_{t,i:}||_2^2) * w_{t,i:}   # row projection
d_t = diag(v_t^{r⊥}(v_t^{r⊥})^T)^{-1/2} v_t^{r⊥}  # ≡ row normalization
w_{t+1} = w_t - η_t (d_t + λ w_t)
```

Reference Code 的关键两行（在 `is_nora and grad.dim() >= 2` 分支里）：

```python
theta_hat = F.normalize(param_data, p=2, dim=-1, eps=eps)       # w 的 row 单位向量
dot_product = torch.sum(m_t * theta_hat, dim=-1, keepdim=True)   # <v, w>/||w||
v = m_t - dot_product * theta_hat                               # row projection
v_hat = F.normalize(v, p=2, dim=-1, eps=eps)                    # row normalization
update_direction = v_hat * scale                                # scale = sqrt(m/n) 或 1
```

注意一个 subtle 点：paper 的 Algorithm 1 写的是直接用 $w_{t,i:}$，但 reference code 用的是 $\hat{w}_{t,i:} = w_{t,i:}/\|w_{t,i:}\|_2$（归一化后），数学上等价：

$$\frac{\langle v_{t,i:}, w_{t,i:}\rangle}{\|w_{t,i:}\|_2^2} w_{t,i:} = \langle v_{t,i:}, \hat{w}_{t,i:}\rangle \hat{w}_{t,i:}$$

代码层面用 normalize 比 divide by squared norm 更 numerical stable（避免 $\|w\|^2$ 接近 0 的极端情况）。

`scale = max(1, sqrt(m/n))` 是一个 dimension-aware 的 scaling，相当于 hidden dimension 比例的修正（在 Muon 实现里也有类似 trick，用于把 update magnitude 校准到合理 scale）。

### Weight decay 的设计选择

Nora 默认 `weight_decay=0`，这一点很重要。因为：
- Tangential update 已经使 $\|w_t\|$ 几乎二阶缓慢增长（discrete update 沿 tangent space，norm 增长率 $\approx \eta^2/2$）
- 加 weight decay 反而会强行收缩 $\|w\|$，破坏 scale-invariant geometry
- 实验对比中 Muon/Mano/RMNP 都用 weight_decay=0.1，Nora 用 0，这本身就是 stability 优势的体现

---

## 4. 理论分析：µP Scaling 与 Convergence

### Theorem 4.1: 学习率 scaling law $\eta \propto 1/\sqrt{n}$

考虑一个 layer $h = wx$，其中 $w \in \mathbb{R}^{m\times n}$，假设输入 activation $\|x\|_2 \leq \gamma\sqrt{n}$（标准初始化下 $\gamma = \Theta(1)$）。

Forward update 量：
$$\Delta h_i = (w_{t+1,i:} - w_{t,i:}) x = -\eta_t \langle d_{t,i:}, x\rangle$$

Cauchy-Schwarz:
$$|\Delta h_i| \leq \eta_t \|d_{t,i:}\|_2 \cdot \|x\|_2 \leq \eta_t \cdot 1 \cdot \gamma\sqrt{n}$$

要 $\Delta h_i = \Theta(1)$（µP 的 stable feature learning 要求），必须有 $\eta_t = \Theta(1/\sqrt{n})$。

这是一个**上界**论证，但 paper 不止于此，Theorem 4.2 给出**tight 渐近**结果。

### Theorem 4.2: 高维极限下的真实 order

假设 $w_{i:} \sim \mathcal{N}(0, \sigma_w^2 I_n / n)$（标准 deep network 初始化），$x_j$ 独立零均值方差 $\sigma_x^2 = \Theta(1)$，error signal $\delta_i = \Theta(1)$，vanilla gradient $g_{i:} = \delta_i x^T$。

经过 projection 后：
$$u_{i:} = \delta_i x - \frac{\delta_i \langle x, w_{i:}\rangle}{\|w_{i:}\|_2^2} w_{i:}$$

用大数律和 CLT 估计 order：
- $\|x\|_2^2 \xrightarrow{p} n\sigma_x^2$，即 $\|x\|_2 = \sigma_x\sqrt{n} + o_p(\sqrt{n})$
- $\|w_{i:}\|_2^2 \xrightarrow{p} \sigma_w^2 = \Theta(1)$（注意 $w_{i:}$ 是 $\mathcal{N}(0, \sigma_w^2/n)$，所以 $\|w_{i:}\|^2$ 是 $\Theta(1)$）
- $\langle x, w_{i:}\rangle \xrightarrow{d} \mathcal{N}(0, \sigma_x^2\sigma_w^2) = \mathcal{O}_p(1)$（n 个独立乘积和，总方差 $\sigma_x^2\sigma_w^2$，CLT）

被去掉的 radial component 的 norm：
$$\|r_{i:}\|_2 = \frac{|\delta_i| \cdot |\mathcal{O}_p(1)|}{\sigma_w^2} \cdot \sigma_w = \mathcal{O}_p(1)$$

主项 $\delta_i x$ 的 norm 是 $\Theta(\sqrt{n})$，被减掉的 radial 是 $\mathcal{O}_p(1)$，所以：

$$\|u_{i:}\|_2 = |\delta_i| \sigma_x \sqrt{n}(1 + o_p(1))$$

Row normalization 后：
$$d_{i:} = \frac{u_{i:}}{\|u_{i:}\|_2} = \frac{\delta_i x - r_{i:}}{|\delta_i|\sigma_x\sqrt{n}(1 + o_p(1))}$$

Inner product with input:
$$\langle d_{i:}, x\rangle = \frac{\delta_i n\sigma_x^2 - \mathcal{O}_p(\sqrt{n})}{|\delta_i|\sigma_x\sqrt{n}(1+o_p(1))} = \text{sgn}(\delta_i)\sigma_x\sqrt{n} + o_p(\sqrt{n})$$

关键观察：**projection 几乎不损失主项**（因为 radial component 在高维下是低阶的）。这就是 $\mu$P 仍然成立的根本原因。

因此 $\Delta h_i = -\eta \langle d_{i:}, x\rangle \approx \mp \eta \sigma_x\sqrt{n}$，要 $\Delta h_i = \Theta(1)$，必须有 $\eta = \eta_0 / \sqrt{n}$。

### Convergence Analysis

定义 row-wise projected gradient 作为 stationarity measure：
$$\mathcal{G}_t := \mathcal{P}_{w_t}^{r\perp}(\nabla f(w_t))$$

这是 Nora 的"真实信号"，因为 radial component 被 projection 过滤了，descent inner product 只依赖这一项。

**Assumption 4.3** 两种 smoothness:
- (a) Frobenius smoothness: $\|\nabla f(w) - \nabla f(w')\|_F \leq L_F \|w - w'\|_F$
- (b) Matched $(\infty, 2)$-smoothness: $\|\nabla f(w) - \nabla f(w')\|_{1,2} \leq L_{\infty,2}\|w - w'\|_{\infty,2}$

其中 $\|x\|_{1,2} := \sum_i \|x_{i:}\|_2$，$\|x\|_{\infty,2} := \max_i \|x_{i:}\|_2$。这两个 norm 是 **row-block structured** 的，正好匹配 Nora 的 row-wise update geometry。

**核心 Lemma**:

- **Lemma A.1 (Non-expansiveness of row projection)**: 在 $\|\cdot\|_F$、$\|\cdot\|_{1,2}$、$\|\cdot\|_{\infty,2}$ 三个 norm 下都有 $\|\mathcal{P}_w^{r\perp}(x)\| \leq \|x\|$。这是因为每个 $P_i = I_n - \hat{w}_{i:}^T \hat{w}_{i:}$ 是正交投影。
  
- **Lemma A.2 (Nora update geometry)**: 
  1. $\langle d_{i:}, w_{i:}\rangle = 0$ (tangent)
  2. $\|d\|_F \leq \sqrt{m}$（每行 norm ≤ 1，m 行）
  3. $\|d\|_{\infty,2} \leq 1$（每行 norm ≤ 1）
  4. $\langle z, d\rangle = \|z\|_{1,2}$（关键恒等式：$z$ 与自身 normalize 的 inner product 等于 $\ell_{1,2}$ norm）
  5. $\langle z, d\rangle \geq \|z\|_F$（$\ell_{1,2} \geq \ell_2$ by Jensen）

- **Lemma A.5/A.6 (Descent lower bound)**:
  $$\langle \nabla f(w_t), d_t\rangle \geq \|\mathcal{G}_t\|_F - (\sqrt{m}+1)\|e_t\|_F$$
  $$\langle \nabla f(w_t), d_t\rangle \geq \|\mathcal{G}_t\|_{1,2} - 2\|e_t\|_{1,2}$$
  其中 $e_t := v_{t+1} - \nabla f(w_t)$ 是 momentum tracking error。

**Theorem 4.4**: Matched $(\infty,2)$-smoothness 下，constant step size:
$$\frac{1}{T}\sum_t \mathbb{E}[\|\mathcal{G}_t\|_{1,2}] \leq \frac{\Delta}{T\eta} + 2\left[\frac{L_{\infty,2}\eta\beta}{1-\beta} + \frac{\sqrt{m}\sigma}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}\right] + \frac{L_{\infty,2}\eta}{2}$$

参数选择 $\eta = \sqrt{(1-\beta)\Delta / (L_{\infty,2}T)}$，$1-\beta = \min\{\sqrt{L_{\infty,2}\Delta}/(2\sqrt{m}\sigma\sqrt{T}), 1\}$，得到：

$$T = \mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$$

注意只有 $m$ 的一次方，比 Frobenius smoothness 下 $T = \mathcal{O}(m^2 L_F \sigma^2 \Delta\epsilon^{-4})$ 好。这是因为 row-wise 结构的 norm 更紧。

**Corollary 4.6**: 在 row-wise scale invariance 假设下，$\mathcal{G}_t = \nabla f(w_t)$，即投影不丢任何信号。证明很 elegant：对角矩阵 $D_i(c) = I + (c-1)e_i e_i^T$，由 $f(D_i(c)w) = f(w)$ 在 $c=1$ 处求导得 $\langle \nabla f(w)_{i:}, w_{i:}\rangle = 0$。

---

## 5. 与 Mano 的关键对比

Mano 来自 Riemannian optimization 思路，做 row/column 交替投影 + normalization：

- 奇数步：$v_t^{r\perp}$ + row normalization
- 偶数步：$v_t^{c\perp}$ + column normalization

Nora 的核心 thesis 是：**Transformer Hessian 的 row block diagonal dominance 才是正确的 structural prior**，column-wise 操作是不必要的 heuristic。

Ablation 实验很关键：把 Mano 的 weight_decay 也设为 0（消除超参差异），Nora 仍然赢：

| Optimizer | 0.003 (ppl/loss) | 0.005 | 0.01 | 0.02 |
|-----------|------------------|-------|------|------|
| Mano | 22.13 / 3.097 | 22.14 / 3.098 | 23.34 / 3.150 | 25.07 / 3.222 |
| Nora | 21.74 / 3.079 | 21.86 / 3.085 | 22.43 / 3.111 | 23.39 / 3.152 |

Nora 在所有 LR 下都赢，证明优势来自算法结构本身，不是 weight decay 配置。

---

## 6. 实验：训练 dynamics 与 wall-clock 优势

### 主结果 (Table 2)

| Optimizer | 60M val loss | 135M val loss |
|-----------|--------------|---------------|
| Muon | 3.44 | 3.142 |
| Mano | 3.39 | 3.097 |
| RMNP | 3.41 | 3.112 |
| **Nora** | **3.37** | **3.079** |

### 训练 dynamics (Figure 1)

一个有意思的现象：
- RMNP 早期收敛最快（loss 最低）
- Muon 早期慢，且 plateau 在较高值
- Mano 和 Nora 起步较高，但在 15k steps 之后持续改善
- Nora 在训练末期超越所有方法，达到最低 final loss

这暗示 Nora 的 tangent-space projection 在早期可能因为过滤 radial 信息"走得更慢"，但长期来看 scale-invariance 的稳定性让它能 sustain 优化。这跟"radial noise 短期看似加速、长期累积破坏 effective learning rate"的 narrative 一致。

### Wall-clock 比较 (Table 4)

Newton-Schulz (5 步) vs Row normalization 的 kernel runtime：

| Model | Shape | Row norm (ms) | NS(5) (ms) | Ratio |
|-------|-------|---------------|------------|-------|
| 60M | 512×512 | 0.0689 | 0.6554 | 9.52× |
| 135M | 768×768 | 0.0686 | 0.6520 | 9.50× |
| 350M | 1024×1024 | 0.0674 | 0.6397 | 9.49× |
| 1B | 2048×2048 | 0.0684 | 2.0552 | 30.06× |
| 1B MLP | 5461×2048 | 0.0985 | 6.9985 | **71.02×** |
| 1B MLP | 2048×5461 | 0.1084 | 7.9678 | **73.51×** |

对 1B 规模 MLP，NS 比 row norm 慢 70 倍以上。这就是 Nora "speed" 原则的实际收益。NS 的 cost 随矩阵规模快速上升（$m^2 n$），而 row norm 几乎常数（element-wise operations）。

---

## 7. 一些值得深挖的 subtle 点

### 7.1 为什么 row block diagonal dominance 在 Transformer 上成立

Zhang et al. (2024) "Why Transformers Need Adam: A Hessian Perspective" 指出 Transformer 的 layer-wise Hessian 有强 row-wise 块对角占优结构，因为 attention 和 MLP 的 computation 主要在 row direction 上独立处理（每个 token 对应一行）。这是 RMNP 和 Nora 都依赖的关键 prior。

参考: https://arxiv.org/abs/2402.05128

### 7.2 Nora vs SSO

SSO (Spectral Sphere Optimizer, Xie et al. 2026) 也试图保持 tangent update，但用 bisection method，实际太慢。Nora 通过 row-wise projection 的代数技巧绕过了这个问题。

参考 SSO: https://arxiv.org/abs/2601.08393

### 7.3 与 µP 的关系

Nora 的 $\eta \propto 1/\sqrt{n}$ scaling law 与 Yang et al. 的 µP 框架兼容，意味着可以 zero-shot hyperparameter transfer across width。这对 LLM scale-up 极其重要——你可以先在小模型上调 LR，然后按 $\sqrt{n}$ scaling 直接用到大模型上。

µP 参考: https://arxiv.org/abs/2203.03466

### 7.4 weight decay = 0 的 implication

传统 wisdom 是 weight decay 是 LLM 训练的 regularizer，但 Nora 实验显示在 scale-invariant geometry 下不需要。这跟 recent 一些 paper 的观察一致——很多 weight decay 的作用其实是"修正非 scale-invariant optimizer 的 radial drift"。Nora 直接从源头消除 radial drift，weight decay 反而是 redundant。

参考 Karpathy 自己的 nanoGPT 训练经验也通常用相对较小的 weight decay（如 0.1），而 Nora 实验显示 0 也能 work 甚至更好。

### 7.5 代码里 `scale = max(1, sqrt(m/n))` 的物理意义

这个 scaling 是为了在不同 aspect ratio 的矩阵上保持 update magnitude 的合理 scale。Muon 的 NS iteration 自动产生正交矩阵（每行 norm 为 1），但 Nora 的 row normalization 让每行 norm 为 1，整体 Frobenius norm 是 $\sqrt{m}$，需要根据 $m/n$ 比例做 correction 让 effective step size 在不同 layer shape 上一致。这跟 Muon 实现里的 `adjust_p_for_muon` 是类似的 calibration trick。

---

## 8. 可能的扩展方向与潜在问题

### 8.1 Scaling 到更大模型

paper 只验证到 135M。Muon 在更大模型（DeepSeek-V3 上 671B）已经验证过 scalable，Nora 的算法复杂度更低，理论上应该 scale 更好，但需要 empirical 验证。

参考 Muon scalability: https://arxiv.org/abs/2502.16982

### 8.2 Column-wise 结构的 loss

Nora 主动放弃了 column-wise projection（vs Mano 的交替）。如果某些 layer 的 Hessian 同时有 row 和 column block diagonal dominance（比如 embedding layer？），Nora 可能 suboptimal。但 paper 的 ablation 显示 Transformer 上 row-only 是更优选择。

### 8.3 Numerical stability of projection

当 $\|w_{i:}\|_2 \to 0$ 时，projection $\frac{\langle v, w\rangle}{\|w\|^2} w$ 数值不稳定。Reference code 用 `F.normalize(param_data, eps=eps)` with `eps=1e-10`，但极端情况下仍需注意。早期训练时 weight 较小，可能是 risk zone。

### 8.4 与 Sophia / Shampoo / Adafactor 的关系

这些 optimizer 各自有 preconditioning 思路，但都没有同时处理 scale-invariance。Nora 的 framework 是否可以扩展到这些 preconditioner 上是个开放问题。

参考 Shampoo: https://proceedings.mlr.press/v80/gupta18a.html
参考 Sophia: https://arxiv.org/abs/2305.14342

---

## 9. 我的 takeaways

Nora 的核心 insight 是一个 **代数兼容性**：row projection 和 row normalization 在保持正交性上"幸运地"兼容，这让我们能把 Muon 的 preconditioning 思想和 scale-invariance 的 stability 要求统一在一个 $\mathcal{O}(mn)$ 的算法里。

这个工作的优美之处在于：
1. **理论 completeness**: 既有 µP scaling law，又有 non-convex convergence，还有 row-wise scale invariance 下的简化
2. **工程友好**: 两行核心代码，与 Adam plug-and-play 兼容
3. **几何 principled**: 不是 ad-hoc 的 trick，而是从 scale-invariance 几何出发的自然推导

潜在 concern：
- 实验规模偏小（135M），需要 scale 验证
- 与 Muon 的比较里，Muon 用 weight_decay=0.1 但 Nora 用 0，这个 confound 在 ablation 里只对 Mano 做了 controlled comparison，对 Muon 没有
- Column-wise prior 的完全抛弃需要更广泛的 model architecture 验证

参考 paper 仓库: 
- 主仓库: https://github.com/yuanjinghui/Nora
- 复现仓库: https://github.com/yuanjinghui/Nora-reproduction

如果你想深挖某个具体方面（比如 NS iteration 的 numerical analysis、µP scaling law 的更紧 bound、或与 Riemannian optimization 的关系），可以告诉我，我可以进一步展开。
