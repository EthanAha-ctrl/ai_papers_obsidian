---
source_pdf: Fantastic Pretraining Optimizers and.pdf
paper_sha256: 478846f2e17b3da286a5731f4640bb794d90f6bcf35f72939e1ced504ada1675
processed_at: '2026-08-04T06:36:06-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Andrej，咱们用大白话来拆解这篇paper。这篇paper本质上是Stanford团队给整个LLM optimizer圈做了一次"打假"和"大洗牌"。

用一句话总结：**过去两年那些声称比AdamW快2倍的新optimizer，大部分都是在"耍流氓"——要么是baseline没调好，要么是评估手段有误导性。如果大家在一个绝对公平的擂台上PK，最快的optimizer确实有提速，但最多只有1.4倍，而且模型越大，提速越弱，到了1.2B参数基本就只剩1.1倍了。**

下面我用最直白的方式给你梳理这篇paper的精髓，同时按照你的要求，把核心的公式、变量定义和实验数据都详细拆解，帮你建立直觉。

---

### 一、 为什么以前paper的"2倍提速"是假象？

Stanford团队指出，以前发optimizer paper的作者们犯了三个致命的方法论错误：

1.  **Baseline严重Under-tuned (调参不公)**
    很多paper拿GPT-3当时的config做baseline。那个config的peak learning rate极小。这篇paper在Figure 1里展示了一个极其打脸的事实：如果仅仅把AdamW的learning rate稍微调大一点，AdamW自己就能获得2倍提速。所以，那些"新optimizer比AdamW快2倍"的claim，本质上只是"新optimizer比一个没调好的AdamW快2倍"。

2.  **盲目共用Hyperparameter**
    很多paper做对比实验时，给所有optimizer套用相同的learning rate和weight decay。这完全违背了各家的物理特性。
    *Intuition*：Lion的更新规则是用sign函数，它的update magnitude固定为 $\eta$；而AdamW的update magnitude是 $\eta / \sqrt{v_t}$。这两种完全不同的scale，怎么可能适用同一个weight decay？实验证明，Lion的最优weight decay在0.6左右，而AdamW在0.1左右。强行共用，就是在坑baseline。

3.  **看中间Checkpoint，被早期假象骗了**
    很多paper在训练跑到一半时截图，说"看，我的optimizer loss更低"。但这非常具有误导性。在learning rate decay阶段，loss曲线的相对位置会**交叉并反转**。早期跑得猛的optimizer，后期往往plateau；早期慢的，后期反而反超。所以，评估optimizer必须看训练完全结束时的final loss。

---

### 二、 核心对决：Scalar-based vs Matrix-based Optimizers

这篇paper最核心的insight是：把11个optimizer放在一起看，可以清晰地分为两大阵营，**Matrix-based optimizer全面碾压Scalar-based optimizer**。

#### 1. Scalar-based Optimizers (AdamW, Lion, NAdamW, Mars, etc.)
这类optimizer把神经网络的参数当成一维的独立数字来看待。它们计算每个参数的gradient历史方差，然后对每个参数进行单独的缩放。

**AdamW 的公式拆解：**
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w_t$$

*   $w_t$: 第 $t$ 步的权重参数。
*   $\eta$: Learning rate。
*   $\hat{m}_t$: Bias-corrected first moment (梯度的均值)。
*   $\hat{v}_t$: Bias-corrected second moment (梯度的方差)。
*   $\lambda$: Weight decay系数。
*   $\epsilon$: 防止除以0的极小数。

*Intuition*：AdamW的逻辑是"如果某个参数的gradient历史上一直很大，我就把它的learning rate缩小"。这完全忽略了矩阵的几何结构。如果一个参数矩阵在某些"方向"（singular directions）上需要大步走，在另一些方向上需要小步走，AdamW是看不到这种row/column级别的相关性的。

#### 2. Matrix-based Optimizers (Muon, Soap, Kron, Scion)
这类optimizer深刻理解Transformer的参数本质上是矩阵。它们用另一个矩阵去precondition（预调）梯度，直接在矩阵层面做操作。在所有model scale下，这个阵营的optimizer都比scalar阵营快。

**Muon 的公式与Newton-Schulz拆解：**
Muon是目前最出圈的matrix-based optimizer，它的核心是Newton-Schulz正交化。

$$w_{t+1} = w_t - \eta \cdot \text{NS}^{(5)}(\beta_2 \tilde{m}_t + (1-\beta_2) g_t) - \eta \lambda w_t$$

*   $w_t$: 第 $t$ 步的权重矩阵。
*   $g_t$: 当前步的梯度矩阵。
*   $\tilde{m}_t$: Momentum。
*   $\text{NS}^{(5)}(M)$: 对矩阵 $M$ 迭代5次Newton-Schulz正交化。

*Intuition*：梯度矩阵 $G$ 做SVD分解后是 $U \Sigma V^T$。Newton-Schulz操作相当于把 $\Sigma$（singular values）全部抹平变成1，只保留 $UV^T$。这等于让更新步骤在所有特征方向上都有相同的"步长"。在Transformer中，梯度的singular values差异极大，Muon这种"强制对齐步长"的做法在训练初中期极其有效，收敛速度暴涨。

**Soap 的公式拆解：**
Soap是Shampoo的改进版，它有状态记忆。

$$w_{t+1} = w_t - \eta Q_A^\top \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right) Q_B^\top$$

*   $Q_A, Q_B$: 两个正交矩阵，通过维护梯度的协方差矩阵 $G_A, G_B$ 并定期做QR分解得到。
*   $m_t, v_t$: 在 $Q_A, Q_B$ 变换后的空间里维护的Adam的first/second moment。

*Intuition*：Soap不光利用了矩阵结构，还维护了长期的梯度统计。它把梯度旋转到一个"各个方向方差最均匀"的空间里，然后在那里面跑Adam。这比Muon的stateless正交化更精细。

---

### 三、 实验数据表：打破幻觉的Scaling Law

paper中最有杀伤力的部分是实验数据。他们在130M到1.2B参数的模型上，用1x到8x的Chinchilla数据量跑了全面实验。

**Table: Optimizer Speedup Ratio 随 Model Size 的衰减**

| Model Size | Muon Speedup vs AdamW | Soap Speedup vs AdamW | 结论 |
| :--- | :--- | :--- | :--- |
| 130M | ~1.4× | ~1.3× | Matrix-based在小模型上优势明显 |
| 300M | ~1.3× | ~1.3× | 优势开始收窄 |
| 520M | ~1.3× | ~1.3× | 优势稳定 |
| 1.2B | ~1.1× | ~1.1× | 优势大幅缩水，接近AdamW |

*Intuition*：为什么模型越大，新optimizer的优势越小？因为大模型的gradient landscape可能本身就更加"isotropic"（各向同性），或者大模型训练的瓶颈已经从"optimizer的搜索方向"转移到了"纯粹的算力/data bottleneck"。这也解释了为什么像DeepSeek、Llama 3这些大厂在生产中依然死守AdamW——因为在他们的scale下，换optimizer带来的收益（1.1倍）已经盖不住工程改造的风险和memory overhead了。

---

### 四、 最Profound的发现：Data Ratio决定胜负

这篇paper揭示了一个之前完全没人注意的规律：**Optimizer的优劣排名，取决于你训练时用的data-to-model ratio（训练数据量与参数量的比值）。**

*   **低 Data Ratio (1x - 4x Chinchilla，即不过度训练)**：
    **Muon 获胜**。因为Muon是stateless的，每步直接正交化当前gradient，不需要warm up任何二阶统计量。在数据量不大、需要快速fit的场景下，它反应最快。
*   **高 Data Ratio (8x - 16x Chinchilla，即重度过度训练)**：
    **Soap / Kron 获胜**。当数据量极大时，需要精确捕捉长期的curvature信息。Soap和Kron维护了长期的 $Q_A, Q_B$ 协方差矩阵，这种long memory在overtraining阶段能榨干最后一点性能。Muon的"每步抹平"反而变得太粗暴。

*Intuition*：这就像是短跑和马拉松。Muon是短跑健将，起跑爆发力极强；Soap是马拉松选手，前期慢热但在长距离拉锯战中笑到最后。现在大厂训练模型（如Llama 3用50x Chinchilla）全是马拉松，这进一步解释了为什么Muon在实验室很火，但大厂生产端 adoption 慢。

---

### 五、 共性现象：Universal Phenomena Across Optimizers

paper在Section 4.3记录了一些与optimizer无关的普遍现象，这对build intuition非常有帮助：

1.  **Parameter norm跟着LR走**：如果有weight decay，参数矩阵的L2 norm会随着learning rate的warmup上升而上升，随着decay下降而下降。这证明了weight decay和LR schedule之间有强烈的耦合作用。
2.  **Gradient norm在LR decay阶段反升**：所有optimizer在训练末期，gradient的norm都会变大。这听起来违反直觉（loss已经收敛了，gradient反而变大？）。原因是到了训练末期，optimizer进入了loss landscape的窄谷，需要非常精细但高频率的震荡来寻找最优解，此时gradient magnitude变大，但互相抵消，导致loss不上升。
3.  **Generalization行为一致**：不同optimizer的train loss和eval loss曲线高度重合。这表明optimizer不会改变模型的泛化本质，只是改变了到达那个最小值的速度。

---

### 六、 总结与Stanford的"警告"

Andrej，这篇paper其实是给整个AI community敲了一个警钟。它的深层意思是：

**算法层面的"微创新"被严重overhyped了。** 很多所谓的进步是实验设计的不严谨导致的artifact。在公平条件下，最好的matrix-based optimizer在1B模型上也就提速1.1-1.4倍。

如果你要自己做实验或者选optimizer，这篇paper给出了几条铁律：
1.  **绝不要blind transfer hyperparameter**。每个optimizer都有自己专属的最优lr和weight decay空间，必须独立做coordinate descent sweep。
2.  **必须跑到final checkpoint再下结论**。中间的loss曲线具有极强的欺骗性。
3.  **考虑你的training regime**。如果你做重度overtraining（高data ratio），优先考虑Soap/Kron这种stateful的matrix optimizer。如果你做短训练或者小模型，Muon是最佳选择。

**Reference Links:**
*   原paper: [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.01440)
*   Muon原版blog: [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
*   Soap paper: [SOAP: Improving and Stabilizing Shampoo Using Adam](https://arxiv.org/abs/2409.11321)
*   Sophia paper: [Sophia: A Scalable Stochastic Second-Order Optimizer](https://arxiv.org/abs/2305.14342)
*   开源代码: [Marin optimizer repo on GitHub](https://github.com/marin-community/marin/tree/kaiyue/optimizers)

---

# 《Fantastic Pretraining Optimizers and Where to Find Them》深度解读

Andrej，这篇paper是Stanford的Kaiyue Wen、Tengyu Ma、David Hall、Percy Liang在2025年9月发布的，本质上是对近年来LLM pretraining optimizer领域的一次**严肃的re-evaluation**。它的核心thesis很清楚：很多人声称的新optimizer的1.4×到2× speedup，大部分是**baseline调得不够好**以及**评估方法有缺陷**造成的artifact。当你rigorously tune hyperparameter、在多个scale和data-to-model ratio下评估、并在training结束时比较，几乎所有optimizer的真实speedup都缩水了。

让我从方法论、optimizer技术细节、核心发现三个层面来讲解。

---

## 一、Methodology：三阶段Hyperparameter Tuning Framework

这篇paper最重要的贡献其实是**方法论**——它build了一个principled的benchmarking pipeline。

### Phase I: Fine-grained Coordinate Descent

对每个optimizer，定义一个discrete grid for每个hyperparameter。比如AdamW的grid包括：learning rate η、weight decay λ、warmup steps、β_1、β_2、ε、gradient-norm clipping g_norm、batch size BSZ。

从原始paper的configuration开始，做coordinate descent：每次只sweep一个hyperparameter，固定其他在当前best value，如果validation loss改进超过 Δ_1 = 3×10^{-3} 就accept新值。重复passes直到converge。

这个phase在6个settings下做：130M、300M、500M at 1× Chinchilla，以及130M at 2×、4×、8× Chinchilla。

**关键观察**：loss对大部分hyperparameter都不sensitive（小扰动不改变final loss），只有少数是scaling-sensitive的。

### Phase II: Coordinate Descent on Scaling-Sensitive Hyperparameters

定义"approximate-optimal configuration"：所有final loss落在 L_r^* + Δ_2 (其中Δ_2 = 6.4e-3)内的configuration集合 C_r。一个hyperparameter c_h是**scaling-insensitive**的当且仅当存在一个single value v_h使得对每个regime r都存在 c ∈ C_r 满足 c_h = v_h。否则就是**scaling-sensitive**的。

Table 4列出了每个optimizer的scaling-sensitive hyperparameters。比如：
- AdamW: learning rate, warmup, weight decay, batch size
- Lion: learning rate, β_2
- Muon: 只有learning rate
- Soap: learning rate, warmup, block size
- Kron: 只有learning rate

**Intuition**：大部分hyperparameter一旦调好就是universal的，只有少数几个需要随scale重新调。这个observation极大降低了benchmarking成本。

### Phase III: Hyperparameter Scaling Law

用以下functional form拟合每个scaling-sensitive hyperparameter h：

$$h(N, D) = \alpha N^{-A} D^{-B} + \beta$$

其中：
- N: model parameter count
- D: data budget (tokens)
- α, A, B, β: learned coefficients via non-linear least-squares

在12个 (N, D, h) triples上fit，然后外推到1.2B scale。验证显示predicted hyperparameter的final loss与ground-truth optimal的差距在3e-3以内。

---

## 二、各Optimizer的技术细节与Intuition

paper benchmark了11个optimizer，分为5大类。我重点讲几个有代表性的。

### 1. AdamW（Baseline）

Update rule：

$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w_t$$

变量含义：
- $w_t$: parameters at step t
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) \hat{g}_t$: first moment (gradient的EMA)
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) \hat{g}_t^2$: second moment (gradient平方的EMA)
- $\hat{m}_t = m_t / (1-\beta_1^t)$: bias-corrected first moment
- $\hat{v}_t = v_t / (1-\beta_2^t)$: bias-corrected second moment
- $\hat{g}_t$: gradient clipping后的gradient, $\hat{g}_t = g_t \cdot \max\{1, g_{\text{norm}} / \|g_t\|_2\}$
- $\eta$: learning rate
- $\lambda$: weight decay coefficient
- $\epsilon$: numerical stability (paper里tune到1e-10甚至1e-25，非常小)

**Intuition**：AdamW的核心idea是**entry-wise adaptive learning rate**。每个parameter i的有效learning rate是 $\eta / \sqrt{v_{t,i}}$，即被其gradient的历史magnitude normalize。这在gradient scale在不同parameter间heterogeneous时很有效。

但AdamW的limitation是：它treat每个parameter entry独立，**完全忽略了parameter的matrix structure**。

### 2. Nesterov AdamW (NAdamW)

$$w_{t+1} = w_t - \eta \frac{\beta_1 m_t + (1-\beta_1) \hat{g}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w_t$$

注意分子：不是用 $\hat{m}_t$（当前momentum），而是用 $\beta_1 m_t + (1-\beta_1) \hat{g}_t$（lookahead的momentum estimate）。

**Intuition**：Nesterov momentum的idea是"look ahead"——用当前gradient预测下一步的position，在那个position算gradient。NAdamW把这个idea植入到Adam框架里。它是一种**variance reduction**技术，让update更准确。

实验上NAdamW始终略微beat AdamW，虽然差距很小。

### 3. Lion

$$w_{t+1} = w_t - \eta \cdot \text{sign}(\hat{m}_t) - \eta \lambda w_t$$

其中 $\hat{m}_t = \beta_1 m_{t-1} + (1-\beta_1) \hat{g}_t$，并维护 $m_t = \beta_2 m_{t-1} + (1-\beta_2) \hat{g}_t$ 给下一步用。

**Intuition**：Lion是symbolic discovery找到的optimizer，关键innovation是**用sign函数替代Adam的element-wise normalization**。这相当于"极端"的update——每个parameter要么走 +η 要么走 -η，magnitude固定。好处是省掉 v_t 的memory（只需要 m_t），坏处是update的information content更低。

实验中Lion与AdamW comparable，但它的**optimal weight decay ≈ 0.6-0.7**，远高于AdamW的 ≈ 0.1。这说明了为什么hyperparameter不能blind transfer。

### 4. Muon（最有趣的matrix-based optimizer）

这是paper里benchmark最好的optimizer之一。核心是**Newton-Schulz orthogonalization**：

$$\text{NS}(M) = M(aM + bM^\top M + c(M^\top M)^2)$$

with appropriate (a, b, c)，可以证明当 $\|M\|_{\text{op}} < 1$ 时，迭代5次的 $\text{NS}^{(5)}(M)$ 近似于：

$$\text{NS}^{(5)}(M) \approx \arg\max_{\|O\|_{\text{op}} = 1} \text{Tr}(O^\top M)$$

也就是说，NS^(5)(M) 近似于 M 的 orthogonal projection（在operator norm约束下）。

Muon的完整update rule：

$$w_{t+1} = w_t - \eta \cdot \text{NS}^{(5)}(\beta_2 \tilde{m}_t + (1-\beta_2) g_t) - \eta \lambda w_t$$

加上一个scaling factor $s = \sqrt{\max(1, \text{rows}(w)/\text{cols}(w))}$ 来处理non-square matrices。

**Intuition**：Muon的key insight是——Transformer的weight matrices的gradient有strong spectral structure（不同singular directions的magnitude差异大）。Newton-Schulz做的事是**把gradient orthogonalize**，让update在所有singular directions上都有similar的scale。

具体来说，如果gradient G = UΣV^T (SVD)，那么NS(G) ≈ UV^T（rank-r的orthogonal matrix）。这相当于"丢弃"了singular values的信息，只保留方向。

这为什么好？因为gradient的singular values反映的是"沿着这个direction，loss变化多快"。AdamW会down-weight那些historically large gradient的directions，但这是entry-wise的。Muon在matrix level做这件事，更principled。

**重要note**：Muon只对transformer layer里的matrix parameters用Newton-Schulz，对embedding和LM head仍然用AdamW（因为这些不是真正的"matrix"参数）。

### 5. Soap (Shampoo + Adam)

Soap是Shampoo的改进版，用两个preconditioner matrices Q_A, Q_B：

Update rule:
1. 变换gradient：$\hat{g}_t = Q_A g_t Q_B$
2. 在变换后的space里跑Adam：维护 m_t, v_t（在transformed space）
3. Update：$w_{t+1} = w_t - \eta Q_A^\top \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} Q_B^\top$
4. 维护 gradient covariance estimates: $G_A = \mu G_A + (1-\mu) \hat{g}_t \hat{g}_t^\top + \epsilon I$, $G_B = \mu G_B + (1-\mu) \hat{g}_t^\top \hat{g}_t + \epsilon I$
5. 每k步更新Q_A, Q_B via QR decomposition: $Q_A = \text{QR}(G_A Q_A)$, $Q_B = \text{QR}(G_B Q_B)$

**Intuition**：Soap维护了gradient在row space和column space的二阶统计量，并用它们来preconditioning。与Muon不同，**Soap是有状态的**——Q_A, Q_B会随着训练progressively更新。这使得Soap能capture长期的gradient structure。

### 6. Kron (PSGD Kron)

Kron是Preconditioned SGD的Kronecker-factored版本。对每个parameter block，维护一组lower-triangular matrices Q_i (i=1,...,n)，preconditioner update通过random probing和conjugate sketch来估计Hessian信息。

完整的update rule很复杂（见Appendix A.10），但核心idea是：用Kronecker product $Q_1 \otimes Q_2 \otimes \cdots \otimes Q_n$ 来近似Hessian的inverse，然后用这个preconditioner乘以gradient。

**Intuition**：Kron在每一步都update preconditioner（with probability p_upd），所以它有long memory of gradient statistics。这与Muon的stateless NS不同。

### 7. Sophia-H

基于Hessian对角近似的optimizer：

$$\theta_{t+1} = \theta_t - \eta_t \cdot \text{clip}\left(\frac{m_t}{\max(\gamma h_t, \epsilon)}, 1\right)$$

其中 $h_t$ 是Hessian对角的估计，通过Hessian-vector product计算：
- 每 k 步，sample random vector r ∈ {±1}^d
- 算 $v = g_t \cdot r$ (Hessian-vector product的中间步骤)
- 算 $u = \nabla_\theta v$
- $\hat{h} = r \odot u$ (Hutchinson estimator)
- $h_t = \beta_2 h_{t-k} + (1-\beta_2) \hat{h}$

**Intuition**：Sophia用Hessian的对角来scale update，相当于"用curvature信息"来precondition。比Adam的second moment更principled，但Hessian-vector product贵且noisy。

paper发现Sophia在他们的setup下不beat AdamW（在<0.5B），原因是原paper用的data loader不是完全random shuffle，导致AdamW的最优learning rate偏小，让Sophia看起来好。

### 8. Mars

Mars是variance-reduced AdamW variant。它通过gradient differencing来reduce variance：

$$c_t = g_t + \gamma \frac{\beta_1}{1-\beta_1} (g_t - g_{t-1})$$

然后用 $c_t$ 替代 $g_t$ 进入Adam的update。这是SARAH/Stochastic Path-Integrated Differential Gradient Estimator (SPIDER)类variance reduction技术。

---

## 三、核心Empirical Findings

### Finding 1: Matrix-based optimizers consistently outperform scalar-based optimizers

Figure 1 (bottom right) 和 Figure 3 都清晰展示了这一点。

- Scalar-based optimizers（AdamW, NAdamW, Mars, Cautious, Lion, Adam-mini）的speedup ratio基本都在1.0-1.2×
- Matrix-based optimizers（Muon, Soap, Kron, Scion）的speedup ratio在1.2-1.4×

**Intuition**：这个结果告诉我们，在modern transformer架构下，gradient的matrix structure比entry-wise的heterogeneity更重要。Scalar-based optimizers只能处理"每个parameter scale不同"的问题，但无法处理"matrix的某些directions需要更aggressive update"的问题。

### Finding 2: Speedup decays with model size

从Figure 3和Figure 4 (left)可以读出：

| Model Size | Muon speedup | Soap speedup |
|------------|--------------|--------------|
| 130M | 1.4× | 1.3× |
| 300M | 1.3× | 1.3× |
| 520M | 1.3× | 1.3× |
| 1.2B | 1.1× | 1.1× |

**Intuition（speculative）**：可能的原因有几个：
1. **Gradient isotropy**: 大模型的gradient可能spectrally更"flat"（不同singular directions的magnitude差异变小），matrix preconditioning的好处减少
2. **Compute vs optimization bound**: 大模型训练更可能是compute-bound而不是optimization-bound，optimizer差异的影响相对变小
3. **Already near-optimal**: 大模型的training dynamics可能已经更接近"natural"的scaling law，留给optimizer的improvement空间小

paper还fit了一个scaling law预测7B 1× Chinchilla下Muon会比AdamW略差。

### Finding 3: Optimal optimizer depends on data-to-model ratio

- 1× Chinchilla: Muon最好
- 4× Chinchilla: Muon仍最好，但Soap/Kron开始catch up
- 8× Chinchilla: Soap/Kron超过Muon
- 16× Chinchilla: Soap明确超过Muon

**Intuition**：这是paper里最profound的发现之一。Muon的Newton-Schulz是**stateless**的——每一步都"reset"，不维护preconditioner的历史。这在"需要快速adaptation"的regime（small data-to-model ratio）下好，因为不需要warm up preconditioner。

但在high data-to-model ratio下（"overtraining" regime），需要**long-term accumulation of gradient statistics**。Soap和Kron维护的second-order momentum (Q_A, Q_B 或 Q_i)能capture这个long-term information，所以在overtraining regime下更好。

这也暗示了：**生产环境里训练frontier model（通常是heavily overtrained）时，Muon可能不如Soap/Kron**。

### Finding 4: Hyperparameter transfer is non-trivial

Figure 1 (top right)展示了一个striking example：
- Lion optimal weight decay ≈ 0.6-0.7
- AdamW optimal weight decay ≈ 0.1
- Kron optimal weight decay ≈ 0.5

**Intuition**：不同optimizer的update有very different的"effective scale"。Lion用sign函数，update magnitude固定为η；AdamW的update magnitude是η/√v_i（per-parameter）。所以为了达到similar的weight shrinkage，weight decay需要不同。

这意味着：**你不能用一个optimizer的最优hyperparameter去run另一个optimizer**。这是很多prior work的hidden pitfall。

### Finding 5: Early-stage loss curves can be misleading

Figure 5 (right)展示了一个case：某个optimizer在early training看起来更快，但后期plateau，最终被另一个optimizer超过。

**Intuition**：Training有"easy features"阶段和"hard features"阶段。Easy features是loss landscape里的"宽阔valley"，几乎所有optimizer都能快速descent。Hard features是"狭窄的"saddle或"精细的minimum"，需要precise的preconditioning。

某些optimizer（如Muon）可能在easy features上特别快（因为Newton-Schulz让early update很aggressive），但在hard features的fine-tuning上不如有long memory的Soap/Kron。

**Practical implication**：评估optimizer必须看**training结束时的loss**，不能用intermediate checkpoints。

### Finding 6: 三个Common Phenomena across all optimizers

Figure 6展示了三个普遍现象：

1. **Parameter norms track learning rate**: 当有weight decay时，所有optimizer的parameter norm都先增后减，与learning rate schedule的增减高度相关。但不同optimizer的绝对norm值差异很大。

2. **Gradient norm increases during LR decay**: 所有optimizer的gradient norm在LR decay阶段都增加，但loss不增加。这与Defazio [2025]的理论一致——LR小时，模型需要"更努力"地descent，gradient自然变大。

3. **Similar generalization behavior**: 不同optimizer的train loss和eval loss趋势几乎相同。这与architecture design不同——不同architecture（如Mamba vs Transformer）的generalization gap可能差很多，但optimizer不会。

---

## 四、Important Methodological Pitfalls Identified

### Pitfall 1: Under-tuned baseline

Figure 1 (top left)展示了一个striking example：在GPT-3 recipe下，仅tuning peak learning rate一个hyperparameter，AdamW就能获得2× speedup。这意味着很多prior work的"2× speedup over AdamW"实际上只是"2× speedup over an under-tuned AdamW"。

### Pitfall 2: Fixed hyperparameters across optimizers

很多prior work用同样的learning rate和weight decay跑不同optimizer。但如前所述，不同optimizer的optimal hyperparameter差异很大（Lion WD=0.6 vs AdamW WD=0.1）。

### Pitfall 3: Single data-to-model ratio

很多prior work只在1× Chinchilla下benchmark。但paper显示optimizer ranking在不同data-to-model ratio下会flip。如果只看1× Chinchilla，你会得出"Moon最好"的结论，但production training通常是8-50× Chinchilla。

### Pitfall 4: Intermediate checkpoint evaluation

Figure 5 (left)展示了Soap vs Mars在520M 8× Chinchilla下，如果用intermediate checkpoint会得到与final checkpoint相反的ranking。

---

## 五、Comparison with Concurrent Work (Semenov et al. 2025)

paper还讨论了与concurrent work [Semenov et al. 2025]的差异。两个work同意：
1. Non-zero weight decay和decay到small LR都essential
2. Variance-reduced AdamW variants（如Mars）有non-trivial speedup

但他们在matrix-based optimizer的relative performance上有分歧：Semenov发现AdEMaMix和Mars beat Muon，而本文发现Muon更好。

**Reason**：Batch size差异。Semenov用batch size 0.1M-0.02M tokens（1-8 H100 GPU regime），本文用≥0.4M tokens（128 TPU-v5lite ≈ 12 H100）。

在小batch regime下，gradient noise dominate，variance reduction methods (Mars, AdEMaMix) 的好处放大。在大batch regime下，noise不那么dominant，matrix preconditioning (Muon) 的好处放大。

**Implication**：Optimizer的ranking**依赖于hardware/batch size regime**。这是benchmarking时必须考虑的factor。

---

## 六、Limitations and Open Questions

paper的limitation：
- 没scale到 >1.2B
- 没考虑communication cost（DION的focus）
- 没考虑memory constraint（对于frontier training很重要）

Open questions：
1. Speedup decay with model size是否持续到7B+？scaling law预测Muon在7B 1× Chinchilla下会比AdamW差。
2. 能否设计一个optimizer在所有scale和data-to-model ratio下都optimal？
3. Matrix preconditioning的theoretical limit是什么？为什么speedup上限是1.4×而不是2×？

---

## 七、对实际Training的Practical Implications

1. **如果你的model < 1B**：Muon可能是最优选择，speedup ~1.3-1.4×
2. **如果你的model在1-7B且overtrained**：考虑Soap或Kron，speedup ~1.1-1.3×
3. **如果你的model是frontier scale (>10B)**：paper的结果暗示matrix-based optimizer的speedup可能进一步diminish，AdamW可能仍是competitive baseline
4. **永远不要blind transfer hyperparameter**：每个optimizer都需要独立的hyperparameter sweep
5. **评估必须看training结束时的loss**，不能用intermediate checkpoints

---

## References

- 原paper: [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.01440) (Wen et al. 2025)
- Muon原始blog: [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) (Jordan et al. 2024)
- Muon scaling work: [Muon is scalable for LLM training](https://arxiv.org/abs/2502.16982) (Liu et al. 2025a)
- Soap: [SOAP: Improving and Stabilizing Shampoo Using Adam](https://arxiv.org/abs/2409.11321) (Vyas et al. 2025)
- Sophia: [Sophia: A Scalable Stochastic Second-Order Optimizer](https://arxiv.org/abs/2305.14342) (Liu et al. 2024a)
- Mars: [MARS: Unleashing the Power of Variance Reduction](https://arxiv.org/abs/2411.10438) (Yuan et al. 2025)
- Lion: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) (Chen et al. 2023)
- Concurrent work: [Benchmarking optimizers for LLM pretraining](https://arxiv.org/abs/2509.01440) (Semenov et al. 2025)
- Algoperf: [Accelerating Neural Network Training: Algoperf Competition](https://arxiv.org/abs/2502.15015) (Kasimbeg et al. 2025)
- Kimi K2 (Muon in production): [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- PSGD Kron: [Preconditioned SGD](https://arxiv.org/abs/2211.04422) (Li 2022)
- Scion: [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529) (Pethick et al. 2025)
- Code: [Marin optimizer repo](https://github.com/marin-community/marin/tree/kaiyue/optimizers)
- WandB runs: [optimizer-scaling](https://wandb.ai/marin-community/optimizer-scaling)

---

## 我的Take

Andrej，这篇paper给我的核心intuition是：

**Optimizer的"算法创新"在fair benchmarking下被overhyped了**。Muon/Soap确实有genuine的matrix-structure利用，speedup是真实的，但远没有claims的2×那么夸张。真正的瓶颈可能不在optimizer本身，而在：

1. **Data quality and mixture**：这影响gradient的"shape"，进而影响optimizer的相对performance
2. **Architecture**：Mamba, hybrid, attention variants等改变了gradient structure，optimizer的optimal choice可能变
3. **Scaling regime**：从0.1B到1.2B的speedup decay强烈暗示，frontier scale下optimizer choice的marginal value在缩小

一个provocative的interpretation是：**AdamW之所以dominant，是因为在frontier scale下，optimizer choice的marginal value真的不大**。Muon/Soap的speedup主要在小model上显现，而小model不是production的重点。

另一个interesting的angle是：matrix-based optimizer的**stateful vs stateless**是一个重要的设计维度。Muon的stateless NS在small data regime好，Soap/Kron的stateful preconditioner在large data regime好。一个可能的future direction是**adaptive state**——根据training progress和data-to-model ratio动态调整preconditioner的memory length。

最后，paper的方法论贡献是meta-important的：**deep learning research需要更多这种rigorous benchmarking**。很多"新方法"的claimed improvement实际上是methodological artifact。这个pattern在optimizer领域最明显（因为容易做小实验），但在architecture、data、regularization等领域都存在。
