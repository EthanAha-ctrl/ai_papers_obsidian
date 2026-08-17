---
source_pdf: GATED DELTA NETWORKS - IMPROVING MAMBA 2 WITH DELTA RULE.pdf
paper_sha256: 55f84f2ae9c4e52ff494bfa699499867f5e9e17514994ac71de43888363a5fb9
processed_at: '2026-08-04T12:20:06-07:00'
target_folder: LLM-engine/TVM
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Gated DeltaNet 用人话讲

Andrej，我换个方式说，尽量少用术语，多打比方。

## 核心故事：一个记性好的笔记本

想象你在上课记笔记。笔记本质上就是 memory state `S_t`，每来一个新信息（key `k_t` 和 value `v_t`），你要决定怎么写到本子上。

**Linear Transformer 的做法**：来了就往上叠，`S_t = S_{t-1} + v_t k_t^T`。本子越写越满，新旧内容糊在一起。本子只有 `d_k × d_v` 那么大，信息一多就 collision，找不回来。

**Mamba2 的做法**：每次写之前，先把整个本子的内容"擦淡"一点，`S_t = α_t S_{t-1} + v_t k_t^T`。`α_t` 是橡皮擦的力度，`α_t` 越小擦得越狠。好处是本子不会爆满，坏处是擦的时候**所有内容一起擦**，你想留着的重要东西也跟着淡了。

**DeltaNet 的做法**：来了新 key，先翻本子找到旧的同 key 内容，**精准替换**掉那个位置，`S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T`。`β_t` 控制替换力度。好处是写入精准，坏处是没法"翻篇"——换 topic 时旧内容还赖在本子上不走。

**Gated DeltaNet 的做法**：两个一起用。先擦淡（gating `α_t`），再精准替换（delta rule `β_t`）：

$$S_t = S_{t-1}\bigl(\alpha_t(I - \beta_t k_t k_t^\top)\bigr) + \beta_t v_t k_t^\top$$

- `α_t → 0`：相当于"翻篇"，整个本子擦干净
- `α_t → 1`：相当于"继续记"，只做精准替换

就这么简单。两个 mechanism 是正交的，一个管全局清空，一个管局部写入。

## 为什么这两个东西是互补的

Table 2 的 S-NIAH 实验把这个故事讲得特别清楚：

**S-NIAH-1**：在重复的废话里藏一个 passkey（比如"密码是12345"），问你密码是多少。这种任务信息量很小，关键看你的本子**记不记得住**。
- DeltaNet 几乎满分：精准写入，不擦，所以记得住
- Mamba2 在 4K 之后崩了：每步都在擦，密码被擦没了
- Gated DeltaNet 表现 ok：虽然有 gating，但 delta rule 帮它把重要东西写牢了

**S-NIAH-2**：在真实 essay 里藏一个数字。这种任务信息量很大，本子很快写满，关键看你会不会**清理垃圾**。
- DeltaNet 崩了：essay 里每句话都往本子写，没法清，信息 collision，找不到 needle
- Mamba2 还行：gating 自动淡忘无关内容
- Gated DeltaNet 最好：gating 清垃圾 + delta rule 精准记 needle

**S-NIAH-3**：needle 从数字变成 UUID（比如 `a3f7-b2c1-...`）。UUID 比数字难记，关键看**写入精度**。
- Mamba2 崩了：只有简单叠加，UUID 记不牢
- DeltaNet 比 Mamba2 好
- Gated DeltaNet 最好：delta rule 帮它精确写入复杂 pattern

三个任务，正好分别测 retention、filtering、memorization。Gated DeltaNet 在三个维度上都拿到了好处。

## 从 Optimization 角度看：就是一个加了 Weight Decay 的 SGD

这个视角我觉得最 elegant。把 memory state `S_t` 当作一个 **fast weight matrix**，每来一个 `(k_t, v_t)` 样本，做一步 test-time SGD 去拟合 `S_t k_t ≈ v_t`：

Loss 是 `L(S_t) = (1/2)||S_t k_t - v_t||^2`，gradient 是 `(S_t k_t - v_t) k_t^T`。

**DeltaNet** 就是纯粹的 SGD：
$$S_{t+1} = S_t - \beta_t (S_t k_t - v_t) k_t^\top = S_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

`β_t` 就是 learning rate。这个公式和 delta rule 一模一样。

**Gated DeltaNet** 就是在这个 SGD 上加了 **weight decay** `α_t`：
$$S_t = \alpha_t \cdot S_{t-1}(\text{SGD update part})$$

Weight decay 在 deep learning 里太标准了（Krogh & Hertz 1991），就是把 weights 往零拉一点，防止 overfitting / 防止 weights 爆炸。这里 `α_t` 做的事情完全一样：把 memory state 往零拉一点，防止 memory 爆炸 / saturated。

所以从 optimization 视角：
- Mamba2 = 只有 weight decay，没有 gradient-based update（只靠 `+v_t k_t^T` 这个 Hebbian-like 项）
- DeltaNet = 只有 SGD，没有 weight decay
- Gated DeltaNet = SGD + weight decay

这个组合在 optimization literature 里是最自然的。Titans (Behrouz et al. 2024) 也独立发现了类似的思路。

## 最难的部分：怎么高效训练

RNN 的训练历来有个矛盾：要 parallel 就不能有 recurrence，要 recurrence 就不能 parallel。

**Mamba2 的 trick**：因为 decay 是 scalar `α_t`，累积 decay `γ_t = ∏ α_i` 也是 scalar，可以表示成一个 mask matrix `Γ`，然后写成 `O = ((QK^T) ⊙ Γ) V`，完全是 matmul，可以 parallel。

**DeltaNet 的问题**：transition 是 matrix-valued `(I - β_t k_t k_t^T)`，累积乘积 `∏(I - β_t k_t k_t^T)` 没法简单表示。直接展开是 `O(L × d^3)`，太慢。

**Yang et al. (2024b) 的 solution**：用 WY representation。这是 1985 年数值线性代数里的一个经典结果（Bischof & Loan），说 Householder matrices 的累积乘积可以写成低秩形式：

$$\prod_{i=1}^r (I - \beta_i k_i k_i^\top) = I - \sum_{i=1}^r w_i k_i^\top$$

其中 `w_i` 可以递推算出来。这样累积乘积就变成了一个低秩 matrix `W K^T`，可以 matmul。

**Gated DeltaNet 的扩展**：作者观察到，gating `α_t` 是 scalar，可以**吸收进** WY representation。具体来说，累积乘积变成：

$$\prod_{i=1}^r \alpha_i(I - \beta_i k_i k_i^\top) = \gamma_r \cdot \prod_{i=1}^r (I - \beta_i k_i k_i^\top) = \gamma_r (I - \sum w_i k_i^\top)$$

其中 `γ_r = ∏ α_i`。所以只要把 `γ_r` 作为 decay factor 乘上去就行，WY representation 的结构完全不变。

最终的 chunkwise algorithm 只需要在 DeltaNet 的基础上，在几个地方插入 `Γ` decay mask（UT transform 里 `diag(β) K K^T` 变成 `diag(β)(Γ ⊙ K K^T)`），其他 matmul 结构不变。这就是为什么 throughput 几乎和 DeltaNet 一样（Figure 3）。

## 架构设计的细节

Figure 1 的 block design 有几个点值得注意：

**Query/Key path**：`linear proj → short conv → SiLU → L2 norm`
- short conv 是为了 local context mixing，Mamba 系列都在用
- L2 norm 是 training stability 的关键。Ablation (Table S.1) 显示 L1 norm 的 ppl 是 30.79，L2 norm 是 27.67，差很多。原因可能是 L2 norm 把 key 限制在 unit sphere 上，避免 `k_t k_t^T` 这个 outer product 爆炸

**Alpha/Beta path**：只用 linear projection，不加 activation
- 因为 `α_t, β_t ∈ (0,1)`，通常用 sigmoid 或 softplus。但论文这里说"linear projection only"，可能是在 input 阶段就 normalize 好了

**Value path**：`linear proj → short conv → SiLU`，没有 L2 norm
- value 不需要 norm，因为它不参与 outer product，只是被加进去

**Output**：normalization → SiLU gating → output projection
- output gate 用 SiLU，ablation 显示去掉 gate ppl 从 27.35 涨到 29.12，重要

## Hybrid 模型的道理

Pure RNN 有两个已知弱点（Arora et al. 2024a）：
1. **Local shift/comparison**：比如比较相邻几个 token 的细微差异，RNN 的 fixed-size state 做不好
2. **Exact retrieval**：state 容量有限，长序列 retrieval 会 collision

**SWA（sliding window attention）** 解决第一个问题：在 2K window 内做精确 attention，local comparison 没问题。

**Gated DeltaNet** 解决第二个问题：delta rule 精准写入 + gating 清理垃圾，比 Mamba2 的 retrieval 更好。

**Mamba2** 在 hybrid 里的作用可能是提供另一种 inductive bias，或者是为了 throughput（Mamba2 比 DeltaNet 快一点）。

Table S.2 的 ablation 显示，ordering 是 `Mamba2 → Gated DeltaNet → SWA` 最好。这个 ordering 的道理可能是：先用 Mamba2 做 global context compression，再用 Gated DeltaNet 做 selective memory update，最后用 SWA 做 local precise comparison。但这个解释有点 ad hoc，需要更多实验验证。

## 实验结果的 takeaway

**Language modeling (Table 3)**：Gated DeltaNet 在 pure RNN 里最好，hybrid 进一步提升。1.3B 模型上 Gated DeltaNet-H1 的 avg accuracy 56.40，比 Mamba2 的 54.89 高 1.5 个点。

**Retrieval (Table 4)**：这是最关键的结果。Pure RNN 里 Gated DeltaNet 的 avg 30.6，超过 Mamba2 的 29.8 和 DeltaNet 的 26.2。Hybrid 模型达到 39-40，超过 Transformer++ 的 37.0。说明 Gated DeltaNet 确实在 retrieval 上有优势，而 retrieval 一直是 RNN 相比 Transformer 的最大弱点。

**LongBench (Table 5)**：Code 任务（LCC, RepoBench-P）上 Gated DeltaNet 优势明显。这很重要，因为 coding 需要 state tracking（跟踪变量作用域、括号匹配等），而 Merrill et al. (2024) 证明 SSM 有 TC^0 限制。DeltaNet 的 identity-plus-low-rank transition 比 Mamba2 的 diagonal transition 更 expressive，可能突破这个限制。

**Throughput (Figure 3)**：Gated DeltaNet 和 DeltaNet 几乎一样快，比 Mamba2 慢 2-3K tokens/sec。Hybrid 模型因为 SWA 的高效 kernel 反而更快。这说明 gated delta rule 的额外计算开销可以忽略。

## 一些更深的联想

**和 LSTM 的对比**：LSTM 有 forget gate、input gate、output gate，都是 vector-valued。Gated DeltaNet 的 `α_t` 相当于 forget gate，`β_t` 相当于 input gate，但 state 是 matrix-valued 而非 vector-valued，且 transition 是 Householder 而非 element-wise。这个 matrix-valued state 是关键区别，它让 memory capacity 从 `O(d)` 提升到 `O(d^2)`。

**和 Titans 的对比**：Titans 也在 test-time SGD 上加 weight decay，但 Titans 用 nonlinear function `f_S(k)` 而 Gated DeltaNet 用 linear `S k`。Titans 的 expressiveness 更高，但需要 chunk-level nonlinear update，parallelism 更差。Gated DeltaNet 保持了 linear recurrence，能 fully parallelize。

**和 RWKV-7 的对比**：RWKV-7 用 `diag(d_t) - a_t b_t^T` 的 diagonal-plus-low-rank transition。`diag(d_t)` 是 vector-valued decay（每个 dimension 独立 decay），比 Gated DeltaNet 的 scalar `α_t` 更 expressive。但 Gated DeltaNet 的 Householder `(I - β_t k_t k_t^T)` 是 data-dependent low-rank，而 RWKV-7 的 `a_t b_t^T` 也是 data-dependent low-rank，两者结构类似。RWKV-7 的 formalism 更一般化，但 Gated DeltaNet 的 WY representation 更 clean。

**Memory capacity 理论**：Gardner (1988) 证明 delta rule 的 capacity 是 `O(d_k)`（能存 `d_k` 个 orthogonal key-value pairs），而 Hebbian learning 的 capacity 只有 `O(d_k / log d_k)`。这是 DeltaNet 在 synthetic retrieval 上远超 linear attention 的根本原因。但 capacity 仍然有限，长序列必须靠 gating 清理。

**State tracking 和 TC^0**：Merrill et al. (2024) 证明 SSM（包括 Mamba2）的状态转移矩阵如果是 diagonal 的，只能表达 TC^0 complexity class 的 computation。DeltaNet 的 `(I - β_t k_t k_t^T)` 是 identity-plus-low-rank，理论上更 expressive。Grazzi et al. (2024) 提出 negative eigenvalues 可以进一步增强 state tracking。Siems et al. (2025) 的 DeltaProduct 用多个 Householder product 乘积实现 high-rank transformation。这些方法都可以直接 apply 到 Gated DeltaNet 上。

**未来的方向**：
1. Nonlinear recurrence（TTT, Titans 方向）vs. linear recurrence（Gated DeltaNet）的 trade-off 还没完全探索清楚
2. Vector-valued gating（像 RWKV-7 的 `diag(d_t)`）可能比 scalar gating 更好，但需要新的 chunkwise algorithm
3. 更大的 scale（7B+）和 instruction tuning 下的表现还没验证
4. Multi-head Gated DeltaNet 的 head 间 interaction 怎么设计最优

## 一句话总结

Gated DeltaNet = DeltaNet 的精准写入 + Mamba2 的全局清空，用 WY representation 把两者无缝融合成一个 hardware-efficient 的 linear RNN，在 retrieval、long context、code 等任务上全面超过 Mamba2 和 DeltaNet。

论文地址：https://arxiv.org/abs/2412.06464  
代码：https://github.com/NVlabs/GatedDeltaNet

---

# Gated DeltaNet 深度解析

Andrej，这篇论文的核心 idea 非常 elegant，让我从 intuition 到 technical details 完整地讲一遍。

## 1. 核心 Insight：两种 Memory 机制的互补性

Linear Transformer 本质上是把 attention 重新表述为 matrix-valued state 的 linear RNN。state `S_t ∈ R^{d_v × d_k}` 存储的是 key-value associations 的 outer product 累积。问题在于这个 memory 的**写入**和**遗忘**机制。

现有两种极端：

**Mamba2 的 gating**：`S_t = α_t S_{t-1} + v_t k_t^T`，其中 `α_t ∈ (0,1)` 是 data-dependent scalar decay。这个机制能**整体快速擦除**所有历史记忆（设 α_t→0），代价是**无法选择性更新**某个特定 key-value pair。如果要忘记某个 association，所有 association 都被同等衰减。

**DeltaNet 的 delta rule**：`S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T`，其中 `β_t ∈ (0,1)` 是 writing strength。这个机制能**精确替换**与当前 key `k_t` 关联的旧 value（通过 Householder reflection `I - β_t k_t k_t^T`），代价是**缺乏全局清除能力**。context switch 时无法快速丢弃过时信息。

Gated DeltaNet 的 insight 就是把两者 unify：

$$S_t = S_{t-1}\bigl(\alpha_t(I - \beta_t k_t k_t^\top)\bigr) + \beta_t v_t k_t^\top$$

变量解释：
- `S_t ∈ R^{d_v × d_k}`：时刻 t 的 memory state matrix，d_v 是 value 维度，d_k 是 key 维度
- `α_t ∈ (0,1)`：data-dependent gating term，控制整体 decay
- `β_t ∈ (0,1)`：data-dependent writing strength，控制 delta update 的强度
- `k_t ∈ R^{d_k}`：当前 input key
- `v_t ∈ R^{d_v}`：当前 input value
- `I ∈ R^{d_k × d_k}`：identity matrix
- 上标 `⊤` 表示 transpose

当 `α_t → 0`：快速清除整个 memory；当 `α_t → 1`：退化为 pure delta rule，做 selective update。

这个 formulation 让我联想到 LSTM 里 forget gate 和 input gate 的关系，不过这里 state 是 matrix 而非 vector，且 transition matrix 是 Householder 形式而非对角。

## 2. 从 Online Learning 视角理解

Table 1 是这篇论文最深刻的 insight 之一。Liu et al. (2024) 的框架把 RNN state update 看作 online learning 问题的 closed-form 解。不同模型对应不同的 online objective：

**Linear Attention**：
$$\min_{S_t} \|S_t - S_{t-1}\|_F^2 - 2\langle S_t k_t, v_t\rangle$$
- 第一项是 regularization，防止 state 偏离前一步太远（memory retention）
- 第二项是 negative inner-product loss，鼓励 `S_t k_t` 对齐 `v_t`
- `||·||_F` 是 Frobenius norm，`⟨·,·⟩` 是 inner product

**Mamba2**：
$$\min_{S_t} \|S_t - \alpha_t S_{t-1}\|_F^2 - 2\langle S_t k_t, v_t\rangle$$
- 把 regularization target 从 `S_{t-1}` 改成 `α_t S_{t-1}`，引入 adaptive scaling
- 当 state saturated 时，`α_t` 可以放松 regularization，允许 controlled deviation

**DeltaNet**：
$$\min_{S_t} \|S_t - S_{t-1}\|_F^2 - 2\langle S_t k_t, \beta_t(v_t - S_{t-1}k_t)\rangle$$
- loss 里出现了 `v_t - S_{t-1}k_t`，这是**prediction error**（old value 与 new value 的差异）
- 这就是 delta rule 的本质：用 prediction error 驱动更新

**Gated DeltaNet**：
$$\min_{S_t} \|S_t - \alpha_t S_{t-1}\|_F^2 - 2\langle S_t k_t, \beta_t(v_t - \alpha_t S_{t-1}k_t)\rangle$$
- 同时 relax regularization（`α_t S_{t-1}`）和引入 prediction error（`v_t - α_t S_{t-1}k_t`）

这个视角让 Gated DeltaNet 的设计变得**自然**：它就是在 DeltaNet 的 objective 上加 adaptive scaling，或者在 Mamba2 的 objective 上引入 prediction error term。

## 3. Fast Weight / Test-Time Training 视角

另一个理解角度：把 `S_t` 当作 **fast weight matrix**，delta rule 是对 online regression objective 做 test-time SGD：

$$\mathcal{L}(S_t) = \frac{1}{2}\|S_t k_t - v_t\|^2$$

SGD update：
$$S_{t+1} = S_t - \beta_t \nabla \mathcal{L}(S_t) = S_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

- `β_t` 是 adaptive learning rate
- `S_t k_t - v_t` 是 gradient 的核心部分

Gated delta rule 相当于在这个 SGD 上加了 **weight decay** `α_t`，这是 deep learning 里广泛使用的技术（Krogh & Hertz, 1991; Andriushchenko et al., 2023）。Concurrent work Titans (Behrouz et al., 2024) 也验证了这个思路。

这个类比让我想到：Mamba2 像是只有 weight decay 没有 gradient-based update 的优化器，DeltaNet 像是只有 SGD 没有 weight decay，Gated DeltaNet 是两者的结合——这在 optimization literature 里是标准组合。

## 4. Chunkwise Parallel Algorithm（最 technical 的部分）

这是论文最硬核的部分。问题在于：gated delta rule 的 recurrence 包含 matrix 乘积（Householder matrices），如何高效并行化？

### 4.1 为什么不能直接用 parallel form？

Mamba2 能用 `O = ((QK^T) ⊙ Γ) V` 的 parallel form，因为 decay 是 scalar，累积成 `Γ` mask 即可。但 gated delta rule 的 transition 是 `α_t(I - β_t k_t k_t^T)`，是 matrix-valued，累积乘积无法简单表示。

### 4.2 WY Representation

Yang et al. (2024b) 的关键 insight：用 **WY representation**（Bischof & Loan, 1985）把 Householder matrices 的累积乘积转化为低秩形式。

对于 DeltaNet，展开 recurrence 得到：
$$S_{[t]}^r = S_{[t]} P_{[t]}^r + H_{[t]}^r$$

其中：
- `S_{[t]}`：chunk t 的 initial state
- `P_{[t]}^r = ∏_{i=1}^r (I - β_{[t]}^i k_{[t]}^i k_{[t]}^{i⊤})`：累积 Householder 乘积
- `H_{[t]}^r`：累积的 value 写入项
- 下标 `[t]` 表示 chunk t，上标 `r` 表示 chunk 内第 r 个位置

WY representation 把 `P_{[t]}^r` 表示为：
$$P_{[t]}^r = I - \sum_{i=1}^r w_{[t]}^i k_{[t]}^{i⊤}$$

其中 `w_{[t]}^r` 通过递推计算：
$$w_{[t]}^r = \beta_{[t]}^r \left(k_{[t]}^r - \sum_{i=1}^{r-1} w_{[t]}^i (k_{[t]}^{i⊤} k_{[t]}^r)\right)$$

- `w_{[t]}^r ∈ R^{d_k}`：WY representation 的 auxiliary vector
- `k_{[t]}^{i⊤} k_{[t]}^r`：key 之间的 inner product

### 4.3 UT Transform

Joffrain et al. (2006) 的 UT transform 把递推写成矩阵形式：
$$T_{[t]} = \left[I + \text{strictLower}(\text{diag}(\beta_{[t]}) K_{[t]} K_{[t]}^\top)\right]^{-1} \text{diag}(\beta_{[t]})$$

- `T_{[t]} ∈ R^{C×C}`：C 是 chunk size
- `strictLower(·)`：严格下三角部分
- `diag(β_{[t]})`：β 的对角矩阵
- `K_{[t]} ∈ R^{C×d_k}`：chunk t 的 key matrix

然后：
$$W_{[t]} = T_{[t]} K_{[t]}, \quad U_{[t]} = T_{[t]} V_{[t]}$$

- `W_{[t]} ∈ R^{C×d_k}`：WY vectors 堆叠
- `U_{[t]} ∈ R^{C×d_v}`：value 的 transformed 版本

### 4.4 Gated DeltaNet 的扩展

对于 gated delta rule，展开 recurrence：
$$S_{[t]}^r = S_{[t]} F_{[t]}^r + G_{[t]}^r$$

其中：
$$F_{[t]}^r = \prod_{i=1}^r \alpha_{[t]}^i (I - \beta_{[t]}^i k_{[t]}^i k_{[t]}^{i⊤})$$

关键观察：`F_{[t]}^r = γ_{[t]}^r P_{[t]}^r`，其中 `γ_{[t]}^r = ∏_{i=1}^r α_{[t]}^i` 是累积 gating。所以 gating 可以**吸收进** WY representation 的 decay terms。

对于 `G_{[t]}^r`，adapt DeltaNet 的形式：
$$G_{[t]}^r = \sum_{i=1}^r \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i} \tilde{u}_{[t]}^i k_{[t]}^{i⊤}$$

其中 `γ_{[t]}^r / γ_{[t]}^i` 是从位置 i 到位置 r 的相对 decay。

最终的矩阵形式：
$$\tilde{U}_{[t]} = \left[I + \text{strictLower}(\text{diag}(\beta_{[t]}) (\Gamma_{[t]} \odot K_{[t]} K_{[t]}^\top))\right]^{-1} \text{diag}(\beta_{[t]}) V_{[t]}$$

- `Γ_{[t]}`：decay-aware mask，`(Γ_{[t]})_{ij} = γ_{[t]}^i / γ_{[t]}^j`
- `⊙`：Hadamard product

这把 DeltaNet 的 chunkwise algorithm 扩展为 gated 版本，只需要在 UT transform 里插入 `Γ_{[t]}` mask，computational overhead 极小。

### 4.5 最终的 Chunkwise Algorithm

State update 和 output：
$$S_{[t+1]} = \overrightarrow{S_{[t]}} + (\overleftarrow{\tilde{U}_{[t]}} - \overleftarrow{\tilde{W}_{[t]}} S_{[t]}^\top)^\top \overrightarrow{K_{[t]}}$$

$$O_{[t]} = \overleftarrow{Q_{[t]}} S_{[t]}^\top + (Q_{[t]} K_{[t]}^\top \odot M)(\overleftarrow{\tilde{U}_{[t]}} - \overleftarrow{\tilde{W}_{[t]}} S_{[t]}^\top)$$

箭头符号（Eq. 2）：
- `←q_{[t]}^r = γ_{[t]}^r q_{[t]}^r`：把 query decay 到 chunk 起始位置
- `→k_{[t]}^r = (γ_{[t]}^C / γ_{[t]}^r) k_{[t]}^r`：把 key decay 到 chunk 末尾位置
- `→S_{[t]} = γ_{[t]}^C S_{[t]}`：state 在整个 chunk 上的 decay
- `M`：causal mask

这个算法全是 matmuls，能充分利用 tensor cores，实现 hardware-efficient training。

## 5. S-NIAH Case Study：直觉验证

Table 2 的三个 observation 非常清晰地验证了设计 intuition：

**S-NIAH-1 (pass-key retrieval)**：合成重复 context，测试长期 retention。DeltaNet 接近完美（97-99%），Mamba2 在 4K 后暴跌（65.4% → 30.4%）。Gated DeltaNet 轻微下降（88.4% → 91.8%，有波动）。结论：**decay 伤害 retention**，delta rule 的精确写入更好。

**S-NIAH-2 (number in haystack)**：真实 essay context，测试 memory management。DeltaNet 在 4K 后崩溃（45.6% → 18.6% → 14.4%），因为无法清除无关信息导致 memory collision。Mamba2 和 Gated DeltaNet 通过 gating 过滤无关信息，保持更好。结论：**gating 促进 filtering**。

**S-NIAH-3 (uuid in haystack)**：value 从 number 变成 UUID，测试复杂 pattern memorization。Mamba2 快速下降（64.4% → 4.6%），Gated DeltaNet 更好（86.6% → 27.6%）。结论：**delta rule 增强 memorization**。

这三个 observation 完美对应了 gating 和 delta rule 的互补性：retention 需要 delta，filtering 需要 gating，memorization 需要 delta。

## 6. 架构设计

Figure 1 展示了 block design：
- **Query/Key path**：linear projection → short conv → SiLU → L2 norm
- **Value path**：linear projection → short conv → SiLU
- **Alpha/Beta path**：linear projection only（不需要非线性，因为要保持在 (0,1)）
- **Output**：normalization → gating（SiLU）→ output projection

Macro architecture 沿用 Llama：token mixer layers 交替 SwiGLU MLP layers。

Hybrid 变体：
- **GatedDeltaNet-H1**：Gated DeltaNet + SWA（sliding window attention）
- **GatedDeltaNet-H2**：Mamba2 + Gated DeltaNet + SWA

SWA 的作用是弥补 linear recurrent models 在 local shifts/comparisons 上的弱点。Mamba2 的作用可能是提供不同的 inductive bias。

## 7. 实验结果分析

### 7.1 Commonsense Reasoning (Table 3)

1.3B 模型，100B tokens，FineWeb-Edu：
- Gated DeltaNet：Avg 55.32（超过 Mamba2 的 54.89，DeltaNet 的 52.14）
- Gated DeltaNet-H1：56.40（最高）
- Gated DeltaNet-H2：56.18

WikiText ppl：Gated DeltaNet 16.42 < Mamba2 16.56 < DeltaNet 17.71

### 7.2 In-context Retrieval (Table 4)

真实世界 retrieval 任务（SWDE, SQuAD, FDA, TriviaQA, NQ, DROP）：
- Gated DeltaNet：Avg 30.6（超过 Mamba2 的 29.8 和 DeltaNet 的 26.2）
- Hybrid 模型达到 39-40，接近 Transformer++ 的 37.0 并超过

注意：DeltaNet 在合成 retrieval 上很强，但真实世界落后于 Mamba2，这验证了 gating 的 filtering 价值。Gated DeltaNet 同时获得两者的优势。

### 7.3 Length Extrapolation (Figure 2)

训练 4K，外推到 20K。Gated DeltaNet 在 RNN 模型中 perplexity 最低，hybrid 模型进一步改善。

### 7.4 LongBench (Table 5)

14 个长 context 任务。Gated DeltaNet 在 single-doc QA、few-shot、code 任务上优势明显，反映其 retrieval、in-context learning、state tracking 能力。Code 任务（LCC, RepoBench-P）尤其重要，因为 coding 需要 state tracking beyond TC^0 complexity（Merrill et al., 2024）。

### 7.5 Throughput (Figure 3)

Gated DeltaNet 与 DeltaNet throughput 基本相同，比 Mamba2 慢 2-3K tokens/sec（因为更复杂的 transition matrices）。Hybrid 模型因为 SWA 的高效 kernel 反而更快。

## 8. Ablation Studies

Table S.1 的关键发现：
- **Naive delta rule（无 gating）**：ppl 30.87 vs 27.35，掉了很多，验证 gating 重要性
- **Short conv**：去掉 ppl 28.95，重要
- **Output gate**：去掉 ppl 29.12，重要
- **L2 norm**：L1 norm 明显更差（30.79 vs 27.67），L2 norm 对 stability 关键
- **Feature map**：SiLU 最好但差异不大（27.35 vs ReLU 27.67 vs 1+ELU 27.58）
- **Head dim**：128 最优，256 略好但计算开销大

Table S.2 的 hybrid ordering：Mamba2 + Gated DeltaNet + SWA 顺序最好（ppl 23.54）。

## 9. 相关联想与扩展

### 9.1 与 RWKV-7 的关系

RWKV-7 用 diagonal-plus-low-rank transitions：`S_t = S_{t-1}(diag(d_t) - a_t b_t^T) + v_t k_t^T`。这比 Gated DeltaNet 的 scalar gating 更一般化（`diag(d_t)` 是 vector-valued decay），但 formalism 更 relax。Flash Linear Attention 库已实现。

### 9.2 Memory Capacity 理论

Gardner (1988) 和 Prados & Kak (1989) 证明 delta rule 的 memory capacity 远超 Hebbian learning。DeltaNet 用 delta rule，linear transformers 用 Hebbian-like rule。这解释了 DeltaNet 在 synthetic in-context learning 上的优势。

### 9.3 State Tracking 和 TC^0

Merrill et al. (2024) 证明 state space models 有 TC^0 complexity 限制。DeltaNet 的 identity-plus-low-rank structure `(I - β_t k_t k_t^T)` 比 Mamba2 的 diagonal `(α_t I)` 更 expressive，可能突破这个限制。Grazzi et al. (2024) 和 Siems et al. (2025) 提出用 negative eigenvalues 和 multiple Householder products 进一步增强。

### 9.4 Nonlinear Recurrence 的 trade-off

Irie et al. (2021, 2022b) 尝试非线性 recurrence 增强 expressiveness，但牺牲 parallelism。TTT (Sun et al., 2024a) 和 Titans (Behrouz et al., 2024) 在 chunk 级别做非线性 update。Mesa layer (von Oswald et al., 2024) 用 RLS-style objective 考虑整个 history。

### 9.5 Hybrid Architectures 趋势

Griffin (De et al., 2024)、Samba (Ren et al., 2024)、MiniMax-01 都在探索 linear recurrent + attention 的混合。Gated DeltaNet 的 hybrid 版本延续这个趋势。单层内的 hybrid（Zancato et al., 2024; Dong et al., 2025）是另一个方向。

## 10. 我的 Intuition 总结

Gated DeltaNet 的 elegance 在于它把两个看似不同的 memory 机制统一到一个 online learning framework 下。从 optimization 视角：
- Mamba2 = SGD with weight decay only
- DeltaNet = SGD without weight decay
- Gated DeltaNet = SGD with adaptive weight decay

这几乎是 deep learning 里最自然的组合。而且 hardware-efficient algorithm 的推导显示，gating 可以**无缝融入** DeltaNet 的 WY representation，只需在 UT transform 里加一个 decay mask，这让 implementation 成本极低。

更深层的 intuition：linear RNN 的 state 是一个**容量有限的 memory buffer**。Delta rule 决定**如何写入**（精确替换 vs. 叠加），gating 决定**何时清空**（全局衰减 vs. 保留）。两者的正交性让 Gated DeltaNet 能同时优化写入质量和清空时机。

从 fast weight programming 视角，这呼应了 Schmidhuber 早年关于 self-referential weight matrices 的思想（Irie et al., 2022b）——network 的 weights 在 test time 通过 learning rule 自我修改，delta rule 是最 principled 的修改方式，weight decay 是最自然的 regularization。

## References

- 论文：https://arxiv.org/abs/2412.06464
- 代码：https://github.com/NVlabs/GatedDeltaNet
- Mamba2 (SSD)：https://arxiv.org/abs/2405.21060
- DeltaNet (Yang et al. 2024b)：https://proceedings.mlr.press/v235/yang24b.html
- WY Representation (Bischof & Loan 1985)：https://api.semanticscholar.org/CorpusID:36094006
- UT Transform (Joffrain et al. 2006)：https://api.semanticscholar.org/CorpusID:15723171
- Longhorn (Liu et al. 2024)：https://arxiv.org/abs/2407.14207
- Titans (Behrouz et al. 2024)：https://arxiv.org/abs/2501.00663
- TTT (Sun et al. 2024a)：https://arxiv.org/abs/2407.04620
- Flash Linear Attention：https://github.com/sustcsonglin/flash-linear-attention
- RULER (Hsieh et al. 2024)：https://arxiv.org/abs/2404.06654
- Zoology (Arora et al. 2023a)：https://arxiv.org/abs/2312.04927
- Merrill et al. 2024 (TC^0)：https://arxiv.org/abs/2404.08819
- Grazzi et al. 2024 (negative eigenvalues)：https://api.semanticscholar.org/CorpusID:274141450
- DeltaProduct (Siems et al. 2025)：https://arxiv.org/abs/2502.10297
- RWKV-7 (Eagle/Finch)：https://arxiv.org/abs/2404.05892
- Samba (Ren et al. 2024)：https://arxiv.org/abs/2406.07522
- Griffin (De et al. 2024)：https://arxiv.org/abs/2402.19427
- MiniMax-01：https://arxiv.org/abs/2501.08313
- Linear Transformers (Katharopoulos et al. 2020)：http://proceedings.mlr.press/v119/katharopoulos20a.html
- Fast Weight Programmers (Schlag et al. 2021)：http://proceedings.mlr.press/v139/schlag21a.html
- Weight Decay (Krogh & Hertz 1991)：https://api.semanticscholar.org/CorpusID:10137788
- Gardner 1988 (memory capacity)：https://api.semanticscholar.org/CorpusID:15378089
