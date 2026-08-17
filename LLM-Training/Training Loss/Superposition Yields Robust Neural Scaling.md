---
source_pdf: Superposition Yields Robust Neural Scaling.pdf
paper_sha256: 162bf1c6bc1454ee58d81f93a96f9aa507a71cad5f2bbe08d883d02c15f7e772
processed_at: '2026-08-12T11:30:52-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话说清楚

这篇 paper 想回答一个特别 fundamental 的问题: **为什么 model 越大 loss 越低, 而且是漂亮的 power law?** 他们给出的答案特别 elegant — 这事儿跟 data distribution 关系没那么大, 根源在 **superposition 的几何**.

## Setup 一下你的 intuition

先想象一个特别简单的场景. 你有一个 m-dim 的 hidden space (比如 m = 100), 你要 represent 一堆 features, 数量是 n (比如 n = 10000), 远远多于 m. 每个 feature i 有一个 frequency $p_i$, 表示它在 data 中出现的概率. Frequency 随 rank 递减 — 第 1 个最常见, 第 10000 个最罕见.

你有两种策略:

**策略 A (weak superposition)**: 挑前 m 个最 frequent 的 features, 给它们每个一个 orthogonal direction, 完美 represent, 剩下的全扔掉. 这种情况下, loss 来自那些被扔掉的 features — 就是它们 frequency 的 sum.

**策略 B (strong superposition)**: 把所有 n 个 features 都塞进 m-dim space, 每个分一个 vector, 但这些 vector 肯定要 overlap. Loss 来自 overlap 干扰.

这篇 paper 核心就是: 这两种策略给出的 scaling law 非常不一样.

## Weak Superposition — "Power law in, power law out"

策略 A 下, loss 就是 tail sum:

$$L \approx \frac{4}{3} \sum_{i > m} p_i$$

(这里 4/3 是因为 $v \sim U(0,2)$ 给 $\langle v^2 \rangle = 4/3$)

如果 $p_i \propto 1/i^\alpha$:

$$\sum_{i > m} \frac{1}{i^\alpha} \sim \int_m^\infty \frac{di}{i^\alpha} \sim m^{-(\alpha-1)}$$

所以 $\alpha_m = \alpha - 1$.

这里关键 insight: **scaling exponent 完全由 data distribution 的 tail 决定**. Data 不是 power law, loss 就不是 power law. 你要 power law out, 必须 power law in.

这其实就是之前大部分 scaling law 理论 [15-20] 隐含的 regime — 它们都假设某种 power-law importance spectrum.

## Strong Superposition — 几何接管一切

策略 B 下, 情况完全不同. 假设只有 feature j 激活, 输出在 feature i 上的 leakage 是 $W_i \cdot W_j$, loss 贡献是 $(W_i \cdot W_j)^2$.

最简单的 ansatz: $W_i$ 是 m-dim unit sphere 上的随机向量. 那两个随机 unit vector 的 dot product² 服从 Beta(1/2, (m-1)/2), mean 是 1/m.

所以:

$$L \sim \sum_j p_j \cdot \sum_{i \neq j} \frac{1}{m} \sim \frac{1}{m} \cdot \langle\text{激活数}\rangle$$

得到 $\alpha_m \approx 1$.

关键点: **这个 1/m 跟 data distribution 长什么样完全无关**. 不管 $p_i$ 是 power-law、exponential、linear, squared overlap 都是 1/m. 这是 high-dimensional geometry 的天然结果.

你想象一下, 在 m-dim space 中随便撒一把 unit vectors, 它们之间的 cosine similarity 平方值就是 ~1/m. 这是 high-dimensional probability 的基础 fact, 跟你的 vectors 怎么分布无关.

## 实际训练的 vectors 不是完全随机 — 它们更像是 ETF

实际训练出来的 $W_i$ 有 structure. 作者发现 norm 分布是 bimodal, 一堆 near 1, 一堆 near 0. Norm > 1 的那批 (strongly represented) 行为很像 **Equiangular Tight Frame (ETF)**.

ETF 是什么? 就是 m-dim space 里塞 ν 个 vectors (ν 可以远大于 m), 让所有 pairwise absolute overlap 都相等, 达到 Welch bound:

$$\max_{i \neq j} |w_i \cdot w_j| \geq \sqrt{\frac{\nu - m}{m(\nu - 1)}} \equiv \kappa \approx \sqrt{1/m}$$

所以 $\kappa^2 \approx 1/m$. ETF 的 variance 是 0, 比随机 vectors 的 variance ~2/m² 还小. 实验上 important features 确实更 ETF-like (Figure 5c, d).

Real space 里 ETF 上限 $\nu \leq m(m+1)/2 \approx m^2/2$. 实验测出来 strongly represented features 数量确实 ~ $m^2/2$ (Figure 16). 漂亮.

## Skewed frequencies 下会 break

当 α 很大 (frequency 分布特别 skewed), important features 会占据更多 angular space, vectors 不再 isotropic. 极端 conjecture: 前 $m^2/2$ 个 features ETF-like (loss 可忽略), 剩下 weakly represented 的 loss 主导:

$$\sum_{i > m^2/2} p_i \sim (m^2)^{-(\alpha-1)} \sim m^{-2(\alpha-1)}$$

所以 $\alpha_m \to 2(\alpha-1)$. 实验上 Figure 5e 显示 $\alpha_m$ 确实从 1 过渡到这个上界.

但这个 transition 没有严格解, 作者也只是 conjecture. 严格解 toy model 是 future work.

## 怎么控制 superposition degree? — Decoupled weight decay

这是个挺 trick 的 move. 对 W 的每一行单独做 weight decay:

- $\gamma > 0$: 标准 weight decay, unimportant features norm → 0 → weak superposition
- $\gamma < 0$: 对应 minimize $(\|W_i\| - 1)^2$, 鼓励 unit norm → strong superposition

用 $\phi_{1/2} = |\{i: \|W_i\| > 1/2\}|/n$ 来 measure superposition degree:
- Weak: $\phi_{1/2} \approx m/n$
- Strong: $\phi_{1\.5} \approx 1$

这个 trick 让作者能在同一个 model 里 sweep superposition degree, 做出 Figure 9b 的相图.

## LLMs 是 strong superposition

证据链:
1. Vocabulary $n \approx 50k$, model dim $m$ 几百到几千 → $n \gg m$
2. 所有 tokens 都有 non-zero representation (Figure 20)
3. Token frequency 是 power law, 但 α ≈ 1 (Figure 22) — 这是 "flat" end, 落在 robust regime

然后他们直接 measure LM head 的 rows 之间的 mean squared overlap (Figure 6a), 发现确实 ~1/m, 跨四个 model family (OPT, GPT-2, Qwen2.5, Pythia).

## Cross-entropy loss 怎么跟 squared overlap 挂钩?

这部分在 Appendix A.2. 假设 hidden state 是 $W_i/\|W_i\|$, target token 是 i. Cross-entropy:

$$L = \ln\left[1 + \sum_{j \neq i} e^{W_i \cdot W_j/\|W_i\| - \|W_i\|}\right]$$

由于 overlap ~1/m ≪ 1, Taylor expand:

$$L \approx (n-1)e^{-\|W_i\|} + \frac{\epsilon_{D,i} e^{-\|W_i\|}}{\|W_i\|} + \frac{1}{2}\sum_j \left(\frac{W_i \cdot W_j}{\|W_i\|}\right)^2 e^{-\|W_i\|}$$

- 第一项: baseline, 跟 m 无关 (intrinsic uncertainty)
- 第二项: data correlation 修正
- 第三项: 跟 m 相关, 因为 cosine sim² ~ 1/m

关键 assumption: $\|W_i\|$ 跟 m 无关 (由 intrinsic data property 控制, 不能无限大否则 hidden space 过度 sharpen). 这个 Figure 20 验证了.

所以 cross-entropy loss 中跟 model size 相关的部分 ~1/m, 跟 squared error loss 一样.

## Fitting 实际 LLM loss

用:

$$L = C_m / m^{\alpha_m} + L_{\backslash m}$$

其中 $L_{\backslash m}$ 是 dataset/model class specific offset (不同 dataset 不同). 跨 4 个 model family × 4 个 dataset 一起 fit, 得 $\alpha_m = 0.91 \pm 0.04$.

## Chinchilla consistency check

Chinchilla: $L \propto N^{-\alpha_N}$, $\alpha_N = 0.35$. 实测 $N \propto m^{2.52}$. 所以 $\alpha_m = 2.52 \times 0.35 = 0.88 \approx 1$. 跟 toy model prediction 吻合. 这个 quantitative consistency 是 paper 最 satisfying 的 moment.

## 我觉得最 elegant 的几个 insight

1. **Scaling 的 robustness 来自 geometry, 不来自 data**. High-dim 空间里, 任意两个 unit vectors 的 dot product² 期望就是 1/m, 这是数学 fact, 不需要任何 data assumption. 这就解释了为什么 neural scaling law 在各种 architecture、各种 task 上都 universal — 因为底层的几何 universal.

2. **Weak superposition 是 "data-limited"**, strong superposition 是 "geometry-limited". 前者 scaling exponent 跟 data distribution 死死绑定, 后者被 high-dim 几何 robustly 钉死在 1 附近.

3. **LLMs 落在 strong superposition + flat frequency (α≈1)**, 这是 robustness 最强的 corner. 这就是为什么实际 LLM scaling law 看起来这么 universal, exponent 这么稳定.

4. **ETF 是 error correction 的最优 configuration**. To do superposition 不崩, 你需要 negative bias 去 cancel interference, 而 ETF (uniform overlap) 让 max overlap 最小, 最容易 cancel. 所以训练 dynamics 自然 drive 向 ETF-like structure. 这是个 self-organization 现象.

## 但这 paper 也有几个 weakness

1. **Toy model 没严格 solve**. ETF ansatz 只在 small α 严格. Large α regime 只是 conjecture, Figure 18 显示 $m^2/2 > n$ 时还是随 α 变, 说明 simple strongly/weakly dichotomy 不够.

2. **LLM 验证是 correlational**. 只展示 overlap ~1/m 和 loss ~1/m 一致, 没做 intervention (比如 enforce ETF structure 看 loss 是否符合).

3. **Token-as-feature 是 naive 假设**. Anthropic 自己的工作就显示真实 atomic features 远多于 vocabulary tokens. 所以 "m 接近 vocabulary size 时 scaling 停止" 这个 prediction 可能不对.

4. **Parsing loss 没独立 measure**. 论文假设 representation loss 和 parsing loss 在 optimal m-ℓ balance 下 comparable, 但没实验直接分离. 真实 LLM 的 loss 有多少来自 representation、多少来自 transformer layers 的 processing, 不清楚.

5. **Depth-width relationship 是 black box**. $N \propto m^{2.5}$ 是 empirical observation, 为什么是 2.5 没解释. 这个 exponent 直接决定 $\alpha_N$ 和 $\alpha_m$ 的 mapping.

## 对实际 LLM 开发的 implication

作者提了几个挺有意思的点:

1. **想加速 scaling exponent? 对 natural language 不行**, 因为 α≈1 已经是 robust 几何下界. 但对 super-skewed domain tasks 可能可以 (回到 weak-superposition-like 行为).

2. **nGPT (在 unit sphere 上做 representation) 和无 weight decay 的 optimizer (如 FOCUS) 可能本质上 promote superposition**, 进而 improve scaling. 但作者承认这些主要 alter coefficient, 不 alter exponent.

3. **Superposition 让 mechanistic interpretability 更难** — 你想想, 一个 hidden dim 同时 encode 一堆 features, 要 disentangle 就难. 所以 promote superposition 是 performance-safety trade-off.

## 我觉得 future work 最值得做的

1. **在真实 LLM 上 enforce ETF structure**, 直接看 loss 是否符合 ~1/m prediction. 这是 causal test, 不是 correlational.

2. **独立 measure representation loss vs parsing loss**. 训一个只有 embedding+LM head 没有 transformer layers 的 "degenerate LLM", 它的 scaling 是不是纯粹 representation-limited?

3. **研究不同 tokenizer 下 vocabulary-features 关系**. BPE vs Unigram vs WordPiece, vocabulary size 从 10k 到 200k, scaling 终点怎么变? 这能验证 "m 接近 n 时 scaling 停止" prediction.

4. **Solve toy model strictly**. Large α regime 的 $\alpha_m$ vs α 关系现在只有 conjecture, 应该能用 statistical mechanics 方法 (类似 compressed sensing 的 replica trick) 解出来.

5. **Depth scaling**: 论文只讲 width. Depth scaling 的 mechanism 是什么? Parsing loss $f_\ell(\ell)$ 怎么 scale? 这是论文 explicitly 留的 open question.

---

整体看, 这篇 paper 的贡献是把 Anthropic 的 superposition toy model recontextualize 成 scaling law 的 mechanistic explanation. 之前 scaling law 理论大多停留在 phenomenological level (假设 power-law data → power-law loss), 这篇往下挖了一层, 说 "power-law loss 的 origin 在 representation geometry, data 只是 modulation factor". 作为 Karpathy, 我觉得这个方向对了 — mechanistic understanding of empirical scaling laws 是我们这个领域最缺的. 但距离真正 quantitative prediction (比如不同 architecture 的 α 差异, depth contribution, data scaling) 还有距离. 这是个 good start, 不是 end.

---

# Superposition Yields Robust Neural Scaling — 深入解读

## 1. 论文核心 Thesis

这篇 paper 的核心 claim 是: **neural scaling laws (loss ~ N^{-α_N}) 的根源在 representation superposition 的几何性质**, 而非 data distribution 本身. 在 strong superposition regime 下, 由于 representation vectors 在 m-dim hidden space 中的 geometric overlap 以 ~1/m scale, loss 自然地以 ~1/m 下降, 几乎独立于 feature frequency 的具体形状. 这给出了一个 robust 的 scaling exponent α_m ≈ 1.

作者要解决的核心 Question:
> 当 superposition degree 和 data structure 都变化时, loss 何时是 power law? 如果是, exponent 由什么决定?

## 2. Toy Model 架构 (Section 2)

基于 Anthropic [27] 的 autoencoder, 但在 data sampling 上做了修改. Architecture 极简:

```
Input x ∈ R^n  →  h = W^T x ∈ R^m  →  y = ReLU(W h + b) ∈ R^n
                                                                  ↑
                                   Loss L = ⟨‖y - x‖_2^2⟩_x
```

其中 W ∈ R^{n×m}, m ≪ n. 关键参数:
- **n**: data dimension = atomic features 数量
- **m**: model dimension (hidden space width)
- **W_i**: W 的第 i 行, 是 feature i 在 hidden space 中的 representation vector

### Data generation (Eq. 1)

每个 sample x_i 由两部分相乘:

$$x_i = u_i v_i, \quad u_i \sim \text{Bernoulli}(p_i), \quad v_i \sim U(0, 2)$$

- $u_i$: 是否激活 feature i, 由 probability $p_i$ 决定 (frequency)
- $v_i$: 激活强度, uniform on (0, 2)
- $p_i$ 随 rank i 递减, 即 $p_1 \geq p_2 \geq \dots \geq p_n$

**Activation density**: $E = \sum_{i=1}^n p_i$ (期望激活 feature 数量). Sparsity 意味着 $E/n \ll 1$.

### Feature frequency 分布

主要 scan 的 case: $p_i \propto 1/i^\alpha$, α 是 **data exponent**. 越大 α, distribution 越 skewed.

## 3. 控制 Superposition 的 degree (Eq. 2)

最 trick 的部分: 用 decoupled weight decay 来 tune superposition 强度. 对 W 的每一行单独做:

$$W_{i,t+1} = \begin{cases} W_{i,t} - \eta_t \gamma W_{i,t}, & \gamma \geq 0 \text{ (standard weight decay, shrink toward 0)} \\ W_{i,t} - \eta_t \gamma W_{i,t}(1/\|W_{i,t}\|_2 - 1), & \gamma < 0 \text{ (grow toward unit norm)} \end{cases}$$

- $\gamma > 0$: 标准 weight decay, 强烈的 penalty 会把 unimportant features 的 norm 推到 0 → **weak/no superposition**
- $\gamma < 0$: 对应 minimize $(\|W_i\| - 1)^2$, 鼓励 unit norm rows → **strong superposition**

### 度量 superposition degree (Eq. 3)

定义 **fraction of represented features**:

$$\phi_{1/2} = |\{i : \|W_i\|_2 > 1/2\}| / n$$

经验观察: $\|W_i\|_2$ 分布是 **bimodal**, 集中在 0 和 1 附近 (Figure 3a). 因此 0.5 是自然分界:
- Weak superposition: $\phi_{1/2} \approx m/n$ (只有 m 个 features 被表示)
- Strong superposition: $\phi_{1/2} \approx 1 \gg m/n$ (几乎所有 features 都被表示, 但有 overlap)

## 4. Result 1: Weak Superposition — "Power Law In, Power Law Out"

### 推导 (Eq. 4)

假设理想化: 前 $\phi_{1/2} n$ 个最 frequent features 被 perfectly represented (无 overlap), 后面被忽略. Optimal bias 为 $b_i = 0$ for $i \leq \phi_{1/2} n$, $b_i = \langle x_i \rangle$ for $i > \phi_{1/2} n$.

Loss 来自未被表示 features:

$$L = \sum_{i > \phi_{1/2} n} \langle(x_i - \langle x_i \rangle)^2\rangle = \sum_{i > \phi_{1/2} n} \left(\langle v^2 \rangle p_i - \langle v \rangle^2 p_i^2\right) \approx \langle v^2 \rangle \sum_{i > \phi_{1/2} n} p_i$$

最后近似成立是因为 $p_i \ll 1$ 时 $p_i^2$ 可忽略. 由于 $v \sim U(0, 2)$, 有 $\langle v^2 \rangle = \int_0^2 v^2 \cdot \frac{1}{2} dv = 4/3$.

用积分近似:
$$\int_m^n p_i \, di \propto m^{-(\alpha - 1)} \quad \text{when } n \gg m, \alpha > 1$$

所以:
$$L \propto m^{-(\alpha - 1)}, \quad \alpha_m = \alpha - 1$$

### 结论 (Result 1)

| Condition | Loss scaling |
|---|---|
| $p_i \propto 1/i^\alpha$ (power-law, α>1) | $L \propto 1/m^{\alpha-1}$ |
| $p_i \propto e^{-i/c}$ (exponential) | $L$ 不再是 power-law |
| $p_i \propto (n - i)$ (linear) | $L$ 不再是 power-law |

**Insight**: Weak superposition 下, scaling exponent 完全由 data 的尾部 sum 决定. **必须 input 是 power-law, output 才是 power-law**. 这与之前的工作 [15-20] 一致, 那些工作隐含假设了 weak superposition regime.

Figure 4b 的实验数据验证了 $\alpha_m \approx \alpha - 1$ (当 weight decay 较大, 接近 no superposition 时).

## 5. Result 2: Strong Superposition — Geometric Origin of 1/m Scaling

这是 paper 的核心创新. Strong superposition 下 loss 的 origin 完全不同.

### Loss 来自 interference

若只有 feature j 激活, 输出激活 $y_j \approx W_i \cdot W_j$ (其它 features 上的 leakage). Loss 来自 squared overlaps:

$$L \sim \sum_{j} p_j \sum_{i \neq j} (W_i \cdot W_j)^2$$

### 随机向量几何 ansatz

最简单的 ansatz: $W_i$ 各向同性均匀分布在 unit sphere 上. 在 $\mathbb{R}^m$ 中, 两个随机单位向量的 squared dot product 服从:

$$(W_i \cdot W_j)^2 \sim \text{Beta}\left(\frac{1}{2}, \frac{m-1}{2}\right)$$

- **Mean**: $\mathbb{E}[(W_i \cdot W_j)^2] = 1/m$
- **Variance**: $\frac{2(m-1)}{m^2(m+2)} \sim 2/m^2$

所以 squared overlap 典型 size 为 $1/m$. 这给出 $L \sim 1/m$ 的 scaling.

### ETF (Equiangular Tight Frame) — 更精细的 ansatz

实际训练的 $W_i$ 不完全随机, 而是 bimodal around 1 (Figure 5a). Important features 倾向有更大 norm (Figure 5b). 作者发现 $W_i$ 倾向于 form **Equiangular Tight Frame (ETF)**.

#### Welch bound (Eq. 5)

对 $\nu$ 个 unit vectors $w_i \in \mathbb{R}^m$ ($\nu \geq m$):

$$\max_{i \neq j} |w_i \cdot w_j| \geq \sqrt{\frac{\nu - m}{m(\nu - 1)}} \equiv \kappa$$

当 $\nu \gg m$, $\kappa \approx \sqrt{1/m}$, 即 $\kappa^2 \approx 1/m$.

#### ETF 性质

- ETF 达到 Welch bound equality, 即所有 pairwise absolute overlap = $\kappa$
- Variance 为 0 (随机向量 variance $\sim 2/m^2$)
- Real space 中 ETF 要求 $\nu \leq m(m+1)/2$

实验观察 (Figure 5c, d):
- Norm > 1 的 vectors (strongly represented) 的 squared overlap variance 显著小于随机情形
- Mean squared overlap 收敛到 $\kappa^2 \approx 1/m$
- Strongly represented vectors 的数量 $\approx m^2/2$ (Figure 16), 接近 ETF 上限

### Strongly vs weakly represented features

将 features 分两类:
- **Strongly represented**: $\|W_i\|_2 > 1$, ~ $m^2/2$ 个, ETF-like
- **Weakly represented**: $\|W_i\|_2 < 1$, norm 较小

$\phi_1 = |\{i : \|W_i\|_2 > 1\}| / n \approx m^2/(2n)$ (Eq. 9).

### 两种 sub-regime (Figure 5e)

**Even frequencies (small α)**: 所有 vectors 大致各向同性, mean squared overlap = 1/m → $\alpha_m \approx 1$.

**Skewed frequencies (large α)**: Important features 占更大 angular space, 有更小 overlap. 极端 conjecture: 前 $m^2/2$ 个 features ETF-like, 贡献可忽略, 其余 weakly represented. 后者 loss:

$$\sum_{i = m^2/2}^{n} p_i \sim (m^2)^{-({\alpha - 1})} \sim m^{-2(\alpha-1)}$$

→ $\alpha_m \approx 2(\alpha - 1)$, 与实验观察接近 (Figure 5e).

### 结论 (Result 2)

| Regime | $\alpha_m$ | Origin |
|---|---|---|
| Strong superposition, small α (flat freq) | ≈ 1 | Isotropic geometry, squared overlap ~ 1/m |
| Strong superposition, large α (skewed) | ≈ 2(α−1), 上界 | Heterogeneous, important features 占更多 angular space |
| Weak superposition, power-law freq | α − 1 | Tail sum of ignored features |

**Robust scaling in strong superposition**: 即使 data distribution 不是 power-law (exponential 或 linear), $\alpha_m \approx 1$ 仍然保持. 这是 "Superposition Yields Robust Neural Scaling" 的核心含义.

## 6. Result 3: LLMs 验证

### LLMs 在 strong superposition regime

证据:
1. Vocabulary size $n \approx 50k$, model dimension $m$ 从几百到几千 → $n \gg m$
2. All tokens 都有 non-zero representation (Appendix D.7, Figure 20)
3. Token frequency 接近 power law with α ≈ 1 (Figure 22) — 即 "flat" end, 落在 robust regime

### Mean squared overlaps of LM head rows

把 LM head 的 weight matrix W 当作 representation vectors. 计算:

$$\text{overlap}(W_i, W_j) = \left|\frac{W_i \cdot W_j}{\|W_i\|_2 \|W_j\|_2}\right|$$

Figure 6a 显示 mean squared overlaps 大致服从 $1/m$ scaling, 跨越 OPT, GPT-2, Qwen2.5, Pythia 四个 model families.

### Cross-entropy loss → squared overlap 的近似 (Appendix A.2, Eq. 10-13)

假设 hidden state 对应 $W_i / \|W_i\|_2$, target 是 token i. Cross-entropy loss:

$$L = -\ln \frac{e^{\|W_i\|_2}}{\sum_j e^{W_i \cdot W_j / \|W_i\|_2}} = \ln\left[1 + \sum_{j \neq i} e^{W_i \cdot W_j/\|W_i\|_2 - \|W_i\|_2}\right]$$

由于 overlap $\sim 1/m \ll 1$, Taylor 展开 (Eq. 11):

$$L \approx (n-1)e^{-\|W_i\|_2} + \frac{\epsilon_{D,i} e^{-\|W_i\|_2}}{\|W_i\|_2} + \frac{1}{2} \sum_{j \neq i} \left(\frac{W_i \cdot W_j}{\|W_i\|_2}\right)^2 e^{-\|W_i\|_2}$$

- 第一项: 与 m 无关的 baseline (intrinsic uncertainty)
- 第二项: 与 data 相关的 correction, $\epsilon_{D,i}$ 是 data-imposed correlation
- **第三项**: $\frac{1}{2} \sum_j (\text{cosine sim})^2 e^{-\|W_i\|}$ → 与 m 相关的部分, 因为 cosine sim² ~ 1/m

### Fitting LLM loss (Eq. 6)

$$L = C_m / m^{\alpha_m} + L_{\backslash m}$$

- $C_m / m^{\alpha_m}$: 与 model size 相关, universal across datasets
- $L_{\backslash m}$: dataset/model class specific offset (intrinsic data uncertainty)

Fitting 结果: $\alpha_m = 0.91 \pm 0.04$ (Figure 6b), 接近 1.

### Chinchilla 一致性

Chinchilla scaling law: $L \propto N^{-\alpha_N}$ with $\alpha_N = 0.35 \pm 0.02$.

实测 $N \propto m^{2.52 \pm 0.03}$ (Appendix D.7, Figure 23), 因此:

$$\alpha_m = 2.52 \cdot \alpha_N = 2.52 \times 0.35 = 0.88 \pm 0.06$$

接近 1, 与 toy model 在 flat frequency (α≈1) 下的 prediction 一致.

## 7. 跨超参数相图 (Figure 9b)

将 $\alpha_m$ 作为 (γ, α) 的函数, 显示三种行为:
1. **Weak superposition (γ > 0)**: $\alpha_m$ 小, 慢速 decay
2. **Strong superposition, small α**: $\alpha_m \approx 1$ (red box, robust)
3. **Strong superposition, large α**: $\alpha_m$ 随 α 增长 (blue box, 上界 ~2(α−1))

R² values (Figure 9a, 10): strong superposition 下, loss 在 log-log 上接近直线, R² ≈ 1, 跨越 exponential / power-law / linear 三种 frequency 分布. 这是 robustness 的直接证据.

## 8. 为什么 LLMs 会落到 strong superposition?

作者给出两个 conjecture:
1. **Sparsity of language**: 预测一个 token 需要的 tokens 数远小于 vocabulary size, error correction 容易实现
2. **Softmax**: 强大的 error correction 能力, 给 superposition 优势

## 9. 对 LLM 开发的启示 (Section 5)

### 能否加速 scaling exponent?

> 不能 (对 natural language), 因为 α ≈ 1 已经是 robust 几何下界. 可能对 super-skewed domain tasks 有效 (回到 weak superposition-like 行为).

### 何时 scaling law 停止?

> 当 m 接近 vocabulary size n (≈50k), superposition 不再必要, scaling law 偏离 power law 并消失. 但如果 "true number of independent things in language" 大于 vocabulary, scaling 可继续.

### 鼓励 superposition 训练策略

- **nGPT** [60]: 约束 hidden states 和 weight rows 在 unit sphere 上, 促进 superposition, 实测性能提升
- **FOCUS** [61]: 无 weight decay 的稳定 optimizer, 可能因 enhanced superposition 而有效

### Trade-off

Superposition 让 mechanistic interpretability 和 AI safety 更难 [27, 62].

## 10. Limitations & Open Questions

1. **Toy model 不严格 solve**: strong superposition 下 $W_i$ 的真实 configuration 比简单 ETF ansatz 复杂, large α regime 的 $\alpha_m$ vs α 严格关系未解
2. **Parsing loss vs representation loss**: 论文只关注 representation. 真实 LLM loss 应分解为 (Eq. 7):
   $$C_m/m^{\alpha_m} = f_m(m) + f_\ell(\ell)$$
   其中 $\ell$ 是 depth, $f_\ell$ 是 parsing-limited loss. 需要独立研究 $f_\ell(\ell)$.
3. **Depth-width relationship**: LLMs 中 $N \propto m^2 \ell$. Optimal $m$-$\ell$ 关系使两者 balanced, 可能解释为何 $\alpha_m \approx 1$ 整体观测到
4. **Dataset size / training steps scaling**: 未涉及. 在 strong superposition 下, 与 representation vector angle 的演化相关, 难以严格解释
5. **Emergent abilities**: 同样 pre-training loss 下, 不同 superposition degree 的 model 可能在 reasoning / RL trainability 上有差异 [63]

## 11. 与 Related Works 的比较

| Work | Regime | Scaling |
|---|---|---|
| Sharma & Kaplan [14], Bahri et al. [15] | Variance-limited (data infinite) | $1/m$ via CLT |
| Bordelon et al. [16, 17], Maloney et al. [18] | Resolution-limited, kernel regime | $\alpha_m = \alpha' - 1$ |
| Michaud et al. (Quantization) [20] | Discrete skills, power-law importance | $\alpha - 1$ type |
| Song et al. [25] | Resource model, overparameterized | $1/m$ via CLT, less relevant to LLMs |
| **This paper** | **Strong superposition, LLM regime** | **$1/m$ via geometry, robust across data distributions** |

关键区分: 本文将 superposition 视为 mechanistic cause, 在 flat-frequency regime 下给出 robust $1/m$, 而 prior work 需要假设 power-law data 才能得到 power-law loss.

## 12. 我的 Intuition Takeaway

1. **Scaling law 的 robustness 来自 geometry, 不来自 data**: 在 strong superposition 下, loss 的 power-law exponent ~1 是 high-dim 空间中向量几何的必然 — 随着维度增加, 任意两个向量的 dot product² 期望是 1/m, 不论 vector 数量. 这是 high-dimensional probability 的基础结果.

2. **Weak superposition regime 是 "data-limited"**: 模型把资源 (m 个 orthogonal directions) 全部分配给 top-m features, 剩余 loss 来自 unlearned tail. Tail sum 的形状决定 scaling, 所以 data shape 完全 control scaling.

3. **Strong superposition regime 是 "geometry-limited"**: 模型选择 represent 所有 features, 但每个都 imperfect. Error 来自 pairwise interference, interference 的 size 由维度几何决定, 不由 data 决定.

4. **LLMs 是 strong superposition**: 因为 (a) features (tokens) 远多于 dimensions, (b) language 是 sparse 的 (error correction 可行), (c) token frequency α ≈ 1 是 flat end of distribution → 落在 robust regime. 这是 neural scaling laws 在 LLM 上有 universal exponent 的原因.

5. **Chinchilla α_N ≈ 0.35 ↔ α_m ≈ 1**: 通过 $N \propto m^{2.5}$ bridge, 这是 paper 最 elegant 的 quantitative consistency check.

6. **Optimal m-ℓ balance 假设**: 论文假设在 Chinchilla-optimal 点, representation loss 和 parsing loss 大致 balanced, 都 ~ 1/m, 从而总 $\alpha_m \approx 1$. 这是个 conjecture, 未直接验证.

7. **Architecture implication**: nGPT (在 unit sphere 上做 representation) 和无 weight decay 的 optimizer 可能本质上 promote superposition, 进而 improve scaling. 但 paper 提到这些主要 alter coefficient 而非 exponent.

## 13. 关键 Reference

- Anthropic toy model (superposition 的奠基): https://transformer-circuits.pub/2022/toy_model/index.html
- Chinchilla scaling: https://arxiv.org/abs/2203.15556
- Kaplan et al. scaling laws: https://arxiv.org/abs/2001.08361
- Bahri et al. "Explaining neural scaling laws": https://www.pnas.org/doi/10.1073/pnas.2311878121
- Michaud et al. Quantization model: https://arxiv.org/abs/2310.10622 (NeurIPS 2023)
- Welch bound: Welch, IEEE Trans. Info. Theory, 1974
- nGPT: https://arxiv.org/abs/2410.01131
- FOCUS optimizer: https://arxiv.org/abs/2501.12243

## 14. 不足之处

1. **No rigorous solution of toy model**: ETF ansatz 只在 α small 时严格成立, α large regime 仅有 conjecture. Figure 18 表明 m 从 50-150 时 $m^2/2 > n$ 已经让所有 features strongly representable, 但 $\alpha_m$ 仍随 α 增长, 说明 simple strongly/weakly dichotomy 不够.
2. **LLM 验证 correlational, not causal**: 只展示 overlap ~ 1/m 和 loss ~ 1/m 一致, 没做 interventional 实验 (如人为 enforce ETF structure 看 loss 是否符合 prediction).
3. **Token-as-feature 假设 naive**: 真实 LLM 中 "atomic features" 远多于 vocabulary tokens (Anthropic 的 superposition work 已显示), 这可能让 vocabulary size 不是 scaling 终点的正确估计.
4. **Parsing loss $f_\ell(\ell)$ 未独立 measure**: 论文假设 balanced, 但没有实验直接分离 representation vs parsing contribution.
5. **Activation density 测试局限**: 仅在 toy model 上验证 E 不影响 exponent (Figure 14, 15), LLM 中真实 sparsity-strength 关系未检验.

## 15. 直觉总结公式集

最关键的几个等式连起来:

$$\text{Squared overlap (isotropic)} \sim \text{Beta}(1/2, (m-1)/2) \Rightarrow \mathbb{E}[\cdot] = 1/m$$

$$\text{Welch bound} \Rightarrow \kappa^2 = \frac{\nu - m}{m(\nu - 1)} \approx 1/m \text{ when } \nu \gg m$$

$$L_{\text{strong}} \approx \langle v^2 \rangle \cdot \frac{1}{m} \cdot E \Rightarrow \alpha_m \approx 1$$

$$L_{\text{weak}} \approx \frac{4}{3} \int_m^n \frac{1}{i^\alpha} di \sim m^{-(\alpha-1)} \Rightarrow \alpha_m = \alpha - 1$$

$$L_{\text{LLM}} = \frac{C_m}{m^{\alpha_m}} + L_{\backslash m}, \quad \alpha_m \approx 0.91 \approx 2.52 \times 0.35$$

整体 picture: **Superposition 把 scaling 的来源从 data distribution 转移到 representation geometry, 后者在 high-dim 下天然给出 ~1/m, 不需要 power-law data 假设. 这就是 robustness 的来源.**

---

这篇 paper 把 Anthropic 的 superposition toy model 重新用作 scaling laws 的 mechanistic 解释, 是 mechanistic interpretability 与 scaling theory 的桥梁. 作为 Karpathy, 我会说这种 "从 mechanism 出发理解 empirical scaling" 的方向非常对路, 但距离真正 quantitative prediction LLM scaling (如不同 architecture family 的 α 差异) 还需要解决 toy model 的严格解和 parsing loss 的分解. 值得 follow-up 的方向包括: 在真实 LLM 上做 interventional ETF-enforcing 实验, 直接 measure representation loss vs parsing loss, 以及研究不同 tokenizer 下 vocabulary-features 关系对 scaling 终点的影响.
