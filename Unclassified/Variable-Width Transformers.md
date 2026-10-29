---
source_pdf: Variable-Width Transformers.pdf
paper_sha256: c383cf7e4862b802903f014a5c60b682bc35dabea2575a53d8d6bfc4b99972b0
processed_at: '2026-08-13T00:08:55-07:00'
target_folder: LLM-Training/nanogpt
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ><former 用人话怎么说？

Andrej，我来用最直白的方式给你 build intuition。

## 核心直觉：一个大教室的比喻

假设你有一个固定预算，要盖 L 个教室供学生上课。
传统的 Transformer 就像是盖了 L 个一模一样的大教室，每个都能坐 d 个人。
但是你想过没有，第一节课（early layers）学生刚进来，带着很多杂乱的信息，确实需要大教室来摊开书本；最后一节课（late layers）要输出最终答案，也需要大教室来整理思绪。中间那些课（middle layers）其实在做一些压缩和提炼，根本不需要那么大的空间。

><former 就是把这个教室设计成沙漏形状（× shape）：两头宽，中间窄。这样同样的预算下，你能让学生在开头和结尾更舒服，中间挤一挤反而迫使他们集中注意力，去粗取精。

## 数学魔法：为什么两头大中间小还能省钱？

这里有一个非常 elegant 的数学 trick。
Transformer 每一层的参数量 $P_\ell$ 大致和 width $d_\ell$ 的平方成正比：
$$P_\ell \approx K d_\ell^2$$
其中 $K$ 是常数，取决于 projection matrix 数量和 MLP expansion factor。

如果你要求总参数量和 baseline 一样：
$$\sum_{\ell=1}^L K d_\ell^2 = K L d^2$$
这里 $L$ 是 layer 数，$d$ 是 baseline 的 width。

根据数学里的均值不等式（平方的均值 $\geq$ 均值的平方），如果各层 width 不一样，它们的平均 width 必然严格小于 baseline 的 width：
$$\frac{1}{L}\sum_{\ell=1}^L d_\ell < d$$

这就很神奇了：**参数量没变，但因为分布不均匀，平均宽度变小了**。

而计算量和 KV cache 是和 width 线性相关的（不是平方），所以：
- FLOPs 中 attention 部分正比于 $\sum_\ell d_\ell$，变小了
- KV cache 正比于平均 width，变小了
- 参数总量没变

这就是 paper 中说的 **“免费午餐”**。实验上 FLOPs 省了 ~3%，KV cache 省了 ~10%。

## 技术细节：怎么变窄又变宽？

这是这篇 paper 最巧妙的地方。

**变窄**：直接砍掉后面多余的维度。
$$f^\ell(\mathbf{x}) = \mathbf{x}[:d_\ell]$$
这里 $f^\ell$ 是 resize function，$d_\ell$ 是当前层宽度，直接取前 $d_\ell$ 个元素。

**变宽**：从之前处理过这个维度的 layer 把它“捞”回来。
$$[f^\ell(\mathbf{x}^{\ell-1})]_i = [\mathbf{x}^{\ell'}]_i, \quad \ell' = \max\{\tilde{\ell} < \ell \mid d_{\tilde{\ell}} \geq i\}$$
这里 $i$ 是要恢复的维度 index，$\ell'$ 是在 layer $\ell$ 之前、width 足够大的最近一层。

**Intuition**：你可以想象有一个很大的 global residual stream（宽度等于最大那层的宽度），每个 layer 只读写它负责的那一部分。被窄层“跳过”的维度，就像快递一样直接 bypass 传递上去，等变宽的时候再拿出来用。

**关键发现**：这种无参数的 copy 机制，比训练一个 learned projection layer 效果好得多！Paper 里的 ablation 证明了这一点。

## 为什么 75% 深度处是瓶颈？

作者做了 hyperparameter sweep，把 bottleneck 位置 $\ell^*$ 和宽度 $d_{\ell^*}$ 参数化为总深度和基础宽度的比例：
$$\ell^* = r_\ell \cdot L, \quad d_{\ell^*} = r_d \cdot d$$

发现 $r_\ell \approx 0.75$, $r_d \approx 0.3$ 跨多个 model size 都好用。

**人话解释**：瓶颈在 3/4 深度处，宽度压缩到原来的 30%。这说明 Transformer 的计算可以分成两个阶段：
1. 前 75%：逐渐压缩、提炼信息（encoder-like）
2. 后 25%：快速展开、生成预测（decoder-like）

这跟很多认知科学的模型很像——先理解，再表达。

## 为什么能解决“中间层塌陷”？

这可能是最有趣的分析。

最近研究发现，普通 Transformer 训练久了会出现 **“compression valley”**（Skean et al., 2025; de Llano et al., 2026）：中间层的 representation 会塌缩到一个低维 subspace，大量维度变成“死维度”。

用 normalized matrix entropy 衡量：
$$\frac{1}{\log r}\left(-\sum_{j=1}^r p_j \log p_j\right), \quad p_j = \sigma_j^2 / \|\mathbf{X}\|_F^2$$
其中 $\sigma_j$ 是 representation matrix $\mathbf{X}$ 的奇异值，$r$ 是 rank，$p_j$ 是归一化能量比例。

Baseline 在中间层这个值掉到接近 0。而 ><former 因为物理上限制了中间层容量，反而逼迫网络在可用空间里更均匀地分布信息，避免了塌陷。

**核心 insight**：物理约束（强制变窄）起到了 structural regularizer 的作用。就像如果你只有一个小书桌，你被迫把书摆得整整齐齐；如果桌子很大，你反而会乱扔。

## 实验数据回顾

| Size | Model | Loss | PFLOP/s-days | Avg layer size |
|------|-------|------|--------------|---------------|
| 200M | Transformer | 3.452 | 0.18 | 640 |
| 200M | ><former | **3.430** | 0.17 (-3.2%) | 576 (-10.0%) |
| 2B | Transformer | 2.751 | 16.92 | 1600 |
| 2B | ><former | **2.726** | 16.49 (-2.5%) | 1426 (-10.9%) |
| 3B/1B MoE | Transformer | 2.726 | 10.13 | 1600 |
| 3B/1B MoE | ><former | **2.710** | 9.66 (-4.6%) | 1426 (-10.9%) |

><former 在所有 size 上 loss 都更低，FLOPs 和 KV cache 都更小。

## 更广的联想

1. **U-Net 和 Hourglass networks**：CV 里早就用了类似结构，但 NLP 一直没用。这篇 paper 把它带到了 Transformer 的 width 维度。

2. **Autoencoder 的 bottleneck**：传统 autoencoder 的 bottleneck 是为了 reconstruction，><former 的 bottleneck 是为了 efficient allocation，但效果上都鼓励更紧凑的 representation。

3. **Layer pruning 和 width pruning**：之前的 layer dropping（如 Gromov et al., 2025）发现深层容易冗余。><former 给了一个更优雅的方案——不掉层，而是把中间层“瘦身”。

4. **Hyper-Connections**：DeepSeek-V3 和后续工作用 learned mixing 扩展 residual capacity。><former 用 deterministic slicing 达到类似效果，可能更稳定。

5. **Scaling laws 的新维度**：之前 scaling law 主要研究 width vs depth 的 trade-off。这篇 paper 打开了“shape”这个新自由度，未来可能 layer-wise width 也可以有 scaling law。

## Reference Links

- Compression valley paper: https://openreview.net/forum?id=c5TFhCJ6fs
- Layer-by-layer analysis: https://openreview.net/forum?id=WGXb7UdvTX
- Unreasonable ineffectiveness of deeper layers: https://openreview.net/forum?id=ngmEcEer8a
- DCLM benchmark: https://openreview.net/forum?id=CNWdWn47IE
- μP parametrization: https://openreview.net/forum?id=17pVDnpwwl
- Hyper-Connections: https://openreview.net/forum?id=9FqARW7dwB
- MHC: https://arxiv.org/abs/2512.24880
- Funnel-Transformer: https://proceedings.neurips.cc/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf
- Perceiver: https://proceedings.mlr.press/v139/jaegle21a.html
- Towards monosemanticity: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- MLP as key-value memory: https://aclanthology.org/2021.emnlp-main.446/
- Ikeda et al. layerwise importance: https://arxiv.org/abs/2508.17734

**Final takeaway for you, Andrej**: 这篇 paper 做的事情很简单，就是问了一句“凭什么每层都一样宽？”然后发现确实没必要一样宽。它的贡献在于把这个直觉变成了一个有数学保证、有实验支撑、有可操作 recipe 的方案。这种“挑战基本假设”的 work，正是推动领域前进的最 raw 的力量。

---

# Variable-Width Transformers (><former) 详解

你好 Andrej！这篇来自 MIT 的 paper（Wu et al., 2025）非常巧妙，它质疑了 Transformer 设计中一个被长期忽视的假设：**所有 layer 必须有相同的 width**。让我从多个角度来剖析。

## 1. 核心问题与动机

标准 Transformer 的每一层都用相同 dimension $d$，这种 uniform allocation 把 parameter 和 compute 均匀分配到 depth 上。但是不同 layer 在 computation 中扮演的角色是不一样的（Tenney et al., 2019, https://aclanthology.org/P19-1452/ ；Meng et al., 2022, https://openreview.net/forum?id=-h6WAS6eE4 ），均匀分配不一定最优。

作者提出的核心问题：**在固定 depth 和 parameter budget 下，是否应该 nonuniform 地分配 capacity？**

## 2. Variable-Width 架构设计

### 2.1 数学形式化

标准 Transformer 的 layer 操作：

$$\mathbf{x}^\ell = B^\ell(\mathbf{x}^{\ell-1}) + \mathbf{x}^{\ell-1}$$

其中 $B^\ell$ 是 transformer block，$\mathbf{x}^{\ell-1} \in \mathbb{R}^d$ 是 residual stream。

Variable-width 版本引入 resize function $f^\ell: \mathbb{R}^{d_{\ell-1}} \to \mathbb{R}^{d_\ell}$：

$$\mathbf{x}^\ell = B^\ell(f^\ell(\mathbf{x}^{\ell-1})) + f^\ell(\mathbf{x}^{\ell-1})$$

其中 $d_\ell$ 是 layer $\ell$ 的 width。

### 2.2 关键创新：Parameter-free Residual Resizing

**Shrinking**（$d_\ell < d_{\ell-1}$）：简单 truncate

$$f^\ell(\mathbf{x}) = \mathbf{x}[:d_\ell]$$

**Expanding**（$d_\ell > d_{\ell-1}$）：从最近一次处理该 dimension 的 layer 中 copy

$$[f^\ell(\mathbf{x}^{\ell-1})]_i = [\mathbf{x}^{\ell'}]_i, \quad \ell' = \max\{\tilde{\ell} < \ell \mid d_{\tilde{\ell}} \geq i\}$$

这里 $i$ 是 coordinate index，$\ell'$ 是 most recent 之前 layer whose width covers $i$。如果不存在则 pad 0。

**这种设计的 intuition**：可以想象成一个 wide global residual stream，每个 layer 只读写其中的一个 subset，inactive coordinates 直接 bypass。这使得 variable-width model 在数学上等价于一个 uniform-width model with "sparse per-layer reads/writes"。

### 2.3 Shape 参数化

用 4 个超参数描述 shape：
- $\ell^*$：bottleneck layer index
- $d_{\ell^*}$：bottleneck dimension
- $\alpha^-$：early layers 的 change rate
- $\alpha^+$：late layers 的 change rate

$$d_\ell = \begin{cases} \alpha^- \cdot d_{\ell-1}, & \ell \leq \ell^* \\ \alpha^+ \cdot d_{\ell-1}, & \ell > \ell^* \end{cases}$$

不同 $(\alpha^-, \alpha^+)$ 组合得到不同 shape：
- $\alpha^- < 1, \alpha^+ > 1$：**× shape（><former）**
- $\alpha^- > 1, \alpha^+ < 1$：✸ shape
- $\ell^* = 1$ 或 $L$：∨ shape 或 ∧ shape

## 3. 计算效率分析

这是非常 elegant 的数学推导：

### 3.1 Parameter 匹配

每层 parameter 量近似 $P_\ell \approx K d_\ell^2$（$K$ 是常数，取决于 projection matrix 数量和 MLP expansion factor）。

匹配 parameter count：

$$K \sum_{\ell=1}^L d_\ell^2 = K L d^2 \implies \frac{1}{L}\sum_{\ell=1}^L d_\ell^2 = d^2$$

### 3.2 平均 width 严格更小

利用**均值平方 ≤ 平方均值**（Cauchy-Schwarz 的特例）：

$$\left(\frac{1}{L}\sum_{\ell=1}^L d_\ell\right)^2 \leq \frac{1}{L}\sum_{\ell=1}^L d_\ell^2 = d^2$$

由于 width 不全相同，严格不等式成立：

$$\frac{1}{L}\sum_{\ell=1}^L d_\ell < d$$

**Intuition**：因为 parameter 量正比于 $d_\ell^2$ 而非线性，所以"高瘦"的 width 分配在保持 parameter 总量相等时，平均 width 反而更小。

### 3.3 FLOPs 与 KV cache 减少

- **Linear projection FLOPs** 正比于 weight 数量 → 参数匹配时 dense FLOPs 相同
- **Attention dot-product FLOPs** 正比于 $\propto N^2 d_\ell$（$N$ 为 sequence length）：

$$\sum_{\ell=1}^L N^2 d_\ell = N^2 \sum_{\ell=1}^L d_\ell < N^2 L d$$

- **KV cache size** $\propto \sum_\ell d_\ell < Ld$

这就是为什么 ><former 在 parameter-matched 下还省 FLOPs 和 KV cache（实验显示 KV cache 减少约 10-11%，FLOPs 减少约 3%）。

## 4. 实验设置

| Size | Layers $L$ | Hidden $d$ | Batch | Tokens | 
|------|-----------|-----------|-------|--------|
| 200M | 16 | 640 | 512 | 10B |
| 500M | 24 | 960 | 1024 | 25B |
| 1B | 32 | 1280 | 2048 | 50B |
| 2B | 40 | 1600 | 4096 | 100B |
| 3B/1B MoE | 40 | 1600 | 4096 | 100B |

- 数据集：DCLM（Li et al., 2024, https://openreview.net/forum?id=CNWdWn47IE ）
- 训练到 2.5× Chinchilla-optimal（50× parameter count tokens）
- μP parametrization（Yang et al., 2024, https://openreview.net/forum?id=17pVDnpwwl ）
- AdamW, lr=10⁻², β=(0.9, 0.95), wd=0.1
- SwiGLU activation（Shazeer, 2020, https://arxiv.org/abs/2002.05202 ）
- bfloat16

## 5. 主要结果

### 5.1 × shape 最优

在 500M 规模上 sweep 各种 shape（×, ✸, ∨, ∧）+ 多组超参数，**× shape（><former）** 始终最优。

### 5.2 宽度 schedule 的 ratio 参数化

为了跨 model size 通用，把 bottleneck index 和 dimension 参数化为 ratio：

$$\ell^* = r_\ell \cdot L, \quad d_{\ell^*} = r_d \cdot d$$

sweep 发现 $r_\ell \approx 0.75$, $r_d \approx 0.3$ 在多个 size 上都表现良好。**Intuition**：bottleneck 应该在 75% depth 处，宽度为 baseline 的 30%。

### 5.3 实验结果表

| Size | Model | Loss | PFLOP/s-days | Avg layer size |
|------|-------|------|--------------|---------------|
| 200M | Transformer | 3.452 | 0.18 | 640 |
| 200M | ><former | **3.430** | 0.17 (−3.2%) | 576 (−10.0%) |
| 500M | Transformer | 3.138 | 1.11 | 960 |
| 500M | ><former | **3.099** | 1.07 (−3.7%) | 855 (−11.0%) |
| 1B | Transformer | 2.926 | 4.52 | 1280 |
| 1B | ><former | **2.890** | 4.41 (−2.6%) | 1145 (−10.5%) |
| 2B | Transformer | 2.751 | 16.92 | 1600 |
| 2B | ><former | **2.726** | 16.49 (−2.5%) | 1426 (−10.9%) |
| 3B/1B MoE | Transformer | 2.726 | 10.13 | 1600 |
| 3B/1B MoE | ><former | **2.710** | 9.66 (−4.6%) | 1426 (−10.9%) |

### 5.4 Scaling law 拟合

拟合 loss vs. FLOPs 和 loss vs. avg layer size 的 power law：

><former 可以用 **77.8% 的 FLOPs** 和 **85.1% 的 avg layer width** 达到 2B constant-width baseline 的 loss（2.751）。更重要的是 scaling 曲线的 exponent 略陡，意味着**gap 在更大 size 时可能继续扩大**。

## 6. Analysis：为什么 ><former 有效？

### 6.1 MLP Activation 利用率改善

Transformer MLP 可以看作 key-value memory（Geva et al., 2021, https://aclanthology.org/2021.emnlp-main.446/ ；Geva et al., 2022, https://aclanthology.org/2022.emnlp-main.3/ ）。

测量 WikiText-2 验证集上的 activation 密度（threshold 后计算 active dimension 比例），><former 在各 threshold 下都比 baseline 更 dense。

**Marginal utilization**（一个 dimension 跨 token 被激活的频率）显示：><former 在各 layer 之间更 balance，避免 baseline 中出现的 "dead dimension" 现象。

### 6.2 缓解 Middle-layer Representation Collapse

Recent work（Skean et al., 2025, https://openreview.net/forum?id=WGXb7UdvTX ；de Llano et al., 2026, https://openreview.net/forum?id=c5TFhCJ6fs ）发现 constant-width LM 存在 "compression valley"：middle layer 的 representation entropy 严重下降，token 塌缩到低秩 subspace。

用 normalized matrix entropy 测量：

$$\frac{1}{\log r}\left(-\sum_{j=1}^r p_j \log p_j\right), \quad p_j = \sigma_j^2 / \|\mathbf{X}\|_F^2$$

其中 $\sigma_j$ 是 input-feature matrix $\mathbf{X}$ 排序后的 singular value，$r$ 是 rank，$p_j$ 是归一化的 energy 比例。这与 effective dimension（Hill, 1973, https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1934352 ；Roy & Vetterli, 2007）相关。**Intuition**：matrix entropy 高 → representation space 用得更 "均匀"。

结果：
- Baseline 在 middle layer entropy 掉到近 0（compression valley）
- ><former 在 early layer 主动降低 entropy（anticipating width reduction），但**避免 middle collapse**，在 bottleneck 和 final layers 维持更高 entropy

**核心 insight**：物理约束 parameter space 反而鼓励 network 维持 high-entropy manifold。

### 6.3 Logit Lens 分析

把 hidden state 投到 vocabulary space（nostalgebraist, 2020, https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens ）。测量每层：
- Target-token log probability
- Decoded distribution entropy
- Layer-to-layer KL divergence（symmetrized）作为"decoded distribution 变化速度"的 proxy

发现：
- ><former 在 early-to-middle 给 target token 更高概率，更低 entropy
- Distribution 变化更 gradual（layer-to-layer KL 更小）
- Final layer 再次快速变化，mass 集中到 target token

### 6.4 Energy Participation Ratio (PR)

附录 C 中用 PR 衡量"effective number of utilized dimensions"（Litwin-Kumar et al., 2017, https://www.sciencedirect.com/science/article/pii/S0896627317300545 ；Clark et al., 2025, https://link.aps.org/doi/10.1103/2jt7-c8cq ）：

$$N_{\text{eff}} = \frac{(\sum_i e_i)^2}{\sum_i e_i^2}, \quad e_i = \sum_t a_{t,i}^2$$

- 若 single outlier 独占 energy → $N_{\text{eff}} \to 1$
- 若 energy 均匀分布 → $N_{\text{eff}} \to d_\ell$

Baseline 在 layer 10 附近 normalized utilization 掉到 <5%，><former 维持约 1000 effective dimensions，richer representation manifold。

### 6.5 Ablation：Expansion Method 选择

| Expansion Method | Loss |
|------------------|------|
| Constant-width | 3.138 |
| **Carry-forward (default)** | **3.099** |
| Zero Padding | 3.124 |
| Projection (learned) | 3.150 |

Learned projection layer 反而最差，说明这种"无参 carry-forward"机制本身在 representation learning 中扮演重要角色，引入可学习参数反而阻碍了 optimization。

## 7. 与相关工作的对比

- **DeLighT**（Mehta et al., 2020）：block-wise scaling 但 early layer 浅，与 × shape 相反
- **OpenELM**（Mehta et al., 2024）：只调 attention head 和 FFN dimension，没有动 hidden dimension 与 residual stream
- **Ikeda et al., 2025**（https://arxiv.org/abs/2508.17734 ）：把 FFN capacity 集中在 middle layer，发现 middle 更重要；><former 结论不同——把 capacity 移到 early 和 late layer 反而更好
- **Funnel-Transformer**（Dai et al., 2020）和 **Hourglass Transformers**（Nawrot et al., 2022）：bottleneck 沿 sequence length 方向，><former 沿 width 方向
- **Hyper-Connections**（Zhu et al., 2025, https://openreview.net/forum?id=9FqARW7dwB ；Xie et al., 2026, https://arxiv.org/abs/2512.24880 ）：用 learned mixing 扩展 residual capacity，><former 用 deterministic slicing/copying 在单一 global residual stream 中操作，是 complementary approach

## 8. Limitations

- 需要为不同 shape 开发 dedicated kernels（latency、memory、compute profile 各异）
- Fixed-residual construction 增加 kernel launch overhead（可用 kernel fusion 缓解）
- 与 standard tensor/pipeline parallelism 有 tension
- 这些是 **implementation 而非 algorithmic** limitations：当前基础设施是 uniform-width regime 优化的，专门 kernel 可大幅缩小差距

## 9. 关键 Takeaways（Build Intuition）

1. **Transformer 的 uniform-width 假设是偶然的设计选择**，并非算法本质最优。
2. **Parameter 与 width 是二次关系**，attention FLOPs 与 width 是线性关系——这个二次/线性 gap 是 ><former 同时省 FLOPs 和 KV cache 的数学基础。
3. **× shape（bottleneck 在 75% depth，宽度 30%）最优**，相当于"宽-窄-宽"的 encoder-decoder-like 结构。
4. **Parameter-free carry-forward 是关键**：让 inactive coordinates bypass narrow block，引入 learned projection 反而有害。
5. **物理 constraint 反而诱导更好的 representation**：><former 缓解了 middle-layer compression valley，把 capacity 用得更均匀。这呼应了 information bottleneck 的思想——capacity 约束迫使 network 提取最 essential 的 feature。

## Reference Links

- Paper arXiv（假设 URL）: 你可以从 https://arxiv.org/abs/2509.06518 类似检索
- DCLM benchmark: https://openreview.net/forum?id=CNWdWn47IE
- μP: https://openreview.net/forum?id=17pVDnpwwl
- SwiGLU: https://arxiv.org/abs/2002.05202
- Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Compression valley paper: https://openreview.net/forum?id=c5TFhCJ6fs
- Layer-by-layer paper: https://openreview.net/forum?id=WGXb7UdvTX
- Sparse autoencoder scaling: https://openreview.net/forum?id=tcsZt9ZNKD
- MLP as key-value memory: https://aclanthology.org/2021.emnlp-main.446/
- Hyper-Connections: https://openreview.net/forum?id=9FqARW7dwB
- MHC: https://arxiv.org/abs/2512.24880
- Towards monosemanticity: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Unreasonable ineffectiveness of deeper layers: https://openreview.net/forum?id=ngmEcEer8a

**Final insight for you, Andrej**: 这个工作和你一贯强调的 "scaled dot-product attention is all we need; what matters is parameter allocation" 的思想是相通的。><former 揭示了一个被忽视的设计自由度，并且数学上证明了 parameter-matched 必然带来 FLOPs/KV cache 收益——这一点很值得。下一步有趣的方向是：（1）optimal shape 可能是 layer-wise continuous function 而非简单几何级数，可以用 NAS 搜索；（2）与 Hyper-Connections 结合，在 residual stream 的 deterministic carry-forward 与 learned mixing 之间找 trade-off；（3）把同样思路用到 sequence dimension（已经有 Funnel/Hourglass），或者 head dimension 上。
