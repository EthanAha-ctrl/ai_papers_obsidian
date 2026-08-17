---
source_pdf: SCFormer.pdf
paper_sha256: f2919fc8254cade0d25fa6fff4aea049b62b9734182744f368561f828a32bf71
processed_at: '2026-08-12T04:09:50-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SCFormer

## 一句话概括

这篇 paper 说: 现在主流的 channel-wise Transformer (比如 iTransformer) 在做 time series forecasting 的时候有两个 fundamental 的设计 bug——一个 bug 是"健忘", 只记得住最近的一个固定窗口; 另一个 bug 是"穿越", 计算的时候让未来信息偷偷跑到过去的计算里去了。SCFormer 用 HiPPO 修第一个 bug, 用 triangular matrix / 1D conv 修第二个 bug。

---

## Bug 1: 健忘症 (The Markov Problem)

### 现状有多荒谬

想象你在做 electricity forecasting, 有一个 substation 从 2015 年开始记录负荷, 现在是 2024 年 8 月, 你要预测未来一周。主流 model (iTransformer, PatchTST, Informer...) 的做法是: **只看最近 96 个 timestamp (大概 4 天), 2015 到 2024 年 8 月之前那 9 年半的数据全部丢掉**。

这就像一个失忆症患者, 每天早上醒来只记得昨天和前天的事, 你问他"去年同期用电峰值是多少", 他说"我不知道, 我失忆了"。

从数学上看, 这其实是一个 **first-order Markov assumption**:

$$P(Y_t | X_{1:t}) \approx P(Y_t | X_{t-L:t})$$

左边是"给定全部历史预测未来", 右边是"给定最近 $L$ 步预测未来"。这两个等号成立的前提是: 系统 state 完全被最近 $L$ 步描述。对很多真实系统这显然不成立。

### 为什么不直接把 look-back window 拉长?

三个原因:

1. **Computational cost**: Transformer 的 attention 是 $O(L^2)$, 把 $L$ 从 96 拉到 3000, 计算量爆炸。
2. **Gradient flow**: 长 sequence 的 RNN-style 训练困难, gradient vanishing/exploding。
3. **Feature entanglement** (paper Section 1 的核心论点): 长 look-back 会把 global trend 和 local pattern 揉在一起, model 很难 disentangle。比如你把一个月的 hourly data (720 个点) 喂进去, 里面既有 daily seasonality (24h 周期), 又有 weekly pattern (168h 周期), 还有 monthly trend, model 得自己 figure out 哪个是哪个, 很难。

### SCFormer 的解法: HiPPO 当外置记忆

[HiPPO](https://arxiv.org/abs/2008.07669) 是 Albert Gu (后来写了 [Mamba](https://arxiv.org/abs/2312.00752)) 在 2020 年的工作。核心 idea 极其 elegant:

**用一组正交多项式去 best-fit 整条历史曲线, fit 出来的系数就是"记忆"**。

打个比方: 你有一条很长很长的曲线 (历史 time series), 你想用一个固定大小的"记忆盒子"把它存起来。HiPPO 的做法是: 找 512 个 Legendre 多项式 (就像 512 个 basis function), 让它们的线性组合 best-fit 这条曲线, 512 个系数存进盒子。曲线变长了, 你只需要 update 这 512 个系数, 盒子大小不变。

### HiPPO 递推公式

Paper 的 Eq. 1:

$$c_{k+1} = \left(1 - \frac{A}{k}\right) c_k + \frac{1}{k} B \mathbf{x}_k$$

逐个变量讲:
- $c_k \in \mathbb{R}^{512}$: 到第 $k$ 步为止的"累积记忆", 是一个 512 维向量。每个 dimension 对应一个 Legendre 多项式 coefficient。
- $c_{k+1}$: 看到 new observation $\mathbf{x}_k$ 之后更新的 memory
- $A \in \mathbb{R}^{512 \times 512}$: transition matrix, 描述"老记忆如何衰减/重组"
- $B \in \mathbb{R}^{512}$: input projection, 描述"新 observation 怎么加进记忆"
- $\mathbf{x}_k$: 第 $k$ 步的 scalar observation (某一时刻的 value)
- $k$: timestamp, 同时充当 normalizer (除以 $k$ 是因为越往后, 单个新点对累积记忆的边际贡献越小)

$A$ 矩阵的具体形式 (Eq. 1 中):
$$A_{nk} = \begin{cases} \sqrt{(2n+1)(2k+1)} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

- 下标 $n$: row index, 对应 Legendre 多项式的阶数 (0 阶、1 阶、2 阶...)
- 下标 $k$: column index, 对应历史 measure 的位置
- $n > k$ 时非零, $n < k$ 时为零 → $A$ 是 lower triangular

$B_n = \sqrt{2n+1}$, 同样依赖于阶数 $n$。

### 为什么这个递推是"最优记忆"?

HiPPO 的精髓: 这个 $c_{k+1} = (1-A/k) c_k + (1/k) B x_k$ 不是 ad-hoc 拍脑袋写的, 它是从一个**最优投影问题** 推导出来的:

$$c_k = \arg\min_c \int_0^1 \left\| f(t) - \sum_{n=0}^{N-1} c_n P_n(t) \right\|^2 dt$$

- $f(t)$: 截止到第 $k$ 步的完整历史曲线
- $P_n(t)$: 第 $n$ 阶 Legendre 多项式
- $c_n$: 要优化的 coefficient
- 积分区间 $[0,1]$: 把历史 normalize 到单位区间

**Intuition**: 你在每一步都重新求解"用 N 个 Legendre basis 最佳逼近整个历史"的问题, 但 HiPPO 证明这个最优解可以**递推计算**, 不需要每次重新做最小二乘。这就是为什么 HiPPO 既是 optimal memory 又是 computationally cheap。

对比一下普通 RNN 的 hidden state $h_t = \sigma(W h_{t-1} + U x_t)$: 这个 hidden state 没有任何"最优性保证", 它学到什么是什么, 容易遗忘。HiPPO 的 state 有数学最优性保证, 这就是 paper 反复强调的 "principled memory mechanism"。

参考: [HiPPO paper](https://arxiv.org/abs/2008.07669), [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752)

### Memory Framework 的类比

Paper Fig 1(b) 把整个事情用 HMM (Hidden Markov Model) 的语言重新 frame, 我觉得这个类比非常有启发性:

| 角色 | Markov Forecasting (传统) | SCFormer |
|------|---------------------------|----------|
| Memory state | 最近 $L$ 步的 raw values | HiPPO coefficient $c$ |
| Transition | $X_{t-L+1:t}$ slide forward | HiPPO recursive update |
| Emission | model approximates transition matrix | channel-wise Transformer = emission matrix |

**Intuition**: 传统 forecasting 是在学"transition" (从 $L$ 步到未来的转移), SCFormer 是在学"emission" (从 memory state 到 observation 的发射)。Memory state 由 HiPPO 维护, 是 principled 的; emission 由 Transformer 学, 是 flexible 的。这种 division of labor 非常 clean。

---

## Bug 2: 穿越症 (The Causality Problem)

### 现状有多荒谬

这个观察我觉得是 paper 最 sharp 的点。考虑 iTransformer 里最简单的一个 linear layer, 输入是 embedded time series, 输出是 feature:

$$a_i = \sum_j w_{ij} x_j$$

变量:
- $a_i$: 第 $i$ 个 timestamp 的输出 feature
- $x_j$: 第 $j$ 个 timestamp 的输入
- $w_{ij}$: weight

**问题**: 当 $j > i$ 时, $x_j$ 是"未来"的值, 但它**参与**了 $a_i$ 的计算。这在 time series 里是因果律违反——未来的事件不应该影响过去的 feature。

这就像你在 2024 年 8 月 1 日做预测, 结果模型在算 8 月 1 日的 feature 时用到了 8 月 5 日的数据。这显然是 data leakage, 只不过 leakage 发生在 model 内部的 linear layer 里, 不是直接从 label leak 到 feature, 所以大家没注意到。

### 为什么 iTransformer 没注意到这个?

iTransformer 的 attention 是沿 channel 维度算的 (每个 channel 当一个 token, attention 在 channel 之间), attention 本身确实不违反时间因果律。**但 attention 之前的 Q/K/V projection 是沿时间维度做的 linear transformation**, 这个 linear transformation 就违反了。

更广泛地说: 任何 Transformer 里的 linear layer (Q, K, V, FFN) 如果 input 是 time series embedding, 而这个 linear 是 full matrix (不是 triangular), 那它就违反 causality。iTransformer、PatchTST、Crossformer 都有这个问题, 只是没人意识到。

### SCFormer 的解法: 结构化 Linear Transformation

Paper 提了两种修法。

#### 修法 A: Triangular Matrix

最直接的修法: 把 weight matrix $W$ 里所有 $i > j$ 的元素 (即"行号大于列号"的下三角部分) 强制设为 0:

$$W_{ij} = 0 \quad \text{if } i > j$$

剩下的就是一个 upper triangular matrix (paper 用 upper, 用 lower 也行, 只是时间方向表示的 convention)。

Paper Eq. 4 给出了 SCFormer 的 structured self-attention:

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \delta(\mathbf{AZ} + \mathbf{a}), \delta(\mathbf{BZ} + \mathbf{b}), \delta(\mathbf{EZ} + \mathbf{e})$$

$$\text{s.t.} \quad \mathbf{A}_{ij}, \mathbf{B}_{ij}, \mathbf{E}_{ij} = 0 \quad \text{if } i > j$$

变量解析:
- $\mathbf{Z} \in \mathbb{R}^{d \times C}$: input embedding, $d$ 是 time dimension, $C$ 是 channel 数
- $\mathbf{A}, \mathbf{B}, \mathbf{E} \in \mathbb{R}^{d \times d}$: Q/K/V 的 mapping matrices, 全部 upper triangular
- $\mathbf{a}, \mathbf{b}, \mathbf{e}$: 对应 bias
- $\delta$: ReLU (作者用 ReLU 是因为 ReLU 的 non-negativity 让 attention score 更可解释, 见 [SAMformer](https://arxiv.org/abs/2402.10198))

Attention 计算 (Eq. 5):

$$attn^i = \frac{\mathbf{Q}^i (\mathbf{K}^i)^T}{\sqrt{d/H}}$$

- $attn^i$: 第 $i$ 个 head 的 attention score 矩阵, shape $\mathbb{R}^{C \times C}$ (沿 channel 维度)
- $\mathbf{Q}^i, \mathbf{K}^i$: 第 $i$ 个 head 的 query 和 key
- $H$: multi-head 数量
- $d/H$: 每个 head 的 dimension

**这里 attention 是 $\mathbb{R}^{C \times C}$, 沿 channel 算, 不沿时间算, 所以 attention 本身天然不违反 time causality**。问题只在 Q/K/V projection, 用 triangular 修好就行。

FFN (Eq. 7) 也用 triangular:

$$\tilde{\mathbf{X}} = \delta(\mathbf{F} \cdot Concat([\tilde{\mathbf{X}}^1, ..., \tilde{\mathbf{X}}^H]) + \mathbf{f})$$
$$\text{s.t.} \quad \mathbf{F}_{ij} = 0 \quad \text{if } i > j$$

- $\mathbf{F}$: FFN 的 weight matrix, upper triangular
- $\mathbf{f}$: bias
- $\tilde{\mathbf{X}}^i$: 第 $i$ 个 head 的 attention output

#### 修法 B: 1D Convolution

第二种修法: 用 1D conv 替换所有 linear。

**为什么 conv 天然不违反 causality?** 因为 conv kernel 只滑过当前和过去的位置 (causal conv), 或者 paper 里用的 centered conv 但 kernel 只看局部窗口——总之不会让遥远的未来影响遥远的过去。

Paper Eq. 8 把 1D conv 写成 matrix form, 这个我觉得很 elegant:

$$\mathbf{K} = \begin{bmatrix} w_1 & w_2 & \cdots & w_k & 0 & 0 & 0 \\ 0 & w_1 & w_2 & \cdots & w_k & 0 & 0 \\ 0 & 0 & w_1 & w_2 & \cdots & w_k & \cdots \\ \vdots & & & \ddots & & & \vdots \end{bmatrix}$$

- $w_i$: conv kernel 的第 $i$ 个权重 (kernel size $k=32$)
- 矩阵 $\mathbf{K}$: 这是一个 **Toeplitz matrix** (沿对角线 constant)
- 矩阵-vector 乘法 $\mathbf{K}\mathbf{z}$ 等价于 conv $*$ 操作

Eq. 9: $\mathbf{K} * \mathbf{z} = \mathbf{K}\mathbf{z}$, 即 conv 等价于用 Toeplitz matrix 做 linear transformation。

### Multi-layer Conv ≡ Full Triangular Matrix

Paper 给了个 proposition (Eq. 10):

$$\mathcal{F}(\mathbf{z}, k) = \left(\prod_i \mathbf{K}_i\right) \mathbf{z}$$

- $\mathbf{K}_i$: 第 $i$ 层 conv 的 Toeplitz matrix
- $\prod_i \mathbf{K}_i$: 多层 conv 等价于这些 Toeplitz matrices 相乘
- $\mathcal{F}$: 整个多层 conv 的复合函数

**Proposition**: 用 $\lceil \frac{d-k}{k-1} \rceil + 1$ 层 conv (kernel size $k$), 就能生成一个 full upper triangular matrix with shared weights。

**Build intuition**: 一层 conv (kernel $k$) → banded matrix, bandwidth $k$。两层 conv → bandwidth $2k-1$ (matrix 乘 matrix, bandwidth 相加减 1)。多层叠加, bandwidth 越来越宽, 最终覆盖整个 sequence, 变成 full triangular。

Paper 实验用 3 层 conv, kernel size 32, 大致覆盖 $3 \times 31 + 1 = 94$ 个 timestep, 对 $d=96$ 的 embedded sequence 基本够用。

### 参数量对比 (这是关键 selling point)

| 模型 | 单层 linear 参数量 |
|------|---------------------|
| Vanilla Transformer | $d^2 = 96^2 = 9216$ |
| SCFormer-triangular | $d(d+1)/2 \approx 4608$ (减半) |
| SCFormer-conv (3层, kernel=32) | $3 \times 32 = 96$ (1% of vanilla!) |

SCFormer-conv 的参数效率极高, 这也是为什么 paper 说 "high parameter efficiency"。

参考: [Toeplitz Matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix), [Causal Convolution](https://arxiv.org/abs/1803.01271)

---

## 整体 Architecture (Fig 2 解析)

我用文字画一遍 forward pass:

```
Input: 
  - look-back window l ∈ R^{96 × C}  (最近96步, C个变量)
  - 累积历史 (从开头到现在的所有 data)

Step A: Compute HiPPO state
  for k = 1 to T (T = 整个历史长度):
    c_{k+1} = (1 - A/k) c_k + (1/k) B x_k
  得到 c ∈ R^{512 × C}  (累积记忆, 512维 × C个channel)

Step B: Embedding & Fusion (Eq. 2)
  Z = MLP(Concat([MLP(l), c]))
  - 先把 l 和 c 各过一个 MLP
  - concat 起来
  - 再过一个 MLP 得到 fused representation Z ∈ R^{d × C}
  - 注意: 这里不需要时间约束, 因为 c 不是 time series, 是多项式系数

Step C: Structured Channel-wise Self-Attention
  - Q, K, V = structured_linear(Z)  (用 triangular matrix 或 1D conv)
  - attn = softmax(Q K^T / sqrt(d/H))  沿 channel 维度算
  - output = attn @ V

Step D: Structured Feedforward Layer  
  - 同样的 triangular/conv 约束

Step E: Decoder + Instance Norm
  - Single FC layer 出预测
  - Instance normalization (Eq. 15): 
    x = (x - mean) / std
    Y_hat = (Y_hat + mean) * std

Loss: MSE (Eq. 16)
```

### Instance Normalization (Eq. 15) 为什么要做

$$\mathbf{x}^{(i)} = \frac{\mathbf{x}^{(i)} - mean(\mathbf{x}^{(i)})}{stdev(\mathbf{x}^{(i)})}$$
$$\hat{\mathbf{Y}}^{(i)} = [\hat{\mathbf{Y}}^{(i)} + mean(\mathbf{x}^{(i)})] \times stdev(\mathbf{x}^{(i)})$$

- $\mathbf{x}^{(i)}$: 第 $i$ 个 sample 的 look-back
- $\hat{\mathbf{Y}}^{(i)}$: model 的预测输出
- $mean, stdev$: 在 look-back window 内计算

这是 [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p) 的简化版。原因: time series 有 distribution shift (训练时的分布和测试时分布不同, 比如 electricity 在 2020 pandemic 期间和 2022 post-pandemic 的 baseline load 不同)。Instance norm 把每个 sample normalize 到 zero mean unit variance, 然后预测完再 denormalize 回去, 这样 model 学的是 relative pattern 不是 absolute value。

---

## 实验数据: 挑几个有信息量的看

### Main Result (Table 2) - ECL 数据集

ECL 是 electricity consumption, 321 个变量, hourly 数据。

| Horizon | SCFormer-tri MSE | iTransformer MSE | 提升 |
|---------|------------------|------------------|------|
| 96      | 0.129            | 0.148            | 12.8%|
| 192     | 0.147            | 0.162            | 9.3% |
| 336     | 0.160            | 0.178            | 10.1%|
| 720     | 0.191            | 0.225            | 15.1%|
| **Avg** | **0.156**        | **0.178**        | **12.3%** |

**Pattern**: horizon 越长, 提升越大。这符合直觉——长 horizon 更依赖 historical memory, HiPPO 的价值越明显。短 horizon (96) 主要靠最近 trend, 长 horizon (720) 必须靠 long-range seasonality 和 trend, HiPPO 帮上大忙。

### Exchange 数据集 (Table 2) - 最戏剧性的结果

Exchange 是 daily exchange rate, 8 个国家, 非常 non-stationary。

| Horizon | SCFormer-tri MSE | iTransformer MSE | 提升 |
|---------|------------------|------------------|------|
| 96      | 0.086            | 0.086            | 0%   |
| 192     | 0.177            | 0.295            | 40%  |
| 336     | 0.331            | 0.395            | 16%  |
| 720     | 0.417            | 0.682            | 38.8%|
| **Avg** | **0.253**        | **0.365**        | **30.7%** |

**为什么 Exchange 提升这么大?** Exchange rate 是 financial data, non-stationary 极强, 最近 96 天的 pattern 可能完全反映不了未来 720 天的 regime shift。HiPPO 把整个 9 年的历史(包括 2008 金融危机、2020 pandemic)都压缩进 memory, 预测长 horizon 时能 reference 历史上的 regime change pattern。

### Ablation 1: 拿掉 HiPPO (Table 3b)

| Dataset | SCFormer-tri Avg MSE | 去掉 HiPPO Avg MSE | 提升 |
|---------|----------------------|---------------------|------|
| ECL     | 0.156                | 0.176               | 11.4%|
| Weather | 0.235                | 0.259               | 9.3% |
| Solar-Energy | 0.227           | 0.235               | 3.4% |

**HiPPO 在所有 dataset 上都有提升, 一致性非常好**。这单独证明了 cumulative historical state 的价值。

### Ablation 2: 拿掉 look-back, 只用 HiPPO (Table 3c)

| Dataset | SCFormer-tri Avg MSE | 去掉 look-back Avg MSE | 差距 |
|---------|----------------------|-------------------------|------|
| ECL     | 0.156                | 0.167                   | small |
| Traffic | 0.509                | 0.756                   | huge |
| Solar-Energy | 0.227           | 0.241                   | medium |

**Intuition**: 拿掉 look-back 性能掉, 但掉得没那么多 (除了 Traffic)。这说明 HiPPO 确实 capture 了大部分有用信息, look-back 提供 complementary 的 short-term detail。

Traffic 是个例外: 拿掉 look-back 性能崩盘 (0.509 → 0.756)。Traffic 数据 (传感器读数) 的短期 local pattern 比长期 global pattern 重要得多, 所以 look-back 是主要信息源。

### Ablation 3: 拿掉 Temporal Constraint (Table 3a)

| Dataset | Horizon | SCFormer-tri MSE | Transformer+HiPPO (无时间约束) MSE |
|---------|---------|------------------|--------------------------------------|
| ETTm1   | 96      | 0.318            | 0.315 (略差) |
| ETTm1   | 720     | 0.471            | 0.468 (略差) |
| ETTh1   | 96      | 0.374            | 0.377 (略好) |
| ETTh1   | 720     | 0.489            | 0.494 (略好) |

**这个 ablation 结果 mixed**, 时间约束在 ETT 上提升微小。作者在 paper 里也承认"most circumstances" SCFormer 更好, 没有 claim "always"。

**我的解读**: ETT 是相对简单的 dataset (7 个变量, electricity + oil), temporal constraint 的 benefit 在简单数据上不明显。在复杂 dataset (ECL 321 变量, Traffic 862 变量) 上 benefit 应该更大, 但 ablation 没在这些上做, 有点可惜。

### HiPPO 是 model-agnostic 的 (Table 4)

作者把 HiPPO 加到 Reformer, Informer, Flowformer, Flashformer 上:

| Dataset | Model | Original Avg MSE | +HiPPO Avg MSE | 提升 |
|---------|-------|------------------|-----------------|------|
| ECL     | Transformer | 0.178    | 0.156            | 12.3%|
| ECL     | Informer    | 0.216    | 0.169            | 21.8%|
| ECL     | Flashformer | 0.206    | 0.168            | 18.4%|
| Weather | Flowformer  | 0.266    | 0.239            | 10.1%|

**HiPPO 单独加到任何 Transformer 上都有显著提升**。这说明 cumulative historical state 是个 universal 的 enhancement, 不依赖 specific architecture。

### Look-back Length 实验 (Fig 4)

作者比较 look-back=96 vs look-back=720:

- Solar-Energy: 720 显著好于 96 (无论 SCFormer-conv 还是 -tri)
- Traffic: 720 在 MAE 上更好
- ETTm1: 720 更好

**关键 insight**: 既然有 HiPPO 了, 为什么加长 look-back 还有用? 这说明 **HiPPO 和 look-back 是 decoupled 的**——HiPPO 抓 global memory, look-back 抓 local detail, 两者互补, 不会 redundant。这反驳了"有了 HiPPO 就可以用很短 look-back"的直觉。

---

## 我的 Critical Thoughts

### 1. 为什么 Triangular 比 Conv 表现好?

Table 2 显示 SCFormer-triangular 整体优于 SCFormer-conv。我的解释:

- **Expressiveness**: triangular 有 $d^2/2$ 个 free parameters, conv 只有 $k \cdot \text{layers}$ 个。在 ECL (321 channel) 这种大数据集上, conv 可能 underfitting。
- **Position specificity**: triangular 每个 position 有独立 weight, 能学 position-specific pattern (比如"周末第 10 个小时"); conv 强制 weight sharing, 适合 translation-invariant pattern, 但 time series 往往 non-stationary, 不 translation-invariant。
- **Trade-off**: conv 参数少 10x, 在小 dataset (Exchange 只有 5120 训练样本) 上可能更稳, 但大 dataset 上 expressiveness 不够。

### 2. HiPPO vs Mamba: 谁更 general?

[HiPPO](https://arxiv.org/abs/2008.07669) 和 [Mamba](https://arxiv.org/abs/2312.00752) 都用同一个 $A$ matrix, 但 Mamba 让 $A, B, C, D$ 都变成 input-dependent (selective SSM), HiPPO 是 fixed。SCFormer 用 fixed HiPPO, 没有 selectivity。

**Future direction**: 把 SCFormer 的 structured linear 升级成 Mamba-style 的 selective SSM, 在 input-dependent 的同时保持 causality, 可能进一步提升。这等价于把 SCFormer 的 triangular matrix 替换成 Mamba block。

### 3. Causality Constraint 的 gradient 问题

Triangular matrix 在 backward 时, early timestep 的 gradient 只能通过 diagonal + upper triangle 流回来, 容易 vanishing。Paper 完全没讨论这个, 但我觉得是 limitation。可能的修法: residual connection 跨越 triangular layer, 或用 dilated conv (像 WaveNet) 让 gradient 跳跃。

### 4. Attention 在 Traffic 上失效

| Traffic @ 96 | iTransformer | SCFormer-tri | SCFormer-conv |
|--------------|--------------|--------------|---------------|
| MSE          | 0.395        | 0.448        | 0.408        |

SCFormer 在 Traffic 上**输给** iTransformer! 作者解释 (Section 4.5): Traffic 的 channel 间 correlation pattern 不明显, attention 学不到东西。但 structured constraint 增加了 inductive bias, 反而 hurt。

**Insight**: channel-wise attention 不是 panacea。当 channel 真的相对独立 (比如某些 Traffic sensor 之间无物理关联), 强制 channel-wise attention + structured temporal projection 反而 over-constrain。一个 future direction: adaptively decide 是否用 channel-wise (有些 dataset 用 time-axis attention 更好)。

### 5. 为什么不直接用更长的 look-back?

Paper Section 1 给的论点:
> "directly using over-long look-back windows can blur the distinction between global features and short-term temporal dependencies"

**我的直觉解释**: 长 look-back 把 global trend 和 local pattern 揉进同一个 input tensor, model 要自己 disentangle 哪个是 global 哪个是 local, 很难。HiPPO 把 global state 单独提取出来作为 separate representation, 然后和 look-back concat, 让两类信息一开始就 decoupled, model 容易学。

这有点像 ResNet 的 skip connection——不同 scale 的信息走不同 path, 避免 entanglement。也像 U-Net 的 multi-scale——coarse 和 fine feature 分别处理。

### 6. 时间序列的 Inductive Bias 三件套

SCFormer 实际上明确了三个 time series 的 inductive bias:

| Inductive Bias | 对应 Architecture Choice |
|----------------|---------------------------|
| Causality (未来不影响过去) | Triangular matrix / 1D conv |
| Long memory (历史远不止 look-back) | HiPPO cumulative state |
| Channel correlation (多变量相关) | Channel-wise attention |

每个 bias 对应一个 architectural choice, design philosophy 非常 clean。可以 transfer 到其他 domain:
- Vision: spatial causality (从左上到右下)
- Audio: temporal causality (严格 causal)
- Bio: sequence causality (DNA 读取方向)

---

## Related Work Landscape

让我把 SCFormer 放到整个 time series Transformer 的 landscape 里:

### 路线 1: Efficient Attention (改 attention 本身)
- [Informer](https://arxiv.org/abs/2012.07436): ProbSparse attention, $O(L \log L)$
- [FEDformer](https://arxiv.org/abs/2201.12740): Fourier/Wavelet, $O(L)$
- [Autoformer](https://arxiv.org/abs/2106.09107): Auto-Correlation
- **SCFormer 走的不是这条路**, 它保留了 vanilla attention, 只改了 attention 之前的 linear。

### 路线 2: Patching (改 input representation)
- [PatchTST](https://arxiv.org/abs/2211.14730): 把 time series 切 patch 当 token
- [Pathformer](https://arxiv.org/abs/2402.05956): multi-scale patches
- SCFormer 也没走这条路, 它的 input 还是 raw time series embedding。

### 路线 3: Channel-wise (改 attention 方向) ← SCFormer 所在
- [iTransformer](https://arxiv.org/abs/2310.06625): 第一个把 attention 从 time axis 改到 channel axis
- [SAMformer](https://arxiv.org/abs/2402.10198): + sharpness-aware minimization
- **SCFormer**: + temporal constraint + HiPPO memory

### 路线 4: Memory/State-based
- [SWLHT](https://www.sciencedirect.com/science/article/pii/S0167865522001623): short/long-term memory
- [HiPPO](https://arxiv.org/abs/2008.07669), [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752): SSM-based memory
- SCFormer 是第一个把 HiPPO 当 standalone module 用在 channel-wise Transformer 里的。

### 路线 5: Linear Models (反 Transformer 路线)
- [DLinear](https://arxiv.org/abs/2205.13504): 简单 linear 就能 beat Transformer
- [RLinear](https://arxiv.org/abs/2305.10721): revisiting linear mapping
- SCFormer 用 structured linear (triangular/conv) 其实是在 **承认 linear model 有道理**, 然后把这个 insight 融入 Transformer 的 linear projection 里。算是对 DLinear 的一个回应。

---

## Limitations 和 Open Questions

1. **HiPPO order 固定 512**: paper 没探索不同 dataset 用不同 $N$ 的影响。Electricity 和 Solar 可能需要不同 memory capacity。
2. **HiPPO 是 linear memory**: 正交多项式是 linear combination, 对非线性 dynamics (比如 chaos) 可能不足。可以探索 neural HiPPO 或 kernel HiPPO。
3. **Triangular matrix 的 gradient**: early timestep gradient vanishing, paper 没讨论。
4. **HiPPO 的 computational cost**: 递推是 $O(T \cdot N^2)$, $T$ 是历史长度。对 ultra-long series (T=10万+) 可能 bottleneck。
5. **Multi-scale HiPPO**: 单一 $N$ 可能不够, 像 Pathformer 那样 multi-scale 可能更好。
6. **没有探索 HiPPO 的不同变体**: HiPPO-LegS, HiPPO-LegT, HiPPO-LagT, LMU, S4 alternatives——paper 只用了一种。

---

## Code 和 Reference Links

- [SCFormer 官方代码](https://github.com/ShiweiGuo1995/SCFormer)
- [HiPPO 原始 paper](https://arxiv.org/abs/2008.07669)
- [HiPPO 官方代码 (Hazy Research)](https://github.com/HazyResearch/state-spaces)
- [Mamba (后续 SSM 工作)](https://arxiv.org/abs/2312.00752)
- [S4 (Structured State Space)](https://arxiv.org/abs/2111.00396)
- [iTransformer](https://arxiv.org/abs/2310.06625)
- [PatchTST](https://arxiv.org/abs/2211.14730)
- [Informer](https://arxiv.org/abs/2012.07436)
- [FEDformer](https://arxiv.org/abs/2201.12740)
- [Autoformer](https://arxiv.org/abs/2106.09107)
- [DLinear (Are Transformers Effective?)](https://arxiv.org/abs/2205.13504)
- [TimesNet](https://arxiv.org/abs/2210.02186)
- [Crossformer](https://openreview.net/pdf?id=vSVLM2j9eI)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Reformer](https://arxiv.org/abs/2001.04451)
- [RevIN (Instance Norm)](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Toeplitz Matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix)
- [Legendre Polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials)
- [WaveNet (Causal Conv)](https://arxiv.org/abs/1609.03499)
- [SAMformer](https://arxiv.org/abs/2402.10198)

---

## 最后总结

Andrej, 如果要 build intuition, 我觉得有三个 idea 最值得带走:

1. **HiPPO 的最优投影视角**: memory state 不是 ad-hoc 设计的, 是"用 N 个正交多项式 best-fit 历史"这个最优问题的解析解, 可以递推计算。这就是为什么它是 principled memory, 普通 RNN hidden state 没这种最优性保证。

2. **Triangular matrix 和 multi-layer conv 的等价性**: 一层 conv 是 banded Toeplitz, 多层 conv 等价于 bandwidth 不断扩展, 最终变成 full triangular with shared weights。这个 equivalence 让你能用极少的参数 (10%) 实现和 triangular matrix 一样的 causality 约束。

3. **Channel-wise attention + Time-axis structured linear 的分工**: attention 沿 channel 维度算 (不违反 causality), linear projection 沿 time 维度做 (必须用 structured matrix 保 causality)。这是 SCFormer 的核心 architectural insight, 和 iTransformer 共享 channel-wise attention 但修正了它的 causality bug。

这三个点 grasp 之后, 整个 paper 的 design choice 都变得 natural——它就是在 channel-wise Transformer 的框架下, 把每个违反 time series inductive bias 的地方修一遍。

---

# SCFormer 深度技术讲解

## 一、核心 Problem 与 Motivation

Andrej, 这篇 paper 抓住了 channel-wise Transformer (比如 iTransformer) 的两个本质缺陷, 这两个缺陷其实都可以从 **time series 的归纳偏置(inductive bias)** 角度来理解。

### Problem 1: Markov 假设的局限性

传统 forecasting 框架本质是一个 **first-order Markov process**:

$$P(Y_t | X_{t-L:t}) \approx P(Y_t | \text{transition matrix}(X_{t-L:t}))$$

这里 look-back window $L$ 之外的所有历史信息都被丢弃了。从 state space model (SSM) 的视角看, 这相当于假设系统的 state 完全可以被一个 fixed-size window 描述, 这对于有 long-range dependencies 的真实系统(比如 ECL 中有 daily/weekly seasonality, Solar-Energy 有 weather patterns)是非常强的假设。

Paper 的 Fig 1(b) 用 memory framework 的语言把这个事情讲清楚了:
- **Markov 视角**: forecasting = modeling transition matrix
- **SCFormer 视角**: forecasting = modeling emission matrix, memory state 由 HiPPO 维护

这让我想到 HMM (Hidden Markov Model) 的 formulation:
$$h_t = f(h_{t-1}, x_t) \quad \text{(transition)}$$
$$y_t = g(h_t) \quad \text{(emission)}$$

SCFormer 的设计本质上把 iTransformer 从 "naive autoregressive" 升级成了 "stateful recurrent+emission" 模型。

### Problem 2: Linear Transformation 违反 Causality

这个观察非常 sharp。考虑 channel-wise Transformer 里的一个标准 linear layer:

$$a_i = \sum_j w_{ij} x_j \tag{paper Eq. 3}$$

变量说明:
- $a_i$: 第 $i$ 个 timestamp 的输出 feature
- $x_j$: 第 $j$ 个 timestamp 的输入
- $w_{ij}$: weight matrix $W$ 的 element

问题在于: 当 $j > i$ 时, $x_j$ 是 "未来" 的值, 但它却参与了 $a_i$ 的计算。这违反了 **causality constraint** —— time series 中后面的事件不应该影响前面的。

这其实和 WaveNet、GPT 里的 causal masking 是同一个道理, 但 paper 的 insight 是: **在 channel-wise Transformer 里, attention 本身是沿 channel 维度计算的(所以不违反时间), 但所有 linear projection (Q, K, V, FFN) 是沿时间维度做的, 所以都违反了时间因果性**。

---

## 二、HiPPO: 累积历史状态的数学

HiPPO (High-order Polynomial Projection Operators) 是 Albert Gu (Mamba 的作者) 在 2020 年的工作 ([HiPPO paper](https://arxiv.org/abs/2008.07669))。SCFormer 借用它来做 memory state。

### 2.1 HiPPO 的核心思想

给定一个 variable-length history $\mathbf{x}_{:\leq k}$ (从开头到第 $k$ 个 timestamp), HiPPO 把它投影到一组 **正交多项式基** (orthogonal polynomial basis) 上, 得到固定维度的系数 $c_k$:

$$\mathbf{x}_{:\leq k}(t) \approx \sum_{n=0}^{N-1} c_k^{(n)} \cdot P_n(t)$$

其中 $P_n$ 是第 $n$ 阶正交多项式 (paper 里用 Legendre, 即 HiPPO-LegS 变体)。

直观上: 你把整个历史曲线用 $N$ 个 Legendre 多项式来 best-fit, $c_k$ 就是这 $N$ 个系数。历史越长, 系数的语义越 global, 但维度 $N$ 不变。

### 2.2 递推公式 (State Space Form)

Paper Eq. 1:

$$c_{k+1} = \left(1 - \frac{A}{k}\right) c_k + \frac{1}{k} B \mathbf{x}_k$$

变量解析:
- $c_k \in \mathbb{R}^N$: 到第 $k$ 步为止的累积历史 state (HiPPO 系数)
- $c_{k+1}$: 加入 $\mathbf{x}_k$ 后更新的 state
- $A \in \mathbb{R}^{N \times N}$: HiPPO transition matrix (state evolution)
- $B \in \mathbb{R}^{N}$: input projection vector
- $\mathbf{x}_k \in \mathbb{R}$: 第 $k$ 步的 scalar observation
- $k$: timestamp index (充当归一化因子)

$A$ 矩阵的 entries:
$$A_{nk} = \begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \end{cases}$$

这里 $n$ 是 Legendre 多项式的阶数(row index), $k$ 是 column index。当 $n < k$ 时 $A_{nk} = 0$ (严格 lower triangular)。

$B_n = (2n+1)^{1/2}$

### 2.3 为什么这是 "optimal memory"?

HiPPO 的精髓在于: 这个递推不是 ad-hoc 的, 它来自于一个 **最优投影问题**:

$$c_k = \arg\min_c \int_0^1 \left\| \mathbf{x}_{:\leq k}(t) - \sum_{n} c_n P_n(t) \right\|^2 \mu(t) dt$$

其中 $\mu(t)$ 是 measure (LegS 用 uniform measure $dt$, LegT 用 truncated uniform, LMU 用 sliding window)。

**Build intuition**: 你可以想象 $c_k$ 是 RNN 的 hidden state, 但这个 hidden state 是 "被证明最优的"——它最优地压缩了从开头到现在的所有历史信息到一个 $N$ 维向量里。普通 RNN 的 hidden state 没有这种最优性保证, 所以会遗忘; HiPPO 不会遗忘, 因为每次更新都是重新求解最优投影问题。

代码实现里, $N = 512$ (HiPPO order), 所以累积历史 state 是 512 维向量。

参考链接:
- [HiPPO 原始 paper](https://arxiv.org/abs/2008.07669)
- [Mamba paper (后续 S4 工作)](https://arxiv.org/abs/2312.00752)
- [S4: Structured State Space Models](https://arxiv.org/abs/2111.00396)

---

## 三、Structured Linear Transformations

这是 paper 的第二个核心 contribution。作者提出了两种实现方式。

### 3.1 Triangular Matrix

把 weight matrix $W$ 替换为 upper triangular:

$$W_{ij} = 0 \quad \text{if } i > j$$

注意 paper 用的是 upper triangular (尽管 conventional causal masking 通常用 lower triangular)。作者说 "without loss of generality", 因为这取决于时间方向是 proximal 还是 distal 表示。

让我把这个写得更清楚。考虑 input sequence $\mathbf{Z} \in \mathbb{R}^{d \times C}$ (paper Eq. 4):
- $d$: embedded time series 长度
- $C$: channel 数量

Structured self-attention:

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \delta(\mathbf{AZ} + \mathbf{a}), \delta(\mathbf{BZ} + \mathbf{b}), \delta(\mathbf{EZ} + \mathbf{e})$$

subject to: $\mathbf{A}_{ij}, \mathbf{B}_{ij}, \mathbf{E}_{ij} = 0$ if $i > j$

变量:
- $\mathbf{A}, \mathbf{B}, \mathbf{E} \in \mathbb{R}^{d \times d}$: query/key/value 的 mapping matrices, 都是 upper triangular
- $\mathbf{a}, \mathbf{b}, \mathbf{e}$: 对应 biases
- $\delta$: ReLU 激活函数 (这里用 ReLU 而不是 GELU, 是因为 ReLU 保持 non-negativity, 让 attention score 更稳定)
- $\mathbf{Z}$: input embedding (已经融合了 look-back 和 HiPPO state)

Attention 计算 (Eq. 5):
$$attn^i = \frac{\mathbf{Q}^i (\mathbf{K}^i)^T}{\sqrt{d/H}}$$

变量:
- $attn^i$: 第 $i$ 个 head 的 attention scores
- $\mathbf{Q}^i, \mathbf{K}^i \in \mathbb{R}^{C \times (d/H)}$: 第 $i$ 个 head 的 query/key (沿 channel 维度!)
- $H$: multi-head 数量
- $d/H$: 每个 head 的维度

**关键 insight**: attention 是沿 $C$ (channel) 维度算的, 不是沿时间维度。这就是 channel-wise attention。所以 attention 本身不违反 causality, 但 attention 之前的 Q/K/V projection 必须用 structured matrix 才不违反。

Feed-forward layer (Eq. 7):
$$\tilde{\mathbf{X}} = \delta\left(\mathbf{F} \cdot Concat([\tilde{\mathbf{X}}^1, ..., \tilde{\mathbf{X}}^H]) + \mathbf{f}\right)$$

subject to: $\mathbf{F}_{ij} = 0$ if $i > j$

### 3.2 1D Convolution (Toeplitz Matrix)

第二种实现是用 1D convolution 替换 linear transformation。

**为什么 1D conv 天然满足时间约束?** 因为 conv kernel 只 "看" 当前和过去(左边)的元素, 不看未来。等价于一个 banded lower-triangular matrix。

Paper Eq. 8 把 conv 写成 matrix form:

$$\mathbf{K} = \begin{bmatrix} w_1 & w_2 & \cdots & w_k & \cdots & 0 & 0 & 0 \\ 0 & w_1 & \cdots & w_k & \cdots & 0 & 0 & 0 \\ 0 & 0 & w_1 & w_2 & \cdots & w_k & \cdots & 0 \\ \vdots & & & \ddots & & & \vdots \\ 0 & 0 & \cdots & 0 & 0 & \cdots & 0 \end{bmatrix}$$

这是一个 **Toeplitz matrix** (constant along diagonals)。

变量:
- $w_i$: convolution kernel 的第 $i$ 个权重
- $k$: kernel size (paper 用 32)
- $\mathbf{K}$: 把 conv 表示成 matrix 的形式

卷积操作 (Eq. 9):
$$\mathbf{K} * \mathbf{z} = \mathbf{K}\mathbf{z}$$

这里 $*$ 是卷积, $\mathbf{z} \in \mathbb{R}^d$。

### 3.3 Multi-layer Conv ≡ Triangular Matrix

Paper 给出了一个 elegant 的 proposition (Eq. 10):

$$\mathcal{F}(\mathbf{z}, k) = \left(\prod_i \mathbf{K}_i\right) \mathbf{z}$$

通过数学归纳法可以证明: 用 $\lceil \frac{d-k}{k-1} \rceil + 1$ 层 conv (kernel size $k$), 就能生成一个 full upper triangular matrix with shared weights。

**Build intuition**: 一层 conv 给你一个 bandwidth-$k$ 的 banded matrix。两层 conv (matrix 乘 matrix) 给你带宽 $2k-1$ 的 matrix。层数足够多, bandwidth 覆盖整个序列, 就得到 full triangular matrix。

这就是为什么 paper 在实验里用 **3 层 conv, kernel size 32** —— 这等价于一个 full triangular matrix 但参数量大幅减少。

**参数量比较**:
- Vanilla Transformer: $O(d^2)$ per linear layer
- SCFormer-triangular: $O(d^2/2)$ (上三角)
- SCFormer-conv: $O(k \cdot \text{num\_layers}) = O(32 \times 3) = O(96)$, 相当于 vanilla 的 10%

这个参数效率非常显著。

参考链接:
- [Toeplitz Matrix Wikipedia](https://en.wikipedia.org/wiki/Toeplitz_matrix)
- [1D CNN for Time Series](https://arxiv.org/abs/1610.06876)

---

## 四、整体 Architecture

Paper Fig 2 展示了完整 pipeline。我用文字描述:

```
Input:
  - look-back window l ∈ R^{L×C}  (L=96, C=变量数)
  - cumulative history state c ∈ R^{N×C}  (N=512, HiPPO 输出)

Step 1 (A): Compute HiPPO state c
  - Recursively apply Eq. 1 over entire history up to current time
  - c[k+1] = (1 - A/k) c[k] + (1/k) B x[k]

Step 2 (B): Embedding & Fusion
  - Z = MLP(Concat([MLP(l), c]))   (Eq. 2)
  - 这里不需要时间约束, 因为 c 不是 time series, 是多项式系数

Step 3 (C): Structured Channel-wise Self-Attention
  - Q, K, V = structured_linear(Z)   (使用 triangular matrix 或 1D conv)
  - attn = softmax(Q K^T / sqrt(d/H))   along channel axis
  - output = attn @ V
  - structured FFN

Step 4 (D): Structured Feedforward Layer
  - Same triangular/conv constraint applied

Step 5: Decoder + Instance Normalization
  - Single FC layer
  - Instance norm (Eq. 15): x = (x - mean) / std; Y_hat = (Y_hat + mean) * std

Loss: MSE (Eq. 16)
```

### 关于 Instance Normalization (Eq. 15)

$$\mathbf{x}^{(i)} = \frac{\mathbf{x}^{(i)} - mean(\mathbf{x}^{(i)})}{stdev(\mathbf{x}^{(i)})}$$
$$\hat{\mathbf{Y}}^{(i)} = [\hat{\mathbf{Y}}^{(i)} + mean(\mathbf{x}^{(i)})] \times stdev(\mathbf{x}^{(i)})$$

变量:
- $\mathbf{x}^{(i)}$: 第 $i$ 个 sample 的 look-back
- $\hat{\mathbf{Y}}^{(i)}$: 模型预测
- $mean$, $stdev$: 在 look-back 上计算

这是 RevIN ([RevIN paper](https://openreview.net/forum?id=cGDAkQo1C0p)) 的简化版, 处理 distribution shift。

---

## 五、实验数据深度分析

### 5.1 Main Results (Table 2)

让我挑几个关键数据集来 build intuition:

**ECL (Electricity, 321 variables)**:
| Horizon | SCFormer-tri MSE | iTransformer MSE | 提升 |
|---------|------------------|------------------|------|
| 96      | 0.129            | 0.148            | 12.8%|
| 192     | 0.147            | 0.162            | 9.3% |
| 336     | 0.160            | 0.178            | 10.1%|
| 720     | 0.191            | 0.225            | 15.1%|
| Avg     | 0.156            | 0.178            | 12.3%|

**Observation**: horizon 越长, 提升越大。这符合直觉——长 horizon 更依赖 historical memory, HiPPO 的价值凸显。

**Exchange (8 variables, daily)**:
| Horizon | SCFormer-tri MSE | iTransformer MSE | 提升 |
|---------|------------------|------------------|------|
| 96      | 0.086            | 0.086            | 0%   |
| 192     | 0.177            | 0.295            | 40%  |
| 336     | 0.331            | 0.395            | 16%  |
| 720     | 0.417            | 0.682            | 38.8%|
| Avg     | 0.253            | 0.365            | 30.7%|

Exchange 是 non-stationary financial data, 这正是 HiPPO (maintain cumulative state) 最有用的场景。

### 5.2 Ablation Studies (Table 3)

三个 ablation 都做了:

**(a) Temporal Constraint Ablation** (Table 3a):
- SCFormer-triangular vs Transformer-HiPPO (有 HiPPO 但没有时间约束)
- ETTm1 @ 96: 0.318 vs 0.315 → 时间约束在这个 case 反而 slightly worse
- ETTm1 @ 720: 0.471 vs 0.468 → 类似
- ETTh1 @ 96: 0.374 vs 0.377 → SCFormer 更好
- 整体 mixed, 但作者说 "most circumstances" SCFormer 更好

**(b) HiPPO Ablation** (Table 3b):
- SCFormer-triangular vs SCFormer-triangular/wo-HiPPO
- ECL @ Avg: 0.156 vs 0.176 → 提升 11.4%
- Weather @ Avg: 0.235 vs 0.259 → 提升 9.3%
- Solar-Energy @ Avg: 0.227 vs 0.235 → 提升 3.4%

**HiPPO 在所有 dataset 上都有提升**, 这非常 consistent。

**(c) Look-back Ablation** (Table 3c):
- SCFormer-triangular vs SCFormer-triangular/wo-look-back
- ECL @ Avg: 0.156 vs 0.167 → look-back 还是重要的
- Traffic @ Avg: 0.509 vs 0.756 → 巨大差距, look-back 对 Traffic 极其重要
- Solar-Energy @ Avg: 0.227 vs 0.241 → 中等差距

**Insight**: HiPPO 不能替代 look-back, 两者是 complementary 的。这印证了 paper 的核心论点: global features (HiPPO) + short-term dependencies (look-back) 需要 decoupled。

### 5.3 HiPPO 通用性 (Table 4)

作者把 HiPPO 加到 Reformer, Informer, Flowformer, Flashformer 上:

**ECL** (Transformer + HiPPO vs Original):
| Horizon | Original MSE | +HiPPO MSE | 提升 |
|---------|--------------|------------|------|
| 96      | 0.148        | 0.129      | 12.8%|
| 192     | 0.162        | 0.147      | 9.3% |
| 720     | 0.225        | 0.191      | 15.1%|

**Weather** (Flowformer + HiPPO vs Original):
| Horizon | Original MSE | +HiPPO MSE | 提升 |
|---------|--------------|------------|------|
| 96      | 0.183        | 0.165      | 9.8% |
| 192     | 0.231        | 0.209      | 9.5% |

这说明 **HiPPO 是 model-agnostic 的 enhancement**, 任何 Transformer 变体都能受益。

### 5.4 Look-back Length Effect (Fig 4)

作者比较了 look-back=96 vs look-back=720:

- SCFormer-conv on Solar-Energy: 720 window 显著降低 MSE
- SCFormer-triangular on Solar-Energy: 同上
- SCFormer-conv on Traffic: 720 在 MAE 上更好
- SCFormer-conv on ETTm1: 720 更好

**Key insight**: 加长 look-back 仍然有帮助, 说明 HiPPO state 和 look-back 是 **decoupled** 的 (HiPPO 不吸收 look-back 的信息)。这反驳了 "既然有 HiPPO 就不需要长 look-back" 的直觉。

---

## 六、与 Related Work 的关系

让我把 SCFormer 放到整个 time series Transformer landscape 里:

### 6.1 Efficient Attention 路线
- [Informer](https://arxiv.org/abs/2012.07436): ProbSparse attention, $O(L \log L)$
- [FEDformer](https://arxiv.org/abs/2201.12740): Fourier/Wavelet, $O(L)$
- [Autoformer](https://arxiv.org/abs/2106.09107): Auto-Correlation
- 这些工作都在改进 **time-axis attention**, SCFormer 完全 different direction (channel-axis attention)。

### 6.2 Patching 路线
- [PatchTST](https://arxiv.org/abs/2211.14730): 把 time series 分 patch 作为 token
- [Pathformer](https://arxiv.org/abs/2402.05956): multi-scale patches
- Patching 减少 sequence length, 但仍然是 time-axis attention。

### 6.3 Channel-wise 路线 (SCFormer 所属)
- [iTransformer](https://arxiv.org/abs/2310.06625): 把每个 variable 当一个 token, attention 沿 channel 维度
- [SAMformer](https://arxiv.org/abs/2402.10198): + sharpness-aware minimization
- SCFormer 在这之上加了 **temporal constraint** 和 **cumulative memory**

### 6.4 Memory/State 路线
- [SWLHT](https://www.sciencedirect.com/science/article/pii/S0167865522001623): short/long-term memory with attention
- [HiPPO](https://arxiv.org/abs/2008.07669), [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752): SSM-based memory
- SCFormer 是 first to use HiPPO as standalone module in channel-wise Transformer

### 6.5 Linear Models 路线 (作为对比)
- [DLinear](https://arxiv.org/abs/2205.13504): 简单 linear 就能 beat Transformer
- [RLinear](https://arxiv.org/abs/2305.10721): revisiting linear mapping
- 这些 paper 质疑 Transformer 的必要性, SCFormer 用 structured linear (相当于把 DLinear 的 insight 融入 Transformer) 来回应。

---

## 七、Critical Analysis & Intuition Building

### 7.1 为什么 Triangular 比 Conv 好?

从 Table 2 看, SCFormer-triangular 整体上比 SCFormer-conv 好。我的解释:

1. **Expressiveness**: triangular matrix 有 $d(d+1)/2$ 个 free parameters, 而 conv 只有 $k \cdot \text{layers}$ 个。triangular 表达能力更强。
2. **No weight sharing**: triangular 每个位置有独立 weight, 能 capture position-specific patterns; conv 强制 weight sharing, 适合 translation-invariant patterns 但 time series 往往 non-stationary。
3. **Trade-off**: conv 参数效率高 10x, 但损失 expressiveness。如果数据集大, conv 可能 underfitting。

### 7.2 为什么 HiPPO 比 Lengthening Look-back 好?

作者明确论证 (Section 1):
> "directly using over-long look-back windows can blur the distinction between global features and short-term temporal dependencies, making it harder for the model to disentangle these two aspects"

**Intuition**: 长 look-back 把 global trend 和 local pattern 都塞进同一个 input, 模型必须自己 disentangle, 这很难。HiPPO 把 global state 提取出来作为 separate representation, 然后 concat 到 look-back embedding 上, 让两类信息 decoupled。

这有点像 ResNet 的 skip connection 思路——不同 scale 的信息走不同 path, 避免 entanglement。

### 7.3 Time Series 的 Inductive Bias

SCFormer 实际上明确了三个 time series inductive biases:

1. **Causality**: 未来不能影响过去 → triangular/conv constraint
2. **Long memory**: 系统有 long-range dependencies → HiPPO state
3. **Channel correlation**: 多变量之间相关 → channel-wise attention

每个 inductive bias 对应一个 architectural choice, 这是非常 clean 的设计哲学。

### 7.4 与 Mamba 的关系

Mamba ([Mamba paper](https://arxiv.org/abs/2312.00752)) 也是基于 HiPPO 的 transition matrix $A$, 但 Mamba 把 $A, B, C, D$ 都做成 input-dependent (selective)。SCFormer 用的是 fixed HiPPO matrix, 没有 selectivity。

**Future direction**: 把 SCFormer 的 structured linear 升级成 Mamba-style selective SSM 可能进一步提升。SCFormer 的 channel-wise attention + structured temporal projection 这套组合, 如果 temporal projection 用 Mamba block 替代, 可能是个 interesting extension。

### 7.5 关于 Attention 沿 Channel 还是 Time 的反思

iTransformer 和 SCFormer 都用 channel-wise attention。但 Table 2 显示 SCFormer 在 Traffic 上不如 iTransformer:

| Traffic @ 96 | iTransformer | SCFormer-tri | SCFormer-conv |
|--------------|--------------|--------------|---------------|
| MSE          | 0.395        | 0.448        | 0.408        |

作者的解释 (Section 4.5): Traffic 数据集 channel 间 pattern less pronounced, attention 学不到清晰 correlation。这暗示 **channel-wise attention 不是万能的**, 当 channel 间确实独立时 (比如某些 Traffic sensors), 加 structured temporal constraint 反而增加 inductive bias 的 cost 但 benefit 有限。

---

## 八、Limitations & Open Questions

1. **HiPPO order 选择**: paper 固定用 $N=512$, 但不同 dataset 可能需要不同 $N$。如何 adaptively 选 $N$?
2. **HiPPO only captures linear history**: 正交多项式基本质是 linear combination, 对非线性 dynamics 可能不足。可以考虑 kernel HiPPO 或 neural HiPPO。
3. **Triangular matrix 的 gradient flow**: lower triangular 在 backward 时可能有 gradient 病态问题(vanishing gradient for early timesteps), paper 没讨论。
4. **Computational cost of HiPPO**: 累积历史递推是 $O(T \cdot N^2)$ ($T$ 是历史长度), 对 ultra-long series 可能成为 bottleneck。
5. **Multi-scale HiPPO**: 单一 $N$ 可能不够, 像 Pathformer 那样 multi-scale HiPPO 可能更好。

---

## 九、Code & Reference Links

- [SCFormer 官方代码](https://github.com/ShiweiGuo1995/SCFormer)
- [HiPPO 原始 paper](https://arxiv.org/abs/2008.07669)
- [HiPPO 官方代码](https://github.com/HazyResearch/state-spaces)
- [Mamba (后续工作)](https://arxiv.org/abs/2312.00752)
- [S4](https://arxiv.org/abs/2111.00396)
- [iTransformer](https://arxiv.org/abs/2310.06625)
- [PatchTST](https://arxiv.org/abs/2211.14730)
- [Informer](https://arxiv.org/abs/2012.07436)
- [FEDformer](https://arxiv.org/abs/2201.12740)
- [Autoformer](https://arxiv.org/abs/2106.09107)
- [DLinear (Are Transformers Effective)](https://arxiv.org/abs/2205.13504)
- [TimesNet](https://arxiv.org/abs/2210.02186)
- [Crossformer](https://arxiv.org/abs/2108.02041)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Reformer](https://arxiv.org/abs/2001.04451)
- [RevIN (Instance Norm)](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Toeplitz Matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix)
- [Legendre Polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials)

---

## 十、总结: 这个工作为什么 Important

SCFormer 的 contribution 不仅是 SOTA numbers, 而是 **明确指出了 channel-wise Transformer 的两个被忽视的 inductive bias 问题**:

1. **Causality violation in linear projections** —— 这个观察非常 sharp, 之前 channel-wise Transformer 的工作都没注意到 Q/K/V/FFN 的 linear transformation 沿时间维度做时会 leak future information。

2. **Markov assumption limitation** —— 把 forecasting 重新 frame 成 "stateful emission" 而不是 "transition", 并用 HiPPO 给出 principled 的 memory mechanism。

这两个 insights 都是 transferable 的——你可以把它们用到其他 channel-wise Transformer 变体上, 甚至用到 vision Transformer (如果考虑 spatial causality)。

Andrej, 如果你要 build intuition, 我会建议你重点理解:
- HiPPO 的最优投影视角 (为什么这个递推是 optimal memory)
- Triangular matrix 和 multi-layer conv 的等价性 (Eq. 10 的归纳证明)
- Channel-wise attention 和 time-axis linear 的分工 (attention 抓 channel correlation, structured linear 抓 temporal pattern)

这三点 grasp 之后, 整个 paper 的 design choice 都变得 natural 了。
