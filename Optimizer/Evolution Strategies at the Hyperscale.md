---
source_pdf: Evolution Strategies at the Hyperscale.pdf
paper_sha256: 02d77e6be9353d5f650caaf771f488aa8aec8decc1f09dfb6db1528b302f4c9f
processed_at: '2026-08-18T11:36:45-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EGGROLL 用人话讲

Andrej, 我换个频道, 把这篇 paper 用更直觉的方式重新讲一遍。

---

## 1. 先说 ES 这个东西

Evolution Strategies (ES) 就是: 你有一个 model, 你想优化它, 但你不想算 gradient。怎么做?

你想成你站在一座山上, 蒙着眼睛, 想找山顶。你没有地图, 没有坡度信息。你能做什么? **往各个方向扔石头, 听石头落地的声音, 判断哪个方向更高**。

数学上就是:
- 当前参数 $\mu$ (你站着的位置)
- 随机生成一堆 perturbations $\mu + \sigma v_i$, $v_i$ 是随机方向
- 评估每个的 fitness $f(\mu + \sigma v_i)$ (听石头落地的声音)
- **fitness 高的方向, 往那边走; fitness 低的, 反方向走**

update rule 就这么简单:
$$\mu_{t+1} = \mu_t + \frac{\alpha}{N} \sum_{i=1}^N v_i \cdot f(\mu_t + \sigma v_i)$$

变量含义:
- $\mu_t$: step $t$ 的参数 (位置)
- $v_i$: 第 $i$ 个随机方向 (standard Gaussian)
- $\sigma$: 扰动幅度 (扔石头扔多远)
- $\alpha$: 学习率 (走多大步)
- $N$: population size (扔多少块石头)
- $f$: fitness function (落地声音有多响)

**为什么这个 idea 之前一直没火?** 因为 backpropagation 太强了。Backprop 给你精确的 gradient, ES 给你 noisy 的 gradient estimate。在大多数 deep learning 场景, 精确 gradient 完胜。

但 ES 有几个 backprop 做不到的事情:
1. **不需要 differentiability** — 你的 model 有 discrete 操作? 有 hard clipping? 有 non-differentiable reward? ES 全都不在乎, 因为它只 query function value, 不 query gradient
2. **不需要 backward pass** — 没有 vanishing gradient, 没有 exploding gradient, 没有 BPTT (backprop through time)
3. **Embarrassingly parallel** — 每个 population member 独立评估, 只需要通信 scalar fitness

Salimans et al. 2017 (OpenAI) 让 ES 在 RL 上火过一阵, 但后来被 PPO/SAC 这些 first-order methods 盖过去了。原因: **ES 在 scale 上太贵了**。

---

## 2. ES 为什么在 scale 上爆炸

现在你要 train 一个 billion-parameter 的 LLM。你的 weight matrices 是 $M \in \mathbb{R}^{m \times n}$, $m, n$ 都是几千几万。

ES 要做: 对每个 population member $i$, sample 一个 full-rank perturbation $E_i \in \mathbb{R}^{m \times n}$, 然后 compute forward pass:
$$u_i (M + \sigma E_i)^\top$$

这里的 $u_i$ 是 input activation。注意 $M + \sigma E_i$ 对每个 $i$ 都不同, 所以这变成一个 **batched matrix multiplication**, batch size = population size $N$。

问题在哪? **arithmetic intensity**。

arithmetic intensity = 算术操作数 / 内存访问字节数。GPU 的 peak performance 需要 ~300 ops/byte (H100 bfloat16)。低于这个, 你是 memory-bound, GPU 大部分时间在等数据从 HBM 搬进来。

naive ES 的 arithmetic intensity:
$$\frac{m}{2 + m} < 1$$

**永远小于 1**, 无论 $m$ 多大, 无论 batch size 多大。为什么? 因为 $E_i$ 的每个元素只参与一次乘法, 用完就扔。这和 sparse matrices 的问题一样: 你 fetch 了一个 element, 做一次 op, 然后这个 element 再也不用了。GPU 的 parallelism 和 caching 完全用不上。

paper 里算了一下: 要达到 300 ops/byte, 每个 perturbation 要被 **reuse 324 次**。这违背了 ES 的核心前提 (每个 member 用不同 perturbation)。

所以 Salimans 2017 只能 scale 到 1440 population, 只能 train small networks。

---

## 3. EGGROLL 的核心 trick: Low-rank Perturbations

### 3.1 Idea

作者说: 既然 full-rank $E_i$ 太贵, 我们 sample **low-rank** 的:
$$E_i = \frac{1}{\sqrt{r}} A_i B_i^\top$$

- $A_i \in \mathbb{R}^{m \times r}$: 瘦长矩阵
- $B_i \in \mathbb{R}^{n \times r}$: 瘦长矩阵  
- $r \ll \min(m, n)$: 通常 $r = 1$ 就够
- $\frac{1}{\sqrt{r}}$: 保证 variance 不随 $r$ 爆炸

存储从 $mn$ 降到 $(m+n)r$。对 $m = n = 8192$, $r = 1$: 从 67M 降到 16K, **4000 倍 reduction**。

这看起来像 LoRA (Hu et al., 2022), 但**用途完全不同**。LoRA 是在 backprop 里 freeze base model, 只 train adapter。EGGROLL 是在 ES 里用 low-rank 来 sample perturbation, 但**最终 update 是 full-rank 的** (因为是 $N$ 个 rank-$r$ 的 sum)。

### 3.2 为什么这个能加速: 关键的 forward pass 分解

这才是 paper 的核心 insight。forward pass 是:
$$u_i (M + \sigma E_i)^\top$$

naive ES: $M + \sigma E_i$ 对每个 $i$ 不同, 所以 batched matmul, intensity $< 1$。

EGGROLL: 代入 $E_i = \frac{1}{\sqrt{r}} A_i B_i^\top$:
$$u_i (M + \sigma E_i)^\top = u_i M^\top + \frac{\sigma}{\sqrt{r}} (u_i B_i) A_i^\top$$

看这个分解:
- $u_i M^\top$: **$M$ 对所有 population members 都一样**! 这是普通的 batched matmul, arithmetic intensity 高, GPU 喜欢这个
- $u_i B_i$: 当 $r=1$, 这是 batch of $N$ 个 vector-vector dot products, 极便宜
- 乘以 $A_i^\top$: batched scalar-vector multiplication, 也便宜

paper 算的 EGGROLL arithmetic intensity:
$$\frac{m + 2r + 1/2}{2 + r(2 + 2/m) + m/B}$$

当 $m = 8192$, $r = 1$, 要达到 300 ops/byte 需要 batch size **352** (vs naive ES 是 $\infty$)。

**人话**: EGGROLL 把 expensive 的部分 (base matmul $u M^\top$) share 给所有 population members, 只让 per-member 的 cheap 部分变 batched。这正好是 vLLM (Kwon et al., 2023) 做 batched LoRA inference 的 trick。所以 EGGROLL 的 throughput 接近纯 batched inference (91%)。

### 3.3 Counter-based RNG: 不存 perturbation

另一个 trick: 用 counter-based deterministic RNG (Salmon et al., 2011; JAX PRNG, Bradbury et al., 2018)。每个 worker 有个 seed, 需要时 reconstruct noise, 不需要把 $E_i$ 存在 memory 里。更新时 reconstruct 所有 $E_j$ 来算 $\sum_j E_j f_j$。

rank-1 时还有个 algebraic trick: 不显式 materialise $E_i$, 直接 reconstruct $A \in \mathbb{R}^{N \times d_{out}}$ 和 $B \in \mathbb{R}^{N \times d_{in}}$, 然后算 $(\text{diag}(f) A)^\top B$, 一个普通 matmul。

---

## 4. 理论: High-Dimensional ES 会 Linearise

这是 paper 最 elegant 的部分, 我尽量用人话讲。

### 4.1 Gaussian Annulus Problem

high dimension 下, standard Gaussian 的 probability mass 集中在半径 $\sqrt{d}$ 的 thin shell 上。直觉: 你在 1000 维空间采样一个点, 它大概率离原点 $\sqrt{1000} \approx 31$ 远, 不太可能离原点 1 远。

这意味着: 如果你的 perturbation 是 $\sigma v$, $v \sim \mathcal{N}(0, I_d)$, 那 $\sigma v$ 大概率离 $\mu$ 有 $\sigma \sqrt{d}$ 远。

如果你想让 perturbation 停留在 $\mu$ 附近的 smooth region (半径 $\rho$), 你需要 $\sigma \sqrt{d} < \rho$, 即 $\sigma = o(d^{-1/2})$。

### 4.2 三个 Regime

paper 用 cubic polynomial $f(x) = a^\top x + \frac{1}{2} x^\top B x + \frac{1}{6} C(x, x, x)$ 来分析, 给出三个 regime:

**Regime I (Linearisation)**: $\sigma = o(d^{-1/2})$
- perturbation 停在 local smooth ball 内
- ES update 收敛到 **first-order gradient** $\nabla f(\mu)$
- 收敛 rate: $\Theta((\sigma \sqrt{d})^\alpha)$, $\alpha$ 是 Hölder exponent

**人话**: 如果 noise 足够小, ES 就是 "noisy gradient descent"。high dimension 反而帮你, 因为 noise 集中在 $\mu$ 附近, 给你一个 local linear approximation。这和 Neural Tangent Kernel (NTK, Jacot et al., 2018) 的 linearisation theorem 是 parallel 的。

**Regime II (Critical)**: $\sigma \asymp d^{-1/2}$
- perturbation 刚好 reach smooth region 的边界
- 二阶项 ($B$) 由于 Gaussian symmetry 消失
- **三阶项 ($C$) 保留**! 
- update 保留 odd-order derivative 信息

**人话**: 在 critical scaling, ES 不是纯 gradient, 它还 "看到" 三阶信息。这其实挺有意思 — ES 在 high dim 下自动 filter 掉二阶项 (因为 Gaussian symmetry), 保留三阶。

**Regime III (Divergence)**: $\sigma \gg d^{-1/2}$
- perturbation 飞出 smooth region
- $\|\nabla_\mu J(\theta) - \nabla f(\mu)\| = \Theta(\sigma^2 d) \to \infty$
- 只要 objective 有 non-degenerate 三阶项, 就 diverge

**人话**: noise 太大, ES 完全跑偏, 在 random directions 上乱走。

### 4.3 EGGROLL 也 Linearise (Theorem 3)

即使 $r = 1$ (每个 perturbation rank-1), EGGROLL update 在 $d \to \infty$ 时也收敛到 true ES gradient, 也 linearise 到 first-order gradient。

收敛 rate:
$$\|\hat{g}_{LR} - \nabla f(\mu)\|_F = \mathcal{O}(L_d (\sigma d)^2) + \mathcal{O}\left(\frac{\sqrt{d}}{\sigma^2} \exp\left(-K \frac{\rho}{\sqrt{d}\sigma}\right)\right) = o(1)$$

变量:
- $L_d$: Hessian 的 Lipschitz constant
- $\rho$: local smooth ball 半径
- $K > 0$: constant

对 overparameterised networks (NTK regime), Hessian spectral norm 随 width 衰减, $L_d = \mathcal{O}(d^{-1/2})$ 或 $\mathcal{O}(d^{-1})$。代入得 $\mathcal{O}(\sigma^2 d^{3/2})$ 或 $\mathcal{O}(\sigma^2 d)$。

**人话**: 大 network 的 Hessian 天然小, 所以 low-rank perturbation 在大 network 上 behave 接近 full-rank Gaussian perturbation。这解释了为什么 $r = 1$ 在实验里 work 得很好。

### 4.4 Rank Convergence: 为什么 $r = 1$ 够 (Theorem 4)

$$\|\hat{g}_{LR}^r - \nabla_\mu J(\theta)\|_F = \mathcal{O}(r^{-1})$$

比一般 CLT 的 $\mathcal{O}(r^{-1/2})$ 快。为什么?

paper 用 **Edgeworth expansion** 展开 $P(E^r)$ 的分布:
$$\hat{p}(v^r) = g(v^r) + \frac{1}{4!r} g(v^r) \sum_{i,j,k,l} \kappa_{i,j,k,l}^4 H_{i,j,k,l}(v^r)$$

- $g(v^r)$: limiting Gaussian density
- $\kappa_{i,j,k,l}^4$: 4th order cumulants
- $H_{i,j,k,l}$: 4th order Hermite polynomial

odd-order cumulants 由于 distribution symmetry 全部为零 (Assumption 1)。所以 convergence 由 4th order term 控制, rate $\mathcal{O}(r^{-1})$。

**人话**: low-rank perturbation 的分布, 即使 $r=1$, 已经是 Gaussian 的 decent approximation。因为你的 perturbation elements 是 $a_i b_j$ 的和, CLT 让它快速 Gaussianise。对称性让 odd moments 消失, 只剩 even moments, 所以收敛比一般 CLT 快一倍。$r = 10$ 就几乎 converge 了。

---

## 5. 实验: 三个 Moonshot

### 5.1 EGG: Pure Integer Language Model

最 provocative 的实验。作者设计了一个叫 **EGG (Evolved Generative GRU)** 的 architecture:

**设计哲学** (Appendix G): 既然 ES 不需要 backprop, 我可以 design 一个对 inference 硬件最友好的 model, 完全不在乎 backprop 的需求。

具体:
- **纯 int8**: 所有 weights 和 activations 都是 int8, 永不 cast 到 float。H100 上 int8 tensor core 是最快的数据类型
- **Nonlinear RNN (minGRU variant)**: Transformer/SSM 不能做 state tracking (Merrill et al., 2024), 但经典 RNN 可以。ES 不需要 BPTT, 所以可以 train on unbounded sequence lengths
- **No activation functions**: 利用 int8 clipping 的 implicit nonlinearity (inspired by Foerster, 2017)。saturated addition 提供足够的 nonlinearity

**关键 operations**:
- Matrix multiplication: $u @ M := \mathbb{I}_8\left(\frac{uM}{16\sqrt{n}}\right)$, int32 accumulation + scaling by $16\sqrt{n}$
- LayerNorm: 用 mean absolute value 而非 RMS (sqrt 在整数上 expensive)
- GRU: sigmoid/tanh 设为 identity, 靠 clipped addition 提供 nonlinearity
- Fitness: log-likelihood via lookup tables (EXP2, LOG2)

**结果** (Fig 2b):
- 6L-256D EGG, character-level on minipile
- Best test loss: **3.40 bits/byte**
- vs 6L-256D Transformer with backprop SGD: **3.58 bits/byte**
- EGG **outperforms** Transformer with backprop!
- Population size 从 2 到 $2^{20} = 1,048,576$
- 最大 population 比 Salimans 2017 大 **三个数量级**, 但只需要 single GPU

**关键发现**: 
- Population size 2 (类似 MeZO, Malladi et al., 2023) **显著 underperform**, 说明大 population 对 pretraining 至关重要
- 大 population 需要约 180x more GPU-hours than backprop baseline, 展示 **compute-only scaling in limited data regimes**

**人话**: 这个实验说明 ES 可以直接 train int8 model from scratch, 这用 backprop 几乎不可能 (gradient of clipping, quantisation noise in backward pass, 需要 fake quantisation in QAT)。如果 int8 硬件比 fp16 快 4-16x (Horowitz, 2014), 且 ES 可以直接用 int8 train, 这暗示 **inference hardware 可能就是 training hardware**, 不需要单独设计训练芯片。

### 5.2 LLM Fine-tuning (RWKV-7)

RWKV-7 (Peng et al., 2025) 是 modern recurrent LM, constant state size 适合 large batch inference。

**Countdown task** (Fig 4b):
- RWKV-7 g1.5B, single GPU
- EGGROLL: 1024 parallel generations, 618 updates → **35%** accuracy
- GRPO: 64 parallel generations, 915 updates → **23%**

**GSM8K** (Fig 5a):
- RWKV-7 g7B, 8 GPUs
- EGGROLL: 8192 parallel generations (1024/GPU, 260 updates)
- GRPO: 256 parallel generations (32/GPU, 340 updates)
- EGGROLL outperforms GRPO

**14B model** (Fig 13b):
- 32 GPUs, 12 hours
- AIME24: 13% → **30%**
- AIME25: 7% → **33%**
- HMMT25: 11% → 13%
- **GRPO infeasible** because Adam optimizer memory for 14B is too much

**Scoring function** (类 GRPO 的 group relative advantage):
$$\bar{s}_i = \frac{1}{m} \sum_{j=1}^m \frac{s_{i,j} - \mu_{q_j}}{\bar{\sigma}}$$

- $s_{i,j}$: accuracy of noise direction $E_i$ on question $q_j$
- $\mu_{q_j}$: mean accuracy across all noise directions on $q_j$
- $\bar{\sigma}$: global variance (GRPO 用 per-question variance)
- 同一 batch 内所有 questions 对 population members 权重相同

**人话**: EGGROLL 在相同 hardware + wall-clock 下 outperform GRPO, 主要因为:
1. 更大 population (1024 vs 64 parallel generations per GPU) — ES 不需要 backprop, memory 全给 inference
2. 不需要 Adam optimizer memory — 14B 上 GRPO 都跑不动, EGGROLL 可以
3. 不需要 backprop — 可以 train 更长 sequence, int8 model, non-differentiable objectives

### 5.3 RL Tasks

16 个 environments, 3-layer 256-hidden MLP:
- EGGROLL competitive on **7/16**, underperform on 2/16, outperform on 7/16 vs OpenES
- Rank-1 perturbations
- 通常有 substantial wall-clock improvements

**人话**: 对小网络, OpenES 可以用。但 as network sizes increase, vanilla OpenES 变 infeasible (arithmetic intensity $< 1$)。EGGROLL 的 low-rank structure 让它 remain tractable。

### 5.4 Non-differentiable objectives: pass@k

EGGROLL 可以直接优化 pass@k (non-differentiable, depends on multiple samples)。Fig 10 显示 optimizing for pass@k 增加 answer diversity, 而 pass@1 reduces it (model collapses towards single answer)。这是 GRPO 的 known limitation (Yue et al., 2025)。

**人话**: 这是 ES 的天然优势 — 只要你能 evaluate objective, 你就能 optimize 它。pass@k 需要采样多次算成功率, backprop 根本无法 differentiate 这个, 但 ES 完全不在乎。

### 5.5 High-Frequency Trading

S5 time series model on LOBSTER data, optimize for PnL (profit and loss):
- Baseline (pretrained): mean PnL ~4,700
- EGGROLL fine-tuning: mean PnL ~**12,000** (155% improvement)
- 65,536 parallel generations, LoRA rank 4
- Rank-based fitness: $F_i = \frac{1}{2} - \frac{\text{rank}(\text{PnL}_i)}{M-1}$

**人话**: PnL 是极其 non-smooth, non-differentiable 的 reward (依赖 order matching, market impact, etc)。Backprop 在这里完全无能为力。ES 天然适合这种 black-box optimization。

---

## 6. 我的 Intuition 和联想

### 6.1 为什么 low-rank 在 ES 里 work, 而在 backprop 里只是 adapter?

在 backprop + LoRA, low-rank 是 **inductive bias** — 你假设 task adaptation 是 low-rank 的。如果 task 不是 low-rank, 你 lose performance。

在 ES, low-rank perturbation 是 **sampling trick**, 但 **population aggregate** 是 full-rank 的 ($\min(Nr, m, n)$)。Theorem 3, 4 证明即使每个 perturbation rank-1, 当 $d \to \infty$ 或 $r \to \infty$, update 收敛到 true ES gradient。每个个体 rank-deficient 不重要, **population averaging 带来 collective full-rank**。

这就像 democracy: 每个选民只懂一个方面 (rank-1), 但多数投票的结果可以 capture 全部信息 (full-rank), 只要选民够多且足够 diverse。

### 6.2 和 NTK 的深层 connection

Theorem 1 的 linearisation 和 NTK theorem (Jacot et al., 2018) 深度 analogous:
- **NTK**: 在 overparameterised regime, neural network 在参数空间 linearise, gradient descent 收敛到 global minimum
- **EGGROLL**: 在 high-dimensional regime, ES update linearise 到 first-order gradient

两者都依赖 dimension-independent scaling (Assumption 4) 和 Hessian 衰减。这暗示 **ES 和 GD 在 high-dim overparameterised regime 下 asymptotically equivalent**, 只是 ES 用 noise 估计 gradient, GD 用 chain rule 计算 gradient。

如果这个 intuition 对, 那 ES 的 "sample efficiency" 在 high-dim 下应该接近 GD, 只是 variance 更高 (需要大 population 来 reduce)。EGG 实验 (3.40 vs 3.58 bits/byte) 似乎支持这个 — ES 在足够大 population 下可以 match 甚至 beat backprop。

### 6.3 Arithmetic intensity 是 ES 的 fundamental bottleneck

naive ES 的 arithmetic intensity $< 1$ 是因为每个 perturbation element 只参与一次 multiplication。这和 sparsity / structured matrices 的问题类似。

EGGROLL 的 decomposition $u_i M^\top + \frac{\sigma}{\sqrt{r}}(u_i B_i) A_i^\top$ 把 **shared base matmul** (high intensity) 和 **per-perturbation correction** (low intensity but cheap) 分开。

这和 **FlashAttention** (Dao et al., 2022) 的 insight 类似: 重新结构化 computation 来 improve arithmetic intensity, 而非减少 FLOPs。EGGROLL 的 FLOPs 其实比 naive ES 多 (多了 $A_i, B_i$ 的操作), 但因为 high-intensity 部分 dominate, 整体更快。

### 6.4 Integer-only training 的 implications

EGG 实验是最 provocative 的。如果这条路线 scale:
- **训练硬件 = 推理硬件**: 不需要 separate training/inference silicon。int8 tensor cores 可能直接做 training
- **Energy efficiency**: int8 比 fp16/bf16 省 4-16x energy (Horowitz, 2014), 训练成本大幅下降
- **On-device training**: 手机/IoT 设备的 inference 芯片可能直接做 fine-tuning
- **Quantisation-aware training 过时**: 不需要 fake quantisation, 不需要 straight-through estimators, 直接在 int8 上 train

### 6.5 Population size as compute scaling

EGG 实验 (Fig 2b) 显示 population size 从 2 到 $2^{20}$ 的 scaling。这类似 "compute-only scaling" — 在 limited data regime, 用更多 population (而非更多 data) 提升 gradient estimate 的 signal-to-noise ratio。

这和 Chinchilla (Hoffmann et al., 2022) 的 data/compute tradeoff 是 **orthogonal dimension**。如果 data 是 bottleneck, EGGROLL 让你 trade data for compute (population size)。

### 6.6 和 GRPO 的关系

GRPO (Shao et al., 2024) 也是 population-based, 但用 PPO-style clipped objectives + backprop。EGGROLL 用 ES, 不需要 backprop。两者 scoring function 相似 (group relative advantage)。

EGGROLL outperform GRPO 的原因:
1. **更大 population**: ES 不需要 backprop, memory 全给 inference。1024 vs 64 parallel generations per GPU
2. **不需要 Adam optimizer**: 14B 上 GRPO 都跑不动, EGGROLL 可以
3. **不需要 backprop**: 更长 sequence, int8 model, non-differentiable objectives 都可以

### 6.7 Open questions / 联想

- **Second-order information**: ES 天然 estimate gradient, 但 Hessian 信息呢? CMA-ES (Hansen & Ostermeier, 2001) 估计 covariance, 但 scaling 到 billion params 是 open problem。EGGROLL 的 low-rank structure 可能 enable low-rank covariance estimation?
- **Non-Gaussian perturbations**: Theorem 6 用 Bessel functions 给出 non-Gaussian 的 score functions。Heavy-tailed perturbations (Lévy flights) 在 multi-modal landscapes 上可能 better, EGGROLL 的 framework 是否 extend?
- **混合 ES + GD**: 某些 layers 用 backprop (differentiable), 某些用 EGGROLL (non-differentiable, e.g. discrete tokens, structured modules)。Neurosymbolic systems (Sarker et al., 2021) 的 end-to-end training?
- **Inference-time harnesses**: Paper 提到可以 train LLMs to be aware of inference-time harnesses (chain-of-thought, tool use, multi-agent interaction)。EGGROLL 可以 optimize 整个 system (LLM + harness) end-to-end, 因为不需要 differentiate through harness
- **Curriculum learning**: ES 可以直接 optimize non-differentiable curriculum metrics (e.g. pass rate, coverage), 这对 backprop 很难
- **Adversarial training**: GAN-style adversarial 可以用 ES (generator + discriminator 都用 ES), 避免 mode collapse 的 gradient issues

### 6.8 最让我 excited 的方向

EGG (int8 language model) 这条路线如果 scale, 改变 AI hardware 的 design space。想象:
- H100/B100/R100 的 int8 tensor cores 直接做 training
- 训练成本降一个数量级 (energy + silicon area)
- 手机 NPU 可以 fine-tune 本地 model
- AI hardware 不再分 training/inference, 统一

这是 ES 一直 promise 但从未 deliver 的愿景。EGGROLL 第一次让这个 vision 看起来 technically feasible。

---

## 7. 总结

EGGROLL 的 elegance 在于:
1. **Simple idea**: low-rank perturbations, 像 LoRA 但用于 ES
2. **Rigorous theory**: 三个 regime (linearisation / critical / divergence), rank convergence $\mathcal{O}(r^{-1})$
3. **Practical impact**: 100x speedup, 91% of batch inference throughput, billion-parameter models
4. **Moonshot demos**: int8 RNN pretraining, 14B LLM fine-tuning, HFT agent, non-differentiable pass@k

**Reference links**:
- Paper: https://eshyperscale.github.io/
- Salimans et al., 2017 (ES as RL alternative): https://arxiv.org/abs/1703.03864
- Hu et al., 2022 (LoRA): https://arxiv.org/abs/2106.09685
- Jacot et al., 2018 (NTK): https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf
- Peng et al., 2025 (RWKV-7): https://arxiv.org/abs/2503.14456
- Malladi et al., 2023 (MeZO): https://arxiv.org/abs/2305.14314
- Kwon et al., 2023 (vLLM): https://arxiv.org/abs/2309.06180
- Hansen & Ostermeier, 2001 (CMA-ES): https://ieeexplore.ieee.org/document/6790628
- Horowitz, 2014 (computing energy): https://ieeexplore.ieee.org/document/6757323
- Vershynin, 2018 (High-Dimensional Probability): https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.pdf
- Dao et al., 2022 (FlashAttention): https://arxiv.org/abs/2205.14135
- Hoffmann et al., 2022 (Chinchilla): https://arxiv.org/abs/2203.15556
- Foerster, 2017 (Nonlinear computation in linear networks): https://blog.openai.com/nonlinear-computation-in-linear-networks/

如果你想 deep dive 任何 section (e.g. Edgeworth expansion 的推导, EGG architecture 的 int8 ops, distributed framework), 我可以展开。

---

# EGGROLL: Evolution Strategies at the Hyperscale — 深度讲解

Andrej, 这篇 paper 解决的核心问题是: **如何让 Evolution Strategies (ES) 在 billion-parameter models + million-sized populations 下变得 computationally feasible**。作者来自 Oxford FLAIR/WhiRL 和 MILA, 包括 Jakob Foerster 和 Shimon Whiteson 的组。

---

## 1. Motivation: 为什么 ES 在 scale 上失效

ES 的 attractiveness 在于三点:
1. **不需要 differentiability** — 可以优化 discrete parametrisations (cellular automata)、non-differentiable objectives (outcome-only rewards in LLM fine-tuning)
2. **Robust to noisy/ill-conditioned landscapes** — population-based exploration smooths irregularities, tolerates discontinuities, mitigates vanishing/exploding gradients
3. **Highly parallelisable** — fitness evaluations 独立, 只通信 scalar fitnesses, maps cleanly 到 inference infrastructure

但是 naive ES 在 GPU 上 scaling 到 billion-parameter models 时遇到 **arithmetic intensity bottleneck**。问题出在: 对每个 population member, 你需要 sample 一个 full-rank matrix perturbation $E \in \mathbb{R}^{m \times n}$, 然后 compute $u_i (M + \sigma E_i)^\top$。这变成一个 **batched matrix multiplication where every element of $M + \sigma E_i$ is used in only a single multiplication**, yielding 极低的 arithmetic intensity (ratio of arithmetic ops to memory traffic)。

**关键 insight**: arithmetic intensity of Gaussian matrix ES 永远 $< 1$, 无论 batch size 多大:
$$\frac{m}{2 + m} < 1$$
对于 H100 roofline (~300 ops/byte for bfloat16), 意味着每个 perturbation需要被 reuse 至少 ~324 次才能 saturate compute, 这违背了 ES 的核心优势 (每个 member 用不同 perturbation)。

---

## 2. EGGROLL 的核心 idea: Low-rank Perturbations

### 2.1 从 full-rank 到 low-rank

类比 LoRA (Hu et al., 2022), EGGROLL 不 sample full-rank $E \in \mathbb{R}^{m \times n}$, 而是 sample:
$$E = \frac{1}{\sqrt{r}} A B^\top$$

其中:
- $A \in \mathbb{R}^{m \times r}$, column matrix
- $B \in \mathbb{R}^{n \times r}$, column matrix  
- $r \ll \min(m, n)$ 是 rank
- $\frac{1}{\sqrt{r}}$ scaling 确保 $E$ 的 variance 对所有 $r$ 保持 bounded

**内存**: 从 $mn$ 降到 $(m+n)r$ per layer
**Tensor movement**: 比例降低

### 2.2 EGGROLL Update 公式

对每个 worker $i$ (并行):
1. Sample $A_{i,t} \sim p(A)$, $B_{i,t} \sim p(B)$
2. Form $E_{i,t} = \frac{1}{\sqrt{r}} A_{i,t} B_{i,t}^\top$
3. Evaluate fitness $f(W = M_t + \sigma E_{i,t})$
4. Update:
$$M_{t+1} \gets M_t + \frac{\alpha_t}{N_{\text{workers}}} \sum_{i=1}^{N_{\text{workers}}} E_{i,t} \cdot f(W = M_t + \sigma E_{i,t})$$

变量含义:
- $M_t \in \mathbb{R}^{m \times n}$: mean matrix parameters at step $t$
- $\alpha_t$: learning rate (absorbs $\frac{1}{\sigma}$)
- $N_{\text{workers}}$: population size
- $\sigma$: noise scale (hyperparameter)

**关键**: 每个 $E_{i,t}$ 是 rank-$r$, 但 overall update 是 $N_{\text{workers}}$ 个 rank-$r$ matrices 的 weighted sum, 所以 update rank = $\min(Nr, m, n)$。在所有实验中 $Nr > \min(m,n)$, 即 **EGGROLL parameter updates are full-rank**!

### 2.3 为什么这能加速: Arithmetic Intensity 的分解

关键 trick 是 forward pass 的分解:
$$u_i (M + \sigma E_i)^\top = u_i M^\top + \frac{\sigma}{\sqrt{r}} (u_i B_i) A_i^\top$$

- $u_i M^\top$: regular matrix multiplication, **shared across all population members** (since $M$ is constant), 高 arithmetic intensity
- $u_i B_i$: 当 $r=1$, 这是一个 batch of $N$ vector-vector dot products of length $d_{in}$, 得到 $N$ 个 scalars
- 再乘以 $A_i^\top$: batched scalar-vector multiplication

这正是 vLLM (Kwon et al., 2023) 的 batched LoRA inference 用的 trick。EGGROLL 因此能达到和 batched LoRA inference 一样的速度。

**Arithmetic intensity of EGGROLL** (bfloat16, $d_{out} = d_{in} = m$, rank $r$):
$$\frac{m + 2r + 1/2}{2 + r(2 + 2/m) + m/B}$$

当 $m = 8192, r = 1$ 时, 要达到 300 ops/byte 需要 batch size **352** (vs 324 for standard inference, vs ∞ for naive ES)。**EGGROLL 可以用 unique perturbations per input 饱和 compute, 这是 naive ES 做不到的。**

### 2.4 Counter-based Deterministic RNG

为了不把 perturbation matrices 存在 memory 中, EGGROLL 用 counter-based RNG (Salmon et al., 2011; JAX's PRNG, Bradbury et al., 2018) 从 known seeds $\varsigma$ 重构 noise on demand。更新时 reconstruct 所有 $E_j$ 用于计算 $\sum_{j=1}^N E_j f_j$。

---

## 3. 理论分析: High-Dimensional Convergence

这是 paper 最 elegant 的部分。作者分析 Gaussian ES 在 $d \to \infty$ 时的行为。

### 3.1 Gaussian Annulus Problem

在 high dimensions, standard Gaussian 的 probability mass 集中在半径 $\sqrt{d}$ 的 thin shell 上。这意味着 perturbations $\sigma v$ 会把参数推离 $\mu$ 太远, 除非 $\sigma$ 随 $d$ 衰减。

### 3.2 Theorem 1: Convergence to Linearity

**假设** $\sigma_d = o(d^{-1/2})$ (即 $\sigma$ 衰减快于 $d^{-1/2}$), fitness $f$ 满足:
- **Locally $C^1$-continuous** with $\alpha$-Hölder continuous gradient (Assumption 2)
- **Polynomial growth** globally (Assumption 3)
- **Bounded derivative** $\|\nabla f(\mu)\| = \mathcal{O}(1)$ (Assumption 4, NTK-style scaling)

则:
$$\|\nabla_\mu J(\theta) - \nabla f(\mu)\| = \Theta((\sigma_d \sqrt{d})^\alpha) = o(1)$$

**Intuition**: 在 high dimensions, 如果 $\sigma$ 足够小, Gaussian perturbations 集中在 $\mu$ 附近的 local ball 内, ES update linearises 到 first-order gradient $\nabla f(\mu)$。这和 Neural Tangent Kernel (NTK, Jacot et al., 2018) 的 linearisation theorem 类似, 但适用于更广的 objective class (包括 discontinuous architectures)。

### 3.3 三个 Regimes

用 cubic polynomial $f(x) = a^\top x + \frac{1}{2} x^\top B x + \frac{1}{6} C(x,x,x)$ 来分析:

**Regime I (Linearisation)**: $\sigma_d = o(d^{-1/2})$
- ES 收敛到 $\nabla f(\mu)$
- 收敛 rate $\Theta((\sigma_d \sqrt{d})^\alpha)$, tight

**Regime II (Critical)**: $\sigma_d \asymp d^{-1/2}$
- 二阶项 ($B$) 由于 symmetry 消失
- **三阶项 ($C$) 保留**! 
- $\|\frac{\sigma_d^2}{2} \mathbb{E}[C(v,v,\cdot)]\| = \Theta(1)$
- High-dimensional update 保留 odd-order derivatives

**Regime III (Divergence)**: $d^{-1/2} = o(\sigma_d)$
- $\|\nabla_\mu J(\theta) - \nabla f(\mu)\| = \Theta(\sigma_d^2 d) \to \infty$
- 只要 cubic tensor 有 non-vanishing Gaussian contraction (non-degenerate case), 就 diverge

**Intuition**: 这给出了 ES 在 high dimensions 下稳定优化的 **necessary and sufficient** 条件。$\sigma$ 必须衰减得足够快 ($> d^{-1/2}$ rate), 否则 perturbations 会 explore 超出 local smooth region。

### 3.4 Theorem 3: EGGROLL 也 Linearise

即使 fixed low-rank (包括 $r=1$), 在额外假设下:
- **$C^2$ local continuity** with Lipschitz Hessian (Assumption 5)
- **Sub-Gaussian tails** for $A, B$ elements (Assumption 6)

EGGROLL update $\hat{g}_{LR}$ 也收敛:
$$\|\hat{g}_{LR} - \nabla_W f(W=M)\|_F = \mathcal{O}(L_d (\sigma_d d)^2) + \mathcal{O}\left(\frac{\sqrt{d}}{\sigma_d^2} \exp\left(-K \frac{\rho}{\sqrt{d}\sigma_d}\right)\right) = o(1)$$

$$\|\hat{g}_{LR} - \nabla_M J(\theta)\|_F = \mathcal{O}(\sigma_d \sqrt{d} \cdot (1 + L_d \sigma_d d^{3/2})) = o(1)$$

变量:
- $L_d$: Hessian 的 Lipschitz constant, 对 overparameterised networks 典型衰减 $d^{-1/2}$ 或 $d^{-1}$
- $\rho$: local ball 半径
- $K > 0$: constant

**关键 insight**: 对于 standard parametrisations (NTK regime), Hessian spectral norm 随 width polynomial衰减, 所以 $L_d = o(1)$, 收敛 rate 变成 $\mathcal{O}(\sigma_d^2 d^{3/2})$ 或 $\mathcal{O}(\sigma_d^2 d)$。这解释了为什么 rank-1 EGGROLL 在 high-dimensional neural networks 上 work!

### 3.5 Theorem 4: Rank Convergence — 为什么低 rank 也够

$$\|\hat{g}_{LR}^r - \nabla_\mu J(\theta)\|_F = \mathcal{O}(r^{-1})$$

这比一般 CLT 的 $\mathcal{O}(r^{-1/2})$ **快**! 原因是:

用 **Edgeworth expansion** (Bhattacharya & Ranga Rao, 1976) 展开 $P(E^r)$ 的分布:
$$\hat{p}(v^r) = g(v^r) + \frac{1}{4!r} g(v^r) \sum_{i,j,k,l} \kappa_{i,j,k,l}^4 H_{i,j,k,l}(v^r)$$

- $g(v^r)$: limiting Gaussian density
- $\kappa_{i,j,k,l}^4$: 4th order cumulants
- $H_{i,j,k,l}$: 4th order Hermite polynomial
- Odd-order cumulants 由于 symmetry 全部为零 (Assumption 1: zero-mean symmetric distributions)

所以 convergence 由 4th order term 控制, rate 是 $\mathcal{O}(r^{-1})$ 而非 $\mathcal{O}(r^{-1/2})$。

**Intuition**: 即使 $r=1$, low-rank perturbation $E = ab^\top$ 的 marginal distribution 已经是 Gaussian 的 decent approximation; $r=10$ 时 nearly converged; $r=50$ 时 visually indistinguishable from limit。这解释了为什么 experiments 里 $r=1$ 就 work 得很好。

---

## 4. Score Function 近似

EGGROLL 的 score function 用 Gaussian 近似 $\hat{S}(E) = -E$ (即 Gaussian $\mathcal{N}(0, I_m, I_n)$ 的 score)。这在理论上有两个 justification:

1. **CLT**: $AB^\top = \sum_{i=1}^r a_i b_i^\top$ 是独立零均值 outer products 的和, 当 $r \to \infty$ 时 CLT 保证收敛到 Gaussian
2. **High-dim linearisation**: Theorem 3 证明即使 fixed $r$, 在 $d \to \infty$ 时 EGGROLL update 和用 Gaussian score 的 update 都收敛到同一个 linearised form

Paper 还在 Appendix D.1 推导了 mean-field approximators, 用 Bessel functions $K_n$ 表示, 但 experiments 显示 Gaussian approximator 最好。

---

## 5. 实验: 三个重要 domain

### 5.1 EGG: Pure Integer Language Model Pretraining

这是最 "moonshot" 的实验。作者设计了一个叫 **EGG (Evolved Generative GRU)** 的 architecture:
- **Pure int8**: 所有 weights 和 activations 都是 int8, 永不 cast 到 float
- **Nonlinear RNN**: modified minGRU, 可以做 state tracking (vs Transformers/SSMs 的 limitation, Merrill et al., 2024)
- **No activation functions**: 利用 int8 clipping 的 implicit nonlinearity (inspired by Foerster, 2017)

**关键设计 choice** (Appendix G):
- Matrix multiplication: $u @ M := \mathbb{I}_8\left(\frac{uM}{16\sqrt{n}}\right)$, 用 int32 accumulation + scaling
- LayerNorm: 用 mean absolute value (not RMS, because sqrt expensive on integers)
- GRU: sigmoid/tanh 设为 identity, 依靠 clipped addition 的 nonlinearity
- Fitness: log-likelihood via lookup tables (EXP2, LOG2)

**结果** (Fig 2b):
- 6L-256D EGG, character-level on minipile
- Best test loss: **3.40 bits/byte**
- vs 6L-256D Transformer with backprop SGD: **3.58 bits/byte**
- Population size 从 2 到 $2^{20} = 1,048,576$ (3 orders of magnitude larger than Salimans et al., 2017)
- **大 population size 至关重要**: population size 2 (类似 MeZO, Malladi et al., 2023) 显著 underperform
- 最大 population 需要约 180x more GPU-hours than backprop baseline, 展示 **compute-only scaling in limited data regimes**

**Intuition**: ES 不需要 backprop through time, 可以 train on unbounded sequence lengths。Integer-only training 对 backprop 是 nightmare (gradients of clipping, quantisation), 但 ES 天然 compatible。

### 5.2 RL Tasks

16 个 environments (Navix, Craftax, Brax, Kinetix, Jumanji), 3-layer 256-hidden MLP policy:
- EGGROLL competitive on **7/16**, underperform on 2/16, outperform on 7/16 vs OpenES
- Rank-1 perturbations
- 通常有 substantial wall-clock improvements

**Intuition**: 对小网络, OpenES 可以用, 但 as network sizes increase, vanilla OpenES 变 infeasible。EGGROLL 的低 rank structure 让它 remain tractable。

### 5.3 LLM Fine-tuning (RWKV-7)

RWKV-7 (Peng et al., 2025) 是 modern recurrent LMs, constant state size enables large batch inference。

**Countdown task** (Fig 4b):
- RWKV-7 g1.5B, single GPU
- EGGROLL: 1024 parallel generations, 618 updates → **35%** validation accuracy
- GRPO: 64 parallel generations, 915 updates → **23%**

**GSM8K** (Fig 5a):
- RWKV-7 g7B, 8 GPUs
- EGGROLL: 8192 parallel generations (1024/GPU, 260 updates)
- GRPO: 256 parallel generations (32/GPU, 340 updates)
- EGGROLL outperforms GRPO

**7B model on DeepScaleR** (Fig 5b):
- 128 GPUs, 24 hours
- AIME24: improvements
- Outperforms GRPO

**14B model** (Fig 13b):
- 32 GPUs, 12 hours
- AIME24: 13% → **30%**
- AIME25: 7% → **33%**
- HMMT25: 11% → 13%
- GRPO infeasible due to Adam optimizer memory

**Scoring function** (类 GRPO):
$$\bar{s}_i = \frac{1}{m} \sum_{j=1}^m z_{i,q_j} = \frac{1}{m} \sum_{j=1}^m \frac{s_{i,j} - \mu_{q_j}}{\bar{\sigma}}$$

- $s_{i,j}$: accuracy of noise direction $E_i$ on question $q_j$
- $\mu_{q_j}$: mean accuracy across all noise directions on $q_j$
- $\bar{\sigma}$: global variance (vs GRPO 的 per-question variance)
- 同一 batch 内所有 questions 对 population members 权重相同

### 5.4 Integer Quantised LLMs (int8 RWKV-7)

Quantisation (Appendix K.1): per-channel absmax, $Q_{i,j} = \text{clip}(\text{round}(W_{i,j}/s_i), -127, 127)$

**EGGROLL + Adam integration**:
- Adam 给 real-valued proposal $u \in \mathbb{R}^{m \times n}$ (bf16)
- Discretise: $z = \frac{u - \mu(u)}{\sigma(u) + 10^{-8}}$, then $\Delta = \text{sign}(z) \cdot \mathbb{1}\{|z| \geq \tau\} \in \{-1, 0, +1\}^{m \times n}$
- Update: $Q \gets \text{clip}(Q + \Delta, -127, 127)$

**Distillation fitness**: 
$$f_{\mu_i}(x_{1:T}) = \sum_{t=1}^T \text{KL}(p_t \| q_t(\cdot; \mu_i))$$

- $p_t$: non-quantised model distribution
- $q_t$: quantised model distribution
- KL divergence at each token

**结果** (Fig 6): progressively recovers ability to solve GSM8K subset。

### 5.5 非 differentiable objectives: pass@k

EGGROLL 可以直接优化 pass@k (non-differentiable, depends on multiple samples)。Fig 10 显示 optimizing for pass@k 增加 answer diversity, 而 pass@1 reduces it (model collapses towards single answer)。这是 GRPO 的 known limitation (Yue et al., 2025)。

### 5.6 High-Frequency Trading (Appendix M)

Fine-tune S5 time series foundation model on LOBSTER data, optimize for PnL:
- Baseline (pretrained): mean PnL ~4,700
- EGGROLL fine-tuning: mean PnL ~12,000 (**155% improvement**)
- 65,536 parallel generations, LoRA rank 4
- Rank-based fitness: $F_i = \frac{1}{2} - \frac{\text{rank}(\text{PnL}_i)}{M-1} \in [-0.5, 0.5]$

---

## 6. Distributed Framework (Appendix J)

### 6.1 Base-3 Fitness Packing

Antithetic pairs 产生 $\{+1, 0, -1\}$ (ternary)。5 个值打包成 1 byte:
$$\text{byte} = \sum_{i=0}^4 v_i \cdot 3^i$$

- Effective bitrate: 1.6 bits/value (near $\log 3 \approx 1.585$ theoretical limit)
- Network payload: $52 + \text{chunk\_size}/10$ bytes
- **Bandwidth 独立于 model size**

### 6.2 Coordinator-Worker Topology

- Coordinator: global state, assigns population chunks
- Workers: GPU fitness evaluation, signal shaping (chunk mean filtering, adaptive thresholding), return packed ternary fitness

---

## 7. 我的 Intuition 和联想

### 7.1 为什么 low-rank 在 ES 里 work, 而在 backprop 里只是 adapter?

在 backprop + LoRA, low-rank 是 **inductive bias** — 你假设 task adaptation 是 low-rank 的。如果 task 不是 low-rank, 你 lose performance。

在 ES, low-rank perturbation 是 **sampling trick**, 但 **population aggregate** 是 full-rank ($\min(Nr, m, n)$)。Theorem 3, 4 证明即使每个 perturbation rank-1, 当 $d \to \infty$ 或 $r \to \infty$, update 收敛到 true ES gradient。这是 population averaging 带来的 **collective full-rank**, 每个个体 rank-deficient 不重要。

### 7.2 和 NTK 的 connection

Theorem 1 的 linearisation 和 NTK theorem (Jacot et al., 2018) 深度 analogous:
- NTK: 在 overparameterised regime, neural network 在参数空间 linearise, gradient descent 收敛到 global minimum
- EGGROLL: 在 high-dimensional regime, ES update linearise 到 first-order gradient

两者都依赖 **dimension-independent scaling** (Assumption 4) 和 **Hessian 衰减**。这暗示 ES 和 GD 在 high-dim overparameterised regime 下是 **asymptotically equivalent** 的, 只是 ES 用 noise 估计 gradient, GD 用 chain rule 计算 gradient。

### 7.3 Arithmetic intensity 是 ES 的 fundamental bottleneck

Naive ES 的 arithmetic intensity $<1$ 是因为: 每个 perturbation element 只参与一次 multiplication。这和 sparsity / structured matrices 的问题类似。EGGROLL 的 decomposition $u_i M^\top + \frac{\sigma}{\sqrt{r}}(u_i B_i) A_i^\top$ 把 **shared base matmul** (high intensity) 和 **per-perturbation correction** (low intensity but cheap) 分开。

这和 FlashAttention 的 insight 类似: 重新结构化 computation 来 improve arithmetic intensity, 而非减少 FLOPs。

### 7.4 Integer-only training 的 implications

EGG 实验是最 provocative 的。Backprop 在 int8 上是 disaster (gradient clipping, quantisation noise in backward pass, 需要 fake quantisation in QAT)。ES 天然 bypass: fitness evaluation 用 inference datatypes, gradient 不需要。

如果 int8 硬件比 fp16/bf16 快 4-16x (Horowitz, 2014), 且 ES 可以直接用 int8 train, 这暗示 **未来 inference hardware 可能就是 training hardware**。No separate training/inference silicon。

### 7.5 Population size as compute scaling

EGG 实验 (Fig 2b) 显示 population size 从 2 到 $2^{20}$ 的 scaling。这类似 "compute-only scaling" — 在 limited data regime, 用更多 population (而非更多 data) 提升 signal-to-noise of gradient estimate。

这和 Chinchilla 的 data/compute tradeoff 是 orthogonal dimension。如果 data 是 bottleneck, EGGROLL 让你 trade data for compute (population size)。

### 7.6 和 GRPO 的关系

GRPO (Shao et al., 2024) 也是 population-based, 但用 PPO-style clipped objectives + backprop。EGGROLL 用 ES, 不需要 backprop。两者 scoring function 相似 (group relative advantage)。在相同 hardware + wall-clock 下 EGGROLL outperform GRPO, 主要因为:
1. 更大 population (1024 vs 64 parallel generations per GPU)
2. 不需要 Adam optimizer memory (14B model 上 GRPO infeasible)
3. 不需要 backprop (更长 sequences, int8, etc.)

### 7.7 Open questions / 联想

- **Second-order information**: ES 天然 estimate gradient, 但 Hessian 信息呢? CMA-ES (Hansen & Ostermeier, 2001) 估计 covariance, 但 scaling 到 billion params 是 open problem。EGGROLL 的 low-rank structure 可能 enable low-rank covariance estimation?
- **Non-Gaussian perturbations**: Theorem 6 用 Bessel functions 给出 non-Gaussian 的 score functions。Heavy-tailed perturbations (Lévy flights) 在 multi-modal landscapes 上可能 better, EGGROLL 的 framework 是否 extend?
- **混合 ES + GD**: 某些 layers 用 backprop (differentiable), 某些用 EGGROLL (non-differentiable, e.g. discrete tokens, structured modules)。Neurosymbolic systems (Sarker et al., 2021) 的 end-to-end training?
- **Inference-time harnesses**: Paper 提到可以 train LLMs to be aware of inference-time harnesses (chain-of-thought, tool use, multi-agent interaction)。EGGROLL 可以 optimize 整个 system (LLM + harness) end-to-end, 因为不需要 differentiate through harness。

---

## 8. 总结

EGGROLL 的 elegance 在于:
1. **Simple idea**: low-rank perturbations, 像 LoRA 但用于 ES
2. **Rigorous theory**: 三个 regime (linearisation / critical / divergence), rank convergence $\mathcal{O}(r^{-1})$
3. **Practical impact**: 100x speedup, 91% of batch inference throughput, billion-parameter models
4. **Moonshot demos**: int8 RNN pretraining, 14B LLM fine-tuning, HFT agent, non-differentiable pass@k

最让我 excited 的是 EGG (int8 language model) — 如果这条路线 scale, 训练硬件和推理硬件的边界会消失, int8 tensor cores 可能直接做 training。这改变 AI hardware 的 design space。

**Reference links**:
- Paper: https://eshyperscale.github.io/
- Salimans et al., 2017 (ES as RL alternative): https://arxiv.org/abs/1703.03864
- Hu et al., 2022 (LoRA): https://arxiv.org/abs/2106.09685
- Jacot et al., 2018 (NTK): https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf
- Peng et al., 2025 (RWKV-7): https://arxiv.org/abs/2503.14456
- Malladi et al., 2023 (MeZO): https://arxiv.org/abs/2305.14314
- Kwon et al., 2023 (vLLM): https://arxiv.org/abs/2309.06180
- Hansen & Ostermeier, 2001 (CMA-ES): https://ieeexplore.ieee.org/document/6790628
- Horowitz, 2014 (computing energy): https://ieeexplore.ieee.org/document/6757323
- Vershynin, 2018 (High-Dimensional Probability): https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.pdf
- Bhattacharya & Ranga Rao, 1976 (Edgeworth expansion)
- Foerster, 2017 (Nonlinear computation in linear networks): https://blog.openai.com/nonlinear-computation-in-linear-networks/

如果你想 deep dive 任何 section (e.g. Edgeworth expansion 的推导, EGG architecture 的 int8 ops, distributed framework), 我可以展开。
