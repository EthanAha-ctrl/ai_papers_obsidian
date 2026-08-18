---
source_pdf: DeepLoop Depth Scaling for Looped Transformers.pdf
paper_sha256: cadcbb862dc42afc2a0c3e7d2b9b6dff27f8b058dd207dcf264c76c072082fab
processed_at: '2026-08-18T04:49:20-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepLoop 的人话版

Andrej，我换个角度，把那篇 paper 用更直觉、更"讲给同事听"的方式重新过一遍。技术细节上一轮已经覆盖了，这一轮重点放在 **为什么这件事听起来理所当然但其实很微妙**，以及 **paper 背后那些没明说但你想知道的事**。

---

## 1. 一句话故事

Looped transformer 是那种"用 K 个 block 循环跑 R 轮"的模型，省参数但 effective depth 大。问题来了——DeepNorm 给 standard deep transformer 设计的 residual scaling 公式，在 looped 模型上**不够保守**。这篇 paper 证明：weight sharing 让同一组参数被反复读写，扰动放大效应比 standard 多一个 $R$ factor。修复就是一行：把 exponent 从 $1/4$ 改成 $1/2$。Empirical 上确实更好，而且 reasoning model (HRM) 也吃这一套。

就这么简单。

Reference: DeepLoop repo: https://github.com/lszshu/DeepLoop

---

## 2. 为什么 DeepNorm 在 untied 模型上能 work

要理解 DeepLoop 为什么要改 exponent，得先承认 DeepNorm 原来那套 perturbation argument 真的挺 elegant。我用一个粗略的类比：

想象 residual stream 是一条大河，每个 sublayer 是一个支流往里灌水。DeepNorm 关心的问题是：如果某个支流的入水量稍微变了一点（因为参数 update 了），下游出水量会变多少？

在 untied transformer 里，每个支流（sublayer）有自己独立的水闸（参数），所以"扰动传播"分析很干净——M 个支流各自独立贡献一个 perturbation 项，sum 起来给 $M \cdot (\beta/\alpha)^2$。设这个 $= O(1)$ 就 stable。

具体公式重述一遍：
- $\alpha$ = 主河道流量放大系数
- $\beta$ = 支流闸门初始开口大小
- $\beta/\alpha$ = 支流相对主河道的"扰动比例"
- $M = 2N$ = 支流总数（每 block 含 attention + FFN，所以 $2N$）

DeepNorm 选 $\alpha = (2N)^{1/4}$, $\beta = (8N)^{-1/4}$，让 $M(\beta/\alpha)^2 = 1/2$，恰好打平。

关键 insight 在 Lemma A.1：RMSNorm 在 $\alpha$ 大的时候近似 identity，但 residual branch 的扰动通过 $1/\alpha$ 因子进入 normalized direction。所以 $\beta$（缩 branch matrix 的 init gain）和 $\alpha$（缩 skip connection）需要一起调整——单独看 $\beta$ 没意义，要看 ratio $\beta/\alpha$。

Reference: DeepNorm paper: https://arxiv.org/abs/2203.06523

---

## 3. Weight sharing 把这套分析搞砸在哪

Looped transformer 的核心 trick 是：$K$ 个物理 block 被复用 $R$ 轮，参数 $\phi_j$ 被同一组，但被 unroll 成 $N = KR$ 层。问题来了——同一个支流的水闸被 R 次访问。

在 untied 模型里，第 $\ell$ 个支流的参数 update 只影响第 $\ell$ 个支流。扰动公式长这样：

$$
\Delta F = -\eta \sum_{i=1}^{M} U_i G_i + O(\eta^2)
$$

$U_i$ 是 visit $i$ 的 sensitivity operator，$G_i$ 是 visit $i$ 的 effective update，$M$ 项 sum，结束。

**Looped 模型里同一个 $\phi_j$ 被 R 次 visit 写入（gradient 累加）**：

$$
\delta \phi_j \propto \sum_{r=1}^{R} G_{r,j}
$$

**然后这个被累加过的 update 又被同样的 R 次 visit 读出（sensitivity 累加）**：

$$
\Delta F_{\text{tied}} = -\eta \sum_{j=1}^{J} \left(\sum_{r=1}^{R} U_{r,j}\right) \left(\sum_{t=1}^{R} G_{t,j}\right) + O(\eta^2)
$$

这就是 paper 的 Eq. (3.6)。注意这个 double sum 展开**有 $R^2$ 项**，其中 $R$ 项是 "self term"（$r = t$），剩下 $R(R-1)$ 项是 cross terms。**Cross terms 就是 tied-depth 额外引入的扰动来源**。

直觉上：在 untied 模型里，"参数变了"和"读取这个参数的 sensitivity"是 1 对 1 的。在 tied 模型里，"参数变了"是被 R 次 gradient 写出来的合体版本，然后这个合体版本同时被 R 次 sensitivity 读——cross terms 不可避免。

---

## 4. $\kappa_R$ 到底在测什么

Paper 定义 visit-alignment coefficient：

$$
\kappa_R := \max_j \frac{\|\sum_{r=1}^{R} U_{r,j}\| \cdot \|\sum_{t=1}^{R} G_{t,j}\|}{R \cdot C_U C_G (\beta/\alpha)^2}
$$

变量含义：
- 分子：实际 sum 范数的乘积
- 分母：如果 R 个 visit **完全 decorrelate** 时的 baseline（每个 visit 独立贡献，sum 范数 $\sim \sqrt{R}$，乘积 $\sim R$）
- $\kappa_R$ = 实际 relative to decorrelate baseline 的放大倍数

Triangle inequality 给 $0 \leq \kappa_R \leq R$。两个极端：

**Decorrelated** ($\kappa_R = O(1)$): R 次 visit 的 gradient 各自方向随机，sum 范数 $\sim \sqrt{R}$，分子 $\sim R$，$\kappa_R \approx 1$。模型行为退化到 untied，DeepNorm 的 $p=1/4$ 够用。

**Fully aligned** ($\kappa_R = \Theta(R)$): R 次 visit 的 gradient 方向高度一致，sum 范数 $\sim R$，分子 $\sim R^2$，$\kappa_R \sim R$。扰动比 untied 多一个 $R$ factor，必须更保守。

类比：想象 R 个工人一起推一辆车。Decorrelated 是他们朝不同方向推，合力 $\sim \sqrt{R}$。Fully aligned 是他们朝同一方向推，合力 $\sim R$。**Weight sharing 的目的就是让多次 visit 实现同一种 operation——也就是强迫 alignment**。所以 looped model 的实际工作 regime 偏向 aligned 这一头。

---

## 5. Exponent 从 1/4 到 1/2 怎么来的

考虑 scaling family $\alpha = (cN)^p$, $\beta = (dN)^{-p}$，所以 $\beta/\alpha = (cd)^{-p} N^{-2p}$。

设 $\kappa_R = \Theta(R^\gamma)$, $\gamma \in [0, 1]$，且 fixed physical depth $K$，让 $R = N/K \to \infty$。代入 stability condition $M \kappa_R (\beta/\alpha)^2 = O(1)$：

$$
2N \cdot R^\gamma \cdot (cd)^{-2p} N^{-4p} = \Theta(N^{1 + \gamma - 4p})
$$

要 bounded 需要 $1 + \gamma - 4p \leq 0$，即

$$
p \geq \frac{1 + \gamma}{4}
$$

| Regime | $\gamma$ | Threshold $p$ |
|---|---|---|
| Decorrelated | 0 | 1/4（DeepNorm）|
| Fully aligned | 1 | 1/2（DeepLoop）|

DeepLoop 选

$$
\alpha = (2N)^{1/2}, \quad \beta = (8N)^{-1/2}
$$

保留了 DeepNorm 的常数 $(c, d) = (2, 8)$，只改 exponent。验证一下 worst-case aligned bound（$K$ 固定，$R = N/K$）：

$$
M R \left(\frac{\beta}{\alpha}\right)^2 = 2N \cdot \frac{N}{K} \cdot \frac{1}{16N^2} = \frac{1}{8K} = O(1)
$$

常数恰好打平。

**这个 derivation 最 elegant 的地方**：DeepLoop 没有引入新机制、新 hyperparameter、新 loss——只是把 exponent 从 1/4 改成 1/2。整个 contribution 是一行代码级别的修改。但背后的 "tied-depth effect" 分析是新的，是真正解释 **为什么** 这一行修改是必要的。

---

## 6. 实验上看到啥

### Validation loss (GPT-2 small/medium, FineWeb-Edu 50BT)

| Setup | $R=1$ | $R=3$ | $R=5$ | $R=7$ |
|---|---|---|---|---|
| Small base | 2.8627 | 2.8077 | 2.7910 | 2.7700 |
| Small DeepLoop | 2.8631 | 2.7917 | 2.7679 | 2.7514 |
| Medium base | 2.6253 | 2.5779 | 2.5640 | 2.5558 |
| Medium DeepLoop | 2.6264 | 2.5627 | 2.5444 | 2.5280 |

三个观察：

**$R=1$ neutral** (+0.0004 / +0.0011 nats)。这其实是 paper 整个理论的自洽性证据——$R=1$ 没有 revisit，weight sharing effect 消失，DeepLoop 跟 baseline 在 perturbation argument 上是同一个 regime，所以不应该有 difference。理论上预测 neutral，实验上确实 neutral。

**$R \geq 3$ 时 DeepLoop 一致 better**，gap 随 $R$ 大致单调扩大。Medium scale 上 $R=7$ gap 是 -0.028 nats，这个量级在 single-seed 上算显著了。

**两种方法都 monotone improve with R**——looping 本身有效（这是 recurrent depth 的卖点），DeepLoop 只是让 scaling 更 stable、更接近最优 learning rate 的可达区间。

### p-sweep 直接验证 threshold

Paper Appendix C 在 GPT-2 small, $R=3$ 上 sweep $p \in \{0.30, ..., 0.60\}$，看是否能 escape unigram floor (≈7.67 nats) 在 2700 步内。Per-seed escape fractions：

| $p$ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|
| escape | 0/5 | 1/5 | 2/5 | 2/5 | 3/5 | 5/5 | 5/5 |

Transition bracket $p = 1/2$ 而不是 pinpoint 它。Conditional on training，smaller $p$ 给 better loss（3.70 at $p=0.45$ vs 3.80 at $p=0.60$）——更 aggressive 的 scaling 学得更快，但更不可靠。$p = 1/2$ 是 conservative sweet spot：worst-case analysis 推荐，empirical transition 也落在它附近。

这个结果其实揭示了一个比较 general 的 trade-off：**conservative exponent = 牺牲一点 learning signal 换取 stability**。如果未来能直接 measure $\kappa_R$ 并发现实际 alignment 低于 worst-case，那就能 safely 用 $p < 1/2$，gain 更强的 learning signal。

### ARC-AGI 上 HRM 的验证

Paper §5.3 把 DeepLoop 直接 apply 到 HRM（Hierarchical Reasoning Model, Wang 2025），只换 residual scaling，其他 hyperparameter 全保留。结果：

| Voting K | 1 | 2 | 10 | 100 | 1000 |
|---|---|---|---|---|---|
| Vanilla HRM | 31.50 | 36.50 | 41.50 | 47.50 | 50.75 |
| DeepLoop | 35.50 | 39.75 | 44.25 | 49.75 | 51.50 |
| Δ | +4.00 | +3.25 | +2.75 | +2.25 | +0.75 |

K=2 是 paper-protocol headline，+3.25 pp。Four-seed std ≈ 0.5 pp，所以这是 ~6σ effect。**关键点**：HRM 跟 single-module looped transformer 是两种不同 architecture，但 paper §4 证明 binding regime 是同一个（aligned tied-loop at fixed physical depth），所以 $p = 1/2$ 同时适用。这是 paper 最强的统一性 claim。

Reference: HRM paper: https://arxiv.org/abs/2506.21734 ; ARC-AGI: https://github.com/fchollet/ARC

---

## 7. HRM 推广里几个有意思的细节

HRM 的结构是两个 module 嵌套循环：
- **High module $\mathcal{H}$**：$K_H$ 个 block，每个 outer cycle 跑一次
- **Low module $\mathcal{L}$**：$K_L$ 个 block，每个 outer cycle 内部跑 $C_L$ 次
- **Outer cycle** 重复 $C$ 次
- **One-step gradient approximation**：backward 只算最后一个 outer cycle，前面 detach

Forward visit count 是 $M = 2C(K_H + C_L K_L)$，但 gradient-visible count 只有 $M_g = 2(K_H + C_L K_L)$，注意 $M = C \cdot M_g$。

**关键 insight**：因为 gradient truncation，$\mathcal{H}$ 在 backward graph 里只 visit 一次（$R_g^{(\mathcal{H})} = 1$），alignment $\kappa_g^{(\mathcal{H})} \leq 1$，自动退化到 untied DeepNorm regime，只需要 $p \geq 1/4$。$\mathcal{L}$ 才是真正承载 effective depth 的——它在 $C_L$ 内部反复 revisit 同一组 shared blocks，aligned $\kappa_g^{(\mathcal{L})} = \Theta(C_L)$，需要 $p \geq 1/2$。

如果两个 module 共享同一个 exponent，binding constraint 是更严的那个，所以 $p = 1/2$ 同时满足。Paper §4.2 还指出：bound 分解成 per-module summands，所以理论上**可以用 asymmetric exponent**：$\mathcal{H}$ 用 $p=1/4$（更 aggressive），$\mathcal{L}$ 用 $p=1/2$（更保守）。这个 asymmetric 方案没在实验里测，是潜在的 easy win。

---

## 8. sandwich block 的两个 Norm 各自干嘛

DeepLoop 用的是 post-LN sandwich block：

$$
\mathbf{x}_{i+1} = \mathrm{Norm}\left(\alpha \mathbf{x}_i + f_j(\mathrm{Norm}(\mathbf{x}_i); \phi_j)\right)
$$

两个 Norm 角色完全不同：

**外层 Norm**（在 $\alpha \mathbf{x}_i + \text{branch}$ 外面）：restore residual stream scale，让每个 visit 输出 unit-RMS，propagate 给下一 visit。这是 perturbation analysis 的关键—— Lemma A.1 证明 RMSNorm 在 high-$\alpha$ limit 下让 branch 通过 $1/\alpha$ factor 进入 normalized direction。

**内层 Norm**（在 branch input 上）：pin branch input 到 unit-RMS，无论 residual stream 在 training 中怎么 drift。这是 architectural 而非 analytical——Lemma A.2 证明 init 时所有 normalization gains 都是 1，所以 inner Norm 在 perturbation analysis 上是 identity，operator norm 1，不改变 visit-wise constants $C_U, C_G$。但实际 training 中 residual stream scale 可能漂移，inner Norm 保证 branch 总看到 unit-RMS input，让 Assumption 3.1 的 local scaling 假设持续成立。

**$\beta$ 的 role 也容易搞错**：$\beta$ 是 **init-only gain**，初始化时 $W^{(0)} = \beta \widetilde{W}^{(0)}$，runtime 不再出现 $\beta$。Paper §3.1 特意强调这点，因为很多人会误以为 $\beta$ 是一个 runtime scalar 乘在 branch 输出上。其实 branch 输出的 scale 由 $\beta$-缩过的 matrix 的 operator norm 决定，runtime 那个 matrix 已经是 $\beta \widetilde{W}^{(0)}$ 了，不需要再乘。

---

## 9. 我自己的几个联想

### 9.1 跟 RNN gradient analysis 的呼应

这篇 paper 的 double-sum 结构让我想起 RNN 的 BPTT 分析。RNN 也是 weight sharing across time steps，gradient 通过 time 维度累加。经典 RNN 的 vanishing/exploding gradient 问题本质上也是 cross-time coupling：$ \prod_t \partial h_t / \partial h_{t-1} $ 的累乘。

但 looped transformer 跟 RNN 有一个本质区别：**RNN 的 weight sharing 是 across time**（处理序列不同位置），**looped transformer 的 weight sharing 是 across depth**（处理同一输入的多次 refine）。RNN 的 alignment 问题来自序列结构（adjacent time steps 高度相关），looped transformer 的 alignment 问题来自 weight sharing 本身——shared block 设计上就是想实现同一种 operation，所以 cross-visit gradient 倾向 aligned。

Reference: RNN gradient analysis: https://arxiv.org/abs/1211.5063 (Pass et al. on RNN capacity, 大致)

### 9.2 跟 stochastic depth / dropout 的互动

Standard dropout 在 weight-shared 模型上行为微妙：每次 visit 用不同 dropout mask，相当于给 cross-visit gradient 加 noise，可能降低 alignment。如果这是真的，那 dropout 在 looped transformer 上可能不止是 regularizer，还可能是 **alignment breaker**——让 $\kappa_R$ 下降，从而允许更小的 $p$（更 aggressive learning rate）。

这个方向 paper 没探，但我觉得值得 follow-up。如果能在 training 中动态 measure $\kappa_R$，并根据它调整 $p$ 或 dropout 强度，可能比固定 $p=1/2$ 更优。

### 9.3 跟 BatchNorm 在共享卷积里的行为

CNN 里跨层共享 conv weight（Siamese network、FCN 等）也会遇到类似 alignment 问题——同一 filter 在不同 spatial location / different layers 反复 apply，gradient 是 sum。但 CNN 的 BatchNorm 是 per-layer 独立的，所以 RMSNorm 这种 per-visit re-scale 在 CNN 里的对应物没那么 clean。Transformer 因为 residual stream 是 explicit state，所以 normalization 能直接 pin state scale——这是 looped transformer 比 looped CNN 更适合做 alignment analysis 的原因之一。

### 9.4 跟 Mixture-of-Depths / early exit 的关系

Mixture-of-Depths（Raposo 2024, https://arxiv.org/abs/2404.02258）让每个 token 动态选择跳过某些 layer。这跟 looped transformer 一样都在"用 depth 做 compute allocation"，但机制完全不同：MoD 是 sparse untied depth（不同 token 走不同 layer subset），looped 是 dense tied depth（所有 token 都走 R 轮 shared blocks）。

两者的 residual scaling 问题也不同：MoD 因为还是 untied，DeepNorm 适用；looped 因为 tied，需要 DeepLoop。如果未来出现 MoD + looping 混合架构，perturbation argument 需要重新写——visit count $M$ 变成 token-dependent，bound 可能变成 per-token expected bound。这是个 open problem。

### 9.5 跟 test-time compute scaling 的关系

最近一堆 paper（Snell 2024, https://arxiv.org/abs/2408.03314; Geiping 2025）研究 test-time compute scaling——让 model 在 inference 时花更多 compute 换 better output。Looped transformer 是 test-time compute scaling 的一种实现：增加 $R$ = 增加 inference compute。

DeepLoop 在这个 framing 下的角色是：**给 test-time compute scaling 提供一个稳定的训练阶段 parameterization**。如果训练阶段不 stable，再多的 inference compute 也用不上。这跟 "pause token"（Goyal 2023, https://arxiv.org/abs/2310.02226）、"latent reasoning"（Hao 2024, https://arxiv.org/abs/2412.06769）这些 test-time compute 方向是 complementary 的。

### 9.6 跟 $\mu$P 的关系

$\mu$P（Yang 2021, https://arxiv.org/abs/2010.09258）和 Depth-$\mu$P（Yang 2023, https://arxiv.org/abs/2310.02244）研究的是 width-depth scaling 下的 hyperparameter transfer。$\mu$P 假设 untied depth，所以它的 depth scaling law 不直接适用 looped transformer。DeepLoop 是 loop-specific correction，理论上可以跟 $\mu$P 叠加：$\mu$P 给 width scaling，DeepLoop 给 loop-depth scaling。但具体怎么 combine、是否有 cross-term 互动，paper 没探，是 future work。

---

## 10. Paper 没说但我想知道的几件事

### 10.1 实际 $\kappa_R$ 在 trained model 上是多少

Paper 用 worst-case $\kappa_R = \Theta(R)$ 推导 conservative exponent。但 trained looped model 的实际 alignment 落在哪？如果在训练后期 measure $\sum_r U_{r,j}$ 和 $\sum_r G_{r,j}$ 的范数，看它们 relative to $\sqrt{R}$ baseline 是多少倍，就能知道实际 alignment 系数。如果实际 $\kappa_R = \Theta(\sqrt{R})$，那 threshold 是 $p = 3/8$，可以更 aggressive。

这个测量不难——log gradient norm per round 就行——但 paper 没做。我觉得这是最直接的 follow-up。

### 10.2 大 scale 验证

Paper 只测了 GPT-2 small (124M) 和 medium (350M)。Looped transformer 的卖点是用小 parameter count 达到大 effective depth，所以最自然的验证场景应该是"小 physical model + 大 loop count"，比如 100M parameter + R=20。Paper 测的 R 最多到 7，在这个 regime 下 improvement 是 -0.028 nats，确实显著但不算 dramatic。

如果 loop count 推到 20+，DeepLoop vs DeepNorm-style 的差距会不会更明显？还是说 alignment 在 large R 下饱和，gap 反而 plateau？这关系到 looped transformer 作为 scaling 轴的长期价值。

### 10.3 跟 RoPE position embedding 的互动

Looped transformer 的 position embedding 有微妙问题：如果 position 只在 entry encode，R 轮循环中 position 信息怎么 propagate？Paper 用的是 GPT-MHA-RoPE，但 paper 没明确说 RoPE 是只在第一轮 apply 还是每轮重新 apply。如果每轮重新 apply，position 信号会被反复 inject，可能影响 cross-round alignment（不同 round 看到同样的 position pattern，gradient 倾向 aligned）。如果只在第一轮 apply，后续 round 是 pure state evolution，alignment 模式会不一样。这个 architectural choice 可能直接影响 $\kappa_R$。

### 10.4 用 learnable $\alpha$ 替代固定 exponent

DeepLoop 用固定 $\alpha = (2N)^{1/2}$。但理论上可以让 $\alpha$ 变成 learnable scalar，让模型自己 discover optimal residual scaling。ReZero (Bachlechner 2021, https://arxiv.org/abs/2003.04887) 已经在 untied transformer 上做过这个。在 looped 设定下 learnable $\alpha$ 会跟 alignment 互动——如果某次 visit 的 gradient 倾向 align，model 可以自动 increase $\alpha$ 来抑制扰动。但 paper 没走这条路，保持了 DeepNorm 的"固定公式"风格。

这点其实是个 trade-off：固定公式的好处是 hyperparameter-free、可预测、可分析；learnable 的好处是 adapt to actual training dynamics。Looped transformer 因为 alignment 不可先验预测，learnable $\alpha$ 可能更优。但 paper 选择 conservative fixed 路线，把 learnable 留作 future work。

### 10.5 跟 RL post-training 的互动

Paper 只测了 pretraining。但 looped transformer 在 RLHF / DPO 之类的 post-training 阶段行为如何？Post-training 的 gradient distribution 跟 pretraining 很不一样（更 sparse、更 high-variance），alignment $\kappa_R$ 可能完全不同。DeepLoop 的 $p=1/2$ 在 post-training 阶段是否仍然 optimal？或者需要更小 $p$（因为 gradient 更 sparse 倾向 decorrelate）？这个 open question 对实际部署很重要。

---

## 11. 总结成一段话

DeepLoop 是 DeepNorm 在 weight sharing 设定下的 natural correction。核心观察：weight sharing 让 perturbation 从 single sum 变成 double sum，引入 cross-visit coupling，量度是 alignment coefficient $\kappa_R \in [0, R]$。Looped transformer 因为 weight sharing 设计上就 align，所以 worst-case $\kappa_R = \Theta(R)$，bound 变成 $MR(\beta/\alpha)^2 = O(1)$，exponent threshold 升到 $p = 1/2$。修复就是 $\alpha = (2N)^{1/2}$, $\beta = (8N)^{-1/2}$，一行代码。实验上 $R=1$ neutral（验证 untied 退化），$R \geq 3$ 一致 improve，ARC-AGI 上 HRM +3.25 pp（6σ）。最 elegant 的部分：DeepLoop 没加任何新机制，只是 exponent correction，但背后"depth parameterization 应该 depend on depth realization mechanism"原则比这个具体修正更 general。

---

## 12. Web references

- DeepLoop GitHub: https://github.com/lszshu/DeepLoop
- DeepNorm paper: https://arxiv.org/abs/2203.06523
- Universal Transformer: https://arxiv.org/abs/1807.03819
- ALBERT: https://arxiv.org/abs/1909.11942
- Recurrent depth (Geiping 2025): https://arxiv.org/abs/2502.05171
- Looped language models (Zhu 2026): https://arxiv.org/abs/2510.25741
- HRM (Wang 2025): https://arxiv.org/abs/2506.21734
- Reasoning with latent thoughts (Saunshi 2025): https://arxiv.org/abs/2502.17416
- Mixture-of-Depths (Raposo 2024): https://arxiv.org/abs/2404.02258
- Pause token (Goyal 2023): https://arxiv.org/abs/2310.02226
- Latent reasoning (Hao 2024): https://arxiv.org/abs/2412.06769
- Test-time compute scaling (Snell 2024): https://arxiv.org/abs/2408.03314
- Depth-$\mu$P (Yang 2023): https://arxiv.org/abs/2310.02244
- $\mu$P (Yang 2021): https://arxiv.org/abs/2010.09258
- ReZero (Bachlechner 2021): https://arxiv.org/abs/2003.04887
- Fixup (Zhang 2019): https://arxiv.org/abs/1901.09321
- Pre-LN analysis (Xiong 2020): https://arxiv.org/abs/2002.04745
- ARC-AGI: https://github.com/fchollet/ARC
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Looped transformers as programmable computers (Giannou 2023): https://arxiv.org/abs/2301.06704

---

**Final intuition**: 如果让我给 Andrej 一句话总结——DeepLoop = DeepNorm 把 "depth" 这个概念重新定义了。Untied depth 下，每个 layer 是独立的扰动源；tied depth 下，扰动是 cross-visit coupled 的。Weight sharing 的目的就是 align，align 就要更保守，保守就是 $p$ 从 1/4 提到 1/2。整个 paper 是关于 "depth scaling 应该 reflect depth 是怎么实现的"，不只是 "layer count 是多少"。这个 principle 比 $p=1/2$ 这个具体值更值得记住。

---

# DeepLoop 深度技术讲解

Andrej，这篇paper其实是把 DeepNorm 的 perturbation argument 推广到 weight-shared / looped Transformer 上的一篇很干净的工作。核心 insight 极其简单——looped model 的 residual scaling exponent 应该从 $p=1/4$ 提到 $p=1/2$——但背后的推导 chain 跟 tied-depth aggregation mechanism 值得仔细拆解。我会从最 bottom 的 perturbation argument 一步步 build up 到 final rule，再聊 HRM 扩展和实验。

---

## 1. 这篇paper想解决的问题

Looped Transformer 的 motivation 直接来自一个 scaling 角度的观察：standard Transformer 的 depth 与 parameter count 是 coupled 的（Kaplan 2020），加一层就加一组 attention + FFN 的 weights。Looped Transformer 解耦这两个 axis：存 $K$ 个 physical blocks，循环应用 $R$ 轮，effective depth $N = KR$，但 stored parameter 仍是 $K$ blocks。Universal Transformer（Dehghani 2018）、ALBERT（Lan 2019）、Subformer（Reid 2021）都是这种 depth-wise sharing 的早期实例，最近的 recurrent-depth models（Geiping 2025, https://arxiv.org/abs/2502.05171）和 looped language models（Zhu 2026, https://arxiv.org/abs/2510.25741）把它推到 language modeling scale。

问题来了：standard residual scaling analysis（DeepNorm, Wang 2024, https://arxiv.org/abs/2203.06523）是给 untied depth 写的——假设每个 unrolled residual sublayer 拥有独立的 parameter tensor。Looped Transformer 的 weight sharing 完全打破这个假设：同一 physical parameter 被 visit R 次，gradient 是 R 个 visit 的 sum；同一 updated tensor 又被同样的 R 个 visit 在下一次 linearized forward pass 中 read。这就是 paper 反复强调的 "tied-depth effect"，一个 shared update 同时被 multiple visits **写入**和**读出**，这两个 path 是 coupled 的。

Reference: DeepNorm paper: https://arxiv.org/abs/2203.06523

---

## 2. 重新审视 DeepNorm 的 perturbation argument

要理解 DeepLoop，必须先理解 DeepNorm 在 untied 设定下到底在做什么。假设我们有 depth-$N$ Post-LN Transformer，$M = 2N$ 个 residual sublayer visits（每 block 含 attention + FFN 两个 sublayer）。DeepNorm 的 forward recurrence 是

$$
\mathbf{x}_{\ell+1} = \mathrm{Norm}\left(\alpha \mathbf{x}_\ell + g_\ell(\mathbf{x}_\ell; \theta_\ell)\right),
$$

其中：
- $\alpha$ 是 skip connection 的 scaling 系数（标量，runtime 乘在 residual stream 上）
- $\beta$ 是 per-matrix 的 initialization gain（仅在 init 时乘 $\widetilde{W}^{(0)}$，runtime 不再出现）
- $\theta_\ell$ 是 sublayer $\ell$ 的 residual-branch 参数
- DeepNorm 选 $\alpha = (2N)^{1/4}$, $\beta = (8N)^{-1/4}$

DeepNet perturbation argument 的关键 Lemma（paper Appendix A.1 重写成 RMSNorm 形式）：

**Lemma A.1**: 设 $\mathrm{RMS}(\mathbf{x}) = 1$，$\mathcal{R}(\mathbf{y}) = \mathbf{y}/\mathrm{RMS}(\mathbf{y})$，若 $\mathrm{RMS}(\mathbf{z})/\alpha \leq c < 1$，则

$$
\mathcal{R}(\alpha \mathbf{x} + \mathbf{z}) = \mathbf{x} + \frac{\mathbf{z} - \langle \mathbf{x}, \mathbf{z} \rangle_d \mathbf{x}}{\alpha} + O\left(\frac{\mathrm{RMS}(\mathbf{z})^2}{\alpha^2}\right),
$$

其中 $\langle \mathbf{x}, \mathbf{z} \rangle_d := d^{-1} \mathbf{x}^\top \mathbf{z}$ 是 dimension-averaged inner product。

**这个 Lemma 的直觉**：RMSNorm 在 high-$\alpha$ limit 下近似一个 identity，但 residual branch $\mathbf{z}$ 通过一个 $1/\alpha$ factor 进入 normalized direction。所以 RMSNorm 不会"压死"residual stream，但会把 branch contribution 缩 $1/\alpha$。这就解释了为什么 DeepNorm 的关键 ratio 是 $\beta/\alpha$ 而不是 $\beta$ 单独——branch 输出经过 $1/\alpha$ 的"折扣"才到达 normalized direction。

DeepNet 的 first-order bound 在 untied 设定下是：

$$
\|\Delta F\| \leq C' M \left(\frac{\beta}{\alpha}\right)^2,
$$

**每个 visit 贡献一个 output sensitivity $O(\beta/\alpha)$（因为 $\beta$ 缩了 branch matrix 的 operator norm）和一个 effective update $O(\beta/\alpha)$（gradient 量级同样被 $\beta$ 控制），乘起来是 $O((\beta/\alpha)^2)$，sum over $M$ 个 visits 给 $M \cdot (\beta/\alpha)^2$**。Sufficient stability condition 就是

$$
M \left(\frac{\beta}{\alpha}\right)^2 = O(1).
$$

代入 DeepNorm 的 $\alpha = (2N)^{1/4}$, $\beta = (8N)^{-1/4}$：

$$
\frac{\beta}{\alpha} = \frac{(8N)^{-1/4}}{(2N)^{1/4}} = \frac{1}{2\sqrt{N}}, \quad M\left(\frac{\beta}{\alpha}\right)^2 = 2N \cdot \frac{1}{4N} = \frac{1}{2} = O(1).
$$

常数恰好打平，exponent $p=1/4$ 是 sufficient 的临界值（对 scaling family $\alpha = (cN)^p$, $\beta = (dN)^{-p}$）。

---

## 3. Tied-depth 的关键修正：双 sum 形式

Looped Transformer 把 DeepNet 的 perturbation argument 改了一个本质的东西。设 $\phi_j$ 是物理 sublayer $j$ 的参数（$j \in \{1, \ldots, J\}$, $J = 2K$），第 $r$ 轮 visit 的"虚拟" gradient 记为 $G_{r,j}$，对应 sensitivity operator 记为 $U_{r,j}$。

**Untied 设定**: $\Delta F = -\eta \sum_{i=1}^{M} U_i G_i + O(\eta^2)$，每个 visit 一项，共 $M$ 项。

**Tied 设定**: $\phi_j$ 的 optimizer update 正比于 $\sum_{r=1}^{R} G_{r,j}$（R 个 visit 的 gradient sum），然后这个 update 被 $\sum_{r=1}^{R} U_{r,j}$（R 个 sensitivity 的 sum）读取。First-order perturbation 因此写成 **double-sum**:

$$
\Delta F_{\mathrm{tied}} = -\eta \sum_{j=1}^{J} \left(\sum_{r=1}^{R} U_{r,j}\right) \left(\sum_{t=1}^{R} G_{t,j}\right) + O(\eta^2).
$$

这就是 paper Eq. (3.6)，**整个 DeepLoop paper 的核心数学对象**。它的展开会产生 $R^2$ 项 cross terms，而不只是 $R$ 项 self terms。Tied-depth effect 完全藏在这个 $R^2$ 项里。

---

## 4. Visit-alignment coefficient $\kappa_R$ 的精确定义

为了 quantify cross terms 的大小，paper 定义

$$
\kappa_R := \max_{j \in \{1,\ldots,J\}} \frac{\left\|\sum_{r=1}^{R} U_{r,j}\right\| \left\|\sum_{r=1}^{R} G_{r,j}\right\|}{R \, C_U \, C_G \, (\beta/\alpha)^2},
$$

其中 $\|U_{r,j}\| \leq C_U \beta/\alpha$, $\|G_{r,j}\| \leq C_G \beta/\alpha$ 是 Assumption 3.1 的 local scaling bounds。

**变量的物理含义**：
- 分子 $\|\sum_r U_{r,j}\| \cdot \|\sum_t G_{t,j}\|$ 是 sensitivity sum 与 update sum 的乘积范数
- 分母 $R \cdot C_U C_G (\beta/\alpha)^2$ 是"如果 R 个 visit 完全 decorrelate"的 baseline（sum-of-squares-like，每个 visit 独立贡献 $C_U C_G (\beta/\alpha)^2$）
- $\kappa_R$ 衡量的是 cross-round alignment 的"放大倍数"相对于 independent baseline 的比值

**Triangle inequality 给的 bound**: $0 \leq \kappa_R \leq R$。

- **Decorrelated visits**: $\kappa_R = O(1)$，sum 范数增长 $\sqrt{R}$ 而非 $R$，所以 ratio 接近 1
- **Fully aligned visits**: $\kappa_R = \Theta(R)$，sum 范数 linear in $R$，ratio 正比于 $R$（因为分子 $R \cdot R = R^2$，分母 $R$，所以 $\kappa_R = \Theta(R)$）

代入 double-sum bound：

$$
\|\Delta F\| \leq C'' \, M \kappa_R \left(\frac{\beta}{\alpha}\right)^2,
$$

sufficient condition 变成

$$
M \kappa_R \left(\frac{\beta}{\alpha}\right)^2 = O(1).
$$

这就是 paper Eq. (3.8)–(3.9)。

---

## 5. Exponent threshold 的推导

Paper 考虑 scaling family $\alpha = (cN)^p$, $\beta = (dN)^{-p}$（保留 DeepNorm 的常数族结构）。那么

$$
\frac{\beta}{\alpha} = (cd)^{-p} N^{-2p}.
$$

设 $\kappa_R = \Theta(R^\gamma)$ for $\gamma \in [0, 1]$，且 $K$ 固定，$R = N/K \to \infty$。代入 stability condition：

$$
M \kappa_R \left(\frac{\beta}{\alpha}\right)^2 = 2N \cdot \Theta(R^\gamma) \cdot (cd)^{-2p} N^{-4p} = \Theta\left(N^{1 + \gamma - 4p}\right).
$$

要 uniformly bounded as $R \to \infty$ 需要

$$
1 + \gamma - 4p \leq 0 \quad \Leftrightarrow \quad p \geq \frac{1 + \gamma}{4}.
$$

**这就是 Proposition 3.2 的核心**。两个 limit case：

| Regime | $\gamma$ | Threshold $p$ | 对应模型 |
|---|---|---|---|
| Decorrelated visits | $0$ | $1/4$ | Untied DeepNorm（recovered）|
| Fully aligned visits | $1$ | $1/2$ | DeepLoop |

**关键 insight**: 当 physical depth $K$ 固定、loop count $R$ 增长时（这是 looped model "用 looping 做深度 scaling"的实际 regime），shared update 倾向于 aligned——这正是 weight sharing 的目的（让多次 visit 实现同一种操作），alignment 不会自然消失。所以 worst case $p = 1/2$ 是 conservative 但不是"过度保守"——它对应于 tied-depth 的实际 scaling axis。

DeepLoop 选

$$
\alpha = (2N)^{1/2}, \quad \beta = (8N)^{-1/2},
$$

保留 DeepNorm 的 $(c, d) = (2, 8)$ 常数，只改 exponent。计算 ratio：

$$
\frac{\beta}{\alpha} = \frac{(8N)^{-1/2}}{(2N)^{1/2}} = \frac{1}{4N}.
$$

代入 worst-case aligned bound $M R (\beta/\alpha)^2$（$K$ 固定，$R = N/K$）：

$$
M R \left(\frac{\beta}{\alpha}\right)^2 = 2N \cdot R \cdot \frac{1}{16N^2} = \frac{2R}{16N} = \frac{R}{8N} = \frac{1}{8K} = O(1).
$$

常数恰好打平，依然是 $O(1/K)$，只要 $K$ 固定就 bounded。这就是 DeepLoop 的"one-line correction"——把 DeepNorm 的 exponent 从 $1/4$ 提到 $1/2$。

---

## 6. Architecture: sandwich block 的精确形式

DeepLoop 的 sublayer 形式是 paper Eq. (3.1)：

$$
\mathbf{x}_{i+1} = \mathrm{Norm}\left(\alpha \mathbf{x}_i + f_j\left(\mathrm{Norm}(\mathbf{x}_i); \phi_j\right)\right), \quad i = 1, \ldots, M.
$$

注意这里有 **两个 Norm**：
- **外层 Norm（post-normalization）**: restore residual stream scale，让每 visit 输出 unit-RMS，propagate 给下一 visit
- **内层 Norm（branch-input normalization）**: pin branch input 到 unit-RMS，无论 residual stream scale 在 training 中怎么 drift

内层 Norm 来自 Geiping 2025（recurrent-depth models），**它的作用是 architectural 而不是 analytical**。Lemma A.2 证明：在 initialization 时所有 normalization gains 都是 1，input-embedding RMSNorm 把 entry 设成 unit-RMS，所以每个 visit 的 input 都满足 $\mathrm{RMS}(\mathbf{x}_i) = 1$，此时内层 Norm 是 identity，且其 Jacobian operator norm 恰好 1。所以 Assumption 3.1 的常数 $C_U, C_G$ 不变，perturbation argument 完全 verbatim 适用 sandwich block。

**$\beta$ 的精确用法**: 对每个物理 sublayer $j$，对 DeepNorm-specified matrices set $\mathcal{S}_j$（attention 的 value/output projection、FFN 的两个 matrix）：

$$
W_{j,q}^{(0)} = \beta_{\mathrm{DL}} \widetilde{W}_{j,q}^{(0)}, \quad q \in \mathcal{S}_j, \quad \beta_{\mathrm{DL}} = (8N)^{-1/2}.
$$

**这里 $\beta$ 是 init-time only**，runtime 不会再乘——这点 paper 在 §3.1 反复强调，避免读者误以为 $\beta$ 是一个 runtime scalar。

---

## 7. 跟 DeepNorm 的对比表

Paper §3.5 给了一个很好的总结 table，我重新格式化：

| 量 | DeepNorm | DeepLoop |
|---|---|---|
| Residual scale $\alpha$ | $(2N)^{1/4}$ | $(2N)^{1/2}$ |
| Init gain $\beta$ | $(8N)^{-1/4}$ | $(8N)^{-1/2}$ |
| Per-matrix use of $\beta$ | $W^{(0)} \leftarrow \beta \widetilde{W}^{(0)}$ | 同 |
| Update-to-residual $\beta/\alpha$ | $1/(2\sqrt{N})$ | $1/(4N)$ |
| Untied bound $M(\beta/\alpha)^2$ | $\Theta(1)$ | $\Theta(N^{-1})$ |
| Aligned tied-loop bound $MR(\beta/\alpha)^2$ (fixed $K$) | $\Theta(R)$ **grows unbounded** | $\Theta(1/K)$ **bounded** |

最后一行是关键：**DeepNorm 在 aligned tied-loop 设定下，bound 增长 $\Theta(R)$——意味着 unbounded as $R \to \infty$，会失稳**。DeepLoop 把它压到 $O(1/K)$，fixed $K$ 时 bounded。

---

## 8. 扩展到 Hierarchical Reasoning Model

Paper §4 把分析推广到 HRM（Wang 2025, https://arxiv.org/abs/2506.21734），这是这个 framework 的"pay-off"——证明 $p=1/2$ 在 reasoning model 上也对。HRM 的结构是：

- **High module $\mathcal{H}$**: $K_H$ physical blocks，每 outer cycle 更新一次
- **Low module $\mathcal{L}$**: $K_L$ physical blocks，每 outer cycle 内部迭代 $C_L$ 次
- **Outer cycle**: 重复 $C$ 次
- **One-step gradient approximation**: backward 只算最后一个 cycle，前面 cycle 的 $\mathbf{z}_H, \mathbf{z}_L$ 被 detach

总 unrolled visit count:

$$
M = 2C(K_H + C_L K_L).
$$

但 training graph 可见的 visit count（gradient-visible）是

$$
M_g = 2(K_H + C_L K_L),
$$

注意 $M = C \cdot M_g$。Per-module gradient-visible round counts:

$$
R_g^{(\mathcal{H})} = 1, \quad R_g^{(\mathcal{L})} = C_L.
$$

**关键观察**: $\mathcal{H}$ module 因为 truncation，每个 physical sublayer 在 backward graph 里只 visit 一次，所以 $R_g^{(\mathcal{H})} = 1$，alignment $\kappa_g^{(\mathcal{H})} \leq 1$，自动退化到 untied DeepNorm regime，$p \geq 1/4$ 即可。

**$\mathcal{L}$ module 才是真正承载 effective depth 的**——它在 $C_L$ 内部迭代中反复 revisit 同一组 shared blocks。Weight sharing 的目的就是让这些 revisit 实现同一种 operation，所以 aligned $\kappa_g^{(\mathcal{L})} = \Theta(C_L)$ 是 binding regime。

Bound 分解成 per-module summand：

$$
\|\Delta F\| \leq C'' \left[ J_H R_g^{(\mathcal{H})} \kappa_g^{(\mathcal{H})} + J_L R_g^{(\mathcal{L})} \kappa_g^{(\mathcal{L})} \right] \left(\frac{\beta}{\alpha}\right)^2.
$$

Sufficient condition:

$$
M_g \bar{\kappa}_g \left(\frac{\beta}{\alpha}\right)^2 = O(1), \quad \bar{\kappa}_g := \frac{J_H + J_L C_L \kappa_g^{(\mathcal{L})}}{M_g}.
$$

考虑 scaling family $\alpha = (cN_g)^p$, $\beta = (dN_g)^{-p}$，$N_g = M_g/2$。在 $C_L \to \infty$, $K_H, K_L$ 固定的极限下（HRM 实际增长的 axis），$M_g = \Theta(C_L)$, $\bar\kappa_g = \Theta(C_L)$，所以 threshold 是 $p \geq 1/2$。

**结论**: HRM 的 binding module 是 $\mathcal{L}$（aligned，fixed physical depth, growing loop count），需要 $p=1/2$。$\mathcal{H}$ 只需要 $p \geq 1/4$，所以 shared exponent $p=1/2$ 同时满足两个 module。这是 paper §4.3 推导出的 Eq. (4.7)。

Reference: HRM paper: https://arxiv.org/abs/2506.21734

---

## 9. 实验：validation loss

Paper §5.1 的主 table（GPT-2 small & medium, FineWeb-Edu 50BT, 100K steps）：

| Method | $R=1$ | $R=3$ | $R=5$ | $R=7$ |
|---|---|---|---|---|
| **GPT-2 small** | | | | |
| baseline (pre-LN) | 2.8627 | 2.8077 | 2.7910 | 2.7700 |
| DeepLoop | 2.8631 | 2.7917 | 2.7679 | 2.7514 |
| Δ | +0.0004 | **-0.0160** | **-0.0231** | **-0.0186** |
| **GPT-2 medium** | | | | |
| baseline (pre-LN) | 2.6253 | 2.5779 | 2.5640 | 2.5558 |
| DeepLoop | 2.6264 | 2.5627 | 2.5444 | 2.5280 |
| Δ | +0.0011 | **-0.0153** | **-0.0196** | **-0.0278** |

**几个关键观察**：
1. **$R=1$ 时 neutral**（+0.0004 / +0.0011 nats, within noise）——因为 $R=1$ 没有 revisit，weight sharing effect 消失，DeepLoop 跟 baseline 在 perturbation argument 上是同一个 regime
2. **$R \geq 3$ 时 DeepLoop 严格 better**，gap 随 $R$ 单调扩大到 $R=7$（medium scale）
3. **Medium scale gap 更大**: $R=7$ 时 -0.028 nats，比 small 的 -0.019 更显著——暗示 scale up 时 tied-depth effect 不会衰减
4. **两种方法都 monotone improve with R**——looping 本身有效，DeepLoop 只是让 scaling 更 stable

---

## 10. Downstream: 8-task lm-eval-harness

Paper §5.2 报了 GPT-2 medium 在 8 tasks（ARC-C, ARC-E, HellaSwag, OBQA, PIQA, SciQ, SIQA, WinoGrande）上的 0-shot / 1-shot accuracy。Avg 列：

| Method / R | 0-shot Avg | 1-shot Avg |
|---|---|---|
| baseline R=1 | 50.86 | 51.69 |
| baseline R=3 | 52.50 | 53.10 |
| baseline R=5 | 52.61 | 54.42 |
| baseline R=7 | 52.95 | 54.62 |
| DeepLoop R=1 | 50.85 | 51.53 |
| DeepLoop R=3 | 52.93 | 53.42 |
| DeepLoop R=5 | 53.67 | 54.19 |
| **DeepLoop R=7** | **53.88** | **55.20** |

**$R=7$ 时 DeepLoop 在 7/8 个 task 上都赢**（只有 PIQA 输），WinoGrande 上 +1.74 pp 0-shot 是最大单 task gain。

Reference: lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness

---

## 11. ARC-AGI: HRM 实验验证 §4 的预测

Paper §5.3 是最 striking 的实验：把 DeepLoop 直接 apply 到 HRM（保留所有其他 hyperparameter，只换 residual scaling），在 ARC-AGI-1 上 evaluate。$M_g = 2(K_H + C_L K_L) = 24$ for HRM 默认 config（$K_H = K_L = 4$, $C_L = 2$），所以 $N_g = 12$。

| Method | K=1 | K=2 | K=10 | K=100 | K=1000 |
|---|---|---|---|---|---|
| Vanilla HRM | 31.50 | 36.50 | 41.50 | 47.50 | 50.75 |
| DeepLoop ($p=1/2$) | 35.50 | 39.75 | 44.25 | 49.75 | 51.50 |
| Δ | **+4.00** | **+3.25** | **+2.75** | **+2.25** | +0.75 |

**K=2（paper-protocol headline metric）: +3.25 pp**。Four-seed control 给 K=2 std ≈ 0.5 pp，所以这个 gain ≈ 6σ——不是 seed draw。

这印证了 §4.3 的 prediction：HRM 的 binding regime 是 $\mathcal{L}$ module 的 aligned tied-loop，所以 $p = 1/2$ 是 correct exponent，跟 single-module looped transformer 一样。**一个 single residual scaling rule，$p=1/2$，同时适用 looped LM 和 hierarchical reasoner**——这是 paper 的最强 claim。

Reference: ARC-AGI: https://github.com/fchollet/ARC

---

## 12. p-sweep 直接验证 Proposition 3.2

Paper Appendix C 在 GPT-2 small, $R=3$ 上 sweep exponent $p \in \{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60\}$，每个 $p$ up to 5 seeds，看是否能 escape unigram frequency floor (≈7.67 nats) 在 2700 steps 内。

Per-seed escape fractions:

| $p$ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|
| escape / total | 0/5 | 1/5 | 2/5 | 2/5 | 3/5 | 5/5 | 5/5 |

**Picture**:
- $p \leq 0.40$: 大多数 seed 失败，alignment effect 太强，DeepNorm-style exponent 不够
- $p = 0.50$ (DeepLoop default): 3/5 escape，still 在 transition 区间
- $p \geq 0.55$: 全部 escape，stable

**Conditional on convergence**，larger $p$ 给 slightly higher loss（less aggressive learning）：
- $p=0.45$: 3.70 nats
- $p=0.50$: 3.73 nats
- $p=0.55$: 3.76 nats
- $p=0.60$: 3.80 nats

**这印证 Proposition 3.2**：worst-case aligned regime 的 threshold 在 $p = 1/2$ 附近，empirical transition bracket 而不是 pinpoint 它。$p < 1/2$ 训练不稳定（多数 seed fail），$p > 1/2$ stable 但 learning signal 弱。$p = 1/2$ 是 conservative sweet spot。

---

## 13. 跟其他 parameterization 的关系

**$\mu$P / Depth-$\mu$P**（Yang 2021, 2023, https://arxiv.org/abs/2310.02244）：这些 work 研究的是 width-depth scaling 下的 hyperparameter transfer，但**没有 explicitly account for weight sharing across loop visits**。DeepLoop 是 loop-specific correction，跟 $\mu$P 是 orthogonal 的——理论上可以叠加（$\mu$P 给 width scaling，DeepLoop 给 loop-depth scaling）。

**Fixup initialization**（Zhang 2019, https://arxiv.org/abs/1901.09321）：不需要 Norm 的 residual scaling，但 init 设定跟 DeepNorm 不同。

**ReZero**（Bachlechner 2021）：用 learned scalar gate 替代 Norm，convergence 性质不同。

**Sandwich Norm placement**（Geiping 2025, https://arxiv.org/abs/2502.05171）：DeepLoop 直接采用这个 placement，但 Lemma A.2 证明 inner Norm 在 perturbation analysis 上是 inert 的，所以 adoption 是 architectural choice 而不是 analytical necessity。

---

## 14. 给你的 intuition 总结

Andrej，如果让我压缩到最核心的几句话：

1. **Untied DeepNorm**: 每个 residual branch 拥有独立参数，first-order bound 是 $M(\beta/\alpha)^2 = O(1)$，给 $p=1/4$
2. **Tied loop**: 参数共享让 update 既被 multiple visits 写（gradient sum）又被 multiple visits 读（sensitivity sum），形成 double-sum $\sum_j (\sum_r U_{r,j})(\sum_t G_{t,j})$，cross terms 引入 alignment coefficient $\kappa_R \in [0, R]$
3. **Aligned regime**（这是 looped model 实际工作方式——weight sharing 的目的就是 align revisit 的 operation）: $\kappa_R = \Theta(R)$，bound 变成 $MR(\beta/\alpha)^2 = O(1)$，threshold 升到 $p=1/2$
4. **One-line fix**: $\alpha = (2N)^{1/2}$, $\beta = (8N)^{-1/2}$，常数跟 DeepNorm 一样，只改 exponent
5. **Empirical confirmation**: GPT-2 scale 上 $R=1$ neutral（验证"no revisit 退化到 untied"），$R \geq 3$ 一致 improve；ARC-AGI 上 HRM +3.25 pp（6σ），证明同一 $p=1/2$ rule 适用 hierarchical reasoning

**DeepLoop 最 elegant 的地方**：它没有引入 gate、learnable scalar、auxiliary loss——只是一个 deterministic exponent correction。这暗示了一个更广的 principle：**residual scaling 应该 depend on depth 是如何实现的**（tied vs untied），not only on nominal layer count。这跟 Recurrent ZOO 那种"加 mechanism"路线不同——是"修正 parameterization"路线。

可能 future work 的几个方向：
- **直接 measure $\kappa_R$**: 在 trained looped model 上记录 cross-round gradient inner products，看实际 alignment 落在 $O(1)$ 与 $\Theta(R)$ 之间哪。如果实际接近 $O(1)$，可以 safely 用 $p$ 接近 $1/4$，gain more learning signal
- **Decorrelation training**: 设计 auxiliary loss 鼓励 cross-round gradient decorrelate，让 $p$ 可以更 aggressive
- **HRM per-module asymmetric exponents**: §4.2 证明 bound 分解成 per-module summands，所以 $(\alpha_\mathcal{H}, \beta_\mathcal{H})$ 用 $p=1/4$（H module untied），$(\alpha_\mathcal{L}, \beta_\mathcal{L})$ 用 $p=1/2$（L module tied）——理论上可以更 aggressive
- **Larger scale**: paper 只测了 GPT-2 small/medium，没有 billion-scale 验证。Loop-depth scaling 跟 standard depth scaling 是否在 Chinchilla-optimal regime 一致是 open question
- **跟 test-time compute scaling laws 的关系**: Geiping 2025 证明了 recurrent depth 可以 scale test-time compute，DeepLoop 让这个 scaling 更 stable。是否有一个 unified scaling law 给"depth via looping" vs "depth via untied layers"的 compute-quality tradeoff？

---

## 15. Web references

- DeepLoop GitHub: https://github.com/lszshu/DeepLoop
- DeepNorm (Wang 2024): https://arxiv.org/abs/2203.06523
- Universal Transformer (Dehghani 2018): https://arxiv.org/abs/1807.03819
- ALBERT (Lan 2019): https://arxiv.org/abs/1909.11942
- Subformer (Reid 2021): https://arxiv.org/abs/2104.04071 (近似)
- Recurrent depth (Geiping 2025): https://arxiv.org/abs/2502.05171
- Looped language models (Zhu 2026): https://arxiv.org/abs/2510.25741
- HRM (Wang 2025): https://arxiv.org/abs/2506.21734
- Reasoning with latent thoughts (Saunshi 2025): https://arxiv.org/abs/2502.17416
- Discoloop (Fu 2026): https://arxiv.org/abs/2607.00341
- Looped transformers as programmable computers (Giannou 2023): https://arxiv.org/abs/2301.06704
- Depth-$\mu$P (Yang 2023): https://arxiv.org/abs/2310.02244
- Fixup (Zhang 2019): https://arxiv.org/abs/1901.09321
- ReZero (Bachlechner 2021): https://arxiv.org/abs/2003.04887
- ARC-AGI: https://github.com/fchollet/ARC
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Pre-LN analysis (Xiong 2020): https://arxiv.org/abs/2002.04745
- Transformers without tears (Nguyen & Salazar 2019): https://arxiv.org/abs/1910.05895
- On training Transformer difficulty (Liu 2020): https://arxiv.org/abs/1906.02643
- Scaling laws (Kaplan 2020): https://arxiv.org/abs/2001.08361
- Chinchilla (Hofmann 2022): https://arxiv.org/abs/2203.15556

---

**Final takeaway**: 这篇 paper 的核心 insight 是 weight sharing 改变了 first-order perturbation 的结构——从 $\sum_i U_i G_i$ 变成 $\sum_j (\sum_r U_{r,j})(\sum_t G_{t,j})$，多了一个 cross-visit coupling。Alignment coefficient $\kappa_R$ 是 quantify 这个 coupling 的精确工具。最 conservative 的 $p=1/2$ 在实际 aligned regime 下是 necessary 的，empirical transition 也恰好在 $p=1/2$ 附近。把 $p=1/4$ 改成 $p=1/2$ 是 DeepLoop 的全部 contribution——一个 one-line 修正，但背后的"depth parameterization 应该 depend on depth realization mechanism"原则比这个具体修正更 general。
