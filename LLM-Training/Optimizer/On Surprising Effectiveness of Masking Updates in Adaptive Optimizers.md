---
source_pdf: On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.pdf
paper_sha256: 1ff3f7f03659dc386f4783ec54b66f4fd1ed36d8894dacf6cad7c0cfff489cf0
processed_at: '2026-08-05T23:39:22-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

训练 LLM 时, 每步把一半的 parameter blocks 直接跳过不更新, 结果模型反而训得更好。

## 为什么这事反直觉

你想啊, backward pass 花了那么多算力把所有 gradients 都算出来了, 现在你告诉我扔掉一半, 剩下的乘以 2 放大, 效果反而更好? 这就像你花 100 块钱买了一桌子菜, 然后服务员说"我建议你扔掉一半, 剩下的我帮你 double 份量", 你会觉得这服务员有病。

传统 optimization 理论也会告诉你这有问题:
- Bernoulli masking = 加噪声, variance 变大, 收敛变慢
- 同样的计算成本, 每个参数只收到一半 update, sample efficiency 应该变差
- 经典 coordinate descent 都是 greedy 选坐标, 随机选是公认的低效

但实验事实就摆在那里: **1B Llama 上, 扔掉一半 updates 的 RMSProp, 比 Muon 这种精心设计的 matrix optimizer 还低 9% perplexity**。这时候你就知道传统理论漏掉了什么东西。

## 关键 insight: 二阶效应

直觉全在 Proposition 1 这一步。我换个角度讲。

### Dense update 的 expected loss

假设你用标准 optimizer 算出一个 update $\Delta_t$, 走一步到 $\theta_t - \Delta_t$, loss 大概降 $\|\Delta_t\|^2$ 这么多(一阶项)。

### SkipUpdate 的 expected loss

现在你把 $\Delta_t$ 乘一个随机 Bernoulli mask $m_t \in \{0, 1\}$, 期望 $p$, 然后除以 $p$ 保持 unbiased。走一步到 $\theta_t - \tilde{\Delta}_t$。

一阶项: $\mathbb{E}[\tilde{\Delta}_t] = \Delta_t$, 所以一阶期望不变, 该降多少还降多少。

二阶项: 这里就是 magic 了。Taylor 展开二阶项是 $\frac{1}{2}\tilde{\Delta}_t^\top H \tilde{\Delta}_t$。注意 $\tilde{\Delta}_t$ 里含 $m_t^2 = m_t$ (因为是 Bernoulli 0/1, 平方等于自己)。所以二阶项的期望里, **对角 block 贡献是 $\frac{1}{p}$ 倍, 非 diagonal block 贡献不变**(因为 $m_t^{(b)}$ 和 $m_t^{(b')}$ 独立, $p \cdot p$ 乘上 $\frac{1}{p} \cdot \frac{1}{p}$ 抵消)。

净效果: 多了一个 $\frac{1-p}{2p}(\Delta_t^{(b)})^\top H_{bb} \Delta_t^{(b)}$ 的项。

### 这多出来的项在说什么

$(\Delta_t^{(b)})^\top H_{bb} \Delta_t^{(b)}$ 就是 update direction 在 block b 局部曲率下的 Rayleigh quotient。

- 如果你的 update 顺着 sharp valley wall 走, $H$ 在那个方向 eigenvalue 大, 这项就大
- 如果你的 update 顺着 flat 方向走, 这项小

**minimizing expected post-update loss = implicitly penalize sharp-direction updates**。

换句话说, random masking 让 optimizer **自动倾向走平的路**, 而平的路就是 flat minima, flat minima 就是 generalization 好的地方。这一切从 stochastic noise 里 free 浮现出来, 你不需要像 SAM 那样额外算一次 forward+backward, 也不需要像 KFAC 那样近似 Hessian。

## 为什么 transformer 特别吃这一套

CNN 的 Hessian 大致是 homogeneous 的: 不同 layer、不同 channel 的 curvature scale 差不多。你 random mask 哪个 block 都差不多。

Transformer 不一样, 它的 loss landscape 极度 heterogeneous:
- 不同 layer 的 Hessian spectrum 跨好几个 order
- Attention 和 MLP 块的 curvature 完全不同
- 不同 head 之间差异大
- Hessian 还呈现 block-diagonal 结构(Kunstner et al., 2024 测过)

Block-diagonal 结构意味着主要的 curvature 交互就在 block 内部, 所以 block-wise quadratic penalty 特别精准。

再加一条: LLM 的 gradient noise 是 heavy-tailed 的 (Zunstner et al., 2024; Zhang et al., 2020)。Paper Figure 3 在 controlled benchmark 上验证: 在 light-tailed 噪声下 Magma 和 Adam 差不多, 但在 heavy-tailed 噪声下 Magma 碾压 Adam。LLM 训练正好是 heavy-tailed, 这就是 mechanistic 解释。

## Magma 在 SkipUpdate 上加了什么

SkipUpdate 是 **homogeneous masking**: 所有 block 都一视同仁地按 $p=0.5$ 扔。但 transformer 的 blocks 这么异质, 一视同仁显然不是最优。

Magma 的想法: 用 **momentum 和 gradient 的 cosine similarity** 作为这个 block 当前是 signal 还是 noise 的代理。

为什么这个代理靠谱? Orvieto & Gower (2025) 在 variational inference 视角下证明: momentum $\mu$ 和 stochastic gradient $g$ 反向对齐的概率 $\mathbb{P}(\mu^\top g < 0) = \Phi(-\|\mu\|/\sigma)$, 其中 $\sigma$ 是 gradient noise std。这个概率随信噪比 $\|\mu\|/\sigma$ **指数衰减**。

换句话说: 如果 momentum 和 gradient 反向, 这是一个统计异常事件, 极大概率意味着这个 block 当前的 gradient 是噪声主导的, 不是真正的 descent direction。

所以 Magma 算一个 alignment score:
$$\tilde{s}_t^{(b)} = \text{sigmoid}\left(\frac{\text{cossim}(\mu_t^{(b)}, g_t^{(b)})}{\tau}\right)$$

- Alignment 高 (同向): $\tilde{s}_t \to 1$, update 放行
- Alignment 低 (反向): $\tilde{s}_t \to 0.12$ ($\tau=2$ 时 sigmoid(-0.5)), update 被抑制

然后 EMA 平滑一下 $s_t^{(b)} = 0.9 s_{t-1}^{(b)} + 0.1 \tilde{s}_t^{(b)}$, 再乘到 update 上: $\theta_{t+1}^{(b)} = \theta_t^{(b)} - s_t^{(b)} m_t^{(b)} \Delta_t^{(b)}$。

注意这里同时保留了 Bernoulli mask $m_t^{(b)}$, 所以 Proposition 1 的几何正则化还在, 只是在此基础上根据 alignment 做了软调制。

## 为什么 Magma 比 Cautious Optimizer 好

Cautious Optimizer (Liang et al., 2024) 也是用 momentum-gradient alignment, 但它是 **deterministic hard mask**: gradient 和 momentum sign 相反就完全扔掉。

问题: deterministic mask 没有 stochastic noise, 所以 Proposition 1 那个二阶 curvature regularization **直接消失**。C-Adam 只是把 destabilizing update 扔了, 但没有 bias 到 flat minima。

Magma 保留了 Bernoulli 随机性, 所以既过滤了 noise update, 又诱导了几何正则化, 两件事一起干。

## 几个细节让人觉得这 paper 是认真的

**Dense momentum 不能省**: Paper 专门 ablation 过, 如果 momentum 也跟着 sparse 更新(像 GaLore 那样), 即使加 damping 也明显劣于 dense momentum, 不加 damping 直接发散。原因: dense momentum 累积了所有"未执行更新"的 gradient, 是一个 variance-reduced estimator; sparse momentum 丢掉了这个性质。

**只对 attention 和 MLP 做 mask**: 不对所有参数 mask, embedding、layer norm 这些保持 dense。这一条配置就鲁棒地 work across model sizes。

**Block-level 而不是 element-level**: 理论上 element-level masking 只惩罚 Hessian 对角元, 丢了 within-block off-diagonal curvature。但实验上 block/column/element 差不多, 因为 Adam/RMSProp 这种 diagonal preconditioner 本来就利用不了 off-diagonal 信息。所以选 block, 计算上还更高效(可以整块 skip)。

**对学习率极不敏感**: Figure A3 显示 Adam 在 lr > 0.003 就崩, Adam+Magma 在 lr = 0.05 还能正常训。这非常实用: hyperparameter tuning 成本大降。

## 整个故事串起来

1. **Empirical surprise**: 扔掉一半 updates 反而更好, 碾压 SOTA optimizer
2. **Theoretical mechanism**: 二阶 Taylor 展开, Bernoulli noise 自然诱导一个 block-wise curvature penalty, 等价于 implicit flatness regularization
3. **Why transformer**: heterogeneous block curvature + block-diagonal Hessian + heavy-tailed noise 三个条件同时满足
4. **Principled improvement**: 用 momentum-gradient alignment 作为 per-block signal/noise 代理, 在 noise block 上抑制, 在 signal block 上放行, 既保留随机性(几何正则化)又过滤 noise
5. **Practical**: drop-in wrapper, 零额外内存, 几行代码, 对 lr 鲁棒, scaling 友好

这 paper 最让我欣赏的是 step 2 那一步。一个看似 hacky 的 trick, 被一个干干净净的二阶展开解释成 implicit curvature regularization, 然后基于这个理解设计出更好的 Magma。从 surprise 到 understanding to improvement, 整条链路非常 clean。

---

## References

- Paper: https://arxiv.org/abs/2502.04654 (推测链接, 需确认)
- Cautious Optimizer: https://arxiv.org/abs/2411.16085
- Muon: https://kellerjordan.github.io/posts/muon/
- GaLore: https://arxiv.org/abs/2403.03507
- Heavy-tailed noise in LLM training: https://arxiv.org/abs/2402.05820
- Transformer Hessian structure (Zhang et al., 2024a): NeurIPS 2024
- Adam's secret sauce (Orvieto & Gower, 2025): https://arxiv.org/abs/2505.21829
- SAM: https://arxiv.org/abs/2010.01412
- Linear transformer benchmark (Ahn et al., 2024): ICLR 2024
- Flat minima (Hochreiter & Schmidhuber, 1997): Neural Computation
- Dropout implicit regularization: https://arxiv.org/abs/2007.00823

---

# On Surprising Effectiveness of Masking Updates in Adaptive Optimizers 深度解读

Andrej，这篇 paper 真的非常 "surprising"——它直接挑战了我们从 backpropagation 时代延续至今的一个根深蒂固的假设: **既然 backward pass 一次性算出了 dense gradients, 那 dense updates 一定是最优的**。作者用一种近乎反直觉的方法证明了: 在 transformer 训练中，扔掉一半 updates 反而更好。

下面我从直觉、理论、实验三个层次详细剖析。

---

## 1. 核心动机: 为什么这个 idea 出乎意料?

考虑一下传统 optimization 视角下的反驳理由:

1. **Noise injection 视角**: 随机 mask 等价于在 update 上加 Bernoulli 噪声, 经典 convergence theory 告诉我们这会增大 stochastic variance, 收敛 worst-case rate 变差。
2. **Sample efficiency 视角**: 同样的 backward pass 计算成本, 但每个参数只收到一半的 update, 计算效率应该变差。
3. **Coordinate descent 视角**: 经典 coordinate descent (Nesterov, 2012) 选择坐标是有讲究的(通常 greedy), 随机选择收敛更慢。

但实验结果完全相反: **SkipUpdate 在 1B 参数 Llama 上比 Muon (Jordan et al., 2024) 还低 9% perplexity**。这说明在我们的传统框架里漏掉了某种重要的正则化机制。

---

## 2. SkipUpdate 算法细节

**Algorithm 1 (SkipUpdate 部分):**
```
For each block b ∈ [B]:
    s_t^(b) = 2                          # rescale factor (p=0.5)
    m_t^(b) ~ Bernoulli(0.5)              # random mask
    θ_{t+1}^(b) = θ_t^(b) - s_t^(b) · m_t^(b) · Δ_t^(b)
```

关键设计:
- $\Delta_t^{(b)} = \eta_t D_t^{(b)} g_t^{(b)}$ 是 base optimizer (e.g. RMSProp) 算出的 update
- $m_t^{(b)} \sim \text{Bernoulli}(p)$, paper 用 $p = 0.5$
- $s_t^{(b)} = 1/p = 2$ 保证 unbiasedness: $\mathbb{E}_t[\tilde{\Delta}_t^{(b)}] = \Delta_t^{(b)}$
- **Momentum states 仍然 dense 更新!** 这是 paper 一个 critical 设计 choice, §2 后半部分专门讨论为什么。

### 为什么必须 dense momentum?

对比 GaLore (Zhao et al., 2024) 这类 subspace optimizer: 它们只在 selected coordinates 上更新参数和 moment。Paper 在 Figure A2 中做了 ablation, 发现 **sparse momentum + 无 damping 直接发散**, 即使加 damping 也明显劣于 dense momentum。

直觉: dense momentum 提供了一个 **variance-reduced estimator of true momentum**, 因为 lazy update scheme 等效于把过去多次"未执行的更新"积累在 $\mu_t$ 里。这给了更稳定的搜索方向。

这与 modern LLM 训练的现实也吻合: optimizer states 占的 memory 在 activation memory 面前微不足道 (Shamshoum et al., 2025; Zhang et al., 2024b), 所以没必要省。

---

## 3. Proposition 1: 几何正则化的诞生

这是 paper 的理论核心, 我详细拆解。

### Setup

- $\theta_t \in \mathbb{R}^d$ 分成 $B$ 个 blocks $\{\theta_t^{(b)}\}_{b=1}^B$
- $g_t^{(b)} \triangleq \nabla_b l(\theta_t)$ (block b 的 stochastic gradient)
- $H_{bb'}(\theta_t)$ 是 Hessian 的 $(b,b')$ block
- $\Delta_t^{(b)}$ 是 base optimizer 给出的 block-b update

### Masked update

$$\tilde{\Delta}_t^{(b)} = s_t^{(b)} m_t^{(b)} \Delta_t^{(b)}, \quad s_t^{(b)} = 1/p$$

**注意: 期望保持 unbiased**: $\mathbb{E}_t[\tilde{\Delta}_t^{(b)}] = \Delta_t^{(b)}$

### 二阶 Taylor 展开

在 $\theta_t$ 附近展开 $l(\theta_t - \tilde{\Delta}_t)$:

$$l(\theta_t - \tilde{\Delta}_t) = l(\theta_t) - \sum_b (g_t^{(b)})^\top \tilde{\Delta}_t^{(b)} + \frac{1}{2}\sum_{b,b'} (\tilde{\Delta}_t^{(b)})^\top H_{bb'}(\theta_t) \tilde{\Delta}_t^{(b')} + R_2(\tilde{\Delta}_t)$$

其中 $R_2(\tilde{\Delta}_t) = O(\sum_b \|\tilde{\Delta}_t^{(b)}\|^3)$。

### 关键期望计算

利用 $\{m_t^{(b)}\}$ 之间的独立性:

$$\mathbb{E}_t\left[(\tilde{\Delta}_t^{(b)})^\top H_{bb'} \tilde{\Delta}_t^{(b')}\right] = \begin{cases} (\Delta_t^{(b)})^\top H_{bb'} \Delta_t^{(b')}, & b \neq b' \\ \frac{1}{p}(\Delta_t^{(b)})^\top H_{bb} \Delta_t^{(b)}, & b = b' \end{cases}$$

**为什么 $b \neq b'$ 期望不变?** 因为 $m_t^{(b)}$ 与 $m_t^{(b')}$ 独立且 $\mathbb{E}[m_t^{(b)}] = p$, 乘积期望 $p \cdot p$, 被 $1/p \cdot 1/p$ 抵消。

**为什么 $b = b'$ 期望变 $1/p$ 倍?** 因为 $m_t^{(b)} \cdot m_t^{(b)} = m_t^{(b)}$, 期望为 $p$, 被 $1/p \cdot 1/p$ 中只剩 $1/p$。

### Proposition 1 最终形式

$$\boxed{\mathbb{E}_t[l(\theta_t - \tilde{\Delta}_t)] = l(\theta_t - \Delta_t) + \sum_{b=1}^B \underbrace{\frac{1-p}{2p}(\Delta_t^{(b)})^\top H_{bb}(\theta_t)\Delta_t^{(b)}}_{\mathcal{R}_t^{(b)}: \text{geometric regularizer}} + O(\sum_b \|\Delta_t^{(b)}\|^3)}$$

变量含义:
- $p$ — survival probability (paper 中 = 0.5)
- $\Delta_t^{(b)}$ — block b 的 update direction
- $H_{bb}(\theta_t)$ — Hessian 在 $\theta_t$ 处 block b 的对角块
- $(\Delta_t^{(b)})^\top H_{bb}(\theta_t)\Delta_t^{(b)}$ — update direction 在 block b 局部曲率下的 Rayleigh quotient

### 几何解读

$\mathcal{R}_t^{(b)}$ 度量的是 **update direction $\Delta_t^{(b)}$ 在 loss 局部 sharp 方向上的投影**。

- 如果 $\Delta_t^{(b)}$ 指向高曲率方向(sharp valley wall)，则 $(\Delta_t^{(b)})^\top H_{bb} \Delta_t^{(b)}$ 大
- Minimizing expected post-update loss 等价于 implicit 惩罚 sharp-direction updates
- 这正是 flat minima 假说 (Hochreiter & Schmidhuber, 1997; Keskar et al., 2016) 想要的!

**关键 insight**: 这个正则化 **emerges from stochastic noise**, 不需要 explicit 计算 curvature! 对比:
- SAM (Foret et al., 2020) 需要额外一次 forward+backward 来近似 sharpness
- KFAC (Martens & Grosse, 2015) 需要 Kronecker-factored curvature matrix
- Magma: 零额外计算

### 为什么 block-wise 对 transformer 特别有效?

Transformer Hessian 有显著 block-diagonal structure (Kunstner et al., 2024; Ormaniec et al., 2025; Zhang et al., 2024a)。主要的 curvature interaction 在 blocks 内部, block-wise quadratic penalty 因此非常 principled。

而 element-wise masking 只惩罚 Hessian 对角元:

$$\sum_{b,i} \frac{1-p}{2p} \{\Delta_t^{(b)}\}_i^2 \{H_{bb}\}_{ii}$$

这丢失了 within-block 的 off-diagonal curvature interaction。但 paper 在 130M Llama 上做 ablation, 发现 column/element/block 几乎一样 (21.78/21.73/21.81), 都远超 RMSProp baseline (22.64)。作者解释: diagonal preconditioner (RMSProp/Adam) 本来就难以利用 dense within-block curvature, 所以 finer masking 没多大用。但 block-wise 计算上更高效 (可以整块 skip)。

---

## 4. Magma: 用 momentum-gradient alignment 调制

SkipUpdate 是 homogeneous masking。但 transformer 参数有强 heterogeneity: 不同 block 的 Hessian spectrum 差别很大 (Zhang et al., 2024a), gradient variance 也不同 (Orvieto & Gower, 2025)。

### Theoretical motivation: 在线变分推断视角

Orvieto & Gower (2025) 把 momentum 解释为 online variational inference 下的 posterior mean, 在该视角下:

$$\mathbb{P}(\mu^\top g < 0) = \Phi\left(-\frac{\|\mu\|}{\sigma}\right)$$

其中:
- $\mu$ — first moment (momentum)
- $g$ — stochastic gradient
- $\sigma$ — gradient noise std
- $\Phi$ — 标准正态 CDF

这个概率随 signal-to-noise ratio $\|\mu\|/\sigma$ **指数衰减**。意味着 negative alignment 是统计异常事件, 通常代表 destabilizing update。

### Magma 算法

**Alignment score:**
$$\tilde{s}_t^{(b)} = \text{sigmoid}\left(\frac{\text{cossim}(\mu_t^{(b)}, g_t^{(b)})}{\tau}\right)$$

变量:
- $\mu_t^{(b)}$ — block b 的 first moment estimate
- $g_t^{(b)}$ — block b 的 stochastic gradient
- $\tau$ — temperature (paper 用 2.0)
- cossim — cosine similarity (scale-invariant, 因为 LLM gradient norm 在不同 block 和 iteration 间差异巨大)

**EMA 平滑:**
$$s_t^{(b)} = 0.9 s_{t-1}^{(b)} + 0.1 \tilde{s}_t^{(b)}$$

**最终 update:**
$$\theta_{t+1}^{(b)} = \theta_t^{(b)} - s_t^{(b)} m_t^{(b)} \Delta_t^{(b)}$$

### 设计哲学

| 现象 | Magma 的反应 |
|---|---|
| $g_t^{(b)}$ 与 $\mu_t^{(b)}$ 同向 (alignment 高) | $s_t^{(b)} \to 1$, update 放行 |
| $g_t^{(b)}$ 与 $\mu_t^{(b)}$ 反向 (alignment < 0) | $s_t^{(b)} \to \text{sigmoid}(-1/\tau) \approx 0.12$, update 抑制 |

这是一种 **damping**, 引入 bias 但换取 stability (Appendix Figure A2 验证)。

### 与 Cautious Optimizer (Liang et al., 2024) 的关键区别

C-Adam 用 deterministic sign-based mask: 当 $\text{sign}(g) \neq \text{sign}(\mu)$ 时完全 mask。Magma 不同点:
1. **软调制** (sigmoid 而非 hard mask)
2. **保留 random Bernoulli masking** $m_t^{(b)}$, 因此 **依然诱导 Proposition 1 的 geometric regularization**

paper §6 强调: 没有 structured stochastic masking 就没有 curvature regularization, C-Adam 因此没有 Magma 那种 smoothing trajectory 的效果。

### 与 RPROP, MGUP 的区别

RPROP (Riedmiller & Braun, 1993) 基于 gradient sign 的 temporal consistency 调整 step size; MGUP (Chang & Yuan, 2025) 也是 momentum-gradient alignment update policy。但它们都 **缺少 structured stochastic masking**, 所以没有 Proposition 1 那种 implicit curvature regularization。

---

## 5. Convergence Theory: Theorem 6 详细解析

### Setup

考虑 constant-step SGD (隔离 masking 效果, 不让 Adam 的 adaptivity 干扰):
- Vanilla SGD: $\theta_{t+1} = \theta_t - \eta g_t$
- Magma: $\theta_{t+1} = \theta_t - \eta S_t \mathcal{M}_t(g_t)$

其中:
- $\mathcal{M}_t(g)^{(b)} \triangleq \frac{m_t^{(b)}}{p} g^{(b)}$ 是 scaled masking operator
- $S_t = s_t \otimes I_{d'}$ 是 alignment scaling (Kronecker product, 每 block 用同一个 $s_t^{(b)}$ 缩放)

### 关键定义

**Block-wise smoothness (Assumption 2):** 存在 $L^{(b)} \geq 0$ 使得
$$l(\theta + U_b u) \leq l(\theta) + u^\top \nabla_b l(\theta) + \frac{L^{(b)}}{2}\|u\|^2$$

其中 $U_b u$ 表示只在 block b 上加 $u$, 其他 block 为 0。这个 assumption 自然 capture transformer 的 heterogeneous landscape。

**Block-wise bounded variance (Assumption 3):**
$$\mathbb{E}[\|g^{(b)}(\theta)\|^2] \leq \|\nabla_b l(\theta)\|^2 + \sigma_b^2$$

**Weighted semi-norms:**
- $\|g_t\|_L^2 \triangleq \sum_b L^{(b)} \|g_t^{(b)}\|^2$
- $\|g_t\|_{\tilde{L}_t}^2 \triangleq \sum_b \tilde{L}_t^{(b)} \|g_t^{(b)}\|^2$ where $\tilde{L}_t^{(b)} \triangleq \frac{\rho_t^{(b)}}{p} L^{(b)}$
- $\sigma_{\tilde{L}_t}^2 \triangleq \sum_b \tilde{L}_t^{(b)} \sigma_b^2$

**Second-moment scaling factor:**
$$\rho_t^{(b)} \triangleq \frac{\mathbb{E}_t[\|s_t^{(b)} g_t^{(b)}\|^2]}{\mathbb{E}_t[\|g_t^{(b)}\|^2]}$$

### Lemma 4: Descent lemma

$$\mathbb{E}_t[l(\theta_{t+1})] \leq l(\theta_t) - \eta \mathbb{E}_t[g_t^\top S_t \nabla l(\theta_t)] + \frac{\eta^2}{2p}\mathbb{E}_t[\|S_t g_t\|_L^2]$$

对比 vanilla SGD (取 $p=1, S_t = I_d$):
$$\mathbb{E}_t[l(\theta_{t+1})] \leq l(\theta_t) - \eta\|\nabla l(\theta_t)\|^2 + \frac{\eta^2}{2}\mathbb{E}_t[\|g_t\|_L^2]$$

**两个变化:**
1. **First-order term 减小**: $g_t^\top S_t \nabla l \leq g_t^\top \nabla l$ a.e. 因为 $S_t \preceq I_d$ (alignment 在 (0,1))
2. **Quadratic penalty 重写**: $\frac{\eta^2}{2p}\mathbb{E}_t[\|S_t g_t\|_L^2] = \frac{\eta^2}{2}\sum_b \frac{\rho_t^{(b)} L^{(b)}}{p}\mathbb{E}_t[\|g_t^{(b)}\|^2]$

**Effective smoothness**: $L^{(b)} \mapsto \tilde{L}_t^{(b)} = \frac{\rho_t^{(b)}}{p} L^{(b)}$

### Lemma 5: Lower bound on descent term

$$\mathbb{E}[g_t^\top S_t \nabla l(\theta_t)] \geq \|(\alpha_t \otimes I_{d'}) \nabla l(\theta_t)\|^2 - \sigma_{C_t}^2$$

其中:
- $\alpha_t^{(b)} \in [\text{sigmoid}(-1/\tau), \text{sigmoid}(1/\tau)]$ — effective descent efficiency factor (block b 上保留的 descent fraction)
- $c_t^{(b)} \in [0, \text{sigmoid}(1/\tau)/2]$ — noise-descent coupling coefficient
- $\sigma_{C_t}^2 \triangleq \sum_b c_t^{(b)} \sigma_b^2$

证明思路: 定义 alignment event $E_t^{(b)} = \{\cos(g_t^{(b)}, \nabla_b l) \geq \gamma^{(b)}\}$, 用 total expectation 分情况 bound。在 alignment 高的事件下 $s_t^{(b)} \geq s_\gamma^{(b)}$, 低的事件下 $s_t^{(b)} \geq s_-$, 然后用 Cauchy-Schwarz 把 cross term bound 掉。

### Theorem 6: 全局收敛率

$$\boxed{\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}[\|\nabla l(\theta_t)\|^2] \leq \frac{2(l(\theta_0) - l_*)}{\eta \bar{\alpha}_T^{eff} T} + \frac{2\sigma_{\bar{C}}^2}{\bar{\alpha}_T^{eff}} + \frac{\eta \bar{\sigma}_{\tilde{L}}^2}{\bar{\alpha}_T^{eff}}}$$

其中:
- $\bar{\alpha}_T^{eff} \triangleq \frac{\sum_t \mathbb{E}[\|(\alpha_t \otimes I_{d'})\nabla l(\theta_t)\|^2]}{\sum_t \mathbb{E}[\|\nabla l(\theta_t)\|^2]}$ — average effective descent efficiency
- $\sigma_{\bar{C}}^2 \triangleq \frac{1}{T}\sum_t \mathbb{E}[\sigma_{C_t}^2]$ — average noise-descent coupling
- $\bar{\sigma}_{\tilde{L}}^2 = \frac{1}{T}\sum_t \mathbb{E}[\sigma_{\tilde{L}_t}^2]$ — average effective noise level

stepsize range: $\eta \in (0, \bar{\alpha}_T^{eff} / \tilde{L}_t^{max}]$

### Theorem 6 的关键 trade-off

Scaling $s_t$ 同时影响三项:
1. **$\bar{\alpha}_T^{eff}$ ↓** (descent efficiency 减少 → 收敛变慢)
2. **$\bar{\sigma}_{\tilde{L}}^2$ ↓** (effective noise floor 降低 → stationary error 减少)
3. **$\tilde{L}_t^{max}$ ↓** (effective smoothness 减少 → admissible stepsize range 扩大)

**最佳 regime**: 当高曲率 block 上 $\rho_t^{(b)} \ll 1$ 时, Magma 同时:
- (i) 扩大 stepsize range
- (ii) 降低 stationary error floor
- (iii) 增加 descent surrogate $\eta g_t^\top S_t \nabla l - \frac{\eta^2}{2p}\|S_t g_t\|_L^2 > 0$ 成立的迭代数

这正是 transformer landscape 的特征: ill-conditioned + heterogeneous, stability 由少数 high-curvature/high-variance blocks 决定。Magma 通过 alignment-based 抑制这些 blocks, 等效于选择性放宽全局 stepsize。

---

## 6. 实验数据深度解析

### Table 1: Llama 2 C4 pretraining

| Method | 60M | 130M | 350M | 1B |
|---|---|---|---|---|
| Adam | 30.79 | 24.77 | 18.42 | 16.35 |
| C-Adam | 29.70 | 23.59 | 18.58 | 15.92 |
| Adam+SGG | 30.31 | 22.18 | 17.28 | 14.30 |
| **Adam+Magma** | **29.09** | 22.08 | 16.41 | 13.71 |
| Muon | 28.93 | 22.34 | 17.09 | 14.52 |
| RMSProp | 29.29 | 22.64 | 17.47 | diverged |
| **RMSProp+Magma** | **28.55** | **21.66** | **16.16** | **13.19** |

几个值得注意的现象:

1. **RMSProp+Magma 一举成为 SOTA**, 1B 上比 Muon 低 9.1% (13.19 vs 14.52), 比 Adam 低 19.3% (13.19 vs 16.35)。
2. **RMSProp 单独在 1B 上 diverge**, 但 +Magma 后变成最好的。Magma 不仅提升性能, 还 **救活了不稳定 optimizer**。
3. **Scaling behavior**: Magma 相对优势随 model size 增大。1B 上 Magma 给 Adam 减 16.0% perplexity, 60M 上只减 5.5%。这符合 paper 的 thesis: 大模型 loss landscape 更 irregular/nonsmooth, 需要更强的 geometric regularization。
4. **Muon+Magma 应该可叠加** (paper §4.2 在 MoE 上验证了), 说明 stochastic masking 和 structured preconditioning 是 **orthogonal** 的优化维度。

### Figure 3: Heavy-tailed gradient noise

paper 用 Ahn et al. (2024) 的 controlled benchmark: linear transformer 学 in-context linear regression。通过把 covariates 从 $\mathcal{N}(0, I_d)$ 改成 $\sqrt{\Gamma_{0.1, 10}} \cdot \text{uniform}(\mathbb{S}^{d-1})$ (heavy-tailed) 诱发 heavy-tailed gradient noise。

**关键发现**: 在 light-tailed 下 Magma ≈ Adam, 但在 heavy-tailed 下 Magma **显著优于** Adam。而且 Magma 一直保持更小的 robust condition number (max eigenvalue / median eigenvalue of Hessian)。

**这是 paper 非常 beautiful 的部分**, 因为 LLM 训练的 gradient noise 几乎都是 heavy-tailed (Kunstner et al., 2024; Zhang et al., 2020), 这给出了 Magma 在 LLM 上特别有效的 mechanistic 解释。

### Figure 4: Heterogeneous quadratics

构造两个 9 维 quadratic $l(w) = \frac{1}{2}w^\top H w$, eigenvalues 都是 $\{1,2,3,99,100,101,4998,4999,5000\}$, 但分块不同:
- **Homogeneous**: $\{1,2,3\}, \{99,100,101\}, \{4998,4999,5000\}$ (每块内 scale 相近, 类似 CNN)
- **Heterogeneous**: $\{1,99,4998\}, \{2,100,4999\}, \{3,101,5000\}$ (每块内 scale 跨度大, 类似 Transformer)

结果:
- Homogeneous: Magma ≈ AdamW
- Heterogeneous: Magma 显著优于 AdamW

**对照实验**: 在 ResNet-50 + CIFAR-10 上 Magma 没改进 (94.46 vs 93.82)。这强力佐证: **Magma 的效果不是 universal 的, 而是针对 transformer-like heterogeneous landscape**。

### Ablation: Dense vs Sparse momentum (Figure A2)

这是 paper 一个 critical 设计 choice 的验证:
- Dense momentum + 无 damping: 稳定收敛, 最佳
- Dense momentum + damping: 稳定收敛, 次佳
- Sparse momentum + 无 damping: **直接发散**, perplexity 爆炸
- Sparse momentum + damping: 稳定但劣于 dense

为什么 dense momentum 这么重要? Lazy update 等效于 momentum state 累积了多次"未执行更新"的 gradient, 给出 variance-reduced 搜索方向。如果 sparse 更新 momentum, 这种 implicit variance reduction 就消失了。

### Ablation: Learning rate robustness (Figure A3)

- Adam 在 lr > 0.003 后 perplexity 爆炸
- C-Adam 同样敏感
- Adam+Magma 在 lr = 0.05 还能正常训练

这是 Theorem 6 中 $\tilde{L}_t^{max}$ 缩小的直接体现: Magma 选择性抑制 high-curvature blocks, 等效 smoothness 大幅降低, admissible stepsize 范围扩张 10× 以上。

---

## 7. 与现有方法的联系

### 与 Dropout (Srivastava et al., 2014) 的关系

Dropout 在 hidden units 上随机 mask, 已被证明 (Mianjy et al., 2018; Wei et al., 2020a; Zhang & Xu, 2024) 诱导 data-dependent weight-space regularizer。Magma 在 parameter updates 上 mask, 机制不同但精神类似: **structured stochastic perturbation → implicit regularization**。

### 与 SAM (Foret et al., 2020) 的关系

SAM 通过 adversarial perturbation 找 flat minima, 但需要额外 forward+backward。Magma 用 Bernoulli noise 实现等效的 flatness-seeking, **零额外计算**。

### 与 NEFTune (Jain et al., 2024) 的关系

NEFTune 在 instruction fine-tuning 时给 embedding 加 structured noise。Magma 是在 parameter updates 上做, 机制更深 (operates on optimization dynamics, 不是 forward pass)。

---

## 8. 我对这篇 paper 的直觉总结

**核心 insight**: 在 transformer 的 heterogeneous loss landscape 上, 标准 dense adaptive optimizer 的 update 经常 **同时包含 signal 和 high-curvature noise**。Bernoulli masking 通过 Proposition 1 的二阶效应, 把 noise 成分的 expected loss 贡献"显化"成一个 quadratic penalty, optimization 自然 bias 到 flat regions。

Magma 进一步: 用 momentum-gradient alignment 作为 **per-block signal-to-noise ratio 的代理**。低 alignment 的 block 大概率是 noise-dominated, 抑制它们既减少 effective smoothness 又减少 noise floor, 同向 alignment 的 block 放行。

为什么 transformer 特别受益:
1. Hessian block-diagonal structure 让 block-wise regularization principled
2. Heavy-tailed gradient noise 让 alignment-based filtering 高效
3. Heterogeneous block curvature (不同 layer、attention vs MLP、head 间 spectrum 差异大) 让 selective masking 有用武之地

为什么 ResNet 不受益: CNN 的 Hessian 更 homogeneous, 没有 high-curvature blocks 让 Magma 抑制。

---

## 9. 一些开放问题

paper §3 末尾提到: "Developing a stable yet unbiased masking scheme for Magma remains an important future direction." 当前 Magma 引入 bias (因为 $s_t$ 不是 $1/p$ rescale), paper 试过用 $\tilde{s}_t$ 作为 survival probability 配 $1/\tilde{s}_t$ rescale 但训练不稳定。

其他可能的方向:
- **Adaptive $\tau$**: 现在固定 $\tau = 2$, 是否应该随训练 schedule?
- **Block 自适应**: 目前 blocks 是预定义的 (attention/MLP), 是否可以 learned?
- **与其他 enhancer 叠加**: Magma 已和 Muon 正交, 是否还能和 SGG、SOAP 等叠加?
- **理论 extension**: Proposition 1 假设 Bernoulli masking, 但如果用 structured mask (e.g. low-rank Gaussian) 是否有更强的 regularization form?

---

## References

- Paper arXiv: https://arxiv.org/abs/2502.04654 (推测, 实际 paper URL 需确认)
- Cautious Optimizer (Liang et al., 2024): https://arxiv.org/abs/2411.16085
- Muon (Jordan et al., 2024): https://kellerjordan.github.io/posts/muon/
- GaLore (Zhao et al., 2024): https://arxiv.org/abs/2403.03507
- SAM (Foret et al., 2020): https://arxiv.org/abs/2010.01412
- Heavy-tailed noise analysis (Kunstner et al., 2024): https://arxiv.org/abs/2402.05820
- Transformer Hessian structure (Zhang et al., 2024a): https://arxiv.org/abs/2402.05820
- Ahn et al. linear transformer benchmark: https://arxiv.org/abs/2310.07737
- Orvieto & Gower "Adam's secret sauce": https://arxiv.org/abs/2505.21829
- Adam (Kingma & Ba, 2015): https://arxiv.org/abs/1412.6980
- SOAP (Vyas et al., 2024): https://arxiv.org/abs/2409.11321
- APOLLO (Zhu et al., 2025): MLSys 2025
- LaProp (Ziyin et al., 2020): https://arxiv.org/abs/2002.04839
- Adafactor (Shazeer & Stern, 2018): https://arxiv.org/abs/1804.04235
- Flat minima (Hochreiter & Schmidhuber, 1997): Neural Computation
- Dropout (Srivastava et al., 2014): JMLR
- Coordinate descent (Nesterov, 2012): SIAM J. Optimization
- nanoMoE: https://github.com/wolfecameron/nanoMoE
- C4 dataset (Raffel et al., 2020): JMLR
- OpenWebText: https://openwebtext.com/
- NEFTune (Jain et al., 2024): ICLR 2024
- KFAC (Martens & Grosse, 2015): ICML 2015
- AdaHessian (Yao et al., 2021): AAAI 2021
- SPAM (Huang et al., 2025): https://arxiv.org/abs/2501.06842
- SGG (Li et al., 2025): ACL 2025
- MGUP (Chang & Yuan, 2025): NeurIPS 2025
- RPROP (Riedmiller & Braun, 1993): IEEE ICNN
- SGLD (Welling & Teh, 2011): ICML 2011

希望这个深度解读对 Andrej 你 build intuition 有帮助! 这篇 paper 最让我兴奋的是它把一个看似 hacky 的 trick (扔掉一半 updates) 通过二阶 Taylor 展开 elegant 地解释成 implicit curvature regularization, 并用 momentum alignment 把"在哪些 blocks 上抑制"做成了 data-driven 的选择。整个 story 从 empirical surprise → theoretical mechanism → principled improvement, 非常 clean。
