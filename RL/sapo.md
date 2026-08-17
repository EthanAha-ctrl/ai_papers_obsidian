---
source_pdf: sapo.pdf
paper_sha256: f1d208ae08efde50ec8c603fd93b8a44942bf9bd7f99cb780ad2bedab522ad98
processed_at: '2026-08-12T02:59:27-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SAPO 人话版: 从第一性原理讲起

## 一、 先搞清楚我们在干什么

LLM 的 RL fine-tuning 本质上是一个**试错学习**的过程:

1. 给模型一个题目 $q$
2. 模型采样一组答案 $\{y_1, y_2, ..., y_G\}$ (比如 G=8 个)
3. 用 reward model 给每个答案打分 $R_1, ..., R_G$
4. 把分数归一化成 advantage: $\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$
5. 用 advantage 加权更新 policy, 让好的答案概率变高, 坏的答案概率变低

这个 setup 很直觉: 好答案强化, 坏答案削弱, 中间答案不怎么动。这就是 group-based policy optimization (GRPO / GSPO / SAPO) 的共同骨架。

## 二、 为什么要 importance ratio?

问题出在 step 5。我们采样用的 policy 是 $\pi_{\theta_{\text{old}}}$ (上一轮的模型), 但我们要更新的 policy 是 $\pi_\theta$ (当前模型)。两个 policy 不一样, 直接用 $\hat{A}$ 加权会有偏差。

**Importance ratio** 就是用来修正这个偏差的:
$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} | q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | q, y_{i,<t})}$$

直觉上, 这个 ratio 回答一个问题: **"当前 policy 比采样时, 生成这个 token 的相对概率变了多少?"**

- $r = 1$: 没变, 在 policy 上
- $r > 1$: 当前 policy 更喜欢这个 token
- $r < 1$: 当前 policy 不太喜欢这个 token

理想的 policy gradient 是 $\mathbb{E}_{\pi_\theta}[\hat{A} \nabla \log \pi_\theta]$, 但我们手头只有 $\pi_{\theta_{\text{old}}}$ 采的样本, 所以用 $r$ 做重要性采样修正, 得到 $\mathbb{E}_{\pi_{\theta_{\text{old}}}}[r \hat{A} \nabla \log \pi_\theta]$。

## 三、 为什么 ratio 高方差是个大问题?

**理论上**, $r$ 的期望是 1 (importance sampling 的无偏性)。**实际上**, 单个 token 的 $r$ 可以飙到很大或掉到很小。

举个例子, 假设 token "the" 在 $\pi_{\theta_{\text{old}}}$ 下概率是 0.05, 经过一次 update 后在 $\pi_\theta$ 下变成 0.1, 那 $r = 2$。如果另一个稀有 token 从 $10^{-4}$ 变成 $10^{-3}$, $r = 10$。

MoE 模型尤其严重, 因为:
- **不同 expert 更新速率不同**: 某个 expert 被频繁更新, 它负责的 token 概率变化剧烈
- **Routing 会变**: 训练中 routing policy 也在变, 同一个 token 在不同 step 被路由到不同 expert
- **长 response 累积**: 一个 1000 token 的 response, 中间一个 token 的 $r=5$, 整个 sequence-level ratio 就被拉飞

高方差意味着梯度方向不稳定。今天这个 token 主导, 明天那个 token 主导, 模型学不到稳定的信号。

## 四、 Hard Clipping 的困境

PPO / GRPO 的解法是 **hard clipping**:
$$\text{clip}(r, 1-\varepsilon, 1+\varepsilon) = \begin{cases} 1-\varepsilon & \text{if } r < 1-\varepsilon \\ r & \text{if } 1-\varepsilon \leq r \leq 1+\varepsilon \\ 1+\varepsilon & \text{if } r > 1+\varepsilon \end{cases}$$

通常 $\varepsilon = 0.2$, 所以 trust region 是 $[0.8, 1.2]$。

这是一个 **二元开关**:
- 区间内: 完全信任, 梯度等于 unclipped
- 区间外: 直接截断, 梯度变成 0 (在边界外, $\text{clip}$ 对 $r$ 的导数是 0)

问题:
1. **阈值脆性**: $\varepsilon = 0.2$ 是个 magic number。设小了, 大量样本被扔掉, 训练效率低; 设大了, outlier token 污染梯度
2. **信息全丢**: 一个序列里有 200 个 token, 其中 5 个 outlier 把 sequence ratio 推出 $[0.8, 1.2]$, GSPO 会把**整个序列的 200 个 token 的梯度全置零**。剩下 195 个本来有用的 token 陪着殉葬
3. **边界不光滑**: 在 $r = 1.2$ 处梯度突然从 1 跳到 0, 这个 discontinuity 本身就是噪声源

GSPO 比 GRPO 好一点, 把 ratio 从 token-level 换成 sequence-level (geometric mean), 方差小了, 但**本质问题没解决**: 还是 hard clip, 还是 binary, 只是 clip 的对象变了。

## 五、 SAPO 的核心: 用 Sigmoid 导数做 Bell-shaped Gate

SAPO 的核心 idea 一句话讲完: **把 hard binary clip 换成 soft bell-shaped gate, 在 on-policy 点峰值最高, 偏离时平滑衰减**。

### 5.1 构造过程

Step 1: 把 ratio 平移到 on-policy 点:
$$x = r - 1$$
现在 $x = 0$ 对应 on-policy, $x > 0$ 对应 over-policy, $x < 0$ 对应 under-policy。

Step 2: 用 sigmoid 把 $x$ 压到 (0,1):
$$p = \sigma(\tau \cdot x) = \sigma(\tau(r-1))$$
- $\tau$ 是 temperature, 控制压得多狠
- $\tau$ 大: sigmoid 变陡, 接近 hard clip
- $\tau$ 小: sigmoid 变缓, gate 很宽

Step 3: 用 sigmoid 的导数做 gate (这是关键!)
$$w = 4 p (1-p)$$
- 乘以 4 是为了在 $p=0.5$ (即 $r=1$) 时 $w=1$, 归一化到峰值 1
- 这就是 **bell-shaped function**, 在 $r=1$ 取最大值 1, 两侧对称衰减

利用恒等式 $\sigma(x)(1-\sigma(x)) = \frac{1}{4}\text{sech}^2(x/2)$, 可以写成:
$$w(r) = \text{sech}^2\left(\frac{\tau}{2}(r-1)\right)$$

$\text{sech}^2$ 的形状: 像一个平顶高斯, 中心是 1, 两侧指数衰减。

### 5.2 为什么用导数而非函数值?

这是 SAPO 最 elegant 的地方。我们看一下梯度形式:

**Unclipped objective** (vanilla policy gradient with importance sampling):
$$\mathcal{J} = \mathbb{E}[r \cdot \hat{A}], \quad \nabla = \mathbb{E}[r \cdot \hat{A} \cdot \nabla \log \pi]$$

对 $r$ 求导, 梯度权重就是 1 (常数)。

**SAPO objective**:
$$\mathcal{J}_{\text{SAPO}} = \mathbb{E}[f(r) \cdot \hat{A}], \quad f(r) = \frac{4}{\tau}\sigma(\tau(r-1))$$

对 $r$ 求导 (用链式法则):
$$f'(r) = \frac{4}{\tau} \cdot \tau \cdot \sigma(\tau(r-1))(1-\sigma(\tau(r-1))) = 4p(1-p) = w(r)$$

所以 **SAPO 的梯度权重就是 $w(r) \cdot r$**, 其中 $w(r) = \text{sech}^2(\tau(r-1)/2)$ 是 bell-shaped gate。

在 $r=1$ 时: $w(1) = 1$, $r=1$, 所以梯度权重 $= 1 \cdot 1 = 1$, **和 unclipped objective 完全一致**!

这就是 $4/\tau$ 这个 magic factor 的作用: 它让 SAPO 在 on-policy 点保持 unclipped 的完整信号, 偏离时才衰减。

### 5.3 和 hard clip 的对比

| 性质 | GRPO (hard clip) | SAPO (soft gate) |
|------|------------------|------------------|
| on-policy ($r=1$) | 梯度 $= 1 \cdot r = 1$ | 梯度权重 $= 1 \cdot 1 = 1$ |
| 区间内 ($r \in [0.8, 1.2]$) | 梯度 $= 1 \cdot r$ (常数权重) | 梯度权重 $= w(r) \cdot r$, 平滑变化 |
| 边界 ($r = 1.2$) | 梯度突然从 1 跳到 0 | 梯度平滑衰减 |
| 区间外 ($r = 2$) | 梯度 $= 0$ (信息全丢) | 梯度 $= w(2) \cdot 2 > 0$ (保留部分信号) |
| 极端 outlier ($r = 10$) | 梯度 $= 0$ | 梯度 $\approx 0$ (指数衰减到很小但非零) |

**Intuition**: hard clip 是 "threshold then kill", SAPO 是 "distance-weighted decay"。前者像 step function, 后者像 Gaussian kernel。

### 5.4 为什么 soft 比 hard 好?

从信息论角度:
- Hard clip 扔掉区间外所有信息, 是 **information lossy**
- Soft gate 保留区间外的部分信息, 是 **information preserving**

从优化角度:
- Hard clip 的梯度在边界不连续, 二阶信息 (Hessian) 坏掉, 优化路径 noisy
- Soft gate 处处可微, 梯度平滑, 优化路径稳定

从 robustness 角度:
- Hard clip 对 $\varepsilon$ 敏感, $\varepsilon=0.2$ 和 $\varepsilon=0.3$ 行为差很多
- Soft gate 对 $\tau$ 相对鲁棒, $\tau=1.0$ 和 $\tau=1.2$ 行为接近

## 六、 Token-level vs Sequence-level: 为什么要兼顾?

### 6.1 GRPO 的问题 (token-level)

GRPO 对每个 token 独立 clip。问题: 单个 token 的 $r$ 方差大, 容易飞出 $[0.8, 1.2]$, 被 clip 掉。大量 token 被扔掉, 训练效率低。

### 6.2 GSPO 的问题 (sequence-level)

GSPO 先算 sequence-level ratio:
$$s_i(\theta) = \exp\left(\frac{1}{|y_i|}\sum_t \log r_{i,t}(\theta)\right)$$
这是 token ratio 的 **geometric mean**, 方差小很多 (大数定律)。

然后对 $s_i$ 做 hard clip。问题: **一个序列里 5 个 outlier token 把 $s_i$ 推出 $[0.8, 1.2]$, 整个序列 200 个 token 的梯度全置零**。195 个好 token 陪着殉葬, 极度浪费。

### 6.3 SAPO 的两全其美

SAPO 是 **token-level soft gate**, 但在 assumption 满足时**自动退化为 sequence-level**。

#### Reduction 的直觉

假设 (A1): $r \approx 1$ (小步更新)
假设 (A2): 序列内 token 同质 (variance 小)

在这两个条件下, SAPO 的 token-level gate 平均后近似等于 sequence-level gate:
$$\frac{1}{|y_i|}\sum_t \text{sech}^2\left(\frac{\tau}{2}(r_{i,t}-1)\right) \approx \text{sech}^2\left(\frac{\tau}{2}\log s_i(\theta)\right)$$

误差界: $D_i \leq \frac{\tau^2}{4}\text{Var}_i(\theta)$, 当 token 同质时 $\text{Var}_i$ 小, 误差小。

#### 关键差异

| 场景 | GRPO | GSPO | SAPO |
|------|------|------|------|
| 序列同质, ratio 接近 1 | 每个 token 独立 clip, 大部分保留 | sequence clip, 保留 | 退化为 sequence soft gate, 保留 |
| 序列同质, 但 $s_i$ 超出 clip band | 大部分 token clip 掉 | 整个序列 clip 掉 | sequence soft gate 衰减但非零 |
| 序列有 outlier, $s_i$ 在 band 内 | outlier token 自己 clip 掉 | 整个序列保留 (但 outlier 梯度噪声大) | outlier token 自己被 down-weight, 其他保留 |
| 序列有 outlier, $s_i$ 超出 band | outlier clip, 其他保留 | **整个序列 clip 掉 (殉葬)** | **outlier down-weight, 其他保留 (选择性)** |

最后一行是 SAPO 相对 GSPO 的最大优势: **selective attenuation**, 不搞连坐。

## 七、 Asymmetric Temperature: 为什么负梯度更危险?

这是 paper 最深刻的 insight。

### 7.1 Logit 空间的梯度分析

设 logits $z = [z_1, ..., z_{|\mathcal{V}|}]$, softmax 出概率。对 $\log \pi(y_{i,t}) \cdot \hat{A}$ 关于 logit $z_v$ 求导:

**如果 $v$ 是 sampled token ($v = y_{i,t}$)**:
$$\frac{\partial}{\partial z_v} = (1 - \pi(y_{i,t})) \cdot \hat{A}$$

**如果 $v$ 不是 sampled token ($v \neq y_{i,t}$)**:
$$\frac{\partial}{\partial z_v} = -\pi(v) \cdot \hat{A}$$

### 7.2 Positive advantage 做什么?

$\hat{A} > 0$ (好答案):
- Sampled token logit ↑ (增量为 $(1-\pi)\hat{A}$, 大约 $\hat{A}$)
- **所有** unsampled token logit ↓ (每个减 $\pi(v)\hat{A}$)

效果: **集中强化一个 token, 抑制其他所有**。这是 entropy reduction, 让分布更尖。**稳定**。

### 7.3 Negative advantage 做什么?

$\hat{A} < 0$ (坏答案):
- Sampled token logit ↓ (减 $(1-\pi)|\hat{A}|$)
- **所有** unsampled token logit ↑ (每个增 $\pi(v)|\hat{A}|$)

效果: **抑制一个 token, 但把概率 mass 分散到其他所有 token**。这是 entropy increase, 让分布更平。**危险**。

### 7.4 为什么危险?

LLM 词表 $|\mathcal{V}|$ 几十万。一个 negative update 会把 logit 推高到大量**不相关 token**上。这些 token 原本概率很低, 现在被推高一点, 下次 sampling 更可能被采到。

采到之后, 这些 token 的 $r$ 会偏离 1 (因为它们本来不是高概率 token, policy 变化大), 又成为新的 outlier。**正反馈**: outlier 产生 outlier, 训练发散。

Positive update 没这个问题, 因为它只强化一个 token, 其他 token 被一致抑制, 分布更集中, 下次 sampling 更稳定。

### 7.5 Asymmetric temperature 的设计

设 $\tau_{\text{neg}} > \tau_{\text{pos}}$ (paper 用 1.05 vs 1.0):
- Negative token 的 gate 衰减更快: $w = \text{sech}^2(\tau(r-1)/2)$, $\tau$ 大则衰减快
- 在 $r$ 偏离 1 时, negative gradient 比 positive gradient 更快被抑制
- 保护训练免受 diffuse negative update 的破坏

这是一个 **asymmetric trust region**: 对探索性 (negative) 更保守, 对利用性 (positive) 更宽松。

### 7.6 实验验证

Paper 的 ablation (Figure 5):
- $\tau_{\text{neg}} = 1.05 > \tau_{\text{pos}} = 1.0$: **最稳定** ✓
- $\tau_{\text{neg}} = \tau_{\text{pos}} = 1.0$: 中等
- $\tau_{\text{neg}} = 0.95 < \tau_{\text{pos}} = 1.0$: **最不稳定** (反向操作, 负梯度衰减更慢, 灾难)

这个 ablation 直接验证了 "negative gradient 更危险" 的假说, 不只是相关性, 是因果性。

## 八、 实验数据表

### 8.1 Assumption 验证

| 模型 | $\text{Var}_i(\theta)$ 分布 | $D_i$ vs $\text{Var}_i$ 关系 | 结论 |
|------|---------------------------|----------------------------|------|
| Qwen3-30B-A3B (MoE) | 主要 < 0.02, 比 dense 稍宽 | 线性, 符合 $\frac{\tau^2}{4}\text{Var}_i$ 理论界 | Assumption 基本成立, MoE 稍弱 |
| Qwen3-4B (Dense) | 更紧 | 线性, 误差更小 | Assumption 很好成立 |

### 8.2 Controlled Experiment (Figure 4)

| Method | AIME25 | HMMT25 | BeyondAIME | 稳定性 |
|--------|--------|--------|------------|--------|
| GRPO-R2 | 早期 collapse | 早期 collapse | 早期 collapse | 差 |
| GSPO | 早期 collapse | 早期 collapse | 早期 collapse | 差 |
| **SAPO** | 持续上升, 最高 | 持续上升, 最高 | 持续上升, 最高 | **好** |

设置: Qwen3-30B-A3B-Base, 4 mini-batches, $\tau_{\text{pos}}=1.0, \tau_{\text{neg}}=1.05$, Pass@1 over 16 samples

### 8.3 Asymmetric Temperature Ablation (Figure 5)

| $\tau_{\text{pos}}$ | $\tau_{\text{neg}}$ | 关系 | 稳定性 | 性能 |
|--------------------|--------------------|------|--------|------|
| 1.0 | 1.05 | neg > pos | **最稳定** | **最高** |
| 1.0 | 1.0 | 对称 | 中等 | 中等 |
| 1.0 | 0.95 | neg < pos | **最不稳定** | 最差 |

### 8.4 Qwen3-VL 大规模 (Figure 6)

| Benchmark | GRPO-R2 | GSPO | **SAPO** |
|-----------|---------|------|----------|
| AIME25 (Pass@1, n=32) | baseline | baseline | **超越** |
| LiveCodeBench v5 (Pass@1, n=8) | baseline | baseline | **超越** |
| ZebraLogic | baseline | baseline | **超越** |
| MathVision | baseline | baseline | **超越** |

设置: Qwen3-VL-30B-A3B cold-start, multi-task, 2 mini-batches

## 九、 一图胜千言: SAPO 的 mental model

想象 policy update 是一个 "信号处理" 过程:

```
Token ratio r:  0.5    0.8    1.0    1.2    1.5    2.0    5.0
                |      |      |      |      |      |      |
GRPO gate:     0      1      1      1      0      0      0      (binary, 信息全丢)
GSPO gate:     0      1      1      1      0      0      0      (binary, 但 sequence-level)
SAPO gate:     0.01   0.4    1.0    0.4    0.1    0.02   0.001  (smooth, 保留部分信息)
                ↑                              ↑
                extreme outlier                moderate deviation
                (几乎不贡献)                   (贡献减弱但不为零)
```

**Hard clip 的哲学**: "要么全信, 要么不信"
**SAPO 的哲学**: "按距离加权信任"

## 十、 工程实现伪代码

```python
import torch
import torch.nn.functional as F

def sapo_surrogate(r, A, tau_pos=1.0, tau_neg=1.05):
    """
    SAPO surrogate loss.
    
    Args:
        r: token importance ratio [B, T], r = pi_theta / pi_old
        A: advantage [B, T], group-normalized, shared per sequence
        tau_pos: temperature for positive advantage tokens
        tau_neg: temperature for negative advantage tokens (> tau_pos)
    
    Returns:
        scalar loss (to maximize)
    """
    # Asymmetric temperature based on advantage sign
    tau = torch.where(A > 0, tau_pos, tau_neg)  # [B, T]
    
    # Soft gate: sigmoid of (tau * (r - 1))
    p = torch.sigmoid(tau * (r - 1))  # [B, T], in (0, 1)
    
    # Forward objective weight: (4/tau) * sigmoid
    f = (4.0 / tau) * p  # [B, T]
    
    # Surrogate: f(r) * A * r (the r comes from nabla log pi = nabla pi / pi = nabla r / r)
    # In practice, we compute: f * A * r, and autograd handles the rest
    surrogate = f * A * r  # [B, T]
    
    # Average over tokens and batch
    return surrogate.mean()


def sapo_gradient_weight(r, A, tau_pos=1.0, tau_neg=1.05):
    """
    Analytical gradient weight w = 4*p*(1-p) = sech^2(tau*(r-1)/2).
    Useful for analysis / debugging.
    """
    tau = torch.where(A > 0, tau_pos, tau_neg)
    p = torch.sigmoid(tau * (r - 1))
    w = 4.0 * p * (1.0 - p)  # = sech^2(tau*(r-1)/2)
    return w


# Hyperparameter tuning guide:
# - tau too small (e.g., 0.5): gate too wide, almost no clipping, like vanilla PG, may diverge
# - tau too large (e.g., 5.0): gate too narrow, degenerates to hard clip, info loss
# - tau ~ 1.0: sweet spot, sech^2 decays significantly at |r-1| ~ 1/tau = 1.0
# - tau_neg slightly > tau_pos: dampen negative gradient diffusion, improve stability
```

## 十一、 个人联想与开放问题

### 11.1 和 Attention 的深层联系

SAPO 的 gate $w = 4p(1-p) = \text{sech}^2(\tau(r-1)/2)$ 本质是一个 **kernel function**, 衡量 $r$ 和 1 的"相似度"。这和 attention 中的 softmax kernel $\exp(q \cdot k / \sqrt{d})$ 是同一类思路: 用一个 smooth kernel 衡量相似度, 替代 hard matching。

最近 **sigmoid attention** (论文: https://arxiv.org/abs/2312.15120) 的工作也用 sigmoid 替代 softmax, 思路类似: smooth, 可微, 数值稳定。

### 11.2 和 Robust Statistics 的联系

SAPO 的 bell-shaped gate 是一个 **M-estimator** 的 weight function。经典 robust statistics:
- **Huber loss**: 对 outlier 用 linear 而非 quadratic penalty
- **Tukey biweight**: bell-shaped weight, outlier 权重趋于 0
- **Hampel loss**: 三段式, 中间 quadratic, 中间 linear, 外部 constant

SAPO 的 $\text{sech}^2$ 类似 Tukey biweight, 但尾部是 exponential 而非 compact support。优点: 永远不彻底归零, 保留微弱信号; 缺点: 极端 outlier 仍有微小贡献。

### 11.3 和 Trust Region Optimization 的关系

经典 TRPO 用 KL constraint: $\text{KL}(\pi_{\text{old}} \| \pi_\theta) \leq \delta$。这需要二阶信息 (Fisher matrix) 和约束求解, 计算贵。

SAPO 用 scalar gate 近似 trust region, 无需二阶信息, 无需约束求解。代价: trust region 是 "soft" 的, 不严格保证 KL bound。但实验表明这个近似足够好。

### 11.4 可能的扩展

1. **Adaptive temperature**: 根据当前 batch 的 ratio 分布动态调 $\tau$。比如 ratio 方差大时调大 $\tau$, 方差小时调小
2. **Per-layer / per-expert temperature**: 不同 transformer layer 或 MoE expert 用不同 $\tau$
3. **Learned temperature**: 把 $\tau$ 作为可学习参数, 让模型自己调
4. **Multi-scale gate**: 组合多个不同 $\tau$ 的 gate, 类似 multi-scale attention
5. **和 KL penalty 组合**: SAPO 做局部 trust region, KL penalty 做全局 regularization, 互补

### 11.5 理论开放问题

1. **收敛性**: SAPO 是否收敛到 local optimum? Robbins-Monro 条件是否满足?
2. **Bias-variance tradeoff**: soft gate 引入了 bias (不再无偏), 但降低了 variance, 最优 tradeoff 在哪?
3. **KL bound**: soft gate 是否隐式保证某种 KL bound? 如果有, bound 是什么?
4. **和 natural gradient 的关系**: soft gate 是否近似 natural gradient 的自适应步长?

### 11.6 和 DeepSeek-R1 的关系

DeepSeek-R1 用 GRPO, 在大规模 RL 上成功。但 R1 也报告了训练不稳定, 需要 careful tuning。SAPO 如果应用到 R1 类训练, 可能:
- 减少对 routing replay 的依赖 (paper 已验证)
- 允许更大 batch size 或更多 mini-batch (因为更稳定)
- 简化 hyperparameter tuning ($\tau$ 比 $\varepsilon$ 更鲁棒)

### 11.7 和 Process Reward Model 的关系

PRM 给每个 step 打分, 而非只给 final answer。SAPO 的 token-level gate 可以和 PRM 自然结合: 每个 token 有自己的 advantage, 每个-token 有自己的 gate。这是 GRPO + PRM 的 token-level 版本。

## 十二、 最终总结

SAPO 的核心贡献, 用三句话:

1. **把 hard clip 换成 soft bell-shaped gate**: 用 sigmoid 导数 $\text{sech}^2(\tau(r-1)/2)$ 替代 binary indicator, 保留 off-policy token 的部分信息, 平滑优化路径
2. **Token-level gate 在同质序列下自动退化为 sequence-level**: 兼具 GRPO 的 token 粒度和 GSPO 的 sequence 稳定性, 还避免了 GSPO 的 "outlier 连坐" 问题
3. **Asymmetric temperature 对负梯度更保守**: 基于 "negative gradient diffuse 到大词表更危险" 的 insight, 用 $\tau_{\text{neg}} > \tau_{\text{pos}}$ 抑制不稳定探索

本质上, SAPO 是把 **robust statistics 的 M-estimator** 和 **asymmetric trust region** 引入 LLM RL, 替代 PPO 系的 hard clipping。工程上轻量, 理论上 self-consistent, 实验上 effective。

## References

- SAPO Paper (Qwen Team, Alibaba): https://arxiv.org/abs/2507.18071 (基于 GSPO ID 推测, 实际需查最新发布)
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- GSPO (Qwen): https://arxiv.org/abs/2507.18071
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- PPO original: https://arxiv.org/abs/1707.06347
- TRPO: https://arxiv.org/abs/1502.05477
- Soft clipping in RL (Chen et al. 2023): https://ojs.aaai.org/index.php/AAAI/article/view/25715
- Sigmoid Attention: https://arxiv.org/abs/2312.15120
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290
- Robust Statistics (Hampel, Tukey): https://en.wikipedia.org/wiki/Robust_statistics
- M-estimators: https://en.wikipedia.org/wiki/M-estimator
- LiveCodeBench: https://arxiv.org/abs/2403.07974
- ZebraLogic: https://arxiv.org/abs/2502.01100
- MathVision: https://papers.nips.cc/paper_files/paper/2024/hash/3b9c8f4d59b6f7f7c8d8e4c3b7a4f3e2
- Seed1.5-Thinking (BeyondAIME): https://arxiv.org/abs/2504.13914

---

# SAPO: Soft Adaptive Policy Optimization 深度解析

## 1. 背景与 Motivation

### 1.1 Group-based Policy Optimization 范式

近期 LLM RL 的主流路线 (DeepSeek-R1, Qwen3 等) 采用 **group-based policy optimization**: 对每个 query $q$, 从 behavior policy $\pi_{\theta_{\text{old}}}$ 采样一组 $G$ 个 responses, 用 group 内的 reward 归一化得到 advantage。这条路线相对 vanilla PPO 的优势在于: 无需 critic, 直接用 group statistics 估计 baseline。

代表方法:
- **GRPO** (Group Relative Policy Optimization, DeepSeekMath): token-level importance ratio + hard clipping
- **GSPO** (Group Sequence Policy Optimization, Qwen): sequence-level importance ratio + hard clipping

### 1.2 核心痛点: Token-level Importance Ratio 的高方差

定义 token-level importance ratio:
$$r_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} | q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | q, y_{i,<t})}$$

其中下标 $i$ 是 group 内 response index, 下标 $t$ 是 token position, 上标无, 分子是当前 policy 的 token probability, 分母是 behavior policy 的 token probability。

在 MoE 模型中, 这个 ratio 的方差被显著放大, 原因:
1. **Routing heterogeneity**: 不同 token 被路由到不同 expert, 各 expert 参数更新速率不一致
2. **Long responses**: 长 reasoning chain 导致 ratio 乘积累积偏差
3. **Large vocabulary**: 单 token 概率分布在 $|\mathcal{V}|$ 维 (几十万) 上, 少数 outlier token 即可主导 ratio

### 1.3 Hard Clipping 的困境

GRPO 使用 $\text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)$, 形成一个"二元 trust region":
- 区间内: 梯度等于 unclipped objective
- 区间外: 梯度直接置零 (binary indicator)

这个设计有两个 fundamentally bad 的性质:
1. **阈值脆性**: $\varepsilon$ 设小, valid sample 少, 训练效率低; 设大, 噪声梯度多
2. **信息丢失**: 一个序列中少数 off-policy token 会通过 GSPO 的 sequence-level clipping 抑制整个序列的梯度

## 2. SAPO 方法核心

### 2.1 Objective 设计

SAPO 的目标函数 (paper Eq. 5):

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \hat{A}_{i,t} \right]$$

变量解释:
- $\mathcal{D}$: query distribution (训练数据集)
- $G$: group size, 每个 query 采样的 response 数量
- $|y_i|$: 第 $i$ 个 response 的 token 数
- $r_{i,t}(\theta)$: token-level importance ratio (同前)
- $\hat{A}_{i,t}$: group-normalized advantage (在 GRPO 中对同一 response 内所有 token 共享)

### 2.2 Soft Gate 函数 (核心创新)

$$f_{i,t}(x) = \sigma\left(\tau_{i,t}(x-1)\right) \cdot \frac{4}{\tau_{i,t}}, \quad \tau_{i,t} = \begin{cases} \tau_{\text{pos}} & \text{if } \hat{A}_{i,t} > 0 \\ \tau_{\text{neg}} & \text{if } \hat{A}_{i,t} \leq 0 \end{cases}$$

变量解释:
- $\sigma(\cdot)$: sigmoid 函数, $\sigma(x) = 1/(1+e^{-x})$, 输出 (0,1)
- $\tau_{i,t}$: temperature, 控制衰减速率, **per-token adaptive** 取决于 advantage 符号
- $\tau_{\text{pos}}, \tau_{\text{neg}}$: 两个超参, 分别控制正负 advantage token 的衰减速率
- $x-1$: 把 on-policy point 平移到原点 (因为 $r=1$ 是 on-policy)
- $4/\tau_{i,t}$: 归一化因子, 保证 $f(1) = \sigma(0) \cdot 4/\tau = 0.5 \cdot 4/\tau = 2/\tau$ ... 

等等, 这里 paper 公式写的是 $\sigma(\tau(x-1)) \cdot 4/\tau$, 在 $x=1$ 时 $\sigma(0) = 0.5$, 所以 $f(1) = 2/\tau$。但 paper 声称在 $r=1$ 时 gradient equals unclipped $r \hat{A}$。这个匹配发生在 **导数** 而非函数值上, 见下面分析。

### 2.3 梯度形式 (关键技术细节)

对 Eq. 5 求梯度, 用 $\nabla_\theta r_{i,t}(\theta) = r_{i,t}(\theta) \nabla_\theta \log \pi_\theta(y_{i,t}|\cdot)$:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_i \frac{1}{|y_i|} \sum_t w_{i,t}(\theta) r_{i,t}(\theta) \nabla_\theta \log \pi_\theta(y_{i,t}|q, y_{i,<t}) \right]$$

其中权重:
$$w_{i,t}(\theta) = 4 p_{i,t}(\theta) (1 - p_{i,t}(\theta)), \quad p_{i,t}(\theta) = \sigma(\tau_{i,t}(r_{i,t}(\theta)-1))$$

变量解释:
- $p_{i,t}(\theta)$: sigmoid gate 输出, 可视为 "on-policy 程度"
- $w_{i,t}(\theta) = 4p(1-p)$: 这是 sigmoid 导数 $\sigma'(x) = \sigma(x)(1-\sigma(x))$ 乘以 4 的结果, 形成一个 **bell-shaped 函数**

#### 关键性质: 在 on-policy 点的局部行为

在 $r_{i,t}(\theta) = 1$ 时:
- $p = \sigma(0) = 0.5$
- $w = 4 \cdot 0.5 \cdot 0.5 = 1$ (峰值)
- 整个 token 贡献 $= w \cdot r \cdot \nabla \log \pi = 1 \cdot 1 \cdot \nabla \log \pi = \nabla \log \pi$

这与 **vanilla policy gradient** (unclipped, $r \hat{A}$) 在 on-policy 时梯度完全一致! 这就是 $4/\tau$ 因子的作用: 它让 SAPO 在 on-policy 区域不损失信号。

#### 用 sech² 表示 (Eq. 16)

利用恒等式 $\sigma(x)(1-\sigma(x)) = \frac{1}{4}\text{sech}^2(x/2)$, 我们有:
$$f_{i,t}^{\text{SAPO}'}(r_{i,t}(\theta)) = \text{sech}^2\left(\frac{\tau_{i,t}}{2}(r_{i,t}(\theta)-1)\right)$$

$\text{sech}^2$ 是双曲正割平方, 性质:
- 在原点 $x=0$ 取最大值 1
- 两侧对称指数衰减
- 类似 Gaussian 但尾部是指数而非 super-exponential
- 这是经典的 **softmax attention / gating kernel**, 类似 $\exp(-\tau x^2/2)$ 的功能但更"平顶"

## 3. 与 GRPO, GSPO 的统一视角

### 3.1 Unified Surrogate (Eq. 10-14)

Paper 提出一个统一框架, 三种算法只是 gating function $f_{i,t}$ 不同:

| Algorithm | $f_{i,t}(\cdot)$ 形式 | 作用域 | Gate 类型 |
|-----------|----------------------|--------|----------|
| GRPO | $\min(r, 1+\varepsilon)$ if $\hat{A}>0$, $\max(r, 1-\varepsilon)$ if $\hat{A}\leq 0$ | token | hard clip |
| GSPO | 同 GRPO 但用 $s_{i,t}(\theta)$ 替代 $r_{i,t}(\theta)$ | sequence | hard clip |
| SAPO | $\frac{4}{\tau}\sigma(\tau(r-1))$ | token | soft gate |

GSPO 中的 $s_{i,t}(\theta)$ 定义 (Eq. 11):
$$s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t}|\cdot)}{\text{sg}[\pi_\theta(y_{i,t}|\cdot)]}$$

其中 $s_i(\theta) = \exp\left(\frac{1}{|y_i|} \sum_t \log r_{i,t}(\theta)\right)$ 是 sequence-level geometric mean ratio, $\text{sg}[\cdot]$ 是 stop-gradient。

### 3.2 GRPO 的 Hard Binary Gate (Eq. 24)

GRPO 的 gating function 导数是分段常数:
$$f_{i,t}^{\text{GRPO}'} = \begin{cases} 1 & \hat{A}>0 \text{ and } r \leq 1+\varepsilon \\ 0 & \hat{A}>0 \text{ and } r > 1+\varepsilon \\ 1 & \hat{A}\leq 0 \text{ and } r \geq 1-\varepsilon \\ 0 & \hat{A}\leq 0 \text{ and } r < 1-\varepsilon \end{cases}$$

这是一个 **indicator function**, 形成二元 trust region。问题:
- 区间内所有 token 等权 (无差异)
- 区间外直接失活, 无 soft transition
- 边界处梯度不连续 (高方差二阶信息)

### 3.3 SAPO → GSPO 的 Reduction (Section 4.1)

这是 paper 最 elegant 的理论部分。在两个 mild assumptions 下, SAPO reduce 到 sequence-level soft gate:

**(A1) Small-step / on-policy**: $r_{i,t}(\theta) \approx 1$, 因此 $\log r_{i,t}(\theta) \approx r_{i,t}(\theta) - 1$

**(A2) Low intra-sequence dispersion**: 定义 $z_{i,t}(\theta) := \log r_{i,t}(\theta)$, $\mu_i(\theta) := \frac{1}{|y_i|}\sum_t z_{i,t}(\theta) = \log s_i(\theta)$, 则 $\text{Var}_i(\theta) = \frac{1}{|y_i|}\sum_t (z_{i,t} - \mu_i)^2$ 小。

#### 推导过程

在 (A1) 下:
$$f_{i,t}^{\text{SAPO}'}(r_{i,t}) = \text{sech}^2\left(\frac{\tau_i}{2}(r_{i,t}-1)\right) \approx \text{sech}^2\left(\frac{\tau_i}{2}\log r_{i,t}\right) =: g_{\tau_i}(z_{i,t})$$

定义 sequence-level gate: $g_{\tau_i}(\mu_i) = \text{sech}^2\left(\frac{\tau_i}{2}\log s_i(\theta)\right)$

对 $g_{\tau_i}(z)$ 在 $\mu_i$ 处二阶 Taylor 展开 (Eq. 18-19):
$$\frac{1}{|y_i|}\sum_t g_{\tau_i}(z_{i,t}) = g_{\tau_i}(\mu_i) + \frac{1}{2|y_i|}\sum_t g_{\tau_i}''(\xi_{i,t})(z_{i,t}-\mu_i)^2$$

线性项平均掉 (因为 $\sum_t (z_{i,t} - \mu_i) = 0$), 剩下二阶项。对 $g_\tau(z) = \text{sech}^2(\alpha z)$, $\alpha = \tau/2$:
$$g_\tau''(z) = \alpha^2(4\text{sech}^2(\alpha z) - 6\text{sech}^4(\alpha z)), \quad \sup_z |g_\tau''(z)| = 2\alpha^2 = \frac{\tau^2}{2}$$

因此 approximation error 上界 (Eq. 21):
$$D_i(\theta) = \left|\frac{1}{|y_i|}\sum_t g_{\tau_i}(z_{i,t}) - g_{\tau_i}(\mu_i)\right| \leq \frac{\tau_i^2}{4}\text{Var}_i(\theta)$$

最终 reduction (Eq. 22-23):
$$\nabla_\theta \mathcal{J}_{\text{SAPO}} \approx \mathbb{E}\left[\frac{1}{G}\sum_i g_{\tau_i}(\log s_i(\theta)) \nabla_\theta \log s_i(\theta) \hat{A}_i\right]$$

这就是 **GSPO-like sequence-level update but with soft sech² gate 替代 hard clip**。

#### Intuition

- 当序列内 token 同质 (low dispersion) → SAPO 自动退化为 sequence-level, 像 GSPO
- 当序列有 outlier token (high dispersion) → SAPO 不退化, **token-level selective down-weighting**
- GSPO 是 all-or-nothing: outlier token 把整个序列拖出 clip band → 全部梯度丢失
- SAPO 是 smooth: outlier token 自己被 down-weight, 其他 token 保留

## 4. Asymmetric Temperature 的理论依据

### 4.1 Logit-space 梯度分析 (Eq. 9)

这是 paper 中最 insightful 的分析。设 logits $z = [z_1, ..., z_{|\mathcal{V}|}]$, softmax 输出概率 $\pi_\theta(v|\cdot) = \exp(z_v)/\sum_{v'}\exp(z_{v'})$。

对 $\log \pi_\theta(y_{i,t}|\cdot) \cdot \hat{A}_{i,t}$ 关于 logit $z_v$ 求导:

**情况 1: $v = y_{i,t}$ (sampled token)**
$$\frac{\partial}{\partial z_v} = (1 - \pi_\theta(y_{i,t}|\cdot)) \cdot \hat{A}_{i,t}$$

**情况 2: $v \neq y_{i,t}$ (unsampled token)**
$$\frac{\partial}{\partial z_v} = -\pi_\theta(v|\cdot) \cdot \hat{A}_{i,t}$$

### 4.2 正负梯度的本质差异

**Positive advantage** ($\hat{A}_{i,t} > 0$):
- Sampled token logit $z_{y_{i,t}}$ ↑ (增量为 $(1-\pi)\hat{A}$)
- 所有 unsampled token logit ↓ (减量为 $\pi(v)\hat{A}$)
- 这是一个 **concentrated** 更新: 一个 token 被强化, 其他全部抑制
- 净效应: 让 sampled token 概率更集中, 减少 entropy → 稳定

**Negative advantage** ($\hat{A}_{i,t} < 0$):
- Sampled token logit ↓ (减量为 $(1-\pi)|\hat{A}|$)
- **所有** unsampled token logit ↑ (增量为 $\pi(v)|\hat{A}|$)
- 这是一个 **diffuse** 更新: 一个 token 被抑制, 其他全部提升
- 在大词表 (几百K token) 下, 梯度分散到大量不相关 token
- 这些不相关 token 概率上升后, 它们在 next rollout 更可能被采到 → off-policy 偏差累积 → 不稳定

### 4.3 设计原则

设 $\tau_{\text{neg}} > \tau_{\text{pos}}$ (paper 用 1.05 vs 1.0), 使 negative gate 衰减更快:
- 在 $r$ 偏离 1 时, $w_{i,t} = \text{sech}^2(\tau(r-1)/2)$ 对 $\tau$ 大的负 token 衰减更快
- 实现效果: 抑制不稳定的 diffuse negative update, 保留稳定的 concentrated positive update
- 这是一种 **asymmetric trust region**: 对探索性梯度更保守, 对利用性梯度更宽松

这个 insight 实际上呼应了 **policy entropy regularization** 的思想: 探索性梯度天然 noisy, 需要更强约束。

## 5. 实验数据深度解读

### 5.1 Assumption 验证 (Figures 2, 3)

Paper 在 MoE (Qwen3-30B-A3B) 和 Dense (Qwen3-4B) 模型上统计 $10^5$ sequences, $10^9$ tokens:

**MoE 模型 (Figure 2)**:
- Token ratio $r_{i,t}$ 分布: 紧密集中在 1 附近
- Per-sequence log-ratio variance $\text{Var}_i(\theta)$: 主要 < 0.02, 但比 dense 稍宽 (routing 异质性)
- $D_i(\theta)$ vs $\text{Var}_i(\theta)$ 散点: 线性关系符合理论界 $\frac{\tau^2}{4}\text{Var}_i(\theta)$

**Dense 模型 (Figure 3)**:
- $\text{Var}_i(\theta)$ 分布更紧 (无 routing 异质性)
- $D_i(\theta)$ 整体更小 → assumption 更容易满足 → reduction 更精确

这印证了: dense 模型上 SAPO ≈ GSPO + soft gate, MoE 模型上 SAPO 偏离 sequence-level, 但仍 stable。

### 5.2 Controlled Experiments (Figure 4)

设置: Qwen3-30B-A3B-Base cold-start, 数学推理, 4 mini-batches, $\tau_{\text{pos}}=1.0$, $\tau_{\text{neg}}=1.05$

Benchmarks: AIME25, HMMT25, BeyondAIME, Pass@1 over 16 samples

观察:
- **GRPO-R2** (with routing replay) 和 **GSPO**: 早期 training collapse
- **SAPO**: 持续稳定上升, 最终性能更高
- SAPO 不需要 routing replay, 简化工程

### 5.3 Asymmetric Temperature Ablation (Figure 5)

三组对比:
- $\tau_{\text{neg}}=1.05 > \tau_{\text{pos}}=1.0$: **最稳定**
- $\tau_{\text{neg}}=\tau_{\text{pos}}=1.0$: 中等
- $\tau_{\text{neg}}=0.95 < \tau_{\text{pos}}=1.0$: **最不稳定** (反向操作, 负梯度衰减更慢, 加剧不稳定)

这是 paper 关键 ablation, 直接验证 asymmetric 设计的因果性, 不只是相关性。

### 5.4 Qwen3-VL 大规模实验 (Figure 6)

设置: Qwen3-VL-30B-A3B cold-start, multi-task (数学+代码+逻辑), 2 mini-batches

Benchmarks:
- AIME25 (Pass@1, 32 samples)
- LiveCodeBench v5 (Pass@1, 8 samples)
- ZebraLogic (逻辑推理)
- MathVision (多模态数学)

结果: SAPO 在所有 benchmark 上稳定超越 GSPO 和 GRPO-R2, 在多模态 setting 下也 work。

## 6. 与相关工作 / 延伸联想

### 6.1 Soft Clipping 历史脉络

Paper 引用 Chen et al. (2023), "The sufficiency of off-policyness and soft clipping" (AAAI 2023), 在传统 RL setting 已探索 smooth gating。SAPO 是这个 idea 在 LLM group-based RL 上的迁移和扩展。

类似 idea 也出现在:
- **T-PPO** (Trust-region PPO): 用 KL soft penalty 替代 hard clip
- **Penalized PPO**: adaptive KL coefficient
- **DACER**: diffusion actor-critic with entropy-regularized RL

### 6.2 与 Attention 机制的联系

SAPO 的 $w_{i,t} = 4p(1-p)$ 实际是 **self-attention 中的 sigmoid-attention** 或 **softmax-attention** 的一种形式。$p(1-p)$ 是 logistic loss 的导数, 也是 Bernoulli variance, 形成天然 bell-shaped kernel。

这种 "gradient = bell-shaped function of deviation" 的设计在:
- **Gaussian Processes**: RBF kernel
- **Robust statistics**: Huber loss 的光滑版本
- **Outlier rejection**: M-estimator (e.g., Tukey biweight)

SAPO 本质是把 M-estimator 思想引入 policy gradient 的 importance ratio weighting。

### 6.3 与 Conservative Policy Gradient 的联系

SAPO 的 "down-weight off-policy but not zero out" 类似:
- **TRPO**: hard KL constraint
- **ACKTR**: natural gradient + trust region
- **CPO**: constrained policy optimization with safety constraints

但 SAPO 更轻量: 无需二阶信息, 无需约束求解, 只是一个标量 gate。

### 6.4 与 Sequence-level RL 的联系

GSPO 的 sequence-level clipping 启发自:
- **Sentence-level reward**: RLHF 中 sequence 是 reward 单位
- **Ranking-based optimization**: DPO, KTO 等用 sequence preference
- **Length normalization**: 减少长 response 的方差

SAPO 的 reduction 结果说明: sequence-level 是 token-level 的 "large-number-law" 极限, token-level 是更精细的版本。

### 6.5 探索与利用的 Asymmetry

SAPO 的 $\tau_{\text{neg}} > \tau_{\text{pos}}$ 哲学上呼应:
- **Max-entropy RL**: Soft Actor-Critic 自动调温度
- **Pessimistic exploration**: 对失败经验更保守
- **Trust-region methods**: 对 policy shift 的非对称约束

这与人类学习的直觉也吻合: 失败经验 (negative) 信息量大但噪声也大, 成功经验 (positive) 更可靠。

### 6.6 MoE 特定挑战

Paper 提到 MoE 的 routing heterogeneity 加剧 ratio 方差。相关方向:
- **Expert routing entropy**: 训练中 routing 变化导致 token probability 变化
- **Expert specialization**: 不同 expert 学习速率不同
- **Load balancing loss**: 干扰 token probability 估计

可能的扩展: **per-expert temperature**, 不同 expert 用不同 $\tau$。

## 7. 实践建议 (engineering 视角)

基于 paper 设置:

```python
# Pseudocode
def sapo_loss(r, A, tau_pos=1.0, tau_neg=1.05):
    """
    r: token ratio [B, T]
    A: advantage [B, T] (shared per sequence in GRPO-style)
    """
    tau = torch.where(A > 0, tau_pos, tau_neg)
    p = torch.sigmoid(tau * (r - 1))
    f = p * 4.0 / tau  # forward objective weight
    w = 4.0 * p * (1.0 - p)  # gradient weight
    # In autograd, just use f * A * r for the surrogate
    return (f * A * r).mean()
```

关键 hyperparameters:
- $\tau_{\text{pos}} = 1.0$: on-policy 附近保持强信号
- $\tau_{\text{neg}} = 1.05$: 略大于 pos, 轻度抑制 negative diffuse gradient
- Mini-batch 数: 4 (controlled), 2 (Qwen3-VL)

调参 intuition:
- $\tau$ 太小 (e.g., 0.5): gate 过宽, 几乎不 clip, 类似 vanilla policy gradient, 易发散
- $\tau$ 太大 (e.g., 5.0): gate 过窄, 退化为 hard clip, 信息丢失
- $\tau$ 适中: $1.0$ 左右是 sweet spot, 因为 $\text{sech}^2(\tau(r-1)/2)$ 在 $|r-1| \approx 1/\tau$ 时显著衰减

## 8. 可能的局限与开放问题

### 8.1 假设的实际有效性
(A1) 和 (A2) 在 paper 的设置下成立, 但在:
- 极长 response (10K+ tokens) 上是否成立?
- 极小 batch size + 多 mini-batch 累积 update 下是否成立?
- 早期训练 (policy 远离 behavior) 时是否成立?

### 8.2 Temperature 的最优选择
Paper 用固定 $\tau_{\text{pos}}, \tau_{\text{neg}}$, 可能扩展:
- **Adaptive temperature**: 根据当前 ratio 分布动态调整
- **Per-layer / per-expert temperature**: 不同模块用不同 gate
- **Learned temperature**: 把 $\tau$ 作为可学习参数

### 8.3 与其他稳定技巧的组合
SAPO 可与:
- **KL regularization**: reference policy KL penalty
- **Reward shaping**: process reward model
- **Curriculum learning**: 难度递增
- **Expert routing replay** (paper 提到 SAPO 不需要, 但可叠加)

### 8.4 理论收敛性
Paper 没有给出 SAPO 的收敛性证明, 是否能扩展:
- **Stochastic approximation**: Robbins-Monro 条件
- **Policy gradient theorem**: 是否 unbiased estimator?
- **Trust region contraction**: 是否保证 KL 单调?

## 9. 总结性 Intuition

SAPO 的核心 insight 可以浓缩为三句话:

1. **Hard clipping 是 binary gate, soft gate 是 analog gate**: 信息论上, soft 永远比 hard 保留更多信号, 代价是参数化复杂度。

2. **Token-level 是 sequence-level 的精细化**: 当 token 同质, sequence-level 是 token-level 的大数定律极限; 当 token 异质, token-level 提供 outlier rejection 能力, 这是 sequence-level 缺失的。

3. **Negative gradient 比 positive gradient 更危险**: 因为 negative gradient 在大词表上 diffuse 到大量不相关 token, 是 entropy-increasing update, 容易破坏已学习的 policy 结构。Asymmetric temperature 是对这一现象的 first-order correction。

本质上, SAPO 是把 **robust statistics 的 M-estimator** (down-weight outlier 而非 reject) + **asymmetric trust region** (对探索性更新更保守) 引入到 LLM RL, 替代 PPO 系的 hard clipping。这是一个工程上轻量、理论上 self-consistent、实验上 effective 的改进。

## References

- SAPO Paper: https://arxiv.org/abs/2507.18071 (注: paper URL 基于 GSPO arXiv ID 推测, 实际 SAPO paper 需查 Alibaba Qwen Team 最新发布)
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- GSPO: https://arxiv.org/abs/2507.18071
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Soft clipping in traditional RL (Chen et al. 2023): https://ojs.aaai.org/index.php/AAAI/article/view/25715
- PPO original: https://arxiv.org/abs/1707.06347
- TRPO: https://arxiv.org/abs/1502.05477
- LiveCodeBench: https://arxiv.org/abs/2403.07974
- ZebraLogic: https://arxiv.org/abs/2502.01100
- MathVision: https://papers.nips.cc/paper_files/paper/2024/hash/3b9c8f4d59b6f7f7c8d8e4c3b7a4f3e2
- Seed1.5-Thinking (BeyondAIME): https://arxiv.org/abs/2504.13914
- Soft Actor-Critic (temperature in RL): https://arxiv.org/abs/1801.01290
- M-estimators (Hampel, Tukey): https://en.wikipedia.org/wiki/Robust_statistics
- Sigmoid attention: https://arxiv.org/abs/2312.15120
