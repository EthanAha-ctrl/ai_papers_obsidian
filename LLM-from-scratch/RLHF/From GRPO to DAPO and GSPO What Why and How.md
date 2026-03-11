

这篇文章详细阐述了**强化学习在大型语言模型训练中的技术演进路径**，从 PPO 开始，经过 GRPO，发展到 DAPO，最终到 GSPO。让我为您详细解读这篇文章的核心内容、技术细节和创新点。

## 一、背景：LLM 强化学习的发展历程

### 1.1 PPO 的局限性

**PPO**曾是主流，但其核心问题在于**依赖价值模型**：

- 对于长文本输出，价值估计不准确
- 从简单任务泛化到复杂任务困难
- 需要维护额外的价值网络，增加计算复杂度

### 1.2 GRPO 的核心突破

**GRPO**移除了对价值模型的依赖，通过**群体相对策略优化**实现：

```python
J_GRPO(θ) = E_{q~P(Q), {o_i}^G_{i=1} ~ π_θ_old(O|q)} [
    (1/G) Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|} (
        min(r_{i,t}(θ)A_i, clip(r_{i,t}(θ), 1-ε, 1+ε)A_i) 
        - β D_KL(π_θ || π_ref)
    )
]
```

其中关键参数：
- **重要性比率**: `r_{i,t}(θ) = π_θ(o_{i,t}|q,o_{i,<t}) / π_θ_old(o_{i,t}|q,o_{i,<t})`
- **优势函数**: `A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})`

## 二、Importance Sampling 的深度分析

### 2.1 基本原理

**Important Sampling** 的数学本质是：
```
E_{p_new}[f(x)] = E_{p_old}[p_new(x)/p_old(x) × f(x)]
```

在 PPO/GRPO 中，我们需要使用旧策略（采样成本低）生成 rollouts，然后用新策略更新。由于两个策略分布不同，需要用重要性比率作为校正权重。

### 2.2 At 和 rt 符号组合的影响

| Adv (At) | Ratio (rt) | 期望行为 | 实际约束 |
|----------|------------|----------|----------|
| >0 | >1 | 强化好行为 | 上限截断 |
| >0 | <1 | 降低好行为概率 | 不合理 |
| <0 | >1 | 增加坏行为概率 | 不合理 |
| <0 | <1 | 惩罚坏行为 | 下限截断 |

**关键洞察**：只有同号组合（同正同负）是期望的，但 clipping 操作会进一步限制。

### 2.3 Clipping 的具体影响

对于 At > 0：
- 当 rt > 1+ε 时，梯度 → 0（token 贡献被完全抑制）
- 当 rt < 1-ε 时，min 操作限制但不截断梯度

对于 At < 0：
- 当 rt < 1-ε 时，梯度 → 0
- 当 rt > 1+ε 时，理论上可以到 +∞

## 三、DAPO：GRPO 的精细化改进

DAPO 目标函数：
```python
J_DAPO(θ) = E_{(q,a)~P(Q), {o_i}^G_{i=1}~π_θ_old(O|q)} [
    (1/Σ|o_i|) Σ_{i=1}^G Σ_{t=1}^{|o_i|} 
    min(r_{i,t}(θ)A_i, clip(r_{i,t}(θ), 1-ε_low, 1+ε_high)A_i)
]
```

### 3.1 Clip-Higher：提升上界，解决"马太效应"

**问题建模**：
```
P_old × (1+ε) = 上界
```

对比两种场景：
- **高概率 token**：P_old = 0.9, ε = 0.2 → 上界 = 1.08（永远不会被截断）
- **低概率 token**：P_old = 0.2, ε = 0.2 → 上界 = 0.24（即使提升到 0.4 也会被截断）

**Clip-Higher 的设计**：
- 保持下界：`1-ε_low`（通常 1-0.2 = 0.8）
- 抬升上界：`1+ε_high`（ε_high > ε）

这使得低概率的好 token 不会过早被截断。

### 3.2 Dynamic Sampling：避免无效样本

**问题**：
```python
假设采样 10 个响应：
- 全部得分为 0：所有 Adv = 0
- 全部得分为 1：所有 Adv = 0
结果：全部贡献零梯度
```

**Dynamic Sampling 策略**：
```
约束：0 < |{o_i | is_equivalent(a, o_i)}| < G
```

即强制同一 query 的采样集中必须同时包含正确和错误样本。

### 3.3 Token-Level Gradient Loss：解决梯度稀释

**GRPO 的梯度权重计算**：
```python
样本 1（200 tokens）：w = (1/200) × (1/2)
样本 2（10 tokens）：  w = (1/10) × (1/2)
```
→ 短样本的 token 权重是长样本的 20 倍

**DAPO 的改进**：
```python
统一权重：w = 1/(200+10)
```

所有 token 不受样本长度影响，平等参与梯度更新。

### 3.4 Overlong Reward Shaping：软惩罚机制

**惩罚曲线设计**：
```
if length > threshold_1:
    penalty = α × (length - threshold_1)
if length > threshold_2:
    penalty 足够大以抵消奖励
```

这防止模型生成冗长但低质量的响应。

## 四、GSPO：从 Token 到 Sequence 的范式转变

### 4.1 MoE 架构中的挑战

**专家激活的波动性**：
```
π_θ_old 和 π_θ 可能激活不同专家集
→ 路由差异导致概率剧烈波动
→ 频繁触发 clipping
→ 梯度信号大量丢失
```

**Traditional Solution - Routing Replay**：
- 记录采样时的专家激活
- 训练时强制使用相同路由
- **缺点**：高工程成本、限制性能提升

### 4.2 GSPO 的核心创新

**目标函数**：
```python
J_GSPO(θ) = E_{q~P(Q), {o_i}^G_{i=1}~π_θ_old(O|q)} [
    (1/G) Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|} 
    min(s_i(θ)A_i, clip(s_i(θ), 1-ε, 1+ε)A_i)
]
```

**Sequence-level Importance Ratio**：
```python
s_i(θ) = [π_θ(o_i|q) / π_θ_old(o_i|q)]^(1/|o_i|)
       = exp[(1/|o_i|) Σ_{t=1}^{|o_i|} 
           log(π_θ(o_{i,t}|q,o_{i,<t}) / π_θ_old(o_{i,t}|q,o_{i,<t}))]
```

### 4.3 数学原理深度解析

#### 为什么要指数化？

**正确形式**：
```
E_{z~π_tar}[f(z)] = E_{z~π_beh}[π_tar(z)/π_beh(z) × f(z)]
```
权重必须是概率比率，非负数。

**错误形式**：
```
E[Δlog p × A]
```
这不再是无偏的重要性采样校正。

#### 为什么长度归一化？

防止长序列的比率爆炸：
```python
长度归一化：1/|o_i| × Σ log p_ratio
指数化后保持量级一致
```

### 4.4 梯度分析对比

**GSPO 梯度**：
```
∇_θ J_GSPO(θ) = E[
    (1/G) Σ_{i=1}^G s_i(θ) A_i × (1/|o_i|) Σ_{t=1}^{|o_i|} ∇_θ log π_θ
]
```
→ 同一序列内所有 token 共享相同权重 `s_i(θ)A_i/|o_i|`

**GRPO 梯度**：
```
∇_θ J_GRPO(θ) = E[
    (1/G) Σ_{i=1}^G (A_i/|o_i|) Σ_{t=1}^{|o_i|} r_{i,t}(θ) ∇_θ log π_θ
]
```
→ 每个位置有不同权重 `r_{i,t}(θ)A_i/|o_i|`

### 4.5 性能差异分析

| 指标 | GRPO | GSPO |
|------|------|------|
| 方差 | 高（token 级波动） | 低（sequence 级平滑） |
| MoE 稳定性 | 差（专家激活抖动） | 优（自然避免） |
| 长序列鲁棒性 | 弱 | 强 |
| 有效 token 比例 | 高 | 低（更激进的 clipping） |
| 训练效率 | 低 | 高 |

## 五、技术演进总结与对比

### 5.1 演进路径图
```
PPO (依赖 value model)
    ↓
GRPO (移除 value model，但 token 优化)
    ↓
DAPO (GRPO 的精细优化)
    ├─ Clip-Higher
    ├─ Dynamic Sampling
    ├─ Token-Level Gradient Loss
    └─ Overlong Reward Shaping
    ↓
GSPO (sequence 级优化，解决 MoE 问题)
```

### 5.2 核心对比表

| 特性 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| Value Model | 需要 | 不需要 | 不需要 | 不需要 |
| 优化粒度 | Token | Token | Token | Sequence |
| Clipping | 对称 | 对称 | 不对称 | 对称 |
| MoE 支持 | 差 | 差 | 差 | 优 |
| 长文本稳定性 | 差 | 中 | 中 | 优 |
| 实现复杂度 | 高 | 中 | 中 | 低 |

### 5.3 适用场景推荐

- **传统 Transformer**：DAPO 即可满足
- **MoE 架构**：强烈推荐 GSPO
- **长文本生成**：GSPO 的 sequence 级优势明显
- **资源受限**：DAPO 在不引入复杂性的前提下提升效果

## 六、实际应用与实践建议

### 6.1 超参数建议

**Clipping 参数**：
- DAPO: `ε_low = 0.2`, `ε_high = 0.5~1.0`
- GSPO: `ε = 0.2`

**采样策略**：
- GRPO/DAPO: G = 4~16
- GSPO: 可适当减少（因 sequence 级噪声更低）

### 6.2 实践陷阱

1. **GSPO 不是万能的**：对于短序列任务，token 级优化可能更细粒度
2. **Clip-Higher 需要平衡**：上界过高可能导致训练不稳定
3. **Dynamic Sampling 的成本**：可能增加额外 forward passes

### 6.3 未来方向

根据文章，可能的演进方向包括：
- Hybrid approaches：结合 token 和 sequence 级优化
- Adaptive granularity：根据任务动态选择优化粒度
- Better reward shaping：更精细的序列级奖励设计

## 七、参考资源

- **原始论文**：DeepSeek-AI 的 GRPO 论文
- **社区讨论**：Hugging Face 社区活跃讨论
- **实装案例**：QWen3 系列已采用 GSPO

这篇文章的价值在于**系统性地梳理了 LLM 强化学习优化的演进逻辑**，特别是从 token 级到 sequence 级的范式转变，为未来研究提供了清晰的技术路线图。对于想要在 MoE 架构上应用强化学习的团队，GSPO 提供了一个简洁而有效的解决方案。