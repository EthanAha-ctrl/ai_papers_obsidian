# MuonClip 详解

## 一、问题背景：QK Score 爆炸

在 Transformer 的 attention mechanism 中，核心计算是 **Query-Key dot product**：

$$\text{QK Scores}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

其中：
- $\mathbf{q}_i = \mathbf{x}_i W_q^T \in \mathbb{R}^{d_k}$ 是第 $i$ 个 token 的 query 向量
- $\mathbf{k}_j = \mathbf{x}_j W_k^T \in \mathbb{R}^{d_k}$ 是第 $j$ 个 token 的 key 向量
- $d_k$ 是 head dimension，$\sqrt{d_k}$ 是缩放因子
- $W_q, W_k \in \mathbb{R}^{d_k \times d_{model}}$ 是 query 和 key 的投影矩阵

**问题**：当模型 scale 到数十亿参数、训练数据达到万亿 token 量级（如 Kimi K2 的 15.5T tokens）时，QK scores 会 **爆炸**——产生 NaN、梯度消失/爆炸、训练崩溃。

---

## 二、为什么不直接用现有方法？

| 方法 | 问题 |
|------|------|
| **Logit soft-capping**（将 scores 钳制到上限） | 不自然地扭曲 attention distribution，导致模型学习到畸形的概率分布 |
| **Query-Key Normalization**（对 Q、K 向量做 normalize） | 没有从 **weight 本身** 出发解决问题，只是处理了 symptom 而非 cause |

---

## 三、Muon 优化器回顾

在讲 MuonClip 之前，需要理解它基于的 **Muon optimizer**。

Muon 的核心思想是利用 **Newton-Schulz iteration** 对梯度矩阵做 **零幂近似**（zeroth power approximation），即将梯度矩阵投影到最近的正交矩阵，从而让更新方向更平衡。

具体来说，给定梯度矩阵 $G \in \mathbb{R}^{m \times n}$，Newton-Schulz 五步迭代如下：

$$X_0 = \frac{G}{\|G\|_F + \epsilon}$$

迭代公式（系数 $a=3.4445, b=-4.7750, c=2.0315$）：

$$A = X_t X_t^T, \quad B = bA + cA^2, \quad X_{t+1} = aX_t + BX_t$$

最终得到 $X_5 \approx \text{sign}(G)$（即 $G$ 的极分解中的正交部分），再加上 momentum 和 Nesterov 加速。

**Muon 的优势**：更新更平衡、训练更快。  
**Muon 的劣势**：更激进（aggressive），在大规模训练中容易导致 $W_q, W_k$ 的权重值增长过快，从而引发 QK score 爆炸。

---

## 四、MuonClip 核心机制：qk-clip

MuonClip = **Muon + QK Clip**，即在每次 Muon 更新之后，对 $W_q$ 和 $W_k$ 做一个 **后处理的缩放安全网**。

### 具体算法

给定：
- **阈值** $t$（例如 Kimi K2 中 $t=100$）
- **平衡参数** $\alpha \in [0, 1]$（通常 $\approx 0.5$）
- 当前 mini-batch 的输入 $\mathbf{x}$

**Step 1**：计算当前 QK scores

$$\mathbf{q} = \mathbf{x} W_q^T, \quad \mathbf{k} = \mathbf{x} W_k^T$$

$$s_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

$$s_{\max} = \max_{i,j} s_{ij}$$

**Step 2**：判断是否需要 clip

- 如果 $s_{\max} \leq t$：**不操作**，训练正常进行
- 如果 $s_{\max} > t$：触发 clip

**Step 3**：计算缩放因子

$$\eta = \frac{t}{s_{\max} + \epsilon}, \quad \text{其中 } \epsilon \text{ 是防止除零的小常数}$$

此时 $\eta < 1$（因为 $s_{\max} > t$）。

**Step 4**：缩放权重

$$W_q \leftarrow \eta^\alpha \cdot W_q$$

$$W_k \leftarrow \eta^{1-\alpha} \cdot W_k$$

**关键洞察**：为什么不是 $W_q$ 和 $W_k$ 都乘以 $\eta^{0.5}$？因为 $\alpha$ 允许你 **不对称地** 分配缩放量。当 $\alpha=0.5$ 时，两边各缩放 $\eta^{0.5}$，即等比缩放；但也可以偏向一侧。

### 为什么这样做有效？

1. **数学上**：clip 后的 QK score 上界恰好为 $t$。因为：

$$s'_{ij} = \frac{(\eta^\alpha \mathbf{q}_i) \cdot (\eta^{1-\alpha} \mathbf{k}_j)}{\sqrt{d_k}} = \eta \cdot s_{ij} \leq \eta \cdot s_{\max} = t$$

2. **不改变更新方向**：缩放是标量乘法，只改变 $W_q, W_k$ 的 **模长**，不改变它们的 **方向**。Muon 的正交化更新方向被完整保留。

3. **不扭曲 attention distribution**：不同于 soft-capping 直接改 score 的值（这会改变 softmax 后的概率分布），qk-clip 是在 **weight 层面** 做等比缩放，attention 的相对模式完全不变。

---

## 五、训练动态：Early vs Late Stage

根据 Jianlin Su 的博客和 Kimi K2 的训练日志：

| 阶段 | 训练步数 | Max QK Score | Clip 行为 |
|------|----------|-------------|----------|
| **Early Training** | ~70k 步之前 | > 100 | **频繁触发 clip**，将 max score 压回 100 |
| **Late Training** | ~70k 步之后 | ~30 | **不再触发 clip**，权重已自然稳定 |

这意味着 qk-clip 本质上是一种 **training warmup 阶段的 stabilizer**：它只在训练最不稳定的早期阶段介入，随着权重逐渐收敛，clip 自然退出，不需要手动 schedule。

---

## 六、从第一性原理理解

### 为什么 QK score 会爆炸？

从第一性原理出发：

1. **Attention score 的量级** $= \|\mathbf{q}_i\| \cdot \|\mathbf{k}_j\| \cdot \cos\theta_{ij} / \sqrt{d_k}$
2. 其中 $\|\mathbf{q}_i\| \propto \|W_q\|$, $\|\mathbf{k}_j\| \propto \|W_k\|$（假设 $\mathbf{x}$ 的范数相对稳定）
3. 因此 **score 的量级主要由 $\|W_q\| \cdot \|W_k\|$ 决定**
4. Muon 的正交化更新虽然方向更好，但 **不直接控制权重的范数增长**
5. 在高学习率 + 大梯度的早期训练中，$\|W_q\| \cdot \|W_k\|$ 可能指数级增长

### qk-clip 如何从根本上解决这个问题？

它直接 **控制了问题的根因**——权重矩阵的范数。通过：

$$\|W_q'\| \cdot \|W_k'\| = \eta^\alpha \|W_q\| \cdot \eta^{1-\alpha} \|W_k\| = \eta \cdot \|W_q\| \cdot \|W_k\|$$

而 $\eta = t / s_{\max}$ 恰好是让 $\|W_q'\| \cdot \|W_k'\|$ 降到一个安全水平的因子。

---

## 七、与 AdamW + QK Norm 的对比

| 特性 | AdamW + QK Norm | MuonClip |
|------|-----------------|----------|
| 优化器 | AdamW（保守但慢） | Muon（激进但快） |
| 稳定性手段 | Normalize Q/K 向量 | 缩放 $W_q/W_k$ 权重 |
| 训练速度 | 较慢 | 更快（Muon 的正交化更新） |
| Attention distribution | 改变（归一化后分布更平坦） | 保留（相对模式不变） |
| 干预层面 | Inference 时也需要 norm | 只在 training 时 clip，inference 无额外操作 |

---

## 八、代码核心逻辑解读

从附件中的代码来看，关键函数是 `apply_clip`：

```python
def apply_clip(W_q, W_k, alpha, t, x, eps=1e-7):
    q = x @ W_q.T          # (batch, seq, fan_out)
    k = x @ W_k.T
    scores = torch.einsum('bid,bjd->bij', q, k) / np.sqrt(W_q.size(0))
    max_score = scores.max()
    if max_score > t:
        eta = t / (max_score + eps)    # η < 1
        scale_q = eta ** alpha          # W_q 缩放因子
        scale_k = eta ** (1 - alpha)   # W_k 缩放因子
        W_q *= scale_q                  # 原地缩放
        W_k *= scale_k
        return True, max_score.item()   # clip 触发
    return False, max_score.item()      # clip 未触发
```

注意几个细节：
- `scores` 的计算使用的是 **当前 batch 的输入 $\mathbf{x}$**，所以 clip 是 **数据依赖** 的（data-dependent），不是简单的 weight clipping
- 缩放后的 `max_score` 恰好等于 $t$（因为 $s'_{\max} = \eta \cdot s_{\max} = t$）
- `eps=1e-7` 防止 $s_{\max}$ 极小时除零

---

## 九、总结

**MuonClip 的本质**：在 Muon 优化器每次更新 $W_q, W_k$ 后，检查 QK score 是否超过阈值 $t$；如果超过，就按 $\eta^\alpha$ 和 $\eta^{1-\alpha}$ 的比例缩放权重，将 score 压回阈值内。这是一种 **非侵入式**（non-intrusive）的稳定性手段——不改更新方向、不改 attention 分布、只在 training 时生效、随训练稳定自动退出。

**核心公式**：
$$\boxed{W_q \leftarrow \left(\frac{t}{s_{\max}}\right)^\alpha W_q, \quad W_k \leftarrow \left(\frac{t}{s_{\max}}\right)^{1-\alpha} W_k}$$

**关键直觉**：不是在 attention score 层面做 post-hoc 修补，而是在 **weight 层面** 做等比缩放，从源头控制 score 的量级，同时完全保留 Muon 正交化更新的方向信息。

---

**参考链接**：
- Kimi K2 博客：https://moonshotai.github.io/Kimi-K2/
- Jianlin Su 详解：https://kexue.fm/archives/11126
- Jianlin Su 的推文：https://x.com/Jianlin_S/status/1943920839487107372
- Keller Jordan 关于 Muon 的博客：https://kellerjordan.github.io/posts/muon/
- 交互式可视化 Demo：https://muon-clip-app-644257448872.us-central1.run.app