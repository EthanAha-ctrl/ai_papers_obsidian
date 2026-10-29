---
source_pdf: THE ADEMAMIX OPTIMIZER.pdf
paper_sha256: f3502dffb8758859fc17f2bdaa113852a0c68c2a9a54c0c83fbe5fc572ce184e
processed_at: '2026-08-12T13:49:59-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AdEMAMix 用人话讲

Andrej，咱们抛开公式，用最朴素的话把这事说清楚。

---

## 一句话说清楚

Adam 用一个"记忆"来记过去梯度，这个记忆有个毛病——**要么记得近的忘了远的，要么记得远的模糊了近的**。AdEMAMix 干脆用两份记忆，一份只管最近几步（快记忆），一份管几千步（慢记忆），俩一拼，问题没了。

---

## 为什么要搞这事？

先看 Adam 怎么工作的。你每步算个梯度 g，Adam 把过去梯度做个加权平均当更新方向。权重是指数衰减的：

$$w_i = \beta^i (1-\beta)$$

- **β**: 衰减系数，越大记得越久
- **i**: 往前回溯第几步
- **w_i**: 第 i 步前的梯度占的权重

β=0.9 的时候，6 步前的梯度就只剩一半权重了。6 步，就这么点 memory。你训练一个 LLM 动辄几十万步，前面 99.99% 的梯度信息全扔了。

那你把 β 调到 0.9999 呢？好，现在 7000 步前的梯度还有一半权重。但问题来了——**最近 6 步的梯度权重也变得很小**。你刚刚算出来的梯度，在更新方向里只占个零头。模型对"当前 loss landscape 的局部变化"几乎没反应，在峡谷里来回撞墙。

看 Fig. 3a 那张图就一目了然：
- β=0.9：尖峰，最近几步吃掉所有权重
- β=0.9999：平摊，每步都一丁点
- **没有任何单一 β 能同时做到"近的尖"+"远的平"**

这就是 single EMA 的根本几何限制，跟调参没关系。

---

## AdEMAMix 怎么搞的

非常简单，加个第二份 EMA：

- **m1**：fast memory，β1=0.9，跟 AdamW 原版一样，管最近几步
- **m2**：slow memory，β3=0.9999，管最近 7000 步
- 最终更新方向 = m1 + α·m2，α 一般取 8

就这点改动。公式：

$$\theta^{(t)} = \theta^{(t-1)} - \eta\left(\frac{\hat{m}_1^{(t)} + \alpha \cdot m_2^{(t)}}{\sqrt{\hat{\nu}^{(t)}} + \epsilon} + \lambda \theta^{(t-1)}\right)$$

- **θ**: 模型参数
- **η**: learning rate
- **m1**: fast EMA
- **m2**: slow EMA，注意没做 bias correction
- **α**: slow memory 的权重，[4, 10] 之间最好
- **ν**: second moment，跟 AdamW 一样
- **λ**: weight decay

m2 不做 bias correction 是故意的——它就是要慢慢填满，强行 correct 早期会爆炸。

---

## 为什么两个 EMA 比一个好？canyon 直觉

这是 paper 最精妙的 intuition。把 loss landscape 想成一条蜿蜒的峡谷：

**沿着峡谷走的方向**：梯度方向几千步都差不多，slow EMA 在这方向上不断累加，effective step size 越来越大——你飞快地往前冲。

**垂直峡谷的方向**：梯度一会正一会负（撞左墙往右推，撞右墙往左推），slow EMA 在这方向上正负抵消，自然变小——不会撞墙。但 fast EMA 还在，能感知"该转弯了"，做局部修正。

单 EMA 用大 β 的问题：转弯的时候反应不过来。用小 β 的问题：沿着峡谷方向没有长期累积，跑不快。

AdEMAMix 同时拥有两者：slow EMA 负责"大方向加速度"，fast EMA 负责"局部纠偏"。Fig. 2 那个 Rosenbrock 函数的图特别清楚——AdEMAMix 又快又不抖，Adam 调大 β1 就抖死，调小 β1 就慢死。

---

## 一个关键 negative result

你可能会想：那我把 AdamW 自己的 β1 调大不就行了？

**不行**。paper 花了整个 App. C.1.6 证伪这个。试了四种：
1. From scratch 直接用大 β1：β1>0.999 直接 diverge
2. From scratch + β1 scheduler 慢慢加：稳定了，但 final loss 比 β1=0.9 baseline 还差
3. From checkpoint 突然提 β1：没改善
4. From checkpoint + scheduler 提 β1：还是没改善

**单一 EMA 的架构性缺陷，调参救不回来**。必须有两份独立 memory，一份快一份慢。

这个 negative result 我觉得是整篇 paper 最有价值的部分之一。它说明"为什么 Adam 用 β1=0.9" 不是因为 0.9 这个数好，是因为 single EMA 在 0.9 附近才不犯病。你想吃大 β 的红利，必须加 fast EMA 做纠偏。

---

## Scheduler 为什么要搞那么复杂

直接 β3=0.9999, α=8 起跑会 diverge。早期 ν 还没填满，分母小，slow EMA 让 update norm 爆炸。

**α scheduler** 简单，线性从 0 涨到 α：

$$\alpha^{(t)} = \min\left(\frac{t \cdot \alpha}{T_\alpha}, \alpha\right)$$

- **T_α**: warmup 长度，一般设成总训练步数 T

**β3 scheduler** 麻烦。直觉上线性加 β3 就行，但 t_half 关于 β 是高度非线性的：

$$t_{\text{half}} = \frac{\ln(0.5)}{\ln(\beta)} - 1$$

- β=0.9 加 0.0001：t_half 几乎不变
- β=0.999 加 0.0001：t_half 增加 77 步
- β=0.9999 加 0.0001：t_half 增加几千步

线性加 β3 等于前期跟没加一样，后期突然跳变。所以作者搞了个 scheduler 让 **t_half 线性增长**，而不是 β3 线性增长：

$$\beta_3^{(t)} = \min\left(\exp\left(\frac{\ln(\beta_{\text{start}})\ln(\beta_3)}{(1-\frac{t}{T_{\beta_3}})\ln(\beta_3) + \frac{t}{T_{\beta_3}}\ln(\beta_{\text{start}})}\right), \beta_3\right)$$

- **β_start**: 起点，设成 β1=0.9
- **β3**: 终点，0.9999
- **T_β3**: warmup 长度

这样前期 t_half 小，slow EMA 短视，不会爆炸；后期 t_half 慢慢拉长，slow EMA 逐渐看到更远。

---

## 实验数据，直接看重点

### Token efficiency（最 striking 的结果）

1.3B LLM on RedPajama：

| Optimizer | Tokens | Final loss |
|---|---|---|
| AdamW | 197B | X |
| AdEMAMix | 101B | ≈ X |

**一半的 token，同样的 loss**。在 scaling law 语境下等于 2x compute efficiency。110M 和 330M 上类似比例。

### ICL downstream（Table 7）

1.3B 模型 131B tokens 训练后：

| Task | AdamW | AdEMAMix |
|---|---|---|
| PubmedQA | 0.556 | 0.632 (+7.6%) |
| Winogrande | 0.563 | 0.580 |
| HellaSwag | 0.426 | 0.436 |
| ARC-Challenge | 0.262 | 0.274 |

大部分 task 小幅改善，PubmedQA 大幅改善（这个有点可疑，可能 noise）。

### ViT 结果（Fig. 6）

- 24M params + 11M images (IN-21k): AdEMAMix trivially 赢
- 86M params + 11M images: 还能赢
- 86M params + 1.3M images (IN-1k, 320 epochs): **赢不了**

关键 insight：**AdEMAMix 几乎总能降低 train loss**。但只有 train loss 降低能 translate 到 test loss 降低时，它才胜出。在 overfitting regime，更好的 optimizer 只是让你 overfit 更快。这说明 AdEMAMix 是个更好的 optimizer，不是更好的 regularizer。

### Forgetting 实验（特别有意思，Fig. 4）

Protocol: 固定一个 batch B，从训练集里删掉 B。训练时在不同时刻 t_B 把 B 注入。

结果：
- **AdamW**: B 注入后 loss 突降，然后**快速反弹**——典型的 catastrophic forgetting
- **AdEMAMix**: B 注入后 loss 平滑下降，**几千步后还在持续下降**——B 的信息长期保留

这给了一个深层 insight：AdEMAMix 的 efficiency gains 部分来自 **slower forgetting**。每个 batch 的梯度信息更充分地"沉淀"到参数里，单 EMA 浪费了这部分信号。

Fig. 10 还证明一个有趣的事：forgetting 主要由 **learning rate decay** 控制。在 LR decay 开始前，old batches 都不被记住。这暗示 cosine decay 不只是"收敛工具"，还是"记忆巩固工具"。

---

## Memory & 省钱技巧

多一份 m2，多一份跟参数同大小的 buffer。1.3B 模型多 ~5GB。

**Trick**: 设 β1=0，那 m1 就退化为当前梯度 g，不用存了。Memory 跟 AdamW 一样。App. C.1.7 显示 1.3B 上 β1=0 的 final loss 甚至略好于 β1=0.9，但训练曲线上 spike 多一些。

```python
# 关键代码
m1.mul_(beta1).add_(grad, alpha=1 - beta1)
m2.mul_(beta3).add_(grad, alpha=1 - beta3)  # m2 无 bias correction
nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
update = (m1 / bias_correction1 + alpha * m2) / (nu.sqrt() / sqrt(bc2) + eps)
```

[FSDP](https://arxiv.org/abs/2304.11277) 可以把 optimizer states 分布到多卡，memory 不是大问题。

---

## 跟相关工作的关系

**DiLoCo / SlowMo**：分布式训练里 N 个 worker 各跑 K 步，每 K 步用 outer momentum 聚合。这本质就是离散版的 slow EMA，outer optimizer 每 500 步应用一次 momentum 相当于 β≈0.998。作者猜测 DiLoCo 的 gains 部分来自这种"slow momentum"效应。AdEMAMix 是这个 idea 的 continuous + single-node 版本。

[DiLoCo](https://arxiv.org/abs/2311.08105)

**Lion**：用 sign(α·m + (1-α)·g)，有点像 β1=0 的 AdEMAMix 变体。但 Lion 用 β=0.99 最好，再大就 diverge。AdEMAMix 靠 scheduler 让 β=0.9999 成为可能。

[Lion](https://arxiv.org/abs/2302.06675)

**Sophia**：用 Hessian preconditioner，second-order 路线。AdEMAMix 是 first-order 但通过 slow EMA 间接实现类似 long-horizon trust region 的效果。

[Sophia](https://arxiv.org/abs/2305.14342)

**Grokfast**：也用 EMA 分离 fast/slow signal，但是 nested EMA (EMA on EMA)，会衰减 recent gradients，跟 AdEMAMix 目标相反。

[Grokfast](https://arxiv.org/abs/2405.20233)

**AggMo**：K 个 momentum 之和，证明大 β 可行。但 AggMo 在 SGD 上，没 Adam 的 second moment，没在 LLM scale 试过，没设计 scheduler。AdEMAMix 在 LLM 上证明 gains 远超 AggMo。

[AggMo](https://openreview.net/forum?id=Hkx-oFZq)

---

## 实操建议

如果你要训 1B+ LLM：

1. 直接换 AdEMAMix，β3=0.9999, α=8, T_α=T_β3=T（总步数）
2. β1=0.9, β2=0.999, weight decay=0.1 保持原样
3. LR 可能要稍微降一点（1.3B 上 AdEMAMix 用 3e-4，AdamW 用 5e-4）
4. Memory 紧就设 β1=0
5. 也可以中途从 AdamW 切到 AdEMAMix：m2 init=0，不用 scheduler，越早切越好（Fig. 5b,c）
6. **绝对不要**天真地把 AdamW 的 β1 调大——这篇 paper 花了大篇幅证明这没用

**短训练 (<64k 步)**: 用 β3=0.999 比 0.9999 好（Fig. 17）。slow EMA 的 t_half 要小于总训练步数才有意义。

---

## 几个我自己的 speculative 联想

**(1) Slow EMA ≈ gradient space 的 Polyak averaging**

m2 在 ~7000 步上平均梯度，如果这 7000 步的 weight 变化主要靠最后几步，m2 在某种意义上是"参数轨迹的方向导数"。这跟 [SWA (Stochastic Weight Averaging)](https://arxiv.org/abs/1803.05407) 有联系，但作用在 gradient space 而非 weight space。

**(2) Hierarchical gradient memory**

多个 EMA 不同 β 形成 "gradient memory hierarchy"，类似 CPU 的 L1/L2/L3 cache。slow EMA 是 L3，fast EMA 是 L1。生物神经元也有 multiple timescale plasticity，这个结构在自然界是普遍的。

Ad3EMAMix（加第三个 EMA）实验显示两个就够，第三个冗余。这说明 L1+L3 已经覆盖了主要 timescale，中间的 L2 收益递减。

**(3) Long-tail gradient relevance**

gradients 在几万步后仍 relevant 暗示 loss landscape 在大部分训练中**局部 quasi-convex**，否则 gradients 会快速反向使 slow EMA 自抵消。这给 [loss landscape smoothness](https://arxiv.org/abs/1712.09913) 研究提供新证据。

**(4) Implicit curriculum**

slow EMA 让早期 batch 持续贡献方向。结合 forgetting 实验，这暗示 LLM 训练中**早期 batch 的 gradient signal 价值很高**，single EMA 浪费了。这可能跟 [data curriculum](https://arxiv.org/abs/2305.02217) 研究联动——如果早期数据特别重要，那 curriculum design 更关键。

**(5) Kalman filtering 视角**

两个 EMA 不同 β 类似两个 Kalman filter 不同 process noise 假设。slow EMA 假设系统 quasi-static，fast EMA 假设系统快变。Mixture 类似 multi-model Kalman filtering。

---

## 局限性

1. **短训练不友好**：T < 64k 时 slow EMA 还没填满，用 β3=0.999 更好
2. **Distribution shift 不友好**：长 memory 对突然的 distribution shift 反应迟钝。Continual learning 可能需要 decay m2
3. **Overfitting regime 不友好**：ViT on IN-1k 320 epochs 上没赢。train loss 降低不等于 test loss 降低
4. **理论空白**：为什么 gradients 几万步后还 relevant？loss landscape 的 local linearity 持续时间远超预期。Variance reduction 视角说 large momentum 会损害 generalization，但 AdEMAMix 没观察到。这两个理论 gap 都没解释

---

## 我的整体判断

这篇 paper 厉害在哪：

1. **极简改动**：多加一份 EMA buffer，代码改动 ~10 行
2. **巨大收益**：2x token efficiency 在 LLM scale
3. **Mechanism 清晰**：canyon intuition + fast/slow decoupling
4. **Ablation 详尽**：每个 design choice 都有对应实验
5. **Negative results 诚实**：花整个 section 证伪"single large β EMA"替代方案

它暴露了 Adam 家族 single-EMA 设计的盲点。过去十年大家都在 β1=0.9 这个 corner 周围调参，没人认真想过"为什么是 0.9"。这篇 paper 给出了答案——single EMA 在 0.9 才不犯病，想吃大 β 的红利必须加 fast EMA 做纠偏。

更深层地说，它打开了一扇门：**gradient memory 的结构化设计**可能是下一个 optimizer frontier。EMA 只是最朴素的 gradient aggregation，更复杂的结构（hierarchical, adaptive timescale, task-conditional）可能还有很大空间。

[AdEMAMix 论文](https://arxiv.org/abs/2409.03437) | [官方代码](https://github.com/apple/ml-ademamix)

---

# AdEMAMix 深度解析

Andrej，这篇 paper 触到了 deep learning 优化器设计中一个被严重忽视的 corner——**单一 EMA 的根本性 trade-off**。下面我从 intuition、math、experiments 三个层面展开，尽量 build up your intuition。

---

## 1. 核心 Insight：单一 EMA 的根本缺陷

Adam / AdamW 用一个 EMA 来 aggregate 过去 gradients：

$$\text{EMA}(\beta, \mathcal{G}^T) = \sum_{i=0}^{T} \beta^{i}(1-\beta)\,\mathbf{g}^{(T-i)}$$

- **β**: decay coefficient, 控制历史梯度的衰减速度
- **i**: 距离当前步 $T$ 的回溯步数（i=0 即当前步）
- **g^(T-i)**: 第 $T-i$ 步的 stochastic gradient
- **(1-β)**: 归一化系数，保证权重和为 1

定义 **half-life**：

$$t_{\text{half}} = \frac{\ln(0.5)}{\ln(\beta)} - 1$$

这是累积权重达到 0.5 所需要的步数。代入数值：

| β | t_half | 含义 |
|---|---|---|
| 0.9 | ≈ 6 | 6 步之前的梯度几乎被遗忘 |
| 0.99 | ≈ 68 | 中等 |
| 0.9999 | ≈ 6,930 | 7k 步前的梯度还有一半权重 |
| 0.99999 | ≈ 69,314 | 70k 步前 |

**Key observation**: 任何单一 β 都无法同时满足 (i) 给 immediate past 高权重，(ii) 给 very old gradients 非零权重。看 Fig. 3a 就明白——β=0.9 是个尖峰，β=0.9999 是个平坦分布，没有中间地带。

直觉上，practitioners 普遍认为 "old gradients are outdated because the iterate moved"，所以用 β=0.9。但作者 empirically 证明：**gradients can stay relevant for tens of thousands of steps**。问题不在 gradient relevance，而在 EMA 这个工具本身。

---

## 2. AdEMAMix 的设计

### 2.1 主公式

在 AdamW 基础上加一个 **slow EMA** $m_2$：

$$
\begin{cases}
\mathbf{m}_1^{(t)} = \beta_1 \mathbf{m}_1^{(t-1)} + (1-\beta_1)\mathbf{g}^{(t)}, & \hat{\mathbf{m}}_1^{(t)} = \frac{\mathbf{m}_1^{(t)}}{1-\beta_1^t} \\
\mathbf{m}_2^{(t)} = \beta_3 \mathbf{m}_2^{(t-1)} + (1-\beta_3)\mathbf{g}^{(t)} & \text{(no bias correction)} \\
\boldsymbol{\nu}^{(t)} = \beta_2 \boldsymbol{\nu}^{(t-1)} + (1-\beta_2)(\mathbf{g}^{(t)})^2, & \hat{\boldsymbol{\nu}}^{(t)} = \frac{\boldsymbol{\nu}^{(t)}}{1-\beta_2^t} \\
\boldsymbol{\theta}^{(t)} = \boldsymbol{\theta}^{(t-1)} - \eta\left(\frac{\hat{\mathbf{m}}_1^{(t)} + \alpha \mathbf{m}_2^{(t)}}{\sqrt{\hat{\boldsymbol{\nu}}^{(t)}}+\epsilon} + \lambda \boldsymbol{\theta}^{(t-1)}\right)
\end{cases}
$$

变量解释：
- **m1**: fast EMA, β1=0.9, 对近期梯度敏感，类似 AdamW 原本的 momentum
- **m2**: slow EMA, β3=0.9999, half-life ≈ 7000 步，捕捉 long-range gradient 信号
- **ν**: second moment, 与 AdamW 一致
- **α**: mixing coefficient, 控制 slow EMA 的权重, 典型值 [4, 10]
- **λ**: weight decay
- **ε**: numerical stability

为什么 m2 不做 bias correction？因为 m2 本身就靠 scheduler 慢慢 fill up，强行 bias-correct 反而会让早期 m2 爆炸。

### 2.2 Schedulers: 防止早期 divergence

直接从 β3=0.9999, α=8 起跑会 diverge。原因：早期 ν 还没 warmup 完，分母太小，slow EMA 会让 update norm 爆炸（见 Fig. 27）。

**α scheduler**（线性）：

$$\alpha^{(t)} = \min\left(\frac{t \cdot \alpha}{T_\alpha}, \alpha\right)$$

- **T_α**: warmup steps，通常设为总训练步数 T

**β3 scheduler**（非线性，让 t_half 线性增长）：

$$\beta_3^{(t)} = \min\left(\exp\left(\frac{\ln(\beta_{\text{start}})\ln(\beta_3)}{(1-\frac{t}{T_{\beta_3}})\ln(\beta_3) + \frac{t}{T_{\beta_3}}\ln(\beta_{\text{start}})}\right), \beta_3\right)$$

为什么非线性？因为 t_half 关于 β 是高度非线性的——在 β=0.9 附近加 0.0001 几乎没变化，但在 β=0.999 附近加 0.0001 会让 t_half 增加 77 步。线性加 β 等价于前期几乎没有 warmup，后期突然跳变。这个 scheduler 让 **t_half 线性增长**，更合理。

---

## 3. Intuition：为什么两个 EMA 比 单个大-β EMA 好？

### 3.1 Canyon intuition（论文最精彩的 intuition）

> While changing the direction of the slow momentum is difficult, any adjustment orthogonal to that direction is easy—which favors fast progress in sinuous canyon-like landscapes.

把 loss landscape 想象成一个蜿蜒的 canyon：
- **沿着 canyon 方向**：梯度方向长期稳定，slow EMA 在这个方向上持续累积，effective step size 变大 → 快速前进
- **垂直 canyon 方向**：梯度方向反复震荡，slow EMA 在这个方向上自我抵消（几何级数累加正负项），但 fast EMA 仍能响应局部 correction → 避免 wall-bouncing

单一 EMA 用大 β 会失去 perpendicular correction 能力，导致在 canyon 中震荡（见 Fig. 2b 中 β1=0.999 的轨迹）。AdEMAMix 同时拥有两者（Fig. 2c）。

### 3.2 2D toy experiment（App. C.1.6）

用一个 Beale-like function $f(x,y) = 8(x-1)^2(1.3x^2+2x+1) + 0.5(y-4)^2$：
- 给 EMA 一个非零初始化（模拟 initial "speed"），相当于一个错误的初始方向
- Adam with β1=0.999：要几百步才能纠正 m1 的方向，期间 iterate 沿着错误方向飞
- AdEMAMix：fast EMA (m1, β=0.9) 几步内就能感知到 gradient 方向变化，slow EMA 的初始 bias 慢慢衰减，最终收敛

这解释了为什么 "single large-β EMA" 不 work（见 App. C.1.6, Fig. 19, 20）：增大 AdamW 的 β1 不管是 from scratch 还是 from checkpoint，加 scheduler 还是不加，**都搞不定**——因为单个 EMA 没有 fast correction 通道。

### 3.3 为什么 m1 + αm2 而不是 (1-α)m1 + αm2？

App. C.1.8 专门讨论这个。理论上两者可以 reparameterize：

$$\eta(m_1 + \alpha m_2) = \hat{\eta}\left((1-\hat\alpha)m_1 + \hat\alpha m_2\right), \quad \hat\eta = \eta(\alpha+1), \quad \hat\alpha = \frac{\alpha}{\alpha+1}$$

但加了 cosine η scheduler + linear α scheduler 后，两者不再等价。Empirically (Fig. 24)：m1 + αm2 在所有 (η, α) 组合上都更好。Convex combination 会随着 α→1 丢掉 fast EMA 信号，破坏了 AdEMAMix 的核心机制。

---

## 4. 实验结果数据

### 4.1 LLM scaling behavior

Table 1: Transformer 架构

| Params | Hidden | Heads | Layers |
|---|---|---|---|
| 110M | 768 | 12 | 12 |
| 330M | 1024 | 16 | 24 |
| 1.3B | 2048 | 16 | 24 |

关键 token efficiency 数据（Fig. 1）：

| Model | AdEMAMix tokens | Equivalent AdamW tokens | Improvement |
|---|---|---|---|
| 110M | 17B | 33B | ~94% |
| 330M | ~30B | ~50B | ~67% |
| 1.3B | 101B | 197B | **+95%** |

1.3B AdEMAMix 用一半 token 就能 match AdamW。在 LLM scaling law 语境下，这相当于 ~2x 的 compute efficiency，非常 dramatic。

### 4.2 ICL results (Table 7, 1.3B on 131B tokens)

| Task | AdamW | AdEMAMix | Δ |
|---|---|---|---|
| ARC-Challenge | 0.262 | 0.274 | +0.012 |
| BoolQ | 0.569 | 0.576 | +0.007 |
| HellaSwag | 0.426 | 0.436 | +0.010 |
| MathQA | 0.226 | 0.236 | +0.010 |
| Winogrande | 0.563 | 0.580 | +0.017 |
| MMLU | 0.244 | 0.248 | +0.004 |
| **PubmedQA** | **0.556** | **0.632** | **+0.076** |
| RewardBench (reasoning) | 0.617 | 0.630 | +0.013 |

PubmedQA 的 +7.6% 非常 striking，但单点 caveat 仍需谨慎。

### 4.3 ViT results (Fig. 6, 30)

| Setting | Data/Capacity | AdEMAMix vs AdamW |
|---|---|---|
| 24M params, 11M images (IN-21k) | high | trivially better |
| 86M params, 11M images (IN-21k) | medium | mostly better |
| 86M params, 1.3M images (IN-1k) | low | hard to beat baseline |

观察：**AdEMAMix 几乎总是降低 train loss**，但只有当 train loss 降低对应 test loss 降低时才胜出。在 overfitting regime（86M + IN-1k + 320 epochs），额外的优化能力转化为 overfit 而非 generalization。这暗示 AdEMAMix 是个 **better optimizer**，不一定是 better regularizer。

### 4.4 Forgetting experiment（Fig. 4, 9, 10）

Protocol：固定一个 batch B，从训练数据中移除 B，训练两个模型（一个见过 B 一个没见过）。在 t_B 时把 B 注入训练，观察 B 上的 loss。

关键发现：
- AdamW: B 的 loss 突降后**快速反弹**——典型的 catastrophic forgetting
- AdEMAMix: B 的 loss 下降更平滑，且**几千步后还在持续下降**——B 的信息长期保留
- 后期训练（t_B > 180k）两者 forgetting 都减缓，主要由 **learning rate decay** 驱动（Fig. 10 证明：在 LR decay 开始前，old batches 都不被记住）

这给了一个深层 insight：AdEMAMix 的 gains 部分来自 **slower forgetting**，让每个 batch 的 gradient 信息更充分地"沉淀"到参数中。

### 4.5 Mamba results (Fig. 3c)

168M Mamba on FineWeb: AdEMAMix 仍胜 AdamW。这说明 gains 不局限于 Transformer 的特定 optimization geometry。

---

## 5. 重要的 Negative Results

### 5.1 "Why not just increase AdamW's β1?"

App. C.1.6 系统性证伪了"single EMA + large β"的方案：
1. From scratch with β1 ∈ {0.99, 0.999, 0.9999, 0.99999}: β1 > 0.999 都 diverge
2. From scratch with β1 scheduler (linear t_half warmup): 稳定但 final loss 都不如 β1=0.9 baseline
3. From AdamW checkpoint (300k steps) 突然提升 β1: 无改善
4. From checkpoint with β1 scheduler: 无改善

这组实验是 paper 最有说服力的部分之一——**问题不是 β 不够大，是 single EMA 的架构限制**。

### 5.2 Ad3EMAMix（App. C.3.3）

加第三个 EMA (β4) 不带来任何 improvement。说明 **两个 EMA 是 sweet spot**：一个 fast for correction，一个 slow for accumulation，再多冗余。

### 5.3 Comparison with Lion (App. C.3.2)

Lion 用 sign(α·m + (1-α)·g) 类似于 β1=0 的 AdEMAMix 变体，但 Lion 用 β=0.99 最好，再增大就 diverge。AdEMAMix 的 scheduler 让 β=0.9999 成为可能，这是 Lion 做不到的。

---

## 6. 与相关工作的 connection

### 6.1 DiLoCo / SlowMo（分布式优化）

DiLoCo 中 N 个 worker 独立训练 K 步，每 K 步用 outer momentum 聚合 delta updates。这本质上是**离散的 slow EMA**——outer optimizer 每 500 步应用一次 momentum，相当于 β ≈ 1 - 1/500 ≈ 0.998 的 EMA on the K-step-averaged gradient。作者猜测 DiLoCo 的成功部分来自这种 "slow momentum" 效应。

[DiLoCo paper](https://arxiv.org/abs/2311.08105) | [SlowMo paper](https://openreview.net/forum?id=HZeVx6YP1D)

### 6.2 Grokfast

[Grokfast](https://arxiv.org/abs/2405.20233) 用 EMA pre-filtering 放大 gradient 的 low-frequency 成分来加速 grokking。思想上类似——分离 fast/slow signal。但 Grokfast 是 nested EMA（EMA on EMA），会衰减 recent gradients（见 Fig. 31），与 AdEMAMix 目标相反。

### 6.3 AggMo

[AggMo (Lucas et al. 2019)](https://openreview.net/forum?id=Hkx-oFZq) 用 K 个 momentum 的 sum，证明大 β 可行。但 AggMo 是 SGD 基础上，没有 Adam 的 second moment + 没有针对 LLM scale 设计 scheduler。AdEMAMix 在大模型上证明 gains 远超 AggMo。

### 6.4 Sophia

[Sophia](https://arxiv.org/abs/2305.14342) 用 Hessian-based preconditioner 来 normalize step size，思路是 second-order。AdEMAMix 是 first-order 的，但通过 slow EMA 间接实现了**类似 long-horizon trust region** 的效果——slow momentum 在低曲率方向放大 step，高曲率方向自抵消。

### 6.5 Polyak / Nesterov momentum

经典 momentum（β=0.9）对应 t_half ≈ 6。AdEMAMix 的 fast EMA 等价于 Polyak momentum，slow EMA 是新引入的"long-horizon Polyak"。Polyak 1964 原始论文证明 momentum 在 quadratic 上达到 optimal convergence rate，但 [Goujaud et al. 2023](https://arxiv.org/abs/2307.11291) 证明在一般 non-convex 上**不保证 acceleration**——single momentum 的局限。

---

## 7. Limitations & Open Questions

1. **Memory overhead**: 多一份 m2，与参数同 size。但 β1=0 可以省掉 m1（用 g 替代），memory 与 AdamW 持平。App. C.1.7 显示 β1=0 在 110M 上 work，1.3B 上 final loss 甚至略好，但 spike 多。
2. **Short-training unfriendly**: β3=0.9999 需要 ~7k 步才填满 m2。T < 64k 时用 β3=0.999 更好（Fig. 17）。
3. **Distribution shift**: 长 memory 对 sudden shift 不友好，slow EMA 会拖后腿。Continual learning 可能需要 decay m2。
4. **Theoretical gap**: 
   - 为什么 gradients 在几十千步后仍 relevant？loss landscape 的 local linearity 持续时间远超预期
   - Variance reduction 视角：large momentum 减小 variance，可能损害 generalization（[Ghosh et al. 2023](https://openreview.net/forum?id=MyxH8x3ja)），但 AdEMAMix 没观察到这个 degradation
   - Forgeting 与 optimization efficiency 的因果关系未完全建立——可能两者都源于"iterate 移动慢"这一共同因素

5. **β3 = 0.99999 的边际收益**: Fig. 12b 显示 β3=0.99999 不如 0.9999，说明有个 sweet spot，再大收益递减。可能是因为太老的 gradients 终究 irrelevant，或者 noise 累积超过 signal。

6. **与 schedule coupling**: AdEMAMix 的 gains 在 cosine decay 下显著，constant LR + linear decay 下也 work（Fig. 3b, 28），但 magnitude 不同。LR schedule 与 momentum schedule 的相互作用值得深挖。

---

## 8. PyTorch 实现要点（App. A.2）

[官方实现](https://github.com/apple/ml-ademamix) 关键代码片段：

```python
m1.mul_(beta1).add_(grad, alpha=1 - beta1)
m2.mul_(beta3).add_(grad, alpha=1 - beta3)
nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

denom = (nu.sqrt() / math.sqrt(bias_correction2)).add_(eps)
update = (m1.div(bias_correction1) + alpha * m2) / denom
update.add_(p, alpha=lmbda)  # weight decay
p.add_(-lr * update)
```

注意：
- `m2` 不做 bias correction
- `m1 / bias_correction1` + `alpha * m2` 一起除以 `denom`
- FSDP 可以分布 optimizer states 缓解 memory

---

## 9. 对 LLM 训练的实际 takeaway

如果你在训 1B+ LLM：
1. **直接 try AdEMAMix**, β3=0.9999, α=8, T_α=T_β3=T (total steps)
2. 保持 AdamW 的 β1=0.9, β2=0.999, weight decay=0.1 不变
3. LR 可能需要略降（1.3B 上 AdEMAMix 用 3e-4 vs AdamW 用 5e-4）
4. Memory: 如果 tight，设 β1=0，省一份 m1 buffer
5. From-AdamW-switch 也可以 work：m2 init=0, no scheduler, 越早切越好（Fig. 5b, c）
6. **不要**天真地增大 AdamW 的 β1——这篇文章证明它不 work

---

## 10. 个人 speculative 思考（build your intuition）

几个值得深挖的方向：

**(a) Slow EMA as implicit EMA on weights**: 每 ~7000 步, m2 ≈ average of last 7000 gradients。如果 LR scheduler 让最后 7000 步占据 weight change 的大部分，m2 在某种意义上是 "iterate trajectory 的方向导数"。这与 EMA on weights (Polyak averaging, [Izmailov et al. SWA](https://arxiv.org/abs/1803.05407)) 有关联，但作用在 gradient space。

**(b) Connection to Kalman filtering**: 两个 EMA with 不同 β 类似于两个 Kalman filter with 不同 process noise assumption。slow EMA 假设系统 quasi-static，fast EMA 假设 system 快变。Mixture 类似 multi-model Kalman。

**(c) Long-tail gradient relevance**: gradients 在几万步后仍 relevant 暗示 loss landscape 在训练大部分时间里**局部 quasi-convex**，否则 gradients 会 rapidly 反转方向使 EMA 自抵消。这给 [loss landscape smoothness](https://arxiv.org/abs/1712.09913) 研究提供新证据。

**(d) Implicit curriculum**: slow EMA 让"早期看到的 batch"持续贡献方向。结合 forgetting 实验，这暗示 LLM 训练中**早期 batch 的 gradient signal 价值很高**，single EMA 浪费了这部分 signal。这与 [data curriculum](https://arxiv.org/abs/2305.02217) 研究可能联动。

**(e) Mixture-of-EMAs as a primitive**: 更一般地，可以用多个 EMA with 不同 β 形成 "gradient memory hierarchy"，类似 CPU cache (L1/L2/L3)。slow EMA 是 L3 cache, fast EMA 是 L1。这种 hierarchical structure 在 biological neurons (multiple timescale plasticity) 中也存在。

---

## Web References

- [AdEMAMix arXiv (pagliardini et al.)](https://arxiv.org/abs/2409.03437)
- [Official code (Apple ml-ademamix)](https://github.com/apple/ml-ademamix)
- [Adam original paper](https://arxiv.org/abs/1412.6980)
- [AdamW](https://arxiv.org/abs/1711.05101)
- [Lion](https://arxiv.org/abs/2302.06675)
- [Sophia](https://arxiv.org/abs/2305.14342)
- [AggMo](https://openreview.net/forum?id=Hkx-oFZq)
- [DiLoCo](https://arxiv.org/abs/2311.08105)
- [SlowMo](https://openreview.net/forum?id=HZeVx6YP1D)
- [Grokfast](https://arxiv.org/abs/2405.20233)
- [Goujaud et al. on heavy-ball non-acceleration](https://arxiv.org/abs/2307.11291)
- [Polyak 1964 momentum](https://www.sciencedirect.com/science/article/pii/0041555363903823)
- [RedPajama v2 dataset](https://github.com/togethercomputer/RedPajama-Data)
- [FineWeb dataset](https://arxiv.org/abs/2406.17557)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [ViT original](https://arxiv.org/abs/2010.11929)

---

**Final takeaway**: 这篇 paper 用一个极简的改动（多加一份 EMA buffer）撬动了 LLM 训练中接近 2x 的 token efficiency，且 mechanism 清晰（fast/slow decoupling）、ablation 详尽、negative results 诚实。它暴露了 Adam 家族 single-EMA 设计的盲点，也打开了一扇门——**gradient memory 的结构化设计**可能是下一个 optimizer frontier。
