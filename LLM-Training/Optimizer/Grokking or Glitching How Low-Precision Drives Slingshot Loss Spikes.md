---
source_pdf: Grokking or Glitching How Low-Precision Drives Slingshot Loss Spikes.pdf
paper_sha256: 0b3f62fa4cca502263f4abe8dc54d81b463d040989ee1db29defff67f7e0b5b8
processed_at: '2026-08-04T22:24:06-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说：Slingshot 到底是个什么鬼

## 一句话版本

神经网络训练里那些周期性出现的 loss 突刺（Slingshot），**不是优化器在搞什么玄学**，是浮点数精度不够把一个本该对冲为零的梯度悄悄掰歪了一点点，这点歪在 Neural Collapse 的几何结构里被放大成指数增长的正反馈环，最后把参数一脚踢飞。

## 一个生活类比

想象你和一个朋友站在一艘小划艇上，两人本来应该站在船的正中间保持平衡。

**正常情况**（精确算术）：你往左偏一点，船往右歪一点，水的浮力把你推回来。两个力严格对冲，船不动。这是 CE loss 的 **zero-sum constraint**——对正确类和错误类的梯度加起来严格为零，全局平均参数 $W_G$ 像船的重心一样纹丝不动。

**精度不够的情况**（float32）：你往左偏的那一下太小了（正确类概率和 1 之间的差，大约 $10^{-7}$ 量级），船的传感器检测不到，只记录了"你朋友往右偏"。于是系统以为"只有朋友在偏"，开始把船往左修正。重心 $W_G$ 开始往一个方向漂。

**Neural Collapse 把单次漂移变成指数爆炸**：船一旦歪了，你站的姿态也被迫跟着调整（feature h 收到来自 $W_G$ 的反作用力）。你调整的姿态又让船歪得更厉害。两人互相"我以为你不动所以我来补"，越补越歪，方向反向平行，幅度指数增长。这就是 **NFI（Numerical Feature Inflation）**。

**最后的 spike**：船摆得越来越猛，某个时刻你的脚突然踩到一块石头（某个样本的 margin 跌破 16 这个 float32 阈值，正确类梯度突然从 0 变回有限值）。但是！你们俩已经习惯了"船永远在动"的节奏，Adam 优化器内部的"有效学习率"早就被推到天上（因为长期没信号，分母 $v_t$ decay 到极小）。突然来了一个真信号，乘上巨大的有效学习率，等于用全力踹一脚——船直接翻掉，loss 飙到 random guessing 水平。

然后船重新扶正，开始下一轮摆动。这就是周期性 Slingshot。

---

## 为什么这个类比是对的

Paper 的三个定理对应类比里的三步：

### 第一步：传感器失灵 = Softmax Collapse

float32 有 24 个有效 bit。两个数相加，如果一个比另一个小 $2^{23}$ 倍以上（约 $1.19\times 10^{-7}$），小的那个被"吃掉"——这就是 **absorption error**。

CE loss 里算 $Z = z_m + \log\sum_k \exp(z_k - z_m)$。当正确类 logit $z_r$ 比其他类高出 16 以上（即 $\ln 2 \times 23 \approx 16$），后面那一坨全被吃掉，$Z = z_r$。

这时正确类梯度 $g_r = e^{z_r - Z} - 1 = 0$。本来不该是 0 的，被浮点数 round 成 0。

### 第二步：对冲失效，重心漂移 = Theorem 3.4

理想算术下，所有类的梯度加起来：
$$\sum_k \nabla_{W_k}\mathcal{L} = \left(\sum_k \hat{y}_k - \sum_k y_k\right)h = (1-1)h = 0$$

意思就是"船左右受力严格抵消"。但 SC 下 $g_r$ 被砍成 0：
$$\sum_k \nabla_{W_k}\mathcal{L} \to \sum_{k\neq r}\hat{y}_k \cdot h = \epsilon \cdot h$$

其中 $\epsilon = \sum_{k\neq r}\hat{y}_k$ 是错误类上残留的概率质量（很小但不为零）。

在 class-balanced batch + NC 假设下，paper 证明：
$$\mathbb{E}[\Delta W_G] = -\frac{\eta\epsilon}{K}\mu_G$$

变量含义：
- $W_G = \frac{1}{K}\sum_k W_k$：所有类别 classifier 权重的平均（船的重心）
- $\mu_G = \frac{1}{B}\sum h$：batch 内 feature 的平均（你站的位置）
- $\eta$：学习率
- $\epsilon$：错误类残留概率
- $K$：类别数

公式说什么：**重心 $W_G$ 朝着 $-\mu_G$ 方向漂移**。船重心往你对面那一边偏。

### 第三步：几何把漂移变成指数爆炸 = Theorem 3.7

$W_G$ 一旦不为零，feature h 的梯度也会被它影响：
$$\text{Proj}_{W_G}(\nabla_h \mathcal{L}) = \epsilon W_G$$

意思是 feature 也被朝 $W_G$ 方向推。但 $W_G$ 朝 $-\mu_G$，所以 feature 被朝 $+\mu_G$ 推——**feature 增长方向和 $W_G$ 漂移方向相反**。

写成耦合动力学：
$$W_G^{(t+1)} = W_G^{(t)} - \frac{\eta\epsilon}{K}\mu_G^{(t)}$$
$$\mu_G^{(t+1)} = \mu_G^{(t)} - \eta\epsilon \cdot W_G^{(t)}$$

这是个 2D 耦合线性系统，矩阵 $\begin{bmatrix}1 & -\alpha \\ -\beta & 1\end{bmatrix}$。它的最大特征值 $\lambda_1 = 1 + \eta\epsilon/\sqrt{K} > 1$，对应特征向量 $W_G = -\mu_G/\sqrt{K}$（反向平行）。

所以：
$$\|W_G^{(t)}\| \propto \left(1 + \frac{\eta\epsilon}{\sqrt{K}}\right)^t \quad \text{指数增长}$$
$$\cos(W_G, \mu_G) \to -1 \quad \text{反向平行}$$

这就是 Figure 1b/c 看到的现象：$\|\mu_G\|$ 进入快速增长阶段，$W_G$ 和 $\mu_G$ 的 cosine 接近 -1。

### 第四步：触发 spike

指数增长不直接等于 spike。还需要一个 trigger：

1. 指数增长让所有 logit 整体压低（因为 $W_G^T\mu_G \to -\infty$ 是 logit 的公共项）
2. 但 class-mean 之间的 margin 还在 classification subspace 里被错误类小梯度维持
3. **outlier samples** 沿 $\mu_G$ 方向跑得更快（因为它们的 $\epsilon$ 大，由 Proposition 3.6），intra-class variance 相对膨胀
4. 某个 outlier 的 sample-level margin 跌破 16 → SC 解除 → 正确类梯度复活
5. Adam 的 effective LR 早已被推到 $\eta/\varepsilon_{Adam} \approx 10^5$ 倍
6. 复活的小梯度 × 巨大 effective LR = 巨大 update → loss spike

Paper 在 Section 4.1 给了一个超漂亮的定量验证：
- Spike 前 gradient ~ $3\times 10^{-9}$
- Spike 时 re-emerged gradient ~ $1.19\times 10^{-7}$（正好等于 $e^{-16}$，float32 absorption 阈值）
- Adam first moment $m_t \approx 1.46\times 10^{-8}$
- Adam second moment $\sqrt{v_t} \approx 2.7\times 10^{-8}$
- Update $\approx 4\times 10^{-4}$

实验观察：spike 时确实在 $\pm 4\times 10^{-4}$ 出现两个 sharp mode，比平时 $10^{-5}$ 大 40 倍。理论预测和实验数字吻合到量级。

---

## 最干净的证据：float64 一招毙命

Paper 的"smoking gun" 在 Figure 1a：

- 全 float32 训练 → 周期性 spikes
- 全 float64 训练 → spikes 消失
- **只把 logits 和 loss 计算改 float64，其他还 float32** → spikes 也消失

这就直接钉死了：instability 来自 loss 计算的精度，**不是 optimizer 的 intrinsic 问题，也不是 model 架构的 intrinsic 问题**。

float64 的 absorption 阈值是 $2^{-52} \approx 2.22\times 10^{-16}$，实际训练根本到不了那个 confidence level，所以 SC 永远不触发，NFI 反馈环永远启动不了。

---

## 干预实验像拼图一样互相印证

Paper 做了一系列干预，每一个都从不同环节切断同一个反馈环：

| 干预 | 切断的环节 | 效果 |
|------|------------|------|
| loss 用 float64 | 不让 SC 发生 | spike 消失 |
| 把 logit gradient 投影到 zero-sum 子空间 | 不让 zero-sum 被打破 | spike 消失 |
| BatchNorm 放在 classifier 前 | 减掉 $\mu_G$ 主漂移分量 | spike 消失 |
| 增大 Adam $\varepsilon$ 到 $10^{-5}$ | 限制 effective LR 上限 | spike 消失 |
| Label Smoothing | 让 $\hat{y}_r < 1$，SC 不触发 | NFI spike 消失，但引入 EOS spike（Hessian 不再 vanish） |
| Weight decay | 限制 logit 范围 | spike 消失 |

**多个不同干预都指向同一机制**，这是 mechanism 真实性的最强证据。如果 Slingshot 是某种混沌的 intrinsic dynamics，不会这么容易被多个不同角度的干预精准打掉。

---

## 反直觉的几个点

### LayerNorm 反而加速 spike

LN 是 per-sample normalize，**不阻止集体 $\mu_G$ 对齐**——所有 sample 可以一起朝同一方向 drift，LN 看不出来。而且 LN 约束 feature norm，让 angle 的变化更容易触发 absorption 阈值。

Thilak et al. 之前观察到 LN 下 last layer norm 是 stepwise 增长，paper 给出机理：LN 把 direction 和 magnitude decouple，模型先优化 angle，angle 稳定后再增大 scalar scale，形成 alternating phase。这就解释了 stepwise pattern 的来源。

### ResNet18 是唯一不 spike 的架构

Table 1 里 Transformer、MLP、VGG11、ViT 都 spike，唯独 ResNet18 不 spike。

原因：NFI 指数增长的前提是 $\epsilon$（错误类残留概率）比 $W_G$ 慢得多。ResNet18 学得快，$\epsilon$ 以 $1/t$ 或 $\log t / t$ 速度下降，累积因子 $(1 + \eta\epsilon/\sqrt{K})^t$ 只能 polynomial 增长，指数爆炸起不来。

其他架构 $\epsilon$ 在 spike 前几乎停滞，指数因子得以累积。这是一个非常 architecture-specific 的预测，很优雅。

### 小 learning rate 更易 spike

大 LR 的 noise 让模型逃出 sharp local minimum；小 LR 让模型忠实收敛到最近的极小，而那些极小常常很 sharp，更容易触发 absorption 阈值。

### Label Smoothing 的陷阱

LS 让目标 $y_r = 1 - \alpha < 1$，CE 的 global minimum 在 finite logit 处达到，SC 不触发，NFI spike 消失。

**但是！** Paper 证明（Theorem A.1 + C.4.3）：标准 CE 下 interpolation 时 Hessian $\lambda_{max} \to 0$，优化在远低于 stability threshold 的地方运行；LS 下 $\text{tr}(H_z) = 1 - \|y^{LS}\|^2 = 2\alpha - \alpha^2(1 + 1/(K-1)) > 0$，Hessian 不 vanish，反而进入 EOS regime。

所以 LS 消除了 NFI spike，却引入了 EOS spike。这是非常 subtle 的 trap。

---

## LLM 里的发现

### SC 在 LLM 中很普遍

nanoGPT 110M + FineWeb，每 step ~13万 token 中约 4000 个 token loss 严格为 0。Top-10 collapse 频率 token：".", "org", "example", ",", "to", "t", "last", " ", " you", "of"。

很多不是高频 token，而是 **数据集 template 造成的可预测上下文**：`example.org` 里的 "org"，表单字段 "first name" / "last name" 里的 "last"。

这是 mechanistic interpretability 的一个有趣入口——SC 的位置揭示了 dataset 的结构性 redundancy。

### LLM 的 logit divergence 方向与分类任务相反

分类任务里 $W_G^T\mu_G \to -\infty$，logit 整体压低。LLM 里：
- float32 训练 1e5 步后 mean logit = 183
- float64 训练同样步数后 mean logit = 498（**更高**）

为什么相反？自然语言 token 频率服从 **Zipf's law**，output embedding $W_k$ 内在 imbalanced toward 高频 token，本身就造出 large $W_G$；features 和 output embedding 在**同一方向**互相加强（frequent-token embedding 引导对应 feature 沿对齐方向增长）。

而 NFI 是 **anti-parallel** 的——所以低精度反而部分抑制了 Zipf-induced 的更快 divergence。

### 统一干预：去掉 last-layer mean

不论 NFI 还是 Zipf-induced 机制，根本因素都是 $W_G$。训练中减掉 $W_G$（或用 BN before classifier，或在 LLM 中 subtract output embedding mean）能同时抑制两种异常 logit 增长。

这对 nanoGPT / micrograd 这类教学实现是直接提示：**output embedding centering 是非常便宜但能提升长期训练稳定性的 trick**。

---

## 对实际训练的意义

### 为什么常规训练里看不到 Slingshot

1. **Weight decay** 限制 logit 范围，absorption 阈值达不到（Xie & Li 证明 AdamW 下 $\|w\|_\infty \lesssim 1/\lambda$）
2. **Mini-batch** 引入 stochastic noise，让模型难以精确达到 absorption threshold；同时 implicit regularization 偏向 flat minima，打破 full-batch 的 "silent regime"

但 VGG11 + batch size 256 训练 1e6 steps 后仍有约一半 sample 进入 SC——没有可见 spike，但 NFI 仍暗中运作，导致 late-stage logit 异常增长。**NFI 不等于 visible spike，但仍在影响 optimization trajectory**。

### 低精度训练的潜在风险

随着 LLM 训练向 BF16/FP8/FP4 推进，absorption threshold 急剧缩小：
- float32: ~$10^{-7}$
- bf16: ~$10^{-3}$
- fp8 e4m3: ~$10^{-1}$

NFI 触发需要的 confidence level 大幅降低，可能是未来大规模低精度训练中被低估的 instability source。其他已知的低精度 instability 源（matmul quantization error、attention sink、Flash Attention rounding 累积）都和 NFI 不同——NFI 发生在 log-probability 计算中。

---

## 局限性

Paper 自己承认：分析基于 Unconstrained Feature Model 假设，假设 backbone 足够 expressive 任意 feature 都能生成。浅网络可能不适用——这解释了 Nanda et al. 报告的浅网络不易观察到 Slingshot。

我额外想到几点：
1. NFI 是 spike 的充分条件还是必要条件？Paper 证明了 SC+NC → NFI → spike，但没排除其他 numerical artifact 也可能造 spike。
2. Adam 换成 vanilla GD（手动 LR=$10^5$）也能在相同时机触发 spike，但 GD 之后无法 re-converge。这暗示 NFI 的 trigger 不依赖 adaptive，但**周期性 spike 的 pattern 依赖 Adam 的 moment adaptation**——这块更深入的耦合分析还可以做。
3. EMA / Lion / Shampoo 等其他 optimizer 的 NFI 表现如何？
4. 能不能在 LLM 内部识别出哪些 attention head 或 MLP neuron 在承担 $\mu_G$ drift 的"载体"角色？这会让 paper 从 last-layer 分析扩展到 mechanistic circuit 层面。

---

## 我的核心 takeaway

1. **Loss computation 的 finite precision 是 first-order factor**，长期训练动力学分析不应忽略它，gradient flow 视角会 miss 这个。
2. **SC 打破 zero-sum 是一切的种子**——一个非常 subtle 的精度问题，能通过 NC 几何放大成指数增长的 feedback loop。这种"小数值 + 几何结构 → 大动力学"的 pattern 在深度学习中可能还有更多类似案例。
3. **NC 与 SC 的相互作用是关键**——单看 SC（Prieto et al.）只看到 plateau，单看 NC（Papyan et al.）只看到 geometry，**两者耦合才造出 NFI**。
4. **Intervention 的层次**：BN before classifier、去 $W_G$、增 $\varepsilon$、float64 loss——它们从不同层面切断了同一个 feedback loop 的不同环节，这种"multi-pronged 一致性"是 mechanism 真实性的强证据。

paper 最漂亮的地方在于：把一个被归为 intrinsic optimization 的神秘现象（Slingshot）归约到一个**具体可证的数值机制**，且干预实验极其干净、互相印证。Theorem 3.4 + 3.6 + 3.7 三个定理形成完整因果链，Adam update magnitude 的定量预测和实验观察一致到量级。这是我读过最 satisfying 的 mechanistic optimization 工作之一。

---

## 参考

主 paper：Liu et al. "Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes"

背景：
- Power et al. 2022 "Grokking" https://arxiv.org/abs/2201.02177
- Thilak et al. 2022 "The slingshot mechanism" https://arxiv.org/abs/2206.04817
- Nanda et al. 2023 "Progress measures for grokking" https://openreview.net/forum?id=2y3hOUuOcV
- Prieto et al. 2025 "Grokking at the edge of numerical stability" https://openreview.net/forum?id=6PqR1idJws

Neural Collapse：
- Papyan, Han, Donoho 2020 https://www.pnas.org/doi/10.1073/pnas.2015509117
- Mixon et al. "NC with unconstrained features" https://link.springer.com/article/10.1007/s43670-022-00007-1

Edge of Stability：
- Cohen et al. 2021 https://openreview.net/forum?id=jfRTy3sJE8
- Ma, Wu, E 2022 https://proceedings.mlr.press/v145/ma22a.html

LLM logit divergence：
- Wortsman et al. 2024 https://openreview.net/forum?id=ouxytrC9J5
- Stollenwerk et al. 2026 https://arxiv.org/abs/2601.02031
- Gao et al. 2019 "Representation degeneration" https://openreview.net/forum?id=HJlNzA5Ym

低精度训练：
- Wortsman et al. 2023 https://openreview.net/forum?id=rp6AicF4UN
- Xiao et al. 2024 "Attention sinks" https://openreview.net/forum?id=EfJcZDlZaj

---

# Grokking or Glitching? Slingshot Mechanism 的数值本质

Andrej，这篇 paper 我读完之后非常兴奋——它做的事情非常符合你的口味：**把一个看起来很神秘的 optimization dynamics 现象（Slingshot），用一个具体的、可证伪的 numerical mechanism（floating-point absorption error）解释清楚**。这就像 grokking 的 mechanistic interpretability 一样，是从"现象学描述"走向"机理归因"的工作。下面我尽量详细地 build 你的 intuition。

---

## 1. 背景：Slingshot 与 Grokking 的纠葛

**Grokking** 是 Power et al. (2022) 在小算法数据集（modular arithmetic）上观察到的现象：模型在 training accuracy 达到 100% 很久之后，test accuracy 才突然跃升到 100%。

**Slingshot Mechanism** 是 Thilak et al. (2022) 在研究 grokking 时观察到的伴随现象：在无 weight decay、CE loss、Adam optimizer 的长期训练中，会出现周期性的 training loss spikes，每次 spike 前都伴随着 last-layer parameter norm 的近指数增长。

之前的解释路径：
- Thilak et al. 把它和 **Edge of Stability (EOS)** 联系起来，认为是 optimizer 周期性穿越 stability boundary。
- Nanda et al. (2023) 在 mechanistic interpretability 工作中注意到 lower precision 会加剧 slingshot，但没给出因果机制。
- Prieto et al. (2025) ICLR 工作识别出了 **Softmax Collapse (SC)**：当 `z_m - max_{k≠m} z_k > (p-1) ln 2`（float32 中约 16）时，PyTorch 的 log-sum-exp 实现里 `exp(z_k - z_m)` 会被 absorption error 吃掉，导致正确类 logit 的梯度严格 round 到 0。但他们停在这里，只是假设 Slingshot 是 optimizer 为了"逃避" SC 的 intrinsic response。

**这篇 paper 的核心 claim**：Slingshot 根本不是 intrinsic optimization 现象，而是 SC 与 Neural Collapse 几何相互作用产生的一个数值反馈环——他们命名为 **Numerical Feature Inflation (NFI)**。把这个数值机制去掉（哪怕只把 loss 计算改成 float64），Slingshot 就消失了。

---

## 2. 核心机制：从 Absorption Error 到 NFI 反馈环

### 2.1 Absorption Error 与 Softmax Collapse

IEEE 754 float32 有 1 个 sign bit、8 个 exponent bit、23 个 mantissa bit（加上 implicit leading 1，有效精度 p=24）。两个浮点数 a, b 相加（|a|≥|b|），需要做 exponent alignment。当 `|b|/|a| < 2^{-(p-1)}`（float32 中约 `1.19e-7`），b 在 mantissa 对齐后落在表示范围之外，结果 `a + b = a`，这就是 absorption error。

PyTorch CE loss 用 log-sum-exp trick：
$$Z = \log \sum_k \exp(z_k) = z_m + \log \sum_k \exp(z_k - z_m)$$

其中 $z_m = \max_k z_k$。当 `z_m - max_{k≠m} z_k > (p-1) ln 2`（float32 约 16），第二个 log 项里所有 `exp(z_k - z_m)` 都被 absorption 吃掉，于是 `Z = z_m`。

此时正确类 r 的梯度：
$$g_r = \hat{y}_r - y_r = e^{z_r - Z} - 1 = e^{z_r - z_r} - 1 = 0$$

但错误类 k≠r 的梯度 `g_k = e^{z_k - z_r} ≠ 0`（虽然很小）。

### 2.2 关键观察：SC 打破了 Zero-Sum Constraint

在精确算术下，CE loss 对最后一层 classifier row $W_k$ 的梯度满足 zero-sum constraint：
$$\sum_{k=1}^K \nabla_{W_k} \mathcal{L} = \left(\sum_k \hat{y}_k - \sum_k y_k\right) h = (1 - 1) h = 0$$

这意味着 global classifier mean $W_G = \frac{1}{K}\sum_k W_k$ 在理想算术下不会移动。

但 SC 下，`g_r` 被 round 到 0，sum 不再抵消：
$$\sum_{k=1}^K \nabla_{W_k} \mathcal{L} \xrightarrow{SC} \sum_{k\neq r} \hat{y}_k \cdot h = \epsilon \cdot h$$

其中 $\epsilon = \sum_{k\neq r} \hat{y}_k$ 是错误类上的 residual probability mass。这就是 zero-sum constraint 被打破的根源。

### 2.3 Theorem 3.4：W_G 的 Drift

**Theorem 3.4**: 在 NC 状态 + SC 条件下，batch size B、learning rate η、class-balanced batch 上：
$$\mathbb{E}_B[\Delta W_G] = -\frac{\eta \epsilon}{K} \mu_G$$

其中：
- $W_G = \frac{1}{K}\sum_{k=1}^K W_k$：global classifier mean
- $\mu_G = \frac{1}{B}\sum_{k,i} h_{k,i}$：batch 内 penultimate feature 的 global mean
- $\epsilon = \mathbb{E}[\sum_{k\neq r}\hat{y}_k]$：错误类 residual probability mass
- η：learning rate
- K：类别数

**证明直觉**（Appendix C.1）：
1. 单样本对 $W_G$ 的梯度 = $\frac{1}{K}\sum_k (\hat{y}_k - y_k) h$，理想下 = 0。
2. SC 下变成 $\frac{\epsilon}{K} h$。
3. Batch 上聚合：$\Delta W_G = -\frac{\eta}{KB}\sum_i \epsilon_i h_i$。
4. 由 NC1（intra-class variability collapse），$h_i \approx \mu_k = \mu_G + \mu_k^*$。
5. 由 NC2（Simplex ETF），$\sum_k \mu_k^* = 0$。
6. Class-balanced batch 下 $\mathbb{E}[\sum_i h_i] = B\mu_G$。
7. 假设 $\epsilon_i \approx \epsilon$ 跨 batch 近似常数，得到 $-\frac{\eta\epsilon}{K}\mu_G$。

这个定理说：**W_G 朝着 $-\mu_G$ 方向 drift**。这非常关键——它给了后续反馈环一个明确的方向。

### 2.4 Proposition 3.6：Feature 也被推着走

W_G 变非零后，原本 NC3 的 self-duality 失效，需要重新定义 NC3'：centered weights $W_k^* = W_k - W_G$ 与 $\mu_k^*$ 对齐形成 ETF。

**Proposition 3.6**: 假设 $W_G \perp \text{span}\{W_k^*\}$，SC 下，feature h 的梯度沿 $W_G$ 方向的投影：
$$\text{Proj}_{W_G}(\nabla_h \mathcal{L}) = \epsilon W_G$$

**证明直觉**（Appendix C.2）：
$$\nabla_h \mathcal{L} = \sum_k (\hat{y}_k - y_k) W_k = \sum_k (\hat{y}_k - y_k)(W_G + W_k^*)$$
SC 下正确类项消失：
$$\nabla_h \mathcal{L} \approx \sum_{k\neq r} \hat{y}_k (W_G + W_k^*) = \epsilon W_G + \sum_{k\neq r}\hat{y}_k W_k^*$$
由于 $W_G \perp W_k^*$，投影到 $W_G$ 上得到 $\epsilon W_G$。

这意味着 feature h 的 update `h ← h - η∇_h L` 在 $W_G$ 方向上加了一个 $\eta\epsilon W_G$ 的分量。由于 $W_G$ 朝 $-\mu_G$ 方向，所以 feature 被朝 $+\mu_G$ 方向推。

### 2.5 Theorem 3.7：NFI 反馈环——指数增长

把 Theorem 3.4 和 Proposition 3.6 串起来，得到一个 coupled linear dynamical system（Appendix C.3）：
$$W_G^{(t+1)} = W_G^{(t)} - \alpha \mu_G^{(t)}, \quad \alpha = \frac{\eta\epsilon}{K}$$
$$\mu_G^{(t+1)} = \mu_G^{(t)} - \beta W_G^{(t)}, \quad \beta = \eta\epsilon$$

写成 block matrix：
$$\begin{bmatrix} W_G^{(t+1)} \\ \mu_G^{(t+1)} \end{bmatrix} = \begin{bmatrix} I & -\alpha I \\ -\beta I & I \end{bmatrix} \begin{bmatrix} W_G^{(t)} \\ \mu_G^{(t)} \end{bmatrix} = M u^{(t)}$$

特征值：
$$\lambda_1 = 1 + \sqrt{\alpha\beta} = 1 + \frac{\eta\epsilon}{\sqrt{K}} > 1$$
$$\lambda_2 = 1 - \sqrt{\alpha\beta} = 1 - \frac{\eta\epsilon}{\sqrt{K}} < 1$$

- $\lambda_1 > 1$ 是不稳定模，对应特征向量 $W_G = -\frac{1}{\sqrt{K}}\mu_G$（anti-parallel）
- $\lambda_2 < 1$ 是稳定模，对应 parallel 配置

随着 $t\to\infty$，$\lambda_1$ 模 dominate：
$$\lim_{t\to\infty} \cos(W_G^{(t)}, \mu_G^{(t)}) = -1$$
$$\|W_G^{(t)}\| \propto \left(1 + \frac{\eta\epsilon}{\sqrt{K}}\right)^t$$
$$\|\mu_G^{(t)}\| \propto \left(1 + \frac{\eta\epsilon}{\sqrt{K}}\right)^t$$

**Intuition**：这是一个 2D 耦合系统，矩阵 M 的几何意义是"轮换 + 微小扰动"，类似 $\begin{bmatrix}1 & -\alpha \\ -\beta & 1\end{bmatrix}$，它会做 90° rotation 加上 stretch，导致两向量交替"互相推 → 反向增长"。这正是 Slingshot 文献里看到的 anti-parallel alignment 现象的机理。

一个重要的额外结论（C.3.3）：feature 不仅整体膨胀，还在 $\mu_G$ 方向 rank-1 collapse：
$$\lim_{t\to\infty} \frac{\|h_\perp\|}{\|h_\|\|} \to 0 \implies \lim_{t\to\infty}\cos(h_t, \mu_G) \to 1$$

因为 parallel 分量的增长率 $\propto \|W_G\|$（指数），而 perpendicular 分量来自 ETF 加权和（linear bounded），parallel dominate。

---

## 3. 从 NFI 到 Loss Spike 的完整链路

光有指数增长还不等于 loss spike。Paper 在 3.3 节给出完整 trigger 链路：

1. **NFI 让 μ_G 和 W_G 指数膨胀**，logit 的公共项 $W_G^T \mu_G \to -\infty$，把所有 logit 整体压低。
2. **在 classification subspace**（垂直于 μ_G/W_G），剩余的错误类 gradient 仍在把不同类 mean 推开，所以 margin 在 class-mean 层面仍然安全。
3. **但 outlier samples** 沿 μ_G 方向跑得更快（因为它们 residual probability mass ε 更大，见 Proposition 3.6），intra-class variance 相对 inter-class variance 增长，某些样本的 sample-level margin 被压缩。
4. **当某个 outlier 的 margin 跌破 absorption threshold**（< 16 in float32），它的正确类梯度突然从 0 变回有限值。
5. **关键**：在这之前，正确类梯度长期为 0，错误类梯度极小（~$\exp(-16)$ 量级），Adam 的 second moment $v_t$ 已经 decay 到极小，effective learning rate $\eta/(\sqrt{v_t} + \varepsilon_{Adam})$ 被 amplified 到接近 $\eta/\varepsilon_{Adam}$（标准设置下可达 $10^5$ 倍）。
6. **突然恢复的有限梯度 × 巨大 effective LR = 巨大 update**，把 model 参数大幅踢飞，loss 爆炸回 random-guessing 量级。

### 3.1 一个非常漂亮的定量验证

Section 4.1 给了一个 scalar toy 估算：
- Adam 标准设置 $\eta=10^{-3}, \beta_1=0.9, \beta_2=0.95$
- Spike 前平均梯度 ~$3\times 10^{-9}$
- Spike 触发时 re-emerged gradient ~$\exp(-(p-1)\ln 2) = 1.19\times 10^{-7}$
- First moment: $m_t = 0.9 \times 3\times10^{-9} + 0.1 \times 1.19\times10^{-7} \approx 1.46\times10^{-8}$
- Second moment sqrt: $\sqrt{v_t} = \sqrt{9\times10^{-18}\times 0.95 + 1.19^2\times10^{-14}\times 0.05} \approx 2.7\times10^{-8}$
- Adam update: $\eta \cdot m_t / (\sqrt{v_t} + \varepsilon) \approx 4\times10^{-4}$

这与 Figure 2a 的实证观察惊人吻合：spike 前的 update ~$10^{-5}$，spike step 形成 $-4\times10^{-4}$ 和 $+4\times10^{-4}$ 两个 sharp mode。50 倍的 update 跳跃，损失被打回 $10^0$ 量级。

这是一个可证伪的预测：换 Adam 超参（如 $\beta_1, \beta_2 = 0.9, 0.999$ 或 Orvieto-Gower 推荐的 0.9, 0.9），spike 的 update magnitude 仍与该估算一致——paper 也验证了。

---

## 4. 实验验证

### 4.1 最干净的证据：float64 消除 Slingshot

Figure 1a 是 paper 的"smoking gun"：
- 全 float32 训练 → 出现周期性 spikes
- 全 float64 训练 → spikes 消失
- **只把 logits/loss 计算改 float64，参数仍 float32** → spikes 也消失

这证明 instability 来自 loss computation 的精度，不是 optimizer 或 model intrinsic。float64 的 absorption threshold 是 `2^{-52}` ≈ `2.22e-16`，实际训练根本达不到那个 confidence level。

参考：Prieto et al. "Grokking at the edge of numerical stability" ICLR 2025 https://openreview.net/forum?id=6PqR1idJws

### 4.2 Zero-Sum Projection 干预

Section 4.1 还做了一个非常聪明的设计实验：把 logit gradient $g = \nabla_z L$ 投影到 zero-sum 子空间：
$$g \leftarrow g - \frac{1}{K}\sum_k g_k$$

这强制每个 sample 的 gradient update 满足 Eq. (4) 的 zero-sum。结果 Slingshot 消失。这是 Theorem 3.4 的直接验证——zero-sum breaking 是 NFI 的必要条件。

### 4.3 架构依赖性：ResNet18 为什么特殊

Table 1 显示：Transformer、MLP、VGG11（±BN）、ViT 都出现 Slingshot，**唯独 ResNet18 没有**。

解释（Section 4.1 末 + Figure 2b）：NFI 指数增长的前提是 $|\dot{\epsilon}| \ll \frac{d}{dt}\|W_G\|$，即 ε 比 W_G 慢得多。
- MLP/VGG11/ViT：ε 在 spike 前缓慢下降或停滞 → 指数因子 $(1+\eta\epsilon/\sqrt{K})^t$ 能累积指数增长
- ResNet18：ε 快速下降（~$1/t$ 或 $\log t / t$）→ 累积只有 polynomial/logarithmic growth → NFI 起不来

ResNet18 学得快，能在 NFI 反馈环 dominate 之前逃出不稳定区。这是一个非常 elegant 的 architecture-dependent 预测。

### 4.4 Mitigation 实验汇总（Section 4.2, Figure 3）

| 干预 | 效果 | 机理 |
|------|------|------|
| Loss 用 float64 | 消除 | 抬高 absorption threshold |
| Zero-sum projection | 消除 | 阻断 NFI 触发条件 |
| 增大 Adam ε 到 $10^{-5}$ | 消除 | 限制 effective LR 上限 $\eta/\varepsilon$ |
| BatchNorm before classifier | 消除 | 减掉 $\mu_G$ 主漂移分量 |
| LayerNorm | **无效，反而加速** | LN 是 per-sample normalize，不阻止集体 μ_G 对齐；且约束 feature norm 让 angle 阈值更容易触发 |
| Label Smoothing | 消除 NFI，但引入 EOS | 让 $\hat{y}_r < 1$，Hessian 不再 vanish（见 C.4.3） |
| Weight decay | 消除 | 限制 logit 范围（Xie & Li 的 AdamW $\|w\|_\infty \lesssim 1/\lambda$） |

特别有意思的两个细节：

**LN 反而加速 spikes**——之前 Thilak et al. 也观察到 LN 下 last layer norm 是 stepwise 增长，paper 给出机理：LN decouple direction 和 magnitude，模型在 angle optimize 完之后才增大 scalar scale，形成 alternating phase。这正好解释了 Thilak 看到的 stepwise pattern。

**Label Smoothing 的双刃**——LS 让目标 $y_r < 1$，CE 的 global minimum 在 finite logit 处达到（$\hat{y} \to y^{LS}$），不会触发 SC。但 Theorem A.1 + C.4.3 证明：标准 CE 下 Hessian 在 interpolation 处 vanish（$\lambda_{max} \to 0$），所以优化在远低于 $2/\eta$ stability threshold 的地方运行；而 LS 下 $\text{tr}(H_z) = 1 - \|y^{LS}\|^2 = 2\alpha - \alpha^2(1 + 1/(K-1)) > 0$，Hessian 不再 vanish，反而进入 EOS regime。这是为什么 LS 后会看到新的、与精度无关的 spikes——非常 subtle 的 trap。

---

## 5. 与 EOS 的本质区别（Appendix A.2）

这是 paper 的一个重要澄清。EOS（Edge of Stability，Cohen et al. 2021 ICLR https://openreview.net/forum?id=jfRTy3sJE8 ）和 Slingshot 长得很像（都是 loss spikes），但：

1. **Hessian 行为**：CE 下 interpolation 时 Hessian $\lambda_{max} \to 0$（Theorem A.1 证明），远离 $2/\eta$ threshold，不该出现 EOS。
2. **精度敏感性**：EOS 是 landscape intrinsic 性质，float64 不会消失；Slingshot 是数值 artifact，float64 直接消失。
3. **共存的实验**（Figure 5）：窄网络（d=20）float32 下早期和晚期都有 spikes；float64 下早期 spikes（真 EOS）保留，晚期 spikes（Slingshot）消失。

**Theorem A.1 证明直觉**（Appendix C.4）：Hessian 分解成 GGN 项 $J^T H_z J$ 和 residual 项。CE+Softmax 的 $H_z = \text{diag}(\hat{y}) - \hat{y}\hat{y}^T$，$\|H_z\|_2 \leq \text{tr}(H_z) = 1 - \|\hat{y}\|_2^2$。Interpolation 时 $\hat{y} \to y$（one-hot），$\|y\|_2^2 = 1$，所以 $\|H_z\|_2 \to 0$，GGN 项 vanish。Residual 项含 $(\hat{y}_k - y_k)$，也 vanish。两项都 vanish，所以 $\lambda_{max}(H_\theta) \to 0$。

这个定理的意义：**CE loss 在晚期训练中，loss landscape 是 flatten 的，不存在 sharpness-driven instability**。这恰好是 Ma et al. 2022（https://proceedings.mlr.press/v145/ma22a.html ）的结论，但 paper 在此基础上加了一步：既然 landscape 不 sharp，那晚期 spike 一定来自数值。

---

## 6. LLM 中的 NFI（Section 4.3）

这部分对你应该特别有意思。Paper 在 nanoGPT（110M）+ FineWeb 上训练 100k steps（~13B tokens），观测到：

### 6.1 Softmax Collapse 在 LLM 中普遍存在

每 step 约 13万 token 中，约 4000 个 token loss 严格为 0（进入 SC）。Top-10 collapse 频率 token：".", "org", "example", ",", "to", "t", "last", " ", " you", "of"。

很多不是高频 token，而是 **数据集 template 造成的可预测上下文**：
- "example.org" 里的 "org"、"example"
- 表单字段 "first name", "last name" 里的 "last"

这是 mechanistic interpretability 角度看 SC 的一个有趣入口——SC 的位置揭示了 dataset 的结构性 redundancy。

### 6.2 LLM 的 logit divergence 方向与分类任务不同

分类任务里 $W_G^T \mu_G \to -\infty$，logit 整体压低（Figure 1c）。但 LLM 训练里：
- float32 训练 1e5 步后 mean logit = 183
- float64 训练同样步数后 mean logit = 498（**更高**！）

这与分类任务的"高精度抑制增长"完全相反。Paper 给的解释：自然语言 token 频率服从 **Zipf's law**，output embedding $W_k$ 内在 imbalanced toward 高频 token，这本身就能造出 large $W_G$（参考 Gao et al. 2019 "Representation degeneration problem" https://openreview.net/forum?id=HJlNzA5Ym ）；features 和 output embedding 在**同一方向**互相加强（frequent-token embedding 引导对应 feature 沿对齐方向增长），而 NFI 是 **anti-parallel** 的——所以低精度反而部分抑制了 Zipf-induced 的更快 divergence。

### 6.3 一个很诚实的复现失败声明

Appendix A.4 有一段很有意思的反思：他们试图复现 Wortsman et al. 2024 ICLR Oral（"Small-scale proxies for large-scale transformer training instabilities" https://openreview.net/forum?id=ouxytrC9J5 ）报告的"logits 漂向 $-\infty$"现象，无论是用 PyTorch 还是 Wortsman 团队使用的 NanoDo JAX/Flax 实现，都复现不出来。三年来也没有独立的第三方成功复现。他们怀疑这可能是 JAX/Flax 在 TPU 上的特殊行为，或者 paper 有未披露的实现细节。

他们观察到的 LLM logit divergence 方向是 **$+\infty$**，与 Stollenwerk et al. 2026 "Output embedding centering for stable LLM pretraining"（https://arxiv.org/abs/2601.02031 ）一致。

### 6.4 统一干预：去除 last-layer mean

不论是 NFI 还是 Zipf-induced 机制，**根本因素都是 $W_G$**。在训练中减掉 $W_G$（或用 BN before classifier，或在 LLM 中 subtract output embedding mean）能同时抑制两种异常 logit 增长。

这给你做 nanoGPT / micrograd 这类教学实现时一个直接的提示：output embedding centering 是一个非常便宜但提升长期训练稳定性的 trick。

---

## 7. Practical Implications

### 7.1 Mini-batch 抑制

Appendix A.3.2 解释为什么 Slingshot 在常规训练中罕见：
1. **Stochastic delay of convergence**：mini-batch noise 让模型难以精确达到 absorption threshold。
2. **Implicit regularization towards flat minima**：mini-batch SGD 偏向 flat 极小值（Keskar et al. 2017 https://openreview.net/forum?id=H1oy2l9gg ；Smith & Le 2018 https://openreview.net/forum?id=Byl unfamiliar ），打破 full-batch 的"silent regime"。

但 VGG11 + batch size 256 训练 1e6 steps 后仍有约一半 sample 进入 SC——没有 spike，但 NFI 仍暗中运作，导致 late-stage logit 异常增长。这意味着即使看不到 spike，NFI 仍可能在影响 optimization trajectory。

### 7.2 低精度训练的潜在风险

A.3.3 给了一个重要预警：随着 LLM 训练向 BF16/FP8/FP4 推进，absorption threshold 急剧缩小：
- float32: `2^{-23}` ≈ 1.19e-7
- float16/bf16: `2^{-10}` ≈ 9.77e-4（mantissa 11 bit 包括 implicit 1，p=11）
- fp8 e4m3: `2^{-3}` ≈ 0.125
- fp4: 量级更小

这意味着 NFI 触发需要的 confidence level 大幅降低，可能是未来大规模低精度训练中一个被低估的 instability source。其他已知的低精度 instability 源包括：
- Matrix multiply quantization error（Wortsman et al. 2023 https://openreview.net/forum?id=rp6AicF4UN ）
- Attention sink（Xiao et al. 2024 https://openreview.net/forum?id=EfJcZDlZaj ）
- Flash Attention rounding error accumulation（Qiu & Yao 2026 https://openreview.net/forum?id=... ）

NFI 是一个**新的、与上述都不同的 failure mode**——发生在 log-probability 计算中，而非 attention 或 matmul。

### 7.3 Learning rate 反直觉

A.1.2 一个反直觉结论：**小 learning rate 反而更易触发 Slingshot**。大 LR 的 noise 让模型逃出 sharp local minimum；小 LR 让模型忠实收敛到最近（常是 sharp）的极小，那里更易触发 absorption。

---

## 8. 局限性

Paper 的 Limitations 部分很诚实：
- 分析基于 Unconstrained Feature Model 假设（Mixon et al. 2022 https://link.springer.com/article/10.1007/s43670-022-00007-1 ），假设 backbone 足够 expressive 任意 feature 都能生成。浅网络可能不适用——这也解释了 Nanda et al. 报告的浅网络不易观察到 Slingshot。

我额外想到几点：
1. **NFI 是充分条件还是必要条件？** Paper 证明了 SC+NC → NFI → spike，但没排除其他 numerical artifact 也可能造 spike。比如 Flash Attention rounding 累积、attention sink 是否会通过类似机制产生 spike？
2. **Adam 之外**：他们把 Adam 换成 vanilla GD（手动 LR=$10^5$）也能在相同时机触发 spike，但 GD 之后无法 re-converge。这暗示 NFI 的 trigger 不依赖 adaptive，但**周期性 spike 的 pattern 依赖 Adam 的 moment adaptation**——这块更深入的耦合分析还可以做。
3. **EMA / Lion / Shampoo 等其他 optimizer** 的 NFI 表现如何？
4. **token-level NFI in LLM**：能不能在 LLM 内部识别出哪些 attention head 或 MLP neuron 在承担 $\mu_G$ drift 的"载体"角色？这会让 paper 从 last-layer 分析扩展到 mechanistic circuit 层面，可能正好接上你熟悉的 mech interp 路线。

---

## 9. 我的整体评价与 takeaway

**这篇 paper 最漂亮的地方**：
- 把一个被归为 intrinsic optimization 的神秘现象（Slingshot）归约到一个**具体可证的数值机制**，且干预实验（float64、zero-sum projection、BN、ε）极其干净、互相印证。
- Theorem 3.4 + 3.6 + 3.7 三个定理形成完整因果链，数学上 self-contained，假设清晰（NC + SC + class-balanced + W_G⊥span）。
- Adam update magnitude 的定量预测和实验观察一致到量级，是难得的"理论给具体数字"的例子。
- 对 LLM 复现失败的诚实声明，以及 NFI 与 Zipf-induced divergence 的区分，显示了科学态度。

**直觉上最值得吸收的几点**：
1. **Loss computation 的 finite precision 是 first-order factor**，长期训练动力学分析不应忽略它，gradient flow 视角会 miss 这个。
2. **SC 打破 zero-sum 是一切的种子**——一个非常 subtle 的精度问题，能通过 NC 几何放大成指数增长的 feedback loop。这种"小数值 + 几何结构 → 大动力学"的 pattern 在深度学习中可能还有更多类似案例。
3. **NC 与 SC 的相互作用是关键**——单看 SC（Prieto et al.）只看到 plateau，单看 NC（Papyan et al.）只看到 geometry，**两者耦合才造出 NFI**。
4. **Intervention 的层次**：BN before classifier、去 $W_G$、增 ε、float64 loss——它们从不同层面切断了同一个 feedback loop 的不同环节，这种"multi-pronged 一致性"是 mechanism 真实性的强证据。

**对你做教学/工程实现的启示**：
- nanoGPT / micrograd 这类教学实现里，CE loss 用 float32 在长期训练时是会埋雷的。把 loss 计算升 float64 是一行代码的事，能彻底避免一类隐藏的 instability。
- BN before final classifier（或减 $\mu_G$）是非常便宜的稳定 trick，特别是当你想 demo unregularized training 时。
- 如果你做 grokking demo，把 SC/NFI 这套机制讲清楚，会让学员对"数值精度如何塑造优化动力学"有非常 concrete 的感受。

---

## 参考与相关 reading

主 paper：
- Liu et al. "Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes" (本文)

Slingshot & Grokking 线：
- Power et al. 2022 "Grokking" https://arxiv.org/abs/2201.02177
- Thilak et al. 2022 "The slingshot mechanism" https://arxiv.org/abs/2206.04817
- Nanda et al. 2023 "Progress measures for grokking via mechanistic interpretability" https://openreview.net/forum?id=2y3hOUuOcV
- Liu et al. "Omnigrok" https://openreview.net/forum?id=d1dUJ4yX0P
- Kumar et al. "Grokking as transition from lazy to rich" https://openreview.net/forum?id=esSIeCKWF8

Softmax Collapse & 数值精度线：
- Prieto et al. 2025 "Grokking at the edge of numerical stability" https://openreview.net/forum?id=6PqR1idJws
- Xu et al. 2025 "Let me grok for you" https://openreview.net/forum?id=l0O0Z5eO0w

Neural Collapse：
- Papyan, Han, Donoho 2020 "Prevalence of neural collapse" https://www.pnas.org/doi/10.1073/pnas.2015509117
- Lu & Steinerberger 2022 "Neural collapse under CE" https://www.sciencedirect.com/science/article/pii/S1063520322000548
- Mixon et al. "NC with unconstrained features" https://link.springer.com/article/10.1007/s43670-022-00007-1

Edge of Stability & Adam instability：
- Cohen et al. 2021 "Gradient descent at edge of stability" https://openreview.net/forum?id=jfRTy3sJE8
- Cohen et al. "Central flows" https://openreview.net/forum?id=tc1N5InbQq
- Molybog et al. "Theory on Adam instability" https://arxiv.org/abs/2304.09871
- Bai et al. 2025 "Adaptive preconditioners trigger loss spikes" https://arxiv.org/abs/2506.04805
- Ma, Wu, E 2022 "Qualitative study of adaptive gradient algorithms" https://proceedings.mlr.press/v145/ma22a.html

LLM logit divergence & 低精度训练：
- Wortsman et al. 2024 "Small-scale proxies" https://openreview.net/forum?id=ouxytrC9J5
- Stollenwerk et al. 2026 "Output embedding centering" https://arxiv.org/abs/2601.02031
- Xiao et al. 2024 "Attention sinks" https://openreview.net/forum?id=EfJcZDlZaj
- Wortsman et al. 2023 "Stable low-precision" https://openreview.net/forum?id=rp6AicF4UN
- Gao et al. 2019 "Representation degeneration" https://openreview.net/forum?id=HJlNzA5Ym

IEEE 754 & 实现：
- IEEE 754-2019 standard https://ieeexplore.ieee.org/document/8766229
- PyTorch CE 实现 https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

如果你接下来想深入，我建议先重读 Prieto et al. 2025 的 SC 部分，再看这篇 paper 的 Appendix C 三个证明——它们结合起来构成了一个完整的"数值精度 → 几何 → 动力学"的故事，是我读过最 satisfying 的 mechanistic optimization 工作之一。
