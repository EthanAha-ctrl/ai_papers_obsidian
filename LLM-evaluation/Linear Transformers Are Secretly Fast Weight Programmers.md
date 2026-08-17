---
source_pdf: Linear Transformers Are Secretly Fast Weight Programmers.pdf
paper_sha256: 6891c3b7f048f436cebb6f4aaee0343b3e3b2b91793f3412819383ccea5c962b
processed_at: '2026-08-05T14:56:53-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

**大家这几年疯狂研究的 linear Transformer，其实是 Schmidhuber 1991 年就搞出来的 Fast Weight Programmer，换了个马甲而已**。而且用 FWP 的老视角一看，linear Transformer 有个致命缺陷——memory 只会往里塞东西，不会删不会改。这篇 paper 就把这个 bug 修了。

---

## 故事从头讲

### Attention 到底在干嘛

标准 Transformer 的 self-attention，你仔细看公式：

$$\mathbf{y}^{(i)} = \mathbf{V}^{(i)} \text{softmax}((\mathbf{K}^{(i)})^\top \mathbf{q}^{(i)})$$

表面上是 query 跟所有 key 算相似度，加权求 value。但你要是**把 softmax 扔掉**，代数变形一下：

$$\mathbf{y}^{(i)} = \big(\sum_{j=1}^{i} \mathbf{v}^{(j)} \otimes \mathbf{k}^{(j)}\big) \mathbf{q}^{(i)}$$

看出来了吗？中间那个 $\sum_j \mathbf{v}^{(j)} \otimes \mathbf{k}^{(j)}$ 就是一个**矩阵在慢慢累积**。每来一个 token，就往这个矩阵里塞一个 outer product $\mathbf{v} \otimes \mathbf{k}$。然后 query 来了，就拿这个矩阵乘一下 query，把东西捞出来。

**这就是一个 key-value associative memory**。Write 用 outer product 累加，read 用 matrix-vector multiply。Softmax 只是个 normalisation 的装饰品。

### Schmidhuber 1991 年就干了这个

1991 年，Schmidhuber 提出 Fast Weight Programmer（FWP）：一个 slow net 学习如何 program 另一个 net 的 fast weights。Update rule 就是：

$$\mathbf{W}^{(i)} = \sigma(\mathbf{W}^{(i-1)} + \mathbf{a}^{(i)} \otimes \mathbf{b}^{(i)})$$

- $\mathbf{a}^{(i)}, \mathbf{b}^{(i)}$：slow net 生成的两个向量（今天叫 key 和 value）
- $\mathbf{W}^{(i)}$：fast weight matrix，每步动态更新
- $\otimes$：outer product，把两个向量拼成一个 matrix

跟上面去掉 softmax 的 attention **一模一样**。Schmidhuber 1993 年的 paper 里甚至用了 "internal spotlights of attention" 这个词——attention 这个概念他早就想到了。

参考：https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html

---

## Linear Transformer 的问题在哪

### 容量有上限

Linear Transformer 把 memory 压成一个固定大小的 matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{value}} \times d_{\text{dot}}}$。问题来了：**一个固定大小的矩阵能存多少个 key-value pair？**

答案：**最多 $d_{\text{dot}}$ 个**。

为什么？因为检索是 $\mathbf{W}\mathbf{q}$，本质是 query 跟存在矩阵里的 keys 做内积。要让不同 key 互不干扰，keys 必须正交。$d_{\text{dot}}$ 维空间里最多 $d_{\text{dot}}$ 个正交向量。

所以序列长度 $L > d_{\text{dot}}$ 时，模型就**爆容量**了，老的记忆被新的冲掉，retrieval 出错。

### 只会加，不会改

更糟糕的是 update rule：

$$\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \mathbf{v}^{(i)} \otimes \phi(\mathbf{k}^{(i)})$$

纯加法。只会往里塞，**永远不会删、不会修正**。

设想一个场景：同一个 key 先关联 value A，后来要改成 value B。纯加法的结果是 memory 里同时有 A 和 B，query 这个 key 时返回 A+B 的混合物。**无法更新**。

标准 Transformer 没这个问题，因为它把所有 KV pairs 显式 concat 起来存（$O(L)$ memory），查的时候 softmax 会自动 attend 到最新的。但代价是 $O(L^2)$ 复杂度。

---

## Delta Rule：让 memory 学会"改"

### 核心思路

老 delta rule（Widrow-Hoff 1960）的思想：**写入前先读出来看看，用误差来修正**。

论文提出的 update rule：

$$\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \beta^{(i)}(\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}) \otimes \phi(\mathbf{k}^{(i)})$$

拆开看：

1. **先 retrieve**：$\bar{\mathbf{v}}^{(i)} = \mathbf{W}^{(i-1)}\phi(\mathbf{k}^{(i)})$，看看当前 memory 里这个 key 对应什么 value
2. **算误差**：$\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}$，新 value 减老 value，就是 prediction error
3. **用 dynamic learning rate 缩放**：$\beta^{(i)} = \sigma(\mathbf{W}_\beta \mathbf{x}^{(i)}) \in [0,1]$，slow net 学习"这次写多猛"
4. **用误差更新**：$\beta^{(i)}(\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}) \otimes \phi(\mathbf{k}^{(i)})$

变量含义：
- $\mathbf{v}^{(i)}$：要写入的新 value
- $\bar{\mathbf{v}}^{(i)}$：memory 里已有的老 value（retrieve 出来的）
- $\beta^{(i)}$：write strength / dynamic learning rate
- $\phi(\mathbf{k}^{(i)})$：key 的非线性投影

这就是 **delta rule with learned learning rate**。网络通过 backprop 学会什么时候大写（覆盖老值）、什么时候小写（保留老值）。

### 为什么比 gated rule 好

Peng et al. (2021) 并发提了个 gated update：

$$\mathbf{W}^{(i)} = (1-\beta^{(i)})\mathbf{W}^{(i-1)} + \beta^{(i)} \mathbf{v}^{(i)} \otimes \phi(\mathbf{k}^{(i)})$$

看起来差不多，但有个致命区别。假设 memory 里有两个正交 associations $\mathbf{v}_1 \otimes \mathbf{k}_1 + \mathbf{v}_2 \otimes \mathbf{k}_2$，现在用 $\mathbf{k}_2$ 写入新 value $\mathbf{v}_3$：

**Gated rule**：
- $\mathbf{k}_2$ 对应的 value 变成 $(1-\beta)\mathbf{v}_2 + \beta\mathbf{v}_3$ ✓
- $\mathbf{k}_1$ 对应的 value 变成 $(1-\beta)\mathbf{v}_1$ ✗ **被无辜衰减了！**

**Delta rule**：
- $\mathbf{k}_2$ 对应的 value 变成 $(1-\beta)\mathbf{v}_2 + \beta\mathbf{v}_3$ ✓
- $\mathbf{k}_1$ 对应的 value 还是 $\mathbf{v}_1$ ✓ **毫发无伤**

原因：delta rule 的 remove 项 $\bar{\mathbf{v}} \otimes \phi(\mathbf{k})$ 只移除跟当前 key 相关的部分（因为 $\bar{\mathbf{v}} = \mathbf{W}\mathbf{k}$ 投影到了 key 方向）。Gated rule 对整个 $\mathbf{W}$ 做 $(1-\beta)$ 衰减，**伤及无辜**。

参考：https://arxiv.org/abs/2103.02144

---

## DPFP：一个简单但聪明的 $\phi$ 函数

### 为什么要 $\phi$

Linear Transformer 用 $\phi(\mathbf{k})^\top \phi(\mathbf{q})$ 替代 $\exp(\mathbf{k}^\top \mathbf{q})$ 来 linearise softmax。$\phi$ 的选择决定 $d_{\text{dot}}$，也就是 memory 容量。

现有方案的毛病：
- **ELU+1**（Katharopoulos）：$\phi(x) = \text{ELU}(x)+1$，简单但不扩维，$d_{\text{dot}} = d_{\text{key}}$，容量固定
- **FAVOR+**（Performer）：用随机特征近似 softmax，$d_{\text{dot}} = 2m$，但随机采样引入 variance，永远收敛不到 0 loss

### DPFP 的设计

核心 idea：**把输入空间切成不重叠的区域，让不同区域的向量在投影后正交**。

2D 例子：把 $\mathbb{R}^2$ 投到 $\mathbb{R}^4$，用四个 partial functions：

$$\phi_1 = r(k_1)r(k_2), \quad \phi_2 = r(-k_1)r(k_2), \quad \phi_3 = r(k_1)r(-k_2), \quad \phi_4 = r(-k_1)r(-k_2)$$

- $r(a) = \max(0, a)$
- $k_1, k_2$：输入向量的两个分量

每个象限的向量只激活 4D 空间中的一个维度，不同象限自然正交。

高维推广：

$$\phi_{i\nu}(\mathbf{k}) = r\big([\frac{\mathbf{k}}{-\mathbf{k}}]\big)_i \cdot r\big([\frac{\mathbf{k}}{-\mathbf{k}}]\big)_{i+\nu}$$

- $[\frac{\mathbf{k}}{-\mathbf{k}}] \in \mathbb{R}^{2d_{\text{key}}}$：$\mathbf{k}$ 和 $-\mathbf{k}$ concat
- $\nu$：超参数，控制容量，$d_{\text{dot}} = 2d_{\text{key}}\nu$
- 确定性、无参数、高度并行

PyTorch 实现就三行：

```python
def dpfp(x, nu=1):
    x = cat([r(x), r(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) for j in range(1,nu+1)], dim=-1)
    x_repeat = cat([x] * nu, dim=-1)
    return x_repeat * x_rolled
```

参考：https://arxiv.org/abs/2102.11174

---

## 实验数据说话

### 合成实验：验证容量限制

$d_{\text{key}} = 64$，变序列长度 $S$，看什么时候 retrieval 开始出错：

| Model | $d_{\text{dot}}$ | 容量极限 |
|-------|------------------|----------|
| Linear-Attention | 64 | ~60 |
| DPFP ν=1 | 128 | ~128 |
| DPFP ν=2 | 256 | ~256 |
| DPFP ν=3 | 384 | ~384 |
| FAVOR+ m=64 | 128 | 永远到不了 0 |
| Softmax | ∞ | >500 才吃力 |

**转折点精确落在 $S \approx d_{\text{dot}}$**，理论完美验证。

### WikiText-103：Delta rule 的威力

Small config：$D=128$，$L=256$，8 heads，per-head $d_{\text{dot}}=16$（严重 overcapacity）。

| Model | Update Rule | Valid PPL |
|-------|-------------|-----------|
| Standard Transformer | - | 33.0 |
| Linear Transformer | sum | 37.1 |
| **Delta Network** | **delta** | **34.1** |
| Performer | sum | 39.0 |
| Performer | **delta** | **36.1** |

Delta rule 在两个 backbone 上都提升 ~3 PPL。

### 不截断 context：最震撼的结果

训练时让 fast weight memory 跨 segment carry，理论上可以**无限运行**：

| Model | State Size (M) | Valid PPL |
|-------|----------------|-----------|
| Linear Transformer (sum) | 0.13 | >260 **崩了** |
| **Delta Network** | **0.13** | **27.8** |
| Transformer-XL | 0.13 | 65.7 |
| Transformer-XL | 6.29 | 24.6 |

**同 state size 下 Delta Network 碾压 Transformer-XL**（27.8 vs 65.7）。Transformer-XL 要用 48 倍大的 state 才能追上。

纯加法的 Linear Transformer 在无限 context 下直接崩溃（PPL > 260），因为 memory 被历史噪声淹没。Delta rule 能 selective update，保持 memory 信噪比。

---

## 为什么这篇 paper 重要

### 历史价值

它揭示了一个尴尬事实：ML community 2017-2020 年疯狂研究 linear Transformer，结果发现 Schmidhuber 1991 年就做完了核心数学。只是 community 喜欢重新命名：Fast Weight Programmer → Linear Attention → Performer → Random Feature Attention，全是一回事。

### 技术价值

**Delta rule 是后来一系列工作的原型**：

- **RetNet (2023)**：用 exponential decay 替代 delta gating
- **Mamba (2023)**：SSM + selective gating，本质是 input-dependent memory update  
- **RWKV**：linear attention RNN 化，gating 机制类似 Peng 的 gated rule
- **Gated DeltaNet (2024)**：直接继承本文 delta rule + gating

公式 $\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \beta^{(i)}(\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}) \otimes \phi(\mathbf{k}^{(i)})$ 是这些后续工作的祖宗。

参考：
- RetNet：https://arxiv.org/abs/2307.08621
- Mamba：https://arxiv.org/abs/2312.00752
- Gated DeltaNet：https://arxiv.org/abs/2412.06464

### Intuition 总结

1. **Attention 本质是 associative memory**：softmax 只是个 retrieval kernel，linear family 换了别的 kernel
2. **固定大小 memory 有容量上限**：$d_{\text{dot}}$ 决定容量，不是参数量
3. **纯加法 update 是蠢 memory**：只会堆，不会改。Delta rule 让 memory 变聪明
4. **$\phi$ 函数决定容量**：ELU+1 不扩容，FAVOR+ 随机扩容有 variance，DPFP 确定性扩容促正交
5. **Position encoding 在 linear attention 里有害**：memory 本身就是 temporal，不需要额外 position 信号
6. **FWP 视角统一 Transformer 和 RNN**：Linear Transformer = RNN with matrix state，Delta rule = RNN with learnable update gate

这篇 paper 的美在于：**用一个 30 年前的老视角，让看似复杂的 linear Transformer 变透明，然后基于这个视角提出实质改进**。这是重新发现旧理论价值的典范。

---

# Linear Transformers Are Secretly Fast Weight Programmers 深度解析

## 1. 核心论点与历史定位

这篇 paper 来自 Schmidhuber 组（Schlag, Irie, Schmidhuber, 2021 ICML），核心 thesis 极其优雅：**当下流行的 linear Transformer 家族（Linear Transformer, Performer, Random Feature Attention 等）在数学形式上完全等价于 Schmidhuber 1991 年提出的 Fast Weight Programmers (FWPs)**，只不过 community 用了不同的词汇重新发明了一遍。

这个观点的价值不在于"谁先发明"的优先权争论（虽然 Schmidhuber 确实很在意），而在于 **FWP 的视角能让我们看清楚 linear Transformer 的本质缺陷，并提出有针对性的改进**。

参考链接：
- 原论文：https://arxiv.org/abs/2102.11174
- Schmidhuber 1991 FWP blog：https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html
- Schmidhuber 1992 Neural Computation paper：https://www.mitpressjournals.org/doi/10.1162/neco.1992.4.1.131

---

## 2. Fast Weight Programmers 历史回顾

### 2.1 概念起源

传统 neural network 中，weights 训练完就固定了，只有 activations 随输入变化。Fast weights 的思想是让 weights 本身也变成 input-dependent，这个概念可以追溯到：

- von der Malsburg (1981) 的 **synaptic modulation**：effective weight = slow weight × fast weight
- Hinton & Plaut (1987) 的 **dual learning rate**：两套 weights 用不同 learning rate
- Feldman (1982) 的 **dynamic connections**

关键问题：1991 年之前，没有任何网络 **通过 gradient descent 学习如何快速修改另一个网络的 fast weights**。Schmidhuber (1991, 1992, 1993) 填补了这个空白。

### 2.2 经典 FWP 形式

Schmidhuber 的 FWP 是一个两网络系统：slow net 学习 program fast net 的 fast weights。其核心 update rule 使用 **outer product**：

$$\mathbf{a}^{(i)}, \mathbf{b}^{(i)} = \mathbf{W}_a \mathbf{x}^{(i)}, \mathbf{W}_b \mathbf{x}^{(i)} \quad (1)$$

$$\mathbf{W}^{(i)} = \sigma\big(\mathbf{W}^{(i-1)} + \mathbf{a}^{(i)} \otimes \mathbf{b}^{(i)}\big) \quad (2)$$

$$\mathbf{y}^{(i)} = \mathbf{W}^{(i)} \mathbf{x}^{(i)} \quad (3)$$

变量含义：
- $\mathbf{x}^{(i)} \in \mathbb{R}^{d_{\text{in}}}$：第 $i$ 步的输入向量，$d_{\text{in}}$ 是输入维度
- $\mathbf{a}^{(i)}, \mathbf{b}^{(i)}$：slow net 生成的两个向量（对应今天说的 key 和 value 的前身）
- $\mathbf{W}_a, \mathbf{W}_b$：slow weights（trainable，通过 backprop 学习）
- $\mathbf{W}^{(i)}$：fast weight matrix，每步动态生成，充当 short-term memory
- $\otimes$：outer product，$\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\top$，结果是矩阵
- $\sigma$：activation function（可以是 identity）

**Intuition**：$\mathbf{W}^{(i)}$ 是一个 associative memory，每步通过 outer product 写入一个 key-value association（Eq. 2 是 write，Eq. 3 是 read）。这就是后来 attention 的雏形，Schmidhuber 1993 甚至讨论了 "internal spotlights of attention"。

参考链接：
- Schmidhuber 1991 技术报告：https://people.idsia.ch/~juergen/FKI-147-91.pdf
- Ba et al. 2016 "Using fast weights to attend to the recent past"：https://arxiv.org/abs/1610.06258

---

## 3. Linear Transformer = FWP 的形式等价证明

### 3.1 标准 Self-Attention 回顾

autoregressive Transformer 的 self-attention：

$$\mathbf{k}^{(i)}, \mathbf{v}^{(i)}, \mathbf{q}^{(i)} = \mathbf{W}_k \mathbf{x}^{(i)}, \mathbf{W}_v \mathbf{x}^{(i)}, \mathbf{W}_q \mathbf{x}^{(i)} \quad (4)$$

$$\mathbf{K}^{(i)} = [\mathbf{K}^{(i-1)}, \mathbf{k}^{(i)}] \in \mathbb{R}^{d_{\text{key}} \times i} \quad (5)$$

$$\mathbf{V}^{(i)} = [\mathbf{V}^{(i-1)}, \mathbf{v}^{(i)}] \in \mathbb{R}^{d_{\text{value}} \times i} \quad (6)$$

$$\mathbf{y}^{(i)} = \mathbf{V}^{(i)} \text{softmax}\big((\mathbf{K}^{(i)})^\top \mathbf{q}^{(i)}\big) \quad (7)$$

变量含义：
- $\mathbf{x}^{(i)} \in \mathbb{R}^{d \times 1}$：第 $i$ 个 token 的输入
- $\mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_q$：key/value/query 的 projection matrices
- $\mathbf{K}^{(i)}, \mathbf{V}^{(i)}$：到第 $i$ 步累积的 key/value 矩阵，沿时间维度 concat
- $[\mathbf{A}, \mathbf{a}]$：matrix-vector 沿时间维度拼接
- $d_{\text{key}}, d_{\text{value}}$：key 和 value 的维度

### 3.2 去掉 softmax 揭示本质

如果把 Eq. 7 的 softmax 去掉：

$$\mathbf{y}^{(i)} = \mathbf{V}^{(i)} \big((\mathbf{K}^{(i)})^\top \mathbf{q}^{(i)}\big) = \big(\mathbf{V}^{(i)} (\mathbf{K}^{(i)})^\top\big) \mathbf{q}^{(i)} = \big(\sum_{j=1}^{i} \mathbf{v}^{(j)} \otimes \mathbf{k}^{(j)}\big) \mathbf{q}^{(i)} \quad (8)$$

关键步骤：$\mathbf{V}^{(i)}(\mathbf{K}^{(i)})^\top$ 就是所有 $\mathbf{v}^{(j)} \otimes \mathbf{k}^{(j)}$ 的求和！

定义 fast weight matrix：

$$\mathbf{W}^{(i)} = \sum_{j=1}^{i} \mathbf{v}^{(j)} \otimes \mathbf{k}^{(j)} \quad (9)$$

则 self-attention 退化为：

$$\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \mathbf{v}^{(i)} \otimes \mathbf{k}^{(i)} \quad (10)$$

$$\mathbf{y}^{(i)} = \mathbf{W}^{(i)} \mathbf{q}^{(i)} \quad (11)$$

**这就是 FWP 的 Eq. 2-3，σ = identity，query 投影 $\mathbf{W}_q$ 保留**。

Intuition：标准 Transformer 的 attention 内部其实在做"用 outer product 累积一个动态矩阵，再用 query 检索"。Softmax 只是给这个矩阵-向量乘法加了一个 normalisation 而已。

### 3.3 Linearised Self-Attention 仍是 FWP（带 normalisation）

Katharopoulos et al. (2020) 等人 linearise softmax 的思路：把 softmax kernel $\kappa(\mathbf{k}, \mathbf{q}) = \exp(\mathbf{k} \cdot \mathbf{q})$ 替换为：

$$\kappa'(\mathbf{k}, \mathbf{q}) = \phi(\mathbf{k})^\top \phi(\mathbf{q})$$

其中 $\phi: \mathbb{R}^{d_{\text{key}}} \to \mathbb{R}^{d_{\text{dot}}}$ 是某个非线性映射。代入后得到：

$$\mathbf{y}^{(i)} = \frac{\sum_{j=1}^{i} \big(\mathbf{v}^{(j)} \otimes \phi(\mathbf{k}^{(j)})\big) \phi(\mathbf{q}^{(i)})}{\big(\sum_{j'=1}^{i} \phi(\mathbf{k}^{(j')})\big) \cdot \phi(\mathbf{q}^{(i)})} \quad (14)$$

引入 fast weight matrix 和 accumulator：

$$\mathbf{W}^{(i)} = \sum_{j=1}^{i} \mathbf{v}^{(j)} \otimes \phi(\mathbf{k}^{(j)}) \quad (15)$$

$$\mathbf{z}^{(i)} = \sum_{j=1}^{i} \phi(\mathbf{k}^{(j)}) \quad (16)$$

递推形式：

$$\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \mathbf{v}^{(i)} \otimes \phi(\mathbf{k}^{(i)}) \quad (17)$$

$$\mathbf{z}^{(i)} = \mathbf{z}^{(i-1)} + \phi(\mathbf{k}^{(i)}) \quad (18)$$

$$\mathbf{y}^{(i)} = \frac{1}{\mathbf{z}^{(i)} \cdot \phi(\mathbf{q}^{(i)})} \mathbf{W}^{(i)} \phi(\mathbf{q}^{(i)}) \quad (19)$$

**结论**：Linear Transformer = FWP + normalisation（denominator $\mathbf{z}^{(i)} \cdot \phi(\mathbf{q}^{(i)})$）。

这个等价性让我们可以借用 FWP 30 年的理论积累来分析 linear Transformer。

参考链接：
- Katharopoulos et al. 2020 "Transformers are RNNs"：https://arxiv.org/abs/2006.16236
- Choromanski et al. 2021 Performer：https://arxiv.org/abs/2009.14794
- Peng et al. 2021 Random Feature Attention：https://arxiv.org/abs/2103.02144

---

## 4. 容量限制分析（论文最重要的理论贡献）

### 4.1 直觉推导

Linear attention 把所有 key-value associations 累加到一个固定大小的矩阵 $\mathbf{W}^{(i)} \in \mathbb{R}^{d_{\text{value}} \times d_{\text{dot}}}$ 里。检索时用 $\mathbf{W}^{(i)} \phi(\mathbf{q}^{(i)})$，本质是 matrix-vector 乘法。

要让不同 associations 互不干扰，**对应的 keys 必须正交**。否则 query 与多个 key 都有非零内积，会返回多个 values 的线性组合（crosstalk）。

在 $d_{\text{dot}}$ 维空间里，最多有 $d_{\text{dot}}$ 个正交向量。所以 **linear attention 的容量上界就是 $d_{\text{dot}}$**。当序列长度 $L > d_{\text{dot}}$，模型进入 **overcapacity regime**。

### 4.2 Tensor Product Representation 理论支撑

这个容量限制可以用 Smolensky (1990) 的 **Tensor Product Representation (TPR)** 理论形式化。TPR 把符号结构编码为 role 和 filler 向量的 outer product：

$$\text{TPR} = \sum_i \text{filler}_i \otimes \text{role}_i$$

直接对应 FWP 里的 $\sum_j \mathbf{v}^{(j)} \otimes \phi(\mathbf{k}^{(j)})$，其中 role = key，filler = value。Smolensky 的 Theorem 3.1 和 3.3 形式化了 crosstalk 和 retrieval error。

**重要区别**：经典 TPR 的 role/filler 向量是预先设计的（用 a priori 符号结构知识）。FWP 的 role/filler 是 **learned** 的——slow net 通过 gradient descent 自己发明 key 和 value 的表示。

Intuition：这就像 classic Hopfield network 用预定义的模式，而 FWP 让网络自己学习要存储什么模式。

参考链接：
- Smolensky 1990 TPR：https://www.sciencedirect.com/science/article/pii/0004370290900077
- Schlag & Schmidhuber 2018 "Learning to reason with third order tensor products"：https://papers.nips.cc/paper/2018/hash/e8e29619253e0c9a67e3f58a2be7c64a-Abstract.html

### 4.3 实验验证容量限制（Setting 1）

合成 retrieval 实验：给定 $L$ 个 key-value pairs，最后用一个 query 检索对应 value。$d_{\text{key}} = 64$，不同 $\phi$ 函数产生不同 $d_{\text{dot}}$。

实验结果（Figure 2）：
| Model | $d_{\text{dot}}$ | 容量极限 |
|-------|------------------|----------|
| Linear-Attention | 64 | ~60 associations 开始出错 |
| DPFP-ν=1 | 128 | ~128 |
| DPFP-ν=2 | 256 | ~256 |
| DPFP-ν=3 | 384 | ~384 |
| FAVOR+ (m=64) | 128 | 永远到不了 0 loss |
| Softmax | $\infty$ | >500 才开始吃力 |

实验完美验证理论：误差出现的转折点正好在 $L \approx d_{\text{dot}}$ 处。

**关键 insight**：容量不是模型参数量决定的，而是 $d_{\text{dot}}$（dot product 空间维度）决定的。FAVOR+ 虽然理论上 $d_{\text{dot}} = 2m$，但随机 features 近似 softmax 会引入 variance，永远无法精确 retrieval。

---

## 5. Delta Rule 更新规则（论文最重要的算法贡献）

### 5.1 动机

在 overcapacity regime 下，理想 memory 应该能 **dynamically 交互 memory 内容，选择性 decide 哪些 associations 保留、哪些遗忘**。但 Eq. 17 的纯加法 update rule 做不到——它只会无脑累加，老的 association 永远不会被删除或修正。

标准 Transformer 通过 concat 存储所有 KV pairs（容量随序列长度线性增长）规避了这个问题，但代价是 $O(L^2)$ 复杂度。

### 5.2 Delta Rule 推导

灵感来自 Schlag et al. (2021) 和经典的 Widrow-Hoff delta rule (1960)。核心思想：**在写入新 value 前，先 retrieve 老的 value，然后用两者的差作为 update 信号**。

给定新输入 $(\mathbf{k}^{(i)}, \mathbf{v}^{(i)})$：

**Step 1: Retrieve 当前 memory 中与 $\mathbf{k}^{(i)}$ 关联的 value**

$$\bar{\mathbf{v}}^{(i)} = \mathbf{W}^{(i-1)} \phi(\mathbf{k}^{(i)}) \quad (20)$$

**Step 2: 计算写入强度（write-strength）**

$$\beta^{(i)} = \sigma(\mathbf{W}_\beta \mathbf{x}^{(i)}) \quad (21)$$

- $\mathbf{W}_\beta \in \mathbb{R}^{1 \times d}$：可训练参数
- $\sigma$：sigmoid，保证 $\beta^{(i)} \in [0, 1]$
- $\beta^{(i)}$ 相当于 dynamic learning rate

**Step 3: 用 interpolation 生成 new value**

$$\mathbf{v}_{\text{new}}^{(i)} = \beta^{(i)} \mathbf{v}^{(i)} + (1 - \beta^{(i)}) \bar{\mathbf{v}}^{(i)} \quad (22)$$

**Step 4: Update memory（write + remove）**

$$\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \underbrace{\mathbf{v}_{\text{new}}^{(i)} \otimes \phi(\mathbf{k}^{(i)})}_{\text{write}} - \underbrace{\bar{\mathbf{v}}^{(i)} \otimes \phi(\mathbf{k}^{(i)})}_{\text{remove}} \quad (23)$$

把 Eq. 22 代入 Eq. 23：

$$\mathbf{v}_{\text{new}}^{(i)} - \bar{\mathbf{v}}^{(i)} = \beta^{(i)}(\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)})$$

所以：

$$\boxed{\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \beta^{(i)} (\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}) \otimes \phi(\mathbf{k}^{(i)})} \quad (24)$$

这就是 **delta rule with dynamic learning rate**！

变量含义：
- $\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}$：prediction error（target - prediction）
- $\beta^{(i)}$：dynamic learning rate，由 slow net 学习
- $\phi(\mathbf{k}^{(i)})$：input pattern（key 的非线性投影）

**Intuition**：这完全是 Widrow-Hoff 的 $\Delta \mathbf{W} = \eta (\mathbf{t} - \mathbf{W}\mathbf{x})\mathbf{x}^\top$ 的形式，只不过 $\eta$ 变成 input-dependent 的 $\beta^{(i)}$。网络通过 gradient descent 学习"何时用大 learning rate 写入、何时用小 learning rate 保留"。

### 5.3 与 Peng et al. (2021) Gated Update 的对比

Peng et al. 并发提出 gated update rule（受 LSTM gating 启发）：

$$\mathbf{W}^{(i)} = (1 - \beta^{(i)}) \mathbf{W}^{(i-1)} + \beta^{(i)} \mathbf{v}^{(i)} \otimes \phi(\mathbf{k}^{(i)}) \quad (52)$$

论文 Appendix B 给出了一个漂亮的对比。假设 memory 中已有两个正交 associations：

$$\mathbf{W} = \mathbf{v}_1 \otimes \mathbf{k}_1 + \mathbf{v}_2 \otimes \mathbf{k}_2$$

现在用 $\mathbf{k}_3 = \mathbf{k}_2$（重复 key）写入新 value $\mathbf{v}_3$。

**Peng 的 gated rule**：
- $\mathbf{W}' \mathbf{k}_3 = (1-\beta)\mathbf{v}_2 + \beta \mathbf{v}_3$ ✓（正确更新）
- $\mathbf{W}' \mathbf{k}_1 = (1-\beta)\mathbf{v}_1$ ✗（**无关 association 被衰减了！**）

**Delta rule**：
- $\mathbf{W}' \mathbf{k}_3 = (1-\beta)\mathbf{v}_2 + \beta \mathbf{v}_3$ ✓（正确更新）
- $\mathbf{W}' \mathbf{k}_1 = \mathbf{v}_1$ ✓（**无关 association 完好无损**）

原因：delta rule 的 remove 项 $\bar{\mathbf{v}} \otimes \phi(\mathbf{k})$ 只移除与当前 key 相关的部分，因为 $\bar{\mathbf{v}} = \mathbf{W}\mathbf{k}$ 已经投影到 key 方向。而 gated rule 对整个 $\mathbf{W}$ 做 $(1-\beta)$ 衰减，伤及无辜。

### 5.4 Normalisation 策略

两种 normalisation 方案：

**Attention normalisation**（沿用 linear Transformer 的 denominator）：

$$\mathbf{z}^{(i)} = \mathbf{z}^{(i-1)} + \phi(\mathbf{k}^{(i)}) \quad (26)$$

$$\bar{\mathbf{v}}^{(i)} = \frac{\mathbf{W}^{(i-1)} \phi(\mathbf{k}^{(i)})}{\mathbf{z}^{(i-1)} \cdot \phi(\mathbf{k}^{(i)})} \quad (27)$$

$$\mathbf{y}^{(i)} = \frac{\mathbf{W}^{(i)} \phi(\mathbf{q}^{(i)})}{\mathbf{z}^{(i)} \cdot \phi(\mathbf{q}^{(i)})} \quad (28)$$

**问题**：$\mathbf{z}^{(i)}$ 单调递增（全是正项累加），长期不稳定。

**Sum normalisation**（论文提出）：对 $\phi(\mathbf{k})$ 和 $\phi(\mathbf{q})$ 做 L1 normalisation：

$$\phi'(\mathbf{q}^{(i)}) = \frac{\phi(\mathbf{q}^{(i)})}{\sum_{j=1}^{d_{\text{dot}}} \phi(\mathbf{q}^{(i)})_j} \quad (29)$$

**Intuition**：matrix-vector 乘法 $\mathbf{W}\mathbf{q}$ 可以看作 $\mathbf{W}$ 的列的 weighted sum，weights 是 $\mathbf{q}$ 的分量。如果 $\mathbf{q}$ 的分量和为 1，这就是对 $\mathbf{W}$ 列的 attention。

Appendix A.2 的推导证明：sum normalisation 让 write 和 remove 操作的权重平衡（positive 项权重 = negative 项总权重），这对于 delta rule 的稳定性至关重要。

参考链接：
- Widrow & Hoff 1960 "Adaptive switching circuits"：https://www-isl.stanford.edu/~widrow/papers/t1960adaptive.pdf
- Schlag et al. 2021 "Learning associative inference using fast weight memory"：https://arxiv.org/abs/2011.07831

---

## 6. DPFP：Deterministic Parameter-Free Projection

### 6.1 现有 $\phi$ 函数的问题

| 方法 | $\phi$ | $d_{\text{dot}}$ | 优点 | 缺点 |
|------|--------|-------------------|------|------|
| Katharopoulos | $\text{ELU}(x)+1$ | $d_{\text{key}}$ | 简单 | 不扩容，容量 = $d_{\text{key}}$ |
| FAVOR+ | $\frac{h(\mathbf{x})}{\sqrt{m}}[\exp(\mathbf{R}\mathbf{x})]$ | $2m$ | 理论严格近似 softmax | 随机采样引入 variance |

FAVOR+ 的 $\phi$：

$$h(\mathbf{x}) = \frac{1}{\sqrt{2}} \exp\big(-\frac{\|\mathbf{x}\|^2}{2}\big) \quad (31)$$

$$\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{m}} [\exp(\mathbf{R}\mathbf{x})] \quad (32)$$

- $\mathbf{R} \in \mathbb{R}^{m \times d_{\text{key}}}$：随机矩阵，每行 $\mathbf{r} \sim \mathcal{N}(0, \mathbf{I}_{d_{\text{key}}})$
- $[\cdot]$：沿 feature 维度 concat（正负两部分，所以 $d_{\text{dot}} = 2m$）
- 理论上当 $m \to \infty$ 时精确近似 softmax，但实践中 $m$ 有限，引入 variance

### 6.2 DPFP 设计思想

目标：**确定性、无参数、能扩容、促进正交性**。

从 2D 到 4D 的 toy example 出发。对 $\mathbf{k} \in \mathbb{R}^2$，定义 4 个 partial functions：

$$\phi_1(\mathbf{k}) = r(k_1) r(k_2) \quad (33)$$
$$\phi_2(\mathbf{k}) = r(-k_1) r(k_2) \quad (34)$$
$$\phi_3(\mathbf{k}) = r(k_1) r(-k_2) \quad (35)$$
$$\phi_4(\mathbf{k}) = r(-k_1) r(-k_2) \quad (36)$$

- $r(a) = \max(0, a)$：rectifier
- $k_1, k_2$：$\mathbf{k}$ 的两个分量

**关键性质**：2D 平面被四个象限切分，每个象限的向量只激活 4D 空间中的一个分量。不同象限的向量在 4D 空间中**正交**（因为非零分量位置不重叠）。

### 6.3 高维推广

一般形式，对 $\mathbf{k} \in \mathbb{R}^{d_{\text{key}}}$ 和 $i \in [1, 2d_{\text{key}}]$：

$$\phi_{i\nu}(\mathbf{k}) = r\Big(\big[\frac{\mathbf{k}}{-\mathbf{k}}\big]\Big)_i \cdot r\Big(\big[\frac{\mathbf{k}}{-\mathbf{k}}\big]\Big)_{i+\nu} \quad (37)$$

- $\big[\frac{\mathbf{k}}{-\mathbf{k}}\big] \in \mathbb{R}^{2d_{\text{key}}}$：$\mathbf{k}$ 和 $-\mathbf{k}$ 的 concat
- $\nu \in \{1, 2, ..., d_{\text{key}}^2 - 1\}$：capacity 控制超参数
- $d_{\text{dot}} = 2 d_{\text{key}} \nu$

**Intuition**：通过把 $\mathbf{k}$ 和 $-\mathbf{k}$ concat 得到 $2d_{\text{key}}$ 维非负向量，然后取所有相隔 $\nu$ 的两两乘积。这创造了大量"特征槽位"，不同输入倾向于激活不同的槽位，促进正交性。

PyTorch 实现（Listing 1）极简：

```python
def dpfp(x, nu=1):
    x = cat([r(x), r(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) 
                    for j in range(1, nu+1)], dim=-1)
    x_repeat = cat([x] * nu, dim=-1)
    return x_repeat * x_rolled
```

两个 concat + 一个 element-wise multiply，高度并行化。

---

## 7. 实验结果详解

### 7.1 合成实验 Setting 2：Update Rule 对比

设定：$L = 40$，20 unique keys/values，**有放回采样**（同一 key 可能被多次赋新 value，需要 update）。

对比的 update rules：
1. **sum rule**（Eq. 17）：纯加法，baseline
2. **Schlag (2021)**：原始 FWP，用 tanh，无 $\phi$，无 sum normalisation
3. **Schlag (2021) + DPFP**：加 DPFP 但不加 normalisation
4. **ours**：delta rule + DPFP + sum normalisation

结果（Figure 3）：只有 **ours** 收敛到接近 0 loss。sum rule 完全失败（无法更新已有 association）。这验证了 delta rule 的必要性。

### 7.2 机器翻译（WMT14 En-De）

BLEU scores（Table 1）：

| Model | $d_{\text{dot}}$ | Valid BLEU | Test BLEU |
|-------|-------------------|------------|-----------|
| Standard Transformer | - | 26.6 | 27.7 |
| Linear Transformer | 64 | 25.5 | 26.8 |
| Performer | 64 / 256 / 512 | 24.2 / 24.9 / 26.7 | 24.4 / 25.3 / 27.7 |
| **DPFP (ours)** | 256 / 512 | **26.2 / 26.2** | **26.9 / 27.1** |

关键观察：
- Performer 需要 $d_{\text{dot}} = 512$（$m = 256$）才能匹配 standard Transformer
- DPFP 在 $d_{\text{dot}} = 256$（$\nu = 2$）时就已经接近，且无随机性
- DPFP 在小 $d_{\text{dot}}$ 时明显优于 Performer（simplicity-effectiveness trade-off）

### 7.3 语言建模（WikiText-103）

#### 7.3.1 Overcapacity regime（Table 2）

Small config：$D = 128$，$L = 256$，8 heads，$d_{\text{dot}} = 16$ per head（严重 overcapacity，$L = 256 \gg d_{\text{dot}} = 16$）。

| Model | Update Rule | Valid PPL | Test PPL |
|-------|-------------|-----------|----------|
| Transformer | - | 33.0 | 34.1 |
| Linear Transformer | sum | 37.1 | 38.3 |
| **Delta Network** | **delta** | **34.1** | **35.5** |
| Performer | sum | 39.0 | 39.6 |
| Performer | **delta** | **36.1** | **37.2** |

Delta rule 在两个 backbone 上都带来 ~3 PPL 的提升。在 overcapacity 下效果尤为显著。

#### 7.3.2 Ablation（Table 3）

| Position Encoding | Attn. Normalisation | Valid | Test |
|-------------------|---------------------|-------|------|
| Yes | Yes | 30.4 | 32.1 |
| No | Yes | 29.2 | 31.2 |
| Yes | No | 29.7 | 31.5 |
| **No** | **No** | **28.1** | **31.1** |

发现：
- **Absolute positional encoding 有害**（证实 Irie et al. 2019a 的发现）
- **Attention normalisation 有害**（对 delta rule 而言，因为 accumulator 会 blow up）
- **Sum normalisation 必需**（去掉就发散）

#### 7.3.3 不截断 context（Table 4）—— 最惊人的结果

训练时把 fast weight memory 从一个 segment carry 到下一个（backprop 仍限制在 segment 内）。这是 linear Transformer 的杀手锏：**理论上可以无限运行**。

| Model | Params (M) | State Size (M) | Valid PPL | Test PPL |
|-------|-----------|----------------|-----------|----------|
| Linear Transformer | 89.8 | 0.13 | >260 | >260 |
| **Delta Network** | **89.9** | **0.13** | **27.8** | **29.4** |
| Transformer-XL | 90.9 | 0.13 | 65.7 | 65.5 |
| Transformer-XL | 90.9 | 1.05 | 29.3 | 30.1 |
| Transformer-XL | 90.9 | 2.10 | 26.4 | 27.4 |
| Transformer-XL | 90.9 | 6.29 | 24.6 | 25.5 |

**关键 insight**：
- Linear Transformer（sum rule）在无限 context 下完全崩溃（PPL > 260）
- Delta Network 在相同 state size（0.13M）下 PPL = 27.8
- Transformer-XL 在 state size = 0.13M 时 PPL = 65.7（远差于 Delta Net）
- Transformer-XL 需要 6.29M state（48 倍大）才能达到 PPL 24.6

**Delta Network 在小 state size 下碾压 Transformer-XL**，这对 memory-constrained 应用至关重要。

Intuition：sum rule 把所有历史无脑累加，长序列下 memory 被噪声淹没。Delta rule 能 selective update，保持 memory 的"信噪比"。

---

## 8. Complexity 与工程实现

### 8.1 复杂度

| Method | Time (per step) | Space |
|--------|-----------------|-------|
| Standard Transformer | $O(L \cdot d^2)$（并行 $O(1)$ depth） | $O(L \cdot d)$ |
| Linear Transformer | $O(d^2)$ per step | $O(d^2)$（固定） |
| Delta Network | $O(d^2)$ per step（+少量 gating 计算） | $O(d^2)$（固定） |

Delta rule 引入的额外计算：每步多一次 retrieve（$\mathbf{W}\phi(\mathbf{k})$）和 gating（$\sigma(\mathbf{W}_\beta \mathbf{x})$）。实测 wall clock：

| Model | Words/sec | Memory |
|-------|-----------|--------|
| Linear Transformer | 63K | 14 GB |
| Delta Network | 66K | 13 GB |
| DPFP | 63K | - |
| Performer | 57K | - |
| Standard Transformer (PyTorch) | 33K | 17 GB |

Delta Network 甚至比 Linear Transformer 略快（可能因为更好的 memory 访问模式），Performer 因为 sampling 逻辑最慢。

### 8.2 CUDA Kernel 实现细节

Appendix F 提到一个重要工程问题：**naive autograd 会为每个 time step 存储一份 fast weights**，GPU memory 立刻爆掉。解决方案是 **custom backward pass**：在 backward 时重新计算每个 time step 的 fast weights（recomputation），只存一份。因为 fast weight 计算很廉价，recomputation 的时间开销可接受。

---

## 9. 与后续工作的联系（build your intuition）

### 9.1 与 Modern Hopfield Network 的关系

Ramsauer et al. (2021) 把 standard Transformer 的 attention 解释为 Modern Hopfield Network 的 retrieval。本文的 FWP 视角是互补的：

- **Hopfield 视角**：attention 是从超大容量 memory 中 retrieval
- **FWP 视角**：attention 是对有限容量 memory 的 program（write + read）

两个视角合在一起：standard Transformer 用无限容量 memory（concat 所有 KV），linear Transformer 用有限容量 memory（固定大小矩阵），delta rule 让后者能像前者一样 dynamic。

参考链接：
- Ramsauer et al. 2021 "Hopfield Networks is All You Need"：https://arxiv.org/abs/2008.02217

### 9.2 与 RetNet / Mamba / RWKV 的关系（2023-2024 发展）

这篇 paper 是后来一系列 **linear recurrence / state space model** 工作的思想源头之一：

- **RetNet (Microsoft, 2023)**：用 exponential decay 替代 delta rule 的 gating，类似 gated update
- **Mamba (Albert Gu, 2023)**：SSM + selective gating，本质也是 input-dependent memory update
- **RWKV**：linear attention 的 RNN 化，gating 机制类似 Peng et al. 的 gated rule
- **Gated DeltaNet (2024)**：直接继承本文的 delta rule，加上 gating 改进

本文的 delta rule 公式 $\mathbf{W}^{(i)} = \mathbf{W}^{(i-1)} + \beta^{(i)}(\mathbf{v}^{(i)} - \bar{\mathbf{v}}^{(i)}) \otimes \phi(\mathbf{k}^{(i)})$ 可以看作这些后续工作的"原型"。

参考链接：
- RetNet：https://arxiv.org/abs/2307.08621
- Mamba：https://arxiv.org/abs/2312.00752
- RWKV：https://arxiv.org/abs/2305.13048
- Gated DeltaNet：https://arxiv.org/abs/2412.06464

### 9.3 与 Neural Turing Machine / Differentiable Neural Computer 的关系

Graves et al. (2014, 2016) 的 NTM/DNC 也用 differentiable memory operations（read/write heads），但：
- NTM/DNC 用 **explicit addressing**（content-based + location-based）
- FWP/Delta Net 用 **implicit addressing**（通过 key-query 相似度）
- NTM/DNC 的 memory 是显式矩阵，FWP 的 memory 是权重矩阵

Delta rule 的 remove-write 分离与 NTM 的 erase-write 机制哲学相似。

参考链接：
- NTM：https://arxiv.org/abs/1410.5401
- DNC：https://arxiv.org/abs/1610.04003

---

## 10. 对你（Karpathy）的 intuition building 建议

作为构建 intuition 的角度，这篇 paper 的几个 takeaway：

1. **Attention 的本质是 associative memory**：不管是 softmax 还是 linear，attention 都在做 key-value retrieval。Softmax 只是一个特殊的 retrieval kernel，linear family 是其他 kernel。

2. **容量 vs 复杂度的 trade-off**：Standard Transformer 用 $O(L)$ memory 换无限容量，linear family 用 $O(1)$ memory 换 $d_{\text{dot}}$ 容量。没有 free lunch。

3. **Update rule 决定 memory 的"智能"**：纯加法（sum rule）= 蠢 memory，只会堆叠。Delta rule = 智能 memory，能修正、能遗忘。Gated rule（Peng）介于中间，但会伤及无关 association。

4. **$\phi$ 函数决定容量上限**：ELU+1 不扩容，FAVOR+ 随机扩容但有 variance，DPFP 确定性扩容且促进正交。容量 = $d_{\text{dot}}$，不是参数量。

5. **Position encoding 在 linear attention 里有害**：因为 memory 本身就是 temporal，不需要额外 position 信息。这个发现后来在 Mamba 等工作中也被验证。

6. **FWP 视角统一了 Transformer 和 RNN**：Linear Transformer = RNN with matrix state。Delta rule = RNN with learnable update gate。这把 attention 和 recurrence 统一在一个框架下。

如果你想深入，建议看：
- Schlag & Schmidhuber 2018（third order tensor products，把 FWP 推广到三阶）
- Schlag et al. 2021 ICLR（associative inference with FWP）
- Schmidhuber 的 blog 关于 FWP 历史：https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html

这篇 paper 的美在于：它用一个 30 年前的视角，让看似复杂的 linear Transformer 变得透明，并且基于这个视角提出了实质性改进。这是"重新发现旧理论价值"的典范，也是对 ML community 倾向于重新命名的温和提醒。
