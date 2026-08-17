---
source_pdf: cka.pdf
paper_sha256: 371a3404ea3c4ffa1220beacc6f929cf1d95c2e57907eaf4041dc16210ce5782
processed_at: '2026-08-03T15:40:48-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 CKA paper

## 先说背景：大家为啥用 CKA

你想比较两个 neural network 学到了啥，最直觉的办法是把每层 activation 拿出来看。但 activation 是高维向量（比如 2048 维），n 个样本就变成 $n \times d$ 的矩阵，你怎么比较两个这样的矩阵 $X$ 和 $Y$ "像不像"？

直接对齐 neuron 没用 — 第 i 个 neuron 在两个网络里完全可能做不同的事。所以大家想：**别比 neuron，比样本和样本之间的相似性结构**。

具体来说，算两个 Gram matrix：
- $K = XX^\top$，$K_{ij} = \langle x_i, x_j \rangle$，意思是"样本 i 和样本 j 在网络 A 看来有多像"
- $L = YY^\top$，同理对网络 B

然后看这俩 matrix 对齐得好不好，用 HSIC 算它们的 inner product，再归一化一下，就是 CKA。

$$\text{CKA}(X, Y) = \frac{\text{tr}(KHLH)}{\sqrt{\text{tr}(KHKH) \cdot \text{tr}(HLHL)}}$$

其中 $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ 是 centering matrix（把每列减去均值）。

**直觉**：CKA 高 = "网络 A 觉得哪些样本相似，网络 B 也觉得这些样本相似"。这听起来很合理。Kornblith et al. 2019 提出 linear CKA 后，社区一片叫好，因为它能识别不同 random init 训出来的同架构网络对应层（SVCCA 等老方法做不到）。

从此大家疯狂用 CKA，用它得出一堆结论：
- "wide network 最后几层学到相似东西"（Nguyen 2021 发现的 block structure）
- "ViT 和 CNN 学得不一样"（Raghu 2021）
- "transfer learning 中什么被转移了"（Neyshabur 2020）
- "什么 layer 容易 forget"（Ramasesh 2021）

参考：原始 CKA paper https://arxiv.org/abs/1905.00414

---

## 这篇 paper 说：等一下，CKA 可能是在骗你

作者发现一件挺吓人的事：**你可以把一个已经训练好的 network，偷偷改一改它内部 representation，让 CKA 变成任意你想要的值，但 network 的 output 几乎不变**。

具体来说，三种"魔改"都成功：

1. 让 first layer 和 last layer 的 CKA 从 ≈0.1 变到 ≈0.9（"早期和晚期学得一样"！）
2. 让所有 layer 两两 CKA 都 ≈0.9（"完美 block structure"！）
3. 让所有 layer 两两 CKA 都 ≈0.1（"每层都独立"！）

魔改之后 network 在 CIFAR10 test accuracy 几乎不掉，OOD（CIFAR-10-C）也不掉，linear probe 也不掉。

**这意味着什么**？意味着 Kornblith 2019 以来大家用 CKA 看出来的"结构"，可能根本不是 model 行为的反映，而是 CKA 这个 metric 本身的 artifact。

---

## 关键 trick：subset translation

最核心的理论发现是 CKA 对一类叫 **subset translation** 的变换极度敏感。

什么意思？你有 n 个 representation 点 $\{x_1, ..., x_n\}$ 在高维空间里。现在你挑出其中一小撮（subset $S$，比如就 1 个点），其他点不动，把这小撮点沿某方向 $\vec{v}$ 推一段距离 $c$。

从 functional 角度看：如果下一层 weight 和 $\vec{v}$ 垂直，这推一下对 output 完全没影响。

从 CKA 角度看：**灾难**。

### Theorem 1 公式

作者证了 closed-form：

$$\lim_{c \to \infty} \text{CKA}(X, X_{S,\vec{v},c}) = \underbrace{\frac{\rho}{1-\rho}}_{\Gamma(\rho)} \cdot \frac{||\mathbb{E}_{x \in S}[x]||^2}{\mathbb{E}_{x \in X}[||x||^2]} \cdot \sqrt{\text{PR}(X)}$$

变量含义：
- $c$：你推这小撮点多远（趋近无穷）
- $\rho = |S|/n$：你推的子集占多大比例
- $\Gamma(\rho) = \rho/(1-\rho)$：比例因子
- $\mathbb{E}_{x \in S}[x]$：subset S 的均值向量
- $\mathbb{E}_{x \in X}[||x||^2]$：所有点的平均 squared norm
- $\text{PR}(X) = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$：participation ratio，effective dimensionality
  - $\lambda_i$ 是 covariance matrix $\frac{1}{n}X^\top X$ 的第 i 个 eigenvalue

**关键观察**：当 $\rho \to 0$（只推一点点点），$\Gamma(\rho) \to 0$，整个极限 → 0。

也就是说，**推得越少，CKA 跌得越狠**。这反直觉到家了。

### 为什么推一点点反而 CKA 跌得多？

直觉是这样：Gram matrix $XX^\top$ 是 $n \times n$，每个元素是 $\langle x_i, x_j \rangle$，是 d 维 inner product。高维下，单个 outlier 的 norm 如果大，它涉及的所有 $2n-1$ 个 entries 全部 dominate Frobenius norm。

你推一个小子集出去，这部分点的 norm 飙大，但它们和其余点的 inner product 巨大，把整个 Gram matrix 的"形状"扭曲了。剩下的 $\rho/(1-\rho)$ 这个系数，反映的就是"被推的部分 vs 没推部分"在 centered Gram matrix 中的相对权重。

当 $\rho = 1/n$（只推一个点），$\Gamma \approx 1/n$。n = 10000 时，CKA 极限值被乘 $10^{-4}$。

---

## 三个让人坐不住的推论

### Corollary 3：单点 outlier 就能毁掉 CKA

最极端：$S = \{\hat{x}\}$，单点。

$$\Gamma(1/n) = \frac{1/n}{1-1/n} \approx \frac{1}{n}$$

对 CIFAR（n=50000）：CKA 极限 $\approx 2 \times 10^{-5} \times \text{其他项}$。

也就是说，两组 representation **只差一个点的位置**，CKA 就能从 1 跌到 0.001。

这立刻解释了 Nguyen et al. 2022 的发现：block structure 其实是少数 dominant data points 撑起来的，把这些点去掉 block 就消失。Cor 3 给了这个现象理论解释。

参考：https://arxiv.org/abs/2202.07184

### Corollary 4：linear separability 不变也没用

更狠的：S 和 X\S 线性可分，存在 hyperplane $(w, k)$ 分开。你取 $\vec{v} \perp w$，推 X\S 沿 $\vec{v}$ 走。

由于 $\langle w, x + c\vec{v} \rangle = \langle w, x \rangle + c\underbrace{\langle w, \vec{v}\rangle}_{=0} = \langle w, x \rangle$，**完全没跨越 hyperplane**，linear separability 和 margin 一点没变。

但 CKA 照样跌到 0。

**这意味着**：两个 representation，**用同一个 linear classifier 都能正确分类**，CKA 说它们"完全不像"。

经典 ML 理论告诉我们 large margin 是 generalization 的关键。CKA 直接无视这个关键。

### Corollary 2：子集多大都行

$\rho > 1/2$ 也成立。所以推一小撮或推一大撮，CKA 都能被你打下去。

---

## 实验怎么验证的

### 实验 1：早期层 CKA 高不代表 features 相似

训练三个 CIFAR10 网络：
- **Generalized**：正常训练
- **Memorized**：用随机 label 训练（让它死记）
- **Random init**：根本不训

测它们 layer-wise CKA：早期层都 ≈ 0.9（高）。

但看第一层 conv filter 视觉化：三个网络的 filter 完全不一样。Generalized 的是 Gabor-like edge detector，Memorized 的是噪声，Random 是随机。

Linear probe accuracy：generalized 的早期 features 已经能分类 CIFAR10 到 ~70%，random 的只有 ~30%。

**结论**：高 CKA 既不代表 features 视觉相似，也不代表 features 有用。这戳穿了 Ramasesh 2021 等用 CKA 当 "feature usefulness" 代理的做法。

### 实验 2：人工数据验证 Theorem 1

造 20000 个点在 1000 维空间，分两 cluster（沿第一维可分）。把第二个 cluster 沿随机方向推 c 远。

Linear CKA 和 RBF CKA 都迅速跌到 < 0.2，即使 c 不大就开始跌。

注意 RBF CKA 也没躲过！只是 $\sigma$ 极小时（0.2 × median distance）才免疫 — 但 Kornblith 2019 自己的 Table 2 显示这种参数下 RBF CKA 几乎没信息量。

### 实验 3：真实 CNN 上验证

CIFAR10 上训 9-layer CNN，取 last hidden layer 的 X。用 SVM 找分类 hyperplane（91% accuracy）。然后把一个 class 的点沿 $\perp$ hyperplane 方向推出去。

SVM 对原 representation 和被推的 representation 都能 91% 分类。但 CKA 跌到 0。

单点 outlier 实验：n=10000 中推 1 个点，CKA 也明显下降（虽然需要较大 c）。

### 实验 4：CKA map optimization — 最强证据

这是 paper 最炸的一部分。作者直接构造一个优化目标：

$$\theta^*_{new} = \arg\min_\theta \left( \mathcal{L}_{distill}(f_{\theta^*}(X), f_\theta(X)) + \lambda \mathcal{L}_{map}(M_{f_\theta(X)}, M_{target}) \right)$$

变量：
- $\theta^*$：原始训好的网络参数
- $\theta$：当前优化的参数
- $\mathcal{L}_{distill}$：distillation loss，让新网络 output 接近原网络 output
- $M[i,j] = \text{CKA}(\text{layer}_i, \text{layer}_j)$：当前网络的 CKA map
- $M_{target}$：你想让 CKA map 长成的样子
- $\mathcal{L}_{map} = \sum_{i,j} \ln\cosh(M[i,j] - M_{target}[i,j])$：soft L1 loss
- $\lambda$：动态平衡权重

动态调 $\lambda$（Algorithm 1）：
- $\delta_{acc} = \text{orig\_acc} - \text{val\_acc}$
- 若 $\delta_{acc} > \eta$（掉太多）：$\lambda \times 0.8$（重视 accuracy）
- 否则：$\lambda / 0.8$（强化 CKA 匹配）

训练超参（Appendix A.3）：

| 项 | Original | Optimized |
|---|---|---|
| Epochs | 200 | 30 |
| Initial LR | 0.01 | 0.001 |
| LR schedule | Cosine | Cosine + 500 warmup |
| Distill temperature | — | 0.2 |
| Initial $\lambda$ | — | 500 |
| $\lambda$ scaling $\alpha$ | — | 0.8 |
| Optimizer | AdamW | AdamW |

### 三个魔改场景

**Scenario 1**：让 first 和 last layer 的 CKA 接近 1
- 成功，accuracy 几乎不掉
- 直接反驳 Kornblith 2019 "well-trained 网络浅深 CKA 应该低"

**Scenario 2**：让所有 layer 两两 CKA 接近 1（全 1 block）
- 成功，accuracy 微跌
- 反驳 Nguyen 2021 "block structure 是 overparameterization 的标志"

**Scenario 3**：让所有 layer 两两 CKA 接近 0
- 成功，accuracy 大致保住

### Wider Network 实验

ResNet-34 在 CIFAR10 上训，宽度 1× 到 16×。原 wide network 显示漂亮 block structure（Nguyen 2021 的发现）。

把每个宽网络优化让 CKA map 模仿 thin network 的样子。结果：**所有宽网络都能被改成 thin network 的 CKA map，accuracy 几乎不变**。

跨 dataset 验证（PatchCamelyon，组织病理图像，和 CIFAR10 像素分布完全不同）：优化后的网络在新数据上 CKA map 仍然呈现 thin network 模式。说明 manipulation 不是表面现象。

参考：ResNet https://arxiv.org/abs/1512.03385，PatchCamelyon https://arxiv.org/abs/1806.03962

### ViT 也跑了一遍

8 个 transformer block，12 heads，hidden 256，200 epochs。CKA map 同样能被任意操纵。说明结论架构无关。

参考：ViT https://arxiv.org/abs/2010.11929

### Mechanism：优化器自己发现了 subset translation

可视化 optimized network 的 last hidden layer PCA：**单个 class 的一部分点被整体推离其他类**。

这就是 Theorem 1 描述的 subset translation！没人告诉优化器这么做，它自己摸索出来这是操纵 CKA 最有效的办法。

---

## 直觉总结

### 1. CKA 测的不是 feature similarity，是 Gram matrix alignment

CKA 比的是 $XX^\top$ 和 $YY^\top$ 的对齐。但 functional behavior 关心的是 linear separability + margin。这两个概念**正交**。

两组 representation 可以**用同一个 linear classifier 都能正确分类**，但 Gram matrix 完全不 align，CKA = 0。

### 2. 高维 + 大 n 下 outlier 主导 Gram matrix

d 大时，每个 inner product 是 d 个分量之和。Outlier 的 norm 大，它涉及的 $2n-1$ 个 Gram entries 全部 dominate Frobenius norm。

### 3. Subset translation 在 NN 里自然发生

某层 representation $X$，下一层 $Y = \sigma(XW + b)$。如果 $W$ 的某行 $W_j$ 满足 $\langle W_j, \vec{v}\rangle = 0$，那 $X \to X_{S, \vec{v}, c}$ 对 neuron $j$ 的输出完全没影响（BatchNorm 还会吸收 centering 偏移）。

**这意味着 NN 训练过程可能"自然"产生 subset translation，CKA 对此敏感但 functional behavior 完全不变**。

### 4. CKA 不是 metric

Williams et al. 2021 指出 CKA 不满足 triangle inequality，所以**不能当距离用**，不能用来做聚类、最近邻、hierarchical 分析。

参考：https://openreview.net/forum?id=L9JM-pxQOl

---

## 这对既往文献意味着什么

| 文献 | 主要 claim | 本文挑战 |
|---|---|---|
| Nguyen 2021 | wide/deep 网络 block structure 说明 representational similarity | block structure 可以人造或消除，accuracy 不变 |
| Raghu 2021 | ViT 和 CNN CKA map 差异说明它们学不同东西 | CKA map 可任意操纵 |
| Ramasesh 2021 | 深层 CKA 高表示 transfer 不易 forget | CKA 不该作 forgetting 代理 |
| Neyshabur 2020 | CKA 表示 transfer learning 中"什么被转移" | CKA 与 functional 未必对应 |
| Kornblith 2019 | well-trained 网络浅层和深层 CKA 低 | 可让浅深 CKA 接近 1 而 accuracy 不变 |

---

## 给实践者的建议

1. **别只用 CKA**：联合多个 metric（orthogonal Procrustes、linear probes、sparsity、可视化）
2. **关注 functional metrics**：accuracy、OOD performance、robustness
3. **training procedure 不受控时格外小心**：比如 open-source pre-trained model 来源不明
4. **检查 outlier**：少数 dominant samples 可能主导 CKA
5. **别把 CKA 值当距离用**：CKA 不是 metric

---

## 我的延伸想法

### 与 Neural Collapse 的联系

Papyan et al. 2020 发现训练良好的 classifier 末期 representation 进入 Neural Collapse：类内方差极小，类均值 collapse 成 simplex ETF。

NC 状态下，每个类中心的位置主导 Gram matrix。**对某个类中心做平移**，按 Theorem 1，CKA 暴跌，但 linear classifier 仍能完美分类。这给 Theorem 1 在 NC 状态下额外直觉。

参考：https://www.pnas.org/doi/10.1073/pnas.2018611117

### 与 Contrastive Learning 的联系

SimCLR / CLIP 等 contrastive objective 关心保 local pairwise structure，和 CKA 关心的 Gram alignment 概念上接近。本文结论暗示 contrastive learning 学到的 representation 也可能易被 outlier 主导。

参考：SimCLR https://arxiv.org/abs/2002.05709，CLIP https://arxiv.org/abs/2103.00020

### Mechanistic Interpretability 的方向

Anthropic circuits work（Elhage et al. 2021）关注 induction heads 等 specific circuit，用 causal interventions 而非 representation similarity。这暗示 mechanistic interpretability 应避免纯 representation-level similarity，倾向 functional/causal analysis。

参考：https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

### 大模型时代的 implication

如今 LLM evaluation 大量依赖 probing 和 representation analysis。如果用 CKA 比较两个 LLM（如 Llama vs Mistral）的中间层，得到的相似度未必反映它们做相同任务的能力。需要 functional probing + behavioral evaluation。

---

## 一句话总结

Linear CKA 把高维 representation 的几何结构压缩成 [0,1] scalar，这个压缩**丢失了 functional behavior 信息**：两个 representation 即使被同一 linear classifier 以相同 margin 分类，CKA 也能完全不同；反之，可以人为操纵 CKA 到任意值而不改变 accuracy。Theorem 1 给出 closed-form：subset translation 的 CKA 极限正比于 $\frac{\rho}{1-\rho} \cdot \frac{||\bar{S}||^2}{\mathbb{E}[||x||^2]} \cdot \sqrt{\text{PR}(X)}$，在标准 deep network 训练条件下（大 n + weight decay + 低 effective dimensionality）趋近 0。给社区敲警钟：依赖 CKA 单一指标得出的结论都该重新审视，未来应联合多种 similarity measures 并与 functional probing 互相印证。

---

# CKA Reliability Paper 深度解读

## 0. Paper 元信息

**标题**: *Reliability of CKA as a Similarity Measure in Deep Learning*
**作者**: MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, Eugene Belilovsky (Concordia + Université de Montréal + Mila)
**arXiv**: 推测在 https://arxiv.org/abs/2207.01162 附近，正式版 ICLR 2023

## 1. 一句话总结

Linear CKA 这个社区默认用来比较 neural representations 的 metric，对一类**非常常见且功能无影响的变换**（subset translation）极度敏感，可以被任意操纵到接近 0 或接近 1，**同时不改变模型的 functional behavior**。这戳穿了 Kornblith et al. 2019 提出 CKA 以来大量文献（Nguyen 2021, Raghu 2021, Ramasesh 2021, Neyshabur 2020）的核心结论根基。

参考：
- 原始 CKA paper: https://arxiv.org/abs/1905.00414
- Davari et al. 这篇: https://arxiv.org/abs/2207.01162 (估)

---

## 2. CKA 基础回顾（用来 build intuition）

### 2.1 HSIC 的定义

给定两组 representations $X \in \mathbb{R}^{n \times d_1}$ 和 $Y \in \mathbb{R}^{n \times d_2}$（n 个样本，维度分别为 $d_1, d_2$），定义 kernel matrices:
$$K_{i,j} = k(x_i, x_j), \quad L_{i,j} = l(y_i, y_j)$$

其中 k, l 是任意 PSD kernels。centering matrix:
$$H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$

- $I_n$: n×n 单位矩阵
- $\mathbf{1}$: n 维全 1 列向量
- $\mathbf{1}\mathbf{1}^\top$: n×n 全 1 矩阵
- $H$ 的作用: 把任一矩阵 M 变成 $HM$，等价于对每列减去列均值。对 kernel matrix K，HKH 就是 centered Gram matrix，相当于在 RKHS 中减去 empirical mean embedding。

HSIC:
$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(KHLH)$$

**直觉**: $\text{tr}(KHLH) = \text{tr}((HKH)(HLH))$，即两个 centered kernel matrices 的 Frobenius inner product。如果 X 的样本相似结构与 Y 的样本相似结构 align 得好，则内积大。HSIC 本质是 Gram matrix 之间的 second-order statistics 对齐。

### 2.2 CKA

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

**Linear CKA** 是 k = l = 内积 的特例:
$$K = XX^\top, \quad L = YY^\top$$
记号 $\text{CKA}(X, Y) := \text{CKA}(XX^\top, YY^\top)$

**性质**:
- 对正交变换（旋转、反射、置换）不变
- 对 isotropic scaling 不变
- 对 invertible linear transformation **不**不变 — 这是 Kornblith et al. 2019 故意设计的，他们论证完全 invariance to invertible linear maps 会让所有 width ≥ n 的 representation 给出相同结果

参考: Gretton et al. 2005 HSIC 原文 https://link.springer.com/chapter/10.1007/11564089_7

---

## 3. 核心理论结果: Theorem 1 (Subset Translation Sensitivity)

### 3.1 定理陈述

设 $X \in \mathbb{R}^{n \times p}$ 已 column-wise centered（每列均值 0），取子集 $S \subset X$，记
$$\rho = \frac{|S|}{|X|} \leq \frac{1}{2}$$

取单位向量 $\vec{v} \in \mathbb{R}^p$（$||\vec{v}|| = 1$）。定义**subset-translated** 版本:
$$X_{S, \vec{v}, c} = S \cup \{x + c\vec{v} : x \in X \backslash S\}$$

也就是: 子集 S 不动，其余点沿 $\vec{v}$ 方向平移距离 $c$。

**核心极限公式**:
$$\lim_{c \to \infty} \text{CKA}_{lin}(X, X_{S, \vec{v}, c}) = \Gamma(\rho) \cdot \frac{||\mathbb{E}_{x \in S}[x]||^2}{\mathbb{E}_{x \in X}[||x||^2]} \cdot \sqrt{\text{dim}_{PR}(X)} \tag{2}$$

### 3.2 变量逐项解析

| 项 | 含义 | 范围/性质 |
|---|---|---|
| $c$ | translation distance（被推向 ∞） | $c \to \infty$ |
| $\vec{v}$ | translation direction, 单位向量 | 任意 |
| $\rho$ | 未平移子集 S 的比例 | $\leq 1/2$（Cor 2 扩展到 > 1/2） |
| $\Gamma(\rho) = \frac{\rho}{1-\rho}$ | 比例因子 | $\rho = 1/2$ 时取最大值 1；$\rho \to 0$ 时 → 0 |
| $\mathbb{E}_{x \in S}[x]$ | subset S 的均值向量 | 因为 X centered，S 均值通常较小 |
| $||\mathbb{E}_{x \in S}[x]||^2$ | S 均值的 squared norm | weight decay 训练下偏小 |
| $\mathbb{E}_{x \in X}[||x||^2]$ | 全体 X 的平均 squared norm | = trace(covariance)，weight decay 下偏小 |
| $\text{dim}_{PR}(X) = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$ | **Participation Ratio**，effective dimensionality | $[1, p]$ |
| $\lambda_i$ | covariance matrix $\frac{1}{n}X^\top X$ 的第 i 个 eigenvalue | 全部 ≥ 0 |

### 3.3 Participation Ratio 直觉

PR 是神经科学/物理学借来的 effective dimensionality 估计 (Litwin-Kumar et al. 2017, Farrell et al. 2019, Mazzucato et al. 2016, Horoi et al. 2020):

- 所有 $\lambda_i$ 相等 → PR = p（最大，全维活跃）
- 一个 $\lambda_1 \gg \lambda_{i>1}$ → PR ≈ 1（最小，一维占主导）
- 一般 deep network 的 representation effective dimensionality 远小于 p，所以 PR 通常较小

直觉：$\text{Var}(X) = \sum_i \lambda_i$，$\sum_i \lambda_i^2$ 是"集中度"，PR 衡量 spectrum 的"扁平度"。

参考: Horoi et al. 2020 https://link.springer.com/chapter/10.1007/978-3-030-47358-7_30

### 3.4 主定理的直觉解读

这个极限公式说: **即使平移只占数据一半以下（甚至单点 outlier）的子集，CKA 都可以被打到接近 0**，因为：

1. $\Gamma(\rho)$ 在 $\rho \to 0$ 时 → 0：移动的子集越小，CKA 落得越快（反直觉！直觉上"动一点点"应该影响小）
2. $\rho = 1/2$ 时 $\Gamma$ = 1，但 weight decay 让 $\mathbb{E}_{x \in S}[x] \to 0$，整体仍然小
3. PR 也通常较小

### 3.5 Theorem 1 的证明骨架

证明关键步骤（见 Appendix C）:

**Step 1: Centering $X_{S, \vec{v}, c}$**

记 $C_1 = S$, $C_2 = X \backslash S$。$X_{S, \vec{v}, c}$ 的均值:
$$\bar{X}_{S, \vec{v}, c} = \frac{1}{n}\sum_{z \in X_{S,v,c}} z = \frac{|C_2| \cdot c\vec{v}}{n}$$

因为 X 已 centered: $\sum_{i \in X} x_i = 0$。

centered 后的 Y:
- $x \in C_1$: $y = x - \frac{|C_2|c\vec{v}}{n}$
- $x \in C_2$: $y = x + \frac{|C_1|c\vec{v}}{n}$

**Step 2: Linear HSIC 展开**

$$\text{HSIC}_{lin}(X, Y) = \frac{1}{(n-1)^2}\sum_{i,j} \langle x_i, x_j \rangle \langle y_i, y_j \rangle$$

三类求和:
- $(i, j) \in C_1 \times C_1$: 贡献 $\langle x_i, x_j \rangle^2 + O(c^0)$（与 c 无关的低次项）
- $(i, j) \in C_1 \times C_2$ 或 $C_2 \times C_1$: 贡献 $O(c^1)$
- $(i, j) \in C_2 \times C_2$: 贡献 $O(c^2)$

**Step 3: 分子取极限**

当 $c \to \infty$，分子主导项（$c^2$）整理后:
$$\text{numerator} \to |C_1| |C_2| \cdot ||\bar{x}_1 - \bar{x}_2||^2$$

其中 $\bar{x}_j = \mathbb{E}_{x \in C_j}[x]$ 是各类均值。

**Step 4: 分母取极限**

分母是 $\sqrt{\text{HSIC}(X,X) \cdot \text{HSIC}(Y,Y)}$。

- $\text{HSIC}(X,X) = \frac{1}{(n-1)^2} \sum_{i,j} \langle x_i, x_j \rangle^2 = \frac{1}{(n-1)^2} ||X X^\top||_F^2$
- 关键: $||X X^\top||_F^2 = \sum_i \lambda_i^2$，其中 $\lambda_i$ 是 covariance matrix $\frac{1}{n}X^\top X$ 的 eigenvalues
- $\text{HSIC}(Y,Y)$ 主导项 $O(c^4)$

整理后分母 → $n^2 \mathbb{E}_{x \in X}[||x||^2] \cdot \text{PR}(X)^{-1/2}$

**Step 5: 合并并使用 centering 约束**

由于 X centered: $|C_1|\bar{x}_1 + |C_2|\bar{x}_2 = 0$，可得 $\bar{x}_2 = -\frac{|C_1|}{|C_2|}\bar{x}_1$。代入:

$$||\bar{x}_1 - \bar{x}_2||^2 = \left(1 + \frac{\rho}{1-\rho}\right)^2 ||\bar{x}_1||^2$$

合并所有 $\rho$ 项:
$$\Gamma(\rho) = \rho(1-\rho)\left(1 + \frac{\rho}{1-\rho}\right)^2 = \frac{\rho}{1-\rho}$$

最终得到公式 (2)。

---

## 4. 三个推论

### 4.1 Corollary 2 (ρ > 1/2 仍成立)

证明中 ρ ≤ 1/2 仅用于 Γ 的 bound。实际表达式在 $\rho \in (0.5, 1)$ 也成立，Γ 仍能趋于 0（如 $\rho \to 1$）。意思是: **移动的子集无论小还是大，CKA 都能被打到极低**。

### 4.2 Corollary 3 (Outlier Sensitivity)

**情形**: $S = \{\hat{x}\}$，单点 outlier。$\rho = 1/n$。

代入主定理:
$$\Gamma(\rho) = \frac{1/n}{1 - 1/n} \approx \frac{1}{n}$$

当 n = 10000 时，CKA 极限值大约被乘以 $10^{-4}$ 量级。即使 X 和 $X_{S,v,c}$ 除了一个点之外完全相同，CKA 都会暴跌到接近 0。

**这解释了 Nguyen et al. 2022 的 block structure 现象** — 他们发现 wide/deep network 的 CKA heatmap 出现"block"是因为少数 dominant data points 主导。Cor 3 给了这个现象形式化解释。

参考:
- Nguyen et al. 2022 https://arxiv.org/abs/2202.07184
- Nguyen et al. 2021 https://arxiv.org/abs/2010.15327

### 4.3 Corollary 4 (Linear Separability Preservation)

**情形**: S 和 X\S 线性可分，存在 hyperplane $(w, k)$:
- $x \in S \Rightarrow \langle w, x \rangle \leq k$
- $x \in X \backslash S \Rightarrow \langle w, x \rangle > k$

取 $\vec{v} \perp w$，则平移不跨越超平面:
$$\langle w, x + c\vec{v} \rangle = \langle w, x \rangle + c\underbrace{\langle w, \vec{v}\rangle}_{=0} = \langle w, x \rangle$$

因此 $x \in X \backslash S \Rightarrow \langle w, x + c\vec{v} \rangle > k$，**linear separability 和 margin 都完全保留**！

但 Theorem 1 仍然适用 — CKA 可以任意低。

**哲学冲击**: 经典 ML 理论 (Bartlett & Shawe-Taylor 1999) 告诉我们 large margin 是 generalization 的关键；deep network 的最后一层 representation 几乎完美线性可分 (Zeiler & Fergus 2014, Oyallon 2017)。两个 representation 即使**用同一个 linear classifier 都能正确分类**，CKA 也可能说它们"完全不同"。

参考:
- Bartlett & Shawe-Taylor 1999 https://dl.acm.org/doi/10.5555/646330.686428
- Oyallon 2017 https://arxiv.org/abs/1610.06962

---

## 5. 实验

### 5.1 Sec 4.1: Early Layer High CKA 不代表 features 相似

**设置**: 训练三种 CIFAR10 网络:
1. **Generalized** — 正常训练 (100 epochs, AdamW, lr=1e-3, wd=5e-4, cosine schedule)
2. **Memorized** — 用随机标签训练（memorize 数据）
3. **Random init** — 不训练，仅前向

**网络架构** (Appendix A.2): 9 层 CNN
- 前 2 层 conv: kernel 7×7
- 后 6 层 conv: kernel 3×3
- 通道: [16, 16, 32, 32, 32, 64, 64]
- 全部 ReLU + BatchNorm
- 最后 FC layer 分类

**结果** (Figure 2, 3):
- 早期层 CKA ≈ 0.9 (high)
- 但 convolution filters 视觉上完全不同 (Figure 3)
- Linear probe accuracy (Figure 16) 显示 generalized network 的早期 features 比 random init 好得多

**直觉**: 浅层的 raw pixel statistics 主导 CKA，与"学到有用 features"无关。这戳穿了 Ramasesh et al. 2021 等用 CKA 做"feature usefulness"代理的做法。

参考:
- Krizhevsky CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
- AdamW: https://arxiv.org/abs/1711.05101
- BatchNorm: https://arxiv.org/abs/1502.03167

### 5.2 Sec 4.2: Subset Translation 实验验证 Theorem 1

#### 5.2.1 人工数据 (Figure 4a)

构造 $X \in \mathbb{R}^{20000 \times 1000}$:
- 10000 点: 均匀采样自 1000 维 unit cube centered at origin
- 10000 点: 均匀采样自相似 cube，center 在 (1.1, 0, ..., 0)
- 两组沿第 1 维 linearly separable

构造 Y: 将第二组点沿随机方向（d-ball 采样）平移距离 c。

**结果**: 
- Linear CKA 和 RBF CKA 都迅速降到 < 0.2
- 即使 c 不大就开始下降
- RBF CKA with $\sigma = 0.2 \times \text{median}$ 不受影响，但 σ 这么小时 RBF CKA 几乎没信息量（Kornblith 2019 Table 2 也证实）

#### 5.2.2 真实 CNN (Figure 4b)

CIFAR10 上训练 9-layer CNN，取 last hidden layer representations X。用 SVM 提取超平面 (≈91% accuracy)。

构造 Y: 平移一个类的点沿不跨越超平面方向 → 保留 margin 和 separability。

**结果**:
- 平移距离增大 → CKA 迅速降到 0
- 即使 SVM 对 X 和 Y 都能 91% 正确分类
- outlier 情形: 单个点平移，CKA 也能显著下降（虽然需要较大 c）

### 5.3 Sec 4.3: CKA Map Optimization — 最强 evidence

#### 5.3.1 优化目标

$$\theta_{new}^{*} = \arg\min_\theta \left( \mathcal{L}_{distill}(f_{\theta^*}(X), f_\theta(X)) + \lambda \mathcal{L}_{map}(M_{f_\theta(X)}, M_{target}) \right) \tag{3}$$

**变量**:
- $\theta^*$: 原始训练好的网络参数
- $\theta$: 当前优化的网络参数
- $f_{\theta^*}(X)$: 原始网络的 output logits
- $\mathcal{L}_{distill}$: 蒸馏 loss，保持 functional behavior
- $M_{f_\theta(X)}$ [i,j] = CKA(layer_i, layer_j) of current network
- $M_{target}$: 任意指定的目标 CKA map
- $\lambda$: 动态平衡权重
- $\mathcal{L}_{map} = \sum_{i,j} \ln\cosh(M[i,j]_{f_\theta(X)} - M[i,j]_{target})$: soft L1 loss

**动态 λ 调整 (Algorithm 1)**:
- 设 $\delta_{acc} = \text{original acc} - \text{current val acc}$
- 若 $\delta_{acc} > \eta$ (threshold): $\lambda \leftarrow \lambda \times \alpha$ (减小 λ 重视 accuracy)
- 否则: $\lambda \leftarrow \lambda / \alpha$ (增大 λ 强化 CKA map 匹配)
- 实验中 $\alpha = 0.8$, 初始 $\lambda = 500$

#### 5.3.2 训练细节 (Appendix A.3)

| Hyperparameter | Original | Optimized w.r.t. Map |
|---|---|---|
| Epochs | 200 | 30 |
| Initial LR | 0.01 | 0.001 |
| LR Schedule | Cosine | Cosine + 500 step warmup |
| Distillation temperature | — | 0.2 |
| Initial λ | — | 500 |
| λ scaling α | — | 0.8 |
| Optimizer | AdamW | AdamW |

#### 5.3.3 三个 target CKA map 场景 (Figure 5)

**Scenario 1**: Maximize CKA(1st layer, last layer)
- 目标: 让浅层和深层 CKA ≈ 1
- 结果: 几乎不损失 accuracy，CKA map 被改为"early-late 全 1"
- **直接反驳 Kornblith 2019 的"well-trained 网络 early-late CKA 低"**

**Scenario 2**: Maximize CKA all pairs (uniform block)
- 目标: 所有层对 CKA ≈ 1
- 结果: 成功，accuracy 仅微跌
- **反驳 Nguyen 2021 "block structure 仅出现在 overparameterized 模型"**

**Scenario 3**: Minimize CKA all pairs (anti-diagonal)
- 目标: 所有层对 CKA ≈ 0
- 结果: 成功，accuracy 大约保持

#### 5.3.4 Wider Networks 实验 (Figure 7)

ResNet-34 不同宽度 (1×, 2×, 4×, 8×, 16×) 在 CIFAR10 上训练。

- Original: 宽网络展现 block structure (Nguyen 2021 的发现)
- Optimized: 把所有宽网络的 CKA map 都改成 thin network 的样子，accuracy 不变

**Cross-dataset 验证** (Figure 10): 在 PatchCamelyon (组织病理数据集，与 CIFAR10 像素分布完全不同) 上测 CKA map，optimized 网络仍然呈现 thin network 模式。说明 manipulation 不是 dataset-specific 的表面现象，而是真正的 representation geometry 改变。

参考:
- He et al. 2016 ResNet: https://arxiv.org/abs/1512.03385
- PatchCamelyon: https://arxiv.org/abs/1806.03962

#### 5.3.5 ViT 实验 (Figure 14)

Vision Transformer (8 blocks, 12 heads, hidden 256, 200 epochs) 同样可以 CKA map 被任意操纵。说明结论**架构无关**。

参考:
- Dosovitskiy et al. 2020 ViT: https://arxiv.org/abs/2010.11929

#### 5.3.6 Representation Mechanism (Figure 8)

PCA 可视化 optimized network 的 last hidden representation: **单个 class 的部分点被整体平移到远离其他 classes** — 这就是 Theorem 1 描述的 subset translation！完全是 emergent behavior，优化器自己发现了这个 trick。

---

## 6. 旁支结果

### 6.1 Invertible Linear Transformation 敏感性 (Figure 9)

虽然 Theorem 1 没覆盖这种变换，但实验显示: 即使 $M \in \mathbb{R}^{d \times d}$ 是接近单位阵的高斯随机矩阵 ($\mu, \sigma$ 小)，$Y = XM$ 的 CKA 也降到 0。

Kornblith 2019 论证对 invertible linear maps invariance 是 undesired 的（因为 width ≥ n 时无意义），但**对 width ≪ n 的场景** (e.g. 全连接最后一层) 这个 invariance 反而是合理的 desirable 性质。Linear CKA 既丢了 invariance 又丢了 sensitivity 的"恰当度"。

### 6.2 OOD Robustness (Figure 17)

把 optimized networks 在 CIFAR-10-C (Hendrycks & Dietterich 2018) 上测试，performance 几乎不变。说明 CKA manipulation 真的没改变 model 的 functional behavior，连 robustness 都保留了。

参考: https://arxiv.org/abs/1903.12261

---

## 7. 建模 Intuition 总结

让我把这套结果组织成几个核心 takeaway:

### Intuition 1: CKA = Gram matrix alignment, 不是 feature alignment

CKA 比较的是 $XX^\top$ 和 $YY^\top$（样本×样本的相似性矩阵）。两个 representation 即使**在 feature space 上完全等价**（用同一 linear classifier 都能分），只要 Gram matrix 不 align，CKA 就低。

这导致 CKA 测的是 "global second-order structure"，而 network 的 functional behavior 是 "linear separability + margin"，这两个概念**正交**。

### Intuition 2: 高维空间中 outlier 主导 Gram matrix

高维 (d 大) + 大样本 (n 大) 下，Gram matrix $XX^\top$ 的每个元素是 $d$ 维内积。Outlier 的 norm 如果显著大于其他点，它会主导整个 Gram matrix 的 Frobenius norm。

$$\mathbb{E}_i[||x_i||^2] = \frac{1}{n}\sum_i ||x_i||^2$$

如果 $x_{outlier}$ 的 $||x||^2$ 是其他点的 $n$ 倍，则它在 $\sum_i ||x_i||^2$ 中占主导。所有 $n$ 个 outlier-related entries $(i, j)$ where $i$ or $j$ is the outlier 共有 $\sim 2n$ 个，每个 squared inner product 都很大。

### Intuition 3: Subset Translation 在 NN 里"自然发生"

考虑一个 layer 的 representation $X$ 和下一层: $Y = \sigma(XW + b)$。如果 $W$ 的某些行（对应某些 neurons）方向 $\vec{v}$ 上的分量 $\langle W_j, \vec{v}\rangle = 0$，则 $X \mapsto X_{S,\vec{v},c}$ 后输出不变（除了一阶 centering 影响，BN 会吸收）。

这意味着 **NN 训练过程中可能 "自然" 产生 subset translation**，CKA 对此敏感但 functional 完全不变。

### Intuition 4: CKA 是非 metric

Williams et al. 2021 指出 CKA 不满足 triangle inequality，所以**不能用作距离**做下游分析（如聚类、最近邻）。

参考: https://openreview.net/forum?id=L9JM-pxQOl

### Intuition 5: 多个 similarity measure 联合使用

作者建议:
- Orthogonal Procrustes (Ding 2021, Williams 2021): 直接学一个正交变换最小化 ||XW - Y||_F
- Sparsity 比较 (Kornblith 2021)
- Linear probes (Alain & Bengio 2016, Davari 2022)
- 可视化 (Nguyen 2019 PHATE, Horoi 2020, Recanatesi 2021)

参考: https://arxiv.org/abs/1610.01644

---

## 8. 与其他 representation similarity 方法的对比

| 方法 | 原理 | 优势 | 弱点 |
|---|---|---|---|
| **Linear CKA** (Kornblith 2019) | $\text{HSIC}(XX^\top, YY^\top)$ 归一化 | 对正交变换和 isotropic scaling 不变 | 对 outlier、subset translation、margin-preserving 变换极敏感 |
| **SVCCA** (Raghu 2017) | 先 SVD 再 CCA 取均值 correlation | 降维去噪 | 只看 dominant singular vectors，对 invertible transforms 不变 |
| **PWCCA** (Morcos 2018) | weighted CCA by projection magnitude | 减弱噪声方向影响 | 同上 |
| **Orthogonal Procrustes** (Ding 2021, Williams 2021) | $\min_{W: W^\top W=I} ||XW - Y||_F$ | 满足 triangle inequality，几何可解释 | 计算 cost 高 |
| **Linear Probes** (Alain & Bengio 2016) | 在 representation 上训练 linear classifier | 直接衡量 separability | 不衡量两个 representation 间的相似性 |
| **Neuron match** (Li 2015, Wang 2018) | 找 neuron-to-neuron 对应 | 细粒度 | 难处理 many-to-many |

参考:
- SVCCA: https://arxiv.org/abs/1706.05806
- PWCCA: https://arxiv.org/abs/1806.05759
- Linear probes: https://arxiv.org/abs/1610.01644

---

## 9. 这篇 paper 的重要含义

### 9.1 对既往文献的影响

| 文献 | 主要 claim | 本文 challenge |
|---|---|---|
| Nguyen et al. 2021 | wide/deep 网络出现 block structure 说明 representational similarity | 可人为生成或消除 block structure 而不改变 functional behavior |
| Raghu et al. 2021 | ViT 和 CNN 的 CKA map 差异说明它们学不同东西 | CKA map 可被任意操纵 |
| Ramasesh et al. 2021 | 深层 CKA 高表示 transfer 不易 forget | CKA 不能用作 forgetting 代理 |
| Neyshabur et al. 2020 | transfer learning 中 CKA 表示"什么被转移" | CKA 与 functional 未必对应 |

### 9.2 给实践者的建议

1. **不要单独用 CKA 下结论**: 联合用 multiple metrics (Procrustes, linear probes, sparsity, 可视化)
2. **关注 functional metrics**: classification accuracy, OOD performance, robustness
3. **当 training procedure 不受控时尤其小心**: 比如 open-source pre-trained models 来源不同
4. **检查 outlier**: 数据中少数 dominant samples 可能主导 CKA
5. **不要把 CKA 值当距离**: CKA 不是 metric

---

## 10. 我自己的延伸思考

### 10.1 与 Neural Collapse 的联系

Papyan et al. 2020 发现训练良好的 classifier 末期 representation 会形成 Neural Collapse:
- 类内方差极小，类均值（NC1）collapse 成 simplex ETF
- classifier weights 与 class means 对齐 (NC4)

在 NC 状态下，每个类中心的位置主导了 Gram matrix。**对某一类中心做平移**，按 Theorem 1，CKA 会暴跌，但 linear classifier 仍可分类。这给了 Theorem 1 在 NC 状态下额外直觉。

参考: Papyan et al. 2020 https://www.pnas.org/doi/10.1073/pnas.2018611117

### 10.2 与 CCA / Procrustes 比较

CCA 关心的是 **subspace alignment**: 找 X 和 Y 的子空间最大 correlation。Subset translation 只改变 mean，不动子空间结构，所以 CCA 不敏感。

Linear Procrustes 关心 **point-wise 对齐**: 找正交 W 让 ||XW - Y||_F 最小。Subset translation 改变了部分点的位置，Procrustes 也敏感，但敏感度与 translation 量级线性相关，不会像 CKA 一样在小 ρ 时直接 collapse 到 0。

### 10.3 与 mechanistic interpretability 的联系

Anthropic 的 circuits work (Elhage et al. 2021) 关注 induction heads 等 specific circuit，用 causal interventions 而非 representation similarity。这暗示 mechanistic interpretability 应避免纯 representation-level similarity，倾向 functional/causal analysis。

参考: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

### 10.4 大模型时代的 implication

如今 LLM evaluation 大量依赖 probing、representation analysis。如果用 CKA 比较两个 LLM (e.g., Llama vs Mistral) 的中间层，得到的相似度未必反映它们做相同任务的能力。需要 functional probing + behavioral evaluation。

### 10.5 与 Contrastive Learning 的联系

SimCLR / CLIP 等 contrastive methods 的 objective（NT-Xent）在某种意义上是**保 local pairwise structure**，与 CKA 关心的 Gram matrix alignment 概念上接近。本文结论暗示 contrastive learning 也可能学到 representation 易被 outlier 主导。

参考:
- SimCLR: https://arxiv.org/abs/2002.05709
- CLIP: https://arxiv.org/abs/2103.00020

---

## 11. 开放问题

1. **Nonlinear CKA 的理论分析**: 本文只对 linear CKA 给出严格定理。RBF CKA 实验也敏感，但理论不清。RKHS 中"对应于 translation"的变换是什么？

2. **Functional equivariant similarity measure**: 能否设计一个度量，**invariant to subset translation 且 sensitive to functional behavior changes**？Procrustes 接近但不完美。

3. **CKA 的 fix**: 是否可以 robustify CKA，例如用 median-based HSIC 替换 mean-based？

4. **Outlier detection in representations**: 既然 CKA 对 outlier 极敏感，可否反向用 CKA 做 outlier detector？

5. **Generalization 与 CKA**: 是否存在特定 CKA 值范围**严格对应**于 generalization 能力？目前看似乎没有。

---

## 12. 推荐阅读路径

如果你刚接触这个领域，建议顺序:

1. **Raghu 2017 SVCCA** - 第一篇系统比较 representation 的工作
2. **Kornblith 2019 CKA** - 当前主流方法
3. **Nguyen 2021 Wide & Deep** - 实证发现 block structure
4. **Ding 2021 Statistical Testing** - 用统计严谨视角看 CKA
5. **Nguyen 2022 Block Structure Origins** - 解释 block structure 来自 outlier
6. **Williams 2021 Shape Metrics** - 提出满足 metric 公理的替代品
7. **Davari 2022 (本文)** - 系统性 reliability analysis

URLs:
- https://arxiv.org/abs/1706.05806
- https://arxiv.org/abs/1905.00414
- https://arxiv.org/abs/2010.15327
- https://arxiv.org/abs/2106.06434
- https://arxiv.org/abs/2202.07184
- https://arxiv.org/abs/2110.14739
- 本文 arXiv (估测): https://arxiv.org/abs/2207.01162

---

## 13. 一段话总结

Linear CKA 把高维 representation 的几何结构压缩成一个 [0,1] scalar，这个压缩过程**丢失了 functional behavior 信息**：两个 representations 即使被同一个 linear classifier 以相同 margin 分类，CKA 也能完全不同；反之，可以**人为操纵 CKA 到任意值而不改变 model accuracy**。Theorem 1 给出 closed-form formula 解释这个现象：subset translation 的 CKA 极限值正比于 $\Gamma(\rho) \cdot ||\bar{S}||^2 / \mathbb{E}[||x||^2] \cdot \sqrt{\text{PR}(X)}$，在大 n + weight decay + 低 effective dimensionality 的标准 deep network 训练条件下趋近 0。这给社区敲了警钟：依赖 CKA 单一指标得出的所有结论都应重新审视，未来应用应联合多种 similarity measures 并与 functional probing 互相印证。
