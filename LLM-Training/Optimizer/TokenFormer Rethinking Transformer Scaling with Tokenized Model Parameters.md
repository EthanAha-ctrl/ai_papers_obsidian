---
source_pdf: TokenFormer Rethinking Transformer Scaling with Tokenized Model Parameters.pdf
paper_sha256: 24ec9dad4ac8f6b37ae6803b4ddad7b502ad8e24d669dd19ed6bcc39d2c8716f
processed_at: '2026-08-12T16:33:56-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 TokenFormer

## 一句话版本

把 Transformer 里的权重矩阵也变成一堆 token，让 attention 不光算 token 之间的事，也算 token 跟参数之间的事。这样参数数量就能像堆积木一样随便加，不用每次从头训。

---

## 问题出在哪

你现在训了一个 124M 的 GPT。想扩到 1.4B，传统做法是把每一层的 hidden dimension 从 768 加宽到 4096，加几层，然后整个权重矩阵全部重新随机初始化，从头训 300B tokens。

这就好比盖房子，你想加一层楼，结果发现得把整栋楼拆了重盖。

为什么？因为 Transformer 里 token 跟权重打交道的方式是 **矩阵乘法** $Y = XW$。这个 $W$ 的形状是写死的，$W \in \mathbb{R}^{d_{in} \times d_{out}}$，你想加宽模型，$d_{in}, d_{out}$ 都得变，$W$ 就废了，老知识全丢。

而 token 跟 token 打交道用的是 **attention**，这个机制天然支持变长序列——你 sequence 长度随便变，attention 都能算。这就是为什么 Transformer 能处理任意长度文本，但死活不能随便改宽度。

---

## 核心招数：把权重也 tokenize

TokenFormer 说：既然 attention 这么灵活，**为什么不让 token 也用 attention 去查询权重？**

具体怎么做：原来一个 linear layer $Y = XW$，$W$ 是 $d \times d$ 的矩阵。现在把 $W$ 拆成两组"参数 token"：
- 一组叫 $K_P$，n 个 token，每个 d 维——相当于"索引"；
- 一组叫 $V_P$，n 个 token，每个 d 维——相当于"内容"。

然后计算变成：

$$\text{output} = \text{softmax}(X \cdot K_P^\top) \cdot V_P$$

翻译成人话：**输入 token 先跟所有"索引 token"算相似度，得到一组权重，再加权求和"内容 token"**。这就是 cross-attention——query 是数据，key/value 是参数。

为什么这等价于 linear layer？想象极端情况：n = d，$K_P$ 是单位矩阵，attention score 退化成"硬选第 i 个"，那就退化成 $XW$ 了。但 n 现在是个**自由变量**，想加几个加几个，跟 $d$ 解耦了。

这就是论文里说的 **Pattention layer**。

参考：https://arxiv.org/abs/2410.23168

---

## 为什么这个能扩展模型

现在你的参数池有 n 个 KV pair。想扩到 n+m 个？直接在后面 concat m 个新的：

$$K_P^{\text{new}} = [K_P^{\text{old}}, K_P^{\text{new}}]$$
$$V_P^{\text{new}} = [V_P^{\text{old}}, V_P^{\text{new}}]$$

关键技巧：**新加的 $K_P^{\text{new}}$ 初始化为零**。

为什么是零？看计算：$X \cdot 0 = 0$，零向量经过 L2 norm + GeLU 还是零，所以新加的 value token 根本不参与输出。**模型输出跟扩展前一模一样**，知识一点没丢。

然后你继续训，新参数慢慢学起来，老参数也微调。相当于在原有知识上"长"出新容量，而不是推倒重来。

这个性质标准 softmax 做不到——$\exp(0) = 1$，新加的 key 会把所有 attention score 拉低，立刻破坏分布。所以作者把 softmax 改成了 **L2 norm + GeLU**，这是个工程性的 patch，但数学上很关键：它让零输入严格映射到零输出。

参考 LoRA 用的是类似思路（A 随机初始化、B 初始化为零）：https://arxiv.org/abs/2106.09685

---

## 实验结果有多香

从 124M 扩到 1.4B，作者的做法：
1. 先从头训 124M，300B tokens；
2. 扩到 354M，只用 30B tokens 接着训；
3. 扩到 757M，再用 30B tokens；
4. 扩到 1.4B，再用 30B tokens。

最终 1.4B 模型 perplexity **11.60**，而从 scratch 训 300B tokens 的 1.4B Transformer 是 **11.63**——基本一样，甚至略好。

但成本：TokenFormer 累计用了 300+30+30+30 = 390B tokens 的 compute；传统做法如果想训出 124M、354M、757M、1.4B 四个尺寸，得 4×300B = 1.2T tokens。**省了一半多**。

更狠的对比：同样只给 30B tokens 预算，从 scratch 训 1.4B Transformer 得到 PPL 13.34，TokenFormer 渐进扩展得到 11.77。**同样算力下 TokenFormer 完胜**。

---

## 还有个副作用：长文本更便宜

标准 Transformer 里 attention 的 FLOPs 是 $4 \cdot n_{\text{layer}} \cdot d_{\text{model}} \cdot T^2$，所以你把 $d_{\text{model}}$ 加宽，长文本的代价 quadratic 增长。

TokenFormer 扩参数时 $d_{\text{model}}$ **不变**（只增 KV pair 数量），所以 token-token attention 的 quadratic 项系数不变。**模型变大了，长文本的相对代价反而下降**。

这对 Chain-of-Thought、长上下文 LLM 是结构性优势。

---

## 有什么代价

1. **Inference 慢**：原来 linear 是一次 matmul，Pattention 是两次 matmul + softmax，FLOPs 大约多 $n/d$ 倍。所以部署时比同尺寸 dense transformer 慢。作者承认这点，说未来要接 MoE 稀疏路由解决。

2. **修改后的 softmax（L2+GeLU）是工程性 hack**：没有深厚理论基础，换到别的规模、别的任务可能要重新调。

3. **最大就训到 1.4B**：scaling law 形态在更大尺寸上未知。

---

## 这玩意儿本质是什么

作者自己在 §5 说了：**这就是一个极端版的 Mixture of Experts**。每个 KV pair 就是一个 expert，attention score 就是 soft routing。区别是传统 MoE 只激活 top-2 expert 省 FLOPs，TokenFormer 默认全 expert 都参与。

换个角度看，它也像 **Neural Turing Machine 的可微版**——参数就是外部 memory，input token 通过 attention 检索 memory。区别是 NTM 的 memory 是临时 state，TokenFormer 的 memory 就是模型参数本身，跟着梯度一起学。

更接地气的类比：**这就是把权重矩阵变成了一个可检索的 key-value store**。Linear layer 是"硬编码查表"，Pattention 是"软检索查表"。软检索的好处是表可以随便加条目，硬编码的表改一下形状就崩了。

参考：
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- Geva et al. FFN as memory: https://arxiv.org/abs/2012.14913
- Switch Transformer (MoE): https://arxiv.org/abs/2101.03961

---

## 我觉得最深刻的点

这篇 paper 真正的 insight 不在于"省了多少 compute"——省 compute 是结果。真正的 insight 是：

**Transformer 的灵活性只存在于 token-token 这一半，token-parameter 这一半一直是死板的矩阵乘法。把后一半也 attention 化，整个模型才真正"原生可扩展"。**

这跟 retrieval-augmented generation、MoE、memory-augmented networks、甚至 Mamba 的 state space 其实都在指向同一个方向：**把"参数"从一个 monolithic 矩阵碎片化成可寻址的单元**。TokenFormer 用 attention 给了一个统一的数学语言。

至于它能不能真扛到 100B+ 规模、能不能跟 sparse routing 结合把 inference 成本降下来、能不能用来 merge 多个独立训练的模型（作者埋了伏笔但没做）——都是开放问题。但作为一个 conceptual contribution，我觉得它是 2024 年最值得读的 architecture paper 之一。

---

# TokenFormer：把 Model Parameters 也 Tokenize，让 Transformer 原生可扩展

## 1. Motivation：Transformer 的扩展瓶颈在 token-parameter interaction

Transformer 的计算本质上分成两类 interaction：

- **token-token interaction**：用 self-attention 实现，对 sequence length T 灵活，可处理任意长度的 tokenized data；
- **token-parameter interaction**：用 linear projection（matmul）实现，对 d_model 是 hard-coded 的，W ∈ R^(d_in × d_out) 一旦固定就要从头训练。

问题就出在后者。当你想从 124M 扩到 1.4B，传统做法是把 d_model 从 768 加宽到 4096，每一层 W^Q, W^K, W^V, W^O, FFN 全部要重新初始化训练——Biderman et al. 2023 [Pythia, https://arxiv.org/abs/2304.01373] 证明 chinchilla-optimal training 几乎每加一个量级就要重训 300B+ tokens，开销随 scale 二次爆炸 [Kaplan et al. 2020, https://arxiv.org/abs/2001.08361]。

TokenFormer 的 insight 是：**把 model parameter 当成一种特殊 token**，让 attention 机制同时处理 token-token 和 token-parameter 两种 interaction，于是 parameter 数量 N_p 就变成了一个独立于 d_model 的轴，可以沿着它渐进扩展而不动其他计算。

作者特别指出这是一种 *extreme MoE instantiation*——每个 key-value pair 就是一个 expert，且 routing 是 soft attention。这点直觉上很优雅，我下面会展开。

参考链接：
- 原论文：https://arxiv.org/abs/2410.23168
- 代码：https://github.com/Haiyang-W/TokenFormer
- nanoGPT（作者的 124M baseline 用的是 Karpathy 的 repo）：https://github.com/karpathy/nanoGPT

---

## 2. Pattention Layer：核心创新

### 2.1 公式与符号

第 4 式是全文最关键的公式：

$$\text{Pattention}(X, K_P, V_P) = \Theta(X \cdot K_P^\top) \cdot V_P$$

变量含义：

- $X \in \mathbb{R}^{T \times d_1}$：input tokens，T 是 sequence length，$d_1$ 是 input channel dim；
- $K_P \in \mathbb{R}^{n \times d_1}$：**key parameter tokens**，n 是 parameter token 的数量，是新引入的轴；
- $V_P \in \mathbb{R}^{n \times d_2}$：**value parameter tokens**，$d_2$ 是 output channel dim；
- $\Theta$：modified softmax（下面专门讲）；
- $X \cdot K_P^\top \in \mathbb{R}^{T \times n}$：input token 与 parameter token 的相似度矩阵；
- 输出 $\in \mathbb{R}^{T \times d_2}$，与 $V_P$ 的列维度一致。

直觉上，这就是一个 **cross-attention**：query 是数据 token，key/value 是参数 token。它把 Linear layer $Y = XW$ 替换成了一个可变容量的"参数池"检索器。

为什么这样可以替代 linear projection？想象 n = d_1, $K_P$ 是 $\mathbb{R}^{d_1 \times d_1}$ 单位矩阵的离散化版本，$V_P = W^\top$，attention score 退化为 hard pick——那就退化成了普通 linear。但只要 n 任意可变，这个"虚拟的中间维度"就脱离了 $d_1, d_2$，可以随 scale 自由增长。

### 2.2 为什么不用标准 softmax？——梯度分析（Appendix A）

标准 softmax：
$$S_i = \frac{\exp(A_i / \sqrt{d})}{\sum_j \exp(A_j / \sqrt{d})}$$
梯度：
$$\frac{\partial S_i}{\partial A_j} = \frac{1}{\sqrt{d}} S_i (\mathbb{1}_{i=j} - S_j)$$

问题：$\exp$ 把高分 logit 推得极尖，softmax 分布非常 sharp，导致 $S_i \to 1$ 或 $\to 0$，于是梯度 $S_i(1-S_j) \to 0$。在 cross-attention 这种长 sequence × 长 parameter 的场景下，vanishing gradient 严重。

Pattention 的 $\Theta$ 用 **L2 norm + GeLU**：
$$\hat{S}_i = f(Z_i) = f\left(\frac{A_i \sqrt{n}}{\sqrt{\sum_j A_j^2}}\right)$$

- 分子：$A_i \sqrt{n}$，$\sqrt{n}$ 是为了把 L2 norm 后的量纲稳定（不随 n 衰减）；
- 分母：$||A||_2$，对所有 n 个 score 做 L2 归一化；
- $f$：GeLU [Hendrycks & Gimpel 2016, https://arxiv.org/abs/1606.08415]，平滑、有非零梯度区。

对应的梯度（Eq 25）：
$$\frac{\partial \hat{S}_i}{\partial A_j} = \begin{cases} f' \frac{1}{\sqrt{n}} \frac{1}{||A||_2} (n - Z_i Z_j) & i = j \\ -f' \frac{1}{\sqrt{n}} \frac{1}{||A||_2} Z_i Z_j & i \neq j \end{cases}$$

关键对比：Softmax 的梯度依赖 $S_i S_j$（指数，分布尖锐），Pattention 的梯度依赖 $Z_i Z_j$（L2 归一化后 $|Z_i| \leq \sqrt{n}$，分布平滑）。Table 4 ablation 显示 GeLU vs $\exp$ 在 ViT-B/16 上 +2.1 acc，L2 vs L1 再 +0.8。

**直觉**：softmax 在 query-key 数量接近、维度接近时是 well-behaved 的（self-attention 那种 T×T），但 Pattention 里 query 维度是 T（输入），key 维度是 n（参数），分布特征完全不同，作者用 L2+GeLU 让 score 分布更均匀，避免训练初期就陷入"winner-take-all"的尖峰。

### 2.3 一个关键性质：zero-init new keys → 输出不变（Appendix B）

设原模型有 n 个 key-value pair，加 m 个新 pair，新 keys 全为 0：

$$\hat{A} = \begin{bmatrix} K_P \\ 0 \\ \vdots \\ 0 \end{bmatrix} X = \begin{bmatrix} A \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

L2 norm 后 $Z_{\text{new}} = 0$，GeLU(0) = 0，所以新 value 完全不参与。**输出 $\hat{O} = O$ 严格不变**。

这个性质标准 softmax 做不到——$\exp(0) = 1 \neq 0$，新加的 key 就会拉低所有旧 score，立刻破坏分布。

**为什么这个性质重要？** 它意味着 scaling 时模型不会"忘记"已学知识，optimizer 状态、loss landscape 都不被打断，相当于 LoRA [Hu et al. 2022, https://arxiv.org/abs/2106.09685] 的"zero-init B, random-init A"思路在 attention 里的等价物。但 LoRA 只动少量 adapter，TokenFormer 是把整套 weight 都放进了这种"可无损扩展"的容器里。

---

## 3. 整体架构

### 3.1 一层 TokenFormer 的计算流

Eq 5-6 是 pre-norm 残差：
$$X_{\text{inter}} = X_{\text{in}} + \text{MHA}(\text{LN}(X_{\text{in}}))$$
$$X_{\text{out}} = X_{\text{inter}} + \text{FFN}(\text{LN}(X_{\text{inter}}))$$

MHA 内部把所有 linear projection 全换成 Pattention（Eq 7-9）：

$$Q = \text{Pattention}(X, K_P^Q, V_P^Q)$$
$$K = \text{Pattention}(X, K_P^K, V_P^K)$$
$$V = \text{Pattention}(X, K_P^V, V_P^V)$$
$$X_{\text{att}} = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$  ← 这一行还是标准 self-attention，token-token
$$O_{\text{att}} = \text{Pattention}(X_{\text{att}}, K_P^O, V_P^O)$$

参数集维度：
- $(K_P^Q, V_P^Q) \in \mathbb{R}^{n_q \times d}$，$(K_P^K, V_P^K) \in \mathbb{R}^{n_k \times d}$，$(K_P^V, V_P^V) \in \mathbb{R}^{n_v \times d}$，$(K_P^O, V_P^O) \in \mathbb{R}^{n_o \times d}$

FFN 也压缩成单个 Pattention（Eq 10）：
$$O_{\text{ffn}} = \text{Pattention}(X_{\text{ffn}}, K_P^{\text{ffn}}, V_P^{\text{ffn}})$$
$(K_P^{\text{ffn}}, V_P^{\text{ffn}}) \in \mathbb{R}^{n_{\text{ffn}} \times d}$。

**这与标准 Transformer 的对应关系**：
- 标准 FFN：两层 MLP，中间维度 $4d$，参数 $8d^2$；
- TokenFormer FFN：一个 Pattention，n_ffn ≈ 4d，参数 $2 \cdot n_{\text{ffn}} \cdot d = 8d^2$，量级一致。

这意味着可以从预训练 Transformer 直接初始化一个等大 TokenFormer（Geva et al. 2021 [https://arxiv.org/abs/2012.14913] 和 Sukhbaatar et al. 2020 [https://arxiv.org/abs/1907.01470] 已经证明 FFN 可以视为 key-value memory），把 $W_1$ 当作 $K_P$，$W_2^\top$ 当作 $V_P$，soft attention 替代 hard ReLU+matmul。

### 3.2 LayerNorm 也改成 non-parametric

Table 5：去掉 LN 的 learnable $\gamma, \beta$ 后 acc 几乎不变（82.5 → 82.6 → 82.5）。作者这么做的目的是让"唯一可学的部分就是 Pattention 里的 key-value pair"，为后续"merge two separately-trained parameter token sets"扫清障碍。这是为 future work 留的伏笔——把两个独立训练的 TokenFormer 的 KV 池 concat 起来，理论上可以直接 merge 模型，类似 model souping。

---

## 4. Progressive Model Scaling：从 124M 到 1.4B 不重训

### 4.1 Scaling 公式

Eq 11-12：

$$K_P^{\text{scale}} = [K_P^{\text{old}}, K_P^{\text{new}}]$$
$$V_P^{\text{scale}} = [V_P^{\text{old}}, V_P^{\text{new}}]$$
$$O = \text{Pattention}(X, K_P^{\text{scale}}, V_P^{\text{scale}})$$

$[\cdot, \cdot]$ 是沿 token 维度 concat，$K_P^{\text{old}} \in \mathbb{R}^{n \times d}$，$K_P^{\text{new}} \in \mathbb{R}^{m \times d}$，扩到 $\mathbb{R}^{(n+m) \times d}$。

**初始化策略**：$K_P^{\text{new}} = 0$（保持输出不变），$V_P^{\text{new}}$ random（保证一旦 keys 开始学就有梯度信号）。这正好对应 LoRA 的 $A$ random, $B = 0$ 的镜像（这里 roles 互换）。

### 4.2 FLOPs 与 scaling cost（Table 3，Figure 5）

Transformer 与 TokenFormer 的参数/FLOPs 对照：

| 模块 | Transformer 参数 | TokenFormer 参数 |
|---|---|---|
| QKV proj | $3 n_{\text{layer}} d^2$ | $2 n_{\text{layer}} d (n_q + n_k + n_v)$ |
| FFN | $8 n_{\text{layer}} d^2$ | $2 n_{\text{layer}} d \cdot n_{\text{ffn}}$ |
| 总非 embed | $12 n_{\text{layer}} d^2$ | $2 n_{\text{layer}} d (n_q+n_k+n_v+n_o+n_{\text{ffn}})$ |

FLOPs：
- Transformer：$2NT + 4 n_{\text{layer}} d T^2$（token-token 是 $T^2$ 项）；
- TokenFormer：$2NT + 4 n_{\text{layer}} d_{\text{token}} T^2$。

**关键差异**：Transformer 里 token-token 项的系数是 $d_{\text{model}}$，scale 时 $d_{\text{model}}$ 上升，长 context 时代价 quadratic 增长；TokenFormer 里 token-token 项系数是 $d_{\text{token}}$，**$d_{\text{token}}$ 在 scaling 时不变**，只增 parameter axis。这就是 Figure 5 里 sequence length 越长 TokenFormer 优势越大的原因。

对 Chain-of-Thought [Wei et al. 2022, https://arxiv.org/abs/2201.11903] 这种需要 long context 的场景，这是结构性优势——传统 scaling 会同时增加 FFN/QKV 的 quadratic 开销和 attention 的 quadratic 开销，TokenFormer 把"加宽模型"和"加长 context"解耦。

### 4.3 实验：渐进 scaling 的具体数字（Table 6, 7）

模型尺寸（Table 9，注意 layers 和 d_model 在 scaling 时不变，只增 n_kv）：

| 模型 | Layers | d_model | Attention KV pairs | FFN KV pairs | Params |
|---|---|---|---|---|---|
| Tokenformer-124M | 12 | 768 | 576 | 2304 | 124M |
| Tokenformer-354M | 12 | 768 | 2140 | 8560 | 354M |
| Tokenformer-757M | 12 | 768 | 4850 | 19400 | 757M |
| Tokenformer-1.4B | 12 | 768 | 8620 | 34480 | 1.4B |

注意：**layers 和 d_model 完全不变**，只增 KV pair 数量！这是与传统 scaling 最显著的区别——传统 1.4B 模型必然要加 layer 数或加宽 d_model。

OpenWebText validation perplexity（Table 7）：

| 模型 | 训练方案 | 训练 tokens | PPL |
|---|---|---|---|
| 124M | from scratch | 60B | 16.41 |
| 124M | from scratch | 300B | 17.06 |
| 354M | reuse 124M | 15B | 14.58 |
| 354M | reuse 124M | 30B | 14.02 |
| 354M | reuse 124M | 60B | 13.59 |
| 354M | from scratch | 300B | 13.02 |
| 757M | reuse 354M | 15B | 13.08 |
| 757M | reuse 354M | 30B | 12.59 |
| 1.4B | reuse 757M | 15B | 12.14 |
| 1.4B | reuse 757M | 30B | 11.77 |
| 1.4B | reuse 757M | 60B | 11.60 |
| 1.4B | from scratch | 300B | 11.63 |

**核心结论**：1.4B TokenFormer 用 60B tokens 训出来比 from-scratch 300B 略好（11.60 vs 11.63）；30B 已达 11.77，几乎打平；累计训练成本（300B+30B+30B+30B=390B）远小于 "三次 scratch 300B × 4 个尺寸" = 1.2T。Figure 3 显示 TokenFormer 在 1.4B 上累计成本约为 Transformer 的 1/3。

每个 scaling step 只需 30B tokens，相当于 chinchilla 1/10 的 compute，这是因为它在"沿用 1.4B 已学的 124M KV 池"，剩下的只是让新加的 KV pair 适应数据。

### 4.4 与 Net2Net / HyperCloning 对比（Table 8, Figure 7）

Net2Net [Chen et al. 2015, https://arxiv.org/abs/1511.05641] 的 width expansion：
$$W_l^{\text{new}} = \begin{bmatrix} W_s^{\text{old}} & W_{l(12)}^{\text{new}} \\ W_{l(21)}^{\text{new}} & W_{l(22)}^{\text{new}} \end{bmatrix}$$

把小模型权重放进大矩阵左上角，其余部分新初始化。问题：新初始化部分会**立刻改变输出分布**，预训练知识被打断。

HyperCloning [Samragh et al. 2024, https://arxiv.org/abs/2409.12903] 用更聪明的复制方案但本质还是 duplication。

Table 8：
- Net2Net 在 757M 上 PPL 12.1，FLOPs 10.1
- HyperCloning 12.0，FLOPs 10.1
- TokenFormer 11.5，FLOPs **3.6**（因为 $d_{\text{token}}$ 不增）

FLOPs 差距巨大：传统 scaling 把 $d_{\text{model}}$ 翻倍，token-token 的 $T^2$ 项 FLOPs 也翻倍；TokenFormer 不增 $d_{\text{token}}$，所以 attention 的 quadratic 项不变，参数扩张纯粹发生在 token-parameter 这一项。

---

## 5. Benchmark：与 from-scratch 同等表达力

### 5.1 Language Modeling（Table 1, 10）

Pile 上训练 300B tokens，与 Pythia [Biderman et al. 2023, https://arxiv.org/abs/2304.01373]、GPT-Neo [Black et al. 2021]、OPT [Zhang et al. 2022, https://arxiv.org/abs/2205.01068]、Mamba [Gu & Dao 2023, https://arxiv.org/abs/2312.00752]、RWKV [Peng et al. 2023, https://arxiv.org/abs/2305.13048] 对比：

| 模型 | Pile PPL | Avg acc |
|---|---|---|
| Pythia-160M | 29.64 | 40.1 |
| **Ours-150M** | 10.45 | 44.7 |
| Pythia-410M | 9.95 | 48.2 |
| **Ours-450M** | 8.28 | 52.0 |
| Pythia-1B | 7.82 | 51.9 |
| **Ours-900M** | 7.38 | 56.4 |
| Pythia-1.3B | 7.51 | 55.2 |
| **Ours-1.5B** | 6.91 | 59.3 |
| Mamba-1.4B | 6.80 | 59.7 |
| RWKV-1.5B | 7.70 | 54.3 |

TokenFormer 在每个尺寸上略优于 Pythia，与 Mamba 接近。**这证明 Pattention 作为 token-parameter 计算单元表达力不弱于 linear projection**——这其实有点 surprising，因为 Pattention 的 routing 是 soft 的，理论上比 hard linear 信息损失更多。

直觉解释：当 n_kv ≥ d_model 时，Pattention 的 effective rank ≥ linear layer；attention 的 soft routing 在数据丰富时反而提供了 conditional computation 的能力，每个 input token 可以"选择"该激活哪些 parameter token，类似 sparse MoE。

### 5.2 Vision（Table 2）

ImageNet-1K，沿用 MAE [He et al. 2022, https://arxiv.org/abs/2111.06377] 的训练 recipe：

| 模型 | #Params | Top-1 |
|---|---|---|
| ViT-B/16 (MAE) | 86M | 82.3 |
| Ours-B/16† | 86M | 82.1 |
| Ours-B/16 | 109M | 82.5 |
| ViT-L/16 (MAE) | 307M | 82.6 |
| Ours-L/16† | 307M | 83.0 |
| Ours-L/16 | 407M | 83.1 |

token-parameter attention 在 vision 上也对得上 ViT，证明这不是 NLP 特有的 trick。Ours-L/16 还超过 ViT-L/16 0.4 分，说明 Pattention 在容量更高时 expression 更优。

---

## 6. 一些值得思考的延伸

### 6.1 与 MoE 的关系（作者 future work §5）

把 Pattention 的每个 key-value pair 看作一个 expert，那它就是 **soft-routed MoE with top-n-all experts**。传统 MoE [Fedus et al. 2022, https://arxiv.org/abs/2101.03961] 用 top-1 或 top-2 router 节省 FLOPs，TokenFormer 默认全 expert 都参与，所以训练 FLOPs = 2N（参数总量），与 dense transformer 同级。但 attention 的 sparsity 天然存在（score 分布是 L2 norm + GeLU，非零 expert 比例可调）——未来引入 top-k routing 就能把 inference FLOPs 降下来。

更激进的想法：**MoE 的 expert 数量可以渐进增长**。训完一个 100-expert 模型后，加 50 个新 expert，zero-init keys，原有 expert 不动——这正是 [Zhu et al. 2024, MoE Jetpack] 想做的事，TokenFormer 给了原生的实现路径。

### 6.2 Parameter-efficient tuning 的天然形态

§5 提到：新任务来了，往 Pattention 池里加新的 KV pair，预训练 KV 全部 freeze。这本质是 LoRA 的"任意宽度版本"——LoRA 限定在低秩，TokenFormer 可以加任意 n 个新 pair，是 rank-agnostic adapter。比 Adapter [Houlsby et al. 2019] 和 Prefix Tuning [Li & Liang 2021] 都更结构化。

### 6.3 Vision-Language 整合

§5 的 vision-language 想法很有意思：把预训练 vision TokenFormer 的 KV 池和预训练 language TokenFormer 的 KV 池 concat，再加一组新 KV 做 alignment。这绕开了传统 VLM [Liu et al. 2023 LLaVA, https://arxiv.org/abs/2304.08485] 的 projection layer 设计——projection layer 是个"接口瓶颈"，而 concat KV 池让两个模态的"知识"都保留原样，alignment 通过新加的 KV pair 完成。

这其实和 Retrieval-Augmented Generation 有结构相似性——KV pair 就是"可检索的知识库"，input token 通过 attention 检索。如果把 Pattention 拆成"freeze 的 retrieved memory KV"和"learnable parametric KV"两组，那就是 parametric + non-parametric memory 的统一框架。

### 6.4 与 Mamba/RWKV 的关系（§G.1）

作者自己提到："This incremental memory expansion has the potential to mitigate excessive information loss, a common limitation in models such as Mamba and RWKV."

Mamba [https://arxiv.org/abs/2312.00752] 的 hidden state $h_t$ 是固定大小的，信息必须经过 selection gate 压缩，长 sequence 信息丢失。如果用 Pattention 替代 SSM 的 hidden state transition——把 $h_t$ 表示为一组 KV pair，可以通过 append 新 KV 扩展 memory——就解决了状态机容量受限的问题。这是 tokenizing state space models 的思路。

### 6.5 Interpretability（§G.2）

Geva et al. 2021 [https://arxiv.org/abs/2012.14913] 证明 FFN 是 key-value memory：第一层是 key（pattern），第二层是 value（output），ReLU 是 hard router。TokenFormer 把这个观察提升为架构 first-class citizen——Pattention 的 attention map 直接可视化为"input token 在检索哪些 parameter"，FFN 的可解释性变成 model 本身的属性，而不是 post-hoc 解释。

### 6.6 Limitation（§G.3）

Token-parameter 计算 $O(N)$ 随参数线性增长，inference 时 dense 计算开销大。比 dense transformer 慢一些（每个 Pattention 是 $T \times n \times d$ 的 matmul + softmax，相比 linear 的 $T \times d^2$ 多一个 softmax 和一个 transpose）。所以作者说必须接 MoE sparsity 才实用。

我自己的疑问：Pattention 的 forward 是 $\Theta(X K_P^\top) V_P$，两次 matmul。第一次是 $T \times n \times d$（query @ key^T），第二次是 $T \times n \times d$（score @ value），相比 linear 的 $T \times d \times d$，当 $n > d$ 时（scale 后必然如此）FLOPs 更高。所以 TokenFormer 的 inference cost 比 same-size dense transformer 高约 $n/d$ 倍。这意味着虽然训练成本省，但部署成本可能反而更高——除非用 sparse routing。

---

## 7. 我对这篇 paper 的整体判断

**真正的贡献**：把 model parameter 也 tokenize 是一个深刻的 conceptual shift。它把"模型尺寸"从一个固定架构参数变成了"知识库大小"，可以像 retrieval 一样扩展。这与 retrieval-augmented LM、MoE、memory-augmented networks [Graves et al. 2014 Neural Turing Machines, https://arxiv.org/abs/1410.5401]、甚至 liquid networks 的思路在数学上接通了——所有这些 model 都试图把"参数"从 monolithic weight 碎片化为 addressable units，TokenFormer 提供了一个用 attention 统一它们的语言。

**真正的弱点**：
1. Inference FLOPs 比 dense transformer 高（每个 linear 变成两次 matmul + softmax）；
2. Pattention 的 $\Theta$（L2 + GeLU）是工程性 patch，理论基础不深，可能在不同规模上需要重新调；
3. 没在更大尺寸（>1.4B）上验证，scaling law 形态未知；
4. Self-attention 还是 dense 的，没解决 token-token 的 $T^2$ 问题（这正是 Mamba/S4 想解决的），所以严格说 TokenFormer 是把 token-parameter 解耦，没把 token-token 解耦。

**值得继续追的方向**：
- Pattention + sparse routing → 真 MoE 原生扩展；
- Pattention 作为 RNN 的 memory cell → 解决 Mamba hidden state 限制；
- 多个独立训练模型的 KV 池 concat → 模型 merge / souping 的新范式；
- LN 改 non-parametric 是为 merge 留的伏笔，但作者没展开。

Reference 论文 list（方便后续 follow）：
- TokenFormer: https://arxiv.org/abs/2410.23168
- Code: https://github.com/Haiyang-W/TokenFormer
- Net2Net: https://arxiv.org/abs/1511.05641
- HyperCloning: https://arxiv.org/abs/2409.12903
- LoRA: https://arxiv.org/abs/2106.09685
- GeLU: https://arxiv.org/abs/1606.08415
- Geva et al. FFN as memory: https://arxiv.org/abs/2012.14913
- Sukhbaatar persistent memory: https://arxiv.org/abs/1907.01470
- Pythia: https://arxiv.org/abs/2304.01373
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- ViT: https://arxiv.org/abs/2010.11929
- MAE: https://arxiv.org/abs/2111.06377
- Switch Transformer (MoE): https://arxiv.org/abs/2101.03961
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- nanoGPT (Karpathy): https://github.com/karpathy/nanoGPT
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- OPT: https://arxiv.org/abs/2205.01068
- LayerNorm: https://arxiv.org/abs/1607.06450
- RMSNorm: https://arxiv.org/abs/1910.07467

如果你想，我可以接下来专门分析 Pattention 与 Linear layer 在 expressiveness 上的等价性（什么条件下 Pattention 退化为 linear）、或者把它与 Retrieval-Augmented Generation 的 memory bank 在数学上做严格对偶、或者推一下 sparse routing 版本 TokenFormer 应该怎么设计。
