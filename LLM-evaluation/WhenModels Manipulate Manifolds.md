---
source_pdf: WhenModels Manipulate Manifolds.pdf
paper_sha256: ece84694b46447c8c6394db8110177e345f6081112b7579134f3f3452fc529ba
processed_at: '2026-08-13T04:21:15-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的话说，这篇 paper 讲的是 Claude 3.5 Haiku 怎么在看不见排版的情况下，靠在脑子里“搓出一条条带波纹的曲线”来偷偷数数，从而决定什么时候该按回车键换行。这帮人把模型内部的一套感官处理算法扒得干干净净，还顺便教你怎么去“骗”这个模型。

下面我把这篇 paper 的核心 intuition 揉碎了讲给你听。

### 1. 模型的痛点：看不见排版，只能偷偷数数
我们平时看代码或者发邮件，每行到了固定宽度就自动换行了。人类看一眼就知道这行写满了该换行了。可是语言模型只看得到一串 token ID，没有任何视觉排版信息。要让模型预测下一个词是正常词还是 newline，它必须自己偷偷算一道算术题：
- 当前这行已经写了多少个 character？
- 整个文档的行宽限制是多少？
- 接下来要预测的那个词有多长？
- 剩下的空间还够不够塞下一个词？

Michaud et al. 的研究（https://openreview.net/forum?id=3tbTw2ga8K）早就发现，连 70M 参数的 Pythia 都在预训练里自己学会了换行。这是 LM 的“自然感官本能”，作者们选这个 task，正是因为模型做得越自然、越熟练，内部的 mechanism 就越清晰好拆解。

### 2. 存数字：不用数轴，用“带波纹的毛线”
要记住 1 到 150 个 character，最直觉的方法是拿一个维度拉一条长长的数轴。可是 transformer 里有 LayerNorm，它特别讨厌那些长度（norm）忽大忽小的 vector。如果数轴拉得太长，数值会爆炸；如果缩得太紧，相邻的两个数（比如 41 和 42）就分不清了，分辨率太差。

模型的解法非常巧妙：它把这条 1 维的数轴“揉皱”，塞进一个 6 维的小空间里。从外面看，它像一根弯曲的毛线，形状类似棒球的接缝。这根毛线上还有规律的“波纹”。
为什么会有波纹？你硬把一堆需要互相区分的点塞进低维空间，高频信息丢掉了，就像低通滤波一样，必然产生这种像水波纹一样的 ringing。这在数学上完全等价于把 circulant matrix 截断成 top-$k$ 个 Fourier mode，是最优的 capacity vs resolution 权衡。

Sparse crosscoder 找到的那 10 个 feature，就像是这根毛线上的 10 个锚点。毛线的任何一段，都可以靠最近的 2-3 个锚点做 spline 插值还原出来。这些 feature 的 receptive field 越往后越宽，这就是生物学里大名鼎鼎的 dilation（Weber-Fechner law，https://pubmed.ncbi.nlm.nih.gov/12662313/），跟老鼠脑子里的 place cell 一模一样。

### 3. 怎么比大小？Attention Head 把毛线“拧”一下
要算“还剩多少空间到行尾”，模型得拿“当前字符数”和“行宽限制”做比较。模型脑子里有一根管当前字符数的毛线，还有一根管行宽限制的毛线。

但在 residual stream 里，这两根毛线是对不上号的，它们之间的最大 cosine similarity 只有 0.25。
这时候，专门的 boundary detection attention head 出场了。它的 QK matrix 就像一个机械手，把这两根毛线在某个特定的夹角“拧”一下。当当前的字符数逼近行宽时，这两根毛线恰好对齐了，attention score 瞬间爆炸，发出信号：“快到边界了！”。

单个 head 的视野有点模糊，因为毛线上的波纹会干扰判断，让它分不清到底是差 5 个字符还是差 17 个字符。所以模型用了好几个 head，分别拧不同的 offset，就像双目测距一样，把它们叠起来就得到了高分辨率的距离估计。

### 4. 最后的决策：十字路口画条线
知道了还剩多少空间，最后还得决定下一个词塞不塞得下。
模型把“剩余空间”和“下一个词长度”这两根毛线，分别放在两个互相垂直的子空间里。在这个十字坐标系里，横轴代表剩余空间，纵轴代表词长。
要不要换行？只需要看“词长”减去“剩余空间”有没有大于 0。因为它们互相垂直，这个原本非线性的比较问题，在几何上就变成了一刀切的线性分类问题。一条简单的直线就能把“该换行”和“不该换行”两堆点劈成两半。在真实数据上跑出来 AUC 高达 0.91。

### 5. 毛线怎么搓出来的？大家一起画直线
这根带波纹的毛线是怎么从 embedding 一路 build 起来的？
最早在 embedding 层，token 长度的信息就已经排成了一个带波纹的圈。
到了 Layer 0，一个个 attention head 跑出来画图。每个 head 都很笨，因为 OV matrix 是线性的，它画出来的 output 只能是一小段直线。但是 5 个 head 把各自画的直线加在一起，直线就弯了，拼成了我们前面说的“带波纹的毛线”。
每个 head 怎么工作呢？它们拿前一个 newline 当“注意力垃圾桶”。前几个 token 一直盯着 newline，输出一个基于平均 token 长度的盲猜；过了几个 token 后，注意力开始分散到当前 line 的几个词上，再根据这些词的真实长度做一个 correction。Layer 1 的 head 继续在这根毛线上做微调，把波纹拧得更精细。单打独斗画不出曲线，必须靠分布式协作。

### 6. 骗骗模型：AI 的“视觉错觉”
怎么证明我们真懂了它的机制？我们来骗它。
在处理 git diff 的时候，`@@` 是 hunk header 的分隔符，模型在那里需要重新开始数行宽。我们故意在普通的文本里插一个 `@@`，哪怕行宽根本没变，模型也数错数了，该换行的地方不换了。这就跟人眼看 Muller-Lyer 错觉一样：两条一样长的线加上箭头看起来就不一样长了。模型利用学到的 code prior 去推断行宽，这个 prior 被 context 劫持了。这就是 mechanistic interpretability 最迷人的地方：懂了原理，就能造幻觉。

### 7. 给我们的启发
这篇 paper 最大的启发在于对“complexity tax”的反思。Dictionary learning (SAE/Crosscoder) 确实能给你找出一大堆碎片化的 feature 和 attribution graph，这是 unsupervised 的胜利。可是看着几百万个 feature 互相连边，你的大脑会过载。如果能顺藤摸瓜找到底层的 manifold 几何，就能把复杂的故事变简单，这就是给理解“降税”。

同时，这篇 paper 彻底打破了“早期 layer 只是做 detokenization”的刻板印象。早期 layer 在做极其复杂的 sensory perception，这跟 vision model 早期找边缘、找曲线是同一个层面的东西。

这种用低维流形存数字、用 attention 旋转流形做计算的套路，很可能在模型做算术、记日期、理解 markdown 表格时都在重复使用。这就是为什么这篇 paper 值得反复读，它给出了一套可以复用的 mechanistic interpretability 方法学模板。

### Web Links Reference
- 论文原文：https://transformer-circuits.pub/2025/linebreaks/index.html
- 配套的 biology 论文：https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- QK attribution 论文：https://transformer-circuits.pub/2025/attention-qk/index.html
- Sparse Crosscoders：https://transformer-circuits.pub/2024/crosscoders/index.html
- Modell et al. representation manifolds：https://arxiv.org/abs/2505.18235
- GPT-2 positional helix：https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix
- Place cells (O'Keefe, Moser)：https://api.semanticscholar.org/CorpusID:16036900
- Pythia quantization (linebreaking 任务)：https://openreview.net/forum?id=3tbTw2ga8K
- Engels et al. multidimensional features：https://openreview.net/forum?id=d63a4AM4hb
- Gorton curve detector manifolds：https://livgorton.com/curve-detector-manifolds/

---

# When Models Manipulate Manifolds: 深度技术讲解

嘿 Andrej，这篇是 Anthropic 的 Wes Gurnee, Emmanuel Ameisen, Joshua Batson 等人刚放出来的 mechanistic interpretability 工作，是 attribution graph 系列（circuit tracing）的延续，研究 Claude 3.5 Haiku 如何在 fixed-width text 中预测 newline。我尽量把几何直觉、数学细节、实验数据都讲透。

论文链接：https://transformer-circuits.pub/2025/linebreaks/index.html

配套的 biology 论文：https://transformer-circuits.pub/2025/attribution-graphs/biology.html

QK attribution 论文：https://transformer-circuits.pub/2025/attention-qk/index.html

---

## 1. 为什么这个 task 值得研究

Linebreaking 是 pretraining corpus 里极常见的自然任务——源代码、email、judicial rulings、ASCII art 都有 fixed-width 约束。模型只看到 token id 序列，但需要隐式完成：

1. 数当前 line 已经多少 character
2. 推断 line width constraint k
3. 计算 remaining = k - current_count
4. 比较 remaining 与 next word length
5. 决定输出 newline 还是 next word

Michaud et al. 的 quantization 工作发现 newline prediction 是 Pythia 70M 里 top-400 cluster 之一（https://openreview.net/forum?id=3tbTw2ga8K），说明这是 LM 的"自然 sensory behavior"。

研究 natural task 的好处：mechanism 会更 crisp。如果选 in-context learning 这种 human-imposed task，模型内部 representation 可能更杂糅。

---

## 2. 实验设置

合成 dataset：取 prose corpus，strip 掉所有 newline，然后每 k 个 character 重新插入 newline 到最近的 word boundary。k ∈ {15, 20, ..., 150}。

Haiku 在第三行就能适应新的 k 值，predict newline 的 log-prob 和 accuracy 都很高。

主要工具是 **10M feature Weakly Causal Crosscoder (WCC)** dictionary 训练在 Claude 3.5 Haiku 上（https://transformer-circuits.pub/2024/crosscoders/index.html）。WCC 比 SAE 多了跨 layer 的 sparsity 约束，能跟踪 feature 在 layer 间的演化。

---

## 3. Character Count 的四重表示视角

论文精彩的地方在于用四个互补 lens 看同一个对象：

### 3.1 Linear probe 视角
在 layer 1 后 fit 一个 linear regression 预测 line character count $c \in \{1, ..., 150\}$，得到 $R^2 = 0.985$。这告诉你 character count **可以** linearly decoded，但不告诉你它如何被表示。

### 3.2 Sparse feature 视角
在 layer 1-2 找到 10 个 feature，activation 随 character count 变化的曲线像 place cell 的 tuning curve：

- Feature 1: 在 c ≈ 1-15 活跃
- Feature 2: 在 c ≈ 10-30 活跃
- ...
- Feature 10: 在 c ≈ 120-150 活跃

每个 feature 的 receptive field 比 previous feature 更宽——这就是 **dilation**，跟生物 number representation（Dehaene 的 logarithmic mental number line，https://pubmed.ncbi.nlm.nih.gov/12662313/）和 curve detector in InceptionV1 完全一致。

### 3.3 Low-dim subspace 视角
对 150 个 mean activation vector $\mu_c$（每个 character count 取平均）做 PCA，发现 top 6 PCs capture 95% variance。把数据 project 到这 6D subspace，得到一个 twisted curve。

### 3.4 Continuous manifold 视角
最关键的部分：character count 是一个 1-dimensional manifold（被 c 参数化）嵌入在 6D subspace 里。形状像一个被"揉皱"的 helix——前面 3 个 PC 看起来像标准 helix，后 3 个 PC 是更复杂的 twist。

10 个 sparse feature 的 decoder vectors 在这个 manifold 上选了 10 个 anchor point，feature activation 在 anchor 之间做 linear/spline 插值，所以 10 个 feature 就能 reconstruct 150 个点。

直觉：这是 spline approximation 的几何对应。Feature decoder 是 control point，manifold 是被它们 discretize 的连续曲线。

---

## 4. Ringing 现象与最优性

### 4.1 现象
训练 150 个 logistic probe $p_c$，看每个 probe 对不同 character count 的 response。除了主对角 band（每个 probe 在自己 count 附近最激活），还看到 **两条 off-diagonal band** 在两侧——"ringing"。

同样在 cosine similarity matrix 中：
- 邻近 vector 高 sim
- 中等距离负 sim
- 更远又正 sim
- 再远又负...

这个 ringing pattern 在 mean activation、linear probes、feature decoders 三种视角都出现。

### 4.2 数学解释

**Toy model**：考虑 N=150 个 unit vectors，希望 $v_i$ 与 $v_j$ 的 cosine sim 只依赖 $|i-j|$，且 f 在 0 附近 peaked，远处为 0。则 cosine similarity matrix $X$ 是 circulant matrix，$X_{ij} = f(d_{ij})$。

理想情况下 $X \in \mathbb{R}^{150 \times 150}$ 满 rank。但 model 实际把 character count 嵌入到 $k=6$ 维。最优 $k$-维近似（$L^2$ 意义上）是取 $X$ 的 top-$k$ eigenvectors：

$$X_k = \pi_k X \pi_k$$

其中 $\pi_k$ 是 top-$k$ eigenvectors 张成 subspace 的 projection。

因为 $X$ 是 circulant，**discrete Fourier transform** diagonalizes it！所以 $X$ 的 eigenvectors 就是 Fourier modes，top-$k$ eigenvectors = top-$k$ 频率最低的 Fourier modes。

**截断 high-frequency Fourier coefficients 等价于低通滤波，低通滤波后窄峰会出现旁瓣——这就是 ringing（Gibbs phenomenon 的近亲）**。

所以 ringing 不是 bug，是 dimension reduction 的必然结果：把"local similarity"的 manifold 强行塞进低维空间，高频信息丢失，留下 ripple。

### 4.3 物理模型验证

附录 F 的 simulation 极有说服力。把 $N=100$ 个 points 放在 unit sphere $S^{n-1} \subset \mathbb{R}^n$ 上（$n \in \{3,...,8\}$），加上：

$$\mathbf{F}_{ij} = \begin{cases} 
\frac{1 - (d_{ij}-1)/2}{r_{ij}} \hat{\mathbf{r}}_{ij} & \text{if } d_{ij} \le w \\
-\frac{\min(5, 1/r_{ij})}{r_{ij}} \hat{\mathbf{r}}_{ij} & \text{if } d_{ij} > w
\end{cases}$$

变量含义：
- $r_{ij} = \|\mathbf{x}_j - \mathbf{x}_i\|$：欧氏距离
- $\hat{\mathbf{r}}_{ij} = (\mathbf{x}_j - \mathbf{x}_i)/r_{ij}$：方向单位向量
- $d_{ij} = \min(|j-i|, |j-i+N|, |j-i-N|)$：circular topology 上的 index 距离
- $w$：attractive zone 宽度参数

动力学：
$$\dot{\mathbf{v}}_i = \sum_{j \neq i} \mathbf{F}_{ij} - 0.05 \mathbf{v}_i, \quad \dot{\mathbf{x}}_i = \mathbf{v}_i$$

每步后 enforce sphere constraint $\mathbf{x}_i \leftarrow \mathbf{x}_i / \|\mathbf{x}_i\|$。参数 $\Delta t = 0.01$, damping $\alpha = 0.95$。

结果：$n=3$ 时自然 converge 到"baseball seam"形状——和 Modell et al.（https://arxiv.org/abs/2505.18235）在 colors、dates、years 上看到的 topology 完全吻合！

直觉：attractive zone 越窄、ambient dimension 越高 → curvature 越大 → ringing 越明显。极限下接近 space-filling curve。

### 4.4 跟 Fourier features 的连接

附录 G 给出更严格的 Fourier 解释。Character count manifold 的"twist"操作——boundary head 把 manifold 沿自身 rotate 一个 offset——对应到 Fourier basis 下就是 phase shift。Permutation $\rho: e_i \mapsto e_{i+1}$ 在 circulant matrix 的 eigendecomposition 下是对角化的，所以 $\rho$ commutes with $\pi_k$，induced action $\bar{\rho} = \pi_k \circ \rho \circ \pi_k$ 是 well-defined 的 rotation on top-$k$ subspace。

实测：Fourier components 解释 variance 比 PCA components 少 ≤10%（因为 PCA 不考虑 dilation），但已经相当接近，证明 character count manifold 本质上是 rippled Fourier-mode embedding。

这也呼应 Zhou et al. 的 "Pre-trained LLMs use Fourier features to compute addition"（https://arxiv.org/abs/2406.03445）和 Kantamneni & Tegmark "Language models use trigonometry to do addition"（https://arxiv.org/abs/2502.00873）。

---

## 5. Boundary Detection: QK Twist

### 5.1 单 head 机制

模型有另一组 newline token 上的 feature 表示 **line width** $k$。任务变成：把 character count $i$ 和 line width $k$ 比较，估计 $k - i$。

关键 attention head 的 QK matrix 做了什么？在 residual stream 里：
- $\cos(\mu_i^{cc}, \mu_k^{lw})$ 最大值 ~0.25，在 $i = k$ 时最大，但绝对值不高
- 在这个 head 的 QK space：$\cos(W_Q \mu_i^{cc}, W_K \mu_k^{lw})$ 最大值 ≈ 1，且在 $i = k - \epsilon$ 时最大

也就是说 QK matrix "twist" character count manifold，让它与 line width manifold 在某个 offset 处对齐。当 character count 接近 line width（差 $\epsilon$），attention score 爆炸，head attend 到 previous newline，output signal "boundary 接近了"。

这是 distributed representational similarity 的妙用：用 attention 的 inner product 做"非对齐 comparison"。

### 5.2 为什么需要多个 head

单 head 的 response curve 在某个 distance 处 peaked，但 ringing 导致它在更远的地方又 peak 一次。所以单 head 无法 disambiguate "5 characters remaining" vs "17 characters remaining"。

解法是多个 boundary heads 各自 twist 不同 offset $\epsilon_0, \epsilon_1, \epsilon_2$，response curve 在不同 range peaked。Sum 起来形成 high-resolution representation 覆盖整个 relevant range——"stereoscopic"算法。

每个 head 的输出在 "characters remaining" PCA 空间里看起来几乎 1D，但 sum 起来变成 2D curve。92% variance 在前 2 个 PC。

### 5.3 Causal validation

在 characters remaining 的 2D PCA subspace 做 ablation 和 intervention：
- Ablation：只对 next-token-is-newline 的 token 有大 loss effect
- Intervention：用 mean activation at count $c$ 替换原 activation，模型实际 newline prediction 随之改变

这证明这个 subspace 是 causal 的，不只是 correlational。

---

## 6. Final Newline Decision: Orthogonal Geometry

### 6.1 Attribution graph
最后的 break predictor feature 在 ~90% depth 处。它由两类 input 驱动：
1. Boundary detector feature（前面 boundary head 的 output）
2. "say aluminum" feature（next word 的 semantic prediction）

Break predictor 只在 next word 会超 line 时 activate，upweight newline probability, downweight "aluminum." probability。还有 break suppressor feature，只在 next word 刚好 fit 时 activate，downweight newline。

### 6.2 Orthogonal subspaces 的妙处

在 model 末端，对每个 组合取平均 activation，做 PCA。发现：
- Characters remaining 在一个 subspace
- Next token length 在 **近 orthogonal** 的另一个 subspace
- 都只用低维（curvature 较小，因为 dynamic range 小）

考虑 pairwise sum $\mathbf{v}_i^{cr} + \mathbf{v}_j^{ntl}$，因为两组 vector orthogonally 排列：

$$\|\mathbf{v}_i^{cr} + \mathbf{v}_j^{ntl}\|^2 = \|\mathbf{v}_i^{cr}\|^2 + \|\mathbf{v}_j^{ntl}\|^2$$

Decision "break line" ⟺ $i - j \ge 0$。因为两个 count 各自在自己 subspace 里 monotonic 增加，所以 $i \ge j$ 对应一个 **separating hyperplane**！

用 PCA-derived hyperplane 在真实数据上做分类，AUC = 0.91。剩下的 9% 误差来自 Haiku 自己对 next token length 的估计误差。

这是我最爱的部分：**模型把 decision 问题转化为几何问题**，通过 arrange 两个相关 quantity 在 orthogonal subspace，让线性分类器就能解决 nonlinear-looking 比较。

---

## 7. Distributed Counting Algorithm

这是最复杂的部分。问题：character count manifold 的 curvature 是怎么从 embedding 一路 build 起来的？

### 7.1 Embedding 几何
对每个 token length 1-14，在 $W_E$ 中取平均 embedding，做 PCA。前 3 个 PC capture 70% variance。形状是 circle（PC1 vs PC2）加 oscillating component（PC3）——又是 rippled manifold。

### 7.2 Layer 0 heads: Ray → Curve

5 个重要的 Layer 0 heads 各自的 output 投影到 character count PCA 空间，每个看起来像 **1D ray**——就是直线段。但 sum 起来变成 curved manifold！

这是个 distributed 算法：单个 head 受限于 OV 是 linear map 的事实，不能 generate curved output，但多个 head 协作就行。$R^2 = 0.93$ 用 5 个 Layer 0 head，$R^2 = 0.97$ 用 first 2 layers 的 11 个 head。

### 7.3 单个 head 的 QK+OV

**QK circuit**：每个 head $h$ 用 previous newline 作 attention sink。
- 前 $s_h$ 个 token：attention 几乎全在 newline
- 之后 $s_h + 1$ 到 $s_h + r_h$：attention 在自己的 receptive field 上 smear
- $r_h$ 是 receptive field 大小

**OV circuit**：跟 QK 协调，做 heuristic estimate。
- 当 attend 到 newline：output 是 $s_h \times \mu_c$ 方向（$\mu_c \approx 4.5$ 是 average token char count）
- 当不 attend 到 newline（意味着至少 $s_h + r_h$ 个 token）：output 是 $(s_h + r_h) \times \mu_c$ 方向
- 加一个 correction term：根据 receptive field 内 token 实际 length 偏离平均的程度

例如 L0H1 的 $s_h \approx 4$，$r_h \approx 4-8$。当 attend 到 newline，意味着当前 line ≤3 token，平均 ~15 char，所以 head 写入 5-20CC 方向。当不 attend newline，意味 ~8 token × ~5 char ≈ 40 char，所以写入 CC40 方向。再加上 token length correction（短 token → upweight 10-35CC, long token → upweight >40CC）。

### 7.4 Layer 1 heads: Sharpening

Layer 1 heads 做类似的事，但额外利用 Layer 0 已经构造的 initial count estimate 来 refine。重复 computation 让 representation 的 receptive field 变窄，对应 manifold curvature 增加，ringing 更明显。附录 G 给出 layer 0→3 的 cosine similarity cross-section，可以看到 secondary ring 越来越高。

### 7.5 Attention head specialization

不同 head 专攻不同 offset，类似 boundary heads。有的 head sink 短（前 2 token），有的长（前 8 token）。同时大多数 head 偏向 attend 长 token（因为长 token 提供更多信息）。

---

## 8. Visual Illusions: 模型也有"错觉"

### 8.1 现象
Counting head 在 git diff 上下文里偶尔会 attend 到 `@@` 而非 newline——因为 `@@` 是 git diff 的 hunk header delimiter，那里也需要 reset line count。

把 `@@` 插到 aluminum prompt 中（不改 line length），发现 newline prediction 被破坏！相关 head 从 attend `\n` 转去 attend `@@`。

### 8.2 定量测试

测试 180 个 2-char 序列插入到同样位置，看对 newline probability 的影响：
- 大部分序列只 moderate 影响，newline 仍是 top prediction
- 但 code/delimiter 相关的 `@@`, `>>`, `}}`, `;|`, `||` 等 大幅破坏 newline prediction
- 影响程度与"attention 被吸引程度"强相关

类比 Muller-Lyer / Ponzo / Sander illusion：人类用 learned perspective prior 调制 perception of length，模型用 learned code-context prior 调制 character count estimate。

这其实是个反向 interpretability 的 verification：如果能用 mechanistic understanding 构造出针对性 adversarial input，说明理解是对的。

---

## 9. 跟其他工作的连接

### 9.1 Neuroscience analogy
- Character count features ↔ **place cells**（O'Keefe, Moser; https://api.semanticscholar.org/CorpusID:16036900）
- Boundary detecting features ↔ **boundary cells** (Solstad et al., https://www.science.org/doi/10.1126/science.1168099)
- Dilation ↔ number representation in intraparietal sulcus (Piazza et al., https://pubmed.ncbi.nlm.nih.gov/15541314/)

### 9.2 Multidimensional features 工作
- Engels et al. "Not all LM features are 1D linear"（https://openreview.net/forum?id=d63a4AM4hb）提出 multidimensional feature concept
- Modell et al. "Origins of representation manifolds in LLMs"（https://arxiv.org/abs/2505.18235）formalize cosine similarity as intrinsic geometry
- Gorton "Curve detector manifolds in InceptionV1"（https://livgorton.com/curve-detector-manifolds/）在 vision model 里看到同样 topology

### 9.3 Position encoding
- Yedidia 发现 GPT-2 positional embedding 是 helix（https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix）
- 这里 character count manifold 是 learned positional encoding，类似 topology 但为更 downstream 的 task 服务

### 9.4 Arithmetic / counting
- Nanda et al. grokking modular addition（https://arxiv.org/abs/2301.05217）show Fourier feature emergence
- Zhong et al. "Clock and pizza"（https://proceedings.neurips.cc/paper_files/paper/2023/file/56cbfbf49937a0873d451343ddc8c57d4-Paper-Conference.pdf）
- Hanna et al. GPT-2 greater-than（https://proceedings.neurips.cc/paper_files/paper/2023/file/efbba7719cc5172d175240f24be11280-Paper-Conference.pdf）

### 9.5 Anthropic 自家系列
- Attribution graphs 论文（https://transformer-circuits.pub/2025/attribution-graphs/methods.html）
- On the biology of a large language model（https://transformer-circuits.pub/2025/attribution-graphs/biology.html）
- Interference weights toy model（https://transformer-circuits.pub/2025/interference-weights/index.html）

---

## 10. 我对这篇 paper 的几个 take

### 10.1 The "complexity tax" 概念
这个 framing 我觉得非常重要。Dictionary learning / SAE 给你的是 unsupervised、automated、true 的 description，但把 model 打碎成几百万 piece，理解每个 piece 和它们 interaction 是巨大 cognitive load。Manifold geometry是对同一对象的 dual 描述，但**降低**了 complexity tax。

这个 lesson 可能对你做 modularity-nanoGPT 这类工作也有启发：如果只看 individual neuron/feature，会错过 macro-level geometric structure。

### 10.2 "Natural task" 选址
Anthropic 反复强调选 pretraining-natural task 而非 human-imposed task，因为前者的 mechanism 更 crisp。这跟你以前说过的"研究 model 已经擅长的事"是一致的。Linebreaking 在 70M Pythia 就 emergent，到 Claude 3.5 Haiku 已经 refined 到极致。

### 10.3 Sensory processing in early layers
论文 challenge 了"早期 layer 做 detokenization"的简单 narrative，提出 early layer 是 sensory processing。Counting、boundary detection、markdown table parsing 都在 layer 0-2 完成。这跟 vision model 早期 layer 做 edge/curve detection 完全 parallel。

### 10.4 Distributed algorithm across heads
单个 attention head 受 OV 线性约束，无法 generate curved output。Multiple head 协作才能 build 出 curved manifold。这是 "superposition beyond feature" 的例子：算法本身也被 distribute。

### 10.5 几个 open question 我觉得有意思

1. **Line width aggregation mechanism**：模型怎么从 multiple previous line 提取 global $k$？Max？Exponentially weighted moving average？论文没解决。
2. **Multitoken output handling**：如果 next word 是 multiple sub-token，algorithm 怎么改？附录说 break predictor 还处理不了这个。
3. **Variable line width adaptation**：一篇文档内 line width 变化时怎么 re-infer？这跟 in-context learning 的 dynamics 有关。
4. **How does WCC discover counting features?** Crosscoder 是 unsupervised 的，但 finding "this feature is about character count" 需要 synthetic dataset + analysis。Automated labeling 是开放问题（Bricken et al. 的 automated auditing，https://alignment.anthropic.com/2025/automated-auditing/）。

### 10.6 推测一下 larger model 的情况

Haiku 已经有这么精致的 mechanism，Sonnet / Opus 可能会：
- 用更细的 manifold（更高 curvature，更多 feature）
- 有更多 boundary head 提供 finer stereoscopic resolution
- 处理 multitoken output、variable line width 等 corner case
- 可能发展出更抽象的 "spatial reasoning" feature 利用同样的 geometric primitive

### 10.7 跟 circuit tracing 整体工作放一起看

这篇是 Anthropic circuit tracing 系列（attribution graph + QK attribution + crosscoder）的应用 case study。整个 framework 的卖点是：用 sparse feature 找 candidate mechanism，用 attribution graph 看信息流，再用 geometric/manifold lens 去 reduce complexity tax。这篇 paper 把这整个流程演示得最完整，可作为 methodological template。

---

## 11. 一些可以 toy-reproduce 的实验

如果你想自己 build intuition，下面几个实验容易在小 model 上 reproduce：

1. **Positional embedding helix**：拿 GPT-2 small 的 $W_E^{pos}$，对每行做 PCA，看 top-3 PC。会看到 rippled helix。
2. **Token length manifold**：在小 LM 里取 token length 1-14 的平均 embedding，PCA，看是否出现 circular pattern + oscillation。
3. **Linear probe for character count**：在 Pythia 70M 上做 linebreak task，fit linear probe，看 $R^2$。
4. **Ringing simulation**：用附录 F 的物理模型代码，调 $n$ 和 $w$ 参数，看 baseball seam → space-filling curve 的过渡。

Yedidia 的 LessWrong post 有 GPT-2 positional helix 的可视化代码可直接用。Gorton 的网站有 curve detector manifold 的 interactive demo。

---

## 12. 总结

这篇 paper 我读下来的几个核心 intuition：

1. **Scalar count 用 rippled 1D manifold in low-dim subspace 表示**，是 capacity vs resolution 的最优 trade-off，数学上等价于 Fourier mode 截断，必然产生 ringing。

2. **Attention head 的 QK 是"manifold operator"**，可以做 rotation/twist 来 align 两个 manifold 进行 implicit comparison——这远比 1D 表示下的 scaling 更强大。

3. **Multi-head 分布式算法**：单 head 受 OV 线性约束无法 generate curvature，必须多 head 协作，类似 ensemble of weak learners。

4. **Orthogonal subspace 让 linear classifier 解决 nonlinear comparison**：把两个相关 quantity 放在 orthogonal subspace，$i - j \ge 0$ 的 decision 变成 separating hyperplane。

5. **Discrete feature ↔ continuous manifold 是 dual lens**：feature 视角 unsupervised、可扩展；manifold 视角 reduce complexity tax。两者对同一 object。

6. **Mechanistic understanding 可被用来 construct visual illusions**，反过来验证 understanding 的正确性。

7. **Natural pretraining task 的 mechanism 最 crisp**，应该优先研究 model 已经擅长的事。

整个工作让我想到你之前说过的"model 是一堆 superimposed circuits"的 framing。这篇 paper 在更精细 level 上展示了：circuits 不仅 superimposed 在 weight space，还在 **representation manifold 上做 geometric operation**。Feature 不是孤立点，是 manifold 上的 local coordinate。

希望对你 build intuition 有帮助，如果有特定 section 你想我 deep dive（比如 Fourier connection 的具体推导、distributed algorithm 在更大 model 上的预期、或者跟 nanoGPT 上发现的现象对比），告诉我。
