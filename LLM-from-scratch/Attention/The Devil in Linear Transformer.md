---
source_pdf: The Devil in Linear Transformer.pdf
paper_sha256: b3b7ba6d45ba957357afa21dd983977865a13390dca12dc48fad76c27eab881b
processed_at: '2026-08-12T14:13:41-07:00'
target_folder: LLM-from-scratch/Attention
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说TRANSNORMER

## 一、这paper到底在干啥

一句话：**别人搞linear transformer想省计算，结果效果拉胯，这paper找出两个罪魁祸首，修好了，效果追平vanilla transformer。**

vanilla transformer的痛点大家都知道 — sequence length $n$ 一长，attention matrix是 $n \times n$，显存爆炸。linear transformer的思路是用kernel trick把 $n \times n$ 的attention matrix拆掉，变成和 $n$ 线性的complexity。听起来很美，但实际用起来效果差一大截。

这paper的贡献就是找出**为什么差**，然后给出fix。

---

## 二、问题一：Gradient会爆炸

### 直觉讲法

你想想softmax为啥数值稳定。softmax里有个 $\exp$，$\exp$ 这个函数有个特别好的性质：**它的导数等于它自己**，$f'(x) = f(x)$。

这意味着啥呢？当你算gradient的时候，会出现 $\frac{f'(x)}{f(x)}$ 这种ratio，对 $\exp$ 来说这个ratio恒等于1。**不管你的logit是100还是-100还是0，这个ratio永远是1**。softmax自带"数值护栏"。

linear transformer的人想偷懒，把 $\exp$ 换成identity function $f(x) = x$。这时候 $f'(x) = 1$，$\frac{f'(x)}{f(x)} = \frac{1}{x}$。

**问题来了**：$x \to 0$ 的时候 $\frac{1}{x} \to \infty$。

而 $x = \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$ 在训练中经常接近0 — 当query和key接近正交时，内积就是0附近。所以gradient爆炸是**常态**，不是edge case。

### paper的构造性证明

他们构造了个case：让所有 $\mathbf{q}_i = \mathbf{k}_j = \phi^{-1}(\mathbf{x}_0)$，其中 $\|\mathbf{x}_0\|_2 \leq \sqrt{\epsilon}$。这时所有similarity $s_{ij} = \mathbf{x}_0^\top \mathbf{x}_0 \leq \epsilon$，gradient大小 $\propto \frac{1}{\epsilon}$。$\epsilon \to 0$ 时gradient $\to \infty$。

这是**存在性证明** — 说明确实存在让gradient任意大的输入configuration，所以gradient是unbounded的。

### 经验验证

他们实际跑了50k步训练，测gradient的relative standard deviation：

- vanilla: 0.25
- Performer: 0.47  
- 1+elu: 0.58
- NORMATTENTION (他们的): 0.20

linear方法gradient波动是vanilla的2倍以上，他们的方法比vanilla还稳。

---

## 三、问题二：Attention被稀释

### 直觉讲法

vanilla transformer有个有意思的现象：**早期层的attention很"local"**，每个token主要看它周围的几个token，远处的几乎不看。这和CNN的inductive bias有点像 — 先抓local pattern，再往上层做global integration。

linear transformer丢了这个性质。你想想，linear attention的score是 $\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$，kernel $\phi$ 通常是非负的（比如1+elu），所以所有score都是正数。长序列上，远处token的score和近处token的score差不多大，attention分布就变得很"平"，均匀洒到整个sequence上。

这叫**attention dilution** — attention被稀释了，每个token都"雨露均沾"，失去了focus。

### 怎么量化的

他们定义了个metric叫"locally accumulated attention score"：以token $i$ 为中心，看周围 $r \cdot N$ 邻域内的attention score加起来多少。

vanilla transformer：40%邻域贡献~80% attention，曲线很陡
linear transformer：40%邻域只贡献~40%，曲线很平

很直观的证据 — vanilla是local-focused，linear是diluted。

### 一个关键的ablation

他们做了个controlled experiment，把12层里的一部分换成vanilla attention：

- 全用1+elu: PPL 4.98
- 前面1+elu，后面vanilla: PPL 3.90
- 前面vanilla，后面1+elu: PPL 3.76

**早期层用vanilla收益更大**。这说明早期层需要local focus，这个local focus正是linear attention丢的东西。

---

## 四、解决方案一：NORMATTENTION

### 思路

既然scaling（就是softmax分母 $\sum_k f(s_{ik})$ 那一项）是gradient爆炸的根源，那就去掉它。

但直接去掉会出事：forward pass的attention output变成unbounded了。他们试了下，PPL从4.98暴涨到797.08，模型直接崩。

所以得找个东西来"稳住"output。他们的方案：去掉scaling，但在attention output后面加个normalization。

$$\mathbf{O}_{\text{norm}} = \text{RMSNorm}(\mathbf{Q}(\mathbf{K}^\top \mathbf{V}))$$

RMSNorm公式：$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\sigma^2 + \epsilon}}$，其中 $\sigma^2 = \frac{1}{d}\sum x_i^2$。

### 为啥这能fix gradient

关键在于RMSNorm里那个 $\epsilon$（通常 $10^{-6}$）。

RMSNorm的Jacobian长这样：
$$\frac{\partial o_{ij}}{\partial t_{ik}} = \frac{1}{\sqrt{\sigma_i^2 + \epsilon}}\left[1\{j=k\} - \frac{t_{ij}t_{ik}}{d(\sigma_i^2 + \epsilon)}\right]$$

那个 $\sqrt{\sigma_i^2 + \epsilon}$ 在分母，因为 $\epsilon > 0$，所以即使 $\sigma_i^2 \to 0$，分母也有lower bound $\sqrt{\epsilon}$。

最终推出gradient bound：
$$\left|\frac{\partial \mathcal{L}}{\partial s_{ij}}\right| \leq \frac{3 c_1 c_2 d}{2\sqrt{\epsilon}} < \infty$$

这里的 $\sqrt{\epsilon}$ 提供了gradient的有界性保证。$\epsilon$ 就像是个"数值安全垫"。

### 直觉总结

softmax用 $\exp$ 的"自正则化"来稳gradient，代价是quadratic complexity。
linear attention去掉了 $\exp$，失去了这个保护。
NORMATTENTION用RMSNorm的 $\epsilon$ 重新建立保护，complexity保持linear。

**本质上是把softmax的隐式正则化，换成了显式的normalization操作**。

---

## 五、解决方案二：DIAGATTENTION

### 思路

既然早期层需要local focus，而linear attention做不到，那早期层就用vanilla attention。

但要保持linear complexity。他们的做法：**把sequence切成不重叠的block，每个block内部算vanilla attention**。

block size = $w$，sequence length = $n$，共有 $n/w$ 个block。每个block内是 $w \times w$ 的attention，总共 $O(n \cdot w \cdot d)$。当 $w$ 固定且 $d \ll n$，对 $n$ 是线性的。

### 直觉

这相当于给早期层一个"局部视野"，强制它只能看周围的token。每个block内部是完整的softmax attention，保留了locality。

和Longformer、BigBird那些方法比，这里更简单粗暴 — 纯block-wise，没有sliding window，没有global token。因为只用6层，简单的设计就够了。

### Block size的trade-off

消融实验：
- $w=32$: PPL 3.92
- $w=64$: PPL 3.82
- $w=128$: PPL 3.72

越大越好，但计算量也线性增长。$w=64$ 是他们的trade-off选择。

---

## 六、Hybrid架构的设计哲学

### 核心insight

**不同depth的层有不同的functional role**。

- 早期层：抓local pattern，token-level features，syntactic structure
- 晚期层：做global integration，semantic understanding，long-range reasoning

这和很多其他架构的设计哲学一致：
- CNN：receptive field随depth增大
- U-Net：multi-scale hierarchy
- ResNet：residual learning的层次化

vanilla transformer是uniform的，所有层都一样。但uniform不等于optimal。TRANSNORMER的hybrid设计实际上是在说：**早期层应该local-biased，晚期层应该global-capable**。

### 消融证据

Table 8 — DIAG和NORM的比例：
- 0 DIAG / 12 NORM: 4.23
- 6 DIAG / 6 NORM: 3.82 ← 最佳
- 12 DIAG / 0 NORM: 4.75

Table 9 — 顺序：
- 早期NORM + 晚期DIAG: 4.13
- 早期DIAG + 晚期NORM: 3.82 ← 最佳

早期用DIAG效果好3个点，非常显著的证据。

---

## 七、结果有多好

### Autoregressive LM (WikiText-103)

TRANSNORMER T2: PPL 29.57 / 31.01 (val/test)
Vanilla: PPL 29.63 / 31.01

**基本持平vanilla**。这是linear transformer第一次在这个benchmark上达到vanilla水平。之前的linear方法最好也就31-32，差2-3个PPL。

### GLUE

TRANSNORMER T1平均79.38，**超过vanilla的78.79**。

特别值得注意的是CoLA（语法acceptability判断）：TRANSNORMER T1是45.38，vanilla是38.63，高了6.75分。CoLA需要细粒度syntactic理解，早期层的local attention可能正好帮上了。

### Long-Range Arena

TRANSNORMER T2平均64.80，新SOTA。
之前linear最好的cosFormer是62.11。
vanilla只有57.37。

在Pathfinder（视觉长程依赖任务）上，T2是76.80，vanilla才65.26，差11.54分。长序列上hybrid设计优势明显。

### 速度

5K sequence inference速度：
- Transformer: OOM
- FLASH: 13.16 steps/sec
- Performer: 31.25 steps/sec  
- TRANSNORMER T2: 36.23 steps/sec

比FLASH快近3倍，同时效果还好得多。

---

## 八、这个工作为啥重要

### 方法论的贡献

1. **诊断先于治疗**：他们先从gradient flow和attention distribution两个first principles出发，找到root cause，再设计fix。这比盲目堆trick更有价值。

2. **重新审视softmax**：揭示了softmax的 $\exp$ 提供的两个隐式功能 — gradient stabilization和sharpness/locality。linear transformer同时丢了这两个，所以效果差。

3. **Normalization作为万能工具**：RMSNorm的 $\epsilon$ 可以替代 $\exp$ 的gradient stabilization功能。这个思路后来被很多工作借鉴。

### 对后续工作的影响

这个方向后来发展出一大票工作：
- **Transnormer-LLM** (2023): 扩展到billion-scale，证明这个架构能scale
- **GLA (Gated Linear Attention)** (Yang et al., 2023): 在NORMATTENTION基础上加forget gate，更强的表达能力
- **RetNet** (Microsoft, 2023): 类似的normalization + gating思路
- **RWKV**: 线性RNN的思路和NORMATTENTION有相通之处
- **Mamba** (2023): selective SSM，也是"不同层不同行为"的哲学

### 更深层的启示

这个paper其实揭示了一个普遍性的问题：**当你用计算效率更高的近似替换某个组件时，往往会丢失一些"隐式正则化"**。

softmax的 $\exp$ 同时提供了：
1. 非线性sharpness（产生locality）
2. gradient stability（log-derivative = 1）
3. 数值boundedness（output在[0,1]）

linear transformer用identity替换 $\exp$，同时丢了这三样。NORMATTENTION用normalization补回了2和3，DIAGATTENTION用block attention补回了1。

这个"识别隐式功能 → 显式恢复"的思路，是个很general的方法论。

---

## 九、一句话总结

**Linear transformer效果差的两个原因：gradient爆炸（因为丢掉了 $\exp$ 的自正则化）和attention稀释（因为丢掉了 $\exp$ 的sharpness）。Fix分别是：用RMSNorm替代scaling，早期层用block attention恢复locality。结果是第一个追平vanilla的linear transformer。**

参考链接：
- Paper: https://arxiv.org/abs/2110.04612
- Code: https://github.com/OpenNLPLab/Transnormer
- 后续LLM版本: https://arxiv.org/abs/2307.14995
- GLA: https://arxiv.org/abs/2312.06635
- Mamba: https://arxiv.org/abs/2312.00752
- RetNet: https://arxiv.org/abs/2307.08621
- RWKV: https://arxiv.org/abs/2305.13048

---

# The Devil in Linear Transformer - 深度技术讲解

## 一、Paper核心问题定位

这篇paper来自SenseTime Research & Shanghai AI Lab,核心诊断linear transformer性能落后的两个根本原因,并提出TRANSNORMER架构。让我从底层原理讲清楚为什么这两个问题是"devil"。

paper GitHub: https://github.com/OpenNLPLab/Transnormer
arXiv链接: https://arxiv.org/abs/2110.04612 (实际是COLM 2022)

---

## 二、Linear Transformer Background

Vanilla attention的核心瓶颈在于:

$$\mathbf{O} = \text{Softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\mathbf{V}$$

其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d}$,$n$是sequence length,$d$是hidden dimension。这里 $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$ 是quadratic的根源。

**Kernel trick**的核心思想是把softmax分解。定义kernel function $\phi(\cdot)$,则:

$$\mathbf{O} = \Delta^{-1}\phi(\mathbf{Q})[\phi(\mathbf{K})^\top \mathbf{V}]$$

$$\mathbf{A} = \text{diag}(\phi(\mathbf{Q})[\phi(\mathbf{K})^\top \mathbf{1}_n])$$

其中 $\Delta$ 是normalization diagonal matrix。关键在于先计算 $\phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{d \times d}$,这是constant w.r.t. $n$,所以总complexity降到 $O(nd^2)$,当 $d \ll n$ 时是linear的。

不同方法选择不同的 $\phi$:
- **Katharopoulos et al. 2020**: $\phi(x) = \text{elu}(x) + 1$ (1+elu)
- **Performer**: random feature approximation (Choromanski et al., 2020, https://arxiv.org/abs/2009.14794)
- **cosFormer**: $\phi$ 用cosine positional re-weighting (Qin et al., 2022, https://arxiv.org/abs/2110.04612)
- **Random Feature Attention**: Peng et al. (https://arxiv.org/abs/2103.02143)

---

## 三、Unbounded Gradients - 核心理论洞察

### 3.1 统一形式的attention

paper最关键的洞察是把vanilla和linear attention统一成:

$$p_{ij} = \frac{f(s_{ij})}{\sum_{k=1}^{n} f(s_{ik})}$$

其中 $s_{ij}$ 是token $i$ 和 $j$ 的相似度,$f: \mathbb{R} \to \mathbb{R}$ 是变换函数。

- **Vanilla attention**: $s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d}$, $f(x) = \exp(x)$
- **Linear attention**: $s_{ij} = \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$, $f(x) = x$

### 3.2 Gradient推导

对 $s_{ik}$ 求偏导(注意这里 $s_{ik}$ 是分子 $s_{ij}$ 中相同下标的项):

$$\frac{\partial p_{ij}}{\partial s_{ik}} = \frac{f'(s_{ik})}{f(s_{ik})}(1_{j=k}p_{ij} - p_{ij}p_{ik})$$

这里 $\frac{f'(s_{ik})}{f(s_{ik})}$ 是关键项。我们分两种情况分析:

**Vanilla attention情况**:
$f(x) = \exp(x)$,所以 $f'(x) = \exp(x) = f(x)$,因此 $\frac{f'(x)}{f(x)} = 1$。结果:

$$\frac{\partial p_{ij}}{\partial s_{ik}} = \begin{cases} p_{ik}(1 - p_{ik}) \in [0, 1/4] & j = k \\ -p_{ij}p_{ik} \in [-1/4, 0] & j \neq k \end{cases}$$

这里用AM-GM不等式 $\sqrt{ab} \leq (a+b)/2$ 取 $a = p_{ik}, b = 1-p_{ik}$,得到 $p_{ik}(1-p_{ik}) \leq 1/4$。

**Linear attention情况**:
$f(x) = x$,所以 $f'(x) = 1$,因此 $\frac{f'(x)}{f(x)} = \frac{1}{x}$。结果:

$$\frac{\partial p_{ij}}{\partial s_{ik}} = \frac{1}{s_{ik}}(1_{j=k}p_{ij} - p_{ij}p_{ik})$$

bound变成:

$$\left|\frac{\partial p_{ij}}{\partial s_{ik}}\right| \leq \frac{1}{4|s_{ik}|}$$

而 $|s_{ik}|^{-1} = |\phi(\mathbf{q}_i)\phi(\mathbf{k}_j)^\top|^{-1}$ 可以任意大!

### 3.3 构造性证明unbounded

**Proposition 3.1** 的构造很巧妙。取 $\mathbf{q}_i = \mathbf{k}_j = \phi^{-1}(\mathbf{x}_0)$,其中 $0 < \|\mathbf{x}_0\|_2 \leq \sqrt{\epsilon}$,那么:

$$s_{ij} = \mathbf{x}_0^\top \mathbf{x}_0 \in (0, \epsilon]$$

$$p_{ij} = \frac{1}{n} \text{ (均匀分布)}$$

$$\left|\frac{\partial p_{ij}}{\partial s_{ik}}\right| = \frac{1}{\|\mathbf{x}_0\|^2} \cdot t_{ijk}$$

其中 $t_{ijk} = \frac{1}{n}(1-\frac{1}{n})$ 或 $\frac{1}{n^2}$。当 $\epsilon \to 0^+$,gradient $\to \infty$。

### 3.4 Intuition构建

这里的核心intuition是:

**Softmax的"自正则化"特性**:vanilla attention的 $f(x) = \exp(x)$ 满足 $f'/f = 1$,这是指数函数的独特性质 — 它的logarithmic derivative是常数。这意味着softmax的Jacobian完全不依赖于logit的绝对值,只依赖于softmax输出本身(在 $[0,1]$ 内)。这是为什么softmax在数值上如此稳定的原因。

**Linear attention失去了这个保护**:用 $f(x) = x$ 代替 $\exp(x)$,logarithmic derivative $\frac{f'}{f} = \frac{1}{x}$ 在 $x \to 0$ 时爆炸。而 $\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j) \to 0$ 在训练中经常发生(尤其是当query和key接近正交时),所以gradient爆炸是常态而非例外。

---

## 四、Attention Dilution - 实证观察

### 4.1 Locally Accumulated Attention Score

paper定义了metric来量化locality:

$$l(i, r, N) = \sum_{j=start}^{end} p_{ij}$$

其中邻域大小为 $r \cdot N$,$r$ 是相对比例,$N$ 是sequence length。

例如 $l(i, 0.4, N) = 0.6$ 表示以token $i$ 为中心的40%邻域贡献了60%的attention score。

### 4.2 实验观察

Figure 2的关键观察:
- **Vanilla transformer**:曲线陡峭上升,40%邻域就贡献了~80%的attention,locality强
- **Linear transformer**:曲线平缓,attention被均匀稀释到整个sequence

这背后的原因是 $\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$ 在长序列上对远距离token的score和近距离token差不多(都是非负kernel的内积,容易产生均匀分布),而softmax的 $\exp$ 会放大差异。

### 4.3 Ablation验证

Table 3的controlled experiment很说明问题:
- Early = 1+elu, Late = 1+elu: PPL 4.98
- Early = 1+elu, Late = Vanilla: PPL 3.90
- Early = Vanilla, Late = 1+elu: PPL 3.76

早期用vanilla attention(locality强)收益更大。这印证了:早期层需要捕获local结构,后期层需要global视野。

---

## 五、NORMATTENTION - 解决unbounded gradient

### 5.1 核心思路

既然scaling operation(即公式4的分母 $\sum_k f(s_{ik})$)是unbounded gradient的来源,那就移除它。但直接移除会导致forward pass的attention unbounded:

Table 1显示直接移除scaling导致PPL从4.98暴涨到797.08!

解决方案:移除scaling后,在attention output上加normalization:

$$\mathbf{O} = \mathbf{Q}(\mathbf{K}^\top \mathbf{V})$$

$$\mathbf{O}_{\text{norm}} = \text{XNorm}(\mathbf{Q}(\mathbf{K}^\top \mathbf{V}))$$

XNorm可以是LayerNorm (Ba et al., 2016, https://arxiv.org/abs/1607.06450) 或 RMSNorm (Zhang & Sennrich, 2019, https://arxiv.org/abs/1910.07467)。paper选RMSNorm因为更快。

### 5.2 RMSNorm回顾

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\sigma^2 + \epsilon}}, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d x_i^2$$

其中 $\epsilon > 0$ 是防止除零的小常数(通常 $10^{-6}$ 到 $10^{-8}$)。RMSNorm相比LayerNorm去掉了mean shift,只保留scale normalization,因此更快。

### 5.3 Gradient bound证明

定义:
- $\mathbf{S} = \phi(\mathbf{Q})\phi(\mathbf{K})^\top \in \mathbb{R}^{n \times n}$
- $\mathbf{T} = \mathbf{S}\mathbf{V} \in \mathbb{R}^{n \times d}$
- $\mathbf{O} = \text{RMSNorm}(\mathbf{T})$

RMSNorm的Jacobian元素:

$$\frac{\partial o_{ij}}{\partial t_{ik}} = \frac{1}{\sqrt{\sigma_i^2 + \epsilon}}\left[1\{j=k\} - \frac{1}{d}\frac{t_{ij}t_{ik}}{\sigma_i^2 + \epsilon}\right]$$

这里 $\sigma_i^2 = \frac{1}{d}\sum_{j=1}^d t_{ij}^2$ 是第 $i$ 行的second moment。

利用 $|t_{ij}t_{ik}| \leq (t_{ij}^2 + t_{ik}^2)/2$ (AM-GM),得到:

$$\left|\frac{\partial o_{ij}}{\partial t_{ik}}\right| \leq \frac{1}{\sqrt{\sigma_i^2 + \epsilon}}\left[1 + \frac{1}{2}\right] \leq \frac{3}{2\sqrt{\epsilon}}$$

关键!这里 $\epsilon$ 提供了lower bound,使gradient有界。

定义映射 $h(\mathbf{X}) = \max_i \|\mathbf{X}_i\|_2$,并令:
- $c_1 = h(\nabla_\mathbf{O}\mathcal{L}) = \max_i \|\nabla_{\mathbf{O}_i}\mathcal{L}\|_2$ (上游gradient范数)
- $c_2 = h(\mathbf{V}) = \max_i \|\mathbf{V}_i\|_2$ (value范数)

利用 $\|\mathbf{R}^{(i)}\|_F \leq \frac{3d}{2\sqrt{\epsilon}}$ 和 $\|\mathbf{V}\|_2 \leq \sqrt{n} c_2$,最终:

$$\left|\frac{\partial \mathcal{L}}{\partial s_{ij}}\right| \leq \frac{3 c_1 c_2 d}{2\sqrt{\epsilon}} < \infty$$

这里 $\sqrt{\epsilon}$ 是关键 — $\epsilon$ 越大gradient bound越紧,但数值上distortion越大;越小越接近原值但bound越松。这是工程trade-off。

### 5.4 Empirical验证

Table 2测量了50k iterations的gradient relative standard deviation:

| Method | Rel. Std. Dev. |
|--------|---------------|
| 1+elu (Katharopoulos) | 0.58 |
| Performer | 0.47 |
| Vanilla | 0.25 |
| NORMATTENTION | **0.20** |

NORMATTENTION的gradient甚至比vanilla attention更稳定!这从经验上验证了理论分析。

---

## 六、DIAGATTENTION - 解决attention dilution

### 6.1 设计动机

既然linear attention在早期层会"稀释"attention到远距离token,而vanilla attention擅长捕获local结构,那就在早期层用block-wise vanilla attention。

### 6.2 实现方式

将sequence分成 $n/w$ 个不重叠的block,每个block大小为 $w$。在每个block内独立计算vanilla attention:

$$\text{DiagAttention}(\mathbf{X})_{[i:i+w]} = \text{Softmax}(\mathbf{Q}_{[i:i+w]}\mathbf{K}_{[i:i+w]}^\top / \sqrt{d})\mathbf{V}_{[i:i+w]}$$

复杂度: $O(n \cdot w \cdot d)$,当 $d \ll n$ 时是 $O(n)$ w.r.t. sequence length。

### 6.3 与相关方法对比

- **Longformer** (Beltagy et al., 2020, https://arxiv.org/abs/2004.05150):sliding window + global tokens
- **BigBird** (Zaheer et al., 2020, https://arxiv.org/abs/2007.14062):random + window + global
- **BlockSparse** (Child et al., 2019, https://arxiv.org/abs/1904.10509):sparse patterns

DIAGATTENTION的特殊之处是只用于早期层,且与NORMATTENTION的hybrid设计是关键。

### 6.4 Block size消融

Table 11显示block size越大效果越好:
- $w=32$: PPL 3.92
- $w=64$: PPL 3.82 (paper选择)
- $w=128$: PPL 3.72

但 $w$ 越大computational cost越高($O(nwd)$),所以选 $w=64$ 作为trade-off。

---

## 七、TRANSNORMER整体架构

### 7.1 Hybrid设计

```
Input → [DIAGATTENTION × 6] → [NORMATTENTION × 6] → Output
        (early stage)         (later stage)
        捕获local structure   捕获global context
```

Table 8的消融显示6/6 split最优:
- 0 DIAG / 12 NORM: PPL 4.23
- 6 DIAG / 6 NORM: PPL 3.82 ← 最佳
- 12 DIAG / 0 NORM: PPL 4.75

Table 9进一步证明:DIAG在early stage比在later stage好(PPL 3.82 vs 4.13)。这说明early层需要locality,late层需要global view,这正好与CNN的receptive field逐渐扩大的设计哲学一致。

### 7.2 两个variants

- **TRANSNORMER T1**: DIAG用ReLA attention (Zhang et al., 2021, https://arxiv.org/abs/2104.07012),NORM用elu激活
- **TRANSNORMER T2**: DIAG用softmax attention,NORM用1+elu激活

### 7.3 FFN选择

Table 10显示GLU (Shazeer, 2020, https://arxiv.org/abs/2002.05202) 优于传统FFN:
- FFN: PPL 3.93
- GLU: PPL 3.82

GLU的公式: $\text{FFN}_{\text{GLU}}(x) = (\text{Swish}(xW_1) \otimes xW_2)W_3$,通过gating mechanism增强表达力。

### 7.4 Combination消融

Table 12证明不应在同一层内串联使用:
- 交替 D→N: PPL 4.19
- 并行: PPL 3.77 (但double computation)
- 串联(TRANSNORMER): PPL 3.82

---

## 八、实验结果深度分析

### 8.1 Autoregressive LM (WikiText-103)

Table 4核心数据:

| Method | PPL (val) | PPL (test) | Params |
|--------|-----------|------------|--------|
| Vanilla | 29.63 | 31.01 | 156M |
| FLASH | 33.18 | 34.63 | 153M |
| 1+elu | 32.63 | 34.25 | 156M |
| Performer | 75.29 | 77.65 | 156M |
| **TRANSNORMER T2** | **29.57** | **31.01** | 156M |

TRANSNORMER T2**匹配甚至超过**vanilla transformer!这是linear transformer首次在autoregressive LM上达到vanilla水平。相比之前SOTA(FLASH-quad val 31.88, LS test 32.59),提升2.31和1.58 PPL。

### 8.2 Bidirectional LM (GLUE)

Table 5平均分:
- Vanilla: 78.79
- FLASH: 76.87
- Performer: 63.41 (基本失败)
- 1+elu: 70.00
- **TRANSNORMER T1: 79.38** (超过vanilla!)
- TRANSNORMER T2: 78.78

在CoLA任务上TRANSNORMER T1达45.38,比vanilla的38.63高6.75分,这个提升非常显著。CoLA是linguistic acceptability判断,需要细粒度的syntactic理解。

### 8.3 Long-Range Arena

Table 6平均:
- Vanilla: 57.37
- cosFormer: 62.11 (之前linear SOTA)
- FLASH: 61.31
- **TRANSNORMER T2: 64.80** (新SOTA)
- TRANSNORMER T1: 63.71

在Pathfinder(视觉spatial reasoning)上TRANSNORMER T2达76.80,比vanilla的65.26高11.54分!这说明hybrid设计在长序列上确实有优势。

### 8.4 速度对比

Table 7 (A6000 GPU):

| Model | 1K infer | 5K infer | 1K train | 5K train |
|-------|----------|----------|----------|----------|
| Transformer | 39.06 | OOM | 15.34 | OOM |
| FLASH | 40.32 | 13.16 | 20.49 | 6.93 |
| Performer | 104.17 | 31.25 | 28.41 | 9.06 |
| **TRANSNORMER T2** | **119.05** | **36.23** | **29.41** | **10.16** |

TRANSNORMER T2在5K sequence上比FLASH快~175% (inference) 和~46% (training)。

---

## 九、深层intuition与思考

### 9.1 为什么vanilla attention如此难以替代?

paper揭示了两个深层原因:

1. **Softmax的logarithmic derivative = 1**:这是softmax的"免费午餐",它的Jacobian完全由output分布决定,不依赖于logit绝对值。任何替代 $f(x) = \exp(x)$ 的函数都需要面对这个问题。

2. **Softmax的sharpness**: $\exp$ 函数的非线性放大了score差异,产生locality。Linear kernel的内积是"线性"的,无法产生sharp分布。

### 9.2 NORMATTENTION的深层意义

NORMATTENTION本质上是用 **数值稳定的normalization** 替换 **不稳定的scaling**。这有两层含义:

- **Forward stability**:RMSNorm的 $\sqrt{\sigma^2 + \epsilon}$ 保证output scale固定
- **Backward stability**:RMSNorm的Jacobian的 $\frac{1}{\sqrt{\epsilon}}$ bound保证gradient有界

这与最近的研究趋势一致 — 用structural normalization替代softmax-based normalization。后续工作如Hyena (https://arxiv.org/abs/2202.08434)、RWKV (https://arxiv.org/abs/2305.13048)、RetNet (https://arxiv.org/abs/2307.08621) 都采用了类似思路。

### 9.3 Hybrid架构的哲学

TRANSNORMER的hybrid设计反映了一个重要intuition: **不同depth的层有不同的functional role**。

- 早期层:token-level features,local syntactic patterns (类似CNN的low-level features)
- 晚期层:semantic integration,long-range dependencies

这与U-Net的multi-scale设计、ResNet的residual hierarchy、以及最近Mamba (https://arxiv.org/abs/2312.00752) 的selective SSM有相似的哲学 — **不是uniform地处理所有层,而是根据functional需求定制**。

### 9.4 与后续工作的联系

这篇paper启发了后续一系列工作:
- **Transnormer-LLM** (2023): 扩展到billion-scale LM (https://arxiv.org/abs/2307.14995)
- **GLA (Gated Linear Attention)** (2023): 在NORMATTENTION基础上加forget gate (https://arxiv.org/abs/2312.06635)
- **Lightning Attention**: 进一步优化NORMATTENTION的并行计算 (https://arxiv.org/abs/2402.01698)

### 9.5 公式变量总结

最后总结一下关键公式中的变量含义:

**Eq. 4** ($p_{ij} = \frac{f(s_{ij})}{\sum_{k=1}^{n} f(s_{ik})}$):
- $p_{ij}$: attention matrix第 $i$ 行第 $j$ 列元素,表示token $i$ 对token $j$ 的attention weight
- $s_{ij}$: token $i$ 和 $j$ 的similarity score
- $f$: 变换函数, vanilla用 $\exp$, linear用identity
- 下标 $k$: 求和索引,遍历所有keys

**Eq. 14** ($\mathbf{O}_{\text{norm}} = \text{XNorm}(\mathbf{Q}(\mathbf{K}^\top \mathbf{V}))$):
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d}$: query, key, value matrices
- $\mathbf{K}^\top \mathbf{V} \in \mathbb{R}^{d \times d}$: key-value外积,与 $n$ 无关
- $\mathbf{Q}(\mathbf{K}^\top \mathbf{V}) \in \mathbb{R}^{n \times d}$: unnormalized output
- XNorm: row-wise normalization (LayerNorm or RMSNorm)

**Eq. 15** ($\left|\frac{\partial \mathcal{L}}{\partial s_{ij}}\right| \leq \frac{3 c_1 c_2 d}{2\sqrt{\epsilon}}$):
- $\mathcal{L}$: loss function
- $s_{ij}$: similarity score
- $c_1$: upstream gradient的max $L_2$ norm
- $c_2$: value matrix的max row $L_2$ norm
- $d$: hidden dimension
- $\epsilon$: RMSNorm的small constant (通常 $10^{-6}$)

---

## 十、总结

这篇paper的价值在于它**不是简单堆砌trick,而是从gradient flow和attention distribution两个first principles出发**,诊断了linear transformer的root cause。NORMATTENTION通过移除scaling + 加normalization解决gradient问题,DIAGATTENTION通过block-wise vanilla attention解决locality问题,两者的hybrid设计实现了vanilla-level performance + linear complexity。

后续工作证明这个方向是正确的 — NORMATTENTION的normalization思路被广泛采纳,成为efficient transformer的核心设计原则之一。

**主要参考链接**:
- TRANSNORMER GitHub: https://github.com/OpenNLPLab/Transnormer
- Transnormer-LLM: https://arxiv.org/abs/2307.14995
- Linear Attention (Katharopoulos): https://arxiv.org/abs/2006.16236
- Performer: https://arxiv.org/abs/2009.14794
- cosFormer: https://arxiv.org/abs/2110.04612
- FLASH: https://arxiv.org/abs/2202.10447
- RMSNorm: https://arxiv.org/abs/1910.07467
- LayerNorm: https://arxiv.org/abs/1607.06450
- Long-Range Arena: https://arxiv.org/abs/2011.04006
- GLU variants: https://arxiv.org/abs/2002.05202
- ReLA (Sparse Attention with Linear Units): https://arxiv.org/abs/2104.07012
- Mamba: https://arxiv.org/abs/2312.00752
- RetNet: https://arxiv.org/abs/2307.08621
- RWKV: https://arxiv.org/abs/2305.13048
