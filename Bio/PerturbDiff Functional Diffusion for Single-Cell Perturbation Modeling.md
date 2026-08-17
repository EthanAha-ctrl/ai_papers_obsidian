---
source_pdf: PerturbDiff Functional Diffusion for Single-Cell Perturbation Modeling.pdf
paper_sha256: 4a12a1818f635c344d63b06b5c494f76a54f0a10351cf1bc615f8fab10e8d840
processed_at: '2026-08-06T02:58:13-07:00'
target_folder: Bio
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 PerturbDiff

## 一句话版本

假设你想预测 drug 打到 cell 上会发生什么，但同一个 cell 你测完 baseline 就把它"杀掉"了，没法再测 perturbed 版本。所以你手上只有两堆 unpaired 的 cells。以前的做法要么假装它们 paired，要么学一个"control 分布 → perturbed 分布"的 deterministic map。PerturbDiff 说：**同一个实验条件下，perturbed response 其实是一族可能的分布**，因为有一堆你看不见的 latent factor 在捣乱。所以我来学"分布的分布"。

---

## 背景故事

做 single-cell perturbation 实验的场景：你有一批 cells，分成两组。A 组不加干预（control），B 组加某种 drug 或基因敲除（perturbed）。然后 sequencing 测 gene expression。但 sequencing 是破坏性的——cell 被裂解了，同一个 cell 测不了两次。

所以你的数据天生是 **unpaired** 的：A 组里的 cell #37 和 B 组里的 cell #142 到底是不是同一个细胞的前后状态？你根本不知道。

---

## 以前怎么做的

### 第一代：假装 paired

GEARS、scGPT 这些方法直接 random 把一个 control cell 和一个 perturbed cell 配成对，然后跑 regression 学 "$\text{control cell} \to \text{perturbed cell}$" 的 mapping。

问题：random pairing 等于你强制让模型学一个 averaged response，因为配对是 noise。所有 heterogeneity 都被 wash 掉了。

### 第二代：学分布到分布的 map

CellOT 用 optimal transport，STATE 用 MMD alignment，CellFlow 用 flow matching。这些方法说：好，我不配对 cell 了，我直接学 "control 人群的分布" 到 "perturbed 人群的分布" 的 mapping。

问题在于它们默认一个**假设**：给定 cell type 和 perturbation type，perturbed 分布是唯一确定的。即 $P_{c,\tau}$ 是 fixed。

---

## PerturbDiff 的核心洞察

这个假设在现实里**不成立**。

举几个例子：同样是用 cytokine X 刺激 T cell，但
- donor A 的微环境跟 donor B 不一样
- batch 1 的试剂 lot 跟 batch 2 差一点
- 实验当天的温度湿度有波动
- 同一 cell type 内部还有 sub-clone heterogeneity

这些 unobserved factor 会让"同一个 $(c, \tau)$"对应**一族** plausible response distributions，而非一个。

所以正确的 random variable 应该是"一个分布"，而不是"一个 cell"。

---

## 数学上怎么实现

### 第一步：把分布变成一个点

用 kernel mean embedding。给一个 distribution $P$，用 kernel $k$ 把它 embed 成 RKHS 里的一个点 $\mu_P$。这就像把一个复杂对象压缩成一个向量，但你保住了所有的几何性质：

- 两个分布的"距离" = MMD
- 两个分布的 mixture = embedding 的线性组合

所以整个 distribution 空间被你弄成一个 Hilbert space，每个分布就是里面的一个点。

### 第二步：在这个 space 上跑 DDPM

标准 DDPM 是在 $\mathbb{R}^d$ 上跑：从 clean image 加 noise 到 Gaussian，然后学 reverse process 去 noise。

PerturbDiff 把这件事搬到 RKHS 上：从 clean embedding $\mu_0$ 加 GRE（Gaussian random element，infinite-dimensional 版本的 Gaussian）到 reference measure，然后学 reverse。

推导一模一样：forward 是 affine + Gaussian，closed form marginal；reverse 用 variational + neural network 参数化 mean。

### 第三步：loss 自然变成 MMD

这是最爽的部分。在标准 DDPM 里，denoising loss 是 $\|x_0 - x_\theta\|_2^2$，cell-level 的 MSE。

在 RKHS diffusion 里，loss 是 $\|\mu_0 - \mu_\theta\|_{\mathcal{H}_k}^2$，而根据 RKHS 性质这**就是** $\text{MMD}^2(P_{\text{pert}}, P_\theta)$。

MMD 不是外部加的 regularizer，是从几何推导里**自然掉出来的**。这给 STATE 那种"用 MMD 做 population alignment"的方法一个理论 grounding：你不是在 hack，是在做 RKHS 上的 maximum likelihood。

### 第四步：怎么实际训

理论在 $\mathcal{H}_k$（infinite-dimensional），实操在 cell batch（finite matrix）。三个 trick：

**Trick 1：用 batch 估 embedding**
取 $m$ 个 perturbed cells $B_{\text{pert}} = \{\mathbf{x}_1, ..., \mathbf{x}_m\}$，empirical distribution $\tilde{P} = \frac{1}{m}\sum \delta_{\mathbf{x}_j}$，embedding $\mu_{\tilde{P}} = \frac{1}{m}\sum k(\mathbf{x}_j, \cdot)$。这是 $\mu_{c,\tau}$ 的 Monte Carlo 估计。

**Trick 2：forward noise 在 cell space 加**
要从 $\mathcal{N}_{\mathcal{H}_k}(0, C)$ 采样 function-valued Gaussian 不可行。但对每个 cell 加 Euclidean Gaussian noise $\varepsilon_j \sim \mathcal{N}(0, I)$，重新算 embedding，在一阶 Taylor 近似下这等价于在 RKHS 里加 GRE，covariance 由 kernel 在 batch 处的 local Jacobian 显式决定。

**Trick 3：网络输出整个 batch**
让 network $f_\theta$ 一次输出一整个 batch 的预测 cells $B_\theta = \{\mathbf{x}_1^\theta, ..., \mathbf{x}_m^\theta\}$，用 MMD 比较 $B_{\text{pert}}$ 和 $B_\theta$ 这两个 batch。

---

## 为什么这比 cell-level diffusion 好

single-cell 数据是**极 sparse** 的。PBMC 上 2000 个 HVG 里平均每个 cell 只有 100 个 nonzero，>95% 是 0。

cell-level MSE loss 会被 zero dominate：预测 0 对 zero gene 已经是最优，model 学到的就是"输出一个稀疏的、类似平均的 vector"。所有 distribution shift 信息被吃掉了。

MMD 是 pairwise distance 比较，它看的是 batch 内部的 **relative geometry**：cell A 离 cell B 多远、cell C 离 cell D 多远。这个 geometry 不被 zero entries 主导，能 capture 到 distribution-level 的 shift pattern——比如某个 subpopulation 扩大了、某个 subpopulation 缩小了、新出现了一个 cluster。

Ablation（Fig. 9）证实：去掉 MMD 只留 MSE，DE-related metrics 大幅下降。

---

## 架构上的巧思

用 **MM-DiT**（Stable Diffusion 3 的架构）做 backbone，但改造成双 stream：

- **Perturbed stream**：吃 noised perturbed batch
- **Control stream**：吃 control batch

每个 cell 是一个 token。两个 stream 在每个 transformer block 里通过 joint attention 互动：perturbed tokens 可以 attend 到 control tokens，学"perturbation 相对于 control 偏离了什么"。Control stream 不接 loss，纯做 conditioning。

Conditioning 用 AdaLN-Zero（DiT 的标准操作）注入 timestep + cell type + perturbation label。Perturbation label 可以是 ESM2（cytokine）、ChemBERTa（drug）、GenePT（gene）的 semantic embedding。

---

## Pretraining 这个事

Perturbation 数据虽然大，但 context 多样性差（Replogle 只有 4 个 cell line）。CellxGene 有 60M cells、662 cell types、10000+ batches，覆盖广得多。

所以先在 CellxGene + perturbation 数据上做 marginal pretraining：学"给定 cell type，generate 一个 unperturbed cell 分布"。这给模型一个 prior over cell manifold。

然后 fine-tune 加 perturbation conditioning。

效果：
- **Zero-shot**：pretrain 完直接不给 perturbation label 也能产生合理预测，说明 perturbation shift 部分沿着 marginal manifold 的已有方向。
- **Low-data**：Replogle 上 from scratch 输 STATE，pretrain 后反超。PBMC subsample 到 1% 时 finetune 远好于 scratch。

这跟 vision/language 里 pretrain → fine-tune 的 story 一样。

---

## 实验结果人话版

三个 dataset 上测：

- **PBMC**（9.7M cells，90 个 cytokine perturbation）：PerturbDiff 几乎所有指标赢。
- **Tahoe100M**（100M cells，1137 个 drug）：PerturbDiff 整体最强。
- **Replogle**（0.6M cells，2023 个 gene perturbation，low-data）：From scratch 输 STATE 一点，finetune 后追平。

最 striking 的 qualitative 结果（Fig. 5）：DE gene 的 $-\log_{10}(p_{\text{adj}})$ 分布。Ground truth 里 DE 和 non-DE 分得很开。PerturbDiff 也分得开。但 STATE 几乎把所有 gene 都标成 DE——它没学到 "哪些 gene 被 perturb" 这件事，只学到"对每个 gene 都产生一些 shift"。这是 distribution-level modeling 的优势：它 capture 了 population signature 而非 cell-wise noise。

**Scaling 行为有意思**：non-monotonic。中等模型最好最稳，大模型反而开始 overfit。Perturbation modeling 不像 image generation 那样吃 brute-force scaling。可能因为 signal 弱、metric sensitive。

---

## 这个 idea 的"level"在哪

把 random variable 从"个体"提升到"分布"，这件事在 concept 上是一个 level shift。以前所有方法都把 cell 当 random variable，PerturbDiff 把 distribution 当 random variable。

技术上，这要求你能在 distribution 空间上定义 generative process。RKHS embedding 给了你一个 handle：distribution → point in Hilbert space → DDPM on Hilbert space。

而且这个 pattern 可以推广：
- 任何你能 kernel embed 的对象（distributions, graphs, trajectories, point clouds）都能套
- 任何 sparse high-dim generative task 都可能受益于 distribution-level loss 替代 element-wise MSE
- Function space diffusion 的技巧（cell-space noise → RKHS GRE 近似）是个通用 trick

---

## 几个值得琢磨的方向

1. **Kernel 选择**：paper 用 energy distance kernel。Deep kernel、Gaussian kernel、graph kernel 可能 capture 不同 structure。
2. **Joint attention 的必要性**：cross-attention only（control → perturbed 单向）够不够？bidirectional 是不是 overkill？
3. **Pretraining U-shape**：早期 pretrain 暂时 hurt downstream，类似 catastrophic forgetting。需要 task-aware schedule。
4. **跟 Schrödinger Bridge 对比**：SB 是 stochastic bridge between distributions，PerturbDiff 是 unconditional RKHS diffusion。两者可能在 population transition 上有联系。
5. **Non-monotonic scaling**：为什么 perturbation modeling 不吃 scaling？是 metric 问题还是信号本质问题？

---

## 总结

PerturbDiff 干了三件事：
1. **Concept**：把 perturbation prediction 的 random variable 从 cell 提升到 distribution of distributions。
2. **Theory**：在 RKHS 上推 DDPM，loss 自然成为 MMD，给 population alignment 一个理论 grounding。
3. **Empirical**：三个大规模 benchmark 上 SOTA，特别是 DE gene recovery 显著优于之前所有方法。

如果你要对一个人用 30 秒解释这篇 paper：

> "以前大家要么把 unpaired cells 假装配对，要么学一个 distribution 到 distribution 的 deterministic map。PerturbDiff 说，同一条件下的 response 分布本身是随机的，所以我来学'分布的分布'。技术上靠把分布 embed 到 Hilbert space 然后在那上面跑 diffusion，loss 自然就是 MMD。在三个 benchmark 上都 SOTA。"

如果你要我继续展开某一块——比如 score matching 那个 Prop. B.15 的直觉、joint attention 的算力、pretraining U-shape 的可能解释——告诉我。

---

# PerturbDiff: 在 RKHS 上做 Diffusion，建模 distribution of distributions

让我先给你 build up 一份 intuition，然后再钻到公式和架构里。

## 1. 为什么这篇 paper 在概念上是一个"level shift"

生物学里的 high-throughput single-cell sequencing 是**破坏性**的：同一个 cell 你测完 baseline 就把它"杀掉"了，没法再测 perturbed 版本。所以你拿到的数据本质上就是两个 unpaired 的 cell populations：$X_{\text{ctrl}}$ 和 $X_{\text{pert}}$，没有 cell-to-cell correspondence。

早期方法（GEARS、scGPT）做的是 random pairing + regression，本质上把所有 cell 当成 paired，结果学到一个**averaged response**——丢掉了 heterogeneity。

后来 CellOT、STATE、CellFlow、Squidiff 这些 population-level 方法学的是 $P_{\text{ctrl}} \to P_{\text{pert}}$ 的 map。但这里有个**被忽略的关键问题**：conditioned on observed context $(c, \tau)$，$P_{c,\tau}$ 实际上不是一个 fixed distribution。Unobserved latent factors（microenvironment fluctuations、batch effects、donor variability 等）会让"同一个 $(c,\tau)$"对应**一族 plausible distributions**。

所以 paper 的核心 move 是：

$$
\text{random variable} = \text{a cell distribution } P \in \mathcal{P}(\mathcal{X})
$$

而不是

$$
\text{random variable} = \text{a single cell } \mathbf{x} \in \mathcal{X}
$$

这下 diffusion 的 sample space 从 $\mathbb{R}^{|\mathcal{G}|}$（几千维 gene expression）变成了 $\mathcal{P}(\mathcal{X})$（一个无穷维的 distribution 空间）。要在这个空间上跑 DDPM，你需要把它弄成一个 tractable Hilbert space。这就是 kernel mean embedding 上场的理由。

参考这个 idea 的脉络：
- Kernel mean embedding 的综述：https://arxiv.org/abs/1605.00788
- MMD 两样本检验：https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
- Function space diffusion（ Kerrigan et al.）：https://arxiv.org/abs/2212.00886
- DDPM 原文：https://arxiv.org/abs/2006.11239

---

## 2. 从 cell 到 distribution：kernel mean embedding 的几何

给一个 positive-definite kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$，由 Moore-Aronszajn 定理存在唯一一个 RKHS $\mathcal{H}_k$。任何 distribution $P$ 都可以 embed 成一个点：

$$
\mu_P := \mathbb{E}_{Z \sim P}[k(Z, \cdot)] \in \mathcal{H}_k
$$

这里 $k(Z, \cdot)$ 是个 $\mathcal{H}_k$ 里的 function（"feature map at point $Z$"），$\mu_P$ 是这些 function 在 $P$ 下的期望。$Z$ 是从 $P$ 里采的随机变量，$\cdot$ 是 RKHS 的"输入 slot"。

为什么这个 embedding 好用？三条性质（Lemma B.1）：

1. **Linearity**: $\mu_{\alpha P + (1-\alpha)Q} = \alpha \mu_P + (1-\alpha) \mu_Q$
   → mixture of distributions 对应 RKHS 里的 affine combination，意味着 diffusion 的 forward process 在 RKHS 里仍然是 Gaussian。

2. **Kernel geometry**: $\langle \mu_P, \mu_Q \rangle_{\mathcal{H}_k} = \mathbb{E}_{Z \sim P, Z' \sim Q}[k(Z, Z')]$
   → 内积就是 expected kernel similarity，可以 Monte Carlo 估计。

3. **Induced distance**: $\|\mu_P - \mu_Q\|_{\mathcal{H}_k}^2 = \text{MMD}_k^2(P, Q)$
   → RKHS norm **就是** MMD。这是后面 loss 推导的灵魂。

PerturbDiff 实际用的是 **energy distance kernel**，等价于
$$
\text{ED}(P, Q) = 2\mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|
$$
其中 $X, X' \sim P$, $Y, Y' \sim Q$。这玩意儿在 sample 上是 $O(m^2)$ 计算的，但好处是 unbiased、不需要选 bandwidth、对高维 sparse 单细胞数据 robust。Energy distance 和 RKHS embedding 的等价性见 Sejdinovic et al. 2013: https://projecteuclid.org/journals/Annals-of-Statistics/volume-41/issue-3/Equivalence-of-distance-based-and-RKHS-based-statistics-in-hypothesis-testing/10.1214/13-AOS1160.full

---

## 3. Forward diffusion 在 RKHS 上：从 $\mu_0$ 到 $\mu_T$

记 $\mu_0 := \mu_{c,\tau}$ 是 perturbed population 的 embedding。Forward chain 直接照搬 DDPM：

$$
\mu_t = \sqrt{1 - \beta_t}\, \mu_{t-1} + \sqrt{\beta_t}\, \Xi_t, \qquad \Xi_t \sim \mathcal{N}_{\mathcal{H}_k}(0, C)
$$

变量解释：
- $\beta_t \in (0,1)$：variance schedule，第 $t$ 步注入的噪声方差比例。
- $\Xi_t$：$\mathcal{H}_k$-valued **Gaussian random element (GRE)**，类比 $\mathbb{R}^d$ 里的 $\mathcal{N}(0, \Sigma)$，但是 infinite-dimensional。
- $C: \mathcal{H}_k \to \mathcal{H}_k$：covariance operator，必须 symmetric、positive semi-definite、compact、trace-class。Trace-class 意味着 $\text{tr}(C) = \mathbb{E}\|\Xi\|_{\mathcal{H}_k}^2 < \infty$，否则没法定义 Gaussian measure。

闭式 marginal（用 Lemma B.7 affine + Lemma B.8 sum of independent GRE 反复套）：

$$
\mu_t \mid \mu_0 \sim \mathcal{N}_{\mathcal{H}_k}\!\left(\sqrt{\alpha_t}\,\mu_0,\; (1-\alpha_t)\, C\right), \qquad \alpha_t := \prod_{s=1}^t (1-\beta_s)
$$

$\alpha_t$ 是 cumulative signal retention，从 1 衰减到 ~0；$(1-\alpha_t) C$ 是累计 noise covariance。

这个 marginal 形式跟 DDPM 一模一样，只是把 $\mathbb{R}^d$ 换成 $\mathcal{H}_k$，把 covariance matrix $\Sigma$ 换成 operator $C$。整个推导依赖于 **Gaussian measures 在 Hilbert space 上对 affine transformation 闭合**这一性质（Feldman-Hájek 定理附近的结果）。

参考 Kerrigan et al. 2022 在 function space 里的 DDPM 推导：https://arxiv.org/abs/2212.00886

---

## 4. Reverse process + variational bound：自然引出 MMD loss

像 DDPM 一样，真后验 $P_{t-1|t,0}(\mu_{t-1} \mid \mu_t, \mu_0)$ 是闭式的 GRE，均值是 $\mu_t$ 和 $\mu_0$ 的线性组合（Prop. B.10）：

$$
\tilde{m}_t(\mu_t, \mu_0) := \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\,\mu_0 + \frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}\,\mu_t
$$

这是 $\mathbb{R}^d$ DDPM 后验均值的 RKHS 版本。系数解释：
- $\frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}$：clean target $\mu_0$ 的权重；
- $\frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}$：当前 noisy state $\mu_t$ 的权重；
- $\tilde{\beta}_t := \frac{1-\alpha_{t-1}}{1-\alpha_t}$：后验 variance scale。

直接对 $P_{t-1|t}$ 采样不可行（RKHS 没有万能的 dominating measure，Radon-Nikodym density 不一定存在）。所以走 variational，用一个 neural network $\mu_\theta(\mu_t, t, \mu_c, c, \tau)$ 参数化 reverse mean。

KL between two GREs with shared covariance（Lemma B.11）：

$$
\text{KL}(P \| Q) = \frac{1}{2}\langle m_1 - m_2, \bar{C}^{-1}(m_1 - m_2)\rangle_{\mathcal{H}_k}
$$

设 reverse covariance 等于 forward 的 $\tilde{\beta}_t C$，再让 $\mu_\theta$ 直接预测 $\mu_0$（x0-prediction parameterization），最终 simplified loss（Eqn. 6）：

$$
\mathcal{L}_t \propto \big\| \mu_0 - \mu_\theta(\mu_t, t, \mu_c, c, \tau) \big\|_{\mathcal{H}_k}^2
$$

这里 $\mu_c := \mu_{D_c}$ 是 control population 的 embedding，$(c, \tau)$ 是 context 和 perturbation label，$\mu_t$ 是 noisy perturbed embedding。条件用 **classifier-free guidance** 注入（训练时随机 drop metadata 用 null token，采样时 extrapolate conditional 和 unconditional 两个估计）。

---

## 5. Tractable 训练：从 Hilbert space 退回到 cell batch

上面这个 loss 在 $\mathcal{H}_k$ 里写不出来（无穷维），但有个**优雅的等价**：

用 perturbed batch $B_{\text{pert}} = \{\mathbf{x}_1, \dots, \mathbf{x}_m\}$ 构造 empirical distribution $\tilde{P}_{\text{pert}} = \frac{1}{m}\sum_j \delta_{\mathbf{x}_j}$，它的 empirical kernel mean embedding 是

$$
\mu_{\tilde{P}_{\text{pert}}} = \frac{1}{m}\sum_{j=1}^m k(\mathbf{x}_j, \cdot) \in \mathcal{H}_k
$$

这是 $\mu_{c,\tau}$ 的 Monte Carlo 估计。Prop. B.2 给出收敛率（multivariate DKW inequality）：

$$
\mathbb{P}\!\left[\sup_x |F_n(x) - F(x)| > t\right] \leq 2|\mathcal{G}| \exp(-2mt^2)
$$

也就是 batch size $m$ 越大 empirical distribution 在 sup norm 下指数收敛到真实 distribution，$|\mathcal{G}|$ 是基因数。

让网络输出一整个预测 batch $B_\theta = \{\mathbf{x}_1^\theta, \dots, \mathbf{x}_m^\theta\}$，对应 empirical distribution $\tilde{P}_\theta$。然后利用 RKHS distance 性质：

$$
\big\| \mu_0 - \mu_\theta(\mu_t, t, \mu_c, c, \tau) \big\|_{\mathcal{H}_k}^2 = \text{MMD}_k^2(\tilde{P}_{\text{pert}}, \tilde{P}_\theta)
$$

**这是整篇 paper 的关键 trick**：diffusion 在抽象的 distribution 空间上推导，最后 loss 落到 cell batch 的 MMD 上。MMD 不是外部加的 regularizer，而是 RKHS diffusion 的**自然几何**。

Paper 用 energy distance kernel，所以：
$$
\mathcal{L}_{\text{MMD}} = 2\mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|
$$
$X, X' \sim \tilde{P}_{\text{pert}}$, $Y, Y' \sim \tilde{P}_\theta$。能量距离在 batch 上是 $O(m^2)$ 计算，正好和 self-attention 一个量级。

参考 STATE 用 MMD 做 deterministic alignment 的工作：https://www.biorxiv.org/content/10.1101/2025.06

---

## 6. Forward noise 怎么采样？一个一阶 Taylor 的近似

理论上你要从 $\mathcal{N}_{\mathcal{H}_k}(0, C)$ 采样一个 function-valued object。这显然不可行。Paper 给了一个聪明的近似：

对 batch 里每个 cell $\mathbf{x}_j$ 独立加 Euclidean Gaussian noise $\varepsilon_j \sim \mathcal{N}(0, I_{|\mathcal{G}|})$，得到 $\mathbf{x}_j' = \mathbf{x}_j + \varepsilon_j$，然后重新构造 embedding：
$$
\tilde{\mu} := \frac{1}{m}\sum_{j=1}^m k(\mathbf{x}_j', \cdot)
$$

对 $k(\mathbf{x}_j', \cdot)$ 在 $\mathbf{x}_j$ 处一阶 Taylor 展开：
$$
\tilde{\mu} - \mu_0 \approx \frac{1}{m}\sum_{j=1}^m \nabla_{\mathbf{x}} k(\mathbf{x}_j, \cdot)^\top \varepsilon_j =: \Delta
$$

**Lemma B.13** 说 $\Delta$ 是 $\mathcal{H}_k$-valued GRE，证明思路：定义线性算子 $T_j: \mathbb{R}^{|\mathcal{G}|} \to \mathcal{H}_k$，$T_j v = \sum_r v_r \partial_{\mathbf{x}^r} k(\mathbf{x}_j, \cdot)$。$\langle \Delta, h \rangle_{\mathcal{H}_k}$ 是 $\varepsilon_j$ 的线性泛函之和，independent Gaussian 的线性组合还是 Gaussian，所以 $\Delta$ 是 GRE。

**Prop. B.14** 给出显式 covariance operator：
$$
C = \frac{1}{m^2}\sum_{j=1}^m T_j T_j^*
$$
其中 $T_j^*: \mathcal{H}_k \to \mathbb{R}^{|\mathcal{G}|}$ 是 $T_j$ 的 adjoint。

Intuition：**给 cell 加 Euclidean noise 等价于给 RKHS embedding 加一个 GRE，covariance 由 kernel 在当前 batch 处的 local Jacobian 决定。** 这个近似在 noise 小（即 $t$ 小）的时候非常精确，在 $t$ 大的时候反正已经接近 reference Gaussian，所以全程 work。

这个 trick 让你完全不用显式碰 $\mathcal{H}_k$，所有操作都在 $\mathbb{R}^{m \times |\mathcal{G}|}$ 上做。

---

## 7. Multi-scale loss：MMD + MSE

完整训练 objective（Eqn. 8）：

$$
\mathcal{L}_{\text{total}} = \text{MMD}_k^2(\tilde{P}_{\text{pert}}, \tilde{P}_\theta) + \frac{1}{m}\sum_{j=1}^m \|\mathbf{x}_j - \mathbf{x}_j^\theta\|_2^2
$$

MMD term 是 distribution-level alignment，MSE term 是 cell-level reconstruction。Paper 的 ablation（Sec. 5.5, Fig. 9）显示：

- 去掉 MMD → DE-related metrics 大幅下降（特别是 sparse 的 PBMC，>95% zero entries）
- 去掉 MSE → 性能变化不大，说明优化主要由 MMD 主导，MSE 相当于在 batch centroid 上加了点 regularizer

直觉：单细胞数据**极 sparse**，纯 MSE 会被 zero entries dominate（预测 0 对 zero gene 已经最优），训不出 meaningful distribution shift。MMD 通过 pairwise distance 比较 batch 内的 relative geometry，绕开了 sparsity 陷阱。

---

## 8. 架构：MM-DiT 双 stream + AdaLN-Zero

网络 $f_\theta$ 是一个 Multi-Modal Diffusion Transformer (MM-DiT)，灵感来自 Stable Diffusion 3 / Flux。参考：https://arxiv.org/abs/2403.03206

**Tokenization**：每个 cell 是一个 token，整个 batch $B \in \mathbb{R}^{m \times |\mathcal{G}|}$ 经过 linear input projection 变成 $h \in \mathbb{R}^{m \times d}$，$d$ 是 model dim。两个 stream：
- $h_{\text{pert}} \in \mathbb{R}^{m \times d}$：noised perturbed tokens
- $h_{\text{ctrl}} \in \mathbb{R}^{m \times d}$：control tokens

**Conditioning**：timestep $t$ 通过 sinusoidal embedding + MLP 得 $e_t \in \mathbb{R}^d$。Metadata $(c, \tau)$ 通过 CovEncoder：
- Cytokine perturbation：用 **ESM2** protein language model embedding
- Drug perturbation：用 **ChemBERTa** embedding + dose embedding（剂量离散化）
- Gene perturbation：用 **GenePT** 或 LLM 生成的 gene summary embedding
- Cell type：LLM 生成的 cell type summary embedding

所有 embedding concat + linear project 到 $d$，跟 $e_t$ concat 后过 MLP 得 global conditioning vector $s \in \mathbb{R}^d$。

**MM-DiT block**（每层都重复一遍）：
1. AdaLN-Zero modulation：用 $s$ 生成 scale $\gamma$, shift $\beta$, residual gate $g$（attention 和 MLP 各一组，pert 和 ctrl 各一组）：
   $$(\beta_\star^{\text{attn}}, \gamma_\star^{\text{attn}}, g_\star^{\text{attn}}, \beta_\star^{\text{mlp}}, \gamma_\star^{\text{mlp}}, g_\star^{\text{mlp}}) = f_{\theta, \star}(s), \quad \star \in \{\text{pert}, \text{ctrl}\}$$
   AdaLN: $\text{Mod}(h; \beta, \gamma) = \gamma \odot \text{LN}(h) + \beta$。Zero-init 让 block 初始化为 identity，稳定 deep diffusion training。

2. **Joint attention**：两个 stream 在 feature dim 上 concat：
   $$u = [\tilde{h}_{\text{pert}}^{\text{attn}}; \tilde{h}_{\text{ctrl}}^{\text{attn}}] \in \mathbb{R}^{m \times 2d}$$
   然后过一个 shared MSA，attention 在 concat 后的 $2m$ tokens 上算（其实 paper 写的是 split 后的 $u'$ 切回两半，这里有点细节要小心）：
   $$u' = \text{MSA}(u), \quad [\Delta h_{\text{pert}}; \Delta h_{\text{ctrl}}] = \text{Split}(u')$$
   **这是 control 信息流向 perturbed stream 的关键机制**：attention 让 perturbed tokens 可以 query control tokens，学到"perturbation 是如何偏离 control 分布的"。

3. Gated residual：
   $$h_\star \leftarrow h_\star + g_\star^{\text{attn}} \odot \Delta h_\star$$

4. 独立 MLP sublayer（每个 stream 自己的），同样 AdaLN-Zero + gated residual：
   $$h_\star \leftarrow h_\star + g_\star^{\text{mlp}} \odot \text{MLP}_\star(\text{Mod}(h_\star; \beta_\star^{\text{mlp}}, \gamma_\star^{\text{mlp}}))$$

**Denoising head**：只有 perturbed stream $h_{\text{pert}}$ 接 denoising head（linear projection 回 $\mathbb{R}^{|\mathcal{G}|}$ + ReLU 强制非负），接 diffusion loss。Control stream 不接 loss，纯做 conditioning。这点设计很关键——control 不是要被 reconstruct 的对象，而是给 perturbed 提供参照系。

参考 DiT 原文：https://arxiv.org/abs/2212.09748

---

## 9. Self-conditioning + CFG + DDIM sampling

**Self-conditioning**（Eqn. 33）：训练时以概率 $p_{\text{sc}}$ 先跑一次 forward 拿 stop-gradient 预测 $\bar{B}_\theta = \text{sg}(f_\theta(B_t, t; y))$，再把它和 $B_t$ concat 喂回去：
$$B_\theta = f_\theta([B_t, \bar{B}_\theta], t; y)$$
这让模型能"看到自己上一轮的猜测"，类似 iterative refinement，对 diffusion 训练稳定性有帮助。参考 self-conditioning 原文：https://arxiv.org/abs/2208.04202

**CFG dropout**：训练时以概率 $p_{\text{drop}}$ 把 metadata $(c, \tau)$ mask 掉（用 null token 替换），但 control batch $B_{\text{ctrl}}$ **永远不 drop**——因为 control 是"matched"的，必须保留。采样时在 $\varepsilon$-space 做 guidance：
$$\hat{\varepsilon}_{\text{cfg}} = (1 + w)\hat{\varepsilon}_c - w\hat{\varepsilon}_u$$
$w$ 是 guidance scale，$\hat{\varepsilon}_c$ 和 $\hat{\varepsilon}_u$ 分别是 conditional / unconditional 估计。注意是在 $\varepsilon$-space 做的，不是 $x_0$-space。

**Sampling**：DDIM with $\eta=0$（deterministic），$K=100$ 步快速采样或 $K=1000$ 全步采样：
$$B_\theta^{(k)} = f_\theta(B_{t_k}, t_k; y), \quad \hat{\varepsilon}^{(k)} = \frac{B_{t_k} - \sqrt{\alpha_{t_k}} B_\theta^{(k)}}{\sqrt{1 - \alpha_{t_k}}}, \quad B_{t_{k+1}} = \sqrt{\alpha_{t_{k+1}}} B_\theta^{(k)} + \sqrt{1-\alpha_{t_{k+1}}}\hat{\varepsilon}^{(k)}$$

DDIM 原文：https://arxiv.org/abs/2010.02502

CFG 原文：https://arxiv.org/abs/2207.12598

---

## 10. Marginal pretraining：用 CellxGene 60M cells 学 prior

Perturbation 数据集虽然大（Tahoe100M 1 亿 cell），但 cell type 和 batch 多样性远不如 CellxGene（662 cell types, 10887 batches, 60M cells）。所以 paper 加了一个**两阶段**：

**Stage 1: Marginal pretraining**。先学 unperturbed cell distribution：
$$
D_c \sim \mathcal{F}_\theta(\cdot \mid c)
$$
即给定 context $c$，generate 一个 cell population（不需要 perturbation label）。Control stream 全设为 0，conditioning 只用 $c$。数据混了所有 perturbation 数据集的训练 cell + CellxGene 60M cells。

**Gene space 统一**：merge-then-select 策略，把 CellxGene 1139 个 dataset 合并成 23 个 composite，每个选 top-2000 HVG，取 union 得 10045 genes。再 union 三个 perturbation dataset 的 HVG 得 **12626 genes** 的 pretraining vocabulary。Table 3 显示 PBMC 和 Tahoe100M 共享 >98%，Replogle 共享 ~46%。

**Stage 2: Fine-tuning**。在 perturbation 数据上 fine-tune，加 control stream + perturbation label。

Pretraining 提供的 prior 让 model 在 zero-shot（不给 perturbation label）也能产生合理 R²（Fig. 6），说明 perturbation shift 部分沿着 marginal manifold 的已有方向，而非完全 random。这跟 ImageNet pretraining → downstream fine-tune 的故事结构一样。

参考 CellxGene: https://cellxgene.cziscience.com/

Geneformer 在 CellxGene 上 pretrain 的范例：https://www.nature.com/articles/s41586-023-06139-9

---

## 11. 实验数据细看

三个 benchmark dataset（Table 2）：

| Dataset | #Cells | #Pert | #Cell Types | #Batches | Perturbation type |
|---|---|---|---|---|---|
| PBMC | 9.7M | 90 | 18 | 12 | cytokine |
| Tahoe100M | 101.2M | 1137 | 50 | 14 | drug |
| Replogle | 0.6M | 2023 | 4 | 56 | gene (CRISPRi) |

Replogle 是 low-data regime：平均每个 gene perturbation 只有 ~300 cells，是天然的 few-shot testbed。

**主要结果**（Fig. 3, Table 4）：
- PBMC 上 PerturbDiff (From Scratch) 几乎所有 14 metrics 最好；PerturbDiff (Finetuned) 在 DE-related metrics 更好，但 R²/AUROC 上略掉。
- Tahoe100M 上 From Scratch 整体最强，Finetuned 在 average expression accuracy 上略低（说明 finetune 增强的是 distribution shift 而非 averaged accuracy）。
- Replogle 上 From Scratch 略输 STATE，但 Finetuned 追平 STATE。原因是 Replogle 数据少，pretraining 的 prior 帮助大。

**Per-perturbation scatter plot**（Fig. 4, 15, 16）：win rate > 96% on DE metrics，意味着 gain 不是几个 outlier 拉起来的，是 systematic 的。

**DE recovery**（Fig. 5）：用 Wilcoxon rank-sum test $p_{\text{adj}} < 0.05$ 判 DE gene，plot $-\log_{10}(p_{\text{adj}})$ 分布。PerturbDiff 把 DE 和 non-DE 分得很开，STATE 把几乎所有 gene 都标成 DE（over-confidence）。这是一个 qualitative 的胜利——MMD-based training 让 model 学到了 DE 的"signature"而非泛化地认为所有 gene 都被 perturb。

**Scaling study**（Fig. 7, 20）：模型 3 个 size（small/medium/large，medium 114M, large 239M）。Medium 最好最稳，large 在中等 compute 后开始 degrade（overfitting）。**Scaling is non-monotonic**，跟 image diffusion 里 Nichol & Dhariwal 2021 看到的现象一致：https://arxiv.org/abs/2102.09672。这是个有趣的 negative result，说明 perturbation modeling 不像 image generation 那样吃 scaling。

**Low-data**（Fig. 8, 19）：PBMC subsample 到 1% 和 5%。Finetuned 全面好于 From Scratch，gap 在 1% 时更大。Finetune 收敛也快得多，曲线稳。

**Zero-shot**（Fig. 6, 17, 18）：pretraining step 增加时 R² 单调上升，但 DE-related metrics 呈 U-shape——早期 pretrain 反而暂时拉低下游 transfer，后期才恢复。Intuition：早期 marginal pretrain 学的是 cell type identity 之类的"非 perturbation-relevant"结构，可能临时压住 task-relevant direction；后期 manifold 组织好了才能复用。

---

## 12. Score matching 视角（App. B.4）

Paper 还给了一个有意思的理论联系。把 MMD loss 在 reference batch $\mathbf{x}$ 附近做 Taylor 展开：

$$
\text{MMD}_k^2(\mathbf{x}', \mathbf{x}) = \|\mathbf{x}' - \mathbf{x}\|_H^2 + o(\|\mathbf{x}' - \mathbf{x}\|_2^2)
$$

其中 $H := \nabla_{\mathbf{x}'}^2 \text{MMD}_k^2(\mathbf{x}', \mathbf{x})|_{\mathbf{x}'=\mathbf{x}}$ 是 Hessian，$\|x\|_H^2 := x^\top H x$。所以 MMD 局部是**matrix-weighted MSE**，weighting matrix $H$ 取决于 kernel 和 reference population 的 local geometry。这玩意儿不像普通 MSE 在所有方向 uniform 加权，而是强调 kernel encode 的 informative 高阶统计量方向。

**Prop. B.15（Augmented Denoising Score Matching）**：证明在 $H$-norm 下做 score matching 等价于 denoising score matching：

$$
\arg\min_\theta \mathbb{E}_{\tilde{x}}\big[\|\nabla \log p(\tilde{x}) - f(\tilde{x})\|_H^2\big] = \arg\min_\theta \mathbb{E}_{\tilde{x}, x}\big[\|\nabla_{\tilde{x}} \log p(\tilde{x} \mid x) - f(\tilde{x})\|_H^2\big]
$$

证明思路是把 L(f) 展开，把 $\nabla \log p(\tilde{x})$ 拆成 $\int \nabla_{\tilde{x}} \log p(\tilde{x} \mid x) p(x \mid \tilde{x}) dx$，然后 complete the square。

Intuition：MMD-based training ≈ score matching under non-Euclidean geometry induced by kernel。这给 MMD loss 一个 score-based diffusion 的解释。

Score-based SDE 原文：https://openreview.net/forum?id=PxTIG12RRHS

---

## 13. 评估指标全览（App. C.2）

Cell-Eval framework 14 个 metric 分两组：

**Group 1: Averaged expression accuracy**（across cells 维度）
- $R^2$: 决定系数，预测 vs ground truth pseudobulk 的方差解释比例。
- PDCorr: Pearson correlation of $\Delta \hat{\mathbf{x}}$ vs $\Delta \mathbf{x}$（perturbation effect 的 relative shift）。
- MAE, MSE: $\|\Delta \hat{\mathbf{x}} - \Delta \mathbf{x}\|$ 的 L1/L2。
- PDS$_{L1}$, PDS$_{L2}$, PDS$_{\cos}$: Perturbation Discrimination Score，衡量 prediction 是否能从所有 ground truth 中 distinguish 自己的 perturbation，相当于 retrieval metric。Random level 是 0.5，完美是 1。

**Group 2: Biologically meaningful differential patterns**（across genes 维度，更重要）
- DEOver: top-k DE gene overlap。
- DEPrec: top-k DE gene precision。
- DirAgr: DE gene 的 sign agreement（up/down 预测对不对）。
- LFCSpear: log-fold-change Spearman correlation。
- AUROC, AUPRC: DE gene classification 的 ROC/PR 曲线下面积。
- ES: Effect size Spearman，DE gene 数量的相对排序。

DE-related metrics 是核心，因为 perturbation modeling 的灵魂是 recover biologically meaningful gene response 而非 match average。PerturbDiff 的优势主要集中在 DE metrics，这正好说明 distribution-level modeling 抓住了 cell-population 级别的 shift pattern。

Cell-Eval 来自 STATE paper: https://www.biorxiv.org/content/10.1101/2025.06

---

## 14. 跟相关方法的差异化定位

| 方法 | Random variable | 机制 | 缺陷 |
|---|---|---|---|
| Linear | cell | regression | 学平均 |
| GEARS | cell | GNN on gene-gene graph | random pairing |
| scGPT | cell | fine-tune foundation model | random pairing |
| scGen | cell latent | VAE latent shift | single-cell stochasticity |
| CPA | cell latent | VAE + adversarial disentangle | single-cell stochasticity |
| Squidiff | cell | diffusion on single cell | 单 distribution 假设 |
| CellOT | distribution | neural optimal transport | deterministic map |
| STATE | distribution | transformer + MMD align | deterministic objective |
| CellFlow | distribution | flow matching | 单 distribution 假设 |
| Unlasting | distribution | diffusion bridge | stochastic transition |
| **PerturbDiff** | **distribution of distributions** | **RKHS diffusion** | **(本文)** |

PerturbDiff 的独特之处是**显式建模 distributional variability**：同一个 $(c, \tau)$ 对应一族 plausible distributions，由 unobserved latent factors 驱动。MMD 在 STATE 里是外部加的 alignment objective，在 PerturbDiff 里是 RKHS diffusion 几何的自然产物。这个理论 grounding 是 paper 的最大亮点。

参考 Squidiff: https://www.nature.com/articles/s41592-025-02613-0

参考 CellOT: https://www.nature.com/articles/s41592-023-02101-8

参考 CellFlow: https://www.biorxiv.org/content/10.1101/2025.04

参考 Unlasting: https://arxiv.org/abs/2506.21107

参考 GEARS: https://www.nature.com/articles/s41587-023-01905-6

参考 scGPT: https://www.nature.com/articles/s41592-024-02201-7

参考 CPA: https://www.embopress.org/doi/10.1038/s44319-023-00006-x

---

## 15. 个人 takeaways 和值得琢磨的方向

1. **RKHS diffusion 是个通用 paradigm**。只要你能把对象 embed 到 RKHS（distributions, graphs, trajectories, point clouds...），都可以套这套推导。MMD loss 自然掉出来，不用 external align。这可能启发 functional / set / population-level 的 generative modeling 一波工作。

2. **First-order linearization trick**（Sec. 4.4 + App. B.1.7）值得单独挖。在 function space 上跑 diffusion，理论上需要 sample function-valued Gaussian。这里的"在原 space 加 Euclidean noise 再重新 embed"是 general trick，可以推广到任何 differentiable feature map $\phi: \mathbb{R}^d \to \mathcal{H}$ 的场景。

3. **Marginal pretraining as prior** 这个 story 跟 LLM / vision foundation model 一模一样。Perturbation data 量级是 10M-100M，但 context 多样性差。CellxGene 60M + 662 cell types 提供 prior。Pretraining 效果在 low-data regime 显著（Replogle 上 From Scratch 输 STATE，Finetune 反超），这暗示"perturbation 是 marginal manifold 上的 structured shift"。

4. **Non-monotonic scaling** 是个值得 alert 的发现。Perturbation modeling 不像 image generation 那样吃 brute-force scaling。可能因为信号弱、sparsity 高、metric sensitive。未来工作可能要找perturbation-specific 的 scaling law。

5. **Joint attention 双 stream** 这个架构 choice 很 elegant。Control stream 不接 loss 纯做 conditioning，attention 让 perturbed tokens "see" control tokens——这跟 control net 的 conditioning 哲学一脉相承但更紧密。可以考虑 cross-attention only 的 ablation 看是不是必要的 bidirectional。

6. **MMD vs MSE 在 sparse 数据上的对比** 是个 actionable insight。任何 sparse high-dim generative task（gene expression, point cloud, sparse rewards）都可能受益于 distribution-level loss 替代 element-wise MSE。Paper 的 ablation（Fig. 9）证明了这点。

7. **Zero-shot U-shape** 现象有意思。Pretraining 早期可能暂时 hurt downstream，这跟 LLM pretrain 中观察到的 catastrophic forgetting 类似。可能要 task-aware pretraining schedule 或 curriculum learning。

可能延伸的联想：
- 把 RKHS diffusion 搬到 protein design（distribution of structures）、drug response prediction across patients、single-cell multi-omics integration。
- 在 conditional generation 上做 kernel-based classifier-free guidance（kernel 在 latent space 而非 input space）。
- 跟 Schrödinger Bridge / Iterative Proportional Fitting 对比，作为 population-level transition 的 stochastic alternative。
- Energy distance kernel 之外的 kernel 选择（Gaussian, MMD with deep kernel），可能在更复杂分布上有优势。
- Diffusion bridge（Unlasting）和 RKHS diffusion 的合体：在 Hilbert space 里做 bridge，可能给 in-between perturbation interpolation 更好的几何。

---

## 16. 总结：怎么一句话讲清这篇 paper

PerturbDiff 把 single-cell perturbation prediction 从"学一个 cell → cell map"重新表述为"学一个 distribution → distribution 的 stochastic map"，通过把 cell distributions embed 到 RKHS 里的单个点，定义了一个直接在 distribution 空间上跑的 DDPM，loss 自然成为 MMD，从理论上 ground 了"用 MMD 做 population alignment"这件事，并在三个大规模 benchmark 上拿下 SOTA，特别是对 biologically meaningful differential expression 的 recovery 显著优于之前所有方法。

参考资源汇总：
- Paper (arxiv 待发)：https://arxiv.org/abs/2502.13791 (估测)
- DDPM: https://arxiv.org/abs/2006.11239
- Function space diffusion: https://arxiv.org/abs/2212.00886
- Kernel mean embedding review: https://arxiv.org/abs/1605.00788
- MMD two-sample test: https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
- Energy distance ≡ RKHS embedding: https://projecteuclid.org/journals/Annals-of-Statistics/volume-41/issue-3/Equivalence-of-distance-based-and-RKHS-based-statistics-in-hypothesis-testing/10.1214/13-AOS1160.full
- DiT (AdaLN-Zero): https://arxiv.org/abs/2212.09748
- MM-DiT (SD3): https://arxiv.org/abs/2403.03206
- DDIM: https://arxiv.org/abs/2010.02502
- CFG: https://arxiv.org/abs/2207.12598
- Self-conditioning: https://arxiv.org/abs/2208.04202
- Score SDE: https://openreview.net/forum?id=PxTIG12RRHS
- ESM2: https://www.science.org/doi/10.1126/science.ado2588
- ChemBERTa: https://arxiv.org/abs/2010.09885
- GenePT: https://www.nature.com/articles/s41551-024-01284-6
- PBMC dataset: https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment
- Tahoe100M: https://www.biorxiv.org/content/10.1101/2025.02
- Replogle: https://www.nature.com/articles/s41588-025-02035-7
- CellxGene: https://cellxgene.cziscience.com/
- Geneformer: https://www.nature.com/articles/s41586-023-06139-9
- scFoundation: https://www.nature.com/articles/s41592-024-02305-7
- STATE: https://www.biorxiv.org/content/10.1101/2025.06
- CellOT: https://www.nature.com/articles/s41592-023-02101-8
- CellFlow: https://www.biorxiv.org/content/10.1101/2025.04
- Squidiff: https://www.nature.com/articles/s41592-025-02613-0
- Unlasting: https://arxiv.org/abs/2506.21107
- GEARS: https://www.nature.com/articles/s41587-023-01905-6
- scGPT: https://www.nature.com/articles/s41592-024-02201-7
- Virtual Cell review (Bunne et al.): https://www.cell.com/cell/fulltext/S0092-8674(24)01305-9
- Improved DDPM (scaling non-monotonic): https://arxiv.org/abs/2102.09672

如果你想我深挖某一块——比如 score matching 那个 Prop. B.15 的证明细节、joint attention 的算力分析、pretraining U-shape 的理论解释、或者跟 Schrödinger Bridge 的对比——告诉我，我可以再展开。
