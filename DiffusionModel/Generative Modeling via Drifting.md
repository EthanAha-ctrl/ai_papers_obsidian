---
source_pdf: Generative Modeling via Drifting.pdf
paper_sha256: 4c6de150102edf01fdde787a2bc6d20962cd2021769db09e41c4ac7fb4ab852f
processed_at: '2026-08-04T14:11:51-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Drifting Model

Karpathy，行，我把刚才那堆公式翻译成"跟同事在白板前聊天"的版本。

---

## 一句话版本

**别在推理的时候迭代了，训练本身就是迭代，用它来演化分布就完事了。**

---

## 背景先对齐一下

Diffusion models 干啥的？给它一个noise $\epsilon$，它学一个网络 $f$，把 noise 映射成 image。但一次映射太难学了，所以它把这个映射拆成 250 步小映射，推理的时候一步一步走（像爬楼梯一样，每步走一点点）。

Flow Matching 同理，也是推理时迭代，只是数学上更干净点。

这些方法的核心思路是：**复杂映射太难一次学到，分解成 250 步简单映射，推理时一步步走。**

Drifting Model 说：凭啥非要推理时走？**SGD 训练本来就是 100 万步的迭代，把这个迭代当作分布演化不就完了？**推理时直接一步出图。

---

## 核心直觉：一个超级简单的物理类比

想象你有一群粒子（generated samples），目标是让它们的分布跟 data 分布重合。

每个粒子同时受两个力：

1. **吸引力** $\mathbf{V}_p^+$：被 real data 吸引。你周围的 real data 像 "质心"，你往那个质心方向走。
2. **排斥力** $\mathbf{V}_q^-$：被其他 generated samples 排斥。你周围其他 fake samples 也是个 "质心"，但你要远离它们（避免大家挤在一起）。

这两个力相减就是 drifting field $\mathbf{V}$：

$$\mathbf{V}(\mathbf{x}) = \mathbf{V}_p^+(\mathbf{x}) - \mathbf{V}_q^-(\mathbf{x})$$

**关键点**：当 generated 分布 = data 分布时，这两个 "质心" 是同一个东西，吸引力 = 排斥力，力平衡，粒子不动了。

这就是 equilibrium。

---

## 为什么 anti-symmetry 这么重要

paper 里花了一整页强调这个，我一开始觉得是数学洁癖，后来发现是真的 physical core。

你看 Table 1 的 ablation：

| 改动 | FID |
|------|-----|
| 正常 anti-symmetric（$V^+ - V^-$） | 8.46 |
| 1.5倍吸引（$1.5V^+ - V^-$） | 41.05 |
| 只吸引不排斥（$V^+$） | 177.14 |

**只吸引不排斥 → FID 177**，完全是灾难。为啥？

因为只有吸引的话，所有粒子全往 data mode 跑，最后塌缩成一个点（mode collapse）。**排斥力的作用就是让粒子之间互相撑开，覆盖整个分布**。

而且必须严格等量。你给吸引力多加 0.5 倍权重，平衡点就偏了——equilibrium 不再是 $p=q$，而是某个 weird 中间状态。

类比一下：这是弹簧系统。两个弹簧刚度必须一样，不然平衡位置就偏了。

---

## 训练 loss 怎么写

到这里就很简单了。我们有 $\mathbf{V}(\mathbf{x})$ 告诉每个 sample 应该往哪走，那 loss 就是：

```python
target = x + V(x)        # 告诉网络：你输出应该往这走
target = stopgrad(target)  # 但 target 本身不参与梯度
loss = mse(x, target)       # 让网络输出逼近 target
```

就这么几行。

**为什么 stopgrad？** 因为 $\mathbf{V}$ 依赖整个 batch 的统计（它是个分布级别的量），直接 backprop through 分布不 trivial。stopgrad 让它变成 "把当前 sample 拉向 target" 的简单 regression。

这跟 BYOL（[Chen & He 2021](https://arxiv.org/abs/2011.10566)）和 iCT（[Song & Dhariwal 2023](https://arxiv.org/abs/2310.14189)）是同一个 trick：frozen target + stop-gradient，避免 representation collapse。

---

## 一个我觉得很妙的设计：feature space drifting

直接在 pixel space 算 kernel 不行——pixel 距离不反映语义。两只猫像素差异可能比猫和狗还大。

所以把 drift 搬到 feature space：

$$\text{loss} = \|\phi(\mathbf{x}) - \text{stopgrad}(\phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x})))\|^2$$

用预训练的 self-supervised encoder $\phi$（MoCo / SimCLR / 他们自己训的 latent-MAE）。

这跟 perceptual loss 长得像但**根本不同**：

- Perceptual loss: $\|\phi(\mathbf{x}) - \phi(\mathbf{x}_{\text{target}})\|^2$，需要给 $\mathbf{x}$ 配对一个 target $\mathbf{x}_{\text{target}}$
- Drifting loss: target 是 $\phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x}))$，由 kernel-based drift 计算，**不需要配对**

这是 distribution-level matching vs sample-level regression 的区别。Drifting loss 不需要"哪只猫对应哪只猫"，只要"这群 fake cats 整体往 real cats 群靠"。

---

## 然后是结果

**ImageNet 256×256 latent space:**
- 之前最好的 1-NFE 方法 iMeanFlow-XL/2（610M params）：FID 1.72
- Drifting L/2（463M params）：**FID 1.54**

更小模型，更好结果。

**Pixel space（不用 VAE，直接出 256×256 图）:**
- 之前最好的 1-NFE 是 GAN（StyleGAN-XL 166M）：FID 2.30
- Drifting L/16（464M）：**FID 1.61**

跟 PixelDiT/16 持平，但 PixelDiT 用 200 步推理，Drifting 用 1 步。FLOPs 上 Drifting 是 87G，StyleGAN-XL 是 1574G，**18倍效率差**。

---

## 一个特别有意思的发现：mode collapse 难以发生

paper 里 Figure 3 有个 toy experiment：target distribution 是双峰的，把 generated distribution 初始化成塌缩到其中一个峰上。

正常 GAN 的话，这种 init 就死了——梯度信号让所有 samples 都往那个 mode 挤，越挤越死。

但 Drifting Model 能恢复。机制是：

1. 一开始 q 全在 mode A，所以 $\mathbf{V}_q^-$（排斥力）让 samples 互相排斥，往外散
2. mode B 的 real data 提供 $\mathbf{V}_p^+$ 吸引力，把散开的 samples 拉过去
3. 最终 q 覆盖两个 mode

**排斥力天然 anti-collapse**。这跟 GAN 形成鲜明对比——GAN 需要各种 trick（minibatch discrimination、diversity loss）来防 collapse，drifting 在数学结构上就 avoid 了。

---

## CFG 这块也很优雅

标准 CFG（[Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)）：推理时跑两次网络（conditional + unconditional），然后外推。所以是 2-NFE。

Drifting 的 CFG：**训练时**就把 unconditional samples 当作额外的 negatives 喂进去，让网络直接学一个 "已经外推过的" distribution。推理时**仍然是 1-NFE**。

形式上：

$$q_\theta(\cdot|c) = \alpha \cdot p_{\text{data}}(\cdot|c) - (\alpha-1) \cdot p_{\text{data}}(\cdot|\emptyset)$$

这就是标准 CFG 的线性组合形式，但**省掉推理时那次无条件 forward**。

---

## 跟之前方法的根本区别

| 方法 | 在哪迭代 | 用什么迭代 | 推理 NFE |
|------|----------|-----------|----------|
| Diffusion | 推理时 | ODE/SDE solver | 250+ |
| Flow Matching | 推理时 | ODE solver | 50-250 |
| Consistency Model | 不迭代 | distill multi-step 成 1-step | 1 |
| MeanFlow | 不迭代 | distill from flow formulation | 1 |
| GAN | 不迭代 | adversarial training | 1 |
| **Drifting** | **训练时** | **SGD 迭代 = 分布演化** | **1** |

Consistency / MeanFlow 本质还是"从 diffusion/flow 蒸馏过来"，**保留了 SDE/ODE 的数学结构**。

Drifting **完全脱离 SDE/ODE 框架**，直接定义向量场驱动分布演化。这个 conceptually clean 的程度我觉得是 Kaiming He 这几年一脉相承的极简主义（MoCo、BYOL、MAE、JiT 都是这风格）。

---

## 跟 MMD 的关系（paper Appendix C.2 我觉得很关键）

[MMD-based GAN](https://arxiv.org/abs/1505.03906)（2015 年的 paper）其实也是用 kernel 衡量两个分布差异，最小化 MMD。Drifting 在某种特殊情况下能约化成 MMD。

但 paper 指出关键区别：

1. **MMD 用未归一化 kernel**：所以 mean-shift interpretation 失效
2. **Drifting 用归一化 kernel** $\tilde{k} = k / Z$：让双 kernel 耦合的 Eq. (11) 成立，这是 anti-symmetry 的基础

paper 直接说："**我们用 MMD framework 在 ImageNet 上跑不出 reasonable 结果**"。

这是让我比较意外的——MMD 在 2015 年就不行了，但大家以为是 kernel 选择问题。Drifting 表明：**问题在于 framework 本身缺少归一化和 V-centric 视角**。

---

## 我觉得最 "Aha" 的几个瞬间

### 1. SGD 本身就是迭代器

这话说出来好像很 trivial，但仔细想想：diffusion 的 250 步推理跟 SGD 的百万步训练，本质上都是"iterative refinement"。那为什么不把推理的迭代性吸收进训练？

这让我想起 [Neural Turing Machine](https://arxiv.org/abs/1410.5401) 当年的 insight：把"外部 memory"变成"internal state"。Drifting 是把"外部迭代"变成"训练动态"。

### 2. Stop-gradient 的反复出现

BYOL 用 stop-grad 避免 collapse，iCT 用 stop-grad 做 consistency target，Drifting 用 stop-grad 处理分布级别 target。

这个 trick 出现得太频繁了，暗示着**深度学习里很多 "self-referential" 训练**都需要 stop-grad 来稳定。可能是个值得理论分析的通用现象。

### 3. Anti-symmetry = 物理平衡

$$\mathbf{V}_{p,q} = -\mathbf{V}_{q,p} \Rightarrow p=q \text{ 时 } \mathbf{V}=0$$

这跟物理学里"作用力 = 反作用力"完全同构。牛顿第三定律的机器学习版本。

---

## 潜在的局限（我自己的看法）

1. **依赖强 SSL feature encoder**：paper 明说不用 feature extractor 在 ImageNet 上跑不出结果。这跟 GAN 类似——GAN 也需要各种 auxiliary classifier/perceptual loss。所以本质上 drifting 的高质量生成是"借用"了 SSL encoder 的 semantic structure。

2. **训练时长**：1280 epochs 才能到 1.54 FID。Diffusion 一般 1000 epochs 左右。所以训练成本并不便宜。

3. **理论 converses 不完全**：paper 在 Appendix C.1 给了 "V=0 ⇒ p=q" 的 heuristic argument，但依赖 basis expansion 假设和线性独立性。严格证明还缺。

4. **Batch size 需求大**：effective batch 8192，而且需要 N_pos=128, N_neg=128 的大 batch 来准确估计 drifting field。这跟 contrastive learning 一样，对硬件要求高。

---

## 一句话总结

**Drifting Model = mean-shift 算法 + anti-symmetric equilibrium + SGD 训练动态 + SSL feature space**

把传统生成模型 "推理时迭代" 的范式，翻转成 "训练时演化"，通过物理直觉清晰的 attraction-repulsion 平衡达到 distribution matching。结果上 1.54 FID on ImageNet 256×256，1-NFE，比之前最好的 1-step 方法还好。

**这 paper 的价值不在于 FID 数字，在于 conceptual cleanliness**。它给了生成建模一个 "outside the SDE/ODE box" 的可行路径，跟 Kaiming 之前的 MoCo/MAE 一样，是 "做简单的事，但做对" 的风格。

---

Links:
- Drifting Model paper: https://arxiv.org/abs/2505.22689 (假设)
- Mean-shift algorithm (Cheng 1995): https://ieeexplore.ieee.org/document/485468  
- BYOL: https://arxiv.org/abs/2006.07733
- Consistency Models: https://arxiv.org/abs/2303.01969
- MeanFlow: https://arxiv.org/abs/2505.13447
- iMeanFlow: https://arxiv.org/abs/2512.02012
- MMD GAN: https://arxiv.org/abs/1505.03906
- CFG: https://arxiv.org/abs/2207.12598
- MAE: https://arxiv.org/abs/2111.06377
- MoCo: https://arxiv.org/abs/1911.05722
- StyleGAN-XL: https://arxiv.org/abs/2203.01856
- Karpathy 的 random thoughts: https://karpathy.ai/

---

# Generative Modeling via Drifting 深度解析

Karpathy好，这篇由Mingyang Deng、He Li、Tianhong Li、Yilun Du和Kaiming He合作的paper，提出了一个**概念上非常优雅的generative modeling新范式**。它把"训练迭代"本身当作分布演化的机制，绕过了diffusion/flow models在inference time的迭代需求。我来把它拆开讲透。

## 1. 核心Paradigm Shift：Training-Time Pushforward Evolution

### 1.1 问题设定

考虑一个神经网络 $f: \mathbb{R}^C \mapsto \mathbb{R}^D$，输入是noise $\epsilon \sim p_\epsilon$（任意分布，通常是Gaussian），输出 $\mathbf{x} = f(\epsilon) \sim q$，其中：

$$q = f_\# p_\epsilon \tag{1}$$

这里 $f_\#$ 是 **pushforward operator**：它把输入分布 $p_\epsilon$ 通过 $f$ 映射到输出分布 $q$。生成建模的目标是找到 $f$ 使得 $f_\# p_\epsilon \approx p_{\text{data}}$。

### 1.2 关键insight

Diffusion/Flow models（[Sohl-Dickstein 2015](https://arxiv.org/abs/1503.03585), [Lipman 2022](https://arxiv.org/abs/2210.02747)）的做法是：在**inference time**通过Euler solver迭代 $\mathbf{x}_{i+1} = \mathbf{x}_i + \Delta \mathbf{x}_i$，每步都eval网络。

Drifting Models反过来：**既然SGD本身就是迭代的**，training过程自然产生一序列 $\{f_i\}$，对应一序列 $\{q_i\}$，其中 $q_i = [f_i]_\# p_\epsilon$。把训练迭代直接当作分布演化机制，inference只需一次forward。

这是一个很beautiful的视角转换。让我画个对比图：

```
Diffusion/Flow:    [Training: learn velocity field] -> [Inference: iterate N steps]
Drifting Model:   [Training: iterate to evolve distribution] -> [Inference: 1 step]
```

直觉上，diffusion把复杂映射分解成一串简单映射（在inference时），drifting则把复杂映射的复杂性"吸收"进SGD的训练动态里。

## 2. Drifting Field的数学构造

### 2.1 Drifting update rule

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \mathbf{V}_{p,q_i}(\mathbf{x}_i) \tag{2}$$

变量解释：
- $\mathbf{x}_i = f_i(\epsilon) \sim q_i$：训练第 $i$ 步的generated sample
- $\mathbf{V}_{p,q_i}(\cdot): \mathbb{R}^d \to \mathbb{R}^d$：drifting field，一个向量场
- 下标 $p, q_i$：表明field依赖于target distribution $p$（即 $p_{\text{data}}$）和当前分布 $q_i$
- $\mathbf{x}_{i+1} \sim q_{i+1}$：drift后样本所属的新分布

### 2.2 Anti-symmetry property（关键的equilibrium条件）

$$\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x}), \quad \forall \mathbf{x} \tag{3}$$

**推论**：$q = p \Rightarrow \mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0}, \forall \mathbf{x}$

**证明**（一行）：当 $q=p$，$\mathbf{V}_{p,q} = \mathbf{V}_{q,p} = -\mathbf{V}_{p,q}$（最后一步用anti-symmetry），所以 $\mathbf{V}_{p,q} = \mathbf{0}$。

这个anti-symmetry是整个框架的"锚点"——它保证当生成分布匹配数据分布时，drift自然归零，达到equilibrium。

直觉上，这类似于**热力学平衡**或者**力学中的势能极小点**：吸引力（被data吸引）和排斥力（被其他generated samples排斥）达到平衡。

### 2.3 Fixed-point training objective

在equilibrium处 $\hat{\theta}$：

$$f_{\hat{\theta}}(\epsilon) = f_{\hat{\theta}}(\epsilon) + \mathbf{V}_{p, q_{\hat{\theta}}}(f_{\hat{\theta}}(\epsilon)) \tag{4}$$

这是一个fixed-point equation。把训练迭代转化成loss：

$$\mathcal{L} = \mathbb{E}_\epsilon \left[ \left\| \underbrace{f_\theta(\epsilon)}_{\text{prediction}} - \underbrace{\mathrm{stopgrad}\left( f_\theta(\epsilon) + \mathbf{V}_{p, q_\theta}(f_\theta(\epsilon)) \right)}_{\text{frozen target}} \right\|^2 \right] \tag{6}$$

变量：
- $f_\theta(\epsilon)$: 当前网络输出（prediction）
- $\mathrm{stopgrad}(\cdot)$: 冻结梯度，target不参与backprop（借鉴自[BYOL Chen & He 2021](https://arxiv.org/abs/2011.10566)和[iCT Song & Dhariwal 2023](https://arxiv.org/abs/2310.14189)）
- $\mathbf{V}_{p, q_\theta}(f_\theta(\epsilon))$: 在generated point处计算的drifting field
- $q_\theta = [f_\theta]_\# p_\epsilon$: 当前参数对应的pushforward分布

注意：loss数值等于 $\mathbb{E}_\epsilon[\|\mathbf{V}(f(\epsilon))\|^2]$，即**drifting field的squared norm**。stopgrad的trick很关键：因为 $\mathbf{V}$ 依赖于整个分布 $q_\theta$（一个batch的统计），直接backprop through distribution是nontrivial的。stopgrad把这个变成"让 $f_\theta(\epsilon)$ 向 $\mathbf{x} + \mathbf{V}(\mathbf{x})$ 移动"的简单回归问题。

## 3. Drifting Field的具体设计：Mean-ShiftInspiration

### 3.1 Attraction + Repulsion分解

借鉴mean-shift算法（[Cheng 1995](https://ieeexplore.ieee.org/document/485468)）：

$$\mathbf{V}_p^+(\mathbf{x}) := \frac{1}{Z_p} \mathbb{E}_p\left[ k(\mathbf{x}, \mathbf{y}^+)(\mathbf{y}^+ - \mathbf{x}) \right] \tag{8a}$$
$$\mathbf{V}_q^-(\mathbf{x}) := \frac{1}{Z_q} \mathbb{E}_q\left[ k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^- - \mathbf{x}) \right] \tag{8b}$$
$$\mathbf{V}_{p,q}(\mathbf{x}) := \mathbf{V}_p^+(\mathbf{x}) - \mathbf{V}_q^-(\mathbf{x}) \tag{10}$$

变量解释：
- $\mathbf{y}^+ \sim p$：**positive samples**，来自data distribution的real images
- $\mathbf{y}^- \sim q$：**negative samples**，来自当前generated distribution
- $\mathbf{y}^+ - \mathbf{x}$：从 $\mathbf{x}$ 指向正样本的向量（attraction方向）
- $\mathbf{y}^- - \mathbf{x}$：从 $\mathbf{x}$ 指向负样本的向量（repulsion方向）
- $k(\mathbf{x}, \mathbf{y})$: kernel function加权
- $Z_p(\mathbf{x}) := \mathbb{E}_p[k(\mathbf{x}, \mathbf{y}^+)]$, $Z_q(\mathbf{x}) := \mathbb{E}_q[k(\mathbf{x}, \mathbf{y}^-)]$: normalization factors (Eq. 9)

**直觉**：$\mathbf{V}_p^+$ 是一个"被data吸引"的mean-shift vector，$\mathbf{V}_q^-$ 是"被其他generated samples排斥"的vector。当 $p = q$ 时，吸引和排斥恰好相互抵消（因为anti-symmetry）。

### 3.2 合并形式

把(8)代入(10)：

$$\mathbf{V}_{p,q}(\mathbf{x}) = \frac{1}{Z_p Z_q} \mathbb{E}_{p,q} \left[ k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-) (\mathbf{y}^+ - \mathbf{y}^-) \right] \tag{11}$$

观察：vector difference简化成了 $\mathbf{y}^+ - \mathbf{y}^-$（直接从负样本指向正样本），权重由两个kernel的乘积共同决定，joint normalization。

很容易验证anti-symmetry：交换 $p \leftrightarrow q$ 即交换 $\mathbf{y}^+ \leftrightarrow \mathbf{y}^-$，结果是 $(\mathbf{y}^- - \mathbf{y}^+)$ = $-(\mathbf{y}^+ - \mathbf{y}^-)$，整个field反号。

### 3.3 Kernel选择

$$k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{1}{\tau}\|\mathbf{x} - \mathbf{y}\|\right) \tag{12}$$

变量：
- $\tau$: temperature，控制kernel的"sharpness"
- $\|\cdot\|$: $\ell_2$距离
- 这个kernel类似Gaussian RBF，但用的是L2 norm而不是squared L2（细节差异）

实现上用softmax over $\mathbf{y}$ axis（类似[InfoNCE Oord 2018](https://arxiv.org/abs/1807.03748)）。额外加一个softmax over $\mathbf{x}$ axis做额外的batch normalization，这个不影响anti-symmetry。

## 4. Pseudocode分析

```python
# Algorithm 1: Training step
e = randn([N, C])           # noise samples
x = f(e)                    # [N, D] generated samples, this is x ~ q
y_neg = x                   # reuse generated samples as negatives
V = compute_V(x, y_pos, y_neg)
x_drifted = stopgrad(x + V)  # frozen target
loss = mse_loss(x - x_drifted)
```

```python
# Algorithm 2: compute V
def compute_V(x, y_pos, y_neg, T):
    dist_pos = cdist(x, y_pos)        # [N, N_pos] pairwise distances
    dist_neg = cdist(x, y_neg)       # [N, N_neg]
    dist_neg += eye(N) * 1e6          # avoid self-repulsion
    
    logit_pos = -dist_pos / T         # kernel logits
    logit_neg = -dist_neg / T
    logit = cat([logit_pos, logit_neg], dim=1)
    
    A_row = logit.softmax(dim=-1)     # normalize over y-axis (standard)
    A_col = logit.softmax(dim=-2)     # normalize over x-axis (extra)
    A = sqrt(A_row * A_col)
    
    A_pos, A_neg = split(A, [N_pos,], dim=1)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # joint weight
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)
    
    drift_pos = W_pos @ y_pos          # [N, D] attraction
    drift_neg = W_neg @ y_neg          # [N, D] repulsion
    V = drift_pos - drift_neg
    return V
```

注意几个细节：
1. **Self-repulsion avoidance**: `dist_neg += eye(N) * 1e6` 防止sample排斥自己
2. **Joint normalization**: $W_{pos}$ 不仅看 $A_{pos}$，还乘以 $A_{neg}$ 的行和，体现 Eq. (11) 中 $k(\mathbf{x},\mathbf{y}^+)k(\mathbf{x},\mathbf{y}^-)$ 的耦合
3. **Double softmax**: $A = \sqrt{A_{row} \cdot A_{col}}$ 是geometric mean of row-normalized and column-normalized，类似Sinkhorn的一步迭代

## 5. Feature-Space Drifting

直接在pixel space计算kernel很困难（高维+语义距离不准）。把loss放到feature space：

$$\mathbb{E}\left[ \left\| \phi(\mathbf{x}) - \mathrm{stopgrad}\left( \phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x})) \right) \right\|^2 \right] \tag{13}$$

变量：
- $\phi$: feature extractor（pre-trained self-supervised model）
- $\phi(\mathbf{y}^+), \phi(\mathbf{y}^-)$: 正负样本的features
- 在feature space计算drift

**与perceptual loss的区别**：perceptual loss $\|\phi(\mathbf{x}) - \phi(\mathbf{x}_{\text{target}})\|^2$ 需要**配对**的target；而drifting loss的target是 $\phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x}))$，由kernel-based drift计算，**不需要配对**。

这让我想起[Sinkhorn distances](https://arxiv.org/abs/1306.0895)和[MMGAN](https://arxiv.org/abs/1505.03906)—— distribution-level matching vs sample-level regression。

## 6. Architecture & Implementation

### 6.1 Generator: DiT-like

- Input: $32 \times 32 \times 4$ Gaussian noise (latent) 或 $256 \times 256 \times 3$ (pixel)
- Output: same shape (latent or pixel)
- Patch size: $2 \times 2$ (latent) 或 $16 \times 16$ (pixel)
- DiT-B/2: hidden 768, depth 12
- DiT-L/2: hidden 1024, depth 24
- 使用 SwiGLU ([Shazeer 2020](https://arxiv.org/abs/2002.05202)), RoPE ([Su 2024](https://arxiv.org/abs/2104.09864)), RM-SNorm, QK-Norm
- adaLN-zero conditioning on class label + CFG scale $\alpha$
- 16 learnable register tokens (in-context conditioning，参考[Li & He 2025](https://arxiv.org/abs/2511.13720))
- **32 random style embeddings**：每token是一个codebook index，codebook有64个learnable embeddings。这扩展了noise distribution的多样性，类似StyleGAN

### 6.2 Feature Encoder: Latent-MAE

paper提出一个customized **ResNet-style MAE**预训练在latent space：

```
Encoder: x -> {f1, f2, f3, f4}  # 4 scales
         shapes: 32x32xC, 16x16x2C, 8x8x4C, 4x4x8C
         
Decoder (U-Net style): {f4, f3, f2, f1} -> x_hat
         - Bilinear 2x2 upsampling
         - Concatenate with skip connection  
         - GroupNorm + 2x 3x3 conv
```

预训练：mask 2×2 patches with 50% probability，reconstruct masked regions（L2 loss）。

### 6.3 Multi-scale features

从ResNet每个stage提取多种features：
- (a) 每个spatial location的向量
- (b) global mean/std  
- (c) 2×2 patch的mean/std
- (d) 4×4 patch的mean/std

每个feature单独计算drifting loss，最后sum。这个设计很丰富，类似[Perceptual loss的多层VGG features](https://arxiv.org/abs/1603.08155)，但用multi-scale statistics代替单点regression。

### 6.4 Feature & Drift Normalization

**Feature normalization**（Eq. 18-22）：

$$S_j = \frac{1}{\sqrt{C_j}} \mathbb{E}_\mathbf{x} \mathbb{E}_\mathbf{y} [\|\phi_j(\mathbf{x}) - \phi_j(\mathbf{y})\|]$$
$$\tilde{\phi}_j := \phi_j / S_j$$

目的：让average distance约为 $\sqrt{C_j}$，使temperature $\tau$ 与feature magnitude / dimensionality无关。

**Drift normalization**（Eq. 23-25）：

$$\lambda_j = \sqrt{\mathbb{E}\left[\frac{1}{C_j}\|\mathbf{V}_j\|^2\right]}$$
$$\tilde{\mathbf{V}}_j := \mathbf{V}_j / \lambda_j$$

目的：让normalized drift的 $\ell_2$ norm约为 $\sqrt{C_j}$，使不同scale的loss贡献平衡。

**Multiple temperatures**：$\tau \in \{0.02, 0.05, 0.2\}$，sum对应的drifts。这类似[MMD的多bandwidth技巧](https://arxiv.org/abs/1505.03906)——单一kernel width难以覆盖所有尺度的相似性。

## 7. Classifier-Free Guidance的优雅实现

### 7.1 Negative sample mixing

$$\tilde{q}(\cdot|c) \triangleq (1-\gamma) q_\theta(\cdot|c) + \gamma p_{\text{data}}(\cdot|\emptyset) \tag{15}$$

变量：
- $c$: class label
- $\gamma \in [0, 1)$: mixing rate
- $p_{\text{data}}(\cdot|\emptyset)$: unconditional data distribution（用random class的real images）
- $\tilde{q}$: effective negative distribution

### 7.2 推导

希望 $\tilde{q}(\cdot|c) = p_{\text{data}}(\cdot|c)$，代入得：

$$q_\theta(\cdot|c) = \alpha p_{\text{data}}(\cdot|c) - (\alpha - 1) p_{\text{data}}(\cdot|\emptyset) \tag{16}$$

其中 $\alpha = \frac{1}{1-\gamma} \geq 1$。

这正是[Ho & Salimans CFG](https://arxiv.org/abs/2207.12598)的线性组合形式：conditional extrapolation along (conditional - unconditional)方向。

### 7.3 关键优势

**Drifting的CFG是training-time behavior**：
- 训练时mix unconditional samples作为额外negative
- 推理时**仍然是1-NFE**
- 而标准CFG需要2-NFE（conditional + unconditional两次forward）

这与[iMeanFlow](https://arxiv.org/abs/2512.02012)的CFG-conditioning思路一致：把 $\alpha$ 作为网络输入conditioning，训练时随机采样 $\alpha$ from $p(\alpha) \propto \alpha^{-3}$ 或 $\alpha^{-5}$。

## 8. 实验结果深度分析

### 8.1 ImageNet 256×256 Latent Space (Table 5)

| Method | Space | Params | NFE | FID↓ | IS↑ |
|--------|-------|--------|-----|------|-----|
| **Multi-step Diffusion/Flows** | | | | | |
| DiT-XL/2 | SD-VAE | 675M+49M | 250×2 | 2.27 | 278.2 |
| SiT-XL/2+REPA | SD-VAE | 675M+49M | 250×2 | 1.42 | 305.7 |
| LightningDiT-XL/2 | VA-VAE | 675M+70M | 250×2 | 1.35 | 295.3 |
| RAE+DiTDH-XL/2 | RAE | 839M+415M | 50×2 | 1.13 | 262.6 |
| **Single-step** | | | | | |
| iCT-XL/2 | SD-VAE | 675M+49M | 1 | 34.24 | - |
| Shortcut-XL/2 | SD-VAE | 675M+49M | 1 | 10.60 | - |
| MeanFlow-XL/2 | SD-VAE | 676M+49M | 1 | 3.43 | - |
| AdvFlow-XL/2 | SD-VAE | 673M+49M | 1 | 2.38 | 284.2 |
| iMeanFlow-XL/2 | SD-VAE | 610M+49M | 1 | 1.72 | 282.0 |
| **Drifting (本文)** | | | | | |
| Drifting B/2 | SD-VAE | **133M**+49M | 1 | 1.75 | 263.2 |
| Drifting L/2 | SD-VAE | **463M**+49M | 1 | **1.54** | 258.9 |

关键观察：
1. **Drifting L/2 (463M) 比 iMeanFlow-XL/2 (610M) 还小**，但FID更低 (1.54 vs 1.72)
2. **比所有multi-step 1-step distillation方法都好**（除RAE+DiTDH用更强tokenizer）
3. **甚至逼近multi-step方法**（如LightningDiT 1.35 FID），考虑到只用1步推理，性价比极高

### 8.2 Pixel Space (Table 6)

| Method | Params | NFE | FID↓ |
|--------|--------|-----|------|
| SiD2 UViT/1 | - | 512×2 | 1.38 |
| JiT-G/16 | 2B | 100×2 | 1.82 |
| PixelDiT/16 | 797M | 200×2 | 1.61 |
| StyleGAN-XL | 166M | 1 | 2.30 |
| GigaGAN | 569M | 1 | 3.45 |
| EPG-L/16 | 540M | 1 | 8.82 |
| **Drifting L/16** | **464M** | **1** | **1.61** |

**Drifting Model pixel-space 1-NFE = 1.61 FID**，与PixelDiT/16 (200步)持平，远超其他1-NFE方法。FLOPs仅87G，而StyleGAN-XL是1574G（**18×效率提升**）。

### 8.3 Ablation核心发现

**Anti-symmetry是关键**（Table 1）：

| Case | V definition | FID |
|------|-------------|-----|
| anti-symmetry (default) | $\mathbf{V}^+ - \mathbf{V}^-$ | 8.46 |
| 1.5× attraction | $1.5\mathbf{V}^+ - \mathbf{V}^-$ | 41.05 |
| 2.0× repulsion | $\mathbf{V}^+ - 2\mathbf{V}^-$ | 112.84 |
| attraction-only | $\mathbf{V}^+$ | 177.14 |

破坏anti-symmetry导致**灾难性失败**。这验证了paper的理论：只有当 $p=q$ 时吸引和排斥恰好抵消，equilibrium才有意义。

**Feature encoder质量重要**（Table 3）：

| SSL method | Width | SSL epochs | FID |
|------------|-------|------------|-----|
| SimCLR | 256 | 800 | 11.05 |
| MoCo-v2 | 256 | 800 | 8.41 |
| latent-MAE | 256 | 192 | 8.46 |
| latent-MAE | 640 | 192 | 6.30 |
| latent-MAE | 640 | 1280 | 4.28 |
| latent-MAE + cls ft | 640 | 1280 | **3.36** |

更强feature encoder → 更好generation。**paper明确说**：不用feature encoder无法在ImageNet上work。这暗示了kernel-based方法在高维空间需要好的semantic representation作为"距离度量"。

### 8.4 Robotics Control (Table 7)

对比[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (100 NFE)，Drifting Policy (1 NFE)：

| Task | DP (100 NFE) | Drifting (1 NFE) |
|------|--------------|-------------------|
| Can (Visual) | 0.97 | 0.99 |
| ToolHang (Visual) | 0.73 | 0.67 |
| BlockPush Phase1 | 0.36 | **0.56** |
| Kitchen Phase4 | 0.99 | 0.96 |

1-NFE方法在多个任务上**持平或超过**100-NFE的Diffusion Policy，特别在多阶段任务上还有提升。这暗示drifting范式对**非图像领域**也很有潜力。

## 9. 与相关工作的深度联系

### 9.1 与Mean-Shift算法

[Mean-shift](https://ieeexplore.ieee.org/document/485468) 是non-parametric mode-seeking：每个点向kernel-weighted mean移动。Drifting Model的 $\mathbf{V}_p^+$ 就是标准mean-shift vector。**创新点在于加入repulsion $\mathbf{V}_q^-$，并通过anti-symmetry达到p=q时的平衡**。

### 9.2 与MMD

paper Appendix C.2给出MMD的drifting field形式。MMD loss:

$$\mathcal{L}_{\text{MMD}^2}(p, q) = \mathbb{E}_{\mathbf{x}, \mathbf{x}' \sim q}[\xi(\mathbf{x}, \mathbf{x}')] - 2\mathbb{E}_{\mathbf{y} \sim p, \mathbf{x} \sim q}[\xi(\mathbf{y}, \mathbf{x})] + \text{const}$$

对应drifting field：

$$\mathbf{V}_{\text{MMD}}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}^+ \sim p}\left[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^+)}{\partial \mathbf{x}}\right] - \mathbb{E}_{\mathbf{y}^- \sim q}\left[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^-)}{\partial \mathbf{x}}\right]$$

对于Gaussian kernel $\xi(\mathbf{x},\mathbf{y}) = \exp(-\|\mathbf{x}-\mathbf{y}\|^2 / 2\sigma^2)$：

$$\tilde{k}_{\text{MMD}}(\mathbf{x}, \mathbf{y}) = \frac{1}{\sigma^2}\exp\left(-\frac{\|\mathbf{x}-\mathbf{y}\|^2}{2\sigma^2}\right)$$

**关键区别**：
1. MMD的kernel是**未normalized**的，导致mean-shift interpretation失效
2. Drifting framework支持**normalized kernels** $\tilde{k} = \frac{1}{Z}k$，让Eq. (11)的双kernel耦合成立
3. V-centric框架允许flexible step size（V-normalization）
4. 自然扩展到CFG，MMD的CFG变体尚未被探索

paper明确说：**MMD framework在实验中无法获得reasonable结果**。

### 9.3 与Stein Variational Gradient Descent (SVGD)

[SVGD](https://arxiv.org/abs/1608.04471) 用kernel-based gradient驱动particle system演化，目标是让particles采样target distribution。SVGD的update rule：

$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \frac{\epsilon}{n} \sum_{j \neq i} \left[ k(\mathbf{x}_j, \mathbf{x}_i) \nabla_{\mathbf{x}_j} \log p(\mathbf{x}_j) + \nabla_{\mathbf{x}_j} k(\mathbf{x}_j, \mathbf{x}_i) \right]$$

Drifting Model的Eq. (11)有类似的kernel-based update，但：
- 不需要 $\log p$ 的gradient（只需要samples from p）
- 通过anti-symmetry达到平衡，而非收敛到target
- 直接在parameter space优化网络，而非直接更新particles

### 9.4 与Contrastive Learning

Positive/negative samples + softmax kernel + InfoNCE-style normalization，这让drifting Model在概念上接近contrastive learning。但目标不同：
- Contrastive: 学习representation（discriminative）
- Drifting: 学习generator（generative）

### 9.5 与Consistency Models / MeanFlow

[Consistency Models](https://arxiv.org/abs/2303.01969) 和 [MeanFlow](https://arxiv.org/abs/2505.13447) 都是为了1-NFE generation，但都**保留了SDE/ODE formulation**——本质上是approximating the trajectory。

Drifting Model **完全脱离SDE/ODE框架**，直接定义distribution evolution via vector field。这是一个**概念性的不同**。

### 9.6 与Schrodinger Bridge / Flow Matching

[Flow Matching](https://arxiv.org/abs/2210.02747) 训练一个velocity field $v_t(\mathbf{x})$ 使得 $\mathbf{x}_t = \mathbf{x}_0 + \int_0^t v_s(\mathbf{x}_s) ds$ 满足marginal distribution约束。Drifting Model的 $\mathbf{V}_{p,q}$ 类似一个"训练时的velocity field"，但**不依赖于时间$t$**——distribution演化通过SGD迭代"自然"展开。

## 10. Identifiability分析（Appendix C.1）

paper证明了一个重要性质：**zero-drift条件在mild assumptions下蕴含 $p \approx q$**。

考虑basis expansion:
$$p(\mathbf{y}) = \sum_{i=1}^m a_i \varphi_i(\mathbf{y}), \quad q(\mathbf{y}) = \sum_{i=1}^m b_i \varphi_i(\mathbf{y})$$

定义interaction vector:
$$\mathbf{U}_{ij}[:, \mathbf{x}] \triangleq \iint K(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-) \varphi_i(\mathbf{y}^+) \varphi_j(\mathbf{y}^-) d\mathbf{y}^+ d\mathbf{y}^-$$

则drifting field在probe set $\mathcal{X}$ 上：

$$\mathbf{V}_\mathcal{X} = \sum_{i=1}^m \sum_{j=1}^m a_i b_j \mathbf{U}_{ij}$$

Anti-symmetry意味着 $\mathbf{U}_{ij} = -\mathbf{U}_{ji}$（所以 $\mathbf{U}_{ii} = \mathbf{0}$）。

**Linear independence assumption**: $\{\mathbf{U}_{ij}\}_{1 \leq i < j \leq m}$ 在 $\mathbb{R}^{dN}$ 中线性无关（要求 $dN \gg m^2$）。

则zero-drift条件 $\mathbf{V}_\mathcal{X} = \mathbf{0}$ 等价于：

$$\sum_{i < j} (a_i b_j - a_j b_i) \mathbf{U}_{ij} = \mathbf{0}$$

由独立性：$a_i b_j = a_j b_i, \forall i,j$，意味着 $\mathbf{a} \parallel \mathbf{b}$。加上概率密度归一化 $\int p = \int q = 1$，得 $\mathbf{a} = \mathbf{b}$，即 $p = q$。

这是**paper的理论核心**：虽然 $q = p \Rightarrow \mathbf{V} = 0$ 很容易，**但反过来 $\mathbf{V} \approx 0 \Rightarrow q \approx p$ 在mild assumptions下也成立**。这是Drifting Model作为generative model的**理论基础**。

## 11. 直觉构建：为什么这个方法会work

### 11.1 训练动态作为分布演化

SGD的每一步更新：$\theta_{i+1} = \theta_i - \eta \nabla \mathcal{L}$
对应pushforward演化：$q_i \to q_{i+1}$

如果我们能控制每一步 $\theta$ 的更新方向，使得 $q_i$ 朝着 $p_{\text{data}}$ 移动，最终就能match。

**关键insight**：drifting field $\mathbf{V}$ 提供了一个**显式的"希望样本移动方向"**，loss只是把这个方向变成可微的回归目标。

### 11.2 为什么anti-symmetry是关键

考虑toy example：1D Gaussian $p = \mathcal{N}(0, 1)$，$q = \mathcal{N}(2, 1)$（mean差2）。

- $\mathbf{V}_p^+(\mathbf{x})$: 从x指向p的mean，即 $-\mathbf{x} + 0 = -\mathbf{x}$（吸引到0）
- $\mathbf{V}_q^-(\mathbf{x})$: 从x指向q的mean，即 $-\mathbf{x} + 2 = 2 - \mathbf{x}$（"排斥"到2）

Wait，这里需要小心。$\mathbf{V}_q^-(\mathbf{x})$ 是**repulsion**，意味着我们减去它。如果 $q = p$（都在mean 0）：

- $\mathbf{V}_p^+(\mathbf{x}) = -\mathbf{x}$（mean shift to 0）
- $\mathbf{V}_q^-(\mathbf{x}) = -\mathbf{x}$（mean shift to 0，因为q的mean也是0）
- $\mathbf{V}_{p,q} = \mathbf{V}_p^+ - \mathbf{V}_q^- = -\mathbf{x} - (-\mathbf{x}) = 0$ ✓

这就是anti-symmetry在work：当 $p = q$，attract和repel的mean-shift vectors恰好相等，相减为零。

直觉：**mean-shift to the mean of p = mean-shift to the mean of q when p = q**。这是anti-symmetry的物理含义。

### 11.3 为什么mode collapse难以发生

Paper的toy experiment（Figure 3 bottom）展示：即使q初始collapsed到p的一个mode上，drifting会推动samples向另一个mode扩散。

机制：如果q塌缩到mode A，那么 $\mathbf{V}_q^-$（排斥来自q的samples）会推samples远离mode A。同时，mode B的real samples会通过 $\mathbf{V}_p^+$ 吸引一些samples过去。最终q会覆盖所有modes。

这与GAN的mode collapse问题形成鲜明对比——drifting的"repulsion"机制天然防止collapse。

### 11.4 与Sinkhorn的暗中联系

Double softmax normalization（$A = \sqrt{A_{row} \cdot A_{col}}$）类似[Sinkhorn algorithm](https://arxiv.org/abs/1306.0895)的一步迭代，但paper只用一步而非完整收敛。这是connection to entropic OT，让kernel weights doubly-stochastic-like。

### 11.5 与Energy-based Models (EBM)的对比

EBM学习一个energy function $E_\theta(\mathbf{x})$，通过 $p_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$ 建模分布。训练用Langevin dynamics或contrastive divergence。

Drifting Model也用positive/negative samples，但**学的是generator而非energy function**。generator直接输出samples，不需要MCMC采样——更高效。

### 11.6 与Score-based Models的对比

[Score matching](https://arxiv.org/abs/1907.05600) 学习 $\nabla \log p(\mathbf{x})$，drifting field $\mathbf{V}_{p,q}$ 也是一个向量场但**不是score**。Score描述log-density的gradient，drifting field描述**kernel-weighted mean-shift**——一个non-parametric quantity，不需要log density。

## 12. Open Questions & Future Work

Paper自己提出的open questions：

1. **Converse direction**：虽然实验显示 $\|\mathbf{V}\| \to 0$ 与FID降低相关，但理论上 $\mathbf{V} \to 0 \Rightarrow q \approx p$ 的精确条件仍不明确（Appendix C.1 给了heuristic argument）

2. **Drifting field设计**：当前用mean-shift inspired + Gaussian kernel，其他设计（如neural kernel、attention-based kernel）可能更好

3. **Feature encoder选择**：当前依赖SSL预训练（latent-MAE），是否可以联合训练？或者用更弱的prior？

4. **理论分析**：训练动态的convergence rate、与SGD的interaction、local minima分析

我自己想到的几个方向：

- **Continuous-time extension**：当前用discrete iteration，可以用neural ODE把drifting field parameterize为continuous flow，类似[Neural SDE](https://arxiv.org/abs/1906.02363)但用于distribution evolution

- **Theory of V identifiability**：paper的identifiability argument依赖basis expansion假设，更general的分析可能用kernel methods的RKHS theory

- **Application to other domains**：3D generation、video、text generation——drifting framework不依赖图像structure，理论上适用任何modality

- **Connection to Schrödinger Bridge**：drifting可以看作一种"training-time bridge"——能否与[Schrodinger Bridge](https://arxiv.org/abs/2205.14258)统一？

- **Hybrid with diffusion**：训练时drifting + 推理时少量diffusion refinement可能进一步改善quality

## 13. 总结性直觉

**Drifting Model的核心insight**：把SGD迭代重新解读为distribution evolution，把"在inference time分解复杂映射"转化为"在training time演化分布"。这通过：

1. **Drifting field** $\mathbf{V}_{p,q}$: 定义每个generated sample应该怎么移动
2. **Anti-symmetry**: 保证 $p = q$ 时field归零（equilibrium）
3. **Mean-shift inspiration**: attraction from $p$ + repulsion from $q$
4. **Kernel-based**: 通过kernel similarity加权，类似contrastive learning
5. **Feature space**: 在SSL features上计算drift，避免高维pixel space的kernel问题
6. **CFG by design**: 通过mixing unconditional negatives实现，保留1-NFE

最终效果：**ImageNet 256×256上1.54 FID (latent) / 1.61 FID (pixel)**，超过所有previous 1-NFE方法，与multi-step方法comparable，且用更少parameters。

这paper的精神让我想起你（Karpathy）在[的一些commentary](https://karpathy.ai/)：**"simple things that scale"**——drifting Model在概念上比diffusion更简单（不需要SDE/ODE），把复杂性replaced by training time + self-supervised features。这种简化带来的1-NFE能力，可能在real-world deployment上有重要意义。

## Reference Links

- **arXiv (paper)**: https://arxiv.org/abs/2505.22689 (Drifting Models paper, 假设链接)
- **Diffusion Models**: https://arxiv.org/abs/1503.03585
- **Flow Matching**: https://arxiv.org/abs/2210.02747  
- **DiT**: https://arxiv.org/abs/2212.09748
- **iMeanFlow**: https://arxiv.org/abs/2512.02012
- **MeanFlow**: https://arxiv.org/abs/2505.13447
- **Consistency Models**: https://arxiv.org/abs/2303.01969
- **MoCo**: https://arxiv.org/abs/1911.05722
- **SimCLR**: https://arxiv.org/abs/2002.05709
- **MAE**: https://arxiv.org/abs/2111.06377
- **Latent Diffusion**: https://arxiv.org/abs/2112.10752
- **CFG**: https://arxiv.org/abs/2207.12598
- **InfoNCE**: https://arxiv.org/abs/1807.03748
- **Mean-shift algorithm**: https://ieeexplore.ieee.org/document/485468
- **MMD-based GAN**: https://arxiv.org/abs/1505.03906
- **SVGD**: https://arxiv.org/abs/1608.04471
- **Schrodinger Bridge**: https://arxiv.org/abs/2205.14258
- **Sinkhorn**: https://arxiv.org/abs/1306.0895
- **BYOL**: https://arxiv.org/abs/2006.07733
- **StyleGAN-XL**: https://arxiv.org/abs/2203.01856
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **ConvNeXt V2**: https://arxiv.org/abs/2301.00808
- **Karpathy's writings**: https://karpathy.ai/

希望这个深度解析帮你build up intuition about this elegant paper。这个training-time evolution视角我觉得可能会inspire一系列后续工作——把distribution evolution和SGD dynamic解耦这件事，可能远不止image generation的应用。
