---
source_pdf: STAR- Rethinking MoE Routing as Structure-Aware Subspace Learning.pdf
paper_sha256: e69d641c90764b547c1cdfca692c9f8948901efc003a3db799bf66b3a14cfb0b
processed_at: '2026-08-12T10:56:33-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# STAR的"人话"版本

好，我来用大白话重新讲一遍，尽量剥掉math外衣，只保留core idea。

---

## 一句话说清楚

**MoE的router本来就该"看懂input长什么样"再决定路由，但现在的router是个瞎子——STAR给它装了副眼镜。**

---

## 1. 现在的MoE router有多糙

你train一个Mixtral、DeepSeek-MoE，router干的事就一行：

```
score = input · weight_matrix
pick top-k experts
```

这个weight matrix是gradient倒推学的。问题是gradient只告诉你"什么对task loss有帮助"，完全没告诉你"input data本身长什么shape"。就像一个厨师，只知道哪道菜客人给了好评，但从来不去看食材长什么样——他怎么做specialization？

更具体一点：假设你的input representation其实cluster成几个blob（比如code token一个blob，自然语言一个blob，math一个blob）。Standard router完全不知道这些blob的存在，它就是在整个空间里随机初始化几个方向，靠gradient慢慢往"有用"的地方挪。

$K$小的时候（8个expert），瞎猫碰死耗子，总还能学到点东西。$K$一大（40、160），router的capacity稀释，开始collapse——几个expert抢所有token，其他expert饿死。这就是Figure 4(d)里Standard MoE在$K=40$时$H_{\text{norm}}$从0.43掉到0.17的原因。

之前大家怎么fix这个？加regularizer逼expert均匀干活（[load balancing loss](https://arxiv.org/abs/1701.06538)）、让expert反选token（[Expert Choice](https://arxiv.org/abs/2202.09368)）、用cosine similarity代替dot product（[Cosine Router](https://arxiv.org/abs/2206.04046)）。这些都是**症状治疗**——expert不均衡？逼它均衡。但router本身瞎不瞎？没人管。

STAR说：咱换个axis，治本——让router先看见input structure。

---

## 2. STAR怎么让router"看见"

### 第一步：online估计input的principal directions

数据在hidden space里不是各向同性散开的，总有几个方向variance大、其他方向variance小。这就是PCA的intuition——找variance最大的方向。

但standard PCA要先把所有数据收集起来算covariance matrix，对streaming training不现实。STAR用GHA（[Generalized Hebbian Algorithm, Sanger 1989](https://www.sciencedirect.com/science/article/pii/0893608089900517)），这是个**online incremental PCA**：每来一个mini-batch，basis vector就往variance大的方向挪一丁点。

最关键的是——GHA是个**local、unsupervised rule**，跟gradient完全无关。它纯粹在追求数据本身的统计structure。

你可以这么想：standard router是个有supervisor教的学生，老师（gradient）说什么它学什么。GHA是个自己翻教科书的学生，没老师，但能看出数据里"哪些方向information多"。

### 第二步：把basis decouple到expert上

最naive的做法：让第$k$个expert对应第$k$个principal component。但这有个致命问题——PCA spectrum衰减得很快（$\lambda_1 \gg \lambda_K$），第一个PC variance可能是第一百个的100倍。如果直接这么做，第一个expert会吃掉所有token，后面的expert全饿死。Expert hierarchy就被variance hierarchy绑架了。

STAR加了mixing matrix $R \in \mathbb{R}^{K \times K}$，让每个expert的routing direction是所有principal components的**线性组合**，不是一一对应。这样expert selection就跟variance大小脱钩了，但仍然在principal subspace里——structure-aware的property保留。

Paper里的Proposition 4.2和4.3证明得很干净：
- $R=I$：energy直接是$\lambda_k$，collapsed
- $R$ random orthonormal：energy期望相等，balanced
- $R$ learnable：实验上也能avoid collapse，而且比random更expressive

这是个很elegant的theoretical insight。

### 第三步：跟standard gate interpolate

完全unsupervised的GHA可能跟task无关（input variance大的方向不一定是task-relevant方向）。所以STAR把GHA gate和standard learnable gate做weighted sum：

```
final_logits = σ(α) * linear_logits + (1 - σ(α)) * GHA_logits
```

$\alpha$是learnable的per-expert coefficient。有意思的是实验观察到$\sigma(\alpha)$训练过程中**单调下降**——model越来越依赖GHA。这说明：当representation stabilize之后，input structure本身比gradient signal更reliable。

---

## 3. 为什么这个思路work？我的几个take

### (a) Unsupervised prior > Learned prior in certain regimes

Deep learning里反复出现这个pattern：batch norm用batch statistics稳定training，contrastive learning用augmentation structure学representation，[SimCLR](https://arxiv.org/abs/2002.05709)用instance discrimination。这些都是把"数据本身的某些structure"注入到model里，而不是全靠gradient。

STAR干的事类似：把input distribution的principal structure作为prior，注入到routing决策里。Gradient负责specialization细节，structure负责routing robustness。两者互补。

### (b) Routing imbalance是症状，不是病因

之前所有工作（[load balancing](https://arxiv.org/abs/1701.06538)、[GShard loss](https://arxiv.org/abs/2006.16668)、[auxiliary-loss-free](https://arxiv.org/abs/2408.15664)）都在治imbalance这个症状。STAR说：imbalance的根因是router没看见structure，所以它选expert是arbitrary的，几个expert碰巧被gradient推到similar位置就collapse了。Structure-aware之后，routing decisions有prior支撑，自然更stable。

Ablation里最informative的是**random basis**那一行（Table 2底部）：把GHA换成固定random orthonormal basis，accuracy掉到79.59，MNLI方差到15.85。这证明**不是"加额外信息"就有用，必须是data-driven的structure**。

### (c) Test-time adaptation是个free lunch

GHA是unsupervised的，所以test时也能继续update。ImageNet-C上severity 5 corruption，STAR+TTA比standard MoE高1.56%——router在线adapt到corrupted input的新分布。这相当于unsupervised domain adaptation的cost-free版本。

这个让我想到[TENT](https://arxiv.org/abs/2006.10926)（test-time entropy minimization），但STAR更轻——连entropy都不用算，直接Hebbian rule就行。

### (d) 跟Attention的analogy

MoE routing本质是 $x W^\top$，attention也是 $Q K^\top$，都是某种"用query去select keys/experts"。Linear attention family（[Performer](https://arxiv.org/abs/2009.14794)、[RWKV](https://arxiv.org/abs/2305.13048)、[RetNet](https://arxiv.org/abs/2307.08621)）其实在用kernel trick把attention限制在某个subspace。STAR的思路完全可以推广到attention：key/value的basis也可以structure-aware。

### (e) 跟old-school MoE literature的精神呼应

[Jacobs 1991](https://doi.org/10.1162/neco.1991.3.1.79)最初的MoE paper里，expert specialization是core motivation。但deep learning时代的MoE基本都focus在engineering（sparse activation、load balance、expert choice），specialization这个原始motivation反而被遗忘了。STAR某种意义上是"回到future"——重新拾起expert应该specialize along input structure这个old idea，但用了modern tool（GHA + online learning）。

---

## 4. 实验到底说明什么

### Synthetic实验的杀手锏

HMM生成的数据有明确的latent structure（entity × property）。STAR的expert selection和latent property的mutual information在$K=40$时是0.98，Standard MoE只有0.24。**Standard MoE在$K=40$时basically完全collapse了——expert selection和真实structure毫无关系**。

这个数字你应该敏感：0.24意味着routing decision基本是noise，expert之间没有真正的specialization。STAR的0.98意味着几乎perfect alignment。

### Scale-up实验

LLaMA-MoE 469M active / 2.6B total，在Pile上pretrain 30B tokens，zero-shot平均43.93% vs Standard MoE 42.69%。+1.24%在zero-shot setting下是**非trivial的improvement**——尤其是所有method都加了load balancing loss，STAR的gain完全来自structure-awareness这一orthogonal axis。

### Ablation里的关键insight

Table 2最下面三行：
- **No R**：去掉mixing matrix，直接用GHA basis做routing → 81.23 (-1.01)
- **No Interpolation**：纯GHA，不要linear gate → 81.60 (-0.64)  
- **Random basis**：GHA换成固定random basis → 79.59 (-2.65, 方差爆炸)

三件事缺一不可：
1. 必须有R decouple variance hierarchy
2. 必须有linear gate提供task supervision
3. 必须是data-driven的basis（GHA），不能random

Random basis的失败尤其重要——它证明STAR的gain不是"加个regularizer"那种trick，而是真的在capture数据structure。

---

## 5. 我觉得哪里可能有问题

### GHA的convergence vs representation drift

LLM pretraining中，hidden representation本身在变化——你训1000步时的$x$和训10000步时的$x$分布完全不同。GHA追的是moving target。Paper的Figure 7显示GHA在fixed representation上能well-approximate SVD，但没显示在moving representation下的tracking质量。

如果representation drift速度 > GHA convergence速度，basis会滞后，structure-aware的gain会打折扣。

### Large K下的R

DeepSeek-V2有160个expert，$R$是$160 \times 160 = 25600$个参数，computational cost $\mathcal{O}(K^2 d)$在$K=160, d=4096$下是1M ops per token，relative to expert compute $\mathcal{O}(k \cdot d \cdot d')$（比如$k=8, d'=11008$）是355M ops——ratio 0.3%。可以接受。但$R$的gradient signal会随$K$增大而稀释，learnable $R$在$K=160$下是否还能well-conditioned？Paper没probe这个。

### Heterogeneous experts

现在所有expert同构。如果expert本身有different architecture/specialization prior（比如[MoE-LLaVA](https://arxiv.org/abs/2401.15947)的vision/language experts），单一principal subspace的assumption可能break。Multi-subspace extension可能需要：每modality一个GHA basis，routing时选subspace + 选expert。

### Spectral bias of GHA

GHA估计top-K eigenvectors，但input里重要的structure未必在top-K principal directions上。比如长尾task-relevant signal在low-variance direction里，GHA会miss掉。是不是该考虑supervised GHA——让basis既track variance又track task-relevance？这就回到了[Supervised PCA](https://www.sciencedirect.com/science/article/pii/S0031320307003838)的范畴。

---

## 6. 一图胜千言

```
            input x
              │
              ├────────────────┐
              │                │
         [GHA update]    [Linear Gate]
         (unsupervised,   (supervised,
          online)          gradient)
              │                │
              ↓                ↓
       V (basis)         W_g (matrix)
              │                │
         [R mixing]            │
              │                │
              ↓                ↓
        l_GHA = xZ^T     l_linear = xW_g^T
              │                │
              └─────interpolate─┘
                   (σ(α) blend)
                        │
                        ↓
                    Softmax
                        │
                        ↓
                   Top-k experts
```

**两个信息流**：上面是unsupervised structure prior，下面是supervised task signal。Interpolate让model自己balance。

---

## 7. 对未来工作的implications

### (a) Routing as subspace learning是个paradigm shift

以后design router，不应该只问"怎么balance load"，应该先问"router看见input structure了吗"。这个axis可以跟现有所有balancing方法正交叠加。

### (b) Hebbian learning的revival

最近[Forward-Forward](https://www.cs.toronto.edu/~hinton/FFA13.pdf)、[equilibrium propagation](https://arxiv.org/abs/1604.05339)这些bio-plausible learning在revive。GHA也属于这个family。Deep learning community可能准备好重新拥抱local learning rules了——尤其是用来做auxiliary signal，不是replace backprop。

### (c) Test-time adaptation的cost-free版本

[TTA](https://arxiv.org/abs/2006.10926) field一直在找"如何不retrain就adapt到新distribution"。STAR的GHA update in test time是个非常lightweight的方案——no gradient, no entropy, just Hebbian。这个思路可以应用到其他architecture。

### (d) Mixture of Subspaces的revival

[Original MoE](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)的精神是mixture of subspaces，每个expert学一个local region。STAR重新显式化这个subspace estimation。可能引出一系列工作：spectral MoE, kernel MoE, manifold-aware MoE...

---

## 8. 如果你只记一句话

**Router的设计不该只追求"balanced"，更该追求"structure-aware"。STAR用online PCA（GHA）给router装上了structure感知的眼镜，再加个mixing matrix $R$ 打破variance hierarchy绑架expert utilization的魔咒，最后跟standard gate interpolate让supervised和unsupervised signal互补。**

这就是全部。剩下的experiment和theory都是supporting evidence。

Code在这里：https://github.com/psmiz/STAR

---

还有什么角度想dive deeper的，比如GHA的具体推导、R的learnable dynamics、或者跟某个具体work的comparison，随时说。

---

# STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning - 深度解读

Karpathy你好，这篇paper提出了一个相当elegant的视角来重新思考MoE routing。我尽量把intuition、技术细节、实验数据都unpack出来，并且尽量多联想相关work。

---

## 1. 核心问题：MoE Router到底在干什么？

当前主流MoE (Mixtral, DeepSeek-MoE, Qwen2.5-MoE, Switch Transformer)的router就是一个**shallow linear projection**：

$$\text{logits} = xW_g^\top, \quad W_g \in \mathbb{R}^{K \times d}$$

其中 $x \in \mathbb{R}^d$ 是token的hidden representation，$W_g$ 是gating matrix，$K$ 是expert数量。然后softmax + top-k选expert。

**问题在哪？** 这个router对input structure完全没有inductive bias。$W_g$的每一行 $g_k \in \mathbb{R}^d$ 是一个**learned**的方向，但这些方向是否align with input的实际分布结构？完全没有保证。特别是当 $K$ 增大时（比如DeepSeek-V2有160个expert），router的capacity不足，routing quality会退化，导致expert collapse、specialization变差。

之前大部分工作（load balancing loss, GShard loss, expert choice routing, ReMoE）都在解决**imbalance**问题。STAR的视角是orthogonal的：router到底有没有**aware of input structure**？这才能保证stable input-expert specialization。

Reference: [Switch Transformer (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961), [Mixtral (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088), [DeepSeek-MoE](https://arxiv.org/abs/2401.06066), [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

---

## 2. STAR的核心Idea：Routing = Subspace Learning

### 2.1 Principal Input Structure的形式化定义

STAR把routing重新formalize为估计input的principal subspace。Definition 3.1：

$$S_K^* := \arg\min_{P \in \mathbb{R}^{d \times K}, \, P^\top P = I} \, \mathbb{E}\bigl[\lVert x - PP^\top x \rVert_2^2\bigr]$$

变量含义：
- $P \in \mathbb{R}^{d \times K}$：basis matrix，每一列是一个basis vector
- $P^\top P = I$：约束条件，要求$P$的列是orthonormal的（$I$是$K \times K$单位矩阵）
- $PP^\top x$：把$x$投影到$P$张成的$K$维子空间
- $\lVert \cdot \rVert_2^2$：squared L2 norm
- 最小化重构误差 → $P$的列就是$\Sigma_x = \mathbb{E}[xx^\top]$的top-$K$ eigenvectors

**Intuition**：principal subspace就是input variance最大的那些方向。如果router基于这些方向做决策，那它就是在"看到数据真正的structure"。

### 2.2 为什么用GHA而不是直接PCA/SVD？

Standard PCA需要显式计算covariance matrix $\Sigma_x$，这对streaming/mini-batch场景不practical。GHA (Generalized Hebbian Algorithm, Sanger 1989)是Oja's rule的generalization，**online incremental**估计top-K principal components。

GHA update rule：

$$v_i \leftarrow v_i + \eta\Bigl(y_i x - y_i \sum_{j=1}^{i} y_j v_j\Bigr), \quad y_i = v_i^\top x$$

逐项解读：
- $v_i \in \mathbb{R}^d$：第$i$个basis vector，会逼近$\Sigma_x$的第$i$个eigenvector
- $x \in \mathbb{R}^d$：当前input sample
- $y_i = v_i^\top x$：$x$在$v_i$上的projection（scalar）
- $\eta$：learning rate（论文里设$2 \times 10^{-5}$到$5 \times 10^{-5}$）
- $\sum_{j=1}^{i} y_j v_j$：这个sum是关键！它reconstruct前面所有components的contribution

这个update rule做了两件事：
1. **Hebbian term** $y_i x$：让$v_i$往input方向align（"cells that fire together wire together"）
2. **Decorrelation term** $-y_i \sum_{j \leq i} y_j v_j$：减去前面components已经capture的部分，保证orthogonality

这本质上是在做online Gram-Schmidt正交化，同时align with variance方向。Reference: [Sanger 1989 - Optimal Unsupervised Learning](https://www.sciencedirect.com/science/article/pii/0893608089900517), [Oja & Karhunen 1985](https://www.sciencedirect.com/science/article/pii/0022247X8590159X)

---

## 3. STAR架构详解

### 3.1 完整forward pass（Algorithm 1）

给定input $X \in \mathbb{R}^{N \times d}$（一个batch），$N$是batch size，$d$是hidden dim：

**Step 1: GHA更新（m次迭代）**
对每个input $x$，做$m$次GHA update（$m$是hyperparameter，论文默认1或3）：
```
for m iterations:
    for k in {1,...,K}:
        y_k = v_k^T x
        v_k ← v_k + η * y_k * (x - Σ_{i=1}^k y_i v_i)
        v_k ← v_k / ||v_k||_2   # normalize
```

每次update后normalize保证$v_k$是unit vector。

**Step 2: 计算两组logits**
$$l_{\text{linear}} = xW_g^\top, \quad l_{\text{GHA}} = xZ^\top, \quad Z = RV$$

- $l_{\text{linear}} \in \mathbb{R}^K$：standard learnable gate的logits，$W_g \in \mathbb{R}^{K \times d}$
- $l_{\text{GHA}} \in \mathbb{R}^K$：structure-aware的logits
- $V \in \mathbb{R}^{K \times d}$：GHA学到的basis，每行是一个principal direction
- $R \in \mathbb{R}^{K \times K}$：**mixing matrix**，关键component
- $Z = RV \in \mathbb{R}^{K \times d}$：mixing后的routing matrix，每行是一个expert-specific routing vector

**Step 3: Interpolation + Softmax**
$$s = \text{Softmax}\Bigl(\sigma(\alpha) \odot l_{\text{linear}} + (1 - \sigma(\alpha)) \odot l_{\text{GHA}}\Bigr)$$

- $\alpha \in \mathbb{R}^K$：per-expert的interpolation coefficient（learnable）
- $\sigma(\cdot)$：sigmoid function，把$\alpha$压到$(0,1)$
- $\odot$：element-wise (Hadamard) product
- $s \in \mathbb{R}^K$：final routing scores

### 3.2 为什么需要Mixing Matrix R？

这是paper最elegant的部分。考虑Lemma 4.1：

定义per-expert routing energy $\mathcal{L}_k := \mathbb{E}_x[\ell_k(x)^2]$，即第$k$个expert的logit的second moment。推导后得到：

$$\mathcal{L}_k = r_k^\top \Lambda r_k = \sum_{i=1}^{K} \lambda_i r_{k,i}^2$$

变量：
- $r_k^\top$：$R$的第$k$行（所以$r_k$是列向量）
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_K)$：对角矩阵，$\lambda_i$是第$i$大的eigenvalue
- $r_{k,i}$：$R$第$k$行第$i$列的元素

**关键insight**：routing energy是eigenvalues的加权和，权重是$r_{k,i}^2$。

**Case 1: No mixing ($R = I$)** → Proposition 4.2

此时$r_k = e_k$（standard basis vector），所以：
$$\mathcal{L}_k = \lambda_k$$

routing energy直接继承variance hierarchy！由于$\lambda_1 \gg \lambda_K$（PCA spectrum衰减），leading PC对应的expert会dominate routing，导致expert collapse。

**Case 2: Random orthonormal R** → Proposition 4.3

如果$R$是random orthonormal matrix，$\mathbb{E}[r_k r_k^\top] = \frac{1}{K}I$，则：
$$\mathbb{E}[\mathcal{L}_k] = \frac{1}{K}\sum_{i=1}^{K} \lambda_i, \quad \forall k$$

所有expert的routing energy在期望上**相等**！spectral bias被removed。

**Case 3: Learnable R (STAR default)**

Learnable R是intermediate case，paper在Figure 5用实验显示learnable R也能避免collapse，同时比fixed random R更expressive。

**Intuition**：$R$的作用是decouple expert selection from variance ordering。Expert不应该按"input variance大小"排序来分配，而应该按"semantic specialization"来分配。$R$允许model学习这种decoupling。

---

## 4. 实验数据深度解读

### 4.1 Synthetic HMM Experiment (Section 4)

用multinomial HMM生成有明确structure的数据：每个token来自$(v_t, s_t)$，$v_t$是entity（15个），$s_t$是property（15个），transition有sticky Markov property。这给了data可解释的latent structure。

**Figure 3 结果**：Test loss对比STAR vs Standard MoE
- $K \in \{10, 20, 30, 40\}$，top-$k \in \{1,2,3,4,5\}$
- STAR在所有16个配置下都outperform
- Gap随$K$增大而widening，说明STAR在scale时更robust

**Figure 4 - Specialization分析**：

(a)(b) **Interpolation coefficient $\sigma(\alpha)$的dynamics**：
- 训练过程中$\sigma(\alpha)$单调下降
- 意味着model越来越依赖GHA-driven gating（structure-aware）
- 这个trend在所有$K$和top-$k$配置下都consistent

(c) **Mutual Information $I(e, s)$**（公式2）：
$$I(e, s) = \sum_{e=1}^{K}\sum_{s \in \mathcal{S}} p(e,s) \log\frac{p(e,s)}{p(e)p(s)}$$

- $e$：selected expert
- $s$：latent property
- $p(e,s)$：joint distribution
- Higher $I(e,s)$意味着expert selection和真实latent structure更correlated

| K | Standard MoE | STAR |
|---|---|---|
| 10 | ~0.5 | ~0.6 |
| 30 | ~0.7 | ~0.85 |
| 40 | **0.24 (collapse!)** | **0.98** |

Standard MoE在$K=40$时collapse，STAR继续上升。这是paper最impressive的结果之一。

(d) **Normalized load entropy $H_{\text{norm}}$**（公式3）：
$$H_{\text{norm}} = -\frac{\sum_{e=1}^{K} p(e) \log p(e)}{\log K}$$

$H_{\text{norm}} \in [0,1]$，1表示完全balanced。Standard MoE从0.43 ($K=30$)跌到0.17 ($K=40$)，STAR保持stable。

### 4.2 LLaMA-MoE Pretraining (Table 1)

两个scale：182M active / 777M total，469M active / 2.6B total。在Pile上pretrain 30B tokens。Top-1 routing，8 experts。

| Model | 182M Avg | 469M Avg |
|---|---|---|
| Standard MoE | 40.65 | 42.69 |
| EC (Expert Choice) | 40.13 | 42.81 |
| ReMoE | 39.94 | 43.29 |
| **STAR** | **41.31** | **43.93** |

STAR在两个scale都highest average zero-shot accuracy across 7 tasks (ARC-c, ARC-e, BoolQ, HellaSwag, LAMBADA, PIQA, RACE)。

注意：这个setting下所有方法**都用了load balancing loss**，所以STAR的gain纯来自structure-awareness，证明orthogonal contribution。

Reference: [ReMoE](https://arxiv.org/abs/2412.14711), [Expert Choice Routing](https://arxiv.org/abs/2202.09368), [LLaMA-MoE](https://arxiv.org/abs/2406.16554)

### 4.3 GLUE Fine-tuning (Table 2)

BERT-large + MoEfication，5个GLUE subtask，3个random seeds平均。

| Algorithm | (K,k) | Average |
|---|---|---|
| DynMoE | (9, 7.1) | 81.64 |
| Cosine Router | (8,4) | 81.77 |
| **STAR** | (8,4) | **82.24** |
| Cosine Router | (16,4) | 81.69 |
| STAR | (16,4) | 82.11 |

**Ablation study (底部)**：
- No R: 81.23 (-1.01) → R mixing是essential
- No Interpolation: 81.60 (-0.64) → 需要task supervision
- **Random basis: 79.59 (-2.65), MNLI 75.52±15.85** → GHA incremental learning是核心，random orthogonal basis导致huge instability

Random basis的失败特别informative：它证明不是"加任何额外信息"都有用，必须是**data-driven**的structure。

### 4.4 OOD / Test-Time Adaptation (Table 3, Figure 6)

这是STAR最novel的feature：GHA在test time也能update！

**ImageNet-C, ViT-S/32, 15种corruption**：

| Severity | Standard MoE | STAR | STAR (TTA) |
|---|---|---|---|
| 1 | 58.48 | 58.96 | 59.07 |
| 3 | 43.46 | 44.06 | 44.14 |
| 5 | 22.63 | 23.88 | **24.19** |

TTA在所有severity都带来额外gain。**Intuition**：当test distribution shift时（corruption），GHA能online adapt basis到新的corrupted distribution，这是unsupervised domain adaptation。

GLUE-X上5/7个task提升，average +0.65%。

---

## 5. Computational Analysis (Appendix A)

STAR的额外overhead：

$$\underbrace{\mathcal{O}((m+1)B K d)}_{\text{GHA + extra logits}} + \underbrace{\mathcal{O}(K^2 d)}_{\text{mixing}} + \underbrace{\mathcal{O}(B k d d')}_{\text{experts (baseline)}}$$

- $B$：batch size
- $K$：expert数
- $d$：hidden dim
- $d'$：expert width (FFN intermediate size)
- $m$：GHA iteration数
- $k$：top-k selection

实际配置$K \ll \min(d, d')$（Switch Transformer等都是$K/d$ ratio约1-4%），$m \in \{1,3\}$，所以：

$$\mathcal{O}((m+1)B K d) + \mathcal{O}(K^2 d) \ll \mathcal{O}(B k d d')$$

Table 4实测：STAR ($m=3$)训练latency 131.1ms vs Standard MoE 130.7ms，**几乎无差别**。DynMoE反而慢4倍。

---

## 6. Intuition Building：为什么这个方法work？

### 6.1 Router的两种learning signal

| | Linear Gate | GHA Gate |
|---|---|---|
| Learning signal | Task gradient (supervised) | Input variance (unsupervised) |
| Structure awareness | 无 | 有 |
| Adaptivity | 随gradient走 | Online incremental |

STAR把两者interpolate。训练初期representation不稳定，linear gate主导（gradient-driven）；训练后期representation stabilize，GHA basis也converge了，shift到structure-aware。这解释了Figure 4(a)(b)中$\sigma(\alpha)$递减的现象。

### 6.2 为什么$R$是必要的？

如果直接用$V$做routing，因为PCA spectrum衰减（$\lambda_1 \gg \lambda_K$），energy集中在leading PC。**Variance hierarchy ≠ Expert specialization hierarchy**。$R$的作用是重新分配energy，让每个expert都能被utilize，同时仍然在$V$张成的subspace内（structure-aware）。

这让我联想到一个deep learning的经典问题：**batch normalization强行把activation distribution拉平，但model实际需要的是selective amplification**。STAR的$R$做的是类似的"redistribution"但是structure-preserving。

### 6.3 Connection to Attention Mechanism

MoE routing和attention有structural similarity：
- Attention: $Q K^\top$ select relevant tokens
- MoE routing: $x W_g^\top$ select relevant experts

STAR的视角可以推广：attention也可以structure-aware。比如linear attention (Performer, RWKV, RetNet)其实也在做某种subspace projection。Reference: [Performer](https://arxiv.org/abs/2009.14794), [RWKV](https://arxiv.org/abs/2305.13048), [RetNet](https://arxiv.org/abs/2307.08621)

### 6.4 Connection to Old MoE Literature

Original MoE (Jacobs 1991, Jordan & Jacobs 1993)就有expert specialization的motivation，但deep learning时代的MoE大部分focus on engineering (sparse activation, load balance)。STAR回归到原始motivation：**expert应该specialize along input structure**。

Reference: [Jacobs et al. 1991](https://doi.org/10.1162/neco.1991.3.1.79), [Jordan & Jacobs 1993](https://ieeexplore.ieee.org/document/298432)

### 6.5 Connection to Mixture of Subspaces

Classic mixture of experts其实是mixture of subspaces：每个expert学习一个input region。STAR显式地estimate这些subspace。这和[Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)的精神是一致的。

### 6.6 Hebbian Learning的复兴

GHA是Hebbian learning的generalization。最近几年有renewed interest in local learning rules：
- Forward-forward algorithm (Hinton)
- Equilibrium propagation
- Various bio-plausible learning

STAR用了GHA因为它的online + unsupervised + local性质，正好适合streaming training。Reference: [Forward-Forward Algorithm](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

### 6.7 Possible Extension: Continual Learning

GHA的online nature让router能adapt to new data分布。这本质上是**unsupervised continual learning**。Test-time adaptation是lightweight版本，但可以扩展到：新domain pretraining时，GHA持续adapt，不需要retrain整个router。

### 6.8 和Spectral Methods in Deep Learning的联系

- Spectral Norm Regularization (for GANs)
- Batch spectral normalization
- Power method for finding top eigenvector (类似GHA但iterative)

GHA可以看作power method的stochastic online version。Reference: [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)

---

## 7. 可能的Limitations和Open Questions

1. **GHA convergence speed**：Figure 7显示$m$需要≥3才能well-approximate SVD。对于non-stationary distribution（比如LLM pretraining中representation本身在变），GHA能否keep up？

2. **V和R的interaction**：$V$是online更新（Hebbian），$R$是gradient更新。这两个dynamic的interaction是否stable？Paper没有fully analyze。

3. **Large K regime**：实验做到$K=16$ (GLUE) 和 $K=8$ (LLaMA-MoE)。DeepSeek-V2有160 expert，STAR能否scale？$K=160$时$R$是$160 \times 160$，computational cost仍是$\mathcal{O}(K^2 d)$但绝对值增大。

4. **Heterogeneous experts**：现在所有expert同构，如果expert本身有different capacity/specialization，router如何adapt？

5. **Multi-modal MoE**：Vision-Language model的MoE，input structure跨modality，单一subspace是否sufficient？

---

## 8. Architecture Diagram解析（Figure 1）

虽然paper里Figure 1的图我没法直接看，但根据Algorithm 1可以reconstruct：

```
Input x ∈ R^d
    │
    ├──→ [GHA Update] ←── V (K×d, online updated)
    │         │
    │         ↓
    │       V (updated)
    │         │
    │         ↓
    │       Z = R·V   ←── R (K×K, learnable)
    │         │
    │         ↓
    │    l_GHA = x·Z^T  (K-dim)
    │
    ├──→ [Linear Gate] ←── W_g (K×d, learnable)
    │         │
    │         ↓
    │    l_linear = x·W_g^T  (K-dim)
    │
    ↓
σ(α)⊙l_linear + (1-σ(α))⊙l_GHA   ←── α (K-dim, learnable)
                │
                ↓
            Softmax
                │
                ↓
            s ∈ R^K (routing scores)
                │
                ↓
          Top-k Selection
                │
                ↓
       Σ_{k∈Topk} s_k · f_k(x)   ←── {f_k} experts
```

这个设计有几个nice properties：
- **Two information streams**：supervised (linear) + unsupervised (GHA)
- **Learnable interpolation**：α让model自动balance
- **Decoupled basis from routing**：R打破variance hierarchy

---

## 9. 总结：STAR的Core Insight

把MoE routing重新理解为**online subspace learning**。Router不应该只是learn a linear map，而是应该**recognize and respond to** input的dominant structure。GHA提供了一个principled、online、unsupervised的方式来estimate这个structure，$R$让expert utilization decouple from variance hierarchy，interpolation让supervised和unsupervised signal互补。

这个视角让我想起deep learning里一个反复出现的theme：**data structure prior比learned prior更stable、更generalizable**。BatchNorm用batch statistics，contrastive learning用augmentation structure，STAR用principal subspace。都是把unsupervised structure注入到learned model中。

Code: https://github.com/psmiz/STAR

---

## 10. 相关Reference汇总

**MoE基础**：
- [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [Switch Transformer (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961)
- [GShard (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [Mixtral](https://arxiv.org/abs/2401.04088)

**Routing variants**：
- [Expert Choice Routing (Zhou et al., 2022)](https://arxiv.org/abs/2202.09368)
- [ReMoE (Wang et al., 2025)](https://arxiv.org/abs/2412.14711)
- [Auxiliary-loss-free Load Balancing](https://arxiv.org/abs/2408.15664)
- [Cosine Router / Sparse MoE as Domain Generalizable Learners](https://arxiv.org/abs/2206.04046)

**Hebbian / PCA learning**：
- [Sanger 1989 - GHA](https://www.sciencedirect.com/science/article/pii/0893608089900517)
- [Oja's Rule](https://www.sciencedirect.com/science/article/pii/0022247X8590159X)

**Subspace methods in DL**：
- [MoEfication (Zhang et al., 2022)](https://arxiv.org/abs/2110.01786)
- [Unlocking Emergent Modularity (Qiu et al., 2024)](https://arxiv.org/abs/2310.10908)
- [Demons in the Detail: Load Balancing Loss Analysis](https://arxiv.org/abs/2501.11873)

**相关evaluation**：
- [GLUE-X](https://arxiv.org/abs/2211.08073)
- [ImageNet-C](https://arxiv.org/abs/1903.12261)
- [In-context Learning as Implicit Bayesian Inference (GINC)](https://arxiv.org/abs/2111.02080)

希望这个解读对你build intuition有帮助。整个paper的核心message是：**routing应该structure-aware，且这个structure可以online、unsupervised地学到**。这个视角对future MoE设计应该有比较深远的影响。
