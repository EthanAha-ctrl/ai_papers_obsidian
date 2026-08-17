---
source_pdf: OmniOpt Taxonomy, Geometry, and Benchmarking of Modern.pdf
paper_sha256: 62afd6af2d5463057172ec575d129d257447803b16b0a7d39bbe872351318a00
processed_at: '2026-08-05T23:22:47-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OmniOpt 人话版

## 一句话总结

这paper干了一件事:把市面上100多个optimizer用同一套语言理清楚,然后跑了个大规模实验告诉你"什么时候该用什么"。

---

## 为什么要搞这个

现在的optimizer论文有个毛病:每个人只讲自己那个小mechanism,什么"我加了Nesterov""我用了sign""我做了low-rank",但你根本看不出来这跟别人有啥关系,能不能组合,为啥有时候work有时候不work。

举个例子:Lion到底算啥?有人说它是"Adam替代品",有人说它是"sign optimizer",有人说它是"memory-efficient method"。这三个描述都对但都没抓住重点。如果你按"memory-efficient"把它跟AdaFactor、8-bit Adam归一类,那就乱套了——因为Lion省内存只是sign机制的副产物,它真正改的是update方向。

所以作者说:咱别按名字分类,按**它在update pipeline的哪个位置动手**来分。

---

## 核心insight: 五个Stage

把一次optimizer update拆成五步:

```
Gradient进来 → 路由(S1) → 变换(S2) → 存状态(S3) → 还原(S4) → 写回(S5)
```

关键发现:**大多数optimizer只在1-2步做实事,剩下都是identity**。

- **AdamW**: 只在S3(存moment)和S5(decoupled weight decay)动手
- **Muon**: 只在S1(把matrix挑出来)和S2(做orthogonalization)动手  
- **GaLore**: S2投影到low-rank + S3在子空间跑Adam + S4投影回来
- **Lion**: S2取sign + S3只存first moment
- **SAM**: S0(扰动后取gradient) + S5(写回)

这个视角的好处:一眼能看出哪些能组合。不同stage的mechanism天然可以stack,比如MARS的variance reduction(S0/S3)能套在AdamW、Lion、Shampoo上,这就是为啥有MARS-AdamW、MARS-Lion、MARS-Shampoo。但同一stage的两个东西要组合就得想清楚顺序,比如low-rank projection和spectral orthogonalization都在S2,你得决定先投影再正交还是先正交再投影——结果完全不同。

---

## 几何统一: LMO

这是paper最漂亮的部分。核心idea一句话:**不同的optimizer只是在不同的norm ball里找最陡方向**。

Linear Minimization Oracle (LMO) 就是:给你一个信号,在一个constraint set里找内积最小的点。当constraint set是norm ball时,就变成找最陡下降方向。

三个经典例子:

**$\ell_2$ ball** → 方向是 $g/\|g\|_2$,就是normalized gradient
**$\ell_\infty$ ball** → 方向是 $\text{sign}(g)$,就是Lion!
**Spectral-norm ball** → 方向是 $UV^\top$ (SVD的polar factor),就是Muon!

所以Lion和Muon其实是同一回事,只是用了不同形状的ball。Lion用cube(每个坐标独立),Muon用spectral ball(矩阵级别的方向)。

更妙的是Adam:它的ball是**动态的**,边界由 $|m_t|/\sqrt{v_t}$ 决定。所以Adam是一个adaptive LMO,constraint set本身在变。

这个统一视角让你能直接比较:sign是coordinate-wise的极值,spectral是matrix-wise的极值,Adam是adaptive的极值。

---

## 四个Axis

作者把上面进一步拆成四个axis,对应update过程的四个决策:

**Axis I: 在哪个空间update?**
- 全空间: AdamW, Muon
- 矩阵空间: Muon (保持矩阵结构)
- Low-rank子空间: GaLore

**Axis II: 用什么信号?**
- Raw gradient: SGD
- Momentum: SGDM
- First + second moment: Adam
- Variance-reduced: MARS

**Axis III: 怎么生成方向?**
- Sign map: Lion
- Diagonal preconditioner: Adam
- Polar/SVD: Muon
- Kronecker metric: Shampoo

**Axis IV: 怎么写回?**
- Learning rate
- Weight decay
- Projection-back (GaLore)
- Layer-wise scaling (LAMB)

中间两个axis最重要:Axis II是"信号质量",Axis III是"方向几何"。把它们分开是关键——variance reduction是改善信号(II),换norm ball是改方向(III),两者独立,可以组合。

---

## 五个Family

按主mechanism分:

**T1 - Adam家族**: 保持element-wise结构,改moment估计、时间尺度、自动调lr。包括AdamW, RAdam, NAdam, AdaBelief, Adan, AdEMAMix, MARS-AdamW, Schedule-Free, Prodigy等。

**T2 - 矩阵方法**: 把weight当矩阵处理。三条路线:
- Spectral orthogonalization (Muon): 保持奇异向量,拍平奇异值
- Kronecker preconditioning (Shampoo, SOAP): 用行列因子近似Hessian
- Low-rank projection (GaLore): 投影到小子空间里跑Adam

**T3 - Sign离散化**: 用sign map丢掉幅度信息。Lion是核心,还有SignSGD, MARS-Lion, FOCUS等。Lion只存一个moment(省内存是副产物),方向是 $d_t = \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t)$。

**T4 - 压缩状态**: 直接减optimizer state。AdaFactor做行列分解,O(mn)→O(m+n);8-bit Adam做量化;Adam-mini做block sharing;APOLLO做随机投影;LOMO直接在backward时流式更新。

**T5 - 几何正则**: 在update外面套一层。SAM做对抗扰动(两次forward-backward),Sophia用对角Hessian,Sophia用clipped Newton,Cautious做方向一致性mask,LAMB做layer-wise trust ratio。

---

## 六个评估维度

作者不只看PPL,看六个东西:

- **O1**: 收敛质量 (PPL)
- **O2**: 每步计算开销 (runtime)
- **O3**: 内存 (optimizer state)
- **O4**: 稳定性 (gradient norm的波动)
- **O5**: 对超参的鲁棒性 (lr扰动测试)
- **O6**: 泛化 (跨数据/长度/架构)

---

## 实验说了啥

### Stage 1: C4短context筛选

24个optimizer,4个scale (60M到1B),seq=256,关掉weight decay和gradient clipping。

1B结果:

| Optimizer | PPL | Memory (GB) | Time (ms) |
|-----------|-----|-------------|-----------|
| APOLLO | 13.53 | 0.79 | 28.65 |
| MARS-Shampoo | 13.72 | 7.48 | 513.7 |
| Muon | 13.72 | 2.50 | 379.0 |
| RMNP | 13.87 | 2.50 | 16.94 |
| SOAP | 14.04 | 29.30 | 1371.5 |
| AdamW | 14.48 | 4.99 | 18.62 |
| Lion | 17.02 | 2.49 | 12.48 |

一眼看出:
- **APOLLO**: 短context之王!PPL最低,内存极低
- **SOAP**: 质量好但1371ms/step,贵到离谱
- **RMNP**: sweet spot!质量接近Muon但快20倍
- **Lion**: 最快但PPL差
- **AdaFactor**: 内存4KB但PPL中等

Pareto分析显示:**没有universal winner**。PPL-runtime-memory三角各有冠军。

### Stage 2: 长context + 跨架构

换FineWeb-Edu,32k context,4个架构(Transformer++, Gated DeltaNet, DeltaNet, GLA),340M和1B。

**核心发现**:

1. **APOLLO崩了**: 从Stage 1的13.53跌到34.08!从冠军变垫底。原因:APOLLO用固定low-rank投影压缩state,长context下gradient的有效rank升高,固定投影丢失太多信息。

2. **SOAP最稳**: 8个场景里7个排第一。Kronecker basis跨架构transfer最好。但代价是runtime/memory极高。

3. **Muon看架构**: 标准Transformer上中等,GLA上最好。spectral geometry跟参数topology耦合。

4. **MARS-AdamW是T1最稳的增强**: STORM-style variance reduction改善信号,保持AdamW几何,跨场景稳定。

### 长度敏感性实验

固定数据和架构,只改context length 256→32k:

| Optimizer | 256 | 32k | 变化 |
|-----------|-----|-----|------|
| APOLLO | 13.53 | 35.40 | +21.87 |
| Muon | 13.72 | 22.54 | +8.81 |
| SOAP | 14.04 | 21.62 | +7.58 |
| AdamW | 14.48 | 21.87 | +7.39 |
| Lion | 17.02 | 23.31 | +6.29 |

APOLLO退化是AdamW的3倍。这是**rank-bounded compression**的硬限制:压缩省的内存,在长context下用质量还了。

### 稳定性分析

看gradient norm的coefficient of variation (GNormCV):

- **Muon最稳**: spectral orthogonalization把update scale和gradient magnitude解耦,damp了相对波动
- **GLA架构让所有optimizer爆炸**: GNormCV从~1飙到10-160,但所有run都完成了(零NaN/Inf)
- **"没diverge"不等于稳定**: 传统判断只看是否NaN/Inf,但GNormCV暴露soft instability

### LR扰动测试

测0.2×, 1×, 5× tuned lr:

- **Lion最flat** (s=0.7%): 但因为fixed-magnitude update本来quality就弱,flat是"低水平稳定"
- **APOLLO最敏感**: 5×时steeply rise,跟它的长context崩溃一致
- **AdamW中等**: 5×侧敏感

T3的local tolerance来自direction-bounded updates: $\|d_t\|_\infty=1$ 限制了LR error的放大。

---

## Muon Ablation: 拆解看什么真正work

这是最mechanistic的实验,拆Muon的组件:

### 核心发现

1. **NS orthogonalization是核心**: 去掉 $v_t$ (Adam的second moment) PPL从17.78→70.74(灾难);加NS恢复到16.86(超过AdamW)。NS就是Muon用来替代AdamW diagonal scaling的东西。

2. **LR scaling和Nesterov是次要增益**: 加LR scaling 16.86→16.76,加Nesterov→16.60。Symmetric two-way scaling最好(16.52)。

3. **顺序很重要**:
   - Momentum在正交化空间积累: 23.01 (烂)
   - Momentum在原始空间积累: 16.60 (好)
   - LR scaling在NS之前: 16.70 (烂)
   - LR scaling在NS之后: 16.52 (好)

### 跨架构验证

| 场景 | 标准Muon | +Sym LR | +Post-NS Nesterov | 两个都加 |
|------|----------|---------|-------------------|----------|
| C4 350M | 16.60 | 16.52 | 16.57 | 16.51 |
| C4 1B | 13.72 | 13.64 | 13.64 | 13.58 |
| GDN 340M | 24.26 | 24.02 | 24.12 | 24.12 |

**标准Transformer上: gains可叠加**
**Gated DeltaNet上: 叠加效应消失!**

这说明optimizer组件的orthogonality本身是architecture-dependent的。同一个trick在不同架构上interaction不同。

---

## 最终建议

作者把24个optimizer分三档:

**Tier I (主推)**:
- **AdamW**: 默认baseline,稳定便宜好调
- **RMNP**: 要质量又要效率时选它,matrix-structured但runtime接近lightweight
- **Muon**: 强力且mechanism清晰,但注意架构敏感性

**Tier II (看场景)**:
- **SOAP**: 长context质量天花板,但贵到 prohibitive
- **MARS-AdamW**: T1里最稳的增强
- **APOLLO**: 短context内存紧张时高回报高风险
- **AdaFactor**: 安全的low-memory baseline
- **Lion**: 廉价探索,预期有quality gap

**Tier III (诊断用)**:
- RAdam, NAdam, Prodigy, AdaBelief, GaLore, Shampoo, Sophia, LAMB等

### 选optimizer的decision rule

先问:你的binding constraint是什么?

- **Stability + 成本**: AdamW
- **Quality + efficiency**: RMNP  
- **Final quality, cost不care**: SOAP
- **Memory, short context**: APOLLO (激进) 或 AdaFactor (保守)
- **Cheap exploration**: Lion

---

## 真正的takeaway

1. **没有universal best optimizer**,只有"匹配你的binding constraint"的optimizer

2. **Geometry-sensitive direction maps (Axis III) 是主要benefit carrier**: Muon的NS, SOAP的Kronecker basis。Scalar tweaks (T1.1那些)贡献很小

3. **State compression是rank-bounded的**: APOLLO短context冠军长context垫底,因为固定rank投影在gradient有效rank升高时丢失信息

4. **Spectral geometry是architecture-conditional的**: Muon在标准Transformer上stackable,在Gated DeltaNet上additivity消失

5. **Composability由locality决定**: 不同axis/stage的可以stack,同slot的要explicit ordering

6. **"没diverge"不等于stable**: GNormCV暴露的soft instability比NaN/Inf判断更informative

---

## 我的联想

1. **APOLLO的失败是information-theoretic的**: 你不能用固定大小的summary去压缩信息量随context增长的gradient。Adaptive rank是明显future direction——让projection rank随gradient effective rank动态调整。

2. **Muon的mystery**: Shumaylov et al. [100] 发现"random or inverted spectra work just as well"。如果真的,那NS的core不在spectral structure本身,而在"把singular values uniform化"这个动作。这跟这篇paper的ablation不矛盾——NS是核心,但为啥work还是open。

3. **Linear attention的instability**: GLA让所有optimizer的GNormCV爆炸。这可能跟linear attention的eigenvalue distribution有关——没有softmax的 normalization,gradient path更长更不稳定。

4. **跟你的nanoGPT**: 60M-130M scale上Muon可能值得试。Stage 1显示Muon在小scale上领先。但Stage 2警告:架构sensitive,标准Transformer应该OK。

5. **Optimizer as kernel method**: LMO本质是kernel operation——选最陡方向。不同norm ball = 不同kernel。这视角可能连到kernel literature。

6. **Composition search = NAS for optimizers**: 把four-axis coordinate作为search space,自动找最优combination。这比手动设计MARS-AdamW等更systematic。

7. **Scaling law interaction**: optimizer改变scaling law的prefactor还是exponent?这paper没回答,但是个key question。如果optimizer只改prefactor,那小scale实验对大scale有指导意义;如果改exponent,那小scale结论可能完全误导。

8. **Architecture-optimizer co-design**: RMNP是一个example——用architecture-induced row-wise structure设计preconditioner。这可能是future的主流方向:不是设计universal optimizer,而是为特定architecture class设计tailored optimizer。

---

## Web Links

- [OmniOpt Paper](https://arxiv.org/abs/2509.02046)
- [Fantastic Pretraining Optimizers](https://arxiv.org/abs/2509.02046)
- [Muon Blog](https://kellerjordan.github.io/posts/muon/)
- [SOAP Paper](https://arxiv.org/abs/2409.11321)
- [GaLore Paper](https://arxiv.org/abs/2403.03507)
- [Lion Paper](https://arxiv.org/abs/2302.06675)
- [Sophia Paper](https://arxiv.org/abs/2305.14342)
- [SAM Paper](https://arxiv.org/abs/2010.01412)
- [Old Optimizer New Norm](https://arxiv.org/abs/2409.20325)
- [Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529)
- [APOLLO Paper](https://arxiv.org/abs/2410.21217)
- [AdaFactor Paper](https://arxiv.org/abs/1804.04235)
- [8-bit Adam](https://arxiv.org/abs/2110.02861)
- [Adam-mini Paper](https://arxiv.org/abs/2410.04391)
- [MARS Paper](https://arxiv.org/abs/2411.10438)
- [Schedule-Free Paper](https://arxiv.org/abs/2407.85866)
- [Cautious Optimizers](https://arxiv.org/abs/2411.16085)
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)
- [Deconstructing Optimizers](https://arxiv.org/abs/2407.07972)

---

# OmniOpt: 一篇关于现代Optimizer的统一框架与基准测试

Hey Andrej! 这篇paper试图解决一个很实际的问题:现在有100多个optimizer,但它们的语言、机制、实验协议都不兼容,很难比较。作者提出一个统一的"坐标系"来理解这些方法,并做了一个大规模benchmark来验证。我详细讲讲。

---

## 1. 核心Motivation: 为什么optimizer选择变成system-level decision

传统上optimizer选择是一个算法问题,但在LLM时代,它变成了一个受多约束的系统设计问题:
- **Compute budget**: 固定GPU数量
- **Memory**: optimizer state memory (Adam需要2个FP32 tensor,7B模型仅Adam state就要56GB)
- **Tuning budget**: hyperparameter search的成本
- **Task diversity**: pretraining vs fine-tuning vs long-context

Memory公式 (BF16训练):
$$M_{\text{train}} \approx 4d + 2d + 2d + S_{\text{opt}}d + M_{\text{act}}$$

其中:
- $4d$: FP32 master weights ($d$是总参数数,FP32占4 bytes/param)
- $2d$: BF16 model copy
- $2d$: BF16 gradients  
- $S_{\text{opt}}d$: optimizer state (Adam = 8 bytes/param,因为$m_t$和$v_t$都是FP32)
- $M_{\text{act}}$: activations + temp buffers

这就是为什么memory-efficient optimizers (T4 family) 在LLM场景如此重要。

---

## 2. Universal Meta-Pipeline: 五阶段抽象

这是paper的核心insight之一。作者观察到**大多数optimizer只在1-2个pipeline stage做实质性工作,其余stage都是identity mapping**。

### 2.1 五个Stage的定义

对于参数矩阵 $W_t \in \mathbb{R}^{m \times n}$,gradient $G_t \in \mathbb{R}^{m \times n}$,state $S_{t-1}$:

$$\Delta_t = \mathbf{S5}(\mathbf{S4}(\mathbf{S3}(\mathbf{S2}(\mathbf{S1}(G_t)); S_{t-1}); S_{t-1}); S_{t-1}, W_t)$$

**S0: Training Signal Acquisition** (信号获取)
- 标准情况: first-order gradient $G_t = \nabla_W \mathcal{L}(\hat{W}_t; \xi_t)$
- Variance-reduced: STORM-style $\tilde{G}_t = G_t - G_{t-1}^{\xi_t} + M_{t-1}$
- Curvature-augmented: Hessian-vector product $h_t \approx u \odot (Hu)$, $u \sim \mathcal{N}(0, I)$ (Sophia)
- Zeroth-order: 边界case,不在主taxonomy内

**S1: Parameter Scoping and Routing** (参数路由)
$$\rho^{(i)} = \mathcal{R}(\text{shape}(\theta^{(i)}), \text{module-type}(\theta^{(i)}))$$
- 决定哪些参数走matrix route (2D weights),哪些走element-wise route (biases, norms)
- AdamW: all params single route → S1是identity
- Muon: matrices走orthogonalization,vector params走SGD fallback

**S2: Gradient Transformation** (梯度变换)
$$\hat{G}_t = \mathcal{T}(G_t; \mathcal{S}_{t-1}) \in \mathbb{R}^{r \times s}$$
- Identity (T1, most T4/T5)
- Newton-Schulz orthogonalization: $\hat{G}_t = \text{NS}_k(M_t)$ 使得 $\hat{G}_t^\top \hat{G}_t \approx I$ (Muon)
- Kronecker preconditioning: $\hat{G}_t = L_t^{-1/4} G_t R_t^{-1/4}$ (Shampoo)
- Low-rank projection: $\hat{G}_t = P_t^\top G_t \in \mathbb{R}^{r \times n}$ (GaLore)
- Sign discretization: $\hat{G}_t = \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) G_t)$ (Lion)

**S3: State Evolution** (状态演化)
$$S_t = f(S_{t-1}, \hat{G}_t)$$
- Adam: $m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat{G}_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat{G}_t^{\odot 2}$
- Shampoo: $L_t = \text{EMA}_\beta(G_t G_t^\top)$, $R_t = \text{EMA}_\beta(G_t^\top G_t)$
- AdaFactor: row/column factored $r_t \in \mathbb{R}^m$, $c_t \in \mathbb{R}^n$, $v_t \approx r_t c_t^\top$
- Stateless: bypass S3 (SignSGD)

**S4: Update Reconstruction** (更新重建)
$$\hat{\Delta}_t = \mathcal{R}(\tilde{\Delta}_t; \mathcal{S}_t) \in \mathbb{R}^{m \times n}$$
- Low-rank: $\hat{\Delta}_t = P_t \tilde{\Delta}_t$ (GaLore)
- Kronecker: $\hat{\Delta}_t = Q_L \tilde{\Delta}_t Q_R^\top$ (SOAP)
- Identity (大多数T1/T3/T5)

**S5: Update Finalization** (更新定稿)
$$W_{t+1} = W_t - \eta_t \cdot \phi_t^{(l)} \cdot \mathcal{C}(\mathcal{F}(\hat{\Delta}_t)) - \eta_t \lambda W_t$$
- $\eta_t$: global learning rate
- $\phi_t^{(l)}$: layer-wise trust ratio (LAMB)
- $\mathcal{F}$: post-update filter (Cautious, gradient centralization)
- $\mathcal{C}$: clipping operator
- $\lambda$: weight decay coefficient

### 2.2 代表性instantiations

| Method | Active stages | Core mechanism |
|--------|---------------|----------------|
| AdamW (T1.1) | S3, S5 | Moment EMAs (S3) + decoupled weight decay (S5) |
| Muon (T2.1) | S1, S2 | Matrix routing (S1) + Newton-Schulz orthogonalization (S2) |
| GaLore (T2.3) | S1-S4 | Low-rank projection (S2/S4), subspace Adam state (S3) |
| Lion (T3) | S2, S3 | Momentum interpolation (S3) + sign discretization (S2) |
| SAM (T5.1) | S0, S5 | Perturbation-induced gradient (S0) + neighborhood-regularized writeback (S5) |

### 2.3 Composition intuition

- 不同stage的mechanism通常可以stack: variance-reduced signal (S0) + AdamW + trust-ratio (S5)
- 同一stage的mechanism需要explicit ordering: low-rank projection vs spectral orthogonalization (都是S2),必须指定哪个先

---

## 3. LMO-driven Four-Axis Decomposition: 几何统一

这是paper最有理论深度的部分。作者用**Linear Minimization Oracle (LMO)** 作为统一语言,把sign、spectral orthogonalization、Kronecker preconditioning等都纳入同一几何框架。

### 3.1 LMO基础

给定凸集 $\mathcal{D}$ 和信号 $s$:
$$\text{lmo}_{\mathcal{D}}(s) \in \arg\min_{x \in \mathcal{D}} \langle s, x \rangle$$

当 $\mathcal{D}$ 是norm ball $\mathcal{D}_\rho = \{x : \|x\| \leq \rho\}$ 时:
$$\text{lmo}_{\mathcal{D}_\rho}(s) = -\rho \cdot u^\sharp(s)$$

其中 $u^\sharp(s)$ 是单位球上的steepest ascent direction:
$$u^\sharp(s) \in \arg\max_{\|u\| \leq 1} \langle s, u \rangle$$

**Key insight**: norm ball的形状决定了什么算"steepest"。不同optimizer其实是在选择不同的norm ball。

### 3.2 三种canonical norm geometries

**(a) Euclidean ball ($\ell_2$)** → normalized gradient:
$$\mathcal{D}_2 = \{x : \|x\|_2 \leq \rho\}, \quad \text{lmo}_{\mathcal{D}_2}(g) = -\rho \frac{g}{\|g\|_2}, \quad \Phi(g) = \frac{g}{\|g\|_2}$$

**(b) Max-norm ball ($\ell_\infty$)** → sign direction:
$$\mathcal{D}_\infty = \{x : \|x\|_\infty \leq \rho\}, \quad \text{lmo}_{\mathcal{D}_\infty}(g) = -\rho \cdot \text{sign}(g), \quad \Phi(g) = \text{sign}(g)$$

这解释了为什么Lion的update天然有bounded per-coordinate magnitude。

**(c) Spectral-norm ball** → matrix polar direction:
$$\mathcal{D}_{S_\infty} = \{X : \|X\|_{S_\infty} \leq \rho\}, \quad \text{lmo}_{\mathcal{D}_{S_\infty}}(M) = -\rho \cdot UV^\top, \quad \Phi(M) = UV^\top$$

其中 $M = U\Sigma V^\top$ 是SVD。这是Muon的几何本质!

**(d) Adam的adaptive box**:
$$\mathcal{D}_t^{\text{Adam}} = \{x : |x_i| \leq \rho_t b_{t,i}, \forall i\}, \quad b_{t,i} = |m_{t,i}| / \sqrt{v_{t,i}}$$

Adam是一个**dynamic-boundary LMO**,constraint set本身由state estimator产生。这是很elegant的视角。

### 3.3 Four Axes

$$\underbrace{(M_t, H_t, \mathcal{D}_t)}_{\text{Axis II}} = \text{StateEstimator}_t(g_t, \text{State}_{t-1})$$
$$D_t = \underbrace{\Phi_t}_{\text{Axis III}}(M_t; H_t, \mathcal{D}_t)$$
$$W_{t+1} = \underbrace{\text{Finalize}}_{\text{Axis IV}}(W_t, D_t)$$

- **Axis I (Update domain)**: $\mathcal{X}_t$ 或 $(Q_L, Q_R)$ — update在哪个空间表达
  - Full space: SGD, AdamW, Muon
  - Low-rank subspace: GaLore ($Z_t = Q_L^\top M_t Q_R$, 然后 $D_t = Q_L \Phi_t(Z_t) Q_R^\top$)
  
- **Axis II (State estimator)**: 产生 $(M_t, H_t, \mathcal{D}_t)$
  - SGD: $M_t = g_t$, $H_t = I$
  - Adam: $M_t = m_t$, $H_t = \text{diag}(v_t)$
  - MARS: $c_t = g_t^{\xi_t} + \gamma_t \frac{\beta_1}{1-\beta_1}(g_t^{\xi_t} - g_{t-1}^{\xi_t})$, 然后 $m_t = \text{EMA}(c_t)$

- **Axis III (Geometry/Precondition operator)**: $\Phi_t$
  - LMO reading: $\Phi_t(M_t) = -\frac{1}{\rho_t} \text{lmo}_{\mathcal{D}_t}(M_t)$
  - Preconditioner reading: $\Phi_t(M_t) = H_t^{-\alpha} M_t$ ($\alpha$是内部指数)

- **Axis IV (Finalization)**: learning rate, weight decay, projection-back, routing

### 3.4 Muon的dual reading

Muon是最elegant的例子。对于 $M_t \in \mathbb{R}^{m \times n}$, $m < n$:

**LMO reading**: spectral-norm ball
$$\Phi_t(M_t) = U_t V_t^\top = \arg\max_{\|X\|_{S_\infty} \leq 1} \langle M_t, X \rangle$$

由von Neumann trace inequality: $\langle M_t, X \rangle \leq \sum_i \sigma_i(M_t) \sigma_i(X) \leq \sum_i \sigma_i(M_t)$,等号在 $X = U_t V_t^\top$ 取得。

**Preconditioner reading**: left Gram Hessian
$$H_t = M_t M_t^\top, \quad \Phi_t(M_t) = H_t^{-1/2} M_t = (M_t M_t^\top)^{-1/2} M_t$$

代入 $M_t = U_t \Sigma_t V_t^\top$:
$$(U_t \Sigma_t V_t^\top V_t \Sigma_t U_t^\top)^{-1/2} U_t \Sigma_t V_t^\top = (U_t \Sigma_t^2 U_t^\top)^{-1/2} U_t \Sigma_t V_t^\top = U_t V_t^\top$$

**两个reading完全一致!** 这说明Muon同时是spectral-norm LMO和Gram preconditioner。

---

## 4. Dual-Dimension Taxonomy

### 4.1 Dimension A: Methodological taxonomy (机制维度)

五个non-overlapping families:

**T1: Element-wise adaptive moment and scalar control** (25 methods)
- T1.1: Direct Adam variants (AdamW, RAdam, NAdam, AdaBelief, Adan, ADOPT, EXAdam)
- T1.2: Multi-timescale / variance reduction (AdEMAMix, MARS-AdamW, MADGRAD, TAM)
- T1.3: Iterate averaging / auto-tuning (Schedule-Free, D-Adaptation, Prodigy, SWATS)

**T2: Matrix-level structural methods** (~19 methods)
- T2.1: Spectral orthogonalization (Muon, RMNP, Dion, AdaMuon, OrthoGrad)
- T2.2: Kronecker preconditioning (Shampoo, SOAP, MARS-Shampoo, PSGD, Kron)
- T2.3: Low-rank projection (GaLore, Fira, Alice)

**T3: Discretization and directional quantization** (7 methods)
- Pure sign (SignSGD, Signum)
- Signed momentum (Lion, MARS-Lion)
- Smooth/hybrid (RLion, FOCUS, Ano)

**T4: State compression and structural aggregation** (~11 methods)
- T4.1: Row-column factorization (AdaFactor, CAME)
- T4.2: Low-bit quantization (8-bit Adam, Q-GaLore)
- T4.3: Block/layer sharing (Adam-mini, APOLLO, SM3, Conda, NovoGrad)
- T4.4: Fused backprop-update (LOMO, AdaLOMO)

**T5: Curvature-aware and geometric regularization** (~25 methods)
- T5.1: SAM family (SAM, ASAM, GSAM, WSAM, bSAM, LookSAM)
- T5.2: Diagonal Hessian (Sophia, AdaHessian)
- T5.3: Post-processing/filtering (Gradient Centralization, AdamP, Cautious, Grams, SPAM, Magma, MGUP)
- T5.4: Layer-wise trust region (LAMB, AGC)

### 4.2 Dimension B: Objective taxonomy (效果维度)

六个evaluation axes:

| Objective | Definition | Measurement |
|-----------|------------|-------------|
| O1 Convergence | Loss reduction under fixed budget | Final PPL, steps-to-threshold |
| O2 Step cost | Extra per-step computation | Wall-clock, FLOPs, extra backward count |
| O3 Memory | Optimizer state + buffers | Peak memory, state bytes |
| O4 Stability | Robustness to spikes/divergence | GNormCV, spike rate |
| O5 Hyperparameter robustness | Sensitivity to LR/decay/batch | LR sweep width, performance variance |
| O6 Generalization | Quality beyond training objective | Validation, downstream, OOD transfer |

### 4.3 Cross-dimension matrix (Table 7)

| Family | O1 | O2 | O3 | O4 | O5 | O6 |
|--------|----|----|----|----|----|-----|
| T1 | ⇑ | ○ | → | ⇑ | ⇑ | ↑ |
| T2 | ⇑ | ↓ | → | ⇑ | ↑ | ↑ |
| T3 | ← | ⇑ | ⇑ | ↑ | ○ | ○ |
| T4 | → | ← | ⇑ | → | ○ | ↑ |
| T5 | ↑ | ↓ | → | ⇑ | ⇑ | ⇑ |

- ⇑/⇓: strong prior (favorable/cost)
- ↑/↓: conditional favorable/cost
- ○: protocol-dependent neutrality
- →/←: trade-off direction

---

## 5. Key Formulas per Family

### 5.1 AdamW baseline
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t$$
$$u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, \quad \theta_{t+1} = (1 - \eta_t \lambda) \theta_t - \eta_t u_t$$

- $\hat{m}_t = m_t / (1-\beta_1^t)$: bias-corrected first moment
- $\hat{v}_t = v_t / (1-\beta_2^t)$: bias-corrected second moment
- $\lambda$: weight decay coefficient
- $\beta_1, \beta_2$: EMA decay (typically 0.9, 0.99)

### 5.2 Muon
$$M_t = \beta M_{t-1} + (1-\beta) G_t, \quad M_t = U_t \Sigma_t V_t^\top$$
$$W_{t+1} = W_t - \eta_t U_t V_t^\top$$

实际用Newton-Schulz迭代近似 $U_t V_t^\top$:
$$\text{NS}_K(M) \approx M (M^\top M)^{-1/2} \approx UV^\top$$

### 5.3 SOAP
$$L_t = \beta_2 L_{t-1} + (1-\beta_2) G_t G_t^\top, \quad R_t = \beta_2 R_{t-1} + (1-\beta_2) G_t^\top G_t$$
$$G_t' = Q_L^\top G_t Q_R, \quad V_t = \beta_2 V_{t-1} + (1-\beta_2) G_t' \odot G_t'$$
$$N_t' = \frac{M_t'}{\sqrt{V_t} + \epsilon}, \quad N_t = Q_L N_t' Q_R^\top$$

- $Q_L, Q_R$: $L_t, R_t$的eigenvectors (每 $f$ 步refresh一次)
- $f$: preconditioning frequency (唯一额外hyperparameter)

### 5.4 GaLore
$$\tilde{G}_t = P_t^\top G_t, \quad \tilde{\Delta}_t = \text{AdamW}_{\text{subspace}}(\tilde{G}_t), \quad \Delta_t = P_t \tilde{\Delta}_t$$

- $P_t \in \mathbb{R}^{m \times r}$, $r \ll \min(m, n)$: projection matrix
- 内存从 $O(mn)$ 降到 $O(r^2)$
- $P_t$ 周期性refresh (通过SVD of gradient或momentum)

### 5.5 Lion
$$z_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad d_t = \text{sign}(z_t)$$
$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t, \quad \theta_{t+1} = (1-\eta_t \lambda) \theta_t - \eta_t d_t$$

- 只存一个moment (没有 $v_t$)
- $\|d_t\|_\infty = 1$, learning rate直接控制最大坐标位移

### 5.6 Sophia
$$h_t = \beta_2 h_{t-1} + (1-\beta_2) \widehat{\text{diag}}(H_t)$$
$$\Delta_t = \text{clip}\left(\frac{m_t}{\gamma h_t + \epsilon}, -1, 1\right)$$

- $h_t$: diagonal Hessian estimate (via Hessian-vector product)
- $\gamma$: clipping threshold
- Sophia-H用HVP,Sophia-G用gradient-based proxy

### 5.7 SAM
$$\epsilon^* = \rho \frac{g_t}{\|g_t\|_2}, \quad \tilde{g}_t = \nabla_\theta \mathcal{L}(\theta_t + \epsilon^*)$$
$$\theta_{t+1} = \theta_t - \eta_t \cdot \mathcal{U}_{\text{base}}(\tilde{g}_t, S_t)$$

- $\rho$: perturbation radius
- 需要两次forward-backward pass

### 5.8 Cautious Optimizers
$$\mathcal{G}^{\text{cautious}}(\Delta, g) = \Delta \odot \mathbf{1}[\Delta \odot g > 0]$$

只保留update和instantaneous gradient方向一致的coordinates。

---

## 6. Benchmark Study: 核心实验

### 6.1 Setup

**24个optimizer**,覆盖T1-T5和four-axis space的主要region。

**Stage 1**: C4 + LLaMA, seq=256, 四个scale (60M, 130M, 350M, 1B)
- 10k/20k/60k/100k steps
- 禁用weight decay和gradient clipping (隔离S2/S3机制)

**Stage 2**: FineWeb-Edu + 32k context, 340M & 1B, 四个architectures
- Transformer++ (standard attention)
- Gated DeltaNet, DeltaNet, GLA (linear attention variants)
- 启用weight decay + gradient clipping

### 6.2 Stage 1 结果 (Table 13)

1B scale的关键数据:

| Optimizer | PPL | Mem (GB) | Time (ms) | Family |
|-----------|-----|----------|-----------|--------|
| APOLLO | 13.53 | 0.790 | 28.65 | T4 |
| MARS-Shampoo | 13.72 | 7.483 | 513.7 | T2 |
| Muon | 13.72 | 2.495 | 379.0 | T2 |
| RMNP | 13.87 | 2.495 | 16.94 | T2 |
| SOAP | 14.04 | 29.299 | 1371.5 | T2 |
| GaLore | 14.29 | 0.790 | 15.29 | T2.3 |
| Conda | 14.25 | 6.317 | 62.33 | T4 |
| AdamW | 14.48 | 4.989 | 18.62 | T1 |
| RAdam | 14.47 | 4.989 | 23.79 | T1 |
| 8-bit Adam | 14.53 | 2.534 | 42.38 | T4 |
| CAME | 14.53 | 4.997 | 87.46 | T4 |
| AdaFactor | 14.92 | 0.004 | 56.46 | T4 |
| Lion | 17.02 | 2.494 | 12.48 | T3 |

**Key observations**:
1. APOLLO在short-context是最优 (PPL 13.53 + low memory)
2. Muon/MARS-Shampoo质量好但runtime爆炸 (513ms/step!)
3. RMNP是matrix-structured方法中的efficiency sweet spot (16.94ms)
4. Lion最快但PPL差 (17.02)

### 6.3 Pareto Analysis (Figure 14)

**PPL vs Runtime frontier**:
- Lion最快但PPL太高
- RMNP占据关键middle region (高质量+低runtime)
- APOLLO是非dominated的quality-memory点
- Heavy matrix methods (SOAP, Muon, Shampoo)质量好但runtime prohibitive

**PPL vs Memory frontier**:
- AdaFactor内存最低但PPL中等
- APOLLO是Stage 1最强memory-frontier点

### 6.4 Stage 2: Long-context generalization (Table 14)

WikiText PPL across architectures (340M):

| Optimizer | Tr++ | GDN | Delta | GLA | CS Avg (Tr++) |
|-----------|------|-----|-------|-----|---------------|
| SOAP | 23.90 | 23.53 | 26.02 | 27.04 | 53.75 |
| RMNP | 24.37 | 23.65 | 26.80 | 28.60 | 53.35 |
| MARS-AdamW | 24.57 | 24.17 | 26.79 | 28.28 | 52.50 |
| AdamW | 24.62 | 24.47 | 27.16 | 28.67 | 52.28 |
| Muon | 25.05 | 24.34 | 27.18 | 27.47 | 53.25 |
| Lion | 26.02 | 24.76 | 28.20 | 29.47 | 51.07 |
| APOLLO | 34.08 | 30.36 | 34.73 | 37.75 | 48.19 |

**Critical findings**:
1. **SOAP最稳定**: 7/8 scenarios排第一,never leaves top 2
2. **APOLLO崩溃**: 从Stage 1的冠军(13.53)跌到Stage 2的最差(34.08)
3. **Muon architecture-dependent**: 在GLA上最好,在Tr++上中等
4. **MARS-AdamW是T1中最稳定的enhancement**

### 6.5 Sequence-length sensitivity (Table 15)

| Optimizer | 256 | 32k | Δ |
|-----------|-----|-----|---|
| APOLLO | 13.53 | 35.40 | +21.87 |
| Muon | 13.72 | 22.54 | +8.81 |
| SOAP | 14.04 | 21.62 | +7.58 |
| AdamW | 14.48 | 21.87 | +7.39 |
| Lion | 17.02 | 23.31 | +6.29 |

**APOLLO的degradation是AdamW的3倍!**

Framework解释: APOLLO通过random projection压缩state到low-dim subspace。Long context下,gradient的有效rank上升,固定low-dim projection会proportionally丢失更多信息。这就是**rank-bounded compression**的本质限制。

### 6.6 Stability analysis (Figure 17)

**GNormCV** = std(||g_t||) / mean(||g_t||)

- Muon最稳定 (spectral orthogonalization decouples update scale from gradient magnitude)
- GLA architecture导致所有optimizer的GNormCV爆炸 (10-160)
- 但completion status相同 (零NaN/Inf)
- **"Did it diverge?" criterion不够,需要看soft instability**

### 6.7 Learning-rate perturbation (Figure 18)

测试 $0.2\eta^*$, $\eta^*$, $5\eta^*$:

- Lion, MARS-Lion: $s_{LR} = 0.7\%, 7.7\%$ (最flat,但tuned quality弱)
- AdamW, MARS-Shampoo, APOLLO: 高sensitivity
- APOLLO在 $5\times$ 时steeply rise

**T3的local tolerance来自direction-bounded updates**: fixed-magnitude限制了LR misspecification的影响。

### 6.8 Family-level summary (Figure 19)

| Family | O1 Quality | O2 Runtime | O3 Memory | O4 Stability | O5 LR Robust | O6 Generalization |
|--------|-----------|-----------|----------|--------------|--------------|-------------------|
| T1 | moderate | moderate | moderate | moderate | moderate | moderate |
| T2 | strong | expensive | variable | strong (Muon) | moderate | strong (SOAP) |
| T3 | weak | cheap | low | moderate | strong | weak |
| T4 | short-ctx strong | variable | strong | weak | fragile | weak |
| T5 | situational | expensive | moderate | situational | situational | situational |

---

## 7. Muon Mechanistic Ablation (Section 6.3)

这是paper最mechanistic的部分,用Muon做case study验证meta-pipeline view。

### 7.1 Single-scene decomposition (C4-350M, Figure 20)

**Block 1: Core mechanism recovery**
- No $v_t$ (移除second moment): PPL 17.78 → 70.74 (灾难)
- + NS orthogonalization: → 16.86 (恢复并超过AdamW)
- **NS是Muon的核心**,替代AdamW的diagonal adaptive scaling

**Block 2: Gain design**
- + LR scaling: 16.86 → 16.76
- + Nesterov: → 16.60 (full Muon)
- Symmetric two-way LR scaling: 16.52 (best)
- Penalizing wide matrices: 16.71 (worse)
- Post-NS Nesterov: 16.57

**Block 3: Operator order constraints**
- Momentum in orthogonalized space: 23.01 (bad!)
- Momentum in original-gradient space: 16.60 (good)
- LR scaling before NS: 16.70 (bad)
- LR scaling after NS: 16.52 (good)

**Key insight**: Operations不是freely permutable的。Momentum必须在original space accumulate,LR scaling必须在NS之后。

### 7.2 Cross-architecture validation (Table 18)

| Scenario | Standard Muon | Sym LR | Post-NS Nesterov | Both | Best |
|----------|---------------|--------|------------------|------|------|
| C4-LLaMA 350M | 16.60 | 16.52 | 16.57 | 16.51 | Both |
| C4-LLaMA 1B | 13.72 | 13.64 | 13.64 | 13.58 | Both |
| GDN-340M | 24.26 | 24.02 | 24.12 | 24.12 | Sym LR only |

**Standard Transformer**: gains stackable
**Gated DeltaNet**: additivity消失!

这说明**optimizer component的orthogonality是architecture-dependent的**。

---

## 8. Tiered Summary (Table 17)

| Tier | Optimizers |
|------|------------|
| **Tier I** (primary candidates) | Muon, RMNP, AdamW |
| **Tier II** (scenario-dependent) | SOAP, MARS-AdamW, AdamP, MARS-Shampoo, Conda, Adan, Lion, MARS-Lion, APOLLO |
| **Tier III** (diagnostic) | RAdam, NAdam, Prodigy, AdaBelief, GaLore, Shampoo, 8-bit Adam, CAME, AdaFactor, Adam-mini, LAMB, Sophia |

### Selection guidance:

1. **General-purpose pretraining**: AdamW (stable, inexpensive, interpretable baseline)
2. **Quality-efficiency balance**: RMNP (matrix-structured但runtime接近lightweight)
3. **Final quality ceiling**: SOAP (long-context最稳定但prohibitive cost)
4. **Memory-constrained short-context**: APOLLO (high-reward high-risk) 或 AdaFactor (safe)
5. **Cheap exploratory**: Lion (但有quality gap)

---

## 9. Technique-Level Lessons (Section 7.2)

### 9.1 Benefit carriers

- **Geometry-sensitive direction maps (Axis III)**: Muon的NS, SOAP的Kronecker basis
- **Structured state (Axis II)**: MARS的variance-reduced estimator
- **Simple memory techniques preserving geometry**: AdaFactor的row/col factorization, 8-bit Adam

### 9.2 Limited returns

- Small element-wise AdamW refinements: RAdam, NAdam, AdaBelief
- Automatic step-size tuning: Prodigy
- Diagonal curvature proxies: Sophia, LAMB
- Geometric wrappers without consistent return

### 9.3 Compatible compositions (by locality)

- Variance-reduced signal (Axis II) + 任何base direction → MARS-AdamW, MARS-Lion, MARS-Shampoo
- Low-rank projection (T2.3) + state quantization (T4.2) → Q-GaLore
- Post-update filters (T5.3) + any base → Cautious variants
- Layer-wise trust ratios (T5.4) + element-wise adaptive → LAMB

### 9.4 Conflicts

- 两个S2 matrix constraints (spectral orth + low-rank): 需要explicit ordering
- Fused backprop-update (LOMO) vs global operations: gradient lifetime conflict
- SAM + Kronecker: double forward-backward × expensive preconditioning = impractical

### 9.5 Empirical quantitative conflicts

- **APOLLO**: rank-bounded compression, +21.87 PPL degradation at 32k
- **Muon**: architecture-conditional, gains stack on Tr++ but not on GDN
- **Operator order**: momentum in orthogonalized space hurts, LR scaling before NS hurts

---

## 10. Open Problems (Section 7.3)

1. **Diagnostics & adaptive compression**: effective-rank estimate, basis-staleness measure, intrinsic-vs-protocol decomposition
2. **Architecture-aware geometry**: 预测何时geometry transfers, RMNP是example
3. **Multi-objective selection**: Pareto-aware, cost-aware comparison
4. **Compositional search**: 自动化axis-compatible combination search
5. **Cost-effective curvature**: cheaper Hessian/Kronecker estimation
6. **Matched protocols**: standardized evaluation conventions

---

## 11. Build Your Intuition

### 11.1 核心mental model

把optimizer理解为一个**structured transformation pipeline**,大多数method只改1-2个stage。选择optimizer时问三个问题:

1. **Where does it act?** (meta-pipeline stage)
2. **Why does the direction have this form?** (LMO geometry)
3. **What objective does it target?** (effect taxonomy)

### 11.2 Adam vs Muon vs Lion的几何对比

- **Adam**: adaptive $\ell_\infty$ box,dynamic boundary by $|m_t|/\sqrt{v_t}$
- **Muon**: spectral-norm ball,polar direction $UV^\top$
- **Lion**: fixed $\ell_\infty$ ball,sign direction

三者都是LMO,只是norm ball不同!

### 11.3 为什么no universal best optimizer

因为O1-O6 objectives genuinely trade off:
- T2质量好但runtime贵
- T4省memory但long-context崩溃
- T3 robust to LR但tuned quality弱
- T5 situational

选择optimizer = 匹配binding constraint到method的dominant strength。

### 11.4 Composition的locality principle

- 不同axis/stage的mechanism → stackable
- 同一slot的mechanism → need explicit ordering
- Architecture-dependent interactions → must validate on target topology

---

## References & Web Links

- [OmniOpt Paper (arXiv)](https://arxiv.org/abs/2509.02046) - 这篇paper本身
- [Fantastic Pretraining Optimizers](https://arxiv.org/abs/2509.02046) - Wen et al.的optimizer benchmark
- [Muon](https://kellerjordan.github.io/posts/muon/) - Keller Jordan的Muon blog
- [SOAP](https://arxiv.org/abs/2409.11321) - SOAP paper
- [GaLore](https://arxiv.org/abs/2403.03507) - GaLore paper
- [Lion](https://arxiv.org/abs/2302.06675) - Lion paper (symbolic discovery)
- [Sophia](https://arxiv.org/abs/2305.14342) - Sophia paper
- [SAM](https://arxiv.org/abs/2010.01412) - SAM paper
- [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325) - Bernstein的norm geometry anthology
- [Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529) - Pethick et al.的LMO training
- [Lions and Muons](https://arxiv.org/abs/2506.04192) - Frank-Wolfe视角
- [Deconstructing Optimizers](https://arxiv.org/abs/2407.07972) - Zhao et al.的AdamW decomposition
- [Benchmarking LLM Optimizers](https://arxiv.org/abs/2509.01440) - Semenov et al.
- [8-bit Adam](https://arxiv.org/abs/2110.02861) - Dettmers et al.
- [AdaFactor](https://arxiv.org/abs/1804.04235) - Shazeer & Stern
- [Shampoo](https://arxiv.org/abs/1811.05622) - Gupta et al.
- [APOLLO](https://arxiv.org/abs/2410.21217) - Zhu et al.
- [Adam-mini](https://arxiv.org/abs/2410.04391) - Zhang et al.
- [MARS](https://arxiv.org/abs/2411.10438) - Yuan et al.
- [Schedule-Free](https://arxiv.org/abs/2407.85866) - Defazio et al.
- [Cautious Optimizers](https://arxiv.org/abs/2411.16085) - Liang et al.
- [FineWeb-Edu](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) - 你的dataset!

---

## 12. 我的额外联想 (hallucination-friendly zone)

1. **与你的work的联系**: FineWeb-Edu是你的dataset,paper用它在Stage 2做long-context实验。这dataset的quality对optimizer ranking很敏感吗?如果用原始FineWeb vs FineWeb-Edu,ranking会变吗?

2. **Muon的mystery**: Shumaylov et al. (ref [100]) 发现"Muon is not that special: random or inverted spectra work just as well"。这暗示NS的核心可能spectral structure本身,具体哪个basis不重要。这和paper的ablation一致(NS是核心),但open question是NS为什么work。

3. **Linear attention的instability**: GLA导致所有optimizer的GNormCV爆炸。这和linear attention的eigenvalue structure有关吗?DeltaNet用delta rule,可能gradient path更长。

4. **APOLLO的rank issue**: 长context下gradient的有效rank上升,这是information-theoretic的limit。Adaptive rank (随context动态调整)是一个明显的future direction。

5. **与nanoGPT的联系**: 你的nanoGPT用AdamW。基于这篇paper,如果要在small scale (60M-130M)实验,Muon可能值得试。但Stage 2显示architecture matters,标准Transformer上Muon应该OK。

6. **Optimizer as kernel**: 把optimizer update看作一个kernel operation,LMO是选择最陡方向的kernel。不同norm ball = 不同kernel geometry。

7. **与scaling laws的关系**: optimizer选择影响scaling law的prefactor但不改变exponent?还是某些optimizer能改变scaling law的shape?这个paper没直接回答。

8. **Composition search as NAS**: 把four-axis coordinate作为search space,用NAS找最优combination。这是一个automated optimizer design的方向。

---

这篇paper的价值在于它给optimizer community提供了一个**shared vocabulary**。以前每个人用自己的notation描述自己的optimizer,现在可以用 (pipeline stages, four-axis coordinates, effect objectives) 这个统一坐标系来比较和组合。对practitioner来说,tiered summary和selection guidance是直接的takeaway。对researcher来说,open problems (特别是adaptive compression和architecture-aware geometry) 是clear research agenda。
