---
source_pdf: LeWorldModel Stable End-to-End Joint-Embedding.pdf
paper_sha256: 36b6dabbf3a37a4229063e76d82854b9e696411abfa0d4bf72325bdd67d64d85
processed_at: '2026-08-05T14:35:45-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲LeWM

## 一句话说清楚

**之前的JEPA world model要么靠"作弊"（用别人训好的encoder），要么靠"炼丹"（调7个loss term的超参数），这篇paper用一个数学上很干净的小trick，把7个超参数砍成1个，还能从头到尾稳定训练。**

## 问题出在哪

想象你要训一个world model。给它看一帧画面，它把画面压缩成一个192维的向量 $z_t$。再给它当前action，它预测下一帧的向量 $\hat{z}_{t+1}$。如果预测对了，就给个糖。

问题来了：模型很懒。它发现一个捷径——**不管输入什么，都输出同一个常数向量**。那predictor也输出同一个常数，预测误差永远是0，完美交差。但这个representation啥用没有，你没法拿它做planning。

这叫**collapse**。

## 之前人们怎么治

三招，都有毛病：

**第一招：EMA + stop-gradient**（I-JEPA, V-JEPA那套）。搞两个encoder，一个是目标encoder，慢慢更新。但这个操作数学上说不清到底在优化什么，就是个trick。

**第二招：直接用DINOv2当encoder，冻住不训**（DINO-WM）。既然你不训encoder，它就不会collapse。但你就丧失了end-to-end的能力，encoder是什么样你就得用什么样。而且DINOv2有86M参数，planning巨慢。

**第三招：VICReg那套多loss**（PLDM）。加一堆regularizer：variance要够大、covariance要够小、时序上要平滑、再加个inverse dynamics……一共7个loss term，6个超参数。调参调到怀疑人生。每个环境都要重新调，而且grid search的复杂度是 $\mathcal{O}(n^6)$，就是六个维度一起搜。

## LeWM的招：SIGReg

核心想法特别优雅：**强制latent embedding服从一个标准高斯分布**。

如果embedding是标准高斯 $\mathcal{N}(0, I)$，那它就不可能是常数（常数是delta分布，不是高斯）。collapse自动被堵死了。

但怎么在高维（192维）空间检验一个分布是不是高斯？经典normality test都是一维的，高维会退化。

这里用了个数学trick，两步走：

### 第一步：Cramér-Wold定理

这个定理说：两个高维分布相等，当且仅当你在**任意方向**投影它们，得到的1D分布都相等。

想象你手里有个3D物体，你看不清它长啥样。但你拿手电筒从各个角度照它，如果每个角度的影子都一样，那这两个物体就是一样的。

所以你不用管192维，你只要在1024个随机方向上投影，看每个1D投影是不是高斯就行。

### 第二步：Epps-Pulley test

每个1D投影 $h$ 上，你用特征函数（就是概率分布的Fourier transform）算它跟标准正态的距离：

$$T = \int w(t) |\phi_N(t) - e^{-t^2/2}|^2 dt$$

直觉上：标准正态的特征函数是 $e^{-t^2/2}$，你拿数据的特征函数跟它比，差异越小越像高斯。

最后把1024个方向的 $T$ 取平均，就是SIGReg loss。

### 为什么这个loss这么干净

- **数学上可证明**：SIGReg → 0 当且仅当 embedding分布真的收敛到 $\mathcal{N}(0, I)$
- **只有一个超参数** $\lambda$（SIGReg的权重），而且ablation显示 $\lambda \in [0.01, 0.2]$ 都work，可以用binary search调
- **不用EMA**，不用stop-gradient，encoder和predictor一起训，梯度全通

总loss就两项：

$$\mathcal{L} = \underbrace{\|\hat{z}_{t+1} - z_{t+1}\|^2}_{\text{预测下一帧}} + \lambda \cdot \underbrace{SIGReg(Z)}_{\text{别collapse}}$$

## 架构上的小细节

**Encoder**：ViT-Tiny，5M参数。把224×224的图变成192维向量。注意最后一层后面要接个BatchNorm，因为ViT自带的LayerNorm会把方差normalize掉，SIGReg就没法干活了。

**Predictor**：ViT-Small，10M参数。action怎么进来？用AdaLN（DiT那个trick），每层都把action信息揉进去。AdaLN的参数初始化为0，让action的影响从0慢慢涨上来，训练初期不会乱。

总共15M参数，单GPU几小时训完。

## 训练完怎么用做planning

给你一个起始画面 $o_1$ 和一个目标画面 $o_g$：
1. 都encode成向量 $z_1$ 和 $z_g$
2. 随机生成一堆action sequence，在latent空间rollout到horizon H
3. 看最后那个 $\hat{z}_H$ 离 $z_g$ 有多远，这就是cost
4. 用CEM（采样→挑好的→更新采样分布）优化action sequence
5. 执行前几步，然后replan（MPC）

因为latent只有192维，而DINO-WM要处理几千个patch token，所以planning快48倍，1秒内搞定。

## 实验结果讲讲人话

**Push-T**（推方块那个任务）：LeWM 96%成功率，PLDM 78%，DINO-WM 92%。LeWM赢了，而且方差还更小。

**Two-Room**（最简单的2D导航）：LeWM反而输了。这个很反直觉，但作者解释得挺诚实——Two-Room的"本质维度"太低了（就agent的x,y坐标），但SIGReg强制192维空间都是高斯，这就有点"杀鸡用牛刀"，很多维度被浪费在拟合高斯上，实际有用的信息反而不够紧凑。

**OGBench-Cube**（3D机械臂）：DINO-WM略赢，尤其在旋转相关的量上。因为DINOv2在124M张图上预训练过，对3D旋转这种geometric信息有强prior，LeWM从零训达不到。

**Probing**：拿latent去线性回归agent位置、方块角度这些物理量，LeWM普遍比PLDM好，跟DINOv2打平。说明latent确实编码了物理state，不只是视觉texture。

## 两个意外发现

### 1. 不用reconstruction loss，decoder也能重建出画面

虽然训练时压根没有reconstruction loss，但训完之后单独训个decoder，能从192维latent重建出场景。早期训练时decoder重建的是slow features（位置、形状），细节慢慢才出来。这印证了JEPA天然偏好编码变化慢的物理量。

### 2. Latent轨迹会自己变直

神经科学里有个发现：人脑会把复杂的时间序列"拉直"成接近线性的轨迹。这篇paper发现LeWM的latent轨迹在训练中**自动变直了**，没加任何显式的平滑约束。

测量方法是看相邻velocity向量的cosine similarity：

$$S = \text{mean} \frac{\langle z_{t+1}-z_t, \quad z_{t+2}-z_{t+1} \rangle}{\|z_{t+1}-z_t\| \cdot \|z_{t+2}-z_{t+1}\|}$$

越接近1说明轨迹越直。LeWM的straightness比PLDM还高，尽管PLDM专门有个loss term鼓励时序平滑。

作者的猜测：SIGReg只在每个time step独立约束分布，不管时序维度。所以encoder可以"选择"在时序上collapse成线性路径，这对planning反而有利——因为线性dynamics意味着你可以用闭式解外推，不用一步步rollout。

## 我觉得最重要的insight

1. **数学干净 > 工程hack**。一个有理论保证的regularizer胜过6个heuristic loss term。这不仅是省事，是让你能分析、能证明、能debug。

2. **超参数从6到1不是量变是质变**。$\mathcal{O}(n^6)$ 到 $\mathcal{O}(\log n)$ 意味着从"不敢跑ablation"到"单卡随便扫"。这democratize了world model research。

3. **Emergent temporal straightening可能是个大deal**。如果JEPA天然倾向于linear dynamics，那latent space的planning可能比我们想的简单得多。这跟Hénaff的neuroscience发现、跟用straightness检测AI视频都是连通的线索。

4. **Low intrinsic dimensionality是SIGReg的 Achilles' heel**。强制高维isotropic Gaussian对简单任务不友好。这暗示了可能需要mixture of Gaussians或者learnable prior，但那就违背了极简哲学。

## 可能的后续方向

- 把SIGReg用到V-JEPA 2上，替掉EMA+stop-gradient，看能不能让self-supervised video learning也有数学保证
- 结合CausalJEPA（同一作者后续工作），去掉对action label的依赖
- 利用emergent linear dynamics做hierarchical planning——既然latent轨迹是直的，能不能直接做closed-form planning而不是CEM sampling
- 把这个方法往更大规模数据推，看SIGReg在natural video上是否还work

---

**一句话总结**：这篇paper证明了，你不需要7个loss term、不需要EMA、不需要预训练encoder，只需要一个数学上干净的分布匹配loss，就能从pixel训出稳定的JEPA world model。代价是对特别简单的任务有点overkill，收益是超参数search从指数级降到对数级，而且latent会emergent地展现出一些neuroscience里观察到的性质。

---

# LeWorldModel: 一种极简但稳定的JEPA World Model

Andrej，这篇paper挺有意思的，它击中了JEPA训练中一个长期的痛点——**collapse**。让我从底层机制到intuition给你拆解一下。

## 1. 核心问题的重新定义

JEPA的collapse问题本质上是一个**自由度过剩**的问题。当你有一个encoder $enc_\theta: \mathcal{O} \to \mathbb{R}^d$ 和一个predictor $pred_\phi: \mathbb{R}^d \times \mathcal{A} \to \mathbb{R}^d$，prediction loss：

$$L_{pred} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = pred_\phi(z_t, a_t)$$

这里 $z_t = enc_\theta(o_t)$ 是latent embedding，$a_t$ 是action。这个loss有一个**平凡的trivial minimum**：让encoder输出常数 $z_t = c \in \mathbb{R}^d$，predictor输出常数 $\hat{z}_{t+1} = c$。Loss直接为0，但representation完全没用。

之前的解法分三类：
- **EMA + stop-gradient** (I-JEPA, V-JEPA [13,14]): 用一个slow-moving target encoder，但theoretically [17]这并不能对应任何well-defined objective的优化
- **Frozen pretrained encoder** (DINO-WM [18]): 用DINOv2 [41]，但丧失了end-to-end的表达能力
- **Multi-term regularization** (PLDM [22]): VICReg [23] + 7个loss term，6个超参数，$\mathcal{O}(n^6)$ search

LeWM的insight：**用一个有理论保证的分布匹配loss替换所有heuristics**。

## 2. SIGReg的数学深度解析

### 2.1 为什么高维normality test很难

直接在高维空间 $\mathbb{R}^d$ 检验 $Z \sim \mathcal{N}(0, I_d)$ 在 $d=192$ 时几乎是不可行的。经典normality test（Shapiro-Wilk, Anderson-Darling, Jarque-Bera）都是univariate的，高维扩展会退化。

### 2.2 Cramér-Wold定理的妙用

Cramér-Wold定理 [39]说：两个概率分布 $P, Q$ 在 $\mathbb{R}^d$ 上相等，当且仅当对所有方向 $u \in \mathbb{S}^{d-1}$，它们的1D投影分布 $P_u, Q_u$ 相等。换言之：

$$P = Q \iff \forall u \in \mathbb{S}^{d-1}: \langle Z, u \rangle \text{ 的分布相等}$$

所以把高维问题reduce成无穷多个1D问题。实践中用 $M$ 个随机方向近似，$M=1024$ 在paper里。

### 2.3 Epps-Pulley test statistic

对每个投影 $h^{(m)} = Z u^{(m)} \in \mathbb{R}^{NB}$（N是history length, B是batch size），计算：

$$T^{(m)} = \int_{-\infty}^{\infty} w(t) |\phi_N(t; h^{(m)}) - \phi_0(t)|^2 dt$$

这里：
- $\phi_N(t; h) = \frac{1}{N} \sum_{n=1}^N e^{ith_n}$ 是经验特征函数，即 $h$ 分布的Fourier transform
- $\phi_0(t) = e^{-t^2/2}$ 是标准正态分布 $\mathcal{N}(0,1)$ 的特征函数
- $w(t) = e^{-t^2/(2\lambda^2)}$ 是权重函数，让积分收敛
- $T$ 越小，$h^{(m)}$ 越接近正态

最终SIGReg：

$$SIGReg(Z) = \frac{1}{M} \sum_{m=1}^M T^{(m)}$$

**intuition**：你在latent space的每个随机"视角"上检查embedding分布是不是高斯，如果所有视角都高斯，那joint就是isotropic Gaussian。这就像你用一个旋转的多面镜照一个3D物体，每个角度都正常就说明物体本身正常。

## 3. 架构细节与工程实现

### 3.1 Encoder

- **ViT-Tiny**: patch size 14, 12 layers, 3 heads, hidden dim 192, ~5M params
- 输入：224×224×3 RGB frame $o_t$
- 取最后一层 [CLS] token embedding
- **关键设计**：后面跟一个1-layer MLP with **BatchNorm** [35]，不是LayerNorm

为什么是BatchNorm而不是LayerNorm？因为ViT最后一层是LayerNorm，会normalize掉维度间的variance，使SIGReg无法有效优化。BatchNorm保留batch内统计，让SIGReg能"看见"分布。

### 3.2 Predictor

- **ViT-S**: 6 layers, 16 heads, ~10M params
- **Adaptive Layer Normalization (AdaLN)** [37] for action conditioning，每层都注入
- **AdaLN参数初始化为0**：这是DiT [37]的trick，让action conditioning的影响从0开始逐渐增长，稳定训练初期
- **10% dropout**：ablation Table 9显示p=0.1最优（96% SR），p=0时只有78%，p=0.5降到66.67%
- **Causal masking**：predictor只看过去的N个frame embedding（N=3 for PushT/Cube, N=1 for TwoRoom）

### 3.3 总参数量

ViT-T encoder (~5M) + ViT-S predictor (~10M) = **15M**。这比DINO-WM的DINOv2-base (~86M) 小一个数量级。

## 4. Loss的极简性对比

这是paper最震撼的对比。看PLDM的loss（Appendix C.2）：

$$\mathcal{L}_{PLDM} = \mathcal{L}_{pred} + \alpha\mathcal{L}_{var} + \beta\mathcal{L}_{cov} + \gamma\mathcal{L}_{time-sim} + \zeta\mathcal{L}_{time-var} + \nu\mathcal{L}_{time-cov} + \mu\mathcal{L}_{IDM}$$

7个term，6个超参数 $(\alpha, \beta, \gamma, \zeta, \nu, \mu)$。paper Table 2里调到 $\alpha=18, \beta=12, \gamma=0.2, \zeta=0.7, \nu=0, \mu=0$。

而LeWM：

$$\mathcal{L}_{LeWM} = \mathcal{L}_{pred} + \lambda \cdot SIGReg(Z)$$

只有一个 $\lambda$，且ablation Fig. 16显示 $\lambda \in [0.01, 0.2]$ 都work（SR > 80%），peak在 $\lambda=0.09$。可以用 **bisection search $\mathcal{O}(\log n)$** 调，而PLDM需要 $\mathcal{O}(n^6)$ grid search。

**Karpathy你会喜欢这个intuition**：6维到1维的降维不是简单的"少一个参数"，而是从指数级的search space降到了logarithmic。这就像从穷举2^N个状态变成了binary search。

## 5. Planning: MPC + CEM

### 5.1 Cost function

$$\mathcal{C}(\hat{z}_H) = \|\hat{z}_H - z_g\|_2^2$$

其中 $\hat{z}_H = pred_\phi(\ldots pred_\phi(z_1, a_1) \ldots, a_{H-1})$ 是rollout，$z_g = enc_\theta(o_g)$ 是goal embedding。

### 5.2 CEM求解

Algorithm 2的pseudocode：
- 每次采样 $N=300$ 个候选action sequence $\{a_{1:H}^{(i)}\} \sim \mathcal{N}(\mu, \Sigma)$
- rollout每个，算cost
- 取top $K=30$ elite
- 更新 $\mu, \Sigma$
- 30 iterations

### 5.3 为什么快

DINO-WM用DINOv2的patch tokens（~200×更多tokens），而LeWM只用一个192维的[CLS] embedding。Planning时间快48×（Fig. 3左），<1秒完成。

## 6. 实验结果深度解读

### 6.1 Push-T (Table 5)

| Model | SR (%) |
|-------|--------|
| DINO-WM | 92.0 ± 1.63 |
| PLDM | 78.0 ± 5.0 |
| **LeWM** | **96.0 ± 2.83** |

注意LeWM的variance比PLDM还低。这说明SIGReg不仅是anti-collapse，还提供了训练稳定性。

### 6.2 Two-Room的反常结果

这是paper最诚实的部分。在Two-Room（最简单的环境）上，LeWM反而不如PLDM和DINO-WM。作者的解释很intuitive-给：

> "the low diversity and low intrinsic dimensionality of this dataset make it difficult for the encoder to match the isotropic Gaussian prior enforced by SIGReg in a high-dimensional latent space"

**intuition**: Two-Room的intrinsic dimensionality可能就2-3维（agent的x,y位置），但SIGReg强制192维空间是isotropic Gaussian。这就像你用192维高斯去fit一个2维流形，会有很多"浪费"的维度，导致representation不够structured。

### 6.3 Probing结果 (Table 1, Push-T)

LeWM在agent location的linear probe MSE = 0.052，比PLDM的0.090还好。Block angle的MLP probe r=0.990，远超PLDM的0.972。这说明LeWM的latent确实编码了物理state。

### 6.4 OGBench-Cube (Table 4)

注意这里DINO-WM在 **rotational quantities**（block quaternion r=0.411, end-effector yaw r=0.917）上明显比LeWM好。作者归因于DINOv2的124M image pretraining。这暗示了**foundation model的visual prior对3D旋转这种高维geometric量是关键的**。

## 7. 两个emergent phenomena

### 7.1 Slow features (Fig. 8)

训练初期decoded image对应slow features——这是Sobal et al. [21]在JEPA中观察到的现象。JEPA天然倾向于编码变化慢的量（位置、形状）而不是变化快的（纹理、噪声）。

### 7.2 Temporal straightening (Fig. 17)

这是我**最感兴趣的发现**。Hénaff et al. [42]在neuroscience里发现人脑会straighten temporal trajectories。这里测量：

$$S_{straight} = \frac{1}{B(T-2)} \sum_{i,t} \frac{\langle v_t^{(i)}, v_{t+1}^{(i)} \rangle}{\|v_t\| \|v_{t+1}\|}, \quad v_t = z_{t+1} - z_t$$

值越接近1，latent轨迹越接近直线。**LeWM没有explicit的temporal smoothness loss，但straightening比PLDM还高**——PLDM有个 $\mathcal{L}_{time-sim}$ 项专门鼓励这个。

作者的hypothesis：因为SIGReg只在每个time step独立施加，不约束temporal维度，encoder可以"选择"在temporal上collapse到线性路径，这反而有利于planning。

**这个发现可能比paper本身还重要**。它暗示了JEPA的representation learning有一个inductive bias toward linear dynamics，这和Hénaff的neuroscience发现、还有Internò et al. [54]用DINOv2 straightness检测AI视频都是connected的。

## 8. Violation-of-Expectation (Fig. 10)

VoE来自developmental psychology [43]，婴儿对"违背物理"的事件会注视更久。这里用surprise = prediction error做proxy。

结果：teleportation perturbation（物理违背）产生显著surprise spike（p<0.01），color perturbation（视觉违背）在OGBench-Cube上不显著。这暗示LeWM的latent space确实是**physics-aware**而不是pixel-similarity-aware。

## 9. 与V-JEPA 2 [14]和LeCun vision的关系

这篇paper直接承接LeCun [5]的JEPA vision。和V-JEPA 2 [14]的区别：
- V-JEPA 2: 自监督video预训练，没有action，用EMA+SG
- LeWM: action-conditioned，end-to-end，无EMA，无SG

**LeWM某种程度上是V-JEPA 2的"action-conditioned + stable"版本**。如果LeWM的方法（SIGReg）能scale到V-JEPA 2的数据规模，可能能替代EMA+SG那套heuristic。

## 10. Limitations & 我的思考

### 10.1 Low intrinsic dimensionality的failure mode

Two-Room的失败暴露了SIGReg的一个fundamental limitation：isotropic Gaussian prior假设了latent space的"信息密度"是均匀的。对于intrinsically low-dim的任务，这强制了信息冗余。

可能的fix：用 **mixture of Gaussians** 或者 learnable prior，类似VQ-VAE的codebook。但那会增加超参数，违背paper的极简philosophy。

### 10.2 短horizon planning

Paper的planning horizon H=5（with frame skip 5 = 25 env steps）。对于long-horizon任务（如Montecraft里的multi-step crafting），这远远不够。Author提到hierarchical world modeling是future work，但没具体说怎么做。

一个可能的direction：LeWM的latent空间已经emergent straightening，那么在latent space做hierarchical planning可能比pixel space自然得多——直线轨迹意味着可以**extrapolate**，可以做coarse-to-fine的plan refinement。

### 10.3 Action labels的依赖

LeWM需要action labels $a_t$。Author提到inverse dynamics modeling [20]是一个方向。CausalJEPA [20]（同一作者的另一篇）已经在做这个：通过object-level latent interventions学action representation。

## 11. 为什么这对world model research重要

之前JEPA world model的landscape是这样的：
- 想要稳定 → 用frozen pretrained encoder (DINO-WM)，但牺牲表达力
- 想要end-to-end → 用multi-term loss (PLDM)，但tuning hell
- 想要scale → I-JEPA/V-JEPA路线，但需要EMA+SG这些"黑魔法"

LeWM证明了：**有理论grounding的单个regularizer (SIGReg) 可以替代所有这些heuristics**。这是一个conceptual win。它把JEPA world model从"工程hack"拉回到"数学优化"。

更重要的是，它把**超参数search的复杂度从 $\mathcal{O}(n^6)$ 降到了 $\mathcal{O}(\log n)$**。这意味着researcher可以在单GPU上做ablation，不需要大规模hyperparameter sweep。这democratize了world model research。

## 12. 给你的几个可能联想

1. **V-JEPA 2 + SIGReg**：能否把V-JEPA 2的EMA+SG替换成SIGReg？如果可以，可能解开non-contrastive SSL的theoretical puzzle [17]
2. **LeWM + causal masking in latent**：CausalJEPA [20] + LeWM的组合，可能实现无action label的world model
3. **Temporal straightening + planning**：emergent linear dynamics能否被利用做closed-form planning？如果 $z_{t+1} \approx z_t + v$ 在latent里成立，planning可以变成一个QP问题
4. **SIGReg for diffusion models**：diffusion的forward process就是isotropic Gaussian，SIGReg可能作为diffusion的alternative regularization

## References

- Paper: [LeWorldModel arXiv (推测链接)](https://arxiv.org/abs/2602.11389) 
- SIGReg origin: [LeJEPA by Balestriero & LeCun 2025](https://arxiv.org/abs/2511.08544)
- JEPA vision: [LeCun 2022 paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- DINO-WM: [Zhou et al. ICML 2025](https://arxiv.org/abs/2411.04985)
- PLDM: [Sobal et al. 2025](https://openreview.net/forum?id=jON7H6A9UU)
- VICReg: [Bardes et al. ICLR 2022](https://openreview.net/forum?id=xm6YD62D1Ub)
- V-JEPA 2: [Assran et al. 2025](https://arxiv.org/abs/2506.09985)
- I-JEPA: [Assran et al. CVPR 2023](https://arxiv.org/abs/2301.08243)
- Cramér-Wold: [original 1936 paper](https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/jlms/s1-11.4.290)
- Epps-Pulley test: [Biometrika 1983](https://academic.oup.com/biomet/article-abstract/70/3/723/257914)
- Temporal straightening: [Hénaff et al. Nature Neuro 2019](https://www.nature.com/articles/s41593-019-0397-x)
- CausalJEPA: [Nam et al. 2026](https://arxiv.org/abs/2602.11389)
- DiT (AdaLN): [Peebles & Xie ICCV 2023](https://arxiv.org/abs/2212.09748)
- OGBench: [Park et al. ICLR 2025](https://openreview.net/forum?id=M992mjgKzI)
- stable-pretraining: [Maes et al. 2025](https://arxiv.org/abs/2511.19484)
- VoE in developmental psych: [Margoni et al. Psychological Review 2024](https://psycnet.apa.org/record/2024-44186-010)
- AI video detection via straightening: [Internò et al. NeurIPS 2025](https://openreview.net/forum?id=LsmUgStXby)

Andrej，这篇paper的beauty在于它把JEPA world model training从一个"七种loss term的炼丹术"还原成了一个"两个term的优化问题"。而且SIGReg的数学是closed-form可分析的——Cramér-Wold + Epps-Pulley的组合让你可以证明SIGReg → 0 iff $\mathbb{P}_Z \to \mathcal{N}(0, I)$。这是LeCun lab最近一系列"去heuristic化"工作的延续。 temporal straightening那个emergent property尤其值得深挖，它可能暗示了JEPA representation learning和neuroscience的perceptual straightening有共同的information-theoretic root。
