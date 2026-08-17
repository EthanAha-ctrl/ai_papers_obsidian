---
source_pdf: Residual-Oriented Multi-Layer Alignment for Spatially-Aware Vision-Language-Action
  Models.pdf
paper_sha256: 8b639f8aeac53b121b3a78ece5ec162ce8464e6cd8e20f96a3169219d2db457b
processed_at: '2026-08-11T22:56:09-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 ROCKET

## 一句话说清问题

VLA 模型看不懂 3D，你给它一张桌面照片，它不知道杯子离机器人多远。解决办法是找个 3D 高手（VGGT）当老师，让 VLA 的中间层表示去"模仿"老师的表示。但之前大家都只对齐一层，挑哪层全靠试。

## 为什么多层对齐会崩

想象 10 个工人一起推一块石头（= 浅层收到的总梯度）。如果每个工人有自己的"导航仪"（= independent projector），他们很快会发现各推各的最省力，于是 10 个人推 10 个方向，石头原地打转。paper 里 Fig. 7 实测了——独立 projector 的参数两两 cosine 几乎是 0，确实正交化了。

Table 8 最惨的数字：Goal suite 从 97.6 掉到 42.2。石头没推动。

## Shared Projector 的妙处

把 10 个工人的导航仪没收，换成一张共用地图。现在大家只能往同一个方向推，石头嗖嗖往前走。理论上是 Eq. 44 那个不等式——只要老师的 error signals 跨层正相关（典型情况），共用 projector 就把 cross-layer gradient 锁在 constructive interference 一侧。

实证：Shared-only 把 80.0 拉回 98.2，10k 步就到 82.5（baseline 才 70.0）。

## Matryoshka 的补刀

但共用地图有副作用——浅层太容易对齐（CKA 图显示浅层师生相似度本来就高），会把共用地图"绑架"成只服务浅层的形状，深层对不上了。

解决办法很朴素：浅层只许看地图的前 20%，深层逐步放开到 100%。浅层任务简单，给小铲子够用；深层要挖硬石头，得上大铲子。这就是 Matryoshka 嵌套激活。

代价是早期略慢（78.0 vs 82.5），但终点更高（98.5 vs 98.2）。

## 结果

- LIBERO 98.5%，打平 SOTA (SF)，但只用 4% 算力
- 跨 PI0.5、RoboTwin、LIBERO-Plus 都涨
- 1% 数据就能到 73.6%，data efficient
- layer selection 鲁棒，随便选都涨

## 我的直觉

这篇 paper 的核心 insight 其实一句话：**gradient interference 是 multi-layer alignment 的隐形杀手，shared projector 是结构性的解药**。独立 projector 让每层"自私地"找自己最优解，结果浅层叠加的梯度互相抵消。共用 projector 强制所有层"协商"，加上 Matryoshka 给浅层限容防止绑架，整体梯度方向就一致了。

更深的联想：这其实是把 multi-task learning 的 gradient conflict 问题，用 architecture design 而非 optimizer trick (如 PCGrad) 来解决。治本而非治标。代价是牺牲了 per-layer 表达力，但 Matryoshka 部分补回来。

还有一个没被 paper 点破的点：Matryoshka 的 prefix mask 等价于强制浅层用低秩 projector，低秩 = smooth mapping = 抑制高频，这恰好匹配浅层"不需要复杂变换"的归纳偏置。深层要拟合 fine-grained geometry，放开满秩。这是个免费的 spectral regularization。

至于为什么 8-16 层是甜区：层数太少浪费信号，太多则 alignment loss 压过 action loss，且 Eq. 30 的 transport error $\mathcal{E}_i$ 随层数累积。24 层掉到 94.1 就是这个原因。

最后一句：这套方法 50 行 PyTorch 就能实现，但背后的 residual dynamics 分析才是真东西。好 paper 的特征——方法简单，道理深刻。

---

# ROCKET: Residual-Oriented Multi-Layer Alignment for Spatially-Aware VLA 深度解析

Karpathy 你好，这篇 paper 我仔细读完了，下面我从 intuition 出发，把它的核心机制、理论推导、实验数据都拆开讲清楚，并尽量补上我自己的联想和延伸思考。

---

## 1. 问题背景：为什么 VLA 需要 3D spatial reasoning

VLA models (如 OpenVLA, PI0, PI0.5) 的 vision-language backbone 基本都在 2D 图像上预训练，缺乏稳定的 3D geometric understanding。在 LIBERO-Spatial 这种依赖 viewpoint changes、fine-grained spatial relations 的任务上，2D-pretrained VLA 会 generalize得很差。

给 VLA 注入 3D 知识的三条路线：
- **Explicit 3D inputs**: 直接喂 point cloud / depth map (SpatialVLA, GeoVLA, 3D-CAVLA)
- **Depth estimation**: 从 2D recover 3D (DPT, Depth Anything 系列)
- **Implicit representation alignment**: 用 frozen 3D foundation model (VGGT, $\pi^3$, Depth Anything 3) 作为 teacher，让 VLA student 的中间层 features 去匹配 teacher 的 features

第三条路线 inference 高效、易 scale，是 ROCKET 选的方向。已有的 SF (Spatial Forcing)、GLaD、REPA、3DRS 都属于这一类，但它们都只对齐 **single layer**。

Single-layer alignment 的两个痛点：
1. **Layer selection 敏感**：Table 1 显示 SF 选 OpenVLA 的第 24/32 层，REPA 选 DiT 的第 8/24 层，3DRS 选 LLaVA 的第 32/32 层——没有统一规律，全靠 post-hoc search
2. **浪费 cross-layer 信号**：residual network 的不同 depth 编码不同 level 的信息（Skean et al., 2025; Lee et al., 2025），shallow 偏 local cue，deep 偏 global semantics

直觉上 multi-layer alignment 应该更好，但作者发现 **naive multi-layer alignment (每个 layer 配一个独立 projector) 在 VLA 上反而崩了**——Table 8 显示加 multi-layer 后 Goal suite 从 97.6% 掉到 42.2%，平均从 96.4% 掉到 80.0%。这个反直觉现象是 ROCKET 要解决的核心问题。

---

## 2. 核心诊断：从 Residual Dynamics 看 Gradient Interference

### 2.1 把 Transformer 看作 dynamical system

Student backbone 写成 Pre-LN residual stream:

$$h_{l+1} = h_l + F_l(\mathrm{LN}(h_l); \theta_l), \quad l = 0, \ldots, L-1 \quad \text{(Eq. 17)}$$

变量含义：
- $h_l \in \mathbb{R}^d$: 第 $l$ 层的 residual representation (flattened)
- $F_l$: 第 $l$ 个 transformer block 的"残差增量"函数 (含 attention + MLP)
- $\theta_l$: 第 $l$ 层参数
- $\Delta_l := F_l(\mathrm{LN}(h_l); \theta_l)$: 残差增量

这种 ODE-like 视角 (Weinan, 2017; Chang et al., 2017) 的关键洞察是：**Pre-LN 下，当 $\|\Delta_l\|$ 小时，反向传播近似 identity**。

形式化地，定义 Jacobian:
$$\frac{\partial h_{l+1}}{\partial h_l} = I + A_l, \quad A_l := \frac{\partial F_l}{\partial u_l} J_{\mathrm{LN}}(h_l) \quad \text{(Eq. 21)}$$

如果 $\|A_k\| \le \varepsilon$ (residual-smallness assumption, Eq. 28)，那么 Lemma 1 给出:

$$\left\| \prod_{k=i}^{l-1} (I + A_k)^\top - I \right\| \le \exp((l-i)\varepsilon) - 1 \approx (l-i)\varepsilon \quad \text{(Eq. 29)}$$

也就是说，从深层 $l$ 反传到浅层 $i$ 的 Jacobian 接近单位矩阵。

### 2.2 关键 Corollary：浅层梯度是未来所有对齐层梯度的叠加

对齐 loss 定义为:
$$\mathcal{L}_{\mathrm{align}}(\theta, \{\phi_l\}) := \sum_{l \in \mathcal{S}} \alpha_l \ell(z_l, t_l), \quad z_l := p_{\phi_l}(h_l(\theta)) \quad \text{(Eq. 20)}$$

变量：
- $\mathcal{S}$: 选中的 student layer 集合
- $\alpha_l$: 第 $l$ 层对齐权重 (默认 uniform)
- $z_l$: projector 输出
- $t_l$: teacher representation (来自 VGGT 对应层)
- $\ell$: similarity loss (默认 $1 - \cos$)

定义 local gradient:
$$g_l := \nabla_{z_l} \ell(z_l, t_l) \quad \text{(Eq. 24)}$$

那么 student 第 $l$ 层的梯度是 $J_{p_{\phi_l}}(h_l)^\top g_l$ (Eq. 25)，反传到浅层 $i$ 后叠加，得到 **Corollary 1**:

$$\boxed{\nabla_{h_i} \mathcal{L}_{\mathrm{align}} = \sum_{l \in \mathcal{S}: l \ge i} \alpha_l J_{p_{\phi_l}}(h_l)^\top g_l + \mathcal{E}_i} \quad \text{(Eq. 30)}$$

变量：
- $v_l := J_{p_{\phi_l}}(h_l)^\top g_l$: 经 projector 映射回 student hidden space 的 local distillation gradient
- $\mathcal{E}_i$: transport error，量级 $O(\varepsilon \sum_{l \ge i} (l-i) \alpha_l \|v_l\|)$

**Intuition**: 浅层 $h_i$ 收到的总梯度 ≈ 所有未来对齐层 $l \ge i$ 的 local gradient 直接相加。这意味着不同层的 alignment 信号会在浅层"叠加干涉"。

### 2.3 干涉项的展开：cross terms 决定 constructive vs destructive

令 $G \approx \sum_{l > i} \alpha_l v_l$，展开平方范数:

$$\|G\|_2^2 = \sum_l \alpha_l^2 \|v_l\|_2^2 + \sum_{a \neq b} \alpha_a \alpha_b \langle v_a, v_b \rangle \quad \text{(Eq. 32)}$$

第一项是各层自身贡献，第二项是 cross terms。**核心命题 (Proposition 1)**:

$$\boxed{\langle v_a, v_b \rangle = g_a^\top \underbrace{\left( J_{p_{\phi_a}}(h_a) J_{p_{\phi_b}}(h_b)^\top \right)}_{=: \mathcal{M}_{ab}} g_b} \quad \text{(Eq. 33)}$$

- $g_a, g_b$: teacher-side error signals (在 projector 输出空间)
- $\mathcal{M}_{ab}$: **interaction matrix**，决定 cross-layer coupling 的方向和强度

**关键观察**: 用 independent projectors 时，$\mathcal{M}_{ab}$ 在不同 $(a, b)$ 上没有结构耦合，既不必对称也不必 PSD，所以 $\langle v_a, v_b \rangle$ 可能符号不稳定、甚至 destructive。

### 2.4 实证：independent projectors 会"正交化"

Appendix E (Fig. 7) 把每个独立 projector 的 MLP 参数 flatten 后计算 pairwise cosine similarity，发现几乎全为 0。这意味着不同 projector 学到了**几乎正交的映射轨迹**。

Fig. 1 上半部分进一步画了不同 alignment loss 在浅层诱导的梯度的 pairwise cosine similarity，训练全程都很低。这说明 independent projectors 在用"互相正交化"来吸收 layer-specific 差异，结果就是 cross-layer coupling 被抑制，浅层叠加的梯度信号互相抵消。

---

## 3. ROCKET 的核心设计：Shared Projector + Matryoshka

### 3.1 Shared Projector 的理论保证

把所有层共用一个 projector $\phi_l \equiv \phi$ 后，interaction matrix 变成:

$$\mathcal{M}_{ab} = J_{p_\phi}(h_a) J_{p_\phi}(h_b)^\top \quad \text{(Eq. 10 上)}$$

虽然它本身仍不必对称/PSD，但 Proposition 2 证明它可以**分解为 PSD reference + controlled deviation**:

$$\langle v_a, v_b \rangle_{\mathrm{share}} = g_a^\top M g_b + \Delta_{ab}, \quad M := JJ^\top \succeq 0 \quad \text{(Eq. 42)}$$

变量：
- $J := J_{p_\phi}(\bar{h})$: 参考层 $\bar{l}$ 处 projector 的 Jacobian
- $M = JJ^\top$: 半正定 reference operator
- $\Delta_{ab}$: 偏差项，由 Jacobian 沿 residual stream 的 Lipschitz 变化控制

具体地，假设 projector Jacobian Lipschitz (Eq. 36): $\|J_{p_\phi}(x) - J_{p_\phi}(y)\| \le L_J \|x - y\|$，结合 bounded residual increments $\|\Delta_l\| \le \delta$ (Eq. 35)，Lemma 2 给出:

$$\|J_{p_\phi}(h_b) - J_{p_\phi}(h_a)\| \le L_J (b - a) \delta \quad \text{(Eq. 37)}$$

于是 $J_{p_\phi}(h_l) = J + E_l$, $\|E_l\| \le L_J |l - \bar{l}| \delta$。

再假设 $M$ 在 error-signal subspace $\mathcal{G} = \mathrm{span}\{g_l\}$ 上 near-isometric (Eq. 39):

$$(1 - \eta) c \|x\|^2 \le x^\top M x \le (1 + \eta) c \|x\|^2, \quad \forall x \in \mathcal{G}$$

变量：
- $c > 0$: isometry scale
- $\eta \in [0, 1)$: distortion 程度，越小越接近 isometry

由 polarization identity (Lemma 3) 得到:

$$\left| x^\top M y - c x^\top y \right| \le \eta c \cdot \frac{\|x\|^2 + \|y\|^2}{2} \quad \text{(Eq. 40)}$$

最终得到 **signal-aligned lower bound (Theorem H.1)**:

$$\boxed{\langle v_a, v_b \rangle_{\mathrm{share}} \ge c g_a^\top g_b - \eta c \cdot \frac{\|g_a\|^2 + \|g_b\|^2}{2} - |\Delta_{ab}|} \quad \text{(Eq. 44)}$$

**Intuition 解读**:
- 第一项 $c g_a^\top g_b$: 当 teacher-side error signals 跨层正相关时（典型情况，因为不同层都在追同一个 teacher 表示流），这一项是正的
- 第二项: near-isometry 的 distortion，量级由 $\eta$ 控制
- 第三项: Jacobian 沿深度变化的偏差，由 $L_J \delta$ 控制

只要这三项加起来还是正的，cross-layer gradients 就是 **constructive interference**，浅层收到的总梯度被强化而不是抵消。这是 shared projector 比 independent projectors 优越的理论核心。

Fig. 1 下半部分实证验证：shared projector 下，cross-layer gradient cosine similarity 全程保持高值。

### 3.2 Matryoshka-style Sparse Activation: 解决另一个不对称

Shared projector 解决了 gradient conflict，但引入新问题：**浅层更容易对齐，会主导 shared projector 的学习**。

Appendix F 用 CKA (Centered Kernel Alignment, Cortes et al., 2012) 测量 OpenVLA-7B 与 VGGT 各层的相似度 (Fig. 8)，发现 VGGT 浅层与 OpenVLA 相似度高，深层相似度低。这意味着浅层 alignment loss 更容易下降，shared projector 会被浅层"绑架"。

CKA 的公式（线性版本）:
$$\mathrm{CKA}(X, Y) = \frac{\|X^\top Y\|_F^2}{\|X^\top X\|_F \|Y^\top Y\|_F}$$

变量：
- $X \in \mathbb{R}^{n \times d_x}$: student 一层的表示矩阵，$n$ 个样本
- $Y \in \mathbb{R}^{n \times d_y}$: teacher 一层的表示矩阵
- $d_x, d_y$ 可以不同，只要 $n$ (样本数) 相同
- CKA 度量两个表示空间如何"组织"同一组样本，对 isotropic scaling 和 orthogonal 变换不变

ROCKET 的解决方案是 **Matryoshka 嵌套激活** (Kusupati et al., 2022)。Projector 参数化为两层 MLP:

$$p_\Phi(x) = W_2 \sigma(W_1 x), \quad W_1 \in \mathbb{R}^{m \times d_S}, W_2 \in \mathbb{R}^{d_T \times m} \quad \text{(Eq. 12)}$$

变量：
- $d_S$: student hidden dim (OpenVLA 是 4096)
- $d_T$: teacher hidden dim (VGGT 是某个值)
- $m$: projector 最大内部宽度
- $\sigma$: GELU

对第 $i$ 个对齐层，只激活前 $m_i$ 个 hidden channels，宽度 schedule 线性递增:

$$\rho_i = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot \frac{i - 1}{\max(N - 1, 1)}, \quad m_i = \lceil \rho_i m \rceil \quad \text{(Eq. 13)}$$

变量：
- $\rho_i \in [0.2, 1.0]$: 第 $i$ 层激活比例
- $\rho_{\min} = 0.2$, $\rho_{\max} = 1.0$: 默认配置
- $N$: 对齐层数 (默认 10)

用 binary mask 实现:
$$g_i[j] = \mathbb{I}(j \le m_i), \quad p_\Phi^{(i)}(x) = W_2 (g_i \odot \sigma(W_1 x)) \quad \text{(Eq. 14, 15)}$$

- $g_i \in \{0, 1\}^m$: prefix mask
- $\odot$: element-wise 乘法
- 浅层用 prefix 子矩阵 $W_1[:m_i, :]$, $W_2[:, :m_i]$
- 深层用完整 $W_1, W_2$

**Intuition**: Matryoshka 让浅层"共享" projector 的一小部分参数（容量小，但浅层本来就好对齐），深层用更多参数（容量大，去拟合更难的 alignment）。这形成嵌套的 projector family $\{p_\Phi^{(i)}\}$，参数共享但有效容量随深度递增。

这与 Matryoshka Representation Learning (Kusupati et al., 2022) 的思想同源：原始 MRL 是让一个表示的前 $k$ 维就能"近似"全表示的语义，这里反过来用——让 projector 的前 $k$ 个 hidden units 就能处理简单（浅层）对齐。

### 3.3 最终训练目标

$$\boxed{\mathcal{L}_{\mathtt{ROCKET}} = \mathcal{L}_{\mathrm{action}} + \lambda \frac{1}{N} \sum_{i=1}^N \ell\left(p_\Phi^{(i)}(h_{s_i}^S), h_{\tau_i}^T\right)} \quad \text{(Eq. 16)}$$

- $\mathcal{L}_{\mathrm{action}}$: action prediction loss (Eq. 2，对 ground-truth action 序列)
- $\lambda = 0.5$: 默认 alignment 权重
- $\ell(a, b) = 1 - \cos(a, b)$: cosine similarity loss
- $N = 10$: 对齐层对数
- $(s_i, \tau_i)$: student-teacher 层对

---

## 4. 实验数据深度解析

### 4.1 LIBERO 主结果 (Table 2)

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| OpenVLA (baseline) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| SpatialVLA (explicit 3D) | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| GeoVLA (explicit 3D) | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA (explicit 3D) | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| GLaD (single-layer align) | 95.0 | 97.4 | 94.4 | 89.4 | 94.1 |
| Spatial Forcing (single-layer) | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| **ROCKET** | 98.2 | 99.8 | 98.8 | 97.0 | **98.5** |

ROCKET 与 SF 持平 (98.5%)，但 Table 7 显示 ROCKET 只用 1×32×50k = 1.6M compute，SF 用 4×64×150k = 38.4M，**ROCKET 只用约 4% compute**。

注意 ROCKET 在 Long-horizon suite 上比 SF 高 1.0 个点 (97.0 vs 96.0)，在 Spatial 上略低 (98.2 vs 99.4)。我猜测原因是 multi-layer alignment 对 long-horizon 的 multi-stage reasoning 更有帮助（不同 stage 可能依赖不同 depth 的表示），而 Spatial suite 已经被 SF 的 single-layer 深层对齐解决得很好。

### 4.2 训练动态 (Table 9, Fig. 3)

| Method | 10k | 20k | 30k | 50k |
|---|---|---|---|---|
| Baseline | 70.0 | 96.2 | 96.4 | 95.4 |
| Spatial Forcing | 66.9 | 94.8 | 96.4 | 95.9 |
| Multi-layer (independent) | 59.0 | 78.0 | 78.5 | 80.0 |
| ROCKET (Shared-only) | 82.5 | 96.4 | 96.9 | 98.2 |
| **ROCKET** | 78.0 | 96.5 | 96.8 | **98.5** |

几个关键观察：
1. **Naive multi-layer 在 10k 就崩了** (59.0)，且训练全程没恢复 (50k 仍 80.0)，证明 gradient interference 是持续性伤害，不是初期噪声
2. **Shared-only 在 10k 就达到 82.5**，比 baseline 高 12.5 个点，说明 shared projector 让 multi-layer alignment 立即生效
3. **加 Matryoshka 后 10k 略低 (78.0 vs 82.5)**，但 50k 更高 (98.5 vs 98.2)。Matryoshka 牺牲一点早期速度换更稳定的 late-stage 收敛——这符合"浅层容量小、深层逐步释放"的设计意图

### 4.3 Ablation (Table 8)

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| Baseline | 96.9 | 98.7 | 97.6 | 94.8 | 96.4 |
| + Multi-layer (independent) | 93.6 | 99.2 | 42.2 | 85.0 | 80.0 |
| + Shared | 99.0 | 99.8 | 97.0 | 96.8 | 98.2 |
| + Matryoshka | 98.2 | 99.8 | 98.8 | 97.0 | 98.5 |

- Multi-layer independent 把 Goal 从 97.6 砸到 42.2——这是 gradient interference 最戏剧化的证据
- Shared projector 救回 18.2 个点 (80.0 -> 98.2)
- Matryoshka 再加 0.3 个点，主要在 Goal (+1.8) 和 Long (+0.2)

### 4.4 Layer 数量选择 (Table 4)

| # Layers | Avg. |
|---|---|
| 4 | 96.3 |
| 8 | 97.4 |
| 16 | 96.7 |
| 24 | 94.1 |

8-16 层是甜区，24 层反而下降。我推测原因是层数太多时 $\sum_l \alpha_l$ 增大，alignment loss 相对 action loss 过强，且 Eq. 30 中的 transport error $\mathcal{E}_i$ 累积变大。

### 4.5 Layer Selection 策略 (Table 5)

| Strategy | Avg. |
|---|---|
| Baseline | 95.0 |
| Uniform-8 | 97.4 |
| E2M-Last1 | 96.3 |
| Shallow | 94.7 |
| Middle | 93.7 |
| Deep | 95.3 |
| Sim-1-Top | 93.6 |
| Sim-1-Last | 95.6 |
| Sim-2-Top | 95.4 |
| Sim-2-Last | 95.1 |

**重要发现**: 几乎所有合理策略都比 baseline 好，说明 ROCKET 对 layer selection 鲁棒。Uniform-8 最好 (97.4)，但默认用 E2M-Last1 (96.3)——作者选 E2M-Last1 可能是因为它在 PI0/PI0.5 (18 层) 上更通用 (Table 10 注脚)。

Sim-based 启发式 (基于 input-output cosine similarity，Gromov et al., 2024; He et al., 2024) 没有显著优势，说明"层重要性"在 VLA + VGGT 这种 cross-model 场景下没那么好用。

### 4.6 LIBERO-Plus 鲁棒性 (Fig. 5)

ROCKET 平均 81.7%，baseline 80.0%。在 **Robot 和 Layout 扰动**上提升最明显——这两类扰动直接改变 spatial geometry 而非 appearance，说明 ROCKET 真的注入了 spatial reasoning，没有走 positional shortcut。

### 4.7 RoboTwin 2.0 (Fig. 6)

Bimanual 任务上，ROCKET 在 Easy setting 明显领先，Hard setting 略低于 fine-tuned baseline。我猜测 Hard 任务的多阶段、多接触特性让 single-view 2D 表示的瓶颈变得更突出，representation alignment 的边际收益递减。

### 4.8 PI0.5 (Table 3)

| Method | Avg. |
|---|---|
| Baseline | 93.0 |
| Spatial Forcing | 94.0 |
| ROCKET | 95.3 |

ROCKET 在 PI0.5 (full fine-tuning, 18 层 backbone) 上比 SF 高 1.3 个点，说明 shared projector + Matryoshka 对小模型/浅模型也有效，不局限于 OpenVLA-7B。

### 4.9 Data Efficiency (Table 6)

| Data Ratio | Avg. |
|---|---|
| 1% | 73.6 |
| 5% | 83.4 |
| 10% | 84.3 |
| 100% | 96.3 |

1% 数据就能到 73.6%，这对 real-robot data 稀缺场景意义重大。我推测 multi-layer alignment 提供了 dense 的 auxiliary supervision，相当于隐式 data augmentation。

---

## 5. 我的 Intuition 与延伸联想

### 5.1 Residual stream 视角的威力

把 multi-layer alignment 看作"对齐一个 residual stream 到另一个 residual stream"是个很漂亮的 reformulation。它把"选哪一层"问题转化为"两个 dynamical trajectory 之间的 mapping"问题。

这让我想到 Neural ODE (Chen et al., 2018) 和 flow matching (Lipman et al., 2023) 的连续深度视角。如果 student 和 teacher 都是 residual ODE $\dot{h} = F(h, t)$，那 alignment 就是在两个 vector field 之间建立对应。ROCKET 的 shared projector 类似于学一个"积分变换"把 student trajectory 推到 teacher trajectory 上。

进一步推：如果对齐所有层（连续极限），shared projector 实际上是在学一个 **conjugacy**——两个 dynamical system 之间的拓扑等价映射。这和 Koopman operator theory 有联系。

### 5.2 Pre-LN vs Post-LN 的隐含假设

理论推导依赖 Pre-LN 的 residual-smallness (Eq. 28)。Post-LN Transformer (如原始 BERT) 的反向传播不接近 identity，因为 LayerNorm 在残差外，信号会被 normalize 掉。这意味着 ROCKET 的理论保证**不直接适用 Post-LN 模型**。

好消息是现代大模型 (Llama, Gemma, PaliGemma) 几乎都是 Pre-LN，所以实际影响不大。但如果有人想在 BERT-style 模型上用 ROCKET，需要重新分析。

### 5.3 Cone Effect 与 Layer-wise Learning Rate Decay 的关系

Paper 提到 hidden states 在同一模型内趋向同一 feature space ("cone effect", Gao et al., 2019; Fig. 4)。这解释了为什么 shared projector 能 work——不同层的 $h_l$ 已经在同一个 cone 里，一个线性+非线性映射就够。

这也联系到 layer-wise learning rate decay (LLRD) 的经验：浅层用更小的 LR，因为它们的更新会影响所有深层。ROCKET 的 Matryoshka 反过来——浅层用更小的 projector 容量。两者方向相反但目标相似：**保护浅层不被过度扰动**。

### 5.4 与 REPA 的对比

REPA (Yu et al., 2024) 在 diffusion transformer (DiT) 上对齐第 8/24 层到 DINOv2 第 24/24 层。它选 shallow student layer 是因为 diffusion model 早期需要建立 visual structure。ROCKET 的 multi-layer 同时覆盖 shallow 和 deep，更通用。

我猜测 REPA 也能从 ROCKET 的 shared projector + Matryoshka 受益，特别是 DiT-XL 这种深模型。值得做的实验：把 REPA 的 single-layer alignment 换成 ROCKET-style multi-layer，看 FID 改善多少。

### 5.5 Matryoshka 的另一种解释

Matryoshka sparse activation 还有一个没被 paper 明说的好处：**它隐式地做了 spectral regularization**。浅层只用 $W_1$ 的前 $m_i$ 行，相当于强制 projector 在浅层只用低秩子空间。低秩 = 高频抑制 = smooth mapping，这恰好匹配浅层"容易对齐、不需要复杂变换"的特性。

深层用满秩，能拟合 teacher 深层的 fine-grained geometric detail。这种"浅层 smooth、深层 sharp"的归纳偏置与神经科学里 visual cortex 的层级处理 (V1 简单特征 -> IT 复杂对象) 有趣地平行。

### 5.6 为什么 Independent Projectors 会正交化

Appendix E 的现象（独立 projector 参数 cosine similarity 趋向 0）可以用 **lottery ticket + 特征竞争**解释。每个 projector 要把 student 第 $l$ 层的 $h_l$ 推到 teacher 第 $\tau_l$ 层的 $t_l$。不同层的 $(h_l, t_l)$ 对的"难度方向"不同，独立优化会让每个 projector 找到对自己有利的子空间。由于梯度叠加在浅层，这些子空间会互相"避让"——正交化是最小化 interference 的解。

Shared projector 强制所有层共用一个子空间，逼它们"协商"出一个共同方向。这类似于 multi-task learning 中的 shared trunk vs per-task head 的 trade-off，但 ROCKET 给出了理论解释。

### 5.7 与 PCGrad / Gradient Surgery 的关系

Multi-task learning 中 gradient conflict 的经典解法是 PCGrad (Yu et al., 2020)、CAGrad (Liu et al., 2021) 等——投影掉冲突分量。ROCKET 不做投影，而是通过 architectural design (shared projector) 让 conflict 在结构上消失。

这是更"根治"的方案：PCGrad 是 symptom-level 的 patch，shared projector 是 root-cause-level 的 redesign。代价是 shared projector 限制了 per-layer 的表达力，但 Matryoshka 部分恢复了它。

### 5.8 对 VLA Scaling 的启示

Table 7 的 compute 对比很震撼：
- OpenVLA-OFT: 25.6× (4×64×150k)
- Spatial Forcing: 24.0× (4×64×150k)
- ROCKET: 1.0× (1×32×50k)

ROCKET 用 4% compute 达到 SOTA。这意味着 representation alignment 是 VLA scaling 的"捷径"——不需要更大 model 或更多 data，只需要更好的 auxiliary supervision。

进一步推：如果 teacher 用更大的 3D foundation model (如 VGGT-XL 或未来的 $\pi^3$-XXL)，ROCKET 的收益可能更大。Teacher quality 是 ROCKET 性能的上限。

### 5.9 局限与未解问题

1. **Teacher 必须是 frozen 3D model**：如果 teacher 本身不够强（比如在 occlusion 重的场景），ROCKET 的天花板就低
2. **Layer selection 仍需经验**：虽然 Table 5 显示鲁棒，但 Uniform-8 (97.4) 比 Sim-1-Top (93.6) 差 3.8 个点，说明策略仍有影响
3. **理论假设较强**：residual-smallness、Jacobian Lipschitz、near-isometry 三个假设在训练初期可能不成立，这解释了为什么 Matryoshka 早期略慢
4. **只在 manipulation 上验证**：navigation、locomotion 等其他 embodied task 上的效果未知

### 5.10 一个可能的扩展：Cross-Model Multi-Teacher

ROCKET 用单一 teacher (VGGT)。如果同时用 VGGT + Depth Anything 3 + $\pi^3$ 作为多 teacher，每个 teacher 对齐不同层子集，会怎样？这相当于 multi-stream alignment，理论分析需要扩展到 multi-teacher interaction matrix。我猜测需要 per-teacher shared projector + cross-teacher orthogonality constraint。

### 5.11 与你的 work 的联系

Karpathy，你在 "micrograd" 和 "makemore" 里强调 build intuition through minimal implementation。ROCKET 的核心实现其实非常简单——一个 shared MLP projector + prefix mask，可能 50 行 PyTorch 就够。但背后的 residual dynamics + gradient coherence 分析才是真正的贡献。这是典型的"简单方法、深刻理论"的好 paper。

你之前在 Tesla 搞 real-world vision 时也遇到过 2D 表示缺乏 3D 理解的问题。ROCKET 这种 implicit alignment 比 explicit depth 输入更轻量，适合 on-robot deployment。如果把 VGGT 换成你那边的 multi-camera 3D reconstruction 模型作为 teacher，理论上可以直接迁移。

---

## 6. 总结：ROCKET 的三步逻辑

1. **诊断**: Naive multi-layer alignment 在 VLA 上崩了，因为 independent projectors 学到正交化映射，浅层叠加梯度互相抵消 (Eq. 30, 33, Fig. 1 上)
2. **治疗**: Shared projector 把 interaction matrix 分解为 PSD reference + controlled deviation，保证 cross-layer gradients constructive (Eq. 42, 44, Fig. 1 下)
3. **精修**: Matryoshka sparse activation 让浅层用小容量（避免主导）、深层用大容量（拟合难对齐），平衡 alignment losses (Eq. 13, 15, Fig. 8)

结果：4% compute 达到 SOTA，跨 LIBERO / LIBERO-Plus / RoboTwin / PI0.5 一致有效。

---

## References (web links)

- ROCKET code: https://github.com/CASE-Lab-UMD/ROCKET-VLA
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- Spatial Forcing: https://arxiv.org/abs/2510.12276
- VGGT: https://arxiv.org/abs/2503.17351 (CVPR 2025)
- $\pi^3$: https://arxiv.org/abs/2507.13347
- Depth Anything 3: https://arxiv.org/abs/2511.10647
- REPA: https://arxiv.org/abs/2410.06940
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
- Patient KD: https://arxiv.org/abs/1908.09355
- ReviewKD: https://arxiv.org/abs/2104.08697 (CVPR 2021)
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- PI0: https://arxiv.org/abs/2410.24164
- PI0.5: https://arxiv.org/abs/2504.16054 (CoRL 2025)
- DINOv2: https://arxiv.org/abs/2304.07193
- CKA (Cortes et al., 2012): https://dl.acm.org/doi/10.5555/2188385.2188469
- Gromov "unreasonable ineffectiveness of deeper layers": https://arxiv.org/abs/2403.17887
- Cone effect (Gao et al., 2019): https://arxiv.org/abs/1907.12009
- Residual dynamics (Weinan, 2017): https://global-sci.org/article/43586
- Skean et al., "Layer by layer": https://arxiv.org/abs/2502.02013
- GLaD: https://arxiv.org/abs/2512.09619
- 3DRS: https://arxiv.org/abs/2506.01946
- PCGrad (Yu et al., 2020): https://arxiv.org/abs/2001.06782
- CAGrad (Liu et al., 2021): https://arxiv.org/abs/2110.14085
- Neural ODE (Chen et al., 2018): https://arxiv.org/abs/1806.07366
- Flow Matching (Lipman et al., 2023): https://arxiv.org/abs/2210.02747
- SpatialVLA: https://arxiv.org/abs/2501.15830
- GeoVLA: https://arxiv.org/abs/2508.09071
- 3D-CAVLA: https://arxiv.org/abs/2505.05800
- GR00T N1.6: https://arxiv.org/abs/2503.14734
- Octo: https://arxiv.org/abs/2405.12213
- WorldVLA: https://arxiv.org/abs/2506.21539
- Dita: https://arxiv.org/abs/2503.19757
- InternVLA-M1: https://arxiv.org/abs/2510.13778
- VOTE: https://arxiv.org/abs/2507.05116
- TraceVLA: https://arxiv.org/abs/2412.10345
- ThinkAct: https://arxiv.org/abs/2507.16815
- UniVLA: https://arxiv.org/abs/2505.06111
- MemoryVLA: https://arxiv.org/abs/2508.19236
- SigLIP: https://arxiv.org/abs/2303.15343
- PaliGemma: https://arxiv.org/abs/2407.07726
- Llama 2: https://arxiv.org/abs/2307.09288
- LoRA: https://arxiv.org/abs/2106.09685
- FiLM: https://arxiv.org/abs/1709.07871
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RDT-1B: https://arxiv.org/abs/2410.07864
- RT-2: https://arxiv.org/abs/2307.15818
- VoxPoser: https://arxiv.org/abs/2303.07826
- CLIPort: https://arxiv.org/abs/2109.12098
- Perceiver-Actor: https://arxiv.org/abs/2209.05451
- DPT (Ranftl et al., 2021): https://arxiv.org/abs/2103.13413
- DiT (Peebles & Xie, 2022): https://arxiv.org/abs/2212.09748
- He et al., "What matters in transformers": https://arxiv.org/abs/2406.15786
- Gong et al., "Beyond logits": https://aclanthology.org/2025.acl-long.1125/
- Miles & Mikolajczyk, "Role of projector in KD": https://ojs.aaai.org/index.php/AAAI/article/view/30130
- VKD (Miles et al., 2024): https://arxiv.org/abs/2405.05753
- Projector ensemble (Chen et al., 2022): https://arxiv.org/abs/2207.11591
- Chang et al., "Multi-level residual networks": https://arxiv.org/abs/1710.10348
- ALOHA 2: https://arxiv.org/abs/2405.02292
- GPT-4o system card: https://arxiv.org/abs/2410.21276
- Reconstructive Visual Instruction Tuning: https://arxiv.org/abs/2410.09575

希望这个深度解析对你 build intuition 有帮助。如果你想让我再展开某一块（比如 Theorem H.1 的完整证明细节、Matryoshka 的 spectral 解释、或与 REPA 的具体对比实验设计），告诉我。
