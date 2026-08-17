---
source_pdf: Temporal Straightening for Latent Planning.pdf
paper_sha256: 4e18ebe0f34ad8d18ecb7cbc24d18ece559fd485fd69845bd44a21a9e42b6f6a
processed_at: '2026-08-12T13:26:13-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Temporal Straightening

## 一句话版本

你用 DINOv2 这种 pretrained encoder 把 image 编码成 vector，然后想在 latent space 做 planning — 但 trajectory 是弯的，直线距离骗你，gradient 也指错方向。这篇 paper 加个 regularizer 把 trajectory 拉直，planning 就好做了。

## 问题是什么

假设你要做 latent planning。流程大概是:

```
image o_t → encoder → z_t (latent vector)
                    ↓
              predictor f(z_t, a_t) → ẑ_{t+1}
                    ↓
              rollout K 步 → ẑ_K
                    ↓
              loss = ||ẑ_K - z_goal||² → backprop to actions a_0...a_{K-1}
```

你想用 gradient descent 调 action sequence，让 agent 走到 goal。这听起来挺自然的对吧？

**但 DINOv2 的 embedding space 几何性质不允许你这么做**。

### 直觉 1: 弯曲的 trajectory 让 Euclidean distance 说谎

想象你在迷宫里。A 点到 B 点的"真实距离"是沿着可行路径走的最短步数 (geodesic)。但 DINOv2 不知道你的迷宫拓扑 — 它只知道两张 image 视觉上像不像。

结果: 迷宫里两个视觉上像 (比如都在墙角) 但实际隔了十万八千里的 state，在 DINOv2 embedding 里挨得很近。你用 $\|z_A - z_B\|_2$ 衡量距离，得到一个骗人的小数字。Planner 以为"快到了"，其实根本没动。

Paper Figure 6 把这个可视化得很清楚。DINOv2 CLS embedding 的 distance heatmap 在迷宫里几乎是乱的，跟 A* 算的 ground-truth geodesic 完全不对应。但如果在 latent trajectory 上加 straightening regularizer，heatmap 突然就 align 上了 — 蓝色 (近) 沿可行路径铺开，红色 (远) 在不可达区域。Encoder 自己学到了"在迷宫拓扑上离 goal 多远"这件事。

### 直觉 2: 弯曲让 gradient 指错方向

Planning 的 loss 是 $\mathcal{L} = \|z_K - z_g\|_2^2$。你想对这个 loss 求 action 的 gradient，告诉 action 该往哪走。

如果 dynamics 是直的 ($z_{t+1} = z_t + B a_t$，即 action 直接等于 state 变化)，那 gradient 非常 clean — 你想让 state 往 $z_g - z_t$ 方向走，action 就该往那个方向。

但 DINOv2 embedding 里 trajectory 是弯的。同一个 action 在不同 state 上造成截然不同的 latent direction 变化。Gradient 有时指向 local minimum，有时根本是误导方向。Paper Figure 4 画了 action space 的 loss landscape — DINOv2 feature 上是高度 non-convex 的一团乱麻，straightened feature 上接近一个 clean bowl。

所以实践中大多数 latent planning 工作 (Dreamer, TD-MPC2, DINO-WM) 都不敢用 gradient descent 做 planning，而是用 CEM / MPPI 这种 sampling-based 方法。能用，但贵 — 每次规划要 sample 几百个 trajectory 来估 gradient。

## Insight: 人脑早就这么干了

Neuroscience 里有个叫 **perceptual straightening** 的 hypothesis (Hénaf et al., *Nature Neuroscience* 2019, https://www.nature.com/articles/s41593-019-0377-4)。

他们记录 macaque猴子 V1 cortex 的 neural population response，给猴子看 natural video。发现: **pixel 空间里 trajectory 是高度弯曲的 (相机一动，pixel 巨变)**，**但 V1 的 neural population response 把 trajectory 拉直了** (相邻时间点 neural state 方向一致)。

为什么？因为直的 trajectory 容易 predict。V1 在为后续 cortex 的 predictive coding 服务 — 如果你不能 predict 下一刻会看到什么，你没法 survive。Evolution 学到了让 representation 在时间上平滑，便于 forward prediction。

这篇 paper 的 insight: 我们 train world model 也想做 prediction，那 encoder 应该有同样的 geometric pressure。但 DINOv2 的训练 objective 是 self-distillation + patch-level contrastive，根本没有 temporal dynamics 信号 — 它会拉直 image 上的 augmentation perturbation，但不会拉直 video trajectory。所以我们需要 explicit regularizer 把它拉直。

LLM 里也有类似现象 (Hosseini & Fedorenko 2023, https://www.biorxiv.org/content/10.1101/2023.06.23.546237): next-word prediction 训练让 sentence trajectory 在 LM 的 representation space 里变直。Predictive learning 似乎 universally favor straight representation。

## 方法: 一个简单到令人发指的 Regularizer

定义三个连续的 latent points: $z_t$, $z_{t+1}$, $z_{t+2}$。计算两个 velocity vectors:

$$v_t = z_{t+1} - z_t, \quad v_{t+1} = z_{t+2} - z_{t+1}$$

这是 latent space 里的"瞬时速度" — 从 $t$ 到 $t+1$ 的 displacement。

然后算它们的 cosine similarity:

$$\mathcal{C} = \frac{v_t \cdot v_{t+1}}{\|v_t\|_2 \cdot \|v_{t+1}\|_2}$$

- $v_t \cdot v_{t+1}$: 内积，方向一致就正
- 分母是两个 velocity 的 L2 norm 乘积，做归一化
- $\mathcal{C} \in [-1, 1]$
  - $\mathcal{C} = 1$: 速度方向完全一致 — trajectory 在这里是直线
  - $\mathcal{C} = 0$: 速度方向垂直 — 90 度急转弯
  - $\mathcal{C} = -1$: 速度反向 — trajectory 折返

Loss 就是:

$$\mathcal{L}_{curv} = 1 - \mathcal{C}$$

最小化这个 loss = 最大化 cosine similarity = 强制连续两步的 velocity 方向一致 = 把 trajectory 拉直。

就这么简单。加到 JEPA 的 prediction loss 上:

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{curv}$$

$\lambda$ 是个系数，paper 里用 0.1 或 0.01。

**没有 negative samples，没有 contrastive pairs，没有 momentum encoder**，就一个 local geometric regularizer。完全是 JEPA philosophy — 用 prediction + 简单 regularization，avoid 复杂的 contrastive learning 工程。

## 为什么"直"在数学上好

这部分是 paper 最 technical 的贡献。考虑 linear dynamics 假设:

$$z_{t+1} = A z_t + B a_t$$

- $A \in \mathbb{R}^{d \times d}$: state 自己怎么 evolve (无 action 时)
- $B \in \mathbb{R}^{d \times d_a}$: action 怎么影响 state
- $d$ 是 latent dimension，$d_a$ 是 action dimension

定义 "$\epsilon$-straight":

$$\|A - I\|_2 \leq \epsilon$$

- $I$ 是单位矩阵
- $\|\cdot\|_2$ 是 spectral norm (最大奇异值)
- 这意思是 $A$ 接近 identity

当 $A = I$ (即 $\epsilon = 0$): dynamics 是 $z_{t+1} = z_t + B a_t$。这意味着**没有 action 时 state 不变** (agent 静止)，**有 action 时 state 沿 action 方向移动**。这是最直的 dynamics — action 是 trajectory 的直接 driver，没有 hidden dynamics 弯曲它。

Planning 是要 minimize $\|z_K - z_g\|^2$，其中 $z_K$ 是从 initial state $z_0$ 用 action sequence $\mathbf{a}$ unroll K 步的结果。Unroll 出来:

$$z_K = A^K z_0 + \sum_{t=0}^{K-1} A^{K-1-t} B a_t$$

这是 $\mathbf{a}$ 的 affine function，所以 loss 是 quadratic，Hessian 是:

$$H = 2 J_\Phi^\top J_\Phi$$

其中 $J_\Phi = [A^{K-1}B \mid A^{K-2}B \mid \cdots \mid B]$ 是 $z_K$ 对 actions 的 Jacobian。

Gradient descent 在 quadratic 上的 convergence rate 是 $(1 - 1/\kappa)^t$，其中 $\kappa$ 是 Hessian 的 condition number (最大和最小奇异值的比)。$\kappa$ 越大，GD 收敛越慢。

**Main theorem** (paper Theorem 4.4):

$$\kappa_{\text{eff}}(H) \leq \kappa(B)^2 \cdot \kappa(A)^{2(K-1)}$$

- $\kappa(B)$: control input matrix 的 condition number (系统属性，无法改)
- $\kappa(A)$: state transition matrix 的 condition number
- $K$: planning horizon

注意 $\kappa(A)^{2(K-1)}$ 这一项 — **horizon 越长，condition number 越爆炸**。如果 $\kappa(A) = 10$，horizon $K = 25$，那就是 $10^{48}$ 这种量级。GD 直接死掉。

但如果 dynamics 是 $\epsilon$-straight ($\|A - I\| \leq \epsilon$):

$$\kappa_{\text{eff}}(H) \leq \kappa(B)^2 \left(\frac{1+\epsilon}{1-\epsilon}\right)^{2(K-1)} \leq \kappa(B)^2 e^{6\epsilon K}$$

当 $\epsilon$ 小，$e^{6\epsilon K}$ 增长缓慢。比如 $\epsilon = 0.01$，$K = 25$，那只增长 $e^{1.5} \approx 4.5$ 倍，可控。

**直觉**: 把 trajectory 拉直 ⟺ 让 $A$ 接近 $I$ ⟺ Hessian condition number 不爆炸 ⟺ GD planning 收敛快且 stable。

这是 paper 最 elegant 的地方 — 把 representation geometry ($A$ 离 $I$ 多远) 直接和 optimization 难度 (condition number) 挂钩。

## 实验结果

主表 (Table 1) 几个关键数字:

**UMaze 环境 (MPC success rate)**:
- DINOv2 patch + projector baseline: 81.33%
- + straightening: **100%**

**Wall 环境 (MPC)**:
- DINOv2 patch baseline: 76.67%
- + straightening: **100%**

**PushT (open-loop)**:
- baseline: 70%
- + straightening: 77.33%

不是小数点的提升 — 是从"勉强能用"到"几乎完美"。

更有意思的是 **ResNet from scratch**:
- 不加 straightening: Wall open-loop **1.33%** (基本完全失败)
- 加 straightening: **84.67%**

这个对比说明 from-scratch encoder 学出的 representation 极度病态 (Figure 5 里它的 cosine similarity 最低)，但 straightening 把它救活了。这印证了 paper 的核心论点 — representation 几何性质是 planning 成败的关键。

### GD vs CEM 对比

这是另一个亮点。CEM 是 sampling-based planner，每次规划 sample 200 个 trajectory 跑 10 iterations — 慢。GD 只需要 backprop 100 步 — 快得多。

| Method | Wall GD | Wall CEM | UMaze GD | UMaze CEM |
|--------|---------|----------|----------|-----------|
| No straighten | 80% | 92% | 44% | 75.33% |
| + straighten | **90.67%** | 100% | **94%** | 94% |

Straightening 之后 GD 几乎赶上 CEM，但速度快几个数量级。这是 paper 工程上最有价值的 takeaway — **不需要昂贵的 sampling-based planning，只要 representation 学得好，简单 GD 就够**。

### Teleported-PointMaze — 最聪明的设计

这是个 killer experiment。他们改 PointMaze: 碰右墙瞬间传送到左边。这创造了一种诡异的 state 分布:
- 右墙附近和左墙附近: 视觉上像 (都靠墙)
- 但 temporal distance 巨大 (必须 teleport 才能到)

DINOv2 这种基于 visual similarity 的 encoder 会说"这俩挺像的"，planner 以为"快到了"实际根本到不了。Straightened encoder 学到了 dynamics，正确反映 teleportation 关系 — distance heatmap 上左右墙区域显示远距离，符合 A* 的 ground-truth。

这证明 straightening 不是 visual shortcut — 它真的在 encoding temporal dynamics。

## 更深的 Intuition

### 为什么 JEPA 训练已经 implicit 拉 直一点

Paper 发现即使不加 explicit straightening loss，只 train prediction loss，trajectory 也会比 DINOv2 baseline 直一些 (Figure 5)。为什么？

JEPA 的 objective 是 predict $z_{t+1}$ from $z_t$ and $a_t$。如果 trajectory 弯弯曲曲，每步 direction 都变，predictor 很难学 — 它要 model 一个高度 nonlinear 的 mapping。如果 trajectory 直，predictor 只需要 model 一个接近 affine 的 mapping — 简单。

所以 prediction pressure 隐式 favor 让 trajectory 变直的 representation。这是 JEPA 的 implicit geometric regularization。

但这不够强 — explicit straightening loss 进一步 amplify 这个效果。这就像 L2 regularization 和 weight decay 的关系 — 你可以靠 weight decay implicit 做，但 explicit L2 更 controllable。

### Spatial Features 比 Global Features 好

实验发现 $14 \times 14 \times 8$ (spatial, 8 channel) 比 $1 \times 384$ (global vector) 好。这看起来反直觉 — global vector 是更抽象的 summary，应该更好 planning？

实际相反。Intuition:
- Spatial features 保留 local geometry，predictor 可以 attend 到具体 patch，model local dynamics
- Global vector 把所有信息 collapse 到一个点，丢掉 where 信息
- 8 channel 已经够 encode planning-relevant info，更多 channel 反而让 optimization landscape 更复杂

这个发现 align 最近 representation learning 趋势: 不要过早 pool，保留 spatial structure 直到必须 aggregate。DINOv2 patch feature 比 CLS token 强，但之前 latent planning 工作多用 CLS 因为简单。

### 为什么不需要 Negative Samples

对比学习 (SimCLR, MoCo) 需要 negative pairs 防止 representation collapse — 如果只有 positive pair，encoder 把所有 image 映射到同一点，loss = 0，但 representation 无用。

这篇 paper 用 stop-gradient (SimSiam, BYOL 的 trick) 防止 collapse。架构上:

```
o_t → encoder → z_t → predictor(z_t, a_t) → ẑ_{t+1}
                                                    ↓
                                              MSE loss
                                                    ↓
                              z_{t+1} ← stop-grad ← encoder ← o_{t+1}
```

Target branch 有 stop-gradient，gradient 只从 prediction loss 流回 online encoder 和 predictor。Target 是个 fixed reference (在每步内)，encoder 不能 trivially 把它 collapse 掉 — 它必须真的 encode $o_{t+1}$ 的 relevant 信息才能被 predictor 准确预测。

加上 straightening loss 也帮助 anti-collapse — cosine similarity 要有意义，velocity 不能全为零，所以 encoder 不能输出常数。

### Long Horizon 仍然 Hard

50 步 horizon 的成功率显著下降 (PushT open-loop 13.33%)。这是 latent planning 的 fundamental challenge — **prediction error compounding**。每步小 error，unroll 50 步累积成大 drift。

Straightening 帮不上 compounding error — 它改善的是单次 planning 的 optimization landscape，不是 predictor 的 long-horizon accuracy。Long-horizon planning 可能需要:
- Hierarchical planning (high-level plan in abstract space, low-level plan in detail)
- Latent diffusion (生成整条 trajectory 而非 step-by-step)
- 多步训练 (直接 train multi-step prediction，而非 1-step)

但这是 open problem，不是这篇 paper 能解决的。

## 我的几个反思

### 1. Representation Geometry 是 Planning 的 Underexplored Dimension

过去 latent planning 工作 focus 在两件事: (a) 学好 dynamics predictor，(b) 用强 planner (CEM/MPPI/diffusion)。这篇 paper 说: 还有第三个维度 — **representation 的几何性质**。即使 predictor 完美，representation 弯曲也会让 planning 难做。

这跟 language modeling 里的 positional encoding、rotary embedding 类似 — representation 的 geometric structure 影响 downstream computation。

### 2. Cross-Disciplinary 的 Power

这篇 paper 把 neuroscience (perceptual straightening) + control theory (controllability Gramian, condition number) + deep learning (JEPA, representation learning) 串起来。三个 field 各自的贡献:
- Neuroscience 给 inspiration
- Control theory 给 formal analysis
- Deep learning 给 scalable implementation

这是好研究的标志 — 不是 incremental 改 SOTA，而是把不同 field 的 insight 联系起来，开新 direction。

### 3. Simple Idea, Strong Theory

Straightening loss 简单到 5 行代码。但有 theorem 证明它改善 Hessian condition number，有实验证明 20-60% 的 success rate 提升。

这让我想到 BatchNorm、Dropout、Skip Connection — 都是简单 idea + 强大经验效果。Straightening 可能是 latent planning 的类似 foundational trick。

### 4. Open Questions

- **Nonlinear dynamics 怎么办**？Paper 假设 linear，真实 ViT predictor 高度 nonlinear。Local linearization + state-dependent Jacobian analysis 是显然的下一步。
- **Real-world robotic manipulation** 能 work 吗？2D navigation + PushT 算简单 visual。3D、occlusion、multi-object 还没测。
- **和 Diffusion Policy 比**？Diffusion Policy 在 action space 用 diffusion 避开 planning non-convexity。Straightening 在 latent space 让 planning landscape 变 convex。两者是 complementary approach — 能结合吗？
- **Connection 到 LLM planning**？LLM 做 planning (e.g. Tree of Thoughts) 也是在 representation space 优化。Representation 的几何性质如何影响 LLM planning 效率？

## 总结

这篇 paper 用一句话总结: **representation 的几何性质决定 planning 的可优化性**。

它从 neuroscience 拿 inspiration，用 control theory 给 formal analysis，用 deep learning 实现，最终在简单任务上拿到强 empirical gain。最重要的是，它指出了一个被忽视的 dimension — representation geometry — 在 latent planning 中的重要性。

如果你做 latent planning，下一步应该测的就是你的 representation 是不是 curved 的。算一下 cosine similarity of consecutive velocities，如果低，加个 straightening regularizer，可能直接 10%+ success rate 提升。

**Web Links**:
- Paper: https://agenticlearning.ai/temporal-straightening
- Hénaf et al. 2019 (perceptual straightening): https://www.nature.com/articles/s41593-019-0377-4
- DINOv2: https://arxiv.org/abs/2304.07193
- DINO-WM: https://arxiv.org/abs/2411.04983
- V-JEPA2: https://arxiv.org/abs/2506.09985
- SimSiam: https://arxiv.org/abs/2011.10566
- BYOL: https://arxiv.org/abs/2006.07733
- Hosseini & Fedorenko 2023 (LLM straightening): https://www.biorxiv.org/content/10.1101/2023.06.23.546237
- PushT / Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DreamerV3: https://arxiv.org/abs/2301.04104

---

# Temporal Straightening for Latent Planning 深度解析

## 1. Paper 的核心 Insight

这篇 paper 来自 Yann LeCun 组 (NYU) + Brown + Toronto，处理 latent planning 中一个 fundamental 但是之前被忽视的问题:**representation 的几何性质如何影响 planning 的可优化性**。

核心 insight 用一句话概括: **pretrained visual encoder (DINOv2) 产生的 latent trajectory 在时间维度上是高度弯曲的 (curved)，导致 Euclidean distance 无法忠实反映 geodesic distance，gradient-based planner 因此陷入 non-convex landscape**。作者通过一个简单的 curvature regularizer，把 latent trajectory "拉直" (straighten)，让 planning objective 的 Hessian condition number 指数级改善。

这背后的灵感来自 neuroscience 中的 **perceptual straightening hypothesis** (Hénaf, Goris & Simoncelli, *Nature Neuroscience* 2019): 人类 V1 cortex 把 natural video 转换成内部 representation 时，trajectory 在时间上变得更接近直线，便于 predict。paper link: https://www.nature.com/articles/s41593-019-0377-4

## 2. 为什么这是个真问题

### 2.1 Latent Planning 的标准范式

Latent world model 包含三个 component:
- Sensory encoder $\mathcal{E}_\phi^s: o_t \mapsto z_t \in \mathbb{R}^d$
- Predictor $f_\theta$: 在 latent space 预测 dynamics
- Planning: minimize $\mathcal{L}(\mathbf{a}) = \|z_K - z_g\|_2^2$，通过 backprop through dynamics 优化 action sequence $\mathbf{a} = (a_0, \dots, a_{K-1})$

### 2.2 DINOv2 Embedding 的几何病态

DINOv2 (Oquab et al., TMLR 2024; https://arxiv.org/abs/2304.07193) 是 strong semantic encoder，但它的 training objective (self-distillation with patch-level features) 没有 reason 关心 temporal dynamics。结果:

- 相邻 frame 的 latent 之间方向变化剧烈 (high curvature)
- MSE cost 在 embedding space 上是 **non-convex** 的 (paper Figure 4a 可见)
- GD planner 容易 stuck 在 local minima
- 大多数 latent planning 工作 (Dreamer, TD-MPC2, DINO-WM) 退而求其次用 CEM/MPPI 等 sampling-based 方法，但 compute 昂贵

DINO-WM (Zhou et al., ICML 2025; https://arxiv.org/abs/2411.04983) 直接在 frozen DINOv2 patch feature 上学 predictor + CEM planning，是这篇 paper 的主要 baseline。

### 2.3 Geodesic vs Euclidean Distance

这是 paper 最深刻的 observation 之一。考虑迷宫中两个 state $s_1, s_2$:
- **Geodesic distance**: 沿 feasible transitions 走的最短步数 (反映真实 planning 难度)
- **Euclidean distance**: $\|z_1 - z_2\|_2$ in embedding space

当 latent trajectory 高度 curved 时，两点之间的 Euclidean 直线穿越不可达区域，与 geodesic 严重不符。Paper Figure 6 把这件事可视化得非常漂亮: DINOv2 CLS embedding 的 distance heatmap 几乎和 geodesic 没关系，而 straightened embedding 的 heatmap 几乎完美对齐 A* 计算的 ground-truth geodesic。

## 3. 方法详解

### 3.1 World Model 架构

完整架构 (paper Figure 3):

```
o_t (RGB image)
   ↓
[Visual Encoder E^s_φ]  ← 可以是 frozen DINOv2 + trainable projector，或 from-scratch ResNet
   ↓
z_t^v ∈ R^{m_v × d_v}   (spatial)  或  z_t^v ∈ R^{d_v}  (global)
   ↓
[Concat with action embedding z_t^a = E^a_ψ(a_t) ∈ R^{d_a}  and  proprioception embedding]
   ↓
[Predictor f_θ: ViT with causal temporal mask]
   ↓
ẑ_{t+1}  (predicted next latent)
```

Predictor 是 ViT-based，使用 temporal causal attention mask，让 time $t$ 的 token 只 attend to $\{t-K, \dots, t-1\}$，实现 frame-level autoregressive prediction。History length $K=3$，frameskip=5。

### 3.2 Curvature Regularizer — 关键贡献

定义三个连续 latent points $z_t, z_{t+1}, z_{t+2}$，构造 approximate latent velocity:

$$v_t = z_{t+1} - z_t, \quad v_{t+1} = z_{t+2} - z_{t+1} \tag{3}$$

这里 $v_t$ 是 latent space 中的"瞬时速度"，即从 $t$ 到 $t+1$ 的 displacement vector。

**Curvature** 通过两个连续 velocity 之间的 angle 衡量:

$$\mathcal{C} = \frac{v_t \cdot v_{t+1}}{\|v_t\|_2 \cdot \|v_{t+1}\|_2} \tag{4}$$

- $v_t \cdot v_{t+1}$: 内积，衡量同向程度
- $\|v_t\|_2$: 速度大小 (norm)，下标 2 表示 L2 norm
- $\mathcal{C} \in [-1, 1]$: 即 cosine similarity
  - $\mathcal{C} = 1$: 速度方向完全一致，trajectory 局部是直线
  - $\mathcal{C} = 0$: 方向垂直，90° 转弯
  - $\mathcal{C} = -1$: 完全反向，trajectory 折返

**Straightening loss**:

$$\mathcal{L}_{curv} = 1 - \mathcal{C} \tag{6}$$

最小化这个 loss 等价于最大化 cosine similarity，等价于最小化连续 velocity 之间的 angle。当 $\mathcal{L}_{curv} \to 0$，trajectory 局部趋近直线。

### 3.3 完整 Training Objective

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{curv} \tag{7}$$

其中:
- $\mathcal{L}_{pred} = \|\hat{z}_{t+1} - \text{sg}(z_{t+1})\|_2^2$ (Eq. 5)
  - $\hat{z}_{t+1}$: predictor 输出
  - $\text{sg}(\cdot)$: stop-gradient，阻止 gradient 流回 target encoder branch (类似 SimSiam, BYOL)
  - 防止 representation collapse
- $\lambda \geq 0$: straightening 强度。Paper 中 $\lambda = 0.1$ (with learnable pooling head) 或 $\lambda = 0.01$ (其他 variant)

**为什么 stop-gradient 能防 collapse**: 如果 target branch 也接收 gradient，encoder 可以 trivially 把所有 $z_t$ 映射到同一点，让 prediction loss 为 0。stop-gradient 让 target 是 fixed reference (in each step)，encoder 必须真正学有意义 representation 才能预测。SimSiam (Chen & He, CVPR 2021; https://arxiv.org/abs/2011.10566) 和 BYOL (Grill et al., NeurIPS 2020; https://arxiv.org/abs/2006.07733) 都用类似 trick。

### 3.4 Spatial Features 的 Cosine Similarity 变体

当 encoder 输出 spatial features $z_t^v \in \mathbb{R}^{m_v \times d_v}$ ($m_v > 1$, 比如 $14 \times 14 = 196$ patches)，velocity 也是 spatial tensor $v_t \in \mathbb{R}^{m_v \times d_v}$。Paper 比较 4 种 cosine similarity 计算方式 (Appendix B.5):

1. **[patch]**: 每个 patch 独立计算 cosine 再平均
   $$\mathcal{C}_t = \frac{1}{m_v} \sum_{i=1}^{m_v} \cos(v_{t,i}, v_{t+1,i})$$
   - $v_{t,i} \in \mathbb{R}^{d_v}$: 第 $i$ 个 patch 的 velocity vector

2. **[mean]**: 先把 patches 平均成一个 vector 再算 cosine
   $$\bar{v}_t = \frac{1}{m_v}\sum_i v_{t,i}, \quad \mathcal{C}_t = \cos(\bar{v}_t, \bar{v}_{t+1})$$

3. **[flatten]**: 把 spatial 维度展平成一个长 vector 再算一次 cosine
   $$\mathcal{C}_t = \cos(\text{vec}(v_t), \text{vec}(v_{t+1}))$$
   - $\text{vec}: \mathbb{R}^{m_v \times d_v} \to \mathbb{R}^{m_v d_v}$

4. **[agg]**: 用 learnable pooling head $h_\phi$ 聚合后计算 (MLP, output dim 128)
   $$\mathcal{C}_t = \cos(h_\phi(v_t), h_\phi(v_{t+1}))$$

实验结果 (Figure 11): **[agg] 效果最好**。直觉: straightening 应该作用于 global trajectory representation，而 spatial token 主要捕捉 local patch-level variation (由于 object motion, occlusion)，时间上不对齐。

## 4. 理论分析 — Build Intuition

### 4.1 Linear Dynamics 假设

为 tractable 分析，假设 latent dynamics 是 linear 的 (Assumption 4.1):

$$z_{t+1} = A z_t + B a_t \tag{9}$$

- $A \in \mathbb{R}^{d \times d}$: state transition matrix (无控制时的 dynamics)
- $B \in \mathbb{R}^{d \times d_a}$: control input matrix (action 如何影响 state)
- $d$: latent dimension
- $d_a$: action dimension

这是 control theory 的经典 LTI (Linear Time-Invariant) 系统。

### 4.2 ε-Straight 定义

**Definition 4.2**: dynamics $f$ 是 $\epsilon$-straight 当

$$\|A - I\|_2 \leq \epsilon \tag{10}$$

- $I$: identity matrix
- $\|\cdot\|_2$: spectral norm (最大 singular value)
- $A - I$: 衡量 transition 偏离 identity 的程度

**直觉**: 当 $\epsilon \to 0$，$A \to I$，dynamics 退化为 $z_{t+1} = z_t + B a_t$。这意味着没有 action 时 state 不变 (静止)，有 action 时 state 沿 action 方向 linearly 移动 — 这是最"直"的 dynamics。Action 是 trajectory 的直接 driver，state 不被 hidden dynamics 弯曲。

### 4.3 Planning Hessian 的 Condition Number

**Goal**: minimize $\mathcal{L}(\mathbf{a}) = \|z_K - z_g\|_2^2$，其中 $z_K = \Phi(\mathbf{a})$ 是 unrolling 后的 terminal state。

Unrolling linear dynamics (Eq. 16):

$$z_K = A^K z_0 + \sum_{t=0}^{K-1} A^{K-1-t} B a_t$$

- $A^K z_0$: initial state 经过 $K$ 步 free evolution
- $\sum_{t} A^{K-1-t} B a_t$: actions 累积的影响，每个 $a_t$ 经过 $K-1-t$ 步 free evolution

Jacobian w.r.t. actions:

$$J_\Phi = \frac{\partial z_K}{\partial \mathbf{a}} = [A^{K-1}B \mid A^{K-2}B \mid \cdots \mid B] \in \mathbb{R}^{d \times K d_a} \tag{17}$$

这是 block matrix，每块 $A^{K-1-t}B \in \mathbb{R}^{d \times d_a}$ 表示 $a_t$ 对 $z_K$ 的 sensitivity。

**Hessian** (Eq. 19):

$$H = \nabla_\mathbf{a}^2 \mathcal{L} = 2 J_\Phi^\top J_\Phi \succeq 0$$

由于 $\mathcal{L}$ 是 $\mathbf{a}$ 的 convex quadratic (因为 $z_K$ 是 $\mathbf{a}$ 的 affine function)，Hessian 是 PSD。

**Controllability Gramian** (Eq. 18):

$$\mathcal{W}_K = J_\Phi J_\Phi^\top = \sum_{k=0}^{K-1} A^k B B^\top (A^\top)^k$$

这是 control theory 中衡量系统可控性的核心量 (Kailath 1980; Sontag 1998; Chen 1999)。$\mathcal{W}_K$ "大" 表示系统容易 control。

**Effective condition number** (Definition C.2): 对 PSD matrix $H$，

$$\kappa_{\text{eff}}(H) = \frac{\sigma_{\max}(H)}{\sigma_{\min}^+(H)}$$

- $\sigma_{\max}$: 最大 singular value
- $\sigma_{\min}^+$: 最小非零 singular value (处理 singular 情况)

### 4.4 Main Theorem (Theorem 4.4)

$$\kappa_{\text{eff}}(H) = \kappa(\mathcal{W}_K) \leq \kappa(B)^2 \kappa(A)^{2(K-1)} \tag{11}$$

其中 $\kappa(M) = \sigma_{\max}(M)/\sigma_{\min}(M)$ 是 condition number。

**如果 $\epsilon$-straight** ($\epsilon = \|A - I\|_2 < 1$):

$$\kappa_{\text{eff}}(H) \leq \kappa(B)^2 \left(\frac{1+\epsilon}{1-\epsilon}\right)^{2(K-1)} \tag{12}$$

对 $\epsilon \leq 1/2$:

$$\kappa_{\text{eff}}(H) \leq \kappa(B)^2 e^{6 \epsilon K}$$

**Intuition**:
- 当 $\epsilon$ 小 (dynamics 接近 straight)，condition number 随 horizon $K$ **指数级增长缓慢** ($e^{6\epsilon K}$)
- 当 $\epsilon$ 大 (curved dynamics)，$\kappa(A)^{2(K-1)}$ 项可以爆炸式增长，导致 Hessian 极度 ill-conditioned
- GD 在 ill-conditioned quadratic 上的 convergence rate 是 $(1 - 1/\kappa)^t$，所以 $\kappa$ 大意味着 GD 慢且 unstable

### 4.5 证明关键步骤

**上界** (max eigenvalue):
$$x^\top \mathcal{W}_K x = \sum_k \|B^\top (A^\top)^k x\|_2^2 \leq \sigma_{\max}(B)^2 \sum_k \sigma_{\max}(A)^{2k}$$

- 第一步: quadratic form 展开
- 第二步: sub-multiplicative norm inequality $\|B^\top (A^\top)^k x\| \leq \|B\| \cdot \|A^k\| \cdot \|x\|$

**下界** (min nonzero eigenvalue): 当 $B$ invertible，
$$\|B^\top (A^\top)^k x\| \geq \sigma_{\min}(B) \sigma_{\min}(A)^k$$

- 用 $\|B^\top u\| \geq \sigma_{\min}(B) \|u\|$ 和 $\sigma_{\min}(A^k) \geq \sigma_{\min}(A)^k$

**关键步骤**: 用 Weyl's perturbation theorem
- $\sigma_{\max}(A) \leq 1 + \epsilon$ (因为 $\|A - I\| \leq \epsilon$，最大 singular value 偏离 1 不超过 $\epsilon$)
- $\sigma_{\min}(A) \geq 1 - \epsilon$

代入得到 $\left(\frac{1+\epsilon}{1-\epsilon}\right)^{2(K-1)}$。再用 $\ln\left(\frac{1+\epsilon}{1-\epsilon}\right) \leq 3\epsilon$ (对 $\epsilon \leq 1/2$)，得到 exponential bound $e^{6\epsilon K}$。

### 4.6 Cosine Similarity 作为 Practical Proxy

实际中我们 minimize cosine similarity loss，但理论保证是基于 $\|A - I\|_2$。Paper Proposition C.9 建立 bridge:

**Assumption C.7**: 常速度 $\|v_t\| = c$ 和平滑 action $\Delta_a = \max_t \|a_{t+1} - a_t\| < \infty$。

**Proposition C.9**: 对每个 $t$,

$$\|(A - I) \hat{v}_t\|_2 \leq \sqrt{2(1 - \mathcal{C}_t)} + \frac{\sigma_{\max}(B) \Delta_a}{c} \tag{23}$$

- $\hat{v}_t = v_t / \|v_t\|$: 单位速度方向
- $\mathcal{C}_t$: cosine similarity
- 第一项: curvature contribution，$\mathcal{C}_t \to 1$ 时为 0
- 第二项: action 变化 contribution，action 越平滑越小

**关键 insight**: cosine similarity 高 ⟹ $(A - I)$ 在 visited directions 上小。要从 directional control 升级到 spectral control ($\|A-I\|_2$)，需要 trajectories 覆盖足够多方向 (Remark C.10)，这在 diverse training data 下通常成立。

## 5. 实验结果深度分析

### 5.1 环境

- **Wall**: 两个 room 通过 narrow door 连接 (Zhou et al. 2025, Sobal et al. 2025)
- **PointMaze UMaze**: MuJoCo-based 2D navigation (D4RL, Fu et al. 2020; https://arxiv.org/abs/2004.07219)
- **PointMaze Medium**: 更复杂的 maze
- **PushT**: Contact-rich pushing task (Chi et al. 2025, Diffusion Policy; https://diffusion-policy.cs.columbia.edu/)
- **Teleported-PointMaze**: Novel environment，碰右墙瞬间传送到左边，测试是否真正学 dynamics 而非 visual similarity

### 5.2 主结果表 (Table 1) 关键数字

| Encoder | Dim | Straighten | Wall Open | Wall MPC | UMaze Open | UMaze MPC | PushT Open | PushT MPC |
|---------|-----|-----------|-----------|----------|------------|-----------|------------|-----------|
| DINOv2 (patch) + proj | 14×14×8 | ✗ | 80.00 | 90.67 | 44.00 | 81.33 | 70.00 | 78.67 |
| DINOv2 (patch) + proj | 14×14×8 | ✓ | **90.67** | **100.00** | **94.00** | **100.00** | 77.33 | 85.33 |
| ResNet from scratch | 14×14×8 | ✗ | 1.33 | 6.67 | 14.67 | 66.00 | 71.33 | 70.67 |
| ResNet from scratch | 14×14×8 | ✓ | 84.67 | 100.00 | 64.67 | 98.67 | 70.67 | 91.33 |

**Key observations**:

1. **Straightening 带来 10-60% 提升**: UMaze open-loop 从 44% → 94% (+50 pp)，Wall open-loop 从 80% → 90.67%

2. **ResNet from scratch 没有 straightening 时崩溃**: Wall 只有 1.33%，因为 from-scratch encoder 学出的 representation 极度 curved (Figure 5 显示)。加 straightening 后 jump 到 84.67%。

3. **Spatial features (14×14×8) 比 global (1×384) 好**: 保留 spatial structure 让 predictor 更准确，channel dimension 可以极度压缩 (384 → 8) 不掉性能

4. **Implicit straightening**: 即使不显式加 $\mathcal{L}_{curv}$，仅用 prediction loss 训练 projector 也会让 trajectory 变直一点 (Figure 5)。这是 JEPA 的 implicit benefit: prediction pressure 自然 favor 容易 predict 的 representation。

### 5.3 Long-Horizon 结果 (Table 2)

50 步 horizon (vs 主实验 25 步):

| Model | PushT Open | PushT MPC | Medium Open | Medium MPC |
|-------|-----------|-----------|-------------|------------|
| DINO-WM | 3.33 | 27.33 | 35.00 | 65.33 |
| + Channel Proj + Straighten | 13.33 | 24.00 | 68.00 | 88.00 |
| + ResNet + Straighten | 10.67 | 33.33 | 76.00 | 98.67 |

Long-horizon 仍然 challenging (compounding prediction error)，但 straightening 优势在 long horizon 更明显，与理论 $\kappa_{\text{eff}} \propto e^{6\epsilon K}$ 吻合: horizon 越长，straightening 的几何 benefit 越大。

### 5.4 GD vs CEM (Table 3)

Paper 一个 sub-goal 是 enable gradient-based planning 替代 sampling-based:

| Method | Straighten | Wall GD | Wall CEM | UMaze GD | UMaze CEM |
|--------|-----------|---------|----------|----------|-----------|
| DINOv2 patch | ✗ | 73.33 | 87.33 | 63.33 | 88.00 |
| + proj | ✗ | 80.00 | 92.00 | 44.00 | 75.33 |
| + proj | ✓ | **90.67** | 100.00 | **94.00** | 94.00 |

GD + straightening 接近 CEM performance，但 GD 比 CEM 快得多 (CEM 需要 200 samples × 10 iterations)。这印证 paper 的核心 motivation: straightening 让 latent space 适合 gradient-based optimization。

### 5.5 Teleported-PointMaze — 关键 Ablation

这是非常聪明的实验设计。环境修改: 碰右墙瞬间 teleport 到左边。这创造 **visual similar 但 temporal distant** 的 states:
- 左墙附近 vs 右墙附近: 视觉相似
- 但 temporal distance 极大 (必须 teleport 才能到达)

DINOv2 这种基于 visual similarity 的 encoder 会失败 (Figure 23a)。Straightened encoder 学到了 dynamics，正确反映 teleportation (Figure 23b)。这证明 straightening 真正学 dynamics 而非 visual shortcut。

### 5.6 Dimension Ablation (Figure 10)

Channel dimension $d_v \in \{2, 8, 32, 128\}$:
- $d_v = 2$: 信息不足，performance 差
- $d_v \in \{8, 32\}$: 最佳
- $d_v = 128$: 反而下降

**Intuition**: 高维 latent 让 dynamics prediction 和 optimization 都变难。这与 manifold hypothesis 一致 — planning-relevant information 应该 low-dimensional。

## 6. 与 Related Work 的深层联系

### 6.1 JEPA Family

Paper 属于 JEPA paradigm (LeCun 2022, "A Path Towards Autonomous Machine Intelligence"; https://openreview.net/pdf?id=BZ5a1r-kVsf)。JEPA 核心思想:
- 不重建 pixel (区别于 Dreamer, World Models)
- 在 latent space 预测
- 通过 predictability 学习 meaningful representation
- Avoid 处理 unpredictable details

V-JEPA2 (Assran et al. 2025; https://arxiv.org/abs/2506.09985) 是大规模 video pretraining 的 JEPA 实例。这篇 paper 在 JEPA 基础上加 geometric regularization，是 JEPA 的 refinement。

### 6.2 Perceptual Straightening 在 Neuroscience

Hénaf et al. 2019 (https://www.nature.com/articles/s41593-019-0377-4) 发现:
- Natural video 在 pixel space 是高度 curved 的
- macaque V1 的 neural population response 把 video trajectory "拉直"
- 直线 trajectory 更容易 temporal prediction

后续工作:
- Harrington et al. ICLR 2023 (https://arxiv.org/abs/2210.02897): 在 deepnets 中研究 implicit straightening
- Niu et al. NeurIPS 2024 (https://arxiv.org/abs/2405.18043): 用 straightening 作 objective 训练 robust video model
- Bagad & Zisserman, NeurIPS 2025: chirality-aware video representation

这篇 paper 是首次把 straightening 用于 planning。

### 6.3 Language Model 中的 Straightening

Hosseini & Fedorenko 2023 (https://www.biorxiv.org/content/10.1101/2023.06.23.546237): next-word prediction 训练让 LLM 的 sentence trajectory 在 representational space 中变直。Hosseini et al. 2026: context structure reshape representational geometry。这暗示 straightening 可能是 predictive learning 的 universal property。

### 6.4 Controllability Gramian 与 Optimal Control

Controllability Gramian $\mathcal{W}_K = \sum_k A^k B B^\top (A^\top)^k$ 是经典 control theory 概念:
- Kalman 1960s 提出
- $\mathcal{W}_K$ 正定 ⟺ 系统 controllable
- LQR, MPC 的 analysis 基础
- Reference: https://www.sciencedirect.com/topics/engineering/controllability-gramian

Paper 把 control theory 的 condition number 分析与 representation learning 联系，是个 elegant 的 cross-disciplinary insight。

### 6.5 Dreamer 系列

Dreamer (Hafner et al. ICLR 2020; https://arxiv.org/abs/1912.01603), DreamerV2 (2021), DreamerV3 (2023; https://arxiv.org/abs/2301.04104) 用 reconstruction-based latent dynamics + actor-critic in latent space。它们不显式做 planning at test time (用 policy network)。这篇 paper 走的是 differentiable planning 路线，更接近 TD-MPC2 (Hansen et al. ICLR 2024; https://arxiv.org/abs/2310.16828) 和 DINO-WM。

### 6.6 Representation Learning 防止 Collapse

- **VICReg** (Bardes et al. ICLR 2022; https://arxiv.org/abs/2105.04906): variance + invariance + covariance regularization
- **SimSiam** (Chen & He, CVPR 2021): stop-gradient + predictor
- **BYOL** (Grill et al. NeurIPS 2020): stop-gradient + momentum encoder
- **LeJEPA** (Balestriero & LeCun 2025; https://arxiv.org/abs/2511.08544): provable 没有 heuristics 的 JEPA

这篇 paper 用最简单的 stop-gradient，足以防 collapse，避免 contrastive negative sampling 的 complexity。

### 6.7 Temporal Contrastive Learning

TCN (Sermanet et al. ICRA 2018; https://arxiv.org/abs/1704.08045), TCLR (Dave et al. 2022) 用 InfoNCE 鼓励 temporally close frames similar，distant dissimilar。问题: 选择 positive/negative pair 需要调，suboptimal trajectories 会错误 push apart geodesically close states。Straightening 不需要 negatives，纯 local regularization，更 robust。

## 7. Intuition Building — 综合理解

### 7.1 为什么 Straightening 改善 Planning

多层次因果链:

1. **Geometric level**: Straightening 让 latent trajectory 局部线性化，Euclidean distance ≈ geodesic distance
2. **Optimization level**: Linear dynamics + straight trajectory → Hessian condition number 指数级改善
3. **Algorithmic level**: GD planner convergence rate $\propto 1/\kappa$，condition number小 → 收敛快且 stable
4. **Empirical level**: 20-60% success rate 提升，Wall/UMaze 达到 100% MPC success

### 7.2 Implicit Straightening 现象

Paper 发现仅用 prediction loss 训练也会 implicit straighten (Section 5.2)。Intuition:
- JEPA 要预测 $z_{t+1}$ from $z_t$
- 如果 trajectory 高度 curved，$z_{t+1}$ 难预测 (方向变化大)
- Encoder 会自然学到让 trajectory 平滑的 representation
- 这与 LLM 的 next-word prediction 导致 sentence straightening 完全 parallel

Explicit straightening loss $\mathcal{L}_{curv}$ 把 implicit effect explicit 化并 amplify。

### 7.3 Spatial vs Global Features

Paper 发现 spatial features ($14 \times 14 \times 8$) 比 global ($1 \times 384$) 好。Intuition:
- Spatial features 保留 local geometry
- Predictor 可以 attend to specific patches，model local dynamics
- Global vector collapse 后信息损失大
- 8 channel 足够编码 planning-relevant info，更高维反而让 optimization 变难

这与最近 representation learning 趋势一致: 不要过早 pool，保留 spatial structure 直到 task requires aggregation。

### 7.4 为什么 Stop-Gradient 够用

JEPA-style architecture:
```
o_t ─[E_φ]→ z_t ─[f_θ, a_t]→ ẑ_{t+1} ── MSE ── sg(z_{t+1}) ←[E_φ]← o_{t+1}
```

- Target branch 有 stop-gradient，相当于 teacher
- Online branch (predictor + encoder) 是 student
- 类似 BYOL 但没有 momentum encoder (simpler)

为什么 work: prediction loss 已经足够约束 encoder 学 meaningful representation (要预测 future 就要 encode 现在 state 的 relevant 信息)。Straightening loss 进一步 shape geometry。两者协同，无需 contrastive negatives。

## 8. Limitations 和 Future Directions

### 8.1 Linear Dynamics 假设

理论分析假设 linear dynamics $z_{t+1} = A z_t + B a_t$。真实 predictor 是 ViT (highly nonlinear)。Paper 在 Remark 4.5 / Section C 提到: nonlinear case 需要 control state-dependent Jacobian 的 product 和 higher-order terms，是 future work。

可能的 nonlinear extension:
- Local linearization: $A_t = \partial f_\theta / \partial z_t$ at each step
- 控制/products of Jacobians $\prod_t A_t$ 的 spectral radius
- 类似 Lipschitz analysis for recurrent networks

### 8.2 Vision-Centric

实验都是 2D navigation + simple manipulation。Real-world robotic manipulation (3D, multi-object, occlusion) 是否 work 未知。V-JEPA2 在 real-world video 上 pretrained，可能更 scalable baseline。

### 8.3 Long Horizon

50 步 horizon success rate 仍然低 (PushT open-loop 13.33%)。Compounding prediction error 是 fundamental challenge。可能需要:
- Hierarchical planning
- Latent space diffusion
- Trajectory-level (而非 step-level) prediction

### 8.4 与 Other Representation Learning Method 结合

可以尝试 combine:
- Straightening + VICReg variance/covariance regularization
- Straightening + contrastive on hard negatives
- Straightening + Equivariance to action (让 latent explicitly equivariant to action)

### 8.5 Straightening 与 Diffusion Policy

Diffusion Policy (Chi et al. IJRR 2025) 在 action space 做 diffusion，避开 planning 的 non-convexity。能否在 straightened latent space 做 diffusion？或用 straightening 解释 diffusion policy 的 success？

### 8.6 Connection to Tokenizer / VQ-VAE

Paper 用 VQ-VAE decoder (van den Oord et al. 2017; https://arxiv.org/abs/1711.00937) 仅作 interpretability。Discrete latent token + straightening 是否有意义？可能 connection 到 world models as language modeling。

## 9. 核心公式速查表

| 公式 | 含义 |
|------|------|
| $v_t = z_{t+1} - z_t$ | Latent velocity，相邻 latent 的差 |
| $\mathcal{C} = \frac{v_t \cdot v_{t+1}}{\|v_t\|_2 \|v_{t+1}\|_2}$ | Cosine similarity，衡量速度方向一致性 |
| $\mathcal{L}_{curv} = 1 - \mathcal{C}$ | Curvature loss |
| $\mathcal{L}_{pred} = \|\hat{z}_{t+1} - \text{sg}(z_{t+1})\|_2^2$ | Prediction loss with stop-gradient |
| $\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{curv}$ | Total objective |
| $\|A - I\|_2 \leq \epsilon$ | ε-straight definition |
| $\mathcal{W}_K = \sum_{k=0}^{K-1} A^k B B^\top (A^\top)^k$ | Controllability Gramian |
| $H = 2 J_\Phi^\top J_\Phi$ | Planning Hessian |
| $\kappa_{\text{eff}}(H) \leq \kappa(B)^2 e^{6\epsilon K}$ | Condition number bound (ε-straight) |

## 10. 总结

这篇 paper 的 elegance 在于:
1. **Simple idea**: cosine similarity regularizer on consecutive latent velocities
2. **Deep insight**: 把 representation geometry 与 planning optimization 直接联系
3. **Strong theory**: control theory 的 condition number analysis 提供 principled 保证
4. **Clean experiments**: simple GD planner + strong empirical gains
5. **Cross-disciplinary**: neuroscience (perceptual straightening) → ML (representation learning) → control (Gramian)

核心 message: **representation 的几何性质决定 planning 的可优化性**。这是 latent planning 一个 underexplored 但 fundamental 的维度，与 representation 的 semantic quality 同等重要甚至更重要。

对未来的启示: 设计 representation learning objective 时，应该同时考虑 (i) 保留 task-relevant information，(ii) 几何性质利于下游 optimization。Straightening 是后者的一个 instance，可能还有其他 geometric regularizer (e.g., isometry, equivariance) 值得探索。

---

**Reference Links**:
- Paper: https://agenticlearning.ai/temporal-straightening
- Perceptual straightening (Hénaf et al. 2019): https://www.nature.com/articles/s41593-019-0377-4
- DINOv2: https://arxiv.org/abs/2304.07193
- DINO-WM: https://arxiv.org/abs/2411.04983
- V-JEPA2: https://arxiv.org/abs/2506.09985
- JEPA (LeCun 2022): https://openreview.net/pdf?id=BZ5a1r-kVsf
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- SimSiam: https://arxiv.org/abs/2011.10566
- BYOL: https://arxiv.org/abs/2006.07733
- VICReg: https://arxiv.org/abs/2105.04906
- PushT / Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- D4RL PointMaze: https://arxiv.org/abs/2004.07219
- LeJEPA: https://arxiv.org/abs/2511.08544
- Harrington et al. ICLR 2023 (straightening in deepnets): https://arxiv.org/abs/2210.02897
- Hosseini & Fedorenko 2023 (LLM straightening): https://www.biorxiv.org/content/10.1101/2023.06.23.546237
- VQ-VAE: https://arxiv.org/abs/1711.00937
- Controllability Gramian: https://www.sciencedirect.com/topics/engineering/controllability-gramian
