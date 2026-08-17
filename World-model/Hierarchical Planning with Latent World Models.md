---
source_pdf: Hierarchical Planning with Latent World Models.pdf
paper_sha256: 0f60bed10fff3d873f246b0655f4f4b176d3f59063e5c643e148453cc79b7f40
processed_at: '2026-08-04T23:44:37-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲HWM

## 一句话版本

**Flat world model planning在长horizon任务上会崩，加一层high-level planner在latent space里先粗规划再让low-level精细执行，就能解决需要"先远离goal才能到达goal"的非贪婪任务。**

---

## 问题是什么

假设你让机器人"把杯子从A放到B"。

Flat planner的逻辑是：每一步都贪心地minimize当前state到goal的距离。但pick-&-place这个任务，你得先**往下抓**，这时候距离goal是**变大**的。Flat planner一看"离goal越来越远了"，就不往下走，结果完全失败——0%成功率。

这就是**non-greedy task**：成功的路径中间必须有一段"远离goal"的detour。

---

## 为什么flat planner不行

两个原因，一个比一个致命：

**原因1：误差累积**

World model每预测一步都有小误差。你要预测75步以后的state，这些误差滚雪球。预测10步还靠谱，预测75步基本是瞎猜。

**原因2：搜索空间爆炸**

你要用CEM采样action sequence。如果horizon是75，action是10维连续空间，你要在这个 $\mathbb{R}^{750}$ 的空间里搜索。sample再多也不够覆盖。

---

## HWM怎么解决

核心idea特别简单：**分两层规划，粗一层细一层。**

### 高层：粗规划

用一个high-level world model，在latent space里规划2步macro-action就能到goal。

这2步macro-action展开后，给你1个intermediate latent state——这就是**subgoal**。

关键：macro-action不是primitive action，是action encoder把一长串primitive action压缩成的4维向量。搜索空间从 $\mathbb{R}^{750}$ 降到了 $\mathbb{R}^8$（2步×4维）。

### 低层：精细执行

拿到subgoal之后，low-level planner用primitive action去reach这个subgoal。因为subgoal很近（只需要几步），所以horizon短，误差不累积，搜索空间也小。

### 为什么work

- 高层只规划2步，不累积误差
- 低层只执行几步，也不累积误差
- subgoal是latent vector，两层共享encoder，直接传递，不需要decode回image
- macro-action压缩了action sequence的结构信息（比如"先上后横"这种non-greedy pattern）

---

## 最核心的insight

**通过限制high-level的表达能力（latent action只有4维），反而让low-level更容易执行。**

如果latent action维度太高，high-level会规划出很复杂的subgoal，low-level根本reach不了。维度太低，high-level又规划不出valid path。4维是个sweet spot——刚好够表达useful trajectory，又不够复杂到让low-level崩溃。

这跟人的直觉有点反——你以为越expressive越好，其实不是。**适当的bottleneck是一种regularization。**

---

## 实验结果有多convincing

最impressive的数字：

- **Pick-&-place：0% → 70%**。这是real robot，不是simulation。flat planner完全失败，加了hierarchy就能work。
- **Push-T d=75：17% → 61%**。horizon越长，hierarchy收益越大。
- **Diverse Maze OOD：44% → 83%**。test的maze layout训练时根本没见过。
- **Compute：3-4× reduction**。不只是更准，还更快。

对比VLA baseline：π0.5用77倍的数据训练，pick-&-place cup只有68%，HWM是80%。

---

## 方法上最优雅的地方

1. **Shared latent space**：两层world model用同一个encoder，subgoal直接是latent vector，不需要inverse model或者额外interface
2. **Model-agnostic**：plug into VJEPA2-AC、DINO-WM、PLDM三个完全不同的backbone都work
3. **Reward-free**：训练时完全不用reward，纯next-latent prediction，zero-shot部署
4. **No policy learning**：不训练policy，test time直接CEM planning

---

## 局限性

- **Top-down only**：low-level执行失败了没法反馈给high-level重新规划
- **只有2层**：更长的horizon可能需要3层、4层hierarchy
- **Waypoint是随机选的**：训练时随机选N=3个waypoint，不是optimal decomposition
- **Latent action维度要手调**：4维是empirical sweet spot，没有理论指导

---

## 一句话intuition

**把"规划75步primitive action"这个impossible的问题，分解成"规划2步macro-action + 每步规划几步primitive action"这个tractable的问题。关键是在shared latent space里做，让两层自然耦合。**

本质上就是经典的means-ends analysis，只不过用learned latent world model在现代deep learning setup下重新实现了一遍。

---

# Hierarchical Planning with Latent World Models (HWM) 深度解析

## 1. Core Motivation: 为什么需要Hierarchical Planning

这篇paper来自FAIR at Meta和NYU的Yann LeCun团队，第一作者Wancong Zhang。核心问题是 **long-horizon planning** 在learned world models上的两大根本挑战：

**Challenge 1: Prediction error compounding**
- World model每一步预测都有误差 $\epsilon_t$
- 经过 $T$ 步autoregressive rollout，误差累积为 $O(T \cdot \epsilon)$ 甚至更差（非线性放大）
- Talvitie (2014) 和 Janner et al. (2019) 都指出了这个问题

**Challenge 2: Curse of horizon**
- 假设action space是continuous $\mathcal{A} \subset \mathbb{R}^d$
- CEM/MPPI sampling需要覆盖 $|\mathcal{A}|^T$ 的搜索空间
- 随着 $T$ 增大，planning复杂度指数级爆炸

经典解法是引入temporal abstraction（Sutton的options framework, Sutton et al. 1999），但现有方法要么需要task-specific reward（RL methods），要么只能处理low-dim state space（classical hierarchical MPC）。这篇paper的关键insight是：**在latent space中做hierarchy，让high-level prediction直接作为low-level的subgoal**。

参考链接：
- Sutton options: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- VJEPA2: https://arxiv.org/abs/2506.09985
- PLDM: https://arxiv.org/abs/2502.14819
- DINO-WM: https://arxiv.org/abs/2411.04983

---

## 2. Architecture 解析

### 2.1 两层World Model的结构

整个系统由以下组件构成：

**Shared Encoder $E$**：
- 输入：observation $s_t$（RGB image + proprioception）
- 输出：latent state $z_t = E(s_t)$
- 在Franka实验中使用frozen ViT-G/16（VJEPA2 backbone）
- 在Push-T中使用frozen DINOv2 ViT-S/14
- 在Diverse Maze中jointly learn一个lightweight conv encoder

**Low-level World Model $P_\theta^{(1)}$**：
$$P_\theta^{(1)}(z_{t+1} | z_t, a_t)$$
- 参数 $\theta$
- 输入：当前latent $z_t$ 和 primitive action $a_t$
- 输出：下一步latent prediction $\hat{z}_{t+1}$
- 时间尺度：1步primitive action

**High-level World Model $P_\phi^{(2)}$**：
$$P_\phi^{(2)}(z_{t+h} | z_t, l_t)$$
- 参数 $\phi$
- 输入：当前latent $z_t$ 和 macro-action $l_t$
- 输出：$h$ 步后的latent prediction
- 时间尺度：variable-length segment（$h$ 可以变化）

**Action Encoder $A_\psi$**：
- 参数 $\psi$
- 输入：primitive action subsequence $a_{t_k:t_{k+1}}$
- 输出：latent macro-action $l_{t_k} \in \mathbb{R}^d$（Franka中 $d=4$）
- 架构：transformer-based，CLS token + MLP head

### 2.2 Top-Down Planning的数据流

参考Figure 2的逻辑：

```
Goal image $s_g$
    ↓ Encoder
$z_g = E(s_g)$
    ↓
[High-Level Planner: CEM over $\hat{l}_{1:H}$]
    ↓ Minimize $\|z_g - P^{(2)}(\hat{l}_{1:H}; z_1)\|_1$
    ↓
Optimal $l^*_{1:H}$ → Unroll → Subgoals $\tilde{z}_1, \tilde{z}_2, ..., \tilde{z}_H$
    ↓
[Low-Level Planner: CEM over $\hat{a}_{1:h}$ toward $\tilde{z}_1$]
    ↓ Minimize $\|\tilde{z}_1 - P^{(1)}(\hat{a}_{1:h}; z_1)\|_1$
    ↓
Execute $a^*_{1:h}$, replan every k steps
```

---

## 3. 关键公式详解

### 3.1 High-Level Energy Function

$$\mathcal{E}_2(\hat{l}_{1:H}; z_1, z_g) \triangleq \Big\| z_g - P^{(2)}(\hat{l}_{1:H}; z_1) \Big\|_1$$

**变量解释**：
- $\hat{l}_{1:H} = (\hat{l}_1, \hat{l}_2, ..., \hat{l}_H)$：候选的macro-action序列
- $H$：high-level planning horizon（Franka中 $H=2$）
- $z_1 = E(s_1)$：初始latent state
- $z_g = E(s_g)$：goal latent
- $P^{(2)}(\hat{l}_{1:H}; z_1)$：从 $z_1$ 出发，autoregressively unroll high-level model，得到最终latent
- $\|\cdot\|_1$：L1 distance（对outlier更鲁棒）

**优化**：
$$l_{1:H}^* = \arg\min_{\hat{l}_{1:H}} \mathcal{E}_2(\hat{l}_{1:H}; z_1, z_g)$$

**Subgoal extraction**：
$$\tilde{z}_i \triangleq P^{(2)}(l_{1:i}^*; z_1), \quad i = 1, \dots, H$$

这里 $l_{1:i}^*$ 是取前 $i$ 个optimal macro-actions，unroll得到第 $i$ 个intermediate latent state作为subgoal。

### 3.2 Low-Level Energy Function

$$\mathcal{E}_1(\hat{a}_{1:h}; z_1, \tilde{z}_1) \triangleq \Big\| \tilde{z}_1 - P^{(1)}(\hat{a}_{1:h}; z_1) \Big\|_1$$

**变量解释**：
- $\hat{a}_{1:h}$：候选的primitive action序列
- $h$：low-level planning horizon（Franka中 $h=2$）
- $\tilde{z}_1$：high-level planner给出的第一个subgoal
- $P^{(1)}(\hat{a}_{1:h}; z_1)$：从 $z_1$ 出发unroll low-level model

**关键设计**：这个energy function是 **latent-state matching**，直接在latent space比较距离，不需要inverse model把subgoal decode回image再处理。

### 3.3 Low-Level Training Loss

Teacher-forcing loss:
$$\hat{z}_{k+1} := P_\theta^{(1)}((a_t, z_t)_{t \leq k})$$
$$\mathcal{L}_{tf}(\theta) := \frac{1}{T}\sum_{k=1}^T \|\hat{z}_{k+1} - z_{k+1}\|_1$$

Multi-step rollout loss:
$$\mathcal{L}_{roll}(\theta) := \sum_{j=2}^T \|P_\theta^{(1)}(a_{1:j}, z_1) - z_{j+1}\|_1$$

Total:
$$L(\theta) := \gamma_{tf} \cdot \mathcal{L}_{tf}(\theta) + \gamma_{roll} \cdot \mathcal{L}_{roll}(\theta)$$

**变量解释**：
- $T$：trajectory length（Franka中 $T=15$）
- $\gamma_{tf}, \gamma_{roll}$：loss coefficients（VJEPA2-AC中都是1.0）
- $P_\theta^{(1)}(a_{1:j}, z_1)$：从 $z_1$ autoregressively rollout $j$ 步

Rollout loss非常关键——它直接supervise多步预测，mitigate compounding error。

### 3.4 High-Level Training Loss

给定trajectory $\tau = (s_1, a_1, s_2, ..., s_T)$，选 $N$ 个waypoint indices $1 = t_1 < t_2 < ... < t_N$：

$$\hat{z}_{t_{k+1}} := P_\phi^{(2)}\Big((l_{t_i}, z_{t_i})_{i \leq k}\Big)$$
$$\mathcal{L}_{tf}(\phi, \psi) := \frac{1}{N}\sum_{k=1}^N \big\|\hat{z}_{t_{k+1}} - z_{t_{k+1}}\big\|_1$$

**变量解释**：
- $\phi$：high-level world model参数
- $\psi$：action encoder参数
- $N$：waypoint数量（Franka中 $N=3$）
- $z_{t_k} = E(s_{t_k})$：waypoint的latent
- $l_{t_k} = A_\psi(a_{t_k:t_{k+1}})$：用action encoder把waypoint之间的primitive action sequence编码成macro-action
- $(l_{t_i}, z_{t_i})_{i \leq k}$：interleaved sequence作为causal input

**关键设计**：
1. **Variable-length segments**：不像固定stride的temporal abstraction，每个high-level transition可以对应不同长度的low-level execution
2. **Shared encoder $E$**：high-level和low-level用同一个encoder，保证latent space对齐，subgoal可以直接传给low-level
3. **No rollout loss for high-level**：从Table 6看 $\gamma_{roll}=0$，可能因为waypoint间隔本来就长，rollout意义不大

---

## 4. CEM Planning的具体实现

### 4.1 Cross-Entropy Method回顾

CEM (Rubinstein & Kroese, 2004) 维护一个factored Gaussian over action sequences：
- 初始化mean $\mu_0$ 和 variance $\sigma_0^2$
- 每轮采样 $N_{samples}$ 个trajectory
- 用world model evaluate，选top $N_{elites}$
- 用elites更新 $\mu, \sigma^2$
- Var EMA smoothing: $\sigma_t^2 = \alpha \cdot \sigma_{t-1}^2 + (1-\alpha) \cdot \hat{\sigma}^2$

### 4.2 Franka的Hyperparameters（Table 10）

| Planner | #elites | #iters | #samples | Var EMA | pred h/H | k |
|---------|---------|--------|----------|---------|----------|---|
| High-level | 22 | 15 | 3000 | 0.65 | 2 | - |
| Low-level | 12 | 5 | 800 | 0.25 | 2 | 1 |

注意：
- High-level用更多samples（3000 vs 800）但horizon短（2 vs 2），因为macro-action是4维连续空间
- Low-level每步replan（$k=1$），标准MPC
- High-level的Var EMA=0.65保持探索性，low-level=0.25更greedy

### 4.3 计算复杂度对比

**Flat planner**：
- 搜索空间：$|\mathcal{A}|^h$，$h=6$（Franka flat setting）
- 每个sample需要6次autoregressive prediction

**Hierarchical planner**：
- High-level：$|\mathcal{L}|^H$，$|\mathcal{L}|=\mathbb{R}^4$, $H=2$ → 大幅压缩
- Low-level：$|\mathcal{A}|^h$, $h=2$ 但只跑短期
- Total prediction calls: $H \cdot (\text{high-level unroll}) + h \cdot (\text{low-level unroll per subgoal})$

实验显示3× compute reduction（Fig 5），这是hierarchical abstraction的直接收益。

---

## 5. 实验数据深度解读

### 5.1 Franka Real Robot Results (Table 1)

| Method | P&P Cup | P&P Box | Drawer |
|--------|---------|---------|--------|
| Octo | 20% | 10% | 43% |
| π0-FAST-DROID | 52% | 18% | - |
| π0.5-DROID | 68% | 36% | - |
| VJEPA2-AC (flat) | 80%* | 80%* | 30% |
| VJEPA2-AC + HWM | **80%*** | **70%** | **70%** |

\* = with oracle subgoals

**关键观察**：
1. **Non-greedy task的0%→70%突破**：Pick-&-place需要先抓取再放置，中间状态距离goal可能增大，flat planner完全失败
2. **HWM用single goal image**就达到了π0.5（用77×数据训练的VLA）comparable的性能
3. **Drawer task的30%→70%**：multi-stage motion（先下后横移）也被hierarchy解开

### 5.2 Push-T Long Horizon (Table 2)

| Method | d=25 | d=50 | d=75 |
|--------|------|------|------|
| GCIQL | 40% | 25% | 7.5% |
| HIQL | 55% | 30% | 20% |
| HILP | 25% | 13% | 0% |
| DINO-WM | 84% | 55% | 17% |
| DINO-WM + HWM | **89%** | **78%** | **61%** |

**关键观察**：
1. **d=75时17%→61%**：flat planner在长horizon崩溃（compounding error + search space爆炸）
2. **RL methods崩溃更快**：HIQL从55%→20%，说明offline RL的long-horizon generalization远不如model-based planning
3. **HWM在d=25时仍有5%提升**：即使short horizon，hierarchy也有regularization效果

### 5.3 Diverse Maze OOD Generalization (Table 3)

| Method | D∈[5,8] | D∈[9,12] | D∈[13,16] |
|--------|---------|----------|-----------|
| PLDM | 100% | 63% | 44% |
| PLDM + HWM | 100% | **95%** | **83%** |

**关键观察**：
1. **Test layout unseen**：maze layout在训练时没见过，pure zero-shot generalization
2. **D∈[13,16]时44%→83%**：hard regime下hierarchy收益最大
3. **RL methods在OOD下完全崩溃**：HILP只有10%，因为学的policy overfit训练layout

### 5.4 Compute Efficiency (Figure 5)

Push-T d=50的Pareto frontier：
- Flat planner在相同success rate下需要3× compute
- Diverse Maze D∈[13,16]：4× compute reduction

**为什么hierarchy更高效**？
1. High-level搜索空间是 $\mathbb{R}^4$（macro-action），远小于low-level的 $\mathbb{R}^{10}$（Push-T primitive action chunk）
2. High-level prediction只需1次unroll，flat需要15次（$d=75$时）
3. Low-level只在短horizon上精细搜索

---

## 6. Ablation Studies的核心洞察

### 6.1 Latent Action vs Delta Pose (Table 4)

| High-Level Action | Action Cos ↑ | Action L1 ↓ |
|-------------------|--------------|-------------|
| Delta Pose | 0.80 ± 0.02 | 0.088 ± 0.005 |
| Latent Action | **0.88 ± 0.03** | **0.080 ± 0.002** |

**为什么latent action更好**？考虑一个pick-&-place trajectory：
- Waypoint 1 → Waypoint 2：先向上移动（抓取），再水平移动
- Delta pose只编码净位移（水平），丢失了"先上后横"的结构
- Latent action可以compactly encode整个non-greedy action subsequence的结构

### 6.2 Prediction Error vs Horizon (Figure 6)

这是最有intuition的一张图：
- **Short horizon (≤1s)**：Low-level model更准（每步精细）
- **Long horizon (≥1.5s)**：High-level model的single-step prediction更准
- **Crossover around 1s**：这是hierarchical decomposition的natural timescale

**数学解释**：假设low-level每步误差 $\epsilon$，autoregressive rollout $T$ 步累积误差 $\sim T\epsilon$（线性）或 $T^2\epsilon^2$（平方）。High-level single-step prediction误差为 $\epsilon_{high}$，当 $T\epsilon > \epsilon_{high}$ 时hierarchy占优。

### 6.3 Latent Action Dimension Trade-off (Figure 7)

这是最subtle的发现：
- $d < 4$：latent space capacity不够，high-level planner产生invalid plans
- $d = 4$：sweet spot，subgoals既valid又reachable
- $d > 4$：subgoals变得too expressive，需要non-greedy low-level execution，low-level planner跟不上

**关键insight**：
> Restricting the latent-action dimensionality biases the planner toward proposing subgoals that are achievable with greedy behavior.

这其实是一种 **implicit regularization**——通过限制high-level的表达能力，迫使它产生"简单"的subgoals，让low-level更容易reach。

**重建质量与性能解耦**：低维latent的重建更blurry（L1 error更高），但语义结构（contact events, motion direction）保留，足够planning用。

---

## 7. 与Prior Work的关系

### 7.1 vs. Director (Hafner et al. 2022)

| 维度 | Director | HWM |
|------|----------|-----|
| Training | Online RL with reward | Offline, reward-free |
| Hierarchy interface | Goal-conditioned policy | Latent subgoal matching |
| Zero-shot | No | Yes |
| Latent space | Hierarchical latents (separate) | Shared latent space |

Director需要训练high-level和low-level policy，HWM直接用pretrained world models做planning，无需policy training。

### 7.2 vs. THICK (Gumbsch et al. 2023)

THICK也做hierarchical latent world models from pixels，但：
1. THICK用reward-predictive model-based RL
2. THICK的evaluation在aligned train/test distribution
3. HWM完全reward-free，task-agnostic

### 7.3 vs. Classical Hierarchical MPC

CAVIN (Fang et al. 2019), LAT (Li et al. 2021)等：
1. 都在structured state space，不能处理pixels
2. HWM在learned latent space，支持high-dim observations
3. HWM用learned macro-actions，不需要hand-designed abstraction

---

## 8. 我的Critical Analysis

### 8.1 Strengths

1. **Clean formulation**：shared latent space让hierarchy自然耦合，不需要额外的interface learning（inverse model, skill policy等）
2. **Model-agnostic**：作为plugin应用到VJEPA2-AC, DINO-WM, PLDM三个完全不同的backbone
3. **Real robot results**：0%→70%在non-greedy pick-&-place是convincing的突破
4. **Compute efficiency**：不仅是performance提升，还3-4× compute reduction

### 8.2 Potential Limitations

1. **Top-down only**：没有feedback from low-level back to high-level。如果low-level执行失败，high-level不知道。Conclusion里作者也承认这点。

2. **Waypoint selection是random的**：训练时 $N=3$ waypoints随机选，可能不是optimal decomposition。Adaptive waypoint selection可能更好。

3. **Latent action dimension是hyperparameter**：$d=4$是empirical sweet spot，但理论上没有principled way选择。

4. **Long-horizon仍有上限**：D∈[13,16]时83%不是100%，说明hierarchy也有极限。可能需要3层或更多层hierarchy。

5. **No uncertainty quantification**：planning时假设model是deterministic的，没有epistemic uncertainty估计。在safety-critical场景可能有风险。

### 8.3 Future Directions

1. **Multi-level hierarchy**：当前只有2层，理论上可以递归扩展到 $K$ 层，每层timescale指数增长
2. **Closed-loop hierarchy**：Low-level执行结果feedback给high-level做replan
3. **Learned waypoint selection**：用meta-learning或curriculum learning自动选择optimal waypoints
4. **Diffusion-based planning**：把CEM替换成diffusion planner（如Diffuser, Decision Diffuser），可能更sample-efficient

---

## 9. Implementation Details值得注意的点

### 9.1 High-level model capacity更大

Push-T实验中（Table 7）：
- Low-level: 25M params, 6 layers, 384 dim
- High-level: **75M params**, 10 layers, **768 dim**

High-level需要更多capacity来建模long-horizon dynamics的complexity。

### 9.2 Action Encoder架构

Transformer-based，CLS token + MLP：
- 输入：variable-length primitive action sequence
- CLS token聚合整个sequence的信息
- MLP head project到latent action space

这其实是借鉴了BERT-style的sequence encoding。

### 9.3 VICReg in PLDM

PLDM用VICReg (Bardes et al. 2021)防止representation collapse：
- Variance: 每个维度variance > threshold
- Invariance: augmented views有similar representations
- Covariance: 不同维度decorrelated

参数 $\alpha=29.4, \beta=17.9, \lambda=2.80, \omega=4.81$（Table 8）。

参考：https://arxiv.org/abs/2105.04906

### 9.4 Decoder只用于visualization

Paper强调decoder reconstructions (Figure 1, 4)只用于human interpretability，planning和training都不用。这是JEPA philosophy的体现——predict in latent space, don't generate pixels。

---

## 10. Connection to Broader Themes

### 10.1 JEPA Philosophy

HWM完全符合LeCun的JEPA framework：
1. Predict in latent space, not pixel space
2. No generative decoder needed
3. Self-supervised learning
4. Hierarchical abstraction

参考LeCun的position paper: https://openreview.net/forum?id=BZ5a1r-kVsf

### 10.2 Test-time Compute Scaling

HWM展示了test-time compute的value——用更多inference-time planning换更好的performance。这和recent trend一致：
- OpenAI o1的test-time compute
- SNELL et al. "Scaling LLM Test-Time Compute Optimally"
- HWM提供了一个robotic control领域的analog

### 10.3 World Models as Foundation Models

VJEPA2-AC trained on 130 hours DROID + RoboSet，然后zero-shot deploy到新task。这是robotic foundation model的方向：
- Pretrain on large-scale offline data
- No task-specific finetuning
- Planning at test time enables zero-shot

参考：
- DROID dataset: https://arxiv.org/abs/2403.12945
- RoboSet: https://arxiv.org/abs/2310.21644

---

## 11. Open Questions & My Speculations

1. **为什么 $H=2$ 就够了？** Franka实验中high-level horizon只有2。可能因为robot task的intrinsic hierarchy不深。Navigation可能需要更深hierarchy。

2. **Can we use diffusion planner instead of CEM?** CEM是gradient-free的，可能sample-inefficient。Diffusion planner（如Planning with Diffusion, Janner et al. 2022）可能更natural fit for continuous action spaces。

3. **How does this relate to MuZero-style MCTS?** MuZero在discrete action space用MCTS，HWM在continuous space用CEM。可能可以用implicit policy + MCTS结合。

4. **Is shared encoder really optimal?** High-level和low-level可能需要different representation granularities。Shared encoder可能限制high-level的abstraction能力。

5. **What about stochastic environments?** 当前formulation假设deterministic model。Stochastic setting需要distributional prediction + risk-sensitive planning。

---

## 12. Summary: Building Your Intuition

HWM的核心intuition可以归纳为：

**"在不同时间尺度上做规划，让每个planner只处理它能handle的复杂度"**

- **High-level planner**：处理"what to achieve"——在abstract latent space规划coarse trajectory到goal
- **Low-level planner**：处理"how to achieve"——在fine-grained latent space规划primitive actions到subgoal

**为什么work**？
1. **Error decomposition**：high-level的single-step prediction避免了low-level的长程rollout error累积
2. **Search space reduction**：macro-action的 $\mathbb{R}^4$ 远小于primitive action的 $\mathbb{R}^{10+}$
3. **Shared latent space**：自然interface，无需额外的skill learning或inverse model
4. **Modularity**：可以plug into任何latent world model

**为什么non-greedy task特别需要hierarchy**？Flat planner每步greedy minimize到goal的距离，遇到需要"先远离goal再approach"的任务就失败。HWM的high-level planner可以规划出non-greedy的subgoal sequence，low-level只负责reach下一个subgoal，不需要考虑全局non-greedy structure。

这其实就是经典AI中的means-ends analysis的modern realization——把大问题分解成小问题，每个小问题greedy solvable。

---

**Paper资源**：
- GitHub: https://github.com/kevinghst/HWM_PLDM
- Project page: https://kevinghst.github.io/HWM/
- VJEPA2: https://arxiv.org/abs/2506.09985
- PLDM: https://arxiv.org/abs/2502.14819
- DINO-WM: https://arxiv.org/abs/2411.04983
- CEM: https://link.springer.com/book/10.1007/978-0-387-30140-5
- Options framework: https://arxiv.org/abs/cs/9905114
- VICReg: https://arxiv.org/abs/2105.04906

希望这个深度解析能帮你build intuition about why hierarchical planning in latent space is a natural and powerful abstraction for long-horizon control!
