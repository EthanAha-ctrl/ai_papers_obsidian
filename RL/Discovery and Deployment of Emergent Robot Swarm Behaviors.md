---
source_pdf: Discovery and Deployment of Emergent Robot Swarm Behaviors.pdf
paper_sha256: 424c1075a716e8563bbd8e0a6cfc60b85d2216aae42d57d2b8879f493777ff45
processed_at: '2026-08-03T22:10:16-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 Paper

## 一句话总结

给一群只有 binary sensor、只跑 if-else 四参数 controller 的便宜 robot，怎么让 computer 自己**发现**这些 robot 能搞出哪些 collective behaviors，而且**真的能 deploy 到 real robots 上跑**。

---

## 核心问题为什么 Hard

你想，一个 robot 只有一个 ToF sensor，要么看到别的 robot（h=1），要么没看到（h=0）。就这一 bit 信息。controller 就 4 个数字 `[u_v0, u_ω0, u_v1, u_ω1]`，根据这一 bit 决定 linear velocity 和 angular velocity。

**8 个这种 dumb robot 放一起**，按理说啥也干不了。但 prior work（[Gauci 2014](https://journals.sagepub.com/doi/10.1177/0278364914525244)）已经证明：这种 setup 能 produce aggregation（聚一堆）、cyclic pursuit（围圈转）、dispersal（散开）这些 emergent behaviors。

问题来了：**complete set of possible behaviors 是什么？** 你不知道 4D controller space 里哪个点对应哪个 behavior。你不能 grid search（4D 太大），而且你连 "behavior 怎么 measure" 都没定义好。

Prior work 的 two big problems：

**Problem 1: 怎么 represent behavior？**

Brown 2018 用 5 个 hand-crafted metrics：average speed、angular momentum、radial variance、scatter、group rotation。每跑一个 controller，把 600 帧的 robot trajectories 算成这 5 个数字。然后 novelty search 在这 5D space 里找 novel behaviors。

问题：**hand-crafted metrics 有 bias**。作者实测发现，hand-crafted representation 把 dispersal 和 random behavior 搞混 27% 的时间。原因很 intuitive——dispersal 就是 robot 散开，random 也是乱跑，你 measure "scatter" 这个 metric 区分不开。结果就是 novelty search **从来没找到过 dispersal**（Table 1: 0.0 ± 0.0），因为它在这个 5D space 里和 random 太像了。

**Problem 2: 怎么 deploy 到 real robots？**

Prior work 只在 simulation 里 discover behaviors。但 sim ≠ real。Collision friction、sensing noise、wheel slip 这些东西 simulator 建模不准。结果是 sim 里发现的 "behavior" 到 real robot 上根本跑不出来。

作者做了个 ablation：用 default simulator（没 RSRS）discover behaviors，然后 deploy 到 real robots。18 个 non-random controllers，只有 22% one-shot 成功。剩下 78% 都 fail。

---

## 方法一：SimCLR 学 behavior representation

**核心 idea**：与其 hand-craft 5 个 metrics，不如让 neural network 自己从 video 学。

流程：
1. 在 simulator 里 random sample 6000 个 controllers
2. 每个 controller 跑 600 timesteps，render 成 grayscale video
3. 从最后 300 timesteps 均匀取 3 帧，resize 到 64×64，组成 `(3, 64, 64)` tensor
4. 用 SimCLR 在这 6000 个 videos 上 train encoder

SimCLR 怎么 work：
- 拿一个 video `x`，apply 两个 random augmentations（random crop + random rotation + horizontal flip），得到两个 views `x̃_i` 和 `x̃_j`
- 两个 views 都过 ResNet18 encoder + projection head，得到 latent vectors `z_i` 和 `z_j`
- Loss 让 `z_i` 和 `z_j` 在 latent space 里 close，让它们和 batch 里其他 2N-2 个 views far

公式（Equation 1）：
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\sin(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\sin(z_i, z_k)/\tau)}$$

直觉讲：batch size 1000，所以有 2000 个 views。对每个 view `z_i`，它的 positive pair `z_j` 是同一个 video 的另一个 augmentation。Loss 就是 softmax 形式的 "从 1999 个 candidates 里选对那个 positive"。

为什么 random rotation 作为 augmentation 很关键：cyclic pursuit 转 90° 还是 cyclic pursuit。SimCLR 自动学会 rotation-invariant 的 representation，这是 hand-crafted metrics 不 explicit handle 的。

为什么 random crop 重要：让 encoder 不依赖 robot 在 frame 里的 absolute position，而是关注 **relative pattern**。

Train 完之后，encoder `φ` 把任何 video map 到 512D latent vector。这个 vector 就是 behavior representation。

**结果**（Figure 4 confusion matrix）：
- Cyclic pursuit：self-supervised 比 hand-crafted 准 **16%**
- Aggregation：持平
- Dispersal：self-supervised 准 **5%**，而且 random confusion 从 27% 降到 19%
- 最关键的 downstream effect：novelty search 现在**能找到 dispersal 了**（Table 1: 2.0 ± 0.57 vs hand-crafted 的 0.0）

---

## 方法二：Novelty Search 发现 Behaviors

有了 encoder `φ`，现在搜 controller space。

genome `g = [u_v0, u_ω0, u_v1, u_ω1]`，4D vector，每个 dimension bounded by robot velocity limits。

fitness function 就是 **novelty**（Equation 2）：
$$\text{Novelty}(b, B) = \frac{1}{k} \sum_{i=0}^{k} \text{dist}(b, B_i)$$

`b` 是新 controller 的 behavior embedding（512D），`B` 是之前见过的所有 embeddings 的 buffer，`k=15` nearest neighbors。

novelty 高 = 离之前见过的 behaviors 远 = 值得 explore。

**Evolutionary search**:
- 50 个 random controllers 起步
- 100 generations，每代 50 个
- crossover rate 0.7，mutation rate 0.15
- 每个 controller 跑 simulation → video → encoder → 512D embedding → 算 novelty → 存进 buffer
- 最后 buffer 有 5000 个 controllers
- k-Medoids clustering，k=10，返回 10 个 representative behaviors

**为什么用 novelty search 不用 gradient descent？**

两个原因：
1. Objective non-stationary：buffer 在长，同一个 controller novelty 一直在变
2. Simulation 不可微分（collision、discretization），gradient 无法 backprop through simulator

Evolutionary search 是 gradient-free 的，天然适合这种 black-box optimization。

---

## 方法三：RSRS 关闭 Sim2Real Gap

这是 paper 最 insightful 的部分。

传统 sim2real 思路：**把 simulator 做得更 realistic**。精确 model collision friction、sensing noise、wheel slip。问题是要 run 1000+ controllers × 600 timesteps × 8 robots，high-fidelity simulator 太慢。而且即使 realistic，仍会 miss 某些 real-world phenomena。

RSRS 的反直觉思路：**让 simulated robots 比 real robots less capable**。

具体怎么 less capable：

**Sensing capability underestimate**：
- Real HeRo+ 的 ToF sensor 能 detect 2m 距离，且高速下也 reliable
- 但 simulator 里把 max velocity 限制到 9 cm/s（real max 是 20 cm/s），1.6 rad/s（real max 3 rad/s）
- Intuition：低速 = 有充足 sensing 反应时间 = behavior 更 robust。作者发现 real robot 在 20 cm/s 时 sensing 来不及反应，9 cm/s 时 reliable。与其在 simulator 里 model "velocity-dependent sensing reliability"，直接 cap velocity 更 cheap

**Actuation capability underestimate**：
- Real HeRo+ 有 bump shield，collision 不卡 wheel
- 但 real collision 还是有 friction，robots 不能 slide past each other
- Simulator 里**手动 tune friction coefficient**，让 collision 行为 match real world。Default frictionless simulator 会 discover "wall-following"（robot 沿 wall slide），但 real robot 因为 friction 跑不出来

**Physical geometry underestimate**：
- Simulator 不 model bump shield 的 exact geometry
- 但通过 hardware 加 bump shield，real robots 能 collide without 损坏

**Hardware upgrade vs simulator upgrade 的决策树**：
- 如果 upgrade hardware 更便宜 → upgrade hardware（bump shield、ToF sensor、encoder reposition）
- Else → make simulator more realistic（tune friction、cap velocity）

**RSRS ablation 结果**（Table 2）：
- No RSRS: 22% one-shot deploy success
- With RSRS: **70% one-shot deploy success**

最关键发现：no-RSRS simulator discover 了 milling 和 wall-following 两个**额外 behaviors**，但这两个在 real robot 上**完全跑不出来**。它们是 simulator artifact——simulator 的 simplification 创造了 real capability 无法 achieve 的 phantom behaviors。

RSRS 通过 conservative under-modeling，simulator 里就**不会 discover** 这些 phantom behaviors，只 discover 真正 deployable 的。

---

## 整体 Pipeline

```
Real HeRo+ robots
    ↓ (measure dynamics: collision friction, sensing range, velocity limits)
RSRS-tuned lightweight simulator
    ↓ (random sample 6000 controllers → 6000 videos)
SimCLR encoder (ResNet18, 512D latent)
    ↓ (novelty search, 100 generations, tournament GA)
5000 controllers in buffer
    ↓ (k-Medoids, k=10)
10 representative behaviors
    ↓ (filter out random behaviors, deploy on real robots)
Real-world emergent behaviors: aggregation, cyclic pursuit, dispersal
```

整个 pipeline **zero human feedback**。Hand-crafted metrics 不需要了，sim2real gap 通过 RSRS hardware + simulator co-design 关闭。

---

## 为什么这个 Work 让人 Excited

**1. Minimalism wins**

4 个 scalar parameters + 1 bit sensor + ResNet18 + novelty search + 8 个 $80 robot = 自动发现 3 种 deployable emergent behaviors。

No LLM、no transformer、no reinforcement learning、no 100 万 step training。纯 contrastive learning + evolutionary search 在小 scale problem 上 work 得很漂亮。

**2. Representation learning 替代 hand-crafted features**

这个 pattern 在 ML 里反复出现：hand-crafted features 总有 blind spot。SimCLR 学到的 512D representation 自动 capture 人类没想到的 behavioral features。Dispersal 能被发现就是最好的证据。

**3. Conservative simulator design 的 wisdom**

"Make simulated robots less capable than they actually are" 这个 insight 通用性很强。不只 swarm robotics，任何 sim2real 问题都可以借鉴。High-fidelity simulation 是 expensive 且 incomplete 的 trap；conservative under-modeling 给 discovered behaviors 自带 safety margin。

**4. Emergent behavior 的 deep lesson**

Individual robot controller 只有 4 个数字，但 collective behavior space 是 rich 的。这说明 **behavior 不在 individual level，在 interaction level**。Representation learning 的 job 就是 capture 这种 interaction pattern。Video input 自然适合——video 里你能看到 N 个 robots 的 spatial relationship 随时间 evolve。

---

## Karpathy 视角的 Connection

这 paper 的 philosophy 和你一直 push 的 minimalism 完全 align：

- **micrograd**：reverse-mode autodiff 100 行 Python 就能 understand backprop
- **nanoGPT**：300 行训练 GPT 级 transformer，去掉所有 framework bloat
- 这篇 paper：4-parameter controller + SimCLR + novelty search，发现 emergent behaviors

Less is more 的 pattern 在这里很 clear。作者没有上 PPO、没有上 diffusion model、没有上 LLM-as-reward。就是 contrastive learning + evolutionary search 在 64×64 grayscale videos 上 work。

另外，self-supervised representation learning + downstream non-differentiable optimization 这个 pattern 很 powerful。Representation learning 提供 differentiable 的 "behavior understanding"，novelty search 在 non-differentiable 的 simulator 上做 search。两者分工明确。

如果你要做 nanoSwarm 之类的 educational project，这篇 paper 是很好的 reference。6000 videos、100 generations、ResNet18，单 GPU 几小时能跑完。Hardware 8 个 robot $640，arena 几十美元。完全在 hobbyist 可及范围。

### References

- [SimCLR paper](https://arxiv.org/abs/2002.05709)
- [Novelty Search original](https://direct.mit.edu/evco/article-abstract/19/2/189/8277/Abandoning-Objectives-Evolution-Through-the-Search)
- [RSRS paper](https://arxiv.org/abs/2301.09018)
- [HeRo 2.0 robot](https://link.springer.com/article/10.1007/s10514-023-10100-0)
- [Gauci 2014 computation-free swarms](https://journals.sagepub.com/doi/10.1177/0278364914525244)
- [Brown 2018 behavior discovery](https://link.springer.com/chapter/10.1007/978-3-319-73008-0_31)
- [Mattson & Brown 2023 human feedback](https://dl.acm.org/doi/10.1145/3583131.3590443)
- [VL53L1X ToF sensor](https://www.pololu.com/product/3415)
- [ResNet paper](https://arxiv.org/abs/1512.03385)

---

# Discovery and Deployment of Emergent Robot Swarm Behaviors 论文深度解析

## 1. Paper Overview & Motivation

这篇paper由 University of Utah 的 Daniel S. Brown 团队联合 George Mason University、University of Hawaii 发表在 AAMAS 2025。核心研究问题是: **给定 N 个 limited-capability robots, 如何自动 discover 并 deploy emergent behaviors?**

传统 swarm robotics 研究 focused on **optimizing** controllers 去实现特定预设 behavior (aggregation, circle formation, milling 等)。而这篇 paper tackle 一个更 **open-ended** 的问题: 在没有 prior knowledge 的情况下, **explore** 整个可能的 behavior space。这与 Lehman & Stanley 的 [Novelty Search](https://direct.mit.edu/evco/article-abstract/19/2/189/8277/Abandoning-Objectives-Evolution-Through-the-Search) 哲学一脉相承——"abandoning objectives"。

Prior work (Brown et al. 2018, Mattson & Brown 2023) 有两个 fundamental limitations:
1. 依赖 **hand-crafted behavior metrics** (average speed, angular momentum, radial variance, scatter, group rotation)
2. 仅在 **simulation** 中 discover behaviors, 无法 deploy 到 real robots

这篇 paper 的三个 main contributions:
- Self-supervised representation learning 基于 [SimCLR](https://arxiv.org/abs/2002.05709)
- HeRo+ hardware 改进 (open-source, ~$80)
- 第一次实现 emergent behavior discovery 的 real-world deployment, 通过 [RSRS](https://arxiv.org/abs/2301.09018) (Reality-to-Simulation-to-Reality for Swarms)

---

## 2. Problem Formulation

Robot 定义为 capability model $C = \langle S, M, A \rangle$:
- $S$: sensing capabilities
- $M$: memory capabilities  
- $A$: actuation capabilities

核心 research question: **Given N robots with capabilities C, what is the complete set of emergent behaviors that can be deployed on these robots?**

这个问题的数学 formalization:
- $\mathcal{U}(C)$: controller space (所有 possible controllers)
- $\mathcal{E}$: environment space
- $\mathcal{B}$: behavior space (隐式, 不可直接 access)
- $\phi: \mathcal{U}(C) \times \mathcal{E} \to \mathcal{B}$: behavior map, 输入 controller 和 environment, 输出 behavior representation

**关键 insight**: behavior space $\mathcal{B}$ 不可直接 access, 只能通过 sampling controllers → simulate → observe video → infer behavior。这就是为什么需要 representation learning——learn 一个 implicit 的 $\phi$。

---

## 3. Methods 详解

### 3.1 Self-Supervised Representation Learning (SimCLR)

#### 3.1.1 SimCLR Framework 直觉

[SimCLR](https://arxiv.org/abs/2002.05709) (Simple Framework for Contrastive Learning of Visual Representations) 由 Chen et al. 2020 提出, 属于 contrastive learning 家族。核心 idea: 同一个 input 的两个 augmented views 应该在 latent space 中 **close**, 不同 inputs 应该 **far**。

为什么 contrastive learning 适合 behavior discovery?
- Swarm behaviors 是 **high-dimensional** (video sequences) 但 underlying structure 是 **low-dimensional** (few behavioral primitives)
- 无需 labels (self-supervised), 适合 open-ended discovery
- 学习的 representation 能 capture **semantic similarity**, 这正是 novelty search 需要的

#### 3.1.2 NT-Xent Loss 详解

Loss function (Equation 1):

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\sin(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\sin(z_i, z_k)/\tau)}$$

**变量详解**:
- $\mathcal{L}_{i,j}$: 第 $i$ 个样本与其 positive pair $j$ 之间的 contrastive loss
- $z_i, z_j, z_k$: latent embeddings, 分别是 augmented views $\tilde{x}_i, \tilde{x}_j, \tilde{x}_k$ 经过 encoder + projection head 后的输出向量
- $\sin(\cdot, \cdot)$: similarity function, 论文中用 cosine similarity: $\sin(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}$
- $\tau$: temperature parameter, 控制分布的 "sharpness"。小的 $\tau$ 使模型更 confident, 大的 $\tau$ 更平滑。SimCLR 原文用 $\tau = 0.5$
- $\sum_{k=1}^{2N}$: 求和范围是整个 batch 的所有 $2N$ 个 augmented views (因为每个原始样本产生 2 个 views)
- $\mathbb{1}_{[k \neq i]}$: indicator function, 当 $k \neq i$ 时为 1, 否则为 0。这避免了 self-similarity 主导 loss
- $N$: batch size (论文中 batch size = 1000 videos, 所以 $2N = 2000$ views)

**Intuition**: 这个 loss 是 [InfoNCE](https://arxiv.org/abs/1807.03748) 的变种, 可以理解为 $2N-1$ way classification:
- Query: $z_i$
- Positive: $z_j$ (应该被选中的类)
- Negatives: 所有其他 $2N - 2$ 个 views
- 目标: 让 $\sin(z_i, z_j)$ 相对于 $\sin(z_i, z_k)$ 足够大

最小化 $\mathcal{L}_{i,j}$ 等价于最大化:
$$\frac{\exp(\sin(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\sin(z_i, z_k)/\tau)}$$

即 positive pair 的 similarity 在所有 candidates 中的 softmax probability。

#### 3.1.3 Data Augmentation 策略

基于 [Chen et al. 2020](https://arxiv.org/abs/2002.05709) 的 ablation study, 这篇 paper 用:
- **Random crop**: scale range $[0.6, 1.0]$, aspect ratio 1:1
- **Horizontal flip**: probability $p = 0.5$
- **Random rotation**: $\theta \in \{0, \frac{\pi}{2}, \pi, \frac{3\pi}{2}\}$, 围绕 image center 旋转

**为什么这些 augmentations 适合 swarm videos?**
- Random crop: 模拟 camera viewpoint 变化, 让 encoder 不依赖于 absolute position
- Rotation: swarm behaviors 应该是 **rotation-invariant** (cyclic pursuit 旋转 90° 仍是 cyclic pursuit)
- Horizontal flip: 让 encoder 不依赖 chirality

#### 3.1.4 Architecture

- **Backbone**: ResNet18 (pretrained), 参见 [He et al. 2016](https://arxiv.org/abs/1512.03385)
- **Final output size**: 修改为 128
- **Encoder**: ResNet18 除最后一层外的所有层, 输出 latent embedding size 512
- **Projection head**: 2-layer MLP, 把 512-dim embedding project 到 contrastive loss 应用 space
- **下游使用**: 只用 encoder 部分 (不用 projection head), 这是 SimCLR 推荐做法

**Video representation**: 每个 video 被 subsample 成 size $(3, 64, 64)$ 的 tensor:
- Channel dimension (3): 3 张 grayscale images, 均匀取自最后 300 timesteps
- Spatial dimension (64, 64): 从 513x426 resize 而来

这个设计选择 **captures final converged emergent behavior**——作者 qualitatively 发现最后 300 timesteps 足以 represent converged behavior。

### 3.2 Behavior Discovery via Novelty Search

#### 3.2.1 Novelty 定义

Novelty function (Equation 2):

$$\text{Novelty}(b, B) = \frac{1}{k} \sum_{i=0}^{k} \text{dist}(b, B_i)$$

**变量详解**:
- $b \in \mathbb{R}^d$: 新观察到的 behavior representation, $d$ 是 latent space 维度 (这里是 512)
- $B$: dynamic buffer, 存储之前观察到的所有 representations
- $k$: nearest neighbors 数量, 论文中 $k = 15$
- $B_i$: $b$ 在 $B$ 中的第 $i$ 个 nearest neighbor
- $\text{dist}(\cdot, \cdot)$: distance function, 在 latent space 中可以用 cosine distance 或 L2 distance

**Intuition**: novelty 是 $b$ 到其 $k$ nearest neighbors 的平均距离。如果 $b$ 远离之前见过的所有 behaviors, 它的 novelty 就高。这是一种 **density estimation** 的简化形式。

#### 3.2.2 Evolutionary Optimization

优化目标 (Equation 3):

$$\max_{g \in \mathcal{U}(C)} \text{Novelty}(\phi(S(g)), B)$$

**变量详解**:
- $g$: genome, 即一个 parameterized swarm controller, 在这篇 paper 中是 4 维向量
- $\mathcal{U}(C)$: controller space, 受 robot capability constraints 限制
- $S(g)$: simulation function, 输入 controller $g$, 返回 behavior video $x$
- $\phi$: trained SimCLR encoder, 把 video $x$ map 到 latent representation
- $B$: novelty buffer

**重要 property**: 这个 objective 是 **non-stationary** 的。同一个 $g$ 在早期 generation 有高 novelty, 但在后期 generation 由于 buffer 已经 populated, 它的 novelty 会降低。这迫使 algorithm 不断 explore 新 regions。

#### 3.2.3 Tournament-Style Genetic Algorithm

基于 [Mattson & Brown 2023](https://dl.acm.org/doi/10.1145/3583131.3590443):
- **Initial population**: 50 controllers 随机 sampled from controller space
- **Generations**: 100 generations, 每代 population size 50
- **Crossover rate**: 0.7
- **Mutation rate**: 0.15
- **Final buffer**: 5000 controllers
- **Clustering**: k-Medoids with $k = 10$, 输出 10 个 representative behaviors

### 3.3 Real2Sim2Real Simulator Design (RSRS)

#### 3.3.1 RSRS Philosophy

[RSRS](https://arxiv.org/abs/2301.09018) (Vega et al. 2023) 的核心 insight: **simulation 中的 robots 应该比 real robots less capable**。这看似 counterintuitive, 但有 deep wisdom:
- Real-world dynamics 难以精确 model (collisions, friction, sensing noise)
- 通过 exaggerating uncertainty, 我们让 simulator 不去 exploit 这些难建模的 dynamics
- 结果: simulator 中 "work" 的 behaviors 更 robust, 更容易 transfer

**Example**: "robots cannot reliably slide past each other on head-on collision" → 在 simulator 中提高 friction coefficient, 让 simulator 不会发现 exploit collision sliding 的 behaviors。

#### 3.3.2 RSRS 4-Step Process

1. **Measure** capabilities and dynamics of real-world robots
2. **Implement** measurement data into simulator
3. **Run experiments** in simulation, modify robots/simulator as needed
4. **Perform experiments** on real robots, modify robots/simulator as needed

步骤 3 和 4 涉及 iterative refinement, 遵循简单 if-else rule:
- If 升级 robot hardware 比 modify simulator 更 cheap → upgrade robots
- Else → make simulator more realistic + 更多 real-world measurements

---

## 4. HeRo+ Hardware Improvements

基于 [HeRo 2.0](https://link.springer.com/article/10.1007/s10514-023-10100-0) (Rezeck et al. 2023), 作者做了三项 hardware 升级:

### 4.1 Bump Shield
**Problem**: chassis-to-wheel collisions 导致 servo motor 受力, 使 robot halt, 形成 pile-up。Cyclic pursuit 等 behavior 对此特别 sensitive。

**Solution**: 3D-printed bump shield, 让 robots collide without actuation faults。这是 RSRS "upgrade hardware" 路径的典型例子——比 modeling 不同 collision types 更 cheap。

### 4.2 Time-of-Flight (ToF) Sensing
**Problem**: 原始 IR sensor 检测距离 < 25cm, 25cm 处 false negative rate 50%。

**Solution**: VL53L1X laser-ranging ToF sensor, 检测距离 2m, 低速下几乎无 false negatives。这是 sensing reliability 对 behavior formation 至关重要的例子。

### 4.3 Encoder Feedback
**Problem**: gear-driven encoders 频繁 reading errors, 影响 low-level PID control。

**Solution**: 把 encoder 移到 wheel 外, 用 directly-driven shaft。减少了 error frequency。

**总成本**: ~$80 USD per robot, 11 robot fleet (8 用于 study)。

---

## 5. Controller & Simulator Details

### 5.1 Controller Representation (Equation 4)

$$
(v_{i,t}, \omega_{i,t}) = \begin{cases} 
(u_{v,0}, u_{\omega,0}) & \text{if } h_{i,t} = 0 \\
(u_{v,1}, u_{\omega,1}) & \text{otherwise}
\end{cases}
$$

**变量详解**:
- $v_{i,t}$: 第 $i$ 个 robot 在 time $t$ 的 forward velocity
- $\omega_{i,t}$: 第 $i$ 个 robot 在 time $t$ 的 angular velocity
- $h_{i,t} \in \{0, 1\}$: 第 $i$ 个 robot 在 time $t$ 的 binary line-of-sight observation (ToF 检测到其他 robot 为 1, 否则 0)
- $u_{v,0}, u_{\omega,0}$: 当 $h_{i,t} = 0$ (没有检测到其他 robot) 时的 forward/angular velocity commands
- $u_{v,1}, u_{\omega,1}$: 当 $h_{i,t} = 1$ (检测到其他 robot) 时的 forward/angular velocity commands

**整个 controller**: 4-tuple $[u_{v,0}, u_{\omega,0}, u_{v,1}, u_{\omega,1}] \in \mathcal{U}(C)$

**Beautiful insight**: 4 个 scalar values 控制 N 个 robots 的任意时间 horizon 的 collective behavior。这是 [Gauci et al. 2014](https://journals.sagepub.com/doi/10.1177/0278364914525244) "computation-free swarms" 的极致简化——每个 robot 只有一个 binary sensor 和一个 4-parameter if-else controller, 但 collective behaviors 却很 rich。

### 5.2 Simulator Parameters

| Parameter | Real Robot Max | Simulated Value | Rationale |
|-----------|----------------|-----------------|-----------|
| Linear velocity | 20 cm/s | 9 cm/s | 给 sensing 足够反应时间 |
| Angular velocity | 3 rad/s | 1.6 rad/s | 同上 |
| Collision friction | (variable) | manually tuned | 防止 exploit frictionless sliding |
| Arena size | - | 170×142 cm | - |
| Wall height | - | 5 cm | 低于 ToF sensor, 防止检测到 wall |
| Init grid | - | 3×4 = 12 points | 8 robots 随机分配 |
| Init orientation | - | $[0, 2\pi]$ | 随机 |
| Timesteps | - | 600 | $\Delta t = 0.1$, total 60s |

---

## 6. Experimental Results 深度分析

### 6.1 Representation Confusion Matrices (Figure 4)

Evaluation 用 500 labeled test videos, 用 triplet $(a, p, n)$:
- $a$: anchor
- $p$: positive (same label as $a$)
- $n$: negative (different label)

Test: $\text{dist}(a, p) < \text{dist}(a, n)$?

**Quantitative results** (diagonal = within-class accuracy):
- **Cyclic Pursuit**: Self-supervised 比 hand-crafted 高 **+16%**
- **Aggregation**: 相等 **+0%**
- **Dispersal**: Self-supervised 高 **+5%**
- **Dispersal vs Random confusion**: hand-crafted 27% 混淆, learned 只 8% better

**Key insight**: hand-crafted metrics 设计 for specific behaviors, 在它设计 target 的 behaviors (aggregation) 上表现 fine, 但对 "unexpected" behaviors (dispersal) 区分能力差。Self-supervised learning 更 **general**。

### 6.2 t-SNE Visualization (Figure 5)

- **Hand-crafted** (5D): 能 distinguish cyclic pursuit 和 aggregation from random, 但 dispersal 和 random 混在一起
- **Self-supervised** (512D, projected to 2D): 所有 3 个 non-random behaviors 都能 distinguish from random, dispersal 更 distinctly embedded

### 6.3 Discovery Frequency (Table 1)

| Behavior | Hand-Crafted | Triplet Learning | Self-Supervised |
|----------|--------------|------------------|-----------------|
| Aggregation | 4.33 ± 0.66 | 3.66 ± 0.66 | 2.0 ± 0.0 |
| Cyclic Pursuit | 0.66 ± 0.33 | 2.0 ± 0.0 | 1.0 ± 0.57 |
| Dispersal | 0.0 ± 0.0 | 0.0 ± 0.0 | **2.0 ± 0.57** |
| Random | 5.0 ± 1.0 | 4.33 ± 0.66 | 5.0 ± 1.0 |
| **Total Unique** | 1.6 ± 0.33 | 2.0 ± 0.0 | **2.6 ± 0.33** |

**Critical observation**: 
- Hand-crafted **从未** discover dispersal (0.0 ± 0.0)——因为它的 representation 把 dispersal 和 random 混淆, novelty search 找不到 dispersal 的 region
- Self-supervised **最 diverse** (2.6 unique behaviors on average)
- Triplet learning (从 [Mattson & Brown 2023](https://dl.acm.org/doi/10.1145/3583131.3590443)) 仍需 human feedback, 但 self-supervised 不需要

### 6.4 RSRS Ablation (Table 2)

| Behavior | No RSRS T1 | No RSRS T2 | No RSRS T3 | RSRS T1 | RSRS T2 | RSRS T3 |
|----------|------------|------------|------------|---------|---------|---------|
| Aggregation | ✓ | — | ✗ | ✓ | ✓ | ✓ |
| Cyclic Pursuit | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Dispersal | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Milling | ✗ | ✓ | ✗ | — | — | — |
| Wall-Following | ✗ | ✗ | ✗ | — | — | — |

Legend: ✓ = one-shot deploy; ✓ (parenthesized) = 2-3 attempts; ✗ = discovered but failed; — = not discovered

**Quantitative summary**:
- **No RSRS**: 22% one-shot success, 27% three-shot success (out of 18 non-random controllers)
- **RSRS**: 70% one-shot success, 90% three-shot success (out of 30 non-random controllers)
- **Improvement**: +48% one-shot, +63% three-shot

### 6.5 Critical Insight: "Discovered" ≠ "Deployable"

**Milling** 和 **Wall-Following** 是 simulator artifacts:
- **Wall-Following**: Real-world friction 阻止 robots slide along walls
- **Milling**: 导致 head-on collisions, real robots 无法 slip past

这揭示一个 deep issue: **behavior discovery without sim2real considerations 会产生 hallucinated behaviors**。Simulator 的 simplifications (frictionless collisions, ideal sensing) 创造了 real world 无法 achieve 的 "phantom" behaviors。RSRS 通过 conservative under-modeling 避免 discover 这些 phantom behaviors。

---

## 7. Intuition Building: 为什么这套方法 work?

### 7.1 为什么 SimCLR > Hand-Crafted?

Hand-crafted metrics 是 **top-down** 设计: researcher 先假设什么 important (speed, momentum, scatter), 然后 measure。问题:
- 对 unknown behaviors, 可能 measure 错 things
- 5D representation capacity 有限

SimCLR 是 **bottom-up** 学习: 从 data 中 learn 什么 distinguishes behaviors。优势:
- 512D latent space capacity 大
- 自动 capture 人类没考虑到的 features (例如 robot 之间的 spacing pattern, temporal dynamics)
- Augmentation invariance (rotation, crop) 让 representation 更 robust

### 7.2 为什么 RSRS > High-Fidelity Simulator?

High-fidelity simulator (e.g., [PyBullet](https://pybullet.org/), [Mujoco](https://mujoco.org/)) 的问题:
- 1000+ controllers × 600 timesteps × N robots = 太慢
- 即使用 high-fidelity, 仍有 sim2real gap (modeling 永远不完全)

RSRS 的智慧:
- Lightweight simulator (fast) + conservative dynamics (robust)
- 通过 making simulated robots **less capable**, 避免 exploit 不 realistic dynamics
- 发现的 behaviors 自带 **safety margin**

### 7.3 为什么 4-Parameter Controller 能产生 Rich Behaviors?

这看似 magic, 实际上 [Gauci et al. 2014](https://journals.sagepub.com/doi/10.1177/0278364914525244) 已经证明: binary sensing + unicycle kinematics + simple if-else controller 足以 produce aggregation, dispersion, cyclic pursuit 等。

**Intuition**: swarm behavior 是 **emergent** 的, 不在 individual robot level, 而在 interaction level。即使 individual controller 简单, N 个 robots 的 interaction 产生 high-dimensional dynamics。这就是为什么 representation learning 重要——我们需要 capture **collective** pattern, 不仅是 individual states。

### 7.4 Behavior Space 的 Topology

从 t-SNE visualization 推测 behavior space 结构:
- **Cluster centers**: aggregation, cyclic pursuit, dispersal
- **Boundaries**: regions where behaviors "transition" (e.g., between dispersal and random)
- **Phantom regions**: milling, wall-following (在 no-RSRS simulator 中 exist, 但在 real capability model 中不存在)

Novelty search 的 job: explore 这个 behavior space, find diverse cluster centers, avoid 在 phantom regions 浪费 samples。

---

## 8. Limitations & Future Directions

### 8.1 Current Limitations
- 只测试了 homogeneous swarm (8 robots, same capabilities)
- Controller space 只有 4D (简单 if-else), 更 complex controllers 未测
- 3 behaviors 成功 deploy (aggregation, cyclic pursuit, dispersal)——是否还有未发现的 deployable behaviors?
- Real-world evaluation 是 qualitative (high-level behavior match), 没有 quantitative metric

### 8.2 Future Work (Paper 提到)
- Heterogeneous swarms (different capabilities)
- Limited communication capabilities
- Environment effect on emergent behaviors
- Task-specific discovery (结合 novelty + task objective)

### 8.3 我的 Speculation (可能 hallucination)

可能 extensions:
- **更 rich controller space**: 用 neural network 作为 controller, search space 变成 weight space, 配合 [NEAT](https://arxiv.org/abs/1702.03705) 或 [CMA-ES](https://arxiv.org/abs/1604.00772) search
- **Hierarchical behavior discovery**: learn behavior primitives, 然后 search compositions
- **Multi-modal representation learning**: 除了 video, 还用 robot state trajectories (positions, velocities) 作为 input, 类似 [VLM](https://arxiv.org/abs/2103.00020) 思路
- **Language-conditioned discovery**: 用 [CLIP](https://arxiv.org/abs/2103.00020)-style training, 让 user 可以用 language query behavior space ("find me a behavior where robots form a line")
- **Real-world evolution**: 在 real robots 上直接 evolve (用 [real-world evolution](https://arxiv.org/abs/2010.05829) 思路), 避开 sim2real gap
- **World models**: 用 [Dreamer](https://arxiv.org/abs/2306.01940)-style world models 学 swarm dynamics, 在 latent space 做 behavior discovery

---

## 9. Connections to Broader ML/Robotics

### 9.1 与 Quality Diversity (QD) 的关系

Novelty search 是 QD 算法家族的一员。其他相关:
- [MAP-Elites](https://arxiv.org/abs/1504.04909): search for high-performing behaviors in each cell of behavior space
- [CMA-ME](https://arxiv.org/abs/1912.02400): 结合 CMA-ES 和 MAP-Elites
- [OpenAI ES](https://arxiv.org/abs/1703.03864): evolution strategies for RL

### 9.2 与 Sim2Real 的关系

RSRS 与其他 sim2real 方法对比:
- [Domain Randomization](https://arxiv.org/abs/1703.06907): 在 simulator 中 randomize dynamics, 让 policy robust
- [System Identification](https://arxiv.org/abs/1907.04758): learn simulator parameters from real data
- [DAgger](https://arxiv.org/abs/1011.0686): dataset aggregation, 适合 supervised settings
- [Real2Sim2Real](https://arxiv.org/abs/2011.07253): iterate between real and sim
- [Meta-Learning](https://arxiv.org/abs/1703.03400): learn to adapt at test time

RSRS 特点: 不 optimize for specific behavior, 而是 design simulator 让 discovered behaviors transferable。

### 9.3 与 Contrastive Learning 在 Robotics 的应用

SimCLR 之外, 相关 contrastive methods:
- [CLIP](https://arxiv.org/abs/2103.00020): image-language contrastive
- [R3M](https://arxiv.org/abs/2203.12601): robotic manipulation representations
- [Voltron](https://arxiv.org/abs/2302.12766): language-driven representation
- [RoboCLIP](https://arxiv.org/abs/2401.02146): video-language for robotics

这篇 paper 用 SimCLR 学 **behavior** representations, 不同于 typical robotics 用法 (学 state/action representations)。

### 9.4 与 Evolutionary Robotics

[Evolutionary Robotics](https://www.frontiersin.org/articles/10.3389/frobt.2015.00004/full) 的 survey (Doncieux et al. 2015):
- Early work: [Nolfi & Floreano](https://mitpress.mit.edu/9780262140705/evolutionary-robotics/) 教科书
- Modern: [Neuroevolution](https://arxiv.org/abs/1912.04906), [Deep Neuroevolution](https://arxiv.org/abs/1712.06567)

这篇 paper 用 **gradient-free** evolution 在 controller parameter space search, 用 **gradient-based** learning 在 representation space learn embeddings。两者 complement。

---

## 10. Reproducibility & Resources

### 10.1 Open-Source Resources
- Project webpage (mentioned in paper, 但具体 URL 未在 attachment 中明确)
- HeRo+ CAD models 和 bill of materials 在 supplementary materials
- HeRo 2.0 原始设计: [Rezeck et al. 2023](https://link.springer.com/article/10.1007/s10514-023-10100-0)
- SimCLR 原始实现: [github.com/google-research/simclr](https://github.com/google-research/simclr)
- RSRS 详细方法: [arxiv.org/abs/2301.09018](https://arxiv.org/abs/2301.09018)

### 10.2 Hardware Cost
- 每个 HeRo+ robot: ~$80 USD
- 8-robot swarm: ~$640 USD
- ToF sensor (VL53L1X): [$5-10 USD](https://www.pololu.com/product/3415)
- 3D-printed parts: 几美元 filament
- Arena (170×142cm) + 5cm walls: DIY 成本低

### 10.3 Software Stack (推断)
- ROS (centralized server for emergency stop, sync start, controller updates)
- Custom simulator (lightweight, 未指明具体 framework)
- PyTorch (SimCLR training)
- [k-Medoids](https://scikit-learn.org/stable/modules/generated/sklearn_extra.cluster.KMedoids.html) clustering (scikit-learn-extra)

---

## 11. 公式汇总 & 变量索引

### Equation 1: NT-Xent Loss
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\sin(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\sin(z_i, z_k)/\tau)}$$

| Variable | Meaning |
|----------|---------|
| $\mathcal{L}_{i,j}$ | Contrastive loss for positive pair $(i, j)$ |
| $z_i, z_j, z_k$ | Latent embeddings of augmented views |
| $\sin$ | Cosine similarity |
| $\tau$ | Temperature (default 0.5) |
| $N$ | Batch size (1000) |
| $2N$ | Total views (each sample → 2 augmented views) |
| $\mathbb{1}_{[k \neq i]}$ | Indicator excluding self |

### Equation 2: Novelty
$$\text{Novelty}(b, B) = \frac{1}{k} \sum_{i=0}^{k} \text{dist}(b, B_i)$$

| Variable | Meaning |
|----------|---------|
| $b$ | New behavior representation in $\mathbb{R}^d$ |
| $B$ | Dynamic buffer of past representations |
| $k$ | Number of nearest neighbors (15) |
| $B_i$ | $i$-th nearest neighbor of $b$ in $B$ |
| $\text{dist}$ | Distance function |

### Equation 3: Optimization Objective
$$\max_{g \in \mathcal{U}(C)} \text{Novelty}(\phi(S(g)), B)$$

| Variable | Meaning |
|----------|---------|
| $g$ | Genome = 4D controller vector |
| $\mathcal{U}(C)$ | Controller space constrained by capability $C$ |
| $S(g)$ | Simulation function returning video $x$ |
| $\phi$ | Trained SimCLR encoder |
| $B$ | Novelty buffer |

### Equation 4: Controller
$$(v_{i,t}, \omega_{i,t}) = \begin{cases} (u_{v,0}, u_{\omega,0}) & \text{if } h_{i,t} = 0 \\ (u_{v,1}, u_{\omega,1}) & \text{otherwise} \end{cases}$$

| Variable | Meaning |
|----------|---------|
| $v_{i,t}$ | Forward velocity of robot $i$ at time $t$ |
| $\omega_{i,t}$ | Angular velocity of robot $i$ at time $t$ |
| $h_{i,t}$ | Binary line-of-sight observation (0 or 1) |
| $u_{v,0}, u_{\omega,0}$ | Velocity commands when no robot detected |
| $u_{v,1}, u_{\omega,1}$ | Velocity commands when robot detected |

---

## 12. 总结

这篇 paper 是 swarm robotics, representation learning, sim2real transfer, evolutionary computation 的 **beautiful intersection**:

1. **Representation learning** (SimCLR) 替代 hand-crafted metrics, 让 behavior discovery 更 general, 更 scalable
2. **Novelty search** 在 learned latent space 中 explore, 不被 prior assumptions 限制
3. **RSRS** 通过 conservative simulator design 让 discovered behaviors 一次 transfer 到 real robots (70% one-shot success)
4. **HeRo+** 开源 hardware platform 让其他 researcher 可以 replicate 和 extend

**最深刻的 insight**: 在 open-ended discovery 问题中, **不 over-model reality** 反而更好。让 simulator 保守, 让 discovered behaviors robust。这与传统 "high-fidelity simulation is better" 的直觉相反, 体现了 evolutionary robotics 和 robust control 的智慧。

**对 Karpathy 的 relevance**: 这 paper 体现了 self-supervised learning + evolutionary search 的 synergy, 类似您在 nanoGPT 和 [micrograd](https://github.com/karpathy/micrograd) 中推崇的 "minimal but powerful" 思路。4 个 scalar parameters + binary sensor + SimCLR encoder + novelty search = rich behavior space exploration。Less is more, again。

### References

1. [SimCLR - Chen et al. 2020](https://arxiv.org/abs/2002.05709)
2. [Novelty Search - Lehman & Stanley 2011](https://direct.mit.edu/evco/article-abstract/19/2/189/8277/Abandoning-Objectives-Evolution-Through-the-Search)
3. [RSRS - Vega et al. 2023](https://arxiv.org/abs/2301.09018)
4. [HeRo 2.0 - Rezeck et al. 2023](https://link.springer.com/article/10.1007/s10514-023-10100-0)
5. [Brown et al. 2018 - Discovery of Novel Swarm Behaviors](https://link.springer.com/chapter/10.1007/978-3-319-73008-0_31)
6. [Gauci et al. 2014 - Self-organized aggregation without computation](https://journals.sagepub.com/doi/10.1177/0278364914525244)
7. [ResNet - He et al. 2016](https://arxiv.org/abs/1512.03385)
8. [MAP-Elites - Mouret & Clune 2015](https://arxiv.org/abs/1504.04909)
9. [Mattson & Brown 2023 - Human Feedback for Behavior Discovery](https://dl.acm.org/doi/10.1145/3583131.3590443)
10. [Evolutionary Robotics Survey - Doncieux et al. 2015](https://www.frontiersin.org/articles/10.3389/frobt.2015.00004/full)
11. [CLIP - Radford et al. 2021](https://arxiv.org/abs/2103.00020)
12. [Domain Randomization - OpenAI 2017](https://arxiv.org/abs/1703.06907)
13. [VL53L1X ToF Sensor](https://www.pololu.com/product/3415)
14. [PyTorch SimCLR implementation](https://github.com/google-research/simclr)
15. [Quality Diversity Overview](https://arxiv.org/abs/2107.04733)
