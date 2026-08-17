---
source_pdf: DINO-WM World Models on Pre-trained Visual Features.pdf
paper_sha256: 7683c366109c091a47b7ae8a3d6520131edd3db1a5aa594d74c7636d2f9745d6
processed_at: '2026-08-03T22:01:50-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DINO-WM 人话版

好，我换个方式讲，像咱们在 NeurIPS 走廊聊天那样。

---

## 这篇paper在搞啥

想象你教一个robot开车。传统方法要么:

- **Online RL**: 让robot在路上反复试错，撞了再来。换个parking lot就要重学。
- **Imitation learning**: 给一堆expert demo让它copy。换task又得重训。
- **World model + RL (DreamerV3)**: 在learned "dream" environment里训policy。但reward signal绑死了，换task要重训。

DINO-WM 的想法特别朴素：**我就学一个环境dynamics，task到test time再说**。给个goal image，我自己planning怎么reach。

类比一下：你学物理，学"物体在重力下怎么运动"，学完之后，无论让你预测抛篮球还是抛石头，都能用同一套物理。不需要为每个object重学。

这个idea本身不新，Ha & Schmidhuber 2018就提过 "World Models"。但为什么之前没work好？DINO-WM 说: **因为之前的 visual representation 太烂了**。

---

## 核心insight：为什么 DINOv2 是game changer

之前world model的pain:

- DreamerV3 / IRIS / TD-MPC2 都要**从头学encoder**，在control data上训。control data才几千条trajectory，perception根本学不好。
- 学出来的latent要么被reward信号拉歪了（TD-MPC2），要么被pixel reconstruction拉歪了（IRIS）。
- 结果就是latent space对control有用但对spatial reasoning没用。

DINOv2 在1.42亿张图上自监督训出来的patch features，**天然就有object的概念**。你给它一张桌面推物体的图，它自动把"推手"、"T block"、"背景"分成不同的patch cluster。这玩意儿根本不是为control训的，但它的spatial understanding恰恰是control最需要的prior。

所以 DINO-WM 就说: **干嘛要重学perception？直接拿DINOv2当frozen encoder，在上面学dynamics prediction就好了**。

这就好比你写code，不用从汇编开始写，直接用PyTorch。DINOv2 就是world model的"PyTorch" —— 一个足够好的perception foundation，你只需要在它上面搭dynamics head。

参考 DINOv2 paper: https://arxiv.org/abs/2304.07193

---

## 为什么不reconstruct pixels

这个点你Andrej肯定特别有共鸣。

之前一堆world model (DreamerV3, IRIS, Visual Foresight) 都要predict pixels，然后从pixels提取action。问题:

- Pixel prediction 浪费大量capacity在无关细节上：光照、阴影、纹理noise
- 这些细节对control毫无帮助，但model必须predict对才能loss低
- Result: model的latent被pixel detail绑架，task-relevant信息被挤掉

DINO-WM 干脆就不predict pixels。它在 DINOv2 patch embedding space里predict next frame的embedding。Decoder只在需要visualization的时候单独训一个，**完全不影响predictor**。

Table 7 那个ablation特别striking: 加decoder loss到predictor，PushT成功率从0.92掉到0.80。**pixel supervision actively hurts representation learning for control**.

这就是Yann LeCun一直倡导的JEPA哲学的empirical evidence。你predict in abstract space，别generate pixels。

参考 I-JEPA: https://arxiv.org/abs/2301.08243

---

## 架构其实特别简单

我画给你看：

```
image o_t
   │
   ▼
[DINOv2 frozen]  →  z_t (196个patch, 每个384维)
   │
   │  + action a_t (MLP后concat到每个patch)
   ▼
[ViT predictor]  →  z_{t+1} predicted
   │
   ▼
(可选: decoder → image，只用来visualization)
```

就这么多。19M参数。比你nanoGPT还小。架构上没有任何fancy的东西。

但有几个细节很关键：

### Patch features, not CLS token

DINOv2 输出有两种：CLS token (一个vector表示整张图) 和 patch tokens (196个vectors保留spatial layout)。

CLS看起来更简洁，但Table 2显示差距巨大：PushT上，DINO CLS只有0.44 success rate，DINO Patch是0.90。

为什么？因为manipulation需要知道"推手在左下角，T block在中间"。CLS token把这些spatial info全压成一个vector，信息丢了。Patch保留了2D layout，predictor可以spatially reason。

这点你肯定懂，就像ViT比ResNet强就是因为保留了spatial tokens不做early pooling。

### Frame-level causal attention

这个设计有点微妙。IRIS (之前的SOTA) 把image当language，token-by-token autoregressive生成。DINO-WM 不同，它**一次预测整帧的所有patches**，跨帧才做causal。

物理直觉：时间上是causal的（未来不能attend过去），但同一时刻的所有spatial location是simultaneous的（一个物体的所有部分同时存在）。Frame-level causal正好匹配这个物理reality。

而且这样训练效率高，inference也快 —— 一次forward出一整帧，不用196次autoregressive。

Table 6那个ablation显示，**加causal mask至关重要**。没mask的话，训练时模型能"偷看"未来帧，test time就崩了。h=3时，有mask 0.92，无mask 0.08。这个gap太可怕了。

### Action conditioning很朴素

Action就是个MLP后concat到每个patch。没有cross-attention,没有FiLM,没有controlnet。就是concat。

为什么这么简单能work? 因为action对dynamics的影响是global的（机械臂移动影响整个scene），broadcast到每个patch是合理的inductive bias。

如果action是spatially localized的（比如只推某个点），可能cross-attention更好。但paper里这些task都是global action，concat够用。

---

## Planning 怎么搞

训练完了，test time给个start image和goal image，怎么plan?

简单粗暴：**Cross-Entropy Method (CEM) 在action space搜索**。

流程:
1. 随机sample 100个action sequence
2. 每个sequence在world model里rollout，算predicted latent和goal latent的L2 distance
3. 选top-K最低cost的，更新Gaussian分布
4. 重复10次
5. 执行最佳sequence的前几步，下一步replan (MPC)

就这么简单。没有什么policy network，没有什么RL，没有什么reward shaping。**world model + MPC = zero-shot policy**。

这点和你一直说的"model + planning > policy"思路完全一致。Policy是model的amortized version，但如果你有model，直接plan更灵活，zero-shot。

Table 8 显示 CEM 完胜 Gradient Descent，这点我之前展开讲过。核心reason: latent space的geometry可能不smooth，GD沿gradient走可能掉进bad region。CEM是sampling-based，对non-convex landscape更鲁棒。

---

## 实验结果一句话总结

六个环境，从简单navigation到deformable object manipulation:

| | DINO-WM | 次优baseline |
|---|---|---|
| 简单navigation | 接近perfect | 也perfect (DreamerV3) |
| Contact-rich manipulation | **碾压** | 差距巨大 |
| Generalization to new configs | **best** | 大多fail |

最impressive的numbers:

- **PushT**: DINO-WM 0.90, DreamerV3 0.30, IRIS 0.32
- **Rope CD**: DINO-WM 0.41, IRIS 1.11, DreamerV3 2.49
- **Granular CD**: DINO-WM 0.26, IRIS 0.37

TD-MPC2 全0那个observation特别重要: 它的representation完全靠reward signal学，没reward就collapse。证明TD-MPC2不是真正的world model，是task-specific value model。

LPIPS metrics: DINO-WM预测的future frames decode出来，**比专门predict pixels的baseline还要visually accurate**。这是最有力的证据 —— 好的representation + latent prediction > 烂representation + pixel supervision。

---

## 这篇paper真正教会我们什么

### 1. Representation is everything

之前world model的pain本质是representation烂。DINOv2一来，pain消失。你不需要在control data上学perception，internet-scale self-supervised perception已经够好。

这个lesson对整个robot learning领域都适用：**别从头学perception**。

### 2. Don't generate, predict in latent space

Pixel-space world model (Visual Foresight, AVDC, Sora-style)看起来cool但控制精度差。Latent-space prediction看起来boring但实用。

Yann说的JEPA philosophy在control domain得到验证。

### 3. Task-agnostic model + test-time planning > task-specific policy

训一个general world model，test time MPC出action。这比训一个task-specific policy灵活得多。换goal不用重训，换task configuration不用重训。

这就是"system 2 thinking" for robots —— slow, deliberate, planning-based, vs "system 1" feed-forward policy。

参考你Andrej自己的talk: "Software 2.0 vs Software 3.0"那个思路。

### 4. Foundation model + lightweight task head is the pattern

DINOv2 frozen + 19M ViT predictor + CEM planner。这个recipe太clean了。没有reward modeling,没有inverse model,没有expert demo,没有keypoint detection。

Foundation model (perception) + small task-specific head (dynamics) + classical optimizer (planning). 这就是future of robot learning的template。

---

## 我觉得还差什么

### Latent action for internet video

DINO-WM 需要ground truth action。但internet video没action label。如果想scale到internet video，需要先infer latent action (like Genie, LAPA)。

参考 LAPA: https://arxiv.org/abs/2410.11758
参考 Genie: https://arxiv.org/abs/2402.15391

### Amortized planner

CEM每次53秒太慢。需要把planner蒸馏成policy network，只在policy失败时fallback到CEM。这就像AlphaGo的policy network + value network + MCTS组合。

### Stochastic dynamics

现在predictor是deterministic的。对contact-rich task可能不够，需要diffusion或flow matching在latent space capture多模态future。

### Real robot

全simulation实验。Real robot的perception noise、actuator delay、sensor latency可能break整个pipeline。但DINOv2 本身在real image上strong，应该能transfer。

### Long-horizon

Planning horizon T有限。要做"把房间打扫干净"这种long-horizon task，需要hierarchical decomposition。

---

## 一句话总结

**DINO-WM = DINOv2 (frozen perception) + ViT (dynamics predictor) + CEM (planner)**

它证明了一个简单但重要的point: **好的pre-trained visual representation + latent space dynamics prediction + test-time planning，就能做task-agnostic world model，无需reward、无需demo、无需task-specific training**。

这是JEPA philosophy在control domain的clean instantiation，也是foundation model时代world model应该长什么样子的template。

你Andrej应该会喜欢这paper，它没搞什么花哨架构，就一个clean idea + 仔细的实验。这种风格越来越稀少了。

Project page: https://dino-wm.github.io

---

# DINO-WM 深度讲解

Andrej, 这篇paper非常对你的口味 —— 它本质上就是把你和Yann长期倡导的 **"predict-then-plan in latent space without pixel reconstruction"** 思路，在modern foundation model (DINOv2)加持下做了一次clean、scalable的实例化。下面我从 motivation → architecture → math → experiments → intuition → connections 多个角度unpack。

---

## 1. Big Picture: 这篇paper真正想解决什么问题

World model领域一直有一个**两难**：

- **Online world model**（DreamerV3, TD-MPC2 这类）：在environment里rollout收集数据，训练得到的model只覆盖当前policy的support。换一个task就要重训，泛化性差。
- **Offline world model**（IRIS, Visual Foresight 这类）：从pre-collected trajectories学习，task-agnostic，但需要auxiliary supervision —— expert demos, keypoints, inverse model, dense reward —— 这些都破坏了generality。

DINO-WM 的 thesis statement：**auxiliary information可以来自pre-trained visual representation本身**，无需task-specific label。DINOv2 patch features 提供了spatially-structured、object-centric、semantically rich的latent space，在这个空间里做forward prediction足够稳定、足够准确，足以支撑test-time MPC。

这跟你自己在 tweet 和 nanoGPT 系列里反复提的 "representation is all you need, predictor learns on top" 是同一条思路。Yann LeCun 是共同作者，JEPA 哲学（Joint Embedding Predictive Architecture）在 DINO-WM 里以"non-generative, latent predictive"的形式被落地。

参考：
- DINOv2: https://arxiv.org/abs/2304.07193
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://openreview.net/forum?id=WFYbBOEOtv
- DINO (original): https://arxiv.org/abs/2104.14294

---

## 2. Architecture 详解

DINO-WM 由三个解耦组件构成：

```
        o_t (RGB 224×224)
             │
             ▼
   ┌─────────────────────┐
   │  DINOv2 (frozen)    │   Observation model enc_θ
   │  ViT-S/14 patch     │
   └─────────────────────┘
             │
             ▼  z_t ∈ R^(N×E),  N=14×14=196, E=384
             │
   ┌─────────────────────┐
   │  Transition Model   │   p_θ  (decoder-only ViT)
   │  causal attn @frame │
   │  + action MLP concat│
   └─────────────────────┘
             │
             ▼  ẑ_{t+1} ∈ R^(N×E)
             │
   ┌─────────────────────┐
   │  Decoder (optional) │   q_θ  (transposed conv stack)
   │  only for viz       │
   └─────────────────────┘
             │
             ▼  ô_{t+1} (reconstructed RGB)
```

### 2.1 Observation Model

Frozen DINOv2 输入 224×224 (实际resize到196×196), patch size 14, 输出 $z_t \in \mathbb{R}^{N \times E}$ with $N=196, E=384$.

关键点：**patch-level, NOT CLS token**。CLS把整张图压成一个vector会丢掉spatial layout，对navigation和contact-rich manipulation是致命的。Table 2 中 DINO CLS 在 PushT 上 SR=0.44, 而 DINO Patch 是 0.90 —— 这32个百分点的gap就是spatial information的价值。

### 2.2 Transition Model

这是一个**修改过的ViT**，本质是decoder-only transformer：

- **输入**：过去 H 帧的 patch embeddings $\{z_{t-H}, \dots, z_{t-1}\}$ + 对应 actions $\{a_{t-H}, \dots, a_{t-1}\}$
- **Action conditioning**：action $a_t \in \mathbb{R}^K$ 经 MLP $\phi$ 映射到高维，**concat到每个patch vector** $z_t^i$ 上（i=1...N）。这种broadcast式的conditioning保证action信号能均匀影响每个spatial location。
- **Causal attention at frame level**：第 t 帧的 patch $z_t^i$ 只 attend to 历史帧的对应 patches $\{z_{t-H:t-1}^i\}_{i=1}^N$。注意，frame内部所有patch可以互相attend（spatial mixing），但跨frame严格causal。
- **Proprioception**（如果有）：同样concat到patch上。

这与 IRIS (Micheli et al.) 的关键区别：IRIS 在 **token level** 做autoregressive —— $z_t^i$ 可以 attend to $z_t^{<k}$（同一帧已生成的patches），相当于把image当language sequence建模。DINO-WM 论证：这种token-level AR会破坏image的整体性，frame-level causal + intra-frame full attention更好捕捉global structure。

ViT 配置（all envs 共享）：
- depth = 6
- heads = 16  
- MLP dim = 2048
- ~19M parameters

### 2.3 Decoder

完全可选，独立训练。用transposed conv stack 把 $z_t$ 解码回像素 $\hat{o}_t$，loss为pixel MSE。**predictor不接收decoder gradient**。这点很重要 —— Table 7 显示，如果让decoder loss反传到predictor，PushT成功率从0.92掉到0.80。原因：pixel reconstruction会引入与task无关的优化signal，把predictor的representation capacity拉向无关细节。

这正是JEPA的核心哲学：**predict in abstract space, never generate pixels**。

---

## 3. 数学详解

### 3.1 POMDP 定义

环境建模为POMDP $(\mathcal{O}, \mathcal{A}, p)$，其中 $p(o_{t+1} | o_{\leq t}, a_{\leq t})$ 是transition。注意DINO-WM **不需要reward**，也不需要discount/termination，纯dynamics modeling。

### 3.2 各组件公式

**Observation model**：
$$z_t \sim \text{enc}_\theta(z_t | o_t)$$

实际是deterministic的：$z_t = \text{DINOv2}(o_t) \in \mathbb{R}^{N \times E}$

**Transition model**：
$$z_{t+1} \sim p_\theta(z_{t+1} | z_{t-H:t}, a_{t-H:t})$$

实际预测的是next frame的patch embeddings，deterministic forward pass。

**Decoder**（optional）：
$$\hat{o}_t \sim q_\theta(o_t | z_t)$$

### 3.3 Training loss (Eq. 1)

$$\mathcal{L}_{\text{pred}} = \left\| p_\theta\!\left(\text{enc}_\theta(o_{t-H:t}), \phi(a_{t-H:t})\right) - \text{enc}_\theta(o_{t+1}) \right\|^2$$

变量解析：
- $o_{t-H:t}$：从time step $t-H$ 到 $t$ 的观测序列，H是context length
- $\text{enc}_\theta$：frozen DINOv2 encoder
- $\phi$：action encoder (MLP)，将原始action $a \in \mathbb{R}^K$ 映射到higher dim
- $p_\theta$：transition model (ViT)
- $\text{enc}_\theta(o_{t+1})$：ground truth next-frame embedding (用作teacher forcing target)
- $\|\cdot\|^2$：L2 norm，对整个 $N \times E$ 张量求element-wise squared sum

注意teacher forcing：trajectory切成 $H+1$ 长度的segments，对每个predicted frame独立计算loss，**predictor在每个时间步都基于真实历史**，不基于自己之前的prediction（避免exposure bias，但open-loop rollout时会有distribution shift，paper Section 4.7 Figure 4 显示rollout仍很准确）。

### 3.4 Decoder loss (Eq. 2)

$$\mathcal{L}_{\text{rec}} = \| q_\theta(z_t) - o_t \|^2, \quad z_t = \text{enc}_\theta(o_t)$$

这个loss只更新 $q_\theta$（decoder），**不更新 enc_θ（frozen）也不更新 p_θ**。

### 3.5 Planning objective (Eq. 3-4)

给定 $o_0$ 和 $o_g$（goal image）：
$$\hat{z}_0 = \text{enc}(o_0), \quad z_g = \text{enc}(o_g)$$

Cost function：
$$\mathcal{C} = \|\hat{z}_T - z_g\|^2$$
$$\hat{z}_t = p(\hat{z}_{t-1}, a_{t-1}), \quad \hat{z}_0 = \text{enc}(o_0)$$

变量：
- $\hat{z}_t$：timestep t 的predicted latent（**recursive**, 用模型自己的prediction作为下一步输入）
- $z_g$：goal的DINO embedding
- $T$：planning horizon

### 3.6 CEM optimization

CEM (Cross-Entropy Method) 是zero-th order optimization：
1. 从Gaussian $\mathcal{N}(\mu_0, \Sigma_0)$ 采样 N 个 action sequences $\{a_0, ..., a_{T-1}\}$
2. 对每个sequence forward rollout计算 cost $\mathcal{C}$
3. 选 top-K 最低cost的sequences (elite set)
4. 用elite set重新估计 $\mu, \Sigma$
5. 重复若干iterations
6. 执行前 k 个actions，下一步重新plan (MPC, receding horizon)

Hyperparams: N=100 samples, 10 CEM iterations。

### 3.7 Gradient Descent alternative (Eq. 6-8)

由于 $p_\theta$ differentiable，可以backprop cost到actions：
$$a_t \leftarrow a_t - \eta \frac{\partial \mathcal{C}}{\partial a_t}$$

但 Table 8 显示 GD 在 PointMaze 上 SR=0.22, CEM 0.80, MPC 0.98。GD差的原因我推测：
- Latent landscape可能highly non-convex，GD stuck in local minima
- CEM 是sampling-based，能escape local minima
- DINOv2 features可能不是distance-preserving的，gradient direction不一定指向semantic正确的方向

---

## 4. Environments & Datasets

六套环境，复杂度递进：

| Env | Dynamics | Dataset | Traj Len | Frameskip | H |
|-----|----------|---------|----------|-----------|---|
| PointMaze (Maze) | 2-DoF ball, velocity/accel/inertia | 2000 | 100 | 5 | 3 |
| Reacher (Reach) | DMC, 2-joint arm match full pose | 3000 | 100 | 5 | 3 |
| Push-T | contact-rich push, T-shape block | 18500 | 100-300 | 5 | 3 |
| Wall | 2-room navigation with door | 1920 | 50 | 5 | 1 |
| Rope | XArm + Nvidia Flex rope | 1000 | 5 | 1 | 1 |
| Granular | ~100 particles | 1000 | 5 | 1 | 1 |

Rope和Granular的trajectory长度只有5步，但每步simulation rollout要3秒（Table 10），所以world model做inference只要14ms，加速 ~200x，这对deformable object planning意义重大。

Generalization variants:
- **WallRandom**: 随机wall/door位置
- **PushObj**: 训练4种shape，测试2种unseen Tetris-like shapes
- **GranularRandom**: 测试时粒子数量不同（OOD）

---

## 5. Baselines 对比

### 5.1 IRIS (https://arxiv.org/abs/2209.00588)
- Discrete VQ-VAE tokenize image
- GPT autoregressively predict next tokens  
- Token-level causal attention
- DINO-WM 在 PushT: 0.90 vs IRIS 0.32
- 在 Wall 上 IRIS 只有 0.04，因为discrete tokens很难capture continuous navigation dynamics

### 5.2 DreamerV3 (https://arxiv.org/abs/2301.04104)
- Categorical latents + RSSM
- Predict reward + continue flag + state
- 在 Maze, Wall 上完美（1.00），但 PushT 只有 0.30
- DreamerV3的latent是reward-shaping过的，对navigation好，对contact-rich manipulation差

### 5.3 TD-MPC2 (https://arxiv.org/abs/2310.16828)
- Decoder-free, 但用reward signal学latent
- **没有reward时完全fail** (Table 1: 全0)
- 这说明TD-MPC2的representation本质是task-specific的，不是general world model

### 5.4 AVDC (https://arxiv.org/abs/2310.08576)
- Diffusion model generate video conditioned on text goal
- 视觉上realistic，但physical implausible (Figure 6)
- 单步变化可能很大，无法精确reach goal
- Latent DINO-WM vs pixel-space diffusion: 精确性 vs 美观性 tradeoff

### 5.5 Encoder ablations (Table 2)
| Encoder | Type | PushT SR |
|---------|------|----------|
| R3M | global | 0.42 |
| ResNet (ImageNet) | global | 0.20 |
| DINO CLS | global | 0.44 |
| **DINO Patch** | **spatial** | **0.90** |

R3M 在 robotics video 上预训练，但global pool丢掉spatial info。DINO Patch 的优势纯粹来自**保留spatial layout**。

参考：
- R3M: https://arxiv.org/abs/2203.12601
- ImageNet ResNet: https://arxiv.org/abs/1512.03385
- Diffusion Policy (Push-T来源): https://arxiv.org/abs/2303.04137

---

## 6. 关键实验结果深度解读

### 6.1 Table 1: 主结果

| Model | Maze | Wall | Reach | PushT | Rope(CD↓) | Granular(CD↓) |
|-------|------|------|-------|-------|-----------|---------------|
| IRIS | 0.74 | 0.04 | 0.18 | 0.32 | 1.11 | 0.37 |
| DreamerV3 | 1.00 | 1.00 | 0.64 | 0.30 | 2.49 | 1.05 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |
| **DINO-WM** | **0.98** | **0.96** | **0.92** | **0.90** | **0.41** | **0.26** |

观察：
- DINO-WM 在简单task上接近DreamerV3 (Maze 0.98 vs 1.00, Wall 0.96 vs 1.00)
- 在复杂contact-rich task上**碾压**所有baseline (PushT 0.90 vs 0.30, Rope 0.41 vs 1.11)
- TD-MPC2 的全0证实：**无reward signal时，TD-MPC2的representation collapse**

### 6.2 Table 5 (Appendix): Scaling law on PushT

| Dataset | SR | SSIM | LPIPS |
|---------|-----|------|-------|
| 200 | 0.08 | 0.949 | 0.056 |
| 1000 | 0.48 | 0.973 | 0.013 |
| 5000 | 0.72 | 0.981 | 0.007 |
| 10000 | 0.88 | 0.984 | 0.006 |
| 18500 | 0.92 | 0.987 | 0.005 |

这是典型的power-law scaling。10000条轨迹是 saturation knee point。对real robot dataset collection有指导意义 —— 10K trajectories for a single task family是合理budget。

### 6.3 Table 6: Causal mask ablation

| h=1 | h=2 | h=3 |
|-----|-----|-----|
| w/o mask: 0.76 | 0.36 | 0.08 |
| with mask: 0.76 | 0.88 | 0.92 |

h=1时mask无效，两者等价。h>1时无mask的模型训练时能"作弊"attend到future frames，test-time不能 → catastrophic forgetting。有mask时，更长history持续提升performance（捕捉velocity/acceleration/momentum）。

这个ablation对build intuition很重要：**world model必须strictly causal in time, otherwise it learns spurious correlations**.

### 6.4 Table 7: Decoder gradient ablation

| Setting | Success Rate |
|---------|--------------|
| w/o decoder loss to predictor | 0.92 |
| with decoder loss to predictor | 0.80 |

支持JEPA hypothesis: **pixel-level reconstruction objective 伤害 predictive representation learning**.

### 6.5 Table 4/9: LPIPS & SSIM

DINO-WM 在所有环境上 LPIPS 最低 (PushT 0.007 vs 次优0.039)，SSIM最高。这说明即使predictor从未见过pixel supervision，它的latent prediction decode出来仍比专门做pixel reconstruction的baseline更准确。

这是一个strong evidence：**good representation + pure latent prediction > bad representation + pixel supervision**.

参考 LPIPS: http://arxiv.org/abs/1801.03924

---

## 7. 我的intuition building

### 7.1 为什么 DINOv2 patch features work so well?

DINOv2 在 LVD-142M (142M images) self-supervised训练，emergent properties包括：
- **Semantic segmentation** 自然涌现
- **Object-centric attention** 不需要supervision
- **Depth estimation** 可线性probe出来
- **Patch-level correspondence** 跨图像匹配

对world model来说，这些恰好是physical reasoning需要的prior。DINO-WM 本质上 transfer了DINOv2 在internet-scale images上学到的"物体性"和"空间结构"先验到control domain，无需在control data上重新学习perception。

这就是为什么 paper Section 4.5 中 GranularRandom（OOD particle count）仍能work —— 每个patch的语义是"颗粒"，颗粒数量变化只是改变了某些patches的激活pattern，仍in-distribution per patch。

### 7.2 为什么不reconstruct pixels?

JEPA 哲学（Yann 反复强调）：
- Pixel space包含大量irrelevant detail（光照、纹理、噪声）
- Predict pixels 强制 model 浪费capacity学这些细节
- Task-relevant information是sparse的、abstract的

DINO-WM 量化验证：Table 7 显示加decoder loss到predictor降低SR。这就是 JEPA 的empirical evidence。

### 7.3 Frame-level vs Token-level causal

这是 paper 一个 subtle 但重要的设计选择。可以这么想：
- Token-level AR (IRIS)：把image当language，每个patch是"word"。好处是expressivity高，坏处是spatial inductive bias弱，且需要大model才emerge spatial reasoning。
- Frame-level causal (DINO-WM)：保留image的2D structure，intra-frame全连接允许global spatial mixing，跨frame严格temporal causal。这正好匹配physical reality —— 时间是1D causal，空间是2D simultaneous。

DINO-WM的19M params就能work，IRIS通常需要更大model，这印证了frame-level的sample efficiency优势。

### 7.4 为什么 CEM > GD?

直觉上GD应该更好（gradient信息利用）。但实验显示CEM完胜。原因猜测：
1. **Latent space的几何**：DINO embedding space 上的L2 distance 不一定correspond to actionable manifold。GD沿L2 gradient走可能离开"plausible latent"区域。
2. **Action space的非凸**：cost landscape 对action sequence有大量local minima（特别是contact-rich task）。
3. **Sampling-based方法的robustness**：CEM不依赖cost的smoothness，对discrete contact event更友好。

这个insight对 real robot planning 很重要 —— 在 learned latent space 上做 MPC，black-box optimizer 往往比gradient方法更可靠。

### 7.5 Zero-shot planning 的意义

"Zero-shot"在这里意思是：训练时**没有task specification**（无reward, 无demo, 无goal），test时给一个RGB goal image，agent直接plan出action sequence reach goal。

这跟你提到的"LLM in-context learning for control"很相近 —— model trained once, applied to many tasks at test time。只不过DINO-WM 的"reasoning"是**通过MPC optimization**完成的，而不是autoregressive generation。

未来如果world model足够大，可能MPC都不需要，直接autoregressive生成action sequence（类似VPT, π0思路）。但在当前scale下，MPC + CEM 是更可靠的test-time reasoning engine。

参考:
- Video PreTraining (VPT): https://arxiv.org/abs/2210.01850
- π0: https://arxiv.org/abs/2410.24164

---

## 8. 与相关工作的更广连接

### 8.1 与 Ha & Schmidhuber "World Models" (https://arxiv.org/abs/1805.12160)
原始 World Models 用VAE encoder + MDN-RNN + controller。DINO-WM 用frozen pre-trained encoder 替代 in-domain VAE，用ViT替代MDN-RNN。本质架构哲学一致，但scale 和 representation quality 完全不同level。

### 8.2 与 Visual Foresight (https://arxiv.org/abs/1812.00568)
Visual Foresight 是 first pixel-space world model for control。用video prediction model + sampling-based planning。DINO-WM 可以看作 Visual Foresight 的 latent-space + foundation-model 版本。

### 8.3 与 Ebert et al. (https://arxiv.org/abs/1812.00568)
同上 lineage。

### 8.4 与 Sora-style video generation world models (https://arxiv.org/abs/2402.17177)
Sora 类工作 在 pixel space 用 diffusion model 生成video。问题：computational expensive, physically implausible, 不适合 MPC。DINO-WM 论文 Section 4.6 显式对比这点。

### 8.5 与 Genie (https://arxiv.org/abs/2402.15391)
Genie 是 DeepMind 的 latent action video world model。但 Genie 的"action"是latent inferred action，DINO-WM 假设有ground truth action。这是 future work 方向之一（paper Section 5 提到）。

### 8.6 与 JEPA 家族
- I-JEPA: image-level JEPA, masked patch prediction
- V-JEPA: video-level JEPA
- DINO-WM: **action-conditioned video JEPA** —— 严格说不是JEPA（JEPA不predict given action），但哲学相同

如果Yann的grand vision是 JEPA-style world model for AGI, DINO-WM 是这个 vision 在 robot control 上的 closest instantiation。

### 8.7 与 Diffusion Policy (https://arxiv.org/abs/2303.04137)
Diffusion Policy 是 generative action policy。DINO-WM 不是policy，是model + planner。但两者可以互补 —— Diffusion Policy 提供action prior，DINO-WM 提供 forward model for verification/refinement。未来work 可以把 Diffusion Policy 作为CEM的proposal distribution。

### 8.8 与 AdaptiGraph (https://arxiv.org/abs/2407.07889)
Graph-based dynamics for rope manipulation。DINO-WM 在同一Rope环境上工作，但用 generic ViT 替代手工graph structure。Trade-off: graph-based sample efficient but task-specific, DINO-WM generic but data hungry.

### 8.9 与 APT/Any-point trajectory (https://arxiv.org/abs/2401.00025)
Track any point + policy learning。DINO patches 本质上隐式做了 dense correspondence，无需显式tracking。

### 8.10 与 Chain-of-Thought MPC (https://arxiv.org/abs/2304.00776)
CoT-style reasoning for control。DINO-WM 的"reasoning"是geometric optimization in latent space，不是language reasoning。两种 reasoning 方式可能未来merge。

---

## 9. Limitations (paper 提到 + 我的补充)

### 9.1 Paper 明确提到的
1. **Assumes offline dataset with sufficient state-action coverage** —— 高度complex env难收集
2. **Requires ground truth actions** —— 不能直接用internet video
3. **Plans in action space** —— 缺hierarchical structure, fine-grained control可能难

### 9.2 我的补充
4. **No stochasticity modeling** —— transition model是deterministic，对highly stochastic env（多人交互、复杂contact）可能不够
5. **DINOv2 frozen 可能suboptimal** —— 没在control data上fine-tune，可能丢掉某些control-relevant细节。但fine-tune会risk representation collapse。
6. **CEM planning cost** —— 53秒 per planning step (Table 10) 对real-time control不够。需要amortized planner (policy distillation from MPC)。
7. **Goal specification 局限** —— RGB image as goal. 如果goal是abstract（"把蓝色方块推到红色区域"），需要额外language grounding。
8. **No long-horizon composition** —— Planning horizon T有限，long-horizon需要hierarchical decomposition。
9. **Image size 224×224** —— 对fine manipulation（比如threading needle）分辨率不够。DINOv2-high-res 可help，但cost高。

---

## 10. Future Directions (paper + 我的)

### Paper 提到
- Combine with exploration for online data collection
- Use internet video (无ground truth action) → 需要 inverse dynamics model (like VPT, LAPA)
- Hierarchical planning (high-level + low-level control)

### 我的延伸
- **Latent action world model**: 把 DINO-WM 和 latent action discovery (LAPA, Genie)结合，可以训练在unlabeled video上
- **Amortized planner**: distill CEM into a policy network, fall back to CEM on failure
- **Multi-modal latents**: 用 diffusion or flow matching 替代deterministic ViT，capture stochastic dynamics
- **3D world models**: DINOv3 or 3D-aware encoder → 6-DoF manipulation
- **Cross-embodiment**: 不同robot platform共享world model
- **VLM-conditioned**: 用VLM (LLaVA, GPT-4V)生成goal image，DINO-WM做planning，实现language→image→action pipeline

参考 LAPA: https://arxiv.org/abs/2410.11758

---

## 11. 总结: 这篇paper的真正贡献

DINO-WM 的核心 intellectual contribution 不是architecture novelty (架构很简洁)，也 不是 SOTA numbers。它的贡献是**empirically establish**:

1. **Foundation visual representations (DINOv2) can serve as universal observation model for world models** —— 无需per-task encoder learning
2. **Latent-space prediction without pixel reconstruction is the right paradigm** —— JEPA philosophy validated in control
3. **Task-agnostic offline world model enables zero-shot test-time planning** —— decouple model learning from task specification

这三点加起来，把world model 从"per-task RL infrastructure" 重新定位为"general reasoning engine for control"。和你Andrej一直说的"let model predict, let planning emerge"思路高度aligned。

Reference project page: https://dino-wm.github.io

---

如果你 (Andrej) 想在这个方向上再深入，几个最 promising 的 open question:
1. **如何scale DINO-WM到internet video?** 需要latent action discovery
2. **如何amortize planner?** 把CEM蒸馏成policy，但保留CEM作为fallback
3. **DINO-WM 在 real robot 上work吗?** Paper 全是simulation，sim-to-real未验证
4. **DINOv2 是最优encoder吗?** I-JEPA / V-JEPA / DINOv3 可能更好
5. **能否merge with autoregressive action generation (like π0)?** World model + policy的dual system

期待你的进一步 thoughts!
