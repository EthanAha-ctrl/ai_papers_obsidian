---
source_pdf: AdaJEPA An Adaptive Latent World Model.pdf
paper_sha256: 8e4dfbf66b1f4b0ef46122dab4a291a470ff55cd736a384de298516df3c4b2f1
processed_at: '2026-07-18T01:47:43-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AdaJEPA 深度解析

## 1. 核心 Motivation: 为什么 Frozen World Model 在 Test-Time 会失败

当前 JEPA world model 的标准范式：offline 预训练 encoder $\mathcal{E}_\phi^s$ + predictor $f_\theta$，test-time freeze 后用 MPC planning。问题出在两个层面：

- **Encoder misalignment**：visual corruptions (Gaussian blur, salt-and-pepper noise, lighting changes, color shifts) 会让 $\mathcal{E}_\phi^s(o_t)$ 偏离 training manifold
- **Predictor misalignment**：dynamics shifts (mass, damping 变化) 让 $f_\theta(z_t, u_t)$ 的 transition structure 失效

关键 insight 来自 neuroscience：生物系统的 cerebellum 会持续用 sensorimotor prediction error 校准 internal model (Shadmehr et al., 2010)。AdaJEPA 把这个机制搬进 MPC closed loop。

## 2. JEPA World Model 形式化

### 2.1 Forward Model (Eq. 1)

$$z_t = \mathcal{E}_\phi^s(o_t), \quad u_t = \mathcal{E}_\psi^a(a_t), \quad \hat{z}_{t+1} = f_\theta(z_t, u_t)$$

变量解释：
- $o_t \in \mathbb{R}^{n_o}$: 高维 observation (image + proprioception)
- $a_t \in \mathbb{R}^{n_a}$: action vector
- $z_t$: latent representation (本文 PushT 用 1×384 global feature，DINO-WM 用 196×384 spatial patch features)
- $u_t$: action embedding
- $\hat{z}_{t+1}$: predicted next latent (not pixel reconstruction)

$f_\theta$ 实际实现是 transformer-based predictor，输入是 history window=3 frames + frameskip=5 的 action chunks 拼接后的 embedding sequence。

### 2.2 Training Objective (Eq. 2)

$$\mathcal{L}_{\text{pred}} = \frac{1}{K}\sum_{k=1}^{K} \ell(\hat{z}_{t+k}, z_{t+k})$$

这里 $K$ 是 prediction horizon，$\ell$ 是 MSE。关键 trick：target $z_{t+k}$ 加 stop-gradient (sg) 防 collapse，配合 curvature regularization 让 latent trajectory 更 "straight" 便于 planning (Wang et al., 2026 的 temporal straightening)。

### 2.3 MPC Planning (Eq. 3)

$$a_{t:t+H-1}^* = \arg\min_{a_{t:t+H-1}} \sum_{k=1}^{H} \alpha_k d(\hat{z}_{t+k}, z_g)$$

变量：
- $H$: planning horizon (=25)
- $\alpha_k$: temporal weights (discount future)
- $d(\cdot,\cdot) = \lVert \cdot \rVert_2^2$: squared Euclidean latent distance
- $z_g = \mathcal{E}_\phi^s(o_g)$: goal latent

Planner 两选：gradient-based (GD, Adam, lr=0.1, 100 steps) 或 sampling-based (CEM, 200 samples, 10 iterations)。执行 action chunk (5 actions per replan due to frameskip=5)。

## 3. AdaJEPA 的核心：Closed-Loop Plan-Execute-Adapt-Replan

### 3.1 Algorithm 1 循环

```
for t = 0...T-1:
    1. Plan: 用当前 model 解 Eq. 3 得到 a_{t:t+H-1}
    2. Execute: 执行第一个 action chunk，observe o_{t+1}
    3. Buffer update: (o_t, a_t, o_{t+1}) → B (cap to size N=5)
    4. Adapt: U=1 gradient step on L_ada(B)
    5. Replan with updated model
```

### 3.2 Adaptation Loss (Eq. 4)

$$\mathcal{L}_{\text{ada}}(\mathcal{B}) = \frac{1}{|\mathcal{B}|}\sum_{(o_i, a_i, o_{i+1}) \in \mathcal{B}} \ell\Big(f_\theta(z_i, \mathcal{E}_\psi^a(a_i)), \text{sg}(z_{i+1})\Big)$$

注意这里 $z_i = \mathcal{E}_\phi^s(o_i)$ 仍通过 encoder，但 target $z_{i+1}$ 加 stop-gradient。这个 self-supervised signal 本质上和 pretraining loss 一样，区别只在数据来源：pretraining 用 offline dataset $\mathcal{D}_{\text{off}}$，adaptation 用 online buffer $\mathcal{B}$。

### 3.3 Parameter Update (Eq. 5)

$$\Omega \leftarrow \Omega - \eta \nabla_\Omega \mathcal{L}_{\text{ada}}(\mathcal{B})$$

$\Omega \subseteq \{\phi, \psi, \theta\}$ 是被更新的子集。Default 配置：**只更新 predictor 最后一个 transformer block + encoder 最后一个 stage (projection head)**，其他全 freeze。这是 paper 最 surprising 的点：1 gradient step + 极少参数 + 单 transition buffer，就能 fix OOD shift。

## 4. 关键 Design Choices 的 Intuition

### 4.1 为什么只更新 last layers 就够了？

Paper 的 ablation (Figure 8) 比较了：
- `predlast+enclast` (default)
- `predfirst+enclast` (对 layout shift 特别有效)
- `predlast+encfirst`
- `predlast+encfrozen`
- `LoRA` (rank 8, α=16, 全模型)

发现：
- **Shape shifts**: encoder frozen 也行，说明 pretrained representation 对 geometry 有 generalization，correction 主要在 predictor
- **Visual shifts**: 必须更新 encoder，因为 mismatch 从 observation 进入
- **Layout shifts**: 更新 predictor **first layer** 最好，因为 first layer 直接接收 latent+action input，能 recalibrate local transition connectivity

Intuition：JEPA 学的是 abstract latent dynamics，OOD shift 通常 manifest 在 "readout" 层面而非 deep representation。更新 last layer 等价于在 frozen feature space 上做 local recalibration，类似 linear probe 的 logic。

### 4.2 Buffer Design

两种策略：
- **recent-N** (default, N=5): 聚焦 local dynamics
- **hard-N**: 保留 prediction error 最大的 N 个 transitions

Ablation (Figure 9) 显示即使 no buffer (只用当前 transition) 也能大幅超过 frozen。Recent-N 给最稳定 gain。

### 4.3 Learning Rate 选择

Default: 直接用 training lr ($\eta_{\text{pred}}=5\times10^{-4}$, $\eta_{\text{enc}}=10^{-5}$)。

Ablation 发现 lr 和 #steps 强耦合：
- 大 lr (5× training) + 1 step: 有效但多 step 会 overshoot
- 小 lr (0.2×): 稳定但需要更多 step，增加 latency
- Training lr + 1 step: 最优 practical default

## 5. 实验数据深度解读

### 5.1 Shape Shifts (PushObj)

训练 shapes: {T, L, Z, +}；测试 held-out: {I, smallT, square}。从 Figure 2 上排：
- Frozen model 在 unseen shapes 上 success rate 大幅 drop
- AdaJEPA **几乎翻倍** planning success
- Gains 随 replan steps 持续增长，而 frozen model 早期 saturate

### 5.2 Visual Shifts (PushT corruptions)

6 种 shift 测试：
1. Gaussian blur
2. Salt-and-pepper (snp) noise  
3. Dark lighting
4. Red moving T (color)
5. Red anchor T
6. Red agent

前 3 个有 clear gains；后 3 个 color shift gains 较 modest。Paper 解释：model 依赖 color 区分 anchor vs manipulated object，需要 data augmentation 或 invariance regularization。

### 5.3 Dynamics Shifts (Table 1)

PointMaze-Medium:
- **low mass (×0.2)**: 同 force 下移动更快。Frozen GD: 77.3% → AdaJEPA predlast+enclast: 78.7% (+1.3)，predfirst+enclast: 82.0% (+4.7)
- **high damping (×20)**: velocity 衰减更快。Frozen CEM: 76.0% → Adapt: 82.0% (+6.0)

有趣点：Frozen model 在 dynamics shift 上 baseline 已经很强 (77%+)，paper 推测是 history=3 frames 的 in-context learning 效应。AdaJEPA 仍有 consistent additional gain。

### 5.4 Layout Shifts (Unseen Mazes)

这是最 dramatic 的结果：
- Frozen GD: 53.3% → predfirst+enclast: **78.7% (+25.3)**
- Frozen CEM: 49.3% → predfirst+enclast: **70.7% (+21.3)**

更新 predictor **first layer** 比 last layer 多 ~15% gain，因为 first layer 直接处理 latent+action 的 local transition structure，对 maze connectivity 变化最敏感。

### 5.5 Latency Analysis (Table 2)

三种 architecture 对比：
1. Temporal Straightening (global feat., 1×384): adapt +0.03s/replan
2. Temporal Straightening (spatial feat., 196×384): +0.01-0.02s
3. DINO-WM (patch, 196×384): +0.02-0.03s

几乎 negligible。而且 adapt 后 agent 用更少 replan 次数 reach goal，total time 反而下降。

### 5.6 Data Scale 实验 (Figure 6, Section 4.4)

这个实验很重要，回答了 "adapt 能否替代 data scaling"：

维度：
- $K$: shape diversity ∈ {1, 2, 4}
- $N$: trajectories per shape ∈ {1k, 2k, 4k, 8k, 16k}

Key findings:
1. **Diversity > Quantity**: 同样 16k total，K=4,N=4k vs K=1,N=16k：unseen adapted 52% vs 46%
2. **Adapt 补偿 low data**: K=1, N=1k + adapt = 60.8% seen success，超过 frozen K=4, N=16k (54%)
3. Adapt 在所有 scale 都有 gain，但 low-data regime gain 最大 (30%+ seen, 15%+ unseen)

Limitation：当 test environment 需要的训练时 absent 的 feature 时，adapt 也补不齐。需要 continual/active learning 扩展 coverage。

## 6. Visualization 的 Insight (Figure 7, 11-14)

Paper 训了一个 decoder 在 pretrained latent space 上，发现 light-weight adaptation 后 decoder 仍能 reconstruct rollout，且 decoded images 倾向于 "训练域结构"：
- Unseen red T block → decoded 成 gray block (训练颜色)
- Unseen square shape → decoded 成 visually similar seen shape (如 +)
- Salt-and-pepper noise → decoder 输出干净 PushT

这说明 AdaJEPA 通过 **exploiting shared latent structure** 改善 planning，而不是学新 representation。它 recalibrate prediction 使其 stay close to learned manifold，这也是为什么 1 step + 少量参数 就够：本质是 manifold 上的 local projection correction。

## 7. 与 Related Work 的 Positioning

### 7.1 vs Parthasarathy et al. (2025)
最 close 的 motivation 工作，但他们的 model 在 planning 时仍 frozen，只在 training 时用 data synthesis 减小 train-test gap，且只 evaluate in-distribution。AdaJEPA 是首个在 planning 期间 adapt JEPA 的。

### 7.2 vs TTT/TTA literature
传统 TTT 用 auxiliary task (Sun et al., 2020)、entropy minimization (Tent, Wang et al., 2021)、masked reconstruction (Gandelsman et al., 2022)。AdaJEPA 的 self-supervised signal 是 **JEPA 自身的 prediction objective**，天然 fit latent planning pipeline。

### 7.3 vs Online Model-Based RL
Dyna-style (Sutton, 1991) 和 TD-MPC (Hansen et al., 2022) 的 model update 和 policy/value learning coupled。AdaJEPA decouple 了：只 adapt world model，不动 policy (因为 policy 是 MPC planning 出来的)。

### 7.4 vs AdaWorld (Gao et al., 2025)
AdaWorld 用 latent actions adapt，但需要 target-domain finetuning data。AdaJEPA 不需要额外数据，用自己 action 产生的 transition。

### 7.5 vs Adapting World Models (Lanier et al., 2025)
Lanier 用 latent-state dynamics residuals，但需要额外 online rollouts。AdaJEPA 直接在 MPC loop 内 1-step adapt。

## 8. 给 Karpathy 的 Intuition Building

### 8.1 为什么 1 gradient step 够？

这个结果反直觉但可解释。考虑 OOD shift 的性质：visual corruption (blur, noise) 或 dynamics change (mass×0.2) 在 latent space 引起的 perturbation 通常是 **smooth, low-rank** 的。1 step SGD 在 last layer 等价于一次 Newton-like correction，对 smooth perturbation 足够。多步反而 overshoot 因为 buffer 太小 (N=5) 提供 noisy gradient。

### 8.2 为什么 freeze encoder 有时也行 (shape shift)？

JEPA encoder 学的是 contact dynamics 的 abstract geometry invariant。不同 shape 共享 "pusher-pushee contact" 的物理结构，latent representation 对 shape geometry 有 natural generalization。Mismatch 主要在 predictor 如何 map latent+action → next latent 的 transition rule，所以 predictor update 就够。

### 8.3 为什么 layout shift 需要更新 predictor first layer？

Maze layout 改变 free space connectivity，即 latent state 的 **transition graph 结构** 变了。Predictor first layer 直接 receive $z_t$ 和 $u_t$，是建模 "where can I go from here" 的 layer，所以对 connectivity 最敏感。Last layer 只是 refine 输出，对 structural change 不够 responsive。

### 8.4 生物学类比 (Paper Section 1)

Paper 引用 Shadmehr & Mussa-Ivaldi (1994) 的 force-field adaptation 实验：人类在 altered dynamics 下用 cerebellum 持续 update forward model。AdaJEPA 是这个机制的 computational analog：每个 action 产生 observation，prediction error 驱动 model update，update 后的 model shape 下一次 decision。这是 **active inference** 的简化版，但工程上 elegantly minimal。

### 8.5 Limitation 与 Future Direction

Paper 诚实地指出：adapt 的 effectiveness bounded by pretrained representation 的 coverage。如果 test environment 需要 training 时 absent 的 feature (比如完全新的 sensor modality)，adapt 补不齐。自然 next step 是 combine lightweight test-time adaptation + continual/active learning 持续扩展 coverage。

## 9. 实操层面的 Takeaway

如果你要在自己的 world model pipeline 上 reproduce AdaJEPA：

1. **Minimal config**: predictor last block + encoder projection head, 1 GD step, training lr, buffer size 5 recent transitions
2. **MPC integration**: 在每个 replan 后、observe $o_{t+1}$ 后插入 adapt step，不动 planner 逻辑
3. **Anti-collapse**: 用 stop-gradient on target (default)；如果想更新更多 encoder 层，考虑 VICReg-style regularization (Bardes et al., 2022) 或 LeJEPA (Balestriero & LeCun, 2025) 的 provable non-collapse
4. **Per-environment tuning**: layout/structural shift → update predictor first layer；visual shift → update encoder last stage；dynamics shift → default 即可
5. **Latency**: 实测 0.01-0.03s/replan，negligible；但注意 adapt 后总 replan 次数减少，端到端 wall-clock 反而更快

## References

- AdaJEPA project page: https://agenticlearning.ai/adajepa
- LeCun (2022) "A path towards autonomous machine intelligence": https://openreview.net/pdf?id=BZ5a1r-kVsf
- V-JEPA 2 (Assran et al., 2025): https://arxiv.org/abs/2506.09985
- DINO-WM (Zhou et al., 2025): https://arxiv.org/abs/2410.04463
- Temporal Straightening (Wang et al., 2026): ICML 2026
- Sobal et al. (2025) "Learning from reward-free offline data": https://arxiv.org/abs/2502.07894
- Parthasarathy et al. (2025) "Closing the train-test gap": https://arxiv.org/abs/2512.09929
- Test-Time Training (Sun et al., 2020): https://arxiv.org/abs/1911.03332
- TENT (Wang et al., 2021): https://arxiv.org/abs/2006.10726
- MAE-style TTT (Gandelsman et al., 2022): https://arxiv.org/abs/2205.07811
- AdaWorld (Gao et al., 2025): https://arxiv.org/abs/2503.07701
- Lanier et al. (2025) latent residuals: https://arxiv.org/abs/2504.02252
- Shadmehr & Mussa-Ivaldi (1994) cerebellar adaptation: https://www.jneurosci.org/content/14/5/3208
- TD-MPC2 (Hansen et al., 2024): https://arxiv.org/abs/2310.16828

整体来看，AdaJEPA 的 elegance 在于它 **不发明新 architecture、不发明新 loss、不发明新 planner**，而是把一个 closed-loop feedback control 的古老 idea (adaptive control, 1979 年 IDCOM 就有了) 精准地插入到 modern JEPA+MPC pipeline 的 correct intervention point，用 minimal intervention (1 step, few params) 获得 maximal effect。这种 "right idea at right place with right scale" 的 work，正是 LeCun lab 一贯风格。
