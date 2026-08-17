---
source_pdf: stable-worldmodel A Platform for Reproducible World Modeling Research
  and Evaluation.pdf
paper_sha256: d9b1ea1261c7aea9f0a382c1992073b62e6f713553533557a24b71f3b498b666
processed_at: '2026-08-12T10:32:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲 stable-worldmodel

Andrej，我用最直白的方式给你捋一遍这篇 paper 在干什么。

## 1. 这个 field 的现状有多乱

World model 这个方向现在的研究基本是"各自为政"的状态。每个 lab 从零搭一套 pipeline：自己写 data loader，自己实现 baseline，自己实现 planner，自己定义 evaluation protocol。结果就是同一个 CEM (Cross-Entropy Method) 在至少五篇 recent papers 里被独立实现了一遍 (TD-MPC, PLDM, DINO-WM, LeWM, V-JEPA2)，每次实现 fidelity 都有细微差别。你拿两个 paper 比 performance，根本不知道 performance gain 是 method 带来的还是 implementation 带来的。这是 reproducibility crisis 的经典症状。

还有个更隐蔽的问题：data loading。World model 训练要吃 video + actions + proprioception 这种 temporal contiguous blocks，跟普通 CV 的 per-frame sampling 完全不同。用 per-frame 存，I/O 爆炸；用 MP4 压缩，取第 N 帧要解码前面 N-1 帧，random access 死掉。结果 GPU 一直饿着等数据。这个问题没人系统解决过。

Evaluation 那块更糟。大家都在 training distribution 附近测，success rate 高就发 paper。但 [From Kepler to Newton](https://arxiv.org/abs/2602.06923) 那篇 paper 早就指出，high-capacity sequence model 可以 fit trajectory 但完全没学到 underlying dynamics law。你 in-distribution success rate 高，可能只是学到了 spurious correlation，跟真懂 dynamics 是两回事。

swm 就是来一次性解决这三个问题的。

## 2. Platform 长什么样

三个核心 abstraction，非常简洁：

**World**：所有 environment 的统一 wrapper，底层是 vectorized Gymnasium + EnvPool。关键设计是 `reset()` / `step()` 不 return observation，而是 update 一个 `world.infos` dict in-place，包含所有 environment 输出。这样在 vectorized 场景下不用做 return tuple 的拷贝。

**Policy**：任何能从 `info` dict 产生 action 的对象。random policy、SAC expert、learned policy 都能插。最关键的是 `MPCPolicy`，它 wrap 一个 world model + 一个 solver，每个 timestep 把 observation encode 成 latent state，然后 delegate 给 solver 找 action sequence。

**Solver**：planning algorithm 的 plug-in。只要 world model 实现 `get_cost` method，任何 solver 都能接上去。这意味着 world model 和 planner 完全 decoupled。

用代码说话，整个 loop 就这么几行：

```python
world = swm.World("swm/PushT-v1", num_envs=8, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy(seed=42))
world.collect(dataset_path="pusht.lance", episodes=5000, seed=42)
model = MyWorldModel(...)  # 你自己的 PyTorch model
planner = swm.solver.CEMSolver(model, ...)
wm_policy = swm.policy.MPCPolicy(planner)
world.set_policy(wm_policy)
metrics = world.evaluate(episodes=100, seed=0, 
                         options={'variation': ['agent.size', 'background.color']})
```

注意 `options={'variation': [...]}` — 这就是 swm 的核心卖点之一：evaluation 时可以 on-the-fly 改 environment 的 visual/geometric/physical properties，测 zero-shot robustness。

## 3. Data Layer：为什么用 Lance

Lance 是个 modern columnar ML-optimized format ([paper](https://arxiv.org/abs/2504.15247), [repo](https://github.com/lancedb/lance))。它解决了 world model 训练的 fundamental trade-off：

| Format | Local throughput | S3 throughput (no cache) | S3 throughput (cache) |
|--------|------------------|---------------------------|----------------------|
| HDF5 | 1,416 s/s | **9 s/s** | 757 s/s |
| Lance | **4,815 s/s** | 3,184 s/s | 3,253 s/s |
| MP4 | 1,331 s/s | - | - |

直觉上：
- HDF5 over S3 不 cache 基本不可用 (9 samples/sec)，GPU 直接饿死
- Lance local 比 HDF5 local 快 3.4x
- **Lance over S3 比 HDF5 local 还快** — 这个数字很重要，意味着你不用把数据全 download 到本地，直接 stream 训练就行，对大规模实验是 game changer
- MP4 最省 disk 但 random access 死，不适合 training

Lance 是 columnar layout，意味着你可以 partial read (比如只读 actions 不读 frames)，这是跟 MP4 的关键差别。而 random access 能力让你能 sub-sample trajectory，这是 video format 做不到的。

swm 也支持 MP4, HDF5, LeRobot format，提供 one-click conversion。你可以拿 LeRobot 的真实机器人数据直接转 Lance 用。

## 4. World Model Baselines 详解

### 4.1 GCRL 这一支

GCRL 方法不学 predictor，直接学 policy $\pi(a|s, g)$。

**GCBC** ([Goal-Conditioned Behavioral Cloning](https://arxiv.org/abs/1912.06088)) 是最简单的：拿 expert trajectories，supervised learning 学一个 mapping from (state, goal) to action。

$$\mathcal{L} = \frac{1}{H} \sum_{i=0}^{H-1} \|\hat{a}_i - a_i\|_2^2$$

- $\hat{a}_i = \pi_\theta(z_i, z_g)$: 预测的 action
- $z_i = \mathcal{E}(o_i)$: DINOv2 embedding (swm 统一加 DINOv2 head)
- $z_g = \mathcal{E}(g)$: goal embedding
- $H$: history length

简单但 strong，因为直接 mimic expert。

**GCIQL** ([Goal-Conditioned IQL](https://arxiv.org/abs/2110.06169)) 更复杂，学 Q-function 和 value function：

$$\mathcal{L}_Q = \mathbb{E}\left[ \left( Q_\psi(s_t, a_t, g) - (r(s_t, g) + \gamma m_t V_{\bar{\theta}}(s_{t+1}, g)) \right)^2 \right]$$
$$\mathcal{L}_V = \mathbb{E}\left[ L_\tau^2(Q_{\bar{\psi}}(s_t, a_t, g) - V_\theta(s_t, g)) \right]$$

变量解释：
- $Q_\psi, V_\theta$: critic 和 value net
- $Q_{\bar{\psi}}, V_{\bar{\theta}}$: target networks (EMA updated)
- $m_t = 1 - \mathbb{1}[o_t = g]$: terminal mask (到 goal 就是 0)
- $r = -m$: reward (没到 goal 就扣分)
- $\gamma$: discount factor
- $L_\tau^2(u) = |\tau - \mathbb{1}(u<0)| u^2$: asymmetric expectile loss，$\tau$ 控制 pessimism 程度

Policy extraction 用 Advantage-Weighted Regression:
$$\mathcal{L}_\pi = \exp(\alpha A) \cdot \|\pi_\phi(z_t, z_g) - a_t\|_2^2$$
- $A = Q_\psi(z_t, a_t, z_g) - V_\theta(z_t, z_g)$: advantage
- $\alpha$: temperature

**GCIVL** 是 IQL 的简化版，只学 value function 不学 Q。

### 4.2 Latent World Models 这一支

这是 paper 的核心 comparison。这类方法学 encoder $\mathcal{E}$ + predictor $\mathcal{P}$，然后用 MPC 在 latent space 里 planning。

**DINO-WM** ([paper](https://arxiv.org/abs/2411.04983)):
- 冻结 pretrained [DINOv2](https://arxiv.org/abs/2304.07193) 当 encoder
- 只学一个 ViT predictor over patch features
- Loss 就是简单的 L2:
$$\mathcal{L} = \frac{1}{H} \sum_{i=0}^{H-1} \|\hat{z}_{i+k} - z_{i+k}\|_2^2$$
  - $z_{i+k} = \mathcal{E}(o_{i+k})$: frozen DINOv2 target
  - $\hat{z}_{i+k}$: ViT predictor 输出 (causal attention, window size $H$)
  - $k$: prediction offset
- 优点：avoid representational collapse (因为 encoder 冻着)
- 缺点：no end-to-end training，representation quality 受限

**PLDM** ([paper](https://arxiv.org/abs/2502.14819)):
- End-to-end JEPA，encoder 和 predictor joint train
- 用 [CLS] token (不是 patch-level)
- 五项 anti-collapse loss:
$$\mathcal{L} = \mathcal{L}_{sim} + \alpha \mathcal{L}_{std} + \beta \mathcal{L}_{cov} + \delta \mathcal{L}_{temp} + \omega \mathcal{L}_{idm}$$
  - $\mathcal{L}_{sim} = \frac{1}{T}\sum \|\hat{z}_i - z_i\|_2^2$: 主 prediction loss
  - $\mathcal{L}_{std}$: VICReg variance (防止 collapse 到一个点)
  - $\mathcal{L}_{cov}$: VICReg covariance (feature decorrelation)
  - $\mathcal{L}_{temp} = \frac{1}{T}\sum \|z_i - z_{i+1}\|_2^2$: temporal smoothness
  - $\mathcal{L}_{idm} = \frac{1}{T}\sum \|\hat{a}_i - a_i\|_2^2$: inverse dynamics
- 问题：**5 个 hyperparameter 要 tune**，reproducibility 噩梦

**LeWorldModel (LeWM)** ([paper](https://arxiv.org/abs/2603.19312)):
- 同架构，但 loss 极度简化:
$$\mathcal{L} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{SIGReg}$$
  - $\mathcal{L}_{SIGReg}$: push latent distribution 到 isotropic Gaussian
  - $\lambda$: **唯一 effective hyperparameter**
- 这是 major engineering win，让 method 更 accessible

**TD-MPC2** ([paper](https://arxiv.org/abs/2310.16828)):
- Decoder-free implicit world model
- 同时学 encoder, dynamics, reward predictor, Q-ensemble, policy prior
- Loss 四项:
$$\mathcal{L} = \beta_c \mathcal{L}_c + \beta_r \mathcal{L}_r + \beta_v \mathcal{L}_v / N_Q + \mathcal{L}_\pi$$
  - $\mathcal{L}_c = \sum_t \rho^t \|z_t' - \text{sg}(\tilde{z}_{t+1})\|_2^2$: consistency (sg = stop gradient)
  - $\mathcal{L}_r = \sum_t \rho^t \text{CE}(\hat{r}_t, \text{TwoHot}(r_t))$: reward prediction
  - $\mathcal{L}_v = \sum_t \rho^t \text{CE}(Q_i(z, a_t), \text{TwoHot}(y_t))$: value prediction
  - $\mathcal{L}_\pi = \sum_t \rho^t (-\bar{q}_t^\pi / S - \beta_\pi |A| \mathcal{H}_t)$: max-entropy policy
  - $\rho^t$: temporal weight
  - $S$: running scale
  - $\beta_\pi$: entropy coefficient
- Online 训练时很强，但 offline 设置下会崩 (后面讲)

## 5. Planning Solvers：怎么在 latent space 里找 action

Solver 在 swm 里就是个 plug-in，任何 world model 有 `get_cost` method 就能接。

### 5.1 Sampling-based (零阶方法)

**Predictive Sampling** ([Howell et al.](https://arxiv.org/abs/2203.04955)):
最简单的。在 nominal plan 周围加 Gaussian noise，sample N 个 candidates，返回 cost 最低的。不 fit distribution，靠 MPC replanning 来 refine。

**CEM** ([book](https://link.springer.com/book/10.1007/978-1-4757-4321-0)):
维护一个 diagonal Gaussian $\mathcal{N}(\mu, \text{diag}(\sigma^2))$ over action sequence:
$$\mu^{\ell+1} = \frac{1}{E}\sum_{i \in \mathcal{E}^\ell} A_i^\ell, \quad \sigma^{\ell+1} = \text{Std}_{i \in \mathcal{E}^\ell}(A_i^\ell)$$
- $A_i^\ell = \mu^\ell + \sigma^\ell \odot \epsilon_i^\ell$: 第 $\ell$ 轮第 $i$ 个 candidate
- $\mathcal{E}^\ell$: top-$E$ elites (cost 最低)
- $\mu \in \mathbb{R}^{H \times d_a}$: mean over planning horizon $H$ and action dim $d_a$
- $\sigma \in \mathbb{R}^{H \times d_a}$: per-coordinate std

每轮 sample N 个，选 E 个 best，refit Gaussian。$\sigma$ 慢慢收缩，focus 到 low-cost region。

**iCEM** ([paper](https://proceedings.mlr.press/v164/pinneri22a.html)):
CEM 的三点改进：
1. **Colored noise** $\xi \sim \mathcal{C}_\beta(H, d_a)$: 时间维度上 correlate noise，产生更 smooth 的 action sequence (好 control 通常需要时间一致性)
2. **Elite retention**: 保留上一轮 top-$E_{keep}$ candidates 下一轮 reuse (防止好 sample 被浪费)
3. **Momentum**: $\mu^{\ell+1} = \alpha \mu^\ell + (1-\alpha)\hat{\mu}^\ell$ 平滑 update，减少 small elite set 下的 instability

**MPPI** ([paper](https://ieeexplore.ieee.org/document/7487281)):
Soft weighting 而非 hard elite:
$$w_i^\ell = \frac{\exp(-(C_i^\ell - C_{min}^\ell)/\lambda)}{\sum_j \exp(-(C_j^\ell - C_{min}^\ell)/\lambda)}$$
$$\mu^{\ell+1} = \sum_i w_i^\ell A_i^\ell$$
- $\lambda$: temperature，低 → 类似 CEM hard selection，高 → 更 averaging
- $C_{min}^\ell$: 减去最小值是 numerical stability trick
- 比 CEM 在 elite set 小时更 stable

### 5.2 Gradient-based (一阶方法)

这类方法要 world model differentiable，可以直接 backprop 通过 predictor。

**Gradient Descent**:
$$A_i^{k+1} = A_i^k - \eta \nabla_{A_i^k} J_\theta(s_0, A_i^k)$$
- 问题：long-horizon backprop 通过 multiple predictor steps，gradient vanishing/exploding

**PGD** (Projected GD, [Henaff et al.](https://arxiv.org/abs/1705.07177)):
对 discrete action 优化 relaxed probability vector:
$$P_i^{k+1} = \Pi_{\Delta(\mathcal{A})^H}(P_i^k - \eta \nabla_P J_\theta + \xi_i^k)$$
- $P \in \Delta(\mathcal{A})^H$: action 概率在 simplex 上
- $\Pi_{\Delta(\mathcal{A})^H}$: [Duchi simplex projection](https://doi.org/10.1145/1390156.1390191)
- 最终取 $\arg\max_a P_{t, a}$ 作为 discrete action

**GRASP** ([paper](https://arxiv.org/abs/2602.00475)):
最 clever 的 gradient solver。同时优化 action sequence $A$ 和 virtual states $z_1, \ldots, z_{H-1}$:
$$\mathcal{L}_k = \sum_{t=0}^{H-1} \left( \|\hat{z}_{t+1}^k - z_{t+1}^k\|_2^2 + \gamma_k \|\hat{z}_{t+1}^k - s_g\|_2^2 \right)$$
- $\hat{z}_{t+1}^k = \mathcal{P}_\theta(\text{sg}(z_t^k), a_t^k)$: one-step prediction (stop gradient on $z_t$)
- $z_t^k$: virtual state，是 optimization variable
- $\gamma_k$: goal-reaching weight
- 关键 trick：**one-step transitions 可以 parallel 计算**，避免 long-horizon sequential backprop
- 初始化用 linear interpolation: $z_t^0 = (1-t/H) s_0 + (t/H) s_g$
- 每 $K_{sync}$ 步 sync 一次 $A$ 到标准 rollout cost

**Lagrangian**:
处理 inequality constraints $g_j(s_0, A) \leq 0$:
$$\mathcal{L}_{aug} = \sum_i [C_i + \lambda^\ell \cdot G_i + \rho^\ell \|[G_i]_+\|_2^2]$$
- $\lambda^{\ell+1} = [\lambda^\ell + \rho^\ell \bar{G}]_+$: dual ascent
- $\rho^{\ell+1} = \min(\rho_{max}, \rho_{scale} \rho^\ell)$: penalty annealing
- 适合 safety constraint 场景

## 6. FoV：怎么系统测 robustness

swm 最大的 contribution 之一是系统化的 distribution shift 测试机制。两层设计：

**Native FoV (simulator-level)**:
对有 source access 的环境 (MuJoCo, PushT 等)，在 reset 时通过 `options={'variation': [...]}` 直接修改 simulator 内部参数：
- Visual: `agent.color`, `block.color`, `background.color`
- Geometric: `agent.size`, `block.shape`, `block.scale`
- Physical: `physics.floor.friction`, `physics.gravity`, body mass

关键设计：在 reset 时 sample，**整个 episode 保持不变**。这避免了每帧独立扰动造成的虚假 fragility 测量。失败可以归因到 persistent environment change。

**Visual Wrappers (boundary-level)**:
对 closed-source simulators (Atari ROMs, Craftax)，用 Gymnasium wrapper 在 observation boundary 上做 transform。11 种 wrapper：

| Wrapper | 作用 |
|---------|------|
| ChromaKeyWrapper | 用 color keying 替换背景 |
| NoiseWrapper | 加 Gaussian/salt-pepper 噪声 |
| BlurWrapper | Gaussian blur |
| ColorJitterWrapper | 扰动 brightness/contrast/saturation/hue |
| GrayscaleWrapper | 去色 |
| RandomShiftWrapper | 随机平移 |
| CutoutWrapper | 随机 mask 矩形区域 |
| OcclusionWrapper | 随机 occluding patches |
| MovingPatchWrapper | 时间上移动的 occluders |
| RandomConvWrapper | 固定随机卷积 |
| ResolutionWrapper | 降采样再上采样 |

两种 mechanism 可以 compose：一个 World 可以同时有 native FoV + visual wrappers。

## 7. 实验结果：最关键的两个发现

### 7.1 In-distribution baseline (Table 1)

| Method | Push-T SR | OGB-Cube SR |
|--------|-----------|-------------|
| TD-MPC2 | 12 | 4 |
| GCBC | 75 | 84 |
| LeWM | **94** | 72 |
| PLDM | 78 | 62 |
| DINO-WM | 92 | **86** |

观察：
- LeWM 在 Push-T 上最强 (94%)，DINO-WM 在 OGB-Cube 上最强 (86%)
- **TD-MPC2 offline 设置崩了** (12 / 4)
- Table 5 + Figure 13 证明 TD-MPC2 implementation 正确 (online DMC 上跟 SAC 持平)，所以 offline 崩是 algorithmic 问题
- Figure 13 的 PCA projection 显示：TD-MPC2 的 actor rollout 很快 drift 出 training manifold，expert rollout 保持在 support 内。这就是 offline 失败 mode — actor 生成 OOD action，predictor 在 OOD 区域被 fool

### 7.2 核心发现：Prediction error ≠ Planning success

这是 paper 最重要的科学贡献。Figure 4 显示：

在四个递进 OOD 程度的 regime (expert train → expert validation → random policy → random + full FoV)，跑 256 条 trajectory，画 prediction MSE 分布：
- **success (blue) 和 failure (red) 的 MSE 分布大量 overlap**
- 即使在强 OOD 下，低 prediction error 不保证 planning 成功
- 高 prediction error 也不一定意味着失败

直觉上：这意味着 **OOD inputs 本身 (而非 error magnitude) 是 planning failure 的主要 driver**。模型在 OOD 时整个 representation breakdown，不是 graceful degradation。

Figure 10 (PLDM) 确认 cross-model 成立。这对整个 field 的 evaluation metric 设计有重大启示：prediction MSE 是 misleading 的 proxy。

### 7.3 Targeted Perturbation (Table 4)

逐个 factor 拆解 success rate，非常 informative：

| FoV | Entity | LeWM | PLDM | DINO-WM |
|-----|--------|------|------|---------|
| None | - | 50.8 | 50.8 | 20.0 |
| Color | Anchor | 14.0 | 10.0 | 20.0 |
| Color | Agent | 12.0 | 8.0 | 18.0 |
| Color | Canvas | 6.0 | 6.0 | 10.0 |
| Size | Anchor | 26.0 | 18.0 | 14.0 |
| Size | Agent | 22.0 | 18.0 | 4.0 |
| Shape | Agent | 26.0 | **52.0** | 18.0 |
| Position | Anchor | 32.0 | 18.0 | 4.0 |

几个有意思的 patterns：

1. **PLDM 对 agent shape 极其 robust (52.0)**，甚至超过 baseline (50.8)。可能因为 PLDM 用 [CLS] token representation，更 invariant 到 shape 变化
2. **Anchor position 对 DINO-WM 致命 (4.0)**。DINOv2 patch features 对 2D 平面位置变化敏感，而 end-to-end JEPA (LeWM 32.0) 好得多
3. **Canvas color 是 universal killer** (6-10%)。所有 model 都依赖 background-foreground color contrast
4. **DINO-WM baseline 最低 (20.0)**。Frozen DINOv2 在 Push-T 这种简单 2D 场景上可能 over-structured，end-to-end JEPA 更 flexible

### 7.4 Visual Distractors (Figure 5a)

Success rate 随 distractor squares 数量呈 **quadratic decay**。前几个 distractors 模型可 tolerate，超过 threshold 后 rapidly collapse。Pattern 在所有 baseline 上一致。

### 7.5 Chromatic Wheel (Figure 11)

LeWM 在背景颜色 chromatic wheel 上 SR：
- 高 SR 区域：**white 附近** 和 **green axis**
- Green axis robust 可解释：Push-T 默认 anchor 是绿色，model 学到了 green-foreground contrast
- Red/blue/purple 高 intensity → collapse

这表明 model 依赖 specific color contrast 而非 task geometry。典型 spurious correlation。

## 8. 整理一下核心 Take-aways

1. **World model 研究的 fragmentation 是真问题**。同一 CEM 在 5 个 paper 各自实现，跨论文比较无法 isolate method gain vs implementation difference。swm 把 data、baseline、solver、evaluation 全统一。

2. **Data I/O 是 silent killer**。HDF5 over S3 不 cache 只有 9 samples/sec，GPU 直接饿死。Lance 解决，且 **Lance over S3 比 HDF5 local 还快** (3,184 vs 1,416)。这让大规模 streaming training 可行。

3. **Prediction error 和 planning success decoupling** 是最重要的科学发现。Figure 4 显示 success 和 failure 的 MSE 分布大量 overlap。这意味着 community 用 prediction MSE 当 model quality proxy 是 misleading 的。

4. **OOD inputs (而非 error magnitude) 是 failure driver**。World model 在 OOD 时整个 representation breakdown，不是 graceful degradation。

5. **Models 对 specific correlations 敏感而非 dynamics**。Green anchor 让 LeWM 对 green background robust 但对 red/blue collapse。这是 spurious correlation 的直接证据。

6. **DINOv2 features 在简单 2D 任务上可能 over-structured**。DINO-WM baseline 20%，end-to-end JEPA (LeWM, PLDM) 在 in-distribution 上更好。

7. **TD-MPC2 的 offline 失败 mode**：actor 生成 OOD actions，predictor 在 OOD 区域被 fool，drift 累积。Figure 13 PCA 清晰展示这个 drift。

## 9. 我觉得 paper 没说透的几个点

1. **FoV 系统实验只在 Push-T 上做**。Atari, MuJoCo, OGBench 上的系统性 robustness 结果 missing。Push-T 是 2D 简单场景，结论能否泛化到高维 realistic 场景？

2. **No scaling experiments**。Paper 提 scaling 是 future work，但 swm 的 selling point 之一就是 enable scaling studies。Lance data layer 的 throughput 有了，但 model/data/compute scale 的 systematic 结果没有。

3. **Lance vs 其他 columnar formats**。Parquet, Arrow IPC, WebDataset 没比。Lance 是否真唯一最优？

4. **No sim-to-real**。Sec 6 说这是 future work，但这才是 world model 的 ultimate test。在 sim 里测 robustness 只是 proxy。

5. **GRASP 在长 horizon 的 numerical stability** 没详细讨论。Virtual state trick 在 H 大时是否会 ill-conditioned?

6. **FoV 的组合效应**没测。Paper 测了 single factor，但 multiple factor joint perturbation (颜色变 + 形状变 + 物理参数变) 的 interaction effect 没系统分析。

## 10. 对你 (Andrej) 来说可能特别 interesting 的几个点

1. **JEPA vs Reconstruction** 的对比。Paper 里 DINO-WM (frozen encoder + reconstruction-like target) vs PLDM/LeWM (end-to-end JEPA)。在简单任务上 end-to-end JEPA 更好，但 DINO-WM 在 OGB-Cube (更复杂 3D) 上最好 (86%)。这暗示 JEPA 的 advantage 可能 task-dependent。

2. **Representation collapse 的不同解法**。DINO-WM 用 frozen encoder (粗暴但有效)，PLDM 用 5 项 regularizer (复杂)，LeWM 用 SIGReg (1 个 hyperparameter)。LeWM 的简化是 major engineering win， SIGReg 推 latent 到 isotropic Gaussian 这个 idea 值得深挖。

3. **Planning solver 的 trade-off**。Sampling-based (CEM, MPPI) 不需要 differentiable model 但 sample inefficient。Gradient-based (GD, GRASP) sample efficient 但需要 differentiable predictor 且 long-horizon unstable。GRASP 的 virtual state trick 是个聪明的 workaround，把 sequential backprop 变成 parallel one-step prediction。

4. **Evaluation metric 的根本问题**。Prediction MSE 和 planning success decoupling 这个发现，对整个 world model field 的 evaluation protocol 设计有重大启示。我们需要更好的 probe 来测 model 是否真懂 dynamics，而非只测 prediction accuracy。

## 11. Reference Links

**Platform & Code**:
- Lance format: https://github.com/lancedb/lance
- Lance paper: https://arxiv.org/abs/2504.15247
- Gymnasium: https://gymnasium.farama.org/
- LeRobot: https://github.com/huggingface/lerobot
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl
- mbrl-lib: https://github.com/facebookresearch/mbrl-lib

**World Model Baselines**:
- DINO-WM: https://arxiv.org/abs/2411.04983
- PLDM: https://arxiv.org/abs/2502.14819
- LeWorldModel: https://arxiv.org/abs/2603.19312
- TD-MPC2: https://arxiv.org/abs/2310.16828
- TD-MPC (original): https://arxiv.org/abs/2203.04955
- Dreamer V3: https://arxiv.org/abs/2301.04104
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DINOv2: https://arxiv.org/abs/2304.07193
- Genie: https://arxiv.org/abs/2401.13689

**Solvers & Planning**:
- Predictive Sampling: https://arxiv.org/abs/2203.04955
- iCEM: https://proceedings.mlr.press/v164/pinneri22a.html
- MPPI: https://ieeexplore.ieee.org/document/7487281
- CEM book: https://link.springer.com/book/10.1007/978-1-4757-4321-0
- PGD (Henaff et al.): https://arxiv.org/abs/1705.07177
- Simplex projection: https://doi.org/10.1145/1390156.1390191
- GRASP: https://arxiv.org/abs/2602.00475

**Environments**:
- DMC (MuJoCo): https://github.com/google-deepmind/dm_control
- MuJoCo: https://github.com/google-deepmind/mujoco
- OGBench: https://arxiv.org/abs/2410.20092
- Atari ALE: https://github.com/mgbellemare/Arcade-Learning-Environment
- Craftax: https://arxiv.org/abs/2402.16801
- Diffusion Policy (PushT): https://arxiv.org/abs/2303.04137
- Distracting Control Suite: https://arxiv.org/abs/2101.02722
- DMC-VB: https://arxiv.org/abs/2407.12960

**RL Foundations**:
- IQL: https://arxiv.org/abs/2110.06169
- SAC: https://arxiv.org/abs/1801.01290
- D4RL: https://arxiv.org/abs/2004.07219
- GCBC: https://arxiv.org/abs/1912.06088
- VICReg: https://arxiv.org/abs/2105.04906
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf

**Critical Analysis References**:
- From Kepler to Newton (Inductive bias): https://arxiv.org/abs/2602.06923
- WorldTest: https://arxiv.org/abs/2510.19788
- WorldMark: https://arxiv.org/abs/2604.21686
- EB-JEPA: https://arxiv.org/abs/2602.03604
- Novel games benchmark: https://arxiv.org/abs/2507.12821
- Chinchilla scaling: https://arxiv.org/abs/2203.15556
- Reproducibility in RL: https://arxiv.org/abs/1708.04133

---

总结一下：swm 的 contribution 是把 world model 研究从"每个组自己造轮子"升级到"shared infrastructure + standardized evaluation"。最重要的科学发现是 prediction error 和 planning success 的 decoupling，这暗示当前 world model 学到的 representation 在 OOD 时整个崩溃，不是 graceful degradation，重新设计 evaluation protocol 是 field 的 critical next step。Lance-based data layer 是让 scaling experiments 真正可行的 enabler。对你 (Andrej) 来说，JEPA 的各种 anti-collapse trick 对比、GRASP 的 parallel planning idea、evaluation metric 的根本问题，可能都是值得深挖的方向。

---

# stable-worldmodel (swm): Reproducible World Modeling 的一站式平台

Andrej，这篇 paper 切入了一个非常实际且被严重低估的问题：world model 研究目前是高度 fragmented 的，每个 lab 都自己造轮子。作者团队包括 Yann LeCun、Randall Balestriero、以及 NYU/Mila/Brown 的合作者，他们把整个 world model pipeline (数据 → 训练 → evaluation → planning) 串起来做成一个 unified open-source 平台。让我从底层架构、算法、到实验发现一层层拆解。

## 1. 核心问题诊断：World Model 研究的三大瓶颈

Paper 在 Sec 3.1 列出了三个根本性 bottleneck，这些直觉可以帮你 build 整个 field 的 mental model：

### 1.1 Implementation Fragmentation

CEM (Cross-Entropy Method) planner 至少在 5 篇 recent papers (TDMPC, PLDM, DINO-WM, LeWM, V-JEPA2) 中被独立 re-implement，每次都有 subtle fidelity differences。这导致跨论文比较时无法 isolate "performance gain 是来自 method 还是 implementation difference"。这是机器学习研究中典型的 reproducibility crisis (引用 [12-14])。

### 1.2 Data Loading I/O Bottleneck

World model 训练需要 contiguous temporal blocks (video frames + actions + proprioception)，这里有一个 fundamental trade-off：

- **Per-frame storage** (CV 风格)：random access 快，但 I/O overhead 高 (file header decoding 冗余)，storage 大
- **MP4 压缩**：disk footprint 小，但 random access 极差 — 取第 N 帧要解码前面 N-1 帧
- 结果：GPU starves，因为 data loading 跟不上 accelerator

swm 用 Lance format 解决这个 (后面细讲)。

### 1.3 Evaluation 闭着眼睛打靶

Standard benchmarks (Gym-style) 通常在 training distribution 附近评估。引用 [25] (From Kepler to Newton paper, https://arxiv.org/abs/2602.06923) 揭示了一个关键 insight：high-capacity sequence models 可以 fit trajectories 但 fail to recover local dynamical laws。这意味着 in-distribution success rate 完全无法反映 model 是否真的学到了 dynamics，还是只学到了 exploitable correlations。Zero-shot 和 OOD evaluation 是必要的。

## 2. swm 的设计哲学：三个核心 Abstractions

swm 的设计哲学是 "impose as few restrictions as possible on model architecture, while standardizing data/eval/control"。这三个 abstractions 是：

### 2.1 World

统一的 environment wrapper，支持：
- Vectorized execution (EnvPool-based)
- Rendering
- Controllable intervention on visual/geometric/physical properties (FoV)

关键设计点：`world.reset()` 和 `world.step()` 不返回 observations/rewards，而是 update 一个 internal `world.infos` dict in-place，包含所有 environment 输出 (RGB frames, states, rewards, terminations, truncations)。这避免了 Gymnasium API 在 vectorized 场景下的 return 拷贝开销。

### 2.2 Policy

只需实现 `get_actions(info)`。可以是 random policy、expert policy (SAC)、learned policy、或 `MPCPolicy` (wraps world model + solver)。`MPCPolicy` 在每个 timestep 把 observation encode 成 latent state，然后 delegate action selection 给 solver。

### 2.3 Solver

Self-contained planning algorithm，只需 world model 实现 `get_cost` method。这是 pluggable design，让任何 world model 都能接任何 solver。

Algorithm 1 展示了完整 loop：
```python
world = swm.World("swm/PushT-v1", num_envs=8, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy(seed=42))
world.collect(dataset_path="pusht.lance", episodes=5000, seed=42)
model = MyWorldModel(...)
planner = swm.solver.CEMSolver(model, ...)
wm_policy = swm.policy.MPCPolicy(planner)
world.set_policy(wm_policy)
metrics = world.evaluate(episodes=100, seed=0, options={'variation': ['agent.size', 'background.color']})
```

## 3. Data Layer：为什么是 Lance？

Lance (https://github.com/lancedb/lance, paper https://arxiv.org/abs/2504.15247) 是 modern columnar ML-optimized format，特点：
- Fast random access via adaptive structural encodings
- High compression ratios
- Zero-copy operations
- Native versioning
- Streaming from cloud object stores (S3)

### Figure 3 的实验数据 (Push-T 环境)

| Format | Throughput (no cache) | Throughput (cache) |
|--------|----------------------|-------------------|
| HDF5 (local) | 1,416 samples/sec | 1,474 samples/sec |
| HDF5 (S3) | **9 samples/sec** | 757 samples/sec |
| Lance (local) | **4,815 samples/sec** | 4,431 samples/sec |
| Lance (S3) | 3,184 samples/sec | 3,253 samples/sec |
| Video (local) | 1,331 samples/sec | 1,348 samples/sec |

注意几个关键 insight：
- HDF5 over S3 不加 cache 几乎不可用 (9 samples/sec)
- Lance local 比 HDF5 local 快 ~3.4x
- **Lance over S3 比 HDF5 local 还快** — 这意味着 remote streaming 可以替代 local storage，对大规模 training 非常重要
- Figure 8 在 Two-Room 环境上验证 Lance 能 maintain 5.0k samples/sec local / 3.4k samples/sec S3

Lance 的 storage 比 MP4 大一些，但提供 random access 能力。这是 ML workload 中的 sweet spot：columnar layout 允许 partial read (只读 actions 不读 frames)，random access 允许 trajectory sub-sampling。

## 4. World Model Baselines 深度对比

swm 实现了两大 paradigm 的 baselines。

### 4.1 GCRL (Goal-Conditioned RL) 一类

这类方法直接 parameterize policy $\pi(a|s, g)$，不涉及 predictor。

**GCBC** (Goal-Conditioned Behavioral Cloning, https://arxiv.org/abs/1912.06088):
$$\mathcal{L} = \frac{1}{H} \sum_{i=0}^{H-1} \|\hat{a}_i - a_i\|_2^2$$
- $\hat{a}_i = \pi_\theta(z_i, z_g)$: predicted action
- $z_i = \mathcal{E}(o_i)$: DINOv2 embedding (swm 统一加 DINOv2 head)
- $z_g = \mathcal{E}(g)$: goal embedding
- 监督学习，纯 imitation，简单但 strong baseline

**GCIQL** (Goal-Conditioned Implicit Q-Learning, https://arxiv.org/abs/2110.06169):

两个 loss 联合训练 critic 和 value：
$$\mathcal{L}_Q = \mathbb{E}_{(s_t, a_t, s_{t+1}, g) \sim \mathcal{D}} \left[ \left( Q_\psi(s_t, a_t, g) - (r(s_t, g) + \gamma m_t V_{\bar{\theta}}(s_{t+1}, g)) \right)^2 \right]$$
$$\mathcal{L}_V = \mathbb{E}_{(s_t, a_t, g) \sim \mathcal{D}} \left[ L_\tau^2(Q_{\bar{\psi}}(s_t, a_t, g) - V_\theta(s_t, g)) \right]$$

变量解释：
- $Q_\psi, V_\theta$: critic 和 value network
- $Q_{\bar{\psi}}, V_{\bar{\theta}}$: target networks (EMA updated with rate $\rho$)
- $m_t = 1 - \mathbb{1}[o_t = g]$: 0 if terminal, 1 otherwise
- $r = -m$: reward (negative non-terminal cost)
- $\gamma$: discount
- $L_\tau^2(u) = |\tau - \mathbb{1}(u<0)| u^2$: asymmetric expectile loss
- $\tau$: expectile parameter (控制 pessimism)

Policy extraction via Advantage-Weighted Regression:
$$\mathcal{L}_\pi = w \cdot \|\pi_\phi(z_t, z_g) - a_t\|_2^2, \quad w = \exp(\alpha A), \quad A = Q_\psi(z_t, a_t, z_g) - V_\theta(z_t, z_g)$$
- $\alpha$: AWR temperature
- $A$: advantage estimate

### 4.2 Latent World Models (planning-based) 一类

这是 paper 的核心 comparison。

**DINO-WM** (https://arxiv.org/abs/2411.04983):
- 冻结 pretrained DINOv2 (https://arxiv.org/abs/2304.07193) 作为 encoder
- 只学 ViT predictor over spatial patch features
- Training loss:
$$\mathcal{L} = \frac{1}{H} \sum_{i=0}^{H-1} \|\hat{z}_{i+k} - z_{i+k}\|_2^2$$
  - $z_{i+k} = \mathcal{E}(o_{i+k})$: frozen DINOv2 target embedding
  - $\hat{z}_{i+k}$: predicted by causal Transformer with windowed attention (window size $H$)
  - $k$: prediction offset
- 优点：避免 JEPA 的 representational collapse (因为 encoder frozen)
- 缺点：no end-to-end training，限制 representation quality

**PLDM** (Planning with Latent Dynamics Models, https://arxiv.org/abs/2502.14819):
- End-to-end JEPA, jointly train encoder $\mathcal{E}_\theta$ 和 predictor $\mathcal{P}_\theta$
- 用 single [CLS] token embedding (而非 patch-level)
- 五项 anti-collapse objective:
$$\mathcal{L} = \mathcal{L}_{sim} + \alpha \mathcal{L}_{std} + \beta \mathcal{L}_{cov} + \delta \mathcal{L}_{temp} + \omega \mathcal{L}_{idm}$$
  - $\mathcal{L}_{sim} = \frac{1}{T}\sum_{i=1}^T \|\hat{z}_i - z_i\|_2^2$: prediction
  - $\mathcal{L}_{std} = \text{VarLoss}(z_{0:T})$: VICReg variance (防止 collapse to point)
  - $\mathcal{L}_{cov} = \text{CovLoss}(z_{0:T})$: VICReg covariance (decorrelation)
  - $\mathcal{L}_{temp} = \frac{1}{T}\sum_{i=0}^{T-1} \|z_i - z_{i+1}\|_2^2$: temporal smoothness
  - $\mathcal{L}_{idm} = \frac{1}{T}\sum_{i=0}^{T-1} \|\hat{a}_i - a_i\|_2^2$: inverse dynamics
  - $\alpha, \beta, \delta, \omega$: 各项权重 (5 个 hyperparameter to tune)

**LeWorldModel (LeWM)** (https://arxiv.org/abs/2603.19312):
- 同架构 (JEPA, end-to-end)
- 关键创新：用 SIGReg 替代五项正则
$$\mathcal{L} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{SIGReg}$$
  - $\mathcal{L}_{SIGReg}$: 推 latent distribution 到 isotropic Gaussian
  - $\lambda$: **唯一 effective hyperparameter**
- 这种简化是 major engineering win，让 reproducibility 更好

**TD-MPC2** (https://arxiv.org/abs/2310.16828):
- Decoder-free world model
- Components: encoder $h$, dynamics $d$, reward predictor $R$, Q-ensemble $\{Q_i\}_{i=1}^{N_Q}$, policy prior $p$
- Loss:
$$\mathcal{L} = \beta_c \mathcal{L}_c + \beta_r \mathcal{L}_r + \beta_v \mathcal{L}_v / N_Q + \mathcal{L}_\pi$$
  - $\mathcal{L}_c = \sum_t \rho^t \|z_t' - \text{sg}(\tilde{z}_{t+1})\|_2^2$: consistency loss (sg = stop gradient)
  - $\mathcal{L}_r = \sum_t \rho^t \text{CE}(\hat{r}_t, \text{TwoHot}(r_t))$: reward prediction (TwoHot encoding)
  - $\mathcal{L}_v = \sum_t \rho^t \text{CE}(Q_i(z, a_t), \text{TwoHot}(y_t))$: value prediction
  - $\mathcal{L}_\pi = \sum_t \rho^t (-\bar{q}_t^\pi / S - \beta_\pi |A| \mathcal{H}_t)$: max-entropy policy
  - $\rho^t$: temporal discount weight
  - $S$: running scale (从 $\bar{q}_t^\pi$ quantile 更新)
  - $\beta_\pi$: entropy coefficient
- 用 sampling-based planning 加 bootstrapped terminal value

## 5. Planning Solvers 算法详解

### 5.1 Sampling-based (Zeroth-order)

**Predictive Sampling** (https://arxiv.org/abs/2203.04955):
Single-shot, 不迭代 distribution fitting。从 nominal plan 加 Gaussian noise，返回 lowest-cost candidate。简单但 effective due to MPC replanning。

**CEM** (https://link.springer.com/book/10.1007/978-1-4757-4321-0):
$$\mu^{\ell+1} = \frac{1}{E}\sum_{i \in \mathcal{E}^\ell} A_i^\ell, \quad \sigma^{\ell+1} = \text{Std}_{i \in \mathcal{E}^\ell}(A_i^\ell)$$
- $A_i^\ell = \mu^\ell + \sigma^\ell \odot \epsilon_i^\ell$: candidates
- $\mathcal{E}^\ell$: top-$E$ elites (lowest cost)
- $\mu^\ell \in \mathbb{R}^{H \times d_a}$: mean over horizon $H$, action dim $d_a$
- $\sigma^\ell \in \mathbb{R}^{H \times d_a}$: per-coordinate std

**iCEM** (https://arxiv.org/abs/2103.11090):
三项改进：
1. **Colored noise** $\xi_i^\ell \sim \mathcal{C}_\beta(H, d_a)$: temporal correlation in perturbations (smoother action sequences)
2. **Elite retention**: 保留 $E_{keep}$ 个 best candidates 下一轮 reuse
3. **Momentum**: $\mu^{\ell+1} = \alpha \mu^\ell + (1-\alpha)\hat{\mu}^\ell$

**MPPI** (https://ieeexplore.ieee.org/document/7487281):
Soft weighting 而非 hard elite selection:
$$w_i^\ell = \frac{\exp(-(C_i^\ell - C_{min}^\ell)/\lambda)}{\sum_j \exp(-(C_j^\ell - C_{min}^\ell)/\lambda)}, \quad \mu^{\ell+1} = \sum_i w_i^\ell A_i^\ell$$
- $\lambda$: temperature (low → elite focused, high → averaging)
- $C_{min}^\ell$: minimum cost (numerical stability)
- 比 CEM 在 small elite set 时更 stable

### 5.2 Gradient-based (First-order)

**Gradient Descent**:
$$A_i^{k+1} = A_i^k - \eta \nabla_{A_i^k} J_\theta(s_0, A_i^k)$$
- 支持 Adam, gradient clipping, multiple random restarts
- 问题：long-horizon backprop 通过 predictor 不稳定

**PGD** (Projected Gradient Descent, https://arxiv.org/abs/1705.07177):
对 discrete actions 优化 relaxed probability:
$$P_i^{k+1} = \Pi_{\Delta(\mathcal{A})^H}(P_i^k - \eta \nabla_P J_\theta(s_0, P_i^k) + \xi_i^k)$$
- $P \in \Delta(\mathcal{A})^H$: action probability vectors on simplex
- $\Pi_{\Delta(\mathcal{A})^H}$: Duchi projection (https://doi.org/10.1145/1390156.1390191)
- 最终 $\arg\max_a P_{i^\star, t, a}^K$ 取最大概率 action

**GRASP** (https://arxiv.org/abs/2602.00475):
最 interesting 的 gradient-based solver。Jointly optimize action sequence $A$ 和 virtual states $z_1, \ldots, z_{H-1}$:
$$\mathcal{L}_k = \sum_{t=0}^{H-1} \left( \|\hat{z}_{t+1}^k - z_{t+1}^k\|_2^2 + \gamma_k \|\hat{z}_{t+1}^k - s_g\|_2^2 \right)$$
- $\hat{z}_{t+1}^k = \mathcal{P}_\theta(\text{sg}(z_t^k), a_t^k)$: one-step prediction (stop gradient on $z_t$)
- $z_t^k$: virtual state (optimization variable)
- $\gamma_k$: goal-reaching weight
- 关键 trick：**所有 one-step transitions 可 parallel 计算**，避免 long-horizon sequential backprop 的 vanishing gradient
- 初始化: $z_t^0 = (1-t/H) s_0 + (t/H) s_g$ (linear interpolation)
- 每 $K_{sync}$ 步同步一次 $A$ 到标准 rollout cost $J_\theta$

**Lagrangian**:
处理 inequality constraints $g_j(s_0, A) \leq 0$:
$$\mathcal{L}_{aug} = \sum_i [C_i + \lambda^\ell \cdot G_i + \rho^\ell \|[G_i]_+\|_2^2]$$
- $\lambda^{\ell+1} = [\lambda^\ell + \rho^\ell \bar{G}]_+$: dual ascent
- $\rho^{\ell+1} = \min(\rho_{max}, \rho_{scale} \rho^\ell)$: penalty annealing
- 适合 safety constraints

## 6. Environments 和 Factors of Variation (FoV)

### 6.1 环境家族

| Family | Type | Action space | Examples |
|--------|------|--------------|----------|
| DMC | continuous control | continuous | cheetah_run, walker_walk |
| OGBench | 3D manipulation | continuous | OGB-Cube |
| Classic Control | low-dim | mixed | CartPole, PushT |
| Fetch-Suite | robotic manipulation | continuous | FetchPush |
| Craftax | open-world | discrete | 2D survival |
| ALE/Atari | arcade | discrete | 100+ games |
| swm extras | custom | continuous | PushT-v1, TwoRoom-v1 |

### 6.2 FoV 机制：两层设计

**Native FoV (simulator-level)**: 通过 `variation_space` hierarchical 字典配置。在 reset 时一次性 set，保持整个 episode。例如 `physics.floor.friction`, `agent.color`, `block.scale`。这意味着 failure 可以 attribute 到 persistent environment change 而非 frame-wise noise。

**Visual Wrappers (boundary-level)**: 用于 closed-source simulators (Atari ROMs, Craftax)。11 种 wrapper：
- ChromaKeyWrapper (color keying)
- NoiseWrapper (Gaussian/salt-pepper)
- BlurWrapper (Gaussian blur)
- ColorJitterWrapper (brightness/contrast/saturation/hue)
- GrayscaleWrapper
- RandomShiftWrapper
- CutoutWrapper
- OcclusionWrapper (random patches)
- MovingPatchWrapper (temporal motion)
- RandomConvWrapper
- ResolutionWrapper

这两种 mechanism 可以 compose: 一个 World 可以同时 stack visual wrappers over native FoV environment。

### 6.3 关键 design choice

FoV 在 episode reset 时 sample，整个 episode 保持。这是 evaluation 的关键：failure 可归因于 persistent environment change 而非 frame-wise noise。这避免了 "每帧独立扰动" 造成的虚假 fragility 测量。

## 7. 实验发现：关键 Insights

### 7.1 In-distribution Baseline (Table 1)

| Method | Push-T SR | OGB-Cube SR |
|--------|-----------|-------------|
| TD-MPC2 | 12 | 4 |
| GCBC | 75 | 84 |
| LeWM | **94** | 72 |
| PLDM | 78 | 62 |
| DINO-WM | 92 | **86** |

几个观察：
- LeWM 在 Push-T 上最佳 (94)，DINO-WM 在 OGB-Cube 上最佳 (86)
- TD-MPC2 offline 设置下表现差 (12 / 4)，作者 conjecture 是 OOD action generation fooled predictor
- Table 5 + Figure 13 验证 TD-MPC2 implementation 正确 (online DMC 上 performant, 与 SAC 持平)，所以 offline 失败是 algorithmic 而非 implementation artifact
- Figure 13 PCA projection 显示：TD-MPC2 actor rollout 很快 drift 出 training manifold，而 expert rollout 保持在 support 内

### 7.2 Robustness 实验 (Section 4.2)

这是 paper 最 important 的实验。两层分析：

**Layer 1: Progressive Distribution Shift (Figure 4)**
四个 regime 的 trajectory-level prediction MSE 分布：
1. Expert train
2. Expert validation
3. Random policy
4. Random policy + full FoV

关键发现：**prediction error 和 planning success 几乎不相关**。即使 OOD 设置下 MSE 整体上升，success 和 failure 分布大量 overlap。这说明 **OOD inputs 本身 (而非 error magnitude) 是 planning failure 的主要 driver**。

Figure 10 (PLDM 对应版本) 确认这个结论 cross-model 成立。

**Layer 2: Targeted Visual Perturbations (Table 5b / Table 4)**

Table 4 非常 informative，逐个 factor 拆解：

| FoV | Entity | LeWM | PLDM | DINO-WM |
|-----|--------|------|------|---------|
| None | - | 50.8 | 50.8 | 20.0 |
| Color | Anchor | 14.0 | 10.0 | 20.0 |
| Color | Agent | 12.0 | 8.0 | 18.0 |
| Color | Block | 22.0 | 18.0 | 18.0 |
| Color | Canvas | 6.0 | 6.0 | 10.0 |
| Size | Anchor | 26.0 | 18.0 | 14.0 |
| Size | Agent | 22.0 | 18.0 | 4.0 |
| Shape | Agent | 26.0 | **52.0** | 18.0 |
| Shape | Block | 12.0 | 14.0 | 8.0 |
| Position | Anchor | 32.0 | 18.0 | 4.0 |

几个有意思的观察：
- **PLDM 对 agent shape 极其 robust (52.0)**，甚至超过 baseline (50.8)。这可能因为 PLDM 的 [CLS] token representation 更 invariant 到 shape
- **Anchor position 变化对 DINO-WM 致命 (4.0)**，但 LeWM 反而最 robust (32.0)
- **Canvas color 是 universal killer** (6-10% SR)
- DINO-WM baseline SR 最低 (20.0)，可能因为 frozen DINOv2 features 在 Push-T 这种 2D 简单场景上 over-structured

### 7.3 Visual Distractor 数量分析 (Figure 5a)

Success rate 随 distractor squares 数量呈 **quadratic decay**。前几个 distractors 模型可 tolerate，但超过某 threshold 后 rapidly collapse。Pattern 在所有 baseline 上一致。

### 7.4 Chromatic Wheel 分析 (Figure 11)

LeWM 在背景颜色 chromatic wheel 上 SR：
- 高 SR 区域：**white background 附近** 和 **green axis**
- Green axis robust 可解释：Push-T 默认 anchor 是绿色，模型学到 green-foreground contrast
- Red/blue/purple 高 intensity → SR collapse
- 这表明 model 依赖 specific background-foreground color contrast，而非 task geometry

## 8. Build Intuition：Paper 的核心 Take-aways

1. **World model 研究的 fragmentation 是真问题**。同一 CEM 在 5 个 paper 各自实现，无法 fair comparison。统一 platform 是 infrastructure 必需。

2. **Data I/O 是 silent killer**。HDF5 over S3 不 cache 只有 9 samples/sec。Lance 解决这个问题，且 over-S3 Lance 比 local-HDF5 还快。这对 scaling 实验 essential。

3. **Prediction error ≠ Planning success**。Figure 4 是 paper 最 important 的图。意味着 evaluation metric 需要重新设计。当前 community 用 prediction MSE 报告 model quality 是 misleading 的。

4. **OOD inputs (而非 error magnitude) 是 failure driver**。这暗示 world models 学到的 representation 在 OOD 时整个 break down，不是 graceful degradation。

5. **Models 对 specific correlations 而非 dynamics 敏感**。Green anchor 让 LeWM 对 green background robust，但对 red/blue collapse。这是典型的 spurious correlation。

6. **DINOv2 features 在简单 2D 任务上可能 over-structured** (DINO-WM baseline 20%)。End-to-end JEPA (LeWM, PLDM) 在 in-distribution 上 better。

7. **TD-MPC2 的 offline 失败 mode**：actor 生成 OOD actions，predictor 在 OOD 区域 fooled，drift 累积。Figure 13 PCA 可视化清晰展示。

## 9. Open Research Directions (Sec K)

Paper 提出 swm 立即可行的几个方向：

1. **Zero-shot world models**: 当前方法对 mild color shift 都 brittle。需要 architectural 和 training advances。
2. **Long-horizon planning**: 当前 evaluation 多用 short horizon in-distribution trajectories。需要 horizon scaling, hierarchical planning。
3. **World model scaling laws**: 类比 LLM 的 Chinchilla scaling (引用 [72], https://arxiv.org/abs/2203.15556)。Lance data layer 让大规模 training 不再 I/O bound。

## 10. 我的 Critical Commentary

几个我觉得可以深挖的点：

- **FoV evaluation 只测 Push-T**: 虽然 platform 支持多环境，但 robustness 实验主要在 Push-T 上。Atari, MuJoCo, OGBench 上的 FoV 系统性结果 missing。
- **No scaling experiments**: Paper 提到 scaling 是 future work，但没展示 swm 在不同 scale 上的 performance。
- **Lance vs other columnar formats**: 没比 Parquet, Arrow IPC 等。Lance 是否真唯一选择？
- **No real-world transfer**: Sec 6 提到 sim-to-real 是 future work，但这正是 world model 的 ultimate test。
- **Predictor dimension dependence**: GRASP 的 virtual state trick 在 H 大时 numerical stability 没详细讨论。

## 11. Reference Links

**Code & Platform**:
- swm paper code: 提到 "Code available here" (paper 中未给出具体 URL)
- Lance format: https://github.com/lancedb/lance
- Lance paper: https://arxiv.org/abs/2504.15247
- Gymnasium: https://gymnasium.farama.org/
- LeRobot: https://lerobot.github.io/ / https://github.com/huggingface/lerobot
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl
- mbrl-lib: https://github.com/facebookresearch/mbrl-lib

**Baselines**:
- DINO-WM: https://arxiv.org/abs/2411.04983
- PLDM: https://arxiv.org/abs/2502.14819
- LeWorldModel: https://arxiv.org/abs/2603.19312
- TD-MPC2: https://arxiv.org/abs/2310.16828
- TD-MPC (原版): https://arxiv.org/abs/2203.04955
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Dreamer: https://arxiv.org/abs/1912.01603
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DINOv2: https://arxiv.org/abs/2304.07193
- Genie: https://arxiv.org/abs/2401.13689

**Solvers & Control**:
- Predictive Sampling: https://arxiv.org/abs/2203.04955 (Howell et al.)
- iCEM: https://proceedings.mlr.press/v164/pinneri22a.html
- MPPI: https://ieeexplore.ieee.org/document/7487281
- CEM book: https://link.springer.com/book/10.1007/978-1-4757-4321-0
- PGD (Henaff et al.): https://arxiv.org/abs/1705.07177
- Simplex projection: https://doi.org/10.1145/1390156.1390191
- GRASP: https://arxiv.org/abs/2602.00475

**Environments**:
- DMC (MuJoCo): https://github.com/google-deepmind/dm_control
- MuJoCo: https://github.com/google-deepmind/mujoco
- OGBench: https://arxiv.org/abs/2410.20092
- Atari ALE: https://github.com/mgbellemare/Arcade-Learning-Environment
- Craftax: https://arxiv.org/abs/2402.16801
- Diffusion Policy (PushT): https://arxiv.org/abs/2303.04137
- Distracting Control Suite: https://arxiv.org/abs/2101.02722
- DMC-VB: https://arxiv.org/abs/2407.12960

**RL Foundations**:
- IQL: https://arxiv.org/abs/2110.06169
- SAC: https://arxiv.org/abs/1801.01290
- D4RL: https://arxiv.org/abs/2004.07219
- GCIQL / GCBC: https://arxiv.org/abs/1912.06088
- VICReg: https://arxiv.org/abs/2105.04906
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf

**Critical References**:
- From Kepler to Newton (Inductive bias in WMs): https://arxiv.org/abs/2602.06923
- WorldTest: https://arxiv.org/abs/2510.19788
- WorldMark: https://arxiv.org/abs/2604.21686
- EB-JEPA: https://arxiv.org/abs/2602.03604
- Novel games benchmark: https://arxiv.org/abs/2507.12821
- Chinchilla scaling: https://arxiv.org/abs/2203.15556

---

总结一下核心 intuition：swm 的贡献是把 world model 研究从 "每个组自己造轮子" 升级到 "shared infrastructure + standardized evaluation"。最重要的科学发现是 **prediction error 和 planning success 的 decoupling**，这暗示当前 world models 学到的 representation 在 OOD 时整个崩溃而非 graceful degradation，重新设计 evaluation protocol (而非仅优化 prediction accuracy) 是 field 的 critical next step。Lance-based data layer 是让 scaling experiments 真正可行的 enabler。
