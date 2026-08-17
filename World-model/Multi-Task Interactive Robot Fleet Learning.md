---
source_pdf: Multi-Task Interactive Robot Fleet Learning.pdf
paper_sha256: c33ca3b5878e40f515dee957100deb267419b6dd72c25e5cbf55181e9dad150b
processed_at: '2026-08-05T21:14:43-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SIRIUS-FLEET 的人话版本

## 一句话总结

你有一堆机器人跑同一个 multi-task policy, policy 不够好会出错, 你不想一直盯着它们, 但又不能完全不盯。这篇 paper 的核心 idea 就是: 训练一个 visual world model 来 "预知未来", 看未来会不会出事, 要出事就喊人来帮忙。而且随着 robot 越来越厉害, 喊人的频率自动降低。

---

## 问题是什么

想象你开了个工厂, 里面 100 台机械臂, 跑同一个 generalist policy, 能干 50 种活。这个 policy 是你用 behavioral cloning 训的, 训练数据 50 个 demo per task。

问题来了:

**Policy 会出错**。Real-world 不是 simulation, 有 visual variability, object 位置变了, 光照变了, layout 变了。Policy 遇到没见过的情况就懵了。

**你不想一直盯着**。100 台机器人, 你雇 100 个人盯着? 太贵了。而且大部分时候 robot 是 OK 的, 盯着是浪费。

**但你又不能完全不盯**。Robot 摔了杯子, 撞了人, 把东西放错地方, 这些都要避免。

所以核心矛盾是: **如何在 minimal human supervision 下保证 fleet 的 safety 和 reliability**。

---

## 核心 Idea: Visual World Model 作为 "预知未来" 的 backbone

### Intuition

人类怎么知道 robot 要出事? 你看 robot 的动作, 脑子里会 "模拟" 接下来几秒会发生什么。比如你看到机械臂抓歪了, 你脑子里会想 "这样下去肯定抓不到", 然后你介入。

SIRIUS-FLEET 就是让 robot 系统也具备这个 "模拟未来" 的能力。

### 怎么做

训练一个 visual world model $\mathcal{W} = (E_\gamma, D_\lambda, T_\psi)$:

**Encoder $E_\gamma$**: 把当前 image $x_t$ 压缩成 latent embedding $z_t$。这个 $z_t$ 是个低维向量, 包含了 image 的 essential information。

**Decoder $D_\lambda$**: 从 $z_t$ 重构回 image, 确保 embedding 保留了 fine-grained visual details。训练 loss 是 L2 reconstruction loss:

$$\mathcal{L}_{recon} = \|x_t - D_\lambda(E_\gamma(x_t))\|_2^2$$

这里 $x_t$ 是原始 image, $D_\lambda(E_\gamma(x_t))$ 是重构的 image, $\|\cdot\|_2^2$ 是 squared L2 norm。

**Transition model $T_\psi$**: 这是核心。输入过去 $T$ 步的 embeddings $(z_{t-T}, ..., z_t)$, 预测未来 $L$ 步的 embeddings $(\hat{z}_{t+1}, ..., \hat{z}_{t+L})$。

用 cVAE (conditional VAE) 而非 deterministic predictor, 因为 future 是 multi-modal 的: 同一个 state 可以有多种 valid futures。

### 为什么 world model 而非直接做 anomaly detection

关键 insight: **World model 在 diverse multi-task data 上 pre-train 后, 学到的 representation 可以 share 给所有 task 的 anomaly predictors**。

这意味着你只需要 train 一次 world model (在所有 task 的 data 上), 然后 freeze 它, 各个 task 的 anomaly predictor 都能用这个 shared representation, training cost 极低 (failure classifier 只需 1.5 小时)。

这就像 BERT 在 text 上 pre-train, 然后 downstream task 只需 fine-tune 最后一层。SIRIUS-FLEET 的 world model 就是 visual domain 的 "BERT"。

---

## Anomaly Detection: 两种 Predictor 互补

### Failure Prediction — "这个动作像不像之前出错的"

**Intuition**: Human 之前纠正过 robot 的错误, 这些错误 trajectory 就是 labeled data。训练一个 classifier 来识别 "这个 state 会不会导致之前犯过的错误"。

**Label 来源**: Trajectory $\tau = (s_t, a_t, r_t, c_t)$ 中, $c_t$ 标记 human 是否 intervene。Human intervention 之前的 segment 就是 failure。

**3-class design**: $y_i \in \{\text{rollout}, \text{failure}, \text{human}\}$, 区分 "正常执行", "即将 failure", "需要 human takeover"。

**Loss**: Cross-entropy with balanced sampling

$$\mathcal{L}_F = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

$n$ 是 batch size, $y_i$ 是 true label (one-hot), $\hat{y}_i$ 是 predicted probability distribution over 3 classes。

### OOD Prediction — "这个场景我见过没"

**Intuition**: 如果当前 state 和之前见过的所有 states 都差很远, 那就很危险, 因为 policy 在没见过的场景下表现不可预测。

**Method**: 无需训练, 纯 unsupervised。

1. World model 生成所有 training data 的 embeddings
2. PCA 降维 (减少计算 cost)
3. k-means clustering 找到 cluster centroids
4. 新 embedding $z$ 到 nearest centroid 的 distance $d$ 大于 threshold $\alpha_g$ 就是 OOD

$$d = \min_{j \in \{1, \dots, c\}} \|z - \mu_j\|_2$$

$\mu_j$ 是第 $j$ 个 cluster centroid, $c$ 是 cluster 数量, $\|\cdot\|_2$ 是 L2 distance。

### 为什么两个都要

Ablation 结果 (Table 2) 很清楚:

- OOD only: ~40-85% accuracy
- Failure only: ~40-87% accuracy  
- Combined: ~45-99% accuracy

**OOD 捕捉 distribution shift** (novel scenario), **Failure 捕捉 known failure modes**。两者互补:

- 一个 state 可以是 OOD 但 not failure (robot 遇到新场景但碰巧做对了)
- 一个 state 可以是 in-distribution 但 failure (robot 在熟悉场景下犯了之前犯过的错)

---

## 最 Key 的 Innovation: Adaptive Threshold

### 问题

Fixed threshold 在 multi-task setting 下会挂。为什么?

假设你有 10 个 task, 5 个 robot 已经很熟练, 5 个还很烂。

如果 threshold 设得很严格: 熟练的 task 会 trigger 太多 false positive, human 累死。
如果 threshold 设得很宽松: 烂的 task 会漏掉真正的 failure, 出事。

**一个 threshold 不可能 fit all tasks**。

### Solution

Threshold $\alpha_g$ 根据 human intervention ratio $p_H$ 动态调整。

$$\theta_g = a + b \cdot e^{c \cdot p_H}$$

变量含义:
- $\theta_g$: percentile threshold, 决定 $\alpha_g$ 的位置
- $p_H$: human intervention ratio = human intervention timesteps / total timesteps
- $a = 95.2$, $b = -17.7$, $c = -3.2$

$\alpha_g = d_g^{\theta_g}$, 其中 $d_g^{\theta_g}$ 是 validation set 中到 nearest centroid distance 的 top $\theta_g$ percentile 值。

### Intuition

- $p_H$ 高 = robot 表现差 = 需要更多 human help = threshold 降低 (更容易判断为 OOD, 更频繁喊人)
- $p_H$ 低 = robot 表现好 = 不需要那么多 help = threshold 升高 (更不容易判断为 OOD, 少喊人)

这实现了 **human workload 随 robot autonomy 提升而自动降低** 的闭环。

### 为什么是 exponential decay

当 $p_H \to 0$ (robot 很好): $\theta_g \to 95.2 - 17.7 = 77.5$, threshold 还在合理范围
当 $p_H \to 1$ (robot 很烂): $\theta_g \to 95.2 - 17.7 \cdot e^{-3.2} \approx 94.5$, threshold 也不会太极端

Exponential form 保证了 smooth transition, 不会因为 $p_H$ 的小变化导致 threshold 大跳。

论文说这个 function 和这组参数在 simulation (RoboCasa) 和 real-world (Mutex) 都 work, 没有调过。这很 impressive, 说明这个 relationship 有 universal property。

---

## Runtime 时的 Prediction Pipeline

Deployment 时, 对每个 timestep:

1. **World model 预测未来**: 从当前 state, 用 cVAE sample $N=20$ 次, 生成 20 个 possible future trajectories, 每个长度 $L=20$ 步

2. **Failure prediction**: 对每个 predicted future embedding, 计算 failure score。最后 average across 20 个 samples:

$$\bar{F} = \frac{1}{N} \sum_{i=1}^{N} F_\sigma(\hat{z}_{t+1:t+L}^{(i)})$$

$N$ 是 sampling 次数, $F_\sigma(\cdot)$ 是 failure classifier 输出的 score, $\hat{z}^{(i)}$ 是第 $i$ 次 sample 的 future trajectory。

3. **OOD prediction**: 对每个 predicted future embedding, 计算到 nearest centroid 的 distance。Average across 20 个 samples, 和 threshold $\alpha_g$ 比较:

$$\bar{d} = \frac{1}{N} \sum_{i=1}^{N} \min_j \|\hat{z}^{(i)} - \mu_j\|_2$$

如果 $\bar{d} > \alpha_g$, 就是 OOD。

4. **Decision**: 任何一个 predictor trigger, 就喊 human。

### 为什么要 sample 20 次再 average

Single prediction 是 noisy 的, 可能误判。通过 Monte Carlo sampling, 你得到的是 **expected anomaly risk**, 更 robust。

这就像天气预报说 "明天 80% 概率下雨" 比 "明天会下雨" 更 informative, 因为它 quantifies uncertainty。

---

## Multi-Task Policy 架构

Transformer-based, 输入:
- Image observations (workspace + wrist camera, concatenated)
- Robot proprioceptive data (joint positions, velocities)
- Language task goal (task embedding)

输出用 GMM (Gaussian Mixture Model) head:

$$\pi(a|s, g) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(a | \mu_k, \Sigma_k)$$

$K=5$ 是 mixture component 数, $\pi_k$ 是第 $k$ 个 component 的 mixture weight, $\mu_k$ 是 mean, $\Sigma_k$ 是 covariance。

**为什么 GMM 而非 Gaussian**: Action distribution 是 multi-modal 的。比如抓一个杯子, 可以从左边抓也可以从右边抓, 两个 valid action modes。Gaussian 只能 capture single mode, GMM 能 capture multi-modality。

---

## Continual Learning Loop

整个系统是这样循环的:

**Round 1**: Full human supervision (没有 anomaly predictor)
- Human 全程盯着, robot 跑 policy, human 随时纠正
- 收集所有 trajectory + intervention labels
- Train initial anomaly predictors

**Round 2**: Anomaly-guided supervision
- Anomaly predictor 决定什么时候喊人
- 只有 predicted anomaly 时 human 才介入
- 收集新 data, fine-tune policy 和 anomaly predictors
- Update adaptive threshold based on new $p_H$

**Round 3**: 重复 Round 2, robot autonomy 进一步提升, human intervention 进一步减少

**数据更新公式**:
$$\mathcal{D}^{i+1} = \mathcal{D}^i \cup \mathcal{D}'$$

$\mathcal{D}^i$ 是之前所有 round 的 aggregated data, $\mathcal{D}'$ 是当前 round 新收集的 data。

**关键**: World model 训练一次后 freeze, 只有 policy 和 anomaly predictors 持续 fine-tune。这避免了 world model continual training 的 computational cost 和 catastrophic forgetting。

---

## 实验结果: 为什么这个方法 Work

### 1. System Performance (Figure 6)

**Simulation (RoboCasa)**:
- Autonomous performance: +13% (3 rounds 后)
- Combined performance: >95% maintained
- ROHE: 持续改善

**Real-World (Mutex)**:
- Autonomous performance: +45% (比 simulation 改善更大, 因为 initial policy 更弱)
- Combined performance: >95% maintained

**Key insight**: Combined performance 始终 >95%, 说明 runtime monitoring + human intervention 有效地 prevent failure, 即使 autonomous policy 还在学习中。这就是 human-in-the-loop 的价值: 你不需要等 policy 完美了才 deploy, 而是在 deploy 过程中持续 improve。

### 2. vs Baselines (Figure 7)

三个 baseline:
- **MoMaRT** (VAE reconstruction): pixel 变化就误报, false positive 多
- **PATO** (ensemble variance + VAE): VAE 部分常 false negative, ensemble 部分振荡大
- **ThriftyDAgger** (risk Q-function): Q-function 需要大量 data, generalization 差

SIRIUS-FLEET 在 CPP 和 ROHE 上都赢。

**为什么赢**: 
1. Visual world model 的 shared representation 比 task-specific models 更 generalizable
2. Adaptive threshold 自动适应不同 task 的 performance level
3. Failure + OOD 互补, 覆盖更多 anomaly types

### 3. Multi-Task vs Single-Task World Model (Table 1)

Multi-task world model 的 future prediction MSE 比 single-task 低 ~2x:

| Task | Single | Multi |
|------|--------|-------|
| CloseDoor | 3.3e-4 | 1.5e-4 |
| CoffeeSetup | 4.4e-4 | 1.8e-4 |
| ... | ... | ... |

**Intuition**: Multi-task training 让 world model 学到 shared dynamics (比如 "gripper 接近 object 时的 visual change pattern" 是所有 manipulation task 共通的)。这些 shared dynamics 在 single-task training 中无法充分利用。

这和 foundation model 的核心 idea 一致: diverse data 上的 pre-training 产生更 generalizable 的 representations。

---

## Limitations

1. **只适合 quasi-static tasks**: Dynamic tasks (juggling, catching) 的 anomaly 用 teleoperation 难以纠正

2. **World model frozen 的 trade-off**: 如果 deployment environment 和 training data 差异很大, world model prediction accuracy 会下降, 但因为它 frozen, 无法 adapt

3. **Single human monitoring 假设**: Fleet size 很大时, 即使有 anomaly prediction, 一个 human 可能也不够

4. **Small human operator group**: 只有 5 个 operators, 不同 human 的 intervention style 差异可能影响 policy learning

5. **Single embodiment**: 只在 Franka Panda 上实验, cross-embodiment transfer 没验证

---

## Build Your Intuition: 三个 Core Insights

### Insight 1: Shared Representation 是 Scaling 的 Key

Multi-task world model 作为 shared backbone, 各个 task 的 anomaly predictor 都用这个 frozen representation。这让你:
- 只 train 一次 world model (expensive)
- 每个 task 的 anomaly predictor training 极快 (1.5 小时)
- 新 task 加入时不需要 retrain world model

这就是 foundation model paradigm 在 robot monitoring 上的应用。

参考: 
- Foundation models in robotics: https://arxiv.org/abs/2310.08864
- RT-X: https://robotics-transformer-x.github.io/

### Insight 2: Adaptive Threshold 是 Multi-Task Scaling 的 Enabler

Fixed threshold 在 single-task 上 work, 但 multi-task 上会挂, 因为不同 task 的 performance level 不同。

Adaptive threshold 基于 $p_H$ (human intervention ratio) 自动调整, 不需要人工 tune 每个 task 的 threshold。这个 exponential decay function 在 simulation 和 real-world 都 universal, 非常 elegant。

这是 "self-calibrating monitoring system" 的 idea: monitoring system 自己根据 robot 的 performance 调整 sensitivity。

### Insight 3: Failure + OOD 是 Complementary 的 Signal

- OOD: "我见过这个场景没?" (distribution shift detection)
- Failure: "这个动作像不像之前犯过的错?" (failure mode detection)

两者捕捉不同的 anomaly types, 单独用都不够, 组合起来效果好得多。

这和 anomaly detection literature 中的 "unknown unknown" vs "known unknown" 区分类似:
- OOD detects unknown unknowns (novel scenarios)
- Failure detects known unknowns (known failure modes)

参考:
- Anomaly detection survey: https://arxiv.org/abs/2110.14051
- OOD detection in robotics: https://arxiv.org/abs/2203.05864

---

## 和 LLM/VLM 的 Connection

这篇 paper 的 world model 和 GPT/Sora 有哲学上的相似性:

1. **Next-token prediction → Next-frame prediction**: GPT 预测下一个 token, SIRIUS-FLEET 的 world model 预测下一个 latent embedding。都是 autoregressive prediction。

2. **Shared backbone + task-specific heads**: GPT pre-train 后 fine-tune for various NLP tasks, SIRIUS-FLEET 的 world model pre-train 后 freeze, 各 task 的 anomaly predictor 是 task-specific heads。

3. **Scale → Generalization**: Multi-task world model 比 single-task 更 accurate, 印证了 "more diverse data → better representation" 的 scaling law。

但 key difference: SIRIUS-FLEET 的 world model 是 action-conditional 的, 预测的是 "如果执行这个 action, 未来会怎样", 而 GPT/Sora 是 unconditional 或 text-conditional 的。Action-conditioning 对于 robot control 至关重要。

参考:
- Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators
- Genie (interactive environments): https://arxiv.org/abs/2402.15391
- GAIA-1 (autonomous driving world model): https://arxiv.org/abs/2309.17080

---

## 最后: 这篇 Paper 的 Real Contribution

用一句话: **它把 visual world model, multi-task learning, interactive imitation learning, fleet learning 这四个方向 converge 到一个 unified framework, 解决了 "如何 scalable 地 deploy robot fleet with minimal human supervision" 这个实际 problem**。

Technical 上最 novel 的点:
1. Visual world model 作为 multi-task anomaly detection 的 shared backbone
2. 基于 $p_H$ 的 adaptive threshold function
3. Failure + OOD 的 complementary 设计

Experimental 上最 convincing 的点:
1. Simulation (12 tasks) 和 real-world (10 tasks) 都验证
2. >95% combined performance maintained
3. Adaptive threshold 在 sim 和 real 通用一组参数

参考链接:
- Project page: https://ut-austin-rpl.github.io/sirius-fleet
- RoboCasa: https://robocasa.github.io/
- Mutex: https://ut-austin-rpl.github.io/mutex/
- 作者之前的工作 (Model-based Runtime Monitoring, ICRA 2024): https://arxiv.org/abs/2402.19309
- 作者之前的工作 (Robot Learning on the Job, RSS 2023): https://arxiv.org/abs/2306.08232

---

# SIRIUS-FLEET: Multi-Task Interactive Robot Fleet Learning 深度解析

## 1. 核心问题与动机

这篇 paper 解决的是一个非常实际的 deployment 问题: 当你有一个 fleet of robots 运行一个 generalist multi-task policy 时, 如何在 real-world 的 unstructured environments 中保持 safety 和 reliability。传统的 IIL (Interactive Imitation Learning) 方法如 HG-DAgger, ThriftyDAgger 等要么需要 continuous human supervision (human workload 太高), 要么只能处理 single-task setting (无法 scale 到 fleet deployment)。

SIRIUS-FLEET 的 key insight 是: 用一个 visual world model 作为 shared representation backbone, 这样不同 task 的 anomaly predictors 可以共享 learned embeddings, 从而实现 cross-task generalization。同时通过 adaptive decision boundaries, 让 anomaly prediction threshold 随 robot autonomy 提升而动态调整, 实现 human workload 的逐渐降低。

参考链接:
- ThriftyDAgger: https://arxiv.org/abs/2109.08273
- Fleet-DAgger: https://proceedings.mlr.press/v205/hoque23a.html
- RoboCasa: https://arxiv.org/abs/2406.02523 (RSS 2024)
- Mutex: https://openreview.net/forum?id=PwqiqaaEzJ

---

## 2. Problem Formulation 的数学细节

### 2.1 MDP 定义

N 个 robots 运行在 N 个 independent MDPs, 第 i 个 robot 的 MDP 定义为:

$$\mathcal{M}_i = (S, A, \tau, H_i, \mu_i^0, R_i)$$

变量含义:
- $S$: shared state space (所有 robots 共享)
- $A$: shared action space
- $\tau: S \times A \to S$: transition dynamics (shared transition function)
- $H_i$: 第 i 个 robot 的 horizon length (task-specific)
- $\mu_i^0$: 第 i 个 robot 的 initial state distribution
- $R_i: S \times A \to \mathbb{R}$: 第 i 个 task 的 reward function

在 sparse-reward 设置下, $R_i$ 被替换为 goal predicate $g_i: S \to \{0, 1\}$, 这是一个 binary indicator function, 表示 state 是否达到 task goal。

### 2.2 Unified Multi-Task MDP

N 个 MDPs $\{\mathcal{M}_i\}_{i=1}^N$ 被 reformulate 成一个 unified MDP, 学习一个 unified multi-task policy:

$$\pi(a \mid s, g_i)$$

这里的 conditioning on $g_i$ 非常关键 — policy 不仅 conditioned on state $s$, 还 conditioned on task identifier $g_i$ (可以是 language embedding)。

优化目标:

$$\max_\pi J(\pi) = \mathbb{E}_{s_t, a_t \sim \pi, \mu_0}\left[\sum_{t=1}^{H} g_i(s_t)\right]$$

变量含义:
- $s_t, a_t$: 从 policy $\pi$ 和 initial distribution $\mu_0$ 采样得到的 state-action pair
- $g_i(s_t)$: goal predicate 在 timestep $t$ 的 binary output
- $H$: horizon length
- 这个 objective 本质上是最大化 expected cumulative goal achievement

---

## 3. Visual World Model 架构深度解析

这是 SIRIUS-FLEET 的 core technical contribution。World model $\mathcal{W} = (E_\gamma, D_\lambda, T_\psi)$ 由三个 component 组成:

### 3.1 Encoder $E_\gamma$ 和 Decoder $D_\lambda$

**架构**: UNet-based (参考文献 Ronneberger et al. 2015, Ho et al. 2020)

**Encoder 作用**:
$$z_t = E_\gamma(x_t)$$

将 image observation $x_t$ 编码成 latent embedding $z_t$。

**Decoder 作用**:
$$\hat{x}_t = D_\lambda(z_t)$$

从 latent embedding 重构 image, 训练时最小化 image reconstruction L2 loss:

$$\mathcal{L}_{recon} = \|x_t - \hat{x}_t\|_2^2$$

**Hyperparameter 细节** (来自 Table 3):
- input/output channels: 3 (RGB)
- latent channels: 4
- block output channels: [32, 64]
- layers per block: 1
- activation: SiLU
- normalization: GroupNorm with 32 groups
- image output activation: Sigmoid

**为什么用 UNet 而不是标准 CNN encoder-decoder?** UNet 的 skip connections 保留了 spatial information, 这对于 manipulation tasks 中的 fine-grained visual details (比如 object position, gripper alignment) 至关重要。普通的 autoencoder 容易丢失这些 spatial precision。

### 3.2 Transition Model $T_\psi$

**架构**: cVAE (conditional Variational Autoencoder) + Transformer

这是最 interesting 的部分。$T_\psi$ 的 input 是过去 $T$ 个 timestep 的 embeddings:

$$(z_{t-T}, z_{t-T+1}, \dots, z_t)$$

Output 是 future embeddings 的 autoregressive prediction:

$$(\hat{z}_{t+1}, \hat{z}_{t+2}, \dots, \hat{z}_{t+L})$$

其中 $L$ 是 prediction horizon, $T$ 是 history length (论文中 $T=10$, $L=20$)。

**cVAE 的数学结构**:

cVAE 的核心是 learning 一个 stochastic latent space, 而不是 deterministic mapping。这非常关键, 因为 robot manipulation 的 future 是 multi-modal 的 (同一个 state 可以有多种 valid future trajectories)。

cVAE 的 ELBO:

$$\mathcal{L}_{cVAE} = \mathbb{E}_{q_\psi(z'|z_{t-T:t}, c)}[\log p_\psi(z_{t+1:t+L}|z', c)] - D_{KL}(q_\psi(z'|z_{t-T:t}, c) \| p(z'|c))$$

变量含义:
- $q_\psi(z'|z_{t-T:t}, c)$: approximate posterior, conditioned on history 和 context $c$
- $p_\psi(z_{t+1:t+L}|z', c)$: likelihood (decoder)
- $p(z'|c)$: prior (learnable GMM prior with 10 nodes)
- $D_{KL}$: KL divergence

**为什么用 GMM prior?** GMM (Gaussian Mixture Model) with 10 components 可以 approximate 任意 multi-modal distribution, 这对于捕捉 manipulation 中的 multi-modal future behaviors 非常重要。

**Transformer 用于 cVAE 的 encoder, decoder 和 prior network**:
- embed dimension: 512
- number of layers: 6
- number of heads: 8
- embedding dropout, attention dropout, block output dropout: all 0.1
- activation: GeLU
- causal masking: False (因为 prediction 而非 language modeling)
- sinusoidal embedding: False

**关键设计决策**: $T_\psi$ 与 $E_\gamma, D_\lambda$ 在同一个 latent space 上 joint training, 确保 predicted future embeddings 与 encoder 输出的 embeddings 在同一个 representation space 中。

### 3.3 Training Loss 的完整形式

World model 的 total loss:

$$\mathcal{L}_{total} = \mathcal{L}_{recon}(E_\gamma, D_\lambda) + \mathcal{L}_{pred}(T_\psi) + \beta \cdot D_{KL}$$

其中:
- $\mathcal{L}_{recon}$: image reconstruction L2 loss
- $\mathcal{L}_{pred}$: embedding reconstruction loss between predicted $\hat{z}_{t+1:t+L}$ 和 ground truth $z_{t+1:t+L}$
- $D_{KL}$: KL divergence regularization term
- $\beta$: KL weight (standard VAE practice)

### 3.4 Stochastic vs Deterministic Latent Space

论文明确说明用 stochastic latent space 而非 deterministic:

> "We use a stochastic latent space rather than a deterministic one since the stochastic latent space of cVAE supports multiple future sampling and facilitates better prediction"

这意味着在 deployment 时, 可以从 cVAE latent space 中 sample $N$ 次 (论文中 $N=20$), 生成 $N$ 个可能的 future scenarios, 然后对每个 future embedding 分别做 anomaly prediction, 最后 aggregate (average failure score / average distance)。

这种 multi-sample prediction 的 intuition: 单次 deterministic prediction 无法 capture future 的 uncertainty, 而通过多次 sampling 可以 estimate expected anomaly risk, 更 robust。

参考 DayDreamer (Wu et al. 2023): https://arxiv.org/abs/2206.14176

---

## 4. Anomaly Predictors 的技术细节

### 4.1 Failure Prediction

**模型**: Small classifier $F_\sigma$ (训练时间仅 1.5 小时, 说明非常 lightweight)

**Input**: Frozen image embeddings from visual world model

**Labels 来源**: 从 trajectory $\tau = (s_t, a_t, r_t, c_t)$ 中获取, 其中 $c_t$ 标记 human intervention timesteps。Intervention 前的 trajectory segments 被标记为 failures。

**Loss function**: Cross-entropy loss with balanced sampling

$$\mathcal{L}_F = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

变量含义:
- $n$: number of samples in batch
- $y_i \in \{\text{rollout}, \text{failure}, \text{human}\}$: 3-class label (autonomous rollout, failure, human intervention)
- $\hat{y}_i$: predicted probability distribution over 3 classes

**3-class 设计的 intuition**: 区分 "robot 正常 rollout", "即将 failure", "需要 human takeover" 三种状态, 比 binary classification 提供更细粒度的 risk assessment。

**Hyperparameters** (Table 6):
- context length: 10
- embed dimension: 512, layers: 6, heads: 8
- predict on future: True (预测未来而非当前)
- threshold count: 2
- evaluation index: [7, 8, 9] (在 future prediction 的第 7-9 步评估)
- use probability: False (用 logit 而非 probability)

### 4.2 OOD Prediction

这是一个 unsupervised 方法, 不需要训练 additional models:

**Step 1: PCA Dimension Reduction**
- Frozen world model 生成 embeddings from sampled trajectories
- PCA 将 embedding dimension 降到 $l$ (降低计算 cost)

**Step 2: k-means Clustering**
- 对每个 task 做 task-specific k-means clustering
- 得到 $c$ 个 cluster centroids

**Step 3: Distance Calculation**
对新 embedding $z$:
1. PCA 降维
2. 找到 nearest centroid
3. 计算 L2 distance $d$

$$d = \min_{j \in \{1, \dots, c\}} \|z - \mu_j\|_2$$

变量含义:
- $\mu_j$: 第 $j$ 个 cluster centroid
- $d$: embedding $z$ 到 nearest centroid 的 L2 distance

**Step 4: Threshold Comparison**
如果 $d > \alpha_g$, 则识别为 OOD。

### 4.3 Adaptive Decision Boundaries — 最关键的 Innovation

这是论文最 important 的 contribution 之一。

**问题**: 固定 threshold 在 multi-task setting 中失效, 因为不同 task 的 performance 不同, 同一 threshold 对 high-performing task 太严格 (过多 intervention), 对 low-performing task 太宽松 (漏掉 failure)。

**Solution**: 基于 human intervention ratio $p_H$ 动态调整 threshold。

$$\theta_g = a + b \cdot e^{c \cdot p_H}$$

变量含义:
- $\theta_g$: percentile threshold (决定 $\alpha_g$ 的位置)
- $p_H$: human intervention ratio (proxy for policy performance, 低 $p_H$ 表示 robot autonomy 高)
- $a = 95.2$, $b = -17.7$, $c = -3.2$: fitted parameters

**Intuition**: 
- 当 $p_H$ 高 (robot 表现差), $\theta_g$ 降低 → $\alpha_g$ 降低 → threshold 更宽松 → 更多 OOD detection
- 当 $p_H$ 低 (robot 表现好), $\theta_g$ 升高 → $\alpha_g$ 升高 → threshold 更严格 → 更少 OOD detection

Wait, 让我重新理解一下。$\alpha_g = d_g^{\theta_g}$, 其中 $d_g^{\theta_g}$ 是 validation set 中距离 nearest centroid 的 top $\theta_g$ percentile 的 distance。

当 $\theta_g = 95.2$ percentile, 意味着只有距离非常大的 sample (top 4.8%) 才被认为是 OOD。
当 $\theta_g = 77.5$ (假设 $p_H = 1$), threshold 降低, 更容易判断为 OOD。

所以:
- $p_H$ 高 (robot 差) → $\theta_g$ 低 → $\alpha_g$ 低 → more aggressive OOD detection (合理, 因为 robot 差, 更需要 human help)
- $p_H$ 低 (robot 好) → $\theta_g$ 高 → $\alpha_g$ 高 → less aggressive OOD detection (合理, robot 好, 不需要那么多 intervention)

这个 exponential decay function 在 simulation 和 real-world experiments 中都 robust, 使用同一组 hyperparameters, 这是一个很强的 universal property。

**Exponential function 的 form choice**: 
$f(p_H) = a + b \cdot e^{c \cdot p_H}$

当 $p_H \to 0$: $\theta_g \to a + b = 95.2 - 17.7 = 77.5$
当 $p_H \to 1$: $\theta_g \to 95.2 - 17.7 \cdot e^{-3.2} \approx 95.2 - 17.7 \cdot 0.0408 \approx 94.5$

这个设计确保 $\theta_g$ 始终在合理范围 (77.5 - 95.2), 不会出现 extreme values。

### 4.4 Anomaly Predictors at Runtime

Deployment 时的 prediction pipeline:

1. Visual world model 预测未来 $L$ 步的 embeddings
2. 从 cVAE latent space sample $N=20$ 次, 生成 $N$ 个 possible futures
3. 对每个 predicted future embedding:
   - Failure prediction: 计算 failure score
   - OOD prediction: 计算到 nearest centroid 的 distance
4. Aggregate: 
   - Failure: average failure score across futures
   - OOD: average distance, compare against threshold $\alpha_g$

这种 multi-sample aggregation 的 intuition 类似 Monte Carlo estimation, 通过多次采样 estimate expected anomaly probability, 比 single prediction 更 robust。

---

## 5. Multi-Task Policy 架构

**架构**: Transformer-based (following RoboMimic and RoboCasa)

**Inputs**:
1. Image observations (workspace + wrist cameras, concatenated)
2. Robot proprioceptive data
3. Language task goals (task embedding)

**Output**: Gaussian Mixture Model (GMM) action distribution

$$\pi(a|s, g) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(a | \mu_k, \Sigma_k)$$

变量含义:
- $K = 5$: number of GMM modes
- $\pi_k$: mixture weight for mode $k$
- $\mu_k$: mean of mode $k$
- $\Sigma_k$: covariance of mode $k$
- min std: 0.005 (防止 mode collapse)
- std activation: Softplus

**GMM 的 intuition**: Action distribution 是 multi-modal 的, 特别在 manipulation 中 (比如抓取 object 可以从左边或右边)。GMM head 可以 capture 这种 multi-modality, 而 Gaussian head 不行。

**Policy Hyperparameters** (Table 4):
- context length: 20
- embed dimension: 512, layers: 6, heads: 8
- dropout: 0.1 (embedding, attention, block output)
- activation: GeLU
- causal: False
- Image encoder: ResNet18 with ConvFiLM
- pretrained backbone: True
- pool: 64

**Optimizer**: AdamW, initial LR 0.0001, constant with warmup scheduler

---

## 6. Experimental Results 详细分析

### 6.1 Evaluation Metrics 的数学定义

**Combined Policy Performance (CPP)**: 
整体 system success rate under monitoring, 反映 human-robot collaboration 的 effectiveness。

**Return of Human Effort (ROHE)**:

$$\text{Normalized ROHE} = \frac{\mathbb{E}_\tau\left[\sum_{t=0}^{T_\tau} r_t^\tau\right]}{1 + \frac{H}{T}}$$

变量含义:
- $\tau$: trajectory
- $T_\tau$: trajectory length
- $r_t^\tau$: reward at timestep $t$ in trajectory $\tau$
- $H$: total human intervention timesteps
- $T$: total timesteps
- $H/T$: human intervention ratio

**Intuition**: 分子是 expected return, 分母 penalize human intervention effort。目标 maximize performance while minimize human workload。

### 6.2 System Performance Over Time (Figure 6)

**Simulation (RoboCasa)**:
- Autonomous Performance: +13% improvement over deployment rounds
- Combined Policy Performance: >95% maintained
- ROHE: consistently improving

**Real-World (Mutex)**:
- Autonomous Performance: +45% improvement (比 simulation 更显著, 因为 real-world 的 initial policy 更弱, improvement space 更大)
- Combined Policy Performance: >95% maintained
- ROHE: consistently improving

**Key insight**: Combined performance 保持在 95% 以上, 说明 runtime monitoring + human intervention 有效地 prevents failures, 即使 autonomous policy 还在 learning curve 上。

### 6.3 Baseline Comparison (Figure 7)

Baselines:
1. **MoMaRT** (Wong et al. 2022): VAE reconstruction loss for OOD detection
   - Paper: https://arxiv.org/abs/2202.06060
   - 弱点: prone to false positives due to pixel changes

2. **PATO** (Dass et al. 2022): ensemble policy variance + VAE reconstruction for future image goals
   - Paper: https://arxiv.org/abs/2210.02814
   - 弱点: VAE 部分 often false negatives, ensemble 部分 high oscillations and extreme values

3. **ThriftyDAgger** (Hoque et al. 2021): OOD detection + failure detection using risky Q-function values
   - Paper: https://arxiv.org/abs/2109.08273
   - 弱点: risk Q-function data-intensive, generalization errors

**SIRIUS-FLEET consistently outperforms** 在 both CPP 和 ROHE。

**Why SIRIUS-FLEET wins**: 
- Fixed thresholds (baselines) 无法 handle varying task distributions
- SIRIUS-FLEET 的 adaptive threshold 基于 human intervention ratio, 自动 align with robot's evolving autonomy
- Visual world model 的 shared representation 比 task-specific models 更 generalizable

### 6.4 Ablation Studies

#### Ablation 1: Multi-Task vs Single-Task World Model (Table 1)

MSE loss for future latent state prediction:

| Task | Single Task | Multi Task |
|------|-------------|------------|
| CloseDoorSingleHinge | 3.3×10⁻⁴ | 1.5×10⁻⁴ |
| PnPCabToCounter | 3.1×10⁻⁴ | 1.7×10⁻⁴ |
| TurnOnMicrowave | 1.7×10⁻⁴ | 1.3×10⁻⁴ |
| TurnOnSinkFaucet | 3.2×10⁻⁴ | 1.2×10⁻⁴ |
| CoffeeSetupMug | 4.4×10⁻⁴ | 1.8×10⁻⁴ |

**Key finding**: Multi-task world model consistently 给出更 accurate future latent prediction (大约 2x better MSE)。

**Intuition**: Multi-task training 让 world model 学习到 shared dynamics (比如 "接近 object 时 gripper 应该如何移动" 这类 common patterns), 这些 shared dynamics 在 single-task training 中无法充分利用。

#### Ablation 2: Failure + OOD Prediction (Table 2)

Human Intervention Overlap Accuracy:

| Task | SIRIUS-FLEET (OOD only) | SIRIUS-FLEET (Failure only) | SIRIUS-FLEET (Combined) |
|------|------------------------|----------------------------|--------------------------|
| CoffeeSetupMug | 85.1 | 87.0 | 99.4 |
| PnPCounterToCab | 43.1 | 56.2 | 73.7 |
| TurnOffMicrowave | 54.9 | 50.0 | 74.5 |
| TurnOffSinkFaucet | 11.3 | 38.8 | 45.0 |
| OpenDoorDoubleHinge | 42.8 | 63.3 | 78.4 |

**Key finding**: Combined OOD + Failure prediction 显著 outperform 单独使用任何一个。

**Intuition**: 
- OOD prediction 捕捉 "robot 处于 unfamiliar scenario" (distribution shift)
- Failure prediction 捕捉 "robot 正在重复之前 human 纠正过的 failure mode"
- 两者 complementary: OOD 可能不是 failure (只是 novel), failure 可能不是 OOD (在 training distribution 内但 still failing)

#### Ablation 3: Multi-Task vs Single-Task Policy (Figure 8)

Multi-task policy 在大多数 task 上 outperform single-task policy, 特别是 CoffeeSetupMug 和 PnPCounterToSink, single-task policy 给出 0% success rate。

**Intuition**: Multi-task training 通过 knowledge sharing 提升 generalization, 特别是对于 data-scarce tasks, 可以 leverage 其他 task 的 learned representations。

---

## 7. Qualitative Analysis 深度解读 (Section 6.4)

论文做了 very thoughtful 的 qualitative analysis, 比较 SIRIUS-FLEET 和 baselines 在 specific failure modes 上的 detection 能力。

**三个 human intervention 的典型场景**:
1. Robot not aiming at the object it is grasping
2. Robot pauses at grasping
3. Arm too stiff, does not bend down

**方法对比**:

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| SIRIUS-FLEET (Failure) | Captures similar error pattern to human judgment | - |
| SIRIUS-FLEET (OOD) | Aligns with human intervention on "not aiming" and "arm too stiff" | - |
| MoMaRT | - | Prone to false positives due to pixel changes |
| PATO (VAE) | - | Often predicts false negatives |
| PATO (Ensemble) | - | High oscillations, extreme values |
| ThriftyDAgger | - | Risk Q-function generalization errors, cannot reflect task progress |

**Critical insight**: SIRIUS-FLEET 的 failure prediction 和 OOD prediction 各自捕捉 different aspects of anomalies, 与 human risk assessment 对齐得很好。

---

## 8. Continual Learning Pipeline

### 8.1 Three-Round Deployment

**Round 1 (Initial)**:
- Full human supervision (no anomaly predictors trained yet)
- Collect deployment data with human intervention labels
- Train initial policy with 50 demos/task over 1200 epochs
- Collect 100 rollouts/task

**Round 2 & 3**:
- Human supervision only requested when anomaly predicted
- Fine-tune policy on newly collected data (400 epochs)
- Fine-tune failure classifier with latest intervention labels
- Update k-means clustering (expand latent space coverage)
- Adjust adaptive threshold based on $p_H$

### 8.2 Data Flow

$$\mathcal{D}^{i+1} = \mathcal{D}^i \cup \mathcal{D}'$$

其中 $\mathcal{D}'$ 包含 trajectories $\tau = (s_t, a_t, r_t, c_t)$, $c_t$ 标记 human control timesteps。

**Key design**: 
- World model 训练一次后 frozen (不 continual update)
- Policy 和 anomaly predictors 持续 fine-tune
- 这避免了 world model continual training 的 catastrophic forgetting 和 computational cost

---

## 9. Implementation Details 补充

### 9.1 Training Configuration

**Simulation (RoboCasa)**:
- World model training: 50 demos × 20 task suites = 1000 trajectories
- Policy training: 50 demos × 12 task suites = 600 trajectories
- Initial BC policy: 1000 epochs
- Fine-tune per round: 400 epochs
- Evaluation: 2 seeds, 50 trials/task, 600 trials total

**Real-World (Mutex)**:
- World model training: 30 demos × 50 tasks = 1500 trajectories
- Policy training: 30 demos × 10 tasks = 300 trajectories
- Initial BC policy: 2000 epochs
- Fine-tune per round: 800 epochs
- Evaluation: 1 seed, 20 trials/task, 200 trials total

### 9.2 Robot Configuration

- Robot: Franka Emika Panda arm (single embodiment)
- Controller: OSC (Operational Space Control)
- Action space: 7D (x-y-z position, yaw-pitch-roll orientation, gripper)
- Observation: workspace camera + wrist camera

---

## 10. Limitations 和 Future Work

### 论文 explicitly 提到的 limitations:

1. **Quasi-static manipulation only**: SIRIUS-FLEET 最适合 quasi-static tasks, 对 dynamic tasks (比如 catching, juggling) 可能 challenging, 因为 anomalies 难以用 teleoperation 纠正。

2. **Small human operator group**: 只用了 5 个 human operators, 需要 large-scale human studies 理解 diverse human interventions 的影响。

3. **Single embodiment**: 只在 Franka Panda 上实验, 需要 cross-embodiment 扩展。

### 我自己的 additional observations:

1. **World model frozen 的 trade-off**: 虽然 frozen world model 避免 continual training cost, 但也意味着 world model 无法 adapt 到 deployment 时遇到的新 visual scenarios。如果 deployment environment 与 training data 差异很大, world model 的 prediction accuracy 可能下降。

2. **Exponential decay function 的 universal applicability**: 虽然 $a=95.2, b=-17.7, c=-3.2$ 在 simulation 和 real-world 都 work, 但这个 function form 和 parameters 是否能 generalize 到更 diverse 的 task domains (比如 surgical robotics, autonomous driving) 还有待验证。

3. **OOD detection 的 PCA + k-means pipeline**: PCA 降维可能丢失 important variance directions, 特别是对于 high-dimensional embeddings。可以考虑用 learned dimensionality reduction (比如 contrastive learning) 替代 PCA。

4. **Single human monitoring assumption**: 论文假设一个 human 可以 monitor 整个 fleet, 但当 fleet size 很大时, 即使有 anomaly prediction, human workload 可能仍然很高。可能需要 hierarchical monitoring structure。

---

## 11. 与 Related Work 的 positioning

### 11.1 在 Robot Fleet Learning 中的位置

参考关键 papers:
- Fleet-DAgger: https://proceedings.mlr.press/v205/hoque23a.html
- OpenBot-Fleet: https://arxiv.org/abs/2402.01431
- Robot Fleet Learning via Policy Merging: https://arxiv.org/abs/2402.10350

SIRIUS-FLEET 是第一个 combine runtime monitoring + continuous policy updates + multi-task setting 的 framework。Prior fleet learning 工作主要关注 resource allocation, decentralized learning, 或 system management, 但没有 address multi-task policy improvement through human interaction。

### 11.2 在 Interactive Imitation Learning 中的位置

参考:
- HG-DAgger: https://arxiv.org/abs/1810.05712
- Robot Learning on the Job: https://arxiv.org/abs/2306.08232 (RSS 2023)
- Model-based Runtime Monitoring: https://arxiv.org/abs/2402.19309 (ICRA 2024)

SIRIUS-FLEET 与 prior IIL 工作的主要区别:
1. **Multi-task**: Prior IIL 主要 single-task, SIRIUS-FLEET 第一个 multi-task IIL
2. **Visual world model backbone**: Prior work 用 task-specific dynamics models, SIRIUS-FLEET 用 pre-trained visual world model
3. **Adaptive thresholds**: Prior work 用 fixed thresholds 或 simple target intervention ratio, SIRIUS-FLEET 用 exponential decay function based on $p_H$

### 11.3 与 World Model 研究的 connection

参考:
- Sora (Video generation models as world simulators): https://openai.com/research/video-generation-models-as-world-simulators
- Genie: https://arxiv.org/abs/2402.15391
- GAIA-1: https://arxiv.org/abs/2309.17080
- DayDreamer: https://arxiv.org/abs/2206.14176

SIRIUS-FLEET 的 visual world model 受到这些工作启发, 但有 key difference: SIRIUS-FLEET 的 world model 专门为 robot manipulation 设计, 关注 fine-grained visual details 和 action-conditioned prediction, 而 Sora/Genie 等更关注 general video generation。

---

## 12. Technical Takeaways for Building Intuition

1. **Shared representation > Task-specific models**: Multi-task world model 的 MSE 比 single-task 低 2x, 证明 shared representation 的 power。这呼应了 foundation model 的 core idea: 在 diverse data 上 pre-train 的 representations 可以 transfer 到 downstream tasks。

2. **Stochastic > Deterministic for future prediction**: cVAE 的 stochastic latent space 允许 multi-sample prediction, 更好地 capture future uncertainty。这在 deployment 时通过 $N=20$ 次 sampling 来 estimate expected anomaly risk。

3. **Adaptive > Fixed thresholds**: 固定 threshold 无法 handle evolving robot performance。Exponential decay function $\theta_g = a + b \cdot e^{c \cdot p_H}$ 提供 principled way 来 dynamically adjust detection sensitivity。

4. **OOD + Failure > Either alone**: 两个 predictor 是 complementary 的, OOD 捕捉 distribution shift, Failure 捕捉 known failure modes, 结合使用 accuracy 从 ~50% 提升到 ~75-99%。

5. **Frozen backbone + Fine-tunable heads**: World model frozen, 只 fine-tune policy 和 anomaly predictors, 这在 computational efficiency 和 continual learning 之间取得 good balance。

6. **Multi-task policy > Single-task policy**: Multi-task training 通过 knowledge sharing 提升 generalization, 特别是 data-scarce tasks 受益最大。

---

## 13. 项目链接

- **Project website**: https://ut-austin-rpl.github.io/sirius-fleet
- **RoboCasa**: https://robocasa.github.io/
- **Mutex**: https://ut-austin-rpl.github.io/mutex/
- **MimicGen**: https://mimic-gen.github.io/
- **RoboMimic**: https://robomimic.github.io/

这篇 paper 的 core contribution 在于将 visual world model 作为 multi-task fleet learning 的 shared backbone, 配合 adaptive anomaly prediction thresholds, 实现了 scalable 的 human-in-the-loop robot deployment。从 technical perspective, 它是 IIL, fleet learning, 和 world model 三个方向 的 nice convergence, 展示了如何将 recent advances in foundation models 和 generative models apply 到 practical robot deployment 问题。
