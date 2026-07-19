---
source_pdf: ACCELERATING GOAL-CONDITIONED REINFORCEMENT LEARNING ALGORITHMS AND RESEARCH.pdf
paper_sha256: 6c1b63f7389ca4ee31476c62259f56ea673cb2cb657fcfafc1ec401f94c94fd6
processed_at: '2026-07-17T23:17:10-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# JaxGCRL: 加速 Goal-Conditioned Reinforcement Learning 研究

非常好的 paper,Andrej! 这篇工作的核心 motivation 是把 self-supervised RL 的研究速度从 "hours/days" 压缩到 "minutes",让 GCRL 研究像 vision/NLP 那样可以快速 iterate。下面我尽量详细拆解。

---

## 1. Big Picture: 为什么这件事有意义

Self-supervised learning 在 NLP (BERT, GPT) 和 vision (SimCLR, MAE) 上已经取得革命性突破,但 RL 中的 self-supervision 有两种 fundamentally 不同的形态:

- **Fixed-dataset self-supervision**: 从 pre-collected dataset 中学习 representation/model (类似 offline RL + world model 路线)
- **Online self-supervision**: agent 在 environment 中无 reward 交互,自己收集数据并 relabel (本文路线)

JaxGCRL 走第二条路,聚焦在 **goal-conditioned RL (GCRL)** —— agent 学习"达到任意 state"的能力。这等价于学习一个 universal policy,可以快速 adapt 到 downstream task。这个 setting 对应 generalist robot 很重要,因为每个 state 都可以看作一个 task。

**Self-supervised RL 的 bottleneck**:
1. Environment simulation 慢 (CPU-bound)
2. 算法不稳定 (CRL 原实现容易 collapse)

JaxGCRL 用 GPU acceleration + 稳定化的 CRL 同时解决这两个问题,实测达到 **22× 加速**。

参考链接:
- JaxGCRL repo: https://github.com/MichalBortkiewicz/JaxGCRL
- 原 CRL paper (Eysenbach et al. 2022): https://arxiv.org/abs/2201.07757
- Brax: https://github.com/google/brax

---

## 2. 数学形式化:把 Goal-Reach 视作 Inference

### 2.1 设定

Controlled Markov Process (CMP): $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, p_0, \gamma)$

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $p(s_{t+1}|s_t, a_t)$: transition dynamics
- $p_0(s_0)$: initial state distribution
- $\gamma$: discount factor

Goal $g \in \mathcal{S}$ 就是某个 state,policy $\pi(a|s,g)$ 接收 state + goal,输出 action。

### 2.2 关键 trick: 把 reward 定义成 transition probability

Paper 定义了一个特殊的 goal-conditioned reward:
$$r_g(s_t, a_t) \triangleq (1-\gamma)\gamma \, p(s_{t+1}=g \mid s_t, a_t)$$

变量解释:
- $g$: 目标 state
- $s_t, a_t$: 当前 state 和 action
- $(1-\gamma)\gamma$: 归一化常数,保证几何分布归一化
- $p(s_{t+1}=g|s_t,a_t)$: 一步到达 g 的概率

### 2.3 Discounted state visitation distribution
$$p_\gamma^\pi(s^+ \mid s, a) \triangleq (1-\gamma)\sum_{t=0}^{\infty}\gamma^t \, p_t^\pi(s^+ \mid s, a)$$

变量解释:
- $p_t^\pi(s^+|s,a)$: 在 policy $\pi$ 下,从 $(s,a)$ 出发 $t$ 步后到达 $s^+$ 的概率
- $(1-\gamma)$: 让几何级数归一化的系数
- 整个表达式等价于: $T \sim \text{Geom}(1-\gamma)$ 时, $T$ 步后 state 的分布

### 2.4 Q-function = Probability 的证明 (Appendix D.1)

核心定理:
$$Q_g^\pi(s,a) = p_\gamma^\pi(g \mid s, a)$$

证明思路:
$$Q_g^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_g(s_t,a_t) \mid s_0=s, a_0=a\right]$$

把 reward 定义代入,加上 Markov 性质,逐项展开后正好得到 discounted visitation distribution。

**Intuition**: Q-function 在这个特殊 reward 定义下,直接编码了 "从当前 state-action 出发,未来访问 goal state 的 discounted 概率"。这就把 value learning 变成了 probability estimation 问题,可以用 contrastive classification 直接学,不需要 TD-learning 的 bootstrap。

### 2.5 总体目标 (Eq. 1)
$$\max_\pi \mathbb{E}_{p_0(s_0)p_\mathcal{G}(g)\pi(a_0|s_0,g)}\left[p_\gamma^\pi(g \mid s_0, a_0)\right]$$

含义: 在初始 state 分布和 goal 分布下,最大化"按 policy 执行 action 后访问 goal 的 discounted 概率"。

---

## 3. Contrastive RL (CRL) 算法详解

### 3.1 Critic 学习

Critic $f(s,a,g)$ 近似 $Q_g^\pi(s,a)$,通过 contrastive classification 学习。

**Energy function** (paper 主推 L2):
$$f_{\phi,\psi}(s,a,g) = -\|\phi(s,a) - \psi(g)\|_2$$

变量解释:
- $\phi(s,a)$: state-action encoder,输出 embedding
- $\psi(g)$: goal encoder,输出 embedding
- 负号: 距离越小,similarity 越大 (paper 中也有写成 $\|\cdot\|_2$ 然后用负号在 loss 外面)

### 3.2 Symmetric InfoNCE Loss (Eq. 中间)

$$\min_{\phi,\psi} \mathbb{E}_\mathcal{B}\left[-\sum_i \log\frac{e^{f(s_i,a_i,g_i)}}{\sum_j e^{f(s_i,a_i,g_j)}} - \sum_i \log\frac{e^{f(s_i,a_i,g_i)}}{\sum_j e^{f(s_j,a_j,g_i)}}\right]$$

- 第一项 (forward): 给定 $(s_i, a_i)$,在 batch 内的多个 $g_j$ 中分辨出真正来自同一 trajectory 的 $g_i$
- 第二项 (backward): 给定 goal $g_i$,在多个 $(s_j, a_j)$ 中分辨出对应它的 $(s_i, a_i)$
- 对称化让 representation 双向都对齐 (类似 CLIP)

**Sampling 策略**: 每个 batch $(s_i, a_i, g_i)$,其中 $g_i$ 从包含 $(s_i, a_i)$ 的 trajectory 的未来 state 中采样。这就是 hindsight relabeling 的核心 —— "你达到了什么 state,那个 state 就是你的 goal"。

### 3.3 Policy 学习 (Eq. 3)

DDPG-style policy extraction:
$$\max_\theta \mathbb{E}_{p(s,a)p(g|s,a)\pi_\theta(a'|s,g)}\left[f_{\phi,\psi}(s, a', g)\right]$$

变量解释:
- $\pi_\theta(a'|s,g)$: deterministic policy (DDPG 风格)
- $p(s,a)$: replay buffer 中的 state-action 分布
- $p(g|s,a)$: 从 $(s,a)$ 的未来 trajectory 中采样 goal

**关键 insight**: Paper 在 Appendix C 中比较了两种 goal sampling:
- 从同 trajectory 未来采样 (本文默认)
- 从 replay buffer 随机采样 (原 CRL 做法)

实验发现 $\alpha=0$ (只从同 trajectory 采样)效果更好。这与原 CRL paper 不同,可能是 online setting 下的差异。

### 3.4 Logsumexp Regularization

每个 contrastive loss 都加上:
$$\mathcal{L}_{\text{logsumexp}} = \beta \cdot \log\sum_j e^{f(s_i,a_i,g_j)}$$

作用: 防止 critic 输出整体漂移 (因为 InfoNCE 只关心相对值)。Paper 中 $\beta = 0.1$。如果没有这一项,InfoNCE 性能会显著下降 —— 这是从 Eysenbach et al. 2022 沿用过来的稳定化 trick。

---

## 4. JaxGCRL 系统设计:22× 加速从哪来

### 4.1 加速来源

| 组件 | 原实现 | JaxGCRL |
|------|--------|---------|
| Environment | MuJoCo (CPU) | BRAX / MuJoCo MJX (GPU) |
| Replay buffer | CPU 内存 | GPU 上的 JIT-compiled buffer |
| Parallel actors | 4-32 CPU threads | 1024 GPU-threaded envs |
| Data transfer | CPU↔GPU PCIe | 全部在 GPU 上 |
| Compilation | Eager | JIT + operator fusion |

**实测**: Ant 环境,UTD=1:16,达到 16,500 env steps/sec,10M steps 只需 10 分钟。

### 4.2 环境列表 (Table 1)

| Environment | Goal distance | Termination | Brax pipeline |
|---|---|---|---|
| Reacher | 0.05 | No | Spring |
| Half-Cheetah | 0.5 | No | MJX |
| Pusher Easy/Hard | 0.1 | No | Generalized |
| Humanoid | 0.51 | Yes | Spring |
| Ant | 0.5 | Yes | Spring |
| Ant Maze (3种) | 0.5 | Yes | Spring |
| Ant Soccer | 0.5 | Yes | Spring |
| Ant Push | 0.5 | Yes | MJX |

**难度梯度**: Reacher 简单 (2D, 2-segment arm) → Ant Push 复杂 (要 push box 出路,错了就无解,需要 exploration + long-horizon reasoning)。

参考链接:
- MuJoCo MJX: https://github.com/google-deepmind/mujoco_mjx
- BRAX: https://github.com/google/brax
- Gymnax: https://github.com/RobertTLange/gymnax

---

## 5. 实验结果详解

### 5.1 Baseline 对比 (Section 5.2, Fig. 3)

**对比方法**: CRL vs SAC vs SAC+HER vs TD3 vs TD3+HER vs PPO

**核心发现**:
- CRL 在大多数环境中胜出,包括高维 (Humanoid) 和 exploration-hard (Pusher, Ant Push) 任务
- HER 帮助 TD3 和 SAC 提升 (符合预期,因为 sparse reward)
- PPO 在 sparse reward 下表现差 (on-policy + sparse reward = 不友好)

**性能 metric**:
- **Success rate**: episode 中是否至少一次达到 goal
- **Time near goal**: 在 goal 附近停留时间比例

注意: 两个 metric 不一定一致 —— 有的方法 success rate 高但 time near goal 低 (比如 DPO)。这暗示 policy "冲到 goal 就走了" vs "稳定停在 goal" 的差异。

### 5.2 Contrastive Objectives 对比 (Section 5.3, Fig. 4)

测试了 10 种 contrastive losses,聚合结果:

| Loss 类型 | 表现 |
|-----------|------|
| InfoNCE family | 最好 (symmetric, fwd, bwd 类似) |
| DPO | success rate 好,但 time near goal 差 |
| NCE-binary, Forward-Backward, IPO, SPPO | 最差 |

**Loss 数学定义** (Appendix A.2):

**DPO** (Direct Preference Optimization):
$$\mathcal{L}_{\text{DPO}} = -\sum_{i,j}\log\sigma\left[f(s_i,a_i,g_i) - f(s_i,a_i,g_j)\right]$$

直接 drive positive 和 negative 的 score 差距变大,不加额外正则化。

**IPO** (Identity Preference Optimization):
$$\mathcal{L}_{\text{IPO}} = \sum_{i,j}\left[(f(s_i,a_i,g_i) - f(s_i,a_i,g_j)) - 1\right]^2$$

强约束 positive 和 negative 差距恰好为 1。

**SPPO** (Self-Play Preference Optimization):
$$\mathcal{L}_{\text{SPPO}} = \sum_{i,j}\left[(f(s_i,a_i,g_i)-1)^2 + (f(s_i,a_i,g_j)+1)^2\right]$$

强约束 positive score = 1, negative score = -1。

**Intuition**: DPO 系列来自 LLM alignment 域。在 RL 这里,过强的约束 (IPO, SPPO) 会破坏 critic 表达 visitation probability 的能力 —— 因为 probability 的绝对值有意义,而不只是相对值。InfoNCE 留下了更大的自由度。

### 5.3 Energy Functions 对比 (Section A.3, Fig. 10, 11)

5 种 energy function:
- Cosine: $\frac{\langle\phi,\psi\rangle}{\|\phi\|_2\|\psi\|_2}$
- Dot: $\langle\phi,\psi\rangle$
- L1: $-\|\phi-\psi\|_1$
- L2: $-\|\phi-\psi\|_2$
- L2 w/o sqrt: $-\|\phi-\psi\|_2^2$

**关键发现**:
- L2 整体最好,尤其在 time near goal 上显著领先
- Cosine 最差 —— 因为 cosine 丢失了 magnitude 信息,而 Q-value (probability) 是有 magnitude 的
- L2 w/o sqrt 性能下降 —— 因为不再满足 triangle inequality,而 quasimetric / metric 性质对 temporal representation 很重要 (Myers et al. 2024b)

参考链接:
- Quasimetric RL (Wang et al.): https://arxiv.org/abs/2305.17684
- Contrastive Successor Features (Myers et al.): https://arxiv.org/abs/2406.17098

### 5.4 Architecture Scaling (Section 5.4, Fig. 5, 6)

测试了 width (256, 512, 1024) 和 depth (1-6 层) 的组合。

**发现**:
1. Width + depth 都增加时,性能提升
2. Width=1024 时,depth 再增加不再有提升 (saturation)
3. **Layer Normalization 是关键** (Fig. 6): 在大网络 (1024 宽,4 层) 上,加 LN 后能持续学习,不加 LN 会在某个点 saturate

这与 Nauman et al. 2024b ("Bigger, Regularized, Optimistic") 的观察一致 —— RL 模型 scaling 远比 supervised learning 复杂,需要 normalization 配合。

### 5.5 Data Scaling (Section 5.5, Fig. 7)

300M steps (vs 默认 50M):
- L2 + InfoNCE 在 locomotion 任务上最佳
- Dot product 在 manipulation (Ant Soccer) 上最佳
- Ant Soccer 和 Ant Big Maze 即使在 300M steps 后 success rate 仍只有 ~40% —— 说明 CRL 在长 horizon manipulation 上仍有空间

### 5.6 UTD Ratio (Section 5.6, Fig. 8)

测试了 1:1, 1:8, 1:16, 1:24, 1:32, 1:48 UTD ratio。

**反直觉发现**: 提高 UTD 只在 Pusher Hard 上明显帮助,其他环境不提升甚至下降。

**Intuition**: 高 UTD 在 sample-efficient learning 上有用,但 CRL 已经在用 hindsight relabeling 提供丰富 signal,过度 update 可能造成 overfitting / representation collapse。这也意味着 JaxGCRL 在更低 UTD 下更快 (1:48 时比 1:16 还快 3 倍)。

参考链接:
- Nauman et al. "Overestimation, Overfitting, and Plasticity": https://arxiv.org/abs/2403.00514
- Nauman et al. "Bigger, Regularized, Optimistic": https://arxiv.org/abs/2405.16158

---

## 6. Pseudocode (Appendix F)

```
Algorithm 1: Contrastive RL
1. Input: contrastive loss L_Critic, energy function f
2. Initialize φ, ψ, π, empty replay buffer D
3. repeat
4.   in parallel over environments:
5.     Observe state s, sample action a ~ π(s, g)
6.     Execute a in environment
7.     Observe next state s', done d
8.     Append (s, a, s') to current trajectory
9.     if s' terminal:
10.      Reset env, sample new goal
11.      Store trajectory in D, start new one
12.  for j = 1..num_updates:
13.    Sample batch B from D (states, actions, future goals)
14.    Update critic:
        (φ,ψ) ← (φ,ψ) - α∇[L_Critic(B;φ,ψ) + β L_logsumexp(B;φ,ψ)]
15.    Update policy:
        π ← π - α∇[L_Actor(B;φ,ψ,π)]
16. until convergence
```

实现要点:
- 全部 JIT-compiled 在 GPU 上
- 1024 parallel envs 同时跑 rollout
- Replay buffer 在 GPU 上
- 每 step 同时做 1 个 critic update + 1 个 actor update (UTD=1:16 时是每 16 个 env steps 做 1 个 update step)

---

## 7. Limitations & 我的思考

### Paper 自述 limitations
1. GCRL 难以表达 "非单 state" 的 goal (比如 "把红色方块放到蓝色方块上" 这种 compositional goal)
2. 假设 full observability
3. Goal 从已知 distribution 采样 —— 真实世界不知道哪些 state 是 reachable / desirable 的
4. 只研究 online setting,没研究 offline GCRL

### 我额外想到的几个点

**A. Exploration 仍然未解**: Ant Push 这类需要探索 (push box 错方向就无解) 的任务,即使 50M steps,success rate 也不高。CRL 用 hindsight relabeling 隐含了 "你达到的 state 就是好 goal",但对于 "你必须先做 X 才能做 Y" 的 bottleneck state,exploration 还是 hard。可能需要 intrinsic motivation (curiosity, empowerment) 叠加。

**B. Goal distribution 的影响**: Paper 假设 goal 从已知分布采样。如果 goal distribution 是 non-uniform (比如 ant 在 maze 中,某些 goal 概率高),policy 会偏向高频 goal。这跟 LLM RLHF 中的 reward distribution 有点像。

**C. 与 Diffusion Policy 的结合**: 现在 manipulation 主流是 diffusion policy。CRL 提供 Q-function (probability),可以用来 guide diffusion sampling。这可能是 self-supervised robot learning 的下一个突破点。

**D. Hierarchical GCRL**: 当前 CRL 是 flat policy。Long-horizon 任务 (Ant Big Maze) 上 flat policy 难学。可以叠加 subgoal generation —— 比如用 contrastive representation 做 option discovery (DIAYN, METRA 路线)。

**E. 跟 Foundation Models 的关系**: Paper introduction 提到 self-supervised RL 可能解决 foundation model 的 counterfactual reasoning 和 long-horizon planning 问题。这个 vision 很宏大。如果 CRL 能 scale 到 pixel-based + real robot,可能就是 robot foundation model 的核心算法。

**F. Successor Features 视角**: CRL 学的 critic $f(s,a,g) \propto p_\gamma^\pi(g|s,a)$ 本质上是 successor representation 的 goal-conditioned 版本。$Q_g^\pi(s,a) = \sum_g \psi(g) \cdot \phi_g(s,a)$ 这种分解可能让 zero-shot transfer 更自然。参考 Barreto et al. 2022 的 successor features。

参考链接:
- DIAYN: https://arxiv.org/abs/1802.06070
- METRA: https://arxiv.org/abs/2405.20152
- Successor Features: https://arxiv.org/abs/1606.05312
- Stabilizing Contrastive RL: https://arxiv.org/abs/2306.12267

---

## 8. 对你 (Karpathy) 可能的特别相关点

考虑到你的背景,几个值得思考的方向:

**A. Self-supervised RL vs LLM self-supervised**: LLM 的 next-token prediction 是 "predict future token given context",CRL 是 "predict future state given current state-action"。结构上类似,但 RL 多了 action conditioning 和 discount factor。如果 RL agent 在 environment 上做 "next-state prediction" 而不是 "next-token prediction",这本质上是 world model + planning。能否把 LLM 的 scaling law 直接搬到 CRL 上?JaxGCRL 让这种实验变得可行。

**B. Micrograd-style intuition**: CRL 的核心 trick 是把 Q-learning 重写成 classification。传统 Bellman backup $\hat{Q} = r + \gamma \max Q(s',a')$ 是 regression with target network,容易 over-estimation。CRL 直接 classify "这个 goal 是不是 future",避免了 bootstrap,梯度更稳定。从 optimization 角度看,这是个更"clean"的 supervision signal。

**C. 实时调试的可能性**: 22× 加速意味着 researcher 可以 interactive debug —— 改一个 hyperparameter,几分钟看到 10 seeds 结果。这跟 PyTorch 早期 "immediate execution" 的开发体验类似。可能彻底改变 RL research 的 workflow。

**D. Eysenbach 路线的总体**: Benjamin Eysenbach 的研究路线一直在 push "RL as classification"。从 C-Learning (2021) → Contrastive RL (2022) → Quasimetric (2023) → Contrastive Successor Features (2024b) → Stabilizing CRL (2024)。JaxGCRL 是这条路线的 infrastructure 实现。可以关注他组的后续工作,可能很快会有 pixel-based + real robot 的版本。

参考链接:
- Eysenbach 主页: https://www.cs.princeton.edu/~beysenbach/
- C-Learning: https://arxiv.org/abs/2102.08403
- Contrastive Difference Predictive Coding: https://arxiv.org/abs/2306.1193

---

## 9. 总结

JaxGCRL 的核心贡献:
1. **Infrastructure**: 把 GCRL 的 experiment time 从 hours 压到 minutes
2. **Benchmark**: 8 个 GPU-accelerated 环境,从 easy 到 long-horizon exploration
3. **Empirical study**: 系统性测试了 CRL 的 design choices —— L2 + symmetric InfoNCE + LN + 中等 UTD 是最佳组合

**对未来研究的启示**:
- GCRL 可以像 NLP/vision 一样快速 iterate
- CRL 在长 horizon + exploration hard 任务上仍有 space
- Architecture scaling + normalization 是 RL scaling 的关键
- 数据规模提升 6× 仍不够,CRL 没饱和

如果你打算进入 self-supervised RL,JaxGCRL 是个非常合适的起点 —— 上手快,实验便宜,可以快速验证直觉。下一步突破可能来自 (a) 把 CRL 跟 diffusion policy 结合,(b) 用 hierarchical structure 处理 long-horizon,(c) 把 self-supervised RL 真正搬到 real robot 上。

希望这个讲解帮你 build 起对 GCRL / CRL 的 intuition!如果对某个具体细节想深入,可以指出来我再展开。
