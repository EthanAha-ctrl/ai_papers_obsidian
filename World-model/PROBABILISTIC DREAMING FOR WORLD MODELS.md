---
source_pdf: PROBABILISTIC DREAMING FOR WORLD MODELS.pdf
paper_sha256: 0c2abe2a14c471f7d0e2a4155e3210736c08572749e1f9b3f596091169857999
processed_at: '2026-08-06T06:27:24-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 Paper

好嘞 Andrej，我换个频道，把这篇 paper 当跟一个聪明的同事喝咖啡时讲清楚。

---

## 一句话总结

**Dreamer 让 agent "做梦" 学 RL，但做梦的方式有点傻——只做一条梦、而且假设未来是单峰的。这篇 paper 想让 agent 同时做 K 条梦、保留多个互斥的假设，结果发现：做梦多了确实更稳，但主动去"剪枝梦境"反而把 agent 带沟里去了。**

---

## 1. 先讲清楚 Dreamer 在干嘛

普通 RL 是 agent 跟环境交互，拿 reward 学 policy。问题：sample inefficiency，要几百万步才能学会个 Atari。

Dreamer 的 idea：**别跟真实环境死磕，自己学个"世界模型"在脑子里演练**。就像你学开车不用真上路，闭上眼想象前面有辆卡车该咋办。

具体怎么做梦？学一个 latent dynamics model $p(z_{t+1} \mid h_t, z_t, a_t)$，然后在 latent space 里 rollout 个 16 步、50 步 trajectory，在这些 imagined trajectory 上跑 actor-critic 更新 policy。真实环境只需要收集少量 data 不断修正 world model 就行。

**收益**：1 step real data → 20 steps imagined training，sample efficiency 暴涨。

Dreamer V3 用的是 **discrete categorical latents**（32 个 class，每个 32 类），因为离散分布天然支持多模态。但代价是梯度不光滑、表示稀疏。

---

## 2. 这篇 paper 抓住的两个痛点

### 痛点 A：做梦只做一条

Dreamer rollout 时从 learned distribution $p(z_t \mid h_t)$ 里只 sample 一个 $z_t$，然后沿着这一条路往下推。

这就像你学开车，每次只想象一种"前面有辆卡车"的场景。但现实里可能有"卡车左转"、"卡车右转"、"卡车急刹"——你只练一种，policy 永远见不到另外的分支。

数学上：learned distribution 是个 $\mathcal{N}(\mu, \sigma)$，你只 sample mean 附近的点，永远摸不到 tail。而 RL 里最 informative 的 transitions（rare events, adversary 突变）恰恰在 tail。

### 痛点 B：单峰 Gaussian 把"左"和"右"平均成"中间"

这是 model-based RL 的经典坑，叫 **mode collapse** 或 **averaging catastrophe**。

设想一个 predator 在你前面，50% 概率往左扑，50% 往右扑。真实未来是 bimodal（两个 mode：左 / 右）。

如果你用 unimodal Gaussian $\mathcal{N}(\mu, \sigma^2)$ 去拟合这个 distribution，最优拟合是 $\mu = 0$（中间），$\sigma$ 巨大。然后你的 world model 预测 predator 会出现在"正前方稍偏中"——这个位置在真实环境里**从来没发生过**。

policy 学到躲"正前方稍偏中"的 predator，结果真实 predator 不是左就是右，policy 直接懵逼。

paper 里描述 BaseDreamer 的行为："freeze momentarily"——就是 policy 看到 predator 突然切换策略，自己脑补的"中间路径"不灵了，瞬间僵住不知道往哪躲。

Dreamer V3 的解法是 categorical latents，自然多模态。但这篇 author 想保留 Gaussian 的 smooth gradient，所以另寻他路。

---

## 3. 三个 Innovations，逐个人话翻译

### Innovation 1：Particle Filter——做 K 条并行的梦

**人话**：与其 sample 一个 $z_t$，不如同时 sample K 个 $z_t^1, z_t^2, \ldots, z_t^K$，每个独立往下 rollout。K 条梦境并行。

```
原 Dreamer：
  z_t → z_{t+1} → z_{t+2} → ... → z_{t+T}
  （1 条梦）

ProbDreamer：
  z_t^1 → z_{t+1}^1 → ... → z_{t+T}^1   （梦 1：predator 往左）
  z_t^2 → z_{t+1}^2 → ... → z_{t+T}^2   （梦 2：predator 往右）
  z_t^3 → z_{t+1}^3 → ... → z_{t+T}^3   （梦 3：noise）
  ...
  z_t^K → z_{t+1}^K → ... → z_{t+T}^K
  （K 条梦）
```

**数学上发生了什么变化**：原来 belief 是 Gaussian $\mathcal{N}(\mu, \sigma^2)$（unimodal），现在变成 empirical distribution：

$$\hat{p}(z_t) = \frac{1}{K} \sum_{k=1}^K \delta(z_t - z_t^k)$$

这是个 Dirac delta mixture。K=1 时退化成单点，K→∞ 时可以逼近任意分布。

**关键 insight**：每个 single-step transition 还是 Gaussian（smooth gradient 保留），但 outer-level 是 non-parametric mixture（multi-modality 保留）。两层 hybrid，鱼和熊掌兼得。

严格讲这不是完整 particle filter——没有 importance weighting 也没有 resampling。更像 "K-parallel rollouts"。但效果上达到了"维持多个 competing hypotheses"的目的。

### Innovation 2：Latent Beam Search——每个 particle 再 branch N 个 action

**人话**：每个 particle 在每个 time step 从 policy sample N 个候选 action，看哪个走得最好。

```
particle 1:
  ├─ action A → z_{t+1}^{1,A} → ...
  ├─ action B → z_{t+1}^{1,B} → ...
  └─ action C → z_{t+1}^{1,C} → ...

particle 2:
  ├─ action A → z_{t+1}^{2,A} → ...
  └─ action B → z_{t+1}^{2,B} → ...
```

总共 K × N 条 branches。这是借 MuZero 的思路（[Schrittwieser 2020](https://www.nature.com/articles/s41586-020-03051-4)）。

**区别**：MuZero 是 inference 时 search（一次决策一次 MCTS）；这里是 training 时 search（imagination 阶段做 branch）。

### Innovation 3：Free Energy Pruning——用 value + uncertainty 给 branches 打分

**人话**：想象时没有 ground-truth 观测，没法用 likelihood 判断哪个 branch 更可能真实。那用啥打分？

作者借了 Friston 的 free energy principle（[Friston 2010](https://www.nature.com/articles/nrn2787)）的名字，但实际公式是：

$$F_t^k = V_\phi(h_t^k, z_t^k) + \beta \cdot \sigma_{ens}^2$$

变量含义：
- $F_t^k$：particle $k$ 在时间 $t$ 的 "free energy score"（其实更应该叫 expected utility + curiosity bonus）
- $V_\phi(h_t^k, z_t^k)$：critic network 预测的 value（exploit 项，越高分越好）
- $\sigma_{ens}^2$：一组 prior model ensemble 在这个 latent state 上的预测方差（explore 项，方差大说明 model 不确定、值得探索）
- $\beta$：scaling factor，平衡 explore vs exploit

**直觉**：高分 particle 留下（继续 exploite）；ensemble 不同意的 particle 也留下（去 explore 未知区域）。这正是 [Plan2Explore](https://arxiv.org/abs/2005.05943) 和 [Bootstrapped DQN](https://arxiv.org/abs/1602.04621) 的思路。

**严格 caveat**：这跟 Friston 的 Expected Free Energy 公式不是一回事。Friston 的 EFE 是：

$$G(\pi) = \underbrace{\mathbb{E}_q[\ln q(\theta) - \ln p(\theta \mid y)]}_{\text{epistemic value}} + \underbrace{\mathbb{E}_q[\ln q(y) - \ln p(y)]}_{\text{pragmatic value}}$$

paper 里只是粗糙的 scalar proxy。命名有点 loose，但思路相似。

---

## 4. 实验结果——人话版

### Domain

[MPE SimpleTag](https://www.pettingzoo.farama.org/environments/mpe/simple_tag/)：agent 躲 3 个 predator。author 给 predator 加了 trick——靠近 agent 时 stochastic 切换 CHASE（直线追）/ INTERCEPT（预判拦截）两种策略。这就**人为构造了 bimodal future**：predator 下一步位置要么在路径 A，要么在路径 B，绝不会在中间。

### 表格讲人话

| Model | 配置 | Score (0 最好) | 人话 |
|---|---|---|---|
| BaseDreamer | K=1, T=16 | $-9.21 \pm 0.80$ | 原版，Gaussian 单梦 |
| **ProbDreamer Lite 1** | **K=2, N=1, T=10** | **$-8.79 \pm 0.68$** | **2 条并行梦，赢！** |
| ProbDreamer Lite 2 | K=4, N=1, T=22 | $-9.43 \pm 1.57$ | 4 条梦反而差（粒子饱和） |
| ProbDreamer Full 1 | K=2, N=4, T=10 | $-53.78 \pm 12.14$ | beam search 灾难 |
| ProbDreamer Full 2 | K=8, N=1, T=22 | $-26.84 \pm 23.03$ | 8 条梦 + 长 horizon 不稳 |

**关键发现**：
1. **K=2 是 sweet spot**：因为 predator 恰好有 2 个策略，2 个 particle 完美对应。多了反而 fit noise。
2. **N=1（不做 beam search）反而最好**：beam search 配 free energy pruning 直接崩溃。
3. **短 horizon（T=10）比长 horizon 稳**：做梦做太久 model 误差累积。

### 行为观察

看 gameplay 录像，ProbDreamer 能在 predator 切策略瞬间反应——因为 particle 1 一直在"假设 CHASE"、particle 2 一直在"假设 INTERCEPT"，哪个成真就跟着哪条 dream 走。

BaseDreamer 反应慢半拍——它的 Gaussian 把 CHASE 和 INTERCEPT 平均成"中间路径"，policy 脑补的位置跟实际不符，愣住 0.5 秒。

---

## 5. 为什么 Full 失败了——人话诊断

这是这篇 paper 最有价值的部分。三个 failure mode：

### Failure A：Particle Saturation

K 不能瞎大。SimpleTag 有 2 个 mode（CHASE / INTERCEPT），K=2 正好，K=4 就开始 fit noise，K=8 完全跑偏。

**直觉**：particle filter 的 K 应该匹配 environment 的 intrinsic modality 数。这不是越多越好。

类比：用 K-means 聚类，你知道数据有 3 类就设 K=3，设 K=10 就把好好的 cluster 拆碎。

后续工作方向：自动学 K。用 Dirichlet Process Mixture Model / Chinese Restaurant Process / Bayesian non-parametric 让 K 随数据增长。

### Failure B：Ineffective Pruning

beam search + value-based pruning 失败。诊断：

- imagination 时没 ground-truth observation 校验
- pruning 信号只能靠 critic $V_\phi$
- 但 critic 早期是 noisy 的、biased 的
- noisy critic 给不切实际的 trajectory 高 value
- 这些 trajectory 被 select 进 training
- policy 在这些"虚假高 value 梦境"上 update
- 学到 spurious policy
- 收集到 worse real data
- critic 更差
- 死循环

这是 model-based RL 的经典死穴，叫 **model exploitation** 或 **hallucination loop**。Lambert 那篇 ["Trust Issues with World Models"](https://arxiv.org/abs/2206.06369) 专门讲这个。

**根本原因**：imagination 是个 self-referential 的 zero-ground-truth game。任何在 imagination 阶段做的决策 signal，都是 model 自己 hallucinate 出来的。

### Failure C：Ensemble Collapse

作者想用 ensemble disagreement 估计 epistemic uncertainty，结果 ensemble members 迅速 collapse 成几乎相同的预测。

原因：相同初始化 + 相同 optimizer + 相同 data → 相同 gradient → 相同 model。

这是 [Bootstrapped DQN](https://arxiv.org/abs/1602.04621) 以来大家都知道的坑。要 diversity 必须显式做：
- 不同 random init
- 不同 replay subset（bootstrapped）
- 不同 optimizer / LR
- 显式 orthogonal regularization
- 不同 network architecture

ablation 显示去掉 curiosity term (σ_ens²) 影响不大——因为 ensemble 已经没 diversity 了。

---

## 6. 最深层 Intuition

**Imagination 是 zero-ground-truth loop。**

这是 model-based RL 的根本困境。任何在 imagination 阶段做的决策（pruning、selecting、weighting）都是 self-referential 的。

- Particle filter 缓解了 **representation** 层面的问题（多模态可以表达）。
- 但 pruning / selection 层面的问题没解决。

下一步关键：找到 **reality-grounded signal in imagination**。可能路径：
1. **Replay buffer anchor**：imagination trajectory 不能离 replay buffer 真实 transitions 太远（trust region, Lambert 2022）
2. **Rollback to real**：周期性用 real transitions 校准 imagination
3. **Adversarial discriminator**：GAN-style world model，discriminator 区分 real vs dreamed
4. **Diffusion-based world model**：diffusion 天然 multi-modal，且可以 ground 在 real data distribution 上
5. **Symmetric ensemble diversity**：显式约束 ensemble members 互相 orthogonal
6. **JAX-vmap K=1000 particle**：把 K 推到极限看 scaling law，看是否能消除"K 选择"的敏感性

---

## 7. 我的整体评价

### 优点
1. **问题抓得准**：mode collapse + single rollout 是 Dreamer 的真实痛点
2. **诚实的 negative result**：Full model 失败、ensemble collapse、pruning 失效都老实交代，这反而最有信息量
3. **Hybrid 思路聪明**：单步 Gaussian 保 smooth gradient，外层 particle 保 multi-modality

### 局限
1. **Domain 太简单**：SimpleTag 是 fully observable、低维、2-mode 的人造环境，K=2 恰好 fit。换到 Atari / Minecraft / 部分可观测环境还能不能 work？存疑。
2. **Particle filter 不完整**：没有 importance weight、没有 resampling，严格讲只是 parallel rollouts。
3. **"Free Energy" 命名 loose**：跟 Friston EFE 公式不是一回事，借名字蹭热度嫌疑。
4. **实验规模小**：单 GPU、5 seeds、6 finalist，统计意义不强。
5. **没解决核心问题**：imagination 的 reality-grounding 仍然 open。

### 给 Andrej 你的 intuition

如果让我从中拎一句最值得记住的话：

> **Particle filter 是表达 multi-modal belief 的好工具，但 imagination 阶段的任何 selection signal 都必须能 anchor 到真实 data，否则就是 self-referential hallucination loop。**

这跟你在 [Tesla AI Day 讲的 "data engine 闭环"](https://www.youtube.com/watch?v=j0z4FweCy4c) 是同一个道理：合成数据生成必须有 real data anchor，否则越合越偏。Dreamer 的 imagination 本质上是"按 learned model 合成数据"，同样需要 anchor。

---

## 关键链接清单

**Core papers**:
- [Dreamer V3 (Hafner 2023)](https://arxiv.org/abs/2301.04104) — 这篇的 backbone
- [Dreamer V1 (Hafner 2020)](https://arxiv.org/abs/1912.01603) — 原版
- [PlaNet / RSSM (Hafner 2019)](https://arxiv.org/abs/1811.04551)
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122)

**Particle filter & RL planning**:
- [POMCP (Silver & Veness 2010)](https://papers.nips.cc/paper/2010/hash/edf28225e278d6605c9c0d47f7c3e89c-Abstract.html)
- [Doucet SMC intro 2000](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_SMCintrointro_2000.pdf)

**Model exploitation 问题**:
- [Trust Issues with World Models (Lambert 2022)](https://arxiv.org/abs/2206.06369) — 必读
- [MBPO (Janner 2019)](https://arxiv.org/abs/1906.05243)
- [MOPO (Yu 2020)](https://arxiv.org/abs/2005.05951)

**Exploration & ensemble**:
- [Plan2Explore (Sekar 2020)](https://arxiv.org/abs/2005.05943)
- [Bootstrapped DQN (Osband 2016)](https://arxiv.org/abs/1602.04621)
- [ICM (Pathak 2017)](https://arxiv.org/abs/1705.05363)
- [RND (Burda 2018)](https://arxiv.org/abs/1810.12894)

**Free Energy / Active Inference**:
- [Friston 2010 FEP](https://www.nature.com/articles/nrn2787)
- [Friston 2021 EFE](https://arxiv.org/abs/2004.08128)
- [Tschantz 2020 AIF = Bayesian RL](https://arxiv.org/abs/2001.07203)

**Multi-modal latents**:
- [Mixture Density Networks (Bishop 1994)](https://publications.aston.ac.uk/id/eprint/953/1/NCRG_94_004.pdf)
- [Normalizing Flows (Rezende 2015)](https://arxiv.org/abs/1505.05770)

**Domain & framework**:
- [PettingZoo SimpleTag](https://www.pettingzoo.farama.org/environments/mpe/simple_tag/)
- [MuZero (Schrittwieser 2020)](https://www.nature.com/articles/s41586-020-03051-4)
- [TD-MPC2 (Hansen 2023)](https://arxiv.org/abs/2310.16828)

**未来方向参考**:
- [Diffusion World Models](https://diffusion-rl.github.io/)
- [TWM Transformer World Model (Robine 2023)](https://arxiv.org/abs/2304.09143)
- [SWAG (Maddox 2019)](https://arxiv.org/abs/1902.02476)
- [Epinet (Osband 2023)](https://arxiv.org/abs/2107.08924)

---

需要我继续展开哪一块？比如：
- 把 RSSM 的数学完整推一遍
- 写个 JAX 伪代码实现 K-particle filter imagination
- 对比 Dreamer V3 categorical vs Gaussian 的 gradient 性质
- 讲讲 Friston EFE 严格公式跟 RL 的关系

告诉我你感兴趣的方向。

---

# Probabilistic Dreaming for World Models — 详细技术分析

你好 Andrej。这篇 paper 是 Yale 的 Gavin Wong 写的一个 short paper（134 行），工作量和规模不大，但是触及了 model-based RL 里几个非常核心的问题：**latent imagination 的多模态性、粒子滤波近似、以及 model exploitation 问题**。我尽量详细地帮你 build intuition。

---

## 1. 背景与动机

Dreamer 系列（[Hafner et al. 2020, v1](https://arxiv.org/abs/1912.01603); [2021, v2](https://arxiv.org/abs/2010.02193); [2023, v3](https://arxiv.org/abs/2301.04104)）的 core pipeline 是：用 **RSSM**（Recurrent State Space Model，[Hafner et al. 2019 PlaNet](https://arxiv.org/abs/1811.04551)）学一个 latent dynamics model $p(h_t, z_t \mid x_{1:t}, a_{1:t-1})$，然后在 latent 空间里 rollout 一条 trajectory 用 actor-critic 更新 policy，完全 bypass 真实环境（"dreaming"）。

这篇 paper 抓住了 Dreamer 的两个非常具体的 weak points：

### 1.1 Single-sample rollout 浪费了 learned distribution

Dreamer 的 posterior encoder / prior 学的都是一个 distribution $p(z_t \mid h_t)$，但 rollout 时只从中 sample 一个 $z_t$ 然后用 $h_{t+1} = f(h_t, z_t, a_t)$ 往前推。这本质上是把 stochastic dynamics collapse 成 deterministic 一条 trajectory。在 RL 里这意味着 imagination 的 sample efficiency 受限于 1/K 的覆盖率，且方差高。

直觉：如果学了一个 Gaussian $\mathcal{N}(\mu, \sigma^2)$，只 sample mean 附近的点，那 policy 永远看不到 distribution 的尾部，而 RL 中最 informative 的 transitions 往往在尾部（rare events、adversarial predator 策略切换）。

### 1.2 Unimodal Gaussian 在多模态未来上的 "averaging catastrophe"

这是经典的 **Mixture Density Network** 问题（[Bishop 1994](https://publications.aston.ac.uk/id/eprint/953/1/NCRG_94_004.pdf)）。当真实 future 是 bimodal（predator 50% chance 往左，50% chance 往右），用 unimodal Gaussian 拟合会把 mode 塌到 mean，得到一个 "中间路径"——而中间路径在真实环境里几乎从不发生。

Dreamer v3 的解决方案是换成 **discrete categorical latents**（如 32 classes × 32 categories），用 categorical distribution 自然支持 multi-modality。但代价是：
- 梯度不连续（Gumbel-softmax / straight-through estimator）
- 表示稀疏（one-hot）
- 对 reward / value head 的 dense gradient 不友好

所以作者想保留 Gaussian 的 dense smooth gradient 优势，**但在外层用 non-parametric particle filter 表达 multi-modality**。这是一个 hybrid 思路。

---

## 2. 方法详解

### 2.1 Backbone: BaseDreamer

直接用 Dreamer-v3 的所有 components，但把 latent 从 categorical 换回 Gaussian：

| Component | 公式 | 作用 |
|---|---|---|
| RSSM | $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$ | deterministic recurrent state |
| Posterior | $q_\phi(z_t \mid h_t, x_t) = \mathcal{N}(\mu_{post}, \sigma_{post})$ | observation-grounded latent |
| Prior | $p_\theta(z_t \mid h_t) = \mathcal{N}(\mu_{prior}, \sigma_{prior})$ | imagination prior（无观测） |
| Decoder | $\hat{x}_t = g_\theta(h_t, z_t)$ | reconstruction |
| Heads | reward / continue predictions | standard |

KL balance trick（[Hafner et al. 2020](https://arxiv.org/abs/1912.01603)）依然保留：
$$\mathcal{L}_{dyn} = \beta \cdot \mathrm{KL}(q \Vert p), \quad \mathcal{L}_{rep} = (1-\beta) \cdot \mathrm{KL}(q \Vert p)$$

其中 $\beta$ 控制 prior 拉近 posterior 的强度，避免 posterior 过拟合观测导致 prior 不可用。

### 2.2 Particle Filter（核心创新 #1）

这是 paper 的灵魂。Imagination 时维护 K 个 particles $\{(h_t^k, z_t^k)\}_{k=1}^K$，每个 particle 用相同的 prior model 独立 propagate：

$$z_{t+1}^k \sim p_\theta(\cdot \mid h_{t+1}^k), \quad h_{t+1}^k = f_\theta(h_t^k, z_t^k, a_t^k)$$

每个 particle 是一次独立的 "dream"，于是我们得到 K 条并行 trajectories。这就把 belief $p(z_t \mid h_t)$ 从 unimodal Gaussian 升级为 empirical distribution：

$$\hat{p}(z_t \mid h_t) = \frac{1}{K} \sum_{k=1}^K \delta(z_t - z_t^k)$$

**关键 intuition**：单步 transition 还是 Gaussian（保留 smooth gradient），但 outer-level 是 non-parametric，所以几步 propagation 后 empirical distribution 可以逼近任意形状的 multi-modal belief。这等价于用 sequential Monte Carlo 近似 belief state（[Doucet, Godsill, Andrieu 2000](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_SMCintrointro_2000.pdf); [Del Moral et al. 2006](https://arxiv.org/abs/1301.4082)）。

不过这里**没有 explicit resampling step**，作者也没提 importance weights。严格来说这不是完整 particle filter，更像 "parallel multi-start rollout"。真正 particle filter 应该有：
1. **Prediction**: 用 prior propagate
2. **Update**: 用 likelihood $p(x_t \mid z_t)$ 算 weight $w_t^k \propto \frac{p(x_t \mid z_t^k) p(z_t^k \mid z_{t-1}^k)}{q(z_t^k \mid x_t, z_{t-1}^k)}$
3. **Resampling**: 按 weight 重采样，避免 particle degeneracy

Dreaming 时没有 ground-truth $x_t$，所以无法做标准 update step。这正是 paper 后面 "free energy pruning" 想解决但失败的问题。

### 2.3 Latent Beam Search（创新 #2）

每个 particle 在每个时间步 branch 出 N 个候选 actions，从 policy 采样：
$$\{a_t^{k,n}\}_{n=1}^N \sim \pi_\theta(\cdot \mid h_t^k, z_t^k)$$

总共 $K \times N$ 条 branches。这是 MuZero 风格的 search（[Schrittwieser et al. 2020 MuZero](https://www.nature.com/articles/s41586-020-03051-4)），但区别：
- MuZero 在真实环境 inference 时 search
- 这里在 imagination 训练阶段 search
- MuZero 用 PUCT，这里用 free energy scoring

### 2.4 Free Energy Pruning（创新 #3，也是失败点）

在 imagination 阶段，没有 ground-truth observation，标准 MLE / ELBO pruning 失效。作者用 Friston 的 Free Energy Principle（[Friston 2010](https://www.nature.com/articles/nrn2787)）作为代理：

$$F_t^k = V_\phi(h_t^k, z_t^k) + \beta \cdot \sigma_{ens}^2$$

变量解释：
- $F_t^k$：particle $k$ 在时间 $t$ 的 "negative free energy"（这里命名不严格，更像 expected utility + epistemic bonus）
- $V_\phi(h_t^k, z_t^k)$：critic network 预测的 value（exploitation 项）
- $\sigma_{ens}^2$：ensemble of prior models 在该 latent state 上的预测方差（exploration / epistemic uncertainty）
- $\beta$：scaling hyperparameter，平衡 explore vs exploit

**直觉**：高 value 的 particle 应该保留（exploit），高 ensemble disagreement 的 particle 也应该保留（explore，因为 critic 可能不准）。这正是 [Plan2Explore](https://arxiv.org/abs/2005.05943) 的 disagreement-based exploration 思想，也是 [Bootstrapped DQN](https://arxiv.org/abs/1602.04621) 的 ensemble uncertainty。

严格说，Friston 的 **Expected Free Energy** (EFE, [Friston et al. 2021](https://arxiv.org/abs/2004.08128)) 是：

$$G(\pi) = \underbrace{\mathbb{E}_q[\ln q(\theta) - \ln p(\theta \mid y)]}_{\text{epistemic value (param info gain)}} + \underbrace{\mathbb{E}_q[\ln q(y) - \ln p(y)]}_{\text{pragmatic value (negative surprise)}}$$

paper 这里只是一个粗糙的 scalar approximation，并没有严格 derive 出来。所以这个 "free energy" 是 loose 命名。

---

## 3. 实验设计与结果

### 3.1 Domain: MPE SimpleTag

[PettingZoo SimpleTag](https://www.pettingzoo.farama.org/environments/mpe/simple_tag/) 是一个 multi-agent predator-prey game。作者的 twist：predator 在 agent 附近时 stochastically 切换 CHASE / INTERCEPT 策略，构造 bimodal future position distribution。这刻意制造一个 ground-truth bimodal 的问题。

### 3.2 训练 loop

- 每 iteration：$10^3$ real env steps + $\bar{2} \cdot 10^4$ imagination steps（replay buffer + dream）
- 这 1:20 的 ratio 是 model-based RL 的典型 sample efficiency 收益

### 3.3 关键结果表（Table 1）

| Model | K | N | T | Score (0 = perfect) |
|---|---|---|---|---|
| BaseDreamer 1 | 1 | 1 | 16 | $-9.21 \pm 0.80$ |
| BaseDreamer 2 | 1 | 1 | 10 | $-9.74 \pm 0.79$ |
| **ProbDreamer Lite 1** | **2** | **1** | **10** | **$-8.79 \pm 0.68$** |
| ProbDreamer Lite 2 | 4 | 1 | 22 | $-9.43 \pm 1.57$ |
| ProbDreamer Full 1 | 2 | 4 | 10 | $-53.78 \pm 12.14$ |
| ProbDreamer Full 2 | 8 | 1 | 22 | $-26.84 \pm 23.03$ |

观察：
1. **Lite 1 (K=2, N=1) 是 winner**：score 提升 4.5%，variance 降低 28%
2. K=2 比 K=4 还好（particle saturation）
3. N>1（beam search）导致 catastrophic degradation（-53, -26）
4. T=10 比 T=16/22 好（短 horizon 更稳）

### 3.4 行为分析

作者观察 gameplay 发现：ProbDreamer 能快速响应 predator 策略切换，BaseDreamer 会"freeze momentarily"——这是 Gaussian mean collapse 的典型症状：当 predator 突然变 INTERCEPT，Gaussian prior 平均了 CHASE/INTERCEPT，predicted next state 在两者中间，policy 不知道往哪边躲，就愣住。

---

## 4. 失败模式深度分析（这是 paper 最有价值的部分）

### 4.1 Particle Saturation（K=2 最优）

K 应该匹配 environment 的真实 modes 数。SimpleTag 有 2 个 predator 策略，K=2 完美 fit，K>2 会 fit noise。这给我们一个深刻 intuition：

> **Particle filter 的 K 不是越多越好，而是要先验知道（或学出）环境的 intrinsic modality。**

这与 [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/953/1/NCRG_94_004.pdf) 选 mixture components 数、[DPMM](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/density-models.pdf) 用 Dirichlet process 自动决定 components 类似。未来工作应该用 **non-parametric Bayesian prior**（如 Dirichlet Process）自动学 K，或用 [Chinese Restaurant Process](https://en.wikipedia.org/wiki/Chinese_restaurant_process) 让 K 随数据增长。

### 4.2 Ineffective Pruning（model exploitation / hallucination）

这是 model-based RL 的经典死穴（[Lambert et al. "Trust Issues with World Models" 2022](https://arxiv.org/abs/2206.06369); [Talvitie 2014](https://arxiv.org/abs/1406.1165)）：

- Critic 在 imagination 阶段被用来 prune branches
- 但 critic 自己是 noisy、early-training biased
- Noisy critic 给 unrealistic trajectory 高 value → 被 select → policy 在这些 trajectory 上 update → 学到 spurious policy → 收集到 worse data → critic 更差 → 死循环

这本质是 **off-policy distribution shift + model bias** 的复合。Lambert 的 paper 给出的解法是 model ensembling + disagreement penalty，但这里 ensemble 又 collapse 了。

**直觉**：imagination 是 self-play 的 zero-ground-truth game，任何 pruning signal 都是 model 自己 hallucinate 的。要打破这个循环，必须引入某种 "reality check"。可能的方案：
- **Trust region**：限制 imagination 离 replay buffer 真实 transitions 不能太远（[Lambert et al.](https://arxiv.org/abs/2206.06369)）
- **Rollback to real data**：周期性用 real transitions 校准 imagination
- **Adversarial training**：discriminator 区分 real vs dreamed transitions（类似 GAN world models, [Ramé et al. 2022](https://arxiv.org/abs/2207.07081)）

### 4.3 Ensemble Collapse

[Bootstrapped DQN](https://arxiv.org/abs/1602.04621) 以来大家都知道 ensemble 要 diversity，但实现上往往 fail：相同初始化、相同 optimizer、相同 data → 同样梯度 → collapse。

作者的三个建议都是经典解法：
1. **Different optimizers + different replays**：类似 bootstrapped ensemble
2. **MC Dropout**（[Gal & Ghahramani 2016](https://arxiv.org/abs/1506.02142)）：但 dropout 的 epistemic estimate 一直被认为 weak
3. **Disagreement in rewards/observations**：[Plan2Explore](https://arxiv.org/abs/2005.05943) 的做法

我补充几个更强的方向：
- **SWAG / Laplace approximation**（[Maddox et al. 2019](https://arxiv.org/abs/1902.02476)）：post-hoc Bayesian
- **Epistemic Neural Networks / Epinet**（[Osband et al. 2023](https://arxiv.org/abs/2107.08924)）：DeepMind 的新思路
- **SNGP**（[Liu et al. 2020](https://arxiv.org/abs/2006.10108)）：Spectral normalization + Gaussian process
- **Hyperbolic embeddings**：把 epistemic uncertainty 编码进 latent space 的曲率

---

## 5. 与相关工作的连接

### 5.1 同血脉
- **[DreamerV1/V2/V3](https://danijar.com/project/dreamerv3/)**：直接 backbone
- **[TD-MPC2](https://arxiv.org/abs/2310.16828)**：Hansen et al., 同样 multi-modal world model 的努力，用 MPC + latent ensemble
- **[PlaNet](https://arxiv.org/abs/1811.04551)**：CEM-based planning in latent space
- **[MuZero](https://www.nature.com/articles/s41586-020-03051-4)**：MCTS in latent space，这篇的 beam search 是简化版

### 5.2 Particle filter in RL
- **[POMCP / POMCPOW](https://arxiv.org/abs/1109.2145)**：Particle Monte Carlo Tree Search
- **[Silver & Veness 2010 Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/2010/hash/edf28225e278d6605c9c0d47f7c3e89c-Abstract.html)**
- **[PPO + particle filter for visual navigation](https://arxiv.org/abs/2010.00738)**

### 5.3 Multi-modal latents
- **[Mixture Density Networks (Bishop 1994)](https://publications.aston.ac.uk/id/eprint/953/1/NCRG_94_004.pdf)**
- **[VAE with mixture prior](https://arxiv.org/abs/1611.05148)**
- **[DreamerV3 discrete latents](https://arxiv.org/abs/2301.04104)**：这条 paper 在正面挑战
- **[VQ-VAE](https://arxiv.org/abs/1711.00937)**：discrete latent 的另一思路

### 5.4 Free energy / Active inference
- **[Friston 2010](https://www.nature.com/articles/nrn2787)**：原始 FEP
- **[Friston et al. 2021 Active Inference](https://arxiv.org/abs/2004.08128)**：EFE 公式
- **[Millidge 2020](https://arxiv.org/abs/2003.06206)**：combining FEP with ML
- **[Tschantz et al. 2020](https://arxiv.org/abs/2001.07203)**：AIF is just Bayesian RL，澄清概念

### 5.5 Model exploitation
- **[Lambert et al. 2022 "Trust Issues with World Models"](https://arxiv.org/abs/2206.06369)**：必读，诊断同类问题
- **[Janner et al. 2019 MBPO](https://arxiv.org/abs/1906.05243)**：用 rollout length schedule 缓解
- **[Yu et al. 2020 MOPO](https://arxiv.org/abs/2005.05951)**：uncertainty penalty in reward

### 5.6 Curiosity / exploration via ensemble
- **[Pathak ICM 2017](https://arxiv.org/abs/1705.05363)**
- **[Burda RND 2018](https://arxiv.org/abs/1810.12894)**
- **[Sekar Plan2Explore 2020](https://arxiv.org/abs/2005.05943)**

---

## 6. 我的整体评价与延伸思考

### 6.1 Paper 的真正贡献

诚实讲，这篇 paper 的实验规模很小（一个 domain、单 GPU sweep、6 finalist），写作也存在概念混淆（"free energy" 命名不严格、particle filter 没 resampling）。但它的价值在于：

1. **明确分离了两个问题**：
   - (A) Latent distribution 的 multi-modality
   - (B) Imagination 的 parallel exploration
   并提出用 particle 解决 A、用 beam search + scoring 解决 B

2. **诚实记录了 negative result**：beam search + free energy pruning 全失败，这反而是最 informative 的部分。大多数 paper 只报 win，这篇把 fail mode 拆得很清楚。

### 6.2 我会进一步追的方向

如果让我接着做，几个最有 promise 的方向：

**(1) 用 Normalizing Flow 替代 Gaussian prior**：保留 continuous + smooth gradient，但 single-component flow 就能表达 multi-modal distribution（[Rezende & Mohamed 2015](https://arxiv.org/abs/1505.05770)）。比 particle filter 更参数化、更可学。

**(2) Object-centric latents**（[Locatello et al. 2020](https://arxiv.org/abs/1906.07355); [Anand et al. 2023](https://arxiv.org/abs/2206.07363)）：多模态本质常常来自不同 object 的不同 future。每个 object 一个 slot，每个 slot 一个 particle，自然 K=object count。

**(3) Transformer-based world model**（[Robine et al. 2023 TWM](https://arxiv.org/abs/2304.09143); [Schmidgall et al. 2024](https://arxiv.org/abs/2403.12398)）：用 attention over particles，类似 set transformer，让 K 自适应。

**(4) Diffusion-based world model**（[Mendonca et al. 2023](https://diffusion-rl.github.io/); [Zhang et al. 2024 DiffSim](https://arxiv.org/abs/2404.16611)）：diffusion model 天然 multi-modal，采样时间步可以并行（[Song et al. 2020 DDPM](https://arxiv.org/abs/2006.11239)），直接解决这篇 paper 的两个 motivation。

**(5) Symmetric ensemble diversity**：用 [Negative Correlation Learning](https://www.cs.bham.ac.uk/~gxk/papers/nc-ijcnn-2000.pdf) 或 [Orthogonal Regularization](https://arxiv.org/abs/1609.07093) 显式约束 ensemble members 正交，解决 collapse。

**(6) World model + JAX-based vectorized particle filter**：在 JAX 里 vmap K particles，K=1000 也能跑（[Hafner personal note](https://danijar.com/project/dreamerv3/)），完全可以把 K 推到 100+ 看 scaling law。

**(7) Active inference 的严格 EFE**：把 Friston 的 EFE 严格写进 RL loss，包括 epistemic value 用 BALD score（[Houlsby et al. 2011](https://arxiv.org/abs/1112.5745)），pragmatic value 用 expected reward。

### 6.3 这篇 paper 的最大直觉收益

**Imagination 是 zero-ground-truth game**——任何在 imagination 阶段做决策的 signal 都是 self-referential。这是 model-based RL 的根本困境。Particle filter 缓解了 representation 层面的问题（多模态可以表达），但 pruning 决策层面的问题没解决。下一步的关键是找到 **reality-grounded signal in imagination**，无论是：
- Replay buffer 的 nearest-neighbor anchor
- 后验 bootstrap 校准
- 还是 model ensemble diversity penalty

只要这个 "ground-truth anchor" 解决了，model-based RL 就能跳出当前的 model exploitation 瓶颈。

---

## 关键参考链接

- [Dreamer V3 paper (Hafner 2023)](https://arxiv.org/abs/2301.04104)
- [Dreamer V1 (Hafner 2020)](https://arxiv.org/abs/1912.01603)
- [PlaNet / RSSM (Hafner 2019)](https://arxiv.org/abs/1811.04551)
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122)
- [MuZero (Schrittwieser 2020)](https://www.nature.com/articles/s41586-020-03051-4)
- [Plan2Explore (Sekar 2020)](https://arxiv.org/abs/2005.05943)
- [Trust Issues with World Models (Lambert 2022)](https://arxiv.org/abs/2206.06369)
- [Bootstrapped DQN (Osband 2016)](https://arxiv.org/abs/1602.04621)
- [MC Dropout (Gal & Ghahramani 2016)](https://arxiv.org/abs/1506.02142)
- [Free Energy Principle (Friston 2010)](https://www.nature.com/articles/nrn2787)
- [Active Inference / EFE (Friston 2021)](https://arxiv.org/abs/2004.08128)
- [Mixture Density Networks (Bishop 1994)](https://publications.aston.ac.uk/id/eprint/953/1/NCRG_94_004.pdf)
- [Particle Filters intro (Doucet 2000)](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_SMCintrointro_2000.pdf)
- [PettingZoo MPE](https://www.pettingzoo.farama.org/environments/mpe/simple_tag/)
- [TD-MPC2 (Hansen 2023)](https://arxiv.org/abs/2310.16828)
- [ICM Pathak 2017](https://arxiv.org/abs/1705.05363)
- [RND Burda 2018](https://arxiv.org/abs/1810.12894)
- [MBPO Janner 2019](https://arxiv.org/abs/1906.05243)
- [MOPO Yu 2020](https://arxiv.org/abs/2005.05951)
- [Normalizing Flows (Rezende & Mohamed 2015)](https://arxiv.org/abs/1505.05770)
- [SWAG (Maddox 2019)](https://arxiv.org/abs/1902.02476)
- [Epinet (Osband 2023)](https://arxiv.org/abs/2107.08924)
- [DDPM (Song 2020)](https://arxiv.org/abs/2006.11239)

---

如果你想我再深入某一块（比如把 RSSM 的 math 推一遍、或写一段 JAX 伪代码实现 K-particle filter、或对比 DreamerV3 categorical vs Gaussian 的 gradient 分析），告诉我，我可以继续展开。
