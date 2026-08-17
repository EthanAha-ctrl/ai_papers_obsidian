---
source_pdf: Simulation Distillation Pretraining World Models.pdf
paper_sha256: f5ce6f9664b4f4932bc84794eaa7bfcb328f2dcb4bbc5f662c612e8a50b69f2a
processed_at: '2026-08-12T06:49:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 SimDist

---

## 故事的开头：sim-to-real 到底卡在哪

你有个 robot，你想让它干活。直接在 real world 上 RL 太贵太慢，所以大家都在 simulation 里训——sim 不要钱，可以跑 4096 个 parallel env，几小时就能训出一个看起来很猛的 policy。

然后你把这个 policy 搬到 real robot 上，**它经常直接趴窝**。

为什么？因为 sim 不准。sim 里的摩擦系数是 0.5，real 上是 0.3。sim 里的 foam 是刚性 wall，real 上的 foam 会压扁变形。sim 里 peg 对准 hole 就插进去，real 里 peg 卡在 hole 边缘因为接触动力学完全不一样。

这就是 **dynamics gap**。

---

## 老办法为什么不行

**办法 1：domain randomization + zero-shot deploy**

在 sim 里把 friction / mass / stiffness 全随机化，训一个 robust policy，希望它能 generalize 到 real。问题是 robust ≠ accurate。你让 policy 学会 "不管 friction 是 0.2 还是 1.0 都能走"，但 real 的 friction 是 0.35 且还在变化，你这个 policy 仍然不是最优的。Fig. 1 左边就是 zero-shot 失败的惨状。

**办法 2：sim pretrain + real finetune（model-free RL）**

把 sim policy 当 warm start，在 real 上继续跑 SAC / PPO / IQL / RLPD。听起来合理，但实际：

- real 上一分钟可能就 300 步，15 分钟才 4500 步——太少
- model-free RL 要 bootstrap value function，value 在低数据下 overestimate 严重
- policy 一更新就把 sim 里学到的好东西忘光了（catastrophic forgetting）
- exploration 根本展不开，real robot 不能随便乱试

Fig. 4 里 IQL / RLPD 的曲线就是：finetune 越 finetune 越烂，甚至 collapse。

**办法 3：online model-based RL（Dreamer / TD-MPC）**

在 real 上直接学一个 world model（dynamics + reward + value + policy），然后用这个 world model plan。问题是你得**同时**从 raw image bootstrap 出来 representation、dynamics、reward、value、policy——所有这些在 real 低数据下都学不准，一个学不准全家崩。这类方法在 real contact-rich task 上经常 saturate。

---

## SimDist 的 aha moment

观察一下 world model 的结构，它其实是有自然 modularity 的：

```
observation → encoder → latent state
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           dynamics    reward     value
           (预测下一步) (打分)    (长远收益)
```

关键观察：**sim 和 real 之间，有些东西变，有些东西不变。**

**变的是 dynamics**——施加一个 torque 后 robot 到底怎么动，sim 算出来的和 real 不一样，因为 friction / contact / compliance 不准。

**不变的是 representation / reward / value 的语义**——

- "peg 离 hole 多远" 这个几何关系，sim 和 real 里是一样的
- "peg 插进 hole 了没" 这个 reward 信号，sim 和 real 里是一样的
- "从这个状态出发，未来能成功插进去的概率" 这个 value，sim 和 real 里是相近的

因为这些东西 depend on 的是 **task structure**（物体语义、空间几何、goal 条件），不是 **low-level physics**（接触力、摩擦、变形）。

所以 SimDist 的做法非常直觉：

> **sim 里把 representation / reward / value / dynamics 全部训好。real 里冻住前三者，只更新 dynamics。**

这就把一个"在 real 上从头学 RL"的噩梦，变成一个"在 real 上做个 supervised regression"的简单活——你只需要教 dynamics model："在 real 里，从这个 latent state 施加这个 action，下一个 latent state 是什么"。target 是 encoder 冻住之后对 real observation encode 出来的值。

---

## 但这里有个 catch

你 real 部署的时候，不是跑 expert policy。你跑的是 **MPPI planner**——它会 sample 几百条 candidate action sequence，很多是 sub-optimal 的、乱来的、exploration 性质的。

这意味着你的 world model 必须能 predict **不在 expert trajectory 上的状态**。如果你 sim pretrain 只用 expert data，那 world model 只见过 "peg 对准了优雅插入" 这种 trajectory，MPPI 一 sample "peg 歪着撞墙" 的 action，dynamics / reward / value 全部 out-of-distribution，直接崩。

这就是 SimDist 第二个关键 idea：**data generation 要故意制造 diversity**。

具体做法很 hacky 但很 effective：

1. PPO 训 expert 的时候，把所有 intermediate checkpoint 都存下来——iter 0 的 policy 是傻的，iter 500 的 policy 是半傻的，iter 5000 是 expert
2. rollout 的时候，50% 概率用 expert，50% 概率随机抽一个傻 checkpoint
3. 再叠加一段一段的 Gaussian action noise——连续几步加噪声，然后几步干净，交替
4. 这样产生的 data 里既有 "完美执行" 也有 "笨手笨脚" 也有 "扰动恢复"

这个 dataset 里大约 36%-56% 是 clean expert，其余全是 sub-optimal / perturbed。但恰恰是这些 "垃圾" data 让 world model 见过了 "peg 歪着会怎样" "foot 打滑了会怎样"——MPPI 后来在 real 上 sample 的那些奇怪 action，world model 全都 in-distribution。

Ablation 里 expert-only data 把 Peg Insertion 从 0.90 干到 0.10——**9 倍 drop**。这是整篇 paper 最有说服力的一行数字。

---

## 部署的时候到底发生了什么

1. Robot 来一帧 observation $o_t$（RGB images + proprioception）
2. Encoder $E_\theta$ 冻住的，encode 成 latent $z_t$（64 维）
3. History encoder $C_\theta$ 也是冻住的，把过去几步的 obs+action encode 成 $h_t$
4. Base policy $\pi_\theta$（冻住）输出一个 "建议 action chunk"——这是个 warm start
5. MPPI 拿这个建议，加噪声，sample 出 250 条 candidate action sequence
6. 每条 candidate 喂给 dynamics model $f_\theta$（**这个是唯一在 real 上 finetune 的**），一次 forward pass 预测 $T$ 步未来 latent state
7. Reward head $R_\theta$（冻住）和 value head $V_\theta$（冻住）对预测出来的 latent trajectory 打分
8. 算每条 candidate 的 return = $\gamma^T \hat{v}_{t+T} + \sum \gamma^s \hat{r}_s$
9. 按 return 做重要性采样，加权平均出最终执行的 action
10. 执行一步，collect $(o_t, a_t, o_{t+1})$ 加进 real dataset
11. 每 20 episode 拿 real dataset 更新一下 $f_\theta$（只更新它，其他全冻），就是普通的 regression

整个 loop 就是：plan → execute → collect → fit dynamics → plan better。

---

## 为什么这个比 model-free RL finetune 好

直觉上：

- **Model-free RL finetune**：real 上 15 分钟数据，你要让 policy 学会新行为。policy 改了 → value 也要重新 estimate → value estimate 错 → policy 跟着错 → catastrophic forgetting。这是一个 **coupled 不稳定系统**。

- **SimDist**：reward 和 value 已经从 sim transfer 过来了，它们是对的（Fig. 5 证明 value 在 real 上能区分 success/failure）。你只需要让 dynamics 更准一点。dynamics 更准 → MPPI plan 出来的 action 更好 → collect 到更好的 data → dynamics 更准。这是一个 **positive feedback loop**，稳定收敛。

而且 dynamics finetune 是 **supervised regression**，不是 RL。regression 的 loss surface 是 convex-ish 的，LR 调好就稳稳下降。RL 的 loss surface 在低数据下是噩梦。

---

## 为什么 reward / value 能 zero-shot transfer

这是整篇 paper 最 magic 的地方，也是最 fragile 的假设。

Fig. 5 做的验证：同一个初始条件，teleop 出一条成功 trajectory 和一条失败 trajectory，看 $V_\theta(\hat{z}_t)$ 怎么变。成功的 value 单调上升，失败的 value 一直低。

**为什么这个能 work？** 因为 value function 本质上学的是 "当前状态离 goal 有多远 + 接下来走得好不好"。这个 depend on 的是 **task geometry**——peg 在 hole 口上方 2cm 就是 "快成功了"，peg 歪在桌上就是 "还远"——这种几何关系 sim 和 real 里是同一个。

但有个前提：encoder $E_\theta$ 必须能从 real image 里 extract 出同样的几何 latent。这就是为什么 paper 用 ImageNet pretrained ResNet + data augmentation（color jitter / blur / crop）——希望 encoder 学到的是 "peg 的位置和朝向" 而不是 "sim 里那个特定光照下的纹理"。

如果这个假设崩了——比如 real 和 sim 的物体长得完全不一样，或者 sim 渲染太假——那 encoder encode 出来的 latent 就是 garbage，value 和 reward 全跟着错，dynamics finetune 也救不了。这是 SimDist 最脆弱的一环。

---

## 一个被忽视但很重要的架构细节

传统 world model（Dreamer / TD-MPC）predict reward 和 value 的时候，是**每个 timestep 独立**用一个 MLP decode：

```
z_t → MLP → r_t
z_t → MLP → v_t
```

SimDist 用的是 **transformer**，attend 整条 predicted trajectory：

```
[z_t, z_{t+1}, ..., z_{t+T}] → transformer → [r_t, ..., r_{t+T}, v_{t+1}, ..., v_{t+T}]
```

为什么这个重要？因为 MPPI 评估一条 candidate 时，它关心的是**这条 trajectory 的整体 return**，不是单步 reward。trajectory-level structure（"前 3 步在加速，第 4 步要接触了，第 5 步能不能对准"）是 sequence property，per-timestep MLP 根本 capture 不到。

Ablation 里把这个换回 MLP，Peg 从 0.90 掉到 0.82，Table Leg 从 0.85 掉到 0.60。Table Leg 掉得狠是因为那个 threading + screw motion 是高度 sequence-dependent 的。

---

## 另一个反直觉的发现：不要加 reconstruction loss

Dreamer 系列一直带 image reconstruction loss，让 latent 能 decode 回 image。SimDist 发现加了反而崩——Peg Insertion 从 0.90 掉到 0.32。

直觉：reconstruction 逼 latent 去编码 image 的所有细节——背景纹理、光照、peg 表面划痕——这些对 dynamics prediction 和 planning 毫无用处，反而占满了 latent capacity，把真正重要的几何信息挤掉了。

SimDist 的 latent 全部来自 behavioral supervision（dynamics target + reward + value + BC），没有任何 reconstruction。latent 被迫只学 "对预测未来和打分有用的东西"。这是一个更激进的 representation learning stance。

---

## 最后总结：一句话版本

**Sim-to-real 难是因为 dynamics 不准。但 task structure（哪是 goal、离 goal 多远、什么状态算好）在 sim 和 real 之间是 shared 的。所以把 world model 拆成 "冻住的结构" 和 "可变的 dynamics"，sim 里大规模蒸馏前者，real 里只 regression 后者——real adaptation 就从一个 RL 问题降维成一个 supervised learning 问题，15 分钟数据就够了。**

听起来简单，但 devil in the details：

- data diversity 靠 checkpoint mixing + action perturbation
- planning throughput 靠 chunked transformer（一次 forward 出 $T$ 步）
- return accuracy 靠 sequence-to-sequence transformer reward/value head
- transfer robustness 靠 ImageNet pretrained encoder + augmentation
- 稳定收敛靠 frozen encoder 提供 stable latent target

每一块单独看都不新，但组合在一起 + 那个 "只冻 dynamics" 的核心 insight，就 make it work 了。

---

Project: https://sim-dist.github.io

---

# Simulation Distillation (SimDist) 深度解读

这篇 paper 来自 UT Austin (Jacob Levy, David Fridovich-Keil) + UW (Tyler Westenbroek, Abhishek Gupta) + FieldAI 的合作，project page: https://sim-dist.github.io 。核心讲的是 **sim-to-real transfer 的新范式**：与其 end-to-end finetune 一个 policy，不如把 world model 拆成模块，只 adapt dynamics，把 reward/value/representation 全部 freeze。

---

## 1. Problem Setup: 为什么 sim-to-real 这么难

经典的 sim-to-real pipeline 是：在 simulation 里用 domain randomization 训一个 robust policy，然后直接 zero-shot deploy。但 Fig. 1 的左图显示，这种 zero-shot deployment 在 precise manipulation 和 quadruped locomotion 上都会失败——dynamics gap 太大。

替代方案是 **finetune in real world**，但这条路上有几个深坑：

- **Catastrophic forgetting**：end-to-end policy 一旦在新 domain 上 finetune，pretraining 学到的 priors 立刻被忘掉（[Wagenmaker et al. 2024](https://arxiv.org/abs/2412.07762)）。
- **Exploration in low-data regime**：real robot 15-30 分钟的数据很少，model-free RL（RLPD、IQL、SAC）根本探索不开，credit assignment 在 long horizon 上爆炸。
- **Model-based RL 的 bootstrapping trap**：在线 MBRL（如 [TD-MPC](https://arxiv.org/abs/2203.07454)、[Dreamer](https://danijar.com/project/dreamer/)）必须从 raw perception 同时 bootstrap representation、value、reward、policy，sample efficiency 极差，在 contact-rich manipulation 上经常 saturate。

SimDist 的 **核心 insight**：world model 是 modular 的——`representation + reward + value + dynamics`。其中前三者属于 **global task structure**（比如 peg 和 hole 的相对位置、离 goal 多远），在 sim 和 real 之间大致 invariant；dynamics 属于 **local structure**（施加这个 torque 后到底滑多远），sim 和 real 差异巨大。所以应该 freeze 前三者，只 finetune dynamics。

这个 decomposition 让 real-world adaptation 从一个 **long-horizon credit assignment + exploration** 问题，降维成一个 **short-horizon supervised system identification** 问题——这就是关键。

---

## 2. World Model 架构详解

参考 Fig. 3 和 Fig. 8。整体结构仿照 TD-MPC，但做了几处关键修改。

### 2.1 公式 (1)：模块化定义

$$
\begin{aligned}
z_t &= E_\theta(o_t) & \text{(latent representation)} \\
h_t &= C_\theta(o_{t-H:t-1},\, a_{t-H:t-1}) & \text{(history encoding)} \\
\hat{z}_{t+1:t+T} &= f_\theta(z_t,\, a_{t:t+T-1},\, h_t) & \text{(latent dynamics)} \\
\hat{r}_{t:t+T-1} &= R_\theta(\hat{z}_{t:t+T},\, a_{t:t+T-1}) & \text{(reward head)} \\
\hat{v}_{t+1:t+T} &= V_\theta(\hat{z}_{t:t+T}) & \text{(value head)} \\
\hat{a}_{t:t+H} &= \pi_\theta(z_t,\, h_t) & \text{(base policy)}
\end{aligned}
$$

变量含义：
- $o_t \in \mathcal{O}$：raw observation（RGB images + proprioception）
- $s_t \in \mathcal{S}$：underlying true state（simulator 里 privileged 可见，real 里看不到）
- $a_t \in \mathcal{A}$：robot action（manipulation 是 6-DoF end-effector target；quadruped 是 12 个 joint position targets）
- $z_t \in \mathbb{R}^{64}$：latent state，64 维
- $h_t$：history encoding，对过去 $H$ 步的 observation + action 做编码
- $H$：history horizon（manipulation $H=5$，quadruped $H=25$）
- $T$：prediction horizon（manipulation $T=5$，quadruped $T=25$）
- $\hat{z}_{t+1:t+T}$：模型预测的未来 $T$ 步 latent state 序列
- $\hat{r}, \hat{v}$：reward / value 预测
- $\pi_\theta$：base policy，用来 warm-start MPPI sampling
- $\gamma \in (0,1]$：discount factor，论文里 $\gamma=0.99$
- $E_\theta, C_\theta, f_\theta, R_\theta, V_\theta, \pi_\theta$：六个网络，全部参数共享 $\theta$

### 2.2 关键架构决策（Section IV-C）

**Minimal history representation**：把 observation 切成 proprioception $o_t^p$（低维，joint state）和 exteroception $o_t^e$（高维，images / height map）。history encoder $C_\theta$ 只吃 $(o_{t-H:t}^p, a_{t-H:t}, o_t^e)$——只有最近一步 exteroception。理由：减少 transformer context length，plan latency 大幅下降，且经验上训练更稳。

**Chunked prediction**：$f_\theta$ 不是一个 autoregressive RNN，而是一个 transformer，**一次 forward pass 直接出 $T$ 步 latent state**。具体来说用 cross-attention 把 history tokens 和 candidate action sequence 关联起来，加 causal mask。这样 MPPI 可以在 GPU 上 batch 几百条 candidate trajectory 并行评估，throughput 高一个数量级。这个设计来自 [AnyCar to Anywhere (Xiao et al. 2025)](https://arxiv.org/abs/2503.06815)。

**Sequence-to-sequence return modeling**：$R_\theta$ 和 $V_\theta$ 也是 transformer，**attend 整条 $\hat{z}_{t:t+T}$**——而不是像 [TD-MPC](https://arxiv.org/abs/2203.07454) / [Dreamer](https://danijar.com/project/dreamer/) 那样每个 timestep 用一个独立 MLP 解码。Ablation (Table I) 显示这一改动在 manipulation 上从 0.82 → 0.90，在 quadruped 上从 19.47 → 22.78。直觉：trajectory-level structure（"这条路是不是接近 goal"）是 sequence-level property，per-timestep MLP 抓不到。

### 2.3 具体 hyperparameters

**Manipulation world model** (Table II)：
- Embedding dim 64
- Dynamics transformer：3 层，4 heads，hidden 256
- Reward/Value transformer：各 1 层 1 head
- Base policy transformer：4 层 8 heads
- Encoder：3 个 ResNet-18 (ImageNet pretrained) → 每个 512-d embedding → concat → MLP → 64-d $z_t$
- 控制频率 5 Hz

**Quadruped world model** (Table VII)：
- 同样 64-d latent
- CNN 处理 21×15 height map（kernel 3，stride 2,2,2，features 8/16/32）
- Dynamics transformer：2 层 8 heads
- 控制频率 50 Hz，在 RTX 4090M laptop 上跑

---

## 3. Pretraining in Simulation: 怎么蒸馏出 "可规划" 的 world model

### 3.1 Expert policy 训练

先用 **PPO** [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) 在 sim 里训一个 state-based expert $\pi^e(s_t)$ + value $V^e(s_t)$。Manipulation 用 [Yin et al. 2026](https://openreview.net/forum?id=nAO9LcV7nE) 的 pipeline；quadruped 用 [IsaacLab](https://github.com/isaac-sim/IsaacLab) + 4096 parallel envs + 490M steps + terrain curriculum。

### 3.2 Diverse data generation（Algorithm 2，这是 paper 的一大亮点）

naive 做法是直接 rollout $\pi^e$ 收集数据，但这样 world model 只见过 expert 的 narrow manifold。real-world MPPI 会 sample 大量 **off-policy、sub-optimal** trajectory，expert-only data 完全 out-of-distribution，planning 就崩了。

SimDist 的做法：
1. 保存 PPO 训练过程中的所有 checkpoints $\{\pi^k\}_{k=1}^K$（quadruped 保存了 iter 0/50/100/.../2000 的 11 个）
2. reset 时 50% 概率随机 assign 一个 sub-optimal $\pi^k$，50% 概率用 expert $\pi^e$
3. 对每个 env，sample 一个 diagonal noise covariance $\Sigma_j = \text{diag}(\sigma)$，$\sigma_i \sim \mathcal{U}[\sigma_i^{min}, \sigma_i^{max}]$
4. rollout 中 **contiguous noise intervals**：从 $\mathcal{U}[1,5]$（manipulation）或 $\mathcal{U}[1,50]$（quadruped）采样 noise 长度，期间加 Gaussian noise 到 action；之后从 $\mathcal{U}[5,10]$（manipulation）或 $\mathcal{U}[25,500]$（quadruped）采样 clean 长度，期间不加 noise，交替进行
5. 同时记录 $b_t^e$（这个 action 是否来自未受扰动的 expert）和 $v_t = V^e(s_t)$（value target）

这样产生的 dataset $\mathcal{D}_{sim} = \{(o_t, a_t, b_t^e, r_t, v_t)\}_{t=0}^N$ 同时包含：expert 行为、early-checkpoint 笨拙行为、加了 noise 的扰动行为、恢复行为。这是 **broad state-action coverage**，让 world model 在 real-world planning 时看到的 trajectory 全部 in-distribution。

Manipulation：100K trajectories，36% 是 optimal expert；Quadruped：100M data points，55.7% 是 optimal expert。Quadruped data gen 在单张 RTX 4500 Ada 上 ~7 小时。

### 3.3 World model pretraining loss (Eq. 2)

$$
\mathcal{L}_t^{sim}(\theta) = \sum_{i=0}^{T} \Big(
\underbrace{\|\hat{z}_{t+i+1} - \text{sg}(E_\theta(o_{t+i+1}))\|_2^2}_{\text{latent dynamics}}
+ c_1 \underbrace{(\hat{r}_{t+i} - r_{t+i})^2}_{\text{reward}}
+ c_2 \underbrace{(\hat{v}_{t+i+1} - v_{t+i+1})^2}_{\text{value}}
+ c_3 \underbrace{\mathbb{1}_e(a_{t+i})\|\hat{a}_{t+i} - a_{t+i}\|_2^2}_{\text{BC}}
\Big)
$$

变量解释：
- $\text{sg}$：stop-gradient operator，latent dynamics loss 用的是 **next observation encode 出来的 target**，但对 $E_\theta$ 反向传播被截断——这样 $E_\theta$ 只通过 reconstruction / contrastive 之类的辅助 objective 学（实际上 paper 里 $E_\theta$ 是通过 reward / value loss 反向传梯度学的，因为 sg 只阻断 dynamics 路径的梯度，reward/value head 仍然 attend 到 $\hat{z}$）
- $c_1, c_2, c_3$：loss 权重，按照每个 target 的 range normalize 到同尺度
- $\mathbb{1}_e(a_t)$：indicator，**仅当** $a_t$ 来自 uncorrupted expert $\pi^e$ 时为 1，否则为 0。这避免 BC loss 被 noise 污染的 action 训练
- $r_t$：从 privileged simulator state 算的 dense reward
- $v_t = V^e(s_t)$：sim 里训好的 optimal value，作为 distillation target

注意 **没有 observation reconstruction loss**——ablation (Table I "Raw Obs. Reconstruction") 显示加 reconstruction 反而让 manipulation 从 0.90 跌到 0.32。直觉：reconstruct 高维视觉会逼着 latent 过度编码 task-irrelevant 细节（光照、背景），干扰 compact dynamics learning。这与 [DreamerV3](https://danijar.com/project/dreamerv3/) 一直带 reconstruction 不同——SimDist 完全放弃 reconstruction，全部信号来自 reward/value/dynamics 的 "behavioral" supervision。

Data augmentation： proprioception 加 zero-mean Gaussian noise；image 加 color jitter + Gaussian blur + random crop。这防止 overfitting 到 sim 的视觉纹理。

Pretraining：2 epochs，batch 256 (manip) / 512 (quadruped)，Adam lr $2 \times 10^{-4} \to 1 \times 10^{-4}$ cosine decay + 10K step linear warmup。Quadruped pretrain ~28 小时单卡。

---

## 4. Real-World Adaptation: 核心机制

### 4.1 只 finetune dynamics (Eq. 3)

$$
\mathcal{L}_t^{real}(\theta) = \sum_{i=0}^{T} \|\hat{z}_{t+i+1} - \text{sg}(E_\theta(o_{t+i+1}))\|_2^2
$$

with $C_\theta, E_\theta, R_\theta, V_\theta, \pi_\theta$ **frozen**, only $f_\theta$ finetunable。

关键：因为 $E_\theta$ 冻住，所以 $E_\theta(o_t)$ 给出的是**稳定的 latent target**——不像 pretraining 那样 $E_\theta$ 一直在动，target 漂移导致训练不稳。这就把 real-world learning 简化成 **regression**：把当前 latent $z_t$ + candidate action $a$ 映射到 next latent $\hat{z}_{t+1}$，target 是真实 next observation encode 出来的 $E_\theta(o_{t+1})$。

不再需要 reward learning（$R_\theta$ 冻住，从 sim 直接 transfer）、不再需要 value bootstrapping（$V_\theta$ 冻住，从 sim 直接 transfer）、不再需要 exploration（MPPI 在 latent space 自动 counterfactual reasoning）。

### 4.2 MPPI planning 部署

部署时用 **Model Predictive Path Integral (MPPI)** [Williams et al. 2016](https://arxiv.org/abs/1509.04841)，TD-MPC 风格：

1. 当前 observation $o_t$ → encode $z_t, h_t$
2. Base policy $\pi_\theta$ 输出 chunk $\hat{a}_{t:t+T-1}$，加 Gaussian noise 得到 $N$ 个 candidate action sequences（manip $N=250+100$，quadruped $N=450+22$）
3. 用 $f_\theta$ 一次性预测每条 candidate 的 $\hat{z}_{t+1:t+T}$
4. 用 $R_\theta, V_\theta$ 算每条 trajectory 的 return：
$$
\mathcal{R}(a_{t:t+T-1}) = \gamma^T \hat{v}_{t+T} + \sum_{s=t}^{t+T-1} \gamma^{s-t} \hat{r}_s
$$
5. MPPI importance-weighting：$w_i \propto \exp(\mathcal{R}_i / \text{temp})$，加权平均得到 executed action
6. Solver iterations（manip 3 次，quadruped 8 次）+ momentum

### 4.3 Iterative improvement (Algorithm 1)

```
for j = 1..J:
    collect M real rollouts with MPPI
    add to D_real
    while not converged:
        sample segments from D_real
        freeze C,E,R,V,π
        update f_θ minimizing L_real
```

每 20 episodes finetune 一次（manipulation），quadruped 也是 iterative。Real-world 总数据量：**manipulation 20 episodes ≈ 15-30 minutes；quadruped Foam 32.1 min，Slippery Slope 35.7 min**。

---

## 5. 实验结果

### 5.1 四个任务

**Manipulation** (UR5e, 3 cameras, 6-DoF end-effector)：
- **Peg Insertion**：16mm square peg（仿 [Factory](https://arxiv.org/abs/2205.03532)），pick + align + insert，Narrow (2cm×2cm) / Wide (35cm×35cm) 初始条件
- **Table Leg**：用 [FurnitureBench](https://github.com/ml-lab-snu/furniturebench) assets，pick + align + thread，需要 screw motion

**Quadruped** (Unitree Go2)：
- **Slippery Slope**：两个 $3^\circ$ 和 $5.7^\circ$ 的斜坡，表面 PTFE (Teflon)，脚裹 thermoplastic——极低摩擦；要求走 1.82m 通过两块板；speed 0.1/0.3/0.5 m/s
- **Foam**：两块 5cm 厚 memory foam，dynamics 不建模；走 3.00m；speed 0.2/0.7/1.2 m/s

### 5.2 Main results (Fig. 4 + Table IX)

**Peg Insertion (Wide)**：SimDist ~0.85 success vs Diffusion Policy ~0.35 vs IQL ~0.0 vs RLPD ~0.0 vs SGFT-SAC ~0.3
**Table Leg**：SimDist+BC ~0.7 vs all others < 0.3
**Slippery Slope 0.3 m/s**：SimDist 1.82±0.00m (5/5 success) vs IQL 0.39±0.56m (0/5) vs RLPD 0.34±0.05m (0/5) vs Pretrained 0.43±0.13m (0/5) vs Single-step BC 1.43±0.25m (1/5)
**Foam 1.2 m/s**：SimDist 3.00±0.00 (5/5) vs IQL 2.73±0.34 (3/5) vs Pretrained 1.70±0.99 (0/5)

整体 SimDist 大约 **2×** 优于 best baseline。

关键观察：
1. **Standard RL finetuning (IQL/RLPD) 经常 collapse**——catastrophic forgetting。RLPD 在 Foam 上直接 destabilize robot，没法报。
2. **SGFT-SAC** ([Yin et al. 2025](https://arxiv.org/abs/2502.02705)，transfer value from sim) 避免了 collapse 但 sample efficiency 仍比 SimDist 差很多——证明 **只 transfer value 不够，必须 transfer 完整 world model**（reward + dynamics）。
3. **Diffusion Policy + sim demos** 在 narrow init 上还能凑合，Wide 上完全崩——因为 BC 只能 mimic demo distribution，无法 generalize。Fig. 6 显示 SimDist 在 init condition 空间上的 success 覆盖远广于 Diffusion Policy。
4. **Demonstrations 帮助 SimDist+BC** 在 Table Leg 上提升明显——precise screw motion 难 transfer，demos 直接给 base policy 提供 action prior。

### 5.3 Value transfer 验证 (Fig. 5)

从同一个初始 condition teleop 一条 success 和一条 failure trajectory，画 $V_\theta(\hat{z}_t)$ 随时间变化。Success 的 value 单调上升，failure 的 value 一直低——证明 sim 训练的 $E_\theta + V_\theta$ 在 real 上能区分 good/bad trajectory，**zero-shot transfer 是真的 work 的**。这是 SimDist 整个故事成立的前提。

### 5.4 Dynamics adaptation 验证 (Fig. 7)

Slippery Slope 上：
- Pretrained dynamics loss 平均 0.076，finetune 后 0.019（降 4×）
- Pretrained model 预测 front-left foot 在 PTFE 上稳定 contact（错！），finetuned model 正确预测 slippage
- 视觉化 MPPI sampling：finetuned model sample 出来的 trajectory 显式 account for slip，pretrained model 完全错位

这正面验证了 **"dynamics finetune 改变了 planner 的 trajectory distribution"** 这件事，而不是单纯降低 loss number。

---

## 6. Ablations (Table I) — 直觉 build

**Scaling simulation data**：
- Full data → 50% → 10%
- Peg Insertion: 0.90 → 0.72 → 0.06
- Table Leg: 0.85 → 0.61 → 0.02
- Quadruped reward: 22.78 → 22.73 → 19.38

10% data 几乎完全崩盘——证明 **broad simulation pretraining 是必要条件**，real-world planning 需要足够覆盖。

**Data diversity (Expert-only)**：
- Peg 0.10 vs full 0.90（9× drop）
- Table Leg 0.05 vs 0.85（17× drop）
- Quadruped 16.68 vs 22.78

这是 paper 最强的 ablation 之一：**只 expert data 完全不行**，因为 MPPI 会 sample 大量 off-policy 的 sub-optimal action sequence，dynamics/reward/value 在这些 region 上完全 OOD。这正是 Section 3.2 设计 perturbation + sub-optimal checkpoint mixing 的原因。

**MLP reward/value models**：
- Peg 0.82 vs 0.90
- Table Leg 0.60 vs 0.85（差距很大）
- Quadruped 19.47 vs 22.78

per-timestep MLP 无法 capture trajectory-level structure——比如 "在 step 3 时的 action 是否会让 step 5 的状态进入 goal region" 这种 sequence property。

**Raw observation reconstruction**：
- Peg 0.32（巨大 drop！）
- Quadruped 23.34（略升）

Reconstruction 在 manipulation 上严重 hurt——latent 被迫编码 visual detail，dynamics 学习被干扰。Quadruped 略升是因为 height map 信息量较低，reconstruction 不算 harmful。这与 [Dreamer](https://danijar.com/project/dreamer/) 系列一直保留 reconstruction 形成鲜明对比——SimDist 主张 **behavioral-only supervision 比 reconstruction + behavioral 更好**。

---

## 7. 与 Related Work 的位置

### 7.1 跟 model-based RL 的关系

- **TD-MPC** [Hansen et al. 2022](https://arxiv.org/abs/2203.07454)：SimDist 直接 inherit TD-MPC 的 MPPI 框架和 world model 架构，但 TD-MPC 是 **online 从 scratch 训练**， SimDist **offload 一切到 sim pretrain + real 只 adapt dynamics**。
- **Dreamer / DreamerV3** [Hafner et al.](https://danijar.com/project/dreamer/)：autoregressive latent rollout + actor-critic on imagined trajectory。Dreamer 也需要从 scratch bootstrap，在 contact-rich real task 上经常 saturate。SimDist 用 privileged expert + checkpoint mixing 一次性产生 diverse 数据。
- **MBRL with uncertainty** ([PETS](https://arxiv.org/abs/1805.00909), [MBPO](https://arxiv.org/abs/1906.08253))：用 ensemble/uncertainty 防止 model exploit。SimDist 不需要——因为 reward/value frozen，dynamics finetune 就是 supervised regression，没有 exploit 问题。

### 7.2 跟 sim-to-real 的关系

- **Domain randomization**（[RMA](https://arxiv.org/abs/2107.04034), [Learning to Walk in Minutes](https://arxiv.org/abs/2208.07860)）：从 sim 训 robust policy 直接 deploy。SimDist 也用 DR（见 Table IV 的 friction/mass/stiffness randomization），但承认 zero-shot 不够，需要 adaptation。
- **Neural physics engines / system identification** ([Neural Robot Dynamics](https://arxiv.org/abs/2508.15755), [ContactNets](https://arxiv.org/abs/2104.14255))：在 high-fidelity sim 上加 residual。这些方法依赖 object pose / contact label，在 partial observability 下 brittle。SimDist 从 raw observation 学 latent dynamics，不需要这些 privileged labels。
- **SGFT** [Yin et al. 2025](https://arxiv.org/abs/2502.02705)：transfer value function from sim 到 real，用 SAC finetune。是 SimDist 的 baseline，证明 **只 transfer value 不够**。
- **Offline-to-online RL** ([RLPD](https://arxiv.org/abs/2302.04874), [IQL](https://arxiv.org/abs/2110.06169))：sample efficient model-free，但仍然 long-horizon credit assignment + value overestimation，在 low-data real 上 collapse。

### 7.3 跟 generative video world model 的关系

- **Genie / Genie 2** [DeepMind](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)、**DreamGen** [Jang et al. 2025](https://arxiv.org/abs/2505.12705)：用 internet video 训 latent dynamics，然后 inverse model 或 BC decode action。**fundamental limit**：action distribution 受 demonstration 限制，无法 improve beyond demos。SimDist 直接在 low-level action 上训 world model，可以 plan 出 demo 没见过的 action。
- **UniPi / UVDM / LDP** [Du et al.](https://arxiv.org/abs/2310.06114)、[Xie et al. 2025](https://arxiv.org/abs/2504.16925)：plan in video space，用 inverse dynamics 恢复 action。同样受 demo 限制。

### 7.4 跟 VLA 的关系

- **π0.5** [Physical Intelligence](https://arxiv.org/abs/2504.16054)、**OpenVLA** [Kim et al.](https://arxiv.org/abs/2406.09246)、**π0.6**：BC-based VLA，作为 baseline 出现在 Fig. 4。SimDist+BC 在 Table Leg 上结合了 BC 的 action prior 和 world model 的 planning——这是一个很有趣的 hybrid。

---

## 8. 直觉总结：为什么这个 works

1. **Modularity buys you free invariance**：world model 把 decision making 拆成 global (representation/reward/value) 和 local (dynamics)。global 在 sim-real 之间 invariant 是因为 **physics 中物体语义、距离、goal 关系不变**；local 变是因为 contact/friction/compliance 在两个域里不同。SimDist 显式 exploit 这个 asymmetry。

2. **Frozen encoder = stable target**：因为 $E_\theta$ 冻住，$E_\theta(o_t)$ 在 finetune 期间是 deterministic target。这避免了 Dreamer 那种 representation 飘移导致 value/reward 跟不上、训练不稳的经典坑。

3. **Frozen reward/value = no bootstrapping**：real-world 不需要重新 estimate value（最贵的一步），sim 的 $V^e$ 直接 transfer。Fig. 5 实证这个 transfer work。

4. **Diverse pretraining = broad planning coverage**：MPPI 在 real 会 sample 大量 sub-optimal action sequence，必须保证 dynamics/reward/value 在这些 region in-distribution。Section 3.2 的 checkpoint mixing + perturbation + recovery data 就是为此设计。Table I "Expert Data Only" 那一行（0.10 vs 0.90）实证这点。

5. **Chunked transformer + MPPI = tractable planning**：传统 autoregressive world model 一次一步 rollout，N candidate × T horizon = $O(NT)$ forward pass。SimDist 一次 forward 出 $T$ 步 → $O(1)$ forward per candidate，可以 batch 几百条 → 50 Hz quadruped control。

6. **MPPI > actor-critic for adaptation**：actor-critic finetune 要 bootstrap policy 和 critic 同时改，critic overestimation 导致 policy collapse。MPPI 直接用 frozen $V_\theta$ 做 terminal value，sample → weighted average，**没有 policy gradient，没有 critic update，没有 exploitation of model error**。

---

## 9. Limitations / Open Questions

Paper 自己提到：
- 只测了 single-task + high-fidelity sim。multi-task world model + 通用 sim 还没做。
- 15-30 min real data 这个数字依赖 sim fidelity；如果 sim 很差（比如连基本 contact 都不准），pretrain 出来的 $E_\theta$/$V_\theta$ 可能根本 transfer 不过去。

我会问的几个问题：
- **What if $E_\theta$ encode 的特征在 real 是 OOD？**（比如 sim 用的材质纹理在 real 完全不同）。Paper 里 data augmentation (color jitter/blur/crop) 缓解，但本质上这是 representation transfer 的 fragile 点。如果 $E_\theta$ 在 real 上 mis-encode，整个 dynamics finetune 就 garbage-in-garbage-out。
- **Frozen $R_\theta$ 真的 robust 吗？** sim reward 是 privileged state-based（比如 peg-to-hole 距离），transfer 到 real 后 $E_\theta$ encode 出来的 $z$ 是否真的 capture 同样的几何？Fig. 5 实证 peg 上 work，但更复杂的 contact rich task（比如 deformable object manipulation）可能不行。
- **Long-horizon task 怎么办？** 当前 $T=5$ (manip) / $T=25$ (quadruped) 是 short-horizon planning。如果任务是 assembly 30 步 sequential sub-task，$T$ 不够。可能的解法是 hierarchical planning（skill-level world model）。
- **MPPI sample efficiency vs learned planner**：MPPI 要 sample 几百条 trajectory 才收敛，compute 大。一个 learned policy head 可以 distill 出来 offline，但 paper 没做。
- **能不能跟 VLA 结合？** 比如 $\pi_0.5$ 作为 base policy，SimDist world model 作为 verifier / planner——这听起来很像最近 RLHF + VLA 的方向。

---

## 10. Code & 项目链接

- Project page: https://sim-dist.github.io
- TD-MPC (架构基础): https://arxiv.org/abs/2203.07454
- RLPD: https://arxiv.org/abs/2302.04874
- IQL: https://arxiv.org/abs/2110.06169
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- π0.5: https://arxiv.org/abs/2504.16054
- SGFT: https://arxiv.org/abs/2502.02705
- FurnitureBench: https://github.com/ml-lab-snu/furniturebench
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- Factory (Peg insertion task): https://arxiv.org/abs/2205.03532
- MPPI (Williams et al.): https://arxiv.org/abs/1509.04841
- PPO: https://arxiv.org/abs/1707.06347
- AnyCar (chunked prediction 灵感): https://arxiv.org/abs/2503.06815

---

**TL;DR**: SimDist 把 sim-to-real 重新 cast 成一个 modular system identification 问题：sim 里用 PPO + checkpoint mixing + action perturbation 蒸馏出 broad-coverage 的 latent world model，real 里冻住 representation/reward/value，只 finetune dynamics 的 transformer，用 MPPI 在 latent space 做 planning。15-30 min real data 就能在 precise manipulation + quadruped locomotion 上达到 ~2× 优于 IQL/RLPD/Diffusion Policy/π0.5/SGFT 的性能。Key insight: world model 的 modularity 让 sim 和 real 之间 "global structure invariant, local dynamics vary" 的 asymmetry 可以被显式 exploit，从而避开 end-to-end finetune 的 catastrophic forgetting 和 long-horizon credit assignment。
