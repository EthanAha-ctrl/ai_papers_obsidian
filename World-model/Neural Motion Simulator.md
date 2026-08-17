---
source_pdf: Neural Motion Simulator.pdf
paper_sha256: 4e795d657cb1c2ca02111c837fd1ea08d0573949cec36c800e58f5bf2a1fb483
processed_at: '2026-08-05T22:20:48-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MoSim 用人话说

## 一句话版本

**他们做了一个"物理引擎替身"**，学会了预测机器人怎么动，然后你可以在它"脑子里的想象世界"里训练RL agent，不用碰真实环境。

---

## 为什么要做这件事

先讲一个观察：动物和人类能在脑子里"预演"动作。你要抓个杯子，脑子里先跑一遍抓的动作，觉得OK了才执行。这个"脑子里的simulator"就是world model。

RL圈一直在做world model，但有个尴尬的问题 —— **没人认真测过world model到底预测得准不准**。

传统做法是：拿world model去跑个RL任务，任务表现好就默认world model好。这逻辑有点像"我徒弟打架赢了，说明我武功高"，太indirect了。

MoSim团队说：咱直接测预测精度吧，给定当前state和action，你预测100步之后的state，跟ground truth比，看差多少。

结果一测发现 —— 现有的world model（DreamerV3、TD-MPC2那些）**其实预测得挺烂的**。DreamerV3在Cheetah上预测16步，MSE是0.87；MoSim做到0.12，差不多**7倍好**。在Reacher上更夸张，100步预测，MoSim比DreamerV3好**100倍**。

---

## 他们怎么做到的

核心idea其实很朴素，分三层：

### 第一层：把物理公式拆开，让网络各学各的

机器人运动有一套经典公式，叫manipulator equation（[link](https://www.cs.cmu.edu/~jbruce/thesis/ch3-kinematics.pdf)），大概长这样：

$$\ddot{q} = M(q)^{-1}[\tau + b(q, \dot{q}) + c(q, \dot{q}, a)]$$

翻译成人话：
- $\ddot{q}$: 各关节的加速度（你想算的东西）
- $M(q)^{-1}$: 惯量矩阵的逆 —— "我有多重，给我这个力能产生多少加速度"
- $\tau$: 电机施加的力（action）
- $b$: 重力、弹性力这些"被动"的力
- $c$: 碰撞、摩擦这些"交互"的力 —— 最难建模

MoSim的策略：**前三项用神经网络结构化地学，最后一项用一个"corrector"网络兜底**。

具体来说：
- $M(q)$ 用ResNet学，但输出强行rearrange成lower triangular matrix $L$，然后算 $M = LL^T$。这样保证 $M$ 是symmetric positive definite —— 这是惯量矩阵的物理要求
- $b$ 用ResNet学
- $\tau$ 用MLP学
- $\epsilon$（corrector，吸收了$c$和所有unmodeled stuff）用另一个ResNet学

**intuition**: 你给网络一个"骨架" —— "加速度 = (惯量) × (重力 + 电机力 + 残差)"，让它在这个框架里填血肉。网络不用从零学"F=ma"，只需要学"这个具体机器人的惯量长什么样"。

这跟你nanoGPT的精神一样：**别让网络学它本来就该知道的**。

### 第二层：用Neural ODE做时间积分

物理是连续的。传统RL里一个step就是离散的一帧，但真实物理是 $dt \to 0$ 的连续过程。

Neural ODE（[link](https://arxiv.org/abs/1806.07366)）的idea：把"从 $t_0$ 到 $t_1$ 的状态变化"formulate成一个积分：

$$s(t_1) = s(t_0) + \int_{t_0}^{t_1} g_\theta(s(t), t) \, dt$$

然后让数值积分器（DOPRI5，[link](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method)）去算这个积分。DOPRI5是adaptive的 —— 平滑的地方用大步长，剧烈变化的地方自动用小步长。

**好处**:
- 模型学会的是连续dynamics，换frame rate不用重训
- 可以预测任意时间长度
- backprop用adjoint method，memory是O(1)（[link](https://arxiv.org/abs/1806.07366)）

**代价**: compute贵3倍左右，因为要跑forward + backward adjoint + backward gradient三次积分。

### 第三层：Multi-stage训练

先train predictor（学smooth dynamics: 惯量、重力），收敛了freeze住，再train corrector（学non-smooth: 摩擦、碰撞）。

**为什么**: 如果一开始joint train，corrector会"偷懒"去学那些predictor本该学的smooth部分，结果predictor没学好，corrector也学不好。分阶段train让predictor先把"骨架"立起来，corrector只补"血肉"。

Ablation（Figure 5）证明这招确实work —— 收敛更快，final MSE更低。

---

## 他们怎么评估的

这是paper我最喜欢的部分。他们设计了一个**直接评估world model**的benchmark：

1. 拿DeepMind Control Suite（[link](https://github.com/google-deepmind/dm_control)）当ground truth
2. 生成两种test data: random policy的 + pretrained agent的
3. 给model一个initial state和一段action序列，让它连续预测3步、16步、100步
4. 算prediction和ground truth的MSE

就这幺简单直接。没有downstream task包装，没有surrogate metric，就是"你预测得准不准"。

---

## 结果有多impressive

### Table 1: MoSim vs DreamerV3

挑几个dramatic的:

| 任务 | Horizon | DreamerV3 | MoSim | 倍数 |
|---|---|---|---|---|
| Reacher | 100 | 0.0988 | 0.0009 | **110x** |
| Panda | 100 | 0.0971 | 0.0043 | **23x** |
| Go2 | 100 | 0.4165 | 0.1401 | **3x** |
| Humanoid | 16 | 6.51 | 1.27 | **5x** |

简单系统碾压，复杂系统显著提升但没那么夸张 —— reasonable，因为Humanoid的contact dynamics非常复杂。

### Table IV: 等价prediction horizon

用DreamerV3在16步的MSE当benchmark，看MoSim多少步才降到同样的MSE：

- Reacher: **>1000步**（DreamerV3 16步的精度，MoSim保持到1000步以上）
- Panda: **>1000步**
- Cheetah: 60步
- Humanoid: 42步

**Reacher和Panda这种简单系统，MoSim基本能"永远"预测下去**。这是zero-shot RL的必要条件。

### Table 6: Lyapunov exponent

Acrobot是chaotic system（对initial condition敏感）。他们测了Lyapunov characteristic exponent（LCE，[link](https://en.wikipedia.org/wiki/Lyapunov_exponent)）：

- MuJoCo ground truth: 1.1738
- MoSim: 1.1728

几乎一样。说明MoSim不仅point prediction准，**连"蝴蝶效应"的发散速率都学对了**。这是很深层次的physical fidelity。

---

## Zero-shot RL：最有野心的实验

### 概念

拿一个pre-trained MoSim（在random data上训练的），**完全替代**real environment。让model-free RL algo（SAC、TQC）在MoSim里训练policy。训练完了，把policy直接丢到real env测，看work不work。

如果work，就是zero-shot RL —— **完全不需要real environment interaction**。

### 成功的case (Figure 3a-d)

- Reacher-Easy: 完美
- Reacher-Hard: 完美
- Cartpole-Balance: 完美
- Acrobot-SwingUp: 基本成功

这些任务的特点：**prediction horizon需求短**。Reacher只需要30步horizon，MoSim能轻松到1000+步，margin巨大。

### 失败的case: Cheetah-Run (Figure 3e)

Cheetah-Run需要500步horizon才能训出好policy，但MoSim只能稳定预测~60步。结果：训练到score~100后开始**decline**，因为model error累积导致policy学到了wrong dynamics。

他们诊断了两个root cause:

**问题1**: Prediction horizon不够。TQC算法在real env上time limit=100时都学不会Cheetah-Run，MoSim只能预测60步，显然不够。

**问题2**: Distribution shift。训练过程中policy变好，generated state distribution偏离MoSim训练时的random data分布。MoSim在OOD region预测不准，policy学到错东西。

### Few-shot: 部分缓解

策略：每5000 virtual steps，收集1000 real steps更新MoSim。同时从real replay buffer采样initial state，模拟longer episode。

结果：训练stabilize在score~100，不再decline，但也没法进一步提升。

### 最cool的idea: 让model知道自己不知道 (Section 3.7)

用一个normalizing flow（[link](https://arxiv.org/abs/1912.02762)）拟合训练数据分布。当policy走到低概率密度区域（OOD），给一个penalty：

$$R_{penalty} = \sigma\left(\frac{\log P(s) - \tau}{\alpha}\right) - 1$$

- $\log P(s)$: 当前state在flow model下的log概率
- $\tau$: threshold（"多低算OOD"）
- $\alpha$: scaling
- $\sigma$: sigmoid，把penalty压到$[-1, 0]$

**intuition**: model在熟悉区域预测准，让policy多待在那；不熟悉的区域，penalty推policy回来。

Figure 3f显示加penalty后，Cheetah训练reward不再decline，甚至超过之前峰值。这是一个非常有潜力的方向 —— **uncertainty-aware world model RL**。

---

## 我的解读

### 这篇paper的真正贡献

不是architecture多novel（rigid body + Neural ODE之前都有人做过），而是：

1. **提出直接评估world model的protocol** —— 这个benchmark本身就有价值
2. **证明world model质量直接决定RL性能** (Figure 6: horizon 10/50/100对应性能递增)
3. **首次show zero-shot RL在部分任务上可行**
4. **诚实分析failure mode**（Cheetah-Run的distribution shift）

### 为什么MoSim能赢

回到你的"bitter lesson"思考（[link](https://karpathy.ai/lesson.html)）—— Sutton说compute + general method最终赢，但MoSim似乎违反了这个规律：它用**更多inductive bias**赢了DreamerV3的**更大网络**。

我的解读：在data稀缺的regime，inductive bias很重要。MoSim在robotics这种structured problem里，physics prior的价值远超堆参数。但如果是language这种unstructured problem，bitter lesson可能仍然hold。

这跟你做nanoGPT的精神一致：**先理解structure，再谈scale**。

### 开放的big questions

1. **Contact-rich tasks**: Humanoid只有42步horizon，离locomotion需要的1000步还远。corrector够吗？要不要显式model contact？
2. **Vision**: 当前只处理proprioceptive state（关节角度等），没有pixel input。真实robot需要vision。
3. **Sim-to-real**: 全在simulation里做的，真实robot的sensor noise、actuator delay怎么办？
4. **Compute cost**: Neural ODE比discrete model慢3-5x，能实时planning吗？
5. **Morphology generalization**: 训练了Cheetah的MoSim能transfer到Humanoid吗？还是每个robot都要重训？

### 更深层的connection

这篇paper其实呼应了LeCun的JEPA vision（[link](https://openreview.net/pdf?id=BZ5a1r-kVsf)）: world model应该是predictive的，不一定要generative（不一定要generate pixels）。MoSim只predict state，不generate image，正符合这个思路。

也呼应了Schmidhuber早期的world model工作（[link](https://arxiv.org/abs/1406.2682)）和Sutton的Dyna架构（[link](https://dl.acm.org/doi/10.1145/122344.122377)）—— 在learned model里planning这个idea有30年历史了，MoSim是把它做到足够precise的开始。

---

## 一句话总结

**MoSim把rigid body physics的structure塞进神经网络，用Neural ODE做连续时间积分，在raw state space做出了比DreamerV3/TD-MPC2准100倍的long-horizon prediction，第一次show了zero-shot RL在简单任务上可行，也诚实揭示了复杂任务上distribution shift的challenge。**

核心takeaway: **world model的prediction accuracy是model-based RL的真正bottleneck，不是RL算法本身。**

---

## Resources

- Paper项目页: https://oamics.github.io/mosim_page/
- Neural ODE: https://arxiv.org/abs/1806.07366
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DM Control: https://github.com/google-deepmind/dm_control
- MuJoCo: https://mujoco.org/
- Normalizing Flows: https://arxiv.org/abs/1912.02762
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Sutton Dyna: https://dl.acm.org/doi/10.1145/122344.122377
- Karpathy's bitter lesson: https://karpathy.ai/lesson.html

---

# Neural Motion Simulator (MoSim): 深度解析

## 1. 这篇paper想解决什么问题

传统world models在RL中通常被**indirectly evaluated** — 我们看下游任务表现来推测world model好不好。这导致一个核心问题被忽略：world model本身到底有多擅长做它该做的事，即**给定当前state和action，预测未来state**？

MoSim团队的核心hypothesis很清晰：如果world model的prediction horizon和accuracy足够，我们能否**完全在predicted space里训练或搜索新policy**，实现zero-shot RL？更进一步，能否把任何model-free RL algorithm直接wrap成model-based？

这是一个非常"Karpathy式"的问题 — 你关于"World Models"的博客（[link](https://worldmodels.github.io/)）和LeCun的JEPA架构（[link](https://openreview.net/pdf?id=BZ5a1r-kVsf)）都指向这个方向：agent需要内部simulator来做planning。

---

## 2. 核心Architecture：Predictor + Corrector + Neural ODE

### 2.1 Physical State的定义

$$s(t) := (q_1, q_2, \ldots, q_n, \dot{q}_1, \dot{q}_2, \ldots, \dot{q}_n)^T = (q^T, \dot{q}^T)^T$$

变量解释：
- $s(t)$: physical state vector at time $t$
- $q \in \mathbb{R}^n$: generalized coordinates (joint angles, spatial positions等)
- $\dot{q} \in \mathbb{R}^n$: generalized velocities
- $n$: degrees of freedom

这是机器人学里的标准configuration space表示。**关键insight**: MoSim在raw state space预测，而DreamerV3/TD-MPC2在latent space预测。这决定了它能否被plug进任意model-free algorithm。

### 2.2 Dynamics Decomposition

核心公式(1)：
$$\dot{s}(t) = f(s(t), a(t)) + \epsilon(s(t), a(t))$$

变量含义：
- $\dot{s}(t)$: time derivative of state (即velocity和acceleration的stack)
- $f$: **Predictor** — deterministic rigid body dynamics
- $\epsilon$: **Corrector** — 残差项，捕捉noise, friction, contact等unmodeled因素
- $a(t)$: action vector (joint torques等)

这是一种**residual modeling**思路，类似ResNet的identity skip connection哲学：让网络只学"难的部分"，容易的部分由inductive bias解决。

### 2.3 Rigid Body Dynamics的Inductive Bias

理想刚体动力学有显式形式（公式3）：
$$\dot{s}_{ideal} = \begin{pmatrix} \dot{q} \\ M(q)[b(q, \dot{q}) + \tau(a) + c(q, \dot{q}, a)] \end{pmatrix}$$

变量详解：
- $M(q)$: **inverse inertia matrix** (注意是inverse! 不是inertia matrix本身)，仅依赖position $q$，symmetric positive definite
- $b(q, \dot{q})$: conservative forces vector (gravity, elastic forces等)
- $\tau(a)$: applied torques from action
- $c(q, \dot{q}, a)$: constraint forces + contact forces (collisions, friction) — 这部分**很难显式建模**

将$c$吸收进$\epsilon$后得到公式4：
$$\dot{s}(t) = \begin{pmatrix} \dot{q} \\ M(s)[b(s) + \tau(a)] \end{pmatrix} + \epsilon(s(t), a(t))$$

**这里的物理直觉**：$M(q)$告诉你"给定力，加速度是多少" — 这是robotics的manipulator equation的核心。$b$是"被动"项（重力总是存在），$\tau$是"主动"项（actuator施加的力），$c$是"交互"项（contact，最难学）。

### 2.4 网络参数化

每个component用不同网络实现（见Table II）：

| Module | Network Type | Input | Output |
|---|---|---|---|
| Position Encoder $M$ | ResNet + Rearrange | $(B, D_q)$ | $(B, D_v \times (D_v+1)/2)$ → $(B, D_v, D_v)$ |
| State Encoder $b$ | ResNet | $(B, D_q + D_v)$ | $(B, D_v)$ |
| Action Encoder $\tau$ | MLP | $(B, D_a)$ | $(B, D_v)$ |
| Corrector $\epsilon$ | ResNet | $(B, D_q + D_v + D_a)$ | $(B, D_v)$ |

#### Cholesky Decomposition的关键trick

$M$必须是symmetric positive definite (SPD)。如何用神经网络保证SPD？

利用Cholesky decomposition的性质（[link](https://en.wikipedia.org/wiki/Cholesky_decomposition)）：任何SPD matrix可分解为 $M = LL^T$，其中$L$是lower triangular matrix。

实现步骤：
1. ResNet输出长度为 $D_v(D_v+1)/2$ 的vector（正好是lower triangular matrix的元素个数）
2. Rearrange成 $D_v \times D_v$ lower triangular matrix $L$
3. 计算 $M = LL^T$ → 自动保证SPD

这是一个非常优雅的inductive bias注入方式。**对比**: Hamiltonian Neural Networks（[link](https://arxiv.org/abs/1906.01563)）用symplectic structure保证energy conservation，思路类似但目标不同。

### 2.5 Neural ODE做时间积分

这是MoSim最关键的设计选择之一。标准Neural ODE形式（公式5-6）：
$$\frac{dz(t)}{dt} = g_\theta(z(t), t)$$
$$z(t_1) = z(t_0) + \int_{t_0}^{t_1} g_\theta(z(t), t) dt$$

变量：
- $z(t)$: latent state (在MoSim里就是 $s(t)$)
- $g_\theta$: dynamics function parameterized by $\theta$
- $t_0, t_1$: 起止时间

**为什么用Neural ODE而不是discrete RNN?**

1. **连续时间**: 物理本身是连续的，discrete step是人为的。Neural ODE让模型学会连续dynamics，可以任意调整step size
2. **Adaptive integration**: 用DOPRI5 integrator（[link](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method)），自适应step size，刚性区域用小步长，平滑区域用大步长
3. **Memory efficient backprop**: adjoint method（公式7-8）避免了存所有中间states

#### Adjoint Method详解

公式7：
$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} \alpha(t) \frac{\partial g_\theta(z(t), t)}{\partial \theta} dt$$

公式8：
$$\alpha(t_0) = \alpha(t_1) - \int_{t_1}^{t_0} \alpha(t) \frac{\partial g_\theta(z(t), t)}{\partial z(t)} dt$$

变量：
- $L$: loss function
- $\alpha(t) := \frac{\partial L}{\partial z(t)}$: **adjoint state** — loss对中间hidden state的梯度
- $\frac{\partial g_\theta}{\partial \theta}$: dynamics对参数的Jacobian
- $\frac{\partial g_\theta}{\partial z}$: dynamics对state的Jacobian

**直觉**: 标准backprop需要存所有中间states（$O(N)$ memory），adjoint method把"反向传播"也formulate成一个ODE，从$t_1$积分回$t_0$，memory是$O(1)$。代价是compute cost约翻3倍（需要3次积分：forward, backward adjoint, backward gradient）。

在MoSim中：
$$g_\theta(s(t), t) = f_\theta(s(t), a(t)) + \sum_i \epsilon_\theta^i(s(t), a(t))$$

即predictor + 多个correctors的叠加。

---

## 3. Multi-Stage Training Strategy

### 3.1 核心思想

```
Stage 1: Train Predictor only (smooth dynamics: inertia, gravity, elastic)
    ↓ (converge)
Stage 2: Freeze Predictor, Train Corrector (non-smooth: friction, contact)
    ↓ (for complex robots)
Stage 3: Add more Correctors for stepwise refinement
```

### 3.2 为什么有效？

这是一种**curriculum learning**。Predictor有强inductive bias（rigid body structure），能快速学到"easy"的smooth dynamics。如果一开始joint train，corrector可能"偷懒"学一些本该predictor学的smooth部分，导致整体suboptimal。

Figure 5的ablation study在Hopper-Hop上验证：multi-stage比end-to-end训练**更快收敛且final performance更好**。

这与你的"micrograd"教学哲学一致：理解inductive bias的作用比堆参数重要。

---

## 4. Benchmark设计：直接评估World Model

### 4.1 评估协议

| Horizon | 对应模型 | 意义 |
|---|---|---|
| 3 steps | TD-MPC2 | 短期planning |
| 16 steps | DreamerV3 | 中期imagination |
| 100 steps | Long-horizon | 真正的planning能力 |

评估方式：
- 给定initial state + action sequence
- 连续预测指定horizon
- 计算prediction与ground truth的MSE

### 4.2 两种测试数据

1. **Random policy (Poisson sampling)**: 覆盖广阔state-action space
2. **Pretrained RL agent policy**: task-specific, meaningful data

**关键发现**: 在random data上训练的model **泛化更好** (Table 2)！这符合你的"更宽泛数据 = 更好world model"直觉，也呼应了LeCun关于JEPA需要diverse training的论点。

---

## 5. 实验结果深度解析

### 5.1 Raw State Space: MoSim vs DreamerV3 (Table 1)

以Reacher 100步为例：
- DreamerV3-r (5-step warmup): 0.0988
- MoSim-r: 0.0009 → **约100x提升**

以Panda 100步（机械臂）：
- DreamerV3-r: 0.0971
- MoSim-r: 0.0043 → **约23x提升**

以Go2 100步（四足机器人）：
- DreamerV3-r: 0.4165
- MoSim-rm: 0.1401 → **约3x提升**

**规律**: 简单系统提升巨大，复杂系统提升相对小。这很reasonable — Humanoid有高DoF，contact dynamics复杂。

### 5.2 Latent Space: MoSim vs TD-MPC2 (Table 3)

Reacher在TD-MPC2 latent space：
- TD-MPC2 (its own latent space!): 4.81e-5
- MoSim (encode predicted raw state): 2.93e-7 → **约160x提升**

这特别striking — MoSim甚至比TD-MPC2在自己设计的latent space里更准！这暗示raw state space + 强inductive bias > learned latent space + 弱inductive bias。

### 5.3 Stochasticity与Chaos (Table 6)

Lyapunov Characteristic Exponent (LCE):
- MuJoCo (ground truth): 1.1738
- MoSim: 1.1728

LCE衡量chaotic system的"蝴蝶效应"程度（[link](https://en.wikipedia.org/wiki/Lyapunov_exponent)）。正值表示chaotic，相近值表示MoSim捕捉到了正确的divergence rate。

这意味着MoSim不仅是point prediction准，连**chaotic性质都preserved**。这通过ensemble sampling实现 — 不同corrector初始化产生不同trajectory，ensemble分布与true divergence一致。

### 5.4 Prediction Horizon对比 (Table IV)

| Environment | DreamerV3 (16-step MSE) | MoSim等价horizon |
|---|---|---|
| Cheetah | 16步 | 60步 (3.75x) |
| Reacher | 16步 | >1000步 (62.5x+) |
| Panda | 16步 | >1000步 |
| Humanoid | 16步 | 42步 |

Reacher和Panda这种简单系统，MoSim能预测**几乎无限长**。这是zero-shot RL的必要条件。

---

## 6. Zero-Shot RL：核心实验

### 6.1 概念

用pre-trained MoSim (在random data上) **完全替代**real environment，让model-free RL algorithm在MoSim里训练。如果在MoSim里学到的policy直接在real env上work，就是zero-shot。

### 6.2 成功案例 (Figure 3a-d)

- Reacher-Easy: 完美zero-shot
- Reacher-Hard: 完美zero-shot
- Cartpole-Balance: 完美zero-shot
- Acrobot-SwingUp: 大部分成功

### 6.3 Cheetah-Run的失败模式 (Figure 3e)

两个fundamental challenges:

**Challenge 1: Prediction Horizon Insufficiency**
- MoSim在Cheetah-Run上稳定预测约100步
- 但model-free TQC算法在real env time limit=100时**根本学不会**
- Table 7: 实际需要500步，MoSim只达到60步

**Challenge 2: Distribution Shift**
- 训练过程中policy变化，生成的state distribution偏离MoSim训练数据
- Figure 3e: 训练到score~100后开始**degrade**

这是model-based RL的经典failure mode，对应你之前讨论过的"model bias accumulates over rollout"问题。

### 6.4 Few-Shot: 部分缓解

策略：每5000 virtual steps收集1000 real steps，update MoSim。同时从real replay buffer采样initial state。

结果：训练stabilize在score~100，但无法进一步提升。**说明distribution shift被缓解但prediction horizon问题未解决**。

### 6.5 让Model知道自己不知道 (Section 3.7)

这是paper最forward-looking的部分。用Residual Flow拟合训练数据分布，将log probability density作为RL reward的penalty：

$$R = R_{original} + R_{penalty}$$

$$R_{penalty} = \sigma\left(\frac{\log P(s) - \tau}{\alpha}\right) - 1$$

变量：
- $\log P(s)$: state在flow model下的log概率密度
- $\tau$: 自定义inflection point (threshold)
- $\alpha$: scaling factor
- $\sigma$: sigmoid function，将penalty限制在$[-1, 0]$

通过normalizing flow（[link](https://arxiv.org/abs/1912.02762)）计算：
$$\log P(s) = \log P_{base}(x) + \log|\det J|$$
$$x, \log|\det J| = f^{-1}(s)$$

- $f^{-1}$: flow model的inverse transform
- $P_{base}(x)$: base distribution (Gaussian)
- $J$: Jacobian of transformation

**直觉**: 当state在训练分布外时，$\log P(s)$低，penalty接近-1，policy被"推回"已知区域。这有点像uncertainty-aware RL ([link](https://arxiv.org/abs/2008.02261))，但用generative model而不是ensemble variance。

Figure 3f显示加penalty后，Cheetah训练reward不再decline，甚至超过之前峰值。

---

## 7. Ablation Studies

### 7.1 Inductive Bias的必要性 (Figure 4)

对比MoSim predictor vs 标准ResNet (相同参数量)在Hopper-Hop上：
- MoSim predictor (with rigid body structure): 快速收敛到低MSE
- Plain ResNet: 收敛慢，final MSE高

**这是最重要的ablation** — 证明了"structure > scale"的thesis。

### 7.2 Multi-Stage Training (Figure 5)

对比full training vs multi-stage，相同predictor+corrector架构：
- Multi-stage: 更快收敛，更稳定
- End-to-end: 训练前期波动大

### 7.3 Prediction Horizon影响RL (Figure 6)

用horizon=10, 50, 100训练policy：
- Horizon=100: 最好
- Horizon=50: 中等
- Horizon=10: 最差

**这量化了"world model quality直接决定RL性能"**。

---

## 8. 与相关工作的positioning

### 8.1 vs DreamerV3 ([link](https://arxiv.org/abs/2301.04104))
- DreamerV3: latent space (RSSM)，pixel-based
- MoSim: raw state space，structure-aware
- DreamerV3需要reconstruction loss；MoSim不需要

### 8.2 vs TD-MPC2 ([link](https://arxiv.org/abs/2310.16828))
- TD-MPC2: latent space + MPC planning
- MoSim: raw space + 任意RL algo
- TD-MPC2 horizon=3；MoSim可达100+

### 8.3 vs Differentiable Simulation ([link](https://arxiv.org/abs/2011.05607))
- Differentiable sim: hand-coded physics + gradients
- MoSim: data-driven + structural prior
- MoSim不需要精确contact modeling

### 8.4 vs PILCO ([link](https://arxiv.org/abs/1206.3935))
- PILCO: Gaussian Process dynamics, sample efficient但scale差
- MoSim: Neural ODE, scale好

---

## 9. My Critical Assessment

### 9.1 Strengths

1. **Direct evaluation protocol**: 终于有人直接measure world model quality
2. **Inductive bias design**: Cholesky + rigid body structure非常elegant
3. **Zero-shot成功case**: Reacher/Cartpole证明concept可行
4. **Honest about limitations**: Cheetah-Run失败分析透彻

### 9.2 Open Questions

1. **Contact-rich tasks**: Humanoid只到42步horizon，real locomotion需要1000+
2. **Visual observations**: 当前只处理proprioceptive state，无vision
3. **Real robot transfer**: 全在simulation，sim-to-real未验证
4. **Computational cost**: Neural ODE + adjoint method比discrete model慢3-5x
5. **Multi-contact**: Corrector能否捕捉stick-slip friction, impact dynamics等complex phenomena?

### 9.3 Connections to Broader Themes

这与你的micrograd/nanoGPT哲学呼应：**简单架构 + 强inductive bias > 复杂黑箱**。MoSim没有用transformer，没有用diffusion，就是ResNet + Neural ODE + 物理structure，却beat了SOTA。

这也呼应LeCun的JEPA vision: world model应该是non-generative, predictive的。MoSim不generate pixels，只predict states，正符合。

更深层的connection到Schmidhuber的RNN world models ([link](https://arxiv.org/abs/1406.2682))和你的blog post "The Bitter Lesson" — 算法进步很重要，但inductive bias + computation同样关键。

### 9.4 Future Directions I'd Pursue

1. **Combine with Vision**: 用JEPA-style encoder处理pixel input，MoSim在latent space操作但保持物理structure
2. **Hierarchical Correctors**: 不同corrector处理不同时间尺度（contact vs inertia）
3. **Meta-learning Correctors**: Quick adaptation到新robot morphology
4. **Uncertainty Calibration**: 把Residual Flow penalty扩展到全trajectory level
5. **Symmetry Exploitation**: SE(3) equivariance ([link](https://arxiv.org/abs/2104.01161)) for locomotion

---

## 10. Code & Resources

- 项目主页: https://oamics.github.io/mosim_page/
- Neural ODE原paper: https://arxiv.org/abs/1806.07366
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DM Control Suite: https://github.com/google-deepmind/dm_control
- MuJoCo: https://mujoco.org/
- MuJoCo Menagerie (Panda, Go2): https://github.com/google-deepmind/mujoco_menagerie
- ResNet: https://arxiv.org/abs/1512.03385
- SAC: https://arxiv.org/abs/1801.01290
- TQC: https://arxiv.org/abs/2005.07504
- Cholesky Decomposition: https://en.wikipedia.org/wiki/Cholesky_decomposition
- DOPRI5: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
- Normalizing Flows: https://arxiv.org/abs/1912.02762
- Hamiltonian Neural Networks: https://arxiv.org/abs/1906.01563
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Sutton's Dyna: https://dl.acm.org/doi/10.1145/122344.122377
- Schmidhuber's World Models: https://arxiv.org/abs/1406.2682

---

## 总结：Build Your Intuition

MoSim的core message: **World model quality, directly measured by long-horizon prediction accuracy, is the bottleneck for model-based RL**。一旦prediction horizon足够长（比如Reacher的1000+步），zero-shot RL就从theory变成practice。

它的recipe很简洁：
1. Physical inductive bias (rigid body equations)
2. Residual learning (Predictor + Corrector)
3. Continuous-time modeling (Neural ODE)
4. Multi-stage curriculum training
5. Wide data distribution (random sampling > task-specific)

这与你反复强调的"simple things done well"完全一致。MoSim没有用fancy的attention或diffusion，但通过**正确地injecting physics structure**，在long-horizon prediction上超越了DreamerV3和TD-MPC2。

下一步要攻克的显然是contact-rich tasks (Humanoid, dexterous manipulation)和visual observations — 这些是真正通往general embodied AI的bottleneck。
