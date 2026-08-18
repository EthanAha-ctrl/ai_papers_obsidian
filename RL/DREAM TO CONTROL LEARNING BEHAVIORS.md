---
source_pdf: DREAM TO CONTROL LEARNING BEHAVIORS.pdf
paper_sha256: 318891d59ba0065f57eb0e735735700e5eb528f53bf08b81c773437385586c64
processed_at: '2026-08-18T06:45:26-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Dreamer: 用人话讲

好, Andrej, 我换种方式, 少用行话, 多打比方, 但核心数学还是保留.

---

## 1. 这论文到底在干啥

一句话: **让 AI 先在脑子里"做梦"模拟未来, 然后从梦里学会怎么行动.**

具体场景: 你给 AI 看屏幕画面 (64×64 像素的图片), 它要学会控制一个机器人走路 / 摆杆 / 跑步. 传统方法 (model-free, 比如 DQN, PPO, SAC) 是让 AI 在真实环境里疯狂试错, 几千万次才学会. Dreamer 只试 500 万次就学会了, 而且学得更好.

类比: 你学开车. Model-free 是你在停车场瞎开 1000 小时, 撞了无数次才学会. Dreamer 是你开 50 小时, 然后在脑子里"模拟"开车 950 小时, 模拟的时候还能精确算出"如果我方向盘往左打 5 度, 0.5 秒后车会到哪, 那个位置好不好", 然后直接优化方向盘怎么打.

参考: Danijar Hafner 的项目主页 https://danijar.com/dreamer/

---

## 2. 三步走: 学, 梦, 动

Dreamer 把整个 agent 拆成三个循环执行的步骤:

### Step 1: 学 (Learn dynamics from experience)

AI 先在真实环境里随便走走, 收集一堆 (图片, 动作, 奖励) 的序列. 然后训练一个 **world model**——一个神经网络, 给它当前画面和动作, 它能预测下一步画面和奖励.

但有个问题: 直接预测 64×64×3 的图片太贵了, 一张图 12288 个数, rollout 15 步就是 15 万个数, GPU 内存爆炸.

解决办法: 把图片压成 30 个数字的 latent state $s_t$. 这 30 个数"浓缩"了当前世界的所有关键信息 (杆子角度, 速度, 机器人姿态等). 然后在 30 维空间里 rollout, 几乎免费.

```
真实图片 o_t (64×64×3)
      ↓ CNN encoder
latent state s_t (30 维)
      ↓ transition model
latent state s_{t+1} (30 维)
      ↓ decoder (只在训练时用)
重建图片 ô_t (64×64×3)
```

训练 loss 大概是: "重建的图片像不像原图" + "预测的 reward 对不对" + "prior 和 posterior 别差太多" (KL term, 防止 encoder 偷懒把所有信息都塞进 latent).

这个 world model 的结构叫 **RSSM** (Recurrent State Space Model), 来自作者前作 PlaNet.

参考: PlaNet 论文 https://arxiv.org/abs/1811.04551

### Step 2: 梦 (Learn behavior in imagination)

World model 训好之后, 把它当成一个"虚拟游戏机". 这个虚拟游戏机的规则是神经网络定义的, 所以 fully differentiable.

从 replay buffer 里取一个真实状态 $s_t$ 作为起点, 然后 rollout 15 步:

$$
s_t \xrightarrow{a_t} s_{t+1} \xrightarrow{a_{t+1}} s_{t+2} \xrightarrow{} \cdots \xrightarrow{} s_{t+15}
$$

每一步:
- Action model (policy) 预测动作: $a_\tau \sim q_\phi(a_\tau \mid s_\tau)$
- Transition model 预测下一个 state: $s_{\tau+1} \sim q_\theta(s_{\tau+1} \mid s_\tau, a_\tau)$
- Reward model 预测奖励: $r_\tau \sim q_\theta(r_\tau \mid s_\tau)$
- Value model 预测"从这个 state 开始, 未来能拿多少分": $v_\psi(s_\tau)$

关键 trick: 所有 sampling 都用 reparameterization, 意思是把随机性抽出来当外部 noise $\epsilon$, 让整个 chain 可以 backprop.

$$
a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

变量解释:
- $\mu_\phi, \sigma_\phi$: 神经网络输出的 mean 和 std (参数是 $\phi$)
- $\epsilon$: 从标准正态分布采的 noise, 跟参数无关
- $\tanh$: 把无界的 Gaussian 压到有界 action space

这样 $\frac{\partial a_\tau}{\partial \phi}$ 可以解析计算, 梯度能穿过 sampling.

### Step 3: 动 (Act in environment)

拿训好的 action model 去真实环境里跑, 收集新数据, 扔回 replay buffer, 回到 Step 1.

每 100 个 gradient step 收集 1 个 episode. Action 加一点 Gaussian noise $\mathcal{N}(0, 0.3)$ 当 exploration.

---

## 3. 为什么 Dreamer 比传统方法强

### 3.1 数据效率

传统 model-free: 每次环境 interaction 产生 1 个 transition, 只能用来 update 1 次.

Dreamer: 每次环境 interaction 之后, 在 latent space rollout 50 batch × 15 horizon = 750 个 imagined transitions, 全部带 analytic gradient. 一个真实 step "衍生出" 750 个训练信号.

这就是为什么 5M steps 能打 D4PG 的 100M steps——**数据利用率差 20 倍**.

### 3.2 Gradient 质量

Model-free policy gradient (PPO / A3C) 用的是 REINFORCE / score function:

$$
\nabla_\phi J \approx \mathbb{E}\left[\nabla_\phi \log \pi(a|s) \cdot R\right]
$$

这是"投票法": 遇到好结果, 把产生这个结果的 action 概率调高. Variance 巨大, 因为同一个 policy 可能产生好结果也可能产生坏结果, 你不知道是 action 选得好还是运气好.

Dreamer 用 analytic gradient:

$$
\nabla_\phi J = \frac{\partial V}{\partial s_{t+15}} \cdot \frac{\partial s_{t+15}}{\partial s_{t+14}} \cdots \frac{\partial s_{t+1}}{\partial a_t} \cdot \frac{\partial a_t}{\partial \phi}
$$

这是"精确计算法": 知道每一步 action 对最终 value 的影响有多大, 直接算出来调多少. Variance 小很多, 不需要 PPO 的 clip / trust region 那些 engineering trick.

参考: 
- PPO: https://arxiv.org/abs/1707.06347
- REINFORCE: https://link.springer.com/article/10.1007/BF00115009
- SVG (类似思路的早期工作): https://proceedings.neurips.cc/paper/2015/hash/8b6dd7db7b5ded432b3a4d4b8f3e1b9a-Abstract.html

### 3.3 Long-horizon 能力

想象一下 acrobot swingup 这个任务: 一个双摆, 你要把它从下垂状态甩到直立状态. 需要来回 swing 好几次, credit assignment 跨越几十步.

- PlaNet (online planning, CEM): 只看未来 15 步的 reward, 15 步不够 swing 一次, 所以 fail (得分 3.2)
- Naive action model (只优化 horizon 内 reward): 同样问题, short-sighted
- Dreamer: 15 步之后用 value model $v_\psi$ 估计"剩下的未来能拿多少分", 相当于把 infinite horizon 折叠到一个 scalar. 所以 15 步 horizon 够用 (得分 365)

---

## 4. V_λ: Dreamer 的核心计算

这是最精巧的部分, 我多花点篇幅.

### 4.1 问题设定

给定 imagined trajectory $\{s_\tau, a_\tau, r_\tau\}_{\tau=t}^{t+H}$, 我们要给每个 $s_\tau$ 估计一个 value $V(s_\tau)$, 用来训 action model 和 value model.

### 4.2 三种估计方式

**方式 A: Monte Carlo return (只看 horizon 内的 reward)**

$$
V_R(s_\tau) = \sum_{n=\tau}^{t+H} r_n
$$

直接把未来 15 步的 reward 加起来. 问题: 15 步之后的 reward 完全忽略, 短视.

**方式 B: n-step return (用 value model 补尾)**

$$
V_N^k(s_\tau) = \sum_{n=\tau}^{h-1} \gamma^{n-\tau} r_n + \gamma^{h-\tau} v_\psi(s_h)
$$

变量:
- $k$: 前 k 步用真实 reward, 之后用 value model 估计
- $h = \min(\tau + k, t+H)$: 截断点
- $\gamma = 0.99$: discount factor
- $v_\psi(s_h)$: value model 的预测

$k$ 大 → bias 小 variance 大 (更像 MC); $k$ 小 → bias 大 variance 小 (更像 TD).

**方式 C: λ-return (Dreamer 实际用的)**

$$
V_\lambda(s_\tau) = (1-\lambda) \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau) + \lambda^{H-1} V_N^H(s_\tau)
$$

变量:
- $\lambda = 0.95$: mixing 系数
- 对所有 n-step return 做指数加权平均, $\lambda$ 接近 1 偏向 long-horizon (low bias, high variance), 接近 0 偏向 1-step TD (high bias, low variance)
- 最后一项 $\lambda^{H-1}$ 把 residual 概率全给 full-horizon return, 保证是 proper convex combination

直觉: 不要押宝在单一 k 上, 各种 k 都试, 加权平均. 这跟 Sutton 1988 的 TD(λ) 一模一样, 只是在 imagination environment 里跑.

参考: Sutton & Barto RL textbook http://incompleteideas.net/book/the-book-2nd.html

### 4.3 实验验证 (Figure 4)

作者做了 ablation:
- **Dreamer** (with value model + V_λ): horizon 从 5 到 25 都稳定, 性能几乎不变
- **No value** (只用 V_R): horizon < 20 就崩, 因为短视
- **PlaNet** (online planning): horizon 越长越慢, 但性能也受 horizon 影响

结论: value model 让 Dreamer **robust to horizon**, 即使只 rollout 5 步也能解 long-horizon 任务.

---

## 5. World Model: RSSM 内部

### 5.1 为什么不直接用 VAE / RNN

普通 VAE: 每帧独立 encode, 没有时间依赖.  
普通 RNN: 有时间依赖, 但没有 stochastic latent, 无法建模环境不确定性.  
RSSM: 两者结合, stochastic + recurrent.

### 5.2 RSSM 结构

每一步有两个 distribution:

```
Prior (不看当前图):     q(s_t | s_{t-1}, a_{t-1})        ← imagination 用这个
Posterior (看当前图):    p(s_t | s_{t-1}, a_{t-1}, o_t)    ← 训练时用这个
```

训练时强制 prior 靠近 posterior (KL term), 这样 imagination 时只靠 prior 就够准.

这模仿了 **Kalman filter** 的 predict-update 循环:
- Predict: 用 transition model 预测下一步 state (prior)
- Update: 看到 observation 后修正 (posterior)

区别: Kalman filter 是线性的, RSSM 是神经网络, 非线性.

### 5.3 训练 loss (ELBO)

$$
\mathcal{L} = \sum_t \left[\underbrace{\ln q(o_t | s_t)}_{\text{重建图片}} + \underbrace{\ln q(r_t | s_t)}_{\text{预测 reward}} - \underbrace{\beta \, \text{KL}(p \| q)}_{\text{prior 靠近 posterior}}\right]
$$

变量:
- $\beta = 1.0$: KL weight
- KL clip 在 3 free nats (3 nats ≈ 4.1 bits), 防止 posterior 完全 collapse 到 prior (信息全丢) 或完全 ignore prior (imagination 不准)

---

## 6. 实验数字

### 6.1 主结果

20 个 DMC tasks 平均分:

| Agent | 输入 | Steps | 平均分 |
|---|---|---|---|
| A3C | proprio (state) | 100M | 243 |
| D4PG | pixels | 100M | 786 |
| PlaNet | pixels | 5M | 333 |
| **Dreamer** | **pixels** | **5M** | **823** |

Dreamer 用 5% 的数据, 打败了 model-free SOTA. 训练时间 15 小时 (单 V100), D4PG 要 24 小时.

### 6.2 几个亮点任务

- **Acrobot Swingup**: D4PG 91, Dreamer 365. Long-horizon credit assignment.
- **Quadruped Run**: D4PG 解不了 (空), Dreamer 888. 3D 接触动力学.
- **Hopper Stand**: D4PG 930, Dreamer 924. 几乎持平, 但 Dreamer 用 5% 数据.

### 6.3 Representation learning 对比 (Figure 8)

三种 representation 学习方式:
1. **Reconstruction** (pixel decoder): 最好, 大部分任务都解
2. **Contrastive** (InfoNCE, 不重建 pixel): 解一半任务
3. **Reward only** (只预测 reward): 大部分 fail, 因为 reward 太 sparse

结论: pixel reconstruction 提供了最强的 learning signal, 即使看起来"浪费". 后续 DreamerV2 用 discrete latent, DreamerV3 用 symlog predictions 来改进.

---

## 7. 跟其他方法的关系

| 方法 | 怎么用 model | Gradient 类型 | Horizon |
|---|---|---|---|
| Dyna (Sutton 1991) | 生成 synthetic transition 训 Q | 无 (1-step bootstrap) | 1 step |
| PlaNet (Hafner 2018) | Online planning (CEM) | derivative-free | fixed H |
| MVE / STEVE | Multi-step Q-learning | analytic (但只到 Q) | fixed k |
| DDPG / SAC | 不用 model | analytic (1-step Q) | 1 step |
| SVG (Heess 2015) | 1-step model gradient | analytic | 1 step |
| **Dreamer** | **Multi-step imagination** | **analytic through dynamics** | **H + λ bootstrap** |

Dreamer 的位置: 把 SAC 的 reparameterized actor-critic 从 1-step 推广到 H-step, 加上 V_λ 的 bias-variance 平衡.

参考:
- Dyna: https://dl.acm.org/doi/10.1145/122344.122377
- MVE: https://arxiv.org/abs/1803.00101
- DDPG: https://arxiv.org/abs/1509.02971
- SAC: https://arxiv.org/abs/1801.01290

---

## 8. 后续工作

### DreamerV2 (2020)
- Latent 从 30-dim Gaussian 换成 categorical (32×32 discrete)
- Atari 55 个游戏达到 human-level
- https://arxiv.org/abs/2010.02193

### DreamerV3 (2023)
- 固定 hyperparameter 跨 150+ tasks (DMC, Atari, Crafter, Minecraft, ProcGen)
- Symlog predictions + free bits + KL auto-encoding
- Minecraft 第一个收集到 diamond 的 model-based agent
- https://arxiv.org/abs/2301.04104

### DayDreamer (2022)
- Dreamer 跑在真实机器人上, 1 小时学会 walk
- https://arxiv.org/abs/2206.14176

---

## 9. 一句话 Intuition

**Dreamer = 学一个世界模型 + 在模型里做梦 + 从梦里用 analytic gradient 学策略 + 用 value 估计把有限梦境延伸到无限未来.**

它把 reinforcement learning 变成了 supervised learning on imagined data. 真实环境只负责"喂"数据给 world model, policy improvement 是 world model 的 byproduct. 这就是为什么 data efficiency 能差 20 倍.

如果你熟悉 differentiable programming, Dreamer 本质上是把整个 imagination trajectory 当成一个 differentiable computation graph, 然后 end-to-end backprop. World model 是 frozen 的 (behavior learning 阶段), 所以梯度只更新 actor 和 critic, 但梯度**穿过** world model 来传递 credit.

这就是 Dreamer 的核心 trick, 剩下的都是 engineering details.

---

## 10. 参考资源汇总

- Dreamer V1 paper: https://arxiv.org/abs/1912.01603
- Dreamer V1 code (TF2): https://github.com/danijar/dreamer
- Dreamer V2: https://arxiv.org/abs/2010.02193
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Dreamer V3 code (JAX): https://github.com/danijar/dreamerv3
- PlaNet (前作): https://arxiv.org/abs/1811.04551
- Hafner 个人主页: https://danijar.com/
- Sutton & Barto RL 教材: http://incompleteideas.net/book/the-book-2nd.html
- DMC Control Suite: https://arxiv.org/abs/1801.00690
- TD(λ) 原始论文: https://link.springer.com/article/10.1007/BF00115009
- SAC: https://arxiv.org/abs/1801.01290
- VAE / Reparameterization: https://arxiv.org/abs/1312.6114
- InfoNCE / CPC: https://arxiv.org/abs/1807.03748
- VIB: https://arxiv.org/abs/1612.00410
- DayDreamer (real robot): https://arxiv.org/abs/2206.14176
- MuZero (另一种 latent dynamics + MCTS): https://arxiv.org/abs/1911.08265

希望这次足够"人话"了, Andrej. 如果还有哪块想再展开 (比如 RSSM 的 GRU 内部, V_λ 的 backprop chain rule, 或者 DreamerV3 的改进细节), 随时说.

---

# Dreamer: Learning Behaviors by Latent Imagination

Andrej, 这篇 paper 我非常熟悉, 仔细给你拆解一下, 重点放在能 build intuition 的技术细节上.

---

## 1. High-level Intuition

Dreamer 的核心 insight 可以这样表述: 如果我们有一个能在 compact latent space 中前向预测的 world model, 那么 policy optimization 就可以完全在 "imagination" 中进行——而且因为整个 imagination trajectory 都由 differentiable neural networks 构成, 我们可以直接把 value 的 analytic gradient backprop through dynamics 来更新 policy, 这比 model-free 的 policy gradient (e.g. PPO / A3C) variance 低很多, 也比 online planning (e.g. PlaNet / CEM) 计算高效很多.

这跟 Sutton 1991 年提出的 Dyna architecture 思想一脉相承, 但 Dreamer 的关键升级在于:
- Dyna 用 model 生成 synthetic transitions 来训 Q-function, 仍然是 1-step bootstrap
- Dreamer 直接把 multi-step return 的 analytic gradient 通过 transition model 反传, 类似 Differentiable MPC / Stochastic Value Gradient (SVG, Heess et al. 2015) 的 multi-step 推广

参考链接:
- Sutton 的 Dyna 论文: https://dl.acm.org/doi/10.1145/122344.122377
- SVG (Heess et al. 2015): https://proceedings.neurips.cc/paper/2015/hash/8b6dd7db7b5ded432b3a4d4b8f3e1b9a-Abstract.html
- Dreamer 项目主页: https://danijar.com/dreamer/

---

## 2. Latent Dynamics Model: RSSM

### 2.1 三个核心组件

Dreamer 沿用了作者前作 PlaNet 的 RSSM (Recurrent State Space Model, Hafner et al. 2018). 关键是有两条路径产生 latent state:

$$
\underbrace{p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)}_{\text{representation (posterior)}} \quad
\underbrace{q_\theta(s_t \mid s_{t-1}, a_{t-1})}_{\text{transition (prior)}}
$$

变量含义:
- $s_t \in \mathbb{R}^{30}$: latent state, 论文里是 30-dim diagonal Gaussian
- $a_{t-1}$: action (continuous 或 discrete)
- $o_t$: 64×64×3 image observation
- $p$ vs $q$ 的约定: $p$ 表示 access 到真实 observation 的分布 (posterior), $q$ 表示仅靠 latent 自身预测的近似 (prior). 这种 notation 借鉴 VAE 的 encoder/decoder 思想

直觉上, 这就是一个**非线性 Kalman filter**: prior 用 RNN 预测下一步的 state distribution, posterior 在看到真实 observation 之后用 CNN encoder 修正. 训练时强制 posterior 和 prior 靠近 (KL term), 这样在 imagination 阶段只靠 prior 就能 rollout, 不需要 decoder 生成图像.

### 2.2 训练目标: Variational Lower Bound

World model 的 loss 是 ELBO (具体是 Variational Information Bottleneck, VIB, Tishby et al. 2000; Alemi et al. 2016):

$$
\mathcal{L}_{\text{REC}} \doteq \mathbb{E}_p\left(\sum_t \left[ \mathcal{L}_O^t + \mathcal{L}_R^t + \mathcal{L}_D^t \right]\right)
$$

各项含义:
- $\mathcal{L}_O^t \doteq \ln q_\theta(o_t \mid s_t)$: observation reconstruction log-likelihood (decoder 用 transposed CNN)
- $\mathcal{L}_R^t \doteq \ln q_\theta(r_t \mid s_t)$: reward prediction log-likelihood (dense MLP)
- $\mathcal{L}_D^t \doteq -\beta \, \text{KL}\big(p(s_t \mid s_{t-1}, a_{t-1}, o_t) \,\|\, q(s_t \mid s_{t-1}, a_{t-1})\big)$: KL regularizer, $\beta$ 默认 1.0, clip 在 3 free nats (来自 PlaNet 的技巧, 防止 posterior 过度 collapse 到 prior)

Appendix B 推导了这其实是最大化 mutual information $\mathbb{I}(s_{1:T}; (o_{1:T}, r_{1:T}) \mid a_{1:T}) - \beta \, \mathbb{I}(s_{1:T}; i_{1:T} \mid a_{1:T})$, 其中 $i_t$ 是 dataset index. 第二项限制 representation 不要从一个 image 把所有信息都 "抄" 进 state, 强制让 model 用 history 来预测, 鼓励 long-term dependency.

VIB 原文: https://arxiv.org/abs/1612.00410  
PlaNet (RSSM 原始论文): https://arxiv.org/abs/1811.04551

### 2.3 网络结构细节

- Encoder: CNN (来自 Ha & Schmidhuber 2018), 输入 64×64×3, 输出 flatten 后 concat 到 recurrent state
- RSSM core: GRU-like recurrent cell, 输出 prior 和 posterior 的 Gaussian 参数
- Decoder: transposed CNN, 重建 64×64×3 image
- Reward head / Discount head: 3-layer dense MLP, hidden 300, ELU activation (Clevert et al. 2015)
- Latent dim: 30, diagonal Gaussian

ELU 论文: https://arxiv.org/abs/1511.07289

---

## 3. Imagination Environment

这里有一个非常重要的概念转换. World model 训练好之后 (或者训练中 frozen 一段时间), 我们把它**当成一个新的 MDP**, 这个 MDP 的 transition / reward / done 都由神经网络给出, 而且 fully observable (因为 $s_t$ 是 Markovian).

Imagination rollout:
$$
s_\tau \sim q_\theta(s_\tau \mid s_{\tau-1}, a_{\tau-1}), \quad r_\tau \sim q_\theta(r_\tau \mid s_\tau), \quad a_\tau \sim q_\phi(a_\tau \mid s_\tau)
$$

时间下标: 真实环境用 $t$, imagination 内部用 $\tau$. 起点是从 replay buffer 里取的真实 batch 的 posterior state $s_t$, 然后 rollout $H=15$ 步 (continuous control) 或 $H=10$ (Atari).

整个 imagination trajectory 是一个 differentiable computational graph:

$$
a_\tau \xrightarrow{q_\phi} s_{\tau+1} \xrightarrow{q_\theta} r_{\tau+1}, v_\psi(s_{\tau+1})
$$

所有 stochastic node 都用 reparameterization trick 转成 deterministic function of parameter + 外部 noise $\epsilon$, 这样 backprop 可以穿过 sampling 操作.

---

## 4. Value Estimation: V_λ

这是 Dreamer 设计上最精巧的部分. 给定 imagination trajectory $\{s_\tau, a_\tau, r_\tau\}_{\tau=t}^{t+H}$, 我们要估计每个 $s_\tau$ 的 value. 论文给出三种估计:

### 4.1 Monte Carlo return (no bootstrap)

$$
V_R(s_\tau) \doteq \mathbb{E}_{q_\theta, q_\phi}\left(\sum_{n=\tau}^{t+H} r_n\right)
$$

直接 sum horizon 内的 rewards, 不管 horizon 之后的事. 这就是 "naive action model" ablation, 在 long-horizon 任务上会短视.

### 4.2 n-step return

$$
V_N^k(s_\tau) \doteq \mathbb{E}_{q_\theta, q_\phi}\left(\sum_{n=\tau}^{h-1} \gamma^{n-\tau} r_n + \gamma^{h-\tau} v_\psi(s_h)\right), \quad h = \min(\tau+k, t+H)
$$

变量:
- $k$: bootstrap 之前展开的步数
- $h$: 实际截断点, 不能超过 imagination horizon $t+H$
- $\gamma = 0.99$: discount factor
- $v_\psi(s_h)$: learned value model 的预测

直觉: $k$ 大 → bias 小 variance 大 (更像 MC), $k$ 小 → bias 大 variance 小 (更像 TD).

### 4.3 λ-return (TD(λ) 的 imagination 版本)

$$
V_\lambda(s_\tau) \doteq (1-\lambda) \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau) + \lambda^{H-1} V_N^H(s_\tau)
$$

变量:
- $\lambda = 0.95$: weighting 系数, 接近 1 偏向 multi-step (low bias, high variance), 接近 0 偏向 1-step TD (high bias, low variance)
- 最后一项 $\lambda^{H-1} V_N^H$ 把 residual 概率全部分给 full-horizon return, 保证 $V_\lambda$ 是个 proper convex combination

这跟 Sutton 1988 的 TD(λ) 是同一个 idea, 只是在 imagination environment 里跑. $\lambda = 0.95$ 意味着 Dreamer 严重偏向 long-horizon bootstrap, 但保留一点点 TD 的 variance reduction.

Sutton & Barto RL textbook (免费): http://incompleteideas.net/book/the-book-2nd.html  
TD(λ) 原始论文: https://link.springer.com/article/10.1007/BF00115009

### 4.4 为什么 V_λ 让 Dreamer 不受 horizon 限制

Figure 4 的 ablation 很关键: 不用 value model (只用 $V_R$) 时, horizon 必须 ≥ 20 才能解 acrobot / hopper 这种 long-horizon 任务; 用 $V_\lambda$ + value model, 即使 $H=5$ 也能解. 原因: value model $v_\psi$ 在 horizon 之外继续 bootstrap, 相当于把 infinite horizon 折叠到一个 scalar prediction. Action model 不需要真的 rollout 1000 步, 只需要 rollout 15 步 + 相信 $v_\psi$ 的 estimate.

---

## 5. Actor-Critic Objective

### 5.1 Action model (policy)

$$
\max_\phi \, \mathbb{E}_{q_\theta, q_\phi}\left(\sum_{\tau=t}^{t+H} V_\lambda(s_\tau)\right)
$$

注意: 这里**不是**直接最大化 reward, 而是最大化 value estimate $V_\lambda(s_\tau)$, 而且 $V_\lambda$ 本身依赖 $\phi$ (因为 trajectory 是 $\phi$ 生成的). 梯度通过 dynamics 反传:

$$
\nabla_\phi V_\lambda(s_\tau) = \frac{\partial V_\lambda}{\partial s_\tau} \cdot \frac{\partial s_\tau}{\partial a_{\tau-1}} \cdot \frac{\partial a_{\tau-1}}{\partial \phi}
$$

其中 $\frac{\partial s_\tau}{\partial a_{\tau-1}}$ 来自 transition model $q_\theta$, $\frac{\partial a_{\tau-1}}{\partial \phi}$ 来自 policy. 这是 Dreamer 的核心计算 trick—— analytic gradient through dynamics.

### 5.2 Action model 的 reparameterization

$$
a_\tau = \tanh\big(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \odot \epsilon\big), \quad \epsilon \sim \mathcal{N}(0, I)
$$

变量:
- $\mu_\phi(s_\tau)$: mean head (输出 scaled by factor 5, 让 action 可以 saturate)
- $\sigma_\phi(s_\tau)$: std head (softplus)
- $\tanh$: 把 unbounded Gaussian 映射到 bounded action space (SAC 的 tanh-Gaussian trick, Haarnoja et al. 2018)

离散 action (Atari) 用 straight-through gradient (Bengio et al. 2013), 因为 categorical sampling 不可微.

SAC 论文: https://arxiv.org/abs/1801.01290  
Straight-through estimator: https://arxiv.org/abs/1308.3432

### 5.3 Value model (critic)

$$
\min_\psi \, \mathbb{E}_{q_\theta, q_\phi}\left(\sum_{\tau=t}^{t+H} \frac{1}{2} \|v_\psi(s_\tau) - V_\lambda(s_\tau)\|^2\right)
$$

Stop gradient on $V_\lambda$ target (Sutton & Barto 2018 标准 trick), 防止 critic 跟着 actor 一起漂移. 注意没有用 target network (作者在 hyperparameter section 提到试过不需要), 这跟 DQN / DDPG 不同.

### 5.4 跟其他 actor-critic 的关键区别

| Method | Gradient path | Horizon |
|---|---|---|
| A3C / PPO | REINFORCE (score function), variance 高 | 1-step bootstrap |
| DDPG / SAC | reparameterization, gradient 只到 immediate Q | 1-step bootstrap |
| MVE / STEVE | multi-step Q-learning with model | fixed k |
| SVG (Heess 2015) | 1-step model gradient, 用 on-policy baseline | 1-step |
| **Dreamer** | **multi-step model gradient through value** | **H steps + λ bootstrap** |

Dreamer 的 "magic" 在于: 它把 SAC 的 reparameterized actor-critic 思路从 1-step 推广到 H-step, 而且整个 trajectory 都是免费的 (在 latent space 里 rollout, 不用 environment step).

MVE 论文: https://arxiv.org/abs/1803.00101  
STEVE: https://arxiv.org/abs/1802.09477  
DDPG: https://arxiv.org/abs/1509.02971

---

## 6. Representation Learning: 三种选择

Dreamer 的 algorithm 跟 representation learning 是 orthogonal 的, 作者对比了三种:

### 6.1 Reconstruction (default)

用 pixel decoder + ELBO, 如上所述. Figure 8 显示这是最强的方式.

### 6.2 Contrastive (InfoNCE)

替换 decoder $q(o_t \mid s_t)$ 为 encoder $q(s_t \mid o_t)$, 用 InfoNCE bound (Oord et al. 2018):

$$
\mathcal{L}_S^t \doteq \ln q(s_t \mid o_t) - \ln\left(\sum_{o'} q(s_t \mid o')\right)
$$

直觉: 第一项让 state 从 image 可预测, 第二项 (negative samples over batch) 防止 state collapse 到常数. Appendix B 给出这是 mutual information 的 lower bound via noise contrastive estimation.

Figure 8 显示 contrastive 能解大约一半任务, 但不如 reconstruction 稳.

CPC / InfoNCE: https://arxiv.org/abs/1807.03748  
NCE 原始: https://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann10AISTATS.pdf

### 6.3 Reward prediction only

只用 $\mathcal{L}_R^t$, 不学任何 visual representation. 在大多数任务上 fail, 因为 reward 太 sparse, state 学不到足够信息.

这个 ablation 对后续工作很有启发: DreamerV2 (discrete latent) / DreamerV3 (symlog predictions, dropout) 主要改进就在 representation learning 上.

DreamerV2: https://arxiv.org/abs/2010.02193  
DreamerV3: https://arxiv.org/abs/2301.04104

---

## 7. 实验结果详解

### 7.1 主结果 (Table in Appendix G)

20 个 DMC tasks, average score:

| Agent | Input | Steps | Avg Score |
|---|---|---|---|
| A3C | proprio | 1e8 | 243.70 |
| D4PG | pixels | 1e8 | 786.32 |
| PlaNet | pixels | 5e6 | 332.97 |
| **Dreamer** | **pixels** | **5e6** | **823.39** |

要点:
- Dreamer 用 **20 倍少** 的 environment steps 超过 D4PG
- PlaNet 用同样的 world model 但靠 online planning (CEM), 只到 332, 说明 analytic gradient >> derivative-free planning
- 单 V100 GPU, 5M steps 训练 ~15 小时 (3 小时 / 1M steps); D4PG 训练 24 小时; PlaNet 55 小时

### 7.2 Long-horizon 任务的具体差异

Figure 7 显示几个典型任务的曲线:
- **Acrobot Swingup**: A3C 41.9, D4PG 91.7, PlaNet 3.2, Dreamer **365.3**. 差距巨大, 因为 acrobot 需要多次 swing, credit assignment 跨越几十步
- **Hopper Stand**: A3C 27.9, D4PG 929.9, PlaNet 5.96, Dreamer **923.7**. Dreamer 接近 D4PG 但只用 5% 的数据
- **Walker Run**: D4PG 567.2, Dreamer **824.7**. Dreamer 反而 final performance 更高
- **Quadruped Run**: D4PG 没解 (空), PlaNet 280, Dreamer **888**. 3D 接触动力学, 之前 model-based 完全 fail

### 7.3 Atari / DMLab (Appendix C)

用 straight-through gradient 处理 discrete action, predict discount factor 处理 early termination. Figure 9 显示能解一些 Atari (如 Boxing, Breakout), 但 overall 不及 SimPLe / Rainbow. 这刺激了 DreamerV2 用 discrete latent states (categorical VQ-VAE 风格) 来处理 high visual complexity.

---

## 8. 算法流程 (Algorithm 1 详解)

```
Initialize dataset D with S=5 random seed episodes
Initialize θ (world model), φ (action model), ψ (value model)

while not converged:
    for c = 1..100 (collect interval):
        # ---- Dynamics learning ----
        Draw B=50 sequences of length L=50 from D
        Compute posterior states s_t ~ p_θ(s_t | s_{t-1}, a_{t-1}, o_t)
        Update θ via ELBO (reconstruction + KL)

        # ---- Behavior learning ----
        Imagine trajectories {s_τ, a_τ}_{τ=t}^{t+H}, H=15
        Predict rewards r_τ, values v_ψ(s_τ)
        Compute V_λ targets (γ=0.99, λ=0.95)
        Update φ: φ ← φ + α ∇_φ Σ V_λ(s_τ)        # α = 8e-5
        Update ψ: ψ ← ψ - α ∇_ψ Σ ½||v_ψ - V_λ||²  # α = 8e-5

    # ---- Environment interaction ----
    Collect 1 episode:
        s_t ~ p_θ(s_t | s_{t-1}, a_{t-1}, o_t)  # posterior from history
        a_t ~ q_φ(a_t | s_t) + Normal(0, 0.3) noise  # exploration
        Add (o_t, a_t, r_t) to D
```

关键 hyperparameter 总结:
- $S=5$ seed episodes, $B=50$ batch, $L=50$ sequence length, $H=15$ imagination horizon
- Learning rates: world model $6\times10^{-4}$, actor/critic $8\times10^{-5}$
- $\beta=1.0$ (KL weight, clipped at 3 nats), $\gamma=0.99$, $\lambda=0.95$
- Action repeat $R=2$ (统一, 不像 PlaNet 每个 task 调)
- Gradient norm clip at 100

---

## 9. 一些值得思考的设计选择

### 9.1 为什么不在 image space imagination?

Image decoder 太贵, 64×64×3 的 trajectory rollout 内存爆炸. Latent 30-dim Gaussian 让 batch=50 × horizon=15 的 imagination 在 GPU 上几乎免费. 这就是 "compact state space" 的物理含义.

### 9.2 为什么 value model 没有 target network?

通常 critic 需要 target network (e.g. DDPG, DQN) 来稳定训练. Dreamer 不需要, 可能因为:
- World model θ 在 behavior learning 阶段是 frozen 的 (虽然实际实现里两个一起训), trajectory 的 dynamics 不漂移
- $V_\lambda$ target 本身是 convex combination, 平滑性强
- Imagination rollout 是从 replay buffer 的真实 state 出发, 起点多样

### 9.3 探索策略

很朴素: action model 的 mode + Gaussian noise $\mathcal{N}(0, 0.3)$. 没有 intrinsic motivation / count-based exploration. 作者认为 world model 的不确定性 (posterior vs prior 的 KL) 隐式提供了 exploration 信号. 后续 DreamerV3 加了 discrete latent 和 KL 自动调温.

### 9.4 Discount prediction

任务有 early termination 时 (Atari), world model 还预测 $\gamma_t$ (binary classifier, soft labels 0 / γ). 然后 Equation 7/8 里的项乘以累积 discount product, 这样 imagination 中提前 "死掉" 的 trajectory 贡献被自动 downweight.

---

## 10. 跟 Model-Free RL 的根本区别

我觉得 Karpathy 你会感兴趣的一点是: Dreamer 把 reinforcement learning 变成了 **supervised learning on imagined data**.

- Model-free (DQN / PPO / SAC): 每个环境 step 只产生 1 个 transition, 用来 update Q 或 policy. 数据利用率 bounded by environment interaction.
- Dreamer: 每个 environment step 之后, 在 latent space rollout 50 batch × 15 horizon = 750 imagined transitions, 全部带 analytic gradient. Environment step 主要用来 *improve world model*, policy improvement 是 "免费" 的 byproduct.

这解释了为什么 data efficiency 能有 ~20× 的提升. 而且 because gradients are analytic (not score-function), variance 大幅降低, 不需要 PPO 那种 clip / trust region 的工程 trick.

---

## 11. 局限与后续工作

### Dreamer V1 的弱点:
1. Pixel reconstruction 在 high visual complexity (Atari 全集, 自然图像) 上 cost 太高
2. Gaussian latent 对 long-tail dynamics 表达力不足
3. 探索完全靠 noise, sparse reward 任务 (Montezuma) 仍然 fail

### Dreamer V2 (2020):
- 把 latent state 从 30-dim Gaussian 换成 categorical distribution (32 classes × 32 categories)
- 借鉴 VQ-VAE 的 discrete representation
- Atari 上 55 个游戏 human-level, 用 200M frames
- https://arxiv.org/abs/2010.02193

### Dreamer V3 (2023):
- **固定 hyperparameter** 跨 150+ tasks (DMC, Atari, Crafter, Minecraft, BSuite, ProcGen)
- Symlog 预测 + free bits + KL auto-encoding 三大 trick
- Minecraft diamond collection 首个 model-based agent
- https://arxiv.org/abs/2301.04104

### DayDreamer (2022):
- Dreamer 跑在真实机器人上 (四足, 机械臂), 训练 1 小时学会 walk
- 显示 world model 能 generalize 到 sim-to-real gap
- https://arxiv.org/abs/2206.14176

---

## 12. 进一步阅读

- Danijar Hafner 个人主页 (所有 Dreamer 系列): https://danijar.com/
- Dreamer V1 official code (TensorFlow 2): https://github.com/danijar/dreamer
- Dreamer V3 official code (JAX): https://github.com/danijar/dreamerv3
- Dreamer PyTorch community port (Antonin Raffin): https://github.com/EvePasta/dreamer-pytorch
- 文章 Reni/DMC Control Suite: https://arxiv.org/abs/1801.00690
- PlaNet 1.0 的 CEM planning 原版: https://arxiv.org/abs/1811.04551
- World Models (Ha & Schmidhuber): https://worldmodels.github.io/
- Differentiable MPC / iLQR 经典 (Amos et al.): https://implicit-layers.github.io/
- MuZero (latent dynamics + MCTS, 不同范式): https://arxiv.org/abs/1911.08265
- TD(λ) 教材式推导: http://incompleteideas.net/book/RLbook2020.pdf (Chapter 12)

---

## 13. Intuition 总结

如果让我给 Dreamer 一个 one-liner: **它把 model-based planning 的 MPC 换成了 analytic gradient backprop through a learned dynamics, 同时用 value bootstrap 把 finite imagination horizon "无限化".**

具体来说:
1. World model 把 high-dim pixels 压到 30-dim Markovian latent (RSSM + VIB)
2. 在 latent 里 rollout H=15 步, 全 differentiable
3. 用 $V_\lambda$ 估计每个 imagined state 的 value, 既 unbiased 又 low variance
4. Action model 通过 ∇V 反传更新, 把 model 当成 computation graph 的一部分
5. Value model regress $V_\lambda$, 标准 critic
6. Environment interaction 只为 grow replay buffer, policy improvement 几乎免费

最终效果: 在 20 个 DMC pixel control tasks 上, 用 5M steps (5% data) 超过 D4PG 用 100M steps 的最终性能, 训练时间 15 小时 vs 24 小时. 这是 model-based RL 第一次在 final performance 上同时打败 model-free, 不只是 data efficiency.

希望这些细节足够 build 你的 intuition. 如果你想深挖某个具体部分 (比如 RSSM 内部结构、V_λ 的具体反传链、或者 V2/V3 的改进), 我可以继续展开.
