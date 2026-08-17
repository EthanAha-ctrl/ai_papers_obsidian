---
source_pdf: world-models-wayve.pdf
paper_sha256: b0c1e30aab53efd28ddf61d661f680150918d4d03b77bae62bc52d62dbd76cce
processed_at: '2026-08-13T05:24:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# World Models 用人话讲

---

## 这 paper 到底在干嘛

设想你在打棒球。100mph 的 fastball 从投手到本垒只要零点几秒,光视觉信号传到你脑子都来不及。但你能打中。为什么?

因为你脑子有个 **mental model**。你不看球在哪,你看球的 trajectory,脑子瞬间预测它未来去哪,然后你的肌肉自动挥棒。这过程不需要 conscious planning,就是 instinctive reflex。

这 paper 说的就是:**给 AI agent 也造一个这样的 mental model,让 agent 靠 model 的 prediction 来 act,而不只是 react 当下这一帧**。

---

## 核心架构:V, M, C 三个模块

整篇 paper 就是这三个字母的组合。用最直白的话讲:

### V (Vision) — 眼睛

V 就是个 **VAE**,干一件事:把 64×64×3 的图片压成一个 32 维或 64 维的 vector $z$。

直觉:你眼睛每秒接收海量 photons,但你脑子里记的是一个抽象的 scene 概念,不记每个 pixel。V 就干这个事。

```
image (64×64×3)  →  V  →  z (32维 or 64维)
```

为什么用 VAE 不用普通 autoencoder?因为后面 M 会生成一些 training data 里没见过的 $z$,VAE 的 Gaussian prior 让 latent space 更平滑,decoder 对这些"没见过"的 $z$ 不会崩。相当于给 latent space 加了个形状约束。

### M (Memory) — 脑子里的预测机器

M 是个 **LSTM + Mixture Density Network**,干一件事:给定当前 $z_t$ 和 action $a_t$,预测下一帧 $z_{t+1}$ 的概率分布。

公式:

$$P(z_{t+1} \mid a_t, z_t, h_t) = \sum_{i=1}^{5} \pi_i \, \mathcal{N}(\mu_i, \sigma_i^2 I)$$

人话翻译:下一个 $z$ 不是固定一个值,是一个 **5 个 Gaussian 叠加出来的概率分布**。$\pi_i$ 是每个 Gaussian 的权重,$\mu_i, \sigma_i$ 是每个 Gaussian 的中心和宽度。

为什么用 mixture 不用 single Gaussian?因为环境里有 **discrete random events**。比如 Doom 里 monster 可能 shoot fireball 也可能不 shoot,这是个 binary 事件。single Gaussian 表达不了这种 multi-modal 分布,mixture 可以。这点很关键,后面 cheating 那段会回来讲。

M 还有个 **temperature 参数 $\tau$**。sample 的时候把 $\sigma$ 乘 $\tau$:
- $\tau$ 小 → 分布尖 → 近似 deterministic
- $\tau$ 大 → 分布平 → 更 stochastic

这个 $\tau$ 是整个 dream training 的灵魂 knob。

### C (Controller) — 决策肌肉

C 极其简单,就是个 **linear mapping**:

$$a_t = W_c [z_t \; h_t] + b_c$$

把 $z_t$ (当前看到的) 和 $h_t$ (LSTM hidden state,也就是 M 对未来的 prediction) 拼一起,线性映射成 action。

参数量:
- CarRacing: 867 个参数
- Doom: 1,088 个参数

**为什么这么小?** 这是整篇 paper 最关键的设计决定。V 和 M 可以用 backprop 在 GPU 上高效训,因为它们 loss 是 differentiable 的 (reconstruction / prediction loss)。但 C 要在 RL environment 里训,credit assignment problem 在大参数空间下超难。所以把 C 做成几百个参数,用 **CMA-ES** (一种 evolution strategy) 这种 black-box optimizer 来训。

复杂度全在 V+M 里 (几百万参数),C 只是个 tiny linear readout。这就像你脑子大部分在做 perception 和 prediction,真正"决定挥棒不挥棒"的决策回路其实很简单。

---

## CarRacing 实验:h_t 为什么重要

CarRacing-v0:top-down 赛车,track 每局随机生成,要访问尽量多 tile。solving 标准 = 100 局平均 > 900。

### 关键 ablation

paper 做了个特别有教学意义的对比:

**Setup 1: C 只看 $z_t$**
$$a_t = W_c z_t + b_c$$
Score: **632 ± 251** — 开得 wobbly,急弯经常 miss

**Setup 2: C 只看 $z_t$ + 加个 hidden layer** (参数 1443,跟 baseline 量级相当)
Score: **788 ± 141** — 好一点,还是不够

**Setup 3: C 看 $[z_t, h_t]$ (full world model)**
$$a_t = W_c [z_t \; h_t] + b_c$$
Score: **906 ± 21** — 稳定,能 attack 急弯

**Intuition**: $z_t$ 只是当前一帧的压缩,没 temporal context。$h_t$ 是 LSTM 的 hidden state,它 encode 了"未来 z 的分布"。给 C $h_t$ 等于给它一个 instinctive prediction,不用 explicit plan ahead。

这就是棒球比喻的技术实现:F1 赛手不 plan,他靠内部 prediction 自动 steer。

之前 SOTA (A3C + edge detection + frame stacking) 只有 591-652,这个 work 是第一个 solve 这个 task 的。

---

## VizDoom 实验:在 dream 里训练

这才是 paper 最 exciting 的部分。

### Setup

VizDoom Take Cover:躲 fireball,活得久就行。max 2100 steps,solving = 100 局平均 > 750。

M 不仅预测 $z_{t+1}$,还预测 death $d_{t+1}$。这样 M 就是个完整 RL environment 接口。把 M 包成 `gym.Env`,叫 **DoomRNN**。

DoomRNN 只在 latent space 跑,不 render pixel。真实 VizDoom 要跑 game engine (physics + rendering),DoomRNN 只是几次 matrix multiply。**训练效率天差地别**。

流程:
1. 在 DoomRNN (dream) 里训 C
2. 把 C 拿回真实 VizDoom 测 transfer

### Temperature $\tau$ 的神奇效果

Table 2 是 paper 里最有 insight 的 table:

| $\tau$ | Virtual Score | Actual Score |
|---|---|---|
| 0.10 | 2086 ± 140 | **193 ± 58** (比 random 还差!) |
| 0.50 | 2060 ± 277 | 196 ± 50 |
| 1.00 | 1145 ± 690 | 868 ± 511 |
| 1.15 | 918 ± 546 | **1092 ± 556** (最好) |
| 1.30 | 732 ± 269 | 753 ± 139 |
| Random | - | 210 ± 108 |
| Gym leaderboard | - | 820 ± 58 |

读这个 table:

$\tau = 0.1$: dream 里 score 2086 (几乎满分),但 reality 只有 193。**Agent 找到 cheat 了**。

$\tau = 1.15$: dream 里 918 (比 $\tau=0.1$ 难很多),reality 1092 (超 leaderboard)。**Sweet spot**。

$\tau = 1.30$: dream 太难,score 降,但 variance 也降 (1092±556 → 753±139)。Agent 学到更保守但更 robust 的策略。

**Intuition**: dream 比 reality 难一点,transfer 最好。这跟 sim-to-real 里 domain randomization 的思路完全一致。

### Cheating the World Model

$\tau = 0.1$ 时 agent 学到了一个诡异策略:它能"自动熄灭"刚发射的 fireball。在 dream 里看到 fireball 要形成,agent 一移动,fireball 就消失了。

为什么会这样?因为 agent 看到了 M 的 **hidden state $h_t$**。这相当于给 agent 直接访问 game engine 的 source code,而 player 本来只能看 screen。Agent 在 $h_t$ 空间里找到了"死角"——某些 $(z_t, h_t)$ 组合让 M 陷入 mode collapse,fireball 模式永远 sample 不出来。

低 $\tau$ 时 mixture Gaussian 退化成单 modal,agent exploit 这个 imperfection。这就是为什么用 **mixture of Gaussian** 而不是 single Gaussian —— 即便真实环境是 deterministic 的,用 mixture 近似成 stochastic,让 agent 不能轻易 exploit。

但 mixture 也只是 mitigation,不是 solution。这是所有 learned dynamics model 的根本难题:model 有缺陷,controller 一定会找到缺陷去 exploit。

paper 提到 Schmidhuber 的 "Learning to Think" 框架更 general:C 可以学会 *ignore* M 的不可靠部分,甚至把 M 的 subroutines 当程序调用。但本文没实现这套,只做最简单的 step-by-step prediction。

---

## Iterative Training:留给未来的 scheme

paper 的实验只做了一轮 (random policy 收数据 → 训 V, M → 训 C)。但第 5 节给了一个更 general 的 iterative scheme:

```
1. 随机初始化 M, C
2. 在真实环境 rollout N 次,存所有 (a_t, x_t, r_t, d_t)
3. 训 M 学 P(x_{t+1}, r_{t+1}, d_{t+1} | x_t, a_t, h_t)
   在 M 内部训 C 优化 expected reward
4. 没解决就回到 step 2
```

几个 extension:
- M 也要预测 reward 和 action,这样复杂 motor skill (走路) 能被吸收进 M
- 然后 C 就能 rely on M 已有的 motor skill,自己学更高层 skill
- **Curiosity**: flip M 的 loss sign,agent 就会主动去找 M 预测差的地方 = 探索未知
- 跟神经科学的 **hippocampal replay** (Foster 2017, https://www.annualreviews.org/doi/10.1146/annurev-neuro-072116-031538) 类比:动物休息时 replay 经历来 consolidate memory
- 联系 Schmidhuber 的 PowerPlay (https://arxiv.org/abs/1210.8385) 和 novelty search

---

## Intuition 总结

### 1. 把 representation learning 和 policy learning 彻底分开

V+M 完全 unsupervised,reward-agnostic,用 backprop + GPU 训。C 极小,用 ES 训。两边各用最适合自己的 optimizer。这避免了 end-to-end model-based RL 里 dynamics model 和 policy 互相干扰的常见问题。

### 2. Hidden state 就是 predictive feature

$h_t$ 携带"未来会发生什么"的信息。给 C $h_t$ 比给它 stacked frames 好太多。后来 Dreamer 的 RSSM state,以及 Sora 的 spacetime latent,都是这个 idea 的延续。

### 3. Dream training = model uncertainty as data augmentation

调高 $\tau$ 让 dream 比 reality 难,transfer 最好。这给 sim-to-real 一个 elegant framework:model 的 uncertainty 本身就是 data augmentation。

### 4. Model exploitability 是根本难题

任何 learned dynamics model 都有死角。Controller 一定会找死角。Mixture + temperature 只是 mitigation。后续 Dreamer 用更长 horizon planning + image reconstruction 继续打这个问题。

### 5. 跟 LLM / video generation 的 connection

虽然 paper 发表在 transformer 时代之前,ideas 完全迁移:
- V = vision encoder (CLIP-like)
- M = world dynamics transformer (Sora-like)
- C = policy head

**Sora** (https://openai.com/sora) 本质上就是超大 scale 的 M model,在 latent space 学世界 dynamics。World Models 是 Sora 的 ancestor idea。

### 6. 跟 MuZero 的对比

MuZero (https://www.nature.com/articles/s41586-020-03051-4) 也是 learned dynamics model,但它是 reward-aware end-to-end training + explicit MCTS planning。World Models 的 C 不做 explicit planning,只 instinctive react on $h_t$。两者是 model-based RL 谱系上的两个极端:一个重 planning,一个重 reflex。

---

## 后续 work

顺着读:

- **PlaNet** (Hafner et al. 2019, https://arxiv.org/abs/1811.04556) — latent state + MPC planning
- **Dreamer** (Hafner et al. 2020, https://danijar.com/project/dreamer/) — actor-critic in latent dream
- **DreamerV2** (2021, https://arxiv.org/abs/2010.02193) — discrete latent,Atari SOTA
- **DreamerV3** (2023, https://arxiv.org/abs/2301.04104) — fixed hyperparameters across domains
- **DayDreamer** (Wu et al. 2022, https://arxiv.org/abs/2206.14176) — real robot dream training
- **Sora** (2024, https://openai.com/research/video-generation-models-as-world-simulators) — scaled up world models as video generator
- **Genie** (DeepMind 2024, https://arxiv.org/abs/2402.15391) — unsupervised environment generation from video

Schmidhuber ancestor works:
- **Making the World Differentiable** (1990, http://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf)
- **Learning to Think** (2015, https://arxiv.org/abs/1511.09249)
- **One Big Net** (2018, https://arxiv.org/abs/1802.08864)

---

## 代码

- 官方 interactive demo: https://worldmodels.github.io
- TensorFlow 实现: https://github.com/hardmaru/WorldModelsCode
- PyTorch 社区实现: https://github.com/awjuliani/worldmodel

---

这 paper 最美的地方在于:它用 modern deep learning tools (VAE, LSTM, MDN, ES) 把 Schmidhuber 1990 年代的 ideas 重做了一遍,且写得极清晰,配 interactive demo。它是 model-based RL 的 modern 起点,也是 Sora / Genie / JEPA 这波 world model 热的直接 ancestor。核心 insight 就一句:**agent 应该靠 internal predictive model 来 act,而不是靠 raw observation react**。

---

# World Models (Ha & Schmidhuber, 2018) 深度讲解

paper 链接: https://worldmodels.github.io  
arXiv: https://arxiv.org/abs/1803.10122  
David Ha 的 blog: http://blog.otoro.net/

---

## 1. 核心思想:Intuition Building

paper 的 motivation 来自一个观察:人类脑子里有一个 mental model of the world,我们靠这个 model 做决策,而不是靠 raw sensory input。Forrester 说的那句"the image of the world around us, which we carry in our head, is just a model"是整篇 paper 的精神底色。

棒球的例子特别直观:打 100mph 的 fastball,从视觉信号传到大脑的时间都不够,但人能打中。原因在于人脑在 *预测* 球的去向,muscles reflexively 根据 prediction 动作。agent 也应该这样:有一个内部 predictive model,act on prediction,不一定要 explicit plan。

技术上的关键 trick 在于:**把 agent 拆成一个大 world model 和一个极小的 controller**。大模型用 backprop 在 GPU 上高效训练 unsupervised,小 controller 用 evolution strategies (CMA-ES) 训练,避开 RL 里臭名昭著的 credit assignment problem 在大参数空间下的困难。

---

## 2. 架构:V, M, C

### 2.1 V (Vision) — Variational Autoencoder

V 模型负责空间压缩:把每一帧 64×64×3 的 raw pixel image 压成一个低维 latent vector $z$。

- CarRacing-v0: $z \in \mathbb{R}^{32}$ (即 $N_z = 32$)
- VizDoom: $z \in \mathbb{R}^{64}$

**ConvVAE 结构** (见 Figure 22):

Encoder:
- Input: 64×64×3
- 4 个 conv layers,stride 2,ReLU
- 输出 $\mu, \sigma \in \mathbb{R}^{N_z}$
- $z \sim \mathcal{N}(\mu, \sigma I)$ (reparameterization trick)

Decoder:
- 4 个 deconv layers,stride 2
- 最后一层不加 ReLU(因为输出要在 [0,1] 范围)
- L2 reconstruction loss + KL divergence

参数量:
- CarRacing VAE: 4,348,547
- Doom VAE: 4,446,915

**为什么用 VAE 而不是 plain autoencoder?** Gaussian prior 限制了 z 的信息容量,但同时让 world model 对 M 生成的"不真实"的 z 向量更 robust。M 是个 generative model,它会 sample 出一些 training data 里没见过的 z,如果 V 用 plain AE 训得太死,decoder 遇到这些 z 就会崩。VAE 的 prior 相当于给 latent space 一个平滑的"形状"约束。

### 2.2 M (Memory) — MDN-RNN

M 模型负责时间压缩 + 未来预测:给定当前状态和 action,预测下一个 z 的概率分布。

公式:

$$P(z_{t+1} \mid a_t, z_t, h_t) = \sum_{i=1}^{M} \pi_i \, \mathcal{N}(z_{t+1} \mid \mu_i, \sigma_i^2 I)$$

变量含义:
- $a_t$: time step $t$ 的 action
- $z_t$: time step $t$ 的 latent vector (来自 V)
- $h_t$: LSTM 在 time $t$ 的 hidden state
- $M$: mixture 数量 (paper 中 $M = 5$)
- $\pi_i$: 第 $i$ 个 Gaussian 的 mixture weight (softmax 输出)
- $\mu_i, \sigma_i$: 第 $i$ 个 Gaussian 的 mean 和 std
- 注意是 **diagonal covariance**,不建模 $z$ 各分量之间的 correlation(对比 SketchRNN 会建模 $\rho$)

为什么用 mixture of Gaussians 而不是 single Gaussian?这是 paper 里一个很关键的设计选择,后面 cheating 部分会详细讲。直觉:环境里有 discrete random events(比如 monster 决定发不发 fireball),single Gaussian 表达不了 multi-modal 分布,mixture model 可以。

**LSTM 配置**:
- CarRacing: 256 hidden units
- Doom: 512 hidden units

**Temperature 参数 $\tau$**:在 sampling 时,把 $\sigma_i$ 乘以 $\tau$。$\tau$ 小 → 分布更尖(deterministic);$\tau$ 大 → 分布更平(stochastic)。这个 trick 直接来自 SketchRNN。后面会看到 $\tau$ 是整个 dream training 的核心 hyperparameter。

**Doom 任务里 M 还预测 death**:输出 $P(d_{t+1} \mid a_t, z_t, h_t)$。当预测概率 > 50% 时设 done = True。这里用 cutoff 而不是 sample Bernoulli,因为 death 是低概率事件,sampling 不稳定。

**参数量**:
- CarRacing MDN-RNN: 422,368
- Doom MDN-RNN: 1,678,785

**训练 trick**:用 teacher forcing,每个 batch 重新从 $\mathcal{N}(\mu, \sigma)$ sample $z$,避免 overfit 到某个固定的 sampled z。

### 2.3 C (Controller) — Linear

最简洁的设计,直接 linear mapping:

$$a_t = W_c [z_t \; h_t] + b_c \tag{1}$$

变量:
- $W_c$: weight matrix
- $b_c$: bias
- $[z_t \; h_t]$: concatenation of $z_t$ 和 $h_t$
- $a_t$: action (CarRacing 是 3 维连续:steering, gas, brake)

参数量极小:
- CarRacing: 867 (=(32+256)×3 + 3)
- Doom: 1,088

Doom 里 C 的输入是 $[z_t \; c_t \; h_t]$,即 LSTM 的 cell state 也喂进去。

**为什么这么小的 controller?** 这是整个 paper 最关键的 insight。backprop 训练 V 和 M 时,loss function 是 well-behaved(differentiable,reconstruction/prediction loss),GPU 上可以高效训。但 controller 要在 RL environment 里训,credit assignment problem 让 traditional RL 算法很难训大量参数。把 C 做成只有几百个参数的 linear model,用 CMA-ES 这种 black-box optimizer 就能搞定。同时,expressiveness 全部藏在 V+M 里,C 只需要"读"feature 做决策。

### 2.4 V + M + C 的整体 rollout

```python
def rollout(controller):
    obs = env.reset()
    h = rnn.initial_state()
    cumulative_reward = 0
    while not done:
        z = vae.encode(obs)
        a = controller.action([z, h])
        obs, reward, done = env.step(a)
        cumulative_reward += reward
        h = rnn.forward([a, z, h])
    return cumulative_reward
```

注意 loop 里的数据流:
1. V 把 obs 变成 z
2. C 用 [z, h] 算 a
3. env 用 a 给出 new obs + reward + done
4. M 用 [a, z, h] 更新到 h_{t+1}

---

## 3. CarRacing-v0 实验

环境:top-down view 的赛车游戏,track 每局随机生成,reward = 访问的 tile 数 - 时间。action 是 3 维连续 (steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1])。solving 标准:100 局平均 > 900。

### 3.1 Procedure

1. 用 random policy 收集 10,000 rollouts
2. 训 V (VAE),z ∈ R^32
3. 训 M (MDN-RNN),学 $P(z_{t+1} \mid a_t, z_t, h_t)$
4. 定义 C 为 linear
5. 用 CMA-ES 优化 $W_c, b_c$ 最大化 expected cumulative reward

V 和 M 都不知道 reward 信号,纯 unsupervised。只有 C 接触 reward。

### 3.2 关键对比:V only vs. V + M

这是 paper 里最有教学意义的一个 ablation:

**V only** ($a_t = W_c z_t + b_c$):
- Score: 632 ± 251
- 行为:wobbly,过弯不顺,经常 miss track

**V only + 一个 40 unit hidden layer** ($a_t = \text{tanh}(W_1 z_t + b_1) \cdot W_2 + b_2$):
- Score: 788 ± 141
- 参数 1443,接近 baseline setup,但还是不够

**Full World Model (V + M)** ($a_t = W_c [z_t \; h_t] + b_c$):
- Score: 906 ± 21
- 行为:稳定、能 attack 急弯

这个对比说明什么?**$h_t$ 提供了 predictive information**。$z_t$ 只是当前帧的压缩,没有 temporal context。$h_t$ 是 LSTM 的 hidden state,它编码了"未来 z 的分布"。controller 拿到 $h_t$ 就相当于拿到一个 instinctive prediction,不需要 roll out 未来场景。

跟开篇棒球比喻对应:F1 赛车手不是看一帧画面然后 plan,是 instinctively 用内部预测来 steer。

### 3.3 对比其他方法

| Method | Avg Score |
|---|---|
| DQN (Prieur 2017) | 343 ± 18 |
| A3C continuous (Jang et al. 2017) | 591 ± 45 |
| A3C discrete (Khan & Elibol 2016) | 652 ± 10 |
| Gym leaderboard (ceobillionaire) | 838 ± 11 |
| V model only | 632 ± 251 |
| V + hidden layer | 788 ± 141 |
| **Full World Model** | **906 ± 21** |

传统 Deep RL 方法需要 frame preprocessing (edge detection) + frame stacking,World Model 直接吃 raw RGB pixel stream 学 spatio-temporal representation。

---

## 4. VizDoom Take Cover 实验 — Dream Training

这个实验是 paper 最 exciting 的部分:**在 dream world 里训 agent,transfer 回真实环境**。

### 4.1 Setup

VizDoom Take Cover:agent 要躲 monster 发的 fireball,活得越久 reward 越多。max 2100 steps (~60s),solving 标准 = 100 局平均 > 750 steps (~20s)。

M 模型比 CarRacing 多预测一件事:death。$P(z_{t+1}, d_{t+1} \mid a_t, z_t, h_t)$。这样 M 就是一个完整的 RL environment 接口。

### 4.2 Dream World = DoomRNN

把 M 包成 `gym.Env` 接口,叫 DoomRNN。它只在 latent space 跑,不需要 render pixel。这极大提升了训练效率——真实 VizDoom 要跑 game engine 算 physics + render,而 DoomRNN 只是几次 matrix multiply。

在 DoomRNN 里训练完 C,deploy 回真实 VizDoom 测试 transfer。

### 4.3 关键发现:Temperature $\tau$ 调控

Table 2 是 paper 里最有 insight 的 table:

| Temperature $\tau$ | Virtual Score | Actual Score |
|---|---|---|
| 0.10 | 2086 ± 140 | 193 ± 58 |
| 0.50 | 2060 ± 277 | 196 ± 50 |
| 1.00 | 1145 ± 690 | 868 ± 511 |
| 1.15 | 918 ± 546 | 1092 ± 556 |
| 1.30 | 732 ± 269 | 753 ± 139 |
| Random policy | N/A | 210 ± 108 |
| Gym leaderboard | N/A | 820 ± 58 |

读这个 table 的 intuition:

**$\tau = 0.1$ 时**,M 几乎变成 deterministic LSTM。在 dream 里 score 2086(接近上限 2100),但 transfer 回真实环境只有 193(比 random 还差!)。原因:agent 找到了 exploit M 的 adversarial policy。

**$\tau = 1.15$ 时**,dream 比 reality 难一些,但 transfer 最好 1092。Sweet spot。

**$\tau = 1.30$ 时**,dream 太难,score 下降,但 variance 也下降(732 ± 269 → 753 ± 139)。意味着 agent 学到更 robust 但更 conservative 的策略。

直觉:**dream 比 reality 难一点,transfer 最好**。这跟 domain randomization / sim-to-real 的思路相通。

### 4.4 Cheating the World Model

paper 里最 philosophically interesting 的部分。$\tau = 0.1$ 时 agent 学到了一个神奇策略:它能"自动熄灭"刚被发射出来的 fireball。即使在 dream 里看到 fireball 形成的迹象,agent 移动方式能让 fireball 消失。

为什么?因为 agent 看到了 M 的 *hidden state* $h_t$。这相当于给 agent 直接访问了 game engine 的 internal state,而 player 本来只能看 observation。agent 可以找到 $h_t$ 空间里的"死角"——某些 $(z_t, h_t)$ 组合会让 M 陷入 mode collapse,fireball 模式永远不被 sample 出来。

低 $\tau$ 时 mixture Gaussian 退化为单 modal,agent exploit 这个 imperfection。MDN-RNN 的 mixture design 就是为了缓解:即便真实环境是 deterministic 的,用 mixture of Gaussian 近似成 stochastic,让 agent 不能轻易 exploit。

这跟 Schmidhuber 早期 deterministic RNN world model (1990a,b) 的问题对应:deterministic model 容易被 controller fool。PILCO 用 Bayesian uncertainty 缓解,但不彻底。World Models 用 mixture model + temperature 缓解,但也不彻底。

paper 也提到更 general 的 "Learning to Think" (Schmidhuber 2015a) 框架:C 可以学会 *ignore* M 的不可靠部分,甚至把 M 的 subroutines 当成可调用的程序。但本文没实现这套,只是最简单的 step-by-step planning。

---

## 5. Iterative Training Procedure

paper 第 5 节给了一个更 general 的 iterative scheme,虽然本 paper 的实验只用了一轮:

```
1. Initialize M, C 随机
2. Roll out 真实环境 N 次,存所有 (a_t, x_t)
3. Train M 学 P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t)
   Train C 在 M 内部优化 expected reward
4. 没解决就回到 step 2
```

关键 extension:
- M 还要预测 reward 和 action,这样复杂运动技能(走路)能被吸收进 M
- 然后 C 就能 rely on M 已经学好的 motor skills,自己学更高层 skill
- 这跟 hippocampal replay 现象 (Foster 2017, https://www.annualreviews.org/doi/10.1146/annurev-neuro-072116-031538) 类比:动物休息/睡眠时 replay 经历来 consolidate memory
- Curiosity 可以通过 flip M 的 loss sign 实现:agent 主动去找 M 预测得差的地方,等于去探索未知区域
- 联系到 PowerPlay (Schmidhuber 2013, https://arxiv.org/abs/1210.8385) 和 novelty search (Lehman & Stanley 2011)

---

## 6. 个人 Intuition 总结

读这篇 paper 几个关键 takeaway:

**1. 分离 representation learning 和 policy learning 的极端版本**

传统 model-based RL 试图 end-to-end 训 dynamics + policy。World Models 把它推到极致:V+M 完全 unsupervised,reward-agnostic,可以用任何 generative modeling SOTA 来训。C 极小,可以用 ES 这种 non-gradient 方法。两边各用最适合自己的 optimizer。

**2. Hidden state as predictive features**

$h_t$ 是 LSTM 内部状态,paper 的实验证明它携带 "未来会发生什么" 的信息。给 controller $h_t$ 比给 stacked frames 好。这跟后来 Dreamer (Hafner et al. 2019, https://danijar.com/project/dreamer/) 的 RSSM state 一脉相承。

**3. Dream training = data augmentation via model uncertainty**

调高 $\tau$ 让 dream 比 reality 难,相当于 domain randomization。在更难的 dream 里学到的 policy transfer 回 reality 反而更好。这给 sim-to-real 一个非常 elegant 的 framework。

**4. Model exploitability 是根本难题**

任何 learned dynamics model 都有"死角"。controller 一定会找这些死角。Mixture density + temperature 只是 mitigation,不是 solution。后续 DreamerV2/V3 用更复杂的方法(image space reconstruction + 更长的 horizon planning)继续打这个问题。

**5. 跟 LLM 的 connection**

虽然 paper 写在 transformer 时代之前,但 ideas 完全迁移:
- V = vision encoder (像 CLIP 的 image encoder)
- M = world dynamics transformer (像 Sora 的 spacetime transformer)
- C = policy head

Sora (https://openai.com/sora) 本质上就是一个超大 scale 的 M model,在 latent space 学世界 dynamics。World Models 是 Sora 的 ancestor idea。

OpenAI Sora technical report: https://openai.com/research/video-generation-models-as-world-simulators

**6. 跟 MuZero / AlphaZero 的对比**

MuZero (Schrittwieser et al. 2020, https://www.nature.com/articles/s41586-020-03051-4) 也是 learned dynamics model,但它是 reward-aware 的 end-to-end 训练,且做 explicit MCTS planning。World Models 的 C 不做 explicit planning,只 instinctive react on $h_t$。两者是 model-based RL 谱系上的两个极端点。

---

## 7. 代码 / 资源

- 官方 interactive demo: https://worldmodels.github.io
- David Ha 的 TensorFlow 实现: https://github.com/hardmaru/WorldModelsCode
- PyTorch 实现 (社区): https://github.com/awjuliani/worldmodel
- Distill.pub-style article with interactive demos (用 p5.js + deeplearn.js)

---

## 8. 后续 work 的延伸

如果这篇 paper 让你兴奋,顺着读:

- **PlaNet** (Hafner et al. 2019): https://arxiv.org/abs/1811.04556 — latent state + MPC planning
- **Dreamer** (Hafner et al. 2020): https://danijar.com/project/dreamer/ — actor-critic in latent dream
- **DreamerV2** (2021): https://arxiv.org/abs/2010.02193 — discrete latent,Atari SOTA
- **DreamerV3** (2023): https://arxiv.org/abs/2301.04104 — fixed hyperparameters across domains
- **DayDreamer** (Wu et al. 2022): https://arxiv.org/abs/2206.14176 — real robot dream training
- **Sora** (2024): https://openai.com/research/video-generation-models-as-world-simulators — scaled up world models as video generator
- **Genie** (DeepMind 2024): https://arxiv.org/abs/2402.15391 — unsupervised environment generation from video

Schmidhuber 系列的 ancestor work:
- Making the World Differentiable (1990): http://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf
- Learning to Think (2015): https://arxiv.org/abs/1511.09249
- One Big Net (2018): https://arxiv.org/abs/1802.08864

---

这篇 paper 的价值在于它把一系列 1990 年代的 ideas (Schmidhuber 那批) 用 modern deep learning 工具 (VAE, LSTM, MDN, ES) 重做了一遍,且写得极其清晰,有 interactive demo。它是 model-based RL 的 modern starting point,也是当前 world model 热 (Sora, Genie, JEPA 等) 的直接 ancestor。从 intuition building 角度,它让你理解:为什么 agent 需要一个 internal model,model 和 policy 应该怎么分工,dream training 为什么 work,以及 learned model 为什么不可避免地被 exploit。
