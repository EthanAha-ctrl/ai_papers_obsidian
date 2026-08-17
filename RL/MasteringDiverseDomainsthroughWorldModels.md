---
source_pdf: MasteringDiverseDomainsthroughWorldModels.pdf
paper_sha256: d0385798e8bada8e81b915c1743be81d9ce8776f12ed937e09351995e099ce37
processed_at: '2026-08-05T16:39:20-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DreamerV3 人话版

好 Andrej，我把刚才那堆术语翻译成"跟朋友喝咖啡时讲的"版本。核心技术点都还在，但语气换一下。

paper: https://arxiv.org/abs/2301.04104
代码: https://github.com/danijar/dreamerv3

---

## 这篇 paper 到底干了啥

一句话：**一个 RL algorithm，零调参，打穿 150+ 游戏，还从零开始在 Minecraft 里挖到了钻石。**

听起来好像没啥，但你想想——以前你要训一个 Atari agent，得调 learning rate、entropy、reward scale、network size 一堆东西；换到 DMC control，又得重新调；再换到 Minecraft，基本就得请专家来 babysit。DreamerV3 的 claim 是：**一套 hyperparameter 全通吃**，而且每个 task 都只用一张 A100，不用分布式集群。

这在 RL 圈是很大的 deal。之前大家默认"RL = 调参地狱"，这个 paper 说"不一定"。

---

## 核心思路：先学世界，再在脑子里练

人类学新东西的时候，会在脑子里"预演"。你要抓个杯子，脑子里先模拟"手伸过去、杯子在哪、会不会碰倒"——这就是 world model 的直觉。

Dreamer 就是把这个搬进 RL：

1. **先学一个 world model**——神经网络学会"给定当前状态和 action，下一步会怎样"
2. **在 world model 里做梦**——从真实状态出发，rollout 一段想象的 trajectory
3. **在梦里学 policy**——actor 和 critic 都在想象出来的 trajectory 上训练，不用真的去环境里试

这跟 model-free (PPO/SAC/DQN) 完全不同。model-free 是"每次都要真的去环境里试，试了才知道好不好"。Dreamer 是"学会环境的规律，然后在自己的模拟器里随便试，试错了也不浪费真实 sample"。

这就是为什么 Dreamer sample efficient——**每条真实 experience 可以反复用来做 imagination rollout，等于数据增广了几十倍**。

---

## 三个网络，各干各的

Figure 3 那张图，说穿了就是三个 network：

**1. World Model（"造梦机"）**
- 吃 observation，吐 latent representation
- 能预测下一步的 latent、reward、episode 结不结束
- 能把 latent decode 回 observation（保证 latent 有信息）

**2. Actor（"做梦的人"）**
- 给定 latent state，输出 action
- 在 imagination rollout 里学，不在真实环境里学

**3. Critic（"评分员"）**
- 给定 latent state，预测 return 的分布
- 告诉 actor "你现在的选择值多少分"

训练循环：
- 真实环境跑一会儿，存 replay buffer
- 从 replay sample batch，训 world model
- 从 replayed latent 出发，actor + world model 做 15 步 imagination
- 在 imagined trajectory 上算 return，更新 actor 和 critic
- 回到真实环境，用 actor 采样新数据

---

## RSSM：world model 的心脏

RSSM = Recurrent State-Space Model，从 PlaNet (https://arxiv.org/abs/1811.04551) 那里继承来的。

核心公式（paper 公式 1）：

$$
\begin{aligned}
h_t &= f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
z_t &\sim q_\phi(z_t \mid h_t, x_t) \\
\hat{z}_t &\sim p_\phi(\hat{z}_t \mid h_t)
\end{aligned}
$$

人话翻译：

- $h_t$：**记忆**。deterministic 的 RNN state，总结过去所有历史。用 GRU 算。
- $z_t$：**当前感知**。stochastic 的 latent，表示"当前这一帧里有什么不可预测的东西"。用 categorical 分布（32 个 categorical，每个 d/16 个 class）。
- $x_t$：observation（图片走 CNN，vector 走 MLP）
- $a_t$：action
- $q_\phi$：**encoder**，看当前 obs 推出 $z_t$（posterior，训练时用）
- $p_\phi$：**dynamics predictor**，不看 obs，只看 $h_t$ 预测 $z_t$（prior，imagination 时用）

**关键直觉**：$h_t$ 负责"可预测的部分"（确定性 dynamics），$z_t$ 负责"不可预测的部分"（随机性、sensor noise）。两者拼起来 $s_t = \{h_t, z_t\}$ 就是 Markov state，actor/critic 都从这里读。

**训练 vs imagination 的区别**：
- 训练 world model 时：有 obs，用 $q_\phi$（posterior）encode
- imagination rollout 时：没 obs，用 $p_\phi$（prior）predict
- 所以 dynamics loss 就是让 prior 学会逼近 posterior

---

## World Model 的 Loss：三个 part 互相制衡

公式 2 和 3：

$$
\mathcal{L} = \beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}
$$

权重 $\beta_{\text{pred}}=1, \beta_{\text{dyn}}=1, \beta_{\text{rep}}=0.1$。

**三个 loss 各是啥**：

**$\mathcal{L}_{\text{pred}}$（prediction loss）**：
- 重建 observation：$\hat{x}_t \sim p_\phi(x_t | h_t, z_t)$
- 预测 reward：$\hat{r}_t$
- 预测 continue flag：$\hat{c}_t$
- 就是让 latent 有信息量

**$\mathcal{L}_{\text{dyn}}$（dynamics loss）**：
$$
\mathcal{L}_{\text{dyn}} = \max(1, \text{KL}[\text{sg}(q_\phi) \| p_\phi])
$$
- 让 prior $p_\phi$ 学会预测 posterior $q_\phi$
- posterior 被 stop-gradient（sg），所以只有 prior 在动
- 这是 imagination 时用的，prior 得准

**$\mathcal{L}_{\text{rep}}$（representation loss）**：
$$
\mathcal{L}_{\text{rep}} = \max(1, \text{KL}[q_\phi \| \text{sg}(p_\phi)])
$$
- 让 posterior $q_\phi$ 靠近 prior $p_\phi$
- prior 被 stop-gradient
- 这是让 latent 更可预测，但权重只有 0.1，别压太狠

**为什么 stop-gradient 一前一后？** 如果两个都不 stop，prior 和 posterior 互相拉扯，可能一起塌缩到 trivial 解（latent 全变成常数）。stop-gradient 让它们各自学各自的，dynamics loss 推动 prior，representation loss 轻轻拉 posterior。

**Free bits（`max(1, KL)`）**：KL 小于 1 nat 时 loss 恒为 1，梯度为 0。这是防止"KL 塌缩"——encoder 为了省事把所有信息丢光，KL=0，latent 变废物。free bits 说"至少保留 1.44 bits 信息"。

**1% unimix**：categorical 分布 mix 1% uniform，防止变成 one-hot。one-hot 的 KL 会 spike 到无穷大，训练崩。这个 trick 从 deep VAE 训练经验里来的 (https://arxiv.org/abs/2011.10650)。

**最重要的 ablation 发现（Figure 6b）**：DreamerV3 的 representation **主要靠 reconstruction loss 学**，reward/value gradient 只是辅助。这跟 MuZero 思路完全不同（MuZero 纯靠 reward/value 学 representation）。这个发现暗示——**未来可以在无 reward 的 internet video 上 pretrain world model**，然后 fine-tune 到下游 task。

---

## Symlog：解决"数字大小差太多"的万能膏药

RL 跨 domain 最大的痛点之一：**reward 和 observation 的 magnitude 天差地别**。

- DMC control：vector obs 可能 ±1e3
- Atari：reward 可能 ±1e2
- BSuite：故意把 reward scale 到 1e6 测试 robustness
- Minecraft：return 0~12

一个 loss function 怎么搞定所有？DreamerV3 的答案：**symlog transformation**。

$$
\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)
$$

性质：
- 原点对称（不像 log 只能处理正数）
- $|x| \ll 1$ 时近似 identity（小值不动）
- $|x| \gg 1$ 时近似 $\text{sign}(x)\ln|x|$（大值压缩）

用法（公式 8）：
- network 输出 $f(x, \theta)$
- target 先 symlog 压缩
- loss 是压缩后的 MSE
- readout 时 symexp 解压回来

$$
\hat{y} = \text{symexp}(f(x,\theta)) = \text{sign}(f)(\exp|f| - 1)
$$

**为什么不用其他方案？**
- log 不支持负数 ❌
- running mean/std normalize 引入 non-stationarity，optimizer 状态被打乱 ❌
- Huber loss 大 target 上梯度恒定，stagnate ❌
- clip 大 target 丢信息 ❌

symlog 的精髓：**在 raw output space 学 compressed representation，梯度 magnitude 跟 target magnitude 解耦**。network 可以很快学到大 target（compressed space 里就是线性外推），也不会因为小 target 的噪声被放大。

参考 bi-symmetric log transformation (Webber 2012, https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001)。

---

## Two-Hot Encoding：把回归变成分类

Critic 要预测 return distribution。return 可能从 0 到 1e6，怎么搞？

DreamerV3 的答案：**把 return 离散化成 255 个 exponentially spaced bin，用 classification 训**。

公式 10：

$$
B = \text{symexp}([-20, \ldots, +20]) \quad \text{(255 bins)}
$$

bins 从 -20 到 +20 在 symexp space 均匀，展开后在 linear space 是 exponential spacing。这样小 return 附近 resolution 高，大 return 也能 cover。

**Two-hot encoding（公式 12）**：

假设 target $y$ 落在 bin $b_k$ 和 $b_{k+1}$ 之间：

$$
\text{twohot}(y)_i = \begin{cases}
|b_{k+1} - y| / |b_{k+1} - b_k| & i = k \\
|b_k - y| / |b_{k+1} - b_k| & i = k+1 \\
0 & \text{else}
\end{cases}
$$

就是把 $y$ 按线性插值分到最近的两个 bin。比如 $y = 5.3$，bin 是 5 和 6，那 two-hot 就是 [0.7, 0.3]（0.7 给 bin 5，0.3 给 bin 6）。

**Loss 是 categorical cross entropy**：
$$
\mathcal{L} = -\text{twohot}(y)^T \log\text{softmax}(f(x))
$$

**Readout 是 weighted average**：
$$
\hat{y} = \text{softmax}(f(x))^T B
$$

**为什么这么搞？**

1. **梯度 scale 完全跟 target magnitude 解耦**——不管 target 是 1 还是 1e6，loss 都是 cross entropy over 255 bins，梯度量级一样
2. **可以表示多模态分布**——return 在 random env 下可能双峰，categorical 自然能表示
3. **可以表示任意连续值**——weighted average 落在 bin 之间，不局限于 bin 中心
4. **不需要像 PopArt (https://arxiv.org/abs/1809.04474) 那样在出现新 extreme value 时调整网络权重**

这个 trick 用在 reward head 和 critic 上。是 DreamerV3 robustness 的核心组件之一。

---

## Critic 训练：λ-Return + EMA + Replay

公式 5：

$$
\begin{aligned}
R_t^\lambda &= r_t + \gamma c_t ((1-\lambda) v_t + \lambda R_{t+1}^\lambda) \\
R_T^\lambda &= v_T
\end{aligned}
$$

变量：
- $\gamma = 0.997$（discount）
- $\lambda = 0.95$（bias-variance tradeoff，类似 GAE）
- $c_t \in \{0, 1\}$（episode 结束 flag）
- $v_t = \mathbb{E}[v_\psi(\cdot|s_t)]$（critic 分布的期望）
- $R_t^\lambda$（bootstrap 的 λ-return）

**人话**：imagination 只 rollout 15 步，但 effective horizon 是 $1/(1-\gamma) = 333$ 步。15 步以后的 reward 由 critic bootstrap。λ=0.95 是在"用真实 imagined reward"和"相信 critic 预测"之间做加权平均。

**EMA target critic**：critic 的 supervision target 是它自己的 EMA copy 输出。这跟 DQN target network 一个道理——防止"自己追自己"导致不稳定。EMA decay = 0.98。

**Critic replay loss**：除了在 imagination trajectory 上训，还在 replay buffer 真实 trajectory 上训（loss scale 0.3）。用 imagination 起点的 $R^\lambda$ 作为 value annotation，然后沿真实 trajectory 算 λ-return。这让 critic 在真实 long horizon 上也学好。

**Zero-init trick**：reward predictor 和 critic 的 output weight 初始化为 0。训练初期不会输出大 reward/value，避免 hallucinated signal 把 learning 拖偏。简单但关键。

---

## Actor 训练：最 subtle 的部分

公式 6, 7：

$$
\begin{aligned}
\mathcal{L}(\theta) &= -\sum_t \text{sg}\left(\frac{R_t^\lambda - v_\psi(s_t)}{\max(1, S)}\right) \log \pi_\theta(a_t|s_t) + \eta H[\pi_\theta] \\
S &= \text{EMA}(\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5), 0.99)
\end{aligned}
$$

变量：
- $\eta = 3 \times 10^{-4}$（固定 entropy scale，所有 domain 通用）
- $S$（return range，5-95 percentile 的 EMA）
- $\text{sg}$（stop-gradient，advantage 不参与 actor gradient）

**这里是最关键的设计**，让我详细讲讲为什么。

---

### 为什么 normalize return 而不是 advantage？

PPO 的经典做法是 normalize advantage：$A / \text{std}(A)$。

问题在于 **sparse reward**：
- Minecraft 里大部分 step reward=0
- advantage 也接近 0
- std(advantage) 接近 0
- normalize 后 advantage 变成噪声被无限放大
- 噪声压过 entropy regularizer
- 探索停滞

**DreamerV3 的方案**：normalize **return**（不是 advantage），用 return range $S$。

$$
\text{normalized advantage} = \frac{R - v}{\max(1, S)}
$$

关键在 $\max(1, S)$：

- **Sparse reward (Minecraft)**：return 大（要拿 diamond 需要长链路）→ $S$ 大 → advantage 被压小 → entropy $\eta H$ 相对放大 → **探索**
- **Dense reward (DMC)**：return 小（每步有 reward）→ $S < 1$ → $\max(1,S) = 1$ → advantage 正常 scale → **exploit**

**同一个 $\eta = 3 \times 10^{-4}$ 全 domain 通吃**。这是整个 paper 最聪明的地方。

---

### 为什么用 percentile 不用 min/max？

$$
S = \text{EMA}(\text{Per}(R, 95) - \text{Per}(R, 5), 0.99)
$$

- min/max 对 outlier 敏感——random env 下某些 episode 可达 return 极高，会把 $S$ 拉到巨大，所有 advantage 被压成 0
- 5-95 percentile 砍掉极端值
- EMA (decay 0.99) 在 batch 间平滑，避免 $S$ 抖动

---

### 为什么有 $\max(1, S)$ 的下限？

- 当 return 已经很小（dense reward 收敛阶段），$S < 1$
- 如果除以 $S$，advantage 被放大，噪声 dominate
- $\max(1, S)$ 说"return 小于 1 时不 normalize"
- 这样小 return 下的 advantage 保持原 scale，不会噪声爆炸

**一句话总结这个 trick**：return 大时 normalize 压小 advantage 让 entropy 主导探索；return 小时不 normalize 保持 advantage 让 policy 精进。自适应，无需调参。

---

## Minecraft：压轴大戏

**任务**：从零开始拿 diamond。12 个 milestone 链：
log → plank → stick → crafting table → wooden pickaxe → cobblestone → stone pickaxe → iron ore → furnace → iron ingot → iron pickaxe → **diamond**

每个 milestone +1 reward，一次性。30 分钟一局，100M env steps 预算。

**结果**：
| Method | Return @ 100M |
|--------|---------------|
| Dreamer | 9.1 |
| IMPALA | 7.1 |
| Rainbow | 6.3 |
| PPO | 5.1 |

Dreamer 是**第一个无 human data、无 curriculum 从 scratch 拿 diamond 的算法**。10 个 seed 全部至少一次拿到 diamond（Figure 5）。

对比 VPT (https://arxiv.org/abs/2206.11795)：
- VPT：720 GPU × 9 天，behavioral cloning on YouTube contractor data + RL fine-tune
- DreamerV3：**1 GPU × 9 天**，无 human data
- compute 差 **720 倍**

这就是 world model + robustness tricks 的威力。

---

## 为什么 Dreamer 能 work？Intuition 拆解

### 1. World model 是 representation engine

- Reconstruction + reward + continue + KL 一起 shape latent
- Reconstruction 是主导 signal（Figure 6b ablation 证实）
- 这让 representation 在 reward 稀疏时也能学好
- Minecraft 前 10M step 基本没 reward，但 world model 已经学会"树木长啥样、怎么砍"

### 2. Imagination 是 sample efficiency 引擎

- 每条真实 trajectory 可以反复 rollout imagination
- replay ratio 32 意味着每 env step 做 32 gradient step
- 这是 model-free 永远做不到的

### 3. Robustness tricks 各管一类 failure mode

| Trick | 防啥 failure |
|-------|-------------|
| Symlog | target magnitude 跨 domain 差太大 |
| Two-hot | regression gradient 跟 target 耦合 |
| Percentile return norm | sparse reward 下探索停滞 |
| Free bits | KL collapse |
| 1% unimix | KL spike |
| Zero-init | early training hallucination |
| EMA critic | self-chasing instability |
| Block GRU | 大 model 参数爆炸 |

**每个 trick 单独看都只解决一类问题，合起来 cover 所有 domain 的 failure mode**。这就是为什么 fixed hparams 能 work——不是某个 trick 神奇，是组合起来刚好 cover 住。

### 4. 大 model = sample efficient

这是 model-based 的反直觉优势：
- World model 越大，对环境理解越快
- Imagination rollout 越准确
- Actor/critic 学得越快
- 所以大 model 用更少 env step 达到同样 score

model-free 里大 model 通常 overfit，model-based 里大 model 反而 sample efficient。Figure 6c 证实了这点。

---

## 跟前两代 Dreamer 的关系

| 维度 | V1 (2019) | V2 (2020) | V3 (2023) |
|------|-----------|-----------|-----------|
| Latent | Gaussian | Discrete categorical | Discrete categorical (随 model size 缩放) |
| Domain | Continuous control only | Atari only | 跨所有 domain |
| Hparam | 需调 | 需调 | Fixed |
| Reward predict | Gaussian | Discrete | Symexp two-hot |
| Critic | Gaussian | Discrete | Symexp two-hot + EMA |
| Return norm | 无 | 无 | Percentile (5,95) EMA |
| KL balance | 无 | 有 | 有 + free bits |
| Observation | Raw | Raw | Symlog transformed |

V1 → V2 是把 Gaussian latent 换成 categorical（Atari 上 work 了）。V2 → V3 是加了一堆 robustness tricks 让它跨 domain work。核心 architecture 没大变，变的是 loss function 和训练细节。

---

## 一个直觉总结

DreamerV3 的哲学：**RL 算法应该像 LLM 一样，一套 recipe 跨所有 task**。

LLM 之所以成功，是因为 transformer + next-token prediction + scale 这个 recipe 足够通用，不需要每个 task 调架构。DreamerV3 想说的是：**world model + imagination + robustness tricks 也可以是 RL 的通用 recipe**。

当然，它还没完全到 LLM 那个程度——还需要 reward signal，还需要 online interaction，Minecraft 成功率才 0.4%。但方向是对的。

未来如果能把 world model pretrain on internet video（无 reward），然后 fine-tune 到下游 task（少 reward），那就真的接近 LLM 的 paradigm 了。Paper 最后一段话就在暗示这个方向。

---

## 想自己跑跑看？

最快路径：
1. clone https://github.com/danijar/dreamerv3
2. 跑 Crafter（单 A100 几小时）
3. 看 tensorboard 里 world model 的 reconstruction、KL、reward prediction 各自怎么收敛
4. 然后 Atari100k（单 A100 半天）
5. 再看 Minecraft（单 A100 9 天）

跑一遍比读十遍 paper 有用。

---

要不要我再展开某个具体部分？比如：
- RSSM 在 imagination 时 prior 和 posterior 的切换逻辑
- Two-hot encoding 的 numpy 实现
- Block-diagonal GRU 到底怎么 block
- Minecraft env 的 action space 为什么要 specially design
- 为什么 DMLab 上 Dreamer 100M 步能打 R2D2 1B 步

---

# DreamerV3 深度解析

paper 链接: https://arxiv.org/abs/2301.04104
项目主页: https://danijar.com/project/dreamerv3/
开源实现: https://github.com/danijar/dreamerv3

Andrej，这篇是 Hafner 团队（DeepMind + Toronto）model-based RL 路线的集大成之作。我把它从架构、loss、robustness tricks 一直讲到 Minecraft 实验，重点 build 你的 intuition。

---

## 一、整体定位：为什么 DreamerV3 重要

RL 领域长期有两套范式：
- **Model-free** (PPO, SAC, DQN)：鲁棒但 sample inefficient
- **Model-based** (MuZero, Dreamer)：sample efficient 但 brittle，hyperparameter 调起来痛苦

DreamerV3 的核心 claim：**一套 fixed hyperparameters** 跨越 150+ task（Atari / DMLab / ProcGen / DMC / BSuite / Crafter / Minecraft），并且**每个 task 都在 single A100 上跑**，9 天训出 Minecraft diamond（无 human data、无 curriculum）。

这条研究线的脉络：
- PlaNet (Hafner 2018) — RSSM 雏形，https://arxiv.org/abs/1811.04551
- DreamerV1 (Hafner 2019) — latent imagination + actor-critic，仅 continuous control，https://arxiv.org/abs/1912.01603
- DreamerV2 (Hafner 2020) — discrete latents，Atari 上超人类，https://arxiv.org/abs/2010.02193
- **DreamerV3 (2023)** — fixed hparams across domains

---

## 二、三大组件与训练循环

DreamerV3 由三个网络组成（Figure 3）：

1. **World Model** — 学环境 dynamics，能在 latent 里 rollout 未来
2. **Critic** — 预测 return distribution
3. **Actor** — 选 action，从 imagination trajectory 上学

训练流程：
- Environment interaction：actor 采样 action，存入 replay buffer
- World model training：从 replay sample batch，重建 + reward + continue + KL
- Imagination rollout：从 replayed latent state 出发，actor + world model 生成 H=15 步 abstract trajectory
- Actor/Critic training：在 imagined trajectory 上算 λ-return，更新 actor 和 critic

关键直觉：**actor 和 critic 从来不直接看 observation，只在 latent imagination 里学**。这样 world model 的 representation quality 决定了 policy quality。

---

## 三、RSSM 架构详解

公式 (1) 是整个 paper 的核心：

$$
\begin{aligned}
h_t &= f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) & \text{(sequence model)} \\
z_t &\sim q_\phi(z_t \mid h_t, x_t) & \text{(encoder posterior)} \\
\hat{z}_t &\sim p_\phi(\hat{z}_t \mid h_t) & \text{(dynamics predictor prior)} \\
\hat{r}_t &\sim p_\phi(\hat{r}_t \mid h_t, z_t) & \text{(reward head)} \\
\hat{c}_t &\sim p_\phi(\hat{c}_t \mid h_t, z_t) & \text{(continue head)} \\
\hat{x}_t &\sim p_\phi(\hat{x}_t \mid h_t, z_t) & \text{(decoder)}
\end{aligned}
$$

**变量解释**：
- $h_t \in \mathbb{R}^{8d}$：deterministic recurrent state（block-diagonal GRU，8 个 block，每个 block size = d）
- $z_t$：stochastic latent，由 32 个 categorical 分布组成，每个有 d/16 个 class（DreamerV2 用 32×32，V3 改成随 model size 缩放）
- $x_t$：observation（image 走 CNN，vector 走 MLP + symlog）
- $a_t$：action（discrete 或 continuous 都支持）
- $\phi$：world model 参数
- $q_\phi$：encoder（posterior，看当前 obs）
- $p_\phi$：dynamics predictor（prior，只看 $h_t$）

**直觉**：
- $h_t$ 像"记忆"，确定性地总结过去
- $z_t$ 像"当前感知"，stochastic 表示当前帧的不可预测部分
- model state $s_t = \{h_t, z_t\}$ 是 Markov 的，actor/critic 都从这里读
- 训练时用 posterior $q_\phi$；imagination rollout 时只用 prior $p_\phi$（因为没 obs 可看）

**Straight-through gradient**：从 categorical 采样用 Gumbel-softmax 风格的 straight-through estimator (Bengio 2013, https://arxiv.org/abs/1308.3432)，让采样步骤可以 backprop。

**Block-diagonal GRU** (Van Keirsbilck 2019, https://arxiv.org/abs/1905.12340)：8 个独立 block，避免 standard GRU 在 hidden=8192 时参数量爆炸。每个 block 内部 full recurrent，block 之间通过 input embedding 混合。

---

## 四、World Model Loss：KL Balance + Free Bits

公式 (2) 总 loss：

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\left[\sum_{t=1}^{T}(\beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}})\right]
$$

权重 $\beta_{\text{pred}}=1, \beta_{\text{dyn}}=1, \beta_{\text{rep}}=0.1$。

公式 (3) 三个分量：

$$
\begin{aligned}
\mathcal{L}_{\text{pred}} &= -\ln p_\phi(x_t|z_t,h_t) - \ln p_\phi(r_t|z_t,h_t) - \ln p_\phi(c_t|z_t,h_t) \\
\mathcal{L}_{\text{dyn}} &= \max(1, \mathrm{KL}[\text{sg}(q_\phi(z_t|h_t,x_t)) \| p_\phi(z_t|h_t)]) \\
\mathcal{L}_{\text{rep}} &= \max(1, \mathrm{KL}[q_\phi(z_t|h_t,x_t) \| \text{sg}(p_\phi(z_t|h_t))])
\end{aligned}
$$

**直觉拆解**：

1. **$\mathcal{L}_{\text{pred}}$**：重建 + reward + continue，让 latent 有信息
2. **$\mathcal{L}_{\text{dyn}}$**：让 prior $p_\phi$ 去拟合 posterior $q_\phi$（posterior 被 stop-gradient），训练 dynamics predictor 在 imagination 时好用
3. **$\mathcal{L}_{\text{rep}}$**：让 posterior $q_\phi$ 去靠近 prior $p_\phi$（prior 被 stop-gradient），让 latent 更可预测，但只施加 0.1 的权重

**为什么 stop-gradient 一前一后？** 防止两个 loss 互相拉扯导致 collapse。如果都一起训，dynamics loss 会把 posterior 拉向 prior，representation loss 又会把 prior 推向 posterior，可能塌缩到 trivial 解（latent 全 0）。

**Free bits (1 nat ≈ 1.44 bits)**：`max(1, KL)` 当 KL < 1 时 loss 恒为 1，梯度为 0。这避免了 encoder 把信息全丢光（KL=0）的退化。当 KL 已经很小，focus 转到 prediction loss。

**1% unimix**：encoder 和 dynamics predictor 的 categorical 都 mix 1% uniform：
$$
p_{\text{mixed}} = 0.99 \cdot p_\phi + 0.01 \cdot \text{Uniform}
$$
防止 categorical 变成 one-hot（deterministic），保证 KL 不会 spike 到无穷。这是 deep VAE 训练里常见的 trick (Child 2020, https://arxiv.org/abs/2011.10650)。

**关键 ablation 发现**（Figure 6b）：DreamerV3 **主要靠 unsupervised reconstruction loss 学 representation**，而 reward/value gradient 只是辅助。这跟 MuZero（纯 reward/value 学 representation）思路完全不同。这条结论很重要——暗示可以在无 reward 的 internet video 上 pretrain world model。

---

## 五、Symlog / Symexp：处理 magnitude 不一的目标

公式 (8, 9)：

$$
\begin{aligned}
\mathcal{L}(\theta) &= \frac{1}{2}(f(x,\theta) - \text{symlog}(y))^2 \\
\hat{y} &= \text{symexp}(f(x,\theta)) \\
\text{symlog}(x) &= \text{sign}(x)\ln(|x|+1) \\
\text{symexp}(x) &= \text{sign}(x)(\exp(|x|)-1)
\end{aligned}
$$

**变量**：
- $f(x,\theta)$：神经网络原始输出
- $y$：target（可以是 vector obs，也可以是 reward）
- $\hat{y}$：readout 后的预测

**性质**：
- 关于原点对称（处理负值，不像 log）
- $|x| \ll 1$ 时近似 identity（小值不动）
- $|x| \gg 1$ 时近似 $\text{sign}(x)\ln|x|$（大值压缩）

**为什么不用 log/running normalize/Huber？**
- log 不支持负数
- running mean/std normalize 引入 non-stationarity，optimizer 状态会被打乱
- Huber 在大 target 上梯度恒定，stagnate
- clip 大 target 会丢失信息

**直觉**：symlog 让 network 在 raw output space 学一个 compressed representation，readout 时解压。梯度 magnitude 解耦于 target magnitude。DMC 的 vector obs 可能 ±1e3，Atari reward 可能 ±1e2，BSuite 故意 reward scale 1e6，全都一个 loss function 搞定。

参考 bi-symmetric log transformation (Webber 2012, https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001)。

---

## 六、Symexp Two-Hot：分布式回归

公式 (10, 11, 12)：

$$
\begin{aligned}
\hat{y} &= \text{softmax}(f(x))^T B \\
B &= \text{symexp}([-20 \ldots +20]) \quad \text{(255 bins, 指数间隔)} \\
\mathcal{L}(\theta) &= -\text{twohot}(y)^T \log\text{softmax}(f(x,\theta))
\end{aligned}
$$

two-hot encoding（公式 12）：

$$
\text{twohot}(x)_i = \begin{cases}
|b_{k+1}-x|/|b_{k+1}-b_k| & i = k \\
|b_k - x|/|b_{k+1}-b_k| & i = k+1 \\
0 & \text{else}
\end{cases}
$$

其中 $k = \sum_j \delta(b_j < x)$。

**直觉**：
- 这是 C51 (Bellemare 2017, https://arxiv.org/abs/1707.06887) 分布式 RL 的推广
- 标量 y 落在 bin $b_k$ 和 $b_{k+1}$ 之间，权重按线性插值分到两个 bin
- network 输出 softmax over bins，loss 是 categorical cross entropy
- readout 是 weighted average，可以表示任意连续值（不局限在 bin 中心）

**为什么这样设计 critic 和 reward head？**
- Return distribution 跨 domain 跨几个数量级（Minecraft return 0~12，Atari return 0~1e6，BSuite 故意 1e6 scale）
- 普通 MSE 在大 target 上发散，Huber 停滞
- 把 regression 变成 classification，**梯度 scale 与 target magnitude 完全解耦**
- 多模态 return 分布可以表示（某些 random env 不同 episode 可达 return 差很多）
- 不像 PopArt (Hessel 2019, https://arxiv.org/abs/1809.04474) 需要在出现新 extreme value 时调整网络权重

**实现细节**：bins 从 -20 到 +20 symexp 后展开，注意 summation order——positive 和 negative bins 要分别从小到大累加再合并，避免数值精度问题。

---

## 七、Critic Learning：λ-Return + EMA Target + Replay Value

公式 (4, 5)：

$$
\begin{aligned}
a_t &\sim \pi_\theta(a_t|s_t) \\
v_\psi(R_t|s_t) &\quad \text{(categorical distribution)} \\
\mathcal{L}(\psi) &= -\sum_{t=1}^{T} \ln p_\psi(R_t^\lambda | s_t) \\
R_t^\lambda &= r_t + \gamma c_t((1-\lambda)v_t + \lambda R_{t+1}^\lambda) \\
R_T^\lambda &\triangleq v_T
\end{aligned}
$$

**变量**：
- $\gamma = 0.997$ discount
- $c_t \in \{0,1\}$：episode continuation flag（done 则 0）
- $\lambda = 0.95$：bias-variance tradeoff，类似 GAE 的 λ
- $v_t = \mathbb{E}[v_\psi(\cdot|s_t)]$：critic 分布的期望值
- $R_t^\lambda$：bootstrapped λ-return，融合 imagined rewards 和 critic 预测

**Imagination horizon H=15**：从 replayed state rollout 15 步，超过的部分由 critic bootstrap。$1/(1-\gamma) = 333$ 是 effective horizon，所以 critic 实际覆盖 333 步以远的 reward。

**EMA target critic**：critic loss 的 target 是它自己的 EMA copy 的输出。这类似 DQN target network，但 trick 在于 return 计算时还是用 current critic（保证 consistency），只有 supervision target 用 EMA。EMA decay = 0.98。

**Critic replay loss (β_repal=0.3)**：除了在 imagination trajectory 上训 critic，还在 replay buffer 的真实 trajectory 上训。trick 是用 imagination rollout 起点 state 的 $R_t^\lambda$ 作为 on-policy value annotation，然后沿 replay trajectory 算 λ-return。这让 critic 在真实 long horizon 上也学好。

**Zero-init trick**：reward predictor 和 critic 的 output weight matrix 初始化为 0。这样训练初期不会输出大 reward/value，避免 hallucinated signal 延迟 learning onset。简单但重要。

---

## 八、Actor Learning：Return Normalization with Percentile

公式 (6, 7)：

$$
\begin{aligned}
\mathcal{L}(\theta) &= -\sum_{t=1}^{T} \text{sg}\left(\frac{R_t^\lambda - v_\psi(s_t)}{\max(1, S)}\right) \log \pi_\theta(a_t|s_t) + \eta H[\pi_\theta(a_t|s_t)] \\
S &= \text{EMA}(\text{Per}(R_t^\lambda, 95) - \text{Per}(R_t^\lambda, 5), 0.99)
\end{aligned}
$$

**变量**：
- $\eta = 3 \times 10^{-4}$：固定 entropy scale（所有 domain 通用）
- $S$：return range，5-95 percentile 的 EMA
- $\text{sg}(\cdot)$：stop gradient，advantage 不参与 actor gradient

**为什么这么设计？这是 paper 最 subtle 的部分**

经典做法的痛点：
- **Normalize advantage**（PPO 做法）：sparse reward 时 advantage 接近 0，normalize 后 noise 被 amplify 到压过 entropy → 探索停滞
- **Normalize by stddev**：sparse reward stddev ≈ 0，除以小数 reward 被无限放大
- **Constrained optimization** (SAC/MPO)：固定 entropy average，sparse reward 下探索太慢
- **直接用 raw advantage**：reward scale 跨 domain 差几个数量级，entropy scale 没法统一

**DreamerV3 的方案**：
1. **Normalize return（不是 advantage）**：$(R - v) / S$，advantage 跟着 return 一起被 scale
2. **$\max(1, S)$ 分母下限**：当 return 小（dense reward 接近收敛）时 $S < 1$，分母变 1，advantage 不被放大 → 避免噪声放大
3. **Percentile (5, 95) 而非 min/max**：鲁棒于 outlier（random env 下某些 episode 可达 return 极高）
4. **EMA smoothing (0.99)**：S 在 batch 间平滑，稳定

**直觉 unification**：
- Sparse reward (Minecraft)：return 大（要拿 diamond 需要长链路）→ S 大 → advantage 被压小 → entropy $\eta H$ 相对放大 → 探索
- Dense reward (DMC locomotion)：return 小（每步都有 reward）→ S 小 → max(1,S)=1 起作用 → advantage 正常 scale → exploit
- 同一个 $\eta = 3\times10^{-4}$ 全 domain 通吃

这个 trick 单独看简单，但配上 entropy regularizer 一起，就是 paper 的 robustness 核心。Ablation 里它排在第二重要（仅次于 world model KL）。

**Reinforce estimator**：actor 用 Williams 1992 的 REINFORCE (https://www.semanticscholar.org/paper/Simple-statistical-gradient-following-algorithms-Williams/...)，对 discrete 和 continuous action 通用。continuous 用 Gaussian + reparameterization-free score function。

---

## 九、Minecraft Diamond：里程碑实验

**任务设定**：
- 64×64×3 first-person image + inventory vector + health/hunger/breath
- 12 milestone：log → plank → stick → crafting table → wooden pickaxe → cobblestone → stone pickaxe → iron ore → furnace → iron ingot → iron pickaxe → **diamond**
- 每个 milestone reward +1（一次性），health loss -0.01/heart
- 30 分钟一局（36000 steps @ 20Hz）
- 100M env steps 预算

**结果（Figure 5, Table 5）**：
| Method | Return @ 100M |
|--------|--------------|
| Dreamer | 9.1 |
| IMPALA | 7.1 |
| Rainbow | 6.3 |
| PPO | 5.1 |

**关键 claim**：DreamerV3 是第一个**无 human data、无 curriculum** 从 scratch 拿到 diamond 的算法。10 个 seed 全部至少一次拿到 diamond（Figure 5），episode-level 成功率 0.4%（Figure 9）。

**对比 VPT** (Baker 2022, https://arxiv.org/abs/2206.11795)：
- VPT: 720 GPU × 9 天，behavioral cloning on YouTube contractor data + RL fine-tune
- DreamerV3: **1 GPU × 9 天**，无 human data
- compute 差 720×

**对比 Voyager** (Wang 2023, https://arxiv.org/abs/2305.16291)：用 LLM API + MineFlayer scripting layer，属于完全不同的 paradigm，不算 fair comparison。

**为什么 Dreamer 能在 Minecraft 上 work？**
- World model 学到 long-horizon dynamics（Figure 7 显示 45 步 future prediction 很清晰）
- Return normalization 让 sparse reward 下探索自动放大
- Reconstruction loss 学到 representation（inventory、画面结构），即使 reward 稀疏也能学

---

## 十、Benchmark 全景（Figure 1, Table 6-13）

| Benchmark | Tasks | Steps | Dreamer 表现 |
|-----------|-------|-------|-------------|
| Atari 200M | 57 | 200M | 超过 MuZero、Rainbow、IQN |
| ProcGen 50M | 16 | 50M | 匹配 PPG (Cobbe 2021, https://arxiv.org/abs/2009.04496) |
| DMLab 100M | 30 | 100M | 100M 步达到 R2D2+ 1B 步水平，1000% data efficiency |
| Atari100k | 26 | 400K | 仅次于 EfficientZero（但 EffZero 改了 env setting） |
| Proprio Control | 18 | 500K | SOTA，超 D4PG/DMPO/MPO |
| Visual Control | 20 | 1M | SOTA，超 DrQ-v2 (Yarats 2021, https://arxiv.org/abs/2107.09645) / CURL |
| BSuite | 23 | - | SOTA，scale 类目尤其强 |
| Crafter | 1 | 1M | - |
| Minecraft | 1 | 100M | 第一个 from scratch diamond |

**特别值得注意的 ablation（Figure 6a）**：所有 robustness tricks 都贡献，但**每个 trick 只在 subset of task 上 critical**。这就是为什么 fixed hparams 难做——每个 trick 都为了 cover 某一类 failure mode。

---

## 十一、Scaling Properties（Figure 6c, 6d）

Model size 从 12M 到 400M 参数（Table 3）：

| Param | Hidden d | Recurrent (8d) | Codes/latent |
|-------|---------|----------------|--------------|
| 12M | 256 | 1024 | 16 |
| 25M | 384 | 3072 | 24 |
| 50M | 512 | 4096 | 32 |
| 100M | 768 | 6144 | 48 |
| 200M | 1024 | 8192 | 64 |
| 400M | 1536 | 12288 | 96 |

**关键发现**：
- 性能随 model size 单调上升
- **更大 model 反而需要更少 env interaction**（同样 score 用更少 step）
- 这违反一般直觉（大 model 通常更易 overfit sample）

**直觉**：world model 越大，对环境理解越快，imagination rollout 越准确，actor/critic 学得越快。这是 model-based 的天然优势——大 model 在 sample efficiency 上有正向收益，跟 model-free 不同。

**Replay ratio**：32~1024 可调，控制 gradient steps / env steps 比率。更大 ratio = 更多 compute = 更少 env interaction。提供了一个 predictable compute-performance tradeoff。

---

## 十二、Implementation Details（值得借鉴的 trick 集合）

1. **Adaptive Gradient Clipping (AGC)** (Brock 2021, https://arxiv.org/abs/2102.06171)：clip 阈值 = 30% weight matrix L2 norm，与 loss scale 无关
2. **LaProp optimizer** (Ziyin 2020, https://arxiv.org/abs/2002.04839)：RMSProp 先 normalize gradient，再 momentum 平滑，$\epsilon=10^{-20}$ 可以取极小（Adam 因为 momentum 和 normalizer 都在 raw gradient 上算，必须 $\epsilon \sim 10^{-8}$）
3. **RMSNorm + SiLU activation**：替代 LayerNorm + ReLU
4. **Online queue + uniform replay**：每个 minibatch 16×64 中先用 online trajectory，剩下从 replay uniformly sample。prioritized replay 也 work 但实现复杂，没用
5. **Latent state 存入 replay buffer**：避免 replay 时重新 burn-in RNN state
6. **Image encoder**：stride-2 CNN 到 6×6 或 4×4，flatten
7. **Image decoder**：transposed stride-2 CNN，sigmoid output
8. **Vector I/O**：3-layer MLP，symlog transform

---

## 十三、与前两代 Dreamer 的 Diff

| 维度 | DreamerV1 | DreamerV2 | DreamerV3 |
|------|-----------|-----------|-----------|
| Latent | Gaussian | Discrete categorical | Discrete categorical (随 model size 缩放) |
| Domain | Continuous control only | Atari | 跨所有 domain |
| Hparam | 需调 | 需调 | Fixed across 150+ tasks |
| Reward predict | Gaussian | Discrete | Symexp two-hot |
| Critic | Gaussian | Discrete (DreamerV2 引入) | Symexp two-hot + EMA target |
| Return norm | 无 | 无 | Percentile (5,95) EMA |
| Observation | Raw | Raw | Symlog transformed |
| KL balance | 无 | 有 | 有 + free bits |
| Optimizer | Adam | Adam | LaProp + AGC |
| Sequence model | GRU | GRU | Block-diagonal GRU |

---

## 十四、Intuition 总结：为什么 DreamerV3 work

我把核心直觉梳理一遍：

**1. World model 是 representation engine**
- Reconstruction + reward + continue + KL 一起 shape latent
- Reconstruction 是主导 signal（ablation 证实）
- 这让 representation 在 reward 稀疏时也能学好

**2. Imagination 是 sample efficiency 引擎**
- 每条真实 trajectory 可以反复 rollout imagination
- replay ratio 32 意味着每 env step 做 32 gradient step（在 imagination 上）
- 这是 model-free 永远做不到的

**3. Robustness 是 generalization 引擎**
- symlog 处理 magnitude 跨 domain
- two-hot 把 regression 变 classification，解耦 gradient scale
- percentile return norm 让 entropy 在 sparse/dense reward 下自适应
- free bits 防 representation collapse
- 1% unimix 防 KL spike
- zero-init 防 early hallucination
- 每个 trick 各管一类 failure mode，合起来 cover 所有 domain

**4. 大 model = sample efficient**
- World model 越准，imagination rollout 越有用
- 大 model 不是 overfit risk，是 sample efficiency gain
- 这跟 model-free 的 scaling 直觉相反

**5. Distributional + symexp = magnitude-agnostic regression**
- Critic 预测 return distribution 而非点估计
- two-hot encoding 让任意 magnitude 都能学
- 这统一了 Minecraft (return 0~12) 和 Atari (return 0~1e6)

---

## 十五、未解决的 / Future Work

paper 自己提到的：
- 在 internet video 上 pretrain world model
- 跨 domain 单一 world model（lifelong learning）
- Minecraft diamond episode 成功率才 0.4%，离 robust 还有距离
- BSuite 的 Deep Sea exploration task Dreamer 完全失败（0%）

我观察到的潜在 limitation（hallucinate 一点）：
- World model 在 stochastic environment 上 imagination rollout 有 compounding error
- Imagination horizon H=15 是固定的，可能某些 task 需要更长 planning
- 没有 tree search（不像 MuZero），完全靠 actor amortize planning
- Symexp two-hot 的 bin 数 (255) 是固定的，对超极端 return 可能不够 resolution
- Critic EMA decay 0.98 是 fixed，可能不同 task 需要不同

---

## Reference Links 汇总

**核心 paper**:
- DreamerV3: https://arxiv.org/abs/2301.04104
- DreamerV2: https://arxiv.org/abs/2010.02193
- DreamerV1: https://arxiv.org/abs/1912.01603
- PlaNet (RSSM 起源): https://arxiv.org/abs/1811.04551
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dyna (Sutton 1991): https://dl.acm.org/doi/10.1145/122344.122377

**对比 baseline**:
- PPO: https://arxiv.org/abs/1707.06347
- MuZero: https://arxiv.org/abs/1911.08265
- SAC: https://arxiv.org/abs/1801.01290
- DQN: https://arxiv.org/abs/1312.5602
- Rainbow: https://arxiv.org/abs/1710.02298
- IQN: https://arxiv.org/abs/1806.06923
- C51 (distributional): https://arxiv.org/abs/1707.06887
- PPG: https://arxiv.org/abs/2009.04496
- SPR: https://arxiv.org/abs/2007.04957
- IRIS: https://arxiv.org/abs/2209.00588
- TWM: https://arxiv.org/abs/2211.05115
- EfficientZero: https://arxiv.org/abs/2111.00210
- DrQ-v2: https://arxiv.org/abs/2107.09645
- CURL: https://arxiv.org/abs/2004.04136
- IMPALA: https://arxiv.org/abs/1802.01561
- R2D2: https://arxiv.org/abs/1809.07628

**Minecraft 相关**:
- VPT: https://arxiv.org/abs/2206.11795
- MineRL competition: https://arxiv.org/abs/1904.10079
- Voyager: https://arxiv.org/abs/2305.16291
- MALMO: https://arxiv.org/abs/1902.09544

**Benchmark**:
- Atari ALE: https://arxiv.org/abs/1207.4708
- ProcGen: https://arxiv.org/abs/1811.12832
- DMLab: https://arxiv.org/abs/1612.03801
- DeepMind Control Suite: https://arxiv.org/abs/1801.00690
- BSuite: https://arxiv.org/abs/1908.03568
- Crafter: https://arxiv.org/abs/2109.06780

**技术细节**:
- Straight-through estimator: https://arxiv.org/abs/1308.3432
- Free bits: https://arxiv.org/abs/1606.04934
- Symlog (bi-symmetric log): https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
- AGC (NFNet): https://arxiv.org/abs/2102.06171
- LaProp: https://arxiv.org/abs/2002.04839
- PopArt: https://arxiv.org/abs/1809.04474
- Block GRU: https://arxiv.org/abs/1905.12340
- GRU: https://arxiv.org/abs/1412.3555
- Two-hot distributional: https://arxiv.org/abs/1707.06887
- REINFORCE: https://link.springer.com/article/10.1007/BF00992696

**代码与项目**:
- DreamerV3 官方实现: https://github.com/danijar/dreamerv3
- Danijar Hafner 主页: https://danijar.com/

---

如果你想 dive deeper，我建议从两个方向 build intuition：

1. **跑 DreamerV3 repo 的 Crafter 实验**（最快，单 A100 几小时）：直接看 RSSM 训练时的 reconstruction + KL 行为，理解 free bits 怎么 work
2. **读 DreamerV2 paper**：V3 的 robustness 是在 V2 的 discrete latent 基础上加的，理解 V2 的 categorical latent 设计先于 V3 的 symlog/two-hot 更顺

要不要我再展开讲某个具体部分？比如 RSSM 在 imagination 时 prior vs posterior 的细节、或者 two-hot encoding 的实现 trick、或者 Minecraft env 的 action space 设计？
