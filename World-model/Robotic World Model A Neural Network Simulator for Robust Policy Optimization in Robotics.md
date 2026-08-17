---
source_pdf: Robotic World Model A Neural Network Simulator for Robust Policy Optimization
  in Robotics.pdf
paper_sha256: 613d81eb472e437a76447f408f46ddeb0d786da29fae81e66089db1bfb1b7a14
processed_at: '2026-08-12T01:58:04-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RWM

## 这 paper 到底想干啥

一句话：**能不能用神经网络学一个"假 simulator"，在这个假 simulator 里训机器人 policy，然后直接扔到真机器人上跑？**

听起来简单，但这是 robotics + RL 领域十年的圣杯问题。之前没人真正在复杂机器人（四足、人形）上可靠地做成过。这篇做成了。

## 为什么这事儿难

想象你在写一个游戏 AI。正常做法：让 AI 在游戏里跑几亿次，慢慢学会怎么玩。这在 simulator 里没问题——simulator 不要钱，跑得飞快。

但真实机器人呢？你让 ANYmal 在实验室跑几亿次？机器人会坏，人会累，电费会爆。

所以大家想了个办法：**先学一个"环境模型"（world model），这个模型能预测"如果我做 action A，世界会变成什么样"。然后在脑子里（imagination）用这个模型跑 RL，等于在脑内 simulator 里训练。**

听起来很美，但有两个老大难问题：

**问题 1：误差滚雪球。** 你预测下一步可能差一点点，但 RL 训练要在脑内 rollout 100 步。每一步的小误差都喂回下一步当输入，几十步后就完全 hallucinate 了——模型预测出来的轨迹跟现实毫无关系。你在一个错误的"世界"里训出来的 policy，扔到真机器人上肯定崩。

**问题 2：不连续动力学。** 四足机器人走路，脚一会儿踩地一会儿抬起来。踩地和抬起的瞬间，dynamics 是突变的。这种 discontinuity 对神经网络 model 特别难学。

之前的工作怎么绕过这俩问题？**靠注入人类知识**。比如告诉模型"脚的落点遵循这个物理公式"、"物体满足这个 invariance"、"用 Lagrangian 力学结构"。有效，但每换一个任务就要重新设计架构，没法泛化。

RWM 的 thesis：**不注入任何 domain knowledge，纯靠一个 general 的 training scheme，就能学到能 rollout 100 步不爆的 world model。**

## RWM 怎么搞的——核心 idea

核心就一个词：**autoregressive training**。

说人话：**训练的时候，就让模型吃自己吐出来的预测结果，而不是总吃 ground truth。**

### 对比一下两种训练方式

**Teacher-forcing（老办法）：**
- 第 1 步：给真实 observation → 预测第 2 步
- 第 2 步：给**真实** observation → 预测第 3 步
- 第 3 步：给**真实** observation → 预测第 4 步
- ...

训练时每一步都用 ground truth，所以可以 fully parallelized，训练快。

问题：**inference 时没有 ground truth 啊！** inference 时只能吃自己上一步的 prediction。模型从来没见过"自己 prediction 出来的 observation"长什么样，一 rollout 就懵了。这就是 train/test distribution mismatch。

**Autoregressive training（RWM 的办法）：**
- 第 1 步：给真实 observation → 预测第 2 步 $o_2'$
- 第 2 步：给**自己预测的** $o_2'$ → 预测第 3 步 $o_3'$
- 第 3 步：给**自己预测的** $o_3'$ → 预测第 4 步 $o_4'$
- ...
- 连续 rollout 8 步（forecast horizon N=8）

训练时就让模型暴露在自己的 prediction error 下，等于逼它学会"如果我上一步预测偏了一点，下一步怎么 self-correct"。

这就是 RWM 跟所有前作的根本区别。Teacher-forcing 是 autoregressive training 的退化情况（N=1）。

### Dual-autoregressive——为什么不是一层 autoregression

RWM 还有个精巧设计叫 dual-autoregressive：

- **Inner autoregression**：在 history horizon M=32 步内，每一步用真实 (observation, action) 更新 GRU 的 hidden state。这是在"读取历史"，相当于让模型从过去 32 步推断当前的真实 latent state（belief state estimation）。
- **Outer autoregression**：在 forecast horizon N=8 步内，把预测的 observation 喂回 GRU 自己。这是在"脑内 rollout"，训练模型在自己生成的轨迹上继续预测的能力。

直觉：**inner AR 学"怎么理解现在"，outer AR 学"怎么想象未来"**。两层合起来，model 同时学会了 belief estimation 和 long-horizon imagination。

### 为什么选 GRU 不选 Transformer

paper 里明确比较了。Transformer 在 autoregressive training 下不 scale，因为 8 步 unrolled backprop 时 attention 的显存爆了。GRU 是 recurrent 的，gradient 自然流过去，显存友好。

而且 GRU 的 reset gate 对 contact switching 这种 discontinuity 天然友好——遇到突变可以 hard forget 旧 belief，attention 是 weighted average，对突变不敏感。

所以这里 GRU > Transformer 不是因为 GRU 表达力强，而是因为 **autoregressive training 的工程约束**正好卡住了 transformer。

### Privileged info 当 auxiliary task

还有一个细节很关键：model 除了预测 observation，还要预测 "privileged info"——contact forces、foot height 这种只有 simulator 才能拿到的 ground truth。

policy 在真机器人上看不到这些，但 world model 内部被强迫 encode 这些信息。等于在 model 的 hidden state 里塞了一个隐式的 system identification module。这让它对 contact 这种关键 event 的预测准很多。

## MBPO-PPO——怎么在 world model 里训 policy

这部分其实 straightforward。流程：

1. 用 current policy 跟真实环境（simulator）交互，存到 replay buffer
2. 用 replay buffer 里的 data autoregressive 训 RWM
3. 从 replay buffer sample initial observation，启动 4096 个 "imagination agents"
4. 每个 imagination agent 用 (policy, RWM) rollout **100 步**——纯在 neural network 里跑，不碰真实环境
5. 拿这 100 步 imagined trajectory 跑 PPO 更新 policy
6. 回到 1

**恐怖点是 100 步 rollout**。之前的 MBPO 只敢 rollout 5 步，Dreamer 也短。RWM 能 rollout 100 步说明 model fidelity 真的够当 simulator 用了。

4096 个并行 imagination agents 等同于 PPO 的 4096 个 parallel envs——保留了 PPO 的 batch efficiency。

### 一个工程细节：必须 pretrain

paper 承认 online fine-tune 之前必须先在 sim data 上 pretrain RWM。原因是：online data 量太小（单环境），而且 immature policy 老摔，生成的 data 都是"摔倒轨迹"，质量太差。从 garbage data 训不出好 model。

但 pretraining 不需要 optimal policy 的 data，suboptimal policy 就够。RWM 对 domain shift robust（Figure 3b 证明），所以 pretrain 完 fine-tune 到 target domain 能 work。

Manipulation 任务不需要 pretrain——因为 dynamics 平滑连续，没 contact 突变，online 就能学。

## 结果怎么样

### Prediction 精度

50Hz 控制下，RWM 在 100+ 步 rollout 后仍跟 ground truth 高度吻合。velocity、joint position 都准。这是 PPO 能收敛的前提。

### Robustness

加 Gaussian noise 到 obs 和 action 上，MLP baseline 几步就爆，RWM 在所有 noise level 下都稳。GRU 的 recurrent memory + autoregressive training 让 model 隐式学会了 denoising。

### 跟 baseline 比

- **RWM-AR（autoregressive training）全场最优**，尤其在 discontinuous 的 legged locomotion 上优势大
- **RWM-TF（teacher-forcing）**在 long rollout 上崩——证明 autoregressive training 是关键，跟架构无关
- **RSSM + autoregressive training** 也能接近 RWM-AR——说明 training scheme 比 architecture 重要
- **Transformer + autoregressive training** 不 scale，爆显存

### Policy 训练 + 硬件部署

对比 SHAC（differentiable sim + first-order gradient）和 DreamerV3：

- **MBPO-PPO**：model error 单调下降，reward 收敛到 ~0.9 ground truth
- **SHAC**：first-order gradient 遇到 contact discontinuity 就不准，policy 学出 chaotic behavior，反过来污染 RWM training data，恶性循环
- **DreamerV3**：收敛但慢，short planning horizon 限制了对 long-horizon 的处理

**ANYmal D 和 Unitree G1 都 zero-shot deploy 成功**——能 track velocity command，能抗 external perturbation（踢、推）。SHAC 和 Dreamer 都训不出可部署 policy。

### 跟 model-free PPO 的对比

reward 持平（0.90 vs 0.90），但 RWM 用了 40 倍少的 transitions（6M vs 250M）。

不过说实话——RWM 的 6M 也是在 sim 里 collect 的，所以这个对比的意义是：**如果 sim 不够准（deformable terrain、fluid、granular media），RWM 可以用 limited data 替代 high-fidelity sim**。这才是 MBRL 真正有价值的场景。

## 还差什么（limitations）

paper 自己也很诚实：

1. **还比不过 model-free on perfect sim**。model-free PPO 在 high-fidelity simulator 上调好了，性能上限更高。RWM 的价值在于 sim 不准或 sim 不存在的场景。

2. **Pretraining 依赖 sim**。虽然 online fine-tune 用 limited data，但 pretraining 阶段还是要 sim。想完全摆脱 sim，需要根本性的 safe exploration + collision recovery 机制。

3. **没做 real online learning**。A.4.4 节承认：online learning 时 policy 会 exploit model 的小误差，导致过于 optimistic 的行为，在真硬件上会摔 20+ 次。sim 里摔了能 reset，真机器人摔了没人扶。所以目前用 "single sim env with domain shift" 近似 real constraint。

4. **没 explicit uncertainty**。model 输出 Gaussian mean/std，但 std 没被 policy 利用。policy 不知道"哪里 model 不确定"，没法 conservative。Ensemble methods（PETS 那种）在这点上有优势。

5. **Reward 必须从 obs + privileged info 能算**。如果 reward 涉及 sim-only 变量（精确 contact force magnitude），RWM 没法 rollout reward。

6. **100 步其实也不长**。50Hz 下 100 步 = 2 秒。对 locomotion 够，对 long-horizon manipulation（pick → place → assemble）可能不够。

## 我自己的几点 intuition

### 1. Train/inference distribution match 是宇宙终极规律

RWM 的成功本质上就是解决了一个 distribution shift 问题。这个原则无处不在：

- LLM RLHF：训 policy 时 response 分布必须跟 inference 时 sample 出来的分布一致，否则 reward hacking
- GAN：generator 训练分布必须跟 inference 分布对齐
- World model：训练时 input 分布必须跟 rollout 时 input 分布对齐

Teacher-forcing 违反这个原则，autoregressive training 遵守这个原则。就这么简单。

### 2. Architecture 没那么重要，training scheme 才是关键

paper 里 RSSM + autoregressive training 也能接近 RWM-AR 的性能。说明架构（GRU vs RSSM vs Transformer）是 secondary 的，**怎么训**才是 primary 的。这个 insight 对整个 field 都有启发意义——大家太沉迷于 design architecture，忽视了 training scheme 的 distribution match 问题。

### 3. "当 simulator 用"是个很高的 bar

之前 world model 的工作，rollout 5 步就算"long horizon"了。RWM 跑 100 步还能保持 fidelity，这跨越了一个 qualitative threshold——从"辅助 planning"升级到"真正当 simulator 用"。这个 threshold 一旦跨过，很多 downstream application 就 unlock 了。

### 4. Pretraining 的必要性暗示了一个更深的问题

完全 from-scratch 的 online MBRL 在 discontinuous dynamics 下可能根本不稳定。policy 在烂 model 上 exploration → 产生 fall data → model 更烂 → 死循环。这是 chicken-and-egg 问题。

解决方向：
- Curriculum learning（先学简单 task，model 好了再学难的）
- Safety-aware exploration（不让 policy 去危险区域）
- Uncertainty-aware model（policy 知道哪里 model 不准，保守行事）
- Recovery policy（摔了能自己起来）

这四个里任何一个 breakthrough 都能让 RWM 真正脱离 sim 依赖。

### 5. 跟 MuZero 的哲学一致

MuZero 也是 learned dynamics model + planning，只是用在 discrete game（围棋、Atari）。RWM 是这个思想在 continuous robotics 上的 realization。

但 RWM 比 MuZero 难一个量级：
- MuZero 的 game dynamics 是 discrete 且 deterministic
- RWM 的 robot dynamics 是 continuous、stochastic、discontinuous（contact）

所以 RWM 用 PPO（zero-order）而不是 MCTS（需要 accurate value estimation），用 GRU 而不是 attention，都是被 problem nature 逼出来的选择。

### 6. 真正的 real-data online learning 是下一个 holy grail

paper 的 deployment 虽然是 zero-shot 到 hardware，但 training 阶段还是在 sim 里。真正的 endgame 是：**在真机器人上边 collect data 边训 model 边训 policy，完全不碰 sim**。

这需要解决：
- Safe exploration（不摔坏机器人）
- Collision recovery（摔了能起来继续）
- Privileged info 的 sensor-based estimation（contact force from FT sensor / tactile skin）
- Uncertainty-aware model（知道哪里不知道）

paper 的 A.4.4 明确说这是 ongoing work。如果做成了，robotics learning 的范式会彻底改变——从 "sim-to-real" 变成 "real-only"。

### 7. 跟 neuroscience 的呼应

"World model" 这个概念来自 neuroscience——大脑确实 maintain internal model 来 predict sensory consequences of action，用于 motor planning。

RWM 的 dual-autoregressive 机制有点像大脑的 predictive coding：
- Inner AR ≈ lower-level sensory processing（处理当前 input）
- Outer AR ≈ higher-level simulation（想象未来）

这种 layered 的 prediction 机制在认知科学里有大量对应。RWM 算是把这个 idea 在工程上实现了。

## 给 Karpathy 的一句话总结

如果你只有 30 秒：**RWM 发现让 world model 在训练时就吃自己的 prediction（autoregressive training，而非 teacher-forcing），就能学到能 rollout 100 步不爆的 dynamics model，配合 PPO 在 imagination 里训出的 policy 能 zero-shot 部署到 ANYmal 和 G1。核心 insight 是 train/test distribution match，架构是 secondary 的。下一步是 real-data online learning，需要解决 safe exploration 和 uncertainty estimation。**

---

## Reference Links

- Project page: https://sites.google.com/view/roboticworldmodel
- Dreamer: https://danijar.com/project/dreamer/
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://tdmpc2.com/
- MBPO: https://arxiv.org/abs/1906.08253
- PPO: https://arxiv.org/abs/1707.06347
- SHAC: https://arxiv.org/abs/2204.07137
- DayDreamer: https://github.com/danijar/daydreamer
- MuZero: https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
- Schmidhuber world models: https://www.schmidhuber.de/sn/worldmodel.html
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- ANYmal: https://www.anybotics.com/
- Unitree G1: https://www.unitree.com/g1/

---

# Robotic World Model (RWM): 技术深度解析

## 1. 一句话核心直觉

这篇 paper 想解决一个根本问题：**能不能用纯神经网络当 simulator 来训练机器人 policy，然后直接 zero-shot 部署到真实硬件上？** 答案是 yes——只要 world model 的 autoregressive rollout 能稳定跑超过 100 步累积误差不爆炸。RWM 的核心贡献就是搞清楚怎么训这个 model。

## 2. 动机：为什么 model-based RL 在机器人上一直没work

先建立大局观。model-free RL (PPO, SAC) 在 simulator 里很强，但需要几亿次 interaction；真实硬件上根本玩不起。MBRL 的卖点是用 learned model 替代 simulator，sample efficiency 高几个数量级。但实际困难：

- **error accumulation**: autoregressive rollout 时，每一步的小误差都喂回下一步，几十步后就 hallucinate
- **partial observability**: 真实机器人 state 不可完全观测，contact forces 这种关键信息经常缺失
- **stochasticity & discontinuity**: legged locomotion 有 contact switching，dynamics 不连续
- **domain-specific inductive bias**: 之前的工作 (rigid body dynamics [20], Lagrangian [28], foot-placement [21]) 都靠手工注入物理知识，没法跨任务泛化

RWM 的 thesis：**一个 architecture-agnostic 的 autoregressive training scheme + 简单 GRU 架构，就能学到 robust 的 world model，不需要任何 domain knowledge。**

## 3. 核心方法：Self-supervised Autoregressive Training

### 3.1 POMDP formulation

环境建模为 POMDP $(S, \mathcal{A}, \mathcal{O}, T, R, O, \gamma)$：

- $S$: state space (隐含的，agent 看不到)
- $\mathcal{A}$: action space (joint position targets)
- $\mathcal{O}$: observation space (agent 实际看到的)
- $T: S \times \mathcal{A} \to S$: transition kernel $p(s_{t+1} | s_t, a_t)$
- $R$: reward function
- $O: S \to \mathcal{O}$: observation kernel $p(o_t | s_t)$
- $\gamma \in [0,1]$: discount factor

Policy $\pi_\theta: \mathcal{O} \to \mathcal{A}$ 最大化 $\mathbb{E}_{\pi_\theta}[\sum_{t \geq 0} \gamma^t r_t]$。

直觉：因为只能看 observation，policy 必须从历史推断 latent state，所以 world model 也必须能吃 history。

### 3.2 Equation 1：autoregressive prediction

$$o_{t+k}' \sim p_\phi\left(\cdot \mid o_{t-M+k:t},\; o_{t+1:t+k-1}',\; a_{t-M+k:t+k-1}\right)$$

逐项拆解：

- $o_{t+k}'$：第 $k$ 步 ahead 的 predicted observation（从分布里 sample 出来的，带 prime 表示是 model 自己生成的，不是 ground truth）
- $p_\phi$：参数为 $\phi$ 的 world model，输出 Gaussian 的 mean/std
- $o_{t-M+k:t}$：从 $t-M+k$ 到 $t$ 的历史真实 observations（注意上界 $t$ 固定，下界随 $k$ 滑动，因为 history buffer 长度固定为 $M$）
- $o_{t+1:t+k-1}'$：**自己之前预测的** $k-1$ 个 observations——这就是 autoregressive feedback 的核心
- $a_{t-M+k:t+k-1}$：对应的 actions（真实的 + 当前 policy 给的）

直觉解读：在第 $k$ 步预测时，model 的 input 是 (a) 滑动窗口里剩的真实 history + (b) 自己之前吐出来的 predictions + (c) 对应的 action 序列。这跟 inference 时的 rollout 完全一致——**train/test distribution 对齐**，这是比 teacher-forcing 的根本优势。

### 3.3 Equation 2：multi-step loss

$$\mathcal{L} = \frac{1}{N} \sum_{k=1}^{N} \alpha^k \left[L_o(o_{t+k}', o_{t+k}) + L_c(c_{t+k}', c_{t+k})\right]$$

- $N$: forecast horizon（training 时 rollout 的步数，paper 用 $N=8$）
- $\alpha$: decay factor，paper 里设 $1.0$（不衰减，所以每一步 loss 等权）
- $L_o$: observation 的 discrepancy（Gaussian NLL）
- $L_c$: privileged info $c$（contacts, foot height 等）的 loss——这个 auxiliary task 很关键，强制 hidden state 把 contact 这种隐变量 encode 进去，否则 long-horizon 没法准
- $c_{t+k}'$ vs $c_{t+k}$：predicted vs ground-truth privileged info

为什么这个 loss 比 teacher-forcing 强？teacher-forcing 是 $N=1$ 的特殊情况，每步都用 ground truth 当 input，训练时 parallelizable 但 inference 时没见过"自己的错误输出"长什么样，一 rollout 就 drift。Autoregressive training 让 model 在训练时就暴露在自己的 prediction error 分布下，学到 self-correction。

### 3.4 Dual-autoregressive mechanism (Figure S6)

这是 paper 最精巧的设计，我画一下：

```
Context horizon M (inner autoregression)         Forecast horizon N (outer autoregression)
[o_{t-M+1}, a_{t-M+1}] → GRU → h_{t-M+1}         [o_{t+1}' ← predict, a_{t+1}] → GRU → predict o_{t+2}'
[o_{t-M+2}, a_{t-M+2}] → GRU → h_{t-M+2}         [o_{t+2}' ← predict, a_{t+2}] → GRU → predict o_{t+3}'
...                                              ...
[o_t, a_t] → GRU → h_t (final context)           [o_{t+N}' ← predict]
   ↓                                                  ↓
   └─────── start forecast from h_t ─────────────────┘
```

- **Inner autoregression**: 在 context horizon $M=32$ 内，每一步用真实 $(o, a)$ 更新 GRU hidden state $h$。直觉：让 GRU 的 memory 把过去 32 步的真实动力学累积起来，相当于 implicit belief state estimation。
- **Outer autoregression**: 在 forecast horizon $N=8$ 内，把 predicted observation 反馈给 GRU 自己。直觉：训练 model 在"自己输出的轨迹"上继续预测的能力。

两层 autoregression 的组合让 model 同时学到 (a) 怎么从 history 提取 belief state，(b) 怎么在自己生成的 trajectory 上 forward rollout。这是它能 rollout 100 步不爆的关键。

### 3.5 Reparameterization trick

为了让梯度能 backprop 通过 sampling $o' \sim \mathcal{N}(\mu, \sigma)$，用 reparameterization：$o' = \mu + \sigma \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。这样 N 步 prediction 的 loss 都能端到端优化，等于让梯度穿越 8 步 autoregressive rollout。

## 4. MBPO-PPO：在 world model 上跑 PPO

### 4.1 Algorithm 1 解析

```
1. 初始化 π_θ, p_φ, replay buffer D
2. for each iteration:
   a. 用 π_θ 跟真实环境交互，存到 D
   b. 用 D 里的 data 按 Eq.2 autoregressive 训 p_φ
   c. 从 D sample initial observations，初始化 4096 个 imagination agents
   d. 每个 imagination agent 用 (π_θ, p_φ) rollout 100 步
   e. 用这些 imagined trajectories 跑 PPO 更新 π_θ
```

### 4.2 Equation 3：imagined action

$$a_{t+k}' \sim \pi_\theta(\cdot \mid o_{t+k}')$$

- $a_{t+k}'$：policy 在第 $k$ 步 imagined action
- $\pi_\theta$：current policy
- $o_{t+k}'$：world model 给的 predicted observation（Eq.1 出来的）

注意这里 policy 只看 predicted observation，**不看 ground truth**。整个 rollout 是纯 imagination，等于在 neural network simulator 里跑 PPO。

### 4.3 关键工程细节

- **imagination steps per iteration = 100**：autoregressive rollout 100 步！这是 RWM 的真正恐怖之处。MBPO 原版只能 5 步 rollout，Dreamer 也短。RWM 能跑 100 步说明 model fidelity 极高。
- **imagination environments = 4096**：并行 4096 个 imagination agents，等同于 PPO 的 4096 个 parallel envs。这让 PPO 的 batch efficiency 不丢。
- **model pretraining 是必需的**：online fine-tuning 之前要先在 simulation data（来自 suboptimal policy 即可）上 pretrain。原因：online data 量太小（单环境），且 immature policy 老摔，data 质量差。Pretraining 给一个 robust initialization。Manipulation task 不需要 pretrain（dynamics 平滑连续）。

### 4.4 Collision handling 的细节

RWM 显式预测 termination（base contact → failure）作为 privileged info 的一部分。PPO 在 imagination 里看到 termination 就当 episode 结束，影响 return 计算。这让 model 学会"什么样的 trajectory 会导致摔"——policy 自然避免。

## 5. Architecture 细节 (Table S7)

```
Input: concat(o_t, a_t) ∈ R^(dim_o + dim_a)
   ↓
GRU [hidden 256, 256, 2 layers]
   ↓
h_t ∈ R^256
   ↓
┌──────────────┬──────────────┐
MLP head (128) MLP head (128) 
ReLU           ReLU
   ↓              ↓
μ_o, σ_o      μ_c, σ_c
```

- 输入维度（ANYmal D）：obs 45 + action 12 = 57
- 输入维度（Unitree G1）：obs 96 + action 29 = 125
- GRU 比 transformer 在 autoregressive training 下更 scalable——transformer 的 multi-step gradient propagation 会爆 GPU memory，paper 明确说了

为什么 GRU 而不是 LSTM？GRU 参数少，且 reset gate 对处理 discontinuity（contact switching）天然友好——遇到 contact 状态变化时 reset gate 可以选择性 forget 旧的 belief。

## 6. Observation / Action / Reward 细节

### 6.1 World model observation (Table S2)

ANYmal D（45维）：
- base linear velocity $v$ [0:3]
- base angular velocity $\omega$ [3:6]
- projected gravity $g$ [6:9]
- joint positions $q$ [9:21]（12个关节）
- joint velocities $\dot{q}$ [21:33]
- joint torques $\tau$ [33:45]

Unitree G1（96维）：类似但更多关节（29 DoF）

注意：world model 看到的 obs 比 policy 多（policy 不看 torque，看 last action）。这是合理的——world model 需要完整动力学信息，policy 只需要决策信息。

### 6.2 Privileged information (Table S3)

- ANYmal D: knee contact [0:4] + foot contact [4:8] = 8维
- G1: body contact [0:26] + foot height [26:28] + foot velocity [28:30] = 30维

这些是 simulator 里才能拿到的 ground truth，作为 auxiliary prediction target。Policy 在 deployment 时拿不到这些，但 world model 内部 encode 了。

### 6.3 Reward (Eq. 各种)

总 reward = 各项加权和（Table S6）：

- 线速度 tracking: $r_{v_{xy}} = w_{v_{xy}} \exp(-\|c_{xy} - v_{xy}\|_2^2 / \sigma_{v_{xy}}^2)$，$\sigma=0.25$
- 角速度 tracking: $r_{\omega_z} = w_{\omega_z} \exp(-\|c_z - \omega_z\|_2^2 / \sigma_{\omega_z}^2)$
- 垂直速度惩罚: $r_{v_z} = w_{v_z} \|v_z\|_2^2$，$w=-2.0$
- roll/pitch 角速度惩罚: $r_{\omega_{xy}} = w_{\omega_{xy}} \|\omega_{xy}\|_2^2$，$w=-0.05$
- joint torque 惩罚: $r_{q_\tau} = w_{q_\tau} \|\tau\|_2^2$，$w=-2.5e^{-5}$
- joint acceleration 惩罚: $r_{\ddot{q}} = w_{\ddot{q}} \|\ddot{q}\|_2^2$
- action rate 惩罚: $r_{\dot{a}} = w_{\dot{a}} \|a' - a\|_2^2$（smoothness）
- feet air time reward: $r_{f_a} = w_{f_a} t_{f_a}$
- undesired contact penalty: $r_c = w_c c_u$，$w=-1.0$
- flat orientation: $r_g = w_g g_{xy}^2$，$w=-5.0$（保持 body 水平）
- foot clearance: $r_{f_c} = w_{f_c} h_{f_c}$
- joint deviation: $r_{q_d} = w_{q_d} \|q - q_0\|_1$（向 default pose 回归）

Exponential 形式的 tracking reward 是为了让 reward landscape 平滑可微——这对 model-based 方法特别重要，因为 world model 的小预测误差不会导致 reward 巨变。$\sigma=0.25$ 是温度系数，控制 reward 衰减 sharpness。

## 7. Experiments 深度解读

### 7.1 Autoregressive trajectory prediction (Figure 3a)

50 Hz 控制，M=32, N=8 训练。可视化显示 RWM 在 100+ 步 rollout 后仍跟 ground truth 高度吻合。具体看：

- velocity tracking 类变量（linear/angular velocity）误差很小
- joint 级别变量也保持一致
- 这种 fidelity 在 100 步后还保持，是 PPO 能在 imagination 里收敛的前提

### 7.2 Robustness under noise (Figure 3b)

加 Gaussian noise 到 obs 和 action 上，比较 RWM (yellow) vs MLP baseline (grey)：

- MLP: 几步内误差爆炸，noise 越大爆得越快
- RWM: 在所有 noise level 下都保持低误差，曲线平

直觉解释：GRU 的 recurrent nature + dual-autoregressive training 让 model 学到 "denoising" 的隐式能力。MLP 没有记忆，每步预测都是 myopic 的，error 没法被后续步修正。

### 7.3 Generality across environments (Figure 4)

Baseline 比较：
- **MLP**：简单前馈网络，autoregressive 训练但没 memory
- **RSSM** (PlaNet/Dreamer 用的): GRU + latent state + VAE prior，categorical latent (32 categories × 64 dim)
- **Transformer** (Decision Transformer 风格)：decoder, dim 64, 8 heads, 2 layers, context 32, sinusoidal PE
- **RWM-TF** vs **RWM-AR**：teacher-forcing vs autoregressive training 的消融

关键结果：
1. **RWM-AR 全场最优**，在 legged locomotion 这种 discontinuous 任务上优势最大
2. **RWM-AR >> RWM-TF**：teacher-forcing 训出来的 RWM 在 long rollout 上崩，证明 autoregressive training 是关键
3. **RSSM 加 autoregressive training 也能接近 RWM-AR**，说明 training scheme 比 architecture 重要
4. **Transformer + autoregressive training 不 scale**：multi-step backprop 爆显存

### 7.4 Policy learning & hardware transfer (Figure 5)

对比 SHAC [38] 和 DreamerV3 [30]：

- **MBPO-PPO (RWM)**: model error 单调下降，reward 收敛到 ~0.9 ground truth
- **SHAC**: first-order gradient through differentiable sim，遇到 discontinuous contact 时 gradient 不准，policy 学出 chaotic behavior，反过来污染 RWM training data → 恶性循环
- **DreamerV3**: 收敛但慢，因为 short planning horizon 限制了对 long-horizon dependencies 的处理

Hardware：ANYmal D 和 G1 都 zero-shot deploy 成功，能 track velocity command，能抵抗 external perturbation（踢、推）。SHAC 和 Dreamer 都训不出可部署 policy。

### 7.5 跟 model-free PPO 的对比 (Table 1)

| Method | state transitions | training time | real tracking reward |
|--------|-------------------|---------------|----------------------|
| RWM pretraining | 6M | 50 min | - |
| MBPO-PPO | - | 5 min | 0.90 ± 0.04 |
| PPO (high-fidelity sim) | 250M | 10 min | 0.90 ± 0.03 |

直觉：reward 持平，但 RWM 用了 250M / 6M ≈ 40 倍少的 transitions。MBPO-PPO 总训练时间（50+5=55min）比 PPO (10min) 长，但 PPO 假设你有 perfect simulator，这在真实场景里通常是 luxury。如果按真实硬件 cost 算，40x sample efficiency 是巨大胜利。

## 8. Ablation: M 和 N 的影响 (Figure S8)

- **History horizon M**: 增大降低 error，但 plateau（M=32 后增长有限）。直觉：32 步 = 0.64s 在 50Hz，足够覆盖一个 gait cycle
- **Forecast horizon N**: 增大显著提升 long rollout 性能，但 training time 线性增长（sequential）。N=1 = teacher-forcing，最快但最差
- **Optimal trade-off**: M=32, N=8

## 9. 与相关工作对比

### 9.1 Dreamer 系列 [29, 11, 30]

- Dreamer 用 RSSM 在 latent space rollout，actor-critic 学 latent policy
- 优势：sample efficient
- 劣势：planning horizon 短，长 horizon rollout 仍会累积误差；visual input 任务为主，低层 control 少
- RWM: 直接在 observation space rollout，rollout 100 步，zero-shot 到 hardware

Web: https://danijar.com/project/dreamer/

### 9.2 MBPO [13]

- MBPO: model-based rollouts + model-free fine-tuning，rollout 短 horizon (~5 步)
- RWM: 扩展到 100 步 rollout + PPO，命名为 MBPO-PPO

Web: https://arxiv.org/abs/1906.08253

### 9.3 TD-MPC2 [36]

- TD-MPC2: latent dynamics + MPC，conservative Q learning
- 优势：continuous control SOTA
- 劣势：planning 时 MPC 在线优化，latency 高；locomotion 任务少
- RWM: 训完 policy 直接 forward，1ms inference

Web: https://tdmpc2.com/

### 9.4 DayDreamer [19]

- Dreamer 在真实机器人上的部署
- 主要做 visual task，简单机器人
- RWM: 复杂 locomotion，legged system

Web: https://github.com/danijar/daydreamer

### 9.5 SHAC [38] / differentiable simulation

- SHAC: first-order gradient through differentiable physics
- 问题：contact discontinuity 让 gradient 不准
- RWM: zero-order (PPO) 避免 gradient 通过 discontinuity

Web: https://arxiv.org/abs/2204.07137

### 9.6 Decision Transformer / Trajectory Transformer [41]

- 用 transformer 做 offline RL，sequence modeling 视角
- RWM 借用了 autoregressive 的思想但 purpose 不同：DT 是 return-conditioned policy，RWM 是 dynamics model

Web: https://arxiv.org/abs/2106.01345

## 10. 我的几点 intuition 和联想

### 10.1 Train/inference distribution match 是核心

RWM 的成功本质上是解决了 distribution shift 问题。Teacher-forcing 的训练分布是 ground truth observation，inference 分布是 model 自己生成的 observation——这俩越走越远。Autoregressive training 把训练分布直接对齐到 inference 分布，等同于在 model 自己的 rollout 上做 data augmentation。

这跟 LLM 的 RLHF training 有点像：你训 policy 时用的 response 分布必须跟 inference 时实际 sample 出来的分布一致，否则 reward hacking。RWM 的 autoregressive training 就是这个原则在 world model 上的应用。

### 10.2 Privileged information 作为 auxiliary task

预测 contact 这种 ground truth privileged info 是非常聪明的 design。它强迫 hidden state encode 真正的 latent dynamics（脚是不是着地、是不是要摔），即使 policy 在 deployment 时看不到这些。这等价于一个隐式的 system identification module。

联想到 Kenneth Stanley 的 novelty search：auxiliary objective 让 representation 更丰富，反而对主任务有帮助。

### 10.3 Pretraining 的必要性暗示了什么

Paper 承认必须 pretrain world model 才能 online fine-tune。这暗示一个更深的问题：**完全 from-scratch 的 online MBRL 在 discontinuous dynamics 下可能根本不稳定**。因为 policy 在烂 model 上 exploration → 产生 fall data → model 更烂 → 死循环。Pretraining 用 suboptimal policy 的 data 就够，说明 RWM 对 domain shift robust，但不能从 garbage data 学起。

未来方向：curriculum learning / safety-aware exploration / uncertainty estimation。

### 10.4 为什么 GRU 比 Transformer 在这里好用

Transformer 在 long sequence 上理论上更强，但有两个问题在这个 setting 下致命：
1. **Multi-step backprop memory**：autoregressive training 需要 8 步 unrolled graph，transformer 的 attention 在每一步都 O(N^2)，backprop 时显存爆
2. **Discontinuity**：attention 是 weighted average，对突然的状态变化（contact event）不友好；GRU 的 reset gate 可以 hard switch

这跟最近一些工作发现 small RNN 在 low-dimensional control 上仍然有优势是一致的。

### 10.5 跟 MuZero / AlphaZero 的联系

MuZero 也是 learned dynamics model + planning，但：
- MuZero 用 MCTS，RWM 用 PPO（更适合 continuous action）
- MuZero 在 discrete game，dynamics 平滑；RWM 在 continuous robot，有 contact discontinuity
- MuZero 的 representation / dynamics / prediction heads 分开，RWM 是 unified GRU

但哲学上是一致的：learn a model that supports long-horizon planning。RWM 算是 MuZero 思想在 legged robotics 上的 realization。

Web: https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/

### 10.6 sim-to-real 角度

RWM 提供了一个新的 sim-to-real 路径：
- 传统：在 high-fidelity sim 训 → domain randomization → deploy
- RWM：在 learned neural sim 训 → deploy

如果 learned model 能 capture real dynamics，理论上 zero-shot 更顺。但 paper 还是在 sim 里 collect data 训 RWM，real deploy 只是 policy transfer。真正的 real-data online MBRL 还没完全 work（A.4.4 节承认了 safety 问题）。

### 10.7 Sample efficiency 的真实意义

40x sample efficiency 看起来好看，但 RWM 的 6M transitions 也是在 sim 里 collect 的。所以这个对比的意义在于：**如果 sim 不够准（比如 deformable terrain, fluid, granular media），RWM 可以用 limited real data 替代 high-fidelity sim**。这才是 MBRL 的真正价值场景。

### 10.8 跟 neuroscience 的联系

"World model" 这个概念本身来自 neuroscience——Craik 1943, Schmidhuber 1990s。大脑确实 maintain internal model 来 predict sensory consequences of action，用于 motor planning。RWM 的 dual-autoregressive 机制有点像大脑的 predictive coding：lower level 处理 immediate sensory input (inner AR)，higher level 做 long-horizon simulation (outer AR)。

Web: https://www.schmidhuber.de/sn/worldmodel.html

## 11. Limitations 我自己的补充

1. **Pretraining 依赖 simulator**：虽然 online fine-tune 用 limited data，但 pretraining 还是要 sim。如果想完全摆脱 sim，需要根本性的 exploration safety mechanism。
2. **Single environment online data**：A.4.4 说"approximate real-world constraints by using only a single simulation environment"——这其实是用 sim 里的 single env 模拟 real world 的 single env。真正的 real online learning 没做。
3. **Reward 必须可从 obs + privileged info 计算**：如果 reward 涉及 sim-only 变量（比如精确 contact force magnitude），RWM 没法直接 rollout reward。这限制了 reward design 灵活性。
4. **没有 explicit uncertainty**：所有 prediction 是 Gaussian mean/std，但 std 没被 policy 利用。如果 policy 能知道"哪里 model uncertain"就能 conservative。Ensemble methods (PETS [12]) 在这里有优势。
5. **Long-horizon 仍是相对的**：100 步在 50Hz 是 2 秒，对 locomotion 够，对 manipulation long task 可能不够。
6. **GPU memory 限制**：autoregressive training 的 N 受限于显存，4090 都跑不动 transformer。这限制了 architecture 选择。

## 12. 公开问题和未来工作方向

- **Uncertainty-aware RWM**: 用 ensemble 或者 Bayesian NN 让 model 输出 calibrated uncertainty，policy 在 high uncertainty region 保守
- **Real-world online learning**: 解决 collision recovery + safe exploration + privileged info estimation (contact force from onboard sensors)
- **Long-horizon tasks**: manipulation 里 multi-step task (pick → place → assemble) 需要 1000+ step rollout，RWM 能否扩展？
- **Multi-modal observations**: 加 vision？高维 observation 让 GRU 不够，可能要 VAE + latent dynamics
- **Hierarchical world model**: high-level (task) + low-level (dynamics) 分层
- **Cross-embodiment**: 一个 RWM 能不能 model 多种机器人？比如 ANYmal 和 G1 共享 backbone？
- **Foundation world model**: 类似 VideoGPT/Genie，pretrain on cross-robot data，fine-tune on specific robot

Web (Genie): https://arxiv.org/abs/2402.15391
Web (VideoGPT): https://arxiv.org/abs/2104.10157

## 13. 总结

RWM 的核心 insight 很 clean：**把 world model 训练时就让它经历 inference 时的 autoregressive rollout 分布，用 dual-autoregressive mechanism 同时学好 belief state estimation 和 self-feeding prediction**。这个简单的 training scheme 让 GRU 这种老架构就能学到能 rollout 100 步的 dynamics model，配合 PPO 在 imagination 里训出的 policy 能 zero-shot deploy 到 ANYmal 和 G1。

它不是想颠覆 Dreamer 或 TD-MPC，而是 fill 一个 gap：**怎么让 learned world model 真的能当 simulator 用**。这个 gap 之前没人可靠地 close 过。Paper 的 limitation 也很诚实——还比不过 model-free on perfect sim，real online learning 没做。但作为一个 milestone，它把 neural network simulator 从 toy 推到了 real robot deployment。

如果你的下一步是把这个推到 real-data online learning，我会关注三个方向：(1) uncertainty estimation 让 policy 不会 exploit model error, (2) safe exploration with collision recovery, (3) privileged info 的 sensor-based estimation。这三个里任何一个 breakthrough 都能让 RWM 真正脱离 sim 依赖。

---

## Reference Links

- Project page: https://sites.google.com/view/roboticworldmodel
- Dreamer: https://danijar.com/project/dreamer/
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://tdmpc2.com/ | https://arxiv.org/abs/2310.16828
- MBPO: https://arxiv.org/abs/1906.08253
- PlaNet: https://arxiv.org/abs/1811.04551
- PPO: https://arxiv.org/abs/1707.06347
- SAC: https://arxiv.org/abs/1801.01290
- DayDreamer: https://github.com/danijar/daydreamer
- Isaac Lab / Orbit: https://isaac-sim.github.io/IsaacLab/
- ANYmal: https://www.anybotics.com/
- Unitree G1: https://www.unitree.com/g1/
- SHAC: https://arxiv.org/abs/2204.07137
- Decision Transformer: https://arxiv.org/abs/2106.01345
- MuZero: https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
- PETS: https://arxiv.org/abs/1805.12114
- Schmidhuber world models: https://www.schmidhuber.de/sn/worldmodel.html
- Genie (foundation world model): https://arxiv.org/abs/2402.15391
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Diffusion for world modeling (DIAMOND): https://arxiv.org/abs/2405.12399
- STORM (stochastic transformer world model): https://arxiv.org/abs/2307.09991
