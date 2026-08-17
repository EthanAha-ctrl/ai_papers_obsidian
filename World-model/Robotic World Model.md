---
source_pdf: Robotic World Model.pdf
paper_sha256: 613d81eb472e437a76447f408f46ddeb0d786da29fae81e66089db1bfb1b7a14
processed_at: '2026-08-12T01:59:31-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说RWM

## 一句话版本

让机器人学会在"脑内"想象自己走路的完整过程, 而且连续想100步都不跑偏, 然后直接拿这个"脑内模拟器"当训练场练policy, 练完直接上真机跑。

---

## 为什么这事难

### Error accumulation 问题

想象你蒙眼走路, 每走一步在脑内预测"我现在在哪"。第一步预测可能差一点, 第二步基于第一步的错误预测继续推, 错误就放大了。走100步后你脑内的位置和实际位置差十万八千里。

这就是world model的老大难问题: **autoregressive prediction的compounding error**。你预测的第2步用了第1步的预测结果, 第3步用了第2步的预测结果...任何一个小错误都会snowball。

以前的MBPO [13] 最多rollout 5步就到极限了, 因为5步之后想象出的世界已经面目全非, policy从里面学不到东西。

### Train-test mismatch 问题

更subtle的问题: 以前的方法训练时用**真实数据**当输入, 测试时却用**自己的预测**当输入。这就像你只在晴天练车, 上路却全是雨天。

具体来说:
- **Teacher-forcing**: 训练时第 $t+1$ 步的输入是ground truth $o_t$, 预测目标 $o_{t+1}$
- **Inference**: 实际用时第 $t+1$ 步的输入是自己的预测 $o'_t$

model从来没见过"自己预测错误后的状态"长什么样, 一旦遇到就hallucinate。

参考: Scheduled sampling [Bengio et al. 2015] https://arxiv.org/abs/1506.03099

---

## RWM的核心insight

**训练时就让model见到自己预测错误的后果**。

用paper的语言: autoregressive training。用大白话: 让model在训练时就"踩自己的坑"。

具体做法:
- 给model 32步真实历史 (M=32)
- 让它往前预测8步 (N=8)
- 但这8步里, 每一步的输入都是**上一步自己的预测**, 不是ground truth
- Loss就是这8步预测和真实值的差距

这样训练完, model就知道"如果我上一步预测偏了, 下一步该怎么修正"。这就像你练车时专门在雨天练, 上路就不怕雨了。

公式 Eq.1 其实就是:
$$o'_{t+k} \sim p_\phi(\cdot | \text{真实历史}, \text{前k-1步的预测}, \text{actions})$$

变量含义:
- $o'_{t+k}$: 第k步的预测observation
- $p_\phi$: world model (参数为$\phi$)
- 竖线右边: condition的真实历史 + 之前的预测 + actions

---

## Dual-autoregressive 到底是啥

Paper说"dual-autoregressive", 听起来fancy, 其实就两层循环:

### Inner loop (处理历史)
GRU把32步历史一步一步吃进去, 更新hidden state:
$$h_{t-31} \rightarrow h_{t-30} \rightarrow ... \rightarrow h_t$$

这就是标准RNN处理sequence, 没啥特别的。

### Outer loop (预测未来)
从 $h_t$ 开始, 预测 $o'_{t+1}$, 然后把 $o'_{t+1}$ 喂回去, 预测 $o'_{t+2}$, 重复8次:
$$h_t \xrightarrow{\text{predict } o'_{t+1}} h_{t+1} \xrightarrow{\text{predict } o'_{t+2}} h_{t+2} \rightarrow ...$$

关键: outer loop里输入的是**预测值**, 不是真实值。这就是"踩自己的坑"的机制。

为什么叫"dual": 因为historical context和future prediction都用autoregressive方式处理, 但historical用真实数据, future用预测数据。

Fig S6 那张图就是画这个: 左边inner autoregression吃真实历史, 右边outer autoregression吃自己预测。

---

## 为什么 GRU 不用 Transformer

Paper里有个小细节值得注意: 他们试过Transformer, 但AR training时GPU memory爆了。

原因: AR training需要gradient通过N步预测回传。Transformer的self-attention每层都要存attention matrix, N步unroll后memory是 $O(N^2 \cdot d)$, N=8时还能扛, N大就崩。

GRU的memory是 $O(N \cdot d)$, linear in N, 所以能scale。

这也是为什么RWM用GRU而非Transformer的practical原因。不是Transformer不行, 是AR training + Transformer的memory cost太高。

参考 TWM (Transformer World Models) https://arxiv.org/abs/2209.00588 用teacher-forcing避免这个问题, 但就失去了AR training的好处。

---

## MBPO-PPO: 在imagination里练policy

### 基本流程

1. 真实机器人随便走走, 收集数据到replay buffer
2. 用这些数据训练RWM
3. 从replay buffer采样initial state, 在RWM里rollout 100步
4. 在这100步的"想象轨迹"上跑PPO更新policy
5. 新policy去真实环境收集更多数据
6. 重复

100步是关键数字。原版MBPO [13] 只rollout 5步, DreamerV3 [30] 也短。RWM能到100步, 因为AR training让world model在long horizon下stable。

### 为什么100步重要

PPO需要long horizon来estimate return:
$$G_t = \sum_{k=0}^{T} \gamma^k r_{t+k}$$

T太短, return estimate variance大, policy gradient noisy。T=100能让PPO看到足够远的未来, 学到因果性的policy。

但如果world model在100步里hallucinate, policy就会学到错误的行为。所以100步rollout的feasibility完全依赖于RWM的long-horizon stability。

参考 MBPO: https://arxiv.org/abs/1906.08253

---

## 实验里的关键数字

### Sample efficiency (Table 1)

| 方法 | 数据量 | 真机reward |
|------|--------|-----------|
| RWM + MBPO-PPO | 6M transitions | 0.90 ± 0.04 |
| PPO (model-free) | 250M transitions | 0.90 ± 0.03 |

42x的sample efficiency提升, 而且真机performance持平。这是paper最impressive的数字。

对real robot learning, 250M transitions意味着机器人要连续跑250M步, 按ANYmal 50Hz算就是5M秒 ≈ 58天不间断。6M transitions只要14万秒 ≈ 1.6天。差距巨大。

### Robustness to noise (Fig 3b)

给observation和action加Gaussian noise, 测试prediction error:
- MLP baseline: error随forecast step快速发散, noise越大越崩
- RWM: error保持低且stable, 即使高noise

为什么? GRU的hidden state作为memory, 对noisy input有smoothing effect。MLP每次prediction完全依赖当前input, noise直接pass through。

### Architecture对比 (Fig 4)

在manipulation + quadruped + humanoid上对比:
- MLP: 最差
- RSSM (Dreamer架构): 中等
- Transformer: 中等 (但AR training不scale)
- RWM-TF (teacher-forcing): 差
- RWM-AR (autoregressive): 最好

关键发现: RSSM如果也用AR training, 能match RWM-AR。说明AR training才是核心, 不是GRU架构多special。但GRU更简单efficient, 所以选GRU。

---

## Pretraining: 不能从零开始

Paper里一个honest的limitation: locomotion task必须pretraining。

为什么? 早期policy很差, 机器人一直摔。摔倒的data对world model训练没用, model学不好 → policy更差 → 摔更多 → 死循环。

Pretraining策略: 用其他similar task的policy数据先训练RWM, 让它有个decent initialization。然后online fine-tune。

关键: pretraining不需要optimal policy。Suboptimal policy的数据也行, 因为RWM对domain shift robust (Fig 3显示)。

Manipulation task不需要pretraining, 因为没摔倒termination, dynamics连续。

这个limitation说明: **从零开始online learning在real robot上仍然很难**。Paper的实验其实是simulation里online fine-tune, 不是真机online learning。

---

## Collision handling 的小trick

RWM的privileged info head预测contact。但paper还用它预测"termination": base接触地面 = 摔倒 = episode结束。

在imagination里, 如果RWM预测"要摔了", MBPO-PPO就terminate这个rollout。这样policy在imagination里就能学会"不要摔", 而不是等到真机才学。

这个设计很关键: 如果imagination里摔了不终止, policy可能学到"摔了也无所谓"的错误行为。

---

## 我的intuition

### 类比: 学车

Teacher-forcing像在驾驶模拟器里, 每次你都从"正确位置"开始练, 模拟器永远给你correct feedback。

AR training像真车上路, 你转错了弯, 下一秒就在错误的位置, 要从错误中recover。

后者更接近实际驾驶, 学到的skill更robust。

### 类比: LLM的autoregressive generation

RWM的AR training和LLM的next-token prediction本质相似:
- LLM: 预测下一个token, 喂回去预测下一个
- RWM: 预测下一个observation, 喂回去预测下一个

LLM也有compounding error问题 (hallucination), 但LLM的训练和inference都是autoregressive, 所以distribution aligned。

RWM把同样的insight用到robotics: 训练时也autoregressive, 和inference对齐。

这可能是paper最deep的insight: **world model应该像LLM一样训练, 而不是像传统sequence model用teacher-forcing**。

### 类比: Diffusion的iterative refinement

Diffusion model [Ho et al. 2020] https://arxiv.org/abs/2006.11239 通过iterative denoising生成图像。每一步都在refine上一步的输出。

RWM的AR training类似: model学会从自己的"noisy prediction"中refine出更好的prediction。这和diffusion的iterative refinement有philosophical connection。

区别: diffusion是spatial refinement, RWM是temporal refinement。

---

## Limitations 的honest评价

### 1. 仍未超越model-free on perfect sim

Table 1显示RWM和PPO的real reward持平 (0.90 vs 0.90)。但PPO在simulation里可能更高, 只是transfer到real有gap。RWM因为直接在learned model上训, 没有sim-to-real gap。

这说明: 如果你有perfect simulator, model-free RL仍然更好。RWM的优势在sim不准确或不存在时。

### 2. Real-world online learning仍难

Paper承认online learning时policy会exploit model error, 导致avg 20+次collision。真机不能这么摔。需要recovery policy、safety constraint、uncertainty estimation。

这其实是整个MBRL领域的open problem, RWM没解决, 只是逼近了。

### 3. Pretraining依赖

对truly novel task, 没有similar task数据怎么办? Paper没回答。可能需要meta-learning或large-scale pretraining across tasks (类似LLM的pretraining)。

### 4. Single Gaussian limitation

Gaussian prediction head无法capture multimodal dynamics。比如脚接触地面 vs 悬空是两个mode, Gaussian会blur。Ensemble或diffusion可能更好。

参考: Diffusion world models https://arxiv.org/abs/2405.12399

---

## 对未来的启发

### World model scaling

如果AR training是关键, 那RWM可以scale吗? 更大的GRU、更长的M和N、更多data, 能达到什么limit?

类似LLM的scaling law, world model可能也有scaling law。但这paper没探索, 只是single robot single task。

### Foundation world model

一个world model跨多个robot (ANYmal, G1, 甚至manipulator)? Paper显示RWM在多种robot上work, 但每个robot单独训练。

如果pretrain一个large world model on all robot data, 然后fine-tune到specific robot, 可能达到更好的sample efficiency。类似GPT for robotics。

### Connection to video prediction

RWM预测state-space observations, 不是pixels。但如果结合video prediction (Sora-style https://openai.com/sora/), 可能学更rich的world model。

难点: pixel prediction的long-horizon stability更难, 而且computational cost高。但AR training的insight应该transfer。

---

## 总结: 三个key takeaways

1. **AR training解决train-test mismatch**: 训练时让model见自己的prediction errors, 学会self-correct, 实现long-horizon stable prediction

2. **100-step imagination rollout可行**: 因为RWM的stability, MBPO-PPO能在imagination里rollout 100步, 远超之前方法的5步

3. **42x sample efficiency + zero-shot real deployment**: 6M vs 250M transitions, ANYmal D和G1上zero-shot deploy

核心philosophy: **train on the distribution you'll test on**。这个intuition简单但powerful, 和domain randomization、adversarial training、self-supervised learning的core idea一脉相承。

---

# Robotic World Model (RWM) 论文详解

这篇paper来自ETH Zurich的Chenhao Li, Andreas Krause, Marco Hutter组, 发表在2025年, 探索model-based reinforcement learning (MBRL) 在真实机器人上的应用。核心贡献: 提出一个general的world model框架, 通过dual-autoregressive training机制, 实现长horizon stable prediction, 并配合MBPO-PPO在ANYmal D quadruped和Unitree G1 humanoid上实现zero-shot硬件部署。

---

## 1. 核心Problem和Motivation

### 1.1 问题定义
机器人控制常用model-free RL (PPO, SAC), 在仿真中效果出色, 但是sample efficiency差, 真实机器人上无法直接应用。World model是解决sample efficiency的natural方案, 通过learned dynamics model模拟环境, 实现policy在imagination中训练 (learning in imagination, Sutton的Dyna思想 [16])。

现有world model方法存在几个关键问题:
- **Long-horizon error accumulation**: autoregressive prediction下, 一步error会compound到future steps, 导致rollout发散
- **Partial observability**: 真实机器人观测存在contact discontinuities, sensor noise, 无法完全observability
- **Sim-to-real gap**: learned model可能overfit training distribution, 在real deployment时失效
- **Domain-specific inductive biases**: 现有方法需要foot-placement dynamics [21], Lagrangian structure [28], rigid body dynamics [20]等handcrafted priors, 限制generalization

### 1.2 RWM的设计哲学
RWM的核心insight: **训练时让model见到自己predictions的distribution**, 而不是只在teacher-forcing下训练。这是处理long-horizon compounding error的关键。

参考链接:
- World Models original: https://worldmodels.github.io/
- DreamerV3: https://danijar.com/project/dreamerv3/
- MBPO: https://bair.berkeley.edu/blog/2019/12/09/mbpo/

---

## 2. RWM Architecture详解

### 2.1 Dual-autoregressive机制 (核心创新)

RWM的精髓在于**双层autoregression**, 这与传统的teacher-forcing训练有本质区别:

**Inner autoregression (处理历史context M)**:
- 给定M个历史observations $o_{t-M+1:t}$和actions $a_{t-M+1:t}$
- GRU hidden state $h_t$按时间步sequentially更新:
  $$h_{t-m+1} \rightarrow h_{t-m+2} \rightarrow ... \rightarrow h_t$$
- 这部分类似标准RNN处理sequence

**Outer autoregression (处理forecast horizon N)**:
- 在forecast阶段, model用自己的predictions作为下一步的input
- 公式 Eq.1:
  $$o'_{t+k} \sim p_\phi(\cdot | o_{t-M+k:t}, o'_{t+1:t+k-1}, a_{t-M+k:t+k-1})$$
  - $o_{t-M+k:t}$: 真实历史observations
  - $o'_{t+1:t+k-1}$: 之前predicted的observations (自己生成的)
  - $a_{t-M+k:t+k-1}$: 历史和当前actions

**为什么这个设计重要**: training时model就经历自己的prediction errors, 学会self-correct, 减少train-test distribution mismatch。Teacher-forcing下, model只见到clean ground truth observations, test时遇到自己的noisy predictions就hallucinate。

### 2.2 网络结构 (Table S7)
```
base: GRU, hidden [256, 256]
heads: MLP, hidden 128, ReLU activation
       - predicts mean μ(o_{t+1})
       - predicts std σ(o_{t+1}) 
       - predicts mean μ(c_{t+1}) for privileged info
       - predicts std σ(c_{t+1})
```

预测是Gaussian distribution, 通过reparameterization trick实现end-to-end gradient flow:
$$o'_{t+k} = \mu_{t+k} + \sigma_{t+k} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 2.3 训练Loss (Eq.2)
$$\mathcal{L} = \frac{1}{N} \sum_{k=1}^{N} \alpha^k \left[ L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k}) \right]$$

变量解析:
- $N$: forecast horizon (论文用N=8)
- $\alpha$: decay factor (论文设为1.0, 即no decay)
- $L_o$: observation prediction loss (likely MSE)
- $L_c$: privileged info prediction loss (contacts, foot height等)
- 加privileged info作为auxiliary objective, 帮助implicit representation learning

### 2.4 为什么GRU而非Transformer?
论文提到一个key practical consideration: Transformer + autoregressive training存在GPU memory bottleneck。Multi-step gradient propagation through N forecast steps需要存储intermediate activations, Transformer的attention mechanism让这个成本exploding。GRU更compact, 可以scale到N=8甚至更大。

参考:
- Decision Transformer: https://sites.google.com/view/decision-transformer
- TWM (Transformer World Models): https://arxiv.org/abs/2209.00588

---

## 3. MBPO-PPO Policy Optimization

### 3.1 算法框架 (Algorithm 1)
RWM的policy optimization基于MBPO (Janner et al. 2019) [13] + Dyna架构 [42]:

```
1. 用policy π_θ与真实环境交互, 收集(o, a, r, o')到replay buffer D
2. 从D采样batches, 用Eq.2的autoregressive loss训练world model p_φ
3. 从D采样initial states, 初始化imagination agents
4. 在imagination中rollout T步:
   for k in 1 to T:
       a'_{t+k} ~ π_θ(· | o'_{t+k})  (Eq.3)
       o'_{t+k+1} ~ p_φ(· | history, predictions, actions)
       r'_{t+k} = R(o'_{t+k}, a'_{t+k}, o'_{t+k+1})
5. 用PPO updates在imagination trajectories上更新π_θ
6. 重复
```

关键超参数 (Table S11):
- Imagination environments: 4096 (并行)
- Imagination steps per iteration: 100 (T=100, 远超MBPO原论文的5步!)
- Buffer size: 1000
- Iterations: 2500
- dt: 0.02s (50Hz control)

### 3.2 为什么能rollout 100步?
这是RWM的关键breakthrough。原版MBPO只能rollout 1-5步, 因为长horizon下compounding errors让policy gradient信号noise极大。RWM通过autoregressive training, 让world model在长horizon下保持stable, 从而支持100-step rollout。

Paper的实验数据显示: 在model error对比中 (Fig. 5), MBPO-PPO的model error随training稳定下降, 而SHAC (first-order gradient method) 维持高error, DreamerV3存在moderate compounding errors。

### 3.3 Reward function设计 (Appendix A.1.2)
ANYmal D和Unitree G1用velocity tracking reward:

Linear velocity tracking (xy):
$$r_{v_{xy}} = w_{v_{xy}} \cdot e^{-\|c_{xy} - v_{xy}\|_2^2 / \sigma_{v_{xy}}^2}$$
- $w_{v_{xy}} = 1.0$: weight
- $c_{xy}$: commanded velocity
- $v_{xy}$: actual base velocity
- $\sigma_{v_{xy}} = 0.25$: temperature (类似exp-kernel)

Angular velocity tracking (z):
$$r_{\omega_z} = w_{\omega_z} \cdot e^{-\|c_z - \omega_z\|_2^2 / \sigma_{\omega_z}^2}, \quad w_{\omega_z} = 0.5$$

Penalty terms:
- Vertical velocity: $r_{v_z} = w_{v_z} \|v_z\|_2^2, \quad w_{v_z} = -2.0$
- Roll/pitch velocity: $r_{\omega_{xy}} = w_{\omega_{xy}} \|\omega_{xy}\|_2^2, \quad w_{\omega_{xy}} = -0.05$
- Joint torque: $r_{q_\tau} = w_{q_\tau} \|\tau\|_2^2, \quad w_{q_\tau} = -2.5e^{-5}$
- Action rate: $r_{\dot{a}} = w_{\dot{a}} \|a' - a\|_2^2, \quad w_{\dot{a}} = -0.01$

Reward预测: 从imagination的observations中计算reward (假设reward function是known的, 这与Dreamer类似, 与model-free RL的implicit reward不同)。

---

## 4. 实验结果深度分析

### 4.1 Autoregressive Prediction Accuracy (Sec 4.1)
实验设置:
- ANYmal D hardware数据
- 50Hz control, M=32, N=8
- Visualize在Fig 3a

结果显示RWM的predicted trajectories与ground truth高度align, 即使超过training forecast horizon N=8, 仍保持稳定prediction。这是autoregressive training带来的generalization: model学会了long-horizon self-correction, 即使训练时只见到8步, inference时能extend到更长。

### 4.2 Robustness under Noise (Sec 4.2)
关键实验: 在observations和actions上注入Gaussian noise, 测试RWM vs MLP baseline (都autoregressively trained)。

Results (Fig 3b):
- MLP error随forecast steps快速增长, 在noise下diverge明显
- RWM保持低error, 即使high noise levels
- Relative prediction error $e$ across forecast steps, yellow curves (RWM)远低于grey curves (MLP)

**Insight**: GRU的hidden state作为memory buffer, smoothing noisy inputs。MLP没有memory, 每次prediction完全依赖当前input, noise直接传递。这从architecture层面解释了RWM的robustness。

### 4.3 Generality across Environments (Sec 4.3)
对比baselines (Fig 4):
- **MLP**: 2-layer, hidden 256, ReLU
- **RSSM** (Dreamer系列): GRU base, latent dim 64, categorical 32 categories
- **Transformer**: decoder, dim 64, 8 heads, 2 layers, context length 32
- **RWM-TF**: RWM architecture但teacher-forcing training (N=1)
- **RWM-AR**: RWM architecture + autoregressive training (N=8)

Tasks: manipulation + quadruped locomotion + humanoid locomotion

Key findings:
1. RWM-AR在所有tasks上achieve lowest prediction error
2. RWM-AR >> RWM-TF, 证明autoregressive training的critical role
3. RSSM + AR training能match RWM-AR (论文承认), 但GRU更简单
4. Transformer + AR training不scale (memory issues)

### 4.4 Policy Learning and Hardware Transfer (Sec 4.4)
对比MBPO-PPO vs SHAC vs DreamerV3 (Fig 5):

**MBPO-PPO**:
- Model error稳定下降
- Predicted reward初期overshoot (policy exploit model optimism), 后期align ground truth
- 最终high reward, 稳定converge

**SHAC** (first-order gradient through differentiable sim):
- High fluctuating model error
- Discontinuous dynamics (legged locomotion的contact)让gradient inaccurate
- Chaotic robot behaviors → 低质量data → 加剧model error (vicious cycle)

**DreamerV3**:
- Short planning horizons限制长horizon dependency
- Moderate compounding errors
- Partial convergence, 不如MBPO-PPO

**Zero-shot hardware deployment**:
- ANYmal D和Unitree G1上成功
- 跟踪velocity commands, 抵抗external disturbances (impacts, terrain)
- SHAC和DreamerV3的policy在training时collapse, 无法deploy

### 4.5 与Model-Free PPO对比 (Table 1)
Critical comparison:

| Method | RWM pretraining | MBPO-PPO | PPO (model-free) |
|--------|-----------------|-----------|------------------|
| State transitions | 6M | - | 250M |
| Total training time | 50 min | 5 min | 10 min |
| Step inference time | - | 1 ms | 1 ms |
| Real tracking reward | - | 0.90 ± 0.04 | 0.90 ± 0.03 |

**Key insight**: RWM + MBPO-PPO用6M transitions (vs PPO的250M), achieve comparable real-world performance (0.90 vs 0.90)。这是~42x的sample efficiency提升, 对real robot learning意义重大。

但paper也承认limitation: 仍未超越well-tuned model-free RL on high-fidelity sim, MBRL的strength在sim不可用或不准确的场景。

---

## 5. Ablation Studies深入分析

### 5.1 History vs Forecast Horizon (Fig S8)
Heatmap分析不同M和N的组合:
- **M (history)**: 增大M降低prediction error, 但plateaus after certain point。M=32足够capture历史dynamics
- **N (forecast)**: 增大N显著改善long-horizon accuracy, 因为model训练时expose到extended rollouts, 学会handle compounding errors
- **Training time**: N增大让训练时间显著增长 (sequential computation), N=1 (teacher-forcing)最快但accuracy差

**Optimal trade-off**: M=32, N=8, balance accuracy和training cost

### 5.2 Pretraining的必要性 (Sec A.4.3)
对于locomotion tasks, pretraining是critical:
1. **Online dataset limited**: 单环境data量小, 直接train from scratch会overfit
2. **Immature policy导致fall**: 早期policy频繁失败, 生成低质量transitions
3. **Domain shift**: 只用chaotic data训练, imagination rollouts也chaotic → poor policy updates

Pretraining策略:
- 用similar tasks under varied dynamics的policy数据
- 不需要optimal policy, suboptimal即可 (Fig 3显示RWM robust to domain shifts)
- Manipulation tasks不需要pretraining (continuous dynamics, 无collision termination)

### 5.3 Collision Handling (Sec A.4.3)
关键设计: RWM explicitly预测terminations:
- Base contact = failure → terminate rollout
- Privileged info head预测这些terminations
- MBPO-PPO在imagination中treat predicted terminations作为episode-ending events
- 影响PPO return computation和state values

这让world model学会unsafe transitions, policy在imagination中就能learn avoid collisions。

---

## 6. 与相关工作的深度对比

### 6.1 vs Dreamer系列
- Dreamer: latent space dynamics + actor-critic, short horizons
- RWM: observation space dynamics + MBPO-PPO, long horizons (100 steps)
- Dreamer的RSSM用categorical latent, RWM直接predict observations
- RWM paper显示RSSM + AR training也能match, 但GRU更simple

### 6.2 vs TD-MPC
- TD-MPC: latent space + MPC, online planning
- RWM: observation space + MBPO-style rollout, policy learning
- TD-MPC2在continuous control上strong, 但需要MPC planning inference
- RWM的policy是reactive (1ms inference), 更适合real-time control

### 6.3 vs MBPO (original)
- MBPO: 1-5 step rollout (短horizon避免compounding errors)
- RWM: 100 step rollout (autoregressive training使长horizon stable)
- MBPO需要uncertainty quantification (ensemble), RWM单model即可

### 6.4 vs Visual Foresight
- Visual foresight: 视觉prediction + planning
- RWM: state-space prediction + RL
- RWM更适合作low-level control (50Hz), visual prediction通常用于high-level planning

### 6.5 vs Differentiable Simulation (SHAC)
- SHAC: gradient through physics engine
- RWM: learned neural network simulator
- Differentiable sim对discontinuous dynamics (contact)的gradient inaccurate
- RWM通过data-driven learning, 隐式capture contact dynamics

---

## 7. 关键技术细节和Insights

### 7.1 为什么Autoregressive Training有效?
**Train-test distribution alignment**: 
- Teacher-forcing: train on $p(o_{t+1} | o_{t:GT})$, test on $p(o_{t+1} | o_{t:pred})$, mismatch
- AR training: train和test都on $p(o_{t+1} | o_{t:pred})$, aligned

**Error correction learning**:
- 当model预测错误, 下一步见到自己的error, 需要learn correct it
- 这就像human learning from mistakes, 比只看correct examples更robust

**Connection to scheduled sampling**: 
- Scheduled sampling gradually从teacher-forcing切换到AR
- RWM直接用full AR training, 更彻底

### 7.2 Partial Observability的处理
不用explicit belief state (Bayesian filter), 而是用GRU的memory:
- M=32步历史隐式encode unobservable state
- Contact events通过privileged info prediction学习
- 这比explicit POMDP inference更简单, 但需要足够长history

### 7.3 Stochastic Dynamics
预测Gaussian distribution (mean + std):
- 比 deterministic prediction更robust to multimodal dynamics
- 没有用ensemble (vs PETS), 单model即可
- 可能limitation: 单Gaussian无法capture multimodal (如contact/no-contact两个mode)

### 7.4 Reward in Imagination
Policy训练时, reward从predicted observations计算:
- Reward function assumed known (domain knowledge)
- 与Dreamer不同 (Dreamer learn reward predictor)
- 限制: 需要explicit reward function, 无法处理reward from human feedback

---

## 8. Limitations和Future Directions

### 8.1 论文承认的limitation
1. **仍不及model-free on high-fidelity sim**: 0.90 vs 0.90, 但model-free用42x more data
2. **Pretraining needed for locomotion**: domain shift和discontinuous dynamics
3. **Real-world online learning困难**: 
   - Policy exploit model errors → collisions (avg 20+ failures during online learning)
   - 需要recovery policy reset robot (对ANYmal/G1困难)
   - Privileged info (contact forces)需要sensor measurement
4. **Computational cost**: AR trainingsequential, 比teacher-forcing慢

### 8.2 Future work方向
1. **Uncertainty-aware world models**: 处理model errors exploitation
2. **Safe online learning on hardware**: 结合recovery policy和safety constraints
3. **Larger forecast horizons**: 更长N可能进一步改善long-horizon prediction
4. **Multi-modal predictions**: 用diffusion或mixture density处理multimodal dynamics
5. **Latent space RWM**: 结合RWM的AR training和latent compression

---

## 9. 我的思考和Critical Analysis

### 9.1 强项
1. **Engineering细节扎实**: 50Hz control, real hardware, zero-shot transfer
2. **Ablation thorough**: M, N, AR vs TF, architecture对比
3. **Practical insights**: pretraining, collision handling, online learning challenges
4. **Honest limitations**: 承认不及model-free, real-world online learning困难

### 9.2 潜在concerns
1. **Pretraining依赖**: 虽然paper说不需要optimal policy, 但仍需要similar task data。对truly novel task如何处理?
2. **Single Gaussian limitation**: contact/no-contact是bimodal, Gaussian head可能blur这些mode
3. **Reward function known**: 不适合reward from demonstrations或human feedback场景
4. **Long-horizon stability的mechanism**: paper显示结果, 但缺乏theoretical explanation
5. **Generalization beyond locomotion**: manipulation实验相对simple, complex manipulation未验证

### 9.3 与Karpathy的intuition联系
从neural network角度, RWM的success可以联系到几个deep learning principles:
1. **Train-test distribution alignment**: 类似domain adaptation, AR training让model见到test-time distribution
2. **Long-context modeling**: GRU的memory处理partial observability, 类似transformer的context window
3. **Self-supervised learning**: 预测未来observations是自监督signal, 无需external labels
4. **Iterative refinement**: model学会从自己predictions中recover, 类似diffusion的iterative denoising

### 9.4 与LLM/Scaling Laws的联系
RWM的long-horizon prediction类似LLM的long-context generation:
- AR training类似autoregressive language modeling
- Error accumulation类似LLM的hallucination
- Long-context (M=32)类似context window
- 未来可能用transformer + efficient AR training (如flash attention) scale到更大N

---

## 10. 关键参考链接

**核心paper和相关工作**:
- RWM project page: https://sites.google.com/view/roboticworldmodel
- World Models (Ha & Schmidhuber): https://worldmodels.github.io/
- DreamerV3: https://danijar.com/project/dreamerv3/
- MBPO: https://bair.berkeley.edu/blog/2019/12/09/mbpo/
- PlaNet: https://ai.googleblog.com/2019/02/introducing-planet-deep-learning.html
- TD-MPC2: https://tdmpc2.com/

**代码和simulation**:
- Isaac Lab (Orbit): https://isaac-sim.github.io/IsaacLab/
- ANYmal: https://www.anybotics.com/
- Unitree G1: https://www.unitree.com/g1/

**Baselines**:
- SHAC: https://arxiv.org/abs/2204.07137
- Decision Transformer: https://sites.google.com/view/decision-transformer
- PETS: https://arxiv.org/abs/1805.12114

**理论背景**:
- Dyna architecture: Sutton 1991
- POMDP: https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
- Scheduled sampling: https://arxiv.org/abs/1506.03099

---

## 总结

RWM的核心贡献是用dual-autoregressive training机制, 让world model在long-horizon prediction下保持stable, 从而支持100-step MBPO-PPO rollout。这在MBRL领域是significant advance, 因为compounding error一直是long-horizon planning的fundamental obstacle。

从engineering角度, paper展示了从simulation到real hardware的完整pipeline, 包括pretraining, online fine-tuning, collision handling, zero-shot deployment。Sample efficiency的42x提升对real robot learning有practical impact。

从research角度, paper启发几个directions:
1. AR training的theoretical analysis (为什么能learn self-correction?)
2. Latent space + AR training的结合
3. Multi-modal prediction (diffusion + AR)
4. Uncertainty quantification for safe online learning

RWM的philosophy: **让model在training时经历test-time distribution**, 这个intuition对general ML也有启发, 类似domain randomization, adversarial training, self-supervised learning的core idea: train on the distribution you'll test on。
