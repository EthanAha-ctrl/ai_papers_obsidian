---
source_pdf: A Survey of Sim-to-Real Methods in RL Progress, Prospects and Challenges
  with Foundation Models.pdf
paper_sha256: 0f2d3f7ee4aabdc85fb7d74f8d33c0584a70c7a903415dd191c9c2a13530755a
processed_at: '2026-08-17T23:32:10-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 这 paper 到底在搞啥

sim-to-real 这个事儿其实特别朴素: 你在电脑模拟器里训了个 robot, 它在 sim 里各种秀操作, 然后你满怀期待把它搬到真实世界——然后它就傻了。

这事让搞 RL 的人特别头疼。你想啊, 训练 robot 在 sim 里摔倒一万次没事, 但在真实世界摔倒一次可能就是几千刀的维修费, 自动驾驶撞一次可能就是人命。

所以问题来了: **能不能让 sim 里学到的本事, 直接迁移到真实世界?**

这篇 paper 干的事情很 simple: 把这个领域乱七八糟几百篇 paper 理了理, 给出一个还挺好用的 mental model。

## 核心的 mental model: 四个 Gap

paper 最 elegant 的地方在于, 它说 sim-to-real 这个模糊的概念, 其实可以拆成四种具体的"对不上":

### 1. Observation Gap — 眼睛对不上

sim 里的 robot 是开了上帝视角的, 什么都能看清楚, $o_t^{sim} = s_t^{sim}$, 感知完美。

real 里的 robot 像个戴老花镜还举着手电筒的老爷爷, 传感器有 noise, 有 motion blur, 光照变化, 有遮挡, $o_t^{real} \neq s_t^{real}$, 这是 POMDP。

### 2. Action Gap — 手对不上

sim 里你说"前进", robot 立马前进; real 里你说"前进", 信号要传到 motor, motor 要响应, mechanical 有 latency。

sim 里 action 往往 discretized 成 "go forward 1 unit" 这种; real 里你给的是 continuous 的 joint torque 值。粒度也对不上。

### 3. Transition Gap — 物理对不上

这是最 fundamental 的。sim 里的 friction coefficient 是你手写的 0.5, real 里的 friction 随温度湿度磨损变化。同样的 state $s_t$ 和 action $a_t$, sim 给出的 $s_{t+1}^{sim}$ 和 real 给出的 $s_{t+1}^{real}$ 不一样。

### 4. Reward Gap — 反馈对不上

你设计的 reward function 是基于对 sim 的理解, 但 sim 不完整。你在 sim 里奖励"快速到达目标", real 里可能 robot 为了快而把人撞了。

---

**Intuition**: 这四个 gap 是 sim-to-real 的 root cause。任何 method 都是在堵某一个或几个 gap。

## 各种方法到底在干嘛

### Domain Randomization — 见多识广派

思路特别直觉: 既然 sim 和 real 总有差异, 那 sim 里就别老老实实, 把 texture、光照、friction、camera 位置全都随机化, 让 agent 见过一百万种"乱七八糟的世界"。

等它部署到 real, real 就是这一百万种里的某一种, 它就 generalizes 了。

OpenAI 搞 Rubik's Cube 那篇 (https://arxiv.org/abs/1910.07113) 就是用 Automatic Domain Randomization, 还加了 curriculum——一开始 randomize 轻一点, policy 学好了再加大难度, 避免一开始就太混乱导致训练崩掉。

### Domain Adaptation — 对齐派

这个派别的思路是: 让 sim 和 real 的 feature 分布对齐。具体做法类似 GAN, 用一个 discriminator 区分"这图像来自 sim 还是 real", encoder 就使劲学, 学到 discriminator 区分不出来为止。

RetinaGAN (https://arxiv.org/abs/2102.07973) 就是这个思路。

VR-Goggles (https://arxiv.org/abs/1902.07352) 反过来想: 既然 sim→real 适配贵, 那部署时把 real image 翻译成 sim 风格, policy 不用改就行。聪明。

### Grounding — 撒谎派

这个我觉得最有意思。Hanna & Stone 的 GAT (https://arxiv.org/abs/1610.07428) 思路是: 既然 sim 的物理和 real 不一样, 那让 sim 学会"撒谎"。

具体来说, policy 说要执行 action $a$, sim 不直接执行 $a$, 而是执行一个 transformed 的 $f(a)$。这个 $f(a)$ 在 sim 物理下产生的结果, 等价于 $a$ 在 real 物理下产生的结果。

Policy 在这个"会撒谎的 sim"里学到的策略, 拿到 real 里直接就 work——因为 sim 的谎言恰好补偿了物理 gap。

后面衍生出一堆变体: SGAT 把它变 stochastic, RGAT 用 RL 端到端学, GARAT 用 GAN, 还有人加 uncertainty quantification。但核心 idea 都是"让 sim 学会撒一个好谎"。

### Sensor Fusion — 多眼看世界派

一个 sensor 在 real 里总有 failure mode, 但多个 sensor 的 failure mode 不相关, 融合起来就 robust。比如 visual + depth + LiDAR, 谁看不清的时候别人补上。

### Distributionally Robust Learning — 数学党

这派思路: 既然我不知道 real dynamics 具体是啥, 那就假设它在 sim dynamics 的某个"邻域"里, 然后学一个对整个邻域都 robust 的 policy。

数学上是 $\max_\pi \min_{P \in \mathcal{U}} \mathbb{E}_P[\sum \gamma^t r_t]$, 其中 $\mathcal{U}$ 是 uncertainty set。

这个方向的好处是有 theoretical guarantee, 比如 Liu & Xu (https://arxiv.org/abs/2402.02192) 给了 provable efficiency。

### Action Delay 处理

这部分 paper 写得特别细, 因为 sim 假设 action 立刻生效, 但 real 有 latency。

两类解法:
1. **State augmentation**: 把历史 action 塞进 state 里, 但复杂度爆炸
2. **Forward prediction**: 学一个 forward model, 根据历史 action 推断"如果没有 delay 现在应该是什么 state"

DCAC (https://arxiv.org/abs/2010.07384)、PAD、D-SARSA 各有各的招, 但本质都是某种 prediction-based compensation。

## Foundation Models 为啥有意思

paper 的新意在于把 foundation models 嵌进这四个 axis 的每一个。我觉得真正有价值的是两点:

### 1. 语言作为 semantic anchor

"Pick up the red cup" 这句话在 sim 和 real 里语义相同, 不管 pixel-level 长得多不一样。语言天然是一个 abstraction layer, filter 掉了 low-level 的 visual difference。

所以用 LLM/VLM 给 observation 加一个 language-based description, policy 学这个 description 而非 raw pixel, robustness 大幅提升。

Yu et al. (https://arxiv.org/abs/2405.10020) 就在搞这个, 叫 "semantic anchors"。

### 2. LLM 的 world knowledge 补 sim 的盲区

sim 是 hand-engineered, 设计者没想到的 scenario sim 里就没有。LLM 预训练过海量 corpus, 它的 world knowledge 可以覆盖 sim 没见过的 cases。

这在 reward design 上特别有用。CARD (https://arxiv.org/abs/2410.14660) 和 Text2Reward (https://arxiv.org/abs/2309.11489) 都是用 LLM 从自然语言生成 reward function code, 减少人的 bias。

Da et al. 的 Prompt to Transfer (https://arxiv.org/abs/2401.06256) 更直接: 用 LLM 推理能力增强 forward model 对 real dynamics 的 prediction, 然后 grounding 方法就更准了。

## 几个 Domain 的特点

paper 还讨论了不同 domain 的特殊挑战, 这个我觉得对 build intuition 很有用:

**Robotics**: 最关心 safety 和 accuracy。一次失误就是物理损坏, 所以 shielding、safety constraint 很重要。

**Transportation**: 多 agent 协调、real-time decision、smooth transition。一个城市几万 traffic light 互相影响, 这是 combinatorial 问题。SUMO、CARLA、CityFlow 是主流 sim。

**Recommender Systems**: 这是最 tricky 的。用户行为模拟器本身就是 sim-to-real 的核心难题——你没法精确 simulate 用户。所以 off-policy evaluation 和 counterfactual analysis 是关键。

## Evaluation 的三档

paper 给出三种 evaluation, 我觉得这个三维图 (cost/safety/realism) 很有 intuition:

1. **Sim-to-Real**: 直接上真实硬件。Realism 最高, 但贵且危险。
2. **Sim-to-Scale-down-Real**: 在受控的缩小版真实环境测。比如 Duckietown 这种小车道。折中。
3. **Sim-to-Sim**: 在另一个 sim 里测。最便宜最安全, 但 realism 最差。

实际上大家都想做 sim-to-sim 然后预测 sim-to-real 表现, 但这个预测性还是个 open problem, largely depends on 选什么 metric。

## 我的 Take

读完这篇, 我有几个直觉:

**1. 这个 taxonomy 真 useful**

之前看 sim-to-real 的 paper 总觉得东一榔头西一棒, 这篇用 MDP 四元素一归类, 顿时清晰: 任何方法都是堵一个 gap, 组合使用效果最好。比如 observation 用 DR, transition 用 grounding, reward 用 LLM shaping。

**2. Grounding 是被低估的方向**

让 sim 学会"撒谎"补偿 physics gap, 这个 idea 本身就 elegant。而且它直接 attack 最 fundamental 的 transition gap, 不是观察层面的 patch。

**3. Foundation model 真正的价值是 abstraction layer**

不是 LLM 能当 reward designer 或者 forward model 这么表面, 而是 language 提供了一个 sim/real 共享的 semantic space, 这个 space 天然 robust to low-level differences。

**4. 终极解法可能是 generative world models**

NVIDIA COSMOS 和 Genesis 这种 generative physics engine, 如果能 data-driven 地学到足够逼真的物理, 那 gap 从根上就消失了, 不需要各种 patch。这是我最看好的方向。

**5. Evaluation 仍是 fundamental problem**

我们没有 theoretical bound 说 sim-sim 表现好就 sim-real 表现好。这是整个领域的 fundamental uncertainty, paper 也承认这点。

---

总之这篇 survey 值得读, 主要价值是那个 MDP taxonomy——以后你看任何 sim-to-real paper, 都能套进 S/A/T/R 四个框里, 瞬间就明白它在 attack 什么 gap, 用什么思路。这比记住几百个 method 名字有用得多。

---

# A Survey of Sim-to-Real Methods in RL: 深度讲解

Andrej, 这篇 paper 我觉得本质上是在做一件很优雅的事情——它把 Sim-to-Real 这个看似混乱的研究领域, 通过 MDP 的四个元素 (S, A, T, R) 重新组织成一个 taxonomy, 同时把 foundation models 的影响也嵌入到这个框架里。让我从底层逻辑开始, 帮你 build intuition。

## 1. Paper 的定位和核心贡献

这篇 survey 来自 Arizona State University 和 DARPA, 作者包括 Longchao Da, Hua Wei 等。它的核心 positioning 在 Table 1 里很清楚地体现——它是第一个同时具备以下特征的 survey:
- **Methodological Taxonomy on MDP**: 把所有 sim-to-real 方法映射到 MDP 的四个元素上
- **Foundation Models 整合**: 把 LLM/VLM 如何赋能 sim-to-real 系统性纳入
- **Formal Evaluation Metrics**: 给出 G(π) 的形式化定义
- **Domain-Specific Analysis**: 区分 Robotics, Transportation, Recommender Systems 的不同 challenges

paper 的 repository: https://github.com/ASU-DA-Lab/Sim2Real-RL

## 2. Sim-to-Real 问题的数学形式化

### 2.1 MDP 基础回顾

paper 用标准的 MDP 框架:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$$

变量含义:
- $\mathcal{S}$: state space, 所有可能的环境状态
- $\mathcal{A}$: action space, agent 可执行的所有动作
- $\mathcal{T}$: transition function, $\mathcal{T}(s_{t+1}|s_t, a_t)$, 给定当前 state $s_t$ 和 action $a_t$ 时下一状态 $s_{t+1}$ 的概率分布, 定义在 $\mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ 上
- $\mathcal{R}$: reward function, 反馈信号
- $\gamma \in [0,1)$: discount factor, 决定 future reward 的重要性

RL 的目标是最大化 expected return:

$$J(\pi) := \mathbb{E}_{(s,a) \sim \mu^{\pi}(s,a)} \left[ \sum_t \gamma^t r_t(s_t, a_t, s_{t+1} \sim p(s_t, a_t)) \right] \quad (1)$$

其中:
- $\mu^{\pi}(s,a)$: policy $\pi$ 下的 stationary state-action distribution (稳态分布)
- $a_t = \pi(s_t)$: policy 把 state 映射到 action
- $p(s_t, a_t) = \mathcal{T}(s_{t+1}|s_t, a)$: 这其实就是 transition, 只是符号变换

### 2.2 Sim-to-Real Gap 的形式化

这是 paper 里我觉得最 elegant 的部分, 它把 gap 定义为一个 evaluation metric 的差值:

$$G(\pi) := \psi_s(\pi_s^i) - \psi_r(\pi_s^i) \mid \pi_s^i \sim \mathcal{M}_s \quad (2)$$

变量解读:
- $G(\pi)$: policy $\pi$ 的 sim-to-real gap (一个标量)
- $\psi_s$: simulator 中的 evaluation metric (比如 success rate)
- $\psi_r$: real world 中的同一 evaluation metric (必须 calibrated, 用 identical 方式应用)
- $\pi_s^i$: 在 simulator $\mathcal{M}_s$ 中学到的 policy, 上标 $i$ 表示第 $i$ 次训练 instance
- $\mathcal{M}_s$: simulator 对应的 MDP
- 隐含的 $\mathcal{M}_r$: real world 对应的 MDP

**Intuition**: 这个公式说, 一个在 sim 里表现 $\psi_s = 0.9$ 的 policy, 如果在 real 里只有 $\psi_r = 0.4$, 那么 gap 就是 0.5。Gap 越大, transfer 越失败。这是一个很干净的 formalization, 因为它把"sim-to-real problem"从模糊的概念变成了可量化的对象。

### 2.3 Cumulative Reward Difference

paper 还给出了 trajectory-level 的 gap 度量:

$$\Delta R = \sum_{t=1}^{T} R_t^{\text{sim}} - \sum_{t=1}^{T} R_t^{\text{real}} \quad (3)$$

变量解读:
- $\Delta R$: 累积 reward 差异 (标量)
- $R_t^{\text{sim}}$: time step $t$ 在 simulator 中获得的 reward
- $R_t^{\text{real}}$: time step $t$ 在 real world 中获得的 reward
- $T$: 总 time steps (sparse reward 时只对最后一步有效)
- 上标 sim/real: 表示环境来源

判断两个 method $i, j$ 谁更好: 如果 $|G(\pi^i)| < |G(\pi^j)|$ 且相对多次 run 的 standard deviation 显著, 那么 method $i$ 更好。

## 3. 四大 Gap 来源的深度分析

paper 把 sim-to-real gap 归因到 MDP 四个元素的 mismatch, 这是最核心的 contribution。Figure 2 展示了这些 gap 的关系。

### 3.1 Observation Gap

形式化: $o_t^{\text{real}} \neq o_t^{\text{sim}}$

两个子原因:

**(1) Completeness gap - $\Delta_{\text{perception}}$**:
- 在 sim 中 $o_t^{\text{sim}} = s_t^{\text{sim}}$ (full observability, 完美感知)
- 在 real 中 $o_t^{\text{real}} \neq s_t^{\text{real}}$ (partial observation, 这是 POMDP 问题)
- 这就是为什么 paper 后面用 $o_t$ 而不是 $s_t$ 来表示 observation

**(2) Feature representation mismatch - $\Delta_{\mathcal{S}}$**:
- 来自 perception resolution 差异, sensor noise 等
- 比如 sim 里的 image 是 64x64 clean, real 里是 1080p noisy

**Intuition**: 想象你训练一个 robot grasping policy, sim 里的物体边界 perfectly clear, 但 real camera 有 motion blur 和光照变化。Policy 学到的 visual features 在 real 中完全失效。

### 3.2 Action Gap

形式化: $a_t^{\text{real}} \neq a_t^{\text{sim}}$

两个子原因:

**(1) Action granularity - $\Delta_{\mathcal{A}}$**:
- $a_t^{\text{sim}} \in \mathcal{A}_s$: sim 中 action 通常是 discretized 或 oversimplified (比如 "move forward 1 unit")
- $a_t^{\text{real}} \in \mathcal{A}_r$: real 中 action 是 continuous 的 (比如 joint torque 的连续值)
- 采样: $a_t^{\text{sim}} \sim \pi_t(\cdot | o_t^{\text{sim}}), o_t^{\text{sim}} \subseteq s_t$

**(2) System state gap - $\Delta_{\text{system}}$**:
- sim 假设 action instantly 生效
- real 中 mechanical components 有 latency (机械延迟)
- 这会进一步 propagate 到 transition 和 reward

### 3.3 Transition Gap (Next State Divergence)

形式化: $P_s(s_{t+1}|s_t, a_t) \neq P_r(s_{t+1}|s_t, a_t)$

这就是 paper 里说的 "next-state divergence": 给定同样的 state $s_t$ 和 action $a_t$, sim 和 real 产生的 $s_{t+1}$ 不同。

根本原因: **system dynamics gap - $\Delta_{\text{system}}$**

这其实是最难解决的 gap, 因为它涉及物理定律的精确建模。比如:
- sim 里的 friction coefficient 是常数 0.5
- real 里 friction 随温度、湿度、表面磨损变化

### 3.4 Reward Gap

形式化: $r_t^{\text{real}} \neq r_t^{\text{sim}}$

reward function: $r_t^{\text{sim}} := r(o_t^{\text{sim}}, a_t^{\text{sim}})$

两个原因:
1. **$\Delta_{\text{system}}$ 导致**: reward 设计基于 sim 的理解, 没覆盖 real 的 cases
2. **$\Delta_{\mathcal{A}}$ cascade**: action delay/granularity 差异导致 reward 信号失真

**Intuition**: reward gap 是其他三个 gap 的下游效应。如果你的 observation 错了, action 错了, transition 错了, 那 reward 必然错。所以 reward 方向的解决方案往往是间接的——通过 shaping 来补偿其他 gap 带来的 reward 失真。

## 4. Techniques 的深度讲解

### 4.1 Observation 方向

#### Domain Randomization (DR)

核心思想: 在 sim 中随机化 visual parameters (textures, lighting, object positions, camera params), 让 policy 见过足够多的 variations, 从而 generalize 到 real。

关键 papers:
- Tobin et al. 2017: https://arxiv.org/abs/1703.06907 - 最早的 DR 工作
- OpenAI Rubik's Cube (ADR): https://arxiv.org/abs/1910.07113 - Automatic Domain Randomization with curriculum

**ADR 的关键创新**: 
- 不需要手动调 randomization range
- Curriculum-based: 随 policy 改善, 逐渐增加环境难度
- 同时训练 policy 和 vision model
- 在 observations 中加入 Gaussian noise with randomized parameters

**技术细节**: ADR 维护一个 performance threshold $\tau$, 当 policy 在当前 difficulty level $d_k$ 上的 success rate $s(d_k) > \tau$ 时, 升级到 $d_{k+1}$。这避免了两个问题:
1. 一开始就 randomize 太多导致训练 diverge
2. randomize 太少导致 overfitting 到 sim

#### Domain Adaptation (DA)

核心思想: 对齐 sim 和 real 的 feature distribution, 让 policy 学到的 features 在两个 domain 都 work。

主要技术:
- **Adversarial training**: 类似 GAN, 用 discriminator 区分 sim/real features, encoder 学 fool discriminator
- **Embedding alignment**: 直接最小化两个 domain 的 feature embedding 距离

关键 papers:
- CycleGAN-based: https://arxiv.org/abs/1703.10593
- RetinaGAN: https://arxiv.org/abs/2102.07973
- Bi-directional DA: https://arxiv.org/abs/2103.02699 - real2sim + sim2real 双向
- VR-Goggles: https://arxiv.org/abs/1902.07352 - 把 real images 翻译回 synthetic modality

**Bi-directional DA 的 insight**: 传统 DA 只 sim→real, VR-Goggles 反过来 real→sim, 在 deployment 时把 real image stream 翻译成 synthetic, 这样 policy 不用 adapt。这大幅降低 training computational cost。

#### Sensor Fusion

多传感器融合补偿单传感器 limitation:
- Visual + Depth: https://arxiv.org/abs/1709.01227
- LiDAR + Camera: 补偿各自 modality 的 blind spots
- GPS + IMU: https://arxiv.org/abs/2007.01278 - 四轮独立驱动电动车, 处理 individual sensor bias/noise

**Intuition**: 单 sensor 在 sim 里可能完美, 但在 real 里各有 failure mode。融合多个 sensor, 它们的 failure mode 不 correlated, 所以 ensemble 更 robust。

#### Foundation Models for Observation

这是 paper 的新意所在。用 LLM/VLM:
- **Semantic anchors**: 自然语言描述作为统一信号, 在 sim 和 real 中都保持一致 (e.g., "a red mug on table")
- **VLM annotations**: 给 sensory input 生成描述性 annotation, 帮助 align sim/real observation
- Paper: https://arxiv.org/abs/2405.10020 - "Natural Language Can Help Bridge the Sim2Real Gap"

**关键 insight**: 语言是 abstraction layer, 它自然地 filter 掉了 sim 和 real 之间的 low-level visual differences。说"红色杯子"在 sim 和 real 里都是"红色杯子", 不管 pixel-level 长得多不一样。

### 4.2 Action 方向

#### Action Space Scale

核心问题: Discrete (sim) → Continuous (real) gap

方法:
- **Subgoal model**: 在 sim navigation graph 里识别 nearby waypoints, 让 low-fidelity policy 在 real 中 work
- **Action Shielding**: safety layer/filter 评估 action 的 feasibility, 拒绝 unsafe action
  - Paper: https://arxiv.org/abs/1708.08626 - Safe RL via Shielding

#### Action Delays

这是 paper 里技术细节最丰富的部分之一。sim 假设 action instant 生效, 但 real 有 latency。

**(1) Constant-Delayed MDP**:
- Firoiu et al. 2018: https://arxiv.org/abs/1810.07286
- 用 predictive model, inspired by 人类 subconsciously anticipate near future

**(2) Random-Delay MDP (RDMDP)**:
- Bouteiller et al. 2020 (DCAC): https://arxiv.org/abs/2010.07384
  - Formally 定义 RDMDP
  - Delay-Correcting Actor-Critic: 用 action buffer + delay measurements 修正 delay
  - 从 off-policy samples 生成 on-policy sub-trajectories

- PAD (Prediction with Arbitrary Delay): https://arxiv.org/abs/2010.07384 相关
  - Multi-step prediction, 但用 single prediction step 而非 iterative updates
  - Gated unit 动态调整 feature extraction layers for 不同 delay

- D-SARSA, D-Q: https://arxiv.org/abs/1003.4144 相关 work
  - Control-delay MDP
  - 不用 state augmentation (避免 complexity exponential growth)
  - 基于 effective delayed actions 更新 Q-values

- Delayed-Q (Derman et al. 2021): https://arxiv.org/abs/2101.11992
  - 用 forward model 推断 future state from delayed action sequence
  - 然后用 inferred future state 更新 Q-values

**Intuition**: Action delay 本质上是 policy 的 decision 和 environment 的 response 之间有时间错位。解决方案分两类:
1. State augmentation: 把历史 action 加到 state 里 (但 complexity 爆炸)
2. Prediction: 学一个 forward model 预测 "如果没 delay 会发生什么"

#### Action Uncertainties

**(1) Action Advising**:
- RCMP (Requesting Confidence-Moderated Policy advice): https://arxiv.org/abs/2002.09879
  - 用 epistemic uncertainty 引导 action selection
  - 学多个 value function estimates, 计算它们的 variance 作为 uncertainty measure
  - 高 uncertainty 时 request advice from external advisor

**(2) Action Robust RL**:
- PR-MDP (Probabilistic Action Robust MDP) 和 NR-MDP (Noisy Action Robust MDP): https://arxiv.org/abs/1906.06031
  - 考虑 adversarial 干扰下的 action outcomes
  - PR-MDP: 以概率 $p$ 把 action 替换成 adversarial action
  - NR-MDP: 在 action 上加 Gaussian noise

- ARRLC: https://arxiv.org/abs/2307.07666
  - 同时用 optimistic 和 pessimistic Q-function estimates
  - 平衡 exploration 和 adversarial planning

#### Foundation Models for Action

- **Local policies + VLM planning**: https://arxiv.org/abs/2410.22332
  - 训练 simple local policies 作为 action pool (pick, open, close...)
  - VLM 提供 planning strategy 协调这些 action 完成 long-horizon task
  - 在 Robosuite benchmark 上 SOTA

- **SayNav**: https://arxiv.org/abs/2401.02917
  - Incremental scene graph generation
  - LLM-based planner
  - Low-level executor
  - 在 Multi-Object Navigation task 上 SOTA

- **RLingua**: https://arxiv.org/abs/2405.02075
  - LLM 生成 rule-based controllers
  - 引导 exploration, 减少 interaction samples

### 4.3 Transition 方向

#### Domain Randomization (Transition)

随机化 physical parameters: friction, motor torque, mass, inertia 等。

- Peng et al. 2018: https://arxiv.org/abs/1710.06537 - dynamics randomization for robotic control
- Active Domain Randomization (ADR): https://arxiv.org/abs/1910.10625
  - 主动识别导致 policy discrepancy 最高的 configuration
  - 在这些 challenging variations 上集中训练

**Intuition**: 普通 DR 是 uniform randomize, ADR 是 importance sampling——把训练资源分配给那些 policy 表现最差的 environment configurations。

#### Domain Adaptation (Transition)

对齐 parameter distributions:
- Adversarial training: https://arxiv.org/abs/1612.07944
- Conditional DA: https://arxiv.org/abs/1705.10667

#### Grounding Methods

这是 paper 里我觉得特别有意思的一类方法。核心思想: 调整 simulator dynamics 让它 match real dynamics, 通过 grounded actions。

**(1) GAT (Grounded Action Transformation)** - Hanna & Stone 2017:
- https://arxiv.org/abs/1610.07428
- 核心思想: 学习一个 action transformation function $f: a^{\text{sim}} \to a^{\text{transformed}}$
- 使得 $\mathcal{T}_s(s'|s, f(a)) \approx \mathcal{T}_r(s'|s, a)$
- 即: 在 sim 里执行 transformed action 后的 next state, 和 real 里执行原 action 后的 next state 相同

**(2) SGAT (Stochastic GAT)** - Desai et al. 2020:
- https://arxiv.org/abs/2010.09880
- 把 GAT 的 deterministic setup 换成 stochastic
- 学一个 distribution over possible next states, 而不是单个 next state
- 更好地 approximate real-world stochasticity

**(3) RGAT (Reinforced GAT)** - Karnan et al. 2020:
- https://arxiv.org/abs/2011.02420
- 把 grounding 本身作为 RL problem
- End-to-end 训练 action transformer 作为 single neural network
- 减少 error accumulation

**(4) GARAT (Generative Adversarial Reinforced Action Transformation)** - Desai et al. 2020:
- https://arxiv.org/abs/2004.09495
- Framing: Imitation from Observation (IfO) problem
- 用 GAN-style adversarial training 最小化 sim/real dynamics distribution mismatch
- 不需要 explicit action labels

**(5) Uncertainty-aware GAT** - Da et al. 2023:
- https://arxiv.org/abs/2305.17975
- 给 GAT framework 加 uncertainty quantification
- 增强 Sim-to-Real policy training 的 reliability

**Intuition on Grounding**: 想象 sim 是一个 "近似 real" 的环境。Grounding 就是教 sim "撒谎"——当 policy 说要执行 action $a$, sim 实际执行 $f(a)$, 这个 $f(a)$ 在 sim physics 下产生的结果, 等价于 $a$ 在 real physics 下产生的结果。Policy 学到的是 "在撒谎的 sim 里" 的最优策略, 但因为这个谎言恰好补偿了 physics gap, policy 在 real 里直接 work。

#### Distributionally Robust Learning

把 sim-to-real formalize 为 distributionally robust optimization:

- Liu & Xu 2024: https://arxiv.org/abs/2402.02192 - provable efficiency with linear function approximation
- Liu & Xu 2025: minimax optimal algorithms for distributionally robust offline RL
- Tang et al. 2024: https://arxiv.org/abs/2411.18612 - f-divergence regularization
- Liu et al. 2024: https://arxiv.org/abs/2409.20521 - tight bounds

**核心 formulation**: 
$$\max_\pi \min_{P \in \mathcal{U}} \mathbb{E}_{P} \left[ \sum_t \gamma^t r_t \right]$$

其中 $\mathcal{U}$ 是 uncertainty set, 包含所有 "离 sim distribution 在某个 divergence measure 下不超过 $\epsilon$" 的 transition distributions $P$。

**Intuition**: 与其精确学 real dynamics, 不如假设 real dynamics 在 sim dynamics 的某个 "邻域" 内, 然后学一个对整个邻域都 robust 的 policy。这避开了精确 modeling 的难题。

#### LLM-Enhanced Approaches

- Prompt to Transfer: https://arxiv.org/abs/2401.06256
  - 用 LLM 推理能力增强 forward model 的 real-world dynamics prediction
  - 更可靠的 $\hat{s}_{t+1}$ 让 inverse model 产生更好的 grounded actions
  - 输入 $(\hat{s}_{t+1}, s_t)$ 输出 grounded action

### 4.4 Reward 方向

#### Reward Shaping

**(1) Potential-based reward shaping**:
$$r'_t = r_t + \gamma \Phi(s_{t+1}) - \Phi(s_t)$$

其中 $\Phi: \mathcal{S} \to \mathbb{R}$ 是 potential function, 编码 prior knowledge。

**关键性质**: 这个 shaping 不改变 optimal policy (theoretical guarantee), 但加速 learning。

**(2) Automaton-guided reward shaping**:
- 用 automata (有限状态机) 作为 structured representation
- 动态更新 reward function based on automaton transition utility
- 缓解 sparse reward 问题
- Paper: https://arxiv.org/abs/2310.19137

**(3) Assistant reward agents**:
- Auxiliary agent 协作生成 supplementary reward signals
- 基于 future-oriented information
- 动态 adapt reward structure

**(4) Trajectory distribution matching**:
- Guo et al. 2025: https://arxiv.org/abs/2410.19093
  - 通过 trajectory distribution matching 实现 reward augmentation
  - 用 imitation learning transfer policy

**(5) Return augmentation with Decision Transformers**:
- Wang et al. 2024: https://arxiv.org/abs/2410.23450
  - 用 decision transformer 做 sim-to-real transfer
  - Return-conditioned supervised learning (RCSL) 方法的改进

#### LLM-Based Reward Design

**(1) Automated reward generation (CARD)**:
- https://arxiv.org/abs/2410.14660
- Iteratively 生成和 refine reward function code
- 无需 human intervention
- Dynamic feedback mechanism 对齐 task objective

**(2) Evolutionary reward design**:
- https://arxiv.org/abs/2405.13977
- LLM 提出多样 candidate reward structures
- Evolutionary search 评估和优化
- Narin 2024

**(3) Text2Reward**:
- https://arxiv.org/abs/2309.11489
- 从 textual task specification 自动生成 dense reward function
- 自然语言 → executable reward code
- 减少 domain-specific expertise 需求

**Intuition**: Reward design 是 RL 里最 "human-in-the-loop" 的部分, 也是最容易 inject sim bias 的部分。LLM 的价值在于: (1) 自动化 reward 生成减少 human bias, (2) LLM 的 world knowledge 可以覆盖 sim 没见过的 real scenarios, (3) 自然语言作为 reward specification 更 robust to sim/real 的 low-level differences。

## 5. Domain-Specific Challenges

### 5.1 Robotics

Research focus: 
1. **Safety and risk mitigation** - 物理 system, 任何 misstep 都可能导致 damage
2. **Multi-task capability** - 一个 platform 处理 varied tasks
3. **High accuracy** - fine manipulation / precision tasks

Key references:
- https://arxiv.org/abs/2010.07594 - 机器人 manipulation
- https://arxiv.org/abs/1910.07113 - Rubik's cube with robot hand

### 5.2 Transportation

Research focus:
1. **Multi-agent coordination** - 大量 agent 交互
2. **Smooth transitions** - mixed control (human + automated)
3. **Real-time decisions** - 快速变化环境

Key simulators:
- SUMO: https://eclipse.dev/sumo/
- CARLA: https://carla.org/
- CityFlow: https://github.com/cityflow-project/CityFlow
- SMARTS: https://github.com/huawei-noah/SMARTS

### 5.3 Recommender Systems

Research focus:
1. **Large-scale online learning** - 持续更新 recommendation policy
2. **Off-policy evaluation** - 评估 policy 而不实际部署
3. **Counterfactual sensitiveness** - what-if 分析

Key papers:
- RecSim: https://arxiv.org/abs/1909.04847
- S2R-Rec: https://arxiv.org/abs/2110.11073
- Virtual-Taobao: https://arxiv.org/abs/1805.10000
- KuaiSim: https://arxiv.org/abs/2309.12645

## 6. Evaluation 的三种 Settings

Figure 6 展示了三种 evaluation setting 在 cost, safety, realism 三维上的 trade-off:

### 6.1 Sim-to-Real
- **Highest realism**, highest cost, lowest safety
- 直接把 sim-trained policy 部署到 real physical system
- 适用于 robotics, autonomous driving 等必须 real interaction 的 domain

### 6.2 Sim-to-Scale-down-Real
- **Balanced** safety + cost, moderate realism
- 用 controlled testbed (motion capture, closed track)
- 例如 Duckietown: https://arxiv.org/abs/1709.01560

### 6.3 Sim-to-Sim
- **Most cost-effective**, safest, but 牺牲 realism
- Policy 在一个 sim 训练, 在另一个 (通常更 realistic) sim 测试
- Vary physics params, sensor noise, dynamics 来评估 robustness

### 6.4 Metrics by Domain (Table 4)

**Robotics**:
- Success rate (task)
- Execution time
- Planning efficiency
- Energy efficiency
- Failure rate (system/task)

**Transportation**:
- Delay
- Throughput
- Queue length
- Travel time
- Pressure (max pressure control)

**Recommender Systems**:
- Click-through rate (CTR)
- Precision
- Recall
- Conversion rate
- Satisfaction score

## 7. Foundation Models 带来的新 Challenges

### 7.1 Hallucination

LLM/VLM 会 generate factually incorrect outputs, 在 RL agent 中这会导致 suboptimal 甚至 unsafe decisions。

Mitigation:
- **RAG (Retrieval-Augmented Generation)**: grounding LLM outputs with real-world data
  - https://arxiv.org/abs/2005.11401
- **LLM uncertainty quantification**: probe LLM 的 real understanding
  - https://arxiv.org/abs/2407.00994

### 7.2 Scalability

LLM inference time ∝ model capability, parameter size 越大, 推理越慢。RL training 每个 step 都调用 LLM 非常昂贵。

Mitigation:
- **Task-specific distillation**: 把大 LLM 蒸馏成小 specialized model
  - https://arxiv.org/abs/2312.15842
- **Lighter foundation model designs**: efficient architectures

## 8. GenAI-based Simulations (Section 4.2.2)

这是 paper 里我觉得很有前瞻性的部分, 讨论了 generative AI 如何增强 simulation 本身:

### 8.1 Brute-Force Scaling

- Cusumano-Towner et al. 2025: https://arxiv.org/abs/2502.03349
  - Scale 到 1.6 billion km self-play driving data
  - 观察 "emergent" realistic and robust behaviors
  - 印证 scaling law 在 sim-to-real 的适用性

### 8.2 Logic-Integrated Simulation

- Signal Temporal Logic (STL) 指定 traffic rules / safety constraints
- 自动生成满足这些条件的 scenarios
- Paper: https://arxiv.org/abs/2306.14118

### 8.3 Differentiable & Physics-Augmented Simulations

- **PAC-NeRF**: https://arxiv.org/abs/2303.05512
  - Physics Augmented Continuum Neural Radiance Fields
  - 从 motion data 学 material/fluid properties
  - Geometry-agnostic system identification

- **Neural PDE solvers**: 
  - https://arxiv.org/abs/2306.08843
  - Learn neural constitutive laws from motion observations

### 8.4 Generative World Models

- **NVIDIA COSMOS**: https://developer.nvidia.com/cosmos
  - Unified "world foundation model"
  - 协调 scene composition, lighting, object configurations at scale

- **Genesis**: https://github.com/Genesis-Embodied-AI/Genesis
  - "Generative and Universal Physics Engine"
  - Data-driven 捕获 broad range of physical behaviors

## 9. 我的 Intuition 和延伸思考

### 9.1 为什么 MDP-based Taxonomy 是好的?

因为 MDP 是 RL 的 first principles。任何 sim-to-real 问题, 本质上都是 sim 的 MDP $\mathcal{M}_s$ 和 real 的 MDP $\mathcal{M}_r$ 之间的 mismatch。把方法映射到 MDP 元素上, 让我们:
1. 精确诊断问题源头 (是 observation 错了还是 transition 错了)
2. 组合使用方法 (observation 用 DR, transition 用 grounding, reward 用 LLM shaping)
3. 发现 research gap (比如某元素的方法明显少)

### 9.2 Foundation Models 的真正价值

paper 里 foundation models 出现在四个元素的每一个, 但我觉得它的真正价值在于:

**Semantic abstraction layer**: 语言/视觉-语言模型提供了 high-level abstraction, 这个 abstraction 自然 robust to low-level sim/real differences。"Pick up the red cup" 这个 instruction 在 sim 和 real 里语义相同, 不管 pixel-level 差多大。

**Prior knowledge injection**: LLM 的预训练知识可以覆盖 sim 没见过的 scenarios, 这对 reward design 和 dynamics prediction 特别有价值。

### 9.3 三个最有潜力的方向

基于我的 reading, 我认为这些方向最有突破潜力:

**(1) LLM-enhanced Grounding (Prompt to Transfer)**:
https://arxiv.org/abs/2401.06256
把 LLM 的 commonsense reasoning 注入 forward model, 这是 grounding 方法的新 paradigm。

**(2) Distributionally Robust RL**:
https://arxiv.org/abs/2402.02192
把 sim-to-real 作为 robust optimization 问题, 有 theoretical guarantee, 这是少数有 provable efficiency 的方向。

**(3) Generative World Models as Simulators**:
Genesis, COSMOS 这类 generative physics engine, 可能从根本上重新定义 simulator——不再是 hand-engineered, 而是 data-driven 学出来的 physics。这可能是 sim-to-real 的终极解法: 如果 sim 足够像 real, gap 自然消失。

### 9.4 评估的根本难题

paper 指出, sim-to-sim 的 predictivity 仍然 largely 依赖 specific metrics。这是 fundamental problem: 我们没有 theoretical bound 说 "sim-to-sim gap < ε ⟹ sim-to-real gap < δ"。

Wagenmaker et al. 2024 (https://arxiv.org/abs/2410.20254) 尝试 "Leveraging Simulation to Learn to Explore for Real-World RL", 这是 interesting direction——用 sim 学 exploration strategy, 而不只是 final policy。

### 9.5 缺失的视角

paper 没有深入讨论的:
1. **Causal inference 视角**: sim-to-real 本质是 distribution shift, causal inference 的 tools (do-calculus, counterfactual) 可能有用
2. **Information bottleneck**: He et al. 2024 (https://arxiv.org/abs/2410.20592) 开始 explore 这个方向
3. **Multi-agent sim-to-real**: 大多数方法假设 single agent, 但 transportation 等 domain 本质 multi-agent
4. **Continual sim-to-real**: policy 在 real 中持续学习, 而非一次性 transfer

## 10. Key Resources

- **Paper repository**: https://github.com/ASU-DA-Lab/Sim2Real-RL
- **DARPA TIAMAT program**: https://www.darpa.mil/news-events/2023-04-04 (Transfer from Imprecise and Abstract Models to Autonomous Technologies)
- **Key simulators**:
  - MuJoCo: https://mujoco.org/
  - PyBullet: https://pybullet.org/
  - CARLA: https://carla.org/
  - SUMO: https://eclipse.dev/sumo/
  - Genesis: https://github.com/Genesis-Embodied-AI/Genesis
  - NVIDIA COSMOS: https://developer.nvidia.com/cosmos
- **Key benchmarks**:
  - Robosuite: https://github.com/ARISE-Initiative/robosuite
  - Meta-World: https://github.com/rlworkgroup/metaworld
  - CityFlow: https://github.com/cityflow-project/CityFlow
  - CALVIN: https://github.com/mees/calvin
  - ORBIT: https://github.com/NVIDIA-Omniverse/ORBIT

## 总结

这篇 survey 的核心价值在于它的 **MDP-based taxonomy**, 把看似混乱的 sim-to-real literature 组织成四个清晰的 axis (S, A, T, R), 每个 axis 上有 classic methods 和 foundation model-enhanced methods。同时它形式化了 gap 度量 $G(\pi)$, 给出了 evaluation 的三种 setting 和 domain-specific metrics。

最重要的 takeaway: **sim-to-real 不是单一问题, 而是 MDP 四元素 mismatch 的集合**, 解决方案需要针对具体 mismatch 的 source。Foundation models 的价值在于提供 semantic abstraction layer 和 prior knowledge injection, 但 hallucination 和 scalability 是新的 challenge。

未来最有潜力的方向是 distributionally robust learning (有 theoretical guarantee) 和 generative world models (可能从根本上消除 gap)。
