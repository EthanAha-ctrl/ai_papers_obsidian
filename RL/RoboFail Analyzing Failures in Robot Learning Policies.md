---
source_pdf: RoboFail Analyzing Failures in Robot Learning Policies.pdf
paper_sha256: 3d129e39f5003f40db21bce442c7d9ea90573482414ec421e6a7e860ec0128c6
processed_at: '2026-08-12T00:56:29-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,抛开公式,用大白话再过一遍。

---

## 一句话总结

**RoboFail 就是训练一个"找茬 agent",专门去搞破坏,看什么环境配置能把另一个 robot policy 搞崩。**

---

## 问题是什么

你训练了一个 robot policy,在 lab 里表现 99% success rate。拿出去部署,改个桌子颜色、换个光照、物体位姿偏一点——崩了。

这就是 **overfit to training distribution**。policy 学到的不是"怎么抓物体",而是"在训练集那种特定视觉条件下怎么抓"。

传统做法是手动试各种 perturbation,或者 uniform sampling 一堆配置跑测试。但 perturbation space 是组合爆炸的:颜色 RGB 三维、位姿六维、光照无数维、物体几何无数维……grid search 根本跑不完。

---

## RoboFail 的 trick

与其手动找,不如 **训练一个 agent 去自动找**。

这个 agent(外层 RL agent)手里有一堆"破坏工具"——discrete actions,比如"把 cube 染成红色"、"把 table 染成绿色"、"把物体往左挪 5cm"。每一步它选一个 action,改一下环境,然后让被测 policy 在新环境里跑一遍任务。

- 被测 policy 失败了 → agent 拿 +1000 reward
- 被测 policy 成功了 → agent 拿 -100/horizon 的 penalty

agent 用 PPO 训练,慢慢学会"哪种破坏组合最容易让 policy 崩"。

训练完之后,agent 的 policy network 给你一个 softmax distribution over actions。这个 distribution 就是 **被测 policy 的"弱点报告"**——哪个 action 的概率高,说明那个环境配置最容易触发 failure。

---

## 一个生活化的 metaphor

想象你是个拳击教练,要找出学生(被测 policy)的防守漏洞。

**Naive 做法**:让学生站各种姿势,你从各种角度出拳,看哪种组合能打中。组合太多,试不完。

**RoboFail 做法**:你训练另一个"陪练对手"(RL agent),让他专门负责打中学生。陪练通过 PPO 学习"什么角度、什么节奏最容易突破防守"。训练一段时间后,陪练的出招分布(softmax over punch types)就是学生的弱点画像——他偏好出左勾拳,说明学生左侧防守弱;偏好出直拳,说明学生正面防守弱。

你拿这个画像去给学生针对性训练。

---

## 为什么用 RL 而不是 grid search

grid search 是 **均匀撒网**,大部分撒在"被测 policy 能搞定"的区域,浪费 compute。

RL 是 **directed search**,agent 会越来越往"高 failure-rate 区域"靠拢。相当于 importance sampling,把探索预算花在刀刃上。

而且 failure 通常是 **sequential** 的:不是单个 perturbation 触发的,而是多个 perturbation 叠加。RL 天然处理 sequential decision,grid search 要枚举组合,指数爆炸。

---

## 实验告诉我们什么

作者测了三种 policy:

| Policy 类型 | Input | 直觉 |
|---|---|---|
| Image(n) | 纯视觉 | 最容易 overfit 到 visual cue |
| Image + Proprioceptive | 视觉 + 自身关节状态 | proprioception 给个 stable anchor |
| Image + Language | 视觉 + 语言指令 | language 提供 semantic grounding |

结果:**纯视觉 policy 最脆弱**,稍微改视觉就崩。加 proprioception 或 language 都能提升 robustness。

这个结果本身不意外,但 RoboFail 给你的是 **定量的 weakness profile**:不是笼统说"纯视觉差",而是告诉你"纯视觉 policy 在 Red Cube 配置下 P_failure = 0.35,在 Green Table 配置下 P_failure = 0.12"——你知道该去 collect 哪种 demo 来 retrain。

---

## Entropy 的直觉

雷达图(Figure 4)里,形状越接近圆,entropy 越高,说明 policy 在所有配置下都失败——**全面脆弱**,问题大。

形状越尖,entropy 越低,说明 policy 只在特定配置下失败——**定向脆弱**,问题可预测,容易 fix。

但有个 caveat:entropy 高也可能是 policy 太好,RL agent 找不到任何 failure,分布退化成 uniform。所以要看 **绝对量级**,不只看形状。这是 paper 没完全讲清楚的地方。

---

## 几个我觉得 clever 的设计点

**Reward 的 asymmetry**:+1000 vs -100/horizon,正负极度不对称。这确保 agent 不会学成"挑最简单的扰动让 policy 成功拿小惩罚",而是真的去追 failure。这个 reward shaping 很 aggressive 但合理。

**ent_coef = 0**:一般 PPO 会加 entropy bonus 鼓励探索,这里故意设 0。因为最终要拿 π_θ 的分布做 failure mode 解释,entropy 太大分布就接近 uniform,失去诊断价值。这是 **"要 determinism 不要 exploration"** 的有意取舍。

**Shared backbone actor-critic**:NatureCNN backbone 同时喂 actor head 和 critic head,省参数,视觉 feature 同时服务 policy 和 value estimation。经典 PPO 做法,干净。

---

## 几个我觉得 weak 的地方

**Action space 太小且 predefined**:agent 只能在预定义的 9 个(或 20 个,paper 里数字不一致)discrete actions 里选。它不会发现"加一个 distractor object"这种没在 action space 里的 failure cause。这是最大 limitation。要真正强大,action space 得是 programmatic 或 continuous,让 agent 能"发明"新破坏方式。

**P_failure 不是 calibrated probability**:softmax 输出只是 preference ranking,不是真实 P(fail | env)。要拿来做 safety certificate,需要额外 calibration(temperature scaling 之类)。paper 直接把 π_θ(a|o) 当 P_failure(a),这个 interpretation 是 proxy,不是严格估计。

**Compute cost 高**:每个 PPO step 都要跑被测 policy 一整个 episode。如果被测 policy 是 VLA 或 diffusion policy,单步几百 ms,整个 RoboFail 训练巨慢。用 world model 替代 sim 能加速,但 paper 没做。

**Sim-to-real gap of failure modes**:sim 里找到的 failure 配置,real world 未必复现。paper 没验证这个 transferability。

---

## 更大的 picture

RoboFail 是一个 **deep pattern** 的 instance:**用 learning 系统 attack 另一个 learning 系统**。

这个 pattern 到处都是:

- **Red teaming LLMs**:用 RL agent 找 jailbreak prompt(Hong et al. https://arxiv.org/abs/2402.19464)
- **Stress testing AV perception**:用 RL agent 找 corner case 让 perception 模型出错(Delecki et al. https://arxiv.org/abs/2202.05807)
- **Adversarial examples**:用 gradient attack 找 NN 的 brittle input(Goodfellow et al. FGSM)
- **Software fuzzing**:AFL / libFuzzer 用 coverage-guided 找 program crash

本质都是同一个故事:**Software 2.0 时代,我们用 learner 当 fuzzer,去 fuzz 另一个 learner**。RoboFail 是这个故事在 robot manipulation 上的 clean instantiation。

---

## 对你(Karpathy)的几个直接联想

**Software 2.0 fuzzing**:你提过 Software 2.0 概念(https://karpathy.medium.com/software-2-0-a64152b37c35)。RoboFail 就是 Software 2.0 的 fuzz testing。Software 1.0 时代我们用 AFL 找 program crash;Software 2.0 时代我们用 RL agent 找 NN policy 的 brittle configuration。这个类比值得写成 blog。

**Data is all you need, but targeted data**:你常说 data 的重要性。RoboFail 揭示的不是"需要更多 data",而是"需要 **targeted** data"——找到 policy 失败的 case 去 label,比多 label 一堆 IID case 有用得多。这是 active data collection 的核心 argument。

**nanoGPT / micrograd 教学价值**:RoboFail 的 PPO 实现几百行能复现,"NN 攻击 NN"的可视化教学价值很高。适合做 lecture demo:左边是被测 policy 抓物体,右边是 RL agent 改环境,看 agent 怎么一步步学会"搞破坏"。这种 demo 比 dry 讲 PPO 公式直观一百倍。

---

## 一句话收尾

RoboFail 的核心 insight 就一句:**别手动找 policy 的弱点,训练一个 agent 去自动找。** 这个 idea 简单、clean、可扩展,是 "learning to break learning" 在 robot manipulation 上的 nice instantiation。limitation 主要在 action space 太小、calibration 缺失、compute cost 高,但 framework 本身的 extensibility 很强,后续工作空间大。

---

这篇 RoboFail 是一篇挺有意思的"meta-learning"性质的工作:它不是去训练一个更好的 robot policy,而是去训练一个 **找 policy 毛病** 的 agent。整个 framework 的核心想法可以概括为 **"用 RL 去 attack 另一个 policy"**,然后把攻击者(RL agent)的 action distribution 当成被测试 policy 的 failure mode 概率分布。下面我把这篇 paper 拆开,从 motivation、formulation、公式、architecture、实验,到与其它工作的联系,都尽量讲细,目标是帮你 build intuition。

---

## 1. 核心动机:为什么 failure discovery 本身要被 formulate 成一个 RL 问题

robot manipulation policy 现在越做越大(RT-1 https://arxiv.org/abs/2212.06817、RT-2 https://arxiv.org/abs/2307.15818、Octo https://arxiv.org/abs/2405.12213、Open X-Embodiment https://arxiv.org/abs/2310.08864),但 **overfit 到 training distribution** 的问题反而越来越严重。这些模型在 IID 测试集上 success rate 漂亮,稍微改一下桌子颜色、物体颜色、光照,就崩。这种 OOD fragility 在 real-world deployment 是危险的。

naive 的做法是 grid search / uniform sampling 各种 perturbation,但 perturbation space 是组合爆炸的:颜色、位姿、物体几何、摩擦、光照……每种都是一维或高维,exhaustive search 不可行。所以作者把这个搜索过程 **本身** 当成一个 sequential decision making 问题,让一个 learner 主动去探索那些高 failure-rate 的区域。这本质上和 **red teaming LLMs**(Hong et al. curiosity-driven red-teaming https://arxiv.org/abs/2402.19464)、**stress testing AV perception**(Delecki et al. https://arxiv.org/abs/2202.05807)、**black-box safety validation**(Corso et al. JAIR survey https://arxiv.org/abs/2008.06270)是同构思路,只是 apply 到 manipulation policy 上,并加了一层 probabilistic interpretation。

关键 insight:与其问"这个 policy 在哪些情况下会失败",不如去 **训练一个 agent 主动构造这些情况**——RL 给你一个 principled 的 importance sampling over failure configurations。

---

## 2. 方法 pipeline(对应 Figure 1)

整个 framework 有三个 stage,串成一个 pipeline:

**Stage 1 — Environment manipulation**:被测试的 policy 在 Robosuite(https://robosuite.github.io/) / robomimic(https://robomimic.github.io/) 的 manipulation task 里 rollout。一个 **外层 RL agent** 通过 discrete actions 修改 sim 的某些参数(扰动 P),例如改 cube 颜色为红、改 table 颜色为绿/蓝、改 object 位姿等等。每次修改后,被测 policy 在新配置下尝试完成任务。

**Stage 2 — Failure discovery via PPO**:外层 agent 用 PPO 训练,reward 是被测 policy 是否失败。训练目标:找一个 action sequence / 配置,最大化被测 policy 失败的概率。

**Stage 3 — FM analysis**:训练完后,把外层 agent 的 policy π_θ(a|o) 拿出来,在固定 observation 下输出一个 softmax distribution over actions,把这个 distribution 当成 **failure mode 概率分布** P_failure(a) = π_θ(a|o_i)。

注意一个微妙点:这里 P_failure(a) 其实是 **"RL agent 在该 observation 下选择 action a 的概率"**,把它解释为 **"action a 对应的环境配置导致 failure 的概率"** 是一个隐含假设:RL agent 已经被训练得"偏好选 high-failure-rate action",所以它的 action preference 就是 failure likelihood 的 proxy。这个 interpretation 不是严格 frequentist 意义上的 P(fail | env),需要 calibration 才能严格说成是校准概率,文末也提到 P_failure(a) = P(Failure | Env),但严格来说这是 conditional proxy,不是频率估计。

---

## 3. MDP 形式化与公式逐项解读

RoboFail 把"找 failure"形式化为 MDP ⟨S, A, P, R, γ⟩:

### 3.1 State Space S
state 是被测 policy 的当前 observation(通常是 RGB image of robot + workspace),即扰动施加后的 sim 状态。这里其实有一个细节:state 是外层 agent 看到的 observation,不是被测 policy 看到的 observation。两者重合度大但不完全相同,因为外层 agent 需要知道"扰动是什么状态"才能决定下一步怎么继续扰动。

### 3.2 Action Space A
discrete actions,每个 action 对应一种环境参数修改(例如 "set cube color to red"、"set table color to green"、"shift object pose")。文中 paper 提到 20 个 actions(Appendix C 里 "naive for-loop that uniformly samples all 20 actions"),但 Architecture B 又写 action space size = 9。这处不一致我猜是因为:9 是 raw action categories,20 是 categories × values 的笛卡尔积。这是 paper 一个小瑕疵,值得在 review 时追问。

### 3.3 Reward Function R(公式 1)

$$
R = \begin{cases} 1000 & \text{if the policy fails to complete the task,} \\ \frac{-100}{\text{horizon}} & \text{if the policy succeeds.} \end{cases}
$$

这个 **asymmetric reward** 设计有几个细节值得琢磨:

- **+1000 vs -100/horizon**:正 reward 数量级远大于负 reward,确保 agent 强烈偏好找 failure,而不是"避免成功"。否则 agent 可能学到一个 trivial 策略,挑最简单的扰动让被测 policy 成功(从而拿到稳定的负 reward 中较少惩罚的那个)。
- **除以 horizon**:如果被测 policy 成功了,episode 越短(被测 policy 越快成功),惩罚越大(因为 -100/horizon 中 horizon 小则负得更多);episode 越长(被测 policy 磨蹭半天才成功),惩罚越接近 0。这个设计鼓励 agent 找那种"被测 policy 拖很久才勉强成功"或者干脆"失败"的 case——这些才是 policy 的边界 case。
- 量级 +1000 vs -100/200(假设 horizon=200)的 ratio 是 ~2000:1,是非常 aggressive 的 reward shaping。如果被测 policy success rate 很高(比如 95%),agent 可能学得很慢,因为大部分时候拿到的是 -0.5 这种小负 reward,gradient signal 弱。这是 PPO 训练 RL-as-attacker 时常见的瓶颈。

### 3.4 Transition Dynamics P(s'|s,a)
由 Robosuite 的 physics engine 决定,带 stochasticity(noise、robot actuation uncertainty)。这里 transition 不是外层 agent 直接控制的,而是"外层 agent 修改环境 → 被测 policy 在该环境中执行 → 拿到结果 → 下一状态取决于 sim physics"。所以 P 实际上是 **environment modification + 被测 policy rollout** 的联合产物,是个 partially observable 的过程。文中用 MDP 而非 POMDP,是一种 simplification。

### 3.5 Discount Factor γ = 0.99
γ ∈ [0,1] 是未来 reward 的折扣因子。γ^k 中 k 是 future time step index,γ^k 让第 k 步后的 reward 按 γ^k 衰减。γ=0.99 意味着 100 步后的 reward 还值 0.99^100 ≈ 0.366,这个值鼓励 long-horizon planning,适合 manipulation 任务(几十到几百步的 episode)。

---

## 4. PPO 公式逐项解读

### 4.1 Clipped Surrogate Objective(公式 2)

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

各项含义:

- **t**:rollout 时间步 index。
- **θ**:policy network 参数(CNN + linear head),是要优化的对象。
- **π_θ(a_t|s_t)**:当前(新)policy 在 state s_t 下选 action a_t 的概率。
- **π_{θ_old}(a_t|s_t)**:rollout 时使用的旧 policy(上一次 update 前的 policy)在同一 state 选同一 action 的概率。
- **r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)**:importance sampling ratio,衡量新 policy 相对旧 policy 在该 action 上的相对变化。r_t = 1 表示没变化,r_t > 1 表示新 policy 更喜欢这个 action,r_t < 1 表示新 policy 更不喜欢。
- **Â_t**:estimated advantage function,通常用 GAE 计算(后面会讲),衡量"该 action 比平均好多少"。Â_t > 0 表示这个 action 比期望好,应该多采样;Â_t < 0 表示差,应该少采样。
- **ε**:clip 范围参数,通常 0.1~0.2,这里没明说,常见值 0.2。clip 把 r_t 限制在 [1-ε, 1+ε] = [0.8, 1.2] 内,防止单次 update 把 ratio 推到极端值,导致 policy 突变、训练崩溃。
- **min(·, ·)**:这是 PPO 的核心 trick,确保目标函数不会 unbounded 增长。具体行为:
  - 当 **Â_t > 0**(好 action):min 取 r_t·Â_t 和 (1+ε)·Â_t 的较小者。如果 r_t 已经 > 1+ε,说明新 policy 已经把 ratio 推到 clip 上界,继续 push 不再增加 loss,梯度被"截断",防止过度利用这个 action。
  - 当 **Â_t < 0**(差 action):min 取 r_t·Â_t 和 (1-ε)·Â_t 的较小者(注意 Â_t 是负的,这里大小关系翻转)。如果 r_t < 1-ε(新 policy 已经更不喜欢这个 action),ratio 被 clip 在下界,但 loss 仍允许"继续减小"这个 action 的概率——PPO 的 negative gradient 不被截断,避免错误增加坏 action 的概率。
- **E_t[·]**:对 rollout 样本的期望(实际是 sample mean)。

intuition:clip 不是简单的"限制梯度大小",而是一种 trust region 的 soft 实现——只在 ratio 离 1 太远时才 freeze gradient。这就是 PPO 比 vanilla policy gradient 更稳定的原因。在 RoboFail 这种 reward 极度稀疏(只在 episode 结尾给 +1000 或小负数)的场景下,稳定性至关重要,否则很容易 collapse 到 trivial 策略(比如永远选同一个 action)。

### 4.2 Value Function Loss(公式 3)

$$
L^{\text{VF}}(\theta_v) = \mathbb{E}_t \left[ \left( V_{\theta_v}(s_t) - R_t \right)^2 \right]
$$

- **V_{θ_v}(s_t)**:critic network(参数 θ_v)估计的 state value,即"从 s_t 出发,期望累计 reward 是多少"。
- **R_t = Σ_{k=0}^{∞} γ^k · r_{t+k}**:discounted return。γ 是 discount factor(0.99),γ^k 中 k 是从 t 起的未来步数,r_{t+k} 是第 t+k 步的 immediate reward。R_t 是 Monte Carlo 估计的 target。
- 这个 loss 就是 standard MSE regression,让 critic 拟合实际 return。

Advantage Â_t 通常用 **Generalized Advantage Estimation (GAE)**(Schulman et al. 2015 https://arxiv.org/abs/1506.02438)计算:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中 **λ ∈ [0,1]** 是 GAE 参数(paper 里设 0.95),控制 bias-variance tradeoff。λ=0 退化为 1-step TD(高 bias 低 variance),λ=1 退化为 Monte Carlo(低 bias 高 variance)。λ=0.95 是接近 Monte Carlo 的混合,适合 long-horizon manipulation。

δ_t 是 TD error,衡量 critic 预测 vs 实际一步观察的偏差。Â_t 是 δ_t 的指数加权和,权重 (γλ)^l,远期 δ 贡献小。

### 4.3 Total Loss(公式 4)

$$
L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta_v) + c_2 H(\pi_\theta)
$$

- **L^CLIP(θ)**:policy gradient 部分,要 **maximize**(论文里写成 L 是要 maximize 的 surrogate,代码里通常取 negative 当 loss minimize)。
- **c_1 L^VF(θ_v)**:critic loss,前面乘 c_1(vf_coef = 0.5)调节权重。**减号** 因为我们要 minimize critic loss(让 V 逼近 R),同时 maximize policy loss,合并到一个标量方便 backprop。
- **c_2 H(π_θ)**:entropy 正则项,c_2 是 ent_coef。**加号** 因为我们想 maximize entropy(鼓励探索)。注意 Appendix A 写 ent_coef = 0.0,作者这里没加 entropy bonus,可能因为 reward design 已经足够 informative(discrete action + 强 reward),也防止 policy 太 uniform 导致 P_failure 解释不清。这是一个有意思的设计选择——很多 PPO 任务 ent_coef 0.01 起步,这里直接设 0,说明作者更看重 policy 的 **determinism**,因为最终要拿 π_θ 的分布做 failure mode 概率解释,entropy 太大会让 distribution 接近 uniform,失去诊断价值。
- **H(π_θ) = -Σ_a π_θ(a|s) log π_θ(a|s)**:Shannon entropy,衡量 policy 在该 state 下选 action 的不确定性。

---

## 5. CNN Policy Architecture(Appendix B)

外层 agent 的 policy network 是一个 **NatureCNN**(沿用 Mnih et al. DQN 论文的经典结构):

| Layer | Type | Filters / Units | Kernel | Stride | Activation |
|-------|------|----------------|--------|--------|------------|
| 1 | Conv2D | 32 | 8×8 | 4 | ReLU |
| 2 | Conv2D | 64 | 4×4 | 2 | ReLU |
| 3 | Conv2D | 64 | 3×3 | 1 | ReLU |
| 4 | Flatten | — | — | — | — | output dim 3136 |
| 5 | FC | 512 | — | — | ReLU |
| 6 (Actor head) | Linear | 9 (= \|A\|) | — | — | — (softmax) |
| 6 (Critic head) | Linear | 1 | — | — | — |

这是一个非常 **轻量** 的视觉 policy。input 假设是低分辨率 RGB(类似 84×84×3,3136 = 64×7×7,反推 input 大概是 84×84),输出 9 维 logits。

architecture diagram 心智模型:

```
RGB image (s_t)
    │
    ▼
[Conv 8×8, stride 4, 32] ── ReLU
    │
    ▼
[Conv 4×4, stride 2, 64] ── ReLU
    │
    ▼
[Conv 3×3, stride 1, 64] ── ReLU
    │
    ▼
Flatten ── 3136
    │
    ▼
[FC 512] ── ReLU
    │
    ├──▶ Actor head [Linear 9] ── softmax ──▶ π_θ(a|s_t)  (action probabilities)
    │
    └──▶ Critic head [Linear 1] ──▶ V_θv(s_t)  (state value)
```

actor 和 critic 共享前面的 convolution + FC 512 backbone,只在最后一层 fork。这种 shared backbone 是 PPO 的常见做法,节省参数且让 visual feature 同时服务 policy 和 value。但也有 paper 主张分开(Kostrikov et al. https://arxiv.org/abs/2006.05990)更稳定。这里用 shared,简化训练。

**注意**:NatureCNN 处理的是 **外层 agent 的 observation**(用来决定下一步怎么 perturb 环境),不是被测 policy 的 observation。被测 policy 可以是任意架构(BCQN、diffusion policy、VLA 等),它和外层 agent 完全解耦——外层 agent 只通过 reward 信号观察被测 policy 的成败。

---

## 6. PPO 超参数(Appendix A)的关键观察

| Hyperparameter | Value | 含义 / 经验值 |
|---|---|---|
| learning rate | 0.0003 | Adam 经验值,偏大,PPO 常用 3e-4 ~ 1e-4 |
| n_steps (rollout length) | 2048 | 每个 update 收集 2048 步 transition |
| batch size | 64 | mini-batch SGD,2048/64 = 32 mini-batches per epoch |
| epochs | 10 | 每批 rollout 数据重复训练 10 次(PPO 多 epoch 的核心) |
| γ | 0.99 | discount factor |
| GAE λ | 0.95 | bias-variance 权衡 |
| ent_coef | 0.0 | **无 entropy bonus**,前面解释过原因 |
| vf_coef | 0.5 | critic loss 权重 |
| max_grad_norm | 0.5 | gradient clip,防 exploding gradient |

**关键 intuition**:n_steps=2048 配合 epochs=10 意味着每条 rollout trajectory 被重用 10 次。在 reward 极稀疏(只在 episode end 给信号)的情况下,这种重用很重要,因为单次 rollout 提供的 signal 极弱。但 epochs=10 也可能 overfit 当前 batch,特别是 ratio clip 上界被频繁触发时。PPO 之所以能稳定,clip 起了关键作用。

---

## 7. Failure Mode 概率(公式 5-8)

训练完后,外层 agent 给定 observation o_i 时输出一个 softmax 分布:

### 7.1 Log probability(公式 5)

$$
\log \pi_\theta(a | o_i) = f_\theta(o_i)_a - \log \left( \sum_{a' \in \mathcal{A}} \exp(f_\theta(o_i)_{a'}) \right)
$$

- **f_θ(o_i)_a**:policy network 输出的 **logit**(unnormalized score),对应 action a。这是 CNN + actor head 的最后一层输出(未经过 softmax)。
- **log Σ_a' exp(f_θ(o_i)_a')**:log-sum-exp,即 **log partition function log Z**。
- 这个公式其实就是 **log-softmax** 的标准写法,数值上比直接 log(softmax) 更稳定(避免下溢)。

### 7.2 Probability(公式 6)

$$
\pi_\theta(a | o_i) = \frac{\exp(f_\theta(o_i)_a)}{\sum_{a' \in \mathcal{A}} \exp(f_\theta(o_i)_{a'})}
$$

这就是 **softmax 函数**。分子是 action a 的 exp(logit),分母 Z = Σ_a' exp(logit_{a'}) 是 partition function,确保 Σ_a π_θ(a|o_i) = 1。

### 7.3 Failure probability 解释(公式 7, 8)

$$
P_{\text{failure}}(a) = \pi_\theta(a | o_i), \quad \sum_{a \in \mathcal{A}} P_{\text{failure}}(a) = 1
$$

**核心 interpretive move**:把外层 RL agent 的 action probability distribution **重新解释** 为被测 policy 在该 observation 下、各个 failure mode(每个 action 对应一种环境扰动 = 一种 failure mode)发生的概率。

这里有一个 subtle issue 值得深挖:

- P_failure(a) 高 → 外层 agent 觉得"在这个 o_i 下,选 action a 最容易让被测 policy 失败"。
- 这个解读的强度依赖于 **RL agent 是否 well-trained**。如果 PPO 还没收敛,π_θ 的分布可能更多反映 exploration 而不是 failure likelihood。
- 而且 PPO 的输出本身 **不是 calibrated probability**——它只是 "preference strength"。要严格得到 P(fail | env),需要做 temperature scaling 或额外的 calibration。文中没有这一步,所以 P_failure(a) 应该理解成 **relative ranking**,而不是绝对概率。

### 7.4 Logits variance 与 distribution shape(Appendix D)

作者用 logits 的 variance 解释分布形态:

- **Low variance logits**(f_θ(o_i)_a ≈ f_θ(o_i)_a' for all a, a') → softmax 接近 **uniform distribution**。发生在 poorly trained model 或 policy 没有足够信息区分 actions 的场景。
- **High variance logits**(某些 logit 显著大于其他)→ softmax 输出 **skewed distribution**,集中在几个 actions 上。表示 policy 有强 preference,说明它能区分 actions。

这个 intuition 用来解释 Figure 4 的雷达图:poorly performing model 在所有 failure mode 上 P_failure 都接近 uniform,因为它在多个场景都失败(无差别脆弱);stronger model 在某些 mode 上 P_failure 集中,只在特定场景失败(定向脆弱,容易 fix)。

---

## 8. 实验:input modality 与 robustness 关系

### 8.1 被测模型(Section 5.1)

| Model | Input Modality | 描述 |
|---|---|---|
| Image(n) | n 张 RGB | 纯视觉,无 proprioception |
| Image + Proprioceptive | RGB + joint angles / positions | 视觉 + 本体感觉 |
| Image + Language | RGB + natural language instruction | 视觉 + 语言指令 |

任务:stacking, pushing, reaching(Robosuite)。

**为什么选这三种 modality**:作者想探究 **不同 input channel 对 OOD robustness 的贡献**。直觉上,proprioceptive 提供 robot 自身状态(对环境扰动相对 robust),language 提供 task grounding(可能让 model 更抽象、更 robust,也可能让 model overfit 到特定 phrasing),pure image 最容易 overfit 到视觉 distractor。

### 8.2 Figure 3 — 跨 model 的 FM 概率堆叠条形图

每个 failure mode 上的 stacked bar 表示各 model 对该 failure mode 的 P_failure 贡献。这能直接看出 **哪个 failure mode 是哪个 model 的"软肋"**。例如某个 mode 上 Image(n) 的 bar 特别高,说明纯视觉 model 在该场景下尤其脆弱;Image + Proprioceptive 在同 mode 上 bar 较低,说明 proprioception 帮 model 抵抗了这类扰动。

### 8.3 Figure 4 — 雷达图(关键 visualization)

每个 model 一个雷达图,雷达的 axes 是不同环境配置(Red Cube, Green Table, Blue Table 等),每条 axis 上的值是 P_failure(a) 在该配置下的概率。雷达图的形状直接告诉你 model 的 vulnerability profile:

- **接近圆形(uniform)**:model 在所有配置下都脆弱,poorly performing。
- **尖锐凸起**:model 只在某些特定配置下脆弱,targeted vulnerability,容易 fix。

文中给出 entropy 数值的对比(Section 5.2):

| Model | Entropy | 含义 |
|---|---|---|
| Batch-Constrained Deep Q-learning | 2.79 | 高熵 → 失败均匀分布,广泛脆弱,难以定向修复 |
| Hierarchical Behavioral Cloning | 2.11 | 低熵 → 失败集中,问题可预测,容易 fix |

**Entropy 公式**:H(π_θ) = -Σ_a P_failure(a) log P_failure(a),最大值为 log|A| = log 9 ≈ 2.197(纯 uniform)。所以 2.79 这个数字看起来超过 log 9,可能是因为 action space size 不止 9,印证了我之前对 9 vs 20 的疑惑——真实 action space 可能是 20 或更多,H_max = log(20) ≈ 3.0,2.79 接近 uniform。

### 8.4 Figure 5 — Heatmap

行是 model,列是 environment,intensity 是 normalized P_failure。darker = more likely to fail。直接看哪些 cell 是深色:就能找出"model M 在 environment E 下特别容易失败"这种 specific vulnerability pair。这对 targeted data augmentation / curriculum learning 很有用——你知道去 collect 哪种环境的 demo 来 retrain。

### 8.5 Figure 6 — Grouped bar chart

每个 failure mode 一组 bar,组内每个 bar 代表一个 model 的 P_failure。直接比较 model-vs-model 在同一 failure mode 下的相对脆弱性。

---

## 9. 与相关工作的联系与联想

### 9.1 Uncertainty Quantification 在 robot learning 中的位置

robotics 里 uncertainty 分 **aleatoric**(known unknowns,sensory noise、actuation noise,用 probabilistic model 表征,如 Thrun's Probabilistic Robotics https://dl.acm.org/doi/10.1145/504714.504721)和 **epistemic**(unknown unknowns,model 没见过的,Kendall & Gal https://arxiv.org/abs/1703.04977 区分了这两种)。RoboFail 本质是在 **主动构造 epistemic gap**:它去找那些 policy 没见过的、会导致失败的环境配置,所以它是一种 **directed OOD probing**,而不是被动 OOD detection。Senanayake 的 survey https://arxiv.org/abs/2405.03164 给了 robot learning uncertainty 的全面综述。

### 9.2 与 OOD detection 的对比

Nitsch et al. https://arxiv.org/abs/2104.02998 做 automotive perception OOD without extra training cost;Wilson et al. SAFE https://arxiv.org/abs/2303.02095 用 sensitivity-aware features 提升 OOD object detection;PyTorch-OOD library(Kirchheim et al. https://arxiv.org/abs/2104.05506)实现各种 OOD method。这些方法都是 **reactive**(看到 OOD input 才警报),RoboFail 是 **proactive**(主动去找会触发 failure 的 input)。两者互补:reactive OOD 用于 runtime,proactive failure discovery 用于 training/evaluation phase。

### 9.3 与 failure detection in generative policy 的关系

最近一批工作用 VLM 来 detect generative policy failure:Agia et al. "Unpacking failure modes" https://arxiv.org/abs/2410.04640 runtime 监控 consistency 和 progress;Duan et al. AHA https://arxiv.org/abs/2410.00371 用 VLM reason over failure。这些是 **post-hoc detection**(policy 已经在跑,看它有没有走偏),RoboFail 是 **ante-hoc discovery**(在 deployment 前主动找 failure case)。可以想象一个 pipeline:RoboFail 找到 failure-prone configurations → 在这些 configurations 上收集 demos → retrain policy → AHA 在 runtime 做 monitoring。这个组合会很 powerful。

### 9.4 与"Failures are fated"系列

作者自己的前作 Sagar et al. https://arxiv.org/abs/2410.16738 把 RL-based failure discovery 用在 vision-language model 上(找 image misclassification、text summarization error、image generation bias)。RoboFail 是把这个 idea 转移到 robot manipulation,核心 insight 一样:**用 RL learner 当 fuzzer,攻击一个 black-box learner**。这是作者实验室的 research thread,值得关注。

### 9.5 与 PAGER(regression failure analysis)

Thiagarajan et al. PAGER https://arxiv.org/abs/2309.10977 做 deep regression model 的 failure analysis,思路类似但 for regression。RoboFail 是 for sequential decision making(manipulation policy),更复杂,因为 failure 不是单点 input → output,而是 trajectory-level。

### 9.6 与 stress testing AV perception

Delecki et al. IROS 2022 https://arxiv.org/abs/2202.05807 用 RL stress test AV perception。Corso et al. JAIR survey https://arxiv.org/abs/2008.06270 综述 black-box safety validation。RoboFail 是这个 family 在 manipulation 的 instance,但加了 **probabilistic interpretation**(把 RL policy 输出当 failure mode 概率),这是它的 contribution 之一。

### 9.7 与 red teaming LLMs

Hong et al. curiosity-driven red-teaming https://arxiv.org/abs/2402.19464 用 curiosity-driven RL 找 LLM 的 prompt injection / jailbreak。RoboFail 用 PPO 找 manipulation policy 的 brittle configuration。同一个 idea 在不同 domain:用 RL learner 当 adversary 去 attack 另一个 learner。这种 "learning to break learning" 是 ML safety 的核心 pattern。

### 9.8 与 generalization benchmark

Colosseum(Pumacay et al. https://arxiv.org/abs/2402.08191)是 manipulation generalization benchmark,系统化 perturbation 测试。RoboFail 不是 benchmark,是一个 **主动找 perturbation 的 method**,可以看作 Colosseum 的"主动学习版"——不用 grid search perturbation,让 RL agent 自己选。

### 9.9 与 curriculum learning / domain randomization

OpenAI dexterous hand(Andrychowicz et al. https://arxiv.org/abs/1910.07113)用 domain randomization 训练 robust policy。RoboFail 是 reverse direction:不训练 robust policy,而是 **发现 policy 在哪里不 robust**。两者可以结合:RoboFail 找到脆弱配置 → 这些配置加入 domain randomization training set → retrain。

### 9.10 与 Bayesian IRL / IRL

Jeon et al. Bayesian GAIL https://arxiv.org/abs/1806.00767、Brown et al. Bayesian reward inference https://arxiv.org/abs/2006.12633、Ramachandran & Amir Bayesian IRL 这些用贝叶斯方法 infer expert 的 reward。RoboFail 是 inverse direction:它不 infer reward,而是 infer **failure condition**。但从某种意义上,RL agent 学到的 policy 隐含一个 "failure likelihood function" 的 representation,可以理解为对被测 policy 弱点的 model。

### 9.11 与 statistical safety certificates

Farid et al. https://arxiv.org/abs/2202.05894、Ren & Majumdar https://arxiv.org/abs/2103.04995 提供 vision-based control 的 statistical failure bounds 和 distributionally robust policy learning;Vincent et al. https://arxiv.org/abs/2310.14289 给 imitation learning full-distribution generalization bound。这些给 **理论保证**,RoboFail 给 **empirical 发现**,互补。

### 9.12 与 formal methods safety

Tůmová et al. http://aaai.org/ocs/index.php/CAA/CAA13/paper/view/6167 minimum-violation LTL planning 是 formal methods 角度的 safety。RoboFail 是 learning 角度,scalable 但没 formal guarantee。两套思路可以融合:RoboFail 找到的 failure mode 作为 LTL violation candidate,formal methods 来 verify。

### 9.13 一个略 wild 的联想:与 generative world model

如果我们有一个 learned world model(Dreamer 系列 https://arxiv.org/abs/1912.01603、PlaNet https://arxiv.org/abs/1811.04551),可以直接在 latent space 里搜索 failure,不用跑真实 sim。RoboFail 每个 PPO step 都要 rollout 被测 policy,这是 expensive bottleneck(尤其被测 policy 是 VLA / diffusion policy 时,单步 inference 几百 ms)。用 world model 替代 sim 可以让 RoboFail 加速几个数量级。

### 9.14 另一个 wild 联想:与 coverage-guided fuzzing

software fuzzing(AFL https://lcamtuf.coredump.cx/afl/)的 coverage-guided 思路可以借鉴:RL agent 不光 maximize failure,还要 maximize **failure mode diversity**——找到尽可能多种不同的 failure pattern,而不是反复触发同一个 failure。这需要在 reward 里加 diversity term,比如把 P_failure(a) 的 entropy 加入 reward,鼓励 agent 探索不同 failure mode 而不是 overfit 到一个。

### 9.15 与 active learning

RoboFail 和 active learning 同构:被测 policy 是 expensive-to-evaluate black-box,我们要 efficiently sample input space。Bayesian Optimal Experimental Design 可以替代 PPO,但 RL 的优势是能处理 sequential decision(连续多个 action 决定最终 configuration)。

---

## 10. Limitations / 我的 critique

1. **Discrete, predefined action space**:action 只能从 predefined set 中选(改颜色、改位姿等),不能发现"未定义"的 failure mode。比如 RL agent 不会发现"添加一个新的 distractor object"作为 failure cause,除非 action space 里就有这个选项。这是最大的 limitation。Future work 提到 expand action space,但如何让 action space 自身 expand(programmatic perturbation、VLM-generated perturbation)是个 open question。

2. **P_failure 的 calibration**:前面提过,π_θ 输出不是 calibrated probability。要做严格 P(fail | env) 估计,需要 temperature scaling、Platt scaling、或 Monte Carlo 校准。这点文中没处理。

3. **Sim-to-real gap of failure modes**:RL agent 在 sim 里找到的 failure 配置,可能在 real world 不会出现,或者反过来 real-world 失败的配置 sim 里没有。需要 sim-to-real 的 failure transferability 验证。

4. **被测 policy rollout 的 compute cost**:每个 PPO step 都要跑被测 policy 一整个 episode。如果被测 policy 是 VLA(每秒几 Hz)、diffusion policy(每个 action 几百 ms),整个 RoboFail 训练极慢。可以想到 batch rollouts、asynchronous sampling、或用 lighter surrogate。

5. **Binary reward 信息量少**:reward 只有 fail/success,信息量 1 bit。可以用 task progress(Agia et al. 的 progress monitoring https://arxiv.org/abs/2410.04640)、Q-value estimate、partial credit 等丰富 signal。

6. **PPO 探索效率**:在 reward 极稀疏时,PPO 也可能 collapse 到 trivial policy(总选一个 action)。文中没 report PPO training curve / success rate of finding failure,无法判断 PPO 是否真 converge 到好的 failure-discovering policy。建议作者加这个 ablation。

7. **Action space size 的不一致**:Section 3.1 提到 20 actions,Appendix B 说 action space size 9。这个不一致要 fix。

8. **Observation 是被测 policy 的还是外层 agent 的**:文中 state 是"current observation from the environment, including visual data of the robot and workspace after the agent's action"。但外层 agent 是否看到被测 policy 的内部 state(Q-value、intermediate feature)?目前看是 black-box,只能看 final result。如果让外层 agent access 被测 policy 的 intermediate representation,可以更精准 attack(类似 white-box adversarial attack)。这是 black-box vs white-box trade-off。

9. **Entropy 解释的方向性问题**:作者说高 entropy = 广泛脆弱,低 entropy = 集中脆弱,容易 fix。但还有第三种 case:policy 完美 robust,RL agent 找不到任何 failure,这时 π_θ 也接近 uniform(因为没什么 preference)。所以 uniform 分布不一定意味着"全面脆弱",也可能意味着"无懈可击"。要区分这两种 case,需要看 **P_failure 的绝对量级**,而不只是分布形状。作者用 softmax 归一化,丢失了绝对量级信息——所有 P_failure 加起来永远等于 1,即使 RL agent 一个 failure 都没找到,分布也"看起来"正常。这是个 subtle issue。建议在 PPO 的 raw value function V(s) 上加一层分析:V(s) 高表示 RL agent 觉得能找到 failure,V(s) 低表示找不到。

10. **单 observation 下的 distribution 而非 trajectory-level**:P_failure(a|o_i) 是 single observation 下的,但 failure 通常是 trajectory 累积的结果。可能需要 marginalize over observations 或考虑 multi-step action sequences。

---

## 11. 对 manipulation policy 设计的 implication

从实验结果可以提炼几个 actionable insight:

- **Pure image policy 最脆弱**:没有 proprioception 或 language grounding,policy 容易被 visual distractor 骗。建议至少加 proprioception。
- **Language grounding 提供抽象层**:language instruction 让 policy 学到 task 的 semantic goal,而不是 visual cue,对 visual perturbation 更 robust。但 language 本身可能 overfit 到特定 phrasing,需要 paraphrase augmentation。
- **Proprioception 提供自指 anchor**:robot 自身状态对环境扰动相对不变,给 policy 一个 stable reference。
- **Failure mode 的 entropy 是 policy health 的一个指标**:训练后跑 RoboFail,看 P_failure 的 entropy。低 entropy = 集中脆弱,容易 targeted fix;高 entropy = 广泛脆弱,需要重新设计 model 或加 data。

---

## 12. 一个 toy mental model 总结 RoboFail

把 RoboFail 想象成一个 **"对抗性教练"**:

1. 教练(外层 RL agent)看着学生(被测 policy)做任务。
2. 教练能轻微改动训练场(改球颜色、改桌子位置)。
3. 教练的目标:让学生做错。
4. 教练通过 PPO 学习"哪种改动最有效"。
5. 训练完后,教练手里的"作弊 playbook"(softmax distribution over perturbations)就成了学生的"弱点报告"。
6. 你拿这个报告去给学生针对性补课。

这个 metaphor 抓住了 framework 的 essence,也暴露了 limitation:教练只能用预定义的"作弊手法"(discrete actions),不能发明新手法。要让学生彻底变强,得让教练也能"创造性作弊"(programmatic / continuous perturbation generation)。

---

## 13. 一些可能的扩展方向

1. **Continuous action space**:把 discrete perturbation 换成 continuous(物体位姿连续变化、颜色 RGB 连续变化、光照角度连续变化),用 SAC https://arxiv.org/abs/1801.01290 或 PPO-continuous。
2. **Programmatic perturbations**:用 DSL 或 LLM 生成 perturbation program,action space 是 program space(类似 genetic programming)。
3. **VLM-conditioned perturbation**:用 VLM 描述"什么样的场景会让 policy 困惑",VLM 生成 perturbation 配置,RL agent refine。
4. **Coverage-guided**:在 reward 里加 diversity term,鼓励发现不同的 failure mode,而不是 overfit 到一个。
5. **World model acceleration**:用 learned world model 替代 sim,在 latent space 搜索 failure,大幅加速。
6. **White-box attack**:让 RL agent access 被测 policy 的 intermediate feature / Q-value,做更精准 attack。
7. **Multi-agent**:多个 RL agent 协作找不同 type 的 failure,类似 multi-agent red teaming。
8. **Self-supervised failure discovery**:不用 task success 作为 reward,而用 "policy 输出 entropy 变化"、"feature distribution shift" 等代理信号,可能更 general。
9. **Failure-aware policy retraining**:把 RoboFail 找到的 failure configurations 当作 hard examples,加入 training set retrain policy(类似 hard negative mining)。
10. **Probabilistic safety certificate**:把 P_failure(a) 和 statistical methods(Farid et al. https://arxiv.org/abs/2202.05894)结合,给出 P(fail | env) 的 confidence interval,作为 deployment safety certificate。

---

## 14. 与你(Karpathy)工作的几个具体联想点

考虑到你的背景,几个值得深思的连接:

1. **micrograd / nanoGPT 视角**:RoboFail 的 PPO 实现可以在几百行内复现,适合做教学示例,讲 RL agent 如何 attack 另一个 NN。这种 "NN 攻击 NN" 的可视化教学价值很高。

2. **"Software 2.0" 视角**:RoboFail 是 Software 2.0 时代的 fuzz testing。Software 1.0 时代我们用 AFL、libFuzzer 找 program crash;Software 2.0 时代我们用 RL agent 找 NN 的 brittle configurations。这是个 deep pattern 值得在 blog 里展开。

3. **Dataset quality 视角**:你常说"data is all you need"。RoboFail 揭示了 **dataset 的盲区**:training distribution 之外的 configuration 才是 policy 的真正敌人。这印证了 active data collection 的重要性——不是更多 data,而是更 **targeted** data(找 policy 失败的 case 来 label)。

4. **Recipe training 视角**:你最近强调 long-horizon RL training 的 recipe 重要性。RoboFail 的 PPO recipe(lr=3e-4, n_steps=2048, batch=64, epochs=10, no entropy bonus)是个 specific config,这种 config 的 reproducibility 和 sensitivity 值得 ablation。

5. **Interpretability 视角**:RoboFail 本质是一种 **behavioral interpretability**——不打开 black box,而通过它的 failure boundary 来理解它。这和 mechanistic interpretability 互补,一个看 inside,一个看 outside(boundary)。

---

## 15. 总结

RoboFail 的 contribution 有三层:

1. **Formulation**:把 failure mode discovery formulates 成 MDP,用 RL agent 主动探索。
2. **Probabilistic interpretation**:把 RL agent 的 action distribution 解释为 failure mode 概率分布,提供 ranking 和 entropy 分析。
3. **Empirical insight**:不同 input modality(image / proprioceptive / language)对 OOD robustness 的贡献差异,提供 actionable 建议。

它的 limitation 在 discrete action space、calibration、compute cost、sim-to-real transferability 上。但 framework 本身有很强的 extensibility,可以与 world model、VLM、programmatic perturbation、coverage-guided fuzzing 等结合,发展空间大。

这篇 paper 的核心 idea——"用 RL learner 当另一个 learner 的对抗者"——是个 deep pattern,在 ML safety、red teaming、stress testing 中反复出现。RoboFail 是这个 pattern 在 robot manipulation 上的 clean instantiation,值得作为 baseline 被未来工作引用和扩展。

---

## 关键 reference links

- **PPO**: https://arxiv.org/abs/1707.06347
- **GAE**: https://arxiv.org/abs/1506.02438
- **Robosuite**: https://robosuite.github.io/ 和 paper https://arxiv.org/abs/2009.12293
- **robomimic**: https://robomimic.github.io/ 和 paper https://arxiv.org/abs/2103.14090
- **RT-1**: https://arxiv.org/abs/2212.06817
- **RT-2**: https://arxiv.org/abs/2307.15818
- **Octo**: https://arxiv.org/abs/2405.12213
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **Failures are fated (Sagar et al.)**: https://arxiv.org/abs/2410.16738
- **AHA (Duan et al.)**: https://arxiv.org/abs/2410.00371
- **Unpacking failure modes (Agia et al.)**: https://arxiv.org/abs/2410.04640
- **Colosseum**: https://arxiv.org/abs/2402.08191
- **Curiosity-driven red-teaming**: https://arxiv.org/abs/2402.19464
- **PAGER**: https://arxiv.org/abs/2309.10977
- **Black-box safety validation survey (Corso et al.)**: https://arxiv.org/abs/2008.06270
- **Stress testing AV perception (Delecki et al.)**: https://arxiv.org/abs/2202.05807
- **Kendall & Gal uncertainty**: https://arxiv.org/abs/1703.04977
- **Senanayake uncertainty survey**: https://arxiv.org/abs/2405.03164
- **OpenAI dexterous hand (domain randomization)**: https://arxiv.org/abs/1910.07113
- **PyTorch-OOD**: https://github.com/kirchheimart/pytorch-ood 和 paper https://arxiv.org/abs/2104.05506
- **SAC (for continuous action space extension)**: https://arxiv.org/abs/1801.01290
- **Dreamer (world model)**: https://arxiv.org/abs/1912.01603
- **Statistical failure bounds (Farid et al.)**: https://arxiv.org/abs/2202.05894
- **Distributionally robust policy (Ren & Majumdar)**: https://arxiv.org/abs/2103.04995
- **Bayesian IRL (Ramachandran & Amir)**: https://arxiv.org/abs/1511.06379 (republished)
- **Safe imitation learning (Brown et al.)**: https://arxiv.org/abs/2006.12633
- **Shared backbone vs separate (Kostrikov et al.)**: https://arxiv.org/abs/2006.05990

希望这些拆解和联想能帮你 build 出对 RoboFail、对"RL-as-attacker"这一 broader pattern、以及对 manipulation policy robustness 评估方法的 intuition。如果某个公式或某个 connection 你想深挖,告诉我具体点,我可以展开更细的 derivation 或对比实验设计。
