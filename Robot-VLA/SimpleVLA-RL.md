---
source_pdf: SimpleVLA-RL.pdf
paper_sha256: c3f9c608ba58bb274dc6d093e78836603edfe3475278be19ff96b91b0654a3e5
processed_at: '2026-08-12T06:34:11-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SimpleVLA-RL

好，之前那版太 technical 了，这次用大白话重讲一遍，核心 idea 其实非常直观。

---

## 一句话版本

**让机器人不看示范、只看"做没做成功"这个结果，自己在仿真里瞎试，试出来的策略比看 500 条人类操作视频学的还好。**

---

## 现在教机器人做事的方法有什么问题

当前 VLA 的训练范式是两步走：

1. **Pretrain**：喂一堆 internet 上的图文 + 机器人操作视频，让 model 理解"世界长什么样"
2. **SFT（Supervised Fine-Tuning）**：找人来 teleop 操作机器人，录下"看到这个画面就做这个动作"的轨迹，让 model 模仿

SFT 这步有两个死穴：

**死穴一：数据太贵。** 录一条机器人操作轨迹，得有真实的机械臂、有人熟练操作、有精心设计的 task scenario。Open X-Embodiment 那种百万级数据集是几十个 lab 一起搞才攒出来的。想 scale SFT，trajectory 数量跟不上。

**死穴二：只会模仿，不会变通。** SFT 本质是"看到画面 A → 做动作 A"，一旦遇到没见过的物体、没见过的空间配置，model 就懵了。论文里做了个实验：9 个 task 训练，留 1 个 unseen task 测试，SFT 训练越久，unseen task 成功率反而掉到 0%——典型的 catastrophic forgetting。

参考：
- Open X-Embodiment: https://arxiv.org/abs/2410.12247
- OpenVLA: https://arxiv.org/abs/2406.09246

---

## R1 给的启发

DeepSeek-R1（https://arxiv.org/abs/2501.12948）证明了一件事：**不教 model 怎么推理，只告诉它"答案对不对"，它自己能摸出推理路径**。甚至能涌现 SFT 数据里没有的 reasoning pattern（self-verification、backtracking 那些）。

那对应到机器人上就是：**不教 model 怎么操作，只告诉它"任务做没做成功"，让它自己摸出动作策略**。会不会也能涌现 SFT 数据里没有的操作模式？

SimpleVLA-RL 的回答是：会，而且出现了叫 "pushcut" 的现象（后面细讲）。

---

## 从 LLM RL 搬到 VLA RL，为什么不能直接搬

LLM 的 RL 和 VLA 的 RL 看起来都是"policy 生成 token、env 给 reward"，但实际有四个本质差异。这几个差异决定了不能把 LLM 的 RL 代码原样跑在 VLA 上。

### 差异一：State 不一样

LLM 的 state 就是 prompt + 已生成的 token，生成过程中 state 完全由 model 自己控制。

VLA 的 state 是 `(视觉图像, 机器人本体状态, 语言指令)` 三件套。关键区别：**机器人每发一个 action，物理世界就变了，下一帧图像就不一样了**。你 push 一下杯子，杯子位置变了，camera 拍到的画面也变了，model 得重新看、重新决定下一步。

所以 VLA 的 rollout 不能像 LLM 那样"一次 forward 出一整条 trajectory"，得 closed-loop 一步一步和 environment 交互。这导致采样慢得多。

### 差异二：Action 不一样

LLM 的 action 是从 vocabulary 里选 token，天然有概率分布 $\pi_\theta(a_t|s_t)$，PPO/GRPO 直接能用。

VLA 的 action 是 7 维连续向量（6-DoF pose + gripper 开合），生成方式有三种：
- **Token-based**：把连续 action 离散化成 token，和 LLM 一样 next-token prediction
- **Diffusion**：在 latent space 上做去噪（RDT-1B、π0 这条路线）
- **MLP regression**：MLP 直接回归连续值（OpenVLA-OFT 官方版）

只有 token-based 天然能给出 $\pi_\theta(a_t|s_t)$ 的概率，PPO/GRPO 的 importance sampling ratio 才有得算。Diffusion 的 likelihood 极难估计，MLP regression 是 deterministic 的没有 exploration。

所以 SimpleVLA-RL 把 OpenVLA-OFT 官方版的 MLP head 换成了 LLaMA2 的 LM head + cross-entropy loss，从头 SFT 了一遍。这是第一个关键 design choice。

### 差异三：Reward 不一样

传统 robot RL 用 dense shaped reward：`r = -distance_to_goal + 0.1*grasp_success - 0.01*action_smoothness` 这种手搓 reward。问题是不可迁移、要 per-task 调。

SimpleVLA-RL 跟 R1 学，用最简的 binary reward：

$$R = \begin{cases} 1, & \text{任务完成} \\ 0, & \text{任务没完成} \end{cases}$$

整条 trajectory 成功就给 1，失败就给 0，这个 reward 均匀 propagate 到 trajectory 里所有 action token 上。

好处：task-agnostic，scale 起来不用人盯每个 task 设计 reward。
坏处：reward 信号极稀疏，credit assignment 模糊（哪个 token 该为成功负责？），cold-start 难。

### 差异四：Rollout 不一样

LLM 的 rollout 是 autoregressive 生成，一次 forward 出一条 trajectory，env 不参与中间过程。

VLA 的 rollout 是 closed-loop，每发一个 action chunk（k 个连续 action）就要 query 一次 environment 拿新观测。这导致：
- 采样慢：每步要 render 物理仿真、跑 forward dynamics
- 采样贵：simulator 通常单线程

SimpleVLA-RL 的工程解法是 **parallel multi-environment rendering**——spawn N 个 simulator 实例到进程池里，policy 一次性给 N 个 state 出 N 个 action chunk，batch submit 给环境 step。这就是 Listing 1 那段伪代码干的事：

```python
for t in range(max_steps):
    actions = policy.generate(states, temperature=1.0)  # 一次性出 N 个 action
    states, dones = env_process_pool.submit(envs.step, actions)  # batch 环境步进
    # 过滤掉已完成的 env
```

参考：veRL framework https://arxiv.org/abs/2409.19256

---

## GRPO 的核心 trick

GRPO 来自 DeepSeekMath（https://arxiv.org/abs/2402.03300），核心 trick 是 **干掉 value network**。

PPO 原版需要训一个 value network $V_\phi(s)$ 来估计 baseline，算 advantage $A = R - V_\phi(s)$。这个 value network 在高维 multimodal state（图像 + proprioception + 语言）上极难训好。

GRPO 的做法：**对同一个 state 采样 G 条 trajectory，用 group 内 reward 的 mean/std 当 baseline**：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\})}{\text{std}(\{R_i\})}$$

- $R_i$ 是第 $i$ 条 trajectory 的总 reward（0 或 1）
- mean / std 都是 group 内 G 条 trajectory 的统计量
- $\hat{A}_i$ 是第 $i$ 条的归一化 advantage

直觉：组内 G 条 trajectory 谁比平均好谁 advantage 正，谁比平均差谁 advantage 负。不用训 value network。

完整 objective（带 PPO clip + KL penalty）：

$$J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \min(r_{i,t}(\theta)\hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

- $r_{i,t}(\theta) = \pi_\theta(a_{i,t}|s_{i,t}) / \pi_{\theta_{\text{old}}}(a_{i,t}|s_{i,t})$ 是新旧 policy 在同一 action 上的概率比
- $\epsilon$ 是 clip ratio（限制 policy 更新幅度）
- $\beta$ 是 KL penalty 系数（让 policy 不要离 reference policy 太远）

---

## 三个让探索更强的 trick（§3.3，方法核心）

paper 的方法部分主要就这三个 trick，都是从 LLM RL 搬过来的，但在 VLA 上效果尤其大（+10~15%）。三个 trick 互相耦合，缺一不可。

### Trick 一：Dynamic Sampling

问题：GRPO 的 advantage 靠 group 内 reward spread 估计。如果一个 group 8 条 trajectory 全成功（reward 全 1）或全失败（reward 全 0），std = 0，所有 advantage = 0，**gradient 为 0，啥也学不到**。

VLA 上这个问题尤其严重——因为 manipulation 成功率天然低，base model 经常 8 条全失败。

解法：rollout 阶段发现某个 group 全成功或全失败，**丢弃这个 group，重新采样**，直到 batch 里所有 group 都是 mixed outcome（有成功有失败）。

数学约束就是 paper Eq. 10：

$$0 < |\{\text{成功的 trajectory}\}| < G$$

人话：**"一批实验全成功或全失败，学不到东西，重做"**。

### Trick 二：Clip Higher

PPO 的 clip 是对称的 $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$。问题在于：上界 1.2 会限制低概率 token 涨概率的速度。

举例：某个 push action 初始概率 0.01，ratio 即使涨到 1.2 上限，概率也才到 0.012，还是几乎不会被采到。这种"低概率但实际有效的 action"被 clip 卡死了。

DAPO（https://arxiv.org/abs/2503.14476）的解法：上界调高到 1.28（$\varepsilon_{\text{low}} = 0.2, \varepsilon_{\text{high}} = 0.28$），让低概率好 token 能涨得更快。

人话：**"允许以前觉得不靠谱的动作涨得更快"**。

在 VLA 上这特别有用，因为 action 分布是 long-tail 的——push 这种动作初始概率极低但实际有效。这也是 pushcut 现象能涌现的机制基础。

### Trick 三：Higher Rollout Temperature

LLM RL 里采样温度一般用 1.0，SimpleVLA-RL 提到 1.6。高温让 softmax 分布更平坦，采样到 low-probability token 的几率变高。

VLA 上温度太低会导致 policy 早早 collapse 到 grasp-move-place 这一条 mode（因为 SFT 数据里全是这个 pattern）。高温让 policy 偶尔去试 push 这种 out-of-distribution action，才有机会命中 reward=1，然后 Clip-Higher 把这个 token 的概率推高。

人话：**"让机器人偶尔瞎试，试到好的就记住"**。

**三个 trick 的协同关系**：Dynamic Sampling 保证有 gradient signal，High Temperature 保证好 token 能被发现，Clip-Higher 保证好 token 发现后能被强化。缺一不可。

参考：
- DAPO: https://arxiv.org/abs/2503.14476
- POLARIS（temperature scheduling）: https://hkunlp.github.io/blog/2025/Polaris

---

## 实验结果：三个最 striking 的发现

### 发现一：1 条 demo + RL 打 500 条 demo 的 SFT

LIBERO 上做的实验，这是 paper 最有 scaling law 意味的结果。

| Setting | Avg Success Rate |
|---------|------------------|
| 1 条 demo SFT | 48.9% |
| **1 条 demo SFT + RL** | **96.9%** |
| 500 条 demo SFT | 91.0% |
| 500 条 demo SFT + RL | 99.1% |

**1 条 demo + RL（96.9%）超过 500 条 demo 的 SFT（91.0%）**。LIBERO-Long（长程任务）更夸张：1 条 demo SFT 只有 17.3%，加上 RL 直接拉到 91.7%。

人话：**"看一条视频 + 自己在仿真里试错，比看 500 条视频模仿学得还好"**。

这条曲线告诉我们：SFT 的 data efficiency 极差，500 条 demo 还不如 1 条 demo + RL。RL 是把"看 demonstration 学动作"换成"在仿真里自己试错学任务"，env 提供的 trial-and-error signal 比 cross-entropy on expert action 强得多。

这和 Karpathy 你讨论 R1 时的直觉一致：**reasoning 不是被 SFT 教出来的，是被 reward 逼出来的**。这里 manipulation strategy 也是同理——不是被 demo 教出来的，是被 success reward 逼出来的。

### 发现二：RL 超过 π0 接近 20 个点

RoboTwin 2.0 双臂 benchmark，按 task horizon 分四档：

| Horizon | π0 | RDT | OpenVLA-OFT | **+ SimpleVLA-RL** |
|---------|-----|-----|-------------|---------------------|
| Short | 45.5 | 24.5 | 21.3 | 64.9 |
| Medium | 58.8 | 47.8 | 47.1 | 72.5 |
| Long+Extra | 43.3 | 27.8 | 46.5 | 69.0 |
| **Overall** | **49.2** | **33.3** | **38.3** | **68.8** |

π0 用了 Physical Intelligence 的私有海量 pretraining data，SimpleVLA-RL 只用 OpenVLA-OFT（open-source）+ RL，反超接近 20 个点。

Long-Horizon 任务上优势尤其明显。Extra-Long（466-637 step）任务 RL 涨了 +11.1 / +18.7。

直觉：**long-horizon 是 SFT 的死穴**。demo trajectory 长，SFT 的 cross-entropy loss 会让 model 每一步都去 mimic expert，但任何一步小误差会指数级累积导致 trajectory 失败。RL 直接 optimize trajectory-level success，自然更鲁棒。

### 发现三：Sim-to-Real 大幅提升

RoboTwin 2.0 的 4 个 task 迁移到真实 Agilex Piper 双臂，训练完全用仿真，real world 零样本测试。

| Task | RDT | OpenVLA-OFT | **+ SimpleVLA-RL** |
|------|-----|-------------|---------------------|
| Stack Bowls | 60.0 | 38.0 | 70.0 |
| Place Empty Cup | 4.0 | 2.0 | 10.0 |
| Pick Bottle | 10.0 | 0.0 | 14.0 |
| Click Bell | 20.0 | 30.0 | 60.0 |
| Avg | 23.5 | 17.5 | **38.5** |

RL 把 sim-to-real 成功率从 17.5% 拉到 38.5%。Pick Bottle 这种对精度要求极高的任务（gripper 对齐稍偏瓶子就掉），SFT 直接 0%，RL 涨到 14%。

反直觉的地方：通常认为 RL 是 high-variance 的，但这里 RL 训出来的 policy action precision 反而更高。原因在于 binary reward + GRPO 的设定下，policy 被"reward=1"吸引到更精准的 trajectory 上。

参考：RoboTwin 2.0 https://arxiv.org/abs/2506.18088

---

## Pushcut：机器人自己发明的新动作（§6.1，paper 最有趣的部分）

这是 paper 的 "Aha Moment"，和 R1 的 Aha Moment 完全同构。

RoboTwin 2.0 的 "Move Can Pot" 任务：把罐子移到锅旁边。

- **Demonstration 数据**（SFT 学的）：grasp 罐子 → move 到锅旁边 → place
- **RL 训练后涌现的行为**：直接 push 罐子滑到锅旁边

"Place A2B Left/Right" 任务也一样——demo 是 grasp A → move → place，RL 学会直接 push A 到目标位置。

### 为什么会出现 pushcut？

我的理解：

1. **SFT data 的 bias**：人 teleop 时倾向于 grasp-move-place，因为对人来说 grasping 最自然。所以 SFT 学到的是"模仿人的操作偏好"，不是"解决任务的最优策略"
2. **Binary reward 的解放**：reward 只看"罐子有没有在锅旁边"，不管你怎么弄过去的。policy 有自由度去探索任何能拿到 reward=1 的 trajectory
3. **Exploration trick 的协同**：High Temperature 让 push 这种 low-probability action 偶尔被采样到，Clip-Higher 让它一旦拿到 reward 就快速涨概率，Dynamic Sampling 保证有效 gradient
4. **Push 比 grasp 简单**：grasp 需要精确对齐 gripper + 控制抓取力 + lift + move + place，每一步都可能失败。Push 只需要一个方向的力，成功率高得多。binary reward 下 policy 自然往简单策略收敛

人话：**"老师教的是抓起来搬过去，学生自己发现直接推过去更省事"**。

这和 R1 的 Aha Moment 是同构的——**reward signal 让 model 发现了 SFT 数据里不存在的、更高效的解题路径**。区别是 R1 涌现的是 reasoning pattern（self-verification、backtracking），这里是 motor pattern（push vs grasp）。

这暗示 RL 不只是 SFT 的"refinement"，而是能解锁 fundamentally 不同的 solution space。对 VLA 的 scaling 来说，这可能比堆更多 demo 数据更有价值。

---

## 失败模式：RL 不是万能的（§6.2）

这个实验很关键，揭示了 **RL 的 cold-start 问题**。

RoboTwin 2.0 上对比三种 base model：

| SFT 数据量 | SFT 成功率 | + RL 成功率 | Δ |
|-----------|-----------|------------|---|
| 0 trajectory | 0% | 0% | 0 |
| 100 trajectory | 7.3% | 25.4% | +18.1 |
| 1000 trajectory | 28.2% | 50.4% | +22.2 |

**0 trajectory SFT 的 base model 上 RL 完全失败**，所有 task 都是 0%。

直觉：GRPO 需要 group 内有 mixed outcome 才有 gradient。base model 0% 成功率，所有 rollout 都失败，Dynamic Sampling 会无限 retry 直到耗尽 budget。**Pure RL from scratch 在 VLA 上不可行**。

这说明：
- **SFT 提供"探索种子"**：哪怕只有 7% 成功率，也够 RL bootstrap 起来
- **RL 有 threshold**：base model 能力越强，RL 涨幅越大。100 条 demo 从 7.3% 涨到 25.4%（3.5x），1000 条 demo 从 28.2% 涨到 50.4%（1.8x），但绝对涨幅后者更大
- **Pretrain ≠ task capability**：OpenVLA-OFT 即使经过大规模 pretrain，zero-shot 在 RoboTwin 上还是 0%。pretrain 给的是 representation，不是 task prior

这和 Karpathy 你讨论 R1-Zero 时的观察一致——R1-Zero 能 from-scratch work 是因为 base model 已经有 strong reasoning prior（pretrain 阶段海量 CoT 数据）。VLA 的 pretrain 还远没到这个水平，所以 SFT bootstrap 是必需的。

---

## 和 LLM RL 的对应关系（一张表说清）

| LLM RL 那边 | VLA RL 这边 | 关键差异 |
|-------------|-------------|---------|
| R1-Zero (from-scratch RL) | SimpleVLA-RL | R1-Zero 能 from-scratch，VLA 需要 SFT bootstrap |
| GRPO | GRPO（一样的算法） | trajectory-level reward 在 VLA 上更 sparse |
| DAPO (Clip-Higher, 去 KL) | DAPO（直接搬过来） | 完全适用 |
| Dynamic Sampling | Dynamic Sampling | VLA 上更关键，因为成功率低 |
| Aha Moment | Pushcut | 都是 emergent behavior |
| Entropy collapse | VLA 也有，靠高温解决 | — |

paper 也提到了一些并行工作：
- RIPT-VLA（https://arxiv.org/abs/2505.17016）用 RLOO 算法
- VLA-RL（https://arxiv.org/abs/2505.18719）用 PPO
- TGRPO（https://arxiv.org/abs/2506.08440）用 Claude 3.7 当 reward model
- RFTF（https://arxiv.org/abs/2505.19767）用 value model 给 dense reward

SimpleVLA-RL 的差异化在于：(1) 完全 rule-based reward，没有 learned reward model；(2) 系统性验证 sim-to-real；(3) 发现 pushcut 现象。

---

## 局限和未来方向

Paper 自己承认的 + 我观察到的：

1. **只支持 token-based VLA**：diffusion policy（RDT-1B、π0）和 MLP regression VLA 暂时用不上这套方法。未来要想法估计 diffusion 的 likelihood（flow matching 那一套），或者换 policy gradient 形式
2. **Simulator-bound**：必须有 high-fidelity simulator 才能 rollout。real-world RL（ConRFT 那条路线 https://arxiv.org/abs/2502.05450）成本高得多
3. **Binary reward 的天花板**：对特别复杂的任务（需要 tool use、多阶段规划），sparse binary reward 探索效率太低。RFTF 用 value model 给 dense reward 是一个方向
4. **Cold-start 依赖 SFT**：RL 不是替代 SFT 而是补充。什么时候能像 R1-Zero 一样 from-scratch RL，是 open question
5. **Pushcut 的 reproducibility**：paper 只展示了两个 task 上的 pushcut，没说在更广泛 task suite 上的频率。emergent behavior 很难 systematize

未来值得关注的几条线：
- **Diffusion VLA + RL**：怎么把 token-based RL 迁移到 diffusion policy 上
- **Hierarchical RL**：long-horizon task 的 subgoal 分解
- **Real-world RL**：sim 和 real 的 gap 怎么弥合
- **Emergent behavior 的 systematization**：pushcut 是 isolated 现象还是 RL 的普遍性质

---

## 给 Karpathy 的总结性 intuition

**VLA 的 SFT 是在学"人怎么做"，RL 是在学"任务怎么做"。当人的操作偏好不是任务最优解时，RL 会发现 SFT 数据里不存在的、更高效的策略。**

这和 R1 在 reasoning 上的发现完全同构——SFT 学的是"人怎么推理"，RL 学的是"怎么推出对答案"。当答案正确性是唯一 reward 时，model 会绕开人类推理的弯路，找到更直接的路径。

对 robotics 的 scaling 来说，这条路径可能比堆更多 demo 数据更 promising——robotic demo 获取成本 O(N) 增长，simulator rollout 成本 O(1)（parallelize 后）。如果 RL 能把 1 条 demo 的价值放大到 500 条 demo 的水平，VLA 的 data bottleneck 就被绕开了。

唯一前置条件：**simulator fidelity 要够用**。RoboTwin 2.0 的 sim-to-real 结果（17.5% → 38.5%）证明这条路可行，但还有很大空间。

代码开源在 https://github.com/PRIME-RL/SimpleVLA-RL，veRL 主仓库在 https://github.com/volcengine/verl。

---

## 一句话再总结

**把 R1 那套 "只看结果、自己探索" 的 RL recipe 搬到教机器人做事上，机器人不仅学得比看 500 条人类操作视频还好，还自己发明了人类没教过的新动作策略（pushcut）。**

希望这版"人话"讲清楚了，Andrej。如果还想深挖某一块（比如 GRPO 的数学推导、veRL 的实现细节、或者 pushcut 的更深层机制），继续聊。

---

# SimpleVLA-RL：把 DeepSeek-R1 的 recipe 搬到 VLA

Andrej，这篇 paper 本质上是在做一件事——**把 R1-Zero 那套 rule-based online RL + GRPO 范式，从语言模型的"思维链推理"迁移到 VLA 的"长程动作规划"**。下面我从 first principles 角度拆开讲，重点放在那些 LLM RL 工作者容易踩坑的"VLA-specific"细节上。

---

## 1. Motivation：为什么 VLA 需要 RL，而不是继续 scale SFT

当前 VLA 主流是 **pretrain → SFT** 两阶段范式（OpenVLA、π0、RDT-1B 都是这个套路）。问题有两个：

- **Data scarcity**：robotic trajectory 极贵。LIBERO 一条 demo 需要人工 teleop，Open X-Embodiment 的 collection 成本在论文里反复被吐槽。SFT 想要 scale，trajectory 数量跟不上。
- **Generalization 差**：SFT 本质是 imitation，只能复现 demonstration 分布内的行为模式。一旦遇到 distribution shift（新物体/新空间配置/新任务），catastrophic forgetting 严重（paper Figure 4 里 SFT 在 unseen task 上直接掉到 0%）。

R1 给的启示是：**outcome-only reward + online RL** 可以诱导出 SFT 数据里没有的 reasoning pattern。VLA 这边等价的猜想就是：能不能用 binary task-success reward，让 policy 自己探索出更优的 manipulation strategy？答案是能，而且论文里观测到了 "pushcut" 这种 emergent behavior（后面详细讲）。

参考链接：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164

---

## 2. LLM RL vs VLA RL：四个关键差异（核心 intuition）

这一节是 paper 的 §2，也是最容易让 LLM 出身的人忽略的部分。

### 2.1 State $s_t$

LLM 的 state 就是 prompt + 已生成 token：

$$s_t = (x_{\text{prompt}}, y_1, y_2, \ldots, y_{t-1})$$

其中 $x_{\text{prompt}}$ 是初始 prompt，$y_t$ 是第 $t$ 步生成的 token。

VLA 的 state 是多模态的：

$$s_t = (o_t^{\text{vis}}, o_t^{\text{prop}}, l_{\text{task}})$$

- $o_t^{\text{vis}}$：视觉观测，可以是 RGB / depth / point cloud
- $o_t^{\text{prop}}$：proprioception（本体感觉），即机器人自身状态，joint angles、end-effector pose
- $l_{\text{task}}$：语言指令（"pick up the red cup"）

关键差异：**VLA 的 $s_t$ 在 rollout 过程中是会被 action 改变的**。LLM 的 $s_t$ 只受自己生成的 token 影响（autoregressive），但 VLA 每发一个 action chunk，environment 物理动力学就会改变下一帧的 $o_{t+k}^{\text{vis}}$ 和 $o_{t+k}^{\text{prop}}$。这是 closed-loop interaction 的根源。

### 2.2 Action $a_t$

LLM：$a_t = y_t \in \mathcal{V}$，直接从 vocabulary 里采样 token，分布是 $\text{softmax}(f_\theta(s_t)/T)$。$f_\theta \in \mathbb{R}^{|\mathcal{V}|}$ 是 logit，$T$ 是采样温度。

VLA：$a_t \in \mathbb{R}^d$（典型 $d=7$，6-DoF pose + gripper open/close）。生成方式有三种：

1. **Token-based**（OpenVLA、OpenVLA-OFT）：把连续 action 离散化成 token 序列，和 LLM 一样 next-token prediction
2. **Diffusion expert**（RDT-1B、π0）：在 latent space 上做 diffusion denoising
3. **MLP regression**（OpenVLA-OFT 官方版）：MLP head 直接回归连续 action

**这是 SimpleVLA-RL 的第一个核心 design choice**——他们选择了 token-based，因为只有 token-based 才能天然给出 $\pi_\theta(a_t|s_t)$ 的概率分布，这是 PPO/GRPO 算 importance sampling ratio $r_{i,t}(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ 必需的。

Diffusion policy 的 likelihood 难以 tractable 估计（要去 score matching 的 score function 上绕一大圈），MLP regression 则是 deterministic，没有 stochasticity 给 exploration 用。所以 SimpleVLA-RL 把 OpenVLA-OFT 的官方 MLP head 替换成了 LLaMA2 的 LM head + cross-entropy loss，从头 SFT 训了一遍。

### 2.3 Reward

LLM：rule-based binary（数学题答案对错）或者 learned reward model $R_\phi(\tau) \in [0,1]$。

VLA：传统 robot RL 用的是 dense shaped reward（distance-to-goal、grasp success 等 hand-crafted term）。这类 reward 不可迁移、需要 per-task design，scaling 不起来。

SimpleVLA-RL 选了 R1-style：

$$R(a_{i,t} | s_{i,t}) = \begin{cases} 1, & \text{if } \mathrm{traj}_i \text{ succeeds} \\ 0, & \text{otherwise} \end{cases}$$

trajectory-level 0/1 reward，**均匀 propagate 到该 trajectory 内所有 action token**。这种 sparse reward 的好处是 task-agnostic、scalable；坏处是 cold-start 难、credit assignment 模糊（哪个 token 该为成功负责？）。GRPO 通过 group-relative normalization 来缓解 credit assignment。

### 2.4 Rollout

LLM rollout：autoregressive 生成到 stop token，中间没有 environment feedback。一次 forward 就出一条 trajectory。

VLA rollout：必须 closed-loop，每隔 $k$ 个 step（action chunk size）就要 query 一次 environment 拿新观测。这导致：

- **采样慢**：每个 step 要 render 物理仿真、跑 forward dynamics
- **采样贵**：simulator 通常单线程，无法像 LLM 那样一个 GPU batch 里出多条 trajectory

SimpleVLA-RL 的工程解法是 **parallel multi-environment rendering**——把 N 个 simulator 实例 spawn 到 process pool 里，policy 端一次性给 N 个 state 出 N 个 action chunk，再 batch submit 给环境 step。Listing 1 的伪代码把这个结构画得很清楚：

```python
for t in range(max_steps):
    actions = policy.generate(states, temperature=1.0)  # batched forward
    states, dones = env_process_pool.submit(envs.step, actions)  # batched env step
    # filter out done envs
```

这是把 LLM 训练里的 "generate batch" 改造成 "generate-env_step-generate" 的 interleaved loop。

参考：veRL/HybridFlow 论文 https://arxiv.org/abs/2409.19256

---

## 3. GRPO 在 VLA 上的具体形式

GRPO（Group Relative Policy Optimization）来自 DeepSeekMath，核心 trick 是 **干掉 value network**，用 group 内的 reward 归一化来估计 advantage。这对于 VLA 很关键，因为 value network 在高维 multimodal state 上很难训。

完整 objective（paper Eq. 7）：

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D}, \{\tau_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \min\left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

变量逐个解释：

- $G$：每个 prompt/state 采样的 trajectory 数（group size，paper 里是 8）
- $|\tau_i|$：第 $i$ 条 trajectory 的 token 长度（VLA 里是 action token 数 × chunk size）
- $r_{i,t}(\theta) = \pi_\theta(a_{i,t}|s_{i,t}) / \pi_{\theta_{\text{old}}}(a_{i,t}|s_{i,t})$：importance sampling ratio，新旧 policy 在同一 上的概率比
- $\hat{A}_i = (R_i - \text{mean}(\{R_i\})) / \text{std}(\{R_i\})$：第 $i$ 条 trajectory 的 group-normalized advantage。注意是 trajectory-level，整条 trajectory 共享同一个 advantage——这是 binary outcome reward 的必然结果
- $\epsilon$：PPO clip ratio，限制 policy update 幅度，保证 trust region
- $\beta$：KL penalty 系数，让 $\pi_\theta$ 不要离 reference policy $\pi_{\text{ref}}$ 太远

### SimpleVLA-RL 的修改（§3.4 Eq. 11）

借鉴 DAPO（字节系的 LLM RL 工作），他们做了三处改动：

$$\mathcal{T}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \min\left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \hat{A}_i \right) \right]$$

s.t. $0 < |\{\text{successful traj}\}| < G$

变化：
1. **去掉了 KL 项**：$\beta = 0$。这样不需要 reference policy $\pi_{\text{ref}}$，省一半显存，也去掉 exploration 的约束
2. **Clip 范围不对称**：$\varepsilon_{\text{low}} = 0.2, \varepsilon_{\text{high}} = 0.28$。下界 0.2 是 PPO 默认值，上界 0.28 是 DAPO 的 Clip-Higher，让低概率 token 的概率能涨得更多
3. **Dynamic Sampling 约束**：约束 group 内必须有成功有失败（$0 < \text{success count} < G$），否则 advantage $\hat{A}_i = 0$，gradient 消失

参考：DAPO 论文 https://arxiv.org/abs/2503.14476 ；DeepSeekMath（GRPO 原始 paper）https://arxiv.org/abs/2402.03300

---

## 4. 三个 Exploration Enhancements 的 intuition

§3.3 是 paper 的方法核心，三个 tricks 都来自 LLM RL 但在 VLA 上效果显著（Figure 3 显示 +10~15% 提升）。

### 4.1 Dynamic Sampling

GRPO 是 critic-free 的，advantage 完全靠 group 内的 reward spread 估计：

$$\hat{A}_i = \frac{R_i - \text{mean}}{\text{std}}$$

如果一个 group 8 条 trajectory 全部成功（$R_i = 1$ for all）或全部失败（$R_i = 0$ for all），那么 std = 0，所有 $\hat{A}_i = 0$，**gradient 直接为 0**。这在 LLM RL 里也会发生，但 VLA 更严重——因为 manipulation task 成功率天然低，base model 经常 8 条全失败。

Dynamic Sampling 的做法：rollout 阶段如果发现某个 group 全成功或全失败，就**丢弃这个 group，重新采样**，直到 batch 里所有 group 都是 mixed outcome（公式 Eq. 10 的约束）。

直觉：这是把"无效的 gradient update"换成"有效的 exploration"。代价是 sampling 变慢（要 retry），但对于 binary reward 是必需的。

### 4.2 Clip Higher

PPO 的 clip 是双向对称的 $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$，目的是限制 $r_{i,t}(\theta)$ 的剧烈变化。但 DAPO 发现：**上界 1.2 会限制低概率 token 涨概率的速度**，因为如果某个好的 action token 初始 $\pi_{\theta_{\text{old}}}(a) = 0.01$，ratio 从 1 涨到 1.2 也才到 0.012，依然很低。

调高上界到 $1 + \varepsilon_{\text{high}} = 1.28$，让"低概率但被 reward 验证为好的 token"能更快涨上去。在 VLA 上这特别有用，因为 action token 分布经常是 long-tail 的——某种 pushing action 可能初始概率极低但实际有效（pushcut 现象的机制基础）。

### 4.3 Higher Rollout Temperature

LLM RL 里温度一般用 $T = 1.0$，paper 提到提到 1.6。高温让 softmax 分布更平坦，采样到 low-probability token 的几率变高，等价于更强的 exploration。

VLA 上温度太低会导致 policy 早早 collapse 到 grasp-move-place 这一条 mode（因为 SFT 数据里全是这个 pattern）。高温让 policy 偶尔去试 push 这种 out-of-distribution action，才有可能命中 reward=1，然后通过 Clip-Higher 把这个 token 的概率推高。

**这三个 trick 是相互耦合的**：Dynamic Sampling 保证有 signal，Clip Higher 保证好 token 能被强化，High Temperature 保证好 token 能被发现。三者缺一不可。

参考：POLARIS（temperature scheduling） https://hkunlp.github.io/blog/2025/Polaris ；Entropy Mechanism https://arxiv.org/abs/2505.22617

---

## 5. 实验结果分析

### 5.1 LIBERO（Table 2）

LIBERO 四个 suite：Spatial / Object / Goal / Long，每个 suite 10 个 task。

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT (SFT only) | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| **+ SimpleVLA-RL** | **99.4** | **99.1** | **99.2** | **98.5** | **99.1** |
| Δ | +7.8 | +3.8 | +8.6 | +12.0 | +8.1 |

值得注意的几点：
- **LIBERO-Long 提升最大（+12%）**：long-horizon 任务（多步骤）对 RL 的探索收益最敏感，因为每一步的微小改进会累积。SFT 只能学到 demonstration 的固定组合，RL 可以重新组合 sub-skill
- 已经 SFT 到 91% 的 base model，RL 还能再涨 8%——这说明 SFT 远没到 data 的上限，只是 imitation 的上限

### 5.2 RoboTwin 2.0（Table 4，按 horizon 分桶）

RoboTwin 2.0 是双臂 benchmark，50 个 task，按平均 step 数分四档：Short (112-130) / Medium (151-223) / Long (283-313) / Extra-Long (466-637)。

| Horizon | π0 | RDT | OpenVLA-OFT | + SimpleVLA-RL |
|---------|-----|-----|-------------|----------------|
| Short | 45.5 | 24.5 | 21.3 | 64.9 |
| Medium | 58.8 | 47.8 | 47.1 | 72.5 |
| Long+Extra | 43.3 | 27.8 | 46.5 | 69.0 |
| **Overall** | **49.2** | **33.3** | **38.3** | **68.8** |

SimpleVLA-RL **超过 π0 接近 20 个点**，这很惊人，因为 π0 用了海量 pretraining data（Physical Intelligence 的私有数据集），而 OpenVLA-OFT 是纯 open-source。RL 的优势在 Long-Horizon 上尤为明显——Extra-Long 任务（466-637 step）RL 涨了 +11.1 / +18.7。

直觉：**long-horizon 是 SFT 的死穴**。demo trajectory 长，SFT 的 cross-entropy loss 会让 model 在每一步都去 mimic expert，但任何一步的小误差会指数级累积导致 trajectory 失败。RL 直接 optimize trajectory-level success，自然更鲁棒。

### 5.3 Data Scarcity（Table 5，最 striking 的结果）

这是 paper 最有 "scaling law" 意味的实验。LIBERO 上对比 One-Trajectory SFT（每个 task 仅 1 条 demo）vs Full-Trajectory SFT（500 条 demo/task）。

| Setting | Spatial | Object | Goal | Long | Avg |
|---------|---------|--------|------|------|-----|
| 1-traj SFT | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| 1-traj SFT + RL | 98.2 | 98.7 | 98.8 | 91.7 | 96.9 |
| 500-traj SFT | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| 500-traj SFT + RL | 99.4 | 99.1 | 99.2 | 98.5 | 99.1 |

**1 条 demo + RL（96.9%）超过 500 条 demo 的 SFT（91.0%）**。LIBERO-Long 从 17.3% 直接拉到 91.7%。

直觉：这条曲线告诉我们，SFT 的 data efficiency 极差——500 条 demo 还不如 1 条 demo + RL。RL 是把"看 demonstration 学动作"换成"在仿真里自己试错学任务"，env 提供的 trial-and-error signal 比 cross-entropy on expert action 强得多。

这和 Karpathy 你之前讨论 R1 时的直觉一致：**reasoning 不是被 SFT 教出来的，是被 reward 逼出来的**。这里 manipulation strategy 也是同理。

### 5.4 Generalization（Figure 4）

实验设计：每个 LIBERO suite 10 个 task，9 个 seen 1 个 unseen，看训练过程中 unseen task 的 success rate 演化。

关键发现：
- **SFT 在 unseen task 上严重过拟合**，success rate 随训练进行掉到 0%（catastrophic forgetting）
- **RL 在 unseen task 上同步提升**，LIBERO-Goal 三个 unseen task 都涨 5-15%

直觉：SFT 让 model 在 seen task 的 action 分布上 overfit，但 manipulation 的"物体-动作"绑定太具体，迁移性差。RL 学的是"task-success oriented"的 policy，它发现 push 比 grasp 更高效，这种 strategy-level 抽象比 token-level imitation 更可迁移。

### 5.5 Sim-to-Real（Table 6）

RoboTwin 2.0 的 4 个 task 迁移到真实 Agilex Piper 双臂。训练**完全用仿真**，real world 零样本测试。

| Task | RDT | OpenVLA-OFT | + SimpleVLA-RL |
|------|-----|-------------|----------------|
| Stack Bowls | 60.0 | 38.0 | 70.0 |
| Place Empty Cup | 4.0 | 2.0 | 10.0 |
| Pick Bottle | 10.0 | 0.0 | 14.0 |
| Click Bell | 20.0 | 30.0 | 60.0 |
| Avg | 23.5 | 17.5 | 38.5 |

SimpleVLA-RL 把 sim-to-real 的成功率从 17.5% 拉到 38.5%。Pick Bottle 这个任务对动作精度要求极高（gripper 对齐稍偏瓶子就掉），SFT 直接 0%，RL 涨到 14%。说明 RL 训出来的 policy 在 action precision 上也更强——这反直觉，因为 RL 通常被认为是 high-variance。但在 binary reward + GRPO 的设定下，policy 被"reward=1"吸引到更精准的轨迹上。

参考：RoboTwin 2.0 https://arxiv.org/abs/2506.18088

---

## 6. Pushcut：RL 涌现的新行为模式（§6.1，paper 最有趣的部分）

这是 paper 的 "Aha Moment"。在 RoboTwin 2.0 的 "Move Can Pot" 任务上：

- **Demonstration 数据**（SFT 学的）：grasp can → move to pot → place
- **RL 训练后涌现的行为**：直接 push can 滑到 pot 旁边

"Place A2B Left/Right" 任务上也类似——demo 是 grasp A → move → place，RL 学会直接 push A 到目标位置。

### 为什么会出现 pushcut？

我倾向于这么理解：

1. **SFT data 的 bias**：人类 teleop 时倾向于 grasp-move-place，因为对人来说 grasping 是最自然的操作。所以 SFT 学到的是"模仿人的操作偏好"，不是"解决任务的最优策略"
2. **Binary reward 的解放**：reward 只看 task 完成（can 是否在 pot 旁边），不规定怎么完成。policy 有自由度去探索任何能拿到 reward=1 的 trajectory
3. **Exploration enhancement 的协同**：High Temperature 让 push 这种 low-probability action 偶尔被采样到，Clip Higher 让它一旦拿到 reward 就能快速涨概率，Dynamic Sampling 保证有效 gradient
4. **Push 比 grasp 简单**：grasp 需要精确对齐 gripper + 控制抓取力 + lift + move + place，每一步都可能失败。Push 只需要一个方向的力，成功率高得多。在 binary reward 下，policy 自然往简单策略收敛

这和 R1 的 "Aha Moment" 是同构的——**reward signal 让 model 发现了 SFT 数据里不存在的、更高效的解题路径**。区别是 R1 是 reasoning pattern（self-verification、backtracking），这里是 motor pattern（push vs grasp）。

直觉：这暗示 RL 不仅仅是 SFT 的"refinement"，而是能解锁 fundamentally 不同的 solution space。对 VLA 的 scaling 来说，这可能比单纯堆更多 demo 数据更有价值。

---

## 7. Failure Modes：RL 不是万能的（§6.2，Table 7）

这个实验很关键，**揭示了 RL 的 cold-start 问题**。

RoboTwin 2.0 上对比三种 base model：

| SFT 数据量 | SFT 成功率 | + RL 成功率 | Δ |
|-----------|-----------|------------|---|
| 0 trajectory | 0% | 0% | 0 |
| 100 trajectory | 7.3% | 25.4% | +18.1 |
| 1000 trajectory | 28.2% | 50.4% | +22.2 |

**0 trajectory SFT 的 base model 上 RL 完全失败**，所有 task 都是 0%。

直觉：GRPO 需要 group 内有 mixed outcome 才有 gradient。如果 base model 0% 成功率，所有 rollout 都失败，Dynamic Sampling 会无限 retry 直到耗尽 budget。**Pure RL from scratch 在 VLA 上不可行**。

这说明：
- **SFT 提供"探索种子"**：哪怕只有 7% 成功率，也够 RL bootstrap 起来
- **RL 的 effectiveness 有 threshold**：base model 能力越强，RL 涨幅越大（100→25.4 涨 3.5x，1000→50.4 涨 1.8x，但绝对涨幅后者更大）
- **pretrain ≠ task capability**：OpenVLA-OFT 即使经过大规模 pretrain，zero-shot 在 RoboTwin 上还是 0%。pretrain 给的是 representation，不是 task prior

这和 Karpathy 你之前讨论 R1-Zero 时的观察一致——R1-Zero 能 work 是因为 base model 已经有 strong reasoning prior（来自 pretrain 阶段的海量 CoT 数据）。VLA 的 pretrain 还远没到这个水平，所以 SFT bootstrap 是必需的。

---

## 8. 与 LLM RL 工作的对应关系

把 SimpleVLA-RL 和近期 LLM RL 工作对照看，能 build 很好的 intuition：

| LLM RL | VLA RL 对应 | 关键差异 |
|--------|------------|---------|
| DeepSeek-R1 (R1-Zero) | SimpleVLA-RL | R1-Zero 能 from-scratch，VLA 需要 SFT bootstrap |
| GRPO | GRPO（一样） | trajectory-level reward 在 VLA 上更 sparse |
| DAPO (Clip-Higher, 去 KL) | DAPO（直接搬过来） | 完全适用 |
| Dynamic Sampling | Dynamic Sampling | VLA 上更关键，因为成功率低 |
| Aha Moment | Pushcut | 都是 emergent behavior |
| Entropy collapse | VLA 也有，靠高温解决 | — |

paper 里也提到了一些并行工作：RIPT-VLA（用 RLOO）、VLA-RL（用 PPO）、TGRPO（用 Claude 3.7 当 reward model）、RFTF（用 value model 给 dense reward）。SimpleVLA-RL 的差异化在于：(1) 完全 rule-based reward，没有 learned reward model；(2) 系统性验证 sim-to-real；(3) 发现 pushcut 现象。

参考：
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- VLA-RL: https://arxiv.org/abs/2505.18719
- TGRPO: https://arxiv.org/abs/2506.08440
- RFTF: https://arxiv.org/abs/2505.19767

---

## 9. 工程：veRL 的 VLA 扩展

veRL（Volcano Engine RL）是字节开源的 LLM RL 框架，底层是 HybridFlow，做 RLHF 的 hybrid resource scheduling。SimpleVLA-RL 在它上面加了三件事：

1. **VLA-specific trajectory sampling**：把 LLM 的 `policy.generate()` 改成 closed-loop env interaction loop
2. **Parallel multi-environment rendering**：simulator 进程池，batched env.step
3. **Optimized loss computation**：action token 的 cross-entropy + GRPO importance ratio

训练配置：8 × A800 80GB，batch size 64，group size 8，learning rate 5e-6，256 个 action token，action chunk size 8（LIBERO）/ 25（RoboTwin）。max env step 200~800 视 task 复杂度。

代码已经开源：https://github.com/PRIME-RL/SimpleVLA-RL

veRL 主仓库：https://github.com/volcengine/verl

---

## 10. 局限性和未来方向

Paper 自己承认的 + 我观察到的：

1. **Token-based only**：排除了 diffusion policy 和 MLP regression 的 VLA。RDT-1B、π0 这种 diffusion-based VLA 暂时用不上这套方法。未来要么想办法估计 diffusion 的 likelihood（flow matching 那一套），要么换 policy gradient 形式
2. **Simulator-bound**：必须有 high-fidelity simulator 才能 rollout。real-world RL（ConRFT 那条路线）成本高得多
3. **Binary reward 的天花板**：对于特别复杂的任务（比如需要 tool use、多阶段规划），sparse binary reward 可能探索效率太低。RFTF 用 value model 给 dense reward 是一个方向
4. **Cold-start 依赖 SFT**：这意味着 VLA pretrain 还得继续做，RL 不是替代 SFT 而是补充。什么时候能像 R1-Zero 一样 from-scratch RL，是 open question
5. **Pushcut 的 reproducibility**：paper 只展示了两个 task 上的 pushcut，没说在更广泛 task suite 上的频率。emergent behavior 这种东西很难 systematize

---

## 11. 给 Karpathy 的总结性 intuition

如果让我用一句话总结这篇 paper 的核心 insight：

> **VLA 的 SFT 是在学"人怎么做"，RL 是在学"任务怎么做"。当人的操作偏好不是任务最优解时，RL 会发现 SFT 数据里不存在的、更高效的策略。**

这和 R1 在 reasoning 上的发现完全同构——SFT 学的是"人怎么推理"，RL 学的是"怎么推出对答案"。当答案正确性是唯一 reward 时，model 会绕开人类推理的弯路，找到更直接的路径。

对于 robotics 的 scaling 来说，这条路径可能比堆更多 demo 数据更 promising——因为 robotic demo 的获取成本是 O(N) 增长的，而 simulator rollout 的成本是 O(1) 的（parallelize 后）。如果 RL 能把 1 条 demo 的价值放大到 500 条 demo 的水平，那 VLA 的 data bottleneck 就被绕开了。

唯一的前置条件是：**simulator 的 fidelity 要够用**。RoboTwin 2.0 的 sim-to-real 结果（17.5% → 38.5%）证明这条路可行，但还有很大空间。

未来值得关注的几条线：
- **Diffusion VLA + RL**：怎么把 token-based RL 迁移到 diffusion policy 上
- **Hierarchical RL**：long-horizon task 的 subgoal 分解
- **Real-world RL**：ConRFT 那条路，sim 和 real 的 gap 怎么弥合
- **Emergent behavior 的 systematization**：pushcut 是 isolated 现象还是 RL 的普遍性质

希望这些分析对你 build intuition 有帮助，Andrej。如果想深挖某一块（比如 GRPO 的数学推导、veRL 的实现细节、或者 pushcut 的更深层机制），可以继续聊。
