---
source_pdf: WhatCanRLBringtoVLAGeneralization.pdf
paper_sha256: 2dc2c73ab51cf2a97f6e3223cf5dd229632afd599a38a9b515f39d46c2e411a9
processed_at: '2026-08-13T04:16:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

Andrej, 咱们抛开公式，用最直白的话聊一下这篇paper在干嘛。

---

## 故事的开头

现在大家都用SFT (supervised fine-tuning) 训VLA, 就是拿expert录的trajectory让model模仿。这玩意儿有个老毛病: **imitation learning的compounding error**。

啥意思呢? 想象你抄学霸的作业, 但你抄的时候手抖了一下写错一个数字, 后面所有步骤就全歪了。因为你不知道这个数字从哪来, 你只会抄, 不会自己想。SFT就是这么个状态 — model在training data上见过的state它知道怎么处理, 一旦走到没见过的state (哪怕只偏了一点点), 就懵了。

RL不一样, RL是让model自己try, 试对了给reward, 试错了给惩罚 (或者不给reward)。model会慢慢学到"哪种action能让我拿到reward", 而不是"这个state下expert怎么做的"。这就是为什么理论上RL应该更robust — 它学的是skill, SFT学的是memorization。

LLM那边已经证明了这个: Chu et al. 2025那篇 "SFT memorizes, RL generalizes" (https://arxiv.org/abs/2501.17161) 说得很清楚。但VLA这边一直没人系统性地验证过。这篇paper就是来填这个坑的。

---

## 他们具体干了啥

很简单, 三件事:

### 1. 搭了一个benchmark

把generalization拆成三个维度:

- **Vision**: 换桌子背景, 往图片上贴乱七八糟的texture, 加noise
- **Semantics**: 换没见过的object, 换receptacle, 换instruction的措辞
- **Execution**: 换robot初始pose, 换object位置, 甚至**task执行到一半把object突然teleport走**

最后这个mid-episode reposition特别狠, 等于你的policy正在执行任务, 突然环境给你来一下, 看你能不能recover。

### 2. 比较了三种RL算法

- PPO (经典actor-critic, 有value function)
- GRPO (DeepSeek那套, 不要value function, 用group baseline)
- DPO (offline, 用preference pair)

结果: **PPO完胜**。

作者的猜测:
- GRPO在LLM上work是因为text generation的token之间相对独立, 但robotics每个action都改变environment state, 是non-stationary的, group baseline会被这种dynamics搞崩
- DPO失败是因为robotics的reward太sparse (就两个阶段reward: grasp 0.1, place 1.0), 很难从trajectory pair里区分谁好谁坏

我的intuition: GRPO的advantage是trajectory级别的, 整条trajectory共享一个advantage值。LLM的reasoning chain最后对错基本能判断, 但robotics的control是dense sequence, 第3步的action好不代表第15步的action好, 需要per-step的credit assignment, 这就是PPO的GAE在做的事。

### 3. 搞了一个能用的PPO recipe

重点:
- **Shared backbone**: actor和critic共享一个Llama-2 7B, 只在第一个action token位置接个3层MLP预测value
- **Warm-up**: 先用140条demo SFT一下, 让action token的分布对齐到当前任务
- **epoch=1**: 每个batch只过一次梯度, PPO对on-policy很敏感

42小时单A100跑完, 还算可复现。

---

## 核心发现

直接看数据:

| 维度 | SFT vs RL | 谁赢 |
|---|---|---|
| Vision | 基本持平 | 平手 |
| Semantics | RL明显好 | RL |
| **Execution** | **RL碾压** | **RL完胜** |

特别是Mid-Episode Object Reposition这个task:
- SFT: 28.6% success
- RL: 74.5% success

差了快3倍。为什么? 因为SFT的training data全是motion planner生成的"完美轨迹", 从来没有"object突然跑了怎么recover"的case。RL在rollout过程中各种出错、各种recover, natural地学到了recovery behavior。

还有个很visual的发现 (Fig 8): SFT的轨迹紧紧贴合motion planner的path, RL的轨迹覆盖的workspace范围大得多, end-effector orientation也更丰富。这就是RL exploration带来的diversity。

---

## Sim-to-real的初步验证

虽然只是preliminary, 但挺说明问题的:

| Metric | SFT | RL |
|---|---|---|
| Grasp | 10% | 43% |
| Pick-and-Place | 0% | 27% |

SFT在real robot上完全崩了, RL还能用。作者观察到的现象: SFT的policy会overshoot (因为一直模仿motion planner的smooth trajectory, 遇到real world的摩擦、perception noise就过头了), RL的policy会jitter, 但会iteratively调整到正确pose — 这正是closed-loop control的典型表现。

---

## 我的takeaway

1. **RL对VLA的价值是真实的, 但维度specific**: 它对execution robustness有巨大帮助, 对semantic understanding有中等帮助, 对visual robustness没什么额外帮助。这个结论很honest, 不oversell。

2. **Vision维度RL没赢是个limitation**: 作者解释说visual robustness主要靠pretrained visual encoder (SigLIP + DINOv2), SFT和RL都只是在action head上fine-tune, 不直接优化visual invariance。我觉得如果RL training时加更狠的visual domain randomization, 这块可能能补上。

3. **PPO是VLA RL的正解**: LLM那边的GRPO/DPO recipe不能直接搬过来, robotics的sequential + non-stationary + partial observable本质需要value function做temporal credit assignment。这个finding对community很有指引性。

4. **42小时单A100**: 这个recipe的cost已经低到大多数lab都能复现了。期待后续工作扩展到long-horizon multi-task, 以及更强的sim-to-real。

5. **更深层intuition**: SFT学的是"what to do in this state", RL学的是"how to achieve the goal from this state"。前者是memory lookup, 后者是planning + control。这就是为什么RL能generalize到没见过的state, 而SFT不能。

---

## 一句话总结

RL fine-tuning让VLA从"只会模仿expert的鹦鹉"变成了"能自己完成任务并从错误中recover的agent", 特别是在execution鲁棒性上提升巨大, 但对visual perturbation没什么额外帮助。PPO是最佳算法, 而且用42小时单A100就能跑出来, recipe已经足够accessible。

Andrej, 这篇paper的value在于它**不搞花哨的新算法**, 就是老老实实地把一个question回答清楚: RL到底给VLA带来了啥。这种empirical clarity在现在的VLA research里其实挺稀缺的。

---

# What Can RL Bring to VLA Generalization? 深度解读

Andrej, 这是一篇很扎实的empirical study, 来自Tsinghua的team, 核心问题非常clean: **RL fine-tuning相比SFT, 到底给VLA带来了什么泛化收益?** 这个问题在LLM那边已经有比较清晰的答案 (SFT memorizes, RL generalizes, 见 Chu et al. 2025 https://arxiv.org/abs/2501.17161), 但在VLA这边一直缺乏systematic study, 因为robotics的RL training成本高、环境复杂、reward设计tricky。这篇paper的核心contribution就是把这个gap补上了, 而且给出了一个**可复现的PPO recipe** + **三维度的generalization benchmark**。

---

## 1. Motivation: 为什么SFT对VLA不够?

VLA的训练主流是behavior cloning (SFT), 即模仿expert demonstrations。但SFT有个fundamental问题: **compounding errors under distribution shift** (Ross & Bagnell 2010, https://proceedings.mlr.press/v9/ross10a.html)。当你policy在step $t$ 稍微偏离expert trajectory时, 系统就进入一个unfamiliar state, 而这个state在training data里从未出现过, policy就不知道怎么recover, 误差会**quadratic增长** w.r.t. task horizon。

这里的关键intuition是: SFT学的是"什么state下采取什么action"的mapping, 是一个**conditional distribution memorization**; 而RL学的是"如何在一个reward结构下达到goal"的**credit assignment**, 它鼓励policy去探索recovery behaviors, 哪怕这些behaviors从未在demonstration里出现过。这就是为什么RL能break out of the imitation manifold。

---

## 2. Problem Formulation: POMDP + 三个loss

作者把每个language-conditioned task $T$ 建模成POMDP:

$$\mathcal{M} = (S, \mathcal{A}, \mathcal{P}, R, \mathcal{O}, \mathcal{L}, P(s_0), \gamma)$$

变量含义:
- $S$: state space (robot + environment states)
- $\mathcal{A}$: action space (control commands, 这里是7-DoF Cartesian delta + binary gripper)
- $\mathcal{P}$: transition function $s_{t+1} \sim \mathcal{P}(\cdot | s_t, a_t)$
- $R$: reward, sparse: 0.1 for grasping + holding, 1.0 for placement
- $\mathcal{O}$: observation space (640×480 RGB)
- $\mathcal{L}$: language instruction space
- $\gamma$: discount factor
- $H$: history length, 这里 $H=1$ (只用当前frame, 不用history)

Policy: $\pi_\theta(a_t | o_{t-H+1:t}, l)$

**SFT loss**:
$$\mathcal{L}_{\text{SFT}}(\theta) = \sum_{(\tau^{(i)}, l^{(i)}) \in \mathcal{D}_T} \sum_{t=0}^{K_i - 1} \ell_{\text{SFT}}(\hat{a}_t^{(i)}, a_t^{(i)})$$

其中 $\hat{a}_t^{(i)} = \pi_\theta(o_{t-H+1:t}^{(i)}, l^{(i)})$, $\ell_{\text{SFT}}$ 是next-token cross-entropy (因为OpenVLA把action discretize成256 bins)。

**RL loss** (policy gradient形式):
$$\mathcal{L}_{\text{PG}}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{M-1} A_t^\pi \log \pi_\theta(a_t | o_{t-H+1:t}, l)\right]$$

$A_t^\pi$ 是advantage estimator, $\tau \sim \pi_\theta$ 表示trajectory是从当前policy sample的 (online)。

---

## 3. OpenVLA架构解析

基于OpenVLA (Kim et al. 2024, https://arxiv.org/abs/2406.09246):

- **Visual encoder**: fused SigLIP (https://arxiv.org/abs/2303.15319) + DINOv2 (https://arxiv.org/abs/2304.07193), 把640×480 image embed成visual tokens
- **Language backbone**: Llama-2 7B (https://arxiv.org/abs/2307.09288)
- **Action discretization**: RT-2 recipe (https://arxiv.org/abs/2307.15818), 每个continuous action scalar映射到256 bins (1st to 99th percentile均匀划分), 生成 $\mathbf{u}_t \in \{0, ..., 255\}^{d_a}$
- 这256个action tokens**覆盖Llama-2词表里最少用的256个tokens**, 这样LLM可以直接"说"action tokens

这里有个subtle的设计: action tokens作为"language"输出, 意味着policy gradient要compute over**所有256 bins的softmax**, 这点和LLM的log-prob计算完全一致, 所以PPO的 $\log \pi_\theta(a_t | s_t)$ 实际上是 **product of per-action-token probabilities**:

$$\log \pi_\theta(\mathbf{u}_t | s_t) = \sum_{k=1}^{d_a} \log p_\theta(u_t^{(k)} | s_t, u_t^{<k})$$

---

## 4. RL Algorithm Comparison: PPO vs GRPO vs DPO

这是paper的核心technical finding之一。作者比较了三种LLM时代的RL算法在VLA上的表现:

### 4.1 PPO

标准clipped surrogate (Schulman 2017, https://arxiv.org/abs/1707.06347):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $r_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ 是importance sampling ratio, $\epsilon = 0.2$ 是clip ratio。

Advantage用GAE (Schulman 2015, https://arxiv.org/abs/1506.02438):

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \left[r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\right]$$

- $\gamma$: discount factor
- $\lambda$: GAE的bias-variance tradeoff参数 (λ=0是1-step TD, λ=1是Monte Carlo)
- $V(s)$: value function, 由critic预测

**ORZ variant** (Hu et al. 2025, https://arxiv.org/abs/2503.24290): 设置 $\gamma = 1, \lambda = 1$, 等价于完全Monte Carlo, 在LLM上work得很好。作者也试了。

### 4.2 GRPO

DeepSeek团队提出的 (Shao 2024, https://arxiv.org/abs/2402.03300), 不需要value function, 而是用**group-relative baseline**:

$$\hat{A}_t^i = \frac{r^i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

其中 $\mathbf{r} = \{r^1, r^2, ..., r^G\}$, $G=8$ 是group size, $r^i$ 是trajectory $i$ 的outcome reward (只在最后一步给)。

### 4.3 DPO/TPO

基于 (Rafailov 2023, https://arxiv.org/abs/2305.18290) 和GRAPE (Zhang 2024b, https://arxiv.org/abs/2411.19309) 的trajectory版本:

$$\mathcal{L}_{\text{TPO}} = -\mathbb{E}_{(\zeta_w, \zeta_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta\left(\log \frac{\pi_\theta(\zeta_w)}{\pi_{\text{ref}}(\zeta_w)} - \log \frac{\pi_\theta(\zeta_l)}{\pi_{\text{ref}}(\zeta_l)}\right)\right)\right]$$

- $\zeta_w, \zeta_l$: preferred/rejected trajectories
- $\beta$: KL penalty strength
- $\pi_{\text{ref}}$: reference policy (frozen)

### 4.4 结果与hypothesis

**PPO > GRPO > DPO** (见Fig 3b)。

作者的explanation:

1. **PPO vs GRPO**: GRPO在LLM里work是因为text generation是**near-stationary**的 (每个token差不多都是独立的decision)。但robotic POMDP是**non-stationary**的: 每个action改变environment state, 后续action的distribution shift会destabilize GRPO的group baseline。

2. **PPO vs DPO**: DPO是offline的, 而**sparse reward**导致很难区分trajectories的quality (一个trajectory成功了vs部分成功, 在binary sparse reward下区分度不够)。另外offline dataset和online execution的distribution shift很大 (Prudencio 2023, https://ieeexplore.ieee.org/document/1032391944)。

我的intuition: 这个发现很重要, 因为它说明**LLM的RL recipe不能直接迁移到robotics**。LLM的RLHF/PPO/GRPO之所以work, 是因为text generation的特殊性 (token-level locality, 完全observable); 而robotics是sequential decision making + partial observability + non-stationary dynamics, 这要求RL算法有**显式的value estimation**来handle credit assignment over long horizons。PPO的actor-critic结构恰好满足这个需求。

---

## 5. PPO Recipe的三个关键设计

这是paper的另一个核心贡献: 一个**42小时单A100 GPU**就能converge的PPO recipe。

### 5.1 Shared Actor-Critic Backbone

设计: 把VLA当actor, 在Llama-2的Transformer backbone上**接一个3-layer MLP value head**, 用最后一个Transformer block在**第一个action-token position**的hidden vector $h^0$ 作为value head的input。

```
[Image tokens] [Language tokens] [Action tokens: u^0, u^1, ..., u^7]
                                    ↑
                                  h^0 → MLP(3 layers) → V(s)
```

Ablation (Fig 4b):
- $h^0$ (first action token): **highest & most stable returns**
- $h^n$ (last token): worse
- $[h^0, ..., h^n]$ (concat): worse

**为什么 $h^0$ 最好?** 我的猜测: action tokens是autoregressive生成的, $h^0$ 是第一个action token, 它的hidden state**包含最多关于scene understanding的信息**, 但还没有被后续action tokens的autoregressive generation污染。$h^n$ 已经经过了好几层action token的conditioning, 信息已经被"committed"到具体的action决策上, 反而不适合预测整体的state value。

Performance: shared backbone比separate backbone**快35%, 省VRAM 83%** (44.4GB vs 81.3GB)。这是个huge win, 因为VLA的backbone是7B参数, 单独跑一个critic backbone成本很高。

### 5.2 VLA Warm-up

直接用OpenVLA的official checkpoint (在OXE上pretrain的) fine-tune效果差, 作者先在**140条demonstration trajectory** (Octo-Small + motion planner生成) 上做SFT warm-up。

效果 (Fig 5a): warm-up让PPO收敛快**~50% environment steps**。

intuition: OpenVLA的action discretization是OXE数据集的统计量决定的, 和ManiSkill WidowX-250S的action distribution有gap。warm-up本质上是把action token的output distribution对齐到当前任务的action space, 让后续RL的exploration更efficient。

### 5.3 Minimal PPO Epoch

update-to-data ratio = 1, 即每个batch只做一次gradient pass。

Ablation (Fig 5b, 5c): epoch > 1 **没有performance gain, 但wall-clock time linearly increase**。

这和LLM RLHF的经验一致 (Gao 2023等): PPO的stability依赖于on-policy, 多次gradient pass会让 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$ 太多, 即使有clipping也撑不住。

---

## 6. Benchmark: 三个维度的Generalization

这是paper最impressive的部分: 一个**systematic的3-axis benchmark**。

### 6.1 Vision (视觉泛化)
- **Unseen Table**: 16个训练桌子 → 5个未见桌子
- **Dynamic Texture (weak/strong)**: 16种texture, 以alpha=0.3/0.5叠加到object/receptacle/robot上, 每帧resize & crop不同
- **Dynamic Noise (weak/strong)**: 同上, 但叠加到**whole image**

### 6.2 Semantics (语义泛化)
- **Unseen Objects**: 16训练物体 → 9未见物体
- **Unseen Receptacles**: 16未见receptacles (替换默认黄plate)
- **Unseen Instruction Phrasing**: 16种新template (e.g., "Move the $O$ from the table to the $R$", "$O$ on the $R$, please.")
- **Multi-Object (both seen/unseen)**: 桌上有两个物体, 选一个
- **Distractive Receptacle**: 桌上有个干扰receptacle
- **Multi-Receptacle (both unseen)**: 两个receptacle, 选一个

### 6.3 Execution (执行鲁棒性)
- **Unseen Object & Receptacle Position**: 扩大位置随机化范围
- **Unseen Robot Init Pose**: 每个joint初始pose随机
- **Mid-Episode Object Reposition**: 第5步把object teleport到新位置 (这是个很harsh的扰动!)

这个benchmark设计很巧妙: Vision测**perception robustness**, Semantics测**grounding & language understanding**, Execution测**control & recovery**。三者正交, 可以独立ablate。

---

## 7. 实验结果深度分析

### 7.1 Data Scaling (Fig 6)

SFT在16k trajectory时plateau, 64k也没有进一步提升。这是个**16% data efficiency ceiling**: OpenVLA在pick-and-place上最多只能吃16k demo。

### 7.2 RL vs SFT主结果 (Fig 6c, 6d)

RL在**约0.4M environment steps**就超过SFT-16k的OOD性能。Convergence时:
- IND: RL ~ SFT (comparable)
- OOD: **RL比SFT高42.6%**

这跟Chu et al. 2025 (https://arxiv.org/abs/2501.17161) 的发现完全一致: **SFT memorizes, RL generalizes**。

### 7.3 三维度详细结果 (Fig 7, Tab 1)

让我挑几个关键cell:

| Task | SFT Suc | RL Suc | Gap |
|---|---|---|---|
| IND | 0.781 | 0.938 | +20% |
| **Obj. Rep. (mid-episode)** | 0.286 | 0.745 | **+161%** |
| **Robot Pose** | 0.339 | 0.797 | **+135%** |
| Obj. (OOD) | 0.453 | 0.714 | +58% |
| Texture-s (strong) | 0.557 | 0.630 | +13% |

**RL在Execution维度碾压SFT**, 尤其是Mid-Episode Object Reposition和Robot Pose。这正是compounding error的痛点: SFT从未见过"object突然移动"或"robot初始pose异常"的情况, 一旦遇到就crash; 而RL在training时见过各种exploration trajectories, 知道如何recover。

### 7.4 为什么Vision维度RL没有显著优势?

作者的hypothesis: Vision generalization (texture, table appearance)主要靠**visual encoder的invariance**, 而这个invariance在pretraining时已经baked in。无论是SFT还是RL, 都没有进一步induce visual robustness的机制 (训练时table就是random的, RL的reward不直接reward visual robustness)。

我的额外思考: 这其实暴露了一个limitation — 如果把visual perturbation也作为randomization放进RL training, RL理论上应该也能学到对visual的invariance。但paper里RL training的visual randomization和SFT相同 (都是16 tables), 所以RL的优势主要来自**action层面的exploration**, 而不是perception层面的robustification。

### 7.5 Trajectory Distribution (Fig 8)

这是个很intuitive的visualization: 
- **SFT trajectories**: 紧贴motion planner的path (因为demonstration就是motion planner生成的)
- **RL trajectories**: 覆盖更广的workspace, end-effector orientation更丰富

这说明RL的policy **diversifies its behavior**, 探索了更多state space, 这就是为什么它能handle OOD execution scenarios。

---

## 8. Sim-to-Real初步验证 (Tab 2)

用Franka Panda替换WidowX, zero-shot部署到real robot, 30 trials:

| Metric | SFT | RL |
|---|---|---|
| Grasp Success | 0.10 | 0.43 |
| Pick-and-Place Success | 0.00 | 0.27 |

**SFT在real world完全失败**, RL有27%的成功率。Qualitative观察: SFT overshoots, RL会jitter但iteratively adjust end-effector pose — 这正是RL学到的**closed-loop correction behavior**的体现。

(虽然只是preliminary, 但27% vs 0%的gap非常说明问题)

---

## 9. Action Chunking (OpenVLA-OFT) extension

为了验证conclusions在更强架构下成立, 作者用OpenVLA-OFT (Kim 2025, https://arxiv.org/abs/2502.19645)的action chunking (chunk size=4) 重做实验。

Adjustments:
- **Action-dim-wise clipping**: clip ratio从0.2降到0.1
- $\gamma = 0.96, \lambda = 0.85$ (tuned for stability)
- Reward是**4步累积**

结果 (Tab 3): RL仍然consistent优于SFT。例如:
- IND Suc: 0.891 (RL) vs 0.776 (SFT)
- Obj. Rep.: 0.318 (RL) vs 0.068 (SFT) — **RL在action chunking下的mid-episode reposition甚至更好**

---

## 10. Articulated Objects Extension (Tab 4)

更复杂的任务: 开articulated objects (微波炉、洗碗机等) 的门, 超过20°。

6个OOD split, RL vs SFT:
- Vision Avg: RL 0.591 vs SFT 0.646 (SFT略好, 但都degradation大)
- **Semantic Avg: RL 0.386 vs SFT 0.373** (Articulated Obj: RL 0.151 vs SFT 0.042, **+260%**)
- **Execution Avg: RL 0.398 vs SFT 0.300** (EE Pose: RL 0.312 vs SFT 0.151, **+107%**)

RL的semantic和execution优势在更complex manipulation task上依然成立。

---

## 11. 我的Intuition & 思考

### 11.1 为什么RL对Execution这么有效?

核心: **Execution的OOD本质是state distribution shift**, 而SFT的training distribution是motion planner生成的"完美轨迹", 完全不包含recovery scenarios。RL的rollout天然包含失败→recover→成功的循环, 所以policy学到了"在abnormal state下如何return to success manifold"。

数学上, SFT minimize的是 $\mathbb{E}_{\tau \sim \mathcal{D}_{\text{expert}}}[\ell]$, 只覆盖expert manifold; RL minimize的是 $\mathbb{E}_{\tau \sim \pi_\theta}[... ]$, 覆盖的是policy's visited states, 这个set会随着exploration不断扩大, 自然包含更多recovery scenarios。

### 11.2 为什么Vision的gap小?

Vision perturbation是**observation-level**的, 不改变underlying MDP的transition / reward。Policy只需要visual invariance, 这个靠pretrained visual encoder (SigLIP+DINOv2) 已经有strong prior。SFT和RL都只是在action head上fine-tune, 不直接optimize visual robustness。

如果要RL在vision上也有优势, 可能需要:
- Domain randomization on textures during RL training (让RL exposure更广的visual perturbation)
- 或者用contrastive reward来encourage visual invariance

### 11.3 PPO vs GRPO的深层原因

GRPO的 $\hat{A}_t^i = (r^i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})$ 是**outcome-level**的advantage, 所有steps in trajectory $i$ 共享同一个advantage值。这在LLM的reasoning task上work, 因为reasoning的"quality"可以由final answer决定。但robotics的trajectory是**dense control sequence**, 不同step的action quality差异很大, outcome-level advantage无法区分"哪一步action是好的"。

PPO的GAE通过 $V(s)$ 把credit spread到temporal维度, 每步action都有自己的advantage estimate, 这对sequential decision making更合理。

### 11.4 Recipe的实用性

42小时单A100, LoRA rank=32, PPO epoch=1, shared backbone — 这是个**非常production-friendly**的recipe。相比之下, FLaRe (Hu 2024, https://arxiv.org/abs/2409.16578) 用了大量computation。这个recipe让RL fine-tuning VLA变得accessible, 期待community在此基础上扩展。

### 11.5 Limitations & Future Directions

1. **只用motion planner demo**: 真实human data有更多variability, SFT/RL的gap可能不同
2. **只有pick-and-place**: 需要扩展到long-horizon, multi-task (论文承认)
3. **Sim-to-real只初步验证**: 27%成功率说明gap还很大, 需要更systematic的real-world eval
4. **Vision维度RL没优势**: 这是个limitation, 也是future work方向 — 能否设计RL training让visual robustness也提升?
5. **Reward design**: 当前是sparse stage reward, 更dense的reward (e.g., distance to goal) 可能进一步提升RL sample efficiency

---

## 12. 参考链接

- **Project page**: https://rlvla.github.io
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **PPO (Schulman 2017)**: https://arxiv.org/abs/1707.06347
- **GAE (Schulman 2015)**: https://arxiv.org/abs/1506.02438
- **GRPO / DeepSeekMath**: https://arxiv.org/abs/2402.03300
- **DPO**: https://arxiv.org/abs/2305.18290
- **Open-Reasoner-Zero (ORZ)**: https://arxiv.org/abs/2503.24290
- **SFT memorizes, RL generalizes (Chu 2025)**: https://arxiv.org/abs/2501.17161
- **FLaRe**: https://arxiv.org/abs/2409.16578
- **ManiSkill3**: https://arxiv.org/abs/2410.00425
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **RT-2 action discretization**: https://arxiv.org/abs/2307.15818
- **Objaverse**: https://objaverse.allenai.org
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **SigLIP**: https://arxiv.org/abs/2303.15319
- **Llama-2**: https://arxiv.org/abs/2307.09288
- **LoRA**: https://arxiv.org/abs/2106.09685
- **Ross & Bagnell (DAgger)**: https://proceedings.mlr.press/v9/ross10a.html
- **GRAPE (TPO)**: https://arxiv.org/abs/2411.19309
- **Stable Diffusion**: https://arxiv.org/abs/2011.13456
- **ControlNet**: https://arxiv.org/abs/2308.16980

---

## 总结

这篇paper的核心message是: **RL fine-tuning对VLA的generalization收益是真实的、可量化的、且维度specific的**。具体来说:
- **Execution维度**: 巨大收益 (RL学到了recovery behaviors, SFT只会模仿)
- **Semantics维度**: 中等收益 (RL对unseen objects更robust, 因为学到的是skill而非memorization)
- **Vision维度**: 平手 (visual robustness来自pretrained encoder, RL/SFT都marginal)

而PPO是VLA RL fine-tuning的最优选择, 因为robotic POMDP的non-stationarity需要显式的value estimation (GAE), 这正是GRPO的group-relative baseline和DPO的offline preference都缺乏的。Recipe本身很简洁: shared backbone + warm-up + epoch=1, 42小时单A100就跑完。

Andrej, 这篇工作的positioning很好 — 它不claim发明新算法, 而是把LLM时代的RL经验systematic地translate到VLA, 并且honestly地report了RL的limitation (Vision维度没优势)。这种empirical study对community的价值远大于又一个"new algorithm paper"。
