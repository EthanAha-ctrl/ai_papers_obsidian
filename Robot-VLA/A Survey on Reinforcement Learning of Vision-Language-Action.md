---
source_pdf: A Survey on Reinforcement Learning of Vision-Language-Action.pdf
paper_sha256: 9205061ef458f560ad6b2c6cfc4c94ec13940d4d93096a73734a306297539260
processed_at: '2026-07-17T21:53:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# RL-VLA Survey 深度解读

这篇 paper 是 2025 年 12 月发表的综述, 系统性地梳理了 **Reinforcement Learning of Vision-Language-Action (VLA) models for robotic manipulation** 这个新兴领域。我帮你 build intuition, 从 motivation 到 architecture 到 deployment 全链路梳理。

---

## I. 核心 Motivation: 为什么 VLA 需要 RL?

### 1.1 VLA 的根本 limitation

当前 VLA models (OpenVLA, π0, RT-2, π0.5 等) 主要依赖 **imitation learning (IL)**, 通过大规模 teleoperation datasets 获取 general visuomotor priors。这有几个根本性缺陷:

1. **OOD generalization 差**: demonstrations 覆盖的 states/actions space 有限, 部署时遇到 OOD scenario 容易失败
2. **无 failure recovery demonstrations**: 数据集大多是成功 trajectory, 机器人不知道怎么从 failure 状态恢复
3. **纯 imitative objective**: 无法 explore 可能更优的策略

Reference: [OpenVLA](https://openvla.github.io/), [π0](https://www.physicalintelligence.company/blog/pi0), [RT-2](https://robotics-transformer2.github.io/)

### 1.2 RL 的价值 proposition

RL 通过 **self-exploration** 和 **result-driven optimization** 弥补这些缺陷。在 LLM 领域 RL post-training (如 DeepSeek-R1 用 GRPO) 已经证明可以显著提升 reasoning 能力, 这个 insight 自然地 transfer 到 VLA:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right]$$

**变量解释**:
- $\pi = \pi_\theta(a_t | s_t)$: policy, 参数化为 $\theta$ (神经网络 weights)
- $\tau = (s_0, a_0, s_1, a_1, \ldots)$: trajectory, 由 policy $\pi$ 在 environment 中 rollout 产生
- $T$: task horizon (episode 最大步数)
- $\gamma \in [0, 1)$: discount factor, 折扣未来 reward (越远 future reward 越不重要)
- $r(s_t, a_t)$: reward function, 在 state $s_t$ 执行 action $a_t$ 获得的即时回报
- $\mathbb{E}_{\tau \sim \pi}$: 对 policy $\pi$ 产生 trajectory 的期望

**Intuition**: IL 最小化 $\mathcal{L}_{BC} = -\log \pi_\theta(a_{expert}|s)$, 只关心 expert 分布; RL 优化 $J(\pi)$, 关心 long-term return, 即使是非 expert 路径只要 return 高就行, 这正是 OOD 场景需要的灵活性。

---

## II. RL-VLA 的形式化: MDP 定义

### 2.1 State Space $S$

State 是 multimodal 且 high-dimensional:

$$s_t = (o_t^{vis}, o_t^{prop}, l_{task})$$

- $o_t^{vis}$: visual observation, 通常是 multi-view RGB images 或 point clouds
- $o_t^{prop}$: proprioceptive information, 比如 joint angles, end-effector pose (位置+姿态)
- $l_{task}$: language instruction (e.g., "pick up the red cup and place it on the plate")

### 2.2 Action Space $\mathcal{A}$

Action $a_t \in \mathbb{R}^d$ 由 VLA decoder 生成。关键点是 VLA 通常输出 **action chunk**:

$$a_{t:t+k-1} = (a_t, a_{t+1}, \ldots, a_{t+k-1})$$

这是为了减少 latency (一次预测多步), 用 diffusion decoder 或 action tokenizer 实现。这给 RL 带来挑战: 传统 RL 是 single-step action, 而这里要处理 sequence-level action。

### 2.3 Reward $r(s_t, a_t)$

通常组合 sparse + dense:
- **Sparse binary signal**: task success (0/1)
- **Dense process reward**: e.g., $-\|p_{ee} - p_{target}\|_2$ (end-effector 到目标距离)

### 2.4 Transition $p(s_{t+1} | s_t, a_t)$

可以是:
- Simulation (Isaac Sim, MuJoCo)
- Real-world physics (implicit, 通过 robot hardware 实现)

---

## III. RL-VLA Architecture: Action, Reward, Transition 三大模块

### 3.1 Action Modeling

这是最关键的设计 decision, 决定 RL 怎么和 VLA 集成。

#### (1) Autoregressive VLA (Token-level RL)

**架构**: 类似 LLM 的 next-token prediction。Action 被 discretize 成 tokens $a = (a^{(1)}, a^{(2)}, \ldots, a^{(L)})$, VLA autoregressive 生成。

**RL formulation**: 每个 token 对应一个 RL decision, 可以用 PPO/GRPO 等 policy gradient 方法:

$$\nabla_\theta J(\pi) = \mathbb{E} \left[ \sum_{l=1}^{L} A_l \nabla_\theta \log \pi_\theta(a^{(l)} | s, a^{(<l)}) \right]$$

其中 $A_l$ 是 token $l$ 的 advantage。

**代表工作**:
- **TGRPO** (Trajectory-wise Group Relative Policy Optimization): 把 policy gradient 重写为 token-level cross-entropy loss, 用 advantage weighting。关键是它不改 action head 结构, 保留了 VLA 原有能力
- **CO-RFT** (Chunked Offline RL Fine-Tuning): 利用 action probability 的 spatio-temporal dynamics, 解决 autoregressive VLA 离散 action 预测的 trajectory 一致性问题
- **SimpleVLA-RL**: 用 GRPO 在 LIBERO benchmark 上取得显著提升
- **DeepThinkVLA**: 引入 Chain-of-Thought (CoT) + causal attention, 用 GRPO causally align reasoning-action sequence

**挑战**: Discrete token design 有 fundamental tradeoff:
- Coarse tokenization → 丢失 dexterous control 精度
- Fine-grained tokenization → token 间 discrimination 难, prediction 难度激增

#### (2) Generative Action VLA (Sequence-level RL)

**架构**: 用 diffusion 或 flow-matching 作为 action head, 直接生成连续 action trajectory。

$$a_{t:t+k-1} = \text{Denoise}(\epsilon, \text{conditioning})$$

其中 $\epsilon$ 是 initial noise, 通过多步 denoising 生成 action sequence。

**挑战**: Diffusion/flow-matching 没有 explicit action probability, 传统 RL policy gradient $\nabla \log \pi$ 没法直接算。

**解决方案**: 通过 reparameterization 近似概率。

**$\pi_{RL}$ 的方法** (参考 [πRL paper](https://arxiv.org/abs/2510.25889)):

引入两个变体:

**(a) Flow-SDE**: 把 denoising process 建模为 discrete-time MDP。Denoising step $i \to i+1$ 视为 state transition, denoising policy $\pi_\phi(a_i | s_i)$ 可计算 probability, 满足 RL 更新要求。

**(b) Flow-Noise**: 把 denoising noise prediction 作为 action, 在 noise space 计算 policy gradient。

**FPO (Flow Policy Optimization)**: 用 importance sampling 改善 efficiency:

$$\nabla_\theta J \approx \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A \nabla_\theta \log \pi_\theta(a|s) \right]$$

其中 importance ratio $\frac{\pi_\theta}{\pi_{old}}$ 控制更新幅度, 避免 distribution drift。

**ARFM (Adaptive Reinforced Flow Matching)**: 用 dynamic scaling factor 调整每个 sample 的 weight, 提升 sample utilization efficiency。

**挑战**: 近似 probability 不完美, multi-step generation + iterative update 会让 small mismatch 累积, 最终 distort 或 collapse action distribution。

#### (3) Dual-system VLA (Bridge-level RL)

**架构**: 高层 VLM (System 2 thinking) + 低层 VLA (System 1 fast control)。

- VLM 理解 human intent, 生成 sub-tasks
- VLA 执行 sub-tasks, 输出 trajectories
- RL 在 bridge 层: 决定哪个 high-level action proposal 传给 fast controller

**代表工作**: **Hume** (参考 [Hume paper](https://arxiv.org/abs/2505.21432))
- 引入 "value-guided thinking" 过程
- 生成多个 action candidates, 用 specialized value-query head 选择最优

**挑战**: VLM 和 VLA 的 heterogeneous representations 和不同 timescales 导致 value estimates 不一致, 联合 RL training 不稳定。

### 3.2 Reward Design

Reward 是 RL 的核心 learning signal, 直接决定 policy 收敛 quality。

#### (1) Intrinsic Rewards

##### a) Potential-based Reward Shaping (PBRS)

原始 reward $r(s, a, s')$ reshape 为:

$$r'(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

**变量解释**:
- $r(s, a, s')$: 原始 reward
- $\gamma$: discount factor
- $\Phi(s)$: potential function, 衡量 state $s$ 的 "潜力"
- $\gamma \Phi(s') - \Phi(s)$: potential difference, 鼓励 agent 移动到高 potential state

**关键理论**: PBRS 不改变 optimal policy (Ng et al. 1999, [paper](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf))。这保证了我们可以安全地 shape reward 而不破坏 optimality。

$\Phi(s)$ 的设计 choice:
- **手动设计**: distance-to-goal, energy reduction
- **学习得到**: approximated value function, latent progress estimator

##### b) Exploration-driven Rewards

- **Curiosity-driven** ([Pathak et al.](https://pathak22.github.io/noreward-rl/)): reward = prediction error of forward dynamics model。高 error = novel state
- **RND (Random Network Distillation)** ([Burda et al.](https://openai.com/research/random-network-distillation)): 训练一个 predictor network 拟合 fixed random target network, prediction error 衡量 state novelty
- **Count-based**: 奖励 under-visited states

**挑战**: Intrinsic reward 不 align with task objective, 可能导致:
- **Reward hacking**: agent exploit intrinsic reward 的漏洞
- **Reward collapse** in high-dimensional spaces
- 在 long-horizon task 中, 大部分 novel states 是 task-irrelevant 的

#### (2) Extrinsic Rewards

##### a) Human-aligned Rewards

**RLHF** (参考 [InstructGPT](https://arxiv.org/abs/2203.02155)):
1. 收集 human preference comparisons $(a_1, a_2)$ pairs
2. 训练 reward model $r_\phi(s, a)$
3. 用 RL (PPO) 优化 policy w.r.t. $r_\phi$

**代表工作**:
- **SEED**: 用 evaluative feedback 克服 reward sparsity
- **DemPref**: 迭代查询 preference labels on policy-generated trajectories
- **Sirius**, **Transic**: 人类在 training 中 refine reward functions

##### b) Model-generated Rewards (这是最 promising 的方向)

利用 foundation models (LLM/VLM) 生成 reward, 实现 scalable supervision:

**Eureka** ([Ma et al.](https://eureka-research.github.io/)):
- LLM iteratively 生成 reward code proposals
- Environment feedback 评估 reward quality
- LLM 根据 feedback 进化 reward
- 实验显示 often 超过 expert-designed rewards

**Video-based rewards**:
- **VIPER** ([paper](https://arxiv.org/abs/2310.07248)): 学习 video prediction transformer 从 expert demos, 用 model likelihood 作为 reward
- **TeViR**: text-to-video diffusion 生成 predicted image sequence, 与 actual observation 对比计算 reward

**Query-based (直接 query VLM)**:
- **RoboCLIP** ([paper](https://arxiv.org/abs/2310.07888)): 用 CLIP-style 模型 query image + text
- **RL-VLM-F**: 直接 query VLM 生成 reward from image + text description
- **GVL** ([paper](https://arxiv.org/abs/2402.10236)): formulate reward estimation 为 temporal ordering of video frames
- **VLAC**: contrastive learning with negative samples

**核心 principle**: reward = distributional alignment。当 agent behavior matches expert distribution 或 internet-scale video distribution 时, reward 高。

**挑战**: 
- Mis-specification
- Domain shift (VLM 在 internet 数据训练, 不一定 align with robot task)
- Perceptual noise

### 3.3 Transition Modeling

Transition modeling 让 VLA 能进行 predictive rollout, 评估 action sequence 的 long-term effect。

#### (1) Physics-based Simulator

- **Isaac Sim** ([link](https://developer.nvidia.com/isaac-sim)): GPU-accelerated, 高 fidelity
- **Gazebo** ([link](http://gazebosim.org/)): 经典 ROS-compatible simulator

**优点**: 物理 accurate
**缺点**: 
- 构建 high-fidelity simulator 需要大量人工
- 物理参数 annotation 不准确
- Computational cost 高, 限制 scalability

#### (2) Learning-based World Model

##### a) State-based Methods

在 compact latent space 建模 transition:

**PlaNet** ([paper](https://planetrl.github.io/)):
- Recurrent State-Space Model (RSSM)
- Predict future latent states 和 rewards
- 不 reconstruct full visual observation

**Dreamer** ([paper](https://danijar.com/dreamer/)), **DreamerV2** ([paper](https://arxiv.org/abs/2010.02193)):
- 增强 latent state space expressiveness
- 实现 long-horizon planning

**TransDreamer**: 用 transformer 替代 recurrent architecture, 更稳定的 long-horizon prediction

**局限**: 把 image reconstruction 作为 auxiliary objective, 视觉建模不足, real-world generalization 有限

##### b) Observation-based Methods

直接在 pixel level 建模 transition:

**iVideoGPT** ([paper](https://thuml.github.io/iVideoGPT/)):
- 基于 large autoregressive video prediction model
- Fine-tune pretrained model for robotic scenarios
- 结合 learned reward model, 可作为 neural simulator for MBRL

**GWM (Gaussian World Model)** ([paper](https://arxiv.org/abs/2508.17600)):
- 多模态数据 encoding
- 更好地 capture 3D geometric structure
- 在 MBRL 任务上提升 visual quality 和 reward prediction accuracy

**EmbodiedDreamer** ([paper](https://arxiv.org/abs/2507.05198)):
引入两个 component:
- **PhysAligner**: incorporate physics simulator priors, 提供 physically consistent transition dynamics
- **VisAligner**: 用 video painting techniques 增强 generated observation 的 realism

##### c) VLA-designed Methods (最前沿)

将 world model 直接集成进 VLA framework:

**VLA-RFT** (Vision-Language-Action Reinforcement Fine-Tuning):
1. VLA 生成 action sequence $a_{t:t+k}$
2. World model 生成多个 rollouts (predicted future states)
3. Reward model 评估这些 rollouts
4. 用 **GRPO** 更新 VLA

GRPO objective (简化版):

$$\mathcal{L}_{GRPO} = -\mathbb{E}_{g \sim \mathcal{G}} \left[ \frac{1}{|G|} \sum_{i=1}^{|G|} \min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) \right]$$

其中 $g$ 是 group, $|G|$ 是 group size, $\rho_i = \frac{\pi_\theta(a_i|s)}{\pi_{old}(a_i|s)}$ 是 importance ratio, $\hat{A}_i$ 是 advantage estimate, $\epsilon$ 是 clip parameter。GRPO 相比 PPO 不需要 value function, 用 group-relative advantage 代替。

**World-Env** ([paper](https://arxiv.org/abs/2509.24948)):
- VLA → action sequence
- World model → future observations
- VLM → semantic reflections
- LOOP (Leave-One-Out Proximal Policy Optimization) 更新 policy

**WMPO** (World Model-based Policy Optimization):
- 生成 pixel-level imagined trajectories
- 用 learned reward model 评估
- GRPO 优化 policy
- **不需要与 real environment 交互**

**挑战**: World model 跨 scene, embodiment, robot morphology generalization 差; balancing data-driven learning 与 physics-consistent dynamics 仍是核心难题。

---

## IV. Training Paradigms

### 4.1 Online RL-VLA

Agent 与 environment 直接交互, trial-and-error。5 个核心方向:

#### (1) Policy Optimization

主流是 PPO variants, 但有不同变体:

**PPO baseline** ([Schulman et al.](https://arxiv.org/abs/1707.06347)):
- **FLaRe**: 第一个 PPO post-training VLA 的工作
- **RLRC**: PPO fine-tune VLA
- **RIPT-VLA**: RLOO (Leave-One-Out) advantage estimation + PPO, 不需要 shaped reward 或 value function
- **VLA-RL**: PPO + Robotic Process Reward Model (用 VLM 作为 process reward)
- **SimpleVLA-RL**: 用 GRPO 在 LIBERO 上取得大幅提升
- **RLVLA** ([paper](https://arxiv.org/abs/2505.19789)): empirical 比较 DPO, PPO, GRPO, 证明 RL fine-tuning 显著提升 OOD generalization

**Flow-based optimization**:
- **FPO**: Flow Policy Optimization, importance sampling for flow-matching VLA
- **$\pi_{RL}$**: Flow-Noise + Flow-SDE

**Preference alignment**:
- **GRAPE** ([paper](https://arxiv.org/abs/2411.19309)): trajectory-wise preference optimization
- **RobustVLA**: lightweight online RL + Jacobian regularization, 增强 robustness against perturbations

#### (2) Sample Efficiency

Real-world interaction 极其昂贵, 需要 sample-efficient 方法:

**RLDG** ([paper](https://arxiv.org/abs/2412.09858)):
- Distill knowledge from generalist policy (在 diverse datasets 训练)
- 用 targeted exploration/exploitation 加速 new task learning

**iRe-VLA** ([paper](https://arxiv.org/abs/2501.16664)):
- Two-stage: SFT warmup → online RL
- 显著减少达到 proficient performance 所需的 interactions

**VLAC** ([paper](https://arxiv.org/abs/2509.15937)):
- Actor-critic architecture within single VLM
- 同时输出 action + dense progress delta + done signal
- 大幅提升 sample efficiency

**SRPO** ([paper](https://arxiv.org/abs/2511.15605)):
- **Self-referential RL**: policy 用自己的 successful trajectories 作为 self-reference
- Progressive rewards
- 不需要 reward labeling

#### (3) Active Exploration

Random rollout 浪费资源, 需要 intelligent exploration:

**Plan-Seq-Learn** ([paper](https://arxiv.org/abs/2405.01534)):
- LLM 生成 high-level task plans
- 转换为 motion-planning waypoints
- 训练 low-level vision-based RL policy follow waypoints

**SIME**: modal-level exploration, 生成 diverse multi-modal interaction behaviors

**SOE**: 学 latent representation of task-relevant factors, constrain exploration to valid action manifold

**ASID** ([paper](https://arxiv.org/abs/2404.13455)):
- Active exploration policy 收集少量 informative real-world data
- 识别 unknown physical parameters
- 构建更 accurate simulator

**RESample**: 自动生成 challenging OOD data, 用 exploratory sampling 创建 failure + recovery trajectories

**PLD** ([paper](https://arxiv.org/abs/2511.00091)):
- Hybrid rollout scheme
- Bias residual interventions toward states frequently visited by base policy
- Align 收集的 trajectories with generalist deployment distribution
- 同时 capture recovery behaviors

#### (4) Training Stability

RL training 不稳定是常见问题, 几个策略:

**RIPT-VLA**: Dynamic Rollout Sampling (rejection sampling) 处理 high variance in rollout returns

**ConRFT**: offline RL pre-training (Cal-QL) → online fine-tuning (HIl-SERL framework)

**TGRPO**: trajectory-level estimation 减小 policy update variance

**World-Env, VLA-RFT**: world model 生成 synthetic rollouts, 减少 real-world interaction 的 variance

#### (5) Online RL-VLA Infrastructure

**RLinf** / **RLinf-VLA** ([paper](https://arxiv.org/abs/2510.06710)):
- 灵活 infrastructure supporting 各种 policy optimization 算法
- 支持不同 model architectures
- 集成 human feedback 和 safety constraints

借鉴 LLM 训练 infrastructure:
- **vLLM** ([link](https://vllm.ai/)): efficient inference
- **VeRL** / **HybridFlow** ([paper](https://arxiv.org/abs/2409.19256)): RLHF framework

### 4.2 Offline RL-VLA

在 static dataset 上训练, 不与环境交互。适合 high-risk 或 resource-limited 场景。

#### (1) Data Utilization

##### Customized Representation

**ReinboT** ([paper](https://arxiv.org/abs/2505.07395)):
- 修改 offline dataset 最大化 cumulative rewards
- 用 Decision Transformer + Return-To-Go (RTG)
- 比 behavior cloning 更 robust

**$\pi^*_{0.6}$** ([paper](https://arxiv.org/abs/2511.14759)):
- 用 pretrained value function 条件化 VLA with binarized value
- 同时利用 failure 和 success data

**NORA-1.5** ([paper](https://arxiv.org/abs/2511.14659)):
- Offline Direct Preference Optimization (DPO)
- Model-generated rewards

DPO objective (参考 [DPO paper](https://arxiv.org/abs/2305.18290)):

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(s, a_w, a_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(a_w|s)}{\pi_{ref}(a_w|s)} - \beta \log \frac{\pi_\theta(a_l|s)}{\pi_{ref}(a_l|s)}\right) \right]$$

其中 $a_w$ 是 preferred action, $a_l$ 是 dispreferred action, $\pi_{ref}$ 是 reference policy, $\beta$ 是 KL penalty strength。

##### Conservative Constraint

**ConRFT** ([paper](https://arxiv.org/abs/2502.05450)):
- BC + **Cal-QL** ([paper](https://arxiv.org/abs/2302.02535))
- Cal-QL 在 Conservative Q-Learning (CQL) 基础上 calibrated, 适合 small datasets

CQL objective:

$$\mathcal{L}_{CQL} = \alpha \left[ \mathbb{E}_{s \sim \mathcal{D}} \log \sum_a \exp(Q(s,a)) - \mathbb{E}_{(s,a) \sim \mathcal{D}} Q(s,a) \right] + \mathcal{L}_{TD}$$

第一项惩罚 OOD actions (高 Q for unseen actions), 第二项是 standard TD loss。

**CO-RFT**: 用 Cal-QL 的 calibration 机制 constrain policy training, 与 action chunking 兼容

#### (2) Objective Modification

**Architecture-aware**: Q-Transformer, PAC (Perceiver-Actor-Critic), ARFM (flow-based offline RL)

**Data-driven**: RL-100 用 offline RL conservatively gate online PPO agent, 生成 high-quality data

### 4.3 Test-time RL-VLA

Deployment 时 lightweight adapt, 不更新 full model。

#### (1) Value Guidance

**V-GPS** ([paper](https://steering-generalists.github.io/)):
- Pre-trained value function re-rank action candidates
- 选择 highest predicted value 的 action
- **不需要 weight updates**

**Hume** ([paper](https://arxiv.org/abs/2505.21432)):
- "Value-guided thinking" 过程
- 生成多个 action candidates
- Value-query head 选择最优

#### (2) Memory Buffer Guidance

**STRAP** ([paper](https://arxiv.org/abs/2411.19025)):
- Compact pattern library 存储 representative spatiotemporal patterns
- 推理时基于 similarity 检索 trajectory sub-segments

**RA-DT**: 外部 memory 存储 past experiences, retrieval-augmented in-context decision making

**ReSA**: 识别并 selectively imitate high-quality successful trajectories, 通过 intrinsic quality assessment

#### (3) Planning-guided Adaptation

**VLA-Reasoner** ([paper](https://arxiv.org/abs/2509.22643)):
- Online **Monte Carlo Tree Search (MCTS)**
- Base policy 的 initial action prediction 作为 search 起点
- Simulate future outcomes 寻找更优 action

MCTS 的 4 个步骤:
1. **Selection**: 根据 UCB 公式选择 node
2. **Expansion**: 展开 unvisited children
3. **Simulation**: rollout 到 terminal state
4. **Backpropagation**: 更新 node statistics

UCB1 公式:
$$UCB1 = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}$$

其中 $w_i$ 是 node $i$ 的 win count, $n_i$ 是 visit count, $N$ 是 parent visit count, $c$ 是 exploration constant。

**BGR (Bellman-Guided Retrials)** ([paper](https://arxiv.org/abs/2406.15917)):
- 训练 value function 估计 time-to-completion
- Test-time continuous monitoring
- Detect inconsistency → trigger corrective actions

**挑战**: Pre-inference future action sequence 计算成本高, 限制 real-time deployment。

---

## V. Real-world Deployment

### 5.1 Sim-to-Real Transfer

#### (1) Domain Randomization (DR)

随机化 simulator parameters (lighting, texture, actuator noise) 期望涵盖 real-world 多样性。

**SimpleVLA-RL**: 跨多个 task simulations 应用 DR, 实现 zero-shot transfer 到 real robots, 不需要额外 fine-tuning。

#### (2) Digital Twin (DT)

创建 physical system 的 synchronized virtual replica:

**Real-Is-Sim** ([paper](https://arxiv.org/abs/2504.03597)): 动态 DT, 用 real sensor streams 持续校正, 保证 policies 总在 familiar simulator-domain states 操作

**RialTo** ([paper](https://arxiv.org/abs/2406.07852)): 从 minimal real data 构建 on-the-fly simulations, inverse-distillation RL robustify manipulation policies

**RoboTwin** ([paper](https://arxiv.org/abs/2506.18088)): 生成框架, 用 3D generative models 和 LLMs, 从 single 2D image 转换为 diverse interactive DTs

**DREAM** ([paper](https://arxiv.org/abs/2502.02443)): differentiable Gaussian Splat 创建 high-fidelity DTs, 同时识别 object mass + 训练 force-aware grasping policies

**挑战**: SimpleVLA-RL 显示 sim-to-real gap 显著, real-robot success rate 远低于 simulation。

### 5.2 Real-world RL

#### (1) Human-in-the-loop (HiL) RL

##### Human Corrective Intervention

**HIL-SERL** ([paper](https://hil-serl.github.io/), [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.ads5033)):
- Human 实时 corrective feedback
- 快速 acquire precise dexterous manipulation skills
- Key insight: human intervention 稳定 learning, 减少 unsafe exploration

**CR-DAgger**: compliant force-sensitive interface + residual policy 用 force feedback

**TRANSIC** ([paper](https://transic-robot.github.io/)): sim-to-real framework, 从 online human corrections 学习 adaptive policy transfer

**Genie Centurion** ([paper](https://arxiv.org/abs/2505.18793)):
- 跨多 robots scale corrective intervention
- VLM 检测 task failures
- 必要时 request human assistance

**ConRFT**: 第一个 integrate HiL interventions into RL for VLA in real-world, 结合 offline + online human-corrected RL fine-tuning

**DAFT**: 自然语言 feedback → semantically grounded corrective actions

##### Human Recovery Assistance

Manual reset robots/environments after failures。

**ARMADA**: learning-based recovery modules, detect failure states, request human-guided recovery

**Generalist**: 限制 human intervention 到 irreversible states 或 prolonged failure

##### Human Curriculum Task Design

**CurricuLLM** ([paper](https://arxiv.org/abs/2406.12207)):
- LLM 自动 decompose complex skills 到 hierarchical sub-tasks
- Align task progression with human-specified difficulty

**Sirius** ([paper](https://arxiv.org/abs/2306.09831)):
- Human-in-the-loop autonomy framework
- Human operators 动态 design 和 gate deployment curricula

**MT-Opt** ([paper](https://arxiv.org/abs/2104.08212)): fleet scale, prioritize low-performing skills, 控制 deployment thresholds

#### (2) Reversibility and Autonomous Recovery

##### Reset-free Learning

**LNT** ([paper](https://arxiv.org/abs/1711.06782)):
- Train goal-conditioned reset policies
- Restore agents 到 initial state distribution

**VaPRL**: curriculum learning + value-aware

**MEDAL** ([paper](https://arxiv.org/abs/2205.05212)):
- Demos guide both task 和 reset policies
- Unified framework

**IBC** ([paper](https://arxiv.org/abs/2307.04018)): 学习 reset goals directly from demonstrations

**MTRF**: 把 reset-free RL 视为 multi-task learning, task terminal states 作为其他 task 的 initial states

##### Functional Reversibility

**Recovery RL** ([paper](https://arxiv.org/abs/2010.11430)):
- Learn recovery policy 干预, 阻止 robot 进入 unsafe/irreversible states
- Learned recovery zones

**PAINT**: 训练 classifier 预测 potential failures, proactive corrective actions

##### Semantic-aware Recovery

**RECOVER**: ontology + logic + language models 检测 failures, 产生 interpretable recovery plans

**Ahmad et al.**: VLM + behavior trees for real-time failure reasoning 和 autonomous correction

#### (3) Safe Exploration

##### Conservative Safety Critics

**Recovery RL**: learned recovery zones
**SLAC** ([paper](https://arxiv.org/abs/2506.04147)): pre-train task-agnostic latent action space in low-fidelity simulation, constrain real-world exploration

##### Structured Task Decomposition

**GRAPE**: VLM decompose complex tasks, 自动 derive spatiotemporal safety constraints using semantic keypoints

##### Real-time Safety Enforcement

**Impedance controllers**: bound end-effector forces 和 velocities in real time

**SafeVLA** ([paper](https://arxiv.org/abs/2503.03480)):
基于 **Constrained Markov Decision Process (CMDP)**:

$$\max_\pi \mathbb{E}\left[\sum_t \gamma^t r(s_t, a_t)\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_t \gamma^t c(s_t, a_t)\right] \leq d$$

其中 $c(s_t, a_t)$ 是 cost function (e.g., collision risk), $d$ 是 cost budget。Min-max perspective against elicited safety risks, 实现 safety-performance tradeoff。

---

## VI. Benchmarks and Evaluation

### 6.1 Simulation Benchmarks

| Benchmark | Tasks | Robot | Parallel | Gym | GPU |
|-----------|-------|-------|----------|-----|-----|
| **LIBERO** ([link](https://lifelong-robot-learning.github.io/libero/)) | 130 | Franka | ✓ | ✓ | ✗ |
| **Meta-World** | 50 | Sawyer | ✓ | ✓ | ✗ |
| **ManiSkill3** ([link](https://maniskill.readthedocs.io/)) | 100+ | Franka | ✓ | ✓ | ✓ |
| **BEHAVIOR** | 1k | Humanoid | ✓ | ✓ | ✗ |
| **RoboVerse** | 1k+ | Multi | ✓ | ✓ | ✓ |
| **RoboCasa** | 100 | Franka (wheeled) | ✗ | ✓ | ✗ |
| **CALVIN** ([link](https://calvinrobot.github.io/)) | 34 | Franka | ✗ | ✗ | ✗ |
| **SIMPLER** ([link](https://simpler-env.github.io/)) | 8 | Google/WidowX | ✓ | ✓ | ✓ |
| **RoboTwin2.0** | 50 | Multi (dual-arm) | ✓ | ✓ | ✓ |

### 6.2 Real-world Benchmarks

**LeRobot** ([link](https://github.com/huggingface/lerobot)):
- Open-source framework
- 统一 dataset organization, data acquisition, training-evaluation pipeline
- Packaging deployable policy artifacts

**SERL** ([link](https://serl.readthedocs.io/)):
- Real-robot RL suite
- Off-policy vision-based learner
- Practical components: reward specification, automated resets, safe control
- Tasks: PCB insertion, cable routing

**FurnitureBench** ([link](https://furniturebench.github.io/)):
- Furniture assembly
- 3D-printable parts
- Skills: grasping, insertion, screwing

**FMB** (Functional Manipulation Benchmark):
- 3D-printed objects
- Multi-view RGB-D + CAD assets
- Staged skills: grasping, fixture-assisted reorientation, precise insertion

### 6.3 Evaluation Metrics

1. **Success Rate**: episode 完成目标的比例
2. **Average Episodic Return**: $\bar{R} = \mathbb{E}\left[\sum_t \gamma^t r_t\right]$, 反映 efficiency + stability
3. **Safety Cost** (SafeVLA): constraint violation 程度
4. **Cycle Time** (RLDG, CO-RFT): real-world learning cycle 时间 (data collection → policy update → deployment)
5. **Episode Length** (ConRFT): 短 episode 通常 policy 不稳定
6. **Intervention Rate** (ConRFT): human 干预频率, 反映 autonomy degree

---

## VII. Open Challenges 与 Future Directions

### 7.1 Scaling to Long-horizon Tasks

RL 只 supervise final actions, 缺乏 intermediate reasoning guidance。

**Promising solutions**:
- Chain-of-Thought-like supervision
- Memory-retrieval mechanisms (STRAP, RA-DT, MAP-VLA)
- Coupling structured reasoning with sequence modeling

### 7.2 Model-based RL for VLA

当前 RL-VLA 依赖 massive simulated rollouts, sample efficiency 差。

**Promising solutions**:
- Predictive world models 学习 dynamics
- Generate informative rewards + synthetic states
- VLA-RFT, World-Env, WMPO 是 promising 方向

### 7.3 Efficient and Scalable Real-robot Training

Real-robot training inefficiency: limited parallelization, heavy human supervision。

**Promising solutions**:
- Reason agents for automatic failure handling
- Reactive agents for safe exploration
- Multi-robot shared training + real-to-sim simulator rollouts
- Generalist 的 self-improving 框架

### 7.4 Reliable and Reproducible RL-VLA

Multimodal RL 对 design choices, hyperparameters, stochastic dynamics 高度敏感。

**Promising solutions**:
- Consistent training pipelines
- Controlled evaluation environments
- Standardized algorithmic settings reporting

### 7.5 Safe and Risk-aware RL-VLA

Real-world 中 imperfect perception, delayed control, limited supervision 导致 irreversible risks。

**Promising solutions**:
- Predictive risk modeling
- Constraint-based policy optimization
- Language-conditioned safety reasoning (SafeVLA 方向)

---

## VIII. 关键 Insights 总结

### 8.1 RL 与 VLA 的 synergistic relationship

- **VLA → RL**: VLA 携带 rich multimodal representations, 显著提升 RL sample efficiency
- **RL → VLA**: RL 使 VLA 超越 suboptimal pretraining behaviors, 实现 OOD generalization

### 8.2 Architecture 选型 tradeoff

| Architecture | RL 作用层级 | 优势 | 挑战 |
|-------------|------------|------|------|
| Autoregressive | Token-level | 直接 probability, stable update | Discrete token 限制 dexterous control |
| Generative (Diffusion/Flow) | Sequence-level | 连续 action, 高 expressiveness | 无 explicit probability, 需近似 |
| Dual-system | Bridge-level | 分层 reasoning + control | Value misalignment 不稳定 |

### 8.3 Reward 设计 trend

从 human-aligned (RLHF) → model-generated (Eureka, RL-VLM-F) → self-referential (SRPO), 自动化程度递增, scalability 提升。

### 8.4 Training paradigm trend

- **Offline RL**: 适合 risk-averse scenarios, 但 data quality 限制大
- **Online RL**: OOD generalization 最强, 但 sample efficiency + safety 是瓶颈
- **Test-time RL**: 轻量 adaptation, 但 compute cost 限制 real-time deployment

### 8.5 Deployment trend

Sim-to-real → real-world RL, 重点从 domain randomization → autonomous recovery + safe exploration。

---

## IX. 我的 Intuition: 这个领域的核心 tension

阅读这篇综述后, 我 build 的 intuition 是 RL-VLA 领域存在几个 fundamental tensions:

1. **Imitation vs Exploration**: VLA pretrained on expert demos 倾向 conservative behavior, 但 RL 需要 exploration 探索非 expert 策略。如何平衡 KL constraint 和 exploration 是关键。Recovery RL, SafeVLA 等 work 都在处理这个 tension。

2. **Discrete tokens vs Continuous control**: LLM 范式的 autoregressive token prediction 与 robot control 需要的 continuous precision 之间存在 fundamental gap。FPO, $\pi_{RL}$ 等 flow-based 方法试图 bridge, 但 approximate probability 不完美。

3. **Simulation fidelity vs Scalability**: High-fidelity simulator (Isaac Sim) 计算成本高, 限制 scalability; learning-based world model scalable 但 physical accuracy 差。EmbodiedDreamer 的 PhysAligner + VisAligner 是 promising 的 hybrid 方向。

4. **Sparse reward vs Dense supervision**: Task success (sparse) 是 ground truth objective, 但 RL 训练困难; dense reward (PBRS, model-generated) 提升 sample efficiency 但可能 mis-specify。这个 tension 类似 LLM 中 sparse task reward vs dense process reward 的 debate。

5. **Autonomy vs Safety**: Fully autonomous RL training 高效但 unsafe; human-in-the-loop safe 但 labor-intensive。Genie Centurion 的 multi-robot fleet + VLM detection + selective human intervention 是 promising 的中间方案。

这些 tension 没有单一 silver bullet, 需要根据具体 deployment scenario 做 tradeoff。这也是为什么这个 survey 的 taxonomy 设计成 multi-dimensional (architecture + training paradigm + deployment + benchmarking) — 不同 dimension 的组合对应不同 tradeoff regime。

---

## X. Key References

- [OpenVLA](https://openvla.github.io/) - Open-source VLA on Llama 2
- [π0 / π0.5](https://www.physicalintelligence.company/blog/pi0) - Flow-matching VLA
- [RT-2](https://robotics-transformer2.github.io/) - Web knowledge transfer
- [LIBERO Benchmark](https://lifelong-robot-learning.github.io/libero/)
- [HIL-SERL](https://hil-serl.github.io/) - Human-in-the-loop RL
- [Eureka](https://eureka-research.github.io/) - LLM reward design
- [Dreamer](https://danijar.com/dreamer/) - World model RL
- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [PPO](https://arxiv.org/abs/1707.06347)
- [SAC](https://arxiv.org/abs/1801.01290)
- [iVideoGPT](https://thuml.github.io/iVideoGPT/)
- [LeRobot](https://github.com/huggingface/lerobot)
- [SERL](https://serl.readthedocs.io/)
- [RoboTwin](https://robotwin-benchmark.github.io/)
- [SIMPLER](https://simpler-env.github.io/)
- [SafeVLA](https://arxiv.org/abs/2503.03480)
- [VLA-RL](https://arxiv.org/abs/2505.18719)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [Hume](https://arxiv.org/abs/2505.21432)
- [VLA-Reasoner](https://arxiv.org/abs/2509.22643)
- [FurnitureBench](https://furniturebench.github.io/)
- [CurricuLLM](https://arxiv.org/abs/2406.12207)
- [TRANSIC](https://transic-robot.github.io/)
- [Recovery RL](https://arxiv.org/abs/2010.11430)
- [GVL](https://arxiv.org/abs/2402.10236)
- [VIPER](https://arxiv.org/abs/2310.07248)
- [RoboCLIP](https://arxiv.org/abs/2310.07888)
- [RLDG](https://arxiv.org/abs/2412.09858)
- [GRAPE](https://arxiv.org/abs/2411.19309)
- [DAgger/CR-DAgger](https://arxiv.org/abs/2506.16685)
- [EmbodiedDreamer](https://arxiv.org/abs/2507.05198)
- [World-Env](https://arxiv.org/abs/2509.24948)
- [VLA-RFT](https://arxiv.org/abs/2510.00406)
- [π*0.6](https://arxiv.org/abs/2511.14759)
- [NORA-1.5](https://arxiv.org/abs/2511.14659)
- [ConRFT](https://arxiv.org/abs/2502.05450)
- [ReinboT](https://arxiv.org/abs/2505.07395)
- [VLAC](https://arxiv.org/abs/2509.15937)
- [PLD](https://arxiv.org/abs/2511.00091)
- [SRPO](https://arxiv.org/abs/2511.15605)
- [DeepThinkVLA](https://arxiv.org/abs/2511.15669)
- [RobustVLA](https://arxiv.org/abs/2511.01331)

---

这篇 survey 的核心 contribution 在于提供了一个 unified framework 理解 RL-VLA 的全 lifecycle: 从 pretraining → architecture design → training paradigm → real-world deployment → benchmarking。对 practitioner 来说, Table I (representative RL-VLA works 总结) 和 Table II (manipulation benchmarks) 是最有价值的 quick reference。对 researcher 来说, Section VII 的 open challenges 指明了最有前景的 future direction, 特别是 model-based RL for VLA, long-horizon tasks, 和 safe real-robot training。

希望这个详细讲解帮到你 build intuition, Andrej! 如果想 deep dive 到某个具体方向 (比如 flow-matching RL 的 mathematical details, 或者 sim-to-real 的 specific techniques), 我可以进一步展开。
