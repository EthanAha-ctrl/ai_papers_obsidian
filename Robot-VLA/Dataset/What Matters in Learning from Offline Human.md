---
source_pdf: What Matters in Learning from Offline Human.pdf
paper_sha256: 18283ff9cbb9645de4ccf32ca2de472f452e89761287e7ed578e3a1066bea5f1
processed_at: '2026-08-13T04:12:01-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej，我换个画风，就像咱在咖啡厅聊这篇paper一样。

---

## 这paper到底在干嘛

一句话总结：**一群Stanford的人想搞清楚，从human teleoperation data里学robot policy，到底什么真正重要**。

他们没提新算法，就是老老实实做了个大规模empirical study。选了8个task（5个sim，3个real），搞了3种dataset（machine-generated、单个高手、6个不同水平的人混合），然后拿6个算法跑了一遍，看看到底什么matter。

这个动机其实挺简单的。你看NLP有GPT，CV有ImageNet，robot manipulation一直没吃到data-driven的红利。大家心里都有个疑问：**我们到底能不能像train GPT那样，拿一堆human demo就train出一个能用的robot policy？**

这个问题没人系统回答过。D4RL那种benchmark用的data是RL agent自己生成的，跟human data完全是两回事。所以这群人就说，ok，那我们自己collect human data，自己做benchmark，自己跑实验，看看到底怎么回事。

---

## 数据是怎么搞的

这块设计得挺巧的。

**3种data source，故意制造对比：**

1. **MG (Machine-Generated)**：用SAC训agent，训的过程中定期存checkpoint，每个checkpoint采300条trajectory。这样data是good policy和bad policy的混合，模拟D4RL那种经典分布。

2. **PH (Proficient Human)**：找了个老手，一个人collect 200条demo。

3. **MH (Multi-Human)**：找6个人，2个高手、2个中手、2个菜鸟，每人50条demo，总共300条。

MH这个设计很妙。你看Table 4的trajectory length——Lift这个简单task，高手平均72步完成，菜鸟要145步。**Trajectory length就是data quality的proxy**，菜鸟磨磨唧唧嘛。这就天然给你制造了multimodality和suboptimal的挑战。

还有一个诊断数据集叫Can-Paired特别有意思。让一个人对每个初始状态collect两条demo：一条成功把can放进bin，一条故意把can扔到workspace外面。200条demo，100好100坏。这个设计就是专门用来测试batch RL算法能不能区分good action和bad action的。

---

## 算法都跑了哪些

6个算法，分两类：

**Imitation Learning派：**
- **BC**：最naive的，state进网络，action出网络，MSE loss，完事
- **BC-RNN**：BC + LSTM，给policy加了历史记忆
- **HBC**：分层BC，高层predict subgoal，低层执行

**Batch RL派：**
- **BCQ**：约束action只在dataset support里，防止OOD overestimation
- **CQL**：给Q function加conservative regularizer，惩罚OOD的Q值
- **IRIS**：HBC + BCQ value function，用value选subgoal

---

## 最核心的发现，一个一个说

### 发现1：RNN是human data的银弹

这是整个paper最robust的结论。看Table 1，BC-RNN基本碾压BC：

- Transport（双臂传锤子）：BC 17.3 → BC-RNN 71.3，**差54个点**
- Square（套螺母）：BC 52.7 → BC-RNN 78.0，差25个点
- 即使是简单的Can，MH data上也有明显gap

**为什么RNN这么work？** 我觉得两个原因：

第一，**人不是Markovian的**。人teleop的时候不会只看当前帧就决策，他脑子里记着过去几秒发生了什么，自己刚做了什么动作。RNN把这个history显式建模了。RL agent是Markovian的，但人不是。

第二，**history能帮助disambiguate multimodality**。当前state可能对应多种valid action（比如6个人用6种策略），但如果看history，"这个人是用策略A的"，那当前action就比较好预测了。RNN实际上在做一种implicit的context-conditioning。

Horizon越长的task，RNN优势越大，这完全符合上面的intuition——Transport要双臂协调，当前action取决于过去几秒的协调状态，光看当前帧根本推断不出来。

### 发现2：Batch RL在human data上崩盘

这是最striking的发现。BCQ在MG data上表现很好（Lift 91.3, Can 75.3），但在human data上就不行了：

- MH data上：Square 14, Transport 2.6
- CQL更惨：MH data上Square 0.7, Transport 0

而BC-RNN在同样data上：Square 78, Transport 65。

**这说明了什么？** D4RL那些benchmark用的RL-generated data，跟real human data本质上不一样。RL-generated data虽然也有noise，但那个noise是"well-behaved"的，是exploration过程中自然产生的。Human data的noise是乱的，有手抖的、有犹豫的、有试错的，batch RL的Q-function在这种noise上根本学不出有用的signal。

Can-Paired那个诊断实验最能说明问题。那个task简单到不能再简单——区分"把can放进去"和"把can扔外面"。BC-RNN能到70%，BCQ只有44.7%，CQL只有6%。**Batch RL连这种最基本的good/bad区分都做不好**，这说明问题不在算法细节，在于整个paradigm对human data不适应。

我的直觉是：human failure跟"bad action"不是一回事。人抓不住can可能是手抖了一下，这个"bad action"在Q-function看来跟good action几乎没区别，但binary reward给了一个0。Q-function在这种noisy reward下根本学不出smooth的value landscape。

### 发现3：Observation space比算法更重要

这个发现对practitioner特别有用。看Table 25：

**给policy加额外的proprioception信息，performance反而暴跌：**
- Square（low-dim）：加end-effector velocity，从84.0掉到42.7，**跌49%**
- 加joint position，掉到39.3，跌53%

但同样的ablation在image policy上，只跌21-29%。

**为什么？** Low-dim policy直接把所有input feature喂进MLP，网络会overfit那些irrelevant signal。Image policy有ResNet做encoder，ResNet的inductive bias天然能忽略irrelevant visual feature，所以更robust。

这其实是个很general的insight：**information hiding很重要**。不要把所有能拿到的信息都塞给policy，只给task-relevant的。Lee et al那个四足机器人paper也讲过这个道理。

**另外两个image policy的必备component：**
- **Pixel shift randomization**（其实就是random crop）：Square掉47%，Transport掉35%
- **Wrist camera**：Transport掉43%（grasp alignment全靠它）

这俩在real world ablation上同样成立，Can（Real）去掉randomization从73.3掉到26.7。

### 发现4：Hyperparameter极其敏感

Table 26是给工程师看的，几个关键点：

- **Learning rate**：image agent从1e-4换到1e-3，performance暴跌35-63%。Low-dim agent反而还行。Image policy对LR的敏感性远高于low-dim。
- **GMM action head**：去掉GMM用deterministic输出，Transport（MH, ld）从65.3掉到27.3，跌58%。Multimodal action distribution必须用GMM建模。
- **RNN hidden dim**：从400降到100，一致掉performance。History capacity不能省。
- **CNN depth**：用shallow CNN替代ResNet-18，掉25-62%。Visual encoder必须是大模型。

### 发现5：Validation loss跟success rate完全不相关

这个发现让人头疼。你train一个policy，想选最好的checkpoint，正常思路是看validation loss。但这个paper发现：

- Square（PH）：best policy 84.0%，validation loss选出来的 7.3%，final checkpoint 74.0%
- Transport（PH）：best 71.3%，valid loss选 4.0%，final 59.3%

**Validation loss选出来的policy基本是垃圾**。增加validation data量也没用，30% validation set依然选不出好policy。

而且有个更诡异的现象：success rate在上升的同时，validation loss也在上升。也就是说policy在environment里越来越好，但在validation set上的loss越来越大。

我的intuition是：BC在early epoch会overfit training data的noise，validation loss开始上升。但这个"overfit"到noise的policy反而在environment里更robust，因为它学到了average化的action distribution。而validation loss最低的那个checkpoint，可能恰好是underfit的，还没学到task structure。

**这对real world deployment是个大问题**。Simulation里你可以每50个epoch跑50个rollout选最好的，real world里你不可能这么做。Offline policy evaluation是个open problem。

### 发现6：Complex task需要更多data

Table 27很直观：

- Lift（简单）：20% data就能到96.7%
- Can（中等）：20% data到76.7%，50%到97.3%
- Square（难）：20%只有38.7%，50%到67.3%，100%到84.0%
- Transport（更难）：20%只有6.7%，50%到44%，100%到71.3%

**Simple task早就饱和了，complex task还在scaling**。这暗示如果你有1000条、10000条human demo，complex task可能还能继续提升。这跟LLM的scaling law有点神似，只是paper没敢往外推。

### 发现7：Sim的insight直接transfer到real

Section 4.7是money shot。用sim上tuned好的hyperparameter直接跑real robot：

- Lift（Real）：96.7%
- Can（Real）：73.3%
- Tool Hang（Real）：3.3%（这个task太难了，多stage assembly）

而且randomization和wrist camera的ablation在real上同样成立。这说明sim study的结论是有意义的，不是sim artifact。

---

## 我自己的看法

### 这paper的价值在哪

这paper没提新算法，但它的价值在于**systematic地确立了robot manipulation imitation learning的benchmark和baseline**。后来的diffusion policy、decision transformer这些工作，基本都在robomimic上验证。它把"从human data学manipulation policy"这个方向legitimize了。

### Batch RL为什么在human data上崩

我觉得根本原因是**reward signal太弱**。Binary task completion reward在human data上极度noisy。人失败可能是手抖，可能是犹豫，可能是策略本身就不对。Batch RL想用这种reward学Q function，然后做argmax选action，这个链条在noisy reward下断了。

BC直接imitate action，不依赖reward，所以反而robust。这是个paradigm层面的差异，不是调参能解决的。

### 后续工作怎么改进

这paper之后的几个direction：
1. **Diffusion policy**（Cheng Chi, 2023）：用diffusion model替代GMM处理multimodal action，效果更好
2. **Decision Transformer**：用Transformer替代LSTM，return-conditioned generation天然处理suboptimal data
3. **Visual pretraining**（R3M, Voltron）：先self-supervised pretrain visual encoder，再fine-tune policy
4. **Action chunking**：不一步步predict action，而是一次predict一整段action sequence，减少compounding error

### 一个我自己好奇的问题

MH（300条demo）比PH（200条demo）表现差，这个现象很有意思。Paper的解释是multimodality和suboptimal。但我好奇：如果6个人都是高手呢？data quantity和data quality的trade-off到底在哪里？这个paper没回答，因为MH的6个人里有4个是中手和菜鸟。

---

## 一句话总结

**从human data学manipulation policy，history modeling是必须的，batch RL目前还不行，observation engineering比algorithm tuning重要，validation loss不能选policy，complex task需要更多data。**

这paper给后来5年的robot learning工作立了个flag，告诉所有人：human data跟RL-generated data是两回事，别拿D4RL的结论直接套。

---

# What Matters in Learning from Offline Human Demonstrations for Robot Manipulation - 深度讲解

Andrej, 这是一篇非常有意思的empirical study paper, 来自 Ajay Mandlekar 等人 (Stanford + UT Austin), 是 RoboTurk 和 robomimic 项目的延续工作。它本质上回答了一个问题：**在 robot manipulation 上, 用 offline human demonstrations 训练 policy, 到底什么 matter**。

Paper link: https://arise-initiative.github.io/robomimic-web/  
arXiv: https://arxiv.org/abs/2109.12938 (实际版本)

---

## 1. Why this paper matters

Robot manipulation 一直没有享受到像 NLP/CV 那种 large-scale data 的红利。一个 fundamental 的问题是: 我们到底能不能用类似 supervised learning 的方式从大规模 human demonstration 中学到 proficient policy? 这篇 paper 做的就是 systematic 的 empirical study, 不提新 algorithm, 而是 deconstruct 现有方法在不同 axis 上的表现。

它继承了 RL unplugged (https://arxiv.org/abs/2006.13888) 和 D4RL (https://arxiv.org/abs/2004.07219) 的 benchmarking 哲学, 但 **关键区别** 在于: D4RL/RL unplugged 大部分 data 是 RL-trained agent 生成的, 这篇 paper 用的是 **real human teleoperation** 生成的 data。这个区别极其重要, 因为它揭示了 batch RL 算法 (BCQ, CQL) 在 RL-generated data 上表现好, 但在 human data 上崩盘的事实。

---

## 2. Study Design - 5 axes 的 dissect

作者从 5 个 challenge (C1-C5) 出发设计 study:

| Challenge | 含义 | 对应 study section |
|---|---|---|
| C1 | Non-Markovian decision process | §4.1: history-dependent models 的作用 |
| C2 | Variance in human quality | §4.2: suboptimal/multi-human data |
| C3 | Dataset size dependence | §4.6: data scaling law |
| C4 | Train/eval objective mismatch | §4.5: policy selection |
| C5 | Sensitivity to design choices | §4.3, §4.4: observation space 和 hyperparameter |

### 2.1 Tasks (8个, 5 sim + 3 real)

复杂度梯度很清晰, 这是 paper 的精妙之处:

1. **Lift** - 抓 cube (simplest, state dim 10)
2. **Can** - 抓 coke can 放入 bin (state dim 14)
3. **Square** - square nut 套 rod (state dim 14, 需要 precision)
4. **Transport** - 双臂传递 hammer (state dim 41, multi-arm coordination)
5. **Tool Hang** - 组装 frame 并挂 wrench (state dim 44, multi-stage, 高精度)

Task horizon 也递增: PH dataset 平均 trajectory length 从 Lift 的 48 步 增长到 Tool Hang 的 480 步。

### 2.2 Dataset sources (3种)

- **MG (Machine-Generated)**: SAC (https://arxiv.org/abs/1801.01290) 训练过程中收集的 checkpoints (mixture of good + suboptimal policies, 共 300 rollouts × 150 steps)。这模拟 D4RL 的经典 data 分布。
- **PH (Proficient Human)**: 单个 experienced operator 收集 200 demos
- **MH (Multi-Human)**: 6 operators (2 better + 2 okay + 2 worse), 每个 50 demos, 总共 300 demos

MH 设计的精妙之处: 通过 Table 4 我们看到 trajectory length 是 data quality 的 proxy。Lift task: Better 72 ± 24, Okay 94 ± 30, Worse 145 ± 40 步 - 这正是 Multimodality 和 suboptimal 的来源。

---

## 3. Algorithm 技术详解

### 3.1 BC (Behavioral Cloning)

最简单的 baseline:

$$\arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} \| \pi_\theta(s) - a \|^2$$

- $\theta$: policy network 参数
- $s$: state observation
- $a$: demonstrator action  
- $\mathcal{D}$: offline dataset
- $\| \cdot \|^2$: MSE loss

BC 的致命问题在 multi-human data 上, 因为不同 human 用不同 strategy, conditional distribution $p(a|s)$ 是 **multimodal**, MSE loss 把它平均成 unimodal, 得到 mean action, 大概率不是 valid action。

### 3.2 BC-RNN (BC with LSTM)

加入 temporal context:

$$a_t, h_{t+1} = \pi_\theta(s_t, h_t)$$

- $h_t \in \mathbb{R}^d$: LSTM hidden state, $d=400$ (low-dim) or $1000$ (image)
- 训练时 sample length-$T$ sequences, $T=10$
- test 时每 $T$ 步 refresh hidden state (重要的 detail)

为什么 BC-RNN 这么 work? 我理解有两个原因:
1. **Non-Markovian modeling**: Human 在 teleop 时 不会只看 current frame 决策, 他们会基于最近几秒的 visual context + 自己的 action history 决策。RNN 把这层信息显式建模。
2. **Multimodality smoothing**: 不同 human 的 multimodal action 分布通过 time aggregation 被 partially averaged。RNN 输出的是 conditioned on history 的 action, 自然更加 unimodal 在 local context 下。

### 3.3 HBC (Hierarchical BC)

参考 https://arxiv.org/abs/2003.06085 (Mandlekar et al, Learning to Generalize across Long-Horizon Tasks)

两层 hierarchy:

- **Low-level** $\bar{\pi}^L_\theta(s, s_g)$: conditioned on subgoal $s_g$ (即 $T$ 步后的 observation), 输出 action sequence 实现 subgoal。本质是 BC-RNN + subgoal conditioning。
- **High-level** $\pi^H_\theta(s)$: 从 $s_t$ 预测 $s_{t+T}$, 用 cVAE (https://arxiv.org/abs/1312.6114) 学 conditional distribution $\pi^H(s_{t+T} | s_t)$

cVAE objective (大致):
$$\mathcal{L}_{\text{HBC}} = \mathbb{E}_{q_\phi(z|s_t, s_{t+T})} [\log p_\theta(s_{t+T} | s_t, z)] - D_{KL}(q_\phi(z|s_t, s_{t+T}) \| p(z|s_t))$$

- $z$: latent variable
- $q_\phi$: encoder (inference network)
- $p_\theta$: decoder (generative network)  
- $p(z|s_t)$: learned prior (这里用 GMM prior 而非标准 normal, 这是 MH data 上的关键 tuning)

HBC 的核心 insight: **Temporal abstraction** - 高层决策频率低 (每 $T$ 步), 低层执行。这样 multimodality 的处理只需要在高层做, 低层保持 unimodal 但 conditioned on subgoal。

### 3.4 BCQ (Batch Constrained Q-Learning)

Paper: https://arxiv.org/abs/1812.02900 (Fujimoto et al, ICML 2019)

Core idea: 标准 offline Q-learning 会 overestimate OOD (out-of-distribution) action 的 Q-value, BCQ 通过 **constrain action sampling to dataset support** 解决。

三个 components:
- $Q_\psi(s,a)$: Q-network (critic)
- $p_\omega(a|s)$: generative action model (cVAE, 用来 sample 类似 dataset 中出现过的 actions)
- $\pi_\theta(s,a)$: perturbation actor (optional, 微调 sampled actions)

Target construction:
$$A = \{a_i + \pi_\theta(s, a_i) \mid a_i \sim p_\omega(\cdot | s)\}_{i=1}^N$$
$$Q_{\text{target}} = r + \gamma \max_{a_i \in A} Q_{\psi'}(s', a_i)$$

- $A$: 候选 action set (从 VAE 采样 + actor 微调)
- $N$: sample 数量 (paper 用 10 for train, 100 for test)
- $\gamma$: discount factor
- $\psi'$: target network 参数

Q-network loss:
$$\mathcal{L}_Q = (Q_\psi(s,a) - Q_{\text{target}})^2$$

Perturbation actor loss (类似 DDPG, https://arxiv.org/abs/1509.02971):
$$\mathcal{L}_\pi = -Q_\psi(s, a + \pi_\theta(s,a)) \mid a \sim p_\omega(\cdot|s)$$

**关键 ablation (Table 18)**: perturbation actor 在 human data 上会导致灾难性 performance drop。Can (PH): 88.7% → 8.0%。这跟 MG data 上完全相反 (MG data 上 actor 帮助提升)。这个现象我没完全想明白, 可能是 human data 已经 noise 足够, actor 加 perturbation 反而 destroy action distribution。这也呼应 paper 强调的 "MG data tuning 不能直接 transfer 到 human data"。

### 3.5 CQL (Conservative Q-Learning)

Paper: https://arxiv.org/abs/2006.04779 (Kumar et al, NeurIPS 2020)

Core idea: 直接在 Q-function 上加 regularizer, 让 OOD action 的 Q-value 被 underestimate, data-supported action 的 Q-value 被 preserve。

$$Q^{k+1} \leftarrow \arg\min_Q \underbrace{\frac{1}{2}(Q(s,a) - Q_{\text{target}})^2}_{\text{standard TD loss}} + \alpha \underbrace{\left(\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a|s)}[Q(s,a)] - \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(a|s)}[Q(s,a)]\right)}_{\text{conservative regularizer}}$$

- $\alpha$: conservative weight (这里用 Lagrangian auto-tune, threshold $\tau=5$)
- $\mu(a|s)$: behavior policy (data distribution)
- $\pi_\theta(a|s)$: learned policy
- 第一项使 Q 估计接近 TD target  
- 第二项 **push down** Q on policy distribution, **push up** Q on data distribution

CQL 在 paper 中表现极差 (Square (PH): 0% Transport (PH): 0%)。这是 paper 的一个重要 finding - **CQL 的 conservative regularization 在 human data 上 over-regularize**, 因为 human data 自身就有大量 noise, 让 CQL 不能区分 noise vs. real signal。

### 3.6 IRIS (Implicit Reinforcement without Interaction at Scale)

Paper: https://arxiv.org/abs/1911.05321 (Mandlekar et al, ICRA 2020)

本质是 **HBC + BCQ value function**:
- Low-level: 与 HBC 相同, BC-RNN + subgoal conditioning
- High-level: cVAE sampler + value function $V(s)$ (用 BCQ 训练, 选最高 value 的 subgoal)

Selection:
$$s_g^* = \arg\max_{s_g \in \{s_g^{(i)} \sim \text{cVAE}\}_{i=1}^N} V(s_g)$$

- $s_g^{(i)}$: 从 cVAE 采样的第 $i$ 个 candidate subgoal
- $V(s_g)$: value function 评估 subgoal 的 goodness

IRIS 的核心 idea: **将 multimodality 处理 与 value learning 解耦** - multimodality 在 cVAE 高层 (低频) 处理, value learning 在 abstracted state space 上做, 避开 low-level action space 的 OOD 问题。

---

## 4. Key Experimental Findings 深度解析

### 4.1 BC-RNN dominates - 为什么 history 这么重要

Table 1 是核心证据。看几个对比:

| Task | BC | BC-RNN | Gap |
|---|---|---|---|
| Transport (PH) | 17.3 | 71.3 | +54 |
| Transport (MH) | 11.3 | 65.3 | +54 |
| Tool Hang (PH) | 29.3 | 19.3 (这里 BC 反而高, anomaly) | -10 |
| Square (MH) | 52.7 | 78.0 | +25 |

Horizon 越长, BC-RNN 的优势越大。我的 intuition: **long-horizon task 中, 当前 action 几乎不能从 current observation 推断**。比如 Transport task, 当前要决定把 hammer 递给哪个 arm, 这取决于过去几秒的协调状态, 当前 frame 看不出。

Paper 中还有一个有趣的 finding (§4.1): MH dataset 表现普遍比 PH 差, **即使 MH 有更多 data** (300 vs 200)。这说明 data quality 的影响 > data quantity, 至少在 medium scale 上。

### 4.2 Batch RL 在 human data 上崩盘

这是 paper 最 striking 的发现。看 Table 1:

- BCQ 在 MG data: Lift 91.3, Can 75.3 (强)
- BCQ 在 PH data: Lift 100, Can 88.7, Square 50, Transport 7.3 (中等)
- BCQ 在 MH data: Lift 100, Can 62.7, Square 14, Transport 2.6 (差)

CQL 全面更差。MH data 上 Square (MH): 0.7, Transport (MH): 0.

为什么 batch RL 在 human data 上 fail? Paper 中作者做了 diagnostic 实验 (Can-Paired, Table 2 最后一行):

- 单个 operator 对每个 initial state 收集 **2 个 demo**: 一个成功, 一个 toss can outside bin (失败)
- 共 200 demos, 100 个 success, 100 个 failure
- 这个 task 设计很简单: 只需区分 "pick and place" vs "pick and throw outside"

结果:
- BC-RNN: 70.0 ± 4.3
- BCQ: 44.7 ± 1.9
- CQL: 6.0 ± 1.6

我的 hypothesis: Batch RL 的 Q-function 在 sparse reward + human noise 下 extremely hard to train。Human failure 不一定是 "bad action", 可能是抖了一下抓不住, 这种 noise 无法用 binary reward 区分。BCQ 的 VAE action sampler 可能学到一个 "blurry" 版本的 human action distribution, Q-function 在这种 noisy action space 上做 argmax 完全没意义。

### 4.3 Observation space 的敏感性

Fig 2a 和 Table 25 是这个 study 的精华之一:

| Variant | Square (ld) | Square (im) |
|---|---|---|
| Default | 84.0 | 82.0 |
| + EEF Vel | 42.7 (-49%) | 64.7 (-21%) |
| + Joint | 39.3 (-53%) | 58.0 (-29%) |
| - Rand | - | 43.3 (-47%) |
| - Wrist | - | 74.7 (-9%) |

Intuition: **Extra information 反而 hurts** when 它不 task-relevant。加上 EEF velocity 和 joint positions 在 low-dim 上 hurt 49-88%, 但 image 上只 hurt 2-29%。我的解释: low-dim policy 直接 feed 这 24-42 维的 proprioception, 网络会 overfit 这些 signal (即使无用)。Image policy 的 image encoder 是预训练的 ResNet, 有 inductive bias 忽略 irrelevant features, 所以更 robust。

这其实是 **information hiding** 的经典观点, 参考了 Lee et al, Science Robotics 2020 (https://arxiv.org/abs/1909.07564) - 四足机器人 paper 中也强调了 feed minimally sufficient observation 的重要性。

### 4.4 Pixel shift randomization + wrist camera 是 game changer

Image-based policy 必须有的两个 component:
- **Pixel shift randomization** (random crop from 84x84 到 76x76, 或 120x120 到 108x108): Square drop 47%, Transport drop 35%
- **Wrist camera**: Square drop 9%, Transport drop 43%

Random crop 的作用其实是 spatial invariance 的 implicit learning, 类似 CNN 中的 translation invariance, 但更 explicit。Wrist camera 对 grasp precision 至关重要 (gripper alignment), Transport 任务的 43% drop 说明 wrist cam 在多 stage task 中对每个 sub-task 都有用。

这也呼应了 Pinto et al 的 "Visual Imitation Made Easy" (https://arxiv.org/abs/2008.06012) - data augmentation 对 imitation learning 的 critical role。

### 4.5 Hyperparameter sensitivity

Table 26 是工程师的宝典:

| Hyperparameter change | Square (MH, ld) | Transport (MH, ld) | Square (MH, im) | Transport (MH, im) |
|---|---|---|---|---|
| Default | 78.0 | 65.3 | 76.7 | 42.0 |
| larger LR (1e-3) | 76.7 | 49.3 | 28.7 | 23.3 |
| no GMM | 58.0 | 27.3 | 61.3 | 41.0 |
| larger MLP | 73.3 | 46.0 | - | - |
| Shallow Conv | - | - | 48.0 | 16.0 |
| smaller RNN dim | 58.7 | 27.3 | 58.0 | 34.0 |

Practical takeaways:
- **GMM policy head 重要**: GMM 能建模 multimodal action distribution, no GMM 在 Transport (MH, ld) 上 drop 58%
- **Large RNN dim 重要**: 100 vs 400 (ld) 或 400 vs 1000 (im) 一致 hurt
- **Large ConvNet 重要**: Shallow CNN (Finn et al 的 spatial autoencoder, https://arxiv.org/abs/1509.06113) drop 25-62%
- **Image agent 对 LR 极敏感**: 1e-3 vs 1e-4 在 image 上 drop 35-63%

GMM 的 eval trick (Appendix J): test 时不用 learned std, 而用 1e-4 的小 std, 等价于 sample mode mean。这跟 RL 中 Gaussian policy 的 mean action eval 一样。

### 4.6 Policy selection 的难点

Fig 4a 和 Appendix G 揭示了一个微妙问题: validation loss 跟 task success rate **不相关**, 甚至 **negatively correlated**。

具体 (Table 29):
- Square (PH, ld): best 84.0, valid loss selection 7.3, final checkpoint 74.0
- Transport (PH, ld): best 71.3, valid loss selection 4.0, final checkpoint 59.3

增加 validation data 也无济于事 (30% validation): Square best 80.7, valid-loss-selected 2.7; Transport best 64.0, valid-loss-selected 0.7。

作者的解释是 training objective (NLL/MSE) 跟 evaluation objective (task success rate) 的 mismatch。我自己的 intuition: BC 在 early epoch 已经 overfit training data 的 noise, validation loss 开始上升, 但 policy 在 environment 上反而因为 underfit "more robust" 的 action distribution 而表现更好。这是 IL 的 generalization 怪相。

这也联系到offline policy evaluation (OPE) 的整个研究领域, 参考 Fu et al "Benchmarks for Deep Off-Policy Evaluation" (https://arxiv.org/abs/2103.16596)。

### 4.7 Dataset scaling - complex tasks 需要更多 data

Table 27 (PH dataset):

| Task | 20% | 50% | 100% |
|---|---|---|---|
| Lift (ld) | 96.7 | 100 | 100 |
| Can (ld) | 76.7 | 97.3 | 100 |
| Square (ld) | 38.7 | 67.3 | 84.0 |
| Transport (ld) | 6.7 | 44.0 | 71.3 |

Simple task (Lift, Can) 用 20% data 就够了。Complex task (Square, Transport) 显示 **strong scaling behavior** - data 增加, performance 明显提升。

这其实暗示了一个更深的 hypothesis: 如果有 **更大规模 human dataset** (比如 10x, 100x), complex tasks 可能能突破当前 BC-RNN 的 performance ceiling。这跟 LLM 的 scaling law 有点类似, 但没在 paper 中探讨。

### 4.8 Real world transfer

Section 4.7 是 paper 的 "money section", 证明 sim insights 能 transfer:

- Lift (Real): 96.7% (BC-RNN, 200 demos)
- Can (Real): 73.3%  
- Tool Hang (Real): 3.3% (极难, 多 stage assembly)

关键 ablation (Can (Real)):
- Default (with Rand + Wrist): 73.3%
- - Rand: 26.7% (-46.6%)  
- - Wrist: 43.3% (-30%)

Sim 上的 -47% (Square) 和 -9% (Square) 的 ablation 结论在 real world 上同样成立。

---

## 5. 对 practitioners 的 key lessons

总结 paper 的 6 个 lessons (L1-L6) 加上我自己的解读:

### L1. History-dependent models 是 human data 的 silver bullet
- BC-RNN > BC 在所有任务上
- HBC 和 IRIS 在 long-horizon task 上额外加分
- 不要用 Markovian BC 处理 human data

### L2. Batch RL 需要 redesign for human data
- BCQ/CQL 在 MG data 上 work, 在 human data 上 fail
- IRIS (history + value) 是 promising direction, 但还不够
- 未来 batch RL 应该在 human data 上 benchmark, 不只在 RL-generated data

### L3. Offline policy selection 是 open problem
- Validation loss 不能用
- Final checkpoint 也不可靠
- 需要 OPE 或 environment-based selection (但 real world 难做)

### L4. Observation space engineering 至关重要
- 信息越少越好 - 排除 task-irrelevant proprioception
- Pixel shift randomization + wrist camera = 必备
- 低维 observations 反而更脆弱 (less inductive bias)

### L5. Large-scale human dataset 能解锁 complex tasks
- 50 demos 对 Tool Hang 不够, 200 demos 对 Tool Hang (sim) 能达到 67.3%
- Scaling behavior 在 complex task 上明显
- 暗示 data-driven manipulation 的未来

### L6. Sim insights 直接 transfer 到 real
- 不需要 real world hyperparameter tuning
- 这让 sim benchmark 有现实意义

---

## 6. 我自己的一些联想和批评

### 6.1 跟 decision transformer 的联系

Paper 的 reference [50, 51] 提到了 Decision Transformer (https://arxiv.org/abs/2106.01345) 和 Trajectory Transformer, 但没实际比较。我觉得这是个 missed opportunity: DT 用 Transformer 替代 RNN 处理 history, 在 long-horizon 上可能更强, 而且 return-conditioned generation 天然处理 suboptimal data。后续 Robomimic 后续工作确实加了 DT 实验。

### 6.2 Multimodality 的处理

HBC 和 IRIS 用 cVAE 处理 subgoal multimodality, BC-RNN 用 GMM action head 处理 action multimodality。但 GMM 在 high-dim action space 上效果有限 (mode collapse)。Diffusion policy (Cheng Chi et al, 2023, https://arxiv.org/abs/2303.04137) 后来证明 diffusion model 处理 multimodal action 更好, 在 robomimic 上能进一步提升。

### 6.3 PH vs MH 的发现

MH (300 demos) 比 PH (200 demos) 表现差是个反直觉发现。Paper 的解释是 multimodality 和 suboptimal data 增加。但我有个 alternative hypothesis: 6 个 human 中 4 个 "okay + worse" 是 suboptimal, 平均拉低了整个 dataset 的 effective signal-to-noise ratio。如果 data 是 6 个 better humans, 结果可能不同。

### 6.4 Tool Hang 的特殊性

Table 1 中 Tool Hang (PH) 的 anomaly: BC (29.3) > BC-RNN (19.3)。我猜原因是 Tool Hang 的 trajectory 极长 (480 步 avg), RNN 的 sequence length T=10 不够 cover 整个 task structure。HBC 的 subgoal abstraction (每 T 步 predict 一个 future state) 在这个 task 上反而更 work (30.0)。这暗示对 ultra-long-horizon task, subgoal abstraction > dense history。

### 6.5 跟 self-supervised learning 的关系

Paper 完全用 supervised imitation, 没用 self-supervised pretraining (e.g., world model, contrastive learning on observations)。后续工作 (e.g., R3M https://arxiv.org/abs/2203.12601, Voltron https://arxiv.org/abs/2302.12766) 证明 visual pretraining 能进一步提升 robomimic-style task 的 performance。

### 6.6 Action representation

Paper 的 action space 是 7-DoF (3 trans + 3 rot + 1 gripper), 直接用 GMM 输出 continuous action。后续工作表明用 discrete action token (RT-1, https://arxiv.org/abs/2212.06817) 或 diffusion action 更适合高维, multimodal manipulation action。

---

## 7. Conclusion

这篇 paper 是 manipulation imitation learning 领域的 **definitive empirical study**, 给后续 5 年的工作奠定了 benchmark 和 baseline。Codebase (robomimic, https://github.com/ARISE-Initiative/robomimic) 成为大量后续工作的标准测试平台。

我个人最喜欢的 3 个 takeaways:
1. **History dependence 是 human data 的 silver bullet** - 这是 deep RL community 经常忽略的, RL agent 是 Markovian, human 不是
2. **Batch RL 在 human data 上全面崩盘** - 揭示了 D4RL benchmark 跟 real human data 之间的 gap, 是一个研究方向
3. **Information hiding + augmentation 比 model capacity 重要** - observation engineering 比 algorithm tuning 影响更大, 这是 robot learning 比 NLP/CV 更难的地方 (reward signal 弱, 所以 input signal engineering 关键)

如果你想 build intuition, 我强烈建议去 https://robomimic-web.github.io/ 看 videos, 并跑一下他们的 baseline。Transport task 的 multi-arm coordination 和 Tool Hang 的 multi-stage assembly 在 video 里看会让你对 task difficulty 有更直观感受。

---

## References (主要 cited papers)

- Robomimic project: https://arise-initiative.github.io/robomimic-web/
- RoboTurk: https://arxiv.org/abs/1911.04052
- IRIS: https://arxiv.org/abs/1911.05321
- HBC: https://arxiv.org/abs/2003.06085
- BCQ: https://arxiv.org/abs/1812.02900
- CQL: https://arxiv.org/abs/2006.04779
- SAC: https://arxiv.org/abs/1801.01290
- D4RL: https://arxiv.org/abs/2004.07219
- RL Unplugged: https://arxiv.org/abs/2006.13888
- robosuite: https://arxiv.org/abs/2009.12293
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RT-1: https://arxiv.org/abs/2212.06817
- DDPG: https://arxiv.org/abs/1509.02971
- VAE (Kingma): https://arxiv.org/abs/1312.6114
- Visual Imitation Made Easy: https://arxiv.org/abs/2008.06012
- Lee et al quadruped (information hiding): https://arxiv.org/abs/1909.07564
- OPE benchmarks: https://arxiv.org/abs/2103.16596
- Finn et al spatial autoencoder: https://arxiv.org/abs/1509.06113
- ResNet: https://arxiv.org/abs/1512.03385

如果你对某个具体实验或 algorithm 想深入聊 (比如 BCQ 的 VAE sampler 具体怎么 work, 或 HBC 的 cVAE 细节, 或为什么 image agent 对 LR 这么敏感), 我可以继续展开。
