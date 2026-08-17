---
source_pdf: FLaRe.pdf
paper_sha256: 9fe4ebf83f4b9246f12898cd643edb07d4d93de54f7959388a5a27cf0315b6a6
processed_at: '2026-08-04T08:34:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FLaRe 用人话讲

## 这篇paper想解决什么问题？

现在robotics圈子里流行一个套路：收集一大堆人类演示数据，用behavior cloning训练一个大的transformer模型，希望能得到一个什么都会的generalist robot policy（比如RT-1, RT-2, SPOC这些）。

**但实际部署效果很烂。**

为啥烂？因为BC本质上就是在模仿人类的动作，它学的目标函数是"我做的动作要跟人类演示的动作一样"，这个目标跟"把任务完成"是两码事。更糟的是，一旦机器人稍微走偏一点，进入一个训练时没见过的状态，它就不知道怎么办了——因为训练数据里全是人类走的"完美路径"，没有"如何从错误中恢复"的数据。一个小错累积成大错，最后彻底崩掉。

**那RL不是正好能解决这个问题吗？** RL直接optimize"把任务完成"这个真正的目标，而且通过trial-and-error，机器人能学会从各种奇怪状态中恢复。

听起来很美好。但问题是：直接拿RL去fine-tune一个大的BC预训练模型，**会崩溃**。gradient update会把这个模型好不容易学到的behavior prior全部毁掉，training curve直接掉到0。

FLaRe的核心贡献就是：**找到了一套"让RL fine-tune大规模BC模型不崩"的配方**，并且在真实机器人上跑通了。

---

## FLaRe的三个核心idea

### Idea 1: 从foundation model开始fine-tune，不要from scratch

这个很直觉。如果你从一个random initialization开始用RL训练，search space太大，机器人连站都站不稳，根本explore不到任何有意义的reward signal。

但从一个已经pretrain好的multi-task model开始，机器人已经会做一些合理的动作了，RL只需要在这个基础上"修正方向"——让它从"模仿人类"转向"真正完成任务"。

而且因为base model是多任务的，它学到的features很general，你甚至可以fine-tune到训练时从没见过的task和机器人身体上。

### Idea 2: 大量用simulation做fine-tune

RL需要海量sample，在真实机器人上根本跑不起。所以fine-tune全在simulation里做（AI2THOR，15万个程序生成的房子）。

但simulation跟reality有gap啊。FLaRe用两个trick解决sim-to-real：

- **Domain randomization**：给图像加各种augmentation（颜色变换、随机裁剪、posterization），逼模型学到不依赖sim特定视觉特征的representation。
- **DINOv2 frozen backbone**：用一个self-supervised pretrain的vision transformer提取视觉特征，整个RL过程中freeze住。DINOv2的feature天生就sim-real通用。这样RL的gradient就不会把visual feature搞坏。

还有个工程trick叫**KV-cache**：transformer每一步都要看之前所有历史，正常是 $O(n^2)$ 复杂度。对于600步的episode，计算量爆炸。KV-cache把之前的key和value存起来复用，降到 $O(n)$。这个让大规模RL fine-tuning在计算上affordable。

### Idea 3: 四个stabilization trick（这是paper最technical的核心）

这四个trick，**任何一个去掉都会崩**。ablation study里，去掉任何一个，Fetch task的success rate瞬间掉到0。

**Trick 1: 用on-policy的PPO，不用off-policy的SAC**

Off-policy RL看起来很美——可以复用旧数据，sample efficient。但它有个臭名昭著的"deadly triad"问题：function approximation + bootstrapping + off-policy三者同时出现时，value function很容易diverge。

既然fine-tune在simulation里跑，sample efficiency不是瓶颈，稳定性才是关键。PPO是on-policy的，虽然每批数据只用一次就扔，但gradient方向靠谱，不会因为stale data把policy带歪。

**Trick 2: Learning rate要小一个数量级**

从scratch训练RL，SoTA用的LR是 $2 \times 10^{-4}$。但fine-tune BC预训练模型，得降一个数量级，否则gradient step太大，直接跳出预训练模型所在的good loss basin，behavior prior全毁。

注意：所有task用同一个LR，不做per-task tuning。

**Trick 3: 关掉entropy bonus**

PPO默认有个entropy bonus，鼓励policy保持一定randomness，促进exploration。这个在from scratch训练时很有用。

但fine-tune时，预训练policy已经在meaningful action上有合理confidence了。你强制增加entropy，等于把probability mass从"有意义的动作"推向"无意义的动作"，直接distort policy gradient，导致unlearning。所以必须关掉。

**Trick 4: Actor和Critic不要共享backbone**

标准practice是actor和critic共享visual feature extractor，联合学习useful features。

但fine-tune时，critic的gradient会backprop到shared backbone，修改那些对action prediction至关重要的预训练features。Actor的action prediction就跟着烂掉了。

FLaRe的做法：actor和critic是两个完全独立的网络，都从预训练SPOC初始化，但critic的policy head换成random init的value head。Critic的gradient完全不会碰到actor的backbone。

---

## 实验结果有多impressive？

### Simulation（CHORES-S benchmark，4个task）

FLaRe用sparse reward（只看任务是否完成），baselines里有dense reward（人工设计的每步reward）+ privileged info（比如GPS级别的物体位置）。

- FLaRe average 79.5% success rate
- 比previous SoTA高+23.6%

特别值得一提的是Fetch task（先导航找物体再抓取，long-horizon composition）：所有baseline基本都0%或个位数，FLaRe做到66.9%。这是唯一一个能在这个task上work的方法。

而且FLaRe训练用的step数是baseline的1/2到1/15。Poliformer在ObjectNav上跑了300M steps，FLaRe只跑20M。

### Novel tasks（训练时从没见过的能力）

测试了3个需要新reasoning能力的task：
- ObjNavRelAttr："找最大的苹果"——需要枚举所有苹果比较属性
- RoomNav："去厨房"——导航到房间类型而非物体
- ObjNavAfford："找能坐的东西"——理解affordance

FLaRe在3个task全部SoTA。这意味着只需指定新的success criteria和language instruction，就能on-the-fly定义并fine-tune到全新task。**指向continual adaptation的方向。**

### Real robot

Stretch RE-1机器人在一个训练时从没见过的6-room公寓里直接部署，zero real-world fine-tuning。

- ObjectNav: 94.4% (SPOC 50.0%, Poliformer 83.3%)
- Fetch: 66.7% (SPOC 33.3%)
- PickUp: 86.7% (SPOC 66.7%)
- RoomVisit: 75.0% (SPOC 50.0%)

Real-world average 80.7%，比previous best高+30.7%。

### Cross-embodiment

SPOC只在Stretch上训练（有机械臂）。FLaRe把它adapt到Locobot（没有机械臂，但camera可旋转且视野更窄、安装更低）。

做法很暴力但有效：mask掉无效的arm action，repurpose两个无效action来控制camera旋转。

30M steps fine-tune后，ObjectNav 72.0% success rate，碾压Poliformer from scratch的44.0%。

### Cross-behavior

想让机器人更高效（加step penalty）或更少碰撞（加collision penalty）。只需在reward里加一项，6小时fine-tune就adapt到新behavior，success rate几乎不掉。

---

## 直觉总结

FLaRe的insight可以这样理解：

**BC预训练的模型，已经站在一个"还算不错的山丘"上。RL的作用是把它从"模仿人类"的方向，拽到"真正完成任务"的方向。但这个拽的力道必须非常温和、非常精准，否则模型会从山丘上滚下去，摔成random policy。**

四个stabilization trick就是在控制这个"拽"的力道：
- On-policy保证gradient来自当前policy的真实experience，方向可靠
- Small LR保证每步update幅度小
- Disable entropy bonus不人为扰动已有的action distribution
- Separate actor-critic不让critic的gradient污染actor的features

这套配方在大scale transformer + real robot上首次跑通，证明了"BC pretrain + RL fine-tune"这条路线的可行性。跟LLM的RLHF有异曲同工之妙——都是SFT/BC先建立behavior prior，再用RL align到真正目标，而且都需要careful stabilization防止catastrophic forgetting。

---

# FLaRe 深度技术解读

## 一、Core Problem & Motivation

### 1.1 BC paradigm的intrinsic limitation

当前robotics foundation model的主流training recipe (RT-1, RT-2, RT-X, SPOC, Octo, OpenVLA) 都follow一个recipe：用large-scale multi-task behavior cloning训练high-capacity transformer。BC的目标函数是maximum likelihood:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a^*) \sim \mathcal{D}_{expert}} \left[ \log \pi_\theta(a^* | s) \right]$$

其中 $a^*$ 是expert demonstration中的action，$\mathcal{D}_{expert}$ 是demonstration dataset。

**问题1: Compounding error + state distribution drift** (Ross, Gordon & Bagnell, 2011 [44], DAGGER paper)。BC只在expert trajectory附近训练，部署时small action prediction error → state drift出training distribution → 后续prediction更不可靠 → error compounding。

**问题2: Imitation ≠ task completion**。BC的surrogate objective（imitation）与真正的task objective（success）之间存在gap。Expert trajectory本身可能不是optimal的，即使完美imicate也未必成功。

### 1.2 RL fine-tuning的promise和challenge

RL直接optimize expected return:

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, l) \right]$$

这恰好是task completion的真正目标。但是RL fine-tuning BC预训练的large-scale model面临**destructive gradient update**问题：abrupt从BC transition到RL会导致oscillation甚至collapse。这一点在Section V-E的ablation study里得到极端验证——任何stabilization technique的缺失都让success rate掉到0。

FLaRe的核心thesis: **通过carefully designed stabilization，可以用sparse reward RL fine-tune大规模BC model，aligning it toward true task completion。**

参考链接：
- DAGGER (compounding error theory): https://arxiv.org/abs/1011.0686
- RT-1: https://arxiv.org/abs/2212.06817
- SPOC (base model): https://openaccess.thecvf.com/content/CVPR2024/papers/Ehsani_SPOC_Imitating_Shortest_Path_Navigation_CVPR_2024_paper.pdf
- Poliformer (SoTA RL from scratch baseline): https://arxiv.org/abs/2406.20083

---

## 二、Method 三大设计选择

### 2.1 Fine-tune from multi-task foundation model

为什么不from scratch？三个reasons:

1. **Robust representations & versatile behavior priors**：multi-task BC pretraining学到比single-task更general的features。
2. **Inductive bias of high-capacity architecture**：transformer架构本身带好的inductive bias，协助generalization。
3. **Multi-task capability → 一次预训练，多次复用**：同一base model可以fine-tune到很多task甚至unseen task/embodiment。

FLaRe选择SPOC作为base model。SPOC是个multi-task transformer，在ProcTHOR houses里用shortest-path expert trajectories训练。

### 2.2 Large-scale fine-tuning in simulation

用AI2THOR simulation，150k ProcTHOR procedurally generated houses + 800k+ annotated 3D objects (Objaverse)。

**Sim-to-real的关键技术**：

a) **Domain randomization**：
- Color augmentation
- Random crops
- Image posterization
- 降低sim-real domain gap

b) **DINOv2 frozen visual backbone**：
DINOv2 (Oquab et al. 2023 [54]) 是self-supervised vision transformer，其dense predictions具有跨sim-real泛化能力。论文freeze DINOv2 weight，仅训练上层policy network。这样visual features不在RL fine-tuning中被corrupt。

参考: https://arxiv.org/abs/2304.07193

c) **KV-cache technique**：
将transformer inference从 $O(n^2)$ 降到 $O(n)$（n是episode length）。在episode内cache早期observations的keys和values。对于long-horizon mobile manipulation task（如CHORES中600-1000步），这个加速至关重要，让大规模RL fine-tuning computationally affordable。

参考: https://arxiv.org/abs/2211.05102

---

## 三、Stabilize RL Fine-tuning — 论文最技术性的核心

这是Section IV-C，4个critical algorithmic choices。

### 3.1 On-policy PPO rather than off-policy SAC

**Off-policy**方法（SAC, DQN系列）能利用historical data，sample efficient，但是被"**deadly triad**"困扰 (Sutton & Barto [8])：

> Function approximation + Bootstrapping + Off-policy training → divergence

 Deadly triad的intuition：当三者同时出现，Bellman operator在function approximator上的fixed point可能不存在或不稳定，导致value function diverge。

**On-policy PPO**：因为是fine-tune in simulation，sample efficiency不是瓶颈。稳定性是关键。

PPO的核心objective (Schulman et al. 2017 [58]):

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \; \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t \right) \right]$$

变量解释：
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ — probability ratio，新policy与old policy在$(s_t, a_t)$处的概率比
- $\hat{A}_t$ — advantage estimate（用GAE计算）
- $\epsilon$ — clipping parameter (论文设为0.1)，限制policy update的幅度
- $\hat{\mathbb{E}}_t$ — 在collected rollout上经验期望

**Advantage via GAE** (Schulman et al. 2015):

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — TD residual
- $\gamma = 0.99$ — discount factor
- $\lambda = 0.95$ — GAE parameter，控制bias-variance tradeoff

完整的PPO loss:

$$L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)$$

- $c_1 = 0.5$ — value function loss coefficient
- $c_2$ — entropy bonus coefficient (FLaRe设为0)
- $S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ — policy entropy

参考: PPO paper https://arxiv.org/abs/1707.06347  
GAE paper: https://arxiv.org/abs/1506.02438  
Deadly triad discussion: http://incompleteideas.net/book/RLbook2020.pdf (Chapter 11)

### 3.2 Small learning rate

From scratch PPO navigation task的SoTA用的LR是 $2 \times 10^{-4}$。FLaRe fine-tuning要把LR降一个数量级。论文强调：所有tasks和所有experiments都用同一个LR，不做task-specific tuning。

**Intuition**：预训练policy已经处于一个good loss basin，过大的gradient step会让它跳出该basin，destroy学到的behavior prior。这与LoRA、small LR fine-tuning LLM的intuition类似。

### 3.3 Disable entropy bonus

PPO默认的entropy bonus $c_2 S[\pi_\theta]$ 鼓励exploration，防止policy过早collapse到suboptimal deterministic policy。

**问题**：fine-tune开始时，预训练policy已经在data distribution上是相对confident的（low entropy on meaningful actions）。如果此时强制increase entropy，会distort policy gradient——把probability mass从有意义的action推到irrelevant action上，造成unlearning。

**Ablation验证**：EB=0.2导致success rate collapse到0。

### 3.4 Disable feature sharing between actor and critic

Standard practice：actor和critic共享visual backbone，可以联合学习useful features。

**问题**：RL fine-tuning时critic gradient会backprop到shared backbone，修改那些预训练时学到的、对action prediction至关重要的features，造成actor prediction deterioration。

**FLaRe解决方案**：
- Actor网络：用预训练SPOC的weight初始化，policy head
- Critic网络：copy预训练SPOC的weight和架构，但policy head被替换为randomly initialized value head

这样critic update完全不会通过shared backbone影响actor的action prediction。

### 3.5 四个tricks的necessity（Ablation）

Section V-E的Fig 6(b) ablation：
- **PPO → SAC**: collapse
- **LR 10x (2e-4 → 2e-4)**: collapse  
  （注意原文有typo，应该是从2e-5提高到2e-4，10倍）
- **Shared AC**: collapse
- **EB=0.2**: collapse

任何一个去掉都导致Fetch task success rate瞬间掉到0。这暗示大规模BC→RL fine-tuning确实存在一个非常微妙的stability regime，必须同时满足所有四个条件。

---

## 四、SPOC Base Model 架构详解 (Appendix E + Fig 7)

```
                   Two RGB Images              Text Instruction
                  (i_a, i_b)                        (l)
                       |                              |
              ┌────────┴────────┐              ┌───────┴───────┐
              │                 │              │                │
         DINOv2 (frozen)   DINOv2 (frozen)   T5 text encoder
              │                 │              │
              ▼                 ▼              ▼
       visual tokens     visual tokens    text features
       ν_a (nav cam)     ν_b (arm cam)    τ
              │                 │              │
              └───+ camera-type embedding +─────┘
                              │
                  ┌───────────┴───────────┐
                  │   STATE token σ       │
                  │   Transformer State    │
                  │   Encoder (non-causal)│
                  └───────────┬───────────┘
                              │
                       state feature s ∈ R^d
                              │
                  ┌───────────┴───────────┐
                  │ + temporal positional  │
                  │   encoding            │
                  │ + previous action     │
                  │   embedding          │
                  │                       │
                  │  Causal Transformer   │
                  │  Decoder (LLaMA 2     │
                  │   inspired)           │
                  └───────────┬───────────┘
                              │
                       belief vector
                              │
                       actor head → action
                       (or value head for critic)
```

### 4.1 Vision encoder

输入：$i_a, i_b \in \mathbb{R}^{H \times W \times 3}$ — 两个camera (navigation + arm) 的RGB图

DINOv2处理：输出patch-wise representation
$$r \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times h}$$

其中 $h$ 是DINOv2 hidden dim (论文应该是ViT-B的768)。$\frac{H}{14} \times \frac{W}{14}$ 是patch数，14是DINOv2 patch size。

Reshape并project到：
$$\nu_{raw} \in \mathbb{R}^{n_{patch} \times d_{encoder}}$$

加learnable **camera-type embedding**区分两个相机，得到最终 $\nu$。

DINOv2 frozen throughout training，保证visual features不被RL corrupt。

### 4.2 Transformer State Encoder

输入：visual representation $\nu$ + text feature $\tau$ + learnable STATE token $\sigma$。

non-causal transformer encoder，输出STATE token对应的输出作为state feature $s \in \mathbb{R}^d$。

这相当于一个text-conditioned visual state representation。

### 4.3 Causal Transformer Decoder

处理partial observability和long-horizon依赖。

输入：当前state feature + 之前所有step的state features。

Additive:
- Sinusoidal temporal position encoding
- Previous time step action embedding

输出 belief vector → 通过actor head生成action prediction。

论文用LLaMA 2 decoder block替换原SPOC decoder，加速训练和inference。  
LLaMA 2: https://arxiv.org/abs/2307.09288

---

## 五、Problem Formulation (Section III)

每个task $T \in \mathcal{T}$是一个language-conditioned POMDP:

$$(S, \mathcal{A}, \mathcal{P}, R, O, \mathcal{L}, P(s_0), \gamma)$$

变量详解：
- $S$ — state space (unobservable)
- $\mathcal{A}$ — action space (actuators)
- $\mathcal{P}$ — Markovian transition model $P(s'|s,a)$
- $R$ — **sparse** reward function, $R: \mathcal{L} \times S \rightarrow \{0, 1\}$，输入language instruction $l$ 和 state $s$，输出binary值表示task是否完成
- $O$ — observation space
- $\mathcal{L}$ — natural language instructions集合
- $P(s_0)$ — 初始state分布
- $\gamma$ — discount factor

每个task $T$定义自己的instruction set $\mathcal{L}_T$。如ObjectNav可以有 "find a mug", "go to an apple"等。

每个episode开始：sample $l_T \in \mathcal{L}_T$ 和 $s_0 \sim P(s_0)$。

目标：训练 $\pi_\theta^T$ 最大化：

$$\mathbb{E}_{\mathcal{L}_T, \pi} \sum_t R(s_t, l)$$

---

## 六、Experimental Results 详解

### 6.1 Seen capabilities (Table I) — CHORES-S benchmark

CHORES-S是4个long-horizon mobile manipulation tasks。

**关键比较维度**：
- **Fair comparison**：baselines用sparse reward (和FLaRe一样)
- **Unfair comparison**：baselines用dense hand-coded reward + privileged info

注意unfair baseline训练到convergence，往往training steps比FLaRe多很多倍。比如Poliformer (Dense)在ObjectNav训练300M steps，是FLaRe的15倍。

**结果**（Success rate, SEL in parentheses）:

| Task | FLaRe | PIRLNav | JSRL | SPOC | Poliformer-Sparse | Poliformer-Dense | EmbSigLIP-Dense |
|------|-------|---------|------|------|-------------------|------------------|-----------------|
| ObjectNav | 85.0 (67.6) | 20.0 (7.0) | 21.0 (15.6) | 55.0 (42.2) | 14.5 (10.4) | 85.5 (61.2) | 36.5 (24.5) |
| Fetch | 66.9 (54.7) | 0.0 (0.0) | 2.9 (2.8) | 14.0 (10.5) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| PickUp | 91.8 (90.4) | 0.0 (0.0) | 50.9 (47.7) | 90.1 (86.9) | 0.0 (0.0) | 90.1 (88.7) | 71.9 (52.9) |
| RoomVisit | 70.4 (67.1) | 12.5 (11.0) | 19.0 (18.6) | 40.5 (35.7) | 12.5 (12.5) | 12.5 (10.9) | 16.5 (11.9) |

Average success rate = (85.0+66.9+91.8+70.4)/4 ≈ 78.5% (论文写79.5%)

关键观察：
1. FLaRe用sparse reward，在4个task里**3个**击败了用dense reward + privileged info的Poliformer-Dense。
2. Fetch task — 最具挑战性，所有baselines基本失败，FLaRe达66.9%。Fetch需要先navigate找object然后pick up，long-horizon composition。
3. PIRLNav和JSRL这两个"closest to our setting"的方法表现都很差——single-task small-scale setting的优势在大规模multi-task上失效。

参考: PIRLNav https://arxiv.org/abs/2301.02132 ; JSRL https://arxiv.org/abs/2202.10366

### 6.2 Novel capabilities (Table II)

测试3个novel task (require capabilities unseen during pretraining):

1. **ObjNavRelAttr**：用relative attributes比较找物体，e.g. "find the largest apple"。需要枚举所有候选物体，reason their properties。
2. **RoomNav**：navigate到room type而不是object，e.g. "go to the kitchen"。
3. **ObjNavAfford**：object affordance理解，e.g. "find something I can sit on"。

Baselines:
- Poliformer (Sparse)
- SPOC++ — 同架构但加1M frames expert demos per task的BC baseline
- Poliformer (Dense) — privileged dense reward

| Task | FLaRe | Poliformer (Sp) | SPOC++ | Poliformer (De) |
|------|-------|-----------------|--------|-----------------|
| ObjNavRelAttr | 71.0 (63.6) | 6.7 (6.7) | 54.5 (44.6) | 36.1 (32.4) |
| RoomNav | 91.6 (85.6) | 57.0 (51.8) | 74.5 (59.9) | 75.0 (62.4) |
| ObjNavAfford | 79.7 (70.6) | 35.5 (29.4) | 62.4 (50.6) | 53.8 (43.1) |

关键结论：**FLaRe在所有3个novel tasks上SoTA**。这意味着只需指定新的 $R_n$ (success criteria) 和 $L_n$ (language instructions)，就能on-the-fly定义新task并fine-tune。Suggests a path toward continual adaptation。

### 6.3 Real robot (Table III)

Stretch RE-1 robot在真实6-room apartment（unseen during training），no real-world fine-tuning，直接部署。

| Task | FLaRe | SPOC | Poliformer (Dense) |
|------|-------|------|--------------------|
| ObjectNav | 94.4 | 50.0 | 83.3 |
| Fetch | 66.7 (55.6) | 33.3 (11.1) | X |
| PickUp | 86.7 (66.7) | 66.7 (46.7) | X |
| RoomVisit | 75.0 | 50.0 | X |

括号内是"policy success (proximity)"，括号外是full success（包含heuristic grasping）。Manipulation tasks用heuristic grasping model（following SPOC）。

Real-world average 80.7% SR，比best prior work (Poliformer Dense on ObjectNav 83.3% + SPOC其他task) 高30.7%。

### 6.4 Cross-embodiment adaptation

SPOC trained on Stretch RE-1（带manipulation DoF），adapt到Locobot（无manipulation DoF，camera可旋转且更窄FOV，安装位置更低）。

技术细节：mask out invalid actions，repurpose两个invalid actions来控制camera。

| Method | Success Rate | SEL |
|--------|--------------|-----|
| FLaRe | 72.0 | 47.2 |
| Poliformer zero-shot | 57.5 | 30.1 |
| Poliformer (Sparse) | 44.0 | 29.7 |

### 6.5 Cross-behavior adaptation (Fetch task)

加入reward shaping term实现行为控制:

$$R_{new}(s_t, l) = R(s_t, l) + \beta \cdot \text{penalty}_t$$

| Behavior | Success Rate ↑ | Episode Length ↓ | # Collisions ↓ |
|----------|----------------|-------------------|----------------|
| FLaRe (baseline) | 66.9 | 258.2 | 10.0 |
| + Step Penalty (-0.01/step) | 65.7 | 222.8 | 10.0 |
| + Collision Penalty (-0.5/collision) | 66.7 | 251.2 | 3.1 |

仅6小时训练就能shape behavior，success rate几乎不变。这非常有用——可以在不retrain整个pipeline的情况下调整deploy behavior。

---

## 七、Hyperparameters (Table IV)

| Parameter | Value |
|-----------|-------|
| Total Rollouts | 32 |
| Learning Rate | 0.0002 |
| Mini Batch per Update | 1 |
| Update Repeats | 4 |
| Max Gradient Norm | 0.5 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| PPO Clipping | 0.1 |
| Value Loss Weight | 0.5 |
| Entropy Loss Weight | 0.0 ← disabled |
| Steps for PPO Update | 128 |
| State Encoder Layers | 3 |
| State Encoder Hidden Dims | 512 |
| State Encoder Heads | 8 |
| Causal Decoder Layers | 3 |
| Causal Decoder Hidden Dims | 512 |
| Causal Decoder Heads | 8 |

**重要：所有tasks、所有experiments用同一组hyperparameters**。No per-task tuning。这是scalability的关键。

Training steps:
- ObjectNav, RoomVisit: 20M
- Fetch, PickUp: 50M
- ObjNavRelAttr, ObjNavAfford: 50M
- RoomNav: 20M
- Cross-embodiment: 30M

Baselines用3x (nav) 或2x (manip) steps做fair comparison。

---

## 八、CHORES Benchmark 细节 (Appendix D)

### 8.1 Observation space

- 2 ego-centric 384×224 RGB cameras (orthogonal directions)
- 一个朝向navigation，一个朝向arm
- + natural language instruction (re-sampled per episode)

### 8.2 Action space (20 discrete actions)

- Move Base (±20 cm) — 2 actions
- Rotate Base (±6°, ±30°) — 4 actions
- Move Arm (x, z) (±2 cm, ±10 cm) — 4 actions
- Rotate Grasper (±10°) — 2 actions
- pickup, dropoff, done with subtask, terminate — 4 actions

### 8.3 Tasks

| Task | Description | Max Steps |
|------|-------------|-----------|
| ObjectNav | Locate an object category: "find a mug" | 600 |
| PickUp | Pick up a specified object in line of sight: "pick up a mug" | 600 |
| Fetch | Find and pick up an object: "locate a mug and pick up that mug" | 600 |
| RoomVisit | Traverse the house, visit every room | 1000 |

191,568 ProcTHOR houses，10:1 train/test split。

参考: ProcTHOR https://arxiv.org/abs/2206.06994 ; Objaverse https://arxiv.org/abs/2212.08051 ; AI2THOR https://arxiv.org/abs/1712.05474

---

## 九、Building the Intuition — 总结核心insight

### 9.1 为什么BC pretrain + RL fine-tune有效？

BC预训练提供两件事：
1. **Behavior prior**：合理的exploration起点，不需要从random policy开始explore huge state-action space。
2. **Semantic representations**：transformer在大规模多task demonstration上学到的features，对未见task也useful。

RL fine-tune提供：
1. **Align surrogate → true objective**：从imitation objective到task completion objective。
2. **Close the distribution gap**：让policy在自己induced distribution上optimize，避免compounding error。

### 9.2 为什么大规模BC + RL这么fragile？

设想你有个100M+参数的transformer，pretrain后处于一个specific loss basin。RL的gradient来自non-stationary value function，来自noisy advantage estimate，方向highly noisy。如果LR大或off-policy（reward distribution shift）或entropy bonus（直接distort action distribution），gradient update会让policy跳出good basin，进入random region。Collapse发生。

**FLaRe的四个stabilization trick组合形成了一个"safe fine-tuning regime"**：on-policy保证gradient来源consistency，小LR保证update amplitude小，disable entropy bonus避免直接distort action distribution，separate actor-critic避免critic gradient污染actor features。

这其实和LLM RLHF中观察到的现象非常类似——RL fine-tuning大语言模型也fragile，需要careful stabilization (e.g. KL penalty to reference policy，small LR，PPO clipping)。

### 9.3 真正novel的贡献

1. **第一个在大规模robotics foundation model + transformer + 真实机器人上验证RL fine-tuning**的工作。前作（JSRL, PIRLNav, AWAC, DQfD等）都是small MLP, single task, no real robot。
2. **Sparse reward + zero reward engineering**。只要能定义binary success criteria，就能on-the-fly fine-tune到新task/embodiment/behavior。
3. **15x training time reduction**。相比Poliformer的300M steps，FLaRe只需20M就能达到competitive performance。

### 9.4 Limitations

依赖simulation environments。对于涉及liquid、soft body、deformable object的task，难以simulate，可能需要在real world直接fine-tune，而real world sample efficiency差，挑战大。

Phone2Proc https://arxiv.org/abs/2308.10756 和 Reconciling Reality through Simulation https://arxiv.org/abs/2403.03949 等real-to-sim-to-real方向可能缓解。

### 9.5 与RLHF的类比

LLM的RLHF pipeline：SFT (BC) → Reward Model → PPO fine-tune，加KL penalty约束到reference policy。

FLaRe的pipeline：SPOC BC pretraining → Sparse Reward (无需reward model，task completion直接binary) → PPO fine-tune with various stabilizations。

区别在于LLM有explicit reward model学习human preference，FLaRe直接用environment的task completion signal，更principled。

但都需要careful stabilization防止catastrophic forgetting / destructive updates。

RLHF PPO paper (InstructGPT): https://arxiv.org/abs/2203.02155

---

## 十、Reference汇总

### Methods
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- SAC: https://arxiv.org/abs/1801.01290
- JSRL: https://arxiv.org/abs/2202.10366
- PIRLNav: https://arxiv.org/abs/2301.02132
- AWAC: https://arxiv.org/abs/2006.09359

### Foundation Models
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- RT-X / Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Octo: https://arxiv.org/abs/2405.12213
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboCat: https://arxiv.org/abs/2306.11506
- ViNT: https://arxiv.org/abs/2306.14846

### Backbone Models
- DINOv2: https://arxiv.org/abs/2304.07193
- LLaMA 2: https://arxiv.org/abs/2307.09288
- T5 (Sentence-T5): https://arxiv.org/abs/2108.08877

### Simulators & Data
- AI2THOR: https://arxiv.org/abs/1712.05474
- ProcTHOR: https://arxiv.org/abs/2206.06994
- Objaverse: https://arxiv.org/abs/2212.08051
- RoboTHOR: https://arxiv.org/abs/2004.06999
- Phone2Proc: https://arxiv.org/abs/2308.10756

### Hardware
- Stretch RE-1: https://hello-robot.com/ ; paper https://arxiv.org/abs/2209.12180
- Locobot: https://arxiv.org/abs/1811.07011 (Learning Robot Skills)

### Theory
- DAGGER (compounding error): https://arxiv.org/abs/1011.0686
- Sutton & Barto RL book (deadly triad): http://incompleteideas.net/book/RLbook2020.pdf
- KV-cache (Pope et al.): https://arxiv.org/abs/2211.05102
- AllenAct framework: https://arxiv.org/abs/2008.12760

### FLaRe project
- Project website: robot-flare.github.io

---

## 十一、Critique & Open Questions

1. **Base model lock-in**: FLaRe依赖SPOC，能否work在RT-2, OpenVLA, Octo等其他foundation model上还需验证。论文claim"in principle can work on any foundational robotics model"，但实证只在一个。
2. **Sparse reward availability**: 假设能定义binary success criteria，对许多real-world task不易。
3. **Sim-to-real gap remain**: Domain randomization + DINOv2 + frozen backbone trick能在特定task work，但对contact-rich manipulation可能不够。
4. **Critic initialization**: Critic用预训练SPOC weight初始化（除value head random init），但预训练policy网络是个action predictor，其internal features可能不直接适合value prediction。是否有更聪明的critic初始化方式？
5. **Continual adaptation risk**: Cross-behavior实验只验证6小时fine-tune。多次sequential adaptation会否导致catastrophic forgetting？需要EWC-style regularization或类似方法保护critical weights。
6. **No comparison with offline RL pretraining**: 没比较offline RL (CQL, IQL) pretraining路线。Cal-QL https://arxiv.org/abs/2403.14578 等方法可能在某些setting下更principled。

整体上，FLaRe是个非常solid的工作，把"用RL fine-tune BC foundation model"这条路线在大scale + real robot上首次跑通，4个stabilization trick的necessity通过ablation充分验证。后续在LLM RLHF与robotics RLHF的类比研究、continual adaptation、reward shaping等方面还有很多延展空间。
