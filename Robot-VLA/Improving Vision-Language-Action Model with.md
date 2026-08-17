---
source_pdf: Improving Vision-Language-Action Model with.pdf
paper_sha256: 8e370aeae996a21c68e6587eb4fc64cbe4effb09d1dfddba2a1bc396b4c54ba8
processed_at: '2026-08-05T09:22:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：iRe-VLA到底在干嘛

---

## 一句话说清楚

**让一个已经会做事的机器人，通过自己不断尝试，变得越来越厉害，同时不会把原来会的忘掉。**

---

## 背景的故事

想象你train了一个机器人，它能听懂"把杯子拿起来"这种话，也能看着桌面做动作。这个机器人脑子里装了一个巨大的"世界知识库"（BLIP-2，3B参数的VLM），它看过几十亿张图片和文字，所以它知道"杯子长什么样""胡萝卜是什么颜色"这种常识。

这个机器人是通过**模仿学习**（SFT）训练的——你给它看2000条人类遥控操作的视频，它学着复现。这就像小孩看大人做饭，记住动作序列。

但模仿学习有个天花板：**机器人只能做人类演示过的动作**。如果你没演示过"拿茄子"，它就不会拿茄子，即使它"认得"茄子是什么。

那我们加RL让它自己探索不就好了？

---

## 核心矛盾：RL会把大模型搞崩

你拿PPO去fine-tune这个3B参数的VLA model，结果很尴尬：**performance掉下来了**。不是一点点，是5个任务里4个变差。

为什么？因为RL的gradient特别noisy。RL靠sparse reward学习（成功给1分，失败给0分），这个信号要backprop回整个3B网络，像拿着一把大锤子去敲一个精密的瑞士手表——把pretrained representation全敲乱了。

RLHF在LLM上能work，是因为ChatGPT那个场景是bandit（一轮对话一个reward），signal干净，还有KL divergence做anchor防止跑偏。Robotics是几百步才给一个reward，signal-to-noise ratio差太多。

---

## 他们的trick：把探索和沉淀分开

核心idea特别简单——**别让RL直接碰大模型，让SL去做沉淀**。

具体就是一个loop，分两步交替跑：

### 第一步：RL探索（frozen VLM）

把VLM那个3B的大脑子**冻住**，只训练底下那个小action head（几百K参数）。这样RL的noisy gradient只影响这个小head，不会搞坏pretrained representation。

这就像你不让一个博士去重新学走路，你只让他练手腕的精细动作。脑子里的常识不变，只是手上的肌肉在调整。

然后让机器人在新任务上自己试，比如"拿茄子"。它一开始用SFT学到的generalization去试，偶尔成功，把成功的trajectory存下来。paper里用SACfD这个算法，还从demo buffer里sample一半数据来加速。

工程细节：VLM forward一次把latent存到buffer里，之后RL训练直接用latent，不用反复跑3B网络。这样4090就能跑local RL。

### 第二步：SL沉淀（full model）

RL收集到一批成功trajectory后，解冻VLM，用LoRA去fine-tune**整个model**。loss还是MSE，数据是**expert data + RL收集的新data**一起喂。

这一步干两件事：
1. 把RL探索到的新行为"蒸馏"进VLM的representation，让大模型真正理解"茄子"这个概念
2. 用expert data防止它忘了原来会的task（catastrophic forgetting）

这就像博士练完手腕动作后，回去写paper总结，把经验固化成知识。

### 循环

每个新任务跑一轮：RL探索 → SL沉淀 → RL探索 → SL沉淀...

---

## 为什么这个work？intuition

我觉得核心insight是：**RL和SL各擅长不同的事，别逼它们干自己不擅长的**。

- **RL擅长探索**：发现新行为，但gradient很脏
- **SL擅长沉淀**：stable地更新representation，但不会自己explore

你把exploration这个任务交给RL（只动小head），把representation update这个任务交给SL（动整个model），就各自在comfort zone工作。

这就像公司里的R&D和production：R&D去试新东西（可能失败很多），production把成功的东西规模化。你不会让production line去天天实验，也不会让R&D去管量产。

---

## 效果

### Simulation

MetaWorld上：
- 原来SFT model在新任务上成功率~40%
- PPO-Replay（标准RL+经验回放）~35%，**比不训还差**
- iRe-VLA ~89%，而且original task没掉

Franka Kitchen上有个left-door-open任务，expert data不够，SFT只有43%成功率。iRe-VLA提到83%——说明即使数据不足，通过online interaction能自己补上。

Unseen task generalization也从0.51提到0.80。这个很有意思：**学会更多task后，泛化能力变强了**。这和人类的intuition一致——会做更多菜的人，拿到没见过的食材也能即兴发挥。

### Real World

Panda arm，local 4090跑RL，remote 4×A100跑SL。

拿茄子/胡萝卜：35% → 80%。一个新任务训练一小时。

这个效率很惊人——real-world RL一小时学会一个新skill，和SERL（专门做real-world RL的SOTA）持平。但iRe-VLA还多了VLM的common sense。

---

## 一个关键ablation

如果两个stage都冻VLM会怎样？**性能下降**。

这说明：**必须让VLM在SL stage更新**，否则RL探索到的信息只停留在action head里，进不到大模型的representation。大模型不知道"茄子"是什么，generalization就上不去。

这验证了整个设计：RL负责发现，SL负责让大模型吸收。

---

## 我联想到的几条线

### 1. Decision Transformer的影子

Decision Transformer把RL变成sequence modeling，用SL的stable training替代RL的unstable training。iRe-VLA也是类似思路——用SL来稳定RL的过程。

### 2. DPO对RLHF的改造

DPO把preference optimization从RL变成SL，思路类似：RL太难训，那就找一个等价的SL formulation。iRe-VLA没法完全用SL替代RL（因为要explore），但可以把RL的"危险部分"隔离在小head里。

### 3. RLHF里KL anchor的类比

InstructGPT用KL divergence约束policy不偏离reference model太远，防止collapse。iRe-VLA的SL stage本质上也是这个作用——用expert data把model"拉回"stable region。

### 4. Continual Learning

iRe-VLA其实是在做continual learning——不断学新task不忘旧task。经典方法是EWC、replay等。这里用的是replay（expert data + RL data一起训）+ 模块化训练（frozen VLM保护core knowledge）。

### 5. Frozen backbone + trainable head这个pattern

这其实是transfer learning的经典套路：pretrained feature extractor + task-specific head。iRe-VLA把这个用到RL上——RL阶段只动head，SL阶段才动backbone。这个pattern在resource constrained场景下特别实用。

### 6. AlphaGo的影子

AlphaGo也是先模仿人类expert（SL），再self-play（RL）。但AlphaGo的model小，能直接RL fine-tune。VLA model太大，得拆开。

### 7. 大model + RL的根本困难

这个paper其实暴露了一个深层问题：**我们目前没有好的方法让billion-scale model做online RL**。RL的variance太大，大model对variance敏感。iRe-VLA是workaround，但不是终极解。

可能的终极方向：
- 用world model做imagination（Dreamer style），把online RL变成offline RL
- 用preference optimization替代reward-based RL（DPO for robotics）
- 用hierarchical decomposition，让RL只在低维空间进行

---

## Limitations

Paper承认：只能提升seen skill type，不能学全新skill。sparse reward下explore不到全新行为模式。

我觉得还有几个没提的：
- Token Learner把VLM所有token压成一个vector，丢了spatial信息
- MSE loss对multi-modal action distribution不友好（两种valid action会被average掉）
- Binary reward对长horizon task太sparse，可以加dense reward from VLM（参考Eureka）

---

## 一句话总结

**大模型做RL会崩，那就让小head去做探索的脏活，大模型只做SL的沉淀活。循环交替，大模型越来越强。**

这个decoupling思路很general，我觉得未来很多"large model + online learning"的场景都能用——game playing、tool use、autonomous agent等。

---

Reference links:
- [iRe-VLA](https://arxiv.org/abs/2411.18257)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [SERL](https://arxiv.org/abs/2401.16013)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [DPO](https://arxiv.org/abs/2305.18290)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Stabilizing Transformers for RL](https://arxiv.org/abs/1910.06764)
- [Eureka (VLM as reward)](https://arxiv.org/abs/2310.12931)
- [AlphaGo](https://www.nature.com/articles/nature24270)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [SACfD](https://arxiv.org/abs/1707.08817)

---

# iRe-VLA: 用Online RL改进Vision-Language-Action Model

Andrej，这篇paper触及了一个非常核心的问题——**如何让large foundation model在物理世界中持续学习**。这其实是robot learning领域长期的一个痛点：我们有强大的pretrained VLM，我们知道SFT的局限，但直接把online RL套到billion-scale model上会崩。我来把这篇paper拆开讲，顺便补充一些我联想到的相关工作。

---

## 1. 问题背景：为什么需要这个工作？

### 1.1 VLA模型的现状

目前的VLA model line（RT-2, HiRT, Roboflamingo, Octo, OpenVLA等）基本都是走**SFT路线**：把expert demonstrations里的action当作target，用MSE loss或者next-token prediction去训练。这条路确实work，但有几个根本问题：

- **Expert data贵且少**：teleoperation收集2k条trajectory就要花很多时间，覆盖的task和object variation有限
- **Distribution shift**：SFT学的是expert的轨迹分布，但部署时policy自己rollout会偏离这个分布（covariate shift问题，参考DAgger [Ross et al. 2011](https://arxiv.org/abs/1011.0686)）
- **无法explore新解法**：SFT本质是imitation，只能复现expert见过的行为

### 1.2 为什么不能直接用RL？

直觉上，RLHF在LLM上非常成功（[InstructGPT](https://arxiv.org/abs/2203.02155)），那把它搬到VLA上不就行了？但paper里empirically发现：**直接用PPO fine-tune整个VLA model会performance drop**。Figure 4里橙色线（full fine-tune）明显比frozen VLM只训action head的蓝色线差，甚至在Metaworld 5个task里有4个任务performance下降。

我联想到几个相关解释：
- [Parisotto et al. 2020](https://arxiv.org/abs/1910.06764) "Stabilizing Transformers for RL"——transformer在RL里容易collapse，需要特殊设计
- 大model的pretrained representation是脆弱的，noisy RL gradient会破坏这些representation
- LLM的RLHF其实是个bandit问题（单步），而robotics是long-horizon sparse reward的POMDP，signal-to-noise ratio差很多

---

## 2. 方法：iRe-VLA的核心设计

### 2.1 Architecture

VLA model结构（Figure 2a）：

```
Image o + Language instruction i
        ↓
   BLIP-2 3B (frozen in RL stage, LoRA in SL stage)
        ↓
   Last hidden representation h ∈ R^(m×d)
        ↓
   Token Learner (permutation invariant pooling)
        ↓
   h' ∈ R^d
        ↓
   MLP Action Head
        ↓
   Action a ∈ R^(d_a)
```

其中：
- $m$ = VLM的token数
- $d$ = embedding dimension
- $d_a$ = action dimension（end-effector pose change + gripper status）

**Token Learner**这个设计来自[Set Transformer (Lee et al. 2019)](https://arxiv.org/abs/1810.00825)，本质是attention-based pooling，把variable-length的token set压成fixed-size representation。这在robotics里很常见，因为action是低维的连续向量，而VLM输出是高维token序列，需要information bottleneck。

### 2.2 三阶段pipeline

**Stage 0: SFT on Expert Data**

$$J^0(\theta, \phi) = \mathbb{E}_{(o, l, a) \sim D_e} \left[ \| \pi_{\theta, \phi}(o, l) - a \|_2^2 \right]$$

变量解释：
- $\theta$ = VLM的LoRA参数
- $\phi$ = action head参数
- $D_e$ = expert dataset
- $(o, l, a)$ = (observation, language instruction, action) tuple
- $\| \cdot \|_2^2$ = L2 norm的平方，即MSE

这一步得到initial policy $\pi_{\theta, \phi}^0$，作为后续RL的warm start。

**Stage 1: Online RL with Frozen VLM**

$$J^1(\phi) = \mathbb{E}_{((s_0, o_0, a_0), (s_1, o_1, a_1), \dots) \sim p_\phi} \left[ \sum_t \gamma^t R(o^t, a^t) \right]$$

变量解释：
- $\gamma$ = discount factor（通常0.99）
- $R(o^t, a^t)$ = reward function，paper里用binary reward（成功=1，失败=0）
- $p_\phi$ = trajectory distribution induced by policy
- 上标$t$ = time step

关键设计：**只更新$\phi$（action head），冻结$\theta$（VLM）**。这相当于在VLM提供的fixed representation上做RL，把问题降维成一个small-scale RL problem。Paper里用SACfD（Soft Actor-Critic with Demonstrations，[Vecerik et al. 2017](https://arxiv.org/abs/1707.08817) + [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)），sample 50% from demo buffer, 50% from online buffer。

还有一个工程trick：VLM的latent只forward一次，存到replay buffer里复用，这样RL训练时不用每次都跑3B model的forward——这对real-world RL是必须的，否则control frequency太低。

**Stage 2: SL on Expert + Online Data**

$$J^2(\theta, \phi) = \mathbb{E}_{(o, l, a) \sim D_e \cup D_{RL}} \left[ \| \pi_{\theta, \phi}(o, l) - a \|_2^2 \right]$$

$D_{RL}$ = Stage 1里收集的成功trajectories。这里**整个model（包括VLM的LoRA）都更新**，目的是：
1. 把RL探索到的新行为"蒸馏"进VLM的representation
2. 用$D_e$防止catastrophic forgetting（[McCloskey & Cohen 1989](https://stanford.io/3abc)的经典问题）

**迭代**：Stage 1 → Stage 2 → Stage 1 → ... 对每个new task都跑一轮。

### 2.3 为什么这个设计work？我的intuition

核心insight是**decoupling exploration和representation learning**：

- RL的本质是exploration，需要noisy gradient去探索新行为，但noisy gradient会破坏pretrained representation
- SL的本质是stable的gradient signal，适合更新large model的representation
- 把两者分到不同stage，既享受RL的exploration能力，又保持SL的stability

这让我联想到几个相关工作：
- [Decision Transformer](https://arxiv.org/abs/2106.01345)：把RL变成sequence modeling，本质也是用SL的stable training替代RL的unstable training
- [APO (Anchor Preference Optimization)](https://arxiv.org/abs/2310.03708)：在preference optimization里用anchor point防止collapse
- [DPO](https://arxiv.org/abs/2305.18290)：把RLHF的RL step变成SL step，思路类似
- [TRPO/PPO](https://arxiv.org/abs/1707.06347)的trust region：用constraint限制policy update幅度，iRe-VLA则是用SL stage来"reset"回stable region

---

## 3. 实验细节

### 3.1 Simulated Experiments (Table I)

**MetaWorld**：
| Method | Original 25 tasks | 5 New RL tasks | 10 Unseen tasks |
|--------|-------------------|----------------|-----------------|
| SFT Policy | 0.83 | avg ~0.41 | 0.51 |
| PPO-Replay | 0.69 ↓ | avg ~0.35 | 0.39 |
| **iRe-VLA** | **0.83** (保持) | avg ~0.89 | **0.80** |

关键观察：
- PPO-Replay在original task上从0.83掉到0.69——**catastrophic forgetting + representation破坏**
- iRe-VLA在original task保持不变，new task大幅提升，unseen task generalization从0.51→0.80

**Franka Kitchen**：
| Method | 5 Expert tasks | 2 Color variation tasks |
|--------|----------------|------------------------|
| SFT | avg ~0.76 | avg ~0.72 |
| PPO-Replay | avg ~0.51 | avg ~0.47 |
| **iRe-VLA** | avg ~0.90 | **0.995** |

特别注意left-door-open从0.43→0.83，说明即使expert data不够，iRe-VLA也能通过online interaction提升。

### 3.2 Real-world Panda Experiments (Figure 6)

- 2000条teleoperation trajectories，5类task
- RL stage在local RTX 4090上跑，SL stage在4×A100 remote server
- Pick eggplant/carrot：0.35 → 0.80
- Unseen objects pick：0.37 → 0.61
- 训练时间每个new task ~1小时（和[SERL](https://arxiv.org/abs/2401.16013)持平）

这个效率很impressive，说明**frozen VLM + small action head RL**的compute budget和小型RL method差不多，但benefit from VLM的representation。

### 3.3 Ablation (Figure 5)

"iRe-VLA-freeze"（两个stage都冻结VLM）比iRe-VLA差——说明**必须让VLM在SL stage更新**，否则online data无法改善VLM representation，generalization就上不去。

---

## 4. 更深层的思考

### 4.1 和其他VLA工作的对比

| Method | SFT | RL | 关键特点 |
|--------|-----|----|---------|
| [RT-2](https://arxiv.org/abs/2307.15818) | ✅ | ❌ | Action tokenization，co-training with web data |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | ✅ | ❌ | 开源7B VLA，LoRA fine-tune |
| [RoboCat](https://arxiv.org/abs/2306.11706) | ✅ | ✅ (limited) | Self-improving via RL但架构不同 |
| [SERL](https://arxiv.org/abs/2401.16013) | ❌ | ✅ | 纯RL，small policy，sample efficient |
| **iRe-VLA** | ✅ | ✅ (iterative) | Decoupled RL+SL for large VLA |

### 4.2 和LLM RLHF的对比

| Aspect | LLM RLHF | iRe-VLA |
|--------|----------|---------|
| Environment | Offline, bandit | Online, long-horizon POMDP |
| Reward | Reward model from preference | Task success binary |
| Model size update | Full model PPO | Only action head in RL |
| Stability mechanism | KL constraint to ref model | Iterative SL reset |

LLM RLHF能直接训整个model是因为：1) bandit环境signal clean，2) KL anchor提供stability。Robotics没这俩luxury，所以iRe-VLA用iterative SL来提供anchor。

### 4.3 局限性（paper承认的 + 我补充的）

Paper承认：只能提升seen skill type，不能学全新skill（sparse reward下explore不到）。

我补充几点：
1. **Token Learner是bottleneck**：把VLM的所有token压成一个vector，丢掉了spatial信息，对需要fine-grained visual reasoning的task可能不够
2. **Binary reward太sparse**：对长horizon task效率低，可以借鉴[RL with dense reward from VLM](https://arxiv.org/abs/2310.12931) (Eureka) 或[VLM as reward](https://arxiv.org/abs/2310.02724)
3. **没有handle multi-modal policy**：一个language condition可能对应多种valid action distribution，MSE loss会average它们
4. **Real-world的action head还很小**：如果让action head更大或hierarchical（参考[HiRT](https://arxiv.org/abs/2405.13713)），可能explore能力更强

### 4.4 未来方向联想

1. **Online DPO for VLA**：把RL stage也变成preference learning，可能更stable
2. **World model + planning**：用[VLM as world model](https://arxiv.org/abs/2308.01399) + Dreamer-style imagination，避免online sample cost
3. **Hierarchical RL**：用LLM做high-level plan + VLA做low-level control，[DoReMi](https://arxiv.org/abs/2307.00329)是早期尝试
4. **Curriculum learning**：自动生成task sequence，参考[POET](https://arxiv.org/abs/1901.01753)或[AMIGO](https://arxiv.org/abs/2406.16782)
5. **Action chunking + diffusion**：用[Diffusion Policy](https://arxiv.org/abs/2303.04137)替换action head，可能更好处理multi-modal action distribution

---

## 5. Reference Links

- **Paper**: [iRe-VLA on arXiv](https://arxiv.org/abs/2411.18257)（paper的arXiv版本，作者Yanjiang Guo et al.）
- **BLIP-2**: [arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
- **RT-2**: [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- **OpenVLA**: [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)（开源VLA，适合复现）
- **SERL**: [arxiv.org/abs/2401.16013](https://arxiv.org/abs/2401.16013)（real-world RL的software suite）
- **SACfD**: [arxiv.org/abs/1707.08817](https://arxiv.org/abs/1707.08817)
- **LoRA**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **MetaWorld**: [arxiv.org/abs/1910.10897](https://arxiv.org/abs/1910.10897)
- **InstructGPT (RLHF)**: [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- **Decision Transformer**: [arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
- **Diffusion Policy**: [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)
- **Set Transformer**: [arxiv.org/abs/1810.00825](https://arxiv.org/abs/1810.00825)
- **Stabilizing Transformers for RL**: [arxiv.org/abs/1910.06764](https://arxiv.org/abs/1910.06764)
- **RoboCat**: [arxiv.org/abs/2306.11706](https://arxiv.org/abs/2306.11706)
- **Open X-Embodiment**: [arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)
- **Eureka (VLM as reward designer)**: [arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)

---

## 6. 总结

这篇paper的核心贡献是**把LLM社区RLHF的pipeline适配到robotics的long-horizon online RL**。关键insight是：large pretrained model对RL gradient很脆弱，但我们可以用iterative SL来"修复"RL带来的representation drift。这个decoupling思路很general，我觉得可以推广到其他large model + online learning的场景，比如game playing with LLM agent、tool use with long-horizon feedback等。

工程上的take-away：**frozen backbone + trainable head for RL, full model for SL**——这个pattern在resource constrained场景下很实用。

如果你对某个细节想深入聊，比如SACfD在latent space的具体实现、token learner的attention机制、或者real-world RL的sample efficiency trick，我可以继续展开。
