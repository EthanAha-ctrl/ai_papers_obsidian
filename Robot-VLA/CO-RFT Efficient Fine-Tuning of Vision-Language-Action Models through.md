---
source_pdf: CO-RFT Efficient Fine-Tuning of Vision-Language-Action Models through.pdf
paper_sha256: 2a4f078bd069ea95aecb2943feca1d05ab3f57b9978499295afeea46c6b633df
processed_at: '2026-08-18T03:35:19-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说CO-RFT

## 1. 一句话概括

**用30个人类演示，通过offline RL，比用100个演示做behavior cloning效果还要好。**

## 2. 问题在哪

现在的VLA模型（像OpenVLA、π0这些）训练方式很"笨"——纯抄作业。你给它100条人类操作的录像，它学着在同样的场景下复现同样的动作。这种"behavior cloning"（BC）有几个致命问题：

**第一个问题：吃数据**。要它学会一个新任务，通常100条以上demo才够。30条根本不行——它只能学到个皮毛，成功率很低。

**第二个问题：换个位置就懵**。你训练时杯子放左边，测试时放右边，它就抓不到。因为它学的是"在这个位置往左伸手"，而不是"杯子的概念"。

**第三个问题：不会纠错**。BC学到的是"专家怎么做我就怎么做"。但机器人执行时手抖了一下，state偏了，BC就不知道怎么办——它的训练数据里没有"偏离后怎么回来"的样本。

RL理论上能解决这三个问题。RL不是抄动作，而是学"这个动作值多少分"。Q-learning能从suboptimal数据中识别出哪些决策是关键的、哪些是冗余的，这种prioritization能力是BC没有的。

## 3. 但RL和VLA天生不兼容

这里有个被多数paper忽略的根本矛盾。

VLA模型有个特性叫**action chunking**——它一次预测一串动作（比如10步），而不是一步。原因很现实：人操作的时候不是"看一眼当前画面就决定下一步"，而是有一连串计划好的动作；而且VLM forward一次很贵，能10步调用一次就别每步调用。

但标准RL的1-step TD learning假设每一步都是独立决策：

> 我在state $s_t$ 做了action $a_t$，得到reward $r_t$，到了 $s_{t+1}$，然后在 $s_{t+1}$ 我会重新决策选 $a_{t+1}$。

这个假设在chunking下完全错了——chunk内的 $a_{t+1}$ 不是在 $s_{t+1}$ 处新决策的，而是 $h$ 步前就定好了的。Bellman equation的递归性被破坏。

之前的工作要么装作没看见这个问题，要么干脆不用value-based RL（改用preference alignment）。CO-RFT正视了这个问题。

## 4. CO-RFT的核心idea

把TD learning从"1步return"扩展到"h步return"。

标准1-step TD：
> $Q(s_t, a_t) = r_t + \gamma Q(s_{t+1}, a_{t+1})$

value每一步往前传一格。如果reward在100步外，需要100次backup才能传回来。sparse reward下这几乎学不动。

CO-RFT的chunked TD：
> $Q(s_t, a_{t:t+h}) = \sum_{t'=t}^{t+h-1} \gamma^{t'-t} r_{t'} + \gamma^h Q(s_{t+h}, a_{t+h:t+2h})$

把整个action chunk当一个"大动作"看。reward在这个chunk内累加，bootstrap直接跳到下一个chunk起点。value传播速度提升h倍。

直觉：原本一帧一帧往前走，现在"快进"播放，h步并成1步。30条demo的有限数据里，value信号能更快地传到critical decision那里。

## 5. Critic网络怎么实现

最naive的做法：训h个独立critic，每个对应不同长度的action prefix。计算太贵。

CO-RFT用一个Transformer解决。输入state + h个action tokens，加causal mask，一次forward输出h个Q-values，分别对应 $Q(s_t, a_t)$、$Q(s_t, a_{t:t+1})$、...、$Q(s_t, a_{t:t+h})$。

causal mask保证第i个Q-value只能看到前i个action，符合prefix结构。这种设计很高效，一次forward就拿到multi-scale的Q估计。

paper还要求critic不仅预测整个chunk的Q，还要预测所有prefix chunk的Q（从1步到h步的return都要会）。这种multi-scale监督让value learning更稳定——类似图像分割中的deep supervision。

## 6. 两阶段训练

**Stage 1: Behavior Cloning**。先用30个demo做BC，full-parameter fine-tune。目的是把VLM backbone和action head从gripper机器人迁移到当前的灵巧手。这个stage结束后policy能勉强干活，成功率低但不会乱动。

**Stage 2: Offline RL**。用同一个30条demo的dataset跑offline RL（基于CalQL），进一步提升。这一步不收集新数据，只在静态dataset上优化。

为什么选CalQL而不是标准offline RL？因为CalQL有个calibration机制——对OOD actions做保守估计，但不会压得太低（有 $V^\mu$ 作为floor）。作者意外发现这个机制对sparse reward特别有效，即使不做online fine-tuning也能受益。

## 7. 一个小但实用的trick

sparse reward是real-world robotic RL的老大难。30条demo大概600 transitions，reward信号只在成功步骤附近，可能只有60个positive samples。

**Reward Upsampling**：teleoperation收集demo时，对成功步骤多记录几次（比如reward=1的step记录5次）。demo数量不变（还是30条），但dataset里reward信号密度提升5倍。

这个trick很trivial，但offline RL没法通过exploration弥补sparse reward，必须在data collection阶段就处理。

## 8. 实验结果

**In-distribution**：
- 比SFT平均提升57%成功率
- 4/6任务接近100%成功率
- 平均cycle time减少22.3%（学到的policy比专家演示更高效）

**Out-of-distribution**（物体放新位置）：
- Random初始化数据训练的CO-RFT：OOD drop只有10-15%
- Fixed初始化数据训练的CO-RFT：OOD drop平均55.3%
- 说明数据多样性对RL的generalization至关重要

这个OOD结果其实挺反直觉的。offline RL理论上没有online exploration，为什么能generalize？

我的理解：BC学的是trajectory pattern，OOD state下没有prior信息。Offline RL学的是value function，只要OOD state和training states在feature space相近，Q-function的function approximation能给出reasonable estimate。而且CalQL的保守性让policy不去赌博unknown regions，反而更稳。

## 9. Limitations

- 没办法判断optimal checkpoint（overtraining问题）
- 用的是deterministic policy（TD3 base），但灵巧操作往往有multiple optimal ways（左手抓 vs 右手抓），确定性policy表达不了multimodal分布。Future work想用diffusion policy解决
- 只测了6个简单task，没long-horizon
- 没和ConRFT等直接对比，只和SFT比

## 10. 我的takeaway

这篇paper给我的启发：

1. **Action chunking和RL的兼容性是个被低估的问题**。VLA community里用RL的工作不少，但认真处理这个矛盾的没几个
2. **SMDP视角的TD learning**其实是正确的数学框架——chunk就是个macro-action，return在这个macro内累加
3. **30 demo + offline RL > 100 demo + BC**这个结论有实际意义。如果成立，意味着collect data的cost可以降低3-5倍
4. **数据多样性 > 数据数量**这个RL的老lesson在VLA fine-tuning上同样成立

一个我没想明白的点：paper说CO-RFT的cycle time比expert demonstrations更短。Offline RL理论上不能超过data中最好的trajectory，为什么能比expert更快？可能是Q-learning学到了"哪些步骤是冗余的可以跳过"，但具体机制paper没讲清楚。这是个值得深挖的点。

---

# CO-RFT: Chunked Offline Reinforcement Fine-Tuning for VLA Models

让我从paper的核心动机讲起，build up the intuition for why this work matters.

## 1. 大图景：VLA models的fine-tuning困境

当前VLA models（如OpenVLA, π0, RT-2, GR-2）的训练范式主要是 **behavior cloning (BC) + 大规模demonstration datasets**。这种范式有几个本质问题：

- **Sample inefficiency**：要让VLA学会一个新task，通常需要100+ demonstrations，因为BC是"复制专家"而非"理解task"
- **OOD failure**：BC会overfit到demonstrations的specific trajectories，对未见过的initial states束手无策
- **No corrective behavior**：BC只能学到"专家做过什么"，学不到"如果state偏了怎么办"

RL理论上能解决这些问题，因为Q-learning本质是 **value-based prioritization**——它能识别哪些decisions是critical的，哪些是可替代的。这种capability在LLM的RLHF中已经被验证（preference learning > imitation learning）。paper的core hypothesis就是：**Offline RL can surpass BC with the same dataset**，并且能在real-world dexterous manipulation上验证这个假设。

## 2. Action Chunking与RL的根本冲突

这里需要深入思考一下action chunking的本质。

VLA models（ACT, OpenVLA, π0）都预测一个action chunk $a_{t:t+h} = [a_t, a_{t+1}, ..., a_{t+h-1}]$ 而非single action $a_t$。原因有三个：

1. **Non-Markovian demonstration**：人类teleoperation时，下一步action往往依赖history而非当前state。chunking让policy在chunk内closed-loop free，绕开non-Markovian问题
2. **Smoothness**：chunk内actions是连贯的，避免high-frequency jitter
3. **Inference efficiency**：每h步才调用一次expensive VLM forward

但标准RL（1-step TD）与chunking根本不兼容：
$$G_t(s_t, a_t) \gets r_t + \gamma Q(s_{t+1}, a_{t+1})$$

这里假设 $a_{t+1}$ 是新policy在 $s_{t+1}$ 处sample的，但实际上chunking中 $a_{t+1}$ 是上一chunk就已经决定了。这就破坏了 **Bellman consistency**——Q-value的backups假设每个action都是新决策，但chunking中chunk内的actions是"已经决定好的"。

之前的工作（ReinboT, ConRFT, iRe-VLA）几乎都忽略了这个问题，要么强行用1-step TD，要么用preference alignment绕过value learning。

## 3. Chunked RL的数学细节

paper提出的核心创新是把TD learning extend到action chunk上。让我详细拆解。

### 3.1 Notation

记 $a_{t:t+h} \in \mathbb{R}^{Ah}$ 表示h个连续actions的concatenation，$A$是action dimension。

**Policy**: $\pi_\psi(a_{t:t+h}|s_t) := \pi_\psi(a_t, a_{t+1}, ..., a_{t+h-1}|s_t)$

**Critic**: $Q_\theta(s_t, a_{t:t+h}) := Q_\theta(s_t, a_t, a_{t+1}, ..., a_{t+h-1})$

注意critic的输入维度从 $|S| + |A|$ 变成 $|S| + h|A|$，把整个chunk作为"一个macro-action"评估。

### 3.2 Chunked TD Target

公式(5)是paper的核心：
$$G_t^{(h)}(s_t, a_{t:t+h}) \gets \sum_{t'=t}^{t+h-1} [\gamma^{t'-t} r_{t'}] + \gamma^h Q(s_{t+h}, a_{t+h:t+2h})$$

变量解释：
- $t'$: 累加索引，从 $t$ 到 $t+h-1$，覆盖整个chunk
- $\gamma^{t'-t}$: discount factor，相对当前step $t$ 的指数权重。当 $t'=t$ 时为 $\gamma^0=1$，$t'=t+h-1$ 时为 $\gamma^{h-1}$
- $r_{t'}$: 第 $t'$ 步的reward
- $\gamma^h$: bootstrap的discount，h步后才用target network
- $Q(s_{t+h}, a_{t+h:t+2h})$: target network在下一个chunk起始处的Q-value，其中 $a_{t+h:t+2h} \sim \pi_\psi(\cdot|s_{t+h})$

这个公式的物理含义是：**把action chunk视作一个macro-action，其return是这个chunk内所有discounted rewards之和 + 下一个macro-action的value**。本质上这是 **Semi-Markov Decision Process (SMDP)** 的Bellman equation。

直觉上，这让value propagation速度提升h倍——single-step TD需要h次backup才能把 $s_{t+h}$ 的信息传到 $s_t$，而chunked TD一次就完成。这对sparse reward场景是game changer。

### 3.3 N-step Critic Loss

公式(6)更有意思：
$$L(\theta) = \frac{1}{Nh} \sum_{k=1}^{N} \sum_{i=1}^{h} \Big( Q_\theta(s_t, a_{t:t+i}) - G_t^{(h)}(s_t, a_{t:t+i}) \Big)^2$$

- $N$: N-step return的最大step数（hyperparameter，paper中应该和h相近）
- $h$: action chunk size
- $k$: N-step索引（outer sum）
- $i$: chunk内action的prefix长度（inner sum，从1到h）

关键insight：critic不仅要预测整个chunk的Q，还要预测**每个prefix chunk** $a_{t:t+i}$（$i=1, ..., h$）的Q。这意味着critic学到的Q是hierarchical的——从单步到h步的return都能预测。这种multi-scale supervision对value learning稳定性很有帮助，类似deep supervision在image segmentation中的作用。

### 3.4 Chunked Critic Network架构

Figure 2描述的架构很关键。Naively，需要训练h个separate critic networks对应h个不同长度的prefix，计算成本太高。paper用一个Transformer解决：

**输入**: state embedding + action chunk的embeddings（h个action tokens）
**架构**: self-attention with causal mask
**输出**: $H+1$ embeddings，后 $H$ 个对应h个Q-values

causal mask确保 $Q(s_t, a_{t:t+i})$ 只能attend到前 $i$ 个action tokens，符合prefix结构。这种设计等价于一个 **autoregressive value estimator**，一次forward就得到所有prefix的Q-value，效率极高。

这种架构让我想起Perceiver / ViT中的class token机制，只不过这里多个"query positions"分别对应不同长度的prefix value。

## 4. CO-RFT两阶段训练

### Stage 1: Behavior Cloning (Full Fine-tuning)

公式(7):
$$\mathcal{L}^{BC} = \mathbb{E}_{(s^e, a^e) \sim \tau^e} \|a^e - \pi^{BC}(s^e)\|^2$$

- $\tau^e$: expert demonstrations集合
- $s^e, a^e$: 某条demonstration中的state和action
- $\pi^{BC}$: 学习的behavior cloning policy

这一步用**full-parameter fine-tuning**而非LoRA，目的是：
1. Transfer VLM backbone（Kosmos-2，原pretrained on gripper data）到当前workspace的视觉distribution
2. Retrain action head适配dexterous hand embodiment（6-DoF灵巧手 vs 标准gripper）
3. 提供一个decent initial policy让后续RL不用从random开始

30 demos足以让BC policy达到"勉强能完成任务但成功率低"的水平——这就是offline RL的起点。

### Stage 2: Chunked Offline RL (CalQL-based)

公式(8)是critic loss（基于CalQL）：
$$L(\theta) = \mathbb{E}_{s,a,s' \sim \mathcal{D}} [(Q_\theta(s, a_{t:t+i}) - B^\pi \bar{Q}(s, a_{t:t+i}))^2] + \alpha(\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi}[\max(Q_\theta(s, a_{t:t+i}), V^\mu(s))] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q_\theta(s, a_{t:t+i})])$$

- $B^\pi \bar{Q}$: Bellman backup operator applied to target network $\bar{Q}$
- $\bar{Q}$: target Q-network（EMA of $\theta$）
- $\alpha$: conservative regularizer权重
- $V^\mu(s)$: 来自behavior policy的value estimate，提供floor防止over-pessimism
- $\max(Q_\theta, V^\mu)$: CalQL的关键创新，确保对OOD actions的penalty不超过 $V^\mu$，避免Q被压得太低导致后续online fine-tuning unlearning

第一项是标准TD error，第二项是conservative regularizer。Regularizer的设计逻辑：
- Minimize $Q$ under learned policy $\pi$ → 压低OOD actions的Q
- Maximize $Q$ under dataset actions → 保持in-distribution actions的Q
- 两项联合让Q-function学到"正确"的value ranking

公式(9)是policy loss：
$$L(\psi) = -\frac{1}{Nh} \sum_{k=1}^{N} \sum_{i=1}^{h} Q_\theta(s_t, a_{t:t+i})$$

就是maximize所有prefix Q-values的mean。这里有个细节：取mean而不是max，是为了让policy平衡不同scale的Q（短chunk vs 长chunk），避免policy只为长chunk优化导致短chunk quality下降。

## 5. Practical Implementation Details

### Model Architecture
- Backbone: **Kosmos-2** (grounded multimodal LLM, https://arxiv.org/abs/2306.14824)
- Framework: **RoboVLMs** (https://arxiv.org/abs/2412.14058)
- History modeling: LSTM or GPT (大多数VLA忽略了history信息)
- Base RL algorithm: **TD3** (https://arxiv.org/abs/1802.09477) - deterministic policy actor-critic
- Deterministic policy可能是个limitation（作者在Conclusion中提到），因为dexterous manipulation往往需要multimodal action distribution

### Reward Upsampling
这是paper中一个简单但很practical的trick。Sparse reward的本质问题是：30 demos × ~20 steps/traj = 600 transitions，但reward信号只在success step附近，可能只有60-100个positive samples。

**Reward Upsampling的做法**：在teleoperation收集demonstration时，对成功步骤多记录几次（如reward=1的step重复记录5次）。这样：
- 不改变demonstration数量（仍是30条）
- 但dataset中reward signal密度提升
- 缓解Q-function在positive samples上的underfitting

这个trick看似trivial但effect显著，因为offline RL无法通过exploration弥补sparse reward，必须在data collection阶段就处理。

## 6. Experimental Setup细节

### Hardware
- **Realman单臂机器人**: 6-DoF robotic arm
- **Inspire灵巧手**: 6 active joints end effector
- **ZED2 stereo camera**: 110°(H) × 70°(V) FoV，装在头部
- 工作台：白色桌面，均匀光照（减少visual confounders）

### 6个Tasks (Figure 4)
1. Grasping a cup
2. Pinching a cube
3. Grasping a sanitizer
4. Holding a barcode scanner
5. Catching a loop
6. Retrieving a mug（需要grasp cup handle，fine-grained manipulation）

每个task用30个demos（fixed或random初始化），每个task 40次trial评估。

### Metrics
- **Success Rate (SR)**: 物体被抓起在空中
- **Cycle Time (CT)**: 成功完成task所需平均steps（越少越好）

## 7. Experimental Results深入分析

### 7.1 IND场景结果 (Figure 5)
- CO-RFT vs SFT: **+57% SR average**
- 4/6 tasks接近100% SR
- 3个SFT完全失败的tasks（cup, sanitizer, mug），CO-RFT也能学到合理performance
- 例外：sanitizer (36%) 和 mug (30%) 仍然较低
  - Sanitizer: 颜色与白色背景相似 → visual识别困难
  - Mug: 需要精细grasp handle → 动作精度要求高

这个结果验证了paper的core hypothesis：**同样的30 demos，offline RL能比BC学到明显更好的policy**。原因是Q-learning能从suboptimal或redundant transitions中区分出critical decisions，而BC对所有expert actions一视同仁。

### 7.2 Cycle Time (Figure 6)
- CO-RFT: **-22.3% CT average**
- 说明CO-RFT学到的policy比expert demonstrations更高效
- 这其实是个surprising结果——offline RL理论上不能超过data中最好的trajectory，但实践中能通过Q-value学到更短的critical path
- 例外：Pinching a cube task没有CT改善，作者归因于fixed initialization导致data diversity不足，value function学不到expressive enough的critical path

### 7.3 OOD Generalization (Figure 7, Table 1)
这是paper最impressive的结果：

**Random-initialized datasets**:
| Task | IND | OOD | Drop |
|------|-----|-----|------|
| Cup | 100% | 90% | -10% |
| Scanner | 95% | 80% | -15% |
| Mug | 30% | 20% | -10% |
| Avg | 75% | 63.3% | -11.7% |

**Fixed-initialized datasets**:
| Task | IND | OOD | Drop |
|------|-----|-----|------|
| Cube | 100% | 0% | -100% |
| Loopy | 100% | 50% | -50% |
| Sanitizer | 36% | 20% | -16% |
| Avg | 78.7% | 23.3% | -55.3% |

Key insight: **数据多样性对OOD generalization影响巨大**（11.7% vs 55.3% drop）。

为什么？因为Q-function是 **state-action value的function**，如果training data覆盖state space少（fixed init），Q在OOD states上extrapolate会非常noisy。Random init让Q在更广state分布上学习，泛化能力自然强。这从RL的角度很容易理解，但BC community往往忽视了这一点——BC可以overfit到fixed trajectory，但Q-learning必须有state coverage。

### 7.4 为什么Offline RL能generalize而BC不能？

paper没有详细讨论，但这是个值得思考的问题。我的理解：

1. **BC学到的是"trajectory pattern"**：BC的loss是逐step模仿，会让policy在expert trajectory上"memorize"。OOD state出现时，policy没有任何"prior"信息。

2. **Offline RL学到的是"value function"**：Q-function是state-action到scalar value的mapping，即使OOD state，只要它和training states在feature space相近，Q仍能给出reasonable estimate（function approximation的inductive bias）。Policy通过maximize Q generalize到OOD。

3. **CalQL的calibration机制**：保守估计OOD actions的Q，让policy不会去"赌博"unknown regions。这种保守性反而让generalization更稳定。

4. **Chunked RL的effect**：value propagation加速h倍，相同数据量下学到的Q更accurate，generalization更好。

## 8. 关联工作和更广的intuition

### 8.1 Cal-QL的背景
Cal-QL (https://arxiv.org/abs/2303.05479) 是Nakamoto et al. 2024的工作，原设计目的是让offline RL pre-training后能高效online fine-tuning。CO-RFT作者surprising地发现：即使不做online fine-tuning，CalQL的calibration本身就能解决sparse reward问题。这是个non-trivial的finding——calibration not only helps online transition, but also helps value learning stability under sparse rewards.

### 8.2 Action Chunking的起源
ACT (Action Chunking with Transformers, Zhao et al. 2023, https://tonyzhaozh.github.io/aloha/) 是action chunking在robotic manipulation中的开山之作。原始motivation是解决human demonstration的non-Markovian特性（人操作时基于history，但policy只看当前state）。

CO-RFT把action chunking从IL extend到RL，是个non-trivial step。相关同期工作：
- Tian et al. 2025 "Chunking the Critic" (https://arxiv.org/abs/2503.03660): Transformer-based SAC with N-step returns
- Li, Zhou, Levine 2025 "RL with Action Chunking" (https://arxiv.org/abs/2507.07969)
- Seo & Abbeel 2025 "RL with Action Sequence"

### 8.3 VLA + RL的前沿工作对比

| Method | RL Type | Action Chunk | Online/Offline | Limit |
|--------|---------|--------------|----------------|-------|
| ReinboT (Zhang 2025a) | Offline RL | ❌ | Offline | 不兼容chunking |
| ConRFT (Chen 2025) | Offline→Online | ❌ | Both | 需online基础设施 |
| iRe-VLA (Guo 2025) | Online RL | ❌ | Online | 需要IL-RL iteration |
| TPO (Zhang 2024) | Preference | ❌ | Offline | Reward信号弱 |
| Hume (2025) | Test-time | ❌ | Test-time | Inference变慢 |
| **CO-RFT** | **Offline RL** | **✅** | **Offline** | **确定性policy** |

CO-RFT是第一个真正把action chunking和offline RL结合的VLA fine-tuning method。

### 8.4 与LLM RLHF的类比
Karpathy你应该很熟悉这个类比。LLM中：
- SFT ~ BC：模仿expert demonstrations
- RLHF (preference learning) ~ Offline RL：从preference信号学到better policy
- DPO ~ direct policy optimization without explicit Q

VLA中：
- BC ~ SFT
- CalQL-based offline RL ~ RLHF with explicit value function
- TPO (Zhang 2024) ~ DPO analogue

LLM中已经验证RLHF能超过SFT在OOD上，CO-RFT在robotics中验证同样的现象。

## 9. Limitations和未来方向

paper自述的两个limitations：

1. **Overtraining issue**: 没有明确的方法判定optimal checkpoint。这可能是因为offline RL training dynamics中Q-function先improve后overfit到dataset biases。需要类似early stopping或value-based checkpoint selection的technique。

2. **Deterministic policy**: TD3 base给的是deterministic action，但dexterous manipulation往往有multimodal optimal actions（如可以left-grasp或right-grasp）。Future work提到用diffusion-based policy（如Diffusion Policy, https://diffusion-policy.cs.columbia.edu/）。

### 我觉得还有几个未讨论的limitations：
3. **Task complexity**: 6个tasks都是single-object manipulation，没有long-horizon或multi-stage tasks
4. **Vision diversity**: 单一白色桌面+均匀光照，对visual generalization要求低
5. **Reward design**: success-based binary reward，没有dense reward或preference
6. **Sample size**: 30 demos对BC不够，但对industrial deployment仍可能过多
7. **Critic capacity**: 用Transformer critic但没讨论参数量、训练时间、和BC policy的computational overhead对比

## 10. 整体评价

这篇paper做对了几件事：

1. **Identify了真正的问题**：action chunking与RL的兼容性，这是被prior work广泛忽略的
2. **Mathematical formulation清晰**：Chunked TD + multi-scale Q supervision，基于SMDP的理论
3. **Real-world验证**：在dexterous manipulation上做实验，不是simulator toy tasks
4. **Simple and effective**：Reward Upsampling这种"小trick"也implement并discuss

但也有一些不足：

1. **没有和ConRFT等直接对比**：只和SFT比，缺少apples-to-apples的offline RL comparison
2. **没有ablation on N and h**：chunk size和N-step的影响没系统study
3. **Reward Upsampling的quantitative effect没单独report**
4. **Critic网络的具体架构图（Figure 2）信息不够**：embedding dimension, layer数, attention head数都未specify

总体上，这篇paper是VLA + Offline RL方向上一个solid contribution，尤其是action chunking与TD learning的theoretical bridge部分。对实际deployment，CO-RFT证明了：**30 demos + offline RL > 100 demos + BC**，这对data-efficient robotic learning有实际意义。

## References

- CO-RFT paper (本paper)
- Cal-QL: https://arxiv.org/abs/2303.05479
- ConRFT: https://arxiv.org/abs/2502.05450
- ReinboT: https://arxiv.org/abs/2501.16664 (similar paper)
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboVLMs: https://arxiv.org/abs/2412.14058
- Kosmos-2: https://arxiv.org/abs/2306.14824
- ACT (Action Chunking Transformers): https://tonyzhaozh.github.io/aloha/
- TD3: https://arxiv.org/abs/1802.09477
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Chunking the Critic (Tian et al.): https://arxiv.org/abs/2503.03660
- RL with Action Chunking (Li, Zhou, Levine): https://arxiv.org/abs/2507.07969
- π0: https://arxiv.org/abs/2410.24164
- GR-2: https://arxiv.org/abs/2410.06158
- iRe-VLA: https://arxiv.org/abs/2501.16664
- TPO/GRAPE: https://arxiv.org/abs/2411.19309
- Hume (test-time thinking for VLA): https://arxiv.org/abs/2505.21432
- Steering VLA via Value Guidance: https://arxiv.org/abs/2410.03680 (Nakamoto 2024a)
