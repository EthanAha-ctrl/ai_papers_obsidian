---
source_pdf: CReF Cross-modal and Recurrent Fusion for Depth-conditioned.pdf
paper_sha256: dfb49fe0c9b95479480a48e6c6b84b81de32faf4622c2ab2c61ed08d2cf66aa0
processed_at: '2026-08-03T17:50:11-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CReF 用人话讲一遍

## 先想象一个场景

你是一个humanoid robot，站在一段楼梯顶端准备下楼。你的眼睛是一个depth camera，装在head位置往前下方看。问题来了——**你往下看的视野被自己的body挡住了**，你看不到即将踩下去的那个step在哪里。

这就是humanoid locomotion最核心的痛点：**当你最需要看脚下的时候，你偏偏看不到脚下**。

这篇paper就是来解决这个问题的。

## 之前的人怎么解决？

### 方法一：先建个map，再用map来控制

Early perceptive locomotion的做法是：先用depth建一个elevation map（就是一个2.5D heightfield，记录每个位置有多高），然后policy看着这个map来决定怎么走。

听起来很合理，compact、interpretable。但有个致命问题——**elevation map假设每个位置只有一个height值**。

那么遇到这些情况就傻了：
- **Hollow pallet**（带孔的托盘）：上面一层板能踩，下面也有空间，但elevation map只能记录一个值
- **Handrail**（扶手）：垂直结构，elevation map表达不了
- **Overhang**（悬垂结构）：上方有突出的东西，下方还能走

elevation map的本质是一种**inductive bias**——你提前假设terrain能用2.5D表达，这个假设本身就是limitation。你把这个map当作perception和control之间的interface，performance就永远被这个interface的expressiveness卡住了。

### 方法二：用depth但加auxiliary supervision

后来有人想：那我不建map了，直接用depth end-to-end。但是depth encoder是个blank slate，怎么保证它学到的是locomotion-relevant的东西？

于是加auxiliary supervision：
- 让encoder reconstruct terrain height map
- 让encoder学一个privileged teacher的latent representation
- 用decoder-constrained latent estimation

这些方法work，但又掉进同一个坑——**你用height map reconstruction作为supervision target，encoder就被迫保留支持height map的信息**。Terrain structure超出height map表达范围的部分，自然被deemphasized。

这就是paper里那句insight的来源：

> "The blind spots of the intermediate are not removed, but inherited through supervision."

intermediate representation的盲区，通过supervision悄悄传递给了encoder。你以为在end-to-end learning，实际上你只是换了个方式继承2.5D representation的局限。

### 方法三：直接end-to-end，让reward自己组织

CReF的思路：**让RL的reward signal自己决定depth encoder该学什么**。不加auxiliary supervision，不建explicit map，直接让locomotion objective来organize visual representation。

听起来simple，但带来一个challenge：depth只看前方，单帧depth信息不足以disambiguate terrain structure和contact timing。怎么办？

这就是CReF三个核心设计的motivation。

## CReF的三个核心trick

### Trick 1: Cross-modal Attention —— "用身体感觉去问眼睛"

**场景类比**：你站在楼梯上，body告诉proprioceptive sensor："我现在处于swing phase，马上要touchdown了"。这个proprioceptive signal相当于一个query，去"询问"depth encoder："前方哪个region是我应该关注的landing zone？"

数学上，就是把proprioception编码成一个token，作为attention的Query；把depth image切成一组tokens，作为Keys和Values。然后用MHA算attention。

$$\bar{\mathbf{e}}_t^d = \text{MHA}(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t)$$

这里 $\mathbf{Q}_t$ 来自proprioception token，$\mathbf{K}_t, \mathbf{V}_t$ 来自depth tokens。

**Key intuition**：depth feature extraction变成state-dependent了。不同locomotion状态下，policy会attend到depth image的不同区域。

Ablation实验证明这是最critical的component。去掉cross-modal attention，overall掉12分，OOD terrain掉得最多。因为没有了state-conditioned feature extraction，depth encoder就退化成static feature extractor，无法adaptively关注task-relevant region。

### Trick 2: Gated Residual Fusion —— "有选择地融合"

Cross-modal attention输出一个depth-conditioned feature，proprioception token还在旁边。两者怎么merge？

Naive做法：concatenate后送MLP。

CReF做法：gated residual fusion。具体来说：

$$\mathbf{f}_t = \mathbf{x}_t + \mathbf{c}_t \odot \sigma(\mathbf{g}_t)$$

- $\mathbf{x}_t$：原input（proprioception + depth-conditioned feature的concat）
- $\mathbf{c}_t$：candidate residual update
- $\sigma(\mathbf{g}_t)$：channel-wise gate

**Intuition**：不是所有depth information对所有joint control都有用。比如前方远处terrain info对ankle control可能没用，但对hip control可能有用。channel-wise gating让network自己决定每个channel该不该被depth信息influence。

还有个设计细节——residual path（$\mathbf{x}_t + ...$）。这个direct path让gradient能直接回传到cross-modal attention输出，training更stable。如果只有gating没有residual，training容易collapse。

Ablation显示去掉GRF掉约7分，主要在hard terrain上。

### Trick 3: Highway Output Gate —— "什么时候信记忆，什么时候信当前"

Recurrent module（GRU）会aggregate temporal context。但有个问题：**何时该trust memory，何时该trust current observation？**

如果stable walking在平地上，current depth observation已经足够，强行integrate history可能引入noise。如果risky state比如pitch > 0.2 rad，需要更多temporal context来disambiguate terrain structure。

Highway gate就是用来做这个decision的：

$$\beta_t = \sigma(\mathbf{W}_\beta[\mathbf{z}_t^{rec}; \mathbf{f}_t] + \mathbf{b}_\beta)$$
$$\mathbf{y}_t = \beta_t \odot \mathbf{z}_t^{rec} + (1 - \beta_t) \odot \mathbf{f}_t$$

- $\beta_t \to 1$：trust recurrent memory
- $\beta_t \to 0$：trust feedforward feature

gate的input是 $[\mathbf{z}_t^{rec}; \mathbf{f}_t]$，既看memory又看current feature，然后decide该信谁。

**Empirical behavior**（从Fig. 6）：

| Condition | Gate value | Interpretation |
|-----------|------------|----------------|
| Step-like terrain vs Flat | Higher | 需要更多anticipation |
| Flight phase vs Support | Higher | 单帧不够，需要memory |
| Risky posture vs Stable | Higher | 紧急情况依赖history |

这验证了highway gate的设计意图——network确实学到了state-dependent temporal integration。

## 第四个trick：Terrain-Aware Foothold Reward

这个设计最精彩。让我用intuitive方式讲。

### 之前的做法：告诉policy"不许踩哪里"

BeamDojo的Foot Contact Quality Reward (FCQR)是prohibitive的——penalize bad contacts但不告诉policy应该往哪踩。

类比：你教小孩走楼梯，只说"不许踩空、不许踩边缘"，但不说"踩中间那块平的"。小孩需要自己explore出good footholds，sample efficiency低。

### CReF的做法：告诉policy"应该踩这里"

CReF从simulation terrain geometry提取supportable foothold candidates，然后reward touchdown close to这些candidates。

具体pipeline：
1. 每只脚维护一个local point cloud buffer
2. 根据commanded velocity过滤near-foot points，保留前方look-ahead region
3. 把remaining points partitioned成overlapping windows（24cm × 10cm，stride 4cm）
4. 每个window计算covariance matrix的特征值，判断是否够flat、够horizontal、够reachable
5. 满足条件的window作为foothold candidate
6. 在liftoff时refresh candidates，fix到下次touchdown
7. Touchdown时计算实际contact位置到nearest candidate的planar distance，reward = exp(-distance / scale)

**Key design choices**：

**为什么在liftoff时refresh而不是swing phase中实时更新？** 因为foot在swing phase中位置变化，如果candidates实时更新，target会不断变化，reward signal chattering，policy很难learn稳定的foot trajectory。Liftoff时fix，直到下次touchdown，让policy有stable target去aim。

**为什么这是anticipatory？** 虽然reward在touchdown时才计算，但policy需要learn在swing phase就把foot aim到supportable region。这就是forward planning。

**实验结果**：descending stairs的touchdown deviation从2.8cm降到1.4cm，ascending从3.0cm降到1.5cm。Ascending还eliminated ankle-riser collisions（脚踩到stair riser上）。

## 为什么下楼比上楼难？

Table III的数据：所有method都是descent failures远多于ascent failures。CReF在OOD stairs上ascent失败90次，descent失败454次。

**Physics reasons**：

1. **Visual occlusion**：下楼时下方step被自身遮挡，depth camera看不到即将踩的地方
2. **Gravity assist**：下楼时gravity推你走，控制margin小，稍有edge contact就slip
3. **Impact loading**：touchdown时冲击大，foot容易bounce
4. **Edge contact risk**：foot edge踩到step edge会catastrophic

CReF的foothold placement reward + recurrent fusion在descent上improvement最明显。因为即使看不到下方step，policy能从前方terrain geometry通过recurrent context推断foot该放在哪里。

## 为什么不需要synthetic depth corruption？

HPL和Hiking in the Wild用synthetic depth corruption（stereo artifacts, calibration noise, random holes等）来bridge sim-to-real gap。CReF完全不用，直接zero-shot transfer就work。

**可能原因**：

1. **Architecture-induced robustness**：Cross-modal attention让depth feature extraction是state-conditioned的，即使depth有noise，proprioception和temporal context能compensate
2. **No binding to artifact distribution**：如果你simulate specific artifact distribution，robustness只对那个distribution有效。CReF不assume任何artifact distribution，generalization更广
3. **Training simplicity**：不用调domain randomization的hyperparameter

这是一个design trade-off：explicit domain randomization vs architecture-induced robustness。CReF证明了后者也能work。

## Zero-shot Transfer到Real World

实验在AGIBOT X2 Ultra humanoid上做，1.31m高，39kg，peak torque 120 N·m。对比HPL的Unitree H1是1.8m高，47kg，knee torque 360 N·m。X2 Ultra小很多，torque也弱很多，但CReF仍能在上面实现strong performance。

Real-world scenes包括：
- Side railings stairs（20cm rise, 26cm tread）
- 40cm high platform
- 80cm gap
- Hollow pallet assemblies
- Severe reflective interference（depth大面积invalid）
- Outdoor dense vegetation clutter

特别impressive的是在**severe depth degradation**下仍work。深度camera遇到强反光会有大块invalid holes，传统方法可能直接fail，但CReF通过proprioception和recurrent context补偿visual information缺失。

## 三个Trick的Synergy

CReF的coherent design philosophy是：**让RL reward自己organize representation，而不是人为impose inductive bias**。

- **Cross-modal attention**: 让policy自己决定depth哪个region relevant
- **Gated Residual Fusion**: 让policy自己决定哪个modality channel该被emphasized
- **Highway Gate**: 让policy自己决定何时trust memory vs feedforward
- **Foothold Reward**: 提供directional supervision而不是prohibition

每个component都support同一个哲学，所以synergy强。

## Limitations

Paper承认：
- 依赖active depth sensing，对illumination和reflective surface敏感
- Depth alone不preserve texture和appearance cues

Future direction：binocular RGB-based sensing，jointly提供depth structure和texture info。

## 最终Intuition

CReF的story可以用一句话总结：

** humanoid robot在complex terrain上locomotion的难点在于visual information的partial observability。解决这个问题的key是让proprioception state主动guide perception，让temporal memory在需要时被recruited，让reward signal提供directional guidance，而不是人为impose geometric representation的inductive bias。**

这是一个elegant的framework，每个design choice都有clear motivation，ablation实验也support每个component的contribution。值得deeply understand的perceptive locomotion work。

**Project page**: https://cometlogic.github.io/cref/

**Reference**: Yuan Hao et al., "CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion"

**Related works**:
- [Humanoid Parkour Learning](https://arxiv.org/abs/2410.11817)
- [BeamDojo](https://roboticsconference.org/program/papers/79/)
- [Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822)
- [Highway Networks](https://arxiv.org/abs/1505.00387)
- [NVIDIA Warp](https://github.com/NVIDIA/warp)

---

# CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion 深度解析

## 1. 论文背景与动机

这篇paper来自AGIBOT团队，解决的是一个humanoid robot在complex terrain上的perceptive locomotion问题。核心矛盾在于：**如何让robot从raw depth直接学习locomotion-relevant features，而不依赖explicit geometric intermediates**。

### 1.1 现有方法的两大局限

**局限一：2.5D terrain representation的representational bias**

之前的perceptive locomotion方法（如elevation mapping）通过一个explicit 2.5D terrain map作为perception和control之间的interface。这个paradigm很吸引人——compact、interpretable、有efficient online elevation mapping支撑。但问题在于performance tightly coupled to 2.5D representation的fidelity和expressiveness。对于：

- **overhangs**（悬垂结构）
- **vertical clutter**（垂直杂物）
- **perforated obstacles**（穿孔障碍物，如hollow pallet）

2.5D elevation map本质上无法表达，因为elevation map假设每个position只有一个height value。

参考: [Elevation mapping for locomotion and navigation using GPU](https://ieeexplore.ieee.org/document/9811832), [Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822)

**局限二：Auxiliary supervision的inductive bias继承**

另一条line of work直接用depth，但用auxiliary targets来shape depth encoder学什么：

- terrain reconstruction [Song et al. 2025]
- decoder-constrained latent estimation [Luo et al. 2024]  
- privileged teacher distillation [Zhuang et al. 2024]

这些supervision策略看似有效，但存在一个subtle问题：**auxiliary target的inductive bias会被继承到learned representation中**。如果target是height map或geometry surrogate，encoder被rewarded primarily for preserving information that supports that target，scene structure outside its representational scope is naturally deemphasized。

用论文原话："the blind spots of the intermediate are not removed, but inherited through supervision"。

这是一个很深刻的insight——你以为在用end-to-end learning，实际上把2.5D representation的局限性通过supervision悄悄传给了encoder。

### 1.2 CReF的核心思路

CReF的核心设计哲学是：**let the locomotion objective alone organize the visual representation**。直接从raw forward-facing depth + proprioception end-to-end学习，让RL的reward signal自己决定depth encoder该提取什么。

这带来一个挑战：当只有一个depth frame时，如何处理partial observability？前向depth camera无法直接observe instantaneous underfoot region。论文的解决方案是：

1. **Cross-modal attention**：用proprioception query depth，实现state-conditioned feature extraction
2. **Recurrent fusion**：用GRU + highway gate处理temporal context
3. **Terrain-aware foothold reward**：提供directional supervision而不是prohibitive constraints

## 2. Policy Architecture详解

### 2.1 整体数据流

```
Proprioception o_t^p ──┐
                       ├──→ Proprioceptive Tokenizer P_φ ──→ e_t^p ──┐
Velocity est. v̂_t ────┘                                              │
                                                                      ├──→ Cross-Modal Attn ──→ ē_t^d
Depth D_t ──→ Depth Tokenizer T_θ ──→ Z_t ──→ LN ──→ E_t^d ──────────┘         │
                                                                                │
                          [e_t^p ; ē_t^d] = x_t ──────────────────────────────→ │
                                                                                ↓
                                                           Gated Residual Fusion (GRF)
                                                                                ↓
                                                                              f_t
                                                                                ↓
                                                         GRU + Highway Output Gate
                                                                                ↓
                                                                              y_t
                                                                                ↓
                                                                           MLP Head
                                                                                ↓
                                                                            a_t (action)
```

### 2.2 Proprioceptive Observation

$$\mathbf{o}_t^p = [\omega_t, \mathbf{r}_t^{grav}, \mathbf{u}_t^{cmd}, \mathbf{q}_t - \mathbf{q}_0, \dot{\mathbf{q}}_t, \mathbf{a}_{t-1}]$$

各变量含义：

| 变量 | 含义 | 维度（典型）|
|------|------|------|
| $\omega_t$ | base angular velocity（基座角速度）| 3 |
| $\mathbf{r}_t^{grav}$ | gravity direction in body frame | 3 |
| $\mathbf{u}_t^{cmd}$ | commanded motion（线速度+角速度指令）| 3-4 |
| $\mathbf{q}_t - \mathbf{q}_0$ | joint position offset from nominal pose | $n_{dof}$ |
| $\dot{\mathbf{q}}_t$ | joint velocities | $n_{dof}$ |
| $\mathbf{a}_{t-1}$ | previous action | $n_{dof}$ |

这里用 $\mathbf{q}_t - \mathbf{q}_0$ 而不是 $\mathbf{q}_t$ 是因为nominal pose subtraction让网络学习的是相对deviation，optimization landscape更友好。

### 2.3 Asymmetric Actor-Critic

Value network用privileged information：

$$\mathbf{s}_t = [\mathbf{o}_t^p, \mathbf{v}_t, \mathbf{m}_t]$$

- $\mathbf{v}_t$：ground-truth base linear velocity（sim only）
- $\mathbf{m}_t$：robot-centric terrain height observation（local elevation samples around robot）

这是standard asymmetric actor-critic pattern——critic训练时用privileged info获得更准确value estimate，actor部署时只用onboard sensors。参考: [Learning to walk in minutes using massively parallel deep reinforcement learning](https://proceedings.mlr.press/v164/rudin22a.html)

### 2.4 Action Space: Residual Parameterization

$$\mathbf{q}_t^{target} = \mathbf{q}_0 + \mathbf{a}_t$$

Action $\mathbf{a}_t \in \mathbb{R}^{n_a}$ parameterizes joint position targets，由low-level PD controller跟踪。

residual parameterization的好处：policy只需要学习相对nominal pose的deviation，learning效率更高，并且nominal pose提供了一个safe prior。

## 3. Cross-modal Attention: 核心创新一

### 3.1 为什么需要Cross-modal Attention？

传统做法是把proprioception和depth feature concatenate后送入MLP。这种做法的问题：depth features是static的，不会根据当前locomotion state调整关注点。

比如：当robot处于swing phase即将touchdown时，应该关注前方即将接触的region；当处于stable double support时，可能更关注远处的terrain planning信息。

Cross-modal attention让proprioception作为query去"询问"depth tokens，实现**state-conditioned depth feature extraction**。

### 3.2 数学细节

**Proprioceptive token:**
$$\mathbf{e}_t^p = \mathcal{P}_\phi([\mathbf{o}_t^p; \hat{\mathbf{v}}_t])$$

这里 $\hat{\mathbf{v}}_t$ 是auxiliary velocity estimator的输出（后面会讲），被concatenate进去是因为velocity信息对terrain anticipation很重要。

**Normalized depth tokens:**
$$\mathbf{E}_t^d = \text{LN}(\mathbf{Z}_t)$$

其中 $\mathbf{Z}_t = \mathcal{T}_\theta(\mathbf{D}_t) \in \mathbb{R}^{N \times d}$，$N$ 是token数量，$d$ 是token dimension。

**Attention computation:**
$$\mathbf{Q}_t = \text{LN}(\mathbf{e}_t^p)\mathbf{W}_q \in \mathbb{R}^{1 \times d}$$
$$\mathbf{K}_t = \mathbf{E}_t^d\mathbf{W}_k \in \mathbb{R}^{N \times d}$$
$$\mathbf{V}_t = \mathbf{E}_t^d\mathbf{W}_v \in \mathbb{R}^{N \times d}$$

$$\bar{\mathbf{e}}_t^d = \text{MHA}(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t)$$

**Fusion input:**
$$\mathbf{x}_t = [\mathbf{e}_t^p; \bar{\mathbf{e}}_t^d]$$

### 3.3 为什么是single query token？

注意 $\mathbf{Q}_t \in \mathbb{R}^{1 \times d}$，只有一个query token。这是一个很efficiency的设计：proprioception本身是一个"compact summary" of robot state，用single token query是合理的，避免N×N attention matrix带来的计算开销。

Multi-head attention (MHA)允许不同head关注不同spatial regions或不同semantic aspects的depth information。

### 3.4 Auxiliary Velocity Estimator

$$\hat{\mathbf{v}}_t = \text{LSTM}([\mathbf{o}_t^p; \mathcal{C}_\psi(\mathbf{D}_t)])$$

- $\mathcal{C}_\psi(\cdot)$：lightweight depth compression module
- supervised by ground-truth base linear velocity with $\ell_2$ loss

这个estimator的作用：proprioception只能提供reactive info，velocity estimator结合depth提供anticipatory info，让policy能更好地plan。

## 4. Gated Residual Fusion (GRF): 核心创新二

### 4.1 设计动机

proprioception token $\mathbf{e}_t^p$ 和 depth-conditioned token $\bar{\mathbf{e}}_t^d$ 拼接后如何融合？简单做法是MLP，但论文用了更精细的gated residual structure。

### 4.2 数学细节

**第一层 - shared projection:**
$$\tilde{\mathbf{x}}_t = \phi(\mathbf{W}_1\text{LN}(\mathbf{x}_t) + \mathbf{b}_1)$$

其中 $\phi(\cdot)$ 是ELU activation。这层在shared latent space中mix proprioceptive和depth-conditioned features。

**第二层 - content + gate分支:**
$$[\mathbf{c}_t; \mathbf{g}_t] = \mathbf{W}_2\tilde{\mathbf{x}}_t + \mathbf{b}_2$$

其中 $\mathbf{c}_t, \mathbf{g}_t \in \mathbb{R}^{2d}$，分别是content branch和gate branch。

**Gated output:**
$$\mathbf{f}_t = \mathbf{x}_t + \mathbf{c}_t \odot \sigma(\mathbf{g}_t)$$

- $\mathbf{c}_t$：candidate residual update
- $\sigma(\mathbf{g}_t)$：channel-wise gate，控制每个channel的contribution
- $\mathbf{x}_t + ...$：direct residual path，确保stable optimization

### 4.3 为什么这样设计？

这个设计有几个key insights：

1. **Residual path** ($\mathbf{x}_t + ...$): 让梯度能直接回传到cross-modal attention输出，训练更稳定
2. **Channel-wise gating** ($\sigma(\mathbf{g}_t)$): 不是所有depth information对所有joint control都有用，gating允许网络选择性融合
3. **ELU而非ReLU**: ELU在negative region有non-zero gradient，避免dead neurons

## 5. Recurrent Fusion with Highway Gate: 核心创新三

### 5.1 为什么需要Recurrent?

前向depth camera的partial observability问题：
- 看不到underfoot region
- 单帧depth无法disambiguate terrain structure
- contact timing需要temporal context

### 5.2 GRU temporal integration

$$\mathbf{h}_t = \text{GRU}(\mathbf{f}_t, \mathbf{h}_{t-1})$$
$$\mathbf{z}_t^{rec} = \mathbf{W}_h\mathbf{h}_t \in \mathbb{R}^{2d}$$

GRU的standard equations（回顾）:
- Update gate: $\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{f}_t; \mathbf{h}_{t-1}])$
- Reset gate: $\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{f}_t; \mathbf{h}_{t-1}])$  
- Candidate: $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}[\mathbf{f}_t; \mathbf{r}_t \odot \mathbf{h}_{t-1}])$
- Hidden: $\mathbf{h}_t = (1-\mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$

GRU aggregating short-horizon temporal context。

### 5.3 Highway Output Gate: 关键创新

如果只是把GRU输出直接送入MLP head，会有一个问题：**何时该信任recurrent memory，何时该信任current observation？**

论文的highway gate设计：

$$\beta_t = \sigma(\mathbf{W}_\beta[\mathbf{z}_t^{rec}; \mathbf{f}_t] + \mathbf{b}_\beta)$$
$$\mathbf{y}_t = \beta_t \odot \mathbf{z}_t^{rec} + (1 - \beta_t) \odot \mathbf{f}_t$$

- $\beta_t$：highway gate value，每个channel一个值
- 当 $\beta_t \to 1$：信任recurrent features
- 当 $\beta_t \to 0$：信任feedforward features
- gate的input是 $[\mathbf{z}_t^{rec}; \mathbf{f}_t]$，让gate能基于两者信息做决策

这个设计借鉴了Highway Networks ([Srivastava et al. 2015](https://arxiv.org/abs/1505.00387))和LSTM的output gate思想。

### 5.4 Highway Gate的empirical behavior

Fig. 6展示了gate在不同conditions下的behavior：

| Condition | Gate value | Interpretation |
|-----------|------------|----------------|
| Step-like terrain vs Flat | Higher | 需要更多temporal anticipation |
| Flight phase vs Stable support | Higher | 单帧不足，需要memory |
| Risky states (roll/pitch > 0.2 rad) vs Stable | Higher | 紧急情况下依赖history |

这验证了highway gate的设计意图：**state-dependent fusion between instantaneous multimodal evidence and recurrent memory**。

参考Highway Networks原理: [Highway Networks](https://arxiv.org/abs/1505.00387)

## 6. Terrain-Aware Foothold Placement Reward: 核心创新四

这是论文最精彩的reward设计，让我详细解析。

### 6.1 设计动机

传统foothold constraints是prohibitive的——penalize bad contacts但不告诉policy应该往哪里踩。论文想提供一个**directional supervision**：鼓励foot touchdown到supportable regions。

### 6.2 完整pipeline

**Step 1: Point-cloud buffer**

对每只脚，maintain一个local foot frame中的point-cloud samples buffer。这些samples来自哪些source呢？从上下文推断是simulator提供的terrain points around foot。

**Step 2: Command-conditioned forward gating**

过滤掉near-foot points。minimum retained forward distance = 当前forward command下0.5s覆盖的距离。

这个设计很巧妙：根据command动态调整look-ahead distance。如果commanded速度大，look更远；如果速度小，look更近。

**Step 3: Candidate windows**

Remaining points partitioned成overlapping windows：
- Size: $24cm \times 10cm$（大致foot size）
- Stride: $4cm$

$$\mathcal{P}_{t,k}^f = \{\mathbf{p}_{t,k,i}^f\}_{i=1}^{n_{t,k}^f}$$

其中 $k$ index window，$i$ index point within window，$n_{t,k}^f$ 是window $k$ 中的point数量。

**Step 4: Mean and Covariance**

$$\boldsymbol{\mu}_{t,k}^f = \frac{1}{n_{t,k}^f}\sum_{i=1}^{n_{t,k}^f}\mathbf{p}_{t,k,i}^f$$

$$\boldsymbol{\Sigma}_{t,k}^f = \frac{1}{\max(n_{t,k}^f - 1, 1)}\sum_{i=1}^{n_{t,k}^f}(\mathbf{p}_{t,k,i}^f - \boldsymbol{\mu}_{t,k}^f)(\mathbf{p}_{t,k,i}^f - \boldsymbol{\mu}_{t,k}^f)^\top$$

这里用 $\max(n_{t,k}^f - 1, 1)$ 避免除零，是unbiased covariance estimator。

**Step 5: Eigen-decomposition**

$$\boldsymbol{\Sigma}_{t,k}^f \mathbf{v}_{t,k,j}^f = \lambda_{t,k,j}^f \mathbf{v}_{t,k,j}^f$$

with $\lambda_{t,k,1}^f \leq \lambda_{t,k,2}^f \leq \lambda_{t,k,3}^f$

三个特征值和特征向量描述point cloud在三个principal axis上的spread。

**Step 6: Roughness metric**

$$\rho_{t,k}^f = \sqrt{\max(\lambda_{t,k,1}^f, 0)}$$

最小特征值的平方根。物理意义：point cloud最薄方向的"thickness"。如果points在一个plane上，最小特征值接近0，roughness接近0。

**Step 7: Acceptance criteria**

Window被接受为foothold candidate当且仅当：

$$\rho_{t,k}^f < r_{th} \quad \text{(足够平坦)}$$
$$|v_{t,k,1,z}^f| > \eta_{th} \quad \text{(最小特征向量z分量大，即normal direction接近垂直)}$$
$$\mu_{t,k,z}^f > h_{min} \quad \text{(不凹陷)}$$

这三个条件capture了foothold的三个核心要求：
1. 平坦性
2. 近似水平
3. 可达性

**Step 8: Candidate foothold position**

$$\mathbf{p}_{t,k}^{f,\star} = \boldsymbol{\mu}_{t,k}^f$$

取window的mean作为candidate位置。

**Step 9: Refresh策略**

Candidate set在**liftoff时刷新**，然后held fixed until next touchdown。这个设计避免chattering——如果在swing phase中实时更新candidates，foot位置变化会让candidates不断变化，导致reward signal不稳定。

**Step 10: Reward computation**

At touchdown，计算realized contact position到nearest candidate的planar distance：

$$d_{xz}^f = \min_k \|[\mathbf{p}_{x,t}^f] - [\mathbf{p}_{x,t,k}^{f,\star}]\|_2$$

注意 $[\cdot]$ 表示取x-z plane components（planar distance）。

$$r_{fh} = \sum_{f \in \mathcal{F}} I_{td}^f \exp\left(-d_{xz}^f / s_{xz}\right)$$

- $I_{td}^f$：touchdown event indicator
- $s_{xz}$：tolerance scale
- $\mathcal{F}$：feet set

### 6.3 为什么这是"anticipatory shaping"?

虽然reward在touchdown时才计算，但它通过reward signal鼓励swing phase的foot trajectory接近supportable regions，这是anticipatory的——policy需要learn forward planning。

### 6.4 与FCQR (BeamDojo)的对比

BeamDojo的Foot Contact Quality Reward (FCQR)是prohibitive的——penalize bad contacts但不提供directional guidance。

实验结果（Fig. 4）：
- **Ascending stairs**: median absolute deviation从3.0cm → 1.5cm，eliminated ankle-riser collisions
- **Descending stairs**: median absolute deviation从2.8cm → 1.4cm

整个distribution都变得更concentrated，不只是mean shift。

参考BeamDojo: [BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds](https://roboticsconference.org/program/papers/79/)

## 7. Reward Function全景分析

Table I列出了所有reward terms，让我分析几个关键设计：

### 7.1 Linear Velocity Tracking

$$r = \exp\left(-\frac{\mathbb{I}_s \|\mathbf{e}_v\|_1 + (1-\mathbb{I}_s)\|\mathbf{e}_v\|_2^2}{\sigma_v}\right)$$

- $\mathbf{e}_v = \mathbf{u}_{xy}^{cmd} - \mathbf{v}_{xy}$
- $\mathbb{I}_s = \mathbb{I}(\|\mathbf{u}_{xy}^{cmd}\|_2 < v_s)$：low-speed indicator

这个设计很有意思：**low-speed时用L1 norm，high-speed时用L2 norm squared**。

为什么？low-speed regime下，robot处于quasi-static状态，small velocity errors应该被sparse penalized（L1 encourages sparsity）；high-speed regime下，需要smooth gradient（L2 squared）。

参考: [Revisiting reward design and evaluation for robust humanoid standing and walking](https://arxiv.org/abs/2407.18135)

### 7.2 Feet Contact Shaping

$$r = \bar{\mathbb{I}}_{stand} + (1 - \bar{\mathbb{I}}_{stand}')\mathbb{I}_{single}^{0.2s}$$

- $\mathbb{I}_{stand} = \mathbb{I}(\|\mathbf{u}^{cmd}\|_2 < u_s)$
- $\mathbb{I}_{single}^{0.2s}$：single-foot contact within last 0.2s

这个reward鼓励：standstill时双脚contact，motion时交替单脚contact（walking gait）。

### 7.3 其他关键penalties

- **Foot slip**: $\sum_f \mathbb{I}(\|\mathbf{F}^f\| > 5) \|\mathbf{v}_{xyz}^f\|$
- **Foot impact vel**: $\sum_f [|v_z^f| - v_0]_+$
- **Stumble**: $\mathbb{I}(\exists f: \|\mathbf{F}_{xy}^f\| > 5|F_z^f|)$ — 检测横向force过大的"绊倒"

这些penalty保护hardware安全，特别是foot impact相关terms防止hardware damage。

## 8. Training Details

- **Simulator**: Isaac Gym
- **Parallel envs**: 4096
- **Control freq**: 50 Hz
- **Depth resolution**: 64×48, 20 Hz
- **GPU**: RTX 4090
- **Training time**: 30 hours, 20,000 iterations
- **Depth rendering**: NVIDIA Warp, capsule approximation + self-occlusion via ray-capsule intersection
- **No synthetic depth corruption**: 这是与HPL等方法的关键区别

参考NVIDIA Warp: [Warp: A High-performance Python Framework for GPU Simulation and Graphics](https://github.com/NVIDIA/warp)

## 9. Experimental Results深度分析

### 9.1 Architectural Ablation (Table II)

| Method | Stairs (OOD) | Gap (OOD) | Platform (OOD) | MuJoCo OOD | Overall |
|--------|--------------|-----------|-----------------|------------|---------|
| Full CReF | 72.75 | 44.70 | 84.35 | 100(20/20) | **90.45** |
| w/o Cross-Attn | 34.00 | 8.60 | 64.80 | 95(19/20) | 78.56 |
| w/o GRF | 49.10 | 29.95 | 68.55 | 95(19/20) | 83.78 |
| w/o Highway Gate | 50.20 | 17.20 | 73.10 | 100(20/20) | 83.29 |
| HPL | 27.40 | 20.85 | 55.15 | 5(1/20) | 74.57 |

**Key observations**:

1. **Cross-Attn是critical component**: 去掉后overall掉12分，OOD terrain掉最多。这验证了state-conditioned depth extraction的重要性。

2. **GRF和Highway Gate作用相当**: 都贡献约7分。GRF负责spatial fusion quality，Highway Gate负责temporal fusion adaptivity。

3. **HPL在MuJoCo OOD上collapse**: 1/20 success，而CReF variants都接近20/20。这说明CReF的representation generalization更强——因为没绑定到2.5D representation的inductive bias。

### 9.2 Stair Failures: Ascent vs Descent (Table III)

| Method | Easy Up/Down | Hard Up/Down | OOD Up/Down |
|--------|--------------|--------------|-------------|
| Full CReF | 1/2 | 5/49 | 90/454 |
| w/o Cross-Attn | 0/75 | 88/338 | 496/824 |
| HPL | 31/63 | 241/338 | 493/959 |

**Critical insight**: Descent failures >> Ascent failures across all methods。下楼比上楼难得多。

为什么？下楼时：
1. Foot需要precisely place在lower step上，但depth camera看不到下方
2. Touchdown timing critical，edge contact catastrophic
3. Weight transfer dynamics复杂

CReF的foothold placement reward + recurrent fusion在descent上改进最明显。

### 9.3 Foothold Distribution (Fig. 4)

| Setting | Method | Median abs deviation |
|---------|--------|---------------------|
| Ascent | FCQR baseline | 3.0 cm |
| Ascent | Foothold reward | 1.5 cm |
| Descent | FCQR baseline | 2.8 cm |
| Descent | Foothold reward | 1.4 cm |

不仅mean改善，整个distribution更concentrated。Ascending还消除了ankle-riser collisions（foot踩到riser上）。

### 9.4 Real-World Results (Table IV)

| Task | Success/Trials |
|------|----------------|
| Stairs (15cm/30cm) | 20/20 |
| Platform (40cm) | 20/20 |
| Gap (80cm) | 18/20 |
| OOD terrain | 19/20 |

**Zero-shot transfer**: 没有task-specific fine-tuning，直接从sim部署。Real-world scenes包括：
- Side railings stairs
- 40cm platform
- 80cm gap
- Handrails, hollow pallet, severe reflection, outdoor clutter

特别impressive的是在**severe depth degradation**下仍work——大块invalid depth holes from reflection, dense vegetation等OOD因素。

## 10. Hardware Setup

- **Robot**: AGIBOT X2 Ultra, 1.31m height, 39kg weight, 120 N·m peak torque
- **Sensor**: Intel RealSense D435i, forward-facing, downward pitch 50°
- **Compute**: NVIDIA Jetson AGX Orin
- **Latency handling**: Linear regression mapping camera time to system time, select frame with 20ms effective latency

对比HPL的Unitree H1: 1.8m height, 47kg, 360 N·m knee torque。X2 Ultra小很多，torque也弱很多，但CReF仍能在上面实现strong performance。

参考AGIBOT X2: [AGIBOT X2 Ultra](https://www.agibot.com/products/X2)

## 11. Key Insights与Intuition

### 11.1 为什么Cross-modal Attention比Concatenation好？

Concatenation是"proprioception和depth各自encode，最后拼接"。Cross-modal attention是"proprioception主动query depth"。

关键区别：**depth feature extraction是state-dependent的**。当robot在swing phase即将touchdown时，proprioception signal会让attention关注前方即将接触的region；在stable support时，可能关注远处planning info。

这是一种**active perception**——用robot state guide perception，而不是passive fusion。

### 11.2 为什么Highway Gate能improve performance？

如果没有Highway Gate，GRU输出直接解码action，问题是：
- 在stable states，current observation可能已经足够，但GRU仍会强行integrate temporal context，可能引入noise
- 在risky states，需要更多temporal context，但GRU的implicit gating可能不够explicit

Highway Gate提供了**explicit state-dependent control** over temporal vs feedforward contribution。从Fig. 6的empirical evidence看，gate确实学到了intended behavior。

### 11.3 为什么Terrain-Aware Foothold Reward有效？

传统prohibitive constraints的问题：告诉policy"不要踩哪里"，但不告诉"应该踩哪里"。Policy需要自己explore出good footholds，sample efficiency低。

CReF的reward提供directional supervision：
- 计算supportable candidates（基于geometry analysis）
- Reward touchdown close to candidates
- 这是**anticipatory shaping**——encourage forward planning

而且candidates在liftoff时refresh并fixed until touchdown，避免chattering，reward signal稳定。

### 11.4 为什么不需要synthetic depth corruption？

HPL和Hiking in the Wild等方法用synthetic depth corruption（stereo artifacts, calibration noise等）来bridge sim-to-real gap。CReF不用。

可能原因：
1. Cross-modal attention + recurrent fusion提供inherent robustness——即使depth有noise，proprioception和temporal context能compensate
2. 不bound到specific artifact distribution，generalization更广
3. Training更简单

这是一个design trade-off: explicit domain randomization vs architecture-induced robustness。

### 11.5 为什么2.5D representation是limitation？

Elevation map假设每个position一个height value。对于：

- **Hollow pallet**: 上表面和下表面都是supportable，但elevation map只能记录一个
- **Vertical wall**: 没有walkable surface，但elevation map会给出height
- **Overhang**: 上方有结构，下方有space，elevation map无法表达

CReF直接从depth学习，不强制2.5D assumption，能处理这些cases。从实验看，CReF在hollow pallet和handrails等场景下zero-shot transfer成功。

### 11.6 为什么下楼比上楼难？

从Table III看，descent failures远多于ascent。物理原因：

1. **Visual occlusion**: 下楼时下方step被自身遮挡，depth camera看不到
2. **Gravity assist**: 下楼时gravity推着你走，控制margin小
3. **Impact loading**: 下楼touchdown时impact更大
4. **Edge contact risk**: foot edge踩到step edge会slip

CReF的foothold placement reward在descent上improvement明显，因为它提供anticipatory guidance——即使看不到下方step，也能从前方terrain geometry推断foot该放在哪里。

## 12. Limitations与Future Work

论文承认的limitation：
- 依赖RealSense active depth，对illumination和reflective surface敏感
- Depth alone不preserve appearance和texture cues

Future direction: binocular RGB-based sensing，jointly提供depth structure和texture info。

## 13. 与相关工作对比

### 13.1 vs HPL (Humanoid Parkour Learning)

HPL用privileged teacher + distillation，teacher用height map等privileged info。CReF直接end-to-end。

- HPL: 2-stage（teacher + student），explicit geometric intermediates
- CReF: single-stage，no geometric intermediates

从Table II看，CReF在所有terrain categories上都outperform HPL，特别是OOD generalization。

参考HPL: [Humanoid Parkour Learning](https://arxiv.org/abs/2410.11817)

### 13.2 vs Hiking in the Wild

Hiking in the Wild也是depth-conditioned humanoid，但用了synthetic depth corruption和complex training design。CReF证明architecture innovation也能achieve robust transfer，不需要explicit domain randomization。

参考: [Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids](https://arxiv.org/abs/2601.07718)

### 13.3 vs BeamDojo

BeamDojo的FCQR是prohibitive constraint。CReF的foothold placement reward是directional supervision，提供更好的guidance。

参考: [BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds](https://roboticsconference.org/program/papers/79/)

### 13.4 vs START

START用terrain reconstruction作为auxiliary supervision。CReF不用auxiliary targets，让reward signal自己organize representation。

参考: [START: Traversing Sparse Footholds With Terrain Reconstruction](https://ieeexplore.ieee.org/document/10849842)

## 14. 项目资源

- **Project page**: https://cometlogic.github.io/cref/
- **GitHub**: 建议查看project page获取最新code release
- **Reference paper**: CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion

## 15. 总结：CReF的设计哲学

CReF的core philosophy：**让RL reward signal自己决定该学什么，而不是人为impose inductive bias**。

三个architectural innovations都serve这个philosophy：

1. **Cross-modal attention**: proprioception query depth，让policy自己决定depth哪个region relevant
2. **Gated Residual Fusion**: channel-wise gating，让policy自己决定哪个modality channel该被emphasized  
3. **Highway Gate**: state-dependent temporal vs feedforward，让policy自己决定何时该trust memory

加上**Terrain-aware foothold reward**提供directional supervision而不只是prohibition，整个system是一个coherent design——每个component都support "let the objective organize the representation"。

这种设计哲学让CReF能处理2.5D representation无法表达的complex terrain，并在zero-shot transfer到real world时保持robustness。这是一个值得deeply understand的perceptive locomotion framework。
