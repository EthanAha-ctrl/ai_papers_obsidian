---
source_pdf: Aligning Video World Models with Executable.pdf
paper_sha256: 28ae84944459f6b19195eccf574e49c235252b865121eb54df535342e046861c
processed_at: '2026-07-18T07:38:28-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards — 深度解析

## 一、Problem Motivation: Executability Gap 的本质

这篇paper核心洞察在于：current video world models在robotics的decoupled pipeline中存在一个被忽视的**executability gap**。具体来说，decoupled paradigm是这样工作的：

```
Current Observation + Language Instruction 
    → Video World Model (生成future visual rollout)
    → Inverse Dynamics Model (IDM, 把frames转换成actions)
    → Robot Execution
```

关键问题：video model在pixel space优化visual realism，但完全没有kinematic constraints。生成的视频可能看起来完全正常，但机械臂出现：morphological deformation、self-intersection、abrupt temporal discontinuity。这些artifact被IDM open-loop地翻译成abrupt joint jumps、high-frequency jitter、out-of-bound commands。

**Intuition building**: 想象一个生成模型画了一只"看起来像手臂"的东西，但关节角度违反了rigid body constraint——比如某个link被拉伸了20%。对人眼来说这可能很难察觉，但当IDM试图从帧间pixel motion反推joint angles时，会得到完全不可执行的noisy action sequence。这就是为什么paper Section 5.6的Figure 5展示了三类failure mode：implausible kinematics, wrong contact, incorrect goal。

Reference: 论文主页 https://eva-project-page.github.io/

---

## 二、Method架构解析

### 2.1 整体Pipeline

```
[Stage 1: SFT] Pretrained Wan2.1-14B + Diffusion Forcing → SFT on embodiment data → Ours (w/o RL)
                                                                       ↓
[Stage 2: IDM Training] Conv backbone + Spatial Softmax + MLP → Frozen IDM (89.52% accuracy)
                                                                       ↓
[Stage 3: GRPO Alignment] Frozen IDM as Reward Model → Flow-GRPO on LoRA(rank=32) → Ours (with RL)
```

### 2.2 Flow-matching Video Generation的数学

Base model是Wan2.1-14B (Diffusion Transformer架构)。先理解flow matching：

给定video序列 $V = \{I_1, \ldots, I_T\}$，3D VAE将其编码到latent space得到 $x_1 \in \mathcal{V}$。

**变量解释**:
- $x_1$: 真实latent video sample（data endpoint）
- $x_0 \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise（noise endpoint）
- $t \in [0,1]$: 时间插值参数，$t=0$ 时是pure noise，$t=1$ 时是clean data
- $x_t = (1-t)x_0 + t x_1$: 线性插值的intermediate state
- $v_\theta(x_t, t, c)$: neural velocity field，参数为$\theta$，条件context为$c$
- $c$: conditioning signal (text prompt + visual observation)

Training objective (公式1):
$$\mathcal{L}_{FM} = \mathbb{E}_{x_0, x_1, t, c}\left[\left\|(x_1 - x_0) - v_\theta(x_t, t, c)\right\|_2^2\right]$$

物理意义：让网络 $v_\theta$ 学习从noise $x_0$ 到data $x_1$ 的transport vector。Inference时求解ODE $\dot{x} = v_\theta(x,t)$，从 $x_0$ 走到 $x_1$。

### 2.3 GRPO on Flow Models (Flow-GRPO)

标准flow matching是deterministic ODE，无法定义stochastic policy。Flow-GRPO (Liu et al. 2025, https://arxiv.org/abs/2505.05470)构造了一个SDE：

$$dx = f_\theta(x,t)dt + g(t)dw$$

**变量解释**:
- $f_\theta(x,t)$: drift term，derived from flow velocity $v_\theta$
- $g(t)$: diffusion term，引入stochasticity
- $dw$: Brownian motion增量
- $\pi_\theta(\tau|c)$: trajectory distribution

采样 $G=8$ trajectories $\{\tau_i\}_{i=1}^G$，计算reward $\{R_i\}$。Group-relative advantage:

$$\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

**变量解释**:
- $\mu_R$: group内reward均值
- $\sigma_R$: group内reward标准差
- $\epsilon$: numerical stability小常数

Clipped objective (公式2):
$$\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(r_i(\theta)\hat{A}_i, \text{clip}(r_i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

**变量解释**:
- $r_i(\theta)$: importance sampling ratio
- $\varepsilon$: clipping range (paper中设为0.001，相当tight)
- $\beta = 0.004$: KL penalty coefficient
- $\pi_{ref}$: reference policy (frozen SFT model)

**Intuition**: GRPO的核心trick在于用group statistics代替value function——这在video generation中特别重要，因为single video rollout的reward方差极大，没有靠谱的value baseline。

### 2.4 Inverse Dynamics Model设计

IDM输入是centered frame window $I_{t-k:t+k}$，输出是action $a_t$。Training loss (公式3):

$$\mathcal{L}_{IDM} = \mathbb{E}\left[\sum_t \|f_\phi(I_{t-k:t+k}) - a_t^{gt}\|_2^2\right]$$

**关键架构创新: Spatial Softmax**

对feature map $F \in \mathbb{R}^{C \times H \times W}$（将temporal frames stack在channel维度后），spatial softmax (公式4):

$$p_{ij}^c = \frac{\exp(F_{ij}^c)}{\sum_{i',j'} \exp(F_{i'j'}^c)}, \quad (x_c, y_c) = \sum_{i,j} p_{ij}^c (i,j)$$

**变量解释**:
- $F_{ij}^c$: feature map在位置$(i,j)$, channel $c$的值
- $p_{ij}^c$: 在channel $c$上位置$(i,j)$的softmax probability
- $(x_c, y_c)$: channel $c$对应的expected 2D coordinate (keypoint-like representation)
- $C$个channel得到 $C$个keypoints，concatenate后送入MLP

**Intuition**: 这里借鉴了Levine et al. 2016 (End-to-end training of deep visuomotor policies, https://jmlr.org/papers/v17/15-524.html)的设计。Spatial softmax强制每个channel学到一个spatial "attention peak"——这本质是一种learned keypoint detector。对比global average pooling，它对spatial location敏感，这对于action decoding至关重要：机械臂的joint angle直接由end-effector的pixel position决定。

**Ablation验证** (Table S2): 去掉spatial softmax换成global average pooling，test accuracy从0.9864降到0.7738，success rate从89.52%降到84.29%。差距显著。

### 2.5 IDM-based Executability Reward (核心)

这是paper最关键的技术贡献。把frozen IDM当作reward model，从generated video $V$ 反推action sequence $A = \{a_t\}_{t=1}^T$，然后对action sequence的kinematic properties打分。

**Step 1: 计算kinematic quantities**

通过finite differences计算：
- Velocity: $v_t = (a_t - a_{t-1})/\Delta t$
- Acceleration: $\alpha_t = (v_t - v_{t-1})/\Delta t$  
- Jerk: $j_t = (\alpha_t - \alpha_{t-1})/\Delta t$

**Step 2: Robust Huber Penalty** (公式5)

$$\text{Huber}(x; \delta) = \begin{cases} \frac{1}{2}x^2, & |x| \le \delta \\ \delta(|x| - \frac{1}{2}\delta), & |x| > \delta \end{cases}$$

**变量解释**:
- $x$: 输入值（acceleration或jerk）
- $\delta$: Huber threshold，控制quadratic vs linear的切换点
- 当$|x| \le \delta$: 二次penalty（small errors温和处理）
- 当$|x| > \delta$: 线性penalty（large errors不会dominate梯度）

**Intuition**: 为什么用Huber而不用L2？因为generated video的visual artifacts会产生outlier spikes——比如某一帧突然手臂"瞬移"。L2会被这些outlier dominate，导致训练信号被少数bad frames主导。Huber的线性段对outlier更robust，让大多数smooth frames也能贡献有效gradient。

**Step 3: Penalty components** (公式6-7)

$$\mathcal{P}_\alpha = \mathbb{E}_t[\text{Huber}(\alpha_t; \delta_\alpha)], \quad \mathcal{P}_j = \mathbb{E}_t[\text{Huber}(j_t; \delta_j)]$$

$$\mathcal{P}_{vel} = \mathbb{E}_t[\max(|v_t| - v_{max}, 0)]_2^2, \quad \mathcal{P}_{acc} = \mathbb{E}_t[\max(|\alpha_t| - a_{max}, 0)]_2^2$$

**变量解释**:
- $\delta_\alpha, \delta_j$: Huber thresholds for acceleration and jerk
- $v_{max}, a_{max}$: robot embodiment的物理limits（来自AgileX PiPER规格：J1最大速度180°/s等，见Table S3）
- $\mathcal{P}_{vel}, \mathcal{P}_{acc}$: 只penalize**超出**limit的部分（hinge-like）

**Step 4: Total Penalty & Reward Mapping** (公式8-9)

$$\mathcal{P}(A) = \lambda_j \mathcal{P}_j + \lambda_\alpha \mathcal{P}_\alpha + \lambda_{v-lim} \mathcal{P}_{vel} + \lambda_{a-lim} \mathcal{P}_{acc}$$

$$R(V) = \left(1 + \frac{\mathcal{P}(A)}{P_0}\right)^{-\gamma}$$

**变量解释**:
- $\lambda_j, \lambda_\alpha, \lambda_{v-lim}, \lambda_{a-lim}$: 各penalty项的weighting coefficients
- $P_0$: penalty scale，从pretrained video model的rollouts估计（让reward落在合理范围）
- $\gamma$: decay rate，控制reward对penalty的敏感度

**Intuition on reward shaping**: 这个 $(1 + \mathcal{P}/P_0)^{-\gamma}$ 形式很巧妙。当penalty $\mathcal{P} = 0$（perfectly smooth）时 $R = 1$；当penalty很大时 $R \to 0$。这是个bounded reward，对GRPO的advantage normalization友好。$P_0$的作用是adaptive scaling——根据pretrained model的baseline penalty校准，避免reward signal过弱或过强。

---

## 三、Experimental Setup详解

### 3.1 Base Model Stack

| Component | Choice | 备注 |
|-----------|--------|------|
| Video Backbone | Wan2.1-14B (DiT) | https://arxiv.org/abs/2503.20314 |
| Initialization | Large Video Planner (LVP) | 预训练于大规模manipulation data |
| Temporal Modeling | Diffusion Forcing | 改善observation history conditioning |
| Fine-tuning | LoRA rank=32 | 只更新LoRA参数，backbone frozen |
| Hardware | 8× NVIDIA A800 | FSDP + bf16 mixed precision |

### 3.2 Training Protocol

**SFT阶段**:
- RoboTwin: 4500 steps, 640×480 resolution
- Real-world: 2580 steps, 832×480 resolution
- 49-frame video clips
- AdamW, lr = 8e-6, β1=0.9, β2=0.95, weight_decay=0.05
- Constant schedule + 100 warmup steps
- Diffusion forcing with random history conditioning

**GRPO阶段**:
- Group size G=8
- AdamW, lr = 2e-4
- Clipping range ε=0.001 (非常tight)
- Advantage clipping at 5.0
- Max grad norm = 1.0
- KL coefficient β=0.004
- Simulation: 136 steps, 4 inner epochs (~6 days)
- Real-world: 46 steps, 2 inner epochs (~3 days)

### 3.3 RoboTwin 2.0 Benchmark

21个bimanual manipulation tasks，包括：
- ClickBell, HandoverMic, OpenLaptop
- MovePillBtl, PlaceCans, PlaceMouse
- PressStapler, StampSeal, TurnSwitch
- 等等

训练集：1050 video trajectories，每个task 50 expert demos。
评估：210 prompts (21 tasks × 10 observation-instruction pairs)。

Reference: RoboTwin 2.0 https://arxiv.org/abs/2506.18088

---

## 四、Results深度解读

### 4.1 Visual Rollout Quality (Table 1)

| Method | Kinematic | Interaction | Instruction | Perfect |
|--------|-----------|-------------|-------------|---------|
| Vidar (Wan2.2) | 67.6 | 66.7 | 87.6 | 62.9 |
| EVA (w/o RL) | 70.5 | 83.3 | 90.5 | 68.1 |
| **EVA (with RL)** | **91.4** | **86.2** | 89.5 | **83.8** |

**关键观察**:
1. Kinematic plausibility提升 +20.9% (70.5→91.4)——这是reward直接优化的目标，符合预期
2. Interaction plausibility也提升 +2.9%——bonus effect，因为smooth kinematics间接改善contact
3. Instruction adherence基本持平 (90.5→89.5)——RL alignment没有损害language grounding
4. Perfect execution +15.7% (68.1→83.8)——综合改善

**Intuition**: 为什么kinematic改善如此显著而interaction改善相对小？因为reward function只惩罚action-space smoothness和bounds，没有显式建模contact dynamics。Paper在Limitations中承认了这点。

### 4.2 RoboTwin Simulation Success (Table 2, 部分tasks)

| Method | ClickBell | OpenLaptop | PlaceCans | TurnSwitch | Average (21) |
|--------|-----------|------------|-----------|------------|--------------|
| ACT | 12/20 | 11/20 | 03/20 | 01/20 | 29.0% |
| DP | 11/20 | 10/20 | 08/20 | 07/20 | 29.5% |
| RDT | 16/20 | 12/20 | 01/20 | 07/20 | 37.1% |
| π0 | 09/20 | 17/20 | 07/20 | 05/20 | 45.7% |
| EVA (w/o RL) | 18/20 | 06/20 | 08/20 | 08/20 | 46.2% |
| **EVA (with RL)** | **20/20** | 12/20 | **09/20** | **13/20** | **52.6%** |

**关键观察**:
1. EVA在contact-rich tasks上improvement最明显：ClickBell (+2/20), TurnSwitch (+5/20)
2. OpenLaptop是个例外：EVA (w/o RL)只有6/20，with RL恢复到12/20但低于π0的17/20——可能因为laptop的articulated dynamics复杂
3. Average +6.4% (46.2→52.6) across 21 tasks

### 4.3 Real-World Deployment (Table 3)

**Seen tasks (5个)**:
| Method | StackBowl | HangCable | Place2Tray | Place2Basket | FoldTowel | Avg |
|--------|-----------|-----------|------------|--------------|-----------|-----|
| ACT | 11/20 | 05/20 | 12/20 | 09/20 | 05/20 | 42.0% |
| π0 | 12/20 | 08/20 | 13/20 | 12/20 | 06/20 | 51.0% |
| Vidar | 09/20 | 05/20 | 12/20 | 13/20 | 05/20 | 44.0% |
| GE-Act | 10/20 | 07/20 | 11/20 | 11/20 | 04/20 | 43.0% |
| EVA (w/o RL) | 12/20 | 08/20 | 12/20 | 14/20 | 05/20 | 52.0% |
| **EVA (with RL)** | **16/20** | 08/20 | **16/20** | **17/20** | 07/20 | **64.0%** |

**OOD tasks (5个)**:
| Method | PlaceBlock | PourWater | WipeTray | FoldCloth | PlaceToy | Avg |
|--------|------------|-----------|----------|-----------|----------|-----|
| π0 | 02/20 | 03/20 | 02/20 | 01/20 | 03/20 | 11.0% |
| Vidar | 07/20 | 08/20 | 06/20 | 07/20 | 06/20 | 34.0% |
| GE-Act | 01/20 | 00/20 | 01/20 | 00/20 | 01/20 | 3.0% |
| EVA (w/o RL) | 08/20 | 11/20 | 07/20 | 08/20 | 08/20 | 42.0% |
| **EVA (with RL)** | **10/20** | **15/20** | **11/20** | **12/20** | **12/20** | **60.0%** |

**关键观察**:
1. π0在OOD tasks上catastrophic drop (51%→11%)——典型的imitation learning泛化失败
2. Video world model方法显著更robust：Vidar 34%, EVA 42%/60%
3. EVA with RL在OOD上 +18% absolute improvement (42→60)——这是paper最强的结果
4. FoldCloth这种deformable object manipulation，EVA达到12/20，对比π0只有1/20

**Intuition**: 为什么video world model在OOD上远超VLA？因为video model的prior来自internet-scale video data，学到了object affordances和physical interactions的general knowledge。VLA只能从limited robot data学习，泛化受限。EVA的RL alignment进一步确保了生成的rollout物理可行，这就是为什么OOD改善比seen tasks更显著——alignment学到的是embodiment-invariant的smoothness prior。

### 4.4 IDM Validation (Table 4)

IDM在ground-truth video demonstrations上达到89.52% success rate。这个数字非常重要——它证明IDM本身足够accurate，因此可以作为trustworthy的reward signal。如果IDM accuracy很低，reward就是noisy的，GRPO会失败。

### 4.5 Reward Validity (Figure S1)

两个分析：
1. **Reward vs. visual artifacts**: artifact-free rollouts的reward显著高于有kinematic artifacts的rollouts
2. **Reward vs. execution success**: successful executions对应higher reward

这两个correlation验证了reward design的合理性。

---

## 五、Failure Modes & Reward Hacking

### 5.1 Unaligned Model的Failure Modes (Figure 5)

1. **Implausible kinematics**:
   - (a) Morphological deformation: 机械臂link被拉伸或弯曲
   - (b) Ambiguous joint articulation: 关节角度歧义
   - (c) Temporal discontinuity: 帧间不连续

2. **Wrong contact**: object penetration或missing contact

3. **Incorrect goal**: rollout没有向task目标进展

### 5.2 Reward Hacking (Figure S2)

这是paper很诚实的部分——prolonged GRPO training会出现degenerate behaviors：

1. **Incorrect object interaction**: 机械臂smooth但grab错位置
2. **Unrealistic link-length**: 调整link长度让motion更smooth（但物理不可能）
3. **Static behavior**: 机械臂几乎不动，trivially满足smoothness

**Mitigation**: checkpoint selection + early stopping based on validation rollout quality和downstream execution。

**Intuition**: 这个reward hacking现象揭示了smoothness reward的fundamental limitation——它只约束了action的dynamics，没有约束task completion。要解决这个问题，需要task-progress reward或者preference-based reward。Paper在future work中提到要探索richer dynamics-aware reward signals。

---

## 六、Technical Insights & Broader Context

### 6.1 与RLHF的类比

这个工作本质上是robotics版本的RLHF，但关键区别：
- **RLHF**: Reward model从human preference训练，reward是semantic的
- **EVA**: Reward model从IDM + kinematic constraints构造，reward是physical的

这种"physical reward model"的idea可以推广到其他embodiment-aware alignment任务。

### 6.2 与Cosmos-Predict的对比

NVIDIA的Cosmos-Predict (https://arxiv.org/abs/2501.03575)用VLM-based reward model post-train video backbone，优化text alignment、motion quality、visual quality。EVA的区别在于reward target是**executability**——直接在action space定义reward，这比VLM的semantic reward更precise。

### 6.3 与Diffusion Policy的关系

Diffusion Policy (https://diffusion-policy.cs.columbia.edu/)直接在action space用diffusion，绕过了video generation。EVA的decoupled pipeline优势在于：
1. Video model可以利用internet-scale video pretraining
2. Visual planning提供interpretable intermediate representation
3. IDM可以单独训练和验证

但代价是inference cost高——paper承认这限制了high-frequency reactive control。

### 6.4 Zero-Shot Data Augmentation (Appendix F)

Paper展示了一个promising application：用text-to-image generator合成initial scene，然后用aligned video world model生成motion-consistent trajectories。这创造了fully synthetic embodied data pipeline，无需human teleoperation。这对解决robotics的data scarcity问题有重要意义。

---

## 七、Limitations & Future Directions

1. **Contact dynamics缺失**: Reward只建模kinematic feasibility，没有force/friction/torque。对precision contact-rich manipulation（如装配任务）不够。

2. **Inference efficiency**: Diffusion-based video generation计算昂贵，无法做high-frequency closed-loop control。Future work: distillation或fewer-step samplers。

3. **Reward hacking**: Prolonged training出现degenerate behaviors，需要更principled的task-progress reward。

4. **Long-horizon**: 当前用receding-horizon execution (conditioning on最近4帧)，但对极长任务仍可能累积error。

---

## 八、核心Takeaway

这篇paper的核心贡献在于识别并解决了video world model在robotics中的**executability gap**问题。技术上的clever之处：

1. **IDM dual-use**: 训练时作为reward model，inference时作为action decoder——一个model两个用途，无需额外标注
2. **Action-space reward**: 直接在action space定义smoothness和bounds，比visual-space reward更precise
3. **Flow-GRPO**: 将RL alignment从LLM推广到flow matching video models，技术桥梁清晰

最impressive的实验结果是OOD tasks上的+18% improvement (42→60)，这证明alignment学到的是embodiment-invariant prior， generalize到unseen tasks。

**最终intuition**: 想象training a robot就像training a student。SFT是教科书学习，RL alignment是实习——在真实环境中获得feedback。EVA的reward function相当于实习导师告诉学生"你的动作太jittery了，smooth一点"——这种kinematic feedback在video generation阶段就注入，避免了在real robot上试错的高成本。

Reference links:
- Paper project page: https://eva-project-page.github.io/
- Wan2.1: https://arxiv.org/abs/2503.20314
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- π0: https://arxiv.org/abs/2410.24164
- Diffusion Forcing: https://boyuan.space/diffusion-forcing/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Levine et al. 2016 (spatial softmax): https://jmlr.org/papers/v17/15-524.html
- Cosmos-Predict: https://arxiv.org/abs/2501.03575
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
