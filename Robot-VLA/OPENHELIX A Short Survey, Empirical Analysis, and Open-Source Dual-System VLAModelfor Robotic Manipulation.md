---
source_pdf: OPENHELIX A Short Survey, Empirical Analysis, and Open-Source Dual-System
  VLAModelfor Robotic Manipulation.pdf
paper_sha256: 0052c13a0a75078446a6b74d9328ae40ee93c001fc45254bd81f0f0398b8f8b5
processed_at: '2026-08-06T00:38:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 OpenHelix

## 一句话概括

这篇paper发现：现在所有"双系统"机器人方案里，那个慢吞吞的大模型（MLLM）其实根本没在看图，它只是把文字指令复述了一遍丢给小模型。所以异步推理这种东西根本是伪需求——因为传下去的信息本来就不随时间变。为了fix这个问题，作者用辅助任务逼大模型真的去看图，性能就上来了。

## 背景故事

机器人控制圈一直有个老矛盾：

- 大模型（MLLM）聪明，见过整个互联网，什么物体都认识，什么指令都听得懂——但太慢了，1-3 Hz，根本没法实时控制机械臂
- 小模型（diffusion policy之类）快，50 Hz没问题——但太笨了，换个没见过的物体就懵逼

所以大家搞了个"双系统"方案：大模型当"大脑"慢慢思考high-level plan，小模型当"小脑"快速生成low-level action。两个系统异步跑，大模型隔一段时间吐一个latent embedding给小模型当指导。

听起来很合理对吧？

## 这篇paper干了啥

作者做了三件事：

### 第一件：survey了一圈现有方案

发现大家的设计五花八门，7个核心维度都有不同选择，没人系统地比过。

### 第二件：做了一堆ablation实验

发现了好几个反直觉的事：

**反直觉1：异步推理完全不影响性能**

理论上大模型应该实时感知环境变化，更新plan给小模型。但实验发现：大模型推理一次，小模型跑1步和跑60步，性能几乎一样。

这不对劲。如果大模型真的在看环境变化，隔60步才更新一次plan，性能应该暴跌才对。

**反直觉2：大模型没在看图**

作者做了个很巧的诊断实验：把大模型输出的latent embedding映射回word embedding空间，看它最近义词是啥。

结果发现：不管机械臂往左还是往右动，latent embedding对应的词几乎不变。它encode的基本就是instruction里的文字语义（"blue block"、"rotate"、"right"这些），对visual information完全不敏感。

说白了，大模型degenerate成了一个**文字转发器**。你看图不看图无所谓，反正输出的latent都差不多。那异步推理当然没用——因为传下去的信息本来就不随时间变。

**反直觉3：不加projector预训练，模型直接崩溃**

如果直接把大模型和小模型接一起joint训练，不管大模型是frozen还是fine-tune，全部fail（成功率0%）。

必须先用一个"pre-alignment"阶段单独训projector，让两个feature space初步对齐，才能开始joint training。

### 第三件：提出了OpenHelix方案

基于以上发现，方案非常simple：

**核心1：Prompt Tuning代替fine-tuning**

不动大模型参数，只加一个learnable token `<ACT>` 到instruction末尾，只训这一个token的embedding。

好处：保留大模型原有generalization能力，不会catastrophic forgetting。

实验证明：在language generalization场景（CALVIN-E）上，prompt tuning远超fine-tuning和frozen。

**核心2：辅助任务逼大模型看图**

这是最key的设计。在latent embedding后面接一个MLP head，让它去预测action的position、rotation、gripper state。

$$\mathcal{L}_{lm} = \text{BCE}(\text{gripper prediction}) + \omega_1 \cdot L_1(\text{location prediction}) + \omega_2 \cdot L_1(\text{rotation prediction})$$

意思就是：你不光要输出个latent embedding给小模型用，你自己还得从这个latent里把具体action预测出来。

这逼着大模型**必须从图像里extract spatial information**（物体在哪、朝哪转），否则auxiliary loss降不下去。

结果：加了auxiliary task，CALVIN-E从2.13涨到2.26，CALVIN从3.45涨到4.01。

**核心3：两阶段训练**

Stage 1：冻住大模型和小模型，只训`<ACT>` token和projector，让两个feature space先align。

Stage 2：冻住大模型，unfreeze小模型，joint训练。

如果不做Stage 1直接joint训练，直接崩。

## 最核心的intuition

整篇paper最深层的insight就一句话：

**Information flow取决于representation quality。**

如果latent embedding里只有textual summary，那再花哨的异步策略都没用——因为传下去的信息本身就不time-sensitive。

要让异步推理真正有意义，必须先让latent embedding包含time-varying visual information。Auxiliary task就是为了实现这个目标。

这给future research指明了方向：与其纠结异步步数、integration strategy这些surface-level的设计，不如先搞清楚latent embedding到底encode了什么information。Representation才是根本。

## 用人话总结

- 双系统机器人 = 大模型当脑子（慢但聪明）+ 小模型当肌肉（快但笨）
- 但现有方案里大模型其实在摸鱼，没看图，只是转发了文字指令
- 所以各种花哨的异步设计都是伪需求
- OpenHelix用辅助任务逼大模型真看图，性能就上来了
- 核心教训：信息质量 > 信息传递策略

---

# OpenHelix: Dual-System VLA 深度解析

## 1. Core Motivation: 为什么需要Dual-System？

Andrej, 这篇paper的核心出发点是robotic control领域的一个根本tension：**generalization vs. efficiency**。

传统paradigm有two extremes：

**Extreme 1 - Large VLAs (如RT-2)**：
- 参数规模：55B (RT-2) 或 5B
- 推理频率：1-3 Hz (55B) 或 ~5 Hz (5B)
- 优势：Internet-scale pretraining带来强大generalization
- 问题：**太慢了**，real-time control需要至少10-20 Hz

**Extreme 2 - Lightweight policies (如BC-Transformer)**：
- 推理频率：~50 Hz
- 优势：满足real-time要求
- 问题：**task-specific**，poor generalization to novel objects/instructions

这个tension催生了dual-system架构，灵感来自Kahneman的**dual-process theory** ([Thinking, Fast and Slow](https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555))：

| Cognitive Science | Robotics Mapping | 特征 |
|---|---|---|
| System 1 | Lightweight policy | Fast, automatic, intuitive, ~50 Hz |
| System 2 | MLLM/VLA | Slow, deliberate, reasoning, ~1-5 Hz |

关键insight：两个系统**异步运行**，System 2低频更新high-level plans，System 1高频生成low-level actions。

## 2. Survey: 现有Dual-System VLA Landscape

Paper总结了6个representative methods，我重新组织成清晰的comparison：

### Table 1 重新解读

| Method | System 2 Model | System 2 Input | System 2 Training | Latent Representation | System 1 Policy | System 1 Sensory | System 1 Training |
|---|---|---|---|---|---|---|---|
| [LCB](https://arxiv.org/abs/2405.04798) | LLaVA-7B | L+R | LoRA FT | Lang(`<ACT>`) | 3D Diffusion Actor | R+P+PC | Pretrain |
| [DP-VLA](https://openreview.net/forum?id=OVtUUY7BkK) | OpenVLA-7B | L+R | Frozen | Vis+Lang | Transformer | R+P | Scratch |
| [HiRT](https://arxiv.org/abs/2410.05273) | InstructBLIP-7B | L+R | LoRA FT | MaxPool(Vis+Lang) | RT-1 | R | Scratch |
| [RoboDual](https://arxiv.org/abs/2410.08001) | OpenVLA-7B | L+R | LoRA FT | Action+Lang | DiT | R+D+T+P | Scratch |
| [DexVLA](https://arxiv.org/abs/2502.05855) | Qwen2-VL-2B | L+R | LoRA FT | Lang | ScaledDP | R+P | Scratch |
| Helix | N/A | L+R+P | N/A | N/A | Transformer | R+P | N/A |

注意paper明确指出：[π₀](https://arxiv.org/abs/2410.24164)、[GR00T-N1](https://arxiv.org/abs/2503.14734)这类方法**不算真正的dual-system**，因为它们缺乏System 1的real-time perception input这个essential characteristic。

## 3. Seven Key Design Dimensions

Paper提出dual-system VLA有7个core design decisions，我逐一深入分析：

### 3.1 MLLM Selection
- 问题：什么样的MLLM既lightweight又sufficient？
- 趋势：[MiniVLA](https://arxiv.org/abs/2412.03304)用Qwen-VL 0.25B，[Flower](https://robolearesearch.github.io/flower/)强调spatial awareness
- Open question：是否需要robot data pretraining？RoboDual实验表明robotic pretraining确实提升language instruction following

### 3.2 Policy Selection
- 共识：[DiT](https://arxiv.org/abs/2212.09748)和[Flow Matching](https://arxiv.org/abs/2210.02747)架构都work
- 新兴方向：[CARP](https://arxiv.org/abs/2412.06782)的coarse-to-fine autoregressive、[Dense Policy](https://arxiv.org/abs/2503.13217)的bidirectional autoregressive
- Modal问题：System 1需要哪些modalities？depth/tactile/point cloud是否essential？

### 3.3 Latent Feature Representation Selection (最complex的dimension)

这是paper认为**最urgent需要research**的aspect。现有approaches差异巨大：

| Approach | 方法 | 代表 |
|---|---|---|
| Last layer hidden | 直接取MLLM最后一层 | DP-VLA |
| Middle layer hidden | 取中间层（更多visual info，更快） | GR00T-N1 |
| MaxPool aggregation | 对last layer的vis+lang features做pooling | HiRT, RoboFlamingo |
| Special `<ACT>` token | 训练special token作为bridge | LCB |
| Multiple `<ACT>` + lang | 多个special tokens + language features | RoboDual |
| Sophisticated latent selection | 更精巧的hidden state利用 | [MetaQuery](https://arxiv.org/abs/2504.06256), [LEGO](https://arxiv.org/abs/2403.14043) |

### 3.4-3.7 其他dimensions
- **MLLM Training**: frozen vs fine-tuning vs prompt tuning
- **Policy Training**: pretrain+FT vs scratch
- **Integration Strategy**: 如何embed latent info作为condition
- **Asynchronous Strategy**: training和inference的async设计

## 4. Empirical Analysis: 核心实验发现

### 4.1 实验设置
- **Base models**: LLaVA-1.0 + 3D Diffusion Actor (统一baseline)
- **Environment**: [CALVIN](https://arxiv.org/abs/2112.03227) benchmark
- **Three test scenarios**:
  1. **CALVIN** (standard): static objects, standard language
  2. **CALVIN-E** (enriched): 测试language generalization
  3. **CALVIN-D** (dynamic): objects以4种pattern运动

### 4.2 Why not single system? (Table 2)

```
Model   | Static | Left | Forward | Diagonal | Circle
RF      | 100    | 0    | 0       | 0        | 0
3DDA    | 82     | 84   | 46      | 67       | 80
```

**关键发现**：[RoboFlamingo](https://arxiv.org/abs/2311.01378)在dynamic scenarios**完全失败**！

原因分析：RF需要处理前6帧图像得到LSTM的latent representation。Training时objects静态所以latent稳定，但testing时objects移动导致latent剧烈变化，train-test distribution gap巨大。

**Intuition**：single-system的MLLM太慢，无法real-time感知environment变化。这justifies dual-system的necessity。

### 4.3 Policy Training Strategy (Table 3)

```
Strategy      | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len
Fine-tuning   | 96     | 83     | 68     | 58     | 48     | 3.53
From-scratch  | 89     | 71     | 49     | 42     | 34     | 2.85
```

**结论**：pre-trained policy + fine-tuning > from scratch。这符合transfer learning的一般规律——pretrained features提供good initialization。

### 4.4 MLLM Training Strategy (Table 4)

```
MLLM       | Integration  | Avg Len
Frozen     | w CLIP Loss  | 3.30
Frozen     | w/o CLIP Loss| 3.33
Fine-tuning| w CLIP Loss  | 3.53 ← best
Fine-tuning| w/o CLIP Loss| 3.13
```

**关键insight**：
- Frozen时CLIP loss影响小（因为本来就不更新MLLM）
- Fine-tuning时CLIP loss **critical**：无CLIP loss会破坏small model已训练的attention mechanism

### 4.5 Prompt Tuning: Paper的核心创新 (Table 5)

```
Benchmark  | MLLM           | Integration    | Avg Len
CALVIN     | Prompt-tuning  | w CLIP Loss    | 3.28
CALVIN     | Prompt-tuning  | w/o CLIP Loss  | 3.45
CALVIN-E   | Prompt-tuning  | w CLIP Loss    | 2.09
CALVIN-E   | Prompt-tuning  | w/o CLIP Loss  | 2.13
CALVIN-E   | Fine-tuning    | w CLIP Loss    | 1.74
CALVIN-E   | Frozen         | w CLIP Loss    | 1.46
```

**核心发现**：在**language generalization** (CALVIN-E)上，prompt tuning **远超** fine-tuning和frozen！

**Intuition**：
- Fine-tuning → catastrophic forgetting，破坏pretrained generalization
- Frozen → 无法adapt到downstream task
- Prompt tuning → 只训练`<ACT>` token的embedding，保留MLLM原有capability的同时allow adaptation

### 4.6 Projector Pre-alignment (Table 6) - 最surprising的发现

```
Pre-align | MLLM          | Avg Len
√         | Frozen        | 3.30
√         | Fine-tuning   | 3.53
√         | Prompt-tuning | 3.45
×         | Frozen        | 0.00 ← 完全失败!
×         | Fine-tuning   | 0.00 ← 完全失败!
×         | Prompt-tuning | 0.00 ← 完全失败!
```

**震撼性发现**：没有pre-alignment，**所有方法都完全失败**！

**Intuition**：upstream MLLM输出和downstream policy input之间存在巨大semantic gap。如果直接joint training，MLLM输出随机init的projector无法提供meaningful signal给policy，policy的gradient也无法有效backpropagate到MLLM。需要先pre-align projector让两个space有初步alignment。

### 4.7 Asynchronous Strategy - Counterintuitive发现 (Figure 4)

Paper测试了asynchronous steps从1到60，发现**性能几乎无变化**！

这非常counterintuitive——如果System 2提供real-time guidance，async step越大应该performance越差。

### 4.8 最核心的诊断实验 (Figure 5)

Paper做了一个brilliant的诊断：将`<ACT>` token的latent embedding映射到semantic space，看它实际encode了什么information。

实验设置：blue block持续向左移动，观察latent embedding对应的semantic content。

**发现**：
1. 无论robot向左还是向右移动，"right"的概率始终高于"left"
2. Top 10 similar words主要是instruction中的target object、spatial relations、action semantics
3. Latent embedding**对visual information不敏感**！

**Conclusion**：当前dual-system的MLLM**仅仅充当language transmitter**，没有真正利用visual reasoning capability。MLLM的输出基本是textual instruction的summary。

### 4.9 Auxiliary Task的必要性 (Table 7)

```
Type                    | Aux Task | Avg Len
MLLM (Prompt Tuning)    | ×        | 3.45
LLM (Prompt Tuning)     | ×        | 1.77 ← 移除visual input
MLLM (Prompt Tuning)    | √        | 4.01 ← +AUX
```

**关键发现**：
- 移除visual input (MLLM→LLM)，性能从3.45暴跌到1.77 → MLLM确实有用，但没有fully utilized
- 添加auxiliary task，性能从3.45提升到4.01 → **+16%提升**！

**Intuition**：auxiliary task强制MLLM必须使用visual information来预测action的location/rotation/gripper state，否则loss无法降低。这"逼迫"latent embedding包含multimodal information。

## 5. OpenHelix方法详解

基于以上empirical findings，paper提出OpenHelix，包含三个key components：

### 5.1 Architecture Overview

```
Input: {l, (o₁,a₁), (o₂,a₂), ...}
       ↓
   ┌───────────────────────────┐
   │ System 2: MLLM f_φ        │
   │ Input: o_t', l', <ACT>     │
   │ Output: z_t^{<ACT>}        │
   │ + Auxiliary prediction     │
   └───────────┬───────────────┘
               │ z_t^{<ACT>}
               ↓
   ┌───────────────────────────┐
   │ Projector (Linear)         │
   │ 4096 → 512                │
   └───────────┬───────────────┘
               │
               ↓
   ┌───────────────────────────┐
   │ System 1: Policy π_θ      │
   │ Input: o_t, z_t, c_t,     │
   │        τ_t^i, i            │
   │ Output: τ_t, a_t^g         │
   └───────────────────────────┘
```

### 5.2 Prompt Tuning机制

在instruction末尾添加learnable token `<ACT>`：

$$l' = \{l, \text{<ACT>}\}$$

其中：
- $l = \{w_i \in \mathbb{R}^d\}_{i=1}^N$：原始instruction，长度$N$，维度$d$
- `<ACT>` $\in \mathbb{R}^d$：新添加的learnable token
- $l'$：augmented instruction

Training时**freeze所有MLLM参数**，只更新`<ACT>` token的embedding。这通过更新lm-head layer中这个specific token的output embedding实现。

### 5.3 Auxiliary Task - Multimodal Reasoning Learning

这是paper的**核心创新**。公式(1)：

$$\mathcal{L}_{lm}(\text{<ACT>}) = \text{BCE}(\text{MLP}(f_\phi^g(o_t', l')), a_{t:t+T}^g) + \omega_1 \cdot ||\text{MLP}(f_\phi^l(o_t', l')) - a_{t:t+T}^l|| + \omega_2 \cdot ||\text{MLP}(f_\phi^r(o_t', l')) - a_{t:t+T}^r||$$

**逐项解析**：

**Term 1 - Gripper prediction loss**:
- $f_\phi^g$: MLLM的gripper head
- $o_t'$: third-view RGB image at timestep $t$
- $l'$: augmented instruction
- $a_{t:t+T}^g \in \{0,1\}$: ground truth gripper binary state序列 (open/close)
- BCE: Binary Cross Entropy
- 为什么用BCE：gripper是binary的

**Term 2 - Location prediction loss**:
- $f_\phi^l$: MLLM的location head
- $a_{t:t+T}^l \in \mathbb{R}^3$: ground truth 3D location trajectory
- $||\cdot||$: L1 norm
- 为什么用L1：location是continuous的，L1比L2更robust to outliers

**Term 3 - Rotation prediction loss**:
- $f_\phi^r$: MLLM的rotation head  
- $a_{t:t+T}^r \in \mathbb{R}^6$: ground truth rotation (6D representation)
- 6D rotation来自[Zhou et al.](https://arxiv.org/abs/1812.07035)，避免quaternion的double cover问题

**超参数**：
- $\omega_1, \omega_2$: balance location和rotation loss的权重

**设计intuition**：通过强制MLLM预测action的具体components，逼迫它必须从visual input中extract spatial information (object position, orientation等)，否则无法完成prediction。这解决了Section 4.8发现的问题——MLLM原本只是text transmitter。

### 5.4 Diffusion Policy Learning

公式(2)：

$$\mathcal{L}_{policy}(\theta, \text{<ACT>}) = \text{BCE}(\pi_\theta^g(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i), a_{t:t+T}^g) + \omega_3 \cdot ||\epsilon_\theta^l(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i) - \epsilon_{t:t+T}^l|| + \omega_4 \cdot ||\epsilon_\theta^r(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i) - \epsilon_{t:t+T}^r||$$

**逐项解析**：

**Inputs**:
- $o_t$: environment observation，两个RGB-D images from different viewpoints
- $z_t^{<ACT>}$: latent embedding from MLLM的`<ACT>` token
- $c_t$: proprioception (robot's own state)
- $\tau_t^i$: noisy trajectory at diffusion step $i$
- $i$: diffusion step index

**Outputs**:
- $\pi_\theta^g$: predict gripper state
- $\epsilon_\theta^l$: predict added noise on location
- $\epsilon_\theta^r$: predict added noise on rotation

**Diffusion process**:
- Training: 给ground truth trajectory $\tau_t^0$添加noise $\epsilon = (\epsilon^l, \epsilon^r)$得到noisy version $\tau_t^i$
- Model预测noise，用MSE/L1 loss监督
- Inference: 从pure noise开始，iteratively denoise得到action trajectory

**Action representation**:
$$a_t = \{a_t^l \in \mathbb{R}^3, a_t^r \in \mathbb{R}^6, a_t^g \in \{0,1\}\}$$

- $a_t^l$: 3D end-effector position
- $a_t^r$: 6D rotation representation (更stable than quaternion)
- $a_t^g$: binary gripper state

### 5.5 Two-Stage Training

**Stage 1 - Pre-alignment (2,000 iterations)**:
- Freeze: MLLM $f_\phi$ + Policy $\pi_\theta$
- Train: `<ACT>` token embedding + Projector (linear layer 4096→512)
- 目的：让projector初步align两个feature space

**Stage 2 - Joint training (until 100,000 iterations)**:
- Freeze: MLLM $f_\phi$
- Unfreeze: Policy $\pi_\theta$ + `<ACT>` token + Projector
- 目的：fine-tune policy to better utilize latent embedding

**Total loss**:
$$\mathcal{L}_{total} = \mathcal{L}_{lm} + \mathcal{L}_{policy}$$

## 6. Final Results (Table 8)

```
Type        | CALVIN Avg Len | CALVIN-E Avg Len
Only Policy | 3.27           | 1.42
MLLM(PT)+P  | 3.30           | 1.72
Full method | 3.45 (Asy=10)  | 2.26 (Asy=10)
Full method | 3.44 (Asy=60)  | 2.20 (Asy=60)
```

**关键conclusions**：
1. Dual-system主要提升**language generalization** (CALVIN-E: 1.42→2.26, +59%)
2. Auxiliary task对standard和generalization都有提升
3. **Async step从10到60几乎无影响**——再次验证Section 4.7的发现

## 7. 深层Intuition与Critical Analysis

### 7.1 为什么Async无效？深层原因

Paper表面结论是"MLLM不敏感environment changes"，但deeper analysis：

**Hypothesis 1 - Temporal coherence**：CALVIN的任务是sequential manipulation，high-level task一旦确定，中间过程不需要frequent replanning。这和[LCD place recognition](https://arxiv.org/abs/2306.13725)的发现类似——long-term planning不需要high frequency update。

**Hypothesis 2 - Information bottleneck**：latent embedding维度只有512 (projector后)，这个bottleneck可能只能transmitstatic task information，无法transmit time-varying visual details。

### 7.2 Prompt Tuning vs LoRA的深层对比

Paper没有直接compare，但可以infer：

| Method | Trainable Params | Generalization | Adaptation |
|---|---|---|---|
| Full fine-tuning | 100% | Worst | Best |
| LoRA | ~1-10% | Medium | Good |
| Prompt tuning | ~0.01% (1 token) | Best | Medium |

Prompt tuning只训练**一个token**的embedding，这是extreme形式的parameter-efficient tuning。为什么work？因为`<ACT>` token本质上是学一个**task-specific adapter**，将MLLM的general representation投影到policy能理解的space。

### 7.3 Auxiliary Task的设计哲学

这个设计让我想到[Princeton的"Learning by Predicting"](https://arxiv.org/abs/2306.07880)哲学。通过auxiliary prediction task，我们给MLLM一个**reason to look at the image**。

更deep的connection：这类似[self-supervised learning](https://arxiv.org/abs/2006.07733)的contrastive loss——通过auxiliary objective迫使模型学到useful representation。只不过这里是supervised auxiliary task。

### 7.4 与Figure-1 Dual-Process Theory的mapping

回到paper开头的Kahneman理论，OpenHelix的设计实际上实现了：

- **System 2 (MLLM)**: 保留deliberate reasoning capability (frozen params)，通过prompt tuning + aux task增强visual grounding
- **System 1 (Policy)**: 高频diffusion policy，利用pretrained features实现fast反应
- **Integration**: Two-stage training模拟human的"先理解再执行"process

### 7.5 Limitations与Future Directions

Paper坦诚承认的limitations：
1. 只在CALVIN simulation验证，no real robot
2. Policy execution speed未optimization
3. No humanoid robot deployment
4. Multi-robot collaboration未实现

我additional的critical observations：
- **Latent dimension bottleneck**: 512维可能限制information throughput
- **Single `<ACT>` token**: 可能需要multiple tokens encode更rich representation (类似[RoboDual的multi-`<ACT>`](https://arxiv.org/abs/2410.08001))
- **Auxiliary task simplicity**: 只predict next action，可以extend到long-horizon planning ([类似HPT](https://arxiv.org/abs/2406.13739))
- **No cross-embodiment**: 只在single robot arm验证

## 8. 相关资源

- [OpenHelix Project Page](https://openhelix-robot.github.io/)
- [OpenHelix GitHub](https://github.com/OpenHelix-robot/OpenHelix/)
- [CALVIN Benchmark](https://calvinrobot.github.io/)
- [3D Diffuser Actor](https://arxiv.org/abs/2402.10885)
- [LLaVA](https://llava-vl.github.io/)
- [Dual-Process Theory (Kahneman)](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- [Prompt Tuning original paper](https://arxiv.org/abs/2104.08691)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [6D Rotation Representation](https://arxiv.org/abs/1812.07035)

## 9. 总结：Build Intuition

这篇paper的核心contribution在于通过systematic empirical analysis揭示了dual-system VLA的**根本问题**：MLLM作为System 2，实际上degenerate成了text transmitter，没有发挥visual reasoning capability。

OpenHelix的solution elegant而simple：
1. **Prompt tuning** → 保留generalization，允许minimal adaptation
2. **Auxiliary task** → 强制visual grounding，避免MLLM degeneration
3. **Two-stage training** → 解决semantic gap，stable optimization

更深层的intuition：**representation决定了information flow**。如果latent embedding只encode textual summary，那async strategy就无所谓——因为transmitted的信息本身就不time-sensitive。要让async真正matter，必须先让latent embedding包含time-varying visual information。Auxiliary task正是为了achieve这个goal。

这给future research指明方向：如何design更好的latent representation，使其既包含high-level task semantics，又包含time-varying visual grounding，是dual-system VLA的key challenge。
