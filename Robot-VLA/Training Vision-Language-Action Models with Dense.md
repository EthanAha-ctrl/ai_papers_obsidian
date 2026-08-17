---
source_pdf: Training Vision-Language-Action Models with Dense.pdf
paper_sha256: 49a78d032d84ebab80bb8f4ee589d9169ca7be8c13c07d5aae0cbcafe7cc8729
processed_at: '2026-08-12T18:06:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ZR-0

## 这篇paper到底在解决什么问题？

想象你训练了一个robot model，它学会了用Franka单臂抓杯子。现在你想让它也能用双臂ALOHA或者人形机器人GR-1干同样的事——**cross-embodiment transfer**，这事儿一直很难搞。

难在哪里？不同robot的"身体构造"差别巨大：
- 6-DoF vs 7-DoF arm，joint 1转轴方向和范围都不同
- 有的控制joint position，有的控制end-effector pose
- 有的fixed base，有的mobile
- 传感器配置也五花八门

之前大家怎么处理？**zero-padding + per-embodiment normalization**——把所有action硬塞进一个固定dimension的vector，缺的dim补零。这样能joint train，但本质上只是"format alignment"，没有真正解决"semantic alignment"问题。同样的dimension（比如joint 1）在不同robot上物理意义完全不同，model学到的可能只是embodiment-specific的pattern，没法transfer。

## ZR-0的核心insight

作者观察到：**虽然low-level control是身体特定的，但high-level thinking是身体无关的**。

不管你用单臂、双臂还是人形，"抓起蓝色盘子→放到碗架"这个cognitive blueprint是一样的。差异只在手腕怎么发力、joint怎么转。

所以核心idea：**用ECoT（Embodied Chain-of-Thought）作为supervision signal，逼VLM学习body-agnostic的reasoning representations**。

## 具体怎么做？

### 架构：两个stream

**System 2 (VLM)**: Qwen3-VL-2B，负责"思考"——看图、读instruction、生成ECoT reasoning text
**System 1 (Action Expert)**: DiT-based diffusion model，负责"行动"——生成连续action chunk

两个stream通过cross-attention连接，VLM的features作为KV，action expert的tokens作为Q。

### ECoT包含什么？

每帧都annotate六个component：

1. **Scene Description** - "桌子上有个蓝盘子，旁边是白色毛巾" → 训练object recognition
2. **Progress Assessment** - "已抓起盘子，任务未完成" → 训练progress感知
3. **Future Plan** - "需要移动盘子到碗架并释放" → 训练long-horizon planning
4. **To-Do Actions** - 分解成原子动作："Grasp the blue plate from the towel" / "Place into the dish rack" → **这是cross-embodiment alignment的关键**
5. **Target Objects** - bounding box: `{"blue plate": [120, 85, 340, 260]}` → visual grounding
6. **Discrete Actions** - FAST tokenizer产生的离散action tokens → bridge到low-level control

第4个component是精髓——"Verb + Object + Prepositional"格式是body-agnostic的，对所有embodiment都适用。

### 训练loss

两个loss联合优化：

**ECoT loss (只更新VLM)**:
$$\mathcal{L}_{\text{ntp}} = -\mathbb{E}_D \left[ \sum_i \log \pi_{\theta'}(r_t^i \mid l, o_t, r_t^{<i}) \right]$$
标准next-token prediction，让VLM学会generate reasoning text。

**Flow matching loss (更新整个model)**:
$$\mathcal{L}_{\text{fm}} = \mathbb{E}_{D, \tau, \epsilon} \left[ \left\| \pi_\theta(l, o_t, s_t, A_t^\tau, \tau) - (A_t - \epsilon) \right\|^2 \right]$$

构造noisy action: $A_t^\tau = (1-\tau)\epsilon + \tau A_t$

变量含义：
- $A_t$: ground-truth action chunk
- $\epsilon$: Gaussian noise
- $\tau \in [0,1]$: noise level, 从Beta(1.5, 1.0)采样（偏向noisier samples）
- $A_t - \epsilon$: target vector field (从noise指向clean action)

Combined: $\mathcal{L} = \mathcal{L}_{\text{ntp}} + \alpha \mathcal{L}_{\text{fm}}$, pre-training时 $\alpha = 5$, post-training时 $\alpha = 1$。

## 最聪明的trick：Training用ECoT，Inference跳过ECoT

传统ECoT approach在inference时也要autoregressively generate reasoning text，慢得要命（几百ms latency对real-time control不可接受）。

ZR-0的solution：**attention mask设计**

Action expert的cross-attention **只attend VLM的input prompt features**（instruction + image），**排除ECoT tokens**。

这带来一个神奇效果：
- Training时，ECoT loss的gradient塑形了VLM的internal representations $f_t$
- Inference时，VLM只forward pass一次input prompt，产生的 $f_t$ 已经encode了ECoT的"knowledge"
- 不需要显式generate ECoT text，但ECoT的representation benefit保留了

相当于**implicit knowledge distillation**——把explicit reasoning internalize到implicit features里。

## 数据：ProcCorpus-60M

- 60M frames, ~1,000 hours, 400K+ trajectories
- 96.8% frames有ECoT annotation（near-dense）
- 来源：DROID, Bridge, Fractal, RH20T, Open X-Embodiment等
- ECoT由automated VLM pipeline生成
- 还混了general VL data（CapsFusion, Pixmo）co-training，防止VLM的general capability被catastrophic forgetting

## 实验结果如何？

### LIBERO (单臂)

| Model | Avg |
|-------|-----|
| OpenVLA | 76.5 |
| π0 | 94.2 |
| π0.5 | 96.9 |
| GR00T-N1.7 | 97.0 |
| **ZR-0** | **97.8** |

LIBERO-10 (long-horizon)上ZR-0拿96.4，比π0.5高4 points——这正是ECoT的To-Do Actions分解带来的优势。

### RoboCasa GR-1 Tabletop (人形)

| Model | Avg |
|-------|-----|
| GR00T-N1.6 | 47.6 |
| JoyAI-RA | 63.2 |
| **ZR-0** | **69.3** |

Pick-and-place tasks大幅领先（PlateToPan 89 vs 46），但Close tasks落后（BottleToCabinetClose 39 vs 84）。原因：pre-training data里pick-and-place多，close action少。**skill diversity决定ECoT能覆盖的范围**。

### RoboTwin 2.0 (双臂)

- ZR-0: 88.70% / 87.98% (Clean/Randomized)
- 与LingBot-VLA相当，但只用1/20的pre-training data
- Clean→Randomized gap只有0.72 points（π0.5是5.98），robustness好——VL data co-training的功劳

### Real-World xArm

| Task | π0.5 | ZR-0 | Δ |
|------|------|------|---|
| Push Blocks (OCR) | 66.1 | 94.0 | +27.9 |
| Clean Table (long) | 63.3 | 73.4 | +10.1 |
| Pick & Place (spatial) | 56.7 | 66.7 | +10.0 |
| Hang Cups (dexterous) | 85.0 | 70.0 | **-15.0** |
| **Avg** | 67.8 | 76.0 | +8.2 |

每个task的gain都能trace到ECoT的具体component：
- Push Blocks巨涨 → VL co-training保留OCR能力
- Clean Table涨 → To-Do Actions分解long-horizon
- Pick & Place涨 → Scene Description + Target Objects
- **Hang Cups跌 → 关键failure case！** 精细dexterous control更依赖action data scale，ECoT管不了

## 用一句话总结intuition

**ECoT在training时是representation的"老师"，逼VLM学会body-agnostic的thinking；inference时这个"老师"功成身退，但教给VLM的knowledge已经内化成features，action expert直接用。**

## 这篇paper为什么interesting？

1. **Reframe了cross-embodiment问题**: 从"format alignment"升级到"semantic alignment via shared cognitive supervision"

2. **解决了ECoT的inference cost问题**: 之前的ECoT方法inference慢，这里通过attention mask巧妙实现training-inference decoupling

3. **每个ECoT component有明确functional role**: 不是堆features，而是targeted supervision——Scene Description管recognition，Target Objects管grounding，To-Do Actions管planning...

4. **Failure case很enlightening**: Hang Cups的失败说明ECoT不是万能的，精细motor control仍需大scale action data。这给future work指明方向。

5. **Human video的潜力**: ECoT是body-agnostic的，理论上可以用Ego4D等human video annotate ECoT来augment pre-training data，不需要robot action label。这可能是scaling的另一个方向。

## Reference Links

- [ZR-0 GitHub](https://github.com/RUCKBReasoning/ZR-0)
- [ProcCorpus-60M / ProcVLM paper](https://arxiv.org/abs/2605.08774)
- [ECoT original paper](https://proceedings.mlr.press/v270/zawalski25a.html)
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [GR00T N1 paper](https://arxiv.org/abs/2503.14734)
- [Training Strategies for Efficient Embodied Reasoning](https://arxiv.org/abs/2505.08243)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
- [Open X-Embodiment](https://doi.org/10.1109/ICRA57147.2024.10611477)
- [DROID dataset](https://doi.org/10.15607/RSS.2024.XX.120)

---

**TL;DR**: ZR-0证明了一个principle——robot的"大脑"（reasoning）可以跨body transfer，"小脑"（motor control）是body-specific的。用ECoT dense supervision训练"大脑"学body-agnostic representations，然后让"小脑"各自适配。Training时用ECoT，Inference时跳过ECoT但保留benefits——这个trick让approach practical。2.6B参数+1,000小时数据，在多个benchmark上beat更大的model，但精细dexterous control仍是短板。

---

# ZR-0: Dense ECoT Supervision for Cross-Embodiment VLA

## Core Thesis & Intuition Building

这篇paper的核心thesis可以这样理解：robot的low-level motor control是embodiment-specific的（比如6-DoF vs 7-DoF arm的joint 1物理意义完全不同），但high-level cognitive process（scene perception, task planning, sub-task decomposition, object identification）是embodiment-agnostic的。作者利用这个observation，把ECoT作为cross-embodiment representation alignment的dense supervision signal。

一个类比：就像人类学做菜，不管你用左手还是右手拿刀，"切洋葱→炒蛋→装盘"这个cognitive blueprint是shared的，只有手腕的muscle activation pattern不同。ECoT监督的就是这个shared blueprint。

Reference: 类似思路在Embodied CoT (ECoT) original paper中提出
- [Robotic Control via Embodied Chain-of-Thought Reasoning](https://proceedings.mlr.press/v270/zawalski25a.html)
- [Training Strategies for Efficient Embodied Reasoning](https://arxiv.org/abs/2505.08243)

---

## Architecture 深度解析

### Dual-Stream System 1 / System 2 Framework

ZR-0借鉴Kahneman的System 1 / System 2 cognitive framework：

**System 2 (VLM)**: Qwen3-VL-2B-Instruct (2.1B params)
- Pre-trained on web-scale multimodal data
- Input: language instruction $l$ + image observations $o_t = [img_t^1, ..., img_t^n]$ from $n$ camera views
- Output: ECoT reasoning sequence $r_t$
- Features: last-layer hidden states $f_t$ extracted for action expert

**System 1 (Action Expert)**: Diffusion Transformer (DiT) (500M params)
- Input: VLM features $f_t$ + robot state vector $s_t$
- Output: action chunk $A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]$ via flow matching
- $H = 32$ during pre-training, $H = 10$ (LIBERO) or $H = 16$ (RoboTwin, RoboCasa, xArm) during post-training

### Cross-Attention Mechanism 的关键设计

DiT blocks遵循 **1:3 ratio** (1 self-attention + 3 cross-attention) 的repeating pattern。这跟GR00T N1的1:1 ratio不同，作者argue增加cross-modal interaction让action expert更彻底地吸收task instructions和visual observations。

**Self-attention layers**: bidirectional attention between state tokens和action tokens，做feature fusion
**Cross-attention layers**: state/action tokens作为queries，VLM的output features作为keys和values

**关键的Attention Mask设计**：
- Action expert **只attend到VLM的input prompt features**（task instruction + images的features）
- **排除ECoT tokens**
- 这样inference时，single forward pass of VLM over input prompt就足以产生所有action expert需要的features
- 不需要autoregressive ECoT decoding

这个设计是整个paper的"trick"——training时ECoT提供dense gradient signal更新VLM representations，但inference时完全skip ECoT generation。VLM的internal representations已经被ECoT supervision塑造成cross-embodiment aligned的，即使不显式generate ECoT text，features $f_t$ 也已经encode了embodiment-agnostic的reasoning。

Reference for GR00T N1:
- [GR00T N1: Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)

Reference for π0 (System 1/System 2 inspiration):
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)

---

## ECoT 的六个components详解

每个ECoT annotation是一个structured sequence，包含6个components，每个针对VLM的特定capability：

### 1. Scene Description
- Textual depiction of current visual scene
- 训练VLM的object recognition能力
- 例如："A blue plate sits on a white towel next to a dish rack."

### 2. Progress Assessment
- Brief reasoning about what has been accomplished + binary completion indicator (Yes/No)
- 训练VLM感知task progress
- 例如："The robot has grasped the blue plate. Task complete: No."

### 3. Future Plan
- Free-form natural language description of remaining work
- 训练VLM的temporal reasoning和long-horizon planning
- 例如："The robot needs to move the plate to the dish rack and release it."

### 4. To-Do Actions (关键component for cross-embodiment alignment!)
- Structured decomposition of future plan into atomic subtasks
- Format: **Verb + Object [+ Prepositional Phrase]**
- 例如：
  - "Grasp the blue plate from the towel."
  - "Place the blue plate into the dish rack."
- **Embodiment-agnostic format** - 同一个sub-task decomposition适用于任何robot hardware
- 这是cross-embodiment alignment的核心机制

### 5. Target Objects
- Bounding boxes in JSON format
- 例如：`{"blue plate": [120, 85, 340, 260]}`
- Visual grounding supervision，directs spatial attention toward task-critical regions
- 提升camera viewpoint和scene layout的generalization

### 6. Discrete Actions
- Embodiment-specific discrete action tokens from FAST tokenizer
- Bridge between high-level embodiment-agnostic reasoning和low-level continuous control
- Reference: [FAST: Efficient Action Tokenization for VLA Models](https://arxiv.org/abs/2501.09747)

---

## Training Objective 数学详解

### Loss 1: Next-Token Prediction (ECoT)

$$\mathcal{L}_{\text{ntp}} = -\mathbb{E}_D \left[ \sum_i \log \pi_{\theta'}(r_t^i \mid l, o_t, r_t^{<i}) \right]$$

变量解释：
- $D$: training dataset
- $\theta'$: VLM parameters
- $l$: task instruction (language)
- $o_t$: image observations at timestep $t$
- $r_t^i$: $i$-th token of ECoT sequence at timestep $t$
- $r_t^{<i}$: all tokens preceding $i$-th token in ECoT sequence (autoregressive context)

这个loss只更新VLM parameters $\theta'$，是standard language modeling loss applied to ECoT sequence。

### Loss 2: Flow Matching (Continuous Actions)

首先构造noisy action chunk：

$$A_t^\tau = (1-\tau)\epsilon + \tau A_t$$

变量解释：
- $A_t$: ground-truth action chunk $[a_t, a_{t+1}, ..., a_{t+H-1}]$
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\tau \in [0,1]$: flow matching timestep
- $A_t^\tau$: interpolated noisy action at timestep $\tau$
- 当 $\tau = 0$: pure noise; 当 $\tau = 1$: clean action

训练目标：approximate denoising vector field $A_t - \epsilon$ (i.e., the direction from noise to clean action)

$$\mathcal{L}_{\text{fm}} = \mathbb{E}_{D, \tau, \epsilon} \left[ \left\| \pi_\theta(l, o_t, s_t, A_t^\tau, \tau) - (A_t - \epsilon) \right\|^2 \right]$$

变量解释：
- $\pi_\theta$: full model (VLM + action expert)
- $s_t$: robot state vector
- $A_t^\tau$: noisy action chunk
- $\tau$: flow matching timestep
- $A_t - \epsilon$: target vector field

**关键的sampling trick**: $\tau$ sampled from $\text{Beta}(1.5, 1.0)$ distribution（不是uniform），这样emphasis放在noisier timesteps（$\tau$ closer to 0），因为预测从纯noise到clean action更难。这跟π0的设计一致。

Reference: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

### Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{ntp}} + \alpha \mathcal{L}_{\text{fm}}$$

- $\alpha \in \mathbb{R}$: trade-off weight
- Pre-training: $\alpha = 5$ (action loss权重高)
- Post-training: $\alpha = 1$

**Gradient flow的关键insight**：
- $\mathcal{L}_{\text{ntp}}$ 只更新VLM parameters $\theta'$
- $\mathcal{L}_{\text{fm}}$ 通过 $f_t$ 同时propagate gradients到action expert和VLM
- 这意味着ECoT supervision不仅训练VLM generate reasoning text，还通过features $f_t$ 间接影响action expert接收的conditioning

---

## Inference 时的 Flow Matching

从Gaussian noise开始，iterative refinement via forward Euler integration：

$$A_t^{\tau + \frac{1}{N}} = A_t^\tau + \frac{1}{N} \cdot \pi_\theta(l, o_t, s_t, A_t^\tau, \tau)$$

变量解释：
- $A_t^0 \sim \mathcal{N}(0, I)$: initial noisy action chunk
- $N$: number of denoising steps
- $\tau$: initialized to 0, incremented by $1/N$ each step
- 经过$N$次iteration后，$A_t^1$就是predicted action chunk

**Latency**: 单个NVIDIA A6000 GPU with bfloat16，generating一个action chunk约90 ms。Single H100约100 ms per action chunk。这对real-time deployment足够。

**Crucially**: ECoT text generation完全skip，通过cross-attention mask，single forward pass of VLM over input prompt就够。这就是为什么ZR-0能保持ECoT的representation benefits又不incur inference cost。

---

## Pre-training Data: ProcCorpus-60M

### Scale
- ~60 million frames
- ~1,000 hours
- 400K+ trajectories
- 96.8% ECoT annotation coverage (近乎dense annotation)

### Sources
Aggregated from:
- DROID
- BridgeData V2
- Fractal (RT-1 data)
- RH20T
- Open X-Embodiment subsets
- 其他

Reference:
- [DROID: Large-Scale In-The-Wild Robot Manipulation Dataset](https://doi.org/10.15607/RSS.2024.XX.120)
- [Open X-Embodiment](https://doi.org/10.1109/ICRA57147.2024.10611477)
- [RH20T](https://doi.org/10.1109/ICRA57147.2024.10611615)
- [BridgeData V2](https://proceedings.mlr.press/v229/walke23a.html)
- [ProcCorpus-60M / ProcVLM](https://arxiv.org/abs/2605.08774)

### VL Data Co-training
Mixed with general-purpose VL datasets:
- **CapsFusion**: image captioning
- **Pixmo**: VQA, image captioning, visual grounding

这些pure VL data只训练VLM（standard language modeling，无action prediction involved），目的是preserve VLM的general visual perception和language understanding，mitigate catastrophic forgetting。

Reference:
- [CapsFusion](https://doi.org/10.1109/CVPR52733.2024.01330)
- [Molmo and PixMo](https://openaccess.thecvf.com/content/CVPR2025/html/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.html)

---

## Implementation Details

### Hyperparameters
- Total params: 2.6B (2.1B VLM + 500M action expert)
- Image size: 224 × 224
- Global batch size: 1024 (pre-training)
- Post-training batch size: 64
- Optimizer: AdamW
  - $\beta_1 = 0.9$
  - $\beta_2 = 0.95$
  - $\epsilon = 10^{-8}$
- Learning rate: cosine schedule
  - 5% linear warmup
  - Peak: $3 \times 10^{-5}$
  - Decay to: $3 \times 10^{-6}$
- Precision: bfloat16 mixed
- Gradient clipping: 1.0
- DeepSpeed ZeRO for memory-efficient distributed training
- Flash-Attention 2
- Gradient checkpointing

### State/Action Handling for Cross-Embodiment
- States and actions **padded to 64 dimensions** with zeros
- Loss on padded dimensions **masked** (no gradient contribution)
- Each dimension **min-max normalized** using 1st and 99th percentiles of training data

这是处理heterogeneous embodiments的format-level technique。但作者强调，这只是让joint training成为可能，真正的cross-embodiment alignment来自ECoT supervision。

---

## Experimental Results 深度解析

### LIBERO (Single-arm, Franka)

Table 1 highlights:
- **ZR-0: 97.8%** average
- LIBERO-Spatial: 97.4
- LIBERO-Object: 99.4
- LIBERO-Goal: 98.0
- **LIBERO-10 (long-horizon): 96.4** ← 关键differentiator
- 比π0.5高4.0 points on LIBERO-10

LIBERO-10是long-horizon suite，chain多个manipulation sub-goals。ZR-0的优势在这里最明显，因为ECoT的To-Do Actions component专门decompose long-horizon goals into atomic sub-tasks。

Comparators:
- OpenVLA: 76.5
- CoT-VLA: 83.9
- π0: 94.2
- NORA-1.5: 94.5
- π0.5: 96.9
- GR00T-N1.7: 97.0
- DeepThinkVLA: 97.0
- MolmoAct2: 97.2
- **ZR-0: 97.8**

### RoboCasa GR-1 Tabletop (Humanoid)

Table 2 highlights:
- **ZR-0: 69.3%** average (24 tasks)
- 比第二名JoyAI-RA (63.2%) 高6.1 points

**Interesting pattern**:
- Pick-and-place tasks大幅领先：
  - CuttingboardToTieredbasket: 80 vs 36 (JoyAI-RA)
  - PlacematToPlate: 88 vs 38
  - PlateToPan: 89 vs 46
  - PlateToBowl: 82 vs 48
- **Close tasks underperform**:
  - BottleToCabinetClose: 39 vs 84
  - CanToDrawerClose: 47 vs 90
  - MilkToMicrowaveClose: 45 vs 84
  - WineToCabinetClose: 40 vs 54

作者的解释：pick-and-place是pre-training corpus中最prevalent的manipulation primitive，所以ECoT-supervised representation alignment效果好。Closing actions（关柜子/抽屉/微波炉）在pre-training data中far less frequent，limiting aligned representations学习。这暗示**scaling pre-training corpus to cover更多manipulation primitives会directly扩展ECoT受益的skill set**。

### RoboTwin 2.0 (Bimanual, ALOHA)

Table 3 highlights:
- 50 tasks
- **ZR-0: 88.70% / 87.98% (Clean/Randomized)**
- 与LingBot-VLA (88.56% / 86.68%) 相当
- 但pre-training corpus小得多：~1,000 hours vs ~20,000 hours

**Clean → Randomized gap** (越小越robust):
- ZR-0: 0.72 points drop
- Motus: 1.64 points drop
- π0.5: 5.98 points drop

ZR-0对visual variations (clutter, lighting, background)最robust，作者attribute到VL data co-training during pre-training。

**Bimanual coordination tasks表现强**:
- HandoverBlock: 93/87
- HandoverMic: 100/99
- PickDualBottles: 97/98

### Real-World xArm Experiments

Table 4:
- 4 tasks, 10 trials each
- Task progress score $S \in [0, 100]$ based on sub-step completion rubric

| Task | π0.5 | ZR-0 | Δ |
|------|------|------|---|
| Push Blocks (OCR) | 66.1 | **94.0** | +27.9 |
| Clean Table (long-horizon) | 63.3 | **73.4** | +10.1 |
| Pick & Place (spatial) | 56.7 | **66.7** | +10.0 |
| Hang Cups (dexterous) | **85.0** | 70.0 | -15.0 |
| **Avg.** | 67.8 | **76.0** | +8.2 |

**Per-task analysis reveals which ECoT components matter for which capability**:

1. **Push Blocks** (+27.9): 需要OCR-based reasoning (read letters on blocks)。ZR-0的巨大gain来自VL data co-training和ECoT reasoning preserve了VLM的text recognition capability，否则action-only fine-tuning会degrade它。

2. **Clean Table** (+10.1): Long-horizon pick-and-place repetition。ECoT的To-Do Actions component decomposes long-horizon goals into atomic, embodiment-agnostic sub-tasks，提供explicit alignment for pick-and-place primitive。

3. **Pick & Place** (+10.0): Spatial reasoning + referential language ("Put the green apple on the plate with a banana")。ECoT的Scene Description和Target Objects components train VLM感知spatial relationships和ground task-relevant regions。

4. **Hang Cups** (-15.0): Fine-grained motor precision + color understanding。π0.5 outperforms ZR-0。**这个failure case很enlightening**：ECoT supervision strengthens scene understanding和planning，但highly precise manipulation可能更depend on scale of action supervision during pre-training。ZR-0只有~1,000 hours action data vs π0.5的10,000+ hours。

---

## Ablation Study

Table 5 on LIBERO:

| Method | Spatial | Object | Goal | LIBERO-10 | Avg. |
|--------|---------|--------|------|-----------|------|
| ZR-0 | 97.4 | 99.4 | 98.0 | 96.4 | **97.8** |
| w/o ECoT PT | 96.8 | 98.6 | 94.8 | 92.6 | 95.7 |

"w/o ECoT PT" baseline: 从Qwen3-VL-2B-Instruct base model + randomly initialized action expert，直接fine-tune on LIBERO，bypass ECoT-supervised pre-training。

**ECoT pre-training贡献2.1 points average improvement**，主要来自LIBERO-10 (long-horizon)的3.8 points gain。这证实cross-embodiment representations learned during pre-training transfer effectively to downstream tasks。

---

## Discussion 部分的Future Directions

### 1. Scaling Robot Data
ZR-0只有~1,000 hours，比领先VLA models少一个order of magnitude:
- π0/π0.5: 10,000+ hours
- LingBot-VLA: ~20,000 hours
- Qwen-RobotManip: 30,000+ hours

RoboCasa实验表明：well-represented skills (pick-and-place) 大幅受益，underrepresented skills (close cabinets/drawers) 适应差。**Scaling pre-training corpus覆盖更多manipulation primitives会directly扩展ECoT受益的skill set**。

Reference: [Qwen-RobotManip Technical Report](https://arxiv.org/abs/2606.17846)

### 2. Learning from Human Egocentric Video
ECoT的structured reasoning (scene descriptions, task planning, sub-task decomposition, object grounding) 是agnostic to robot vs human manipulation的。这打开了利用海量human egocentric video data的可能性：
- Ego4D: 3,000 hours
- EPIC-KITCHENS

通过annotating human manipulation videos with ECoT，VLM可以acquire richer visual和semantic representations of manipulation behaviors，scale远超robot-only data，**且不需要robot action labels**。

Reference:
- [Ego4D](https://doi.org/10.1109/CVPR52688.2022.01842)
- [EPIC-KITCHENS](https://doi.org/10.1109/TPAMI.2020.2991965)

### 3. Efficient ECoT Annotation
Annotating every frame with dense ECoT需要substantial computational resources (每annotation一个VLM forward pass)。Future direction: select most informative frames for annotation rather than exhaustive annotation，match representation quality with significantly reduced cost。

---

## 与 Related Work 的 Positioning

### VLA Model Landscape
1. **Discrete action tokenization**: RT-2, OpenVLA, FAST - sequential decoding overhead, precision loss from tokenization
2. **Flow matching + VLM**: π0 (Mixture-of-Transformers), π0.5 (+ subtask planning), GR00T N1 (cross-attention DiT)
3. **ZR-0**: dual-stream + dense ECoT supervision (novel contribution)

### VL Data Co-training
1. General-purpose VL corpora: COCO, visual QA, image captioning, OCR, visual grounding
2. **Embodied reasoning VL data from robot trajectories**: ECoT framework属于这类
   - 更directly useful for downstream action prediction
   - ZR-0 focus: align VLM representations across heterogeneous embodiments through dense reasoning supervision at scale

Reference for OpenVLA: [OpenVLA: An Open-Source Vision-Language-Action Model](https://proceedings.mlr.press/v270/kim25c.html)

Reference for π0.5: [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)

---

## Build Intuition: 为什么这个approach work?

### 关键Insight 1: Decoupling "What" from "How"
- **What**: scene perception, task progress, planning, object identification - embodiment-agnostic
- **How**: motor control, joint torques, end-effector trajectories - embodiment-specific

ECoT监督的是"What"部分，让VLM的representations encode shared cognitive blueprint。Action expert只负责把"What"映射到"How"。

### 关键Insight 2: ECoT as Representation Regularizer
即使inference时不generate ECoT，training时的ECoT loss已经通过gradients塑形了VLM的internal representations $f_t$。这些features已经encode了scene understanding, planning, object grounding等capability，即使不显式decode成text。

可以类比：人类做熟练动作时不需要显式verbalize每个step，但learning阶段verbalization帮助build mental model。

### 关键Insight 3: Attention Mask as "Knowledge Distillation" Bypass
传统ECoT approaches (像ECoT原paper) 在inference时需要autoregressively generate reasoning text，latency高。ZR-0的attention mask设计相当于：
- Training: full ECoT supervision (rich gradient signal)
- Inference: 只用input prompt的features (ECoT的"knowledge"已经internalized)

这是一种implicit knowledge distillation，从"explicit reasoning"distill到"implicit features"。

### 关键Insight 4: Cross-embodiment Alignment via Shared Sub-task Vocabulary
To-Do Actions component用"Verb + Object [+ Prepositional Phrase]"格式：
- "Grasp the blue plate from the towel"
- "Place the blue plate into the dish rack"

这个format是embodiment-agnostic的。同一个sub-task decomposition对6-DoF arm, 7-DoF arm, bimanual, humanoid都适用。VLM学习这些sub-task的representations时，被迫extract真正transferable的semantic content，而非memorize embodiment-specific patterns。

---

## Potential Limitations & Open Questions

1. **Action supervision scale**: Hang Cups实验显示fine-grained dexterous control仍depend on large-scale action data。ECoT能帮助"what to do"但不能完全替代"how to do precisely"的data需求。

2. **ECoT annotation cost**: 96.8% frame coverage需要substantial VLM inference。如何select informative frames是open problem。

3. **Close tasks underperformance**: RoboCasa的Close tasks underperform暗示ECoT对underrepresented primitives的benefit有限。需要更diverse的pre-training data。

4. **ECoT quality dependency**: 整个approach依赖automated VLM-based annotation pipeline的质量。annotation noise会propagate到learned representations。

5. **Long-horizon evaluation**: LIBERO-10有4个sub-goals chained。Real-world long-horizon (10+ sub-goals)表现如何？Paper没评估。

6. **Cross-embodiment transfer quantification**: Ablation只比较了"w/o ECoT PT" vs "with ECoT PT"，但没有分离ECoT的cross-embodiment alignment effect vs general representation learning effect。需要zero-shot cross-embodiment transfer实验（train on embodiment A, test on embodiment B without fine-tuning）。

---

## Code & Resources
- GitHub: https://github.com/RUCKBReasoning/ZR-0
- ProcCorpus-60M paper: https://arxiv.org/abs/2605.08774
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631

---

## Summary: ZR-0 的 Contribution 可以这样概括

1. **Conceptual**: 把cross-embodiment transfer问题reframe为"align high-level cognitive representations via embodiment-agnostic supervision" (ECoT)
2. **Architectural**: 1:3 self-attention:cross-attention ratio + attention mask that excludes ECoT tokens at action expert input → training-time ECoT supervision without inference-time ECoT generation cost
3. **Empirical**: 2.6B model with 1,000 hours pre-training data achieves competitive results across single-arm, bimanual, humanoid, and real-world settings, outperforming larger models on specific capabilities (long-horizon, OCR, spatial reasoning)
4. **Practical**: Demonstrates structured reasoning supervision as scalable, embodiment-agnostic approach to cross-embodiment representation learning for VLA models

整体来说，ZR-0是一个设计精巧的work，核心创新在于把ECoT从"explicit reasoning at inference"转变为"dense supervision for representation alignment during training"，通过attention mask实现training-inference decoupling。这种implicit knowledge distillation思路对future VLA research有启发价值。
