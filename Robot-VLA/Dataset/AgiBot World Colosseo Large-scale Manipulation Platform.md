---
source_pdf: AgiBot World Colosseo Large-scale Manipulation Platform.pdf
paper_sha256: 62d5e6eeb33167b144f3cb89a08b07c4e90abf2d6dfda0285ccf843b708e091a
processed_at: '2026-07-18T06:09:07-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AgiBot World Colosseo 深度技术解读

Andrej，这篇paper来自AgiBot团队，我觉得它在robotics foundation model领域是一个相当重要的milestone，下面我从数据、硬件、model架构、实验四个维度给你做一个深度技术拆解，并尽量build up你的intuition。

---

## I. Paper背景与Motivation

### A. 核心问题
robotics foundation model长期受限于**high-quality data的scarcity**。对比NLP（GPT-4 [1]）和CV（SAM 2 [2]），robot data有几个根本痛点：

1. **Heterogeneous hardware**: 不同robot embodiment导致action space不一致
2. **Unstandardized collection**: 各家lab各搞各的，data质量参差不齐
3. **Short-horizon tasks**: 大部分dataset（如RT-1 [14] 130k traj, BridgeData V2 [17] 60.1k traj）都局限在tabletop的pick-and-place
4. **Lack of quality assurance**: OXE [6]虽然是1.4M traj但质量极度不均，DROID [7]虽然scene多但没有human-in-the-loop verification

### B. AgiBot World的positioning
他们的核心claim：通过**scale + diversity + quality**三位一体，把pre-trained policy在real-world manipulation task上的success rate提升了30%（相对OXE）。

参考链接：
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- RT-1 paper: https://arxiv.org/abs/2212.06817

---

## II. 数据集深度解析

### A. 规模对比 (Table I)

让我把关键数字拉出来对比：

| Dataset | Trajectories | Skills | Scenes | Long-horizon | Dex. Hand | Human-in-loop |
|---------|--------------|--------|--------|--------------|-----------|---------------|
| OXE [6] | 1.4M | 217 | 311 | ✗ (mostly <5s) | ✗ | ✗ |
| DROID [7] | 76k | 86 | 564 | ✗ (5-20s) | ✗ | ✗ |
| RoboMIND [18] | 55k | 36 | n/a | partial | √ | ✗ |
| **AgiBot World** | **1M+** | **87** | **106** | **√ (30-60s)** | **√** | **√** |

注意几个关键观察：
- AgiBot World的traj数量（1M+）虽然略少于OXE（1.4M），但**单条trajectory的时长显著更长**（30-60s vs OXE大部分<5s），所以总时长其实接近OXE
- 这是**第一个真正大规模dual-arm + dexterous hand**的real-world dataset
- **唯一带human-in-the-loop verification**和**failure recovery data**的dataset

### B. 长horizon的重要性 (Fig 3b)

这是这篇paper我觉得最被低估的一个insight。看Fig 3b的histogram：
- DROID: 主要分布在5-20s
- OXE: 主要分布在<5s
- **AgiBot World: 主要分布在30-60s，部分超过120s**

为什么这很重要？因为real-world task（如"make a coffee"）本质上是**多个atomic skill的组合**。短horizon dataset只能学到single-step policy，无法捕获：
1. **Sub-task sequencing**: 任务之间的temporal dependency
2. **Error recovery**: 中途出错后如何继续
3. **Skill composition**: pick → place → pour → stir这种chain

### C. Failure Recovery Data (~1% of dataset)

这是一个非常聪明的design choice。teleoperator在操作中偶尔会犯错（比如drop object），但他们能recover并完成任务。AgiBot保留了这些trajectory，并标注failure reason + timestamp。

这与近期RLHF/preference alignment的思路一致（参考GRAPE [28] arxiv.org/abs/2411.19309）：
- Failure recovery trajectory可以作为**contrastive signal**让policy学到"什么不该做"
- 对应Reflexion [29] (NeurIPS 2023)的verbal reinforcement思路在robotics上的迁移

### D. Human-in-the-Loop Pipeline (Fig 2)

三阶段pipeline：

**Phase 1 - Preliminary acquisition**: 
- 验证task feasibility
- 建立collection standard

**Phase 2 - Formal collection**: 
- Skilled teleoperator按standard采集
- Local verification (e.g., 检查丢帧)
- 上传到cloud

**Phase 3 - Post-processing**: 
- Annotator按Phase 1 standard验证
- 提供language annotation

关键的iterative loop：**collect small set → train policy → deploy → 评估data quality → refine collection protocol**。

举一个paper里提到的具体例子：deployed model会在action onset出现prolonged pause，对应annotator反馈的"inconsistent transitions + excessive idle time"。AgiBot据此修改protocol并加post-processing step剔除idle frames。

这种closed-loop data quality improvement在prior dataset里几乎没有。

---

## III. 硬件平台：AgiBot G1

### A. 机器人规格
- **Dual 7-DoF arms** (总共14 DoF)
- **Mobile chassis** (whole-body mobile manipulation)
- **Adjustable waist**
- **Modular end-effectors**: gripper OR 6-DoF dexterous hand
- **Visuo-tactile sensors** (在gripper上)
- **8 cameras**: 
  - Front: 1 RGB-D + 3 fisheye
  - Wrist: 2 RGB-D or fisheye (每个end-effector一个)
  - Rear: 2 fisheye
- **Control frequency: 30 Hz**

### B. 两种Teleoperation系统
1. **VR headset control**: 
   - Hand gesture → end-effector translation/rotation → IK解算到joint angle
   - Thumbsticks控制base/body
   - Trigger控制actuation
   - 限制：dexterous hand只能用几个predefined gesture

2. **Whole-body motion capture**:
   - 记录human joint数据（包括手指）
   - Mapping到robot posture
   - 可以做individual finger movement, torso pose, head orientation
   - 这是解锁dexterous manipulation的关键

这里我的intuition是：VR虽然fast but loses nuance，motion capture保留dexterity但更expensive。AgiBot同时support两者，让data collection可以根据task复杂度选合适的modality。

---

## IV. GO-1模型架构：ViLLA Framework (核心创新)

这是这篇paper最technically interesting的部分。GO-1提出的是**Vision-Language-Latent-Action (ViLLA)** framework，区别于主流的VLA (Vision-Language-Action)。

### A. 架构动机

为什么需要latent action这个intermediate representation？

VLA model（如OpenVLA [4]）直接predict离散化的low-level action，问题在于：
1. **Action space巨大**: 例如7-DoF arm × 30Hz × chunk size H，离散化后vocabulary太大
2. **Heterogeneous data难以unify**: 不同robot的action space维度、范围都不一样
3. **VLM预训练知识难以迁移**: VLM学到的是semantic-level reasoning，直接跳到continuous action太abrupt

ViLLA的解法是引入**latent action token**作为bridge：
- 用web-scale video（含human video如Ego4D）先学一个inverse dynamics model
- 把consecutive frames $I_t, I_{t+H}$ 压缩成discrete latent token $z_t$
- 这个token是embodiment-agnostic的（因为只描述"画面发生了什么变化"）
- VLM latent planner预测这个latent token
- Action expert再从latent token解码出robot-specific action

### B. 三阶段训练详解

#### **Stage 1: Latent Action Model (LAM)**

目标：从unlabeled video学习universal manipulation dynamics。

**Encoder** (inverse dynamics):
$$\mathbf{I}(z_t | I_t, I_{t+H})$$

变量含义：
- $I_t$: timestep $t$ 的image frame
- $I_{t+H}$: timestep $t+H$ 的image frame（H是prediction horizon）
- $z_t$: latent action token，长度为k=4，即 $z_t = [z_t^0, ..., z_t^{k-1}]$
- $\mathbf{I}$: inverse dynamics encoder（spatial-temporal transformer with causal temporal mask）

**Decoder** (forward dynamics):
$$\mathbf{F}(I_{t+H} | I_t, z_t)$$

- 给定初始frame + latent action，预测下一帧
- $\mathbf{F}$: spatial transformer
- 输入是 $I_t$ 和 quantized latent action tokens

**Quantization**: 用VQ-VAE objective [32] (NeurIPS 2017)
- Codebook size: $|C|$
- 这一步把continuous latent变成discrete token，方便后续用masked language modeling训练

这里参考Genie [30] (ICML 2024)的思路：Bruce et al.在generative interactive environment里用类似方法学latent action。LAPA [27] (ICLR 2025)也用了latent action pretraining，但AgiBot指出"LAPA的latent planning capability在downstream task没保留"，这是GO-1的关键改进点。

#### **Stage 2: Latent Planner**

**Backbone**: InternVL2.5-2B [33] (arxiv.org/abs/2412.05271)
- 2B参数量，preliminary实验证明这个scale对robotic task足够

**Input**:
- Multi-view images: $(I_t^h, I_t^l, I_t^r)$（head, left wrist, right wrist）
- Language instruction: $l$

**Output**: 预测latent action tokens
$$\mathbf{P}(z_t | I_t^h, I_t^l, I_t^r, l)$$

**Supervision**: 来自Stage 1的LAM encoder，只取head view
$$z_t := \mathbf{I}(I_t^h, I_{t+H}^h)$$

**Architecture details**:
- Latent planner: 24 transformer layers
- Full bidirectional attention
- Layer-by-layer conditioning from VLM backbone
- Multi-view image: InternViT编码 → project到language space

我的intuition：这相当于把VLM从"understanding"扩展到"planning"，但planning的target是离散latent action（codebook size远小于discretized low-level action），所以预训练VLM的semantic knowledge可以高效transfer。

#### **Stage 3: Action Expert**

**Objective**: Diffusion [34] (arxiv.org/abs/2410.08001) modeling of continuous low-level action distribution

**Input**:
- 多视角images: $(I_t^h, I_t^l, I_t^r)$
- Proprioceptive state: $p_t$
- Language instruction: $l$
- 隐式conditioning on latent planner output

**Output**: Action chunk
$$A_t = [a_t, a_{t+1}, ..., a_{t+H}], \quad H = 30$$

变量含义：
- $a_t$: timestep $t$ 的low-level action（joint angles, gripper, etc.）
- $H = 30$: chunk size，即一次预测30步action
- $p_t$: proprioceptive state（当前joint positions, end-effector pose等）

**为什么用diffusion而非直接回归？**
- Continuous action distribution往往是multi-modal的（比如同一个task可以有多种valid execution方式）
- Diffusion可以model这种multi-modality
- 这与RDT [10] (ICLR 2025)和π0 [26] (arxiv.org/abs/2410.24164)的思路一致

**Hierarchical conditioning**: 
- Latent planner通过masked language modeling生成discrete latent action tokens
- Action expert通过iterative denoising生成continuous actions
- Action expert的conditioning是hierarchical的，包括latent planner的输出

### C. Inference Flow
```
Multi-view images + Language instruction
       ↓
   VLM backbone (InternVL2.5-2B)
       ↓
   Latent Planner (24 layers, bidirectional)
       ↓
   k=4 latent action tokens (z_t)
       ↓
   Action Expert (diffusion denoising, H=30 steps)
       ↓
   Low-level action chunk A_t
```

### D. 与相关工作的对比

| Method | Pretrain data | Action representation | Long-horizon | Generalization |
|--------|---------------|------------------------|--------------|----------------|
| RT-2 [24] | Web VLM | Continuous | weak | moderate |
| OpenVLA [4] | OXE | Discretized | weak | moderate |
| RDT [10] | Heterogeneous | Diffusion | moderate | moderate |
| π0 [26] | Web + robot | Flow matching | strong | strong |
| LAPA [27] | Video | Latent (no preservation) | n/a | n/a |
| **GO-1** | **Web + AgiBot World** | **Latent → Diffusion** | **strong** | **strong** |

---

## V. 实验结果分析

### A. Data Scaling Experiment (Fig 6)

对比AgiBot World alpha (14% of full dataset, 236 hours) vs OXE (~2000 hours)：

**In-distribution**:
- AgiBot World: 0.77
- OXE: 0.47
- **+30% improvement**

**Out-of-distribution**:
- AgiBot World: 0.67
- OXE: 0.38
- **+29% improvement**

更remarkable的是：**AgiBot World alpha用1/10的数据量（236h vs ~2000h）依然surpass OXE**，这说明数据quality比raw scale更重要。

### B. GO-1 vs RDT vs Baseline (Fig 5)

在5个不同复杂度task上：
- GO-1 with latent planner: 平均+0.12 task completion score vs GO-1 without latent planner
- GO-1 vs RDT: 在"Pour Water"和"Restock Beverage"上显著超越

特别是在"Fold Shorts"（deformable object manipulation）上，latent planner带来大幅提升，这印证了latent action对long-horizon planning的必要性。

### C. Scaling Law Analysis (Fig 7a)

用三个数据规模：
- 10% alpha (9.2k traj)
- 100% alpha (~92k traj)
- beta (1M traj)

在4个seen task上评估out-of-the-box performance，得到**Pearson correlation coefficient r = 0.97**，强烈支持power-law scaling relationship。

这非常符合Chinchilla-style的scaling law intuition，只不过在robot data上这是首次观察到这么强的power-law。

### D. Data Quality Ablation (Fig 7b)

在"Wipe Table"任务上fine-tune RDT：
- Verified data (528 traj): higher score
- Unverified data (482 traj): lower score
- 差距: +0.18 completion score

这个ablation很有说服力——**数据quality > 数据quantity**，对应human-in-the-loop verification的价值。

---

## VI. Limitations & 我的思考

### Paper自己提到的limitation:
- 评估都在real-world，没有simulation environment（正在开发中）

### 我看到的几个潜在问题：

1. **Single embodiment**: 全部数据都是AgiBot G1采集的，虽然demonstrates cross-embodiment transfer的latent action理论应该work，但实际cross-embodiment generalization没充分验证

2. **Language annotation粒度**: paper提到"language annotations for both the overall task and each individual sub-steps"，但没具体讲sub-step annotation的quality control

3. **Latent action的interpretability**: codebook size |C|没明确给值，latent token的semantic meaning不明确

4. **Comparison with π0**: paper没直接和π0对比，只对比了RDT。π0是flow-based，GO-1是diffusion-based，两者都是SOTA generalist policy，直接对比会更有说服力

5. **Long-horizon evaluation的具体数字**: 虽然提到"over 60% success rate on complex tasks"，但具体哪些task、horizon多长，没展开讲

### Intuition building的几个关键insight：

**Insight 1: Latent action作为knowledge bridge**
传统的VLA直接从VLM跳到robot action，跨度太大。ViLLA引入latent action作为intermediate bridge，让VLM的semantic knowledge能通过一个embodiment-agnostic的中间表示transfer到robot-specific action。这和LLM里Chain-of-Thought的思路类似——**intermediate representation matters**。

**Insight 2: Human-in-the-loop的工程价值**
机器学习paper往往忽略data collection的engineering细节，但AgiBot的human-in-the-loop证明：**iterative refinement of data collection protocol比单纯堆data更有价值**。这是RLHF思路在data curation上的应用。

**Insight 3: Long-horizon是next frontier**
短horizon task（pick-and-place）已经基本solved，但real-world task本质上是long-horizon的。AgiBot World的30-60s traj（vs OXE的<5s）才是真实部署需要的data分布。这一点对整个field的direction有指导意义。

**Insight 4: Quality > Scale**
1/10数据量surpass full OXE，这说明robot data的bottleneck不是volume而是quality。这和Language model早期数据curation的发现一致。

---

## VII. 参考链接汇总

**主要paper**:
- AgiBot World: https://agibot-world.com/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- RDT-1B: https://arxiv.org/abs/2410.07839
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/
- LAPA: https://latentactionpretraining.github.io/
- Genie: https://arxiv.org/abs/2401.04024
- VQ-VAE: https://arxiv.org/abs/1711.00937
- InternVL2.5: https://arxiv.org/abs/2412.05271
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

**相关data collection工作**:
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RoboMIND: https://arxiv.org/abs/2412.13877
- RT-1: https://arxiv.org/abs/2212.06817

**Scaling law相关**:
- Data scaling in imitation learning: https://arxiv.org/abs/2410.18647

---

## VIII. 总结

AgiBot World Colosseo这篇paper在三个层面都有显著贡献：

1. **Data层面**: 1M+ long-horizon, dual-arm, dexterous manipulation trajectories with human-in-the-loop quality assurance，这是目前最大规模的高质量real-world robot dataset

2. **Model层面**: ViLLA framework通过latent action作为bridge，成功将web-scale VLM knowledge迁移到robot control，解决了VLA model直接predict action的难题

3. **Methodology层面**: 证明了power-law scaling (r=0.97)在robot data上同样存在，且data quality > data quantity

对你Karpathy这样关注scaling law和foundation model的人来说，这篇paper最有价值的可能是：
- **第一次在robotics上看到如此clean的power-law scaling**
- **Latent action作为intermediate representation的paradigm**，某种程度上类似LLM里token的作用
- **Human-in-the-loop data curation**作为quality scaling的methodology

期待你的feedback或者具体技术点上的深入讨论。
