---
source_pdf: RoboRefer Towards Spatial Referring with Reasoning.pdf
paper_sha256: 43871d961a3d088a85fd660f44f02f25a848f55616a63a913ef9ff25e61c8ac6
processed_at: '2026-08-12T01:22:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboRefer 人话版

## 这篇论文在搞啥？

想象你在厨房里跟机器人说："把离你最近的那个杯子的logo前面的苹果拿起来，放到那一排苹果的末尾对齐。"

对人类来说，这指令很自然。但对机器人来说，这是地狱级难度：
- 要先找到离自己最近的杯子
- 然后判断杯子的logo朝向
- 再找到在logo前方的苹果
- 最后找到那排苹果的末尾位置
- 把苹果放过去，还得对齐

这就是**spatial referring**问题——给定一张图片和一个带空间约束的指令，模型得预测出一个**2D pixel point**，告诉机器人"往这儿操作"。

为什么是point不是bounding box？因为point可以非常精准地指代一个位置，在遮挡场景下只指代visible部分，而且天然能通过depth map转换成3D坐标。bbox会框进无关东西，不够"surgical"。

之前的VLM在单步空间理解上还行，但这种"多步组合推理"基本没人做。Gemini 2.5 Pro这么强，在这类任务上成功率也就20-40%左右。

参考：https://zhoues.github.io/RoboRefer

---

## 之前的方法为啥不行？

### 问题1：Depth被当成"假RGB"喂进去

之前的方法比如SpatialRGPT，把depth map当成一张普通图片，和RGB共用同一个encoder。这就像让一个只学过英语的人突然去看中文——他的英语能力会被"污染"。

具体后果：
- 预训练的image encoder weights被depth input搞乱
- 需要大量额外的RGB-only data来"补课"（SpatialRGPT用了2倍于spatial data的RGB data来co-training）
- 性能还是会有损失

### 问题2：只有SFT，模型在"背答案"

SFT（Supervised Fine-Tuning）的问题在于，模型倾向于**memorize**训练数据的答案，而不是真正学会**reasoning**。遇到没见过的spatial relation组合就懵了。

### 问题3：没有multi-step reasoning的数据

之前的数据集最多2步推理，31种spatial relations只有15种被覆盖。复杂场景下（cluttered桌面、多约束组合）完全hold不住。

参考：
- SpatialRGPT: https://spatialrgpt.github.io/
- RoboPoint: https://robopoint.web.berkeley.edu/

---

## RoboRefer的三个核心招数

### 招数1：给Depth配个"专属翻译官"

```
RGB image  → RGB encoder  → RGB projector  ─┐
                                           ├→ LLM
Depth map → Depth encoder → Depth projector ┘
```

Depth encoder结构跟RGB encoder一模一样，weights也从RGB encoder初始化。但训练时**各管各的**：
- RGB encoder不受depth影响，保持原有VQA能力
- Depth encoder独立学习depth特征
- 不需要大量RGB co-training data来"补课"

这个设计虽然简单，但很关键。实验证明在有限的RGB-only data下，dedicated encoder比shared encoder更好地保留image understanding能力。

### 招数2：两阶段训练 SFT + RFT

**Stage 1: SFT（学会基础空间感知）**

分两步：
1. 先只训depth projector，让depth和language对齐
2. 再全参数微调，用混合数据（RGB + RGB-D + instruction tuning + referring data）

SFT之后模型能做单步空间理解，但multi-step reasoning还是靠"背"。

**Stage 2: RFT（学会真正推理）**

这是最核心的创新。用GRPO（Group Relative Policy Optimization，DeepSeekMath提出的方法）做reinforcement learning。

核心idea：对同一个问题采样N=8个回答，用reward function打分，好的回答被强化，差的被抑制。

### 招数3：Metric-Sensitive Process Reward

这是RFT最精妙的地方。传统process reward需要训一个Process Reward Model（PRM）来评估中间步骤，但PRM本身可能不准，而且成本高。

RoboRefer用**rule-based reward**直接评估中间perception步骤，而且根据不同的perception type用不同的metric：

**Reward组成：**

1. **Outcome Format Reward (R_OF)**：输出格式对不对（regex匹配，1或0）

2. **Point L1 Reward (R_P)**：最终预测点离ground truth在50 pixels以内 → 1，否则0

3. **Process Format Reward (R_PF)**：中间推理步骤格式对不对
   ```
   [Perception Type] [Target Object]: [Value]
   ```
   比如：
   - `[Position] [the second largest cup]: [(0.245, 0.147)]`
   - `[Orientation] [the handle of the cup]: (1.000, 0.000, 0.000)`
   - `[Size] [the cup]: 0.12`

4. **Accuracy Reward (R_Acc)**：只对key steps打分
   - Position: L1 distance < 50px → 1
   - Orientation: cosine similarity > 0.8 → 1
   - Size: 误差 < 15% → 1

最终reward：$r_i = R_{OF} + R_P + 0.25 \times (R_{PF} + R_{Acc})$

Process reward乘0.25是为了不让多步骤累积reward盖过outcome reward。

**两个关键insight：**

1. **Metric-sensitive**：position、orientation、size是完全不同的数学对象，用同一个metric评估不合理。position用L1 distance，orientation用cosine similarity，size用relative error。

2. **Order-invariant**：空间推理不要求严格顺序。比如"keyboard和mouse之间的空地"，先找keyboard还是先找mouse对结果没影响。这跟数学推理的严格sequential非常不同。

参考：
- GRPO: https://arxiv.org/abs/2402.03300
- R1-V (implementation基础): https://arxiv.org/abs/2503.10617

---

## 数据怎么搞的？RefSpatial数据集

20M QA pairs，2.5M samples，31种spatial relations（之前只有15种），最高5步推理。

这是一个**bottom-up**的数据pipeline，三层逐步叠加：

### 第一层：2D Web Images（OpenImages）

**目的**：教模型基础空间概念，覆盖室内外各种场景。

**流程**：
1. 从1.7M OpenImages里filter出466k高质量图片
   - SigLIP2粗筛（1.7M → 934k）：去掉macro shots、文字图、GUI截图等
   - Qwen2.5-VL精筛（934k → 846k）：去掉artwork、dim lighting、B&W、distortion、collage
   - Spatial variance check（846k → 466k）：只保留有足够spatial diversity的

2. 构建pseudo-3D scene graph
   - RAM (Recognize Anything) → 物体类别标签
   - GroundingDINO → bounding boxes
   - UniDepth V2 → metric depth estimation
   - WildeCamera → camera intrinsics
   - SAM 2.1 → instance masks
   - 组成object-level point clouds → axis-aligned 3D bounding boxes

3. 生成hierarchical descriptions
   - Qwen2.5-VL生成object/image dense captions
   - Heuristic添加spatial ordering（"the third cup from left to right"）

4. 用QwQ-32B生成reasoning QA pairs

**为什么用OpenImages？** 室内室外都有，depth scale和category diversity都广。直接从2D提取3D信息很难，但可以构建pseudo-3D scene graph来做。

### 第二层：3D Embodied Videos（CA-1M）

**目的**：给模型indoor scene的fine-grained 3D空间感知。

CA-1M有per-frame 2D/3D oriented bounding boxes、camera intrinsics/extrinsics、depth maps。比ARKitScenes、ScanNet等更适合，因为它们要么缺3D bounding box，要么图像质量差。

**流程**：
1. 每20帧采样1帧（2M → 100k，减少temporal redundancy）

2. Bidirectional 2D bounding box matching
   - GroundingDINO + RAM + Florence-2 给unlabeled objects打标签
   - 用IoU matching把predicted boxes和CA-1M original boxes对齐
   - 保留有strong semantic alignment的boxes

3. Gravity alignment → top-down occupancy maps
   - 把point cloud和3D bounding box transform到gravity-aligned frame
   - Y-axis沿gravity vector，方便top-down projection

4. Free space sampling
   - Front/behind/left/right: 90°扇形区域，半径 = max(footprint对角线, 20cm)
   - Above/below: 投影缩小到80%避免overestimation
   - Between: 两个物体投影之间的planar area
   - Minimum free area: 0.036m²（半张A4纸大小）
   - 每个方向采样9000点，至少2000点visible才保留

5. Visibility filtering
   - 把3D points投影回2D image
   - 和aligned depth image比较，差异 > 2.5cm视为occluded

6. 生成更多spatial relation的QA pairs（28种，比2D多）

**为啥要这么复杂的free space sampling？** 因为placement任务需要找到"可以放东西的空地"。不能只看bounding box，要考虑occlusion、platform support等物理约束。

### 第三层：Simulation（Infinigen + Objaverse）

**目的**：教模型multi-step spatial reasoning。

2D和3D数据scalability不够，simulated data可以程序化生成大量带reasoning process annotation的数据。

**流程**：
1. Infinigen生成3k+ indoor scenes
   - 过滤条件：sufficient tabletop area、acceptable lighting、scene realism、camera accessibility
   - Lighting randomization: [0.6I, 1.4I]
   - Camera pose: pitch [-60°, -30°], height 0.3-0.8m above tabletop

2. Objaverse选3k+ 3D assets（从46k里筛）
   - Category filtering: 可放置在surface上、max dimension < 1m
   - Attribute filtering: axis alignment、single object、color diversity、no ground plane、high quality、distinguishable views
   - Manual filtering: 去掉irregular geometry影响bounding box的

3. GPT-4o + OrienText300K生成asset annotations
   - Orientation descriptions ("on the handle side of...")
   - Color labels
   - Object labels

4. Scene population
   - 3-9 assets per scene
   - 增加orientation vectors在XY平面的物体比例
   - 增加同类不同feature的物体共现

5. Blender Cycles rendering: 960×540, 2048 samples

6. QA generation with structured reasoning processes
   - 生成unique referring expressions（color + rank + distance + height等组合）
   - QA templates: Locate from Description, Identify from Relations, Locate Empty Space
   - 每个QA pair生成对应的thought process（reasoning steps）

**为啥simulation数据这么重要？** 只有simulated data能程序化生成大量带explicit reasoning process annotation的样本，这给RFT提供了训练signal。2D和3D数据很难大规模标注中间推理步骤。

参考：
- Infinigen Indoors: https://arxiv.org/abs/2406.11824
- CA-1M: https://arxiv.org/abs/2412.04458
- Objaverse: https://objaverse.allenai.org/

---

## 实验结果怎么样？

### Single-step Spatial Understanding (Table 1)

RoboRefer-8B-SFT在CV-Bench、BLINK、RoboSpatial、SAT、EmbSpatial上全面领先：
- CV-Bench: 96.90/98.33/93.50（Gemini 2.5 Pro是93.54/91.00/90.67）
- 平均超过Gemini 2.5 Pro 5% absolute
- 2B variant超过base NVILA-2B 21.7% absolute
- Depth input带来1.5% relative improvement（3D vs 2D benchmarks）

### Multi-step Spatial Referring (Table 2)

这是paper的核心evaluation：
- RefSpatial-Bench Location: RoboRefer-2B-RFT 52.00，Gemini 2.5 Pro 46.96
- RefSpatial-Bench Placement: RoboRefer-2B-RFT **54.00**，Gemini 2.5 Pro 24.21
- Unseen combinations: RoboRefer-2B-RFT **41.56**，2B-SFT 33.77（+9.1%）
- **2B-RFT超过Gemini 2.5 Pro 17.4% absolute average**

为啥Gemini在placement上这么差（24.21）？因为Gemini擅长2D referring（颜色、image-space localization），但在3D spatial relations involving distance（比如"second-farthest object"）上struggle。多个spatial constraint组合时性能急剧下降。

### Per-Step Analysis (Table 8)

RFT vs SFT在不同reasoning steps上的gain：
- Step 1: +3.34%
- Step 2: +4.17%
- Step 3: +9.09%
- Step 4: +4.16%
- **Step 5: +25.00%!**

RFT在更高reasoning steps上gain更大。Step 5上SFT是0%，RFT能到25%——这说明RFT真正学到了reasoning能力，不只是背答案。

### Real-World Robot Evaluation (Table 6)

"Pick the specific hamburger closest to the mug nearest the camera and place it in front of the teddy bear":
- OpenVLA: **0%**
- RoboPoint: **0%**
- RoboRefer: **80%**

"Pick the apple in front of the leftmost cup's logo side, navigate to the nearest table, and place it aligned with the apple row":
- OpenVLA: **0%**
- RoboPoint: **0%**
- RoboRefer: **60%**

**只有RoboRefer能handle long-horizon tasks requiring multi-step spatial referring**。其他方法在第一步就fail了。

RoboRefer以2.5Hz运行，当mug被移动时，能reactively更新target prediction，重新plan。当teddy bear旋转90°时，也能重新调整placement location。

### Simulation (Table 5)

Open6DOR V2 benchmark:
- Octo: 43.2% avg, 0.04s
- OpenVLA: 43.6% avg, 0.04s
- SoFar: 72.4% avg, 40s
- RoboRefer: **79.2% avg, 29s**

RoboRefer比SoFar快27.5%（因为compact size替代GPT-4o），success rate高6.8%。

---

## Ablation Studies的关键发现

### Data Recipe (Table 7)

去掉任何一类数据都会hurt performance：
- 去掉2D: BLINK（outdoor-centric）严重下降
- 去掉3D: CV-Bench（indoor-focused）下降
- 去掉Simulation: spatial diversity下降

三类数据缺一不可，各自弥补其他类的limitation。

### Depth Encoder (Table 7)

有dedicated depth encoder: CV-Bench 94.77
没有depth encoder: CV-Bench 91.24
**Depth encoder带来3.5%提升**

### Process Reward (Table 7)

有process reward: RefSpatial-Bench 53.00
没有process reward: RefSpatial-Bench 48.00
**Process reward带来5%提升**

### RGB+RGB-D Combination (Table 16)

只用RGB-D训练: RGB inference时 87.69/86.83/82.50
RGB+RGB-D混合训练: RGB inference时 **96.15/95.83/90.67**

混合训练让image encoder也学会从RGB alone提取spatial cues，避免over-reliance on depth encoder。

---

## 真实世界的系统集成

### UR5机械臂

运行流程：
1. RoboRefer以2.5Hz预测2D target point
2. 如果坐标变化大 → 触发motion interruption + replanning
3. Grasping:
   - RoboRefer → 2D point
   - SAM2 → segmentation mask
   - 过滤RealSense L515的point cloud
   - AnyGrasp → grasp pose in camera frame
   - Eye-to-hand calibration → UR5 base frame
4. Placement:
   - RoboRefer → 2D placement point
   - Camera intrinsics + depth → 3D coordinates
   - Transform到robot coordinate system

### G1人形机器人

- Head-mounted D435抓取+placement
- Chest-mounted L515做navigation
- FAST_LIO_LOCALIZATION_HUMANOID做SLAM
- Spatial referring统一了manipulation和navigation

### 语音中断

Whisper ASR转写语音 → RoboRefer处理成新2D坐标 → 任务redirect。这让human-robot collaboration真正interactive。

---

## 局限性

### 人类意图理解

实际human instruction往往ambiguous。比如：
- "pick the one facing the drink" — 场景里4个drink瓶子，只有中间2个对齐某个plate，但人通常基于概率偏见选那个plate
- "place another sushi between the plate and soy sauce dish" — 多对plate-soy sauce，只有最近的那对有足够物理空间

这些需要human prior knowledge和visual-linguistic reasoning，RefSpatial目前缺乏这类数据。

### 3D感知

- 主要依赖qualitative spatial relations（left, right）
- 预测2D image-plane coordinates，需depth-based conversion到3D
- Future work可能直接predict 3D points + visual traces

### 计算成本

整个pipeline粗略估算200+ A100 GPU days，replicability挑战大。RFT比Qwen2.5-VL-based方法慢2x，因为modified NVILA architecture incompatible with vLLM/SGLang acceleration。

参考：
- NVILA: https://arxiv.org/abs/2412.04468
- DepthAnything V2: https://arxiv.org/abs/2406.09414
- SAM 2: https://arxiv.org/abs/2408.00714

---

## 我的思考

RoboRefer的核心贡献在于把RL-based reasoning从text-only domain扩展到了spatial-visual domain。几个关键insight：

1. **Modality-specific encoding matters**: depth和RGB是fundamentally different modalities，shared encoder会互相干扰

2. **Metric-sensitive process reward**: position用L1、orientation用cosine、size用relative error——不同spatial attribute需要不同metric，这比generic process reward更精确

3. **Order-invariance is key for spatial reasoning**: 空间推理天然non-sequential（A在B左边 ⟺ B在A右边），这跟mathematical reasoning的strict sequential非常不同

4. **Point formulation unifies robotics tasks**: navigation waypoint、manipulation target、placement location都是point，一个formulation搞定所有

5. **Bottom-up data curation works**: 2D broad concepts → 3D indoor fine-grained → simulation multi-step reasoning，每层address不同limitation

但也有一些open questions：

- Human intent understanding beyond precise descriptions（probabilistic preference, spatial compatibility）
- Direct 3D reasoning without 2D-3D conversion
- Closed-loop integration for truly dynamic environments（现在是2.5Hz open-loop updates）
- Depth encoder的更sophisticated初始化（depth-specific pretraining？）
- 为什么order-invariance在spatial reasoning上work的theoretical justification

总体来说，RoboRefer为spatial intelligence in embodied AI建立了新的benchmark，特别是multi-step spatial reasoning with RL-based training。它不是终点，但确实是一个important step towards真正spatially intelligent robots。

项目主页：https://zhoues.github.io/RoboRefer

---

<answer>格式，1或0
2. R_P (Point L1 Reward): 预测点与GT的L1 distance < 50 pixels → 1，否则0
3. R_PF (Process Format Reward): 检查[Perception Type][Target Object]: [Value]格式
4. R_Acc (Accuracy Reward): 只对key steps计算
   - Position: L1 < 50px → 1
   - Orientation: cosine similarity > 0.8 → 1
   - Size: |pred - gt|/gt < 15% → 1

Advantage归一化：
A_i = (r_i - mean({r_j})) / std({r_j})

KL divergence regularization稳定更新。

**Process Format要求：**
[Perception Type] [Target Object]: [Value]

Perception Type ∈ {Position, Orientation, Size}
- Position: [(x, y)], normalized [0,1], 3 decimal places
- Orientation: 3D unit vector (x,y,z) in camera coordinates
- Size: scalar in meters

**RefSpatial数据集：**
20M QA pairs, 2.5M samples
- 2D web images (OpenImages): 466k images (from 1.7M after filtering)
- 3D embodied videos (CA-1M): 100k frames (from 2M)
- Simulation (Infinigen + Objaverse): 3k assets (from 46k)
- 31 spatial relations (vs 15 prior)
- Up to 5 reasoning steps

数据pipeline:
2D → pseudo-3D scene graphs via RAM + GroundingDINO + UniDepth V2 + SAM 2.1
3D → gravity alignment, top-down occupancy maps, free space sampling
Simulation → procedural generation with reasoning process annotations

**Key技术创新：**
1. Order-invariant process reward: reasoning steps不需要严格sequential
2. Metric-sensitive reward: 不同perception type用不同metric
3. No PRM needed: rule-based, 不需要Process Reward Model

**实验结果：**
Table 1: Single-step spatial understanding
- RoboRefer-8B-SFT: CV-Bench 96.90/98.33/93.50, BLINK 91.61/92.74
- 超过Gemini-2.5-Pro 5% absolute average
- Depth input带来1.5% relative improvement on 3D benchmarks

Table 2: Multi-step spatial referring
- RefSpatial-Bench上2B-RFT超过Gemini-2.5-Pro 17.4% absolute average
- Unseen combinations: 2B-RFT vs 2B-SFT +9.1%

Table 6: Real-world evaluation
- 2.5Hz operation
- OpenVLA/RoboPoint都是0%，RoboRefer能完成long-horizon tasks

**Ablation (Table 7):**
- Data recipe: 2D+3D+Simulation最优
- Depth encoder: dedicated > shared
- Process reward: +5 points

让我组织回答，重点build intuition，给出技术细节，公式讲解，并提供web links。

我应该讲：
1. 论文的核心insight和motivation
2. Architecture关键设计决策
3. SFT训练细节和公式
4. RFT和reward设计的巧妙之处
5. RefSpatial数据构建pipeline
6. 实验亮点和ablation insight
7. Limitations和future work
8. 与相关工作的对比

Web links需要提供：
- Project page: https://zhoues.github.io/RoboRefer
- arXiv: 应该有arXiv版本
- 相关论文的链接

让我写一个详尽的回答。# RoboRefer 深度技术解析

## 1. 核心问题与直觉构建

这篇paper tackles一个在robotics中极其fundamental但被严重低估的问题：**spatial referring**。给定一张RGB(D)图像和一个spatially constrained instruction（比如"place the object between the pen holder and keyboard, lined up with the cup's logo"），模型需要预测一个**2D pixel point** (x, y)来指代目标location或placement point。

为什么是2D point而不是3D point或bounding box？这里有一个非常elegant的insight：
- 2D point可以trivially map到3D coordinates via depth + camera intrinsics，提供accurate spatial anchors
- 在occlusion场景下，point能accurately localize一个visible object part，而2D bbox往往include irrelevant objects
- point formulation naturally unifies navigation waypoints、manipulation targets、placement locations——enabling multi-task learning

任务分两个level：
1. **Single-step spatial understanding**：识别object properties (position, orientation) 和 inter-object relations (distance, direction)
2. **Multi-step spatial reasoning**：compositional reasoning来sequentially resolve complex references，比如"the plate closest to the observer" → "the soy sauce dish" → "free space between them"

Prior work (SpatialRGPT, SpatialVLM, RoboPoint)主要focus在level 1，level 2基本unexplored。

参考链接：
- Project page: https://zhoues.github.io/RoboRefer
- SpatialRGPT: https://spatialrgpt.github.io/
- RoboPoint: https://robopoint.web.berkeley.edu/
- SpatialVLM: https://spatial-vlm.github.io/

## 2. Architecture: Dedicated Depth Encoder的关键决策

RoboRefer基于NVILA (2B/8B)作为base VLM，但做了一个非常关键的设计决策——**separate RGB encoder和depth encoder**，而不是share一个encoder。

### 为什么shared encoder有问题？

Prior work (SpatialRGPT, SpatialBot)把depth当作RGB-like input，share同一个image encoder。这导致**modality interference**：
- 预训练的image encoder weights被depth input"pollute"
- 需要大量额外的RGB-only co-training data来compensate（SpatialRGPT用了over 2x的RGB-only data）

### RoboRefer的设计：

```
RGB image → RGB encoder (SigLIP-so400m-patch14-448) → RGB projector → 
                                                              ↓
                                                            LLM (Qwen2)
                                                              ↑
Depth map → Depth encoder (initialized from RGB weights) → Depth projector →
```

关键点：
- **Depth encoder结构mirror image encoder**，weights初始化自RGB encoder
- Joint training时，**image encoder不受depth input影响**，depth encoder独立更新
- 保留general VQA性能不需要extensive RGB-only co-training

这个设计决策的实验验证见Table 4：dedicated encoder在limited RGB-only data (1/20 RefSpatial QA)下，比shared encoder更好preserve image understanding。

SigLIP-so400m-patch14-448支持448×448分辨率，采用dynamic resolution——高分辨率图像产生更多visual tokens via finer patch division。这对point prediction这种需要fine-grained perception的任务crucial。

参考：NVILA paper https://arxiv.org/abs/2412.04468

## 3. SFT (Supervised Fine-Tuning)细节

SFT分两步，对应两个不同的training objective。

### Step 1: Depth Alignment

只训练depth projector来align depth space和textual space。

Loss function（Eq. 1）：

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(\mathcal{O}, \mathcal{Q}, A) \sim \mathcal{D}} \sum_{t=1}^{T} \log \pi_\theta(y_t \mid \mathcal{O}, \mathcal{Q}, y_{<t})$$

变量解释：
- $\mathcal{O}$：sensor observation，这里只用RGB-D（depth input）
- $\mathcal{Q}$：textual question
- $A$：answer，可能是direct point coordinate或含中间reasoning steps
- $y_t$：第$t$个output token
- $y_{<t}$：$t$之前的所有generated tokens（autoregressive context）
- $\pi_\theta$：model parameterized by $\theta$的token distribution
- $T$：answer序列的长度

Hyperparameters：
- Max learning rate: 1e-4
- Weight decay: 0
- Warm-up ratio: 0.03
- Batch size: 7/GPU (2B), 3/GPU (8B)
- 1 epoch
- 2B: 10 nodes × 12 hours
- 8B: 8 nodes × 40 hours

### Step 2: Spatial Understanding Enhancement

全参数微调，用混合数据：
- RefSpatial (RGB) + RefSpatial (RGB-D)：6.8M samples after slicing
- LLaVA-1.5 + LRV：965k instruction tuning
- RefCOCO/+/g：321k referring
- SAT：176k
- EmbSpatial：127k

Total 8.5M samples after slicing.

关键trick：**RefSpatial既用RGB也用RGB-D训练**，enforce image encoder学习spatial understanding beyond depth cues。这让model支持both RGB-only和RGB-D inference。

Hyperparameters：
- Max learning rate: 5e-5
- Batch size: 6/GPU (2B), 2/GPU (8B)
- 1 epoch
- 2B: 10 nodes × 2 days
- 8B: 10 nodes × ~1 week

## 4. RFT (Reinforcement Fine-Tuning)的精妙设计

这是这篇paper最core的技术创新。SFT会**memorize答案而非generalize**，RFT用GRPO (Group Relative Policy Optimization from DeepSeekMath)来fix这个问题。

### 为什么不用PPO而用GRPO？

PPO需要costly value network。GRPO通过**intra-group reward comparison**来estimate relative advantages，避免value network，computation更轻、optimization更简单。

### Sampling策略

给定input state $s = (\mathcal{O}, \mathcal{Q})$，从current policy采样N个actions：

$$a_i \sim \pi_\theta(a \mid \mathcal{O}, \mathcal{Q}), \quad \text{for } i = 1, 2, \ldots, N$$

- $N = 8$ in implementation
- $a_i$：第$i$个sampled response
- $\pi_\theta$：current policy（initialized from $\pi_{\text{SFT}}$）

### Reward Function组合

这是paper最innovative的地方——**metric-sensitive process reward functions**：

$$r_i = R_{OF}(a_i) + R_P(a_i) + \alpha R_{PF}(a_i) + \alpha R_{Acc}(a_i)$$

其中 $\alpha = 0.25$。

四个reward components详解：

**1. Outcome Format Reward $R_{OF}$**

检查输出格式是否严格遵守：
```
<answer>(x, y)</answer>
```
- Format正确 → 1
- Format错误 → 0

这reward很cheap，纯regex匹配。

**2. Point L1 Reward $R_P$**

评估最终point prediction的accuracy：
- $|predicted\_point - GT\_point|_1 < 50$ pixels → 1
- 否则 → 0

阈值50 pixels灵感来自Seg-Zero。

**3. Process Format Reward $R_{PF}$**

强制中间reasoning步骤遵守structured format：
```
[Perception Type] [Target Object]: [Value]
```

- Perception Type ∈ {Position, Orientation, Size}
- Position Value: [(x, y)] normalized to [0,1], 3 decimal places
- Orientation Value: 3D unit vector (x, y, z) in camera coordinate system
- Size Value: scalar in meters

Examples：
- `[Position] [the second largest cup]: [(0.245, 0.147)]`
- `[Orientation] [the handle of the second largest cup]: (1.000, 0.000, 0.000)`
- `[Size] [the second largest cup]: 0.12`

**4. Accuracy Reward $R_{Acc}$**

这是最关键的innovation。对key steps（RefSpatial中annotated的中间perception steps）apply metric-specific reward：

- **Position**：$|pred - GT|_1 < 50$ pixels → 1, else 0
- **Orientation**：$\cos(\vec{pred}, \vec{GT}) > 0.8$ → 1, else 0
- **Size**：$|pred - GT| / GT < 0.15$ → 1, else 0

### 为什么process reward不需要PRM？

传统process-based reward需要Process Reward Model（fine-tuned LLM/VLM来评估中间step）。RoboRefer用**rule-based reward**直接评估intermediate perception，原因有二：

1. **LLM无法处理images**，无法判断predicted coordinates是否match target object
2. **VLM虽然在textual coordinates上visual understanding不精确**（prior work显示）

RoboRefer利用RefSpatial提供的ground-truth step-wise annotations，用regex匹配target object，然后apply metric-specific verification。

### Order-Invariance的关键insight

Reasoning过程**不严格要求sequential**。比如"the free area between keyboard and mouse"——先identify keyboard或先identify mouse，对最终结果没有影响。所以reward design是order-invariant的，不constrain reasoning trajectory到fixed sequence。

### Advantage Normalization

$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

- $A_i$：第$i$个response的relative advantage
- $r_i$：第$i$个response的reward
- $\{r_j\}$：group内所有N个responses的reward集合
- mean/std：group内的均值和标准差

这个normalization measures how each reward compares to mean in units of standard deviation。高reward的response被reinforced，低reward被suppressed。

### KL-divergence Regularization

为稳定RL training，update被KL divergence约束：

$$\text{Loss} = -\mathbb{E}[A_i \log \pi_\theta(a_i|s)] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

- $\pi_{\text{ref}}$：reference policy（通常是SFT model）
- $\beta$：KL penalty coefficient
- 这防止policy drift太远，保持incremental updates

### RFT训练细节

- 只对2B model做RFT（compute限制）
- 2 epochs, batch size 1/GPU, 8 outputs per GRPO group
- 用R1-V implementation (modified to support 3D-aware architecture)
- Training data：RefSpatial中3 reasoning steps的中等难度样本，100k samples
- 1 node × 3 days
- Note：不能用vLLM/SGLang加速因为修改了NVILA架构且需RGB-D input

## 5. RefSpatial数据集构建Pipeline

这是paper的另一个大贡献。2.5M samples, 20M QA pairs, 31 spatial relations（prior只有15）。

### Data Recipe: Bottom-Up Design

**2D Web Images (OpenImages)**：
- Goal：core spatial concepts + comprehensive depth perception across indoor/outdoor
- Filtering pipeline：
  - Stage 1: SigLIP2-giant-opt-patch16-384 coarse filtering (1.7M → 934k)
  - Stage 2: Qwen2.5-VL-7B fine-grained filtering (934k → 846k → 466k after spatial variance check)
- Pseudo-3D scene graph construction：
  - RAM (Recognize Anything) → semantic labels
  - GroundingDINO → bounding boxes
  - UniDepth V2 → metric depth estimation
  - WildeCamera → camera intrinsics
  - SAM 2.1 → instance masks
  - Object-level point clouds → axis-aligned 3D bounding boxes
- Hierarchical descriptions：
  - Qwen2.5-VL生成object/image dense captions
  - Heuristic appends spatial ordering info（"the third cup from left to right"）

**3D Embodied Videos (CA-1M)**：
- Goal：focused spatial understanding of indoor scenes with finer-grained perception
- Sampling：1 frame per 20 frames (减少temporal redundancy)
- Bidirectional 2D bounding box matching：GroundingDINO+RAM predictions ↔ CA-1M annotations
- Gravity alignment → top-down occupancy maps
- Free space sampling：
  - Front/behind/left/right: 90° sector, radius = max(footprint diagonal, 20cm)
  - Above/below: 80% shrink to mitigate overestimation
  - Between: planar area enclosed by two objects' projections
  - Minimum free area: 0.036m² (half A4 sheet)
  - Visibility filtering: 9000 points sampled per direction, retain if ≥2000 visible

**Simulation (Infinigen + Objaverse)**：
- Goal：multi-step referring with reasoning processes
- 3k+ unique indoor scenes (after filtering for tabletop area, lighting, realism, camera accessibility)
- 3k+ curated 3D assets (from 46k, after manual filtering for axis alignment, single object, color diversity, etc.)
- Asset annotation via GPT-4o with OrienText300K orientation data
- Scene population: 3-9 assets per scene
- Blender Cycles rendering: 960×540, 2048 samples
- QA generation with structured thought processes

### 31 Spatial Relations分类

从RefSpatial的visualization (Fig. 36-39)看，31 spatial relations包括：
- Left & Right, Close & Far, Depth
- Above & Below (World), Above & Below (Image)
- Tall & Short, Big & Small, Wide & Thin
- Between, Free Space, Corner & Edge
- Rotation (Horizon & Vertical), Angle
- Front & Behind, Distance
- Face & Back, Touch & Far from
- Inside & Outside

## 6. RefSpatial-Bench评估Benchmark

新benchmark填补multi-step spatial referring评估gap。

### Statistics (Table 11)

**Location Task** (100 samples):
- Step 1: 30 samples, avg prompt length 11.13
- Step 2: 38 samples, avg 11.97
- Step 3: 32 samples, avg 15.28
- Overall avg: 12.78

**Placement Task** (100 samples):
- Step 2: 43 samples, avg 15.47
- Step 3: 28 samples, avg 16.07
- Step 4: 22 samples, avg 22.68
- Step 5: 7 samples, avg 22.71
- Overall avg: 17.68

**Unseen Set** (77 samples with novel spatial relation combinations):
- Step 2: 29, Step 3: 26, Step 4: 17, Step 5: 5
- Avg prompt length: 19.45

### Step定义

每个step对应：
- An explicitly mentioned anchor object
- A directional phrase linked to an anchor that greatly reduces ambiguity

Excludes：
- "Viewer"作为anchor
- "on"（typically refers to implied surface, minimal disambiguation）
- Intrinsic attributes (color, shape, size, image-relative position)

Step ≥3 exhibits substantial spatial complexity。Empirically beyond 5 steps diminishing returns。

### Evaluation Metric

Average success rate of predicted points within ground-truth mask。

## 7. 实验结果深度分析

### Table 1: Single-step Spatial Understanding

对比多类models在CV-Bench, BLINK, RoboSpatial, SAT, EmbSpatial上的表现：

**Proprietary Models**：
- Gemini-2.5-Pro: CV-Bench 93.54/91.00/90.67, BLINK 91.61/87.90
- GPT-4o: 84.62/86.50/83.33, 82.52/78.23
- Claude-3.7-Sonnet: 74.15/85.83/84.17, 74.83/67.74

**Open-Source VLMs**：
- NVILA-8B (base): 91.54/91.83/90.67, 76.92/76.61
- Qwen-2.5-VL-72B: 84.15/86.17/84.15, 78.32/73.55

**Spatial Specialists**：
- SpatialRGPT-8B: 91.00/89.8/88.50, 81.12/89.51
- RoboPoint-13B: 75.85/77.83/44.50, 60.84/61.29

**RoboRefer**：
- 2B-SFT RGB: 96.15/95.83/90.67, 83.92/88.71
- 2B-SFT RGB-D: 96.31/97.17/90.83, 87.41/91.13
- 8B-SFT RGB-D: **96.90/98.33/93.50, 91.61/92.74**

关键takeaway：
- RoboRefer-8B-SFT超过Gemini-2.5-Pro 5% absolute average
- 2B variant超过NVILA-2B base 21.7% absolute
- Depth input带来1.5% relative improvement on 3D benchmarks vs 2D

### Table 2: Multi-step Spatial Referring

**RefSpatial-Bench Location**：
- Gemini-2.5-Pro: 46.96
- Molmo-72B: 45.77
- RoboRefer-2B-SFT: 47.00
- RoboRefer-8B-SFT: 52.00
- RoboRefer-2B-RFT: 52.00

**RefSpatial-Bench Placement**：
- Gemini-2.5-Pro: 24.21
- RoboRefer-2B-SFT: 48.00
- RoboRefer-8B-SFT: 53.00
- RoboRefer-2B-RFT: **54.00**

**RefSpatial-Bench Unseen**：
- Gemini-2.5-Pro: 27.14
- RoboRefer-2B-SFT: 33.77
- RoboRefer-8B-SFT: 37.66
- RoboRefer-2B-RFT: **41.56** (vs 2B-SFT +9.1% absolute, showing RFT generalization)

2B-RFT超过prior SOTA (Gemini-2.5-Pro) 17.4% absolute average on RefSpatial-Bench。

### Table 8: Per-Step Success Rates (Appendix)

**Location Task**：
- Step 1: 2B-SFT 63.33 → 2B-RFT 66.67 (+3.34)
- Step 2: 39.58 → 43.75 (+4.17)
- Step 3: 27.27 → 36.36 (+9.09)
- Total: 47.00 → 52.00 (+5.00)

**Placement Task**：
- Step 2: 55.56 → 55.56 (+0.00)
- Step 3: 41.67 → 41.67 (+0.00)
- Step 4: 41.67 → 45.83 (+4.16)
- Step 5: 0.00 → **25.00** (+25.00!)
- Total: 48.00 → 54.00 (+6.00)

RFT在更高reasoning steps上gain更大——这是非常重要的发现。

### Table 5: Simulation Results (Open6DOR V2)

- Octo: L.1 51.2%, L.2 12.7%, L.3 0.0%, Avg 43.2%, Time 0.04s
- OpenVLA: 51.6/13.1/0.0/43.6/0.04
- SoFar: 75.3/65.6/50.0/72.4/40s
- **RoboRefer**: 79.6/68.4/53.2/79.2/29s

RoboRefer相比SoFar：
- Success rate +6.8% absolute
- Time -27.5% (vs GPT-4o)

### Table 6: Real-world Robot Evaluation

Tasks requiring multi-step spatial referring in cluttered dynamic environments：
- "Pick the specific hamburger closest to the mug nearest the camera and place it in front of the teddy bear"：OpenVLA 0%, RoboPoint 0%, RoboRefer **80%**
- "Pick the apple in front of the leftmost cup's logo side, navigate to the nearest table, and place it aligned with the apple row"：OpenVLA 0%, RoboPoint 0%, RoboRefer **60%**

**只有RoboRefer能handle long-horizon tasks requiring complex multi-step spatial referring**。

### Table 10: Depth Noise Robustness

Real-world evaluation with DepthAnything V2 vs Real Camera depth：

| Task | DepthAnything V2 | Real Camera |
|------|------------------|-------------|
| Pick specific hamburger | 80 | 70 |
| Place hamburger | 90 | 90 |
| Pick apple in front of cup logo | 80 | 80 |
| Place apple aligned with row | 60 | 40 |

DepthAnything V2更robust，但real camera depth下RoboRefer仍能maintain decent performance（利用RGB priors from mixed RGB+RGB-D training）。

## 8. Ablation Studies深度解析

### Table 7: Data Recipe Ablation

| 2D | 3D | Sim | Depth Enc | CV-Bench | BLINKval |
|----|----|----|-----------|---------|----------|
| ✗ | ✓ | ✓ | ✓ | 84.17 | 74.48 |
| ✓ | ✗ | ✓ | ✓ | 81.83 | 74.61 |
| ✓ | ✓ | ✗ | ✓ | 83.96 | 75.10 |
| ✓ | ✓ | ✓ | ✗ | 91.24 | 85.27 |
| ✓ | ✓ | ✓ | ✓ | **94.77** | **89.27** |

关键insights：
- 去除2D数据：severely degrades performance on outdoor-centric BLINK
- 去除3D数据：hurts indoor-focused CV-Bench (no Sim2Real gap mitigation)
- 去除simulated data：reduces spatial diversity
- 去除depth encoder：drops from 94.77 to 91.24 on CV-Bench

### Table 7: RFT Reward Ablation

| Process Reward | Depth Enc | RefSpatial-Bench |
|----------------|-----------|-------------------|
| ✗ | ✗ | 40.00 |
| ✗ | ✓ | 48.00 |
| ✓ | ✓ | **53.00** |

Process reward带来5-point improvement，validating其importance。

### Table 16: RGB/RGB-D Combination for SFT

只用RGB-D vs RGB+RGB-D combination：
- 只用RGB-D inference: 87.69/86.83/82.50/79.02/81.45
- Combination, RGB inference: 96.15/95.83/90.67/83.92/88.71
- Combination, RGB-D inference: **96.31/97.17/90.83/87.41/91.13**

Combination training让image encoder也能从RGB alone学习spatial cues，避免over-reliance on depth encoder。

## 9. Real-World System Integration

### UR5 Manipulation

- RoboRefer runs at **2.5Hz**，enabling reactive updates
- Significant shifts in predicted 2D coordinates trigger motion interruption + re-planning
- Grasping pipeline：
  1. RoboRefer预测2D point
  2. SAM2生成segmentation mask
  3. Filter target object point cloud from RealSense L515
  4. AnyGrasp预测grasp pose in camera frame
  5. Eye-to-hand calibration转换到UR5 base frame
- Placement：
  1. RoboRefer预测2D placement point
  2. Camera intrinsics + depth → 3D coordinates
  3. Transform到robot coordinate system

### G1 Humanoid Mobile Manipulation

- Head-mounted Intel RealSense D435
- Chest-mounted L515 for navigation
- SLAM via FAST_LIO_LOCALIZATION_HUMANOID
- Spatial referring unifies manipulation + navigation under single formulation

### Voice Interruption Demo

Whisper ASR transcribes speech → RoboRefer processes into new 2D coordinates → task redirection。这enables真正interactive的human-robot collaboration。

## 10. 关键创新总结与Intuition

### Innovation 1: Dedicated Depth Encoder

Insight：modality interference是shared encoder的fundamental问题。Dedicated encoder虽然简单，但preserves pretrained image encoder weights + enables independent depth learning。

### Innovation 2: Metric-Sensitive Process Reward

Insight：不同spatial attributes有fundamentally different representations（points, vectors, scalars），需要不同metrics。这比generic process reward更精确。

### Innovation 3: Order-Invariant Reasoning Reward

Insight：spatial referring的reasoning过程naturally不sequential——先identify keyboard还是mouse对最终结果无影响。这比strict sequential reasoning更符合spatial cognition的nature。

### Innovation 4: Bottom-Up Data Recipe

Insight：从2D broad concepts → 3D indoor fine-grained → simulation multi-step reasoning，progressively build up capabilities。每个data source addresses不同limitation。

### Innovation 5: Point-Based Formulation

Insight：point比bbox更适合robotics——maps自然到3D via depth，handles occlusion via part localization，unifies navigation/manipulation/placement under single representation。

## 11. Limitations与Future Work

### Limitation 1: Human Intent Understanding

Current model依赖precise textual descriptions。Real human instructions往往ambiguous。Paper给出两个examples (Fig. 35)：

**Probabilistic Preference**："pick the one facing the drink" — 4个drink bottles中只有中间2个align with second sushi plate，但人通常基于probabilistic bias选择该plate。

**Spatial Compatibility**："place another sushi between the plate and soy sauce dish" — 多个plate-soy sauce pairs中只有closest to observer那对有enough physical space。

### Limitation 2: 3D Perception

- 主要依赖qualitative spatial relations (left, right)
- 预测2D image-plane coordinates，需depth-based conversion到3D
- Future：直接model quantitative geometry, predict 3D points + visual traces

### Future Direction 1: Intent-Aware Data

Procedural synthesis of intent-aware data，或co-training with intent-rich datasets like PixMo-Points。

### Future Direction 2: Direct 3D Reasoning

直接predict 3D points和visual traces，bypass 2D-3D conversion。

## 12. 与Related Work的深度对比

### vs SpatialRGPT

| 维度 | SpatialRGPT | RoboRefer |
|------|-------------|-----------|
| Task | VQA with region input | Spatial referring with text input |
| Model Usage | Needs masks/detection tools | Text-only object reference |
| Data Pipeline | 2D-only | 2D + 3D + Simulation progressive |
| Training | SFT only | SFT + RFT |
| Encoder | Shared | Dedicated depth encoder |
| Reasoning | Single-step | Multi-step up to 5 steps |

### vs RoboPoint

RoboPoint incorporates basic spatial cues via images，但struggles with complex environments和multi-step reasoning。RoboRefer通过RFT实现了generalization到novel spatial relation combinations。

### vs 3D-LLM

3D-LLM demands costly 3D reconstruction of multi-view images，causing modality gaps。RoboRefer用single-view RGB-D + dedicated depth encoder避免此问题。

### vs Molmo

Molmo-72B在RoboRefIt上表现不错（74.2），但在3D spatial relations involving distance (e.g., "second-farthest object")上struggles，reducing performance when multiple spatial constraints combined。

## 13. 技术细节联想与Open Questions

### 关于Reward Design的进一步思考

α = 0.25的choice很关键。Process reward被scaled down by 0.25防止reward accumulation from multi-step processes。但这个值是empirical的，更systematic的ablation会更有说服力。

50 pixels的L1 threshold也很magic。For 448×448 resolution image，50 pixels ≈ 11% of image dimension。这个threshold在不同benchmark上是否optimal？

### 关于Order-Invariance的Theoretical Justification

Order-invariance是spatial referring特有的吗？For mathematical reasoning，order matters a lot。For spatial reasoning，为什么order不matter？可能因为spatial relations是symmetric的（A is left of B ⟺ B is right of A），不像mathematical derivation有strong dependency。

### 关于Depth Encoder初始化

从RGB encoder weights初始化depth encoder是合理starting point，但depth和RGB的statistics很不同。是否需要更sophisticated initialization（比如depth-specific pretraining）？或者depth encoder训练更久？

### 关于Sim2Real Gap

Simulation data用于multi-step reasoning训练，但simulated scenes的visual realism和real-world差距。Paper用gravity alignment、光照randomization等mitigate，但更深层的sim2real问题（texture, material appearance）如何处理？

### 关于Long-Horizon Task Execution

Real-world experiments展示long-horizon tasks，但实际执行pipeline是open-loop（2.5Hz updates）。For truly dynamic environments with moving objects, closed-loop with continuous feedback会更robust。如何integrate closed-loop control with RoboRefer's reasoning?

### 关于Multimodal Co-training Effect

Table 4显示joint RGB+RGB-D training preserves commonsense knowledge。但更deep的问题是：depth information如何interact with RGB information in LLM's representation space？是否internal representation encode了3D structure，还是仍surface-level pattern matching？

## 14. 实现细节的Practical Considerations

### Compute Resources (Appendix E.1)

整个pipeline的compute成本：
- 2D filtering: 1 node × 8.5h (SigLIP2) + 1 node × 2.5 days (Qwen2.5-VL)
- Pseudo-3D scene graphs: 3 nodes × 10h (depth) + 10h (other) + 4 nodes × 18h (captions)
- Reasoning QA generation: 4 nodes × 3.75 days (QwQ-32B for 2D) + 4 nodes × 1.5 days (for 3D)
- Synthetic data: 4× RTX 4090 × 1 week
- SFT depth alignment: 10 nodes × 12h (2B) / 8 nodes × 40h (8B)
- SFT spatial understanding: 10 nodes × 2 days (2B) / 10 nodes × 1 week (8B)
- RFT: 1 node × 3 days (2B)

Total粗略估算：~200+ A100 GPU days。这是非常expensive的training，replicability挑战大。

### RFT Training Bottleneck

RoboRefer RFT比Qwen2.5-VL-based methods (R1-V, VLM-R1)慢2x，因为：
- RGB-D input需要modified NVILA architecture
- Incompatible with vLLM/SGLang group inference acceleration

这是engineering bottleneck，未来通过better inference framework可缓解。

## 15. Final Thoughts

RoboRepresent一个重要方向：**将RL-based reasoning从text-only扩展到spatial-visual domain**。它展示了几个关键insights：

1. **Modality-specific encoding matters**：dedicated depth encoder > shared encoder
2. **Process reward with metric sensitivity**：不同spatial attributes需要不同metrics
3. **Order-invariance enables flexible reasoning**：spatial reasoning天然non-sequential
4. **Bottom-up data curation**：2D → 3D → simulation progressive design
5. **Point formulation unifies robotics tasks**：navigation, manipulation, placement under single representation

但也有一些open questions值得future work探索：
- Human intent understanding beyond precise descriptions
- Direct 3D reasoning without 2D-3D conversion
- Closed-loop integration for truly dynamic environments
- More sophisticated depth encoder initialization
- Theoretical understanding of why order-invariance works for spatial reasoning

这个工作为spatial intelligence in embodied AI建立了新的benchmark，特别是multi-step spatial reasoning with RL-based training。期待后续工作能address human intent、direct 3D、closed-loop等open questions。

参考资源：
- RoboRefer project: https://zhoues.github.io/RoboRefer
- GRPO paper (DeepSeekMath): https://arxiv.org/abs/2402.03300
- NVILA: https://arxiv.org/abs/2412.04468
- R1-V: https://arxiv.org/abs/2503.10617
- VLM-R1: https://arxiv.org/abs/2504.07615
- DepthAnything V2: https://arxiv.org/abs/2406.09414
- SAM 2: https://arxiv.org/abs/2408.00714
- UniDepth V2: https://arxiv.org/abs/2502.18937
- Infinigen Indoors: https://arxiv.org/abs/2406.11824
- CA-1M: https://arxiv.org/abs/2412.04458
- Open6DOR: https://arxiv.org/abs/2410.05171
- SoFar: https://arxiv.org/abs/2501.08481
