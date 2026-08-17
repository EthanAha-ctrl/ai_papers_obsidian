---
source_pdf: MolmoAct2_ Action Reasoning Models for Real-world Deployment.pdf
paper_sha256: 1734fb13d01e8145371f73d0b3b4ec4a55a1ff854720b800b27d7067f40796ce
processed_at: '2026-08-05T20:05:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MolmoAct2

---

## 一句话总结

AI2 做了一个**完全开源**的 robot 大脑，能直接装到三种便宜机器人上干活，性能吊打 π0.5，还能"想一想再动手"但速度快得离谱。

---

## 为什么这事难

先想象你要造一个机器人保姆。你给它一个指令："把桌上的苹果放进碗里"。它需要：

1. **看懂**桌子在哪、苹果在哪、碗在哪、它们之间多远
2. **想清楚**先伸手去抓苹果、抓多紧、往哪移
3. **动手**控制 14 个关节协调运动

现在业界的做法各有各的坑：

**π0.5 这种**：很厉害，但代码和数据都不给你，你没法改、没法在自己的机器人上用。

**带 reasoning 的那种**：每次动手前先生成几百个 token 的"思考链"，或者预测一张"未来图"。等你算完，苹果都被人吃了。

**OpenVLA 这种**：开源了，但只能跑在 Franka 上（一台好几万美金），普通 lab 用不起。

**Fine-tune 之后**：成功率还是不够高，没法真正 deploy。

MolmoAct2 想把这四个坑一次全填上。

---

## 他们怎么做的——五个核心 idea

### 1. 给 VLM 补"空间感"

通用 VLM（比如 GPT-5）看图很厉害，但问它"这个杯子离机器人多远"、"手往左挪 10 厘米会碰到啥"，它答得很烂。因为这些技能在 web 数据里几乎没有。

Molmo2-ER 的做法：拿 Molmo2 当底子，喂 330 万条**专门练空间感**的数据——距离判断、视角切换、free space 指认、ego-exo 对应。先猛练专项（Stage 1），再把原来的通用能力混回来排练一遍（Stage 2），防遗忘。

结果：在 13 个 embodied reasoning benchmark 上赢了 9 个，把 GPT-5 和 Gemini Robotics ER 都压下去了。

**人话**：就好比一个很聪明的文科生，脑子好使但没学过物理。你给他补半年几何和空间课，他再看真实世界就"开眼"了。

---

### 2. 三个新数据集

VLA 要 data，而且要又多又杂又准。

**YAM Bimanual**：720 小时双臂数据，28 种任务，从折衣服到打包药。整套硬件不到 6000 美金。这是目前最大的开源双臂数据集。

**SO-100/101**：从 LeRobot 社区 1222 个 dataset 里筛的。377 个用户贡献，背景杂、对象杂、布置杂。先用四道过滤（结构合法性、移除 eval 集、license 检查、TOPReward 质量门），留下干净的。

**DROID filtered**：DROID 原始数据质量参差，他们用 extended language annotations 和 idle-frame filter 筛了一遍，还重新做了语言标注。

**Language re-annotation 的小故事**：原始标注烂得离谱。BC-Z 这个数据集 39350 条 episode 只有 104 条 unique instruction（0.26%）。SO-100 社区数据里到处是 "lerobot_test" 这种废话。他们用 Qwen3.5-27B 重新标注，unique labels 直接翻倍。

**人话**：好模型要吃好 data。他们既自己种有机蔬菜（YAM），又从农贸市场淘货筛选（SO-100），还把烂标签重贴了一遍。

---

### 3. OpenFAST Tokenizer

Robot action 是连续的浮点数（14 个关节角度），但 LLM 只认 token。怎么桥接？

FAST 的思路：把 1 秒的动作轨迹做离散余弦变换（DCT），量化系数，再用 BPE 压成 2048 个 token 的 vocabulary。1 秒的连续动作变成几个 token。

MolmoAct2 做了开源版，训练数据透明：YAM/SO/DROID 各 30%，剩下给小数据集拓宽控制模式。所有动作先 pad 到 32 维，做 1-99 percentile normalization，gripper 单独处理。

**人话**：就像把一段音乐压成 MIDI 音符序列。原来要存 30 个浮点数（每秒 30 帧控制），现在只要几个 token。LLM 就能像处理文字一样处理动作了。

---

### 4. Per-layer KV Connection（架构核心）

这是全篇最有想象力的创新。

**传统做法**：VLM 是个 36 层的 Transformer，最后一层输出一个 hidden state，action expert 只看这个。就好比让一个将军（VLM）做完所有侦察分析，最后只给前线士兵（expert）一句口令。信息瓶颈。

**MolmoAct2 的做法**：VLM 每一层的 Key 和 Value cache 都单独投影出来，给 action expert 的对应层用。Expert 有自己的 36 层，每层都能 cross-attend 到 VLM 同深度的 KV。

浅层能看到 edge、texture 这些 visual primitives，深层能看到 semantic 概念。Expert 在每个深度都能"问 VLM"。

更妙的是**detach**：expert 的梯度不回传到 VLM。VLM 被 language loss 训（predict discrete action tokens），expert 被 flow loss 训（generate continuous trajectory）。两条路并行不干扰，但 expert 能"读"VLM 的全部注意力状态。

**Flow matching 怎么生成 action**：从纯噪声出发，预测一个速度场，积分到目标动作。训练时让 expert 预测从 noise 到 data 的方向。每个 action chunk 采 4-8 个噪声样本，重用同一个 VLM context，提高效率。

**人话**：想象 VLM 是个侦察营，36 个侦察兵分别报告不同层级的情报（从"左边有个红色物体"到"那是个杯子可以抓"）。Action expert 是前线战斗班，每个战斗员能直接对口问对应层级的侦察兵。以前是侦察营长听完所有汇报，只传一句命令给战斗班。现在是 mesh 通信，信息带宽大了几十倍。

---

### 5. MolmoAct2-Think：Adaptive Depth（延迟杀手）

**问题**：MolmoAct 前作让模型先预测一张 10×10 的 depth map（100 个 token）再生成 action。几何 grounding 有了，但每步都要生成 100 个 token，太慢。

**观察**：机器人干活时，大部分场景是静态的。机械臂在动，但背景的墙、桌子、碗都不动。100 个 depth cell 里可能 90 个没变。

**Adaptive 方案**：
- 把图像切成 10×10 个 32×32 的 patch
- 比较当前帧和上一帧对应 patch 的 RGB cosine 相似度
- 低于 0.996 的才标记为"变了"，重新预测 depth code
- 没变的直接 replay 上次的 cache

推理延迟和**场景变化量**成正比，不再是固定 100 token。第三人称视角（大部分静态）收益最大。

**Training 的两个技巧**：
1. **Depth noise injection**：训练时 10% 的 depth token 随机替换成噪声。因为推理时模型自己预测的 depth 会有误差，训练时先"见过噪声"能鲁棒化。
2. **Depth gate**：每层一个 sigmoid gate，bias 初始化 -4（sigmoid(-4)≈0.018，几乎关死）。训练从不用 depth 开始，慢慢学着用。避免一开始 depth 的噪声就把 action 带歪。

**人话**：你在厨房洗碗，不需要每秒重新画整张厨房的 3D 图。只有水龙头动了一下、碗移了个位置，那两个小区域重画就行，其他地方用记忆。人脑就是这么干的——你不会每秒重新感知整个房间。

---

## 训练 pipeline 全貌

```
Molmo2-ER（VLM backbone）
    ↓ 补空间感
MolmoAct2-Pretrain（200K steps）
    ↓ 学会预测离散 action tokens
MolmoAct2-Post（100K steps）
    ↓ 挂上 flow matching action expert + per-layer KV
MolmoAct2-Finetune（50-100K steps）
    ↓ 针对具体机器人微调
部署到 YAM / DROID / SO-100
```

三个阶段加起来约 9000 GPU hours（64 H100）。全程 open weights + open data + open code。

---

## 结果有多强

### Embodied Reasoning（Molmo2-ER）

| Model | Avg |
|---|---|
| GPT-5 | 57.9% |
| Gemini Robotics ER 1.5 Thinking | 61.3% |
| **Molmo2-ER** | **63.8%** |

### 真实机器人 DROID（15 trials/task）

| Model | Avg |
|---|---|
| π0.5-DROID | 45.2% |
| MolmoBot | 48.4% |
| **MolmoAct2-DROID** | **87.1%** |

87.1% vs 45.2%，差不多翻倍。

### SO-100/101

| Model | Avg |
|---|---|
| SmolVLA | 2.3% |
| π0-SO100/101 | 45.3% |
| **MolmoAct2-SO100/101** | **56.7%** |

### LIBERO（仿真）

| Model | Avg |
|---|---|
| OpenVLA | 76.5% |
| π0.5 | 96.9% |
| GR00T N1.7 | 97.0% |
| **MolmoAct2** | **97.2%** |
| **MolmoAct2-Think** | **98.1%** |

### 真实双臂 YAM（8 task × 50 trials）

MolmoAct2 avg 50.1%，比第二名 OpenVLA-OFT 高 15 个点。

### Inference 速度（H100, horizon 10）

- **MolmoAct2 原始**：23 Hz
- **+ caching + CUDA Graph**：**55.79 Hz**

55.79 Hz 意味着每秒能出 5.6 个 action chunk，每个 chunk 覆盖 10 步控制，实际控制频率绰绰有余做闭环。

MolmoAct2-Think 因为 adaptive depth 的 autoregressive 部分，CUDA Graph 收益小，只到 12.71 Hz。但 Think 版本主要追求质量提升，速度够用。

---

## 为什么这些设计能 work

### Backbone 特化能 transfer 到 action

Table 9 是关键 ablation：同样架构，只用 discrete FAST tokens，换 backbone：
- Molmo2：77.6%
- Molmo2-ER：83.6%

**+6 个点纯来自 VLM 的空间感提升**。证明 embodied reasoning 特化不只刷 benchmark，直接 transfer 到 policy 质量。

### Per-layer KV > Final hidden

Table 10：per-layer KV 95.9% vs hidden state 94.0%。差 2 个点看似不多，但这是在所有其他条件相同的情况下。信息带宽的差距确实体现在性能上。

### Discrete + Continuous 共训

Table 12 显示，只训 action expert（不碰 VLM）只有 93.05%，full fine-tuning + discrete co-training 是 97.20%。离散 action loss 提供额外 regularization，让 VLM 保持对 action space 的理解。

### Adaptive Depth 的价值

Table 8：Think 版本在 LIBERO Long（最难）上 +2.2 个点。越难的 task，depth grounding 越有用。简单 task 已经接近 100% ceiling，提升空间小。

### Trajectory 质量

Figure 6 的 RoboEval：MolmoAct2 不仅 success rate 高，轨迹还更短（joint path length 减半）、更稳（jerk 更低）、更少 self-collision。这意味着真 deploy 时更安全、更省电、更少磨损。

---

## 这篇 paper 的意义

**对学术圈**：第一个 fully open 的 frontier VLA。Data、code、weights 全有，任何一个 PhD 学生都能复现、改进、扩展。之前 π 系列只给 weights 不给 recipe，社区只能猜。

**对机器人圈**：三个 embodiment（YAM < $6k、SO-100 更便宜、DROID Franka）覆盖 low-to-medium cost。不再只有有钱 lab 才能玩 VLA。

**对架构研究**：Per-layer KV connection 是 VLM-to-expert interface 的新范式。之前要么 final hidden bottleneck，要么把 expert 塞进 VLM。现在 expert 能 dense 地访问 VLM 各层 attention state，又通过 detach 保持训练稳定。

**对 reasoning 研究**：Adaptive depth 证明了 reasoning-augmented policy 不一定要付 prohibitive latency。利用 trajectory 的时间冗余，reasoning cost 可以和 scene change 成比例。这给未来"thinking robot"指出了一条路。

---

## 可以挑刺的地方

**Depth Anything V2 的误差**：monocular depth 对透明物体、反光表面、occlusion 都不准。MolmoAct2-Think 的 grounding 质量被这个上限卡住。未来用 active depth sensing 或 multi-view stereo 可能更好。

**28 个 YAM task**：相对真实世界多样性还是少。折衣服、打包药这些 task 的 variability 不足以覆盖所有 household 场景。

**12.71 Hz for Think**：虽然够用，但比 55.79 Hz 差很多。Adaptive depth 的 eager execution 是瓶颈。未来需要 dynamic shape 的 CUDA Graph 或编译优化。

**K=8 的 memory 限制**：fine-tuning 时 8 个 flow sample 是 GPU memory 撑着，没法更大。更长 horizon 或更复杂 action space 可能受限。

**No explicit failure analysis**：paper 报告 success rate 但没深入分析 failure mode。哪些 task 失败、为什么失败、是 perception 还是 control 还是 planning 的锅，不清楚。

---

## Reference Links

- Project: https://allenai.org/blog/molmoact2
- Code: https://github.com/allenai/molmoact2
- MolmoAct (前作): https://arxiv.org/abs/2508.07917
- Molmo2: https://arxiv.org/abs/2601.10611
- FAST: https://arxiv.org/abs/2501.09747
- π0.5: https://arxiv.org/abs/2504.16054
- DROID: https://arxiv.org/abs/2403.12945
- LIBERO: https://arxiv.org/abs/2311.04064
- RoboEval: https://arxiv.org/abs/2507.00435
- RoboPoint: https://arxiv.org/abs/2406.10721
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Flow Matching: https://arxiv.org/abs/2210.02747
- SigLIP2: https://arxiv.org/abs/2502.14786
- RefSpatial: https://arxiv.org/abs/2506.04308
- SAT: https://arxiv.org/abs/2412.07755
- RoboVQA: https://arxiv.org/abs/2311.00899
- GR00T N1: https://arxiv.org/abs/2503.14734
- Ego-Exo4D: https://arxiv.org/abs/2311.18259
- SmolVLA: https://arxiv.org/abs/2506.01844
- Tulu 3: https://arxiv.org/abs/2411.15124
- Knowledge Insulation: https://arxiv.org/abs/2505.23705

---

总结一句：MolmoAct2 把 VLA 的四个老大难问题（闭源、延迟、贵硬件、成功率低）用一套组合拳系统性解决了。Per-layer KV connection 和 adaptive depth 是两个最有想象力的设计——前者打通了信息瓶颈，后者让 reasoning 变得经济。全开源意味着接下来半年社区能在这个底子上玩出花来。

---

# MolmoAct2 深度技术解析

Andrej，这篇 paper 是 AI2 团队的重磅工作，我尽量把每个技术细节拆开讲，帮你 build intuition。

---

## 1. 核心问题：VLA 模型的四个痛点

paper 开篇直接指出当前 VLA (Vision-Language-Action) 模型的四大缺陷，这构成了 MolmoAct2 的设计动机：

1. **Frontier VLA 闭源**：π 系列、Gemini Robotics 只释放 weights，data 和 recipe 都保留
2. **Reasoning-augmented policy 延迟爆炸**：要先生成几百个 token 或整帧预测图像才能 emit 一个 action
3. **Open-weight VLA 绑定昂贵硬件**：大部分只能在 Franka 这种平台上跑，学术 lab 用不起
4. **Fine-tune 后 success rate 依然不够**：达不到 dependable deployment 的 threshold

MolmoAct2 在五个轴上做改进：**VLM backbone**、**数据集**、**tokenizer**、**架构**、**reasoning paradigm**。

---

## 2. Molmo2-ER：为 Embodied Reasoning 特化的 VLM Backbone

### 2.1 核心洞察

通用 VLM 对 metric distance、free space、cross-view object tracking、scene geometry 这些 embodied 技能**under-specify**。Molmo2-ER 的目标是把 Molmo2 这个 base VLM 特化到 spatial 和 embodied reasoning 上。

### 2.2 数据组成（Table 1）

Molmo2-ER corpus 约 **3.3M samples**，分六个能力支柱：

| Pillar | #Samples | Weight |
|---|---|---|
| Image Embodied QA | 1.33M | 0.11 |
| Image Pointing | 780K | 0.11 |
| Image Detection | 100K | 0.01 |
| Video Embodied QA | 703K | 0.10 |
| Multi-image/Ego-Exo | 700K | 0.09 |
| Abstract Reasoning | 150K | 0.04 |

每个 pillar 都用 2-3 个 dataset 覆盖互补的 supervision 来源：
- **SAT** (Ray et al., 2025)：simulator-grounded 的 dynamic reasoning
- **RefSpatial** (Zhou et al., 2026)：CoT referring，250K MCQ + 250K CoT + 80K pointing
- **VST-P** (Yang et al., 2025c)：统一虚拟相机视角，给出 metric-consistent depth/distance/direction/size
- **VSI-590K** (Yang et al., 2025d)：3D-grounded label propagation
- **SIMS-VSI** + **RoboVQA**：video embodied QA 的两端（simulator clean labels vs. human-annotated long-horizon）
- **RoboPoint** pointing 700K + LVIS detection 100K
- **SenseNova-SI** (Cai et al., 2026)：multi-image 和 ego-exo correspondence
- **CLEVR** + **GRiD-3D**：compositional reasoning 和 frame-of-reference distinction

### 2.3 Specialize-then-Rehearse 两阶段训练

**Stage 1: Embodied specialization**
- 从 Molmo2 (Qwen3-4B) mid-training checkpoint 出发
- Fine-tune 20K steps on Molmo2-ER corpus + 8% Tulu-3 text-only data（防止 language forgetting）
- Sequence length 4200，global batch size 64（device batch 4 × 2 nodes × 8 H100）
- 这阶段快速把模型移到 embodied data manifold

**Stage 2: Joint refinement**
- 继续训练 1.5K steps
- Mixture: interleaves embodied corpus 和 Molmo2 原始 multimodal mid-training data
- NLP rate 保持 8%，剩下 92% 分配为 $p \cdot 0.92$ embodied + $(1-p) \cdot 0.92$ general
- Sweep $p \in \{0.30, 0.50, 0.70, 0.90\}$，发现 $p = 0.5$ 是最佳 Pareto trade-off
- Sequence length 增加到 16384（容纳 multi-image 和 long-video），device batch 减到 1

**Intuition**：Stage 1 是 "overspecialization"，Stage 2 是 "rehearsal" 防止 catastrophic forgetting。两阶段都用同一个 NLP 比例保持语言能力，这是个很干净的 curriculum 设计。

### 2.4 结果（Table 3）

Molmo2-ER 在 13 个 embodied reasoning benchmark 上：
- **9/13 benchmark 上胜出**
- Overall average **63.8%**，超过 GR-ER 1.5 Thinking 的 61.3% 和 GPT-5 的 57.9%
- 比 base Molmo2 提升 **17 个百分点**

关键 benchmark 包括：Point-Bench、RefSpatial、RoboSpatial-Point、Where2Place、BLINK、CV-Bench、ERQA、EmbSpatial、MindCube、SAT、OpenEQA、VSI-Bench。

---

## 3. 数据：三大新数据集

### 3.1 MolmoAct2-BimanualYAM Dataset

**规模**：720 小时 teleoperated bimanual trajectories，34.5k demonstrations，**迄今最大开源 bimanual 数据集**。

**硬件**：YAM (Yet Another Manipulator)，整个 setup **<$6,000 USD**，2 个月收集完成。

**任务多样性**：28 个 unique real-world tasks，涵盖折叠衣服、解电缆、收桌子、扫描杂货、打包药物等。

**质量控制**：严格的 failure retry 数量和 no-op segment 时长限制。

### 3.2 MolmoAct2-SO100/101 Dataset

从 **1222 个 LeRobot public datasets**（377 用户贡献）中过滤：
- 原始：38,059 episodes，19.8M frames，184 小时
- 四阶段过滤：
  1. 结构合法性检查
  2. 移除 eval-style datasets
  3. License/codebase 合格性
  4. **TOPReward quality gate**（Chen et al., 2026）：用 mean TOPReward 阈值过滤

**Intuition**：community-sourced data 提供 background、object、task annotation 的多样性，比中心化收集的数据集泛化更好。

### 3.3 MolmoAct2-DROID Dataset

基于 DROID (Khazatsky et al., 2024) 的 Franka subset：
- 用 extended language annotations（95% episodes 有 3 条 instruction）
- Idle-frame filter：保留至少 1 秒的连续 non-idle 段
- 结果：74,604 valid episodes，17.76M frames
- 还做了 language re-annotation

### 3.4 Language Annotation Pipeline

发现两个问题：
- 重复 instruction（BC-Z 有 104 unique instructions for 39350 episodes，仅 0.26%）
- Crowd-sourced 数据 annotation 不准（如 "lerobot_test"、"Test run"）

**解决方案**：用 Qwen3.5-27B re-annotate，prompt 包含 frames + 原 instruction，随机要求 word count 增加多样性。结果：unique labels 从 71,121 (22%) 增加到 146,485 (46%)，**翻倍**。

---

## 4. MolmoAct2 架构：核心创新

### 4.1 三阶段训练 pipeline

```
Molmo2-ER → Pretrain → Post-train → Fine-tune (per-embodiment)
```

### 4.2 OpenFAST Tokenizer

**问题**：Robot actions 是 continuous、embodiment-specific、不同 control rate，无法直接插入 LM pre-training。

**方案**：基于 FAST (Pertsch et al., 2025) 的开源实现，把 1 秒 action trajectory 转成 discrete tokens。

**流程**：
1. Frequency-domain transform（DCT）
2. Quantize coefficients
3. Byte-pair encoding → 2048-token vocabulary

**Embodiment mixture**（Table 2）：
- YAM 30%、SO-100/101 30%、DROID 30%
- Fractal 3.33%、BC-Z 3.33%、Bridge 3.33%

**统一表示**：
- 每条 sequence = 1 秒动作
- Action padded 到 32 维（不同 embodiment 共享 input space）
- 1-99 percentile normalization（限 outlier 影响但保留 dynamic range）
- Gripper 单独处理（binary 或 narrow-range）

### 4.3 Pre-training Recipe

- 初始化自 Molmo2-ER checkpoint
- Vision encoder: **SigLIP2 ViT** (Tschannen et al., 2025)
- Connector: 用 ViT 第三层和第九层 features，image 用 2×2 pooling，video 用 3×3 pooling（减少 token 数）
- Pooling 后用 MLP 投影到 LLM embedding space

**数据配比**：
- 10% multimodal data
- 90% robot trajectories
- Robot 内部：YAM/SO/DROID 各 30%，剩下 10% 给 BC-Z、BridgeData V2、RT-1、MolmoAct Dataset

**训练细节**：
- 200K steps，max sequence length 4200
- Vision encoder + connector LR: $5 \times 10^{-6}$
- LLM LR: $1 \times 10^{-5}$
- Global batch 128，64 H100，约 5760 GPU hours
- **On-the-fly packing**：多个短 example 合并到一个 4200-token sequence，但 attention mask 隔离

**Robot prompt 格式**：
```
<setup_start>bimanual yam robotic arms in molmoact2<setup_end>
<control_start>absolute joint pose<control_end>
[state tokens]
<action_output>
```

### 4.4 Post-training：Per-layer KV Connection（核心架构创新）

#### 4.4.1 Flow Matching Action Expert

给定 normalized target action chunk $a$，Gaussian noise $\epsilon$，sampled time $t \in [0,1]$：

$$x_t = (1-t)\epsilon + t a, \quad u^{\star} = a - \epsilon \tag{1}$$

这里：
- $x_t$：noisy action chunk（noise 和 data 的线性插值）
- $t$：flow time，$t=0$ 纯噪声，$t=1$ 纯数据
- $u^{\star}$：target velocity field（从 noise 到 data 的方向）

Loss function：

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{a,\epsilon,t}\left[\left\| m \odot \left(f_{\theta}(x_t, t, c) - u^{\star}\right)\right\|_2^2\right] \tag{2}$$

变量说明：
- $f_{\theta}$：DiT-style action expert 网络
- $c$：VLM context（task、visual observations、setup/control descriptors、discrete state tokens）
- $m$：mask tensor，屏蔽 padded action steps 和 padded action dimensions
- $\odot$：element-wise 乘法

**Inference**：从 Gaussian noise 出发，积分预测的 velocity field 得到 continuous action trajectory。

#### 4.4.2 Action Expert 架构（公式 3-5）

Expert 和 VLM 同深度，都用 $L = 36$ 层。每个 block 包含 self-attention、cross-attention 到 VLM、MLP，时间 embedding 产生 DiT-style shift/scale/gate 参数：

$$h_{\ell}^{\prime} = h_{\ell} + g_{\ell}^{\text{sa}} \text{SA}(\text{AdaRMS}_{\ell}^{\text{sa}}(h_{\ell}, t)) \tag{3}$$

$$\bar{h}_{\ell} = h_{\ell}^{\prime} + g_{\ell}^{\text{ca}} \text{CA}(\text{AdaRMS}_{\ell}^{\text{ca}}(h_{\ell}^{\prime}, t), \tilde{K}_{\ell}, \tilde{V}_{\ell}) \tag{4}$$

$$h_{\ell+1} = \bar{h}_{\ell} + g_{\ell}^{\text{ff}} \text{MLP}(\text{AdaRMS}_{\ell}^{\text{ff}}(\bar{h}_{\ell}, t)) \tag{5}$$

变量：
- $h_{\ell}$：第 $\ell$ 层的 hidden state
- $g_{\ell}^{\text{sa}}, g_{\ell}^{\text{ca}}, g_{\ell}^{\text{ff}}$：分别给 self-attention、cross-attention、MLP 的 gate
- AdaRMS：Adaptive RMSNorm，由 time embedding $t$ 调制
- $\tilde{K}_{\ell}, \tilde{V}_{\ell}$：从 VLM 投影来的 key/value

#### 4.4.3 Per-layer KV Connection（公式 6-7）

**核心创新**：连接 VLM 和 expert 的不是 hidden state，是 **per-layer KV-cache**。

对 VLM 第 $\ell$ 层，收集 self-attention 产生的 keys 和 values $(K_{\ell}^{\text{vlm}}, V_{\ell}^{\text{vlm}})$，投影到 expert 宽度：

$$\tilde{K}_{\ell} = \text{reshape}(P_K K_{\ell}^{\text{vlm}}), \quad \tilde{V}_{\ell} = \text{reshape}(P_V V_{\ell}^{\text{vlm}}) \tag{6}$$

- $P_K, P_V$：linear adapter projections（VLM-to-expert），独立于 VLM self-attention 的 projection
- reshape：把投影后的 cache 组织到 expert 的 attention heads

Cross-attention：

$$\text{CA}(Q_{\ell}, \tilde{K}_{\ell}, \tilde{V}_{\ell}) = \text{softmax}\left(\frac{Q_{\ell} \tilde{K}_{\ell}^{\top}}{\sqrt{d_h}}\right)\tilde{V}_{\ell} \tag{7}$$

- $Q_{\ell}$：expert 的 query
- $d_h$：expert head dimension

**Intuition**：
- Per-layer 让 expert 每 block 直接访问**同深度**的 VLM attention state
- 比 final-hidden-state conditioning 信息丰富得多（dense hierarchical features）
- Detach 这条 path（knowledge insulation）：flow loss 只更新 expert 和 adapter，不 backprop 到 VLM

#### 4.4.4 Post-training Loss

$$\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{flow}} \tag{9}$$

- $\mathcal{L}_{\text{LM}}$：pre-training 的 next-token prediction（包括离散 action tokens）
- $\mathcal{L}_{\text{flow}}$：continuous action chunk 的 flow matching

**Multiple flow samples**：每个 action chunk 抽 $K$ 个独立 noise-time pair：

$$\mathcal{L}_{\text{flow}}(a, c) = \frac{1}{K}\sum_{i=1}^{K}\left\| m \odot \left(f_{\theta}(x_{t_i}, t_i, c) - (a - \epsilon_i)\right)\right\|_2^2 \tag{8}$$

- $K = 4$（post-training），$K = 8$（fine-tuning）
- 同一 VLM context 重用 $K$ 次，提高 data efficiency

**Knowledge insulation**：
- Expert 的 cross-attention 条件化 VLM KV cache，但 cache 是 **detached**
- $\mathcal{L}_{\text{flow}}$ 不通过 VLM backprop
- VLM 只被 $\mathcal{L}_{\text{LM}}$ 更新

**Training setup**：
- Robot batch sequence length 2100（额外跑 action expert + 4 flow samples）
- Non-robot VLM batch 保持 4200
- 100K updates，global batch 128，64 H100，约 2300 GPU hours
- Expert LR: $5 \times 10^{-5}$（比 VLM 大 5x）

### 4.5 Embodiment-specific Fine-tuning

四个 embodiment 的 fine-tune 配置：

| Embodiment | Action chunk | Camera | Batch | GPUs | Steps |
|---|---|---|---|---|---|
| Bimanual YAM | 30 steps (30Hz) | top/left/right 固定 | 128 | 64 H100 | 100K |
| DROID | 15 steps (15Hz) | exterior + wrist | 64 | 32 H100 | 100K |
| SO-100/101 | 30 steps (30Hz) | 随机顺序 | 64 | 32 H100 | 100K |
| LIBERO | 10 steps (10Hz) | front + wrist 固定 | 64 | 32 H100 | 50K |

Fine-tuning 与 post-training 的**四个差异**：
1. Robot-only（无 multimodal VLM mixture）
2. Flow samples $K$ 从 4 增到 8
3. **取消 knowledge insulation**（gradient 允许更新 VLM）
4. 不 tune added-token input embeddings，tune output head + final norm

---

## 5. MolmoAct2-Think：Adaptive Depth Reasoning

### 5.1 核心思想

MolmoAct (Lee et al., 2025) 引入 depth-token prediction 作为中间 reasoning step。MolmoAct2-Think 的创新：**adaptive across time**——只对场景变化的区域重新预测 depth tokens。

**Intuition**：Robot trajectory 有大量时间冗余，很多 depth grid cell 在 control step 之间不变。重新预测静态区域浪费计算。

### 5.2 Depth Representation

- Depth map 量化为 **10×10 grid**（100 个空间位置）
- 每个 position 取 128 个 learned depth-code 之一
- 用 Depth Anything V2 (Yang et al., 2024) 估计 dense monocular depth
- VQ-VAE 操作在 320×320 depth image，downsampling factor 32，输出 10×10 grid
- Codebook indices $\in \{0, \ldots, 127\}$

### 5.3 Adaptive Update Mechanism（公式 10）

维护 buffer $b_t$ 和 update mask $m_t \in \{0,1\}^{100}$：

$$m_{t,i} = \mathbf{1}[\cos(x_{t,i}, x_{t-1,i}) < 0.996]$$

$$b_{t,i} = \begin{cases} d_{t,i}, & m_{t,i} = 1 \\ b_{t-1,i}, & m_{t,i} = 0 \end{cases} \tag{10}$$

- $x_{t,i}$：第 $t$ 帧第 $i$ 个 cell 的 RGB patch（resize 到 320×320 后切 10×10 个 32×32 patch）
- $\cos$：cosine similarity
- 阈值 0.996：低于这个值才标记为 updated
- $d_{t,i}$：当前帧的 VQ depth code
- $b_{t,i}$：buffer，updated 时用 $d_{t,i}$，否则沿用 $b_{t-1,i}$

**关键**：模型监督在 **buffer codes** $b_t$ 上（与 inference 时维持的表示一致）。

### 5.4 Training（公式 11-12）

Post-training 时三种 output style 等概率采样：
1. **Action**：`<action_output>`
2. **Depth**：`<depth_output>` + 100 depth-buffer tokens
3. **Depth-and-action**：`<depth_output><action_output>` 联合输出，expert 条件化 depth tokens 之后的 KV cache

Fine-tuning 时三个特化：
1. 只采样 action 和 depth-and-action（移除纯 depth 预测）
2. **Depth token noise injection**：10% input depth codes 替换为 uniform 随机 codes，target 不变（模拟 inference 时 imperfect depth）
3. **Learned per-layer depth gate**：

$$c_{\ell} = \frac{\sum_t A_t(1-M_t) V_{\ell,t}^{\text{vlm}}}{\sum_t A_t(1-M_t)}, \quad g_{\ell} = \sigma(w_{\ell}^{\top} c_{\ell} + b_{\ell}) \tag{11}$$

- $M_t$：depth-output trigger、delimiters、depth-code tokens 的位置 mask
- $A_t$：valid context positions
- $c_{\ell}$：non-depth context 的 mean pooling
- $g_{\ell}$：sigmoid gate，per-layer
- Bias 初始化为 $-4$（开始时 close to standard action-conditioning path）

应用到 depth tokens 的 keys/values：

$$\bar{K}_{\ell,t}^{\text{vlm}} = (1-M_t + M_t g_{\ell}) K_{\ell,t}^{\text{vlm}}, \quad \bar{V}_{\ell,t}^{\text{vlm}} = (1-M_t + M_t g_{\ell}) V_{\ell,t}^{\text{vlm}} \tag{12}$$

**Intuition**：gate 让模型学习每层用多强的 depth prefix，从 $-4$ bias 开始（几乎不用 depth）逐渐学习，避免训练初期 depth noise 污染 action 生成。

### 5.5 Adaptive Inference

1. Prefill over prompt + images
2. 无 cache 时：autoregressive 预测完整 `<depth_start>` + 100 codes + `<depth_end>`
3. 有 cache 时：
   - 比较当前 image 和 cached previous image 的 10×10 RGB patch cosine
   - Updated cells：argmax decoding from depth-token logits
   - Unchanged cells：replay from previous predicted depth buffer
   - 连续 unchanged spans 一起 replay，changed spans 逐 token decode
4. Action 生成：expert 接收 prompt + filled depth prefix 的 VLM KV cache，积分 flow matching

### 5.6 Inference 优化

- **Action expert**：固定 shape 的 flow-matching step，用 CUDA Graph capture（减少 kernel launch overhead）
- **Adaptive depth**：eager 执行（数据依赖），但 preallocated static KV cache 稳定 decode state
- Regenerated depth tokens：post-attention 到 next layer pre-attention 用 CUDA Graph，attention 保持 eager（KV length 变化）

---

## 6. 实验：VLA 领域最广泛的实证研究

### 6.1 Molmo2-ER vs. 其他 VLM（Table 3）

13 个 benchmark：
- **Molmo2-ER: 63.8%**（9/13 胜出）
- GR-ER 1.5 Thinking: 61.3%
- GPT-5: 57.9%
- Molmo2 base: 46.8%（提升 17 点）

### 6.2 Out-of-the-box Deployment

**Simulation - MolmoSpaces**（Table 4）：

| Model | Pick | Pick&Place | Open | Close | Avg |
|---|---|---|---|---|---|
| π0.5-DROID | 36.4 | 13.6 | 22.7 | 65.1 | 34.5 |
| **MOLMOACT2-DROID** | **43.7** | **26.7** | 9.5 | **70.8** | **37.7** |

**Simulation - MolmoBot**（Table 5，1000 episodes/task）：

| Model | Pick MSProc | Pick Classic | Pick | Pick Rand-Cam | Avg |
|---|---|---|---|---|---|
| π0.5-DROID | 18.1 | 6.4 | 7.0 | 8.0 | 10.0 |
| **MOLMOACT2-DROID** | **35.6** | **18.9** | **20.5** | **15.4** | **20.6** |

**Real-world DROID**（Table 6，15 trials/task）：

| Model | Apple | Pipette | Cube | Knife | Bowl | Avg |
|---|---|---|---|---|---|---|
| π0.5-DROID | 66.7 | 33.3 | 53.3 | 26.7 | 46.2 | 45.2 |
| MolmoBot | 86.7 | 53.3 | 33.3 | 40.0 | 28.6 | 48.4 |
| **MOLMOACT2-DROID** | **100** | **86.7** | **93.3** | **93.3** | 62.0 | **87.1** |

**Real-world SO-100/101**（Table 7）：

| Model | Fork | Stack | Tissues | Pen | Block | Avg |
|---|---|---|---|---|---|---|
| SmolVLA | 3.3 | 5.0 | 0.0 | 3.3 | 0.0 | 2.3 |
| π0-SO100/101 | 30.0 | 6.7 | 20.0 | 80.0 | 90.0 | 45.3 |
| **MOLMOACT2-SO100/101** | **70.0** | **20.0** | **73.3** | **86.7** | 33.3 | **56.7** |

### 6.3 Fine-tuning on LIBERO（Table 8）

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| MolmoAct-7B-D | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T N1.7 | 97.7 | 97.5 | 98.5 | 94.4 | 97.0 |
| **MOLMOACT2** | 97.8 | **100** | 97.8 | 93.2 | 97.2 |
| **MOLMOACT2-THINK** | **98.8** | 99.8 | **98.5** | **95.4** | **98.1** |

### 6.4 Real-world Bimanual YAM（Sec 6.3）

8 个 task，50 trials/task：
- **MolmoAct2: 50.1%** avg success
- 比 OpenVLA-OFT 高 15%
- 7/8 tasks 上超越所有 baselines

### 6.5 Trajectory Quality（RoboEval, Figure 6）

不仅看 success rate，还看：
- **Completion time**：MolmoAct2 在 Stack Two Blocks 上 4.70s（π0.5: 5.87s，Diffusion: 7.27s）
- **Joint path length**：1.04（π0.5: 2.16，约 2× 缩短）
- **Cartesian/joint jerk**：更稳定
- **Self-collisions**：更少

### 6.6 Systematic Ablations（Tables 9-13）

**Backbone ablation**（Table 9）：
- Molmo2 + discrete FAST: 77.6% (LIBERO Long)
- Molmo2-ER + discrete FAST: **83.6%**（+6 点）

证明 embodied reasoning 特化不仅帮助 VLM benchmark，直接 transfer 到 action prediction。

**VLM-to-expert connection**（Table 10）：
- Per-layer hidden states: 94.0%
- Per-layer KV (per-head): 94.8%
- **Per-layer KV (standard)**: **95.9%**

Per-layer KV connection > hidden state conditioning。Standard 比 per-head 略好，更简单。

**Multiple flow samples**（Table 11）：
- K=1: 94.15%
- K=2: 95.05%
- K=4: 95.15%
- **K=8: 95.90%**

更多 flow samples 给 expert 更密 supervision。

**Fine-tuning design**（Table 12）：
- Discrete co-training: 关键
- Knowledge insulation: 略差于无
- LoRA: Spatial 强但 Long 弱 2.8 点
- Action expert only: **失败**，93.05%
- **Full FT + discrete co-training**: **97.20%**

**Depth-aware fine-tuning**（Table 13）：
- 全 disable: 97.50%
- 只 mixed training: 97.65%
- **全 enable (noise + gate + mixed)**: **98.10%**

### 6.7 Inference Speed（Figure 8）

H100，horizon 10：
- **MolmoAct2 original**: 23.02 Hz
- **MolmoAct2 + caching**: 27.39 Hz
- **MolmoAct2 + CUDA Graph**: **55.79 Hz**（2.42× speedup）
- **MolmoAct2-Think original**: 8.04 Hz
- **MolmoAct2-Think + caching**: 9.72 Hz
- **MolmoAct2-Think + CUDA Graph**: 12.71 Hz（1.58× speedup）

MolmoAct2-Think 收益小因为 adaptive depth 的 autoregressive decode 难以 graph capture。

---

## 7. 关键 Intuition 总结

### 7.1 Per-layer KV Connection 的本质

传统做法：VLM final hidden state → action expert。这是信息瓶颈。

MolmoAct2 的做法：每层 VLM 的 KV cache 投影后给 expert 同层 cross-attend。这意味着：
- Expert 能看到 VLM 浅层的 visual features 和深层的 semantic features
- 信息流是 dense 的、hierarchical 的
- Detach 防止 flow loss 干扰 VLM 训练，但 LM loss 仍更新 VLM

### 7.2 Discrete + Continuous Co-training

Pre-training 用离散 FAST tokens（与 next-token prediction 统一，便于混合 multimodal data）。Post-training 加 flow matching expert，但**保留** discrete loss。

为什么保留？Table 12 显示：移除 discrete co-training 让 Spatial/Object 掉点，Long 涨点。整体平均相似但分布变。Co-training 提供 regularization。

### 7.3 Adaptive Depth 的经济性

100-token depth grid 每步全预测太慢。Adaptive 机制：
- 静态区域：replay cached tokens（几乎零计算）
- 动态区域：autoregressive decode

成本**与 scene change 成比例**，而不是固定 100 tokens。在 third-person view（大部分场景静态）下收益最大。

### 7.4 Knowledge Insulation 的时机

Post-training 用（detach VLM）：
- VLM 在 pre-training 已经学好 robot tokens
- Flow loss 主要训练 expert
- 避免 expert 的 noisy gradient 干扰 VLM

Fine-tuning 不用：
- 此时 expert 已成熟
- 允许 VLM 适配特定 embodiment 的视觉特征
- 实验显示 fine-tune 阶段 detach 没有收益

### 7.5 Depth Gate 的训练技巧

Bias 初始化 $-4$：sigmoid(-4) ≈ 0.018，意味着训练初期 depth prefix 几乎被屏蔽。这让模型从 standard action path 开始，逐渐学习怎么用 depth。如果初始 gate 大，depth noise 会立即污染 action 生成。

### 7.6 数据策略

- **Specialize-then-rehearse**：先特化再排练，防止遗忘
- **Community-sourced + centrally-collected**：SO-100/101 提供 diversity，YAM 提供质量
- **Language re-annotation**：VLM 重新标注，unique labels 翻倍

---

## 8. 与相关工作对比

| 维度 | π0/π0.5 | GR00T N1 | MolmoAct | **MolmoAct2** |
|---|---|---|---|---|
| Open data | ✗ | 部分 | ✓ | **✓ 全开** |
| VLM backbone | 通用 | 通用 | Molmo | **Molmo2-ER 特化** |
| Action expert | Flow matching | Diffusion | Discrete FAST | **Flow + discrete co-train** |
| VLM-expert 连接 | Final hidden | Final hidden | Discrete only | **Per-layer KV** |
| Reasoning | 无 | 无 | Depth tokens | **Adaptive depth** |
| Bimanual dataset | ✗ | ✗ | ✗ | **720h YAM** |
| Latency | 中 | 中 | 高（全 depth） | **低（adaptive）** |

---

## 9. 局限与未来方向

paper 没有显式讨论局限，但可推断：
- **Bimanual YAM 覆盖任务种类仍有限**：28 task 相比真实世界多样性还是少
- **Depth Anything V2 的误差**：monocular depth 在 occlusion 和 transparent 物体上不准
- **CUDA Graph 对 adaptive depth 收益小**：未来需要更动态的优化
- **Per-layer KV 的内存开销**：36 层全 cache 投影，memory footprint 不小
- **Flow samples K=8 在 fine-tuning 受 GPU memory 限制**

---

## 10. 参考 Link

- **Project page**: https://allenai.org/blog/molmoact2
- **Code**: https://github.com/allenai/molmoact2
- **FAST tokenizer**: Pertsch et al., 2025 - https://arxiv.org/abs/2501.09747
- **MolmoAct (前作)**: Lee et al., 2025 - https://arxiv.org/abs/2508.07917
- **Molmo2**: Clark et al., 2026 - https://arxiv.org/abs/2601.10611
- **DROID**: Khazatsky et al., 2024 - https://arxiv.org/abs/2403.12945
- **Depth Anything V2**: Yang et al., 2024 - https://arxiv.org/abs/2406.09414
- **SigLIP2**: Tschannen et al., 2025 - https://arxiv.org/abs/2502.14786
- **LIBERO**: Liu et al., 2023
- **RoboEval**: Wang et al., 2025b - https://arxiv.org/abs/2507.00435
- **RoboPoint**: Yuan et al., 2024 - https://arxiv.org/abs/2406.10721
- **RefSpatial**: Zhou et al., 2025 - https://arxiv.org/abs/2506.04308
- **Flow Matching**: Lipman et al., 2022 - https://arxiv.org/abs/2210.02747

---

这篇 paper 的核心贡献是把 VLA 的几个痛点系统性解决：**open everything + 专门 backbone + per-layer KV 连接 + adaptive reasoning**。Per-layer KV connection 和 adaptive depth 是两个最有想象力的架构创新，前者解决了 VLM-to-expert 的信息瓶颈，后者解决了 reasoning-augmented policy 的延迟问题。实验覆盖范围（7 benchmark + 3 embodiment）在 open VLA 里史无前例。
