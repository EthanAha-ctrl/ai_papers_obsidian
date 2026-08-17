---
source_pdf: Thinking in Space.pdf
paper_sha256: 41e79948c538ad464e82a6bd4f47034d6301944befe6e21c5aeaf31ced4ae826
processed_at: '2026-08-12T15:33:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心故事其实非常直观。想象你拿着手机录了一段家里的视频，然后发给 AI，问它："帮我想想，从冰箱走到沙发，中间需要左转还是右转？" 或者 "我家那个柜子有多高？" 

这篇 paper 就是把这种"空间认知能力"做成了一个严谨的 benchmark 叫 VSI-Bench，然后测了目前市面上最强的 15 个 video MLLMs。结论非常戏剧性：**AI 能看懂视频里有什么，但在脑子里构建 3D 空间地图的能力极差，且用传统的语言推理方法不仅救不了它，反而会让它变更笨。**

下面我拆解里面的技术细节，帮你 build intuition。

## 1. 为什么 MLLM 缺乏 Spatial Intelligence？从 Token 压缩说起

现在的 video MLLM 架构，本质上是一个 Vision Encoder 加一个 LLM。视频被抽成几十帧 image，每帧打成 patch，经过 Vision Transformer 压缩成 visual tokens，然后跟 text tokens 拼在一起喂给 LLM 做 next-token prediction。

这里面的根本矛盾在于：**3D 空间信息被有损压缩成了 2D patch 的 semantic features**。
假设视频里有张桌子，桌子在画面左边，下一帧相机右 pan，桌子到了画面右边。在 LLM 眼里，这只是 sequence 里两个不同位置的 token 激活，它没有内在的机制把这些 2D 观测 stitch 成一个 allocentric（上帝视角）的绝对坐标。

这篇 paper 在 Appendix D 里发现了一个极度违背直觉的现象：
| Input Sequence | Avg. Performance |
|---|---|
| Video first | 48.8% |
| Question first | 46.3% |

把 question 放在 video 前面（Question first），模型反而表现更好。并且重复输入两次视频，模型表现会提升 2.1%。
**Intuition**: 在标准 transformer attention 机制 $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ 中，$Q, K, V$ 分别是 query, key, value 矩阵，$d_k$ 是 key 的 dimension。如果先输入 video，随着 autoregressive generation 往后走，text token 越来越多，video tokens 的 attention 权重会被严重稀释。重复输入视频，相当于强行拉高了 video features 在 KV cache 里的 L2 norm，让模型在 decoding 时能重新 attend 到关键的空间画面。这证明了当前 MLLM 的 spatial working memory 极其脆弱。

## 2. MRA Metric 的数学直觉

评测空间感知经常要回答数值题（如柜子多高、房间多大）。Exact match 太苛刻。作者搞了个 Mean Relative Accuracy (MRA)。

公式是：
$$ \mathcal{MRA} = \frac{1}{10} \sum_{\theta \in \mathcal{C}} \mathbb{1} \left( \frac{|\hat{y} - y|}{y} < 1 - \theta \right) $$

变量解释：
- $\hat{y}$ (y-hat): MLLM 预测的数值，比如估计房间面积 15 sqm。
- $y$: Ground truth 真实数值，比如 20 sqm。
- $\theta$: Confidence threshold 阈值，从集合 $\mathcal{C} = \{0.5, 0.55, \hdots, 0.95\}$ 中取值，一共 10 个点。
- $\mathbb{1}(\cdot)$: Indicator function，条件成立返回 1，否则返回 0。

相对误差率定义为 $\frac{|\hat{y} - y|}{y}$。如果 $\theta = 0.95$，意味着模型预测值与真实值的相对误差必须小于 $1 - 0.95 = 0.05$（即 5%）才算对。把 10 个阈值下的结果取平均，就得到了 MRA。这个 metric 的好处是对微小的偏差不惩罚过度，但保留了连续评估的 discriminative power。

## 3. CoT 为什么在 Spatial Task 上有毒？

这是这篇 paper 最深刻的发现。他们试了 Chain-of-Thought, Self-Consistency, Tree-of-Thoughts，结果在 Object Size 任务上，CoT 让性能暴跌了 21%！

**Intuition**: 空间感知原本是 System 1（直觉式）。视频里出现一个 fridge，模型根据 pretrain 时见过的 fridge 图像，直接 output 一个大概的 cm 数值。这是一种 pattern matching。
引入 CoT 后，你强迫模型用 System 2（语言逻辑链）去推导。模型开始说："我看到 fridge 旁边有 3 块 floor tiles，通常一块 tile 是 30 cm，所以 fridge 大概 90 cm..."
这会导致两个灾难：
1. **Error Amplification**: 语言链条 $T_1, T_2, ... T_n$ 中，如果某一步 $T_k$ 基于幻觉（比如把地板花纹数错了），后续的 $T_{k+1}$ 全部建立在错误前提上。
2. **Dual Coding Theory Conflict**: 认知心理学认为，verbal processing 和 visual processing 是大脑里两个独立的 channel（参考 Paivio 1991, https://link.springer.com/article/10.1007/BF01326232）。强制把空间问题转译成自然语言 tokens，会破坏原本隐含在 network weights 里的 visual-spatial 局部流形，产生严重的 modality mismatch。

## 4. Cognitive Map：从 Local World Models 到 Global Stiching

为了探究 MLLM 内部到底记住了什么，作者让 Gemini-1.5 Pro 生成一个 10x10 grid 的 cognitive map，输出 JSON 格式的坐标。

评估时，计算任意两个 object 之间的 Euclidean distance。如果 MLLM 预测的 object A 和 object B 的距离 $d_{pred}$ 与 Ground truth 距离 $d_{gt}$ 误差在 1 grid unit 以内，就算对。

数据结果极度诡异：
- 0-1 grid distance: 64% accuracy
- 7-8 grid distance: ~20% accuracy

**Intuition**: 模型内部形成了一堆 "Local World Models"。就像 SLAM 里的 Visual Odometry，每一帧附近的物体相对位置它算得很清楚，但完全没有 Loop Closure 机制。一旦距离拉远，全局坐标系就 drift 了。这跟神经科学里的发现高度吻合：Hippocampus 里的 Place Cells 负责局部位置编码，但全局地图需要 Entorhinal Cortex 里的 Grid Cells 提供周期性的六边形坐标系。MLLMs 现在只 emerge 了 Place Cells，没有 Grid Cells。

接着作者做了一个神级 intervention：
| Case | Rel. Dist. Acc. |
|---|---|
| w/o Cog. map | 46.0% |
| w/ Cog. map (MLLM) | 56.0% |
| w/ Cog. map (GT) | 66.0% |
| w/ Cog. map (GT 20x20) | 78.0% |

逼模型先在草稿纸上画个 map，再看着自己的 map 答题，准确率直接涨 10%。给模型喂 Ground Truth 的 map，涨 20%。给更高分辨率的 GT map (20x20)，涨 32%。
这说明：**Spatial Reasoning 的瓶颈根本不在于 LLM 的逻辑推理能力，而在于缺乏一个显式的、可持久化的 Spatial State Memory。**

## 5. 架构层面的联想与未来路线

这篇 paper 给我最大的震动是，它揭示了当前 LLM 架构在处理 $n$-dimensional 空间时的根本缺陷。

1. **AlphaFold Paradigm**: AlphaFold 之所以能解决 3D 蛋白质折叠，是因为它没有纯靠 1D 的 amino acid token 硬刚。它引入了 Structure Module，用 Invariant Point Attention (IPA) 直接在 3D space 里做几何变换，维护一个显式的 3D 坐标 state。
未来的 Video MLLM 如果要真正 "Think in Space"，光靠 stack vision tokens 是没用的，必须在 Transformer 内部或者外挂一个 module，专门维护 latent 的 3D allocentric coordinate frame。比如参考最近的 VGGT (https://vggt.github.io/)，这种 feed-forward 3D reconstruction model 已经能从 2D frames 直接 predict camera pose 和 point map。

2. **Spatial KV Cache**: 当前的 KV cache 是为 text token 设计的。如果我们能在 attention 里区分 "Spatial Tokens"，让它们参与计算时使用一套独立的位置 encoding（比如 3D RoPE 替代 1D RoPE），或许能让模型在 autoregressive decoding 时动态 update 内部的 spatial memory，而不是每次问空间问题都要从头 "回想" 视频。

3. **Embodied Pretraining Loss**: 现在的 video-text contrastive loss 根本不惩罚 spatial inconsistency。未来可能需要在 pretrain 阶段加入 metric depth prediction loss (像 Depth Anything, https://depth-anything.com/)，或者 multi-view consistency loss，强迫 latent representation 里保留 metric geometry。

4. **Object-Centric State Spaces**: 模型回答 "哪个离冰箱近" 这种问题失败，说明它没有 tracking discrete objects 的 state。参考 Slot Attention 或者 Sora 尝试的 object-centric factorization，把视觉输入 decompose 成 $N$ 个 object query $\{o_1, o_2, ... o_N\}$，每个 query 携带 $(x, y, z, \text{semantic})$ 的 latent vector，再输入给 LLM 做推理，这可能才是 spatial AGI 的正确路径。

总而言之，这篇 paper 是一个响亮的耳光，打在那些认为 "只要 scale up video tokens，AI 就能理解物理世界" 的乐观主义脸上。它用严密的 error analysis 和 cognitive map probing 证明：**Spatial reasoning is an emergent property that current 1D autoregressive LLM architectures fundamentally struggle to capture, and visual prosthesis like explicit map generation might be the necessary bridge.**

Reference Links:
- Paper: https://arxiv.org/abs/2412.14171
- AlphaFold & IPA: https://www.nature.com/articles/s41586-021-03819-2
- VGGT (Feed-forward 3D reconstruction): https://vggt.github.io/
- Depth Anything: https://depth-anything.com/
- Sora technical report (Object-centric discussion): https://openai.com/research/video-generation-models-as-world-simulators
- Invariant Point Attention explanation: https://www.biorxiv.org/content/10.1101/2021.08.30.458961v1
- Slot Attention: https://slot-attention.video/
- SpatialVLM (Related work): https://spatialvlm.github.io/

---

# Thinking in Space: 深度解析

## 一、核心问题与动机

人类在逛宜家买柜子时, 可以在脑中调出客厅的mental image, 估算新家具能否塞下。这种visual-spatial intelligence涉及一个连续的过程: perceive → memorize → recall → reason。这篇paper问的核心问题是: **MLLMs trained on million-scale video能否从egocentric video中"think in space"?**

这问题的根本性在于: 现有video MLLM benchmarks (VideoMME, EgoSchema, MVBench) 评测的多是**content-level understanding**——"视频里出现了什么"、"发生了什么事件", 这只是2D image understanding的temporal extension。而VSI-Bench评测的是**3D spatial understanding from 2D video**——一个根本 harder 的inverse problem, 因为模型需要从2D像素流inference出3D scene layout。

这与经典SLAM问题、NeRF、3D reconstruction的motivation相通, 但区别在于: SLAM通过explicit几何优化求解, MLLM通过implicit neural representation "记住"空间。作者关心的不是geometric accuracy per se, 而是**spatial intelligence是否emerge in frontier models**——一个capability-oriented而非task-oriented的问题。

Project page: https://thinking-in-space.github.io/
Code: https://github.com/MLLM-Space/Thinking-in-Space

---

## 二、Visual-Spatial Intelligence Taxonomy

作者基于认知心理学(Gardner multiple intelligences, Tolman cognitive maps, Baddeley working memory)提出一个4-component taxonomy (Fig. 2):

1. **Visual perception**: 识别物体、分类
2. **Linguistic intelligence**: 逻辑/数学/语言推理
3. **Temporal processing**: 时序记忆
4. **Spatial reasoning**:
   - **Relational reasoning**: 通过distance和direction判断物体间关系, 也包括借助visuospatial common sense (如可乐罐~12cm)做相对大小推理
   - **Egocentric-allocentric transformation**: 在self-centered view和environment-centered view间切换

这个taxonomy的insight在于: **spatial reasoning与visual perception在神经层面是distinct的**(参考Chabris et al. 2006, [11])。一个模型可以认出sofa和TV, 但仍然不知道sofa在TV的哪一侧。这解释了为什么模型可以正确描述视频内容但答错spatial question。

**与认知科学的联系**:
- Tolman 1948 "Cognitive maps in rats and men" (https://psycnet.apa.org/record/1949-03405-001): 老鼠在迷宫中形成mental map
- O'Keefe & Nadel 1978: hippocampus as cognitive map, place cells
- Dual coding theory (Paivio 1991, [18]): linguistic和visual processing是distinct yet complementary的channels

---

## 三、VSI-Bench设计与构建

### 3.1 Data sources

| Dataset | Samples | 类型 | FPS |
|---|---|---|---|
| ScanNet [19] | 88 | RGB-D indoor, 1513 scenes | 24 (从frames合成) |
| ScanNet++ [94] | 50 | 高保真indoor, 280 scenes | 30 |
| ARKitScenes [5] | 150 | iPhone LiDAR, 4.5K scenes | 30 |

总计288个视频, 5000+ QA pairs。选择这3个dataset的智慧在于: 它们都是3D reconstruction datasets, 有accurate object-level 3D annotations, 可以自动生成ground truth answers。视频统一为640×480, 旋转归一化。

Meta-info unified format: `{dataset, video_path, room_size, room_center, object_counts, object_bboxes (Open3D OrientedBoundingBox)}`。

Room size用Alpha shape算法从point cloud计算。这本身就是一个approximation——alpha shape是concave hull, 比convex hull紧但比true floor plan松。

### 3.2 8个Tasks

按category分3类:

**Configurational (4)**:
- Object Count: "How many {category}(s) are in this room?"
- Relative Distance: "which of these objects ({a},{b},{c},{d}) is closest to the {category}?"
- Relative Direction: 3个difficulty levels
  - Easy: left/right
  - Medium: left/right/back (back = 至少135°)
  - Hard: front-left/front-right/back-left/back-right (Cartesian quadrants)
- Route Plan: 人类标注, "Go forward until X, [turn left/right/back], Go forward until Y..."

**Measurement Estimation (3, 数值题)**:
- Object Size: 物体最长dimension (cm)
- Room Size: 平方米
- Absolute Distance: 两object closest points的Euclidean距离 (m)

**Spatiotemporal (1)**:
- Appearance Order: 4个category首次出现顺序

### 3.3 MRA Metric (关键创新)

对于numerical answer tasks, exact match太苛刻。作者定义:

$$\text{Relative Accuracy at threshold } \theta = \mathbb{1}\left(\frac{|\hat{y} - y|}{y} < 1 - \theta\right)$$

其中:
- $\hat{y}$: 模型预测值
- $y$: ground truth数值
- $\theta \in \mathcal{C}$: confidence threshold
- $\mathbb{1}(\cdot)$: indicator function

注意阈值条件是 $|\hat{y}-y|/y < 1-\theta$, 即相对误差率 < $1-\theta$。当θ=0.95, 容忍5%误差; θ=0.5, 容忍50%误差。

**Mean Relative Accuracy** across 10 thresholds:

$$\mathcal{MRA} = \frac{1}{10}\sum_{\theta \in \{0.5, 0.55, ..., 0.95\}} \mathbb{1}\left(\frac{|\hat{y} - y|}{y} < 1 - \theta\right)$$

这个metric的好处: 平滑、可微的思想——越接近GT得分越高, 而非0/1 hard match。类似PASCAL VOC的IoU threshold averaging [22]。

### 3.4 Baselines

- **Chance Level (Random)**: MCA任务随机选
- **Chance Level (Frequency)**: 总是选最频繁答案
- **Human Level**: 400题子集 (VSI-Bench tiny), 50题/task, 人类可无限次重看视频

---

## 四、主实验结果

### 4.1 Top numbers (Table 1, 8)

| 模型 | Avg | Config | Meas. | Spatiotemp. |
|---|---|---|---|---|
| Human | **79%** | 94-100% | 较弱 | 100% |
| Gemini-1.5 Pro | 48.8% | - | 接近human | - |
| GPT-4o | 35.6% | - | - | - |
| LLaVA-NeXT-Video-72B | 39.3% | - | - | - |
| LLaVA-OneVision-72B | 41.6% | - | - | - |
| InternVL2-40B | 37.0% | - | - | - |
| LongVILA-8B (32 frames) | 19.1% | - | - | - |

7/12 open-source models 低于chance level——这是shocking的。

### 4.2 Blind vs. Vision-enabled (Fig. 6)

定义:
- **Enabled − Disabled**: 视频带来的提升
- **Disabled − Chance**: 纯language prior超过频率baseline的能力

Key findings:
- 视频普遍有帮助, 但在absolute distance、route plan、relative direction上几乎无提升——这些tasks连视觉都没救
- **Object size上blind模型已显著超越chance**, 说明LLM pretraining中学到了size common sense (e.g. "kitchen cabinet ~80cm")

这给出一个重要insight: **看起来像spatial intelligence的能力, 可能只是language prior**。这是经典的"AI做对的事但理由错"的问题。

---

## 五、Error Analysis: Spatial Reasoning是主bottleneck

作者人工review了VSI-Bench(tiny)的163个error cases, 分4类:

1. **Visual perception error**: 物体识别错误或类别混淆
2. **Linguistic intelligence error**: 逻辑/数学/语言理解错误
3. **Relational reasoning error**: distance/direction/size关系推理错
4. **Egocentric-allocentric transformation error**: allocentric布局错或perspective-taking错

结果(Fig. 8): **~71% errors来自spatial reasoning** (relational + egocentric-allocentric)。

### 关键案例 (Fig. 7)

**成功案例**: Relative Direction任务
- 模型说"orient yourself, locate dishwasher, visualize quadrants"
- 表明模型可能build了implicit global coordinate system

**失败案例**: Relative Direction
- 视频相机向右pan从bed转向wall/window
- 模型顺着egocentric视角说"to face wall you must turn right"
- 真实allocentric布局: door → bed需要turn left
- 模型被**视频的视角"绑架"了**, 无法做egocentric→allocentric转换

这个failure mode极具启发性: **MLLMs的"thinking"是跟着观测stream走的, 而不是构建独立的bird's-eye view**。这类似于SLAM中只有visual odometry没有loop closure——local consistent但global drift。

---

## 六、CoT方法在Spatial任务上失效(关键insight)

测试3种linguistic prompting (Fig. 9):

1. **Zero-Shot CoT**: 加 "Let's think step by step"
2. **Self-Consistency w/ CoT**: temperature=1.0, 5次majority vote
3. **Tree-of-Thoughts (ToT)**: plan generation + answer prediction, 多轮投票

结果:
- Zero-Shot CoT: **-4%** (avg)
- Self-Consistency: -1.1%
- ToT: -4% (avg)
- Object size task: -21% (catastrophic!)
- Room size task: -8%
- Absolute distance & appearance order: 略有提升

对比: 在VideoMME上CoT给Gemini-1.5 Pro带来 +1.6% (Table 2)。**所以CoT在general video understanding上有效, 但在spatial reasoning上有害**。

### 为什么CoT有害? 我的interpretation

这是这篇paper最反直觉但最深刻的发现。几个可能原因:

1. **Modality mismatch**: CoT forcing verbal chains on inherently spatial problems. Dual coding theory (Paivio)说verbal和visual是两个独立channel。强制verbal chain可能**干扰而非辅助visual imagery**。

2. **Hallucination amplification**: 在size estimation上, "think more"让模型chain更多中间假设(如地板plank宽度), 每个假设都有误差, 误差累积放大。一步直觉猜反而可能更准。

3. **Attention dilution**: 在spatial reasoning中, 关键cues在视频某几帧。CoT分散attention到narrative reasoning, 反而miss关键frames。

4. **Anchoring bias**: CoT生成的early steps成为后续reasoning的anchor, 即使early steps基于错误spatial inference, 后续无法纠正。

这让我想起Kahneman的System 1 vs System 2: **spatial estimation对人类是System 1 (直觉), 强行用System 2 (verbal)反而变差**。

类似发现: Visualization-of-Thought (VoT) [87] (https://arxiv.org/abs/2405.08219) — NeurIPS 2024, 用visual reasoning代替verbal reasoning帮助LLM做spatial任务。

---

## 七、Cognitive Maps: Probing MLLMs的内部空间表示

### 7.1 实验设计

受dual coding theory启发, 作者直接prompt Gemini-1.5 Pro生成cognitive map:

```
This video captures an indoor scene. Your objective is to identify specific 
objects within the video, understand the spatial arrangement of the scene, 
and estimate the center point of each object, assuming the entire scene is 
represented by a 10x10 grid.
...
Present the estimated center locations for each object as a list within a 
dictionary. STRICTLY follow this JSON format: {"category name": [(x_1, y_1), ...], ...}
```

MLLM输出JSON格式的object center positions on 10×10 grid (Fig. 10可视化)。

### 7.2 Locality metric

对每对category计算Euclidean distance:
- 多instance category取**最短pairwise距离**
- MLLM预测距离与GT距离相差 ≤ 1 grid unit则算correct
- 距离分8个bins (0-1, 1-2, ..., 7-8 grid units)

结果 (Fig. 11):
- 相邻object (0-1 grid): **64% accuracy**
- 1-2 grid: ~50%
- 距离增大时accuracy急剧下降
- 远距离: ~20-30%

**结论: MLLMs形成series of local world models而非unified global model**。

这个发现的深度: 模型在每一帧附近都build一个coherent local map (类似sub-map in SLAM), 但无法将这些sub-maps stitch成global map。**没有loop closure mechanism in MLLM spatial memory**。

这与神经科学发现一致: hippocampus的place cells形成local maps, 但global consistency需要entorhinal cortex的grid cells提供metric。MLLMs可能emerge了"place cell-like"局部表示但缺"grid cell-like"全局坐标。

参考:
- Tolman 1948 cognitive maps
- Behrens et al. 2018 "What is the cognitive map?" Nature Neuroscience
- Ladyman et al. on mental representation

### 7.3 Cognitive map作为intervention (Table 3)

让模型先输出cognitive map, 再基于map回答relative distance问题:

| Setting | Rel. Dist. Acc. |
|---|---|
| Baseline (no map) | 46.0% |
| w/ MLLM-generated map | 56.0% (+10%) |
| w/ GT map (10×10) | 66.0% (+20%) |
| w/ GT map (20×20) | 78.0% (+32%) |

观察:
- MLLM自生成map就有 +10% 提升
- 用GT map提升 +20% — 说明map作为external memory确实有用
- 20×20 GT (78%) vs 10×10 GT (66%) — 分辨率有用, 表示表示能力是bottleneck
- MLLM生成的map与GT map的差距 (~10%) 来自locality问题

### 7.4 这个发现的implications

这给出一个非常具体的actionable insight: **让MLLM在spatial reasoning前先输出cognitive map作为scratchpad**。这相当于"thinking with images"而非"thinking with words"。

类似思想:
- Scratchpad in GPT-3 (https://arxiv.org/abs/2112.00114)
- Tool use让模型输出intermediate representation
- Program-aided language models (PAL)
- Voyager的skill library

但cognitive map的特殊之处在于: 它是**visual/structured representation而非verbal chain**。

未来方向猜测:
1. **Joint training**: 让MLLM在pretraining时学习从video预测cognitive map作为auxiliary loss (类似depth/motion prediction in self-sup learning)
2. **Multi-step map refinement**: 让模型iteratively refine map (类似SLAM的pose graph optimization)
3. **3D map**: 用voxel grid或NeRF-like latent代替2D grid
4. **Differentiable rendering**: 让模型render from map验证一致性

---

## 八、Ablations: Input Sequencing和Repetition (Table 5, Appendix D)

### 8.1 Video-first vs Question-first

| Order | Avg |
|---|---|
| Video first | 48.8% |
| Question first | 46.3% |

Question先于video展示给模型更好——这跟人类相反! 人类看视频前知道问题更有助于focus attention。但MLLM在question-first模式下反而更好, 可能因为attention机制被question激活, 然后处理video时更有的放矢。

### 8.2 Video Repetition

| Times | Avg |
|---|---|
| 1 | 48.8% |
| 2 (重复video输入) | 50.9% (+2.1%) |

**惊喜发现**: 即使autoregressive MLLM理论上能在answer generation时多次"reference" video, 实际上explicit repeat video输入2次带来 +2.1%。

这说明: **当前MLLM的video token attention机制是suboptimal的**。Autoregressive generation时, 模型 attend to video tokens的能力可能被text context length稀释。重复输入video tokens相当于增加video signal的信噪比。

这跟"re-reading"在人类学习中的作用类似, 但机制可能不同。

---

## 九、与相关工作的联系

### 9.1 MLLM Spatial Awareness

- **SpatialVLM** (Chen et al. CVPR 2024, https://spatialvlm.github.io/): 用internet 3D数据endow VLM with metric spatial reasoning
- **SpatialRGPT** (Cheng et al. NeurIPS 2024): depth-aware VLM
- **SpatialBot** (Cai et al. 2024): precise spatial understanding
- **V-IRL** (Yang et al. ECCV 2024): grounding virtual intelligence in real life

这些工作多基于2D images, VSI-Bench基于video——更接近真实embodied场景。

### 9.2 Video MLLM Benchmarks

- **VideoMME** (https://github.com/BradyFU/VideoMME): comprehensive但content-level
- **EgoSchema** (Mangalam et al. NeurIPS 2023): egocentric长视频理解
- **OpenEQA** (Majumdar et al. CVPR 2024): embodied QA
- **MMBench-Video** (Fang et al. NeurIPS 2024): long-form multi-shot

VSI-Bench区别: explicit 3D spatial focus。

### 9.3 Embodied Agents

- **RT-1/RT-2** (Brohan et al.): vision-language-action models
- **OpenVLA** (Kim et al. CoRL 2024, https://openvla.github.io/)
- **PaLM-E** (Driess et al. ICML 2023)

这些都需要spatial intelligence才能work in real world。VSI-Bench可作为这些agent的capability probe。

### 9.4 Cognitive Maps in AI

- **DayDreamer** (Hafner et al.): world models for embodied control
- **DreamerV3**: mastering diverse domains through world models
- **Neural SLAM** (Gupta et al.): cognitive map in navigation agents
- **TAMER / Mapping networks**: emerging map-like representations

### 9.5 Mental Imagery debates

- Kosslyn (imagery debate): mental images are depictive
- Pylyshyn: mental images are tacit knowledge
- VoT (https://arxiv.org/abs/2405.08219): visualization-of-thought给LLM

MLLM的cognitive map output是第一个quantitative evidence that frontier models有implicit depictive representations。

---

## 十、Intuition Building: 我的几个观察

### 10.1 Spatial Intelligence作为 emergent property的边界

VSI-Bench像一个"stress test"暴露emergent capability的边界:
- Visual perception: strong emergence (CLIP-scale pretraining)
- Linguistic: very strong (LLM base)
- Temporal: medium (video pretraining)
- Spatial reasoning: weak/fragmentary emergence

这暗示current video pretraining objectives (next-token prediction on video-text pairs) **不直接训练spatial reasoning**。Spatial reasoning可能需要**explicit 3D supervision or embodied interaction**。

### 10.2 为什么Video MLLMs形成local maps而非global maps?

可能原因:
1. **Attention span**: video token sequence long, local attention window → local map
2. **No geometric consistency loss**: pretraining不penalize globally inconsistent maps
3. **Tokens ≠ coordinates**: video tokens是semantic features, 2D position信息lossy

未来工作: 引入geometric consistency loss或在pretraining中加入cognitive map prediction auxiliary task。

### 10.3 Embodied Pretraining Hypothesis

如果MLLM在pretraining时同时学:
- Predict next frame
- Predict camera pose
- Predict depth
- Predict object positions in allocentric frame

可能emerge真正的global spatial reasoning。参考:
- **V-JEPA** (https://ai.meta.com/research/publications/v-jepa/)
- **Depth Anything** (https://depth-anything.com/)
- **VGGT** (https://vggt.github.io/): feed-forward 3D reconstruction

### 10.4 Cognitive Map as Universal Interface

我推测未来MLLM architecture会有一个**显式spatial state module**:
- 类似context cache但专门存spatial representation
- 接受video frames, 更新map
- 接收query, render from map
- 不同modality queries用同一map

这跟robotics中的neural SLAM、video model中的latent state、LLM中的KV cache都有相似处。

### 10.5 与AlphaFold-like Leap

AlphaFold2的突破在于从sequence直接预测3D structure, 用structure module显式represent 3D。MLLMs目前缺这个"structure module"——所有spatial info压缩在text tokens里。一个cognitive map module可能是MLLM的"structure module"。

### 10.6 Failure mode的哲学意味

Egocentric-allocentric transformation失败案例很有哲学意味: **模型的语言输出欺骗了我们**。它说"orient yourself, locate X, visualize quadrants"——看起来在thinking spatially, 实际上只是verbal theater。这是"stochastic parrot"在spatial domain的体现。

参考Bender & Koltun 2021, "On the Dangers of Stochastic Parrots" (https://dl.acm.org/doi/10.1145/3442188.3445922)

---

## 十一、Limitations (paper未明说但可推断)

1. **Indoor focus**: 只测室内, outdoor navigation、urban scene、自然场景未覆盖
2. **Static environment**: 没有dynamic objects
3. **No embodied interaction**: 模型只"看"不能"动", 无法通过locomotion update map
4. **Single traversal**: 没有loop closure机会
5. **2D grid representation**: cognitive map是2D, 真实空间是3D
6. **GT map quality**: alpha shape的room size本身是approximation
7. **Gemini-1.5 Pro only for probing**: 其他MLLMs是否也form cognitive maps? 不同architecture可能不同
8. **English questions only**: 跨语言spatial reasoning未测

---

## 十二、个人猜想: 未来6-12个月可能的工作

1. **VSI-Bench v2**: 加入navigation任务, 模型可主动选择 viewpoints (active perception)
2. **3D cognitive maps**: 用voxel或Gaussian splat作为map representation
3. **Pretraining objectives**: "cognitive map prediction"作为auxiliary loss
4. **Embodied VLM**: VLA models用VSI-Bench作capability评估, fine-tune提升
5. **Cross-modal spatial transfer**: 从synthetic 3D scene (e.g. Habitat, http://habitatai.org/) 学到的spatial能力能否transfer到real video?
6. **Spatial CoT**: 设计spatial-specific reasoning prompt, 在verbal chain中插入visual tokens或coordinate references

---

## 十三、最深刻的takeaway

这篇paper给我三个deep insights:

1. **Emergence有清晰的boundary**: visual perception和linguistic能力emerge了, 但spatial reasoning还在threshold。Pretraining data和objective决定了什么emerge——video-text pairs训练semantic content understanding, 不训练spatial reasoning。

2. **Verbal ≠ Spatial**: CoT对language reasoning有效但对spatial有害, 是modality mismatch的实证。未来reasoning research应该考虑**representation-appropriate reasoning**, 而不是把verbal CoT当作universal tool。

3. **External representation作为cognitive prosthesis**: Cognitive map作为intermediate output提供 +10% gain, 说明外部化spatial representation能补足internal representation的不足。这与"tools for thought"、Sketchpad、Extended Mind thesis (Clark & Chalmers)相通。

这个工作虽然看似是benchmark paper, 但其核心贡献是**diagnostic**——精准定位了MLLMs在spatial intelligence上的failure mode和潜在solution。它为下一代embodied MLLM指明了方向: **explicit spatial state representation是必要的architectural inductive bias**。

---

## 参考链接

- Paper (arXiv): https://arxiv.org/abs/2412.14171
- Project page: https://thinking-in-space.github.io/  
- Code: https://github.com/MLLM-Space/Thinking-in-Space
- VSI-Bench: 见project page
- VideoMME: https://video-mme.github.io/
- SpatialVLM: https://spatialvlm.github.io/
- OpenVLA: https://openvla.github.io/
- VoT: https://arxiv.org/abs/2405.08219
- EgoSchema: https://github.com/karthiksharma97/EgoSchema-Benchmark-Code
- ScanNet: http://www.scan-net.org/
- ScanNet++: https://kaldir.vc.in.tum.de/scannet++/
- ARKitScenes: https://github.com/apple/ARKitScenes
- Open3D: http://www.open3d.org/
- LMMs-Eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
- Tolman 1948 cognitive maps: https://psycnet.apa.org/record/1949-03405-001
- Dual coding theory (Paivio): https://link.springer.com/article/10.1007/BF01326232
- Bender & Koltun "Stochastic Parrots": https://dl.acm.org/doi/10.1145/3442188.3445922
- VGGT (3D foundation model): https://vggt.github.io/
- Habitat: http://habitatai.org/
- DayDreamer: https://danijar.com/daydreamer/
- Voyager: https://voyager-minecraft.github.io/
