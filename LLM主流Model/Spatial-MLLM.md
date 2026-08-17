---
source_pdf: Spatial-MLLM.pdf
paper_sha256: a4b2d301d4fd8fef5f16f8ea011be6044f8138c666d4a50f7aefd91b4d32008a
processed_at: '2026-08-12T09:20:44-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Spatial-MLLM: 人话版

---

## 这篇paper想干啥?

简单说: 让AI看一段房间里拍的视频, 然后能回答"这房间多大?"、"桌子离床多远?"、"沙发左边是什么?"这类空间问题。

人类看个视频就能脑补出3D房间布局, 但现在的AI死活做不到 — 它能告诉你视频里有个沙发、是皮的、米色的, 但你问它沙发离墙多远, 它就开始瞎猜。

---

## 问题出在哪?

作者发现一个非常朴素的道理:

现在所有video AI的"眼睛"(visual encoder)都是用**图片配文字描述**训练出来的。这种训练方式让AI学会了"这是沙发、那是台灯", 但**完全没学会空间感**。

就好比一个人从小只看带标签的2D照片长大, 从没摸过实物、没感受过远近, 你突然让他估一张照片里桌子的实际尺寸, 他肯定估不准 — 他脑子里根本没有"3D世界"这个概念。

而传统做3D理解的AI, 都是直接喂它point cloud(激光扫出来的3D点云数据)或depth map(深度图), 相当于直接给它3D信息。但现实中很多场景只有普通2D视频, 没有这些额外数据。

**核心矛盾**: 2D视频里其实藏着3D信息(多视角、运动视差等), 但现有AI的"眼睛"提取不出来。

---

## 他们怎么解决的?

非常直接的思路: **给AI换一双能看3D的眼睛**。

具体来说, 给AI装两只眼:

- **左眼**(Qwen2.5-VL的ViT): 原来的眼睛, 擅长认东西, 知道"这是个沙发"
- **右眼**(VGGT): 新加的眼睛, 专门从2D图片里脑补3D结构, 知道"沙发在3米外、靠左墙"

这只新"右眼"来自一个叫VGGT的模型 — 这玩意儿是专门用"像素↔3D点"配对数据训练的, 你给它几张2D照片, 它能forward pass一次就吐出相机参数和深度图, 相当于一个"一秒建图"的几何引擎。

然后把两只眼看到的信息简单加起来, 喂给LLM去推理。就这么简单 — 没用什么复杂的cross-attention, 两个MLP一拼就完事。

作者自己也说, 试了更复杂的融合方式, 但简单的加法就够了, 因为两只眼本来就各管各的事, 不打架。

---

## 还有一个聪明的小动作: 选帧

一段房间视频可能有2000多帧, 但GPU只允许喂16帧。传统做法是均匀抽帧 — 每隔一定时间抽一帧。

问题是: 假如相机对着门口静止了30秒, 均匀抽帧会抽到一堆几乎一样的画面; 而相机快速扫过某个角落的那几帧(信息量最大的), 可能完全被跳过。

作者的做法特别巧妙:

1. 先抽128帧, 用VGGT算出每帧能看到场景的哪些3D区域
2. 把3D空间切成小方块(voxel)
3. 然后问: 选哪16帧能覆盖最多的小方块?

这是一个经典的"最大覆盖"问题 — 就像选超市地址, 要选能覆盖最多居民区的点位。用贪心算法(每次选能新增最多覆盖的那帧)就能近似解。

**关键: 16帧选定后, 3D特征不用重算** — 因为128帧的特征已经预算好了, 直接拿来用。聪明。

---

## 训练: 三步走

**第一步SFT**: 拿12万条空间问答数据, 教模型基本的空间问答能力。两只眼睛都冻住不动(保留各自的预训练能力), 只训练连接器和LLM。

**第二步Cold Start**: 先让模型学会"思考过程"的输出格式。用大模型(Qwen2.5-VL-72B)生成一堆带推理过程的答案, 筛出对的, 小规模微调200步。

**第三步GRPO**: 用强化学习进一步提升推理质量。对每道题生成8个答案, 互相比较打分, 好的强化、差的弱化。只跑1000步就有效果。

reward设计也分类型 — 选择题就严格对错, 数字题看误差大小(越接近满分越高), 文字题看字符串相似度。这个设计很关键, 因为数字题如果只用"对/错"二分法, 强化学习几乎学不动。

---

## 效果有多炸?

VSI-Bench上:

- **4B参数的Spatial-MLLM: 48.4分**
- 72B参数的Qwen2.5-VL: 37.0分
- Google的Gemini-1.5 Pro: 45.4分

也就是说, 40亿参数的小模型, 碾压了720亿参数的大模型, 甚至超过了Google的旗舰模型。而且Spatial-MLLM只看了16帧, Gemini看了85帧。

这充分说明: **瓶颈不在模型大小, 在于眼睛好不好使**。给它一双能看3D的眼睛, 4B就够用。

---

## 一句话总结

**给video AI加一只"3D眼"**, 用专门的几何模型从2D视频里提取空间结构信息, 和原来的语义信息拼在一起喂给LLM。小模型也能在空间理解上吊打大模型。选帧的时候用3D覆盖最大化来挑, 比瞎均匀采样聪明得多。强化学习再推一把reasoning能力。

核心insight就一个: **representation比scale重要**, 用对inductive bias, 事半功倍。

---

# Spatial-MLLM: 从2D视频重建空间智能的深度技术讲解

Andrej,这篇paper核心攻击的是一个非常fundamental的问题: **现有video MLLMs在spatial intelligence上系统性地薄弱**, 根源不在LLM backbone, 而在visual encoder的inductive bias。下面我从intuition层面、architecture层面、training层面逐层拆解, 并尽量补全每个公式里变量和上下标的语义。

---

## 1. 问题诊断: 为什么video MLLM在spatial reasoning上拉胯?

paper的第一段intuition非常关键。作者观察到一个asymmetry:

- 现有video MLLMs (Qwen2.5-VL, LLaVA-Video, InternVL2 等) 的visual encoder都是follow **CLIP paradigm** — 即在image-text pairs (主要是image-caption data) 上pretrain。这种supervision signal让encoder的feature space在**high-level semantics**上极度sharpened, 但geometric/spatial structure的信息基本被丢弃。
- 当task需要的是"这个房间的尺寸是多少平方米"、"A和B的最短距离是多少米"、"从我站的位置朝C看, B在左边还是右边"这类**metric-aware** reasoning时, CLIP feature提供的信息是**degenerate**的。

这是典型的 **representation bottleneck**: LLM backbone本身有reasoning能力, 但喂给它的visual token本身就是spatially impoverished的, 后面再怎么prompt、CoT、RL都没法recover丢失的几何信号。

paper给出的关键洞察是: **feed-forward visual geometry foundation models** (DUSt3R, VGGT, MegaSAM 这条lineage) 在pixel-point pairs上pretrain, 它们的backbone feature天然encode了3D structure prior, 而且能从pure 2D input forward inference出dense 3D结构。这两类pretrain是**互补**的, 一个抓semantics, 一个抓geometry。

参考链接:
- VGGT (CVPR 2025): https://vgg-t.github.io/
- DUSt3R (CVPR 2024): https://dust3r.europe.naverlabs.com/
- Qwen2.5-VL technical report: https://github.com/QwenLM/Qwen2.5-VL
- VSI-Bench ("Thinking in Space"): https://visual-arms.github.io/

---

## 2. Architecture: Dual-Encoder + Connector的精巧设计

### 2.1 整体数据流

输入是一个scene video $\mathcal{V} = \{\mathbf{f}_i\}_{i=1}^N$, 其中 $\mathbf{f}_i \in \mathbb{R}^{H \times W \times 3}$。由于GPU memory限制, 实际只喂 $N_k$ 帧 (inference时 $N_k=16$)。这 $N_k$ 帧分两路:

```
        ┌─── E_2D (Qwen2.5-VL ViT) ──→ e_2D (semantic)  ───┐
frames ─┤                                                  ├──→ Connector ──→ unified tokens e ──→ LLM f_θ ──→ answer
        └─── E_spatial (VGGT backbone) ──→ e_3D (geometry) ─┘
```

### 2.2 2D Encoder分支

公式(1):
$$
\mathbf{e}_{2\mathrm{D}} = \mathcal{E}_{2\mathrm{D}}(\{\mathbf{f}_i\}_{i=1}^{N_k}), \quad \mathbf{e}_{2\mathrm{D}} \in \mathbb{R}^{N_k' \times \lfloor H/p_{2\mathrm{D}} \rfloor \times \lfloor W/p_{2\mathrm{D}} \rfloor \times d_{2\mathrm{D}}}
$$

变量含义:
- $\mathbf{f}_i$: 第 $i$ 帧, $H \times W \times 3$ 的RGB图
- $N_k$: 实际输入帧数
- $p_{2\mathrm{D}}$: 2D encoder的patch size (Qwen2.5-VL中通常是动态的, 这里implicit假设固定)
- $d_{2\mathrm{D}}$: encoder输出feature的channel dimension
- $N_k' = \lceil N_k / 2 \rceil$: **关键细节** — Qwen2.5-VL把consecutive两帧打包成一组做temporal pooling, 所以feature的frame维度减半

直觉上, 这一支提供的是"这是什么" (what) 的信息 — object category, scene type, appearance。

### 2.3 Spatial Encoder分支 (VGGT backbone)

公式(2):
$$
\mathbf{e}_{3\mathrm{D}}, \mathbf{e}_c, \mathbf{e}_{\mathrm{register}} = \mathcal{E}_{\mathrm{spatial}}(\{\mathbf{f}_i\}_{i=1}^{N_k}), \quad \mathbf{e}_{3\mathrm{D}} \in \mathbb{R}^{N_k \times \lfloor H/p_{3\mathrm{D}} \rfloor \times \lfloor W/p_{3\mathrm{D}} \rfloor \times d_{3\mathrm{D}}}
$$

变量含义:
- $\mathbf{e}_{3\mathrm{D}}$: **dense 3D feature**, 每个token对应一个pixel patch, 但feature本身已经吸收了多帧cross-view的几何信息
- $\mathbf{e}_c$: per-frame的camera feature (后续接camera head出extrinsic $\mathbf{E}$ 和 intrinsic $\mathbf{K}$)
- $\mathbf{e}_{\mathrm{register}}$: register tokens, 来自Darcet et al. 2023的工作 "Vision Transformers Need Registers" (https://arxiv.org/abs/2309.16588), 是为了吸收high-norm的artifact tokens, 让dense feature更干净
- $p_{3\mathrm{D}}$: VGGT的patch size
- $d_{3\mathrm{D}}$: VGGT feature的channel dim

VGGT内部的机制是alternating **frame-wise self-attention** (单帧内的spatial attention) 和 **global self-attention** (跨帧的geometry aggregation), 这种alternating pattern让每帧的dense feature在保持pixel-level resolution的同时, 也能"看见"其他帧的perspective, 从而隐式recover 3D structure。

直觉上, 这一支提供的是"在哪儿、多大、什么形状" (where, how big, what shape) 的信息。

### 2.4 Connector: 极简但足够的融合

这里paper做了一个很pragmatic的设计选择 — 不用cross-attention, 直接用两个MLP然后相加。

公式(3) — 维度对齐:
$$
\mathbf{e}_{3\mathrm{D}}' = \mathrm{Rearrange}(\mathbf{e}_{3\mathrm{D}}), \quad \mathbf{e}_{3\mathrm{D}}' \in \mathbb{R}^{N_k' \times \lfloor H/p_{2\mathrm{D}} \rfloor \times \lfloor W/p_{2\mathrm{D}} \rfloor \times d_{3\mathrm{D}}'}
$$

这里要解决一个misalignment: VGGT是per-frame的 ($N_k$ 帧), Qwen2.5-VL是per-frame-pair的 ($N_k'$), 且两者的patch size可能不同 ($p_{2\mathrm{D}}$ vs $p_{3\mathrm{D}}$)。Rearrange操作把spatially/temporally相邻的信息aggregate到channel维度, 实现"位置对位置"的correspondence。

公式(4) — 融合:
$$
\mathbf{e} = \mathrm{MLP}_{2\mathrm{D}}(\mathbf{e}_{2\mathrm{D}}) + \mathrm{MLP}_{3\mathrm{D}}(\mathbf{e}_{3\mathrm{D}}')
$$

- $\mathbf{e} \in \mathbb{R}^{S \times d_{\mathrm{llm}}}$: 最终喂给LLM的visual token
- $S = N_k' \times \lfloor H/p_{2\mathrm{D}} \rfloor \times \lfloor W/p_{2\mathrm{D}} \rfloor$: token序列长度
- $d_{\mathrm{llm}}$: LLM的hidden dimension

**Intuition**: 这相当于在每个pixel-position上, 把semantic embedding和geometric embedding做element-wise add (在各自MLP投影到同一空间后)。简单相加之所以work, 是因为两个encoder已经被pretrain成在complementary subspace上活跃, 加法相当于channel-wise concat的廉价近似。如果用cross-attention, 反而可能overfit或破坏pretrain feature的structure。

参考链接:
- ViT Registers paper: https://arxiv.org/abs/2309.16588
- VGGT paper (CVPR 2025): https://arxiv.org/abs/2503.11651

---

## 3. Space-Aware Frame Sampling: 把frame selection变成set cover问题

这一块是paper的一个亮点, 设计得非常elegant。

### 3.1 问题动机

VSI-Bench里的scene video平均2000+帧, 但GPU memory只允许喂8-32帧。传统uniform sampling的assumption是"时间均匀=信息均匀", 这对temporal action recognition合理, 但对**3D scene coverage**完全错误:

- 相机静止时, uniform sample会选到大量redundant viewpoints
- 相机快速pan到某个新区域又回来时, 那个brief的transient region在uniform sample下可能完全missed

### 3.2 算法三阶段

**Stage 1: Scene geometry preprocessing**

先uniform subsample $N_m = 128$ 帧 (远多于最终需要的 $N_k = 16$), 用VGGT的backbone + camera head + depth head联合预测:

公式(5):
$$
\{\mathbf{E}_i^m, \mathbf{K}_i^m\}_{i=1}^{N_m} = f_c(\mathbf{e}_c), \quad \{\mathbf{D}_i^m\}_{i=1}^{N_m} = f_d(\mathbf{e}_{3\mathrm{D}})
$$

- $\mathbf{E}_i^m$: 第 $i$ 帧的camera extrinsic (4×4 SE(3)矩阵)
- $\mathbf{K}_i^m$: camera intrinsic (3×3 upper triangular)
- $\mathbf{D}_i^m$: per-pixel depth map

然后通过depth反投影重建3D point map, 公式(8):
$$
\mathcal{P}_i^m = \mathbf{D}_i^m \cdot \mathbf{K}_i^{-1}[\mathbf{u}|\mathbf{v}|]^\top \cdot \mathbf{E}_i^{-1}
$$

- $(\mathbf{u}, \mathbf{v})$: pixel coordinates (齐次化后)
- $\mathbf{K}_i^{-1}[\mathbf{u}|\mathbf{v}|]^\top$: 把pixel ray投影到camera坐标系的3D方向
- 乘 $\mathbf{D}_i^m$: 沿ray方向scale到具体3D点
- 乘 $\mathbf{E}_i^{-1}$: 从camera坐标系变换到world坐标系

每个point还有一个confidence $c(p) \in [0, 1]$, 来自VGGT的depth head。

**Stage 2: Voxelization & coverage calculation**

公式(9): 用confidence过滤valid points:
$$
\mathcal{P}_{\mathrm{valid}} = \bigcup_{i=1}^{N_m} \{p \in \mathcal{P}_i^m \mid c(p) > 0.1 \land c(p) \geq \mathrm{Percentile}(\{c(p)\}, 50\%)\}
$$

——既要绝对confidence > 0.1, 又要在该frame的相对top-50%, 双重过滤去除noise points。

公式(10): adaptive voxel size:
$$
\Delta = \frac{1}{\lambda} \cdot \min(\max(\mathcal{P}_{\mathrm{valid}}) - \min(\mathcal{P}_{\mathrm{valid}}))
$$

- $\lambda = 20$: hyperparameter, 表示把scene bounding box的最短维度切20份
- $\min(\max - \min)$: 取bounding box的shortest edge length
- 这样voxel size是**scene-scale adaptive**的, 因为VGGT输出是up-to-scale的

公式(11): 每帧的voxel coverage set:
$$
V(\mathbf{f}_i^m) = \left\{\left\lfloor \frac{p - \min(\mathcal{P}_{\mathrm{valid}})}{\Delta} \right\rfloor \Big| p \in \mathcal{P}_i^m \cap \mathcal{P}_{\mathrm{valid}}\right\}
$$

每个frame对应一个voxel index的集合, 表示"这帧能看到哪些voxel"。

**Stage 3: Greedy maximum coverage**

公式(12) — 优化目标:
$$
\max_{\mathcal{S} \subseteq \{1,\dots,N_m\}} \left| \bigcup_{i \in \mathcal{S}} V(\mathbf{f}_i^m) \right| \quad \text{s.t.} \quad |\mathcal{S}| = N_k
$$

这是经典的**maximum coverage problem** (Nemhauser-Wolsey-Fisher 1978, https://link.springer.com/article/10.1007/BF01588971), NP-hard但submodular, 所以greedy算法有 $(1 - 1/e) \approx 0.632$ 的approximation ratio。

Algorithm 1的greedy逻辑: 每次选能让已覆盖voxel集合 $\mathcal{C}$ 增长最多的frame, 直到选满 $N_k$ 个或没有更多coverage gain。

**优化trick**: 因为已经预计算了128帧的3D feature $\mathbf{e}_{3\mathrm{D}}^m$, 选出16帧后**直接复用**对应的feature, 不需要重新跑VGGT backbone。这是把inference cost和selection cost解耦的巧妙设计。

直觉上, 这套sampling策略相当于在做"active viewpoint selection" — 用geometric prior指导哪些view对3D understanding最informative。

---

## 4. 训练Pipeline: SFT → Cold Start → GRPO 三阶段

### 4.1 数据: Spatial-MLLM-120k

数据来源:
- ScanQA training set
- SQA3D 
- 自创spatial QA (follow VSI-Bench的pipeline)

数据schema: $\mathcal{T}_i = \langle \mathcal{Q}_i, \mathcal{A}_i, \mathcal{V}_i, \mathcal{M}_i \rangle$ — (question, answer, video ID, meta-info)

QA类型覆盖7类spatial task:
1. **Object counting** (numerical): "How many <category> are in this room?"
2. **Object size** (numerical): OBB longest side in cm
3. **Room size** (numerical): alpha-shape算法算的房间面积 in m²
4. **Absolute distance** (numerical): 两OBB内uniformly sampled point clouds的最短Euclidean距离 in m
5. **Appearance order** (multiple choice): 第一次visible pixel count超过threshold的时间戳排序
6. **Relative distance** (multiple choice): anchor object + 4 candidates with 15-30cm separation
7. **Relative direction** (multiple choice): triple {position, facing, query}, 计算vector夹角并discretize成left/right/front-left等

关键防泄漏: 排除了VSI-Bench evaluation set用到的312个scene video。

总数据量约120k (70k自创 + ScanQA + SQA3D)。

### 4.2 Stage 1: SFT

公式(6) — 标准cross-entropy loss:
$$
\mathcal{L}_{\mathrm{ce}}(\theta) = -\sum_i \log P(\mathbf{o}^{(i)} \mid \mathbf{o}^{(1:i-1)}, \mathbf{q}, \{f_j\}_{j=1}^{N_k})
$$

- $\mathbf{o}^{(i)}$: ground-truth answer的第 $i$ 个token
- $\mathbf{o}^{(1:i-1)}$: 前缀tokens (teacher forcing)
- $\mathbf{q}$: system prompt + question
- $\{f_j\}_{j=1}^{N_k}$: input video frames

**冻结策略**: $\mathcal{E}_{2\mathrm{D}}$ 和 $\mathcal{E}_{\mathrm{spatial}}$ 都frozen (它们已经被大规模pretrain过, 保留semantic和geometric prior), 只训练connector和LLM backbone。这避免了catastrophic forgetting, 让LLM学会"如何读"dual-encoder的输出。

Optimizer: Adam, 1 epoch, batch size 16, peak LR $10^{-5}$, linear schedule。

### 4.3 Stage 2: Cold Start (CoT format alignment)

GRPO之前需要一个**cold start**阶段 (200 steps) 来让模型熟悉long-CoT的输出format, 否则直接做RL会因为output distribution mismatch而崩。

构造流程 (Algorithm 2):
1. 从Spatial-MLLM-120k sample $N_s = 5000$ 条
2. 用Qwen2.5-VL-72B作为teacher, 每条生成 $K=3$ 条独立的reasoning路径 $\hat{\mathcal{T}}_i^{(k)}$ + 答案 $\hat{\mathcal{A}}_i^{(k)}$
3. 用GT答案算reward $r_i^{(k)}$
4. 每条选reward最高的: $k^* = \arg\max_k r_i^{(k)}$
5. **Adaptive filtering**: 按问题类型分组, 取每组reward的50%分位数作为type-specific threshold $\tau_t$
6. 保留 $\hat{r}_i \geq \tau_{t(i)} \land \hat{r}_i > 0$ 的item

最终cold start set有 **2459** 条 (从5000×3=15000次generation中筛出)。这个adaptive thresholding很关键, 避免了不同task类型下数据 imbalance 的问题 — 数值题的reward分布和选择题完全不同, 全局threshold会偏袒某类。

### 4.4 Stage 3: GRPO (Group Relative Policy Optimization)

公式(7) — GRPO目标:
$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q, o_i} \left[\frac{1}{G}\sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\mathrm{old}}}(o_i \mid q)} A_i, \mathrm{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\mathrm{old}}}(o_i \mid q)}, 1 \pm \epsilon\right) A_i\right) - \beta \mathrm{KL}[\pi_\theta \| \pi_{\mathrm{ref}}]\right]
$$

变量含义:
- $q$: prompt (含visual tokens)
- $o_i$: 第 $i$ 个sampled completion (一组共 $G=8$ 个rollouts)
- $\pi_\theta$: current policy
- $\pi_{\theta_{\mathrm{old}}}$: 旧policy (用于importance sampling ratio)
- $\pi_{\mathrm{ref}}$: reference policy (KL anchor, 防止policy跑飞)
- $A_i$: advantage, 用group-relative reward计算: $A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}$
- $\epsilon$: PPO clip range
- $\beta = 0.04$: KL coefficient

GRPO相比PPO的核心优势: **不需要value network**, 用group statistics代替baseline, 显著降低memory和compute。这对video MLLM这种本就memory-tight的场景特别合适。

GRPO参考链接:
- DeepSeekMath (GRPO原始paper): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1

### 4.5 Reward设计: Task-dependent

公式(13):
$$
\mathrm{Reward}(\mathcal{A}_{\mathrm{pred}}, \mathcal{A}_{\mathrm{gt}}) = \lambda_1 R_{\mathrm{format}} + \lambda_2 \begin{cases} R_{\mathrm{MC}} & \text{multiple-choice} \\ R_{\mathrm{MRA}} & \text{numerical} \\ R_{\mathrm{Verbal}} & \text{verbal} \end{cases}
$$

$\lambda_1 = \lambda_2 = 1$。

**Multiple-choice** (公式14): exact match
$$
R_{\mathrm{MC}} = \mathbb{I}(\psi(\mathcal{A}_{\mathrm{pred}}) = \psi(\mathcal{A}_{\mathrm{gt}}))
$$
$\psi$ 做whitespace stripping之类的normalization。

**Numerical** (公式15): Mean Relative Accuracy
$$
R_{\mathrm{MRA}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbb{I}\left(\frac{|\alpha(\mathcal{A}_{\mathrm{pred}}) - \alpha(\mathcal{A}_{\mathrm{gt}})|}{|\alpha(\mathcal{A}_{\mathrm{gt}})| + \epsilon} < \tau\right)
$$

- $\mathcal{T} = \{0.50, 0.55, \dots, 0.95\}$: 10个relative error threshold
- $\alpha$: numeric normalization (解析"大约3米" → 3.0)
- $\epsilon = 10^{-8}$: 防0除
- 直觉: 在多个容忍度下累计accuracy, 越接近GT得分越高, 不是binary的

这个MRA设计很关键 — 它给RL提供**dense, differentiable-ish**的信号, 如果用binary exact match, RL几乎没法收敛, 因为数值题的答案空间是连续的。

**Verbal** (公式16): Levenshtein ratio
$$
R_{\mathrm{Verbal}} = 1 - \frac{D_{\mathrm{Lev}}(\phi(\mathcal{A}_{\mathrm{pred}}), \phi(\mathcal{A}_{\mathrm{gt}}))}{|\phi(\mathcal{A}_{\mathrm{pred}})| + |\phi(\mathcal{A}_{\mathrm{gt}})|}
$$

- $D_{\mathrm{Lev}}$: Levenshtein edit distance
- $\phi$: text normalization
- 直觉: 字符串相似度的soft reward

另外还加了**reasoning length reward** (follow Video-R1, https://arxiv.org/abs/2502.13503), 鼓励更长的CoT, 显式地push模型"think more before answer"。

---

## 5. 实验数据解读

### 5.1 VSI-Bench主战场 (Table 1)

VSI-Bench有8个task, 分3类:
- **Configurational reasoning**: object counting, relative direction, absolute direction, route planning
- **Measurement estimation**: object size, room size, absolute distance
- **Spatiotemporal reasoning**: appearance order

关键数字对比:

| Model | Params | Avg. |
|-------|--------|------|
| Gemini-1.5 Pro | proprietary | 45.4 |
| GPT-4o | proprietary | 34.0 |
| LLaVA-Video-72B | 72B | 40.9 |
| LLaVA-OneVision-72B | 72B | 40.2 |
| Qwen2.5-VL-72B | 72B | 37.0 |
| Qwen2.5-VL-7B | 7B | 33.0 |
| Qwen2.5-VL-3B (base) | 3B | 30.6 |
| **Spatial-MLLM-4B** | **4B** | **48.4** |

最striking的发现: **4B的Spatial-MLLM比72B的Qwen2.5-VL高11.4个点, 比Gemini-1.5 Pro高3.0个点**, 且Spatial-MLLM只用16帧, Gemini用1FPS sampling (VSI-Bench平均85帧)。

这证明: **spatial intelligence的瓶颈不在scale, 在representation**。给对了geometry prior, 4B就能碾压72B。

特别看numerical question (最考验geometry的):
- Object counting: Spatial-MLLM **65.3** vs Gemini 56.2 vs Qwen2.5-VL-72B 25.1
- Object size: Spatial-MLLM 63.1 vs Gemini 64.1 vs Qwen2.5-VL-72B 54.5
- Room size: Spatial-MLLM 45.1 vs Gemini 43.6 vs Qwen2.5-VL-72B 38.8

Object counting上拉开9个点的gap尤其惊人, 说明VGGT prior让模型能更accurately segment和distinguish object instances。

### 5.2 ScanQA & SQA3D (Table 2, 4, 5)

ScanQA (val set, 4675 QA pairs):
- Spatial-MLLM: BLEU-1=44.4, CIDEr=91.8, EM-1=26.3
- Qwen2.5-VL-72B: BLEU-1=26.8, CIDEr=66.9, EM-1=24.0
- 提升: +17.6 BLEU-1, +24.9 CIDEr

SQA3D (test set, 3519 QA pairs):
- Spatial-MLLM: EM-1=55.9, EM-R1=58.7
- Qwen2.5-VL-72B: EM-1=47.0, EM-R1=50.9
- 在"Is"类问题+15.3, "Which"类+13.9

特别值得注意的是: Spatial-MLLM甚至**超过了一些用3D/2.5D input的模型**, 比如3D-LLM, LL3DA, Chat-Scene (这些都用point cloud作为input)。只有3D-LLaVA (用point cloud) 和Video-3D-LLM (用depth map) 在ScanQA上略胜。

这意味着: **2D video + geometry foundation model已经能implicit recover出足够用的3D structure**, 不需要显式point cloud input。

### 5.3 Ablation studies (Table 3)

三个核心ablation:

**RL的效果**:
- Spatial-MLLM-SFT-16: 46.1
- Spatial-MLLM-16 (with GRPO): 48.4
- +2.3个点, 仅用1000步GRPO就达到, ROI非常高

**架构的效果**:
- Qwen2.5-VL-3B-SFT-16: 40.0
- Qwen2.5-VL-7B-SFT-16: 42.0
- Spatial-MLLM-SFT-16: 46.1
- 同样数据训练, dual-encoder比7B单encoder还高4.1个点

**Frame sampling策略**:
- Spatial-MLLM-Uni-8: 43.8 → Spatial-MLLM-8: 46.1 (+2.3)
- Spatial-MLLM-Uni-16: 47.1 → Spatial-MLLM-16: 48.4 (+1.3)
- Spatial-MLLM-Uni-32: 48.4 → Spatial-MLLM-32: 49.3 (+0.9)

观察: **帧数越少, space-aware sampling的gain越大** (8帧时+2.3, 32帧时仅+0.9)。这非常符合intuition — 当budget紧张时, 选哪几帧更critical; budget充裕时uniform也基本够用。这个特性对real-world deployment很有价值, 因为production环境通常受memory严格约束。

---

## 6. 我的几个观察和延伸思考

### 6.1 Inductive bias injection > Scale

这篇paper最deep的take-away是: **当task的representation requirement和encoder的pretrain objective mismatch时, 加参数没用, 换encoder才有用**。这和Kolmogorov complexity的视角一致 — 如果信息在input stage就丢了, 后面再大的LLM都recover不回来。

类似philosophy的paper:
- SpatialVLM (Chen et al. CVPR 2024, https://arxiv.org/abs/2401.02311) — 用internet-scale 3D-aware data pretrain
- Geometry of LLMs系列工作

### 6.2 Feed-forward geometry models作为MLLM的"眼睛"

VGGT, DUSt3R, MegaSAM这一类feed-forward geometry model正在变成新的"foundation perceptual component"。传统SLAM是iterative + sparse, 而这些模型是**single-forward-pass + dense**, 天然适合作为MLLM的plug-in encoder。

我预期接下来会出现:
- 类似SAM之于detection的"VGGT之于MLLM"标准化
- Dual-encoder成为video MLLM的标配
- 把frame sampling、active view selection这类geometric reasoning前置于MLLM

### 6.3 GRPO在多模态RL中的潜力

Spatial-MLLM用1000步GRPO就拿到+2.3的提升, 远比纯SFT高效。结合R1在reasoning上的成功, 我猜测**多模态long-CoT + GRPO**会是接下来一年的主旋律。关键瓶颈在**reward design** — 这篇paper的task-dependent reward (MRA for numerical, EM for MC, Levenshtein for verbal) 是个很好的template。

延伸阅读:
- Video-R1: https://arxiv.org/abs/2502.13503
- R1-VL: https://arxiv.org/abs/2503.12937
- SpatialBot: https://arxiv.org/abs/2506.05198

### 6.4 可能的limitation / 我会push的方向

1. **VGGT的frame数限制**: VGGT本身对输入帧数有上界 (paper里用128), 超长video可能需要hierarchical approach
2. **Connector太简单**: 加性融合可能在更难的任务上吃亏, cross-attention或gated fusion可能更好
3. **RL只跑了1000步**: scaling RL step数 + 优化reward shaping可能还能再涨
4. **VSI-Bench数据 leakage风险**: 虽然作者排除了312个eval scenes, 但ScanNet本身是公开的, 自创QA的prompt template可能和eval重合
5. **Outdoor / unbounded scene**: 全部实验在indoor ScanNet上, outdoor (KITTI, nuScenes) 是否transfer未知

### 6.5 Architecture图的关键走读 (Figure 2)

把Figure 2拆开看:
- **左边input**: 原始video sequence (2000+帧)
- **Space-aware sampling module (中间偏左)**: 用VGGT预compute 128帧的geometry → greedy select 16帧
- **Dual-encoder (中间)**: 上下两路并行
  - 上路: Qwen2.5-VL ViT → semantic feature
  - 下路: VGGT backbone → dense 3D feature
- **Connector (中间偏右)**: Rearrange + 双MLP + add
- **LLM (右边)**: Qwen2.5-VL的LLM backbone, 输入unified visual tokens + text prompt → long-CoT → answer

整个pipeline的设计哲学是**"让专业的encoder做专业的事"** — Qwen2.5-VL管semantics, VGGT管geometry, LLM管reasoning, 三者各司其职, 通过轻量connector串联。

---

## 7. 总结: 这篇paper的intellectual贡献层级

1. **Diagnostic层**: 精准定位video MLLM spatial intelligence瓶颈在encoder pretrain objective, 而非LLM规模
2. **Architectural层**: Dual-encoder + lightweight connector, 优雅地把VGGT的geometry prior注入MLLM, 不需要任何3D/2.5D input
3. **Inference-time层**: Space-aware frame sampling, 把active view selection formalize为maximum coverage, 用geometric prior指导information-theoretic最优frame selection
4. **Training层**: SFT + cold start + GRPO三阶段, task-dependent reward设计, 让long-CoT spatial reasoning可RL化
5. **Empirical层**: 4B模型在VSI-Bench / ScanQA / SQA3D上全面SOTA, 部分超过用3D input的模型

这是一篇**"用对的inductive bias换掉brute-force scale"**的教科书式工作, 跟你之前在Yi Tay的种种讨论中"representation matters more than scale for specific capability"的thesis高度resonance。

Project page: https://diankun-wu.github.io/Spatial-MLLM/

如果你想进一步深挖, 我特别推荐看:
- VGGT的alternating attention设计 (CVPR 2025)
- VSI-Bench原始paper的human evaluation部分 (https://arxiv.org/abs/2412.14171) — 人类在spatial task上几乎是ceiling, MLLM的gap还很远
- DeepSeekMath的GRPO derivation (https://arxiv.org/abs/2402.03300) — 比R1 paper更详细地讲了GRPO的math

需要我针对某一节再展开 (比如GRPO的优势函数推导、VGGT的alternating attention的具体mask pattern、或者ScanNet OBB metadata的计算细节), 尽管push。
