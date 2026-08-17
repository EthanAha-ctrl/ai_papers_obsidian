---
source_pdf: SOFAR.pdf
paper_sha256: 23afde6a58ab2752bdcebcae8c3937f1a4973e3b8664677c841419fc9c2c6298
processed_at: '2026-08-12T08:40:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SOFAR

## 一句话说清楚这篇 paper 在干嘛

让机器人听懂"把刀刃朝下放"这种话。

## 为啥这事难

你跟机器人说"把杯子放到右边"，现在的 VLM 能搞定——它知道"杯子"在哪，"右边"是哪边。但你要是跟它说"把杯子倒过来"或者"把 USB 插进去"，它就懵了。

为啥懵？因为"倒过来"和"插进去"涉及的是 **orientation（朝向）**，不是 position（位置）。现在的 AI 对 position 理解还不错，对 orientation 基本上是瞎的。

你可以这么想：现在的 AI 看一张桌子上的刀，它能告诉你"刀在桌子左边"，但它分不清刀刃是朝左还是朝右。对人来說一眼就看出来了，对 AI 来说特别难，因为：

- 传统的 orientation 表示用的是 quaternion 或者 Euler angles，就是一堆数字，没有 semantic meaning
- 机器人要插 USB，它得知道"插进去那个方向"，而不是"rotation matrix 是多少多少"
- 传统方法要么需要 CAD model 模板，要么需要 predefined reference frame，换个没见过的东西就不 work

## SOFAR 的核心 idea

作者提出了一个概念叫 **Semantic Orientation**：用自然语言来定义方向。

啥意思呢？拿一把刀举例：
- "cutting direction" → 刀刃指向的那个向量
- "handle direction" → 刀把指向的那个向量
- "side direction" → 刀的侧面指向的那个向量

每个 language description 对应一个 3D unit vector。你给模型一个 object 的 3D point cloud 和一句话，它给你吐回一个方向向量。

这玩意妙在哪？妙在它是 **reference-frame-free** 的。你不用管坐标系怎么摆，不用管 quaternion 怎么转，直接用人类语言就能拿到 geometric direction。而且同一个 object 可以有多个 semantic orientations，组合起来就能完整描述它的 rotation。

## 数据从哪来——OrienText300K

作者从 Objaverse（一个 80 万 3D models 的数据集）里搞数据。但原始数据很脏，所以做了一轮严格的 filtering。

### Filtering 的六条标准

1. 只保留 standard 6 views 对齐的 objects
2. 移除带 ground plane 的
3. 保留有 spatial reasoning 价值的
4. 过滤 blurry 和错误的
5. 移除抽象无意义的
6. 只留 single object，不要 scene

### 关键 trick：用 GPT-4o 当 judge

手动 filter 80 万个 3D model 根本不现实，所以作者用 GPT-4o 来做 filtering 和 annotation。

这里有个 insight 挺有意思的：**VLM 是 poor generator 但是 excellent discriminator**。你让 GPT-4o 直接输出一个 orientation 的数值，它输出的基本都是 garbage。但你给它 6 张 standard view 图片，让它判断"哪张图里刀刃正对着镜头"，它能判断得很准。

所以整个 annotation pipeline 就是：
1. 用 Blender 渲染每个 object 的 6 个 standard view
2. 把 6 张图拼起来喂给 GPT-4o
3. GPT-4o 输出 (language description, view index) pairs
4. view index 对应一个已知的 direction vector（因为 standard view 的 camera 方向是固定的）

最后验证下来，GPT-4o 在 filtering 上 88.3% accuracy，在 annotation 上 97.1% accuracy，质量相当不错。

最终搞出来 **350K objects + 8M images**，作者还给这个数据集起了个名叫 OrienText300K。

## PointSO 模型怎么设计的

### 架构

整体就是个 plain Transformer，input 是：
- Object 的 3D point cloud（10000 个点）
- 一句 language description（比如 "handle"）

3D point cloud 的处理：
1. **FPS (Farthest Point Sampling)** 采 512 个 seed points——目的是在空间上均匀采样，不是随机采
2. 每个 seed point 周围用 **KNN** 圈 32 个邻居点，形成一个 patch
3. 每个 patch 用一个小 **PointNet** 编码成 feature vector
4. 这些 feature vector 当作 tokens 喂进 Transformer encoder

Language 的处理：用 **CLIP text encoder**（frozen，不训练），取它的 [CLS] token 当作 text feature。

### 关键 design choice：Fusion 用 addition

这是 paper 里一个反直觉的发现。多模态融合有四种常见做法：
- Cross-attention
- Multiplication（element-wise 乘）
- Addition（element-wise 加）
- Concatenation

Empirical 结果是 **addition 最好**：

| Fusion Method | Average Accuracy |
|--------------|-----------------|
| Cross-attn | 66.21% |
| Multiplication | 65.04% |
| **Addition** | **72.46%** |
| Concat | 55.86% |

为啥 addition 反而最好？作者的 hypothesis 是：semantic orientation 的 language input 通常都很短（"handle", "top", "plug-in"），CLIP 的 [CLS] token 已经编码了 sufficiently high-level 的 semantic information。用 addition 的话，text feature 在每一层都直接加到每个 point token 上，influence 能 propagate 得更深。Cross-attention 反而可能 overfit 或者产生 information bottleneck。

### Loss function

用 negative cosine similarity：

$$\mathcal{L}_{\text{cos}}(\mathbf{v}, \mathbf{k}) = 1 - \frac{\mathbf{v} \cdot \mathbf{k}}{\|\mathbf{v}\| \cdot \|\mathbf{k}\|}$$

其中 $\mathbf{v}$ 是 predicted vector，$\mathbf{k}$ 是 ground truth vector。

为啥用 cosine？因为 semantic orientation 是 unit vector，只关心 direction 不关心 magnitude。Cosine similarity 直接 measure 两个 vectors 的 angle：
- 完全对齐：loss = 0
- 垂直：loss = 1
- 反向：loss = 2

总 loss 就是把所有 object 的所有 orientation pair 的 cosine loss 加起来：

$$\min_{\theta_{\text{SO}}} \sum_{X_i \in \mathcal{D}} \sum_{\ell_j^i \in L_i} \mathcal{L}_{\text{cos}}\left(\mathcal{F}_{\text{SO}}(X_i, \ell_j^i), \mathbf{s}_j^i\right)$$

其中 $\theta_{\text{SO}}$ 是模型参数，$X_i$ 是第 $i$ 个 object，$L_i$ 是它的 language description 集合，$\mathbf{s}_j^i$ 是对应的 ground truth direction。

### Robustness 怎么样

Real world 的 point cloud 通常不完整、有噪声。作者测试了三种 perturbation：
- **Single-View**: 只从一个随机视角观测（partial point cloud）
- **Jitter**: 加 Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.01^2)$
- **Rotate**: 随机 SO(3) rotation

PointSO-L 在 "All" corruption（三种叠加）下还能保持 74.22% accuracy（45° threshold），相当 robust。

## SOFAR 整个 system 怎么跑的

PointSO 只能处理单个 object，但真实任务是在 scene 里做的。SOFAR 就是把 PointSO 和一堆 foundation model 串起来，做成一个完整的 system。

### Pipeline

给一张 RGB-D image 和一句 instruction，流程是：

1. **VLM 提取 object phrases**：比如指令是"把刀拿起来切面包"，VLM 提取出"刀"和"面包"
2. **SAM + Florence-2 做 segmentation**：在 image 里分割出刀和面包的 mask，用 RealSense D415 的 depth 信息转成 3D point cloud
3. **VLM 生成 orientation descriptions**：对刀生成 "handle direction" 和 "cutting direction"
4. **PointSO 推断 orientations**：输入刀的 point cloud 和 "handle direction"，输出对应的 3D vector
5. **构建 6-DoF Scene Graph**：每个 object 作为 node，编码 position、size、orientation set
6. **VLM 做 Chain-of-Thought reasoning**：基于 scene graph 和 image，算出 target pose
7. **Motion planning 执行**：用 OMPL 规划轨迹，机械臂去抓

### 6-DoF Scene Graph 长啥样

每个 node $\mathbf{o}_i$ 包含：
- Object phrase + unique ID
- 3D position $\mathbf{c}_i = (x, y, z)$（centroid）
- Bounding box size $\mathbf{b}_i = (h, w, l)$
- Semantic orientation set $S_i$ + description set $L_i$

每个 edge $\mathbf{e}_{ij}$ 包含：
- 两个 objects 之间的 relative translation
- Size ratio

### Rotation 怎么算的

这里用了一个经典算法叫 **Kabsch-Umeyama algorithm**。

场景：你有一个 mug 的初始 orientation set $S_i = \{\mathbf{s}_1, \mathbf{s}_2, \ldots\}$（比如 handle direction 和 opening direction），你算出了 target orientation set $\tilde{S}_i = \{\tilde{\mathbf{s}}_1, \tilde{\mathbf{s}}_2, \ldots\}$。现在要找一个 rotation matrix $\mathbf{R}$ 把 $S_i$ 转到 $\tilde{S}_i$。

目标：

$$\mathbf{R}^* = \arg\min_{\mathbf{R}} \sum_{k} \|\tilde{\mathbf{s}}_k - \mathbf{R} \mathbf{s}_k\|^2$$

算法步骤：
1. 算 cross-covariance matrix：$\mathbf{H} = \sum_k \tilde{\mathbf{s}}_k \mathbf{s}_k^T$
2. SVD 分解：$\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$
3. Optimal rotation：$\mathbf{R} = \mathbf{V} \mathbf{U}^T$（保证 $\det(\mathbf{R}) = 1$，是 proper rotation）

妙在哪？如果你只用一个 vector 对齐，rotation 还有一个 degree of freedom 不确定（绕这个 vector 的自转）。但如果你同时 align handle direction 和 opening direction，两个 vector 就能把 rotation 完全 lock 住。

## 实验结果有多强

### Open6DOR V2（6-DoF Object Rearrangement Benchmark）

**Perception tasks**（Isaac Sim，只算 final pose 不实际执行）：

| Method | Position | Rotation | 6-DoF | Time |
|--------|----------|----------|-------|------|
| GPT-4V | 45.2 | 9.2 | - | - |
| Open6DOR-GPT | 74.9 | 41.1 | 35.6 | 126.3s |
| **SOFAR** | **93.0** | **57.0** | **48.7** | **8.5s** |

Rotation track 上 SOFAR 比 Open6DOR-GPT 高了 16%，这是 semantic orientation 带来的直接收益。Time cost 从 126 秒降到 8.5 秒，快了 15 倍。

**Execution tasks**（Libero，真要机械臂执行）：

| Method | 6-DoF Overall |
|--------|---------------|
| Octo | 8.0 |
| OpenVLA | 8.2 |
| **SOFAR** | **18.4** |

Octo 和 OpenVLA 是用大量 robot trajectory data 训练的，SOFAR 是 zero-shot，结果反而高了一倍多。

### SIMPLER-Env（Google Robot 模拟环境）

**Visual Matching setting**：

| Policy | Average |
|--------|---------|
| RT-1-X | 53.4% |
| RT-2-X | 60.6% |
| OpenVLA | 27.7% |
| **SOFAR** | **74.9%** |

这个结果挺惊人的。RT-2-X 是 Google 用 OXE 大规模 robot data 训练的 VLA model，SOFAR zero-shot 就能超过它 14%。

### 6-DoF SpatialBench（VQA benchmark）

| Method | Orientation (relative) | Total |
|--------|----------------------|-------|
| GPT-4o | 44.2 | 36.2 |
| SpatialBot | 39.6 | 32.7 |
| **SOFAR** | **54.6** | **43.9** |

Orientation track 上比 GPT-4o 高了 10.4%，说明 PointSO 给 VLM 加上 orientation information 之后，reasoning 能力确实上了一个台阶。

### Real-world 实验

60 个 task，100+ objects，每个 task 跑 3 次：

| Track | CoPa | ReKep | SOFAR-LLaVA | **SOFAR** |
|-------|------|-------|-------------|-----------|
| Position | 74.7% | 72.0% | 81.3% | **85.3%** |
| Orientation | 21.7% | 23.3% | 35.0% | **43.3%** |
| 6-DoF | 20.0% | 26.7% | 33.3% | **48.9%** |

Orientation track 上 SOFAR 是 baseline 的两倍多。这个 track 要求很高的 angle precision（upright/upside-down 要 <5°），能到 43.3% 已经相当不错。

## Ablation 告诉我们什么

### Semantic Orientation 到底有没有用

| CoT | Orient. | Rotation Overall | 6-DoF Overall |
|-----|---------|-----------------|---------------|
| ✗ | ✗ | 13.0 | 14.2 |
| ✓ | ✗ | 12.9 | 13.7 |
| ✗ | ✓ | 52.3 | 45.8 |
| ✓ | ✓ | **57.0** | **48.7** |

关键发现：**只加 CoT 没用**（13.0 → 12.9），**加 Semantic Orientation 才有大提升**（13.0 → 52.3）。这说明 performance 提升的核心确实是 orientation information，不是 reasoning framework 带来的。

### Data scaling

| Data Scale | Average Accuracy |
|-----------|-----------------|
| 15K | 42.58% |
| 35K | 47.27% |
| 150K | 67.97% |
| 350K | 72.46% |

从 15K 到 350K（23x），accuracy 从 42% 提到 72%。这表明 PointSO 有 foundation model 的 scaling property，数据越多效果越好。作者提到如果用 Objaverse-XL（10M+ objects）效果应该会更进一步。

## 失败案例分析

作者统计了 real-world 实验的 failure cases：

- **31% grasping 出问题**：object 太小、抓不稳、滑掉
- **23% orientation 预测不准**：特别是需要 <5° precision 的任务
- **20% object detection 出错**：Florence-2 对 OOD objects 误检
- **16% motion planning 死锁**：没考虑 arm workspace 和 collision
- **10% VLM planning 出错**：复杂 orientation 推理失败

Intuition：low-level execution 还是 bottleneck，特别是 grasping。作者说未来可能需要更强的 grasping policy 或者把 decoupled method 和 end-to-end 结合起来。

## 跟其他工作的区别

### vs. ReKep

ReKep 用 spatial keypoint constraints，有三个问题：
1. 每个任务要手写 complex system prompt，不是真 zero-shot
2. 只用 keypoint 对齐，capture 不了完整 6-DoF（比如水壶倒水只对齐 spout 到杯子，水壶可能翻了）
3. 依赖第一帧的所有 keypoints，pipeline 引入一堆 hyperparameters

SOFAR 真的是 zero-shot，不需要 per-task prompt engineering。

### vs. Orient Anything

Orient Anything 也是学 orientation，但只学 "front", "top" 这种 basic direction。SOFAR 是 language-conditioned，可以处理 "plug-in", "cutting", "handle" 这种 functional semantics，能直接 integrate 到 VLM reasoning pipeline。

### vs. End-to-End VLA

RT-2, OpenVLA 这种 end-to-end VLA model 需要 robot trajectory data training，数据获取成本高。SOFAR 用 3D object data + foundation model reasoning，zero-shot 就能超过它们，data efficiency 高很多。而且 SOFAR 是 embodiment-agnostic 的，换个 gripper、suction cup、dexterous hand 都能用。

## 这篇 paper 的意义

### Concept 层面

提出了一个新 representation：**language-grounded orientation**。这个 representation 把 functional semantics 和 geometric direction 绑在一起，解决了 robot manipulation 里的 orientation understanding 难题。

类似 CLIP 把 image 和 language 对齐、act 作为 vision-language foundation model，PointSO 把 object geometry 和 language 对齐、act 作为 orientation foundation model。

### Technical 层面

1. **OrienText300K**：第一个大规模 semantic orientation dataset，用 GPT-4o auto-annotate
2. **PointSO**：第一个 semantic orientation prediction model，architecture simple 但 effective
3. **SOFAR system**：把 PointSO 和 VLM/SAM 串起来，build 6-DoF scene graph
4. **Open6DOR V2**：6-DoF rearrangement benchmark，支持 open-loop 和 closed-loop evaluation
5. **6-DoF SpatialBench**：第一个 orientation-aware VQA benchmark

### 实验层面

SOFAR 在多个 benchmark 上 zero-shot SOTA：
- Open6DOR: 48.7%（vs 35.6% baseline）
- SIMPLER-Env: 74.9%（vs 60.6% RT-2-X）
- 6-DoF SpatialBench: 43.9%（vs 36.2% GPT-4o）

## 我的一些直觉

读完这篇 paper，我有几个感受：

1. **Language-grounded representation 是 trend**。从 CLIP 到 PointSO，把 geometric/mathematical representation 和 language 对齐，能让 model 借用 VLM 的 reasoning 能力和 world knowledge，这是比纯 geometric approach 更 scalable 的路径。

2. **Data-centric thinking 很关键**。作者没有去 design 更复杂的 architecture，而是去 build 一个高质量的大规模 dataset。PointSO 的 architecture 其实很 simple（plain Transformer + addition fusion），但 data quality 和 scale 才是 performance driver。这和 Scaling Law 的精神一致。

3. **Decoupled approach 还是有竞争力**。虽然 end-to-end VLA 是热点，但 SOFAR 证明 decoupled approach（reasoning + execution）在 zero-shot generalization 上仍有优势。Future direction 可能是 hybrid：用 decoupled approach 做 high-level reasoning，用 end-to-end model 做 low-level execution。

4. **GPT-4o 当 annotator 很 powerful**。作者花了 $10K API cost 就 auto-annotate 了 350K objects，这个 cost-efficiency 非常高。未来 research 里 LLM-as-annotator 会越来越普遍。

5. **Failure mode 很 informative**。31% failure 来自 grasping，说明 low-level execution 还是 bottleneck。这暗示未来 robot learning 的瓶颈可能在 low-level control，不在 high-level reasoning。

参考链接:
- [NeurIPS 2025](https://neurips.cc/)
- [Objaverse Dataset](https://objaverse.allenai.org/)
- [GPT-4o System Card](https://arxiv.org/abs/2410.21276)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [PointNet](https://arxiv.org/abs/1612.00593)
- [SAM](https://arxiv.org/abs/2304.02643)
- [Florence-2](https://arxiv.org/abs/2311.06242)
- [Kabsch Algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)
- [Open6DOR](https://arxiv.org/abs/2406.11118)
- [SIMPLER-Env](https://arxiv.org/abs/2410.24048)
- [ReKep](https://arxiv.org/abs/2409.01652)
- [Orient Anything](https://arxiv.org/abs/2412.18605)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RT-2](https://arxiv.org/abs/2307.15818)

---

# SOFAR: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation 深度解析

## 1. Core Insight: Semantic Orientation 的本质

这篇 paper 的核心洞察在于将 object orientation 从 geometric representation 解耦为 language-grounded semantic representation。传统的 6-DoF pose estimation 依赖 quaternion 或 Euler angles，这些 representation 对于 robot manipulation 来说存在一个根本问题：它们是 reference-frame-dependent 的，且缺乏 semantic grounding。

SOFAR 提出的 **Semantic Orientation** 定义为：

$$\mathbf{s}_{\ell}^{X} = \mathcal{F}(X, \ell) \tag{1}$$

其中：
- $X$ 表示一个 3D object
- $\ell$ 表示 open-vocabulary language description（如 "handle", "plug-in", "cutting"）
- $\mathbf{s}_{\ell}^{X} \in S(2)$ 表示单位球面上的一个向量，即对应于 $\ell$ 的 semantic direction
- $\mathcal{F}$ 表示从 object-language pair 到方向向量的映射函数

这个 formulation 的精妙之处在于它将 object 的 functional semantics 直接编码为 geometric direction。一个 object $X$ 可以关联多个 semantic orientations，形成集合 $S_X = \{\mathbf{s}_{\ell_1}^X, \mathbf{s}_{\ell_2}^X, \ldots, \mathbf{s}_{\ell_n}^X\}$，这实际上构成了一个 semantic basis for describing object's rotation space。

**Intuition building**: 想象你让机器人"把刀的刀刃朝下放"，传统方法需要定义什么是"刀刃方向"，可能需要 CAD model alignment 或 template matching。Semantic Orientation 让你直接用 "blade direction" 这个 language query 来获取对应的 unit vector，这个 vector 就是刀刃指向的 geometric direction。

参考链接:
- [NeurIPS 2025 Official Site](https://neurips.cc/)
- [Objaverse Dataset](https://objaverse.allenai.org/)

## 2. OrienText300K: Data Construction Pipeline

### 2.1 Data Filtering Strategy

从 Objaverse 的 ~800K 3D models 出发，论文设计了六条严格的 filtering criteria：

1. **Standard orthogonal view only**: 只保留在 standard 6 views (front/back/left/right/top/bottom) 下对齐的 objects
2. **Clean objects without ground**: 移除用于辅助可视化的 ground plane
3. **Reasonable objects**: 保留具有 sufficient spatial reasoning potential 的 objects
4. **High-quality objects**: 过滤 blurry 或错误样本
5. **Distinguishable objects**: 移除 abstract 或 meaningless 的 objects
6. **Non-scene objects**: 专注于 object-centric understanding

**Key innovation**: 使用 GPT-4o 作为 human-aligned judge 来执行 filtering。将 multi-view images 拼接后输入 GPT-4o，让它判断样本是否满足上述六条标准。验证集显示 GPT-4o 在 filtering 上达到 **88.3% accuracy**，在 annotating 上达到 **97.1% accuracy**。

### 2.2 Data Annotation Process

这里有一个非常巧妙的 insight：**VLMs 是 poor generators 但 excellent discriminators**。

VLMs 无法直接产生准确的 orientation 数值，但它们能通过 multimodal understanding 区分不同 views。因此 annotation pipeline 设计为：

1. 假设 filtering 阶段已经移除了大量 misaligned data，剩下的 objects 能够产生 standard orthogonal views
2. 让 GPT-4o interpret 六个 views 的 semantic content
3. 生成 semantic-view pairs，即 (language description, view index) pairs

例如，对于一把刀，GPT-4o 可能生成：
- ("blade direction", view_index=2)  # blade 朝向 camera 的 view
- ("handle direction", view_index=4)

**Technical detail**: 整个 annotation 过程结合了 human modelers 在 Objaverse 中的原始 annotations 和 ChatGPT 的 semantic understanding，cost 约 $10K API 调用费用。

最终数据集统计：
- **350K+ clean 3D objects**
- **8M rendered images** (通过 Blender 在精心设计的 lighting conditions 下渲染)
- Diverse orientation-text pairs 覆盖 intra-object spatial reasoning 和 inter-object manipulation contexts

参考链接:
- [GPT-4o System Card](https://arxiv.org/abs/2410.21276)
- [Objaverse-XL](https://objaverse.allenai.org/objaverse-xl.pdf)

## 3. PointSO: Model Architecture Deep Dive

### 3.1 Architecture Overview

PointSO 是一个 plain Transformer-based architecture，专门设计用于 cross-modal 3D-language fusion。其核心设计 choice 是 **simplicity over complexity**——用 token-wise addition 替代 cross-attention 来实现 multi-modal fusion。

**Input processing**:
- **3D point cloud**: $X = \{\mathbf{x}_i \in \mathbb{R}^3 | i = 1, 2, \ldots, N\}$ with $N$ 3D points in Cartesian space
- **Language description**: arbitrary text $\ell$

**3D Embedding pipeline**:
1. **FPS (Farthest Point Sampling)**: 采样 $N_s$ seed points，确保 spatial coverage
2. **KNN grouping**: 对每个 seed point，用 K-Nearest Neighbors 形成 local patches
3. **Patch encoder**: 使用 lightweight PointNet 提取 local geometric features
4. **Transformer encoder**: 标准 Transformer 处理 patch tokens
5. **[CLS] token head**: MLP 将 [CLS] token 映射到 3D vector space，输出 predicted direction

**Language embedding**: 使用 frozen CLIP text encoder，取 global token 作为 cross-modal fusion input。

### 3.2 Cross-Modal Fusion: Why Addition Works

论文 empirically 比较了四种 fusion methods：

| Fusion Method | 45° Acc | 30° Acc | 15° Acc | 5° Acc | Avg |
|--------------|---------|---------|---------|--------|-----|
| Cross-attn | 74.22 | 70.31 | 63.28 | 57.03 | 66.21 |
| Multiplication | 74.22 | 69.53 | 60.16 | 56.25 | 65.04 |
| **Addition** | **79.69** | **77.34** | **70.31** | **62.50** | **72.46** |
| Concat | 66.41 | 60.94 | 52.34 | 43.75 | 55.86 |

**Intuition for why addition wins**: Semantic orientation 的 language inputs 通常是 short phrases（如 "handle", "top", "plug-in"）。CLIP 的 [CLS] token 已经编码了 sufficiently high-level semantic information。通过 token-wise addition，text feature 在每一层都 reinforce 其 influence on point tokens，避免了 cross-attention 可能带来的 overfitting 或 information bottleneck。

数学上，如果 $\mathbf{p}_i$ 是第 $i$ 个 point token，$\mathbf{t}$ 是 text global token，则 addition fusion 为：

$$\mathbf{h}_i^{(l)} = \text{TransformerLayer}(\mathbf{p}_i^{(l-1)} + \mathbf{t})$$

这种 simple broadcast 机制确保了 language condition 在每一层都直接参与 point feature 的更新。

### 3.3 Optimization Objective

PointSO 使用 negative cosine similarity loss：

$$\min_{\theta_{\text{SO}}} \sum_{X_i \in \mathcal{D}_{\text{OrienText300K}}} \sum_{\ell_j^i \in L_i} \mathcal{L}_{\text{cos}}\left(\mathcal{F}_{\text{SO}}(X_i, \ell_j^i), \mathbf{s}_j^i\right) \tag{2}$$

其中：
- $\theta_{\text{SO}}$ 是 PointSO 的可训练参数（CLIP 保持 frozen）
- $X_i$ 是 OrienText300K 中的第 $i$ 个 object point cloud
- $L_i = \{\ell_j^i | j = 1, 2, \ldots, Q\}$ 是第 $i$ 个 object 的 language description 集合
- $\mathbf{s}_j^i$ 是对应于 $\ell_j^i$ 的 ground truth semantic orientation
- $\mathcal{L}_{\text{cos}}(\mathbf{v}, \mathbf{k}) = 1 - \frac{\mathbf{v} \cdot \mathbf{k}}{\|\mathbf{v}\| \cdot \|\mathbf{k}\|}$ 是 negative cosine similarity

**Why cosine similarity**: Semantic orientation 是 unit vector，我们只关心 direction 而非 magnitude。Cosine similarity 直接 measure 两个 vectors 之间的 angle，天然适合 directional prediction task。当 predicted vector 与 ground truth 完全对齐时，loss = 0；当正交时，loss = 1；当反向时，loss = 2。

### 3.4 Model Variants & Scaling

| Model | CLIP | Layers | Hidden size | MLP size | Heads | #Params |
|-------|------|--------|-------------|----------|-------|---------|
| Small | ViT-B/32 | 12 | 256 | 1024 | 4 | 11.4M |
| Base | ViT-B/32 | 12 | 384 | 1536 | 6 | 19.0M |
| Large | ViT-B/32 | 12 | 512 | 2048 | 8 | 43.6M |

**Scaling law analysis**: 从 15K → 350K data scale，performance 从 42.58% Avg 提升到 72.46% Avg，提升约 **30%**。这表明 PointSO 具有良好的 data scaling property，随着数据增加性能持续提升。

**Robustness evaluation**: 在三种 perturbations 下测试：
- **Single-View**: 随机 camera viewpoint 生成 single FoV 观测
- **Jitter**: Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.01^2)$
- **Rotate**: Random SO(3) rotation $(\alpha, \beta, \gamma) \sim \mathcal{U}(-\pi, \pi)$

PointSO-L 在 "All" corruption 下仍保持 74.22% accuracy (45° threshold)，显示 strong robustness。

参考链接:
- [PointNet Original Paper](https://arxiv.org/abs/1612.00593)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 4. SOFAR System: 6-DoF Scene Graph Construction

### 4.1 Pipeline Overview

SOFAR 的核心 system design 是将 object-centric orientation understanding 扩展到 scene-level spatial reasoning。整个 pipeline 包括：

1. **VLM extracts object phrases**: 给定 language query $Q$，VLM $\mathcal{F}_{\text{VLM}}$ 提取 task-relevant object phrases $\mathcal{P} = \{p_i | i = 1, 2, \ldots, M\}$
2. **SAM & Florence-2 segmentation**: Language-conditioned segmentation 得到 object set $\mathcal{X} = \{X_i | i = 1, 2, \ldots, M\}$，每个 object 分配 unique ID 用于 Set-of-Mark (SoM) prompting
3. **PointSO orientation estimation**: VLM 生成 task-specific orientation descriptions $L_i$，PointSO 推断 semantic orientations $S_i$
4. **6-DoF Scene Graph construction**: 构建 graph $\mathcal{G} = (\mathbf{V}, \mathbf{E})$

### 4.2 6-DoF Scene Graph Structure

**Node attributes** $\mathbf{o}_i \in \mathbf{V}$:
1. **Object phrase** $p_i$ with unique instance ID
2. **3D position** $\mathbf{c}_i = (x, y, z) \in \mathbb{R}^3$ from object centroid
3. **Bounding box size** $\mathbf{b}_i = (h, w, l) \in \mathbb{R}^3$
4. **Semantic orientation set** $S_i$ with corresponding description set $L_i$

**Edge attributes** $\mathbf{e}_{ij} \in \mathbf{E}$:
- Relative translation between connected objects $\mathbf{o}_i$ and $\mathbf{o}_j$
- Size ratio between two objects

**Intuition**: 这个 scene graph 实际上是一个 **6-DoF spatial knowledge graph**。传统 scene graph 只包含 "on", "next to", "behind" 等 positional relations，SOFAR 的 scene graph 额外编码了每个 object 的 functional orientations。例如，对于一个 mug，graph 不仅记录它的位置，还记录它的 "handle direction" 和 "opening direction"，这使得 6-DoF manipulation planning 成为可能。

### 4.3 Chain-of-Thought Spatial Reasoning

SOFAR 采用三步 CoT reasoning process：

**Step 1: Scene Analysis**
分析 scene 与 query $\mathcal{Q}$ 和 object nodes $\mathbf{V}$ 的关系。

**Step 2: Desired Pose Computation**
计算 target object 的 desired position 和 orientation。

**Step 3: Transformation Prediction**
预测每个 object 的 target position $\tilde{\mathbf{c}}_i$ 和 semantic orientation set $\tilde{S}_i$。

给定 initial state $\mathbf{c}_i$ 和 $S_i$，full 6-DoF transformation $\mathbf{P}_i$ 的计算：

**Translation**: $\mathbf{t}_i = \tilde{\mathbf{c}}_i - \mathbf{c}_i$

**Rotation**: 使用 **Kabsch-Umeyama algorithm** 从 $S_i$ 和 $\tilde{S}_i$ 估计 rotation matrix $\mathbf{R}_i$

**Kabsch-Umeyama algorithm intuition**: 给定两组 vectors (initial semantic orientations $S_i$ 和 target semantic orientations $\tilde{S}_i$)，找到 optimal rotation matrix $\mathbf{R}$ 使得：

$$\mathbf{R}^* = \arg\min_{\mathbf{R}} \sum_{k} \|\tilde{\mathbf{s}}_k - \mathbf{R} \mathbf{s}_k\|^2$$

算法步骤：
1. 计算两组 vectors 的 cross-covariance matrix $\mathbf{H} = \sum_k \tilde{\mathbf{s}}_k \mathbf{s}_k^T$
2. SVD decomposition: $\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$
3. Optimal rotation: $\mathbf{R} = \mathbf{V} \mathbf{U}^T$ (需要确保 proper rotation，即 $\det(\mathbf{R}) = 1$)

这种方法的优势在于它利用多个 semantic orientation vectors 来约束 rotation，比单 vector 对齐更 robust。例如，同时 align "handle direction" 和 "opening direction" 可以完全确定 mug 的 6-DoF pose（除了沿 axis 的 rotation）。

### 4.4 Low-Level Motion Execution

SOFAR 集成了 task-specific grasping 和 motion planning：

1. **Object/Part Segmentation**: Florence-2 + SAM
2. **Grasp Candidate Generation**: GSNet 生成 grasp candidates
3. **Optimal Grasp Selection**: 考虑 grasp quality 和 heuristics
4. **Trajectory Planning**: OMPL (Open Motion Planning Library) 生成 collision-free trajectory
5. **Joint Initialization**: 初始 joint positions 设为 midpoint 确保 smooth motion

参考链接:
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Florence-2 Paper](https://arxiv.org/abs/2311.06242)
- [Kabsch Algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)
- [OMPL Library](https://ompl.kavrakilab.org/)

## 5. Experimental Results Analysis

### 5.1 Open6DOR V2 Benchmark

**Perception Tasks (Isaac Sim)**:

| Method | Position Overall | Rotation Overall | 6-DoF Overall | Time Cost |
|--------|-----------------|-----------------|---------------|-----------|
| GPT-4V | 45.2 | 9.2 | - | - |
| Dream2Real | 15.9 | 31.3 | 13.5 | 358.3s |
| VoxPoser | 32.6 | - | - | - |
| Open6DOR-GPT | 74.9 | 41.1 | 35.6 | 126.3s |
| SOFAR-LLaVA | 78.7 | 48.6 | 40.3 | 9.6s |
| **SOFAR** | **93.0** | **57.0** | **48.7** | **8.5s** |

**Key observations**:
1. SOFAR 在 Rotation Track 上大幅领先 (57.0% vs 41.1%)，证明 semantic orientation 的有效性
2. Time cost 从 126.3s 降到 8.5s，efficiency 提升约 **15x**
3. SOFAR-LLaVA (fine-tuned version) 也表现强劲，显示 framework 的 flexibility

**Execution Tasks (Libero)**:

| Method | Position Overall | Rotation Overall | 6-DoF Overall |
|--------|-----------------|-----------------|---------------|
| Octo | 47.2 | 17.2 | 8.0 |
| OpenVLA | 47.6 | 17.6 | 8.2 |
| **SOFAR** | **67.0** | **25.7** | **18.4** |

SOFAR 在 execution tasks 上 zero-shot 达到 18.4% success rate，显著优于经过 robot data training 的 Octo 和 OpenVLA。

### 5.2 SIMPLER-Env Evaluation

**Google Robot Setup (Variant Aggregation)**:

| Policy | Pick Coke Can (Avg) | Move Near | Open/Close Drawer (Avg) | Average |
|--------|--------------------|-----------|------------------------|---------|
| RT-1-X | 0.490 | 0.323 | 0.294 | 0.397 |
| RT-2-X | 0.823 | 0.792 | 0.353 | 0.661 |
| Octo-Base | 0.006 | 0.031 | 0.011 | 0.012 |
| OpenVLA | 0.545 | 0.477 | 0.177 | 0.411 |
| **SOFAR** | **0.907** | **0.740** | **0.297** | **0.676** |

**Google Robot Setup (Visual Matching)**:

| Policy | Average |
|--------|---------|
| RT-1-X | 0.534 |
| RT-2-X | 0.606 |
| Octo-Base | 0.168 |
| OpenVLA | 0.277 |
| **SOFAR** | **0.749** |

**Critical insight**: SOFAR 作为 zero-shot method，在 Visual Matching setting 下达到 **74.9% success rate**，超过了使用 OXE data 训练的 RT-2-X (60.6%)。这证明 semantic orientation + VLM reasoning 的 decoupled approach 可以 outperform end-to-end VLA models，即使后者使用了大量 robot trajectory data。

### 5.3 6-DoF SpatialBench

| Method | Position (rel/abs) | Orientation (rel/abs) | Total |
|--------|--------------------|-----------------------|-------|
| GPT-4o | 49.4 / 28.4 | 44.2 / 25.8 | 36.2 |
| SpatialBot | 50.9 / 21.6 | 39.6 / 22.9 | 32.7 |
| RoboPoint | 43.8 / 30.8 | 33.8 / 25.8 | 33.5 |
| **SOFAR** | **59.6 / 33.8** | **54.6 / 31.3** | **43.9** |

SOFAR 在 orientation track 上达到 54.6% relative accuracy，比 GPT-4o 的 44.2% 高出 **10.4%**，证明 PointSO 提供的 orientation information 显著增强了 VLM 的 spatial reasoning capability。

### 5.4 Failure Case Distribution

| Failure Source | Percentage |
|---------------|------------|
| Grasping issues | 31% |
| Incorrect Semantic Orientation prediction | 23% |
| Object analysis and detection errors | 20% |
| Motion Planning issues | 16% |
| Task Planning (VLM) errors | 10% |

**Intuition**: 31% 的 failure 来自 grasping，这表明 low-level execution 仍是 bottleneck。Semantic Orientation prediction 的 23% error 主要发生在需要 <5° precision 的 tasks（如 upright/upside-down），这指向 future work 需要 higher precision orientation models。

参考链接:
- [Open6DOR Benchmark](https://arxiv.org/abs/2406.11118)
- [SIMPLER-Env](https://arxiv.org/abs/2410.24048)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Octo Model](https://arxiv.org/abs/2405.12213)

## 6. Ablation Studies Insights

### 6.1 Semantic Orientation Contribution

| CoT | Orient. | Rotation Overall | 6-DoF Overall |
|-----|---------|-----------------|---------------|
| ✗ | ✗ | 13.0 | 14.2 |
| ✓ | ✗ | 12.9 | 13.7 |
| ✗ | ✓ | 52.3 | 45.8 |
| ✓ | ✓ | **57.0** | **48.7** |

**Critical finding**: CoT alone 几乎没有提升 (13.0 → 12.9)，而 Semantic Orientation 带来巨大提升 (13.0 → 52.3)。这说明 **orientation information 是核心 enabler**，CoT 只是辅助 reasoning framework。这反驳了"可能是 CoT reasoning 带来提升"的 hypothesis。

### 6.2 Data Scaling Property

| Data Scale | 45° | 30° | 15° | 5° | Average |
|-----------|-----|-----|-----|-----|---------|
| 15K | 57.03 | 46.09 | 39.84 | 27.34 | 42.58 |
| 35K | 61.72 | 53.13 | 43.75 | 30.47 | 47.27 |
| 150K | 76.56 | 72.66 | 66.41 | 56.25 | 67.97 |
| 350K | 79.69 | 77.34 | 70.31 | 62.50 | 72.46 |

**Scaling law insight**: 从 15K 到 350K (23x increase)，average accuracy 从 42.58% 提升到 72.46% (+30%)。这种 consistent improvement 表明 PointSO 具有 foundation model 的 scaling property。可以预见使用 Objaverse-XL (10M+ objects) 会进一步提升 performance。

### 6.3 Detection Module Comparison

| Method | Position Overall | Rotation Overall | 6-DoF Overall | Time |
|--------|-----------------|-----------------|---------------|------|
| YOLO-World | 53.3 | 44.9 | 27.8 | 7.4s |
| Grounding DINO | 86.7 | 55.5 | 44.6 | 9.2s |
| **Florence-2** | **93.0** | **57.0** | **48.7** | 8.5s |

Florence-2 虽然在 COCO benchmark 上并非 SOTA，但在 in-the-wild detection tasks 上表现出色，generalization 优于 Grounding DINO 和 YOLO-World。

## 7. Comparison with Concurrent Works

### 7.1 vs. ReKep

ReKep 使用 spatial keypoint constraints，存在三个主要 issues：

1. **Overly customized prompt engineering**: 每个任务需要 manually designed complex system prompts，不是真正的 zero-shot
2. **Keypoint constraints 无法 capture full 6-DoF**: 例如 "pouring water" task，仅 align spout 到 cup 可能导致 kettle overturning
3. **Requires all keypoints in first frame**: 从 mask extraction 到 clustering 引入大量 hyperparameters

SOFAR 的优势在于 genuine zero-shot transfer 和完整的 6-DoF representation。

### 7.2 vs. Orient Anything

Orient Anything 只学习 basic directions ("front", "top")，而 SOFAR 的 Semantic Orientation 是 **language-conditioned**，可以处理任意 open-vocabulary queries 如 "plug-in", "cutting", "handle"。这种 language grounding 使得 SOFAR 能直接 integrate 到 VLM reasoning pipeline。

### 7.3 vs. End-to-End VLA Models

SOFAR 作为 decoupled approach，相比 RT-2-X, OpenVLA 等 end-to-end VLA models 的优势：

1. **Zero-shot generalization**: 不需要 robot trajectory data training
2. **Interpretability**: Scene graph 和 CoT reasoning 提供可解释的 decision process
3. **Embodiment agnostic**: 同一 system 可用于 gripper, suction cup, dexterous hand
4. **Data efficiency**: 只需要 3D object data，不需要 expensive robot data

参考链接:
- [ReKep Paper](https://arxiv.org/abs/2409.01652)
- [Orient Anything](https://arxiv.org/abs/2412.18605)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)

## 8. Technical Details: Training Recipes

### 8.1 PointSO Training Configuration

| Config | Small | Base | Large |
|--------|-------|------|-------|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 5e-5 | 5e-5 | 2e-5 |
| Weight decay | 5e-2 | 5e-2 | 5e-2 |
| LR scheduler | cosine | cosine | cosine |
| Training epochs | 300 | 300 | 300 |
| Warmup epochs | 10 | 10 | 10 |
| Batch size | 256 | 256 | 256 |
| Drop path rate | 0.2 | 0.2 | 0.2 |
| #Points | 10000 | 10000 | 10000 |
| #Point patches | 512 | 512 | 512 |
| Point patch size | 32 | 32 | 32 |
| Augmentation | Rot&Part&Noise | Rot&Part&Noise | Rot&Part&Noise |
| GPU | 8×H800 | 8×H800 | 8×H800 |

**Data augmentation strategy**: "Rot&Part&Noise" 包括：
- **Rotation**: Random SO(3) rotation 增强 rotation invariance
- **Part**: Partial point cloud simulation，增强 robustness to incomplete observations
- **Noise**: Gaussian noise 模拟 real-world sensor noise

### 8.2 SOFAR-LLaVA Fine-tuning

SOFAR-LLaVA 是通过 visual instruction tuning fine-tuned 的 VLM variant。Pipeline:
1. JSON-formatted 6-DoF scene graph 通过 text tokenizer 处理
2. Image 通过 SoM prompting refined
3. 输入 LLM (e.g., LLaMA) 进行 supervised fine-tuning
4. Training data: ~3000 6-DoF object manipulation instructions，从 Objaverse 检索并 manual annotated

## 9. Real-World Experiment Details

### 9.1 Task Design

60 real-world tasks 分为三个 tracks：

**Position Track (25 tasks)**:
- Simple: front/back/left/right spatial relationships
- Hard: between, center, customized positions

**Orientation Track (20 tasks)**:
- Simple: part-level orientation (handle, cap, tip)
- Hard: upright/upside-down (需要 <5° precision)

**Comprehensive & 6-DoF Track (15 tasks)**:
- Complex instruction understanding
- Simultaneous position + orientation control

### 9.2 Embodiment Generality

SOFAR 在三种不同 end-effectors 上验证：
- **Franka Panda with gripper**: Standard 6-DoF rearrangement
- **UR5e with LeapHand**: Dexterous hand manipulation
- **Flexiv with suction tool**: Articulated object manipulation

这种 embodiment agnostic property 来源于 SOFAR 只规划 target pose，不学习 embodiment-specific trajectories。

### 9.3 Navigation Extension

SOFAR 还扩展到 orientation-aware navigation:
- Robot dog (Unitree GO2) 需要从 functional side 接近 object
- 例如：从 front 接近 microwave 以打开 door
- Semantic orientation 提供 facing direction constraint，enhancing navigation precision

## 10. Limitations & Future Directions

### 10.1 Current Limitations

1. **Decoupled system fragility**: Sub-module error (grasping, perception) 会导致 execution failure
2. **Precision bottleneck**: 23% failure 来自 orientation prediction，特别是需要 <5° precision 的 tasks
3. **Open-vocabulary detection instability**: 20% failure 来自 Florence-2/Grounding DINO 在 OOD objects 上的误检测
4. **Motion planning limitations**: 未考虑 robotic arm workspace 和 collision，导致 occasional deadlocks

### 10.2 Future Work Directions

1. **Data expansion**: 使用 Objaverse-XL (10M+ objects) 进一步 scale OrienText300K
2. **Self-supervised pretraining**: 结合 MAE, contrastive learning 提升 PointSO representation
3. **End-to-end integration**: 探索 decoupled methods 与 end-to-end VLA 的结合
4. **Application expansion**: Navigation, mobile manipulation, lifelong learning, spatio-temporal reasoning, humanoid robots, human-robot interaction

## 11. Broader Impact & Connections

### 11.1 Relation to Affordance Learning

Semantic Orientation 与 affordance 有 conceptual connection 但 beyond:
- **Affordance**: 表示 potential actions/interactions
- **Semantic Orientation**: 包含 intra-object part-level spatial understanding

两者都 present potential actions，但 Semantic Orientation 更 generalizable，可以从 Internet 3D data auto-label，达到 higher scalability。

### 11.2 Connection to 6-DoF Pose Estimation

Semantic Orientation + 3-DoF translation = same DoF completeness as 6-DoF pose。关键区别在于：
- Traditional 6-DoF: reference-frame-dependent, template-based
- Semantic Orientation: language-grounded, template-free, open-world

### 11.3 Impact on Robotics Community

SOFAR 代表了 robotics 从 "end-to-end learning" 到 "reasoning + execution" paradigm shift 的一个重要 milestone。通过将 semantic understanding 与 geometric precision 结合，SOFAR 展示了如何利用 Internet-scale 3D data 和 foundation models 来 achieve general-purpose robot manipulation，而不依赖 expensive robot trajectory data。

这种 approach 对 robotics democratization 有重要意义：researchers 可以使用 cheap 3D data 和 pre-trained foundation models 来 build capable manipulation systems，而不需要 expensive robot data collection infrastructure。

参考链接:
- [NeurIPS 2025 Paper Checklist](https://neurips.cc/Conferences/2025/PaperChecklist)
- [Galbot Company](https://www.galbot.com/)
- [Tsinghua University Robotics](https://www.cs.tsinghua.edu.cn/)

---

**Final intuition summary**: SOFAR 的核心贡献在于发现了 object orientation 可以用 natural language 来 ground，并通过 large-scale 3D data learning 实现 open-world generalization。这种 language-grounded orientation representation 桥接了 spatial reasoning (VLM 擅长) 和 object manipulation (需要 geometric precision) 之间的 gap，为 build general-purpose robots 提供了新的 paradigm。PointSO 作为 orientation foundation model，类似于 CLIP 之于 vision-language alignment，为 robotics community 提供了一个可复用的 semantic orientation predictor。
