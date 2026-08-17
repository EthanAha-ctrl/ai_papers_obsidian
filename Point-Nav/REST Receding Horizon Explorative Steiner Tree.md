---
source_pdf: REST Receding Horizon Explorative Steiner Tree.pdf
paper_sha256: 4f97a0094d3f60f54570308a0224237993ab2aafa71d08efa51eaf11de8e2017
processed_at: '2026-08-11T23:00:19-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# REST 用人话说

## 一句话说清楚

之前 ZSON 的 agent 选下一个去哪儿，就像用 google maps 只看终点 POI 评分——"这个点 4.5 星，那个点 3.8 星，去 4.5 的"。问题是：去某个终点 *路上* 你可能路过一堆 doorway，真正的信息藏在路上而不是终点。而且一堆 waypoint 各自独立打分，看不出"这 4 个点其实都是走走廊然后分叉"的结构。

REST 说：别看点了，看 *路径树*。把所有可能的 path 组织成一棵 Steiner tree，共享 trunk 的合并起来，LLM 在 fork 处做"走哪个 branch"的决策，而不是在每个 leaf 上独立打分。

---

## 想象你找厕所

你在陌生办公楼找 toilet。

**老方法**（VLFM 这种）：地图上每个 frontier 点都用 CLIP 算 `<toilet, 这个点的图像> cosine similarity`，谁高去谁。走廊尽头那张图看起来就是走廊，相似度 0.67，旁边 dining room 图里有 chair，`<toilet, dining room> ≈ 0.73`，于是你去 dining room。但你其实应该走走廊——路上会路过 4 个门，其中一个就是厕所。

**REST 方法**：把"走廊这条路"作为一个 branch 整体描述给 LLM——"Option B: 一条长走廊，沿途能看到 3 个未探索 doorway，末端有 3.2m³ unknown region"。LLM 一想，走廊通向多房间，厕所概率高，选 B。走 trunk 段不决策，到 fork 才再问 LLM。

这就是从"点估计"到"方向估计"的升级。

---

## 系统怎么搭的

三个组件串起来：

### 1. Belief — 感知世界

**几何**：UFOMap 做 3D octree，每个 voxel 标 unknown/free/occupied。关键是保留 unknown 状态，后面要数"能看到多少未知"。

**语义**：三段式 pipeline——
- Qwen3-VL 2B 先给图打 tag："这图里有 sofa, plant, door"
- YOLO-World 接 tag 做 open-vocab detection 出 bbox
- EdgeTAM (SAM2 系) 把 bbox 精修成 mask
- 投回 3D voxel 做 confidence-weighted majority voting

跨帧有个很实际的问题：同一个杯子这帧叫 "blue cup" 下帧叫 "teacup"，怎么知道是同一个？REST 用 spatial overlap (IoS > 0.3) AND semantic cosine sim > 0.8 双条件合并。

**路图**：RT-RRT\* 持续维护一棵 agent-rooted 的最短路径树。agent 移动时重 root，不重建。还有一个很实际的检查：在 agent 脚下挂 thin AABB 查 occupied 比例，拒绝悬空 waypoint（防止从楼梯掉下去）。

### 2. Option — 这篇 paper 的核心

RT-RRT\* 有上千条 path，LLM 看不过来。两步压缩：

**(a) 压缩候选 viewpoint**

Poisson-disc sampling 半径 1.25m 内只保留一个候选，避开昂贵 ray-casting。然后对每个候选 $\theta$ 算 information gain：

$$
r(\theta \mid \mathcal{M}) = \left| \left\{ v \in \mathcal{M} \cap \mathcal{F}(\theta) : \mathrm{I}_{\mathrm{visible}}(v, \theta) \cdot \mathrm{I}_{\mathrm{unknown}}(v) = 1 \right\} \right|
$$

意思就是：站在这个位姿，视锥 $\mathcal{F}(\theta)$ 内能看到多少 unknown voxel $v$（visible 表示 ray 没被遮挡）。阈值 $\tau_{IG} = 10$ 才保留。

直觉：这就是 next-best-view——"从这里能看到多少没探索过的空间"。

**(b) Steiner Tree 合并共享段**

15 个 viewpoint 各自有 RRT\* 最短 path，但很多 path 前半段重合。Steiner Tree 优化把这些共享段合并，引入 auxiliary Steiner point 把"三脚架"压成"Y 字形"。Figure 3 里 85m → 47m，更重要的是暴露了 4 个 natural fork point——corridor 分叉、room entrance。

算法很朴素：warm start 用 RRT\* subtree，然后反复 (1) Weiszfeld 算 branching node 的 geometric median 当 Steiner point，(2) Kruskal 在所有 node 上算 MST，(3) prune 非 terminal leaf，直到 cost 不降。

后台跑，每次迭代都产出 valid tree，agent 用当前 best 不阻塞。

### 3. Policy — LLM 决策

**Tree → 语言**：每条 edge 模拟虚拟 camera 走一遍，在 informative viewpoint 处切 panoramic FOV 扫 360°，产出结构化 annotation：
- Known: "pet carrier @ 0.5m, sofa @ 2.0m"
- Unknown: "3.2m³ unknown region near sofa"

top-down 装配成 tree，summarizer LLM 把每个 root-level subtree 转成一句 caption：
- `{chairs, dining table}` → "a dining area"  
- `{coffee table, sofa, pet carrier, cat litter bin}` → "a cluttered cozy living room with a pet-care area"

**LLM 选 branch**：CoT 让 LLM 先想每个 option 的可能性再打分 0-100。比如找 bed："Option A 是 living room 很少有 bed；Option B 通往 hallway 接近大块 unexplored，可能通向 bedroom"。

**Receding horizon**：选了某个 subtree 后沿 trunk 直走，不重复调 LLM。只在 (1) 走到 branching node (2) committed subtree 内 topology 变了，才重新调 LLM。

**Fallback**：LLM 选 "none of above"（语义信号不足）就退化到纯几何，去最近 informative viewpoint，到了再重跑。

---

## 实验结果

| Benchmark | REST | 最强对手 |
|---|---|---|
| Gibson | SR 85.1 / SPL 53.5 | GAMap 85.7 / 55.5（接近饱和）|
| HM3D | SR 57.3 / **SPL 33.4** | ApexNAV 59.6 / 33.0 |
| HSSD | **SR 56.7 / SPL 29.1** | ImagineNav 51.0 / 24.9 |

HM3D 的 SR 略输 ApexNAV 的原因有意思：HM3D 是真实扫描，镜面/玻璃在 UFOMap 里产生 phantom opening，information gain 被虚高，agent 被引向死胡同耗光 500-step budget。2D frontier 方法对这种 scan artifact 免疫。但完成的 episode 路径效率最高（SPL #1）。

HSSD 完胜——scene 多样且语义稀疏，point-wise ranking 容易被 nearest irrelevant 物体 trap 进 greedy local minima，tree reasoning 在 sparse scene 优势放大。

**Ablation 很关键**：
- 去掉 LLM: SR 56.7 → 29.1（砍半，LLM 是 backbone）
- 去掉 Steiner tree: SR 56.7 → 53.1, SPL 29.1 → 25.3。原因不是单纯性能下降，是 compounding degradation——冗余 edge → caption 冗长 → LLM 失焦。**option space 的 compactness 直接影响 LLM 推理质量**。

---

## 给你 Karpathy 的 intuition

这篇本质上在做一件事：把 navigation 从 "endpoint scoring" 重新 framing 成 "branch reasoning"。

LLM 在这里的角色非常 limited——它没在做规划，在做 **ranking over 4 abstracted options**。真正规划在前端：RT-RRT\* + Steiner tree + information gain。LLM 是个 **semantic prior injector**，把 commonsense "走廊通向多房间所以可能藏厕所" 注入到 geometric planner 输出上。

这种 hybrid 比 pure VLM nav (PanoNav, ImagineNav) 数据效率高，比纯 frontier 方法 (VLFM, ApexNAV) 少短视。本质上和 Eureka 用 LLM 做 reward shaper 而不是 policy 是一个套路——把 LLM 放在它擅长的位置（语义抽象 + commonsense），而不是让它做它不擅长的（几何规划、连续控制）。

Steiner Tree 在这里其实是 **presentation layer**：LLM 看到的不是 metric cost，是 textual caption。Steiner 优化的真正价值是暴露 natural decision junction（corridor fork、room entrance），把指数级 path 组合压缩成 $O(\log N)$ 层级 branch。从 information-theoretic 角度，它是个 information bottleneck，压缩比 ~3.75×（15 viewpoints → 4 branches）。

最深的 insight：**option space 的 representation 决定了 policy 的推理效率**。孤立 waypoint 让 LLM 在 leaf 层做 $O(N)$ 次独立判断；Steiner tree 让 LLM 在 branch 层做 $O(\log N)$ 次层级判断。这是把 decision tree induction 的思想反向用在 navigation planning 上——先粗后细，先 direction 后 destination，跟人类导航的直觉完全一致。

参考链接：
- VLFM: https://arxiv.org/abs/2312.03275
- SG-Nav: https://arxiv.org/abs/2410.20619
- VoroNav: https://arxiv.org/abs/2406.04192
- UFOMap: https://arxiv.org/abs/2003.05551
- RT-RRT*: https://dl.acm.org/doi/10.1145/2822013.2822016

---

# REST: Receding Horizon Explorative Steiner Tree — 深度技术解析

## 1. 论文核心 Thesis — 一句话直觉

ZSON 的 hierarchical agent 把 option space 设计成"孤立 waypoint 集合 + 独立打分"的范式，这丢失了两件事：**(a)** 路径上的累积信息增益，**(b)** 候选之间的结构关系。REST 的核心 insight 是把 option space 升级成 **path-grounded 的 Steiner Tree**——让 LLM 在"branch"层级而非"leaf"层级做 coarse-to-fine 推理，把指数级路径组合压缩成层级化决策树。

让我先 build your intuition：考虑 Figure 1 里找 toilet 的例子。走廊尽头的 waypoint 在 cosine similarity 上得分很低（`<toilet, corridor> ≈ 0.67`），但走过去的过程会路过 4 个 doorway，其中之一大概率藏有 toilet。waypoint 范式只能看到"终点 7,8,9,10 各自得分低"，tree 范式把它们压缩成"corridor 子树"，让 LLM 推理"走廊这个 *direction* 值得探索"。这是从**点估计**到**方向估计**的本质转变，类似 policy gradient 从 score function 到 trajectory distribution 的过渡。

---

## 2. 三层 Belief-Option-Policy 拆解

### 2.1 Belief Update — 在线几何/语义/路图三件套

**Geometric mapping: UFOMap**
- UFOMap 是一种基于 octree 的 probabilistic 3D volumetric mapping framework，每个 voxel 分类为 `unknown / free / occupied`。
- 关键设计：**显式保留 unknown**（不是 binary occupancy），这使得 information gain 的 ray-casting 计算成为可能（后面公式会用到）。
- voxel size = 5cm，这个粒度比典型 2D grid map (10–25cm) 细，因为要做 3D swept volume collision checking。
- 论文链接：https://github.com/UnknownFreeOccupied/ufomap

**Semantic mapping: cascaded recognize-detect-segment**

这是 engineering 上很巧妙的设计，三个 foundation model 串联互补：

| Stage | Model | Role | Why |
|---|---|---|---|
| Recognize | Qwen3-VL 2B (8-bit quantized) | 给 image 打 salient entity tags | VLM 有 zero-shot understanding，但缺 grounding 精度 |
| Detect | YOLO-World | open-vocab 2D bbox | 需要 text prompt 才能 ground |
| Segment | EdgeTAM (SAM2 系) | refine bbox → mask | dense grounding |

直觉：VLM 像"知道房间里有什么"的 captioner，detector 像"框出来"的 grounder，segmentor 像"精确分割"的精细器。三者串联克服了 VLM 不会 localize、detector 不会 caption、segmentor 需要 prompt 的各自短板。

**3D semantic fusion 投票机制**

```
for each frame:
    retrieve visible occupied voxels in camera frustum
    project to image plane
    for each pixel in instance mask:
        cast weighted vote for that label
voxel.label = argmax(per-label vote histogram)
```

权重是 detection confidence。多帧 majority voting 平滑掉单帧噪声。

**Entity tracking + cross-label merging**——这里有个非常实际的细节：

- 跨帧 IoU 匹配 bounding box（阈值 τ_IoU = 0.5）
- Open-vocab tagging 会产生 label 不一致：同一个杯子可能被打成 "blue cup", "ceramic cup", "teacup"。REST 用 **spatial-semantic joint clustering** 合并：
  - 空间重叠：Intersection over Smaller (IoS) > τ_IoS = 0.3
  - 语义相关：MobileCLIP2 embedding cosine similarity > τ_sim = 0.8
  - 两个条件同时满足才合并

直觉：纯空间合并会把相邻但语义无关的物体揉一起；纯语义合并会把远距离同类物体当成一个。joint 条件确保"这就是同一个东西"。

**Road mapping: RT-RRT\***

- Real-Time RRT*（Naderi et al. 2015）：增量维护 agent-rooted 的 asymptotic optimal path tree
- Hybrid sampling (Schmid et al. 2020)：node 密度低于阈值时局部采样保连通性，否则全局采样扩覆盖
- **Traversability check**：在 agent base 下方挂一个 thin AABB，查询 occupied voxels 占比；拒绝悬空 waypoint（防止机器人掉下楼梯）
- **Edge feasibility**：cylindrical bounding volume + PCA 估计 OBB 表示 swept volume，与 occupied/unknown voxel 无交集才通过
- 关键：**tree 跨 decision epoch 持续存在**，agent 移动时重新 root 而不重建

这一步输出一个 roadmap，本质是"agent 能去哪里 + 怎么去最短"的数据库。

---

### 2.2 Option Space — 本文的核心 contribution

#### 2.2.1 Informative Viewpoint Sampler

RT-RRT* tree 有上千条 path，远超 LLM reasoning 容量。REST 用两阶段 filter 压缩：

**(a) Spatial thinning: Poisson-disc sampling**
- 接受 candidate ⟺ 半径 r = 1.25m 内无现存 viewpoint
- 产出 well-distributed 低冗余集合
- 廉价预筛，避开昂贵 ray-casting

**(b) Information-gain gating**

核心公式：

$$
r(\theta \mid \mathcal{M}) = \left| \left\{ v \in \mathcal{M} \cap \mathcal{F}(\theta) : \mathrm{I}_{\mathrm{visible}}(v, \theta) \cdot \mathrm{I}_{\mathrm{unknown}}(v) = 1 \right\} \right|
$$

变量解释：
- $\theta \in \mathrm{SE}(3)$：candidate viewpoint 的 6-DoF 位姿
- $\mathcal{M}$：UFOMap（当前 3D occupancy octree）
- $\mathcal{F}(\theta)$：camera 在位姿 $\theta$ 的 view frustum（视锥体）
- $v$：octree 中的一个 voxel
- $\mathrm{I}_{\mathrm{visible}}(v, \theta) = 1$：当且仅当从 $\theta$ 到 $v$ 中心的 ray 未被遮挡（line-of-sight）
- $\mathrm{I}_{\mathrm{unknown}}(v) = 1$：当且仅当 $v$ 被分类为 unknown
- $|\cdot|$：set cardinality

直觉：这就是 next-best-view literature 里的体积信息增益——"从这个位姿能看到多少未知空间"。阈值 $\tau_{IG} = 10$ 个 voxel 才被保留。和 frontier 的区别：frontier 是 grid 边界，REST 的 informative viewpoint 是 SE(3) 位姿，附带方向感。

#### 2.2.2 Euclidean Steiner Tree Optimizer — 这是 paper 名字的来源

**问题形式化**：

给定 root $r \in \mathcal{F}$（agent pose）和 terminals $V_T = \{v_1, \ldots, v_N\} \subset \mathcal{F}$（informative viewpoints），求解：

$$
\min_{T=(V,E)} C(T) = \sum_{e \in E} \|e\|_2
$$

约束：$\{r\} \cup V_T \subseteq V \subseteq \mathcal{F}$，且 $T$ 是嵌入 free space $\mathcal{F} \subseteq \mathbb{R}^2$ 的 tree。

这是 **Obstacle-Avoiding Euclidean Steiner Minimum Tree (OAESMT)** 问题，NP-hard。允许引入 auxiliary Steiner points 来合并共享 segment，从而降低总 cost。

**REST 的近似算法**（Algorithm 1）——非常聪明：

1. **Warm start**：从 RT-RRT* subtree（连接 root 到所有 terminals）开始——这本身就是一个 feasible Steiner tree，只是不最优
2. **迭代局部改进**：
   - Post-order 遍历每个 branching node $v$ 及其 children
   - 用 **Weiszfeld's algorithm** 计算 geometric median $s$（到 $v \cup \mathrm{Children}(v)$ 距离之和最小的点）
   - 把 $s$ snap 到最近的 RRT* node $s^*$（保持 collision-free）
   - 把 collision-free 的 Steiner points 加入 node set $V$
   - 用 **Kruskal's algorithm** 在 $V$ 上计算 MST（边限制为 line-of-sight segments）
   - **Prune** 非 terminal leaves
   - 如果 $C(T') < C^*$，更新 best
3. **收敛**：直到 cost 不再下降

Figure 3 给了一个直观对比：
- 输入 RT-RRT* subtree：85m 总长
- 输出 Steiner tree：47m 总长
- 减少 44.7%，并且暴露了 trunk-to-branch-to-leaf 的拓扑结构

**为什么这是关键 contribution**：

第一直觉——节省距离。但更深层的作用是 **暴露 decision junction**。看 Figure 3 右图，Steiner tree 把 15 个 informative viewpoint 整合成 4 个 root-level subtree，每个对应一个 corridor fork 或 room entrance。LLM 不再面对 15 个独立候选，而是 4 个语义化 branch——这是把"哪 15 个 waypoint 哪个最好"压缩成"哪 4 个方向最好"，决策复杂度从 $O(N)$ 降到 $O(\log N)$ 的层级。

Algorithm 1 还有个工程上的细节：每次迭代都产出 valid tree，agent 可以 acting on current best 同时后台 refine，避免阻塞。这就是 "Receding Horizon" 的来源——类似 MPC，每次决策用当前最优解，然后持续优化下一刻。

---

### 2.3 Hierarchical Policy — Tree → Language → Decision

#### 2.3.1 Tree Narration: 三阶段 textualization

**(a) Per-Edge Annotation（并行）**：

对 tree 每条 edge 模拟一台 forward-facing virtual camera 从 parent 走到 child，查询 geometric + semantic map：
- 在 informative viewpoint 处切换到 panoramic FOV 做 360° 扫描
- 产出两类结构化描述：
  - **Known**：entity ID + label + caption + sighting distance（如 `"pet carrier @ 0.5m, sofa @ 2.0m"`）
  - **Unknown**：DBSCAN 聚类 unknown voxels → cluster volume + centroid 到 nearest known entity 的距离（如 `"3.2m³ unknown region near the sofa"`）

**(b) Tree-Level Assembly**：

top-down traversal 从 root 到 leaves，利用 entity unique ID 跨 edge 跟踪 sighting distance 趋势。例如 entity 的 sighting distance 沿 root-to-leaf 递减，则标记该 subtree "approaching" 该 entity。

**(c) Subtree Captioning**：

summarizer LLM 把结构化 annotation 转成自然语言 option，每个 root-level subtree 一个 caption。鼓励 region-level / room-level 抽象：
- `{chairs, dining table}` → `"a dining area"`
- `{coffee table, sofa, pet carrier, cat litter bin, stuffed animals}` → `"a cluttered cozy living room with a pet-care area"`

**关键设计选择**：summarizer LLM ≠ decision-making LLM。前者负责语义压缩，后者负责选择。这避免了 "既要描述又要决策" 的角色冲突。

#### 2.3.2 Receding-Horizon LLM Planning

系统被解耦成两层：

**Fast reactive layer**：sensor rate 更新 RT-RRT* + Steiner tree
**Event-triggered deliberative layer**：只在 decision-relevant moments 调用 LLM（秒级延迟）

**Commitment model**：选中 path 后，agent 沿非 branch node 路径直走，不重新调用 LLM。执行粒度由 topology 决定，不是固定 metric segment——避免走廊里重复无意义 query。

**Re-invocation triggers**：
1. 当前位置是 live tree 的 branching node（root 有多个 children）
2. committed subtree 内发生 decision-relevant structural change（committed node 被移除或 Steiner 优化重写了 topology）
3. distant map 变化 deferred 到下次 branching node

**Chain-of-Thought Scoring**：

LLM 输入：target category + 编号 subtree captions + 显式 "none of the above" 选项

CoT 流程：
1. 分析每个 caption，基于 semantic co-occurrence 和 spatial layout priors 推理目标可能性
2. 给每个 option 打 0–100 分数
3. 最高分确定 next subgoal

论文给的例子：找 bed 时——"Option A 是 living room，bed 出现概率低；Option B 通往 hallway 接近一大块 unexplored region，可能通向 bedroom"——典型的 commonsense reasoning。

**Geometric fallback**：选 "none" 时（语义信号不足，常见于探索早期或 monolithic scene），fallback 到纯几何策略——导航到最近 informative viewpoint，重跑 pipeline。

---

## 3. 实验结果 — 三个 benchmark

### 3.1 主结果 (Table I)

| Dataset | REST SR / SPL | Best competitor SR / SPL | REST 排名 |
|---|---|---|---|
| Gibson | 85.1 / 53.5 | GAMap 85.7 / 55.5; VLFM 84.0 / 52.2 | Top-3 SR, Top-3 SPL |
| HM3D | 57.3 / **33.4** | ApexNAV 59.6 / 33.0 | #2 SR, #1 SPL |
| HSSD | **56.7 / 29.1** | ImagineNav 51.0 / 24.9; VoroNav 41.0 / 23.2 | **#1 SR, #1 SPL** |

**关键观察**：
- Gibson 已接近饱和，区分度低
- HM3D 上 SR 略低于 ApexNAV（59.6 vs 57.3），但 SPL 最高（33.4）。论文诊断原因：HM3D 真实扫描的 mirror/glass 在 UFOMap 里产生 phantom openings，inflate information gain，把 agent 引向死胡同耗尽 500-step budget。2D frontier method (ApexNAV) 对此免疫
- HSSD 上 REST 完胜——scene 多样、semantic 稀疏，点式 waypoint ranking 容易被 nearest irrelevant 物体 trap 进 greedy local minimum，tree reasoning 在 sparse scene 优势放大

### 3.2 vs VoroNav 隔离分析

VoroNav 是最接近的 graph-based baseline。REST 在 HM3D 上 SR +15.3, SPL +7.4；HSSD 上 SR +15.7, SPL +5.9。两者都基于 navigational graph，所以这个 gap **隔离出** REST 的两个设计选择的增量价值：
- epistemic（information-gain driven）vs rigid geometric skeleton
- path-grounded tree vs clearance-maximizing Voronoi

### 3.3 Ablation (Table II, on HSSD)

| Variant | SR | SPL |
|---|---|---|
| REST Full | 56.7 | 29.1 |
| w/o LLM reasoning | 29.1 | 18.7 |
| w/o Steiner tree | 53.1 | 25.3 |

直觉：
- **去掉 LLM**：SR 砍半（56.7 → 29.1），证明 LLM commonsense 是 backbone
- **去掉 Steiner tree**：SR 掉 3.6, SPL 掉 3.8。原因不是单一性能，而是 **compounding degradation across pipeline**——RRT* subtree 早分支、trunk 短、冗余 edge 多 → tree narrator 工作量大、wall-clock latency 高 → caption 冗长重复 token → SVLM summarizer 性能降 → decision LLM 看到的 option 描述冗长模糊，关键 topological junction 被 bury

这个 ablation 的深层 intuition：**option space 的 compactness 直接影响 LLM 推理质量**，冗长 input 让 LLM 失焦，这和 prompt engineering 里 "less is more" 的现象一致。

---

## 4. 我的批判性思考 — 给 Andrej 的 intuition 角度

### 4.1 Steiner Tree 在这里其实是 "presentation layer"

仔细看：Steiner Tree 优化器输出 cost-47m 的 tree，但 LLM 看到的不是 metric cost，而是 textual caption。Steiner 优化的真正价值是 **拓扑结构**——暴露 corridor fork 和 room entrance 这些 natural decision junction。这等价于在 RANSAC 出来一堆 line 之后再做一次"结构抽象"。从 information-theoretic 角度看，Steiner tree 是一个 **information bottleneck**：高维 path space → 低维 branch space，压缩比 ~3.75×（15 → 4）。

### 4.2 RT-RRT\* 的隐藏价值

为什么用 RT-RRT* 而不是直接做 frontier-based path planner？因为 RRT* 的 rewiring 给你 **asymptotic shortest path**，所以从 agent root 到每个 viewpoint 的距离估计是 tight 的。Steiner 优化需要 warm start——RT-RRT* 提供了高质量的 warm start。如果用 A* + frontier，每次重规划都要从零计算，丢失了历史信息。

### 4.3 "Receding Horizon" 和 MPC 的对应

MPC：每个 timestep 解一个 finite-horizon optimal control 问题，执行第一个 action，滚动 horizon。REST 对应：
- Horizon = 当前 decision tree 的所有 leaves
- Control = 选择哪个 subtree
- Receding = 走到下一个 branching node 后 tree 重新 grow
- 优化目标 = LLM CoT scoring + Steiner cost minimization 的混合

### 4.4 LLM 在 ZSON 中的真正角色

读完这篇 paper 你会发现，LLM 在 navigation 里其实做的事情非常 limited：**给 4 个 abstracted option 打分**。它没在做规划，在做 **ranking**。真正的"规划"在前端的 geometric + informative path planning。LLM 是一个 **semantic prior injector**——把 VLM/commonsense 的高层知识注入到 geometric planner 的输出上。这种 framing 让我想到 NVIDIA 的 Eureka——LLM 作为 reward shaper，不是 policy 本身。

### 4.5 Limitation 没明说但明显的：

1. **HM3D 的 phantom opening 问题**：UFOMap 对 mirror/glass 敏感，导致 SR 下降。这是 3D volumetric mapping 在真实扫描上的老问题。一个可能的 fix 是引入 semantic-aware occupancy（mirror voxel 标记为 reflective 而不是 free）
2. **Steiner tree 优化是 NP-hard 的近似**：用 Weiszfeld + Kruskal 收敛到 local minima，没有保证。但因为是 receding horizon，每次都重新优化，所以 local minima 问题被 mitigate
3. **LLM 调用频率**：虽然 event-triggered，但每次 CoT scoring 仍要秒级。在 500-step budget 下，如果 branching 频繁，可能吃掉可观比例。论文没报告平均 LLM 调用次数
4. **2B Qwen3-VL 8-bit**：summarizer + decision maker 共用一个 2B model，规模偏小。在 complex scene 里 commonsense 可能不够 rich。如果要 scale up 到 GPT-4V / Claude 级，latency 会成更大瓶颈

### 4.6 跟你 [PointNet] [Neural MMO] 等 prior work 的呼应

你写过 "software 2.0"——dataset 替代 hand-crafted rules。REST 是 **software 3.0 的 navigation instantiation**：foundation model 替代 hand-crafted frontier scoring。但 REST 的设计哲学很有趣——它没有端到端用 LLM 做 navigation，而是把 LLM 当作 hierarchy 中一个 specific node 的 reasoner。这让我想到你在 1-hour LLM intro 里讲过的 "LLM as OS kernel call"——REST 就是把 LLM 作为 navigation stack 里一个 specific subroutine，配合传统 SLAM + RRT* + Steiner 优化。这种 hybrid 比 pure VLM navigation (PanoNav, ImagineNav) 在数据效率上明显更优。

### 4.7 Steiner Tree 的几何直觉

为什么 Steiner Tree 比 MST 在这里好？Steiner points 允许 tree 在原 terminal 集合之外添加 junction node，从而把"三脚架"形状的 path 合并成"Y 字形"。在 Euclidean plane 里，三个点构成的 Steiner Tree 比 MST 平均节省 ~13%（Graham-Pollak 上界）。在 navigation context，这个节省等于 agent 走过的多余距离，更重要的是 trunk 段成为"共享 commitment"，agent 走 trunk 时不需要再决策，到了 fork 才决策。

### 4.8 跟 Frontier-based exploration 的本质区别

Frontier = "已知 vs 未知的边界 voxel"。它本质是 2D grid 上的 boundary detection，与 agent 当前 pose 无强耦合。
REST informative viewpoint = "能最大化 unknown voxel 可见性的 SE(3) 位姿"。它本质是 next-best-view (NBV) 的 3D 形式，附带 agent 视角。

Frontier 范式 = "去边界"（explore-then-plan）。
REST 范式 = "去看"（informative path planning integrated with exploration）。

NBV literature (e.g., Next-Best-View planning in robotics) 早就有这套思路，REST 把它和 LLM reasoning 桥接起来，并且用 Steiner tree 把 exponential path space 压成 tractable hierarchy。

---

## 5. 参考 Web Links

**核心方法论文**：
- VLFM (Yokoyama et al. ICRA 2024): https://arxiv.org/abs/2312.03275
- SG-Nav (Yin et al. NeurIPS 2024): https://arxiv.org/abs/2410.20619
- VoroNav (Wu et al. ICML 2024): https://arxiv.org/abs/2406.04192
- ImagineNav (Zhao et al. ICLR 2025): https://arxiv.org/abs/2410.03034
- ApexNAV (Zhang et al. RAL 2025): https://arxiv.org/abs/2503.14258

**基础组件**：
- UFOMap (Duberg et al. RAL 2020): https://arxiv.org/abs/2003.05551
- RT-RRT* (Naderi et al. MIG 2015): https://dl.acm.org/doi/10.1145/2822013.2822016
- Schmid et al. informative path planning (RAL 2020): https://arxiv.org/abs/1912.04770
- SAM2 (Ravi et al. 2024): https://arxiv.org/abs/2408.00714
- YOLO-World (Cheng et al. CVPR 2024): https://arxiv.org/abs/2401.17270
- MobileCLIP2 (Faghri et al. TMLR 2025): https://arxiv.org/abs/2503.09183
- Weiszfeld algorithm 原始论文（1937，数学史）：https://www.jstage.jst.go.jp/article/tmj1911/43/0/43_0_355/_article
- OAESMT (Zachariasen 1999): https://link.springer.com/chapter/10.1007/978-3-540-48867-5_19

**Steiner Tree / MST 基础**：
- Kruskal 原始论文：https://www.ams.org/journals/proc/1956-007-01/S0002-9939-1956-0078686-7/
- Euclidean Steiner Tree NP-hardness 综述：https://www.sciencedirect.com/science/article/pii/S0304397502002023

**Habitat / Benchmark**：
- Habitat Challenge 2023: https://aihabitat.org/challenge/2023/
- HM3D: https://arxiv.org/abs/2109.08238
- HSSD-200: https://arxiv.org/abs/2405.17957
- Gibson Env: https://arxiv.org/abs/1801.07506

---

## 6. 一句话总结给 Karpathy

REST 的核心 thesis 是把 ZSON 的 option space 从 "set of independently scored waypoints" 升级为 "Steiner-tree-structured path tree textualized into language options"，让 LLM 做 coarse-to-fine branch-level reasoning 而不是 leaf-level scoring。工程上用 UFOMap + RT-RRT* + Weiszfeld + Kruskal 搭建一个 model-based backbone，把 LLM 退化为 semantic prior injector。结果上 HSSD 完胜，HM3D 上 SPL 最高，Gibson 接近饱和。本质是把 navigation 重新 framing 成 informative path planning + LLM ranking 的 hybrid，避开了端到端 VLM nav 的数据稀缺陷阱，也避开了纯 frontier-based 的短视问题。
