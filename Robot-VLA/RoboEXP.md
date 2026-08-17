---
source_pdf: RoboEXP.pdf
paper_sha256: 884b64b36b4518791787ba398e5a56792927e44ce330ddeca6af0aed44c46fba
processed_at: '2026-08-12T00:49:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RoboEXP

## 一句话版本

**机器人怎么"翻箱倒柜"找东西，并且记住"打开冰箱会看到苹果"这种因果关系。**

---

## 问题是什么

想象一个家庭机器人要准备早餐。它得翻冰箱、开抽屉拿餐具、掀开盖子找剩菜。这里的核心难题：很多东西**藏着**，你得动手才能看到。

以前的 robotic exploration 基本都是"走来走去换角度看"——典型的 frontier-based exploration，机器人找个没探索过的边界走过去，扫一圈地图。这够用于 navigation，对于 manipulation 远远不够。

你要拿冰箱里的牛奶，光靠看外面不行，得**打开冰箱门**。而且有时候冰箱门把手还被个调味瓶挡住了，得先挪开瓶子。这种"要做A得先做B"的因果链，传统 scene graph 根本表达不了。

RoboEXP 要解决的就是这个：**让机器人通过主动交互来探索环境，构建一个带"动作因果关系"的场景图**。

---

## ACSG 是什么东西

普通 scene graph 长这样：节点是物体，边是"挨着"、"属于"这类静态关系。

ACSG (Action-Conditioned Scene Graph) 把 **action 也变成节点**。

举个例子，table 上有个 cabinet，cabinet 门把手上挡着个 condiment 瓶，cabinet 里有 tape。

普通 scene graph：
```
cabinet — condiment — tape (都是 spatial 关系)
```

ACSG：
```
condiment → [pick condiment] → [open cabinet door] → tape
   (object)    (action)            (action)        (object)
```

这条路径就是"要拿 tape，先 pick condiment，再 open cabinet，然后 tape 就露出来了"。

这种图的核心价值：**给定任意目标物体，沿着图上从 root 到 target 的路径顺序执行 action，就能拿到**。这把复杂的任务规划问题变成了图遍历问题。

---

## 系统怎么转的

RoboEXP 是个闭环，四个 module 轮转：

### Perception（眼睛）

wrist 上装个 RealSense D455 RGBD 相机。用 GroundingDINO 做 open-vocabulary detection（你说"找 fridge"它就能找，不用专门训练 fridge detector），用 SAM-HQ 做 segmentation，用 CLIP 提每个物体 instance 的 semantic feature。

关键：这些全都是 **off-the-shelf foundation model**，没训练任何东西。这是 zero-shot 的。

### Memory（记性）

这里分两层。

**Low-level memory**：把不同视角看到的同一个物体 merge 起来。用 3D IoU + CLIP feature 相似度 + label 一致性 + detection confidence 四个判据。存成 voxel-based 表示，方便后续取几何信息。

**High-level memory**：就是 ACSG 本身。每看到新东西、执行完一个 action，就 update 图——加 node、加 edge、删过期的。

这里有个关键设计：**跨时间的一致性**。比如上一帧门是关的，这一帧机器人把门打开了，memory 里门的状态要更新。用 depth test 判断 memory 里的东西还有效否。

### Decision-Making（脑子）

GPT-4V 干两件事：

**第一件事：action proposer**。给它看一个物体的多视角图（绿色 bbox 标出），问它"这东西该怎么探索？"GPT-4V 回答"open_door" / "pick_object" / "no_action"。

比如给它看 fridge，它会说"open the doors or drawers"。给它看 chair，它说"no action"（椅子没啥好翻的）。

关键设计：**这一步只看物体本身，不管周围环境**。这样降低 GPT-4V 的认知负担，避免它一次性处理太多 context 出错。

**第二件事：action verifier**。proposer 说"open the cabinet door"之后，verifier 检查"这个 action 现在能做吗？有东西挡着吗？"

如果有 condiment 挡着门把手，verifier 会说"不能直接开，得先 pick condiment"。这就生成了 precondition edge $e_{a \to a}$。

这两步解耦的设计非常聪明：**选 skill 用 object-level commonsense，判断可行性用 scene-level reasoning**，两者所需的信息粒度不同。

### Action（手）

7 个 handcrafted action primitives：open_door, open_drawer, close_door, close_drawer, pick_to_idle, pick_back, move_camera。

对 door/drawer 的处理挺有意思：
1. 从 voxel memory 取 handle 的点云
2. **PCA** 算 handle 主方向 → 对齐 gripper
3. 分析 handle 邻域点的 normal，取众数作为 opening direction
4. Drawer 是 prismatic joint，直接沿 axis 平移
5. Door 是 revolute joint，用 inferred motion parameters 模拟旋转过程中方向的变化

对 pick：简化为 top-down grasping，取物体点云最高点的 mean 作为 grasp point。

这些 heuristic 在 tabletop controlled 场景下够用，但离 general household skill 还远。作者自己也承认这点。

---

## 怎么处理"套娃"这种递归场景

Matryoshka doll（俄罗斯套娃）是 5 层嵌套，只看得到最外层。你得 pick 外层 → 看到第二层 → pick 第二层 → ... 直到最内层。

纯 LLM 一次性 code generation（VoxPoser、VILA 那套）处理不了这种，因为需要根据每一步的 observation feedback 决定下一步。

RoboEXP 的解法：**action stack**。

- pick 外层 doll → 发现里面还有 doll → decision module 说"pick 内层 doll" → push 到 stack top
- 这个新 action 优先级高于"pick back 外层 doll"
- 处理完内层再 pop，依次 pick back

简单粗暴但有效。局限是只能处理 linear recursion，branching recursion 就不够了。

---

## Reward 函数到底有没有用

论文形式化了一个 reward：

$$R^t = R_{\text{graph}}^t + R_{\text{explore}}^t + R_{\text{time}}^t$$

- $R_{\text{graph}}^t = |V^t| - |V^{t-1}|$：发现新节点就 +1
- $R_{\text{explore}}^t = \max(0, |U^{t-1}| - |U^t|)$：减少未探索节点就奖励
- $R_{\text{time}}^t = -\lambda$：每步小惩罚，鼓励高效

但说实话，**实际系统没用 RL 训练 policy**，直接用 GPT-4V 做 greedy decision。这个 reward 更像是 **intuition justification 和 evaluation metric**，给你解释"为什么这套逻辑 make sense"。

其实这更像 frontier-based exploration 的图结构版本：传统 frontier method 在 occupancy grid 上找 known/unknown 边界移动，RoboEXP 在 ACSG 上找 explored/unexplored 节点边界，用 LMM 决定怎么探索。每个 frontier 节点带 semantic context，决策更有针对性。

---

## 实验讲了啥

### 主结果

5 种任务 × 10 variations，跟 GPT-4V baseline、heuristic baseline、random baseline 比。

几个关键数字：

**Drawer-Only（简单场景）**：RoboEXP 90% success，Heuristic-Open 也 90%。简单场景 heuristic 够用了。

**Recursive（套娃）**：RoboEXP 70%，GPT-4V 0%，Heuristic-Open 0%，Heuristic-Full 40%。GPT-4V 彻底失败因为不会 closed-loop multi-step reasoning，heuristic 失败因为不会 pick。

**Occlusion（遮挡）**：RoboEXP 50%，所有 baseline 0%。这是唯一所有 baseline 全军覆没的场景，凸显 precondition reasoning 的价值。GPT-4V 不会推理"门被挡住了要先挪开瓶子"。

### 效率

在所有方法都成功的 setting 里数 action 数：

| Method | Drawer-Only | Recursive | Occlusion |
|--------|-------------|-----------|-----------|
| Human GT | 4.0 | 4.6 | 6.0 |
| RoboEXP | 4.0 | 4.6 | 6.3 |
| GPT-4V | 6.0 | - | - |
| Heuristic-Full | 7.0 | 6.0 | - |

RoboEXP 几乎跟人类动作数完全一致，GPT-4V 多 50%，Heuristic-Full 多 75%。说明 LMM commonsense guidance 确实 aligned with human intuition。

### Error 来源

RoboEXP 主要失败在 perception（detection/segmentation 错误），decision 和 action error 很少。GPT-4V baseline 即使人工辅助消除 action error，decision error 仍高。

这指向一个 future direction：**temporal semantic fusion**，用连续观测增强 perception robustness，而非依赖 single-frame detection。

### 不同 LMM 对比

GPT-4V: 95% success, 0.1 GED  
LLaVA-v1.6-34b: 25% success, 2.5 GED

框架对 LMM choice modular，但 LMM 能力影响巨大。GPT-4V 的 commonsense reasoning 明显更强。

---

## 我的直觉与批判

### 为什么 ACSG 比 pure LLM memory 好

VoxPoser、VILA 这类让 LLM 在 context window 里记住所有信息，但：
1. Long-horizon 下 context 爆炸：Matryoshka 5 层嵌套，每层都要描述物体+action+状态变化，token 指数增长
2. LLM 难以稳定推理 precondition chain："要拿 tape 得先移 condiment 再 open cabinet"这种链式逻辑，GPT-4V 在 context 里经常丢

ACSG 把这种结构外化为图，LLM 只需在 local node 做 decision，global consistency 由图结构保证。本质上是 **neuro-symbolic** 思路——LLM 提供 local commonsense，symbolic graph 保证 global coherence。

这跟你常说的 "system 2 thinking" 很契合：ACSG 是一种 **externalized working memory**，把 LLM 的 implicit reasoning 外化为可遍历、可编辑、可验证的图结构。这种范式比纯 LLM context-pumping 更 scalable，尤其在 long-horizon 任务上。

### Action Stack 的精妙与局限

用 stack 处理 recursive reasoning 非常工程化但有效。对比 hierarchical RL 或 tree search：
- 实现极简
- 与 LMM sequential query 模式天然契合
- 自然支持 greedy state recovery（pop 时执行 reverse action）

局限：只能处理 linear recursion。如果同时打开两个抽屉各有嵌套（branching recursion），stack 不够，需要 tree 或 DAG 结构。

### Perception 是真瓶颈

Fig. 4b error breakdown 说明主要失败在 perception。GroundingDINO + SAM-HQ 在 tabletop controlled 场景尚可，但：
- Open-vocabulary detection 对小物体、遮挡物体 recall 不足
- SAM-HQ 对透明、反光、deformable 物体（布料）不稳定
- CLIP feature 在 instance-level 区分相似物体（两个不同水果）时 confusion

这指向 temporal semantic fusion——用连续观测时序信息增强 robustness。

### Action Primitive 的 generalization 限制

7 个 handcrafted primitives 在 tabletop 够用，但：
- "pick object to idle space" 假设 top-down grasping，侧抓倒抓不行
- "open the door" 假设 handle 能被 PCA 对齐，hidden handle、touch-latch 失败
- 没有 pour、push、pull 等丰富 skill

要扩展到真实 household，需要 **learnable skill library**（diffusion policy、ACT）与 ACSG 集成。

### 跟 ConceptGraphs 的关系

ConceptGraphs 也是 open-vocabulary 3D scene graph，但是 static、passive perception，没有 action node。RoboEXP 是 dynamic、active，action 是 first-class citizen。

两者可以互补：ConceptGraphs 风格的 edge prediction 可以增强 RoboEXP 的 $e_{o \to o}$ reasoning，RoboEXP 的 action node 可以给 ConceptGraphs 加上 interactive dimension。

### Reward 没用于训练

虽然形式化了 reward，实际没做 RL，只用 GPT-4V greedy decision。这个 reward 更像 evaluation metric。如果未来用它训 RL policy（PPO）warm-start LMM query，可能进一步提升效率。比如用 RL 学个 node selection policy，决定"先探索哪个 frontier 节点"，GPT-4V 只在不确定时 query。

---

## 更远的联想

### Sim-to-Real with ACSG

ACSG 的 symbolic 性质可能助力 sim-to-real。在 Habitat 或 Isaac Sim 预训练 ACSG 构建 policy，transfer 到 real。图结构对 observation noise 相对 robust，因为离散化掉了连续感知的不确定性。

### ACSG + Diffusion Policy

用 ACSG 的 action node 作为 high-level plan，diffusion policy 作为 low-level executor。这样能突破 handcrafted primitives 限制，支持丰富 skill。ACSG 提供"做什么"，diffusion policy 提供"怎么做"。

### Multi-Robot ACSG

多个机器人协同构建 ACSG。需要解决图融合（merge 各自局部 ACSG）、action 冲突（两机器人同时想开同一扇门）、通信效率（只传 graph delta）。

### Causal ACSG

当前 $e_{a \to a}$ 只表达 precondition，能否扩展为 causal graph 支持 counterfactual reasoning？"如果没移走 condiment，打开 cabinet 会失败"——这种反事实推理能让机器人更智能地规划。

### Active ACSG Refinement

当 LMM 对某 action 信心不足，主动 query 人类或执行试探性 action 来 refine graph。这本质是 active learning 在 graph 结构上的应用。

---

## 最核心的 takeaway

RoboEXP 的核心 insight：**把 action 显式纳入 scene graph，让机器人能 reasoning "做什么才能看到什么"**。

它结合 LMM 的 commonsense（local decision）和 symbolic graph 的 structural guarantee（global consistency），用 modular 闭环系统在 real-world 验证了 effectiveness。

从你的 "system 2 thinking" 视角，ACSG 是一种 **externalized working memory**——把 LLM 的 implicit reasoning 外化为可遍历、可编辑、可验证的图结构。这种 neuro-symbolic 范式比纯 LLM context-pumping 更 scalable，尤其 long-horizon 任务。

主要瓶颈在 perception robustness 和 action primitive generalization。如果用 learnable skill policies 替换 handcrafted primitives，用 temporal fusion 增强 perception，这套框架有潜力成为 household robot exploration 的 backbone。

---

## 参考

- Project page: https://jianghanxiao.github.io/roboexp-web/
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM-HQ: https://arxiv.org/abs/2306.01567
- CLIP: https://arxiv.org/abs/2103.00020
- ConceptFusion: https://arxiv.org/abs/2302.07241
- OVIR-3D: https://proceedings.mlr.press/v229/lu23a.html
- GPT-4V System Card: https://cdn.openai.com/papers/GPTV_System_Card.pdf
- VoxPoser: https://arxiv.org/abs/2307.05973
- VILA: https://arxiv.org/abs/2311.17842
- ConceptGraphs: https://arxiv.org/abs/2309.16650
- PaLM-E: https://arxiv.org/abs/2303.03378
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- SayCan: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753
- Ditto (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Ditto_Building_Digital_Twins_of_Articulated_Objects_From_Interaction_CVPR_2022_paper.pdf
- Frontier-based Exploration (Yamauchi 1997): https://ieeexplore.ieee.org/document/620822
- Active Perception (Bajcsy 1988): https://ieeexplore.ieee.org/document/99966
- Curiosity-driven Exploration: https://arxiv.org/abs/1705.05363
- 3D Scene Graph (Armeni et al. ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/papers/Armeni_3D_Scene_Graph_A_ICCV_2019_paper.pdf

---

# RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation

## 1. 论文核心动机与定位

这篇 paper 来自 Columbia University (Yunzhu Li)、UIUC (Shenlong Wang) 和 Amazon 等团队，发表于 2024 年。核心想解决的问题是：**机器人如何在一个部分可观测的复杂环境中，通过主动交互来构建一个"动作条件化"的场景图，并用它来支撑下游 manipulation 任务**。

传统 robotic exploration 主要关注 navigation 场景下的 viewpoint 改变（frontier-based、information-theoretic），而 RoboEXP 把 exploration 扩展到了 **manipulation-driven exploration**——机器人不是仅仅"看"，而是"动手翻找"，通过打开抽屉、挪开遮挡物、掀起布料等方式揭示隐藏的物体。

项目主页：https://jianghanxiao.github.io/roboexp-web/

---

## 2. 核心概念：Action-Conditioned 3D Scene Graph (ACSG)

### 2.1 为什么需要 ACSG

传统 3D scene graph（如 ConceptGraphs、3D Scene Graph for Robotics）只编码静态的空间关系和语义关系，无法表达"打开冰箱会露出苹果"这种**action-conditioned 关系**。ACSG 把 action 作为 first-class citizen 引入图结构中。

### 2.2 形式化定义

ACSG 是一个有向无环图：

$$\mathbf{G} = (\mathbf{V}, \mathbf{E})$$

其中节点集 $\mathbf{V}$ 包含两类节点：

**Object node**:
$$\mathbf{o}_i = (\mathbf{s}_i, \mathbf{p}_i) \in \mathbf{V}$$

- $\mathbf{s}_i$：semantic feature（来自 CLIP 的 per-instance feature）
- $\mathbf{p}_i$：geometry information（位置、朝向、点云、bbox）

**Action node**:
$$\mathbf{a}_k = (a_k, \mathbf{T}_k) \in \mathbf{V}$$

- $a_k$：high-level action type（如 "open_door"、"pick_object"）
- $\mathbf{T}_k$：low-level motion primitives（具体的轨迹参数、grasping pose、motion axis）

边集 $\mathbf{E}$ 分四类：

| 边类型 | 符号 | 例子 |
|--------|------|------|
| Object → Object | $\mathbf{e}_{o \to o}$ | door handle belongs to fridge |
| Object → Action | $\mathbf{e}_{o \to a}$ | toy can be picked up |
| Action → Object | $\mathbf{e}_{a \to o}$ | opening cabinet reveals banana |
| Action → Action | $\mathbf{e}_{a \to a}$ | move condiment before opening cabinet (precondition) |

### 2.3 关键优势：topological retrieval

只要给定 ACSG 和目标 object node，机器人只需从 root 到 target 的路径上**按 topological order 顺序执行所有 action 节点**就能 retrieve 物体。例如要拿 tape，路径是：移走 condiment → 打开 cabinet → 取 tape。这种"显式 precondition 链"是 ACSG 相比传统 scene graph 最有价值的属性。

---

## 3. 问题形式化：POMDP-inspired Formulation

由于 partial observability，作者把 ACSG 构建形式化为 active perception 问题：

在时间步 $t$：
- 基于 past graph estimation $\mathbf{G}^{t-1}$
- 基于 past sensor observations $\mathbf{O}^{t-1}$
- Agent 采取 action $\mathbf{A}^t$
- 环境状态转移
- Agent 收到 new observation $\mathbf{O}^t$
- 更新 graph $\mathbf{G}^t$
- 更新未探索节点集 $\mathbf{U}^t \subset \mathbf{V}$

### 3.1 Reward 函数

$$\mathbf{R}^t = \mathbf{R}_{\text{graph}}^t + \mathbf{R}_{\text{explore}}^t + \mathbf{R}_{\text{time}}^t$$

逐项解释：

**Graph discovery reward**:
$$\mathbf{R}_{\text{graph}}^t = |\mathbf{V}^t| - |\mathbf{V}^{t-1}|$$

- $|\mathbf{V}^t|$：当前时刻图中的总节点数
- $|\mathbf{V}^{t-1}|$：上一时刻图中的总节点数
- 含义：每发现一个新节点（无论 object 还是 action）就给 +1，鼓励发现更多结构

**Exploration reward**:
$$\mathbf{R}_{\text{explore}}^t = \max(0, |\mathbf{U}^{t-1}| - |\mathbf{U}^t|)$$

- $\mathbf{U}^{t-1}$：上一时刻的未探索节点集合
- $\mathbf{U}^t$：当前时刻的未探索节点集合
- 注意：执行 action 可能让一个未探索节点变成 explored，但同时这个 action 又揭示了新的未探索 object（例如打开抽屉发现了 banana，banana 成为新未探索节点）
- max(0, ...) 保证不会因为发现新未探索节点而被惩罚

**Time penalty**:
$$\mathbf{R}_{\text{time}}^t = -\lambda, \quad 0 < \lambda < 1$$

- 一个小常数惩罚，鼓励高效探索
- 当所有节点都 explored 时终止

**Intuition**: 这个 reward 设计的本质是 frontier-based exploration 的图结构版本——优先探索当前 graph 中"可能揭示新节点"的未探索 frontier 节点，而非随机动作。

---

## 4. RoboEXP 系统架构

系统由四个模块组成闭环：**Perception → Memory → Decision-Making → Action → Perception**。

### 4.1 Perception Module

输入：多个 viewpoint 的 RGBD 图像（来自 wrist-mounted RealSense D455）

Pipeline：
1. **Open-vocabulary detection**: GroundingDINO [1] 使用预定义 tag list 检测物体
2. **Segmentation**: SAM-HQ [2] 生成高质量 mask
3. **Semantic feature extraction**: per-instance CLIP feature [3]，采用 Jatavallabhula et al. (ConceptFusion) [4] 的 local-global merging 策略，并融合 label text feature 增强 semantic 表达
4. 仅保留 instance-level feature（丢弃 pixel-level），加速后续融合

GroundingDINO 论文：https://arxiv.org/abs/2303.05499  
SAM-HQ 论文：https://arxiv.org/abs/2306.01567  
CLIP 论文：https://arxiv.org/abs/2103.00020

### 4.2 Memory Module

这是论文最关键的工程创新之一，分两层：

#### 4.2.1 Low-Level Memory (跨 viewpoint 融合)

借鉴 OVIR-3D [5] 的 instance merging 策略，但增加两个判据：
- **3D IoU**（空间重叠）
- **Semantic feature similarity**（CLIP cosine）
- **Label similarity**（文本标签一致性，作者新增）
- **Instance confidence**（detection 置信度，作者新增）

实现上采用 **voxel-based representation + filtering**，处理 tabletop 场景中物体拥挤的情况，得到更干净的几何表示。

#### 4.2.2 Cross-time Memory Update

环境是 dynamic 的（机器人自己改了环境状态），需要处理 outdated memory：
- 通过 depth test 判断 memory 中的元素是否仍然 valid
- 例如上一时刻门关着，本时刻机器人打开了门，需要更新门的状态

#### 4.2.3 High-Level Graph Update

根据 low-level memory 的变化，更新 ACSG 的 nodes 和 edges：
- 添加新发现的 object node
- 添加 action node 及其 precondition edge $\mathbf{e}_{a \to a}$
- 删除失效的 node/edge
- 修改现有 node 的 state

### 4.3 Decision-Making Module

GPT-4V [6] 在这里承担两个角色，使用 fixed prompt（所有实验不改 prompt）：

#### Role 1: Action Proposer

输入：query object 的 multi-view images（绿色 bounding box 标注）  
输出：推荐的 skill 类型（open_door / open_drawer / pick_object / no_action）

关键设计：**仅分析 object-centric attributes，忽略周围信息**——这是为了降低 LMM 的认知负担，避免一次性处理过多 context。

#### Role 2: Action Verifier

输入：proposed action + 当前 ACSG 中周围物体的信息  
输出：
1. action 是否 feasible
2. 是否有 obstacle 需要 precondition action

这个两阶段设计的 intuition 是把"选择什么 skill"和"判断可行性"解耦——前者需要 object 本身的常识（fridge 有门可以打开），后者需要 scene-level reasoning（门被 condiment 挡住了）。

GPT-4V System Card: https://cdn.openai.com/papers/GPTV_System_Card.pdf

平均响应时间约 8 秒/query，由于只在 high-level planning 调用，低层 motion planning 由 action primitives 处理，满足实时性。

### 4.4 Action Module

设计 7 个 action primitives：

| Primitive | 输入 | 实现 |
|-----------|------|------|
| open the [door] | object node | handle geometry + PCA 主方向 + 邻域 normal 推断 opening direction；revolute joint 用 inferred motion parameters 模拟 evolving opening direction |
| open the [drawer] | object node | prismatic joint，沿 motion axis 平移 |
| close the [door] | object node | 反向运动 |
| close the [drawer] | object node | 反向运动 |
| pick [object] to idle space | object node | top-down grasping，提取点云取最高点 mean 作为 grasp point |
| pick back [object] | object node | 把之前 pick 走的物体放回原位 |
| move wrist camera to [position] | target pose | 视角调整，常用在打开 door/drawer 后看内部 |

**Handle geometry 的处理细节**：
1. 从 voxel-based memory 提取 handle 的点云
2. PCA 确定 handle 主方向，用于对齐 gripper
3. 分析邻域点的 normal 分布，取众数作为 opening direction
4. 对 revolute joint，根据 inferred motion parameters 模拟 opening 过程中方向的变化

这种 heuristic 在 tabletop setting 下可靠，但作者在 Limitation 中承认对更 generalizable 的 skill module 仍有需求。

---

## 5. Algorithm 细节

主算法（Algorithm 1: Interactive Exploration）的伪代码核心逻辑：

```
while |U^{t-1}| != 0:
    if choose object o_i ∈ U^{t-1}:  # 探索 object
        add spatial relations (Algorithm 2)  # e_{o→o}
        obtain action a to explore o_i  # decision-making proposer
        if a ∉ V^{t-1}:
            V^t = V^{t-1} ∪ {a}, U^t = U^{t-1} ∪ {a}  # 添加 action node
            E^t = E^{t-1} ∪ {e_{o_i→a}}  # object → action 边
        U^t = U^t \ {o_i}  # 标记 object 为 explored
    else:  # choose action a_k ∈ U^{t-1}
        if no obstruction:
            take action a_k  # 执行
            obtain new observation O^t
            if found new objects O ∉ V^{t-1}:
                V^t = V^t ∪ {O}, U^t = U^{t-1} ∪ {O}  # 添加新 object node
                E^t = E^t ∪ {e_{a_k→O}}  # action → object 边
            U^t = U^t \ {a_k}  # 标记 action 为 explored
        else:
            add action preconditions (Algorithm 3)  # 添加 e_{a→a}
```

**Algorithm 2 (Add Spatial Relations)**：遍历所有 object node，用 spatial heuristics 判断物体间的从属/相邻关系（如 handle belongs to fridge），添加 $\mathbf{e}_{o \to o}$ 边。

**Algorithm 3 (Add Action Preconditions)**：当 verifier 判断 action 有 obstacle 时：
1. 选择 obstacle object 的处理 action a
2. 添加 a 到 $\mathbf{V}^t$ 和 $\mathbf{U}^t$
3. 添加 $\mathbf{e}_{o \to a}$（obstacle object → clear action）
4. 添加 $\mathbf{e}_{a \to a_k}$（clear action → original action，构成 precondition 链）

### 5.1 Action Stack：multi-step reasoning 的实现

为了处理 Matryoshka doll 这种 recursive scenario，系统维护一个 action stack：
- 当 pick 起顶层 doll 后发现内层还有 doll，decision module 决定 pick 内层 doll
- 新的 pick action 被 push 到 stack top
- 优先级高于"pick back 外层 doll"
- 这构建了一个 deep、coherent 的 exploration 逻辑

这种设计让系统能处理超过 single-step LLM code generation [7,8] 能力的场景。

### 5.2 Greedy State Recovery

每完成一个 object 的探索，greedily 把物体放回原状态（关上门、放回 pick 的物体），保持环境接近 pre-exploration 状态，提高实用性。

---

## 6. 实验设计

### 6.1 硬件 Setup

**Tabletop**:
- Robot arm: UFACTORY xArm 7
- End effector: soft gripper
- Camera: RealSense D455（wrist-mounted）

**Mobile**:
- Hello Robot Stretch2 + official upgraded kits
- 仅需修改 action module 适配新 kinematics，其他模块几乎不变

### 6.2 五种任务类型（每种 10 个 variations）

| Task Type | 描述 | 难点 |
|-----------|------|------|
| Drawer-Only | 仅抽屉场景 | 基础 prismatic joint |
| Door-Only | 仅门场景 | revolute joint |
| Drawer-Door | 混合场景 | 多种 joint 类型组合 |
| Recursive | Matryoshka doll 嵌套 | deep graph，multi-step reasoning |
| Occlusion | 物体遮挡 door handle | precondition edge $\mathbf{e}_{a \to a}$ |

### 6.3 五个评估指标

1. **Success %**：最终 ACSG 与 GT 完全匹配的比例（binary）
2. **Object Recovery %**：发现的隐藏物体比例
3. **State Recovery %**：探索后环境状态是否恢复到初始
4. **Unexplored Space %**：未探索的需要探索的空间比例（越低越好）
5. **Graph Edit Distance (GED)**：输出图与 GT 图的编辑距离，6 种操作（node add/delete/edit, edge add/delete/edit），每种 cost 1

### 6.4 Baselines

- **GPT-4V Baseline**：纯 GPT-4V + CoT，闭环接收 RGB，每轮生成 scene graph 和 next action；为公平比较，action 由人工辅助执行（eliminating action errors），是 baseline 的上界
- **Heuristic-Open**：打开所有 handle
- **Heuristic-Full**：打开所有 handle + pick 所有 movable object
- **Random**：随机选择 action（包括 no action）

所有 baseline 共享 RoboEXP 的 perception module，隔离 decision strategy 的影响。

---

## 7. 实验结果分析

### 7.1 主表（Table 1 & Table 2）

关键数据点：

**Drawer-Only**:
- RoboEXP Success: 90% vs GPT-4V: 20% vs Heuristic-Open: 90%
- GED: 0.2 (Ours) vs 2.8 (GPT-4V) vs 0.2 (Heuristic-Open)
- 注意：Heuristic-Open 在这个简单场景能达到和 RoboEXP 相当的效果

**Recursive (Matryoshka)**:
- RoboEXP Success: 70%, GED: 2.1
- GPT-4V: 0%, GED: 8.8
- Heuristic-Open: 0%, GED: 10.0（完全失败，因为不会 pick）
- Heuristic-Full: 40%, GED: 1.8（pick 所有物体反而有时能 work，但 Success 低因为 graph 结构混乱）
- 这里展示了 LMM-guided 决策的必要性

**Occlusion**:
- RoboEXP Success: 50%, GED: 2.5
- 所有 baseline Success: 0%
- 这是唯一所有 baseline 都彻底失败的场景，凸显 precondition reasoning 的价值

### 7.2 效率比较（Table 3）

在所有方法都成功找到所有物体的 setting 中比较 action 数量：

| Method | Drawer-Only | Door-Only | Drawer-Door | Recursive | Occlusion |
|--------|-------------|-----------|-------------|-----------|-----------|
| GT (Human) | 4.0 | 4.0 | 8.0 | 4.6 | 6.0 |
| Heuristic-Full | 7.0 | 8.0 | 12.0 | 6.0 | - |
| GPT-4V | 6.0 | 4.4 | 8.0 | - | - |
| Ours | 4.0 | 4.0 | 8.0 | 4.6 | 6.3 |

RoboEXP 的 action 数几乎与人类 GT 完全一致，证明其决策高度 aligned with human commonsense。

### 7.3 Error Analysis（Fig. 4b）

错误分为四类：perception、decision、action、no-error。
- RoboEXP 的主要失败来自 perception（detection/segmentation 错误）
- GPT-4V baseline 即使消除了 action error（人工辅助），decision error 仍很高

### 7.4 不同 LMM 的对比（Table 4）

- GPT-4V: Success 95%, GED 0.1
- LLaVA-v1.6-34b: Success 25%, GED 2.5

证明 RoboEXP 框架对 LMM choice 是 modular 的，但 LMM 能力显著影响整体性能。GPT-4V 的 commonsense reasoning 明显更强。

### 7.5 Robustness 实验

- **Extreme illumination + random background**（Fig. 10）：4 个场景 × 4 种条件，共 20 个 setting，系统在所有条件下都能成功构建 ACSG
- **Human intervention**（Fig. 11）：
  - Type 1: 添加新 cabinet，系统自动检测并探索
  - Type 2: 人工从 cabinet 中添加/移除物体，系统监测并 re-explore
- **Room-level scenarios**（Fig. 12）：手持 D455 采集 4 个 RGBD 视角，ICP multi-way alignment，在 dining area 和 bedroom 成功构建 scene graph

---

## 8. 与相关工作的关联

### 8.1 Scene Graph 方向

- 传统 2D scene graph: Johnson et al. (Image Retrieval using Scene Graphs) [9]
- 3D scene graph: Armeni et al. (3D Scene Graph for ICCV 2019) [10]
- LLM-assisted: ConceptGraphs [11], 3D Scene Graph via Language-Enabled Spatial Ontologies [12]
- **RoboEXP 的差异**: action 作为 first-class node，而非仅 spatial/semantic relation

### 8.2 Robotic Exploration

- Frontier-based: Yamauchi 1997 [13]
- Information-theoretic: Charrow et al. (RSS 2015) [14]
- Curiosity-driven: Pathak et al. (ICML 2017) [15], Burda et al. (ICLR 2019) [16]
- Manipulation-based exploration: Agrawal et al. (Learning to Poke, NeurIPS 2016) [17], Ditto (CVPR 2022) [18]
- **RoboEXP 的差异**: 用 LMM commonsense 替代 handcrafted rules / information gain / RL policy

### 8.3 Active Perception

- Bajcsy 1988 (Active Perception) [19]
- Next-Best-View planning: Bircher et al. (ICRA 2016) [20]
- Affordance landscapes: Nagarajan & Grauman (NeurIPS 2020) [21]
- **RoboEXP 的差异**: 显式 ACSG 作为 exploration objective，而非连续 viewpoint optimization

### 8.4 LLM/LMM for Robotics

- SayCan [22], Code as Policies [23], VoxPoser [24], VILA [25]
- PaLM-E [26]
- **RoboEXP 的差异**: 用 explicit memory (ACSG) 而非 brute-force LLM memory；modular 设计避免 single-shot code generation 的局限

---

## 9. 我的 Intuition 与批判性思考

### 9.1 为什么 ACSG 比 implicit LLM memory 更好

VoxPoser、VILA 这类工作让 LLM 在 context window 里"记住"所有信息，但这有两个问题：
1. **Long-horizon 场景下 context爆炸**：Matryoshka 5 层嵌套，每层都要描述物体 + action + 状态变化，token 消耗指数增长
2. **No structured precondition reasoning**：LLM 难以稳定地推理"要拿 tape 必须先移走 condiment 再打开 cabinet"这种 chain

ACSG 把这种结构外化为图，LLM 只需在 local node 上做 decision，global consistency 由图结构保证。这本质上是 **neuro-symbolic** 思路——用 LLM 提供 local commonsense，用 symbolic graph 保证 global coherence。

### 9.2 Action Stack 的简洁性

用 stack 处理 recursive reasoning 是个非常工程化的设计。对比 hierarchical RL 或 tree search，stack 的优势是：
- 实现极简
- 与 LMM 的 sequential query 模式天然契合
- 自然支持 greedy state recovery（pop stack 时执行 reverse action）

但局限是 stack 只能处理 linear recursion，对 branch recursion（同时打开两个抽屉各有嵌套）会受限。

### 9.3 Perception 是瓶颈

Fig. 4b 的 error breakdown 显示 RoboEXP 的主要失败来自 perception。GroundingDINO + SAM-HQ 在 tabletop controlled 场景下尚可，但：
- Open-vocabulary detection 在小物体、遮挡物体上 recall 不足
- SAM-HQ 对透明、反光、deformable 物体（如布料）的 segmentation 不稳定
- CLIP feature 在 instance-level 区分相似物体（如两个不同的水果）时 confusion

这指向一个 future direction：**temporal semantic fusion**——用连续观测的时序信息增强 perception robustness，而非依赖 single-frame detection。

### 9.4 Action Primitive 的 generalization 限制

7 个 handcrafted primitives 在 tabletop 工作，但：
- "pick object to idle space" 假设 top-down grasping，对侧抓、倒抓不适用
- "open the door" 假设 handle 可被 PCA 主方向对齐，对 hidden handle、touch-latch 失败
- 没有 pour、push、pull 等更丰富 skill

要扩展到真实 household，需要 **learnable skill library**（如 diffusion policy、ACT）与 ACSG 集成。

### 9.5 与 ConceptGraphs 的对比

ConceptGraphs (Gu et al. 2023) [11] 也是 open-vocabulary 3D scene graph，但：
- ConceptGraphs 是 static、passive perception
- RoboEXP 是 dynamic、active perception with action nodes
- ConceptGraphs 用 LLM 边权预测 relation，RoboEXP 用 LLM 节点级 action proposal + verifier

两者可以互补：ConceptGraphs 风格的边预测可以增强 RoboEXP 的 $\mathbf{e}_{o \to o}$ 推理。

### 9.6 Reward 函数的实际使用

虽然论文形式化了 reward $\mathbf{R}^t$，但实际系统**并没有用 RL 优化 policy**，而是用 LMM 直接做 greedy decision。这个 reward 更像是 **evaluation metric** 或 **intuition justification**，而非训练信号。如果未来用这个 reward 训练 RL policy（如 PPO）来 warm-start LMM query，可能进一步提升效率。

### 9.7 与 Frontier-Based Exploration 的关系

作者在 appendix 提到 "RoboEXP can be viewed as a special form of frontier-based method with LMM-guided action selection"。这个类比很精确：
- 传统 frontier-based：在 occupancy grid 上找 known/unknown 边界，朝边界移动
- RoboEXP：在 ACSG 上找 explored/unexplored 节点边界（即 $\mathbf{U}$ 中的节点），用 LMM 选择如何探索

这种图结构 frontier 的优势是：每个 frontier 节点带有 semantic 和 action context，决策更有针对性。

---

## 10. Potential Extensions 与 Open Questions

1. **Sim-to-Real with ACSG**: 能否在 simulation（如 Habitat、Isaac Sim）中预训练 ACSG 构建 policy，再 transfer 到 real？ACSG 的 symbolic 性质可能有助于 sim-to-real alignment。

2. **ACSG + Diffusion Policy**: 用 ACSG 的 action node 作为 high-level plan，用 diffusion policy [27] 作为 low-level executor，可能突破 handcrafted primitives 的限制。

3. **Multi-Robot ACSG**: 多个机器人协同构建 ACSG，需要解决图融合、action 冲突、通信效率问题。

4. **ACSG Editing via Natural Language**: 让人类用自然语言修改 ACSG（"把 tape 移到上层抽屉"），机器人 incremental 更新图并验证。

5. **Causal ACSG**: 当前 $\mathbf{e}_{a \to a}$ 只表达 precondition，能否扩展为 causal graph（counterfactual reasoning）？例如"如果没移走 condiment，打开 cabinet 会失败"。

6. **Active ACSG Refinement**: 当 LMM 对某个 action 信心不足时，主动 query 人类或执行试探性 action 来 refine graph。

---

## References

[1] GroundingDINO: https://arxiv.org/abs/2303.05499  
[2] SAM-HQ: https://arxiv.org/abs/2306.01567  
[3] CLIP: https://arxiv.org/abs/2103.00020  
[4] ConceptFusion: https://arxiv.org/abs/2302.07241  
[5] OVIR-3D: https://proceedings.mlr.press/v229/lu23a.html  
[6] GPT-4V System Card: https://cdn.openai.com/papers/GPTV_System_Card.pdf  
[7] VoxPoser: https://arxiv.org/abs/2307.05973  
[8] VILA: https://arxiv.org/abs/2311.17842  
[9] Image Retrieval using Scene Graphs: https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf  
[10] 3D Scene Graph (Armeni et al.): https://openaccess.thecvf.com/content_ICCV_2019/papers/Armeni_3D_Scene_Graph_A_ICCV_2019_paper.pdf  
[11] ConceptGraphs: https://arxiv.org/abs/2309.16650  
[12] Spatial Ontologies: https://arxiv.org/abs/2312.11713  
[13] Frontier-based Exploration: https://ieeexplore.ieee.org/document/620822  
[14] Information-theoretic Planning: http://www.roboticsproceedings.org/rss11/p03.pdf  
[15] Curiosity-driven Exploration: https://arxiv.org/abs/1705.05363  
[16] Large-scale Curiosity: https://arxiv.org/abs/1808.04355  
[17] Learning to Poke: https://proceedings.neurips.cc/paper/2016/hash/38786c064d1a3e3a4c6b9c1f9c2c3f2-Abstract.html  
[18] Ditto: https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Ditto_Building_Digital_Twins_of_Articulated_Objects_From_Interaction_CVPR_2022_paper.pdf  
[19] Active Perception (Bajcsy): https://ieeexplore.ieee.org/document/99966  
[20] Next-Best-View Planner: https://ieeexplore.ieee.org/document/7487281  
[21] Affordance Landscapes: https://proceedings.neurips.cc/paper/2020/hash/0c5f11ffcd5c14c5b2d3ca3b3b3b3b3-Abstract.html  
[22] SayCan: https://arxiv.org/abs/2204.01691  
[23] Code as Policies: https://arxiv.org/abs/2209.07753  
[24] VoxPoser: https://arxiv.org/abs/2307.05973  
[25] VILA: https://arxiv.org/abs/2311.17842  
[26] PaLM-E: https://arxiv.org/abs/2303.03378  
[27] Diffusion Policy: https://arxiv.org/abs/2303.04137  
[28] RoboEXP Project Page: https://jianghanxiao.github.io/roboexp-web/

---

## 总结

RoboEXP 的核心贡献是把 **action 显式纳入 scene graph**，让机器人能 reasoning "做什么才能看到什么"。它巧妙地结合了 LMM 的 commonsense（local decision）和 symbolic graph 的 structural guarantee（global consistency），用 modular 闭环系统在 real-world tabletop 和 mobile 场景验证了 effectiveness。

从 Karpathy 你常强调的 "system 2 thinking" 视角看，ACSG 实际上是为机器人提供了一种 **externalized working memory**——把 LMM 的 implicit reasoning 外化为可遍历、可编辑、可验证的图结构。这种 neuro-symbolic 范式可能是通向 long-horizon robotic reasoning 的有希望路径，远比纯 LLM context-pumping 更 scalable。

主要瓶颈在 perception 的 open-vocabulary robustness 和 action primitive 的 generalization。如果未来用 learnable skill policies 替换 handcrafted primitives，并用 temporal fusion 增强 perception，这套框架有潜力成为 household robot 的标准 exploration backbone。
