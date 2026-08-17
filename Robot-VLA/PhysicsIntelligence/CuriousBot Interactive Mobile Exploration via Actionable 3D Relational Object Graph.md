---
source_pdf: CuriousBot Interactive Mobile Exploration via Actionable 3D Relational
  Object Graph.pdf
paper_sha256: 57e2f8c0778204f76e9ecae58a295657878acae96c9c546a2416d84d4666b982
processed_at: '2026-08-03T18:05:11-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CuriousBot 人话版

Andrej，咱换个方式聊。前面那版太 paper 腔了，这次我就当在 whiteboard 前跟你唠。

---

## 故事的起点

你想想家里的 Spot 机器人进了一个房间，想"摸清楚"这里有啥。传统做法（TARE、frontier-based exploration、Active Neural SLAM）就是让它绕一圈，把 occupancy grid 填满，unknown area 趋零就算完事。这套逻辑在空旷仓库里 OK。

问题是真实 household 里，一大堆东西你光靠"走一圈"根本看不到：
- 抽屉里塞着玩具（cabinet 把它遮住了）
- 椅子背后有个盒子（chair 挡住视线）
- 桌布下压着剪刀（cloth 覆盖）
- 翻倒的箱子里面有东西（box 闭合状态）

**这些 unknown 是物理遮挡，几何扫描解不了。必须动手。**

这就是 paper 的 motivation。前作 RoboEXP（Yunzhu Li 组里的工作，tabletop scale）已经证明 action-conditioned 3D scene graph 这个 idea 可行。CuriousBot 干的事就是把这套搬到 mobile robot 上——Spot 在房间里走来走去，建一个"带动作语义的关系图"。

---

## 核心直觉：让 representation 自己告诉你该干嘛

最关键的一句话：**robot 要决定下一步 explore 哪，需要一个能直接映射到 action 的 representation。**

你给 LLM 一堆 RGB 图像，说"探索吧"——它懵。因为 RGB 帧里没显式编码"这个抽屉我还没开过"这个信息，LLM 得靠记忆力去推理"哦我刚才看了一眼那个 cabinet 但没动手，现在该开它了"。长 horizon 下这种 implicit memory 必崩。

CuriousBot 的做法：robot 自己维护一个 graph $G = (V, E)$，每个 node 是物体（cabinet、handle、chair、cloth、box），每条 edge 是一种关系，关系一共 5 种：

| Relation | 含义 | 对应 action |
|---|---|---|
| `behind` | A 挡在 B 前面 | push A aside |
| `inside` | A 在 B 容器内 | open B |
| `on` | A 压在 B 上面 | lift A |
| `under` | A 在 B 下方被遮 | lift B or sit |
| `of` | A 是 B 的一部分（handle of cabinet） | open A 联动 B |

LLM 看到这个 graph 序列化后的文本，直接就能 reason："哦 cabinet_0 有 inside 关系但还没探索过，那下一步 open(cabinet_0)。"——决策逻辑就在 representation 里，不用 LLM 凭空推。

这跟 SayCan 的 affordance 思路是同源：**显式 ground action 进 representation，比让 LLM 从 raw observation 推 action 靠谱得多。**

---

## 四个 module 串起来怎么跑

```
RGBD + odometry → [SLAM] → camera pose T_t
                          ↓
              [Graph Constructor]
              ├─ YOLO-World 检测 2D box
              ├─ SAM 精修 mask
              ├─ back-project 到 3D point cloud
              ├─ 和历史 node 做 IoU association
              └─ rule-based 推 5 种 relation
                          ↓
                  [LLM Task Planner]
                  ├─ DFS 序列化 graph
                  └─ few-shot prompt → action plan
                          ↓
                  [Low-Level Skills]
                  open / flip / lift / push / sit / collect
                          ↓
                    执行 → 新 observation → 回到 SLAM
```

闭环。每执行一个 action，perception 更新，graph 增长，LLM 重新 plan。这跟 Inner Monologue、Code as Policies 的 closed-loop LLM planning 范式一脉相承。

---

## 唯一的公式其实就一个核心

整个 paper 数学层面很轻，最重要的就是 cross-frame data association 的 value matrix：

$$
C_{ij} = 
\begin{cases}
\text{IoU}(p_t^i, p_{t-1}^j), & \text{if same label} \\
0, & \text{otherwise}
\end{cases}
\tag{1}
$$

变量含义：
- $i \in \{1, \dots, K\}$：当前帧第 $i$ 个检测到的物体
- $j \in \{1, \dots, N\}$：历史 graph 里第 $j$ 个 node
- $p_t^i$：当前帧第 $i$ 个物体的 3D point cloud（YOLO-World + SAM 拿 mask，再用 depth + camera intrinsics 反投影）
- $p_{t-1}^j$：上一帧第 $j$ 个 node 对应的 point cloud
- $\text{IoU}(\cdot, \cdot)$：3D point cloud IoU（应该是 voxel-level，paper appendix 提了但主文没细说）
- $C \in \mathbb{R}^{K \times N}$：value matrix，行是当前检测，列是历史 node

匹配规则：对每个 $i$，找 $\arg\max_j C_{ij}$。若 max 低于 threshold → new node；否则 associate 到对应 node，做 point cloud fusion。

**为啥这公式重要？** 它直接决定 graph 会不会冗余。SLAM drift 一大，同一物体的 point cloud 在两帧里错位，IoU 掉下来，机器人就误判"这是新物体"，graph 越长越乱。paper 的 failure breakdown 里 perception failure 占大头，根因就是这。

要改进的话，ConceptGraphs 那一套用 CLIP/DINO feature embedding 做 association 会鲁棒得多，因为语义 embedding 对 pose drift 不敏感。但 CuriousBot 走了 simple route，工程上 YOLO-World label + 3D IoU 够用，跑得快。

---

## 5 种 relation 是怎么推出来的

paper 没给完整 pseudo-code，只能从描述 reverse-engineer。我推测的规则：

**`on`（A on B）**：A 的 bbox 底部 z 坐标 ≈ B 的 bbox 顶部 z 坐标（容差 2-3 cm），且 A 的 footprint 落在 B 的 footprint 内。几何上就是 vertical support。

**`under`（A under B）**：A 的顶部 ≈ B 的底部，A 在 B 的 footprint 投影内。是 `on` 的 reverse，但 action 不同——`on` 要 lift 上面的物体，`under` 要 lift 上面那个或者 sit down 看。

**`behind`（A behind B）**：从 camera 当前位置 ray-cast，B 的 point cloud 在 A 的 point cloud 后面被遮挡。这是个 view-dependent 关系，每帧重算。tabletop 不会有这种关系（因为相机环绕），是 mobile 特有的。

**`inside`（A inside B）**：A 的 centroid 落在 B 的 convex hull 内，B 是 articulated 物体（cabinet、fridge、drawer）。这里 articulated 的判断可能用 label（YOLO-World 直接给 cabinet label）+ 几何（B 是个 box shape with door）。

**`of`（A of B，partonomic）**：A 是 handle / knob 这类 part label，且 A 超过一半 point cloud 距离 B surface < 0.03 m。paper 明确给了这个 0.03m 阈值。

这 5 种覆盖了 household 大部分遮挡场景。比 RoboEXP 多了 mobile-specific 的 `behind`，比 ConceptGraphs 多了 actionable 的所有 5 种（ConceptGraphs 只编码 spatial proximity，没有"哪种 action 能改变这个关系"）。

---

## LLM 怎么"读"graph

graph 是 $G = (V, E)$ 这种结构化数据，LLM 直接吃不了。CuriousBot 用 **DFS 序列化**：从 root 出发，深度优先遍历，每个 node 输出 depth、name、index、ID、可执行的 action list。文本大概长这样：

```
ROOT (depth=0)
├─ cabinet_0 (depth=1) [actions: open]
│  ├─ handle_0 (depth=2) [of cabinet_0]
│  └─ [inside: unexplored]
├─ chair_0 (depth=1) [actions: push]
│  └─ [behind: unexplored]
└─ cloth_0 (depth=1) [actions: lift]
   └─ [under: unexplored]
```

LLM 看到这个，加上 7 个 few-shot example（每个 example 是 "graph → action sequence" 的 demonstration），就能输出下一步该干啥。

**为啥 7 个 example 是临界点？** Table II 的 ablation：example 从 7 减到 1，success 从 89% 掉到 11%，GED 从 0.33 飙到 2.67（8 倍恶化）。GED 恶化超线性说明 LLM 在 example 不足时不是单纯漏 action，而是开始 hallucinate 出 graph 里不存在的 edge——它"猜"关系，猜错。这个 ablation 间接证明 LLM 确实在 reason about graph 结构，单纯 perception 和 skill 没变，只改 prompt context 就这么大差距。

---

## 6 个 skill 的 engineering 细节

这块 paper 写得最 engineering-heavy，我挑重点说：

**`open`（开 articulated object）** 最 tricky：
1. PCA on handle point cloud → 第一主成分 = handle 长轴方向
2. PCA on cabinet surface → 最小特征值对应特征向量 = cabinet 法向（朝外）
3. End-effector 沿法向 approach handle，grasp
4. **Impedance control** 开门——这是关键。cabinet 的 hinge axis、range 都未知，硬 position control 会卡。用 impedance control：

$$
F = M_p(\ddot{x}_d - \ddot{x}) + K_p(x_d - x) + K_d(\dot{x}_d - \dot{x})
$$

- $M_p$：desired inertia（虚拟质量）
- $K_p$：desired stiffness（虚拟弹簧）
- $K_d$：desired damping（虚拟阻尼）
- $x_d, \dot{x}_d, \ddot{x}_d$：期望位置/速度/加速度（机器人命令）
- $x, \dot{x}, \ddot{x}$：实际位置/速度/加速度（传感器反馈）

door 的 hinge 限制让 end-effector 必须沿圆弧走，直线 $x_d$ 跟不上 → 误差 $(x_d - x)$ 变大 → force 增大但被 $K_p$ 限幅 → 机器人"顺从"物理 constraint，沿门自然轨迹走。这是处理未知 articulation 的经典 trick。

5. Grasp 失败 retry：force feedback 不达标就换 offset 重 grasp。

**`flip`**：假设 box 已 open（lid 打开状态）。Top-down approach，push box 一侧边缘让它倾覆。contact point 要在重心偏外侧，产生足够 torque。

**`lift`**：只用于 cloth（deformable）。Top-down grasp center of mass（point cloud centroid 估计）。cloth 因为 draping 可以被 gripper 捏住，rigid object 不行。

**`push`**：Spot 走到 object 前方，arm 伸直，body 横向移动。whole-body coordination，用 base 提供推力。

**`sit`**：Boston Dynamics Spot 内置 API。sit down 让 camera 高度降低，看 table 底下。这是 active perception 的子集——用 articulation 改变 viewpoint，不用 locomotion。

**`collect`**：off-the-shelf grasping planner（可能 GraspNet 或 Spot 内置）抓物体，place 到 robot 背上 blanket。用于 Fig. 1c 的 retrieve task。

---

## 实验结果的"人话翻译"

Table I 是核心，我直接说哪些数字有意思：

**Flipping boxes 任务：所有 baseline 0%，CuriousBot 80%**

这是最 dramatic 的对比。为啥所有 VLM 都 0%？VLM 看一张 RGB，没法判断 "这个 box 是 open 状态、可以从边缘 flip"。它在 2D 像素里看不出 box 的 lid 状态，更不知道 flip 这个 action 存在。Heuristics baseline 只会 "open all handles"，box 没 handle，也 0%。

CuriousBot 通过 graph 里 "box is open, no object inside detected yet" + flip skill 在 skill library 里，能 reason 出 "flip it to check underneath"。

**GPT-4o 在 lifting cloth 100% 但 underneath 0%**

非常 asymmetric。cloth lift 是 single-step：VLM 看 RGB "哦这是布，掀开它" → done。underneath 需要 multi-step："桌子底下可能有东西 → 但我看不到 → 我得 sit down → 但 sit 这个 action 不在我的 affordance 里因为我只看 RGB" → fail。

这印证了核心 thesis：**single-frame VLM 搞不定 multi-step interactive exploration**，需要 explicit memory + action library。

**Heuristics baseline 平均 12% 最差**

"open all handles" 这种盲目策略在 heterogeneous scene 直接崩。很多任务根本没 handle（flip box、push chair、lift cloth、sit under table）。说明 hand-crafted heuristics 不可扩展，必须有 reasoning。

**CuriousBot 平均 82%，GED 1.28**

GED 1.28 意味着 final graph 跟 ground truth 平均只差 1.28 个 edit operation——基本就是完美建图。对比 GPT-4o 的 3.32，差了一个数量级。

---

## Failure breakdown 的 18% 怎么分配

paper 没给精确百分比，但描述里 perception failure 占大头：
- **SLAM drift**：RTAB-Map 在长期运行 + interaction 后位姿漂移，导致 point cloud 错位，graph association 出错
- **Open-vocab detector 误检**：YOLO-World 把椅子背误认为 cabinet 之类

decision failure 占小部分：graph 对了但 LLM 选错 skill（可能 few-shot example 不够 cover 这个 case）。

action failure 也小部分：gripper 早 release、grasp 不牢、robot 和物体意外碰撞。

这指向改进路线：
1. 用 **3DGS-based SLAM**（如 SplaTAM、Photo-SLAM）替代 RTAB-Map，drift 小得多
2. 用 **Grounding DINO 或 OWLv2** 替代 YOLO-World，open-vocab 更准
3. 用 **diffusion policy** 替代 heuristic skills，robust to pose variation
4. 用 **graph transformer** 替代 DFS serialization，保留拓扑信息

---

## 这篇 paper 在 research landscape 里的位置

画个 family tree：

```
3D Scene Graph 谱系:
  Hydra (2022, structural graph)
    └─ CLIO (2024, task-driven)
       └─ ConceptGraphs (2024, open-vocab via CLIP)
          └─ RoboEXP (2024, action-conditioned, tabletop)
             └─ CuriousBot (this, mobile)
```

LLM planning 谱系：
```
SayCan (2022, affordance-grounded LLM)
  └─ Code as Policies (2023, LLM generates code)
     └─ VoxPoser (2023, LLM composes 3D value maps)
        └─ PaLM-E (2023, embodied multimodal)
           └─ RoboEXP / CuriousBot (closed-loop graph + LLM)
```

CuriousBot 的 unique combo：**explicit actionable 3D relational graph + mobile platform + closed-loop LLM planning**。三者交集之前没人做过。

---

## 我（作为读者）的 takeaway

如果让我用一句话给你总结这篇 paper 的 contribution：**它把 RoboEXP 的 action-conditioned scene graph idea 从 tabletop 推到 mobile，证明 explicit relational representation + LLM planner 在 mobile interactive exploration 上 work，且显著优于 VLM 直接看 RGB。**

更深一层的 takeaway：**LLM 在 robotics 里当 planner，需要 structured substrate 才能 work。** 给它 raw observation 让它 implicit memory + reasoning，长 horizon 必崩。给它 explicit graph、actionable edge、clear affordance，它就能稳定 plan。这个 pattern 在 SayCan、Code as Policies、VoxPoser、RoboEXP 里反复出现，CuriousBot 在 mobile scale 又验证了一遍。

未来方向我会盯着三件事：
1. **Skill learning**：把 6 个 heuristic skill 换成 diffusion policy 或 VLA（vision-language-action model），paper 自己也指了这个 limitation
2. **Better SLAM**：3DGS-based dense SLAM 解决 drift，让 graph association 鲁棒
3. **Graph-native LLM**：用 graph transformer 或 structured prompt 替代 DFS serialization，保留 multi-parent / cross-subtree 关系

如果顺着这条线往下读，我会推荐：
- **RoboEXP**：https://arxiv.org/abs/2402.15487 （前作，必读）
- **Hydra**：https://arxiv.org/abs/2201.13360 （scene graph foundation）
- **ConceptGraphs**：https://conceptgraphs.github.io/ （open-vocab graph alternative）
- **CLIO**：https://arxiv.org/abs/2404.13696 （task-driven scene graph）
- **SayCan**：https://arxiv.org/abs/2204.01691 （LLM + affordance 经典）
- **Code as Policies**：https://arxiv.org/abs/2209.07753 （LLM planner baseline）
- **VoxPoser**：https://arxiv.org/abs/2307.05973 （LLM composes 3D maps）
- **PaLM-E**：https://arxiv.org/abs/2303.03378 （embodied multimodal LLM）
- **OpenMask3D**：https://arxiv.org/abs/2306.13631 （open-vocab 3D segmentation）
- **YOLO-World**：https://arxiv.org/abs/2401.12270 （open-vocab detector）
- **SAM**：https://arxiv.org/abs/2304.02643 （segment anything）
- **RTAB-Map**：https://github.com/introlab/rtabmap （SLAM 库）
- **Pathak curiosity**：https://arxiv.org/abs/1705.05363 （curiosity-driven exploration 起源）
- **Affordance landscapes**：https://proceedings.neurips.cc/paper/2020/hash/da5ba6649d184f93e7d0c2c1c0533f7-Abstract.html （interaction exploration in 3D）

聊完了。这 paper 我觉得最值得记住的就一件事：**explicit structured memory beats implicit VLM memory，在 long-horizon interactive robotics 上尤其明显。** 这跟你过去几年反复强调的 "let LLM reason over structured substrate" 思路完全一致。

---

# CuriousBot 深度技术讲解

Andrej 你好，这篇 paper 让我想起你之前在 Eureka Labs 讲 embodied AI 的语境——把 perception, reasoning, action 三个 modality 用一种 explicit representation 黏合起来。这篇工作做的恰恰是这件事：用 **3D relational object graph** 作为 LLM 的 "memory and grounding substrate"，让 Spot 这种 quadruped 在 household environment 里做 **active interaction exploration**。下面我会拆得很细，公式里每个上下标都讲清楚，再串到 broader research landscape。

参考链接先行：
- CuriousBot 项目页面（推测）：https://curiousbot.github.io/ （作者团队 MIT-SPARK + Yunzhu Li）
- RoboEXP（前作）：https://arxiv.org/abs/2402.15487
- YOLO-World：https://arxiv.org/abs/2401.12270
- SAM：https://arxiv.org/abs/2304.02643
- RTAB-Map：https://github.com/introlab/rtabmap
- Hydra（3D scene graph 经典工作）：https://arxiv.org/abs/2201.13360
- CLIO（task-driven scene graph）：https://arxiv.org/abs/2404.13696
- ConceptGraphs：https://conceptgraphs.github.io/
- SayCan：https://arxiv.org/abs/2204.01691
- Code as Policies：https://arxiv.org/abs/2209.07753

---

## 1. 论文核心 intuition（先讲 why，再讲 how）

传统 mobile exploration 的工作（TARE [1]、frontier-based exploration [36]、Active Neural SLAM [22]、VLFM [25]）几乎全都在做一件事：**active perception**——找一个 best next viewpoint 来 minimize unknown area。这背后的隐含假设是 "未知空间 = 几何上未观测的空间"，相机扫一遍就能消除。

CuriousBot 想要说的是：**很多未知空间是物理上被遮挡的**——cabinet 抽屉里的玩具、椅子背后的物体、桌布下的剪刀、箱子里的物体。这类 unknown 没法靠移动相机解决，必须靠 **articulation + non-prehensile interaction** 把遮挡物移走。这就是 "active interaction" 的 motivation。

更进一步的直觉：当 robot 决定要不要 push 椅子、open cabinet、lift cloth，它需要的不是一个 occupancy grid，而是一个 **relational, actionable** 的 representation：
- relational：椅子挡住了什么，cabinet 包含了什么，cloth 覆盖了什么
- actionable：每种关系对应一组 affordance / skill（behind → push，inside → open，under → lift/sit）

这是这篇 paper 区别于 RoboEXP 的关键。RoboEXP 是 tabletop，object 关系相对简单、相机视野能 cover。CuriousBot 把这件事 push 到 mobile scale，加了 navigation 维度、加了 articulated/deformable objects、加了更大的 action space。

---

## 2. System Architecture 详解

整条 pipeline 是 4 个 module，串成一个 perception→representation→planning→control 的 loop：

```
RGBD stream O_t + Odometry
        │
        ▼
    [SLAM]  ──►  T_t (camera pose), M_t (map)
        │
        ▼
[Graph Constructor]  ──► G_t = (V_t, E_t)
   ├─ YOLO-World (open-vocab 2D boxes)
   ├─ SAM (mask refinement)
   ├─ 3D back-projection → P_t
   ├─ Cross-frame association via IoU value matrix C
   └─ Rule-based relation inference (5 relations)
        │
        ▼
 [Task Planner]  ──► action plan
   ├─ DFS serialization of G_t
   └─ LLM (prompted with skill library + examples)
        │
        ▼
[Low-Level Skills]  ──► Spot + arm execution
   open / flip / lift / push / sit / collect
        │
        ▼ (new observation)
    回到 SLAM ...
```

这种 4-module 拆分是 robotics foundation model 时代一种很流行的 recipe：**把 VFM/LLM 用作 perception and reasoning plugin，把传统 robotics stack（SLAM、control、skill primitives）当作 grounding substrate**。类似的还有 SayCan、Code as Policies、VoxPoser、RoboEXP。

---

## 3. 数学定义与公式逐项解释

### 3.1 Object graph

$$
G = (V, E), \quad V = \{\nu^0, \nu^1, \dots, \nu^N\}, \quad E = \{e^0, e^1, \dots, e^M\}
$$

- $V$：node 集合，上标 $i \in \{0, \dots, N\}$ 是 node index
- 每个 $\nu^i$ 携带两类 attribute：
  - semantic：object label（"cabinet", "handle", "cloth", ...）
  - geometric：point cloud + normal estimate（normal 用来判断 contact surface 朝向）
- $E$：edge 集合，每个 $e = (\nu^i \to \nu^j, r)$ 是一条 directed edge，$r \in \{\text{behind, of, inside, on, under}\}$ 是 relation label

**Intuition**：用 directed graph 而不是 undirected 是因为 "behind" 是 non-symmetric（A 在 B 后面 ≠ B 在 A 后面），"inside" 也是；"on/under" 互为 reverse；"of"（part-of，比如 handle is part of cabinet）也是从 child part 指向 parent whole。

### 3.2 SLAM 输入输出

$$
\text{SLAM}: \{O_0, O_1, \dots, O_t\} \mapsto (T_t, M_t)
$$

- $O_t \in \mathbb{R}^{H \times W \times 4}$：RGBD 帧，4 通道 = RGB (3) + Depth (1)
- $T_t \in SE(3)$：camera pose（rigid transform）
- $M_t$：current map（这里其实就是用来给 graph constructor 提供 spatial context，没有显式公式）

工程上用 RTAB-Map [85]，RGBD SLAM + loop closure。Spot 自带 odometry 给 SLAM 做 prior。

### 3.3 Cross-frame data association（核心公式）

设第 $t$ 帧 YOLO-World + SAM 给出 $K$ 个 mask，每个 mask back-project 到 3D 得到 point cloud $p_t^i$。前帧 graph 有 $N$ 个 nodes，每个对应 point cloud $p_{t-1}^j$。我们构造一个 value matrix：

$$
C \in \mathbb{R}^{K \times N}, \quad
C_{ij} = 
\begin{cases}
\text{IoU}(p_t^i, p_{t-1}^j), & \text{if they share the same label} \\
0, & \text{otherwise}
\end{cases}
\tag{1}
$$

- $i \in \{1, \dots, K\}$：当前帧 detection index
- $j \in \{1, \dots, N\}$：历史 node index
- $\text{IoU}(\cdot, \cdot)$：3D point cloud IoU（应该是 voxel-level IoU，paper 里 appendix 提了 definition）

匹配规则：
- 对每个 $i$，找 $\arg\max_{j} C_{ij}$
- 若 $\max_j C_{ij} < \tau$（threshold）→ 标为 **new node**
- 否则 → associate $p_t^i$ 到 node $j$，做 point cloud fusion（incremental update）

**Intuition**：label consistency 是 hard gate，IoU 是 soft score。这本质上是 multiple object tracking 里的 "label + geometry" data association，只是从 2D IoU 换成 3D IoU。它在长期运行里会遇到 drift 问题——SLAM pose drift 让同一物体的 3D point cloud 在不同 frame 错位，IoU 掉下来 → 误判为 new node → graph 冗余。这也是 paper 后面 "perception failure" 中"imprecise SLAM"那条 failure mode 的来源。

更好的做法（CuriousBot 没做但值得联想）：用 **semantic feature embedding**（CLIP/DINO feature）做 association，加上一个 Hungarian matching 而不是 greedy argmax。ConceptGraphs [74] 就是这么做的。也可以学 RoboEXP 那样用一个 action-conditioned update（每次 interaction 后强制 re-associate）。

### 3.4 Object Recovery 与 Graph Edit Distance

$$
\text{OR} = \frac{|V_{gt} \cap V|}{|V_{gt}|}
$$

- $V_{gt}$：ground truth node set（人在 episode 结束后 manual annotation）
- $V$：robot 发现的 node set
- 分子是 intersection（同一物体的 node），分母是 ground truth 总数
- 类似 recall，不惩罚 false positive

$$
\text{GED}(G, G_{gt}) = \text{cost to edit } G \text{ into } G_{gt}
$$

- add / delete / move 一个 edge 或 node 都算 cost = 1
- GED 是经典 graph matching metric，NP-hard 但 paper 里 graph 足够小所以 brute force / A* 都能跑
- 既惩罚 missing nodes/edges（欠探索），也惩罚 spurious edges（错误关联），还惩罚 edge endpoint 错配（relation 接到错误 node 上）

这俩 metric 一起用很关键：OR 衡量 "你看到多少东西"，GED 衡量 "你的关系理解有多准确"。一个 robot 可以 OR 高但 GED 也高（看到了物体但关系搞错），这是 RoboEXP、ConceptGraphs 都没区分的，CuriousBot 在 metric 上比之前工作更细。

---

## 4. Graph Constructor 详解（最重要的 module）

### 4.1 Perception pipeline

1. **YOLO-World**：open-vocabulary 2D detector，给 prompt like "cabinet, handle, chair, cloth, box, table, ...，输出 bounding boxes。
2. **SAM**：以这些 box 为 prompt，输出 pixel-accurate mask。
3. **3D back-projection**：用 depth $D_t$ + camera intrinsics $K_{cam}$ 把每个 mask 反投影成 3D point cloud：
   $$
   p = K_{cam}^{-1} \cdot [u \cdot d, v \cdot d, d]^T
   $$
   再用 $T_t$ transform 到 world frame。
4. **Point cloud fusion**：和已存在 node 的 point cloud 做 voxel-level union / running average，更新 node 的 geometric attribute。

### 4.2 Relation rule 推断（这是 actionable 的关键）

Paper 没给完整 pseudo-code，但从描述里可以 reverse-engineer 5 类规则：

| Relation | 触发条件（推测） | Action 选项 |
|---|---|---|
| `on` | object A 的底部 bbox 接触 object B 的顶部 bbox，且 A 体积 < B | lift A |
| `under` | object A 的顶部 bbox 接触 object B 的底部 bbox | lift B（or sit if B is table） |
| `behind` | 从 camera 视角，A 被 B 在 ray-cast 中挡住 | push B aside |
| `inside` | A 的 centroid 落在 B（cabinet/fridge）的 convex hull 内，且 B 是 articulated | open B |
| `of` | A 的部分点云（>X%）在 B 的 0.03m 内，且 A 是 handle/knob 类 | open A（连带 B） |

举例：handle is part of cabinet，是因为 handle 50% 以上 point cloud 距离 cabinet surface < 0.03m，且 label 是 "handle"。这种 rule 类似 Hydra [54] 的 "object → element → room" 层次化建图，只是 CuriousBot 把 relation 设计成 "action-friendly"。

### 4.3 5 种 relation 的 intuition

- `behind`：空间遮挡（x-y 平面 + depth 维度同时考虑），解法是 lateral push
- `of`：part-whole（partonomic），handle 属于 cabinet，handle 是 manipulation interface
- `inside`：container 关系，需要 articulation（joint angle estimation）
- `on`：vertical support，物体可能在上面，需要 lift
- `under`：horizontal occlusion，需要 lift or sit-down view

这 5 种是 household 场景的高频 relation，比 RoboEXP 多了 mobile 特有的 `behind`（tabletop 没有 behind，因为相机环绕），也多了 partonomic 的 `of`。

---

## 5. Task Planner：从 graph 到 action sequence

### 5.1 Serialization via DFS

把 graph $G$ 序列化成 LLM 可读的 text：

```
ROOT
├── cabinet_0 [open]
│   ├── handle_0 [of cabinet_0]
│   └── (inside, unexplored)
├── chair_0 [behind: unexplored space]
└── cloth_0 [under: unexplored]
```

DFS 遍历，每个 node 输出：
- depth（在 graph 里的层数）
- object name + index
- node ID
- related action details（哪些 skill 对这个 node 可用）

**Intuition**：LLM 不擅长直接消化 graph 结构（attention 没有拓扑先验），但擅长消化 nested text。DFS 嵌套结构是最接近 graph 的 1D 表达。ConceptGraphs 和 SceneGPT [75] 也用过类似 trick。一个潜在改进是直接用 graph neural network 编码，然后 cross-attention 进 LLM token space，类似 GraphGPT 或 GRAM。

### 5.2 LLM prompt 结构

包含三部分：
1. **System prompt**：描述任务（"你是一个探索机器人，目标是 reduce unknown space by interacting with objects"）
2. **Skill library description**：每个 skill 的 name + precondition + effect（"open(target_idx): opens an articulated object, revealing its inside"）
3. **Serialized graph + few-shot examples**：paper 用 7 examples（ablation study 从 7 → 5 → 3 → 1）

输出是 action plan，格式类似：

```json
{"skill": "push", "target": "chair_0"}
{"skill": "open", "target": "cabinet_0"}
{"skill": "sit"}
```

LLM 输出后逐个 execute，每执行一次回到 perception 重新更新 graph，再 plan。这是经典的 **closed-loop LLM planning**。

### 5.3 Few-shot example 的数量学

Table II 的 ablation：

| # Examples | Success | OR | GED |
|---|---|---|---|
| 7 (full) | 89% | 89% | 0.33 |
| 5 | 67% | 67% | 0.89 |
| 3 | 56% | 56% | 1.00 |
| 1 | 11% | 11% | 2.67 |

Success 和 OR 完全一致说明 **任务完成 = 找到所有物体**，没有"完成但找漏"的情况。GED 随 examples 减少超线性上升（1→2.67 是 8 倍），说明 LLM 在 examples 不足时不仅是漏 action，还会产生 **错误 relation 推断**——这是很 reasonable 的：few-shot 不足时 LLM hallucinate 出 graph 中不存在的 edge。

这个 ablation 也间接说明 LLM 在 task planner 位置上确实在 "reason about graph"，因为 perception 和 skill 都没变，唯一变量是 prompt context。

---

## 6. Low-Level Skills 技术细节

这是 paper 里最 engineering-heavy 的部分，每个 skill 都有 heuristic 设计：

### 6.1 `open`（articulated object）

1. 检测 handle node 和 cabinet node
2. **PCA on handle point cloud**：取第一主成分作为 handle axis（长条形的把手第一主成分就是其 length direction）
3. **PCA on cabinet surface normal**：取最小特征值对应的特征向量作为 cabinet 的 normal（朝外）
4. End-effector approach handle，沿 normal 方向 grasp
5. **Impedance control** 开门：因为 cabinet 的 articulation（hinge axis, range）未知，强行 position control 会卡住。用 impedance control（compliance），end-effector 跟随 handle 物理运动轨迹，同时根据 force feedback 调整 target position/orientation
6. **Grasping failure retry**：若 grasp force feedback 不达标，re-approach with different offsets

Impedance control 的标准形式：

$$
F = M_p (\ddot{x}_d - \ddot{x}) + K_p (x_d - x) + K_d (\dot{x}_d - \dot{x})
$$

- $M_p, K_p, K_d$：desired inertia, stiffness, damping
- $x_d, \dot{x}_d, \ddot{x}_d$：desired position/velocity/acceleration
- 当 door hinge 限制 end-effector 沿圆弧运动，end-effector 实际 $x$ 会偏离 $x_d$，但 impedance control 让 force 维持在安全范围，机器人"顺从"物理 constraint

### 6.2 `flip`

- 假设：box 已经 open（lid open 状态）
- Top-down approach，push box 的一侧边缘使其倾覆
- 关键是 contact point 选择：要在 box 重心偏外侧，才能产生足够 torque 让它翻

### 6.3 `lift`（仅 cloth）

- Top-down grasp center of mass（com 通过 point cloud centroid 估计）
- 只用于 cloth 这种 deformable，因为 rigid object lift 需要 stable grasp，cloth 由于 draping 可以被 gripper "捏住"

### 6.4 `push`（large rigid object）

- Spot 走到 object 前方
- Arm 伸直，body 横向移动（Whole-body coordination）
- 利用 Spot 的 base 提供推力，arm 维持接触

### 6.5 `sit`

- Boston Dynamics Spot 内置 API
- Robot sit down 让 camera 高度降低，看 table 底下
- 这是 **camera repositioning without manipulation**——active perception 的子集，但是用 articulation 而不是 locomotion 来实现

### 6.6 `collect`

- 用 off-the-shelf grasping planner（可能 GraspNet 或 Spot 内置）
- Grasp 后 place 到 robot back 上的 blanket
- 用于 "retrieve" task（Fig. 1c）

---

## 7. 实验数据深读

### 7.1 Setup

- Robot: Spot + 外挂 RealSense D455（前置）
- Compute: RTX A6000 + AMD 128GB（足够跑 YOLO-World + SAM + LLM）
- Scene: 3m × 4m room
- Tasks: 5 类，每类 10 trials，不同 initial condition

### 7.2 主结果（Table I）

| Method | Flipping | Opening | Underneath | Pushing | Lifting | Avg Success |
|---|---|---|---|---|---|---|
| LLaVa | 0% | 40% | 60% | 0% | 10% | 22% |
| Gemini | 0% | 80% | 40% | 0% | 40% | 32% |
| GPT-4o | 0% | 60% | 0% | 0% | 100% | 32% |
| Heuristics | 0% | 60% | 0% | 0% | 0% | 12% |
| **Ours** | **80%** | **80%** | **90%** | **70%** | **90%** | **82%** |

几个关键观察：

1. **Flipping boxes 全 baseline 都 0%**：因为 VLM 只看 current RGB，无法判断 box 是否 open、不知道 push 哪里才能翻；Heuristics 只会 open handles，box 没 handle。CuriousBot 通过 graph 里 "box is open, no object inside" 这种状态判断 + flip skill 解决。

2. **GPT-4o 在 lifting cloth 100% 但 underneath 0%**：很 asymmetric。cloth lift 是 single-step direct action，VLM 能从 RGB 看出来；underneath 需要先 reasoning "table 底下可能有东西，需要 sit down 查看"，这是 multi-step reasoning，VLM 单帧输入做不到。

3. **Heuristics 12% 最差**：说明 "open all handles" 这种盲目策略在 heterogeneous scene 失败——很多任务根本没有 handle。

4. **GED**：CuriousBot 平均 1.28，baselines 3+。GED 是 GED 越低 graph 越接近 ground truth。这里 CuriousBot 不仅找物体准，关系也对。

### 7.3 Failure breakdown（Fig. 5）

18% 失败分三类：
- **Perception failure**：SLAM drift 导致 graph 不准 + open-vocab detector 误检
- **Decision failure**：graph 对了，LLM 选错 skill
- **Action failure**：plan 对了，execution 出错（gripper 早 release、grasp 不牢、robot-object interference）

paper 没给三类 failure 的具体百分比，从描述看 perception 占大头（SLAM + detector 都不稳定）。这指向一个明确改进方向：**用更鲁棒的 dense SLAM（如 NICE-SLAM、Gaussian Splatting SLAM）替代 RTAB-Map**，或者用 foundation model-based detector（Grounding DINO、OWLv2）替代 YOLO-World。

### 7.4 与 RoboEXP 的隐性对比

RoboEXP 的 tabletop success rate（paper [3]）大概 70% 多，CuriousBot 82% 看似更高，但场景不同。RoboEXP 的 challenges 是 fine-grained manipulation（小物体多、关系密集），CuriousBot 是 mobile scale（spatial 较稀疏但 navigation 难）。两者其实互补，一个未来方向是把 CuriousBot 的 mobile layer 加上 RoboEXP 的 tabletop fine manipulation。

---

## 8. Broader context 与我（作为读者）的联想

### 8.1 与 3D Scene Graph 谱系的关系

CuriousBot 在一个很长的 lineage 里：
- **Hydra** [54]：最早把 3D scene graph 做成 real-time，但 relation 主要是 structural（room → object → element）
- **CLIO** [55]：task-driven open-set scene graph，强调 dynamic subset
- **ConceptGraphs** [74]：open-vocabulary 3D graph，用 CLIP feature 做 association
- **OpenMask3D** [69]、**OpenScene** [63]：open-vocabulary 3D segmentation
- **RoboEXP** [3]：第一个 action-conditioned scene graph
- **CuriousBot**：把 RoboEXP 的 idea 移植到 mobile，并加 navigation layer

CuriousBot 相对 ConceptGraphs 的优势是 **actionable**：每条 edge 都映射到一个 skill。ConceptGraphs 是 pure perception，没有 action affordance。

### 8.2 与 LLM planning 谱系

LLM-as-planner 这条线：SayCan [78] → Code as Policies [80] → VoxPoser [76] → PaLM-E [77] → Inner Monologue [79] → RoboEXP → CuriousBot。

CuriousBot 的特点：
- 不是直接 "LLM outputs action"，而是 "LLM outputs (skill, target)"，skill 是 parameterized primitive
- 这是 SayCan 风格（skill 是 high-level action），不是 RT-2 风格（skill 是 low-level token）
- 优点：skill 有 ground truth semantics，可解释；缺点：skill library 需要手工设计（paper 在 conclusion 里也承认）

### 8.3 与 Active Learning / Curiosity 的联系

paper 标题叫 CuriousBot，但内部其实不是经典 curiosity-driven（ICM [49]、RND [50]、disagreement-based）。它是 **task-driven exploration**：目标是 reduce unknown space，"curiosity" 体现在 robot 主动选择哪个 occluded region 去 explore。这更像 active perception 里的 next-best-view，只是 view 换成 interaction。

真正的 curiosity 版本会是：robot 内部一个 predictor $f_\phi$ 预测 "如果 open cabinet 会看到什么"，prediction error 高 → 高 curiosity → 去 open。这种 model-based active interaction 是 open research direction，paper 没碰但很值得做。Affordance landscape [51] 那篇 NeurIPS 2020 是相关参考。

### 8.4 Skill acquisition 的 open problem

paper 自己指出：当前 skills 都是 roboticist 手工设计 + 调参。下一步应该是：
- **Learning from video**（如 RT-2, Vid2Robot, π0）
- **Diffusion policy**（如 Diffusion Policy, 3D Diffusion Policy）
- **VLM-as-skill-generator**（如 Eureka, Code as Policies 自动生成 reward/code）

特别地，把每个 skill 换成 diffusion policy，让 VLM 做 high-level planning、diffusion 做 low-level control，是当前 obvious path（参考 RT-2、OpenVLA、π0）。

### 8.5 Limitation：serialization 丢信息

DFS serialization 把 graph 压成 1D text，丢失了：
- Multi-parent relations（一个 node 可能 inside 一个、on 另一个）
- Cross-subtree comparisons（两个 cabinet 哪个更值得先 open）

更优做法是用 graph transformer 直接 encode graph，或用 structured prompt（如 JSON with explicit edges）保留拓扑。

---

## 9. 我会给 Andrej 的 TL;DR

如果让我一句话总结给你：**CuriousBot 是 RoboEXP 的 mobile version**，把 action-conditioned 3D scene graph 这套 idea 从 tabletop 推到 Spot 上，靠 4 个 module（SLAM + Graph + LLM planner + Heuristic skills）的 tightly-coupled engineering。最大的 takeaway 是 **显式 relational representation 比 VLM 看一串 RGB 更适合 long-horizon interactive exploration**——这是 LLM agent 在 robotics 里的一个 recurring 信号：**explicit structured memory beats implicit VLM memory**。

如果让我推荐下一篇要看的：**RoboEXP** [3] 是必读的前作，**Hydra** [54] 是 scene graph foundation，**Code as Policies** [80] 是 LLM planner 的 baseline，**ConceptGraphs** [74] 是 open-vocab graph 的 alternative。如果要 follow-up 改进，我会盯着三件事：（1）用 3DGS-based SLAM 替 RTAB-Map 解决 drift；（2）用 diffusion policy 替 heuristic skills；（3）用 graph transformer 替 DFS serialization。

参考链接汇总：
- RoboEXP：https://arxiv.org/abs/2402.15487
- Hydra：https://arxiv.org/abs/2201.13360
- CLIO：https://arxiv.org/abs/2404.13696
- ConceptGraphs：https://conceptgraphs.github.io/
- YOLO-World：https://github.com/AILab-CVC/YOLO-World
- SAM：https://github.com/facebookresearch/segment-anything
- RTAB-Map：https://github.com/introlab/rtabmap
- SayCan：https://arxiv.org/abs/2204.01691
- Code as Policies：https://arxiv.org/abs/2209.07753
- VoxPoser：https://arxiv.org/abs/2307.05973
- PaLM-E：https://arxiv.org/abs/2303.03378
- Curiosity-driven exploration (Pathak)：https://arxiv.org/abs/1705.05363
- Affordance landscapes (Nagarajan & Grauman)：https://proceedings.neurips.cc/paper/2020/hash/da5ba6649d184f93e7d0c2c1c0533f7-Abstract.html
- OpenMask3D：https://arxiv.org/abs/2306.13631
- OpenScene：https://arxiv.org/abs/2212.08851
