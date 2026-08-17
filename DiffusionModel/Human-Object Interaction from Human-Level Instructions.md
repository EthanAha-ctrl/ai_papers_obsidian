---
source_pdf: Human-Object Interaction from Human-Level Instructions.pdf
paper_sha256: abc21ebd6b0aa5a402522c8e8a92c823a55a2181da62339c2e9ea8ffc8212c49
processed_at: '2026-08-05T07:48:25-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个讲法，像在 whiteboard 前跟你聊天那样讲。

---

## 这 paper 到底在干嘛

一句话：**给它一句话"帮我把 workspace 搭起来"，它就能生成一个 virtual human 在 3D 场景里走过去、搬桌子、摆显示器、放椅子，全程带手指细节，还物理可信**。

就这么个事。听起来简单，实际上每一步都踩在前面一堆工作的痛点上。

---

## 为什么这件事难

你想搭这样一个系统，最直觉的做法就是 end-to-end：喂个 language instruction 进去，出来 motion。但这条路走不通，三个硬 blocker：

**Blocker 1：LLM 给不出精确坐标。** 你问 GPT-4o"把 monitor 放 table 上面对 chair"，它懂意思，但你让它直接输出 `(x=1.2, y=0.8, z=0.75, qx=0.1, ...)`，数字乱飘。LLM 对离散符号推理很强，对连续几何数值很烂。这是 known issue，3D-LLM [16]、Holodeck [65] 都撞过这堵墙。

**Blocker 2：没有合适的数据集。** 你要训一个 model 同时输出 full-body + fingers + object motion，需要配对数据。但现实是：
- FullBodyManipulation [26] 有 full-body + 大物体 manipulation，没手指
- GRAB [46] 有手指，但只捏小东西，人不走动
- 两个 dataset 合不到一起

你没法 supervised 训一个"三合一"model，因为 label 不存在。

**Blocker 3：Kinematic motion 不物理。** Diffusion model 生成 motion 一定会有 foot sliding、hand 穿进物体、物体悬空这类 artifact。在 graphics 里这是 dealbreaker——看起来就假。

这三个 blocker 就决定了整个系统的三段式结构。

---

## 系统怎么拆

```
Instruction → [LLM Planner] → scene map + plan
                                ↓
              [Diffusion Generator] → kinematic motion
                                ↓
                [Physics RL Tracker] → physically plausible motion
```

每段对应一个 blocker。下面我讲每段里我觉得最 clever 的 design。

---

## LLM Planner：让 LLM 只做它擅长的事

最关键 idea：**别让 LLM 输出坐标，让它输出关系**。

他们定义了三个 symbol：
- `on(A, B)`：A 在 B 上面
- `adjacent(A, B, direction, distance)`：A 在 B 东/西/南/北多少米
- `facing(A, B)`：A 朝向 B

LLM 只负责推理"哪个物体该 on 哪个、哪个该 adjacent 哪个"，具体 3D 坐标由一个 algorithm 算出来。

这个 algorithm 本质上是**带几何 offset 的 topological sort**。伪代码在 supplementary Algorithm 1：

```
V_P ← 静态节点（如 floor）
while 还有节点没处理:
    找一个节点 v，它的所有前驱都已处理
    L(v) = Average over 前驱 u: L(u) + ComputeOffset(u, v)
```

`ComputeOffset` 看 edge 类型：
- `on` → 高度对齐前驱顶面，水平位置 sample（保留随机性，不至于每次都摆正中）
- `adjacent` → 直接用 LLM 给的 direction + distance

`facing(A, B)` 单独处理 orientation：把 A 的 canonical front direction 旋到指向 B。

**为什么这 design 好**：LLM 的 discrete reasoning 能力被充分利用，geometric calculation 这种 LLM 不擅长的活儿被剥离到 algorithm 里。这是个**符号-数值解耦**的经典 pattern——在 robotics 里 task and motion planning (TAMP) 也是这思路，先 symbolic plan 再 continuous refine。

他们对比了 baseline（让 LLM 直接输出 3D pose）：

| Method | Position Error | Orientation Error |
|---|---|---|
| Baseline | 21.9% | 12.5% |
| Ours | 3.1% | 1.6% |

差一个数量级。Human study 也压倒性 prefer Ours。

**吐槽**：relation 集合是手定的，处理不了"靠墙但别正对门"这种隐式 constraint。对任意 CAD 模型还要预先标 canonical direction。但作为 proof-of-concept，这个 abstraction 漂亮。

---

## Diffusion Generator：分而治之解决数据缺失

这是 paper 最核心技术。没有"三合一"数据，怎么办？**拆成四阶段 pipeline，每阶段用合适的数据训，靠中间 condition 传递信息**。

### Stage 1: CoarseNet

用 CHOIS [25]（同组前作）生成 full-body + object motion，**没手指**。

输出 $\mathbf{x} = \{\mathbf{H}, \mathbf{O}, \mathbf{L}\}$：
- $\mathbf{H} \in \mathbb{R}^{T \times D}$：human motion，每帧是 global joint positions + 6D rotations（6D rotation [74] 是 rotation matrix 前两行，避免 quaternion 的 double cover 不连续问题）
- $\mathbf{O} \in \mathbb{R}^{T \times 12}$：object motion，3 位置 + 9 rotation matrix
- $\mathbf{L} \in \mathbb{R}^2$：左右手 contact label

Condition $c = \{S, G, T\}$：
- $S$：masked motion——首帧填 human+object pose，末帧填 object pose，每 30 帧填 2D waypoint，其余 padding 0
- $G$：object geometry，用 BPS [42] 编码（把 mesh 投影到 basis point 上的距离向量）
- $T$：CLIP text embedding

生成完根据 $\mathbf{L}$ 把 motion 切 pre-contact / contact / post-contact 三段，取 contact 段平均 wrist pose $w$ 喂给下一阶段。

### Stage 2: Grasp Pose Generation

用 DexGraspNet [53] 优化出一个 grasp pose $\hat{g} = \Omega(\hat{w}, \hat{\theta})$。这原本是 robotic grasp 方法，但只用 kinematic chain 信息，所以跟 SMPL-X [37] 人体手 model 兼容。

优化目标（paper 没显式写，但 DexGraspNet 原文是这结构）：

$$E = E_{\text{force closure}} + \lambda_1 E_{\text{surface}} + \lambda_2 E_{\text{natural}}$$

- $E_{\text{force closure}}$：grasp 能否抵抗外力（friction cone 分析）
- $E_{\text{surface}}$：手指贴近物体表面
- $E_{\text{natural}}$：手型自然

两个改动：
1. **采样点 2000 → 20000**：DexGraspNet 原本处理小物体，对大物体采样太稀，手指会穿模或浮空
2. **双手任务去掉 force closure 项**：搬箱子靠双手配合，硬逼单手 force closure 会出反物理姿势

### Stage 3: RefineNet

CoarseNet 的 wrist pose 跟 Stage 2 优化的 $\hat{w}$ 对不上，硬拼会有 artifact。RefineNet 是基于 CoarseNet 再加 condition 的 diffusion。

这里有个**我觉得全 paper 最 clever 的 loss**——Wrist-Object Relative Pose Loss。

问题：你想约束"wrist 跟 object 的相对关系"对，但 object 在动，直接约束 absolute pose 没意义。怎么表达"相对关系"？

做法：在 object rest pose 表面 uniform sample 100 个点 $\mathbf{K}_{\text{rest}} \in \mathbb{R}^{100 \times 3}$。预计算这些点在 ground truth wrist frame 下的位置 $\mathbf{K}_w$。推理时每帧把 rest 点经 object 变换到 global，再经 wrist 变换到 wrist local frame：

公式 (2) $\mathbf{K}_{\text{global}} = R_o \mathbf{K}_{\text{rest}} + T_o$
- $R_o$：predicted object rotation
- $T_o$：predicted object translation

公式 (3) $\hat{\mathbf{K}}_w = R_w^{-1}(\mathbf{K}_{\text{global}} - T_w)$
- $R_w, T_w$：predicted wrist rotation / translation

公式 (4) loss：

$$\mathcal{L}_{\text{relative}} = \sum_{t=1}^{T} \mathbf{L}_t \| \hat{\mathbf{K}}_{w,t} - \mathbf{K}_{w,t} \|_1$$

- $\mathbf{L}_t$：contact label，非接触帧 mask 掉

**直觉**：这个 loss 说"无论 object 怎么动，你 wrist 在 object local frame 里看到的 object 表面点分布要一致"。用 point cloud alignment 表达 contact，比 binary contact label 连续可微，比直接约束 absolute pose 又对齐了 moving object 的情况。

这个 idea 可以单独抽出来用——任何"两刚体保持固定相对关系"的 task 都能用这个 loss。

### Stage 4: FingerNet

Contact 阶段保持 grasp pose 不变，pre/post-contact 阶段用 FingerNet 生成 finger motion，在 GRAB [46] 上训，只用 grasp 前后各 1 秒数据。

输出 $\mathbf{F} \in \mathbb{R}^{T \times D'}$，每帧手指 local 6D rotation。

Condition $c_f = \{P, F_s, F_e\}$：
- $P \in \mathbb{R}^{T \times 100}$：100 个 palm-side mesh vertex 到 object 表面的最近距离（proximity feature，算时手指设 mean pose 避免 self-effect）
- $F_s, F_e$：start (rest) / end (grasp) finger pose

Mirror 策略处理双手：训练时 left hand 数据 mirror 到 right，推理时 mirror input → predict → mirror back。跟 ManipNet [70] 一样。

### Navigation Module

距离物体 >1m 用 navigation diffusion（在 HumanML3D [12] 上训），<1m 切到 interaction。切换时上一模块末帧作为下一模块首帧 + 插值平滑。

---

## Physics Tracker：让 motion 物理可信

Kinematic motion 喂进 IsaacGym [33] 让 RL policy track。Character 62 body link，49 controllable joint（30 个是手指）。Policy 30Hz，仿真 120Hz。PPO [45] 训，2048 并行环境。

### Reward 设计的三个聪明点

公式 (S1) 总 reward：

$$r = 0.8 \, r_{\text{body}} + 0.2 \, r_{\text{hand}} + 0.05 \, r_{\text{energy}}$$

权重不 sum-to-1，是 unnormalized 加权。

**聪明点 1：reward 不用 contact state。** PhysHOI [55]、OmniGrasp [32] 都用 contact-based reward，但 diffusion 生成的 contact label 不可靠（有 penetration、mislabel）。作者干脆放弃 contact 信号，只用 pose tracking error。这是个工程妥协，但避开了上游 noise。

**聪明点 2：hand reward 用 $\alpha$ smooth transition。** 公式 (S3)：

$$r_{\text{hand}} = \exp\left(-\frac{5}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \alpha \|\mathbf{e}_{f,o}\| + (1-\alpha) \|\mathbf{e}_{f,w}\|\right)$$

- $\mathcal{F}$：手指集合
- $\mathbf{e}_{f,o}$：手指在 object local frame 下的位置误差
- $\mathbf{e}_{f,w}$：手指在 wrist local frame 下的位置误差
- $\alpha$：手距 object $\le 0.25m$ 时 $\alpha=1$，$\ge 1m$ 时 $\alpha=0$，中间线性插值

**直觉**：手远离物体时，约束"手指相对手腕的姿势"（保持 rest pose）；手靠近物体时，约束"手指相对物体的姿势"（保证 grasp 接触）。$\alpha$ 平滑切换避免 reward 突变让 policy 不稳。

这 trick 可以推广到任何"接近 vs 接触"两种控制目标的 manipulation RL。

**聪明点 3：Importance sampling。** Motion 长达数分钟，从头训 policy 极慢。作者把 2048 并行环境分 $|\mathcal{O}|$ batch，每 batch 分配一个 target object，**初始化直接放 pre-grasp pose**。

好处：
- 集中学"pre-grasp → grasp → 后续 locomotion"这个最难转移
- 同时学多 object 交互，避免 exploration local minima
- 不用每 episode 从头跑

这思路跟 HRL 的 option-critic 有联系——分 batch 训 sub-policy 再 composition。

---

## 实验数据我的解读

Table 1 最关键看三列：

| Method | $CC$↑ | $IV$↓ | $C_{F1}$↑ |
|---|---|---|---|
| CNet+GRIP [48] | 3.99% | 49.00 | 0.70 |
| CNet | 3.54% | 48.56 | 0.84 |
| C+RNet | 3.84% | 24.78 | 0.92 |
| C+R+FNet (Ours) | **5.53%** | **19.06** | 0.92 |

- $IV$（intersection volume）从 49 → 19，RefineNet 起决定性作用——relative pose loss 大幅减少穿模
- $CC$（contact coverage）从 3.5% → 5.5%，FingerNet 让接触更密
- $C_{F1}$ 在 RefineNet 阶段就饱和，说明 contact label 主要靠 RefineNet 保证，FingerNet 只补 detailed motion

Physics tracker tracking error：human joint $E_h = 5.45$ cm，object $E_o = 4.67$ cm。对比 PhysHOI [55] 直接 fail（无法 grasp general object）——PhysHOI 原本学篮球，transfer 不行，说明 general object manipulation 必须有多 object 训练分布。

---

## 我会吐槽的点

1. **慢得离谱**：4 秒 motion 要 stage 1/3/4 各 4 秒 + stage 2 优化 3 分钟，RL 训练每 4 秒 motion 2 小时。离 interactive agent 十万八千里
2. **Stage 2 优化依赖 object mesh 和 SMPL-X 兼容**，对 arbitrary CAD 不一定稳健
3. **RefineNet wrist 轨迹改了要 IK 投回 full-body**，IK 可能让 shoulder/neck 不自然，paper 没量化
4. **FingerNet 只在 GRAB 上训**，对小物体 grasp pattern 偏向强，搬大物体可能 extrapolation 不足
5. **Reward 权重 0.8/0.2/0.05 手 tuned**，迁移性不明
6. **LLM planner 用规则约束 execution order**，复杂场景（嵌套依赖、临时目标）会失效

---

## 我觉得可以抽出来复用的 idea

如果让我从这 paper 偷东西，我会偷这三个：

1. **Symbolic intermediate layer**：LLM 输出 symbol relation，algorithm 编译成数值。任何"LLM 推理 + 精确数值"的 task 都能用这 pattern。

2. **Point cloud alignment 表达 relative pose constraint**：不判别 contact，用 surface point 在 local frame 下的分布对齐。比 binary contact label 连续可微，比 absolute pose constraint 对 moving object 鲁棒。

3. **$\alpha$ smooth transition reward**：接近目标时切 reward 项，用 distance-based smooth interpolation 避免突变。任何"两阶段控制目标"的 manipulation RL 都能用。

---

## 参考链接

- Paper PDF: 搜 "Human-Object Interaction from Human-Level Instructions Stanford"
- CHOIS: https://arxiv.org/abs/2312.03913
- FullBodyManipulation / OMOMO: https://arxiv.org/abs/2305.18066
- GRAB: https://grab.is.tue.mpg.de/
- DexGraspNet: https://dexgraspnet.github.io/
- GRIP: https://arxiv.org/abs/2308.11617
- UniHSI: https://arxiv.org/abs/2312.06472
- PhysHOI: https://arxiv.org/abs/2312.04393
- OmniGrasp: https://arxiv.org/abs/2407.11385
- DeepMimic: https://arxiv.org/abs/1804.02717
- AMP: https://arxiv.org/abs/2105.07648
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Gym: https://arxiv.org/abs/2108.10470
- CLIP: https://arxiv.org/abs/2103.00020
- BPS: https://arxiv.org/abs/1908.09199
- 6D rotation: https://arxiv.org/abs/1812.07035
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Holodeck: https://arxiv.org/abs/2312.09067
- 3D-LLM: https://arxiv.org/abs/2307.12981
- HumanML3D: https://arxiv.org/abs/2105.14874
- ManipNet: https://arxiv.org/abs/2105.05303
- InterDreamer: https://arxiv.org/abs/2403.19652

人话版本就这样。核心就是：LLM 当 symbolic reasoning engine，diffusion 当 motion prior，RL 当 physics enforcer，三个各管一段，靠 clever condition 串起来。paper 的价值不在端到端系统，在那些 component-level design pattern——尤其那个 wrist-object relative pose loss，我觉得挺漂亮的。

---

这篇 paper 是 Stanford 的 C. Karen Liu 组的工作（Zhenwu, Jiaman Li, Pei Xu, Karen Liu），核心目标是**从 human-level 语言指令合成 physically plausible、long-horizon、含 full-body + 手指 + 物体 motion 的交互序列**。整个系统是个三级流水线：LLM Planner → Diffusion Motion Generator → Physics RL Tracker。最大的卖点在于它**首次把 detailed finger motion 跟 full-body + 物体 motion 三者协同合成**，又通过 RL tracker 拿到了物理真实性。

下面我按它的 pipeline 讲，公式变量上下标我都会拆开说明，并穿插我自己的理解与批评。

---

## 1. 系统动机与总体设计

任务的困难在哪里？拆开看三个根本性 bottleneck：

1. **指令到具体动作的 grounding gap**：你说"在电视前做 yoga"，agent 要知道"电视前"通常意味着先清场，把挡路的东西挪走。这种 common-sense 推理 LLM 擅长，但**精确给出 3D 坐标和 quaternion LLM 不行**。
2. **数据模态对不齐**：FullBodyManipulation 数据集 [26] 有 full-body + 大物体 manipulation，但没有手指；GRAB 数据集 [46] 有手指，但只针对小物体、没有 locomotion。两份 dataset 没法合一训出端到端模型。
3. **Kinematic motion 不物理**：diffusion model 生成 motion 容易产生 foot sliding、hand-object penetration、object 悬空等 artifact，必须在物理仿真里 re-track 才能可信。

作者的对应设计是分而治之：

| 模块 | 解决的问题 | 技术 |
|---|---|---|
| High-Level Planner | 指令 grounding | LLM + spatial relation 中间表示 + scene graph algorithm |
| Low-Level Motion Generator | 数据模态对齐 | 四阶段 diffusion pipeline，分而治之 |
| Physics Tracker | 物理真实性 | PPO + importance sampling，30/120 Hz 控制器 |

这种"先 kinematic 再 physics track"思路跟 DeepMimic [39] 一脉相承，但扩展到了 full-body + fingers + 多物体长序列。

---

## 2. High-Level LLM Planner

### 2.1 输入输出

- 输入：scene description（3D 场景里每个 object 的初始 layout）+ human-level instruction（如"I want to do yoga in front of the TV"）
- 输出：scene map $\{\{o_1, p_1, q_1\}, \dots, \{o_n, p_n, q_n\}\}$，其中 $o_i$ 是物体 id，$p_i$ 是目标 3D 位置，$q_i$ 是目标 orientation（quaternion 形式）；以及一个文本动作序列 execution plan $\{l_1, l_2, \dots, l_T\}$，每个 $l_t$ 形如"lift the object, move the object, put down the object"。

### 2.2 关键设计：用 spatial relations 当中间表示

直接让 LLM 输出 $(p, q)$ 数字会让数字飘、单位混乱、坐标系随意。作者抽象出三个 relation function：

1. `on(o1, o2)`：o1 在 o2 顶面上
2. `adjacent(o1, o2, direction, distance)`：o1 在 o2 某方向（east/west/south/north）若干米处
3. `facing(o1, o2)`：o1 朝向 o2

让 LLM 只输出这些符号化关系，然后用 algorithm 把它们 compile 成精确 3D 坐标。这跟 Holodeck [65]、3D-LLM [16]、Aguina-Kang 等的 layout generation 一脉相承——**符号化中间层把 LLM 的离散推理能力与几何计算解耦**。

### 2.3 算法细节（Algorithm 1）

输入 scene graph $G(V, E)$，节点是物体，边是 spatial relation，边是 directional 从 object2 指向 object1。$V_S$ 是 static 节点，$V_M$ 是要移动的节点。

```
Procedure ComputePositions(G, L):
    V_P ← V_S                      # 已处理集合初始化为静态节点
    while V_M 非空:
        V' ⊆ V_M：所有前驱都在 V_P 中的节点
        for v in V':
            L(v) ← Update(v, G, L) # 计算位置
            V_P ← V_P ∪ {v}
            V_M ← V_M \ {v}
    return L

Procedure Update(v, G, L):
    P ← []
    for u in predecessors(v):
        O ← ComputeOffset((u,v), L(u))
        P.append(L(u) + O)        # 候选位置
    return Average(P)             # 多前驱时取平均

Procedure ComputeOffset((u,v), L(u)):
    if edge == "on":
        O_height ← u 顶面高度
        O_horizontal ← 在 u 水平范围内 uniform sample
        return [O_horizontal, O_height]
    elif edge == "adjacent":
        使用 LLM 给的 direction 和 distance 算 O
        return O
```

这本质上是**带几何 offset 的 topological sort**：先处理静态节点（如 floor），再一层层展开。注意 `on` 关系里水平位置是 sample 的（保留随机性，避免每次输出都摆到正中央）；高度是 deterministic 对齐顶面。

Orientation 通过 `facing(o1, o2)` 算：把 o1 在自身 local frame 里的 canonical direction 旋到与 $(p_{o_2} - p_{o_1})$ 向量对齐。这里要求每个物体预先定义好"canonical front direction"，这点在 robotic dataset 里通常很自然（如电视的屏幕面）。

### 2.4 Execution Plan

LLM 还要排序——比如 vase 在 table 上就要先搬 vase 再搬 table。论文用了三条硬规则：
- 一次只动一个物体
- 同一物体不动两次
- 不能搬"上面有东西"的物体

然后 A* planner 在 2D 平面上生成 collision-free waypoints 喂给下游。

**我的看法**：这层 abstraction 设计得挺聪明，把 LLM 当作"自然语言到符号图的翻译器"。缺陷也很明显：relation 集合是手工定义的，处理不了"靠墙但不要正对门"这种隐式约束；canonical direction 需要物体预处理，对 arbitrary CAD 模型不友好。

---

## 3. Low-Level Motion Generator

这是 paper 的技术核心，分**Interaction Module（四阶段）** + **Navigation Module**。

### 3.1 背景：Conditional Diffusion

公式 (1) 是 reverse process：

$$p_\theta(\mathbf{x}_{n-1} \mid \mathbf{x}_n, \mathbf{c}) := \mathcal{N}\left(\mathbf{x}_{n-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_n, n, \mathbf{c}), \sigma_n^2 \mathbf{I}\right)$$

- $\mathbf{x}_n$：第 $n$ 步（从 $N$ 倒数到 $0$）的 noisy motion
- $\mathbf{x}_{n-1}$：少一步噪声的 motion
- $\mathbf{c}$：condition（在这里是 $\{S, G, T\}$ 等）
- $\boldsymbol{\mu}_\theta$：网络预测的 mean（即下一步 clean 化的方向）
- $\sigma_n^2$：固定 variance，通常按 noise schedule 给定
- $\theta$：网络参数

训练 loss：

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, n} \| \hat{\mathbf{x}}_\theta(\mathbf{x}_n, n, \mathbf{c}) - \mathbf{x}_0 \|_1$$

用 L1 而不是 L2，对 motion 这种有结构性稀疏误差的数据更稳。所有四个 diffusion model 都用 transformer-based architecture，supplementary Fig. S4 给了 RefineNet 的结构示意。

### 3.2 Stage 1: CoarseNet

基于预训练的 CHOIS [25]，输出**没有手指细节**的 full-body + object motion。

**Data representation**：

- Human motion $\mathbf{H} \in \mathbb{R}^{T \times D}$，每帧包含 global joint positions 和 6D rotations [74]。6D rotation 是用 (first 2 rows of rotation matrix) 表示，避免 quaternion 的 discontinuity 问题。$D$ 是去除手指关节的 pose 维度。
- Object motion $\mathbf{O} \in \mathbb{R}^{T \times 12}$：global position（3 维）+ 3x3 rotation matrix（9 维），共 12 维。
- Contact labels $\mathbf{L} \in \mathbb{R}^2$：左右手各一个 contact binary label。
- 联合输出 $\mathbf{x} = \{\mathbf{H}, \mathbf{O}, \mathbf{L}\}$。

**Condition** $c = \{S, G, T\}$：
- $S \in \mathbb{R}^{T \times (D+12)}$：masked motion representation。首帧填 human+object pose，末帧填 object pose，每 30 帧填一个 2D waypoint，其余 padding 为 0。这种 sparse masking 让模型既能 anchor 起止状态又能 follow waypoint。
- $G$：object geometry，用 BPS (Basis Point Sets) [42] 编码——把 object mesh 投影到一组预定义 basis point 上的距离向量。
- $T$：CLIP [43] text embedding。

**Contact phase segmentation**：根据预测的 $\mathbf{L}$ 把 motion 切成 pre-contact / contact / post-contact 三段（左右手分别切），取 contact 段的平均 wrist pose $w$ 作为下一步输入。这一步是后面 alignment 的 anchor。

### 3.3 Stage 2: Grasp Pose Generation

这步用 DexGraspNet [53] 这个 robotic grasp 优化方法。它原本是为机器人手设计的，但因为它只用 kinematic chain 信息，所以直接 compatible with SMPL-X [37] 人体手模型。

输入：object mesh、wrist pose $w$、rest finger pose。优化目标是 minimize：

$$E = E_{\text{force closure}} + \lambda_1 E_{\text{surface}} + \lambda_2 E_{\text{natural pose}}$$

（paper 没显式写出公式，但从 DexGraspNet 的原始 paper 知道是这个结构）

- $E_{\text{force closure}}$：grasp 是否能抵抗外力（基于 friction cone 分析）
- $E_{\text{surface}}$：手指贴近物体表面
- $E_{\text{natural pose}}$：手型自然正则

作者做的两个改动很关键：
1. **采样点从 2000 增到 20000**：因为 DexGraspNet 原本处理小物体，对大物体（如箱子、桌子）原本采样密度不够，会出现手指穿模或浮空。
2. **双手任务去掉 force closure 项**：搬箱子靠双手配合而非单手 force closure，硬加会逼出反物理姿势。

输出 $\hat{g} = \Omega(\hat{w}, \hat{\theta})$，其中 $\hat{w}$ 是优化后的 wrist pose，$\hat{\theta}$ 是 finger pose。整个 contact phase 都保持这个 grasp。

### 3.4 Stage 3: RefineNet

CoarseNet 输出的 wrist pose 跟 $\hat{w}$ 不一定对齐，强行拼会有 artifacts。RefineNet 是基于 CoarseNet 再加两个 condition 的 conditional diffusion：

$$c_r = \{W, S_r, G, T\}$$

- $W \in \mathbb{R}^{T \times 18}$：wrist-object relative pose。每帧 18 维 = 双手 × (3 位置 + 6D rotation)。只在 contact phase 填值，其余为 0。
- $S_r$：跟 $S$ 一样但加了"object static"约束——pre-contact 和 post-contact 阶段填 static object pose，防止物体在没接触时晃动。

**Wrist-Object Relative Pose Loss**：这是个非常巧妙的 design。问题在于直接约束 wrist 跟 object 的 absolute pose 不够——因为物体在动。要约束的是 **wrist 在 object local frame 里的相对 pose**，这样物体怎么动，wrist 就怎么跟。

具体做法：

1. 在 rest object surface 上 uniform sample 100 点 $\mathbf{K}_{\text{rest}} \in \mathbb{R}^{100 \times 3}$
2. 预计算这 100 点在每帧每个 wrist local frame 中的位置 $\mathbf{K}_w \in \mathbb{R}^{2 \times T \times 100 \times 3}$（2 是双手，T 是帧数，100 是点数，3 是 xyz）
3. 推理时，每帧用 predicted object pose 和 wrist pose 把 $\mathbf{K}_{\text{rest}}$ 变换到 wrist local frame：

公式 (2)：$\mathbf{K}_{\text{global}} = R_o \mathbf{K}_{\text{rest}} + T_o$
- $R_o \in SO(3)$：predicted object rotation
- $T_o \in \mathbb{R}^3$：predicted object translation
- $\mathbf{K}_{\text{rest}}$：在 rest pose 下 object 表面采样点
- $\mathbf{K}_{\text{global}}$：这些点在 global frame 下的位置

公式 (3)：$\hat{\mathbf{K}}_w = R_w^{-1}(\mathbf{K}_{\text{global}} - T_w)$
- $R_w$：predicted wrist rotation
- $T_w$：predicted wrist translation
- $\hat{\mathbf{K}}_w$：把 global 点变换到 wrist local frame

公式 (4) loss：

$$\mathcal{L}_{\text{relative}} = \sum_{t=1}^{T} \mathbf{L}_t \| \hat{\mathbf{K}}_{w,t} - \mathbf{K}_{w,t} \|_1$$

- $\mathbf{L}_t$：第 $t$ 帧 contact label，非接触时 mask 为 0
- $\hat{\mathbf{K}}_{w,t}$：网络预测的 wrist frame 下的 object 表面点位置
- $\mathbf{K}_{w,t}$：ground truth（来自 Stage 2 优化后）的 wrist frame 下的 object 表面点位置

**直觉**：这个 loss 在说"无论物体在 world frame 里怎么移动，你手腕在物体坐标系里的相对关系要一致"。这本质上是把 contact 用 point cloud 对齐的思路表达——不用判别 contact，用 distance 即可。

**Post-processing**：还有个细节 trick——RefineNet 输出仍有小 misalignment，作者用一个 boundary correction：

- 找 pre-contact / contact 边界处，预测 object pose 与 static pose 的差 $\Delta d \in \mathbb{R}^6$（position + rotation in axis-angle）
- 在 contact phase 用 $\hat{O}_t = O_t + \alpha_t \Delta d$ 平滑过渡，$\alpha_t$ 从 1 衰减到 0
- 然后 wrist 轨迹用 $\hat{w} = (\hat{R}, \hat{T})$ 重新算：rotation = $R_o \hat{R}$，position = $R_o \hat{T} + T_o$
- 最后用 IK 把 wrist 轨迹投回 full-body pose

这层 IK 是必要的，因为 wrist pose 改了，肩膀肘部也得跟着调。

### 3.5 Stage 4: FingerNet

生成 pre-contact 和 post-contact 阶段的 finger motion（contact 阶段保持 grasp pose 不变）。模型在 GRAB dataset [46] 上训练，只用 grasp 前后各 1 秒。

输出 $\mathbf{F} \in \mathbb{R}^{T \times D'}$，是每帧手指关节的 local 6D rotation。

Condition $c_f = \{P, F_s, F_e\}$：
- $P \in \mathbb{R}^{T \times 100}$：hand-object spatial relationship。采样 100 个 palm-side mesh vertex，算它们到 object 表面的最近距离，作为 proximity feature。计算时手指设为 mean pose（这样 $P$ 不被手指自身 pose 影响）。
- $F_s, F_e$：start 和 end finger pose（rest pose 和 grasp pose）

**Mirror 策略**：GRAB 数据集左右手是分开的数据，作者把 left hand 数据镜像后合并到 right hand 训练。推理时 mirror left input → predict right → mirror back。这跟 ManipNet [70] 一致，是个常见技巧。

### 3.6 Navigation Module

生成 locomotion 的 conditional diffusion model，输出 $\mathbf{H} \in \mathbb{R}^{T \times D}$。

Condition：
- initial human pose
- 2D waypoints
- waypoint orientations（normalized direction vectors，控制朝向）

**模块切换**：距离物体 >1m 用 navigation，<1m 切到 interaction。切换时把上一模块末帧作为下一模块首帧，再加插值平滑。

---

## 4. Physics Tracker

### 4.1 仿真设置

- 引擎：IsaacGym [33]，GPU 物理仿真
- character：62 个 body link，49 个 controllable joint（其中 30 个是手指）
- control policy 工作频率 30Hz，仿真 120Hz
- state $\mathbf{s}_t \in \mathbb{R}^{(62+|\mathcal{O}|) \times 13}$：每个 body link 和当前 target object 的 position (3) + orientation (4 quaternion) + linear velocity (3) + angular velocity (3) = 13 维
- target observation $\boldsymbol{\sigma}_{O_t} \in \mathbb{R}^{(62+|\mathcal{O}|) \times 7}$：target position + orientation
- action $\mathbf{a}_t \in \mathbb{R}^{49 \times 3}$：每个 joint 的 3-DoF PD target
- 训练：PPO [45]，Adam optimizer
- 2048 个并行环境

### 4.2 Reward 设计

公式 (S1)：

$$r = 0.8 \, r_{\text{body}} + 0.2 \, r_{\text{hand}} + 0.05 \, r_{\text{energy}}$$

权重 0.8:0.2:0.05 不是 sum-to-1，这是个 unnormalized 加权。Body 权重最高因为运动轨迹最关键；hand 权重相对低但要求高精度；energy 是个小正则防 jitter。

公式 (S2)：

$$r_{\text{body}} = 0.5 \exp\left(-15 \sum_{b \in \mathcal{B}} w_{q,b} \|\mathbf{e}_{q,b}\|^2\right) + 0.5 \exp\left(-15 \sum_{b \in \mathcal{B}} w_{p,b} \|\mathbf{e}_{p,b}\|^2\right)$$

- $\mathcal{B}$：character 的 body link + 当前激活的 target object
- $\mathbf{e}_{q,b}$：link $b$ 的 orientation error，用把 link 从当前朝向旋到 target 朝向所需的 radian 衡量
- $\mathbf{e}_{p,b}$：link $b$ 的 Euclidean 位置误差
- $w_{q,b}, w_{p,b}$：权重（Table S1）
- 系数 15 控制指数衰减速度——误差平方增长时 reward 快速下降

Table S1 的权重很有意思：
- root (pelvis) 和 target object 权重 $w_q=1, w_p=1$（最高）
- wrists $w_q=0.3, w_p=0.3$
- thighs $w_q=0.5$（影响整体重心）
- 大多数 link 只有权重 0.2

Position tracking 只对 root 和 end effectors（foot、hand 不含手指）加权，其它 link 靠 orientation tracking 约束自然性。这是 DeepMimic 风格的简化。

公式 (S3)：

$$r_{\text{hand}} = \exp\left(-\frac{5}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \alpha \|\mathbf{e}_{f,o}\| + (1-\alpha) \|\mathbf{e}_{f,w}\|\right)$$

- $\mathcal{F}$：手指集合
- $\mathbf{e}_{f,o}$：手指 $f$ 相对 target object 的位置误差，在 object local frame 下计算
- $\mathbf{e}_{f,w}$：手指 $f$ 相对 wrist 的位置误差，在 wrist local frame 下计算
- $\alpha$：插值系数。当 hand 距 object $\le 0.25m$ 时 $\alpha=1$，$\ge 1m$ 时 $\alpha=0$，中间线性插值

**直觉**：手远离物体时，应该约束"手指相对手腕的姿势"（保持 rest pose）；手靠近物体时，应该约束"手指相对物体的姿势"（保证 grasp 接触）。这个 $\alpha$ 的 smooth transition 非常聪明，避免了"靠近时 reward 突变导致 policy 不稳"。

公式 (S4)：

$$r_{\text{energy}} = \exp\left(-\frac{1}{900} \sum_{e \in \mathcal{E}} \|\mathbf{a}_e\|^2\right)$$

- $\mathcal{E}$：key end effectors（feet + hands 不含手指）
- $\mathbf{a}_e$：end effector 的 linear acceleration
- 系数 1/900 让这个项在正常加速度下接近 1，只有 jitter 时才大幅惩罚

**关键设计**：reward **不用 contact state**。作者明确说 diffusion 生成的 contact label 不可靠，所以放弃基于 contact 的 reward。这跟 PhysHOI [55]、OmniGrasp [32] 等用 contact-based reward 的做法不同——这是个工程妥协，本质是上游 diffusion 模型还不够准。

### 4.3 Importance Sampling Strategy

由于 motion 长达数分钟、涉及多个 object，从头训练 policy 极慢。作者把 2048 个并行环境分成 $|\mathcal{O}|$ 个 batch，每个 batch 分配一个 target object，**初始化时直接放进 pre-grasp pose**。

这样的好处：
- policy 集中学习"pre-grasp → grasp → 后续 locomotion"这个最难的转移
- 同时学习多个 object 交互，避免 exploration 陷入 local minima
- 不用每个 episode 从头跑——可以聚焦 hard region

---

## 5. Experiments

### 5.1 High-Level Planner

- 25 个 instruction，GPT-4o
- Baseline：LLM 直接输出 3D 位置和 orientation
- Metrics：$PE_p$（位置错误率）、$PE_o$（orientation 错误率）

| Method | $PE_p$↓ | $PE_o$↓ |
|---|---|---|
| Baseline | 21.9% | 12.5% |
| Ours | 3.1% | 1.6% |

差距巨大。Baseline 常见错误包括：monitor 放错位置/朝向、椅子跟桌子相交。Human study 也支持 Ours。

### 5.2 Low-Level Motion Generator

数据集分工：
- CoarseNet、RefineNet → FullBodyManipulation [26]（10 小时，15 个 object，无手指）
- FingerNet → GRAB [46]（10 subject，51 object，有手指但无 locomotion）
- Navigation → HumanML3D [12]（28 小时）

Metrics 跨多个维度：
- Condition：$T_{xy}$（waypoint error, cm）
- Human motion：$FS$（foot sliding）、$H_{\text{feet}}$（foot height）
- Interaction：$C_{\text{prec}}, C_{\text{rec}}, C_{F1}$（contact precision/recall/F1）、$CC$（contact coverage，hand 点在物体表面 ±2mm 内的百分比）、$IV$（intersection volume, cm³）
- GT difference：$MPJPE$、$T_{\text{obj}}, O_{\text{obj}}$

Table 1 关键数据：

| Method | $CC$↑ | $IV$↓ | $C_{F1}$↑ |
|---|---|---|---|
| CNet+GRIP [48] | 3.99% | 49.00 | 0.70 |
| CNet | 3.54% | 48.56 | 0.84 |
| C+RNet | 3.84% | 24.78 | 0.92 |
| C+R+FNet (Ours) | **5.53%** | **19.06** | 0.92 |

几点观察：
- $IV$ 从 49 降到 19，RefineNet 居功至伟（relative pose loss 大幅减少穿模）
- $CC$ 从 3.5% 升到 5.5%，FingerNet 让接触更密
- $C_{F1}$ 在 C+RNet 阶段就饱和了，说明 contact label 主要靠 RefineNet 保证，FingerNet 只补 detailed motion

### 5.3 Physics Tracker

| Metric | Ours |
|---|---|
| $E_h$（human joint error, cm） | 5.45 |
| $E_o$（object error, cm） | 4.67 |

20 个序列，平均 30 秒，每序列 2 个不同 object。对比 PhysHOI [55]：PhysHOI 直接 fail（无法 grasp），Ours 成功。PhysHOI 原本是学篮球的，transfer 到 general object 不行——说明**general object manipulation 的 policy 必须有多 object 的训练分布**，单 task policy 不 generalize。

### 5.4 Inference Speed

- 4 秒 motion：stage 1/3/4 各约 4 秒，stage 2 优化约 3 分钟
- Physics tracker 训练：每 4 秒 motion 约 2 小时
- 单 NVIDIA RTX 4000 GPU

Stage 2 的 DexGraspNet 优化是主要 bottleneck，离 real-time 还很远。

---

## 6. 我的整体评论与联想

**优点**：
- 分阶段 diffusion 解决数据模态对齐问题，思路非常务实
- Wrist-object relative pose loss 用 point cloud 对齐表达 contact，比直接判别 contact 信号更鲁棒
- Physics tracker 的 reward 设计（不依赖 contact state、$\alpha$ smooth transition、importance sampling）非常工程化
- Importance sampling 把多 object 训练分 batch，回避了 long-horizon RL 训练困难

**可质疑点**：
- **Inference 速度不可用**：3 分钟优化 + 多个 diffusion step + 2 小时 RL 训练，离 interactive agent 差得很远
- **Stage 2 优化依赖物体 mesh 和 SMPL-X 兼容性**，对 arbitrary CAD 物体可能不稳健
- **RefineNet 的 wrist 轨迹仍需 IK 投回 full-body**，IK 可能产生 shoulder/neck 的不自然姿态，paper 没量化
- **LLM planner 用规则约束 execution order**，复杂场景（嵌套依赖、临时目标）会失效
- **Physics tracker 不依赖 contact reward 是个妥协**，意味着如果 diffusion 生成 grasp 本身错位，tracker 不会主动纠正
- **Reward 权重 0.8/0.2/0.05 是手工 tuned**，对其它 task 是否可迁移不明
- **FingerNet 只在 GRAB 上训练**，对小物体 grasp pattern 偏向强，搬大物体（如箱子边缘抓握）可能 extrapolation 不足

**与相关工作的关联**：
- 沿袭了 DeepMimic [39] → AMP [40] → PhysHOI [55] → OmniGrasp [32] 的 physics tracking 路线，但首次扩展到 long-horizon 多物体
- 跟 UniHSI [60] 都用 LLM + low-level controller 思路，UniHSI 用 chain-of-contact，本工作用 spatial relation
- 跟 CHOIS [25]、OMOMO [26] 系出同门（Karen Liu 组），CHOIS 是 Stage 1 的 base
- 跟 GRIP [48] 是 baseline 关系，GRIP 是 finger-only refinement
- 跟 Holodeck [65]、3D-LLM [16] 共享"LLM 输出符号化关系而非直接坐标"的设计哲学
- 跟 InterDreamer [64]、CHOIS 一样在 text → motion 用 CLIP embedding

**未来方向（paper 自己提到）**：
- 用 voxel grid 或 egocentric vision 输入 LLM/VLM
- Agent 能主动调整 head 朝向观察环境

我自己会延伸想的方向：
- Stage 2 优化换成 neural grasp regressor（如 ContactGrasp [3]、GraspField [23]）能省时间
- RefineNet 的 wrist-object relative loss 可以推广到 RL reward 里——直接用 point cloud alignment 当 reward，绕过 contact label 不可靠问题
- Physics tracker 的 reward 用 $\alpha$ 切换是个漂亮 trick，可以推广到任何"接近目标物体时优先约束相对位姿"的 manipulation RL 任务
- Long-horizon 训练的 importance sampling 思路可以跟 HRL 的 option-critic 结合——分 batch 训 sub-policy 然后 composition

**参考链接**：
- Project 与 paper：作者通常在 Stanford CS 页面有 project page，可搜 "Human-Object Interaction from Human-Level Instructions Stanford"
- CHOIS (Stage 1 base): https://arxiv.org/abs/2312.03913
- OMOMO / FullBodyManipulation dataset: https://arxiv.org/abs/2305.18066
- GRAB dataset: https://grab.is.tue.mpg.de/
- DexGraspNet: https://arxiv.org/abs/2304.12635 (also at https://dexgraspnet.github.io/)
- GRIP: https://arxiv.org/abs/2308.11617
- UniHSI: https://arxiv.org/abs/2312.06472
- PhysHOI: https://arxiv.org/abs/2312.04393
- OmniGrasp: https://arxiv.org/abs/2407.11385
- DeepMimic: https://arxiv.org/abs/1804.02717
- AMP: https://arxiv.org/abs/2105.07648
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Gym: https://arxiv.org/abs/2108.10470
- CLIP: https://arxiv.org/abs/2103.00020
- BPS (Basis Point Sets): https://arxiv.org/abs/1908.09199
- 6D rotation representation [Zhou et al.]: https://arxiv.org/abs/1812.07035
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Holodeck: https://arxiv.org/abs/2312.09067
- 3D-LLM: https://arxiv.org/abs/2307.12981
- InterDreamer: https://arxiv.org/abs/2403.19652
- ManipNet: https://arxiv.org/abs/2105.05303
- GPT-4o: https://openai.com/index/hello-gpt-4o/
- HumanML3D: https://arxiv.org/abs/2105.14874

如果你（Karpathy）想 build intuition 的话，我建议盯住三个"工程化美学"：
1. **spatial relation 中间层** = 把 LLM 的离散推理与几何计算解耦的通用模式
2. **wrist-object relative pose loss** = 用 point cloud alignment 表达 contact，比 binary contact label 更连续可微
3. **reward 的 $\alpha$ smooth transition** = 在 manipulation RL 里处理"接近 vs 接触"两种控制目标的优雅方式

这三个 idea 都可以单独抽出来用到别的系统里——这篇 paper 真正的价值不在端到端系统，而在这些 component-level design pattern。
