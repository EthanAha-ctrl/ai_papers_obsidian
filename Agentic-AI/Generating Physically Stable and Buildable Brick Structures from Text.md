---
source_pdf: Generating Physically Stable and Buildable Brick Structures from Text.pdf
paper_sha256: bc4e753bee1239144e2f581898fe25d8c5b52dce27eba3f9fce746524b0e5707
processed_at: '2026-08-04T13:51:00-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# BRICKGPT 的人话版本

## 一句话总结

让一个 LLM 学会 "拼 LEGO"，每拼一块就检查一下物理上站不站得住，站不住就退回几步重新拼。

---

## 为什么这件事难

你让 ChatGPT 画个椅子，它能给你一段文字描述，也能给你一段代码画个 3D 模型。但那个模型是"假的"——它存在于电脑里，可以悬浮、可以穿模、可以没有支撑。你拿去 3D 打印或者用真 LEGO 拼出来，直接塌了。

物理世界有两条铁律 digital generation 从来不管：

1. **Buildability**：你得能用标准零件一步步拼出来，中间不能有一步需要"把一块砖塞进两块已经粘死的砖中间"
2. **Stability**：拼完之后它不能倒，每块砖都得有支撑，friction 得够用

现有 text-to-3D 方法（DreamFusion、Hunyuan3D 这些）完全 ignore 这两条。它们优化的是 visual appearance，不是 structural integrity。

---

## 核心 insight

LEGO 结构本质上是一个 **序列**。你拼 LEGO 的时候就是一块一块往上加的，这个顺序天然就是一个 sequence。

LLM 最擅长什么？生成 sequence。

所以这里的 trick 就是：把每块砖的位置和尺寸写成一行文字，比如 `2x4 (3,5,7)`，整个 LEGO 结构就变成了一段文本。LLM 的 next-token prediction 直接变成 next-brick prediction。

```
2x4 (0,0,0)
2x4 (0,4,0)
1x2 (1,0,1)
...
```

就这么简单。LLM 不需要学新的 architecture，它的 inductive bias 直接 transfer。

---

## 数据怎么来的

没人会手动拼 47000 个 LEGO 模型来训练模型。所以他们搞了个 pipeline：

1. 从 ShapeNetCore 拿 28000 个 3D mesh（椅子、桌子、吉他、船、车……）
2. 把 mesh voxel化 成 $20 \times 20 \times 20$ 的小方块网格
3. 用一个 **delete-and-rebuild** 算法把 voxel 转成标准 LEGO 砖块（只用 8 种尺寸：$1\times1, 1\times2, 1\times4, 1\times6, 1\times8, 2\times2, 2\times4, 2\times6$）
4. 这个算法用了人类砌墙的常识：**错缝搭接**（上下层砖的方向交叉）、**优先 interlock**（一块砖尽量压住下面多块砖）、**大砖优先**
5. 对每个 mesh 生成多个不同的砖块布局，增加多样性
6. 用 stability analysis 过滤掉物理上会塌的，只留 stable 的

最后拿到 47000+ 个 stable structure。每个 structure 渲染 24 个视角的图，喂给 GPT-4o 生成 5 条 caption（从粗到细）。这就构成了 text-brick pair。

---

## 物理稳定性到底怎么算的

这是整篇 paper 最 hard-core 的部分。来自 CMU 同一个组的另一篇 paper StableLego。

想象一块 LEGO 砖，它身上可能受哪些力：

- **重力**：往下
- **上面砖压它的力**（pressing，蓝色）
- **下面砖顶它的力**（supporting，紫色）
- **knob 连接处的摩擦力**（dragging，绿色）——这是 LEGO 不散架的关键
- **侧面相邻砖的 shear force**（黄色）

一块砖要 **static equilibrium**，需要两个条件：

$$\sum_j F_i^j = 0 \quad \text{(合力为零)}$$

$$\sum_j \tau_i^j = 0 \quad \text{(合力矩为零)}$$

- $F_i^j$ 是第 $i$ 块砖上的第 $j$ 个力
- $\tau_i^j = L_i^j \times F_i^j$ 是对应的力矩，$L_i^j$ 是力臂（从砖的质心到力作用点的向量）

然后整个结构变成一个 **非线性优化问题**：找到一组力分配方案，使得所有砖都满足 equilibrium，同时让所需的 friction 尽量小、尽量均匀。

$$\arg\min_{\mathcal{F}} \sum_i^N \left\{ \left| \sum_j F_i^j \right| + \left| \sum_j \tau_i^j \right| + \alpha \mathcal{D}_i^{\max} + \beta \sum \mathcal{D}_i \right\}$$

这里 $\mathcal{D}_i$ 是所有 dragging force（摩擦力候选）的集合，$\mathcal{D}_i^{\max}$ 是其中最大的那个。$\alpha = 10^{-3}$, $\beta = 10^{-6}$。

- 前两项：惩罚不平衡的砖
- 第三项：惩罚 worst-case friction（防止单点 stress concentration）
- 第四项：惩罚总 friction（鼓励均匀分担）
- $\alpha \gg \beta$：宁可总摩擦大一点，也不能让某一点爆掉

约束条件：
1. 所有力 ≥ 0（你不能有"负的支撑力"）
2. pulling 和 pressing 互斥（同一连接点要么在拉要么在压，不能同时）
3. Newton's third law：上面砖受的支撑力 = 下面砖受的压力

用 **Gurobi**（一个商业优化求解器）解这个 nonlinear program。对 <200 块砖的结构，平均 0.35 秒搞定。

最后每个砖拿到一个 stability score $s_i \in [0,1]$：

$$s_i = \frac{F_T - \mathcal{D}_i^{\max}}{F_T}$$

- $F_T = 0.98$ N 是实测的 LEGO 连接处 friction capacity
- 如果 $\mathcal{D}_i^{\max} > F_T$，说明这块砖需要的摩擦力超过了 LEGO 实际能提供的 → $s_i = 0$ → 结构会塌

**Intuition**：这套方法不 simulate dynamics（不做物理仿真），而是直接问 "存不存在一组力分配让结构静止？"。这比跑一遍 bullet physics engine 快得多，而且给的是 hard guarantee 而非 probabilistic estimate。

---

## Inference 的两个关键 trick

光靠 fine-tuned LLM 生成的结构，100% 里有大量 collision、out-of-bounds、unstable 的情况。所以他们加了两个机制：

### Trick 1: Brick-by-brick Rejection Sampling

每生成一块砖，检查三件事：
1. 格式对不对（dimension 在 library 里）
2. 位置在不在 $20^3$ grid 范围内
3. 跟已有的砖有没有碰撞：$\mathcal{V}_t \cap \mathcal{V}_i = \emptyset$

不对就 resample。这个很便宜，因为只是 voxel-level 的 collision check。

**为什么不在每步都做 stability check？** 因为很多结构在搭建过程中是 unstable 的，但拼完之后 stable。最经典的例子就是 **arch**（拱门）—— 放 keystone（拱顶石）之前，两边的砖一直在往中间塌。如果你每放一块砖就检查 stability，拱门永远拼不出来。

### Trick 2: Physics-Aware Rollback

等整个结构生成完了（模型输出 EOF token），再做一次完整的 stability analysis。

如果发现有 unstable 的砖：
1. 找到 **最早** 的那块 unstable 砖（index 最小的）
2. 把它和它之后的所有砖全部删掉
3. 从那个点重新开始生成

这个可以迭代，最多 rollback 100 次。实际中位数只要 2 次，平均 40.8 秒生成一个结构。

**为什么 rollback 到最早的 unstable brick 而不是最后一块？** 因为如果第 5 块砖就 unstable 了，后面第 6、7、8 块都是基于一个 unstable foundation 拼的，留着也没用。回到第 4 块重新来。

**Intuition**：这本质上是 **backtracking search**，跟下棋 AI 的 minimax + alpha-beta pruning 思路类似——LLM 负责 "展开"（生成候选 move），physics solver 负责 "评估"（这个 position 能不能站住），不能站住就回退。

---

## Ablation 结果有多 dramatic

看 Table 1 的 ablation：

| 配置 | % valid | % stable |
|---|---|---|
| 去掉 rejection sampling + 去掉 rollback | 37.2% | 12.8% |
| 只有 rejection sampling，没有 rollback | 100% | 24.0% |
| Full method | 100% | 98.8% |

翻译成人话：
- **什么约束都不加**：模型 37% 的输出连格式都不对，stable 的只有 12.8%。LLM 学到了大概的 pattern 但细节上各种出错。
- **只加格式和碰撞检查**：格式 100% 对了，但物理稳定性只有 24%。LLM 学会了 "怎么写字" 但没学会 "什么结构能站住"。
- **加上 physics-aware rollback**：98.8% stable。

Rollback 把 stability 从 24% 拉到 98.8%，这是 **4x** 的提升。这说明 LLM 本身确实没学到 deep physics understanding，它学的是 spatial pattern。真正保证物理正确性的是那个 external physics solver。

---

## 跟其他方法比

Baseline 是 LLaMA-Mesh、LGM、XCube、Hunyuan3D-2 这些 text-to-3D 方法。做法是：先用它们生成 mesh，再用同样的 delete-and-rebuild 算法转成 LEGO。

结果：

| 方法 | % stable |
|---|---|
| LGM | 25.2% |
| XCube | 75.2% |
| Hunyuan3D-2 | 75.2% |
| Hunyuan3D-2 + stability analysis (生成 100 个取第一个 stable 的) | 88.4% |
| BRICKGPT | 98.8% |

Mesh-to-brick 的根本问题：mesh 是 continuous geometry，转成 discrete brick 时会丢失 interlocking information。一块砖可能正好卡在 mesh surface 上，但跟下面的砖没有任何 interlock。BRICKGPT 直接在 brick space 里生成，interlocking 是 native 的。

而且 Hunyuan3D-2 要生成 100 个结构才找到一个 stable 的，BRICKGPT 基本一次就成。

---

## 一些有意思的细节

**Brick ordering**：bricks 按 raster-scan from bottom to top 排列。这个 ordering 同时是 assembly sequence——你按这个顺序拼就行。这不是 trivial 的 design choice，因为 assembly sequence 的 buildability 也是一个 hard constraint（你不能先拼上面再拼下面）。

**Novelty check**：他们用 Chamfer distance 找每个生成结构在 training set 里的 nearest neighbor。结果生成结构跟 nearest neighbor 都有明显距离，说明模型不是在 memorize，而是在学习 combinatorial rules。

**Robot assembly**：真的用两个 Yaskawa 机械臂拼出来了。流程是 assembly-by-disassembly search（把正向拼装顺序反过来找拆除顺序，保证每步中间状态都 stable）→ action masking → asynchronous multi-robot planner → closed-loop force control。

**Color/Texture**：用 FlashTex 给 brick mesh 生成 UV texture，或者把 texture color 平均到每块砖上，match 到 LEGO 标准色库。这个是 post-hoc 的，不是 end-to-end 生成。

---

## 我的 take

这篇 paper 最 valuable 的 contribution 不是 BRICKGPT 这个模型本身，而是它 demonstrate 了一个 **pattern**：

> **LLM 生成 + symbolic solver 验证 + backtracking search = 解决 constrained combinatorial generation**

这个 pattern 可以 generalize 到很多地方：

- **Circuit design**：LLM 生成 netlist，EDA tool 检查 timing/DRC
- **分子设计**：LLM 生成 SMILES，cheminformatics 检查 valence/synthesis
- **代码生成**：LLM 生成代码，type checker 验证（这个已经在做了）
- **建筑结构设计**：LLM 生成 floor plan，structural engineer 检查 load path

LEGO 只是这个 pattern 的一个特别 clean 的 testbed，因为：
1. 零件有限（8 种）
2. 网格离散（$20^3$）
3. 物理约束可以精确求解（static equilibrium）
4. 有 ground truth（真的能拼出来）

真正的 limitation 是 $20^3$ grid 太小、brick library 太窄、只考虑 static loading。但作为 proof-of-concept，它把 "text → physically realizable design" 这条路打通了。

参考：
- StableLego (stability analysis): https://arxiv.org/abs/2407.08962
- FlashTex (texturing): https://arxiv.org/abs/2408.08242
- Apex-MR (multi-robot planning): https://arxiv.org/abs/2502.01880
- Assembly-by-disassembly: https://arxiv.org/abs/2112.10430
- 项目主页: https://avalovelace1.github.io/BrickGPT/

---

# BRICKGPT: 从 Text 生成物理稳定的 Brick 结构

## 论文核心 Idea 与 Intuition

这篇 paper 的核心 insight 是将 **next-token prediction** paradigm 重新用于 **next-brick prediction**。传统 text-to-3D 方法 (DreamFusion, XCube, Hunyuan3D) 生成的 mesh 在 digital world 看起来好看，但**物理上不可实现** —— 漂浮的部件、无法 assemble 的几何、没有 structural support。BRICKGPT 把 brick assembly 重新 formulate 为 autoregressive sequence generation 问题，并在 inference 时把 physics laws 当作 hard constraints 注入。

这是 "LLM as a world model / LLM as a planner" 思路在 physical design 领域的一个非常 clean 的实例化：language model 学的是 token 序列分布，而 brick 结构本身可以被 tokenize 成纯文本序列，于是 LLM 的 inductive bias 直接 transfer 过来。

参考链接:
- 项目主页: https://avalovelace1.github.io/BrickGPT/
- StableLego (stability analysis method): https://arxiv.org/abs/2407.08962
- LLaMA-Mesh: https://arxiv.org/abs/2411.09595
- FlashTex: https://arxiv.org/abs/2408.08242

---

## 1. StableText2Brick Dataset 构建

### 1.1 Brick Representation

每个 brick structure 表示为 $B = [b_1, b_2, \ldots, b_N]$，其中每个 brick state:

$$b_i = [h_i, w_i, x_i, y_i, z_i]$$

变量解释:
- $h_i$: brick 在 X 方向的长度 (单位为 stud)
- $w_i$: brick 在 Y 方向的长度
- $x_i, y_i, z_i$: brick 的 stud 位置 (最靠近 origin 的那个 stud 的坐标)
- $x_i \in [0, 1, \ldots, H-1]$, $y_i \in [0, 1, \ldots, W-1]$, $z_i \in [0, 1, \ldots, D-1]$
- $H, W, D$ 是 grid world 的尺寸，论文中固定为 $20 \times 20 \times 20$

**Intuition**: 这种表示法 encode 了 brick 的 dimension 和 orientation (h, w 顺序决定 orientation about vertical axis)，所有 brick 都是 1-unit tall 的 axis-aligned cuboid。这是一个非常重要的 design choice —— 把连续 6DOF pose 离散化成 grid 上的 5-tuple，让 tokenization 变得 trivial。

### 1.2 Shape-to-Brick Pipeline

输入 ShapeNetCore mesh → voxelization 到 $20^3$ grid → **delete-and-rebuild algorithm**:

1. **Greedy layer-by-layer placement**：从下往上，用 8 种标准 brick ($1\times1, 1\times2, 1\times4, 1\times6, 1\times8, 2\times2, 2\times4, 2\times6$) 贪心填充 voxel
2. **Placement priority heuristic**：
   - (a) 优先放置只在下一层 partially supported 的 brick
   - (b) 优先放置 touch 多个下层 brick 的 brick (interlocking)
   - (c) 大 brick 优先
   - (d) orientation 与下层相反 (cross-pattern，类似真实 LEGO 砌墙)
3. **Stability-aware repair**：用 stability analysis 找出 weak regions，delete 后 rebuild，stochastic 处理 tie

**Intuition**: 这套 heuristic 实际上是把人类 LEGO 设计师的"常识"显式 encode 进去 —— 错缝搭接、避免悬挑、优先 interlock。这跟 real-world masonry 和 bricklaying 的工程经验一致。

### 1.3 Stability Score (核心物理模型)

这是整篇 paper 最 technically deep 的部分，来自 StableLego (Liu et al., RAL 2024)。对每个 brick $b_i$，定义一组 candidate forces $\mathcal{F}_i = \{F_i^j\}_{j=1}^{M_i}$:

- **Gravity** (黑色箭头): 向下的重力
- **Vertical forces with top brick** (红色 = pulling, 蓝色 = pressing)
- **Vertical forces with bottom brick** (绿色 = dragging, 紫色 = supporting)
- **Horizontal shear forces**: knob 连接处 (cyan)、adjacent brick 之间 (yellow)

Static equilibrium 条件:

$$\sum_j^{M_i} F_i^j = 0, \qquad \sum_j^{M_i} \tau_i^j = \sum_j^{M_i} L_i^j \times F_i^j = 0 \quad (2)$$

变量解释:
- $F_i^j \in \mathbb{R}^3$: 第 $i$ 个 brick 上第 $j$ 个 candidate force vector
- $\tau_i^j$: 对应的 torque
- $L_i^j$: force lever (从 brick 质心到 force 作用点的位移向量)，所以 $L_i^j \times F_i^j$ 是叉乘算 torque

整个 stability analysis 是一个 **nonlinear program**:

$$\arg\min_{\mathcal{F}} \sum_i^N \left\{ \left| \sum_j^{M_i} F_i^j \right| + \left| \sum_j^{M_i} \tau_i^j \right| + \alpha \mathcal{D}_i^{\max} + \beta \sum \mathcal{D}_i \right\} \quad (3)$$

subject to:
1. $\mathcal{F}$ 中所有 force ≥ 0 (non-negative)
2. 互斥约束: pulling & pressing 不能共存；dragging & supporting 不能共存
3. Newton's third law: 连接点处 upper brick 受的 supporting force = bottom brick 受的 pressing force

变量解释:
- $\mathcal{D}_i \subset \mathcal{F}_i$: 作用在 $b_i$ 上的所有 dragging force (绿色箭头) 集合 —— 实际上就是 friction 候选
- $\mathcal{D}_i^{\max}$: $\mathcal{D}_i$ 中的最大值 (worst-case friction demand)
- $\alpha, \beta$: hyperparameter weights (论文取 $\alpha = 10^{-3}, \beta = 10^{-6}$)

**Objective 设计 intuition**: 
- 前两项惩罚 non-equilibrium (force/torque 不为零)
- 第三项惩罚 worst-case friction demand (避免单点 stress concentration)
- 第四项惩罚总 friction (鼓励均匀分布)
- $\alpha \gg \beta$ 意味着更怕出现 extreme value 而非总量大

Per-brick stability score:

$$s_i = \begin{cases} 0 & \text{if } \sum_j F_i^j \neq 0 \\ 0 & \text{if } \sum_j \tau_i^j \neq 0 \\ 0 & \text{if } \mathcal{D}_i^{\max} > F_T \\ \frac{F_T - \mathcal{D}_i^{\max}}{F_T} & \text{otherwise} \end{cases} \quad (4)$$

变量解释:
- $F_T$: 实测的 brick 连接处 friction capacity (论文取 $0.98$ N)
- $s_i \in [0, 1]$: 离 fully stable 越远，分数越低；只要 friction demand 超过 capacity，直接归零
- 结构整体 stable 当且仅当 $\forall i, s_i > 0$

**关键 intuition**: 这套方法把 LEGO 的 stud-and-tube 连接抽象成 friction-limited 接触力学，跟传统 rigid body simulation 不同 —— 它不需要 simulate dynamics，而是 directly solve 一个 LP/QP-like 问题来检查是否存在 feasible force distribution 使结构 static。这就是为什么对 <200 bricks 的结构平均只要 0.35 秒。

---

## 2. Model Fine-tuning

### 2.1 Base Model 与 Tokenization

Base: **LLaMA-3.2-1B-Instruct**。为什么不用更大? Paper 没明说，但 1B 已经足以 capture 20^3 grid 上的 combinatorial structure，且 LoRA fine-tuning 成本低。

**Brick 文本格式** (替代 LDraw):

```
{h}x{w} ({x},{y},{z})
```

例如: `2x4 (3,5,7)` 表示一个 2×4 的 brick 放在 (3,5,7)。

Why not LDraw:
1. LDraw 不直接 encode dimension (需要查表)
2. LDraw 包含 rotation/scale 等冗余信息 (这里是 axis-aligned 的)

Bricks 按 **raster-scan order from bottom to top** 排列 —— 这个 ordering 对 autoregressive model 很关键，因为它隐含了一个 buildable 的 assembly sequence。

### 2.2 Autoregressive Factorization

$$p(b_1, b_2, \ldots, b_N | \theta) = \prod_{i=1}^N p(b_i | b_1, \ldots, b_{i-1}, \theta) \quad (1)$$

这是标准 chain rule factorization，每个 brick 是一组 tokens (dimension tokens + position tokens)。

**Training setup**:
- LoRA rank 32, alpha 16, dropout 0.05
- 只对 query & value matrices 做 LoRA → 3.4M 可训练参数 (1B model 的 0.34%)
- AdamW, lr 0.002, cosine scheduler, 100 warmup steps
- 8× NVIDIA RTX A6000, global batch 64, 12 hours, 3 epochs
- 240k distinct prompts, 47k+ distinct structures, max 4096 tokens/sample

**Intuition**: 只对 Q/V 做 LoRA 是为了 preserve LLM 的 general capability (avoid catastrophic forgetting) 同时让 attention pattern 适应 brick structure 的 spatial reasoning。

---

## 3. Inference: Rejection Sampling + Physics-Aware Rollback

### 3.1 Brick-by-Brick Rejection Sampling

每生成一个 brick $b_t$，检查:
1. **Well-formatted**: dimension 在 library 里 (1×1, 1×2, ..., 2×6)
2. **In bounds**: position 在 $[0, 19]^3$ 内
3. **No collision**: $\mathcal{V}_t \cap \mathcal{V}_i = \emptyset, \forall i \in [1, t-1]$
   - $\mathcal{V}_i$: brick $b_i$ 占据的 voxel 集合

如果违反，resample。Temperature 每次 rejection 加 0.01 防止 stuck loop。

**为什么不做 step-by-step stability check**: Paper 明确论证 —— 很多结构在 construction 过程中 unstable，但 fully assembled 后 stable。例如 arch 在放 keystone 之前一直在坍塌边缘。Step-by-step check 会 overly constrain exploration。

### 3.2 Physics-Aware Rollback (Algorithm 1)

```
1. B ← empty
2. loop:
   3. for k = 1..max_rejections:
      4. context ← prompt ⊕ B.to_text_format()
      5. b ← θ.predict_tokens(context)
      6. if b is valid: break
   8. B.add_brick(b)
   9. if b is EOF:
      10. if B stable or max_rollbacks exceeded: return B
      11. while B unstable:
         12. I ← indices of unstable bricks
         13. i ← min(I)  // 最早的 unstable brick
         14. B ← [b_1, ..., b_{i-1}]  // truncate
      15. continue generation from B
```

**关键 insight**: rollback 不是从头重新生成，而是 truncate 到第一个 unstable brick 之前，让 model 从 partial stable structure 继续。这既保留了已经生成的好部分，又给 model 一次 "retry" 的机会。

- Max rollbacks: 100
- Median rollbacks: 2
- Median generation time: 40.8 seconds
- 失败率 (超过 max rollbacks): 1.2%

**Intuition**: 这其实是 **beam search 的退化版** + **constraint-guided decoding**。不像 chess engine 用 minimax，这里用 "trust LLM + verify with physics + backtrack on failure" 的策略，跟 AlphaCode 的 approach 思想上一致。

---

## 4. Brick Texturing & Coloring

### 4.1 UV Texture Generation (Eqn. 5)

$$I_{\text{texture}} = \text{FlashTex}(\mathcal{M}, \text{UV}_\mathcal{M}, c) \quad (5)$$

- $\mathcal{M}$: 把 brick structure 合并成的 mesh (移除 fully occluded bricks $B_{\text{occ}}$，保留 visible bricks $B_{\text{vis}}$)
- $\text{UV}_\mathcal{M}$: cube projection 生成的 UV map
- $c$: appearance text prompt
- $I_{\text{texture}}$: 输出 texture map

### 4.2 Uniform Brick Color (Eqn. 6, 7)

把 structure 转 voxel grid $\mathcal{V}$ → 每个 voxel $v$ 有 $N_v$ 个 visible face ($0 \le N_v \le 6$)。每个 face $f_i^v$ 拆成 2 个 triangle，映射到 UV region $S_i^v$。

Voxel color:

$$\mathcal{C}(v) = \frac{1}{N_v} \sum_{i=1}^{N_v} \mathcal{C}(f_i^v), \quad \forall v \in \mathcal{V} \quad (7)$$

其中 face color:

$$\mathcal{C}(f_i^v) = \frac{1}{|S_i^v|} \sum_{(x,y) \in S_i^v} I_{\text{texture}}(x, y)$$

- $|S_i^v|$: UV region 中 pixel 数量

Brick color = 平均其 voxel color，然后 nearest neighbor 到标准色库。

**Intuition**: 这是 texture-to-color 的 down-sampling —— 先用 FlashTex 生成 high-res texture，再 aggregate 到 brick 级别。这跟 "from NeRF to mesh texture" 的 bake-down 思路类似。

---

## 5. Experiments 解析

### 5.1 Quantitative Results (Table 1)

| Method | % valid | % stable | mean stab | min stab | CLIP | DINO |
|---|---|---|---|---|---|---|
| Pre-trained LLaMA (0-shot) | 0.0% | 0.0% | N/A | N/A | N/A | N/A |
| In-context learning (5-shot) | 2.4% | 1.2% | 0.675 | 0.479 | 0.284 | 0.814 |
| LLaMA-Mesh | 94.8% | 50.8% | 0.894 | 0.499 | 0.317 | 0.851 |
| LGM | 100% | 25.2% | 0.942 | 0.231 | 0.300 | 0.851 |
| XCube | 100% | 75.2% | 0.964 | 0.686 | 0.322 | 0.859 |
| Hunyuan3D-2 | 100% | 75.2% | 0.973 | 0.704 | 0.324 | 0.868 |
| Hunyuan3D-2 + stab analysis | 100% | 88.4% | 0.976 | 0.813 | 0.324 | 0.868 |
| Ours w/o rejection/rollback | 37.2% | 12.8% | 0.956 | 0.325 | 0.329 | 0.888 |
| Ours w/o rollback | 100% | 24.0% | 0.947 | 0.228 | 0.322 | 0.882 |
| **Ours (BRICKGPT)** | **100%** | **98.8%** | **0.996** | **0.915** | 0.324 | 0.880 |

**关键观察**:

1. **Pre-trained LLaMA 0-shot 完全失败** (0% valid) —— LLM 不能 zero-shot 生成结构化输出，需要 task-specific fine-tuning。

2. **In-context learning (5-shot) 仅 2.4% valid** —— Few-shot 远远不够，需要大规模 instruction tuning。

3. **Mesh-to-brick baselines 的痛点**: LGM 只有 25% stable，因为 Gaussian splatting 转 mesh 再 voxelization 的过程中丢失了 interlocking 信息。即使加 stability analysis 也只到 32.5%。

4. **Ablation 揭示的设计哲学**:
   - 去掉 rejection + rollback: valid 掉到 37.2% (大量 collision / out-of-library)
   - 只保留 rejection (去 rollback): valid 100% 但 stable 只 24% —— 说明 LLM 学到了 format 但没学到 physics
   - Full method: stable 98.8% —— rollback 是把 physics 真正注入生成的关键

5. **CLIP / DINO**: BRICKGPT 在 prompt alignment 上跟最强 baseline (Hunyuan3D-2) 持平甚至略优 (DINO 0.880 vs 0.868)。这很 surprising，因为 brick 结构是 discrete + low-res，理论上比 continuous mesh 难匹配 prompt。Paper 的解释是: discrete representation 强制 model 学 semantic structure，反而 alignment 更好。

### 5.2 Novelty Analysis (Figure 9)

对每个生成结构，用 Chamfer distance 在 voxel space 找 training set 中最近的 neighbor。结果显示生成结构与最近 neighbor 明显不同，证明 model 不是 memorization，而是在 learning combinatorial rules。

---

## 6. Robotic Assembly (Appendix B)

**Hardware**: Dual Yaskawa GP4 robot arms + ATI force-torque sensors + calibrated baseplate。

**Software stack**:
1. **Assembly-by-disassembly search** (Tian et al., SIGGRAPH 2022): 把 brick sequence 重排，使每个 intermediate structure 都 stable
2. **Action mask** (Liu et al., RAL 2025): data-free action masking，保证每步可执行
3. **Asynchronous planner Apex-MR** (Huang et al., RSS 2025): 多机器人任务分配 + motion planning
4. **Manipulation policy** (Liu et al., ISFA 2024): closed-loop force control for robust brick manipulation

**Intuition**: 这是 paper 的 "end-to-end physical realizability" claim 的真正落地 —— 从 text prompt → stable structure → executable robot plan → real LEGO assembly。整个 pipeline 没有 human-in-the-loop。

---

## 7. Critical Analysis & Open Questions

### 7.1 强项
1. **Constraint injection 优雅**: 用 rejection sampling + rollback 把 hard physics constraints 注入 autoregressive generation，比 soft penalty term 干净得多。
2. **数据集贡献**: StableText2Brick (47k structures, 21 categories) 是第一个大规模 stable brick-text pair dataset。
3. **Real-world validation**: 真的用 LEGO 拼出来了 + 机器人组装成功。

### 7.2 局限 / 未来方向
1. **Resolution limit**: $20^3$ grid 太粗，无法表达 fine detail。Scaling 到 Objaverse-XL 是 obvious next step。
2. **Brick library 限制**: 只有 8 种 axis-aligned brick。LEGO 实际有 slopes, tiles, brackets, hinges 等 hundreds of types。
3. **Stability model 简化**: 只考虑 static loading，不考虑 dynamic (push, vibration, drop)。Real LEGO 结构在搬运时会因 dynamic load 失败。
4. **No color-aware generation**: Color 是 post-hoc 加上去的，不是 end-to-end 生成。理想情况应该让 model 同时生成 shape + color。
5. **Single baseplate assumption**: 无法生成 multi-baseplate 或 free-standing structures。
6. **No 6DOF brick orientation**: 所有 brick axis-aligned，不能斜放。这极大限制了 expressiveness。

### 7.3 更深层的思考

这篇 paper 其实是 **"LLM for combinatorial reasoning under hard constraints"** 的一个 specific instance。Generalize 这个 idea:
- **Circuit design**: 用 LLM 生成 netlist，constraint check = timing closure + DRC
- **Molecule design**: 用 LLM 生成 SMILES，constraint check = valence + ring closure
- **Code generation with type safety**: LLM 生成 code，constraint check = type checker (已经有 typed lambda calculus 的工作)

BRICKGPT 的 contribution 本质上是证明了: **pre-trained LLM 的 sequence prior + lightweight constraint solver** 可以 solve real-world combinatorial design problems，不需要从头训练 specialized architecture。这跟 AlphaCode、FunSearch 的哲学是一脉相承的。

Stability analysis 用 Gurobi 求解 nonlinear program 是 bottleneck (40 秒/structure)。如果用 learned physics surrogate (像 PhysGym 或 neural physics simulator) 可以大幅加速，但会牺牲 hard guarantee。这个 trade-off 值得探索。

参考:
- AlphaCode: https://deepmind.google/discover/blog/competitive-programming-with-alphacode/
- FunSearch: https://www.nature.com/articles/s41586-023-06247-5
- Typed lambda calculus LLM: https://arxiv.org/abs/2310.13516
- StableLego: https://arxiv.org/abs/2407.08962
- Apex-MR: https://arxiv.org/abs/2502.01880

---

## 8. Summary

BRICKGPT 的核心 thesis: **physical design 是 sequence generation + constraint verification 的 interleaving process**。通过把 brick structure 编码成 text，让 LLM 负责 "creative generation"，把 physics solver 作为 "verifier"，二者通过 rejection sampling + rollback 闭环。这是 neurosymbolic architecture 的一个 clean instantiation —— neural model 提供 prior，symbolic solver 提供 guarantee。

工程上最 impressive 的不是模型大小 (1B) 或数据规模 (47k)，而是 **整个 pipeline 的端到端可执行性**: text → stable design → human/robot assembly。这是 few 3D generation works 能做到的。
