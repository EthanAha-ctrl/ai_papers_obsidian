---
source_pdf: Code-as-Monitor Constraint-aware Visual Programming for Reactive and Proactive
  Robotic Failure Detection.pdf
paper_sha256: 1541b85cfe283d6f0a29929932834380915f5adde69c96de377e0416b9b9fff6
processed_at: '2026-08-03T16:24:05-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，如果用一句大白话来概括这篇 paper 的核心直觉：**把 VLM 从“苦哈哈盯着屏幕找错”的质检员，提拔成了“写脚本写一次就下岗”的架构师，把苦活累活全甩给了确定性的 Python code 和 NumPy 算术运算。**

下面我尽量抛开学术黑话，把里面的技术逻辑用人话拆解给你，同时保留必要的公式和架构细节来 build your intuition。

---

### 1. 痛点到底在哪？

假设你让机器人“端起装着龙虾的锅放到炉子上，别让龙虾掉出来”。

传统 VLM 怎么做监控？比如 DoReMi [16]，就是机器人每动一下，就拿当前的 RGB 图去问 VLM：“龙虾还在锅里吗？锅平吗？”。这种做法有三个致命伤：
1. **VLM 看单张 2D 图没有 3D 概念**：锅稍微倾斜了 15 度，单看照片根本看不出来，但龙虾马上就要滑出去了。
2. **太慢太贵**：每秒调用好几次 GPT-4o，token 消耗巨大，机器人还得停下来等它思考。
3. **事后诸葛亮**：等 VLM 终于看出“龙虾掉桌子上了”，事情已经发生，只能从头再来。

**CaM 的思路**：VLM 别干这种需要精细数值判断的活儿了。你只在任务开始时看一眼，写一段 Python 监控代码，剩下的实时判断全交给代码去算。

---

### 2. CaM 的三步走架构

整个 system 就干三件事：

#### Step 1: VLM 当一次架构师
输入当前看到的多视角画面 $\mathcal{O}$ 和总指令 $\mathcal{L}_{\mathrm{global}}$，GPT-4o 把长任务拆成小目标 $l_{\mathrm{next}}$，并生成两类约束：
$$l_{\mathrm{next}}, \mathcal{C}_d, \mathcal{C}_u = \mathcal{F}_{\mathrm{VLM}}(\mathcal{O}, \mathcal{L}_{\mathrm{global}}, l_{\mathrm{pre}}, f_{\mathrm{pre}})$$
变量解释：
- $l_{\mathrm{next}}$: 下一个小目标（比如“把锅端起来”）
- $\mathcal{C}_d$: **过程约束**（比如“移动中锅必须保持水平”）—— 对应 proactive 检测，防患于未然
- $\mathcal{C}_u$: **结果约束**（比如“锅必须在炉灶正上方”）—— 对应 reactive 检测，事后验收
- $\mathcal{F}_{\mathrm{VLM}}$: GPT-4o 模型
- $l_{\mathrm{pre}}, f_{\mathrm{pre}}$: 上一轮干了啥，失败了是因为啥

#### Step 2: 找“几何替身”
这是 paper 里最有技术含量的创新。你让 VLM 去判断“锅平不平”，太勉为其难。那就把锅抽象成一个“面”，把龙虾抽象成几个“点”。

他们训练了一个叫 **ConSeg** 的模型，基于 LISA [27] 改的。输入一句约束 $c$ 和一张图 $o$，它吐出相关的 mask 和几何类型：
$$l_{\mathrm{e}}, h_{<\mathrm{SEG}>} = \mathcal{F}_{\mathrm{VLM}}(c, o), \quad \mathcal{M} = \mathcal{F}_{\mathrm{dec}}(\mathcal{F}_{\mathrm{MLP}}(h_{<\mathrm{SEG}>}), \mathcal{F}_{\mathrm{enc}}(o))$$
变量解释：
- $c$: 约束文本（如“面包必须留在锅里”）
- $\mathcal{F}_{\mathrm{VLM}}$: 里面的 LLaVA
- $h_{<\mathrm{SEG}>}$: VLM 输出 token 里 `<SEG>` 这个特殊标记的最后一层隐向量，里面蕴含了“该切哪里”的语义信息
- $\mathcal{F}_{\mathrm{MLP}}$: 一个多层感知机，把 $h_{<\mathrm{SEG}>}$ 翻译成 SAM decoder 认识的格式
- $\mathcal{F}_{\mathrm{enc}}(o)$: SAM [26] 的图像 encoder 提取的视觉特征
- $\mathcal{M}$: 最终输出的 segmentation mask
- $l_{\mathrm{e}}$: VLM 额外输出的几何类型（点、线、面）

拿到 mask 后，用 RGB-D 的 depth 信息把它反投影到 3D 空间，用 DBSCAN 聚类挑出几个代表点，连成 constraint element $\mathcal{E}$。比如“锅面”就取 3-4 个点连成一个多边形，“龙虾”取 2 个点连成一条线段。然后把这些 3D 几何体打上数字标号，画回原图上。

#### Step 3: 写代码当保安
把带标号的图给 GPT-4o 看，让它写一段 Python 代码。这段代码的输入是这些标号点的 3D 坐标，里面全是一堆 NumPy 算术运算，比如计算锅面法向量和 Z 轴的夹角。输出就是一个 bool 值和失败原因。

执行任务时，用 CoTracker [24] 实时追踪这些 2D 标号点，配合 depth 换算成 3D 坐标喂给代码。一旦代码算出夹角 > 15 度，立刻返回 False，机器人当场急停，拿着失败原因回去重新规划。**全程再也没 VLM 什么事了。**

---

### 3. 为什么“几何替身”这么神？（Ablation 解读）

这其实是 Table 6 那张消融实验表的核心价值。我用人话翻译一下：

- **不搞多视角，只用单视角 (MV=✗)**：成功率从 65% 掉到 40%。原因很直白——从正前方看，一个“面”在 2D 图里退化成了一条“线”，VLM 写代码时就懵了，把面当线算，全错。
- **不用 ConSeg，用 DINOv2 瞎找点 (CS=✗)**：成功率掉到 42.5%。DINOv2 [45] 只能靠特征聚类找点，它不懂任务。你要找“书边缘上垂直的两个点”，它给你找一堆乱七八糟的点，根本没法算垂直度。
- **不把点连成几何体 (CP=✗)**：成功率掉到 55%。如果只给代码一堆散乱的 3D 点，VLM 在写代码时就很难把“这三个点代表锅面”这种 prior 结构化进去，算几何关系容易乱套。

直觉就是：**你帮 VLM 把物理世界的几何结构抽象得越干净，它写的监控代码就越靠谱。** 

---

### 4. 实验数据里的“打脸”时刻

在 OmniGibson 的倒茶任务里，DoReMi 的成功率直接是 **0%**。为什么？因为茶壶倾斜这种事，单看一张 RGB 图，VLM 的视觉系统根本分辨不出那点微小的角度变化。而 CaM 用代码实时算 surface normal，又快又准。同时，因为 VLM 只在开头调用一次，token 消耗比 DoReMi 暴降了 52.2%。

最让我觉得有启发的是 Real-world 的 Table 4。Dex-GraspNet 2.0 [72] 是个 open-loop policy，人一干扰，它就傻乎乎照着原计划抓空气，成功率 0%。加上 CaM 后，代码实时计算动物和水果的 3D 距离，人把水果拿走了，距离变了，代码立刻报警重规划。**相当于用一个纯视觉的外挂，硬生生把一个 open-loop 的傻 policy 给闭环化了，成功率拉到 90%。**

---

### 5. 我的直观评价

这个 paper 的核心 reformulation 就是：**承认 VLM 在细粒度时空感知上不行，但擅长语义抽象和写代码。那就让 VLM 去做几何抽象和写代码，把判断真假的活儿全丢给确定性的代码执行。**

这跟早期的 Code as Policies [35] 思路很像，只不过 Code as Policies 是用 code 生成 action，CaM 是用 code 做 runtime monitor。这种 “decoupling perception precision from VLM” 的设计模式非常聪明，在现在的 robotics LLM 圈子里极有推广价值。

当然，它也有明显的短板。比如没位移的 force-direction 失败（死死卡住抽屉拉不开），光靠点云追踪的代码是算不出来的；再比如 threshold（15度这种）是 hardcode 的，换个奇怪的锅可能就不适用了。

但总体而言，这个几何抽象 + 代码监控的范式，非常 build intuition。

参考链接：
- LISA (ConSeg 的基础架构): https://arxiv.org/abs/2308.00674
- CoTracker (点追踪模型): https://arxiv.org/abs/2407.07624
- DoReMi (被暴打的 baseline): https://arxiv.org/abs/2402.14635
- Code as Policies (思想渊源): https://arxiv.org/abs/2209.07753

---

# Code-as-Monitor 深度解析

Andrej，这篇 paper 我读完之后直觉上的核心 take-away 是：**把"看图判断失败"这件事从 perception 问题转换为 geometry + code evaluation 问题**。这个 reformulation 解决了一个根本矛盾 —— VLM 擅长语义理解但不擅长实时、精确、3D 的 spatio-temporal reasoning。

---

## 1. 问题动机的深层逻辑

现有 closed-loop robotic system 的 failure detection 面临两个维度的张力：

**Reactive detection（事后识别）** 的痛点：DoReMi [16] 这类方法反复用 VLM 做 VQA 查询。问题在于 VLM 看单张 RGB 图判断"绿块是否在红块上"会出错 —— 因为单图缺乏 3D 深度信息和时空连续性。Table 1 的 Stack in Order 实验里，DoReMi 误判后反复重做上一个 subgoal，浪费时间。

**Proactive detection（事前预防）** 几乎没人做：要预判"pan 倾斜可能导致 lobster 掉出"，必须实时高精度监控 pan 的法向量与重力轴的夹角，这种 1cm/15° 级别的判断，靠 VLM VQA 根本做不到。

把两者**统一为 spatio-temporal constraint satisfaction problem**，再用 VLM-generated code 当 runtime evaluator，是这个 paper 的核心 reformulation。

参考：DoReMi paper https://arxiv.org/abs/2402.14635
ReKep paper https://arxiv.org/abs/2409.01614

---

## 2. Framework 三模块解析

### 2.1 Constraint Generator

$$l_{\mathrm{next}}, \mathcal{C}_d, \mathcal{C}_u = \mathcal{F}_{\mathrm{VLM}}(\mathcal{O}, \mathcal{L}_{\mathrm{global}}, l_{\mathrm{pre}}, f_{\mathrm{pre}})$$

变量：
- $\mathcal{O}$: multi-view RGB-D observations（front + top 两个视角）
- $\mathcal{L}_{\mathrm{global}}$: long-horizon task instruction
- $l_{\mathrm{pre}}$: 上一轮 subgoal
- $f_{\mathrm{pre}}$: 上一轮 failure feedback（成功 or 失败原因字符串）
- $l_{\mathrm{next}}$: 下一 subgoal
- $\mathcal{C}_d = \{c_d^0, \dots, c_d^n\}$: **during execution** 必须保持的约束（例如 pan 在 transfer 时必须保持水平 → proactive 监控对象）
- $\mathcal{C}_u = \{c_u^0, \dots, c_u^k\}$: **upon completion** 必须满足的约束（例如 pan 中心必须正对 stove 中心 → reactive 监控对象）

这个 $\mathcal{C}_d$ vs $\mathcal{C}_u$ 的二分是统一 reactive 和 proactive 的关键 —— 同一个 framework，不同时间点检不同约束集合。

### 2.2 Constraint Painter（最有技术含量的部分）

这一步把 textual constraint $c$ 映射到 image 上的 constraint elements $\mathcal{E} = \{e^0, \dots, e^{n+k}\}$。

**ConSeg 架构**（基于 LISA [27]）：

$$l_{\mathrm{e}}, h_{<\mathrm{SEG}>} = \mathcal{F}_{\mathrm{VLM}}(c, o), \quad \mathcal{M} = \mathcal{F}_{\mathrm{dec}}(\mathcal{F}_{\mathrm{MLP}}(h_{<\mathrm{SEG}>}), \mathcal{F}_{\mathrm{enc}}(o))$$

变量：
- $c$: textual constraint（如 "bread must remain in the pan"）
- $o$: RGB image
- $\mathcal{F}_{\mathrm{VLM}}$: LLaVA [38]
- $h_{<\mathrm{SEG}>}$: VLM 输出 token 序列中 `<SEG>` 这个 special token 的最后一层 hidden embedding
- $\mathcal{F}_{\mathrm{MLP}}$: 把 $h_{<\mathrm{SEG}>}$ 投影到 SAM decoder 所需的 prompt embedding space
- $\mathcal{F}_{\mathrm{enc}}$: SAM [26] 的 image encoder（frozen）
- $\mathcal{F}_{\mathrm{dec}}$: SAM 的 mask decoder
- $\mathcal{M}$: 输出的 segmentation mask
- $l_{\mathrm{e}}$: VLM 额外输出的 element type 描述（point/line/surface），作为 part-level segmentation 的 text response

直觉：VLM 不只输出"哪里相关"，还输出"是点/线/面哪种几何抽象"。这个 $l_{\mathrm{e}}$ 是后续 voxelization 决定 voxel size 的依据（surface 需要 ≥3 个点，所以划 $2\times2$ voxel；line 需要 2 个点；point 需要 1 个点）。

**Pipeline**:
1. ConSeg 输出 instance-level mask $\mathcal{M}_i$ + part-level mask $\mathcal{M}_p$ + type $l_e$
2. 用 depth 把 mask 投影成 3D point cloud
3. Voxelization（voxel size 由 $l_e$ 决定）
4. DBSCAN 聚类 + 每个 voxel 取一个代表点 + 按类型过滤到指定数量
5. 在 $\mathcal{M}_i$ 内部 connect points 形成 element $e$（point 是单点，line 是两点的连线 segment，surface 是 ≥3 点的 polygon）
6. 多视角并行推理，合并 3D element，回标到各视角图像，得 $\mathcal{O}_\mathcal{E}$

**End-effector 特例**：dexterous hand 的 fingertip、palm center 等点直接从 forward kinematics 取，跳过 segmentation pipeline。这是合理的简化。

### 2.3 Constraint Monitor

GPT-4o [1] 输入：$l_{\mathrm{next}}$ + $\mathcal{C}$ + $\mathcal{O}_\mathcal{E}$（带数字标号和 unique color 的多视角 image）→ 输出 Python monitor code。

代码特征：
- 输入：elements 的 3D positions（current + historical trajectory）
- 主体：NumPy 算术运算
- 输出：boolean flag + failure reason string
- tolerance 由代码内部 hardcode（如 pan surface normal 与 z-axis 夹角 > 15° → fail）

执行机制：CoTracker [24] 实时追踪 elements 的 2D 位置，配合 depth 反投影成 3D。**VLM 只在 subgoal 开始时调一次**生成 code，之后 runtime 完全执行 code，不调 VLM。这把 inference cost 从 $O(T)$（每帧 VQA）降到 $O(1)$（每 subgoal 一次 code 生成）。

代码可靠性：用 White-box Testing 在 subgoal 开始时验证每个 if-else 分支的 path coverage，错误就 regenerate。

参考：LISA https://arxiv.org/abs/2308.00674
CoTracker https://arxiv.org/abs/2407.07624

---

## 3. 关键 Intuition：为什么 constraint elements 而不是 raw keypoints

这个 ablation（Table 6）信息量很大，我把每行的逻辑梳理一下：

| MV | CS | CP | Avg SR | 直觉解释 |
|---|---|---|---|---|
| ✗ | ✓ | ✓ | 40.0% | 单视角导致 (1) occlusion 下 element 缺失 (2) dimension collapse：surface 在 front view 下被压成 line，line 被压成 point，code 生成时类型识别错 |
| ✓ | ✗ | ✓ | 42.5% | 用 DINOv2 [45] 提 semantic keypoints 替代 ConSeg。DINOv2 基于特征聚类找点，无法精确选"book 垂直边上的两个点"这种 task-specific keypoint |
| ✓ | ✓ | ✗ | 55.0% | 不 connect points，单独 3D 点作为 element。code 生成时 VLM 看到一堆散点，需要在脑中重新组合成有意义的几何关系，正确率下降 |
| ✓ | ✓ | ✓ | 65.0% | 完整方法 |

**深层 insight**：pre-formed elements 里 encode 了 prior constraint info（"这两点构成 line"作为 visual cue），等于把 constraint structure 显式地传给 VLM 让它写 code，而不是让 VLM 一边看图一边推断几何关系。这是从 perception → structured representation → code generation 的两步抽象，每步都降低 VLM 的负担。

参考：DINOv2 https://arxiv.org/abs/2304.07193

---

## 4. 实验数据的关键观察

### 4.1 CLIPort Stack in Order with noise q=3（最严苛）

| Method | SR | Time(s) |
|---|---|---|
| CLIPort (CP) | 15.00 | — |
| + Inner Monologue [20] | 40.00 | 58.2 |
| + DoReMi [16] | 40.00 | 54.2 |
| + CaM | **63.33** | **36.8** |

q=3 表示放置位置有 [0, 3] cm 均匀噪声。CP 是开环 policy 直接崩。IM 只在 subgoal 结束时检测，已经堆完了才发现塌。DRM 反复 VQA 检测慢且易误判。**CaM 在堆放绿块时就 proactively 检测到位置偏差大、继续堆蓝块会塌，于是先稳住绿块再继续**，所以 SR 高、Time 短。这是 proactive 检测的典型胜利。

### 4.2 OmniGibson Pour Tea（surface-level）

| Method | SR (None/Dist.a/b/c) | Token (k) |
|---|---|---|
| ReKep | 20/20/20/10 | — |
| +DRM | 0/0/0/0 | 44.19 |
| +CaM | (表中未列全) | — |

DoReMi 全 0% 的原因发人深省：teapot 的 pitch/roll 角变化从单张 RGB 看不出来，VLM VQA 判断不了。**CaM 用 monitor code 计算 surface normal 与 z-axis 的夹角，能精确检出**。这印证了 paper 的核心论点：code 在数值精度上碾压 VLM perception。

Token：CaM 比 DRM 减少 52.2%。因为 DRM 每帧调 VLM，CaM 每 subgoal 一次。

### 4.3 Real-world Reasoning Pick & Place（Table 4）

"Grasp the animals according to their distances to fruits, from nearest to farthest" 这种任务，open-loop policy DGN 单独完全失败（0%），DRM 也失败（0%），CaM 拿到 90%。

关键场景：人移动 horse 或 pear 时，"nearest" 关系动态变化。CaM 的 monitor code 实时计算 distance(animals, fruits)，重新选择 grasping target，相当于用 reactive + proactive detection 把 open-loop policy 变成 closed-loop。这是 paper 把 method 卖给社区的核心 narrative —— 任何 open-loop policy 加上 CaM 都能闭环化。

### 4.4 Segmentation Benchmark（Table 5）

ConstraintSeg part-level gIoU/cIoU：

| Method | gIoU | cIoU |
|---|---|---|
| LISA-13B | 23.4 | 24.3 |
| PixelLM | 24.1 | 22.6 |
| FMC (GPT-4o + Grounded SAM + Semantic SAM) | 40.8 | 39.3 |
| **ConSeg-13B** | **60.2** | **65.3** |

ConSeg 在 part-level 上比 LISA 高 ~40 gIoU，比 pipeline 拼接的 FMC 高 ~20。说明 end-to-end 训练 + 多粒度数据（Table 7）确实把"constraint-aware 的 part 是什么"内化进了模型。这是 abstraction layer 的关键 enabling 技术。

---

## 5. 与 ReKep 的本质差异

ReKep [21] 也用 keypoint constraint，但是：
- ReKep 用 DINOv2 cluster 出 semantic keypoint，generic 不 task-aware
- CaM 用 ConSeg 输出 task-conditioned 的 part mask → 3D element，更精确

paper 里给的例子：要捕捉 book 的垂直朝向，需要 book 边缘上精确两点，DINOv2 给不出。ConSeg 因为是 task-conditioned（输入就是 constraint 文本），能精确选出"book 垂直边"。

另一个差异：ReKep 的 keypoint 直接喂给优化器做 closed-loop control。CaM 的 element 喂给 code 做 failure detection，policy agnostic。这两个 design choice 决定了 CaM 可以叠加任何 open-loop policy 之上做检测，而 ReKep 本身就是个 policy。

---

## 6. Limitations 与我对它的批评

paper 自己承认的：
1. **Force-direction 失败检测不到**：gripper 拉不开 drawer 这种没明显位移的失败，element tracking 看不出来。这是 representation 的本质缺陷 —— 元素抽象掉了 force 信息。
2. **多模态丢失**：流动的水、半关的水龙头声音都被简化掉。
3. **VLM hallucination** 仍在 code 生成和 ConSeg segmentation 中存在。

我自己补充几点批评：
- **依赖 RGB-D**：ConSeg 用 depth 投影 3D，但很多真实场景 depth 传感器对透明物体、反光物体不准。Table 3 显示 CaM 在 transparent 物体上 SR 90-93%，已经不错，但绝对值仍非完美，可能就是 depth noise 导致。
- **Subgoal granularity**：约束监控只在 subgoal 层级。如果 subgoal 太长，中间过程的 proactive detection 仍可能滞后。
- **Element type 依赖 VLM 输出**：ConSeg 输出 $l_e$ 决定 voxel size 和点数。如果 VLM 误判 surface 为 line，element 几何就错了。这个 failure mode 在 ablation Table 6 的 "w/o MV" 行已经被验证（维度退化导致类型误判）。
- **Threshold 是 hardcode**：paper 说用 "external knowledge base" 或 "VLM 内部知识" 定 threshold。但 15° 这种数字在不同任务语义下含义不同，缺乏自适应机制。

---

## 7. 数据集细节（值得学习的地方）

ConSeg 训练数据：
- 来源：BridgeData V2 [64]（60,096 trajectories，24 environments，13 skills）
- 处理：用 GPT-4o 分解 trajectory-level instruction 为 subgoals + constraints + object-part associations
- 利用 gripper open/close 状态把 frame 分配到 Approach/Grasp/Transfer/Place 四阶段
- Grounded SAM [53] 做 instance mask，Semantic SAM [32] 做 part mask
- 人工 filter 后得 10,181 trajectories, 219,356 images

训练用 8×H800 GPU 2 天，LoRA 微调，VLM 和 SAM encoder 都 frozen。

**Sim-to-real 处理**：simulation 中用 ConSeg-ft（每个 simulator 收 100 trajectory 微调），real-world 用 ConSeg-base 不微调，证明 generalization。

参考：BridgeData V2 https://arxiv.org/abs/2308.12952
Grounded SAM https://arxiv.org/abs/2401.14159
Semantic SAM https://arxiv.org/abs/2307.04767

---

## 8. 我对 method 整体的直觉总结

把整篇 paper 浓缩成一句：**VLM 的角色从"perceive and judge" 退化为 "abstract and code"，把 perception 的精度负担转移给了 code 的算术运算 + 几何抽象**。

这个 reformulation 的本质是把 VLM 用在它擅长的地方（语义推理、code 生成、constraint 生成），把 VLM 不擅长的地方（实时、3D、精确数值判断）让位给 deterministic 的 code execution 和 tracking model。

这个思路其实和 Code as Policies [35] 一脉相承，但 Code as Policies 用 code 直接生成 action，CaM 用 code 做 runtime monitor —— 是把"code-as-X"范式从 control 扩展到 verification。

参考：Code as Policies https://arxiv.org/abs/2209.07753

对你做 robotics LLM/VLM 工作的可能启示：
- **Decoupling perception precision from VLM** 是一个值得推广的设计模式
- **Geometric abstraction layer** 在 sim-to-real、object generalization 上都起作用，可以借鉴到其他 task
- **Multi-view 至关重要**这个 ablation 警示：单视角 + VLM 哪怕代码生成得再好，dim collapse 也是硬伤

如果你想 build intuition，我建议看的几个最关键的图：
- Figure 2（pipeline overview）
- Figure 4（ConSeg architecture）  
- Figure 16-18（OmniGibson 三种 disturbance 的 detection demo，能直观感受 reactive vs proactive 的差异）
- Table 6 + Table 11（两个 ablation 表，信息密度最高）

paper project page: https://zhoues.github.io/Code-as-Monitor
