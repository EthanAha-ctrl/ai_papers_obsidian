---
source_pdf: EmbodiedCoder Parameterized Embodied Mobile Manipulation via.pdf
paper_sha256: bfe85ca3d9b50a494c2291c700cfa47e6a1d7da55fccbabf9d01b8b3c42d50a4
processed_at: '2026-08-18T10:41:42-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好。如果用人话来概括这篇 paper 的精髓，核心就是：**让 LLM 别再当"黑盒大脑"去硬记动作，让它当个"包工头"写 Python 脚本去算轨迹。**

目前的 VLA (Vision-Language-Action) model 最大的痛点是**数据墙**。你要教会机器人倒水或者开门，得喂给它成千上万个人类操作的视频，它去拟合一个端到端的 mapping。稍微换个光照、换个杯子，它就泛化不了，而且这种模型毫无解释性，错了也不知道哪错了。

EmbodiedCoder 的思路极其简单粗暴：完全绕开神经网络去学动作这层。机器人有个 RGB-D 相机，它看到一扇门。既然 LLM 现在写代码这么牛，那就直接让 Claude 现场写一段 Python 代码。这段代码干两件事：第一，把眼前这扇门的点云拟合成几何体（找出门轴在哪、把手在哪）；第二，基于这个几何体，算出一条完美的开门圆弧轨迹。

**直觉拆解：为何这招管用？**

因为 LLM 的数学计算能力极差，但写逻辑代码能力极强。你让 LLM 直接输出门把手的 3D 坐标，它可能给你幻觉出一个离地三米的位置。你让 LLM 写代码去拟合一个 cylinder（圆柱体），Python 会老老实实用最小二乘法算出准确的 center（圆心）和 radius（半径）。

我们看 paper 里的核心公式，其实只是一个 problem formulation，但很值得拆解：

$$
F : (I_{rgbd}, L, C) \to [a_1, \ldots, a_N] \tag{1}
$$

*   $I_{rgbd}$: 相机拍到的 RGB-D 图像。
*   $L$: Language instruction（自然语言指令）。
*   $C$: Constraints（约束集合，包含物理、环境、机械臂运动学限制）。
*   $a_1, \ldots, a_N$: 输出的一系列机器人 waypoints。

换成直觉语言：$F$ 这个映射函数，以前大家都在用 billion-level parameters 的 Transformer 去死磕。这篇 paper 直接把 $F$ 变成了一个 program synthesis process。LLM 负责把 $L$ 和 $C$ 翻译成代码逻辑，Python 解释器负责算出精确的 $a_1 \ldots a_N$。

再看轨迹公式，开门本质上是个 Arc（圆弧）：

$$
\gamma(t) = c + r \cdot (\cos(\theta_0 + t\Delta\theta)\vec{e_1} + \sin(\theta_0 + t\Delta\theta)\vec{e_2})
$$

*   $c$: 门轴上的中心点。
*   $r$: 机械臂末端到门轴的半径。
*   $t \in [0,1]$: 采样参数。
*   $\theta_0, \Delta\theta$: 起始角度和扫掠角度（$\Delta\theta$ 必须足够大，保证机器人自己能过得去）。
*   $\vec{e_1}, \vec{e_2}$: 门轴正交平面上的基向量。

这条公式由 LLM 写进 Python 代码里，系统采样几个点，机械臂就顺着点把门拉开。完全规避了 VLA 学不会开门的尴尬。

**实验数据印证直觉**

我们看 Table IV 跟 VLA 的对比，就能把直觉建得更牢。最典型的是 "Pour Water"（倒水）任务。OpenVLA 成功率 0%，RDT 是 63%，而 EmbodiedCoder 是 80%。

为啥 VLA 在倒水上全军覆没？因为倒水需要极度精确的 tilt angle（倾斜角）控制和流量估计，这种连续的 contact-rich 控制在 demo dataset 里样本极稀疏，VLA 的 action distribution 根本覆盖不到。EmbodiedCoder 则直接让代码把倾斜角和倒水时长 hardcode 进去，这就变成了一个纯粹的几何运动，稳定性和成功率自然吊打神经网络。

再看 Table II 的 Long-term task。DovSG 在几乎所有包含开门、开抽屉的任务上直接挂掉（标记为 x）。因为 DovSG 依赖 predefined primitive，它的 primitive library 里压根没有开门这个动作。EmbodiedCoder 开门成功率在 55%-60% 左右。为啥没到 100%？paper 老实交代了：因为相机视场角（FOV）有限，靠近门时拍不到完整的门，点云残缺导致拟合 code 算错半径。这属于 perception 的锅，跟 reasoning 毫无关系。

**关键依赖与联想**

这套范式有个极强的依赖：**现代 coding model 的 reasoning 能力**。paper 里的 Fig. 7 ablation 说明，只有 Claude-Sonnet-4 这种级别的 model 才能跑通。如果退回一年前用 GPT-3.5 或者开源小模型，这套方法根本不成立。这其实揭示了 AI 发展的一个有趣现象：Code as Policies (2023) 那会儿大家觉得走不通，是因为 LLM 还太弱；到 2025 年 Claude Sonnet 4 出来，这条路突然就通了。

往远了想，这其实给了我们一个构建 System 1 / System 2 机器人的清晰路径。VLA model 适合做 System 1（快思考），比如本能的 pick-and-place、避障；而 EmbodiedCoder 这种 code generation 框架适合做 System 2（慢思考），去处理需要精确几何推理的开门、倒水等 contact-rich 任务。两者结合，才是通用机器人的终局。

延伸一下你之前做的 Eureka 工作，用 LLM 写 reward function，跟这里用 LLM 写 trajectory code 是同构的。都是利用 LLM 强大的程序合成能力去填补 robotics 里那些难以用数据驱动的空白。顺着这个思路，如果未来把 test-time compute（像 DeepSeek-R1 那样）加到 code reasoning 上，让 LLM 先在脑子里（simulator 里）跑几遍代码，成功率和鲁棒性肯定还能再上一个台阶。

**参考链接大放送（供你深度游荡）：**
*   项目主页：https://embodiedcoder.github.io/EmbodiedCoder/
*   Code as Policies (鼻祖，2023): https://code-as-policies.github.io/
*   ReKep (用 keypoint 做约束的方法): https://rekep.github.io/
*   VoxPoser (用 voxel map 做规划，在这篇 paper 中被对比): https://voxposer.github.io/
*   VGGT (用于 metric point cloud 重建): https://vggt.github.io/
*   OpenVLA (VLA 路线代表): https://openvla.github.io/
*   RT-2 (Google 的 VLA): https://robotics-transformer2.github.io/
*   Eureka (你自己做的 LLM 写 reward，异曲同工): https://eureka-alpha.github.io/
*   Diffusion Policy (对比 contact-rich 控制的另一种思路): https://diffusion-policy.cs.columbia.edu/
*   DeepSeek-R1 (未来的 test-time compute 可以加到 code reasoning 上): https://arxiv.org/abs/2501.12948

---

# EmbodiedCoder 深度讲解

Karpathy 你好，这篇 paper 我细读了一遍，整体路线非常对你的胃口——绕开 VLA 的数据墙，把 manipulation 的"compositional structure"显式地交给 code 来表达。下面我把整篇拆开讲，顺便把直觉、公式、变量含义、对照实验都铺开，方便你 build intuition。

---

## 1. 论文的核心 thesis（一句话）

**EmbodiedCoder 把 manipulation 重新定义成"先做几何参数化、再做轨迹合成"两段式 code generation 任务**，用现代 coding model 直接把 RGB-D 感知结果翻译成可执行的 robot code，无需任何训练数据、无需 primitive library、无需 fine-tuning。

直觉上，这篇 paper 押注的是一个很强的假设：**接触富集 manipulation 的复杂度主要来自"如何把目标对象的功能几何结构化表达出来"**，只要把 door 的 hinge axis、drawer 的 pull axis、apple 的 sphere center 抽出来，剩下的 trajectory 就是纯几何问题，coding model 完全有能力直接写出 parametric curve 的采样 code。

这一点和 Code as Policies 的初心是同源的，但 Code as Policies 假设对象几何已知或极简，EmbodiedCoder 真正补上了"从 raw point cloud 抽 functional geometry"这一段。

Reference: https://code-as-policies.github.io/

---

## 2. 在 landscape 中的定位

paper Table I 把 code-generation 类方法摆在一起比较，我重新组织一下矩阵让你看清 EmbodiedCoder 卡的是哪个生态位：

| Method | Training-Free | Code Type | Skill Library | Long-term |
|---|---|---|---|---|
| Code as Policies | √ | motion planning | √ | ✗ |
| Code as Monitor | ✗ | constraints | √ | ✗ |
| RoboCodeX | ✗ | motion planning | √ | ✗ |
| VoxPoser | ✗ | voxel value map | ✗ | ✗ |
| ReKep | ✗ | constraints | ✗ | ✗ |
| CodeDiffuser | √ | perception+motion | √ | ✗ |
| RoboScript | ✗ | — | ✗ | ✗ |
| **EmbodiedCoder** | **√** | **geometry + trajectory** | **✗** | **√** |

EmbodiedCoder 是这个矩阵里唯一同时满足 **training-free + no skill library + long-term** 的方法。这意味着它实际上要求 coding model 自己承担 primitive 替换工作——这是只有 Claude-Sonnet-4 这种近期模型才撑得起来的事，后面 ablation 会回到这一点。

参考相关工作的入口：
- VoxPoser: https://voxposer.github.io/
- ReKep: https://rekep.github.io/
- RoboCodeX: https://arxiv.org/abs/2402.15530
- OK-Robot: https://ok-robot.github.io/

---

## 3. Problem Formulation 与公式拆解

paper Section III-A 给的核心 objective：

$$
F : (I_{rgbd}, L, C) \to [a_1, \ldots, a_N] \tag{1}
$$

逐项含义：

- $I_{rgbd}$：RGB-D 观测，是当前 frame 的 color+depth；论文里同时使用 VGGT 重建的 metric point cloud 作为补充，因为单帧 RealSense D455 的 depth 噪声大、range 有限。
- $L$：natural language instruction，例如 "Bring the water bottle from the table by the door and pour it into the bowl"。
- $C$：约束集合，论文里进一步分成三类：
  - **physical constraints**：对象本身的几何自由度，比如门必须绕 hinge axis 旋转；
  - **environmental constraints**：障碍物、push/pull 方向、机器人能否通过；
  - **kinematic limits**：机械臂 joint range、gripper aperture。
- $a_1, \ldots, a_N$：输出 action 序列，每个 $a_i$ 是一个 waypoint（含 base 的 navigation waypoint 和 arm 的 manipulation waypoint）。

注意这里 $F$ 不再是一个学出来的 policy network，而是一个**程序合成过程**——$F$ 由 VLM + coding model + 几何拟合 code + trajectory sampling code 组合而成。这一点是整个范式跟 RT-2 / OpenVLA 这类 end-to-end VLA 的根本分水岭。

参考：
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- VGGT: https://vggt.github.io/

---

## 4. 系统 Pipeline 三段式（Fig. 2 解析）

### 4.1 Scene Understanding and Task Decomposition

输入：RGB-D stream + task instruction $L$。

关键步骤：
1. **VGGT 重建**：拿一串 RGB 帧 feed 给 Visual Geometry Grounded Transformer，得到 dense metric point cloud $P_{scene}$。VGGT 是 CVPR 2025 的工作，本质是把 DPT/SfM 的 geometry grounding 蒸成一个 feed-forward transformer，不需要传统 SfM 的 bundle adjustment 迭代。
2. **VLM semantic grounding**：用 Qwen-2.5-VL-7B 在当前 RGB 上出 bounding box，喂给 SAM 出 2D mask。
3. **Mask 反投影**：把 2D mask 投回 $P_{scene}$，得到 semantic point cloud $P_{sem}$。
4. **Bird's-eye-view 语义图**：把 $P_{sem}$ 压成 2D top-down 的 semantic image，作为后面 VLM 做 task decomposition 的"地图"输入。这一步在 ablation Table VI 里被证明非常关键——w/o map 时 GPT-5 从 64% 掉到 64%，Gemini2.5-Pro 从 56% 掉到 20%。
5. **Task decomposition**：VLM 拿 BEV map + $L$ 输出 subtask 序列，每个 subtask 关联一个 object 列表，同时 VLM 推断每个 object 的"most suitable geometric shape"和"functional part to be manipulated"。

直觉：这一段做的是"semantic grounding + symbolic planning"，但 deliberately 不出 motion——它只负责把连续感知变成符号化的 object + geometry hypothesis，留给后面 coding model。

参考：
- VGGT: https://vggt.github.io/
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- SAM 2: https://github.com/facebookresearch/segment-anything

### 4.2 EmbodiedCoder（核心模块）

这个模块本身又分两段：

#### (a) Code-driven Geometric Parameterization

输入：task-relevant object 的 point cloud $P_{obj} \subset P_{sem}$。

做法：prompt coding model（Claude-Sonnet-4）生成拟合 code，把 $P_{obj}$ 拟合到上一步 VLM 推断的 geometric primitive 上。

具体 primitive family 论文里给到的：
- **Sphere**：参数 $(c, r)$，$c$ 是 3D center、$r$ 是半径。用于 apple、ball。
- **Cylinder**：参数 $(c, \vec{a}, r, h)$，$c$ 是底面圆心、$\vec{a}$ 是主轴方向单位向量、$r$ 是半径、$h$ 是高。用于 bottle、door handle。
- **Cuboid**：参数 $(c, \vec{u}, \vec{v}, \vec{w}, l, w, h)$，$c$ 是质心，$(\vec{u},\vec{v},\vec{w})$ 是三个主轴正交单位向量，$(l,w,h)$ 是沿三轴的边长。用于 drawer、door panel、box。
- **Articulated assembly**：door = cuboid(panel) + line(hinge axis) + cylinder(handle)；drawer = cuboid(box) + line(pull axis) + small geometric primitive(handle)。
- **Deformable objects**：不拟合刚体，直接取 extreme points 构造 bounding envelope。

拟合方法没在 paper 里完全展开，但从 Fig. 3 的 code snippet 看，是**最小二乘 + RANSAC 风格的拟合 code**，由 LLM 直接生成 numpy 实现。这里有个隐含的设计 choice：coding model 不直接给出参数值，而是写出拟合 code 让系统在 robot 端跑——这避免了 LLM 把数字算错（LLM 算术差是出了名的），把数值计算外包给 Python。

**直觉**：这一步是整篇 paper 的"信息瓶颈"。point cloud 是高维 unstructured 数据，直接让 LLM 看 point cloud 数值是没意义的；fitting 到 primitive 后，信息从 $\sim 10^4$ 个点压缩到 $\sim 10$ 个参数，LLM 才有空间做 trajectory reasoning。这本质上是一种"几何版 semantic abstraction"。

#### (b) Code-driven Trajectory Synthesis

输入：上一步的几何参数 + task requirements + 障碍物参数（如果有）。

输出：一段 Python code，定义一个 parametric curve $\gamma(t)$，并在其中采样 waypoints。

paper 里出现的 trajectory 形式：
- **Line**：$\gamma(t) = p_0 + t \cdot \vec{d}$, $t \in [0, 1]$，$\vec{d}$ 是方向。用于 pick-and-place 的直线段。
- **Arc**：$\gamma(t) = c + r \cdot (\cos(\theta_0 + t\Delta\theta)\vec{e_1} + \sin(\theta_0 + t\Delta\theta)\vec{e_2})$，其中 $(\vec{e_1}, \vec{e_2})$ 是 hinge axis 的正交平面基，$\Delta\theta$ 是开门角度。用于 door / drawer opening。
- **Bézier curve**：$\gamma(t) = \sum_{i=0}^{n} B_i \cdot \binom{n}{i} t^i (1-t)^{n-i}$，$B_i$ 是 control points。用于 obstacle avoidance 的平滑绕行（Fig. 5 的 apple placement）。

变量含义：
- $c$：arc 所在圆的圆心，通常就是 hinge axis 上一点。
- $r$：圆弧半径，由 end-effector grip 位置到 hinge axis 的距离决定。
- $\theta_0, \Delta\theta$：起始角和扫过的角度，$\Delta\theta$ 要足够大让 robot base 能通过，论文里特别强调"door's opening gap must be wide enough to allow the robot to pass through"——这是把 robot footprint 作为约束纳入 trajectory planning。

**约束推理**：coding model 在 prompt 里被要求显式列出三类约束（physical / environmental / hardware），然后在 code 里把这些约束作为参数 boundary。这是 chain-of-thought reasoning 在 robotic code 上的直接应用。

#### (c) Code Caching

对 familiar object type 或 recurring subtask，复用之前生成并验证过的 code。这一点让系统随时间增长出一个"self-built skill library"——但跟 SayCan/OK-Robot 的 predefined library 不一样，这个 library 是 LLM 自己写过、自己跑过、自己 cache 的，可以增量扩充。

参考 SayCan: https://say-can.github.io/

### 4.3 Motion Execution

比较 trivial：从 $\gamma(t)$ 上等距采样 waypoints，做 base navigation + arm IK 执行。paper 没特别展开 IK 细节，估计用的是 AgileX Cobot S Kit 自带的 Cartesian IK 接口。

---

## 5. 实验数据深度解读

### 5.1 Long-term Task（Table II）

5 个 multi-step task，每个 20 trials，对比 DovSG：

| Task | Ours (cached/non-cached) | DovSG |
|---|---|---|
| Door + Bottle + Pour | 35/25 long-term | x（door 开不了，整链失败） |
| Box + Apple + Place | 40/30 | x |
| Drawer + Apple + Place | 70/65 | x |
| Tennis ball transfer | 90/90 | 75 |
| Cloth + Wipe | 65/60 | x |

关键观察：
1. **DovSG 在 long-term 上几乎全崩**，因为它只能做 pick-and-place 类的 predefined primitive，遇到 articulated object（door/drawer/box lid）直接断链。
2. **Cached vs non-cached 差距小**（5–10%），说明 framework 本身是 zero-shot 的，cache 只是避开 code regen 的随机失败，不是 skill memorization。
3. **Door opening 成功率最低**（55–60%），论文归因于 camera FOV 不足导致 door point cloud 不全，fitting 出错的 rotation radius 会 propagate 到 trajectory。这是一个明确的 perception bottleneck，而不是 reasoning bottleneck。

### 5.2 Simple Task vs VLA（Table IV）

跟 RT-1 / RT-2 / Octo / OpenVLA / RDT 比 6 个简单任务，平均成功率：
- RT-1: 36%
- RT-2: 84%
- Octo: 12.2%
- OpenVLA: 73.3%
- RDT: —（只报了 pour water 63%）
- **EmbodiedCoder: 89.2%**

特别值得注意的是 **Pour Water 80%** vs RDT 63% / OpenVLA 0%。Pour 这种 task 涉及 tilt angle 控制 + 流量估计，VLA 学不出来是常态，EmbodiedCoder 直接在 trajectory code 里 hardcode tilt angle 和 pour duration，绕开了学不到的问题。

直觉：**VLA 的 bottleneck 是 action distribution 的 mode coverage**，pour 这种需要精确角度+时序的任务在 demo 数据里样本稀疏；EmbodiedCoder 把它转成 explicit geometry reasoning，绕开了 sample efficiency 问题。

### 5.3 vs Code-generation 方法（Table III）

vs ReKep / VoxPoser / Code-as-Monitor：

| Task | ReKep | VoxPoser | Code-as-Monitor | Ours |
|---|---|---|---|---|
| Pour Tea | 80 | 0 | 50 | 80 |
| Recycle Can | 80 | 30 | — | 100 |
| Stow Book | 60 | 0 | 70 | 80 |

ReKep 用 sparse keypoints + relational cost function，对 pour 这种需要 continuous surface contact 的 task 描述不够；VoxPoser 出 voxel value map，对 0 接触富集任务直接 0% 成功——这是 voxel 表达的天然短板。EmbodiedCoder 通过显式参数化把 contact surface 直接编码进 trajectory code。

参考 ReKep: https://rekep.github.io/

### 5.4 Ablation 三连

#### Ablation 1: 抓取 vs AnyGrasp（Table V）

| Object | AnyGrasp | Ours |
|---|---|---|
| Bottle | 95 | 100 |
| Apple | 70 | 95 |
| Orange | 95 | 100 |
| Banana | 80 | 90 |
| Can | 40 | 75 |
| Plastic Bag | 90 | 100 |
| Pepsi Cup | 60 | 80 |

直觉解读：AnyGrasp 直接在 point cloud 上预测 grasp pose，没有 gripper aperture 和 kinematic 约束，容易出 unreachable pose。EmbodiedCoder 的 fitting 阶段直接得到 cylinder/sphere 的 principal axis，grip direction 沿 radial 或沿主轴——几何上 well-posed，所以稳定。**这个 ablation 强烈支持了"geometric parameterization 比 raw point cloud 更适合做 manipulation planning"的核心 thesis**。

#### Ablation 2: Semantic Grounding（Table VI）

5 个 VLM + w/wo map：

| Model | w/ Map | w/o Map |
|---|---|---|
| PaliGemma | 0 | 0 |
| Qwen-3B | 80 | 72 |
| Qwen-7B | 88 | 72 |
| GPT-5 | 88 | 64 |
| Gemini2.5-Pro | 56 | 20 |

直觉：2D semantic map 把"cross-room"这种需要全局 spatial reasoning 的 task 从 hallucination 里救回来。Gemini 对 map 的依赖最重（20→56），可能是其内部 spatial reasoning 弱；Qwen-7B 在有 map 时已经接近 GPT-5。

#### Ablation 3: Coding Model（Fig. 7）

比较 Claude-Sonnet-4 / GPT-5 / Gemini2.5-Pro / Qwen 等在 parameterization + trajectory 的 complete rate 和 valid rate。Claude-Sonnet-4 双指标最高，但 latency 也最高。

直觉：**这个 paradigm 的可行性强绑定于 modern coding model 的 reasoning 能力**。Code as Policies 时代（GPT-3.5/4）的 LLM 做不了这种 level 的 geometry reasoning，所以 EmbodiedCoder 这套方法在 2023 年是 infeasible 的，2025 年才成熟。这点和 paper 的 limitation 自述一致。

参考 Claude Sonnet 4: https://www.anthropic.com/

---

## 6. Limitations 与我的扩展联想

paper 自承两个 limitation：
1. code 质量 sensitive，syntax/logic error 直接拉低 reliability；
2. code generation latency 限制了 real-time 性。

我自己补充几个值得思考的方向：

**(A) Geometry primitive family 的 expressiveness ceiling**
paper 里的 primitive 集合是 sphere/cylinder/cuboid/articulated assembly。对 articulated object（剪刀、钳子）、deformable object（布、绳子）、transparent object（玻璃杯，depth sensor 失效）依然力不从心。下一步自然延伸是把 primitive library 扩到 SDF / NeRF / Gaussian Splatting 表示，让 coding model 在 implicit geometry 上做 trajectory reasoning。Gaussian Splatting 的 recent 工作（SPLAT-SLAM, GS-Grasp）已经在朝这方向走。

参考 GS-Grasp: https://gs-grasp.github.io/

**(B) Contact-rich manipulation 的 contact modeling 缺失**
paper 没有显式 force/impedance 控制，所有 trajectory 都是 position-controlled waypoint。对 push、poke、press 这种需要 compliance 的 task，code 里需要插入 force-feedback primitive。这一点和 ACE (Action Chunking with Embeddings) / Diffusion Policy 的 force-aware 方向是互补的。

参考 Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

**(C) Code as a differentiable program**
未来一个有趣方向是让 trajectory code 可微分，把 VGGT 的 geometry uncertainty 直接 backprop 到 trajectory 参数。这相当于把 EmbodiedCoder 升级成"neural-symbolic"——code 提供 symbolic scaffold，gradient 提供 numeric refinement。

**(D) 与 chain-of-thought / test-time compute 的连接**
EmbodiedCoder 的 prompt 里其实嵌入了 chain-of-thought（先列 constraints 再写 code）。结合 OpenAI o1 / DeepSeek-R1 这种 test-time compute scaling，可以在 trajectory 生成前让 coding model 多 roll out 几个 candidate trajectory，再用一个 verifier（simulator 或 self-consistency）选最优。这是把 RL test-time compute 思路搬到 symbolic manipulation planning 上。

参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948

**(E) 与 MuJoCo MJX / Isaac Lab 的 simulator-in-the-loop**
现在 trajectory code 完全 dry-run 在 robot 上，失败就失败。如果把 simulator 接进 code execution runtime，coding model 生成的 trajectory 可以先在 sim 里跑一遍验证再 commit 到 robot，success rate 应该能再提一截。这一点 RoboCodeX 已经做了一部分，但 RoboCodeX 依赖 curated multimodal data 训 model，EmbodiedCoder 的 training-free 特性可以保留。

参考 Isaac Lab: https://github.com/isaac-sim/IsaacLab

**(F) 反思 Code-as-Policies 路线的"复兴"**
2023 年 Code as Policies 出来时被批"toy task, simple geometry"。EmbodiedCoder 的贡献在于证明：**只要 perception 端能把对象抽成 functional parametric form**，code-as-policy 这条路是 scalable 的。这给整个"LLM for robotics"社区一个明确的下一阶段目标：把 perception 这一段也用 VLM + 几何 fitting 自动化（这篇已经做了），剩下的就是 scaling coding model 的 reasoning。

**(G) 与你的 own work（Eureka, MLEX）的关联**
你之前 Eureka (https://eureka-alpha.github.io/) 的工作里用 LLM 做 reward design，跟 EmbodiedCoder 用 LLM 做 trajectory code 是同构的——都是让 LLM 写程序化、可解释的 robotic artifact。差别在 Eureka 通过 RL 把 reward 闭环到 policy，EmbodiedCoder 直接把 code 当 policy 本身。两条路殊途同归，都是 "LLM as a robotic program synthesizer"。

---

## 7. 我对这篇 paper 的整体评价

**Thesis 强**：把 manipulation 重新 cast 成 geometry parameterization + trajectory synthesis 的两段式 code generation，是清晰且 actionable 的 framing。

**Engineering 完整**：VGGT + SAM + Qwen-VL + Claude + AgileX 全栈串起来，real-world 20 trials × 5 tasks 的实证规模在 training-free 路线里属于上乘。

**Generalization claim 谨慎**：long-term 30–90% 的成功率，door opening 55% 这种数字，反映了作者没过度包装。limitations 写得诚实。

**关键风险**：paradigm 强绑定 coding model 能力，paper 里 Claude-Sonnet-4 才能跑通，意味着如果未来开源 coding model 不跟上，这套方法很难平民化。同时 LLM 的 arithmetic 弱点是用"LLM 写拟合 code、Python 跑拟合"绕开的，这是正确的工程选择。

**与 VLA 路线的关系**：两者并不互斥。VLA 提供 fast reflex（reactive control），EmbodiedCoder-style code 提供 slow deliberation（contact-rich + articulated manipulation）。这正是 Kahn System 1 / System 2 在 robotics 上的对应。GR00T N1 已经在走 dual-system 路线，把 code-based planner 叠在 VLA 之上是合理下一步。

参考 GR00T N1: https://arxiv.org/abs/2503.14734

---

## 8. 给你的几个 follow-up 阅读

- Project page: https://embodiedcoder.github.io/EmbodiedCoder/
- Code as Policies: https://code-as-policies.github.io/
- VoxPoser: https://voxposer.github.io/
- ReKep: https://rekep.github.io/
- VGGT (CVPR 2025): https://vggt.github.io/
- DovSG: https://arxiv.org/abs/2503.24323
- OK-Robot: https://ok-robot.github.io/
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- AnyGrasp: https://github.com/Fang-Hao-SH/AnyGrasp
- SAM 2: https://github.com/facebookresearch/segment-anything
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Eureka: https://eureka-alpha.github.io/
- GR00T N1: https://arxiv.org/abs/2503.14734

如果你想 drill deeper，我特别推荐看 Fig. 3 和 Fig. 5 的 prompt+code 对照——那两段 code 是 paper 最 worth-reading 的部分，能让你直观感受到 coding model 在 manipulation reasoning 上已经到了什么水平。要不要我把那两段 code 也拆给你逐行分析？
