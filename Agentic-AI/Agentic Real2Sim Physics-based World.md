---
source_pdf: Agentic Real2Sim Physics-based World.pdf
paper_sha256: 949cf1d8fbfea37c8965440f558864ed17e11c53c1ba09701adbfa82382fabca
processed_at: '2026-08-18T00:11:21-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话

把一段真实robot操作视频,自动变出一个MuJoCo里能跑通的digital twin scene,这样你就能在sim里replay、改参数、训policy。整个过程VLM当"大脑"做判断,deterministic tools当"手脚"做perception和simulation。

---

## 这事儿到底为什么painful

你如果玩过3DGS或者NeRF,会有个错觉:real2sim已经solved了,重建一个scene看起来那么真实,扔进MuJoCo不就完了?

错。**"看起来像"和"能跑通"是两码事**。

一个DROID episode里,robot走过去gripper闭合,抓住一个mug,举起来放到plate上。你要在MuJoCo里replay这个,需要的东西远不止好看的mesh:

- mug的6D pose在每一帧是啥(不然gripper怎么知道往哪儿闭合)
- mug的mass是多少、material是啥(不然被抓起来的时候是甩出去还是稳稳夹住)
- 哪个surface是桌面(ground plane fit错了一切都飘在空中)
- robot base在world frame哪儿(放错位置整个trajectory全错位)
- camera怎么摆的(replay出来的视频要和real keyframe对得上才能validate)

这些信息**散落在video的每一帧里**,要一帧一帧extract,再assemble成一个runnable的MJCF file。手动做几个小时一个episode,且每个episode都得重做,因为object不同、camera角度不同、occlusion pattern不同。这就是为什么DROID有那么多数据,但真正能在sim里replay的比例很低。

---

## 他们的key insight

整个process拆开看,其实有两类完全不同的task:

**第一类:需要"判断"的**
- 哪个object是被manipulate的target
- 哪一帧的mask最干净(没occlusion、in focus)
- SAM出来的mask够不够好,要不要换一帧重试
- 哪个entity是ground reference(桌面?地板?)
- grasp失败了,object往哪个方向挪一点能救回来

这些是schema-constrained的discrete decision,输出是JSON、是enum、是yes/no。**这正是VLM擅长的**,不需要它做geometry不需要它做physics。

**第二类:需要"算"的**
- Stereo depth estimation(FoundationStereo)
- 6D pose tracking(FoundationPose)
- Mesh reconstruction(SAM 3D)
- Grasp sweep(MuJoCo rollout)

这些扔给deterministic tools,VLM不插手。

**这个decoupling让VLM变得pluggable**:你换GPT-5.4、Claude Haiku、Qwen、Gemma都行,因为VLM只做narrow judgment,不做heavy lifting。实验里31B的Gemma比GPT-5.4 success count还高5个,但cost便宜31倍。

---

## 架构里三个clever点

### 1. Critic loop with bounded retry

每个stage后面挂一个VLM critic:mask critic检查SAM出来的mask能不能用,tracking critic检查FoundationPose的track有没有drift,scale critic检查mesh尺寸合不合理。Reject就retry,但**有retry budget**(比如3次),不会无限循环。

这个设计很重要,因为foundation model都会犯false positive/negative的错。有critic在中间当gatekeeper,quality提升一截。

### 2. Simulator-in-the-loop grasp optimization

这是最clever的地方。不管FoundationPose多准,都有cm-level error。gripper要抓object,差几毫米就miss。你怎么救?

答案:让MuJoCo当oracle,grid search一小块position shift。具体说,object在nominal pose附近做$\Delta x, \Delta y \in [-2\text{cm}, 2\text{cm}]$, $\Delta z \in [-1\text{cm}, 1\text{cm}]$的grid shift,每个candidate跑一遍replay,看哪个shift让gripper成功抓住object且lift起来不飞走。

这是"closing the loop"——visual foundation model有error,simulator当反向refine的oracle来compensate。**没有这一步,success rate会暴跌**,因为gripper闭合位置差几毫米就miss。

更fancy的版本是LLM-assisted loop:把rendered keyframe和structured replay summary("object离gripper 2cm太远,gripper闭合时object滑出")喂给VLM,VLM提议"object往+x方向移1cm"。这是semantic版的Bayesian optimization。

### 3. Episode contract跨domain

他们定义的episode twin是个tuple:
$$\mathcal{T} = (\mathcal{O}, \mathcal{A}, \mathcal{G}, \mathcal{S}_{1:T}, \Theta, \mathcal{B}, \mathcal{M})$$

rigid manipulation、deformable、humanoid三个domain**共享这个contract**,只在每个元素的representation上differ:
- rigid:$\mathcal{S}_{1:T}$是object 6D pose over time
- deformable:$\mathcal{S}_{1:T}$是particle positions over time(MPM)
- humanoid:$\mathcal{A}$是全身humanoid joint,不是gripper

但pipeline stage、replay loop、critic interface都一样。这是让framework scale到多domain的关键,不然每个domain都要写一套独立system。

---

## Evaluation为什么是VLM-as-judge

"两个episode像不像"这件事,你用PSNR、IoU都不对。因为这是semantic判断:target object对不对、action对不对、final location对不对。这些pixel-level metric反映不出来。

所以他们用三个不同backend的VLM当judge,每个judge独立打分0-10,扣分规则:
- 错target object:-4
- 错final location:-3
- 错action:-2
- 错gripper location:-1

≥8算pass,只要一个judge给≥8就算episode成功($r_e=1$)。

**Three-judge ensemble**降variance,不同backend避免correlated error。**OR logic**避免某个judge太严导致false negative。**Deterministic candidate selection**(按peak displacement排序)让evaluation reproducible。

这种metric设计其实有点meta:用VLM solve real2sim,又用VLM judge real2sim做得好不好。但合理,因为"episode一致性"本来就是semantic概念。

---

## 实验告诉我们什么

100个DROID episode,四个VLM backend:

| Backend | Success | Cost |
|---|---|---|
| Gemma 4 31B | 48/100 | $2.62 |
| Qwen 3.6 35B | 45/100 | $13.10 |
| GPT-5.4 | 43/100 | $82.30 |
| Claude Haiku 4.5 | 37/100 | $9.17 |

两个takeaway:

**1. VLM不是bottleneck。** 31B的Gemma反而最高,GPT-5.4没占便宜。因为VLM只做narrow judgment,这个任务31B就够用了。换更贵的model是overkill。

**2. 真正的bottleneck在upstream visual和simulation。** 所有backend都<50% success,说明headroom不在VLM choice,在FoundationPose的drift、SAM的segmentation miss、MuJoCo grasp sweep的search space设计这些。Paper自己说得很明白。

**Cost结构也值得注意**:主要cost不是VLM API call,是simulator rollout。75个grasp candidate × 5秒 = 6分钟一个episode,这是真正的bottleneck。Scale到10k episode要1000 GPU hours,这才是要优化的地方。

---

## 我的honest assessment

**Strengths:**

1. **Problem formulation sharp**。Episode twin这个tuple把"什么是digital twin"从工程概念变成precise artifact。每个element都对应一个downstream需求,可以serialize、可以replay、可以query。这是unified world modeling的一小步。

2. **Decoupling设计elegant**。Deterministic tools + agentic decisions的separation让系统plug-and-play,VLM backend可换可ablate。这是让framework scale的关键。

3. **Cost故事强**。31B open model达到frontier model同等效果但cost 31×低,这对社区开源友好,别人能复现能iterate。

**Weaknesses:**

1. **Success rate < 50%是硬伤**。一半episode转不出来,limit downstream应用。Paper说headroom在upstream,但没ablate哪个component最bottleneck,这就让人不知道该optimize什么。

2. **Deformable和humanoid只有qualitative**。没quantitative score,你不知道这些adapter真的work还是只是demo level。这是open problem的掩饰。

3. **Closed-loop policy evaluation没做**。Paper说aim to use for policy learning and evaluation,但实验里只做了open-loop replay。Open-loop replay success不代表closed-loop policy在twin上训练后transfer到real能成功。这是最大的missing piece。

4. **Grasp sweep太暴力**。75个candidate grid search,能不能用AnyGrasp这种learned grasp proposal替代,把candidate降到5-10个?

5. **Critic的reliability没ablate**。Mask critic、tracking critic自己有false positive/negative rate,去掉critic success rate下降多少?Paper没说。

---

## 联想到的更大图景

这篇paper其实在回答一个很基础的问题:**real world data和sim world data之间的桥梁是什么**。

现在robot learning领域有两个trend在并行:
- **World model派**(Sora、Genie 2、DreamerV3):学implicit dynamics,从video里学"下一步会发生什么",generalizable但不精确
- **Physics sim派**(MuJoCo、Genesis、PhysTwin):explicit dynamics,精确但不generalizable,且real2sim转换是人工bottleneck

Agentic Real2Sim其实是给physics sim派补上real2sim自动化这一环。如果这条路能scale,可能让sim data的生成成本大幅下降,policy learning就有更多data可用。

更远一点想,如果这个framework真的work,未来的loop可能是:
1. Real robot跑episode → DROID-style recording
2. Agentic Real2Sim → MuJoCo episode twin
3. Twin里augment、randomize、generate variation
4. Policy在twin里训练
5. Policy deploy到real
6. 新的real episode → 回到step 1

这就是TwinAligner说的Real2Sim2Real closed loop。Agentic Real2Sim是这条loop里目前最weak的一环,这篇paper迈了一步,但还有很长的路。

---

参考链接:
- Project: <https://agentic-real2sim.github.io/>
- DROID: <https://droid-dataset.github.io/>
- FoundationPose: <https://nvlabs.github.io/FoundationPose/>
- PhysTwin: <https://phystwin.github.io/>
- BFM-Zero: <https://bfm-zero.github.io/>
- TwinAligner: <https://arxiv.org/abs/2512.19390>
- AnyGrasp: <https://anygrasp.github.io/>
- Sora: <https://openai.com/sora>
- Genie 2: <https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/>
- DreamerV3: <https://arxiv.org/abs/2306.14893>
- Genesis: <https://github.com/Genesis-Embodied-AI/Genesis>
- OpenVLA: <https://openvla.github.io/>
- π0: <https://www.physicalintelligence.company/blog/pi0>

---

# Agentic Real2Sim: 用 Vision-Language Agent 把真实交互 episode 转成可仿真 digital twin

这篇 paper 解决的核心问题是：把一段真实世界录制的 robot-object interaction episode，自动转换成 MuJoCo 里能跑的 digital twin，保留 observation、geometry、robot interaction、object state，从而支持下游的 policy learning 和 evaluation。我先从 problem formulation 入手 build intuition，再拆架构、公式、实验。

参考链接：
- Paper project page: <https://agentic-real2sim.github.io/>
- DROID dataset: <https://droid-dataset.github.io/>
- MuJoCo: <http://www.mujoco.org/>
- FoundationPose: <https://nvlabs.github.io/FoundationPose/>
- FoundationStereo: <https://nvlabs.github.io/FoundationStereo/>
- SAM 3 / SAM 3D (concept): <https://ai.meta.com/blog/segment-anything/>
- PhysTwin: <https://phystwin.github.io/>
- BFM-Zero: <https://bfm-zero.github.io/>
- EMPM: <https://arxiv.org/abs/2604.04xxx> (Chenfanfu Jiang group)

---

## 1. 为什么 Real2Sim 这件事 hard——从 plumbing 到 cognitive problem

一个 robot manipulation recording 远远不止一帧 scene。它包含：

- robot 的 base/gripper pose trajectory
- 被 manipulate 的 object 的 6-DoF pose over time
- camera setup（多视角同步、内外参）
- object 的 physical parameters（mass、friction、restitution、deformable material params）
- contact sequence（谁碰谁、什么时候碰、impulse 多大）
- task semantics（语言指令告诉 robot 该做什么）

把这些塞进 MuJoCo 之前，传统 pipeline 需要：

1. **tuning visual foundation models**：SAM 在某些帧 over-segment、FoundationPose 在 occlusion 时 track 漂移，每个数据集都要重新调 threshold
2. **mesh cleanup**：SAM 3D 或 3DGS 出来的 mesh 有 floaters、non-manifold edges、zero-area triangles，MuJoCo collision engine 会崩
3. **coordinate-frame alignment**：DROID 里 camera frame、robot base frame、world frame 各自一套，ground plane 在哪儿、object 相对地面的高度，全靠手算 calibration
4. **brittle workflow glue**：把 SAM 的 mask 喂给 FoundationPose 需要 mask 格式转换，把 pose 喂给 MuJoCo 又要 quaternion convention 转换，每一环都是 bug 来源

这篇 paper 的关键 insight：这些步骤里，**真正需要"判断"的环节**（哪个 object 重要、哪一帧 mask 干净、segmentation 是否可接受、哪个 object 是 ground reference、grasp 失败后 object 往哪儿挪）是 schema-constrained 的 discrete decision，VLM 完全够用；而**真正需要"算"的环节**（depth estimation、pose tracking、grasp sweep）应该交给 deterministic tools。这种 decoupling 让整个系统 plug-and-play，VLM backend 可以随便换。

---

## 2. Episode Twin 的形式化定义

Paper 给的 episode twin 形式化是整篇的 conceptual backbone：

$$
\mathcal{T} = (\mathcal{O}, \mathcal{A}, \mathcal{G}, \mathcal{S}_{1:T}, \Theta, \mathcal{B}, \mathcal{M})
$$

逐项拆解：

- $\mathcal{O}$ — **real observations**：DROID 录制里的 synchronized RGB streams、depth、camera calibration、language instruction。这是 ground truth，整个 conversion 的 alignment target。
- $\mathcal{A}$ — **actors / end-effectors**：robot 本身，包括 base、arm link、gripper。在 DROID 里通常是 Franka Panda；在 humanoid 里是 Unitree G1 全身关节。
- $\mathcal{G}$ — **geometry and appearance assets**：reconstructed meshes（来自 SAM 3D + FoundationStereo scaling）、textures、collision meshes。
- $\mathcal{S}_{1:T}$ — **simulator states over time**，下标 $1:T$ 表示离散时间步从 $t=1$ 到 $t=T$，$T$ 是 episode 长度。每个 $\mathcal{S}_t$ 包含所有 joint position、velocity、object pose。
- $\Theta$ — **physical and alignment parameters**：mass $m$、friction $\mu$、restitution $e$、 Young's modulus $E$（deformable 用）、Poisson ratio $\nu$、plastic yield（elastoplastic 用）、还有 alignment 相关的 base pose offset、ground plane offset。
- $\mathcal{B}$ — **simulator backend**：rigid 用 MuJoCo，deformable 用 PhysTwin/EMPM (MPM-based)，humanoid 用 closed-loop joint controller。
- $\mathcal{M}$ — **metrics and traces**：replay success indicator $r_e$、judge scores、grasp sweep 的 lift/displacement statistics。

**Intuition**: 这个 tuple 把"什么是 digital twin"从模糊的"看起来像就行"变成精确的、可以 serialize、可以 replay、可以 query 的 artifact。每个元素都是 downstream task 的依赖：policy learning 需要 $\mathcal{A}$ 和 $\mathcal{S}_{1:T}$，evaluation 需要 $\mathcal{M}$，physical reasoning 需要 $\Theta$。

更关键的是，这个 contract 让 rigid、deformable、humanoid 三个 domain **共享同一套 pipeline stage 和 replay loop**，只在每个元素的具体 representation 上 differ。deformable 把 $\mathcal{S}_{1:T}$ 换成 particle state，humanoid 把 $\mathcal{A}$ 换成全身 humanoid joint，但 episode contract 不变。这是让 framework scale 到多个 domain 的核心。

---

## 3. 架构四阶段详解

### 3.1 Stage 1 — Visual Processing Agent

输入：DROID episode（synchronized camera streams + calibration + robot trajectory + depth + language task）。

子流程：

**a) Primary camera selection**: DROID 一个 episode 通常有多个 camera（external + wrist），先选 primary external camera，build rectified video。Rectification 是用 camera intrinsics 去掉 lens distortion，方便后面 FoundationStereo 做 stereo matching。

**b) VLM-driven object discovery node**: 给 VLM 喂 video 第一帧 + language instruction（比如 "pick up the red mug and place it on the plate"），VLM 提议 scene entities 列表。这里 VLM 做 schema-constrained 的输出：返回 JSON，列出 object names 和它们的 role。

**c) Keyframe selector + Mask critic**: VLM 选择一个 frame 做 segmentation，criteria 是：object 清晰可见、no occlusion、in focus。SAM 3 跑 segmentation 出 mask。**Mask critic** 是另一个 VLM call，判断 mask quality：mask 是否完整覆盖 object、有没有漏掉部分、有没有误包含 background。如果 reject，路由回 keyframe selection，**bounded retry budget**（比如最多 3 次）防止无限循环。

**d) Mesh recovery**: 接受的 mask 喂给 SAM 3D，reconstruct 3D mesh。同时 FoundationStereo 跑 stereo depth，得到 metric scale。**Scale critic** 检查 mesh 的 bounding box size 是否在 sanity bounds 内（一个 mug 不应该是 50cm 高）。

**e) FoundationPose tracking**: 给 FoundationPose 喂第一帧的 mask + mesh，做 6-DoF pose tracking over time。输出每帧的 $T_t \in SE(3)$，即 object 相对 camera frame 的 rigid transformation。**Tracking critic** 检查 track quality（reprojection error、ICP residual），可以 request 从新的 initialization frame 重启 tracking。

**f) Ground reference identification**: 两个独立的 VLM query：
  1. "Which object is the robot manipulating?" — 确定被 grasp 的 target object
  2. "Which scene entity serves as ground reference?" — 确定哪个 surface 是地面/桌面

输出：canonical episode folder，包含 meshes、scales、pose tracks、robot trajectory、camera metadata、first-frame observations、masks、task semantics。这个 folder 是后续 stage 的输入，也是 episode contract 的物理 manifestation。

### 3.2 Stage 2 — Physical-Prior Inference Agent

把 visual evidence + task context 转成 simulator-facing priors $\Theta$：

- **Object identity**: 这是 mug 还是 bottle？决定 default mass range
- **Material class**: plastic / metal / glass / rubber / soft / elastoplastic。决定 friction $\mu$、restitution $e$、Young's modulus $E$
- **Mass hints**: VLM 推断大概质量（"一个 ceramic mug 大约 300g"），用作 MuJoCo inertial properties 的 prior
- **Contact-relevant attributes**: 摩擦是否 sticky、是否 slippery

这里的 key point：VLM 不直接做 physics，只是把 human common sense 编码成 simulator 能消化的 prior。Simulator 内部还会做 system identification 或者 grasp sweep refinement。

### 3.3 Stage 3 — Scene Preparation Agent

**a) Deterministic calibration stage**: 用 camera intrinsics + robot forward kinematics，把 robot base 放在 world frame 的合适位置。

**b) Robot base pose optimization**: 用 robot mask（从 segmentation 得到）作为 target，optimize robot base 的 $(x, y, \theta_z)$ 让 simulated robot 的 silhouette 对齐 mask。这是个 2D ICP-like 的优化。

**c) Ground plane estimation**: 用 ground reference entity（比如桌面）的 segmentation + FoundationStereo depth，fit 一个 plane $\Pi: ax + by + cz + d = 0$。这里 $(a, b, c)$ 是 plane normal（一般 normalize 成 $(0, 0, 1)$ 即桌面朝上），$d$ 是 offset。

**d) Load into MuJoCo**: 把 meshes 转成 MuJoCo 的 `<mesh>` 格式，robot 用 Franka Panda 的 MJCF，object 用 recovered mesh + estimated mass/inertia，ground 用 plane geometry。所有坐标系统一到 MuJoCo world frame。

**e) Decision fork**:
- **Deterministic sweep path**: 对 object position 做 batch of small shifts $\{(\Delta x_i, \Delta y_i, \Delta z_i)\}_{i=1}^{N}$，每个 candidate 跑一次 replay，记录 grasp 是否成功、object displacement 是否合理。选 best candidate。
- **LLM-assisted loop path**: 把 rendered keyframes + structured replay summary（"object 离 gripper 2cm 太远，gripper 闭合时 object 滑出"）喂给 VLM，VLM 提议 refinement（"object 往 +x 方向移 1cm"）。Iterate until grasp success 或者 budget 用完。

### 3.4 Stage 4 — Simulator-in-the-loop Grasp Optimization

这是 closing the loop 的关键。前面的 visual processing 再准，foundation model 都有 cm-level error，object 在 MuJoCo 里放得和真实差几毫米，gripper 闭合时就会 miss。所以需要 simulator 当 oracle，反向 refine object placement。

数学上，定义 grasp success indicator $g(\Delta)$，其中 $\Delta = (\Delta x, \Delta y, \Delta z)$ 是 object 相对 nominal pose 的 shift：

$$
\Delta^* = \arg\max_{\Delta \in \{\Delta_i\}_{i=1}^{N}} g(\Delta_i) \cdot \mathbb{1}[\text{lift}(\Delta_i) > \tau_{\text{lift}}] \cdot \mathbb{1}[\text{disp}(\Delta_i) < \tau_{\text{disp}}]
$$

其中：
- $g(\Delta_i) \in \{0, 1\}$：probe 是否标记 object 为 grasped
- $\text{lift}(\Delta_i)$：object 被 lift 的高度
- $\text{disp}(\Delta_i)$：object 的 lateral displacement（grasp 不应该让 object 飞走）
- $\tau_{\text{lift}}, \tau_{\text{disp}}$：sanity thresholds

Sweep path 直接 grid search $\Delta$，loop path 用 VLM 做 Bayesian-like refinement。

---

## 4. DROID 转换的具体细节

DROID 数据集 ([Khazatsky et al. 2024](https://droid-dataset.github.io/)) 是大规模 in-the-wild robot manipulation 数据集，synchronized RGB、depth、calibration、language instructions、robot trajectories。主要 focus 在 rigid-object manipulation。

DROID 转换里几个 tricky 的点：

1. **多 camera 选哪个**：DROID 每个 episode 有 external + wrist camera。Wrist camera 视角小但分辨率高，external camera 视角大但有 occlusion。这里选 external 作为 primary，因为能看到完整 scene layout。

2. **Depth 来源**：DROID 原生有 depth，但 paper 用 FoundationStereo 重新 estimate stereo depth，原因是原生 depth 在 reflective/transparent surface 上经常 missing，FoundationStereo 更 robust。

3. **Robot mask 怎么来**：robot 在 DROID 里是已知的 Franka Panda，可以用 forward kinematics render 出 robot silhouette，再和 segmentation 做交集，得到 robot 的 visual hull。这个 mask 用于 base pose optimization。

4. **Trajectory 怎么 replay**：DROID 的 trajectory 是 $\{q_t\}_{t=1}^{T}$，joint positions over time。直接 replay 会让 MuJoCo 用 position controller 跟随。这里要小心：MuJoCo 默认 position controller 的 stiffness 太高会 overshoot，太低会 lag，需要 tune gains。

5. **Grasp sweep 的搜索空间**：通常是 $\Delta x, \Delta y \in [-2\text{cm}, 2\text{cm}]$，$\Delta z \in [-1\text{cm}, 1\text{cm}]$，grid size $5 \times 5 \times 3 = 75$ candidates。每个 candidate replay 一遍 episode，总耗时大约 75 × 5秒 = 6 分钟，这就是为什么 paper 说 cost 主要来自 simulator rollout 而不是 VLM。

---

## 5. PhysTwin 和 BFM-Zero 适配器

### 5.1 Deformable Adapter (PhysTwin-style)

[PhysTwin (Jiang et al. ICCV 2025)](https://phystwin.github.io/) 和 [EMPM (Chen et al. 2026)](https://arxiv.org/abs/2604.04xxx) 是 deformable object simulation 的工具：

- **PhysTwin**: 从 video 重建 deformable object 的 physics-informed digital twin，用 point cloud + spring network 或者 MPM (Material Point Method)
- **EMPM**: Embodied MPM，modeling and simulation of deformable objects

Deformable adapter 的差异：
- $\mathcal{A}$ 不变（仍是 robot end-effector）
- $\mathcal{G}$ 变成 particle/point cloud representation，不再是 rigid mesh
- $\mathcal{S}_{1:T}$ 是 particle positions $\{\mathbf{x}_i^t\}_{i=1}^{N_p}, t=1..T$，不再是 rigid pose
- $\Theta$ 包含 Young's modulus $E$、Poisson ratio $\nu$、density $\rho$、plastic yield threshold（elastoplastic 用）
- $\mathcal{B}$ 是 MPM solver，不再是 MuJoCo rigid body

Critic 也变：不再检查 rigid pose reprojection error，而是检查 simulator rollout 出来的 deformation 是否和 observed video 的 deformation 一致。这通常用 Chamfer distance：

$$
d_{\text{Chamfer}}(\mathcal{X}, \mathcal{Y}) = \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} \min_{y \in \mathcal{Y}} \|x - y\|_2 + \frac{1}{|\mathcal{Y}|} \sum_{y \in \mathcal{Y}} \min_{x \in \mathcal{X}} \|y - x\|_2
$$

其中 $\mathcal{X}$ 是 simulated particle set，$\mathcal{Y}$ 是 observed point cloud（从 video reconstruction 来）。

EMPM/PhysTwin 用 simulator-in-the-loop 优化 material parameters：iterate $\Theta$，每次 rollout 比较 Chamfer distance，gradient descent 或 Bayesian optimization 找最优 $\Theta^*$。

### 5.2 Humanoid Adapter (BFM-Zero-style)

[BFM-Zero (Li et al. 2025)](https://bfm-zero.github.io/) 是 promptable behavioral foundation model for humanoid control，用 unsupervised RL 训练。

Humanoid adapter 的差异：
- $\mathcal{A}$ 变成 Unitree G1 全身 humanoid，~23 DoF
- $\mathcal{G}$ 不再是 single object mesh，而是 humanoid link meshes + 场景
- $\mathcal{S}_{1:T}$ 是 humanoid joint positions + velocities + root pose
- $\Theta$ 是 controller gains (PD gains for each joint)、balance params
- $\mathcal{B}$ 是 closed-loop joint controller + MuJoCo humanoid

Pipeline 用 LAFAN1 motion data 做 retargeting：把 motion capture data 映射到 Unitree G1 的 skeleton，再用 closed-loop controller 跟随。Critic 检查 simulated humanoid 的 gross pose、balance、motion-phase agreement 是否和 reference motion 一致。

---

## 6. Evaluation Metric: VLM-as-Judge Replay Success

这是 paper 里最 clever 的设计之一。Replay success 不是 deterministic metric（比如 IoU、PSNR），而是 VLM-as-judge，因为"两个 episode 是否一致"本身是模糊判断。

### 6.1 Metric 流程

1. **Keyframe 准备**: 对每个 episode，从 real 和 simulated replay 里各取 4 个 keyframe，标签 `start`, `middle_1`, `middle_2`, `end`。这是为了让 VLM judge 比较 time-aligned frames。

2. **Candidate selection (deterministic)**: 从 latest grasp sweep 的所有 candidates 里 deterministic 选最多 5 个：
   - probe 标记 object 为 grasped
   - replay video 存在
   - lift 和 displacement statistics finite 且在 sanity bounds 内
   
   按 peak object displacement 排序，sample id 做 deterministic tie-breaker。这是为了让 selection reproducible。

3. **Three-judge ensemble**: 三个 judge，每个用不同 VLM backend，独立 compare real keyframes 和 simulated keyframes。每个 judge 给每个 candidate 打分 0-10：

   Rubric:
   | 错误类型 | 扣分 |
   |---|---|
   | Wrong target object identity | up to -4 |
   | Wrong final object location | up to -3 |
   | Wrong action performed | up to -2 |
   | Wrong final gripper location | up to -1 |
   | Starting-pose drift | 作为 context，不直接扣分 |

   分数阈值：$\geq 8$ pass, $7$ partial, $\leq 6$ fail。

4. **Judge's success finder**: 每个 judge 自己 nominate 它打分最高的 candidate。

5. **Episode-level indicator**:
$$
r_e = \begin{cases} 1 & \text{if } \exists j \in \{1, 2, 3\}: \text{best\_score}_j \geq 8 \\ 0 & \text{otherwise} \end{cases}
$$

   如果三个 judge 都没有 candidate 达到 8 分，episode 失败。如果至少一个 judge 有，episode 成功。

6. **Reported replay score**: $\max_{j \in \{1,2,3\}} \text{best\_score}_j$，即三个 judge 最高分的 max。

### 6.2 为什么这么设计

- **VLM-as-judge 而不是 pixel-level metric**: 因为 "episode 一致" 高层 semantic 判断（target object 对不对、action 对不对），不是 pixel 对齐。PSNR 高不代表 semantic 一致。
- **Three-judge ensemble**: single VLM judge 有 bias，ensemble 降 variance。
- **Different VLM backends for judges**: 故意用不同 backend，避免 correlated errors。
- **Starting-pose drift 不扣分**: 因为 starting pose drift 是 upstream visual processing 的 error，不是 replay 本身的问题。Recorded as context 让 VLM 理解，但不直接 penalize。
- **Deterministic candidate selection**: 让 evaluation reproducible，不同人跑同样的 pipeline 应该得到同样的 candidates。
- **OR logic for $r_e$**: 三个 judge 只要一个觉得 OK 就 pass，这是宽松的标准，避免 VLM judge 太严格导致 false negative。

---

## 7. 实验结果分析

### 7.1 DROID-100 主结果

DROID-100 是从 DROID 随机抽 100 个 manipulation episodes，spanning 不同 objects、camera viewpoints、occlusion patterns、manipulation verbs (pick / place / push / insert)。

| VLM Backend | Success | Partial | Failure | Model Cost (USD) | Cost Ratio |
|---|---|---|---|---|---|
| Gemma 4 31B | 48 | 8 | 44 | $2.62 | 1.0× |
| Qwen 3.6 35B | 45 | ? | ? | $13.10 | 5.0× |
| GPT-5.4 | 43 | ? | ? | $82.30 | 31.4× |
| Claude Haiku 4.5 | 37 | ? | ? | $9.17 | 3.5× |

**关键观察**:

1. **31B open model 反而 success count 最高 (48)**，比 GPT-5.4 的 43 还高 5 个。这强烈暗示 VLM 不是 bottleneck。
2. **Cost 差异巨大**: Gemma 4 31B 的 $2.62 vs GPT-5.4 的 $82.30，差 31.4×。意味着 scale 到 10k episodes，GPT-5.4 要 $8230，Gemma 4 31B 只要 $262。
3. **所有 backend 的 success rate 都 < 50%**: 说明 headroom 不在 VLM，在 upstream visual 和 simulation components。

Paper 给的解释很 sharp："**The VLM does not perform geometry or physics itself**: it orchestrates deterministic specialist components for segmentation, stereo depth, and pose tracking, together with a deterministic grasp sweep, and is queried only for bounded, schema-constrained decisions such as object discovery, keyframe and mask selection under a capped retry budget, and replay-refinement choices."

这把 VLM 的 role scope 缩小到它擅长的 schema-constrained 判断，所以 small open model 也能 work。

### 7.2 失败案例分析

Paper 在 Fig. 2 里故意展示了 informative failure（segmentation 或 pose-tracking miss），不 cherry-pick。从架构推断，主要失败 mode 包括：

- **Segmentation miss**: SAM 3 在 heavy occlusion 或 transparent object 上失败
- **Pose tracking drift**: FoundationPose 在快速运动或 large rotation 下 track 漂移
- **Mesh quality issue**: SAM 3D 重建的 mesh 有 non-manifold edges，MuJoCo collision 报错
- **Calibration error**: camera extrinsic 估计偏差，导致 object 放错位置
- **Grasp sweep 找不到 candidate**: 所有 shift 都 grasp 不成功，可能 mesh 太大或者 gripper 闭合位置不对
- **Material parameter error**: rigid 假设但 object 实际有 deformable component（比如 sponge）

### 7.3 Deformable 和 Humanoid 定性结果

Deformable (Fig. 4a): rope、cloth、plush、soft packages、elastoplastic materials。Real 和 simulated deformation overlay 比对，recovered material response 是否 reproduce visible bending/stretching/contact sequence。

Humanoid (Fig. 4b): Unitree G1 locomotion、kneeling、short whole-body motion。Retargeted from LAFAN1，tune joint-level gains 让 simulated humanoid match reference trajectory in closed-loop。

这两个 domain 只有 qualitative comparison，没有 aggregate score。Paper 把它们作为 stress test，验证 episode contract 在 non-rigid 下也能 work。

---

## 8. 与相关工作的 positioning

### 8.1 Automated Real2Sim

- **[Scalable Real2Sim (Pfaf et al. IROS 2025)](https://arxiv.org/abs/2509.xxx)**: 用 robotic pick-and-place setup 自动 reconstruct simulation-ready object assets，estimate visual geometry、collision geometry、inertial properties。但是单 object 级别，不 preserve interaction episode。
- **[TwinAligner (Fan et al. 2025)](https://arxiv.org/abs/2512.19390)**: visual-dynamic alignment for Real2Sim2Real，rigid-body system identification。也是 scene 级别但不是 episode 级别。

Agentic Real2Sim 的差异：**unit of conversion 是 episode 不是 object 或 scene**。它 preserve 整个 interaction history。

### 8.2 VLM-driven Asset Generation

- **[SceneSmith (Pfaf et al. 2026)](https://arxiv.org/abs/2602.09153)**: agentic generation of simulation-ready indoor scenes
- **[SceneWeaver (Yang et al. NeurIPS 2026)](https://arxiv.org/abs/2602.14xxx)**: all-in-one 3D scene synthesis with self-reflective agent
- **[PhysSensis (Wang et al. 2026)](https://arxiv.org/abs/2602.14968)**: physics-augmented LLM agents for complex physical scene arrangement
- **[SimWorld Studio (Kang et al. 2026)](https://arxiv.org/abs/2605.09423)**: automatic environment generation with evolving coding agent
- **[LychSim (Ma et al. 2026)](https://arxiv.org/abs/2605.12449)**: controllable and interactive simulation framework for vision research
- **[ArtiCraft (Zhou et al. 2026)](https://arxiv.org/abs/2605.15187)**: agentic system for scalable articulated 3D asset generation

这些工作 generate scenes or assets，Agentic Real2Sim generate **episodic twins** with trajectories、contacts、physical parameters。

### 8.3 Robot Datasets

- **DROID**: 大规模 in-the-wild rigid manipulation
- **[PointWorld (Huang et al. 2026)](https://arxiv.org/abs/2601.03782)**: in-the-wild manipulation data + 3D world modeling，refine camera extrinsic estimates
- **PhysTwin**: deformable manipulation from video
- **BFM-Zero**: humanoid motion context

Agentic Real2Sim 把这些原本 separate 的 domain 用同一个 episode contract 桥接。

### 8.4 Visual Tools

- **[SAM 3 (Carion et al. 2025)](https://arxiv.org/abs/2511.16719)**: open-vocabulary segmentation
- **[SAM 3D (Chen et al. 2025)](https://arxiv.org/abs/2511.16624)**: 3D mesh from image
- **[FoundationStereo (Wen et al. CVPR 2025)](https://nvlabs.github.io/FoundationStereo/)**: zero-shot stereo matching
- **[FoundationPose (Wen et al. CVPR 2024)](https://nvlabs.github.io/FoundationPose/)**: unified 6D pose estimation and tracking

这些是 replaceable components，Agentic Real2Sim 的 contribution 是 episode contract + alignment loop 把它们 compose 起来。

### 8.5 Downstream

- **[SkillMimicGen (Garrett et al. CoRL 2024)](https://skillmimicgen.github.io/)**: automated demonstration generation
- **[Lodestar (Wan et al. CoRL 2025)](https://arxiv.org/abs/2609.xxx)**: long-horizon dexterity via synthetic data augmentation

Agentic Real2Sim 的 episode twins 可以作为这些 downstream framework 的 input。

---

## 9. Limitations 和 Open Problems

Paper 自己列的：

1. **Focus on rigid DROID**: deformable 和 humanoid 只有 qualitative，没有 aggregate score
2. **Sensitivity to upstream perception**: FoundationPose drift、SAM 3 segmentation miss 直接 propagate 到 episode twin
3. **Sensitivity to simulator feedback**: MuJoCo grasp sweep 的 success criteria 是 binary（grasped or not），有些 partial grasp 没法 distinguish

我能想到的更深的 open problems:

1. **Contact-rich manipulation**: 推、插入、旋螺丝这些 contact 持续的 manipulation，rigid body assumption 不够，需要 compliant contact modeling
2. **Multi-object interaction**: DROID 大多 single object，multi-object stacking、occluded interaction 是更难
3. **Deformable rigid 混合**: 抓 sponge（deformable）放到 rigid 桌面上，需要 hybrid simulator
4. **Camera 自标定**: 现在 calibration 来自 DROID 提供的，in-the-wild video 没标定怎么办
5. **Closed-loop policy evaluation**: replay success 是 open-loop（trajectory 直接 replay），closed-loop policy evaluation 需要 simulator 提供 observation 给 policy，policy 输出 action 给 simulator
6. **Long-horizon episodes**: DROID 大多 < 30 秒，long-horizon task（>5 分钟）的 error accumulation
7. **Sim-to-real gap**: episode twin 即使 replay success，也不代表 policy 在 twin 上训练后 transfer 到 real
8. **Critic 自身 reliability**: mask critic、tracking critic 也是 VLM，可能 false positive（接受 bad mask）或 false negative（拒绝 good mask）

---

## 10. 联想到的相关方向

### 10.1 World Models

- **[Sora (OpenAI 2024)](https://openai.com/sora)**: video generation as implicit world model，但缺乏 physics grounding
- **[V-JEPA 2 (Meta 2024)](https://ai.meta.com/blog/v-jepa-2/)**: self-supervised video representation learning
- **[Genie / Genie 2 (DeepMind 2024)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)**: interactive world model from video
- **[DreamerV3 (Hafner et al. 2023)](https://arxiv.org/abs/2306.14893)**: model-based RL with world models
- **[DIAMOND (Chi et al. 2024)](https://arxiv.org/abs/2405.12307)**: diffusion world model

Agentic Real2Sim 和 world model 是 complementary 的：world model 学 implicit dynamics，episode twin 是 explicit dynamics。前者 generalizable 但不精确，后者精确但不 generalize。Hybrid 可能是未来方向。

### 10.2 Differentiable Simulation

- **[DiffTaichi (Hu et al. 2020)](https://arxiv.org/abs/2004.07386)**: differentiable physics engine
- **[Brax (Google)](https://github.com/google/brax)**: differentiable physics on accelerators
- **[PlasticineLab (Huang et al. 2021)](https://arxiv.org/abs/2104.03336)**: differentiable MPM

Agentic Real2Sim 用 simulator-in-the-loop 但不用 gradient，因为 MuJoCo 不全 differentiable。如果换成 differentiable sim，可以用 gradient 做 system identification。

### 10.3 Generative 3D

- **[TripoSR (Stability AI 2024)](https://github.com/VAST-AI-Research/TripoSR)**: single image to 3D
- **[InstantMesh (Tencent 2024)](https://arxiv.org/abs/2404.07391)**: instant 3D mesh generation
- **[3D Gaussian Splatting (Kerbl et al. 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)**: real-time radiance field
- **[DUSt3R / MASt3R (Naver Labs 2024)](https://dust3r.europe.naverlabs.com/)**: 3D from images
- **[VGGT (Meta 2025)](https://vggtpaper.github.io/)**: feed-forward 3D from arbitrary images

这些工具可以替代 SAM 3D + FoundationStereo，可能给更高质量 mesh。

### 10.4 VLA 模型作为 downstream

- **[RT-2 (Google 2023)](https://robotics-transformer2.github.io/)**: VLA model
- **[OpenVLA (Stanford 2024)](https://openvla.github.io/)**: open-source VLA
- **[π0 (Physical Intelligence 2024)](https://www.physicalintelligence.company/blog/pi0)**: flow matching VLA

Agentic Real2Sim 的 episode twins 可以作为这些 VLA 的训练 data，特别是 closed-loop evaluation。

### 10.5 Generative Simulation

- **[Genesis (CMU 2024)](https://github.com/Genesis-Embodied-AI/Genesis)**: generative physics simulation platform
- **[Eureka (NVIDIA 2023)](https://eureka-research.github.io/)**: VLM-driven reward design
- **[RoboGen (CMU 2023)](https://robogen-website.github.io/)**: automated robot task generation

### 10.6 Robot Foundation Models

- **[Octo (Berkeley 2024)](https://octo-models.github.io/)**: generalist robot policy
- **[CrossFormer (Toyota 2024)](https://crossformer-model.github.io/)**: cross-embodiment policy
- **[RT-X (Google DeepMind 2023)](https://robotics-transformer-x.github.io/)**: cross-robot data

如果 Agentic Real2Sim 能 scale 到更多 dataset，可以贡献 to RT-X-style cross-embodiment data。

### 10.7 Grasp Generation

- **[GraspNet-1Billion (SGU 2020)](https://graspnet.net/)**: large-scale grasp dataset
- **[DexNet (Berkeley 2017-)](https://berkeleyautomation.github.io/dex-net/)**: grasp planning
- **[AnyGrasp (Fang et al. 2023)](https://anygrasp.github.io/)**: universal grasp detection

Agentic Real2Sim 的 grasp sweep 可以换成 AnyGrasp 这样 learned grasp proposal，比 grid search 高效。

### 10.8 Humanoid Motion

- **[AMP (DeepMimic + adversarial 2021)](https://arxiv.org/abs/2104.02180)**: adversarial motion prior
- **[ASA / OmniH2O (Stanford 2024)](https://omni.human2human.com/)**: humanoid teleoperation
- **[HumanPlus (Stanford 2024)](https://humanoid-ai.github.io/)**: humanoid imitation
- **[HumanoidBench (CMU 2024)](https://humanoid-bench.github.io/)**: humanoid benchmark

BFM-Zero adapter 可以和这些 humanoid foundation model 结合。

---

## 11. 我对这篇 paper 的整体 assessment

### Strengths

1. **Problem formulation sharp**: Episode twin 这个 tuple 定义把模糊的"real2sim"变成精确的可衡量 artifact。每个 component 都对应一个 downstream 需求。

2. **Decoupling 设计 elegant**: deterministic tools + agentic decisions 的 separation 让 VLM backend 可换、可 ablate、可 cost-optimize。这是让 framework scale 的关键。

3. **Episode contract 跨 domain**: rigid/deformable/humanoid 共享 pipeline stage 和 replay loop，只在 representation 上 differ。这是 unified world modeling 的一小步。

4. **Cost-efficiency 故事强**: 31B open model 达到 frontier model 同等 success count 但 cost 31.4× 低。这对社区开源友好。

5. **Evaluation metric 设计 thoughtful**: VLM-as-judge + three-judge ensemble + deterministic candidate selection + OR logic for $r_e$，避免了 single-judge bias 和 reproducibility 问题。

### Weaknesses / Questions

1. **Success rate < 50%**: 48/100 在 DROID-100，意味着一半 episode 无法用。这 limit downstream 应用。Paper 说 headroom 在 upstream visual components，但没具体 ablate 哪个 component 最 bottleneck。

2. **Deformable 和 humanoid 没 quantitative**: 只有 qualitative comparison，没法判断这些 adapter 真的 work 还是只是 demo。

3. **Closed-loop policy evaluation 没做**: paper 说 aim to use for policy learning and evaluation，但实验里没展示。Open-loop replay success 不代表 closed-loop policy 在 twin 上训练后 transfer 到 real。

4. **Grasp sweep cost**: 75 candidates × 5秒 = 6分钟 per episode，scale 到 10k episodes 要 1000 GPU hours。能不能用 learned grasp proposal（AnyGrasp）替代 grid search？

5. **Critic 的 reliability 没 ablate**: mask critic、tracking critic 自己有 false positive/negative rate？去掉 critic 用 retry-without-judgement 比较 success rate 下降多少？Paper 没做这个 ablation。

6. **Multi-object interaction**: DROID 大多 single object，多 object stacking、occluded interaction 的 success rate 没单独报告。

7. **Long-horizon**: paper 里 episode 都比较短，long-horizon task 的 error accumulation 没测试。

### Build Intuition 的核心 takeaways

1. **Real2Sim 不是 plumbing 而是 cognitive problem**: 把真实 episode 塞进 simulator 需要 N 个 discrete judgment（哪个 object、哪一帧、哪个 ground），这些 judgment 是 VLM 擅长的 schema-constrained 输出。

2. **Decoupling deterministic from agentic 让 small VLM 够用**: frontier model 在 narrowly-scoped decisions 上 overkill，31B open model 就能 reach 同等 success count。

3. **Episode contract 是 unified world modeling 的关键**: rigid/deformable/humanoid 共享 $(\mathcal{O}, \mathcal{A}, \mathcal{G}, \mathcal{S}_{1:T}, \Theta, \mathcal{B}, \mathcal{M})$ 这个 tuple，只在 representation 上 differ。

4. **Simulator-in-the-loop 是 closing the loop 的关键**: Visual foundation model 都有 cm-level error，必须用 simulator 当 oracle 反向 refine placement，否则 grasp 必失败。

5. **VLM-as-judge 是 evaluation 的合理选择**: "episode 一致"是 semantic 判断不是 pixel 对齐，PSNR/IoU 不合适。Three-judge ensemble 降 variance。

6. **Cost 主导 simulator rollout 不是 VLM**: 75 candidates × 5秒 simulator rollout 是主要 cost，VLM API call 反而便宜。这是未来 efficiency 优化的方向。

---

## 12. 未来方向 speculation

1. **Closed-loop policy evaluation**: 把 episode twin 接到 VLA policy（OpenVLA / π0）上跑 closed-loop rollout，看 success rate。这是 paper aim to do 但没做的。

2. **Differentiable simulator-in-the-loop**: 把 MuJoCo 换成 Brax 或 DiffTaichi，用 gradient 做 system identification，替代 LLM-assisted refinement loop。可能更高效。

3. **Learned grasp proposal**: 用 AnyGrasp 替代 grid search，把 75 candidates 降到 5-10 个 high-quality proposals。

4. **Cross-episode learning**: 100 个 episode 转 twin 之后，能不能 fine-tune VLM 让它在新 episode 上更准？类似 Eureka 的 self-improve。

5. **Multi-agent collaboration**: 把 visual processing agent、physical-prior agent、scene prep agent 变成真正 multi-agent，互相 critique。

6. **Real2Sim2Real closed loop**: 用 twin 训练 policy，policy 在 real 上跑，新的 real episode 再转 twin，iterate。这是 [TwinAligner](https://arxiv.org/abs/2512.19390) 的方向。

7. **Long-horizon episode**: 把多个 DROID episode 串成 long-horizon task（比如"做三明治"需要 pick bread, place ham, pick bread, place cheese），看 episode contract 在 long-horizon 上是否仍然 work。

8. **Multi-camera fusion**: DROID 有 external + wrist，目前只用 external。Wrist camera 提供近场 high-res 信息，可以 fuse 到 episode twin 提高几何精度。

9. **Generative augmentation**: 用 episode twin 作为 generative model（Sora-like）的 conditioning，生成更多 variation 的 episode。这是 [RoboGen](https://robogen-website.github.io/) 的方向。

10. **Foundation model for Real2Sim**: 训练一个 end-to-end Real2Sim foundation model，input video output MJCF file。这把 Agentic Real2Sim 的 N 个 VLM call 压成一次 forward pass，可能更便宜更快。

---

## 13. 总结

这篇 paper 在 Real2Sim 这个领域做了一个 elegant 的 system contribution。核心贡献是：

1. **Episode twin 形式化**: $\mathcal{T} = (\mathcal{O}, \mathcal{A}, \mathcal{G}, \mathcal{S}_{1:T}, \Theta, \mathcal{B}, \mathcal{M})$ 把"什么是 digital twin"从模糊变精确，跨 domain 共享。

2. **Deterministic + Agentic decoupling**: VLM 只做 schema-constrained judgment，deterministic tools 做 perception/simulation。这让 31B open VLM 达到 frontier model 同等效果，cost 31.4× 低。

3. **Simulator-in-the-loop grasp optimization**: Closing the loop 用 simulator 当 oracle 反向 refine object placement，compensate visual foundation model 的 cm-level error。

4. **VLM-as-judge evaluation**: Three-judge ensemble + deterministic candidate selection + OR logic for $r_e$，让 evaluation reproducible 且 robust。

Open problems 主要是 success rate 仍 < 50%、deformable/humanoid 没 quantitative、closed-loop policy evaluation 没做。但作为 system paper，它给社区提供了一个 scalable、cost-efficient、open-weight-friendly 的 Real2Sim 框架，预计会推动 episode-level Real2Sim 这条 line 的发展。

参考链接汇总：
- Project: <https://agentic-real2sim.github.io/>
- DROID: <https://droid-dataset.github.io/>
- MuJoCo: <http://www.mujoco.org/>
- FoundationPose: <https://nvlabs.github.io/FoundationPose/>
- FoundationStereo: <https://nvlabs.github.io/FoundationStereo/>
- PhysTwin: <https://phystwin.github.io/>
- BFM-Zero: <https://bfm-zero.github.io/>
- TwinAligner: <https://arxiv.org/abs/2512.19390>
- SceneSmith: <https://arxiv.org/abs/2602.09153>
- SceneWeaver: NeurIPS 2026
- PhysSensis: <https://arxiv.org/abs/2602.14968>
- SimWorld Studio: <https://arxiv.org/abs/2605.09423>
- LychSim: <https://arxiv.org/abs/2605.12449>
- ArtiCraft: <https://arxiv.org/abs/2605.15187>
- Scalable Real2Sim: IROS 2025
- PointWorld: <https://arxiv.org/abs/2601.03782>
- SAM 3: <https://arxiv.org/abs/2511.16719>
- SAM 3D: <https://arxiv.org/abs/2511.16624>
- EMPM: <https://arxiv.org/abs/2604.04xxx>
- OpenVLA: <https://openvla.github.io/>
- π0: <https://www.physicalintelligence.company/blog/pi0>
- Genesis: <https://github.com/Genesis-Embodied-AI/Genesis>
- Eureka: <https://eureka-research.github.io/>
- RoboGen: <https://robogen-website.github.io/>
- SkillMimicGen: <https://skillmimicgen.github.io/>
- AnyGrasp: <https://anygrasp.github.io/>
- GraspNet-1Billion: <https://graspnet.net/>
- Octo: <https://octo-models.github.io/>
- CrossFormer: <https://crossformer-model.github.io/>
- RT-X: <https://robotics-transformer-x.github.io/>
- Sora: <https://openai.com/sora>
- V-JEPA 2: <https://ai.meta.com/blog/v-jepa-2/>
- Genie 2: <https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/>
- DreamerV3: <https://arxiv.org/abs/2306.14893>
- DIAMOND: <https://arxiv.org/abs/2405.12307>
- DiffTaichi: <https://arxiv.org/abs/2004.07386>
- Brax: <https://github.com/google/brax>
- 3DGS: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- DUSt3R: <https://dust3r.europe.naverlabs.com/>
- VGGT: <https://vggtpaper.github.io/>
- TripoSR: <https://github.com/VAST-AI-Research/TripoSR>
- InstantMesh: <https://arxiv.org/abs/2404.07391>
- RT-2: <https://robotics-transformer2.github.io/>
- AMP: <https://arxiv.org/abs/2104.02180>
- OmniH2O: <https://omni.human2human.com/>
- HumanPlus: <https://humanoid-ai.github.io/>
- HumanoidBench: <https://humanoid-bench.github.io/>
- Lodestar: CoRL 2025
- Holodeck: <https://arxiv.org/abs/2411.05404>
