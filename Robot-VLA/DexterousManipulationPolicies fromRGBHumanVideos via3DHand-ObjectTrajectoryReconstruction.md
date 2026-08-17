---
source_pdf: DexterousManipulationPolicies fromRGBHumanVideos via3DHand-ObjectTrajectoryReconstruction.pdf
paper_sha256: a7f2e8a51c7b82237d00209ae1bd3db96682f774b3434e93aba5d067d28a814f
processed_at: '2026-08-03T20:49:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VIDEOMANIP

## 这篇 paper 在干啥

简单说：**你拿手机拍一段人手抓东西的视频，30 秒，一段就够，系统就能教机器人灵巧手学会同样的动作**。不用手套、不用动捕、不用机器人示教、不用预先扫好物体 3D 模型。

这是 CMU + Stanford + Georgia Tech 的工作，2026 年初挂出来的。

---

## 为什么这事难

你想啊，从"人手抓苹果的视频"到"LEAP Hand 抓苹果"，中间隔着几道大坎：

**第一道坎：视频是 2D 的，机器人需要 3D 动作。** 你拍到的是一串 RGB 像素，没有深度。机器手需要知道每个手指该放在空间的哪个位置。

**第二道坎：人手和机器手长得完全不一样。** 人手 27 个自由度、柔软、指节比例固定。LEAP Hand 22 个自由度、硬、motor 在指节里。你怎么把人的动作"翻译"给机器手？这叫 retargeting 问题。

**第三道坎：就算你重建出 3D 动作，经常是错的。** HaMeR 这种 hand reconstruction model 在 occlusion 重的视频里会瞎猜，导致手指穿进物体里、或者根本没碰到物体。你拿这种数据训 policy，机器人要么抓空气，要么撞物体。

**第四道坎：一个视频就一条轨迹。** 现在的 imitation learning（比如 DP3）要 1000 条 demo 才能 generalize。你只有 1 条，policy 训出来 overfit 到死，物体换个位置就不会抓了。

这四个坎，paper 一个一个打掉。

---

## 他们怎么打掉的

### 第一道坎：2D → 3D，用现成 vision model 串 pipeline

这是个典型的"2025-2026 年 robotics 套路"——不 train from scratch，而是把一堆 foundation model 串起来：

1. **MoGe-2** 拿 metric depth + camera intrinsics（这是关键，给整个 pipeline 一个统一的物理尺度参考）
2. **SAM 2** 分割物体 → crop 出来
3. **MeshyAI** 把 crop 出来的物体图片变成 3D mesh（但 metric scale 不准）
4. **GPT-4.1** 估计物体大概多大（比如"apple 直径 8cm"）做粗校准
5. **FoundationPose** 在 [0.5×, 2×] 范围扫一遍 scale，选 rendering error 最小的
6. **HaMeR** 重建人手 mesh（MANO 参数化）
7. **AnyTeleop** 把人手动作 retarget 到 robot hand joints

这个 pipeline 的精髓在于 **MoGe-2 提供的 metric depth**。HaMeR 用的是 weak-perspective camera，depth 是 ambiguous 的——它输出的人手可能离物体 50cm，也可能 5cm。MoGe-2 给你绝对深度（以米为单位），你从 HaMeR 预测的 2D keypoint 处采样深度，就能把人手 anchor 到正确的物理位置。

直觉：这就像给一个 2D 画师配了一个测距仪。画师画得对不对另说，但至少他知道画的东西该放在多远。

### 第二道坎：In-the-wild video 的相机是歪的

In-scene video（机器人在场、相机标定过）好办，有个 transform $\text{}^{\text{world}}T_{\text{cam}}$ 直接把 camera frame 转到 robot frame。

但 in-the-wild video（你在家里随便拍的）没这个标定。相机可能是斜的，重建出来的物体在空中飘着，不在桌面上。

解决：**GeoCalib** 从单帧图像推断 gravity direction，得到一个旋转矩阵 $\text{}^{\text{grav}}R_{\text{cam}}$，把所有重建结果转一下，让 gravity 朝下。

直觉：你不需要完整知道相机的 6-DoF pose，只要知道哪边是"下"就够了。因为大多数 manipulation task 的 physics 都围绕重力展开——倒茶要往下倒、挂帽子要往上挂、关抽屉要水平推。gravity 对齐了，task semantics 就对了。

Ablation 数据很扎眼：in-the-wild Pour Tea 不做 gravity alignment，success rate 从 7/10 暴跌到 **0/10**。因为 policy 学到的是"往相机下方倒"，部署时重力方向错了，茶倒在桌子上。

### 第三道坎：重建的 grasp 经常穿模或者不接触

这是 paper 的核心技术贡献之一。

HaMeR 重建的人手，跟 MeshyAI 重建的物体，分别来自不同 model，几何上经常不匹配。手指可能穿进物体里，也可能离物体 2cm 飘在空中。你拿这种 grasp 去训 DRO grasping model，成功率只有 30.7%。

解决：**ContactOpt** 做可微优化。它学了一个 prior："人手抓物体时，哪些区域应该接触"。然后调整 hand pose 参数，让当前 contact map 接近 prior 预测的 contact map。

公式很简单：
$$E(\mathbf{h}) = |C_{\mathcal{O}}(\mathbf{h}) - \hat{C}_{\mathcal{O}}| + |C_{\mathcal{H}}(\mathbf{h}) - \hat{C}_{\mathcal{H}}|$$

其中 $C_{\mathcal{O}}(\mathbf{h})$ 是 hand pose $\mathbf{h}$ 下的 object contact map（每个 object vertex 到最近 hand vertex 的距离的函数），$\hat{C}_{\mathcal{O}}$ 是 prior 预测的理想 contact。

直觉：相当于你重建出来一个 grasp，然后有个"老师"说"人手抓苹果时拇指应该贴在苹果顶部、食指应该托住底部"，你就微调手指位置去贴合这个 prior。不是从零学，而是 refine。

加上 ContactOpt 后，grasp 成功率从 30.7% → 63.75%。**翻倍多**。这是 paper 里最 striking 的 ablation。

### 第四道坎：1 条轨迹 → 1000 条

用 **DemoGen**。核心 idea：对 object point cloud 和 robot trajectory 同时做 SE(3) 变换（旋转 + 平移），因为 point cloud 和 trajectory 都是 SE(3)-equivariant 的，同时变换不破坏 hand-object 的相对接触关系。

具体：原始轨迹里 robot 从位置 A 抓住物体，移动到位置 B。DemoGen 把整个场景旋转 30 度、平移 20cm，得到新轨迹——robot 从 A' 抓物体，移动到 B'。手和物体的相对接触不变，但 spatial location 变了。

1 条 → 1000 条。Pour Tea 的 ablation：从 1/15 → 13/15。

直觉：很多 manipulation task 的难点不是"动作多复杂"，而是"物体位置一变就不会了"。SE(3) augmentation 专门打这个点。就像学开车，你在一条路上开 100 遍没用，但在不同路、不同方向开 1000 遍就熟练了。

### 最后：DP3 policy

用 **3D Diffusion Policy (DP3)**——输入 robot hand point cloud + object point cloud + proprioceptive state，输出 $\Delta q$（关节角增量），closed-loop 执行。

有个 tricky 的 approximation：执行时 LEAP Hand 遮挡严重，DP3 看不到物体，所以假设 grasp 后 hand-object 相对 pose 固定，用 robot 的 forward kinematics 间接更新 object point cloud。

这个 approximation 对 quasi-static task（倒茶、关抽屉）OK，对 dynamic task（in-hand rotation）会崩。所以 paper 没做 in-hand manipulation task——这是个 honest limitation。

---

## 实验结果

**Grasping（IsaacGym 模拟器，Inspire Hand 18-DoF，20 个物体）：**
- 整体 63.75%，加上额外 video 补 5 个失败物体后 70.25%
- 不加 ContactOpt 只有 30.7%
- 评估方式：6 个方向施力扰动（0.5× 物体质量），位移 < 3cm 算成功

**Manipulation（真机 LEAP Hand + xArm，7 个 task）：**
- 平均 62.86%
- 比 LVP（video generation 路线）高 15.87 个百分点
- $\pi_{0.5}$ 几乎全军覆没（因为训练数据主要是 parallel gripper）

最有意思的对比是跟 **LVP**。LVP 走"用 video diffusion 生成未来帧 → 从生成帧提取 hand motion → retarget"的路线。失败模式很有趣：
- 生成出 robot gripper 而不是人手
- 静止不动
- 左右手搞混
- 生成出看不见的手

这是 video generation 路线的 fundamental 问题：pixel-level generation 没有 3D geometry grounding，容易 hallucinate 出物理上不可行的 motion。

VIDEOMANIP 走 reconstruction 路线，虽然失去 generation 的 diversity，但物理 grounding 更强。在 dexterous manipulation 这个对 contact 极其敏感的 task 上，reconstruction 更可靠。

---

## 这篇 paper 的 bigger picture

### 1. Foundation Model Composition 是新范式

2026 年 robotics 的 pattern 已经清晰：**不再 monolithic train，而是 compose foundation models**。

VIDEOMANIP 的 pipeline 用了：MoGe-2, SAM 2, MeshyAI, GPT-4.1, FoundationPose, HaMeR, ContactOpt, AnyTeleop, DRO, DemoGen, DP3——11 个 component。

好处：每个 component 升级，整个 pipeline 受益。MoGe-2 明天出一个 v3 更准，VIDEOMANIP 直接换上就行。
坏处：compounding errors，debug 困难，pipeline 难以端到端优化。

这跟 LLM 时代的 "agentic workflow" 是一个套路——不是训一个大模型，而是组合一堆 specialized model + glue logic。

### 2. Dexterous Manipulation 的 "ImageNet moment" 可能不来自 VLA Foundation Models

$\pi_{0.5}$ 在这个 benchmark 上惨败。这说明 dexterous manipulation 还远没到"一个 foundation model 解决一切"的阶段。

我认为 dexterous manipulation 的 ImageNet moment 更可能来自 **internet-scale hand-object video reconstruction**——把 YouTube 上 1000 万个 cooking / crafting / 工具使用视频，用类似 VIDEOMANIP 的 pipeline 重建出 3D hand-object trajectories，作为 pre-training data。

这对应你之前讲过的 "Software 2.0" 延伸——不是 collect robot demos（teleop bottleneck 是 human hours），而是从 human video 间接 distill（vision model bottleneck 是 model accuracy，而 model 在快速进步）。

### 3. Reconstruction vs. Generation 路线之争

2025-2026 年 video-to-robot 有两条路线：
- **Reconstruction**（VIDEOMANIP, DexMV, Web2grasp）：显式恢复 3D geometry → retarget → policy。物理 grounding 强，diversity 弱。
- **Generation**（LVP, NovaFlow, Video2Policy）：video diffusion 预测未来 → 提取 motion → retarget。Diversity 强，物理 grounding 弱。

VIDEOMANIP 的实验清楚证明：**对 contact-rich 的 dexterous task，reconstruction 路线更可靠**。

但 generation 路线在 long-horizon planning 和 in-the-wild diversity 上有优势。未来大概率 hybrid：generation 做 augmentation 和 planning，reconstruction 做 grounding 和 verification。

### 4. 为什么 Single Video 能 work？

这是 paper 最 counterintuitive 的点。1 个 demo 学不出 robust policy 是常识。

但 DemoGen 的 SE(3) augmentation + DP3 的 point cloud representation 把 1 条变 1000 条 spatially randomized trajectories。

**核心 insight**：很多 manipulation task 的难度不在 trajectory 复杂度，而在 spatial generalization。SE(3) augmentation 正好 cover 这个 dimension。

但 Screw Bulb 表现最差（4/10）——因为旋拧动作需要 fine-grained finger coordination 和 force control，纯 spatial augmentation 不够，需要 temporal / force diversity。这暴露了 DemoGen 的边界：**它只 augment 空间，不 augment 时间和力**。

---

## Limitations，作者 honest 地讲了

1. **Compounding errors**：11 个 model 串联，error 难 trace
2. **Static camera**：假设相机不动，hand-held video 会崩
3. **Frozen contact during execution**：grasp 后假设 hand-object 相对 pose 不变，dynamic task 会失效
4. **Force closure 不保证**：Pan、Hand Bag 失败说明 kinematic grasp 不够
5. **Internet-scale video 还不行**：目前还是作者自己拍的"controlled in-the-wild"，真正 YouTube 视频 background 太复杂

---

## 一句话总结

**VIDEOMANIP 证明了：通过精心组合 vision foundation models + contact optimization + SE(3) data augmentation，单个 RGB 人手视频足以训练一个能泛化的灵巧手操作 policy。**

这是 dexterous manipulation data scaling 的一条 promising 路径——不是更多 teleop，而是更多 YouTube 视频 + 更好的 vision models。

参考：
- 项目主页：https://videomanip.github.io
- MoGe-2：https://arxiv.org/abs/2507.02546
- ContactOpt：https://arxiv.org/abs/2103.14234
- DRO-Grasp：https://arxiv.org/abs/2503.22221
- DemoGen：https://arxiv.org/abs/2502.16932
- DP3：https://arxiv.org/abs/2403.03954
- LVP（对比 baseline）：https://arxiv.org/abs/2512.15840
- $\pi_{0.5}$（对比 baseline）：https://arxiv.org/abs/2504.16054

如果你想深入某个 module 或者某个 experiment 细节，告诉我具体哪个，我可以展开。

---

# VIDEOMANIP: 从 RGB Human Videos 直接学习 Dexterous Manipulation

这篇 paper 解决的核心问题是：**是否可以仅用 RGB human videos（in-the-wild 录制的、无传感器、无机器人演示）直接训练灵巧手抓取和操作策略**。让我从 pipeline、技术细节、实验直觉、和相关延伸四个层面展开。

---

## 1. Problem Setting 与 Motivation

Dexterous manipulation 的瓶颈在于 data acquisition。传统路径有三个 expensive assumptions：

1. **Teleoperation with wearables**（如 DexCap [30] 用 mocap 手套，AINA [2] 用 smart glasses， headset-based tracking [3][4]）——硬件依赖强，scalability 差
2. **Multi-camera studio**（如 GigaHands [5]）——环境受限
3. **Pre-scanned object meshes**（如 DexMV [12]）——需要先验 CAD 知识

VIDEOMANIP 的 setting 极其 aggressive：单 video per task、egocentric static camera、无 camera intrinsics、无 object 尺寸先验、无 robot demonstration。这逼着作者去解决 metric scale、contact feasibility、trajectory diversity 三个核心难题。

参考：项目页面 https://videomanip.github.io

---

## 2. Pipeline 三大模块

### 2.1 3D Hand-Object Trajectory Reconstruction

输入：$\mathcal{V} = \{I_1, \ldots, I_T\}$，其中 $I_t \in \mathbb{R}^{H \times W \times 3}$。

**Step 1: Metric Depth + Intrinsics via MoGe-2 [51]**
MoGe-2 输出 metric depth map 和 camera intrinsics，建立一个 joint metric 3D coordinate frame。这是后续 hand 和 object 能对齐到同一物理空间的关键。MoGe-2 论文：https://arxiv.org/abs/2507.02546

**Step 2: Object Mesh Reconstruction + Scale Estimation**
- 用 SAM 2 [52] 拿到 object mask → crop → 喂给 MeshyAI（image-to-mesh 生成模型）→ 得到外观对、但 metric scale 不准的 mesh $\mathcal{O}$
- 两阶段 scale 估计：
  - **Coarse**：GPT-4.1 估计物理尺寸做粗 rescale
  - **Fine**：候选 scale ∈ [0.5×, 2×] 用 FoundationPose [54] 跑 pose estimation，rendering error 最小化：
  
  $$\text{scale}^* = \arg\min_s \sum_t \| \text{render}(\mathcal{O}_s, \xi_t) - M_t^{\text{SAM2}} \|$$
  
  其中 $\xi_t$ 是第 $t$ 帧的 6D pose，$M_t^{\text{SAM2}}$ 是 SAM 2 mask。这个 trick 解决了"image-to-mesh 没有 metric scale 但 FoundationPose 需要准确 scale"的矛盾。

**Step 3: Hand Mesh via HaMeR [37]**
HaMeR 输出 MANO 参数 $h = (\theta, \beta)$，$\theta$ 是 hand pose（关节角），$\beta$ 是 hand shape。HaMeR 用 weak-perspective camera model，depth-ambiguous。

修正：从 MoGe-2 metric depth map 上，在 HaMeR 预测的 2D keypoints 处采样深度值取平均，得到 corrected hand depth $t_z'$。这样 hand mesh $\mathcal{H}$ 和 object mesh $\mathcal{O}$ 共享同一 depth reference。

**Step 4: Retargeting via AnyTeleop [49]**
Human hand $(\theta_t, \beta_t)$ → robot joint config $q_t$，通过最小化 robot link keypoints 与 human hand joints 的距离：
$$q_t = \arg\min_q \sum_k \| \text{FK}_k(q) - p_k^{\text{human}} \|$$
$\text{FK}_k$ 是 forward kinematics 把第 $k$ 个 link 的位置算出来。给 URDF 文件，可以从 $q$ 生成 robot mesh $\mathcal{R}$。

### 2.2 In-the-Wild Calibration via GeoCalib [55]

In-the-wild video 没有 $\text{}^{\text{world}}T_{\text{cam}}$。GeoCalib 从单帧图像估计 gravity direction 在 camera frame 中的方向，得到旋转 $\text{}^{\text{grav}}R_{\text{cam}} \in \text{SO}(3)$，使得 $\mathbf{g}_{\text{cam}} = [0, 0, -1]^\top$（aligned with negative z-axis）。

把这个 rotation apply 到所有 reconstructed meshes $\mathcal{R}, \mathcal{O}$ 和 robot configs $q$，得到 gravity-aligned trajectories。这只是部分恢复 camera-to-world transform（缺少 translation），但足够把 in-the-wild trajectories 投到 robot table plane 上。

**直觉**：你不需要完整 extrinsic——只要 gravity 对齐了，pour tea / hang hat 这种 task 的物理 semantics 就正确了，因为 task-relevant 的 motion 主要在 horizontal plane + vertical lifting 上。

GeoCalib 论文：https://arxiv.org/abs/2504.16517 (ECCV 2024)

### 2.3 Contact Optimization + Interaction-Centric Grasp Modeling

这是 paper 的 key technical contribution 之一。Reconstructed grasps 经常 invalid——interpenetration 或 no contact（Fig 5(b)）。

**ContactOpt [47] Differentiable Optimization**

定义 hand pose-dependent contact map：

$$C_{\mathcal{O}}(v_{\mathcal{O}}^i; \mathbf{h}) = \max\left(0, 1 - \frac{\min_j \| v_{\mathcal{H}}^j(\mathbf{h}) - v_{\mathcal{O}}^i \|}{c_{\text{rad}}}\right)$$

变量说明：
- $v_{\mathcal{O}}^i$：object mesh 的第 $i$ 个顶点
- $v_{\mathcal{H}}^j(\mathbf{h})$：hand mesh 在 pose $\mathbf{h}$ 下的第 $j$ 个顶点
- $\min_j$：找 hand 上离 object vertex $i$ 最近的 vertex
- $c_{\text{rad}}$：控制 contact falloff 的半径参数，超过这个距离 contact 为 0

Contact value ∈ [0, 1]，distance 越小越接近 1。$C_{\mathcal{H}}(\mathbf{h})$ 对 hand vertices 对称定义。

ContactOpt 预测 desirable contact regions $\hat{C}_{\mathcal{H}}, \hat{C}_{\mathcal{O}}$（learning-based prior），然后用 differentiable objective 优化 hand pose：

$$E(\mathbf{h}) = |C_{\mathcal{O}}(\mathbf{h}) - \hat{C}_{\mathcal{O}}| + |C_{\mathcal{H}}(\mathbf{h}) - \hat{C}_{\mathcal{H}}|$$

直觉：这个 prior 把"哪里应该接触"从大规模 hand-object interaction dataset 学到的分布 inject 进来，弥补 reconstruction 误差。

**DRO Model [56]: Interaction-Centric Grasp Representation**

DRO（D(R,O)-Grasp）预测 robot hand point cloud $\mathbf{P}^{\mathcal{R}} \in \mathbb{R}^{N_{\mathcal{R}} \times 3}$ 与 object point cloud $\mathbf{P}^{\mathcal{O}} \in \mathbb{R}^{N_{\mathcal{O}} \times 3}$ 之间的 dense point-to-point distance matrix：

$$\mathcal{D}(\mathcal{R}, \mathcal{O})^{\text{Pred}} \in \mathbb{R}^{N_{\mathcal{R}} \times N_{\mathcal{O}}}$$

训练 loss 是 L1：
$$\mathcal{L} = \mathcal{L}_{\text{L1}}\left(\mathcal{D}(\mathcal{R}, \mathcal{O})^{\text{Pred}}, \mathcal{D}(\mathcal{R}, \mathcal{O})^{\text{GT}}\right)$$

输入是随机初始化的 robot hand point cloud $\mathbf{P}_{\text{init}}^{\mathcal{R}}$ 和 zero-centered object point cloud $\mathbf{P}^{\mathcal{O}}$。

推理时：用 multilateration [57]（一个代数解法，类似于 GPS 三角定位）从 distance matrix 和 object points 反解出 robot hand points 在 target grasp pose $\mathbf{P}_{\text{grasp}}^{\mathcal{R}}$ 的位置，再通过 optimization 反推 $q^{\text{grasp}}$。

**直觉**：DRO 不直接学 robot joint angles，而是学一个"shape-conditioned distance field"——这个 representation 可以跨 embodiment 迁移（cross-embodiment），因为 distance matrix 是 geometry-only 的。

DRO 论文：https://arxiv.org/abs/2503.22221

### 2.4 DemoGen [20]: Trajectory Synthesis for Generalization

单 video 只产生一条 trajectory，DP3 [58] 需要 ~1000 条才能 generalize。DemoGen 通过 SE(3) equivariant transformation 同步变换 object point cloud 和 robot trajectory：

$$\mathbf{P}^{\mathcal{O}\prime} = g \cdot \mathbf{P}^{\mathcal{O}}, \quad \tau' = g \cdot \tau, \quad g \in \text{SE}(3)$$

这保持 hand-object contact 不变（因为相对几何关系不变），但 spatial location 变化。Ablation（Fig 3(d)）：从 1 条 → 1000 条，Pour Tea success rate 从 1/15 → 13/15。

DemoGen 论文：https://arxiv.org/abs/2502.16932

### 2.5 DP3 Manipulation Policy [58]

3D diffusion policy，输入：
- Robot hand point cloud at grasp pose $\mathbf{P}_{\text{grasp}}^{\mathcal{R}}$
- Robot proprioceptive state $q^{\text{grasp}}$
- Object point cloud $\mathbf{P}^{\mathcal{O}}$

输出：$\Delta q$（关节角变化），closed-loop 执行。

一个关键 approximation：执行时 object 被 LEAP Hand 遮挡严重，DP3 无法 closed-loop track object。所以假设 grasp 后 hand-object relative pose 固定，object point cloud 通过 $q^{\text{grasp}}$ 计算的 hand-to-object transform 更新。这是 paper 的一个 limitation（见 Section V）。

DP3 论文：https://arxiv.org/abs/2403.03954

---

## 3. 实验：核心数据解读

### 3.1 Grasping（IsaacGym + Inspire Hand 18-DoF）

| 指标 | 数值 |
|------|------|
| Overall success（20 objects, single video each） | 63.75% |
| With ContactOpt optimization | 63.75% |
| Without ContactOpt | 30.7%（降 33 个点）|
| Augmented with 2 extra videos for 5 failed objects | 70.25% |
| Top-15 successfully grasped objects avg | 82.13% |

Disturbance test：300 步、6 个方向（±x, ±y, ±z）、力大小 = 0.5 × object mass；displacement < 3cm 算成功。

Failed objects：Pan（handle 被 hand 严重遮挡 → pose estimation 失败）、Hat、Bowl、Glasses Case、Hand Bag（force closure 不够，kinematically OK 但动态不稳）。

**Key insight**：reconstruction error 不是均匀分布的，occlusion-heavy regions 是主要失败模式。多 viewpoint videos 能互补——这指向"用 internet-scale videos 的 diversity 来 robustify reconstruction"的未来方向。

### 3.2 Manipulation（LEAP Hand + xArm 7-DoF, 7 tasks）

| Task | Video type | $\pi_{0.5}$ | LVP(-H) | LVP | Ours |
|------|-----------|-------------|---------|-----|------|
| Pour Tea | in-scene | 0/10 | 1/10 | 7/10 | **8/10** |
| Close Drawer | in-scene | 1/10 | 2/10 | 6/10 | **9/10** |
| Pick&Place Can | in-scene | – | 0/10 | 4/10 | **5/10** |
| Pour Tea (in-the-wild) | wild | 0/10 | 1/10 | 7/10 | **7/10** |
| Hang Hat | wild | 0/10 | – | 4/10 | **6/10** |
| Screw Bulb | wild | 1/10 | 0/10 | 1/10 | **4/10** |
| Move Jenga Box | wild | 0/10 | 0/10 | 2/10 | **5/10** |

平均 62.86%，比 LVP 高 15.87%。

**为什么 LVP [35] 输给 VIDEOMANIP**：LVP 走 video generation → retargeting 路线，pixel-level supervision 缺乏 3D geometry grounding。失败模式包括：
- Hallucinate robot gripper instead of human hand
- 不动（missed action）
- Infeasible grasps
- Confuse left/right hand
- Undetectable hands in generated frames

**为什么 $\pi_{0.5}$ [59] 表现差**：训练数据主要是 parallel-jaw gripper，LEAP Hand fine-tune 只有 200 demos 不够改变 dexterous 能力。$\pi_{0.5}$ 论文：https://arxiv.org/abs/2504.16054

### 3.3 In-the-wild Calibration Ablation

Pour Tea in-the-wild 不加 $\text{}^{\text{grav}}R_{\text{cam}}$ → success rate 从 7/10 暴跌到 0/10。

直觉：policy 学到的是 camera-frame specific 的 trajectory（比如倾斜的 pour 方向），部署时重力方向错了，tea 倒不出 bowl。

---

## 4. 与 Related Work 的精细对比

### 4.1 vs. DexMV [12]
DexMV 也从 human video 学 dexterous manipulation，但**需要 pre-scanned object meshes** + 多 camera 系统 + mocap 设备。VIDEOMANIP 把这三个 dependency 都去掉了。

### 4.2 vs. HOI pre-training [9] (Singh et al.)
HOI 在 video 上做 self-supervised pretraining，但下游 still 需要 robot demos fine-tune。VIDEOMANIP 直接从 video 到 policy，无 robot demos。

### 4.3 vs. DexWild [10] / SPIDER [50]
这些工作也 leverage in-the-wild videos，但 DexWild 需要 robot demos，SPIDER 专注 retargeting 而非 policy learning。

### 4.4 vs. Web2grasp [14]（同一作者前作）
Web2grasp 用 web images 学 functional grasps，但需要 heavy filtering。VIDEOMANIP 用 ContactOpt 替代 filtering，更 robust。

### 4.5 vs. LVP [35] / Video Generation Methods
LVP 用 video generator 预测未来 hand motion 再 retarget。问题：generation 是 pixel-level，没有显式 3D contact model，会 hallucinate infeasible motion。VIDEOMANIP 走 reconstruction-based 路线，物理 grounding 更强但失去 generation 的 diversity。这两条路线其实可以互补——future work 可以用 video generation 做 augmentation，再过 reconstruction pipeline。

### 4.6 vs. AINA [2]
AINA 用 smart glasses + point cloud，依赖硬件。VIDEOMANIP 只用 RGB。

---

## 5. 技术深度的几个细节

### 5.1 为什么需要 metric depth 而不是 relative depth？

Hand mesh（HaMeR）和 object mesh（MeshyAI）来自不同 model，scale 不一致。MoGe-2 提供 metric depth 作为共同 reference，把两者 anchor 到同一物理空间。没有 metric scale，grasp 的 finger placement 会偏移。

### 5.2 GPT-4.1 在 scale estimation 中的作用

这是个有意思的 trick——LLM 作为 object size prior。比如"apple 通常直径 8cm"。给个 coarse initialization，FoundationPose 再 fine-tune。如果 LLM 估计错 50%，fine stage 的 candidate range [0.5×, 2×] 还能 cover。这种 LLM-as-prior + CV refinement 的 hybrid pattern 值得关注。

### 5.3 Multilateration [57] 在 DRO 中的作用

给定 distance matrix $\mathcal{D} \in \mathbb{R}^{N_R \times N_O}$ 和 object points $\mathbf{P}^{\mathcal{O}}$，要反解 hand points $\mathbf{P}^{\mathcal{R}}$ 的位置。

每个 hand point $i$ 满足：$\| \mathbf{P}_{\mathcal{R}}^i - \mathbf{P}_{\mathcal{O}}^j \| = \mathcal{D}_{ij}$ for some $j$。

这是经典 multilateration 问题（GPS 定位原理），可以代数 closed-form 解。

### 5.4 SE(3) Equivariance in DemoGen

为什么 DemoGen 的 SE(3) augmentation 有效？因为 DP3 input 是 point cloud，point cloud 天然 SE(3)-equivariant（旋转/平移 point cloud 等价于旋转/平移 reference frame）。同时 robot trajectory 在 SE(3) 下也可以一致 transform。所以 augmentation 不破坏 hand-object contact 物理一致性。

### 5.5 Closed-Loop Approximation 的风险

执行时假设 hand-object relative pose 固定。这对 Pour Tea、Close Drawer 这种 quasi-static task 还行，但对需要 dynamic re-grasp 的 task（比如 in-hand rotation）会失效。这是为啥 paper 没做 in-hand manipulation task 的原因之一。

---

## 6. Limitations 与 Future Directions（作者自陈 + 我的延伸）

### 6.1 Compounding Errors
Pipeline 串联 MoGe-2 → SAM 2 → MeshyAI → FoundationPose → HaMeR → ContactOpt → DRO → DemoGen → DP3。每一环都有 error，最终 effect 难以 trace。作者 future work 提到 trajectory verification module。

**我的延伸**：可以借鉴 VLM-as-judge 的思路，用 GPT-4V 评估 reconstructed trajectory 的物理合理性。或者用 dynamics simulation 做 forward rollout 验证。

### 6.2 Static Camera Assumption
所有 video 假设 egocentric static camera。In-the-wild hand-held video 会有 camera motion，gravity alignment 还不够——需要 full SLAM。作者引用 C4D [60] 和 Dyn-HAMR [61] 作为 future direction。

### 6.3 Object Tracking During Execution
Grasp 后 object point cloud 是"冻结的"——假设 contact 不变。这对 rigid object + stable grasp OK，对 deformable object 或 sliding contact 会崩。

**延伸**：可以用 tactile sensing（如 DIGIT [MIT]）做 closed-loop contact verification，但这违反"device-free"约束。或者用 visual 6D pose tracker（如 FoundationPose 的 tracking mode）实时更新 object pose——但 LEAP Hand 遮挡严重，需要 multi-view camera setup。

### 6.4 Force Closure 不保证
Pan、Hand Bag 失败案例说明 kinematic grasp 不够，需要 force analysis。可以引入 grasp wrench space (GWS) analysis 或者 differentiable physics（如 DiffSim）做 post-hoc filtering。

### 6.5 Internet-Scale Videos
目前还是 controlled "in-the-wild"（作者自己拍）。真正的 internet video（YouTube cooking videos 等）会有 extreme camera motion、complex backgrounds、multi-object interaction——paper 的 pipeline 撑不住。

**延伸**：可以结合 recent video understanding models（如 Video-LLaMA, InternVideo2）做 semantic parsing + segment task-relevant clip，再过 reconstruction pipeline。

---

## 7. 我对这篇 paper 的几点直觉

### 7.1 "Reconstruction vs. Generation" 路线之争
2025 年 robotics learning 有两条主流：
- **Reconstruction**：从 video 显式恢复 3D geometry → retarget → policy。代表：VIDEOMANIP, DexMV, Web2grasp。
- **Generation**：video diffusion 预测 future → extract motion → retarget。代表：LVP, NovaFlow [27], Video2Policy [16]。

VIDEOMANIP 的实验清楚证明 reconstruction 路线在 dexterous manipulation 上更 reliable，因为 3D contact geometry 是 dexterous 任务的关键。但 generation 路线在 diversity 和 in-the-wild scalability 上有优势。**未来大概率是 hybrid**：generation 做 augmentation，reconstruction 做 grounding。

### 7.2 Foundation Models as Building Blocks
这篇 paper 体现了 2025 年 robotics 的一个 pattern：**不再 train-from-scratch，而是 compose foundation models**。MoGe-2, SAM 2, HaMeR, FoundationPose, ContactOpt, DRO, DemoGen, DP3——每一个都是 community 的成熟 model。VIDEOMANIP 的贡献是 composition strategy + 两个 glue components（contact optimization + DemoGen synthesis）。

这种 modular approach 的好处是 scalability——每个 foundation model 升级，整个 pipeline 都受益。坏处是 compounding errors 和 debugging 困难。

### 7.3 为什么 Single Video per Task 能 work？
这是 paper 最 surprising 的点。直觉上 single demo 学不出 robust policy。但 DemoGen 的 SE(3) augmentation + DP3 的 point cloud representation 让 1 个 demo 变成 ~1000 个 spatially randomized trajectories。

**核心 insight**：很多 manipulation task 的 difficulty 不在 trajectory complexity，而在 spatial generalization。SE(3) augmentation 正好 cover 这个。

但 Screw Bulb 表现最差（4/10），因为旋拧动作的 fine-grained finger coordination 难以通过纯 spatial augmentation 学好——需要 temporal / force diversity。

### 7.4 与 Foundation Models（π0.5, RT-2, Octo）的关系
π0.5 在这个 benchmark 上惨败（平均 ~3%），说明 dexterous manipulation 还远没到 foundation model 可解决的阶段。原因：
- 训练数据中 parallel-jaw gripper 占主导
- Multi-finger coordination 的 action space 复杂度是 gripper 的 ~10×
- Tactile / contact-rich data 严重缺乏

**延伸思考**：dexterous manipulation 的 "ImageNet moment" 可能不来自 VLA foundation models，而是来自大规模 hand-object interaction video reconstruction（比如 VIDEOMANIP scaled to 10M YouTube videos）。这条 path 对应 Karpathy 你之前提的 "software 2.0" 在 robotics 上的延伸——不是收集 robot demos，而是从 human video 间接 distill。

参考：DROID dataset https://droid-dataset.com，Open X-Embodiment https://robotics-transformer-x.github.io

### 7.5 关于 Embodiment Gap
Human hand → LEAP Hand / Inspire Hand 有显著 morphology gap。Retargeting 通过 keypoint matching 缓解，但 finger length、joint limit、actuator dynamics 都不同。这是 manipulation learning 的 fundamental challenge。

**延伸**：可以用 differentiable simulation（如 MuJoCo MJX, Brax）做 cross-embodiment optimization——把 human trajectory 作为 soft constraint，在 robot dynamics 下找最优 trajectory。SPIDER [50] 在这个方向有探索：https://arxiv.org/abs/2511.09484

### 7.6 与 Karpathy 你之前讲过的 "Software 2.0" / "Software 3.0" 的联系
- Software 1.0：手写 heuristic 控制 robot
- Software 2.0：用 NN 学 policy from demos（imitation learning）
- Software 3.0：用 LLM/VLM 做 reasoning + planning

VIDEOMANIP 在 Software 2.0 范式内，但它的 data 来源是 human video 而非 robot demos——这其实指向一个"Software 2.5"：用 vision foundation models 把非 robot data distill 成 robot data。这个 pattern 在 2025-2026 会越来越主流。

---

## 8. 推荐阅读路径

如果想深入理解：

1. **Foundation components**：
   - MoGe-2: https://arxiv.org/abs/2507.02546
   - HaMeR: https://arxiv.org/abs/2312.13262
   - FoundationPose: https://arxiv.org/abs/2312.08344
   - SAM 2: https://arxiv.org/abs/2408.00714
   - GeoCalib: https://www.geocalib.github.io

2. **Contact & Grasp**:
   - ContactOpt: https://arxiv.org/abs/2103.14234
   - DRO-Grasp: https://arxiv.org/abs/2503.22221
   - DexGraspNet (large grasp dataset): https://sites.google.com/view/dexgraspnet

3. **Trajectory Synthesis**:
   - DemoGen: https://arxiv.org/abs/2502.16932
   - MimicGen (predecessor): https://arxiv.org/abs/2306.00937
   - RoboCasa (similar synthesis idea): https://arxiv.org/abs/2306.00937

4. **Policy Learning**:
   - DP3: https://arxiv.org/abs/2403.03954
   - Diffusion Policy: https://arxiv.org/abs/2303.04137

5. **Related Video-to-Robot works**:
   - DexMV: https://arxiv.org/abs/2108.05877
   - Web2grasp (same group): https://arxiv.org/abs/2505.05517
   - LVP: https://arxiv.org/abs/2512.15840
   - Track2Act: https://arxiv.org/abs/2401.17615
   - Vid2Robot: https://arxiv.org/abs/2403.12943

6. **Dexterous Hands**:
   - LEAP Hand: https://leaphand.com/
   - Inspire Hand: https://www.inspire-render.com/en/inspire-hand

---

## 9. 总结

VIDEOMANIP 是 2026 年初 dexterous manipulation learning 的一个重要 milestone。它证明：

1. **Pipeline-level innovation** 比 single-model breakthrough 在 robotics 更 feasible——compose vision foundation models 能解决"从 video 到 robot action"的鸿沟
2. **Contact modeling** 是 dexterous manipulation 区别于 gripper manipulation 的核心——ContactOpt + DRO 把这个做到 practical
3. **Data augmentation via SE(3) synthesis** 让 single-video learning 变 feasible——这降低 data collection cost 一个数量级
4. **Reconstruction > Generation** for dexterous tasks in 2026——但 generation 路线在 long-horizon diversity 上仍占优

它的 limitation（compounding errors, static camera, frozen contact during execution）指明了 2026-2027 年的关键 research direction：dynamic scene reconstruction for manipulation, tactile-in-the-loop closed-loop control, internet-scale video-to-trajectory pipelines。

如果你在思考的"如何 scale robotics data"问题，这条路线（vision foundation models + contact-aware reconstruction + SE(3) augmentation）是一个非常 promising 的非 teleop 路径。 teleop 的 bottleneck 是 human hours，video reconstruction 的 bottleneck 是 vision model accuracy——而 vision model 在快速进步，所以这条 path 的 marginal cost 在下降。
