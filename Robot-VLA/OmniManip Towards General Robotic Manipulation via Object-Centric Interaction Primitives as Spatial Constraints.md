---
source_pdf: OmniManip Towards General Robotic Manipulation via Object-Centric Interaction
  Primitives as Spatial Constraints.pdf
paper_sha256: 63679d2250f492ce242c1a134c2352003f2d3a8b5b52905535b33ee3adf96f2c
processed_at: '2026-08-05T23:21:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 OmniManip

好的 Karpathy，我换个腔调，把 academic 包装扒掉，用大白话讲讲这 paper 到底在干啥。

---

## 一句话概括

**让 VLM 在物体的"标准坐标系"里思考 manipulation，而不是在像素或世界坐标系里瞎猜。**

就这么一句话。剩下的 engineering 都是为了让这个 idea 跑得稳。

---

## 之前方法的痛点在哪

你想象一个 robot 要"倒茶"。VLM 听得懂这句话，它知道 teapot 要 tilt 过来、spout 对准 cup。但 VLM 是 2D 图片训练出来的，它对 3D 空间的精确位置其实没概念。

社区里两条路：

**路一 (VLA)**: 直接 fine-tune VLM 让它输出 action chunk。问题是要海量 teleop data，而且 representation 被某台 robot 的 proprioception 锁死，换台 robot 就废了。RT-2、OpenVLA、π0 都在这条路上。https://openvla.github.io/

**路二 (VLM-as-planner)**: 让 VLM 只负责 high-level reasoning，输出一些 spatial primitive（point、direction、constraint），然后交给传统 motion planner 去执行。VoxPoser、ReKep、CoPa 在这条路。https://voxposer.github.io/ , https://rekep-robot.github.io/

OmniManip 走路二，但指出路二前人有两个 bug：

1. **Primitives 是在物体表面 / image plane 上采样的**。比如 ReKep 用 semantic clustering 在 mesh surface 上聚一堆 keypoint，让 VLM 挑。问题是从正面看 battery，你想抓的那个接触点在物体内部，根本采不到。Table 2 实测：ReKep 在 frontal view 下 0/10，top-down view 下 7/10。view 一变 primitive 就崩。

2. **Primitives 跟 task 没必然关系**。VLM 在一堆 random surface points 里挑 task-relevant 的那个，挑错了就 hallucinate，plan 是 open-loop 的，执行完才发现错。

---

## OmniManip 的 trick

**核心 insight**：每个物体有个"canonical space"，这个坐标系是被物体的功能定义的。

举个例子。teapot 的 canonical frame 大概是：
- +x 轴：spout 指向（因为倒水方向是 functional axis）
- +z 轴：lid 朝上（因为打开方向是 functional axis）
- +y 轴：handle 朝外（因为抓握方向是 functional axis）

这个 frame 不是随便选的，它是 teapot 这个物体"用来干什么"的几何投影。叫做 **functional affordance defines canonical frame**。这条 cognitive science 里有 support，Gibson 1979 那本 ecological perception 是经典 reference。

那么"倒茶"这个 task 在 canonical space 里描述就极简单：
- interaction point $\mathbf{p}$ = 壶口中心（canonical space 里固定一个点）
- interaction direction $\mathbf{v}$ = +x（spout 方向）

不管你相机从哪个角度拍，不管 teapot 摆在桌上哪个位置，这两个 primitive 在 canonical space 里永远是同一个。 viewpoint invariance 自动拿到。

实现上怎么落地：
1. 用 GroundingDINO + SAM 把 scene 里所有物体抠出来 https://github.com/IDEA-Research/GroundingDINO , https://github.com/facebookresearch/segment-anything
2. 用 single-view 3D AIGC（TripoSR / One-2-3-45++）从一张图重建 mesh https://arxiv.org/abs/2403.02151
3. 用 Omni6DPose（同组 prior work）估计 6D pose 把物体 canonicalize 到 functional frame https://jzyao.github.io/Omni6DPose/
4. 在 canonical space 里 VLM 推断 $\mathbf{p}$ 和 $\mathbf{v}$

---

## Primitive 怎么提取

**Interaction point**：用 SCAFFOLD 的 trick，在 image 上画 Cartesian grid，让 VLM 报 (x, y) 坐标。Visible point（比如 teapot handle）直接定位；invisible point（比如 teapot 壶口中心，从外面看不到）用 multi-view reasoning：主视角歧义就切 orthogonal view 推断。然后用 RGB-D 把 2D 坐标反投影到 3D canonical space。https://arxiv.org/abs/2402.12058

**Interaction direction**：这是最 elegant 的设计。不让 VLM 在 SO(3) 上 uniform sample（无穷多个方向，VLM 选不过来），而是只取 object 在 canonical space 的 3 个 principal axes（±x, ±y, ±z），共 6 个候选方向。然后 VLM 给每个 axis 写个 caption（"this axis points along the spout"），LLM 评分 caption 跟 task 的相关性，排序后选 top-1。

Table 3 实测：在 "Recycle Battery" 上，uniform SO(3) sampling 是 50% success，OmniManip 的 principal-axis sampling 是 80%。Iteration 数差不多（1.8 vs 1.7），说明不是 search 更努力，是 candidate pool 质量更高。

---

## Dual Closed-Loop —— 真正的 novelty

之前方法最多一个 loop（ReKep 用 point tracking 做 execution loop，planning 还是 open-loop）。OmniManip 两个 loop。

### Loop 1: Planning 的 RRC (Resample-Render-Check)

VLM 推断完 primitive 和 constraint 后，先不执行，而是：

1. **Render**：根据 candidate constraint 把"假如执行后"的场景 render 成一张图。比如 teapot 已经 tilt 到位、spout 对准 cup 之上 5cm。
2. **Check**：把这张图丢回 VLM，问它 "this looks like successful tea pouring?" VLM 返回 success / fail / refine。
3. **Resample**：如果 refine，在原 direction $\mathbf{v}$ 周围均匀采 6 个新方向，重新 render、check。

Intuition：VLM 一次说"spout 对准 cup"可能是 hallucination，但你给它 render 一张图，它一眼就能看出来对歪了。这本质是把 chain-of-thought 外化成 visual chain。VLM 自己 review 自己的 plan，比一次性 inference 靠谱得多。

Table 1 显示关掉这个 loop，rigid task 从 68.3% 掉到 51.7%，articulated task 从 61.7% 掉到 45%。15-20% 的提升是 RRC 拿到的。

### Loop 2: Execution 的 pose tracking

Plan 定下来后，execution 是个 MPC-style optimization：

$$\mathbf{P}^{ee*} = \arg\min_{\mathbf{P}^{ee}} \{\mathcal{L}_C + \mathcal{L}_{collision} + \mathcal{L}_{path}\}$$

- $\mathcal{L}_C$：当前 active/passive object 位姿与 target constraint $\mathcal{C}$ 的偏差
- $\mathcal{L}_{collision}$：end-effector 跟障碍物距离的 hinge loss
- $\mathcal{L}_{path}$：跟当前位姿的 translation/rotation 平滑度

关键在于 active/passive object 的当前 pose $\mathbf{P}_t^{active}, \mathbf{P}_t^{passive}$ 不是估计一次就完了，而是用 6D pose tracker **实时更新**。这样物体在 task 中途被碰歪、被移动、被遮挡，system 都能跟上。

为什么这事重要？Figure 8 给两个典型 failure case：
- 抓 teapot 时 grasp 偏了一点，teapot 在 gripper 里 shifted，整个 active object pose 跟 plan 假设的不一致
- target cup 在 task 中途被 robot 碰歪了

ReKep 用 point tracking 做 closed-loop，但 point 一旦被 occluded 就 track 不到，reported 47% failure rate。OmniManip 用 object-level 6D pose tracking（基于 mesh + RGBD），即使 primitive point 本身不可见，也能从 object pose 反算 canonical-space primitive 位置。这是 object-centric representation 的 inherent 优势。

---

## 为什么这思路有价值

我看完 paper 的 intuition 是：**这是一个 structured representation + VLM reasoning + classical planning 的 clean blueprint**。

它给的启示是：你不需要 fine-tune VLM，只要给 VLM 一个 structured 3D substrate（canonical space primitives）和一个 self-check 机制（RRC），它就能 zero-shot 解决精确 manipulation。

短期看，这类 structured approach 在 data efficiency 和 generalization 上吊打 VLA。长期看，VLA 可能用这类 system 做 data engine 来 bootstrap 自己 —— Table 4 已经显示端倪：用 OmniManip zero-shot 生成 150 条 trajectory 训 diffusion policy，能拿到 86-95% 成功率。这就是把 symbolic reasoning distill 进 implicit policy 的雏形。

https://diffusion-policy.cs.columbia.edu/

---

## 我看完的几个 concern

1. **Pipeline 太重**。GroundingDINO + SAM + 3D AIGC + Omni6DPose + VLM × 多次 + 6D tracker + motion planner，每一步都有 failure mode，复合 reliability 有限。real-time 跑应该挺慢。

2. **Mesh quality 是 bottleneck**。3D AIGC 对 transparent / thin / texture-less object 还是 noisy。teapot lid 在主实验里只有 5/10 成功率，明显是被 mesh 重建拖累的。

3. **Canonical frame 的 "functional alignment" 是个强假设**。Omni6DPose 必须把 object pose 估计到 functional canonical frame，而不是任意 canonical frame。对"非典型使用方式"的物体（用 teapot 当花瓶）就会出问题。Functional frame 的定义本身有歧义。

4. **Stage decomposition 全靠 VLM**。对 long-horizon task（"泡一杯茶"包含开盖、抓壶、倒水、放回、开盖、加茶、搅拌）VLM 容易 hallucinate stage 切分。Paper 里没 stress test 这个。

5. **没 bimanual**。整个 formulation 是 single end-effector 的，折衣服、拧瓶盖这种 bimanual task 需要重新设计 primitive 关系。

6. **6D pose tracker 在 specular / transparent 上会崩**。可以结合 recent tactile-based pose refinement 来 mitigate，比如北大同组的 visuo-tactile work。https://arxiv.org/abs/2409.17549

---

## 跟其他 line 的 connection

我读的时候冒出来几个联想：

- **Object-centric RL**：canonical primitive 这个 idea 跟 NOCS 那条 category-level manipulation line 是远房亲戚。https://arxiv.org/abs/1901.02970
- **Affordance theory**：canonical axes = functional axes 这事直接对应 Gibson 1979 的 affordance theory，"geometry is shaped by function"。
- **Visual Servoing**：closed-loop execution via pose tracking 本质是 PBVS（Position-Based Visual Servoing），Hutchinson 1996 那篇 tutorial。https://www.cs.cmu.edu/~hager/tt/tt-paper-5.pdf
- **NeuroSymbolic**：整个 system 是 neuro（VLM）+ symbolic（constraint optimization）的 hybrid。这跟 recent "NeuroSymbolic RL" 思路一致。
- **TAMP**：Eq. (2)-(5) 是经典 numerical IK with constraints，跟 CuRobo 的思路一致。https://curobo.org/
- **VLM-as-judge**：RRC 里的 "VLM 检查自己 render 出来的图" 跟 recent VLM-as-judge line 是一脉相承的。https://arxiv.org/abs/2310.12875

---

## 整体 takeaway

这 paper 不在 SOTA 数字上 revolutionary，但它给社区一个 **clean blueprint**：VLM 不需要 fine-tune，只要给它 structured 3D substrate 和 self-check 机制，就能 zero-shot 干精确 manipulation。

我个人觉得最 exciting 的不是 manipulation 本身，是 **Table 4 那个 demo generation 应用**。用 zero-shot structured planner 批量生成 trajectory 给 imitation learning policy 当训练数据，绕开 teleop bottleneck —— 这条路如果 scale 起来，可能比 VLA 直接 fine-tune 更早 unlock general robot manipulation。

短期看 OmniManip 这类 structured approach 占优，长期看 VLA 可能用 OmniManip 做 data engine bootstrap 自己。这个 symbiosis 是我最想看的下一步。

---

References：
- Paper: https://omnimanip.github.io
- ReKep (主 baseline): https://rekep-robot.github.io/
- VoxPoser: https://voxposer.github.io/
- CoPa: https://arxiv.org/abs/2403.08248
- Omni6DPose: https://jzyao.github.io/Omni6DPose/
- FoundationPose: https://foundationpose.github.io/
- TripoSR: https://arxiv.org/abs/2403.02151
- SCAFFOLD: https://arxiv.org/abs/2402.12058
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- UMI: https://universal-mani.github.io/
- OpenVLA: https://openvla.github.io/
- NOCS: https://arxiv.org/abs/1901.02970
- CuRobo: https://curobo.org/

---

# OmniManip: 一个 Object-Centric Approach 来 Bridge VLM 与 Manipulation

Karpathy 你好，这篇 paper 我读完之后感觉非常有意思，因为它触及了一个我一直很关心的核心问题：**如何让 VLM 这种 2D-pixel-pretrained 的 reasoning 引擎真正驱动 3D 精确 manipulation**。下面我从 intuition、公式细节、架构、实验几个层面深挖一下。

---

## 1. 高层 Motivation：为什么这是一个 hard problem

当前 robotics 社区大致有两条路线：

1. **VLA route** (RT-1/RT-2, OpenVLA, π0): fine-tune VLM on robotic data，直接输出 action。问题是 data cost 巨大、agent-specific representation 锁死了 generalization。
   - 参考：https://openvla.github.io/ , https://pi0.org , https://robottransformer.github.io/

2. **VLM-as-planner route** (VoxPoser, ReKep, CoPa, MOKA): VLM 负责 high-level reasoning，输出某种 spatial primitive / value map，再交给传统 motion planner 执行。
   - VoxPoser: https://voxposer.github.io/
   - ReKep: https://rekep-robot.github.io/
   - CoPa: https://arxiv.org/abs/2403.08248
   - MOKA: https://moka-manipulation.github.io/

OmniManip 走的是第二条路，但指出前人方法的几个关键缺陷：

- **Primitives proposal 是 task-agnostic 的**：ReKep 用 semantic clustering 在 surface 上聚类 keypoint，CoPa 用 pixel segmentation 提 parts，但这些 proposal 经常与 task 不相关，导致 VLM 在一堆无意义 candidates 里挑。
- **Primitives 是 view-dependent 的**：直接在 image plane / object surface 上 sampling，viewpoint 一变 primitive 就变了。Table 2 显示 ReKep 在 0° frontal view 下 0/10 成功，90° top-down 才 7/10，这就是个实证。
- **Open-loop planning**：VLM 一次 inference 出 plan 后不能验证，hallucination 直接导致 failure。

---

## 2. Core Insight：Canonical Space 作为 Reasoning Substrate

OmniManip 的核心 insight 是 **object 的 canonical space 是由 functional affordances 定义的**，所以在 canonical space 里描述 interaction primitive 会自带 semantic structure 和 viewpoint invariance。

举例：teapot 的 canonical space 通常是以"壶嘴方向为 +x、壶口朝上 +z、手柄朝外 +y"这样的 functional frame。那么"pour tea" 这个 task 的 interaction direction 自然就落在 +x 或 -x 轴附近，而 interaction point 自然是壶口中心或手柄中心。这种 alignment 是 prior work 在 surface 上采点所没有的。

这背后的物理直觉：**functional artifacts 的几何是被 affordance 塑造的**，所以 principal axes 本身就是 affordance 的 proxy。这一点和 "Shape for function" 这条认知科学线是一致的，参考：
- https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30123-6

具体实现上：
- 用 single-view 3D generation (One-2-3-45++, TripoSR, TriplaneGaussian) 重建 mesh
- 用 Omni6DPose / FoundationPose 做 6D pose estimation canonicalize 到 functional frame
- 参考：https://arxiv.org/abs/2403.02151 (TripoSR), https://foundationpose.github.io/ , https://jzyao.github.io/Omni6DPose/

---

## 3. Formulation 细节

### 3.1 Stage decomposition

Task $T$ 被分解成 stages $s = \{S_1, S_2, ..., S_n\}$，每个 stage：

$$S_i = \{A_i, \mathcal{O}_i^{active}, \mathcal{O}_i^{passive}\}$$

- $A_i$: action verb (grasp, pour, insert, ...)
- $\mathcal{O}_i^{active}$: the object that initiates interaction (e.g. teapot 在 pour 阶段)
- $\mathcal{O}_i^{passive}$: the object being acted upon (e.g. cup 在 pour 阶段)

注意 active/passive 角色在同一个 task 的不同 stage 是会 swap 的 —— teapot 在 grasp 阶段是 passive，在 pour 阶段变成 active。这个 role switching 在 ReKep 里被 relational keypoint 隐式表达，OmniManip 是 explicit 的。

### 3.2 Interaction primitive

每个 object 的 primitive：

$$\mathcal{O} = \{\mathbf{p}, \mathbf{v}\}, \quad \mathbf{p} \in \mathbb{R}^3, \quad \mathbf{v} \in \mathbb{R}^3$$

- $\mathbf{p}$: interaction point (在 canonical space 里定义，比如壶口中心、抽屉把手中心)
- $\mathbf{v}$: interaction direction (沿 principal axes 之一，比如抽屉拉出方向)

整套 spatial constraint 在 stage $i$ 是：

$$\mathcal{C}_i = \{\mathcal{O}_i^{active}, \mathcal{O}_i^{passive}, d_i, \theta_i\} \tag{1}$$

- $d_i$: distance constraint (active point 到 passive point 的目标距离，pour 时壶口高于杯口 5cm)
- $\theta_i$: angular constraint (active direction 到 passive direction 的目标夹角，pour 时壶嘴朝向与杯口法向反向对齐)

这个 formulation 把 manipulation task 变成了一个 geometric constraint satisfaction problem，是经典的 TAMP (Task and Motion Planning) 思路 (Kaelbling & Lozano-Pérez)。和 ReKep 的 relational keypoint constraint 思路很接近，但关键区别在 primitive 是在 **canonical space** 定义的，不是 world frame。

### 3.3 Primitive extraction 的两个 tricks

**Interaction point grounding**: 用 SCAFFOLD (https://arxiv.org/abs/2402.12058) 在 image 上画 Cartesian grid，让 VLM 报坐标。Visible point 直接定位；invisible point (e.g. 壶口中心) 通过 multi-view reasoning —— 主视角有歧义时切到 orthogonal view 推断。这里 OmniManip 用了 RGB-D + reconstructed mesh 把 2D grid 点反投影到 3D canonical space。

**Interaction direction sampling**: 这是最 elegant 的设计之一。不直接在 SO(3) 上 uniform sample，而是先提取 object 在 canonical space 的 3 个 principal axes（PCA 或 mesh 的 inertial axes），然后让 VLM 给每个 axis 写一个 semantic caption ("this axis points along the spout of the teapot"), 再让 LLM 评分这些 caption 与 task 的相关性。这样把 SO(3) 的 3 维 search 压缩到 6 个 candidate directions (±x, ±y, ±z)，极大地降低了 VLM 的选择难度。

Table 3 给出了一个 ablation：在 "Recycle Battery" 和 "Pour Tea" 上对比 uniform SO(3) sampling 和 OmniManip 的 principal-axis sampling：
- Battery: 50% → 80% success, iteration 1.8 → 1.7
- Tea: 30% → 70% success, iteration 3.4 → 1.8

Iteration 数差不多但 success rate 大幅提升，说明 candidate quality 提升是主要 driver —— 这正是 canonical space alignment 的价值。

---

## 4. Dual Closed-Loop System —— 这是真正的 novelty

之前 work 大多是单 loop (ReKep 是 execution loop via point tracking，CoPa/VoxPoser 基本开环 planning)。OmniManip 提了 dual loop。

### 4.1 Closed-loop planning via RRC (Resample-Render-Check)

Algorithm 1 的精髓：

```
Input: task T, stage S_i, candidate constraints K_i = {C_i^(1), ..., C_i^(N)}
for k = 1..N:
    I_i = Render(C_i^(k))    # 在当前 scene 下 render 出 active/passive primitive 的相对位姿
    state = VLM(T, S_i, I_i, refine)   # VLM 看 render 出来的图判断 success/fail/refine
    if state == "Refine" and not refined:
        K_i = Resample(C_i^(k))  # 在 v_i 周围均匀采 6 个 directions
        restart loop
    elif state == "Success":
        return C_i^(k)
return Fail
```

Intuition：把 VLM 的"hallucinated plan"通过 **rendering 转回 image** 让 VLM 自己审查。这本质是把 chain-of-thought 外化成 visual chain —— VLM 一次说"壶嘴对准杯子"是幻觉，但给它 render 一张图，它一眼就能看出来对歪了。这是 self-correction 的关键。

这种思路在 visual reasoning 的 LLM-as-judge line 里有先例，比如 VIBE / VLM-as-judge 之类，参考 https://arxiv.org/abs/2310.12875。但 OmniManip 把它具体化到 manipulation planning 的 spatial constraint validation 上，且 refine 阶段做了 local resampling（绕 v_i 采 6 个方向），这是一个 coarse-to-fine search。

### 4.2 Closed-loop execution via 6D pose tracking

Execution 是一个 optimization：

$$\mathbf{P}^{ee*} = \arg\min_{\mathbf{P}^{ee}} \sum_{j=1}^{N} \mathcal{L}_j(\mathbf{P}^{ee}) \tag{2}$$

with $\mathcal{L} = \{\mathcal{L}_C, \mathcal{L}_{collision}, \mathcal{L}_{path}\}$.

**Constraint loss**:
$$\mathcal{L}_C = \rho(\mathcal{C}, \mathbf{P}_t^{active}, \mathbf{P}_t^{passive}), \quad \mathbf{P}_t^{active} = \Phi(\mathbf{P}_t^{ee}) \tag{3}$$

- $\mathcal{C}$: target constraint (来自 planning)
- $\mathbf{P}_t^{active}, \mathbf{P}_t^{passive}$: 当前时刻 active 和 passive object 的 6D pose（从 6D pose tracker 来）
- $\Phi(\cdot)$: forward kinematics map，把 end-effector pose 映射到 grasped active object 的 pose（已知 grasp transform）
- $\rho(\cdot)$: 一般是 $|d - d_{target}|^2 + |\theta - \theta_{target}|^2$ 这种，paper 里没写具体形式

**Collision loss**:
$$\mathcal{L}_{collision} = \sum_{j=1}^N \max(0, d_{min} - d(\mathbf{P}^{ee}, \mathbf{O}_j))^2 \tag{4}$$

- $\mathbf{O}_j$: 第 j 个 obstacle
- $d(\mathbf{P}^{ee}, \mathbf{O}_j)$: end-effector 到 obstacle 的 signed distance
- $d_{min}$: safety margin
- 这是一个 hinge loss，只在距离小于 safety margin 时才 penalize

**Path smoothness loss**:
$$\mathcal{L}_{path} = \lambda_1 d_{trans}(\mathbf{P}_t^{ee}, \mathbf{P}^{ee}) + \lambda_2 d_{rot}(\mathbf{P}_t^{ee}, \mathbf{P}^{ee}) \tag{5}$$

- $\mathbf{P}_t^{ee}$: 当前 end-effector pose
- $\mathbf{P}^{ee}$: candidate next pose
- $d_{trans}, d_{rot}$: translation 和 rotation 的 displacement (rotation 一般用 geodesic distance on SO(3))
- $\lambda_1, \lambda_2$: weighting factors

这是一个 weighted-sum multi-objective optimization，相当于 MPC 风格的 cost function。整个式子写成 one-step MPC，rollout horizon = 1，比较保守但 fast，适合 real-time。

**为什么 closed-loop execution 重要？** Figure 8 给出两个典型 failure case：
1. Grasp 偏移 → 抓物体时物体在 gripper 里 shifted，导致 active object 的实际 $\Phi(\mathbf{P}^{ee})$ 与假设不符
2. Target object 在 task 进行中 moved (e.g. collision 后被动移位)

这两种情况下，ReKep 的 point tracking 在 occlusion 下会 fail（point 被遮住就 track 不到，47% failure rate）。OmniManip 用 object-level 6D pose tracking（基于 mesh + RGBD），即使 primitive point 不可见也能从 object pose 反算 canonical-space primitive 位置。

---

## 5. Experimental 分析

### 5.1 主实验 (Table 1)

12 个 task，10 trials each。前 6 个 rigid，后 6 个 articulated：

| Method | Rigid total | Articulated total |
|---|---|---|
| VoxPoser | 15.0% | 16.7% |
| CoPa | 30.0% | 26.7% |
| ReKep | 45.0% | — (ReKep 不支持 articulated) |
| OmniManip (open-loop) | 51.7% | 45.0% |
| OmniManip (closed-loop) | **68.3%** | **61.7%** |

几个 takeaway：
- Closed-loop planning 普遍带来 15-20% 提升，这是 RRC 的核心证据
- ReKep 在 "Pick up the cup on the dish" 上反而强 (9/10)，说明在某些 keypoint 容易 surface-sampling 的简单 pick-and-place 上 ReKep 也够用
- OmniManip 在 "Fit the lid onto the teapot" 上 5/10，相对低，因为 lid 这种 thin object 的 mesh 重建和 pose 估计 noise 大
- Articulated object (drawer, jar, laptop) 是 ReKep 完全不支持的领域，因为 relational keypoint 在 articulated DOF 上没定义清楚

### 5.2 Viewpoint consistency (Table 2)

在 "Recycle the battery" 上从 0° 到 90° 测试 viewpoint：

- ReKep: 0/10, 1/10, 3/10, 5/10, 7/10 —— 强 viewpoint 依赖
- OmniManip: 7/10, 8/10, 8/10, 7/10, 7/10 —— 几乎 invariant

这是 canonical space representation 的最强证据。ReKep 在 frontal view 下 keypoint 飘到空中（因为 battery 的 contact 点在 3D 内部，frontal view 看不到），OmniManip 通过 multi-view reasoning + canonical mapping 直接锁定 battery 的 -z axis（插入方向）。

### 5.3 Sampling efficiency (Table 3)

Uniform SO(3) vs canonical principal axes:
- Battery: 50% → 80%, iter 1.8 → 1.7
- Tea: 30% → 70%, iter 3.4 → 1.8

iter 差不多，但 success rate 翻倍 —— 说明 candidate pool 本身 task-relevance 提升了，不是 search 更努力而是 search 空间更对。

### 5.4 Behavior cloning from OmniManip demos (Table 4)

OmniManip 还能用来自动 generate demo data，然后 train diffusion policy：
- Pick up the cup on the dish: 95.24%
- Recycle the battery: 91.30%
- Insert the pen in holder: 86.36%

每个 task 150 trajectories，零人工标注。这其实是 OmniManip 的"killer app" —— 它作为 zero-shot 的 demonstration generator 来 bootstrap imitation learning policies，绕开 manual teleop 的 data bottleneck。这个角度跟 recent work like "Bootstrapping" 思路一致，参考 https://bootstrap-robotic-manipulation.github.io/

---

## 6. Limitations 和我的 intuition

Paper 自己列了三个 limitation：
1. **Deformable object**: pose representation 不适用
2. **Mesh quality 依赖 3D AIGC**: 单视图重建对 thin / transparent / texture-less object 还是 noisy
3. **VLM call 成本**: 多次 VLM inference 慢，即使 parallel

我觉得还有几个 implicit limitation：

- **Canonical space 的 "functional alignment" 假设**：Omni6DPose 必须把 object pose 估计到 functional canonical frame，而不是 arbitrary canonical frame。这对"非典型 functional object"（比如一个被改作花瓶的茶壶）会有问题。Functional frame 的定义本身就有歧义。
- **6D pose tracker 的 robustness**：occlusion / specular / transparent object 上 pose tracker 会 fail，这会直接 break closed-loop execution。可以结合 recent tactile-based pose refinement 来 mitigate，参考 https://arxiv.org/abs/2409.17549 (北大同组的 visuo-tactile work)
- **Stage decomposition 靠 VLM 的 planning 能力**：对 long-horizon task (e.g. 泡茶 = 开盖 + 抓壶 + 倒水 + 放回 + 开盖 + 加茶 + 搅拌) VLM 会 hallucinate stages，paper 里没充分 stress test 这个。
- **No bimanual**: 整个 formulation 假设 single end-effector，bimanual task (e.g. 折衣服) 需要重新设计 primitive 关系。

---

## 7. 与 related work 的延伸联想

几个让我想到的 connection：

1. **与 "Object-centric RL" 的 connection**：OmniManip 的 object-centric canonical primitive 很像 object-centric RL 里的 "object frame" 思路，比如 seminal work on category-level manipulation (NOCS, 等)。参考 https://arxiv.org/abs/1901.02970 (NOCS)

2. **与 Diffusion Policy / π0 的关系**：OmniManip 是 symbolic / structured representation 的 extreme，而 diffusion policy / VLA 是 implicit representation 的 extreme。这两者其实是 representation spectrum 的两端。一个可能的 future direction：用 OmniManip generate 的 trajectories 作为 VLA 的 demonstration data 来 distill symbolic reasoning 进 implicit policy。Paper 里 Table 4 已经是这个方向的开端。

3. **与 "Affordance" literature**：canonical principal axes = functional axes 这个 idea 和 Gibson's affordance theory 的 "geometry-for-function" 是直接对应的。Reference: Gibson 1979 "The Ecological Approach to Visual Perception"。

4. **与 Geometric Constraint Solving**：Eq. (2)-(5) 的 formulation 本质上是经典的 numerical IK with constraints，和 CuRobo (https://curobo.org/) 的思路一致。OmniManip 的 novelty 在于 constraint 是从 VLM 来的，而不是 hard-coded。

5. **与 "Visual Servoing" 的 connection**：Closed-loop execution via pose tracking 本质是 PBVS (Position-Based Visual Servoing)。Reference: Hutchinson, Hager & Corke 1996 "A Tutorial on Visual Servo Control"。

6. **与 NeuroSymbolic 的 connection**：整个 system 是个 neuro-symbolic hybrid —— VLM 是 neuro，constraint optimization 是 symbolic。这跟 recent "NeuroSymbolic RL" line 一致，参考 https://arxiv.org/abs/2305.01024

7. **与 ReKep 的核心区别再 clarify**：ReKep 的 keypoint 是在 image / world frame 里采样，靠 point tracker (CoTracker) 维护；OmniManip 的 primitive 是在 object canonical space 里采样，靠 6D pose tracker 维护。前者依赖 point visibility，后者依赖 mesh + pose estimation。两者各有 failure mode，但 OmniManip 在 occlusion 下更 robust，在 mesh 缺失时更脆弱。

---

## 8. 整体评价

**Strong point**:
- Canonical space 作为 reasoning substrate 这个 idea clean 且 well-motivated
- Dual closed-loop 真正 mitigate 了 VLM hallucination 和 execution dynamic
- Zero-shot generalization 数据 strong，ablation 全面
- Demo generation application 有实用价值

**Weak point**:
- Pipeline 很重：GroundingDINO + SAM + 3D AIGC + Omni6DPose + VLM + 6D tracker + motion planner，每一步都有 failure mode，复合 reliability 有限
- Latency 高，多次 VLM call + 3D generation 不适合高频 control
- 依赖 mesh quality，对 transparent / thin / articulated part 的 mesh 重建是 known weakness

**My take**: 这个工作其实是个很漂亮的 "structured representation + VLM reasoning + classical planning" 范式 showcase。它的价值不在 SOTA 数字，而在给社区一个 clean 的 blueprint：**你不需要 fine-tune VLM，只要给它一个 structured 3D substrate（canonical space primitives）和 self-check 机制（RRC），它就能 zero-shot 地 solve 精确 manipulation**。

这个 direction 和 fully end-to-end VLA 是互补的 —— 短期看 OmniManip 这类 structured approach 在 data efficiency 和 generalization 上占优，长期看 VLA 可能用 OmniManip 这类 system 做 data engine 来 bootstrap 自己。这个 symbiosis 是我最期待的下一步。

---

## 关键 References

- OmniManip: https://omnimanip.github.io
- ReKep: https://rekep-robot.github.io/
- VoxPoser: https://voxposer.github.io/
- CoPa: https://arxiv.org/abs/2403.08248
- FoundationPose: https://foundationpose.github.io/
- Omni6DPose: https://jzyao.github.io/Omni6DPose/
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- SAM: https://github.com/facebookresearch/segment-anything
- SCAFFOLD: https://arxiv.org/abs/2402.12058
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- UMI: https://universal-mani.github.io/
- TripoSR: https://arxiv.org/abs/2403.02151
- OpenVLA: https://openvla.github.io/
- AnyGrasp: https://github.com/graspnet/anygrasp_sdk
- MOKA: https://moka-manipulation.github.io/
