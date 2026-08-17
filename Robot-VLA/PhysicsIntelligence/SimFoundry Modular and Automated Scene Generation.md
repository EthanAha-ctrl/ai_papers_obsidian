---
source_pdf: SimFoundry Modular and Automated Scene Generation.pdf
paper_sha256: 22afd934befb3329670bc10454eb551aef727759665cebe73dbab64cc99522a2
processed_at: '2026-08-12T06:29:33-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SimFoundry 用人话讲

好，我换一种讲法，用大白话把 intuition 讲清楚，技术细节保留但用更接地气的方式表达。

---

## 这篇 paper 到底干了啥

想象你是个机器人研究员，你有一台 Franka 机械臂，桌上摆着碗、杯子、marker、垃圾桶。你想训一个 policy 让机器人收拾桌子。问题来了：

- **收集 real demo 很贵**：teleop 一小时可能就 20 条 trajectory，训一个好 policy 动不动要几千条
- **在 real 上 eval 很慢**：每跑一次 rollout 要 reset scene、摆物体、记录结果，跑 25 trials 一下午就没了
- **想测 5 个不同 policy 哪个好？** 那得 5 个下午，还要保证每次 reset 完全一致

SimFoundry 说：你拍一段手机视频扫一下桌面，我自动给你重建一个 sim scene，物体几何、位置、背景全都有，物理可交互。然后你可以在 sim 里：
1. **eval 任何 policy**，结果跟 real 高度相关（Pearson 0.911）
2. **自动生成大量 synthetic demo** 训 policy，zero-shot deploy 到 real 能用
3. 还能自动变出 object / scene / task 的变体，让 policy 泛化到没见过的东西

一句话：**把 real world 的一个场景，变成 sim 里的一整个 training + eval ecosystem**。

paper link: <https://research.nvidia.com/labs/gear/simfoundry/>

---

## 为什么以前做不到

以前的 real2sim 系统大概分两类，都有硬伤：

**第一类：只做 scene 重建，不管下游**
比如 SAM3D [74]、Scenesmith [63]，给你一个 3D scene 就完事了。但这个 scene 能不能跑机器人？物理对不对？能不能生成训练数据？没人管。

**第二类：做 sim-to-real 或 eval，但 scene 是手搭的**
比如 PolaRiS [32] 做 real-to-sim eval，但你需要手动在他们的 composer 里摆物体、调位置。而且他们 reconstruct 的 scene 跟 real 差太远，policy 在上面 eval 直接 collapse，必须 shallow finetuning policy 才能 correlate。

SimFoundry 的差异点：**一个 video 进去，sim-ready scene + cousins + eval + training data 全自动出来**。Table 1 跟 14 个 prior work 对比，SimFoundry 是唯一全打勾的。

参考：
- PolaRiS: <https://arxiv.org/abs/2512.16881>
- SAM3D: <https://arxiv.org/abs/2511.16624>
- Scenesmith: arxiv 搜 Scenesmith Tedrake

---

## Pipeline 长啥样——用做菜来类比

把 pipeline 想象成一道菜的流水线：**备菜 → 烹饪 → 加料**。

### Stage 1: Extraction（备菜）

你拍一段视频，SimFoundry 做三件事：

**第一件：选一帧代表性画面 + 估深度**
用 DepthAnything3 [50] 从单张 RGB 估出 depth map，然后用 camera intrinsics $K$ 把 RGB-D backproject 成 scene point cloud $\mathbf{P}_s$。这个 point cloud 就是整个 scene 的几何骨架。

**第二件：对齐 ground plane**
用 SAM3 [10] 分出地面，把 scene 的世界坐标系跟 simulator 对齐——z 轴朝上，ground plane 在 z=0。

**第三件：iterative 抠物体**（这是我觉得最 clever 的 trick）
用 VLM (Gemini-Pro-3) 列出场景里有哪些物体，然后逐个：
1. SAM3 分割出 object $o_i$ 的 mask $m_i$
2. 抠出 RGB crop + depth crop
3. **用 inpainting 把这个物体从画面里擦掉**
4. 在擦干净的 residual 画面上重新检测下一个物体

为什么要 iterate？因为物体互相遮挡。如果一次性分割所有物体，被挡住的部分 seg 不出来。擦掉前面的物体，后面的就露出来了。这就像剥洋葱——一层一层来。

参考：
- DepthAnything3: <https://arxiv.org/abs/2511.10647>
- SAM3: Meta SAM 系列
- PriorDepthAnything (inpainting): <https://arxiv.org/abs/2505.10565>

### Stage 2: Generation（烹饪）

对每个抠出来的 object RGB crop：

**生成 mesh**：用 Hunyuan2.1 [29] 或 TRELLIS.2 [97] 从单图生成 3D mesh $\mathcal{M}_i$。这两个是当前最强的 image-to-3D 模型。

**估 pose 和 scale**：把 mesh 放回 scene point cloud 里，用 FoundationPose [89] 做 6-DoF pose refine。这一步的目标是找到 $\mathbf{p}_i \in SE(3)$ 和 $\mathbf{s}_i \in \mathbb{R}^+$ 让 mesh 跟 point cloud 对齐。

**处理 articulated object**：如果物体是柜子、抽屉、垃圾桶这种有 joint 的：
1. VLM 看多角度渲染图，列出有哪些可动 part + joint type（prismatic / revolute）
2. P3-SAM [54] 对 mesh 做 face-level segmentation
3. VLM 把 segments 归到 part 上，merge 成 per-part mesh
4. **Actor-critic loop**：VLM 写代码调 URDF API 放 joint → simulator 跑出 video → 另一个 VLM critic 看视频打分 → 不满意就重写
5. VLM 还估 physical parameters（mass, friction, damping）

**生成 collision mesh**：用 CoACD [87] 给 mesh 算一个近似凸分解的 collision geometry。这一步必须做，因为原始 mesh 太复杂，物理引擎跑不动。

**物理稳定**：所有物体丢进 PyBullet，可能有轻微 penetration，step physics 直到物体 settle。每 step 后强制 velocity = 0 避免 depenetration 爆炸。最终的 settled pose 缓存下来。

**导出**：scene 导出成 IsaacLab [59] 或 OmniGibson 格式，sim-ready。

参考：
- Hunyuan3D: <https://arxiv.org/abs/2506.15442>
- FoundationPose: <https://github.com/NVlabs/FoundationPose>
- P3-SAM: <https://arxiv.org/abs/2509.06784>
- CoACD: <https://github.com/NVlabs/coacd>
- Articulate-Anything: <https://arxiv.org/abs/2410.13882>
- Isaac Lab: <https://github.com/isaac-sim/IsaacLab>

### Stage 3: Augmentation（加料——这是最核心的创新）

一个 digital twin 只是一个点。SimFoundry 把这个点撑成一个分布，方法是用三种 cousin：

**Object Cousins：换个形状差不多的物体**
比如原场景里有个马克杯，SimFoundry 生成 9 个变体——高一点、矮一点、把手形状不同、颜色不同——但都能抓、都能用。流程：
1. 把物体分解成 functional component（handle / body / lid / base）
2. 对每个 component 沿三个 dimension vary：geometry（形状）、topology（结构）、visual（纹理材质）
3. Image gen model 只改指定 component，其他保持
4. VLM check 合不合理，不合理的丢掉

关键 insight：**per-component editing** 才能保 affordance。整体改容易丢失物体 identity。

**Scene Cousins：换个 layout**
比如原来 marker 在 plate 右边，变体里 marker 可以在 plate 上面、里面、前面。用 semantic spatial predicate `[LeftOf, RightOf, InFrontOf, Behind, OnTopOf, Inside]` 控制。还可以加 distractor 物体。

关键：这**不是 pose perturbation**——它改变 semantic relation，所以是 task-meaningful 的新 layout。

**Task Cousins：换个任务**
这是最 idea 的。给重建好的 scene，VLM 提出十几个 feasible task。比如场景里有 bowl、marker、eraser、baseball、organizer，VLM 可能提：
- "Place Baseball in Bowl"
- "Place Black Eraser on Organizer"
- "Place Red Marker in Cup"
- "Place Orange Cup on Organizer"

每个 task 都 share 一些 object 和 predicate，所以训练时形成 intra-task transfer——学一个 task 帮助学另一个。

参考：
- Digital Cousins (前作): <https://arxiv.org/abs/2410.07408>
- MimicGen (data generation): <https://arxiv.org/abs/2310.17596>
- BEHAVIOR dataset: <https://behavior.stanford.edu/>

---

## Background 怎么搞——两条路

前景物体用 mesh，背景（桌面、墙面、地板）需要 photorealistic。SimFoundry 给两条 pipeline：

**Automatic（单视频、全自动）**
用同一段视频，SAM2 [68] 把前景 mask propagate 到所有帧，VOID [60] 做 two-pass video inpainting 把前景擦掉，然后在擦干净的视频上训 3DGS [38]。

这里有个关键 design choice：3DGS 训练时除了 photometric loss，还加 depth loss（用 DepthAnything3 的 depth supervision），而且**每个 camera 学一个小的 $SO(3) \times \mathbb{R}^3$ pose perturbation**。paper 说这是"single most impactful design choice for splat sharpness"——没有它 splat 一定 blurry。

为什么？因为 DepthAnything3 的 pose 估出来有 sub-pixel noise，inpainting 也会引入小 offset。如果不让每个 camera 微调自己的 pose，这些误差累积起来 splat 就糊了。

**Manual（拍第二段清场视频）**
用户把前景物体搬走，再拍一遍，直接在干净视频上训 3DGS。然后用 interactive editor 手动对齐到前景 scene。

**反直觉的结果**：Table L.6 显示 automatic 比 manual 好。PSNR 15.29 vs 12.91，NCC 0.749 vs 0.549。

为什么？因为 automatic 的 background-to-world transform 是**解析推导**的——3DGS 用的 camera pose 跟前景 Extraction 共享同一个 anchor frame，registration by construction。Manual 是人眼手调 6-DoF，小旋转误差随 depth 放大成大像素 misalignment。

但 manual 在 textureless 平面上更清晰（无 inpainting artifact），各有 trade-off。

参考：
- 3DGS: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- NerfStudio: <https://github.com/nerfstudio-project/nerfstudio>
- SAM2: <https://github.com/facebookresearch/sam2>
- VOID: <https://arxiv.org/abs/2604.02296>
- COLMAP: <https://github.com/colmap/colmap>
- Umeyama alignment: <https://doi.org/10.1109/34.88573>

---

## 重建质量怎么样

Table L.2，12 个 scene 分 Easy / Medium / Hard（按 occlusion 程度）：

| | SAM3D zero-shot | SimFoundry zero-shot | SimFoundry + 3min/object tuning |
|---|---|---|---|
| Easy F1 | 0.71 | 0.92 | 0.99 |
| Medium F1 | 0.66 | 0.87 | 0.97 |
| Hard F1 | 0.68 | 0.81 | 0.93 |
| Easy Chamfer (m) | 0.0081 | 0.0042 | 0.0026 |

直觉：SimFoundry zero-shot 就比 SAM3D 好 0.2 F1，3 分钟人工调一下就到 0.93–0.99。平均每个 object 重建花 5 分钟，调 3 分钟，总共一个 scene 10–15 分钟搞定。

Ground truth 怎么来的也 clever（Appendix L.1.1）：把物体从远到近一个一个放到桌上，每个物体在 unoccluded view 下用 FoundationPose + 已知 CAD mesh 估 pose 作为 GT。最后留 fully-occluded scene 给 reconstruction method 测。

---

## Real-to-Sim Eval：sim 能预测 real 吗

这是 SimFoundry 的核心 claim 之一：在 sim 里 eval policy，结果能预测 real world 表现。

### 两个 metric

**Pearson correlation $\rho$**：看 sim 和 real 分数的线性相关性，越接近 1 越好。

$$\rho(\mathbf{x}, \mathbf{y}) = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}}$$

- $\mathbf{x} = (x_1, ..., x_N)$：N 个 policy 的 real world success rate
- $\mathbf{y} = (y_1, ..., y_N)$：同 N 个 policy 的 sim success rate
- $\bar{x}, \bar{y}$：均值
- $\rho = 1$：完美线性正相关

**MMRV (Mean Maximum Rank Violation)**：看 sim 是否 preserve real 的 ranking。

$$\mathrm{MMRV}(\mathbf{x}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} \max_{j} \left[ |x_i - x_j| \cdot \mathbb{1}\left(\mathbb{1}[y_i < y_j] \neq \mathbb{1}[x_i < x_j]\right) \right]$$

- 外层：对每个 policy $i$ 取平均
- $\max_j$：遍历所有其他 policy $j$，找最大 violation
- $|x_i - x_j|$：real 分数差，作为 penalty 权重
- $\mathbb{1}[y_i < y_j] \neq \mathbb{1}[x_i < x_j]$：如果 sim 排序跟 real 排序不一致，触发 penalty
- MMRV → 0 表示 sim 完美 preserve real ranking

### 实验结果

7 个 task × 5 个 policy（π0 [5], π0.5 [31], GR00T N1.6 [61], GR00T N1.7, DreamZero [100]）：

| Task | SimFoundry Pearson r | PolaRiS Pearson r |
|---|---|---|
| Stack Dishware | 0.883 | 0.500 |
| Store Marker | 0.915 | 0.822 |
| Throw Away Trash | 0.910 | 0.253 |
| Serve Fruits | 0.960 | 0.480 |
| Cup in Bowl | 0.907 | -0.396 |
| Marker in Cup | 0.995 | 0.512 |
| Clear Table | 0.810 | -0.037 |
| **Mean** | **0.911** | **~0.32** |

PolaRiS 在 Cup in Bowl 和 Clear Table 上甚至**负相关**。这太离谱了——sim eval 反而跟 real 反着来。

为什么 PolaRiS 这么差？paper 解释：PolaRiS reconstruct 的 scene 跟 real 差太远，policy 在上面是 OOD，直接 collapse。PolaRiS 的解法是 shallow finetuning policy 在 PolaRiS sim data 上，才能 correlate。SimFoundry 完全 zero-shot eval，不需要任何 adaptation。

### Sub-task Eval Trick

长 horizon task（比如 Store Marker = open drawer + pick marker + place + close drawer）有个问题：如果第一步 open drawer 失败，后面根本没机会执行，end-to-end success 被 bottleneck 在第一步。

SimFoundry 的 trick：**从 sub-task 已完成的状态开始 eval**。比如初始化时 drawer 已经打开，看 policy 能不能完成剩下的 pick + place + close。

Table G.3 显示 mean Pearson 从 0.902 提到 0.951。

直觉：这相当于 unit test——把 long-horizon task 拆成 atomic 能力分别测，避免 cascading failure 主导评估。这个 trick 我觉得对所有 robot eval 都应该用。

参考：
- π0: <https://www.physicalintelligence.company/blog/pi0>
- π0.5: <https://arxiv.org/abs/2504.16054>
- GR00T N1: <https://arxiv.org/abs/2503.14734>
- DROID: <https://droid-dataset.github.io/>
- DreamZero: <https://arxiv.org/abs/2602.15922>

---

## Sim-to-Real Training：sim 训的 policy 能 deploy 到 real 吗

这是另一个核心 claim。三个 setting：

### Zero-shot Sim-to-Real

只用 SimFoundry 生成的 synthetic data 训 policy，zero-shot deploy 到 real。

Table G.4 摘要：
- **YAM Pot on Stove**：99% real success（+9 object cousins）
- **DROID Stack Dishware**：100% real success
- **YAM Stack Dishware Real Cousin**（held-out 物体）：twin only 21% → +9 cousins 42%

### Object Cousin 的 scaling（Table G.8）

Pot On Stove Sim Cousins：
- Twin only: 17%
- +1 cousin: 27%
- +3 cousins: 35%
- +9 cousins: 93%

从 3 到 9 之间有跳跃。直觉：cousin 数量到某个 threshold 才够 cover 测试分布，过了 threshold policy 才从 overfit specific instance 转到学 affordance。

### Scene Cousin（Table G.5）

```
Store Marker- cousin scene:
  twin only: 0%
  + scene cousin: 16%
```

twin only policy 在 cousin layout 上完全失败。加 scene cousin 才能 transfer。这证明 scene cousin 提供的是**layout generalization**，object cousin 给不了。

### Task Cousin（Table G.6）

```
Throw Away Trash:
  twin only: 8%
  +1 task: 44%
  +7 tasks: 44%
  +13 tasks: 68%
```

加 1 个 task 就从 8% 跳到 44%。这说明 intra-task transfer 很强——trash can task 跟其他 "open + put + close" task 共享 structure。

### Sim + Real Co-training（Table G.7）

```
π0.5 Store Marker-Real:
  sim only: 20%
  real only: 60%
  co-train: 92%

π0.5 Throw Away Trash-Real:
  sim only: 20%
  real only: 48%
  co-train: 96%
```

Co-training 在所有 task 上拿到 best 或 near-best。sim data 给 scale + diversity，real data 给 visual realism grounding，互补。

### Multi-task Generalist（Table 2）

用 SimFoundry 重建一个 cluttered scene，VLM 自动提 13 个 task，每个 task 收 10 条 human demo + MimicGen [58] 扩到 100 条，训一个 multi-task policy。

| | π0.5-DROID (base) | π0.5-FT (sim only) | π0.5-DROID-FT (sim+real) |
|---|---|---|---|
| Sim | 30% | 51% | 61% |
| Real | 28% | 45% | 46% |
| Real held-out | 26% | 29% | 26% |

π0.5-FT 在**没见过的 held-out task** 上达到 29% success。这是 task cousin 的 zero-shot generalization 证据。

参考：
- MimicGen: <https://arxiv.org/abs/2310.17596>
- DexMimicGen (bimanual): <https://arxiv.org/abs/2410.24185>
- JoyLo teleoperation: <https://arxiv.org/abs/2503.05652>

---

## 为什么 SimFoundry correlate 这么好——我的理解

读完之后我自己 build 的 mental model：

**1. Shared camera pose**
Background splat 用 DepthAnything3 在原视频上估的 pose，跟前景 Extraction 共享 anchor frame。sim 渲染的 viewpoint 跟 real 测试 viewpoint 几何一致。PolaRiS 用 COLMAP 重新跑 SfM，camera trajectory 不一致，policy 看到的视角 OOD。

**2. Per-camera pose optimizer in 3DGS**
吸收 DepthAnything3 sub-pixel noise + inpainting offset。paper 明确说这是 splat sharpness 的关键。没有它 splat 一定 blurry，policy 看到的视觉就 OOD。

**3. Iterative inpainting decomposition**
occluded object 在 residual 中重新 detect，比一次 seg 全部更鲁棒。reconstruction 质量直接上去了。

**4. Cousins 撑分布**
一个 twin 是一个点，policy 在这个点上 overfit。三种 cousin 把点撑成 manifold，real test 样本就 likely in-distribution。

---

## 局限性

Paper 自己承认：

1. **依赖 VLM 稳定性**：Gemini 偶尔输出 inconsistent inpainting
2. **Mono depth scale 误差**：单目深度估的 scale 不一定准
3. **Articulated mesh seg 难**：image-to-mesh 生成的 mesh 内部结构复杂
4. **只支持 tabletop**：物理 stability 假设 single flat surface
5. **3DGS 近场 clipping**：end-effector 离 splat 太近 render 出问题，robotics 实验用 mesh background 替代

我自己再加几个：

6. **Articulated joint range 估计无强保证**：actor-critic loop 可能不收敛。可以用真实运动视频作 supervision
7. **Cousin quality 靠 VLM judge**：可能系统性偏向某些 variation。可以引入 affordance metric 反向 check
8. **Pipeline 是 one-shot**：real eval 失败的 case 没有自动反馈回 reconstruction。可以做 closed loop real2sim2real

---

## 跟其他方向的联系

**World Model**：SimFoundry 的 reconstruct + simulate + render 等价于显式 world model——state 是 per-object pose + joint state，transition 是 PyBullet，rendering 是 3DGS/mesh。跟 DreamZero、Genie Sim 3.0 [101] 这种 implicit world model 互补。显式 advantage：可干预、可 reset、可 spawn cousin。

**VLA Scaling**：π0 / π0.5 / GR00T 都是 VLA。VLA 最大瓶颈是 data scale + diversity。SimFoundry 给 VLA 提供可控 diverse training data。如果跑 10000 个 real scenes × 100 cousins = 1M scenes，能 push VLA scaling curve。

**Scaling Law for Cousins**：Table G.8 显示 cousin 数量 1→3→9 不是严格 monotonic，Pot On Stove 在 +3→+9 之间跳 35→93。暗示有 "effective diversity threshold"。这个 threshold 跟 task/object 复杂度有关，值得系统化研究。

**Real2Sim2Real 范式**：SimFoundry + RialTo [79] + X-Sim [19] + GSWorld [34] + Re3Sim [27] + RobotArena∞ [33] + SAGE [96] + Holoscene [94] 一起标志这个范式 mature。未来预期：大规模 real2sim dataset、standardized benchmark、cross-embodiment 重建、end-to-end real2sim foundation model。

参考：
- RialTo: <https://arxiv.org/abs/2403.03949>
- X-Sim: <https://arxiv.org/abs/2505.07096>
- GSWorld: <https://arxiv.org/abs/2510.20813>
- Re3Sim: <https://arxiv.org/abs/2502.08645>
- RobotArena∞: <https://arxiv.org/abs/2510.23571>
- SAGE: arxiv 搜 SAGE 3D scene generation
- Holoscene: <https://arxiv.org/abs/2510.20813>
- Genie Sim 3.0: <https://arxiv.org/abs/2601.02078>

---

## 一句话总结

SimFoundry 把 real2sim 从"重建场景"升级成"构建可 scale 的 sim-based robot learning ecosystem"——modular pipeline 让 foundation model 可热插拔，三种 cousin 把单点撑成分布，faithful reconstruction 让 sim eval 跟 real 高度相关。这是 foundation model era robot learning 的正确姿势：不赌单一 model，用 system design drive performance。

paper: <https://research.nvidia.com/labs/gear/simfoundry/>
GEAR Lab: <https://research.nvidia.com/labs/gear/>

---

# SimFoundry 深度讲解

非常感谢 Andrej 把这篇 paper 丢给我。读完之后直觉上这是一个非常"Karpathy 风格"会喜欢的系统工作——把一堆 foundation model 像乐高一样拼起来，端到端打通 real→sim→real，而且每个模块可以热插拔。我下面尽量把直觉、公式、架构、实验数据全部摊开来讲。

---

## 1. 一句话 Intuition

SimFoundry 把"一段 real-world RGB video"喂进去，吐出一个**物理可交互、sim-ready 的 digital twin + 一组 affordance-preserving 的 cousins**。这个 twin 既能用来**eval 已有 policy（real→sim eval）**，也能用来**自动生成 synthetic demonstrations 训练新 policy（sim→real training）**。整个 pipeline 是 modular 的，每一个 stage 的 foundation model 都可以替换。

核心 paper link: <https://research.nvidia.com/labs/gear/simfoundry/>
作者团队主页面：NVIDIA GEAR Lab，跟 MimicGen [58]、Digital Cousins [17] 是同一拨人。
代码/数据通常在： <https://github.com/NVlabs> 下发布（具体仓库可在上述页面找到）。

---

## 2. 为什么这个工作有意义（与 prior work 的差异）

Table 1 把 SimFoundry 跟 14 个 prior real-to-sim 系统做了对比。这里我把它压缩成"四象限"的直觉：

| 维度 | 仅做 reconstruction | 做 reconstruction + 下游用 |
|---|---|---|
| **静态 digital twin** | SAM3D [74]、Scenesmith [63]、Tabletopgen [85]、SAGE [96] | RialTo [79]、X-Sim [19]、RoboGen [83]、MimicGen [58] |
| **可扩展 cousins** | (空) | **SimFoundry（本文）**、Digital Cousins [17]（仅 object cousins）|

SimFoundry 唯一同时具备：
- ✓ Sim-to-real training
- ✓ Real-to-sim policy eval（zero-shot）
- ✓ Automatic scene construction
- ✓ Articulated objects
- ✓ Multi-embodiment（DROID + YAM bimanual）
- ✓ Asset generation
- ✓ Background reconstruction
- ✓ Object cousins + Scene cousins + Task cousins

这个"feature-complete"是关键。PolaRiS [32] 只做 eval，不做 sim-to-real training，而且需要 shallow finetuning 才能 correlate。SimFoundry 端到端全打通。

参考链接：
- PolaRiS: <https://arxiv.org/abs/2512.16881>
- Digital Cousins: <https://arxiv.org/abs/2410.07408>
- MimicGen: <https://research.nvidia.com/labs/gear/mimicgen/> 或 <https://arxiv.org/abs/2310.17596>

---

## 3. Pipeline 三阶段架构详解

整个 pipeline 在 Figure 2 画出来，分成 Extraction、Generation、Augmentation 三层。我把它拆成数据流和模块两个视角。

### 3.1 数据流视角

```
Raw RGB video V
   │
   ▼
[Representative frame I_s + Depth D_s]  ← V_im2depth (DepthAnything3 / FoundationStereo)
   │
   ▼
[Scene point cloud P_s = D_s backproject via K]   ← camera intrinsics K
   │
   ▼
[Ground plane fit, world frame alignment]   ← V_seg^image (SAM3 [10])
   │
   ▼ iterate:
[Detect object o_i  →  mask m_i  →  RGB-D crop (p_i^rgb, p_i^depth)]
   │ (each iteration inpaints out the extracted object)
   ▼
Per-object {RGB crop, mask, depth, point cloud}
```

这里有一个 trick 我觉得非常聪明：**iterative decomposition + inpainting**。每次抠一个 object 之后用 V_inpaint^depth (PriorDepthAnything [84]) 和 image inpainting 把它从 RGB-D observation 中"擦掉"，下一个 object 就从 residual 中重新检测。这样即使被前面的 object 遮住一部分，只要 inpainting 还原得到位，后面的 object 也能稳定抠出来。

参考：
- DepthAnything3: <https://arxiv.org/abs/2511.10647>
- FoundationStereo: <https://research.nvidia.com/labs/oundationpose>
- SAM3: <https://arxiv.org/abs/...> (Meta SAM 系列最新)
- PriorDepthAnything: <https://arxiv.org/abs/2505.10565>

### 3.2 Generation Stage

```
RGB crop p_i^rgb
   │
   ▼
[V_image upsample]   ← Gemini-Pro-3-Image-Preview
   │
   ▼
[V_mesh: 2D → 3D mesh M_i]   ← Hunyuan2.1 [29] / TRELLIS.2 [97]
   │
   ▼
[6-DoF pose p_i, scale s_i alignment]
   │  (point cloud + mask + FoundationPose [89] refine)
   ▼
if articulated:
   ├── [V_articulation: list parts + joint types]
   ├── [V_seg^mesh (P3-SAM [54]): segment mesh faces]
   ├── [VLM assign segments → parts, merge per part]
   └── [Articulate-Anything actor-critic: predict joint axis, range, mass, friction]
   │
   ▼
[CoACD [87] → collision mesh]
   │
   ▼
[Physical properties from V_scene]
   │
   ▼
[Compose in PyBullet, depenetration, settle]
   │
   ▼
[Export to IsaacLab / OmniGibson]
```

Articulated object 这块是继承了 Articulate-Anything [43] 的 actor-critic 框架。直觉是：VLM 写代码调用 URDF API 放 joint，simulator 跑出 video，另一个 critic VLM 看视频打分，分数低于阈值就让 actor 重写。这个 loop 很像 LLM 的 refiner pattern。

参考：
- Hunyuan3D 2.1: <https://arxiv.org/abs/2506.15442>
- TRELLIS: <https://research.nvidia.com/labs/oundationpose/trellis>
- FoundationPose: <https://github.com/NVlabs/FoundationPose>
- Articulate-Anything: <https://arxiv.org/abs/2410.13882>
- P3-SAM: <https://arxiv.org/abs/2509.06784>
- CoACD: <https://github.com/NVlabs/coacd>
- PyBullet: <https://pybullet.org>
- Isaac Lab: <https://github.com/isaac-sim/IsaacLab>

### 3.3 Depenetration 的物理直觉

foundation model 估出来的 pose 经常会让两个 mesh 略微相交。SimFoundry 用 CoACD 给每个 mesh 算 collision mesh，丢进 PyBullet，每个 step 后强制 velocity = 0（避免 depenetration 爆炸），step 直到 settle。最终的 pose 缓存下来，作为后续 sim-ready scene 的初始 state。

为什么 force velocity = 0：当两个 mesh 有 intersection，物理引擎会产生一个很大的法向接触力，simulation step 一下物体就会被弹飞。强制 zero velocity 让物体只靠位置约束"挤出去"，每步推进一点点，直到不再相交。这个 trick 在 NVIDIA ManipualtorThesis 系工作中很常见。

### 3.4 Background Reconstruction 的两条路

这是我个人觉得 SimFoundry 最有意思的设计选择。它给了两条 pipeline：

**A. Automatic pipeline（单视频、无人干预）**
```
Same raw video V
   │
   ▼
[V_scene enumerate foreground categories on keyframe]
   │
   ▼
[V_seg^image: pixel-accurate masks]
   │
   ▼
[V_seg^video (SAM2 [68]): propagate masks across all frames]
   │
   ▼
[V_inpaint^video (VOID [60]): two-pass chunked inpainting]
   │  (pass 1 fill, pass 2 re-inpaint residual hallucinations)
   ▼
[DepthAnything3 on original frames → sharp camera poses]
[DepthAnything3 on inpainted frames → depth consistent with RGB]
   │
   ▼
[Umeyama [80] fit to align chunk trajectories]
[Backproject inpainted depth → seed PLY]
   │
   ▼
[3DGS training in NerfStudio [72]]:
   Loss = L_photo + λ_depth * L_depth(masked by confidence)
   + per-camera SO(3) × R³ pose optimizer (KEY design choice!)
   │
   ▼
[Compose cam2world anchor from DepthAnything3 + Extraction → M_{src→tgt}]
   │
   ▼
[Registered splat in simulator world frame]
```

**B. Manual pipeline（需要第二段清场视频）**
```
User re-films scene without foreground
   │
   ▼
[COLMAP [71] SfM → camera intrinsics + per-frame extrinsics]
   │
   ▼
[splatfacto-big 3DGS in NerfStudio, photometric loss only]
   │
   ▼
[Interactive editor: human SE(3) + isotropic scale aligns splat to foreground]
```

Table L.6 给了量化对比（5 个 dorm scenes 平均）：

| Variant | PSNR↑ | SSIM↑ | MAE↓ | RMSE↓ | NCC↑ | ΔE↓ | EdgeMAE↓ |
|---|---|---|---|---|---|---|---|
| Manual | 12.91 | 0.497 | 0.1712 | 0.2294 | 0.549 | 20.13 | 0.0283 |
| **Automatic** | **15.29** | **0.605** | **0.1275** | **0.1758** | **0.749** | **15.23** | **0.0248** |

直觉上反常识——**automatic 反而更好**。原因是 automatic 的 background-to-world transform 是解析推导出来的（共享 camera pose），registration by construction；manual 是人眼手调 6-DoF + scale，小旋转误差会随 depth 放大成大像素 misalignment。NCC 这个 metric 把 global brightness / contrast 因子化掉之后，最能反映几何 alignment，automatic 在 NCC 上 0.749 vs 0.549 拉开差距。

但 manual 在 textureless 平面和 silhouette 上更清晰（avoid inpainting artifacts），各有 trade-off。Automatic 的代价是 90 分钟 GPU 时间用于 two-pass video inpainting。

参考：
- 3DGS: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- NerfStudio: <https://github.com/nerfstudio-project/nerfstudio>
- SAM2: <https://github.com/facebookresearch/sam2>
- VOID: <https://arxiv.org/abs/2604.02296>
- COLMAP: <https://github.com/colmap/colmap>
- Umeyama 1991: <https://doi.org/10.1109/34.88573>

---

## 4. Digital Cousins 三种 augmentation

这是 SimFoundry 最 core 的 contribution。一个 digital twin 只能 cover 一个分布点，cousins 把它撑成一片分布。

### 4.1 Object Cousins

Affordance-preserving 的物体变体。流程：
1. **Canonicalize** input object image + retrieve original scene image
2. **Functional decomposition** via VLM：handle / lid / body / base...
3. **Per-component variation proposal**：3 个 dimension（geometry / topology / visual appearance）
4. **Scene-aware image synthesis**：image gen model 改指定 component，其他保持不变
5. **Plausibility check**：VLM 过一遍，filter 掉不合理 / scene-inconsistent 的

Figure F.1 给了 prompt template。这里的关键 insight 是 **per-component localized editing**——直接对整张图做 variation 容易失去 identity，分解到 functional component 再 vary 才能保 affordance。

### 4.2 Scene Cousins

改 layout，保语义。流程：
1. 选 anchor object
2. 对每个其他 object，sample 一个或多个 spatial predicate from `[LeftOf, RightOf, InFrontOf, Behind, OnTopOf, Inside]`
3. Instantiate object 根据 predicate
4. 可选加 distractor object from BEHAVIOR [45] dataset，按 mass / volume / density / category filter

注意：这里**不是简单 pose perturbation**——它改变 semantic relation（从 RightOf 变 OnTopOf），所以产生的是 task-meaningful 的新 layout。

### 4.3 Task Cousins

这在我看来是三个 cousins 里最有 idea 的一个。流程：
1. 抓 scene 2D image + interactable object list
2. 注入 robot constraints（gripper length, single/bimanual）
3. VLM 作为 "robotics expert" 提 N 个 task，要求每个 task 都要 induce **meaningful state change**
4. 输出 structured task definition + goal predicates (OnTop / Inside / Under...)
5. 自动 compile 成 sim-compatible goal spec

Figure F.2 给了 prompt template。Figure F.3 + 后续 list 给了 13 个生成 task 的例子：
- "Place Baseball in Bowl"
- "Place Black Eraser on Organizer"
- "Place Orange Marker on Organizer"
- "Place Red Bottle in Bowl"
- ...

这些 task 都 share 一些 object、predicate、intermediate behavior，所以训练时形成 intra-task transfer。

参考：
- BEHAVIOR: <https://behavior.stanford.edu/>
- Digital Cousins: <https://arxiv.org/abs/2410.07408>

---

## 5. Real-to-Sim Policy Evaluation 的 metrics

这部分是 SimFoundry 跟 PolaRiS 对标的硬指标。两个 metric 都来自 [32, 47]：

### 5.1 Pearson Correlation

给定 N 个 policy $\Pi = \{\pi_1, ..., \pi_N\}$，每个 policy 在 real 拿到分数 $x_i \in [0, 1]$，在 sim 拿到 $y_i \in [0, 1]$。

$$\rho(\mathbf{x}, \mathbf{y}) = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}}$$

变量解释：
- $\mathbf{x} = (x_1, ..., x_N)$：real-world score 向量
- $\mathbf{y} = (y_1, ..., y_N)$：sim score 向量
- $\bar{x} = \frac{1}{N}\sum_i x_i$：real 分数均值
- $\bar{y} = \frac{1}{N}\sum_i y_i$：sim 分数均值
- $\rho \in [-1, 1]$，越接近 1 说明 sim 和 real 的**线性趋势**越一致

Pearson 的局限：它只看 score-level agreement，不看 ranking。

### 5.2 MMRV (Mean Maximum Rank Violation)

为了补 ranking 这个维度，定义：

$$\mathrm{MMRV}(\mathbf{x}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} \max_{j \in \{1,...,N\}} \left[ |x_i - x_j| \cdot \mathbb{1}\left( \mathbb{1}[y_i < y_j] \neq \mathbb{1}[x_i < x_j] \right) \right]$$

变量解释：
- 外层 $\frac{1}{N}\sum_{i=1}^{N}$：对每个 policy $i$ 取平均
- $\max_{j}$：找最大的 violation
- $|x_i - x_j|$：real 分数差（penalty 权重）
- $\mathbb{1}[y_i < y_j] \neq \mathbb{1}[x_i < x_j]$：indicator，1 当且仅当 sim 和 real 的 ranking 不一致
- 整体直觉：**对每一次 sim-vs-real ranking inversion，按 real 分数差加权，取最大**

理想情况 MMRV → 0，意味着 sim 完美 preserve real 的 ranking。

### 5.3 实验数据

Table G.1（SimFoundry）vs Table G.2（PolaRiS），我抽几个 task 看：

**Stack Dishware**：π0 real=100, sim=34；π0.5 real=100, sim=64；GR00T N1.6 real=40, sim=0
Pearson r = 0.883, MMRV = 0.000

**Marker in Cup**：π0 real=40, sim=40；π0.5 real=92, sim=88；GR00T N1.6 real=28, sim=28；GR00T N1.7 real=88, sim=88；DreamZero real=88, sim=80
Pearson r = **0.995**, MMRV = 0.008

**Clear Table**（最难，7 个 sub-step）：r = 0.810, MMRV = 0.016

而 PolaRiS 同样 7 个 task 的 Pearson：
- Stack Dishware: 0.500
- Store Marker: 0.822
- Throw Away Trash: 0.253
- Serve Fruits: 0.480
- Cup in Bowl: **-0.396**（负的！）
- Marker in Cup: 0.512
- Clear Table: -0.037

Mean Pearson SimFoundry = **0.911**, PolaRiS = ~0.32。差距超过 0.59。

直觉解释为什么 PolaRiS 这么差：PolaRiS 要求**shallow finetuning** policy 在 PolaRiS sim data 上才能 correlate，因为他们 reconstruct 出来的 scene 视觉/几何上跟 real 差太远，policy 在 OOD 上直接 collapse。SimFoundry 不需要 finetuning，因为 reconstruct 够 faithful。

### 5.4 Sub-task Evaluation 的 trick

这是另一个我觉得很聪明的 trick。对于 long-horizon task（e.g., Store Marker = open drawer + pick marker + place + close drawer），如果直接 evaluate end-to-end success，前一个 sub-task 失败后面根本没机会执行，correlation 会被前 step 主导。

SimFoundry 的做法：**从 sub-task 已完成的状态开始 evaluate**。比如 Store Marker，初始化时 drawer 已经打开，看 policy 能不能完成剩下的 pick + place + close。

Table G.3 显示 mean Pearson 从 0.902 提到 **0.951**。

直觉上这相当于 isolation 测试——把 long-horizon task 拆成 atomic 能力分别打分，避免前 step failure 的 cascading effect。这个 trick 我觉得对所有 long-horizon robot eval 都适用，应该被更广泛采用。

参考：
- PolaRiS: <https://arxiv.org/abs/2512.16881>
- RoboArena: <https://arxiv.org/abs/2510.23571>
- Li et al. "Evaluating real-world robot manipulation policies in simulation": <https://arxiv.org/abs/2405.05941>

---

## 6. Sim-to-Real Training 实验

这部分证明 SimFoundry 不只是 eval tool，还能 train policy。

### 6.1 Zero-shot Sim-to-Real

Table G.4（YAM bimanual + DROID）摘要：
- **YAM Pot on Stove**：twin only Real Twin = 91%，+9 cousins Real Twin = 99%
- **DROID Stack Dishware**：twin only Real Twin = 88%，+9 cousins = 96%
- **YAM Pot on Stove Real Cousin**（held-out 物体）：twin only = 14%，+9 cousins = **64%**（50-point gain）

这个 50-point gain on held-out 是 paper 的 highlight。直觉解释：训练只用 twin object，policy overfit 到这个 specific geometry；加了 9 个 affordance-preserving cousins，policy 学到的是 **affordance 本身**而不是具体形状。

### 6.2 Object Cousin Ablation（Table G.8）

```
Stack Dishware
                  Twin  +1 Cousin  +3 Cousins  +9 Cousins
Sim Twin          83    89         100         92
Sim Cousins       43    44         65          66
Real Twin         39    41         37          43
Real Cousins      21    32         27          42

Pot On Stove
Sim Twin          85    100        93          100
Sim Cousins       17    27         35          93
Real Twin         91    100        94          99
Real Cousins      14    38         16          64
```

注意 Pot On Stove 的 Sim Cousins 从 17 → 93（+76 points），这个 curve 不是 monotonic 但整体趋势明显。说明 cousin 数量到某个 threshold（这里 ~9）才够 cover 测试分布。

### 6.3 Scene Cousins（Table G.5，DROID）

```
                              twin only  + scene cousin
Stack Dishware                80         88
Stack Dishware- cousin        28         64
Store Marker                  20         24
Store Marker- cousin          0          16
Throw Away Trash              8          36
Throw Away Trash- cousin      0          36
```

最 striking：**Store Marker- cousin 从 0% 到 16%**——twin only policy 完全无法 generalize 到 cousin layout，加 scene cousins 才能 transfer。

### 6.4 Task Cousins（Table G.6）

```
                     twin only  +1 task  +7 tasks  +13 tasks
Stack Dishware       80         88       100       100
Store Marker         20         36       48        60
Throw Away Trash     8          44       44        68
```

Throw Away Trash 从 8 → 68（+60 points）是最大 gain。直觉：trash can 这种 task 跟很多"open + put + close" task 共享 structure，加 cousin task 等于在学 task template 而非 specific instance。

### 6.5 Multi-task Generalist Policy（Table 2）

| | π0.5-DROID | π0.5-FT | π0.5-DROID-FT |
|---|---|---|---|
| Sim | 30 | 51 | 61 |
| Sim held-out | 37 | 45 | 33 |
| Real | 28 | 45 | 46 |
| Real held-out | 26 | 29 | 26 |

SimFoundry-finetuned policy 比 base DROID checkpoint 高 31% (sim) / 18% (real)。**π0.5-FT 在 held-out task 上达到 29% success without task-specific demos**——这是 task cousin 的 zero-shot generalization 证据。

### 6.6 Sim + Real Co-training（Table G.7）

```
                          π0-S  π0-R  π0-co  π0.5-S  π0.5-R  π0.5-co
Stack Dishware-Sim        92    34    76     88      64      100
Stack Dishware-Real       96    100   100    96      100     100
Store Marker-Sim          16    4     40     60      20      60
Store Marker-Real         4     48    80     20      60      92
Throw Away Trash-Sim      0     0     36     48      4       60
Throw Away Trash-Real     0     20    76     20      48      96
```

Co-training 在所有 task 上都拿到 best 或 near-best。**Throw Away Trash-Real 从 0 → 76**（π0）和 20 → 96（π0.5）。

直觉：sim data 提供 scale + diversity，real data 提供 visual realism grounding。两者互补。

参考：
- MimicGen: <https://arxiv.org/abs/2310.17596>
- DexMimicGen: <https://arxiv.org/abs/2410.24185>
- π0: <https://www.physicalintelligence.company/blog/pi0>
- π0.5: <https://arxiv.org/abs/2504.16054>
- GR00T N1: <https://arxiv.org/abs/2503.14734>
- DROID: <https://droid-dataset.github.io/>
- JoyLo (Behavior Robot Suite): <https://arxiv.org/abs/2503.05652>

---

## 7. Reconstruction Fidelity（Table L.2）

12 个 scenes，分 Easy / Medium / Hard（按 occlusion 程度），对比 SAM3D [74] zero-shot vs SimFoundry zero-shot vs SimFoundry + 3min/object tuning：

| Difficulty | Metric | SAM3D ZS | SimFoundry ZS | SimFoundry Tuned |
|---|---|---|---|---|
| Easy | Chamfer ↓ | 0.0081 | 0.0042 | **0.0026** |
| Easy | F1 ↑ | 0.71 | 0.92 | **0.99** |
| Easy | Pos Err ↓ | 0.016 | 0.0060 | **0.0041** |
| Medium | F1 ↑ | 0.66 | 0.87 | **0.97** |
| Hard | F1 ↑ | 0.68 | 0.81 | **0.93** |

直觉：
1. SimFoundry zero-shot 就已经超过 SAM3D zero-shot（F1 高 0.2 左右）
2. 3 分钟 human-in-the-loop tuning 把 F1 推到 0.93–0.99
3. Hard scene 的 zero-shot F1 = 0.81，gap 主要来自 occlusion 下 point cloud 不完整导致 pose 估计偏差

这里 Quasi-Ground-Truth 的生成也很 clever（Appendix L.1.1）：把 object 从远到近一个一个加进去，每个 object 在 unoccluded view 下用 FoundationPose + 已知 CAD mesh 估 6-DoF pose 作为 GT，最后留 fully-occluded scene 给 reconstruction method。这样能在 occluded scene 下也能 evaluate per-object pose。

参考：
- SAM3D: <https://arxiv.org/abs/2511.16624>

---

## 8. 为什么 SimFoundry correlate 这么好——我的 Intuition

读完之后我自己 build 的 mental model 是这样的：

**SimFoundry 的 fidelity advantage 来自三个协同设计**：

1. **Shared camera pose**：background splat 用 DepthAnything3 在**原视频**上估的 pose，跟 Extraction 的 ground-plane-fit 共享同一个 anchor frame。这意味着 sim 渲染出来的 viewpoint 跟 real 测试 viewpoint 几何一致。PolaRiS 用 COLMAP 重新跑 SfM，camera trajectory 不一定一致，policy 看到的视角就 OOD。

2. **Per-camera pose optimizer in 3DGS training**：吸收 DepthAnything3 sub-pixel pose noise 和 inpainting-induced 小 offset。这个 trick 是 paper 里明确说"single most impactful design choice for splat sharpness"。没有它 splat 一定 blurry。

3. **Iterative inpainting decomposition**：occluded object 在 residual 中重新 detect，比一次 seg 全部更鲁棒。

**Cousins 的作用是 distribution coverage**：
- Object cousins：从 single instance 撑成 affordance distribution
- Scene cousins：从 single layout 撑成 spatial relation distribution
- Task cousins：从 single task 撑成 task template distribution

三个一起做相当于把 policy training distribution 从一个点撑成一个 manifold，real test 时候的样本就 likely in-distribution 了。

---

## 9. Limitations 和 Future Work

Paper 里 honest 地承认几个：

1. **依赖 VLM 输出稳定性**：Gemini-Pro-3 偶尔产生 inconsistent inpainting（重复 object、degenerate output）。这个 inherent 在 black-box API 调用上。
2. **Mono depth 的 scale/shape 误差**：monocular 输入下 point cloud scale 不一定准。
3. **Articulated mesh segmentation 难**：image-to-mesh 生成的 mesh 内部结构复杂，V_seg^mesh 容易出错。
4. **Tabletop-only**：物理 stability 假设 single flat reference surface，不能 multi-level。
5. **3DGS 近场 clipping**：robot end-effector 离 splat 太近时 render artifact。所以 robotics 实验部分用 mesh background 替代。

我会再加几个 my own observations：

- **Articulated object 的 joint range 估计**：actor-critic loop 虽然好，但收敛性没有强保证。可以用 demos with motion（drawer 开合 video）作为 supervision signal。
- **Cousin quality evaluation**：object cousin 是 VLM judge 的，可能 systemically 偏向某些 variation type。可以引入 affordance metric（grasp success rate, physics stability rate）做 reverse check。
- **Task cousin 的 action-space coverage**：现在 task cousin 都 share same robot + scene，action distribution 不一定 diverse。可以引入 cross-embodiment task cousin。
- **Real2Sim2Real loop closure**：现在 pipeline 是 one-shot。可以做成 closed loop——real eval failure case 自动 re-reconstruct + retrain + redeploy。

参考：
- GaussGym: <https://arxiv.org/abs/2510.15352>
- Re3Sim: <https://arxiv.org/abs/2502.08645>
- GSWorld: <https://arxiv.org/abs/2510.20813>

---

## 10. 一些值得关注的联想

读完 SimFoundry 我有几个 cross-paper 联想：

### 10.1 跟 World Model 的关系
SimFoundry 的 reconstruct + simulate + render 等价于一个**显式 world model**——state 是 per-object 6-DoF pose + articulated joint state，transition 是 PyBullet physics，rendering 是 3DGS / mesh。跟 DreamZero [100]、Genie Sim 3.0 [101] 这类 implicit world model 是 complementary 的。显式 world model 的 advantage：可干预、可 reset 到 arbitrary state（sub-task eval 用的就是这个）、可 spawn cousin。

### 10.2 跟 Diffusion Policy 的关系
Diffusion Policy [15] 在 SimFoundry pipeline 里没有特别强调，但 paper 提到 YAM 用 "flow-matching policy"。flow matching 是 diffusion 的 continuous-time 版本。SimFoundry data 多样性足够支持 diffusion / flow-matching policy 训练，这跟 MimicGen 的 finding 一致——SDG + diffusion policy 在 manipulation 上 scaling 很好。

### 10.3 跟 VLA 的关系
π0 / π0.5 / GR00T N1.6 / GR00T N1.7 都是 VLA architecture。SimFoundry 给 VLA 提供**可控 diverse training data**——目前 VLA 的最大瓶颈是 data scale 和 diversity。如果 SimFoundry pipeline 能跑 10000 个 real scenes × 100 cousins each = 1M scenes，能显著 push VLA scaling curve。

### 10.4 跟 Scaling Laws 的关系
Table G.8 显示 cousin 数量从 1 → 3 → 9 性能不是严格 monotonic。Pot On Stove 的 Sim Cousins 在 +3 → +9 之间从 35 → 93 跳跃。这个 jump 暗示有个 "effective diversity threshold"——cousin 数量过了 threshold 之后 policy 才真正学到 affordance 而不是 instance。这个 threshold 跟 task / object complexity有关，值得系统化研究。

### 10.5 跟 Real2Sim2Real 范式
SimFoundry 加上最近 RialTo [79]、X-Sim [19]、GSWorld [34]、Re3Sim [27]、RobotArena∞ [33]、SAGE [96]、Holoscene [94]、EmbodiedGen [82]、Real2Render2Real [102]、Genie Sim 3.0 [101] 一起，标志 real2sim2real 范式 mature。我预期未来 12-18 个月会有：
- 大规模 real2sim dataset（100k+ scenes）
- Standardized real2sim benchmark（per-object F1, render PSNR, eval Pearson）
- Cross-embodiment real2sim（一个 scene 多 robot 重建）
- Real2sim foundation model（end-to-end video → sim-ready scene，替代 modular pipeline）

---

## 11. 关键公式总结

把 paper 里的公式重新整理一下：

### Pearson Correlation (Eq. 1)
$$\rho(\mathbf{x}, \mathbf{y}) = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}}$$

- $\mathbf{x}, \mathbf{y} \in \mathbb{R}^N$：real / sim 分数向量
- $\bar{x}, \bar{y}$：均值
- $N$：policy 数量
- $\rho \in [-1, 1]$

### MMRV (Eq. 2)
$$\mathrm{MMRV}(\mathbf{x}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} \max_{j} \left[ |x_i - x_j| \cdot \mathbb{1}\left(\mathbb{1}[y_i < y_j] \neq \mathbb{1}[x_i < x_j]\right) \right]$$

- $\mathbb{1}[\cdot]$：indicator function
- $\max_j$：遍历 policy $j$ 找最大 violation
- $|x_i - x_j|$：real 分数差作权重
- 内层 indicator：ranking flip detector

### 3DGS Loss (Appendix E.5.1)
$$\mathcal{L} = \mathcal{L}_{\text{photo}}(I_{\text{render}}, I_{\text{inpaint}}) + \lambda \cdot \mathcal{L}_1(D_{\text{render}}, D_{\text{inpaint}}) \cdot \mathbb{1}[\text{conf} > \tau]$$

- $\mathcal{L}_{\text{photo}}$：photometric loss（rendered RGB vs inpainted RGB）
- $\mathcal{L}_1$：L1 depth loss
- $\lambda$：depth weight
- $\text{conf}$：DepthAnything3 per-pixel confidence
- $\tau$：confidence threshold

### Umeyama Alignment
用于 align 两个 point cloud / camera trajectory：
$$\hat{R}, \hat{t}, \hat{s} = \arg\min_{R, t, s} \sum_i \| y_i - (s R x_i + t) \|^2$$

- $R \in SO(3)$：rotation
- $t \in \mathbb{R}^3$：translation
- $s \in \mathbb{R}^+$：scale
- $x_i, y_i$：source / target point pair
- SimFoundry 用它 align chunk trajectories 和 inpainted-stream 到 original-stream world

### SimFoundry scene 表示
$$\mathcal{S}_{\text{sim}} = \{(\mathcal{M}_i, \mathbf{s}_i, \mathbf{p}_i)\}_{i=1}^{N}$$

- $\mathcal{M}_i$：mesh
- $\mathbf{s}_i$：scale（isotropic）
- $\mathbf{p}_i \in SE(3)$：6-DoF pose

### Policy 表示
$$\pi_\theta : \mathcal{O} \rightarrow \mathcal{A}$$
$$a_t = \pi_\theta(o_t)$$

- $o_t$：observation at timestep $t$（RGB, proprioception）
- $a_t$：action（joint position command + gripper）
- $\theta$：neural network parameter

---

## 12. 总结一句话

SimFoundry 把 **real2sim 从"做 scene 重建"提升到"做可 scale 的 sim-based robot learning ecosystem"**——通过 modular pipeline + 三种 cousin augmentation + faithful reconstruction，让 sim 既成为 reliable evaluator，又成为 scalable data generator。Mean Pearson 0.911 + zero-shot sim-to-real success 是硬证据。

Paper page: <https://research.nvidia.com/labs/gear/simfoundry/>
作者团队其他相关工作：
- MimicGen: <https://research.nvidia.com/labs/gear/mimicgen/>
- Digital Cousins: <https://research.nvidia.com/labs/gear/digital-cousins/>
- GEAR Lab: <https://research.nvidia.com/labs/gear/>
- JoyLo / Behavior Robot Suite: <https://behaviorrobot.github.io/>

希望这个 walkthrough 帮你 build intuition。我个人的 take-away：SimFoundry 的 modular + VLM-as-lego-brick 设计哲学就是 robot learning 在 foundation model era 的正确姿势——不赌单一 model 解决所有问题，而是把每个 capability 抽象成可替换组件，让 system-level design 来 drive performance。
