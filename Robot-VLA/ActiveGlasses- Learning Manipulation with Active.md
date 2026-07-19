---
source_pdf: ActiveGlasses- Learning Manipulation with Active.pdf
paper_sha256: cbb3f28f6681b01fc87e6c24b2f78981ac45a19b6a6fb4d5e419532930861b68
processed_at: '2026-07-18T01:26:55-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ActiveGlasses: 从egocentric人类演示学习带active vision的manipulation

这篇paper来自SJTU的Cewu Lu团队，主推一个核心insight：**让机器人学会"主动看"和"动手"是耦合的两件事**，而不是把vision当成被动的输入。下面我会从动机、系统设计、数学公式、policy架构、实验数据几个层面铺开讲解。

---

## 1. 核心动机：数据收集的三个tension

作者从两个dimension批评现有data collection pipeline：

**Manipulation dimension**: teleoperation（ALOHA [7]、GELLO [8]）强制操作者模仿机器人运动学，physical exhausting；handheld device（UMI [9]、FastUMI [10]）虽然1:1 mapping硬件，但设备重600g+，长时间拿着手腕酸痛，且操作者必须用"夹子kinematics"完成任务，丢失了human natural smoothness。

**Perception dimension**: 现有setup大多用固定第三视角camera（容易occlusion）或wrist-mounted camera（被动跟随end-effector轨迹）。但人类做精细任务时，**头部会主动移动**——伸头绕过遮挡、靠近凑近看细节、独立于手调整视角。wrist camera把这个intent-driven perceptual signal给丢了。

所以ActiveGlasses的目标是：让操作者**用bare hand + 戴smart glasses**完成自然任务，同时记录下6-DoF head trajectory作为"active vision"的ground truth信号，再zero-shot transfer到机器人上。

参考：UMI论文 https://arxiv.org/abs/2402.10329 ；EgoMimic https://arxiv.org/abs/2510.15156 ；EgoZero https://arxiv.org/abs/2505.20290

---

## 2. 硬件与Interface设计

**Perception device构成**：
- **XREAL Air 2 Ultra**：提供6-DoF head pose tracking（SLAM-based）。因为XREAL出于隐私考虑不开放raw camera API，所以作者额外挂了ZED Mini做stereo vision
- **ZED Mini**：stereo RGB-D相机，挂在glasses前方做visual input
- 作者实际测过ZED自带的IMU tracking，发现numerical jump明显（Fig 3b），所以最终用XREAL的6-DoF数据

**Deploy时的硬件**：
- **Manipulation arm**: Flexiv Rizon4 + Robotiq 2F-85 gripper（也测过UR5）
- **Perception arm**: I2RT YAM Robot（6-DoF tabletop arm），末端通过3D-printed adapter挂着同一套glasses+ZED Mini
- Perception arm mounted on wheeled table，每次rollout时base position在小范围内randomize，模拟人上身整体位移

**Unity GUI**: glasses上显示实时camera view（因为ZED FOV小于人眼FOV，UI加了bottom canvas提示）+ 手势检测作为episode start/end signal + audio feedback。

---

## 3. 数据处理Pipeline（关键公式逐一拆解）

给定输入：left video $V_L = \{l_i\}_{i=0}^K$，right video $V_R = \{r_i\}_{i=0}^K$，head trajectory $H = \{h_i\}_{i=0}^K$。

每帧输出：
- depth map $d_i$
- object mask $m_i^{\text{object}}$ + hand mask $m_i^{\text{hand}}$
- object 6-DoF pose in camera frame $T_{\text{cam},i}^{\text{object}}$
- camera→world transform $T_{\text{world}}^{\text{cam},i}$

### 3.1 Depth & Segmentation
- **FoundationStereo** [36] 从stereo pair $(l_i, r_i)$ 估计depth $d_i$，back-project成camera frame下的RGB point cloud $p_i^{\text{cam}}$
- **Grounded-SAM** [37] 分割hand，移除对应points（去除human-specific artifacts）
- **SAM2** [38] 分割manipulated object，得到 $m_i^{\text{object}}$ 用于下游pose estimation

### 3.2 Object Trajectory Estimation
- **FoundationPose** [39] 输入 left image + depth + object mask + object mesh M (geometric prior)，输出6-DoF object pose序列 $\mathcal{O} = \{T_{\text{cam},i}^{\text{object}}\}_{i=0}^K$
- 用mesh做prior是为了robustness（小物体、occlusion时仍能track）

### 3.3 Calibration：建立world frame（公式1-4）

这块是全文最有math美感的部分。作者没用Aruco marker（理由：head viewpoint变化时detection不稳定），而是在tabletop放3个orange spheres，构成一个planar Cartesian坐标系。

**公式1** — 定义world frame的三个axis：
$$\hat{\mathbf{x}} = \frac{\mathbf{b}_2 - \mathbf{b}_1}{\|\mathbf{b}_2 - \mathbf{b}_1\|}, \quad \hat{\mathbf{y}} = \frac{\mathbf{b}_0 - \mathbf{b}_1}{\|\mathbf{b}_0 - \mathbf{b}_1\|}, \quad \hat{\mathbf{z}} = \frac{\hat{\mathbf{x}} \times \hat{\mathbf{y}}}{\|\hat{\mathbf{x}} \times \hat{\mathbf{y}}\|}$$

变量解释：
- $\mathbf{b}_0, \mathbf{b}_1, \mathbf{b}_2 \in \mathbb{R}^3$：三个sphere centers在initial camera frame下的3D坐标（由pixel + depth反投影得到）
- $\hat{\mathbf{x}}$：从 $\mathbf{b}_1$ 指向 $\mathbf{b}_2$ 的单位向量（x轴方向）
- $\hat{\mathbf{y}}$：从 $\mathbf{b}_1$ 指向 $\mathbf{b}_0$ 的单位向量（y轴方向）
- $\hat{\mathbf{z}}$：通过 $\hat{\mathbf{x}} \times \hat{\mathbf{y}}$ 叉乘得到（右手定则，垂直tabletop向上）

直觉：3个非共线点天然定义一个平面，挑其中一点做origin，另外两点定义两个axis，叉乘补全第三轴。比Aruco鲁棒是因为sphere的几何中心可以通过depth直接3D定位，不依赖marker detection在2D pixel space的稳定性。

**公式2** — initial camera→world transform：
$$T_{\text{cam},0}^{\text{world}} = \begin{bmatrix} [\hat{\mathbf{x}}\ \hat{\mathbf{y}}\ \hat{\mathbf{z}}]^\top & \mathbf{b}_1 \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

变量解释：
- 4×4齐次变换矩阵
- 左上3×3 = $[\hat{\mathbf{x}}\ \hat{\mathbf{y}}\ \hat{\mathbf{z}}]^\top$：rotation matrix（注意是转置，因为是camera→world的方向）
- 右上3×1 = $\mathbf{b}_1$：translation（world原点放在 $\mathbf{b}_1$ 处）
- 底部 $[0\ 0\ 0\ 1]$：齐次坐标padding

**公式3** — 通过head pose propagation：
$$T_{\text{cam},i}^{\text{world}} = T_{\text{cam},0}^{\text{world}} \cdot T_{\text{cam},i}^{\text{cam},0}$$

变量解释：
- $T_{\text{cam},i}^{\text{cam},0}$：从head pose tracking得到的，frame $i$ 相对frame 0的相对运动
- 用乘法把initial calibration propagate到整个episode，避免每帧都跑SAM（计算cost高）+ 处理sphere被occluded或离开FOV的情况

**公式4** — point cloud到world frame：
$$\mathbf{p}_i^{\text{world}} = (T_{\text{cam},i}^{\text{world}})^{-1} \mathbf{p}_i^{\text{cam}}$$

变量解释：
- $\mathbf{p}_i^{\text{cam}}$：原始camera frame下的point
- $\mathbf{p}_i^{\text{world}}$：统一到world frame后的point
- 取逆是因为 $T_{\text{cam},i}^{\text{world}}$ 表示的是world在camera frame下的描述，要反过来变换points

**直觉**：所有不同帧、不同head pose下的point clouds都被"踩"到同一个world坐标系里，policy看到的scene geometry是spatially consistent的。这是后面3D point cloud policy能work的几何基础。

参考：FoundationStereo https://arxiv.org/abs/2503.05518 ；FoundationPose https://arxiv.org/abs/2403.07041 ；SAM2 https://arxiv.org/abs/2408.00714 ；Grounded-SAM https://arxiv.org/abs/2401.14159

---

## 4. Policy设计（这是paper最巧妙的engineering）

整个task分三stage：

### 4.1 Pre-grasp stage
- **AnyGrasp** [40] 生成grasp pose
- 高精度任务用fixed strategy

### 4.2 Motion Planning stage（核心）

Policy基于**RISE** [13] backbone（3D point cloud diffusion policy），关键改动：

**关键设计1: Point cloud as input**
- ActiveGlasses只有单个active camera，每个episode起始head pose不同，2D image space observation高度inconsistent
- 3D point cloud in world frame天然解决这个问题（视角变化时scene geometry不变）

**关键设计2: Two separate diffusion heads**
- **Manipulation head**: 预测object的6-DoF absolute trajectory $\{T_{\text{world}}^{\text{obj}}\}_{t}^{t+T}$
- **Perception head**: 预测head的6-DoF relative trajectory $\{T_{\text{world}}^{\text{head}}\}_{t}^{t+T}$

为什么object用absolute，head用relative？这是经过ablation反复打磨的：
- **Object absolute**: 与3D policy representation对齐，policy更容易学correlation
- **Head relative**: 人头和perception arm end-effector的height/spatial position inherent不同，absolute会让perception arm在policy开始时move一大段，逼近workspace boundary导致IK失败

**关键设计3: 不把current object pose作为condition**

这个ablation结果反直觉但深刻（Table II）：
- abs w/o curr pose: 14/20 ✅ (default)
- abs w/ curr pose: 3/20 ❌
- rel w/ curr pose: 10/20
- rel w/o curr pose: 一（不work）

作者的解释非常insightful：当输出absolute trajectory时，如果额外condition on current object pose，policy会发现"shortcut solution"——直接根据当前pose做小offset即可，**忽略visual observation**，学到near-fixed trajectory。本质上模型overfit到dataset里的dominant motion pattern，丧失对scene variation的响应能力。

这跟behavior cloning里的"action shortcut"问题同源：当你给policy一个能"作弊"的强signal时，它会放弃从raw observation中reasoning。

**关键设计4: Object→EE transform**

借用**SPOT** [41]的思路：
$$T_{\text{obj}}^{\text{EE}} = T_{\text{cam}}^{\text{EE}} \cdot T_{\text{obj}}^{\text{cam}}$$

通过camera calibration得到object到end-effector的固定transform，policy只预测object trajectory，下游再映射成robot action。这正是cross-embodiment的key：不同robot arm只要标定 $T_{\text{cam}}^{\text{EE}}$ 就能复用同一policy。

**关键设计5: Termination**
- 作为policy output的额外维度
- Training时episode最后5帧label为1，其余为0
- Inference时 $f_t > \text{threshold}$ 即终止

### 4.3 Algorithm 1流程
```
for t = 0 to T:
    1. 从 (l_t, r_t) 估计depth d_t
    2. 重建point cloud p_t^cam
    3. if t=0: calibrate得到 T_world^cam,0
       else: T_world^cam,t = T_world^cam,0 · T_cam,0^cam,t
    4. 转world frame: p_t = T_world^cam,t · p_t^cam
    5. Clip distant points
    6. Policy inference: π(p_t) → {T_world^obj}, {T_world^head}, f_t
    7. Execute: robot traj = T_world^obj · T_obj^EE; head traj = T_world^head
    8. if f_t > threshold: break
```

参考：RISE https://arxiv.org/abs/2410.09623 ；SPOT https://arxiv.org/abs/2410.23039 ；AnyGrasp https://arxiv.org/abs/2307.13495

---

## 5. 实验：三个challenging tasks

### 5.1 Task设计
**Book Placement**: 把书插入已有3本书的书架空位，camera被书架侧壁occluded。需要先靠近再旋转观察空位。
**Bread Insertion**: 把面包片插入toaster第一个slot，slot开始时不可见。需要倾斜重定向camera观察slot。
**Occluded Distant Water Pouring**: 越过屏风，把teapot定位到cup上方倒水。cup被screen occluded。

每个task分3 stage评估，反映task progress。

### 5.2 Main Results (Table I)

| Method | Book S1/S2/S3 | Bread S1/S2/S3 | Pour S1/S2/S3 |
|---|---|---|---|
| ActiveGlasses | 20/16/**14** | 20/15/**11** | 20/15/**10** |
| w/o active vision | 20/8/7 | 11/1/0 | 18/10/4 |
| Pi0.5 | 20/9/7 | 20/18/6 | 20/12/4 |

训练数据：Book 200 demos, Bread 100 demos, Pour 100 demos。

**核心发现**：
1. Stage 1（approach）所有方法都work，差异主要在Stage 2/3
2. w/o active vision在Bread上Stage 1就崩（11/20），因为toaster slot不可见时fixed camera根本没法定位
3. Pi0.5（pretrained VLA）在小数据下没法从2D image学head-hand joint distribution，manipulation arm倾向于ignore image走near-fixed trajectory
4. ActiveGlasses的6-DoF head movement补偿了wrist camera的缺失，point cloud representation稳定了visual observation

**为什么Pi0.5输给from-scratch 3D policy**？作者的解释很精彩：Pi0.5是pretrained大数据VLA，但in-the-wild bimanual setting下，head image + manipulation action的joint distribution在2D image space高度diverse，**小dataset下policy提取不出visual pattern**。而point cloud representation把observation space"压缩"到geometrically meaningful的子空间，小数据也能学。这印证了一个deep lesson：**representation的选择比model capacity更重要**。

### 5.3 Scalability (Fig 5)

比较ActiveGlasses vs UMI vs Quest 3 VR teleop vs Sigma.7 haptic teleop：
- **Completion time**: ActiveGlasses最快（bare hand没有teleop mapping overhead，没有UMI夹子操作delay）
- **Success rate**: ActiveGlasses在Bread insertion上优势最明显（高精度任务）
- **Device weight**: ActiveGlasses≈XREAL+ZED Mini重量，远低于UMI的600g+
- Sigma标记为0g是因为zero-gravity mode

### 5.4 Cross-embodiment (Table III)

| Robot | Book S1/S2/S3 |
|---|---|
| Flexiv Rizon 4 | 20/16/14 |
| UR5 | 20/16/11 |

同一policy在UR5上zero-shot deploy，Stage 1/2完全持平，Stage 3略差（UR5 workspace小，靠近limit时flexibility差）。证明object-centric representation天然cross-embodiment。

---

## 6. 我的Intuition总结

读完整篇paper，我能提炼出几个deep insights：

### Insight 1: Active vision是manipulation的一部分，不是perception的附加
人脑里"看"和"动"是coupled的——你伸手抓杯子时眼头会自动follow。wrist camera的passive sensing本质上把人这个coupling切断，policy只能学到"瞎摸"。ActiveGlasses通过显式记录+预测head movement，恢复了head-hand joint distribution的学习。

### Insight 2: Representation决定data efficiency ceiling
Pi0.5的2D image representation在小dataset下学不出head-hand joint distribution；3D point cloud representation因为geometric invariance，小数据也能work。这不是model size的问题，是**信息bottleneck的设计**问题。

### Insight 3: Object-centric是cross-embodiment的"免费午餐"
不retargeting human hand kinematics到robot，直接预测object 6-DoF trajectory，让policy与embodiment解耦。代价是失去了end-effector层面的直接control，但gain是cross-platform transfer和更自然的human demonstration。

### Insight 4: Condition signal要"刚好够用"，多了会让policy偷懒
Table II的ablation最impressive：给policy current object pose作为condition，性能从14/20掉到3/20。模型找到了shortcut（直接offset当前pose）就放弃perception。这跟LLM里"chain-of-thought被奖励hack"的问题同构——你给的inductive bias太强，模型就学不到真正reasoning。

### Insight 5: Absolute vs Relative action representation的选择是工程性的
Object用absolute好（geometric consistency），head用relative好（避免workspace limit + IK failure）。没有普适最优，需要根据具体物理约束选。

---

## 7. 局限与未来方向

paper没深入讨论的几点：
- **数据collection cost**: FoundationStereo + SAM2 + FoundationPose pipeline每帧推理很慢，离real-time data processing还远
- **Object mesh prior**: FoundationPose需要object mesh作为输入，对novel object不友好
- **Pre-grasp依赖AnyGrasp或fixed strategy**: 高精度任务还是hand-engineered，没实现full end-to-end
- **Perception arm wheeled table randomization**: 虽然模拟了torso movement，但real human场景里body movement更复杂
- **没有scalability到long-horizon task**: 每个task都是single object manipulation，复杂chain没测

未来的自然延伸：
- 用SLM-based VLA替代RISE backbone（结合EgoVLA [20]的思路）
- 跟DexCap [14]/DexUMI [16]结合做dexterous manipulation with active vision
- 用NeRF/3DGS替代point cloud，得到更dense的scene representation
- 结合Eye, Robot [33]的BC-RL perception-action loop，让active vision有RL fine-tune的可能

---

## 8. 参考链接

主paper参考（无arxiv链接，作者affiliation为SJTU + Noematrix）：
- 项目推测GitHub：https://github.com/Haojintai/ActiveGlasses （如有）

引用的关键工作：
- UMI: https://arxiv.org/abs/2402.10329
- RISE: https://arxiv.org/abs/2410.09623  
- FoundationStereo: https://arxiv.org/abs/2503.05518
- FoundationPose: https://arxiv.org/abs/2403.07041
- SAM2: https://arxiv.org/abs/2408.00714
- Grounded-SAM: https://arxiv.org/abs/2401.14159
- AnyGrasp: https://arxiv.org/abs/2307.13495
- π0.5: https://arxiv.org/abs/2504.16054
- SPOT: https://arxiv.org/abs/2410.23039
- EgoMimic: https://arxiv.org/abs/2510.15156
- EgoZero: https://arxiv.org/abs/2505.20290
- EgoMi: https://arxiv.org/abs/2511.00153
- ViA: https://arxiv.org/abs/2506.15666
- Eye, Robot: https://arxiv.org/abs/2506.10968
- DexCap: https://arxiv.org/abs/2403.07788
- DexUMI: https://arxiv.org/abs/2505.21864
- Active Perception (Bajcsy 1988): https://ieeexplore.ieee.org/document/5968

---

总的来说，这是一篇engineering非常solid的system paper，核心贡献是把"active vision"从一个concept落成一个data collection + policy training的完整pipeline。最让我兴奋的是Table II那个condition ablation——它揭示了一个普适的learning principle：**给模型的conditional information要刚好让它必须从observation reasoning，给多了它就作弊**。这个intuition不止适用于robotics，对LLM prompting、RL reward design都适用。
