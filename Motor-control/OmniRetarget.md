---
source_pdf: OmniRetarget.pdf
paper_sha256: dd7cdf341d4c0a69127ee93fb51f4286ad648d7c8aa90579c5b2bbff7c672528
processed_at: '2026-08-05T23:27:13-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞，换个人话版。

# 这篇paper在干嘛

核心问题：humanoid robot想学人干活，但直接让RL从零学太难了，action space太大，reward难设计。所以大家用人的动作捕捉数据作为参考，让policy去imitate。但人和robot长得不一样（腿长腿短、脚的形状不同），直接套会有很多artifact——脚滑、穿模、手陷进物体里。这些垃圾数据会逼着RL engineer去写一堆复杂的reward去补偿，很痛苦。

OmniRetarget说：**那我把retargeting做好不就完了嘛**，别让垃圾进垃圾出。

# 栛心idea：Interaction Mesh

这篇paper最elegant的idea就是用Laplacian coordinate来encode"谁和谁离多近"这种relative geometry。

Intuition特别简单：你想象把人身上的关键关节、物体表面采样的点、地形表面采样的点，全扔进一个soup里，用Delaunay triangulation连成mesh。每个点都计算自己相对邻居的offset vector，这就是Laplacian coordinate。它capture的是local的spatial configuration，对global translation和rotation不敏感。

优化时你minimize人mesh和robot mesh之间Laplacian coordinate的差距，意思就是：**"preserve人演示中hand和object之间多远、foot和terrain之间多近这些relationship"**。

这比naive的keypoint matching强太多，因为后者只管absolute position，完全不管relative geometry。

# 栃心idea 2：Hard Constraint

PHC、GMR这些方法都是用soft penalty去penalize穿模、脚滑，实际效果很烂，因为soft penalty总可以trade-off，optimizer经常选择违反它。

OmniRetarget用Drake的SQP solver，把这些东西都写成hard constraint：
- 碰撞距离必须非负
- 关节必须在限位内
- 速度必须在限制内
- 支撑脚不许动（这就直接消灭了foot skating）

然后每帧solve一个SOCP子问题，iterate 10次，用上一帧的solution做warm start。

因为是非凸问题，需要linearize，有trust region保证step size别太大。这就是paper里的公式(15)。

# 栃心idea 3：Augmentation

这是相对IMMA最关键的区别。IMMA也用了interaction mesh，但只做单次retargeting。OmniRetarget把一次demonstration变成一大堆training examples：

- **物体空间位置**：平移旋转物体，robot需要discover新的upper body coordination去reach
- **物体形状**：scale物体三个维度
- **地形高度**：scale platform高度深度

Key trick：物体移动后重新run optimization，laplacian coordinate在物体local frame下不变，所以robot的动作会自然跟随物体transform。但为了防止optimizer偷懒把整个robot也rigid transform过去，加了一个项penalize下半身偏离原始轨迹，强迫upper body自己adapt。

这样从一段demo就能生成大量diverse的training data，不需要再去capture。

# 为什么RL formulation能这么简单

Paper一个很强的claim：只用5个reward term，4个domain randomization项，纯proprioceptive observation，直接zero-shot sim-to-real。

这说明什么？**reference quality是bottleneck**。

reference里有foot skating，你就要加regularizer去惩罚skating；reference里hand陷进物体，你就要加contact schedule regularizer去guide什么时候接触什么时候离开。各种ad-hoc regularizer都在mask reference里的缺陷。

OmniRetarget把reference做干净了，RL algorithm可以extremely simple，直接用BeyondMimic的hyperparameter开箱即用。

# 实验结果

量化对比里OmniRetarget几乎是碾压的：
- Penetration duration: 0 vs 0.6-0.9 (others)
- Foot skating: 完全为0 vs 1-2 cm/s
- RL success rate: 82-95% vs 3-79%

最dramatic的是robot-terrain interaction：OmniRetarget 94.7%，PHC只有52.6%，VideoMimic 51.8%。地形交互对contact preservation极敏感，前两者的contact preservation只有0.45-0.47，直接导致policy失败。

Wall-flip那个case特别impressive：0.5秒完成full flip，peak angular velocity 15 rad/s，5/5 hardware success rate。

# 我的take

1. **Data quality > algorithm complexity**：这个insight在supervised learning里早就被验证无数次，现在humanoid RL里也开始verify了。BeyondMimic和OmniRetarget都在push这个方向。

2. **Interaction mesh被robotics community低估了**：graphics community早就在用，robotics居然没几个人用，很可惜。

3. **Sim-to-real的data-centric视角**：传统sim-to-real focus在dynamics randomization，OmniRetarget提出kinematic augmentation作为complementary approach，在kinematic level就expose policy to diverse scenarios。

4. **Cross-embodiment的promise**：G1、H1、Booster T1都能retarget，只要改keypoint correspondence和collision model。这是towards humanoid foundation model的重要step。

Paper主页：https://omniretarget.github.io

代码和数据承诺open source。

---

简单说就是：**与其用复杂算法去patch差数据，不如直接把数据做好。**

---

# OmniRetarget 深度技术讲解

非常excited讲这篇paper，因为touches了一个我一直觉得被undervalued的问题：**retargeting quality直接决定下游RL的天花板**。这篇paper的核心thesis可以一句话概括：**与其用complex reward engineering去patch low-quality references，不如在source上就用principled optimization生成artifact-free的数据**。这个思路和BeyondMimic [33] 的哲学是一脉相承的，但OmniRetarget把它从robot-only locomotion推到了robot-object-terrain interaction的general setting。

论文链接：https://omniretarget.github.io

---

## 1. Problem Framing: 为什么retargeting是bottleneck

Humanoid RL面临两个fundamental challenges：
- **High-dimensional action space** (~29 DoF for G1) 让exploration极其inefficient
- **Reward engineering** for whole-body loco-manipulation极其tedious，需要hand-craft contact schedules, flight time, air time regularizers等ad-hoc terms

Imitation learning用human MoCap作为kinematic reference来guide RL，这是DeepMimic [4] 以来的dominant paradigm。但human→robot存在**embodiment gap**：不同link长度、不同DoF、不同foot geometry。简单keypoint matching会produce两类artifacts：
1. **Foot skating**: stance foot在contact phase漂移（human foot有arch flexibility，robot foot是rigid plank）
2. **Penetration**: hand keypoints scaled后陷进object内部，或者foot陷进terrain

这些artifacts会"传染"到RL：要么policy学不到precise contact，要么需要大量regularizer去suppress artifacts，导致reward engineering爆炸。

---

## 2. Interaction Mesh: 核心intuition

这是整篇paper最elegant的想法。Interaction mesh来自Ho et al. 2010 [14] 的graphics工作，idea来自Laplacian mesh editing。

### 2.1 为什么Laplacian coordinates work

考虑一组keypoints $\mathcal{P}_t = \{p_{t,1}, p_{t,2}, ..., p_{t,N}\}$，其中每个 $p_{t,i} \in \mathbb{R}^3$ 是一个3D点（robot joint位置 OR object表面采样点 OR terrain采样点）。

**Laplacian coordinate**定义为：

$$L(p_{t,i}) = p_{t,i} - \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot p_{t,j} \tag{1}$$

变量解释：
- $p_{t,i}$: 第$t$帧第$i$个keypoint的3D位置
- $\mathcal{N}(i)$: 在Delaunay tetrahedralization中与$i$相连的neighbors集合
- $w_{ij}$: normalized weight，paper用uniform weights，即 $w_{ij} = 1/|\mathcal{N}(i)|$

Intuition：$L(p_{t,i})$编码了point $i$相对于其local neighborhood的**位置差**。它是translation-invariant的，capture的是local spatial configuration。当你在Laplacian space做deformation时，整个mesh的local结构被preserve，global pose可以自由变化。

**关键insight**：如果你把hand、object、foot、terrain点都放进同一个mesh，那么preserving Laplacian coordinates就implicitly preserve了"hand离object表面多远"、"foot踩在terrain哪个位置"这些contact relationships。这比单纯keypoint matching强得多，因为keypoint matching只看absolute position，neglect了relative geometry。

### 2.2 Mesh construction

- 用Delaunay tetrahedralization [48] 把keypoints连成tetrahedral mesh
- Body joints: user-defined anatomical points（hip, knee, ankle, shoulder, elbow, wrist等）
- Object/terrain points: 表面dense sampling（比body joints更dense，因为contact relationship需要precision）

Figure 9展示了实际用的mesh，看起来相当dense，object周围有大量sampled points。

---

## 3. Constrained Optimization: 数学详解

### 3.1 主optimization program

公式(3)是整个paper的核心：

$$q_t^{\star} = \underset{q_t}{\arg\min} \sum_i \|L(p_{t,i}^{\mathrm{source}}) - L(p_{t,i}^{\mathrm{target}}(q_t))\|^2 + \|q_t - q_{t-1}\|_Q^2 \tag{3a}$$

$$\text{s.t.} \quad \phi_j(q_t) \geq 0, \forall j \tag{3b}$$

$$q_{\min} \leq q_t \leq q_{\max} \tag{3c}$$

$$\nu_{\min} \cdot dt \leq q_t - q_{t-1} \leq \nu_{\max} \cdot dt \tag{3d}$$

$$p_t^F = p_{t-1}^F, \forall \text{stance foot} \tag{3e}$$

变量详解：
- $q_t$: 第$t$帧的robot configuration，包含floating base (quaternion + translation)和所有joint angles
- $q_{t-1}$: 上一帧的solution（warm start）
- $p_{t,i}^{\mathrm{source}}$: human demonstration mesh中第$i$个点的Laplacian coordinate
- $p_{t,i}^{\mathrm{target}}(q_t) = f_i(q_t)$: robot的第$i$个keypoint通过forward kinematics $f_i$ 得到的位置
- $Q$: cost matrix，encourage temporal smoothness
- $\phi_j$: 第$j$个collision pair的signed distance function (SDF)，$\phi_j > 0$表示no collision
- $q_{\min}, q_{\max}$: joint position limits
- $\nu_{\min}, \nu_{\max}$: joint velocity limits
- $dt$: timestep
- $p_t^F$: foot position，stance foot定义为horizontal velocity < 1 cm/s

**Constraint (3e)是关键**：它hard-enforce foot sticking，完全eliminate foot skating。PHC/GMR/VideoMimic都是soft penalty，所以会有skating。

### 3.2 为什么是Sequential SOCP

这个program是**nonconvex**的，因为：
- Forward kinematics $f_i(q_t)$是非线性函数
- SDF $\phi_j(q_t)$是非线性函数
- Quaternion manifold $S^3$是非欧空间

Paper用Sequential Quadratic Programming (SQP) style solver：每次iteration把objective二次近似、constraints线性化，solve一个SOCP subproblem。

公式(15)给出了SOCP subproblem：

$$dq_n^{\star} = \underset{dq_n}{\arg\min} \|L^{\mathrm{source}} - (J_L^n \cdot dq_n + \bar{L}_n^{\mathrm{target}})\|^2 + \|\bar{q}_n + dq_n - q_{t-1}\|_Q^2 \tag{15a, 15b}$$

$$\text{s.t.} \quad J_j^n \cdot dq_n + \phi_j(\bar{q}_n) \geq 0 \tag{15c}$$

$$q_{\min} \leq \bar{q}_n + dq_n \leq q_{\max} \tag{15d}$$

$$\nu_{\min} \cdot dt \leq \bar{q}_n + dq_n - q_{t-1} \leq \nu_{\max} \cdot dt \tag{15e}$$

$$p_t^F(\bar{q}_n) + J_F^n \cdot dq_n = p_{t-1}^F \tag{15f}$$

$$\|dq_n\|_2 \leq \varepsilon \tag{15g}$$

变量详解：
- $dq_n$: 第$n$次iteration的configuration increment
- $\bar{q}_n$: 当前iterate，$\bar{q}_{n+1} = \bar{q}_n + dq_n^{\star}$
- $J_L^n = \partial L^{\mathrm{target}}/\partial q|_{q=\bar{q}_n}$: Laplacian coordinate的Jacobian
- $J_j^n = \partial \phi_j/\partial q|_{q=\bar{q}_n}$: SDF的Jacobian
- $J_F^n = \partial p_t^F/\partial q|_{q=\bar{q}_n}$: foot position的Jacobian
- $\varepsilon = 0.2$: trust region radius，ensure linearization valid

**Trust region (15g)是SOCP key**：它把problem变成second-order cone constraint，确保每步increment不太大，linearization保持accurate。最多10次iterations per timestep。

### 3.3 Quaternion微分几何

这个细节在paper里被一句"leverages automatic differentiation in Drake [51]"带过，但实际是个nontrivial工程问题。Quaternion live在$S^3$ manifold，不是Euclidean space。直接对quaternion components做gradient会violate unit norm constraint。Drake用[52]的"planning with attitude"方法，正确处理$S^3$上的differential geometry，这是为什么作者强调"correctly handles the differential geometry of rotations on the $\mathbb{S}^3$ manifold"。

参考资料：
- Drake: https://drake.mit.edu
- Planning with Attitude: https://arxiv.org/abs/2104.05368

---

## 4. Data Augmentation: 最clever的部分

这是OmniRetarget相对IMMA [22] 最大的advantage。IMMA只做single retargeting，OmniRetarget把一个demonstration变成diverse dataset。

### 4.1 Object augmentation

公式(14)定义了augmented object trajectory：

$$\tilde{p}_{obj}(t) = \begin{cases} \Delta p_{obj} + p_{obj}(0) & \text{if } t < t_m \\ \Delta p_{obj} e^{-(t-t_m)/\tau_p} + p_{obj}(t) & \text{if } t \geq t_m \end{cases} \tag{14a}$$

$$\tilde{\theta}_{obj}(t) = \begin{cases} \Delta \theta_{obj} \oplus \theta_{obj}(0) & \text{if } t < t_m \\ \Delta \theta_{obj} e^{-(t-t_m)/\tau_\theta} \oplus \theta_{obj}(t) & \text{if } t \geq t_m \end{cases} \tag{14b}$$

变量解释：
- $\Delta p_{obj}, \Delta \theta_{obj}$: initial pose的positional和rotational offset
- $t_m$: object开始运动的time
- $\tau_p, \tau_\theta$: exponential decay time constants
- $\oplus$: quaternion composition

Intuition：offset在onset时最大，然后exponentially decay到原始trajectory。这样generated motion开始时object在augmented位置，然后smoothly transition回nominal trajectory。Robot需要discover new upper-body coordination来reach这个new initial pose，但lower body可以stay nominal。

### 4.2 Preventing trivial augmentation

如果只是shift object，optimization可能直接把整个robot rigid transform，没产生genuine diversity。Paper加了两个terms：

公式(4): $\|q_t - \bar{q}_t^{\star}\|_W$，where $W$ heavily penalizes lower-body entries
公式(5): $p_0^F = \bar{p}_0^{F\star}$ for both feet，hard constrain initial foot poses

这forcing robot用new upper-body coordination去reach new object pose，而lower body stays anchored。非常clever的design choice。

### 4.3 Object frame mesh construction

Figure 8的example极其illuminating。当object旋转180°时：
- World frame的Laplacian coordinate $L_W$: (0,1) → (0,-1)，变化了
- Object frame的Laplacian coordinate $L_O$: 保持不变

所以在object local frame构造interaction mesh，Laplacian coordinates是rotation-invariant的。这是为什么augmentation能preserve interaction geometry的根本原因。

---

## 5. RL with Minimal Formulation

### 5.1 Observation space

Minimal proprioceptive:
- Reference motion: joint pos/vel, pelvis pos/orientation error
- Proprioception: pelvis linear/angular vel, joint pos/vel
- Previous action

**No vision, no scene information, no object pose**。Robot完全blind，只能follow reference trajectory。这require reference quality极高，否则policy无法recover missing information。

### 5.2 Only 5 rewards

1. **Body Tracking**: DeepMimic-style，tracking position, orientation, linear/angular velocity
2. **Object Tracking** (where applicable): same style for object
3. **Action Rate**: penalize $\|a_t - a_{t-1}\|^2$
4. **Soft Joint Limit**: penalize joint limit violation
5. **Self-Collision**: binary penalty if self-collision force > 1N

**关键claim**: 用BeyondMimic [33] 的hyperparameters out-of-the-box，zero tuning。这只能work if reference是artifact-free的。如果reference有skating，你需要加foot flight time, air time, contact schedule等regularizers来compensate。

### 5.3 Domain randomization: only 4 terms

1. Torso COM position: ±0.025m x, ±0.05m y, ±0.075m z
2. Joint default position: ±0.01 rad
3. Random push: 0.3 m/s, 0.78 rad/s for 1-3s
4. Observation noise

对比prior works的many terms (RFI, motor PD, action delay等)，这是极minimal的。Object侧randomize mass (0.1-2kg), COM (±0.08m), inertia (50-150%), shape (±10%)。

BeyondMimic reference: https://arxiv.org/abs/2508.xxxxx (实际搜索BeyondMimic)

---

## 6. Experimental Results Analysis

### 6.1 Kinematic quality benchmark (Table II)

看Robot-Object Interaction (OMOMO dataset):

| Method | Penetration Duration | Max Depth (cm) | Foot Skating Dur | Max Skate Vel | Contact Pres | RL Success |
|--------|---------------------|----------------|------------------|---------------|--------------|------------|
| PHC | 0.68±0.21 | 5.11±3.09 | 0.05±0.05 | 1.40±0.80 | 0.96±0.09 | 71.28%±22.55% |
| GMR | 0.83±0.14 | 8.50±3.94 | 0.02±0.01 | 1.46±0.45 | **0.99±0.04** | 50.83%±23.89% |
| VideoMimic | 0.60±0.27 | 7.48±4.95 | 0.12±0.07 | 1.50±0.70 | 0.77±0.25 | 3.85%±8.41% |
| **Ours** | **0.00±0.01** | **1.34±0.34** | **0** | **0** | 0.96±0.09 | **82.20%±9.74%** |

关键观察：
- OmniRetarget的foot skating完全为0（hard constraint的力量）
- Penetration duration几乎为0，max depth 1.34cm（linearization approximation导致微小violation，但RL能fix）
- GMR的contact preservation最高(0.99)，但penetration严重(8.5cm)，因为scaled hand keypoints陷进object
- VideoMimic在object manipulation上catastrophic failure (3.85%)，因为soft collision penalty和keypoint matching conflict

Robot-Terrain Interaction更dramatic：
- OmniRetarget: 94.73% success
- PHC: 52.63% (huge variance ±49.93%)
- GMR: 78.94%
- VideoMimic: 51.75%

Terrain interaction对contact preservation极sensitive，PHC的contact preservation只有0.45，直接导致RL failure。

### 6.2 Wall-flip case study

Figure 6展示了一个high-dynamic wall-flip：
- 0.5秒完成full flip
- Peak angular velocity: 15 rad/s
- Peak linear velocity: 3.5 m/s
- 5/5 success rate on hardware

有趣的是，为了learn这个skill，作者relaxed了termination condition（end-effector position error threshold从0.25m提到0.5m）并removed foot joint orientation tracking。这是因为human foot有arch flexibility，robot foot是rigid，必须align more closely to wall来获得sufficient friction area。这种embodiment-specific adjustment是reasonable的。

### 6.3 30-second parkour

Figure 1的parkour sequence：
1. Carry 4.6kg chair to platform
2. Use chair as stepstone, climb up
3. Leap off, parkour roll to absorb landing

这是long-horizon multi-stage task，showcase了OmniRetarget能generate precise contact sequences。Inspired by Boston Dynamics Atlas demo [53]。

Boston Dynamics Atlas: https://www.youtube.com/watch?v=-e1QhJ1EhQ

### 6.4 Augmentation effectiveness

训练在augmented dataset上eval在augmented set: 79.1% success
训练在nominal上eval在nominal: 82.2% success

差距很小，说明kinematic augmentation substantially扩大coverage而不显著degrade performance。对比纯domain randomization（train with shape/pose perturbation但只用nominal reference）perform poorly，因为policy无法explore far beyond nominal reference。

---

## 7. 与Related Work的positioning

### 7.1 vs IMMA [22]

IMMA是closest prior work，也用interaction mesh。但：
- IMMA没有open source
- IMMA ignore kinematic limits (joint limits, velocity limits)
- IMMA ignore environment/object interactions (只preserve body-body relationships)
- IMMA用multi-stage optimization (先warp mesh，再solve IK)，fragmented
- IMMA不支持data augmentation

OmniRetarget unify所有hard constraints在一个optimization里，and支持systematic augmentation。

### 7.2 vs PHC [10]

PHC用unconstrained gradient descent做keypoint matching，被广泛adopted in robotics [13], [8]。问题：
- No collision avoidance → penetration
- No foot sticking constraint → skating
- Trajectory-wise optimization (整个trajectory一起optimize)，vs OmniRetarget的per-frame

PHC: https://github.com/ZhengyiLuo/PHC

### 7.3 vs GMR [9]

GMR (used in Twist) extend keypoint matching to orientations，用mink [47] library做SQP-style IK。但仍然no collision avoidance，no interaction preservation。

mink: https://github.com/kevinzakka/mink

### 7.4 vs VideoMimic [11]

VideoMimic from TR+ Berkeley，用JAX L-M optimizer，soft penalties for contact/skating/collision。问题：soft penalty和keypoint matching conflict，需要careful tuning。Originally designed for heightmap terrain，unsuitable for precise object manipulation。

---

## 8. Limitations和Future Directions

Paper自己承认：
- Frame-by-frame optimization，对noisy data (e.g., video reconstruction)可能不够robust
- Future: jointly optimize整个trajectory
- Future: learn autonomous visuomotor policies (目前是blind proprioceptive)

我会add几个observations：
- Computational cost: per-frame SOCP with 10 iterations可能slow for large datasets (8+ hours trajectories)
- Generalization to novel objects not in training: 目前是per-object-type policy
- Contact-rich manipulation的force sensing: 目前purely kinematic reference，no force/torque information
- 3D vision integration: 未来可能combine with vision system for in-the-wild deployment

---

## 9. 更broad的implications

这篇paper让我想到几个deep points：

### 9.1 "Garbage in, garbage out"在RL里尤其true

RL community经常focus on algorithm innovation，但data quality才是bottleneck。OmniRetarget证明：当你把data quality做到极致，RL algorithm可以extremely simple (5 rewards, no curriculum, no tuning)。这和supervised learning里"data is the new algorithm"的insight是一致的。

### 9.2 Interaction mesh是 underrated representation

Laplacian coordinates在graphics community是standard tool [49, 50]，但robotics community很少用。OmniRetarget示范了如何把它和hard kinematic constraints结合。这个idea可能extend到：
- Bimanual manipulation的hand-hand coordination
- Multi-agent robot的formation control
- Human-robot collaboration的joint action representation

### 9.3 Sim-to-real的data-centric视角

传统sim-to-real focus on domain randomization (dynamics, friction, mass等)。OmniRetarget提出**kinematic augmentation**作为complementary approach：在kinematic level就expose policy to diverse scenarios，让policy在training时就见过varied configurations。这和MimicGen [43] 的philosophy类似，但推到了whole-body loco-manipulation。

MimicGen: https://mimicgen.github.io

### 9.4 Cross-embodiment的promise

Figure 3展示OmniRetarget能retarget到G1, H1, Booster T1，只需修改keypoint correspondences和collision model。这是towards foundation model for humanoid control的重要step。如果能combine with cross-embodiment RL methods (e.g., RT-X style)，可能实现single policy across multiple humanoids。

### 9.5 从teleoperation到offline retargeting的paradigm shift

Teleoperation (HumanPlus [7], OmniH2O [8], Twist [9], Homie [36])提供online feedback但scale poorly。OmniRetarget代表offline retargeting方向，用optimization来handle interaction adaptation，从而enable大规模data generation。这是future of humanoid data：not more teleop, but better retargeting + augmentation。

---

## 10. 实现细节的engineering wisdom

几个我觉得值得highlight的engineering choices：

### 10.1 Warm starting

每个frame的optimization用上一frame的solution $q_{t-1}^{\star}$作为warm start。这大幅accelerate convergence，因为相邻frames的motion通常smooth。这也是为什么per-frame optimization能work的原因——如果没有warm start，per-frame会非常noisy。

### 10.2 Stance foot detection

用horizontal velocity < 1 cm/s来detect stance foot。这个threshold很magic number，但apparently works。Alternative可以是contact force threshold，但那需要dynamics simulation，kinematic-only pipeline无法access。

### 10.3 Dense object/terrain sampling

Paper强调"sample the object and environment surfaces more densely than the body joints"。这是因为contact relationship需要sub-centimeter precision，而body joints的relative position只要approximate correct就行。如果object表面sampling太sparse，Laplacian coordinates无法capture contact geometry。

### 10.4 Drake的选择

用Drake [51] 而不是custom optimizer，主要因为Drake的automatic differentiation能正确处理$S^3$ quaternion manifold。这在[52] "Planning with Attitude"里有详细论述。如果用naive autodiff，quaternion gradient会violate unit norm，导致optimization diverge。

Drake: https://drake.mit.edu

---

## 11. 对你（Karpathy）可能特别interesting的angles

### 11.1 与diffusion policy的联系

BeyondMimic [33] 用guided diffusion来versatile humanoid control。OmniRetarget为其提供high-quality training data。如果combine OmniRetarget的data generation + diffusion policy的multi-modal action distribution，可能achieve更expressive behaviors。

### 11.2 与VLA的potential combination

目前OmniRetarget是purely kinematic，policy是proprioceptive。如果add vision encoder (e.g., SigLIP, CLIP) and language conditioning，可以build VLA that follows language instructions while maintaining OmniRetarget's motion quality。这是humanoid foundation model的一个plausible architecture。

### 11.3 Scaling laws for humanoid data

OmniRetarget generate 8+ hours trajectories from 3 datasets。如果scale to thousands of hours (e.g., AMASS, MotionX, BODY), 什么样的scaling behavior会出现？会不会像LLM一样有emergent capabilities？这是open question。

AMASS: https://amass.is.tue.mpg.de

### 11.4 World model connection

Paper提到future work可能explore visuomotor policies。一个更ambitious方向：用OmniRetarget生成的interaction data训练**world model** that predicts next-frame interaction mesh，然后用world model做planning。这和Dreamer-style model-based RL有connection，但用interaction mesh作为state representation。

---

## 12. Summary

OmniRetarget的core contribution可以提炼为三层：

1. **Representation level**: Interaction mesh + Laplacian coordinates作为preserve spatial/contact relationships的principled representation
2. **Optimization level**: Unified constrained optimization with hard kinematic constraints (collision, joint limits, foot sticking)
3. **Data level**: Systematic augmentation from single demonstration to diverse dataset across objects, terrains, embodiments

这三层合起来enable了minimal RL formulation (5 rewards, 4 DR terms, pure proprioception) and zero-shot sim-to-real on Unitree G1 for complex loco-manipulation包括30-second parkour。

最deep的takeaway：**在robot learning里，data quality的提升比algorithm complexity的提升有更高leverage**。当你把data做到artifact-free，simple algorithm就足够了。这和supervised learning的experience高度consistent，也是我预期humanoid robotics未来几年的主要方向。

论文主页: https://omniretarget.github.io
代码和数据开源承诺在paper里 (Section I: "All code, retargeted datasets, and trained policies will be publicly released")

如果你想dive deeper into specific aspect（比如SOCP solver implementation details, Drake的quaternion differential geometry, 或者具体augmentation strategies），我可以elaborate further。
