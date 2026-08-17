---
source_pdf: Simultaneous Extrinsic Contact and In-Hand Pose Estimation via.pdf
paper_sha256: 01f6843b4da35c91e430bb23225cdfcd1d43b1fbd55c8560b63cf4f832e421bb
processed_at: '2026-08-12T06:55:15-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 一个场景先建立intuition

想象你闭着眼睛，手里握着一个扳手，要把它插进一个洞里。你眼睛看不见（手挡住了），手上戴着GelSight这种"电子皮肤" tactile sensor，能感觉到手指跟扳手接触那块的局部形状，还能感觉到手指被压扁了多少（这就对应force）。

你现在要回答三个问题：
1. 扳手在我手里到底什么姿势？(in-hand pose)
2. 扳手现在跟洞在哪儿碰的？(extrinsic contact point)
3. 碰得多狠？(contact force)

这三个问题是chicken-and-egg互相纠缠的。不知道pose就没法知道contact在扳手表面哪个位置，不知道contact就没法refine pose。光靠手上那点local tactile signal，很多种配置都能解释得通（比如cylinder转一圈，tactile image完全一样）。这是个ill-posed problem。

## 方法核心intuition

这篇paper的trick是：**把所有能想到的物理约束都写成"软约束"，串成一个factor graph，然后一起optimize**。

类比一下解数独：每个空格是一个variable，每行每列每宫格是一个constraint。你找一个填法让所有constraint尽量满足。这里也是一样——pose、contact point、force是variables，物理约束是factors。

四个factor的"人话"翻译：

**Factor 1 (Geometric Consistency)**："我手指摸到的那些点，必须落在物体表面上。"
- 公式：$SDF(o_r^{-1} P_i^T | \mathcal{M}_o) = 0$
- $P_i^T$ 是tactile sensor测到的点，$o_r^{-1}$ 把它变到object frame，然后算到mesh $\mathcal{M}_o$ 表面的有符号距离。理想情况下这个距离=0（点在表面）。

**Factor 2 (Non-Penetration)**："物体不能穿进环境里。"
- 公式：$S_i = \min(0, SDF(g_t \delta_t o_r P_i^{obj} | \mathcal{M}_e))$
- 在物体表面撒点 $P_i^{obj}$，变到world frame后查到环境mesh $\mathcal{M}_e$ 的SDF。如果点在环境内部（SDF负），就有penalty；外部就0。这是个one-sided constraint，像软墙。

**Factor 3 (Contact Kinematics)**："接触点必须同时在物体表面和环境表面。"
- 公式：$h_3 = [SDF(c_t | \mathcal{M}_e); SDF((g_t\delta_t o_r)^{-1}c_t | \mathcal{M}_o)]$
- $c_t$ 是contact point，左边约束它在环境表面，右边约束它变到object frame后在物体表面。两条约束同时=0，$c_t$ 就被钉在两个surface的intersection上。

**Factor 4 (Force Balance)**："我手上感受到的wrench，必须能用一个作用在contact point上的force解释。"
- 公式：$\hat{w}_t = J(g_t^{-1} c_t) f_t$，然后 $h_4 = \hat{w}_t - w_t$
- $J(p) = [I; [p]_\times]$ 是contact Jacobian，把3D force映射到6D wrench。物理上就是：在位置 $p$ 施加力 $f$，原点感受的wrench = $[f; p \times f]$。tactile sensor测了 $w_t$，你得找个 $c_t, f_t$ 让predicted wrench和observed吻合。

这四个factor各有侧重：Factor 1用几何，Factor 2用空间可行性，Factor 3用contact的几何约束，Factor 4用力的约束。它们通过shared variables（$o_r, c_t, f_t$）互相影响。比如Factor 4说force应该长这样，那contact point必须在能产生这个wrench的位置；Factor 3说contact必须在表面上，这又约束了pose；Factor 1说pose必须让tactile点落在物体表面。**约束越多，解空间越小，ambiguous问题变well-posed**。

## 一个关键reparameterization trick

公式(1)是整篇paper最优雅的地方：

$$o_t = g_t \cdot \delta_t \cdot o_r$$

- $g_t$: gripper在world的pose（robot告诉你）
- $\delta_t$: 物体在grasp内的micro-displacement（tactile sensor测）
- $o_r$: 物体相对gripper的"rest pose"（要估计的不变量）
- $o_t$: 物体在world的real pose

关键insight：GelSight是compliant的，外部force一压，物体在grasp里会微微滑动（$\delta_t$），但force撤掉又回去。这意味着物体在grasp里有个"回家位置" $o_r$，是grasp的内在属性，**时间不变**。

这个trick把"估计每帧的pose"（时变变量）变成"估计一个rest pose"（不变量）+ 用每帧的observation算displacement。在factor graph里，$o_r$ 是被所有time step共享的variable，所有factor都来constrain它，information accumulation非常efficient。这是为什么iSAM2能高效求解。

## 为什么需要particles

factor graph的optimization是non-linear least squares，会卡local minimum。tactile-only下，cylinder这种rotational symmetric物体，绕轴转任何角度tactile image都一样。你从某个initial guess开始优化，可能收敛到一个错的orientation。

所以撒K个initial particles（用ICP对齐到point cloud得到），每个独立跑完整iSAM2 inference，最后用cost $H(\cdot)$ 评分选最优。这是multi-start + local optimization的hybrid。

## 实验结果讲人话

**Pose estimation (Table I)**:
- 有vision时：TacGraph在所有object上~1mm精度，baselines在slanted rectangle这种难物体上会崩到12mm
- 无vision时（tactile-only）：ICP/CHSEL直接崩到17-30mm（没vision它们就废了），SCOPE靠particle filter能到4-18mm但high variance，**TacGraph稳定在1-8mm**

**Contact estimation (Table II)**:
- Tactile-only overall: TacGraph 3.72mm，SCOPE v2 10.40mm，其他更差
- 因为contact accuracy直接依赖pose accuracy，TacGraph的joint optimization让两者互相refine

**Peg insertion (Table IV)**:
- 这是最hardcore的test：tactile-only，3mm clearance，open-loop insertion
- TacGraph 23/40 (57.5%)，SCOPE v2只有3/40
- SCOPE v2用64 pose + 200 contact particles，但pure particle filter在高维空间sample efficiency太差；TacGraph用deterministic local opt + multi-start，efficiency高得多

## 为什么这个方法work

核心insight：**多模态约束融合**。tactile、vision、contact kinematics、force balance，每种信息单独看都ambiguous，但组合起来互相disambiguate。Factor graph是表达这种多源约束融合最自然的language——每个factor独立建模，通过shared variables耦合，iSAM2高效求解。

这跟SLAM用factor graph融合里程计+回环+IMU是一个philosophy。把problem formulation成graph，物理约束做factor，sensor observation做factor，想加什么约束就加什么factor，modularity极强。比起end-to-end learning，这种model-based approach更interpretable、sample efficient、debuggable，缺点是需要显式建模（mesh geometry、Jacobian、SDF），表达力不如neural implicit。

## 局限性和我自己的疑问

1. **Grasp is elastic假设**：如果真的slip了，$o_r$ 就变了，整个estimator失效。paper说可以加slip detection重新初始化，但这只是patch，没解决根本问题
2. **Single contact point**：现实接触往往是patch，用point contact Jacobian会有systematic bias
3. **Mesh geometry必须已知**：如果是novel object就废了，除非先reconstruct
4. **Local optimization**：依赖particle initialization，对highly symmetric object还是可能漏掉正确解
5. **Computational cost**：K个particles每个跑iSAM2，real-time性能存疑（paper没报latency）

## 给Karpathy的intuition总结

这篇paper给你的核心intuition应该是：**在contact-rich manipulation里，perception不该是"先估计pose再估计contact"的pipeline，而是把pose、contact、force当成coupled variables，用物理约束做regularizer，在一个probabilistic graph里jointly infer**。Factor graph是表达这种coupled inference的最佳工具之一，每个物理定律（kinematics、dynamics、non-penetration）天然就是一个factor。

如果你想往这个方向延伸，可以想象把这套framework扩展到：
- Multi-contact（加多个 $c_t, f_t$ pair）
- Deformable object（$o_r$ 变成deformation field）
- Active perception（用mutual information选action主动降uncertainty，参考 [RUMI](https://ieeexplore.ieee.org/document/10803998)）
- Online control loop（把estimator接进servoing）
- Tactile SLAM（environment也unknown，simultaneous localization + mapping + contact estimation）

每个extension都是把一个新的物理constraint或observation变成factor加进graph，framework是scalable的。这是model-based robotics perception的优雅之处。

**References**:
- TacGraph project: https://tacgraph.github.io
- GTSAM (iSAM2 implementation): https://github.com/borglab/gtsam
- Factor Graphs for Robot Perception (Dellaert & Kaess): https://www.nowpublishers.com/article/Details/ROB-003
- CHSEL (SDF + free-space + QD): https://roboticsproceedings.github.io/rss2023/paper
- SCOPE / MultiSCOPE (particle filter baseline): https://ieeexplore.ieee.org/document/9981182
- RUMI (active perception via mutual information): https://ieeexplore.ieee.org/document/10803998
- NeuralFeels (visuotactile in-hand tracking): https://www.science.org/doi/10.1126/scirobotics.adl0628
- Contact Particle Filter (CPF): https://ieeexplore.ieee.org/document/7759551

---

# TacGraph: 基于Factor Graph的Simultaneous In-Hand Pose与Extrinsic Contact Estimation 深度解析

## 1. 核心Intuition：为什么这个问题难

这篇paper要解决一个robotics里的经典ill-posed problem：当你用手爪(GelSight Mini)抓住一个物体，物体又跟环境发生extrinsic contact时，你手上传感器返回的local tactile signal根本不能唯一确定物体位姿和接触位置。

关键洞察在于**耦合性**：物体pose决定了哪些contact是可能的（如果物体在A位置，那contact只能在物体表面上对应的位置发生），反过来contact的位置又约束了pose（接触点必须在物体表面又在环境表面，这个intersection大大缩小pose空间）。再加上force balance的约束（触觉传感器感受到的wrench必须能由一个作用在contact point上的force解释），四个约束同时作用就把解空间大大缩小了。

这种**多模态约束融合**的思想，跟SLAM里factor graph处理里程计+回环检测的思想是一样的（参考 Dellaert & Kaess 的 [Factor Graphs for Robot Perception](https://www.nowpublishers.com/article/Details/ROB-003)）。论文用 GTSAM 实现 iSAM2 incremental solver。

---

## 2. 系统架构分解

整个pipeline分两大块：

### 2.1 Tactile Models（object-agnostic）

这是非常聪明的设计点——三个模型都是object-agnostic的，不需要对每个物体重新训练：

**Model A: Tactile Depth Prediction**
- 输入：单张GelSight tactile image $I_t^L$ 或 $I_t^R$，shape $\mathbb{R}^{H \times W \times 3}$
- 输出：depth map $D \in \mathbb{R}^{H \times W}$
- 后处理：threshold → de-project → tactile point cloud $P^T \in \mathbb{R}^{N_T \times 3}$
- 训练数据：用已知geometry的fixture，render depth作为supervision

**Model B: In-Hand Displacement**
- 输入：两张tactile images $I_t^L, I_t^R$（左右手指各一张）
- 输出：$\delta_t \in SE(3)$，物体在hand内的相对位移
- 物理意义：GelSight是compliant的，外部contact施加时物体会微微在grasp内滑动，这个sliding就是 $\delta_t$

**Model C: In-Hand Wrench**
- 输入：两张tactile images
- 输出：$w_t \in \mathbb{R}^6$，grasp处感受到的wrench [force(3) + torque(3)]
- 同时用来做contact detection：$b_t = \|w_t\|_\Sigma > \epsilon$

这三个model都是convolutional，用supervised learning训练，论文没详细给架构细节，但按照GelSight相关工作的惯例（参考 [GelSight original paper](https://ieeexplore.ieee.org/document/8238100) 和 [Tac2Pose](https://journals.sagepub.com/doi/10.1177/02783649231181066)），大概率是ResNet/UNet类型的encoder-decoder结构。

### 2.2 TacGraph Factor Graph Estimator

这是论文的真正核心。整体factor graph结构：

```
Variables: o_r (rest pose, 共享), {c_t, f_t}_{t=1..T}
Factors:
  h1: Geometric Consistency (only t=0, 连接 o_r 与 P^T + P^V)
  h2: Non-Penetration (每个t, 连接 o_r 与 g_t, δ_t)
  h3: Contact Kinematics (仅当 b_t=True, 连接 o_r, c_t)
  h4: Force Balance (仅当 b_t=True, 连接 c_t, f_t)
```

---

## 3. 关键数学：变量reparameterization

公式(1)是理解整个方法的关键：

$$o_t = g_t \cdot \delta_t \cdot o_r$$

其中各变量的frame定义：
- $g_t \in SE(3)$: gripper pose in world frame（来自robot proprioception）
- $\delta_t \in SE(3)$: in-hand displacement，在gripper frame下表达
- $o_r \in SE(3)$: rest pose，物体相对gripper的pose，是个**不变量**（grasp is elastic假设）
- $o_t \in SE(3)$: 物体在world frame下的真实pose at time t

**物理意义**：grasp is elastic意味着只要外部force撤掉，物体就回到grasp里的"rest position"。所以 $o_r$ 是grasp的内在属性，时间不变。物体的真实pose = gripper在world的pose × 物体相对gripper的rest pose × 物体在grasp内的微扰。

这个reparameterization非常关键，因为它把一个时变变量 $o_t$ 转换成了一个**全局共享的不变量 $o_r$** + 一组观测 $\delta_t$。这让factor graph变成了"local + shared"结构，iSAM2可以非常高效处理。

---

## 4. Factor Graph的MAP估计

公式(2)(3)给出MAP优化：

$$o_r^*, c_{1:T}^*, f_{1:T}^* = \arg\min_{o_r, c_{1:T}, f_{1:T}} H(o_r, c_{1:T}, f_{1:T})$$

$$H(\cdot) = \|h_1(o_r)\|^2_{\Sigma_1} + \sum_{t=1}^{T}\left\{ \|h_2(o_r)\|^2_{\Sigma_2} + \mathbf{1}[b_t]\left(\|h_3(o_r, c_t)\|^2_{\Sigma_3} + \|h_4(c_t, f_t)\|^2_{\Sigma_4}\right)\right\}$$

变量解读：
- $\|\cdot\|^2_{\Sigma_i}$: Mahalanobis distance，$\|x\|^2_\Sigma = x^T \Sigma^{-1} x$
- $\Sigma_i$: 各factor的covariance，empirically selected
- $\mathbf{1}[b_t]$: indicator function，只有当b_t=True（检测到contact）时才激活 h3, h4

这种Gaussian noise model假设下，MAP = nonlinear least squares，GTSAM的iSAM2正是为此设计（[iSAM2 paper](https://journals.sagepub.com/doi/10.1177/0278364911430039)）。

---

## 5. 四个Factor的深度解析

### Factor 1: Geometric Consistency (公式4-5)

$$h_1(o_r; P, \mathcal{M}_o) = S$$

$$S_i = SDF(o_r^{-1} P_i^T | \mathcal{M}_o)$$

变量：
- $P_i^T$: tactile point cloud中的第i个点（在gripper frame下）
- $o_r^{-1}$: 把点从gripper frame变换到object frame
- $\mathcal{M}_o$: 物体的triangle mesh
- $SDF$: signed distance function，点到表面的有符号距离

**直觉**：tactile sensor测到的surface points必须落在物体surface上，所以 $SDF \to 0$。SDF对pose的gradient给出了refinement方向，这正是ICP的连续可微版本。SDF + gradient用 [Bilinear SDF computation](https://github.com/UM-ARM-Lab/chsel) 的高效mesh-based方法计算。

注意：这个factor只在t=0时应用（公式里有 $P$ 但没有 t 下标，且作者在论文里明确说"at time t=0"）。这是因为grasp is elastic假设下，rest pose不变，只需在initial time约束几何一致性即可。

### Factor 2: Non-Penetration (公式6-7)

$$h_2(o_r | \mathcal{M}_o, \mathcal{M}_e, g_t, \delta_t) = S$$

$$S_i = \min(0, SDF(g_t \delta_t o_r P_i^{obj} | \mathcal{M}_e))$$

变量：
- $P_i^{obj} \in \mathbb{R}^{N_P \times 3}$: 在object surface上采样的一组点
- $g_t \delta_t o_r P_i^{obj}$: 把object上的点变换到world frame
- $\mathcal{M}_e$: environment mesh
- $\min(0, \cdot)$: 只有当点在environment内部时（SDF为负）才有penalty，外部时为0

**直觉**：物体不能穿透环境。这个factor是 one-sided penalty，即"软墙"——穿透才惩罚，不穿透则无约束。这跟CHSEL（[CHSEL paper](https://roboticsproceedings.github.io/rss2023/paper)）的free-space reasoning有思想上的相似，但TacGraph用的是sampled point cloud而非整个mesh。

### Factor 3: Contact Kinematics (公式8)

$$h_3(o_r, c_t; \mathcal{M}_e, \mathcal{M}_o) = \begin{bmatrix} SDF(c_t | \mathcal{M}_e) \\ SDF((g_t \delta_t o_r)^{-1} c_t | \mathcal{M}_o) \end{bmatrix}$$

变量：
- $c_t \in \mathbb{R}^3$: contact point in world frame
- 第一行：contact point必须在environment表面 → SDF to $\mathcal{M}_e$ = 0
- 第二行：contact point必须变换到object frame后也在object surface上 → SDF to $\mathcal{M}_o$ = 0

**直觉**：contact point是物体与环境的intersection，必须同时位于两个surface上。这两个约束把6维的 $c_t$（实际3D） + 6维的 $o_r$（SE(3)）联合约束到一个低维流形上。这是factor graph的威力——同时优化所有变量，让contact和pose相互约束。

### Factor 4: Force Balance (公式9-10)

$$\hat{w}_t = J(g_t^{-1} c_t) f_t$$

$$h_4(c_t, f_t; g_t) = \hat{w}_t - w_t$$

变量：
- $f_t \in \mathbb{R}^3$: contact force in world frame
- $c_t$: contact point in world frame
- $g_t^{-1} c_t$: contact point变换到gripper frame
- $J(\cdot) \in \mathbb{R}^{6 \times 3}$: contact Jacobian，把3D contact force映射到6D wrench at gripper
- $w_t \in \mathbb{R}^6$: tactile sensor预测的observed wrench

**Contact Jacobian的物理意义**：如果你在gripper frame内的点 $p = (p_x, p_y, p_z)$ 施加force $f = (f_x, f_y, f_z)$，那么在原点（gripper frame原点）感受到的wrench是 $[f; p \times f]$。所以：

$$J(p) = \begin{bmatrix} I_{3 \times 3} \\ [p]_\times \end{bmatrix}$$

其中 $[p]_\times$ 是skew-symmetric matrix:
$$[p]_\times = \begin{bmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix}$$

**直觉**：触觉感受到的wrench是contact force通过Jacobian转换的结果。给定 $c_t$ 和 $f_t$，可以预测gripper感受到的wrench，必须和tactile observation吻合。这个约束非常关键，因为tactile sensor直接观测wrench（force + torque），这个factor把contact force和location直接和sensor signal联系起来。

注意：$f_t$ 是个隐变量，没有直接的observation，完全通过 h4 因子被优化。同时h4与h3耦合——h3约束 $c_t$ 必须在surface上，h4进一步用wrench信息disambiguate到底surface上哪个点是真正的contact。

---

## 6. Inference: Particle + iSAM2的混合

公式(11)给出multi-hypothesis处理：

$$o_r^*, c_{1:T}^*, f_{1:T}^* = \arg\min_k H(o_r^k, c_{1:T}^k, f_{1:T}^k)$$

流程：
1. 用ICP把初始particles $\{o_r^1, ..., o_r^K\}$ 对齐到 available point cloud $P$
2. 对每个particle独立运行完整的iSAM2 inference → 得到K个完整解
3. 用 $e^{-H(\cdot)}$ 作为particle weight，选cost最小的那个

为什么需要particles？因为factor graph的non-linear least squares会陷入local minimum。tactile-only场景下，物体如果是rotational symmetric（比如cylinder），tactile observation可能完全无法区分不同的orientation。此时particles提供multi-start。

这跟SCOPE/MultiSCOPE的particle filter思路相似（[SCOPE paper](https://ieeexplore.ieee.org/document/9981182), [MultiSCOPE](https://roboticsproceedings.github.io/rss2023/paper)），但区别在于TacGraph只在初始化时用particles，每个particle内部是deterministic的global optimization（iSAM2），而SCOPE是纯particle filter，每步都maintain particle distribution。

**Quality Diversity的可能性**：论文在Discussion里提到可以引入 [CHSEL的Quality Diversity](https://github.com/UM-ARM-Lab/chsel) 思想来enforce particle diversity，避免particles坍缩到同一个local minimum。

---

## 7. 实验结果深度分析

### 7.1 Setup
- Robot: KUKA LBR iiwa Med R820
- Gripper: WSG-50 parallel jaw
- Tactile: GelSight Mini × 2
- Vision: Intel Realsense D435
- Segmentation: SAM (Segment Anything Model, [SAM paper](https://arxiv.org/abs/2304.02643))
- F/T ground truth: ATI Gamma × 2（仅用于训练和eval，不参与inference）

### 7.2 Pose Estimation结果（Table I）

关键观察：

**Vision+Tactile条件下**：
- TacGraph在所有6个object上都接近最优，Train object上0.68/1.48/1.34 mm，Test object上1.05/0.92/1.62 mm
- ICP受vision noise影响大，slanted rectangle上达到4.86 mm
- CHSEL在大多数object上很好（1-2 mm），但slanted rectangle上崩溃到12.28 mm——这是因为QD search在该object上陷入bad basin
- SCOPE（v1/v2）一直表现差（5-16 mm），因为particle filter在高维pose space下采样效率低

**Tactile-only条件下**：
- ICP完全崩溃（12-23 mm），因为没有vision约束，纯geometric alignment无法工作
- CHSEL更糟（13-30 mm），QD反而引入更多bad solutions
- SCOPE表现尚可（4-18 mm），但受限于particle filter的稀疏采样
- **TacGraph依然强势**：Train上2.96/1.29/8.45 mm，Test上7.58/0.78/1.54 mm

特别值得注意的是wrench这个object——tactile-only下TacGraph达到0.78 mm，比SCOPE(v2)的4.75 mm好6倍。这是因为wrench是non-symmetric的复杂shape，contact kinematics + force balance的联合约束特别有用。

quarter cylinder在tactile-only下TacGraph是8.45 mm，相对较差——作者解释是GelSight在sensor normal方向sensitivity差，导致depth prediction噪声大。这其实揭示了tactile sensor本身的物理限制。

### 7.3 Contact Point Estimation结果（Table II）

Tactile-only下overall：
- ICP: 17.53 mm
- CHSEL: 24.41 mm
- SCOPE v1: 16.20 mm
- SCOPE v2: 10.40 mm
- **TacGraph: 3.72 mm**

这个差距非常大——TacGraph几乎是SCOPE的1/3。原因：contact point的accuracy直接依赖于pose accuracy，而tactile-only下其他方法的pose都不准，contact自然也不准。TacGraph通过factor graph的joint optimization让pose和contact互相refine，形成正反馈。

### 7.4 Force Estimation结果（Table III）

所有方法都差不多（0.6-0.7 N），因为都用同一个tactile force model。TacGraph略好（0.61 N），因为contact location更准 → Jacobian更准 → force estimation更准。

### 7.5 Peg Insertion结果（Table IV）

最实际的test——tactile-only下做open-loop peg insertion，clearance只有3 mm：

| Method | Overall Success |
|--------|----------------|
| ICP | 11/40 |
| CHSEL | 14/40 |
| SCOPE v1 | 7/40 |
| SCOPE v2 | 3/40 |
| **TacGraph** | **23/40** |

23/40 = 57.5% 的成功率，对tactile-only 3mm-clearance insertion来说是很impressive的。SCOPE v2虽然有64 pose particles + 200 contact particles，但只有3/40成功——这印证了pure particle filter在高维空间下sample efficiency极差的问题。TacGraph的deterministic local optimization（iSAM2）配合multi-start particles，效率高得多。

---

## 8. 与Related Work的对比

### 8.1 vs. Tac2Pose / SIMPLE
[Tac2Pose](https://journals.sagepub.com/doi/10.1177/02783649231181066) 和 [SIMPLE](https://www.science.org/doi/10.1126/scirobotics.adi8808) 是object-specific tactile model，需要为每个object训练专门model。TacGraph的优势是object-agnostic tactile models + 用mesh geometry做后端inference，新object只需要mesh就能用。

### 8.2 vs. SCOPE / MultiSCOPE
[SCOPE](https://ieeexplore.ieee.org/document/9981182) 用F/T sensing + particle filter，假设environment上装F/T sensor。MultiSCOPE扩展到environment pose估计。两者都不考虑geometric consistency。TacGraph的改进：① 触觉替代F/T（更现实）；② 加入geometric consistency；③ 从particle filter换成factor graph（deterministic global opt）。

### 8.3 vs. CHSEL
[CHSEL](https://roboticsproceedings.github.io/rss2023/paper) 用SDF + free-space + Quality Diversity做pose estimation，但不考虑contact。TacGraph借用了SDF的思路，但加入了contact kinematics + force balance。

### 8.4 vs. NeuralFeels
[NeuralFeels (Suresh et al.)](https://www.science.org/doi/10.1126/scirobotics.adl0628) 用neural fields + visuotactile做in-hand pose estimation，但不显式处理extrinsic contact。NeuralFeels更多是reconstruction + tracking，TacGraph更聚焦在contact-rich interaction下。

### 8.5 vs. Vitascope
[Vitascope (Lee & Fazeli, 2025)](https://roboticsproceedings.github.io/rss2025) 用implicit representation同时估计pose和contact patch，是TacGraph的近期竞争者。Vitascope用neural implicit representation，TacGraph用显式factor graph + mesh，trade-off是expressiveness vs interpretability。

---

## 9. Limitations和潜在Extension

论文Discussion部分提到的limitations：

1. **Geometry assumption**：必须知道object和environment的mesh。可以用 [NeuralFeels-style reconstruction](https://www.science.org/doi/10.1126/scirobotics.adl0628) 或 [Tac2Pose的object model](https://journals.sagepub.com/doi/10.1177/02783649231181066) 来relax这个假设。

2. **No slip assumption**：grasp is elastic，但实际可能slip。可以引入 [slip detection](https://ieeexplore.ieee.org/document/7353481) 来re-initialize estimator。

3. **Single contact point**：现实接触往往是patch。可以扩展到multi-contact（参考 [Ota et al. 2024](https://ieeexplore.ieee.org/document/10611067) 的contact patch estimation）。

4. **Local optimization**：iSAM2有local minimum问题，依赖initialization。可以结合 [RUMI (Zhong et al.)](https://ieeexplore.ieee.org/document/10803998) 的mutual information action selection来主动drive down uncertainty。

5. **Online control loop**：目前是offline inference，没有closed-loop control。可以接入contact servoing（参考 [Van der Merwe et al. 2023](https://proceedings.mlr.press/v205/van-der-merwe23a.html)）做reactive control。

---

## 10. 我的Critical Thoughts

**Strengths**：
- 把物理约束（kinematics + dynamics）和sensor feedback统一到一个probabilistic framework，是经典的"model-based + learning" hybrid
- Factor graph + iSAM2是个非常mature的tool（GTSAM），工程上immediately deployable
- Object-agnostic tactile models是关键设计，避免了per-object training
- 实验baseline齐全，tactile-only setting很有挑战性

**Weaknesses**：
- Particles + iSAM2的K倍计算开销（论文没说K是多少，但应该至少5-10）
- 只测了简单的poke interaction，没测更复杂的multi-step manipulation
- SDF computation对mesh quality敏感，mesh得是watertight的
- 没讨论computational latency，对online control重要
- Tactile model training需要F/T ground truth，限制了dataset规模

**Open Questions for Intuition Building**：
1. 如果grasp is elastic假设被violate（slip真的发生），factor graph会怎么reactive？是不是只能re-initialize？
2. Contact Jacobian $J(g_t^{-1} c_t)$ 假设contact force是point force，如果实际是patch contact，这个Jacobian就不准——force balance factor会有systematic bias
3. 如果object是非rigid的（比如deformable object），整个factor graph需要redesign
4. 如果environment geometry unknown，需要simultaneous localization + mapping，整个问题变成tactile SLAM，复杂度大幅上升

---

## 11. 总结：Building the Intuition

这篇paper的beauty在于：**把tactile sensing的local信息、vision的global信息、contact的物理约束三者用factor graph统一成一个optimization problem**。

Factor graph的真正威力是**modularity**——你可以加任意多的factor，每个factor代表一种约束（geometric, kinematic, dynamic），它们各自独立但通过shared variables相互influence。这比"end-to-end learning everything"更interpret、更sample efficient、更debuggable。

如果你要做contact-rich manipulation的perception系统，这篇paper的framework非常值得借鉴。可以想象在此基础上加：slip factor、deformation factor、multi-contact factor、dynamic factor（acceleration/velocity），每个factor都是对物理reality的一个维度的约束，越多约束 → 解空间越小 → estimate越准。

---

**Reference Links**:
- Paper project page: https://tacgraph.github.io
- GTSAM: https://github.com/borglab/gtsam
- iSAM2 paper: https://journals.sagepub.com/doi/10.1177/0278364911430039
- Factor Graphs for Robot Perception: https://www.nowpublishers.com/article/Details/ROB-003
- GelSight: https://ieeexplore.ieee.org/document/8238100
- Tac2Pose: https://journals.sagepub.com/doi/10.1177/02783649231181066
- CHSEL: https://roboticsproceedings.github.io/rss2023/paper
- SCOPE: https://ieeexplore.ieee.org/document/9981182
- MultiSCOPE: https://roboticsproceedings.github.io/rss2023/paper
- NeuralFeels: https://www.science.org/doi/10.1126/scirobotics.adl0628
- SAM: https://arxiv.org/abs/2304.02643
- Contact Particle Filter: https://ieeexplore.ieee.org/document/7759551
- RUMI: https://ieeexplore.ieee.org/document/10803998
- Vitascope: https://roboticsproceedings.github.io/rss2025
