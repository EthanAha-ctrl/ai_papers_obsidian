---
source_pdf: Elastic Motion Policy.pdf
paper_sha256: ed3d4de1f05663e7cc38f553c40f910ba2020668d8e55d0b9b4e76ae158a402b
processed_at: '2026-08-04T03:03:58-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 EMP

## 一句话讲完

**教 robot 一个动作看一遍就会，然后物体挪位置了它还能跟着调整路径，整个过程中保证它不会发疯跑飞。**

---

## 用最直白的类比

想象你在教小朋友写字。传统做法有两种：

**做法 A（典型 BC）**：让小朋友临摹一万遍 "木"字。结果：他临摹得很好。但你让他换个位置写，或者手被人撞了一下，他就懵了——因为没人教过他在新情况下怎么写。

**做法 B（EMP）**：只让小朋友看一遍写"木"字，但你告诉他两个 key info：
- 笔从纸的左上角起笔
- 笔最终落到右下角收笔

然后你跟他说："以后不管纸放哪，你记住起笔点和收笔点跟着纸走就行，中间怎么运笔保持原来那种感觉"。

小朋友懂了：**起笔和收笔必须贴在纸上指定位置（hard constraint），中间运笔保持原来的局部形状（soft constraint）**。这其实就是 EMP 的核心 idea——Laplacian editing。

---

## 为什么这么简单的东西之前没人做

之前的 stable DS 方法（LPV-DS）已经能做到了"看一遍就学，且不会跑飞"。但有个 gap：学完之后 policy 就定死了，物体一挪就废。

那为什么不能直接"把整个 policy 平移旋转到新位置"？——因为很多 task 的 constraint 不是 rigid 的。

举个 paper 里的 Book Placing 例子：你要把书插进书架。原始 demo 里书从正面进，斜着插进槽里。现在你把书架转 90 度。如果只是 rigid transform 整个 policy，那"斜着插"这个动作可能让书撞书架侧面——因为 transform 之后的 policy 保证收笔在槽里，但中间路径的几何关系不一定是 task 想要的。

EMP 的解法：把 GMM 看成一根"橡皮筋上挂着铃铛"，两端铃铛强行贴到新位置，中间铃铛按原比例自适应分布。这就是 Laplacian editing 的本质。

**用一个画面想**：一根橡皮筋挂在两个钉子上，上面串着 5 个珠子，珠子间距固定。你把两个钉子挪到新位置，珠子会自动重新分布，保持局部间距不变。EMP 就是把 motion policy 的 GMM 当成这根橡皮筋。

---

## 技术细节讲清楚（给 Karpathy 看的）

### 为什么 DS 适合这个场景

DS 形式：$\dot{x} = f(x)$，policy 直接是 vector field。好处：
1. 任何 state $x$ 都有 defined action，没有 "OOD undefined" 问题
2. 可以用 Lyapunov 函数证明 convergence
3. 自然 compliant（一阶 dynamics 遇到外力会"让"一下再回来）

而 neural network policy $\pi(x)$ 不一定有这些性质——你不知道它在 OOD state 会输出什么。

### LPV-DS 公式再讲一遍变量

$$\dot{x} = \sum_{k=1}^{K} \gamma_k(x) \mathbf{A}_k (x - x^*)$$

- $K$：把 motion 分成几段（GMM 的 component 数）。简单 task 比如 pick-and-place 可能 3-4 个，复杂 handwriting 可能 10+
- $\gamma_k(x)$：当前位置属于第 k 段的"概率权重"，所有 $\gamma_k$ 加起来等于 1。离第 k 个 Gaussian center 越近，权重越大
- $\mathbf{A}_k$：第 k 段的 local linear dynamics。是 negative definite 的（保证 stability）
- $x^*$：attractor，最终要收敛到的目标

**直觉**：当前 state 决定"我现在在哪一段"，每一段有自己的 linear 规则（往目标拉），混合起来就是 nonlinear 规则。

### Lyapunov 稳定性的物理直觉

$$V(x) = (x - x^*)^T \mathbf{P} (x - x^*)$$

这是把"距离目标的远近"加权求和——一个椭圆碗。$\mathbf{P}$ 决定碗的形状：往哪个方向拉得紧（高 P 值方向），哪个方向松。

稳定条件：$\dot{V}(x) < 0$，意思是沿着任何轨迹，碗里的"能量"都单调减少，最终归零到 attractor。

**为什么 $\mathbf{P}$ 难学？** 因为它和 $\mathbf{A}_k$ 是耦合的：给定 $\mathbf{A}_k$，求能证明 stable 的 $\mathbf{P}$，这本身是非凸 feasibility problem。原始方法用 nonconvex optimization，慢得要死。

### EMP 的 convex 化魔法

$$\min_{\mathbf{P}} \sum_{i=1}^{N} \text{ReLU}(\dot{x}_i^T \mathbf{P} (x_i - x^*))$$

**直白解释**：对每个 data point，检查它是否违反 stability 条件 $\dot{V} < 0$。如果违反了（$\dot{x}_i^T \mathbf{P}(x_i - x^*) > 0$），用 ReLU 罚它；如果没违反，loss 为 0。

这跟 SVM 的 hinge loss 思路完全一样：
- SVM: $\text{ReLU}(1 - y \cdot wx)$，惩罚违反 margin 的点
- 这里: $\text{ReLU}(\dot{x}^T \mathbf{P}(x-x^*))$，惩罚违反 stability 的点

凸性来自：
1. ReLU 是凸的
2. $\dot{x}_i^T \mathbf{P}(x_i - x^*)$ 对 $\mathbf{P}$ 是 affine 的（线性加常数）
3. 凸函数的 affine 复合是凸的
4. 凸函数 sum 是凸的
5. 约束 $\mathbf{P} \succeq \epsilon I$ 是 PSD cone，凸的

所以可以 QP 高效求解，**2.62s → 0.09s**，30x 加速。这个加速才是 real-time 30Hz update 的关键 enabler。

### Laplacian Editing 的数学

Graph Laplacian $L$ 是 $D - A$，D 是 degree matrix（每个 node 的 degree），A 是 adjacency matrix。

Laplacian coordinates $\Delta = L \beta$，其中 $\beta$ 是 node 位置。$\Delta_i = \beta_i - \frac{1}{|N(i)|}\sum_{j \in N(i)} \beta_j$，即"node i 相对邻居平均的偏离"。

EMP 的优化：

$$\min_\beta \|L\beta - \Delta\|^2 \quad \text{s.t. endpoints fixed}$$

意思是：新 graph 的 Laplacian coordinates 和原来一样（保留局部细节），但首尾 node 位置被 hard constraint 固定到新 keypose。

这个 closed-form 解是：

$$\beta^* = (L^T L)^{-1} (L^T \Delta + \text{constraint terms})$$

实际中用 sparse linear system 求解，毫秒级。

**为什么这聪明？** 因为它把"复杂 nonlinear policy 的 adaptation"降维成了"sparse linear system 求解"。NN policy adaptation 需要 retraining 或 fine-tuning，EMP 是 closed-form。

---

## 实验结果人话版

| Task | Baseline OOD | EMP OOD |
|---|---|---|
| Book Placing | 4/10 | 10/10 |
| Cube Pouring | 4/10 | 9/10 |
| Pick-and-Place | 1/10 | 7/10 |

Baseline 就是 SE(3)-LPVDS 加 rigid transform。ID 场景两边都满分（因为本来就在 demo 位置）。OOD 场景 EMP 完胜。

**Cube Pouring 的 9/10 失败很有意思**：不是 EMP 的错，是 robot 的 reachability 限制——path 被 morph 出来后超出 robot 工作空间。这是 task space adaptation 的 inherent limitation，不是 method 本身问题。Future work 可以加 workspace constraint 到 Laplacian editing 优化里。

**Pick-and-Place 双低的原因**：用了 UVD 自动分割长 horizon video，分割点不准会传播错误。比如把"放下"动作切早了，keypose 落在 box 外面，robot 就不会把物体放进 box。这暴露了 multi-step decomposition 的 brittle 性。

---

## Empathy 一下作者的思路

我猜作者 Tianyu Li 当时的思考路径大概是这样：

1. 先做 stable DS（LPV-DS），发现 generalize 不行
2. 想做 task-parameterized GMM（Calinon 的工作），但太 rigid
3. 灵感：computer graphics 里有 Laplacian mesh editing，能 elastic 变形保局部结构 → 能不能用在 GMM 上？这就是 Elastic-DS
4. 但 Elastic-DS 只能 position，慢得要命（分钟级）
5. 需要加 orientation → quaternion manifold 上做 tangent plane projection
6. 需要加速 → convex 化 P-QLF learning
7. 需要全自动化 perception pipeline → LLM + SAM + FoundationPose
8. 需要长 horizon → UVD 自动分段

每一步都是 reasonable extension，最后堆出 1-shot → real-time adaptable → stable 的完整系统。这是 incremental 但 cumulative 的工作，工程价值高。

---

## 让我兴奋的点 vs 让我皱眉的点

### 让我兴奋

1. **Reproducing kernel 的思路本质**：GMM + Laplacian 等价于 implicit 定义了一个 kernel function，"在 demonstration 流形上做插值"。这跟 KBR、GMR 的内核思想一致，但更 generalizable
2. **Convex 化的 elegance**：ReLU + affine = convex，这个 move 看着简单但很多 control 论文没注意到
3. **DS 的 modularity**：DS 可以自然 combine obstacle avoidance (modulation)、multi-step (activation function)，因为它是 vector field 不是 blackbox policy。Neural policy 想加这些得重训
4. **Mathematical 结构的美感**：Laplacian editing 既在 graphics 验证过、又能在 motion policy 上用、还能 extend 到 manifold——这是 cross-domain 的数学工具迁移

### 让我皱眉

1. **Single attractor limitation**：QLF 证明稳定本质要求 single attractor。但很多 task 是 cyclic（搅拌、拧螺丝），EMP 框架直接无法表达。Paper future work 提 contraction theory 是正路
2. **Single tangent space approximation**：quaternion 大 rotation 时 distort。Paper 自己承认了 (citing Jaquier 2024)。Lie group formulation 应该是 next step
3. **Perception stack 的 fragility**：GPT-4o + Grounded SAM + FoundationPose，三层串联每层都可能 fail。Paper 没量化 perception 失败率
4. **Task complexity 天花板**：linear mixture + quadratic Lyapunov 表达力有限。复杂 choreography（比如跳舞、操作柔性物体）大概率 encode 不进去
5. **10 trials 的统计意义**：每个 task 只 10 trials，OOD 4/10 vs 9/10 这种差距，统计上未必显著。需要更多 trials 或 cross-validation
6. **跟 diffusion policy 比吗？** Paper 没直接比。Diffusion policy 用大数据表达高 multi-modality，EMP 用极少数据表达 single stable mode。两者解决不同问题，但 paper framing 上"1-shot"容易让人错觉它替代了 diffusion policy

---

## 横向对比一下其他 paradigm

### vs Diffusion Policy

Diffusion policy 是 distribution-level policy：学 $p(a|x)$ 的多模态分布。EMP 是 deterministic DS：$\dot{x} = f(x)$，单模态稳定收敛。

**Diffusion 强**：多模态（同一 state 可以多个合理 action）、表达力强、大数据学复杂 task
**Diffusion 弱**：没 stability guarantee、OOD 行为不可预测、retrain 才能适应新场景

**EMP 强**：1-shot、stable、real-time adapt、closed-form adaptation
**EMP 弱**：单模态、表达力有限、依赖 GMM 假设（local linearity）

**互补而非替代**。想象 future：Diffusion policy 在大数据上学 task class，EMP 在 1-shot 上做 task instance adaptation。

### vs RMP (Riemannian Motion Policy)

RMP 是 tree of policies，每个 leaf 是一个 task（如 attractor、avoid obstacle），组合成整体 policy。

**RMP 强**：modular、composable、显式 control 每个行为
**RMP 弱**：每个 sub-policy 仍需手动 design 或 learn，组合后 stability 分析复杂

**EMP 强**：从 demonstration 自动 extract structure，adaptation 是 closed-form
**EMP 弱**：policy 不能自由组合（除了 obstacle modulation）

### vs Task-Parameterized GMM (Calinon)

TP-GMM 把每个 demo 在多个 frame（object、world、robot）中表达，组合出适应新 frame 的 policy。

**TP-GMM 强**：principled、多 frame、可以多 demo
**TP-GMM 弱**：rigid transformation，没有 stability guarantee

**EMP 强**：elastic transformation（非 rigid）、stability guarantee、real-time
**EMP 弱**：single frame（attractor frame）、需要 SDP re-learn

### vs LfD on Riemannian Manifolds (Zeestraten et al.)

Zeestraten 的工作在 manifold 上做 GMM + GMR，处理 quaternion 但 rigid。

**EMP 区别**：在 manifold tangent 上做 Laplacian editing（非 rigid），加 stability 保证。

---

## 我会怎么 extend 这个工作

如果让我继续做 EMP，会从这几个方向切：

### 1. Contraction Theory + LPV-DS

Lohmiller & Slotine 1998 的 contraction theory 证明：如果 Jacobian 的 symmetric part uniformly negative definite，则所有轨迹互相收敛。这比 Lyapunov 更宽松——不要求 single attractor，可以是 limit cycle、multi-stable 等。

把 LPV-DS 的 $\mathbf{A}_k$ 约束从 Lyapunov equation 改成 contraction condition，能 encode cyclic task。

Paper: https://ieeexplore.ieee.org/document/661067

### 2. Neural Laplacian Editing

Laplacian editing 用 hand-crafted graph。如果 graph structure 也 learnable，比如用 GNN 学习 task-aware graph，可以做更 complex 的 adaptation。Inspired by Neural ODE (Nawaz et al. ICRA 2024)。

Paper: https://arxiv.org/abs/2406.00137 (Nawaz et al.)

### 3. Self-Supervised Object Keypose

当前 perception stack 是 GPT-4o + SAM + FoundationPose。可以让 robot 自己 explore，对每个新物体自动发现"task-relevant keypose"——通过 affordance prediction 或者 interactive perception。

Affordance 相关：https://github.com/UT-Austin-RPL/Where2Act

### 4. Adaptive Graph Topology

当前 GMM structure (K) 在 demonstration 后固定。如果 task 复杂度变化（比如原来简单 task 现在加 obstacle），应该能 dynamic add/remove Gaussians。Non-parametric Bayesian 类似 Dirichlet Process GMM 可以做到。

### 5. Multi-robot / Bimanual Extension

EMP 目前 single arm。Bimanual task 有 relative pose constraint（两 hand 协同），可以扩展 Laplacian editing 到 multi-agent graph。

### 6. Vision-Conditioned Lyapunov

当前 $\mathbf{P}$ 在 task space 上 constant。如果 $\mathbf{P}$ 是 vision feature 的 function（如 CNN 输出），可以在不同 visual context 下自动选择不同 stability profile。

类似 Visual RL with stability 的方向。

### 7. Cross-Embodiment Transfer

UMI gripper demo → Franka robot execution。如果用不同 morphology robot（比如 UR5 vs Franka），EMP 框架应该直接 work（因为都在 end-effector space）。这个 paper 没测，但理论上很 natural。Google 的 RT-X 在做类似事情。

RT-X: https://robotics-transformer-x.github.io/

---

## 最终一句话直觉

**EMP 把"motion policy adaptation"问题，从"重新学习"降维到"几何变形 + 凸优化"问题**。

这一降维的代价是：表达力受限（只能做 stable reaching motion）、需要 perception 提供物体 pose、假设 task 可以用 keypose 概括。

收益是：1-shot、real-time、stable、compliant、composable with obstacle avoidance。

这种 "drop the constraint, keep the structure" 的设计哲学，在 system design 里反复出现。比如 ResNet drop 了"层必须直接学 target"的约束，保留 forward structure，结果表达力暴涨。EMP 也类似：drop 了"policy 必须直接模仿 demo"的约束，保留 GMM structure 和 stability，得到 adaptation 能力。

这种 trade-off 思路是我看完这篇 paper 最 takeaway 的东西。

---

## 给 Karpathy 的 TL;DR

- **核心**：1-shot DS imitation learning + Laplacian editing for elastic adaptation + convex Lyapunov optimization for real-time update
- **Smart move 1**：把 GMM 当 graph，用 graph Laplacian 做 elastic deformation，preserving local geometry while fixing endpoints to new object pose
- **Smart move 2**：把 Lyapunov matrix learning 从 nonconvex 转成 convex via ReLU hinge loss，30x speedup enabling 30Hz
- **Smart move 3**：Quaternion 通过 attractor tangent space projection 降到 3D 做欧式 Laplacian editing，再升回 quaternion manifold
- **Limitation**：single attractor, single tangent plane approx, linear mixture expressiveness ceiling, perception stack fragility
- **Real value**：shows that for the right task class (stable reaching with adaptation), simple mathematical structure + convex optimization beats deep learning in data efficiency and adaptation speed

这 paper 适合作为 "Riemannian geometry + control theory + imitation learning" 的入门 reading，因为它把这些技术 concrete 在一个 working system 里。读 paper 之前建议先看 Billard 团队的 LPV-DS 原始 paper 和 Elastic-DS 原始 paper，会更顺。

最深入的相关 reading：
- LPV-DS (CoRL 2018): https://proceedings.mlr.press/v87/figueroa18a.html
- Elastic-DS (CoRL 2023): https://proceedings.mlr.press/v229/li23b.html
- SE(3)-LPVDS (IROS 2024): https://arxiv.org/abs/2406.16824

---

# Elastic Motion Policy (EMP) 深度讲解

## 1. 论文核心思想

这篇 paper 解决的核心问题是：**如何用极少（1-2 个）demonstration 让 robot 学到稳定、compliant、reactive 且能适应环境变化的 motion policy**。

传统 Behavior Cloning (BC) 的痛点在于：
- 即使收集大量数据，OOD performance 仍然差
- 没有 convergence guarantee
- 物理交互（比如人推一下 robot）会让 state 跑到 OOD，行为不可预测
- 环境/物体位置变化后，learned policy 失效

EMP 的核心 insight：**motion policy 应该 extract task 信息并基于 scene change 调整，单纯 mimic demonstration 是不够的**。这其实是把 motion representation 分解成 "task-relevant invariants" + "geometric constraints" 的思想，类似 task-parameterized GMM 但更具弹性。

项目主页：https://elastic-motion-policy.github.io/EMP/

---

## 2. 背景技术：SE(3) LPV-DS

EMP 建立在 SE(3) LPV-DS（Linear Parameter Varying Dynamical System）之上。需要先理解这块才能 build intuition。

### 2.1 LPV-DS（位置部分）

LPV-DS 把 nonlinear DS 表示成多个 stable LTI（Linear Time-Invariant）系统的加权混合：

$$\dot{x} = \sum_{k=1}^{K} \gamma_k(x) \mathbf{A}_k (x - x^*) \tag{1}$$

变量含义：
- $x, \dot{x} \in \mathbb{R}^m$：robot state 和 velocity（m 是维度，对 end-effector 位置 m=3）
- $x^*$：attractor（目标点）
- $K$：LTI 系统总数（等价于 GMM 的 component 数）
- $\gamma_k(x)$：state-dependent mixing function（来自 GMM 的 posterior responsibility）
- $\mathbf{A}_k$：第 k 个 LTI 系统的 matrix，需要学习

$\gamma_k(x)$ 的参数 $\Theta_\gamma = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ 通过 GMM/DAMM 拟合 reference trajectory 得到。

**为什么这样做？** 一个 nonlinear DS 可以 locally 看作多个 linear DS 的拼接，每个 Gaussian region 内部行为是 linear 的，整个组合起来就是 nonlinear 的。GMM natural 地提供了 "局部 linear" 的 segmentation。

### 2.2 Stability Guarantee via P-QLF

每个 $\mathbf{A}_k$ 通过 SDP（Semi-Definite Program）学习，约束保证 Globally Asymptotic Stability (GAS)：

$$\min_{\Theta_{DS}} J(\Theta_{DS}) = \sum_{i=1}^{N} \|\dot{x}_i^{ref} - f(x_i^{ref})\|_2^2$$
$$\text{s.t. } (\mathbf{A}_k)^T \mathbf{P} + \mathbf{P}\mathbf{A}_k = \mathbf{Q}_k, \quad \mathbf{Q}_k = (\mathbf{Q}_k)^T \prec 0, \quad \forall k \tag{3}$$

Lyapunov function 形式：

$$V(x) = (x - x^*)^T \mathbf{P} (x - x^*) \tag{2}$$

变量含义：
- $\mathbf{P} = \mathbf{P}^T \succ 0$：正定矩阵，定义 Lyapunov function 的 elliptical 等高线形状
- $\mathbf{Q}_k \prec 0$：负定矩阵，保证每个 LTI 系统在 V 上递减

**直觉**：$V(x)$ 是一个碗状能量函数，attractor $x^*$ 是碗底。Lyapunov 条件 $\dot{V} < 0$ 意味着沿着任何轨迹"能量"都递减，最终落到碗底。$\mathbf{P}$ 决定碗的形状（各方向陡峭程度），$\mathbf{Q}_k$ 决定每个 LTI 区域内的下降速率。

约束 $(\mathbf{A}_k)^T \mathbf{P} + \mathbf{P}\mathbf{A}_k = \mathbf{Q}_k \prec 0$ 是 continuous-time Lyapunov equation 的负定版本——保证 $\mathbf{A}_k$ 是 stable matrix（所有特征值实部为负）。

---

### 2.3 Quaternion-DS（旋转部分）

旋转处理在 quaternion manifold $S^3$ 的 tangent plane 上做。

$$ (\hat{\mathbf{q}}_{att})^{des} = \sum_{k=1}^{K} \gamma_k(\mathbf{q}) \mathbf{A}_k \log_{\mathbf{q}_{att}} \mathbf{q} \tag{4}$$

变量含义：
- $\mathbf{q}, \mathbf{q}_{att}$：当前 orientation 和 target orientation（单位 quaternion）
- $\log_{\mathbf{q}_{att}} \mathbf{q}$：Riemannian logarithmic map，把 quaternion 投影到 attractor 处的 tangent plane，得到 3D 偏差向量
- $(\hat{\mathbf{q}}_{att})^{des}$：下一个 desired orientation（在 tangent plane 表示）
- $\mathbf{A}_k \prec 0$：约束为负定，保证 stability

**为什么用 tangent plane？** Quaternion 空间 $S^3$ 是 4 维单位球面，不是欧式空间。Riemannian geometry 工具（log/exp map）允许在局部 tangent space 上做欧式运算，再映射回 manifold。attractor 处的 tangent plane 是 "flat approximation"。

Recovering angular velocity 需要 parallel transport + exponential map（Appendix A）：
- Parallel transport $\Gamma_{\mathbf{q}_{att} \to \mathbf{q}}$ 把 attractor 处 tangent vector 搬到当前 state 处
- $\exp_{\mathbf{q}}$ 把 tangent vector 映射回 manifold 得到 desired quaternion
- $\omega = (\bar{\mathbf{q}} \circ (\hat{\mathbf{q}})^{des}) / dt$ 通过 quaternion 乘法算角速度

---

## 3. EMP 核心方法：SE3 LPV-DS Policy Morphing

### 3.1 Laplacian Editing 的直觉

这是 EMP 的灵魂。**把 GMM 看作一个 graph，nodes 是 Gaussians，通过约束 endpoints 让中间 nodes 弹性 morph**。这与计算机图形学里的 Laplacian mesh deformation 一脉相承。

定义 "joints" $\beta_{i,k,k+1}$ 为相邻两个 Gaussians 之间的"中间点"，通过 product of two Gaussians 的 normalization 计算：

$$\Sigma_t = (\Sigma_k^{-1} + \Sigma_{k+1}^{-1})^{-1} \tag{7}$$
$$\beta_{i,k,k+1} = \Sigma_t (\Sigma_k^{-1} \mu_k + \Sigma_{k+1}^{-1} \mu_{k+1}) \tag{8}$$

变量含义：
- $\Sigma_t$：两个相邻 Gaussians 的 product 归一化后的 covariance
- $\beta_{i,k,k+1}$：两个 Gaussians 的"信息融合中点"（精度加权平均）

**直觉**：两个 Gaussian 的 product（不是 mixture）是一个新的未归一化 Gaussian，其 mean 就是这个精度加权平均，covariance 是精度之和的逆。$\beta$ 就是 Gaussians 之间的"关节"。

### 3.2 Laplacian Editing 优化

$$\min_{\beta_i} J(\beta_i) = \|L\beta_i - \Delta\|_2^2 \tag{9}$$
$$\text{s.t. } T_{0,1}(\beta_{i,0}, \beta_{i,1}) = O_{start}, \quad T_{n-1,n}(\beta_{i,n-1}, \beta_{i,n}) = O_{end}$$

变量含义：
- $L$：graph Laplacian matrix（图论中的 $L = D - A$，D 是 degree matrix，A 是 adjacency matrix）
- $\Delta$：original Laplacian coordinates（每个 node 与邻居平均位置的差）
- $T_{0,1}, T_{n-1,n}$：endpoints 处的 homogeneous transformation 约束
- $O_{start}, O_{end}$：从相关物体 pose 得到的 new geometric constraints

**直觉**：$L\beta$ 是当前 graph 的 Laplacian coordinates（局部几何细节），$\Delta$ 是原始的。最小化 $\|L\beta - \Delta\|^2$ 意味着"保持局部几何关系不变"——这就是"elastic" 的来源，像橡皮筋一样两端固定后中间自然分布。endpoints 约束把首尾贴到新的物体 pose 上。

这个 idea 跟 mesh editing 中 Taubin/Laplacian surface editing、Sorkine 的 laplacian coordinates 是一回事，只不过用在了 motion policy 的 GMM 上。

参考：Elastic-DS 原始 paper (Li & Figueroa, CoRL 2023)
https://proceedings.mlr.press/v229/li23b.html

### 3.3 Orientation Space 上的 Laplacian Editing

这里有个关键的技术挑战：quaternion GMM 的 mean 和 covariance 都在 tangent space 中定义，不能直接做欧式 Laplacian editing。

EMP 的解法：**通过 attractor 处的 null space，把 4D quaternion 表示降到 3D 欧式空间，做完 editing 再升回 4D**。

**Step 1**: 找 attractor quaternion $\mathbf{q}_{att}$ 的 null space（perpendicular hyperplane）：

$$\text{Null}(\mathbf{q}_{att}) = \{\mathbf{q}_i \in \mathbb{R}^4 \mid \mathbf{q}_i \cdot \mathbf{q}_{att} = 0\} \tag{10}$$

构造 basis $\Lambda_{\mathbf{q}_{att}} = [\mathbf{q}_1 \mathbf{q}_2 \mathbf{q}_3] \in \mathbb{R}^{4 \times 3}$。

**Step 2**: 把每个 GMM mean $\tilde{\mu}_k$ 投影到 3D：

$$\hat{\mu}_k = \Lambda_{\mathbf{q}_{att}}^T \log_{\mathbf{q}_{att}}(\tilde{\mu}_k) \in \mathbb{R}^3 \tag{12}$$

**Step 3**: 把 covariance 也投影到 3D。这里 tricky：每个 covariance 是相对于自己的 mean 定义的，所以每个 Gaussian 需要自己的 basis：

$$\Lambda_{\mu_k} = [\mathbf{q}_1 \mathbf{q}_2 \mathbf{q}_3], \quad \mathbf{q}_i \in \text{Null}(\tilde{\mu}_k) \tag{13}$$
$$\hat{\Sigma}_k = \Lambda_{\mu_k}^T \tilde{\Sigma}_k \Lambda_{\mu_k} \in \mathbb{R}^{3 \times 3} \tag{14}$$

**Step 4**: 在 3D 空间做 Laplacian editing（同 Eq. 9）。

**Step 5**: 反向恢复：

$$\tilde{\mu}_k^* = \exp_{\mathbf{q}_{att}}(\Lambda_{\mathbf{q}_{att}} \hat{\mu}_k^*) \in \mathbb{H} \subset \mathbb{R}^4 \tag{15}$$
$$\tilde{\Sigma}_k^* = \Lambda_{\mu_k} \hat{\Sigma}_k^* \Lambda_{\mu_k}^T \in \mathbb{R}^{4 \times 4} \tag{16}$$

**为什么用 attractor 处的单一 tangent space？** 这是计算 trade-off。理论上每个 quaternion 应该在它自己的 tangent space 上处理（更精确），但这样无法做统一的 Laplacian editing。统一到 attractor tangent space 会有 distortion，但 paper 实验中观察到 conservative 的 QLF 约束让方法保持稳定。Paper 也指出 limitation：对于大 rotation 可能需要 Lie group formulation。

参考 single tangent space fallacy discussion：
https://arxiv.org/abs/2310.07902

---

## 4. 凸优化加速 Lyapunov Learning

### 4.1 传统方法的痛点

原 LPV-DS 中 $\mathbf{P}$ 矩阵通过 nonconvex optimization 学习（Khansari-Zadeh & Billard 2014 的方法）。这导致：
- 慢（更新一次要几秒到几分钟）
- 无法 real-time 适应

EMP 的关键贡献：**把 P-QLF learning 重新公式化为 convex optimization**。

### 4.2 Lyapunov Stability Condition

稳定性条件：沿着轨迹 V 递减

$$\dot{V}(x) = \dot{x} \cdot \nabla V = 2\dot{x}^T \mathbf{P} (x - x^*) < 0, \quad \forall x \neq x^* \tag{17}$$

变量含义：
- $\dot{V}(x)$：Lyapunov function 沿轨迹的导数
- $\dot{x}$：当前 velocity
- $x - x^*$：相对 attractor 的 displacement
- $\mathbf{P}$：要学习的 Lyapunov matrix

### 4.3 Convex Formulation

$$\min_{\mathbf{P}} \sum_{i=1}^{N} \text{ReLU}(\dot{x}_i^T \mathbf{P} (x_i - x^*)) \tag{18}$$
$$\text{s.t. } \mathbf{P} \succeq \epsilon I$$

变量含义：
- $\text{ReLU}(\cdot)$：max(0, ·)，只惩罚违反 stability 的点
- $\epsilon > 0$：small value 保证 $\mathbf{P}$ 严格正定
- $I$：单位矩阵

**为什么是凸的？** ReLU 是凸函数。$\dot{x}_i^T \mathbf{P} (x_i - x^*)$ 对 $\mathbf{P}$ 是 affine 的。凸函数的 affine 复合是凸的。凸函数求和是凸的。约束 $\mathbf{P} \succeq \epsilon I$ 是 PSD cone，凸的。所以整个问题凸，可以用 QP 高效求解。

**直觉**：原始 P-QLF 同时学 $\mathbf{P}$ 和 $\mathbf{A}_k$，coupling 导致非凸。这里 fixed 住一个，单独学另一个，且只惩罚 violation 而不是追求严格负，使得问题简化为分类式的"hinge loss"问题——就像 SVM 的 hinge loss 思路，让违反约束的点 contribute loss，没违反的不算。

参考 Boyd & Vandenberghe Convex Optimization：
https://web.stanford.edu/~boyd/cvxbook/

### 4.4 GMM-informed 加速

进一步加速：用每个 Gaussian 的平均 position 和 velocity 代替所有 data points。因为每个 Gaussian 代表"相似特性的局部区域"，平均后保留主要信息。

| Methods | Single Traj Time | Single Traj Violation | All Traj Time | All Traj Violation |
|---|---|---|---|---|
| Baseline P-QLF | 0.332s | 14.0% | 2.62s | 14.9% |
| Convex P-QLF | 0.038s | 11.1% | 0.24s | 12.3% |
| GMM P-QLF | 0.007s | 15.1% | 0.09s | 15.4% |

**关键 take-away**：GMM P-QLF 在 all trajectory 上从 2.62s 降到 0.09s，**约 30x 加速**（paper 中说 50x，看具体 baseline 配置），violation 从 14.9% 到 15.4%，几乎无 loss。这让 30Hz online update 成为可能。

---

## 5. EMP 完整 Pipeline

### 5.1 Data Collection

- 用 UMI gripper（Universal Manipulation Interface, Chi et al. RSS 2024）做 in-the-wild demonstration
- 外部 RGBD 相机记录
- AprilTags 多面贴在 UMI cube 上避免 occlusion 问题
- Microcontroller-based contact sensor 记录 binary gripper state
- AprilTag cube 作为伪 robot base frame

UMI paper: https://universal-manipulation-interface.github.io/

### 5.2 Keypose/Attractor Extraction

这是 perception pipeline 的核心：

1. **GPT-4o 识别 semantic label**：输入 demonstration 的 first/middle/last frame，让 GPT-4o 输出 short phrase（如 "yellow mustard bottle on black stand"）
2. **Grounded SAM 生成 mask**：用 phrase 作为 prompt
3. **FoundationPose 估计 6D pose**：基于 mask 和 3D mesh

最后把 demonstration 中 end-effector 的 last pose 在 object frame 中记录为 keypose $O_{key,obj}$。当 object 移动时，keypose 跟随移动，作为 Laplacian editing 的 endpoint constraint。

**为什么用 first/middle/last 三帧？** 简化 inference，降低 latency，同时给 LLM 足够 context 理解整个 motion。

FoundationPose: https://nvlabs.github.io/FoundationPose/
Grounded SAM: https://github.com/IDEA-Research/Grounded-SAM

### 5.3 Multi-step Decomposition

用 UVD (Universal Visual Decompose, Zhang et al. ICRA 2024) 自动分解长 horizon video。每个 segment 单独训练一个 stable DS motion policy。

Multi-step DS stitching:

$$\dot{x} = \sum_k \sum_j \delta(\xi) f_{kj}(x)$$

变量含义：
- $f_{kj}(x)$：每个 single stable DS policy
- $\delta(\xi)$：one-hot activation function，当前 DS 达到 attractor 时激活下一个
- $k$：task subgoal level index
- $j$：subgoal 内部的 DS index

UVD: https://universal-visual-decomposer.github.io/

### 5.4 Runtime Inference

1. Track object pose $O_{obj}$（FoundationPose online）
2. Compute keypose $O_{key}$ in world frame
3. Update EMP policy via Laplacian editing (Eq. 9) using new keypose as endpoint constraint
4. Re-optimize $\mathbf{P}$ via convex QP
5. Re-learn $\mathbf{A}_k$ via SDP（这个其实是最慢的环节）
6. Output $\dot{x}, \omega$ 给 passive impedance controller（Kronander & Billard 2015）

整个 update loop 在 30Hz 运行。

---

## 6. 实验：与 Baseline 对比

### 6.1 Baseline：SE(3)-LPVDS with Global Transform

Baseline 用 object-centric 版本的 SE(3)-LPVDS：learned DS policy 做全局刚体变换（rotation + translation）跟随 object keypose。

| Task | Method | ID | OOD |
|---|---|---|---|
| Book Placing | Baseline | 10/10 | 4/10 |
| Book Placing | EMP | 10/10 | 10/10 |
| Cube Pouring | Baseline | 10/10 | 4/10 |
| Cube Pouring | EMP | 10/10 | 9/10 |
| Pick-and-Place | Baseline | 8/10 | 1/10 |
| Pick-and-Place | EMP | 7/10 | 7/10 |

### 6.2 关键 task 分析

**Book Placing**：需要两个 constraints 同时满足——approaching angle（避免撞书架）和 placement pose。全局刚体变换下，变换后一个 constraint 满足可能另一个不满足。EMP 通过 Laplacian editing 严格保证 endpoints 同时满足，中间 morph 保持局部几何关系，成功率高 100% OOD。

**Cube Pouring**：arch-shaped trajectory + wrist rotation。OOD 场景中，从另一侧开始时 baseline 走到未知区域，可能画大圆或撞桌子。EMP 因 Laplacian editing 的 local geometric preservation，path 形状保持。失败案例来自 robot workspace 限制（adapted path 超出 reachability）。

**Pick-and-Place (multi-step)**：UVD 分解 + sequential DS。两边都有失败，因为 segmentation 误差会传递——如果 placing motion 切早了，keypose 落到 box 外面，导致收敛错误。

### 6.3 Obstacle Avoidance

用 DS modulation（Khansari-Zadeh & Billard 2012）：

$$\dot{x}_{new} = \mathbf{M}(x) f(x)$$

变量含义：
- $f(x)$：原 DS velocity field
- $\mathbf{M}(x)$：modulation matrix，通过 eigenvalue decomposition 构造，根据 obstacle 边界的 normal/tangent 方向
- $\dot{x}_{new}$：modulated velocity，绕过 obstacle

**直觉**：M(x) 在 obstacle 附近"重塑" velocity field，让 flow 沿 obstacle 表面 tangent 方向滑过，远离 obstacle 后 M 渐近 identity 不影响原 DS。这跟控制障碍函数 (CBF) 的思路类似，但更几何化。

DS Obstacle Avoidance 原始 paper: https://link.springer.com/article/10.1007/s10514-012-9294-y

---

## 7. 直觉总结与相关联想

### 7.1 EMP 的本质

EMP 本质上把"motion policy learning"分成了三层 abstraction：

1. **Structure layer** (GMM/K)：任务复杂度，从 demonstration 一次拟合后固定
2. **Geometric layer** (means/covariances/joints)：跟随 scene 变化 elastic morph
3. **Stability layer** ($\mathbf{A}_k$, $\mathbf{P}$)：保证 convergence，每次 morph 后 re-learn

这种"层化解耦"思想类似 task-parameterized GMM (Calinon 2016) 但更灵活——TP-GMM 用 rigid transformation，EMP 用 Laplacian elastic editing。

### 7.2 为什么稳定 DS 比 Neural Policy 适合这个场景

- DS 是 closed-form，可以做 formal stability analysis via Lyapunov
- 1-2 个 demonstration 就能拟合（神经网络需要几百几千）
- Adaptation 是 algebraic operation（Laplacian editing）而非 retraining
- Compliance 和 reactivity 是 first-order DS 的天然属性

### 7.3 与 Diffusion Policy 的对比

Diffusion Policy (Chi et al. 2023) 用 diffusion model 学 action distribution：
- 优势：高表达力，多模态 action distribution
- 劣势：需要大量数据，没有 stability guarantee，难以 online adapt

EMP 的定位是另一极端：极少数据、强 stability guarantee、可 online adapt，但表达力受限（linear mixture + 单 attractor）。两者互补，适合不同 task。

Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 7.4 与 Contraction Theory 的联想

Paper 在 future work 提到 contraction theory。Contraction theory（Lohmiller & Slotine 1998）是比 Lyapunov 更强的 stability 概念：保证所有轨迹互相收敛（增量稳定性）。这可以放松"单 attractor"的限制，允许多 attractor 或 limit cycle。

Neural-ODE with contraction（如 Nawaz et al. ICRA 2024）可以学更复杂的 motion plan，但仍需要 training。EMP 思路 + contraction 可能是 next step。

### 7.5 Limitation 与机会

1. **依赖 object tracking**：FoundationPose 等 tracker 失败则 EMP 失败。Future: end-to-end vision input
2. **Single tangent space approximation**：大 rotation 时 distortion。Future: Lie group formulation
3. **P-QLF 限制 expressive motion**：复杂 nonlinear motion 难以 encode。Future: Neural ODE 或 contraction
4. **Multi-demonstration extension**：DAMM 可以 fit 多 demo 为 directed graph，general Laplacian editing 适用 graph，所以 EMP 自然扩展
5. **Object shape 信息缺失**：当前只用 keypose，复杂 task 需要更多 geometric awareness

### 7.6 与 Optimization-based Motion Planning 的关系

EMP 在某种意义上把 motion planning 和 motion execution 统一到 DS framework：
- "Planning" = Laplacian editing（解析的、连续的）
- "Execution" = DS integration + impedance control

这与 CHOMP/TrajOpt 等 optimization-based planner 不同：那些 plan 出 trajectory 再 tracking，EMP 直接 morph 整个 vector field，任何 state 都有 defined action。

### 7.7 联想到的其他相关工作

- **Riemannian Motion Policies (RMPs)** (Mukadam et al.): 类似 DS 但更 modular，多树状组合
- **GeoMap / Pullback** (Ratliff et al.): geometry-aware motion generation
- **Stable-BC** (Mehta et al. 2024): 同方向但用 error dynamics stabilization
- **Lyapunov Density Models** (Kang et al. ICML 2022): distribution-level Lyapunov 约束
- **Euclideanizing Flows** (Rana et al.): diffeomorphism 让 DS 更易学

RMPflow: https://arxiv.org/abs/1811.07049

---

## 8. Implementation 上的细节与思考

### 8.1 关键参数

- 30Hz update rate（必须达到 real-time reactive control 频率）
- UVD hyper-parameter $\gamma = 0.003$（控制 segmentation granularity）
- Obstacle sphere diameter 0.18m（实验设置）
- AMD Ryzen 7 5800X + 32GB RAM（实验硬件）

### 8.2 Computational Bottleneck

EMP 的 30Hz 主要瓶颈在 re-learning $\mathbf{A}_k$（SDP 求解）。Convex P-QLF 让 $\mathbf{P}$ 学习快了 30-50x，但 $\mathbf{A}_k$ 仍是 SDP。如果未来能用 closed-form 或 iterative QP 替代 SDP，可以更快。

### 8.3 Perception Pipeline 健壮性

GPT-4o + Grounded SAM + FoundationPose 是合理的"perception stack"，但每层都可能失败：
- GPT-4o 给错 object label（multi-object scene）
- Grounded SAM mask 不准
- FoundationPose 6D pose 在 occlusion 下漂移

论文未量化 perception 失败对 end-to-end success rate 的贡献，这是一个 future evaluation gap。

---

## 9. 我对这篇 paper 的整体评价

**强项**：
- 把 stability guarantee 与 online adaptation 结合，是个 nontrivial achievement
- Convex P-QLF formulation 是漂亮的工程创新，把非凸变凸
- Quaternion Laplacian editing 通过 tangent plane projection 实现，技术细节扎实
- Pipeline 把 LLM + SAM + Pose Estimator 串起来，实用性强
- 实验 ID vs OOD 对比直接显示 adaptation 价值

**弱项**：
- 表达力受限（linear mixture + 单 attractor + quadratic Lyapunov）
- Perception stack 复杂，failure mode 未充分分析
- Single tangent space 是 approximation，paper 自己承认
- Real robot 实验任务数量较少，每任务 10 trials 统计意义有限

**最值得学的设计**：
1. **Joints $\beta_{i,k,k+1}$ 的引入**：把 GMM 的"组件间关系"显式化，让 elastic morph 成为可能
2. **ReLU-based convex Lyapunov learning**：用 hinge loss 思路把 stability violation 变成可优化目标
3. **Endpoint constraints vs Laplacian coordinates preservation**：保留局部结构的同时让 endpoints 严格满足 task constraint

**对我自己 research 的启发**：层化解耦 policy 是 design pattern：结构 / 几何 / 稳定性可以分开 adapt，不同 layer 用不同 update rate 和 method。这种思路可推广到 RL、model-based RL、甚至 LLM agent design。

---

## 10. 推荐补充阅读

1. LPV-DS 原始 paper (Figueroa & Billard CoRL 2018): https://proceedings.mlr.press/v87/figueroa18a.html
2. SE(3)-LPVDS (Sun & Figueroa IROS 2024): https://arxiv.org/abs/2406.16824
3. Elastic-DS (Li & Figueroa CoRL 2023): https://proceedings.mlr.press/v229/li23b.html
4. DAMM (Sun et al. RA-L 2024): https://arxiv.org/abs/2406.00137
5. UMI (Chi et al. RSS 2024): https://universal-manipulation-interface.github.io/
6. FoundationPose (Wen et al. CVPR 2024): https://nvlabs.github.io/FoundationPose/
7. Single Tangent Space Fallacy (Jaquier et al.): https://arxiv.org/abs/2310.07902
8. DS for Imitation Learning Book (Billard, Mirrazavi, Figueroa): https://mitpress.mit.edu/9780262046132/learning-for-adaptive-and-reactive-robot-control/
9. Laplacian Mesh Editing (Sorkine et al.): https://doi.org/10.1007/3-540-26808-7_25
10. Contraction Theory (Lohmiller & Slotine 1998): https://ieeexplore.ieee.org/document/661067

总结一句话：EMP 把"motion policy 的结构（GMM/K）"和"geometric constraint（物体 keypose）"和"stability guarantee（P-QLF）"分层解耦，用 Laplacian editing 做 elastic morph、用 convex optimization 做 real-time stability re-learning，把 1-shot imitation learning 在 DS framework 下推到了 30Hz real-time adaptation 的能力。
