---
source_pdf: Cloth-Splatting 3D Cloth State Estimation.pdf
paper_sha256: 853a664d9fd8b6f8fe114cb93c2d336eb9e300a0bc6636b73b6f1e12ca2c6f39
processed_at: '2026-08-03T16:05:03-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Cloth-Splatting 用人话说

好，我换个更直觉的方式讲。想象你在教一个 robot 折衣服，robot 需要时刻知道"衣服现在长啥样、在哪儿"。这就是 paper 要解决的问题。

---

## 问题场景：为什么 cloth 难 track

假设你看一块布在桌上，robot 抓住一个角往上提。你站在旁边用几个 camera 拍照。问题是：

- 衣服会 **fold**、**wrinkle**、**self-occlude**（自己挡自己）
- 你从某些角度根本看不见被折起来那一半
- 布的 state 是连续的 3D shape，不是 rigid box 那种 6 DOF pose 就能描述的

之前的人怎么搞？两类思路：

**思路 A**：只用 2D image。搞个 2D tracking 或者 optical flow，但是丢了 3D 结构信息，布折起来你完全不知道背面长啥样。

**思路 B**：用 depth sensor 拿 point cloud。问题是 point cloud 只告诉你看得见的 surface，没告诉你看不见的部分，而且没 texture 信息（一块白布全是一样的点，你分不清哪个点对应哪个点）。

还有人用 GNN 学布的 dynamics，给定 robot action 预测布下一秒长啥样。问题是 sim-to-real gap，再加上 rollout 久了 error 累积。

**Cloth-Splatting 的 idea**：把 dynamics prediction (GNN) 和 vision observation (RGB) 用 Bayesian filtering 的方式结合起来。GNN 给个先验 guess，RGB 来修正这个 guess。

---

## 核心 trick：怎么用 RGB 修正 3D state

这是 paper 最关键的地方。你要用 image 去 update 3D mesh state，那你得有一个从 3D state 到 image 的 **differentiable mapping**——给定一个 mesh，能 render 出一张图，而且这个 render 过程能 backprop gradient。

**Gaussian Splatting (GS)** 恰好就是个 differentiable renderer。GS 用一堆 3D Gaussian "泼溅"到 image plane 上合成图像，整个过程可微。

但 standard GS 有个问题：它用百万个 free 3D Gaussians，每个 Gaussian 的 position、rotation、scale、color 都是 free parameter。你只有 3-4 个 camera views，去优化百万参数，这是严重 ill-posed。MD-Splatting 这种方法需要 50+ 个 cameras 才能跑得好。

**Cloth-Splatting 的 key insight**：把 Gaussians "钉"在 mesh 上。每个 mesh face 放几个 Gaussians，Gaussians 的位置用 barycentric coordinates 表示——也就是说，Gaussian 的位置是它所在 face 三个 vertices 的加权平均：

$$\mu = b_1 \mathbf{v}^1 + b_2 \mathbf{v}^2 + b_3 \mathbf{v}^3$$

变量含义：
- $\mu$：Gaussian 的 3D 位置
- $\mathbf{v}^1, \mathbf{v}^2, \mathbf{v}^3$：这个 face 的三个 vertex 的 3D positions
- $b_1, b_2, b_3$：barycentric weights，满足 $b_1 + b_2 + b_3 = 1$

这俩 weights 是 static 的，cloth 怎么动它们都不变。所以 mesh vertices 一动，所有 Gaussians 跟着动，rendered image 也跟着变。

**这样做的后果**：你只需要 optimize 几千个 mesh vertex positions，不需要 optimize 百万个 Gaussian positions。Search space 暴跌几个数量级。3-4 个 camera views 就能 converge。

这就是为什么 paper 说比 MD-Splatting 快 85%。

参考：3D Gaussian Splatting 原始 paper https://repo-splatting.doc.city.ac.uk / project repo https://github.com/graphdeco-inria/gaussian-splatting

---

## 整个 pipeline 走一遍

让我用一个具体场景走一遍。假设 t 时刻你有布的 state estimate $\mathbf{M}_t$（一组 vertex positions + velocities），robot 做了个 action $\mathbf{a}_t$（gripper 移动多少），你收到 t+1 时刻的 RGB observation $\mathbf{Y}_{t+1}$（4 个 camera views）。

**Step 1: Prediction**

把 $\mathbf{M}_t$ 和 $\mathbf{a}_t$ 喂给 GNN。GNN 预测 $\hat{\mathbf{M}}_{t+1}$：每个 vertex 的下一个 position 和 velocity。

GNN 怎么处理 action？它假设被 grasp 的那个 vertex 是 rigidly attached 到 gripper 的，所以直接根据 gripper velocity 更新那个 vertex 的 velocity。然后通过 mesh graph 的 message passing 把这个 perturbation 传播到整个布。这个 trick 来自 Wang et al. 的 Visual Haptic Reasoning paper https://arxiv.org/abs/2208.04870

**Step 2: Refinement via GS**

GNN 给的 prediction 不准（sim-to-real gap + accumulated error）。所以加一个 residual correction：

$$\tilde{\mathbf{M}}_{t+1} = \hat{\mathbf{M}}_{t+1} + \delta\hat{\mathbf{M}}_{t+1}$$

变量：
- $\hat{\mathbf{M}}_{t+1}$：GNN prediction，fixed
- $\delta\hat{\mathbf{M}}_{t+1}$：要学的 residual
- $\tilde{\mathbf{M}}_{t+1}$：最终 refined state

这个 residual 怎么学？用一个 MLP $u_\psi$，输入是 normalized time step（用 sinusoidal positional encoding，NeRF 那种），输出是所有 vertex 的 3D offset。

然后用 mesh-constrained GS 把 $\tilde{\mathbf{M}}_{t+1}$ render 成 4 张 predicted images $\tilde{\mathbf{Y}}_{t+1}$。

**Step 3: Compute loss and backprop**

$$\mathcal{L}_{obs} = ||\mathbf{Y}_{t+1} - \tilde{\mathbf{Y}}_{t+1}||_2^2$$

变量：
- $\mathbf{Y}_{t+1}$：真实 RGB observation
- $\tilde{\mathbf{Y}}_{t+1}$：从 $\tilde{\mathbf{M}}_{t+1}$ render 出来的 predicted images

用 gradient descent 更新 MLP $u_\psi$ 的 weights，从而更新 $\delta\hat{\mathbf{M}}_{t+1}$，从而更新 $\tilde{\mathbf{M}}_{t+1}$，从而更新 rendered image，loss 下降。

iterate 几千次直到 converge。

**Intuition**：MLP weights 是 trajectory 的 compressed representation，每个 time step 通过 MLP decode 出当前 offset。优化 MLP weights = optimize trajectory。这类似 implicit neural representation 思想，跟 NeRF 用 MLP 表示 scene 是一个套路。

---

## 为什么要 residual MLP，不直接 fine-tune GNN

你可能会问：为啥不直接 backprop 通过 GNN 来 fine-tune GNN weights？

三个原因：

**原因 1：Vanishing gradient through long rollout**

GNN 内部有 15 个 GN blocks with residual connections，再 unroll 16 步，gradient 要 backprop 通过 15 × 16 = 240 个 GN blocks。即使有 residual connections，gradient 也会衰减。

**原因 2：Catastrophic forgetting**

GNN 是在大量 training scenes 上学的 general dynamics。test-time fine-tune 它会破坏这个 general knowledge，overfit 到当前 scene。Residual MLP 是 separate 的 lightweight network，不动 GNN weights。

**原因 3：Speed**

Fine-tune GNN 需要 retrain 整个网络，per-scene。Residual MLP 是小 network（3-layer ReLU，width 256），optimize 几千 iter 就行。

---

## Regularization 为啥这么重要

去掉 regularization，MTE 从 8.923 mm 暴涨到 15.772 mm，差 76.7%。为什么？

Inverse rendering 本质上 ill-posed。多个 3D configurations 都能 render 出同样的 2D image。没有 prior 的话，optimizer 会找到 weird 的 cloth shape——比如把 mesh 拉成不自然的 stretched form 去匹配 image 里的视觉 feature。

Regularization 干啥：
- $\mathcal{L}_{iso}$：强制 neighboring vertices 之间距离不变（cloth 不可拉伸）
- $\mathcal{L}_{magn}$：鼓励最小 motion（GNN prediction 已经是 good guess，别乱改）
- $\mathcal{L}_{SSIM}$：不只看 pixel-wise color，看 structural similarity

具体公式：

**Isometric loss**:
$$\mathcal{L}_{iso} = \sum_{t=0}^{T-1} \sum_{i=0}^{N-1} \sum_{\mathcal{N}(v_{t,i})} |d(v_{t,i}, v_{t,j}) - d(v_{t+1,i}, v_{t+1,j})|$$

变量：
- $T$：trajectory 长度
- $N$：vertex 数量
- $\mathcal{N}(v_{t,i})$：$v_{t,i}$ 的 neighboring vertices（在 mesh graph 上相邻的）
- $d(\cdot, \cdot)$：Euclidean distance
- $v_{t,i}, v_{t,j}$：t 时刻 vertex i 和它的 neighbor j 的 positions
- $v_{t+1,i}, v_{t+1,j}$：t+1 时刻对应 positions

Intuition：物理上布几乎不可拉伸，所以 mesh edges 长度应该不变。这个 prior 把 solution space 限制在 physically plausible 的 cloth configurations 上。

这就是 As-Rigid-As-Possible (ARAP) 的思想，参考 Sorkine & Alexa 2007 https://dl.acm.org/doi/10.1145/1276377.1276468

---

## 一些重要细节

### Freeze Gaussians after 6k iterations

非常有意思的细节：前 1.5k iterations 只 optimize Gaussians 的 appearance（color, scale, opacity）让它 fit 布的 visual appearance。接下来 5.5k-6.5k iterations 联合 optimize Gaussians + residual MLP。之后 **freeze Gaussians**，只 optimize residual MLP。

为什么？如果不 freeze，Gaussians 会"偷懒"——它会去 fit deformed appearance，而 residual MLP 就 lazy 了不去学正确的 offset。这是 **curse of flexibility** 问题：当有多个 flexible components 都能 explain data，它们会互相 push responsibility，结果都不学正确的 thing。

Freeze Gaussians 之后，只有 residual MLP 能调整 mesh，迫使它去学正确的 dynamics offset。

### Negative barycentric coordinates trick

允许 barycentric coordinates 取负值，这样 Gaussian 可以跑到 face 外面。当 Gaussian 跑出 face 时，它应该被 reassign 到另一个 face。这是个 clever 的 bookkeeping trick，让 GS 的 pruning/densification 机制能继续 work。

### 每 face 只放 2 个 Gaussians

Total 才 4k 个 Gaussians，远少于 standard GS 的百万级。因为 cloth appearance 是相对简单的（一块布颜色一致），不需要那么多 Gaussians 来 model appearance。

### GNS architecture 细节

GNN 用的是 Sanchez-Gonzalez et al. 2020 的 Graph Network Simulator (GNS) architecture https://arxiv.org/abs/2002.09405

三个部分：
- Encoder：两个 MLP $\phi_p, \phi_e$，分别 encode vertex features 和 edge features
- Processor：15 个 GN blocks with residual connections
- Decoder：MLP 输出每个 vertex 的 acceleration

用 forward-Euler integration：
$$\dot{\mathbf{V}}_{t+1} = \dot{\mathbf{V}}_t + \ddot{x}_i \cdot \Delta t$$
$$\mathbf{V}_{t+1} = \mathbf{V}_t + \dot{\mathbf{V}}_{t+1} \cdot \Delta t$$

训练用 one-step MSE loss，200 epochs，Adam optimizer。

有意思的是 GNN 只在 TOWEL 上训练，但 generalize 到 TSHIRT 和 SHORTS。说明 GNN 学的是 local physics propagation rules（spring、stretching、bending），这些 rules 是 cloth-mesh invariant 的。

---

## ROLLOUT vs ITERATIVE 两种模式

**ROLLOUT**：一次性 unroll GNN 整个 trajectory（16 步），然后同时 refine 所有 16 个 states。Fast but error accumulates。

**ITERATIVE**：predict H 步，refine，用 refined state 作为下一轮 GNN input。Slow but accurate。

具体数据：

| 模式 | H | MTE (mm) | 时间 |
|------|---|----------|------|
| ITERATIVE | 1 | 0.767 | 53:35 |
| ITERATIVE | 2 | 0.890 | 30:14 |
| ITERATIVE | 4 | 0.819 | 15:53 |
| ITERATIVE | 8 | 1.170 | 9:32 |
| ROLLOUT | 16 | 5.328 | 2:12 |

Intuition：H 加倍，时间近似 halve。Tracking 用 ROLLOUT（要快），manipulation 用 ITERATIVE（要准）。

---

## 实验结果直觉

Table 1 主结果：

| Method | MTE (mm) | δ_avg | Survival |
|--------|----------|-------|----------|
| RAFT-Oracle | 18.324 | 0.683 | 0.715 |
| DynaGS | 10.924 | 0.804 | 0.835 |
| MD-Splatting | 3.635 | 0.847 | 0.887 |
| GNN only | 13.853 | 0.747 | 0.800 |
| **Cloth-Splatting** | **3.284** | **0.862** | **0.910** |

几个观察：

1. **GNN only 已经不错**（13.853 mm），说明 GNN prior 很强。GS 把它从 13.853 拉到 3.284，refine 了 4 倍多。

2. **RAFT-Oracle 最差**，说明 2D tracking 根本不够用。即使给 oracle view selection，2D 信息严重缺失 3D 结构。

3. **比 MD-Splatting 快 85%**，因为 GNN warm start + mesh constraint 大幅减少 search space。

4. **SHORTS 最难**，因为 self-occlusion 最严重。被折起来的部分相机看不见，GS 没 supervision，只能靠 GNN 预测。这是 deformable tracking 的 fundamental challenge。

参考：
- MD-Splatting https://md-splatting.github.io
- DynaGS https://dynamic3dgaussians.github.io
- RAFT https://arxiv.org/abs/2003.12039

---

## Manipulation 应用：闭环折衣服

他们还展示了 Cloth-Splatting 能 enable closed-loop manipulation。任务是 half-folding：给 pick & place positions，优化它们之间的 trajectory。

为什么 linear trajectory 不行？单 gripper 抓一个角直线拉到对面，布会 drag、滑掉、不折好。需要 mid-trajectory 调整高度、速度等。

他们用 MPC (Model Predictive Control)：
1. Sample N 个 candidate actions
2. 用 GNN rollout 预测每个 candidate 的结果
3. 算 cost：$||\hat{\mathbf{M}}_{h+1} - \mathbf{M}_g||_2^2$（离 goal 多远）
4. 选最优 action 执行一步
5. 用 Cloth-Splatting refine state
6. 重复

结果：

| Method | TOWEL MSE (×10⁻³) | TSHIRT MSE (×10⁻³) |
|--------|-------------------|---------------------|
| FIXED | 2.2 ± 0.4 | 2.4 ± 0.4 |
| MPC-OL | 1.8 ± 2.1 | 7.3 ± 5.2 |
| MPC-CS | 0.6 ± 0.6 | 1.2 ± 0.8 |
| MPC-ORACLE | 0.4 ± 0.2 | 0.8 ± 0.5 |

变量：
- MSE：mean squared error between final cloth state 和 goal state
- FIXED：linear trajectory
- MPC-OL：open-loop MPC，不更新 state
- MPC-CS：MPC + Cloth-Splatting refine
- MPC-ORACLE：MPC + ground-truth state access

MPC-CS 接近 oracle！说明 state estimation 质量足够好，closed-loop manipulation 能 work。

MPC-OL 在 TSHIRT 上方差巨大（7.3 ± 5.2），说明 open-loop 在 complex cloth 上不稳定——一开始小 error 累积，后期完全失控。

参考 Garcia-Camacho et al. 2020 bimanual cloth benchmark https://arxiv.org/abs/1910.01745

---

## Real-world 实验

用 Franka Emika Panda 7-DOF robot，3 个 RealSense d435 cameras（只用 RGB），折一块 rectangular cloth。

Pipeline：
1. t=0 时用 depth 初始化 mesh（Delaunay triangulation from point cloud）
2. 用 Grounding-DINO (prompt "cloth") + SAM (prompt "robot gripper") 做 segmentation
3. XMEM 做 video tracking 维护 mask
4. Cloth-Splatting 全程 refine state

结果显示 GNN prediction 部分预测对但累积 error 大，Cloth-Splatting 成功 refine 出更准确的 mesh。

参考：
- Grounding-DINO https://github.com/IDEA-Research/GroundingDINO
- SAM https://github.com/facebookresearch/segment-anything
- XMEM https://github.com/hkchengrex/XMem

---

## Robustness to initialization error

很 practical 的实验。他们对 initial mesh 加 noise：
- TRANS：平移 ±5cm
- ROT：旋转 ±30°
- SCALING：scale 0.8-1.2x
- NOISE：Gaussian noise variance 0.005

结果：Cloth-Splatting 对这些 perturbation 相当 robust。MTE 从 error-free 的 2.193 涨到 augmented 平均 2.961 mm，涨幅不大。

特别 interesting：如果先用 t=0 observation refine initial mesh 再 unroll GNN，可以 avoid catastrophic failure（某些 ROT augmentation 之前 tracking 太差没法 evaluate）。

**Intuition**：即使初始 mesh 错了，GS vision supervision 能在 t=0 时就把它纠正回来。这是 inverse rendering 的力量——visual observation 能 disambiguate 很多 initial uncertainty。

---

## Limitations 直觉化

**Speed**：即使快了 85%，还是要 minutes per trajectory。Real-time control 需要的是 Hz 级别，差几个数量级。

**Multi-camera**：3-4 个 calibrated camera 在 real-world setup 里不便宜不方便。Dust3r https://github.com/naver/dust3r 这种 calibration-free 方法可能能缓解。

**Static appearance**：布的颜色、texture 假设不变。实际中 lighting 会变，shadow 会动，这些会被误认为是 cloth deformation。这是 photometric loss 的 inherent limitation。

**Occlusion-free initialization**：crumpled cloth 你看不见全貌，没法 initialize mesh。Wang et al. 2024 TRTM https://arxiv.org/abs/2410.19312 这种 template-based reconstruction 可能能解决。

**GNN training data**：需要 simulator 生成训练数据。Sim-to-real gap 永远是个问题。不过 GNN 只做 prior，GS 会 refine，所以 gap 能被 partially compensated。

---

## 整体 intuition 总结

整个 paper 的核心 insight 可以浓缩成一句话：**用 mesh 做 structural prior，用 GS 做 differentiable rendering bridge，用 GNN 做 dynamics prior，三者结合把 ill-posed inverse problem 变成 tractable optimization**。

更直觉的比喻：

想象你画一个 caricature（漫画像）。Free GS 是给你一张白纸随便画——画啥都行，但你只有几张参考照片，画出来可能完全不像。Mesh-constrained GS 是给你一个 face template，你只需要调整 face 的几个 control points（eyes 大小、nose 位置）——search space 大大减小，几张参考照片就够了。

GNN 是你的"face drawing prior"——你知道人脸大概长啥样，先画个 rough sketch。GS refinement 是你对着照片 fine-tune 这个 sketch。两者结合就是又快又像。

这种 pattern (differentiable dynamics + differentiable rendering + Bayesian filtering) 是 modern robotics + graphics 的 powerful paradigm。可以推广到 rope tracking、fluid surface、soft body manipulation 等其他 deformable problems。

---

## 更多 reference

- Cloth-Splatting project: https://kthrpl.github.io/cloth-splatting
- 3D Gaussian Splatting: https://repo-splatting.doc.city.ac.uk / https://github.com/graphdeco-inria/gaussian-splatting
- GNS paper: https://arxiv.org/abs/2002.09405
- MD-Splatting: https://md-splatting.github.io
- DeformGS: https://bduisterhoft.github.io/deformgs/
- PhysGaussian: https://github.com/Xiangyu1Xu/PhysGaussian
- SoftGym: https://github.com/Xingyu-Lin/softgym
- SpeedFolding: https://github.com/berkeleyautomation/speedfolding
- AdaFold: https://arxiv.org/abs/2403.06210
- NeRF: https://arxiv.org/abs/2003.08934
- CoTracker: https://arxiv.org/abs/2307.07635
- TAP-VID: https://tapvid.github.io
- RAFT: https://arxiv.org/abs/2003.12039
- Dust3r: https://github.com/naver/dust3r
- SAM: https://github.com/facebookresearch/segment-anything
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- XMEM: https://github.com/hkchengrex/XMem
- Lips et al. 2024 synthetic cloth: https://arxiv.org/abs/2401.01734
- RoMa rotation toolbox: https://github.com/naver/roma
- Garcia-Camacho et al. 2020 bimanual cloth benchmark: https://arxiv.org/abs/1910.01745
- Wang et al. 2024 TRTM: https://arxiv.org/abs/2410.19312
- NVIDIA Flex: https://developer.nvidia.com/flex
- Blender: https://www.blender.org
- Delaunay triangulation: standard computational geometry
- ARAP (Sorkine & Alexa 2007): https://dl.acm.org/doi/10.1145/1276377.1276468
- SSIM (Wang et al. 2004): https://ieeexplore.ieee.org/document/1285907

---

# Cloth-Splatting 深度解析

## 1. Paper 核心思想总览

这篇 paper 的核心 contribution 是把 **Bayesian filtering** 的 prediction-update 框架移植到 deformable cloth 的 3D state estimation 上，并且 update step 用 **Gaussian Splatting (GS)** 作为可微的 measurement model。这 very elegant，因为 GS 提供了一条从 3D mesh state 直接到 RGB image 的 differentiable pathway，使得 gradient-based optimization 可以直接用 photometric loss refine state。

Key insight 拆解：
- 把 cloth 表示成 augmented mesh $\mathbf{M}_t = (\mathbf{V}_t, \dot{\mathbf{V}}_t, \mathbf{E}_t)$（vertex positions + velocities + edges）
- 用 GNN 学一个 action-conditioned dynamics model 做 prediction
- 把 mesh 的每个 face populate 上 3D Gaussians，Gaussians 的位置用 barycentric coordinates 相对于 mesh vertices 表达
- 这样 mesh 一动，Gaussians 跟着动，渲染出来的 image 也变
- 用 RGB observation 算 photometric loss，反向 propagate gradient 修正 mesh state

这种 mesh-constrained GS 是 critical，因为 standard GS 是 free 3D Gaussians（百万级参数），这里把 Gaussians "glue" 到 mesh 上，自由度急剧减少，剩下的 only 是 mesh vertices（几千个），所以 optimization 能在 sparse views 下 converge。

项目主页：https://kthrpl.github.io/cloth-splatting

---

## 2. Problem Formulation 详细解析

### 2.1 State 表示

$$\mathbf{M}_t = (\mathbf{V}_t, \dot{\mathbf{V}}_t, \mathbf{E}_t)$$

- $\mathbf{V}_t \in \mathbb{R}^{N \times 3}$：N 个 vertex 的 3D positions（待估计）
- $\dot{\mathbf{V}}_t \in \mathbb{R}^{\tilde{N} \times 3}$：vertex velocities（待估计）
- $\mathbf{E}_t \in \mathbb{Z}_+^{L \times 2}$：L 条 edges 定义 mesh 的 connectivity，**time-invariant**（$\mathbf{E}_0 = \mathbf{E}_1 = \cdots = \mathbf{E}_t$）

注意这里 $\tilde{N}$ 和 $N$ 不一定相等（可能 boundary vertices 有特殊 velocity 处理），这个细节 paper 没展开。

### 2.2 Observation 与 Action

- $\mathbf{Y}_{t+1} = \{\mathbf{I}^0_{t+1}, \dots, \mathbf{I}^K_{t+1}\}$：K 个 multi-view RGB images
- 每个 $\mathbf{I}^k_{t+1} \in \mathbb{R}^{w \times h \times 3}$
- camera matrices $\mathbf{P} = \{\mathbf{P}^0, \dots, \mathbf{P}^K\}$，每个 $\mathbf{P}^k \in \mathbb{R}^{4 \times 4}$
- action $\mathbf{a}_t \in \mathbb{R}^3$：Cartesian end-effector velocities

### 2.3 Bayesian Filtering 公式

Prediction step（Eq.1）:

$$p(\mathbf{M}_{t+1} | \mathbf{Y}_{1:t}, \mathbf{a}_{1:t}) = \int p(\mathbf{M}_{t+1} | \mathbf{M}_t, \mathbf{a}_t) \, p(\mathbf{M}_t | \mathbf{Y}_{1:t}, \mathbf{a}_{1:t-1}) \, d\mathbf{M}_t$$

变量含义：
- $p(\mathbf{M}_{t+1} | \mathbf{M}_t, \mathbf{a}_t)$：transition probability，由 GNN 近似
- $p(\mathbf{M}_t | \mathbf{Y}_{1:t}, \mathbf{a}_{1:t-1})$：previous timestep 的 posterior，作为新的 prior
- 这就是经典的 Chapman-Kolmogorov equation

Update step（Eq.2）:

$$p(\mathbf{M}_{t+1} | \mathbf{Y}_{1:t+1}, \mathbf{a}_{1:t}) = \frac{1}{\eta} p(\mathbf{Y}_{t+1} | \mathbf{M}_{t+1}) \, p(\mathbf{M}_{t+1} | \mathbf{Y}_{1:t}, \mathbf{a}_{1:t})$$

- $p(\mathbf{Y}_{t+1} | \mathbf{M}_{t+1})$：measurement likelihood，由 GS 近似
- $\eta$：normalization constant
- 这就是 Bayes' rule

所以整个 framework 是 EKF / particle filter 的精神，但 transition 和 measurement 都用 learnable + differentiable models 代替。

---

## 3. GNN Dynamics Model 细节

### 3.1 架构（GNS-style）

GNN 基于 Sanchez-Gonzalez 等人的 Graph Network Simulator (GNS) architecture，三个核心模块：

**Encoder**：两个 MLP $\phi_p, \phi_e$
- $\phi_p$：vertex features → latent embedding $h_i$
- $\phi_e$：edge features → latent embedding $g_{jk}$

**Processor**：L = 15 个 Graph Network (GN) blocks with residual connections
- 每个 GN block 包含 edge update MLP, vertex update MLP, global update MLP
- 信息通过 mesh graph 结构 propagate

**Decoder**：MLP $\psi$ 输出每个 vertex 的 acceleration
$$\ddot{x}_i = \psi(h_i^L)$$

然后用 forward-Euler integration 算 velocity 和 position：
$$\dot{\mathbf{V}}_{t+1} = \dot{\mathbf{V}}_t + \ddot{x}_i \cdot \Delta t$$
$$\mathbf{V}_{t+1} = \mathbf{V}_t + \dot{\mathbf{V}}_{t+1} \cdot \Delta t$$

### 3.2 Input features

- Vertex features：过去 k=3 个 timestep 的 velocities + vertex type（binary flag 区分 grasped vs non-grasped）
- Edge features：relative distance vector $(\mathbf{v}_j - \mathbf{v}_k)$ 和它的 norm $||\mathbf{v}_j - \mathbf{v}_k||$

### 3.3 Action conditioning

为了 condition 在 action 上，他们直接把 grasped particle 的 velocity 根据 robot action 更新（假设 grasped vertex 刚性 attached 到 gripper）。这是 Wang et al. 的 Visual Haptic Reasoning 里的 trick。这样 action 信息通过 mesh graph 传播到整个 cloth。

### 3.4 Training

- Dataset: TOWEL only（但 generalizes 到 TSHIRT, SHORTS）
- Loss: one-step MSE between predicted and ground-truth meshes
- 200 epochs，Adam optimizer

**Intuition**: GNN 学的是 local physics propagation rules（弹簧、拉伸、弯曲），这些 rules 是 cloth-mesh invariant 的，所以 training 在 TOWEL 上但能在 TSHIRT 上 generalize。这点很重要，因为这意味着 dynamics model 不需要 per-scene retraining。

---

## 4. Mesh-Constrained Gaussian Splatting — 核心 Technical Contribution

这是 paper 最关键的部分。我详细讲一下。

### 4.1 Standard 3D GS 回顾

Original 3D Gaussian Splatting (Kerbl et al. 2023): https://repo-splatting.doc.city.ac.uk
Project: https://github.com/graphdeco-inria/gaussian-splatting

每个 3D Gaussian 由：
- Mean $\mu \in \mathbb{R}^3$
- Covariance $\Sigma \in \mathbb{R}^{3 \times 3}$（必须 positive semi-definite）
- Color $c$（spherical harmonics）
- Opacity $\alpha$

Covariance 分解（Eq.4）：
$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

- $\mathbf{R} \in SO(3)$：rotation matrix（可由 quaternion 参数化）
- $\mathbf{S}$：diagonal scale matrix（控制 Gaussian 在各方向的 spread）
- 这种分解保证 $\Sigma$ PSD

投影到 image space（Eq.5）：
$$\Sigma' = \mathbf{J} \mathbf{P} \Sigma \mathbf{P}^T \mathbf{J}^T$$

- $\mathbf{P} \in \mathbb{R}^{4 \times 4}$：camera matrix
- $\mathbf{J}$：Jacobian of affine approximation of projective transformation
- $\Sigma'$：2D image-space covariance

Pixel color via α-blending（Eq.6）：
$$\mathbf{c}^p = \sum_{i \in N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

- $N$：覆盖该 pixel 的有序 Gaussians
- $\mathbf{c}_i$：第 i 个 Gaussian 的 color
- $\alpha_i$：projected 2D Gaussian evaluated at pixel × learned opacity
- $\prod_{j=1}^{i-1}(1-\alpha_j)$：front-to-back compositing 的 transmittance

### 4.2 关键 trick：Mesh Constraint

把 Gaussians "绑"到 mesh 上。每个 Gaussian 分配到一个 mesh face，position 用 barycentric coordinates 表示（Eq.7）：

$$\mu = b_1 \mathbf{v}^1 + b_2 \mathbf{v}^2 + b_3 \mathbf{v}^3$$

- $\mathbf{v}^1, \mathbf{v}^2, \mathbf{v}^3 \in \mathbf{V}$：assigned face 的三个 vertices
- $b_1 + b_2 + b_3 = 1$：barycentric coordinates
- 这些 barycentric coords 在 mesh deformation 时**保持不变**

**Rotation** 也 relative to face：
$$\mathbf{R} = \mathbf{R}^F \mathbf{R}'$$

- $\mathbf{R}'$：Gaussian 相对于 face 的 static orientation
- $\mathbf{R}^F$：face 当前 orientation

### 4.3 Face Orientation Estimation

通过 vector registration 找 $\mathbf{R}^F$（Eq.9）：
$$\min_{\mathbf{R}^F} \sum_{i=\{1,2,3\}} ||\mathbf{R}^F \mathbf{v}^i_0 - \mathbf{v}^i_t||^2$$

- $\mathbf{v}^i_0$：face 在 initial mesh 上的 vertex positions
- $\mathbf{v}^i_t$：当前 deformed mesh 上对应 vertex positions
- 用 RoMa toolbox (Bregier 2021) 求解
- 这是一个 Procrustes-style 3D rotation 求解问题

### 4.4 Initialization 与 Gaussian 数量

- 每个 mesh face 放 2 个 Gaussians
- Barycentric coordinates 从 $\mathcal{N}(1/3, 0.05)$ 采样，然后 normalize 让它们和为 1
- 总共 ~4k Gaussians（远少于 standard GS 的百万级）

### 4.5 关键 Optimization 细节

非常重要的 detail：
- 前 1.5k iterations：only optimize Gaussians with $\mathcal{L}_{obs}$，no regularization
- 接下来 5.5k-6.5k iterations：jointly optimize Gaussians + residual network
- 之后 **freeze Gaussian attributes**，only optimize residual dynamics

为什么 freeze？因为如果不 freeze，Gaussians 会"overfit"到 deformed appearance，而 residual dynamics model 就 lazy 了不去学正确的 offset。这是一种 curse of flexibility 问题——flexibility 太大反而分不清谁该负责。

### 4.6 Pruning 细节

为了 keep Gaussian 数量少，他们 increase pruning opacity threshold（假设 cloth 不透明）。

### 4.7 Negative barycentric coordinates trick

允许 barycentric coordinates 取负值——这会让 Gaussian 跑到 face 外面，从而可以 detect when Gaussian 应该被 reassign 到另一个 face。这是个 clever 的 bookkeeping trick。

### 4.8 为什么这个 design 这么 powerful

**Intuition**：Standard GS 的 optimization 需要从 scratch 学习百万 Gaussians 的位置、旋转、颜色——这个 search space 巨大，需要 dense multi-view observations（50+ cameras in MD-Splatting）。Mesh-constrained GS 把这个 search space 大大 reduce：
- Gaussians 的位置由 mesh vertices 决定
- 只需要 refine 几千个 vertex positions
- 视觉 appearance（color, scale, opacity）由 Gaussian attributes 学到，但是 static across time

所以只需要 sparse 3-4 cameras + GNN warm start 就能 converge。这就是为什么 85% faster。

---

## 5. State Update via Residual Learning

### 5.1 为什么不直接优化 GNN

直接 backprop 通过 GNN roll-out 会有 vanishing gradient 问题（GNN 内部 propagations + recursive roll-out）。这是 RNN 训练中经典的 issue。

### 5.2 Residual trick (Eq.8)

$$\tilde{\mathbf{M}}_{t+1} = \hat{\mathbf{M}}_{t+1} + \delta\hat{\mathbf{M}}_{t+1}$$

- $\hat{\mathbf{M}}_{t+1}$：GNN prediction（fixed，不更新）
- $\delta\hat{\mathbf{M}}_{t+1} = u_\psi(t+1)$：learned residual state update，由 MLP $u_\psi$ 参数化
- $\psi$：MLP 参数

### 5.3 Residual MLP 架构

- 3-layer ReLU MLP，width 256
- Input: normalized time step $t/T \in [0, 1]$，用 sinusoidal frequency encoding（6 frequencies，NeRF-style positional encoding）
- Output: $3 \times N$（N 个 vertex 的 3D offsets）
- 初始化：output layer weights 用 zero-centered normal distribution with covariance 0.0001，让初始 residual 接近 0

### 5.4 为什么用 time-conditioned MLP

这是个 interesting design choice。Residual 不是直接预测每个 vertex 的 offset，而是从 time step encode 出来。这类似 implicit neural representation 的思想——把 trajectory 上的所有 offsets encode 在一个 MLP 的 weights 里。

**Intuition**：MLP weights 像 trajectory 的 compressed representation，每个 time step 通过 MLP decode 出当前 offset。优化 MLP weights = optimize trajectory。

这种 design 共享了 information across time steps，使得 trajectory 上相邻 frames 的 offsets 是 smooth 的（因为 MLP 是连续的 function of time）。

### 5.5 两种 Update 模式

**ITERATIVE**：predict H steps ahead，refine，再用 refined state 作为下一轮 GNN input。H=1,2,4,8。

**ROLLOUT**：unroll GNN 整个 trajectory，refine 所有 states 同时。

Table 4 显示 ROLLOUT (H=16) 用 2:12 minutes，但 MTE 5.328；ITERATIVE (H=1) 用 53:35 minutes，MTE 0.767。这是 speed-accuracy tradeoff。

---

## 6. Loss Functions 详细解析

### 6.1 Observation Loss (Eq.3)

$$\mathcal{L}_{obs} = ||\mathbf{Y}_{t+1} - h_{GS}(\tilde{\mathbf{M}}_{t+1}, \mathbf{P})||_2^2$$

- $\mathbf{Y}_{t+1}$：ground-truth RGB observation
- $h_{GS}$：GS rendering function
- $\tilde{\mathbf{M}}_{t+1}$：refined state
- $\mathbf{P}$：camera matrices

这是 photometric consistency loss。

### 6.2 Regularization Loss

$$\mathcal{L}_{reg} = \mathcal{L}_{SSIM} + \mathcal{L}_{iso} + \mathcal{L}_{magn}$$

#### SSIM Loss (Eq.11)

$$\mathcal{L}_{SSIM}(v, w) = \frac{(2\mu_v \mu_w + c_1)(2\sigma_{vw} + c_2^p)}{(\mu_v^2 + \mu_w^2 + c_1)(\sigma_v^2 + \sigma_w^2 + c_2)}$$

- $v, w$：两个 image windows
- $\mu_v, \mu_w$：windows 的 mean color
- $\sigma_v^2, \sigma_w^2$：color variances
- $\sigma_{vw}$：color covariance
- $c_1, c_2$：stabilization constants

SSIM 比 pure pixel-wise loss 好，因为它考虑 local neighborhood 的 structural information。

#### Isometric Loss (Eq.10)

$$\mathcal{L}_{iso} = \sum_{t=0}^{T-1} \sum_{i=0}^{N-1} \sum_{\mathcal{N}(v_{t,i})} |d(v_{t,i}, v_{t,j}) - d(v_{t+1,i}, v_{t+1,j})|$$

- $T$：trajectory 长度
- $N$：vertex 数量
- $\mathcal{N}(v_{t,i})$：$v_{t,i}$ 的 neighboring vertices
- $d(\cdot, \cdot)$：Euclidean distance

这个 loss 强制 neighboring vertices 在 consecutive frames 间保持 constant distance，即 cloth 是 inextensible 的（As-Rigid-As-Possible, ARAP behavior，参考 Sorkine & Alexa 2007）。

**Intuition**：cloth 在物理上几乎不可拉伸，所以 mesh edges 长度应该不变。这个 prior 防止 optimization 把 cloth 拉成不自然形状去 fit 视觉。

#### Motion Magnitude Loss (Eq.12)

$$\mathcal{L}_{magn} = \sum_{t=0}^{T-1} \sum_{i=0}^{N-1} ||v_{t,i} - v_{t+1,i}||_2^2$$

- 鼓励最小 motion per vertex
- 数值稳定性

**Intuition**：从 GNN 出发的 prediction 已经是合理的初始 guess，optimization 应该只做最小必要的修正。这个 loss 防止 optimization 走极端。

### 6.3 Ablation 关于 Regularization (Table 2 - A3)

去掉 $\mathcal{L}_{reg}$ 后 MTE 从 8.923 升到 15.772，差了 76.7%。这说明 regularization 极其重要——只有 photometric loss 的话，cloth 会被拉成不自然形状来 fit 视觉。这是 ill-posed inverse problem 的典型问题。

---

## 7. Architecture 总览图

```
┌─────────────────────────────────────────────────────────────┐
│  Input: M_{t-m:t} (history), a_t (robot action), Y_{t+1}  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │     Prediction Step (GNN f_θ)        │
        │  ┌──────┐   ┌──────┐   ┌──────┐     │
        │  │Encoder│→ │Processor│→│Decoder│    │
        │  │φ_p,φ_e│   │15 GN   │   │  ψ   │    │
        │  └──────┘   └──────┘   └──────┘     │
        │  Output: M̂_{t+1} (predicted state)  │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │     Residual Update (MLP u_ψ)        │
        │  Input: time step (sinusoidal enc)   │
        │  Output: δM̂_{t+1}                   │
        │  M̃_{t+1} = M̂_{t+1} + δM̂_{t+1}     │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Mesh-constrained GS (h_GS)          │
        │  - Gaussians on mesh faces           │
        │  - Barycentric coordinates           │
        │  - Render predicted RGB Ỹ_{t+1}     │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Photometric Loss + Regularization   │
        │  L_obs = ||Y_{t+1} - Ỹ_{t+1}||²     │
        │  L_reg = L_SSIM + L_iso + L_magn    │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Backprop to ψ                       │
        │  Update residual MLP weights         │
        └──────────────────────────────────────┘
                              │
                              ▼
                    Output: M̃_{t+1}
```

---

## 8. Experimental Setup 详细

### 8.1 Synthetic Dataset

- 75 scenes 总计
- 3 object categories：TOWEL, SHORTS, TSHIRT
- 5 mesh variations per object × 5 trajectories per variation = 25 scenes per category
- Mesh variations 用 Lips et al. 2024 的方法 procedurally generated
- NVIDIA Flex simulator 做 physics simulation
- 4 camera views per scene
- RGB-D（depth 只给 baseline methods 用）
- Blender 渲染 photo-realistic

### 8.2 Trajectory Generation

- Quadratic Bézier curves with 3 control points
- Pick & place locations 作为 primary control points
- 中间 control point：pick-place 之间，height 在 [0.05, 0.15] cm
- 随机 tilt 在 $[-\pi/4, \pi/4]$ rad
- Gripper velocity 在 [0.5, 2] cm/s 随机
- Discretize 成 $\Delta x_1, \dots, \Delta x_T$ 满足 $x_{pick} + \sum_{i=1}^T \Delta x_i = x_{place}$

### 8.3 Real-world Setup

- 3 calibrated RealSense d435 cameras（only RGB used）
- Franka Emika Panda 7-DOF robot
- Cartesian position controller
- 假设已知 pick & place locations，cloth 已经 grasped
- Segmentation: Grounding-DINO (prompt "cloth") + SAM (prompt "robot gripper")
- Video tracking: XMEM

### 8.4 Metrics

- **MTE (Median Trajectory Error)** [mm]：估计 track 和 ground-truth track 的 median distance
- **Position accuracy δ**：在 thresholds 10, 20, 40, 80, 160 mm 内的 track 百分比
- **Survival rate**：average number of frames until error > 50mm

---

## 9. Quantitative Results 深度解读

### 9.1 Table 1 - 主结果

| Method | MTE (mean) | δ_avg | Survival |
|--------|-----------|-------|----------|
| RAFT-Oracle | 18.324 ± 23.821 | 0.683 | 0.715 |
| DynaGS | 10.924 ± 11.246 | 0.804 | 0.835 |
| MD-Splatting | 3.635 ± 6.235 | 0.847 | 0.887 |
| GNN only | 13.853 ± 12.674 | 0.747 | 0.800 |
| **Cloth-Splatting** | **3.284 ± 3.722** | **0.862** | **0.910** |

关键观察：
1. Cloth-Splatting 比 MD-Splatting MTE 低 9.66%
2. 比 best baseline (MD-Splatting) 快 85%
3. **GNN alone 已经相当 competitive**（13.853 MTE），说明 GNN prior 非常 strong，GS 只是 refine
4. RAFT-Oracle（2D tracking 加 oracle view selection）表现最差，说明 2D 信息严重不足
5. SHORTS 是最难的场景（self-occluded deformation），所有方法都 worst 在 SHORTS

### 9.2 Per-Object 性能分析

SHORTS 性能最差的原因：large self-occluded deformations。当 cloth fold 后看不到部分 surface，GS 渲染也看不到，所以 photometric loss 对这些部分没有 supervision。

**这是 deformable tracking 的 fundamental challenge**——self-occluded 部分只能靠 dynamics model 预测，没法用 vision 修正。

### 9.3 Time Ablation (Figure 4)

Cloth-Splatting converge 比 MD-Splatting 快得多。原因：
- MD-Splatting 从 scratch 学百万 Gaussians
- Cloth-Splatting 有 GNN warm start，只需要 refine
- 只用 4k Gaussians vs MD-Splatting 更多

Initial 26.55s 是 Gaussian initialization 时间，1.36s 是 GNN prediction 时间，剩下的是联合优化。

---

## 10. Ablation Studies 关键 Insight

### 10.1 Table 2

| Ablation | MTE [mm] |
|----------|----------|
| (A1) Only GNN | 21.552 |
| (A2) No GNN | 16.135 |
| (A3) No L_reg | 15.772 |
| (A4) No R_t | 10.799 |
| (A5) 1 view | 16.525 |
| (A6) 2 views | 9.535 |
| (A7) 3 views | 9.004 |
| Full (4 views) | 8.923 |

**关键 insights**：

1. **(A1) Only GNN (21.552)**：比 full (8.923) 差 2.4 倍。GNN alone 累积 error 大，特别是 OOD shape (SHORTS)。GS update 至关重要。

2. **(A2) No GNN (16.135)**：比 only GNN (21.552) 还好。这说明 GS vision supervision 比 GNN dynamics 更 informative，但同时 GNN 给的 prior 能极大加速 converge。两者结合最好。

3. **(A3) No L_reg (15.772)**：去掉 regularization 退化 76.7%。这印证了 inverse rendering 是 ill-posed 问题，没有 prior 的话 cloth 会被拉成不自然形状。

4. **(A4) No R_t (10.799)**：这个 ablation 含义不太清晰，可能指某种 temporal consistency 项。

5. **Views 数量影响**：
   - 1 view (16.525) → 2 views (9.535)：大幅提升
   - 2 → 3 (9.004)：小幅提升
   - 3 → 4 (8.923)：基本无提升
   - 说明 2-3 views 已经 saturate

### 10.2 Table 4 - ITERATIVE vs ROLLOUT

| Method | H | MTE | δ_avg | Survival | Time |
|--------|---|-----|-------|----------|------|
| ITERATIVE | 1 | 0.767 | 0.893 | 0.937 | 53:35 |
| ITERATIVE | 2 | 0.890 | 0.891 | 0.950 | 30:14 |
| ITERATIVE | 4 | 0.819 | 0.893 | 0.947 | 15:53 |
| ITERATIVE | 8 | 1.170 | 0.867 | 0.907 | 9:32 |
| ROLLOUT | 16 | 5.328 | 0.753 | 0.776 | 2:12 |

**Key insight**：H 增加，时间近似 halve，但精度也下降。Trade-off 明显。Tracking 用 ROLLOUT (快)，manipulation 用 ITERATIVE (准)。

为什么 ITERATIVE 更准？因为每 refine 一步就用 refined state 重新 condition GNN，避免了 long-horizon rollout 的 compounding error。

---

## 11. Manipulation Use Case

### 11.1 任务

Half-folding task（Garcia-Camacho et al. 2020 的 benchmark）：
- 给定 pick & place positions
- 优化它们之间的 trajectory
- 用 MPC 闭环 planning

### 11.2 Baselines

- **FIXED**：线性 trajectory between pick & place
- **MPC-OL**：open-loop MPC，不更新 state
- **MPC-CS**：MPC + Cloth-Splatting refine state
- **MPC-ORACLE**：MPC + ground-truth state access

### 11.3 Results (Table 3, 6)

| Method | TOWEL MSE (×10⁻³) | TSHIRT MSE (×10⁻³) |
|--------|------------------|---------------------|
| FIXED | 2.2 ± 0.4 | 2.4 ± 0.4 |
| MPC-OL | 1.8 ± 2.1 | 7.3 ± 5.2 |
| MPC-CS | 0.6 ± 0.6 | 1.2 ± 0.8 |
| MPC-ORACLE | 0.4 ± 0.2 | 0.8 ± 0.5 |

**Insight**：
1. MPC-CS 接近 oracle，说明 state estimation 质量足够好
2. MPC-OL 在 TSHIRT 上方差巨大 (7.3 ± 5.2)，说明 open-loop 在 complex cloth 上不稳定
3. FIXED 表现差，因为单 gripper linear trajectory 无法 fold cloth

### 11.4 Algorithm 2 解析

Pseudo-code 给出了 closed-loop manipulation 算法：
1. 用 $\mathbf{PC}_0$ 初始化 mesh
2. 对每个 timestep：
   - Sample N 个 candidate actions
   - 每个 candidate rollout GNN 预测 H 步未来
   - 算 cost $\mathcal{T}^h(\mathbf{a}_h^n) = ||\hat{\mathbf{M}}_{h+1} - \mathbf{M}_g||_2^2$
   - 选最优 $a^*_{t:t+H}$
   - Execute $a_t^*$
   - Collect new observation $\mathbf{Y}_{t+1}$
   - 用 Cloth-Splatting refine state

这是 standard MPC + state estimation 的 combination，novelty 是用 GS 做 state estimation。

---

## 12. Initialization Robustness (Table 5)

为了测试 robustness to mesh initialization error，他们对 initial mesh 加 augmentation：
- TRANS：xy 翻译 [-0.05, 0.05] m, z 翻译 [-0.003, 0.003] m
- ROT：yaw rotation [-30, +30] degrees
- SCALING：scale coefficient [0.8, 1.2]
- NOISE：multivariate Gaussian noise, variance 0.005
- TRSN：all combined

结果显示 Cloth-Splatting 对 initialization error 相当 robust。特别是当先用 t=0 的 observation refine initial mesh 再 unroll GNN，可以避免某些 catastrophic failure（ROT augmentation 中两个 case 之前 tracking 太差无法 evaluate）。

**Key takeaway**：即使初始 mesh 有 error，GS vision supervision 能纠正回来。这给 real-world deployment 更大 margin。

---

## 13. 与 Related Work 的对比

### 13.1 与 Dynamic 3D Gaussians (DynaGS) 的区别

DynaGS: https://dynamic3dgaussians.github.io
- 单独 model 每个 Gaussian 的 position 和 rotation
- 适合 rigid/mostly-rigid scene tracking
- 对 cloth 这种 large deformation 表现差

### 13.2 与 MD-Splatting 的区别

MD-Splatting: https://md-splatting.github.io
- Extend GS with non-metric → metric projection
- 需要 dense observations (50+ cameras)
- Per-scene optimization from scratch

Cloth-Splatting 优势：
- Sparse views (3-4 cameras)
- GNN warm start
- Mesh-constrained 大幅减少参数

### 13.3 与 DeformGS 的区别

DeformGS (Duisterhof et al. WAFR 2024)：
- 学习 object-centric masks
- 简化 regularization
- 仍需 dense observations

### 13.4 与 PhysGaussian 的区别

PhysGaussian: https://github.com/Xiangyu1Xu/PhysGaussian
- 用 explicitly modeled dynamics (FEM-based physics)
- 不用 visual observation refinement
- 限制 tracking capability

### 13.5 与 Self-supervised Cloth Reconstruction (Huang et al. 2023) 的区别

这是 closest baseline 思想上：
- 也用 action-conditioned dynamics + test-time optimization
- 但是用 point cloud supervision，不用 RGB
- 不利用 texture 信息

Cloth-Splatting 用 RGB supervision 能 leverage texture clues，这在 depth-only 方法里是 missing 的。

---

## 14. Limitations 与 Future Directions

Paper 列出的 limitations：
1. **速度不够 real-time**：2-9 分钟 per trajectory
2. **需要 calibrated multi-camera setup**（但 Dust3r 等方法可缓解）
3. **Static appearance 假设**：shadows 和 lighting changes 会 cause tracking of visual artifacts
4. **需要 occlusion-free initial observation**：限制了对 crumpled cloth 的适用性

我补充几个 potential future directions：
1. **Dynamic appearance modeling**：把 lighting/shadow 作为 latent variable 学进来，类似 NeRF-W
2. **Single-view extension**：用 diffusion prior 或 stronger dynamics prior 补足单 view 信息不足
3. **Real-time optimization**：用更轻量 residual model（不是 MLP，是直接 vertex offsets），更少 iterations
4. **Template-based initialization for crumpled cloth**：结合 Wang et al. 2024 (TRTM) 的方法做 crumpled cloth 初始化

---

## 15. Build Intuition 的核心要点

我想强调几个让你 build intuition 的关键点：

### 15.1 为什么 mesh-constrained GS 比自由 GS 好

自由 GS 是 ill-posed：百万级参数，sparse views，possible 多个 configurations 都能渲染出同样 image。Mesh constraint 把 solution space 限制在 physically plausible 的 cloth configurations 上。这本质上是把 cloth 的 topological prior 烧进了 representation。

### 15.2 为什么 GNN + GS 比单独 GNN 好

GNN 是 forward model，会 accumulate error。GS 提供 backward correction（observation → state）。两者结合是 prediction-update framework 的力量——GNN 给 prior，GS 给 likelihood，Bayes 把它们 fuse。

### 15.3 为什么 residual learning 比直接 fine-tune GNN 好

直接 fine-tune GNN 在 test-time 有几个问题：
1. Vanishing gradients through long rollout
2. Catastrophic forgetting of learned dynamics
3. Per-scene retraining 太慢

Residual MLP 是 lightweight "correction function"——不动 GNN weights，只学一个 time-conditioned offset。这保留了 GNN 的 general dynamics knowledge，只对当前 scene 做少量修正。

### 15.4 整个 framework 的 elegant 之处

这是一个 differentiable simulation + differentiable rendering 的 perfect marriage：
- GNN = differentiable forward dynamics simulator
- GS = differentiable renderer
- Photometric loss = differentiable measurement
- 整个 chain 都 differentiable，gradient 可以从 image pixel 一直 backprop 到 mesh vertex

这种 differentiable pipeline 是 modern robotics + graphics 的 powerful paradigm。

---

## 16. 与 Broader Research 的关联

### 16.1 Differentiable Simulation Literature

- **DiffCloth** (Li et al.)：differentiable cloth simulator
- **Brax, MJX**：differentiable physics simulators
- **SoftGym** (Lin et al. 2021)：benchmark for deformable manipulation https://github.com/Xingyu-Lin/softgym

Cloth-Splatting 跳过了 explicit differentiable simulator，用 GNN 学习 dynamics。Trade-off：GNN 不保证 physical accuracy 但 fast，differentiable physics 准但慢。

### 16.2 Differentiable Rendering Literature

- **NeRF** https://arxiv.org/abs/2003.08934
- **D-NeRF, Nerfies, Neural Scene Flow Fields**：dynamic scene extensions
- **3D GS** https://repo-splatting.doc.city.ac.uk

Cloth-Splatting 是 GS 在 robotics 上的 application，显示了 GS 的 differentiability + speed 优势。

### 16.3 Cloth Manipulation Literature

- **SpeedFolding** (Avigal et al. 2022) https://github.com/berkeleyautomation/speedfolding
- **AdaFold** (Longhini et al. 2024) https://arxiv.org/abs/2403.06210
- **SoftGym**

Cloth-Splatting 提供了 closed-loop manipulation 的 state estimation 基础设施。

### 16.4 Visual Tracking Literature

- **CoTracker** https://arxiv.org/abs/2307.07635
- **TAP-Vid** benchmark
- **RAFT** https://arxiv.org/abs/2003.12039

这些都是 2D tracking，Cloth-Splatting 是 3D tracking。

---

## 17. 我的 Critical 评价

### 17.1 Strengths

1. **Conceptual elegance**：Bayesian filtering framework 干净，prediction-update 分离明确
2. **Strong empirical results**：57% more accurate, 85% faster than baselines
3. **Sim-to-real transfer demonstrated**：real-world Franka experiment
4. **Practical applicability**：closed-loop manipulation experiment 展示了实用价值
5. **Ablation thorough**：每个 component 都 ablate 了

### 17.2 Weaknesses

1. **Speed limitation**：即使快了 85%，仍然 minutes per trajectory，远非 real-time
2. **Multi-camera requirement**：real-world deployment 受限
3. **Static appearance assumption**：long-term deployment 中 lighting 会变
4. **Occlusion-free initialization**：crumpled cloth 场景受限
5. **GNN training data dependence**：需要 simulator-generated training data
6. **No comparison to NeRF-based methods**：只比 GS-based baselines

### 17.3 Future Work I'd Like to See

1. Single-view extension with diffusion prior
2. Real-time implementation with truncated optimization
3. Dynamic lighting modeling
4. Crumpled cloth initialization
5. Comparison with differentiable physics simulators (DiffCloth)
6. Bimanual manipulation experiments
7. Long-horizon tasks (multi-step folding)

---

## 18. 总结

Cloth-Splatting 的核心 contribution 是把 Bayesian filtering framework 应用到 cloth state estimation，关键是 mesh-constrained GS 提供了 differentiable state→image mapping。这个 design choice 把 GS 的 expressive power 和 mesh 的 structural prior 结合起来，使得 sparse-view real-time-ish 3D cloth tracking 成为可能。

对 robotics 社区的意义：展示了 differentiable rendering 在 state estimation 上的潜力。这种 pattern 可以推广到其他 deformable objects（rope, fluid, soft body）。

对 graphics 社区的意义：展示了 GS 不只是 novel view synthesis 的工具，也能作为 inverse problem 的 differentiable forward model。这是 GS 应用范围的扩展。

References:
- Paper project page: https://kthrpl.github.io/cloth-splatting
- 3D Gaussian Splatting: https://repo-splatting.doc.city.ac.uk / https://github.com/graphdeco-inria/gaussian-splatting
- GNS (Sanchez-Gonzalez et al. 2020): https://arxiv.org/abs/2002.09405
- MD-Splatting: https://arxiv.org/abs/2312.00583 / https://md-splatting.github.io
- DynaGS: https://dynamic3dgaussians.github.io
- DeformGS: https://bduisterhoft.github.io/deformgs/
- PhysGaussian: https://github.com/Xiangyu1Xu/PhysGaussian
- Self-supervised Cloth Reconstruction (Huang et al. 2023): https://arxiv.org/abs/2303.00149
- SoftGym: https://github.com/Xingyu-Lin/softgym
- SpeedFolding: https://github.com/berkeleyautomation/speedfolding
- AdaFold: https://arxiv.org/abs/2403.06210
- RAFT: https://arxiv.org/abs/2003.12039
- CoTracker: https://arxiv.org/abs/2307.07635
- TAP-VID: https://tapvid.github.io
- Dust3r: https://github.com/naver/dust3r
- NeRF: https://arxiv.org/abs/2003.08934
- SAM: https://github.com/facebookresearch/segment-anything
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- XMEM: https://github.com/hkchengrex/XMem
- Lips et al. 2024 (synthetic cloth): https://arxiv.org/abs/2401.01734
- RoMa (rotation toolbox): https://github.com/naver/roma
- Garcia-Camacho et al. 2020 (bimanual cloth benchmark): https://arxiv.org/abs/1910.01745
- Wang et al. 2024 (TRTM): https://arxiv.org/abs/2410.19312
- NVIDIA Flex: https://developer.nvidia.com/flex
