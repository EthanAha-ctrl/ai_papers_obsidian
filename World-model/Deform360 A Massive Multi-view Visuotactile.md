---
source_pdf: Deform360 A Massive Multi-view Visuotactile.pdf
paper_sha256: 3f23966cb31f3614a8b27aba9ae558ef4684e812dd86ef6cd072d1788c6faf95
processed_at: '2026-08-18T04:58:08-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Deform360 用人话讲

## 一句话说清楚这篇 paper 在干嘛

想象你在教机器人叠衣服、捏海绵、拧绳子。机器人要干活，得先在脑子里"想象"这个物体被我推一下会变成什么样——这就是 **world model**。但 deformable object 太难了：衣服会皱、海绵会塌、绳子会缠，而且 gripper 一抓住，接触点就被挡住了，看不见。

这篇 paper 干了三件事：
1. 搭了一个 super expensive 的 capture rig：41 个 camera 围一圈 + 两个带 tactile sensor 的 gripper，拍了 198 个日常物体、1980 个 interaction、215 小时视频
2. 写了一个 pipeline 把这些 raw video 转成 dense 3D particle tracking annotation——每个物体表面有上千个点，每帧都知道它们在哪
3. 用这个 data 拿当前最强的 2D video model（NVIDIA 的 Cosmos）和 3D particle model（PhysTwin、PGND、ParticleFormer）打擂台，看谁更厉害

核心 finding 是一个 trade-off：**数据少的时候，懂物理的 3D 模型赢；数据多要泛化的时候，靠 internet pretraining 的 video model 赢**。

Project: https://deform360.lhy.xyz

---

## 为什么要花这么大代价搞 41 个 camera？

你去摸一个 plush toy 的瞬间，你的手指把 toy 那一块全挡住了。从单个 camera 看，你根本不知道 toy 被捏成了什么形状。这是 deformable object manipulation 的核心痛点——**contact-induced local deformation 几乎总是被 occlude**。

作者的解法很 brute force：41 个 camera 围一圈，总有几个角度能"偷看"到 contact region。这听起来简单，但之前没人这么干，因为：
- 41 个 camera 的同步、calibration 本身就是工程噩梦
- 数据量爆炸：41 views × 30 FPS × 10 秒 = 12300 帧/episode，1980 episodes 就是 23.3M 帧
- Storage 和 3DGS reconstruction 的 compute 都不是小事

但作者证明这是必要的。任何 < 20 view 的 setup 都会有大块区域长时间 invisible，导致 3D reconstruction 和 tracking 在 contact region 彻底失败。之前的 dataset（Robo360 有 86 views 但没 tactile，PhysTwin 只有 3 views 11 objects）都缺一块。

---

## Tactile sensor 到底解决什么问题？

这是这篇 paper 最 clever 的地方。

想象你用 gripper 抓住一块布往上提。从 visual 看，布被提起来了，但 gripper 和布接触的那一小块完全看不见。这时候你的 3D particle tracker 会怎么办？它会"猜"——根据周围 particle 的 motion 推断 contact region 的 particle 在哪。但这个猜很容易错，因为布在 gripper 下面可能滑了、皱了、扭转了。

Tactile sensor 给你一个 **ground-truth anchor**：它告诉你 gripper 的 finger 在哪个 3D 位置、施加了多少 normal pressure。这就像你在黑暗中摸东西，虽然看不见，但手能感觉到。

所以作者把 tactile 不是当成一个 input modality，而是当成一个 **constraint**——在 contact region，particle 的 velocity 必须跟 tactile sensor 测出来的一致。这个 constraint 是 soft 的（因为 tactile 只测 normal pressure，测不到 tangential slip），但它足以把 visual tracker 在 occlusion 下"拉回正轨"。

Quantitative evidence：加上 tactile optimization 后，warped point cloud 的 Chamfer distance 从 $1.41 \times 10^{-4} m^2$ 降到 $2.71 \times 10^{-5} m^2$，5 倍提升。这不是 marginal improvement，是 qualitative difference。

参考 tactile gripper 的设计：UMI (Universal Manipulation Interface) https://universal-manipulation-interface.github.io/

---

## Annotation pipeline 为什么这么复杂？

你有了 41 个 camera 的视频，怎么把它变成 "每个 particle 在每一帧的 3D 位置"？这是整个 paper 最 technically challenging 的部分。

### Naive approach 的问题

最 naive 的做法是直接用 4D Gaussian Splatting（Dynamic 3DGS）——让每个 Gaussian 自己动，track 它的 trajectory。但这有个根本问题：3DGS 是为 **rendering** 设计的，不是为 **tracking** 设计的。每个 Gaussian 独立 motion，如果两个相邻 Gaussian 朝不同方向飘，render 出来的图可能很好看，但 particle trajectory 物理上不合理——比如布被拉了一下，应该整体跟着动，但 Gaussian 可能"撕裂"。

### Deform360 的解法：decouple and fuse

作者把这个 problem 拆成三步：

**Step 1: Per-frame 3DGS reconstruction**
每一帧独立 reconstruct 一个 3DGS model。这一步只管 geometry——每一帧的 3D shape 对不对，不管 temporal consistency。

**Step 2: 2D tracking with CoTracker3**
在每个 camera view 上用 CoTracker3（Meta 的 point tracker）track 1600 个点。这一步只管 temporal correspondence——同一个点在 frame 1 到 frame 10 是不是 track 对了，不管 3D geometry。

**Step 3: 3D lifting + physics-informed optimization**
把 2D track 用 3DGS 的 depth map lift 成 3D position，然后用一个 optimization 把多 view 的 estimate fuse 起来，同时 enforce 几个 physical constraint：
- **Shape loss**：particle 不能飘离物体表面
- **ARAP (As-Rigid-As-Possible)**：相邻 particle 之间的距离应该保持不变（small deformation 下）
- **Laplacian smoothness**：velocity field 在空间上 smooth
- **Tactile consistency**：contact region 的 particle velocity 要跟 tactile sensor 一致

这个设计的 intuition 是：每个 module（3DGS、CoTracker3、physics constraint）单独看都不完美，但组合起来互补。3DGS 给 geometry，CoTracker3 给 temporal correspondence，physics constraint 给 physical plausibility，tactile 给 contact ground-truth。这是典型的 modular robotics pipeline 思路。

---

## Benchmark 的三个 level 是什么意思？

这是 paper 最有 insight 的部分。作者定义了三个 generalization 难度：

### Level 1: Per-episode（Frame Generalization）
同一个 episode 内，用前半段预测后半段。测试的是"给你看 5 秒，你能不能预测第 6 秒"。

结果：**PhysTwin（physics simulator）完胜**。因为数据量太少，learning-based model 没法学。Physics model 不需要学，它直接用 spring-mass equation 算。

### Level 2: Multi-episode（Episode Generalization）
同一个物体，用 5 个 episode 训练，预测第 6 个 episode。测试的是"给你看这个物体被各种捏抓推，你能不能预测一个新的 interaction"。

结果：**ParticleFormer 赢**。因为数据量稍微多了，3D structural prior 帮助 generalize 到新 configuration。Video model（Cosmos）在 reconstruction 上更好（preserve texture），但在 future prediction 上不如 3D model——说明它还没学到 underlying dynamics，只是在 memorize visual pattern。

### Level 3: Multi-object（Object Generalization / Zero-shot）
用 150 个物体训练，预测剩下 48 个没见过的物体。测试的是"给你看一堆布和海绵，你能不能预测一个新玩具被捏会怎样"。

结果：**Cosmos 赢**。因为 internet-scale pretraining 让它有 general visual prior，能 generalize 到新 object appearance。3D model 没有 pretraining，只能 rely on training data 里的 pattern，泛化能力弱。

### 这个 trade-off 为什么重要？

这三个 level 画出了一个非常清晰的 picture：

- **Low data + specific object** → physics-based 3D model 赢（structural prior 是 free lunch）
- **Medium data + specific object** → learning-based 3D model 赢（structural prior + learned dynamics）
- **High data + diverse objects** → 2D video model 赢（scalability + pretraining）

这就是 paper 标题说的 "structural priors vs scalability" trade-off。这对未来 robotics foundation model 的设计有直接 implication：你需要一个 hybrid——3D structural prior 作为 inductive bias，2D pretraining 作为 scalable feature learner。这是 LeCun 的 JEPA、Hafner 的 DreamerV3、以及 PointWorld 都在朝向的方向，Deform360 第一次为 deformable object 给出了 empirical evidence。

---

## 那个 "Cosmos 不 follow action" 的 failure mode 是什么意思？

这是 paper 里最 subtle 也最 interesting 的 observation。

在 zero-shot object generalization 下，Cosmos 生成的视频看起来 physically reasonable——布会皱、海绵会塌——但如果你仔细对比，它生成的 motion 和你给它的 robot action **不匹配**。比如你让它预测"gripper 往左拉"，它可能生成"gripper 往右拉"的视频。

这说明 Cosmos 学到的不是 "action → motion" 的因果 mapping，而是 "object appearance → plausible motion distribution" 的 association。它知道布会被拉扯、海绵会被捏，但不知道**这个具体 action 会导致什么具体 motion**。

这是 LLM-style world model 的本质问题：当你训练 $p(\text{future} | \text{past}, \text{action})$，如果 action 和 past 在训练分布外，model 会 fallback 到 $p(\text{future} | \text{past})$——即基于 visual prior 的 association，忽略 action。这是为什么 3D structural prior 在 OOD 下更可靠——它不依赖 visual similarity，而是依赖 physical law。

---

## 为什么 2D video model 没法做 MPC planning？

paper 在 Section 5.4 提了一句但没展开，我觉得这是整个 paper 最 deep 的 insight。

MPC（Model Predictive Control）的工作原理是：给定 goal state，搜索一系列 action，让 world model 预测的 future state 尽量接近 goal。这需要你有一个 **reward function** 来衡量"预测的 future state 离 goal 有多近"。

对 3D particle model，这个 reward 天然 well-defined：用 Chamfer distance between predicted particle positions 和 goal particle positions。几何上很直观。

对 2D video model，这个 reward 极难定义。你有一个 generated video，你怎么衡量"这个视频里的物体离 goal state 有多近"？你没法直接从 pixel 比较，因为视角、光照、texture 都会变。你可以 train 一个 reward model，但那又引入新的 failure mode。

所以 3D representation 在 robotics planning 上有结构性优势——它让 reward design 变得 tractable。这是为什么作者只 deploy PhysTwin 做 real-world MPC，没 deploy Cosmos。

---

## 这个 dataset 对未来研究意味着什么？

### 1. 3D world model 需要 pretraining
Paper 明确指出 3D model 在 zero-shot 下输给 video model，因为后者有 internet-scale pretraining。PointWorld [28] 是 recent attempt 给 3D world model 做 pretraining，但还没 open source。这是一个 open problem：怎么给 3D particle model 做 large-scale pretraining？

### 2. Visuotactile fusion 是 promising direction
Paper 证明 tactile 作为 ground-truth constraint 能 5× improve tracking。但 tactile 只测 normal pressure，测不到 tangential slip。Future work 应该用 richer tactile sensor（比如 DIGIT、GelSight）来 capture slip，这样能 model 更复杂的 contact physics。

参考 GelSight: https://github.com/gelsightinc/gelsight

### 3. 2D + 3D hybrid 是 future
最 promising 的方向是把 2D video model 的 scalability 和 3D structural prior 的 physical plausibility 结合。DINO-WM [90] 是一个 attempt——用 pretrained visual feature + 3D dynamics model。但还没人在 deformable object 上做这件事。Deform360 提供了 perfect benchmark。

### 4. Action-conditioned video model 的 action alignment
Cosmos 的 failure mode（不 follow action）说明当前 video world model 的 action conditioning 不够强。需要更好的 action representation 或 training paradigm 来 ensure action fidelity。这关系到 video model 能不能真正用于 robotics planning。

---

## 几个我觉得值得深挖的技术细节

### 1. ARAP loss 的物理含义
ARAP (As-Rigid-As-Possible) 是 deformation graph 的经典 constraint。它的物理直觉是：对于 small deformation，物体的局部结构应该保持 rigid——相邻点之间的距离不变。这在 thin-shell（布、纸）和 soft body（海绵、plush toy）上都成立，但对 highly plastic material（橡皮泥、clay）会失败，因为 plastic deformation 允许局部 distance 改变。这是 paper 的 limitation 之一。

参考原论文：Sorkine-Hornung & Alexa, "As-Rigid-As-Possible Surface Modeling", Symposium on Geometry Processing 2007. https://igl.ethz.ch/projects/ARAP/

### 2. 2D-to-3D lifting 公式的几何直觉
$$\mathbf{P}_{n,t} = \mathbf{E}_n^{-1} \mathbf{D}_{n,t}(\mathbf{u}_{n,t}) \mathbf{K}_n^{-1} \tilde{\mathbf{u}}_{n,t}$$

这个公式的几何含义：
- $\tilde{\mathbf{u}}_{n,t} = [u, v, 1]^T$：2D pixel 的 homogeneous coordinate
- $\mathbf{K}_n^{-1} \tilde{\mathbf{u}}_{n,t}$：把 pixel 转成 camera frame 下的 ray direction（normalized）
- $\mathbf{D}_{n,t}(\mathbf{u}_{n,t})$：沿这条 ray 的 depth（标量）
- 两者相乘得到 camera frame 下的 3D point
- $\mathbf{E}_n^{-1}$：把 camera frame 转成 world frame

这本质上是 pinhole camera model 的逆运算，用 depth 来 disambiguate ray 上的哪个点。

### 3. Tactile loss 的 no-slip assumption
$$\mathcal{L}_{tactile} = \frac{1}{|S_{tactile}|} \sum_{i \in S_{tactile}} \|\mathbf{v}_{i,t} - \mathbf{v}_{sensor}\|^2$$

这个 loss 假设 contact region 没有 slip——particle velocity 等于 sensor velocity。这是 soft constraint，但实际中 slip 是常见的。作者在 data collection 时主动避免 slip（通过 protocol），但这是 limitation。Future work 应该用 6-axis force/torque sensor 或 GelSight 这种能测 tangential force 的 sensor 来 capture slip。

### 4. Multi-episode 下 PhysTwin 为什么被排除？
PhysTwin 是 optimization-based——它需要对每个 episode 做 system identification（fit spring stiffness、damping 等 physical parameter）。这没法 zero-shot 跨 episode，因为新 episode 的 initial configuration 不同，需要重新 optimize。这是 physics-based model 的结构性 limitation：它们 generalize across configuration 但不 generalize across episode without re-fitting。

---

## 和 Karpathy 你自己工作的 connection

你之前提过 world model 是 "model that predicts future given past and action"——这和 Deform360 的 formulation 完全一致。Deform360 的 contribution 是把这个 formulation 落地到 deformable object 这个 hardest case，并且用 visuotactile data 提供了 contact region 的 ground-truth。

你关注的 "LLM as world model" 争议（simulator vs associative memory）在这篇 paper 里也有 empirical evidence：Cosmos 在 zero-shot 下生成 physically plausible 但 action-misaligned motion，这正是 "associative memory" 的表现——它 associate object appearance with plausible motion，但不 simulate action-causality。

Deform360 的 benchmark 暗示：如果要让 video model 真正成为 world model（而不是 video generator），需要更强的 action conditioning 机制。这可能是 architectural innovation（比如 action-conditioned attention）、training paradigm innovation（比如 reinforcement learning on action fidelity）、或 hybrid approach（3D prior + 2D generation）。

---

## 最后的 takeaway

这篇 paper 不是一个 fancy algorithm paper，是一个 **infrastructure + benchmark paper**。它的价值不在提出新 method，而在：
1. 提供了 deformable object world modeling 的第一个大规模 real-world benchmark
2. Systematic 对比了 2D vs 3D paradigm，揭示了 structural prior vs scalability 的 trade-off
3. 证明了 tactile 作为 contact ground-truth 的价值
4. 给未来 robotics foundation model 设计提供了 empirical guidance

这种 paper 在 ML 里 value 很高——它不 solve 一个 problem，但 define 了一个 problem space。Deform360 之于 deformable world modeling，就像 ImageNet 之于 image classification、MuJoCo 之于 RL——它让后续工作有了一个 fair comparison 的 ground。

如果你要 build 一个 generalist robot policy，Deform360 告诉你：你需要 3D structural prior（不然 contact physics 建模不准），你需要 tactile sensing（不然 contact region 是 black box），你需要 large-scale pretraining（不然 zero-shot 泛化不行），你需要 action-conditioned training（不然 model 不 follow action）。这四条是 future robotics foundation model 的 checklist。

---

# Deform360: A Massive Multi-view Visuotactile Dataset for Deformable World Models

## 1. 论文的核心叙事

这篇 paper 的核心论点是：当前 deformable object 的 world modeling 有两大 paradigm——2D video model（在 pixel space 中预测 dynamics）和 3D particle model（在显式几何空间中建模 dynamics）——但一直没有一个 fair、大规模、real-world 的 benchmark 去对比这两者各自的 strength 和 weakness。作者 build 了一个 41-camera surround-view + bimanual tactile gripper 的 capture rig，采集了 198 个日常物体、1980 个 interaction sequences、共 215.7 小时、23.3M 帧的数据，然后用一个 markerless visuotactile 3D tracking pipeline 把这些 raw videos 转成 dense particle annotation，进而系统对比 SOTA 2D video model（Cosmos-Predict 2.5 [51]）和 3D particle model（PhysTwin [30]、PGND [86]、ParticleFormer [27]）。

核心 empirical finding 是一个 "structural priors vs scalability" 的 trade-off：在 low-data regime（per-episode、multi-episode）下，3D physics-based model 因为有 spring-mass、ARAP 这类 structural prior 而更准确；但在 zero-shot object generalization 下，pretrained video model 因为 internet-scale pretraining 而 generalize 得更好——虽然它有时候 "ignores" 给定的 action 而生成 physically plausible 的 motion。这其实是一个非常 clean 的 scaling law 视角下的 3D prior vs 2D data 之争。

Project website: https://deform360.lhy.xyz

---

## 2. Dataset 设计的几个关键直觉

### 2.1 为什么是 41 cameras？

Deformable object manipulation 的核心困难是 **contact-induced local deformation**——gripper 抓住的瞬间，contact region 被 end-effector 完全遮挡。任何 single-view 或 sparse-view 的 setup 都会丢掉这部分信息。41 个 surround cameras 提供 360° 覆盖，至少有一个视角能看到 contact region 的 surface deformation。同时多视角使得后续 3DGS reconstruction、2D-to-3D lifting 可以在 occlusion 下仍然 recover 出 3D particle position。

### 2.2 为什么是 tactile + visual？

这里的关键 intuition 是：visual modality 提供 global geometry，tactile modality 提供 **contact-ground-truth**——gripper 在哪里、施加了多少 normal pressure。这个 pairing 解决了一个长期问题：在 gripper 完全 occlude object 的瞬间，纯 visual tracking 的 particle 会"漂走"，因为没有任何 visual cue 来 anchor 它们。Tactile signal 在那一刻成为唯一的 ground-truth。论文的 Fig. 4 展示了加上 tactile optimization 后 warped point cloud 的 Chamfer distance 从 $1.41 \times 10^{-4} m^2$ 降到 $2.71 \times 10^{-5} m^2$，5× improvement。

### 2.3 Object taxonomy：1D / 2D / 3D

198 个物体分成三类，按 topological dimension：
- **1D deformables**（28 个）：rope, cable, wire——一维远超另外两维
- **2D deformables**（98 个）：cloth, fabric, paper, bag——thin-shell
- **3D volumetric**（72 个）：plush toy, sponge, squeezable——有体积

这个划分对应不同的 physical model：1D 用 elastic rod theory（DDER [9]），2D 用 cloth simulation（DifCloth [44]），3D 用 spring-mass 或 MPM（EMPM [10]）。Deform360 横跨全部三类，这是它的核心 contribution 之一。

---

## 3. Annotation Pipeline 的技术细节

整个 pipeline 的设计哲学是 **decouple per-frame geometry recovery from temporal tracking**。3DGS [34, 49, 74] 的 original formulation 是为 rendering 优化的，每个 Gaussian 独立 motion——这意味着 tracking 出来的 trajectory 缺乏 temporal consistency。所以作者把 geometry 重建和 tracking 分成两步，再用一个 physics-informed optimization 把它们 fuse。

### 3.1 Per-frame 3DGS Reconstruction

每个 Gaussian $k$ 由以下参数化：
- $\mu_k \in \mathbb{R}^3$：mean position
- $\Sigma_k = R_k S_k S_k^T R_k^T$：covariance，分解为 rotation matrix $R_k$ 和 scaling matrix $S_k$，保证 positive semi-definite
- $\alpha_k \in [0, 1]$：opacity
- $c_k$：spherical harmonic coefficients 表示 view-dependent color

Loss function 是 3DGS 标准 formulation：
$$\mathcal{L}_{gs} = (1 - \lambda_{gs}) \mathcal{L}_1 + \lambda_{gs} \mathcal{L}_{SSIM}$$
其中 $\lambda_{gs} = 0.2$，$\mathcal{L}_1$ 是 rendered image 和 segmented object frame $S_n$ 之间的 mean absolute error。Object mask $M_{obj,n} \in \{0,1\}^{T \times H \times W}$ 用 SAM-style [8, 61] segmentation model 生成，然后用 $\mathbf{S}_n = \mathbf{V}_n \odot M_{obj,n}$ 把 background 抠掉。

### 3.2 Markerless 2D Tracking + 3D Lifting

用 CoTracker3 [33] 在每个 view 上 track 最多 $M=1600$ 个 mask-filtered grid points，提供 persistent 2D trajectory $\mathbf{u}_{n,t} \in \mathbb{R}^{M \times 2}$。tracking 在 15-frame clips 上做，stride 5，处理 self-occlusion。

然后是关键的 2D-to-3D back-projection。给定一个 2D track $\mathbf{u}_{n,t}$，它的 3D 位置由如下公式给出：
$$\mathbf{P}_{n,t} = \mathbf{E}_n^{-1} \mathbf{D}_{n,t}(\mathbf{u}_{n,t}) \mathbf{K}_n^{-1} \tilde{\mathbf{u}}_{n,t}$$

变量解释：
- $\mathbf{E}_n \in \mathbb{R}^{4 \times 4}$：camera $n$ 的 extrinsic matrix（world-to-camera transform），$\mathbf{E}_n^{-1}$ 是 camera-to-world
- $\mathbf{K}_n \in \mathbb{R}^{3 \times 3}$：intrinsic matrix，$\mathbf{K}_n^{-1}$ 把 pixel coordinate 转成 normalized camera coordinate
- $\mathbf{D}_{n,t}(\mathbf{u}_{n,t})$：在 2D track $\mathbf{u}_{n,t}$ 处从 3DGS rendered depth map 采样的 depth value（标量）
- $\tilde{\mathbf{u}}_{n,t}$：$\mathbf{u}_{n,t}$ 的 homogeneous coordinate $\in \mathbb{R}^3 = [u, v, 1]^T$

直觉：先 $\mathbf{K}_n^{-1} \tilde{\mathbf{u}}$ 把 pixel 转成 camera frame 下的 ray direction，乘以 depth 得到 camera frame 下的 3D point，再用 $\mathbf{E}_n^{-1}$ 转到 world frame。

### 3.3 Physics-informed Tracking Optimization

单 view 的 3D lifting 由于 occlusion 和 depth error 会 noisy。论文用 RANSAC 把多 view 估计 fuse 成一个 velocity field $\mathbf{v}_t$，然后用 gradient descent 最小化一个 physics-informed objective：

$$\mathcal{L}_{track} = \mathcal{L}_{shape} + \lambda_{local} \mathcal{L}_{local} + \lambda_{lap} \mathcal{L}_{lap} + \lambda_{tactile} \mathcal{L}_{tactile}$$

权重 $\lambda_{local} = 20.0$，$\lambda_{lap} = 0.1$，$\lambda_{tactile} = 1.0$。

四个 loss 的具体形式和直觉：

**(a) Shape loss $\mathcal{L}_{shape}$**：bidirectional Chamfer distance between predicted next-step positions 和 target 3DGS point cloud $\mathbf{P}_{t+1}^{target}$。直觉：保证 particle 在 next step 仍然落在 object surface 上，不会"飘"出去。注意：被 tactile sensor 影响的 particle（contact set $S_{tactile}$）被排除在外，避免和 contact constraint 冲突。

**(b) Local rigidity (ARAP) loss $\mathcal{L}_{local}$**：
$$\mathcal{L}_{local} = \frac{1}{M} \sum_i \sum_{j \in \mathcal{N}(i)} \left( \|\mathbf{P}_{i,t+1} - \mathbf{P}_{j,t+1}\| - l_{ij}^{rest} \right)^2$$
其中 $l_{ij}^{rest}$ 是 particle $i$ 和 $j$ 之间的 rest length（precomputed），$\mathcal{N}(i)$ 是 $i$ 的 neighbor set。直觉：ARAP（As-Rigid-As-Possible）约束——相邻 particle 之间的 distance 在 small deformation 下应该保持不变。这是 thin-shell 和 soft body deformation 的经典 prior [Sorkine-Hornung & Alexa, "As-Rigid-As-Possible Surface Modeling"]。

**(c) Laplacian regularization $\mathcal{L}_{lap}$**：
$$\mathcal{L}_{lap} = \frac{1}{M} \sum_i \left\| \mathbf{v}_{i,t} - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{v}_{j,t} \right\|^2$$

直觉：每个 particle 的 velocity 应该接近它邻居 velocity 的平均——即 velocity field 在空间上 smooth。这个对应 Laplacian smoothing 的 discrete form，相当于隐式 viscosity prior。

**(d) Tactile loss $\mathcal{L}_{tactile}$**：
$$\mathcal{L}_{tactile} = \frac{1}{|S_{tactile}|} \sum_{i \in S_{tactile}} \|\mathbf{v}_{i,t} - \mathbf{v}_{sensor}\|^2$$

其中 $S_{tactile}$ 是 contact set——在 activated taxel 的 radius $r$ 范围内的 object particle 集合。$\mathbf{v}_{sensor}$ 是 tactile sensor 提供的 velocity estimate。直觉：在 contact region，particle velocity 应该 follow tactile signal。这是 **soft constraint**，因为 tactile sensor 只 measure normal-axis pressure，无法 observe tangential slip——所以这个 loss 假设 localized no-slip。Slip 在 data collection 时通过 protocol 主动避免。

### 3.4 关键设计决策的解读

把这个 pipeline 拆开看，它的设计哲学和 PhysGaussian [79]、DeformGS [19] 这些方法的根本区别在于：后者把 3DGS 的 dynamic deformation 和 temporal tracking 耦合在一起——一个 Gaussian 既要 render 好又要 track 好——导致 optimization objective 互相 conflict。Deform360 显式 decouple 这两个 objective：3DGS 只管 per-frame geometry，2D tracker 只管 temporal correspondence，最后用一个 lightweight optimization 把它们 fuse。这种 "modular pipeline" 设计实际上是 robotics + graphics 交叉领域的常见 trick（参考 BundleSDF [72] 也是类似思路）。

---

## 4. Benchmark 设计：三个 generalization level

这是 paper 最有意思的部分。作者定义了三个 generalization 难度递增的 setting：

### 4.1 Per-episode（Frame Generalization）

在同一个 episode 内，用前 $T_{train}$ 帧 train，predict 剩余的 $T - T_{train}$ 帧。这里测试的是 "given 一个具体 object + 具体 interaction，模型能否 generalize 到这个 interaction 中的 unseen continuation"。Table 3 的结果：

| Method | CD ↓ (pred) | Track Err ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|
| PGND | 0.073 | 0.073 | 25.296 | 0.963 |
| ParticleFormer | 0.044 | 0.041 | 26.288 | 0.964 |
| PhysTwin | 0.014 | 0.025 | 26.574 | 0.964 |

PhysTwin（physics-based differentiable simulator）显著优于 learning-based method，因为每个 episode 内数据量太少，learning-based method 没法 infer 出 physical parameter。Cosmos 不参与这个 setting，因为单 episode 数据量 insufficient for post-training。

### 4.2 Multi-episode（Episode Generalization）

对于给定 object，用 $E_{train}$ 个 episode 训练，test 在 unseen episode 上。Table 4：

| Method | CD ↓ (pred) | Track Err ↓ | PSNR ↑ |
|---|---|---|---|
| PGND | 0.130 | 0.144 | 23.788 |
| ParticleFormer | 0.051 | 0.079 | 25.203 |
| Cosmos | – | – | 24.950 |

这里出现 trade-off：Cosmos 在 reconstruction 上 PSNR 最高（27.748），因为它直接在 2D space 训练，preserve texture 更好；但 ParticleFormer 在 future prediction 上 PSNR 25.203 > Cosmos 24.950——意思是当数据量稍微多一些（多 episode），3D structural prior 帮助 generalize 到 unseen configuration，而 video model 还没学到 underlying dynamics。PhysTwin 被排除，因为它需要 per-episode system identification，没法 zero-shot 跨 episode。

### 4.3 Multi-object（Object Generalization / Zero-shot）

训练在 $O_{train}$ 个 object 上，test 在完全 unseen object 上。Table 5：

| Method | CD ↓ | Err ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|
| PGND | 0.429 | 0.320 | 22.049 | 0.969 |
| ParticleFormer | 0.038 | 0.048 | 23.312 | 0.969 |
| Cosmos | – | – | 25.042 | 0.958 |

在 zero-shot object generalization 下，Cosmos PSNR 25.042 最高——这是 internet-scale pretraining 的胜利。但作者观察到一个有趣的 failure mode：Cosmos 在 long-horizon prediction 中常常 **不严格 follow 给定的 action**——但它生成的 motion 仍然 physically reasonable。这暗示 2D video model 学到的是 "what motion is plausible for this kind of object"，而不是 "what motion does this specific action cause"。这是 LeCun [37] 和 Wang [71] 等人讨论的 "world model 是 simulator 还是 associative memory" 之争的实证。

---

## 5. Visuotactile Contact Prediction

这个 sub-task 单独看很有意思。作者 discretize tactile data 成 binary contact/no-contact signal，train 一个 transformer encoder 把 visual stream + robot action 映射到 expected contact signal。准确率 88.67%（random baseline 50.31%），F1 = 0.8909。

直觉解读：这意味着 visual signal 中**已经包含**了大量关于 contact 的信息——fold 时的褶皱、stretch 时的 surface strain、poke 时的局部 indentation 都是 contact event 的 visual proxy。Tactile signal 在 contact region 提供 ground-truth，而 visual signal 在 non-contact region 提供 global context。这个 cross-modal prediction task 实际上是测试 dataset 是否 capture 到了 visuotactile coupling——如果 model 学不会，说明 dataset 中 visual 和 tactile 没有 correlate。88.67% 说明 correlate 得不错，但 11% 的 error 也提示 visual-only contact prediction 仍然 ambiguous（self-occlusion 时无法判断 contact 是否真的发生）。

---

## 6. Real-World Robot Planning with MPC

最后作者用 PhysTwin（3D particle model）在 **完全不同的 robot setup（xArm）和不同 lab** 上做 zero-shot MPC planning。这意味着 Deform360 训出来的 model 可以 transfer 到新的 embodiment 和 environment。

为什么不 deploy Cosmos？作者给出两个理由：
1. Video model 对 appearance difference 跨 environment 更敏感，post-training scale 不够 support OOD visual generalization
2. Designing reward function on generated video 困难——3D model 可以直接用 Chamfer distance 作为 reward

这第二个理由其实是很深的 insight：3D representation 让 reward design 变得 tractable，因为几何 metric（Chamfer, Hausdorff）天然 well-defined；而 2D video 缺乏 explicit 3D structure，所以无法直接定义 "object 在 goal state" 这个 reward。这是 3D world model 在 robotics planning 上的结构性优势。

---

## 7. Limitations 和 Open Questions

作者承认：
1. **Heavy self-occlusion**：如果大区域长时间 invisible，tracking quality 仍然下降
2. **Highly plastic material**：violate ARAP 和 Laplacian smoothness assumption
3. **Visible slip**：tactile no-slip regularizer 会 over-constrain
4. **Tactile 只 measure normal pressure**：无法 detect micro-slip

这些都是 future work 方向。从 model 角度看，更深层的问题是：3D particle model 没有 internet-scale pretraining（PointWorld [28] 是 recent attempt），所以 zero-shot 不如 2D video model；而 2D video model 在 long-horizon 下 action 不严格 follow。Future direction 是把两者结合——用 3D structural prior 做 inductive bias，但用 2D pretraining 做 scalable feature learning。这其实是 DINO-WM [90]、V-JEPA [Wang et al., 70] 这条线想做的事。

---

## 8. 与相关工作的位置

Deform360 在 deformable object dataset 生态中的位置：

| Dataset | 主打 | 缺陷 |
|---|---|---|
| ClothSim2Real [6] | Cloth sim-to-real | 1 view, 3 objects |
| DDER [9] | Cable, elastic rod | 5 views, 11 objects |
| Robo360 [47] | 86 views, multi-material | 无 tactile, 无 annotation |
| PokeFlex [52] | Volumetric + wrench | 18 objects, 6 views |
| PhysTwin [30] | Physics twin from video | 3 objects, 11 views, 无 tactile |
| PGND [86] | Particle-grid dynamics | 4 objects, 8 views |
| **Deform360** | **198 objects, 41 views, tactile, markerless** | – |

Deform360 在 object 数量（198 vs 次多 86）、view 数量（41）、modality richness（visual + tactile + 3DGS reconstruction + particle tracking）上都是 SOTA。

在 world model paradigm 维度上，Deform360 是第一个**systematically compare 2D video model vs 3D particle model** 的 benchmark。之前的工作要么只做 video model（Vid2World [26]、PAN [64]、Cosmos [51]）要么只做 3D（PhysTwin [30]、PGND [86]、ParticleFormer [27]）——从未在 unified dataset 上对比。这个 benchmarking contribution 可能比 dataset 本身更有价值。

---

## 9. 我对这篇 paper 的几个观察

1. **41 cameras 的 cost-benefit**：这个 setup 极其 expensive（41 个同步 camera + bimanual tactile gripper + calibration rig）。但作者证明这是必要的——任何 < 20 view 的 setup 都会有大量 occlusion，导致 3DGS reconstruction 和 particle tracking 在 contact region 失败。这也解释了为什么之前 dataset 都没有 dense annotation——不是不想做，是 capture rig 不够。

2. **Decoupled pipeline 的优雅**：3DGS for geometry + CoTracker3 for 2D tracking + physics-informed fusion。每个 module 都是 SOTA component，但组合在一起形成了一个大于部分之和的系统。这种 modular design philosophy 在 LeCun [37] 的 JEPA 路线、Hafner [24] 的 DreamerV3 路线、以及 Anthropic 的 constitutional AI 中都能看到。

3. **Tactile as ground-truth, not modality**：很多 tactile paper（V-HOP [41]、ViTa-Zero [39]）把 tactile 当作 primary input modality。Deform360 的角度不同——tactile 是 **contact event 的 ground-truth**，用来 constrain visual tracking。这个 perspective shift 解释了为什么 tactile 在 annotation pipeline 里只是 soft regularizer 而非 hard constraint——它在 contact region 提供物理 anchor，在 non-contact region 不参与。

4. **Action misalignment 的哲学**：Cosmos 在 zero-shot 下生成 physically plausible motion 但不严格 follow action——这其实是 LLM-style world model 的本质问题。Video model 学的是 $p(\text{future frames} | \text{past frames, action})$，但当 action 和 visual context 在训练分布外时，model fallback 到 $p(\text{future frames} | \text{past frames})$——即"基于 visual prior 的 association"。这是 Long-tail generalization 的体现，也是为什么 3D structural prior 在 OOD 下更可靠——它不依赖 visual similarity。

5. **MPC 上的 reward design insight**：3D model 可以用 Chamfer distance 做 reward，video model 不行——这是 robotics planning 社区长期 know 但很少 explicitly state 的事情。Deform360 在 Section 5.4 用一句话点出，但这个 insight 值得单独写一篇 paper。

---

## References

- Deform360 project: https://deform360.lhy.xyz
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- CoTracker3: https://github.com/facebookresearch/co-tracker
- Cosmos World Foundation Model: https://github.com/nvidia-cosmos
- PhysTwin: https://arxiv.org/abs/2506.10167 (Jiang et al., ICCV 2025)
- PGND (Particle-Grid Neural Dynamics): https://arxiv.org/abs/2506.15680 (RSS 2025)
- ParticleFormer: https://arxiv.org/abs/2506.23126 (CoRL 2025)
- Universal Manipulation Interface (UMI): https://universal-manipulation-interface.github.io/
- DeformGS: https://arxiv.org/abs/2312.00583
- PhysGaussian: https://arxiv.org/abs/2311.12098 (CVPR 2024)
- DINO-WM: https://arxiv.org/abs/2410.15205 (ICML 2025)
- PointWorld: https://arxiv.org/abs/2601.03782
- Vid2World: https://arxiv.org/abs/2505.14357
- V-HOP: https://arxiv.org/abs/2502.18415 (RSS 2025)
- LeCun "A Path Towards Autonomous Machine Intelligence": https://openreview.net/forum?id=BZ5a1r-kVsf
- DreamerV3: https://arxiv.org/abs/2301.04104
- Sorkine-Hornung & Alexa "As-Rigid-As-Possible Surface Modeling": https://igl.ethz.ch/projects/ARAP/

---

## 总结

Deform360 的核心 contribution 是把 deformable world modeling 从 "we have a dataset" 推进到 "we have a **fair benchmark**"，并且揭示出 2D video model vs 3D particle model 之间的 trade-off 不是 technical detail 而是 **fundamental scaling law**——structural prior 在 low-data 下赢，data scale 在 zero-shot 下赢。这对 future robotics foundation model 的设计有直接 implication：如果你 build 一个 generalist robot policy，你需要一个 hybrid——3D structural prior 作为 inductive bias，2D pretraining 作为 scalable feature learner。这是 LeCun [37] 的 JEPA、Hafner [24] 的 Dreamer、以及 PointWorld [28] 都在朝向的方向，而 Deform360 第一次为 deformable object 给出了 empirical evidence。
