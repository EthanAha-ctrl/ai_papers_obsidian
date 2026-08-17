---
source_pdf: Toward Physically Consistent Driving Video.pdf
paper_sha256: eab32e8c703f3be6080a6ac7238b89a243de1b30cbc4c70610a0a894c5691ebf
processed_at: '2026-08-12T16:52:33-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PhyGenesis

## 一句话概括

**现有的 driving video world model 都是"画皮"——你给它什么 trajectory 它就硬画什么，哪怕 trajectory 物理上根本不通，它也会给你画出车穿透铁栏杆的鬼畜画面。PhyGenesis 加了一个"物理常识脑"先纠错，再让 video generator 画画，同时用 CARLA 模拟的撞车数据教会 model "撞车长什么样"。**

---

## 一、问题到底出在哪？

先讲一个场景。假设你是 end-to-end planner，你 output 一条 trajectory 说："ego vehicle 在 0.5s 后以 50km/h 撞上前面的 guardrail"。这条 trajectory 是 planner 在某个异常状态下吐出来的，可能根本不合理。

你把这条 trajectory 喂给 MagicDrive-V2 或者 DiST-4D，让它生成 video。会发生什么？

**它真的会画一辆车"穿过"guardrail**。因为 model 压根没有"碰撞"这个概念。它的 training data 是 nuScenes，nuScenes 里 99.9% 是乖乖开车，collision event 几乎为零。model 学到的就是 "trajectory condition → pixel mapping"，trajectory 说位置在 (x=3, y=5)，那我就画车在 (x=3, y=5)，管你那里有没有 guardrail。

这就好比你教一个小孩画画，只给他看过正常行驶的车，从来没给他看过撞车。你让他画"撞车"，他只会把车画到墙的位置上，但车和墙是叠在一起的——因为他不知道"撞"是什么意思。

这里有**两个 independent 的 problem**，paper 分得很清楚：

**Problem A: 输入的 trajectory 本身就是 physics-violating 的**
比如两个 agent 的 path 穿透彼此、车开到河里、trajectory 让车穿过建筑物。这种 trajectory 来自 imperfect simulator 或 buggy planner。

**Problem B: 即使 trajectory 是 physics-plausible 的，model 也不会画 physical interaction**
比如 trajectory 说"车在 t=15 帧 hit guardrail 并停下"，这条 trajectory 本身是合理的。但 model 从来没见过 collision，它不知道 collision 瞬间 velocity 应该突降到 0、车头应该 pitch forward、可能有 debris 飞出。它会画一个"慢慢减速停下来的车"，完全不像真实撞车。

现有工作两个 problem 都没解决。PhyGenesis 两个都治。

---

## 二、PhyGenesis 怎么治这两个 problem？

### 治 Problem A：Physical Condition Generator

思路特别直觉：**既然 input trajectory 可能是坏的，那我先加一个 module 把它"修好"再传给下游。**

这个 module 是一个小 transformer，输入 N 个 agent 的 2D trajectory $\mathcal{T}^{orig} = \{(x_{i,t}, y_{i,t})\}$，output 每个 agent 的 6-DoF trajectory $\hat{\mathcal{T}}^{6dof}$，包括 $(x, y, z, \text{pitch}, \text{yaw}, \text{roll})$。

为什么 2D 升 6-DoF？想想撞车瞬间——车头撞 guardrail，车体会 pitch forward（绕 y 轴前倾），可能还有轻微 roll（绕 x 轴侧倾）。z 方向也可能有变化（比如撞上路肩弹起）。2D $(x,y)$ 完全无法表达这些。所以必须升维。

那这个 module 怎么训？关键 trick 是 **counterfactual training pair**：

- 从 CARLA 拿一个真实的撞车 clip
- 撞车**前**的 trajectory 保持原样
- 撞车**后**的 trajectory 故意 corrupt 成"假如没撞会怎样"——让车继续按撞车前的速度匀速直线开
- 这就合成了一条"穿透式"的 counterfactual trajectory（车会穿过 guardrail 继续飞）
- supervision target 是真实撞车的 trajectory（车在 guardrail 处停下）

让 model 看 (坏的输入, 好的 target) 这种 pair，强迫它学会"识别坏 trajectory 并修正"。这本质上是 inverse dynamics learning，让 model 内化物理常识。

具体架构里几个设计点比较精巧：

**Spatial Cross-Attention**：trajectory token 和 multi-view image feature 做 deformable cross-attention。直觉上，这是让 trajectory token "看一眼" image 上对应位置有什么——如果有 guardrail / wall / other car，token 就能感知到可能的碰撞风险。

**Agent Self-Attention**：让 agent 之间互相 attend。一辆车必须知道邻居车在哪才能避免穿透。这是解决穿透 conflict 的核心。

**Map Cross-Attention**：让 trajectory token 看 vectorized map，知道 drivable area 在哪。这是 off-road awareness 的来源。

**Time-Wise Output Head**（这个设计我特别喜欢）：传统 MLP head 把 trajectory 输出"平滑化"了，但碰撞瞬间 velocity 是 discontinuous 的——从 50km/h 瞬间掉到 0。MLP 因为参数共享 + smooth activation 会把这个 impulse 滤掉，画出来的 trajectory 像车"慢慢减速停下"，不符合真实碰撞物理。

PhyGenesis 的做法：对每个 timestep $t$ 都给一个独立的 learnable temporal embedding $\mathbf{E}_{time}(t)$，concat 到 agent token 上，过 TCN，再投影到 6-DoF。

$$\mathbf{h}_{i,t} = \text{TCN}\Big(\text{Proj}\big(\mathbf{q}_f[i] \,\|\, \mathbf{E}_{time}(t)\big)\Big)$$

$$\hat{\mathcal{T}}_{i,t}^{6dof} = \text{MLP}(\mathbf{h}_{i,t}) \in \mathbb{R}^6$$

直觉解释：每个 timestep 有自己独立的 "slot"（through $\mathbf{E}_{time}(t)$），不会被其他 timestep 的输出通过共享 MLP 参数"平均化"。TCN 捕捉 local 时序变化，但保留 per-timestep 的 freedom 来表达 discontinuity。

看 paper Figure 4 就一目了然：MLP head 是 velocity 渐变下降曲线，Time-Wise head 是 velocity 瞬间 drop 到 0，跟 GT 对齐。

Loss 也设计得有意思——weighted L1：

$$\mathcal{L}_{phy} = \frac{1}{N \times T} \sum_{i=1}^{N} \sum_{t=1}^{T} W_{i,t} \big\| \hat{\mathcal{T}}_{i,t}^{6dof} - \mathcal{T}_{i,t}^{gt} \big\|_1$$

$W_{i,t}$ 有两个 weight：
- $\lambda_{\text{event}}=10$：在 collision/off-road 时刻 $t_e$ 附近 ±1 到 +10 timestep 的 window 内，loss 权重从 10 指数衰减到 1。这把 supervision "focus" 到 critical event 附近。
- $\lambda_{\text{agent}}=5$：对参与碰撞的 agent（ego + collision partner）额外加权 5 倍。

直觉：撞车那一瞬间最难学，最重要，所以 loss 要 heavily weight 在那里。

---

### 治 Problem B：Physics-Enhanced Video Generator + Heterogeneous Data

这里思路也特别直觉：**model 不会画撞车，是因为它没见过撞车。那就用 CARLA 模拟一堆撞车给它看。**

具体做法：

1. 在 CARLA simulator 上跑 Bench2Drive routing
2. 通过 perturb ego vehicle 或 nearby non-ego vehicle 的 lateral offset 和 target speed，制造 collision / off-road event
3. 配置 6 camera + 1 LiDAR + 5 radar + IMU，sensor setup 严格 align nuScenes
4. 关键：加 collision sensor 和 HD map metadata，精确记录碰撞 timestamp
5. 收集了 31 小时数据，filter 出 9.7 小时 high-quality physically-challenging clips
6. 和 nuScenes 4.6 小时 nominal data 混合，1:1 balanced sampling 训练

1:1 这个 ratio 是关键 trick。如果简单 mixing，nuScenes 因为数据量大就会 dominate batch，CARLA 的撞车 signal 被稀释到没。1:1 强制每个 batch 里一半是撞车 / off-road，model 才能真正学到物理 dynamics。

Video generator 用的是 Wan2.1（阿里的 high-capacity DiT video model，https://github.com/Wan-Video/Wan2.1），改造为 multi-view controllable 版本。

Multi-view 处理有个偷懒但高效的设计：把 6 个 view 的 latent 在 spatial 维度 concat 成 $\mathbb{R}^{T \times C \times h \times (V \cdot w)}$，这样原始 self-attention 自然就能 capture cross-view dependencies，无需额外 cross-view attention module。

Layout condition：把 future T-frame 的 3D agent boxes + map polylines 通过 camera intrinsics/extrinsics 投影到每个 view，得到 control image，VAE 编码后和 noisy latent 在 channel 维度 concat，进 DiT。

训练用 Rectified Flow（Wan2.1 原生的 framework）：

$$\mathbf{z}_t = t\mathbf{z}_1 + (1-t)\mathbf{z}_0$$

$$\mathcal{L}_{FM} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \big\| u_\theta(\mathbf{z}_t, t, \mathbf{c}_{init}, \mathbf{c}_{text}, \mathbf{c}_{layout}) - \mathbf{v}_t \big\|_2^2$$

- $\mathbf{c}_{init}$：initial frame latent
- $\mathbf{c}_{text}$：scene caption
- $\mathbf{c}_{layout}$：multi-view layout image latent
- $\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$：target velocity

Curriculum：224×400 低分辨率先训 2850 步学几何和 layout mapping，再升到 448×800 训 350 步 polish visual fidelity。在 48 张 H20 GPU 上跑。

**这里有个关键 design choice**：在这个 stage **不用 counterfactual trajectory**，只用 ground-truth physical trajectory。这是为了 decouple——stage 1 专门做物理纠错，stage 2 专门做 rendering。如果 stage 2 也用 counterfactual，model 就要同时学 rectification + rendering，两个任务会互相干扰。

---

## 三、实验结果到底有多炸？

直接看 Table 1 在 CARLA Ego（physics-violating trajectory input）上的对比：

| Method | FID↓ | FVD↓ | PHY↑ | Pref.↑ |
|---|---|---|---|---|
| UniMLVG | 34.50 | 260.21 | 0.55 | 0.13 |
| MagicDrive-V2 | 32.19 | 207.64 | 0.60 | 0.06 |
| DiST-4D | 19.84 | 197.57 | 0.39 | 0.10 |
| **PhyGenesis** | **11.03** | **72.48** | **0.71** | **0.71** |

DiST-4D 的 PHY score 只有 0.39（满分 1.0）——这意味着生成的 video 里大量 object 穿透、变形、melting。PhyGenesis 到 0.71，human preference 直接到 0.71，也就是说 71% 的 case 人类选 PhyGenesis。

Table 3 看 Physical Condition Generator 单独的效果（6-DoF L2 distance to GT）：

| Setting | nuScenes | CARLA Ego | CARLA Adv |
|---|---|---|---|
| W/o Phy Cond Gen | 0.21 | 1.78 | 1.05 |
| W/ Phy Cond Gen | 0.19 | 0.65 | 0.86 |

CARLA Ego 上 trajectory error 从 1.78 降到 0.65，**降了 63%**。这就是 counterfactual rectification 的威力。

Ablation Table 4 也讲得很清楚：
- 去掉 Physical Condition Generator：CARLA Ego Pref 从 0.55 掉到 0.19
- 去掉 Mixed Data（只用 nuScenes）：CARLA Adv Pref 从 0.57 掉到 0.15
- 两个都去掉：基本回到 baseline 水平

两个 component 是 additive 的，缺一不可。

---

## 四、我最喜欢这个 paper 的几个点

**1. Problem decomposition 非常 clean。**

很多 driving world model paper 把一堆东西揉一起，loss function 写半页纸。PhyGenesis 把 problem 拆成 trajectory feasibility + rendering quality 两块，每块一个 module，每个 module 一个明确的 loss。这种 clean decomposition 让 ablation 特别清楚——Table 4 直接告诉你两个 module 各自的贡献。

**2. Counterfactual trajectory corruption 是 self-supervised 物理推理。**

不需要在 inference time 跑物理 engine，但训练时用 engine 生成 supervision 信号，把 simulator 的物理知识"蒸馏"到轻量 transformer 里。这跟 model-based RL 里的 "imaginary rollout correction" 思路类似，可以推广到很多 domain。

**3. Time-Wise Output Head 这个细节特别 sharp。**

这个设计我第一次看没 get，后来想到 collision 是 impulse event，物理上就是 discontinuous 的。MLP head 本质是 low-pass filter，会滤掉高频。Per-timestep token + TCN 给每个 timestep 独立的 "slot" 来表达 discontinuity。这个 insight 对任何需要建模 piecewise dynamics 的 task 都有用——robot contact transition、foot strike in locomotion、sudden obstacle appearance。

**4. 1:1 heterogeneous co-training 是 sim-to-real 的核心 hyperparameter。**

简单 mixing 会让 nominal data dominate。1:1 强制 balance，让稀有 event 的 signal 不被淹没。这个 trick 在任何 long-tail distribution 训练里都适用。

**5. 6-DoF representation 比 2D 更匹配 target dynamics 的自由度。**

condition representation 的维度必须匹配 target dynamics 的自由度，否则会 bottleneck 整个 system。这跟 latent world model 里 latent dimension 选择的 trade-off 是同一类问题。

---

## 五、可能的延伸思考

如果你在做 model-based RL / world model，这篇 paper 给了几个值得借鉴的方向：

**Decoupled physical reasoning + rendering** 对应 model-based RL 中 "latent dynamics model" + "decoder" 的解耦。让 dynamics model 专门学 action feasibility，让 decoder 专门学 observation generation，避免互相干扰。

**Counterfactual trajectory as self-supervised physical prior** 可以推广到任何 domain：让 model 看 "如果违反物理会发生什么" 和 "真实物理结果" 的 pair，强迫它内化物理常识。这比 hardcode 物理规则要 flexible，比纯 data-driven 要 sample efficient。

**Heterogeneous data ratio** 是 sim-to-real 的核心 hyperparameter。在 model-based RL 里，real trajectory 和 model-generated imaginary trajectory 的混合 ratio 也是类似的 trade-off——太多 imaginary 会 accumulate error，太少又 sample inefficient。

可能的 follow-up 方向：
- 把 Physical Condition Generator 替换成 symbolic physics engine + learned residual（hybrid），让物理保证更强
- 用 3D Gaussian Splatting 替代 DiT latent representation，让 collision 后的 deformation 在 3D space 而非 image space 表达
- 把 trajectory input 升级成 intention-level（"想要变道"）而非 coordinate-level，让 model 自己 propose physical trajectory
- 加入 LiDAR / occupancy 这种 3D-aware modality 作为 output，让物理一致性更可验证

---

## 六、几个 paper 没明说但能看出来的事

1. **DiST-4D 在 CARLA 上 PHY 只有 0.39**——这个数字其实非常糟糕，说明现有 SOTA 在 challenging trajectory 下基本是 unusable 的。PhyGenesis 拉到 0.71，离 perfect 还有距离，但已经从"完全 unusable"变成"usable for safety testing"。

2. **9.7 小时 CARLA clips + 4.6 小时 nuScenes 的混合**，相比 nuScenes 全集 ~20 小时，数据量并不大。能在这种数据量下取得这么大提升，说明 problem formulation 对了比堆数据量重要。

3. **Style transfer model 的存在**说明 sim-to-real 还有 visual gap 问题。CARLA 渲染再逼真也和真实相机有差异，所以需要 Wan2.1-based transfer model 把 CARLA video 翻译成 nuScenes 风格才能公平 benchmark。这是 closed-loop simulation 评估 protocol 里的一个 practical trick。

4. **48 张 H20 GPU + 两 stage curriculum**，training cost 不算夸张。Physical Condition Generator 是轻量 transformer，training 和 inference 都 fast。这意味着这个 framework 可以 plug-in 到现有 driving stack 里做 real-time trajectory sanity check。

---

## 七、最后用一段话总结

PhyGenesis 的 contribution 不在于单点 SOTA，而在于一个 **system-level insight**：driving video world model 在 challenging trajectory 下崩坏，根因是 training distribution 缺 physical interaction + inference input 可能 physics-violating。这两个 problem 必须同时治。用 CARLA-generated physics-rich heterogeneous data 解决 distribution 缺失，用 Physical Condition Generator + counterfactual rectification training 解决 input 不合理。paper 在工程上做得很扎实——sensor 配置严格 align nuScenes、style transfer 保证公平 benchmark、curriculum co-training 稳定 large-scale DiT 训练、event-aware weighted loss focus supervision 到 critical moment。

它把 driving video generation model 从"condition-to-pixel translator"推向"physics-aware world model"，在 closed-loop evaluation 和 safety-critical scenario synthesis 这种下游应用里是质变的 enabler。

参考链接：
- Project page: https://wmresearch.github.io/PhyGenesis/
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- nuScenes: https://www.nuscenes.org/
- CARLA: https://carla.org/
- Bench2Drive: https://github.com/ID-Canvas/Bench2Drive
- Rectified Flow (SD3): https://arxiv.org/abs/2403.03206
- WorldModelBench: https://arxiv.org/abs/2502.20694
- MagicDrive-V2: https://arxiv.org/abs/2411.13844
- DiST-4D: https://arxiv.org/abs/2503.06542

---

# PhyGenesis: Physics-Aware Driving World Model 深度解析

## 一、核心 Problem 与 Motivation 直击

这篇 paper 来自 Xiaomi EV + Zhejiang University，攻击的是当前 driving video world model 的一个**致命缺陷**：它们本质上是 condition-to-pixel translators，缺乏 physical reasoning。

具体有两层 limitation：

1. **Trajectory feasibility awareness 缺失** — 主流 trajectory simulator 或 planner 给出的 trajectories 可能是 physics-violating 的（例如让两个 agent 的 path 穿透彼此），但现有 model 照单全收，会渲染出 object melting、geometry distortion 这种 artifacts。

2. **Physics-consistent generation 能力缺失** — 现有 model 在 nuScenes 这类 nominal real-world data 上训练，collision 和 off-road 这类事件在 training distribution 中极度稀疏，导致即便给出 physical-feasible 的 challenging trajectory，model 也无法生成正确的 dynamics（如碰撞瞬间的速度突降到 0）。

PhyGenesis 的核心 insight 是：**trajectory feasibility 和 physics-consistent video generation 必须被联合处理**，分成两个 stage：Physical Condition Generator 先把任意 2D trajectory rectify 成 physically-plausible 6-DoF motion；Physics-Enhanced Video Generator 再在 hybrid dataset 上 co-train 来 render 物理一致的 multi-view video。

项目主页：https://wmresearch.github.io/PhyGenesis/

---

## 二、整体 Architecture 详解

系统的输入定义：

- $\mathcal{T}_0$：initial multi-view images
- $\mathcal{M}$：static map
- $\mathcal{T}^{orig}$：所有 N 个 agent 的 future trajectories
  - $\mathcal{T}^{orig} = \{\mathcal{T}_i^{orig}\}_{i=1}^{N}$
  - $\mathcal{T}_i^{orig} = \{\mathcal{T}_{i,t}^{orig}\}_{t=1}^{T}$
  - $\mathcal{T}_{i,t}^{orig} = (x_{i,t}, y_{i,t})$ — agent $i$ 在时刻 $t$ 的 2D location

这里的 $(x_{i,t}, y_{i,t})$ 上标/下标含义：
- $i$：agent index（车）
- $t$：time step
- $x, y$：2D 平面坐标

选择 2D 表示是为了 align 主流 trajectory simulator 和 end-to-end planner 的 output format。

输出：multi-view video sequence $\mathcal{V}_{1:T}$，要满足 high fidelity + physical consistency。

### 2.1 Physical Condition Generator 架构（对应 Figure 2 (b) 左侧）

这是一个 sequence-to-sequence trajectory transformer，作用是把可能 physics-violating 的 2D trajectory 转成 physically plausible 的 6-DoF motion（$x, y, z, \text{pitch}, \text{yaw}, \text{roll}$）。为什么升到 6-DoF？因为 collision 和 off-road 会引入 $z$ 方向的颠簸和 pitch/roll 这种 rotation，2D 完全无法表达。

**Step 1: Token encoding**

原始 2D trajectories 先过 sine-cosine positional encoding，再过一个 MLP encoder，得到 agent tokens：

$$\mathbf{q} \in \mathbb{R}^{N \times D}$$

- $N$：agent 数量
- $D$：token dimension（hyperparameter）

**Step 2: Spatial Cross-Attention（接地气到 multi-view PV feature）**

为了让 token "看到" visual environment，对 multi-view Perspective View features $\mathcal{F}_{pv}$ 做 deformable spatial cross-attention：

$$\mathbf{q}_s = \text{SpatialCrossAttn}(\mathbf{q}, \mathcal{F}_{pv}) \tag{1}$$

这里用 deformable attention 是因为 trajectory 坐标提供了 query point，可以 deformable 采样 PV feature map 上对应位置。这一步把 trajectory token 锚定到 camera view 上的具体语义内容。

**Step 3: Agent Self-Attention（agent 间互动推理）**

让 token 之间感知彼此的 position 和 kinematic state：

$$\mathbf{q}_a = \text{AgentSelfAttn}(\mathbf{q}_s) \tag{2}$$

这是解决 overlapping 和 penetration conflict 的关键设计 — 一个 agent 必须知道邻居 agent 在哪才能避免穿插。

**Step 4: Map Cross-Attention（off-road 感知）**

引入 vectorized map embeddings $\mathbf{E}_{map}$：

$$\mathbf{q}_m = \text{MapCrossAttn}(\mathbf{q}_a, \mathbf{E}_{map}) \tag{3}$$

这一步让 agent 知道哪里是 drivable area、哪里是 guardrail / curb，避免 trajectory 跑出路面。

**Step 5: FFN**

$$\mathbf{q}_f = \text{FFN}(\mathbf{q}_m) \tag{4}$$

非线性聚合上面所有信息。

**Step 6: Time-Wise Output Head（关键设计）**

这一步是论文里一个特别巧妙的设计。传统 MLP head 会把 trajectory 输出"平滑化"，无法表达 collision 瞬间 velocity 突降到 0 这种 high-frequency dynamic impulse。所以 paper 设计了 per-timestep 的 head：

对第 $i$ 个 agent 的 refined token $\mathbf{q}_f[i]$，先沿 $T$ 个 future step 复制，然后和 step-specific learnable temporal embedding $\mathbf{E}_{time}(t)$ 拼接，过 TCN：

$$\mathbf{h}_{i,t} = \text{TCN}\Big(\text{Proj}\big(\mathbf{q}_f[i] \,\|\, \mathbf{E}_{time}(t)\big)\Big) \tag{5}$$

- $\|$：concatenation
- $\mathbf{E}_{time}(t)$：让每个 timestep 携带其独立的可学习时间身份
- TCN：捕捉相邻 timestep 之间的 local 动态变化

最后 MLP 投影到 6-DoF：

$$\hat{\mathcal{T}}_{i,t}^{6dof} = \text{MLP}(\mathbf{h}_{i,t}) \in \mathbb{R}^6 \tag{6}$$

参考 Figure 4 的对比：MLP head 在 collision 后是 velocity 渐变下降的曲线，GT 和 Time-Wise head 是 velocity 瞬间 drop to zero。这个细节非常重要——collision 是 impulse event，物理上就是 discontinuous 的。

### 2.2 Counterfactual Training Pair 构造

这是让 Physical Condition Generator 学会"rectification"的核心 trick：

- 拿一条 CARLA 中的 collision clip
- Collision **之前**的 trajectory 保持原样
- Collision **之后**的 trajectory 故意 corrupt：让所有 agent 沿 collision 前的 velocity 继续匀速直线运动 → 合成"穿透式" counterfactual trajectory
- 监督 target 是真实 simulation log（即真正碰撞后的 trajectory）

这就创造了一对（physics-violating input, physics-plausible target），强迫 model 学到"如何把穿透 trajectory 改成碰撞 trajectory"。

同时为了避免 distorting nominal driving，还会把 nuScenes 的 real trajectory 不做 corruption 直接 pair 起来。

### 2.3 Physical Condition Generator 的 Loss

$$\mathcal{L}_{phy} = \frac{1}{N \times T} \sum_{i=1}^{N} \sum_{t=1}^{T} W_{i,t} \big\| \hat{\mathcal{T}}_{i,t}^{6dof} - \mathcal{T}_{i,t}^{gt} \big\|_1 \tag{7}$$

- $\|\cdot\|_1$：L1 距离（比 L2 对 outlier 更鲁棒）
- $W_{i,t}$：per-agent, per-timestep 的权重，由两个 scalar 组成

**$W_{i,t}$ 的设计**（详见 Appendix E）：

Temporal weight $w_e(t)$：在 event timestep $t_e$ 附近定义一个 forward window $[s_e, e_e]$，其中 $s_e = \max(0, t_e - 1)$，$e_e = \min(T-1, t_e + 10)$，在这个 window 内 weight 从 $\lambda_{\text{event}}$ 指数衰减到 1：

$$w_e(t) = \lambda_{\text{event}} \exp\Big( \frac{\log(1/\lambda_{\text{event}})}{e_e - s_e} (t - s_e) \Big), \quad t \in [s_e, e_e]$$

- $\lambda_{\text{event}}$：在 collision/off-road onset 处 loss 放大倍数（论文设 10）
- 衰减常数 $\frac{\log(1/\lambda_{\text{event}})}{e_e - s_e}$：保证 window 末尾 weight 衰减到 1
- 多个 event window 重叠时取 max，window 外权重为 1

Agent weight $\lambda_{\text{agent}}$：对参与物理事件的 agent（碰撞对方 vehicle / pedestrian）额外放大 loss。论文设 $\lambda_{\text{agent}} = 5$。

---

## 三、Physics-Enhanced Multi-View Video Generator (PE-MVGen)

基于 **Wan2.1**（https://github.com/Wan-Video/Wan2.1）这个 high-capacity DiT 改造，原本是 image+text 条件的视频生成模型，这里把它改装成 autonomous driving 领域的 multi-view controllable generator。

### 3.1 Multi-View & Layout Conditioning

输入 multi-view clips 经过 pre-trained 3D VAE 编码到 latents：

$$\mathbf{z} \in \mathbb{R}^{V \times T \times C \times h \times w}$$

- $V$：view 数量（nuScenes 是 6）
- $T$：time
- $C$：latent channel
- $h, w$：latent spatial 维度

为了**不引入额外参数**就能做 multi-view modeling，paper 把 view 维度 reshape 到 spatial axis：

$$\mathbb{R}^{T \times C \times h \times (V \cdot w)}$$

这样原始的 self-attention 自然就能 capture cross-view dependencies，无需额外 cross-view attention 模块——这是从 MagicDrive-V2 沿袭过来的高效设计。

Layout conditioning：把 future T-frame 的 3D agent boxes 和 map polylines 用 camera intrinsics $\mathbf{K}_v$ 和 extrinsics $\mathbf{E}_v$ 投影到每个 camera view，得到 view-specific control image $\mathbf{M}_v$，再用 VAE encoder 编码到 $\mathbf{z}_c$，reshape 后在 channel dimension 与 noisy latent $\mathbf{z}_t$ 拼接，过 patch embedder 进入 DiT。

### 3.2 Rectified Flow 训练目标

Wan2.1 用的是 Rectified Flow formulation（参考 https://arxiv.org/abs/2403.03206）。

给定 clean video latent $\mathbf{z}_1$ 和 noise $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，时间步 $t \in [0, 1]$ 从 logit-normal 分布采样，noisy latent 为线性插值：

$$\mathbf{z}_t = t\mathbf{z}_1 + (1-t)\mathbf{z}_0 \tag{8}$$

- $t=0$：纯噪声
- $t=1$：clean signal
- linear interpolation 比 DDPM 的 cosine schedule 更适合 ODE 求解

Ground-truth velocity vector：

$$\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$$

DiT 模型参数 $\theta$ 预测 velocity $u_\theta$，loss 是 MSE：

$$\mathcal{L}_{FM} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \big\| u_\theta(\mathbf{z}_t, t, \mathbf{c}_{init}, \mathbf{c}_{text}, \mathbf{c}_{layout}) - \mathbf{v}_t \big\|_2^2 \tag{9}$$

- $\mathbf{c}_{init}$：单个 initial context frame 的 latent feature
- $\mathbf{c}_{text}$：scene caption（语义描述）
- $\mathbf{c}_{layout}$：future multi-view layout images

**关键 design choice**：在这个 stage，generator 只用 ground-truth physical trajectories 监督，**不用 counterfactual**。这就把 physical correction（在 stage 1 完成）和 rendering（stage 2）解耦了——generator 不会被 trajectory rectification 任务干扰，专注生成 high-fidelity physically consistent video。

### 3.3 Curriculum Co-Training

- Stage 1：224×400 低分辨率，2850 steps，lr $5\times 10^{-5}$，batch size 480 — 快速学习 multi-view geometry 和 physical layout mapping
- Stage 2：448×800 高分辨率，350 steps，lr $1\times 10^{-4}$，batch size 240 — 提升 visual fidelity

在 48 张 NVIDIA H20 GPU 上训练，生成 33-frame videos at 12 Hz。

---

## 四、Heterogeneous Dataset 构造细节

这是整篇 paper 的"脏活累活"。

### 4.1 Real-World 部分

nuScenes（https://www.nuscenes.org/），正常 urban driving，6 cameras，~1000 scenes，~20s/scene @ 2Hz。但严重 bias 到 safe driving，collision/off-road 数据几乎为零。

### 4.2 CARLA Simulated 部分

基于 CARLA（https://carla.org/）+ Bench2Drive routing setup（https://github.com/ID-Canvas/Bench2Drive）。

配置的 sensor suite：
- 1 LiDAR
- 6 surround-view cameras @ 900×1600
- 5 radars
- 1 IMU/GNSS
- **额外 collision sensor** — 精确记录碰撞 timestamp
- **HD map metadata** — 精确记录 off-road timestamp

记录频率 12Hz（比 nuScenes 的 2Hz 高 6 倍），annotation 格式与 nuScenes 完全一致。

两个 subset：
- **CARLA Ego**：perturb ego vehicle 自身，ego 与 environment/其他 agent 交互
- **CARLA Adv**：perturb 一个 nearby non-ego vehicle，non-ego 与环境/ego 交互

Perturbation mechanism（详见 Appendix D）：
- 在 Bench2Drive 默认 route 上做 lateral offset 和 target speed 的扰动
- Target speed 在 0-30 m/s 间随机采样
- Lateral offset 在 -200 到 200 m 间随机采样
- 三种 perturbation mode 等概率：(i) 零 lateral + 随机速度，(ii) 固定 10 m/s + 随机 lateral，(iii) 都随机
- Ego 模式下，前 24 steps（2 秒）走 autopilot，之后切换到 perturbed route
- Event 触发后继续录 48 帧（4 秒），无 event 上限 120 步（10 秒）

总数据规模：

| Dataset | Hours | Bounding Boxes |
|---|---|---|
| CARLA Adv | 15.5h | 760K |
| CARLA Ego | 15.2h | 830K |
| 高度物理挑战 clips（filter 后）| 9.7h | - |
| nuScenes（混合用）| 4.6h | - |

Figure 3 显示了 nuScenes vs CARLA Ego vs CARLA Adv 的 maximum ego-acceleration 分布对比——CARLA 数据明显 shift 到更高加速度，说明 dynamics 更 aggressive。

### 4.3 1:1 Balanced Sampling

训练时强制 simulated-to-real clip ratio 接近 1:1，避免 nominal nuScenes 因为数据量过大而 dominate 训练信号。

---

## 五、实验结果深度解读

### 5.1 主结果（Table 1）

baseline 是 UniMLVG、MagicDrive-V2、DiST-4D。PhyGenesis 在三个 dataset 上 PHY score 都最高，visual quality（FID/FVD）也最好，特别是 CARLA Ego 和 CARLA Adv 这种 physically challenging set 上提升最明显。

CARLA Ego 上：
- FID：DiST-4D 19.84 → PhyGenesis **11.03**（44% 相对降低）
- FVD：DiST-4D 197.57 → PhyGenesis **72.48**（63% 相对降低）
- PHY：DiST-4D 0.39 → PhyGenesis **0.71**（接近翻倍）
- Pref.：PhyGenesis **0.71** vs DiST-4D 0.10

PHY 在 DiST-4D 上掉到 0.39 这么低，正是因为它直接拿 physics-violating trajectory 喂 video generator，导致 object 严重变形/穿透。PhyGenesis 通过两阶段处理规避了这个问题。

### 5.2 GT Trajectory 下的 Video Generator 单独评估（Table 2）

这里都给 ground-truth trajectory 作为 input，剥离掉 Physical Condition Generator 的贡献，单独看 Video Generator。

| Method | nuScenes FID | CARLA Ego FID | CARLA Adv FID |
|---|---|---|---|
| UniMLVG | 16.63 | 34.03 | 30.06 |
| MagicDrive-V2 | 13.17 | 32.92 | 33.62 |
| DiST-4D | 10.48 | 19.94 | 16.12 |
| PhyGenesis | **10.20** | **10.98** | **9.07** |

即使 input trajectory 是 GT 的，baseline 在 CARLA 上依然很差——因为他们的 training distribution 里没有 collision / off-road 这种 dynamics，根本不知道怎么 render。这就是为什么需要 heterogeneous co-training。

CtrlErr（controllability error）PhyGenesis 也最好，说明 trajectory-following 也没牺牲。

### 5.3 Physical Condition Generator 的效果（Table 3）

| Setting | nuScenes | CARLA Ego | CARLA Adv |
|---|---|---|---|
| W/o Phy Cond Gen | 0.21 | 1.78 | 1.05 |
| W/ Phy Cond Gen | 0.19 | 0.65 | 0.86 |

6-DoF L2 距离 GT 的 error：
- CARLA Ego 上从 1.78 掉到 0.65，**63% 相对降低**
- nuScenes 也小幅改善（0.21 → 0.19），因为 model 帮 nuScenes 补齐了缺失的 4 个 DoF

Figure 8 给了一个 qualitative 例子：input trajectory 直接穿过 guardrail，rectification 后 trajectory 与 guardrail 发生碰撞并停止。这正是 counterfactual training 想要 model 学到的 behavior。

### 5.4 Ablation Study（Table 4）

| Mixed Data | Phy-Model | nuScenes FID | CARLA Ego PHY | CARLA Adv PHY |
|---|---|---|---|---|
| ✓ | ✓ | 10.24 | 0.71 | 0.87 |
| ✓ | ✗ | 10.70 | 0.65 | 0.85 |
| ✗ | ✓ | 10.53 | 0.71 | 0.84 |

- 去掉 Phy-Model：CARLA Ego Pref 从 0.55 掉到 0.19
- 去掉 Mixed Data：CARLA ADV Pref 从 0.57 掉到 0.15

两个 component 都不可或缺。

### 5.5 Weighting Ablation（Table 5, 6）

$\lambda_{\text{event}}$ 和 $\lambda_{\text{agent}}$ 在较宽范围内都对结果影响有限，说明 model 对这些超参相对 robust，weighting 机制主要起 "focused supervision" 的作用。

---

## 六、Style Transfer Model（Appendix B）

为了公平对比，因为 baseline 主要在 nuScenes 上训练，paper 训了一个 style transfer model 把 CARLA video 翻译到 nuScenes 视觉风格。

- Backbone：Wan2.1-Fun-V1.1-1.3B-Control
- 条件：per-frame depth（来自 Depth Anything V2 https://github.com/DepthAnything/Depth-Anything-V2）+ video-level caption（来自 Qwen2.5-VL https://qwenlm.github.io/blog/qwen2.5-vl/）
- Loss：同样 Rectified Flow MSE

$$\mathcal{L}_{\text{transfer}} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \big\| u_\theta(\mathbf{z}_t, t, \mathbf{c}_{text}, \mathbf{c}_{depth}) - (\mathbf{z}_1 - \mathbf{z}_0) \big\|_2^2 \tag{11}$$

关键 trick：**不用 initial frame 作为条件**，只用 depth + caption，这样 model 的生成自然 reflect nuScenes 风格而不是 input video 的外观。Style transfer model 完全在 nuScenes 上训练。

---

## 七、构建 Intuition 的核心 takeaways

1. **Two-stage decoupling 是关键**：把 trajectory feasibility（structural reasoning）和 video rendering（pixel synthesis）解耦。前者需要轻量 transformer 学 counterfactual rectification，后者需要大容量 DiT 学 visual generation。耦合在一起会互相干扰。

2. **Time-Wise Output Head 的意义**：物理事件（collision）是 discontinuous impulse，传统 MLP head 因为参数共享 + smooth activation 会把它们"低通滤波"。Per-timestep token + TCN 把每个 timestep 当独立 slot，可以学高频 discontinuity。

3. **Counterfactual Training 是一种 self-supervised 物理推理**：用 "假设不碰撞会怎样"（继续匀速）作为 input，用真实碰撞 trajectory 作为 target，强迫 model 学会"识别 + 修正"违反物理的 trajectory。这本质上是 inverse dynamics learning。

4. **Heterogeneous data ratio (1:1) 的必要性**：CARLA 物理挑战数据虽然只有 9.7h，远少于 nuScenes 全集，但 1:1 sampling 保证 physical 事件不会在 mini-batch 中被 nominal 数据淹没。这是 sim-to-real co-training 的常用技巧。

5. **6-DoF 的必要性**：2D trajectory 在 collision 后无法表达 vehicle 的 pitch / roll 变化（例如车头撞 guardrail 会 pitch forward）。升到 6-DoF 后 video generator 才能 render 正确的 3D posture。

6. **Deformable cross-attention 锚定 trajectory 到 visual context**：trajectory coordinate 提供了天然的 reference point，deformable attention 在该点附近采样 PV feature，相当于"告诉 token 这条 trajectory 在 image 上对应哪个区域"，是 condition grounding 的关键。

---

## 八、Related Work Map

### Driving World Models 主线演进
- **BEVGen**（https://arxiv.org/abs/2310.13360）：用 BEV layout 控制，但丢 height
- **BEVControl**（https://arxiv.org/abs/2308.01661）：加 height-lifting 模块
- **MagicDrive**（https://arxiv.org/abs/2310.02605）：3D geometric constraint + cross-view attention
- **MagicDrive-V2**（https://arxiv.org/abs/2411.13844）：DiT backbone + 高分辨率
- **Drive-Dreamer**（https://arxiv.org/abs/2309.09777）：hybrid Gaussians 保障 temporal consistency
- **DiST-4D**（https://arxiv.org/abs/2503.06542）：metric depth + 4D scene representation
- **WorldSplat**（https://arxiv.org/abs/2509.23402）：Gaussian-centric feed-forward 4D
- **Genesis**（https://arxiv.org/abs/2506.07497）：multimodal (LiDAR + RGB) joint generation
- **UniScene**（https://arxiv.org/abs/2502.05283）：occupancy-centric voxel representation
- **GAIA-1**（https://arxiv.org/abs/2309.17080）：早期 generative world model

### High-risk Driving Video Generation
- **AVD2**（https://arxiv.org/abs/2501.01414）：单目 accident video
- **DrivingGen**（IEEE ICME 2024）：latent diffusion 安全关键场景
- **Ctrl-Crash**（https://arxiv.org/abs/2506.00227）：可控 crash diffusion
- **SafeMVDrive**（https://arxiv.org/abs/2505.17727）：trajectory simulator + multi-view generator
- **Challenger**（https://arxiv.org/abs/2505.15880）：affordable adversarial video generation
- **ReSim**（https://arxiv.org/abs/2506.09981）：用 synthetic 数据 augment world model 训练（但单视角 + 仅 ego-trajectory）

### 通用技术组件
- **Wan2.1**（https://arxiv.org/abs/2503.20314）：base video DiT
- **Rectified Flow**（https://arxiv.org/abs/2403.03206）：Esser et al. 的 scaling rectified flow transformer
- **Depth Anything V2**（https://arxiv.org/abs/2406.09414）：depth condition 提取
- **Qwen2.5-VL**（https://arxiv.org/abs/2502.13923）：video-level captioning
- **CARLA**（https://carla.org/）：开源 urban driving simulator
- **Bench2Drive**（https://arxiv.org/abs/2406.21296）：closed-loop end-to-end benchmark
- **nuScenes**（https://www.nuscenes.org/）：标准 multi-modal autonomous driving dataset
- **WorldModelBench**（https://arxiv.org/abs/2502.20694）：VLM-as-judge 评估 world model
- **ViPE**（https://arxiv.org/abs/2508.10934）：video pose extraction，用于 CtrlErr 计算

---

## 九、可能的延伸思考

如果你（Andrej）在做 model-based RL / world model 的工作，这篇 paper 给出几个值得借鉴的方向：

1. **Decoupled physical reasoning + rendering** 对应 model-based RL 中 "latent dynamics model" + "decoder" 的解耦。Counterfactual trajectory corruption 类似于 model-based RL 中的 "imaginary rollout with bad dynamics → ground truth correction" — 让 model 显式学会区分 feasible vs infeasible action sequences。

2. **Time-Wise Output Head 对 impulse event 的处理** 可以推广到任何 piecewise dynamics：robot contact transition、foot strike in locomotion、sudden obstacle。MLP head 是 prior work 默认设计，但对高频 discontinuity 本质不适配。Per-timestep token 是 inductive bias 上更好的选择。

3. **1:1 heterogeneous co-training ratio** 是 sim-to-real 的核心 hyperparameter，比简单的 data mixing 更激进。可以联想 OpenAI 的 "mixing real and synthetic in fixed ratio" 训练 paradigm。

4. **6-DoF vs 2D trajectory representation** 提醒我们：condition representation 的维度必须匹配 target dynamics 的自由度。Low-dimensional condition 在 challenging regime 下会 bottleneck 整个 system。这跟 latent world model 里 latent dimension 选择的 trade-off 是同一类问题。

5. **Counterfactual trajectory as self-supervised physical prior** 是一种非常有意思的 training paradigm — 不需要 explicit physical engine 在 inference time，但训练时用 engine 生成 supervision 信号。这跟 distillation 思路类似，把 simulator 的物理知识"蒸馏"到一个轻量 transformer 里，从而在 inference 时无需 simulator。

6. **Style transfer 用于公平 benchmark**：当 baseline 在 A domain 训练、你的 model 在 A+B domain 训练时，把 B 的 GT 翻译到 A domain 来计算 FID/FVD 是必要的 protocol trick，否则评估不公平。

可能的 follow-up 方向：
- 把 Physical Condition Generator 替换成 symbolic physics engine + learned residual（hybrid）
- 用 3D Gaussian Splatting 替代 DiT 的 latent representation，让 collision 后的 deformation 在 3D space 而非 image space 表达
- 把 trajectory input 升级成 intention-level（"想要变道"）而非 coordinate-level，让 model 自己 propose physical trajectory
- 加入 LiDAR / occupancy 这种 3D-aware modality 作为 output，让物理一致性更可验证

---

## 十、总结

PhyGenesis 的 contribution 不在于单点 SOTA，而在于一个 **system-level insight**：autonomous driving world model 在 challenging trajectory 下崩坏，是因为 training distribution 缺 physical interaction + inference input 可能 physics-violating。这两个 problem 必须同时治：
- 前者用 CARLA-generated physics-rich heterogeneous data 解决
- 后者用 Physical Condition Generator + counterfactual rectification training 解决

并且 paper 在工程上做得很扎实：sensor 配置严格 align nuScenes、style transfer 保证公平 benchmark、curriculum co-training 稳定 large-scale DiT 训练、event-aware weighted loss focus supervision 到 critical moment。

它把"driving video generation model"从 "condition-to-pixel translator" 推向 "physics-aware world model"，这在 closed-loop evaluation 和 safety-critical scenario synthesis 这种下游应用里是质变的 enabler。

参考链接：
- Project page: https://wmresearch.github.io/PhyGenesis/
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- nuScenes: https://www.nuscenes.org/
- CARLA: https://carla.org/
- Bench2Drive: https://github.com/ID-Canvas/Bench2Drive
- Rectified Flow (SD3 paper): https://arxiv.org/abs/2403.03206
- WorldModelBench: https://arxiv.org/abs/2502.20694
- MagicDrive-V2: https://arxiv.org/abs/2411.13844
- DiST-4D: https://arxiv.org/abs/2503.06542
- ReSim: https://arxiv.org/abs/2506.09981
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Qwen2.5-VL: https://qwenlm.github.io/blog/qwen2.5-vl/
