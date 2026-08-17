---
source_pdf: Motion Forcing A Decoupled Framework for Robust Video Generation in Motion
  Dynamics.pdf
paper_sha256: b2786999409e70e7ca6982cd8879e12c81f4119ceae8f6fbd26043316515b09d
processed_at: '2026-08-05T20:37:56-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Motion Forcing

## 一句话核心

这篇 paper 想解决一件事：现在的 video generation model 在复杂场景（多辆车博弈、碰撞、机器人抓东西）下，画面好看但物理规律乱套。作者把生成分成三层"漏斗"——先画骨架、再填几何、最后画皮——强制模型先想清楚 3D 几何再渲染 RGB。这样画面保住的同时，物理一致性也保住了。

## 为什么这是个真问题

你拿一个 Sora 级别的 model 生成一辆车变道插队，画面是好看的，但仔细看可能：
- 车突然出现在某个位置，没有 inertia（不该瞬间变方向）
- 两辆车重叠了但谁前谁后不对
- 撞车后没有正确的碰撞反应

根因是 end-to-end 模型把"物体往哪走"（dynamics）和"物体长啥样"（appearance）混在同一个 loss 里算。模型偷懒偏好的是 **texture 这种高频细节**——gradient 容易下降、loss 容易收敛。**物理一致性**这种跨多帧的 long-term property，单帧 loss 抓不到，模型就给忽略了。

这就是论文里说的 trilemma：visual quality、physical consistency、controllability 三角平衡，场景一复杂就崩。

参考：VideoPhysics 论文系统讨论了这个问题 https://arxiv.org/abs/2406.03520

## 解法：Point-Shape-Appearance 三层

人话讲就是"先画火柴人，再画立体模型，再上色"。

### 第一层 Point：稀疏骨架
每个 object 抽象成一个**最大内切圆**，参数就两个：
- centroid $(x_t^i, y_t^i)$：圆心，控制 2D 平面位置
- radius $r_t^i$：半径，通过 perspective projection 隐式编码 **depth ordering**（远的物体投影后半径小，近的物体投影后半径大）

为什么用圆？因为它在 2D canvas 上 explicit，用户画箭头、写语言指令、写 kinematic script，都能 temporal interpolation 转成圆心轨迹。更重要的是 **derivative explicit modulation**——你想控制每帧的 instant velocity，直接对 centroid 求导就能 fine-tune。

### 第二层 Shape：dynamic depth maps
从稀疏 point 生成连续的 depth map。这是论文最关键的选择。为什么不选 segmentation 或 optical flow？

**Depth vs Segmentation**：
- depth 是连续 metric 值，编码 surface distance、occlusion ordering、spatial relationship
- segmentation 是 categorical mask，把连续空间信息丢了，只是"这是车、那是路"
- Ablation 显示 depth 完胜：FVD 157.8 vs 167.0，FVMD 205.2 vs 228.8，Physics-IQ 33.2 vs 29.7

**Depth vs Optical Flow**：
- optical flow 只是 2D pixel displacement，没 3D awareness——远物体小位移和近物体大位移在 flow 上看一样
- depth 直接在 3D metric space 里告诉你 surface 在哪
- Ablation 显示 depth 完胜：FVD 157.8 vs 173.0，Physics-IQ 33.2 vs 28.4

**关键直觉**：depth 是 control-level（稀疏）和 pixel-level（稠密）的**自然桥梁**。从 point 到 depth 都是几何，domain gap 小；从 depth 到 RGB 都是 dense，domain gap 也小。直接从 sparse point 跳到 dense RGB，那个 leap 太大，模型会偷懒。

### 第三层 Appearance：最终 RGB
有了 verified geometric layout，最后再渲染 texture/lighting/shadow/material。这步就是 CogVideoX1.5-5B-I2V 的本职工作。

参考 CogVideoX: https://arxiv.org/abs/2408.06072

## Camera Motion 怎么处理：一个关键的 trick

标准做法是把 6-DoF camera pose $(R_t, t_t)$ 和 intrinsics $K_t$ 压成 vector 通过 AdaLN 注入。这在 driving domain 有两个致命问题：

### 问题 1: entanglement
driving 数据里 camera trajectory 和 scene layout 强相关——车直行 ego 多半前进，车转弯 ego 多半转。模型直接把 camera motion 和 scene content 缠在一起，**factorization 极其困难**，没有海量数据多样性解不开。

### 问题 2: spatial precision 不够
6-DoF 压成 vector 后，网络内部要重新 expand 成 dense per-pixel displacement。这是 **lossy bottleneck**，跟 conv/attention 的 pixel-aligned 处理范式不匹配。autonomous driving 要求几何级精确的 6-DoF 控制，vector embedding 做不到。

### Motion Forcing 的解法：把 camera motion 也表示成 depth warping

公式 (1) 的整体流程：
$$W_t = \mathrm{Splat}\big(D_0, \nabla\Pi_t \circ \Pi_0^{-1}(\mathbf{u}, D_0(\mathbf{u}))\big)$$

变量解释：
- $D_0$：第一帧的 depth map
- $\Pi_0$、$\Pi_t$：frame 0 和 frame t 的 projection function
- $\mathbf{u} = (u, v)$：frame 0 中的像素坐标
- $D_0(\mathbf{u})$：像素 $\mathbf{u}$ 处的 depth 值
- $\nabla$：forward warping (splatting) 操作
- $\mathrm{Splat}(\cdot)$：forward splatting 函数

展开看公式 (2) unprojection（像素 → world）：
$$\mathbf{p}_{\mathrm{world}} = \mathbf{R}_0^\top\left(D_0(\mathbf{u}) \cdot \mathbf{K}_0^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix} - \mathbf{t}_0\right)$$

- $\mathbf{R}_0 \in \mathbb{R}^{3\times3}$：frame 0 的 rotation matrix（camera-to-world）
- $\mathbf{t}_0 \in \mathbb{R}^3$：frame 0 的 translation vector
- $\mathbf{K}_0^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix}$：把 pixel homogeneous coordinate 转到 normalized camera coordinate（ray 方向）
- 乘 $D_0(\mathbf{u})$：沿 ray 缩放到 camera coordinate 下的 3D 点
- 减 $\mathbf{t}_0$ 再乘 $\mathbf{R}_0^\top$：camera coordinate → world coordinate

公式 (3) projection（world → frame t pixel）：
$$\mathbf{u}_t = \pi\left(\mathbf{K}_t(\mathbf{R}_t\mathbf{p}_{\mathrm{world}} + \mathbf{t}_t)\right)$$

- $\mathbf{R}_t, \mathbf{t}_t, \mathbf{K}_t$：frame t 的 extrinsics 和 intrinsics
- $\pi$：perspective division（除以第三维）

最后把 $D_0(\mathbf{u})$ splat 到 $\mathbf{u}_t$ 位置，形成 warped depth $W_t$。

**人话**：你把第一帧每个像素的 depth 值，按照"如果相机走到 t 时刻位置，这个点应该出现在新画面哪个位置"重新投一遍，得到一张 dense 的 depth warping map。camera motion 就变成了**几何形变**，和 intermediate representation 同模态。

Ablation 证明这个 trick 显著优于 AdaLN：FVMD 205.2 vs 243.8，Physics-IQ 33.2 vs 29.1。

## 单个 backbone 同时干两个 stage 怎么做到的

这是架构上最巧妙的部分。作者没用两个 cascade 的 model（那样会 error propagation），用**单个 3D DiT backbone 同时处理两个 stage**。

机制核心是 **dual independent diffusion timesteps**：
- $\tau_d$：depth latent 的 noise level
- $\tau_v$：video latent 的 noise level

两条 latent stream $\mathbf{z}_{\tau_d}^d$ 和 $\mathbf{z}_{\tau_v}^v$ 沿 temporal axis 拼接后 jointly 处理。

### Dual AdaLN
公式 (5)：
$$\mathbf{h}_{1:T}' = \mathrm{LN}(\mathbf{h}_{1:T}) \odot [\gamma(\tau_d) \| \gamma(\tau_v)] + [\beta(\tau_d) \| \beta(\tau_v)]$$

- $\mathbf{h}_{1:T}$：hidden state sequence（depth 半部分 + video 半部分）
- $\mathrm{LN}$：layer normalization
- $\gamma(\cdot), \beta(\cdot)$：学到的 affine scale / shift 函数
- $\odot$：element-wise 乘法
- $\|$：沿 temporal axis 的 concatenation

**人话**：一个 transformer block 里，depth 半部分和 video 半部分 receive 不同的 scale/shift，但 **attention 层共享 representation**。这样既允许 task-specific modulation，又避免了维护两套独立 parameter 的开销。

### Forcing 训练：每次只解一个 sub-problem

每次 iteration 强制模型只学一个 stage，另一个被 push 到极端：

**Mode I: Physical Reasoning（Point + Camera → Depth）**
- $\tau_v = T_{\max}$：video latent 是纯噪声，模型完全没 RGB 信息可用
- 采样 $\tau_d \sim \mathcal{U}\{0, \dots, T_{\max}-1\}$
- 模型只能从 $\mathbf{P}$、$\mathbf{W}$、$\mathbf{I}_0$ 推理 $\mathbf{D}$

Loss 公式 (6)：
$$\mathcal{L}_{\mathrm{reason}} = \mathbb{E}\bigg[\big\|\epsilon_d - \hat{\epsilon}_d\big\|^2\bigg], \quad \mathrm{s.t.}\ \tau_v = T_{\max}$$

**Mode II: Neural Rendering（Depth → Appearance）**
- $\tau_d = 0$：depth latent 是 ground truth（clean）
- 采样 $\tau_v \sim \mathcal{U}\{0, \dots, T_{\max}-1\}$
- 模型在 perfect geometry 条件下渲染 RGB

Loss 公式 (7)：
$$\mathcal{L}_{\mathrm{render}} = \mathbb{E}\Big[\big\|\epsilon_v - \hat{\epsilon}_v\big\|^2\Big], \quad \mathrm{s.t.}\ \tau_d = 0$$

**为什么叫 Motion Forcing**：模型被 force 在两个 capability 之间交替训练。这个灵感来自 Diffusion Forcing（next-token prediction 和 full-sequence diffusion 统一）https://arxiv.org/abs/2407.01392

## Masked Point Recovery：让模型主动思考

如果 input 完整，模型会 **passive pattern matching**——直接把 input 复制到 output，不学物理定律。论文设计三种 masking 强制 active reasoning：

### Temporal Ego Masking
从 frame $t = \lfloor \tau_{\mathrm{ego}} T \rfloor$ 起 mask 掉 ego-motion，cutoff ratio $\tau_{\mathrm{ego}} \sim \mathcal{U}(0.3, 1.0)$。

**人话**：camera 后半段轨迹被抹掉，模型必须从初始 momentum 推断后续 ego 走向。强迫学 **inertia**。

### Temporal Object Masking
对 object control points $\mathbf{P}$ 在 cutoff $\tau_{\mathrm{obj}} \sim \mathcal{U}(0.3, 1.0)$ 后 temporal mask。

**人话**：抹掉某个物体后半段的轨迹，模型从 initial velocity 推断它继续怎么走。

### Spatial Object Masking
公式 (8)：
$$m_{\mathrm{spatial}}^{(i)} \sim \mathrm{Bernoulli}(1 - p_{\mathrm{drop}})$$

- $m_{\mathrm{spatial}}^{(i)} \in \{0, 1\}$：物体 $i$ 是否保留（1=保留，0=drop）
- $p_{\mathrm{drop}}$：drop probability

**人话**：以一定概率把某个物体整条轨迹抹掉。模型必须从 target depth 和**其他可见 agent 的 reactive behavior** 推断这个看不见的物体存在并影响别人。强迫学 **object permanence**（物体恒存）和 **implicit multi-agent interaction**（隐式多智能体交互）。

### 还有一个 secondary objective
training 时轨迹是 dense 的，inference 时用户给的是 sparse 的。masking 提前让模型适应这种 partial input，bridge 这个 gap。

**这个思想的直觉**：跟 BERT 的 MLM、MAE 的 masked autoencoding 异曲同工，但应用在 motion control 这个新场景。

## Inference 走两阶段串行

### Stage 1: Depth Generation
- 初始化 $\mathbf{z}^d, \mathbf{z}^v \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- DDIM 采样**只对 depth stream denoise**，$\tau_v$ 固定在 $T_{\max}$（video noise 全程不变）
- 模型在 Mode I 下从 $\mathbf{P}$、$\mathbf{W}$ 合成 clean depth latent $\hat{\mathbf{z}}^d$

### Stage 2: Appearance Synthesis
- depth latent 固定到 $\hat{\mathbf{z}}^d$（$\tau_d = 0$）
- $\mathbf{z}^v$ 用 fresh noise 重新初始化
- 运行第二个 DDIM loop 对 video stream denoise（Mode II）
- 生成的 depth 当 fixed geometric blueprint

**人话好处**：中间 depth $\hat{\mathbf{D}}$ 可视化、可验证、可编辑。用户 commit 到昂贵的 rendering 前，可以先看 depth 对不对，甚至手动改 depth（删个 agent、挪个位置）再 re-render。这对 safety-critical 的 autonomous driving 是大杀器。

## 实验结果用大白话讲

### Waymo 100 个测试视频对比

| Method | FVD↓ | FVMD↓ | Physics-IQ↑ |
|---|---|---|---|
| MOFA-Video | 272.6 | 421.3 | 21.6 |
| Seed Dance 2.0（闭源） | 112.5 | 345.6 | 30.5 |
| Wan 2.6（闭源） | 118.3 | 316.2 | 31.2 |
| One-stage*（无 depth） | 152.4 | 218.4 | 28.7 |
| **Motion Forcing** | **157.8** | **205.2** | **33.2** |

关键观察：
1. **Seed Dance 2.0 / Wan 2.6 的 FVD 更低**（112.5, 118.3 vs 157.8）——大规模 pretraining 在 perceptual quality 上有优势。**但 FVMD 和 Physics-IQ 都不如 Motion Forcing**。说明 perceptual quality 替代不了物理一致性，闭源大模型在 physical reasoning 上还有 gap。
2. **One-stage\* 变体 FVD 反而更低**（152.4 < 157.8），但 FVMD 和 Physics-IQ 显著退化。这强烈证明 **intermediate depth 是 motion coherence 和 physical plausibility 的关键**，代价是稍微损失 distributional similarity。
3. **MOFA-Video 全面最差**，印证 coarse optical flow + softmax splatting 的 mismatch 在复杂 multi-agent 场景下崩盘。

### Ablation

| 替换项 | FVD↓ | FVMD↓ | Physics-IQ↑ |
|---|---|---|---|
| 完整 Motion Forcing | 157.8 | 205.2 | 33.2 |
| Segmentation 替代 depth | 167.0 | 228.8 | 29.7 |
| Optical Flow 替代 depth | 173.0 | 224.3 | 28.4 |
| Softmax Splatting 替代 instance flow | 160.3 | 251.7 | 28.5 |
| AdaLN 替代 depth warping | 159.6 | 243.8 | 29.1 |

**人话**：
- Depth > Segmentation > Optical Flow（连续 3D 几何完胜）
- Instance flow > Softmax Splatting（coarse pixel warping 破坏 temporal coherence）
- Depth warping > AdaLN embedding（spatially explicit 6-DoF conditioning 必不可少）

### 跨域 generality
- **Physion**（rigid-body physics）：多米诺骨牌场景，MOFA-Video 即使 fine-tune 在论文数据集上仍崩，Motion Forcing 保持 plausible collision dynamics。证明 Point-Shape-Appearance hierarchy 能 transfer 到通用物理场景。
- **Jaco Play**（robotic manipulation）：directional inputs 控制机械手抓取，模型灵活按方向移动。证明 point-based control primitive 跨域灵活。

## 几个 takeaway 直觉

### 1. Decoupling 优于 entanglement
end-to-end model 看起来"统一"，但在物理一致性这种 long-term property 上吃亏。显式 decompose 让 verification 成为可能，safety-critical domain 必需。

### 2. Intermediate representation 的选择极其关键
depth 在 driving domain 是自然几何桥梁。换到 fluid simulation，可能是 velocity field；换到 articulated body，可能是 joint angle。**关键是找该 domain 的"自然中间表示"**。

### 3. Domain gap 最小化是 decomposition 设计的核心原则
Point → Shape → Appearance 每一步都是"相邻层的自然桥梁"。如果哪一步 leap 太大（比如直接 sparse point 到 dense RGB），模型会偷懒 collapse。

### 4. Dual-timestep 是 unified backbone 的关键 trick
单 model 处理两 stage 听起来不可能，但 dual AdaLN + mode switching 实现了 parameter sharing 但 task-specific modulation。比 cascade model 优越在于避免了 error propagation。

### 5. Masking 是 active reasoning 的催化剂
这个思想类似 BERT MLM、MAE masked autoencoding，但应用在 motion control。任何 sparse control → dense output 的任务都可以考虑 training 时破坏 input 强制 model 推理。

### 6. Camera motion 用 geometric representation 而非 embedding
非常 specific to driving domain 的洞察。driving 数据 camera 与 scene 强相关，必须用 geometric representation 才能 decouple。其他 domain（比如 VR、cinematography）可能不同。

### 7. Trilemma 是个 useful framing
把 video generation 三个目标的平衡明确化为 trilemma，让后续工作有了清晰的 evaluation axes：visual quality / physical consistency / controllability。

### 8. Inference 的两阶段串行带来 interpretability + editability
中间 depth 是 verifiable、editable 的，这对 safety-critical domain 是大杀器。用户可以在 commit 到 expensive rendering 前先 inspect。

## 局限性

论文自己承认两个 open problem：
1. **Dense non-motorized traffic**（人群、自行车群）：sparse point control 难以 capture 大量小 agents 的多样 motion patterns。Point abstraction 的固有限制。
2. **Highly occluded multi-agent interactions**：depth representation 在多 vehicle 显著重叠时 occlusion ordering 歧义性增加。

可能的未来方向：
- Multi-resolution point representation（小物体用多个细粒度 point）
- Explicit occlusion reasoning module
- 或者引入 mesh / voxel 等更结构化的中间表示

## 参考

- 项目仓库: https://github.com/Tianshuo-Xu/Motion-Forcing
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- CogVideoX: https://arxiv.org/abs/2408.06072
- MoFA-Video: https://arxiv.org/abs/2405.20222
- STANCE: https://arxiv.org/abs/2510.14588
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- Wan 2.1: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- DragAnything: https://arxiv.org/abs/2403.07420
- DragNUWA: https://arxiv.org/abs/2308.08089
- DriveDreamer: https://arxiv.org/abs/2309.09777
- GAIA-1: https://arxiv.org/abs/2309.17080
- Gen3C: https://arxiv.org/abs/2503.11594
- PhysGen: https://arxiv.org/abs/2409.18964
- Physion: https://arxiv.org/abs/2106.08261
- Jaco Play: https://github.com/clvrai/clvr_jaco_play_dataset
- VGGT: https://arxiv.org/abs/2503.11651
- Waymo Open Dataset: https://waymo.com/open/
- Driving Dojo: https://drivingdojo.github.io/
- FVD: https://arxiv.org/abs/1812.01717
- FVMD: https://arxiv.org/abs/2407.16124
- Physics-IQ: https://arxiv.org/abs/2501.09038
- VideoPhysics: https://arxiv.org/abs/2406.03520
- PhysDreamer: https://arxiv.org/abs/2404.13026
- Physics3D: https://arxiv.org/abs/2406.04338
- VideoRepa: https://arxiv.org/abs/2505.23656
- AnimateDiff: https://arxiv.org/abs/2307.04725
- DeepSpeed ZeRO: https://arxiv.org/abs/1911.04611
- AdamW: https://openreview.net/forum?id=Bkg6RiCqY7
- Stable Diffusion: https://arxiv.org/abs/2112.10752

---

# Motion Forcing 论文深度解读

这是 HKUST(GZ) Ying-cong Chen 组的工作。让我从 motivation、架构、训练策略到实验结果逐层拆解。

## 1. 问题定位：trilemma 与 dynamics-appearance 的 entanglement

现有 video generation models 在简单场景下能同时满足三个目标:
- **Visual quality**: texture/lighting 的高频细节
- **Physical consistency**: inertia, collision dynamics, object permanence 等物理定律
- **Controllability**: 用户对 motion 的精确控制

但场景复杂化后(dense traffic, collisions) 这种 balance 就崩塌。作者把根因归结为 end-to-end model 中 dynamics 和 appearance 的 entanglement: **模型天然倾向于最小化高频视觉 loss**(texture 像素差异容易 gradient 下降)，**忽略 long-term 物理一致性**(物理约束跨越多帧, 单帧 loss 难以捕获)。

先前的两类尝试都没解决这个 gap:
- **MoFA-Video** [31] 用 coarse optical flow + softmax splatting: coarse 控制信号与 pixel-precise warping 机制之间存在 mismatch, 复杂场景下模型为了 visual stability 直接忽略用户指令
- **STANCE** [10] 直接生成 RGB, 用 auxiliary motion loss 作 soft constraint: sparse control 与 dense pixel 之间 learning gap 仍然没有 bridge

## 2. 核心 idea: Point-Shape-Appearance 三级解耦

insight 是把生成分解成三个**密度递增**、**相邻 stage 之间 domain gap 最小**的阶段:

### Point: sparse control signal
每个 object $i$ 在 frame $t$ 抽象为**最大内切圆**, 参数化为:
- centroid $(x_t^i, y_t^i)$: 控制 planar motion
- radius $r_t^i$: 通过 perspective projection 隐式编码 **depth ordering**(远的物体投影后半径小)

这些 primitives 渲染到 canvas 上, 再用 VAE encode 成 latent。

这个设计的好处很巧妙:
1. centroid + radius 在 2D canvas 上是 explicit 的几何参数, 用户可以画箭头、写 language instruction、写 script, 都能通过 temporal interpolation 转换过来
2. **支持对 derivatives 的 explicit modulation**: 可以对每帧 instant velocity 直接 fine-tune, 这对 kinematic properties 控制很重要

### Shape: dynamic depth maps
为什么选 depth 作 intermediate 而非 segmentation 或 optical flow? 关键在 depth 捕获**连续 3D 几何信息**:
- surface distances (metric space)
- spatial relationships
- occlusion ordering

这是 control-level reasoning(稀疏)和 pixel-level rendering(稠密)的**自然桥梁**。ablation 证实 depth 显著优于 segmentation(FVD 167.0 vs 157.8, Physics-IQ 29.7 vs 33.2)和 optical flow(FVD 173.0, Physics-IQ 28.4)。

### Appearance: 最终 RGB
high-fidelity texture/lighting/shadow/material, 条件是 verified geometric layout。

## 3. Camera Motion Encoding: Depth Warping 而非 AdaLN Embedding

这部分是论文一个重要 contribution。标准做法是把 extrinsics $(R_t, t_t)$ 和 intrinsics $K_t$ 通过 cross-attention / AdaLN 注入。作者论证这在 driving domain 有两个致命问题:

### 问题 1: entanglement under limited data diversity
driving 数据集中, camera trajectory 与 scene layout / agent behavior 强相关(车直行时 ego 多半前进, 车转弯时 ego 多半转)。模型会把 camera motion 与 scene content 缠在一起, **factorization 极其困难**, 缺乏大规模数据多样性无法解。

### 问题 2: insufficient spatial precision for 3D-grounded control
6-DoF transformation 压缩成 vector, network 内部要重新 expand 成 dense per-pixel displacement。这是 **lossy bottleneck**, 与 conv/attention 的 pixel-aligned 处理范式不匹配。

### 解决方案: 把 camera motion 也表示成 warped depth map $W_t$

公式 (1) 给出整体流程:
$$W_t = \mathrm{Splat}\big(D_0, \nabla\Pi_t \circ \Pi_0^{-1}(\mathbf{u}, D_0(\mathbf{u}))\big)$$

变量解释:
- $D_0$: 第一帧的 depth map
- $\Pi_0$: frame 0 的 projection function
- $\Pi_t$: frame t 的 projection function
- $\mathbf{u} = (u, v)$: frame 0 中的像素坐标
- $D_0(\mathbf{u})$: 像素 $\mathbf{u}$ 处的 depth 值
- $\nabla$: forward warping (splatting) 操作
- $\mathrm{Splat}(\cdot)$: forward splatting 函数

公式 (2) unprojection(像素 → world):
$$\mathbf{p}_{\mathrm{world}} = \mathbf{R}_0^\top\left(\mathbf{D}_0(\mathbf{u}) \cdot \mathbf{K}_0^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix} - \mathbf{t}_0\right)$$

逐项展开:
- $\mathbf{R}_0 \in \mathbb{R}^{3\times3}$: frame 0 的 rotation matrix(camera-to-world)
- $\mathbf{t}_0 \in \mathbb{R}^3$: frame 0 的 translation vector(camera origin in world)
- $\mathbf{K}_0 \in \mathbb{R}^{3\times3}$: frame 0 的 intrinsics matrix
- $\mathbf{K}_0^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix}$: 把 pixel homogeneous coordinate 转到 normalized camera coordinate(ray 方向)
- 乘以 $D_0(\mathbf{u})$: 沿 ray 缩放到 camera coordinate 下的 3D 点
- 减 $\mathbf{t}_0$ 再乘 $\mathbf{R}_0^\top$(等价于 $\mathbf{R}_0^{-1}$): camera coordinate → world coordinate, 得到 $\mathbf{p}_{\mathrm{world}}$

公式 (3) projection 到 frame t(world → 像素):
$$\mathbf{u}_t = \pi\left(\mathbf{K}_t(\mathbf{R}_t\mathbf{p}_{\mathrm{world}} + \mathbf{t}_t)\right)$$

- $\mathbf{R}_t, \mathbf{t}_t, \mathbf{K}_t$: frame t 的 extrinsics 和 intrinsics
- $\mathbf{R}_t\mathbf{p}_{\mathrm{world}} + \mathbf{t}_t$: world → frame t 的 camera coordinate
- $\mathbf{K}_t(\cdot)$: camera coordinate → image homogeneous coordinate
- $\pi$: perspective division(除以第三维), 得到 frame t 上的 pixel 坐标 $\mathbf{u}_t$

最后把 depth 值 $D_0(\mathbf{u})$ splat 到 $\mathbf{u}_t$ 位置, 形成 $W_t$。

**这套设计的关键 intuition**: camera motion 被显式表现为**几何变形**, 与 intermediate representation(depth)同模态。network 不需要学习"6-DoF → dense displacement"的抽象映射, 而是直接在 depth space 上观察 camera 带来的几何效果。

## 4. 统一 Diffusion Backbone: Dual-Timestep Motion Forcing

这是架构上最巧妙的部分。论文没有用两个 cascaded model(那样会 error propagation), 而是**单个 3D DiT backbone 同时处理两个 stage**, 机制核心是 **dual independent diffusion timesteps**:

- $\tau_d$: depth latent 的 noise level
- $\tau_v$: video latent 的 noise level

设 $\mathbf{z}_{\tau_d}^d$ 和 $\mathbf{z}_{\tau_v}^v$ 分别是 D 和 V 的 VAE-encoded latents, 被加噪到对应 level。两个 latent stream **temporal concatenation** 后 jointly 处理。

统一 objective 公式 (4):
$$\mathcal{L} = \mathbb{E}_{\mathbf{D}, \mathbf{V}, \tau_d, \tau_v, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\big(\mathbf{z}_{\tau_d}^d \oplus \mathbf{z}_{\tau_v}^v, \tau_d, \tau_v, \mathbf{I}_0, \mathbf{P}, \mathbf{W}\big)\right\|^2\right]$$

- $\oplus$: temporal concatenation(沿时间维度拼接)
- $\mathbf{I}_0$: reference image(首帧)
- $\mathbf{P}$: point control
- $\mathbf{W}$: camera depth warping
- 这些 condition 都先 VAE-encode, 再沿 channel 维 concat 到 noisy input

### Dual Adaptive Layer Normalization

公式 (5):
$$\mathbf{h}_{1:T}' = \mathrm{LN}(\mathbf{h}_{1:T}) \odot [\gamma(\tau_d) \| \gamma(\tau_v)] + [\beta(\tau_d) \| \beta(\tau_v)]$$

- $\mathbf{h}_{1:T}$: hidden state sequence(depth 半部分 + video 半部分)
- $\mathrm{LN}$: layer normalization
- $\gamma(\cdot)$: 学到的 affine scale 函数
- $\beta(\cdot)$: 学到的 affine shift 函数
- $\odot$: element-wise multiplication
- $\|$: 沿 temporal(sequence) axis 的 concatenation

intuition: 一个 transformer block 内, **depth 半部分和 video 半部分 receive 不同的 scale/shift**, 但 **attention 层共享 representation**。这避免了维护两套独立 parameter 的开销, 同时允许 task-specific modulation。输出层 norm_out 依次应用两个 embedding 产生最终 prediction。

### Forcing Strategy: 模式切换训练

受 **Diffusion Forcing** [7] 启发(把 next-token prediction 和 full-sequence diffusion 统一), 这里采用 stochastic mode-switching。每次 iteration 强制模型只解一个 sub-problem:

**Mode I: Physical Reasoning** (Point + Camera → Depth)
- 设 $\tau_v = T_{\max}$(video latent 是纯噪声, 没有 RGB 信息可用)
- 采样 $\tau_d \sim \mathcal{U}\{0, \dots, T_{\max}-1\}$
- 模型只能从 $\mathbf{P}, \mathbf{W}, \mathbf{I}_0$ 推理出 $\mathbf{D}$

Loss 公式 (6):
$$\mathcal{L}_{\mathrm{reason}} = \mathbb{E}\bigg[\big\|\epsilon_d - \hat{\epsilon}_d\big\|^2\bigg], \quad \mathrm{s.t.}\ \tau_v = T_{\max}$$

**Mode II: Neural Rendering** (Depth → Appearance)
- 设 $\tau_d = 0$(depth latent 是 ground truth, clean)
- 采样 $\tau_v \sim \mathcal{U}\{0, \dots, T_{\max}-1\}$
- 模型在 perfect geometry 条件下 hallucinate texture/lighting/material

Loss 公式 (7):
$$\mathcal{L}_{\mathrm{render}} = \mathbb{E}\Big[\big\|\epsilon_v - \hat{\epsilon}_v\big\|^2\Big], \quad \mathrm{s.t.}\ \tau_d = 0$$

这两个 mode 交替训练, $\epsilon_\theta$ 在**同一 latent space 内**同时充当 physics engine 和 renderer。这是为什么叫 "Motion Forcing": 模型被 force 在两种 capability 之间交替。

## 5. Masked Point Recovery: 强制 active physical reasoning

这是论文的另一个关键 contribution。直觉是: 如果不故意破坏 input, 模型会**passive 模式**地做 pattern matching, 而不学习物理定律。论文设计三种 masking:

### Temporal Ego Masking
从 frame $t = \lfloor \tau_{\mathrm{ego}} T \rfloor$ 起 mask 掉 ego-motion conditioning, cutoff ratio $\tau_{\mathrm{ego}} \sim \mathcal{U}(0.3, 1.0)$。

intuition: 模型必须从**初始 momentum** 推断后续 ego-trajectory。这强迫模型内化 **inertia**(惯性)概念。

### Temporal Object Masking
对 object control points $\mathbf{P}$ 在 cutoff $\tau_{\mathrm{obj}} \sim \mathcal{U}(0.3, 1.0)$ 后 temporal mask。

intuition: 模型从 initial velocity 预测 ongoing object motion, 同样强迫 inertia 学习。

### Spatial Object Masking
公式 (8) 独立 drop 每个物体 $i$ 的整条 trajectory:
$$m_{\mathrm{spatial}}^{(i)} \sim \mathrm{Bernoulli}(1 - p_{\mathrm{drop}})$$

- $m_{\mathrm{spatial}}^{(i)} \in \{0, 1\}$: 物体 $i$ 是否保留(1=保留, 0=drop)
- $p_{\mathrm{drop}}$: drop probability
- $\mathrm{Bernoulli}(1 - p_{\mathrm{drop}})$: 以 $1 - p_{\mathrm{drop}}$ 的概率取 1

intuition: 模型必须从 target depth $\mathbf{D}$ 和**其他可见 agents 的 reactive behavior** 推断被 mask 掉的物体的存在和 trajectory。这强迫模型学习 **object permanence**(物体恒存)和**隐式 multi-agent 交互**(周围车辆对被遮挡车辆的反应)。

**这套策略还有一个 secondary objective**: 桥接 dense training trajectory 与 sparse real-world inference 之间的 gap。用户在真实使用时往往只提供稀疏 trajectory, training-time 的 masking 让模型提前适应这种 partial input。

## 6. Inference: 两阶段串行

**Stage 1: Depth Generation**
1. 初始化 $\mathbf{z}^d, \mathbf{z}^v \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. 执行 DDIM sampling loop, **只对 depth stream denoise**($\tau_v$ 固定在 $T_{\max}$)
3. 模型在 Mode I 下, 从 $\mathbf{P}$ 和 $\mathbf{W}$ 合成 clean depth latent $\hat{\mathbf{z}}^d$
4. RGB noise $\mathbf{z}^v$ 全程保持 constant(即保持纯噪声)

**Stage 2: Appearance Synthesis**
1. depth latent 固定到 $\hat{\mathbf{z}}^d$(设 $\tau_d = 0$)
2. $\mathbf{z}^v$ 用 fresh noise 重新初始化
3. 运行第二个 DDIM loop 对 video stream denoise(模型在 Mode II)
4. 生成的 depth 作为 fixed geometric blueprint

这种 pipeline 提供了 **interpretability**: 中间 depth $\hat{\mathbf{D}}$ 是 verifiable 的 3D scene layout, 用户可以在 commit 到昂贵的 rendering 前先 inspect。它还提供 **natural editing interface**: 用户可手动修改 depth(删除/重定位 agent), 再 re-render 出 consistent appearance。

## 7. 训练细节

- **Base model**: CogVideoX1.5-5B-I2V [49] fine-tune
- **Hardware**: 8× NVIDIA H100, DeepSpeed ZeRO Stage-2 [33], bfloat16 mixed precision
- **Optimizer**: AdamW [26], lr = $1 \times 10^{-5}$, $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay $1 \times 10^{-4}$, 100 warmup steps
- **Sample spec**: 33 frames @ 320×480 resolution (stride 2)
- **Schedule**: 10 epochs, effective batch size 8

### 数据集
- **Driving**: Waymo [36] + Driving Dojo [44] + YouTube curated
- **Physics**: Physion [2] (rigid-body physics)
- **Robotics**: Jaco Play [11] (robotic manipulation)

### 评测 Metrics
- **FVD** [39]: Fréchet Video Distance, distributional similarity
- **FVMD** [24]: Fréchet Video Motion Distance, temporal motion coherence
- **Physics-IQ** [29]: physical plausibility(基于 VideoPhysics 的物理常识 benchmark)

## 8. 实验结果深度分析

### Table 1: Waymo 100 个测试视频对比

| Method | FVD↓ | FVMD↓ | Physics-IQ↑ |
|---|---|---|---|
| MOFA-Video | 272.6 | 421.3 | 21.6 |
| Seed Dance 2.0 | 112.5 | 345.6 | 30.5 |
| Wan 2.6 | 118.3 | 316.2 | 31.2 |
| One-stage* (无 depth intermediate) | 152.4 | 218.4 | 28.7 |
| **Motion Forcing (Ours)** | **157.8** | **205.2** | **33.2** |

关键观察:
1. Seed Dance 2.0 / Wan 2.6 因为大规模 pretraining, FVD 更低(112.5, 118.3), 但 motion coherence 和 physical plausibility 都不如 Motion Forcing。这说明**perceptual quality 不能替代物理一致性**, 闭源大模型在物理 reasoning 上仍有 gap。
2. **One-stage\* 变体 FVD 反而更低(152.4 vs 157.8)**, 但 FVMD(218.4 vs 205.2)和 Physics-IQ(28.7 vs 33.2)显著差, 甚至 Physics-IQ 比 Seed Dance 和 Wan 还低。这强烈说明 **intermediate depth 是 motion coherence 和 physical plausibility 的关键**, 代价是稍微损失 distributional similarity。
3. MOFA-Video 全面最差, 印证 coarse optical flow + softmax splatting 的 mismatch 在 complex multi-agent 场景下崩盘。

### Table 2: Ablation

| Method | FVD↓ | FVMD↓ | Physics-IQ↑ |
|---|---|---|---|
| Motion Forcing | 157.8 | 205.2 | 33.2 |
| **Motion Representation** | | | |
| Segmentation (替代 depth) | 167.0 | 228.8 | 29.7 |
| Optical Flow (替代 depth) | 173.0 | 224.3 | 28.4 |
| **Object Motion Embedding** | | | |
| Softmax Splatting [31] (替代 instance flow) | 160.3 | 251.7 | 28.5 |
| **Ego Motion Encoding** | | | |
| AdaLN (替代 depth warping) | 159.6 | 243.8 | 29.1 |

逐项分析:

**Motion Representation 对比**:
- Depth vs Segmentation: depth 在 FVD(-9.2), FVMD(-23.6), Physics-IQ(+3.5)全面胜出。原因: depth 在 metric space 编码连续 surface distance 和 occlusion ordering, 而 segmentation 是 categorical mask, 丢弃 continuous spatial structure。
- Depth vs Optical Flow: depth 更优(FVD -15.2, Physics-IQ +4.8)。Optical flow 缺 3D awareness, 只是 2D pixel displacement, 无法区分"远物体小位移"和"近物体大位移"。

**Object Motion Embedding**:
- 用 Softmax Splatting 替代 instance flow: FVD 略好(160.3 vs 157.8, 可能因为 splatting 本身就是 pixel-wise 操作), 但 **FVMD 严重退化(251.7 vs 205.2)** 和 **Physics-IQ 退化(28.5 vs 33.2)**。intuition: coarse pixel-level warping 在 multi-body dynamics 下破坏 temporal coherence, 物体之间相互作用难以保持。

**Ego Motion Encoding**:
- 用 AdaLN 替代 depth warping: FVD 影响小(159.6 vs 157.8), 但 FVMD(243.8 vs 205.2)和 Physics-IQ(29.1 vs 33.2)显著退化。intuition: spatially explicit, pixel-aligned 6-DoF conditioning 对 motion coherence 和 physical reasoning 必不可少, embedding 形式把 6-DoF 压缩成 vector 是 lossy bottleneck。

## 9. 跨域 generality 验证

### Physion [2] (rigid-body physics)
Fig. 4 展示多米诺骨牌场景: "leftmost piece falls rightward, second piece falls leftward, they collide"。结果显示 MOFA-Video(即使 fine-tune 在论文数据集)在复杂 multi-object collision 下物理 incoherent, 而 Motion Forcing 保持 plausible collision dynamics。这证明 Point-Shape-Appearance hierarchy 能 transfer 到通用物理场景。

### Jaco Play [11] (robotic manipulation)
Fig. 6 通过 directional inputs 控制 robotic hand 抓取物体。模型能灵活地引导机械臂和被抓物体按指定方向移动。这证实 **point-based control primitive 跨域灵活**, 不局限于 driving。

## 10. 局限性

论文自己承认两个 open problem:
1. **Dense non-motorized traffic**(crowds of pedestrians, cyclists): sparse point control 难以 capture 大量小 agents 的多样 motion patterns。这是 point abstraction 的固有限制。
2. **Highly occluded multi-agent interactions**: depth representation 在多 vehicle 显著重叠时可能无法 resolve occlusion ordering。Depth 在大幅 occlusion 下歧义性增加。

## 11. 与相关工作脉络

让我把这篇工作放在更大的研究脉络中:

**Video Diffusion 主流演进**:
- UNet-based LDM 阶段: Stable Diffusion [35] → AnimateDiff [13] (temporal attention) → VideoCrafter [8, 9] / ModelScope [41] / Moonshot [52] / Show-1 [53]
- DiT 范式迁移: Sora [32] → CogVideoX [49] → HunyuanVideo [19] → Wan 2.1 [38], 配合 Flow Matching [22]

**Controllable Motion Generation**:
- 轨迹控制: DragNUWA [50], DragAnything [46], SG-I2V [30]
- Motion field: MoFA-Video [31] (这篇文章直接 baseline)
- Sparse anchor: STANCE [10] (sparse-to-dense anchored encoding)

**World Models for Driving**:
- DriveDreamer [42], DriveDiffusion [21], GAIA-1 [15], GenAD [47]
- 3D-aware: Gen3C [34] (3D-informed cache), NeoVerse [48] (monocular 4D), PhiGenesis [27] (stereo forcing), GeoDrive [6] (3D geometry-informed)

**Physically Coherent Generation**:
- Hybrid: PhysGen [25] (集成 explicit physics engine)
- LLM/VFM 推理: Think Before You Diffuse [54], VideoRepa [56]
- Latent physics: PhysVideoGenerator [20], STANCE [10]
- 3D 物理交互: PhysDreamer [55], Physics3D [23]

Motion Forcing 的独特定位: **decoupled framework**, 显式把 physical reasoning 与 visual rendering 分离, 通过 depth 中间表示 bridge。这避开了 hybrid 方法对外部 simulator 的依赖, 同时避开了 latent-based 方法无法显式 verify 的弱点。

## 12. 我的整体 intuition

这篇工作最值得借鉴的点:

1. **Representation 的层次设计**: Point-Shape-Appearance 不是随便选的, 是 domain gap 最小化下的密度递增。每一层都是相邻层的"自然桥梁", 避免 sparse-to-dense 的巨大 leap。

2. **Dual-timestep 是 unified backbone 的关键 trick**: 单 model 处理两 stage 听起来不可能, 但通过 dual AdaLN + 模式切换, 实现了 parameter sharing 但 task-specific modulation。比 cascaded model 优越在于避免了 error propagation。

3. **Masked Point Recovery 的"主动 vs 被动"洞察**: 模型不 mask 就会 passive pattern matching, mask 后被迫 active reasoning。这跟 BERT 的 MLM、MAE 的 masked autoencoding 思想相通, 但应用在 motion control 这个全新场景。

4. **Camera Motion 用 depth warping 而非 embedding**: 这是非常 specific to driving domain 的洞察。driving 数据 camera 与 scene 强相关, 用 geometric representation 才能 decouple。

5. **Trilemma 概念本身**: 把 video generation 三个目标的平衡明确化为 trilemma, 是个 useful 的 framing, 让后续工作有了清晰的 evaluation axes。

参考链接:
- 项目仓库: https://github.com/Tianshuo-Xu/Motion-Forcing
- Diffusion Forcing [7]: https://arxiv.org/abs/2407.01392
- CogVideoX [49]: https://arxiv.org/abs/2408.06072
- MoFA-Video [31]: https://arxiv.org/abs/2405.20222
- STANCE [10]: https://arxiv.org/abs/2510.14588
- Sora [32]: https://openai.com/research/video-generation-models-as-world-simulators
- Wan 2.1 [38]: https://arxiv.org/abs/2503.20314
- HunyuanVideo [19]: https://arxiv.org/abs/2412.03603
- Flow Matching [22]: https://arxiv.org/abs/2210.02727
- DragAnything [46]: https://arxiv.org/abs/2403.07420
- DragNUWA [50]: https://arxiv.org/abs/2308.08089
- DriveDreamer [42]: https://arxiv.org/abs/2309.09777
- DriveDiffusion [21]: https://arxiv.org/abs/2310.07771
- GAIA-1 [15]: https://arxiv.org/abs/2309.17080
- Gen3C [34]: https://arxiv.org/abs/2503.11594
- NeoVerse [48]: https://arxiv.org/abs/2601.00393
- PhiGenesis [27]: https://arxiv.org/abs/2509.20251
- GeoDrive [6]: https://arxiv.org/abs/2505.22421
- PhysGen [25]: https://arxiv.org/abs/2409.18964
- Physion [2]: https://arxiv.org/abs/2106.08261
- Jaco Play [11]: https://github.com/clvrai/clvr_jaco_play_dataset
- VGGT [40]: https://arxiv.org/abs/2503.11651
- Waymo Open Dataset [36]: https://waymo.com/open/
- Driving Dojo [44]: https://drivingdojo.github.io/
- FVD [39]: https://arxiv.org/abs/1812.01717
- FVMD [24]: https://arxiv.org/abs/2407.16124
- Physics-IQ [29]: https://arxiv.org/abs/2501.09038
- VideoPhysics [1]: https://arxiv.org/abs/2406.03520
- PhysDreamer [55]: https://arxiv.org/abs/2404.13026
- Physics3D [23]: https://arxiv.org/abs/2406.04338
- VideoRepa [56]: https://arxiv.org/abs/2505.23656
- Think Before You Diffuse [54]: https://arxiv.org/abs/2505.21653
- AnimateDiff [13]: https://arxiv.org/abs/2307.04725
- Animate Anyone [16]: https://arxiv.org/abs/2311.17117
- SG-I2V [30]: https://arxiv.org/abs/2411.04989
- DeepSpeed ZeRO [33]: https://arxiv.org/abs/1911.04611
- AdamW [26]: https://openreview.net/forum?id=Bkg6RiCqY7
- Stable Diffusion [35]: https://arxiv.org/abs/2112.10752
- VideoCrafter [8]: https://arxiv.org/abs/2310.19512
- ModelScope [41]: https://arxiv.org/abs/2308.06571
- Moonshot [52]: https://arxiv.org/abs/2401.01827
- Show-1 [53]: https://arxiv.org/abs/2309.15818

---

# 总结性的几个思考

这篇 paper 的几个重要启发:

1. **Decoupling 优于 entanglement**: 即使 end-to-end model 看起来"统一", 在物理一致性这种 long-term property 上反而吃亏。显式 decompose 让 verification 成为可能, 这是 safety-critical domain(autonomous driving, robotics)的关键需求。

2. **Intermediate representation 的选择极其关键**: depth 在 driving domain 是自然的几何桥梁。换到其他 domain(eg. fluid simulation, articulated body), 应该思考该 domain 的"自然中间表示"是什么。

3. **Forcing 的命名很贴切**: 模型被 force 在两个 capability 之间交替训练, 类似于 curriculum learning。单 stream 训练可能 collapse 到某种退化解, dual-timestep forcing 强制探索两个 mode 的解空间。

4. **Masking 是 active reasoning 的催化剂**: 这个思想可以推广到很多 generative model。任何 sparse control → dense output 的任务, 都可以考虑在 training 时破坏 input 强制 model 推理。

5. **Open problem 方向**: dense non-motorized traffic 和重度 occlusion 的 multi-agent 场景仍然未解。可能需要 multi-resolution point representation 或者 explicit occlusion reasoning module。
