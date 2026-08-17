---
source_pdf: Physical Simulator In-the-Loop Video Generation.pdf
paper_sha256: 7f54b6114da1b4580dfc026a2fc6a036dff12153e63498964900b9b7563b807d
processed_at: '2026-08-06T03:19:56-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PSIVG

好, 换个姿势, 咱们像在咖啡馆白板上画图那样讲。

---

## 这篇paper到底在搞啥

你玩过 Sora 或者 Runway 这些 video generator 没? 画面是真好看, 但是仔细一看, 球撞 pin 之后pin会凭空消失, 保龄球会突然飞起来, 人走路会穿模。**好看是好看, 物理全是错的**。

为啥? 因为这些 model 本质上就是在学"pixel 之间的 statistical correlation"。它见过一万张球滚动的图, 所以能画出球滚动的样子, 但它从来没有真正理解"球为啥会滚"——gravity、momentum、collision 这些概念它一点都没有。

PSIVG 的 idea 特别暴力: **既然 generator 自己不懂物理, 那就在它生成 video 的过程中, 塞一个真正的 physics engine 进去, 让 engine 来"指挥"它怎么动**。

这就好比让一个只会画画的小孩 (diffusion model) 画一个保龄球撞 pin 的场景。小孩画得很好看但动作乱七八糟。现在你旁边站一个物理老师 (simulator), 老师先用真物理算好每帧每秒球和 pin 应该在哪, 然后告诉小孩"你就照这个轨迹画, 别自己瞎发挥"。小孩画出来的就既有美术功底又符合物理了。

---

## 整个pipeline, 一图流讲完

```
Text Prompt
    │
    ▼
[SD3 生成图 + CogVideoX] ──→ Template Video (好看但物理错)
                                    │
                                    ▼
                          [Perception Pipeline]
                          把 2D video "翻译"成 3D 场景
                          • 物体的 3D mesh
                          • 背景的 3D geometry
                          • 相机怎么动的
                          • 物体初始速度多少
                                    │
                                    ▼
                          [MPM Physics Simulator]
                          在 Taichi 里跑真物理
                          • 放好物体, 给密度/弹性等参数
                          • 给初始速度
                          • 跑 forward simulation
                          • 得到物理正确的轨迹
                                    │
                                    ▼
                          [Mitsuba 渲染]
                          渲染出 RGB + mask + pixel correspondence
                          (注意: 画面很丑, 但动得对)
                                    │
                                    ▼
                          [Go-with-the-Flow]
                          把 simulator 的 motion 作为 optical flow
                          去 guide video generator 重新生成
                                    │
                                    ▼
                          [TTCO 优化]
                          test-time 微调一下 text embedding
                          让物体 texture 在旋转时不闪烁
                                    │
                                    ▼
                          Final Video (好看 + 物理对)
```

核心思想就一句话: **generator 负责"好看", simulator 负责"动得对", 两者凑一块儿**。

---

## 几个关键技术点, 拆开讲

### 1. 为什么需要先生成一个 "错的" template video?

你可能会想: 既然最后要靠 simulator 指挥, 为啥不直接从 prompt 跳到 simulator?

因为 simulator 是个 "瞎子"。它不知道你 prompt 里说的 "a teddy bear falling on a wooden floor" 长啥样。它需要有人告诉它:
- 场景里有啥物体
- 物体长啥样 (3D mesh)
- 背景长啥样
- 相机怎么动的
- 物体一开始往哪跑

这些信息从哪来? 只能从某个 visual output 里 "perceive" 出来。所以先让 video generator 生成一个 "错的但好看的" template, 然后从这个 template 里 extract 出场景信息, 再喂给 simulator。

**这其实是个很 general 的 pattern**: learnable model 负责 "perception + generation", classical system 负责 "constraint enforcement"。两者互补。

### 2. Perception Pipeline: 从烂 video 里抠出 3D 信息

这是最难的一步。你想, 一个物理都错的 video, 你要从中重建出 3D mesh, 这不是扯淡吗?

作者发现一个反直觉的事: **用 single image (第一帧) 重建 mesh, 比用多帧重建更靠谱**。

为啥? 因为 video generator 在不同 frame 之间几何不一致。你拿多帧去做 multi-view reconstruction, 假设是"这些 frame 是同一个 3D 物体的不同视角", 但这个假设 video generator 恰好违反了——它每帧的 3D 几何都在飘。所以 multi-view 方法直接崩掉。

用 InstantMesh 这种 single-image-to-3D 的 model 反而稳, 因为它靠的是 pretrained 的 object prior, 而非 frame 间的 consistency。

具体 chain 是:
- **Grounding DINO**: 从 prompt 检测物体在哪
- **SAM 2**: 在 video 里把物体 mask 出来, 一帧帧 propagate
- **XMem**: 长时序 memory-based segmentation
- **InstantMesh**: 从第一帧 crop 出来重建 3D mesh

背景和相机运动用 NVIDIA 的 **ViPE**, 它做 4D reconstruction, 输出 background point cloud 和 camera poses。

### 3. 物理参数怎么估? 让 GPT-5 来猜

Simulator 需要物体的 density、Young's modulus 这些物理参数。你没法从 video 里直接 measure, 怎么办?

作者让 **GPT-5 看第一帧图**, 然后prompt 它: "这玩意是啥材质? 弹性如何? 表面粗糙吗?"

但直接让 LLM 输出数字不靠谱, 它可能说 density = 500 kg/m³, 下次又说 5000。所以做了个 **hierarchical prompting**:

```
第一轮: 问定性属性
  - 物体是啥组成的 (木头/金属/橡胶...)
  - 弹性如何 (高/中/低)
  - 表面粗糙度如何

第二轮: 把定性标签映射到定量参数
  - "木头" → density = 700 kg/m³, Young's modulus = 10 GPa
  - "橡胶" → density = 1100 kg/m³, Young's modulus = 0.01 GPa
```

这个 pattern 很实用: **LLM 不擅长输出 calibrated 数值, 但擅长输出 qualitative 判断, 你可以用一个 lookup table 把 qualitative 变 quantitative**。

### 4. MPM Simulator: 为什么不用 PyBullet/MuJoCo?

**Material Point Method (MPM)** 是 particle-based 的物理模拟方法, 原本是 Disney 做雪的效果发明的 (Stomakhin et al. 2013, 《Frozen》里的雪就是它)。

它比 rigid body simulator 的优势:
- 能处理 deformable body (软体、布料)
- 能处理 fracture (碎裂)
- 能处理 fluid-like 行为

劣势:
- 不能很好处理 articulated structure (人、机器人这种有关节的)
- 计算量大

作者用 Taichi 实现的 MPM。Taichi 是个很 clever 的 DSL, 专门写 spatially sparse 的并行计算, MPM 这种 particle-grid hybrid 方法用 Taichi 写特别顺。

### 5. Go-with-the-Flow: 怎么把 simulator 的 motion "灌"给 generator?

直接把 simulator 的 rendered RGB 当 conditional input 喂给 video generator? 不行, 因为 simulator 渲染出来的画面太丑, 风格和真实 video 差太远。

作者用了一个叫 **Go-with-the-Flow (GwtF)** 的方法。它的核心 idea: **与其 condition on RGB, 不如 condition on motion, 用 optical flow**。

具体做法:
1. 从 simulator 渲染的 RGB 用 RAFT 算 optical flow (前景物体的物理运动)
2. 从 template video 用 RAFT 算 optical flow (保留背景和相机的运动)
3. 用 segmentation mask 把两个 flow 融合: 前景用 simulator 的, 背景用 template 的
4. 用这个 hybrid flow 去 warp noise latent, 让 diffusion 从一个 motion-correlated 的 noise 开始生成

为啥用 RAFT 算 flow, 而不直接用 simulator 给的 pixel correspondence? 因为 GwtF 是在 RAFT flow 上 train 的, 分布匹配更好。这是一个 "train-test distribution alignment" 的细节考虑。

### 6. TTCO: 最聪明的一步

即使有准确的 motion guidance, generator 还是会有个毛病: **texture 闪烁**。

想象一个旋转的骰子, 每面点数不同。即使旋转轨迹完全正确, generator 可能在某些 frame "幻觉" 出错误的点数, 因为它没有机制 enforce "frame 1 的这个 pixel 和 frame t 的那个 pixel 是同一个物理点, 应该颜色一样"。

TTCO 的 idea: **simulator 知道 frame 1 的 pixel A 和 frame t 的 pixel B 是同一个粒子, 那 generator 在这两个 pixel 上应该输出一样的颜色**。

#### 公式讲清楚

$$\mathcal{L}_{\text{tex}}(t) = \sum_{j=1}^{J} \left\| \left[ De\left( h_0(\hat{L}_\tau) \right) \right]_{q_{t,j}} - \left[ W_t(\hat{I}_1) \right]_{q_{t,j}} \right\|_2^2$$

逐个符号人话翻译:

- $\hat{L}_\tau$: diffusion model 在去噪第 $\tau$ 步时预测的 latent。下标 $\tau$ 是 diffusion timestep, 范围 $[0, 1000]$, 越大越 noisy
- $\hat{I}_1$: template video 的第一帧 image, 作为 texture reference
- $W_t(\hat{I}_1)$: 用 simulator 给的 pixel correspondence 把第一帧 warp 到第 $t$ 帧的结果
- $p_{1,j}$: 第一帧里第 $j$ 个对应点的位置
- $q_{t,j}$: 第 $t$ 帧里对应的点位置 (同一个物理粒子)
- $J$: 总共多少对对应点
- $h_0(\cdot)$: 一个 deterministic mapping, 把当前 timestep $\tau$ 的 latent 一步跳到 timestep 0 的预测, 类似 DDIM 的一次大跳
- $De(\cdot)$: VAE decoder, latent → pixel
- $[\cdot]_q$: 取位置 $q$ 的 pixel value

人话: **"你在第 $t$ 帧这个位置生成的颜色, 应该和第一帧那个位置的颜色一样, 因为它们是同一个物理点"**。

总 loss 是所有 frame 累加:
$$\mathcal{L}_{\text{TTCO}} = \sum_{t=2}^{T} \mathcal{L}_{\text{tex}}(t)$$

#### 优化什么?

关键 design choice: **只优化 text embedding 里对应物体名词的那些 token**, 加一个 learnable residual; 以及 DiT 里对应这些 token 的 feature modulation。

为啥不直接用 LoRA? 作者做了 ablation:
- LoRA 会影响整个网络, 背景也跟着变
- 直接优化 spatiotemporal token 会产生 grid-like artifact
- 只调 text token 最 localized, 背景几乎不动

这和最近一些工作 (TokenVerse, DiTCtrl) 的发现一致: **text token 和对应物体 appearance 有强 coupling, 可以做 object-specific control**。

#### 两个 implementation detail 很重要

1. **只优化 noisy step (700-1000)**: texture 的 layout 在早期 step 决定, 后期 step 只做 refinement。你要改 texture, 得在早期改
2. **旋转角度太大时 fallback**: 如果物体转太多, 第一帧的 pixel 都转出视野了, correspondence 找不到, 就用 simulator 渲染的 reconstructed object 的 pixel value 作为 target

---

## 实验数据, 挑重点说

### Table 1: 主实验

| Metric | PSIVG | 最强 baseline | 差距 |
|--------|-------|---------------|------|
| SAM mIoU (mask 一致性) | **0.84** | 0.75 (SG-I2V) | +0.09 |
| Corr. Pixel MSE (pixel 运动误差) | **0.007** | 0.012 (PISA-Base) | -42% |
| Subject Consistency | 0.95 | 0.95 (HunyuanVideo) | 持平 |
| Background Consistency | 0.96 | 0.96 (HunyuanVideo) | 持平 |

关键: **motion 准确性大幅领先, 画面质量不牺牲**。

PISA-Seg 那个 baseline 看起来 temporal flickering 很好 (0.99), 但仔细看是它基本不动, 几乎是 static video, 当然没 flickering。这就是个陷阱 metric。

### Table 2: User Study

32 个人选哪个最"物理上合理":
- PSIVG: **82.3%**
- 其他所有加起来: 17.7%

这个 gap 非常大, 说明人类对 physical plausibility 的感知和 simulator-based metric 高度相关。

### Table 3: TTCO ablation

| | SAM mIoU | Corr. Pixel MSE | Subject Consis. |
|--|----------|-----------------|----------------|
| 不加 TTCO | 0.82 | 0.009 | 0.93 |
| 加 TTCO | **0.84** | **0.007** | **0.95** |

Corr. Pixel MSE 降 22%, 这直接反映 rotation 的 pixel-level accuracy。Subject Consistency 升 2%, texture 更稳了。

---

## 这篇paper真正聪明的几个地方

1. **Training-free**: 不需要重新 train 任何 model, 全在 inference time 做。这意味着可以直接 plug 进任何现有 video generator
2. **Simulator 不需要渲染好看**: 它只提供 motion guidance, 不提供 final RGB。这绕开了 simulator 渲染质量差的死结
3. **Hybrid flow**: 前景用 simulator 的物理 flow, 背景用 template 的 flow, 用 mask 融合。这样既物理正确又保留了 simulator 建不了的复杂背景动态 (水、树叶、火)
4. **Text token 做 localized control**: TTCO 只调 text embedding, 不碰网络权重, 背景不受影响。这个 insight 对 future work 很有价值
5. **Hierarchical LLM prompting for physics**: 让 LLM 输出定性判断再映射到定量参数, 绕开 LLM 数值 calibration 差的问题

---

## 它的局限和我的思考

1. **MPM 处理不了 articulated body**: 人、机器人这种有关节的, MPM 不擅长。要换成 MuJoCo 或 hybrid
2. **Perception 是 bottleneck**: InstantMesh 重建不准, 全盘皆错。整个 pipeline 的 error propagation 没有建模
3. **One-shot, 不是 closed loop**: 现在是 generate template → perceive → simulate → guide, 一锤子买卖。如果做成 iterative refinement, 可能更好
4. **计算慢**: TTCO 要 50 iterations optimization, 加上 perception pipeline, 比 vanilla generation 慢一个数量级

### 一个更大的 question

Karpathy 你之前讲过 "software 2.0" 的概念——神经网络替代传统软件。但这篇 paper 展示了一个 complementary 的方向: **传统软件 (physics engine) 神经网络化 (in-the-loop integration)**。

这其实是 hybrid intelligence 的一种形式。纯 end-to-end 的 world model 现在还做不到准确物理, 纯 simulator 又没有 generation 能力。把它们 in-the-loop 组合起来, 现在就能用, 还能为 future end-to-end world model 生成高质量 training data。

一个 wild 的联想: 这和 **AlphaGo 的 architecture** 有点像。AlphaGo = policy network (learned) + value network (learned) + MCTS (classical search)。PSIVG = video generator (learned) + perception pipeline (learned) + physics simulator (classical)。都是 learned model 提供 "直觉", classical system 提供 "正确性 guarantee"。

如果未来 differentiable physics simulation 成熟了, 这套东西可以 end-to-end train, 那就真的逼近 "learned world model with physics inductive bias" 了。

---

## 参考链接

**核心组件:**
- PSIVG Project: https://vcai.mpi-inf.mpg.de/projects/PSIVG
- CogVideoX: https://arxiv.org/abs/2408.06072
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Go-with-the-Flow: https://arxiv.org/abs/2501.01822
- InstantMesh: https://arxiv.org/abs/2404.07191
- RAFT: https://arxiv.org/abs/2003.12039
- SuperGlue: https://arxiv.org/abs/1911.11773
- Grounding DINO: https://arxiv.org/abs/2303.05499
- SAM 2: https://arxiv.org/abs/2408.00714
- XMem: https://arxiv.org/abs/2207.07115
- DDIM: https://arxiv.org/abs/2010.02502

**物理模拟:**
- Taichi: https://arxiv.org/abs/1904.04931
- 原 MPM paper (Stomakhin 2013, Disney 雪): https://dl.acm.org/doi/10.1145/2461912.2461948
- Mitsuba/Dr.Jit: https://rgl.epfl.ch/publications/Jakob2022Drjit
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis

**相关工作:**
- PISA: https://arxiv.org/abs/2502.01863
- PhysGen3D: https://arxiv.org/abs/2406.04148
- WonderPlay: https://arxiv.org/abs/2506.03614
- MotionClone: https://arxiv.org/abs/2412.19015
- SG-I2V: https://arxiv.org/abs/2411.04965
- Do generative video models understand physics? https://arxiv.org/abs/2501.09038
- How far is video generation from world model: https://arxiv.org/abs/2501.14909
- ViPE (NVIDIA): https://research.nvidia.com/labs/toronto-ai/vipe/

**你可能感兴趣的 broader context:**
- Yann LeCun 的 JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- Sora technical report 里提到的 "world simulator" vision

---

一句话总结这篇 paper: **承认 generator 不懂物理, 然后给它配个物理老师, 在 generation loop 里手把手教它怎么动**。方法上没啥惊天动地的新 theory, 但工程上把一堆 existing pieces (InstantMesh, ViPE, Taichi MPM, RAFT, GwtF, DiT) 组合得很 clever, 解决了一个大家都看到但没人这么解过的问题。这种 "pragmatic hybrid" 的思路, 在纯 end-to-end world model 成熟之前, 有很长的实用价值。

---

# PSIVG: Physical Simulator In-the-Loop Video Generation 深度解析

Andrej, 这篇paper触及一个相当fundamental的问题——diffusion models在video generation上虽然visual fidelity已经非常striking, 但是它们完全缺乏physical understanding。Karpathy你之前在多次talk中提到过这个观点: 这些models本质上是pixel-level denoiser, 它们learned了appearance statistics, 但是没有learned world dynamics。这篇PSIVG的工作就是直接攻击这个gap。

---

## 1. 核心Intuition: 为什么Diffusion Models不懂物理?

论文的核心观察非常sharp: modern video generation models都是trained on denoising或reconstruction objectives, 这些objectives本质上鼓励model去"denoise" individual pixels或patches, 整个过程没有任何mechanism去enforce physical constraints。

考虑一个bowling ball撞pin的场景。一个标准video generator会生成一个视觉上plausible的序列——ball rolling, pins scattering——但是如果你仔细追踪motion vectors, 会发现它们是chaotic的, 违反momentum conservation, 违反gravity, objects可能突然vanish或teleport。

PSIVG的核心idea可以概括为: **既然generator本身无法enforce physics, 那就在它的generation loop里塞进一个真正的physical simulator作为external constraint**。这是一个training-free, inference-time的framework——你不需要重新train任何model, 只需要在inference时把simulator的output作为guidance signal feed back给generator。

这个paradigm叫**simulation-in-the-loop generation**, 它和reinforcement learning里的model-based planning有相似的flavor——你有一个learned policy (这里是diffusion model) 和一个physics engine, 两者协作来产生physically grounded的输出。

---

## 2. PSIVG Pipeline全貌

整个pipeline分为四个major stages, 让我逐层解析:

### Stage 1: Template Video Generation

```
Text Prompt → SD 3 (image) → CogVideoX-I2V-5B / HunyuanVideo-I2V → Template Video
```

这一步生成了一个visually appealing但是physically broken的template。这个template的作用是提供:
- Scene composition (objects, background layout)
- Camera movements
- Object geometry 和 textures 的initial estimate
- Intended dynamics的大致方向

这里有一个关键的design choice: 为什么不直接从prompt生成physical video, 而是先生成template? 因为simulator需要知道场景里有什么objects, 它们的geometry, 初始pose——这些信息必须从某个visual output中perceive出来。Template video充当了这个"物理世界的草图"。

### Stage 2: Perception Pipeline (2D → 4D Lifting)

这是整个framework最technically challenging的部分。我们需要从一个physically inconsistent的video中extract出simulator-ready的assets, 包括三件事: foreground object geometry, background scene geometry, 和object dynamics。

#### 2.1 Foreground Object Geometry Reconstruction

```
Video → Grounding DINO + SAM 2 + XMem → Per-frame masks
First frame → Object-centric crop → InstantMesh → 3D Mesh per object
```

这里用到了三个off-the-shelf models的chain:
- **Grounding DINO** [30]: open-set object detection, 从text prompt找到objects
- **SAM 2** [38]: video segmentation, propagate masks across frames
- **XMem** [9]: long-term video object segmentation with Atkinson-Shiffrin memory model

然后用**InstantMesh** [50]做single-image 3D mesh reconstruction。这里有一个empirical insight非常值得注意: 作者发现用single image (first frame)重建的mesh比用multi-view方法直接从different frames重建更reliable。原因是video generator本身在frames间存在geometry和texture inconsistency, 这会break multi-view reconstruction的假设。Single-image reconstruction leveraging pretrained object priors反而更robust。

这给我们一个deep的intuition: **video generators的frame-to-frame consistency问题, 反而让传统的multi-view 3D reconstruction方法失效, 因为这些方法的underlying assumption (multi-view consistency)恰好是video generator所违反的**。

#### 2.2 Background Scene Geometry & Camera Motion

```
Video (foreground masked) → ViPE → 4D Reconstruction
                          → 3D background point cloud (world frame)
                          → Camera poses per frame
                          → Rough object positions
```

使用**ViPE** (Video Pose Engine) [20], 这是一个NVIDIA的4D geometric perception system。它的pipeline包含:
- 对key frames做bundle adjustment
- Transform per-frame metric depth pointmaps到scene-level world frame
- Aggregate static background points from all frames

然后做aggressive sub-sampling和filtering来remove floating artifacts——这些artifacts是template video inconsistency的直接产物。

#### 2.3 Foreground Object Dynamics Estimation

这是最math-heavy的部分。我们需要估计每个object的initial state: position, linear velocity, rotational velocity。

**Linear Velocity:**
```
v_linear = (p_t2 - p_t1) / Δt
```
其中 `p_t1, p_t2` 是两个key frames中object的3D position, `Δt` 是real-world时间间隔。

**Rotational Velocity:**
这个更tricky。方法:
1. 在两个frames之间做2D feature matching用**SuperGlue** [39]
2. 计算相对于matched feature points centroid的2D flow field
3. 从这个flow field isolate出rotational motion component
4. Combine linear和rotational components得到per-point initial instantaneous velocity

这个per-point velocity representation很关键, 因为MPM simulator需要每个particle的initial velocity, 而rigid body simulators只需要center of mass velocity + angular velocity。MPM的particle-based nature让这个formulation natural。

---

## 3. Physical Simulation Setup

### 3.1 Simulation Domain Design

这里有一个很elegant的scaling trick:

```
1. Bound foreground dynamics range → green box
2. Bound background geometry → red box  
3. Apply spatial offset coefficient C → blue cube (simulation domain)
4. Normalize domain to [0, 2] in x, y, z
5. Compute metric-to-simulation scale S
6. Scale physics constants (gravity, Young's modulus) by S
```

为什么要normalize到[0, 2]? 因为MPM在fixed domain上更稳定, 而且这样可以让simulation resolution和metric scale解耦。Scale S用于把real-world physics constants转换到simulation space。

### 3.2 Physical Property Estimation via LLM

这是一个非常creative的design choice。用**GPT-5**作为vision-language model来infer object的physical properties:

```
First frame + Prompt → GPT-5 → Material descriptors (composition, elasticity, roughness)
                            → Physical parameters (density, Young's modulus)
```

作者发现直接让LLM输出numerical values不稳定, 所以设计了一个**hierarchical prompting framework**:
1. First query: intermediate material descriptors (qualitative)
   - Object composition (wood, metal, rubber...)
   - Elasticity / bounce characteristics
   - Surface roughness
2. Then map these qualitative properties到quantitative simulation parameters

这个design pattern很generalizable——当你需要LLM输出连续数值但它的calibration不可靠时, 先让它做qualitative reasoning, 再用一个deterministic mapping function把qualitative labels转成numbers。

### 3.3 MPM Simulation & Rendering

使用**Taichi** [19]实现的**Material Point Method** [41]。MPM是particle-based method, 每个object表示为一组particles, 这些particles carry mass, velocity, 和deformation信息。MPM的优势:
- 自然支持deformable bodies (不像rigid body simulators)
- 可以handle fractures, large deformations
- Lagrangian-Eulerian hybrid: particles carry material properties, grid用于collision detection和force computation

然后使用**Mitsuba** [22]渲染:
- RGB frames
- Segmentation masks  
- Frame-to-frame pixel-to-pixel correspondences (这是TTCO的关键input)

**Key insight**: Simulator的rendered RGB visual quality很差——artificial style, 没有realistic lighting/shadows, low resolution, mesh imperfections会放大。但是这些renders encapsulate faithful motion physics。所以simulator的role是guidance signal, 而非final output。

---

## 4. Physically-Consistent Video Generation

### 4.1 Optical Flow Conditioning via Go-with-the-Flow

使用**Go-with-the-Flow (GwtF)** [6]作为video generation backbone。GwtF的核心idea是: 用optical flow来warp noise latents, 这样diffusion process从correlated noise开始, 自然产生遵循flow的motion。

Hybrid flow field construction:

```
Flow_foreground = RAFT(simulator_rendered_RGB)      # physics-grounded
Flow_background = RAFT(template_video)              # preserve camera/scene dynamics  
Flow_hybrid = mask_blend(Flow_foreground, Flow_background, seg_masks)
```

用**RAFT** [43]计算optical flow, 即使simulator本身提供pixel correspondences, 也用RAFT重新计算, 因为GwtF是在RAFT flow上trained的, 分布更匹配。

这个hybrid design很重要: foreground用simulator flow来enforce physics, background用template flow来preserve那些simulator无法model的complex dynamics (water, foliage, fire等)和camera movement。

### 4.2 Test-Time Texture Consistency Optimization (TTCO)

这是论文的第二个核心contribution, 解决一个specific failure mode: 即使有accurate motion guidance, flow-conditioned models仍然会有texture flickering和appearance drift, 特别是object rotation时。

#### Intuition

考虑一个spinning cube with不同faces有不同textures。即使motion trajectory完全正确, diffusion model可能在某些frames "hallucinate" wrong texture on visible face, 因为它没有explicit mechanism来enforce texture consistency across rotations。

TTCO的想法: 既然simulator知道frame 1的pixel A对应frame t的pixel B (因为same physical particle), 那么generated video在这两个pixels上应该有相同texture。我们可以用这个correspondence来定义一个loss, 然后在test-time优化某些learnable parameters。

#### Mathematical Formulation

核心loss function:

$$\mathcal{L}_{\text{tex}}(t) = \sum_{j=1}^{J} \left\| \left[ De\left( h_0(\hat{L}_\tau) \right) \right]_{q_{t,j}} - \left[ W_t(\hat{I}_1) \right]_{q_{t,j}} \right\|_2^2$$

变量逐个解析:

- $\hat{L}_\tau$: diffusion model在denoising timestep $\tau$ 预测的**latent**。这里 $\tau$ 是diffusion timestep, 不是frame index。下标 $\tau$ 表示这个latent是在去噪过程的第 $\tau$ 步。
  
- $\hat{I}_1$: template video的**第一帧** image (作为texture reference)

- $W_t(\hat{I}_1)$: 使用simulator pixel correspondences $\{(p_{1,j}, q_{t,j})\}_{j \in J}$ 将第一帧warp到第 $t$-th frame的结果
  - $p_{1,j}$: frame 1中第 $j$ 个corresponding pixel location
  - $q_{t,j}$: frame $t$ 中对应的pixel location (same physical point)
  - $J$: total number of correspondence pairs

- $h_0(\cdot)$: **deterministic DDIM-style step mapping**到final denoising iteration。这是一个one-step prediction, 把当前timestep $\tau$ 的latent直接映射到final (timestep 0) 的latent prediction。参考DDIM [40]的formulation, 这相当于做一次large skip的denoising step来获得clean image的estimate。

- $De(\cdot)$: VAE的**decoder**, 把latent space映射回pixel space

- $[\cdot]_q$: indexing operator, retrieve pixel value at location $q$

整个loss的含义: 对于第 $t$ frame, 找到所有与frame 1对应的pixels $q_{t,j}$, 检查generated video在这些位置的pixel values是否等于frame 1对应位置 $p_{1,j}$ 的pixel values (经过warping后)。

**Total TTCO loss**:

$$\mathcal{L}_{\text{TTCO}} = \sum_{t=2}^{T} \mathcal{L}_{\text{tex}}(t)$$

对所有frames (从2到 $T$) 累加。

#### What Gets Optimized?

这是一个很subtle的design choice。只优化foreground-related parameters:

1. **Learnable residual token** added to text embeddings for object phrases
2. **Feature-wise modulations** in DiT (Diffusion Transformer) layers corresponding to object tokens

为什么这样设计而不是用LoRA或直接优化spatiotemporal tokens?

作者做了ablation (Fig. 6):
- LoRA-based: degrades video quality, particularly in background
- Direct spatiotemporal token optimization: produces grid-like artifacts
- Text token modulation: lightweight, localized, preserves global consistency

这个observation aligns with recent work [7, 14] showing text tokens在diffusion models中strongly control corresponding object appearance。这是一个非常deep的insight——text embeddings在DiT里不只control semantic content, 它们和spatial features有strong coupling, 可以做localized appearance control。

#### Implementation Details

- **Optimizer**: AdamW, learning rate $2 \times 10^{-4}$
- **Iterations**: 50
- **Diffusion steps**: sample steps 700-1000 (noisier steps)

为什么要focus on noisier steps? 这是一个重要的empirical finding: texture的generation主要在早期(high noise)steps决定, 后期(low noise)steps主要做refinement。如果在后期steps做optimization, 只能改变high-frequency details, 无法改变整体texture layout。这个insight和classifier-free guidance的analysis有类似的flavor——structural decisions happen early in diffusion process。

#### Edge Cases

当object rotation角度大时, frame 1的pixels可能全部rotate出view, 这时pixel-to-pixel warping loss无法直接apply。Solution: 使用simulator中rendered的reconstructed object的pixel values作为fallback target。这是为什么保留simulator-rendered RGB很关键。

---

## 5. 实验数据分析

### 5.1 Quantitative Results (Table 1)

关键metrics解读:

| Metric | PSIVG | Best Baseline | Interpretation |
|--------|-------|---------------|----------------|
| SAM mIoU | **0.84** | 0.75 (SG-I2V) | Object mask与simulator trajectory的一致性, 越高越好 |
| Corr. Pixel MSE | **0.007** | 0.012 (PISA-Base) | Frame-to-frame pixel correspondence error, 越低越好 |
| CLIP Text | 0.35 | 0.35 (multiple) | Text-prompt alignment, 不牺牲semantic fidelity |
| Subject Consistency | 0.95 | 0.95 (HunyuanVideo) | Object appearance consistency |
| Background Consistency | 0.96 | 0.96 (HunyuanVideo) | Background stability |
| Motion Smoothness | 0.99 | 0.99 (HunyuanVideo) | Temporal smoothness |
| Temporal Flickering | 0.97 | 0.99 (PISA) | 略低但acceptable |

**关键observation**: PSIVG在motion controllability metrics上大幅领先(SAM mIoU +0.09, Corr. Pixel MSE -0.005), 同时在general quality metrics上保持competitive。这说明physics guidance没有sacrifice visual quality。

值得注意的baseline analysis:
- **PISA-Seg/Depth** [25]: temporal flickering很好(0.99), 但是motion很小——基本上是static videos, 所以flickering低是因为没有motion
- **MotionClone** [28] 和 **SG-I2V** [35]: training-free motion control, 但是Corr. Pixel MSE较高, 说明它们难以precisely follow trajectories, 尤其是rotations
- **DragAnything** [47]: SAM mIoU只有0.43, 说明entity representation-based control在物理精确性上有限

### 5.2 User Study (Table 2)

| Method | Preference Rate |
|--------|----------------|
| **PSIVG** | **82.3%** |
| CogVideoX | 7.2% |
| HunyuanVideo | 4.5% |
| PISA-Seg | 2.6% |
| SG-I2V | 2.5% |
| MotionClone | 0.9% |

32个participants, 82.3%的preference rate是一个相当dominant的结果。Human evaluators对physical plausibility的感知和simulator-based metrics高度correlated。

### 5.3 Ablation: TTCO Impact (Table 3)

| Setting | SAM mIoU ↑ | Corr. Pixel MSE ↓ | Subj. Consis. ↑ |
|---------|------------|-------------------|-----------------|
| w/o TTCO | 0.82 | 0.009 | 0.93 |
| w/ TTCO | **0.84** | **0.007** | **0.95** |

TTCO的贡献:
- Corr. Pixel MSE从0.009降到0.007 (-22%)——这是pixel-level rotation accuracy的直接度量
- Subject Consistency从0.93升到0.95——texture consistency improvement
- SAM mIoU也有小幅提升(0.82→0.84), 说明更好的texture consistency也有助于trajectory adherence

---

## 6. Limitations & 未来方向

作者honestly列出了几个limitations:

1. **MPM的限制**: MPM适合continuum mechanics (soft bodies, fluids), 但是难以handle articulated structures如humans或vehicles。Rigid body simulators (PyBullet, MuJoCo)或hybrid approaches可能更适合这些cases。这是一个自然的extension方向——根据object type选择合适的simulator backend。

2. **Perception quality bottleneck**: 整个pipeline的上限被perception pipeline的质量限制。如果InstantMesh重建的mesh不准, simulator的dynamics也会有误差。未来的方向可能是iterative refinement或更好的4D reconstruction methods。

3. **GwtF继承的限制**: 难以generate very small或thin objects。这是underlying video generator的limitation, 不是PSIVG特有的。

4. **Computation cost**: Test-time optimization需要50 iterations, 加上perception pipeline的overhead, 整个inference比vanilla generation慢很多。对于real-time applications可能是个问题。

---

## 7. 更深的Intuition与Reflections

### 7.1 这为什么重要?

这篇工作触及一个deep question: **generative models和symbolic/physics-based systems如何hybridize?** 

Karpathy你多次提到过"software 2.0"的概念——neural networks replacing traditional software。但这篇paper展示了一个complementary的方向: **traditional software (physics simulators) augmenting neural networks**。Physics simulator是"software 1.0"的极致——deterministic, interpretable, provably correct within its assumptions。把它in-the-loop with diffusion model, 我们得到一个hybrid system, 它combines:
- Generative model的visual realism和diversity
- Physics simulator的dynamical correctness

这个paradigm可能extend到很多domains: 
- **Robotics**: Diffusion policy + MuJoCo for verification
- **Autonomous driving**: Generative scenarios + physics engine for safety validation
- **Game AI**: Procedural content generation + game engine physics

### 7.2 和World Models的关系

这是目前一个非常hot的话题。Video generation models是否能成为world models? Kang et al. [23] 和Motamed et al. [34]的工作表明, 当前video models并不真正understand physical principles。

PSIVG采取了一个pragmatic stance: 不要求generator本身learn physics, 而是externalize physics到一个dedicated module。这是一个reasonable engineering choice, 但从AGI perspective来看, 我们最终需要models内生地understand physics。PSIVG这样的hybrid systems可以作为:
1. **当前的practical solution** (works now)
2. **Future training data generator** (用physical-grounded videos来train下一代end-to-end world models)
3. **Evaluation benchmark** (simulator-grounded metrics)

### 7.3 Open Questions

1. **Perception bottleneck**: 如果perception pipeline出错, 整个system fails。如何quantify和propagate uncertainty through the pipeline?
2. **Multi-object interactions**: 当前paper主要demo了single或few objects。Complex multi-body scenarios的scalability如何?
3. **Non-rigid dynamics**: MPM支持deformable bodies, 但是cloth, fluid, granular materials的handling还需要更多validation
4. **Closed-loop refinement**: 当前是one-shot pipeline (generate template → perceive → simulate → guide)。Iterative refinement是否能进一步提升quality?
5. **Learnable integration**: TTCO是一个test-time optimization。如果把simulator integration变成end-to-end trainable, 会怎样? 但这需要differentiable physics simulation, 增加complexity

---

## 8. 参考资源

**主paper与project:**
- PSIVG Project Page: https://vcai.mpi-inf.mpg.de/projects/PSIVG
- arXiv (待更新, 目前从MPI page获取)

**Core components:**
- CogVideoX: https://arxiv.org/abs/2408.06072
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Go-with-the-Flow: https://arxiv.org/abs/2501.01822
- InstantMesh: https://arxiv.org/abs/2404.07191
- RAFT: https://arxiv.org/abs/2003.12039
- SuperGlue: https://arxiv.org/abs/1911.11773
- Grounding DINO: https://arxiv.org/abs/2303.05499
- SAM 2: https://arxiv.org/abs/2408.00714
- XMem: https://arxiv.org/abs/2207.07115
- DDIM: https://arxiv.org/abs/2010.02502

**Physics simulation:**
- Taichi: https://arxiv.org/abs/1904.04931
- Original MPM paper (Stomakhin et al. 2013): https://dl.acm.org/doi/10.1145/2461912.2461948
- Mitsuba/Dr.Jit: https://rgl.epfl.ch/publications/Jakob2022Drjit
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- PhysGaussian (related): https://arxiv.org/abs/2311.12131

**Related physics-aware generation:**
- PISA: https://arxiv.org/abs/2502.01863
- PhysGen3D: https://arxiv.org/abs/2406.04148
- WonderPlay: https://arxiv.org/abs/2506.03614
- MotionClone: https://arxiv.org/abs/2412.19015
- SG-I2V: https://arxiv.org/abs/2411.04965
- Force Prompting: https://arxiv.org/abs/2502.00231
- Do generative video models understand physical principles? https://arxiv.org/abs/2501.09038
- How far is video generation from world model: https://arxiv.org/abs/2501.14909

**ViPE (NVIDIA):**
- https://research.nvidia.com/labs/toronto-ai/vipe/

**Karpathy你可能感兴趣的相关discussion:**
- 你的"Intro to LLMs"和micrograd相关talks里提到的"software 2.0"和differentiable programming的discussion
- Yann LeCun的JEPA framework, 同样关注world models的predictive learning
- Sora的技术报告里提到的"world simulator"野心——这个paper某种程度上是对那个vision的一个pragmatic implementation

---

这篇paper给我最大的启发是: **解决hard AI problems不一定要end-to-end differentiable everything**。Hybrid systems combining neural networks with classical structured modules (physics engines, symbolic reasoners, planners)可能比pure end-to-end approaches更快deliver practical solutions, 同时为future end-to-end learning提供high-quality training data。这是一个很Karpathy-style的pragmatic engineering wisdom。

如果你想更深入, 我建议关注几个follow-up方向: (1) differentiable physics integration, (2) uncertainty-aware perception pipelines, (3) scaling到complex multi-agent scenarios, (4) closed-loop iterative refinement。这些方向都有potential成为独立的impactful work。
