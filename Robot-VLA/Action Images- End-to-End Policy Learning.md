---
source_pdf: Action Images- End-to-End Policy Learning.pdf
paper_sha256: 3125d87b98b2ff821335ae0ac4e6fcff4acda980eacca208f0adc25c2f4fd3db
processed_at: '2026-07-18T00:48:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Action Images: 把 7-DoF 控制塞进像素空间做 Video Generation

这篇 paper 是 MIT/UMass/Genesis AI 等合作的 work,核心 thesis 极其清晰:**robot action 的表示方式决定了 policy 的 generalization 上限**。在 video world model 火爆的当下,大家发现"会预测未来帧 ≠ 会做动作",根本症结在于 action 没有跟 observation 共享同一个表示空间——传统做法把 action 编成一个 7-dim token / latent code,再外挂一个 policy head 去解码,这个外挂的网络就是 generalization 的 bottleneck。作者把 7-DoF action 渲染成多视角的 RGB heatmap video(就是 paper title 里的 "action images"),让 action 跟 observation 同住一个 video latent space,video backbone 直接当 zero-shot policy 用。

项目主页: https://ActionImages.github.io

---

## 1. Big Picture:为什么需要 pixel-grounded action

### 1.1 现有 world-action model 的两条路线和它们的痛点

Robotics world model 这一波基本沿着 video diffusion 的成功路线走,但把"预测未来"转成"决定动作"始终有 gap:

- **路线 A:外挂 policy head / action module**([TesserAct](https://arxiv.org/abs/2504.20995), [Cosmos-Policy](https://arxiv.org/abs/2601.16163), [DreamZero](https://arxiv.org/abs/2602.15922), [UniVA](https://arxiv.org/abs/2503.00200))。Video backbone 学到的 world prior 经过一层抽象 feature,再丢给一个新的 control decoder。pretrain 的 video knowledge 在这个 interface 处发生"信息瓶颈",distribution shift 的时候最先崩掉的就是这个 head。
- **路线 B:latent action code**([Cosmos-Policy](https://arxiv.org/abs/2601.16163) 用低维 action token 喂进 video backbone)。Action 没有空间结构,video model 把它当作一个 foreign modality 处理,无法 reuse 任何 spatial-temporal prior。

paper 的诊断特别 Karpathy-style:"the burden of generalization is shifted to a specialized control module, which is often exactly where transfer breaks down."(generalization 的负担被甩给一个专门的 control module,而转移失败恰好发生在这里。)→ 既然如此,那就把 action 编码到 video model 已经精通的表示空间里——像素。

### 1.2 关键 insight:把 action 当成 tracking 问题

如果把"输出 7-DoF action"重新表达成"在多视角图像里预测 end-effector 的几个 semantic 关键点的 2D 位置",那这个问题就跟 [CoTracker](https://arxiv.org/abs/2507.01578)/[TAPIR](https://arxiv.org/abs/2306.08379) 这类 point tracking 长得一模一样,video model 对这种 spatio-temporal localization 是 pretrain 过海量数据的。Action images 本质上是把 tracking 当 action representation。一旦做到这点,zero-shot policy 就是 video generation 的副产品——你 generate 出 action video,decode 回 7-DoF 即可。

---

## 2. Action Images Encoding:7-DoF → 3 个 semantic 3D point → multi-view heatmap

### 2.1 把 7-DoF 拆成 3 个语义点(公式 1)

原始 action:
$$\mathbf{a}_t = [\mathbf{p}_t, \boldsymbol{\theta}_t, g_t] \in \mathbb{R}^7$$

- $\mathbf{p}_t \in \mathbb{R}^3$:end-effector position(末端位置)
- $\boldsymbol{\theta}_t \in \mathbb{R}^3$:end-effector orientation(通常是 axis-angle 或 RPY)
- $g_t \in \mathbb{R}$:gripper openness(实数 0~1,抓爪开度)
- 下标 $t$:时间步

转换成 3 个 semantic 3D 点:

$$\mathbf{q}_t^{\mathrm{pos}} = \mathbf{p}_t$$
$$\mathbf{q}_t^{\mathrm{up}} = \mathbf{p}_t + \ell\, \mathbf{R}(\boldsymbol{\theta}_t)\, \mathbf{e}_x$$
$$\mathbf{q}_t^{\mathrm{normal}} = \mathbf{p}_t + \ell\, \mathbf{R}(\boldsymbol{\theta}_t)\, (-\mathbf{e}_z)$$

变量解释:
- $\mathbf{q}_t^{\mathrm{pos}}, \mathbf{q}_t^{\mathrm{up}}, \mathbf{q}_t^{\mathrm{normal}} \in \mathbb{R}^3$:三个语义 3D 点;上标 pos/up/normal 表示语义
- $\mathbf{R}(\boldsymbol{\theta}_t) \in SO(3)$:由 orientation 构造的 rotation matrix
- $\mathbf{e}_x, \mathbf{e}_z \in \mathbb{R}^3$:canonical 单位基向量(分别对应 gripper 局部坐标系的 x 轴和 z 轴)
- $\ell$:延伸长度(implementation 里设 0.1 米)
- $\mathbf{e}_x$ 给 "up" 方向(in-plane,沿抓爪开口平面里的一个 canonical 方向)
- $-\mathbf{e}_z$ 给 "normal" 方向(垂直于抓爪平面的法线方向,加负号是为了让法线指向"指向物体"的方向)

**intuition**:3 个点刚好完整确定了 6-DoF 的 rigid pose——position 一个点 fix 住平移,up + normal 两个方向 fix 住旋转(第三个方向叉乘出来)。这跟 graphics 里 camera look-at 的 up/forward 表示完全同构。

### 2.2 Multi-view 投影(公式 2)

对每个 camera view $v$,用 projection function $\pi_t^{(v)}(\cdot)$(即 intrinsics × extrinsics)把 3D 点投到 2D:

$$\mathbf{u}_t^{\mathrm{pos},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\mathrm{pos}})$$
$$\mathbf{u}_t^{\mathrm{normal},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\mathrm{normal}})$$
$$\mathbf{u}_t^{\mathrm{up},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\mathrm{up}})$$

- $\mathbf{u}_t^{(\cdot),(v)} \in \mathbb{R}^2$:投影后的 2D 像素坐标
- 上标 $(v)$:第 $v$ 个视角

### 2.3 Heatmap 渲染(公式 3-5)—— RGB 三通道玩得很巧

把每个 2D 点 rasterize 成 Gaussian heatmap,塞进一张 RGB 图 $\mathbf{A}_t^{(v)} \in \mathbb{R}^{H \times W \times 3}$:

**Red channel**(位置点):
$$\mathbf{A}_t^{(v)}(:,:,1) = \mathcal{G}(\cdot; \mathbf{u}_t^{\mathrm{pos},(v)}, \sigma)$$

**Green channel**(法线点):
$$\mathbf{A}_t^{(v)}(:,:,2) = \mathcal{G}(\cdot; \mathbf{u}_t^{\mathrm{normal},(v)}, \sigma)$$

**Blue channel**(up 点 + gripper openness,共用一个通道):
$$\tilde{\mathbf{A}}_t^{(v)}(:,:,3) = \mathcal{G}(\cdot; \mathbf{u}_t^{\mathrm{up},(v)}, \sigma)$$
$$\mathbf{A}_t^{(v)}(i,j,3) = \begin{cases} \tilde{\mathbf{A}}_t^{(v)}(i,j,3), & \tilde{\mathbf{A}}_t^{(v)}(i,j,3) > 0.25 \\ 0.25 \cdot g_t, & \text{otherwise} \end{cases}$$

变量:
- $\mathcal{G}(\cdot; \mathbf{u}, \sigma)$:中心在像素 $\mathbf{u}$、标准差 $\sigma$ 的 2D Gaussian(implementation 里 $\sigma = 0.05$ 相对 image resolution)
- $(i,j)$:像素坐标
- $g_t$:gripper openness ∈ [0, 1]
- 阈值 0.25:区分"heatmap 高响应区域"和"背景"

**这个设计非常聪明**:blue channel 复用了——高响应(>0.25)区域存 up 点的 Gaussian,低响应(≤0.25)的背景区域统一存 $0.25 \cdot g_t$。这样一张图同时编码了"空间信息"(up 点位置)和"标量信号"(gripper 开度),decode 的时候用阈值切分即可恢复。

### 2.4 Stack 成 action video(公式 6)

$$\mathcal{A}^{(v)} = \{\mathbf{A}_1^{(v)}, \dots, \mathbf{A}_T^{(v)}\} \in \mathbb{R}^{T \times H \times W \times 3}$$

- $\mathcal{A}^{(v)}$:view $v$ 上的 action video
- $T$:时间帧数(implementation 里 41 frames per modality per view)

这就跟 RGB observation $\mathcal{O}^{(v)} \in \mathbb{R}^{T \times H \times W \times 3}$ 完全同构,**unified video-space representation of observation and action**。

### 2.5 为什么 multi-view 是必须的

paper 在 Sec 3.1 强调:"a single view often provides only an ambiguous projection of motion"。从 2D 单视角去 lift 7-DoF action 在数学上是 underdetermined 的——单视角的 2D 点反推 3D 点存在 depth ambiguity。多视角让 representation reconstructable,同时 partial occlusion 时也有 redundancy。

---

## 3. Action Images Decoding:从生成 heatmap 反推 7-DoF

### 3.1 Gripper openness decode(公式 7)

$$\hat{g}_t = \frac{1}{0.25} \cdot \frac{1}{|\mathcal{Q}_t|} \sum_{(i,j,v) \in \mathcal{Q}_t} \mathbf{A}_t^{(v)}(i,j,3)$$
$$\mathcal{Q}_t = \{(i,j,v) \mid \mathbf{A}_t^{(v)}(i,j,3) < 0.25\}$$

变量:
- $\hat{g}_t$:decoded gripper openness
- $\mathcal{Q}_t$:低响应像素的集合(跨所有 view)
- $|\mathcal{Q}_t|$:$\mathcal{Q}_t$ 的元素个数

**intuition**:背景像素理论上都是 $0.25 \cdot g_t$,所以对它们求均值再除 0.25 就得到 $g_t$。multi-view 平均能去噪。

### 3.2 Heatmap centroid 提取(公式 8)

对 main view 的 heatmap $\mathbf{H}_t^{(1)} \in [0,1]^{H \times W}$,用加权平均找 2D 中心:

$$\hat{\mathbf{u}}_t^{(1)} = \frac{\sum_{i,j} \mathbf{H}_t^{(1)}(i,j) [i+0.5, j+0.5]^\top}{\sum_{i,j} \mathbf{H}_t^{(1)}(i,j)}$$

- $\hat{\mathbf{u}}_t^{(1)} \in \mathbb{R}^2$:主视角 heatmap 的 2D 中心点
- $[i+0.5, j+0.5]$:加 0.5 是为了避免像素边界 off-by-half 误差
- 上标 (1) 表示 main view

加权平均比 argmax 更平滑、可微、对噪声鲁棒。

### 3.3 Multi-view ray casting + side-view matching(公式 9)

从 main view 的 2D 点 $\hat{\mathbf{u}}_t^{(1)}$ 出发,沿 ray 在 near plane 和 far plane 之间采 $K$ 个候选 3D 点 $\{\mathbf{x}_{t,k}\}_{k=1}^K$,投影到 side view,选跟 side-view heatmap 匹配最好的:

$$\hat{\mathbf{x}}_t = \arg\max_{\mathbf{x}_{t,k}} \mathbf{H}_t^{(2)}\big(\pi_t^{(2)}(\mathbf{x}_{t,k})\big)$$

变量:
- $\mathbf{x}_{t,k} \in \mathbb{R}^3$:沿 ray 采样的第 $k$ 个 3D 候选
- $\pi_t^{(2)}(\cdot)$:side view 的投影函数
- $\mathbf{H}_t^{(2)}$:side view 的 heatmap
- $\hat{\mathbf{x}}_t$:最终选定的 3D 点
- 上标 (2):side view

**intuition**:main view 给 2D anchor,side view resolve depth ambiguity。这就是经典的 stereo matching / multi-view triangulation 思路,只是把"找对应"换成"在 heatmap 上找最大响应"。

对三个 semantic point 分别重复这个过程,得到 $\hat{\mathbf{q}}_t^{\mathrm{pos}}, \hat{\mathbf{q}}_t^{\mathrm{up}}, \hat{\mathbf{q}}_t^{\mathrm{normal}}$。

### 3.4 从 3D 点反推 7-DoF

$$\hat{\mathbf{p}}_t = \hat{\mathbf{q}}_t^{\mathrm{pos}}$$
$$\hat{\mathbf{e}}_t^x = \mathrm{norm}(\hat{\mathbf{q}}_t^{\mathrm{up}} - \hat{\mathbf{q}}_t^{\mathrm{pos}})$$
$$\hat{\mathbf{e}}_t^z = \mathrm{norm}(\hat{\mathbf{q}}_t^{\mathrm{pos}} - \hat{\mathbf{q}}_t^{\mathrm{normal}})$$
$$\hat{\mathbf{e}}_t^y = \hat{\mathbf{e}}_t^z \times \hat{\mathbf{e}}_t^x$$

- $\hat{\mathbf{e}}_t^x, \hat{\mathbf{e}}_t^y, \hat{\mathbf{e}}_t^z \in \mathbb{R}^3$:recovered 旋转矩阵的三列(局部坐标轴在世界坐标系下的表示)
- 叉乘重构 $y$ 轴保证正交
- $\hat{\boldsymbol{\theta}}_t$:由这三个 basis vectors 唯一确定 rotation matrix,再 parameterize 回 $\mathbb{R}^3$

最终 decoded action:$\hat{\mathbf{a}}_t = [\hat{\mathbf{p}}_t, \hat{\boldsymbol{\theta}}_t, \hat{g}_t]$

### 3.5 Decode 误差来源分析

paper 在 Discussion 里说,信息损失主要来自两个 discretization:
- **ray 采样间隔** → 控制 depth 精度
- **heatmap 空间分辨率** → 控制 image-plane localization 精度

这两个都是 predictable 的——加分辨率 / 加采样密度就能线性降误差。这给了 representation 一个非常好的可调精度特性,不像 latent action code 那种黑盒 bottleneck。

---

## 4. Unified World Action Model 训练

### 4.1 Latent packing(公式 10)

对每个 view $v$,RGB observation clip $\mathbf{V}_{1:T}^{(v)} \in [0,1]^{T \times H \times W \times 3}$ 和 action clip $\mathbf{A}_{1:T}^{(v)} \in [0,1]^{T \times H \times W \times 3}$ 经过 3D-VAE 编码到 latent space,沿时间维度拼接:

$$\mathbf{X}_v = [\mathbf{V}_{1:T}^{(v)}, \mathbf{A}_{1:T}^{(v)}] \in \mathbb{R}^{(2T) \times h \times w \times c}$$

- $\mathbf{X}_v$:第 $v$ 个 view 的 unified latent sequence
- $2T$:双倍时间长度(video 在前, action video 在后)
- $h, w, c$:latent space 的空间分辨率和通道数

implementation 里 $T = 41$,full two-view 设置就是 $2 \times 2 \times 41 = 164$ 帧。

### 4.2 Multiple Mask Strategies—— 一个 backbone 切换 4 种 task

这个设计非常 elegant。同一个 flow matching 框架下,通过不同 mask 模式实现 4 种行为:

| Mask 模式 | Visible | Masked | 训练目标 |
|---|---|---|---|
| 1) Action & video joint gen | 第一帧 V | 其余 V + 全部 A | 给定初始观测,联合生成未来 video 和 action |
| 2) Action-conditioned video gen | 全部 A | 全部 V (除第一帧) | 给定 action,生成对应 video |
| 3) Video-to-action labeling | 全部 V | 全部 A | 从 video 推 action(video → action)|
| 4) Video-only generation | 第一帧 V | 其余 V | 标准 video generation(给没 action label 的数据用)|

任务混合比例:85% joint generation,5% each for 其他三个。

**intuition**:这种 masking 思路跟 [MAE](https://arxiv.org/abs/2111.06377)、[JEPA](https://arxiv.org/abs/2301.08243)、[Drop-Diffusion](https://arxiv.org/abs/2312.02548) 是一脉相承的——一个 backbone 一个 objective,通过 conditioning pattern 切任务。对 generalization 友好,因为不同 task 互相 regularize。

### 4.3 Camera conditioning(Plücker embedding)

借鉴 [ReCamMaster](https://arxiv.org/abs/2511.16727),用 Plücker embedding $\mathbf{cam}_t$ 表示 camera,通过一个 lightweight conv encoder $E_c$ 注入:

$$\mathbf{F}_i = \mathbf{F}_o + E_c(\mathbf{cam}_t)$$

- $\mathbf{F}_o$:spatial-attention layer 的输出 feature
- $\mathbf{F}_i$:下一个 3D-attention layer 的输入 feature
- $E_c$:camera encoder

Plücker embedding 用 6 维 $(origin, direction)$ 表示每条 ray,是 multi-view / camera-control video generation 的事实标准。

### 4.4 Flow matching 训练目标(公式 11)

用 [flow matching](https://arxiv.org/abs/2210.02747)(Lipman et al.)替代传统 $\epsilon$-prediction:

$$\mathbf{v} = \boldsymbol{\epsilon} - \mathbf{X}$$
$$\mathcal{L} = \mathbb{E}\left[\big\| M \odot (\mathbf{v} - \mathbf{v}_\theta(\mathbf{X}, \mathcal{T}, \mathbf{cam}))\big\|_2^2\right]$$

变量:
- $\mathbf{v}$:target velocity field
- $\boldsymbol{\epsilon}$:从 prior distribution(高斯)采样的 noise
- $\mathbf{X}$:clean latent(公式 10 的 $\mathbf{X}_v$,跨 view 拼起来)
- $M$:mask tensor(mask strategy 决定的)
- $\mathbf{v}_\theta$:模型预测的 velocity
- $\mathcal{T}$:text instruction 的 embedding
- $\mathbf{cam}$:camera Plücker embedding
- $\odot$:Hadamard 积(element-wise)
- $L_2$ 范数 squared

只对 masked tokens 计算 loss(visible tokens 不参与),这跟 [Masked Diffusion](https://arxiv.org/abs/2503.06682) 的做法一致。

### 4.5 Backbone & 训练超参

- Backbone: [Wan 2.1-I2V-14B-480P](https://arxiv.org/abs/2503.20314)(开源 video diffusion 大模型)
- 训练硬件:32 × A100 GPU
- 并行策略:DeepSpeed ZeRO + bfloat16 mixed precision + gradient checkpointing
- Per-device batch size:1
- Learning rate:$5 \times 10^{-7}$,constant-with-warmup,1000 step warmup
- Gradient clipping:max norm 1.0
- 总步数:100,000 steps
- Gaussian 渲染参数:$\ell = 0.1$, $\sigma = 0.05$(paper 说这些 hyperparameter 不敏感)

### 4.6 数据集混合(Table 1)

| Dataset | Trajectories | Views | Real Action | Camera Calib | Camera Motion |
|---|---|---|---|---|---|
| DROID | 80k | 2 | ✓ | noisy | Static |
| RLBench | 180k | 4 | ✗ (sim) | ✓ | Diverse |
| BridgeV2 | 30k | 1-4 | ✗ | ✗ | Static |

混合比例:Bridge 0.2,RLBench 0.5,DROID 0.3。每个数据集优势互补:
- DROID 提供真实 action annotation
- RLBench 提供精确 camera + 多视角(用 [Robot-Colosseum](https://arxiv.org/abs/2402.08191) 做 background augmentation 增强 visual diversity)
- BridgeV2 用于 video-only training(camera 用 [VGGT](https://arxiv.org/abs/2503.05274) 估计)

---

## 5. 实验结果深度解析

### 5.1 Zero-shot success rate(Table 2)—— 真正的"以一敌百"

RLBench 上(zero-shot 表示 task 没在训练 split 里,但 robot arm 和环境是 seen):

| Method | pick cup | reach target | close drawer | close laptop |
|---|---|---|---|---|
| MV-Policy | 0 | 0 | 0 | 0 |
| $\pi_{0.5}$ | 0 | 5 | 35 | 20 |
| MolmoAct | 20 | 5 | 10 | 0 |
| TesserAct | 0 | 0 | 0 | 0 |
| Cosmos-Policy | 0 | 5 | 20 | 0 |
| **Ours** | **30** | **60** | **50** | **15** |

Real-world 上(object + environment + robot arm xArm 全部 unseen):

| Method | Place Cup | Pick Unseen Toy | Pick Tissue | Close Drawer | Close Box |
|---|---|---|---|---|---|
| MV-Policy | 0 | 0 | 0 | 0 | 0 |
| $\pi_{0.5}$ | 5 | 0 | 0 | 0 | 0 |
| MolmoAct | 10 | 5 | 5 | 5 | 0 |
| TesserAct | 0 | 0 | 0 | 0 | 0 |
| Cosmos-Policy | 0 | 0 | 0 | 0 | 0 |
| **Ours** | **40** | **20** | **15** | **45** | **10** |

**震撼点**:TesserAct 和 Cosmos-Policy 都用相同的 Wan 2.2 backbone 在相同数据上 fine-tune,但它们的成绩几乎是 0,而 Action Images 在所有任务上都显著更高。这说明性能差异**不是来自 backbone 容量或者数据,而是来自 action representation**。这跟 paper 的核心 thesis 完美对齐。

特别地,real-world 上 $\pi_{0.5}$ 这种顶级 VLA 也只能挣扎在 5%——这说明 zero-shot 跨物体跨环境的 generalization 仍然是非常未解的难题,而 pixel-grounded action 在这件事上表现显著更好。

**one-trial open-loop evaluation**(很关键!):模型从单次 forward pass 生成全部未来帧和 action,**没有 online replanning**。这个 setting 比 close-loop reactive policy 难得多。这意味着成绩反映的是 representation 本身的质量,而不是控制循环的鲁棒性。

### 5.2 In-domain RLBench(Table 3)—— 加 action head 进一步拉升

这里 paper 测试了 in-domain 任务,还加了一个 optional action head(在 unified backbone 上挂个 MLP,输入 video latents + camera params + decoded action + observation,直接回归 7-DoF action sequence):

| Method | Avg |
|---|---|
| MV-Diffusion Policy | 17.8 |
| MolmoAct (zero-shot) | 3.3 |
| $\pi_{0.5}$ | 14.4 |
| TesserAct | 20.6 |
| Cosmos-Policy | 20.0 |
| Ours | 20.6 |
| **Ours w/ action head** | **36.7** |

**intuition**:这说明 action image representation 不仅本身足够支撑 zero-shot policy,还能作为一个 strong feature 给更精细的 decoder 用。这是 representation 设计的最高境界——既能独立工作,又能作为插件 boost 其他方法。

### 5.3 Joint Generation 质量(Table 4)—— 视频 + action 双指标

| Model | PSNR↑ | SSIM%↑ | FVD↓ | LPIPS↓ | 2DErr↓ | 3DErr×10³↓ |
|---|---|---|---|---|---|---|
| Cosmos-Predict2.5-14B† | 17.92 | 50.77 | 208.65 | 0.409 | - | - |
| Cosmos-Policy | 18.29 | 53.41 | 192.58 | 0.418 | 2.11 | 19.4 |
| TesserAct | 20.83 | 59.20 | 154.38 | 0.351 | 1.84 | 19.0 |
| TesserAct-RGB | 20.31 | 60.19 | 147.83 | 0.372 | 1.55 | 14.2 |
| **Ours** | **23.48** | **78.62** | **143.74** | **0.209** | - | **12.2** |

- **PSNR**(Peak Signal-to-Noise Ratio):pixel-level fidelity,越高越像 ground truth
- **SSIM**(Structural Similarity):结构相似度,percent
- **FVD**(Fréchet Video Distance):用预训练 I3D 提取特征后计算 Fréchet distance,衡量 video distribution 整体相似度
- **LPIPS**:用 deep feature 衡量 perceptual distance,越低越好
- **2DErr**:2D 轨迹误差(把所有 baseline 的 3D 输出投影到 2D 比对)
- **3DErr×10³**:3D 轨迹误差,乘 1000 标度

Ours 在所有视频指标上都领先,而且 3D action 误差也是最低的。这非常关键:pixel-grounded representation 不光让 video 生成更好,action 本身也更准。这个 mutual benefit 强烈支持"observation 和 action 共享表示空间"的 thesis——它们在 latent space 互相 constrain。

### 5.4 Action-Conditioned Video Generation(Table 5)

给定 action,生成未来 video。对比 [Tora](https://arxiv.org/abs/2410.12790)(2D trajectory-conditioned video gen):

| Model | PSNR↑ | SSIM%↑ | FVD↓ | LPIPS%↓ |
|---|---|---|---|---|
| Tora | 19.76 | 52.43 | 187.41 | 39.62 |
| **Ours** | **31.35** | **67.16** | **115.02** | **21.78** |

差距非常大(PSNR 差 11.6 dB),说明 action image 不仅是好的输出,也是好的 conditioning signal。

### 5.5 Video-to-Action Labeling(Table 6)

给 video,推 action。对比 point tracking 方法 [TAPIR](https://arxiv.org/abs/2306.08379) 和 [CoTracker3](https://arxiv.org/abs/2507.01578):

| Model | Traj Err↓ | Jaccard@4↑ | Avg Jaccard↑ |
|---|---|---|---|
| TAPIR | 14.80 | 40.26 | 29.77 |
| CoTracker3 | 12.91 | 46.15 | 31.20 |
| **Ours** | **5.785** | **64.92** | **46.71** |

- **Jaccard@4**:在 visibility threshold 4 下,Jaccard 相似度
- **Avg Jaccard**:跨不同 threshold 的平均 Jaccard

这个结果非常有意思——action image 训练出来的 model 居然比专门做 point tracking 的方法在 tracking 任务上也强。可能原因:
1. 训练目标更聚焦(end-effector 而非任意点)
2. Video backbone 的 prior 更强
3. Multi-view 提供了几何约束

**practical implication**:可以拿这个 model 给现有无 action label 的 robot video 数据集自动标 action。这对 scaling up robot learning 是巨大价值。

### 5.6 推理效率(Table 7)—— system-level 优化

| Model | Size | GPU | Steps | Frames | Res | Time(s) |
|---|---|---|---|---|---|---|
| TesserAct | 5B | 1 H100 | 50 | 49 | (480,640) | 137.5 |
| DreamZero | 14B | 1 H100 | 16 | 48 | (176,320) | 5.7 |
| DreamZero-Flash | 14B | 2 GB200 | 1 | 48 | (176,320) | 0.15 |
| Ours | 5B | 1 H100 | 50 | 164 | (512,512) | 49.1 |
| + Parallelism | 5B | 8 H100 | 50 | 164 | (512,512) | 11.8 |
| + Caching | 5B | 8 H100 | 16 | 164 | (512,512) | 2.3 |

加速技术:
- [Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719):4-8 GPU 并行
- CFG parallelism
- VAE parallelism
- Caching(KV cache 复用)
- torch.compile

最终 71 FPS / 2.3s 一段 video——已经接近 closed-loop control 可用的速度。DreamZero-Flash 虽然更快(0.15s),但 paper 指出它用 1 step denoising,video quality 严重退化。所以 Action Images 在质量-速度曲线上占了 sweet spot。

---

## 6. 架构图解析

### Fig 2: Action as Image 流程
- 7-DoF vector → 3 个 3D semantic points(position, normal, up)→ multi-view 投影 → RGB Gaussian heatmap stacking → action image with blue-channel 编码 gripper

### Fig 3: Decoding 流程
- Main view heatmap → 2D centroid → ray cast → sample 候选 3D points → project 到 side view → 选 best match → 重建 3D point → 反推 7-DoF

### Fig 4: Unified Training
- RGB video + action video → 3D-VAE → latents → temporal concat → unified sequence → backbone(Wan 2.2)→ flow matching with random mask strategy
- Camera Plücker embedding 通过 $E_c$ 注入 3D-attention 层之间
- Text embedding 通过 cross-attention 注入

### Fig 5: Real-world Zero-shot Rollouts(xArm)
- 左:输入观测 + 预测轨迹
- 中:生成的 future video frames + predicted action-image trajectories
- 右:用 VGGT 重建的 3D 点云里 visualize decoded 3D trajectory
- 颜色编码时间(blue → red)

---

## 7. 我的理解 / Intuition Building

### 7.1 这篇 paper 的真正贡献在哪

paper 标题写的是 "End-to-End Policy Learning via Multiview Video Generation",但我觉得真正的 contribution 是 **"用像素表示 action,让 video prior 直接流到 policy 里"**。这个 idea 看起来 obvious,但执行起来有几个非 trivial 的设计:

1. **3 个 semantic point 的选取**:position + up + normal 这三件套就是 SO(3) 上 minimal 的 pose representation,跟 SLAM、graphics 里的"camera coordinate frame"完全同构。说明作者理解了"action 的本质是 6-DoF pose,7-DoF 只是 parameterization"
2. **Blue channel 复用**:把 gripper openness 藏在背景里这种 trick 看起来 hack,但实用——action image 维度不增加,decoder 又能无损恢复
3. **Multi-view 而非单视角**:不是图方便,而是数学上必须——单视角 2D heatmap 无法 resolve depth
4. **Multiple mask strategy**:一个 backbone 一个 loss,通过 masking 切换 4 种任务,这种 multi-task 设计借鉴了 LLM 的 in-context versatility

### 7.2 跟其他工作的关系

- **跟 [Genie 2](https://arxiv.org/abs/2412.13614) / [Genie 3](https://arxiv.org/abs/2507.07995) 对比**:Genie 系列 learns latent action space from video,Action Images 把 action 显式 ground 到像素。前者 unsupervised,后者 supervised,但后者的 action 可执行性更强。
- **跟 [Pi0 / $\pi_{0.5}$](https://arxiv.org/abs/2504.16054) 对比**:VLA 路线把 action tokenize 成离散 token,flow matching 出来。Action Images 把 action 表示成连续的 spatio-temporal signal,更适合 video prior。
- **跟 [DreamZero](https://arxiv.org/abs/2602.15922) 对比**:DreamZero 也是 zero-shot policy from world model,但它用 latent action code,需要一个 action decoder。Action Images 让 backbone 直接当 policy,跳过了 decoder 这个 generalization bottleneck。
- **跟 [CoTracker](https://arxiv.org/abs/2507.01578) / [TAPIR](https://arxiv.org/abs/2306.08379) 对比**:point tracking 是 general task,Action Images 是 task-specific tracking(end-effector only)。但实验显示 task-specific 反而更准——因为 model 学的是 task-relevant structure。

### 7.3 局限性 & 未来方向(自己的思考)

paper 自己承认 limitation:**只有 open-loop results,没做 closed-loop**。这是大问题——真实 robot deployment 需要 reactive control。但 paper 路线清晰:用 [distillation](https://arxiv.org/abs/2310.13895) + [consistency model](https://arxiv.org/abs/2503.03007) 加速 inference,然后接 closed-loop。

我看到几个潜在 issue:

1. **Heatmap 的 quantization**:image resolution 限制 action 精度。Table 4 里 3DErr 是 12.2×10³ = 0.0122 米 = 1.22 cm——对精细 manipulation 还不够。提高分辨率能改善,但计算量线性涨。
2. **Single object point 假设**:3 个 semantic point 全在 end-effector 上,意味着这个 representation 隐含假设 "action = move end-effector"。对 bimanual / whole-body / mobile manipulation 需要扩展。
3. **Multi-view 数据依赖**:Table 1 显示 RLBench 4 views,但真实世界多视角数据稀缺(DROID 只 2 views,BridgeV2 大多 1 view)。Cross-view generalization 在 data-sparse regime 还需验证。
4. **Occlusion robustness**:heatmap 被 occlusion 时怎么处理?paper 提到 multi-view 提供 redundancy,但没系统测试严重 occlusion 下的退化曲线。
5. **Closed-loop 时序累积误差**:open-loop 一次生成已经看到不错成绩,但 closed-loop 时每步小误差会不会放大?需要 [trajectory optimization](https://arxiv.org/abs/2403.03954) 类的后处理。

### 7.4 更深的 connection——为什么 pixel grounding 这么 powerful

让我推测一下深层原因:

- Video pretraining 的所有 prior——depth estimation、optical flow、object permanence、physics intuition——都工作在 pixel space
- Action 用 token / latent code 时,这些 prior 没法 transfer,因为它们不是以"abstract code"形式存在的
- Action 用 pixel grounding 时,video model 处理 action 跟处理 observation 用的是同一组 circuits
- 这就像 LLM 用 natural language 表示 reasoning:不引入新 interface,prior 可以 fully transfer

所以这个 paper 其实给了一个更通用的 insight:**要让 pretrain knowledge 流到 downstream task,downstream 的 input/output 必须住在 pretrain 的 native representation space**。这个 principle 可以推广到很多场景:
- Tool use prediction → 用 pixel-grounded affordance map
- Audio synthesis → 用 spectrogram 而非 MIDI
- Code generation → 用 character-level 而非 token
- 等等

### 7.5 为什么 multi-view 是必须的(再强调)

我重新算一下:假设单视角,heatmap 中心是 $\hat{u} = (u, v)$,这只能告诉你 end-effector 在 image plane 上的 2D 位置。要 lift 到 3D,需要一个 depth scalar $\lambda$,使得:
$$\hat{q} = \lambda \cdot K^{-1} \cdot [u, v, 1]^\top$$

但 $\lambda$ 从 heatmap 上看不出来(heatmap 只是 2D 强度分布)。Multi-view 提供了第二个视角的 constraint,使得 triangulation 可以解 $\lambda$。这就是 paper 说 "single view often provides only an ambiguous projection" 的精确含义。

### 7.6 跟 NeRF / Gaussian Splatting 的精神连接

Action images 渲染的是 **2D Gaussian heatmap**,这跟 [3D Gaussian Splatting](https://arxiv.org/abs/2208.05232) 用 2D Gaussian 表示 3D scene 的做法本质上一脉相承——都是把 3D 信号 rasterize 到 image space,让 NN 在 image space 处理。Action Images 可以看作 "minimal Gaussian splatting",只用 3 个 splat 表示一个 action。

### 7.7 跟 Diffusion Policy 的关系

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 用 diffusion 直接生成 action sequence vector。Action Images 用 diffusion 生成 action sequence image。前者是"low-dim signal diffusion",后者是"spatio-temporal visual diffusion"。后者能 fully leverage video pretrain,前者不能。这是核心区别。

### 7.8 一个 potential concern:为什么 action image 比直接 3D point trajectory 更好

理论上 3D trajectory(直接用 3D coordinate 序列)信息量更大,但实验显示 Action Images 更好。我推测原因:
- 3D coordinate 没有空间结构,video model 不擅长处理这种 signal
- Action image 跟 observation 在同一个 image space,attention 可以直接 cross-modal
- Action image 蕴含了"哪些像素在动"的信息,这跟 visual grounding 直接挂钩
- Multi-view heatmap 自带几何 consistency 约束,而 raw 3D 没这种 structure

---

## 8. 总结

这篇 paper 的核心 contribution 可以浓缩成一句话:**"在 video world model 里,action 跟 observation 应该住同一个表示空间——像素。然后 video backbone 就直接是 policy。"**

它做到了:
1. **Representation design**:7-DoF → 3 个 semantic 3D point → multi-view RGB Gaussian heatmap,信息无损且 decode 容易
2. **Unified training**:multiple mask strategy 让一个 backbone 同时学 joint gen / action-cond gen / video-to-action labeling / video-only gen
3. **Empirical validation**:zero-shot 在 RLBench 和 real-world 上都碾压用相同 backbone 的 baseline(TesserAct / Cosmos-Policy),证明 representation 才是关键
4. **System efficiency**:5B 模型 + system-level 优化,2.3s 出 164 帧视频,接近 closed-loop 可用

**My take**:这是我近期看到的 robotics world model 方向最 elegant 的工作之一。它没有引入新 module、没有改 loss function、没有特殊训练 trick,只是改了 **action 表示的方式**,就带来了大幅 zero-shot generalization 提升。这种"用 representation 设计代替 architecture engineering"的思路,正是 Karpathy 你过去几年一直在 neural network philosophy 里强调的"substance over form"。

参考链接:
- 项目主页: https://ActionImages.github.io
- Wan 2.1 backbone: https://arxiv.org/abs/2503.20314
- ReCamMaster(camera control): https://arxiv.org/abs/2511.16727
- TesserAct(对照 baseline): https://arxiv.org/abs/2504.20995
- Cosmos-Policy: https://arxiv.org/abs/2601.16163
- DreamZero: https://arxiv.org/abs/2602.15922
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- VGGT(camera estimation): https://arxiv.org/abs/2503.05274
- Flow Matching: https://arxiv.org/abs/2210.02747
- CoTracker3: https://arxiv.org/abs/2507.01578
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DROID dataset: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RLBench: https://arxiv.org/abs/1909.12271
- Robot Colosseum: https://arxiv.org/abs/2402.08191
- Unified Sequence Parallelism: https://arxiv.org/abs/2405.07719
