---
source_pdf: RealWonder Real-Time.pdf
paper_sha256: be54705f7d25f2cb3d6262c16398e3c1010bd1d6edc7a30d64846054d696a5fd
processed_at: '2026-08-11T21:31:55-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RealWonder 用人话说

## 一句话说清楚

**你给一张照片，再告诉它"往这个杯子施加一个向上的力"或者"让机器人抓一下这个布"，它能在 1 秒内开始给你实时生成视频，看着就像真的一样。**

13.2 FPS，480×832 分辨率，单张 H200 GPU 跑。

---

## 这事为什么难？

想象你让 Sora 这种 video model 去"推一个杯子"：

**问题 1：它听不懂"力"**

你跟它说"施加 5 牛顿向上的力在杯子底部"，它懵了。Video model 认识的是 pixel、是 "杯子往左飘" 这种 visual 描述，不认识 Newton、不认识 3D 坐标。Force 是连续的——可以是 0.001N 也可以是 10000N，方向任意，作用点任意，你没法把它变成离散的 token 喂给 model。

**问题 2：没训练数据**

你想教 model "force → video"，需要百万条"某个力 + 对应视频"的 pair。但现实里你拍个视频，根本反推不出来当时有什么力作用在物体上。所以 action-video 数据几乎没法收集。

---

## RealWonder 的妙招：找个"翻译官"

核心 insight 就一句话：**别让 video model 直接理解 force，让 physics simulator 先把 force 翻译成"画面怎么动"，video model 只负责把"怎么动"画好看。**

具体三步：

### 第一步：从照片建个 3D 场景（~13 秒一次性 cost）

拿一张照片，用 SAM 2 分出前景物体和背景，用深度估计模型算出每个 pixel 离镜头多远，然后"撑"成 3D 点云。物体看不见的背面怎么办？用 SAM3D 猜一个完整 mesh 补上。

再用 GPT-4V 看一眼："这是布还是水还是石头？"估出 material 参数。

### 第二步：物理模拟算"后果"

你输入一个 action（"在 (x,y,z) 施加力 f"），physics simulator（用的是 Genesis 引擎）算出每个点下一刻跑到哪、速度多少。这一步 < 2ms，极快。

然后把 3D 运动投影到 2D，得到 **optical flow**——就是"每个 pixel 下一帧该往哪移"的图。再渲染一张粗糙的 RGB 预览（point cloud 拼出来的，丑但能看 structure）。

### 第三步：Video model 画漂亮

关键是这里的两个 trick：

**Trick A：Flow-warped noise（来自 Go-with-the-Flow）**

Diffusion model 一开始是从纯噪声出发去噪。RealWonder 不用纯随机噪声，而是先生成一帧噪声，然后让噪声"跟着 flow 走"——下一帧的噪声是上一帧噪声按 flow 挪过来的。

妙处：噪声的 marginal 分布还是 Gaussian（数学没坏），但 noise 里已经"预装"了 motion 信息。Model 不用学"怎么跟随 motion"，只要学会"利用已经有 motion 结构的 noise"就行。不需要加任何额外 network module。

**Trick B：4 步蒸馏（DMD + Self-Forcing）**

原始 video diffusion 要 50 步去噪，太慢。RealWonder 用 Distribution Matching Distillation 把它压到 4 步，还从 bidirectional（全帧互看）改成 causal（只看前面已生成的帧），这样才能 streaming。

蒸馏时用 Self-Forcing 训练——让 student 在训练时就"rollout 自己的输出"当 context，避免 train-test gap。但标准 Self-Forcing 长序列会崩，作者 fix 了：存 RoPE 之前的 KV cache + 加 attention sink。

**Trick C：SDEdit 融合 RGB 预览**

Student model 训练时只见过 flow conditioning，但推理时还想用那张粗糙 RGB 预览（提供 occlusion 等信息）。怎么办？不重新训练，用 SDEdit：4 步去噪不从第 4 步（纯噪声）开始，从第 3 步开始，起始点是"粗糙 RGB 编码 + flow-warped 噪声"的混合。让模型先用 1 步消化 RGB 的 structure，剩下 3 步正常去噪。

---

## 为什么这招 work？帮你 build intuition

### Intuition 1：分工明确

Physics simulator 擅长算因果——"力 → 位移"，但渲染丑。Video model 擅长画好看，但不懂物理。让它俩各干各的，中间用 optical flow 当接口。

就像拍电影：physics 是"剧本导演"规划谁怎么动，video model 是"特效团队"把剧本拍成大片。导演不用会特效，特效不用懂物理。

### Intuition 2：数据难题被绕过

你不需要"force → video"这种难收集的数据。只需要"flow → video"——随便找个视频用 RAFT 提 flow 就行。Action 这一头被 physics simulator 完全解耦，训练时根本不碰 action。

### Intuition 3：Video prior 反过来"补全"物理

Paper 里有个有意思的例子：simulator 只算了船怎么动，没算水。但生成的视频里有波浪、涟漪、水花——video model 在大量真实视频上学过"船动起来水会这样"，自动补上了。

这是双向 benefit：simulator 给 video model 物理因果骨架，video model 给 simulator 视觉细节补全。

### Intuition 4：对错误容忍度高

作者做了 stress test：把深度估计扰动 20%，或者把 material 从"雪"故意改错成"沙"，生成的视频还是看着合理。因为 video model 对 conditioning noise 有 tolerance，不会因为 physics 输出有点糙就崩。

---

## 实验结果讲人话

**对比 baseline：**

- **PhysGaussian**（纯 3D 物理渲染）：物理对但画面丑，0.2 FPS
- **CogVideoX / Tora**（普通 video model）：画面漂亮但根本不 follow action。你让它"把船往右推"，它给你"船往前开"——因为训练数据里船都是往前开的，它不懂"右"这个 action
- **RealWonder**：又对又好看又快，13.2 FPS

**400 人 user study：** 79%-89% 的人觉得 RealWonder 比 baseline 好，尤其在 "action following" 和 "physical plausibility" 上碾压。

**Teacher-Student 对比：** 4 步蒸馏几乎无损，有些指标甚至略升（causal context 帮了 motion 一致性）。

---

## 一些 Limitation 讲人话

1. **开头要等 13 秒建 3D 场景**——切换场景不流畅，得预 reconstruction
2. **13.2 FPS 对 AR/VR 不够**——人眼舒适要 30+ FPS，robotics simulation 够用
3. **Material 只分 6 类**——现实世界比这复杂多了，半透明、各向异性搞不定
4. **可能"幻觉物理"**——Video prior 学错了物理的话，会生成"看着合理但实际违反物理"的结果，这在 robotics 里 dangerous
5. **长序列 drift 没量化**——streaming 几分钟会怎样没说

---

## 我觉得最有意思的点

**把 physics simulator 当 "action translator" 这个 idea 本身就值这篇 paper。**

之前大家要么硬让 video model 学物理（难、没数据），要么纯用 physics 渲染（丑、没细节）。RealWonder 说：别选了，两个都要，用 optical flow 当接口就行。

这个 decomposition 思路其实是 generalizable 的：任何"连续高维 action + 视觉输出"的问题，都可以想想能不能找个 simulator 先把 action 落到某种 visual representation，再让 generative model 去 refine。

比如 robot learning——与其让 video model 直接学"joint torque → video"，不如用 simulator 把 torque 变成 motion，让 model 学"motion → video"。训练数据从稀缺变成充足。

再比如 AR/VR——把 physics 跑在 headset 端（30 FPS），video generation 跑在云端（13 FPS），异步 pipeline 可能就能做出真正实时的物理交互体验。

这篇 paper 的 contribution 不只是"做了个快系统"，更重要的是给"action-conditioned generation"指了一条可 scale 的路：**不要硬教 generative model 物理要拿物理引擎当翻译官**。

---

# RealWonder: Real-Time Physical Action-Conditioned Video Generation 深度解析

## 1. Paper 整体定位与 Motivation

RealWonder 来自 Stanford（Wei Liu, Ziyu Chen, Hong-Xing Yu, Jiajun Wu）和 USC（Yue Wang），目标是构建第一个 **real-time physical action-conditioned video generation system**，从单张 RGB image 出发，接受 3D physical actions（forces、robot end-effector commands、camera poses），生成 13.2 FPS @ 480×832 的 streaming video。

核心 motivation 来自一个 fundamental mismatch：

- Video diffusion models operate in pixel/latent space，擅长 visual pattern matching
- Physical actions（force、torque、robot joint commands）operate in force/kinematic space，是 continuous、unbounded 的
- 直接把 action tokenize 不可行：force 的 magnitude/direction/application point 都是连续的，且 action-video pair 数据几乎无法收集（无法从 observed motion 反推 causative force）

RealWonder 的核心 insight：**把 physics simulation 作为 intermediate representation bridge**，把 continuous action 转换为 optical flow 和 coarse RGB preview——这两个 signal 天生落在 video model 能消化的 visual domain 里，同时保留了 action→outcome 的 causal relationship。

项目主页：https://liuwei283.github.io/RealWonder

---

## 2. 系统架构深度解析

系统由三个 stage 组成，参考 Figure 2：

```
Single Image I  ──► [3D Reconstruction] ──►  Scene S = B ∪ O
                                                    │
                                                    ▼
User Actions a_t ──► [Physics Simulation] ──►  (p_t, v_t)
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              ▼                                            ▼
                       Optical Flow F_t                          Coarse RGB Ṽ_t
                              │                                            │
                              └───────────────┬────────────────────────────┘
                                              ▼
                              [4-step Distilled Video Generator G]
                                              │
                                              ▼
                                  Photorealistic Video V_t (13.2 FPS)
```

### 2.1 Single-Image 3D Scene Reconstruction

Scene 表示为 $S = B \cup \mathcal{O}$：

**Background** $B = \{(\mathbf{p}_i^B, \mathbf{c}_i^B)\}_{i=1}^{N_B}$
- $\mathbf{p}_i^B \in \mathbb{R}^3$：3D position
- $\mathbf{c}_i^B \in \mathbb{R}^3$：RGB color
- 用 SAM 2 [48] 分割 static region，FLUX inpainting [4] 补全 occluded area，MoGE-2 [59] 估 monocular depth + intrinsics，unproject 到 3D

**Objects** $\mathcal{O} = \{(\mathbf{p}_j^O, \mathbf{c}_j^O, \mathbf{v}_j)\}_{j=1}^{N_O}$
- 多了一个 $\mathbf{v}_j \in \mathbf{R}^3$ velocity（用于 physics）
- 关键 trick：visible surface 用 unprojected pixels，invisible surface（如 object 背面）用 SAM3D [13] feed-forward reconstruction 的 mesh vertices 补全，再用 DUSt3R [60] 做 pose estimation + least-square 求解 scale $s$ 和 translation $\mathbf{T}$ 对齐坐标系

**Materials**：用 VLM（GPT-4V [46]）把每个 object 分到 6 类之一：rigid / elastic / cloth / smoke / liquid / granular，并估参数 $m$（density、friction、Young's modulus、Poisson ratio、viscosity 等）。用户可 override。

整段 reconstruction 在 H200 上耗时 ~13.5s（one-time cost，streaming 开始前的 init phase）。

### 2.2 Physics Simulation as Intermediate Bridge

这是 paper 的核心 idea。统一三类 action 到 3D scene space：

1. **External forces** $\mathbf{f}_t(x,y,z) \in \mathbb{R}^3$：直接作用在 3D 位置
2. **Robot end-effector** $\mathbf{r}_t = \{\mathbf{p}_t^{ee}, \mathbf{q}_t^{ee}, g_t\}$：position、orientation（quaternion）、gripper state，通过 inverse kinematics 转成 joint torques 驱动 Franka model
3. **Camera poses** $\mathbf{C}_t = \{\mathbf{R}_t, \mathbf{t}_t\}$：rendering 时应用

#### Physics step 公式

$$
(\mathbf{p}_{t+1}, \mathbf{v}_{t+1}) = \text{PhysicsStep}(S_t, \mathbf{a}_t) \quad (1)
$$

变量含义：
- $S_t$：当前 scene state（包含所有 dynamic points 的 position 和 velocity）
- $\mathbf{a}_t$：当前 timestep 的 action
- $\mathbf{p}_{t+1}, \mathbf{v}_{t+1}$：下一时刻所有 dynamic point 的位置和速度

Solver 选择：
- **Rigid body**：shape matching [43]（Müller et al. 2005，meshless deformation）
- **Elastic / Cloth / Smoke**：Position-Based Dynamics (PBD) [7, 42]
- **Liquid / Granular**：Material Point Method (MPM) [27]

Single physics step < 2ms，simulation + rendering stream 跑 30 FPS，远快于 13.2 FPS 的 video generator，所以 physics 不是 bottleneck。

#### Optical Flow 公式

$$
\mathbf{F}_t(u, v) = \Pi(\mathbf{p}_t + \Delta t \cdot \mathbf{v}_t) - \Pi(\mathbf{p}_t) \quad (2)
$$

变量含义：
- $\mathbf{F}_t \in \mathbb{R}^{\tilde{H} \times W \times 2}$：pixel-space optical flow
- $\Pi$：camera projection（3D → 2D pixel）
- $(u, v)$：pixel coordinates
- $\mathbf{p}_t$：当前 3D position
- $\mathbf{v}_t$：3D velocity
- $\Delta t$：时间间隔

intuition：用 forward projection 直接算出每个 pixel 下一帧应该跑到哪里，这比让网络学 motion 要容易得多，且天然 ground truth。

#### Coarse RGB Preview

$\tilde{\mathbf{V}}_t \in \mathbb{R}^{\check{H} \times W \times 3}$：用 point cloud rasterization 渲染的粗略 RGB，提供 occlusion、structure cue 这些 flow 表达不了的信息。

### 2.3 Real-Time Conditional Video Generation

这是把 physics 输出转成 photorealistic video 的关键 module。

#### Stage 1: Flow-Conditioned Teacher Model

Base model：VideoXFun [2] 的 Wan2.1-1.3B-InP [57]（image-to-video inpainting variant）。

关键 trick 来自 Go-with-the-Flow [9]：**flow-warped noise**。

具体做法：sample 单帧 Gaussian noise $z \sim \mathcal{N}(0, I)$，根据 flow field $\mathbf{F}$ 做 temporal warp：

$$
z^{\mathbf{F}} = \text{Warp}(z, \mathbf{F})
$$

warp 后的 noise $z^{\mathbf{F}}$ 仍是 Gaussian distribution（保证 diffusion math 仍成立），但已经在 noise structure 里 encode 了 motion pattern。这样不需要额外的 embedding module 或者 architectural change，直接通过 initial noise 注入 control。

Training 用 flow-matching objective，把 base model 微调到能 take flow conditioning。冻结原 weight，每个 attention block 注入 LoRA [22] rank=2048（这是非常大的 rank），300K iterations，lr=1e-5。

#### Stage 2: Causal Distillation for Streaming

Teacher 是 bidirectional model，处理整个 sequence，无法 streaming。需要蒸馏成 causal student（4 step denoising）。

采用 Distribution Matching Distillation (DMD) [70, 71]：

$$
\nabla L_{\text{DMD}} = \mathbb{E}_t [\nabla_\theta \text{KL}(p_{\text{fake},t} \| p_{\text{real},t})] \quad (3)
$$

变量含义：
- $p_{\text{fake},t}$：student 在 timestep $t$ 估计的 distribution
- $p_{\text{real},t}$：real data distribution（通过 teacher 的 ODE solver 估计）
- $\theta$：student 参数
- $\nabla_\theta$：对 student 参数求梯度

intuition：直接匹配 output distribution 而非逐 sample regression，更稳定。

引入 Self-Forcing [25] 的 autoregressive rollout 训练，让 student 在训练时就 exposure 到自己的 output distribution（解决 train-test gap）。但标准 Self-Forcing 在长序列上 quality 退化，作者 fix 了这个：

- **存储 RoPE [51] 之前的 KV cache**（避免 attention bias 累积）
- **加 attention sink**（参考 [29, 37, 50]）

蒸馏三阶段细节：
1. ODE regression（Self-Forcing 风格）：从 post-trained teacher sample 2K ODE trajectories，train student 3K iterations with MSE loss，让 bidirectional model 适应 causal attention
2. Distribution matching distillation：600 iterations，batch size=64
3. 总 compute：128 A100 GPU-days

### 2.4 Streaming Inference：SDEdit 融合 RGB Preview

蒸馏后的 student 只 trained on flow conditioning，但 RGB preview $\tilde{\mathbf{V}}_t$ 在 inference 时也有用（occlusion、structure）。怎么融合？用 SDEdit [41]。

从 4-step denoising 的第 3 步开始（而非第 4 步即纯 noise）：

$$
\mathbf{V}_{t,(3)} = \alpha_{(3)} \cdot \mathcal{E}(\tilde{\mathbf{V}}_t) + \sqrt{1 - \alpha_{(3)}^2} \cdot z_t^{\mathbf{F}} \quad (4)
$$

变量含义：
- $\mathcal{E}$：VAE encoder
- $\alpha_{(3)}$：diffusion step 3 对应的 noise schedule coefficient
- $z_t^{\mathbf{F}}$：flow-warped noise
- $\mathbf{V}_{t,(3)}$：第 3 步（共 4 步）的 noisy latent 起始点

intuition：让模型先用 1 个 step 去 denoise 这个 mixture（相当于把 coarse RGB "refine" 一下），剩下 3 个 step 做 standard denoising。这样 motion accuracy 来自 flow，structure cue 来自 RGB preview，二者各司其职。

#### Streaming 生成公式

$$
\mathbf{V}_{t+1} = \mathcal{G}(\text{text}, \mathbf{I}, \mathbf{F}_{t+1}, \tilde{\mathbf{V}}_{t+1}, \{\mathbf{V}_j\}_{j \leq t}) \quad (5)
$$

变量含义：
- $\mathcal{G}$：distilled causal video generator
- $\text{text}$：text prompt
- $\mathbf{I}$：input image（始终作为 anchor）
- $\mathbf{F}_{t+1}$：当前 frame 的 optical flow condition
- $\tilde{\mathbf{V}}_{t+1}$：当前 frame 的 coarse RGB preview
- $\{\mathbf{V}_j\}_{j \leq t}$：之前所有已生成的 frame（causal context）

整条 pipeline 见 Algorithm 1（附录），latency 分两部分：
- Init phase（reconstruction + material estimation）：~13.5s
- Streaming loop（physics + 4-step diffusion）：13.2 FPS，sub-100ms latency per frame

---

## 3. 实验数据表深度解读

### 3.1 Table 1: Quantitative Comparison

| Methods | Visuals ↑ | Aesthetics ↑ | Consistency ↑ | PhysReal ↑ |
|---------|-----------|--------------|---------------|------------|
| PhysGaussian | 0.454 | 0.517 | 0.221 | 0.468 |
| CogVideoX | 0.696 | 0.603 | 0.234 | 0.624 |
| Tora | 0.700 | 0.588 | 0.223 | 0.578 |
| **RealWonder** | **0.708** | 0.593 | **0.265** | **0.705** |

关键观察：
- **PhysGaussian** Visuals 低（0.454）符合预期，因为它完全依赖 3D representation 渲染，没有 video prior 的 photorealism
- **CogVideoX** 和 **Tora** Visuals 高但 PhysReal 不如 RealWonder：它们生成看起来漂亮的 video，但根本不 follow physical action（如 Tora 把 "向右移动的船" 误解成 "向前移动"）
- RealWonder 在 Consistency 上显著领先（0.265 vs 0.221-0.234），说明 causal + flow conditioning 提供了强 temporal consistency
- Aesthetics RealWonder 不是最高，但接近：4-step distilled model 会有少量 quality loss（参考 Table S1：teacher 0.605 vs student 0.593）

### 3.2 Table 2: 2AFC Human Study（400 participants）

| Comparison | Action Following | Motion Fidelity | Visual Quality | Physical Plausibility |
|------------|------------------|------------------|----------------|----------------------|
| over PhysGaussian | 88.4% | 82.0% | 88.6% | 87.1% |
| over CogVideoX-I2V | 89.6% | 71.0% | 75.3% | 85.9% |
| over Tora | 83.9% | 67.9% | 75.4% | 79.7% |

intuition：人对 physical plausibility 的偏好非常显著（79.7%-87.1%），这是 RealWonder 最大的优势点。Motion Fidelity 相对较低（67.9% over Tora）说明 Tora 在 motion 自然度上仍有竞争力，但 Action Following 上 RealWonder 完胜（83.9%）。

### 3.3 Table 3: Runtime Performance

| Method | FPS | Latency |
|--------|-----|---------|
| Tora | 0.107 | — |
| CogVideoX-I2V | 0.225 | — |
| PhysGaussian | 0.207 | 4.84s |
| **RealWonder** | **13.2** | **0.73s** |

RealWonder 比 baselines 快了 **~60-120x**，这是 distillation + 4-step + causal streaming 带来的指数级提升。注意 baseline 们的 "FPS" 实际是 batch generation time 除以 frame count，并非真正 streaming。

### 3.4 Table S1: Teacher-Student Comparison

| Methods | Imaging ↑ | Aesthetic ↑ | Consistency ↑ | PhysReal ↑ |
|---------|-----------|--------------|----------------|-------------|
| Teacher | 0.713 | 0.605 | 0.271 | 0.698 |
| Student | 0.708 | 0.593 | 0.265 | 0.705 |

distillation 几乎无损（甚至 PhysReal 略升，可能因为 causal context 帮助了 motion 一致性），这是非常重要的结果：证明 4-step causal 蒸馏可行。

---

## 4. 关键 Ablation Studies

### 4.1 Physics Simulator 的影响（Figure 7）

去掉 physics simulator，只用 text prompt 描述 "wind blowing from the right"，结果 smoke 完全不改变方向。说明 text-only conditioning 无法传达 physical action 的精确性。

### 4.2 Conditioning Signals 的影响（Figure 8）

- **Full model**（flow + RGB preview）：正确响应 3D point force upward
- **w/o RGB preview**（只用 flow）：不 adhere to simulated overall motion
- **w/o flow**（只用 RGB）：video model 倾向 ignore motion signal，产生 static video

intuition：flow 提供 dense motion guidance，RGB preview 提供 structural cue（occlusion、object identity），二者互补，缺一不可。

### 4.3 Reconstruction Robustness（Figure S1）

扰动 depth ±20% 或把 material 从 "snow" 改成 "sand"，visual realism 仍 robust，因为 video generator 本身对 conditioning noise 有 tolerance。这是 important property——让 pipeline 不依赖于 perfect 3D reconstruction。

### 4.4 Ambient Dynamics Compensation（Figure S2）

Simulator 只算 boat motion，不 model water dynamics。但 video generator 自动 synthesize 周围的 waves 和 ripples。这说明 video prior 起到了 "physics 补全" 的作用，类似 generative model 把粗糙 simulation "翻译" 成 photorealistic video。

---

## 5. Training Data 细节

- **200K flow-video pairs**：
  - 180K real-world clips from OpenVid [44]，filter 到 80-120 frames
  - 20K synthetic videos from Wan2.1-14B-T2V [57]，使用 VidProM [61] 的 prompts
- Flow 用 RAFT [54] 提取
- 关键：**不需要 action-video pairs**，只需要 flow-video pairs，这是 scalability 的关键

intuition：作者通过 physics simulation 把 action 解耦掉了，training 时根本不接触 action，只在 inference 时用 physics 把 action 转成 flow。这是非常 elegant 的设计——把无法收集的 supervision（action→video）替换成容易收集的 supervision（flow→video）。

---

## 6. Material Parameters 详解（Table S2）

附录里给出了详细的物理参数：

| Solver | Material | 关键参数 |
|--------|----------|----------|
| General | — | Step time 1e-2, substeps 10, gravity (0,0,-9.8) |
| Shape matching | Rigid | friction coefficient 0.1 |
| MPM | Liquid | Young's modulus 1e7, Poisson ratio 0.2 |
| MPM | Granular | Young's modulus 1e6, Poisson ratio 0.2, friction angle 45° |
| PBD | Elastic | stretch/bending/volume compliance 0, relaxation 0.3/0.3/0.1 |
| PBD | Cloth | stretch compliance 1e-7, bending compliance 1e-5 |
| PBD | Smoke | viscosity 0.1 |

注意 granular 比 liquid 的 Young's modulus 低一个数量级（1e6 vs 1e7），这符合 sand 比 water 更软的直觉。Friction angle 45° 表示 sand 的 internal friction（典型 dry sand 值）。

---

## 7. Build Intuition：为什么这个 approach 能 work？

让我从几个 angle 帮你 build intuition：

### 7.1 "Action Tokenization" 困境的绕过

直接 tokenize force 的问题：force 是 $\mathbb{R}^3$ 的 continuous 量，magnitude 跨度从 micro-Newton 到 mega-Newton，application point 任意，direction 任意。任何 discretization 都会 lose 信息或爆炸式扩张 vocabulary。

RealWonder 的 trick：force 先经过 physics simulator 算成 displacement field，再 project 成 optical flow。**Physics simulator 天生 handle continuous unbounded force**，因为它是数值积分器。Video model 只需要懂 flow（2D pixel motion），这是它已经擅长的事。

### 7.2 为什么 flow-warped noise 是 elegant？

Go-with-the-Flow [9] 的核心 insight：diffusion model 的初始 noise 决定了生成的 random structure，但是如果你让 noise 在时间上"跟着 flow 走"，那么 noise 在 latent space 里就预 encode 了 motion。具体地：

- 单帧 noise $z$ 是 i.i.d. Gaussian
- Warp 后 $z^{\mathbf{F}}_t = z^{\mathbf{F}}_{t-1} \text{ warped by } \mathbf{F}_t$
- Marginal distribution 仍是 Gaussian（因为 warp 是 deterministic bijection），所以 forward diffusion math 仍然成立
- Conditional distribution 已经 encode 了 motion 信息

这相当于把 "teach model to follow motion" 的问题转化成 "model 学会 leverage 已经 motion-aware 的 noise structure"，更易学。

### 7.3 Causal + Self-Forcing + Attention Sink

标准 video diffusion 是 bidirectional（全 frame 互相 attend），无法 streaming。改成 causal 后有 train-test gap：训练时用 ground truth context，推理时用自己生成的 context（可能有 error）。

Self-Forcing [25] 的核心：训练时就 rollout 自己的 output 作为 context，让 model 适应自己的 distribution。但长序列会 quality 退化，原因之一是 RoPE [51] 在 attention 时引入 position bias，KV cache 中存储的 K、V 是 RoPE 之后的，每生成新 frame，相对 position 编码都会变。

RealWonder 的 fix：**store KV cache before RoPE**，这样存储的是 "原始" K、V，每次 query 时再实时 apply RoPE，避免 bias 累积。再加 attention sink（保留前几个 token 作为 "anchor"）让 attention 不会 drift 到无意义 region。

### 7.4 SDEdit 融合的妙处

为什么从 step 3 而不是 step 4 开始？因为：
- 4-step diffusion 中 step 4 是最 noisy（接近纯 noise）
- 如果直接从 coarse RGB encoded latent 开始（step 0），会 over-commit 到 preview 的 artifacts
- Step 3 是个折中：保留 RGB 的 structural information（α_(3) 权重），同时留 1 step 让 model 用 flow-warped noise 的 motion 信息做 refine

公式 (4) 里的 $\alpha_{(3)}$ 控制 trade-off：太大 → 被 preview artifact 主导；太小 → preview 没用。这是 SDEdit 的标准用法，但在 flow-conditioned video 里是新颖的。

### 7.5 Video Prior 作为 "Physics Enhancer"

Figure S2 的 boat 例子非常有意思：simulator 不算 water dynamics（只算 rigid body motion of boat），但生成的视频里有 waves、ripples、water splash。这说明 video prior（在大量 real video 上训练）"补全" 了 simulator 没建模的部分。

这是个非常重要的 insight：**physics simulator 提供因果骨架，video prior 提供 photorealistic 细节**。两者各取所长。Simulator 的弱点（视觉粗糙、缺细节）被 video prior 弥补，video prior 的弱点（不懂物理因果）被 simulator 弥补。

---

## 8. 与 Related Work 的对比

### 8.1 vs WonderPlay [33]

WonderPlay 是同一作者组的前作，也用 physics simulation + video generation，但需要 "slow optimization of explicit 4D representations"——几分钟生成短视频。RealWonder 用 distillation 把生成速度提到 13.2 FPS，且不需要 4D 优化。

WonderPlay: https://arxiv.org/abs/2501.05311

### 8.2 vs Go-with-the-Flow [9]

RealWonder 直接借鉴 Go-with-the-Flow 的 flow-warped noise trick，但 Go-with-the-Flow 是 general motion-controlled video generation（用户提供 flow），RealWonder 的 flow 来自 physics simulation，从而把 action→flow 这一步显式建模。

Go-with-the-Flow: https://arxiv.org/abs/2501.08331

### 8.3 vs MotionStream [50]

MotionStream（concurrent work）也做 real-time streaming + trajectory control，但用 VideoXFun control variant + 额外 action module。RealWonder 不用额外 module，直接 inject flow 到 initial noise，简化了 student model 用于 DMD distillation。

MotionStream: https://arxiv.org/abs/2511.01266

### 8.4 vs PhysGaussian [64]

PhysGaussian 把 MPM 直接 integrate 到 3D Gaussian Splatting，能做 physics simulation + rendering，但视觉质量受限（无 video prior），且需要 3DGS optimization。RealWonder 把 3DGS 这层去掉了，用 video model 替代 rendering。

PhysGaussian: https://arxiv.org/abs/2311.12913

### 8.5 vs Genie [8] / GameGen-X [10]

这些是 game world model，action space 是离散的（按键、joystick），action-video pair 容易获取。RealWonder 处理的是 continuous physical action，且不需要 action-video pair。

Genie: https://arxiv.org/abs/2402.15391
GameGen-X: https://arxiv.org/abs/2411.00769

---

## 9. Limitations 与 Future Direction

作者明确指出：

1. **3D reconstruction 不准**：depth estimation error 会导致 simulation 和 video 都受影响。Future work 可以用 large reconstruction model 如 VGGT [58] 或 GS-LRM [73]
2. **Physical correctness ≠ physical plausibility**：当前只追求 "看起来合理"，不严格 enforce physical law。Video prior 可能 synthesize 违反物理的细节
3. **Material 估计粗糙**：VLM 估的参数不一定准，依赖 user override

VGGT: https://arxiv.org/abs/2403.12947
GS-LRM: https://arxiv.org/abs/2404.19705

---

## 10. 公式变量总览

让我把所有公式的变量汇总，方便 cross-reference：

**公式 (1)** PhysicsStep
- $S_t$：scene state at time $t$
- $\mathbf{a}_t$：action at time $t$
- $\mathbf{p}_{t+1}, \mathbf{v}_{t+1}$：next position and velocity
- 下标 $t$：time index

**公式 (2)** Optical Flow
- $\mathbf{F}_t \in \mathbb{R}^{\tilde{H} \times W \times 2}$：optical flow
- $\Pi$：camera projection function
- $\mathbf{p}_t$：3D position
- $\mathbf{v}_t$：3D velocity
- $\Delta t$：time step
- $(u,v)$：pixel coordinate

**公式 (3)** DMD Loss
- $L_{\text{DMD}}$：distribution matching distillation loss
- $p_{\text{fake},t}$：student distribution at diffusion timestep $t$
- $p_{\text{real},t}$：real data distribution
- $\theta$：student model parameters
- $\nabla_\theta$：gradient w.r.t. $\theta$
- $\mathbb{E}_t$：expectation over diffusion timesteps

**公式 (4)** SDEdit Mixing
- $\mathbf{V}_{t,(3)}$：noisy latent at diffusion step 3 (out of 4)
- $\alpha_{(3)}$：noise schedule coefficient at step 3
- $\mathcal{E}$：VAE encoder
- $\tilde{\mathbf{V}}_t$：coarse RGB preview
- $z_t^{\mathbf{F}}$：flow-warped noise
- 下标 $(3)$：diffusion step index

**公式 (5)** Streaming Generation
- $\mathcal{G}$：distilled causal video generator
- $\text{text}$：text prompt
- $\mathbf{I}$：input image
- $\mathbf{F}_{t+1}$：current frame optical flow
- $\tilde{\mathbf{V}}_{t+1}$：current frame RGB preview
- $\{\mathbf{V}_j\}_{j \leq t}$：causal context (all previous frames)

---

## 11. 我对这篇 paper 的整体评价

### Strengths

1. **Insight 漂亮**：physics simulation 作为 intermediate bridge 解决了两个 fundamental obstacle（action tokenization、action-video pair 数据稀缺），非常 elegant
2. **工程整合度高**：3D reconstruction + physics + distillation + SDEdit 整合到一个 streaming pipeline，13.2 FPS 在 480×832 上是真 real-time
3. **Ablation 充分**：physics、flow、RGB preview 三个 condition 各自贡献都验证了
4. **Robustness 好**：对 reconstruction error 容忍度高，这是 deploy 到真实世界的关键

### 潜在 Concerns

1. **Init phase 13.5s**：reconstruction + material estimation 一次性 cost，对 interactive 应用来说还是太长。可以预 reconstruction 解决，但 image 切换场景时不流畅
2. **13.2 FPS 是否够 "interactive"**：人眼舒适阈值通常 30 FPS，13.2 在 robotics simulation 可能够，但 AR/VR 体验受限
3. **Material 分到 6 类**：现实世界 material 远比这复杂（半透明、各向异性、复合材料），VLM 分类可能 mislead simulation
4. **Video prior 可能"幻觉"物理**：Figure S2 里 video model 自动 synthesize water ripples 是好事，但如果 video prior 学错了物理（比如 reverse gravity 的视频在训练集里），就可能产生 physically implausible but visually plausible 的结果，这是 dangerous 的
5. **Long-horizon drift**：虽然 Self-Forcing + attention sink 缓解了 drift，但 streaming 几分钟后会怎样？paper 没量化
6. **Single object / 多 object 交互**：robot gripper 与 fluid 的交互很惊艳，但复杂 multi-body collision（比如一袋多 object 互相碰撞）的效果未充分展示

### Inspirations for Future Work

1. 把 physics simulator 替换成 differentiable simulator，让 video prior 的 error signal 能 backpropagate 回 simulation parameter，做 online calibration
2. 把 action space 扩展到 non-physical action（比如"打开灯"、"点燃火"）——可能需要 hybrid semantic + physics simulator
3. 把 video generator 的 distribution matching distillation 换成 consistency model [Song et al.]，可能更稳
4. 用 RLHF-style feedback 让人 judge physical plausibility，fine-tune video prior
5. AR/VR 场景：把 latency 拆开，physics 跑 30 FPS（headset 端），video generation 跑 13 FPS（cloud 端），异步 pipeline

---

## 12. 参考 Web Links

- RealWonder 项目主页：https://liuwei283.github.io/RealWonder
- Wan2.1 视频生成模型：https://arxiv.org/abs/2503.20314
- VideoXFun 框架：https://github.com/aigc-apps/VideoX-Fun
- Go-with-the-Flow (flow-warped noise)：https://arxiv.org/abs/2501.08331
- Distribution Matching Distillation (DMD)：https://arxiv.org/abs/2311.18828
- Improved DMD：https://arxiv.org/abs/2405.14867
- Self-Forcing：https://arxiv.org/abs/2506.08009
- SDEdit：https://arxiv.org/abs/2108.01073
- LoRA：https://arxiv.org/abs/2106.09685
- RAFT (optical flow)：https://arxiv.org/abs/2003.12039
- Genesis physics engine：https://github.com/Genesis-Embodied-AI/Genesis
- SAM 2：https://arxiv.org/abs/2408.00714
- DUSt3R：https://arxiv.org/abs/2312.14132
- MoGE-2：https://arxiv.org/abs/2507.02546
- WonderPlay (前作)：https://arxiv.org/abs/2501.05311
- PhysGaussian：https://arxiv.org/abs/2311.12913
- PhysDreamer：https://arxiv.org/abs/2404.13026
- Tora：https://arxiv.org/abs/2410.10173
- CogVideoX：https://arxiv.org/abs/2408.06091
- RoFormer (RoPE)：https://arxiv.org/abs/2104.09864
- Position-Based Dynamics：https://dl.acm.org/doi/10.1016/j.jvcir.2007.01.005
- Material Point Method (SIGGRAPH course)：https://arxiv.org/abs/2501.03575
- Shape Matching (Müller 2005)：https://dl.acm.org/doi/10.1145/1073204.1073216
- VGGT：https://arxiv.org/abs/2403.12947
- Genie：https://arxiv.org/abs/2402.15391
- MotionStream：https://arxiv.org/abs/2511.01266
- Cosmos World Foundation Model：https://arxiv.org/abs/2501.03575
- SAM3D：https://arxiv.org/abs/2511.16624
- Matrix-Game 2.0：https://arxiv.org/abs/2508.13009
- WorldScore benchmark：https://arxiv.org/abs/2501.05311 (related, by same group)

---

## 13. 总结

RealWonder 的核心贡献可以归纳为三句话：

1. **Conceptual**：把 physics simulation 提升为 "action → visual" 的 universal translator，绕过 action tokenization 和 action-video pair 数据稀缺两个 fundamental obstacle
2. **Architectural**：flow-warped noise + DMD + Self-Forcing + SDEdit 的组合，把 multi-step bidirectional video diffusion 蒸馏成 4-step causal streaming generator，且几乎无损
3. **System**：3D reconstruction + physics + video generation 三 module co-design for streaming，达到 13.2 FPS 真实时性能

最重要的 takeaway：**让 physics simulator 干 physics 的活，让 video model 干 visual 的活，通过 optical flow 这个 visual representation 做接口**。这个 decomposition 比试图让 video model end-to-end 学物理要 tractable 得多，也更有可解释性。

未来如果这条路走通，会带来 robot learning（在 video prior 上做 motion planning）、AR/VR（实时物理交互预览）、game engine（生成式物理引擎）的重大变化。我预期会有大量 follow-up work 把 action space 扩展（contact-rich manipulation、tool use）、把 simulator 换成 differentiable version、把 video model 换成更 efficient variant（如 consistency model）。这是个非常 promising 的方向。
