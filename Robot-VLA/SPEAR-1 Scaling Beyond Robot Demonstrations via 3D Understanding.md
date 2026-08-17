---
source_pdf: SPEAR-1 Scaling Beyond Robot Demonstrations via 3D Understanding.pdf
paper_sha256: 2a58afc5cb789bda91c18e3cd6cff3ee637b1aec282ba343bbbc178d7eb5dc6c
processed_at: '2026-08-12T09:49:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 SPEAR-1

Andrej, 我用更 intuitive 的方式再讲一遍。

---

## 一句话版本

当前所有 VLA 都有个通病: 它们基于的 VLM 只有 2D 视觉理解, 完全不懂 3D 空间关系。SPEAR-1 的核心 idea 非常 simple — 在训练 robot 之前, 先用 cheap 的 3D-annotated 普通 images 把 VLM 变成 3D-aware, 这样你后面就不需要那么多昂贵的 robot demonstrations 来 implicit 学习 3D 理解了。

---

## 为什么这是个 real problem

先 build 一下 intuition。想象你从没学过 depth perception, 只能从 2D 图像判断 world。你看到一个 cup 在桌上, 你知道 "这是 cup", 但你不知道它离 camera 多远, 也不知道它和旁边的 plate 的实际 distance 是多少。你看到的只是 pixels 的 2D projection。

传统 VLM (PaliGemma, LLaVA 等等) 就是这个状态。它们在 internet-scale 2D image-text data 上 pretrained, 学会了非常好的 semantic understanding (这是 cup, 那是 plate), 但是对 3D spatial relations 一无所知。

然后我们期望这样的 VLM 去 control robot arm 做精细操作。Robot arm 需要知道: 物体在 3D space 哪里? 我应该往哪个方向移动多少 distance? 这完全是 3D reasoning。

之前大家怎么解决这个 gap? 用大量 robot demonstrations! 反正 demonstrations 里包含了 action 信息, model 可以 implicit 地从 action supervision 中 "infer" 出 3D structure。π0, π0.5 都是这么做的, 用了 900M+ frames 的 robot data。

**问题**: robot data 太 expensive 了。每个 demonstration 都需要 human teleoperation, 一个 setup 一个 setup 地收集, 很难 scale。而且 robot data 通常 environment-specific, 对 generalization 帮助有限。

---

## SPEAR-1 的 key insight

作者说: wait, 我们干嘛非要从 robot data 中 implicit 学 3D? 3D understanding 本身不需要 robot, 我们可以用普通 images + 3D annotations 来 explicitly 教 VLM 3D reasoning, 这样:

1. 数据 source: 普通 2D images (海量存在, internet 上到处都是)
2. Annotation cost: 用现成 foundation models 自动标注 (Gemini + SAM2 + MoGe), 几乎 free
3. 任务设计: 让 VLM 做一些 embodied-inspired 3D VQA tasks

完成后, 这个 3D-aware VLM 就有了 spatial reasoning, 再 fine-tune 到 robot control 时就 data-efficient 得多。

**效果**: SPEAR-1 用 ~45M robot frames, 匹配了用 900M+ frames 训练的 π0.5, 在 zero-shot Franka (DROID) 上。**20× data efficiency**。

---

## 具体怎么做的

### Stage 1: Build SPEAR-VLM (3D-aware VLM)

**Architecture**: 拿 PaliGemma (Google 的 VLM), 加一个 MoGe depth encoder 作为 second vision backbone。两个 encoder 的 features 都 project 到 LLM embedding space, 然后 average fuse 起来。

为什么选 MoGe? 因为它做的是 **affine-invariant depth estimation**。这个很关键 — 不同的 camera 有不同的 intrinsics, 如果你的 depth estimator 假设 fixed intrinsics, 那跨 environment generalize 就差。Affine-invariant 允许 model 适应不同 cameras, 这对 generalist robot policy 很重要。

**Tokenizer 扩展**: 加 1024 个新的 3D tokens。为什么要新 tokens? 因为 3D coordinates 和普通 text/visual tokens 概念上不同, 需要 dedicated representation。距离值近似 normal distribution, 所以用 non-uniform bins — mean 附近 fine, tails 处 spread, 让 token distribution 近似 uniform。

**3D VQA tasks**: 这是最 core 的 design。作者设计了一些 control-inspired 问题:
- "Object X 的 3D bounding box vertices 是什么?"
- "Object X 和 Object Y 之间的 3D distance xyz components?"
- "Camera 到 Object X 的 distance?"
- "哪个 object 更近?"

这些都是 VLM 可以用 text 回答的问题, 但回答它们需要真正理解 3D 空间关系。而且这些问题直接 inspired by robot 要做的 embodied tasks。

**Annotation pipeline** (全自动):
1. Gemini 检测 2D bounding boxes + semantic labels
2. SAM2 用 boxes prompt 出 instance segmentation masks
3. MoGe 给出整张图的 3D point cloud
4. 用 mask filter point cloud 得到 object-level point cloud
5. Open3D 做 statistical outlier removal + oriented 3D bounding box
6. 自动生成 templated Q&A pairs

只用 230k images (200k from EgoExo4D + 30k from Bridge V2), 就能让 VLM 学会 3D understanding。这比 900M robot frames 便宜无数倍。

**Two-stage training** (像 LLaVA):
- Stage 1: 短 (2k steps), 只 train 新 init 的 weights (MoGe projector + 3D token embeddings) + SigLIP projector
- Stage 2: 长 (10k steps), freeze SigLIP + MoGe encoders, fine-tune 其他, 3D token loss scaled by 2

### Stage 2: Build SPEAR-1 (Robot Foundation Model)

Architecture 跟 π0 一样: SPEAR-VLM + 一个 action expert (也是 Gemma-style transformer, ~300M params)。Action expert 通过 shared attention 来 attend VLM 的 intermediate key-value pairs。

**Flow matching formulation** (这是最 technical 的部分, 我尽量讲 intuition):

核心 idea: action prediction 是个 generative modeling 问题。我们有个 ground truth action sequence `A_t`, training 时把 noise 混进去得到 `A_t^τ` (τ 是 noise level, 0=pure noise, 1=clean), 让 model 预测 denoising vector field。Inference 时从 pure noise 开始, 沿着 learned vector field integrate 得到 clean action。

**Translation**: 在 R³ 上做 linear interpolation 就好
$$x_t^\tau = \tau x_t + (1-\tau) x_\epsilon$$

- `x_t`: ground truth translation
- `x_ε`: sampled Gaussian noise
- `τ`: timestep, 0 到 1
- `x_t^τ`: noisy intermediate

**Rotation**: 这里就有意思了。Rotation 用 unit quaternion 表示, 住在 S³ manifold 上 (4D unit sphere)。如果用 linear interpolation, intermediate states 不是 unit quaternion, 需要 project, 这会 break geometry。

作者用 **spherical linear interpolation (SLERP)** 在 S³ 上:
$$q_t^\tau = \frac{\sin((1-\tau)\theta)}{\sin(\theta)} q_\epsilon + \frac{\sin(\tau\theta)}{\sin(\theta)} q_t$$

- `q_t`: ground truth quaternion
- `q_ε`: uniform random quaternion on S³
- `θ = cos⁻¹(q_ε · q_t)`: 它们之间的 angular distance
- 整个 formula 保证 `q_t^τ` 始终在 S³ 上 (是 valid unit quaternion)

**Inference 也要在 manifold 上 integrate**:
- Translation: Euler integration, `x^{τ+δ} = x^τ + δ·v(x^τ)`
- Rotation: 先把 predicted velocity 转成 angular velocity `ω = 2·Im(q* ⊗ q̇)`, 构造 small delta quaternion `Δq = [cos(Δφ/2), ω̂ sin(Δφ/2)]`, 然后 `q^{τ+δ} = q^τ ⊗ Δq`

这保证了整个 denoising trajectory 都在 S³ 上, mathematically elegant 且 empirically better (ablation 显示 S³ flow matching 比 linear flow matching 高 ~7%)。

**其他 engineering tricks**:
- Image resolution: 280×210 external, 112×112 wrist。**不 distort aspect ratio** — naive resize 会 break camera intrinsics, 对 depth estimation 有害
- VLM training 时 SigLIP + MoGe 都 trainable, VLA training 时 freeze MoGe (VLA training 会 degrade MoGe representations)
- Global quantile normalization across all datasets (encourage cross-dataset knowledge sharing, 不要 memorize 每个 dataset)
- EMA checkpointing (stabilize final performance, 这个 trick 在 RL/diffusion training 中很有用)
- Action chunk H=5, 5Hz control frequency

---

## 结果有多强

**Main result**: 在 Franka (DROID) zero-shot unseen environments 上, SPEAR-1 显著 outperform π0-FAST, 匹配 π0.5, 但用了 20× less robot data。

这个特别 impressive 因为 DROID 是出了名的难 — 20× more unique scenes than Bridge V2, camera viewpoints vary a lot, Franka arm 在 real-world environments 而不是 toy kitchens。之前能在 DROID 上做 zero-shot Franka 的只有 π0-FAST 和 π0.5。

**3D Ablation** (Table 1, 最关键 experiment):
- PaliGemma baseline: 20.8% avg success
- SPEAR-VLM 只用 random pixel 3D coords: 20.8% (没提升!)
- SPEAR-VLM 用 object-level 3D tasks: 35.4% (+14.6%)

这说明什么? 不是简单 "加 3D 信息" 就行, 要加 **semantic-grounded object-level 3D supervision**。Random pixels 的 3D coords 对 robot control 没用, 因为 robot 关心的是 object 的位置, 不是 random 几何。

**SIMPLER simulation** (Table 3): +10% over SpatialVLA, +56% over OpenVLA

**WidowX real-world** (Figure 4): +10% avg task progress over OpenVLA

---

## 为什么这个工作 important

1. **Scaling path**: 它 demo 了一个新 scaling 路径 — 不是更多 robot data, 而是更多 non-robotic 3D-annotated data。Non-robotic data 便宜无数倍, 几乎无限 scale。

2. **VLM capability enhancement**: 直接 attack 了 VLA 的 root bottleneck — VLM 的 3D understanding deficit。之前大家的 solution 都是 "用更多 robot data implicit 学", 这个工作给了 explicit 的 alternative。

3. **Open & reproducible**: Open weights, open datasets, open annotation pipeline。对比 Gemini Robotics 1.0 (closed, much larger, undisclosed details), SPEAR-1 在 open data 上验证了 3D pretraining 的 isolation effect。

4. **Mathematical elegance**: S³ flow matching for rotations 是 principled treatment, 比 linear flow matching 然后 project 要 sound 得多。

---

## Caveats & Limitations

- **Deformable objects**: MoGe 的 affine invariant depth 对 deformable/复杂 shape 物体不好。Paper 中 tasks 都是 rigid objects (carrot, cup, spoon 等)。
- **Non-metric space**: MoGe 的 depth 是 affine invariant的, 不是真实 metric distance。对 gross motor control (reach, grasp) 够用, 对 precision tasks (insertion) 可能不够。
- **Still needs embodiment fine-tuning**: 不是完全 zero-shot 跨 embodiment, 需要在 target embodiment 上 fine-tune (SPEAR-1 (DROID), SPEAR-1 (Bridge))
- **Scaling laws 没研究**: 3D pretraining data quantity/quality 与 downstream performance 的关系还是 unknown
- **3D VQA tasks 是 hand-designed**: 未来可能用 automatic task discovery

---

## 对你 Andrej 的 takeaways

我知道你一直关注 data efficiency 和 scaling。这个工作直接 address 你的 concerns:

1. **Data efficiency**: 用 cheap non-robotic data 替代 expensive robot data, 20× improvement
2. **Scaling**: 用 3D-annotated internet images scale, 不用 scale robot teleoperation
3. **Foundation model insight**: VLM 的 3D deficit 是 root cause, explicit fix 比 implicit learn 更 efficient

如果你要 build 一个 VLA, 我觉得最 actionable 的 takeaways 是:
- **Image 不要 naive resize**, preserve aspect ratio
- **Object-level 3D supervision** 在 VLM pretraining 有奇效, random 3D 没用
- **Freeze depth encoder 在 VLA training**, 不然 VLA training 会 degrade 它
- **S³ flow matching for rotations**, 不要用 linear
- **Global normalization across datasets**, 不要 per-dataset
- **EMA checkpointing**, 简单但 effective

希望这个 human-readable 版本更能 build 你的 intuition!

---

# SPEAR-1 深度解析: 用 3D 理解超越 Robot Demonstrations 的 Scaling

你好 Andrej! 这篇 paper 让我非常兴奋, 它直击当前 VLA (Vision-Language-Action) models 的一个 fundamental bottleneck: **2D VLM 缺乏 3D spatial reasoning**, 而这个问题之前只能用昂贵的 robot demonstrations 来弥补。SPEAR-1 通过一个 elegant 的 staged training pipeline, 用 non-robotic 3D-annotated images 替代了大量 robot data, 这是一条非常有潜性的 scaling 路径。

---

## 1. Core Thesis & Motivation

### 1.1 The Fundamental Problem

当前 Robotic Foundation Models (RFMs) 的 recipe 基本是: **internet-pretrained VLM + 大规模 robot demonstrations fine-tuning**。这个 recipe 有两个 hidden assumptions:

1. VLM 从 internet 2D image-language data 学到的 semantic priors 足够支撑 robot control
2. Robot demonstrations 可以教模型学会 3D spatial reasoning

SPEAR-1 的作者认为第二个 assumption 是错的: **2D VLMs inherently lack 3D spatial reasoning**, forcing models to implicitly learn 3D structure from expensive robot data. 这是 generalization bottleneck 的根源 — especially in zero-shot Franka (DROID) scenarios with varied camera positions and OOD backgrounds.

### 1.2 The Key Insight

与其用 robot data 来 "teach" 3D understanding (expensive, embodiment-specific, hard to scale), 不如:

1. 用 easy-to-collect non-robotic 2D images (大量存在)
2. Enrich 它们 with 3D annotations (用现成 foundation models 自动标注)
3. Train VLM on 3D-aware VQA tasks (cheap, scalable)
4. 再 fine-tune 到 robot control

这个 insight 的威力在 numbers 中体现: **SPEAR-1 用 ~45M frames 超越/匹配 π0-FAST 和 π0.5, 它们用了 900M+ frames (20× more)**。

参考链接:
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 paper: https://arxiv.org/abs/2504.16054  
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

---

## 2. SPEAR-VLM: 3D-aware Vision-Language Model

### 2.1 Architecture

SPEAR-VLM 是 Stage 1 的产物, 它 extends PaliGemma [4] 通过:

1. **Adding MoGe [47] depth encoder** 作为 supplementary vision backbone
2. **Extending tokenizer with N=1024 3D tokens**

```
┌─────────────────────────────────────────────────────────┐
│                    SPEAR-VLM Architecture                │
├─────────────────────────────────────────────────────────┤
│  Image ──┬──> SigLIP Encoder ──> Linear Projector ──┐   │
│          │                                          │   │
│          └──> MoGe Encoder (last 4 layers concat)    │   │
│                ──> Linear Projector ─────────────────┤   │
│                                                        │
│                          (Average fused features)       │
│                                 │                       │
│                                 ▼                       │
│              ┌──────────────────────────────────┐     │
│              │     Gemma LLM + 1024 new 3D tokens│     │
│              └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

**Why MoGe?** 它的 affine-invariant modeling approach 能 fit cameras with different intrinsics, 这对 cross-environment generalization 至关重要。Depth Anything V2 [48] 在 SpatialVLA [35] 中用过, 但没有 VLM alignment pretraining。

**Encoder Fusion Strategy**: 作者 ablation 了两种方式:
1. Concatenate SigLIP + MoGe (last 4 layers) features, project to LLM space — **这个更好**
2. Add MoGe point cloud to SigLIP features (像 SpatialVLA) — 输出 grammar 不稳定 (e.g., 输出 22/23 tokens 而非 24)

### 2.2 3D Tokenization

这是 paper 中一个 subtle 但关键的 design decision。3D coordinates 与 visual/language tokens 概念上不同, 所以扩展 PaliGemma tokenizer 加入 1024 个 new 3D tokens (类似于 PaliGemma 扩展 Gemma tokenizer 加入 pixel location tokens)。

**Quantization strategy**:
- 距离值近似服从 Normal distribution
- 用 non-uniform bins: mean 附近 fine-grained, tails 处 spread out
- 使 3D token distribution 近似 uniform
- 新 token embeddings 从 multivariate normal distribution 初始化, mean/covariance 来自 pretrained embeddings

参考: John Hewitt's vocab expansion notes https://www.cs.columbia.edu/~johnhew/vocab-expansion.html

### 2.3 VQA Tasks for 3D Pretraining

这是最 core 的创新之一。作者设计了 control-inspired VQA tasks:

| Task Category | Example Question |
|---------------|------------------|
| 3D keypoints | "Output the 3D coordinates of the closest/furthest/center points of object X" |
| 3D bounding box | "Output the vertices of the 3D bounding box of object X" |
| Object-to-object distance | "Output xyz components of distance between object X and Y" |
| Backprojection | "Locate the 3D bounding box vertices on 2D image" |
| Chain-of-thought | "Distance from camera to X? Distance to Y? Which is closer?" |

**重要细节**: 每个 training example 用 1-4 个 question-answer pairs (随机), 对应不同 prompts 和 objects, 这 encourage model "reason" over 信息并 attend to right objects。如果同类 object 有多个 instances, 直接 filter out (避免歧义)。

### 2.4 Semi-automatic Data Annotation Pipeline

这是使 scaling 成为可能的关键:

```
┌─────────────────────────────────────────────────┐
│      Annotation Pipeline (only 2D images needed)│
├─────────────────────────────────────────────────┤
│                                                 │
│  2D Image                                       │
│     │                                           │
│     ▼                                           │
│  Gemini [9] ──> 2D bboxes + semantic labels     │
│     │                                           │
│     ▼                                           │
│  SAM2 [36,37] ──> instance segmentation masks   │
│     │                                           │
│     ▼                                           │
│  MoGe [47] ──> 3D point cloud (affine-invariant)│
│     │                                           │
│     ▼                                           │
│  Filter point cloud by mask                     │
│     │                                           │
│     ▼                                           │
│  Open3D [52]: statistical outlier removal       │
│     │                                           │
│     ▼                                           │
│  Oriented 3D bounding box (consistent vertex    │
│  ordering w.r.t. camera frame)                  │
│     │                                           │
│     ▼                                           │
│  Templated Q&A pair                             │
└─────────────────────────────────────────────────┘
```

**Data sources**:
- EgoExo4D [13] cooking & bike repair: ~200k images (有 GT segmentation masks)
- Bridge-V2 [45]: 30k frames (downsampled to 10% in VLM training)
- Total: ~230k annotated images

**Important nuance**: 用 GroundingDINO [29] 替代 Gemini 时, semantic labels 明显 less accurate 和 consistent。MoGe 在 input image size 840×630 下 resize 以保证 consistent depth scales。

### 2.5 Two-stage Training Process

类似 LLaVa [25]:

**Stage 1 (short, 2k steps)**: 
- 从 PaliGemma + MoGe weights 初始化
- MoGe projector + LLM 3D token embeddings 随机初始化
- 只 train 随机初始化 weights + SigLIP projector
- 其他 frozen

**Stage 2 (long, 10k steps)**:
- 只 freeze SigLIP + MoGe encoders
- 3D token next-token-prediction loss scaled by λ=2
- 18hrs on 16 H200 GPUs

---

## 3. SPEAR-1: Robotic Foundation Model

### 3.1 Overall Architecture

SPEAR-1 构建在 π0 [5] architecture 上, 但用 SPEAR-VLM 初始化:

```
┌──────────────────────────────────────────────────────────┐
│                     SPEAR-1 Architecture                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Image(s) + Language Instruction                         │
│       │                                                  │
│       ▼                                                  │
│  ┌────────────────────────────────────┐                 │
│  │   SPEAR-VLM (with 3D understanding)│                 │
│  │   - SigLIP + MoGe fusion           │                 │
│  │   - Gemma LLM + 3D tokens          │                 │
│  └────────────────────────────────────┘                 │
│       │ intermediate key-value pairs                     │
│       ▼                                                  │
│  ┌────────────────────────────────────┐                 │
│  │   Action Expert (~300M params)     │                 │
│  │   - Gemma-style transformer       │                 │
│  │   - Token size 4096, hidden 4096   │                 │
│  │   - Shared attention with VLM     │                 │
│  └────────────────────────────────────┘                 │
│       │                                                  │
│       ▼                                                  │
│  Action Chunk A_t = [a_t, ..., a_{t+H-1}], H=5          │
│  where a_t = [x_t, q_t, g_t] (translation, rotation,    │
│                               gripper)                   │
└──────────────────────────────────────────────────────────┘
```

**Attention pattern**: Block-wise causal attention over blocks `[I_t, l_t], [p_t], [a_{t+1}, ..., a_{t+H-1}]`。每个 block 内 bidirectional attention, 可以 attend to previous blocks, 不能 attend to future blocks。

### 3.2 Flow Matching Formulation - The Mathematical Core

这是 paper 中最 technical 的部分, 也是最值得深挖的。

#### 3.2.1 Preliminaries

Goal: learn π(·) mapping observation `o_t` to action sequence `A_t = [a_t, ..., a_{t+H-1}]` over horizon H.

**Observation**: `o_t = [I_t^1, ..., I_t^n, p_t, l_t]`
- `I_t^i`: i-th image from uncalibrated camera
- `p_t`: robot state (end-effector pose + gripper state)
- `l_t`: language instruction tokens

**Action decomposition**: `a_t = [x_t, q_t, g_t]`
- `x_t`: translation (R³)
- `q_t`: rotation (unit quaternion, S³)
- `g_t`: gripper (binary)

#### 3.2.2 Training: Noisy Action Construction

Sample timestep `τ ~ Beta(α, β)` and noise:
- `x_ε ~ N(0, I)` for translation
- `q_ε ~ U(S³)` for rotation (uniform on unit quaternion manifold)

**Noisy translation**: linear interpolation
$$x_t^τ = τx_t + (1-τ)x_ε$$

**Noisy rotation**: spherical linear interpolation (SLERP) on S³:
$$q_t^τ = \frac{\sin((1-τ)θ)}{\sin(θ)} q_ε + \frac{\sin(τθ)}{\sin(θ)} q_t \quad (1)$$

**变量解释**:
- `q_t^τ`: noisy quaternion at flow time τ
- `τ ∈ [0,1]`: flow matching timestep (0=noise, 1=clean)
- `θ = cos^(-1)(q_ε · q_t)`: angular distance between noise and target quaternions
- `q_ε`: sampled uniform noise quaternion
- `q_t`: ground truth target quaternion

**关键 insight**: 这个 SLERP formulation 在 S³ manifold 上做 interpolation, 保证了 noisy intermediate states 始终是 valid unit quaternions, 这比在 R^4 上做 linear interpolation 然后 project 到 S³ 要 elegant 得多。

#### 3.2.3 Denoising Vector Field & Losses

Model 预测 denoising vector field `v_θ(A_t^τ, o_t)`, ground truth 是 `u(A_t^τ | A_t) = dA_t^τ/dτ`。

**Translation Loss** (MSE on R³):
$$\mathcal{L}_{\mathbb{R}^3}(\theta) = ||v_\theta(A_t^\tau, o_t)[X_t] - u(A_t^\tau | A_t)[X_t]||^2 \quad (2)$$

**变量解释**:
- `v_θ(...)`: neural network prediction
- `[X_t]`: indexing operator extracting translation component
- `u(...)`: ground truth denoising vector field

**Rotation Loss** 是组合的:

1. **Cosine loss on velocity predictions**:
$$\mathcal{L}_t^{cos}(\theta) = 1 - v_\theta(A_t^\tau, o_t)[q] \cdot u(A_t^\tau | A_t)[q] \quad (8)$$

2. **Geodesic loss on integrated rotation**:
$$\mathcal{L}_t^{geo}(\theta) = \min |q_t^{\tau+\delta} \pm q_{\theta,t}^{\tau+\delta}| \quad (9)$$

其中 `q_{θ,t}^{τ+δ}` 是通过 integrating predicted velocity 得到的 quaternion prediction。

**Geodesic loss 的 derivation**: ground truth denoising vector field for quaternions:
$$u_t(q_t^\tau | q_t) = \frac{\theta}{\sin\theta}[-\cos((1-\tau)\theta) q_\epsilon + \cos(\tau\theta) q_t] \quad (7)$$

**Total rotation loss**:
$$\mathcal{L}_{\mathbb{S}^3}(\theta) = \sum_{k=t}^{t+H} [\mathcal{L}_k^{cos}(\theta) + \mathcal{L}_k^{geo}(\theta)] \quad (10)$$

**Total loss**:
$$\mathcal{L}(\theta) = \mathbb{E}_{p(A_t|o_t), q(A_t^\tau|A_t)} [\mathcal{L}_{\mathbb{R}^3}(\theta) + \mathcal{L}_{\mathbb{S}^3}(\theta)] \quad (3)$$

#### 3.2.4 Quaternion Integration on S³

这是 paper 中一个 subtle 但 critical 的 mathematical detail。

Given current quaternion `q_t ∈ S³` 和它的 time derivative `q̇_t ∈ R^4`:

**Angular velocity**:
$$\omega_t = 2.0 \cdot \text{Im}(q_t^* \otimes \dot{q}_t) \in \mathbb{R}^3$$

其中:
- `q_t^*`: quaternion conjugate
- `⊗`: quaternion multiplication
- `Im(...)`: imaginary part (vector part of quaternion)

**Delta rotation** for small timestep Δt:
- `ω = ω/||ω||`: unit rotation axis
- `Δφ = ||ω||Δt`: rotation angle
- `Δq = [cos(Δφ/2), ω sin(Δφ/2)]` (公式 6)

**Integrated quaternion**:
$$q_{t+\Delta t} = q_t \otimes \Delta q \in S^3$$

**关键**: 所有 quaternions in training/inference 都保持在 same half-space defined by `Re(q) = q_w > 0`, 这解决了 unit quaternion group 的 double coverage 问题 (q 和 -q 表示 same rotation)。

#### 3.2.5 Inference

从 `τ=0` (pure noise) integrate to `τ=1` (clean action):

**Translation** (Euler on R³):
$$x_t^{\tau+\delta} = x_t^\tau + \delta v_\theta^x(A_t^\tau, o_t) \quad (4)$$

**Rotation** (manifold integration on S³):
$$q_t^{\tau+\delta} = q_t^\tau \otimes q_t^\delta(v_\theta^q(A_t^\tau, o_t)) \quad (5)$$

其中 `q_t^δ(...)` 通过公式 (6) 从 predicted velocity 构造的 delta quaternion。

### 3.3 Key Engineering Decisions

#### 3.3.1 Image Resolution

- External camera: **280×210** (4:3 aspect ratio)
- Wrist camera: **112×112**

**Critical**: 不 distort aspect ratio! Naive resizing 会 distort camera intrinsics, negatively 影响 depth 和 point cloud estimates。这是与 OpenVLA [19] 的重要区别。

#### 3.3.2 Vision Encoder Training

Ablation (Table 9) 显示:
- Trainable SigLIP: **72.9%** avg success
- Frozen SigLIP: 56.3%
- Frozen then trainable: 70.8% (comparable but needs extra hyperparameter tuning)
- Lower lr SigLIP: 56.3%

**MoGe**: 在 VLM training 时 trainable, 在 VLA training 时 frozen (Table 1 实验 5 vs 4)。

作者 hypothesis: SigLIP 只 train for image-level semantics, 而 MoGe 已 train for dense depth prediction (更接近 manipulation nature)。

#### 3.3.3 Control Frequency & Data Normalization

- Action chunk size H=5
- Frequency 5Hz
- Non-5Hz datasets: linear interpolation resampling
- **Global quantile normalization** (across entire training mixture) — encourage learning motion across datasets instead of "memorizing" each dataset

Translation normalization ablation (Table 10):
- Min-max const: **65.6%**
- 99th quantile: 61.5%
- Mean-std: 49.0% (significantly worse)

#### 3.3.4 Rotation Representation

Ablation (Table 11) comparing linear vs S³ flow matching:

| Flow Matching | Velocity Loss | Avg Success |
|---------------|---------------|------------|
| linear (R^4 → S³) | MSE | 50.0% |
| linear | cos | 52.1% |
| **S³** | MSE | 57.3% |
| **S³** | **cos** | **59.4%** |

**S³ flow matching consistently outperforms linear**。Half-space unit quaternions 比 Gram-Schmidt orthonormalization on rotation matrices 更 stable。

#### 3.3.5 Reference Frames

- Translation delta: in **robot base frame**
- Rotation delta: in **end-effector frame**
- Gripper: binary action

这个 decomposition 选择很关键, 让 translation 和 rotation 各自在最 natural 的 frame 中学习。

#### 3.3.6 EMA Checkpointing

**Exponential Moving Average** checkpointing significantly stabilizes final checkpoint performance。这是一个 undervalued 的 trick, 在 RL 和 diffusion-style training 中特别有用。

### 3.4 Data Mixture

Open X-Embodiment [32] 24 datasets, 手动 set sampling weights based on:
- Dataset size
- Visual diversity
- Task diversity
- Quality of language annotations

主要 datasets (Table 4):
- DROID: 35.0 weight
- Bridge: 18.0
- Fractal20220817: 12.0
- Kuka: 4.0
- 等等

**Total ~45M frames** vs π0/π0.5 的 900M+ frames。

---

## 4. Experimental Results Deep Dive

### 4.1 3D Ablations - The Crucial Experiment (Table 1)

这是验证 core thesis 的关键 experiment。在 Bridge V2 single environment subset 上训练, 在 SIMPLER WidowX 上 evaluate (inducing distribution shift):

| Exp | VLM Arch | 3D Tasks | VLM SigLIP | VLM MoGe | VLA SigLIP | VLA MoGe | Avg Success |
|-----|---------|----------|------------|----------|-------------|------------|-------------|
| 1 | PaliGemma | None | train | - | train | - | 20.8% |
| 2 | SPEAR-VLM | points only | frozen | frozen | train | frozen | 20.8% |
| 3 | SPEAR-VLM | objects | frozen | frozen | train | frozen | 29.1% |
| 4 | SPEAR-VLM | objects | frozen | frozen | train | **train** | 18.8% |
| 5 | SPEAR-VLM | objects | **train** | **train** | train | frozen | **35.4%** |

**关键 insights**:

1. **Random pixel 3D coordinates 没用** (exp 2 vs 1): 只有 object-level 3D tasks 才有效。这说明 VLM 需要 semantic-grounded 3D understanding, 不是 random geometric supervision。

2. **VLM training 时 trainable encoders 重要** (exp 5 vs 3): 必须 fine-tune SigLIP 和 MoGe 在 VLM stage, 让它们 aligned with 3D tasks。

3. **VLA training 时 frozen MoGe 更好** (exp 5 vs 4): VLA training 会 degrade MoGe 的 representations (验证了 ReVLA [10] 的发现)。

### 4.2 Franka (DROID) 3D Ablations (Table 2)

Real-world 验证, 在 DROID 上从头训练:

| Method | Carrot on Plate (Dist) | (Elev.) | Marker in Cup (Dist) | Avg |
|--------|------------------------|---------|---------------------|-----|
| π0-PaliGemma (DROID) | 0% | 32% | 67% | 34% |
| π0-SPEAR-VLM (DROID) | **42%** | **52%** | 43% | **46%** |

"Carrot on plate" 不在 DROID training set 中, 所以这显示了 SPEAR-VLM 的 generalization 优势 (+12% avg)。

### 4.3 SIMPLER Simulation (Table 3)

| Model | Carrot | Eggplant | Spoon | Stack | Avg |
|-------|--------|----------|-------|-------|-----|
| OpenVLA [19] | 0% | 4.1% | 0% | 0% | 1.0% |
| SpatialVLA [35] | 25.0% | 100.0% | 16.7% | 29.2% | 42.7% |
| **SPEAR-1** | **58.3%** | 62.5% | **62.5%** | **45.8%** | **57.3%** |

**+10% over SpatialVLA**。作者 note: SIMPLER 结果只 indicative of relative performance, 不是 absolute real-world performance。

### 4.4 Real-World Franka - The Main Result (Figure 5)

**这是 paper 的 headline result**。在 5 个 tasks, M=5 initial conditions, N=3 trials (75 trials per model) 上:

**Zero-shot (no target environment fine-tuning)**:
- SPEAR-1 significantly outperforms π0-FAST [33]
- SPEAR-1 matches π0.5 [6]

**Data efficiency**:
- SPEAR-1: ~45M frames
- π0-FAST / π0.5: 900M+ frames
- **20× less robotic data**

### 4.5 Real-World WidowX (Figure 4)

5 tasks, M=4, N=3 (60 trials per model):
- SPEAR-1: ~10% higher avg task progress than OpenVLA

Note: 无法与 π0/π0.5 对比因为没公开 WidowX weights。

### 4.6 The "Carrot on Plate" Generalization Insight

在 Franka experiments 中, "Carrot on Plate" 是 OOD task (不在 DROID training data)。

- π0-PaliGemma (DROID): 0% / 32% (dist / elev.)
- π0-SPEAR-VLM (DROID): 42% / 52%

SPEAR-VLM 在 OOD task 上明显 better, 说明 3D pretraining 提供的不仅仅是 in-distribution 优势, 而是 true generalization capability。

---

## 5. Critical Analysis & Limitations

### 5.1 Strengths

1. **Elegant insight**: 用 cheap 3D annotations 替代 expensive robot data
2. **Strong empirical results**: 20× data efficiency, matching SOTA
3. **Open weights and datasets**: 真正 reproducible
4. **Principled S³ flow matching**: 数学上 sound 的 rotation handling
5. **Comprehensive ablations**: 验证了每个 design choice

### 5.2 Limitations (作者承认的)

1. **Deformable/complex objects**: MoGe 的 affine-invariant depth 对这些 object 不友好
2. **Non-metric coordinates**: 3D bounding box labels 不在 metric space, 对 fine-grained manipulation 可能有限制
3. **Scaling laws 未探索**: 3D pretraining data quantity/quality 与 downstream performance 的关系未研究
4. **Still needs embodiment fine-tuning**: 不能完全 zero-shot 跨 embodiment, 需 fine-tune on target embodiment

### 5.3 我的 Critical Thoughts

**关于 affine-invariant depth**: 这是 trade-off。Affine-invariant 让 cross-environment generalization 更好, 但失去 metric information。对于 gross motor control (reach, grasp) 够用, 但对 precision tasks (insertion, assembly) 可能不够。Future work 可以 explore hybrid: 用 metric depth estimator 但 normalize per-scene。

**关于 VQA task design**: 这些 3D VQA tasks 是 hand-designed。如果能让 model 自动 discover 控制-relevant 3D tasks (通过 RL 或 self-supervised learning), 可能更 scalable。

**关于 EMA checkpointing**: 这个 trick 在 paper 中只有一句 mention, 但实际可能很 critical。VLA training variance 是社区中 known issue, EMA 是一个简单的 mitigation。建议关注这个方向。

**关于 Rotation Representation**: S³ flow matching > linear flow matching 是 expected result, 但 ablation 中 cos loss > MSE for velocity prediction 是 interesting finding。可能因为 cos loss 在 S³ 上更 natural (与 geodesic distance aligned)。

---

## 6. Broader Context & Related Work

### 6.1 VLM 3D Understanding 历史

- **SpatialVLM [8]** (Chen et al.): 类似 data annotation approach, 但没 integrate pretrained depth estimator, 未公开
- **RoboSpatial [39]**: 高级 spatial relationships, 不是 explicit 3D coordinates
- **SpatialBot [7]**: spatially-aware VLM, 但 multi-step inference, too slow for real-time control

### 6.2 VLA Models 对比

| Model | Key Feature | 3D Awareness |
|-------|-------------|--------------|
| OpenVLA [19] | Open VLA, 2D VLM | None |
| SpatialVLA [35] | MoGe encoder, no VLM pretraining | Learned from robot data |
| MolmoAct [20] | Spatial reasoning at inference | Too slow |
| π0 [5] | Flow matching action expert | None |
| π0-FAST [33] | Action tokenization | None |
| π0.5 [6] | Open-world generalization, 900M+ data | None explicit |
| **SPEAR-1** | **3D VLM pretraining** | **Pretrained on non-robotic 3D data** |

### 6.3 Most Closely Related: Gemini Robotics 1.0 [44]

Gemini Robotics 1.0 也用 3D pretraining fine-tune significantly larger Gemini 2.0 [34], 然后 distill 到 smaller VLA with reasoning capabilities。但:
1. 大部分 method details 未公开
2. Significantly larger model
3. 在 less diverse open data 上未验证

SPEAR-1 在 isolation 中验证了 3D pretraining benefits, 用小 model 和 open data。

### 6.4 Bridge V2 vs DROID for Generalization (Appendix A.4)

这是 paper 中一个 undervalued 的 insight。作者 argue:

- **Bridge V2 + WidowX**: 多数 VLA works 在这里 evaluate zero-shot, 但 environment 不 diverse (toy kitchen), camera viewpoints 限制, WidowX payload/reach 限制
- **DROID + Franka**: significantly more diverse, real-world environments, 20× more unique scenes, camera viewpoints vary, Franka 更 capable

**Only a handful of works** [6, 33] 成功在 DROID 上做 zero-shot Franka control without target environment fine-tuning。所以 SPEAR-1 在 DROID 上的 results 是 significantly stronger claim than 在 Bridge V2 上。

参考: RoboArena [2] https://arxiv.org/abs/2502.05029

---

## 7. Practical Implications & Future Directions

### 7.1 For Practitioners

如果你要 train 一个 VLA, key takeaways:

1. **Don't naively resize images** — preserve aspect ratio
2. **Use object-level 3D supervision** in VLM pretraining — random pixel 3D coords 没用
3. **Freeze MoGe during VLA training** — 让它 retain 3D representations
4. **Use S³ flow matching for rotations** — 显著优于 linear
5. **Use global normalization across datasets** — encourage cross-dataset knowledge sharing
6. **Use EMA checkpointing** — stabilize final performance

### 7.2 For Researchers

Open research questions:

1. **3D pretraining scaling laws**: How does 3D data quantity/quality affect downstream performance?
2. **Metric depth integration**: Can SOTA metric depth estimators resolve affine-invariant limitation?
3. **Deformable objects**: 如何 capture geometry of deformable objects?
4. **Zero-shot cross-embodiment**: 如何 alleviate need for target embodiment fine-tuning?
5. **Automatic 3D task discovery**: 让 model 自动 discover control-relevant 3D tasks

### 7.3 Connection to Broader Trends

SPEAR-1 验证了一个重要 hypothesis: **enhancing VLM capabilities with non-robotic embodied knowledge is scalable**。这与以下 trends 一致:

1. **Foundation models 在 robotics**: 用更大 pretrain, less fine-tune
2. **Multi-modal pretraining**: 3D 作为新 modality
3. **Data efficiency in robotics**: robot data expensive, leverage other data sources

参考:
- Foundation Models for Robotics: https://arxiv.org/abs/2504.16054
- RT-2: https://arxiv.org/abs/2307.15818

---

## 8. Final Thoughts

SPEAR-1 是一个 important step toward data-efficient generalist robot policies。它的核心 contribution 不是 architecture innovation, 而是 **training strategy insight**: 用 cheap 3D-annotated non-robotic data 来 inject 3D understanding into VLM, 这比用 expensive robot data implicitly learn 3D 更 scalable。

最 exciting 的是这个 approach 的 scaling potential: 用更多 non-robotic 3D-annotated images, 不需要更多 robot demonstrations。如果 future work 验证了 scaling laws, 这可能是 robotic foundation model 训练 paradigm 的一个重要 shift。

对于你 Andrej, 我知道你一直关注 data efficiency 和 scaling, 这个 paper 直接 address 这两个 concerns。Flow matching on S³ manifold 也是 elegant 的数学 treatment, 值得深入理解。

---

## References (Key Links)

- **SPEAR-1 Paper**: (要找官方 link)
- **π0**: https://arxiv.org/abs/2410.24164
- **π0.5**: https://arxiv.org/abs/2504.16054
- **PaliGemma**: https://arxiv.org/abs/2407.07726
- **MoGe**: https://arxiv.org/abs/2410.19115
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **DROID**: https://arxiv.org/abs/2403.12945
- **Bridge V2**: https://arxiv.org/abs/2308.12952
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **SpatialVLA**: https://arxiv.org/abs/2501.15830
- **SAM2**: https://arxiv.org/abs/2408.00714
- **EgoExo4D**: https://arxiv.org/abs/2311.18258
- **SpatialVLM**: https://arxiv.org/abs/2401.02370
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **SIMPLER**: https://arxiv.org/abs/2405.05941
- **ReVLA**: https://arxiv.org/abs/2409.15250
- **Gemini Robotics**: https://arxiv.org/abs/2503.20020
- **LLaVA**: https://arxiv.org/abs/2304.08485
- **SigLIP**: https://arxiv.org/abs/2303.15343
- **Gemma**: https://arxiv.org/abs/2403.08295
- **Flow Matching Guide**: https://arxiv.org/abs/2412.06264
- **RoboArena**: https://arxiv.org/abs/2502.05029
- **π0-FAST**: https://arxiv.org/abs/2501.09747
- **Depth Anything V2**: https://arxiv.org/abs/2406.09414
- **GroundingDINO**: https://arxiv.org/abs/2303.05499
- **Learning with 3D Rotations**: https://arxiv.org/abs/2402.12159

---

希望这个 deep dive 帮你 build intuition about SPEAR-1。如果你想 discuss 任何 specific aspect (e.g., quaternion integration math, ablation interpretations, 或 future directions), 我很乐意 go deeper!
