---
source_pdf: 3D and 4D World Modeling A Survey.pdf
paper_sha256: 2be3ba2b3545cb4af11fabd83af25101c3e09615c3d0429a876d9fe13e87984b
processed_at: '2026-07-17T09:17:30-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 《3D and 4D World Modeling: A Survey》深度解读

## 1. Paper 整体定位与动机

这篇由 WorldBench Team 撰写的 survey 是**首个**专门聚焦 native 3D 和 4D world modeling 的系统性综述。Lingdong Kong 等人(主要来自 National University of Singapore、Salesforce、Shanghai Jiao Tong University 等)在 2025 年推出此 work，目的是清理一个长期混乱的术语生态。

核心痛点在于: "world model" 这个词在文献里被滥用到几乎失去意义。有人用 GAIA-1 这种 sensory data 生成模型，有人把 video diffusion 笼统叫 world model，还有人把 predictive forecasting、simulators、decision-making frameworks 全部塞进去。Yann LeCun 原本关于 world model 的定义 (https://openreview.net/forum?id=BZTBaMY2zD) 强调的是 joint embedding predictive architecture (JEPA) 那种 latent-space 自监督预测，但工业界把这个概念无限外延了。这份 survey 想做的事情就是给 native 3D/4D world modeling 一个**精确的**定义和层次化 taxonomy。

- Project Page: https://worldbench.github.io/survey
- GitHub Repo: https://github.com/worldbench/awesome-3d-4d-world-models

**为什么 native 3D/4D 关键?** 这是这篇 survey 的核心 thesis。2D projections (image pixels) 把 metric geometry、visibility、motion 全部折叠成 RGB tensor，丢失了物理作用所依赖的坐标信息。Native 3D/4D signals (RGB-D、occupancy grids、LiDAR point clouds) 在物理作用发生的坐标系中编码这些信息，是 actionable modeling 的一等公民。它们提供 multi-view consistency、egocentric consistency、rigid-body kinematics、scene-scale occlusion reasoning、map/topology adherence 这些 inductive bias。在 safety-critical 系统 (autonomous driving、robotics) 中，agents 仅仅生成 photorealistic frames 是远远不够的，必须遵守 geometry、causality、controllability。

这一点可以从图 1 的整体框架中直观感受到: 整个 taxonomy 围绕三类 native 3D/4D representation 展开 (VideoGen、OccGen、LiDARGen)，由三类 conditions ($\mathcal{C}_{geo}$、$\mathcal{C}_{act}$、$\mathcal{C}_{sem}$) 引导，分成 generative 和 predictive 两个 paradigm，再细化为四种 functional types。

---

## 2. Preliminaries: 数学表示与定义

### 2.1 三类核心 3D/4D 表示

**(1) Video Streams**
$$\mathbf{x}_v \in \mathbb{R}^{T \times H \times W \times C}$$

变量含义:
- $T$ = 帧数 (temporal extent)
- $H, W$ = 单帧图像的高度和宽度
- $C$ = 通道数 (通常 3 表示 RGB)

这里的关键 insight 是: 在 3D/4D world modeling 语境下，video 不仅仅是 2D 帧的时序堆叠，而是要求 geometric coherence (跨视角几何一致) 和 temporal consistency (时间连贯)，这是与一般 video generation 工作 (Sora、Stable Video Diffusion) 的根本区别。

**(2) Occupancy Grids**
$$\mathbf{x}_o \in \{0,1\}^{X \times Y \times Z}$$

变量含义:
- $X, Y, Z$ = 三个空间维度上的 voxel 数量
- 每个 voxel 取值 0 或 1，表示该位置是否被占据

时序扩展为 4D:
$$\mathbf{x}_o^t \in \{0,1\}^{T \times X \times Y \times Z}$$
- 上标 $t$ 表示这是时间序列形式
- $T$ = 时间步数

Voxelized geometry 强制 spatial constraints，对 physics-consistent scene generation 友好。实际 driving 场景下，常见配置是 $(50, 50, 8)$ 到 $(100, 100, 16)$ 这种 coarse-to-fine 的 resolution (见 Table 9)。

**(3) LiDAR Point Clouds**
$$\mathbf{x}_l = \{(x_i, y_i, z_i)\}_{i=1}^N$$

变量含义:
- $(x_i, y_i, z_i)$ = 第 $i$ 个点的 3D Cartesian 坐标
- $N$ = 单帧点数 (通常 1-10 万)

时序 LiDAR:
$$\mathbf{x}_l^t = \{(x_i, y_i, z_i, t_i)\}_{i=1}^{N_t}$$
- $t_i$ = 第 $i$ 个点的捕获时间戳
- $N_t$ = 时刻 $t$ 捕获的点数

LiDAR 直接捕获几何，对 texture、lighting、weather variations 鲁棒，所以 weather-robust 3D perception 任务都依赖它。

**(4) Neural Representations**

- **NeRF** (Neural Radiance Fields): 一个 MLP $f_\theta$ 把 $(r, d) \mapsto (c, \sigma)$
  - $r$ = ray origin (相机位置)
  - $d$ = ray direction
  - $c$ = 颜色 (RGB)
  - $\sigma$ = 体密度

- **Gaussian Splatting (GS)**: 场景表示为一组 Gaussian primitives，每个 primitive 由 position、covariance、color 参数化
  - 与 NeRF 的隐式表示相比，GS 是 explicit primitive，渲染快得多
  - 时序扩展到 4D Gaussians 支持动态场景重建

### 2.2 三类 Conditions 的形式化

Table 1 列出了一个非常丰富的 condition 集合。我把它压缩成三类:

**Geometric conditions** $\mathcal{C}_{geo}$ — 描述 spatial layout:
- Camera pose: 6-DoF 位姿
- Depth map: 每像素深度
- BEV map: 鸟瞰图几何
- HD map: 高分辨率道路语义地图
- 3D bounding box: 物体 3D box
- Flow field: 光流 / scene flow
- Past occupancy: 历史占据
- LiDAR pattern: 传感器扫描配置 (beam count、FOV、resolution)
- Object coordinate、Partial point cloud、RGB frame、Surface mesh

**Action conditions** $\mathcal{C}_{act}$ — 描述 agent motion:
- Ego-trajectory、Ego-velocity、Ego-acceleration、Ego-steering
- Ego-command、Route plan、Action token、Scan path

**Semantic conditions** $\mathcal{C}_{sem}$ — 描述 abstract scene intent:
- Semantic mask、Text prompt、Scene graph、Object label、Weather tag、Material tag

把 conditions 和 functions 解耦是这篇 survey 的关键设计决策。文献里经常混淆 "what the model consumes" (conditions) 和 "what the model does" (function)，导致比较困难。

### 2.3 Generative vs Predictive World Models

**(1) Generative World Models**:
$$\mathcal{G}(\mathbf{x}_i, \mathcal{C}_{geo}, \mathcal{C}_{act}, \mathcal{C}_{sem}) \rightarrow S_g \tag{1}$$

变量含义:
- $\mathbf{x}_i$ = 可选输入表示，下标 $i \in \{\varnothing, v, o, l\}$
  - $\varnothing$ = 无输入 (纯噪声)
  - $v$ = partial video
  - $o$ = partial occupancy
  - $l$ = partial LiDAR
- $S_g$ = 生成的 3D/4D scene (可以是 video sequence、occupancy grid 或 LiDAR sweep sequence)

**(2) Predictive World Models**:
$$\mathcal{P}(\mathbf{x}_i^{-t:0}, \mathcal{C}_{act}) \rightarrow S_p^{1:k} \tag{2}$$

变量含义:
- $\mathbf{x}_i^{-t:0}$ = 从过去 $t$ 步到当前 step 0 的历史观察序列
- $\mathcal{C}_{act}$ = agent actions (control commands 或 planned trajectories)
- $S_p^{1:k}$ = 未来 $k$ 步的预测 scene 表示

这两个 paradigm 抓住了 world model 的 dual capability: **imagine** diverse and controllable worlds (generative)，**anticipate** plausible future evolution under specific conditions (predictive)。这种 formulation 与 Danijar Hafner 在 Dreamer 系列里的想法 (https://danijar.com/project/dreamer/) 在精神上一致，但这里强调的是 native 3D/4D representation。

### 2.4 四类 Functional Types

| Type | Input | Output | 核心目标 |
|------|-------|--------|---------|
| **Data Engines** | $\mathcal{C}_{geo}, \mathcal{C}_{act}$ (opt), $\mathcal{C}_{sem}$ | $S_g$ | plausibility + diversity, 用于大规模数据增强 |
| **Action Interpreters** | $\mathbf{x}_i^{-t:0}, \mathcal{C}_{act}$ | $S_p^{1:k}$ | action-aware forecasting |
| **Neural Simulators** | $S_a^t, \pi_{agent}$ | $S_g^{t+1}$ | closed-loop agent-environment interaction |
| **Scene Reconstructors** | $\mathbf{x}_i^p, \mathcal{C}_{geo}$ (opt) | $\hat{S}_g$ | 从 partial observations 恢复完整 scene |

这里 $\pi_{agent}$ 是 agent policy，$S_a^t$ 是当前 scene state，$\mathbf{x}_i^p$ 是 partial observation。Neural Simulators 是闭环的，前两个是开环的，第四个是补全任务。

### 2.5 四个生成式家族的数学

**(1) Variational Autoencoders (VAEs)**

encoder 定义 variational posterior:
$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \text{diag}(\sigma_\phi^2(\mathbf{x})))$$

- $\mu_\phi(\mathbf{x})$ = encoder 输出的 latent mean
- $\sigma_\phi^2(\mathbf{x})$ = encoder 输出的 latent variance (diagonal covariance)
- $\phi$ = encoder 参数

reparameterization trick:
$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \epsilon \sim \mathcal{N}(0, I)$$
- $\odot$ = element-wise multiplication
- 这个 trick 让梯度可以 backprop 通过采样过程

训练目标 (ELBO):
$$\log p_\theta(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularization}} \tag{3}$$

- $\theta$ = decoder 参数
- $p(\mathbf{z})$ = latent prior，通常 $\mathcal{N}(0, I)$
- 第一项鼓励重建，第二项让 posterior 接近 prior

VAEs 训练稳定，latent space 可解释，但容易生成模糊样本。3D world modeling 里 VAE 用于 occupancy 压缩 (如 UniScene 的 VAE encoder)，把 voxel grid 压到 low-dim latent。

**(2) Generative Adversarial Networks (GANs)**

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))] \tag{4}$$

- $G$ = generator (参数 $\theta$)
- $D$ = discriminator (参数 $\phi$)
- $p_{data}$ = 真实数据分布
- $p(\mathbf{z})$ = latent prior

DUSty (IROS'21) 就是 GAN-based 的 LiDAR 合成方法，explicitly disentangle depth map 和 measurement uncertainty。GAN 高保真但训练不稳定、mode collapse，3D 数据上更难。

**(3) Diffusion Models (DMs)**

Forward (noising) process:
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

- $\beta_t$ = 第 $t$ 步的 noise variance (variance schedule)
- $\sqrt{1-\beta_t}\mathbf{x}_{t-1}$ = signal 的衰减部分 (保持 scale)
- $\beta_t \mathbf{I}$ = 加的 Gaussian noise 协方差

训练损失:
$$\mathbb{E}_{\mathbf{x}, \epsilon, t}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2] \tag{5}$$

- $\epsilon \sim \mathcal{N}(0, I)$ = 真实添加的噪声
- $\epsilon_\theta(\mathbf{x}_t, t)$ = 神经网络预测的 noise
- $t$ = 随机采样的 diffusion timestep

DMs 是当前 3D/4D world modeling 的主流。R2DM (https://arxiv.org/abs/2312.08012) 首次把 DDPM 应用到 LiDAR range image generation，后续 R2Flow 用 flow matching (https://arxiv.org/abs/2210.02747) 加速。

**(4) Autoregressive Models (ARs)**

$$p(\mathbf{x}) = \prod_{i=1}^n p(x_i | x_{<i})$$

- $x_{<i}$ = 前 $i-1$ 个 token
- 因子分解让 joint distribution 变成条件分布的乘积

Transformer-based ARs 提供 exact likelihood，序列建模灵活，但生成是顺序的，慢。Copilot4D (https://arxiv.org/abs/2405.00380) 把 MaskGIT 改造成 discrete diffusion，做并行的 denoising 和 decoding，缓解了顺序生成的瓶颈。DrivingGPT、DrivingWorld 用 GPT-style 的 next-token prediction 统一建模 visual 和 action tokens。

---

## 3. 三大 Modality Tracks 深度解析

### 3.1 VideoGen: World Modeling from Video Generation

这是 driving world model 里最热闹的方向。Figure 4 把 VideoGen 模型按功能分成 Data Engines、Action Interpreters、Neural Simulators 三类。

#### 3.1.1 Data Engines — 三大 application

**(a) Perception Data Augmentation**

时间线很清晰。早期工作 BEVGen (RA-L'24, https://ieeexplore.ieee.org/document/10384657) 用 autoregressive transformer + cross-view transformation 从 BEV layout 生成 spatially consistent surrounding images。BEVControl (https://arxiv.org/abs/2308.01661) 用 diffusion model 提升 synthetic data 质量，尤其针对 long-tail scenarios。

MagicDrive (ICLR'24, https://arxiv.org/abs/2310.02699) 是个里程碑，把 3D geometry、semantic descriptions、camera poses 联合 conditioning 生成高保真 multi-view 图像。后续 SyntheOcc (https://arxiv.org/abs/2410.00337) 用 3D semantic multi-plane images (MPIs) 做 spatially aligned conditioning，PerLDiff (ICCV'25, https://arxiv.org/abs/2411.10203) 用 perspective 3D geometry 增强 realism 和 consistency。

时序方向: Panacea (CVPR'24)、DrivingDiffusion (ECCV'24)、SubjectDrive (AAAI'25) 引入 4D attention、keyframes、subject control 改善 temporal consistency。NoiseController (https://arxiv.org/abs/2504.18448) 提出 multi-level noise decomposition + multi-frame collaborative denoising。

Long-horizon video 是个开放问题。DiVE (https://arxiv.org/abs/2409.01595)、MagicDrive-V2 (ICCV'25)、Cosmos-Drive (https://arxiv.org/abs/2501.03575) 用 DiT (Diffusion Transformer, Peebles & Xie, ICCV'23, https://arxiv.org/abs/2212.09748) 的 flexibility 和 scalability 生成长视频。Glad (ICLR'25, https://arxiv.org/abs/2412.01046) 用 latent-variable propagation，STAGE (https://arxiv.org/abs/2506.13138) 用 hierarchical temporal feature transfer 做 streaming 式长视频生成。

**(b) Planning-Oriented Data Mining**

Delphi (https://arxiv.org/abs/2406.01349) 用 diffusion-based long video generation + failure-case-driven approach + pre-trained VLM 合成 failure scenarios 类似的数据，提升 end-to-end planning 的 sample efficiency。DriveDreamer-2 (AAAI'25, https://arxiv.org/abs/2402.12248) 用 LLM 把 user queries 转成 agent trajectories，再生成 traffic-compliant HDMaps 做 corner case generation。Nexus (https://arxiv.org/abs/2504.10485) 从 fine-grained tokens + independent noise states 模拟 regular 和 challenging scenarios。Challenger (https://arxiv.org/abs/2505.15880) 用 physics-aware multi-round trajectory refinement 识别 adversarial maneuvers。

**(c) Scene Editing & Style Transfer**

WoVoGen (ECCV'24, https://arxiv.org/abs/2403.14002) 通过 world volume-aware synthesis 保证 cross-sensor consistency。SyntheOcc 用 occupancy grids 做 occlusion-aware scene editing。SimGen (NeurIPS'24, https://arxiv.org/abs/2406.18440) 通过 simulator-conditioned cascade diffusion 桥接 sim-to-real gap。DrivePhysica (https://arxiv.org/abs/2412.08410) 用 CARLA 模拟 cut-in 等复杂场景，引入 motion representation learning 和 instance flow guidance。GeoDrive (https://arxiv.org/abs/2505.22421) 集成 explicit 3D geometry conditions + dynamic editing，支持 interactive trajectory 和 object manipulation。

#### 3.1.2 Action Interpreters

**(a) Action-Guided Video Generation**

GAIA-1 (https://arxiv.org/abs/2309.17080) 是 Wayve 的开创性工作，把 video、text、action 三类输入 fusion 起来合成 driving scenarios，精细控制 ego-vehicle 行为和 scene attributes。GAIA-2 (https://arxiv.org/abs/2503.20523) 扩展到 agent configurations、environmental factors、road semantics。

GenAD (CVPR'24, https://arxiv.org/abs/2403.09630) 释放 OpenDV dataset (YouTube 上 2139 scenes、60M frames)，支持 zero-shot、language- 和 action-conditioned predictions。Vista (NeurIPS'24, https://arxiv.org/abs/2405.17358) 在多场景下做 robust action conditioning。GEM (CVPR'25, https://arxiv.org/abs/2406.14825) 给出 multimodal outputs + 精确 ego-motion control。MaskGWM (CVPR'25, https://arxiv.org/abs/2405.12148) 用 mask-based diffusion 提升 fidelity 和 long-horizon predictions。

长视频合成有 error accumulation 问题。InfinityDrive (https://arxiv.org/abs/2412.01522) 用 memory injection，Epona (ICCV'25, https://arxiv.org/abs/2503.06115) 用 chain-of-forward training strategy。

**(b) Forecasting-Driven Action Planning**

这条线让 planner 在 commit 之前 virtual 地 "test" 多个 futures，避免真实世界的 unsafe trial-and-error。Drive-WM (CVPR'24, https://arxiv.org/abs/2311.13584) 生成 candidate maneuvers 的 video rollouts，用 image-based rewards 评分选 trajectory。DriveDreamer (ECCV'24, https://arxiv.org/abs/2309.09777) 提出 ActionFormer 预测未来状态和 ego-environment 交互。ADriver-I (https://arxiv.org/abs/2311.13549) 结合多模态 LLMs 和 autoregressive control signals。

GPT-style 设计: DrivingGPT (https://arxiv.org/abs/2412.18607)、DrivingWorld (https://arxiv.org/abs/2412.19505) 联合建模 visual 和 action tokens 做 next-token prediction planning。Doe-1 (https://arxiv.org/abs/2412.09627) 统一 perception、prediction、planning 做闭环自动驾驶。

#### 3.1.3 Neural Simulators

**(a) Generation-Driven Simulation**

DriveArena (ICCV'25, https://arxiv.org/abs/2408.02090) 是首个 closed-loop framework，包含两个核心组件:
- TrafficManager: scalable traffic synthesis
- WorldDreamer: autoregressive scene generation

DreamForge (https://arxiv.org/abs/2409.04003) 通过 object-wise position encoding + novel temporal attention 增强 long-term scenario modeling。DrivingSphere (CVPR'25, https://arxiv.org/abs/2503.04504) 引入 4D semantic occupancy modeling，unify static environments + dynamic objects，外加 visual synthesis module 保证 multi-view video generation 的 spatiotemporal consistency。

**(b) Reconstruction-Centric Simulation**

这条线用 NeRF (https://arxiv.org/abs/2003.08934) 和 3D Gaussian Splatting (https://arxiv.org/abs/2308.16425) 把 driving logs 转成 interactive neural environments。StreetGaussian (ECCV'24, https://arxiv.org/abs/2406.07134) 把动态街道表示为 point clouds + semantic logits + 3D Gaussians 的集合，每个 Gaussian 关联到 foreground vehicle 或 background。

HUGSIM (https://arxiv.org/abs/2412.01718) 集成 physical constraints 和 3D GS 做 aggressive behavior synthesis。OmniRe (ICLR'25, https://arxiv.org/abs/2410.17360) 通过 neural scene graph 增强动态实体建模。ReconDreamer (CVPR'25, https://arxiv.org/abs/2411.00748) 用 progressive refinement 消除 dynamic scenes 的 ghosting artifacts。Stage-1 (https://arxiv.org/abs/2412.05280) 通过 multi-view point cloud completion 做 controllable 4D synthesis。

Table 2 列出了 63 个 VideoGen 模型，覆盖了从 GAIA-1 到 STAGE 的整个生态。值得一提的是 dataset 维度上，nuScenes 是绝对主流，OpenDV-YouTube 因为有 caption 越来越重要。

### 3.2 OccGen: World Modeling from Occupancy Generation

Occupancy grid 是 geometry-centric 的表示，编码 3D 世界的语义和结构细节。Figure 5 把 OccGen 分成 Scene Representors、Occupancy Forecasters、Autoregressive Simulators 三类。

#### 3.2.1 Scene Representors

**(a) 3D Perception Robustness Enhancement**

SSD (https://arxiv.org/abs/2301.00527) 开创了用 discrete diffusion (Austin et al., NeurIPS'21, https://arxiv.org/abs/2107.02806) 和 latent diffusion (Rombach et al., CVPR'22, https://arxiv.org/abs/2112.10752) 做 scene-level 3D categorical data generation，把 sparse occupancy inputs 映射到 dense semantic reconstructions。SemCity (CVPR'24, https://arxiv.org/abs/2403.05437) 进一步通过在 initial SSC outputs 上 conditioning diffusion，提升 geometric 和 semantic fidelity。

**(b) Generation Consistency Guidance**

WoVoGen 用 4D temporal occupancy volumes 驱动 multi-view video generation，保证 intra-world 和 inter-sensor consistency。UrbanDif (https://arxiv.org/abs/2403.11697) 用 semantic occupancy grids 作为 3D-aware image synthesis 的 geometric priors。DrivingSphere 把动态 4D occupancy 转 temporally consistent video。UniScene (CVPR'25, https://arxiv.org/abs/2503.04232) 用 Gaussian-based rendering + prior-guided sparse modeling 统一 video 和 LiDAR synthesis。

#### 3.2.2 Occupancy Forecasters

**(a) Predictive Model Pretraining**

Emergent-Occ (https://arxiv.org/abs/2205.11430, https://arxiv.org/abs/2306.13060) 引入 differentiable rendering 从 4D occupancy predictions 重建 point clouds，实现 self-supervised training from raw sequences。UnO (CVPR'24, https://arxiv.org/abs/2405.15274) 建 continuous 4D occupancy field 做 joint perception 和 forecasting。

DriveWorld (CVPR'24, https://arxiv.org/abs/2312.06347) 是个关键 work，用 4D pre-training 做 scene understanding，可微调到 detection 和 planning，减少对 dense labels 的依赖。UniWorld (https://arxiv.org/abs/2308.07234) 类似。

**(b) Ego-Conditioned Occupancy Forecasting**

OccWorld (ECCV'24, https://arxiv.org/abs/2401.09530) 联合建模 ego motion 和周围环境 evolution 在 3D occupancy 空间里。OccSora (https://arxiv.org/abs/2405.20337) 生成 trajectory conditioned 4D occupancy 在长 horizon 上 (16 秒)。

Cam4DOcc (CVPR'24, https://arxiv.org/abs/2403.12534) 是 vision-centric 的 pipeline，把 world model 集成到 end-to-end planning。OccLLaMA (https://arxiv.org/abs/2409.03272) 和 Occ-LLM (ICRA'25, https://arxiv.org/abs/2412.10437) 用 semantic occupancy 作为共享 representation 统一 vision、language、action modalities，支持 embodied question answering。UniOcc (https://arxiv.org/abs/2503.24381) 建立 benchmark 结合真实和模拟数据做标准化评估。

#### 3.2.3 Autoregressive Simulators

**(a) Scalable Open-World Generation**

PDD (ECCV'24, https://arxiv.org/abs/2311.12085) 提出 scale-varied diffusion framework，progressively 从 coarse layouts 到 fine details 生成 outdoor scenes。XCube (CVPR'24, https://arxiv.org/abs/2312.03890) 用 hierarchical voxel-based latent diffusion 做 multi-resolution generation。

InfiniCube (ICCV'25, https://arxiv.org/abs/2411.00749) 和 X-Scene (https://arxiv.org/abs/2506.13558) 集成 voxel-based occupancy 和 consistent visual synthesis 做 realistic、editable simulation worlds。

**(b) Long-Horizon Dynamic Simulation**

OccSora 在 16-second horizons 上做 trajectory-conditioned generation。DynamicCity (ICLR'25, https://arxiv.org/abs/2409.08860) 支持 layout-aware 和 command-conditioned generation。DrivingSphere 构造由 static backgrounds + dynamic objects 组成的 4D world 做 closed-loop simulation。UniScene 生成 layout-conditioned 4D occupancy with rich semantic 和 geometric detail。

### 3.3 LiDARGen: World Modeling from LiDAR Generation

LiDAR-based 生成模型提供 geometry-aware 和 appearance-invariant 的表示。Figure 6 把 LiDARGen 分成 Data Engines、Action Forecasters、Autoregressive Simulators 三类。

#### 3.3.1 Data Engines

**(a) Perception Data Augmentation**

DUSty (IROS'21, https://arxiv.org/abs/2108.04017) 是 GAN-based 的，通过 explicitly disentangling underlying depth map 和 measurement uncertainty 合成 realistic LiDAR scans。DUSty v2 (WACV'23, https://arxiv.org/abs/2210.06534) 集成 implicit neural representations，支持任意分辨率生成。

LiDARGen (ECCV'22, https://arxiv.org/abs/2207.08631) 是开创性的，首次用 Langevin dynamics 做 LiDAR point cloud generation，性能超过 GANs 和 VAEs。R2DM (ICRA'24, https://arxiv.org/abs/2312.08012) 是首个把 denoising diffusion paradigm 应用到这个 domain 的工作 (DDPM-based)，用 explicit positional encoding 达到更高精度。R2Flow (ICRA'25, https://arxiv.org/abs/2403.16080) 用 flow matching (Lipman et al., ICLR'23, https://arxiv.org/abs/2210.02747) 大幅加速生成。

LiDM (CVPR'24, https://arxiv.org/abs/2312.02945)、RangeLDM (ECCV'24, https://arxiv.org/abs/2404.18655)、3DiSS (https://arxiv.org/abs/2503.21449) 采用 latent diffusion，先用 pretrained VAE 把 raw-scale 数据压缩到 low-dim latent，再在 latent space 训练 diffusion model，最后 reconstruct 到 original resolution，大幅加速。

LiDARGRIT (https://arxiv.org/abs/2404.05505) 用 VQ-VAE (van den Oord et al., NeurIPS'17, https://arxiv.org/abs/1711.00937) discretize latent space，用 autoregressive transformer 生成 latent codes，外加 raydrop estimation loss 显式增强 raydrop noise 建模。

SDS (https://arxiv.org/abs/2410.11628) 提出 simultaneous diffusion sampling 做 multi-view LiDAR scene generation，所有 views 一起生成，比单独生成每个 view 的几何一致性好得多。SPIRAL (NeurIPS'25, https://arxiv.org/abs/2505.22643) 开创了生成 segmentation-labeled LiDAR data，引入 closed-loop inference strategy 增强 geometry 和 semantics 之间的一致性。

La La LiDAR (AAAI'26, https://arxiv.org/abs/2508.03691) 提出 layout-guided generative framework，集成 scene graph-based layout diffusion + foreground-aware control injector，explicitly 建模 object relations。Veila (https://arxiv.org/abs/2508.03690) 用 monocular RGB image 引导 panoramic LiDAR generation。

**(b) Scene Completion**

UltraLiDAR (CVPR'23, https://arxiv.org/abs/2304.11164) 用 discrete voxel-based VQ-VAE 做 sparse-to-dense completion，efficient 且 controllable。LiDiff (CVPR'24, https://arxiv.org/abs/2310.10348) 和 DiffSSC (https://arxiv.org/abs/2409.18092) 用 DDPM 的 denoising process 重新放置 duplicated points，densifying LiDAR point cloud 同时完成 occluded areas。

LiDAR-EDIT (ICRA'25, https://arxiv.org/abs/2503.14754) 集成 UltraLiDAR (background) 和 AnchorFormer (CVPR'23, https://arxiv.org/abs/2211.13342) (foreground)，支持 flexible LiDAR scene editing 包括 object removal 和 insertion。LiDPM (IV'25, https://arxiv.org/abs/2501.03863) 扩展 LiDiff，能从 sparse inputs 和 pure Gaussian noise 生成 dense point clouds，合成 entirely novel scenes。

SuperPC (CVPR'25, https://arxiv.org/abs/2503.14558) 提出统一框架，把 point clouds 转成 representation features 适合 completion、upsampling、denoising、colorization，避免 sequentially 应用 separate models 导致的 error accumulation。

**(c) Rare Condition Modeling**

Text2LiDAR (ECCV'24, https://arxiv.org/abs/2404.18507) 是 Transformer-based 架构，集成 textual information 做 text-controlled LiDAR point cloud generation。WeatherGen (CVPR'25, https://arxiv.org/abs/2405.19072) 针对 rainy、snowy、foggy 条件，用 Mamba (Gu & Dao, https://arxiv.org/abs/2312.00752) 作为 backbone，统一 controllable generative model。OLiDM (AAAI'25, https://arxiv.org/abs/2411.04037) 通过两阶段 pipeline 解决 object-level fidelity: 先生成 foreground objects，再用它们作为 conditions 生成 scene。LOGen (https://arxiv.org/abs/2412.07385) 提出 object-level point cloud generation model 合成 traffic participants，conditioned on 相对传感器的 orientation 和 distance。

**(d) Multimodal Generation**

X-Drive (ICLR'25, https://arxiv.org/abs/2411.15271) 是 dual-branch diffusion 架构，jointly 生成 aligned LiDAR point clouds 和 multi-view camera images，关键创新是 cross-modality epipolar condition module。HoloDrive (https://arxiv.org/abs/2412.01407) 也是 jointly 生成 multi-view camera 和 LiDAR，在 2D generative model 中引入 depth prediction branch 改善 2D-3D alignment。

#### 3.3.2 Action Forecasters

**(a) Temporal Modeling**

Copilot4D (ICLR'24, https://arxiv.org/abs/2405.00380) 是这里的关键 work，提出 scalable approach:
1. 用 VQ-VAE tokenize complex、unstructured point cloud inputs
2. 把 MaskGIT (Chang et al., CVPR'22, https://arxiv.org/abs/2202.04200) 改造成 discrete diffusion model，使能 parallel denoising 和 decoding

输入 1-3 秒的过去 LiDAR frames + 未来 ego actions (poses)，预测未来 1-3 秒的高保真 LiDAR frames。Vi-DAR (CVPR'24, https://arxiv.org/abs/2402.01968) 接受历史 camera frames 输入，预测未来 LiDAR frames，enable pre-training for perception、prediction、planning。

**(b) Multi-Modal Action Forecasters**

BEV-World (https://arxiv.org/abs/2407.05679) 引入 multi-modal tokenizer 把生成能力扩展到 surround-view images + LiDAR point clouds。DriveX (https://arxiv.org/abs/2505.19239) 支持 multi-modal outputs (point clouds、camera images、semantic maps)，用 decoupled latent world modeling strategy 把 world representation learning (spatial) 和 latent future decoding (future state prediction) 分开。HERMES (ICCV'25, https://arxiv.org/abs/2504.08206) 集成 LLMs 生成 textual descriptions of future frames 之外加 LiDAR，增强人机交互。

#### 3.3.3 Autoregressive Simulators

**(a) Sequential Autoregressive LiDAR Generation**

HoloDrive jointly 生成 multi-view camera + LiDAR。LiDARCrafter (AAAI'26, https://arxiv.org/abs/2508.03692) 把 La La LiDAR 的 layout-based two-stage framework 扩展到 4D domain，autoregressive LiDAR sequence generator 支持 fine-grained control、long-term temporal coherence、diverse editing。

**(b) Scene-Scale Simulation from Meshes**

LidarDM (https://arxiv.org/abs/2404.02903) 构造方法很巧妙:
1. 从多帧 point clouds 移除 dynamic objects 构造 mesh grids
2. 训 diffusion model conditioned on BEV layout，生成 mesh world
3. 把 dynamic objects with motion trajectories incorporate 到 mesh world
4. 通过 ray projection 合成 long sequential LiDAR point clouds

这种基于 mesh 的物理仿真保证了 dynamic feasibility，对 closed-loop simulation 友好。

Table 4 列出了 34 个 LiDARGen 模型，从 DUSty 到 LiDARCrafter 覆盖 5 年发展。架构从 GAN、Enc-Dec、LDM、AR 到 DiT 演进。

---

## 4. Datasets 深度解读

Table 5 是数据集统计表，关键 dataset:

| Dataset | Venue | Scenes | Frames | Occupancy | LiDAR | Freq | 关键 Conditions |
|---------|-------|--------|--------|-----------|-------|------|----------------|
| KITTI (https://www.cvlibs.net/datasets/kitti/) | CVPR'12 | 22 | 15k×4 | - | 15k | 10 Hz | D, 3D, F |
| nuScenes (https://www.nuscenes.org/) | CVPR'20 | 1000 | 1.4M×6 | 40k | 400k | 2 Hz | 3D, B, H, T, V, S |
| Waymo Open (https://waymo.com/open/) | CVPR'20 | 1150 | 1M×5 | - | 230k | 10 Hz | 3D, B, H, T, V, S |
| SemanticKITTI (http://www.semantic-kitti.org/) | ICCV'19 | 22 | - | 43k | 23k | 10 Hz | S, T |
| Argoverse 2 (https://www.argoverse.org/) | NeurIPS'21 | 1000 | 2.7M×9 | - | 150k | 10 Hz | 3D, T, H |
| OpenDV-YouTube (https://github.com/OpenDriveLab/DriveWorld/) | CVPR'24 | 2139 | 60M×1 | - | - | 10 Hz | T, C |
| NAVSIM (https://github.com/autonomousvision/navsim) | NeurIPS'24 | 115k | 920k×8 | - | 115k | 2 Hz | 3D, T, H |
| Occ3D-nuScenes (https://tsinghua-mars-lab.github.io/Occ3D/) | NeurIPS'23 | 900 | 240k×6 | 40k | 40k | 2 Hz | 3D, T, S |
| OpenOccupancy (https://github.com/JeffWang98/OpenOccupancy) | ICCV'23 | 850 | 200k×6 | 34k | 34k | 2 Hz | 3D, T, S |
| KITTI-360 (http://www.cvlibs.net/datasets/kitti-360/) | TPAMI'22 | 379 | 150k×4 | - | 80k | 10 Hz | 3D, L, P, T, S, H |

几个值得注意的 pattern:
- **nuScenes** 是绝对主流 benchmark，几乎所有 VideoGen、OccGen 方法都在它上面 report
- **OpenDV-YouTube** 用 web-scale YouTube 视频 + caption，trade calibration for scale 和 diversity，对 generalization 重要
- **Waymo Open** 10 Hz 帧率，比 nuScenes 的 2 Hz 高 5 倍，对时序建模更友好
- **Synthetic datasets** (CARLA, CarlaSC) 提供 clean ground truth 和完全可控 layout，对 ablation 和 stress test 不可替代

---

## 5. Evaluation Metrics 全景

Table 14 是个超长的 metrics 总览。我把核心 metrics 按类别梳理:

### 5.1 Generation Quality

**Perceptual Fidelity** ( Fréchet family):
- **FID** (Fréchet Inception Distance, Heusel et al., NeurIPS'17, https://arxiv.org/abs/1706.08500): 在 Inception features 上拟合 Gaussians，算 Fréchet distance (Wasserstein-2 between two multivariate Gaussians)
  $$d^2 = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$
  - $\mu_r, \mu_g$ = real / generated features 的均值
  - $\Sigma_r, \Sigma_g$ = real / generated features 的协方差矩阵
- **FVD** (Fréchet Video Distance, Unterthiner et al., https://arxiv.org/abs/1812.01717): 用 I3D features (Carreira & Zisserman, https://arxiv.org/abs/1705.07750)，capturing temporal coherence
- **FRD** (Fréchet Range Distance, Nakashima & Kurazume, https://arxiv.org/abs/2312.08012): RangeNet++ features (Milioto et al., IROS'19, https://arxiv.org/abs/1908.09801) on LiDAR range images
- **FPD** (Fréchet Point Cloud Distance): PointNet features (https://arxiv.org/abs/1612.00593) on raw 3D point clouds
- **FSVD, FPVD**: LiDM (https://arxiv.org/abs/2312.02945) 引入，sparse voxel 和 hybrid point-voxel 表示
- **F3D**: 在 occupancy grid features 上算，用于 volumetric realism
- **S-FRD, S-FPD**: SPIRAL (https://arxiv.org/abs/2505.22643) 引入 semantic-aware variants
- **KID** (Kernel Inception Distance, Binkowski et al., https://arxiv.org/abs/1801.01401): MMD with polynomial kernel，unbiased
- **IS** (Inception Score, Salimans et al., https://arxiv.org/abs/1606.03498): 不需要 real reference data，奖励 confident 和 diverse class predictions

**Statistical Fidelity**:
- **JSD** (Jensen-Shannon Divergence): symmetric divergence between occupancy histograms
- **MMD** (Minimum Matching Distance): 每个 real sample 到最近 generated neighbor 的平均 Chamfer distance
- **COV** (Coverage): 至少被一个 generated output 匹配的 real samples 比例
- **1-NNA** (1-Nearest-Neighbor Accuracy): 50% 表示两个分布等价

**Spatial Consistency**:
- **VCS** (View Consistency Score, BEVGen): LoFTR (Sun et al., CVPR'21, https://arxiv.org/abs/2104.00680) keypoint confidences 在 overlapping views 上的求和
- **KPM** (Key Points Matching, Drive-WM)
- **DAS** (Depth Alignment Score, X-Drive): 投影 point clouds vs estimated monocular depth (Depth Anything V2, https://arxiv.org/abs/2406.09414) 的统计差异

**Temporal Consistency**:
- **CTC** (CLIP Temporal Consistency, DiVE): 连续帧 CLIP features (https://arxiv.org/abs/2103.00020) 的 cosine similarity
- **DTC** (DINO Temporal Consistency): 用 DINOv2 (https://arxiv.org/abs/2304.07193)
- **TTCE** (Temporal Transformation Consistency Error, LiDARCrafter): 时序上 generated 和 ground-truth point clouds 之间的 registration error
- **CTC** (Chamfer Temporal Consistency, LiDARCrafter): 不同 timestamp 间 generated point clouds 的 Chamfer distance
- **ICP Energy Outlier**: ICP alignment residuals 和 outlier ratios

**Subject Consistency**:
- **SC** (Subject Consistency, VBench, https://arxiv.org/abs/2311.17913): subject-region features 的 cosine similarity
- **FDC** (Foreground Detection Confidence, LiDARCrafter): 用 pretrained detector 检测 generated samples 中 foreground objects 的置信度

**Controllability**:
- **CDA** (Conditional Detection Accuracy): 用 pretrained 3D detector 在 generated point clouds 上算 detection accuracy (with box conditioning)
- **CLIP-Sim**: generated 和 reference frames 的 CLIP embeddings 的 cosine similarity
- **RotErr, TransErr** (PosePilot, https://arxiv.org/abs/2505.01729): camera trajectory rotation 和 translation 误差

**Human Preference**:
- **VQ** (Visual Quality, 2AFC, LongDWM, https://arxiv.org/abs/2506.01546): two-alternative forced choice 的 win rates
- **MR** (Motion Rationality, 2AFC): 运动合理性的 win rates
- **DMOS** (Differential Mean Opinion Score)

### 5.2 Forecasting Quality

**Spatial Predictive Accuracy**:
- **L1 Error, L2 Error**: frame-level pixel 或 depth 距离
- **IoUc, IoUf, IoUwf** (Cam4DOcc): current / future / weighted-future timestamp 的 IoU
- **CD** (Chamfer Distance, Copilot4D): bidirectional nearest-neighbor distance between point clouds
- **L1 Med, AbsRelMed, L1 Mean, AbsRelMean**: 沿 LiDAR rays 的 depth error 统计

**Temporal Predictive Accuracy**:
- **KODP** (Key Object Dimension Probability, UniOcc): 用 category priors 惩罚 implausible object dimensions
- **TFSC** (Temporal Foreground Shape Consistency): voxel-level IoU of dynamic objects across frames
- **TBEC** (Temporal Background Environment Consistency): 在 ego-motion compensation 下 static voxels 的一致性

### 5.3 Planning-Centric Quality

**Open-Loop Planning**:
- **ADE, FDE** (GenAD, https://arxiv.org/abs/2403.09630): average / final displacement error
- **SLE, SALE** (ADriver-I): speed / steering angle L1 error
- **CR** (Collision Rate, Drive-WM): collision 比例
- **PDMS** (Predictive Driver Model Score, NAVSIM, https://arxiv.org/abs/2406.02504): aggregate score combining progress、spacing、comfort，discard unsafe rollouts
- **AHE, FHE, MR** (nuPlan): average / final heading error，miss rate

**Closed-Loop Planning**:
- **SR, ID** (CARLA): success rate，infraction distance
- **ADS** (Arena Driving Score, DriveArena)
- **NAC** (No At-Fault Collisions, nuPlan)
- **DAC, DDC** (Drivable-Area / Driving-Direction Compliance, nuPlan)
- **TTC** (Time-to-Collision), **PAR** (Progress Along Route), **SLC** (Speed-Limit Compliance), **Comfort**

### 5.4 Reconstruction-Centric Quality

- **PSNR** (https://ieeexplore.ieee.org/document/4514586), **SSIM** (Wang et al., https://ieeexplore.ieee.org/document/1284395), **LPIPS** (Zhang et al., https://arxiv.org/abs/1801.03924)
- **NTA-IoU, NTL-IoU** (DriveDreamer4D, https://arxiv.org/abs/2410.22903): novel viewpoint 下 foreground agents 和 lane 的 IoU

### 5.5 Downstream Evaluation

- **Detection**: mAP、mAP-3D、LET-3D-AP、NDS (nuScenes Detection Score，复合 mAP + mATE + mASE + mAOE + mAVE + mAAE)
- **Segmentation**: mIoU、BEV-Map-IoU
- **Tracking**: MOTA、MOTP、3D-AMOTA、3D-AMOTP
- **Occupancy Prediction**: Occupancy-IoU、VPQ (Voxelized Panoptic Quality)
- **VQA**: Top-1 Acc (OccLLaMA)

---

## 6. Quantitative Benchmarks 关键发现

### 6.1 VideoGen (Table 6)

**Single-View Video Generation on nuScenes**:
- DriveDreamer (ECCV'24): FID 14.90, FVD 340.80
- GenAD (CVPR'24): FID 15.40, FVD 184.00
- Vista (NeurIPS'24): FID 6.90, FVD 89.40 @ 576×1024 @ 10Hz
- MaskGWM (CVPR'25): FID 4.00, FVD 59.40 @ 288×512 @ 10Hz — **SOTA**
- GeoDrive (arXiv'25): FID 4.10, FVD 61.60 @ 480×720 @ 12Hz

**Multi-View Video Generation on nuScenes**:
- BEVControl (arXiv'23): FID 24.85 (high)
- Drive-WM (CVPR'24): FID 15.80, FVD 122.70 @ 192×384
- MagicDrive (ICLR'24): FID 16.20, FVD 218.12 @ 224×400 @ 12Hz
- DriveScape (CVPR'25, https://arxiv.org/abs/2501.07457): FID 8.34, FVD 76.39 @ 576×1024 @ 10Hz
- DiST-4D (ICCV'25, https://arxiv.org/abs/2503.08821): FID 6.83, FVD 22.67 @ 424×800 @ 12Hz — **SOTA**
- UniScene (CVPR'25): FID 6.45, FVD 71.94 @ 256×512 @ 12Hz

Insight: resolution 和 frame rate 强烈影响 generation fidelity。explicit multi-view modeling 很困难，许多方法降低 FID 但 temporal coherence 仍难。DiST-4D 用 disentangled spatiotemporal diffusion + metric depth 显式 enforce spatial structure 和 temporal consistency，达到 FVD 22.67 这种惊人的低值。

### 6.2 VideoGen Downstream (Table 7, 8)

- Real nuScenes baseline (UniAD, https://arxiv.org/abs/2212.10156): 37.98 mAP, 49.85 NDS, 25.93 lane mIoU
- MagicDrive: 12.92 NDS, 28.36 lane, 51.46 drivable mIoU
- DrivingSphere: 21.45 mAP, 22.29 NDS, 34.16 lane, 57.99 drivable
- DreamForge: 16.63 mAP, 16.61 NDS, 30.57 lane, 58.98 drivable
- DrivePhysica (https://arxiv.org/abs/2412.08410): 35.50 mAP, 43.67 NDS (StreamPETR)
- SubjectDrive (AAAI'25, https://arxiv.org/abs/2410.08437): 28.00 mAP, 41.10 NDS (StreamPETR)
- DrivingSphere (BEVFusion): 22.71 mAP, 31.79 NDS — **生成数据上的 SOTA**

Insight: photorealistic generation 单独不够，必须 explicit modeling of geometry、temporal consistency、motion dynamics。生成数据上的检测性能 (22.71 mAP) 距离真实数据 (37.98 mAP) 仍有 ~40% gap，这是当前 driving world model 的核心瓶颈。

### 6.3 OccGen Reconstruction (Table 9)

| Method | Type | Resolution | mIoU ↑ | IoU ↑ |
|--------|------|------------|--------|-------|
| OccSora (https://arxiv.org/abs/2405.20337) | VQVAE | (×, 25, 25, 512) | 27.40 | 37.00 |
| OccLLaMA (https://arxiv.org/abs/2409.03272) | VQVAE | (50, 50, 128) | 65.93 | 57.66 |
| OccWorld (https://arxiv.org/abs/2401.09530) | VQVAE | (50, 50, 128) | 66.38 | 62.29 |
| UrbanDif (https://arxiv.org/abs/2403.11697) | VQVAE | (50, 50, 2048) | 80.00 | 98.80 |
| I2World (https://arxiv.org/abs/2507.09144) | VQVAE | (50, 50, 128) | 81.22 | 68.30 |
| DOME (https://arxiv.org/abs/2410.10429) | VAE | (25, 25, 64) | 83.08 | 77.25 |
| UniScene | VAE | (100, 100, 8) | 92.10 | 87.00 |
| T3Former (https://arxiv.org/abs/2503.07338) | Triplane-VAE | (100, 100, 16, 8) | 85.50 | 72.07 |
| **X-Scene (https://arxiv.org/abs/2506.13558)** | **Triplane-VAE** | (100, 100, 16, 8) | **92.40** | **85.60** |

Insight: latent representation design 决定 reconstruction fidelity。Triplane factorization (T3Former、X-Scene) 通过把 3D 体素分解成三组 2D planes，enforce geometric consistency，enable finer spatial detail。简单地增大 latent dimensionality (如 UrbanDif 的 2048 channels) 收益有限。OccSora aggressive compression (×, 25, 25, 512) 严重牺牲精度。

### 6.4 OccGen 4D Forecasting (Table 10)

| Method | mIoU@1s | mIoU@2s | mIoU@3s | IoU@1s | IoU@2s | IoU@3s |
|--------|---------|---------|---------|--------|--------|--------|
| GaussianAD (https://arxiv.org/abs/2412.10371) | 6.29 | 5.36 | 4.58 | 14.13 | 14.09 | 14.04 |
| OccWorld | 25.78 | 15.14 | 10.51 | 34.63 | 25.07 | 20.18 |
| OccLLaMA | 25.05 | 19.49 | 15.26 | 34.56 | 28.53 | 24.41 |
| DOME | 35.11 | 25.89 | 20.29 | 43.99 | 35.36 | 29.74 |
| UniScene | 35.37 | 29.59 | 25.08 | 38.34 | 32.70 | 29.09 |
| T3Former | 46.32 | 33.23 | 28.73 | 77.00 | 75.89 | 76.32 |
| **I2World** | **47.62** | **38.58** | **32.98** | 54.29 | 49.43 | 45.69 |

Insight: naive autoregressive 或 generative approaches 在 longer horizons 上迅速 deteriorate (OccWorld 从 25.78 → 10.51)，凸显 structured priors 的必要性。Triplane factorization 大幅提升 spatial fidelity (T3Former IoU@1s 77.00)。I2World 通过 intra-inter tokenization 平衡 spatial 和 temporal。

### 6.5 OccGen Planning (Table 11)

| Method | L2@1s ↓ | L2@2s ↓ | L2@3s ↓ | CR@1s ↓ | CR@2s ↓ | CR@3s ↓ |
|--------|---------|---------|---------|---------|---------|---------|
| ST-P3 (ECCV'22, https://arxiv.org/abs/2207.07605) | 1.33 | 2.11 | 2.90 | 0.23 | 0.62 | 1.27 |
| UniAD (CVPR'23, https://arxiv.org/abs/2212.10156) | 0.48 | 0.96 | 1.65 | 0.05 | 0.17 | 0.71 |
| OccWorld | 0.43 | 1.08 | 1.99 | 0.07 | 0.38 | 1.35 |
| GenAD | 0.36 | 0.83 | 1.55 | 0.06 | 0.23 | 1.00 |
| T3Former | 0.32 | 0.91 | 1.76 | 0.08 | 0.32 | 0.51 |
| Drive-OccWorld (https://arxiv.org/abs/2404.00698) | 0.32 | 0.75 | 1.49 | 0.05 | 0.17 | 0.64 |
| **Occ-LLM** | **0.12** | **0.24** | **0.49** | - | - | - |

Insight: 集成 occupancy world models 到 planning pipeline 系统性超越 pure trajectory-based 方法。Hybrid designs 如 Drive-OccWorld、DFIT-OccWorld 在 accuracy 和 safety 上同时提升。Occ-LLM 报告了惊人的 0.12m L2 error，可能因为 LLM 引入了 language grounding 和 common sense reasoning。

### 6.6 LiDARGen Fidelity (Table 12)

| Method | Resolution | FRD ↓ | FPD ↓ | JSD ↓ | MMD ↓ |
|--------|------------|-------|-------|-------|-------|
| LiDARGen (https://arxiv.org/abs/2207.08631) | 64×1024 | 681.37 | 115.17 | 0.1323 | 2.19e-3 |
| LiDM | 64×1024 | - | 108.70 | 0.0456 | 2.90e-4 |
| R2DM | 64×1024 | 192.81 | 19.29 | 0.0373 | 1.60e-4 |
| Text2LiDAR | 64×1024 | 522.32 | 11.09 | 0.0750 | 4.29e-4 |
| **WeatherGen** | **64×1024** | **184.11** | **11.42** | **0.0290** | **3.80e-5** |

Insight: WeatherGen 用 Mamba backbone 在所有 metrics 上达到 SOTA，说明 state space models 对 LiDAR 时序建模有优势。Text2LiDAR 虽然强 textual conditioning，但 FRD 反而高 (522.32 vs R2DM 的 192.81)，提示 aligning with semantic prompts 可能 compromise geometric fidelity，这是个重要 trade-off。

### 6.7 LiDARGen 4D Temporal (Table 13)

| Method | TTCE@3 ↓ | TTCE@4 ↓ | CTC@1 ↓ | CTC@2 ↓ | CTC@3 ↓ | CTC@4 ↓ |
|--------|----------|----------|---------|---------|---------|---------|
| UniScene | 2.74 | 3.69 | 0.90 | 1.84 | 3.64 | 3.90 |
| OpenDWM (https://github.com/SenseTime-FVG/OpenDWM) | 2.68 | 3.65 | 1.02 | 2.02 | 3.37 | 5.05 |
| OpenDWM-DiT | 2.71 | 3.66 | **0.89** | **1.79** | **3.06** | **4.64** |
| **LiDARCrafter** | **2.65** | **3.56** | 1.12 | 2.38 | 3.02 | 4.81 |

Insight: end-to-end autoregressive methods (UniScene, OpenDWM-DiT) 在 short-horizon (1-2 frames) 几何一致性上有优势，但 fixed-length generation 限制了 broader applicability，long horizons 上 error accumulation 增长。BEV-based generation 时序连续性好但牺牲了 raw point cloud pattern fidelity，range-based (LiDARCrafter) 更好保留 LiDAR-specific sensing characteristics 但需要 careful design embed conditions 和 sustain long-term consistency。

---

## 7. Qualitative Insights (Section 4.4)

### 7.1 VideoGen

Figure 7, 8 对比 MagicDrive、DreamForge、DriveDreamer-2、OpenDWM。OpenDWM 在 OpenDV + nuScenes + Waymo Open 多数据集上训练，达到最 realistic、consistent、controllable 的结果，**underscores the role of dataset diversity in improving generalization**。

常见问题:
- **Physical Plausibility**: 没有 explicit physics constraints 时，generated videos 出现 vehicle-background interpenetration、incorrect shadows、scale distortions。连续视频时这些 subtle 问题显著降低 realism。
- **Controllability**: appearance-level controls (weather、time-of-day、style) 可以通过 text conditioning 可靠控制。**precise geometric control** over object position、orientation、velocity 仍很难，typically 需要 dedicated control embeddings 或 structured conditioning。
- **Long-Tail Categories**: rare 和 small-scale classes (pedestrians、cyclists、traffic signs) 难以 convincingly 生成。Long-tail data imbalance 导致 unrealistic shapes、distorted geometry、甚至 omission。

### 7.2 OccGen

Figure 9 展示 X-Scene 的 conditioned generation 结果。生成 videos 和 occupancies 展示 strong spatial alignment across perspectives，cross-view coherence 对 multi-camera settings 关键。

挑战:
- **Fine-Grained Geometry**: thin lane boundaries、small dynamic agents 经常 misalignment 或 incomplete reconstruction
- **Generalization**: rare structures 和 long-tail categories (bicycles、pedestrians) poorly represented

### 7.3 LiDARGen

Figure 10 对比 OpenDWM (voxel-based)、UniScene (occupancy-based)、LiDARCrafter (range-based):

- **OpenDWM** (voxel): 强调 coherent scene geometry，但 voxel-level modeling 产生 overly regularized patterns
- **LiDARCrafter** (range): 更好保留 native scanline structure with sharper rings，occlusion boundaries 可能引入 artifacts
- **UniScene** (occupancy): 重现 global distributions 但 oversmooth fine details，leading to discontinuities

关键 attributes:
1. **Global patterns**: coherent scene geometry + sensor-specific scan structures
2. **Point sparsity**: realistic density distributions 匹配 LiDAR characteristics
3. **Object completeness**: 准确 capturing dynamic agents with sharp contours 和 consistent surfaces

---

## 8. Applications 拓展

### 8.1 Autonomous Driving

- **Traffic Simulation**: heterogeneous agents、diverse motion、physics-compliant interactions。Volumetric representations (occupancy grids、multi-frame LiDAR、scene-level meshes) 比 image-only platforms 提供更丰富 geometry 和 temporal coherence
- **Closed-Loop Driving Evaluation**: couple generative models 和 agents，assess perception→planning→control stacks over long horizons。Modular conditioning (HD maps、text queries、ego trajectories) enables targeted stress testing
- **Scenario Synthesis**: 生成 rare 或 safety-critical scenes (severe occlusions、sudden intrusions、multi-agent conflicts、adverse weather)。Physics- 和 motion-aware models ensure dynamic feasibility

### 8.2 Robotics

- **Embodied Navigation**: world models 预测 dynamic layouts，enable long-horizon exploration、obstacle avoidance、localization
- **Object-Centric Manipulation**: capture object geometry 和 physical transitions，anticipate contact dynamics，plan stable grasps
- **Scene Generation for Simulation**: diverse synthetic environments 减少手动设计成本，improve robustness 和 sim-to-real transfer

### 8.3 Video Games & XR

- **Procedural World Generation**: automates expansive virtual worlds design，支持 open exploration 和 emergent gameplay
- **Interactive Scene Rendering**: NeRF 和 Gaussian Splatting advance photorealistic synthesis，temporal extensions model motion 和 state change
- **Playable Environment Adaptation**: adaptive worlds 调整 geometry、layout、agent behavior 维持 challenge 和 engagement

### 8.4 Digital Twins

- **City-Scale Scene Modeling**: multimodal sensing (LiDAR、RGB-D、aerial photogrammetry、drone surveys)
- **Event Replay & Forecasting**: 从 sparse sensor logs 重建 past 或 hypothetical events
- **Scene Control & Editing**: vehicle removal、weather alteration、layout modification

### 8.5 Emerging Applications

- **Scientific Discovery**: climate/weather prediction、glacier retreat monitoring、wildfire spread simulation (FourCastNet, https://arxiv.org/abs/2202.11214; GraphCast, https://arxiv.org/abs/2212.12794)
- **Healthcare**: anatomy deformation、tissue behavior、rehabilitation、prosthetics
- **Industrial**: virtual prototyping、robotic assembly、material handling
- **Security & Disaster Response**: tactical operations simulation、structural collapse prediction、fire spread、chemical dispersion

---

## 9. Challenges & Future Directions

### 9.1 Standardized Benchmarking & Evaluations

不同 paper 用不同 datasets 和 ad hoc metrics，难以 fair comparison。需要 unified benchmarks 涵盖 physical plausibility、temporal consistency、controllability，且包含 closed-loop simulation tests 和 real-world scenarios。

### 9.2 High-Fidelity & Long-Horizon Generation

Short-term predictions 准确，但 small errors 在 longer sequences 上 accumulate，导致 unrealistic behaviors 和 scene consistency degradation。Dynamic urban environments 中 multi-agent 和 environmental factors 持续 evolve，compound 这个 challenge。需要 advanced generative techniques 探索 novel training paradigms 和 memory mechanisms，effectively penalize long-term divergences。

### 9.3 Physical Fidelity、Controllability & Generalizability

- **Physical Realism**: 当前 world models 产生 physically implausible events (non-deforming collision impacts、objects lacking temporal consistency)
- **Controllability**: editing capabilities 仍然 coarse，confined to adjusting traffic agents' positions 或 appearances，neglecting granular control over environmental elements
- **Generalizability**: tend to overfit training data，failing to generalize to new urban environments 和 rare objects

### 9.4 Computational Efficiency & Real-Time Performance

Heavy architectures + multi-step sampling strategies 导致 substantial latency 和 memory overhead，undermines practicality for large-scale data generation 和 simulation。需要 sparse computation 和 inference acceleration techniques (sub-linear time complexity, https://arxiv.org/abs/2406.12039)。

### 9.5 Cross-Modal Generation Coherence

Visual、geometric、semantic modalities 必须 jointly interact 形成 coherent representation。Misalignment 导致 generated imagery 与 underlying 3D structure 冲突，undermines reliability in downstream perception 和 planning tasks。需要 integrated architectures jointly learn from diverse sensor data while enforcing strict consistency constraints during generation。

---

## 10. 我的整体 Intuition 与思考

读完这篇 survey，我构建的 mental model 是:

**World model 的本质**: 是一个 learnable simulator，能够 **imagine** 多种 plausible futures (generative)，**anticipate** 给定 actions 下的 future evolution (predictive)，同时 maintain geometry、causality、controllability 这些 actionable constraints。

**2D vs 3D/4D 的本质区别**: 2D video diffusion 模型 (如 Sora) 在 pixel space 学到了 appearance manifold，但缺乏 metric geometry 的 inductive bias。3D/4D 表示把信号放在 physical 作用发生的坐标系中，提供了 hard constraints (multi-view consistency、rigid body kinematics、occlusion reasoning) 让模型无法 cheat。所以 driving world model 必须 native 3D/4D，**这是这篇 survey 的核心 thesis**。

**Generative vs Predictive 的 dual capability**: 这两个 paradigm 实际上对应了人类认知的两种模式 — 一种是基于 prior 和 condition 的 imagination (像做梦、像 what-if 推理)，一种是给定当前 state 和 action 的 forward simulation (像物理仿真、像 model-based RL)。理想 world model 应该 unify 这两种能力。

**四种 Functional Types 的层次**: 
- Data Engines (无历史 observation，从 conditions 生成) — 最基础
- Action Interpreters (有历史 + action conditions) — 增加 temporal context
- Neural Simulators (closed-loop + policy-in-the-loop) — 增加 interaction
- Scene Reconstructors (从 partial 观察补全) — 处理 missing information

这个 hierarchy 实际上对应了 difficulty 递增，从 unconditional generation → conditional generation → forecasting → closed-loop simulation → completion。

**Triplane factorization 是当前 SOTA 的关键 insight**: T3Former 和 X-Scene 在 OccGen 上的成功表明，把 3D voxel grid 分解成三组 2D planes (XY、XZ、YZ) 在数学上类似 CNN 的 weight sharing，但显式 enforce 了 geometric consistency。这让我联想到 NeRF 的 positional encoding 也是用 low-dim functions 表达 high-dim signals。

**Diffusion 之外，Autoregressive + Discrete Tokens 在崛起**: Copilot4D 用 VQ-VAE tokenize + discrete diffusion 把 MaskGIT 改造成 parallel denoising，DrivingGPT 用 next-token prediction 统一 visual 和 action tokens。这暗示着 LLM 的 scaling law 可能 transfer 到 3D world modeling — 当我们有足够大的 tokenizer 和 sequence model，3D world modeling 可能也变成 next-token prediction 问题。Genie 2 (DeepMind, https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model) 和 Cosmos (NVIDIA, https://arxiv.org/abs/2501.03575) 都在验证这个方向。

**LiDAR vs Video 的 trade-off**: LiDAR 直接捕获 geometry，对 appearance 变化鲁棒；video 包含丰富 semantic 和 texture，但缺乏 metric geometry。理想的 driving world model 应该 multi-modal aligned (X-Drive、HoloDrive、UniScene 都是这条路)。Veila 用 monocular RGB 引导 panoramic LiDAR generation 是个有趣的 cross-modal 实验。

**Open Challenges 的优先级排序** (我自己的判断):
1. **Cross-modal alignment** — video 和 LiDAR 之间的一致性是部署的硬约束
2. **Physical fidelity** — non-deforming collisions 这种问题在 closed-loop driving 中是 deal-breaker
3. **Long-horizon coherence** — 当前 ~3s horizon 远不够，driving 至少需要 10s+ 才能做 real planning
4. **Generalization** — rare scenarios (construction zones、emergency vehicles、unusual weather) 是 deployment 的 long tail
5. **Real-time performance** — DiT-based 方法 inference latency 高，部署需要 distillation 或 flow matching 加速

**与 LeCun JEPA 的联系**: LeCun 的 JEPA (https://openreview.net/forum?id=BZTBaMY2zD) 强调在 latent space 做 predictive learning，avoid pixel-level reconstruction。这篇 survey 里的 Emergent-Occ、UnO、UniWorld 等 self-supervised pretraining 方法实际上在做类似事情 — occupancy forecasting 作为 pretext task 学 generalizable representation。但 driving 这种 safety-critical domain，pixel-level 和 voxel-level fidelity 也很重要，不能完全 abstract away。

**LLM 与 World Model 的融合趋势**: OccLLaMA、Occ-LLM、HERMES、DrivingGPT 都在 unify vision + language + action modalities。这是受 GPT-4o、Gemini 等多模态 LLM 启发的方向。如果 LLM 的 reasoning 能力 + world model 的 physical grounding 能结合，可能产生真正 general 的 embodied AI agent。

---

## 11. 参考 Web Links 汇总

**Survey 自身**:
- Project Page: https://worldbench.github.io/survey
- GitHub Repo: https://github.com/worldbench/awesome-3d-4d-world-models

**Foundational Methods**:
- NeRF: https://arxiv.org/abs/2003.08934
- Gaussian Splatting: https://arxiv.org/abs/2308.16425
- VAE: https://arxiv.org/abs/1312.6114
- GAN: https://arxiv.org/abs/1406.2661
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- DiT: https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747
- VQ-VAE: https://arxiv.org/abs/1711.00937
- MaskGIT: https://arxiv.org/abs/2202.04200
- Mamba: https://arxiv.org/abs/2312.00752

**Key VideoGen Methods**:
- GAIA-1: https://arxiv.org/abs/2309.17080
- GAIA-2: https://arxiv.org/abs/2503.20523
- MagicDrive: https://arxiv.org/abs/2310.02699
- MagicDrive-V2: https://arxiv.org/abs/2408.03126
- Vista: https://arxiv.org/abs/2405.17358
- GenAD: https://arxiv.org/abs/2403.09630
- DriveDreamer: https://arxiv.org/abs/2309.09777
- DriveDreamer-2: https://arxiv.org/abs/2402.12248
- DriveDreamer4D: https://arxiv.org/abs/2410.22903
- Drive-WM: https://arxiv.org/abs/2311.13584
- DriveArena: https://arxiv.org/abs/2408.02090
- DrivingSphere: https://arxiv.org/abs/2503.04504
- DreamForge: https://arxiv.org/abs/2409.04003
- DiST-4D: https://arxiv.org/abs/2503.08821
- UniScene: https://arxiv.org/abs/2503.04232
- DrivingGPT: https://arxiv.org/abs/2412.18607
- DrivingWorld: https://arxiv.org/abs/2412.19505
- Doe-1: https://arxiv.org/abs/2412.09627
- ReconDreamer: https://arxiv.org/abs/2411.00748
- Cosmos-Drive: https://arxiv.org/abs/2501.03575

**Key OccGen Methods**:
- OccWorld: https://arxiv.org/abs/2401.09530
- OccSora: https://arxiv.org/abs/2405.20337
- OccLLaMA: https://arxiv.org/abs/2409.03272
- Occ-LLM: https://arxiv.org/abs/2412.10437
- DOME: https://arxiv.org/abs/2410.10429
- Cam4DOcc: https://arxiv.org/abs/2403.12534
- RenderWorld: https://arxiv.org/abs/2409.11356
- UniWorld: https://arxiv.org/abs/2308.07234
- DriveWorld: https://arxiv.org/abs/2312.06347
- UnO: https://arxiv.org/abs/2405.15274
- T3Former: https://arxiv.org/abs/2503.07338
- X-Scene: https://arxiv.org/abs/2506.13558
- DynamicCity: https://arxiv.org/abs/2409.08860

**Key LiDARGen Methods**:
- DUSty: https://arxiv.org/abs/2108.04017
- DUSty v2: https://arxiv.org/abs/2210.06534
- LiDARGen: https://arxiv.org/abs/2207.08631
- UltraLiDAR: https://arxiv.org/abs/2304.11164
- Copilot4D: https://arxiv.org/abs/2405.00380
- R2DM: https://arxiv.org/abs/2312.08012
- R2Flow: https://arxiv.org/abs/2403.16080
- LiDM: https://arxiv.org/abs/2312.02945
- LiDiff: https://arxiv.org/abs/2310.10348
- LiDARGRIT: https://arxiv.org/abs/2404.05505
- Text2LiDAR: https://arxiv.org/abs/2404.18507
- WeatherGen: https://arxiv.org/abs/2405.19072
- OLiDM: https://arxiv.org/abs/2411.04037
- LiDARCrafter: https://arxiv.org/abs/2508.03692
- La La LiDAR: https://arxiv.org/abs/2508.03691
- Veila: https://arxiv.org/abs/2508.03690
- X-Drive: https://arxiv.org/abs/2411.15271
- HoloDrive: https://arxiv.org/abs/2412.01407
- LidarDM: https://arxiv.org/abs/2404.02903
- OpenDWM: https://github.com/SenseTime-FVG/OpenDWM
- SPIRAL: https://arxiv.org/abs/2505.22643

**Datasets**:
- KITTI: https://www.cvlibs.net/datasets/kitti/
- nuScenes: https://www.nuscenes.org/
- Waymo Open: https://waymo.com/open/
- Argoverse 2: https://www.argoverse.org/
- SemanticKITTI: http://www.semantic-kitti.org/
- OpenDV: https://github.com/OpenDriveLab/DriveWorld
- NAVSIM: https://github.com/autonomousvision/navsim
- nuPlan: https://www.nuscenes.org/nuplan
- KITTI-360: http://www.cvlibs.net/datasets/kitti-360/
- Occ3D: https://tsinghua-mars-lab.github.io/Occ3D/
- OpenOccupancy: https://github.com/JeffWang98/OpenOccupancy

**Industry World Models**:
- NVIDIA Cosmos: https://arxiv.org/abs/2501.03575
- DeepMind Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model
- DeepMind Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models
- V-JEPA 2 (Meta): https://arxiv.org/abs/2506.09985

**Other Surveys**:
- Ding et al. world models survey: https://arxiv.org/abs/2411.14499
- Cosmos-Reason1: https://arxiv.org/abs/2503.15558

希望这份深度解读帮你 build 出 3D/4D world modeling 的完整 intuition — 从 mathematical formulation、taxonomy、methods landscape、datasets、evaluations、benchmarks、applications 到 open challenges，形成一张可操作的 mental map。后续如果想深入某个具体 method (比如 Copilot4D 的 discrete diffusion、DiST-4D 的 disentangled spatiotemporal factorization、X-Scene 的 triplane VAE)，可以再细聊。
