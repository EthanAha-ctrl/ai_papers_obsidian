---
source_pdf: A Comprehensive Survey on World Models for Embodied AI.pdf
paper_sha256: 6e5bf7339186149d2743a1bf0bfdad1c2709a7ed0f0e57a047c1d6852b1ec9ae
processed_at: '2026-07-17T19:49:25-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# World Models for Embodied AI: 一份综合 Survey 的技术深读

Andrej, 这篇由 Xinqing Li 等人写的 survey 是我近期看到的对 embodied AI 中 world models 最系统的梳理之一。它的核心贡献在于提出一个 **三轴 taxonomy**, 把散落在 robotics, autonomous driving, video generation 各自孤岛的几百篇工作整合进统一坐标。我会按"数学基础 → taxonomy → 数据/metrics → 实验对比 → 开放挑战"这条线深讲, 并穿插给你 build intuition。

参考链接:
- 论文 arXiv 版本 (持续更新): https://arxiv.org/abs/2411.05664 之类 (搜索 paper title)
- 作者维护的 GitHub: https://github.com/Li-Zn-H/AwesomeWorldModels
- Dreamer 系列原始页: https://danijar.com/project/dreamerv3/
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2/
- Sora 技术报告: https://openai.com/index/video-generation-models-as-world-simulators/

---

## 1. 数学框架: POMDP + Variational State-Space Model

这是整篇 paper 的基石, 你需要把这套 ELBO 形式深深嵌入到 intuition 里。

### 1.1 POMDP 设定

环境建模成 POMDP。每个 timestep $t \geq 1$:
- $o_t$: observation (agent 看到的)
- $a_t$: action (agent 执行的)
- $s_t$: **true state, 不可观测** — 这就是 partial observability 的核心
- $z_t$: **learned latent state**, 用 neural network 推断的

为了把 dynamics 写成统一形式, 引入 null initial action $a_0$ (在 $t=0$ 时, 让 $p_\theta(z_1 | z_0, a_0)$ 有意义)。

### 1.2 三大分布

$$
\begin{aligned}
& \text{Dynamics Prior:} \quad & p_\theta(z_t | z_{t-1}, a_{t-1}) \\
& \text{Filtered Posterior:} \quad & q_\phi(z_t | z_{t-1}, a_{t-1}, o_t) \\
& \text{Reconstruction:} \quad & p_\theta(o_t | z_t)
\end{aligned}
$$

**Build intuition**:
- $p_\theta$ 是"无观测情况下的演化规律", 类似物理引擎的 prior — 给我一个 state 和 action, 我告诉你下一个 state 怎么走
- $q_\phi$ 是"看到 $o_t$ 之后对 $z_t$ 的后验估计", 类似 Bayesian filter (Kalman filter 的 learned 版本)
- 这两个分布通过 KL 项被对齐 — 这是 world model 区别于单纯 VAE 的关键: dynamics prior 在 inference 时 (没观测了) 还能 rollout

这个设计就是 RSSM (Recurrent State-Space Model) 的核心, 由 PlaNet (Hafner 2019) 提出, 被 Dreamer 系列发扬光大。

### 1.3 Markov factorization 下的联合分布

$$
p_\theta(o_{1:T}, z_{0:T} | a_{0:T-1}) = p_\theta(z_0) \prod_{t=1}^T p_\theta(z_t | z_{t-1}, a_{t-1}) \, p_\theta(o_t | z_t)
$$

这个 chain 是 standard HMM / state-space model 的结构: $z_{t-1} \to z_t \to o_t$。把 $a_{t-1}$ 当成 conditioning, 就是 action-conditioned 版本。

### 1.4 Time-factorized Variational Posterior

$$
q_\phi(z_{0:T} | o_{1:T}, a_{0:T-1}) = q_\phi(z_0 | o_1) \prod_{t=1}^T q_\phi(z_t | z_{t-1}, a_{t-1}, o_t)
$$

注意, 这里 $q_\phi$ **不是 global smoothing posterior** (那种 $q(z_t | o_{1:T})$, 类似 RTS smoother), 而是 **filtering posterior** — 只用 $o_t$ 和上一时刻的 $z_{t-1}$。这保证了 online inference 时只需要 streaming 计算, 不需要 backward pass。

### 1.5 ELBO

直接最大化 $\log p_\theta(o_{1:T} | a_{0:T-1})$ intractable (因为 marginalize $z$)。引入 variational lower bound:

$$
\log p_\theta(o_{1:T} | a_{0:T-1}) \geq \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(o_{1:T}, z_{0:T} | a_{0:T-1})}{q_\phi(z_{0:T} | o_{1:T}, a_{0:T-1})}\right] =: \mathcal{L}(\theta, \phi)
$$

在 Markov factorization 假设下, ELBO 分解为:

$$
\mathcal{L}(\theta, \phi) = \underbrace{\sum_{t=1}^T \mathbb{E}_{q_\phi(z_t)}[\log p_\theta(o_t | z_t)]}_{\text{Reconstruction}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z_{0:T} | o_{1:T}, a_{0:T-1}) \,\|\, p_\theta(z_{0:T} | a_{0:T-1})\big)}_{\text{KL regularization}}
$$

**Build intuition**:
- Reconstruction 项让 latent state 编码足够多信息, 才能 decode 出 $o_t$
- KL 项让 filtered posterior $q_\phi$ 跟 dynamics prior $p_\theta$ 对齐 — 训练时 q 看 observation, 但推理时 (imagination rollout) 只有 p, 两者必须匹配

这就是 RSSM 训练的全部损失。可以扩展为 recurrent model (DreamerV1/V2/V3), Transformer (TWM, IRIS, STORM), 或 diffusion decoder (EnerVerse, Drive-WM, RoboDreamer)。

---

## 2. 三轴 Taxonomy

这是 paper 的核心组织原则, 把 hundreds of methods 切成 $2 \times 2 \times 4$ 的 grid (实际不是所有 cell 都有 work)。

### 2.1 Axis 1: Functionality

- **Decision-Coupled**: dynamics 专门为某个决策任务学习 — 通常是 model-based RL (Dreamer 系列) 或 task-specific driving model (MILE, SEM2)
- **General-Purpose**: task-agnostic simulator, 大规模预训练, 可以迁移到多种 downstream task (Sora, V-JEPA 2, Genie, GenAD)

**Intuition 区别**: Decision-Coupled 关心的是 "dynamics 帮我做好这个 task", General-Purpose 关心的是 "dynamics 准确预测未来世界"。前者 sample-efficient (因为 reward signal 引导), 后者 transferable (因为不绑定 task)。

### 2.2 Axis 2: Temporal Modeling

这是 paper 最有洞察力的 axis, 我觉得是整个 survey 最有价值的 contribution。

- **Sequential Simulation and Inference**: 自回归地一步一步展开 — $z_t \to z_{t+1} \to z_{t+2} \to \dots$。优点: 紧凑, sample-efficient, 可以 closed-loop。缺点: **error accumulation** — 第 100 步的预测严重依赖前 99 步预测的准确性。
- **Global Difference Prediction**: 直接并行预测整个 future 序列 (或 future state 与 current state 的 difference)。优点: 避免 autoregressive error accumulation, multi-step consistency 好, 可以多模态采样。缺点: 计算重, closed-loop 交互弱 (很难做 streaming)。

**Deep intuition**: 你可以联想 transformer decoder (autoregressive, KV-cache 维护 history) vs masked prediction (BERT-style, 一次性预测多个 token)。world models 的两种 paradigm 就是这两个思想的对应:
- Sequential 类比 GPT-style: 每次生成下一个 step, history 通过 recurrent state 或 KV cache 维护
- Global 类比 BERT-style / diffusion: 给定 initial state, 一次 forward 生成整个 future trajectory

**实际工作的分布**: 论文 Section III 显示, 早期 RSSM 系列都是 Sequential; 后来 Transformer 和 Diffusion 兴起后, Global Difference Prediction 越来越多 (Sora, V-JEPA, WorldDreamer, MiLA, Epona)。这是 field 演化的主轴。

### 2.3 Axis 3: Spatial Representation

这是 4 个 sub-strategy:

**(1) Global Latent Vector**: 把整个 scene 压成一个 vector $z_t \in \mathbb{R}^d$。代表: Dreamer 系列, HRSSM, DWL。优点: 计算超快, 适合 real-time robot control。缺点: 丢失 spatial detail, 难处理 occlusion, multi-object 场景。

**(2) Token Feature Sequence**: 把 scene 编码成 token 序列 $[z_t^1, z_t^2, \dots, z_t^K]$。代表: TWM, IRIS, iVideoGPT, Genie, DrivingGPT, Doe-1。优点: 可以 reuse LLM 的 next-token prediction paradigm, 多模态融合容易 (vision + action + language 都是 token)。缺点: token 的 spatial 归纳偏置弱 (跟 CNN 相比), 需要大量数据学。

**(3) Spatial Latent Grid**: 用 BEV 或 voxel grid 编码, 保留 spatial structure。代表: DriveWorld, OccWorld, OccLLaMA, Cam4DOcc, BEVWorld, GaussianWorld。优点: 保留 locality, 适合 convolution / sparse attention, planner-friendly (路径规划可以直接在 BEV 上做)。缺点: discretization 误差, voxel grid 计算开销大。

**(4) Decomposed Rendering Representation**: 用 3DGS 或 NeRF 这类 explicit primitives, 通过 differentiable rendering 生成 view-consistent observation。代表: ManiGaussian, DreMa, PIN-WM, DriveDreamer4D, ReconDreamer, InfiniCube。优点: view consistency 极强, 物理结构 explicit, 可以接入 digital twin。缺点: dynamic scene 下 Gaussian 数量爆炸, NeRF 渲染慢, 训练不稳定。

**Intuition for spatial axis**: 从 (1) 到 (4), geometric fidelity 越来越高, 但计算开销和训练难度也越来越高。Robotics manipulation 任务早期倾向 (1) (因为任务简单, 不需要 high-fidelity geometry), 现在越来越多用 (4) (ManiGaussian)。Autonomous driving 走了 (1) → (3) → (4) 的演化, 现在 occupancy-based methods (3) 和 3DGS-based methods (4) 是主流。

---

## 3. Representative Methods 深度解析

### 3.1 RSSM 家族: 从 PlaNet 到 DreamerV3

**PlaNet (Hafner 2019, ICML)**: 引入 RSSM, 关键创新是 deterministic + stochastic 分支混合:
- Deterministic branch: $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$ — 类似 GRU, 维护长程 memory
- Stochastic branch: $s_t \sim q(s_t | h_t, o_t)$ (posterior) 或 $s_t \sim p(s_t | h_t)$ (prior) — 处理环境随机性
- 完整 latent: $z_t = (h_t, s_t)$

**DreamerV1 (2020, ICLR)**: 在 RSSM 基础上做 actor-critic policy learning, **在 latent space imagination 内做 long-horizon planning**, 不需要真实 environment 交互。

**DreamerV2 (2021, ICLR)**: 把 continuous latent 换成 **discrete latent** (categorical distribution), 用在 Atari 上达到 SOTA。

**DreamerV3 (2025, Nature)**: 关键创新是 **fixed hyperparameters across domains** — 同一组超参在 150+ tasks 上 work, 包括 Atari, DMC, Minecraft, robotic manipulation。这极大降低了 model-based RL 的调参成本, 是 paradigm shift。

参考: https://danijar.com/project/dreamerv3/

### 3.2 Transformer-based: TWM, IRIS, STORM

**TWM (Robine et al. 2023, ICLR)**: 用 Transformer 替代 recurrent core, attention 维护 history。证明 100k interaction 就能学好 Atari。

**IRIS (Micheli et al. 2023, ICLR)**: 完全 discrete token, VQ-VAE 把 image tokenize, 然后 Transformer autoregressive 预测。类似 video GPT, 但用于 RL imagination。

**STORM (Zhang et al. 2023, NeurIPS)**: Stochastic Transformer world model, 在 latent space 做 stochastic 预测, 兼顾随机性和长程依赖。

### 3.3 Diffusion-based: Drive-WM, EnerVerse, RoboDreamer

**Drive-WM (Wang et al. 2024, CVPR)**: Multi-view diffusion model, 用 image-based reward 选 safest trajectory。Spatial Latent Grid + Sequential。

**EnerVerse (Huang et al. 2025)**: Chunk-wise autoregressive video diffusion, sparse memory mechanism + 4DGS, 用于 robotic manipulation。

**RoboDreamer (Zhou et al. 2024, ICML)**: 把 instruction 分解成 low-level primitives, 用这些 primitives 引导 video diffusion, 合成 compositional novel scene。Global Difference Prediction + Token Feature Sequence。

### 3.4 JEPA 家族: V-JEPA, V-JEPA 2

**V-JEPA (Bardes et al. 2024, TMLR)**: LeCun 的 JEPA 视频版, 预测 masked spatiotemporal region 的 latent feature, **不做 pixel reconstruction, 不做 contrastive learning**。核心思想: 在 feature space 学 representation 比在 pixel space 学更好 (pixel 含太多 irrelevant detail)。

**V-JEPA 2 (2025)**: 把 V-JEPA scaling 到 22M videos (VideoMix22M dataset), 加入 limited robot interaction data 做 post-training, transfer 到 robotic planning。这是 LeCun "predictive world model + planning" 路线的最重要 milestone。

**Key formula intuition for JEPA**:
$$\mathcal{L}_{\text{JEPA}} = \|g_\theta(x_{\text{visible}}) - \text{sg}(f_\phi(x_{\text{target}}))\|_2^2$$

- $g_\theta$: predictor 网络, 输入 visible patches + mask 信息, 预测 target patches 的 representation
- $f_\phi$: target encoder, 输入 target patches (从原视频取), 输出 representation
- $\text{sg}$: stop-gradient, 防止 collapse
- $\phi$ 通过 EMA (exponential moving average) 跟随 $\theta$ 更新, 防止 trivial solution

参考: https://ai.meta.com/blog/v-jepa-2/

### 3.5 3DGS-based World Models

**ManiGaussian (Lu et al. 2024, ECCV)**: Predict per-point Gaussian variations to generate future Gaussian scene, 用于 manipulation。每个 Gaussian point 都有 learned motion attribute。

**GaussianWorld (Zuo et al. 2025, CVPR)**: 把 scene evolution 分解成 three components:
1. Ego-motion: 自车运动导致的 scene transform
2. Object dynamics: 动态物体的运动
3. Newly observed regions: 新进入视野的部分

Iterative update Gaussian primitives。

**InfiniCube (Lu et al. 2025, ICCV)**: Hybrid pipeline, voxel generation + video synthesis + dynamic Gaussian reconstruction, conditioned on HDMaps + bounding boxes + text。这是 **city-scale dynamic 3D driving scene** 生成的代表。

参考: https://infinicube.github.io/

---

## 4. Data Resources 全景

Paper Table III 系统列出 4 类资源:

### 4.1 Simulation Platforms

- **MuJoCo** (Todorov 2012): articulated body physics, contact dynamics。Robotics control 标准
- **NVIDIA Isaac Sim / Gym / Lab**: GPU-accelerated, photorealistic rendering + large-scale RL (Isaac Gym 可以跑 thousands of parallel envs)
- **CARLA** (Dosovitskiy 2017): 基于 Unreal Engine 的 urban driving simulator, closed-loop eval protocol
- **Habitat** (Savva 2019): 室内 navigation 高性能 sim, photorealistic 3D scan

### 4.2 Interactive Benchmarks

- **Atari** (Bellemare 2013) + **Atari100k** (Kaiser 2020): sample efficiency 评估 (限制 100k steps)
- **DMC (DeepMind Control)**: MuJoCo-based, 30+ continuous control tasks
- **Meta-World**: 50 robotic manipulation tasks, multi-task / meta-RL
- **RLBench**: 100 tabletop manipulation, 7-DoF Franka arm
- **LIBERO**: lifelong manipulation, 130 procedurally generated tasks
- **nuPlan**: 1500+ hours real driving logs, closed-loop planning benchmark

### 4.3 Offline Datasets (重要, 这是预训练燃料)

- **RT-1**: 130k demos, 700+ tasks, 13 robots, 17 months 收集 — Google DeepMind
- **Open X-Embodiment (OXE)**: 60 sources, 21 institutions, 22 robot embodiments, 1M+ trajectories。这是 cross-embodiment pretraining 的关键 corpus
- **nuScenes**: 1000 20s scenes, Boston+Singapore, 6 cameras + 5 radars + 1 LiDAR
- **Waymo Open Dataset**: 1150 20s scenes, 10Hz, 5 LiDARs + 5 cameras
- **Occ3D**: voxel-level occupancy supervision, 0.4m (nuScenes) 或 0.05m (Waymo) 分辨率
- **SSv2 (Something-Something v2)**: 220k clips, 174 action categories, fine-grained action understanding
- **OpenDV**: 2059 hours, 65.1M frames, YouTube + 7 个公开数据集, 40+ countries 244 cities — 这是 driving video pretraining 最大 corpus
- **VideoMix22M**: V-JEPA 2 提出, 2M → 22M samples, 来自 YT-Temporal-1B + HowTo100M + Kinetics + SSv2 + ImageNet

### 4.4 Real-world Robot Platforms

- **Franka Emika**: 7-DoF 协作机械臂, 1kHz torque control
- **Unitree Go1**: 四足, 1.5 TFLOPS onboard, 4.7 m/s — quadruped locomotion 标准
- **Unitree G1**: humanoid, 43-DoF, knee torque 120 N·m, integrated 3D LiDAR + depth cameras

---

## 5. Metrics 详细解析

Paper 把 metrics 分三层: pixel → state → task。

### 5.1 Pixel Generation Quality

**FID (Fréchet Inception Distance)**:
$$\text{FID}(x, y) = \|\mu_x - \mu_y\|_2^2 + \text{Tr}\left(\Sigma_x + \Sigma_y - 2(\Sigma_x \Sigma_y)^{1/2}\right)$$

- $\mu_x, \mu_y \in \mathbb{R}^{2048}$: real 和 generated image 通过 ImageNet-pretrained Inception-v3 后的 feature mean
- $\Sigma_x, \Sigma_y \in \mathbb{R}^{2048 \times 2048}$: feature covariance
- 第一项 $\|\mu_x - \mu_y\|_2^2$: fidelity (mean shift)
- 第二项 $\text{Tr}(\dots)$: diversity (covariance mismatch, 衡量 mode collapse)
- Lower is better

**FVD (Fréchet Video Distance)**: 把 Inception-v3 换成 I3D (Kinetics-400 预训练), 评估 video 的 appearance + dynamics。

**SSIM (Structural Similarity)**:
$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\Sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\Sigma_x^2 + \Sigma_y^2 + C_2)}$$

- $\mu_x, \mu_y$: 局部 patch 的 luminance mean
- $\Sigma_x^2, \Sigma_y^2$: variance (contrast)
- $\Sigma_{xy}$: covariance (structure)
- $C_1 = (0.01 L)^2$, $C_2 = (0.03 L)^2$: $L$ 是 pixel dynamic range, 防止除零
- Values closer to 1 = higher similarity

**PSNR**:
$$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

- MAX: 255 (8-bit) 或 1 (normalized)
- MSE: $\frac{1}{N}\sum_i (x_i - y_i)^2$

**LPIPS (Learned Perceptual Image Patch Similarity)**:
$$\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\|w_l \odot (\hat{f}_{h,w,x}^l - \hat{f}_{h,w,y}^l)\right\|_2^2$$

- $\hat{f}_{h,w,x}^l \in \mathbb{R}^{C_l}$: layer $l$ 在 spatial position $(h,w)$ 的 unit-normalized activation
- $w_l \in \mathbb{R}^{C_l}$: learned channel weights (per-layer)
- $H_l, W_l$: layer $l$ 的 spatial dimensions
- Lower = more similar

**VBench**: 16 个维度, 分 Video Quality (subject consistency, motion smoothness 等) 和 Video-Condition Consistency (object class, human action 等)。

### 5.2 State-level Understanding

**mIoU**:
$$\text{IoU}_c = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}, \quad \text{mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c$$

**mAP** (per class $c$, IoU threshold $\tau$):
$$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}$$
$$\text{AP}_{c,\tau} = \int_0^1 P_{c,\tau}(r) \, dr, \quad \text{mAP} = \frac{1}{|C|} \sum_{c \in C} \frac{1}{|T|} \sum_{\tau \in T} \text{AP}_{c,\tau}$$

- $P_{c,\tau}(r)$: precision-recall envelope, monotonic interpolation
- $T$: set of IoU thresholds (通常 $[0.5, 0.95]$ 步长 0.05)

**Chamfer Distance**:
$$\text{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2^2 + \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2^2$$

- $S_1, S_2$: predicted 和 ground truth point sets
- 第一项: $S_1$ 中每个点到 $S_2$ 最近点的距离平方之和
- 第二项: 反向, $S_2$ 中每个点到 $S_1$ 最近点
- Bidirectional, 对称。可用于训练 loss + eval

**Displacement Error** (ADE / FDE):
- ADE = Average Displacement Error: 整条 trajectory 的 L2 平均
- FDE = Final Displacement Error: 终点的 L2

### 5.3 Task Performance

- **Success Rate (SR)**: navigation 到达目标 / manipulation 物体放置成功 / driving 不碰撞完成路线
- **Sample Efficiency (SE)**: 达到 target performance 需要的 samples
- **Reward**: cumulative reward $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$, $\gamma$ 是 discount factor
- **Collision Rate**: 至少一次碰撞的 episode 比例 (driving 用 collision/km 或 collision/hour normalized)

---

## 6. 实验数据深度解读

### 6.1 Table IV: nuScenes 视频生成

| Method | Pub | Resolution | FID↓ | FVD↓ |
|---|---|---|---|---|
| DriveDreamer | ECCV'24 | 128×192 | 52.6 | 452.0 |
| MagicDrive3D | arXiv'24 | 224×400 | 20.7 | 164.7 |
| GenAD | CVPR'24 | 256×448 | 15.4 | 184.0 |
| Drive-WM | CVPR'24 | 192×384 | 15.8 | 122.7 |
| Delphi | arXiv'24 | 512×512 | 15.1 | 113.5 |
| Vista | NeurIPS'24 | 576×1024 | 6.9 | 89.4 |
| DrivingWorld | arXiv'24 | 512×1024 | 7.4 | 90.9 |
| Epona | ICCV'25 | 512×1024 | 7.5 | 82.8 |
| MaskGWM | CVPR'25 | 288×512 | 8.9 | 65.4 |
| DriveDreamer-2 | AAAI'25 | 256×448 | 11.2 | 55.7 |
| GeoDrive | arXiv'25 | 480×720 | 4.1 | 61.6 |
| GEM | CVPR'25 | 576×1024 | 10.5 | 158.5 |
| LongDWM | arXiv'25 | 480×720 | 12.3 | 102.9 |
| UniFuture | arXiv'25 | 320×576 | 11.8 | 99.9 |
| DriVerse | ACMMM'25 | 480×832 | 18.2 | 95.2 |
| STAGE | IROS'25 | 512×768 | 11.0 | 242.8 |
| **DrivePhysica** | arXiv'24 | 256×448 | **4.0** | **38.1** |
| **MiLA** | arXiv'25 | 360×640 | 4.1 | **14.9** |

**Intuition**:
- DriveDreamer (ECCV'24) FID=52.6, FVD=452 — 一年前的 baseline 现在看起来很差
- DrivePhysica 用 physics-informed constraints, FID=4.0 是 appearance fidelity 最佳
- MiLA 用 coarse-to-fine + anchor frame interpolation, FVD=14.9 是 temporal consistency 最佳
- Vista (NeurIPS'24) 在两个 metric 都进入 top 5, 综合最强, 576×1024 大分辨率
- 演化趋势: resolution 越来越大, FID/FVD 持续下降, physics-aware 方法崛起

### 6.2 Table V: Occ3D-nuScenes 4D Occupancy Forecasting

输入 2s 历史 3D occupancy, 预测后续 3s。关键数字:

- Copy & Paste (naive baseline): avg mIoU = 11.33% (1s=14.91, 2s=10.54, 3s=8.52)
- OccWorld-O: 17.14% avg
- OccLLaMA-O: 19.93%
- RenderWorld-O: 20.80%
- DOME-O (with GT ego): 27.10%
- DTT-O: 30.85% — triplane transformer, autoregressive
- **COME-O (with GT ego): 34.23%** — best avg mIoU

**Key insight**: 
1. 时间越长, IoU 衰减越严重 (1s vs 3s 差 5-10%) — 这是 long-horizon forecasting 的 inherent difficulty
2. 加入 GT ego trajectory 显著提升 (COME-O 从 21.29 → 34.23) — ego motion prediction 是 bottleneck
3. Camera-based 方法 (OccWorld-S self-supervised: 0.26%) 远远落后于 occupancy-input 方法 (20%+), 说明 geometry representation 极其重要

### 6.3 Table VI: DMC

- PlaNet: 5M steps, 333 avg / 20 tasks
- Dreamer: 5M steps, 823 avg
- DreamerPro: 1M, 857
- HRSSM: **500k**, 938
- DisWM: 1M, 879

**Intuition**: HRSSM 用 500k steps 达到 938, 是 sample efficiency 的 SOTA — dual-branch architecture + reconstruction-free。Dreamer 系列 sample efficiency 持续提升, 但绝对性能上限差不多。

### 6.4 Table VII: RLBench Manipulation

- ManiGaussian: IDM, 67 avg / 18 tasks (传统 GS)
- ManiGaussian++: 45/10 (bimanual extension)
- **VidMan: 63/10** — video diffusion model adapted into IDM, 用 self-attention adapter
- EnerVerse: DiT, video diffusion + 4DGS

**Intuition**: VidMan 表现出色说明 video foundation model 适应 manipulation IDM 是 promising direction — 视频预训练的 motion prior 可以 transfer 到 robotic control。

### 6.5 Table VIII: nuScenes Open-Loop Planning

观察 2s 历史, 预测 3s BEV waypoints。Metric: L2 error (m) 和 collision rate (%)。

- UniAD: L2=1.03, collision=0.31 (5 auxiliary supervisions: Map/Box/Motion/Tracklets/Occ)
- UniAD+DriveWorld: **L2=0.69**, collision=0.19 — 加入 4D pretrained scene understanding 显著提升
- **SSR**: L2=0.75, collision=**0.15** (no auxiliary supervision!) — sparse scene representation, 最强 sample efficiency
- OccWorld-S (self-supervised): L2=1.83, collision=2.02 — 性能最差, 但完全无监督
- DTT-F: L2=1.08, collision=0.44 — triplane transformer + occupancy supervision

**Key insight**: 
1. Camera-based 方法 (SSR, DTT-F) 已经接近甚至超过 occupancy-input 方法 — 说明 end-to-end 视觉规划成熟了
2. Drive-OccWorld (no supervision): L2=0.85, collision=0.29 — 接近 UniAD, 但不用 map/box/motion supervision
3. Tradeoff: 辅助 supervision 越多, L2 越低, 但泛化能力可能下降

---

## 7. 开放挑战与趋势

### 7.1 Data & Evaluation

**Challenges**:
- 跨 domain unified dataset 缺失 (robotics manipulation, navigation, driving 各自孤岛)
- FID/FVD 只看 pixel fidelity, 不看 **physical consistency, dynamics, causality**
- EWM-Bench 是 recent attempt, 但仍 task-specific

**Future**: 构建 unified multimodal cross-domain dataset, 评估从 perceptual realism 转向 physical consistency, causal reasoning, long-horizon dynamics。

参考 EWM-Bench: https://arxiv.org/abs/2505.09694

### 7.2 Computational Efficiency

**Challenges**:
- Transformer 和 Diffusion 推理成本高, 跟 real-time robot control 的需求冲突 (10Hz-1kHz)
- RNN 和 global latent vector 仍然是 real-time 部署首选, 但 long-range modeling 弱

**Future**: Quantization, pruning, sparse computation 降延迟; SSM (Mamba) 提供 linear-time long-range reasoning 可能。

### 7.3 Modeling Strategy

**Challenges**:
- Autoregressive 紧凑 sample-efficient 但 error accumulation
- Global prediction coherence 好但计算重, closed-loop 弱
- Spatial representation trade-off: latent vector 快但 coarse, 3DGS 高保真但 dynamic scene 下 scaling 差

**Future**: 
- SSM (Mamba) + autoregressive — linear-time scalability + long-horizon
- Masked approaches (JEPA) — 更高效 representation learning, 但 closed-loop integration 难
- **Hybrid autoregressive + global prediction** — explicit memory / hierarchical planning / CoT-style task decomposition
- 统一 architecture 平衡 fidelity / efficiency / interactivity

---

## 8. 我对这篇 survey 的几个直觉

### 8.1 最有价值的 axis: Temporal Modeling

Paper 的三轴 taxonomy 中, **Temporal Modeling (Sequential vs Global)** 是最有洞察力的。这把 world model field 跟 LLM / video generation 的演化路径对齐了:
- Sequential = GPT-style autoregressive (Dreamer 系列, TWM, IRIS, DrivingGPT)
- Global = BERT-style masked / diffusion (V-JEPA, Sora, WorldDreamer, MiLA)

这跟你 Andrej 在 LLM 上长期讨论的 "autoregressive vs masked prediction" debate 完全平行。在 world models 里, autoregressive 现在主导 decision-coupled RL (因为需要 closed-loop), global prediction 主导 general-purpose video generation (因为需要 multi-step coherence)。

### 8.2 Hidden trend: LLM 范式迁移到 world models

Token Feature Sequence 这一栏里, 越来越多方法直接借 LLM 范式:
- Doe-1, DrivingGPT: 把 perception + prediction + planning 统一成 next-token prediction
- WorldVLA: VLA unified tokenized representation
- ECoT, NavCoT, MineDreamer: 用 Chain-of-Thought 解构 long-horizon 任务
- Statler, Inner Monologue: LLM 直接当 world model

这是 **paradigm convergence**: LLM 的训练范式 (tokenize + autoregressive + next-token prediction) 被迁移到 world model, 跟 RL 中 model-based 的 imagination 训练融合。你长期关注的 "LLM as world model" 想法正在被多线验证。

### 8.3 3DGS 正在接管 high-fidelity representation

Decomposed Rendering Representation 这一栏 (3DGS / NeRF) 是增长最快的方向:
- Manipulation: ManiGaussian, GAF, DreMa, PIN-WM, DTT
- Driving: DriveDreamer4D, ReconDreamer, MagicDrive3D, GaussianWorld, InfiniCube, MaskGWM

3DGS 相比 NeRF 的优势: explicit point-based, real-time rendering, 容易接入 digital twin 和 differentiable physics。我猜未来 1-2 年 3DGS 会成为 high-fidelity world model 的默认 representation, 尤其在 manipulation 和 digital twin 场景。

### 8.4 Long-horizon consistency 是真正的 bottleneck

所有 benchmark 都显示: prediction horizon 越长, 性能衰减越严重 (Table V 的 1s → 3s IoU 下降 50%+)。这是 fundamental challenge:
- Autoregressive 路径靠 hierarchical planning / memory mechanism (LongDWM distillation, VRAG retrieval, StateSpaceDiffuser Mamba+diffusion)
- Global 路径靠 anchor frame + interpolation (MiLA coarse-to-fine)
- Hybrid 路径: explicit memory + hierarchical planning + CoT-style decomposition

这跟 LLM 的 long context 问题完全同构。LongDWM 用 distillation 让 fine-grained DiT 指导 coarse model, MiLA 用 sparse anchor + refinement — 这些都是 LLM 中 long-context attention 的对应解法。

### 8.5 Metrics 是当前最大的 gap

Paper 反复强调: FID/FVD 只看 pixel fidelity, **忽略 physical consistency, dynamics, causality**。但这是 field-level 的根本问题:
- 一个 FID=4 的 driving video 可能 car 在空中飞, 或者穿越墙体
- 一个 L2=0.7 的 trajectory planning 可能完全违反 traffic rules
- EWM-Bench 是初步尝试, 但仍然 task-specific

未来 critical 的方向是 **physical consistency benchmark**: physics-aware metrics (能量守恒, 动量守恒, 碰撞合理性), causal metrics (counterfactual rollout 一致性), behavioral metrics (是否符合 driving norms)。

---

## 9. 结语

这份 survey 把散落的 world model work 整合进三维框架, 而且提供了标准化 benchmark 对比。对你长期关注的 "neural network 学到 world model" 这个大问题, 现在的 field 状态是:

- 数学框架 (POMDP + ELBO + reconstruction-regularization) 已经成熟, 大部分 SOTA method 都在这套框架内做 variant
- 三大 representation (latent vector / token / spatial grid / 3DGS) 各有 trade-off, 没有统一 winner
- LLM 范式 (tokenize + next-token) 正在 fast 演化到 world model, 但 closed-loop control 是 bottleneck
- Long-horizon consistency 和 physical consistency 是真正未解决的问题
- Data scale 还远不够 (V-JEPA 2 的 22M videos 跟 LLM 的 trillion tokens 比是小数)

建议你重点关注:
1. V-JEPA 2 (LeCun 路线) vs DreamerV3 (RL 路线) 的 convergence
2. Sora + Cosmos-Transfer (OpenAI/NVIDIA) 在 general-purpose simulator 上的 scaling
3. 3DGS-based world models (ManiGaussian, InfiniCube) 在 digital twin 上的应用
4. Long-horizon consistency 的 hybrid solutions (Mamba + diffusion, hierarchical planning + CoT)

希望这份深度讲解帮你的 intuition 更清晰。如果你想深挖某个 sub-area (比如 JEPA 的数学推导, 或者 3DGS 在 dynamic scene 的具体 formulation), 我们可以继续。
