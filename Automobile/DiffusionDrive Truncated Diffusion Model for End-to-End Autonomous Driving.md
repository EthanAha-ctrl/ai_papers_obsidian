---
source_pdf: DiffusionDrive Truncated Diffusion Model for End-to-End Autonomous Driving.pdf
paper_sha256: 6ad4f8a379494eeb099cf7d489b75bec1c0ec3321428bcd0b0ca0f05662b8600
processed_at: '2026-08-03T21:57:04-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DiffusionDrive

好，我换个节奏，跟你聊天的口吻讲。这篇 paper 其实干的事情特别朴素，但背后的 intuition 很漂亮。

---

## 这 paper 到底干了啥

一句话：**让 diffusion model 在自动驾驶里跑得又快又好，快到 45 FPS 实时，好到 NAVSIM 88.1 PDMS 破纪录**。

但是这个故事要讲清楚，得从为什么之前的 driving planner 都不太对劲开始讲。

---

## 1. 为什么单条 trajectory regression 根本是错的

你看 Transfuser（https://arxiv.org/abs/2205.15997）、UniAD（https://arxiv.org/abs/2212.10156）、VAD（https://arxiv.org/abs/2303.12077）这些主流 end-to-end planner，它们的 planning head 都是同一个套路：拿一个 ego query，过个 MLP，回归出一条未来 4 秒的 trajectory，8 个 $(x_t, y_t)$ waypoints。

这个范式在 intuition 上有个根本性的 bug：**driving 的 action distribution 是 multimodal 的**。比如你现在高速上跟在一辆慢车后面，前方左车道空着、右车道也空着，那合理的选择至少有：跟车减速、左侧换道超车、右侧换道。这三个 action 在 BEV 平面上几何差异巨大，**根本不在同一个 mode 里**。

如果你拿 L2 loss 去回归，模型会学这三个 mode 的「平均」——可能就是一个「往左前方轻轻漂一点同时减速」的诡异轨迹，几何上根本不可执行，立刻撞墙或者冲出 lane。

这就是为什么 VADv2（https://arxiv.org/abs/2402.13243）想搞 8192 个 anchor trajectory 组成的 vocabulary，让模型对每个 anchor 打分再采样。思路对，但 vocabulary 太大、算力太贵、out-of-vocabulary 会 fail。

---

## 2. 为什么直接套 diffusion policy 也炸了

Diffusion policy（https://arxiv.org/abs/2303.04137）在 robotics 里火得一塌糊涂，Chi et al. 那篇 RSS 2023 证明 diffusion 能 capture multimodal action distribution。作者心想，那我把 Transfuser 的 MLP head 换成 conditional diffusion UNet 不就完事了？这就是 Transfuser$_{DP}$。

结果两个大坑：

### 坑 1: 太慢

Vanilla DDIM 要 20 step denoising，每 step 6.5ms，total 130ms，FPS 从 60 掉到 7。自动驾驶 real-time 要至少 20Hz，7 FPS 直接 GG。

### 坑 2: Mode collapse

这个是更恶心的。作者采样 20 个不同的 Gaussian noise 起点，跑 20 step denoise，发现最后 20 条 trajectory **几乎全重叠在一起**。论文里专门定义了一个 mode diversity score $\mathcal{D}$，Transfuser$_{DP}$ 的 $\mathcal{D}$ 只有 11%。

直觉上为啥会 collapse？你从 $\mathcal{N}(0, I)$ 采样，所有 sample 在 reverse process 早期都长得差不多——就是一团 mean 附近的 noise。score function 的梯度在 early step 几乎只指向 data distribution 的整体 centroid，mode 信息要到最后几步才分叉。但 DDIM 只有 20 step，分叉时间不够，结果所有 sample 都塌到 dominant mode。

driving 这个 task 比 robotics manipulation 更难，因为 scene 是 open-world 的，mode 之间几何距离很大，pure Gaussian 起点离任何 mode 都很远，模型很难在 20 step 内把 sample 推到正确的 mode 邻域。

---

## 3. Truncated Diffusion 的核心 insight

作者的 insight 我觉得特别 elegant：**人开车不是从「完全随机」开始想的，而是从几个固定的 driving pattern 开始，根据当前 scene 微调**。

你想想自己开车，脑子里其实有几个 template：直行跟车、左换道、右换道、减速刹停、转弯入弯。你看一眼 scene，挑一个 template，然后根据具体车流微调。你从来不会从「random noise 轨迹」开始构思。

所以作者做了两件事：

### 3.1 用 K-Means 聚类出 prior anchor

在 training set 上对所有 ground truth trajectory 跑 K-Means，聚出 $N_{anchor}=20$ 个 cluster center $\mathbf{a}_k$。这 20 个 anchor 就覆盖了 driving 的主要 pattern：直行、左转、右转、各种换道、刹停等。

注意这个 20 vs VADv2 的 8192——**400× 减少**。为什么 20 个够？因为 diffusion model 本身有 distribution expressivity，每个 anchor 周围采样加 noise 后能覆盖一个邻域，20 个邻域叠加起来基本覆盖整个 action space。而 VADv2 的 anchor 是 deterministic point，必须靠数量堆 coverage。

### 3.2 Truncate diffusion schedule

标准 DDPM 是 1000 step 加噪到 pure Gaussian，DDIM 推理用 20 step 反向。作者训练时只加噪到 step 50（50/1000 truncation），forward process 公式：

$$
\tau_k^i = \sqrt{\bar{\alpha}^i}\, \mathbf{a}_k + \sqrt{1-\bar{\alpha}^i}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

变量拆解：
- $\tau_k^i$：第 $k$ 个 anchor 在 diffusion timestep $i$ 的 noisy trajectory
- $i \in [1, 50]$，$T_{trunc}=50 \ll T=1000$
- $\mathbf{a}_k$：第 $k$ 个 K-Means cluster center，是 prior driving pattern
- $\bar{\alpha}^i = \prod_{s=1}^{i}(1-\beta^s)$：累积信号保留率，$i$ 越大保留越少、noise 越多
- $\beta^s$：DDPM linear noise schedule 第 $s$ 步的噪声增量
- $\epsilon$：标准高斯噪声

当 $i=50$ 时，$\bar{\alpha}^{50}$ 大概在 0.6~0.7 量级，意思是 noisy trajectory = 0.8×anchor + 0.6×noise。这个起点既保留了 anchor 的 mode 信息，又给了足够的 exploration 自由度。

这个 anchored Gaussian distribution 本质上是 **mixture of Gaussians**：

$$
p_{init}(\tau) = \frac{1}{N_{anchor}} \sum_{k=1}^{N_{anchor}} \mathcal{N}\!\left(\tau;\, \sqrt{\bar{\alpha}^{50}}\mathbf{a}_k,\, (1-\bar{\alpha}^{50})\mathbf{I}\right)
$$

20 个 mode 各自鼓包，每个鼓包里有 noise 探索空间。**这是和 pure Gaussian 最大的区别**——pure Gaussian 是单鼓包，mixture 是多鼓包。从多鼓包采样，每个 sample 天然带 mode label，reverse process 只需要在邻域内 refine，mode 之间不会互相 pull 塌成 mean。

### 3.3 推理只要 2 step

因为起点已经离 final distribution 很近了（每个 sample 都在某个 anchor 邻域），reverse process 只需要补 2 步 small refinement。Tab.4 显示 1 step 就能跑 87.9 PDMS，2 step 88.1，3 step 还是 88.1，**完全 saturate**。

这个 1-step 能力在 diffusion literature 里非常罕见。Consistency model（https://arxiv.org/abs/2303.01469）能 1-step 但需要特殊训练。这里 1-step 能 work 纯粹是因为 anchored starting point 太好了，reverse process 几乎是 identity。

---

## 4. Cascade Diffusion Decoder 为什么比 UNet 好

Transfuser$_{TD}$ 用 truncated diffusion + UNet，已经能把 step 从 20 降到 2，FPS 从 7 升到 27，PDMS 85.7。但 UNet 有个问题：它本来是给 2D image 设计的，处理 1D trajectory（8 waypoints × 2 dim = 16 个 scalar）是 overkill。UNet param 102M，里头大量卷积核在空转。

作者设计的 cascade diffusion decoder 是 transformer-based，直接对 trajectory query 做 cross-attention 交互。每层结构：

```
noisy trajectory τ_k^i (N_infer 条)
    ↓
spatial cross-attention ↔ BEV feature (deformable, 用 trajectory 坐标当 reference point)
    ↓
cross-attention ↔ agent/map queries (来自 perception module)
    ↓
FFN
    ↓
timestep modulation (encode diffusion step i)
    ↓
MLP → (confidence score s_k, trajectory offset Δτ_k)
    ↓
cascade 下一层 (refine)
```

两个 cascade layer，参数 60M，比 UNet 少 40%，PDMS 还涨 2.4（85.7→88.1）。这个设计的 intuition 是：

- **Trajectory 是 sparse signal**，用 deformable attention 让每个 waypoint 去 BEV feature 上对应位置采信息，比 UNet 全局卷积高效得多
- **Driving 需要 scene context interaction**，agent query 和 map query 提供 high-level 语义（前方有车、右边有 lane）
- **Cascade refine** 类似 Deformable DETR（https://arxiv.org/abs/2010.04159）的多层 decoder，第一层粗定位，第二层精修
- **Timestep modulation** 让 model 知道当前是 denoise 第几步，第 1 步该粗修、第 2 步该精修

这个架构工程上特别 clean，也特别适合 production：60M param 可以塞进 Orin 这种 edge GPU。

---

## 5. Loss 和 label assignment 的妙处

训练 loss 公式 (6)：

$$
\mathcal{L} = \sum_{k=1}^{N_{anchor}} \left[ y_k\, \mathcal{L}_{rec}(\hat{\tau}_k, \tau_{gt}) + \lambda\, \mathrm{BCE}(\hat{s}_k, y_k) \right]
$$

变量：
- $y_k$：第 $k$ 个 anchor 的 label，离 ground truth $\tau_{gt}$ 最近的那个 anchor 标 1（positive），其他标 0（negative）
- $\hat{\tau}_k$：model 预测的 denoised trajectory
- $\hat{s}_k$：model 预测的 confidence score
- $\mathcal{L}_{rec}$：L1 reconstruction loss
- $\mathrm{BCE}$：binary cross entropy
- $\lambda$：balance weight

这个 loss 设计的精髓在于 **one-to-one matching + scoring**：
- Reconstruction 只在 positive anchor 上算 → 梯度集中，避免 mean-seeking collapse
- BCE 在所有 anchor 上算 → model 学会「scene 对应哪个 anchor」的分类能力

这其实是 DETR-style Hungarian matching 的简化版，省了 Hungarian 算法，直接用 anchor 距离匹配。

---

## 6. 实验结果有多炸

NAVSIM navtest 上的对比（Tab.1）：

- **DiffusionDrive**: 88.1 PDMS, 45 FPS, 20 anchors, 60M param
- **Hydra-MDP-$\mathcal{V}_{8192}$-W-EP**: 86.5 PDMS（这个还是用了 rule-based scorer 蒸馏 + post-processing 的 tricked 版本）
- **VADv2-$\mathcal{V}_{8192}$**: 80.9 PDMS
- **Transfuser**: 84.0 PDMS, 60 FPS, no anchor, 56M param

DiffusionDrive 比 Transfuser +4.1 PDMS，比 Hydra-MDP-W-EP +1.6 PDMS（而且没用任何 post-processing），比 VADv2 +7.2 PDMS（anchor 数少 400×）。

nuScenes open-loop（Tab.7）：
- vs VAD：L2 error -20.8%，collision rate -63.6%，FPS 1.8×
- vs SparseDrive：L2 -0.04m，collision 持平

这个结果在 end-to-end driving 领域是 record-breaking 的。

---

## 7. 我的 intuition 和联想

### 7.1 为什么这个 work 本质上是「先验注入」

Diffusion model 的 reverse process 是一个 score matching 的迭代。从 pure Gaussian 开始，score 在早期 step 几乎只指向 data centroid，mode 信息要到后期才分叉。**Truncated + anchored 本质上是把 mode prior 注入到 starting distribution**，让 reverse process 跳过「从 centroid 分叉到 mode」这个最难的阶段，直接做「mode 内 refine」。

这跟 image generation 上的 SDEdit（https://arxiv.org/abs/2108.01073）思路类似——给个 prior 起点加小噪声再 denoise，比从 pure noise 生成容易得多。

### 7.2 Anchor cluster 是 implicit codebook

VADv2 用 8192 个 deterministic anchor 像 VQ-VAE 的 codebook。DiffusionDrive 用 20 个 anchor + Gaussian noise 像褪色的 VQ——anchor 是 codebook entry，noise 是 exploration。这种「sparse codebook + dense exploration」组合在 generative model 里很常见，比如 Stable Diffusion 的 text embedding + noise。

### 7.3 跟 Classifier-Free Guidance 的区别

Image diffusion 用 CFG（https://arxiv.org/abs/2207.12598）来 steer generation，公式 $\hat{\epsilon}_{cfg} = (1+w)\epsilon_\theta(x,c) - w\epsilon_\theta(x,\varnothing)$。但 driving 的 condition $z$ 是 high-dim BEV + agent + map，unconditional $z=\varnothing$ 语义不明。DiffusionDrive 通过 anchor selection + confidence scoring 实现 implicit guidance，避开了 CFG 的训练复杂度。

### 7.4 跟 TDPM 的对比

TDPM（https://arxiv.org/abs/2202.0965）也做 truncated diffusion，但起点是 implicit intermediate distribution（pre-trained model 的中间 latent）。DiffusionDrive 的 anchored Gaussian 是 **explicit domain prior**，用 K-Means 直接定 mixture component 中心。这是「数据驱动先验」vs「模型自学先验」的区别，前者更可控、更可解释。

### 7.5 可能的扩展方向

我脑子里蹦出来的几个联想：

1. **Instruction-conditioned anchor selection**：用 LLM/VLM 输出 navigation intent（「下个路口右转」），动态 activate 对应 anchor 子集。Anchor 本身是 universal codebook，inference 时按 intent 筛选。

2. **Learnable anchor embedding**：现在 anchor 是离线 K-Means 的 deterministic point，可以做成 learnable embedding，让 model 自己学 anchor。类似 DETR 的 learnable object query。

3. **Anchor 跨场景迁移**：Tab.9 已经证明 NAVSIM cluster 的 anchor 能迁到 CARLA。能不能做一个 universal anchor library？全世界 driving dataset 聚类出几百个 universal anchor，所有 model 共享。

4. **Risk-aware diffusion**：BEV occupancy 上加 risk field，让 score function 自动避开 high-risk region。类似 Diffusion-ES（https://arxiv.org/abs/2310.02447）但搬进 end-to-end。

5. **Latent space diffusion**：现在 diffusion 在 waypoint 空间，可以搬到 trajectory latent space，类似 Latent Diffusion（https://arxiv.org/abs/2112.10752），进一步降算力。

6. **RL fine-tuning**：现在是 pure imitation learning。能不能加 collision-free + rule-based reward 做 RL fine-tuning，让 confidence head 学到 rule knowledge？类似 Hydra-MDP 的 distillation 但用 RL。

7. **Multi-agent joint diffusion**：现在只扩散 ego trajectory。能不能同时扩散所有 agent trajectory，做 joint planning？类似 CTG（https://arxiv.org/abs/2305.00963）但搬进 end-to-end。

8. **Closed-loop reactive eval**：NAVSIM 是 non-reactive。DiffusionDrive 在 reactive eval 上表现如何？multimodal sampling 可能让 ego 更激进，需要 reactive 仿真验证。nuPlan 的 reactive challenge 是个好的 testbed。

9. **World model 整合**：diffusion 同时 predict future scene + ego trajectory，类似 GenAD（https://arxiv.org/abs/2407.15018）但 trajectory 用 truncated diffusion，scene 用 latent diffusion。

10. **Consistency model 替代**：2 step 已经 saturate，能不能用 consistency model（https://arxiv.org/abs/2303.01469）直接 1-step？省一个 cascade layer。

---

## 8. 关键 reference 链接

- **DiffusionDrive**: https://github.com/hustvl/DiffusionDrive
- **Diffusion Policy (RSS 2023)**: https://arxiv.org/abs/2303.04137
- **Transfuser**: https://arxiv.org/abs/2205.15997
- **VADv2**: https://arxiv.org/abs/2402.13243
- **Hydra-MDP**: https://arxiv.org/abs/2406.06978
- **NAVSIM**: https://arxiv.org/abs/2406.14149
- **SparseDrive**: https://arxiv.org/abs/2405.19620
- **DDIM**: https://arxiv.org/abs/2010.02502
- **TDPM**: https://arxiv.org/abs/2202.0965
- **Diffuser**: https://arxiv.org/abs/2205.09991
- **Deformable DETR**: https://arxiv.org/abs/2010.04159
- **VAD**: https://arxiv.org/abs/2303.12077
- **GenAD**: https://arxiv.org/abs/2407.15018
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **SDEdit**: https://arxiv.org/abs/2108.01073
- **Consistency Model**: https://arxiv.org/abs/2303.01469
- **Latent Diffusion**: https://arxiv.org/abs/2112.10752
- **CTG**: https://arxiv.org/abs/2305.00963
- **CARLA**: https://arxiv.org/abs/1711.03938
- **UniAD**: https://arxiv.org/abs/2212.10156

---

## 9. 最后一句

DiffusionDrive 真正漂亮的地方在于：它没有发明新理论，把 diffusion model、anchor prior、truncated schedule、transformer decoder 这些已有零件拼到了 driving 这个 task 的 sweet spot。anchor 解决 mode collapse，truncation 解决 speed，cascade decoder 解决 param/interaction。**三个工程 trick 每个都 5% 提升，叠起来 +4.1 PDMS @ 45 FPS**。这种 work 在 production autonomous driving 里比纯理论 paper 值钱得多，因为它把 generative model 真正塞进了 real-time closed-loop 评估的 fast lane。

---

# DiffusionDrive：把 Diffusion Policy 真正塞进 Real-time End-to-End Autonomous Driving

这篇 paper 解决的核心矛盾是：diffusion model 在 robotics 里 generation 质量很好但是 traffic scene 里太慢而且 mode collapse 严重。作者团队（HUST + Horizon Robotics）给出了一整套**truncated diffusion policy + cascade diffusion decoder**的设计，把 Transfuser 这种 single-mode regression 的 baseline 拔到 NAVSIM navtest 88.1 PDMS，同时 NVIDIA 4090 上 45 FPS，几乎是这一两年 end-to-end planning 里最漂亮的「换 head 翻盘」工作。

---

## 1. 为什么直接套 vanilla diffusion policy 在 driving 上不行

我们先把背景捋清楚。

Transfuser 的传统设计（arXiv:2205.15997, https://arxiv.org/abs/2205.15997）是用 ego-query 跑一个 MLP head 回归一条 trajectory $\tau = \{(x_t, y_t)\}_{t=1}^{T_f}$，$T_f=8$ (NAVSIM 上 4 秒 8 waypoints)。这种 single-mode regression 有一个根本问题：driving action distribution 是 multimodal 的——同样的 scene context，ego vehicle 可以选择直行、左换道、右换道、减速跟车等多条合理 trajectory。强行 regression 会落到一个 mean-seeking 的 mode collapse trajectory，几何上经常表现为「不可执行」的中间路线。

VADv2 (https://arxiv.org/abs/2402.13243) 用 4096~8192 个 clustered anchor 轨迹做 vocabulary，让模型打分再采样，把 action space 离散化。这条路有两个问题：(1) vocabulary size 决定 generalization ceiling，遇到 out-of-vocabulary 就 fail；(2) 8192 个 anchor 全部 forward 一次 scoring head 的算力开销很大。

Diffusion Policy (https://arxiv.org/abs/2303.04137) 在 robotics 上证明 generative 模型可以直接 capture multimodal action distribution。作者把 Transfuser 的 regression head 换成 UNet-based conditional diffusion，得到 Transfuser$_{DP}$。结果有惊喜也有惊吓：

- **惊喜**: PDMS 从 84.0 → 84.6，确实把 generative 的好处带进来了；
- **惊吓**: 
  - 推理需要 20 step denoising，每 step 6.5ms，FPS 从 60 掉到 7；
  - **mode collapse 严重**: 不同 Gaussian noise 起点 denoise 完会塌缩到重叠的 trajectory。论文里定义了一个 mode diversity score $\mathcal{D}$ 量化这个现象。

直觉上解释 mode collapse：pure Gaussian $\mathcal{N}(0, I)$ 是 unimodal 的，要用它经过 score matching 推到 multimodal data distribution，reverse process 早期 step 的 noise 都很大、几乎看不出 mode，于是不同 sample 早期都跟着 score 的均值走，到后期 mode 选择其实已经被 loss gradient 拉平。这就是为什么 DDPM 在 image 上常常见到 sample 都长得差不多的现象，在 driving 这种高自由度时序轨迹下尤其明显。

---

## 2. Truncated Diffusion Policy 的核心 idea

作者的 insight 一句话：**human driver 不是从 random noise 开始想的，而是从某些 prior driving pattern 开始根据 scene 调整**。这个 insight 落地成 anchored Gaussian distribution + truncated schedule。

### 2.1 Anchored Gaussian distribution 的构造

先用 K-Means 在 training set 上对所有 ground truth trajectory 聚类，得到 $N_{anchor}$ 个 cluster center $\{\mathbf{a}_k\}_{k=1}^{N_{anchor}}$，每个 $\mathbf{a}_k = \{(x_t, y_t)\}_{t=1}^{T_f}$。论文里默认 $N_{anchor}=20$（NAVSIM）/18（nuScenes），从 8192 缩到 20 是 **400× 减少**。

然后 truncated forward diffusion 写成：

$$
\tau_k^i = \sqrt{\bar{\alpha}^i}\, \mathbf{a}_k + \sqrt{1-\bar{\alpha}^i}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
\tag{4}
$$

变量含义：
- $\tau_k^i$: 第 $k$ 个 anchor 在 diffusion timestep $i$ 的 noisy sample（一条带噪 trajectory）
- $i \in [1, T_{trunc}]$, $T_{trunc} \ll T$，论文取 $T_{trunc}=50$, $T=1000$
- $\mathbf{a}_k$: 第 $k$ 个 cluster center，是 prior driving pattern
- $\bar{\alpha}^i = \prod_{s=1}^{i}(1-\beta^s)$: 累积保留系数，$i$ 越大 noise 越多
- $\beta^s$: noise schedule 第 $s$ 步加噪幅度（DDPM linear schedule）
- $\epsilon$: i.i.d. standard Gaussian noise

直观上，这个分布其实是 **mixture of Gaussians**：

$$
p_{init}(\tau) = \frac{1}{N_{anchor}} \sum_{k=1}^{N_{anchor}} \mathcal{N}\!\left(\tau; \sqrt{\bar{\alpha}^{T_{trunc}}}\mathbf{a}_k,\; (1-\bar{\alpha}^{T_{trunc}}) \mathbf{I}\right)
$$

当 $\bar{\alpha}^{T_{trunc}}$ 比较大（比如 0.8），每个 Gaussian 还集中在 anchor 附近，整体看起来是 $K$ 个鼓包；当 $T_{trunc} \to T$，鼓包互相融合退化为单一 Gaussian。**截断的位置就是 prior 信号保留 vs. exploration 自由度的 trade-off**。$T_{trunc}=50$ 这个值是经验选择，对应 $\bar{\alpha}^{50}$ 大概在 0.6~0.7 量级（线性 $\beta$ schedule 0.0001→0.02 时）。

### 2.2 训练目标

Diffusion decoder $f_\theta$ 接收 $N_{anchor}$ 条 noisy trajectory + conditional scene context $z$，输出每条轨迹的 confidence score $\hat{s}_k$ 和 denoised trajectory $\hat{\tau}_k$：

$$
\{\hat{s}_k, \hat{\tau}_k\}_{k=1}^{N_{anchor}} = f_\theta(\{\tau_k^i\}_{k=1}^{N_{anchor}}, z)
\tag{5}
$$

这里有一个关键的 label assignment：离 ground truth $\tau_{gt}$ 最近的那条 noisy trajectory 被标为 positive ($y_k=1$)，其他都是 negative ($y_k=0$)。这本质上是 DETR-style 的 Hungarian matching 简化版（直接用 anchor 距离代替 matching，省了 Hungarian 的计算）。

Loss：

$$
\mathcal{L} = \sum_{k=1}^{N_{anchor}} \left[ y_k\, \mathcal{L}_{rec}(\hat{\tau}_k, \tau_{gt}) + \lambda\, \mathrm{BCE}(\hat{s}_k, y_k) \right]
\tag{6}
$$

- $\mathcal{L}_{rec}$: L1 reconstruction loss，只对 positive sample 计算（one-to-one matching）
- $\mathrm{BCE}$: binary cross entropy，所有 anchor 都参与，让模型学会区分「这条 anchor 是否对应当前 scene 的正确 mode」
- $\lambda$: 平衡权重（论文里没显式给值，常用 1~5 范围）

这个 loss 设计的妙处在于：
- Reconstruction 只在 positive 上算 → 梯度集中，避免 average-to-mean collapse
- BCE 在所有 anchor 上算 → 强制 model 学习 anchor 的语义区分能力，对应于 inference 时的 top-1 选择

### 2.3 推理流程

训练时 $N_{anchor}=20$，但**推理时可以任意调整 $N_{infer}$**（inference flexibility，这是 conditional diffusion 的天然优势）。流程：

1. 从 anchored Gaussian distribution 采样 $N_{infer}$ 条 noisy trajectory
2. 经过 cascade diffusion decoder 2 step denoising（DDIM update rule）
3. 每步输出 confidence + trajectory offset
4. 最后按 confidence 排序取 top-1 作为 final output

注意：DDIM update rule 这里其实是把 model 预测的 $\hat{\tau}^0$ 当作 x0-prediction，再按 DDIM 公式：

$$
\tau^{i-1} = \sqrt{\bar{\alpha}^{i-1}}\hat{\tau}^0 + \sqrt{1-\bar{\alpha}^{i-1}}\epsilon
$$

每步用新的 $\epsilon$ sample 引入 stochasticity，但这里因为 anchor 已经是 prior 集中分布，noise 的探索空间小，所以 2 步就够。

---

## 3. Cascade Diffusion Decoder 架构详解

参考 https://arxiv.org/abs/2010.04159 (Deformable DETR) 和 https://arxiv.org/abs/2205.15997 (Transfuser)。

整体数据流：

```
N_infer 条 noisy trajectory τ_k^i
        │
        ├── spatial cross-attention (deformable) ↔ BEV/PV features
        │       (用 trajectory 当 spatial query)
        ├── cross-attention ↔ agent/map queries (from perception)
        ├── FFN
        ├── timestep modulation layer (encoding diffusion step i)
        └── MLP → outputs (s_k, Δτ_k)  
                       │
                       v
                  cascade 下一层
```

每一层 decoder 的输入是当前 denoising step 的 noisy trajectory，输出是对它们的 refinement，参数在两个 denoising step 之间共享。**Cascade layer 之间参数不共享**（类似 DAB-DETR/Deformable DETR 的 6 个 decoder layer）。

### 3.1 Spatial Cross-Attention

Trajectory 本身是 8 个 $(x_t, y_t)$ 坐标点，作者用 deformable attention 让 trajectory 点「去 BEV feature map 上对应位置采信息」。BEV feature 通常分辨率是 H×W（比如 50×50），每个 trajectory waypoint $(x_t, y_t)$ 映射到 BEV 上的一个 2D reference point，然后 deformable attention 在这个 point 周围采 4~8 个 offset 位置加权聚合。

这是为什么 Tab.3 ID-2（没有 spatial cross-attn）PDMS 直接掉到 55.1——trajectory 没有 BEV 几何信息就只能靠 perception 的 high-level query，对 metric 规划是不够的。

### 3.2 Agent/Map Cross-Attention

Trajectory query 再去跟 perception module 出来的 agent query（车、人、自行车）和 map query（lane, boundary, traffic light）做 cross-attention。这一步让 trajectory 知道「前方有车」「右边有 lane change 机会」。

### 3.3 Timestep Modulation

Diffusion step $i$ 怎么进 network？常见做法是 sinusoidal embedding + MLP → scale-shift 调制 feature。这样不同 denoising step 知道自己该「粗修」还是「细修」。

### 3.4 Confidence Head

每条 trajectory 输出一个 scalar $\hat{s}_k$，用 BCE 训练做 top-1 选择。这相当于 model 同时学了「生成 trajectory」和「评估 trajectory 在当前 scene 下好不好」两个 task。Inference 时取 top-1 这一步类似 VADv2 的 scoring 但更轻量（不需要 8192 个 anchor 全部 forward）。

### 3.5 Cascade 机制

Tab.5 显示 1→2 stage: 87.4→88.1 (+0.7)，2→4 stage: 88.1→88.2 (+0.1)。**2 stage 已经是收益边际递减点**，论文默认 2 stage。这种 refine 思路和 DAB-DETR / Cascade R-CNN 同源：第一阶段粗定位，第二阶段用第一阶段结果作 query 再 refine。

---

## 4. Mode Diversity Score $\mathcal{D}$

论文公式 (3)：

$$
\mathcal{D} = 1 - \frac{1}{N}\sum_{i=1}^{N} \frac{\mathrm{Area}\!\left(\tau_i \cap \bigcup_{j=1}^{N}\tau_j\right)}{\mathrm{Area}\!\left(\tau_i \cup \bigcup_{j=1}^{N}\tau_j\right)}
\tag{3}
$$

变量含义：
- $\tau_i$: 第 $i$ 条 denoised trajectory（在 BEV 平面上是多边形/曲线带）
- $N$: 采样轨迹数（论文用 20）
- $\bigcup_j \tau_j$: 所有轨迹在 BEV 上扫过的 union 区域
- 分子: $\tau_i$ 与 union 的 intersection area
- 分母: $\tau_i$ 与 union 的 union area（其实就是 union，因为 union ⊇ τ_i）
- 内层分数: $\tau_i$ 与 union 的 IoU
- 整体: 1 - 平均 IoU，越高代表每条轨迹越独特

这个 metric 不算太严格（一条非常古怪的轨迹也会有高 diversity），但配合 PDMS 一起看可以辨别 model 是不是「真实地生成了多 mode」而不是「乱生成」。从 Tab.2 可以看出，Transfuser$_{DP}$（vanilla DDIM）diversity 只有 11%，DiffusionDrive 是 74%——这说明 truncated + anchored 把 diversity 拉回去了。

---

## 5. Roadmap: 从 Transfuser 到 DiffusionDrive

Tab.2 是论文最关键的 ablation，把它当 roadmap 看最直观：

| Method | PDMS | Step time | Steps | Plan time | $\mathcal{D}$ | Param | FPS |
|---|---|---|---|---|---|---|---|
| Transfuser | 84.0 | 0.2ms | 1 | 0.2ms | 0% | 56M | 60 |
| Transfuser$_{DP}$ (vanilla DDIM) | 84.6 | 6.5ms | 20 | 130.0ms | 11% | 101M | 7 |
| Transfuser$_{TD}$ (truncated + UNet) | 85.7 | 6.9ms | 2 | 13.8ms | 70% | 102M | 27 |
| DiffusionDrive (truncated + cascade decoder) | **88.1** | 3.8ms | 2 | 7.6ms | **74%** | 60M | **45** |

三步贡献：
1. **Transfuser → Transfuser$_{DP}$**: 引入 vanilla diffusion policy，质量 +0.6 PDMS 但 +650× runtime
2. **Transfuser$_{DP}$ → Transfuser$_{TD}$**: truncated + anchored Gaussian，denoise step 20→2，diversity 11%→70%，PDMS +1.1，但 UNet 还是贵
3. **Transfuser$_{TD}$ → DiffusionDrive**: 用 cascade diffusion decoder 替代 UNet，param 102M→60M（-40%），PDMS +2.4，FPS 27→45

第三步其实是 architecture engineering 的胜利：UNet 原本是 2D image 模型，处理 1D trajectory (8 waypoints × 2 dims = 16) 其实是 over-engineered。换成 transformer-based cascade decoder 一举减少 param 又能更好和 BEV/agent/map context 交互。

---

## 6. 与 State-of-the-art 的对比

### 6.1 NAVSIM navtest (Tab.1)

NAVSIM 数据集（arXiv:2406.14149, https://arxiv.org/abs/2406.14149）是 nuPlan 子集，用 non-reactive closed-loop simulation + PDMS metric (PDM score)。

PDMS 是 5 个 sub-metric 的加权组合：
- **NC** (No at-fault Collision): 无责任碰撞率
- **DAC** (Drivable Area Compliance): 在可行驶区域内
- **TTC** (Time-to-Collision): 时间到碰撞
- **Comfort**: 加速度/jerk 平滑性
- **EP** (Ego Progress): 行驶进度

DiffusionDrive 全部刷新纪录：
- 88.1 PDMS vs Hydra-MDP-$\mathcal{V}_{8192}$-W-EP 的 86.5（+1.6，**注意**：Hydra-MDP-W-EP 是用 rule-based evaluator 蒸馏 + 加权 confidence post-processing 的 tricked 版本）
- vs Hydra-MDP-$\mathcal{V}_{8192}$ 原版 83.0（+5.1）
- vs VADv2-$\mathcal{V}_{8192}$ 80.9（+7.2）
- vs Transfuser 84.0（+4.1）
- anchor 数从 8192 降到 20（**400× 减少**）

### 6.2 nuScenes open-loop (Tab.7)

参考 SparseDrive (https://arxiv.org/abs/2405.19620) 的 recipe。在 ResNet-50 backbone 下：

| | L2 1s | L2 2s | L2 3s | Avg | Col 1s | Col 2s | Col 3s | Avg | FPS |
|---|---|---|---|---|---|---|---|---|---|
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 | 4.5 |
| SparseDrive | 0.29 | 0.58 | 0.96 | 0.61 | 0.01 | 0.05 | 0.18 | 0.08 | 9.0 |
| DiffusionDrive | 0.27 | 0.54 | 0.90 | **0.57** | 0.03 | 0.05 | 0.16 | 0.08 | 8.2 |

vs VAD: L2 -20.8%, Col -63.6%, FPS 1.8×。但 nuScenes 大部分是 trivial scenario，所以这个 open-loop 优势相对小，主要价值是验证 method 在不同 dataset 都能 fit。

---

## 7. Ablation 关键洞察

### 7.1 Denoising step 数 (Tab.4)

1/2/3 step: PDMS 87.9 / 88.1 / 88.1。**1 step 已经能跑出 87.9**，因为 anchored Gaussian 起点已经接近 final distribution。这个 1-step 能力非常罕见，正常 DDIM 即便 10 step 都会塌掉。本质原因是 truncated 后 reverse process 只需补 small noise。

### 7.2 Cascade stage (Tab.5)

1/2/4 stage: 87.4 / 88.1 / 88.2。2 stage 后 saturation。Param 59→60→65M。**默认 2 stage** 是 param-efficient 最优点。

### 7.3 $N_{infer}$ (Tab.6)

10/20/40: 84.9 / 88.1 / 88.2。**$N_{infer}=10$ 已经够用**，但 20 性价比最优。这个特性允许部署时根据硬件 budget 动态调整——高端 GPU 可以 $N_{infer}=40$，低端 10。

### 7.4 Driving prior 选择 (Tab.8)

这一组 ablation 极其关键，回答了「prior anchor 到底用什么」：
- Anchored Gaussian (20 anchor) 训练 + 推理: **88.1**
- Anchored Gaussian 训练 + extrapolated trajectory 推理（只采样 current status 的外推）: 81.3 (-6.8)
- Extrapolated trajectory 训练（单 anchor）+ 推理: 84.7 (-3.4)

**结论**: 用 single extrapolated trajectory 作 prior 会 fail 在 obstacle avoidance 和 turning 这种需要 mode switch 的场景（与 NAVSIM paper 里 ego-status planner 的 fail mode 一致）。**multi-anchor 是覆盖潜在 action space 的核心**。

### 7.5 Anchor source 泛化 (Tab.9)

把 NAVSIM 上 cluster 出来的 anchor 放到 CARLA Longest6 上训练 DiffusionDrive，结果 DS (Driving Score) 64.27 vs Transfuser 47.30（+16.97）。这证明 anchor 不是数据泄露，而是「真实 driving pattern 的覆盖」——NAVSIM 的 20 个 anchor 就能覆盖 CARLA 的 driving mode space。

---

## 8. 与相关工作的连接和 intuition

### 8.1 与 TDPM 的差异

TDPM (https://arxiv.org/abs/2202.0965, ICLR 2023) 也做 truncated diffusion，但它的起点是 implicit intermediate distribution（pre-trained model 学到的中间 latent）。DiffusionDrive 的 anchored Gaussian 是 **explicit driving prior**，用 K-Means cluster 直接定下 mixture component 中心。这是「引入 domain knowledge」vs「依赖 model 自学」的区别。

### 8.2 与 Diffusion Bridge 的关系

Diffusion Bridge (https://arxiv.org/abs/2304.11527) 是给定起点/终点的 diffusion，bridge 两端的 anchor。DiffusionDrive 可以看作是「multi-bridge」的特例：每个 anchor 是一个 prior mean，diffusion 把它 bridge 到 final mode。这给后续做「conditional anchor」（比如 navigation instruction conditioned anchor selection）留了空间。

### 8.3 与 Consistency Model / Flow Matching 的联系

论文 related work 提到 Flow Matching (https://arxiv.org/abs/2210.02747) 和 Rectified Flow (https://arxiv.org/abs/2209.03003)。这些方法理论上可以做到 1 step generation，但 driving scene 的 multimodality 难以从 pure Gaussian flow 过来。DiffusionDrive 走 anchor-truncated 路线，绕开 flow matching 的训练复杂度，工程友好。后续可以试 anchor-conditioned flow matching：flow 从 anchored Gaussian 流到 data。

### 8.4 与 Classifier-Free Guidance 的关系

Conditional diffusion 的 CFG（https://arxiv.org/abs/2207.12598）在 image gen 上很成功，公式：

$$
\hat{\epsilon}_{cfg} = (1+w)\epsilon_\theta(x, c) - w\epsilon_\theta(x, \varnothing)
$$

但 driving 上 conditional $z$ 是 high-dimensional BEV + agent + map，unconditional $z=\varnothing$ 不太有意义。DiffusionDrive 通过 anchor + score head 实现了 implicit guidance：anchor 提供 action prior，confidence head 提供 scene-conditioned scoring。

### 8.5 与 Diffuser / Janner et al. 的区别

Diffuser (https://arxiv.org/abs/2205.09991) 用 unconditional diffusion 做规划，靠 inpainting 把 state 固定成当前 observation。Driving 上 inpaint 是不现实的（ego state 在 BEV 上是单点），所以 conditional diffusion + scene context cross-attention 是必要的。DiffusionDrive 把 conditional interaction 做成 cascade decoder 而不是 UNet 上加 cross-attn，更适合 sparse query + dense BEV 的混合信号。

### 8.6 Hydra-MDP 的对比

Hydra-MDP (https://arxiv.org/abs/2406.06978) 也是 multimodal 但走 VADv2 的 8192 anchor + scoring。它加了一个 rule-based scorer 做蒸馏，让 scoring head 学到 rule knowledge。DiffusionDrive 不需要 rule-based supervision，纯 human demo 学习就能超 Hydra-MDP-W-EP（86.5）1.6 PDMS。这是 generative 模型比 discriminative scoring 在 coverage 上的根本优势。

### 8.7 与 GenAD 的关系

GenAD (https://arxiv.org/abs/2407.15018) 也想做 generative planning，用 VAE-based latent world model。DiffusionDrive 走的是显式 trajectory diffusion，更直接更可控。

---

## 9. 我对这篇 paper 的整体 intuition 总结

把它拆成几个层次：

1. **为什么 single-mode regression 不行**: Action distribution 本质 multimodal，强行 regression 落到 mode average，几何不可执行。

2. **为什么 vanilla diffusion 在 driving 上 fail**: Two reasons——pure Gaussian 起点离 data manifold 太远，需要 20+ step reverse；reverse 过程中不同 noise 的 mean-seeking 倾向导致 mode collapse。

3. **为什么 truncated + anchored 解决问题**:
   - Anchor cluster 提供 prior mean → 缩短 reverse process 距离 → 2 step 够用
   - Anchored Gaussian 是 mixture → sample 起点自带 mode diversity → 避免 collapse
   - Truncated schedule (50/1000) → 加噪幅度有限 → final exploration 在 anchor 邻域

4. **为什么 cascade decoder 比 UNet 好**:
   - Trajectory 是 sparse 1D signal (8 waypoints)，UNet 2D inductive bias 浪费
   - Driving 需要和 dense BEV + sparse agent query 交互，cross-attention 更直接
   - 参数从 102M 砍到 60M，FPS 从 27 飙到 45

5. **Confidence head 为什么重要**: Top-1 selection 用于 closed-loop eval（NAVSIM PDMS 取 top-1），同时训练时 one-to-one matching 避免模式塌缩。这其实是一种 implicit classifier guidance——anchor 提供 proposal，scene context + confidence 提供 selection。

6. **Inference flexibility 的工程价值**: 训练时 $N_{anchor}=20$，部署时可以根据硬件动态调 $N_{infer}$，这跟 VADv2 的 8192 fixed vocabulary 截然不同。这种 elasticity 在 production 自动驾驶车队（不同车型算力不同）非常有用。

---

## 10. 可能的扩展方向（hallucinate 一点）

读完这篇 paper 我有几个联想方向：

1. **Instruction-conditioned anchor**: 用 LLM/VLM 给出 navigation instruction，把 anchor cluster 按 intent condition 动态选择。比如「right turn」时只 activate 右转 anchor cluster。

2. **Anchor learning end-to-end**: 现在 anchor 是 K-Means 离线 cluster 出来的，可以做成 learnable embedding，让 model 自己学 anchor。

3. **Risk-aware diffusion**: 在 BEV occupancy 估计上加 risk field，让 diffusion process 的 score function 自动避开 high-risk region。类似 Diffusion-ES (https://arxiv.org/abs/2310.02447) 但搬到 end-to-end。

4. **DiffusionDrive + world model**: 用 diffusion 同时 predict future scene + ego trajectory，类似 GenAD 但 trajectory 用 truncated diffusion，scene 用 latent diffusion。

5. **Closed-loop reactive eval**: NAVSIM 是 non-reactive，agent 不反应。DiffusionDrive 在 reactive eval 上会怎样？multi-modal 的 trajectory 采样可能让 ego 更激进，需要 reactive 仿真验证。

6. **RL fine-tuning**: 现在是 imitation learning，能不能加 rule-based + collision-free reward 做 RL fine-tuning，让 confidence head 学到 rule knowledge，类似 Hydra-MDP 的 distillation 但用 RL。

7. **Anchor 跨场景 transfer**: Tab.9 已经证明 anchor 跨 dataset 可迁移。能不能做一个 universal anchor library？把全世界 driving dataset 聚类出几百个 universal anchor，所有 model 共享，类似 codebook 思路。

8. **Latent space diffusion**: 现在 diffusion 在 waypoint 空间，可以在 trajectory latent space 上 diffusion，类似 Latent Diffusion (https://arxiv.org/abs/2112.10752)，进一步降算力。

---

## 11. 关键参考链接

- **DiffusionDrive 主页**: https://github.com/hustvl/DiffusionDrive
- **Diffusion Policy (RSS 2023)**: https://arxiv.org/abs/2303.04137
- **Transfuser (TPAMI 2022)**: https://arxiv.org/abs/2205.15997
- **VADv2**: https://arxiv.org/abs/2402.13243
- **Hydra-MDP**: https://arxiv.org/abs/2406.06978
- **NAVSIM**: https://arxiv.org/abs/2406.14149
- **SparseDrive**: https://arxiv.org/abs/2405.19620
- **DDIM**: https://arxiv.org/abs/2010.02502
- **TDPM (Truncated Diffusion Probabilistic Models)**: https://arxiv.org/abs/2202.0965
- **Diffuser (ICLR 2022)**: https://arxiv.org/abs/2205.09991
- **Deformable DETR**: https://arxiv.org/abs/2010.04159
- **VAD (ICCV 2023)**: https://arxiv.org/abs/2303.12077
- **GenAD**: https://arxiv.org/abs/2407.15018
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **CARLA**: https://arxiv.org/abs/1711.03938

---

## 12. 一句话总结

DiffusionDrive 的真正贡献是把「prior anchor + truncated schedule」这一组合做到了 driving 这种 real-time + multimodal + open-world 场景的工程最优解：anchor 用 K-Means 极简获取、truncation 把 1000 step 压到 50、reverse 把 20 step 压到 2、UNet 换成 cascade transformer 把 param 砍半——每一步都不是发明新理论，而是把现有理论 engineering 到 NAVSIM 88.1 PDMS @ 45 FPS 这个 sweet spot。这种「直觉 + 工程」的 work 比纯理论 paper 在 production autonomous driving 里更有价值，因为它把 generative model 真正送进了 closed-loop 评估的 fast lane。
