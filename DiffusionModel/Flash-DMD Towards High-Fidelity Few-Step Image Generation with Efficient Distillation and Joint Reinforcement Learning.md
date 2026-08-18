---
source_pdf: Flash-DMD Towards High-Fidelity Few-Step Image Generation with Efficient
  Distillation and Joint Reinforcement Learning.pdf
paper_sha256: 078c52afb2565e3f6c521778f5190396625c33c3cfa4b9a930d7eee9a5525d1c
processed_at: '2026-08-18T13:07:35-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Flash-DMD

## 一句话总结

把 1000 步的 SDXL 压成 4 步还能打的 distillation 方法，DMD2 要烧 24000 步 batch=128，这 paper 用 batch=64 跑 1000 步（2.1% cost）就打平甚至超越。关键是发现 DMD2 把两个 loss 乱炖在一起浪费了算力。

---

## 问题出在哪

DMD2 训 student generator 时，每个 timestep 同时塞两个 loss：

- **DM loss**：让 student 输出分布对齐 teacher 输出分布，相当于"照着老师抄大方向"
- **Adversarial loss**：拿 student 输出跟 real image 比，逼它像真实照片，相当于"细节上别偷懒"

两个 loss 直接相加，gradient 在 loss landscape 上互相打架。更要命的是，DMD2 让 score estimator $\mu_{gen}$ 同时干两件事：追踪 student 分布 + 当 discriminator。一个人打两份工，必须 TTUR=5（student 走 1 步，score estimator 跑 5 步追）才稳定，train cost 直接爆炸。

---

## 三个关键观察

### 观察 1：不同 timestep 干的活不一样

4-step 模型的 4 个 timestep，任务分工很清楚：

- **前几步（high noise, low SNR）**：决定构图、姿态、全局结构。这时候像素都糊成噪声，谈"纹理细节"没意义，该学的是 teacher 的 ODE 轨迹大方向
- **最后一步（low noise, high SNR）**：细节、材质、颜色、光照。这时候该学的是怎么把 latent 渲染成真实照片

DMD2 的做法是在每个 timestep 都同时上 DM loss + adversarial loss。但 high-noise 时 adversarial loss 根本没信号（图像还是噪声，discriminator 看个寂寞），low-noise 时 DM loss 信号又很弱（score function 在低噪声区域陡峭不稳定）。

**Flash-DMD 的做法**：DM loss 只在 high-noise timestep 用，adversarial loss 只在 low-noise timestep 用。各干各的擅长活儿，互不打架。

---

### 观察 2：Pixel-space discriminator 早期就把 mode 钉死

DM loss 本质是 reverse KL，有 mode-seeking 倾向——容易塌到 sharp but low-diversity 的 mode。DMD2 用 latent-space discriminator 缓解，但 latent space 的判别信号偏 abstract，对 texture、local geometry 不敏感。

Flash-DMD 直接上 **Pixel-GAN**：discriminator backbone 用 frozen SAM vision encoder，在 pixel space 判真假。SAM 在 1B mask 上预训练，对 local geometric structure、fine texture 极其敏感。这样训练早期 generator 就被钉在 high-fidelity mode 上，没机会塌成 high-contrast + 无纹理的垃圾 mode。

---

### 观察 3：Score estimator 不用当 discriminator

DMD2 让 $\mu_{gen}$ 身兼两职是低效根源。Flash-DMD 让 $\mu_{gen}$ 只做一件事：用 diffusion loss 追踪 student 分布。再加一个 EMA trick：

$$\psi \leftarrow \lambda_{ema} \psi + (1 - \lambda_{ema}) \theta$$

每次 generator 更新完，直接把 generator 参数 $\theta$ 以 EMA 方式注入 score estimator $\psi$。等于参数空间的 soft copy，省去反复用 diffusion loss 拟合。结果 TTUR 从 5 降到 1 或 2 就稳了，train throughput 直接 2-5x。

---

## Stage 1 的人话版流程

```
sample 一个 timestep τ
if τ 是 high-noise:
    只算 DM loss：让 student 在这个 noisy latent 上的 score 对齐 teacher
else if τ 是 low-noise:
    用 backward simulation 把 student 输出推到 clean image x_0
    在 x_0 上 forward 加一点噪声 → x̂
    student 再 denoise x̂ → 喂给 Pixel-GAN discriminator
    只算 adversarial loss

每 TTUR 步：
    更新 generator (DM loss + adv loss)
    EMA 更新 score estimator

每步都更新：
    score estimator (diffusion loss only)
    discriminator (SAM-based Pixel-GAN hinge loss)
```

---

## Stage 2：为什么 previous RL 方法都翻车

HyperSD、PSO 这些 few-step RL 方法，做法是：训完 distillation 拿到 4-step generator，再用 reward model (ImageReward / PickScore) 做 preference optimization 修细节。

结果全翻车：
- **HyperSD + ImageReward**：生成像油画，overexposed
- **PSO + PickScore**：生成像塑料，oversmoothed

原因是它们在 **clean image** 上做 RL，gradient 只 backprop 到 low-noise timestep。Reward model 偏好某些 superficial feature（特定 color palette、smoothness），模型就在最后几步拼命讨好 reward，high-noise 步建立的 structure 被破坏。这叫 **reward hacking**。

---

## Flash-DMD 的 RL 解法

### Trick 1：在 high-noise phase 做 RL

Paper 里 Fig. 3 做了个实验：固定初始噪声，分别在 t=999 和 t=499 采样多个 candidate。发现 t=999 的样本在 layout、细节上 diversity 大得多；t=499 的样本基本已经定型，diversity 小。

**直觉**：high-noise phase 是"做决策"的阶段，layout 空间大；low-noise phase 是"渲染"阶段，大局已定。RL 应该在决策阶段发力，才能真正影响 composition，否则只在最后几步搞 surface feature 就成 reward hacking 了。

### Trick 2：用 LRM (Latent Reward Model)

普通 reward model (PickScore, ImageReward, HPSv2) 都在 pixel space 评分，需要 VAE decode。在 high-noise timestep 评 noisy latent 时它们全瞎——训练时没见过带噪声的 latent。

LRM 直接在 latent space 评分，输入 noisy latent + timestep，输出 preference score。它复用 pretrained diffusion model 的 native latent understanding，任何 timestep 都能评。

### Trick 3：Joint training，distillation 当 anchor

这是最关键的设计。不单独跑 RL，而是 RL loss 和 Stage 1 的 distillation loss 交替更新 generator。

```
每 5 次 RL 更新，配 1 次 distillation 更新 (5:1 ratio)
```

为什么 5:1 最优？Table 4 ablation：
- 1:1 → RL 信号太弱，没提升
- 2:1 → 略好
- 5:1 → 最佳平衡
- 10:1 → RL 太强，开始 reward hacking 苗头

**Intuition**：DM loss + Pixel-GAN 是一条 anchor rope，把模型拴在 teacher 分布 + 真实图像分布附近。RL loss 是 explore 信号，想让模型往 reward 偏好的方向走。两者交替，RL 把模型推出去探索，distillation 把模型拉回来别跑偏。这就是为什么 Flash-DMD 不会出 oil painting / oversmoothing。

---

## 实验数字直觉

### Stage 1 (Table 1, 4-step SDXL)

| 方法 | ImgRwd | Pick | HPSv2 | MPS | Cost |
|---|---|---|---|---|---|
| DMD2 | 0.8748 | 0.2309 | 0.2937 | 12.41 | 128×24k (100%) |
| Flash TTUR1-1k | **0.9509** | 0.2322 | 0.2968 | **12.67** | 64×1k (**2.1%**) |
| Flash TTUR2-8k | **0.9740** | 0.2327 | 0.2981 | **12.71** | 64×8k (8.3%) |

2.1% cost 意味着 DMD2 跑完一次，Flash-DMD 能跑 47 次。对算力有限的小组太友好了。

### Stage 2 (Table 3, 4-step SDXL RL)

| 方法 | Pick | MPS | GPU Hours |
|---|---|---|---|
| Hyper-SDXL | 0.2324 | 12.45 | 400 A100 |
| PSO-DMD2 | 0.2338 | 12.53 | 160 A100 |
| LPO-SDXL | 0.2342 | 12.58 | 92 A100 |
| Flash-DMD | **0.2346** | **12.84** | **12 H20** |

H20 比 A100 稍弱（~0.7-0.8x），换算成等效 A100 hours 大约 15-17 小时。HyperSD 用了 400 小时，Flash-DMD 用 15 小时，差 25x。而且 HyperSD 的 ImageReward 高是假象——油画 artifact 骗过了 ImageReward，但 MPS（多维人类偏好）低。

---

## 为什么这套设计能 work

把三个 trick 串起来看：

1. **Timestep decoupling** 解决了 loss 冲突。DM loss 和 adv loss 在各自擅长的 regime 发力，gradient 不再互相对消
2. **Pixel-GAN (SAM backbone)** 解决了 mode-seeking。早期就把 generator 钉在 high-fidelity mode
3. **Score estimator EMA** 解决了 dual-role 低效。TTUR 大降，throughput 起飞
4. **High-noise RL** 解决了 reward hacking 的战场选择问题。在决策阶段 explore 才有意义
5. **Joint training** 解决了 RL 稳定性。Distillation loss 当 anchor，防止 policy collapse

五个 trick 环环相扣，单独拎出来都不新（timestep-aware 在 Pose 里有过，SAM discriminator 在 ADM 里用过，EMA 是老 trick，DPO form 也是 standard），但组合在一起在 few-step distillation + RL 这个具体场景下，把 cost 砍了两个数量级还提了质量。这就是工程整合的价值。

---

## 一句话 takeaway

Distillation 不是无脑叠 loss，要按 timestep 分工；RL 不是无脑在 clean image 上 optimize，要在决策阶段发力并拿 distillation loss 当 anchor 拴着。把这两个"无脑"改掉，cost 砍 98%，质量还更好。

---

# Flash-DMD: 高效 Diffusion Distillation 与联合 RL 精修框架深度解析

这篇 paper 来自 Tencent、SJTU、HUST、CUHK，核心目标是用 < 3% DMD2 的训练成本，在 4-step inference 下达到 SOTA 的 text-to-image 生成质量。下面从 motivation、架构、公式、实验四个层面拆解，目标是 build intuition。

参考链接：
- DMD2 (NeurIPS 2024): https://arxiv.org/abs/2311.18828
- DMD (CVPR 2024): https://arxiv.org/abs/2310.02557
- SDXL: https://arxiv.org/abs/2307.01952
- SD3 / Rectified Flow: https://arxiv.org/abs/2403.03206
- SAM: https://arxiv.org/abs/2304.02643
- Hyper-SD: https://arxiv.org/abs/2410.06841
- PSO (ICLR 2025): https://openreview.net/forum?id=...
- LRM (LPO): https://arxiv.org/abs/2502.01051

---

## 1. Motivation: DMD2 究竟卡在哪里

DMD 系列 (Distribution Matching Distillation) 通过最小化 teacher 分布 $p_\tau$ 与 student 分布 $p_{gen}$ 之间的 reverse KL divergence，把多步 teacher 压成 few-step student。但 DMD2 在工程上有三处明显低效：

### 1.1 Gradient 冲突
DMD2 把 DM loss 与 adversarial loss 在每个 timestep 上做直接 sum：

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{DMD}} + \lambda \nabla_\theta \mathcal{L}_{\text{AdvGen}}$$

这两个 gradient 在高维 loss landscape 上指向不同 basin，简单相加导致优化方向漂移到 suboptimal state。作者实测：去掉 adversarial loss，纯 DM loss 会让 generator 跑到 high-contrast、无 fine-grained texture 的 mode（reverse KL 的 mode-seeking 本性，ADM paper 也讨论过 https://arxiv.org/abs/2507.18569）。

### 1.2 Score estimator 双重职责
DMD2 让 $\mu_{gen}^\psi$ 同时承担：
- 用 diffusion loss (Eq. 6) 追踪 student 分布 $p_{gen}$
- 用 adversarial loss (Eq. 7) 充当 discriminator

两个目标互相拉扯，必须用 TTUR=5（每更新 1 次 generator，更新 5 次 score estimator）才能勉强稳定，训练吞吐大幅下降。

### 1.3 DM loss 在 distillation 后期失效
DM loss 是 distribution-level 的对齐信号，对 pixel-level 细节、texture 没有直接约束，到了 distillation 后期无法继续指导细节学习。

---

## 2. Stage 1: Timestep-aware Decoupling + Pixel-GAN

### 2.1 核心观察
Few-step distilled model 的不同 timestep 负责不同任务：

| Timestep regime | SNR | Generator 主要任务 |
|---|---|---|
| High-noise (low SNR) | 低 | global composition, structure, ODE trajectory 对齐 |
| Low-noise (high SNR) | 高 | fine details, texture, color, photorealism |

这和 Pose (https://arxiv.org/abs/2508.21019) 在 video generation 中的观察一致：adversarial loss 在 high SNR 时 active，DM loss 在 high-noise regime 强。

### 2.2 Timestep-aware Loss 解耦

作者把训练 timestep 分成两段：

**High-noise segment**：只用 DM loss
$$\nabla_\theta \mathcal{L}_{\text{DMD}}^{\text{AT}} = -\mathbb{E}_{z,t}\left[ (s_\tau(\mathcal{G}_\theta(\cdot)) - s_{gen}(\mathcal{G}_\theta(\cdot))) \frac{d\mathcal{G}_\theta(\cdot)}{d\theta} \right]$$

变量解释：
- $\mathcal{G}_\theta(\cdot) = \mathcal{G}_\theta(x_t, t)$：student generator
- $s_\tau, s_{gen}$：teacher 和 student 的 score function，定义为 $s(x_t,t) = -\frac{x_t - \alpha_t \mu(x_t,t)}{\sigma_t^2}$ (Eq. 3)
- $\alpha_t, \sigma_t$：noise schedule 决定的标量
- $z \sim \mathcal{N}(0, \mathbf{I})$：初始 latent noise
- $t \sim \mathcal{U}(0, T)$：均匀采样的 timestep

**Low-noise segment**：只用 Pixel-GAN adversarial loss
通过 backward simulation $\mathcal{B}$ 把 generator 输出 propagate 到 clean image：
$$x_{t_1} = \mathcal{G}_\theta(x_t, t); \quad x_0 = \text{Detach}(\mathcal{B}(x_{t_1}, 0))$$

然后在 low-noise timestep $\hat{t}$ 上做 forward diffusion 得 $\hat{x}$，再算 generator adversarial gradient：
$$\nabla_\theta \mathcal{L}_{\text{AdvGen}}^{\text{TA}} = \mathbb{E}_{\hat{t}, \hat{x}} \left[ \log \mathcal{D}(\mathcal{V}(\mathcal{G}_\theta(\hat{x}, \hat{t}))) \frac{d\mathcal{G}_\theta(\cdot)}{d\theta} \right]$$

变量解释：
- $\mathcal{D}$：pixel-level discriminator（基于 SAM vision encoder）
- $\mathcal{V}$：VAE decoder，把 latent 解到 pixel space
- $\hat{t}$：low-noise timestep
- $\hat{x}$：从 $x_0$ 在 $\hat{t}$ 上 forward diffusion 得到的 noisy sample

**Intuition**：DM loss 在 noise 大时 gradient signal 清晰（因为有明确的 score direction），到 noise 小时 score 本身变得陡峭、不稳定；adversarial loss 反过来，需要 clean-ish image 才能判别 texture，在 noise 大时毫无意义。把两者按 timestep 分工，等于给每个 loss 配上它最擅长的 regime。

### 2.3 Pixel-GAN：基于 SAM 的 Discriminator

Discriminator 结构：
- Backbone：frozen SAM vision encoder（hierarchical features）
- Heads：多个 trainable discriminator head（2D 结构，沿用 ADM 设计）

Discriminator loss：
$$\mathcal{L}_{\text{AdvDisc}}^{\text{PG}} = \mathbb{E}_{x_{real}}[-\log \mathcal{D}_\omega(\cdot)] + \mathbb{E}_z[\log \mathcal{D}_\omega(\mathcal{V}(\cdot))]$$

为什么用 SAM？SAM 在 1B mask 上预训练，对 local geometric structure 和 fine-grained texture 极其敏感。Pixel-space 而非 latent-space 的对抗，能在训练早期就把 generator 钉到 high-fidelity mode，避免 DM loss 的 mode-seeking 导致 blur。

### 2.4 Stabilized Score Estimator + EMA

作者把 $\mu_{gen}^\psi$ 从 discriminator 角色中解放，只用 diffusion loss 训练：

$$\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{x_{t-1}, t, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \mu_{gen}^\psi(x_t, t) - \epsilon \|_2^2 \right]$$

变量解释：
- $\mu_{gen}^\psi(x_t, t)$：fake score estimator 预测的 noise
- $\epsilon$：真实添加的 Gaussian noise
- $\psi$：score estimator 参数

EMA 更新（关键 trick）：
$$\psi \leftarrow \lambda_{ema} \psi + (1 - \lambda_{ema}) \theta$$

变量解释：
- $\theta$：generator 参数
- $\lambda_{ema}$：EMA decay 系数
- $\psi$：score estimator 参数

这相当于让 score estimator 通过参数空间的 soft copy 来追踪 generator 的 distribution 演化，避免反复用 diffusion loss 重新拟合。结果：TTUR 从 DMD2 的 5 降到 1 或 2 就够稳定。Ablation (Fig. 6) 显示 EMA 在 ImageReward 和 PickScore 上后期有明显增益。

### 2.5 Stage 1 实验数据

Table 1 (SDXL 4-step)：
- DMD2: ImgRwd 0.8748, Pick 0.2309, HPSv2 0.2937, MPS 12.41, cost = 128×24k
- Flash-DMD TTUR1-1k: ImgRwd **0.9509**, Pick 0.2322, HPSv2 0.2968, MPS **12.67**, cost = 64×1k (**2.1%**)
- Flash-DMD TTUR2-8k: ImgRwd **0.9740**, MPS **12.71**, cost = 64×8k (8.3% DMD2)

Table 2 (SD3-Medium 4-step, LoRA)：
- Flash-DMD TTUR2-4k: ImgRwd 1.0193 > teacher SD3 1.0173 > SD3-Flash 0.8459
- 仅 32×4k 训练成本，Flow Matching 框架同样适用

---

## 3. Stage 2: 联合 Latent RL 修细节

### 3.1 Reward Hacking 的根因分析

HyperSD、PSO 等 few-step RL 方法在 clean image 上做 preference optimization，gradient 只 backprop 到 low-noise timestep。结果：
- HyperSD + ImageReward → overexposed + oil painting artifacts
- PSO + PickScore → oversmoothed images

本质：reward model 偏好某些 superficial features (color palette, smoothness)，gradient 集中在 low-noise 步导致 model 在这些步过拟合，丢掉 high-noise 步建立的 structure。

### 3.2 LRM (Latent Reward Model) 选择

Table 5 系统对比了主流 reward model：
- PickScore / ImageReward / MPS / HPSv2：pixel space, time-agnostic → 不能评 noisy latent
- LRM：latent space, time-aware → 能在任意 timestep 评 noisy latent，省去 VAE decode

LRM 用 pretrained diffusion model 的 native latent understanding，输入 noisy latent + timestep，直接输出 preference score。

### 3.3 High-noise Phase Stochastic Sampling

Fig. 3 实验：固定初始 noise，在 t=999（high-noise）采样 vs t=499（mid-noise）采样，前者 layout 与 detail diversity 明显更大。

结论：在 high-noise phase 做 stochastic sampling + reward 筛选，能有效 explore layout 空间，避免 reward hacking 把模型锁在低 diversity 的局部最优。

形式化：
- 给定 $z_t$，在 high-noise timestep 采样 $k$ 个候选 $\{z_{t-1}^1, ..., z_{t-1}^k\}$
- LRM 给每个打分，归一化后选最高/最低构造 win-lose pair $(z_t, z_{t-1}^w, z_{t-1}^l)$
- 训练集 batch 中 $t \in \{749, 999\}$，$k=4$

### 3.4 Preference Optimization Loss

$$\mathcal{L}_{rl} = -\mathbb{E}\left[ \log \sigma(\beta \mathcal{H}(w, l)) \right]$$

$$\mathcal{H}(w, l) = \log \frac{p_\theta(z_{t-1}^w | z_t, c)}{p_{ref}(z_{t-1}^w | z_t, c)} - \log \frac{p_\theta(z_{t-1}^l | z_t, c)}{p_{ref}(z_{t-1}^l | z_t, c)}$$

变量解释：
- $\sigma$：sigmoid 函数
- $\beta$：regularization 强度常数
- $p_\theta(\cdot | z_t, c)$：当前 policy (LCM scheduler) 的 backward transition probability
- $p_{ref}(\cdot | z_t, c)$：reference policy（RL 训练前的初始 generator）
- $w, l$：win / lose sample 的下标
- $z_t$：当前 timestep 的 noisy latent
- $c$：text prompt

这是 Bradley-Terry 形式的 DPO loss，关键在于 $\mathcal{H}(w,l)$ 是 log-likelihood ratio 的差，相当于在 latent 空间上拉近 win、推远 lose。

### 3.5 Joint Training：Distillation 作 Regularizer

最关键的设计：RL loss 不单独训，而是和 Stage 1 的 distillation loss (DM loss + Pixel-GAN) 一起 alternating update。Algorithm 1 中：

```
if iteration mod TTUR == 0:
    update G_θ with L_DMD + λ·L_adv
if FLAG == Stage2:
    sample K candidates at high-noise timesteps
    construct win-lose pairs via LRM
    update G_θ with L_rl
```

Table 4 ablation 显示 RL 与 DM loss 的最优 ratio 是 5:1（每 5 次 RL 更新配 1 次 distillation 更新）。

**Intuition**：DM loss 提供稳定、well-defined 的 distribution alignment 信号，相当于一条 anchor rope；RL loss 是 explore 信号，但容易把模型拽向 reward 偏好的局部最优。两者交替更新时，DM loss 周期性把模型拉回 teacher 分布附近，防止 policy collapse。这就是为什么 Flash-DMD 不会出现 oil painting 或 oversmoothing。

### 3.6 Stage 2 实验数据

Table 3 (4-step SDXL, RL comparison)：
- Hyper-SDXL: ImgRwd 1.085 (highest), 但 overexposed, 400 A100 hours
- PSO-DMD2: 160 A100 hours, oversmoothed
- LPO-SDXL: 92 A100 hours, 40 NFE (慢), CLIP 最高但 oversmoothed
- **Flash-DMD**: PickScore 0.2346 (best), MPS 12.84 (best), ImgRwd 1.0035, **12 H20 hours**

GPU 消耗对比：Flash-DMD 用 ~12 H20 小时达到 HyperSD 用 400 A100 小时的水平（H20 算力约为 A100 的 0.7-0.8 倍），实际成本差距 30x+。

---

## 4. Algorithm 1 流程图解

完整训练循环（Stage 1+2）：

```
Input: teacher μ_real, dataset D_real, generator G_θ,
       score estimator μ_φ, discriminator D_ω, 
       LRM R, timestep set S, high-noise set T_noisy

Init: G_θ ← μ_real, μ_φ ← μ_real, D_ω ← random heads

for iter = 1 to max_iter:
    z ~ N(0, I)
    τ_i ~ S
    x_real ~ D_real
    x_τi ← backwardSim(z, τk→τi)   # back-simulate noisy
    x_τ1 ← backwardSim(x_τi, τi→τ1) # back-simulate clean
    
    # Generator forward
    x ← G_θ(x_τi, τi)
    p_real, p_fake ← D_ω(V(x_τ1))  # pixel-space disc
    
    if iter % TTUR == 0:
        t_j ← T_noisy
        L_DMD ← DM_loss(μ_real, μ_ω, x, t_j)   # Eq.4
        L_adv ← AdvGen_loss(p_real)             # Eq.10
        L_Gθ ← L_DMD + λ·L_adv
        update G_θ with L_Gθ
        μ_φ ← EMA(θ, φ, λ_ema)                  # Eq.11
    
    # Update score estimator (diffusion loss only)
    x ← x.detach()
    t ~ U(0,1)
    x_t ← forwardDiff(x, t)
    L ← ||μ_φ(x_t, t) - x||²
    update μ_φ with L
    
    # Update discriminator
    L_Dω ← Hinge(p_real, p_fake)
    update D_ω with L_Dω
    
    # Stage 2: RL
    if FLAG == Stage2:
        s_pool ← iterSample(S_noisy, K)        # K=4 candidates
        s_win, s_loss ← filter(R(s_pool))      # LRM scoring
        L_rl ← PrefOpt(s_win, s_loss)          # Eq.12-13
        update G_θ with L_rl
```

---

## 5. 关键 Insight 总结

1. **Timestep 是 distillation 的 first-class citizen**：不同 timestep 的任务不同，loss 也应该不同。DMD2 的"全 timestep 叠加 loss"是低效的源头。

2. **Pixel-space 对抗在早期 anchor mode**：SAM-based Pixel-GAN 提供 local geometry prior，能在 mode-seeking 出现前把 generator 钉在 high-fidelity mode。

3. **Score estimator 单一职责**：解开 dual role 后，TTUR 大幅下降，训练 throughput 提升 2-5x。

4. **EMA 是 cheap distribution tracking**：用参数空间 soft copy 替代数据空间反复拟合，cost 几乎为零。

5. **High-noise phase 是 RL 的正确战场**：low-noise phase 做 RL 必然 reward hacking，因为 layout 已经被 high-noise phase 锁定；high-noise phase diversity 大，RL 能真正影响 composition。

6. **Distillation loss 是 RL 的 anchor**：joint training 比 post-training 好（Table 4：online 0.9808 vs post-train 0.9795），因为 DM loss 持续把模型拉回 teacher 分布，防止 reward-driven drift。

---

## 6. 局限与可能的扩展

- Stage 2 的 LRM 依赖 SDXL/SD1.5 backbone，换到 Flow Matching 模型（如 Flux https://github.com/black-forest-labs/flux）需要重新训 LRM
- 4-step 是当前 setting，1-step 是否仍稳定未充分验证（DMD2 在 1-step 上有结果，Flash-DMD paper 没展示）
- Pixel-GAN 的 SAM encoder 推理成本不低，虽然训练成本低，但 discriminator 更新开销仍在
- RL 阶段只用了 LRM，未尝试 VisionReward (https://arxiv.org/abs/2412.13746) 等更新的多维 reward model

如果你想进一步 build intuition，建议从两个角度切入：(a) 在 SDXL 上复现 TTUR2-4k 的 Stage 1 结果，验证 timestep decoupling 的 ablation；(b) 把 LRM 换成你自己训的 reward model，看 joint training 的 anchor 效果是否还成立。
