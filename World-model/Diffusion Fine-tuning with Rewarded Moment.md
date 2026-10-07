---
source_pdf: Diffusion Fine-tuning with Rewarded Moment.pdf
paper_sha256: 43a90af91595133ff160c1ace4938e4a3f1a4f0914bbcc9099a3202e6acc69be
processed_at: '2026-08-03T21:23:51-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RMMD

## 核心故事：一个 model 既要快又要听话

Imagine 你 train 了一个 diffusion model，生成质量屌爆了，but 要跑 1000 步才能出一张图——慢得要死。你做了 distillation 把它压到 8 步，but 现在你又想让它听你的话（比如生成 "red" 的图，或者天气预报更准）。问题是这两个 operation 互相打架：

- **先 distill 再 fine-tune**：fine-tune 时要 backprop through 8 步 sampling，memory 爆炸。如果只 backprop 最后一步（DRaFT-1 那样），gradient signal 很弱，只能改 high-frequency detail，改不了 content
- **联合 train**：reward 把 student 往 high-reward 方向推，student 输出就跑出 teacher 见过的 distribution，distillation signal 失效，model 逐渐崩坏

RMMD 的 insight：**phase 1 蒸馏时 train 的那个 auxiliary network 别扔**，它 learned student 的 conditional moment，freeze 住当 distributional anchor。Phase 2 fine-tune 时用它当 regularizer，让 student 既能追 reward 又不跑偏。

参考：[MMD paper](https://arxiv.org/abs/2411.17623)

---

## 为什么这事 hard：先讲清楚 baseline 的痛点

### DI++ 的问题：off-policy 的 cap

DI++ 的做法：拿 $\hat{x}_0$（student 在 pure noise $x_1$ 上的预测），re-noise 到 random timestep $s$，拿到 $x_s'$，然后用 score difference 做 gradient。

$$\nabla_\theta \mathcal{L}_{\text{DI++}} = \mathbb{E}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top \left(s_{\text{student}}(s, x_s') - s_{\text{teacher}}(s, x_s') - \lambda \nabla_{\hat{x}_0} R(\hat{x}_0)\right)\right]$$

变量：$\frac{\partial \hat{x}_0}{\partial \theta}$ 是 student output 对参数的 Jacobian；$s_{\text{student}}, s_{\text{teacher}}$ 是两 model 在 $x_s'$ 处的 score；$\lambda$ 是 reward weight。

两个 problem：
1. $\hat{x}_0$ 是从 re-noised **off-policy data point** $x_s'$ 预测的，所以 reward 被 dataset 的 reward ceiling 限制住——你 train data 里没见过的 high-reward region，student 永远摸不到
2. Student 被 reward 推得 distribution shift 之后，$p_t$（data marginal）不再 represent student 自己的 intermediate distribution，regularization 失效

### DRaFT 的问题：backprop through 多步

DRaFT 要 backprop through 整个 sampling chain（K 步）。Memory O(K) × activation。In practice 只能做 DRaFT-1（只 backprop 最后一步），但这样 gradient 只影响 high-frequency——你能让图变 "更红"，but 改不了 content（Figure 4 里 DRaFT-1 保留了 bird 形状，just 加了些红色 shadow/artifact）。

### HyperNoise 的问题：只改 noise input

HyperNoise 训一个轻量网络 perturb noise input $x_1$，让 output distribution shift 到 high-reward。But 因为 perturbation 发生在 pure noise level，只能 affect low-frequency content。Figure 4 里 Picasso reward 上 HyperNoise 只是把图染成蓝白色，没真的画出 Picasso 风格。

### Joint distillation + reward (RG-LCD, Hyper-SD)

这些方法 in principle 同时 optimize distillation + reward，但 reward 让 student 漂移出 teacher distribution，distillation signal 逐渐 invalidate。Fragile。

---

## RMMD 怎么 work：two-phase 的 mental model

### Phase 1：纯 MMD distillation，train 出 distributional anchor

Student $\Phi_\theta$ 用 8 步采样。同时 train 一个 auxiliary network $\Phi_{\theta_{\text{aux}}}$，它 learn 从 $x_s'$（re-noised sample at稍早的 timestep $s$）预测 student 的 first moment $m_{\text{student}}(s, x_s') = \mathbb{E}[\hat{x}_0 | x_s']$。

MMD loss：
$$\nabla_\theta \mathcal{L}_{\text{MMD}}(\theta) = \mathbb{E}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (m_{\text{student}}(s, x_s') - m_{\text{teacher}}(s, x_s'))\right]$$

变量：$m_{\text{teacher}}$ 是 teacher 在 $x_s'$ 处预测的 $\mathbb{E}[x_0 | x_s']$；$m_{\text{student}}$ 是 auxiliary network 预测的 $\mathbb{E}[\hat{x}_0 | x_s']$。

**Intuition**：在 DDPM posterior 是 Gaussian 的假设下，conditional first moment 完全 determine conditional distribution。所以 match first moment everywhere = match marginal everywhere。这就是 MMD work 的 reason。

Phase 1 结束后，freeze student 和 auxiliary network。这个 auxiliary network 现在是一个 **learned distributional anchor**——它知道 "phase 1 的 student 在每个 $x_s'$ 处应该输出什么"。

### Phase 2：on-policy reward fine-tuning，auxiliary 当 anchor

关键 move：不用 data marginal $p_t$，改用 student 自己 generate 的 $\tilde{x}_0$，re-noise 到 random timestep $t$，得到 $x_t^{\text{pol}} \sim p_{\text{noise}}(\cdot | \text{sg}[\tilde{x}_0])$。注意 stop-gradient 阻止 backprop through sampling chain。

然后 student 在 $x_t^{\text{pol}}$ 上做一次 denoising step，输出 $\hat{x}_0$。Loss combine 三项：

$$\mathcal{L} = \mathcal{L}_{\text{MMD}}^{\text{online}} - \lambda R + \lambda_{\text{reg}} \mathcal{L}_{\text{L2}}$$

具体 gradient：
$$\nabla_\theta \mathcal{L}_{\text{RMMD}} = \mathbb{E}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (m_{\text{student}}^{\text{frozen}}(s, x_s') - m_{\text{teacher}}^{\text{frozen}}(s, x_s') - \lambda \nabla_{\hat{x}_0} R(\hat{x}_0))\right]$$

L2 项：
$$\mathcal{L}_{\text{L2}}(\theta) = \mathbb{E}_{t, x_t^{\text{pol}}, \xi}\left[\|\Phi_\theta(x_t^{\text{pol}}, t, \xi) - \Phi_{\theta_0}(x_t^{\text{pol}}, t)\|^2\right]$$

变量：$\Phi_{\theta_0}$ 是 frozen phase-1 student（无 dropout noise $\xi$），$\Phi_\theta$ 是当前 student（有 dropout）。$\lambda_{\text{reg}} = \lambda / 2$。

**Three things going on simultaneously**：
1. **Reward term** $-\lambda R$：把 student 推向 high-reward region
2. **MMD term**：用 frozen anchor 检测 student 是否漂离 phase-1 distribution，pull back
3. **L2 term**：更直接地约束 student output 接近 phase-1 output，防止 catastrophic shift

### 为什么这三项都 necessary？

- 只用 reward：student 漂到 reward hacking（CLIP-red 全涂红）
- 只用 MMD：correct distributional constraint 但不够 tight，复杂 reward（如 Inception Score）可能 overshoot
- 只用 L2：会 collapse conditional variance（公式里看到，L2 penalize Tr(cov) + mean shift，如果 variance 是 learned 的会被压到 0）

实验 Figure 7 显示 MMD + L2 combination 在 IS reward 上 outperform 单独任一个。RMMD 用 dropout 产生 variance（not learned），所以 L2 的 collapse 问题不那么严重，但 MMD 仍提供额外 distributional constraint。

---

## On-policy 的 magic：为什么这么关键

### Off-policy vs on-policy 的 distribution mismatch

Off-policy：$x_t$ 来自 noisy real data $p_t$。Student 学的 conditional distribution $p_\theta(\hat{x}_0 | x_t)$ 是 evaluated on $p_t$。

On-policy：$x_t^{\text{pol}}$ 来自 noisy student sample $p_{\text{noise}}(\cdot | \tilde{x}_0)$，$\tilde{x}_0 \sim p_\theta$。Student evaluated on 它自己 generate 的 distribution。

Reward shift 之后，$p_\theta$ 已经和 $p_0$ 不一样了。Off-policy 仍在 $p_t$ 上 evaluate，相当于 train 和 inference distribution mismatch。On-policy 修复了这点。

### Stop-gradient 的妙处

$\tilde{x}_0$ 上 stop-gradient，意味着 gradient 只 flow through single denoising step $\Phi_\theta(x_t^{\text{pol}}, t, \xi) \to \hat{x}_0$，不 flow through 整个 K-step sampling chain。

Memory cost：O(1) activation，不是 O(K)。
Compute cost：每 step 要先 K-step sample 得到 $\tilde{x}_0$，所以 2× slowdown vs off-policy。But still 远 cheap 于 backprop through K steps。

GenCast 实验显示 on-policy +0.4% CRPS improvement, +3.8% win rate over off-policy。这 2× compute 换来 accuracy gain 是 worth 的。

---

## GenCast 应用：为什么这是这篇 paper 的 showpiece

### GenCast 是什么

[GenCast](https://arxiv.org/abs/2312.15796) 是 DeepMind 的 diffusion-based weather forecasting model。给定当前大气状态 $x^t$（包括温度 T、湿度 Q、位势 Z、风 U/V/W at 多个 pressure level，加上 surface variables），预测 12h 后的 $x^{t+\delta}$。

每个 state 是 $180 \times 360$ 网格 × 多个 variables × 多个 pressure levels = 5,313,600 dimensions。Teacher 用 59 NFE per 12h forecast，auto-regressive rollout 到 longer horizon（7.5 天需要 15 步）。

### CRPS：proper scoring rule

CRPS formula：
$$\text{CRPS}_r(\{\hat{x}^{\tau,i}\}_{i=1}^M, x^\tau) = \frac{1}{M}\sum_{i=1}^M |\hat{x}_{(r)}^{\tau,i} - x_{(r)}^\tau| - \frac{1}{2M(M-1)}\sum_{i,j} |\hat{x}_{(r)}^{\tau,i} - \hat{x}_{(r)}^{\tau,j}|$$

变量：$M$ 个 ensemble members $\hat{x}^{\tau,i}$，ground truth $x^\tau$，dimension $r$。
- 第一项：每个 member 到 ground truth 的 MAE（penalize bias）
- 第二项：member 之间的平均距离（penalize over-dispersion，鼓励 spread）

CRPS 在 forecast distribution = true distribution 时最小。所以优化 CRPS 不是 gaming metric，是 distributional alignment。

### 为什么 GenCast 需要 RMMD

Diffusion model 有 systematic under-dispersion：predict 的 ensemble spread 比 actual forecast error 小。Teacher 用 stochastic churn（$S_{\text{noise}}=1.05$）来 compensate，但 MMD-distilled model 不兼容 churn（Table 4：MMD + churn CRPS 反而从 0.82% 降到 0.11%）。

RMMD with CRPS reward 直接 fix 这个 under-dispersion：CRPS 的第二项鼓励 spread，student 学到 produce more diverse ensemble。Figure 5 显示 spread-skill ratio 从 MMD-only 的 under-dispersive 提升到接近 1（well-calibrated）。

### 结果：7.5× speedup + 93% variables 更准

| Model | CRPS improvement | Win rate | Speedup |
|---|---|---|---|
| Teacher | 0% | N/A | 1× (59 NFE) |
| Plain MMD | -1.32% | 4.9% | 7.5× (8 NFE) |
| MMD tuned ($\eta=0.5, \rho=100$) | 0.82% | 75.0% | 7.5× |
| RMMD offline | 1.11% | 89.2% | 7.5× |
| RMMD online | **1.51%** | **93.0%** | 7.5× |

7.5× speedup + 93% variables 比 teacher 更准 + better calibration。这是 strong evidence that RMMD 不 just distill，actually improve distributional modeling。

### Auto-regressive rollout 的 surprising gain

RMMD 只在 12h lead time optimize CRPS。But Figure 11 显示 improvement 随 lead time 增加：1-day 改进小，7-day 改进大。

If RMMD 只是 reward hacking next-state CRPS，auto-regressive rollout 应该 diverge（errors accumulate）。相反，longer horizon 改进更大说明 RMMD 修复了 transition distribution 的 systematic bias。这个 bias 在每步 rollout 都被 "corrected"，所以 compound effect 随 lead time 增加。

### Hyper-parameter $\eta$ 的深刻影响

DDPM posterior：
$$p_{\text{cond}}(x_s | x_t, \hat{x}_0) = \mathcal{N}(\alpha_s \hat{x}_0 + \sqrt{1-\alpha_s^2 - \gamma_{s,t}^2}\hat{\epsilon}, \gamma_{s,t}^2 I)$$

$$\gamma_{s,t} = \eta \frac{\sigma_s}{\sigma_t}\sqrt{1 - \frac{\alpha_t^2}{\alpha_s^2}}$$

变量：$\eta$ 控制 stochasticity（$\eta=1$ 标准 DDPM，$\eta=0$ deterministic），$\hat{\epsilon}$ 是 predicted noise，$\gamma_{s,t}$ 是 posterior std。

$\eta=0.5$ 把 stochasticity 减半，CRPS 从 -1.32% 跳到 +0.22%。$\rho$ 从 7 改到 100（uniform in log-SNR）又提升到 +0.82%。

Why？MMD 需要 $x_s'$ 是 non-deterministic function of $x_t, \hat{x}_0$。太多 stochasticity（$\eta=1$）让 $x_s'$ 跟 $x_t$ 关系太 random，auxiliary network 难学。太少 stochasticity（$\eta \to 0$）让 $x_s'$ 变 deterministic，MMD 退化。$\eta=0.5$ 是 sweet spot。

---

## ImageNet 实验：Pareto front 怎么读

### FID-Reward Pareto 的解读

对每个 $\lambda$ 值 fine-tune 一个 model，每 2500 steps evaluate 一次，画 (FID, Reward) 点。Pareto front 是 Pareto-optimal 点的集合（不能同时 improve FID 和 reward）。

Lower-left corner（FID 低 reward 低）= 保留 quality 但没怎么追 reward。Upper-right（FID 高 reward 高）= 追 reward 但 quality 崩。

**Better method = Pareto front 更 upper-left**（同样 FID 下 reward 更高，或同样 reward 下 FID 更低）。

### Figure 2：8-step RMMD vs 1-step DI++

8-step MMD-distilled model FID = 1.26（teacher 1.19）。1-step DI++ 起点 FID = 2.65。

8-step RMMD 在所有 reward 上 Pareto front 都 dominate 1-step DI++。这暗示 multi-step regime 有 inherent advantage：质量 ceiling 更高，reward fine-tuning 的 starting point 更好。

### Figure 3：vs DRaFT 和 HyperNoise

ImageNet 64 (8-step) 和 ImageNet 512 (2-step)：

- **DRaFT-1**（LoRA + L2 reg）：Pareto 较弱，因为只 backprop final step，只能改 high-freq
- **DRaFT-2**（backprop 2 steps）：reward hacking 严重，CLIP-red 时整张图变红
- **HyperNoise**：low-freq limit，Picasso/watercolor 这种 complex style reward 上只能 broad color shift
- **RMMD**：在 CLIP 和 IS 这种 neural-network-based reward 上优势最大，因为 single-step gradient on corrupted on-policy sample 能 affect full frequency range

### Figure 4 的 qualitative 对比

同样的 initial noise $x_1$ 和 random seed，CLIP alignment "red"/"Picasso"/"watercolor"：

- **DRaFT-1**：bird shape 保留，但加了红色 shadow / cubic shape / patchy texture（adversarial artifact exploiting CLIP feature）。Sharpness 下降
- **DRaFT-2**：distribution shift 严重，image quality 崩
- **HyperNoise**：Picasso 上只能 blue-white broad shift，watercolor 上只能 white shift。无法 capture 复杂 style
- **RMMD**：subtle modification 整合进 image，red reward 时红色融入自然（not 整张涂红），Picasso 时真的画出 cubist style（not just 颜色变化）

---

## Intuition：为什么 RMMD work？

### 1. Distributional anchor 的 reuse

Phase 1 train 的 auxiliary network 是 learned representation of "phase-1 student 在每个 $x_s'$ 处应该输出什么"。Phase 2 freeze 它当 anchor，任何 student shift 都被 $m_{\text{student}} - m_{\text{teacher}}$ detect。这比传统 KL on marginal distribution 更 tight，因为 matching conditional moment 比 matching marginal 更强。

### 2. On-policy 解决 distribution shift

Off-policy（DI++）用 $p_t$，student shift 后 $p_t$ 不 represent student intermediate distribution。On-policy 用 $p_{\text{noise}}(\cdot | \tilde{x}_0)$，$\tilde{x}_0 \sim p_\theta$，确保 regularization 在 student 自己 distribution 上 apply。Stop-gradient 保持 single-step gradient 的 memory efficiency。

### 3. Single-step backprop 的 frequency coverage

与 HyperNoise（只在 noise input perturb，局限 low-freq）不同，RMMD 在 corrupted on-policy sample $x_t^{\text{pol}}$ 上做 single denoising step。$x_t^{\text{pol}}$ 已包含 student 的 high-freq structure（因为 $\tilde{x}_0$ 是 full student sample），denoising step 修改这些 structure 可以 affect high-freq。这解释 RMMD 在 CLIP/Picasso 上 outperform HyperNoise。

### 4. Multi-step ceiling

8-step student FID = 1.26，1-step DI++ = 2.65。Reward fine-tuning 在更高 quality baseline 上 work 更好，Pareto front 起点更优。1-step 结构 cap 住 reward 优化 ceiling。

### 5. CRPS as proper scoring rule

GenCast 上 CRPS 不是 arbitrary reward。它 minimum when forecast = true distribution。所以 optimizing CRPS = distributional alignment = 真正修复 transition distribution 的 systematic bias（under-dispersion）。Auto-regressive rollout improvement 是 strong evidence：不是 reward hacking，是 real improvement。

---

## Limitations 的诚实承认

1. **Quality ceiling**：RMMD 不 improve FID during phase 2。质量上限由 phase 1 MMD 决定。所以 phase 1 必须 train 到最强（8-step, dropout, shifted schedule）
2. **On-policy cost**：每 step 要 K-step sample，2× slowdown vs off-policy。But still 远 cheap 于 backprop through K steps
3. **Differentiable reward**：无法直接 optimize black-box reward。需要 differentiable reward 或 surrogate

---

## Open questions 我觉得值得想

1. **Adaptive $\lambda$**：现在 fix $\lambda$，可以 Lagrangian method 自动 tune per training step
2. **Higher-order moment matching**：MMD 只 match first moment。Match covariance（second moment）可能更好 calibrate uncertainty，especially for weather
3. **Non-differentiable reward via surrogate**：用 reward model 学 differentiable approximation，RMMD framework 应该能 extend
4. **Theoretical connection break**：MMD regularization 在 $x_s'$ 来自 forward diffusion 时 recover DI 的 score-matching objective（[Salimans et al. 2024](https://arxiv.org/abs/2411.17623)），but on-policy 时 $x_s'$ 来自 conditional diffusion，这个 connection break。理论性质需要更深分析
5. **Long-horizon reward**：GenCast 只 optimize 12h CRPS，rollout 改进是 emergent property。If 直接 optimize rollout reward（e.g. 7-day CRPS），可能 further improve 但 cost 增加
6. **Architecture interaction**：dropout 在 RMMD phase 2 设为 0，stochasticity 全靠 on-policy sampling。其他 stochasticity source（stochastic depth, weight averaging）可能有不同 effect
7. **Mode coverage vs mode seeking**：KL 是 mode-seeking，CRPS 是 mode-covering。RMMD 的 MMD regularization 实际上 encourage 哪种 behavior 取决于 moment matching geometry

---

## Final takeaway

RMMD 的 elegance 在于：它不是 invent 全新 framework，而是 reuse phase 1 已经 train 出来的 auxiliary network 当 distributional anchor。这个 anchor 是 learned（not hand-crafted），tight（conditional moment matching），efficient（single-step gradient via stop-gradient）。

ImageNet 上 Pareto front dominate DRaFT/HyperNoise/DI++。GenCast 上 7.5× speedup + 93% variables 更准 + better calibration。这是 strong evidence that unified distillation + reward fine-tuning is practical, not just theoretical。

For scientific diffusion model post-training，RMMD 提供了 general paradigm：find a proper scoring rule（CRPS for weather, energy score for financial distribution, etc），use it as reward，RMMD handles the rest。这个 pattern 应该 generalize beyond weather。

参考链接：
- [RMMD paper (本篇)](https://arxiv.org) 
- [MMD (Salimans et al., 2024)](https://arxiv.org/abs/2411.17623)
- [DRaFT (Clark et al., 2024)](https://arxiv.org/abs/2309.17400)
- [HyperNoise (Eyring et al., 2025)](https://arxiv.org/abs/2502.00064)
- [DI++ (Luo, 2024)](https://arxiv.org/abs/2405.15771)
- [GenCast (Price et al., 2023)](https://arxiv.org/abs/2312.15796)
- [GenCast GitHub](https://github.com/google-deepmind/graphcast)
- [EDM (Karras et al., 2022)](https://arxiv.org/abs/2206.00364)
- [CRPS (Ferro, 2014)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2243)
- [DDPO (Black et al., 2024)](https://arxiv.org/abs/2305.13301)
- [DPOK (Fan et al., 2023)](https://arxiv.org/abs/2305.10401)
- [ReFL (Xu et al., 2023)](https://arxiv.org/abs/2310.18855)
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01969)
- [ERA5 (Hersbach et al., 2020)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803)
- [Simple Diffusion (Hoogeboom et al., 2023)](https://arxiv.org/abs/2306.16666)
- [Simpler Diffusion SiD2 (Hoogeboom et al., 2025)](https://arxiv.org/abs/2504.04497)
- [Reward-Instruct (Luo et al., 2025)](https://arxiv.org/abs/2501.01752)
- [Implicit Diffusion (Marion et al., 2025)](https://arxiv.org/abs/2412.16101)

---

# Rewarded Moment Matching Distillation (RMMD) 深度技术解析

## 1. 核心问题与动机

Diffusion post-training 面临两个正交的 challenge：**distillation**（加速 sampling）和 **reward fine-tuning**（alignment）。现有方法 either 串联执行（先 distill 再 fine-tune，memory-intensive due to multi-step backprop）或 联合执行（reward maximization 让 student 漂移出 teacher distribution，invalidating distillation signal）。RMMD 的核心 insight 是：**把 distillation loss 重新 purposing 为 on-policy KL regularization**，让两个 phase 在一个 principled framework 里 mutually constrain。

参考链接：
- [Moment Matching Distillation (Salimans et al., 2024)](https://arxiv.org/abs/2411.17623)
- [DRaFT (Clark et al., 2024)](https://arxiv.org/abs/2309.17400)
- [HyperNoise (Eyring et al., 2025)](https://arxiv.org/abs/2502.00064)
- [Diff-Instruct (Luo et al., 2023)](https://arxiv.org/abs/2310.19044)
- [GenCast (Price et al., 2023)](https://arxiv.org/abs/2312.15796)

---

## 2. Background：Diffusion 与 Distillation 的 mathematical scaffold

### 2.1 Forward process

给定 clean data $x_0 \in \mathbb{R}^d \sim p_0$，forward process 按 schedule corrupt：

$$x_t = \alpha_t x_0 + \sigma_t \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_d)$$

变量解释：
- $x_t$：timestep $t \in (0,1]$ 时的 corrupted state
- $\alpha_t, \sigma_t$：signal / noise coefficient，满足 $\alpha_0=1, \sigma_0=0$（clean），$\alpha_1=0, \sigma_1=1$（pure noise）
- $\text{SNR}_t = \alpha_t^2 / \sigma_t^2$：signal-to-noise ratio，monotonically decreasing in $t$

Score function 定义：
$$s(t, x) = \nabla_x \log p_t(x) = \frac{\alpha_t \mathbb{E}[x_0 | x_t = x] - x}{\sigma_t^2}$$

这告诉我们：score 完全 determined by conditional first moment $\mathbb{E}[x_0 | x_t]$，而 diffusion training 本质上就是在 learn 这个 moment（通过 denoiser $\Psi(x_t, t) \approx \mathbb{E}[x_0 | x_t]$）。

### 2.2 DDPM posterior sampling

Inference 时 iterate $K$ 步，每步从 conditional posterior 采样：
$$x_{t-\delta} \sim p_{\text{cond}}(x_{t-\delta} | x_t, \hat{x}_0^{(t)}) = \mathcal{N}(\mu_t(x_t, \hat{x}_0^{(t)}), \Sigma_t)$$

其中 $\mu_t, \Sigma_t$ 由 $\alpha_t, \alpha_{t-\delta}, \sigma_t, \sigma_{t-\delta}$ 决定。DDPM 的 stochasticity（$\eta$ 参数）控制这步的 noise 注入，在 GenCast 实验里 $\eta=0.5$ 起到关键作用。

### 2.3 Distributional distillation 的 general formulation

Student $\Phi_\theta(x_t, t, \xi)$（$\xi$ 是 auxiliary noise，e.g. dropout）需要匹配 teacher posterior：

$$\hat{x}_0 | x_t \sim p_{\text{teacher}}(\cdot | x_t)$$

If exact equality at $t=1$，则 one-step student 完美 match teacher。否则 multi-step student 用 $K_{\text{student}} \ll K$ 步采样。

---

## 3. 三种 baseline 方法：DI, DI++, MMD 的 gradient 比较

### 3.1 Diff-Instruct (DI)

固定 $t=1$，student 单步生成 $\hat{x}_0 \sim p_\theta(\hat{x}_0 | x_1)$，然后 re-noise 到随机 $s$：
$$x_s' \sim p_{\text{noise}}(x_s' | \text{sg}[\hat{x}_0])$$

Loss 是 integral KL divergence：
$$\mathcal{L}_{\text{DI}}(\theta) = \mathbb{E}_s\left[w(s) \text{KL}(p_\theta(x_s') \| p_{\text{teacher}}(x_s'))\right]$$

Gradient 利用 score difference：
$$\nabla_\theta \mathcal{L}_{\text{DI}}(\theta) = \mathbb{E}_{s, x_s'}\left[w(s) \left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (s_{\text{student}}(s, x_s') - s_{\text{teacher}}(s, x_s'))\right]$$

变量解释：
- $w(s)$：timestep-dependent 权重
- $s_{\text{student}}, s_{\text{teacher}}$：在 $x_s'$ 处的 score function
- $\frac{\partial \hat{x}_0}{\partial \theta}$：Jacobian of student output w.r.t. parameters

**Intuition**：这个 gradient 把 student 推向 teacher 的 score，但只在 $t=1$（pure noise）做 student prediction，single-step generation 质量受限。

### 3.2 DI++ (Diff-Instruct++)

在 DI 基础上加 differentiable reward term：

$$\nabla_\theta \mathcal{L}_{\text{DI++}}(\theta) = \mathbb{E}\left[w(s) \left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top \left(s_{\text{student}}(s, x_s') - s_{\text{teacher}}(s, x_s') - \lambda \nabla_{\hat{x}_0} R(\hat{x}_0)\right)\right]$$

$\lambda$ 是 reward 与 KL regularization 的 trade-off weight。

**问题**：$\hat{x}_0$ 由 re-noised off-policy data point $x_s'$ 预测，fine-tuning 被 dataset cap 住；且 student distribution shift 后 $p_t$ 不再近似 student intermediate distribution。

### 3.3 Moment Matching Distillation (MMD)

扩展到 arbitrary timestep $t \sim \text{Uniform}[0,1]$，匹配 generalized moments：

$$\mathcal{L}_{\text{MMD}}(\theta) = \mathbb{E}_{t, s, x_t, \hat{x}_0, x_s'}\left[\hat{x}_0^\top \text{sg}(m_{\text{student}}(s, x_s') - m_{\text{teacher}}(s, x_s'))\right]$$

Gradient（因为 stop-gradient on $x_s'$）：
$$\nabla_\theta \mathcal{L}_{\text{MMD}}(\theta) = \mathbb{E}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (m_{\text{student}}(s, x_s') - m_{\text{teacher}}(s, x_s'))\right]$$

变量解释：
- $m_{\text{teacher}}(s, x_s') = \mathbb{E}_{p(x_0 | x_s')}[x_0 | x_s']$：teacher 预测的 first moment
- $m_{\text{student}}(s, x_s') = \mathbb{E}_{p_\theta(\hat{x}_0 | x_s')}[\hat{x}_0 | x_s']$：auxiliary network 预测的 student first moment
- $s \in [t - \delta_{\text{student}}, t)$：稍早于 $t$ 的 timestep

**关键性质**：当 $m_{\text{student}} = m_{\text{teacher}}$ everywhere，student 与 teacher marginals coincide。这意味着 MMD matching first moment 已经足够 enforce distributional equality（在 DDPM posterior 的 Gaussian 假设下）。

Auxiliary network training loss（Appendix A.1）：
$$\mathcal{L}_{\text{auxiliary}}(\theta_{\text{aux}}) = \mathbb{E}_{t, \hat{x}_0, x_s'}\left[\|\Phi_{\theta_{\text{aux}}}(x_s') - \hat{x}_0\|^2 + \|\Phi_{\theta_{\text{aux}}}(x_s') - \Phi_{\theta_0}(x_s')\|^2\right]$$

第一项是 regression to student output，第二项是 L2 regularization to initial weights（防止 auxiliary 漂移）。

---

## 4. RMMD 的核心构造

### 4.1 From off-policy to on-policy 的关键 jump

**Naive baseline**（直接在 MMD gradient 加 reward term）：
$$\nabla_\theta \mathcal{L}_{\text{naive}}(\theta) = \mathbb{E}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (m_{\text{student}}(s, x_s') - m_{\text{teacher}}(s, x_s') - \lambda \nabla_{\hat{x}_0} R(\hat{x}_0))\right]$$

这有两个 problems：
1. $\hat{x}_0$ 由 re-noised off-policy data point $x_s'$ 预测，需要 extra dataset，且 reward 被 data cap 住
2. Student distribution shift 后，$p_t$（data marginal）becomes poor approximation of student intermediate distribution

**On-policy 解决方案**：从 $p_t$ 改为从 noisy on-policy samples 采样：
$$x_t^{\text{pol}} \sim p_{\text{noise}}(x_t | \text{sg}[\tilde{x}_0]), \quad \tilde{x}_0 \sim p_\theta$$

其中 $\tilde{x}_0$ 由 student 用 $K_{\text{student}}$ 步采样得到，stop-gradient 阻止 backprop through 整个 sampling chain。

Reward objective 变成：
$$\mathcal{L}_{\text{Reward}}(\theta) = \mathbb{E}_{t, x_t^{\text{pol}}, \xi}\left[R(\Phi_\theta(x_t^{\text{pol}}, t, \xi))\right] \simeq \mathbb{E}_{\tilde{x}_0 \sim p_\theta}[R(\tilde{x}_0)]$$

**Intuition**：这个 approximation 直接优化 student 的 expected reward，approximation quality 取决于 distilled model 的质量（Eq.(1)）。Stop-gradient 让 gradient 只 flow through single denoising step，避免 multi-step backprop 的 memory cost。

Gradient：
$$\nabla_\theta \mathcal{L}_{\text{Reward}}(\theta) = \mathbb{E}_{\hat{x}_0}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top \nabla_{\hat{x}_0} R(\hat{x}_0)\right]$$

### 4.2 RMMD 完整 objective

Combining on-policy reward gradient 与 on-policy MMD regularization：

$$\nabla_\theta \mathcal{L}_{\text{RMMD}}(\theta) = \mathbb{E}_{t, s, x_t^{\text{pol}}, \hat{x}_0, x_s'}\left[\left(\frac{\partial \hat{x}_0}{\partial \theta}\right)^\top (m_{\text{student}}(s, x_s') - m_{\text{teacher}}(s, x_s') - \lambda \nabla_{\hat{x}_0} R(\hat{x}_0))\right]$$

注意：$m_{\text{teacher}}$ 可以 replace 为 **frozen auxiliary model**（在 phase 1 distillation 学到的），无需再 query teacher。

### 4.3 L2 regularization 的额外约束

$$\mathcal{L}_{\text{L2}}(\theta) = \mathbb{E}_{t, x_t^{\text{pol}}, \xi}\left[\|\Phi_\theta(x_t^{\text{pol}}, t, \xi) - \Phi_{\theta_0}(x_t^{\text{pol}}, t)\|^2\right]$$

$\Phi_{\theta_0}$ 是 frozen MMD-distilled model（无 noise $\xi$）。

**为什么 L2 + MMD 都要？** L2 alone 在 conditional variance 存在时会 reduce diversity：
$$\mathbb{E}[\|\hat{x}_{0,\theta} - \hat{x}_{0,\theta_0}\|^2 | \tilde{x}_t] = \text{Tr}(\text{cov}[\hat{x}_{0,\theta} | \tilde{x}_t]) + \text{Tr}(\text{cov}[\hat{x}_{0,\theta_0} | \tilde{x}_t]) + \|\mathbb{E}[\hat{x}_{0,\theta} | \tilde{x}_t] - \mathbb{E}[\hat{x}_{0,\theta_0} | \tilde{x}_t]\|^2$$

如果 variance 是 learned（不是 dropout），L2 会 collapse variance。RMMD 用 dropout 产生 stochasticity，避免了这点；但 MMD 提供额外的 distributional constraint，实验显示 MMD + L2 combination 在 Inception Score reward 上 outperform 单独任一个（Figure 7）。

### 4.4 Multi-sample reward 扩展

对于 CRPS 这种 multi-sample reward，sample $(\hat{x}_0^{(i)})_i$ 并 combine associated MMD regularization losses with $R(\hat{x}_0^{(1)}, \hat{x}_0^{(2)}, \ldots)$。这 enable 了 GenCast 应用。

---

## 5. Architecture 与 training pipeline

### 5.1 Two-phase procedure

**Phase 1: Pure MMD distillation**
- Train student via MMD without reward
- 用 8-step sampling（optimal speed-quality trade-off）
- Auxiliary network 同时 trained to predict $m_{\text{student}}$
- Freeze student as stable distributional reference

**Phase 2: On-policy reward fine-tuning**
- 用 same hyper-parameters（batch size, optimizer, LR）
- 10,000 steps, LR decay to zero
- Loss: $\mathcal{L} = \mathcal{L}_{\text{MMD}}^{\text{online}} - \lambda R + \lambda_{\text{reg}} \mathcal{L}_{\text{L2}}$
- Fix $\lambda_{\text{reg}} = \lambda / 2$
- Sweep $\lambda$ 得到 Pareto front

### 5.2 On-policy sampling 的两种 variant

**Continuous sampling**：从 $p_{\text{noise}}(x_t | \text{sg}[\tilde{x}_0])$ 在任意 continuous $t$ 采样

**Discrete/ReFL sampling**：early stop denoising at random step $t_{\text{stop}} \in \{0, \delta, \ldots, (1-K_{\text{student}})\delta\}$
- 优点：training timestep 与 inference timestep 完全一致
- 缺点：限制 MMD 在 continuous timestep 上的 generalization

Figure 6 显示两者性能接近，discrete 在 CLIP reward 稍好，continuous 在 Inception Score 稍好。

### 5.3 Architecture details

| Dataset | ImageNet 64 | ImageNet 512 | ERA5 |
|---|---|---|---|
| Backbone | U-ViT | U-ViT | Graph Transformer |
| Teacher | SiD2 | SiD2 | GenCast |
| MMD dropout | 0.1 | 0.1 | 0.0 |
| RMMD dropout | 0.1 | 0.0 | 0.0 |
| MMD steps | 50k | 50k | 300k |
| RMMD steps | 10k | 10k | 300k |
| Batch size | 2048 | 2048 | 16 |
| Hardware | 16 TPU-v5 | 16 TPU-v5 | 16 TPU-v6 |
| LR | 1e-5 | 1e-5 | 1e-7 |

Dropout 在 RMMD phase 2 在 high-resolution 设为 0，这表明 fine-tuning 时 stochasticity 由 on-policy sampling 提供，不再需要 network 内部 dropout。

---

## 6. ImageNet 实验：FID-Reward Pareto fronts

### 6.1 8-step vs 1-step 的 multi-step 优势

**Table 2：MMD distillation quality**
| Setting | I64-8steps | I64-2steps | I512-2steps |
|---|---|---|---|
| MMD (paper) | 1.24 | 3.86 | - |
| MMD (ours) | 1.35 | 2.0 | 9.7 |
| MMD + dropout | 1.37 | 1.66 | 5.4 |
| MMD + dropout + shift | 1.26 | 1.4 | - |

Teacher 在 I64-8steps 的 FID 是 1.19，RMMD 的 MMD 起点是 1.26，extremely close。

Figure 2 显示 8-step RMMD consistently outperform 1-step DI++（FID 2.65 起点）across all reward functions。这验证了 multi-step regime 的 inherent advantage。

### 6.2 Reward functions

- **Black-and-white**：pixel-wise distance to grayscale version
- **Laplacian smoothing**：average distance to 4 neighbors
- **IS (Inception Score)**：直接用 IS 作为 reward
- **CLIP-red**：CLIP alignment with "red"

### 6.3 与 multi-step competitor 比较

Figure 3 比较 RMMD vs HyperNoise vs DRaFT 在 64×64 (8-step) 和 512×512 (2-step)：
- RMMD 在 neural-network-based reward（CLIP, IS）上优势最大
- DRaFT-1（只 backprop final step）只产生 high-frequency artifact
- DRaFT-2（backprop whole 2-step chain）prone to reward hacking（CLIP-red 时全部涂红）
- HyperNoise 局限在 low-frequency modification（Picasso/watercolor reward 时只能 broad color shift）

### 6.4 Qualitative analysis (Figure 4)

- **DRaFT-1**：preserves semantic content but introduces adversarial artifact（red shadow, cubic shape, patchy texture），reduces sharpness
- **DRaFT-2**：drastic distribution shift，图像质量严重 deteriorate
- **HyperNoise**：struggle with complex style reward，default to broad color shift
- **RMMD**：成功 integrate subtle modification without significantly deviating from original distribution

---

## 7. GenCast 应用：scientific domain 的 scaling

### 7.1 GenCast 背景

GenCast 学习 transition distribution $p(x^{t+\delta} | x^t, x^{t-\delta})$，$\delta = 12h$。变量包括：
- 6 atmospheric variables at multiple pressure levels: T, Q, Z, U, V, W
- Pressure levels: 50 hPa to 1000 hPa
- Surface variables: t2m, msl, 10u, 10v
- Resolution: $1^\circ$，地图 $180 \times 360$
- Total dimension: **5,313,600**

Teacher 需要 59 NFE per 12h forecast，auto-regressive rollout 到 longer horizon。

### 7.2 CRPS as reward

Continuous Ranked Probability Score：
$$\text{CRPS}_r(\{\hat{x}^{\tau,i}\}_{i=1}^M, x^\tau) = \frac{1}{M}\sum_{i=1}^M |\hat{x}_{(r)}^{\tau,i} - x_{(r)}^\tau| - \frac{1}{2M(M-1)}\sum_{i=1}^M\sum_{j=1}^M |\hat{x}_{(r)}^{\tau,i} - \hat{x}_{(r)}^{\tau,j}|$$

变量解释：
- $\hat{x}^{\tau,i}$：第 $i$ 个 forecast member（共 $M$ 个）
- $x^\tau$：ground truth
- $r$：dimension index
- 第一项：MAE of individual members
- 第二项：spread between members（鼓励 diversity）

**Intuition**：CRPS 在 forecast distribution 完全 match ground truth distribution 时最小。第一项惩罚 bias，第二项鼓励 spread。Under-dispersed forecast（variance 太小）会因第二项小而被惩罚，over-dispersed 会因第一项大而被惩罚。

### 7.3 Training pipeline

- Teacher: 500k steps, batch 128
- MMD: 300k steps, batch 16（7.5% of initial compute）
- RMMD: 300k steps, batch 16, $\lambda = 0.3$, 2-sample variant
- Final: 8-step DDPM sampling（vs 59 for teacher）= **7.5× speedup**

### 7.4 关键 ablation（Table 1）

| Model | CRPS improv. | Win rate |
|---|---|---|
| Teacher | 0% | N/A |
| Plain MMD | -1.32% | 4.9% |
| MMD (best, $\eta=0.5, \rho=100$) | 0.82% | 75.0% |
| RMMD w/ CRPS (offline) | 1.11% | 89.2% |
| RMMD w/ CRPS (online) | **1.51%** | **93.0%** |

**On-policy 的优势**：+0.4% CRPS, +3.8% win rate，但 only 2× compute slowdown。原因：
1. Network 在自己 generation 的 noisy version 上训练，closer to inference distribution
2. For given $(x^{t-\delta}, x^t)$，training set 只有一个 transition，on-policy 允许 sample 更多 transition，减少 overfitting

### 7.5 Spread-skill ratio 与 calibration

Spread-skill ratio：
$$\text{SpreadSkillRatio}_k = \sqrt{\frac{M+1}{M}} \frac{\text{Spread}_k}{\text{EnsMeanRMSE}_k}$$

其中 $\text{Spread}_k = \sqrt{\frac{1}{T}\sum_t \frac{1}{M}\sum_i \|x^{t+k\delta,i} - \frac{1}{M}\sum_j x^{t+k\delta,j}\|^2}$

- Ratio < 1：under-dispersed（variance 太小）
- Ratio > 1：over-dispersed
- Ratio = 1：perfect calibration

Figure 5 显示 RMMD 优化 CRPS 同时 improves dispersion，matching 或 improving GenCast 的 dispersion，except humidity。Surprisingly, on-policy version（更好 CRPS）有 slightly worse dispersion than off-policy——这提示 CRPS 与 dispersion 不是完全 correlated。

### 7.6 Hyper-parameter tuning 的深刻影响（Table 3）

| Setting | CRPS improv. | Win rate |
|---|---|---|
| Plain MMD | -1.32% | 4.9% |
| MMD w/ $\eta=0.5$ | 0.22% | 50.7% |
| MMD (best, $\eta=0.5, \rho=100$) | 0.82% | 75.0% |

DDPM posterior：
$$p_{\text{cond}}(x_s | x_t, \hat{x}_0^{(t)}) = \mathcal{N}(\alpha_s \hat{x}_0^{(t)} + \sqrt{1-\alpha_s^2 - \gamma_{s,t}^2} \hat{\epsilon}, \gamma_{s,t}^2 I)$$

其中 $\hat{\epsilon} = (x_t - \alpha_t \hat{x}_0^{(t)})/\sigma_t$，$\gamma_{s,t} = \eta \frac{\sigma_s}{\sigma_t}\sqrt{1 - \frac{\alpha_t^2}{\alpha_s^2}}$。

$\eta=1$：classical DDPM；$\eta=0$：deterministic（与 MMD 不兼容）。$\eta=0.5$ 减少 stochasticity，improves CRPS。$\rho$ 从 7 改到 100（roughly uniform in log-SNR）also helps。

### 7.7 Churn analysis（Table 4）

GenCast 用 EDM sampling with stochastic churn（$S_{\text{noise}}=1.05$）。Churn 在 teacher 上 helpful（增加 spread），但在 MMD-distilled model 上 harmful：
- MMD w/ churn: 0.11% CRPS, 50.2% win rate（vs MMD w/o churn: 0.82%, 75.0%）

**Why?** MMD 依赖 $x_s'$ 是 non-deterministic function of $x_t, \hat{x}_0$，churn 干扰了这个 dependency。RMMD 是 better way to fix under-dispersion。

### 7.8 Auto-regressive rollout 的 surprising gain

RMMD 只在 12h lead time 优化 CRPS，但 Figure 11 显示 improvement 随 lead time 增加而 increase。If reward hacking（overfit next-state prediction），auto-regressive rollout 不会 improve。这 suggest RMMD 实际 improves distributional modeling，not just gaming the metric。

---

## 8. Limitations

1. **Quality ceiling**：RMMD 不 improve FID during reward phase，质量上限由 phase 1 MMD distillation 决定
2. **On-policy cost**：需要每 step sample from current policy，2× slowdown vs off-policy
3. **Differentiable reward requirement**：无法直接 optimize black-box reward，需要 differentiable surrogate

---

## 9. Intuition building：为什么 RMMD work？

### 9.1 Distributional anchor 的 reuse

Phase 1 MMD 训练的 auxiliary network $\Phi_{\theta_{\text{aux}}}$ 学会了 student 的 conditional first moment $m_{\text{student}}(s, x_s')$。Phase 2 freeze 它作为 distributional anchor，任何 student shift 都会被 $m_{\text{student}} - m_{\text{teacher}}$ 检测到。这相当于一个 **learned KL regularizer**，比传统 KL on marginal distribution 更 tight（因为 matching conditional moment 比 matching marginal 更强）。

### 9.2 On-policy 解决 distribution shift

Off-policy 方法（DI++）用 data marginal $p_t$，但 student shift 后 $p_t$ 不再代表 student intermediate distribution。On-policy 用 $x_t^{\text{pol}} \sim p_{\text{noise}}(\cdot | \tilde{x}_0)$，$\tilde{x}_0 \sim p_\theta$，确保 regularization 在 student 自己的 distribution 上 apply。Stop-gradient 避免 backprop through 整个 sampling chain，保持 single-step gradient 的 memory efficiency。

### 9.3 Single-step backprop 的 frequency coverage

与 HyperNoise（只在 noise input perturb，局限 low-frequency）不同，RMMD 在 corrupted on-policy sample $x_t^{\text{pol}}$ 上做 single denoising step。$x_t^{\text{pol}}$ 已经包含 student 的 high-frequency structure（因为 $\tilde{x}_0$ 是 full student sample），denoising step 修改这些 structure 可以 affect high-frequency。这解释了为什么 RMMD 在 CLIP/Picasso 等 complex reward 上 outperform HyperNoise。

### 9.4 Multi-step 的 inherent advantage

8-step student 比 1-step student 在 FID 上 advantage 巨大（1.26 vs 2.65）。Reward fine-tuning 在更高 quality baseline 上 work 更好，因为 Pareto front 起点更优。DI++ 的 1-step 结构 cap 住了 reward 优化的 ceiling。

### 9.5 GenCast 的科学意义

CRPS 是 proper scoring rule（minimized when forecast = true distribution），所以 optimizing CRPS 本质上是 distributional alignment。RMMD 在这里不是 reward hacking，是 真正的 distributional refinement。Auto-regressive rollout improvement 是 strong evidence：如果只 gaming next-state CRPS，rollout 会 diverge。Improvement 随 lead time 增加说明 RMMD 修复了 transition distribution 的 systematic bias（under-dispersion）。

---

## 10. 与其他方法的 positional 比较

| Method | Distillation | Reward | Multi-step backprop | On-policy | Frequency coverage |
|---|---|---|---|---|---|
| DI | ✓ (1-step) | ✗ | ✗ | ✗ | Full |
| DI++ | ✓ (1-step) | ✓ | ✗ | ✗ | Full |
| MMD | ✓ (multi-step) | ✗ | ✗ | ✗ | Full |
| DRaFT | ✗ | ✓ | ✓ (K steps) | ✗ | Full |
| DRaFT-1 | ✗ | ✓ | ✓ (1 step) | ✗ | High-freq only |
| ReFL | ✗ | ✓ | ✓ (random step) | ✗ | Noisy latent |
| HyperNoise | ✗ | ✓ | ✗ | ✗ | Low-freq only |
| DDPO/DPOK | ✗ | ✓ | ✗ (RL) | ✓ | Full |
| **RMMD** | **✓** | **✓** | **✓ (1 step)** | **✓** | **Full** |

RMMD 是唯一同时具备所有 desirable property 的方法。

---

## 11. 可能的 extension 与 open question

1. **Non-differentiable reward**：结合 reward model learning（如 RLHF 中的 preference model）或 score function estimation
2. **Adaptive $\lambda$**：当前 fix $\lambda$，可以用 Lagrangian method 自动 tune
3. **Higher-order moment matching**：MMD 只 match first moment，可以 extend to second moment（covariance）以更好 calibrate uncertainty
4. **Architecture interaction**：dropout 在 RMMD phase 设为 0，但其他 stochasticity source（e.g. stochastic depth）可能有不同 effect
5. **GenCast 的 CRPS-only optimization**：如果能 design reward 直接 targeting auto-regressive rollout quality（而非 single-step CRPS），可能 further improve long lead time
6. **Theoretical connection to score matching**：MMD regularization 在 $x_s'$ 来自 forward diffusion 时 recover DI 的 score-matching objective，但 on-policy 时这个 connection break，理论性质需要更深入分析
7. **Mode coverage vs mode seeking**：KL divergence 是 mode-seeking，CRPS 是 mode-covering，RMMD 的 MMD regularization 实际上 encourage 哪种 behavior 取决于 moment matching 的具体 geometry

参考链接：
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01969)
- [DDPO (Black et al., 2024)](https://arxiv.org/abs/2305.13301)
- [DPOK (Fan et al., 2023)](https://arxiv.org/abs/2305.10401)
- [ReFL (Xu et al., 2023)](https://arxiv.org/abs/2310.18855)
- [Multi-step Consistency Models (Heek et al., 2024)](https://arxiv.org/abs/2403.06807)
- [MeanFlow (Geng et al., 2025)](https://arxiv.org/abs/2505.13447)
- [Shortcut Models (Frans et al., 2025)](https://openreview.net/forum?id=0lzB6LnXcS)
- [Reward-Instruct (Luo et al., 2025)](https://arxiv.org/abs/2501.01752)
- [Implicit Diffusion (Marion et al., 2025)](https://arxiv.org/abs/2412.16101)
- [EDM (Karras et al., 2022)](https://arxiv.org/abs/2206.00364)
- [ERA5 dataset (Hersbach et al., 2020)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803)
- [CRPS (Ferro, 2014)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2243)
- [GenCast GitHub](https://github.com/google-deepmind/graphcast)

这篇 paper 的核心 contribution 是 unify distillation 与 reward fine-tuning 在一个 principled framework 里，通过 reusing MMD 的 auxiliary network 作为 distributional anchor，让 on-policy single-step gradient 既能 capture full frequency 又能 prevent distribution drift。GenCast 应用展示了 scientific domain 的 scaling 能力，CRPS 作为 proper scoring rule 让 reward optimization 等价于 distributional alignment，这给 scientific diffusion model 的 post-training 提供 general paradigm。
