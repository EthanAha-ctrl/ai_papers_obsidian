---
source_pdf: GenCast Diffusion-based ensemble forecasting for medium-range weather.pdf
paper_sha256: 0b5e06f0e0eb3d93279ab9f3837e07193f074800e442c559f34a96116e559a77
processed_at: '2026-08-04T13:23:28-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenCast 用人话版

Andrej，我把前面那篇技术讲解翻译成"咖啡桌聊天"版本。核心还是 build intuition，但去掉一些八股和公式堆砌。

参考：
- [GenCast Nature paper](https://www.nature.com/articles/s41586-024-08252-9)
- [Karras 2022 EDM paper](https://arxiv.org/abs/2206.00364)
- [GraphCast](https://arxiv.org/abs/2212.12794)

---

## 一句话概括

DeepMind 用 diffusion model 替代 ECMWF 的 operational ensemble forecast（ENS），在 1° 分辨率、15 天、84 个 variables 上 96%+ 的指标打败 ENS，单条 trajectory 在一块 TPU v4 上 1 分钟出。

---

## 为什么这件事重要

先说一个尴尬的事实：**之前所有的 ML weather model（GraphCast、Pangu、FourCastNet、FengWu、FuXi）虽然 RMSE 上打赢了 NWP，但都没真正解决 ensemble forecasting 问题**。

为什么？

因为它们训练目标都是 **MSE / RMSE**。MSE 最优解是条件期望 $\mathbb{E}[X_t | X_{t-1}]$，也就是 ensemble mean。当 lead time 长、未来分布 spread 大的时候，**条件期望本身就是模糊的**——这是数学决定的，跟模型多大、数据多干净无关。

举个直觉例子：10 天后的某个时刻，天气可能在一个低压槽里，也可能在一个高压脊里，两种 state 都 plausible。MSE-trained 模型会输出一个"半槽半脊"的中间态，物理上根本不存在。这就像让 GAN 去 generate 人脸，结果它输出一张"所有人脸的平均"——没有 pores、没有 hair、没有人看起来那样。

ensemble forecasting 的本质是采样联合分布 $p(X_{1:T} | O)$，每个 sample 是一个物理上 self-consistent 的天气演化。**Diffusion model 天生就是采样器**，这是 GenCast 的核心 insight。

---

## "Blurry" 这个毛病到底有多严重

论文 Figure 3 是最直观的证据。看 GraphCast 10 天 lead time 的预测——specific humidity at 700hPa，全球场像被 Gaussian blur 滤过一样，front、jet streak、convective blobs 全被抹平。

更糟的是看 power spectrum（Figure 4）：把预测场展成 spherical harmonics，GraphCast-Perturbed 在 high wavenumbers（小尺度结构）上 power 直接掉好几个数量级。ENS（NWP）和 GenCast 都能保持。这就是 blurry 在频域的 signature。

还有个隐蔽但致命的问题：**nonlinear derived variables 有 systematic bias**。比如 wind speed = $\sqrt{u^2 + v^2}$。MSE-trained 模型输出的是 $\mathbb{E}[u]$ 和 $\mathbb{E}[v]$，然后你去算 $\sqrt{\mathbb{E}[u]^2 + \mathbb{E}[v]^2}$。但 sqrt 是 concave function，Jensen 不等式告诉你 $\mathbb{E}[\sqrt{u^2+v^2}] < \sqrt{\mathbb{E}[u]^2 + \mathbb{E}[v]^2}$。所以 GraphCast-Perturbed 对 wind speed 有 **系统性负 bias**（Figure 4j-l）。GenCast 每个 ensemble member 都是 sharp 的 (u,v) 预测，先 member 内算 wind speed 再 ensemble 平均，没这个毛病。

---

## Diffusion 到底怎么用过来

### 整体 pipeline 的直觉

想象你在画一张天气图。Diffusion 的流程是：

1. 拿一张纯噪声图（high noise level $\sigma_{max}$）
2. 反复去噪，每一步降低一点 noise level
3. 走到 $\sigma = 0$，得到一张干净的天气图 sample

这是 standard DDPM/EDM 的套路。GenCast 的两个特别之处：

**第一，不是直接 sample $X_t$，而是 sample residual $Z_t = X_t - X_{t-1}$**。

直觉：天气是 slowly-evolving field，大部分 $X_t$ 的内容已经被 $X_{t-1}$ 解释了，真正不确定的是增量。在增量上做 diffusion 比在绝对场上做更稳定，模型 capacity 集中在真正 hard 的部分。这跟 ResNet 的 residual learning、或者 VAE 的 residual decoder 思想是一脉的。

**第二，autoregressive rollout**。一步 12h 的 diffusion sample 完，把结果塞回去当下一步的 condition，循环 30 次得到 15 天 trajectory。

### 关键 trick 1: Spherical noise distribution

标准 diffusion 的噪声是 i.i.d. Gaussian on pixels。但天气在 **球面** 上！在 equiangular lat-lon grid 上 sample i.i.d. noise 会出问题：极点附近 grid cells 物理面积小但数量一样多，导致噪声在极点 over-represented，spherical harmonic power spectrum 不平。

GenCast 的 fix：在 spherical harmonic domain 里直接 sample isotropic white noise（flat power spectrum），然后用 inverse spherical harmonic transform 投影回 grid。

直觉：**让 prior 的对称群匹配 data 的对称群**。如果 prior 破坏了球面旋转对称性，模型要 waste capacity 去学习补偿这个 symmetry breaking。这是 group-equivariant ML 的思想隐式应用。

### 关键 trick 2: Karras preconditioning

这个 trick 来自 Karras 2022 的 EDM paper，非常关键。

直接让神经网络 $f_\theta$ 预测 clean data 会遇到数值问题：不同 noise level $\sigma$ 下 target 的 scale 差几个数量级（$\sigma$ 大时 target 几乎是噪声，$\sigma$ 小时 target 是数据），梯度分布会很病态。

Karras 的解决方案：把 denoiser 写成
$$D_\theta(Z_\sigma) = c_{skip}(\sigma) \cdot Z_\sigma + c_{out}(\sigma) \cdot f_\theta(c_{in}(\sigma) Z_\sigma)$$

四个 $c$ 系数是 $\sigma$ 的解析函数，专门 chosen 让：
- 网络输入的 magnitude 在所有 $\sigma$ 下一致（$c_{in}$）
- 网络输出的 magnitude 在所有 $\sigma$ 下一致（$c_{out}$）
- 高 noise level 时 skip connection 主导（直接输出输入），低 noise level 时网络主导（$c_{skip}$）

直觉：**把数值条件问题丢给解析公式，让神经网络专心学结构**。这就像 BatchNorm 之于 ResNet——本身不增加 model capacity，但让 optimization landscape 平滑很多。

### 关键 trick 3: Noise level distribution 匹配 sampling

Karras 推荐 training 时 $\sigma$ 用 log-normal 分布。GenCast 改成：构造一个分布，其 quantiles 精确匹配 inference 时使用的 noise level schedule。

直觉：**importance sampling 思想**。如果 train 时模型在某些 noise level 上见得少，inference 时遇到那些 level 就 underfit。让 train 分布和 inference 分布对齐，no capacity wasted。

---

## 架构选择

GenCast 用 GraphCast 的 Encoder/Decoder，但把中间的 Processor 从 GNN 换成 **sparse transformer**。

- Mesh：5-refined icosahedral mesh，10242 nodes，61440 edges
- 16 个 transformer block，feature dim 512，4-head attention
- 每个 node attends to **16-hop neighborhood** on mesh

直觉：GNN 要传远距离信息需要很多层 message passing，transformer 一层就能 attend 到 16-hop（在 icosahedral mesh 上对应 synoptic scale，~1000 km）。Transformer 的 expressive power 比 message passing 强，配合 sparse attention（不是 global attention）保持 locality 和 efficiency。

为什么 16-hop？太大计算贵，太小不能覆盖 synoptic-scale weather patterns（气旋、锋面这些）。16 是 sweet spot。

---

## Conditioning 两个东西

### Condition on 前两步天气 state

简单粗暴：把 $X_{t-2}, X_{t-1}$ 和 noisy target $Z_\sigma^t$ 在 channel 维 concatenate，喂给网络。这是 second-order Markov——单步给位置，两步给位置+速度（trend）。

物理模型在完整 state 上是 first-order Markov，但 ML 在有限分辨率 reconstruction 上工作，second-order 给的 trend 信号有用。

### Condition on noise level $\sigma$

用 **conditional layer norm**（AdaSpeech / DiT 家族思想）。把 $\log\sigma$ 编码成 Fourier features → MLP → 16-dim encoding → 每个 layer norm 的 scale 和 offset 由这个 encoding 经 linear layer 输出。

直觉：不同 noise level 下网络看到的"任务"完全不同——高 noise level 是粗 structure recovery，低 noise level 是 fine detail refinement。让 $\sigma$ 控制 normalization 的 affine 参数，等于让网络在不同 $\sigma$ 下 effectively 是不同的网络。比简单把 $\sigma$ concatenate 到 input 更 expressive，因为 $\sigma$ 影响 every layer。

---

## Initial conditions 的故事

GenCast 用 ERA5 deterministic analysis + ERA5 EDA perturbations（9 个 EDA members 减去 mean）。

关键观察：**GenCast 不需要额外的 ad-hoc Gaussian Process perturbations**（GraphCast-Perturbed 需要）。因为 diffusion model 自己就能 generate ensemble spread——noise initialization 和 stochastic churn 已经提供 stochasticity。

deterministic ML model 没内在 stochasticity，必须靠 initial condition perturbation 制造 spread。GenCast 不需要——这是 generative model 的 inherent 优势。

---

## Verification 的微妙之处

这部分很容易被忽略但极重要。Weather forecasting 的 fair comparison 是出了名的难。

### Look-ahead 问题

ERA5 的 12h assimilation windows 是 21z-09z 和 09z-21z。

- 00z 的 ERA5 value 落在 window 早期，**用了 9 小时之后的观测**（look-ahead）
- 06z 的 ERA5 value 落在 window 中间，**只用了 3 小时之后的观测**

ECMWF operational ENS 用 00z/12z 初始化，享受 look-ahead 优势。为了公平，ML models 用 06z/18z（3h look-ahead）。但 ENS 在 TIGGE archive 只有 00z/12z 数据，所以 ENS 还是有 ~2h look-ahead 优势给 ENS。

Figure A.11 量化了这个 effect：用 00z/12z 初始化的 GenCast 比 06z/18z 版本 CRPS 和 RMSE 都明显好。这就是 weather verification 里 classic 的 "future leakage" 问题。

### Ensemble vs deterministic analysis 评分

WMO 标准是用 deterministic analysis 作 ground truth。但这 penalize 真实 ensemble：lead time=0 时最优解应该是 spread 围绕 analysis，但用 deterministic analysis 评分会 reward 把所有 members 设成 analysis（zero spread）。所以 GenCast 在 12h 看起来 over-dispersed，部分是 evaluation protocol 的 artifact。

---

## 结果的关键数字

- **CRPS**：97.3% targets 上优于 ENS，lead time > 36h 是 100%。平均改善 atmospheric 4.8%，surface 7.9%。
- **Ensemble-Mean RMSE**：96.4% targets 上优于 ENS。平均改善 atmospheric 3.5%，surface 5.4%。
- **Brier score（extreme events）**：97.6% 的 900 cases 上优于 ENS，平均改善 12.6%。
- **Spread/skill ratio**：3-4 天后收敛到 ~1（well-calibrated）。ENS 1-2 天后开始 under-dispersed。GraphCast-Perturbed 始终 under-dispersed。
- **Rank histograms**：3 天和 15 天 lead time 上 GenCast 接近 flat（理想），ENS 在 3 天略 U-shape。

CRPS 公式直觉：
$$\text{CRPS} \propto \underbrace{\text{accuracy}}_{\text{ensemble 到 truth 的距离}} - \underbrace{\text{spread}}_{\text{ensemble 内部离散度}}$$

CRPS 同时 reward 准和有 spread。Under-dispersed ensemble spread 项小，CRPS 高（差）。Over-dispersed accuracy 项大，CRPS 也高。最优是 calibrated。

---

## 为什么 power spectrum 是物理 plausibility 的好 proxy

天气场的 spherical harmonic power spectrum 反映了不同空间尺度能量的分布。真实大气在所有尺度上都有能量（turbulence cascade），所以 power spectrum 应该是 broadband 的。

MSE-trained 模型在 long lead time 上 high-frequency power 大幅下降——这就是 blurring。每个 ensemble member 都 blurry，因为模型学的是 mean，high-frequency 结构在 conditional expectation 中被 cancel 了。

Diffusion model 学的是 data 的 full distribution，每个 sample 都从分布里 draw，包括 high-frequency 结构。所以 GenCast 的 power spectrum 能保持。

ENS（NWP 物理模型）也保持得好，因为物理方程 evolve 所有尺度。GenCast 在这一点上 match 了 NWP 的物理性质，这是 ML weather model 之前一直做不到的。

---

## Computational cost 的意义

- GenCast：单条 15-day trajectory，**~1 minute on single Cloud TPU v4**
- N ensemble members on N TPUs：还是 ~1 minute
- ENS：跑在 ECMWF 超算上，hours

这开启了一个可能性：**未来可以 generate 1000+ ensemble members**，远超 ENS 的 50。Ensemble size 越大，对 distribution tails（极端事件）的 capture 越好。这是 GenCast 在 Brier score 上改善 12.6% 的部分原因——更多 effective samples 意味着更准的极端事件概率估计。

不过 diffusion 本身比 deterministic ML 慢：每步 39 次 denoiser forward（DPMSolver++2S 是 second-order solver，2N-1 = 39 evaluations for N=20 steps），15 天 30 步共 ~1170 次 forward。GraphCast 单步 1 次 forward，30 步共 30 次。所以 GenCast inference cost 是 GraphCast 的 ~40 倍，但仍然比 NWP 快得多。

---

## Limitations 要诚实看待

1. **1° 分辨率**：ENS 现在 0.081°（2023 后）。GenCast 在 1° 是为了 training/inference speed。高分辨率需要更多算力，但应该 yield 更好结果。

2. **Diffusion 比 deterministic ML 慢**：39 function evaluations per step。这是 generative model 的 inherent cost。

3. **依赖 NWP initial conditions**：还需要 EDA 系统。完全 end-to-end ML weather forecasting 还没实现。data assimilation 这块 NWP 还是很强。

4. **单步训练**：虽然 work，但 multi-step fine-tuning 可能进一步提升 long lead time 表现。GraphCast 用 12-step rollout fine-tune。GenCast 没做，这是个 opportunity。

5. **Precipitation**：在 Brier score 上 systematically 输给 ENS。Precipitation 是 highly non-Gaussian、sparse 的 field，可能需要 dedicated model 或不同 objective。

---

## 一些和 ML literature 的连接（你的领域会感兴趣）

- **Conditional Layer Norm** 和 DiT (Peebles & Xie)、AdaLN-Zero、FiLM 同家族——用 conditioning signal 控制 normalization affine 参数。
- **Sparse transformer on spherical mesh** 和 ViT 的 window attention 思想一样，但 on icosahedral mesh。
- **Score-based diffusion + probability flow ODE** (Song et al. 2021)：GenCast 用 ODE formulation，因为 ODE solver 更稳定，配合 stochastic churn 注入额外随机性。
- **Spherical equivariance** 和 Equiformer、MACE 等 equivariant GNNs 的 motivation 类似。但 GenCast 没有 strictly equivariant architecture，只是 noise distribution 是 isotropic——一个轻量级的 geometric prior。
- **Autoregressive rollout without teacher forcing** 和 video diffusion models（SVD、Video Diffusion Models）面临的 challenge 类似——single-step training 但 multi-step rollout inference。GenCast 证明在 weather 上这个 gap 可以跨过。

---

## 一句话 build intuition

**MSE 训练让模型学 mean，diffusion 训练让模型学 distribution**。Weather forecasting 在 long lead time 上本质是分布采样问题，所以 diffusion 是 mathematically right tool。GenCast 把这个 right tool 配上 weather-specific 的 inductive biases（spherical geometry、second-order Markov、residual prediction、icosahedral mesh transformer），实现了 ML ensemble forecasting 对 NWP ensemble 的首次系统性超越。

---

延伸阅读：
- [Aurora (2024)](https://arxiv.org/abs/2405.13014) - 后续 3D SwinTransformer + diffusion 的 weather foundation model
- [NeuralGCM](https://arxiv.org/abs/2311.07222) - ML + differentiable physics hybrid 路线
- [SEEDS](https://arxiv.org/abs/2306.14066) - Google 类似思路的 ensemble emulation with diffusion
- [WeatherBench 2](https://arxiv.org/abs/2308.15560) - 标准化 benchmark
- [Karras 2022 EDM](https://arxiv.org/abs/2206.00364) - diffusion design space 系统化，GenCast 的方法基础

---

# GenCast: Diffusion-based Ensemble Weather Forecasting 深度讲解

Andrej，这篇 paper 是 DeepMind 在 probabilistic weather forecasting 上的一个 milestone——把 diffusion model 用到全球 15-day ensemble forecasting，1° 分辨率，84 个 variables，并且 96%+ 的 verification targets 上击败了 ECMWF 的 ENS（top operational ensemble forecast）。我从机制、数学、架构、verification、physical plausibility 多个角度详细 build intuition。

参考链接：
- Paper 本体：[Nature 2024 GenCast](https://www.nature.com/articles/s41586-024-08252-9)（后续正式发表版本）
- Karras et al. 2022 "Elucidating the design space of diffusion-based generative models"：[arxiv 2206.00364](https://arxiv.org/abs/2206.00364)
- GraphCast (Lam et al. 2023)：[arxiv 2212.12794](https://arxiv.org/abs/2212.12794)
- Song et al. 2021 score-based SDE：[arxiv 2011.13456](https://arxiv.org/abs/2011.13456)
- DPMSolver++：[arxiv 2211.01095](https://arxiv.org/abs/2211.01095)
- ERA5 reanalysis：[ECMWF ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- ECMWF ENS documentation：[ENS docs](https://www.ecmwf.int/en/forecasts/documentation-and-support/medium-range-forecasts)

---

## 1. 高层 motivation：为什么 deterministic ML 不够

之前 GraphCast、Pangu-Weather、FourCastNet、FengWu、FuXi 等 ML 模型都训练在 **RMSE**（或者说 MSE）objective 上。这有一个深层的、根本性的问题：

- RMSE 最优的预测，是 **条件期望** $\mathbb{E}[X^t | X^{t-2:t-1}]$，即 ensemble mean。
- 当 lead time 变长，未来分布的 spread 增大，**条件期望会越来越模糊**——这是数学必然，不是 bug。
- 模糊的预测有两条致命后果：
  1. **物理不可信**：spherical harmonic power spectrum 在 high frequencies 上掉得很厉害（小尺度结构消失），全球大气里的湍流、front、jet streak 都被抹平。
  2. **derived 变量有 bias**：例如 wind speed $=\sqrt{u^2+v^2}$ 是 (u,v) 的非线性函数。$\mathbb{E}[\sqrt{u^2+v^2}] \ne \sqrt{\mathbb{E}[u]^2+\mathbb{E}[v]^2}$，所以 RMSE-trained 模型对 wind speed 有系统性的 **负 bias**（Jensen 不等式方向）。GenCast 论文里 Figure 4(j-l) 直观展示了这一点。

ensemble forecasting 的本质是 sampling 联合分布 $p(X^{1:T}|O)$，而非回归到 mean。**Diffusion model 是天然采样器**——这是 GenCast 的核心 insight。

---

## 2. 任务定义和 factorization 的直觉

### 2.1 联合分布

目标是 sampling：
$$p(X^{1:T} \mid O)$$

其中：
- $X^{t}$：timestep $t$ 的 atmospheric state（在 1° × 1° lat-lon grid 上，6 surface variables + 6 atmospheric variables × 13 pressure levels）
- $O$：observations（assimilation window 里的观测）
- $T = 30$：15 天，12 小时一步

直接建模 $p(X^{1:T}|O)$ 不可行（维度天文数字）。论文做 factorization：

$$p(X^{-1:T} \mid O) = p(X^{-1:0} \mid O) \prod_{t=1}^{T} p(X^t \mid X^{t-2:t-1}) \tag{1}$$

### 2.2 第二个因子：**second-order Markov**

注意是 **second-order** Markov（条件 on 过去两步 $X^{t-2:t-1}$），不是 first-order。这是与 NWP 物理模型的传统设定不同的地方。直觉上的 motivation（Appendix A.1）：

- 物理模型在 **完整物理 state** 上是 first-order Markov。
- 但我们在 **有限分辨率重建** 上工作（grid 是离散的、variables 是稀疏的 subset）。在有限分辨率下，过去两步能为 ML 模型提供 **trend / acceleration** 信号——单步只给位置，两步给位置+速度。
- 这和 GraphCast 用两步输入的 motivation 一致。

### 2.3 第一个因子：$p(X^{-1:0}|O)$

不直接 condition on observations，而是用 NWP 的 **reanalysis ensemble**（ERA5 + ERA5 EDA）来近似采样 initial conditions。也就是说 GenCast 把 data assimilation 这块让给了传统 NWP，自己只做 forecasting。这是 current ML weather 的共同限制——operational deployment 还需要 EDA 系统在背后跑。

### 2.4 Residual formulation

不直接 sample $X^t$，而是 sample residual：
$$Z^t = S^{-1}(X^t - X^{t-1})$$

其中 $S$ 是 diagonal matrix，每个 variable × level 一个 entry，用来 invert per-variable-level normalization（unit variance）。然后 $X^t = X^{t-1} + S Z^t$。

直觉：weather 是一个 slowly-evolving field，大部分 $X^t$ 的能量已经被 $X^{t-1}$ 解释，真正不确定的、需要 modeling 的是 **增量**。在 residual 上做 diffusion 比在 absolute field 上做更稳定，模型 capacity 集中在真正不确定的部分。

---

## 3. Diffusion model specification

### 3.1 整体 pipeline（Figure 1 的解析）

```
Input: X^{t-2}, X^{t-1}（已知两步）
       noise Z_0^t ~ p_noise(·|σ_0)  （high noise level σ_0）
                ↓
       iterative refinement via r_θ
                ↓  (N=20 steps)
       Z_N^t = Z^t  （at σ_N = 0, clean sample）
                ↓
       X^t = X^{t-1} + S Z^t
                ↓
       autoregressive: feed (X^{t-1}, X^t) → predict X^{t+1}
```

每个 refinement step $r_\theta$ 是一个 **DPMSolver++2S** 的 ODE step（second-order，需要两次 function evaluation of denoiser），加上 Karras 的 stochastic churn / noise inflation 注入额外随机性。

总 cost per timestep：$2N - 1 = 39$ function evaluations（最后一 step 只需一次）。15 天 = 30 步 × 39 = ~1170 次 denoiser forward。

### 3.2 Probability flow ODE 的直觉

Diffusion 的核心是把 noise → data 的过程建模为一条 ODE trajectory（probability flow ODE, Song et al. 2021）。在 Karras 的 parameterization 下，noise level $\sigma$ 是这条 trajectory 的参数：

- $\sigma = \sigma_{max}$（很大）：纯噪声
- $\sigma = 0$：纯 data

每个 $\sigma$ 对应一个 corrupted distribution $p(Z_\sigma | \text{data})$。Denoiser $D_\theta(Z_\sigma; \sigma)$ 学习预测 clean data $\hat{Z}$ given corrupted $Z_\sigma$。

### 3.3 Preconditioning（Equation 3）

这是 Karras et al. 2022 的核心 trick。直接让神经网络 $f_\theta$ 预测 clean data 在数值上不稳定（不同 $\sigma$ 下 target 的 scale 差好几个数量级）。Karras 把 denoiser 写成 preconditioned 形式：

$$D_\theta(Z_\sigma^t; X^{t-2:t-1}, \sigma) := c_{skip}(\sigma) \cdot Z_\sigma^t + c_{out}(\sigma) \cdot f_\theta(c_{in}(\sigma) Z_\sigma^t; X^{t-2:t-1}, c_{noise}(\sigma)) \tag{3}$$

变量解释：
- $Z_\sigma^t$：noise-corrupted version of target $Z^t$ at noise level $\sigma$
- $X^{t-2:t-1}$：conditioning（前两步天气状态）
- $f_\theta$：核心神经网络（详见 §4）
- $c_{skip}(\sigma)$：skip connection 权重，$\sigma$ 大时（高噪声）让 denoiser 直接输出输入（噪声）；$\sigma$ 小时让网络主导
- $c_{out}(\sigma)$：scaling 输出，控制网络输出贡献的 magnitude
- $c_{in}(\sigma)$：scaling 输入，让网络看到的 input 在不同 $\sigma$ 下 magnitude 一致
- $c_{noise}(\sigma)$：noise level encoding（喂给网络告诉它现在在哪个 $\sigma$）

在 Karras 的设定下，$\sigma_{data} = 1$（因为 target 已被 normalized 到 unit variance），这让 $c_{skip}, c_{out}, c_{in}, c_{noise}$ 的 closed form 简化（详见 Karras paper Table 1）。

直觉：**preconditioning 把数值条件问题丢给解析公式，让神经网络专心学 residual structure**。这是 Karras paper 能 work 得这么好的关键。

### 3.4 Spherical noise distribution（关键创新 1）

这是 GenCast 对 Karras 框架的 **第一个 nontrivial 修改**。

标准 diffusion：$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$，i.i.d. on pixels。

问题：weather state 在 **球面** 上（地球！）。如果直接在 equiangular lat-lon grid 上 sample i.i.d. Gaussian noise：
- 极点附近 grid cells 密度大（同样经度间隔对应物理距离小）
- i.i.d. noise 在 grid 上等权，但在物理上不等权
- 后果：spherical harmonic power spectrum 在 high frequencies 上 power 偏高（极点贡献的）

GenCast 的 fix：
1. 在 spherical harmonic domain 里 sample isotropic Gaussian white noise（**flat power spectrum**）
2. 截断到 grid 能 resolve 的 wavenumbers
3. 用 inverse spherical harmonic transform (Driscoll & Healy 1994) 投影回 grid

结果：noise 在球面上各向同性、power spectrum 期望平坦，符合球面几何。

直觉：**让先验 noise 的对称群匹配 data 的对称群**。这是 group-equivariant ML 的思想隐式体现。在大气、地球科学这种 spherical data 上，几何先验很重要。

### 3.5 Training-time noise level distribution（关键创新 2）

Karras 推荐 $p_{train}$ 用 log-normal，让训练时 noise level 在对数尺度上均匀分布。

GenCast 改成：构造一个 distribution，其 **quantiles match sampling noise schedule**。具体（Appendix A.3.2）：

inverse CDF：
$$F^{-1}(u) = \left( \sigma_{max}^{1/\rho} + u (\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho$$

其中 $u \sim U[0, 1]$。采样 schedule：
$$\sigma_i = \left( \sigma_{max}^{1/\rho} + \frac{i}{N-1}(\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho, \quad i \in \{0, \ldots, N-1\}$$

$\rho = 7$（让 noise level 在低端更密集，因为低 noise level 时 denoiser 需要更精细）。

直觉：**让训练时模型见到的 noise level 分布和 inference 时遇到的 noise level 分布匹配**。如果 train 时 noise level 在 inference 关键区间稀疏，模型在那里 underfit。这是 importance sampling 的思想。

训练时 $\sigma_{min}, \sigma_{max}$ 略宽于 inference（0.02 vs 0.03，88 vs 80），避免 boundary 效应。

---

## 4. Denoiser 架构（Figure 1 + §3.2.3）

### 4.1 总体：基于 GraphCast，但 Processor 换 sparse transformer

```
[Inputs: Z_σ^t, X^{t-2}, X^{t-1}, σ]
        ↓
  Encoder (same as GraphCast)
  grid → icosahedral mesh (5-refined, 10242 nodes, 61440 edges)
        ↓
  Processor: 16-layer sparse transformer (NEW)
  - feature dim 512
  - 4-head self-attention
  - each node attends to 16-hop neighbourhood on mesh
        ↓
  Decoder (same as GraphCast)
  mesh → grid
        ↓
  Output: prediction for Z^t
```

### 4.2 为什么换 Processor

GraphCast 的 Processor 是 GNN（message passing on multimesh）。GNN 在 weather 上 work 得好，但：
- 多次 message passing 才能传远距离信息
- 计算图 sparse 但 expression 受限

GenCast 选 sparse transformer，每个 node attends to **16-hop neighbourhood**（mesh 上的局部 patch）。这本质上是 spherical mesh 上的 sliding-window attention。好处：
- 单 layer 就能 aggregate 远距离信息（16-hop）
- transformer 的 expressive power 比 message passing GNN 强
- mesh 上 sparse attention，不像 dense transformer 那么 expensive

直觉：**16-hop neighbourhood 在 icosahedral mesh 上覆盖的物理距离足够大，能捕捉 synoptic-scale（天气尺度，~1000 km）的依赖，又保持 locality**。

### 4.3 Conditioning 机制

两种 conditioning：

1. **前两步天气 state** $X^{t-2:t-1}$：和 noisy target $Z_\sigma^t$ 一起在 channel dimension concatenate，作为 network input。简单直接。

2. **Noise level $\sigma$**：用 **conditional layer norm** (Chen et al. 2021, AdaSpeech)。具体：
   - 把 $\log \sigma$ 变成 32 个频率的 sine/cosine Fourier features（base period 16）
   - 过 2-layer MLP → 16-dim noise-level encoding
   - 每个 layer norm 用一个 linear layer 把这个 encoding 映射成 layer norm 的 scale 和 offset（替代 standard layer norm 的固定参数）

直觉：**用 $\sigma$ 控制 normalization 的 affine 参数，让 network 在不同 noise level 下 effectively 是不同的网络**。这比简单 concatenate $\sigma$ 更 expressive，因为 $\sigma$ 影响 every layer 而非只 input。

### 4.4 Mesh 细节

5-refined icosahedral mesh：
- 起点：20 面的 icosahedron
- 5 次 refinement（每次三角剖分）：$20 \times 4^5 = 20480$ 三角形，$10242$ nodes
- 比 GraphCast 的 multimesh（多层合并）简单——GenCast 只用单层 mesh

---

## 5. Training objective（Equation 5）

$$\sum_{t \in \mathcal{D}_{train}} \mathbb{E}\left[ \lambda(\sigma) \frac{1}{|G||J|} \sum_{i \in G} \sum_{j \in J} w_j a_i (Y_{i,j}^t - Z_{i,j}^t)^2 \right] \tag{5}$$

变量解释：
- $t$：训练 sample 的时间 index
- $i \in G$：grid location（lat-lon）
- $j \in J$：variable × pressure level（例如 $z_{1000}, z_{850}, \ldots, 2t, msl$）
- $Y_{i,j}^t$：denoiser 的 output（预测 clean residual）
- $Z_{i,j}^t$：真实 clean residual
- $w_j$：per-variable-level loss weight（继承 GraphCast；SST 特别 weight 0.01，因为 SST 变化慢）
- $a_i$：grid cell area，按 latitude 变化（极点小，赤道大），normalized 到 unit mean。**这是 area-weighted loss**，避免极点 over-represented。
- $\lambda(\sigma)$：Karras 的 per-noise-level loss weight
- expectation over $\sigma \sim p_{train}$, $\epsilon \sim p_{noise}(\cdot | \sigma)$

注意几个细节：
- **Area weighting** 是 weather-specific 修改，Karras 原版没有
- **Per-variable-level weighting** 也是 weather-specific，让重要变量（如 z500）权重高
- **不像 GraphCast 那样做 multi-step rollout training**——只在单步 12h transition 上训练。但能 rollout 到 15 天还比 GraphCast 强，说明 single-step diffusion objective 本身够好。这是很 surprising 的点。

---

## 6. Initial conditions（§4.2 + Appendix A.4）

### 6.1 GenCast 的 initial conditions

- 用 deterministic ERA5 analysis（high quality）
- 加上 ERA5 EDA perturbations $\Delta_i^{EDA} = e_i - \bar{e}$（$e_i$ 是 9 个 EDA members，$\bar{e}$ 是它们 mean）
- **不加** Gaussian Process perturbations（这对 GraphCast-Perturbed 有用，对 GenCast 没用——diffusion model 自己就能 generate spread）

直觉：**diffusion model 的 stochasticity 来自 sampling process 本身（noise initialization + stochastic churn）**，不需要靠 initial condition perturbation 制造 ensemble spread。deterministic model 才需要外部 perturbation。

### 6.2 GraphCast-Perturbed 的 initial conditions（baseline）

为了公平，GraphCast-Perturbed 用：
- ERA5 deterministic + EDA perturbations + **ad-hoc Gaussian Process perturbations**
- GP 在球面上，decorrelation length-scale 1200 km，marginal std = 0.085 × 6h differences 的 std
- 只施加于 5 个 variables（z, t, u, v, 2t），其他不施加

这个 GP perturbation 类似 ECMWF operational ENS 里的 singular vector perturbations——解决 EDA 单独 under-dispersed 的问题。

---

## 7. Verification protocol 的微妙之处（§4.3）

这部分很容易被忽略但 **极重要**，因为 weather forecasting 的 fair comparison 极其微妙。

### 7.1 Assimilation window look-ahead

ERA5 的 12h assimilation windows：21z-09z 和 09z-21z。
- 06z 的 ERA5 value：window 21z-09z，06z 在 window 中间偏后，**用了 ~3h look-ahead**（06z 之后的 observations）
- 00z 的 ERA5 value：window 21z-09z，00z 在 window 早期，**用了 ~9h look-ahead**

ECMWF 的 ENS operational forecast 用 00z/12z 初始化，享受 ~9h look-ahead（实际 operational 用 3-5h，比 ERA5 reanalysis 少）。

为了公平，**ML models 用 06z/18z 初始化**（3h look-ahead）。但 ENS 在 TIGGE archive 只有 00z/12z，所以论文里 ENS 还是有 ~2h 的 look-ahead 优势。

Figure A.11 直接证明了 look-ahead 影响：06z/18z 初始化的 GenCast 比 00z/12z 初始化差很多（CRPS 和 RMSE 都更差）。这是 weather verification 里很 classic 的"future leakage"问题。

### 7.2 Evaluation against deterministic analysis

标准做法：用 deterministic analysis 作 ground truth（WMO 标准）。但这对 ensemble 不利——lead time = 0 时，最优 ensemble 应该是围绕 analysis 的 spread，但用 deterministic analysis 评分会 reward 把所有 members 设成 analysis（zero spread）。这是 short lead time 上 GenCast 和 ENS 看起来 over-dispersed 的部分原因。

### 7.3 50-member ensemble

TIGGE 公开 ENS 的 50 members 但只在 8/13 pressure levels。所有比较用 50 members。

---

## 8. Results 深度解读

### 8.1 Headline 数字

- GenCast 在 97.3% 的 CRPS targets 上优于 ENS
- 100% 在 lead time > 36h
- 平均 CRPS 改善：atmospheric 4.8%，surface 7.9%
- Ensemble-Mean RMSE 改善：atmospheric 3.5%，surface 5.4%
- Brier score（extreme events）：97.6% 的 900 cases 上优于 ENS，平均改善 12.6%

### 8.2 CRPS 公式（Appendix A.6.1）

$$\text{CRPS} := \frac{1}{M} \sum_m \frac{1}{|G|} \sum_i a_i \left( \frac{1}{N} \sum_n |x_{i,m}^n - y_{i,m}| - \frac{1}{2N^2} \sum_{n, n'} |x_{i,m}^n - x_{i,m}^{n'}| \right) \tag{A.1}$$

变量解释：
- $M$：forecast cases 数
- $N$：ensemble size（50）
- $x_{i,m}^n$：第 $n$ 个 member 在 location $i$、case $m$ 的预测
- $y_{i,m}$：观测 / analysis
- 第一项 $\frac{1}{N}\sum_n |x - y|$：ensemble 和 truth 的平均距离（**accuracy**）
- 第二项 $\frac{1}{2N^2}\sum_{n,n'}|x^n - x^{n'}|$：ensemble 内部 pairwise 距离（**spread**，越大越被 reward）

直觉：**CRPS 同时 reward accuracy 和 spread**。Under-dispersed ensemble 第二项小，CRPS 高（差）。Over-dispersed 第一项大，CRPS 也高。最优是 well-calibrated。

注意：论文用 traditional estimator，不用 "fair CRPS"（fair CRPS 估计无限 ensemble 的极限 CRPS）。因为 initial conditions 来自 finite NWP ensemble，不能无限 scale。

### 8.3 Spread/skill ratio（Appendix A.6.4）

$$\text{Spread} := \sqrt{\frac{1}{M}\sum_m \frac{1}{|G|}\sum_i a_i \frac{1}{N-1}\sum_n (x_{i,m}^n - \bar{x}_{i,m})^2} \tag{A.6}$$

$$\text{Skill} := \text{Ensemble-Mean RMSE} \tag{A.7}$$

$$\text{SpreadSkillRatio} := \sqrt{\frac{N+1}{N}} \cdot \frac{\text{Spread}}{\text{Skill}} \tag{A.9}$$

完美 ensemble 下 SpreadSkillRatio ≈ 1。$\sqrt{(N+1)/N}$ 是 finite-size correction（因为 ensemble mean 比 single member 更接近 truth）。

Figure 5a 显示：
- GenCast：3-4 天后收敛到 ~1（well-calibrated）
- ENS：1-2 天后开始 under-dispersed
- GraphCast-Perturbed：始终 under-dispersed（因为 RMSE-trained 模型模糊，spread 不足）

### 8.4 Rank histograms（Figure 5b-d）

Rank histogram：把 truth 在 ensemble members 排序里的 rank 直方图化。完美 ensemble → uniform。
- U-shape：under-dispersed（truth 经常在 ensemble 之外）
- ∩-shape：over-dispersed（truth 经常在 ensemble 中间）

观察：
- 12h：GenCast 和 ENS 都 ∩-shape（部分因为 verification 用 deterministic analysis，penalize 真实 spread）
- 3d, 15d：GenCast 接近 flat，ENS 在 3d 略 U-shape

GraphCast-Perturbed 的 12h rank histogram 有 10 个 peaks（modulated U-shape）——这暴露了它用 9 个 EDA members + GP perturbations，EDA members 之间方差主导，GP 没起作用。这是 ad-hoc perturbation 的局限。

---

## 9. Physical plausibility 的直觉

### 9.1 Spherical harmonic power spectrum（Figure 4）

把 field $f(\theta, \phi)$ 展成 spherical harmonics $Y_l^m$，power spectrum $P(l) = \sum_m |a_l^m|^2$ 是 power vs wavenumber $l$。

观察：
- **GenCast** 的 power spectrum 在所有 lead times 都 close to ground truth（ERA5 analysis）
- **GraphCast-Perturbed** 在 long lead times 上 high-$l$（small spatial scale）power 大幅下降——这就是 "blurring" 的频域 signature
- ENS 的 power spectrum 也保持得很好（NWP 物理模型不会 blur）

例外：z500 这种很 smooth 的 field，GenCast 在 high $l$ 上有少量 extra noise（比 ground truth 大 6-7 个数量级小于 dominant power）。这是 diffusion model 在 low-power regimes 的小毛病，但不影响实用。

直觉：**diffusion model 学的是 data 的 full distribution，包括 high-frequency 结构**。RMSE-trained model 学的是 mean，high-frequency 结构在 ensemble averaging 中被 cancel，所以训练时 model 没有动力保留它们。

### 9.2 Derived variables（Figure 4g-l）

Wind speed $= \sqrt{u^2 + v^2}$ 是 nonlinear。RMSE-trained 模型预测 (u, v) 的 mean，但 $\mathbb{E}[\sqrt{u^2+v^2}] < \sqrt{\mathbb{E}[u]^2 + \mathbb{E}[v]^2}$（Jensen 不等式，因为 sqrt 是 concave）。

GraphCast-Perturbed 在 10u, 10v 上 bias 接近 0，但 wind speed 上有 **强负 bias**。GenCast 没有——因为 GenCast 每个 ensemble member 都是 sharp 的 (u, v) 预测，对每个 member 算 wind speed 再 ensemble 平均，没有 Jensen 不等式问题。

这是 diffusion model 的根本优势：**each sample is a physically plausible state, not a mean**。

---

## 10. Computational cost

- 训练：未详细披露（推测 O(days) on TPU pod）
- Inference：单 15-day trajectory，**~1 minute on single Cloud TPU v4**
- N ensemble members on N TPUs：still ~1 minute
- 对比 ENS：跑在 ECMWF 超算上，耗时 hours

这开启了一个可能性：**未来可以 generate 1000+ ensemble members**，远超 ENS 的 50，从而更好地 capture distribution tails（极端事件）。

---

## 11. Limitations 和 open questions

1. **1° 分辨率**：ENS 现在 0.081°（2023 后）。GenCast 在 1° 是为了 training/inference speed。高分辨率需要更多算力。
2. **Diffusion 比 deterministic ML 慢**：39 function evaluations per step vs GraphCast 的 1 次。这是 generative model 的 inherent cost。
3. **依赖 NWP initial conditions**：还需要 EDA 系统。完全 end-to-end ML weather forecasting 仍没实现。
4. **单步训练**：虽然 work，但 multi-step fine-tuning 可能进一步提升 long lead time 表现。GraphCast 用 12-step rollout fine-tune。GenCast 没做，可能是个 opportunity。
5. **Precipitation**（Appendix A.7.1）：GenCast 在 RMSE、CRPS、SEEPS 上 often 优于 ENS，但在 Brier score 上 systematically 输给 ENS。Precipitation 是 highly non-Gaussian、sparse 的 field，需要特殊处理（可能要 dedicated model）。

---

## 12. 一些和 ML literature 的 connection（你的领域）

- **Conditional Denoising Estimator** (Batzolis et al. 2021)：conditioning on previous timesteps 的方式。
- **Conditional Layer Norm** (AdaSpeech, Chen et al. 2021)：和 FiLM、AdaLN-Zero (DiT) 同家族——用 conditioning signal 控制 normalization affine 参数。
- **Sparse transformer on mesh**：和 Vision Transformer 的 window attention 思想一样，但 on spherical icosahedral mesh 而非 image patches。
- **Score-based diffusion + probability flow ODE** (Song et al. 2021)：GenCast 用 ODE formulation（不是 SDE），因为 ODE solver 更稳定且 deterministic（除了 stochastic churn）。
- **Spherical equivariance**：和 Equiformer、MACE 等 equivariant GNNs 的 motivation 类似——match data symmetry。但 GenCast 没有 strictly equivariant architecture，只是 noise distribution 是 isotropic。
- **Autoregressive rollout without teacher forcing**：和 video diffusion models（Video Diffusion Models, Ho et al. 2022b；SVD）面临的 challenge 类似——single-step training 但 multi-step rollout inference。

---

## 13. 总结：build intuition 的关键 takeaways

1. **RMSE objective → blurry forecasts** 是数学必然，不是工程问题。Diffusion 是 principled 修复。
2. **Second-order Markov + residual prediction** 是 ML weather forecasting 的好 inductive bias。
3. **Spherical noise distribution** 把 prior 的 symmetry 匹配 data 的 symmetry——几何先验重要。
4. **Preconditioning (Karras)** 让数值 stability 和 network learning 解耦。
5. **Conditional Layer Norm** 是注入 noise level conditioning 的优雅方式。
6. **Sparse transformer on icosahedral mesh** 兼顾 long-range dependency 和 locality。
7. **Verification 的微妙之处**（assimilation look-ahead、deterministic analysis 评分）决定了 fair comparison 怎么做。
8. **Power spectrum 和 derived variable bias** 是 physical plausibility 的好 proxy，直接揭示 RMSE-trained 模型的 fundamental limit。

延伸阅读推荐：
- [Karras 2022](https://arxiv.org/abs/2206.00364) - diffusion design space 的系统化
- [GraphCast](https://arxiv.org/abs/2212.12794) - 架构基础
- [Aurora (2024)](https://arxiv.org/abs/2405.13014) - 后续 3D SwinTransformer + diffusion 的 weather foundation model
- [NeuralGCM](https://arxiv.org/abs/2311.07222) - ML + differentiable physics hybrid 路线
- [WeatherBench 2](https://arxiv.org/abs/2308.15560) - 标准化 benchmark
- [SEEDS](https://arxiv.org/abs/2306.14066) - 类似思路的 ensemble emulation with diffusion

如果你对某一块想 deeper dive（比如 spherical harmonic transform 的实现、DPMSolver++2S 的具体公式、或 GraphCast multimesh 和 GenCast single mesh 的 trade-off），告诉我就好。
