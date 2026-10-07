---
source_pdf: Local Intrinsic Dimension Unveils.pdf
paper_sha256: ecef34c61629eec622c27c97e2f0aa89f1084f4ba4a84c1058ba4489ae16362e
processed_at: '2026-08-05T15:44:20-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Paper

## 故事从头讲

Diffusion model 画手的时候，经常画出六根手指。这事困扰了整个社区很久。统计上这张图完全合理——像素分布跟训练集很像——但结构上完全荒谬。你问模型"你画的是啥"，它自己也不知道哪里出了问题。

这篇 paper 干了一件事：**把 hallucination 这个玄学问题，翻译成了几何问题**。

---

## 核心直觉：Manifold 上长了"假分支"

想象 DM 学到的 data manifold 是一张巨大的橡皮膜。大部分区域平滑规整——5 根手指的手就在一个低维的"凹槽"里安分守己。但有些区域，橡皮膜鼓起来了，鼓出一些不该存在的方向。

这些鼓起来的方向就是 hallucination 的温床。模型在这些地方多给了几个"自由度"——本该固定 5 指的地方，它允许变 6 指、7 指。

**关键问题**：怎么量化"鼓起来"？答案就是 **LID（Local Intrinsic Dimension）**——局部有效维度。manifold 上每个点有多少个真正能变的方向？正常手部图像可能就 50 个有效方向（角度、光照、肤色...），hallucinated 的手可能有 60 个——多出来的 10 个就是模型瞎发明的。

---

## 三个 Filter 的故事

### TVF（前人的方法）：看时间轴抖不抖

Aithal 2024 的思路：生成过程中如果 $\hat{\mathbf{x}}_0$ 预测一直在剧烈震荡，说明模型"犹豫不决"，大概率在 hallucinate。这就像看一个厨师做菜——如果他在盐和糖之间反复横跳，最后端出来的多半是黑暗料理。

公式回顾：

$$\text{TVF}(\mathbf{x}_0) = \int_{t_1}^{t_2} \|\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t) - \overline{\hat{\mathbf{x}}_{0,t_1:t_2}^\theta}\|_2^2 \, dt$$

- $\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t)$：时间 $t$ 时模型预测的最终图
- $\overline{\hat{\mathbf{x}}_{0,t_1:t_2}^\theta}$：$[t_1, t_2]$ 区间内这些预测的平均值
- 整个积分衡量"预测偏离自己平均的程度"

问题：这只是 temporal 视角，没触及 spatial 几何本质。

### LMI（本文的中间产物）：戳一下看抖不抖

本文先提出 LMI——在初始 noise $\mathbf{x}_1$ 旁边戳一个小扰动 $\varepsilon$，看最终生成的图偏离多少：

$$\text{LMI}(\mathbf{x}_1) = \text{Var}_\varepsilon(\mathcal{G}_\theta(\mathbf{x}_1 + \varepsilon))$$

- $\mathbf{x}_1$：初始 Gaussian noise
- $\varepsilon \sim \mathcal{N}(\mathbf{0}, \beta^2 \mathbf{I})$：小球面扰动，$\beta$ 很小
- $\mathcal{G}_\theta$：完整 generator（noise → image 的映射）
- 直觉：输入稍微动一下，输出飞多远

类比：你在一座山上踩一个小脚印，看脚印在山脚放大成多大的坑。坑大 = 这座山陡峭不稳定 = 容易 hallucinate。

实验结果：LMI 比 TVF 稍好，但还不够干净。

### LID（本文的最终武器）：直接数方向

作者发现 LMI 膨胀有两个原因：

**R1**：真实的 valid 方向变化过快（如光照角度的敏感度被夸大）
**R2**：凭空发明了 invalid 方向（如允许手指数量变化）

LMI 把 R1 和 R2 混在一起测。而 **LID 专门测 R2**——它只数"有多少个显著方向"，不管这些方向变化多快。

Proposition 1 的公式讲清了关系：

$$\text{LMI}(\mathbf{x}_1) \approx \beta^2 \sum_{i=1}^{\lfloor \text{LID}_\theta(\mathbf{x}_0) \rfloor} \sigma_i^2$$

- $\sigma_i$：generator Jacobian 的第 $i$ 大奇异值
- $\text{LID}_\theta(\mathbf{x}_0)$：生成图 $\mathbf{x}_0$ 在 manifold 上的局部维度
- $\lfloor \cdot \rfloor$：取整（理论 LID 是整数）
- 求和上限被 LID 截断——LID 之后的奇异值被视为数值噪声

**人话**：总抖动 = 前 LID 个方向的抖动之和。LID 越大，能抖的方向越多，越容易出事。

实验（Figure 2）：LID 作为 filter **显著超越** LMI 和 TVF。在 11kHands 上 Cohen's d 远高于其他，separability 最佳。

---

## 神 insight：Loss 就是 LID 估计器

这是整个 paper 最漂亮的发现。

正常训练 DM 用的 DSM loss：

$$\mathcal{L}_{DSM}(\mathbf{x}_0, t, \theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t)\|_2^2]$$

- $\epsilon$：forward process 加的标准 Gaussian noise
- $\epsilon_\theta$：模型预测的 noise
- $\mathbf{x}_t = \mathbf{H}_t \mathbf{x}_0 + \Sigma_t^{1/2} \epsilon$：noisy 状态

Proposition 2 证明：当 $t \to 0$，

$$\mathbb{E}_\epsilon[\mathcal{L}_{DSM}] = \text{LID}_\theta(\mathbf{x}_0)$$

**人话**：模型自己的训练 loss，就是 LID 的无偏估计。你不需要额外训练任何东西——任何已训练的 DM 都原生支持 LID 估计。

为什么？因为当 $t$ 很小，$\hat{\mathbf{x}}_0^\theta$ 充当 manifold 上的投影器。DSM 的 residual 就是"投影到切空间后残留的误差"，而这个误差的期望等于切空间的维度（即 LID）。证明里用了一个经典事实：**投影后的标准 Gaussian 的期望平方范数 = 投影矩阵的 trace = rank**。

这跟 [Yeats et al. 2025](https://arxiv.org/abs/2501.01127) 的工作相关——他们证明 loss 上界 LID，本文进一步证明是**无偏估计**。

---

## 从 Filter 到 Corrector：IQ 登场

Filter 只是事后筛选——你得先生成一堆废图再扔掉。作者想：既然知道 LID 高是病根，能不能在生成过程中**主动把 LID 压下去**？

### Theorem 1：Bottleneck 结构

把 generator 拆成两段：

$$\mathcal{G}_\theta = \mathcal{G}_\theta^{\leq\tau} \circ \mathcal{G}_\theta^{>\tau}$$

- $\mathcal{G}_\theta^{>\tau}$：$t > \tau$ 的 macroscopic flow（主生成阶段）
- $\mathcal{G}_\theta^{\leq\tau}$：$t \leq \tau$ 的 terminal projection（最后清理）

定理结论：

$$\text{LMI}(\mathbf{x}_1) \lesssim K \beta^2 \sum_{i=1}^{\lfloor \text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau)) \rfloor} (\sigma_i^{>\tau})^2$$

- $K$：terminal projection 的谱范数平方
- $\sigma_i^{>\tau}$：macroscopic Jacobian 的奇异值
- 关键：求和上限由 terminal 阶段的 LID 截断

**人话**：macroscopic flow 可以放大很多方向，但 terminal projection 像漏斗——只有 LID 个方向能通过。所以总不稳定性被 LID 严格卡住上限。

这意味着：**主动降 LID = 从求和里 drop 方差项 = 降低 LMI 上界**。可证明的改进。

### Theorem 2：Boltzmann Quenching

修正 score function：

$$\tilde{\mathbf{s}}_\theta(\mathbf{x}_t) = \mathbf{s}_\theta(\mathbf{x}_t) - \lambda_t \nabla_{\mathbf{x}_t} \mathcal{E}(\mathbf{x}_t)$$

- $\mathbf{s}_\theta(\mathbf{x}_t)$：原始 score（推样本走向高概率区）
- $\mathcal{E}(\mathbf{x}_t) = \text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t))$：能量函数 = 当前 predicted $\hat{\mathbf{x}}_0$ 的 LID
- $\lambda_t$：引导强度
- $\nabla_{\mathbf{x}_t} \mathcal{E}$：LID 的梯度（往 LID 降低的方向推）

定理说：这等价于采样自 Boltzmann 加权分布：

$$p_t^{\theta,\lambda_t}(\mathbf{x}_t) \propto p_t^\theta(\mathbf{x}_t) \cdot \mathbb{E}[\exp(-\lambda_t \text{LID}_\theta(\mathbf{x}_0))]$$

- $\exp(-\lambda_t \text{LID}_\theta)$：Boltzmann factor，惩罚高 LID 终态
- 直觉：把概率质量从高维 stratum（6 指手）转移到低维 stratum（5 指手）

**人话**：热力学淬火。高温状态（高 LID）被快速冷却，锁定到低能构型（低 LID）。名字 **Intrinsic Quenching** 就来自这。

### Corollary 2：Mode-Seeking

继续推导发现，最小化 energy 朝向：
- $\nabla \log p_t^\theta = 0$：log-density 极值点
- $\text{Tr}(\Sigma_t \nabla^2 \log p_t^\theta) \ll 0$：最大负曲率（局部极大）

**人话**：IQ 把样本推向概率分布的山峰，避开山谷。山谷就是 mode 之间的 interpolation 区域——Aithal 说的 hallucination 来源。两个视角殊途同归。

---

## IQ 怎么实现

### 自适应尺度

$$\lambda_t = \lambda \cdot \frac{\|\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t) - \mathbf{x}_t\|_2}{\|\Sigma_t \nabla_{\mathbf{x}_t} \mathcal{E}(\mathbf{x}_t)\|_2 + \epsilon}$$

- 分子：自然 update 的范数（score 推力大小）
- 分母：energy gradient 投影到 data space 的范数
- $\lambda$：用户设定比例（0.05-0.2 最优）
- 目的：让引导项的大小始终是自然更新的固定比例，避免数值爆炸

### 动态过滤

只在 LID 高于阈值 $q_t$ 的样本上应用：

$$\bar{\lambda}_t = 0 \text{ if } \mathcal{E}(\mathbf{x}_t) < q_t$$

- $q_t$：用 2048 个参考样本预校准的 LID 分布的 $q$-th percentile
- $q \approx 0.4$ 最优（11kHands）

**人话**：只治有病的，别折腾健康的。稳定样本本来就 LID 低，强行压反而破坏多样性。

### 时间窗口

只在 $[t_1, t_2]$（$t_2$ 很小）应用。原因：
- Theorem 2 假设 $t \leq \tau$
- Figure 3 显示 LID gradient 从全局结构 → 局部 hallucinatory 特征 → 分散 cohesive 的演变
- 对应 diffusion 的"细节雕刻"阶段

11kHands 上用 $t_1=0.025, t_2=0.0625$（归一化后），非常窄的窗口。

### Pseudocode 精简版

```python
if t < t_1 or t > t_2:
    return net(x, t)  # 窗口外不干预

x.requires_grad_(True)
x_0_hat = net(x, t)                    # 预测 clean image
LID = dsm_loss(x_0_hat, x, t)          # 估 LID
grad = autograd(LID.sum(), x)          # 算 LID 梯度

# 自适应尺度
scale = lambda * norm(x_0_hat - x) / (norm(t**2 * grad) + 1e-8)

# 过滤
mask = (LID >= quantile(baseline_LID(t), q)).float()

# 修正
guided_x_0_hat = x_0_hat - mask * scale * (t**2 * grad)
return guided_x_0_hat
```

---

## 实验结果人话版

### 11kHands：碾压级表现

| Method | HR↓ | UP↑ |
|--------|-----|-----|
| Baseline | 29.3% | 39.8% |
| RODS-CAS | 25.8% | 40.2% |
| **IQ** | **9.0%** | **68.0%** |

- HR（Hallucination Ratio）：从 29.3% 降到 9.0%，**相对降 69%**
- UP（User Preference）：从 39.8% 升到 68.0%，**用户偏好翻倍**

这是整篇 paper 最炸的结果。其他方法基本没动 needle，IQ 直接把 hallucination 砍到三分之一。

### FID 的谎言

注意 FID 略有回退（16.3 → 16.6）。这是 [Stein et al. 2023](https://arxiv.org/abs/2311.04389) 和 [Jayasumana et al. 2024](https://arxiv.org/abs/2401.03652) 指出的经典问题：**FID 偏好 texture over structure**。六指手在 FID 眼里跟五指手没区别——像素分布差不多。所以 FID 对 hallucination 几乎盲。

附录 ablation 更夸张：当 HR 降到几乎 0%（全无 hallucination），FID 反而指数增长。这进一步证明 feature-based metrics 在 hallucination 场景下完全失灵。

### MNIST：一致性改善

| Method | HR↓ | FID↓ |
|--------|-----|------|
| Baseline | 37.3% | 32.3 |
| **IQ** | **10.2%** | **31.8** |

MNIST 的 hallucination 主要是"生成多余形状"。这些 outliers 同时拉高 FID 和拉低 diversity，移除后所有指标一致改善。这是理想情况——hallucination 与 feature metrics 对齐。

### GaussianGrid：理论验证

2D mixture of Gaussians 的 toy problem。IQ 在真实 LID = ambient dimension（2D）时仍有效。这验证 Theorem 2 的概率解释——Boltzmann factor 不依赖真实 LID 的绝对值，依赖 model-induced LID 的相对差异。

### 医学影像：零样本匹配 supervised 方法

低剂量 CT 重建任务，用 ResNet50 observer 检测脑出血类型：

| Method | mAP↑ | 需要标签？ |
|--------|------|-----------|
| Baseline | 0.27 | - |
| DG | 0.31 | **是**（需训练 classifier） |
| **IQ** | **0.31** | **否**（零样本） |

IQ 完全零样本，却匹配了需要 ground-truth hemorrhage labels 的 DG 的诊断增益。同时 IQ 严格保持 PSNR/SSIM/LPIPS 不变——重建质量无损。这对临床部署意义重大。

---

## Ablation 的人话

### $\lambda$ 太大太小都不行

Figure 7：$\lambda \in [0.05, 0.2]$ 是平台期。太小没效果，太大开始破坏正常样本。0.08 是 11kHands 的最优。

### 时间窗口必须精准

Figure 8：尖锐的转变——过早应用无效（结构还没形成），过晚应用反而增加 hallucination（已经固化错误结构）。这跟 [Sclocchi et al. 2025](https://www.pnas.org/doi/10.1073/pnas.24129) 的 diffusion phase transition 理论一致。

### 过滤参数 $q$ 的 trade-off

Figure 9：$q=0$（全应用）会把所有样本挤到 mode 中心，丧失 variance。$q \approx 0.4$ 是甜蜜点——只治 LID 高的 60% 样本，保留 40% 稳定样本的原貌。

GaussianGrid 上 $q=0$ 的可视化最直观：所有样本挤到 25 个 Gaussian mode 的中心点，variance 几乎归零。$q>0$ 后恢复原始分布。

### 自适应 vs 常数 scaling

Figure 10：常数 scaling 在最优 HR 范围内引入 artifacts——过饱和颜色、严重结构扭曲。自适应 scaling 无此问题。因为不同时间步、不同样本的 gradient magnitude 差异巨大，常数会被某些情况 dominate。

---

## 跟 Memorization 的对称性

[Ross et al. 2025](https://openreview.net/forum?id=aZ1gNJu8wO) 显示 LID 坍塌指示 memorization——模型直接复制训练样本，没有任何创造力，LID 趋近 0。

本文显示 LID 膨胀指示 hallucination——模型过度创造，发明虚假维度。

```
LID → 0        memorization（没创造力）
LID 适中       optimal generation（健康）
LID → ∞        hallucination（瞎创造）
```

**LID 是 DM 创造力的度量衡**。控制 LID 在合理区间可能是 universal regularization 目标。这个 insight 比具体的 IQ 方法更有 long-term 价值。

---

## Limitations 人话

1. **评估难**：hallucination 主观，标注者之间也常有分歧。FID 等自动指标在 hallucination 场景失灵。需要更好的 benchmark。

2. **慢 4.3 倍**：主要开销是 autograd + 32 次 noise 采样估 LID。降低 $k$ 或用更便宜 estimator 是即时改进方向。

3. **只在 unconditional 验证**：conditional settings（text-to-image）的 hallucination 风险更高，但 IQ 还没在那验证。

4. **时间窗口敏感**：得调 $[t_1, t_2]$，不同 dataset 差异大。11kHands 用 [0.025, 0.0625]，SimpleShapes 用 [0.48, 0.96]——跨度极大。缺少自动选择机制。

---

## 一句话总结

**Hallucination 是 manifold 上长了假分支，LID 量化分支数量，IQ 用梯度把它们压回去**。

---

## 参考

- Paper: https://openreview.net/forum?id=LocalIntrinsicDimension
- Aithal 2024 (TVF): https://arxiv.org/abs/2406.09313
- Yeats 2025 (LID estimator): https://arxiv.org/abs/2501.01127
- Tian 2025 (RODS): https://openreview.net/forum?id=fhuqIxoPcr
- Ross 2025 (memorization): https://openreview.net/forum?id=aZ1gNJu8wO
- Stanczuk 2024 (DM encode LID): https://arxiv.org/abs/2402.17563
- Karras 2022 (EDM): https://arxiv.org/abs/2206.00364
- Song 2021 (SDE): https://arxiv.org/abs/2011.13456
- Sohl-Dickstein 2015 (thermodynamics): https://arxiv.org/abs/1503.03585
- Sobieski 2026 (SDB): https://openreview.net/forum?id=cipx3rwfWp
- Stein 2023 (metrics flaws): https://arxiv.org/abs/2311.04389
- Sclocchi 2025 (phase transition): https://www.pnas.org/doi/10.1073/pnas.24129
- RSNA dataset: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

---

# Local Intrinsic Dimension Unveils Hallucinations in Diffusion Models — 深度技术讲解

## 1. 问题定位：Structural Hallucinations 的本质

Diffusion models 生成图像时会出现一类棘手的失败模式——**structural hallucinations**：样本在统计层面匹配 training data 分布，却违反底层物理/逻辑/形态规则。典型例子包括六指手、错位眼睛、多余拇指。这些 sample 看起来"像手"但结构上不合理。

此前 Andrej 你在 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 makemore 系列里讨论过 generative model 的 "memorization vs generalization" 谱系，这篇 paper 恰好把这个谱系扩展到 spatial 几何维度：**hallucination 是过度创造（LID 膨胀），memorization 是缺乏创造（LID 坍塌）**，二者共同 span 了 DM 的生成谱。

参考链接：
- Aithal et al. 2024 "Understanding Hallucinations in Diffusion Models through Mode Interpolation": https://arxiv.org/abs/2406.09313
- Ross et al. 2025 "A Geometric Framework for Understanding Memorization": https://openreview.net/forum?id=aZ1gNJu8wO
- Stanczuk et al. 2024 "Diffusion Models Encode the Intrinsic Dimension of Data Manifolds": https://arxiv.org/abs/2402.17563

---

## 2. 既有假说 vs 本文新视角

### Aithal 的 Temporal 视角
Aithal et al. 2024 提出 hallucinations 源于 **mode interpolation**——DM 在 data distribution 的 modes 之间插值，放大低概率区域。他们提出 **Trajectory Variance Filter (TVF)**：

$$\text{TVF}(\mathbf{x}_0) = \int_{t_1}^{t_2} \|\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t) - \overline{\hat{\mathbf{x}}_{0,t_1:t_2}^\theta}\|_2^2 \, dt$$

- $\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t)$: 时间 $t$ 的 posterior mean 预测
- $\overline{\hat{\mathbf{x}}_{0,t_1:t_2}^\theta}$: 区间 $[t_1, t_2]$ 内 posterior mean 的时间平均
- 直觉：hallucinated sample 在 reverse process 中 $\hat{\mathbf{x}}_0$ 预测剧烈震荡

### 本文的 Spatial 视角
作者主张从 **temporal generative behavior** 转向 **local geometry of model-induced manifold**。核心问题：hallucination 是否也表现为 manifold 上的局部空间不稳定？

定义 **Local Manifold Instability (LMI)**：

$$\text{LMI}(\mathbf{x}_1) \triangleq \text{Tr}\big(\text{Cov}_\varepsilon(\mathcal{G}_\theta(\mathbf{x}_1 + \varepsilon))\big) = \text{Var}_\varepsilon(\mathcal{G}_\theta(\mathbf{x}_1 + \varepsilon))$$

- $\mathbf{x}_1$: 初始噪声（$t=1$ 的 terminal state）
- $\varepsilon \sim \mathcal{N}(\mathbf{0}, \beta^2 \mathbf{I})$: 小球面扰动，$\beta > 0$ 很小
- $\mathcal{G}_\theta$: 完整 generator mapping
- 直觉：测量 noise space 中一个小球被 transport 到 data space 后的总空间扩散。扩散大 → 该区域 manifold 不稳定 → 易 hallucinate

这与 [sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) 中的 local sensitivity 概念同源（Cacuci et al. 2005）。

---

## 3. 数学框架：SDE、Manifold、Generator

### Forward / Reverse SDE

Forward process（eq 1）：

$$d\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_t \, dt + \mathbf{G}_t \, d\mathbf{w}_t$$

- $\mathbf{x}_t \in \mathbb{R}^n$: 时间 $t$ 的状态，$n$ 是 ambient dimension（如图像像素数）
- $\mathbf{F}_t \in \mathbb{R}^{n \times n}$: 线性 drift 矩阵，时间依赖
- $\mathbf{G}_t \in \mathbb{R}^{n \times n}$: diffusion coefficient 矩阵
- $\mathbf{w}_t \in \mathbb{R}^n$: forward Wiener process（标准 Brownian motion）
- $t \in [0,1]$: $t=0$ 是数据分布 $p(\mathbf{x}_0)$，$t=1$ 是 terminal（通常 Gaussian）

Reverse process（eq 2）：

$$d\mathbf{x}_t = [\mathbf{F}_t \mathbf{x}_t - \mathbf{G}_t \mathbf{G}_t^\top \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)] \, dt + \mathbf{G}_t \, d\overline{\mathbf{w}}_t$$

- $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$: **score function**——log probability 的梯度
- $\overline{\mathbf{w}}_t$: 时间反向的 Wiener process
- $\mathbf{G}_t \mathbf{G}_t^\top \nabla \log p$: drift 的 score 修正项，把样本推向高概率区

参考 Song et al. 2021 "Score-Based Generative Modeling through SDEs": https://arxiv.org/abs/2011.13456

### Probability Flow ODE 与 Generator

PF-ODE 是 reverse SDE 的确定性对应物：

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{F}_t \mathbf{x}_t - \frac{1}{2} \mathbf{G}_t \mathbf{G}_t^\top \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$$

差异在于 reverse SDE 有 $\mathbf{G}_t \, d\overline{\mathbf{w}}_t$ 随机项，PF-ODE 用 $\frac{1}{2}$ 系数替代。代入 $\theta$-参数化 score，定义 **generator**（eq 3）：

$$\mathcal{G}_\theta(\mathbf{x}_1) = \mathbf{x}_1 - \int_0^1 \mathbf{F}_{ODE}^\theta(\mathbf{x}_t, t) \, dt$$

这是从 noise $\mathbf{x}_1$ 到 data $\mathbf{x}_0$ 的确定性映射。

### Stratified Manifold Hypothesis (Assumption 1)

[Goresky & MacPherson 1988 的 stratified Morse theory](https://link.springer.com/book/10.1007/978-1-4612-1410-7) 指出高维数据并非单一低维 manifold，而是一组**不相交的低维 submanifolds（strata）的并集**，不同 strata 维度不同。例如手部图像可能由 5 维 stratum（5 指手）和 6 维 stratum（6 指手，训练中极少）组成。

基于此定义 **model-induced manifold**：

$$\mathcal{M}_\theta = \{\mathbf{x}_0 \mid \exists_{\mathbf{x}_1 \in \mathbb{R}^n} \mathbf{x}_0 = \mathcal{G}_\theta(\mathbf{x}_1)\}$$

即 generator 所有可能输出的集合。这个构造由 [Fefferman et al. 2016](https://www.ams.org/journals/jams/2016-29-04/S0894-0347-2016-00852-3/) 和 [Pidstrigach 2022](https://arxiv.org/abs/2202.01027) 的 manifold hypothesis 检验和 DM-manifold 几何研究支撑。

### Forward Kernel 与 Tweedie

Forward kernel（条件分布）：

$$p(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{H}_t \mathbf{x}_0, \Sigma_t)$$

- $\mathbf{H}_t = \Phi(t, 0)$: state transition matrix
- $\Phi(t, s) = \exp\big(\int_s^t \mathbf{F}_u \, du\big)$: 假设 $\mathbf{F}_t$ 可交换
- $\Sigma_t = \int_0^t \Phi(t, \tau) \mathbf{G}_\tau \mathbf{G}_\tau^\top \Phi(t, \tau)^\top \, d\tau$: 协方差累积

[Tweedie's formula](https://www.tandfonline.com/doi/abs/10.1198/016214504000001547)（Efron 2011）连接 score 与 posterior mean（eq 6）：

$$\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t] = \Phi(t, 0)^{-1}\big(\mathbf{x}_t + \Sigma_t \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)\big)$$

直觉：posterior mean = 把 noisy $\mathbf{x}_t$ 沿 score 方向修正后映射回 $t=0$。当 $t$ 很小时，$\hat{\mathbf{x}}_0^\theta$ 充当 manifold 上的近似 orthogonal projector。

---

## 4. Proposition 1：LMI ≈ LID 加权的奇异值平方和

**命题**：在 $\beta$ 足够小、一阶线性近似成立的前提下，

$$\text{LMI}(\mathbf{x}_1) \approx \beta^2 \sum_{i=1}^{\lfloor \text{LID}_\theta(\mathbf{x}_0) \rfloor} \sigma_i^2$$

变量解释：
- $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0$: generator Jacobian $\nabla_{\mathbf{x}_1} \mathcal{G}_\theta(\mathbf{x}_1)$ 的奇异值，降序排列
- $\text{LID}_\theta(\mathbf{x}_0)$: 生成点 $\mathbf{x}_0$ 在 $\mathcal{M}_\theta$ 上的 local intrinsic dimension
- $\lfloor \cdot \rfloor$: 取整操作（理论上 LID 是整数，连续估计器给出连续值）
- $\beta^2$: 扰动方差的缩放

### 证明直觉（自下而上）

**Step 1: 一阶 Taylor 展开**

$$\mathcal{G}_\theta(\mathbf{x}_1 + \varepsilon) \approx \mathcal{G}_\theta(\mathbf{x}_1) + \nabla_{\mathbf{x}_1}\mathcal{G}_\theta(\mathbf{x}_1) \cdot \varepsilon = \mathcal{G}_\theta(\mathbf{x}_1) + \mathbf{J}\varepsilon$$

其中 $\mathbf{J} = \nabla_{\mathbf{x}_1}\mathcal{G}_\theta(\mathbf{x}_1)$。

**Step 2: 协方差传播**

$\mathcal{G}_\theta(\mathbf{x}_1)$ 对 $\varepsilon$ 是确定性的常数，从 Cov 中消失：

$$\text{Cov}_\varepsilon(\mathcal{G}_\theta(\mathbf{x}_1 + \varepsilon)) = \mathbf{J} \cdot \text{Cov}(\varepsilon) \cdot \mathbf{J}^\top = \beta^2 \mathbf{J}\mathbf{J}^\top$$

**Step 3: Trace = Frobenius 范数平方**

$$\text{LMI} = \text{Tr}(\beta^2 \mathbf{J}\mathbf{J}^\top) = \beta^2 \|\mathbf{J}\|_F^2$$

**Step 4: Frobenius = 奇异值平方和**

$$\|\mathbf{J}\|_F^2 = \sum_{i=1}^n \sigma_i^2$$

**Step 5: Stratified manifold 截断**

虽然神经网络 Jacobian 实际上满秩（所有 $\sigma_i > 0$），但 stratified manifold hypothesis 意味着 off-manifold 方向被强烈压缩。$\text{LID}_\theta$ 之后的奇异值是无穷小的数值噪声，不代表结构自由度：

$$\sum_{i=1}^n \sigma_i^2 \approx \sum_{i=1}^{\lfloor \text{LID}_\theta(\mathbf{x}_0) \rfloor} \sigma_i^2$$

### Intuition

LMI 衡量"输入扰动 → 输出扰动"的总放大率。这个放大率只在前 $\text{LID}_\theta$ 个方向显著，因为只有这些方向是 manifold 的有效切方向。所以 LMI 本质上是 **"LID 决定的有效方向上的能量总和"**。

### Hallucination 的两种来源

Proposition 1 揭示 LMI 膨胀的两个独立机制：

- **R1**：过度估计真实奇异值（valid 方向变化过快）
- **R2**：发明虚假的 off-manifold 方向（invalid 自由度，如可变手指数）

LMI 同时被 R1 和 R2 膨胀，这解释了为何 LMI 作为 filter 不如 LID 干净——LID 主要捕捉 R2。

---

## 5. Proposition 2：Loss 即 LID 无偏估计器

[Yeats et al. 2025](https://arxiv.org/abs/2501.01127) 证明 DSM/ISM loss 上界 LID。本文扩展为**无偏估计**：

$$\mathbb{E}_\varepsilon[\mathcal{L}_{DSM}(\mathbf{x}_0, t, \theta)] = \text{LID}_\theta(\mathbf{x}_0)$$

$$\mathbb{E}_\varepsilon[\mathcal{L}_{ISM}(\mathbf{x}_0, t, \theta)] = -\frac{1}{2}\big(n - \text{LID}_\theta(\mathbf{x}_0)\big)$$

### DSM Loss (eq 4)

$$\mathcal{L}_{DSM}(\mathbf{x}_0, t, \theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t)\|_2^2]$$

- $\epsilon$: forward 引入的标准 Gaussian noise
- $\epsilon_\theta$: 模型预测的 noise
- $\mathbf{x}_t = \mathbf{H}_t \mathbf{x}_0 + \Sigma_t^{1/2} \epsilon$

### ISM Loss (eq 5)

$$\mathcal{L}_{ISM}(\mathbf{x}_0, t, \theta) = \mathbb{E}_\epsilon\big[\text{Tr}(\Sigma_t \nabla_{\mathbf{x}_t} \mathbf{s}_\theta(\mathbf{x}_t)) + \frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x}_t)\|_{\Sigma_t}^2\big]$$

- $\mathbf{s}_\theta$: score prediction
- $\text{Tr}(\Sigma_t \nabla \mathbf{s}_\theta)$: 加权 Hessian trace（曲率项）
- $\|\mathbf{s}_\theta\|_{\Sigma_t}^2 = \mathbf{s}_\theta^\top \Sigma_t \mathbf{s}_\theta$: 加权 score 范数

### 证明直觉

**关键步骤**：通过 Tweedie，predicted score 定义 posterior mean：

$$\mathbf{s}_\theta(\mathbf{x}_t) = -\Sigma_t^{-1}(\mathbf{x}_t - \mathbf{H}_t \hat{\mathbf{x}}_0^\theta(\mathbf{x}_t))$$

DSM residual 化为：

$$\epsilon - \epsilon_\theta(\mathbf{x}_t) = \Sigma_t^{-1/2} \mathbf{H}_t(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t) - \mathbf{x}_0)$$

当 $t \to 0$，$\hat{\mathbf{x}}_0^\theta$ 是 manifold 上的 oblique projection（在 Mahalanobis 内积下）：

$$\mathbf{H}_t(\hat{\mathbf{x}}_0^\theta - \mathbf{x}_0) \approx \mathbf{P}_\mathcal{T}^{\Sigma_t, \mathbf{H}_t} \Sigma_t^{1/2} \epsilon$$

$\mathbf{P}_\mathcal{T}^{\Sigma_t, \mathbf{H}_t}$ 是切空间 $\mathcal{T}_{\mathbf{x}_0}\mathcal{M}_\theta$ 上的 oblique projector。变换 $\mathbf{M}_\mathcal{T} = \Sigma_t^{-1/2} \mathbf{P}_\mathcal{T} \Sigma_t^{1/2}$ 通过相似变换保持 rank，并满足对称 + 幂等 → 标准 Euclidean orthogonal projector。其 rank = $\text{LID}_\theta(\mathbf{x}_0)$。

最后用标准恒等式：投影后的标准 Gaussian 的期望平方范数 = 投影矩阵的 trace = rank：

$$\mathbb{E}_\epsilon[\epsilon^\top \mathbf{M}_\mathcal{T}^\top \mathbf{M}_\mathcal{T} \epsilon] = \text{Tr}(\mathbf{M}_\mathcal{T}) = \text{LID}_\theta(\mathbf{x}_0)$$

ISM 通过 Stein's Lemma 与 DSM 关联：$\mathbb{E}[\mathcal{L}_{DSM}] = n + 2\mathbb{E}[\mathcal{L}_{ISM}]$，代入即得 $-\frac{1}{2}(n - \text{LID})$。

### Intuition

- DSM loss 大 → 模型在切空间投影后仍残留大误差 → 有效维度高
- ISM loss 大 → score Hessian trace 大负值 → 高曲率 → 多余维度
- 二者从不同角度量化"模型允许的有效自由度"

实践意义：**任何已训练 DM 都原生支持 LID 估计**，无需额外训练。这正是 IQ 可零样本应用的基础。

---

## 6. 实验 1：Filter 性能对比

数据集：11kHands（手部图像）+ EDM model + 40-step Euler solver + 128 samples + 人工标注。

| Filter | 优势 | 劣势 |
|--------|------|------|
| TVF | 时序视角，cheap | 需优化 $[t_1, t_2]$ 区间 |
| LMI | 空间视角，直观 | 需 32 次完整 reverse process |
| LID | 最佳 separability | 需选择合适小 $t$ |

Figure 2 关键观察：
- LID 在最优 $t$ 下显著超越 LMI 和 TVF
- 性能对 $t$ 高度敏感，必须选足够小的 $t$（遵循 Proposition 2 假设）
- 11kHands 上 LID 的 Cohen's d 远超其他 filter

附录 Table 3 跨数据集结果：LID 在多数情况下 Cohen's d 最高（FFHQ 0.43, 11kHands 隐含最高），但 AFHQV2 和 MNIST 上 LMI 偶尔占优——这与 dataset 的 hallucination 性质有关。

### 几何解释

实验直接验证 R1/R2 假说：hallucination 主要由 **R2（虚假维度发明）** 驱动，所以 LID（专门测 R2）> LMI（同时测 R1+R2）。当 dataset 的 hallucination 也含大量 R1 成分时（如 AFHQV2 的纹理噪声），LMI 反而更敏感。

---

## 7. Theorem 1：Spectral Bottleneck

为了让 IQ 在生成过程中可操作（而非仅在终点 filter），作者将 generator 分解：

$$\mathcal{G}_\theta = \mathcal{G}_\theta^{\leq \tau} \circ \mathcal{G}_\theta^{>\tau}$$

- $\mathcal{G}_\theta^{>\tau}: \mathbf{x}_1 \mapsto \mathbf{x}_\tau$: macroscopic flow（$t > \tau$，主生成阶段）
- $\mathcal{G}_\theta^{\leq \tau}: \mathbf{x}_\tau \mapsto \mathbf{x}_0$: terminal projection（$t \leq \tau$，最后清理）

**定理（一般形式，含 anisotropic 噪声如 SDB）**：

$$\text{LMI}(\mathbf{x}_1) \lesssim K \beta^2 \sum_{i=1}^{\lfloor \text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau)) \rfloor} (\sigma_i^{>\tau})^2$$

变量：
- $\sigma_1^{>\tau} \geq \cdots \geq \sigma_n^{>\tau} \geq 0$: macroscopic Jacobian $\nabla_{\mathbf{x}_1}\mathcal{G}_\theta^{>\tau}(\mathbf{x}_1)$ 的奇异值
- $K = \|\nabla_{\mathbf{x}_\tau}\mathcal{G}_\theta^{<\tau}(\mathbf{x}_\tau)\|_2^2 \geq 1$: terminal oblique projection 的平方谱范数
- $\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau)$: 时间 $\tau$ 的 Tweedie posterior mean 估计

### 证明关键步骤

**Chain rule**：

$$\mathbf{J} = \nabla_{\mathbf{x}_1}\mathcal{G}_\theta = \underbrace{\nabla_{\mathbf{x}_\tau}\mathcal{G}_\theta^{\leq\tau}}_{\mathbf{J}_{\leq\tau}} \cdot \underbrace{\nabla_{\mathbf{x}_1}\mathcal{G}_\theta^{>\tau}}_{\mathbf{J}_{>\tau}}$$

**Von Neumann trace inequality**（奇异值乘积不等式）：

$$\|\mathbf{J}_{\leq\tau} \mathbf{J}_{>\tau}\|_F^2 \leq \sum_{i=1}^n (\sigma_i^{\leq\tau})^2 (\sigma_i^{>\tau})^2$$

**Terminal projector 谱结构**：$\mathbf{J}_{\leq\tau}$ 作为 oblique projector 有恰好 $\lfloor \text{LID}_\theta \rfloor$ 个非零奇异值（其余严格为 0）：

$$(\sigma_i^{\leq\tau})^2 \leq K \text{ for } i \leq \lfloor \text{LID}_\theta \rfloor, \quad \sigma_i^{\leq\tau} = 0 \text{ otherwise}$$

代入得严格截断：

$$\sum_{i=1}^n (\sigma_i^{\leq\tau})^2 (\sigma_i^{>\tau})^2 \leq K \sum_{i=1}^{\lfloor \text{LID}_\theta \rfloor} (\sigma_i^{>\tau})^2$$

**Topological equivalence**：当 $\tau \to 0$，Tweedie 估计 $\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau) \to \mathbf{x}_0$（MMSE estimator + 局部曲率 → 距离 manifold 距离趋于 0）。LID 是局部常量拓扑不变量，故：

$$\text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau)) = \text{LID}_\theta(\mathbf{x}_0)$$

### Intuition：Bottleneck 概念

Macroscopic flow（$t > \tau$）可以放大很多方向，但 terminal projection（$t \leq \tau$）像一个"漏斗"——只有 $\text{LID}_\theta$ 个方向能通过，其余被 annihilate。所以**总不稳定性被 terminal 阶段的 LID 严格截断**。

这给了 IQ 操作空间：**主动降低 $\text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau))$ 等价于从求和中 drop 方差项**，可证明降低 LMI 上界。重要的是，这只需 posterior mean $\hat{\mathbf{x}}_0^\theta(\mathbf{x}_\tau)$ 而非最终 $\mathbf{x}_0$，使得修正可在生成中途进行。

---

## 8. Theorem 2：IQ 的 Boltzmann 分布解释

**修正 score**（eq 10）：

$$\tilde{\mathbf{s}}_\theta(\mathbf{x}_t) = \mathbf{s}_\theta(\mathbf{x}_t) - \lambda_t \nabla_{\mathbf{x}_t} \mathcal{E}(\mathbf{x}_t)$$

- $\mathcal{E}(\mathbf{x}_t) = \text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t))$: energy function
- $\lambda_t$: 时间依赖引导强度

**定理**：在 $t \leq \tau$（posterior variance $\sigma_t^2 \to 0$）下，使用 $\tilde{\mathbf{s}}_\theta$ 等价于采样自 **ideal Boltzmann distribution**：

$$p_t^{\theta, \lambda_t}(\mathbf{x}_t) \propto p_t^\theta(\mathbf{x}_t) \cdot \mathbb{E}_{p^\theta(\mathbf{x}_0 \mid \mathbf{x}_t)}\big[\exp(-\lambda_t \text{LID}_\theta(\mathbf{x}_0))\big]$$

### 证明直觉

**Step 1**：定义 ideal guided density = marginal × expected terminal energy 的 Boltzmann factor。

**Step 2**：log + 求导得 ideal modified score：

$$\nabla \log p_t^{\theta,\lambda_t} = \nabla \log p_t^\theta + \nabla \log \mathbb{E}[\exp(-\lambda_t \text{LID}_\theta)]$$

**Step 3**：当 $\sigma_t^2 \to 0$，posterior $p^\theta(\mathbf{x}_0 \mid \mathbf{x}_t)$ 退化为以 $\hat{\mathbf{x}}_0^\theta$ 为中心的 Dirac delta。log-expectation → expectation-log：

$$\lim_{\sigma_t^2 \to 0} \log \mathbb{E}[\exp(-\lambda_t \text{LID}_\theta)] = -\lambda_t \text{LID}_\theta(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t))$$

**Step 4**：理论上 LID 是局部常量整数（梯度为零），但 operational estimator $\mathcal{L}_{DSM}$ 是神经网络参数化的连续可微函数，梯度非零：

$$\nabla_{\mathbf{x}_t}\mathcal{E}(\mathbf{x}_t) \approx \nabla_{\mathbf{x}_t}\mathcal{L}_{DSM}(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t), t, \theta)$$

### Intuition

Boltzmann factor $\exp(-\lambda_t \text{LID}_\theta)$ 惩罚高 LID 终态——把概率质量从高维 stratum（如 6 指手）转移到低维 stratum（如 5 指手）。这本质是 **thermodynamic quenching**：高温（高 LID）状态被快速冷却，锁定到低能（低 LID）构型。这正是方法名 **Intrinsic Quenching** 的由来。

参考 thermodynamics of diffusion: [Sohl-Dickstein 2015](https://arxiv.org/abs/1503.03585)。

---

## 9. Corollaries：连接到 Mode-Seeking

### Corollary 1：DSM 与 ISM 梯度共线

$$\nabla_{\mathbf{x}_t}\mathcal{L}_{DSM}(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t), t, \theta) = 2 \nabla_{\mathbf{x}_t}\mathcal{L}_{ISM}(\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t), t, \theta)$$

二者精确共线，DSM 是 ISM 的 2 倍。意义：**用 DSM 的数值稳定性 + ISM 的概率可解释性**。

### Corollary 2：Mode-Seeking Behavior

代入 $\mathbf{s}_\theta = \nabla \log p_t^\theta$ 到 ISM：

$$\mathcal{L}_{ISM} = \mathbb{E}_\epsilon\big[\text{Tr}(\Sigma_t \nabla_{\mathbf{x}_t}^2 \log p_t^\theta(\mathbf{x}_t)) + \frac{1}{2}\|\mathbf{s}_\theta\|_{\Sigma_t}^2\big]$$

当 $t \to 0$，最小化 energy 等价于朝向：
- **Stationary points**：$\nabla_{\mathbf{x}_t} \log p_t^\theta(\mathbf{x}_t) = \mathbf{0}$（log-density 极值）
- **Maximal negative curvature**：$\text{Tr}(\Sigma_t \nabla^2 \log p_t^\theta) \ll 0$（log-density 局部极大）

这正是 **mode-seeking**——把样本推向概率分布的局部极大值，避开 mode 间的低概率 interpolation 区域。

### 与 Aithal 假说的统一

Aithal 主张 hallucination 来自 mode interpolation；IQ 通过 mode-seeking 直接消除这些 interpolation。**两个视角殊途同归**：temporal 视角看到"trajectory 在 modes 间游走"，spatial 视角看到"manifold 上有虚假高维方向"。IQ 从 spatial 切入，结果在 temporal 层面表现为 mode 收敛。

---

## 10. IQ 算法实现

### 自适应尺度 $\lambda_t$

为稳定修正，$\lambda_t$ 动态计算使得 energy 项投影到 data space 的范数 = 自然更新的固定比例 $\lambda$：

$$\lambda_t = \lambda \cdot \frac{\|\hat{\mathbf{x}}_0^\theta(\mathbf{x}_t) - \mathbf{x}_t\|_2}{\|\Sigma_t \nabla_{\mathbf{x}_t}\mathcal{E}(\mathbf{x}_t)\|_2 + \epsilon}$$

- $\|\hat{\mathbf{x}}_0^\theta - \mathbf{x}_t\|_2$: 自然 update 的范数（score 推力大小）
- $\|\Sigma_t \nabla \mathcal{E}\|_2$: energy gradient 投影到 data space 的范数
- $\lambda$: 用户设定的比例（如 0.05–0.2）
- $\epsilon$: 数值稳定项

在 EDM 框架中 $\Phi(t, 0) = \mathbf{I}$，$\Sigma_t = t^2 \mathbf{I}$。

### 动态过滤

为避免 IQ 限制所有样本（包括本就稳定的），引入 mid-generation filter：

$$\bar{\lambda}_t = 0 \text{ if } \mathcal{E}(\mathbf{x}_t) < q_t$$

- $q_t$: 时间依赖阈值 = reference set 在时间 $t$ 的 LID 分布的 $q$-th percentile
- 用 2048 个无条件生成样本预校准 $q_t$

### Pseudocode（Algorithm 1）

```
# 仅在 [t_1, t_2] 窗口应用
if t < t_1 or t > t_2:
    return stopgrad(net(x, t))

x.requires_grad_(True)
x_0_hat = net(x, t)                    # posterior mean 预测
LID = dsm_loss(x_0_hat, x, t)          # LID 估计
grad = autograd(LID.sum(), x)          # energy gradient
raw_guidance = (t**2) * grad           # 投影到 data space (EDM)

# 自适应尺度
nat_update = x_0_hat - x
scale = (lambda * norm(nat_update)) / (norm(raw_guidance) + 1e-8)

# 过滤
q_t = quantile(baseline_LID(t), q)
mask = where(LID >= q_t, 1.0, 0.0)

guided_x_0_hat = x_0_hat - (mask * scale * raw_guidance)
return stopgrad(guided_x_0_hat)
```

### 时间窗口选择

只在 narrow interval $[t_1, t_2]$（$t_2$ 小）应用，理由：
- Theorem 2 假设 $t \leq \tau$
- Figure 3 显示 LID gradient 的 coarse-to-fine 演变：早期是全局结构，中期是局部 hallucinatory 特征，晚期是分散 cohesive 状态
- 与 [Park et al. 2023](https://papers.nips.cc/paper_files/paper/2023/hash/24129-Abstract.html) 和 [Sclocchi et al. 2025](https://www.pnas.org/doi/10.1073/pnas.24129-Abstract) 的 diffusion phase 分解一致

---

## 11. 实验结果分析

### Table 1：跨 6 个数据集的定量对比

| Dataset | Method | HR↓ | UP↑ | FID↓ | IV↑ | DSV↑ |
|---------|--------|-----|-----|------|-----|------|
| 11kHands | Baseline | 29.3 | 39.8 | 16.3 | 0.032 | 0.15 |
| 11kHands | RODS-CAS | 25.8 | 40.2 | 15.8 | 0.033 | 0.15 |
| 11kHands | **IQ** | **9.0** | **68.0** | 16.6 | 0.030 | 0.13 |
| FFHQ | Baseline | 8.2 | 45.3 | 13.5 | 0.067 | 0.40 |
| FFHQ | **IQ** | **4.2** | **46.1** | 13.9 | 0.065 | 0.39 |
| MNIST | Baseline | 37.3 | - | 32.3 | 0.050 | 0.13 |
| MNIST | **IQ** | **10.2** | - | 31.8 | 0.051 | 0.14 |
| GaussianGrid | Baseline | 20.2 | - | - | - | - |
| GaussianGrid | **IQ** | **8.9** | - | - | - | 0.016 |

### 关键观察

**11kHands 上的突破**：HR 从 29.3% → 9.0%（69% 相对降低），UP 从 39.8% → 68.0%（70% 相对提升）。FID 略有回退（16.3 → 16.6），DSV/IV 微降——这恰好印证 [Stein et al. 2023](https://arxiv.org/abs/2311.04389) 和 [Jayasumana et al. 2024](https://arxiv.org/abs/2401.03-Abstract) 的发现：**feature-based metrics 与人类偏好错位**，尤其在 hallucination 场景。

**MNIST 的戏剧性**：HR 从 37.3% → 10.2%，同时 FID 改善（32.3 → 31.8）和 diversity 提升。原因：MNIST 的 hallucination 主要是"生成多余形状"，这些 outliers 同时拉高 FID 和拉低 diversity，移除后所有指标一致改善。

**GaussianGrid 的理论验证**：作为 toy problem，IQ 在真实 LID = ambient dimension 时仍有效，验证了 Theorem 2 的概率解释——Boltzmann factor 不依赖真实 LID，而依赖 model-induced LID 的相对差异。

### Ablation Studies（Appendix B.6）

**$\lambda$ ablation**（Figure 7）：最优区间 [0.05, 0.2]，HR 在此区间平台期约 0.3，超出后其他指标指数增长。

**时间窗口 ablation**（Figure 8）：尖锐转变——过早应用无效，过晚应用反而增加 hallucination。FID/HR trade-off 清晰可见：FID 在 hallucination 频繁时反而更好，在 HR 降低时恶化。

**过滤参数 $q$ ablation**（Figure 9）：$q=0$（无条件应用）过度压缩样本到 modes，丧失 variance。$q \approx 0.4$ 是最优——稳定样本不被干扰。GaussianGrid 上 $q=0$ 的 IQ* 把所有样本挤到 mode 中心，几乎 zeroing 每个 Gaussian component 的 variance；$q>0$ 后恢复原始 variance。

**Constant vs Adaptive scaling**（Figure 10）：常数 scaling 在最优 HR 范围内引入 artifacts（过饱和颜色、严重结构扭曲），adaptive scaling 无此问题——验证 $\lambda_t$ 自适应的必要性。

### Runtime（Table 6）

| Method | Runtime (s) |
|--------|-------------|
| Baseline | 11.49 |
| IQ | 49.83 |
| RODS-CAS | 63.86 |
| RODS-SAS | 60.33 |

IQ 比 RODS 快约 20%，但仍比 baseline 慢 4.3×——主要开销来自 autograd + 32 次 noise 采样估计 LID。这是当前主要 limitation。

---

## 12. 医学影像应用：LDCT 重建

### System-Embedded Diffusion Bridges (SDB)

[Sobieski et al. 2026](https://openreview.net/forum?id=cipx3rwfWp) 的 SDB 通过把 measurement 参数嵌入 diffusion coefficients 解决线性 Gaussian inverse problem：

$$\mathbf{H}_t \mathbf{x} = \mathbf{A}^+\mathbf{A}\mathbf{x} + \alpha_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})\mathbf{x}$$

$$\Sigma_t = \gamma_t \mathbf{A}^+\Sigma\mathbf{A}^{+\top} + \beta_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})$$

- $\mathbf{A}$: forward projection matrix（X-ray line integrals）
- $\mathbf{A}^+$: Moore-Penrose pseudoinverse
- $\alpha_t, \beta_t, \gamma_t$: 时间依赖系数

SDB 把 measurement $\mathbf{y}$ 映射到 reconstruction $\mathbf{x}$ 通过 matrix-valued diffusion，保留 measurement range space。

### 实验设置

- 数据：RSNA brain CT scans（[Anouk Stein et al. 2019](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)）
- 任务：low-dose sparse-view CT 重建
- Observer：ResNet50 检测 5 种 hemorrhage 类型，训练于 17,948 ground-truth slices
- 评估：mAP（multi-label Average Precision）和 mROC（multi-label ROC AUC）

### Table 2 结果

| Method | FID↓ | PSNR↑ | SSIM↑ | LPIPS↓ | mAP↑ | mROC↑ |
|--------|------|-------|-------|--------|------|-------|
| Baseline | 33.3 | 33.54 | 0.898 | 0.0387 | 0.27 | 0.85 |
| DG | 35.6 | 33.55 | 0.899 | 0.0388 | 0.31 | 0.86 |
| AAM | 35.6 | 33.54 | 0.898 | 0.0388 | 0.29 | 0.86 |
| RODS-CAS | 33.3 | 33.54 | 0.898 | 0.0388 | 0.29 | 0.86 |
| **IQ** | 33.3 | 33.54 | 0.901 | 0.0388 | **0.31** | 0.86 |

### 关键洞察

- **诊断安全性的 sharp distinction**：所有方法都把 mROC 微提到 0.86，但 IQ 和 DG 把 mAP 提升约 15%（0.27 → 0.31）
- **DG 的不公平优势**：DG 需要独立训练的 classifier，直接访问 ground-truth hemorrhage labels——本质是 supervised guidance
- **IQ 的零样本优势**：尽管完全零样本（不访问 labels），IQ 匹配 DG 的诊断增益，同时严格保持重建质量（PSNR/SSIM/LPIPS 不变）
- **FID 变化**：DG 和 AAM 明显恶化 FID（33.3 → 35.6），因为它们的干预破坏 null space 信息；IQ 和 RODS 保持 FID，因 SDB 严格保留 measurement range space

这指向 IQ 的**通用性**——可直接集成到任意 inverse problem framework（[Chung et al. 2023](https://arxiv.org/abs/2209.14693), [Luo et al. 2023](https://arxiv.org/abs/2306.13-Abstract)）。

---

## 13. Limitations & Broader Implications

### 评估的根本困难

Structural hallucination 的识别依赖人类感知，主观性强。即使独立标注者也常产生分歧。构建完全客观、大规模自动化 benchmark 仍是开放问题。当前 proxy metrics（FID/IV/DSV）与人类偏好错位，尤其在 hallucination 场景：
- FID 偏好 texture over structure（[Geirhos et al. 2018](https://arxiv.org/abs/1811.12231)）
- Diversity metrics 在 hallucination 频繁时反而 inflate（Appendix B.6 ablation 观察到）

### 计算开销

IQ 仍比 baseline 慢 4.3×。降低 LID 估计成本（如更便宜的 estimator）是即时改进方向。当前 $k=32$ noise samples 估计 DSM expectation，可能可降低。

### 与 Memorization 的对称性

[Ross et al. 2025](https://openreview.net/forum?id=aZ1gNJu8wO) 显示 LID 坍塌指示 memorization。本文显示 LID 膨胀指示 hallucination。**LID 是 DM 创造力的 proxy**：

```
LID 坍塌 ←—— memorization（缺乏创造）
          ←—— optimal generation
          → hallucination（过度创造）→ LID 膨胀
```

这给了一个统一的几何框架理解 DM 生成谱。控制 LID 在合理区间可能是 universal regularization 目标。

### Conditional Settings 的开放问题

Unconditional generation 是 cornerstone，但 hallucination 风险最高在 conditional settings（real-world decision-making, diagnosis, [Antun et al. 2020](https://www.pnas.org/doi/10.1073/pnas.1919501117)）。把 IQ 扩展到 conditional DM（text-to-image, image-to-image）是关键未来方向。

---

## 14. 与相关工作的定位

| 方法 | 视角 | 假设 | 关键机制 |
|------|------|------|----------|
| TVF (Aithal) | Temporal | mode interpolation | trajectory variance |
| DG (Triaridis) | Conditional | 需 classifier | classifier guidance to most probable class |
| AAM (Oorloff) | Attention | 需 training data + anomaly detector | attention temperature optimization |
| RODS (Tian) | Continuation | 零样本 | vector field instability intervention |
| **IQ (本文)** | Spatial/Geometric | 零样本 | LID gradient deflation |

IQ 和 RODS 是仅有的两个**对 DM 无额外假设**的方法。IQ 的优势来自更精准的 hallucination 定位（LID 直接测 R2，而 RODS 测 vector field instability，间接）。

---

## 15. 给你的 Intuition 总结

1. **Manifold 不是单一曲面，是 stratified 拼接**——不同 strata 维度不同。手部图像主要在低维 stratum，hallucination 是 model 在不稳定区域"发明"了虚假高维 stratum（允许 6 指）。

2. **LID 是局部有效自由度**——在某个生成点，model 允许多少个独立变化方向。高 LID = model 认为这里有很多 valid 变化 = 易 hallucinate。

3. **Loss 即 LID 估计器**——任何已训练 DM 的 DSM loss 在 $t \to 0$ 时无偏估计 LID。这是 IQ 零样本的基础。

4. **Spectral Bottleneck**——terminal projection 截断 macroscopic flow 的方差。主动降 LID 等于从方差求和中 drop 项，可证明降 LMI 上界。

5. **IQ = Boltzmann Quenching**——修正 score 等价于采样自 Boltzmann 加权分布，惩罚高 LID 终态。热力学淬火的几何对应。

6. **Mode-Seeking 是 emergent property**——Corollary 2 显示 IQ 朝 log-density 局部极大 + 最大负曲率移动，自然消除 mode interpolation。

7. **LID 是创造力 proxy**——坍塌 = memorization，膨胀 = hallucination，中间 = optimal。

8. **Feature metrics lie**——FID/diversity 与人类偏好错位，hallucination 场景下尤其严重。人类评估是 ground truth。

9. **医学影像的 zero-shot 价值**——IQ 不需 labels 即可提升诊断 mAP 15%，匹配 supervised DG。这对临床部署意义重大。

10. **时间窗口很窄**——只在 $t$ 很小时有效，对应 diffusion 的"细节雕刻"阶段。早期干预无效（结构未定），晚期干预有害（已固化）。

---

## 参考链接汇总

- Paper 本身（假设 OpenReview）：https://openreview.net/forum?id=LocalIntrinsicDimension
- Aithal et al. 2024: https://arxiv.org/abs/2406.09313
- Yeats et al. 2025: https://arxiv.org/abs/2501.01127
- Tian et al. 2025 (RODS): https://openreview.net/forum?id=fhuqIxoPcr
- Karras et al. 2022 (EDM): https://arxiv.org/abs/2206.00364
- Ross et al. 2025 (memorization): https://openreview.net/forum?id=aZ1gNJu8wO
- Stanczuk et al. 2024: https://arxiv.org/abs/2402.17563
- Song et al. 2021 (SDE): https://arxiv.org/abs/2011.13456
- Ho et al. 2020 (DDPM): https://arxiv.org/abs/2006.11239
- Sohl-Dickstein 2015: https://arxiv.org/abs/1503.03585
- Pope et al. 2021: https://arxiv.org/abs/2104.02334
- Sobieski et al. 2026 (SDB): https://openreview.net/forum?id=cipx3rwfWp
- Stein et al. 2023 (metrics flaws): https://arxiv.org/abs/2311.04389
- Geirhos et al. 2018 (texture bias): https://arxiv.org/abs/1811.12231
- Antun et al. 2020 (AI instability in reconstruction): https://www.pnas.org/doi/10.1073/pnas.1919501117
- Pidstrigach 2022 (DM detect manifolds): https://arxiv.org/abs/2202.01027
- Fefferman et al. 2016 (manifold hypothesis testing): https://www.ams.org/journals/jams/2016-29-04/S0894-0347-2016-00852-3/
- Levina & Bickel 2004 (LID MLE): https://papers.nips.cc/paper/2004/hash/83982b8d9a03f89d4f9d5f4a94b8f97c-Abstract.html
- Park et al. 2023 (latent space geometry): https://papers.nips.cc/paper_files/paper/2023/hash/24129-Abstract.html
- Sclocchi et al. 2025 (phase transition): https://www.pnas.org/doi/10.1073/pnas.24129
- Efron 2011 (Tweedie's formula): https://www.tandfonline.com/doi/abs/10.1198/016214504000001547
- Chung et al. 2023 (DPS): https://arxiv.org/abs/2209.14693
- RSNA dataset: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

希望这个讲解能 build 出清晰的 geometric intuition——hallucination 不是神秘的"AI 失灵"，是 model-induced manifold 在不稳定区域膨胀了虚假维度，而 IQ 通过热力学淬火把这些维度压回去。
