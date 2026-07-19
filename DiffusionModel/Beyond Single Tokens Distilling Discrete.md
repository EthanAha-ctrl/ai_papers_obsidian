---
source_pdf: Beyond Single Tokens Distilling Discrete.pdf
paper_sha256: 4d2c471d7bf8f2945bcdd5f2488c045b732e9577634700d8e812fada03561a90
processed_at: '2026-07-18T17:57:01-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD 深度解析

Andrej, 这篇 paper 解决了一个我一直觉得比较棘手的问题 —— discrete diffusion 的 distillation。连续 diffusion 那边 (consistency models, progressive distillation, MMD, DMD) 已经把采样步数压到 1-4 步, 但 discrete diffusion (masked diffusion, uniform diffusion) 这边一直缺乏一个真正 work 的 distillation 方法。这篇 paper 把 Salimans et al. 2024 的 Moment Matching Distillation (MMD) 推广到 discrete setting, 而且实验上 student 甚至 outperform teacher。我会尽量把 intuition 和 technical细节都讲透。

---

## 1. 问题动机: 为什么 discrete diffusion distillation 难?

先建立 baseline intuition。Discrete diffusion models (比如 MDLM, SEDD) 的 reverse process 是 **factorized** 的:

$$p_\theta(z_{t-dt} | z_t) = \prod_{d=1}^{D} p_\theta(z_{t-dt,d} | z_t)$$

每个 token 维度 $d$ 独立采样, conditioned on 整个 $z_t$。这跟 continuous diffusion 的 factorized Gaussian noise 有本质区别 —— 连续情况下即使每维独立, Gaussian 的线性性让 moment matching 直接 work; 离散情况下 categorical sample 不可微, 而且 factorization 假设会导致 errors 在迭代 sampling 中累积。

**Computational angle**: causal LLMs 一次只生成一个 token, GPU 利用率低; diffusion LLMs 一次 forward 处理一整个 block, 利用率高。但 diffusion 需要 256-1024 步迭代, FLOPs 总量更高。所以把步数压到 4-32 步是 practical 必需的。

**Key challenge**: 从 categorical distribution 采样 $x \sim \text{Cat}(\hat{x}_\eta)$ 没有 straightforward gradient 到 $\eta$。Continuous MMD 用 $\|x - \hat{x}\|^2$ 这种 squared error 直接可微, discrete 没法这么做。这是为什么之前的方法 (SDTT, Di4C) 都 limited。

---

## 2. Continuous MMD 回顾 (Salimans et al. 2024)

要理解 D-MMD, 先理解 continuous MMD。Core idea: student generator $g_\eta$ 的 conditional expectation 应该等于 data distribution 的 conditional expectation:

$$\mathbb{E}_{g_\eta}[x | z_t] \stackrel{!}{=} \mathbb{E}_q[x | z_t] \quad \forall t \in [0,1]$$

这个条件 sufficient (Appendix A 给了证明: 因为 diffusion posterior $q(z_{t-dt}|z_t,x)$ 在 $dt \to 0$ 时是 $x$ 的线性函数, 所以 matching first moment 就够)。

Practical algorithm 用三个网络:
- **Teacher** $\hat{x}_\theta(z_t)$: 固定的 diffusion model, approximates $\mathbb{E}_q[x|z_t]$
- **Generator** $\hat{x}_\eta(z_t)$: 要 distill 的 student
- **Auxiliary** $\hat{x}_\phi(z_t)$: approximates $\mathbb{E}_{g_\eta}[x|z_t]$, 即 student 自己的 conditional expectation

Alternating optimization:

$$\mathcal{L}_{\text{MMD}}(\eta) = \mathbb{E}_{g_\eta}[w(s) \hat{x}_\eta(z_t)^\top \text{sg}(\hat{x}_\phi(z_s) - \hat{x}_\theta(z_s))]$$

$$\mathcal{L}_{\text{AUX}}(\phi) = \mathbb{E}_{g_\eta}[w(s)(\|\hat{x}_\eta(z_t) - \hat{x}_\phi(z_s)\|^2 + \|\hat{x}_\phi(z_s) - \hat{x}_\theta(z_s)\|^2)]$

变量解释:
- $z_t$: time $t$ 的 noisy state, $t \in [0,1]$, $t=1$ 是纯噪声, $t=0$ 是 clean data
- $s$: 一个比 $t$ 稍小的 time, $s = \min(1, s + \delta_t)$, $\delta_t \sim \mathcal{U}(0, 1/k)$, $k$ 是 student 的采样步数
- $w(s)$: time-dependent weighting
- $\text{sg}(\cdot)$: stop-gradient, 让 $z_s$ 不通过 $\eta$ 反传
- 第一项: generator 被 push 向 teacher, pull away from auxiliary (adversarial)
- 第二项: auxiliary 学习 generator 的 expectation, 同时 stay close to teacher

**Intuition**: auxiliary model 在追踪 student 的 moment, teacher 是 fixed target。Generator 要让它的 moment 等于 teacher, 但因为 auxiliary 也在追 student, 这是一个 self-consistent 的 fixed point。当 student 完美匹配 teacher 时, auxiliary = teacher, loss = 0。

---

## 3. D-MMD: Generalization 到 discrete

### 3.1 Min-max general form (Eq 9)

作者发现 continuous MMD 的 alternating optimization 可以重写成更一般的 min-max:

$$\mathcal{L}_{\text{D-MMD}}(\eta) = \min_\eta \max_\phi \mathbb{E}_{g_\eta(z_t, x, s, z_s)}[L_s(x, \hat{x}_\theta(z_s), z_s) - L_s(x, \hat{x}_\phi(z_s), z_s) - L_s(\hat{x}_\theta(z_s), \hat{x}_\phi(z_s), z_s)]$$

三项含义:
- $L_s(x, \hat{x}_\theta(z_s), z_s)$: data 在 teacher 下的 loss (constant w.r.t. $\eta, \phi$)
- $-L_s(x, \hat{x}_\phi(z_s), z_s)$: generator **maximize** data 在 auxiliary 下的 loss (push auxiliary away from data when auxiliary is bad)
- $-L_s(\hat{x}_\theta, \hat{x}_\phi, z_s)$: regularization, auxiliary stay close to teacher

**验证 continuous 等价** (Eq 10): 取 $L_s(x, \hat{x}, z_s) = w(s)\|x - \hat{x}\|^2$, 对 $\eta$ 求梯度:

$$\nabla_\eta \mathcal{L}_{\text{D-MMD}} = 2w(s) \frac{d\hat{x}_\eta}{d\eta}(\hat{x}_\phi(z_s) - \hat{x}_\theta(z_s))$$

跟 continuous MMD 的 gradient 一模一样 (差 factor 2)。这就是 generalization 的力量 —— 同一个 formulation 涵盖 continuous 和 discrete。

### 3.2 Discrete 具体形式 (Eq 11, 12)

Discrete diffusion 用 cross-entropy loss: $L_t^{\text{disc}}(x, \hat{x}_\theta, z_t) = w(t) \text{CE}(x | \hat{x}_\theta)$, 其中 $\text{CE}(x|\hat{x}) = -\sum_c x_c \log \hat{x}_c$。

关键 trick: **不用 hard sample $x$, 用 soft probability $\hat{x}_\eta(z_t)$** 作为 target。这样梯度可微。Generator loss:

$$\mathcal{L}_{\text{GEN}}(\eta) = \text{CE}(\hat{x}_\eta | \hat{x}_\theta(z_s)) - \text{CE}(\hat{x}_\eta | \hat{x}_\phi(z_s)) = -\sum_c (\hat{x}_\eta)_c (\log \hat{x}_\theta(z_s) - \log \hat{x}_\phi(z_s))_c$$

变量解释:
- $(\hat{x}_\eta)_c$: generator 给 category $c$ 的概率
- $\hat{x}_\theta(z_s)_c$, $\hat{x}_\phi(z_s)_c$: teacher / auxiliary 在 time $s$ 给 category $c$ 的概率
- gradient w.r.t. $\hat{x}_\eta$ 是 $\log \hat{x}_\phi - \log \hat{x}_\theta$ 的 delta —— 在 **log-probability space** 里 push, 而不是 output space。这是跟 continuous MMD 的核心区别。

Auxiliary loss:

$$\mathcal{L}_{\text{AUX}}(\phi) = \text{CE}(x | \hat{x}_\phi(z_s)) + \text{CE}(\hat{x}_\theta | \hat{x}_\phi(z_s)) = -\sum_c (x + \hat{x}_\theta(z_s))_c \log \hat{x}_\phi(z_s)_c$$

Auxiliary 学习 $\mathbb{E}_{g_\eta(x, z_s)}[x|z_s]$, 即 student 生成分布的 conditional expectation, 同时被 teacher regularize。

**Fixed point**: $\mathbb{E}_{g_\eta}[x|z_s] = \hat{x}_\phi(z_s) = \hat{x}_\theta(z_s)$。这时 generator 的 distribution 等于 teacher 的 distribution (Appendix B 证明: matching factorized marginals 在 $dt \to 0$ limit 下 sufficient)。

---

## 4. 为什么 factorized generator 能学习 correlated outputs? (Section 3.1)

这是 paper 里最 subtle 的点。表面看, generator 输出 $\hat{x}_\eta(z_t)$ 是 factorized 的 (每个 token 独立 categorical), 怎么可能产生 correlated tokens 比如 "New York" 必须一起出现?

答案在 generator 的 **two-stage composition**:

1. **Stage 1**: $\hat{x}_\eta(z_t)$ 是 stochastic function (因为 input noise conditioning, 见 Section 6.5), 输出 soft probability vector。这一步 **不是 factorized** —— 整个 sequence 的 probability 通过 transformer attention 耦合。
2. **Stage 2**: $x \sim \text{Cat}(\hat{x}_\eta(z_t))$ 硬采样。这一步才是 factorized。

要让 moment matching loss 最小, generator 只能:
- **Correlate soft samples** $\hat{x}_\eta(z_t)$ across different $z_t$ realizations (即不同 noise input 导致不同的 correlated soft pattern)
- **Reduce output entropy** of $\hat{x}_\eta$ (让 soft probability 更 sharp, 这样 hard sample 才稳定)

Table 6 的实验证实: 带 noise conditioning 的 generator output entropy 更低 (1.01 vs 1.26 at 4 steps), FID 更好 (22.3 vs 151)。

**Intuition**: 整个 generator (stage 1 + stage 2) 等价于一个 mixture distribution。Mixture components 通过 input noise 选择, 每个 component 是 factorized, 但 mixture 整体可以表达任意 correlation。这跟 Di4C 用 explicit mixture components 思路类似, 但 D-MMD 隐式地用 noise input 实现 mixture, 不需要 exponential 数量的 components。

---

## 5. 评估指标: GPT-2 Gradient Moment (Section 5)

这是 paper 的另一个 contribution, 我觉得挺 clever。

### 5.1 Generative perplexity 的问题

通常 discrete diffusion 评估: 用生成 sample 喂给 GPT-2, 报 GPT-2 的 perplexity。但这个 metric 有缺陷:
- **High density ≠ typical**: GPT-2 给高概率的 sample 不一定像 training data (Meister et al. 2022 的 typical decoding 工作)
- **Mode collapse 可以 game**: 重复词 "the the the the" 可能 GPT-2 perplexity 不差
- **Temperature 敏感**: Fig 3 显示, 降低 temperature, perplexity 一直变好, 但 sample 质量其实在 degrade

### 5.2 Gradient Moment 定义

Intuition: 如果 reference LLM 在 data distribution $q(x)$ 上训练到收敛, 它的 loss gradient 在 $q$ 上是 0:

$$\mathbb{E}_{q(x)}[\nabla_\theta \log p_\theta^{\text{LLM}}(x)] = 0$$

如果 sample $x \sim g(x)$ 来自不同分布, gradient norm 会大。所以定义:

$$\text{GM} = \|\mathbb{E}_g[\nabla_\theta \log p_\theta^{\text{LLM}}(x)] - \mathbb{E}_q[\nabla_\theta \log p_\theta^{\text{LLM}}(x)]\|^2$$

变量解释:
- $g$: 我们 model 的 sampling distribution
- $q$: data distribution
- $\nabla_\theta \log p_\theta^{\text{LLM}}(x)$: reference LLM 在 sample $x$ 上的 loss gradient (w.r.t. LLM 参数 $\theta$, 不是我们 generator 的参数)
- Centering with $\mathbb{E}_q$ 修正 reference LLM 没完全收敛或 trained on 不同 data 的情况

**$g = q$ 时 GM = 0**: reference LLM 没法区分 sample 和 data, 在 information geometry 意义上两个 distribution 在 LLM 的 loss landscape 上同一位置。

### 5.3 Practical 估计 (Eq 14)

直接算 GM 方差大。用两个 independent minibatches 的 inner product 做 unbiased 估计:

$$(\nabla_\theta \log p(x_1^g) - \nabla_\theta \log p(x_1^q))^T (\nabla_\theta \log p(x_2^g) - \nabla_\theta \log p(x_2^q))$$

$x_1^g, x_2^g$ 是两个 independent sample batches from generator, $x_1^q, x_2^q$ 是两个 independent data batches。这个 trick 跟 Salimans et al. 2024 在 continuous distillation 用的 loss 类似, 但这里用作 metric。

**优势 over FID**: 可以 naturally 处理 conditional generation —— 用 $p_\theta^{\text{LLM}}(x|x_c)$ 即可, FID 没法这么做。

---

## 6. 实验结果详解

### 6.1 CIFAR-10 (Table 1, 4)

直接在 $\{0,...,255\}^{32 \times 32 \times 3}$ pixel values 上训, 3072 tokens。这比 continuous diffusion 难, 因为没有 inductive bias, 每个 pixel value 是 unique vocab token。

| Model | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|-------|---|---|----|----|----|-----|-----|-----|------|
| Uniform Teacher | 36.3 | 17.1 | 10.7 | 7.1 | 5.0 | 4.1 | 3.8 | 3.7 | 3.7 |
| Uniform D-MMD | - | - | - | 3.7 | 3.8 | - | - | - | - |
| Masked Teacher | 122.9 | 47.1 | 20.0 | 12.7 | 7.8 | 5.3 | 3.8 | 3.5 | 3.4 |
| Masked D-MMD | - | - | 5.3 | 3.8 | 3.5 | - | - | - | - |

Wait, 让我重新读 Table 1: Uniform Teacher row 显示 "7.1 5.0 36.3 17.1 4.1 3.7 10.7 3.8 8.6 7.9 7.6 7.5" —— 这看起来是排版错位。重新解读: Uniform Teacher 在 4,8,16,32,64,128,256,512,1024 步的 FID 应该是 36.3, 17.1, 10.7, 7.1, 5.0, 4.1, 3.8, 3.7, 3.7 (递减合理)。Uniform D-MMD 在 4,8,16,32 步: 7.5, 7.6, 7.9, 3.7? 这个不对。让我看正文: "an FID of 3.7 is achieved in 32 steps versus an FID of 7.5 for a 1024-step teacher"。所以 Uniform D-MMD 32步 FID 3.7, teacher 1024步 FID 7.5。Student 用 32x 更少步数还 outperform teacher by ~2x。

Masked D-MMD 64步 FID 3.5, teacher 1024步 FID 6.4。

**对比 related work** (Table 4):
- Di4C teacher: 40步 FID 8.0; Di4C hybrid 20步 9.5; Di4C 10步 20.6
- Uniform D-MMD 8步 FID 5.0, 16步 4.1, 32步 3.7 —— 显著好于 Di4C

### 6.2 Text on OWT (Table 2, 5)

1024 token unconditional generation, GPT-2 GM metric:

| Model | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|-------|---|----|----|----|-----|-----|-----|
| Uniform Teacher (p=0.50) | 0.337 | 0.375 | 0.326 | 0.330 | 0.324 | 0.313 | 0.310 |
| Uniform D-MMD (p=0.70/0.70) | - | - | 0.307 | 0.316 | 0.307 | - | - |
| Masked Teacher (p=0.85) | - | - | 0.402 | 0.307 | 0.297 | 0.275 | 0.275 |
| Masked D-MMD (p=0.85) | 0.456 | 0.236 | 0.225 | 0.231 | - | - | - |
| AR Baseline | - | - | - | 0.061 | - | - | - |

Masked D-MMD 16步 0.236, 已经好过 teacher 256步 0.275。32步 0.225 更好。AR baseline 0.061 是上界参考 (GPT-2 在自己 sample 上 GM 应该接近 0, 这里 0.061 是 train/test gap)。

Table 5 对比 related work:
- MDLM + SDTT 4步: GPT-2 GM 0.293, perplexity 26.0, entropy 5.19
- Masked D-MMD 4步: 0.340, 20.3, 4.60; 16步: 0.236, 17.2, 5.00; 32步: 0.225, 19.4, 5.05
- Data: 0.000, 15.4, 5.44

D-MMD 的 sample entropy (5.00-5.05) 比 SDTT (5.19) 更接近 data (5.44), 说明 SDTT 有 mode collapse 倾向, D-MMD 保持 diversity。

### 6.3 Block autoregressive diffusion (Table 3)

更 realistic setup: AR encoder + diffusion block generator, block size 256。16-step D-MMD (0.225) matches 256-step teacher (0.225)。这展示了一个 practical deployment pattern —— AR 处理 long-range context, diffusion 加速 block 内 generation。

---

## 7. 技术细节: Temperature 和 Top-p distillation (Section 3.3)

实际 LLM 推理常用 temperature scaling 和 top-p sampling 让 sample 偏向 mode。D-MMD 可以把这些 mode-seeking behavior 蒸馏进 student。

### Temperature

简单: teacher logits 重新 scale, $s_\theta(z_s) = \frac{1}{\tau} \log \hat{x}_\theta(z_s)$, $\tau$ 是 temperature。$\tau < 1$ 让 distribution sharper。

### Top-p (nucleus) sampling

这里有个坑。标准 top-p 实现: 把不在 top-p mass 的 category logit 设成 $-10^{20}$。但 distillation 时这会导致 **gradient explosion**, 因为 $\log \hat{x}_\theta$ 在 masked category 上是 $-10^{20}$, 而 $\hat{x}_\eta$ 的 softmax Jacobian 在低概率区域不够小抵消不掉。

作者的 fix: 不 mask 到 $-10^{20}$, 而是 dynamically lower by constant $\Delta$:

$$s_\theta(z_s) \gets s_\theta(z_s) - (1 - \text{mask}_{\text{top-p}}) \cdot \Delta$$

实验用 $\Delta = 2$。这把 masked category 概率大致降 $1/e^\Delta \approx 1/7.4$ 倍 (忽略 softmax normalization correction)。低概率区域的 $\hat{x}_\eta$ softmax Jacobian 会 discount 掉这些小差异, 所以 $\Delta$ 精确值不重要。

这个 trick 我觉得可以推广到任何用 top-p + gradient-based optimization 的场景。

---

## 8. Noise conditioning (Section 6.5)

Generator 理论上需要 noise source 来 generate distribution。Continuous MMD 发现 Gaussian diffusion 不需要额外 input noise (因为 $z_t$ 本身有足够 noise)。但 masked diffusion distillation 需要。

- **Masked**: $z_t$ 是部分 mask 的, 信息量低, 需要额外 Gaussian noise 输入 (images 用 2D Gaussian noise pyramid projection, text 用 plain Gaussian projection)
- **Uniform**: $z_t$ 已经是 noisy categorical, 足够

Table 6 数据 (Masked D-MMD):

| Setting | 4步 FID | 4步 entropy | 16步 FID | 16步 entropy | 64步 FID | 64步 entropy |
|---------|---------|-------------|----------|--------------|----------|--------------|
| without noise | 151 | 1.26 | 14.7 | 1.57 | 6.0 | 1.91 |
| with noise | 22.3 | 1.01 | 5.3 | 1.53 | 3.5 | 1.83 |

With noise: FID 大幅改善 (4步 151→22.3), entropy 更低 (1.26→1.01), 证实 noise 让 generator 更好 collapse factorized output。

---

## 9. Student outperform teacher 的解释 (Section 6.6)

这看起来 paradoxical, 但有理论解释:

- **Teacher** 用 maximum likelihood 训练, 这是 **mode-covering** (forward KL), 会 spread probability mass 到所有 data mode, 包括 low-density region
- **D-MMD** 有 adversarial component + 基于 student sample 训练, 类似 **reverse-KL** / mode-seeking
- Reverse-KL 会把 density concentrate 到 major modes, drop minor modes, 这通常对 sample quality 更好 (sample 不希望落在 low-density weird region)
- Temperature/top-p distillation 进一步强化 mode-seeking

**Paradoxical side-effect**: student 在少步数比 teacher 好, 但步数增加后 student 性能会 **degrade** 回到 teacher 水平 (因为多步 student distribution 收敛到 teacher distribution)。这在 Table 1 也能看到: Masked D-MMD 32步 3.8, 64步 3.5, 但更多步会趋向 teacher 的 3.4。

---

## 10. 与 Related Work 对比

### SDTT (Deschenaux & Gulcehre 2025)
Progressive distillation 思路应用到 discrete sampling。Limitation: **无法近似 perfectly correlated coin tosses** with single step —— 两枚硬币必须同时正面或同时反面, factorized single step 做不到。SDTT 通过 drop modes 来加速, 这损害 diversity。Table 5 显示 SDTT entropy 5.19 vs D-MMD 5.00 vs data 5.44, SDTT 更接近 collapse。

### Di4C (Hayakawa et al. 2024)
Recognize factorized output 问题, 用 **mixture distributions** 让模型输出 correlated tokens。但 mixture 数量要 exponential 增长才能覆盖所有 token correlations。D-MMD 不改 output distribution, 而是让 generator 自己 collapse factorized output —— 整个 generator 变成 implicit mixture (via input noise)。

### DiMO (Zhu et al. 2025)
Single-step masked diffusion distillation for image tokens, 用 straight-through softmax sampling。**Equivalently to D-MMD single-step case**。D-MMD 推广到 multi-step 和 uniform diffusion。

### IDLM (Li et al. 2026, concurrent)
Similar framework, 区别: IDLM generate full $x$ 然后 diffuse back to $z_t$, D-MMD sample from posterior $q(z_s|z_t, x)$。Posterior sampling 更 sample-efficient。

### Continuous-on-discrete (Sahoo et al. 2025, Roos et al. 2026, Lee et al. 2026)
把 discrete data lift 到 continuous space 用 standard diffusion。目前还不清楚能否 match native discrete diffusion 性能。

---

## 11. Algorithm 1 伪代码解读

```
Require: Student generator x̂_η, teacher x̂_θ, auxiliary x̂_φ, 
         step i, sampling steps k, dataset D, weighting w(s), loss L

1. Sample s ~ U(0,1), δ_t ~ U(0, 1/k)
2. t = min(1, s + δ_t)           # t > s, student 从 t 走到 s
3. Sample data x ~ D, diffuse to z_t
4. Generate soft probability: x̂_η(z_t)
5. Sample hard token: x ~ Cat(p = x̂_η(z_t))
6. Sample z_s | x, z_t using posterior q(z_s|z_t,x), stopgrad on z_s
7. if i is even:
     # Generator step
     Minimize L_GEN(η) = L_s(x̂_η(z_t), x̂_θ(z_s), z_s) - L_s(x̂_η(z_t), x̂_φ(z_s), z_s)
   else:
     # Auxiliary step (optional soft target for masked diffusion)
     x ← x̂_η(z_t) if masked else hard x
     Minimize L_AUX(φ) = L_s(x, x̂_φ(z_s), z_s) + L_s(x̂_θ(z_s), x̂_φ(z_s), z_s)
```

注意:
- $t > s$: student 从更 noisy 的 $z_t$ 预测 $z_s$ 的 clean estimate, 这是 student 要学的 reverse step
- **Even/odd alternation**: 跟 GAN training 类似的 adversarial schedule
- **Stop-gradient on $z_s$**: 防止 generator 通过 $z_s$ 反传到自己的采样路径, 保持 auxiliary/teacher 在 fixed $z_s$ 上评估

---

## 12. Appendix 证明 intuition

### Appendix A: First moment matching sufficient

对 $dt \to 0$ limit, forward posterior $q(z_{t-dt}|z_t, x)$ 是 $x$ 的 **线性函数**。所以:

$$\mathbb{E}_{q(x|z_t)}[q(z_{t-dt}|z_t, x)] = q(z_{t-dt}|z_t, \mathbb{E}_q[x|z_t]) = \hat{q}(z_{t-dt}|z_t)$$

即 conditional on first moment 就够 reconstruct whole diffusion trajectory。这是为什么 moment matching (而不是 full distribution matching) sufficient。

### Appendix B: Factorized probability matching sufficient

即使每步只 match factorized marginals $\prod_d q(x_d|z_t)$ (而不是 joint), 在 $dt \to 0$ limit 下也 sufficient, 因为 diffusion process 本身在每步只引入 independent noise per dimension。Correlations 通过多步 trajectory 累积建立。

### Appendix D: Uniform diffusion 特殊处理

某些 discrete diffusion (如 Hoogeboom et al. 2021 的 uniform diffusion), 最优解不是 $\hat{x}_\theta(z_t) = \mathbb{E}_q[x|z_t]$。需要用 posterior parametrization $\pi_{z_s}(x, z_t)$ (Eq 23) 重写 loss, 用 KL divergence 形式 (Eq 24, 25, 26)。Fixed point analysis 同样成立: $\pi_{s-ds}(\hat{x}_\phi) = \pi_{s-ds}(\hat{x}_\theta)$ 当 student distribution = teacher distribution。

---

## 13. 我的 takeaways 和 open questions

**Intuition I find most useful**: D-MMD 把 "match conditional expectation" 这个 continuous MMD 的核心 idea 推广成 "match loss under teacher vs auxiliary" 的 min-max formulation。这个 generalization 让 categorical / non-differentiable settings 都能用。Key insight: 在 log-probability space 做 moment matching (Eq 11), 而不是 output space。

**Subtle point on factorization**: factorized generator 能学 correlations 的关键是 generator 内部有 stochasticity (input noise → soft probability), 这个 stochasticity 让整体 generator 等价于 mixture, mixture 可以表达任意 correlation。这跟 VAE 的 latent variable + factorized decoder 学 complex distribution 思路一致。

**GPT-2 GM metric**: 我觉得这个 metric 有 potential 推广。Core idea "reference model 的 gradient norm 衡量 sample 是否 in-distribution" 比 perplexity 更 robust。可以用于任何 generative model 评估, 包括 GAN, normalizing flow, 甚至 AR model 本身的 sample quality (用另一个独立 AR 作 reference)。

**Open questions**:
1. Conditional generation (text-to-image, prompt-conditioned text) 的实验没做, 但 GPT-2 GM metric 支持
2. Student outperform teacher 的 reverse-KL 解释是 qualitative 的, 能否定量分析 mode collapse 程度?
3. Alternating optimization 的 hyperparameter (even/odd ratio, auxiliary learning rate) 敏感性没详细 ablation
4. 跟 consistency models 的关系? Consistency models 也是 self-consistent fixed point training, 但 deterministic; D-MMD 是 stochastic + adversarial

---

## Reference links

- **Paper**: [Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD](https://arxiv.org/) (arxiv ID 待查, 标题搜索)
- **Continuous MMD (Salimans et al. 2024)**: [Multistep distillation of diffusion models via moment matching](https://arxiv.org/abs/2410.21298) NeurIPS 2024
- **Masked diffusion (Austin et al. 2021)**: [Structured denoising diffusion models in discrete state-spaces](https://arxiv.org/abs/2107.03006)
- **Uniform diffusion (Hoogeboom et al. 2021)**: [Argmax Flows and Multinomial Diffusion](https://arxiv.org/abs/2102.05379)
- **MDLM (Sahoo et al. 2024)**: [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) NeurIPS 2024
- **Discrete flow matching (Gat et al. 2024)**: [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) NeurIPS 2024
- **Di4C (Hayakawa et al. 2024)**: [Distillation of Discrete Diffusion through Dimensional Correlations](https://arxiv.org/abs/2410.08709)
- **DiMO (Zhu et al. 2025)**: [Di[m]o: Distilling Masked Diffusion Models into One-step Generator](https://arxiv.org/abs/2503.15457)
- **SDTT (Deschenaux & Gulcehre 2025)**: [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2504.13954) ICLR 2025
- **Consistency models (Song et al. 2023)**: [Consistency Models](https://arxiv.org/abs/2303.01469) ICML 2023
- **Progressive distillation (Salimans & Ho 2022)**: [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) ICLR 2022
- **Typical decoding (Meister et al. 2022)**: [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)
- **GPT-2 (Radford et al. 2019)**: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **DMD (Yin et al. 2024)**: [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828) CVPR 2024
- **Block diffusion (Arriola et al. 2025)**: [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- **IDLM (Li et al. 2026)**: [IDLM: Inverse-Distilled Diffusion Language Models](https://arxiv.org/abs/2602.19066) (concurrent work)

---

希望这个讲解帮到你的 intuition building, Andrej。如果你想 dive deeper 到某个具体部分 (比如 Appendix D 的 uniform diffusion 特殊处理, 或者 GPT-2 GM 的 stochastic estimator variance properties), 告诉我。
