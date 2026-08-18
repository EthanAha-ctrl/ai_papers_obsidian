---
source_pdf: Flash-WAM Modality-Aware Distillation for World Action Models.pdf
paper_sha256: 186f859f2dca5049393e47cc3207d0f66021b377516c24a00e0be6cdcb5ac678
processed_at: '2026-08-18T13:11:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Flash-WAM

## 先讲个故事

想象你是个机器人，正在准备抓桌上那个红瓶子。

每动一下之前，你得先**做梦**——在脑子里想象未来几秒钟会发生什么画面：手伸过去、碰到瓶子、握住、提起来。然后你从梦里**反推**自己每一步该做什么动作。

这就是 WAM（World-Action Model）的工作方式：先 generate video，再从 video 里 decode actions。

问题在于，这个"梦"和"动作"都不是一步想出来的。Diffusion model 的工作原理是：从一团纯噪声开始，一次去掉一点雾，几十次之后雾散干净，看清画面。LingBot-VA 这个 SOTA 模型，video 要 25 步去雾，action 要 50 步去雾，加起来 8.1 秒。

机器人控制圈里有个共识：要 real-time，每步反应得在 500ms 内。8.1 秒？你手都伸过去了机器人还没想完第一步。

## 加速的老办法：distillation

图像和视频生成圈早就解决了类似问题。核心 idea 叫 **consistency distillation**：让一个 student 网络学 teacher 的行为，但 student 只需要一两步就能出图，不用几十步。

直觉上，teacher 像老司机开山路，几十次微操方向盘；student 被训练成"看到任何弯道状态，直接知道终点在哪"，一步到位。

这技术在图像（[LCM](https://arxiv.org/abs/2310.04378)）、视频（[VideoLCM](https://arxiv.org/abs/2312.09109)）上都 work 得很好。

## 但搬到机器人上直接崩

Flash-WAM 这篇 paper 第一句话的潜台词就是：**我们把 LCM 直接套到 WAM 上，success rate 从 91% 掉到 24%**。

这奇怪吗？太奇怪了。同一个 backbone，同样的 flow matching 框架，同样的 distillation loss，为什么视频 work 而机器人崩？

Paper 的核心贡献就是搞清楚这件事。

## 关键：两种"雾"的浓度不一样

WAM 里 video 和 action 用的 noise schedule 是不同的——这本来是 WAM 自己的 design choice，不是 bug。

Video latent 是高维数据，128×128 经过 VAE 还有几千维，空间上有大量冗余。所以可以容忍每步加大噪声，schedule 把训练 mass 推到高噪声区域（参数 $s^v = 5.0$）。

Action 是 30 维的关节角度，几个弧度不对就抓空。它需要 precision，所以 schedule 是温和的，噪声分布几乎均匀铺在 [0,1] 上（$s^a = 1.0$）。

**人话翻译**：video 的雾永远很浓，action 的雾稀浓都有，特别在稀雾（低噪声）区域有很多训练样本。

## Consistency function 的数学结构

任何 consistency function 都长这样：

$$f(\mathbf{x}_\sigma, \sigma) = a(\sigma)\mathbf{x}_\sigma + b(\sigma)v_\theta(\mathbf{x}_\sigma, \sigma)$$

变量解释：
- $\mathbf{x}_\sigma$：当前 noisy 状态
- $v_\theta$：网络预测的 velocity（要去噪的方向）
- $a(\sigma), b(\sigma)$：两个标量函数，决定怎么 mix input 和 network prediction
- 边界条件：$\sigma=0$ 时 $a=1, b=0$，保证干净数据映射回自己

**关键洞察**：网络参数 $\theta$ 只通过 $v_\theta$ 进入 $f$。所以梯度 $\nabla_\theta \mathcal{L}$ 的大小 pointwise 正比于 $|b(\sigma)|$。

**翻译成人话**：$b(\sigma)$ 是个"音量旋钮"，决定网络在 noise level $\sigma$ 处听得到多大的训练信号。$b$ 小的地方，网络就聋了，学不动。

## LCM 的致命伤：低噪声区域失聪

LCM 选的 $b(\sigma) = -\sigma^2 \sigma_d / \sqrt{\sigma^2 + \sigma_d^2}$，$\sigma_d = 0.5$。

在 $\sigma = 0$ 附近做 Taylor 展开：
- $b(0) = 0$ ✓（满足边界条件）
- $b'(0) = 0$ ← **这是问题**
- $|b(\sigma)| \approx \sigma^2 / \sigma_d$，**二次衰减**

在 $\sigma = 0.1$ 时，$|b|$ 比 high-σ 区域小 **36 倍**。意思是：低噪声区域，网络几乎拿不到梯度。

Paper 给了个 Proposition 1 证明这不是 LCM 的 specific 问题，是整个 family 的结构性限制：任何 $C^1$ consistency function 在 $\sigma \to 0$ 的最优 scaling 是 $O(\sigma)$（linear），当且仅当 $b'(0) \neq 0$ 才达到。LCM 偏偏 $b'(0) = 0$，退化到 $O(\sigma^2)$。

## Action stream 正好踩在 LCM 的死穴上

Action 的训练 mass 在低噪声区域很重（因为 $s^a = 1.0$，均匀分布）。LCM 在低噪声区域梯度二次衰减。结果：**action head 在它最常看到 noise level 上学不到东西**。

这就解释了为什么 Naive Joint LCM 崩了。Video 没崩是因为 $s^v = 5.0$ 把 mass 推到高噪声区域，LCM 在那里信号充足。

Paper Table 4 把这个体现得淋漓尽致：Naive Joint LCM 在 horizon=1 还有 41%，horizon=2 掉到 4%，horizon=3 是 0%。Action 误差累积是 cubic 增长。

## Flash-WAM 的 fix：给两边配不同眼镜

### Action stream：最朴素的 linear scaling

最简单的 $b'(0) \neq 0$ 的选择：

$$a(\sigma) = 1, \quad b(\sigma) = -\sigma$$

代进 consistency function：

$$f^a(\mathbf{x}_\sigma^a, \sigma) = \mathbf{x}_\sigma^a - \sigma \cdot v_\theta(\mathbf{x}_\sigma^a, \sigma)$$

你看，这不就是 flow matching 里从 noisy 估 clean 的标准公式 $\hat{\mathbf{x}}_0 = \mathbf{x}_\sigma - \sigma v_\theta$ 嘛。

但 paper 的概念贡献是把它**重新解读为 consistency function family 中的 canonical low-σ realization**——由 framework 的 matching principle 自然选出来，不是随便挑的 parametrization。

性质：
- $|b(\sigma)| = \sigma$ 全程线性，达到 Proposition 1 的 lower bound
- 没有 hyperparameter（LCM 的 $\sigma_d$ 在这里是多余的）
- $a(\sigma) = 1$ 常数，consistency target 不被 skip term 衰减

### Video stream：保持 Karras parametrization

Video 在高噪声区域训练，linear scaling 在这里反而有害：
- $\text{Var}[\mathbf{x}_\sigma - \sigma v_\theta]$ 随 $\sigma$ 增长，把 prediction error 放大 $\sigma$ 倍
- 输出 unbounded，early training 容易 drift 出 manifold

Flash-WAM 保留 LCM 原本的 Karras parametrization：

$$f^v(\mathbf{x}_\sigma^v, \sigma) = c_{\text{skip}}(\sigma)\mathbf{x}_\sigma^v + c_{\text{out}}(\sigma)\hat{\mathbf{x}}_0^v$$

其中 $c_{\text{skip}} = \sigma_d^2/(\sigma^2 + \sigma_d^2)$，$c_{\text{out}} = \sigma\sigma_d/\sqrt{\sigma^2+\sigma_d^2}$。

性质：
- Variance preservation：$\text{Var}[f] \approx \sigma_d^2$ 恒定
- 高噪声时 $c_{\text{out}} \to \sigma_d$，输出有界
- 这正是 [Karras et al. 2022](https://arxiv.org/abs/2206.00364) 设计的初衷

### Joint training

总 loss 是两边的和：

$$\mathcal{L} = \mathcal{L}^v + \lambda_a \mathcal{L}^a$$

每个 modality 用自己的 consistency function 算 consistency loss。两边 target 都来自 teacher 的 guided Euler step。CFG 只用在 video teacher（$w \sim \mathcal{U}[2.0, 10.0]$），action 用 unguided prediction——因为 video 需要语言 condition，action 是 deterministic inverse dynamics。

**架构零改动**：video 和 action tokens 仍 concatenate 进同一个 transformer，flex attention 处理 block-causal mask。两个 consistency function 只影响 per-stream loss head。

## 结果

RoboTwin 2.0（50 个 bimanual 任务）：

| 配置 | 成功率 | 加速 |
|---|---|---|
| LingBot-VA teacher (25v/50a) | 91.25% | 1× |
| Flash-WAM (1v/2a) | 85.54% | 19× |
| Flash-WAM (1v/1a) | 81.41% | 23.3× |
| Naive Joint LCM (1v/2a) | 23.97% | — |
| Video-only LCM (1v/1a) | 73.68% | 23.3× |

23× 加速把 8.1 秒压到 348ms，跨过 500ms 的 real-time 门槛。

真机 Unitree G1 humanoid（3 个 manipulation 任务）：
- Teacher (3v/10a)：66.7%
- Flash-WAM (1v/2a)：60.0%
- Video-only LCM (1v/2a)：43.3%
- LingBot-VA reduced NFE (1v/2a)：40.0%

最难的 T1（开锅盖放土豆）在 1v/1a 时 reduced NFE 只有 10%，Flash-WAM 拉到 40%——这个任务最依赖 action precision，linear-gradient-scaling 在低噪声区域的优势最显著。

## 直觉总结

**一句人话**：机器人要同时想象画面和决定动作，两种东西的噪声分布不一样浓。常用的 distillation 方法在"稀雾区域"（动作的栖息地）几乎听不见训练信号，所以动作学不会；给动作单独配一副能听清稀雾的"眼镜"，给画面保留原来那副，两边各得其所，一步到位。

**更深一层的 intuition**：这其实是 modality asymmetry 在 shared-backbone 模型里的必然表现。Video 和 action 共享 backbone，但 loss head 必须按各 modality 的 noise regime 单独 design。这跟 MoE 的 expert specialization 是不同维度的同一种思想——backbone 共享、head 专精。

**为什么 linear $b = -\sigma$ 不是 ad-hoc hack**：Proposition 1 说任何 $C^1$ consistency function 在 $\sigma \to 0$ 的最优 gradient scaling 是 $O(\sigma)$，由 $b'(0) \neq 0$ 达到。Linear $b = -\sigma$ 是这个 lower bound 的 canonical 实现。Paper 把一个看似只是"换个公式"的选择，提升到了 structural 的 principled 选择。

**可以推广的 principle**：任何 multi-modal diffusion 模型，如果各 modality 的 noise schedule 不对称（这其实是 multi-modal 设定的常态），distillation head 都应该按 modality 单独 design。这个 insight 应该可以 transfer 到 distribution-matching distillation（[DMD2](https://arxiv.org/abs/2405.14867)）和其他 multi-modal 扩散设定，paper Section C 也在 limitations 里暗示了这点。

**一个有意思的观察**：$f^a = \mathbf{x}_\sigma - \sigma v_\theta$ 在 low σ 接近 $\mathbf{x}_0 - \sigma v_\theta$。当 $\sigma = 0.01$ 时，consistency target 几乎是 $\mathbf{x}_0 - 0.01 v_\theta$，相当于直接监督 $v_\theta$ 学到正确方向。这其实在低噪声区域**等价于 flow matching**，只是被包装成 consistency loss。但 consistency loss 还多给了一个 property：trajectory invariance——trajectory 上任何点都映射到同一 endpoint，这就是 single-step inference 的理论保证。普通 flow matching 训出来的模型仍需 iterative Euler integration。

这个 paper 的优雅之处在于：诊断是 structural 的（Proposition 1 给出 lower bound 并证明 attainability），fix 是 minimal 的（只改 loss head，backbone 不动），验证是 end-to-end 的（23× 加速 + 真机 deployment）。是 robotics × diffusion distillation 交叉点上一个扎实的工程贡献，limitation 列表显示 generalization 还要更多工作，但核心 insight 已经站得住。

---

# Flash-WAM 深度技术解析

## 1. 核心动机：WAM 的 inference bottleneck

World-Action Models (WAMs) 把 robot policy 拆成两个 coupled diffusion stages：
- **Visual dynamics**：$p_\theta(\mathbf{x}^v \mid \mathbf{C})$，预测 K 个 future video latents
- **Inverse dynamics**：$p_\theta(\mathbf{x}^a \mid \mathbf{x}^v, \mathbf{C})$，从 predicted video 中 decode actions

每个 chunk 需要 $N^v + N^a$ 次 sequential transformer forward passes。LingBot-VA 默认 $25v/50a$，在 L40S 上 3550ms (video) + 4550ms (action) = 8.1s per chunk，完全无法 real-time。参考 RT-2、$\pi_0$ 这类 VLA，real-time budget 大概是 500ms（2Hz chunk rate），见 [Real-time execution of action chunking flow policies](https://arxiv.org/abs/2506.07339)。

工程级优化（KV cache、partial denoising、async pipeline）只能降 wall-clock，无法减少 NFE 本身。这条路是 orthogonal 的，Flash-WAM 走 step distillation 这条正路。

## 2. 为什么 off-the-shelf distillation 全部失败

这是 paper 最有洞察力的部分，要 build intuition。

### 2.1 不对称 SNR shift 的本质

WAM 给 video 和 action 配了不同 noise schedule：

$$\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0,1]$$

变量含义：
- $\tilde{\sigma}$：均匀采样的 base noise level
- $s$：shift 参数，$s \geq 1$
- $s$ 越大，$\sigma$ 分布越往 high noise 偏

LingBot-VA 设定 $s^v = 5.0$，$s^a = 1.0$。为什么？因为：
- **Video latents**：高维（128×128×3 latent，经过 VAE 仍有数千 dim）、空间冗余强、容忍 per-step 大 noise
- **Action sequences**：30-dim（G1 humanoid），precision-critical，差几个弧度就抓不到物体

直觉上：高维冗余数据 ≈ 高 SNR shift 容忍；低维 precision 数据 ≈ 必须温和 schedule。

### 2.2 Consistency function 的 gradient signal 分析

Consistency function 的 general form：

$$f(\mathbf{x}_\sigma, \sigma) = a(\sigma)\mathbf{x}_\sigma + b(\sigma)v_\theta(\mathbf{x}_\sigma, \sigma)$$

变量含义：
- $\mathbf{x}_\sigma$：noisy input at noise level $\sigma$
- $v_\theta$：网络预测的 velocity field
- $a(\sigma), b(\sigma)$：标量函数，满足 boundary condition $a(0) = 1, b(0) = 0$（保证 $\sigma=0$ 时 $f(\mathbf{x}_0, 0) = \mathbf{x}_0$）

**Key observation**: $f$ 仅通过 $v_\theta$ 依赖 $\theta$，所以 $\nabla_\theta \mathcal{L} \propto |b(\sigma)|$ pointwise。$b(\sigma)$ 决定了 network 在哪个 noise level 拿到有效学习信号。

### 2.3 LCM 的 quadratic vanishing 问题

LCM 选 $b_{\text{LCM}}(\sigma) = -\sigma^2 \sigma_d / \sqrt{\sigma^2 + \sigma_d^2}$，这里 $\sigma_d$ 是 data scale (默认 0.5)。

Taylor 展开在 $\sigma = 0$：
- $b_{\text{LCM}}(0) = 0$ ✓
- $b'_{\text{LCM}}(0) = 0$ ← **致命**
- $|b_{\text{LCM}}(\sigma)| = \sigma^2/\sigma_d + O(\sigma^4)$

也就是说，gradient signal 在 low σ 区域 **二次** vanishing。在 σ=0.1 时，$|b_{\text{LCM}}|$ 比 high-σ 区域小 **36×**。

### 2.4 Proposition 1：最优 gradient scaling

**定理**: 任何 $C^1$ consistency function 满足 boundary condition，$|b(\sigma)| = O(\sigma)$ as $\sigma \to 0$，且当且仅当 $b'(0) \neq 0$ 时 attain 这个 linear bound。

证明用 Taylor's theorem：$b(\sigma) = b'(0)\sigma + O(\sigma^2)$。leading term 消失 iff $b'(0) = 0$，此时退化到 $O(\sigma^2)$。

这是一个 **结构性** 而非 parametric 的 obstruction。对任何 $\sigma_d$ 都有 $\sigma^2\sigma_d/\sqrt{\sigma^2+\sigma_d^2} \leq \sigma$，所以 LCM family 永远达不到 linear bound。

### 2.5 为什么 naive joint LCM 必然崩溃

Action stream 的训练 mass 主要在 low-σ 区域（因为 $s^a = 1.0$，分布 uniform 在 [0,1]）。但 LCM 在 low-σ 给出 quadratic-decaying gradient，意味着 action head 在其训练分布主体上几乎拿不到学习信号。**网络无法在它实际见到最多的 noise level 上学**。

这就是 Table 1 里 Naive Joint LCM 在 1v/2a 跌到 23.97%，在 horizon 2/3 直接掉到 0–4% 的根本原因——long horizon 暴露了 action precision 问题。

## 3. Flash-WAM 的 modality-aware consistency functions

### 3.1 Action stream：linear-gradient-scaling

最简单满足 $a(0)=1, b(0)=0, b'(0)\neq 0$ 的选择：

$$a(\sigma) = 1, \quad b(\sigma) = -\sigma$$

代入得：

$$f^a(\mathbf{x}_\sigma^a, \sigma) = \mathbf{x}_\sigma^a - \sigma \cdot v_\theta(\mathbf{x}_\sigma^a, \sigma)$$

注意这其实就是 flow matching 里 recover clean estimate 的标准公式 $\hat{\mathbf{x}}_0 = \mathbf{x}_\sigma - \sigma v_\theta$。但 paper 把它**重新解释为 consistency-function family 中的 canonical low-σ realization**，由 framework 的 matching principle 选出来。这个 reframing 是 paper 的概念性贡献。

性质：
- $|b(\sigma)| = \sigma$ 全程线性，达到 Proposition 1 的最优 bound
- $a(\sigma) = 1$ constant，consistency target 不被 skip term 衰减
- 无 hyperparameter（不像 LCM 有 $\sigma_d$）

### 3.2 Video stream：Karras parametrization (variance-preserving)

Video 在 high-σ 训练，linear scaling 在这里反而有害：
- $\text{Var}[\mathbf{x}_\sigma - \sigma v_\theta]$ 随 $\sigma$ 增长，把 prediction error 放大 $\sigma$ 倍
- 输出 unbounded，early training 时容易 drift 出 data manifold

Flash-WAM 选 Karras parametrization：

$$f^v(\mathbf{x}_\sigma^v, \sigma) = c_{\text{skip}}(\sigma)\mathbf{x}_\sigma^v + c_{\text{out}}(\sigma)\hat{\mathbf{x}}_0^v$$

其中：
- $c_{\text{skip}}(\sigma) = \sigma_d^2/(\sigma^2 + \sigma_d^2)$：skip connection 权重
- $c_{\text{out}}(\sigma) = \sigma\sigma_d/\sqrt{\sigma^2 + \sigma_d^2}$：输出 scaling
- $\sigma_d = 0.5$：data scale

性质：
- $\text{Var}[f] \approx \sigma_d^2$ 一致（variance preservation）
- $\sigma \to \infty$ 时 $c_{\text{out}} \to \sigma_d$，输出有界
- 这正是 [Karras et al. 2022](https://arxiv.org/abs/2206.00364) 的设计

### 3.3 Joint training objective

$$\mathcal{L} = \mathcal{L}^v + \lambda_a \mathcal{L}^a$$

每个 modality 用自己的 consistency function：

$$\mathcal{L}^v = d\Big(f_{\theta_S}^v(\mathbf{x}_{\sigma_s}^v, \sigma_s), f_{\theta_{S'}}^v(\tilde{\mathbf{x}}_{\sigma_e}^v, \sigma_e)\Big)$$

$$\mathcal{L}^a = d\Big(f_{\theta_S}^a(\mathbf{x}_{\sigma_s}^a, \sigma_s), f_{\theta_{S'}}^a(\tilde{\mathbf{x}}_{\sigma_e}^a, \sigma_e)\Big)$$

变量：
- $\theta_S$：student（online）
- $\theta_{S'}$：EMA target student
- $\sigma_s$：起始 noise level
- $\sigma_e$：经过 k 步 Euler 推进后的 target noise level
- $\tilde{\mathbf{x}}_{\sigma_e} = \mathbf{x}_{\sigma_s} + \hat{v}_{\text{cfg}}(\sigma_e - \sigma_s)$：teacher Euler step 给的 target
- $d$：Huber loss ($c = 0.001$)

CFG 只用在 video teacher step（$w \sim \mathcal{U}[2.0, 10.0]$），action 用 unguided prediction——这反映了 video 需要 text guidance 而 action 是 deterministic inverse dynamics。

**架构层面零改动**：video 和 action tokens 仍 concatenate 进同一个 transformer，flex attention 处理 block-causal mask，两个 consistency function 只影响 per-stream loss head。

## 4. 实验数据深度解读

### 4.1 RoboTwin 2.0（50 tasks，bimanual）

| Method | $N^v$ | $N^a$ | Clean | Rand. | Avg | Speedup |
|---|---|---|---|---|---|---|
| LingBot-VA teacher | 25 | 50 | 91.64 | 90.86 | 91.25 | 1.0× |
| **Flash-WAM** | **1** | **2** | 88.42 | 82.66 | **85.54** | **19×** |
| **Flash-WAM** | **1** | **1** | 82.56 | 80.26 | **81.41** | **23.3×** |
| Naive Joint LCM | 1 | 2 | 25.88 | 22.07 | 23.97 | — |
| Naive Joint LCM | 1 | 1 | 39.68 | 32.96 | 36.32 | — |
| Video-only LCM | 1 | 2 | 80.66 | 76.92 | 78.79 | 19× |
| Video-only LCM | 1 | 1 | 77.90 | 69.46 | 73.68 | 23.3× |
| DMD2 | 1 | 2 | 85.08 | 72.36 | 78.74 | — |
| DMD2 | 1 | 1 | 52.66 | 48.46 | 50.56 | — |
| Motus | — | — | 88.66 | 87.02 | 87.8 | — |
| $\pi_{0.5}$ | — | — | 82.74 | 76.76 | 79.8 | — |

关键观察：
1. **Naive Joint LCM 在 1v/1a 居然比 1v/2a 还高**（36.32 vs 23.97）。这是因为 1v/2a 时 action stream 的 gradient signal 严重不足导致训崩，而 1v/1a 在某种意义上让训练更早 stop 住没有崩得那么彻底——这其实是 model collapse 的不同阶段。
2. **Flash-WAM 1v/2a vs 1v/1a 仅差 4 个点**（85.54 → 81.41），说明 action 单步已经够用，因为 linear scaling 让一步就能学好。
3. **Video-only LCM 在 1v/2a 时保留了 50 step teacher 的 action**，所以还有 78.79%，但 1v/1a 跌到 73.68%，因为 action 没蒸馏直接从 50 步砍到 1 步。

### 4.2 Horizon breakdown（Table 4 / 7）

在 RoboTwin 上按 horizon=1/2/3（sequential steps）拆分。1v/2a 配置：
- Horizon 1：Flash-WAM 92.30/88.47 vs Video-only LCM 87.10/82.73
- Horizon 2：Flash-WAM 84.88/76.63 vs Video-only LCM 73.13/68.19
- Horizon 3：Flash-WAM 73.50/63.25 vs Video-only LCM 62.50/68.25

Naive Joint LCM 在 horizon 2/3 直接 0–4%，**action 误差累积是 horizon-cubed 增长**。

### 4.3 LIBERO（4 suites）

| Method | $N^v$ | $N^a$ | Spatial | Object | Goal | Long | Avg | Speedup |
|---|---|---|---|---|---|---|---|---|
| LingBot-VA | 20 | 50 | 98.5 | 99.8 | 98.0 | 98.3 | 98.6 | 1.0× |
| **Flash-WAM** | 1 | 2 | 97.0 | 92.8 | 96.4 | 98.0 | **95.7** | 13.7× |
| **Flash-WAM** | 1 | 1 | 96.0 | 92.6 | 96.0 | 95.8 | **95.1** | 16.3× |
| Video-only LCM | 1 | 2 | 95.1 | 92.0 | 96.0 | 97.8 | 95.2 | 13.7× |

LIBERO 任务比 RoboTwin 简单（500 demos/suite vs RoboTwin 的 bimanual long-horizon），所以即便 Video-only LCM 在 1v/2a 也有 95.2%——但 1v/1a 时它跌到 94.2，而 Flash-WAM 仍保持 95.1。

### 4.4 真机 Unitree G1 humanoid

三个任务：T1（pot+potato， hardest）、T2（red bottle with yellow distractor）、T3（pink object to target）。

| Method | $N^v/N^a$ | T1 | T2 | T3 | Avg |
|---|---|---|---|---|---|
| LingBot-VA teacher | 3/10 | 50% | 70% | 80% | 66.7% |
| LingBot-VA reduced NFE | 1/2 | 30% | 50% | 40% | 40.0% |
| Video-only LCM | 1/2 | 30% | 50% | 50% | 43.3% |
| **Flash-WAM** | **1/2** | **50%** | **60%** | **70%** | **60.0%** |
| LingBot-VA reduced NFE | 1/1 | 10% | 30% | 30% | 23.3% |
| Video-only LCM | 1/1 | 20% | 40% | 40% | 33.3% |
| **Flash-WAM** | **1/1** | **40%** | **50%** | **60%** | **50.0%** |

注意 T1 在 1v/1a 时 reduced NFE 只有 10%，Flash-WAM 拉到 40%——这个任务最依赖 action precision（开盖 + 放入），所以 linear-gradient-scaling 在 low-σ 的优势最显著。

## 5. DMD2 baseline 适配的工程细节

Paper Appendix 给了把 DMD2 改造到 joint video-action 的完整方案，这个细节很有价值。

### 5.1 三网络架构
- $\theta_T$ (frozen reference)：原 LingBot-VA
- $\theta_S$ (student)：K-step generator，$K=4$
- $\theta_C$ (critic)：track student 分布

### 5.2 Student rollout（K=4 steps）
第 $i$ 步：student 算 $v_{\theta_S}(\mathbf{x}_{\sigma_i}^v, \mathbf{x}_{\sigma_i}^a, \tilde{\mathbf{x}}_0^v, \tilde{\mathbf{x}}_0^a, \sigma_i, y)$，恢复 $\hat{\mathbf{x}}_0^{v,(i)} = \mathbf{x}_{\sigma_i}^v - \sigma_i v_{\theta_S}^v$，然后 re-noise：

$$\mathbf{x}_{\sigma_{i+1}}^v = (1-\sigma_{i+1})\hat{\mathbf{x}}_0^{v,(i)} + \sigma_{i+1}\epsilon^{v,(i+1)}$$

memory bound：只有最后一步 $i = K-1$ 保留 autograd，前几步 no_grad。

### 5.3 Single-pass joint scoring
Student 输出 re-noise 到独立采样的 $\sigma^v, \sigma^a \sim \mathcal{U}(0.02, 0.98)$，喂给 critic 和 reference：

$$\hat{\mathbf{x}}_0^{v,\text{fake}} = \tilde{\mathbf{x}}^v - \sigma^v v_{\theta_C}^v(\tilde{\mathbf{x}}^v, \tilde{\mathbf{x}}^a, \sigma^v, \sigma^a, y)$$

$$\hat{\mathbf{x}}_0^{v,\text{real}} = \tilde{\mathbf{x}}^v - \sigma^v v_{\theta_T}^{v,\text{cfg}}(\cdot)$$

CFG 只用在 video real score，$w^v = 3.0$。

### 5.4 Distribution-matching gradient
$$g^v = \frac{\hat{\mathbf{x}}_0^{v,\text{fake}} - \hat{\mathbf{x}}_0^{v,\text{real}}}{\|\hat{\mathbf{x}}_0^v - \hat{\mathbf{x}}_0^{v,\text{real}}\|_1 + \varepsilon}$$

per-sample $L_1$ normalization（DMD2 的 gradient-norm fix），$\varepsilon = 10^{-8}$。

### 5.5 两种 DMD2 variant
- **Joint DMD2**：action 也从 noise denoise，加 $\mathcal{L}_{\text{DM}}^a$，无 regularizer
- **Video-only DMD2 + reg**：action 输入直接从 ground truth perturb，加 MSE regularizer

Joint DMD2 在 1v/1a 只有 52.66%，比 Video-only DMD2+reg 的 66.53% 低 14 点——同样验证了 Section 4.1 的诊断，distribution-matching 在 asymmetric schedule 下也崩溃。

## 6. 与相关工作对比的位置感

- **$\pi_0$ / $\pi_{0.5}$** ([Black et al. 2024](https://arxiv.org/abs/2410.24164), [2025](https://arxiv.org/abs/2504.16054))：VLA flow matching，单 modality action diffusion，没有 video generation 这条 path
- **Motus** ([Bi et al. 2025](https://arxiv.org/abs/2512.13030))：Mixture-of-Transformers，VLM + video gen + action gen 通过 cross-attention coupled。RoboTwin 87.8% 但比 Flash-WAM 的 85.54% 略高，不过 Motus 没公开 inference latency 是否 real-time
- **DreamZero** ([Ye et al. 2026](https://arxiv.org/abs/2602.15922))：在 architecture level 集成 inference optimization，与 Flash-WAM 的 distillation 思路正交
- **Fast-WAM** ([Yuan et al. 2026](https://arxiv.org/abs/2603.16666))：彻底 skip test-time video gen，用 video DiT 当 single-pass encoder。这条路放弃了 video generation 的 spatiotemporal prior
- **GigaWorld-Policy** ([Ye et al. 2026](https://arxiv.org/abs/2603.17240))：把 future visual dynamics 当 reasoning signal（causal mask），不是显式 prediction

Flash-WAM 的独特位置：**保留 WAM 推理结构，把每个 modality 的 denoising 压到 1 步**。

## 7. Hyperparameter 关键设定

Table 6 重要项：
- Image resolution: 128×128
- Action dimension: 30 (G1 humanoid)
- Actions per video frame: 4（一个 video frame 对应 4 个 action steps）
- Frame chunk size K: 4
- $s^v = 5.0$，$s^a = 1.0$
- $\lambda_a = 1.0$（action loss 权重）
- $\lambda_r = 0.2$（action regularizer 权重，用于 ablation）
- EMA decay $\alpha = 0.995$
- $\sigma_d = 0.5$
- CFG range $[2.0, 10.0]$
- Optimizer: AdamW ($\beta_1, \beta_2) = (0.9, 0.999)$
- LR: $5 \times 10^{-6}$
- Effective batch size: 48 (4× H100)
- Training: 2000 steps per LIBERO suite，约 24 小时

## 8. 我的直觉与延伸思考

### 8.1 Modality-aware 这个 principle 可推广
Paper Section C 提到这个 principle 可能 transfer 到 distribution-matching distillation 和其他 multi-modal diffusion。我认为这指向一个更普遍的设计：**任何 shared-backbone 多模态生成模型，如果各模态 noise schedule 不对称，它们的 distillation head 应该独立 design**。

类比：MoE 中的 expert specialization。Flash-WAM 在 loss head 层面做了 specialization，而 backbone 仍共享。这跟 [Motus](https://arxiv.org/abs/2512.13030) 的 Mixture-of-Transformers 是不同维度的 specialization。

### 8.2 线性 consistency function 的数值稳定性
$f^a = \mathbf{x}_\sigma - \sigma v_\theta$ 在 low σ 接近 $\mathbf{x}_0$。当 σ=0.01 时，consistency target 几乎就是 $\mathbf{x}_0 - 0.01 v_\theta$，这相当于直接监督 $v_\theta$ 学到正确方向。这其实**等价于在 low σ 区域做 flow matching**，只是被包装成 consistency loss 的形式。所以 action stream 在 Flash-WAM 下本质上是用 consistency loss 形式做 conditional flow matching distillation。

### 8.3 为什么不直接用 action-only flow matching distillation
如果 action 本质上等价于 low-σ flow matching，为什么不直接训？因为 consistency loss 提供 trajectory invariance——任何 σ 上的点都映射到同一 endpoint。这给 single-step inference 提供了理论保证（在 trajectory 上任意一点都能一步 jump 到 $\sigma=0$）。普通 flow matching 训出来的模型仍需要 iterative Euler integration。

### 8.4 Karras parametrization 的另一个优势
Paper 没强调但我觉得重要的：Karras 的 $c_{\text{out}}$ 在 high σ 时让网络输出 scale 不爆炸，这对 **teacher guided Euler step** 的数值稳定性是关键。Flash-WAM 在 distillation 时 teacher 给 target $\tilde{\mathbf{x}}_{\sigma_e} = \mathbf{x}_{\sigma_s} + \hat{v}_{\text{cfg}}(\sigma_e - \sigma_s)$，如果 $\hat{v}$ unbounded，target 就会漂移，student 跟不上。Karras 的 bounded output 隐式给 teacher 提供了 stability。

### 8.5 一个值得做的 ablation
Paper 没做的：把 $s^a$ 调到 5.0（和 video 一样），看 Naive Joint LCM 是否崩溃消失。直觉上会——因为这样 action mass 也集中 high σ，LCM 的 quadratic vanishing 就不痛了。但 WAM 性能本身会降，因为 action 在 high σ 学不到精度。这个实验会闭环验证 paper 的诊断。

### 8.6 跟 shortcut models ([Frans et al. 2025](https://arxiv.org/abs/2410.12557)) 的关系
Shortcut models 也做一步 inference，但用 self-conditioning 预测 step-conditioned velocity。Flash-WAM 没用这个机制，因为 consistency loss 已经 implicit 学到了 trajectory invariant mapping。两个方法可以 combine——用 shortcut 的思想给 action stream 一个 fallback path。

### 8.7 Limitations 诚实度
Paper Section C 承认：
- 实验主要在 simulation，真机只有 3 个 task × 10 rollouts（样本小）
- 只针对 shared-backbone WAM，没测 multi-model 架构
- high-σ regime 的 corresponding optimal scaling 分析还没做
- 没扩展到 distribution-matching distillation

这个 limitation 列表其实暴露了 paper 的 generalization claim 还需更多验证。特别是真机 60% vs teacher 66.7%，gap 是 6.7 个点——这个 gap 在 long-horizon 任务上可能放大。

## 9. 相关参考链接

- Paper project page: flashwam.github.io
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [Consistency Models (Song et al.)](https://arxiv.org/abs/2303.01469)
- [Latent Consistency Models (Luo et al.)](https://arxiv.org/abs/2310.04378)
- [Karras et al. EDM](https://arxiv.org/abs/2206.00364)
- [DMD2 (Yin et al.)](https://arxiv.org/abs/2405.14867)
- [LingBot-VA](https://arxiv.org/abs/2601.21998)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [$\pi_0$](https://arxiv.org/abs/2410.24164)
- [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054)
- [Motus](https://arxiv.org/abs/2512.13030)
- [DreamZero](https://arxiv.org/abs/2602.15922)
- [Fast-WAM](https://arxiv.org/abs/2603.16666)
- [GigaWorld-Policy](https://arxiv.org/abs/2603.17240)
- [Shortcut Models](https://arxiv.org/abs/2410.12557)
- [Real-time action chunking](https://arxiv.org/abs/2506.07339)
- [VideoLCM](https://arxiv.org/abs/2312.09109)
- [Progressive Distillation](https://arxiv.org/abs/2202.00512)
- [ECM (Consistency Models Made Easy)](https://arxiv.org/abs/2406.14548)
- [AnimateLCM](https://arxiv.org/abs/2402.00769)
- [T2V-Turbo-V2](https://arxiv.org/abs/2410.05677)
- [Phased DMD](https://arxiv.org/abs/2510.27684)
- [One-step f-divergence](https://arxiv.org/abs/2502.15681)
- [GR00T N1 (NVIDIA)](https://arxiv.org/abs/2503.14734)
- [X-VLA](https://arxiv.org/abs/2510.10274)
- [VLA robustness study](https://arxiv.org/abs/2603.22078)
- [World models survey](https://arxiv.org/abs/2605.00080)
- [World models for embodied AI](https://arxiv.org/abs/2510.16732)
- [Video generators are robot policies](https://arxiv.org/abs/2508.00795)
- [VOTE](https://arxiv.org/abs/2507.05116)
- [TIDAL](https://arxiv.org/abs/2601.14945)

## 10. 总结直觉

Flash-WAM 的核心 insight 可以浓缩成一句话：**consistency function 在 low-σ 的 gradient scaling 是由 $b'(0)$ 决定的，LCM 在这点上是 0 所以 action stream（low-σ dominated）学不动；用 linear $b = -\sigma$ 修这个，video stream 不变（因为 high-σ 不受这个 obstruction 影响）**。

这个 paper 的优雅之处在于：诊断是 structural 的（Proposition 1 给出最低 bound 并证明 attainability），解决方案是 minimal 的（只改 loss head，backbone 不动），实验验证是 end-to-end 的（23× speedup + 真机 G1 deployment）。这是 robotics + diffusion distillation 交叉点上一个扎实的 engineering contribution，虽然 limitation 列表显示 generalization 还需要更多工作，但核心 analytical insight 已经站得住脚。
