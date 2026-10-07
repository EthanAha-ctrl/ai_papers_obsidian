---
source_pdf: Self-Refining Video Sampling.pdf
paper_sha256: dc5f39de93e33763fe9b5469c59a1e52fd3347690b167aa489f19a3b5cd98841
processed_at: '2026-08-12T04:54:06-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

好，我换个口吻，就当咱俩在白板前聊天。

---

## 这篇 paper 到底干了啥

一句话：**让 video generator 在 inference 的时候自己改自己，不加任何外部 model，不 retrain，只改 sampling 的 inner loop。**

就这么简单。剩下的全是"怎么改"和"为什么这么改 work"。

---

## 出发点：video generator 现在啥问题

你看 Wan2.2、Cosmos 这些 SOTA video generator，生成的视频漂亮是漂亮，但一旦遇到复杂 motion 就翻车。比如：

- 体操运动员在 pommel horse 上转圈 → 手臂被画成两份（Figure 4 第一行红框）
- 机器人去抓碗 → 碗没被抓起来，prompt 说要放到蓝布上，结果根本没放（Figure 5 第一行）
- 小孩扒开沙子 → 沙子凭空出现在手里，没有因果（Figure 6 第一行）

这些都是 **physical realism 问题**——motion 不连贯、physics 不对、object interaction 假。

之前的 fix 方法两类：

1. **外部 verifier + rejection sampling**：生成 N 个视频，让另一个 model 挑最好的。比如 Cosmos-Reason1 7B 当 critic (https://arxiv.org/abs/2503.15558)。问题是 acceptance rate 低，要生成 4 个才挑 1 个，4× 推理时间，还依赖一个 domain-specific verifier。

2. **Post-training / RLHF**：拿 synthetic data 或 reward model 去 fine-tune。比如 Video-T1 (https://arxiv.org/abs/2502.05286)、PhyGDPO (https://arxiv.org/abs/2512.24551)。问题是要高质量数据 + 大量 compute，reward model 抓 fine-grained motion 也很难。

这篇 paper 的态度是：**先别加新东西，先看看现有 model 自己能干啥**。大型 video generator 已经在海量数据上学到了 motion 和 physics 的 prior，这个 prior 在 inference 时只用了一次就被扔了，太浪费。

---

## 核心 reframe：flow matching 其实就是个 DAE

这个是全文最妙的 insight。

### 普通 flow matching 训练怎么训

RGB video $x$ 经 VAE 压成 latent $z$，然后学一个 vector field $u_\theta(z_t, t)$，让它逼近 $z_1 - z_0$（clean latent minus noise）。

训练 loss：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, z_0, z_1}\left[ || u_\theta(z_t, t) - (z_1 - z_0) ||_2^2 \right]$$

- $t \in [0,1]$：noise level（$t=0$ 全噪声，$t=1$ 全数据）
- $z_0 \sim \mathcal{N}(0, \mathbf{I})$：噪声
- $z_1 \sim p_1$：clean latent
- $z_t = (1-t)z_0 + tz_1$：插值路径

### 关键 algebraic trick

定义一个新量 $\hat{z}_1^\theta := z_t + (1-t) u_\theta(z_t, t)$。

直觉：因为 $u_\theta$ 学的是 $z_1 - z_0$，而 $z_t = (1-t)z_0 + tz_1$，那 $z_t + (1-t)(z_1 - z_0) = z_1$。所以 $\hat{z}_1^\theta$ 就是模型对 clean data 的直接预测。

把这个代进 loss，化简：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, z_0, z_1}\left[ \frac{1}{(1-t)^2} || \hat{z}_1^\theta - z_1 ||_2^2 \right]$$

注意这个 $\frac{1}{(1-t)^2}$ 权重，$t \to 1$ 时爆炸——意思是 model 越靠近 clean 端，对 $\hat{z}_1$ 的预测必须越准。

而 Bengio 2013 那篇 generalized DAE (https://arxiv.org/abs/1305.6663) 的 loss 是：

$$\mathcal{L}_{DAE}(\theta) = \mathbb{E}\left[ || \hat{z}_1^\theta - z_1 ||_2^2 \right]$$

——**就差一个 weighting。** 所以 flow matching 训练时实际上在每个 noise level $t$ 上都训了一个 DAE，只不过 inference 时你只用它做一次 forward 就走了。

Bengio 2013 那篇里讲过：DAE 可以用 pseudo-Gibbs sampling 反复 corrupt→reconstruct，构成一个 markov chain，不变分布偏向 data manifold，迭代会把样本拉向 high-density region。

**所以 inference 时本来就可以反复来。** 只是大家没这么用。

---

## P&P 算法：两个算子反复套

定义两个算子：

**Predict**（denoise / reconstruct）：
$$D_\theta(z_t, t) := z_t + (1-t) u_\theta(z_t, t)$$
- 输入：当前 noisy latent $z_t$
- 输出：对 clean $z_1$ 的预测 $\hat{z}_1$

**Perturb**（corrupt / local resample）：
$$R_\epsilon(z, t) := tz + (1-t)\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
- 输入：clean 预测 $\hat{z}_1$
- 输出：在 noise level $t$ 上重新加噪
- 几何上就是在 $\hat{z}_1$ 和 fresh noise $\epsilon$ 之间做插值，noise 比例 $1-t$

一次 P&P 迭代：
$$z_t^{(k+1)} = R_{\epsilon_k}(D_\theta(z_t^{(k)}, t))$$

先 predict 出 $\hat{z}_1$，再用同样的 noise level $t$ 加回去。这就是 pseudo-Gibbs 的一步。

每步采新的 $\epsilon_k$，让它探索。

反复 $K_f$ 次（默认 2-3 次就够），把 $z_t$ 拉向 high-density region。然后把这个 refined $z_t^*$ 交给 ODE solver 往下一步推：

$$z_{t_{i+1}} = z_{t_i}^* + \Delta t \cdot u_\theta(z_{t_i}^*, t)$$

就 plug-and-play 嵌进去。

---

## 关键经验观察：只在早期 timestep 应用 P&P

Video diffusion 有个大家都观察到的现象（VideoJAM https://arxiv.org/abs/2502.05446、FlowMo https://arxiv.org/abs/2506.01144、Frame Guidance https://arxiv.org/abs/2506.07177）：**motion 和 physics 在前 20% timestep 就基本定死了。** 后期 timestep 只是在 refine texture、color 这些细节。

所以 P&P 只在 $t < 0.2$ 应用，后期不动。这就把额外 NFE 控制在 1.5× 左右。

---

## 实验数据看几个亮点

### Dynamic-bench (Table 1)

自己用 Gemini 3 生 120 个 prompt（40 multi-object interaction、40 complex human motion、40 physics-driven dynamic），让 Wan2.2-A14B T2V 生成，VBench + 20 人 human eval。

| Method | Motion win rate vs ours | NFE | Time |
|---|---|---|---|
| Default UniPC | 73.57% 偏好 ours | 40 | 1.0× |
| +NFE×2 | 74.05% 偏好 ours | 80 | 2.0× |
| +FlowMo | 70.57% 偏好 ours | 40 | 3.9× |
| +Ours | — | 60 | 1.5× |

关键看点：**NFE 翻倍几乎没用**（74% vs 73%），说明这不是"步数不够"的问题，是"sampling 路径走错了"。P&P 用 1.5× 时间干翻了 3.9× 的 FlowMo。

### PAI-Bench 机器人 I2V (Table 2)

Cosmos-Predict-2.5-2B 上：

| Method | Grasp ↑ | Robot-QA ↑ | NFE | Time |
|---|---|---|---|---|
| Default | 79.2 | 71.7 | 35 | 1.0× |
| +Verifier best-of-4 (Cosmos-Reason1 7B) | 84.4 | 72.3 | 140 | 4.0× |
| +Ours | **89.6** | **76.3** | 57 | 1.6× |

Grasp +11 个点，比 best-of-4 还高 5 个点，cost 是它 40%。这个对比很有杀伤力——你用 7B 的专门 verifier 反复 sample 4 次，还不如让 generator 自己 refine 自己。

Wan2.2-I2V 上也是：default 77.3 → ours 85.7，best-of-4 才 80.5。

### PhyWorldBench (Table 3)

物理常识评估，PC = physical commonsense：

| Method | VideoPhy2 PC↑ | PhyWorldBench PC↑ |
|---|---|---|
| Default | 54.5 | 28.6 |
| +NFE×2 | 53.1 | 29.3 |
| +CFG-Zero | 50.6 | 29.3 |
| +Ours | **55.6** | **40.0** |

PhyWorldBench 上 28.6 → 40.0，绝对 +11.4，相对 +40%。这是个大跳。

### Spatial consistency (Table 5)

让 camera 转 >360° 回到原视角，用 MegaSAM (https://arxiv.org/abs/2438.09012) 估 pose，对比 revisit frame：

| Method | SSIM ↑ | PSNR ↑ |
|---|---|---|
| Default | 0.401 | 14.96 dB |
| +Ours | 0.485 | 17.21 dB |

PSNR +2.25 dB 是很大提升。说明 P&P 不仅 fix motion，还让 early-stage latent 更稳，spatial layout 也跟着稳了。

### Visual reasoning 的反例 (Figure 18, 19)

Wiedemer et al. 2025 (https://arxiv.org/abs/2509.20328) 的两个任务：

- **Graph traversal**（水沿连通节点流）：base 0.1 成功率 → +P&P 0.8。**大提升**
- **Maze solving**（红方块沿白路径走到绿方块）：base 接近 0 → +P&P 还是接近 0。**没救**

为什么？graph traversal 的错误是"传播过程的中途抖动"，local refinement 能修；maze solving 的错误是"路径选择错"，需要全局 planning，local search 救不了。

这跟 LLM 的 self-refine 限制很像——self-refine 能修"过程小错"，修不了"知识盲区"。

---

## Uncertainty-aware P&P：为什么需要，怎么做的

### 问题：多次 P&P 会 over-saturate

P&P 每步都做 CFG update (https://arxiv.org/abs/2207.12598)。CFG scale 在 predict 算子里是 $(1-t)$，在 $t$ 小时接近 1。正常 ODE step 用的系数是 $\Delta t$（很小）。

在 static region（背景），多次 P&P 之后 $z_t^{(k)}$ 几乎不变，但每次 predict 都用大 CFG scale 重推一遍，guidance 累积，导致背景过饱和、色调漂移、水面夸张反光（Figure 9b）。

### 解决：用 self-consistency 当 uncertainty

不需要外部 model，不要 N 次 stochastic forward pass（不像 BayesDiff https://arxiv.org/abs/2306.05453 那种），直接看 P&P 内部：

$$\mathbf{U}(z_t^{(k-1)}, z_t^{(k)}) := \frac{1}{C} || D_\theta(z_t^{(k-1)}, t) - D_\theta(z_t^{(k)}, t) ||_1$$

就是两次连续 P&P 的 clean 预测 $\hat{z}_1$ 之差，在 channel 维平均，spatio-temporal 维算 L1。

阈值化：
$$M_{t_i}^{(k)} := \mathbb{1}(\mathbf{U} > \tau), \quad \tau = 0.25$$

$M=1$ 的地方 refine（moving object），$M=0$ 的地方保留前一轮（background）。

Figure 3 可视化显示 mask 正好落在运动物体上——model 自己的预测一致性已经天然"看见"了哪里需要改。

### 更新规则：零额外 NFE

$$z_{t_{i+1}}^{(k)} \leftarrow M \odot z_{t_{i+1}}^{(k)} + (1-M) \odot z_{t_{i+1}}^{(k-1)}$$

注意 $z_{t_{i+1}}^{(k-1)}$ 是上一轮已经算好的 next-step latent，不重算。Mask 计算复用已经 forward 过的 $D_\theta$，所以 0 额外 NFE。

Algorithm 2 line 11 还有：mask 累加，某个区域一旦被标 uncertain 就一直是 uncertain，避免 flicker。

---

## Video 跟 Image 的本质区别：cross-frame consistency

这是 paper Discussion section 最有意思的部分（Sec 6.1）。

Figure 13a 用 SDEdit (https://arxiv.org/abs/2108.11402) 改 prompt（orange cat → brown dog）：
- Image：明显 semantic 变化
- Video：内容几乎不变

Figure 13b 多次 P&P：
- Image：单次 P&P 就大幅偏离
- Video：多次 P&P 几乎不变

**为什么 video 这么 robust**：相邻帧共享 layout 和 motion trajectory，单帧扰动被邻帧拉回。

→ 这给 P&P 在 video 里能多次迭代不崩的理论支持：image 里 K=8 就 mode-seeking collapse 成"白山羊"（Figure 22），video 里 K=8 只是 reduce temporal jitter。

Paper 把这叫 "intended temporal mode-seeking"——mode-seeking 在 video domain 是好事，因为 temporally inconsistent video 本来就在 low-density region，推向 high-density mode 就是去 flickering。

---

## 跟其他 inference-time 方法的对比

| Method | 怎么用 stochasticity | 在哪 refine | 目标 |
|---|---|---|---|
| Annealed Langevin Dynamics (https://arxiv.org/abs/1907.05600) | 多个 annealed noise scale | 整个 schedule | 严格 MCMC |
| Restart (https://arxiv.org/abs/2306.14878) | macro forward-backward | 后期 | 减累积误差 |
| FreeInit (https://arxiv.org/abs/2410.18054) | 改初始 $z_{t_0}$ 重跑全 denoise | 起点 | temporal consistency |
| **P&P** | 同 noise level local resample | 早期 timestep 内部 | mode-seeking |

P&P 跟 ALD 区别：ALD 跨多 noise level，P&P 在固定 $t$ 内反复。P&P 跟 Restart 区别：Restart 往前跳再 ODE 回，P&P 在固定 $t$ 内 fine-grained。P&P 跟 FreeInit 区别：FreeInit 改 $z_{t_0}$ 整个重跑，P&P 改 intermediate $z_t$ 在同一 trajectory 内 refine，便宜得多。

---

## 一个我特别想强调的直觉

你看 Eq. (3) 那个 $\frac{1}{(1-t)^2}$ 权重，$t \to 1$ 时爆炸。意思是 model 越靠近 clean 端，对 $\hat{z}_1$ 的预测越准。但 inference 时 ODE solver 在每个 $t$ 只 forward 一次——你花了大代价训练出的"在所有 noise level 上都很准的 DAE"，只用了一次就扔了。

P&P 本质就是把这个"被压扁的 markov chain"重新展开成"完整的 pseudo-Gibbs"，让你"白嫖"训练时已经 encode 进去的 capacity。

这跟 LLM 里的 test-time scaling（Karan & Du 2025 https://arxiv.org/abs/2510.14901 用 base LLM 做 MCMC 不用 RL）是同一个哲学：**base model 已经很聪明了，你只是没给它时间想。**

---

## 可能的延伸联想

- **跟 consistency model 的关系**：consistency model (https://arxiv.org/abs/2303.01469) 训练目标就是"任意 noise level 直接预测 endpoint"，跟 $\hat{z}_1^\theta := z_t + (1-t)u_\theta$ 一回事。P&P 可以看作"用 consistency-style predictor 做 inference-time refinement"。能不能用 consistency distill 出来的小 model 做 P&P 的 predict 算子？这样 NFE 还能再降。

- **跟 AlphaGo 的结构对应**：P&P 是 local rollout，predict 是 value evaluation，perturb 是 exploration。能不能在 video generation 上做真正的 tree search？uncertainty mask 当 prior，predict 的 $\hat{z}_1$ 当 value，多次 P&P 当 MCTS 的多分支。这是 visual analog of "let's verify step by step" (https://arxiv.org/abs/2306.15812)。

- **Reward gradient + P&P 混合**：现在 P&P 是 self-supervised，只用 model 自己的 consistency。能不能加 external reward gradient（Liu et al. 2025b https://arxiv.org/abs/2501.13918 的 human feedback reward）做 guided P&P？perturb 之后用 reward gradient 微调一下 $\hat{z}_1$，再 predict。这就把 self-refine 和 RLHF inference-time 版结合了。

- **World model / simulator 角度**：Sora (https://openai.com/research/video-generation-models-as-world-simulators) 把 video generator 当 world model，但物理一致性不够。P&P 让 inference 时自我仿真校正，不需要 retrain。这可能是让 video diffusion 真的能用做 robot simulator 的关键 trick——robot 不需要每次都 retrain model，只要在 planning 时多跑几次 P&P refine 出 physically plausible 的 trajectory。

- **Audio / 3D 上的类比**：cross-frame consistency 是 video 独有的 inductive bias。audio 有 cross-time frequency consistency，3D 有 multi-view consistency。这些 medium-specific structure 都可能支撑类似的 inference-time self-refinement algorithm。比如 3D generation 里 Novel View Synthesis 的"相邻视角 consistency"能不能让 multi-step refine 在 3D 上也稳定？

- **Time-dependent $\tau_t$**：paper Sec B.4 提到早期 uncertainty 大、后期小，但用固定 $\tau = 0.25$。如果让 $\tau$ 跟 $t$ 走，早期放宽后期收紧，可能更 robust。

- **Maze solving 的可能解法**：paper 承认 local search 救不了 maze。但能不能混合 P&P + VLM verifier——P&P 在 local refine motion，VLM 在每几步 check 一次"path 是否有效"，类似 AlphaGo 的 policy + value network。或者更激进，用 VLM 给的 reward signal 引导 perturb 的方向，把 random $\epsilon$ 换成 reward-weighted perturbation。

- **Diffusion model 上的推广**：Figure 23 已经验证 CogVideoX (https://arxiv.org/abs/2408.06026) 这种 DDPM-based 的也 work。说明 DAE 视角对 DDPM/Flow Matching 通用。FLUX.1-dev (https://github.com/black-forest-labs/flux) 上 image generation 也 work（Figure 15），2 NFE 就改善 text rendering。

---

## 最后说回 intuition

这篇 paper 最 elegant 的地方：**它没加任何新东西**。没新 model、没新 training、没新 verifier、没新 reward。只是重新解读 flow matching 的 loss function，发现它其实在每个 noise level 上都训了个 DAE，然后 inference 时把这个"被压扁的 markov chain"展开。

这跟 LLM test-time scaling 的哲学一致：base model 已经有 capacity，你只是没让它"多想几步"。LLM 那边是 sample multiple reasoning chains，video 这边是 same noise level 内 multi-step P&P。

Cross-frame consistency 是 video 独有的 free lunch——image 上 multi-step P&P 会 mode-seeking collapse，video 上反而稳定 accumulate refinement。这种"利用 medium-specific structure 做 inference-time algorithm"的思路，可能比单纯 scale up model 更接近下一个突破点。

如果你 Karpathy 在思考 micro-world model / simulator 这条路，P&P 给了一个 useful building block：让 pre-trained generator 在 inference 时自我仿真校正。这跟你一直强调的"let the model use itself"的哲学其实是一脉相承的。

---

# Self-Refining Video Sampling: 深度技术解读

Andrej, 这篇 paper 我读完后最大的 takeaway 是：**它把 flow matching 重新解读成 time-conditioned DAE，然后用 pseudo-Gibbs 在同一 noise level 内做 self-refinement**。这个 reframe 非常优雅,而且跟 diffusion model 的 score matching → Langevin dynamics 的 connection 一脉相承。下面我从底层 math 开始,逐层 build intuition。

---

## 1. 核心 insight: Flow Matching = weighted DAE

### 1.1 Flow matching 的回顾

在 video diffusion 里，RGB video $x \in \mathbb{R}^{T \times H \times W \times 3}$ 先经 video VAE 压成 latent $\bar{z} \in \mathbb{R}^{f \times h \times w \times c}$（其中 $f, h, w$ 是时空下采样后的分辨率，$c$ 是 latent channel）。

Flow matching 学一个 time-dependent vector field $u_\theta: \mathcal{Z} \times [0,1] \to \mathcal{Z}$，把 prior $p_0 = \mathcal{N}(0, \mathbf{I})$ 拉到 data distribution $p_1$。直线插值路径 $z_t = (1-t)z_0 + t z_1$，目标 vector field $v_t = z_1 - z_0$，训练 loss：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, z_0, z_1}\left[ || u_\theta(z_t, t) - (z_1 - z_0) ||_2^2 \right] \tag{2}$$

- $t \in [0,1]$：noise level / flow time（$t=0$ 纯噪声，$t=1$ 纯数据）
- $z_0 \sim p_0$：初始噪声样本
- $z_1 \sim p_1$：clean data latent
- $u_\theta$：神经网络学的 vector field

### 1.2 改写：从 vector field 到 endpoint predictor

关键 trick：把 $u_\theta$ 改写成对 clean data 的预测 $\hat{z}_1^\theta := z_t + (1-t)u_\theta(z_t, t)$。

直觉：因为 $u_\theta$ 逼近 $z_1 - z_0$，而 $z_t = (1-t)z_0 + t z_1$，所以

$$z_t + (1-t)(z_1 - z_0) = (1-t)z_0 + tz_1 + (1-t)z_1 - (1-t)z_0 = z_1$$

代入 loss (2)，把 $z_1 - z_0 = (z_t + (1-t)u_\theta - z_t)/(1-t)$ 这种代换形式整理一下，最终得到：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, z_0, z_1}\left[ \frac{1}{(1-t)^2} || \hat{z}_1^\theta - z_1 ||_2^2 \right] \tag{3}$$

- $\hat{z}_1^\theta := z_t + (1-t)u_\theta(z_t, t)$：模型对 clean data $z_1$ 的 endpoint prediction
- $\frac{1}{(1-t)^2}$：随 $t \to 1$ 权重爆炸，所以靠近数据端（$t$ 大）时预测必须精确
- 通用 DAE objective (Bengio et al., 2013)：

$$\mathcal{L}_{DAE}(\theta) = \mathbb{E}_{t, z_0, z_1}\left[ || \hat{z}_1^\theta - z_1 ||_2^2 \right] \tag{4}$$

→ **Eq. (3) 就是 Eq. (4) 的 weighted 版本**。这意味着 flow matching 在训练时同时训练了所有 noise level 的 DAE。

### 1.3 Build intuition: 为什么这个 reframe 重要

正常 inference 时，ODE solver 把 $z_{t_0} \to z_{t_T}$ 一路推下去，每步只 forward 一次。但既然 $u_\theta(\cdot, t)$ 在每个固定 $t$ 上都是 DAE，那就可以在固定 $t$ 上反复"腐蚀–重建"做 pseudo-Gibbs。这是 Bengio 2013 "Generalized Denoising Auto-encoders as Generative Models" 的核心思路 (link: https://arxiv.org/abs/1305.6663)。

Pseudo-Gibbs 的 markov chain：给一个 corrupted state $z_t$，先 denoise 得到 $\hat{z}_1$，再把 $\hat{z}_1$ 用同样 noise level $t$ 重新 corrupt 回去。这就构成 invariant distribution 偏向 data manifold 的 markov chain，迭代会拉向 high-density 区域。

---

## 2. Predict-and-Perturb (P&P): 算法细节

### 2.1 两个算子

**Predict** (reconstruction / denoiser)：

$$D_\theta(z_t, t) := z_t + (1-t) u_\theta(z_t, t) \tag{5}$$

- 输入：当前 noisy latent $z_t$
- 输出：对 clean latent $\hat{z}_1$ 的预测

**Perturb** (corruption / local resampling)：

$$R_\epsilon(z, t) := tz + (1-t)\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) \tag{6}$$

- 输入：clean 预测 $\hat{z}_1$
- 输出：在 noise level $t$ 上重新加噪后的 latent
- 注意 $R_\epsilon(\hat{z}_1, t) = t\hat{z}_1 + (1-t)\epsilon$，几何上就是在 $\hat{z}_1$ 和 noise $\epsilon$ 之间做插值

### 2.2 一次 P&P 迭代

$$\hat{z}_1^{(k)} := D_\theta(z_t^{(k)}, t)$$
$$z_t^{(k+1)} := R_{\epsilon_k}(\hat{z}_1^{(k)}, t) \tag{7}$$

合并写成：

$$z_t^{(k+1)} = \mathrm{P\&P}_{\epsilon_k}(z_t^{(k)}, t) := R_{\epsilon_k}(D_\theta(z_t^{(k)}, t)) \tag{8}$$

- $k$：refinement 迭代索引，$k = 0, 1, ..., K_f$
- $\epsilon_k$：第 $k$ 步新采样的 Gaussian noise（每步新 noise，让它探索）
- $K_f$：refinement 总轮数，paper 里默认 2-3 就够

### 2.3 和 ODE solver 集成

把 P&P 嵌进 Euler ODE solver：

$$z_{t_{i+1}} = z_{t_i}^* + \Delta t \cdot u_\theta(z_{t_i}^*, t), \quad \Delta t = t_{i+1} - t_i \tag{9}$$

- $z_{t_i}^*$：P&P 跑完 $K_f$ 步后的 refined latent
- 只替换 ODE 在 timestep $t_i$ 处的"种子"latent，下一步照样推

### 2.4 早期 lock-in 现象 → 只在早期 timestep 加 P&P

Video diffusion 有个 well-known 现象 (VideoJAM, FlowMo, Frame Guidance, Jang et al. 2025)：**motion 和 physics 在前 20% timestep 就基本定死了**。所以 P&P 只在 $t < 0.2$ 应用就够，后期 refinement 几乎无效。

paper Algorithm 1 的核心：
- 第 3 行：base ODE step
- 第 4 行 `if i \leq \alpha T`：定义 "motion stage"（早期）
- 第 5-12 行：在 motion stage 内做 $K_f$ 轮 P&P
- 第 13 行：用 refined latent 替换
- 第 14-16 行：非 motion stage 直接走 base ODE

**额外 NFE**：$K_f=3$ 时大约多 20 NFE，对 40-NFE 的 base sampler 是 1.5× 推理时间。比起 FlowMo 的 3.9× 和 best-of-4 rejection sampling 的 4.0× 便宜很多。

---

## 3. Uncertainty-aware P&P: 防止 over-saturation

### 3.1 问题：为什么多次 P&P 会 over-saturate

每个 P&P 步之后，要做 CFG update (Ho & Salimans, 2022)。CFG 的本质是放大 scale，paper 里说 "amplified scale (i.e. $1-t$ instead of $\Delta t$)"——意思是 predict 算子里 $u_\theta \cdot (1-t)$ 这个系数在 $t$ 小时很大（接近 1），而正常 ODE step 用的系数是 $\Delta t$（很小）。

→ 在 static region（背景），多次 P&P 后 $z_t^{(k)}$ 几乎没变化，但每次 predict 都用大 CFG scale 重新推一遍，guidance 不断累积，导致 over-saturation（Sadat et al., 2024, "Eliminating oversaturation and artifacts of high guidance scales", link: https://openreview.net/pdf?id=B0WqjD5Ol3）。

### 3.2 Self-consistency uncertainty

不需要外部 model，直接用 P&P 内部信息：

$$\mathbf{U}(z_t^{(k-1)}, z_t^{(k)}) := \frac{1}{C} || D_\theta(z_t^{(k-1)}, t_i) - D_\theta(z_t^{(k)}, t_i) ||_1 \tag{10}$$

- $C$：latent channel 维度
- 范数：先在 channel 维度 average，再在 spatio-temporal 维度算 L1
- $D_\theta(z_t^{(k)}, t)$：第 $k$ 次 P&P 之后的 clean 预测 $\hat{z}_1^{(k)}$

直觉：如果两次 predict 出来的 $\hat{z}_1$ 差异大，说明这个 spatio-temporal 位置 model 自己也不确定；差异小（如背景）说明 model 很确信，不需要 refine。

阈值化得 mask：

$$M_{t_i}^{(k)} := \mathbb{1}(\mathbf{U}(z_t^{(k-1)}, z_t^{(k)}) > \tau) \tag{10}$$

- $\tau$：confidence threshold，默认 0.25
- $M = 1$：refine（uncertain）
- $M = 0$：保留前一轮（certain）

### 3.3 Mask 怎么用：零额外 NFE

更新规则：

$$z_{t_{i+1}}^{(k)} \leftarrow M_{t_i}^{(k)} \odot z_{t_{i+1}}^{(k)} + (1 - M_{t_i}^{(k)}) \odot z_{t_{i+1}}^{(k-1)} \tag{11}$$

- $\odot$：element-wise multiplication
- 注意 $z_{t_{i+1}}^{(k-1)}$ 是上一轮 P&P 已经算出来的 next-step latent，不重算
- → 关键 trick：mask 计算复用 Predict 里已经 forward 过的 $D_\theta(z_t^{(k-1)}, t)$ 和 $D_\theta(z_t^{(k)}, t)$，所以 0 额外 NFE

Algorithm 2 line 11 还有个细节：mask 是累加的 (`m_unc = (uncertainty > tau) | buffer[2]`)，意思是某个区域一旦被标 uncertain，后面所有迭代都保持 refine（避免 flicker）。

### 3.4 直觉：uncertainty map 物理含义

Fig. 3 可视化显示：uncertainty 高的地方就是 moving object（人体运动），uncertainty 低的地方就是 background。这说明 model 自己的预测一致性已经自然地"看到"了哪里需要 refine。这跟传统 variance-based uncertainty estimation（比如 BayesDiff, Kou et al. 2024, link: https://arxiv.org/abs/2306.05453）需要 N=5 次 stochastic forward pass 相比，paper 这版是免费的。

---

## 4. 实验：核心数据解读

### 4.1 Motion coherence (Dynamic-bench, Tab. 1)

Wan2.2-A14B T2V，对 baseline 的 tie-adjusted win rate：

| Method | Motion (%) ↑ | Text (%) ↑ | NFE | Time |
|---|---|---|---|---|
| Default (UniPC) | 73.57 | 57.64 | 40 | 1.0× |
| +NFE×2 | 74.05 | 57.55 | 80 | 2.0× |
| +CFG-Zero | 81.53 | 65.71 | 40 | 1.0× |
| +FlowMo | 70.57 | 61.71 | 40* | 3.9× |
| **+Ours** | **(win rate, 73% over default)** | | 60 | 1.5× |

- "Motion (%)" 对 default row 是 win rate vs ours，对 ours row 留空（73% 人偏好 ours）
- FlowMo 70.57% 也表示偏好 ours（差 70.57%）
- 注意 NFE×2 几乎没用（73→74%），说明单纯增 NFE 没法解决 motion 问题
- VBench 上 ours 的 Motion 98.41（最高）、Consistency 91.33（最高）

### 4.2 Robotics I2V (PAI-Bench, Tab. 2)

Cosmos-Predict-2.5-2B：

| Method | Grasp ↑ | Robot-QA ↑ | Quality ↑ | NFE | Time |
|---|---|---|---|---|---|
| Default | 79.2 | 71.7 | 75.1 | 35 | 1.0× |
| +Verifier (best-of-4) | 84.4 | 72.3 | 75.3 | 140 | 4.0× |
| **+Ours** | **89.6** | **76.3** | 75.1 | 57 | 1.6× |

Wan2.2-I2V-A14B:

| Method | Grasp ↑ | Robot-QA ↑ |
|---|---|---|
| Default | 77.3 | 77.4 |
| +Verifier (best-of-4) | 80.5 | 78.1 |
| **+Ours** | **85.7** | **80.3** |

- Grasp +11.0% (Cosmos) / +8.4% (Wan)，对 robotics deployment 是巨大的
- 比 verifier-based best-of-4 又快又好——这个结果很有说服力，因为 best-of-4 用了 Cosmos-Reason1 7B 当 critic (link: https://arxiv.org/abs/2503.15558)，相当于把 verifier 的知识也算进去了

### 4.3 Physics alignment (VideoPhy2, Tab. 3)

VideoPhy2 + PhyWorldBench 自动评估，PC = Physical Commonsense, SA = Semantic Alignment：

| Method | VideoPhy2 PC↑ | VideoPhy2 SA↑ | PhyWorldBench PC↑ | PhyWorldBench SA↑ |
|---|---|---|---|---|
| Wan2.2 T2V | 54.5 | — | 28.6 | — |
| +NFE×2 | 53.1 | 66.1 | 29.3 | 78.1 |
| +CFG-Zero | 50.6 | 67.0 | 29.3 | 80.1 |
| **+Ours** | **55.6** | **66.2** | **40.0** | **78.6** |

- PhyWorldBench 上 PC 从 28.6 → 40.0（+11.4 绝对，~40% 相对提升），这是大跳
- Human eval: 84% 偏好 ours over default，74% 偏好 ours over NFE×2

### 4.4 Spatial consistency (Tab. 5)

Wan2.2 T2V，让 camera 做 >360° rotation 再回到原视角，用 MegaSAM (link: https://arxiv.org/abs/2438.09012) 估计 camera pose，对比 revisit viewpoint 的 frame：

| Method | SSIM ↑ | L1 ↓ | PSNR (dB) ↑ | NFE |
|---|---|---|---|---|
| Default | 0.401 | 37.26 | 14.96 | 40 |
| +Ours | 0.485 | 30.16 | 17.21 | 60 |

- SSIM 0.401 → 0.485 (+21%)
- PSNR +2.25 dB（这个 gap 很大）
- 这告诉我们 P&P 不仅 refine motion，还把"early-stage latent 推向 high-density region"这件事让 spatial layout 更稳定

### 4.5 Visual reasoning (Fig. 18, 19)

Wiedemer et al. 2025 "Video models are zero-shot learners and reasoners" (link: https://arxiv.org/abs/2509.20328) 引入的 visual reasoning 任务：

- **Graph traversal**: base 成功率 0.1 → +Ours 0.8（巨大提升）
- **Maze solving**: 几乎 0 → 几乎 0（没改善）

直觉解释：graph traversal 错误可以被局部 motion 修正（"水沿着连通节点流"的传播错误可以在每帧 P&P 时纠正），而 maze solving 需要全局 path planning（离散决策），local search 救不了。这其实跟 LLM 的 self-refine 限制类似——self-refine 能修局部错误，但修不了 fundamental knowledge gap。

---

## 5. Discussion section 的精彩部分

### 5.1 Cross-frame consistency: video vs image 的根本区别

Fig. 13(a) 用 SDEdit 改 prompt (orange cat → brown dog)：
- Image: 明显 semantic 转换
- Video: 内容几乎不变

Fig. 13(b) 多次 P&P 比较：
- Image: 单次 P&P 就大幅偏离
- Video: 多次 P&P 几乎不变（local search）

**为什么 video 这么 robust**：相邻帧共享 layout 和 motion trajectory，单帧扰动被邻帧"拉回"。这跟 video diffusion 训练时强制 temporal attention 一致性有关。

→ 这给 P&P 在 video 里能多次迭代而不崩的理论支持：在 image 里 K=8 就 collapse 成"白山羊"（Fig. 22 mode-seeking），video 里 K=8 只是 reduce temporal jitter。

### 5.2 Mode-seeking behavior

Fig. 21 toy example：2D Gaussian mixture 上反复 P&P，sample 集中到 high-density mode。

- Image domain: K=8 把"an animal" collapse 成白山羊（Fig. 22）
- Video domain: 反而保留 content 但 reduce temporal variance（去除 flickering）

→ 称之为 "intended temporal mode-seeking"——mode-seeking 在这里反而是好事，因为 temporally inconsistent video 本来就在 low-density region，refinement 就是把它推向 temporally consistent mode。

### 5.3 跟 ALD / Restart / FreeInit 的对比

| Method | Stochasticity 位置 | 目标 | 在哪里 refine |
|---|---|---|---|
| Annealed Langevin Dynamics (Song & Ermon, 2019) | 多个 annealed noise scale | 严格 MCMC 逼近 target | 整个 noise schedule |
| Restart (Xu et al., 2023) | macro forward-backward（往前跳再 ODE 回） | 减少累积误差 | 后期 |
| FreeInit (Wu et al., 2024) | 只 refine 初始 noise $z_{t_0}$ | 全程重新 denoise | 起点 |
| **P&P** | 同 noise level 内 local resampling | mode-seeking, non-strict MCMC | 早期 timestep 内部 |

P&P 跟 ALD 的核心区别：ALD 跨多个 annealed noise level，P&P 在固定 $t$ 内反复扰动。P&P 跟 Restart 的区别：Restart 是 macro 往前跳到高 noise 再回 ODE，P&P 是 fixed $t$ 内 fine-grained refinement。P&P 跟 FreeInit 区别：FreeInit 改 $z_{t_0}$ 然后整个 denoise 重跑，P&P 改 intermediate $z_t$ 在同一 trajectory 内 refine。

---

## 6. Ablation (Sec. 5.6, A.7)

### 6.1 $K_f$ vs $\tau$ (Fig. 16)

- $K_f = 1$：refinement 不够，大 motion 修不动
- $K_f = 3, \tau = 0.25$：默认，最好 trade-off
- $K_f = 5, \tau = 0$：over-saturation，水面上夸张反射
- $K_f = 5, \tau = 0.5$：refine 局限到 very uncertain 区域，artifact 消失但 refine 弱

### 6.2 $\alpha$ (Fig. 17)

P&P 在早期 (3-14 step, $t < 0.1$) 应用效果好；后期 (step 6-10, $t \in (0.05, 0.1)$) 应用几乎没用；只后期应用则效果有限（motion 已经定死了）。

### 6.3 P&P plan (Table 6)

实际配置更细：用 mapping `step_range : K_f`，比如 `{3-6: 3, 7-14: 1}` 表示 step 3-6 各做 3 轮 P&P，step 7-14 各做 1 轮。这样 total NFE 控制 ~1.5×。

---

## 7. 跨模型泛化（B.6, Fig. 23）

虽然 paper 主打 flow matching 模型（Wan, Cosmos），但 P&P 在 diffusion-based CogVideoX 上也 work，修了"光剑被截断"和"泰迪熊嘴部扭曲"的 artifact。这印证了 "DAE 视角"的普适性——DDPM/DDIM 的 noise prediction $\epsilon_\theta$ 也能类似地改写成 endpoint predictor。

FLUX.1-dev (link: https://github.com/black-forest-labs/flux) 上 image generation 也 work（Fig. 15），只加 2 NFE (4% overhead) 就显著改善 text rendering。这暗示 P&P 在所有 flow matching / diffusion generator 上都适用。

---

## 8. 关键直觉总结（build your intuition）

1. **Flow matching 训练时已经训了所有 noise level 的 DAE**，inference 时只用一次太浪费。P&P 让你"白嫖"这个 hidden structure。

2. **Video 的 cross-frame consistency 是 P&P 能多次迭代的关键**——在 image 里 multi-step P&P 会 collapse，在 video 里反而能稳定 accumulate refinement。

3. **Mode-seeking 是 feature 不是 bug**：在 video domain，"high-density mode"对应"temporal consistent video"，所以 mode-seeking 就是去 flickering。

4. **早期 lock-in 让 P&P 只需在前 20% timestep 应用**，这是 free lunch——加一点 NFE，refine 决定 motion 的关键时刻。

5. **Uncertainty mask 是免费的 self-supervised signal**——compare consecutive P&P predict 的差异就够，不需要外部 verifier，不需要 5x stochastic forward pass。

6. **Limitation: local search 修不了 fundamental knowledge gap**——maze solving 这种需要全局 path planning 的任务，P&P 帮不上。这跟 LLM self-refine 修不了 reasoning gap 类似。

7. **Verifier-based best-of-4 又贵又弱**：PAI-Bench 上 best-of-4 (Cosmos-Reason1 当 critic) Grasp 84.4%，P&P 直接 89.6%。这暗示 inference-time refinement 比外部 verifier + rejection sampling 更 fundamental。

---

## 9. 可能的延伸联想

- **跟 test-time scaling for LLM 的关系**：Karan & Du 2025 "Reasoning with Sampling" (link: https://arxiv.org/abs/2510.14901) 用 base LLM 做 MCMC 不用 RL——和 P&P 哲学一致。video 这边 P&P 是 visual analog。

- **跟 inference-time search 的关系**：可以视为 video domain 的 MCTS-lite——P&P 是 local rollout，每次 predict 就是 evaluate，perturb 就是 explore。

- **跟 consistency models 的关系**：consistency model 训练目标就是"任意 noise level 直接预测 endpoint"，跟 $\hat{z}_1^\theta := z_t + (1-t)u_\theta$ 是一回事。P&P 可以看作"用 consistency-style predictor 做 inference-time refinement"。

- **World model 角度**：Brooks et al. 2024 Sora (link: https://openai.com/research/video-generation-models-as-world-simulators) 把 video generator 当 world model，P&P 让 world simulation 更物理一致，可能是让 video diffusion 真的可用作 robot simulator 的关键 inference-time 技巧。

- **跟 score-based diffusion 的 Langevin connection**：Song & Ermon 2019 (link: https://arxiv.org/abs/1907.05600) 用 score + Langevin 生成，跟 P&P 的 perturb-predict 结构镜像——可以把 P&P 理解为"在 flow matching 上做 approximate Langevin"。

- **Potential extension**: 把 P&P 跟 reward model guided sampling 结合，比如 Liu et al. 2025b "Improving video generation with human feedback" (link: https://arxiv.org/abs/2501.13918) 的 reward gradient guidance，可以作为 P&P 的"外部信号"补充。

- **Potential extension**: uncertainty mask 可以做 time-dependent $\tau_t$（paper Sec B.4 提到了但留作 future work），早期容忍更多 refine，后期保守。

- **Limitation 的潜在解法**：maze solving 这种离散/语义任务，可以混合 P&P + VLM verifier，把局部 refine 和全局 check 结合。这跟 tree search + value network 的 AlphaGo 结构类似。

---

## 10. Paper 信息

- 标题：Self-Refining Video Sampling
- 作者：Sangwon Jang, Taekyung Ki, Jaehyeong Jo (KAIST, 共同一作), Saining Xie (NYU), Jaehong Yoon (NTU), Sung Ju Hwang (KAIST/DeepAuto.ai)
- Project page: https://agwmon.github.io/self-refine-video/
- Base models: Wan2.1 / Wan2.2 (link: https://arxiv.org/abs/2503.20314), Cosmos-Predict-2.5 (link: https://arxiv.org/abs/2511.00062)
- Benchmarks: VideoPhy2 (link: https://arxiv.org/abs/2503.06800), PhyWorldBench (link: https://arxiv.org/abs/2507.13428), PisaBench (link: https://arxiv.org/abs/2504.18476), PAI-Bench (link: https://arxiv.org/abs/2512.01989), VideoJAM-bench (link: https://arxiv.org/abs/2502.05446)

---

## 最后的 intuition

这篇 paper 最 elegant 的地方在于：它没有加任何新东西——没新 model、没新 training、没新 verifier、没新 reward——只是"重新解读"已有 flow matching model 的训练目标，把 inference 当成"被压扁的 markov chain"展开成"完整 pseudo-Gibbs"。这种"用对的方式使用已有的东西"的思维方式，比单纯堆参数/堆数据更接近 physical AI 的下一个突破点。

Cross-frame consistency 是 video 独有的"spatial–temporal inductive bias"，P&P 利用了这一点让 multi-step refinement 在 video 上稳定——这跟 image 上的 SDEdit 类方法形成鲜明对比。这种"利用 medium-specific structure 做 inference-time algorithm"的思路，未来在 audio、3D generation 上可能也有类似机会。

如果你（Karpathy）在思考 micro-world model / simulator 方向，P&P 给了一个很有用的 building block：让 pre-trained generator 在 inference 时"自我仿真校正"，不需要 retrain。这跟你在 Eureka Labs / nanoGPT 后续工作里强调的"让模型自己用自己"的哲学其实非常契合。
