---
source_pdf: Inference-Time Policy Steering through Human Interactions.pdf
paper_sha256: 270df8a7a4b7feb68a19f429033d650f92979cea4c0504f7ac15ef873088c0bc
processed_at: '2026-08-05T09:32:01-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说ITPS

## 一句话版本

你训好了一个robot policy, 它自己干活挺行, 但有时候你想插嘴说"别走左边, 走右边", 或者"别抓红碗, 抓蓝碗"。你怎么插嘴?

最naive的做法是你直接把robot的手掰到右边去——但这会把policy搞懵, 因为它从来没见过这种state, 后续动作全崩。这篇paper说: 别硬掰, 咱们在diffusion sampling的过程中间偷偷加一个gradient, 让policy自己往右边靠, 但始终留在它"认识"的distribution里。

就这么个事儿。

---

## 为什么这是个真问题

你训Diffusion Policy的时候, 给它看了几百万条human demonstration。它学到了一个**motion manifold**——一个"我知道怎么动是合理的"的高维空间。这个manifold里包含了collision-free的轨迹、valid的grasp姿态、合理的pick-and-place顺序等等。

Problem是, inference的时候policy自己sample出来的trajectory可能不是user想要的。比如厨房场景里policy随机决定"把碗放水槽", 但你想让它放微波炉。

你能怎么办?

**Option 1: Fine-tune**。拿language correction数据再训一遍。贵, 慢, 而且language这种discrete modality根本说不清"我要你走的轨迹稍微弧一点"这种low-level continuous intent。

**Option 2: 硬override**。直接把robot end-effector掰到你想要的位置。但这就是经典的DAgger问题 [13] (https://arxiv.org/abs/1011.0686)——你把policy推到了OOD state, 它见都没见过这个state, 下一步prediction直接崩。Ross, Gordon, Bagnell 2011年就说过这个事儿。

**Option 3: 这篇paper的ITPS**。不硬掰, 而是在diffusion的denoising process里加一个soft gradient, 引导sampling往user intent走, 但每一步都保持在policy的data manifold附近。

---

## 三种Human Interaction的直觉

Paper考虑了三种user表达intent的方式:

### Point Input — "去那儿"

你在camera画面上点一下, 说"去这个位置"。系统用RGB-D把pixel反投影成3D坐标 $\mathbf{z}^{\text{point}}$。

Objective就是让generated trajectory的所有state平均距离这个point越近越好:

$$\xi(\tau, \mathbf{z}^{\text{point}}) = \sum_{t=1}^{T} \frac{1}{T} \|\mathbf{s}_t - \mathbf{z}^{\text{point}}\|_2$$

这里 $\tau = (\mathbf{s}_1, \ldots, \mathbf{s}_T)$ 是trajectory, $\mathbf{s}_t \in \mathbb{R}^3$ 是第$t$步的end-effector位置, $\mathbf{z}^{\text{point}} \in \mathbb{R}^3$ 是用户点的那个3D点。$\frac{1}{T}$ 就是时间平均, 让objective不因为trajectory长短而scale不同。

### Sketch Input — "沿着这条路走"

用户在workspace里画一条partial trajectory, 系统resample到匹配的temporal length后作为target $\mathbf{z}^{\text{sketch}} \in \mathbb{R}^{T \times 3}$。

$$\xi(\tau, \mathbf{z}^{\text{sketch}}) = \sum_{t=1}^{T} \|\mathbf{s}_t - \mathbf{z}_t^{\text{sketch}}\|_2$$

注意这里没有 $\frac{1}{T}$, 每一步都要match对应的sketch点。这个比point input更具体——你不仅指定终点, 还指定路径形状。

### Physical Correction — "我来掰你"

用户直接物理干预robot, 把end-effector推到某个位置。前 $k$ 步被hard override:

$$\tau = [\mathbf{z}_1^{\text{nudge}}, \ldots, \mathbf{z}_k^{\text{nudge}}, \mathbf{s}_{k+1}, \ldots, \mathbf{s}_T]$$

这是最直接的intervention, 也是最容易出事的——因为nudge出来的state完全可能在policy的data manifold之外。

---

## 六种Steering方法的直觉

Paper比较了6种把user intent注入policy的方法。我按从naive到sophisticated的顺序讲:

### 1. Random Sampling (RS) — 啥都不做

直接 $\tau \sim \pi_\theta$。Baseline。

### 2. Output Perturbation (OP) — 先sample再改

先让policy sample一条trajectory, 然后把前 $k$ 步替换成用户的nudge, 剩下的从nudge后的state重新sample。

**问题**: nudge后的state可能OOD, 后续sample直接崩。Table III的real-world kitchen里OP的success rate只有37%。

### 3. Post-Hoc Ranking (PR) — 生成一堆, 挑最像的

Sample一个batch (比如32条trajectory), 用 $\xi$ 算每条和user intent的距离, 选最近的那条。

**优点**: 零cost (就是多sample几条), 而且每条sample都是in-distribution的。

**缺点**: 前提是batch里至少有一条接近target。如果policy在这个state下是unimodal的, 或者所有mode都离target很远, PR就没办法。

Table I里ACT (unimodal policy) 上PR的Min L2 = 0.26, 和RS一模一样——因为ACT就一个mode, 挑来挑去都一样。DP (multimodal) 上PR从0.27降到0.16, 明显提升。

**直觉**: PR本质是approximate rejection sampling。你从 $\pi_\theta$ 里采样, 然后用 $\xi$ 做selection。只要 $\pi_\theta$ 有足够的diversity, PR就能发现接近target的sample。

### 4. Biased Initialization (BI) — 从用户input开始denoise

正常diffusion从纯噪声 $\tau_N \sim \mathcal{N}(0, I)$ 开始denoise。BI改成用Gaussian-corrupted user input作为起点:

$$\tau_N = \sqrt{\bar{\alpha}_N} \mathbf{z}^{\text{sketch}} + \sqrt{1 - \bar{\alpha}_N} \epsilon$$

这里 $\bar{\alpha}_N = \prod_{i=1}^N \alpha_i$ 是DDPM的cumulative variance schedule, $\epsilon \sim \mathcal{N}(0, I)$。

**直觉**: 你把sketch加上对应noise level $N$ 的Gaussian noise, 作为diffusion的起点。这样denoise过程从一开始就"偏向"sketch的方向。

**问题**: diffusion的每一步denoise都在把sample往data manifold拉, 所以initialization的偏置会逐渐被"washed out"。Table I里BI的collision rate = 0.06, 是SS的6倍——说明initialization偏置确实把sample推到了离manifold有点远的地方。

灵感来自Yoneda et al. [24] (https://arxiv.org/abs/2302.12244) 的"To the noise and back"。

### 5. Guided Diffusion (GD) — 每步加alignment gradient

这是classifier-guided diffusion在trajectory generation上的应用。每个denoising step里, 除了denoising gradient, 再加一个alignment gradient:

$$\tau_{i-1} = \alpha_i \left(\tau_i - \gamma_i \big(\epsilon_\theta(\tau_i, i) + \beta_i \nabla_{\tau_i} \xi(\tau_i, \mathbf{z})\big)\right) + \sigma_i \eta$$

变量逐个解释:
- $i$: diffusion timestep, 从 $N$ (纯噪声) 到 $1$ (clean trajectory)
- $\tau_i$: 第 $i$ 步的noisy trajectory
- $\epsilon_\theta(\tau_i, i)$: denoising network预测的noise, 参数是 $\theta$
- $\nabla_{\tau_i} \xi(\tau_i, \mathbf{z})$: objective $\xi$ 对 $\tau_i$ 的gradient, 通过backprop计算
- $\beta_i$: guide ratio, 控制alignment gradient的强度
- $\alpha_i, \gamma_i, \sigma_i$: DDIM scheduler的系数
- $\eta \sim \mathcal{N}(0, I)$: 随机噪声

**直觉**: 每步denoise时, 有两个力在拉——denoising gradient把sample往data manifold拉, alignment gradient把sample往user intent拉。$\beta_i$ 控制两者的相对强度。

**关键问题**: 这个gradient sum $\nabla \log p + \beta \nabla \log q$ 看起来对应product distribution $p \cdot q$ 的score, 但**实际上单步更新不够让sample mix到product distribution**。Du et al. [21] (https://arxiv.org/abs/2302.08458) 指出, 这更像是采样了 $p + q$ 的mixture而不是 $p \cdot q$ 的product。

Figure 3的toy example特别直观:
- 两个data mode (两个高斯峰), 一个target point在中间远离两个mode
- GD的sample会spread到两个mode和target之间的中间区域——这是mixture的行为
- SS的sample会找到离target最近的那个data mode, 然后在那里align——这是product的行为

所以GD在guide ratio大的时候会产生OOD sample。Table I里GD的collision rate = 0.06, Table III里GD with $\beta_i = 100$ 的success rate只有15%。

### 6. Stochastic Sampling (SS) — 每步多跑几轮MCMC

这是paper的核心贡献。核心idea: 在每个diffusion timestep, 不只做一次gradient update, 而是做 $M$ 步MCMC, 让sample充分mix到product distribution $p_i(\tau) \cdot q(\tau)$。

Update equation:

$$\tau_i = \tau_i - \gamma_i \big(\epsilon_\theta(\tau_i, i) + \beta_i \nabla_{\tau_i} \xi(\tau_i, \mathbf{z})\big) + \sigma_i \eta$$

重复 $M-1$ 次 (注意: timestep不变, 还是 $i$), 然后用公式(5)做一次reverse step到 $\tau_{i-1}$。

和GD的区别就一行: GD每步直接 $\tau_i \to \tau_{i-1}$, SS每步先在 $\tau_i$ 上iterate $M$ 次再 $\tau_i \to \tau_{i-1}$。

Algorithm 1:

```
Input: diffusion policy π_θ, user interaction z, alignment objective ξ(·)
1: Initialize plan τ_N ∼ N(0, I)
2: for i = N, . . . , 1 :        // denoising steps
3:   for j = 1, . . . , M :      // sampling steps (NEW)
4:     ϵ ← π_θ(τ_i)              // denoising gradient
5:     δ ← ∇ξ(τ_i, z)            // alignment gradient
6:     if j < M:                 // (NEW)
7:       τ_i ← reverse(τ_i, ϵ + β_i·δ, i)      // stay at timestep i (NEW)
8:     else:
9:       τ_{i−1} ← reverse(τ_i, ϵ + β_i·δ, i−1) // advance to i-1
```

就加了4行代码。

**为什么这work?** Langevin MCMC的理论: 多步iterate能让sample converge到target distribution $\propto e^{-U(\tau)}$ where $U(\tau) = -\log p_i(\tau) - \log q(\tau)$。单步更新不够, sample卡在gradient descent的中间状态。多步iterate让sample在fixed noise level充分explore, 最终mix到product distribution。

**实现trick**: 在fixed noise level $i$ 上做MCMC update, 需要先reverse sample得到intermediate clean prediction $\tilde{\tau}_0$, 然后forward diffuse回noise level $i$。这是DDIM inversion的变体。

**代价**: $M=4$ 意味着总denoising calls是 $4N$, 4x慢。Paper在limitation里承认这一点, 说future work要distill。

---

## 为什么Multimodal Policy是前提

这是paper的一个隐含insight, 在实验里特别明显:

**ACT** [7] (https://arxiv.org/abs/2303.04137) 是VAE-based, latent space是unimodal的。Table I里ACT上所有steering方法都失败——Min L2 stuck at 0.26, GD和SS甚至跑不起来 (因为ACT没有explicit score function)。

**Diffusion Policy** [6] (https://arxiv.org/abs/2303.04137) 是multimodal的, 能同时表示"去左边"和"去右边"两个mode。Table I里DP + SS的Min L2 = 0.10, collision = 0.01。

**直觉**: Steering本质上是在data manifold里找"最接近user intent的那个mode"。如果policy本身就是unimodal的, 只有一个mode, 那steering再怎么搞也只能在这个mode附近动, 没法discover新mode。

PR的依赖更直接: PR是sample一个batch然后select, 如果所有sample都来自同一个mode, select来select去都一样。

---

## Maze2D实验的核心发现

Maze2D [25] (https://arxiv.org/abs/2004.07219) 是个2D navigation环境, policy只在collision-free的random walk上训过, 没有任何goal objective。

Inference时给一个可能collision-violating的sketch, 看各种steering方法能不能既align sketch又不撞墙。

Table I的核心数据:

| Policy | Method | Min L2 ↓ | Collision ↓ |
|--------|--------|----------|-------------|
| DP | RS | 0.27 | 0.01 |
| DP | PR | 0.16 | 0.01 |
| DP | BI | 0.11 | 0.06 |
| DP | GD | 0.11 | 0.06 |
| DP | **SS** | **0.10** | **0.01** |

**读法**: 
- RS baseline: L2=0.27, 几乎不撞墙
- PR: L2降到0.16, collision没涨——免费的alignment
- BI和GD: L2降到0.11, 但collision涨到0.06——alignment提升但开始OOD
- SS: L2=0.10 (最低), collision=0.01 (和RS持平)——最佳trade-off

**直觉**: SS找到了离sketch最近的collision-free mode。sketch可能穿过墙, 但SS不会傻乎乎跟着sketch走然后撞墙, 而是找到data manifold里离sketch最近的那个collision-free trajectory。

---

## Block Stacking实验的Guide Ratio Schedule Insight

这个实验里有个特别elegant的发现。Figure 7展示了四种情况:

(a) Unconditional DP sampling: multimodal, 但可能miss user intended plan
(b) PR: 如果batch里没有接近的sample, 无法recover
(c) GD with constant $\beta_i$: recover了intended plan, 但trajectory是curved的, 像sketch——这其实是OOD! 因为CuRobo [28] (https://arxiv.org/abs/2310.17274) 生成的training data都是straight-line的, curved trajectory不在data manifold里
(d) **Modified GD with $\beta_{i \leq 50} = 0$**: 前50步 (高noise)用guidance align低频成分 (overall direction), 后50步 (低noise)关闭guidance让policy自己refine high-frequency细节。结果得到straight-line trajectory, in-distribution, 但discrete alignment正确

**Insight**: Diffusion的early steps (高noise)决定trajectory的low-frequency结构 (去哪个方向, 抓哪个block), late steps (低noise)决定high-frequency细节 (exact path, smoothness)。Guidance应该只在early steps用, late steps让policy自己处理。

这和classifier-free guidance的training-inference asymmetry有异曲同工之妙——guidance在coarse level有效, fine level会破坏sample quality。

Table II的数据:

| Method | TA ↑ | CS ↑ | AS ↑ |
|--------|------|------|------|
| PR | 33% | 100% | 33% |
| GD ($\beta_{i<50}=0$) | 83% | 84% | 67% |
| GD ($\beta_i=100$) | 86% | 15% | 15% |

四个category:
- **AS** (Aligned Success): 对了且成功了
- **AF** (Aligned Failure): 对了但失败了 (over-steering导致OOD)
- **MS** (Misaligned Success): 没对但成功了 (policy自主选了不同但valid的plan)
- **MF** (Misaligned Failure): 没对且失败了

GD with $\beta_i=100$ 的AF=71%——大部分aligned sample都fail了, 因为strong guidance把trajectory推到了OOD region。

---

## Real World Kitchen实验的Practical Takeaway

真实厨房场景, 两个task (放碗进微波炉 vs 放碗进水槽), DP训练40K步。

关键challenge: 合并两个task的dataset后, 有些skill sequence是infeasible的——比如"放碗进微波炉"之前需要"开微波炉门", 如果sequence反了就fail。

Table III:

| Method | TA ↑ | CS ↑ | AS ↑ |
|--------|------|------|------|
| RS | - | 90% | ~34% |
| GD ($\beta_i=5$) | 38% | 82% | 34% |
| **SS** ($\beta_i=100$) | **71%** | 73% | **55%** |
| OP | 89% | 37% | 30% |

**注意GD的 $\beta_i=5$ vs SS的 $\beta_i=100$**: 为什么SS能用这么大的guide ratio而GD不能?

Figure 11解释: 
- 小 $\beta_i$: GD和SS都ineffective
- 大 $\beta_i$: GD产生incoherent trajectory (OOD崩了), SS成功识别intended skill

**直觉**: GD的单步gradient update在大 $\beta_i$ 下直接把sample推到OOD region, 没有自我纠正机制。SS的MCMC多步iterate让sample在fixed noise level充分explore, 即使大 $\beta_i$ 也能stay in-distribution——因为每步MCMC都在往 $p_i(\tau) \cdot q(\tau)$ 的mode靠, 而这个mode一定在data manifold里 (因为 $p_i(\tau)$ 的support在data manifold上)。

SS比RS的AS提升21% (55% vs ~34%), without any fine-tuning。这是paper最亮眼的real-world result。

---

## 核心Intuition: 为什么Product > Sum

这是整篇paper最深的insight, 来自Du et al. [21] (https://arxiv.org/abs/2302.08458)。

你有两个distribution:
- $p(\tau)$: policy的data distribution, support在valid trajectories上
- $q(\tau) \propto e^{-\xi(\tau, \mathbf{z})}$: user intent的EBM, energy低的地方是user想要的

你想要的是 $p \cdot q$: 既valid又align的trajectory。

**GD的做法**: 每步用 $\nabla \log p + \beta \nabla \log q$ 做gradient update。这看起来是 $\nabla \log(p \cdot q) = \nabla \log p + \nabla \log q$, 但单步gradient descent不够让sample converge到 $p \cdot q$。实际上sample会卡在 $p$ 和 $q$ 之间的某个中间区域——特别是当 $p$ 和 $q$ 的mode不overlap时, sample会被拉到两个mode之间的"no man's land", 也就是OOD。

**SS的做法**: 每步做 $M$ 步Langevin MCMC。Langevin MCMC的theory保证多步iterate后sample converge到target distribution $\propto e^{-U}$ where $U = -\log p - \log q$, 也就是 $p \cdot q$。多步iterate让sample有足够时间"发现" $p \cdot q$ 的mode在哪里——这个mode一定在 $p$ 的support内 (因为 $p \cdot q$ 只在 $p > 0$ 的地方非零), 所以一定in-distribution。

**Figure 3的toy example最直观**: 
- $p$ 有两个mode (左峰和右峰)
- $q$ 在中间有个target point
- $p \cdot q$: 在离target最近的那个mode附近有density, 另一个mode的density被 $q$ 压低了
- $p + q$: 在两个mode和target之间都有density, 中间区域也有非零density——这就是OOD

GD sample出来的contour lines像 $p + q$, SS的像 $p \cdot q$。

---

## 这篇Paper的Limitation

1. **慢**: SS的 $M=4$ 让inference 4x慢。Real-time HRI里7Hz的rollout频率对M=4的SS来说很紧。Future work: distill steering process成interaction-conditioned policy, 一次性output aligned trajectory。

2. **没做user study**: Paper说了要做但没做。Steerability的human factors evaluation缺失。

3. **只测了DP和ACT**: 没测更新的generalist policies like OpenVLA [4] (https://arxiv.org/abs/2406.09246) 或 Octo [3] (https://arxiv.org/abs/2405.12213)。这些是VLA-based, 不是pure diffusion, steering机制可能不同。

4. **Sketch resampling太naive**: 用uniform resampling而不是DTW [22] (https://link.springer.com/chapter/10.1007/978-3-540-74048-3_4), 对temporal misalignment不robust。

5. **Guide ratio schedule是手动调的**: Block stacking里 $\beta_{i<50}=0$ 的50是手动选的, 没有principle的selection method。

---

## 对你的Intuition Building

如果你要从这篇paper带走一个idea, 就是这个:

**Compositionality of generative models at inference time**。

你有一个pretrained policy $p(\tau)$ 和一个user-specified objective $q(\tau)$。你想compose它们得到 $p \cdot q$。怎么做?

- 如果你用gradient sum $\nabla \log p + \nabla \log q$ 并只做单步update, 你approximate的是 $p + q$ (mixture), 不是 $p \cdot q$ (product)。
- 如果你做足够多步的MCMC, 你能approximate $p \cdot q$。

这个insight不仅适用于robotics, 也适用于任何diffusion-based generation with external guidance——image generation with classifier guidance, text-to-image with layout constraints, protein structure generation with symmetry constraints, 等等。

Du et al. [21] 的"Reduce, Reuse, Recycle"是理论基础, ITPS是robotics HRI的application。如果你对这个方向感兴趣, 还可以看:
- PoCo [47] (https://arxiv.org/abs/2402.02511): 更general的policy composition framework
- BESO [43] (https://arxiv.org/abs/2304.02532): Goal-conditioned diffusion with classifier-free guidance
- SE(3) Diffusion Fields [45] (https://arxiv.org/abs/2305.12734): Cost function gradients for grasp + motion planning

以及更远的connection到consistency models和flow matching——如果能把SS的MCMC过程distill成单次forward pass, 就能解决inference speed的问题。这是我认为这个方向最promising的未来路线。

---

# ITPS: Inference-Time Policy Steering through Human Interactions 深度解析

## 1. Paper的Big Picture与Motivation

这篇paper来自MIT (Yanwei Wang, Lirui Wang, Yilun Du, Julie Shah) 与NVIDIA (Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia Perez-D'Arpino, Dieter Fox)的collaboration。核心problem statement可以一句话概括:

**当下behavior cloning训练出的generalist policies (例如Diffusion Policy [6], ACT [7], OpenVLA [4], Octo [3])在inference阶段把human踢出了control loop**, 当policy prediction和user intent misalign时, 缺乏直接干预机制。Naive human intervention会加剧distribution shift——这是imitation learning里臭名昭著的compounding error问题 (Ross et al. DAgger [13])。

ITPS的核心insight: 把policy steering formulation成**conditional sampling from the likelihood distribution of a learned generative policy**。Likelihood constraints保证actions valid (来自successful demonstrations的data manifold), conditional sampling保证align with user objectives。

项目主页: https://yanweiw.github.io/itps/
arXiv: https://arxiv.org/abs/2410.08005 (推断的arxiv link, 根据作者主页)
Yanwei Wang主页: https://yanweiw.github.io/

---

## 2. 问题Formalization: 三类Human Interaction + 三类Metric

### 2.1 Metrics定义

Paper定义了三个核心metric:

- **Task Alignment (TA)**: 离散任务中预测skill执行intended task的百分比
- **Motion Alignment (MA)**: 连续motion中generated trajectory与target trajectory的负L2距离
- **Constraint Satisfaction (CS)**: 生成的plan满足physical constraints (collision avoidance, task completion)的百分比

**Steering的核心目标**: 最大化TA/MA的同时最大化CS。CS通过从pretrained policy的分布内sampling保证; TA/MA通过最小化objective function $\xi(\tau, \mathbf{z})$ 实现。

### 2.2 三种Human Interaction类型与Objective Function

#### (a) Point Input — 公式(1)

$$
\xi(\tau, \mathbf{z}^{\text{point}}) = \sum_{t=1}^{T} \frac{1}{T} \|\mathbf{s}_t - \mathbf{z}^{\text{point}}\|_2
$$

变量解释:
- $\tau = (\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_T) \in \mathbb{R}^{T \times 3}$: generated trajectory, $T$是temporal length
- $\mathbf{s}_t \in \mathbb{R}^3$: 第$t$步的3D end-effector state
- $\mathbf{z}^{\text{point}} \in \mathbb{R}^3$: 用户在RGB-D camera image上点击的pixel, 通过depth信息反投影到3D scene坐标
- $\frac{1}{T}$: 时间维度上的平均化, 让objective scale-invariant
- $\|\cdot\|_2$: Euclidean distance

Intuition: 这个objective是average L2 distance, 让trajectory的所有states尽可能接近一个target point。在kitchen场景中用户click某个object来specify"接下来去抓这个碗"。

#### (b) Sketch Input — 公式(2)

$$
\xi(\tau, \mathbf{z}^{\text{sketch}}) = \sum_{t=1}^{T} \|\mathbf{s}_t - \mathbf{z}_t^{\text{sketch}}\|_2
$$

变量解释:
- $\mathbf{z}^{\text{sketch}} \in \mathbb{R}^{T \times 3}$: 用户在workspace里画的partial trajectory sketch
- $\mathbf{z}_t^{\text{sketch}}$: sketch在第$t$步的3D位置
- 注意没有$\frac{1}{T}$归一化 (相对公式1)

如果sketch长度和trajectory不一致, paper用uniform resampling把sketch对齐到trajectory的temporal dimension。这本质上是Dynamic Time Warping [22]的简化版本——只用linear interpolation而非DTW的elastic matching。

Intuition: 这允许用户specify trajectory的**shape preference**, 例如"绕过左边障碍物走"。

#### (c) Physical Correction (Nudge) Input — 公式(3)

$$
\xi(\tau, \mathbf{z}^{\text{nudge}}) = \begin{cases} 0, & \mathbf{s}_t = \mathbf{z}_t^{\text{nudge}} \text{ for } t \leq k \\ \infty, & \text{otherwise} \end{cases}
$$

变量解释:
- $\mathbf{z}^{\text{nudge}}$: 用户对robot end-effector的physical correction序列
- $k$: override的步数 (例如前$k$步)
- 当user硬性override前$k$步时, $\tau$的前$k$个state被强制替换:

$$
\tau = [\mathbf{z}_1^{\text{nudge}}, \ldots, \mathbf{z}_k^{\text{nudge}}, \mathbf{s}_{k+1}, \ldots, \mathbf{s}_T]
$$

Intuition: 这是hard constraint——前$k$步必须match用户nudge, 后面步骤由policy生成。这种interaction最直接但最容易induce distribution shift, 因为override部分可能完全out-of-distribution。

---

## 3. 六种Sampling Methods的架构解析

Figure 2展示了5种主要方法的conceptual架构。我把6种全列出来:

### 3.1 Random Sampling (RS) — Baseline

直接$\tau \sim \pi_\theta$, 无任何modification。Serve作为reference baseline。

### 3.2 Output Perturbation (OP)

Pipeline: Sample $\tau \sim \pi_\theta$ → post-hoc perturbation minimize $\xi(\tau, \mathbf{z}^{\text{nudge}})$ → 从$\mathbf{z}_k^{\text{nudge}}$重新sample剩余trajectory。

**致命缺陷**: perturbation后的state $\mathbf{z}_k^{\text{nudge}}$可能完全OOD, policy再从这个state sample时无法保证constraint satisficing。Table III的real-world kitchen结果显示OP的CS只有37%——最aggressive但最容易fail。

### 3.3 Post-Hoc Ranking (PR)

Pipeline: Sample一个batch $\{\tau_j\}_{j=1}^N$ from $\pi_\theta$ → 选$\tau^* = \arg\min_j \xi(\tau_j, \mathbf{z})$。

Intuition: 这个方法本质是**rejection sampling的近似**——前提是batch里至少有一个sample接近target。如果policy prediction是unimodal (例如ACT在某些state下), PR根本无法discover新mode。Table I显示ACT上PR的Min L2 = 0.26, 完全和RS一样, 因为ACT没多modality让PR挑选。

DP multimodal时PR的Table I: Min L2 = 0.16 vs RS的0.27——明显improvement。

### 3.4 Biased Initialization (BI)

灵感来自Yoneda et al. "To the noise and back" [24] (https://arxiv.org/abs/2302.12244)。

Pipeline: 不用$\tau_N \sim \mathcal{N}(0, I)$初始化reverse diffusion, 而是用Gaussian-corrupted user input作为$\tau_N$。

具体: 给定$\mathbf{z}^{\text{sketch}}$, 加上对应noise level $N$的Gaussian noise作为initialization:
$$
\tau_N = \sqrt{\bar{\alpha}_N} \mathbf{z}^{\text{sketch}} + \sqrt{1 - \bar{\alpha}_N} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中$\bar{\alpha}_N = \prod_{i=1}^N \alpha_i$是cumulative variance schedule。

**缺陷**: initialization的偏置会被reverse diffusion process"洗掉", 因为每个denoising step都向data manifold靠拢, 偏置会逐渐消散。

### 3.5 Guided Diffusion (GD) — 公式(5)

这是最经典的classifier-guided diffusion在trajectory上的应用, 来自Janner et al. "Planning with Diffusion" [18] (https://arxiv.org/abs/2205.09991)。

Update equation:

$$
\tau_{i-1} = \alpha_i \left(\tau_i - \gamma_i \big(\epsilon_\theta(\tau_i, i) + \beta_i \nabla_{\tau_i} \xi(\tau_i, \mathbf{z})\big)\right) + \sigma_i \eta
$$

变量详解:
- $i \in \{N, N-1, \ldots, 1\}$: diffusion timestep, 从$N$(纯noise)到$1$(clean trajectory)
- $\tau_i$: 第$i$步的noisy trajectory
- $\epsilon_\theta(\tau_i, i)$: denoising network (U-Net/Transformer) with parameters $\theta$, predicts the noise added to $\tau_i$
- $\nabla_{\tau_i} \xi(\tau_i, \mathbf{z})$: alignment gradient w.r.t. $\tau_i$, 通过backprop通过$\xi$计算
- $\beta_i$: guide ratio, 控制alignment gradient相对denoising gradient的strength
- $\alpha_i, \gamma_i, \sigma_i$: DDPM/DDIM scheduler的hyperparameters
- $\eta \sim \mathcal{N}(0, I)$: stochastic noise注入

**关键insight (Figure 3)**: GD实际采样的是$p_i(\tau) \cdot q(\tau)$的近似, 但严格说是$p_i(\tau) \cdot q(\tau)$的score近似 $\approx \nabla \log p_i(\tau) + \nabla \log q(\tau)$。这里$q(\tau) \propto e^{-\xi(\tau, \mathbf{z})}$是EBM。

但paper指出 (基于Du et al. [21] https://arxiv.org/abs/2302.08458): **gradient sum $\nabla \log p + \nabla \log q$ 对应的分布其实是$p \cdot q$**, 而传统GD在每步只做一次更新时, 由于 Langevin dynamics没有充分mixing, 实际采样的是$p + q$的混合 (mixture)而非$p \cdot q$的product。

具体来说, 在Figure 3的toy example中:
- GD的contour lines近似$p_i(\tau) + q(\tau)$: 当target point远离data mode时, 会把density spread到两个mode之间, 导致OOD samples
- SS的contour lines近似$p_i(\tau) \cdot q(\tau)$: 找到closest in-distribution mode, 然后在那里align

这是一个**精妙但常被忽视的细节**: 单步gradient descent ≠ 真正的product distribution sampling, 因为Langevin MCMC需要多步才能mix到正确的product distribution。

### 3.6 Stochastic Sampling (SS) — Paper的核心贡献

基于Du et al. "Reduce, Reuse, Recycle" [21] (https://arxiv.org/abs/2302.08458)的annealed ULA (Unadjusted Langevin Algorithm) MCMC。

核心思想: 在每个diffusion timestep $i$做$M$步MCMC sampling, 充分mix到$p_i(\tau) \cdot q(\tau)$, 然后再做一次reverse step到$p_{i-1}(\tau) \cdot q(\tau)$。

Update equation — 公式(6):

$$
\tau_i = \tau_i - \gamma_i \big(\epsilon_\theta(\tau_i, i) + \beta_i \nabla_{\tau_i} \xi(\tau_i, \mathbf{z})\big) + \sigma_i \eta
$$

重复$M-1$次, 然后用公式(5)做final reverse step得到$\tau_{i-1}$。

注意公式(6)和公式(5)的**唯一区别**:
- 公式(5): $\tau_{i-1}$ (新timestep)
- 公式(6): $\tau_i$ (同一timestep, 仅iterate)

**Algorithm 1的四行改动**:
```
Input: diffusion policy π_θ, user interaction z, alignment objective ξ(·)
1: Initialize plan τ_N ∼ N(0, I)
2: for i = N, . . . , 1 :  // denoising steps
3:   for j = 1, . . . , M :  // sampling steps (NEW!)
4:     ϵ ← π_θ(τ_i)  // denoising gradient
5:     δ ← ∇ξ(τ_i, z)  // alignment gradient
6:     if j < M:  // (NEW!)
7:       τ_i ← reverse(τ_i, ϵ + β_i δ, i)  // stay at timestep i (NEW!)
8:     else:
9:       τ_{i−1} ← reverse(τ_i, ϵ+β_i δ, i−1)  // advance to i-1
```

**实现trick**: paper提到为了实现公式(6)在固定noise level的更新, 需要先做reverse sampling得到intermediate clean trajectory prediction $\tilde{\tau}_0$, 然后再做forward diffusion step with noise level $i$回到$\tau_i$。这是DDIM inversion的一个变体。

**计算开销**: $M=4$意味着总denoising calls是$NM = 400$ (vs GD的$N=100$)。Paper在limitation里承认这是expensive sampling procedure, future work要distill成interaction-conditioned policy。

---

## 4. Maze2D实验: 连续Motion Alignment

### 4.1 Setup

- Environment: Maze2D from D4RL [25] (https://arxiv.org/abs/2004.07219)
- Training data: 4M collision-free navigation steps (random walk between random locations)
- **No goal-oriented objective** during training — policy只学collision-free motion manifold
- Models: 
  - ACT [7] (https://arxiv.org/abs/2304.13705): VAE-based action chunking transformer, unimodal
  - DP [6] (https://arxiv.org/abs/2303.04137): DDIM scheduler with $N=100$ training steps
- Inference: 100 random maze locations, 每个配一个可能collision-violating的sketch
- Batch size: 32 trajectories per trial
- DDIM inference steps: 10
- Hyperparams: $\beta_{i \leq N} = 20$ for GD, $\beta_{i \leq N} = 60$ for SS, $M=4$

### 4.2 Table I 数据解析

| Policy | Metric | RS | PR | OP | BI | GD | SS |
|--------|--------|----|----|----|----|----|----|
| ACT | Min L2 ↓ | 0.26 | 0.26 | 0.26 | 1 | - | - |
| ACT | Avg L2 ↓ | 0.26 | 0.26 | 0.26 | - | - | - |
| ACT | Collision ↓ | 0.16 | 0.16 | 0.35 | - | - | - |
| DP | Min L2 ↓ | 0.27 | 0.16 | 0.16 | 0.11 | 0.11 | **0.10** |
| DP | Avg L2 ↓ | 0.28 | 0.28 | 0.28 | 0.14 | 0.18 | **0.12** |
| DP | Collision ↓ | 0.01 | 0.01 | 0.02 | 0.06 | 0.06 | **0.01** |

关键观察:

1. **ACT上所有steering方法都失败** (Min L2 stuck at 0.26, GD/SS甚至无法运行): 因为ACT unimodal, 无法discover新mode, 也没有GD需要的score function。Figure 6显示ACT对input perturbation极其sensitive, 整个manifold脆弱。

2. **DP + PR**: Min L2从0.27→0.16, Collision保持0.01。**几乎免费**的alignment提升, 前提是multimodal policy。

3. **DP + SS最佳trade-off**: Min L2 = 0.10 (vs RS 0.27, 62% reduction), Collision = 0.01 (和RS持平)。这是SS的核心价值——在product distribution下采样, 既能align又能stay in-distribution。

4. **DP + GD**: Min L2 = 0.11 (和SS相近), 但Collision = 0.06 (是SS的6倍!)。验证了Figure 3的论断——GD sample自sum distribution, 容易OOD。

5. **DP + BI**: Min L2 = 0.11, 但Collision = 0.06。BI的initialization偏置被diffusion process部分washed out。

### 4.3 Figure 4的Trade-off可视化

Figure 4展示alignment vs collision的Pareto frontier:
- (1) 任何steering都improve alignment at cost of constraint satisfaction
- (2) Multimodal DP + PR提升alignment而不显著增加collision
- (3) Unimodal ACT难steer, 尤其缺乏robustness时
- (4) DP + SS实现最佳Pareto point

### 4.4 Figure 5的Qualitative Comparison

Figure 5展示两种policy (ACT/DP)在不同steering method下的trajectory visualization:
- Trajectory thickness反映ranking后的sketch similarity
- 白色tint表示collision samples
- SS的trajectory既贴近sketch又无collision (color-coded blue→red over time)
- GD的trajectory有部分白tint (collision)

---

## 5. Block Stacking实验: 离散Task Alignment

### 5.1 Setup

- Environment: Isaac Sim [27] (https://arxiv.org/abs/2301.00568), NVIDIA的robotics simulator
- Motion planner: CuRobo [28] (https://arxiv.org/abs/2310.17274) — NVIDIA的parallel collision-free motion generation library
- Task: 4-block stacking, planner随机pick-and-place, 有时disassemble partial tower重建
- Training: DP with DDIM $N=100$, 5M steps from CuRobo dataset
- Policy学到的是**motion manifold of valid pick-and-place actions**, 不含goal-oriented behavior
- Interaction: VR-based 3D sketch系统, 用户在simulation内画3D sketch

### 5.2 Figure 7的Qualitative Analysis

Figure 7(a): Unconditional sampling from DP产生multimodal trajectory set, 但可能miss user intended plan。

Figure 7(b): PR无法recover intended plan (如果initial batch里没有接近的sample)。

Figure 7(c): GD可以recover, 但产生curved trajectory resembling sketch — 这其实是OOD问题!

Figure 7(d): **Modified GD with $\beta_{i \leq I} = 0$ for $i \leq I$**: 在diffusion的early steps (高noise)用guidance, 后期 (low noise)关闭guidance回归unconditional sampling。结果是从CuRobo training dataset retrieve的straight-line trajectory, 但discrete alignment正确。

这是一个**精妙的insight**: sketch的low-frequency成分 (overall direction)在early diffusion step align, high-frequency细节 (exact path)让policy自己生成, 避免硬贴sketch导致OOD。

### 5.3 Table II 数据解析

| Method (DP) | PR | GD ($\beta_{i<50}=0$) | GD ($\beta_i=100$) |
|------|-----|----------------------|---------------------|
| **TA** (Alignment: AS+AF) | 33% | 83% | 86% |
| **CS** (Success: AS+MS) | 100% | 84% | 15% |
| Aligned Success (AS) | 33% | 67% | 15% |
| Aligned Failure (AF) | 0% | 16% | 71% |
| Misaligned Success (MS) | 67% | 17% | 0% |
| Misaligned Failure (MF) | 0% | 0% | 14% |

四个category定义:
- **AS**: aligned AND successful (理想)
- **AF**: aligned BUT failed (over-steering导致OOD)
- **MS**: misaligned BUT successful (policy自主选了不同但valid的plan)
- **TA = AS + AF** (是否对齐)
- **CS = AS + MS** (是否成功)

关键观察:
1. **PR (33% TA, 100% CS)**: 完全不steer, 等价于让policy自己sample再select。CS满分因为policy always in-distribution。
2. **Modified GD $\beta_{i<50}=0$ (83% TA, 84% CS)**: 最佳trade-off, AS=67%。
3. **Full GD $\beta_i=100$ (86% TA, 15% CS)**: 高alignment但灾难性CS, AF=71%——大部分aligned samples都fail!

**核心结论**: Guide ratio $\beta_i$需要**schedule**, 不能constant。Early steps align低频成分, late steps回归data manifold。这其实是classifier-free guidance的implicit insight的一个变种。

---

## 6. Real World Kitchen实验: 真实场景验证

### 6.1 Setup

- Toy kitchen环境, kinesthetic teaching收集demonstrations
- 两个task: (1) place bowl in microwave, (2) place bowl in sink
- 60 demos/task, 合并成dataset
- DP训练40K steps
- Figure 8展示multimodal skills based on end-effector pose和gripper state
- **关键challenge**: 合并dataset引入**infeasible skill sequences** — 例如 "place bowl in microwave" 之前需要 "open microwave door", 如果skill sequence反了就fail

Figure 9展示一个minimum 6-step的valid skill sequence, steering要选preferred legal sequence直到terminal state。

### 6.2 Table III 数据解析

| Method (DP) | RS | GD | SS | OP |
|------|-----|-----|-----|-----|
| Interaction Type | N/A | Point | Point | Point | Correction |
| **TA** (AS+AF) | - | 38% | 71% | 89% |
| **CS** (AS+MS) | 90% | 82% | 73% | 37% |
| Aligned Success (AS) | - | 34% | **55%** | 30% |
| Aligned Failure (AF) | 4% | 5% | 16% | 59% |
| Misaligned Success (MS) | 56% | 50% | 18% | 7% |
| Misaligned Failure (MF) | 6% | 13% | 11% | 4% |

Hyperparams:
- GD: $\beta_i = 5$ for all $N=100$ steps (weak steering baseline)
- SS: $\beta_i = 100$ (strong steering, 因为SS robust to high $\beta_i$)

关键观察:
1. **RS (90% CS, 4% AF)**: Policy自主rollout, 几乎都成功, 但很多misaligned (56% MS) — policy没align user intent。

2. **OP (89% TA, 37% CS)**: 最aggressive, 几乎都align但灾难性fail, AF=59%。

3. **SS (71% TA, 73% CS, 55% AS)**: **最佳**。比RS的AS (推断为~34% based on alignment rate)提升**21%** without any fine-tuning。

4. **GD $\beta_i=5$ (38% TA)**: 这其实steering不足。Figure 11解释: GD对high $\beta_i$敏感——增大$\beta_i$时GD产生incoherent trajectory, SS反而能识别intended skill。

### 6.3 Figure 11 — Guide Ratio Sensitivity

X轴: $\beta_i$ (guide ratio)
- 小$\beta_i$: GD和SS都ineffective (steering不够)
- 大$\beta_i$: GD开始产生incoherent trajectories (OOD), SS成功识别intended skill

**这印证了GD的理论缺陷**: gradient sum不对应product distribution, 大gradient会把sample拉到OOD region。SS的MCMC sampling在固定noise level iterate, 让distribution mixing充分, 即使大$\beta_i$也能stay in-distribution。

### 6.4 Figure 10 — Alignment-Distribution Shift Trade-off

可视化展示user steering与distribution shift的矛盾。当用户过度steer (OP), robot end-effector被推到OOD state, 后续policy prediction完全失败。

---

## 7. 与Related Work的Position

### 7.1 Policy Composition系列

- **PoCo** [47] (Wang et al., https://arxiv.org/abs/2402.02511): Policy composition from heterogeneous robot learning, gradient-based composition across diverse domains/modalities。ITPS是PoCo的inference-time HRI specialization。
  
- **BESO** [43] (Reuss et al., https://arxiv.org/abs/2304.02532): Goal-conditioned imitation using score-based diffusion, 用classifier-free guidance做goal-conditioning。

- **SE(3) Diffusion Fields** [45] (Urain et al., https://arxiv.org/abs/2305.12734): Learned cost functions生成gradient for joint grasp + motion planning。

- **V-GPS** [46] (Nakamoto et al., https://arxiv.org/abs/2410.13816): 用learned value function re-rank generalist policy output, 类似PR但更sophisticated。

- **Compositional Diffusion** [37, 42] (Liu et al., Yang et al.): Compose multiple diffusion models for structured generation。

### 7.2 Diffusion Steering系列

- **Diffusion for Shared Autonomy** [24] (Yoneda et al., https://arxiv.org/abs/2302.12244): BI的灵感来源, 用noisy user input作为diffusion initialization。

- **Reduce, Reuse, Recycle** [21] (Du et al., https://arxiv.org/abs/2302.08458): SS的理论基础, energy-based diffusion + MCMC实现compositional generation。

- **Planning with Diffusion** [18] (Janner et al., https://arxiv.org/abs/2205.09991): 把diffusion作为planner, classifier-guided steering的robotics应用。

### 7.3 HRI系列

- **Yell at Your Robot** [9] (Shi et al., https://arxiv.org/abs/2403.12910): Language corrections fine-tuning, ITPS的对比方向。

- **RT-Trajectory** [10] (Gu et al., https://arxiv.org/abs/2311.01977): Hindsight trajectory sketches for task generalization, sketch input的灵感。

- **Point-and-Click Interface** [11] (Kemp et al., 2008): 早期的point-and-click robot interface, point input的鼻祖。

- **Counterfactual Perturbations** [12] (Wang et al., https://arxiv.org/abs/2403.17124): 同一作者的prior work on grounding language plans in demos。

---

## 8. 核心Intuition总结

### 8.1 为什么SS比GD好?

**数学层面**: GD的gradient sum $\nabla \log p + \beta \nabla \log q$ 对应score of product distribution $p \cdot q$, 但单步更新无法让sample mix到这个product distribution——只会在两个mode之间"骑墙", 产生intermediate OOD samples。

SS通过$M$步MCMC在固定noise level充分mix, 真正采样到product distribution。当user intent和data mode部分overlap时, SS能找到closest in-distribution mode然后align。

### 8.2 为什么multimodal policy是前提?

PR完全依赖multimodality——batch里必须有多个mode才能select。SS虽然能"discover"新mode, 但也只能在data manifold内interpolate, 无法extrapolate到completely novel behavior。

ACT的unimodal VAE latent space无法提供这种diversity, 所以steering失败。

### 8.3 为什么guide ratio要schedule?

Figure 7(d)的modified GD给了insight: 
- Early diffusion steps (高noise): 学习trajectory的low-frequency成分 (overall direction, target object), 此时用guidance align user intent
- Late diffusion steps (low noise): 精refine high-frequency细节 (exact path, smooth motion), 此时回归policy自身分布避免OOD

这和classifier-free guidance的训练-推理不对称有点像——guidance在coarse level有效, fine level会破坏sample quality。

### 8.4 Open Question

Paper的limitation: SS的$M=4$让inference慢4x。Future work: distill steering过程成interaction-conditioned policy, 让distilled model一次性output aligned trajectory, 无需MCMC迭代。

这与**flow matching** [20] (Ye & Gombolay, https://arxiv.org/abs/2403.10809)和**consistency models**方向相关——用OT-based或一致性蒸馏加速sampling。

---

## 9. 实用Takeaway for Practitioners

1. **Multimodal policy是steering的必要条件**: 用diffusion policy而非VAE/Gaussian policy
2. **PR是免费lunch**: 如果policy已经multimodal, PR几乎零成本提升alignment
3. **SS是strong steering的最佳选择**: 高$\beta_i$下SS仍robust, GD会崩
4. **Guide ratio要schedule**: Early steps强steering, late steps弱steering
5. **避免OP**: OP的hard override最容易导致OOD failure
6. **CS比TA更重要**: aligned failure比misaligned success更糟——前者robot损坏, 后者user重新干预即可

---

## 10. Reference Links汇总

**核心paper**:
- ITPS Project Page: https://yanweiw.github.io/itps/
- Diffusion Policy [6]: https://arxiv.org/abs/2303.04137
- ACT [7]: https://arxiv.org/abs/2304.13705
- Reduce Reuse Recycle (SS理论基础) [21]: https://arxiv.org/abs/2302.08458
- Planning with Diffusion [18]: https://arxiv.org/abs/2205.09991
- To the noise and back (BI灵感) [24]: https://arxiv.org/abs/2302.12244
- DAgger [13]: https://arxiv.org/abs/1011.0686
- D4RL [25]: https://arxiv.org/abs/2004.07219
- DDIM [26]: https://arxiv.org/abs/2010.02592
- Isaac Sim/Orbit [27]: https://arxiv.org/abs/2301.00568
- CuRobo [28]: https://arxiv.org/abs/2310.17274

**Related work**:
- PoCo [47]: https://arxiv.org/abs/2402.02511
- BESO [43]: https://arxiv.org/abs/2304.02532
- SE(3) Diffusion Fields [45]: https://arxiv.org/abs/2305.12734
- V-GPS [46]: https://arxiv.org/abs/2410.13816
- Compositional Diffusion [37]: https://arxiv.org/abs/2206.01714
- Yell at Your Robot [9]: https://arxiv.org/abs/2403.12910
- RT-Trajectory [10]: https://arxiv.org/abs/2311.01977
- Counterfactual Perturbations [12]: https://arxiv.org/abs/2403.17124
- OpenVLA [4]: https://arxiv.org/abs/2406.09246
- Octo [3]: https://arxiv.org/abs/2405.12213
- Open X-Embodiment [2]: https://arxiv.org/abs/2310.08864
- Mobile ALOHA [8]: https://arxiv.org/abs/2401.02117
- Latent Actions [36]: https://arxiv.org/abs/2403.03181
- Efficient Trajectory Forecasting [20]: https://arxiv.org/abs/2403.10809
- Energy-based Models as Zero-shot Planners [41]: https://arxiv.org/abs/2304.14391
- Generative Skill Chaining [44]: https://proceedings.mlr.press/v229/mishra23a.html
- Temporal Logic Imitation [17]: https://arxiv.org/abs/2206.04632

**作者主页**:
- Yanwei Wang: https://yanweiw.github.io/
- Yilun Du: https://yilundu.github.io/
- Julie Shah: https://biomechatronics.cit.mit.edu/
- NVIDIA Robotics: https://developer.nvidia.com/isaac-sim

这篇paper精妙之处在于把HRI的intervention问题formulate成compositional sampling问题, 通过SS的product distribution sampling elegantly解决alignment-distribution shift trade-off。从engineering角度, 只需改diffusion policy的4行代码就能实现strong steering——这种简洁的algorithmic contribution非常优雅。
