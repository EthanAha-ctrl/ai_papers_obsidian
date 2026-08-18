---
source_pdf: Diffusion-Based Approximate MPC.pdf
paper_sha256: bc58a7ee80db1e1878617d15f06c14c8115290af7ecc4551122264e79f8a2245
processed_at: '2026-08-18T05:46:13-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Diffusion-Based AMPC 人话版

好，我换个节奏，用更直觉的方式重新讲一遍，讲讲背后的"为什么"。

## 故事的开头: MPC 又强又慢这个老问题

MPC 本质是个"每一步都重新做一次小优化"的控制方法。你有一个机器人，当前 joint 状态 $x_t$（7 个 joint 角度），有个目标 end-effector pose $y_t^d$（位置 + 朝向），然后你求解一个 optimization problem: 未来 N=20 步、每步 0.1 秒、总共 2 秒的 horizon，找一组 joint velocities $u_{\cdot|t}$ 让 cost 最小（tracking error 小 + control effort 小），同时满足 joint limits、self-collision、obstacle avoidance 这些约束。

MPC 的魅力在于: 每一步都重新解，所以它自然处理 constraint、自然 robust、有 stability guarantee。但代价是你要在 10-20 ms 内解一个 nonlinear program，对 KUKA 7-DOF arm 用 IPOPT 大概要 47 ms mean / 65 ms 95th percentile，控制频率只能到 ~10-20 Hz。10 Hz 对做快速 tracking 是不够的，机械臂会 overshoot、会震荡。

AMPC (Approximate MPC) 的 idea 很朴素: 我把 MPC 当 teacher，offline 跑一大堆 $(x_t, y_t^d) \to u^*_{\text{MPC}}$ 的 pair，训练一个神经网络当 student，online 的时候 forward network 就行，微秒级。这个 idea 从 1995 年 [Parisini & Zoppoli](https://www.sciencedirect.com/science/article/abs/pii/S0005109894100094) 就有了，[Nubert et al. 2020](https://arxiv.org/abs/1909.09142) 在 KUKA 上做得挺成功。

## 真正的坑: multi-modality

大部分人讲 AMPC 故事到这里就结束了，但这篇 paper 戳中了一个真实存在但容易被忽略的问题: **MPC 的解不是唯一的，而且常常是 multi-modal 的**。

为什么会 multi-modal? 几个来源:

1. **冗余 DOF**: 7-DOF arm 跟踪一个 6-DOF end-effector pose，有 1 个 nullspace DOF。elbow-up / elbow-down / elbow-sideways 都能到达同一个 pose，每个都是 global optimum。这是 set-valued solution。
2. **非凸 constraints**: workspace 中间有个 obstacle，机器人可以绕左边也可以绕右边，两条 trajectory 都是 locally optimal。
3. **数值 solver 的 local optima**: IPOPT 这种 local solver，不同的 initial guess $\xi^{\text{init}}$ 会落到不同的 local minimum。

所以给定一个 $(x_t, y_t^d)$，MPC expert 给你的不是一个 deterministic $u^*$，是一个 **distribution** $Q_s(\cdot | x_t, y_t^d)$。

传统的 L2 regression 做什么? 它学 conditional mean:
$$
\mathcal{L} = \mathbb{E}_{u^* \sim Q_s} \| u^* - f_\theta(x_t, y_t^d) \|^2
$$

mean 在 unimodal distribution 下是最优预测，但在 multi-modal 下是**两个 mode 的平均**，这个平均既不是 mode A 也不是 mode B，很可能根本不是一个 feasible trajectory。Fig. 2 右边那个 LSM heatmap 看得很清楚: 它学到了一个夹在两个真实 mode 中间的"鬼影"分布，这个鬼影对应的 joint command 让机械臂根本到不了目标。

[Li et al. 2022](https://www.sciencedirect.com/science/article/pii/S0005109822001543) 这篇 paper 直接给出结论: 做 AMPC 必须 avoid multi-modality。怎么 avoid? 限制 robot 到 6-DOF (去掉冗余)、限制 workspace (去掉 obstacle)、用 fixed init guess (强制 deterministic)。这些 work-around 都很 hacky，限制 applicability。

## 为什么选 Diffusion?

Diffusion model 本质是个 **non-parametric conditional density estimator**。给它一堆 samples from $Q_s(\cdot | x_t, y_t^d)$，它能学到任意形状的 distribution，不管你有 2 个 mode 还是 10 个 mode。

对比其他选择:

**Mixture Density Network (MDN)** [Bishop 1994](https://publications.aston.ac.uk/id/eprint/373/1/CRG_PSTR_NCRG_1994.pdf): 用 $K$ 个 Gaussian 的 mixture。问题是你得预先指定 $K$，但 MPC 在不同 $(x_t, y_t^d)$ 下 mode 数量不一样 (workspace 中间有 obstacle 时可能是 bimodal，无 obstacle 时可能 unimodal，redundant DOF 在不同 pose 下 IK 解的数量也不同)。$K$ 不好选。

**Normalizing Flow**: 也可以，但训练麻烦，architecturally 更复杂。

**Diffusion**: 一个 model 搞定任意 $K$，训练简单 (就是个 denoising loss)，sampling 灵活 (可以加 guidance)。这就是 paper 选 diffusion 的核心 reason。

## Diffusion 的极简回忆

如果你忘了 DDPM 怎么回事，这里有个 30 秒回顾:

**Forward**: 给 clean data $x_0$ 逐步加 noise，加 $N$ 步之后变成 pure noise $x_N \sim \mathcal{N}(0, I)$。
$$
q(x_i | x_{i-1}) = \mathcal{N}(x_i; \sqrt{1-\beta_i} x_{i-1}, \beta_i I)
$$
这里 $\beta_i$ 是第 $i$ 步加的 noise variance，$\bar{\alpha}_i = \prod_{j \leq i}(1-\beta_j)$ 是 cumulative signal retention。

**Reverse**: 训练一个网络 $f_\theta$ 学怎么从 $x_i$ 预测 $x_0$ (或者预测被加的 noise $\epsilon$)。Inference 时从 $x_N \sim \mathcal{N}(0, I)$ 开始，逐步去噪，得到 $x_0 \sim p_{\text{data}}$。

这里 $x_0$ 在我们场景下就是 joint velocity trajectory $u_{\cdot|t} \in \mathbb{R}^{7 \times N}$，条件是 $(x_t, y_t^d)$。

paper 用 5 个 denoising steps，这是个激进选择。原版 DDPM 用 1000 步，Diffusion Policy 用 20-100 步，Consistency Policy 蒸馏到 1-3 步。这里直接训 5 步、用 [cosine schedule](https://arxiv.org/abs/2102.09772) + [Min-SNR weighting](https://arxiv.org/abs/2303.09551) + [最后一步 0 SNR rescaling](https://openaccess.thecvf.com/content/WACV2024/html/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.html)，省掉了 distillation 麻烦。

## 第一个坑: closed-loop mode swapping

训练完 diffusion，sample 一个 trajectory 看着挺好，多模态都 capture 到了。但你把它放到 closed loop 里跑，问题就来了。

每个 control step 你 sample 一次，因为 diffusion 是 stochastic 的，每一步可能 sample 到不同 mode。step $t$ 你 sample 到 elbow-up，step $t+1$ 你 sample 到 elbow-down，step $t+2$ 又回到 elbow-up... 机械臂就在 mode 之间疯狂切换，jerk 飙到几千 rad/s³，safety stop 触发，机器人停。

这在 MPC 里不是问题，因为 IPOPT warm-start from previous solution，自然 stay in same basin of attraction。但 vanilla diffusion 没有"记忆"，每一步独立 sample。

paper 的 fix 是 **Gradient Guidance (∇G)**: 用上一步的 solution $x_{0|t-1}$ 当 anchor，通过 Bayes rule 把当前 sample 拉向 anchor。数学上，diffusion reverse step 的 score function 是 $\nabla \log p(x_i)$，加 guidance 变成 $\nabla \log p(x_i) + \nabla \log p(o | x_i)$，其中 $o$ 是 prior info (上一步 solution)。具体形式是加一个 Gaussian " attraction potential" toward $x_{0|t-1}$。

这个 trick 的理论基础是 [Vincent 2011](https://direct.mit.edu/neco/article-lookup/doi/10.1162/NECO_a_00142) 把 diffusion 和 score matching 联系起来，以及 [Dhariwal & Nichol 2021](https://arxiv.org/abs/2105.05233) 的 classifier guidance。

效果: Tab. III 里，mean mode swaps per episode 从 3.00 降到 0.08，甚至比 MPC 的 0.15 还低。这是 paper 最 critical 的 contribution，没有这个根本 deploy 不了 hardware。

## 第二个坑: jerk from noise injection

∇G 解决了 mode swapping，但 jerk 还是太大。原因是 reverse process 的最后几步还在 inject noise $\tilde{\beta}_i \epsilon$，这个 noise 在最后 refine 阶段让 $x_0$ 抖动。

paper 的 fix 是 **Early Stopping (ES)**: 最后 25% 的 denoising steps (即 $i < i^{\epsilon}_{\min} = \lfloor 0.75 N_I \rfloor$) 不加 noise，只走 mean 项。

$$
\tilde{\beta}'_i = \tilde{\beta}_i \cdot \mathbf{1}_{i > i^{\epsilon}_{\min}}
$$

直觉: 早期 denoising step (高 noise level) 加 noise 是好事，让模型 explore 多模态；后期 (低 noise level) 加 noise 是坏事，让 final sample 抖。

效果: median jerk norm 从 2403 rad/s³ 降到 57.76 rad/s³，**40× reduction**。这个 trick 单看很简单，但效果惊人，我觉得可以推广到所有 diffusion-for-control 的工作。

## 第三个 trick: parallel sampling 选最好的

GPU 上一次 forward 可以并行 sample 100 个 candidates，几乎没 extra cost (GPU 上 1.3-4.6 ms vs 单 sample 0.85 ms)。然后从中选最好的:

- **Cost ranking**: 用 MPC cost function 给每个 candidate 打分，选 cost 最低的。需要 forward kinematics 计算 tracking error，CPU 上做。
- **Safe sampling**: 100 个 sample 里 filter 掉 colliding 的，从剩下的随机挑。
- **Clustering**: 没有 cost function 知识时，按 Euclidean distance 聚类，选 density 最大的 cluster 的 representative (democratic voting)。

效果: 1% tolerance SR 从 88.5% 提到 95.1% (safe sampling)，96.09% (clustering)，99.20% (cost ranking)。

这个 idea 是把 diffusion 当 generative model 用，你不仅有 single point estimate，你有 posterior distribution 的 samples，可以 reject bad samples。

## 实验里的几个反直觉发现

**1. DM success rate 比 teacher MPC 还高** (Tab. I, II): simulation 96.33% vs MPC 93.18%, hardware 93% vs 87.8%。原因:
- 数据收集时 filter 掉了 5% 不收敛的 MPC solution (filtering criterion: $d_y > 0.01$)，DM 学不到这些 bad modes
- DM 在 250 Hz 跑，MPC 在 10 Hz 跑，DM 的 fast update 让 transient 更 smooth、overshoot 更少
- DM 在 multi-modal 数据上 trained，学到的是 weighted ensemble of modes

类似 RLHF 中 reward model 偶尔超 human annotator 的现象，但更系统化。

**2. DM steady-state error 比 MPC 大** (5-6 mm vs 0.5 mm): 因为数据收集时加了 $\sigma_{\min} = 0.35$ rad/s 的 exploration noise，DM 学到的 distribution 在 steady state 附近也有 spread，sample 出来的 $u$ 不正好是 0。这是 trade-off: 你要 multi-modal coverage 就得加 exploration noise，但加多了 steady-state 精度就降。未来工作可能需要 noise scheduling based on distance to target。

**3. 5 个 denoising steps 几乎不掉点**: 5 steps ATE 5.47 mm, SR 96.33%; 40 steps ATE 5.14 mm, SR 96.18%。这说明对于 1D control signal (joint velocities)，不需要 image generation 那种 1000 步的精细去噪，5 步就够 capture distribution shape。

**4. GPU 并行 100 个 sample 几乎免费**: 单 sample 0.85 ms，100 个 sample 1.3-4.6 ms。GPU 上的 batching overhead 几乎可以忽略。这让 cost ranking / safe sampling 变得 practically free。

## 几个 architecture 细节值得注意

**Plain MLP，不是 U-Net**: 7-layer, 1000 neurons/layer, 6.6M params。joint velocity 是 1D vector signal，没有 spatial structure，U-Net 的 inductive bias 没用。

**6D rotation representation**: target pose $y_t^d$ 的 rotation 用 [Zhou et al. 2019](https://arxiv.org/abs/1812.07035) 的 6D representation (两列 rotation matrix)，不用 quaternion 避免 discontinuity，不用 Euler 避免 gimbal lock。

**Virtual trajectory formulation** [Kohler et al. 2020](https://www.sciencedirect.com/science/article/pii/S0005109820301939): MPC 里引入 virtual trajectory $x^s_{\cdot|t}$ 当 "moving anchor"，real trajectory $x_{\cdot|t}$ tracking virtual trajectory，virtual trajectory 收敛到 target。这让 MPC 的 tracking error 项是 $\|x_{N|t} - x^s_{N|t}\|$ 而不是 $\|x_{N|t} - x^d\|$，避免了 target 在 workspace 边界时的 infeasibility 问题。

**CoClusterBridge**: 数据收集用 multi-process 把 Isaac Lab vectorized sim 和 CasADi/IPOPT solver cluster 桥接起来，能跑出 55.5M predictions 的数据集。这个工程量不小。

## 我觉得 paper 没说清楚的地方

1. **Gradient guidance 的 strength**: Eq. 13 里 guidance term 的权重是隐式的 (因为 Gaussian log-likelihood 的 coefficient 是 $1/(1-\bar{\alpha}_i)$，跟 noise schedule 耦合)。paper 没讨论这个 strength 怎么 tune，跟 classifier guidance 的 scale $\gamma$ 怎么对应。

2. **Stability guarantee 完全缺失**: MPC 的核心 selling point 是 Lyapunov stability，AMPC 一般会通过 constrained learning [Hose et al. 2025](https://arxiv.org/abs/2312.06225) 或 ISS Lyapunov certificate 保留某种 stability。这篇 paper 完全没提，DM 本质是 black-box，没法 certify。这是 diffusion-based control 普遍的 open problem。

3. **Out-of-distribution behavior**: 当 $x_t$ 在 training distribution 之外 (e.g. joint 突然被外力推到 weird pose)，DM 行为不可预测。MPC 至少还能 re-plan，只是慢。

4. **Multi-modal 的数量随环境变化**: paper 里场景相对简单 (最多 elbow-up/down + 绕左/绕右)，mode 数量有限。更复杂场景 (e.g. 多个 obstacle、双手协调) mode 数量会爆炸，diffusion 还能 handle 吗?

5. **Training data 成本**: 55M predictions 用 IPOPT + multi-process cluster，paper 没说采集多久，但估计是几天到一周。这个对快速 iteration 不友好。

## 我会怎么 extend 这工作

**Direction 1: Consistency Policy 替代 DDPM**: [Prasad et al. 2024](https://arxiv.org/abs/2405.0784) 已经 distill 到 1-3 步，搭配 ES 应该能跑 kHz。但 distillation 需要 teacher 模型，多一道工序。

**Direction 2: Score-based SDE + control Lyapunov**: 用 [Song et al. 2021](https://arxiv.org/abs/2011.13456) 的 SDE formulation，把 reverse process 当 stochastic dynamical system，尝试找 Lyapunov function certify stability。这个理论上 hard 但价值大。

**Direction 3: Diffusion warm-start + MPC refinement**: 类似 [DiffuSolve](https://arxiv.org/abs/2407.07455)，用 diffusion 给 IPOPT 一个好 init，IPOPT 跑几步 refine。这样保留 MPC 的 safety guarantee，同时大幅 speedup。可能比纯 diffusion deployable 到更 critical 的场景。

**Direction 4: Goal-conditioned RL with diffusion policy + MPC teacher**: 把这套 framework 推到更 general 的 RL setting，MPC 当 demo source，diffusion 当 policy class，gradient guidance 用 task-specific reward 当 likelihood。

**Direction 5: Online adaptation**: 当 obstacle 移动时，conditional distribution 会 time-varying。把 DM 做成 online-adaptive，用最近几步的 MPC solution 做 few-shot finetune，可能比 re-train 整个 model 实用。

## 最后的 takeaway

如果用一句话讲这篇 paper 的贡献: **三个 inference-time tricks (∇G + ES + parallel sampling) 让 diffusion 从 generative model 变成 closed-loop controller**。每个 trick 单看都不复杂，但组合起来解决了 multi-modal learning、closed-loop consistency、smoothness、constraint satisfaction 四个问题，才能把 rate 从 10 Hz 推到 250 Hz。

数学上最 elegant 的部分是 gradient guidance 通过 Bayes rule 把 prior information 嵌入 reverse process，这个 framework 应该可以推广到其他 conditional generation 场景。工程上最 impressive 的部分是 5 个 denoising step + GPU 并行让 inference 跑到 0.85 ms，比 MPC 快 70× 还反超 success rate。

未来的关键 question 是 stability guarantee 怎么补，否则 diffusion-based control 永远只能在 non-safety-critical 应用里用。这也是我觉得最值得 follow up 的方向。

希望这个版本更 readable，Andrej 你看哪部分还需要展开。

---

# Diffusion-Based Approximate MPC 深度解析

很高兴和 Andrej 一起深入解读这篇 2024/2025 的 ICRA/RAL 风格工作。这篇 paper 把 Diffusion Model 从 high-level planning 拉到了 250 Hz 的 low-level joint velocity control 上，并且在 7-DOF 机械臂上做到了 70× 的 speedup，甚至 success ratio 反超 teacher MPC。我尽量把里面的数学细节、设计直觉、实验数字都摊开讲。

## 1. Why this paper matters: 问题动机

### 1.1 MPC 的痛点

MPC (Model Predictive Control) 是一种 receding horizon 的优化控制方法，每个 control step 都要求解：

$$
\min_{u_{\cdot|t}, x_{\cdot|t}} J_N(u_{\cdot|t}; x_t, y_t^d)
$$

subject to dynamics $x_{k+1|t} = f(x_{k|t}, u_{k|t})$ 和 constraints $g_j(x_{k|t}, u_{k|t}) \le 0$。

其中变量：
- $x_t \in \mathcal{X} \subseteq \mathbb{R}^n$: current state (e.g. 7 个 joint positions)
- $u_{\cdot|t} \in \mathcal{U}^N$: 控制输入序列，整个 horizon 的 N 步
- $y_t^d \in \mathcal{Y}$: desired output (end-effector pose in SE(3))
- $N$: prediction horizon
- $k \in [0, ..., N-1]$: time index within horizon
- $j$: constraint index

对于 KUKA LBR4+ 这种 7-DOF arm，他们用 IPOPT (interior point method) 在 CPU 上求解，每个 step 大约 **47.54 ms (mean)，65.63 ms (95th percentile)**，对应控制频率只能做到 ~10-20 Hz。这个 rate 对于做 dynamic trajectory tracking（比如快速 SE(3) tracking）是远远不够的。

### 1.2 AMPC (Approximate MPC) 的思路

AMPC 的核心思想: 用 IL (Imitation Learning) 去拟合 MPC 的 policy $\pi_{\text{MPC}}(x_t)$，online 的时候 forward 一个 neural network 就够了。

经典做法 [Parisini & Zoppoli 1995, Nubert et al. 2020]: 训练一个 MLP with L2 (least squares) regression:

$$
\mathcal{L}_{LS} = \| u^*_{\text{MPC}}(x_t, y_t^d) - f_\theta(x_t, y_t^d) \|_2^2
$$

### 1.3 关键问题: multi-modality

这就是这篇 paper 要解决的真正问题。非线性 MPC 因为 non-convex constraints (obstacles)、redundant DOF、numerical solver 的 local optima，会产生 **set-valued / multi-modal solution distributions**。形式化地，给定 $x_t$ 和 $y_t^d$，solver $s$ 找到的是：

$$
Q_s = \{ s(\xi^{\text{init}}, x_t, y^d) \mid \xi^{\text{init}} \sim P(\Xi) \}
$$

这里 $\xi^{\text{init}}$ 是 solver initialization，不同 init 会落到不同 local optima，甚至不同 global optima（redundant DOF 的情况下，比如 elbow-up vs elbow-down 都能到达同一个 end-effector pose）。

L2 regression 在 multi-modal distribution 下的失败模式非常经典: **LS 学到的是 conditional mean**，而 conditional mean 在 set-valued distribution 下根本就不是一个 feasible solution！这个观察 [Bishop 1994, Li et al. 2022] 在 robotics 场景下被反复印证。

Fig. 2 的 heatmap 非常直观: 
- 左边: 真实 MPC 的 open-loop trajectory 在 x-y 平面和 x-z 平面上的 density 是 multi-modal 的（多个 cluster）
- 中间: DDPM 学到了这些 modes
- 右边: LSM (least squares model) collapse 到 dominant mode，而且这个 mode 经常不是 target-reachable 的 mode

## 2. Method: Diffusion-based AMPC (DAMPC)

### 2.1 DDPM 数学回顾

DDPM [Ho et al. 2020, https://arxiv.org/abs/2006.11239] 的 forward process 给数据加 noise:

$$
q(x_i | x_{i-1}) := \mathcal{N}(x_i; \sqrt{1-\beta_i} x_{i-1}, \beta_i \mathbf{I})
$$

变量解释:
- $x_i$: 第 $i$ 步 noisy version of $x_0$
- $\beta_i$: 第 $i$ 步的 noise variance schedule (small)
- $\bar{\alpha}_i := \prod_{j=1}^{i} (1-\beta_j)$: cumulative product

closed form:
$$
q(x_i | x_0) = \mathcal{N}(x_i; \sqrt{\bar{\alpha}_i} x_0, (1-\bar{\alpha}_i)\mathbf{I})
$$

训练时学一个 network $f_\theta^{x_0}(x_i, i)$ 直接预测 clean sample $x_0$，loss 是 ELBO 简化版:

$$
\mathcal{L}_{\text{ELBO}}^{x_0} := \| x_0 - f_\theta^{x_0}(\sqrt{\bar{\alpha}_i} x_0 + \sqrt{1-\bar{\alpha}_i} \epsilon_i, i) \|^2
$$

这里的 $\epsilon_i \sim \mathcal{N}(0, \mathbf{I})$ 是 sampled noise。注意等价的另一形式是预测 $\epsilon$ 而不是 $x_0$。

Reverse process:
$$
x_{i-1} := \frac{\sqrt{\bar{\alpha}_{i-1}} \beta_i}{1-\bar{\alpha}_i} f_\theta^{x_0}(x_i, i) + \frac{\sqrt{\alpha_i}(1-\bar{\alpha}_{i-1})}{1-\bar{\alpha}_i} x_i + \tilde{\beta}_i \epsilon
$$

变量:
- $\tilde{\beta}_i := \frac{1-\bar{\alpha}_{i-1}}{1-\bar{\alpha}_i} \beta_i$: posterior variance
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 采样 noise

### 2.2 关键转换: predict $\epsilon$ vs predict $x_0$

$x_0$ 和 $\epsilon$ 之间有解析关系:

$$
x_0 = \frac{x_i - \sqrt{1-\bar{\alpha}_i} \epsilon}{\sqrt{\bar{\alpha}_i}}
$$

这个等式重要，因为它把 DDPM 和 **score matching** 联系起来 [Vincent 2011, https://direct.mit.edu/neco/article-lookup/doi/10.1162/NECO_a_00142]:

$$
\nabla_x \log p(x) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

这个 connection 是 gradient guidance 的理论基础。

### 2.3 数据收集策略 (Sec. IV-A)

这部分有重要的 imitation learning 细节。他们用 DAgger-like 的 noise injection 来 mitigate covariate shift [Spencer et al. 2021, https://arxiv.org/abs/2102.02872]:

$$
\pi'_{\text{MPC}}(x_t) := \pi_{\text{MPC}}(x_t) + \sigma_{u_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

其中
$$
\sigma_{u_t} := \max\left(\frac{|\pi_{\text{MPC}}(x_t)|}{\text{SNR}^d}, \sigma_{\min}\right)
$$

变量:
- $\text{SNR}^d = 0.8$: desired signal-to-noise ratio
- $\sigma_{\min} = 0.35$ rad/s: minimum noise floor

这个 $\sigma_{\min}$ 保证在接近 steady state (where $|\pi_{\text{MPC}}|$ 很小) 时仍然有 exploration，否则模型在 steady state 附近会 overfit 到一个 deterministic point。这个 trick 我觉得挺巧妙。

数据集规模: **55.5M / 33.9M open-loop MPC predictions**，$N_d = 80$ steps per episode。这是相当大的数据集，作者也在 conclusion 里承认 training data 太多是问题。

### 2.4 MPC formulation 细节 (Sec. V-A)

他们用了一种 **virtual trajectory formulation** [Kohler et al. 2020, https://www.sciencedirect.com/science/article/pii/S0005109820301939]:

state $x_t \in \mathbb{R}^7$ (joint positions), input $u_t \in \mathbb{R}^7$ (joint velocities)。

引入 **virtual trajectory** $x^s_{\cdot|t} \in \mathcal{X}^{N+1}$ 和 $u^s_{\cdot|t} \in \mathcal{U}^{N+1}$，加上 terminal constraint $f(x^s_{N|t}, u^s_{N|t}) = x^s_{N|t}$ (即 virtual trajectory 在终点是 equilibrium)。

cost function:

$$
J_N = \| x_{N|t} - x^s_{N|t} \|_\mathbf{P} + d_y(y^d_t, h(x^s_{N|t})) + \sum_{k=0}^{N-1} l(x_{k|t}, u_{k|t}, x^s_{N|t}, u^s_{N|t})
$$

其中 stage cost 是 quadratic:
$$
l(x, u, x^d, u^d) = \|x - x^d\|_\mathbf{Q}^2 + \|u - u^d\|_\mathbf{R}^2
$$

输出误差 $d_y$ 是 SE(3) 上的:
$$
d_y = w_p \|y_p - h(x^s)_p\|_2^2 + w_R \|(\log(y_R^{-1} h(x^s)_R))_\vee\|_2^2
$$

变量:
- $y_p, y_R$: SE(3) element 的 position 和 rotation 部分
- $\log(\cdot)$: Lie group SE(3) → Lie algebra $\mathfrak{se}(3)$ 的映射
- $[\cdot]_\vee$: $\mathfrak{se}(3) \to \mathbb{R}^6$ 的 vee operator
- $w_p, w_R$: weights

obstacle avoidance constraint (ellipsoidal):
$$
\| (h(x^i_{k|t})_p - o_j^p) \odot o_j^s \|_2 \ge 1
$$

变量:
- $o_j^p \in \mathbb{R}^3$: obstacle position
- $o_j^s \in \mathbb{R}^3$: ellipsoid scaling (per-axis)
- $\odot$: element-wise (Hadamard) product

这是 sphere / ellipsoid obstacle 的 standard formulation。

## 3. 三个核心 inference-time 改进

这部分是 paper 的精髓。naive DDPM 拿到 multi-modal distribution 之后，直接 sample 在 closed loop 下会有什么问题? Mode swapping - 下一个 timestep 可能跳到另一个 mode，导致 jerky motion，jerk 大到无法部署到 hardware。

### 3.1 Gradient Guidance (∇G) for closed-loop consistency

inspired by classifier-free guidance [Dhariwal & Nichol 2021, https://arxiv.org/abs/2105.05233] 和 motion planning diffusion [Carvalho et al. 2023, https://arxiv.org/abs/2310.06122]。

Bayes rule 在 reverse process 上:
$$
p(x_{i-1|t} | x_{i|t}, i, o) \propto p(x_{i-1|t} | x_{i|t}, i) \cdot p(o | x_{i|t}, i)
$$

取 log 和 gradient:
$$
\nabla_x \log p(x_{i-1|t} | x_{i|t}, i, o) = \nabla_x \log p(x_{i-1|t} | x_{i|t}, i) + \nabla_x \log p(o | x_{i|t}, i)
$$

这里的 $o$ 是 prior information，他们用 Gaussian 来 force 当前 sample 靠近 previous step 的 solution $x_{0|t-1}$:

$$
p(o | x_{i|t}, i) = \mathcal{N}(x_{i|t}; x_{0|t-1}, (1-\bar{\alpha}_i) \mathbf{I})
$$

把这个 guidance 加到 $\epsilon$-prediction 上:

$$
f_\theta^\epsilon(x_{i|t}, i, o) := f_\theta^\epsilon(x_{i|t}, i) - \nabla_{x_{i|t}} \log \mathcal{N}(x_{i|t}; x_{0|t-1}, (1-\bar{\alpha}_i) \mathbf{I})
$$

直觉: 把上一个 timestep 的 solution 作为 soft anchor，current sample 偏离 anchor 时被拉回来，这模拟了 MPC solver 的 warm-starting behavior。MPC solver 在 closed loop 中之所以 mode-consistent，是因为 warm-start 总是从 previous solution 出发，不会跳到另一个 local minimum。

从 Tab. III 看，效果惊人:
- Vanilla DDPM (no G): mean mode swaps per episode = 3.00
- DDPM with G: 0.08
- 这比 MPC 本身 (0.15) 还低，比 LSM (0.00，但 LSM 是 deterministic) 接近

### 3.2 Early Stopping (ES) of noise injection

reverse process 公式里 $\tilde{\beta}_i \epsilon$ 这一项是 stochasticity 来源。低 SNR step (即 $i$ 大的时候) 加 noise 让模型 explore multi-modality，是好事；但 high SNR step (即 $i$ 小，接近 $x_0$ 时) 还加 noise 会让 final sample 抖动。

修改:
$$
\tilde{\beta}'_i := \tilde{\beta}_i \cdot \mathbf{1}_{i > i^{\epsilon}_{\min}}
$$

其中 $i^{\epsilon}_{\min} = \lfloor 0.75 \cdot N_I \rfloor$。

也就是说，最后 25% 的 denoising steps 完全 deterministic (除了 mean 项)。Tab. III 的 jerk 数据:
- DDPM 5 steps with G, no ES: median jerk norm = 2403.78 rad/s³
- DDPM 5 steps with G and ES: 57.76 rad/s³
- **40× 的 jerk reduction!**

直觉: 多模态 explore 阶段需要 stochasticity，但 commit 到某个 mode 之后，refine 阶段需要 smoothness。这是个挺通用的 trick，应该可以推广到其他 diffusion-for-control 的工作。

### 3.3 Informative sampling

一次 forward 可以 GPU 并行 sample L 个 candidates $\{u_l^*\}_{l=1}^L$，然后 select 最好的。三种 ranking 方法:

**a) Cost-based (full state knowledge)**: 直接用 MPC 的 cost function $d_y(y_t^d, h(x_{N|t}))$ 给每个 candidate 打分。需要在 CPU 上跑 forward kinematics，但效果最好 (Tab. V)。

**b) Feasibility-based (DDPM-Safe)**: 从 L=100 个 samples 里筛掉 colliding 的，从 non-colliding subset 里随机挑一个。Tab. VI 显示 1% tolerance 下 simulation SR 从 88.5% → 95.1%，hardware 从 81.25% → 85%。

**c) Democratic voting via clustering (no full state knowledge)**: 用 Euclidean distance 在 command space 聚类，选 density 最大的 cluster 的 representative。$P(C_k) \approx |C_k|/L$。

这个 idea 是把 diffusion 当 generative model 用: 你不仅有一个 sample，你有整个 conditional distribution 的 samples。这可以 reject 不好的 local minima 或 unfaithful denoising。

## 4. Experimental results 深度解读

### 4.1 Tracking performance (Tab. I, II)

Simulation (8750 random start-goal):
- MPC: ATE 0.54 mm, SR 93.18%, TRT 1.18 s (gold standard，但慢)
- LSM: ATE 13.84 mm, SR 10.26% (失败！LSM 在 multi-modal 下基本不可用)
- DDPM (5 steps, ∇G, ES): ATE 5.47 mm, SR 96.33% (反超 MPC!)
- DDPM (40 steps): ATE 5.14 mm, SR 96.18% (5 steps 几乎没掉点)

Hardware (100 trials, KUKA LBR4+, 250 Hz):
- MPC (10 Hz): ATE 3.71 mm, SR 87.80% (因为 update rate 慢，overshoot 严重)
- LSM: ATE 15.77 mm, SR 12.90% (基本不可用)
- DDPM (5 steps, ∇G, ES): ATE 6.16 mm, SR 93.00%

DDPM 比 MPC steady-state error 大 (5-6 mm vs 0.5 mm)，这部分归因于:
- 数据采集加了 $\sigma_{\min} = 0.35$ rad/s 的 noise，让模型在 steady state 附近也 "shaky"
- 训练数据更多在 dynamic region

但 SR 反超 MPC 因为 MPC 偶尔不收敛 (warm-start 失败，IPOPT numerical issues)，这些 failing cases 被从 training dataset 里 filter 掉了，所以 DM 学不到这些 bad modes。

### 4.2 Hardware 上的关键发现

Tab. II 显示 vanilla DDPM (no ∇G) 在 hardware 上完全不可部署 (因为 jerk 导致 safety stop 触发)。这呼应了之前的工作 [Chi et al. 2023 Diffusion Policy, https://arxiv.org/abs/2303.04137] 在 end-effector space + 10 Hz 才跑得动的情况，必须做 mode consistency 才能 closed-loop deploy。这是 paper 一个很重要的 contribution: 把 Diffusion Policy 从 planning-style 推到了 low-level control。

### 4.3 Computational time (Tab. IV, VII)

- MPC: 47.54 ms mean (CPU)
- DDPM 5 steps: 9.77 ms (CPU), **0.85 ms (GPU)** — 70× speedup
- DDPM 40 steps: 107.18 ms (CPU), 3.79 ms (GPU)
- DDPM + Safe sampling (100 samples): 28.38 ms (CPU), 3.305 ms (GPU)
- DDPM + Cost sampling: 27.539 ms (CPU), 4.499 ms (GPU)

GPU 并行 sampling 100 个 trajectories 几乎没有 overhead (1.3-4.6 ms)，这是非常 impressive 的工程结果。CPU 上 batch 100 大约 4× slowdown，对 embedded system 也勉强可用。

### 4.4 Constraint satisfaction (Fig. 5, Tab. VI)

加入 spherical obstacle (40 cm) 在 workspace 中心:
- LSM: simulation 10% tolerance SR 84.2%, 1% tolerance 80.3% (差)
- DDPM (5 steps): 10% → 97.4%, 1% → 88.5%
- DDPM + Safe sampling: 10% → 99.2%, 1% → 95.1%

DM 显著好于 LSM 在 non-convex constraint 上，这印证了 multi-modal learning 的价值。

## 5. Architecture & training details

### 5.1 Network architecture

- Diffusion model: 7-layer MLP, 1000 neurons/layer, **6.6M parameters**
- Temporal encoder: MLP (encode step $i$)
- Observation encoder: MLP (encode $x_t$ and $y^d$, where $y^d$ 用 6D rotation representation [Zhou et al. 2019, https://arxiv.org/abs/1812.07035] 避免 discontinuity 问题)
- LSM baseline: 6-layer, 2.8M parameters

不是 transformer，不是 U-Net，是 plain MLP。这反映 1D control signal (joint velocities) 不需要 spatial inductive bias。

### 5.2 Diffusion training tricks

- Cosine noise schedule [Nichol & Dhariwal 2021, https://arxiv.org/abs/2102.09772]
- 5 denoising steps (!) — 这是激进的选择
- Rescaling 确保最后一步 0 SNR [Lin et al. 2024, https://openaccess.thecvf.com/content/WACV2024/html/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.html] — 这避免最后一步的 color shift / DC offset 类问题
- Min-SNR weighting [Hang et al. 2023, https://arxiv.org/abs/2303.09551] with $\gamma = 5$ — 这个 trick 让训练 loss 在不同 timesteps 上 weight 更均衡

5 denoising steps 是 deployment 的关键。Consistency Policy [Prasad et al. 2024, https://arxiv.org/abs/2405.0784] 用 distillation 把 Diffusion Policy 压缩到 1-3 步，但这篇直接训 5 步，避免了 distillation 的复杂度。

## 6. 相关工作脉络

### 6.1 AMPC 谱系

- [Parisini & Zoppoli 1995, https://www.sciencedirect.com/science/article/abs/pii/S0005109894100094]: 最早用 NN 近似 MPC
- [Nubert et al. 2020, https://arxiv.org/abs/1909.09142]: Safe and fast tracking on robot manipulator with robust MPC + NN — 这个组的 previous 工作，也是 baseline
- [Hose et al. 2025, https://arxiv.org/abs/2312.06225]: Safety-augmented NN for AMPC
- [Li et al. 2022, https://www.sciencedirect.com/science/article/pii/S0005109822001543]: 用 stochastic programming 训 NN approximation of MPC，强调 multi-modal 必须避免
- [Carius et al. 2019, https://arxiv.org/abs/1908.05622]: MPC-Net，mixture density networks for multi-modal

### 6.2 Diffusion for robotics 谱系

- [Janner et al. 2022 Diffuser, https://arxiv.org/abs/2205.09991]: Diffusion 当 planning 的 world model
- [Chi et al. 2023 Diffusion Policy, https://arxiv.org/abs/2303.04137]: Visuomotor policy，10 Hz end-effector
- [Carvalho et al. 2023 Motion Planning Diffusion, https://arxiv.org/abs/2310.06122]: trajectory prior + gradient guidance for obstacle avoidance
- [Prasad et al. 2024 Consistency Policy, https://arxiv.org/abs/2405.0784]: 1-3 步 denoising，用 consistency distillation
- [Huang et al. 2025, https://arxiv.org/abs/2411.03995]: Diffusion 找 global optima，但 200 ms+ 太慢
- [Romer et al. 2024, https://arxiv.org/abs/2412.09342]: Diffusion predictive control with constraints
- [Li et al. 2025 DiffuSolve, https://arxiv.org/abs/2407.07455]: Diffusion warm-start nonlinear solver

### 6.3 Score matching / guidance 理论

- [Vincent 2011, https://direct.mit.edu/neco/article-lookup/doi/10.1162/NECO_a_00142]: 连接 score matching 和 denoising autoencoders
- [Song et al. 2021 Score-based, https://arxiv.org/abs/2011.13456]: SDE formulation
- [Dhariwal & Nichol 2021, https://arxiv.org/abs/2105.05233]: Classifier guidance，beat GANs

## 7. Critical takeaways & intuition building

### 7.1 为什么 diffusion 比 GMM/Mixture Density Network 好?

MDN [Bishop 1994, https://publications.aston.ac.uk/id/eprint/373/1/CRG_PSTR_NCRG_1994.pdf] 需要 pre-specify mode 数量 $K$，这在 MPC 里很难知道 (redundant DOF 在不同 pose 下可能不同数量的 IK solutions，obstacle configuration 不同时 mode 数量也不同)。Diffusion 是 non-parametric density estimator，single model 能 capture arbitrary number of modes。

### 7.2 为什么 joint space 比 end-effector space 难?

Diffusion Policy [Chi et al. 2023] 在 end-effector space 做 10 Hz，因为 EE trajectory 是相对 low-dim 且 smooth 的。Joint space 直接输出 7-dim joint velocities，每一步都要保证 robot 不违反 joint limits、self-collision、obstacle collision，constraint satisfaction 更难。这篇 paper 把 rate 推到 250 Hz，关键贡献。

### 7.3 三个 trick 的角色分工

| Trick | 解决的问题 | 效果 |
|---|---|---|
| ∇G | mode swapping (closed-loop consistency) | 3.00 → 0.08 mode swaps |
| ES | jerk (last-mile smoothness) | 2403 → 57.7 rad/s³ |
| Sampling | pick best mode / feasibility | SR 95.76 → 96.09-99.20 |

三者组合才能 hardware deployable，少任何一项都有问题。

### 7.4 反直觉的结果: DM 超越 teacher

最 surprising 的: DM 的 SR (96.33% sim, 93.00% hw) 比 MPC (93.18% / 87.80%) 还高。这有几个原因:
1. **MPC numerical failures filtered**: 5% unsatisfactory local minima 从 training set 移除，DM 学不到 bad modes
2. **DM 在 fast update rate 下更稳**: 250 Hz vs MPC 的 10 Hz，避免 overshoot
3. **Multi-modal exploration during training**: 数据包含不同 init 找到的不同 optima，DM 学到的是 weighted ensemble

这有点像 RLHF 中 reward model 偶尔超越 human annotator 的现象，但更系统化。

### 7.5 没解决的问题 & future directions

作者自己提到的:
- 高维 input/output (vision)
- Training data 量大 (55M+ predictions，几天/几周的数据采集)
- Embedded deployment (MLP 6.6M params + diffusion，对 microcontroller 还是重)

我想到的:
- **Non-stationary environments**: 当 obstacle 移动时，conditional distribution 会 time-varying，DM 需要在线适应
- **Long horizon**: $N=20$ 在 0.1s dt 下是 2 秒，对 dynamic task 可能不够
- **Closed-loop stability guarantees**: paper 完全没讨论 Lyapunov-style stability，而 MPC 的核心 selling point 就是 stability guarantee。Diffusion policy 本质是 black-box
- **Out-of-distribution generalization**: 当 $x_t$ 在 training distribution 之外时，DM 行为不可预测。MPC 仍会 work (虽然可能慢)

### 7.6 给 Andrej 的几个 potential research directions

1. **Diffusion + MPC residual**: 用 diffusion 给 IPOPT warm-start (类似 DiffuSolve)，但让 MPC 做 refinement，保留 safety guarantee + speedup
2. **Consistency model 替代**: 1-step generation，跑 kHz 应该没问题
3. **Score-based SDE formulation**: 把 control 当作 stochastic differential equation，可能能给 stability analysis
4. **Diffusion policy with state observer**: 当前 DM 直接 predict $u_t$ from $x_t$，没显式 state estimation。把 EKF + DM 结合可能更 robust
5. **Energy-based model 替代 diffusion**: 同样能 capture multi-modal，且 likelihood-based，可能更稳定

## 8. 实操细节 cheat sheet

如果你要复现:

- **Solver**: IPOPT via CasADi + Pinocchio (RBD)
- **Simulator**: Isaac Lab / Orbit [Mittal et al. 2023, https://arxiv.org/abs/2305.14244] — GPU vectorized
- **Parallel data collection**: CoClusterBridge (multi-process bridge from sim cluster to solver cluster)
- **Robot**: KUKA LBR4+ (Apollo platform [Kappler et al. 2018, https://arxiv.org/abs/1804.05011])
- **Hardware**: Intel Core i9-13900K + NVIDIA RTX 4070
- **Low-level**: PI controller tracking joint velocities at 1 kHz
- **Diffusion inference rate**: 250 Hz (commanded), 留 4 ms margin from 0.85 ms GPU time

## 9. 总结

这篇 paper 我觉得最重要的 contribution 不是 diffusion 本身，而是把 **三个 inference-time tricks (∇G, ES, parallel sampling) 组合起来让 diffusion 可用** 在 low-level control 上。每个 trick 单独看都不复杂，但组合起来解决了 closed-loop consistency, smoothness, constraint satisfaction 三个问题，才能从 10 Hz end-effector planning 推进到 250 Hz joint velocity control。

数学上最 elegant 的部分是 gradient guidance 和 score matching 的联系 (Eq. 10, 12, 13)，把 diffusion model 的 reverse step 当作 Bayes posterior update，extra information 作为 likelihood。这个 framework 应该可以推广到其他 conditional generation 场景 (e.g. goal-conditioned RL with diffusion policies)。

实验上最 striking 的数字是 Tab. III 的 jerk reduction (40×) 和 Tab. IV 的 73× speedup on GPU，以及 Fig. 5 中 DM 在 non-convex constraint 上对 LSM 的碾压。

我希望未来工作能在 stability guarantee 上有突破，否则 diffusion-based controller 永远只能在 "non-safety-critical" 应用里用，没法替代 MPC 在真正安全相关场景的位置。

## References (web links)

1. DDPM (Ho et al. 2020): https://arxiv.org/abs/2006.11239
2. Diffusion Policy (Chi et al. 2023): https://arxiv.org/abs/2303.04137
3. Diffuser (Janner et al. 2022): https://arxiv.org/abs/2205.09991
4. Motion Planning Diffusion (Carvalho et al. 2023): https://arxiv.org/abs/2310.06122
5. Consistency Policy (Prasad et al. 2024): https://arxiv.org/abs/2405.0784
6. Classifier Guidance (Dhariwal & Nichol 2021): https://arxiv.org/abs/2105.05233
7. Improved DDPM (Nichol & Dhariwal 2021): https://arxiv.org/abs/2102.09772
8. Min-SNR weighting (Hang et al. 2023): https://arxiv.org/abs/2303.09551
9. Flawed noise schedules (Lin et al. 2024): https://openaccess.thecvf.com/content/WACV2024/html/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.html
10. 6D rotation representation (Zhou et al. 2019): https://arxiv.org/abs/1812.07035
11. Score-based SDE (Song et al. 2021): https://arxiv.org/abs/2011.13456
12. Vincent 2011 (score matching connection): https://direct.mit.edu/neco/article-lookup/doi/10.1162/NECO_a_00142
13. Mixture Density Networks (Bishop 1994): https://publications.aston.ac.uk/id/eprint/373/1/CRG_PSTR_NCRG_1994.pdf
14. Nubert et al. 2020 (safe and fast tracking): https://arxiv.org/abs/1909.09142
15. Hose et al. 2024 (parameter-adaptive AMPC): https://arxiv.org/abs/2403.08510
16. Li et al. 2022 (stochastic programming AMPC): https://www.sciencedirect.com/science/article/pii/S0005109822001543
17. DiffuSolve (Li et al. 2025): https://arxiv.org/abs/2407.07455
18. Huang et al. 2025 (near-globally optimal MPC): https://arxiv.org/abs/2411.03995
19. Romer et al. 2024 (diffusion predictive control): https://arxiv.org/abs/2412.09342
20. Kohler et al. 2020 (nonlinear tracking MPC): https://www.sciencedirect.com/science/article/pii/S0005109820301939
21. Spencer et al. 2021 (covariate shift): https://arxiv.org/abs/2102.02872
22. IPOPT (Wächter & Biegler 2006): https://link.springer.com/article/10.1007/s10107-004-0559-y
23. CasADi: https://arxiv.org/abs/1903.07605
24. Pinocchio: https://hal.inria.fr/hal-01866228v1
25. Orbit / Isaac Lab: https://arxiv.org/abs/2305.14244
26. Apollo (Kappler et al. 2018): https://arxiv.org/abs/1804.05011
27. Carius et al. 2019 (MPC-Net): https://arxiv.org/abs/1908.05622
28. Rawlings MPC book: https://sites.engineering.ucsb.edu/~jbraw/mpc/

希望这个解读能 build 出足够的 intuition，Andrej 你看哪里需要进一步深入。
