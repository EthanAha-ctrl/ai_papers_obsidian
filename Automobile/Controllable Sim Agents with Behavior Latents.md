---
source_pdf: Controllable Sim Agents with Behavior Latents.pdf
paper_sha256: 4d91eacaf0378cc09f1beb12eefe449be85ae46a8780e99262355b71c2a7da48
processed_at: '2026-08-18T04:12:52-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CNeVA 人话版

Andrej，来，我当你是同事，咱在白板前面站着聊这篇 paper。

---

## 这帮人到底想解决什么 pain point？

你知道现在 sim agent 这摊事儿的尴尬处境。搞 imitation learning 的那帮人（TrafficBots, SMART, Trajeglish, BehaviorGPT），trajectory 生成得挺像真的，但有个致命问题——**你没法控制生成出来的 agent 到底是什么风格**。

你想测一个自动驾驶 policy，你需要能说"给我一个 aggressive 的 agent"、"给我一个超级保守的 agent"、"给我一个专门爱超速的 agent"。imitation learning 给你的就是一个 distribution over trajectories，你 resample 一下得到不同的 trajectory，但你说不出这个 trajectory 是"激进"还是"慢悠悠"，它就是随机的。

VAE 那帮人（TrafficSim, TrafficBots, STAGE）确实搞了个 latent z，通过 resample z 可以得到不同 behavior。但 z 是个纯黑盒——它从 data 里 emerge 出来，没有任何 semantic axis 对齐。你 resample 出来一个 z，你只能说"这是一个 z"，你说不出"这个 z 更激进"还是"这个 z 更保守"。

Diffusion + guidance 那帮人（SceneDiffuser, Guided Conditional Diffusion, ScenarioDiffusion）可以在生成过程里注入一个 control signal 来 steer，但每个新 behavior 你都要 redesign guidance，而且 guidance 调大了 trajectory 就变得不 realistic。

Self-play RL 那帮人（CTRL-Sim, SPACeR）能学 desired behavior，但 online training 贵得要死，reward 变了你就得 retrain。

所以整个 field 的痛点是：**我想要一个 sim agent 模型，它既 real 又 controllable，而且 controllable 的那个"把手"最好有 explicit semantic meaning，不要是黑盒。**

CNeVA 这帮 Purdue 和 Tokyo 的人就想：咱能不能搞个"driving style"的 explicit 接口？

---

## 核心想法：把"driving style"变成对 reward 的 weight

他们的核心 insight 极其 simple：**一个 agent 的 driving style，本质上就是它怎么 weight 各种 reward**。

什么意思呢？你想想，所有 driver 都在意安全、都在意地图合规、都在意速度、都在意加速舒适度。aggressive driver 和 conservative driver 的区别，仅仅在于 aggressive driver 对 speed reward 给的 weight 更高，对 safety reward 给的 weight 更低。conservative driver 反过来。

所以他们说：咱固定一个 reward basis，就 K=4 个 channel：safety, map, speed, accel。每个 agent 的 driving style 就是它在这 4 个 channel 上各自的 weight，记为一个 K-dim vector **λ_n ∈ R^K**。

这就把无限维的"driving style"压缩成一个 4-dim vector，而且每个 dimension 都有 explicit semantic meaning：第 1 维 = 这个 agent 多在乎安全，第 2 维 = 多在乎地图合规，第 3 维 = 多在乎速度，第 4 维 = 多在乎加速舒适度。

这个 formulation 一下子给你了三件事：
1. λ_n 有 explicit semantic meaning
2. inference 超便宜——closed-form Bayesian update
3. deployment 时 operator 直接 specify λ_n^op = ρ e_k 就能 steer 第 k 个 channel

---

## 怎么从 logged trajectory 反推出每个 agent 的 λ_n？

这是 paper 的数学核心，但其实 intuition 很简单。

你手上有一堆 logged data：agent n 在时间步 1..T 走了一条 trajectory，每一步它都获得一个 K-dim reward vector r(o_{t,n})。把这个 K-dim reward 沿时间 discount 累加，得到 **per-channel discounted return**:

$$
G_n = \sum_{t=1}^T \gamma^{t-1} r(o_{t,n}) \in \mathbb{R}^K
$$

变量解释：
- **t**: time step index，从 1 到 T
- **γ**: discount factor，0 到 1 之间，γ^{t-1} 让后面步骤的 reward weight 逐渐衰减
- **r(o_{t,n})**: 第 t 步第 n 个 agent 拿到的 K-dim reward vector
- **G_n**: 最终的 K-dim cumulative return，每个 channel 一个数

现在问题变成：已知 agent n 拿到了 G_n 这个 return，反推它的 behavior latent λ_n 是什么？

他们设计了一个 exponential preference factor:

$$
\psi(\tau_n, \lambda_n) = \exp(\lambda_n^\top G_n)
$$

这个 exp 的妙处在于：它把 trajectory-level factor 变成 λ_n 和 G_n 的 **inner product**。然后如果你给 λ_n 一个 Gaussian prior p(λ_n) = N(μ_0, Σ_0)，Bayesian conjugate update 直接给你 closed-form posterior：

$$
q^*(\lambda_n) = \mathcal{N}(\mu_0 + \Sigma_0 G_n, \Sigma_0)
$$

变量解释：
- **μ_0**: prior mean，population 平均 driving style
- **Σ_0**: prior covariance，控制 prior 对 posterior 的 influence
- **G_n**: 这个 agent 的 observed per-channel return
- posterior mean 是 μ_0 + Σ_0 G_n：先验均值加上 return 的 contribution
- posterior covariance还是 Σ_0，因为 likelihood exp(λ_n^T G_n) 在 λ_n 上是线性的，不携带 covariance 信息

这里有个**特别 elegant 的 dual interpretation**：

**Tilt view**: exp(λ_n^T G_n) 把 probability mass 向"能解释 observed return"的 profiles 倾斜

**Bayesian regression view**: λ_n 在 G_n 上做 ridge regression，被 prior precision Σ_0^{-1} regularize 向 μ_0

这两种 view 等价，但 regression view 给你一个直接的 shrinkage intuition：

$$
\text{shrinkage factor} = \Sigma_0 (\Sigma_0 + \Sigma_{\text{noise}})^{-1}
$$

- High SNR channel（dense, trajectory-level penalty 比如 speed）：强 shrinkage toward observed return → 易 steer
- Low SNR channel（sparse, event-dependent penalty 比如 collision）：prior-dominated → 难 steer

这个 shrinkage factor 直接 predicts 实验中的 controllability hierarchy。speed 和 accel 这种 dense channel 会很好 steer，safety 和 map 这种 sparse channel 难 steer。这是 paper 的理论 contribution——**identifiability 是从 return-shrinkage factor 直接推出来的**，不需要 ablation 才能发现。

---

## 一个关键细节：per-channel standardization

我必须强调这个细节，因为它看起来 mundane 但实际上决定了整个框架能不能 work。

他们测了一下 WOMD 上四个 channel 的 raw return 统计：

- Safety: mean -36.84, std 19.40
- Map: mean -39.81, std 33.63
- Speed: mean -50.06, std 12.93
- Accel: mean -3.03, std 5.23

注意 speed 的 mean 是 -50，accel 的 mean 只有 -3，**差了 17 倍**。

如果直接把 raw G_n 喂进 conjugate update 用 Σ_0 = I，speed channel 会 dominate posterior 一个数量级，accel channel 的 contribution 几乎被淹没。conditional generator 根本没法用 accel channel 来 condition。

所以他们做了一个 per-channel standardization:

$$
\widetilde{G}_{n,k} = (G_{n,k} - \mu_{G,k}) / \sigma_{G,k}
$$

把每个 channel 的 return 用 calibration split 上测得的 mean 和 std normalize 一下。这样四个 channel 在 posterior update 里就处于 equal footing，每个 channel 的 posterior mean 都在 order-unity scale。

这个 trick 是让 K=4 channel 同时 work 的前提条件。你要 build CNeVA，不做这一步，accel channel 永远 steer 不起来。

---

## Generator：Rectified Flow + λ-conditional

inference 出 λ_n 之后，他们用一个 **rectified flow** generator 来 generate trajectory。

rectified flow 的 idea：把 noisy trajectory x_s 沿直线 push 回 clean trajectory y_n。flow time s ∈ (0,1)，s=1 是纯噪声，s=0 是 clean target。

$$
x_s = (1-s)y_n + s\epsilon, \quad v_n^{\text{target}} = \epsilon - y_n
$$

- **y_n**: clean future displacement target
- **ε**: standard normal noise
- **x_s**: noisy latent 在 linear interpolation path 上
- **v_n^target**: 网络要预测的 velocity，就是从 clean 到 noise 的方向

网络是一个 Transformer decoder，输入 x_s、flow time s、history o_{<t}、map m、还有那个 4-dim behavior latent λ_n。λ_n 作为一个 extra cross-attention token prepended 到 scenario-context。

训练 loss 就是 MSE:

$$
\mathcal{L}(\theta) = \sum_n \mathbb{E}[\|v_\theta(x_s, s, o_{<t}, m, \lambda_n) - v_n^{\text{target}}\|^2]
$$

为什么选 rectified flow 不选 diffusion？rectified flow 的 path 是直线，Euler integrator 只需要 10 步就收敛，diffusion 通常要几十上百步。对于 8s rollout 这种实时性要求不强的场景，rectified flow 的 efficiency 优势巨大。

---

## Mixed Channel-Mask CFG Curriculum：最聪明的工程 trick

这一块是 paper 里我觉得最聪明的工程 contribution。

问题是这样：你想用 classifier-free guidance（CFG）。标准 CFG 是在 fully-conditional（λ=真实 inferred value）和 fully-unconditional（λ=null embedding）之间 flip。训练时网络只见过这两种 extreme。

但 inference 时，operator 想说"我只 steer speed channel，其他 channel 我不 care"，于是他给一个 λ^op = ρ e_k，只有第 k 个 channel 有值，其他全是 0。

**这个 one-hot probe 严格 outside training support**——网络从来没见过这种 sparse λ。

更糟的是，如果你用标准 classifier guidance（Dhariwal & Nichol 2021 那套），在 noised trajectory x_s 上训一个 regressor p_φ(G|x_s, s)，然后沿 ∇_{x_s} log p_φ steer，在高维 λ-space 这其实类似一个 gradient-based adversarial attack——high-confidence directions 不等于 high data density directions，容易崩。

他们的解决方案：**让训练时见过所有 partial mask 配置**。

四分支 mask curriculum：
- null (20%): 所有 channel masked，用 learned null embedding e_∅
- single-channel (40%): 恰好一个 channel kept
- two-channel (20%): 恰好两个 channel kept
- full (20%): 所有 channel kept

而且 mask 不能只 mask value——你得告诉网络"这个 channel 是 unobservable"vs"这个 channel value 就是 0"，这两者 semantic 完全不同。所以他们 concat value-zeroed λ 和 binary mask indicator：

$$
\widetilde{\lambda}_n = [(1-b_n) \odot \lambda_n; b_n] \in \mathbb{R}^{2K}
$$

- **b_n**: K-dim binary mask，b_{n,k}=1 表示 channel k masked-out
- **(1-b_n) ⊙ λ_n**: 把 masked channel 的 value 置零，保留 kept channel 的 value
- **[ ; ]**: concat，前 K 维是 value-zeroed λ，后 K 维是 mask indicator

当 b_n = 1_K（全 masked），整个 projection 被 null embedding e_∅ override，这就是 unconditional branch。

inference 时 CFG combination:

$$
\widetilde{v}_\theta^w = (1+w) v_\theta(\text{cond}) - w v_\theta(\text{null})
$$

w=1.5 是 calibrated operating point。

这个 curriculum 的妙处：inference 时 operator 给 one-hot probe λ = ρ e_k，恰好 match single-channel training branch 的 distribution。网络不会 outside support。

---

## Soft Eligibility Gates：fix reward hacking 和 safety erosion 的关键

这一块是 paper 里第二个核心工程 contribution，也是 honest 的地方。

先说问题。safety 和 map 这两个 channel 的 reward 是 sparse 的——大多数 agent 根本不撞车，也不 offroad，所以大多数 agent 的 safety return 和 map return 都是 trivial 的。

他们一开始用 **hard eligibility gate**：只对 clearance < 5m 或 TTC < 6s 的 agent 标 safety label。结果发现 safety controllability 直接崩了——CSM diagonal 只剩 +0.21，统计上跟零没区别。

原因很明显：hard gate 把绝大多数 safe agent 排除在 supervision 之外，generator 看到的 safety signal 几乎为零，自然学不到东西。

他们还遇到一个 **reward hacking** 问题。早期 checkpoint（40K steps）测了一下，speed CSM 高达 +51.3，看起来 speed steering 效果爆炸好。但仔细看 physical plausibility——75.9% 的 agent stall（基本不动），retained speed 只有 61% GT。

这帮人通过 stalling 来 maximize speed return！因为 speed reward 是 negative penalty，agent 不动就没速度 penalty，return 就高。这是教科书级的 reward hacking。

他们的 fix 是 **soft eligibility gates**——用 smooth exponential decay 替代 binary threshold:

$$
\widetilde{r}_{t,ij}^{\text{safe}} = r_{t,ij}^{\text{safe}} \cdot \exp\Big(-\frac{\max(c_{t,ij}, 0)}{\tau_c}\Big) \cdot \exp\Big(-\frac{\max(\text{ttc}_{t,ij}, 0)}{\tau_t}\Big)
$$

- **c_{t,ij}**: agent i, j 在 t 时刻的 pairwise clearance
- **ttc_{t,ij}**: pairwise time-to-collision
- **τ_c = 2.0 m, τ_t = 3.0 s**: decay scales

乘积形式很关键：clearance 和 TTC **都**要大才能让 risk 消失。一个 distant but fast-approaching agent（高 clearance 但低 TTC）仍然 retain 高 risk weight。这 capture 了真实 driving risk 的本质。

结果：
- Safety CSM 从 +0.21（hard gate, statistical zero）跳到 +0.66（soft gate, significant）
- Stall fraction 只增加 +0.9 pp
- Retained speed 97.9% GT
- minADE 与 hard gate 版持平（1.113 vs 1.112）

**这证明 safety steering 真的是 defensive driving**（spacing, yielding），不是 slow-equals-safe confound。soft eligibility 把 signal 留下来给 near-threshold agent，gradient 不再被 binary threshold 切断。

---

## Context-Residual Return：把 scene difficulty 从 driving style 里剥出来

raw return G_{n,k} 有个本质混淆：它同时 encode 了 driving style 和 scene difficulty。

举例：highway scene 无论 driver 怎么开，offroad penalty 都低，因为 highway 本来就没啥 offroad risk。你不能因为 highway agent 的 map return 高，就说这个 agent "更 map-compliant"——它只是恰好在一个 easy scene 里。

他们的 fix：

$$
G_{n,k}^{\text{cr}} = G_{n,k} - \bar{G}_k(m_n)
$$

- **Ḡ_k(m_n)**: 同 map context m_n 下所有 agent 的 mean return，从 calibration split 估计一次
- 这个 residual 把 scenario structural component 减掉，留下 pure behavioral signal

这个 trick 看起来简单，但实际效果巨大。Figure 8 显示 map controllability 在三种 return measure 下的对比：
- Context-residual: +0.61 ✓
- Physical-offroad: -0.12 ✗
- Lane-centerline: ≈ -0.002 ✗

只有 context-residual 能 produce measurable steering response。其他两个 measure 完全 fail。

**这是 paper 的 honest limitation**：map controllability 强烈依赖 return 定义。当前 reward basis 无法做到 lane-keeping 这种 coordinate-level control，需要 richer decomposition 把 spatial 和 temporal map compliance 分开。

---

## 实验结果速读

### Benchmark (WOSAC)

CNeVA Realism = 0.7145，minADE = 1.80m，mid-spectrum。Leading 是 SMART-R1 (0.7855) 和 TrajTok (0.7852)，都是 tokenized closed-loop imitation。CNeVA 没做 closed-loop fine-tuning，没做 autoregressive token prediction，但已经 mid-spectrum。gap 集中在 collision 和 offroad，与 open-loop drift 一致。

### Controllability (CSM diagonal)

ρ=1, w=1.5, open-loop:

| Channel | Baseline | CNeVA w=0 | CNeVA w=1.5 |
|---------|----------|-----------|-------------|
| Safety  | +0.06    | +0.29     | +0.66 ± 0.10 |
| Map     | +0.06    | +0.24     | +0.61 ± 0.14 |
| Speed   | -3.33    | +3.21     | +8.15 ± 0.07 |
| Accel   | +4.77    | +4.19     | +8.76 ± 0.07 |

Hierarchy 与 shrinkage prediction 完全一致：dense kinematic (speed, accel) >> sparse semantic (safety, map)。

w=0 vs w=1.5 对比揭示：dense channel 已经 strongly steerable 只靠 latent；sparse channel 需 CFG 翻倍才达到 significant response。

### Physical Plausibility (Reward Hacking Diagnostic)

| Model | minADE | Stall% | v/v_GT | ΔR_speed | ΔR_safety | Offroad% |
|-------|--------|--------|--------|----------|-----------|----------|
| CNeVA (soft) | 1.113 | 65.1 | 94.7% | +8.15 | +0.66 | 32.5 |
| Early (40K) | 1.238 | 75.9 | 61% | +51.3 | +2.18 | 34.6 |
| Hard-elig (200K) | 1.112 | 65.4 | 94% | +9.5 | +0.21 | 32.9 |

早期 checkpoint 的 speed CSM 是 hack 出来的——75.9% stall, 61% GT speed。Main model 是真的开得更快。Hard-elig 的 safety CSM 崩到统计零，main model 通过 soft eligibility 救回 +0.66。

### Pairwise Trajectory Divergence

speed-accel 之间最大分离 (0.96m)，safety-map 之间几乎重叠 (0.10m)。sparse channel 在 trajectory space 几乎 indistinguishable——几何上就解释了为什么 sparse channel 难 steer。

### Operating Regime

ρ=5 时所有 channel 仍 monotone positive，但 guardrails degrade（stall +2.1~4.6pp, retained speed 94.7%→89.1%）。main results 用 ρ=1, w=1.5 作为 calibrated point。

---

## 整个 paper 的 conceptual story

让我用一个连贯 narrative 把所有 piece 串起来：

1. **Problem**: sim agent 既要 real 又要 controllable，而且 controllable 接口要有 explicit semantic meaning

2. **Insight**: driving style = how agent weights reward channels。固定 reward basis (safety/map/speed/accel)，每个 agent 的 style 就是一个 K-dim vector λ_n

3. **Inference**: 从 logged trajectory 的 per-channel return G_n 出发，用 Gaussian prior + exponential tilt 得到 closed-form conjugate posterior。posterior mean = μ_0 + Σ_0 G_n，O(K) 复杂度

4. **Standardization**: raw return channel 间量级差 17 倍，必须 per-channel standardize 才能让四个 channel 同时 work

5. **Context residual**: raw return 混淆 style 和 scene difficulty，residualize 掉 scene baseline 才能 isolate behavioral signal

6. **Generator**: rectified flow + λ-conditional Transformer，10-step Euler sampler，receding-horizon patch (ℓ=16) 缓解 open-loop drift

7. **Mixed mask curriculum**: 让网络见过所有 partial mask 配置，inference 时 one-hot probe 不 outside support。mask indicator concat 区分"unobservable"vs"value=0"

8. **CFG**: w=1.5 calibrated operating point，sparse channel 需 CFG 翻倍才 significant

9. **Soft eligibility gates**: smooth exponential decay 替代 binary threshold，保留 near-threshold agent 的 gradient，同时 fix reward hacking（stall hack）和 safety erosion（hard gate kill signal）两个 failure mode

10. **Physical plausibility guardrails**: CSM 必须配合 stall fraction / v/v_GT / offroad rate 一起读。CSM 高不代表真 steer 了，可能是 hack

11. **Identifiability hierarchy**: shrinkage factor 直接 predict dense channel (speed, accel) 易 steer，sparse channel (safety, map) 难 steer。实验完全 confirm

12. **Honest limitation**: map controllability 强依赖 return measure。context-residual work (+0.61)，physical-offroad (-0.12) 和 lane-centerline (≈0) 都 fail。coordinate-level control 需要 richer reward decomposition

---

## 为什么我觉得这 paper 值得读

Andrej，从 build intuition 的角度，这篇 paper 有几层 value：

**第一层 method value**: 把 inverse-RL 的 utility identification 问题变成 closed-form Bayesian regression，cheap 且 interpretable。这个 trick 不仅适用 traffic sim，任何"从 observed behavior infer latent preference"的场景都能用——比如 robotics, game AI, recommendation system。

**第二层 engineering value**: mixed channel-mask curriculum 和 soft eligibility gates 是两个非常实用的工程 trick。前者解决 partial conditioning 的 outside support 问题，后者解决 sparse reward 的 gradient 断裂问题。这两个 trick 都可以 transfer 到其他 conditional generation 场景。

**第三层 conceptual value**: tilt-vs-regression dual view + return-shrinkage factor 给你一个 **a priori predict controllability** 的工具。你不需要跑实验就知道哪些 channel 易 steer 哪些难——直接看 reward 的 density 和 SNR 就行。

**第四层 honesty value**: physical plausibility guardrails + reward hacking diagnostic + map controllability limitation 的 honest discussion，这在现在的 paper 里挺少见的。大多 paper 只 report positive CSM，不 check 是不是 hack 出来的。

最后，如果让我用一句话总结 CNeVA：**它把 driving style inference 和 controllable generation 用 conjugate Gaussian posterior 优雅地串起来，同时 honest 地 acknowledge 自己的 limitation 并通过 reward hacking diagnostic 提醒 community——steering metrics 必须配合 physical-plausibility guardrails 一起读。**

---

# CNeVA: Controllable Sim Agents via Behavior Latents 深度解析

Andrej，这篇 paper 的核心 idea 其实非常 elegant：把 traffic simulation 中每个 agent 的"driving style"用一个 **per-agent Gaussian behavior latent** λ_n 来 encode，这个 latent 不是黑盒，而是 explicit 对应到一个**预定义的 reward basis**（safety / map / speed / accel）上。然后通过一个**closed-form conjugate variational update**从 logged trajectory 的 per-channel discounted return 直接 infer 出来。最有意思的地方是它把 inverse-RL 的 utility identification 问题、Bayesian regression 的 shrinkage 直觉、和 rectified flow + classifier-free guidance 这一套生成模型工程**串成了一条完整的因果链**。

让我从 intuition 开始 build，然后一层层 drill down。

---

## 1. Core Intuition: Why Behavior Latent over a Reward Basis?

传统 imitation learning 学的是 p(τ | m, history)，给你一个 distribution over trajectories，但这个 distribution 是"搅拌在一起的"——你不知道某个 trajectory 是 aggressive 还是因为前方堵车。如果想"让这个 agent 开得更激进一点"，没有接口。

VAE-based methods（TrafficBots, TrafficSim, STAGE）有 latent z，但 z 是**纯结构 latent**：它从数据里 emerge 出来，没有任何 semantic axis 对齐。你 resample 一个 z 得到不同的 trajectory，但你说不出这个 z "更激进"或"更保守"。

Diffusion + guidance 可以注入 control signal，但 guidance signal 通常是一个 hand-crafted differentiable cost，每个新 behavior 都要 redesign guidance，而且 excessive guidance 会产生 unrealistic trajectory。

Self-play RL（CTRL-Sim, SPACeR）可以学 desired behavior，但 online training 很贵，reward 变了就要 retrain。

CNeVA 的核心 move：**先把 reward basis 写死**（K=4 channels: safety, map, speed, accel），然后假设 agent i 的 driving style 完全可以由它如何**weight 这 K 个 channel** 来描述。这就把"behavior style"压缩到一个 K-dim vector λ_n ∈ R^K。每个 channel 是一个**negative sum of penalties**（higher G_{n,k} = less penalized quantity）。

这个 formulation 的 power 在于：
1. λ_n 有 explicit semantic meaning（第 k 维 = 第 k 个 reward channel 的 weight）
2. Inference 是 closed-form Gaussian conjugate update，超便宜
3. Operator 在 deployment 时可以直接 specify λ_n^op = ρ e_k 来"steer 第 k 个 channel"
4. Conditional generator 用 rectified flow + CFG，自然支持 partial conditioning（mask 掉某些 channel）

---

## 2. The Probabilistic Graphical Model

Standard WOSAC PGM 是一个 HMM:
- s_t: latent world state
- o_{t,n}: per-agent observation
- m: static map
- o_{<t}: history

最大化 HMM likelihood 复现 logged behavior，但**没有 steering 接口**。

CNeVA 在 PGM 上加一个 **per-agent behavior profile** λ_n ∈ R^K，capture 每个 agent 如何 weight shared reward signals。

### Per-step preference factor

$$
\psi_t(o_{t,n}, \lambda_n) \triangleq \exp\Big(\gamma^{t-1} \lambda_n^\top r(o_{t,n})\Big) \quad (1)
$$

变量解释：
- **t** ∈ {1, ..., T}: time step index（下标）
- **γ** ∈ (0, 1]: discount factor，控制 future rewards 的 weight 衰减
- **λ_n** ∈ R^K: 第 n 个 agent 的 behavior latent（K-dim vector）
- **r(o_{t,n})** ∈ R^K: 第 t 步第 n 个 agent 的 per-step K-dim reward vector
- **⊤**: 转置
- **γ^{t-1}**: geometric discount，γ^0=1 at t=1

为什么用 exp？因为这样 trajectory-level factor 就 collapse 成 inner product，方便 conjugate update。

### Trajectory-level factor

$$
\psi(\tau_n, \lambda_n) = \prod_{t=1}^T \psi_t(o_{t,n}, \lambda_n) = \exp\Big(\lambda_n^\top G_n\Big) \quad (2)
$$

其中 **per-channel discounted return**:

$$
G_n \triangleq \sum_{t=1}^T \gamma^{t-1} r(o_{t,n}) \in \mathbb{R}^K
$$

这一步的关键 insight：通过 belief propagation，T 个 per-step factor 的 product collapse 成 λ_n 和 G_n 的 inner product。这就把一个 trajectory-level inference 问题降维成一个 K-dim Bayesian regression 问题。

### Relaxed joint distribution

$$
p(\tau, \lambda | m) \propto \prod_{n=1}^N p(\lambda_n) \psi(\tau_n, \lambda_n) \prod_{t=1}^T p(o_{t,n} | o_{<t}, m) \quad (3)
$$

- **p(λ_n)**: population-level Gaussian prior
- **ψ(τ_n, λ_n)**: reward-tilted likelihood
- **p(o_{t,n} | o_{<t}, m)**: autoregressive emission（由 CFM generator 建模）

PGM 见 Figure 1b，把 latent state s_t marginalize 掉之后，per-agent optimality chain 独立 decouple，所以可以 agent-by-agent 做 inference。

---

## 3. Conjugate Gaussian Posterior: The Mathematical Heart

### Variational objective

$$
q^*(\lambda_n) = \arg\min_{q \in \mathcal{Q}} \mathrm{KL}\big[q(\lambda_n) \,\|\, p(\lambda_n | \tau_n)\big] = \arg\max_{q \in \mathcal{Q}} \mathbb{E}_q[\lambda_n^\top G_n] - \mathrm{KL}[q(\lambda_n) \,\|\, p(\lambda_n)] \quad (4)
$$

直觉：
- 第一项 E_q[λ_n^T G_n]：鼓励 q 把 mass 放在能 explain observed return 的 profile 上
- 第二项 KL[q || p]：把 q 拉回 population prior，防止 overfitting 到单条 trajectory

### Closed-form solution

如果 prior p(λ_n) = N(μ_0, Σ_0)，那么：

$$
\boxed{q^*(\lambda_n) = \mathcal{N}\big(\mu_0 + \Sigma_0 G_n, \Sigma_0\big)} \quad (5)
$$

**这是整篇 paper 最 elegant 的结果**。注意 covariance 不变——因为 reward-tilted factor exp(λ_n^T G_n) 在 λ_n 上是**线性**的，所以 likelihood 不携带 covariance 信息，posterior covariance 就等于 prior covariance。

**Derivation (completing the square)**:

$$
\log p(\lambda_n | o_{1:T,n}) = -\frac{1}{2}(\lambda_n - \mu_0)^\top \Sigma_0^{-1}(\lambda_n - \mu_0) + \lambda_n^\top G_n + \text{const.} \quad (13)
$$

展开 quadratic:
$$
-\frac{1}{2}\lambda_n^\top \Sigma_0^{-1}\lambda_n + \lambda_n^\top \Sigma_0^{-1}\mu_0 + \lambda_n^\top G_n + \text{const.}
$$

collect λ_n 一次项：
$$
\lambda_n^\top (\Sigma_0^{-1}\mu_0 + G_n)
$$

precision matrix 是 Σ_0^{-1}（不变，因为 likelihood 线性），complete square 得 mean = Σ_0(Σ_0^{-1}μ_0 + G_n) = μ_0 + Σ_0 G_n。

### Reparameterization trick

$$
\lambda_n = \mu_0 + \Sigma_0 G_n + \Sigma_0^{1/2}\eta_n, \quad \eta_n \sim \mathcal{N}(0, I) \quad (6)
$$

- **Σ_0^{1/2}**: 协方差矩阵的 matrix square root（Cholesky 分解）
- **η_n**: 标准 normal 噪声，让 λ_n 可以 backprop

### Tilt-vs-Regression Dual Interpretation

这个 interpretation 是 paper 中最 valuable 的概念工具：

**Tilt view**: exp(λ_n^T G_n) 把 probability mass 向"解释 observed return"的 profiles 倾斜

**Bayesian regression view**: λ_n 在 G_n 上做 ridge regression，regularized 向 μ_0，regularization strength 是 prior precision Σ_0^{-1}

**Return-shrinkage factor**:
$$
\Sigma_0 (\Sigma_0 + \Sigma_{\text{noise}})^{-1}
$$

- High SNR channel（dense, trajectory-level penalties 如 speed）：强 shrinkage toward observed return → 易 steer
- Low SNR channel（sparse, context-dependent events 如 collision）：prior-dominated → 难 steer，靠 λ alone 不够，要靠 CFG

**这个 asymmetry 直接 predicts 实验中 controllability hierarchy**:
- Speed, accel (dense): ΔR ≈ +8
- Safety, map (sparse): ΔR ≈ +0.66, +0.61

这是 paper 的核心理论 contribution——**identifiability hierarchy 是从 return-shrinkage factor 直接推导出来的**。

---

## 4. Per-Channel Standardization (Critical Preprocessing)

Raw return G_n 的 channel 间量级差 huge：

| Channel | Mean μ_k | Std σ_k |
|---------|----------|---------|
| Safety  | -36.84   | 19.40   |
| Map     | -39.81   | 33.63   |
| Speed   | -50.06   | 12.93   |
| Accel   | -3.03    | 5.23    |

(From Eq. 23, measured on WOMD calibration split with γ=0.99, T_f=80)

如果直接用 G_n 进 Eq. 5，speed/map channel 会**主导** conjugate update 一个数量级。

标准化：
$$
\widetilde{G}_{n,k} \triangleq (G_{n,k} - \mu_{G,k}) / \sigma_{G,k}, \quad k = 1, \dots, K \quad (7)
$$

用 Σ_0 = I 时，每个 channel 的 posterior mean 都是 order-unity scale，四个 channel 在 conditional generator 处于**equal footing**。

这个细节看似 mundane，但实际上是让整个框架工作的**关键工程 trick**。没有这一步，speed channel 会"吃掉"其他 channel 的 conditioning signal。

---

## 5. Conditional Flow Matching Generator

### Optimization objective

$$
\theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim \mathcal{D}} \mathbb{E}_{\lambda \sim q^*(\lambda|\tau)} [\log p(\tau | m, \lambda; \theta)] \quad (8)
$$

用 conditional flow matching (Lipman et al. 2023) 作为 simulation-free surrogate。

### Rectified flow formulation

设 y_n 是 future motion target (displacement features: y_{t,n} = o_{t,n} - o_{t-1,n}):

$$
x_s = (1-s)y_n + s\epsilon, \quad v_n^{\text{target}} \triangleq \epsilon - y_n \quad (9)
$$

- **y_n** ∈ R^{T_f × d}: clean future displacement target (T_f=80 步, d=6 motion dim)
- **ε** ~ N(0, I): noise sample
- **s** ∈ (0, 1): flow time (从 logit-normal 采样，让中间步更密集)
- **x_s**: noisy latent on linear interpolation path
- **v_n^target**: 网络要预测的 velocity field target

直觉：rectified flow 把 noisy x_s 沿直线 push 回 clean y_n，所以 target velocity 就是 (ε - y_n) 的方向。

### Velocity field architecture

$$
v_\theta(x_s, s, o_{<t,n}, m, \lambda_n)
$$

输入：
1. noisy displacement latent x_s
2. flow time s（Fourier features embed）
3. cross-attention conditioning set:
   $$\text{cond}_n = [\text{Dense}_{K d_h}(\lambda_n); \text{embed}(o_{<t,n}); \text{embed}(m)]$$
4. λ_n 作为 extra cross-attention token prepended 到 scenario-context

Decoder block (6 layers, 8 heads, d_h=512):
- temporally causal self-attention over future axis
- unmasked spatial self-attention across agents
- cross-attention to cond_n
- feed-forward sub-block
- adaptive layer-norm（scale/shift 从 flow-time embedding 预测）

---

## 6. Mixed Channel-Mask CFG Curriculum

这是 paper 中**最聪明的工程 contribution 之一**。

### Why not vanilla CFG?

标准 CFG (Ho & Salimans 2022) 在 fully-conditional (λ=λ_n) 和 fully-unconditional (λ=e_∅) 之间 flip。但 inference 时 operator 给一个 one-hot probe λ = ρ e_k，只有 channel k informative，其他 K-1 个 = 0。这**严格 outside training support**——网络从来没见过这种 sparse λ。

### Why not classifier guidance?

在 noised trajectory x_s 上训一个 regressor p_φ(G | x_s, s) 来 steer，类似 Dhariwal & Nichol 2021。问题：
1. 大 s 时 x_s 几乎 isotropic Gaussian，supervisory signal 无信息
2. stepping along ∇_{x_s} log p_φ 在高维 λ-space 类似 adversarial attack——high-confidence directions 不等于 high data density directions

### Mixed channel-mask curriculum

四分支采样 mask b_n:
- **null branch** (b_n = 1_K, prob 0.2): 所有 channel masked，用 learned null embedding e_∅
- **single-channel** (Mask_1(K), prob 0.4): 恰好一个 channel kept
- **two-channel** (Mask_2(K), prob 0.2): 恰好两个 channel kept
- **full** (b_n = 0_K, prob 0.2): 所有 channel kept

### Mask indicator concatenation

$$
\widetilde{\lambda}_n \triangleq [(1-b_n) \odot \lambda_n; b_n] \in \mathbb{R}^{2K} \quad (10)
$$

- **(1-b_n) ⊙ λ_n**: 把 masked channel 的 value 置零，保留 kept channel 的 value
- **b_n**: binary mask indicator，告诉 projection 哪个 channel 是 unobservable

为什么 concat mask indicator？因为 projection 要区分"channel k 的 value 是 0"vs"channel k 是 unobservable"——这两者在 value-zeroed 之后无法区分，但 semantic 完全不同。

当 b_n = 1_K 时，整个 λ̃_n projection 被 learned null embedding e_∅ ∈ R^{d_h} **override**，这就是 unconditional branch。

### Full training loss

$$
\mathcal{L}(\theta) = \sum_{n=1}^N \mathbb{E}_{q^*(\lambda_n), b_n, s, \epsilon} \Big[\big\|v_\theta(x_s, s, o_{<t,n}, m, \widetilde{\lambda}_n) - v_n^{\text{target}}\big\|_2^2\Big] \quad (11)
$$

---

## 7. Inference: CFG + Euler ODE Sampler

### CFG combination

$$
\widetilde{v}_\theta^w(x_s, \lambda_n^{\text{op}}) \triangleq (1+w) v_\theta(x_s, s, o_{<t,n}, m, \lambda_n^{\text{op}}) - w v_\theta(x_s, s, o_{<t,n}, m, e_\emptyset) \quad (12)
$$

- **w** ≥ 0: guidance scale
- **λ_n^op**: operator-supplied preference
- 第一项: conditional pass
- 第二项: unconditional pass (用 e_∅)

CFG 的隐式 classifier interpretation (Ho & Salimans)：

$$
\nabla_{x_s} \log p^i(\lambda | x_s) = \nabla_{x_s} \log \hat{p}(x_s | \lambda) - \nabla_{x_s} \log \hat{p}(x_s)
$$

所以 sample 近似从 p(x_s | λ) p^i(λ | x_s)^w 来。

### Euler integrator

$$
x_{s-\Delta s} \leftarrow x_s - \Delta s \widetilde{v}_\theta^w(x_s, \lambda_n^{\text{op}}), \quad s \in \{1, 1-\Delta s, \dots, \Delta s\} \quad (27)
$$

10 steps，每步两次 forward pass（conditional + unconditional）。

### Receding-horizon patch scheme

每 ℓ=16 output timesteps 重新 evaluate Eq. 27，history buffer 用之前 prediction 更新。这样 closed-loop rollout 中 conditioning 在每个 patch boundary refresh，缓解 open-loop drift。

---

## 8. Return Labeling Extensions

### 8.1 Context-Residual Returns

Raw G_{n,k} conflate **driving style** with **scenario difficulty**。比如 highway scene 无论 driver 怎么开，offroad penalty 都低。

$$
G_{n,k}^{\text{cr}} \triangleq G_{n,k} - \bar{G}_k(m_n) \quad (19)
$$

- **Ḡ_k(m_n)**: 同 map context m_n 下所有 agent 的 mean return，从 calibration split 估计一次
- Residualize 把 scenario structural component 减掉，留下 behavioral signal

### 8.2 Lane-Centerline Return (alternative map reward)

$$
r_{t,n}^{\text{lc}} \triangleq -|d_{t,n}^\perp| / w_{t,n}^{\text{lane}} \quad (20)
$$

- **d_{t,n}^⊥**: agent n 在时刻 t 到最近 lane centerline 的 perpendicular distance
- **w_{t,n}^lane**: 对应 lane 的 half-width

purely geometric, coordinate-specific map compliance 定义。

### 8.3 Soft Eligibility Gates

这是 paper 中**第二个核心工程 contribution**，直接 fix 了 safety controllability erosion 问题。

**Hard gate baseline**: 只对 clearance < 5m 或 TTC < 6s 的 agent 标 safety label。结果：大多数 safe agent 收不到 safety supervision，generator 看到的 safety signal 几乎为零。

**Soft gate (safety)**:

$$
\widetilde{r}_{t,ij}^{\text{safe}} \triangleq r_{t,ij}^{\text{safe}} \cdot \exp\Big(-\frac{\max(c_{t,ij}, 0)}{\tau_c}\Big) \cdot \exp\Big(-\frac{\max(\text{ttc}_{t,ij}, 0)}{\tau_t}\Big) \quad (21)
$$

- **c_{t,ij}**: agent i, j 在时刻 t 的 pairwise clearance
- **ttc_{t,ij}**: pairwise time-to-collision
- **τ_c = 2.0 m**: clearance decay scale
- **τ_t = 3.0 s**: TTC decay scale

**乘积形式**很关键：clearance 和 TTC **都**要大才能让 risk 消失。一个 distant but fast-approaching agent (low TTC, high clearance) 仍然保留高 risk weight——这 capture 了真实 driving risk 的本质。

**Soft gate (map)**:

$$
\widetilde{r}_{t,n}^{\text{map}} \triangleq r_{t,n}^{\text{map}} \cdot \exp\Big(-\frac{\max(m_{t,n}, 0)}{\tau_m}\Big) \quad (22)
$$

- **m_{t,n}**: agent 到最近 road boundary 的 signed margin
- **τ_m = 1.0 m**

**Effect**: 所有 valid agent 都 receive label，但远离 hazard 的 agent 贡献 negligibly。这 preserve 了 near-threshold agent 的 gradient signal——也就是 reward hacking 和 safety erosion 的根本 fix。

### 8.4 Contrastive Conditioning

同一 flow-time noise 下同时算 steered forward pass (λ_n ~ q*) 和 null forward pass (e_∅)，鼓励网络学 steered-vs-unsteered **difference**而不是 absolute position。提高 closed-loop drift 下的 conditioning robustness。

---

## 9. Experiment Deep Dive

### 9.1 Setup

- WOMD (Ettinger et al. 2021) + WOSAC protocol (Montali et al. 2023)
- 1.1s @ 10Hz history → 8s rollout
- N ≤ 128 agents per scenario
- O_{t,n} ∈ R^9 (pose, heading, planar velocity, bbox)
- K=4 channels, d_h=512, 6 layers, 8 heads
- (μ_0, Σ_0) = (0, I)
- Euler 10 steps, ℓ=16 patch
- 200K training steps

### 9.2 WOSAC Benchmark (Table 1)

| Model | Kinematic ↑ | Interactive ↑ | Map-based ↑ | Realism ↑ | minADE ↓ |
|-------|-------------|---------------|-------------|-----------|----------|
| Constant Velocity | 0.2253 | 0.4327 | 0.4535 | 0.3985 | 7.5148 |
| TrafficBots V1.5 | 0.4304 | 0.7114 | 0.8360 | 0.6988 | 1.8825 |
| SceneDiffuser | 0.4295 | 0.7681 | 0.7756 | 0.7030 | 1.7670 |
| VBD | 0.4169 | 0.7819 | 0.8137 | 0.7200 | 1.4743 |
| TrajTok | 0.4887 | 0.8116 | 0.9207 | 0.7852 | 1.3179 |
| SMART-R1 | 0.4940 | 0.8109 | 0.9194 | 0.7855 | 1.2990 |
| Oracle | 0.5565 | 0.8576 | 0.9593 | 0.8330 | 0.0000 |
| **CNeVA (Ours)** | 0.4732 | 0.7482 | 0.8091 | 0.7145 | 1.8029 |

CNeVA 在 mid-spectrum。Leading 是 tokenized closed-loop imitation models（SMART-R1, TrajTok）。CNeVA 的 gap 集中在 collision 和 off-road，与 open-loop drift 在 8s rollout 上的 error accumulation 一致。

关键 takeaway：CNeVA **没做 closed-loop fine-tuning**，**没做 autoregressive token-prediction**，但已经 mid-spectrum。Top methods 都 lack per-channel controllability interface。

### 9.3 Channel Steering Matrix (Table 2) — Core Controllability Result

CSM diagonal 定义:
$$
\Delta R_k = G_{k,k}^{\text{steered}} - G_k^{\text{base}}
$$

drift-paired evaluation: steered 和 baseline 用同一 flow-time random seed，直接 attribute 差异到 conditioning。

Open-loop, ρ=1, context-residual:

| Channel | Uncond. | CNeVA (w=0) | CNeVA (w=1.5) |
|---------|---------|-------------|----------------|
| Safety  | +0.06   | +0.29       | +0.66 ± 0.10   |
| Map     | +0.06   | +0.24       | +0.61 ± 0.14   |
| Speed   | -3.33   | +3.21       | +8.15 ± 0.07   |
| Accel   | +4.77   | +4.19       | +8.76 ± 0.07   |

**关键观察**:

1. **Unconditional baseline 接近零**——确认 null path by construction 不 steer
2. **Hierarchy 与 tilt-vs-regression prediction 完全一致**: dense kinematic (speed, accel) > sparse semantic (safety, map)
3. **w=0 (latent only) vs w=1.5**: kinematic 已经 strongly steerable 只靠 latent；sparse channel 需 CFG 翻倍才达到 significant response
4. **Null ADE = 1.113 ± 0.011 m, offroad = 32.5%**：与 hard-elig. ablation (1.112 m) 持平，证明 soft eligibility 不 degrade fidelity

### 9.4 Reward Hacking (Figure 4 + Table 5) — Critical Diagnostic

这是 paper 中**最 honest 也最 valuable 的实验**。

**Early-stage ablation (40K steps)**:
- Speed CSM = +51.3 (vs main model +8.15)
- Stall fraction = 75.9% (vs main 65.1%)
- v/v_GT = 61% (vs main 94.7%)

**Reading**: early checkpoint 通过**让 agent stall 不动**来 maximize speed return——因为 speed reward 是 negative penalty，agent 不动 = 没速度 penalty = 高 return！这是经典的 reward hacking。

**Main model**: +8.15 是 physically valid——agent 真的开得更快（retain 94.7% GT speed，stall 只 +0.9pp）。

**Hard-elig. ablation (200K)**:
- Safety CSM = +0.21 ± 0.22（统计上 zero）
- 因为 hard gate 排除 majority agents → generator 看不到 safety signal
- minADE 1.112（与 soft 持平）

**Main model with soft eligibility**:
- Safety CSM = +0.66 ± 0.10（统计显著）
- Stall increase only +0.9pp
- 97.9% GT speed

**这证明 safety steering 真的是 defensive driving**（spacing, yielding）**不是 slow-equals-safe confound**。Context-residual labeling 成功 decorrelate safety from speed。

### 9.5 Map Controllability Across Measures (Figure 8) — Structural Limitation

| Return Measure | ΔR_map |
|----------------|--------|
| Context-residual | +0.61 |
| Physical-offroad | -0.12 |
| Lane-centerline | ≈ -0.002 |

**这是 paper 的 honest limitation**: map controllability 强烈依赖 return 定义。Context-residual 把 scenario baseline 减掉后才能 attribute 到 behavior。Coordinate-specific measures (physical offroad, lane-centerline) 都 fail。

**Implication**: 当前 reward basis 无法做到 lane-keeping 这种 coordinate-level control。需要 richer reward decomposition 把 spatial 和 temporal map compliance 分开。

### 9.6 Pairwise Trajectory Divergence (Table 4)

D(e_a, e_b) at ρ=1, w=1.5 (m):

|         | safety | map  | speed | accel |
|---------|--------|------|-------|-------|
| safety  | -      | 0.10 | 0.63  | 0.49  |
| map     | -      | -    | 0.62  | 0.47  |
| speed   | -      | -    | -     | 0.96  |
| accel   | -      | -    | -     | -     |

Mean D̄ = 0.54 m。

**Reading**: speed 和 accel 产生最大几何分离（0.96 m），safety-map 几乎重叠（0.10 m）——与弱 CSM response 一致。这说明 sparse channel 在 trajectory space 几乎 indistinguishable，是 identifiability problem 的几何体现。

### 9.7 Operating Regime Analysis

ρ=1, w=1.5 是 calibrated point。
ρ=5: 所有 channel 仍 monotone positive:
- ΔR_speed = +14.10
- ΔR_accel = +11.73
- ΔR_safety = +0.85
- ΔR_map = +0.42

但 guardrails degrade:
- Speed-steered stall: +2.1 to +4.6 pp
- Retained speed: 94.7% → 89.1%

**Trade-off**: 大 ρ 增强 steering 但 cost 物理 plausibility。这就是为什么 main results 用 ρ=1。

---

## 10. The "Why This Works" Synthesis

把所有 piece 拼起来，CNeVA 的 conceptual elegance 在于：

1. **Behavior = reward weighting** assumption: 把 unbounded driving style 压缩成 K-dim vector on fixed reward basis
2. **Closed-form conjugate posterior**: 把 inverse-RL 的 hard identification 问题变成一个 O(K) Bayesian regression
3. **Tilt-vs-regression dual view**: 直接 predict identifiability hierarchy，无需 ablation
4. **Mixed channel-mask curriculum**: 让 one-hot inference-time probe 严格 match training support
5. **Soft eligibility gates**: smooth exponential decay 让 near-threshold agent retain gradient，同时 fix reward hacking 和 safety erosion 两个 failure mode
6. **Physical-plausibility guardrails**: CSM 必须配合 stall fraction / v/v_GT / offroad rate 一起读，单独看 CSM 会被 reward hacking 欺骗

---

## 11. Open Questions / Future Directions

Paper 自己提到:
1. **Richer reward decomposition**: separate spatial from temporal map compliance for lane-keeping
2. **Closed-loop training with contrastive objectives**: 缓解 open-loop drift 在 collision/offroad 上的 gap
3. **Stronger latent-conditioning beyond one-hot**: 当前只能 single-axis steering，需要 multi-axis joint steering

我额外想到的:
- **Compositional behavior**: 当前 4 channel 是 additive (λ^T r)，能不能让 channel 之间有 nonlinear interaction?
- **Hierarchical latents**: per-agent λ + per-scenario λ_global，capture 一群 agent 共同 style
- **Reward basis learning**: 当前 K=4 是 hand-pick 的，能不能从数据里 emerge 出来?
- **Closed-loop CFG refresh**: 当前 patch boundary 才 refresh conditioning，能不能让 CFG 在 Euler step 内部 dynamically adapt?
- **Inference-time compositionality**: 类似 ControlNet，能不能把多个 reward channel 作为可组合的 steering signal 在推理时自由搭配?

---

## 12. Reference Links

理论 background:
- Waymo Open Motion Dataset: https://waymo.com/open/data/motion/
- Waymo Open Sim Agents Challenge: https://waymo.com/open/challenges/sim-agents/
- Flow Matching (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- Classifier-Free Diffusion Guidance (Ho & Salimans 2022): https://arxiv.org/abs/2207.12598
- Variational Inference (Jordan 1999): https://mitpress.mit.edu/9780262600321/learning-in-graphical-models/
- Diffusion Models Beat GANs (Dhariwal & Nichol 2021): https://arxiv.org/abs/2105.05233

WOSAC benchmark methods:
- BehaviorGPT (Zhou et al. 2024): https://arxiv.org/abs/2410.07412
- TrafficBots (Zhang et al. 2023): https://arxiv.org/abs/2310.15913
- SceneDiffuser (Jiang et al. 2024): https://arxiv.org/abs/2310.07415
- SMART (Wu et al. 2024): https://arxiv.org/abs/2410.12178
- Trajeglish (Philion et al. 2024): https://arxiv.org/abs/2310.18245
- MotionLM (Seff et al. 2023): https://arxiv.org/abs/2309.16534

Controllability & sim agent methods:
- CTRL-Sim (Rowe et al. 2024): https://arxiv.org/abs/2410.18948
- ScenarioDreamer (Rowe et al. 2025): https://arxiv.org/abs/2503.15423
- SPACeR (Chang et al. 2026): https://openreview.net/forum?id=placeholder
- SAFE-SIM (Chang et al. 2024): https://arxiv.org/abs/2409.04868
- ScenarioDiffusion (Pronovost et al. 2023): https://arxiv.org/abs/2310.15008
- Guided Conditional Diffusion (Zhong et al. 2023b): https://arxiv.org/abs/2306.06687

VBD (Vectorized Behavior Diffusion): https://arxiv.org/abs/2408.16979
GoalFlow (Xing et al. 2025): https://arxiv.org/abs/2503.08649

---

## 13. Final Intuition

Andrej，给你一个 mental model 来 internalize 这篇 paper:

想象每个 agent 的 driving style 是一个 **4-dim knob** (safety, map, speed, accel)，每个维度代表这个 agent 多 care 这个 reward channel。从 logged trajectory 我们看到的是这个 agent 在特定 scene 下的 actual return G_n，但 G_n 本身是 style × scene_difficulty 的混淆。Context-residual 把 scene_difficulty 减掉，剩下 pure style signal，进 conjugate Gaussian update 得到 posterior over knob（Eq. 5）。

Generator 是一个 rectified-flow 速度场，吃 noisy trajectory + history + map + 这个 4-dim knob，输出 clean trajectory velocity。训练时用 mixed channel-mask curriculum 让网络见识过所有 partial knob 配置，inference 时 operator 给任意 partial knob 都不会 outside support。

Soft eligibility gates 是关键修正——把"binary on/off threshold"换成"continuous exponential decay"，让 near-threshold agent 保留 gradient signal，避免了"safety channel 因为大多数 agent 离 collision 太远而收不到信号"和"speed channel 因为 stalling agent 而被 hack"两个 failure mode。

最后，**controllability 的成败完全由 return-shrinkage factor Σ_0(Σ_0 + Σ_noise)^{-1} 决定**——dense, trajectory-level penalty 的 channel SNR 高，易 steer；sparse, event-dependent penalty 的 channel SNR 低，prior-dominated，难 steer。这个 prediction 完全被实验 confirm。

这就是 CNeVA——一个**把 driving style inference 和 controllable generation 用 conjugate Gaussian posterior 优雅串起来**的 framework，同时 honest 地 acknowledge 自己的 limitation（map coordinate-specific）并通过 reward hacking diagnostic 提醒 community: **steering metrics 必须配合 physical-plausibility guardrails 一起读**，否则会被 false positive 欺骗。
