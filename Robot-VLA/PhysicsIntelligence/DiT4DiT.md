---
source_pdf: DiT4DiT.pdf
paper_sha256: 72f9c5fa8337e67f2301e9519c64836b6b9d3cc349ba2fce6036afd30c1df010
processed_at: '2026-08-03T22:38:36-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiT4DiT 用人话讲:让 Robot 学会"脑补"未来再行动

Karpathy, 我用更直觉的方式重新讲一遍,就像在 YouTube lecture 上那样,从 high-level intuition 到为什么这些技术 choices work。

## 1. 一句话总结

**让 robot 先"想象"未来会发生什么,再根据这个想象去决定动作。**

这跟人类做事很像。你要抓一个杯子,脑子其实先"模拟"了手伸过去、碰到杯子、合拢 gripper 的整个过程,然后才真正执行。DiT4DiT 就是把这个"脑补"过程 explicit 做出来,用 video generation model 当 robot 的"想象力引擎"。

## 2. 为什么之前 VLA 不 work?用一个类比

想象你要教一个从没见过物理世界的人玩 Jenga (叠叠乐)。这个人只看过照片 (static image-text pairs),知道"这是木块"、"那是桌子",但完全没看过东西怎么运动、怎么倒塌、怎么碰撞。

你让他去抽木块,他得从零开始学:
- 木块会掉
- 手要稳
- 角度要对
- 力度要合适

这就是当前 VLA 的困境。RT-2, OpenVLA, GR00T 这些 model 的 backbone 在 image-text 上预训练,学到的是 **"what" 和 "where"** — semantic knowledge,比如"这是红色杯子,在桌子上"。但 robot 需要的是 **"how"** — 杯子被推会倒,手指要这样捏才抓得稳,等等。

这种 "how" knowledge 我们叫做 **physical dynamics**,它本质上是个 temporal 现象,static images 学不到。

Video generation models (Wan, Cosmos, HunyuanVideo) 天天在 internet videos 上学预测下一帧,它们自然学到了:
- 物体不会凭空消失 (object permanence)
- 重力让东西往下掉
- 手碰东西会推它 (contact dynamics)
- 先 A 后 B 的因果

所以 hypothesis 就是: **video prediction 是个 better pretraining task for robot control**,因为它逼着 model 学 physics。

Paper Section 3 的实验直接验证了这个 hypothesis:
- Grounding (detect objects): semantic,学不到 physics
- FLARE-style (VLM predict future features): 勉强,但是 VLM backbone 本身物理盲
- **Video generation**: 学到 pixel-level dynamics,convergence 快 7×,data efficient 10×

实验用 Qwen3-2B 和 Cosmos-Predict2.5-2B,trainable params 一样,fair comparison。结果 video generation 一骑绝尘。

参考: https://arxiv.org/abs/2503.14734 (GR00T), https://arxiv.org/abs/2505.15659 (FLARE)

## 3. DiT4DiT 的核心 trick:别用最终生成的视频

这里有个反直觉的设计。你想,既然 video generation 好,那我直接生成 future frames,再让 action policy 看 future frames 来决定动作?

**NO。** 这样做效果很差。Paper 的 ablation (Figure 8b) 显示:
- 1 个 denoise step: 最佳
- 32 个 denoise step: 性能单调下降

为什么?因为完全 denoise 后的 future frame **过度具体**了。

举个例子:video model 预测未来帧,它会生成"5秒后杯子在桌子左边,光线这样,纹理这样"。这些 pixel-level details 其实 irrelevant for action — robot 只需要知道"杯子要往左移",具体纹理不重要。

过度 commit 到 pixel reconstruction 反而丢失了 **actionable information** — 也就是 action 需要的 abstract spatiotemporal structure。

DiT4DiT 的 trick: **hook 中间 layer 的 hidden features**。

公式 9:
$$h_t^{\tau_f} = \mathcal{H}[v_\theta^{\text{video}}](z_{t+1}^{\tau_f}, \tau_f | z_t^0, l)$$

变量解释:
- $\mathcal{H}[\cdot]$: hook operator,从 forward pass 中间 intercept activations
- $v_\theta^{\text{video}}$: video DiT 的 velocity prediction network
- $z_{t+1}^{\tau_f}$: future latent 在 flow timestep $\tau_f$ 时的状态 (noisy intermediate)
- $\tau_f$: 固定的 feature extraction timestep (不做完整 denoise)
- $z_t^0$: clean current observation latent (VAE encode 后)
- $l$: language instruction
- $h_t^{\tau_f}$: 提取出的 hidden features,喂给 action DiT

**关键: 单步 forward pass 就够了**,不用跑完整 video generation loop。这让 action inference 极快 (6Hz real-time)。

Ablation 还发现 (Figure 8a):
- Early layers (2-8): 差,只有 texture
- **Layer 18: 最佳**,middle-deep layer 有 actionable semantics
- Late layers (24-28): 崩了,过度 specialized 于 pixel reconstruction

这跟 LLM 里 "middle layers encode semantics, late layers task-specific" 的直觉一致。

参考: https://arxiv.org/abs/2212.09748 (DiT original)

## 4. Tri-timestep:这个设计是 paper 的精髓

Paper 最聪明的 design 是 **asymmetric tri-timestep**。三个 timestep 分别 control 三个不同的 process,各自最优。

你想想,如果三个 module 用同一个 timestep $\tau$,会怎样?

- Video generation 在 $\tau=0.9$ (高噪声) 时要学 global structure
- Action prediction 在 $\tau=0.9$ 时要学从 noise 开始 denoise 出 action
- Feature extraction 在 $\tau=0.9$ 时给 action 的 condition 是 high-noise 的 representation

三者需求完全 conflict。Video 要看所有 noise levels,feature 要 stable,action 要 focus 在 critical phases (从 noise 开始)。

DiT4DiT 用三个独立 timestep:

### $\tau_v \sim \mathcal{U}[0,1]$ — Video timestep
Uniform sampling,standard diffusion training。Model 要学整个 denoising trajectory,从 pure noise 到 clean frame,所有 noise levels 都要见。

### $\tau_f$ — Feature extraction timestep
**Fixed**。从 $\{0/T, 1/T, ..., T/T\}$ 里挑一个固定值。这选了 backbone 的一个 "operating point"。

为什么 fixed?因为 action policy 需要 consistent input。如果 $\tau_f$ 每次 random,feature 每次都不一样,action model 会 confused。固定后,feature extraction 始终从同一个 "视角" 看 backbone 的 representation,稳定。

实验中发现 $\tau_f$ 对应 single denoise step (即 $\tau_f$ 接近 1,噪声最大时) 效果最好。

### $\tau_a = 1 - \sigma, \sigma \sim \text{Beta}(\alpha, \beta)$ — Action timestep
这里 $\alpha = 1.5, \beta = 1.0$。Beta 分布让 $\sigma$ 偏向 0,所以 $\tau_a$ 偏向 1 (高 noise)。

**Intuition**: Action 生成的难点是从 pure noise 开始 denoise,这些 high-noise phases 是 "hard examples"。Beta sampling 让 model 多看这些 hard cases,类似 focal loss 给难样本更多 weight。

可以想象 Beta(1.5, 1.0) 的概率密度,在 $\sigma \to 0$ 处密度高,对应 $\tau_a \to 1$。Training 更多 sample 这些 critical phases。

### 为什么这三个必须 decoupled?

它们各自的 "difficulty distribution" 不一样:
- Video: uniform,因为要学完整 trajectory
- Feature: deterministic,因为要 stable condition
- Action: biased toward high-noise,因为那是生成的起点

强行用同一个 timestep 就是 suboptimal compromise。Decoupling 后每个 module 在自己最需要的 timestep distribution 上训练。

参考: https://arxiv.org/abs/2210.02747 (Flow Matching), Beta distribution: https://en.wikipedia.org/wiki/Beta_distribution

## 5. Joint training 的 magic:action loss 反过来 shape video backbone

这是 paper 最 deep 的发现。

传统做法: 先 train video backbone (fixed),再 train action policy on top。这叫 **decoupled training**。

DiT4DiT: 同时 train video backbone 和 action policy。叫 **joint training**。

Joint training 的好处不是 "训练更方便"那么浅。真正的 magic 是 **action loss 的 gradient 反过来 regularize video backbone 的 latent space**,让它变得更 actionable。

Ablation (Figure 8c) 用 t-SNE 可视化 hidden features,按 episode 的 temporal phase 上色 (Early/Middle/Late):

- **Decoupled**: features 聚类了,但 temporal phase 在 cluster 内部 entangled,混乱
- **Joint**: features 在每个 task cluster 内部形成 smooth temporal flow,从 Early (蓝) → Middle (黄) → Late (红)

Quantitative: silhouette score 从 0.09 → 0.17,几乎 2× 改善。

**这意味着什么?**

Joint training 强迫 video backbone 学到的 representation 不仅用于 reconstruct frames,还要直接 inform action。所以 features 自然编码了 "我现在在 task 的哪个阶段,接下来该干什么" 这种 temporal progression。

这解释了为什么 DiT4DiT 在 LIBERO-Long (长 horizon task) 上特别强 (97.6% vs 其他 ~95%) — 它的 latent space 有 explicit temporal structure,能 reason about 多阶段 execution。

参考: Silhouette score: https://en.wikipedia.org/wiki/Silhouette_(clustering)

## 6. Flow Matching 数学,用直觉讲

为什么用 Flow Matching 而不是 DDPM?用爬山类比。

DDPM 像走一条曲曲折折的山路: forward process 加噪声,要 reverse 整个 stochastic trajectory,采样要 1000 steps,路径复杂。

Flow Matching 像坐缆车直上去: 用 **linear interpolation** 把 noise 和 data 直线连起来:

$$x_\tau = (1-\tau) \cdot x_0 + \tau \cdot z$$

- $x_0$: clean data point
- $z$: Gaussian noise
- $\tau \in [0,1]$: $\tau=0$ 是 data,$\tau=1$ 是 noise
- $x_\tau$: 线性插值点

这条路径是 **optimal transport** — 最短距离。Velocity (target flow) 是常数:

$$v^*(x_\tau, \tau) = \frac{dx_\tau}{d\tau} = z - x_0$$

注意这个 velocity 跟 $\tau$ 无关!这意味着 model 要学的 function 更简单 (constant target),training 更 stable。

DDPM 的 score function $\epsilon(x_\tau, \tau)$ 依赖于 $\tau$,是 time-varying 的,难学。

训练 loss:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, z, \tau}\left[\|v_\theta(x_\tau, \tau) - (z - x_0)\|^2\right]$$

让 neural network $v_\theta$ 预测 velocity,跟 ground truth $z - x_0$ 算 L2 loss。

采样: 用 Euler 法积分 ODE,从 $\tau=1$ (noise) 倒推到 $\tau=0$ (data):

$$x_{\tau - \Delta\tau} = x_\tau - \Delta\tau \cdot v_\theta(x_\tau, \tau)$$

每步用 predicted velocity 修正一点,几步就够 (DiT4DiT action inference 只用 4 steps)。

**为什么 FM 适合 joint training?**

两个 flow matching loss 可以直接相加:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda \cdot \mathcal{L}_{\text{video}}$$

它们数学形式一样,gradient 可以 straightforward backward 到 video backbone 和 action DiT。如果是 DDPM + flow matching 混合,数学上很 awkward。

参考: https://arxiv.org/abs/2210.02747

## 7. 架构整体流程,走一遍

**Training**:
1. Input: current observation $o_t$, future frames $o_{t+1}$ (ground truth), action $a_t$, robot state $s$, language $l$
2. VAE encode: $o_t \to z_t^0$, $o_{t+1} \to z_{t+1}^0$
3. Video branch: sample $\tau_v$, add noise to $z_{t+1}^0$ 得 $z_{t+1}^{\tau_v}$,predict velocity,算 video loss
4. Feature extraction: sample $\tau_f$ (fixed),fresh noise $\tilde{z}_{t+1}$,forward through video DiT,hook layer 18,得 $h_t^{\tau_f}$
5. Action branch: sample $\tau_a$ (Beta),add noise to $a_t$ 得 $a_t^{\tau_a}$,predict velocity conditioned on $h_t^{\tau_f}$ 和 $s$,算 action loss
6. Total loss = action loss + $\lambda$ × video loss
7. Backward,update video DiT $\theta$ 和 action DiT $\phi$

**Inference**:
1. Input: $o_t$, $s$, $l$
2. VAE encode: $o_t \to z_t^0$
3. Single forward pass through video DiT at fixed $\tau_f$,得 $h_t^{\tau_f}$ (这一步关键,只算一次)
4. Action DiT 从 noise 开始,4 steps Euler integration,每步用 $h_t^{\tau_f}$ 和 $s$ condition,得 action $\hat{a}_t$
5. 同时可以选做 video generation (可选,for visualization)

**Computational bottleneck 在哪?**

Step 3 (video DiT forward) 是大头,但只算一次。Action DiT 是个小 model (DiT-B),4 steps 很快。整体 6Hz,够 closed-loop real-time control。

如果 task 固定,LLM text features 可以 cache,进一步加速。

## 8. 实验结果用人话讲

### LIBERO (Franka arm, 7-DoF)
- DiT4DiT: **98.6%** average
- 击败 π0.5 (96.9%), CogVLA (97.4%), GR00T-N1.5 (94.1%)
- Long-horizon task (LIBERO-Long): 97.6%,显著领先
- 注意是 "from scratch",没用外部 action data,vs baselines 用大规模预训练

### RoboCasa-GR1 (humanoid, 29-DoF, 24 tasks)
- DiT4DiT: **50.8%** average
- GR00T-N1.5: 41.8%
- Qwen3DiT (parameter-matched baseline, 用 Qwen3-VL 替代 video backbone): 36.2%
- **关键对比**: DiT4DiT 比 Qwen3DiT 高 14.6%,直接证明 video backbone > VLM backbone
- 16/24 tasks 最高成功率

### Real-world G1 (Unitree humanoid, 16-DoF, 7 tasks)
- Qwen3DiT 几乎全崩 (多数 0%)
- DiT4DiT 在所有 tasks 领先
- 高精度 task 优势最大:
  - Arrange Flower: 75% vs GR00T 25%
  - Stack Cup: 60% vs 25%
  - Drawer Interaction: 90% vs 40%
- Pre-training data 仅 15% of GR00T-N1.5

### Zero-shot generalization
- 训练只见 bottle,测试 Can/Cup/Milk/Wine: DiT4DiT 54.5% vs Qwen3DiT 32%
- Real-world:
  - Category 变 (杯子换成金属杯): Arrange Flower 70% vs GR00T 10%
  - Object 换 (eggplant 换 corn): 成功
  - Number 变 (3 杯变 4 杯): 50% success

**Intuition**: Video model 学的是 physics (杯子怎么插进 vase),不依赖 surface appearance,所以换 object 还能 work。VLM 学的是 semantic appearance,换 object 就崩了。

## 9. 为什么这个 work?我的 hypothesis

我 (Karpathy 视角) 觉得这 paper 触到一个 deep 的点。

**Representation learning 的本质是学 "invariant features"**。

VLM 通过 image-text contrastive 学到的是 "semantic invariance" — 红杯子在哪儿都是红杯子,跟光照、角度无关。这 great for recognition,但 robot 不需要 recognize 杯子,需要 manipulate 杯子。

Video generation 通过预测 future frames 学到的是 **"physical invariance"** — 不管杯子是红的蓝的,被推了都会倒;不管 gripper 是金属还是塑料,接触物体都会施加力。这些是 action-irrelevant features (颜色、纹理、光照) 之上的 invariant structure。

所以 video backbone 给 action policy 的 condition 是 **physics-aware, appearance-invariant**,正好是 robot 需要的。

**跟 LAD (Latent Action Discovery) 之类工作的 connection**:

最近有些 work 想从 video 里 discover latent actions (比如 LAPA, Ye et al. 2024)。DiT4DiT 走了另一条路 — 不显式 extract actions,而是用 video backbone 的 intermediate features 作为 rich condition,让 action model 自己 learn inverse dynamics。

这两条路殊途同归,但 DiT4DiT 的方法更 flexible,因为 action model 可以用 supervised action data fine-tune,latent action 方法依赖 unsupervised discovery,精度难保证。

参考: https://arxiv.org/abs/2410.11758 (LAPA)

## 10. Limitations 和 future work 我觉得有意思的

### Single egocentric camera
这个 setup 在 bimanual task 容易 occlusion。Robot 自己的手会挡住视线,visual feature temporal continuity 断掉。

Future: 加 wrist camera 或者 tactile。Wrist camera 给 close-up view,tactile 给 contact feedback。但怎么 fuse 进 video DiT backbone 是个开放问题。

### Cross-embodiment scaling
现在只测了 GR1 和 G1。如果能 train 一个 video backbone across many embodiments (Franka, GR1, G1, UR5, ...),可能 emerge 出真正的 generalist robot foundation model。

Video generation 的好处是 embodiment-agnostic — 同一个 video model 可以 generate 不同 robot 的 future,只要 train data 够多。

### Hierarchical planning
现在的 action horizon 是 16 steps。Long-horizon task (LIBERO-Long) 还是靠 chunking。如果能在 video generation 层面做 hierarchical planning — 先 generate "high-level" future (几秒后),再 generate "low-level" future (几帧后),action conditioned on both,可能能 extend 到 minutes-long tasks。

### Video-conditioned RL
现在是 imitation learning (behavior cloning)。如果加 RL fine-tuning,用 video generation 作为 world model 来 rollout,可能 sample efficiency 更好。Dreamer 系列 work 已经证明 world model + RL 强大,DiT4DiT 提供了 high-quality world model 基础。

参考: Dreamer V3: https://arxiv.org/abs/2301.04104

## 11. 跟其他工作的关系,梳理一下

### 跟 mimic-video (Pai et al., 2025) 的区别
Mimic-video 是最接近的工作。区别:
- Mimic-video: pre-trained video backbone (frozen) + action decoder,condition 在 partially denoised video latents
- DiT4DiT: **joint training**,video backbone 也 update,action loss 反过来 regularize video backbone

这导致 latent space 性质完全不同。Mimic-video 的 video backbone latent 是为 video generation 优化的,action decoder 要 adapt 它。DiT4DiT 的 latent 是 joint optimized,天生 actionable。

实验 ablation 里 DiT4DiT 在 single denoise step 的 sensitivity 比 mimic-video 更 extreme (monotonic decline),作者认为是 joint training 的 action loss 在 first step 就 regularize latent。

### 跟 Cosmos Policy (Kim et al., 2026) 的区别
Cosmos Policy 直接 fine-tune video diffusion model 让它 output actions 和 future values,encode 成 latent frames。这是 unified 但失去 modularity。

DiT4DiT 用 separate action DiT,保留 modularity,可以独立 optimize action module。Action DiT 也可以 swap (从 GR00T-N1 改编)。

### 跟 π0 (Black et al., 2024) 的关系
π0 也是 flow matching for action generation,但 backbone 是 VLM (PaLI-3 之类)。DiT4DiT 把 backbone 换成 video DiT,展现 video prior 的优势。

π0.5 (Intelligence et al., 2025b) 在 π0 基础上加 web data 和 open-world generalization,但是 still VLM-based。

DiT4DiT 在 LIBERO-Long 上超 π0.5 (97.6% vs 92.4%),说明 video prior 对 long-horizon 特别有帮助。

参考:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- mimic-video: https://arxiv.org/abs/2512.15692
- Cosmos Policy: https://arxiv.org/abs/2601.16163

## 12. Final intuition:这 paper 在我脑里是什么样子

我把它想成一个"导演 + 演员"的 setup:

- **Video DiT = 导演**: 看着当前场景和剧本 (language),脑子里规划"接下来镜头会怎么走" — 物体怎么动、手怎么伸、场景怎么变化。它不需要精确到每个像素,只要 high-level 的动作设计。
- **Hook layer 18 = 导演的 notes**: 不看最终成片 (太具体),看导演工作时的草稿和思路。这些草稿里有 actionable information — "这里杯子要往左,手要合拢"。
- **Action DiT = 演员**: 拿着导演的 notes,加上自己的 body state (proprioception),决定具体怎么动 joint。它做 inverse dynamics — "要实现这个效果,我关节该怎么转"。
- **Joint training = 导演和演员一起排练**: 不是导演先拍完再让演员跟着演,而是边拍边演边 feedback。演员说"你这个 notes 我执行起来不好用",导演就调整 notes 让它更 actionable。最后导演的 notes 既服务于画面,也服务于演员执行。

这个 analogy 还能解释 tri-timestep:
- Video timestep $\tau_v$: 导演练习各种 planning granularities (从粗到细)
- Feature extraction timestep $\tau_f$: 固定看一个 planning stage (e.g. 草稿阶段),稳定输出
- Action timestep $\tau_a$: 演员重点练习从"什么都没有"开始执行的过程 (high noise = 从零开始)

## 13. 一些没在 paper 里明说但我觉得 important 的点

### 为什么 VLM + future frame prediction (FLARE) 不如 video generation?
FLARE 用 VLM attend to future frame embeddings。但 VLM 本身在 static images 上训练,它的 attention 机制是为 semantic alignment 设计的,不是为 temporal dynamics 设计的。即使你给它 future frame,它也只会 extract "这是什么",不会 extract "这怎么变化的"。

Video DiT 是为了 predict next frame 训练的,它的中间 features 天生 encode temporal transformation — "从 state A 怎么变成 state B"。这种 transformation knowledge 直接 actionable。

### 为什么 hook middle layer 而不是 last layer?
DiT 的 layer progression 大概是:
- Early: low-level texture features
- Middle: spatiotemporal transformation features (最 actionable!)
- Late: pixel-level reconstruction features (over-committed)

Action 要的是 "how things change",不是 "what things look like",所以 middle layer 最合适。这跟 LLM 里 "middle layers encode semantics" 的发现类似。

参考: https://arxiv.org/abs/2402.17716 (Anthropic 的 middle layer interpretability work,大意类似)

### 为什么 action timestep 用 Beta(1.5, 1.0) 而不是其他?
Beta(1.5, 1.0) 让 $\sigma$ 偏向 0,所以 $\tau_a = 1-\sigma$ 偏向 1。这给 high-noise phases (action 生成的起点) 更多 training。

Alpha=1.5 比 alpha=1.0 稍微多偏向 0 一点,但 not too extreme。如果 alpha 太大 (e.g. 5.0),model 几乎只看 high noise,low noise phases (fine refinement) 训练不足,action 精度差。Alpha=1.5 是个温和的 bias。

Paper 没做 Beta parameter 的 ablation,我觉得是个 future direction。可能 Beta(2.0, 1.0) 或者 Beta(1.5, 1.5) 会有不同 trade-off。

### 6Hz 够用吗?
对大多数 manipulation task 够。Human manipulation 频率大概 5-10Hz (你抓杯子不需要 100Hz 控制)。但高 dynamic task (juggling, catching) 可能需要更高频率。

Config 里 Num Inference Timesteps = 4 for action,如果减到 2 可能能到 10Hz+,但精度可能下降。Trade-off 没在 paper 里 explore。

## 14. 我觉得这 paper 真正的贡献

不是 98.6% 或 50.8% 这些数字。真正的 contribution 是 establish 一个 paradigm:

**Video generation 是 robot policy 的有效 pretraining objective**。

这听起来简单,但 implications 巨大:
- Internet 上有海量 videos,但 robot action data 极稀缺
- 如果 video pretraining 有效,我们可以用 YouTube videos 训 robot foundation model
- 不需要每条 data 都 paired with action label

这跟 LLM 用 next-token prediction pretrain on internet text 然后 fine-tune 到各种 task 一个道理。DiT4DiT 是把这个思路 port 到 robot — 用 video prediction pretrain on internet videos,然后 fine-tune 到 robot action。

只不过 video prediction 比 next-token prediction 复杂得多 (要学 spatiotemporal dynamics),而且现在的 video models 还不够好。但随着 Wan, Cosmos, Sora 这些 video foundation model 进步,这个 paradigm 的 ceiling 会越来越高。

参考:
- Wan: https://arxiv.org/abs/2503.20314
- Cosmos: https://arxiv.org/abs/2511.00062
- Sora: https://openai.com/sora (技术 report 有限)

## 15. 一句话 take-away

**让 robot 学会先"脑补"未来 (video generation),再根据脑补的中间过程决定动作 (action prediction)。用 tri-timestep decoupling 和 joint training 让脑补过程天生 actionable,从而 bypass video generation 的计算瓶颈,同时获得 physics-aware representations。**

这 paper 是我 (Karpathy 视角) 看到 video-conditioned robot control 里最 clean 的 design,tri-timestep scheme 和 joint training 的 intuition 都很 elegant。我期待看到 cross-embodiment scaling 和 RL integration 的后续 work。

---

附: 一些可能有用的 links

- DiT4DiT project page: https://dit4dit.github.io/
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748
- Cosmos-Predict: https://arxiv.org/abs/2511.00062
- GR00T N1: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- mimic-video: https://arxiv.org/abs/2512.15692
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- FLARE: https://arxiv.org/abs/2505.15659
- LAPA (latent action): https://arxiv.org/abs/2410.11758
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523
- Dreamer V3: https://arxiv.org/abs/2301.04104
- OpenVLA: https://arxiv.org/abs/2406.09246
- CogVLA: https://arxiv.org/abs/2508.21046
- UniVLA: https://arxiv.org/abs/2505.06111
- HunyuanVideo: https://arxiv.org/abs/2412.03603

Karpathy, 我讲完了。这个 paper 我觉得是 foundation model for robotics 一个重要 milestone,虽然现在看 50.8% 在 RoboCasa 还不算惊艳,但 paradigm 是对的,后续 scaling 会很有意思。

---

# DiT4DiT 深度解析:Video Generation 作为 Robot Policy 的 Scaling Proxy

这篇 paper 来自 Mondo Robotics 和 HKUST(GZ),核心贡献是把 video generation 作为 robot control 的 foundation backbone,通过 Dual-DiT 架构和 Tri-timestep scheme 实现 end-to-end joint training。我下面详细讲解技术细节,构建你的 intuition。

## 1. 核心问题与 Motivation

当前 VLA 模型(RT-2, OpenVLA, GR00T, π family)继承自 VLM 的 backbone,这些 backbone 主要在 static image-text pairs 上预训练,导致 **spatiotemporal structure 和 physical dynamics 完全依赖下游 action data fine-tuning**。这有几个根本问题:

- Image-text pretraining 学到的是 "what" 和 "where" 的 semantic alignment,不是 "how things move"
- 需要 thousands of hours 的 action data 才能 learn low-level physics
- Sample efficiency 极差,scaling 受限于 action-labeled data

Video generation models (Wan, Cosmos, HunyuanVideo) 通过预测 internet-scale 的 future frames,自然 internalize 了:
- Motion priors (物体如何运动)
- Causal structure (先 A 后 B)
- Implicit physics (重力, 碰撞, 接触)
- Temporal coherence (状态如何 transition)

**核心 hypothesis**: video generation 不仅仅是 auxiliary task,它是 robot policy 的 effective scaling proxy。

Paper Section 3 用实验验证这个 hypothesis (Figure 1):
- Grounding (object detection auxiliary): semantic-centric
- FLARE-style (VLM latent feature prediction with future frames): implicit world modeling 但缺乏 pixel-level dynamics
- Video generation: physically plausible future dynamics

结果:Video generation objective 比 FLARE 快 **7× convergence**, **10× data efficient**,scaling curve 更陡峭。这个实验用 Qwen3-2B 和 Cosmos-Predict2.5-2B 保证 trainable params 一致,decoupled pre-training + downstream action expert fine-tuning。

## 2. Flow Matching 数学基础

DiT4DiT 用 Flow Matching (Lipman et al., 2022) 而不是 DDPM,原因是 FM 基于 optimal transport 的 linear interpolation,training 更稳定,sampling 更直接。

**Interpolation path** (公式 1):

$$x_\tau = (1-\tau) \cdot x_0 + \tau \cdot z, \quad \tau \in [0,1]$$

变量解释:
- $x_0 \sim p_{\text{data}}$: clean data point (可以是 video latent 或 action)
- $z \sim \mathcal{N}(0, I)$: standard Gaussian noise
- $\tau$: flow timestep,$\tau=0$ 时是 clean data,$\tau=1$ 时是 pure noise
- $x_\tau$: interpolation point

这个 linear path 是 optimal transport displacement map,与 DDPM 的 forward SDE 不同 — DDPM 用累积噪声,FM 用线性插值,所以 trajectory 更短更直。

**Target velocity** (公式 2):

$$v^*(x_\tau, \tau) = \frac{dx_\tau}{d\tau} = z - x_0$$

这是 ground truth flow — 从 $x_0$ 到 $z$ 的恒定速度向量。注意这个 velocity 与 $\tau$ 无关 (因为是 linear path),所以 FM 比 DDPM 的 score function 更容易学。

**Training objective** (公式 3):

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, z, \tau}\left[\left\|v_\theta(x_\tau, \tau) - (z - x_0)\right\|^2\right]$$

变量:
- $v_\theta(x_\tau, \tau)$: neural network 预测的 velocity field,参数 $\theta$
- $\tau \sim \mathcal{U}[0,1]$: uniform sampled timestep
- 目标是最小化 predicted velocity 和 target velocity 的 L2 距离

**Inference: ODE integration** (公式 4-5):

$$\frac{dx}{d\tau} = v_\theta(x, \tau), \quad x_1 \sim \mathcal{N}(0,I)$$

$$x_{\tau - \Delta\tau} = x_\tau - \Delta\tau \cdot v_\theta(x_\tau, \tau)$$

- 从 $\tau=1$ (noise) 积分到 $\tau=0$ (data)
- First-order Euler discretization
- $N$ steps,$\Delta\tau = 1/N$
- 每步用 predicted velocity 更新 latent

**Intuition**: FM 就是学习一个 vector field,把 noise distribution transport 到 data distribution。Linear path 让 transport 路径最短,training signal 最 clean。

参考: https://arxiv.org/abs/2210.02747

## 3. Problem Formulation

传统 VLA: $\pi_\theta(a_t | o_t, l)$ — 直接 observation → action mapping。

DiT4DiT 采用 **predict video dynamics + inverse dynamics** 范式:

$$o_{t+1} \sim p_v(\cdot | o_t, l) \tag{6}$$

$$a_t \sim p_a(\cdot | o_t, \mathcal{H}(o_{t+1}^{\tau_v})) \tag{7}$$

变量:
- $p_v$: video generation distribution
- $p_a$: action generation distribution
- $o_{t+1}^{\tau_v}$: future frame 在 flow step $\tau_v$ 的 intermediate state
- $\mathcal{H}$: hidden state extraction operator
- 当 $\tau_v \to 0$,$o_{t+1}^{\tau_v} \to o_{t+1}$ (clean future)

**Joint distribution** (公式 8):

$$o_{t+1}, a_t \sim p_{va}(\cdot | o_t, l)$$

DiT4DiT 要 model joint distribution,不是 marginal。这是与 mimic-video (Pai et al., 2025) 的关键区别 — mimic-video 是 multi-stage (先 video backbone 固定,再 train action decoder),DiT4DiT 是 end-to-end joint training。

## 4. Dual-DiT Architecture 详解

### 4.1 Video DiT

Backbone: **Cosmos-Predict2.5-2B** (Ali et al., 2025) 作为初始化。

架构组成:
1. **Causal video VAE**: 把 high-dimensional pixel space $o_t, o_{t+1} \in \mathbb{R}^{T \times 3 \times H \times W}$ 压缩到 compact latent space $z_t^0, z_{t+1}^0$,做 spatial 和 temporal downsampling
2. **Video Diffusion Transformer**: flow-prediction parameterization,conditioned on language instructions via multi-layer embeddings from Cosmos-Reason1 (Azzolini et al., 2025)

关键创新 — **forward hook 机制提取中间 features** (公式 9):

$$h_t^{\tau_f} = \mathcal{H}[v_\theta^{\text{video}}](z_{t+1}^{\tau_f}, \tau_f | z_t^0, l) \tag{9}$$

变量:
- $\mathcal{H}[\cdot]$: hook operator,intercept 中间 hidden activations
- $v_\theta^{\text{video}}$: video velocity network
- $z_{t+1}^{\tau_f}$: future latent 在 flow step $\tau_f$ 的状态
- $\tau_f$: 固定的 feature extraction timestep
- $z_t^0$: clean current observation latent
- $l$: language goal
- $h_t^{\tau_f}$: 提取的 hidden features,作为 action DiT 的 condition

**为什么用中间 features 而不是 reconstructed frames?**
- Reconstructed frames 是 pixel-level specific,over-commit 到 specific visual details
- 中间 denoising features 编码 abstract, actionable semantics
- Single forward pass 足够,bypass multi-step video generation bottleneck
- Feature extraction 在 fixed $\tau_f$ 保证 representation 稳定一致

Config: Extract Layer = 18 (从 ablation 发现),Hidden Feature Dim = 2048,Attention 用 flash_attention_2。

### 4.2 Action DiT

从 **GR00T-N1** (Bjorck et al., 2025) 改编。架构细节:

- **DiT-B** backbone,Hidden Size 2560,16 layers
- **Adaptive Layer Normalization (AdaLN)** (Peebles & Xie, 2023): inject diffusion timestep 信息
- **Cross-attention layers**: attend to visual features $h_t^{\tau_f}$ 从 video backbone,Cross Attention Dim 2048
- **Input sequence** = concatenation of:
  - Proprioceptive state embeddings (State Dim 64)
  - Encoded noisy action trajectories (Action Dim 32, Action Horizon 16, Future Action Window 15)
  - Learnable "future tokens": compressed queries for motion planning
- **Output**: linear projection 预测 velocity vector field
- Dropout 0.2, Final Dropout True, Interleave Self Attention True

**Intuition**: Action DiT 是 inverse dynamics model — 给定 current observation 的 visual features 和 robot state,预测实现 future dynamics 的 action。Cross-attention 让 action head fuse spatiotemporal visual context with robot state。

参考:
- Cosmos-Predict: https://arxiv.org/abs/2511.00062
- GR00T N1: https://arxiv.org/abs/2503.14734
- DiT: https://arxiv.org/abs/2212.09748

## 5. Tri-timestep Scheme (核心创新)

这是 paper 最 critical 的设计 — **asymmetric tri-timestep** decouples video generation, feature extraction, 和 action generation。

### 5.1 为什么需要 decoupling?

三个模块有 conflicting requirements:
- **Video generation**: 需要 learn full denoising trajectory (所有 noise levels)
- **Feature extraction**: 需要 deterministic, consistent representations across iterations
- **Action generation**: 需要 focus on critical control phases (从 noise 到 action 的关键阶段)

如果用同一 timestep,action module 会受 video denoising 的 noise level 干扰,feature extraction 会不稳定。

### 5.2 三个 timestep

**$\tau_v \sim \mathcal{U}[0,1]$ (Video)**: Standard uniform sampling,expose 所有 noise levels,learn full denoising trajectory。这与 Cosmos-Predict 原生训练一致。

**$\tau_f$ (Feature Extraction)**: Fixed deterministic timestep,从 $\{0/T, 1/T, ..., T/T\}$ 中选。这个 fixed point 选择 backbone 的 "operating point":
- Early diffusion stages: global structure
- Later stages: fine-grained details
- 固定 $\tau_f$ 让 latent representations 稳定,downstream action prediction 收到 consistent input

实验中 $\tau_f$ 在 single denoise step 时最佳 (Section 5.4 ablation)。

**$\tau_a = 1 - \sigma, \sigma \sim \text{Beta}(\alpha, \beta)$ (Action)**: Biased continuous-time sampling。
- $\alpha = 1.5, \beta = 1.0$
- $\sigma$ 偏向 0,所以 $\tau_a$ 偏向 1 (高 noise)
- 这把更多 training capacity 分配给 action 生成的关键阶段 (从 noise 开始 denoise)
- 类似于 flow matching 的 "difficulty weighting" — 高 noise 区域是 action 生成的起点

**Intuition**: Action module 从 pure noise 开始 denoise 到 clean action,所以 high-noise ($\tau_a \approx 1$) 的训练样本更重要。Beta sampling 让 model 更多看到这些 critical phases。

### 5.3 Algorithm 1 (Training) 详解

```
Require: o_t, o_{t+1}, a_0, s, l, action mask M

// Video DiT Forward
1. z_t^0 = VAE_enc(o_t)           // Encode observation
2. z_{t+1}^0 = VAE_enc(o_{t+1})   // Encode future frames
3. τ_v ~ U[0,1]                   // Sample video timestep
4. z ~ N(0,I)                     // Sample video noise
5. z_{t+1}^{τ_v} = (1-τ_v)·z_{t+1}^0 + τ_v·z   // Noisy future latent
6. v̂_video = v_θ^video(z_{t+1}^{τ_v}, τ_v | z_t^0, l)
7. v*_video = z - z_{t+1}^0
8. L_video = ||v̂_video - v*_video||^2

// Extract Hidden States
9. τ_f ~ U{0/T, 1/T, ..., T/T}
10. z̃_{t+1} ~ N(0,I)
11. h_t^{τ_f} = H(θ, z̃_{t+1}, τ_f, z_t^0, l)

// Action DiT Forward
12. σ ~ Beta(α,β); τ_a = 1-σ
13. ε ~ N(0,I)
14. a_t^{τ_a} = (1-τ_a)·a_t^0 + τ_a·ε
15. v̂_action = v_φ^action(a_t^{τ_a}, τ_a | h_t^{τ_f}, s)
16. v*_action = ε - a_t^0
17. L_action = ||(v̂_action - v*_action) ⊙ M||^2 / ||M||_1

// Backward
18. L_total = L_action + λ·L_video
19. Update θ, φ via ∇L_total
```

注意:
- Video noise $z$ 和 action noise $\epsilon$ 是独立的 samples
- Feature extraction 用新的 noise $z̃_{t+1}$,不是 video denoising 的 trajectory
- Action loss 有 mask $M$ 处理 missing action dimensions
- $\lambda$ 平衡两个 loss (config 中没明确,可能 1.0)

### 5.4 Joint Loss (公式 10)

$$\mathcal{L}_t^{\text{total}} = \underbrace{\mathbb{E}_{\tau_a, \epsilon}\left[\left\|v_\phi^{\text{action}}(a_t^{\tau_a}, \tau_a | h_t^{\tau_f}, s) - (\epsilon - a_t^0)\right\|^2\right]}_{\text{Action Flow Matching Loss}}$$
$$+ \lambda \underbrace{\mathbb{E}_{\tau_v, z}\left[\left\|v_\theta^{\text{video}}(z_{t+1}^{\tau_v}, \tau_v | z_t^0, l) - (z - z_{t+1}^0)\right\|^2\right]}_{\text{Video Flow Matching Loss}}$$

变量全解:
- $v_\phi^{\text{action}}$: action DiT velocity network,参数 $\phi$
- $v_\theta^{\text{video}}$: video DiT velocity network,参数 $\theta$
- $a_t^{\tau_a}$: flow step $\tau_a$ 时的 noisy action
- $h_t^{\tau_f}$: flow step $\tau_f$ 时从 video backbone 提取的 hidden features
- $s$: robot proprioceptive state
- $\epsilon$: action Gaussian noise
- $a_t^0$: clean target action
- $z_{t+1}^{\tau_v}$: flow step $\tau_v$ 时的 noisy future latent
- $z_t^0$: clean current observation latent
- $l$: language goal
- $z$: video Gaussian noise
- $z_{t+1}^0$: clean future latent
- $\lambda$: scalar balancing coefficient

**为什么 joint training 比 decoupled 好?** (Section 5.4 ablation)
- Joint training 让 action loss regularize video backbone 的 latent space
- Video backbone 学到的 features 更 actionable
- t-SNE 显示更清晰的 temporal clustering (Early/Middle/Late phase 分离)
- Silhouette score 从 0.09 (decoupled) 提升到 0.17 (joint) — 几乎 2× 改进
- 这让 action policy 能 reason about long-horizon execution 和 state transitions

## 6. Inference Procedure (Algorithm 2)

DiT4DiT inference 是 decoupled sampling,可以 synthesize future video, predict action, 或两者同时。

### 6.1 Video DiT Sampling

```
1. z_t^0 = VAE_enc(o_t)
2. ẑ_{t+1} ~ N(0,I)              // Initialize from noise
3. Δτ_v = 1/N_v
4. for i = 0, 1, ..., N_v-1:
5.     τ_v = 1 - i·Δτ_v
6.     v̂ = v_θ^video(ẑ_{t+1}, τ_v | z_t^0, l)
7.     ẑ_{t+1} = ẑ_{t+1} - Δτ_v·v̂   // Euler step backward
8. ô_{t+1} = VAE_dec(ẑ_{t+1})
```

### 6.2 Action DiT Sampling (关键)

```
1. â_t ~ N(0,I)                  // Initialize action from noise
2. z̃_{t+1} ~ N(0,I)             // New noise for feature extraction
3. h_t^{τ_f} = H(θ, z̃_{t+1}, τ_f, z_t^0, l)   // Single forward pass
4. Δτ_a = 1/N_a
5. for i = 0, 1, ..., N_a-1:
6.     τ_a = 1 - i·Δτ_a
7.     ṽ = v_φ^action(â_t, τ_a | h_t^{τ_f}, s)
8.     â_t = â_t - Δτ_a·ṽ       // Euler step backward
9. return â_t, ô_{t+1}
```

**Critical insight**: Action conditioning 只需 single forward pass through video backbone,不需要 multi-step video generation loop。这 bypass 了 video generation 的 computational bottleneck,实现 6Hz real-time control。

Action DiT inference 用 Num Inference Timesteps = 4 (从 config),所以 action sampling 4 步,极快。

## 7. 实验结果深度分析

### 7.1 LIBERO Benchmark (Table 1)

DiT4DiT (from scratch) 达到 **98.6% average** SOTA:

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 |
| GR00T-N1.5 | 96.2 | 94.0 | 96.0 | 90.0 | 94.1 |
| Qwen3DiT (from scratch) | 98.0 | 98.8 | 96.0 | 93.6 | 96.6 |
| **DiT4DiT (from scratch)** | **98.4** | **99.6** | **98.6** | **97.6** | **98.6** |

关键观察:
- LIBERO-Object (99.6%) 和 LIBERO-Goal (98.6%) 最高 — 强 generalization 到 unseen objects 和 instructions
- LIBERO-Long (97.6%) 显著领先 — explicit spatiotemporal dynamics modeling 帮助 multi-stage tasks
- "from scratch" 意味着没用 benchmark 外 action data,vs baselines 用大规模预训练
- Qwen3DiT (相同 action DiT + Qwen3-VL) 96.6% — DiT4DiT 用 video backbone 替换 VLM backbone,提升 2%

### 7.2 RoboCasa-GR1 (Table 2)

24 个 challenging tasks,**50.8% average** SOTA:

| Method | Average |
|--------|---------|
| GR00T-N1.5 | 41.8 |
| GR00T-N1.6 | 40.8 |
| Qwen3DiT (from scratch) | 36.2 |
| **DiT4DiT (from scratch)** | **50.8** |

- 比 GR00T-N1.5 高 9%, 比 GR00T-N1.6 高 10%
- 比 parameter-matched Qwen3DiT 高 14.6% — 这是关键 ablation!
- 16/24 tasks 最高 success rate
- 需要精确 spatial coordination 的 task 提升最大:
  - CanToDrawerClose: 74% vs 56%
  - FromCuttingboardToPan: 76% vs 62%
  - FromPlateToPan: 68% vs 56%

**Intuition**: GR1 是 29-DoF humanoid,task 复杂度高。Video backbone 的 implicit physics priors 比 VLM 的 semantic priors 更适合 low-level continuous control。

### 7.3 Real-world G1 (Figure 5)

7 个 tasks,Unitree G1 (16-DoF),仅 ego-view camera:

| Task | GR00T-N1.5 | Qwen3DiT | DiT4DiT |
|------|------------|----------|---------|
| Pick & Place | ~40% | <10% | ~60% |
| Arrange Flower | 25% | 0% | 75% |
| Stack Cup | 25% | 0% | 60% |
| Insert Plate | ~30% | 0% | ~55% |
| Box Packing | ~20% | 0% | 50% |
| Move Spoon | 15% | 0% | 40% |
| Drawer Interaction | ~40% | 0% | 90% |

- Qwen3DiT 几乎完全 collapse (most tasks 0%) — static image-text priors 无法 ground 到 3D physics
- DiT4DiT 即使 pre-training data 仅 15% of GR00T-N1.5,仍大幅领先
- 高精度 tasks (Arrange Flower 75% vs 25%, Stack Cup 60% vs 25%) 提升最大 — video backbone 保留 fine-grained visual details

### 7.4 Zero-shot Generalization (Figure 7)

**Simulation** (训练仅 bottle,测试 Can/Cup/Milk/Wine):
- ToDrawerClose: DiT4DiT 54.5% vs Qwen3DiT 32%
- ToCabinetClose: 34.0% vs 24.5%
- ToMicrowaveClose: 30.5% vs 17.0%

**Real-world** 4 scenarios:
- Category (material/shape change): Arrange Flower 70% vs Qwen3DiT 0% vs GR00T 10%
- Object substitution: Box Packing with corn instead of eggplant
- Quantity variation: Stack Cup with 4 cups instead of 3 — DiT4DiT 50%

**Intuition**: Video generation 学习的是 physics-aware representations,不依赖 surface-level visual features。当 object 外观变化,underlying physics interaction 保持不变,所以 model 能 generalize。

### 7.5 Efficiency (Table 3)

| Model | Trainable Params | Deploy Freq |
|-------|------------------|-------------|
| GR00T-N1.5 | 2.7B | 13Hz |
| Qwen3DiT | 2.3B | 9Hz |
| DiT4DiT | 2.2B | 6Hz |

- DiT4DiT 参数最少 (2.2B)
- 6Hz control rate (vs 13Hz GR00T) — video backbone 计算开销
- 但 single forward pass for feature extraction,bypass multi-step video generation
- LLM features 可 pre-extract cache 进一步加速

## 8. Ablations 的启示

### 8.1 Feature Extraction Layer (Figure 8a)

5 tasks from RoboCasa-GR1:
- Layer 2-8: 差 — 低级 texture,缺乏 actionable semantics
- Layer 18: 最佳 — middle-deep blocks 平衡 spatiotemporal physics 和 high-level scene understanding
- Layer 24-28: collapse — 过度 specialized 于 pixel-level reconstruction
- Average all layers: 略低于 layer 18

**Intuition**: DiT 的不同 layer 编码不同 abstraction level。Early layers = texture, middle = physics-aware semantics, late = pixel reconstruction。Action prediction 需要中间的 actionable representation。

### 8.2 Denoise Steps (Figure 8b)

- 1 step: 最佳
- 32 steps: monotonic decline
- Excessive denoising 让 hidden states over-commit 到 specific reconstructed future
- Joint training 让 action loss 在 first step 就 regularize latent space
- 这 validate single forward pass 足够 high-frequency control

### 8.3 Joint vs Decoupled (Figure 8c)

t-SNE of hidden features colored by execution phase (Early/Middle/Late):
- Decoupled: fragmented, entangled temporal distributions
- Joint: smooth temporal flows within each task cluster
- Silhouette score: 0.09 → 0.17 (2× improvement)

**Intuition**: Joint training forces video backbone embed continuous, physics-aware temporal progression。Action loss 的 gradient 直接 shape video backbone 的 latent space,让它 actionable。

## 9. 与相关工作的对比

### 9.1 VLA 模型

- **RT-2, OpenVLA, GR00T**: 继承 VLM backbone,semantic-centric
- **π0, π0.5**: flow matching for action,但 backbone 仍是 VLM
- **CogVLA**: instruction-driven routing,sparsification
- **UniVLA**: task-centric latent actions

这些都没有用 video generation 作为 backbone。

### 9.2 Video Generation in Robotics

- **mimic-video** (Pai et al., 2025): 最接近的工作。Pre-trained video backbone + separate flow-matching action decoder,conditioned on partially denoised video latents at intermediate flow time。
  - 区别: DiT4DiT 是 joint training,mimic-video 是 multi-stage
  - DiT4DiT 让 action model learn 跨不同 video generation stages 提取 features
  
- **Cosmos Policy** (Kim et al., 2026): fine-tune video diffusion 直接输出 actions 和 future values,encode 成 contiguous latent frames
  - 区别: DiT4DiT 用 separate action DiT,explicit decoupling

- **VidAR** (Feng et al., 2025): embodied video diffusion for bimanual manipulation
- **Motus** (Bi et al., 2025): unified latent action world model
- **UniVLA** (Cen et al., 2025): autoregressive action world model

### 9.3 Implicit World Modeling

- **FLARE** (Zheng et al., 2025): VLM + learnable queries,align with future observation latents
  - 问题: 没有真正 diffusion process,pixel-level dynamics 缺失
  - DiT4DiT 实验证明 FLARE-style 不如 video generation (Section 3)

参考:
- mimic-video: https://arxiv.org/abs/2512.15692
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- FLARE: https://arxiv.org/abs/2505.15659
- UniVLA: https://arxiv.org/abs/2505.06111

## 10. Training Configuration 细节

从 Table 4:

**Video DiT**:
- Base: Cosmos-Predict2.5-2B
- Hidden Feature Dim: 2048
- Extract Layer: 18
- flash_attention_2

**Action DiT**:
- DiT-B, Hidden 2560
- 16 layers
- Action Dim 32, State Dim 64
- Action Horizon 16, Future Action Window 15
- Repeated Diffusion Steps (train): 4
- Noise: Beta(α=1.5, β=1.0), s=0.999
- Num Timestep Buckets: 1000
- Num Inference Timesteps: 4

**Training**:
- Per Device Batch Size: 8
- 32 GPUs
- Max Train Steps: 100,000
- VGM LR: 1e-5
- Action Model LR: 1e-4
- LR Scheduler: cosine_with_min_lr
- Min LR: 5e-7
- Gradient Clipping: 1.0
- AdamW, β1=0.9, β2=0.95, ε=1e-8
- Weight Decay: 1e-8

注意 action LR (1e-4) 比 VGM LR (1e-5) 高 10× — action module 需要更快适应,video backbone 是 fine-tune。

## 11. Dataset Configuration (Table 5)

| Dataset | Episodes | Embodiment | DoF |
|---------|----------|------------|-----|
| Fourier_GR1_Unified_1K | 24,000 | GR1 | 29 |
| Fourier_GR1_Pretrain_10K | 241,450 | GR1 | 29 |
| LIBERO | 1,693 | Franka | 7 |
| Real Robot | 1,400 | G1 | 16 |

Real-world pipeline:
1. Pre-train on Fourier_GR1_Pretrain_10K (241,450 episodes) — 仅 15% of GR00T-N1.5 data
2. Fine-tune on Real Robot (1,400 episodes, 200/task × 7 tasks)

## 12. Limitations

1. **Single egocentric camera**: 容易 occlusion,complex bimanual tasks 时 arms/objects block 视线
2. **Pre-training data scale**: 仅 15% of GR00T-N1.5,future work scale across embodiments
3. **6Hz control rate**: video backbone 计算开销,需要 optimization

## 13. Core Insights 总结

1. **Video generation 是 robot policy 的 effective scaling proxy** — 比 semantic-centric baselines 快 7× convergence, 10× data efficient

2. **中间 denoising features 比 reconstructed frames 更好** — 避免过度 commit 到 specific visual details,保留 actionable semantics

3. **Tri-timestep decoupling 是必要的** — video (uniform), feature extraction (fixed), action (Beta) 各有不同需求

4. **Joint training > decoupled training** — action loss regularize video backbone,latent space 更 actionable (silhouette 2× improvement)

5. **Single forward pass 足够** — bypass multi-step video generation bottleneck,实现 real-time control

6. **Video backbone > VLM backbone** — implicit physics priors 比 semantic priors 更适合 low-level continuous control (Qwen3DiT 36.2% vs DiT4DiT 50.8%)

## 14. 我的 Intuition 构建

从 Karpathy 的视角,这篇 paper 的 elegance 在于:

**Video generation 是 "world simulation"**。当 model 预测 future frames,它必须理解:
- 物体 permanence (东西不会消失)
- Physical constraints (杯子不能穿过桌子)
- Causal dynamics (推 A 会撞到 B)
- Spatial relationships (gripper 要 align with object)

这些正好是 robot control 需要的 priors。VLM 只学到 "what is this",video model 学到 "how things move"。

**DiT4DiT 的 trick 在于 "hijack" denoising process**:
- 不用最终 reconstructed frames (太 specific)
- 用中间 hidden states (编码 actionable semantics)
- 单步 forward pass 就够 (joint training 让 first step 就 actionable)

这类似于 CLIP 的 contrastive learning — 不是直接学 task,而是学一个 representation space 让 downstream task 容易。Video generation 是 proxy task,真正的 representation 在中间 layer。

**Flow matching 的选择很关键**:
- Linear interpolation 比 DDPM 的 forward SDE 更直接
- Velocity field 与 timestep 无关 (constant target),training 更稳定
- 适合 joint training (两个 flow matching losses 容易 balance)

**Beta sampling for action 的 intuition**:
Action 从 pure noise 开始 denoise。High-noise 区域 ($\tau_a \approx 1$) 是 action 生成的起点,这些样本对学习 inverse dynamics 最 critical。Beta(1.5, 1.0) 让 $\sigma$ 偏向 0,所以 $\tau_a = 1-\sigma$ 偏向 1,更多 training capacity 给 high-noise phases。

这与 flow matching 的 "hard examples" 类似 — 接近 noise 的 samples 更难,需要更多 training。

**Future directions 我会探索**:
1. Multi-camera fusion (wrist + ego-view)
2. Cross-embodiment scaling (不同 robot platforms)
3. Hierarchical planning (long-horizon with sub-goals)
4. Tactile feedback integration
5. Real-time video generation for closed-loop feedback

## 15. 相关 References

- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT (Scalable Diffusion Transformers): https://arxiv.org/abs/2212.09748
- Cosmos-Predict: https://arxiv.org/abs/2511.00062
- Cosmos-Reason1: https://arxiv.org/abs/2503.15558
- GR00T N1: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- mimic-video: https://arxiv.org/abs/2512.15692
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- FLARE: https://arxiv.org/abs/2505.15659
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523
- UniVLA: https://arxiv.org/abs/2505.06111
- CogVLA: https://arxiv.org/abs/2508.21046
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Wan: https://arxiv.org/abs/2503.20314
- DiT4DiT project page: https://dit4dit.github.io/

这篇 paper 是 video-conditioned robot control 的重要进展,tri-timestep scheme 和 joint training 的设计 choices 都很有 intuition。它 validate 了 video generation 作为 robot foundation model 的可行性,我期待看到 cross-embodiment scaling 的后续工作。
