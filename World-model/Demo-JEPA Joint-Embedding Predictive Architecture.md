---
source_pdf: Demo-JEPA Joint-Embedding Predictive Architecture.pdf
paper_sha256: a4b084d8b213a8cd6d6bc1d0e27f3073e97bd88dd2041d53836f9cf97365e287
processed_at: '2026-08-03T19:45:31-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

好，我用人话重新讲一遍。

---

## 这篇 paper 到底在干嘛

你想想一个场景：你看一个人用 UR5e 机械臂抓杯子，然后你让 Franka 机械臂去模仿这件事。两个机器人长得不一样、关节不一样、动作空间不一样，怎么模仿？

老办法大概两条路：

**第一条路**，VLA 那帮人走的：搞海量数据，把各种机器人的 action 放在一起 co-training，指望模型自己学会"跨机器人的 control 规律"。π0、OpenVLA、RT-X 都是这路子。问题是你需要巨量数据，而且 action space 对不上的话很痛苦。

**第二条路**，retargeting 那帮人：想办法把 source 的 action 映射到 target 的 action space，或者搞个 shared action space。问题是同一个语义动作（比如"抓杯子"）在不同机器人上 motor command 完全不同，硬要对齐很别扭。

Demo-JEPA 说：你们都想复杂了。

人类怎么模仿？你看别人开抽屉，你不会去想"他肩膀转了几度、肘部弯了多少"，你想的是"哦，抽屉被拉开了"，然后你自己用自己的胳膊去把抽屉拉开。**你模仿的是 outcome，不是 motor command。**

所以 Demo-JEPA 的核心 idea 就一句话：**把 demonstration 理解成"未来想达到什么状态"，而不是"怎么执行这个动作"。**

---

## 怎么实现这个 idea

分三步。

### 第一步：给 target 机器人装个"脑子"

先在 Franka 自己的数据上训练一个 world model，基于 V-JEPA 2.1。这个 world model 干嘛呢？给它当前画面 + 当前关节状态 + 一个 action，它能预测"执行这个 action 之后画面会变成什么样"——在 latent space 里预测，不预测 pixel。

为啥在 latent space 而不是 pixel space？因为 pixel space 会花大量容量去预测背景纹理、光照、机械臂长什么样这些和 task 无关的东西。latent space 只保留"abstract world structure"，比如"杯子从 A 移到了 B"。这个对 cross-embodiment 特别重要，因为你要的是 source 和 target 能共享一个 representation space，机械臂外观是 nuisance，要自动 abstract 掉。

### 第二步：教模型"看视频推断目标"

这是整篇 paper 的核心，叫 Dreamer Predictor。

输入是三样东西：
- source 视频当前帧（比如 UR5e 视频的第 k 帧）
- source 视频未来帧（第 k+n 帧，代表"source 想达到的 future"）
- target 当前帧（Franka 现在看到什么）

输出是一个 latent goal：target 机器人应该往哪个 latent state 去。

怎么算的？两个 cross-attention：

**第一个 attention**：拿 target 当前帧去 source 当前帧里找"语义对应"。比如 target 看到 Franka 前面有个杯子，这个 query 就会去 source 视频里找到杯子的位置，忽略 source 里那个 UR5e 机械臂。相当于做了个 soft correspondence——"你视频里那个杯子，对应我现在看到的这个杯子"。

**第二个 attention**：完全在 source 内部，用 source 的 future frame 去查询 source 的 current frame，提取"source 的运动趋势"。比如"杯子从左边移到右边"这个 motion pattern。

然后两个 feature 用 3D conv 拼起来，过一个 transformer decoder，输出 latent goal。

训练的监督信号是什么？target 自己真实未来帧的 latent。训练数据里 source 和 target 是 paired 的（都是做同一个 task），所以 target 真实未来已知。Dreamer Predictor 要学会：给我 source 视频"未来长啥样"，我能推出 target 应该达到的"未来 latent"。

**这里有个关键 trick**：训练时故意把 source 的时间戳抖动一下（$\delta \sim \mathcal{U}(-r, r)$），用 $(o_{k+\delta}^s, o_{k+n}^s)$ 而不是精确的 $(o_k^s, o_{k+n}^s)$。因为 inference 时 planner 执行不完美，target 不会精确在预期时间点到位，predictor 要对这种 temporal jitter 鲁棒。

### 第三步：让 world model 和 goal distribution 对齐

Stage 1 训完 Dreamer Predictor 之后，freeze 住它，然后微调 world model 的 dynamics predictor。目的是让 world model 的 rollout 分布能"接住"Dreamer Predictor 推出来的 goal 分布。

为啥要这步？因为 world model 原本是在 Franka 自己数据上学出来的，它见过的 latent transition 分布和 Dreamer Predictor 推出来的 goal 分布不一定对齐。Stage 2 让它们对齐一下，inference 时 CEM 才更容易搜到匹配 goal 的 action。

---

## Inference 的时候怎么跑

部署的时候流程是这样的：

1. 拿到 source 视频，切成一串 reference frame pair
2. 当前 reference pair 是 $(o_i^s, o_{i+\Delta}^s)$，Dreamer Predictor 推出 latent goal $\hat{z}_{\mathrm{goal}}^t$
3. 用 CEM 在 world model 里搜 action 序列：sample 一堆 action，每个 rollout 看预测的 future latent 离 goal 有多远，选最好的那批，更新 sampling 分布，重复几轮
4. 执行第一个 action，观察新状态
5. 算一下新状态离 goal 有多近：$D = d(z_{\mathrm{next}}^t, \hat{z}_{\mathrm{goal}}^t)$
6. **如果 $D < \epsilon$**（够近了，subgoal 达成）：推进到下一个 reference pair，重算 goal
7. **否则**：保持当前 goal，继续搜 action

这个 adaptive 机制很重要。source 和 target 的动作快慢不一样，固定频率推进 source 会出问题——target 还没完成当前 subgoal 就被推到下一个了。adaptive 让 target 按自己的节奏来，完成一个再推进。

---

## 实验结果说了啥

三个评测，distribution shift 递增：

| | Behavior Grounding（见过） | Cross-Embodiment（部分见过） | Zero-Shot（完全没见过） |
|---|---|---|---|
| VPP | 最好 | 一般 | 崩 |
| XSkill | 一般 | 差 | 崩 |
| Demo-JEPA | 稍弱 | **最好** | **最好** |

最有意思的 trend：**shift 越大，Demo-JEPA 优势越明显**。

Behavior Grounding 上 Demo-JEPA 不如 VPP，因为 VPP 直接学 visual representation → action 的 mapping，in-domain 下精准。但 Zero-Shot 上 VPP 是 0.04（sim）/ 0.00（real），基本全崩。Demo-JEPA 是 0.36 / 0.25。

这正好验证了 JEPA 哲学的预测：abstract predictive representation 在 distribution shift 下 generalize 更好，因为它没有 overfit pixel-level 或 action-level 的 surface statistics。

---

## 一个最有说服力的对比

论文里有个 ablation 我觉得最 informative：

- **Naive**：直接拿 source 的 future latent 当 goal，完全失败（全 0）
- **Oracle**：用 target 真实 future latent 当 goal（deployment 时拿不到，是个 upper bound）
- **Demo-JEPA**：用 Dreamer Predictor 推的 latent goal

结果 Demo-JEPA 在 cross-embodiment 上几乎追平 Oracle（real 0.55 vs 0.58）。这说明：**你不需要 target 的未来信息，只需要 source 视频和 target 当前帧，Dreamer Predictor 就能推出 target 应该达到的 future latent**，而且这个推出来的 latent 离真实 future latent 很接近。

这是整篇 paper 最强的 evidence。

---

## 还有个有意思的对比：Demo-DP

论文里还有个叫 Demo-DP 的对照：把 Dreamer Predictor 的输出当作 Diffusion Policy 的 conditioning，替代 CEM planner。结果：

- Demo-DP 在 Behavior Grounding 上比标准 DP 强（0.28 vs 0.23），说明 Dreamer Predictor 的 future-conditioning 是有效信号
- 但 Demo-DP 在 Zero-Shot 上只有 0.18，Demo-JEPA 是 0.36

为啥？因为 DP 是个 amortized 网络，一次 forward 出 action，只能 interpolate 训练分布；CEM 是 online optimization，每次都根据当前 state 和 goal 重新 search，能 extrapolate 到没见过的 goal region。

直觉上：DP 是 System 1（fast, amortized, brittle under shift），CEM 是 System 2（slow, deliberative, robust under shift）。Demo-JEPA 把 System 2 的 deliberation 放在 execution 层，System 1 的 fast inference 放在 goal prediction 层。这个分工很优雅。

---

## Scaling：task diversity 比 per-task data 重要

这个发现和 LLM pretraining 的 scaling law 同构。你保留全部 task 但每 task 减到 20% episode，比保留 20% task 但每 task 全部 episode，generalization 更好（Cross-Emb 0.27 vs 0.18，Zero-Shot 0.27 vs 0.18）。

学"source motion semantics → target latent goal"的 mapping，泛化性取决于见过的 source motion *种类*，而不是每种 motion 的 episode 数。这和 LLM 里"覆盖更多 semantic modes 比同一 mode 看更多 sample 更重要"是完全一样的直觉。

---

## 局限和我怎么看

论文自己说了三个局限：
1. World model fidelity 限制复杂高精度 task
2. 还需要 temporal alignment 准备 training data（real-world 用 GTCC 对齐）
3. 真正 unaligned learning 没实现

我补充几点：

**CEM 慢**。每 step 多轮 sampling + world model rollout，real-time Hz 论文没报，大概率不快。这是 MPC-style 方法的通病。

**Target 一侧没真正 generalize**。World model 是 Franka-specific，换 target 机器人要重训 7 天 8×A100。cross-embodiment 只在 source 一侧实现了。

**Real-world zero-shot 0.25**。绝对值还是低，离实用有距离。6 个 task 也少。

**GTCC 依赖**。本质上还是 implicit alignment，和 "no alignment needed" 的 claim 有 tension。

---

## 一句话总结

**让 demonstrator 告诉你 "what state to reach"，让 target agent 用自己的 world model figure out "how to reach it"。**

这个 decoupling 把 imitation 的难点拆成两块独立问题：intent inference（Dreamer Predictor 解决）和 execution planning（CEM + world model 解决），用 JEPA latent space 当 shared interface。

这篇 paper 最大的价值不在绝对性能，而在它给了非常清晰的 evidence：**representation matters more than scale for transfer**。VLA 阵营信 data scale 解决一切，JEPA 阵营信 right representation + small data 也能 transfer。Demo-JEPA 在 cross-embodiment 这个最有挑战性的 transfer 场景上，用几千 episode + 明确 architectural prior，达到了 VLA 在 zero-shot 上达不到的 generalization。这个对比强烈支持 "representation first" 路线。

---

Reference：
- Project page: https://log2r.github.io/Demo-JEPA/
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- VPP: https://arxiv.org/abs/2503.04398
- XSkill: https://arxiv.org/abs/2307.09030
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246

---

# Demo-JEPA: Cross-Embodiment Imitation as Latent Goal Planning

Andrej 你好,这篇 paper 我看完之后直觉上是 LeCun 那套 JEPA 哲学在 robotics imitation 上的一个非常自然的延伸——把"模仿 demonstrator 的 action"这个老问题,重新 formulate 成"在 shared predictive latent space 里推断 demonstrator 想达到的 future state,然后让 target agent 用自己的 world model 去实现它"。这跟你之前在 LLM 和 world model 上的很多思考高度同构。下面我把整篇 paper 拆开讲,公式、架构动机、实验数字、和它跟相关 work 的关系都串起来。

---

## 1. High-Level Idea: Why Decouple Intent from Execution?

传统 cross-embodiment imitation 的痛点非常清楚:Sawyer 和 Franka、UR5e 和 Franka,虽然都能"grasp a cup",但 motor command、torque profile、joint configuration 完全不一样。VLA 模型(π0、π0.5、RT-2、OpenVLA)走的是 **action alignment** 路线——在海量 multi-embodiment data 上 co-training,让模型隐式学会 cross-robot regularities;XIRL、XSkill、MimicDreamer 则尝试 retargeting、shared action space 或者 skill prototype discovery。这些方法本质都还在 action space 里折腾。

Demo-JEPA 的核心 hypothesis 是:**demonstration 应该被理解为对未来 state 的隐式 specification,而非 motor primitive 的序列**。人类看一段别人开抽屉的视频,不会去模仿对方的关节角度,而是理解"抽屉被拉开"这个 outcome,然后用自己的身体去实现。这个 insight 直接对应 LeCun 在 JEPA position paper (https://openreview.net/pdf?id=BZ5a1r-kVsf) 里反复强调的:**predictive representation 比 reconstructive representation 更能抽象掉 nuisance variation**。

所以 Demo-JEPA 把问题写成:

$$
\mathbf{a}_{k:k+H-1}^{t*} = \arg\min_{\mathbf{a}} d\big(F_{wm}(z_k, s_k^t, \mathbf{a}),\ z_{\mathrm{goal}}^t\big)
$$

变量含义:
- $\mathbf{a}_{k:k+H-1}^{t*}$: target embodiment $\mathcal{E}^t$ 在 time step $k$ 起始、horizon $H$ 的最优 action 序列,上标 $t$ 指 target,下标 $k:k+H-1$ 指时间区间
- $z_k = E(o_k^t)$: 当前 target observation 经过 encoder $E(\cdot)$ 得到的 latent state
- $s_k^t$: target robot proprioceptive state(关节角度、gripper 状态等)
- $F_{wm}(\cdot)$: action-conditioned world model 的 dynamics predictor
- $z_{\mathrm{goal}}^t$: 从 source demonstration 推出来的 target-compatible latent goal
- $d(\cdot,\cdot)$: latent distance metric,实现里用 $\ell_1$

这个 formulation 的妙处在于,optimization 完全发生在 target agent 自己的 latent dynamics 里,source demonstration 只提供 *where to go*,不提供 *how to get there*。

---

## 2. Architecture: Three Building Blocks

整体 pipeline 是三块拼起来的:

### 2.1 Action-Conditioned World Model (V-JEPA 2.1)

这部分不是 paper 的主要 contribution,直接基于 V-JEPA 2.1 (https://arxiv.org/abs/2506.09985, 注:实际 V-JEPA 2 是 arXiv:2506.09985,V-JEPA 2.1 是其 dense feature 改进版)。架构上:

$$
z_k = E(o_k^t), \qquad \hat{z}_{k+1} = F_{wm}(z_k, s_k^t, a_k^t)
$$

$E(\cdot)$ 是 ViT encoder,patch size 16, tubelet size 2(意味着时间维度上每 2 帧聚成一个 token),embed dim 1024,predictor 24 层 transformer,16 个 attention heads,frame-causal masking。这是典型的 ViT-based video JEPA 配置。

**关键 insight**:为什么选 JEPA 而不是 Dreamer (https://arxiv.org/abs/1912.01603) 那种 pixel-reconstructive world model?因为 pixel reconstruction 会把 representational capacity 浪费在 background texture、lighting、embodiment-specific hardware appearance 这些 task-irrelevant 细节上。而 cross-embodiment transfer 恰恰需要抽象掉这些。JEPA 的 latent space 通过 predictive objective 被优化来捕获 *abstract world structure*,这正是 source 和 target 能共享一个 representation space 的前提。

这点和 VPP (Video Prediction Policy, https://arxiv.org/abs/2503.04398) 形成鲜明对比——VPP 在 pixel space 做视频预测,所以它必须在 source embodiment 的视觉细节上花大量容量,这也解释了为什么 VPP 在 Behavior Grounding 上强(同分布下视觉细节帮得上忙),但在 Zero-Shot 上崩盘(细节 mismatch)。

### 2.2 Dreamer Predictor: The Core Innovation

这是整篇 paper 的灵魂。它要解决的问题是:**给定 source demonstration 的两帧 $(o_k^s, o_{k+n}^s)$ 和 target 当前帧 $o_k^t$,推断出一个在 target embodiment 的 world model 里 *可达* 的 latent goal $\hat{z}_{\mathrm{goal}}^t$**。

注意这个 framing 的微妙之处。如果直接拿 source 的 future latent $z_{k+n}^s = E(o_{k+n}^s)$ 当 goal(论文里叫 "naive reference"),会完全失败——Table 4 和 5 里 Naive 那一行是空的或者全 0。原因很直觉:source 的视觉里 UR5e 机械臂出现的位置、姿态和 Franka 完全不同,JEPA latent 虽然抽象了 background,但机械臂本身的 appearance 是保留的,直接 cross-embodiment 用 latent 当 goal 会把 planner 带偏。

Dreamer Predictor 的架构很有意思,是两个 cross-attention module + 3D conv fusion + Transformer decoder:

**Cross-attention 1: Embodiment Correspondence**
$$
f_{\mathrm{emb}} = \mathrm{Attn}(Q=z_k^t,\ K=z_k^s,\ V=z_k^s)
$$

这个 attention 的语义是:以 target 当前 latent $z_k^t$ 为 query,去 source 当前 latent $z_k^s$ 里 *检索* 语义对应的 region。比如 target Franka 看到 cup 在桌上,query 就会 attend 到 source 视频里 cup 的位置,忽略 source 视频里的 UR5e 机械臂。这相当于做了一个 soft cross-embodiment correspondence。

**Cross-attention 2: Motion/Trajectory**
$$
f_{\mathrm{mot}} = \mathrm{Attn}(Q=z_{k+n}^s,\ K=z_k^s,\ V=z_k^s)
$$

这个 attention 完全在 source side 内部:用 source future frame $z_{k+n}^s$ 作为 query,去 source current frame $z_k^s$ 里找 temporal motion 的对应。这一步其实是把 source demonstration 的"运动趋势"编码出来——"cup 从 A 移到 B"这种 motion pattern 在 source latent space 里被自对齐地提取出来。

注意 $f_{\mathrm{mot}}$ 完全不依赖 target,这是有意的解耦:source 的 motion 信息(independent of embodiment)和 source-to-target 的 correspondence 是两个独立的 computation,后面再 fuse。

**3D Conv Fusion**
$$
f_{\mathrm{fused}} = \phi\left(\left[\mathbf{z}_k^t \oplus f_{\mathrm{emb}} \oplus f_{\mathrm{mot}}\right]\right)
$$

$\oplus$ 是 channel-wise concatenation,$\phi$ 是 3D convolution。为什么用 3D conv 而不是 mean pooling 或者 linear projection?Ablation (Table 4, 5) 显示,对于简单 task 比如 Basketball in Hoop,mean pooling 就够了(0.33 vs 0.33);但对于 Change Channel(需要 twist wrist)、Close Box(articulated closure)、Remove Plate(coordination-heavy)这种需要结构化时空 motion 的 task,Conv3D 显著更好(simulation 上 avg 0.31 vs 0.21,real-world 上 0.43 vs 0.35)。

直觉上:Conv3D 在(空间 H × 空间 W × 时间 T)三个维度上做 local 卷积,能保留 latent token 之间的 spatial layout 和 temporal order;mean pooling 是 permutation-invariant 的,会把 "先抓后提" 这种时序信息平均掉。JEPA latent 本身是 spatially structured(ViT patch grid),Conv3D 能尊重这个结构。

**Transformer Decoder**
$$
\hat{z}_{\mathrm{goal}}^t = \mathcal{T}(f_{\mathrm{fused}})
$$

4 层 self-attention refinement,把 fused feature decode 成 latent goal。整个 Dreamer Predictor 参数不算大,关键是它在一个 *已经 pretrained* 的 JEPA latent space 上工作,不需要重新学习视觉 representation。

### 2.3 CEM Planner

Cross-Entropy Method (Rubinstein 1997, https://www.sciencedirect.com/science/article/pii/S0305048396001886) 是个 model-predictive control (MPC) 风格的 sampling-based optimizer。算法步骤(Algorithm 1):

1. 初始化 Gaussian $\mathcal{N}(M, \mathrm{diag}(S^2))$,$M \in \mathbb{R}^{H \times \dim(a)}$, $S \in \mathbb{R}^{H \times \dim(a)}$
2. 每轮采样 $N$ 个 candidate action sequences $\mathbf{a}_i$
3. 每个候选 rollout 通过 $F_{wm}$ 得到 $\hat{z}_{t+H}^{(i)} = F_{wm}(z_t, s_t, \mathbf{a}_i)$
4. 算 loss $\mathcal{L}_i = d(\hat{z}_{t+H}^{(i)}, z_{gt})$
5. 选 top-$K$ lowest-loss 作为 elites
6. Momentum update:$M \gets \beta M + (1-\beta) M_{\mathrm{elite}}$, $S \gets \beta S + (1-\beta) S_{\mathrm{elite}}$
7. 重复 $L$ 轮,返回 $M$ 作为最优 action

实现里 $H$ 是 planning horizon,$N$ 是 population size,$K$ 是 elite 数量,$\beta$ 是 momentum(典型值 0.1-0.5)。

**为什么用 CEM 而不是直接学一个 inverse model $a = G(z_t, z_{gt})$?** 因为 inverse model 在 multi-modal goal distribution 下会 mode-collapse(平均多个 valid action 导致无效 action)。CEM 是 sampling-based,天然支持 multi-modal solutions。这个 trade-off 和 Diffusion Policy (https://arxiv.org/abs/2303.04137) 选择 diffusion 而非 regression 的动机一致——只是 Demo-JEPA 选择在 *planning 层* 做 sampling,而 DP 选择在 *action 层* 做 sampling。论文 Section 4.2 的 Demo-DP 对比正好把这两条路径区分开。

---

## 3. Three-Stage Training

这个 training pipeline 是 staged 的,有点类似 pretrain-then-finetune 的层级:

### Stage 0: Pretrain Action-Conditioned World Model

直接复用 V-JEPA 2 训练流程,在 target embodiment (Franka) 自己的 interaction data 上训练 $E$ 和 $F_{wm}$。8×A100 训 7 天。这一步得到的 latent space 是后续所有 cross-embodiment 工作的 *锚点*。

### Stage 1: Train Dreamer Predictor

数据是 paired visual trajectories:source (Sawyer 或 UR5e) 的视频 + target (Franka) 的视频,二者在时间上对齐(simulation 用 retargeted end-effector pose replay;real-world 用 GTCC (https://openaccess.thecvf.com/content/CVPR2024/papers/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.pdf) 做 progress-aware frame-level alignment)。

Loss 是 latent reconstruction:
$$
\mathcal{L}_{\mathrm{pred}} = \|\hat{z}_{\mathrm{goal}}^t - z_{k+n}^t\|_2^2
$$

$z_{k+n}^t = E(o_{k+n}^t)$ 是 target 真实未来帧的 latent(训练时有,target 自己的 data),作为监督信号。Dreamer Predictor 要学会从 source 视频的"未来样子"推断 target 应该达到的"未来 latent"。

**Temporal Perturbation Trick**:训练时不直接用 $(o_k^s, o_{k+n}^s)$,而是采样 $\delta \sim \mathcal{U}(-r, r)$,用 $(o_{k+\delta}^s, o_{k+n}^s)$。这是个关键 regularization——inference 时 planner 会执行不完美,target 不会精确处于 $o_k^t$ 状态,所以 Dreamer Predictor 要在 source reference 时间轴有 jitter 的情况下仍能输出合理 goal。这个 trick 直接对应 BCQ/TD3 里 target policy smoothing 的思想,在 goal prediction 上做了 temporal smoothing。8×A100 训 2.5 天。

### Stage 2: Action Co-Training

Stage 1 训完后 freeze Dreamer Predictor,unfreeze $F_{wm}$,用 planning loss 微调:

$$
\mathcal{L}_{\mathrm{plan}} = \|F_{wm}(z_k^t, s_k^t, \mathbf{a}_{k:k+n-1}^t) - \hat{z}_{\mathrm{goal}}^t\|_2^2
$$

这步的目的是 *align* world model 的 rollout 分布和 Dreamer Predictor 的 goal 分布。直觉上:Dreamer Predictor 推断出的 $\hat{z}_{\mathrm{goal}}^t$ 是个特定分布的 latent point,world model 原本是在 Franka 自己数据上学出来的,可能没见过这种"被 source 视频引导"出的 latent region。Stage 2 让 world model 适应这个新分布,使 inference 时 CEM 更容易找到匹配 goal 的 action。8×A100 训 1 天。

**为什么这种 staged training 重要?** 如果一上来 joint train,world model 还在变化的时候 Dreamer Predictor 的监督信号 $z_{k+n}^t = E(o_{k+n}^t)$ 也会漂移,容易 collapse。Stage 1 先 freeze world model 学一个稳定的 predictor,Stage 2 再让 world model 适配 predictor 输出——这是典型的 *commit-then-adapt* 训练模式,和 ALBEF、BLIP 那种 contrastive-then-generative 的 staged 思路一脉相承。

---

## 4. Inference: Adaptive Goal Updating

这部分是个工程上很巧的设计。Naive 做法:每个 time step 推进 source reference 一帧。问题是 source 和 target 的 kinematics 不一样,source 视频 10 帧完成的动作 target 可能需要 20 帧,或者反过来。固定频率推进会导致 target 还没完成当前 subgoal 就被推到下一个 subgoal,compounding error 累积。

Adaptive Goal Updating 机制:

1. 当前 reference pair 是 $(o_i^s, o_{i+\Delta}^s)$,$\Delta$ 是 temporal offset
2. Dreamer Predictor 算出 $\hat{z}_{\mathrm{goal}}^t$
3. CEM 规划 + 执行第一个 action $a_0^*$
4. 观察 new state $o_{\mathrm{next}}^t$, encode 得 $z_{\mathrm{next}}^t$
5. 计算 discrepancy $D = d(z_{\mathrm{next}}^t, \hat{z}_{\mathrm{goal}}^t)$
6. **如果 $D < \epsilon$**:subgoal 达成,推进 $i \gets i+1$,更新 reference pair,重算 goal
7. **否则**:保持当前 goal,继续 CEM 优化

这个机制本质上是 *latent-space reachability check*,类似 classical motion planning 里的 goal-conditioned RRT 在 latent space 里的版本。$\epsilon$ 是个 distance threshold,起到 subgoal completion detector 的作用。

直觉上,这个设计让 source demonstration 的 *temporal granularity* 和 target execution 的 *temporal granularity* 解耦——target 按自己的速度完成当前 subgoal,完成后再推进到下一个 reference。对 long-horizon task 非常关键,也是 Demo-JEPA 在 zero-shot setting 上远超 VPP/XSkill 的一个重要原因(后两者没有这种 adaptive 机制)。

---

## 5. Experiments: Distribution Shift as a Stress Test

实验设计非常讲究,分三个 suite,distribution shift 递增:

1. **Behavior Grounding**:从 Stage 1 + Stage 2 都见过的 task,fully supervised 条件下测 execution quality
2. **Cross-Embodiment Bridging**:Stage 1 见过、Stage 2 没见过的 task,测 predictor 推的 goal 能不能 drive execution
3. **Zero-Shot Generalization**:Stage 1 也没见过的全新 task 配置

Simulation (RLBench, https://arxiv.org/abs/1909.11271) 用 Sawyer → Franka,real-world 用 UR5e → Franka。

### 5.1 Main Results Analysis

| Method | Sim Behavior Grounding | Sim Cross-Embodiment | Sim Zero-Shot |
|---|---|---|---|
| VPP | 0.47 | 0.28 | 0.04 |
| XSkill | 0.39 | 0.17 | 0.03 |
| Demo-JEPA | 0.31 | **0.45** | **0.36** |

| Method | Real Behavior Grounding | Real Cross-Embodiment | Real Zero-Shot |
|---|---|---|---|
| VPP | 0.65 | 0.53 | 0.00 |
| XSkill | 0.45 | 0.40 | 0.05 |
| Demo-JEPA | 0.43 | **0.55** | **0.25** |

**关键观察**:Demo-JEPA 在 Behavior Grounding 上 *不如* VPP!这很反直觉。论文解释是 VPP 在 in-domain trajectory learning 上有优势(它直接学 visual representation → action 的 mapping),但 Demo-JEPA 用 CEM planner 在 inference 时是 sampling-based,有一定 overhead,可能不如直接 regression 精准。

但 *shift 越大 Demo-JEPA 优势越明显*。Zero-Shot 上 sim 是 9× 优势(0.36 vs 0.04),real-world 是 ∞ 优势(0.25 vs 0.00)。这个 trend 几乎完美对应 JEPA 哲学的 prediction:abstract predictive representation 在 distribution shift 下 generalize 更好,因为它没有 overfit pixel-level 或 action-level surface statistics。

### 5.2 Goal Reference Ablation: 最 informative 的对比

Table 4 和 5 里有三种 goal:
- **V-JEPA 2.1 (Naive)**:直接用 $z_{k+n}^s$ 当 goal,完全失败(表格里空着意味着全部 0)
- **V-JEPA 2.1 (Oracle)**:用 target ground truth future $z_{k+n}^t$,这是个 *upper bound*,deployment 时拿不到
- **Demo-JEPA**:用 Dreamer Predictor 推出来的 $\hat{z}_{\mathrm{goal}}^t$

对比 Oracle vs Demo-JEPA:
- Sim: Oracle 0.55 / 0.42 vs Demo-JEPA 0.31 / 0.36 (Cross-Embodiment / Zero-Shot avg)
- Real: Oracle 0.58 / 0.28 vs Demo-JEPA 0.55 / 0.25

Demo-JEPA 在 Cross-Embodiment 上几乎追平 Oracle(real 0.55 vs 0.58),说明 Dreamer Predictor 推的 goal 离 target 真实未来 latent 很接近。这是整篇 paper 最强的 evidence:**你不需要 target 的未来信息,只需要 source 视频和 target 当前帧,就能推断出 target 应该达到的 future latent**。

### 5.3 Conv3D Ablation

| Variant | Sim avg (3 suites) | Real avg (3 suites) |
|---|---|---|
| Demo-JEPA | 0.31/0.45/0.36 | 0.43/0.55/0.25 |
| w/o Conv3D | 0.21/0.44/0.29 | 0.35/0.40/0.23 |

Behavior Grounding 上 Conv3D 影响最大(real 0.43 vs 0.35),因为 in-domain 时 motion 复杂度对执行质量最敏感。Cross-Embodiment 上(sim)几乎没影响(0.45 vs 0.44),说明 goal inference 本身在简单 task 上不需要复杂时空建模,但 execution 端需要。这印证了 Conv3D 的作用是 *保留 latent 间的 spatiotemporal structure*,在难任务上才显出价值。

### 5.4 Scaling Study: Task Diversity > Per-Task Data

Table 8 是个很 thoughtful 的 ablation:
- 20% data scaling (保留全部 task,每 task 减到 20% episode):Cross-Emb 0.27, Zero-Shot 0.27
- 20% task scaling (保留全部 episode,task 数减到 20%):Cross-Emb 0.18, Zero-Shot 0.18
- 50% data scaling: 0.38 / 0.29
- 50% task scaling: 0.33 / 0.25
- Full: 0.45 / 0.36

**Task diversity 比 per-task data 重要**。这和 LLM pretraining 的 scaling law 同构:覆盖更多 *semantic modes* 比在同一个 mode 上看更多 sample 更能学到 transferable representation。对 cross-embodiment 这个任务,本质是学"source motion semantics → target latent goal"的 mapping,这个 mapping 的泛化性取决于见过的 source motion 种类,而非每种 motion 的 episode 数。

### 5.5 Demo-DP: Planner vs Direct Policy

Demo-DP 是个非常有趣的对照组:把 Dreamer Predictor 的输出 $\hat{z}_{\mathrm{goal}}^t$ 作为 Diffusion Policy 的 conditioning,替代 Demo-JEPA 的 CEM planner。结果:

| Method | Sim Behavior Grounding | Sim Cross-Emb | Sim Zero-Shot |
|---|---|---|---|
| DP | 0.23 | - | - |
| Demo-DP | 0.28 | 0.44 | 0.18 |
| Demo-JEPA | 0.31 | 0.45 | **0.36** |

Demo-DP 在 Behavior Grounding 上比标准 DP 强(0.28 vs 0.23),证明 Dreamer Predictor 提供的 future-conditioning 是有效 signal。但 Demo-DP 在 Zero-Shot 上(0.18)显著弱于 Demo-JEPA(0.36),说明 **planner-based execution 在 distribution shift 下远比 direct policy robust**。直觉:DP 是个 amortized inference 网络,只能 interpolate 训练分布;CEM 是 online optimization,每次都根据当前 latent state 和 goal 重新 search action,可以 extrapolate 到没见过的 goal region。这个对比强烈支持 Demo-JEPA 的 MPC-style 设计选择。

这个 observation 让我想到你之前多次提到的 *System 1 vs System 2* 区分:DP 是 System 1(fast,amortized,但 brittle under shift),CEM planner 是 System 2(slow,deliberative,robust under shift)。Demo-JEPA 整体架构把 System 2 的 deliberation 放在了 execution 层,而 System 1 的 fast inference 放在了 goal prediction 层。这个分工比纯 DP 或纯 MPC 都更优雅。

---

## 6. Connection to Broader Landscape

### 6.1 JEPA 谱系

Demo-JEPA 是 LeCun 系 JEPA 在 robotics 上的延伸:
- I-JEPA (https://arxiv.org/abs/2301.08243):image-level predictive learning
- V-JEPA (https://arxiv.org/abs/2404.08471):video-level,masked latent prediction
- V-JEPA 2 (https://arxiv.org/abs/2506.09985):加了 action conditioning,能用于 planning
- V-JEPA 2.1:dense feature 改进版
- Demo-JEPA(本文):在 V-JEPA 2.1 latent space 上加 cross-embodiment goal translation

整条线都在验证同一个核心 thesis:**predictive latent representation > reconstructive pixel representation**,对于 abstract reasoning、planning、transfer 都更有效。Demo-JEPA 把这个 thesis 推到了 cross-embodiment imitation 这个具体场景上。

### 6.2 World Models 谱系

World model 这条线从 Ha & Schmidhuber (https://arxiv.org/abs/1803.10122) 的 VAE+MDN-RNN,到 Dreamer (https://arxiv.org/abs/1912.01603) 的 RSSM + actor-critic,到 TWM、DIAMOND,再到 JEPA-based world models。核心区别在 *representation objective*:
- VAE/Dreamer:reconstructive,latent 必须 encode 所有 pixel 信息
- JEPA:predictive,latent 只需要 encode *predictable* 的信息

对 cross-embodiment 这件事,这个区别是决定性的——reconstructive latent 会 over-represent 机械臂外观,导致 source 和 target 的 latent 不可比;predictive latent 只关心 "cup 会从 A 移到 B" 这种 abstract dynamics,机械臂外观是 nuisance,会被自动 abstract 掉。

### 6.3 Cross-Embodiment Imitation 谱系

- **XIRL (https://arxiv.org/abs/2108.12459)**:cross-embodiment inverse RL,学 reward
- **XSkill (https://arxiv.org/abs/2307.09030)**:discover skill prototypes,match video to prototype
- **VPP (https://arxiv.org/abs/2503.04398)**:video prediction-based policy
- **Open X-Embodiment (https://arxiv.org/abs/2310.08864)** + RT-X:大数据 co-training
- **π0/π0.5 (https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054)**:VLA flow model,大 scale
- **Universal Actions (https://arxiv.org/abs/2501.10154)**:unified action space
- **MimicDreamer (https://arxiv.org/abs/2509.22199)**:align human + robot demo
- **Demo-JEPA(本文)**:latent goal planning,完全 avoid action alignment

Demo-JEPA 的独特位置:它和 π0/OpenVLA 这条线是完全 orthogonal 的——后者靠 data scale 解决 embodiment gap,前者靠 *representation-level abstraction* + *planning* 解决。两条路线在哲学上对应 LLM 里的 *scale vs architecture* debate。Demo-JEPA 的 evidence 显示,即使 data 有限(每个 task 几百 episode),只要 representation 选对了,小 data 也能 cross-embodiment。

### 6.4 和 Visual Foresight 的关系

Visual Foresight (Finn & Levine 2017, https://arxiv.org/abs/1706.05269) 是 early 的 video-prediction-based planning,核心 idea 是预测 future video 然后在 video space MPC。Demo-JEPA 是这个 idea 的 latent-space + cross-embodiment 版本。关键改进:
1. 在 latent 而非 pixel 上预测,避免 irrelevant detail
2. 加了 cross-embodiment goal translation(Dreamer Predictor)
3. Adaptive goal updating 处理 source/target temporal mismatch

### 6.5 和 Diffusion Policy 的关系

DP (https://arxiv.org/abs/2303.04137) 用 diffusion 在 action space 做 sampling,处理 multi-modal action distribution。Demo-JEPA 用 CEM 在 *action sequence space* 做 sampling。两者都是 sampling-based,但 DP 是 amortized(网络一次 forward 出 action),CEM 是 iterative(多轮 search)。Demo-DP 那个 ablation 显示:在 distribution shift 下,iterative search 比 amortized inference 鲁棒得多。这对应你之前说过的一个直觉——*test-time compute* 在 OOD 场景下价值更高。

---

## 7. Limitations 和 My Take

论文自己列了三个 limitation:
1. World model fidelity 限制复杂高精度 task 性能
2. 仍需要 temporal/progress-aware alignment 做 training data 准备(real-world 用 GTCC)
3. 真正 unaligned learning 还没实现

我的看法:

**强**:
- Formulation 干净,把 cross-embodiment 从 action alignment 问题转化为 latent goal planning 问题,瞬间绕过 VLA 那条 data-hungry 路线
- Dreamer Predictor 设计很 minimal 但 effective——两个 cross-attention 分别建模 correspondence 和 motion,3D conv 保留时空结构,很 elegant
- Adaptive goal updating 是个工程上的 *right call*,解决 long-horizon 下 source/target temporal mismatch
- 实验设计非常 informative,Oracle vs Naive vs Demo-JEPA 这个三角对比直接验证了核心 thesis
- Scaling study 给出了 actionable insight:task diversity > data

**潜在问题**:
- CEM inference 慢,每 step 多轮 sampling + world model rollout,real-time 性未知。论文没报告 Hz,应该补
- World model 是 Franka-specific,换个 target embodiment 要重训 Stage 0(7 天 8×A100),cross-embodiment 的 *target* 一侧其实没真正 generalize
- Real-world 只有 6 个 task,zero-shot 上 0.25 的绝对成功率还是低,离实用有距离
- 对 GTCC 的依赖本质上还是一种 implicit alignment,和 "no alignment needed" 的 claim 有 tension
- Dreamer Predictor 需要 paired source-target data,这比纯 self-supervised 视频预训练还是贵很多

**哲学共鸣**:
这篇 paper 在我最关心的一个问题上给了非常清晰的 evidence:**representation matters more than scale for transfer**。VLA 阵营相信 data scale 解决一切,JEPA 阵营相信 right representation + small data 也能 transfer。Demo-JEPA 在 cross-embodiment 这个最有挑战性的 transfer 场景上,用相对小的数据(几千 episode)和明确的 architectural prior(predictive latent + cross-embodiment attention + CEM planner),达到了 VLA 在 zero-shot 上达不到的 generalization。这个对比强烈支持"representation first"的路线。

如果让我做 next step,我会:
1. 把 Dreamer Predictor 扩展到 multi-task/multi-source-embodiment co-training(同时看 Sawyer + UR5e + 人类手 + 各种 robot 的 video),测试 predictor 本身的 cross-embodiment scaling
2. 把 CEM 替换成 latent-space diffusion planner(类似 Diffuser, https://arxiv.org/abs/2205.09991),可能更 amenable to amortized training
3. 用 V-JEPA 2 的 *non-action-conditioned* 版本做 source representation(不依赖 target embodiment 的 action space),真正实现 action-free source learning
4. 把 world model 也做成 multi-embodiment,实现 *target* 一侧的 generalization(目前 target 只能 Franka)

---

## 8. Final Intuition

如果要把 Demo-JEPA 压缩成一句话:**"让 demonstrator 告诉你 *what state to reach*,让 target agent 用自己的 world model figure out *how to reach it*。"**

这个 decoupling 把 imitation 的难点拆成了两块独立可解决的问题:
- *Intent inference*:从视频里提取"想达到什么状态"——Dreamer Predictor 解决
- *Execution planning*:在自己的 dynamics 里搜出 action 序列——CEM + world model 解决

两块用 JEPA latent space 作为 shared interface。这种 *modular decoupling* 思路,和你一直在神经网络模块化、可组合性、System 1/System 2 分工上的思考是高度一致的。Demo-JEPA 是这个 philosophy 在 robotics 上的一个具体、workable 的 instance。

---

**References**:
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- V-JEPA: https://arxiv.org/abs/2404.08471
- I-JEPA: https://arxiv.org/abs/2301.08243
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- RLBench: https://arxiv.org/abs/1909.11271
- CEM (Rubinstein): https://www.sciencedirect.com/science/article/pii/S0305048396001886
- Dreamer: https://arxiv.org/abs/1912.01603
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Visual Foresight: https://arxiv.org/abs/1706.05269
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Diffuser: https://arxiv.org/abs/2205.09991
- VPP: https://arxiv.org/abs/2503.04398
- XSkill: https://arxiv.org/abs/2307.09030
- XIRL: https://arxiv.org/abs/2108.12459
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- RT-2: https://arxiv.org/abs/2307.15818
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- MimicDreamer: https://arxiv.org/abs/2509.22199
- Universal Actions: https://arxiv.org/abs/2501.10154
- GTCC: https://openaccess.thecvf.com/content/CVPR2024/papers/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.pdf
- Project page: https://log2r.github.io/Demo-JEPA/

如果你对某个具体模块(比如 Dreamer Predictor 的 cross-attention 设计、CEM 调参、Stage 1/2 训练 dynamics)想再深挖,告诉我,我可以再展开讲。
