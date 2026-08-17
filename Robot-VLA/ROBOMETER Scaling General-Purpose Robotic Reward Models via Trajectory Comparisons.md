---
source_pdf: ROBOMETER Scaling General-Purpose Robotic Reward Models via Trajectory
  Comparisons.pdf
paper_sha256: f6c6c7878c0254b3ceae697363b84407ffd627d0a69dccede116ecc444b1f88e
processed_at: '2026-08-12T01:11:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 ROBOMETER

## 一句话总结

以前的 robot reward model 只会看 expert demo 打分，遇到 failure data 就傻眼了。ROBOMETER 的核心 idea 就是：别光看绝对分数，学着比较——哪条 trajectory 更好，哪条更差。这样就能把大量 previously useless 的 failure data 利用起来，训练出更 general 的 reward model。

---

## 为什么这件事是个问题

想象你在训练一个 robot policy，需要一个 reward function 告诉 robot "做得好不好"。传统做法是：拿 expert demo，linearly 从 0 到 1 标 progress，然后 train 一个 model 去 predict 这个 progress。简单粗暴，对 expert demo 挺管用。

但 real world 不是这样的。Real robot data collection 会产生大量 garbage trajectory——robot 掉东西了、卡住了、走错方向了、重复同一个错误动作。这些 trajectory 的 progress 你没法标。标 0.5？它可能走对了 80% 然后掉下来。标 0.0？但它确实做过一些 correct motion。标一个随时间波动的曲线？那是 ill-posed problem。

结果就是：大量 data 被浪费。而 robot data collection 又贵又慢，浪费 data 就是浪费钱和时间。

---

## ROBOMETER 的 key insight

人类判断事物的时候，relative comparison 比 absolute scoring 更 fundamental。你问一个人"这个 pizza 几分"，他可能说不清楚。但你给他两个 pizza 问"哪个更好"，他马上能回答。

ROBOMETER 把这个 intuition 搬到 robot reward model 上：

- **Progress prediction**：沿着单条 trajectory，告诉 model "这段视频的第 5 帧 task 完成了 50%"。这是 absolute grounding。
- **Preference prediction**：给两条 trajectory，告诉 model "第一条比第二条更好地完成了 task"。这是 relative ordering。

两个 signal 叠加，model 既能知道 absolute scale，又能知道 global ranking。关键好处是：**preference 不需要 absolute progress label**。你有一条 expert demo 和一条 failure trajectory，你只需要知道 expert 比 failure 好，不需要知道 failure 的 progress 具体是多少。这就 unlock 了所有 previously unusable 的 failure data。

---

## 怎么实现的

Base model 是 Qwen3-VL-4B，一个 vision-language model。关键 design 是 token sequence 的构造：

```
[语言指令] [视频1的每一帧 + progress token] [分隔符] [视频2的每一帧] [preference token]
```

- 每一帧后面插一个 `<|prog_token|>`，因为 causal mask 的关系，第 t 帧的 progress token 只能 attend to 前面 1 到 t 帧的 visual token，天然给你 dense per-frame progress estimate。
- 末尾放一个 `<|pref_token|>`，能 attend to 两个 trajectory 的所有 frame，做 cross-video comparison。

这个 design 比 Bradley-Terry loss（给每条 trajectory 算一个 scalar score 再比较）强得多。Table XII 显示 Kendall τ 从 0.325 提升到 0.655。原因很直觉：VLM 的 attention 机制是最强的 inductive bias，BT loss 把 comparison 退化成两个 scalar 的比较，浪费了 VLM 的 cross-attention 能力。而 dedicated token 让 VLM 能 explicit 地让一个 video 的 frame 直接 attend 到另一个 video 的 frame，做 fine-grained 对比。

Progress prediction 没有 直接 regress scalar，而是用 C51 distributional formulation，把 [0,1] 分成 10 个 bin，predict categorical distribution。这个 design choice 让 model 能表达 uncertainty——当 frame 的 progress 是 ambiguous 的时候，model 可以输出一个 spread-out distribution 而不是被迫给一个 point estimate。

---

## 数据从哪来

RBM-1M，1,059,370 条 trajectory，21 个 robot embodiment。来源包括：

- OXE Mix（449k expert demo）
- AGIBotWorld（216k long-horizon bimanual）
- Galaxea Open World（108k humanoid）
- RoboReward OXE Mix（45k，带 counterfactual label）
- 各种 human-robot paired data
- **Mixed expertise data**（~140k，包含 paired success/fail）
- Epic-Kitchens（37k human video，只用于 preference）

关键的 design philosophy 不是盲目堆 quantity，而是堆 diversity——viewpoint、scene、embodiment 都要 diverse。而且有意 inclusion failure data，这是 prior work avoid 的东西。

有些 dataset 的 trajectory termination time 和实际 task completion 严重不同（teleop delay），这些 data 如果做 progress prediction 会引入 noise。但 ROBOMETER 的 design 让它们仍可用于 preference prediction——因为 preference 只关心 relative ordering，对 end-frame noise 更 robust。

---

## 怎么造 preference pair

三种 strategy：

**1. Different Expertise**：拿一条 expert（progress=1）和一条 failure（progress=None），expert 永远 preferred。这教会 model 区分 execution quality。

**2. Different Task**：拿两条不同 task 的 trajectory，随机选一个 instruction，preference 指向对应 trajectory，另一条的 progress target 设 0。这 ground reward 在 language 上，防止 model 给 visually plausible 但 semantically wrong 的 motion 高 reward。

**3. Video Rewind**：从单条 expert trajectory 里截一段，正常 forward 作为 chosen，倒放作为 rejected。这显式 model "undoing progress"——RL 探索时极常见的 failure mode（policy 走对了又走回来）。Ablation 显示这个 augmentation 贡献最大。

---

## 结果怎么样

**Reward evaluation**：
- RBM-EVAL-OOD 上 VOC r 达到 0.95（vs RoboReward-8B 的 0.88）
- Kendall τ_a 达到 0.66（vs RoboReward-8B 的 0.47）
- 即使只 用 RoboReward 的 45k data 训练，ROBOMETER 也比 RoboReward 本身好（VOC 0.93 vs 0.88），证明方法论本身有效

**Ablation 三个 hypothesis**：
- H1：没有 failure data，加 preference supervision 就能提升 Kendall τ 从 0.63 → 0.74。Global comparative constraint 本身就 induce 更 structured 的 internal reward representation。
- H2：加 failure data，Kendall τ 跳到 0.92，Suc-Fail Diff 从 0.11 → 0.46（4× improvement）。Failure data 提供 contrastive signal。
- H3：换掉 pre-trained VLM backbone，用 from-scratch transformer，所有 metric 崩溃，Kendall τ 变成 -0.14。Large-scale multimodal pre-training 是 essential 的。

**Downstream applications 四个 use case**：

1. **Automatic Online RL**：用 DSRL steer π_0 policy。Single-stage task 从 20% → 85% success（vs RoboReward 的 55%）。Multi-stage task 从 20% → 70%（vs RoboReward 的 20%，完全没提升）。RoboReward 的致命问题是给 wrong object manipulation 打 max reward，导致 RL 学错行为。

2. **Offline RL**：用 IQL 在 expert + noisy trajectory mix 上训练。ROBOMETER 的 dense reward 让 IQL 在更低 discount factor γ=0.9 上 work best（dense reward 减少 long-horizon credit assignment 需求）。Average 2.4× improvement。

3. **Data Filtering & Retrieval**：从 unlabeled play dataset retrieve task-relevant subtrajectory。ROBOMETER retrieve 的 data 训出的 policy 平均 4.5× success rate。Baselines retrieve 了太多 failed 但 task-relevant 的 subtrajectory，反而 degrade policy performance。

4. **Failure Detection**：Zero-shot 检测 deployment 时的 failure。Average F1 = 0.81。Detection mechanism 是 sliding window 算 progress 和 time 的 Pearson correlation，correlation 变负就 flag failure。能 detect irreversible failure（object drop 导致 progress sharp regression）、insufficient-progress failure（progress 停滞）、semantic failure（wrong object，progress consistently low）。

---

## 为什么这个 work 重要

从更宏观的角度看：

**1. Reward model 的 scaling hypothesis 验证**。1M trajectory + diverse embodiment + intentional failure data，三者缺一不可。Data composition 比 raw quantity 更关键。

**2. Preference + Progress 的 dual objective 类比了 LLM 里的 SFT + RLHF**。Progress loss 像 SFT（absolute grounding），preference loss 像 RLHF（relative alignment）。Robotics 正在重走 LLM 的路，但 grounding 问题让每一步都更难。

**3. VLM pre-training 是 non-negotiable 的**。H3 ablation 证明 from-scratch transformer 完全不行。VLM 的 visual-semantic pre-training 已经 encode 了 task understanding，reward learning 只是把这个 understanding 适配到 reward space。

**4. Synthetic preference 可以 scale**。不需要 human labeling，三种 augmentation strategy 就能 generate 足够的 preference signal。这 unlock 了 robotics reward modeling 的 data bottleneck。

**5. Test-time compute via reward-ranked sampling**。Appendix G 的 DreamZero experiment 虽然只是 proof-of-concept，但思路有意思——用 reward model rank world model 生成的 candidate future trajectory，3.5× success rate improvement。这是 best-of-N sampling 在 robotics 的 instance。

---

## 我觉得最 clever 的几个点

**1. Preference-only data 的 inclusion**。有些 dataset termination time 太 noisy，不能做 progress prediction，但能做 preference prediction。这个 design 让 dataset utilization 最大化。

**2. Success cutoff 的 per-dataset manual annotation**。看起来 mundane 但很重要。Teleop data 的 trajectory termination ≠ task completion，operator 有 delay。用 wrong cutoff 会让 success label noisy。花 2 min/dataset manually 标，ROI 很高。

**3. Fixed trajectory length T for preference training**。Anti-shortcut design，防止 model 学会"短=expert，长=failure"这种 spurious correlation。

**4. C51 distributional progress prediction**。不 regress scalar 而是 predict categorical distribution，让 model 能表达 uncertainty。这在 ambiguous frame 上很重要。

**5. Video rewind augmentation**。从 single expert trajectory 合成 failure data，explicit model "undoing progress" 这个 RL 里常见的 failure mode。Ablation 证明它贡献最大。

---

## Open questions

- Preference supervision 为什么能帮助 progress prediction？作者说有 "mutual reinforcement effect" 但没 fully explain。我的猜测是 global ordering constraint 让 VLM 学到更 structured 的 internal representation。
- Frame subsampling 到 32 帧对 long-horizon task 可能不够。Hierarchical temporal modeling 可能是 future direction。
- Vision-only 无法感知 contact force、grasp stability。Multi-modal reward model（vision + force + tactile）是 natural extension。
- Generalization 只在 manipulation 上验证了，locomotion、navigation、social interaction 没有 cover。

---

Project page: https://robometer.github.io/

---

# ROBOMETER 深度解析:构建 scalable 的 general-purpose robotic reward model

## 1. 核心 Motivation:为什么 prior work 不够 scale

Andrej, 这篇 paper 的核心洞察其实非常 elegant,我先从 motivation 说起。

传统 robotic reward modeling 走的是一条"绝对 progress"路线:给定 expert demo,linearly interpolate 一个从 0 到 1 的 progress label,然后 regress。这个范式在 expert demo 上 work 得很好,但遇到 real-world data 就崩溃了。原因有两层:

**第一层:labeling ambiguity**。Failed trajectory 里的 progress 是 fluctuating 的——机器人可能走对了 80% 然后掉下来,progress 应该怎么标?标 0.8?标 0?标一个随时间递减的曲线?这是一个 ill-posed 问题。

**第二层:data waste**。Real-world robot data collection 里 failure trajectory 是 abundant 的——RL exploration、compounding execution errors、noisy teleop 都会产生大量 suboptimal data。这些 data 如果无法被 reward model 利用,就被丢掉了,极其浪费。

ROBOMETER 的 key insight 借鉴自 human cognition 里的 comparative judgment 机制([Laming 1984](https://doi.org/10.1111/j.2044-8317.1984.tb00867.x); [Stewart et al. 2005](https://doi.org/10.1037/0033-295X.112.4.881); [Sharif & Oppenheimer 2016](https://doi.org/10.1177/0956797616648539))。人类 internalize calibrated scale 的时候,relative comparison 比 absolute judgment 更 fundamental。把这个 intuition 搬到 reward modeling 上:

- **Progress labels** anchor reward magnitude 沿着单条 trajectory(intra-trajectory supervision)
- **Pairwise preferences** impose global ordering constraints across trajectories(inter-trajectory supervision)

这两者是 complementary 的。Progress 给你 absolute scale 的 grounding,preference 给你 cross-trajectory 的 ordering 约束。组合起来,你可以从 unlabeled failure trajectory 里学到东西——只需要 relative comparison,不需要 absolute score。

这个 insight 让我想到 RLHF 里 reward model 也是 preference-based 的([Christiano et al. 2017](https://arxiv.org/abs/1706.03741)),但有一个关键区别:RLHF 用 preference 作为 primary supervision,而 ROBOMETER 用 preference 作为 auxiliary signal 来 complement progress prediction。这个 design choice 很重要,后面我会展开。

---

## 2. RBM-1M Dataset:数据组成与 design philosophy

数据集的 design philosophy 很值得注意。Table IX 列出了完整的 1,059,370 条 trajectory,来自 21 个 robot embodiments。这里有一个 subtle 但重要的 design choice:

> "Rather than maximizing trajectory quantity, RBM-1M focuses on viewpoint, scene, and embodiment diversity."

这个 statement 透露出作者对 scaling law 的理解——不是盲目堆 data,而是堆 diversity。让我 break down 数据组成:

| Category | 代表 dataset | 轨迹数 | 角色 |
|---------|------------|-------|------|
| Expert demos | OXE Mix | 449,475 | Progress supervision 的主源 |
| Long-horizon bimanual | AGIBotWorld-Alpha subset | 216,911 | 多 skill 场景 |
| Humanoid bimanual | Galaxea Open World | 108,118 | Embodiment diversity |
| Counterfactual-labeled | RoboReward OXE Mix | 45,072 | 加入 RoboReward 的 pseudo-failure data |
| Paired human-robot | RH-20T, H2R, PH2D, MotIF | ~36k | Embodiment-invariant representation |
| **Mixed expertise** | RoboArena, SOAR, FAILSafe, RACER, AutoEval, LIBERO failures, Fino-Net | ~140k | **Failure data 的核心来源** |
| Human-only | Epic-Kitchens | 37,030 | Scene diversity |

几个关键 observations:

**(1) "Preference Only" data 的存在**。有些 dataset 的 trajectory termination time 和实际 task completion 时间严重不同(teleop delay)。这些 data 如果用来做 progress prediction 会引入 noise,但 ROBOMETER 的 design 让这些 data 仍然可以用于 preference prediction。这是一个 elegant 的 design——preference supervision 对 noisy end-frame 更 robust,因为它只关心 relative ordering。

**(2) Failure data 的多样性**。Mixed expertise data 包含:
- RoboArena: real-world policy evaluation failures,带 partial progress score
- SOAR: autonomous VLM-guided rollouts,success/fail label 由 VLM 生成(很 noisy,作者用 Qwen3-VL-4B filter 掉了 45%)
- FAILSafe: ManiSkill simulation failures
- RACER: RLBench non-prehensile failures
- LIBERO: 作者自己生成的,通过给 demo action 加 Gaussian noise

这种 diversity 很关键——failure mode 是高维空间里的 manifold,需要 diverse failure data 才能 cover。

**(3) Epic-Kitchens 的 inclusion**。Human-only data 在 robot reward model 里听起来 counterintuitive,但作者的 hypothesis 是 Epic-Kitchens 的 background scene diversity 和 clutter 帮助 model 学到更 general 的 visual-semantic representation。EgoDex 试过但不 work,作者 speculate 是因为缺乏 scene diversity。

**Frame downsampling**:所有 trajectory downsample 到 max 32 frames,image shortest edge 240 pixels。总数据量 ~6TB。

---

## 3. Architecture:Token-level design 的精妙之处

这是 paper 里我最喜欢的部分。ROBOMETER 用 Qwen3-VL-4B-Instruct 作为 backbone,但 token 设计非常 thoughtful。

### 3.1 Token 序列构造

核心 idea 是在一个 causal VLM 里同时处理两个 video trajectory,通过插入 learned special tokens 来 extract 不同的 prediction。Eqn (1) 展示了 token sequence:

```
Tok(l) <|video_start|> [Tok(o_t^1) <|prog_token|>]_{t=1}^T 
<|split_token|> [Tok(o_t^2)]_{t=1}^T <|pref_token|>
```

这里:
- `Tok(l)`:language instruction 的 token
- `Tok(o_t^i)`:第 i 个 trajectory 第 t 帧的 visual token
- `<|prog_token|>`:插在第一个 trajectory 每一帧后面的 learned token,用于 predict 那一帧的 progress
- `<|split_token|>`:分隔两个 trajectory 的 separator
- `<|pref_token|>`:放在序列末尾的 learned token,attend to 两个 trajectory 来做 preference prediction

为什么这么设计?有几个深层原因:

**(a) Causal mask 的 natural 利用**。Qwen3-VL 是 causal decoder,所以 `<|prog_token|>` at position t 只能 attend to frames 1..t of $o^1$。这天然就给你一个 dense,frame-level progress estimate——第 t 帧的 progress 只依赖前面看到的 frames,符合时间因果性。这比用一个 bidirectional encoder 然后做 per-frame prediction 要 elegant 得多,而且 inference 时可以 streaming。

**(b) 为什么只在 $o^1$ 插 prog_token**。Inference 时只需要 predict 单条 trajectory 的 progress,所以只在第一个 trajectory 插入。如果 $o^2$ 也插 prog_token,这些 token 会 attend 到 $o^1$(因为 causal mask),引入不想要的 cross-trajectory contamination。

**(c) 为什么 `<|pref_token|>` 放末尾**。Preference 需要 compare 两个 trajectory,所以这个 token 必须能 attend to 两个 trajectory 的所有 frames。放在末尾 + causal mask = 能 attend 到前面所有 token。

**(d) Fixed length T for both trajectories**。这是一个 anti-shortcut design:如果不固定长度,model 可能学会"短 trajectory = expert,长 trajectory = failure"这种 spurious correlation。固定 T 强制 model 从 content 判断 quality,不从 length。

### 3.2 MLP Heads

三个 lightweight MLP head 接在 VLM hidden state 上:
- `MLP_progress`:接在 `h_{<|prog_token|>,t}` 上,output 一个 N-bin categorical distribution(N=10, C51 style)
- `MLP_success`:同样接在 `h_{<|prog_token|>,t}` 上,output binary success probability
- `MLP_pref`:接在 `h_{<|pref_token|>}` 上,output 单个 logit 表示 trajectory 1 是否 preferred over trajectory 2

MLP config:2-layer MLP + LayerNorm + GELU + dropout 0.1,hidden dim 2048(是 Qwen hidden size 4096 的一半,heuristic choice)。

### 3.3 为什么用 dedicated `<|pref_token|>` 而不是 Bradley-Terry

这是 Appendix B-2d 里一个很关键的 design choice,值得深挖。Table XII 的 ablation 显示:

| Pref. Loss | VOC r | Kendall τ | Succ-Fail Diff |
|-----------|-------|----------|----------------|
| Bradley-Terry | 0.862 | 0.325 | 0.242 |
| BCE (dedicated token) | 0.948 | 0.655 | 0.320 |

BT loss 算的是给每条 trajectory 一个 scalar score,然后通过 softmax 比较。而 ROBOMETER 的设计是让 `<|pref_token|>` 直接 attend 到两个 trajectory 的所有 token。

这背后的 intuition:pre-trained VLM 的 attention 机制是 model 最强的 reasoning inductive bias。BT loss 把 cross-trajectory comparison 退化成两个 scalar 的比较,浪费了 VLM 的 attention 能力。而 dedicated token 让 VLM 能 explicitly 做 cross-video attention——一个 trajectory 的 frame 可以直接 attend 到另一个 trajectory 的对应 frame,做 fine-grained 对比。

这个 idea 来自 language reference game literature([Monroe et al. 2017](https://doi.org/10.1162/tacl_a_00064); [Mitra et al. 2024](https://arxiv.org/abs/2404.01261); [Bao et al. 2022](https://arxiv.org/abs/2203.13685))。

---

## 4. Training Objectives:三个 loss 的数学细节

Total loss: $\mathcal{L} = \mathcal{L}_{\text{pref}} + \mathcal{L}_{\text{prog}} + \mathcal{L}_{\text{succ}}$

### 4.1 Preference Loss (Eqn 2)

$$\mathcal{L}_{\text{pref}} = -\left[\mathbb{I}_{y=1} \log \sigma\left(\text{MLP}_{\text{pref}}\left(h_{<|\text{pref\_token}|>}\right)\right) + \mathbb{I}_{y=2} \log\left(1 - \sigma\left(\text{MLP}_{\text{pref}}\left(h_{<|\text{pref\_token}|>}\right)\right)\right)\right]$$

变量解释:
- $y \in \{1, 2\}$:ground-truth preferred trajectory index(1 表示 trajectory 1 更好,2 表示 trajectory 2 更好)
- $\mathbb{I}_{y=k}$:indicator function,当 $y=k$ 时为 1,否则为 0
- $\sigma(\cdot)$:sigmoid function,$\sigma(x) = 1/(1+e^{-x})$
- $\text{MLP}_{\text{pref}}$:preference head,输出 scalar logit
- $h_{<|\text{pref\_token}|>}$:`<|pref_token|>` 的 hidden state embedding

这本质是 binary cross-entropy。当 $y=1$,maximize $\log\sigma(\text{logit})$;当 $y=2$,maximize $\log(1-\sigma(\text{logit}))$。

### 4.2 Progress Loss (Eqn 3) - C51 distributional formulation

这里有个非常 thoughtful 的设计选择。他们**没有**直接 regress scalar progress,而是用 C51 formulation([Bellemare et al. 2017](https://arxiv.org/abs/1707.06870))把 progress 离散化成 N=10 个 bin。

Ground-truth progress target at frame $t$:$p_t = t/T$ for $t \in \{1, ..., T\}$

这个 scalar 被投影到 categorical distribution 上,通过 linear interpolation between neighboring bin centers。Progress head 输出 $\hat{p}_t \in \Delta^N$(N-simplex 上的 categorical distribution)。

$$\mathcal{L}_{\text{prog}} = \frac{1}{T}\sum_{t=1}^{T} \text{CE}\left(\text{Proj}(p_t), \text{MLP}_{\text{progress}}\left(h_{<|\text{prog\_token}|>,t}\right)\right)$$

- $\text{Proj}(p_t)$:把 scalar $p_t$ 投影成 categorical distribution 的 operation
- $\text{CE}$:cross-entropy
- $\Delta^N$:N-dimensional probability simplex(所有元素非负且和为 1)

Inference 时 recover continuous estimate:
$$\hat{p}_t = \sum_{i=1}^{N} z_i \hat{p}_{t,i}$$

- $z_i$:第 $i$ 个 bin 的 fixed center(对于 [0,1] 区间 N=10 bins,$z_i \in \{0.05, 0.15, ..., 0.95\}$)
- $\hat{p}_{t,i}$:model 预测的 第 $i$ 个 bin 的 probability

**为什么用 distributional 而不是 scalar regression?** 这是 build intuition 的关键。我的理解:

1. **Multi-modal uncertainty**:当 frame $t$ 的 progress 是 ambiguous 的(比如 suboptimal trajectory 里 robot 重复 motion),progress 的 ground truth 本身是 uncertain 的。Scalar regression 强制 model 输出一个 point estimate,distributional 让 model 表达 uncertainty。
2. **Gradient stability**:scalar regression 在 progress 接近 0 或 1 时 gradient 容易 saturate;C51 的 cross-entropy 对所有 bin 都有 well-behaved gradient。
3. **Downstream compatibility**:distributional reward 在 RL 里有理论优势(reward distribution 而非 expected reward),对未来可能的 distributional RL integration 友好。

### 4.3 Success Loss (Eqn 4)

$$\mathcal{L}_{\text{succ}} = \text{BalancedBCE}\left(s_{1:T}, [\text{MLP}_{\text{success}}(h_{<|\text{prog\_token}|>,t})]_{1:T}\right)$$

Target 定义:$s_t = 0$ for $t < T$,$s_t = 1$ for $t = T$。

但这里有 nuance:作者发现 teleop data 里 trajectory 往往在 task 完成后还有 trailing frames。所以他们 manual annotate 了每个 data source 的 "success cutoff"——比如 DROID 是 0.95(即 trajectory 95% 处算 success),Berkeley RPT 是 0.76,Galaxea 是 0.80 等等(Table XI)。这个 cutoff 用于 truncate training target,避免 trailing noise frames 污染 success supervision。

Success supervision 的另一个 subtle design:**只在 progress 严格低于 $\tau_{\text{succ}}$ 或严格等于 1.0 的 frames 上 apply success loss**。中间 transitional frames 被排除。这是因为接近 completion 的 transitional frame 视觉上 ambiguous,success label 不可靠。这个细节体现了对 real-world data 噪声的 respect。

BalancedBCE 用 per-batch adjusted class weights 处理 negative sample imbalance(success frame 只在最后,大量 negative)。

---

## 5. Data Sampling Strategies:三种 preference pair 构造

Section III-D 描述了三种 preference pair 构造策略,这是让 unlabeled failure data 可用的关键 mechanism。

### 5.1 Progress-Based Comparisons (Different Expertise)

从 mixed expertise dataset sample 两条 trajectory:
- $\tau_1$ with progress $p^1$
- $\tau_2$ with progress $p^2$
- 如果 $p^1 > p^2$,$y = 1$;否则 $y = 2$
- 特殊情况:expert demo($p=1$)vs unlabeled failure($p=\text{None}$),expert 总是 preferred

这个 strategy 让 model 学到"什么样的 execution quality 更好"。注意 unlabeled failure 不需要 progress label——只需要知道它比 expert 差就行。

### 5.2 Instruction Negatives (Different Tasks)

Sample $\tau_1$ 和 $\tau_2$ with $l^1 \neq l^2$。Random 选一个 instruction 作为 conditioning text $l$,preference label 指向对应 trajectory,另一条 trajectory 的 progress target 设为 0。

这个 strategy 的作用:**ground reward 在 language instruction 上**。它 enforce"correct behavior for wrong task = 0 reward"。这对 multi-task setting 至关重要——防止 model 给 semantically wrong 但 visually plausible motion 高 reward。

### 5.3 Video Rewind (Augmented Failures)

这是最 clever 的 augmentation。从 single expert trajectory $\tau$ sample indices $1 \leq t_1 < t_2 < t_3 \leq T$:
- **Chosen** $o^c = o_{t_1:t_3}$(正常 forward)
- **Rejected** $o^r = [o_{t_1:t_3}, o_{t_3-1:t_2}]$(先 forward 再 rewind)或者 $[o_{t_3:t_1}]$(完全 reverse)

Rejected sequence 的 progress target 是 **decreasing**,matched to frame indices。

这个 augmentation 显式 model "undoing progress"——RL 探索时常见的 failure mode(policy 走对了然后又走回来)。Table XVIII 的 ablation 显示 rewind 是所有 augmentation 里贡献最大的:去掉它 LIBERO-90 的 Succ-Fail Diff 从 0.455 掉到 0.241,VOC r 从 0.976 掉到 0.818。

### 5.4 Subsequence Trimming

Random sample T frames with uniform start/end indices from full video,防止 model overfit to fixed trajectory length。

### 5.5 Same-source vs cross-source for different-task

以概率 $\rho_{\text{same}} = 0.5$ 从同一 dataset sample different-task pair(disicourage dataset-specific visual cues),$1 - \rho_{\text{same}} = 0.5$ 从不同 dataset sample(encourage robustness to domain shift)。

---

## 6. Experimental Results:关键 numbers 与 interpretation

### 6.1 Main Reward Evaluation (Table I)

| | GVL | VLAC | RoboDopamine | RoboReward-4B | RoboReward-8B | ROBOMETER (RR data) | ReWiND | **ROBOMETER (RBM-1M)** |
|---|---|---|---|---|---|---|---|---|
| VOC r (ID) | 0.16 | 0.16 | 0.13 | 0.77 | 0.82 | 0.84 | 0.46 | **0.92** |
| VOC r (OOD) | 0.21 | 0.17 | 0.08 | 0.88 | 0.88 | 0.93 | 0.51 | **0.95** |
| Kendall τ_a (OOD) | 0.19 | 0.08 | 0.11 | 0.50 | 0.47 | 0.55 | 0.01 | **0.66** |

几个重要 observations:

**(a) GVL/VLAC/RoboDopamine 表现差**。GVL 用 GPT-5-mini zero-shot prompt,VOC r 只有 0.21。这 confirms pre-trained VLMs 直接用做 zero-shot reward 是 noisy 的。VLAC 和 RoboDopamine 都是 fine-tuned 但数据量小(300k 和 100k)。

**(b) RoboReward-4B vs 8B 差距小**。在 OOD eval 上,4B 和 8B 几乎一样(0.88 vs 0.88 VOC,0.50 vs 0.47 Kendall)。这暗示 RoboReward 的 bottleneck 不在 model size,而在 training paradigm。

**(c) ROBOMETER 用 RoboReward data 也更好**。VOC r 0.93 vs 0.88,Kendall 0.55 vs 0.50。Same data,better methodology。这是 dual objective + augmentation 的功劳。

**(d) RBM-1M 带来的进一步提升**。VOC 0.95,Kendall 0.66。Scale + diversity + failure data 的综合贡献。

### 6.2 RoboRewardBench (Table II)

| Model | MAE |
|-------|-----|
| ROBOMETER | 0.72 |
| ROBOMETER (RoboReward data only) | 0.75 |
| RoboReward-4B | 0.85 |
| Qwen3-VL-4B-Instruct | 1.03 |
| RoboReward-8B | 0.67 |
| GPT-5-mini | 0.69 |

这里 RoboReward-8B 和 GPT-5-mini 比 ROBOMETER 略好,作者的解释很有道理:RoboRewardBench 用 5 个 discrete labels 和 final-frame-only evaluation protocol,这对更大 model 有利。而 ROBOMETER 的 dense reward formulation 在更 demanding 的 OOD eval 上才体现优势。

### 6.3 Ablation:为什么 ROBOMETER work (Table IV)

这是理解 paper 的核心。三个 hypothesis:

| Ablation | VOC r | Kendall τ | Suc-Fail Diff |
|----------|-------|----------|----------------|
| H1 Prog. Only (LIBERO-90) | 0.96 | 0.63 | 0.11 |
| H1 +Preference | 0.90 | 0.74 | 0.22 |
| H2 +Failed Data | 0.98 | 0.92 | 0.46 |
| H3 ReWiND Arch. | 0.48 | -0.14 | -0.02 |

**H1**:即使没有 failure data,加 preference prediction 就能提升 Kendall τ 从 0.63 → 0.74。这说明 global comparative constraint 本身就 induce 更 structured 的 internal reward representation。VOC r 从 0.96 略降到 0.90,这是 trade-off——preference 让 model 更会 ranking 但可能 slightly less calibrated on absolute progress。

**H2**:加 failure data,Kendall τ 跳到 0.92,Suc-Fail Diff 从 0.11 跳到 0.46(4× improvement)。这是最大的 gain。Failure data 提供 contrastive signal,model 才能学会区分 suboptimal 和 successful。

**H3**:换掉 pre-trained VLM backbone,用 ReWiND 的 transformer architecture(500M params,32 layers),所有 metric 崩溃。Kendall τ 变成 -0.14(比 random 还差)。这 strongly confirms:**large-scale multimodal pre-training 是 essential 的**,reward learning 需要 generalizable visual-semantic representation,from-scratch transformer 学不到。

### 6.4 RL with Ablated Reward Models (Figure 5)

这个 experiment 把 reward quality metric 的改进 translate 到 policy learning success rate。在 LIBERO-90 的两个 task 上,ROBOMETER(H2 model)比 sparse reward 和 H1 ablation 都有 2-4× 的 sample efficiency 提升。这证明了 reward quality metric(VOC, Kendall)和 downstream policy performance 的 correlation。

---

## 7. Downstream Applications:四个 use case

### 7.1 Automatic Online RL (Figure 6)

用 DSRL([Wagenmaker et al. 2025](https://arxiv.org/abs/2410.21201))steer π_0 policy pre-trained on DROID。

Single-stage task: "put the bowl on the table"
- π_0 baseline: 20% success
- DSRL + ROBOMETER: 85% success(45 min,10k steps)
- DSRL + RoboReward: 55% success

Multi-stage task: "put corn in pot" → "put lid on pot"
- DSRL + ROBOMETER: 70% success
- DSRL + RoboReward: 20% success(no improvement)

RoboReward 的关键 failure mode:在 cluttered environment 里给 wrong object manipulation 打 maximum reward。Table XXII 量化:RoboReward 45 个 false positives,ROBOMETER 0 个。这个 difference 直接导致 RL 学到 wrong behavior。

### 7.2 Offline RL with Mixed Data (Figure 7)

用 IQL([Kostrikov et al. 2021](https://arxiv.org/abs/2110.06169))在 SO-101 robot 上 train。Expert + noisy trajectory mix。

关键 finding:ROBOMETER 的 dense reward 让 IQL 在更低 discount factor γ=0.9 上 work best,而 sparse reward 和 RoboReward 需要更高 γ。这符合理论——dense reward 减少 long-horizon credit assignment 需求,允许 trajectory stitching with smaller γ,降低 value function variance。

Average 2.4× success rate improvement over best baseline。

### 7.3 Data Filtering & Retrieval (Figure 8)

从 bimanual play dataset(unlabeled,multi-task)retrieve top-100 subtrajectories per task。

ROBOMETER 用两种 mode:
- **Pref mode**:pairwise comparison,aggregate 成 win matrix,rank by estimated preferences
- **Prog mode**:per-timestep progress value + VOC,select top VOC scores

Retrieval relevance:ROBOMETER 一致高于 RoboReward、SigLIP、STRAP。

下游 policy learning:LoRA fine-tune π_0.5 on retrieved data。ROBOMETER-retrieved data 训出的 policy success rate 平均 4.5× 高于 best baseline。Baselines retrieve 了太多 failed/suboptimal 但 task-relevant 的 subtrajectory,degrade policy performance。

### 7.4 Failure Detection (Table V, Figure 9)

在 MIT Franka 上 100 条 trajectory(30 success,70 failure),zero-shot detection。

ROBOMETER average F1 = 0.81,最高。VLAC 倾向 flag everything as failure(high TPR,low TNR)。GPT-5-mini 倾向 predict success(high TNR,low TPR)。RoboReward-4B 中间。

Figure 9 展示了 detection mechanism:
- **Irreversible failure**(object drop):progress sharp regression,detect shortly after event
- **Non-terminal failure**(oscillation):progress stagnates/oscillates without convergence
- **Semantic failure**(wrong object):progress consistently low despite smooth execution

Detection method:Pearson correlation between progress values and time over sliding window,flag failure when correlation < threshold(-0.5)。Window size 5-9。

---

## 8. Limitations 与 Future Directions

作者自己承认的 limitations:

1. **Frame-based, temporally subsampled**:只处理 8 frames per trajectory,fine-grained temporal dynamics 和 long-horizon structure 无法 capture。未来可以加 denser temporal modeling。

2. **Failure mode coverage 不足**:real-world failure modes 是高维 manifold,training data 可能无法 cover 所有 rare/subtle/task-specific failures。

3. **缺乏 latent physical state access**:vision-only,无法感知 contact force、grasp stability、compliance。这些 failure-driven factors 在 visually observable 之前无法 detect。

Future work suggestions:
- VQA-style supervision 推理 task structure
- Off-domain data 改善 generalization([Zhang et al. 2026](https://arxiv.org/abs/2601.15224))
- Systematically curated failure datasets([Tian et al. 2026](https://cmu-intentlab.github.io/pdf/tian_icml_26_position.pdf))

---

## 9. 我的 interpretation:为什么这个 work 重要

Andrej, 从你的 perspective 我觉得这个 work 有几个深层 significance:

**(a) Reward model 的 "scaling hypothesis" 验证**。RBM-1M(1M trajectory)vs RoboReward(45k)vs VLAC(300k)vs RoboDopamine(100k)。数据显示 scale + diversity + failure data 三者都重要,但 data composition 比 raw quantity 更关键。这和 LLM scaling law 的精神一致,但有 robotics-specific 的 nuance。

**(b) Preference + Progress 的 dual objective 类比了 LLM 里的 RLHF + SFT**。SFT 给 absolute grounding,RLHF 给 relative alignment。ROBOMETER 的 progress loss 像 SFT,preference loss 像 RLHF。这种类比提示 robotics 可能正在重走 LLM 的路,只是更难 because of the grounding problem。

**(c) VLM as reward model 的 emergent capability**。H3 ablation 证明 from-scratch transformer 完全不行,pre-trained VLM 是必需的。这暗示 VLM 的 visual-semantic pre-training 已经 encode 了 task understanding,reward learning 只是把这个 understanding 适配到 reward space。这和 LLM as reward model 的发现([Ma et al. 2025](https://arxiv.org/abs/2403.20379))呼应。

**(d) Test-time compute via world model ranking**。Appendix G 的 DreamZero experiment 虽然只是 proof-of-concept,但思路很有意思——用 reward model rank 6 个 candidate future trajectories from world model,3.5× success rate improvement。这是 test-time compute scaling 在 robotics 的 instance,类比 LLM 里的 best-of-N sampling。

**(e) Preference token 的 architectural innovation**。让 VLM 通过 cross-attention 直接 compare 两个 video,比 BT loss 的 scalar comparison 强得多。这个 idea 来自 language reference game,但移植到 robotics reward modeling 是新的。Table XII 的 0.325 → 0.655 Kendall τ improvement 量化了这个 architectural choice的价值。

---

## 10. 与 broader landscape 的连接

这个 work 坐在几个 research thread 的 intersection:

1. **Foundation model as reward** ([Eureka](https://arxiv.org/abs/2310.12931), [Language to Rewards](https://arxiv.org/abs/2306.08647), [Text2Reward](https://arxiv.org/abs/2306.09633)):LLM/VLM 生成 reward code 或 shaping function。ROBOMETER 走的是 fine-tune VLM 直接 predict dense reward,更 general 但需要 training data。

2. **VLM zero-shot as reward** ([VLM-RM](https://arxiv.org/abs/2402.05641), [GVL](https://arxiv.org/abs/2403.20379)):直接 prompt pre-trained VLM。ROBOMETER 的 GVL baseline(GPT-5-mini)只有 0.21 VOC r,证明 zero-shot 不够,fine-tuning 必要。

3. **Robot-specific progress predictor** ([ReWiND](https://arxiv.org/abs/2411.00126), [VLAC](https://arxiv.org/abs/2509.15937), [RoboDopamine](https://arxiv.org/abs/2512.23703)):fine-tune 但数据量小。ROBOMETER 的 contribution 是 scale + dual objective。

4. **RLHF in robotics** ([Christiano et al.](https://arxiv.org/abs/1706.03741), [Biyik et al.](https://arxiv.org/abs/2007.05116), [Hejna & Sadigh](https://arxiv.org/abs/2208.11695)):human preference-based。ROBOMETER 用 synthetic preference,不需要 human labeling。

5. **VLA + reward co-training** ([VLAC](https://arxiv.org/abs/2509.15937), [π_0.6*](https://arxiv.org/abs/2511.14759), [Self-improving embodied FMs](https://arxiv.org/abs/2509.15937)):joint action + reward prediction。ROBOMETER 是 reward-only,但可以作为 plug-in 给 VLA。

---

## 11. 我想强调的几个 technical subtleties

**(1) Success cutoff 的 per-dataset annotation**。Table XI 列了每个 dataset 的 success cutoff(0.70-1.00)。这个看起来 mundane 的 detail 其实很重要。Teleop data 的 trajectory termination 不等于 task completion,operator 有 delay。用 wrong cutoff 会让 success label noisy。作者花 2 min/dataset manually 标,这个 effort 的 ROI 很高。

**(2) SOAR data 的 VLM filtering**。SOAR dataset 的 VLM-generated labels 很 noisy,作者用 Qwen3-VL-4B 过滤掉 45% 的 trajectory。Filtering prompt 分 stage,empirically 改善 quality。这暗示 real-world robot data 的 label noise 是 significant bottleneck,VLM-assisted data curation 越来越重要。

**(3) Prompt engineering for training**。Appendix B-1b 给出了完整的 training prompt:
```
Given these two trajectories for the task "{task}", evaluate which one makes more progress towards the task. Return A for the first trajectory and B for the second trajectory. Additionally, predict the task progress at each frame of the first trajectory as a float between 0 and 1, where 0 corresponds to the initial state and 1 corresponds to task completion. If the robot is not performing the specified task, predict 0 progress.
```

最后一句话 "If the robot is not performing the specified task, predict 0 progress" 是 instruction grounding 的 explicit prompt-level enforcement。

**(4) Inference time 的 practicality**。Appendix G 提到 ROBOMETER forward pass 只需 0.6-1 second,而 DreamZero world model forward pass 需要 ~27 second。这意味着 ROBOMETER 在 real-time RL loop 里是 affordable 的,不是 bottleneck。

**(5) LoRA fine-tuning 的 efficiency**。Appendix F 显示 LoRA(76M params)on ROBOMETER-4B 在 RoboFAC 上 fine-tune 500 steps,8 小时单 GPU,performance 几乎等同 full fine-tuning。这让它 practical for domain adaptation。

---

## 12. 一个 deeper question:为什么 preference supervision 帮助 progress prediction?

H1 ablation 显示,即使没有 failure data,加 preference 就能提升 Kendall τ。这个 "mutual reinforcement effect" 作者在 abstract 里提到。让我 speculate 一下 mechanism:

**Hypothesis 1: Better representation learning**。Preference loss 是 auxiliary task,regularize VLM hidden state 学习更 task-relevant 的 feature。这类似 multi-task learning 的 general benefit。

**Hypothesis 2: Global structure constraint**。Progress loss 只约束单条 trajectory 内的 monotonicity。Preference loss 约束 cross-trajectory 的 ordering。这两者 jointly 约束 reward function 的 shape,防止 model 学到 locally monotonic 但 globally inconsistent 的 reward。

**Hypothesis 3: Attention head specialization**。`<|pref_token|>` 的 cross-trajectory attention 可能 force VLM 学到更 fine-grained 的 task-progress-relevant visual feature,这些 feature 通过 shared backbone benefit progress prediction。

**Hypothesis 4: Hard negative mining**。Different-task preference pairs 是 hard negative,force model 学到 task-grounded representation 而不是 visual-motion pattern。

我觉得 Hypothesis 2 和 3 都 contribute,但目前的 ablation 无法 fully disentangle。这是一个 future work 的 direction——mechanistic interpretability of how preference supervision shapes VLM internals。

---

## 13. 与你之前的 work 的潜在连接

Andrej, 从你的 [micrograd](https://github.com/karpathy/micrograd)、[nanoGPT](https://github.com/karpathy/nanoGPT)、[llm.c](https://github.com/karpathy/llm.c) 的角度,以及你对 software 2.0 和 AI 的 thinking,这个 work 有几个值得 connect 的点:

**(a) Reward model as "software 3.0"**。你之前提过 software 1.0(explicit code)、software 2.0(neural network weights learned from data)。ROBOMETER 这种 VLM-based reward model 其实是 software 3.0 的 instance——用 natural language + learned model 来 specify behavior,而不是手工 reward function。

**(b) Token as interface**。ROBOMETER 的 `<|prog_token|>` 和 `<|pref_token|>` 是 learned interface,类似 CLS token in BERT。这种 "insert learned token to extract structured output from VLM" 的 pattern 越来越 common,可能成为 future VLA/VLM architecture 的标准 design。

**(c) Test-time compute scaling**。Appendix G 的 DreamZero + ROBOMETER ranking 是 best-of-N over world model rollouts。这和你最近对 test-time compute 的 interest 一致。Robotics 的 test-time compute 还没被充分探索,ROBOMETER 提供了一个 scalable reward signal 让 best-of-N 变得 practical。

**(d) Data curation as first-class concern**。RBM-1M 的 curation 细节(VLM filtering of SOAR、per-dataset success cutoff、preference-only data flag)体现了 robotics data curation 的 complexity 远超 LLM text data。这和你强调的 "data is the bottleneck" 完全一致。

---

## 14. Open questions 与我的 critique

**(a) Preference supervision 的 quality ceiling**。ROBOMETER 用 synthetic preference(video rewind、different-expertise、different-task)。这些是 heuristic-generated,可能 miss 人类会做的 subtle preference。Future work 可以 explore 人类 preference annotation 的 marginal value,但 RLHF 的经验是 synthetic preference often sufficient for alignment。

**(b) Frame subsampling 的 information loss**。32 frames max 可能对 long-horizon task 不够。ROBOMETER 的 limitation 里也提了。Future work 可能需要 hierarchical temporal modeling,或者 recurrent processing。

**(c) Metric 的 validity**。VOC Pearson r 和 Kendall τ 是 proxy metric。真正验证需要 downstream RL/IL performance,而 paper 提供了 4 个 application 的 evidence。但更 systematic 的 benchmark 还是 missing——类似 ImageNet 之于 vision 的标准化 benchmark。

**(d) Failure mode 的 coverage**。Figure 9 展示的 failure 都比较 visually salient(drop、oscillation、wrong object)。对于 visually subtle failure(grasp unstability、wrong force、compliance issue),vision-only model 有 fundamental limitation。这需要 multi-modal sensing(force-torque、tactile)的 integration。

**(e) Generalization 到 completely unseen task type**。RBM-1M 虽然有 21 embodiment,但都是 manipulation。Locomotion、navigation、social robot interaction 的 task type 没有 cover。Generalization claim 的 scope 应该 qualify。

---

## 15. 总结:这个 work 在 landscape 里的位置

ROBOMETER 代表了 robotic reward modeling 从 task-specific、small-scale、expert-only 的 paradigm 向 general-purpose、large-scale、mixed-expertise 的 paradigm 的 transition。它的 contribution 是 multi-fold:

1. **Conceptual**:dual objective (progress + preference) 让 failure data 可用,解决 real-world robot data 的 core bottleneck。
2. **Architectural**:learned special token + cross-video attention 比 scalar BT loss 更好 leverage VLM 的 attention capability。
3. **Data**:RBM-1M 是 robotics reward learning 里最大的公开 dataset,且包含 intentional failure data。
4. **Empirical**:在 4 个 downstream application 上 2.4-4.5× improvement,validate 了 reward quality 的 practical impact。

这个 work 让我更坚信 robotics 正在 follow LLM 的 trajectory:large-scale pre-training + instruction tuning + RLHF-style alignment。只是 robotics 的 grounding 问题让每一步都更难。ROBOMETER 是这个 trajectory 上的一个 significant milestone,因为它证明了 synthetic preference supervision 可以 scale,不需要 human labeling。

Project page: https://robometer.github.io/

---

如果你想 dig deeper,我建议重点关注几个 future direction:
- Hierarchical reward model for long-horizon tasks
- Multi-modal reward model(vision + force + tactile)
- Reward model interpretability(哪些 attention head 在做什么)
- Test-time compute scaling for robotics via reward-ranked sampling

希望这个 deep dive 帮你 build intuition。如果有特定部分你想更深入,比如 IQL 的 offline RL setup 或者 DreamZero integration,我可以再 expand。
