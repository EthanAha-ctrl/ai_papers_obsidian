---
source_pdf: What to Ignore, What to React Visually Robust RL.pdf
paper_sha256: 1dece03210b63b2029e8ead70e4f51d86416af31819bbee82141358bad252ddf
processed_at: '2026-08-13T04:14:20-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PAIR-VLA

Andrej，我换一种讲法，像我们在 NeurIPS 走廊里聊天那样，从最底层的 intuition 开始，再往上叠技术细节。

---

## 一句话总结

RL fine-tuning 一个 VLA model 时，光靠 task reward（成功/失败）**不够**——reward 只告诉你"做对了没"，不告诉你"刚才那个 visual change 你该不该理"。这篇 paper 的 core insight 就是：**主动构造两类视觉扰动，一类告诉 policy "这个变化 ignore 掉"，一类告诉它 "这个变化你得 react"**。两条信号加到 PPO 上，policy 就学会了区分"应该装作没看见"和"应该改变行为"。

---

## 1. 问题到底出在哪

先讲 standard PPO fine-tuning VLA 的 pain point。

假设你在 train 一个 robot 抓杯子放盘子上。场景里有：
- target object（杯子）
- receptacle（盘子）
- distractors（随便摆几个干扰物）
- table texture
- lighting

PPO 给的 reward 就一句话：**抓起来放对位置 +1，否则 0**。可能再加个 grasp 的 dense reward。

问题来了。Training 时 policy 见过蓝色 distractor A，没见过红色 distractor B。Deployment 遇到 B，policy 可能懵了——但其实 B 对 task 毫无影响，policy 本该完全 ignore。

或者 target 杯子从朝上变成朝下了——这种变化 policy **必须 react**（grasp 策略完全不同），但 PPO reward 不会主动告诉你"这个变化 important"。

**根本痛点**: reward 只有 outcome supervision，没有 **behavior-level supervision**——它不说"哪些视觉变化要 ignore，哪些要 react"。

传统做法（domain randomization）的逻辑是：train 时见够多 variation，自然就 robust。但这个逻辑有个 leak：policy 可能学会"对一切变化都过度敏感"或者"对一切变化都麻木"。它没学到 **区分**。

---

## 2. PAIR-VLA 的核心 trick

核心 idea 就一个词：**paired views**。

每个 observation $o_t$，人工构造两个 "变异版本":

### 2.1 Task-preserving view $\tilde{o}_t^{\mathrm{prev}}$

把 $o_t$ 里 task-irrelevant 的东西改了，task-relevant 的东西保留。具体做：
- 移除 distractors
- 换 table texture / background

构造公式（paper 里的公式 5）：

$$\tilde{o}_t^{\mathrm{prev}} = m_t \odot o_t + (1 - m_t) \odot o_k^{\mathrm{bg}}$$

逐个符号拆：
- $m_t$：binary segmentation mask，1 表示 robot/target/receptacle 这些 task-relevant 像素，0 表示 background/distractor
- $\odot$：element-wise 乘法
- $o_k^{\mathrm{bg}}$：从 $K$ 个 pre-rendered background 里随机选一个（训练前把所有物体隐藏，渲染 K 张不同 scene 的背景图）
- 整个公式：foreground 用原图，background 用一张随机的预存背景拼上去

Intuition：相当于给 policy 看"如果没 distractor、桌面是另一个样子"会怎样。

### 2.2 Task-altering view $\tilde{o}_t^{\mathrm{alt}}$

把 task-relevant 的东西改了——具体是 **target object 的 pose**：
- translation: 从 Gaussian 采样扰动
- rotation: 从 categorical 分布采样扰动
- 然后 re-render 整个 scene

Intuition：给 policy 看"如果杯子位置/朝向变了"会怎样。这种变化会改变 required grasp、motion、placement。

---

## 3. 两个 auxiliary objectives

有了 paired views，怎么把它们变成 training signal？用 KL divergence 在 action distribution 上做约束。注意是 **action distribution**，不是 representation——这点很关键，后面会展开。

### 3.1 Invariance objective（公式 2）

$$\mathcal{L}_{\mathrm{inv}}(\theta) = \mathbb{E}_{(o_t, l)}\left[D_{\mathrm{KL}}\left(\pi_\theta(\cdot | o_t, l) \ || \ \mathrm{sg}[\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{prev}}, l)]\right)\right]$$

逐个符号拆：
- $\theta$：policy 参数
- $\pi_\theta(\cdot | o_t, l)$：原始 observation 下的 action distribution
- $\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{prev}}, l)$：task-preserving view 下的 action distribution
- $l$：language instruction（episode 内固定）
- $\mathrm{sg}[\cdot]$：stop-gradient，让 $\tilde{o}_t^{\mathrm{prev}}$ 那边的 distribution 不更新参数，只当 fixed target
- $D_{\mathrm{KL}}(P || Q) = \sum_a P(a) \log \frac{P(a)}{Q(a)}$：KL 散度，measure 两个分布的差异

人话：**让 original view 的 action distribution 去逼近 task-preserving view 的 action distribution**。换句话说，"如果 distractor 没了、桌面变了，你的 action 应该跟原来一样"。

为什么 stop-gradient？想象两个人都在动，会互相追逐、震荡。让一个人站着不动（stop-gradient），另一个人走过去靠拢，稳定多了。这就是 BYOL 那套思路。

Reference: [BYOL - Grill et al., 2020](https://arxiv.org/abs/2006.07733)

### 3.2 Sensitivity objective（公式 3）

$$\mathcal{L}_{\mathrm{sens}}(\theta) = -\mathbb{E}_{(o_t, l)}\left[\min\left(c, \ D_{\mathrm{KL}}\left(\pi_\theta(\cdot | o_t, l) || \mathrm{sg}[\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{alt}}, l)]\right)\right)\right]$$

逐个符号拆：
- $\tilde{o}_t^{\mathrm{alt}}$：task-altering view
- $c$：clipping threshold，防止 KL 无限增长
- 前面的负号：要 **maximize** 这个 KL

人话：**让 original view 和 task-altering view 的 action distribution 尽量分开**。换句话说，"如果 target pose 变了，你的 action 应该明显不一样"。

为什么需要 $\min(c, \cdot)$ clip？没 clip 的话，policy 会找到 degenerate 解法——比如对 task-altering view 输出完全随机 action 来 maximize KL，但这个 random action 对 task 毫无意义。Clip 在 KL 达到 $c$ 后就停止 push，让 PPO 的 reward signal 接管，告诉它 "分开可以，但得往正确的方向分"。

### 3.3 合体（公式 4）

$$\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{PPO}}(\theta) + \alpha \mathcal{L}_{\mathrm{inv}}(\theta) + \beta \mathcal{L}_{\mathrm{sens}}(\theta)$$

- $\alpha, \beta > 0$：两个 auxiliary objective 的权重
- $\alpha$ 太大 → policy 对什么都麻木（过度 invariant）
- $\beta$ 太大 → policy 对什么都过度敏感（过度 sensitive）

两个力配合，shape 出 action space 里的 "不变区" 和 "应变区"。

**关键卖点**: 这两个 objective 只在 training 时用。Deployment 时 policy 结构原封不动，**零额外 inference cost**。

---

## 4. 为什么是 action distribution 而不是 representation

这是这篇 paper 和之前一大堆 invariant representation 工作的根本区别。

之前 CURL ([Laskin et al., 2020](https://arxiv.org/abs/2004.04136))、CPC、bisimulation ([Zhang et al., 2020](https://arxiv.org/abs/2006.10742)) 这些做法，都是在 **latent representation** 上做 invariance。但 representation invariant **不保证** action invariant——policy head 完全可能从一个 invariant representation 里读出不同的 action。这是 representation learning 和 policy learning 之间的 gap。

PAIR-VLA 直接在 action distribution 上做 invariance，end-to-end，没有这个 gap。Representation 怎么变它不管，它只管 "最终 action distribution 一致"。

类比：之前是规定"两个人的内心活动要一样"，PAIR-VLA 是规定"两个人最后做的决定要一样"。后者更直接，也更符合 task 的实际要求。

---

## 5. 实验数据走一遍

### 5.1 主结果（Table 1）

| Model | Method | Table Texture | Lighting | Target Pose | Clutter | Avg. | Δ Avg. |
|-------|--------|---------------|----------|-------------|---------|------|--------|
| OpenVLA | PPO | 86.98 | 72.14 | 83.59 | 68.88 | 77.90 | - |
| OpenVLA | Ours | 94.53 | 80.47 | 90.63 | 82.36 | 87.00 | **+9.10** |
| π0.5 | PPO | 63.54 | 28.54 | 56.46 | 36.46 | 46.25 | - |
| π0.5 | Ours | 80.21 | 51.67 | 69.38 | 50.21 | 62.87 | **+16.62** |

几个值得咂摸的点：

1. **π0.5 提升比 OpenVLA 大得多**（+16.62 vs +9.10）。猜测：π0.5 是 flow-matching model，action space 更复杂、PPO baseline 更弱，auxiliary objective 的相对收益更大。也暗示 flow-based VLA 在 visual robustness 上有更多 "可塑空间"。

2. **Lighting 列提升最猛**（π0.5 上 +23.13）。但 lighting **从没被用来构造 paired views**！Training 时 lighting 固定不变。这是 **abstract robustness transfer** 的强证据——policy 学到的不是"对 distractor invariant"，而是"对 task-irrelevant variation invariant"这条 meta-rule，自动 apply 到 lighting 上。

这一点我特别 excited，因为它暗示了 behavior-level supervision 可以学到比 observation-level diversity 更 general 的 robustness。

3. **Clutter 也提升明显**（π0.5 +13.75）。Clutter 是 invariance objective 直接训练的，提升在意料之中。

### 5.2 Ablation（Table 2，π0.5 上）

| Method | Table Texture | Lighting | Target Pose | Clutter | Avg. | Δ Avg. |
|--------|---------------|----------|-------------|---------|------|--------|
| PPO | 63.54 | 28.54 | 56.46 | 36.46 | 46.25 | - |
| PPO + $\mathcal{L}_{\mathrm{inv}}$ | 72.92 | 44.58 | 67.92 | 50.00 | 58.86 | +12.61 |
| PPO + $\mathcal{L}_{\mathrm{sens}}$ | 63.96 | 33.54 | 53.33 | 41.10 | 47.98 | +1.73 |
| Full | 80.21 | 51.67 | 69.38 | 50.21 | 62.87 | +16.62 |

关键现象：

- **Invariance 是主力**（+12.61），sensitivity 单独效果小（+1.73）
- 但 **联合有 synergy**（+16.62 > 12.61 + 1.73 = 14.34）

人话解释：sensitivity 单独说"你应该变"，但不说"往哪变"。Policy 可能瞎变，反而搞坏 Target Pose（53.33 < 56.46）。Invariance 提供 "不变" 的 anchor，sensitivity 在这个 anchor 基础上 push "应变方向"，两个一起才 shape 出有意义的 action space 几何。

### 5.3 RL Efficiency（Figure 2）

- ID scenario (1 distractor): 我们的 80 steps 达到 90%，PPO 要 240 steps，**3× 加速**
- OOD scenario (4 distractors): 同样加速趋势

这个我挺意外。Auxiliary objective 不只是改善 final performance，还 **加速 convergence**。猜测：KL objective 提供了额外 supervision，减少了 PPO 纯 reward signal 下 exploration 的盲目搜索。

### 5.4 Clutter 随数量增加的 degradation（Figure 3）

8 distractors（一半 held-out）时：
- OpenVLA: 72% vs 56%
- π0.5: 33% vs 21%

曲线都下降，但我们的下降更平缓。Invariance 学到的是 "distractor = ignore" 这条 abstract rule，不是记住 specific distractor 长啥样，所以多几个 distractor 也能 handle。

### 5.5 Viewpoint extrapolation（Figure 5）

训练视角 $[0°, 20°]$，测试到 $28°$：
- 24°: 38.34% vs 27.92%
- 28°: 32.92% vs 20.63%

这里用 **viewpoint-based task-preserving view**：同一个 scene state 从不同 camera 渲染。说明 framework 不依赖特定的 view 构造方式，可以 instantiate 不同类型的 task-preserving perturbation。

---

## 6. 几个值得展开的技术细节

### 6.1 Flow-based VLA 怎么做 PPO

OpenVLA 是 autoregressive，有 explicit action log-likelihood，PPO 直接能套。

π0.5 是 flow-matching，action 通过 iterative ODE denoising 生成，**没有 tractable log-likelihood**——标准 PPO 的 likelihood ratio 算不出来。

他们 follow $\pi_{\mathrm{RL}}$ ([Chen et al., 2026](https://arxiv.org/abs/2509.15965)) 的 trick：把 ODE 转成 SDE（加 noise），formulate 成 two-layer MDP：
- Outer MDP: environment interaction
- Inner MDP: denoising step

这样 denoising 过程就有 tractable log-likelihood，能算 PPO 的 likelihood ratio，也就能算 KL divergence。

这个技术细节很关键——不然 sensitivity/invariance 的 KL 都没法定义。

Flow-SDE 用 noise level $\sigma = 0.5$，denoise steps = 4（Table 3）。

### 6.2 Stop-gradient 为什么重要

回到 invariance 的公式：

$$D_{\mathrm{KL}}\left(\pi_\theta(\cdot | o_t, l) \ || \ \mathrm{sg}[\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{prev}}, l)]\right)$$

KL 是非对称的：$D_{\mathrm{KL}}(P || Q)$ 让 $P$ 主动靠近 $Q$，$Q$ 当 target。Stop-gradient 强制 $Q$ 不更新，避免两边互相追逐。

类比 SimSiam / BYOL 那一套 self-supervised learning 的设计——一个 branch stop-gradient 当 anchor，另一个 branch 学过去。Reference: [BYOL](https://arxiv.org/abs/2006.07733), [SimSiam](https://arxiv.org/abs/2011.10566)

### 6.3 Sensitivity 的 clip 防止 degenerate

$$\min\left(c, \ D_{\mathrm{KL}}(\cdot)\right)$$

不 clip 的话，policy 可能学到：对 task-altering view 输出完全随机 action，KL 最大化，但 action 毫无意义。Clip 把 KL 限制在 $c$ 内，超过就停止 push。

这和 PPO 的 clip 是同一个哲学：**trust region**，限制 update 幅度防止崩溃。

### 6.4 LoRA 配置

- OpenVLA: LoRA rank 32, 所有 linear layers
- π0.5: VLM backbone 上 LoRA + **action expert full fine-tune**

π0.5 这个配置偏离了 $\pi_{\mathrm{RL}}$ 推荐（只 train action expert）。作者发现加 LoRA 到 VLM 上 OOD 更好——可能因为 VLM backbone 的 visual representation 也需要 adapt 才能 robust，光改 action expert 不够。

Reference: [LoRA - Hu et al., 2021](https://arxiv.org/abs/2106.09685)

### 6.5 Action generation

| 参数 | OpenVLA | π0.5 |
|------|---------|------|
| Action prediction horizon $H$ | 1 | 8 |
| Action replan horizon $H'$ | 1 | 5 |

π0.5 预测 8 步 action chunk，每 5 步 replan——这是 flow-based VLA 的标准做法。OpenVLA 是 1-step autoregressive。

---

## 7. 跟 related work 的根本区别

### 7.1 vs Domain Randomization

Domain randomization ([Tobin et al., 2017](https://arxiv.org/abs/1703.06907)) 的逻辑：见多识广。但它是 **observation-level** 的——只增加 visual diversity，不指定 action response。Policy 可能对什么变化都过度敏感，因为没有任何信号告诉它 "这个变化不重要"。

PAIR-VLA 是 **behavior-level**——直接告诉 policy "对这个变化 action 不变"。

### 7.2 vs CURL / bisimulation / contrastive

CURL ([Laskin et al., 2020](https://arxiv.org/abs/2004.04136))、bisimulation ([Zhang et al., 2020](https://arxiv.org/abs/2006.10742)) 在 **representation** 上做 invariance。但 representation invariant 不保证 action invariant——policy head 可能从 invariant representation 里读出不同 action。

PAIR-VLA 在 **action distribution** 上做 invariance，end-to-end，bypass 掉 representation 的中间环节。

### 7.3 vs RobustVLA

RobustVLA ([Zhang et al., 2025](https://arxiv.org/abs/2511.01331)) 用 Jacobian / smoothness regularization 对付 **observation noise** 和 **action perturbation**——这些是 local corruption。

PAIR-VLA 对付 **scene-level shifts**（distractor、target pose）——这些可能改变 "task 是否需要 adapt action" 的本质。前者是 robustness to noise，后者是 robustness to semantic visual change。

### 7.4 vs BiPS

BiPS ([Zhang et al., 2025](https://arxiv.org/abs/2512.22120)) 在 VLM reasoning 上做 consistency-separation。PAIR-VLA 借了这个 principle，但 grounding 不同：
- VLM 输出 discrete token，VLA 输出 continuous action
- VLM 的 visual change 影响 reasoning，VLA 的 visual change 直接影响 manipulation

所以 PAIR-VLA 用 **manipulation consequence** 定义 task-preserving/task-altering。

---

## 8. 一些直觉性的类比

### 8.1 类比 contrastive learning

Invariance 像正样本对：拉近（same action）。
Sensitivity 像负样本对：推远（different action）。

但 contrastive 的是 **action distribution**，不是 representation。这是 action-level contrastive learning。

### 8.2 类比 causal intervention

Task-preserving view = $do(\text{distractor} = \text{removed}, \text{texture} = \text{changed})$
Task-altering view = $do(\text{target\_pose} = \text{perturbed})$

KL measure 这些 intervention 的 causal effect on action。
- Invariance 要求 task-preserving intervention 的 causal effect ≈ 0
- Sensitivity 要求 task-altering intervention 的 causal effect > $c$

Reference: [Causal Inference - Pearl](https://en.wikipedia.org/wiki/Causal_inference)

### 8.3 类比 information bottleneck

Invariance 在 compress task-irrelevant info（distractor/texture 不影响 action）。
Sensitivity 在 preserve task-relevant info（target pose 影响动作）。

合起来是一个 action distribution 上的 information bottleneck。Reference: [IB - Tishby et al.](https://arxiv.org/abs/physics/0004057)

### 8.4 类比 equivariance

Sensitivity 在鼓励对 task-relevant transformation 的 **equivariance**：transformation 在 input 上，action distribution 应该相应地 transform。这是 group equivariance 在 RL 里的软实现。Reference: [Equivariant RL - Yang et al.](https://arxiv.org/abs/2207.03046)

---

## 9. 我觉得有意思的 open question

### 9.1 Automatic discovery of task-relevant factors

现在 task-preserving 和 task-altering views 是 **人工设计** 的——人决定 "distractor 是 task-irrelevant，target pose 是 task-relevant"。能不能自动 discover？

可能用 causal discovery、attention analysis、或者 gradient-based saliency。这是下一步很自然的方向。

### 9.2 Curriculum: invariance 先，sensitivity 后

Ablation 显示 sensitivity 单独效果小，联合 invariance 才有大效果。直觉上：先建立 "不变" 的 anchor，再 push "应变" 方向，可能比同时启动更稳。可以试试 curriculum learning。

### 9.3 Language invariance

同样 framework 扩到 language:
- Paraphrase invariance: "pick up the cup" vs "grasp the mug" → same action
- Negation sensitivity: "pick up the cup" vs "don't pick up the cup" → different action

VLA 的 language 端也有 robustness 问题，这个 framework 原则上能直接套。

### 9.4 Lighting transfer 的机制

最 magic 的现象：lighting 没用于构造 views，但 lighting 上提升最大。为什么？

一种解释：invariance objective 让 policy 学到 "ignore task-irrelevant variation" 这条 **meta-rule**，lighting 是 task-irrelevant variation 的一种，自动被 ignore。

更深的解释可能涉及 **visual representation 的 disentanglement**：invariance 促使 representation 把 task-relevant 和 task-irrelevant factor 分开。Lighting 落在 task-irrelevant subspace 里，自然被 ignore。

### 9.5 Multi-task extension

不同 task 的 task-relevant factor 不同。Pick-and-place 关心 target pose，pouring 关心 liquid level。Framework 需要 per-task 的 view constructor。能不能 learn 一个 general view constructor？

### 9.6 Sim-to-real 的 mask 依赖

现在用 simulator 的 ground-truth mask。Real-world 用 SAM 3 ([Carion et al., 2025](https://arxiv.org/abs/2511.16719)) 估计 mask，但 mask 误差会 propagate 到 view construction，可能稀释 robustness gain。这是个 real 的 deployment concern。

---

## 10. 一句话 takeaway

这篇 paper 把 visual robustness 从 "见多识广"（observation diversity）升级为 "知所进退"（behavior-level guidance）。**reward 不告诉你的，auxiliary objective 告诉你**：哪些 visual change 是 noise，哪些是 signal。用 KL 在 action distribution 上做 paired contrastive，加 stop-gradient 稳定训练，加 clip 防 degenerate，最后 zero inference cost deploy。

最让我觉得 promising 的不是 +16.62 的数字，而是 **lighting transfer 那个现象**——它暗示了 behavior-level supervision 能学到 abstract robustness rule，超越具体 augmentation 的 scope。如果这个现象 robust，它可能改写我们做 visual generalization 的方法论：从 "throw more data" 到 "specify behavior constraints"。

---

## References

- [PAIR-VLA paper - Peng et al., 2026](https://arxiv.org/abs/2506.05018)（注：实际 arxiv ID 以 paper 发布为准）
- [PPO - Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [GAE - Schulman et al., 2015](https://arxiv.org/abs/1506.02438)
- [OpenVLA - Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- [π0 - Black et al., 2024](https://arxiv.org/abs/2410.24164)
- [π0.5 - Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054)
- [π_RL - Chen et al., 2026](https://arxiv.org/abs/2509.15965)
- [ManiSkill3 - Tao et al., 2024](https://arxiv.org/abs/2410.00425)
- [LoRA - Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- [CURL - Laskin et al., 2020](https://arxiv.org/abs/2004.04136)
- [Bisimulation - Zhang et al., 2020](https://arxiv.org/abs/2006.10742)
- [Domain Randomization - Tobin et al., 2017](https://arxiv.org/abs/1703.06907)
- [DRQ - Kostrikov et al., 2020](https://arxiv.org/abs/2004.13649)
- [RobustVLA - Zhang et al., 2025](https://arxiv.org/abs/2511.01331)
- [BiPS - Zhang et al., 2025](https://arxiv.org/abs/2512.22120)
- [SAM 3 - Carion et al., 2025](https://arxiv.org/abs/2511.16719)
- [BYOL - Grill et al., 2020](https://arxiv.org/abs/2006.07733)
- [SimSiam - Chen & He, 2020](https://arxiv.org/abs/2011.10566)
- [Information Bottleneck - Tishby et al.](https://arxiv.org/abs/physics/0004057)
- [Equivariant RL - Yang et al.](https://arxiv.org/abs/2207.03046)
- [SimpleVLA-RL - Li et al., 2025](https://arxiv.org/abs/2504.05118)
- [RLinf - Yu et al., 2025](https://arxiv.org/abs/2509.15965)
- [RL4VLA - Liu et al., 2026](https://arxiv.org/abs/2502.04078)
- [FLaRe - Hu et al., 2025](https://arxiv.org/abs/2409.16578)
- [Open X-Embodiment - O'Neill et al., 2024](https://arxiv.org/abs/2310.08864)
- [DROID - Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945)

希望这版 "人话" 版本让你 build 起更清晰的 intuition。如果哪个点想再深挖（比如 flow-SDE 的 two-layer MDP 怎么 formulate、或者 lighting transfer 的机制假说），随时说。

---

# What to Ignore, What to React: 深度解析 PAIR-VLA

Andrej，这篇 paper 来自 Microsoft Research Asia、HKUST 和 Zhejiang University，处理的是 VLA model RL fine-tuning 中一个被 underexplored 的问题：**如何让 policy 知道哪些 visual changes 应该 ignore，哪些应该 react**。这其实是一个 behavior-level 的 credit assignment 问题，而他们用 paired views + KL objectives 给出了一个相当 elegant 的解法。

---

## 1. 核心问题的 Intuition

传统做法（domain randomization、image augmentation）的逻辑是：让 policy 在 training 时见过更多 visual variation，自然就 robust。但这只提供了 **observation diversity**，没有提供 **behavior-level supervision**。

考虑两个场景：
- 场景 A：桌上多了一个 distractor 物体 → task 没变，action 应该不变
- 场景 B：target object 的 pose 变了 → task 变了，action 必须变

Standard PPO 的 reward 只告诉你 task 成功与否，**不告诉你 visual change 和 action change 之间的因果关系**。policy 可能学到对 A 也改变 action（over-sensitive），或者对 B 也不改变 action（under-sensitive）。

PAIR-VLA 的核心 insight：把 visual variation 从 "observation diversity" 提升为 "behavior-level guidance"，显式指定 policy 应该对哪些变化 invariant、对哪些变化 sensitive。

这一点让我想到 **causal inference** 里的 intervention：task-preserving view 和 task-altering view 就像两个 do-interventions，KL divergence 在 measure 它们的 causal effect on action distribution。

---

## 2. 方法细节与公式解析

### 2.1 整体框架

在 PPO 之上加两个 auxiliary objectives，构成 augmented objective：

$$\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{PPO}}(\theta) + \alpha \mathcal{L}_{\mathrm{inv}}(\theta) + \beta \mathcal{L}_{\mathrm{sens}}(\theta)$$

变量含义：
- $\theta$: policy 参数
- $\mathcal{L}_{\mathrm{PPO}}$: standard PPO objective
- $\mathcal{L}_{\mathrm{inv}}$: invariance objective (task-preserving)
- $\mathcal{L}_{\mathrm{sens}}$: sensitivity objective (task-altering)
- $\alpha, \beta > 0$: 两个 auxiliary objectives 的权重系数

关键：两个 objective 只在 training 时用，**deployment 时 policy 结构不变，零额外 inference cost**。

### 2.2 PPO Baseline

先回顾 PPO 的 clipped surrogate objective (公式 1)：

$$\mathcal{L}_{\mathrm{PPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{T}\sum_{t=1}^{T}\min\left(w_t(\theta)\hat{A}_t, \ \mathrm{clip}\left(w_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$

变量解析：
- $\tau$: trajectory，从 old policy $\pi_{\theta_{\mathrm{old}}}$ 采样
- $T$: trajectory 长度
- $w_t(\theta) = \frac{\pi_\theta(a_t | o_t, l)}{\pi_{\theta_{\mathrm{old}}}(a_t | o_t, l)}$: likelihood ratio，新策略与旧策略在 action $a_t$ 上的概率比
- $\hat{A}_t$: advantage estimate，通过 GAE 计算
- $\epsilon$: clipping parameter（本文用 0.2），限制 policy update 幅度

对于 flow-matching VLA（如 $\pi_{0.5}$），action 通过 iterative ODE denoising 生成，没有 tractable log-likelihood。他们 follow $\pi_{\mathrm{RL}}$ 的方法：把 ODE 转成 SDE，formulate two-layer MDP，从而获得 tractable log-likelihoods 来做 PPO。这是一个技术细节，但很关键，因为没有这个就没办法算 KL divergence。

### 2.3 Invariance Objective (公式 2)

$$\mathcal{L}_{\mathrm{inv}}(\theta) = \mathbb{E}_{(o_t, l)}\left[D_{\mathrm{KL}}\left(\pi_\theta(\cdot | o_t, l) \ || \ \mathrm{sg}[\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{prev}}, l)]\right)\right]$$

变量解析：
- $o_t$: 时间步 $t$ 的原始 observation
- $\tilde{o}_t^{\mathrm{prev}}$: **task-preserving view**，通过移除 distractors + 替换 background 构造
- $l$: language instruction（episode 内固定）
- $\pi_\theta(\cdot | o_t, l)$: 原始观察下的 action distribution
- $\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{prev}}, l)$: task-preserving view 下的 action distribution
- $\mathrm{sg}[\cdot]$: **stop-gradient operator**，把 task-preserving view 的 distribution 当作 fixed target
- $D_{\mathrm{KL}}(P || Q) = \sum_a P(a) \log \frac{P(a)}{Q(a)}$: KL divergence

**Intuition**: 这是 KL minimization。我们希望原始观察下的 action distribution 去逼近 task-preserving view 下的 action distribution。注意 KL 的方向：$P$ 是 original（有 gradient），$Q$ 是 perturbed（stop-gradient）。这样 gradient 只流回 original view 的 distribution，perturbed view 作为 anchor。

为什么用 stop-gradient？如果两边都有 gradient，两个 distribution 会互相追逐，可能导致 training unstable。Stop-gradient 让 perturbed view 成为 fixed target，类似 knowledge distillation 中的 teacher。

### 2.4 Sensitivity Objective (公式 3)

$$\mathcal{L}_{\mathrm{sens}}(\theta) = -\mathbb{E}_{(o_t, l)}\left[\min\left(c, \ D_{\mathrm{KL}}\left(\pi_\theta(\cdot | o_t, l) || \mathrm{sg}[\pi_\theta(\cdot | \tilde{o}_t^{\mathrm{alt}}, l)]\right)\right)\right]$$

变量解析：
- $\tilde{o}_t^{\mathrm{alt}}$: **task-altering view**，通过 perturb target object 的 pose 构造
- $c$: **clipping threshold**，防止 KL divergence 无限增长
- 负号: 我们要 **maximize** 这个 KL divergence（鼓励分离）

**Intuition**: 负号 + KL = 鼓励两个 distribution 分开。当 target object pose 变化时，required manipulation 变了，policy 应该输出不同的 action distribution。

为什么需要 $\min(c, \cdot)$ clip？如果不 clip，policy 可能为 maximize KL 而 produce degenerate behavior——比如对 task-altering view 输出完全随机 action。Clip 把 divergence 限制在 $c$ 以内，超过 $c$ 后 gradient 为 0，stabilize PPO training。

这是一个很巧妙的设计：sensitivity 是一个 "推" 的力，但有上限；invariance 是一个 "拉" 的力。两个力配合，塑造 action distribution 的几何结构。

### 2.5 Task-preserving View Construction (公式 5)

$$\tilde{o}_t^{\mathrm{prev}} = m_t \odot o_t + (1 - m_t) \odot o_k^{\mathrm{bg}}$$

变量解析：
- $m_t \in \{0, 1\}^{H \times W}$: binary segmentation mask
  - $m_t = 1$: pixels 属于 robot、target object、receptacle（task-relevant foreground）
  - $m_t = 0$: pixels 属于 distractor、background（task-irrelevant）
- $o_k^{\mathrm{bg}}$: 预渲染的 background snapshot，从 $K$ 个不同 scene configuration 中均匀采样
- $\odot$: element-wise multiplication

**实现细节**: 训练前 pre-render $K$ 个 background snapshots（把所有 objects 设为 invisible 渲染）。训练时对每个 $o_t$ 用 simulator 的 object identifier 拿 mask，然后 composited with 随机 background。Real-world 可以用 SAM 3 ([SAM 3 paper](https://arxiv.org/abs/2511.16719)) 近似 mask。

### 2.6 Task-altering View Construction

对 target object 采样：
- Translation perturbation: Gaussian distribution
- Rotation perturbation: categorical distribution

然后 re-render scene 得到 $\tilde{o}_t^{\mathrm{alt}}$。这会产生 target pose 变化，可能需要不同的 grasp、motion、placement。

---

## 3. 实验数据深度分析

### 3.1 主实验 (Table 1)

| Model | Method | Table Texture | Lighting | Target Pose | Clutter | Avg. | Δ Avg. |
|-------|--------|---------------|----------|-------------|---------|------|--------|
| OpenVLA | PPO | 86.98 | 72.14 | 83.59 | 68.88 | 77.90 | - |
| OpenVLA | Ours | 94.53 | 80.47 | 90.63 | 82.36 | 87.00 | **+9.10** |
| π0.5 | PPO | 63.54 | 28.54 | 56.46 | 36.46 | 46.25 | - |
| π0.5 | Ours | 80.21 | 51.67 | 69.38 | 50.21 | 62.87 | **+16.62** |

**关键观察**：
1. **π0.5 的提升更大**（+16.62 vs +9.10）。这可能因为 π0.5 是 flow-based model，action space 更复杂，PPO baseline 更弱，auxiliary objectives 的相对收益更大。
2. **Lighting 提升最显著**（π0.5: +23.13）。但 lighting **没有**被用于构造 paired views！这说明 invariance objective 学到的是 **abstract 的 robustness**，不是针对特定 augmentation 的 robustness。这是 transfer 的证据。
3. **Clutter 也提升明显**（π0.5: +13.75），说明对 distractor 的 robustness 有强 transfer。

### 3.2 Ablation Study (Table 2) — 在 π0.5 上

| Method | Table Texture | Lighting | Target Pose | Clutter | Avg. | Δ Avg. |
|--------|---------------|----------|-------------|---------|------|--------|
| PPO | 63.54 | 28.54 | 56.46 | 36.46 | 46.25 | - |
| PPO + $\mathcal{L}_{\mathrm{inv}}$ | 72.92 | 44.58 | 67.92 | 50.00 | 58.86 | +12.61 |
| PPO + $\mathcal{L}_{\mathrm{sens}}$ | 63.96 | 33.54 | 53.33 | 41.10 | 47.98 | +1.73 |
| Full | 80.21 | 51.67 | 69.38 | 50.21 | 62.87 | +16.62 |

**关键观察**：
1. **Invariance 是主力**（+12.61），sensitivity 单独效果小（+1.73）
2. 但 **联合起来有 synergy**（+16.62 > 12.61 + 1.73 = 14.34）。这说明 sensitivity objective 不只是 additive 的贡献，它 shape 了 action space 的 geometry，让 invariance 更 effective。
3. Sensitivity 单独在 Target Pose 上反而 **下降**（56.46 → 53.33），这可能因为 sensitivity 鼓励 distribution 分离，但如果没有 invariance 约束，分离方向可能不对。

**Intuition**: Invariance 提供 "应该不变" 的 anchor，sensitivity 提供 "应该变" 的 push。单独 sensitivity 没有 anchor，policy 可能往错误方向变。联合起来，两个力共同定义了 action space 中 "不变区" 和 "应变区" 的边界。

### 3.3 RL Fine-tuning Efficiency

ID scenario (1 distractor)：
- Our method: 80 steps 达到 90%
- PPO: 240 steps 达到 90%
- **3× 加速**

OOD scenario (4 distractors)：同样的加速趋势。

这说明 auxiliary objectives 不只是 improve final performance，还 **加速 convergence**。可能因为 KL objectives 提供了额外的 supervision signal，reduce 了 exploration 的 blind search。

### 3.4 Clutter Robustness (Figure 3)

8 distractors (一半 held-out):
- OpenVLA: 72% (Ours) vs 56% (PPO)
- π0.5: 33% (Ours) vs 21% (PPO)

随着 distractor 数量增加，两条曲线都下降，但我们的方法 **degradation 更平缓**。这说明 invariance objective 让 policy 学会了 "distractor = ignore" 的 abstract rule，而不是记住 specific distractor。

### 3.5 Invariance Coefficient Sweep (Figure 4)

$\alpha \in \{0, 1, 2, 4\}$，$\beta = 0$：
- $\alpha = 0$ → PPO baseline
- $\alpha = 1$ → 最佳
- $\alpha = 4$ → 仍然稳定

这说明 method 对 $\alpha$ 不敏感，robust to hyperparameter choice。

### 3.6 Viewpoint Generalization (Figure 5)

训练：$[0°, 20°]$，每 $4°$ 一个点
测试：$0°$ 到 $28°$，OOD range 是 $\{24°, 28°\}$

| Angle | PPO | Ours |
|-------|-----|------|
| 24° | 27.92% | 38.34% |
| 28° | 20.63% | 32.92% |

这里用 **viewpoint-based task-preserving view**（同一个 scene state 从不同 camera pose re-render）。ID range 内性能相当（70.36% vs 70.97%），OOD range 显著提升。这说明 method 可以 instantiate with 不同类型的 task-preserving view，framework 是 general 的。

---

## 4. 与 Related Work 的对比

### 4.1 vs Domain Randomization

Domain randomization ([Tobin et al., 2017](https://arxiv.org/abs/1703.06907); [Tremblay et al., 2018](https://arxiv.org/abs/1810.10093)) 只增加 observation diversity，不指定 action response。PAIR-VLA 显式 shape action distribution，是 **behavior-level** 而非 **observation-level** 的 robustness。

### 4.2 vs Representation-level Invariance

CURL ([Laskin et al., 2020](https://arxiv.org/abs/2004.04136))、CPC、bisimulation ([Zhang et al., 2020](https://arxiv.org/abs/2006.10742)) 等在 **representation space** 做 invariance。但 representation invariance **不保证** action invariance——policy head 可以从 invariant representation 学到不同 action。PAIR-VLA 直接在 action distribution 上做 invariance，end-to-end。

### 4.3 vs RobustVLA

RobustVLA ([Zhang et al., 2025](https://arxiv.org/abs/2511.01331)) 用 Jacobian 和 smoothness regularization 对付 observation noise 和 action perturbation。它针对 **local corruption**，PAIR-VLA 针对 **scene-level visual shifts**（distractor、target pose），这些 shifts 可能改变是否应该 preserve action。

### 4.4 vs BiPS

BiPS ([Zhang et al., 2025](https://arxiv.org/abs/2512.22120)) 在 VLM reasoning 中引入 consistency-separation objectives。PAIR-VLA 借鉴了这个 principle，但 **grounding 不同**：VLA 的 output 是 continuous action，VLM 的 output 是 discrete token；VLA 的 visual change 直接影响 manipulation，VLM 的 visual change 影响 reasoning。所以 PAIR-VLA 用 manipulation consequence 来定义 task-preserving vs task-altering。

---

## 5. 训练细节

### 5.1 Hyperparameters (Table 3)

| Parameter | OpenVLA | π0.5 |
|-----------|---------|------|
| RL train steps | 240 | 280 |
| Global batch size | 640 | 5120 |
| Update epochs | 1 | 5 |
| Actor LR | 1e-4 | 7.91e-6 |
| Critic LR | 3e-3 | 1.55e-4 |
| PPO clip ε | 0.2 | 0.2 |
| **α (invariance)** | 1 | 4 |
| **β (sensitivity)** | 0.2 | 4 |
| **c (sensitivity clip)** | 0.8 | 0.08 |

注意 $\alpha, \beta, c$ 在两个 model 上差异大，因为 autoregressive 和 flow-based 的 KL divergence 在不同 numerical scale 上。

### 5.2 LoRA 配置

- OpenVLA: LoRA rank 32, 所有 linear layers
- π0.5: LoRA on VLM backbone + **full fine-tune action expert**。这偏离了 $\pi_{\mathrm{RL}}$ 的推荐（只 train action expert），但作者发现这样 OOD 性能更好。

### 5.3 Reward 设计

- Outcome reward: +1 if success
- Is_grasp reward: +0.1 if grasp
- Consecutive grasp reward: +0.1 for sustained grasp

Dense reward 有助于 exploration，特别是 consecutive grasp reward 鼓励 stable grasp。

### 5.4 Action Generation

| Parameter | OpenVLA | π0.5 |
|-----------|---------|------|
| Action prediction horizon $H$ | 1 | 8 |
| Action replan horizon $H'$ | 1 | 5 |
| Denoise steps | - | 4 |
| Noise level $\sigma$ (Flow-SDE) | - | 0.5 |

π0.5 用 flow-matching，需要 denoise steps 和 noise level 参数。Flow-SDE formulation 让 PPO 可行。

### 5.5 Compute

8× H100 GPU
- OpenVLA: ~1 day
- π0.5: ~3 days

---

## 6. 评估 Benchmark 细节

### 6.1 Task Setup

ManiSkill3 ([Tao et al., 2024](https://arxiv.org/abs/2410.00425))，pick-and-place task：
- Observation: 480×640 third-person image + joint-poses + instruction
- Action: delta end-effector pose
- Object 位置: 6×6 grid 离散采样，避免 overfit 到 specific spatial configuration

### 6.2 OOD Scenarios

1. **Unseen Table Texture**: 16 train textures, 5 eval textures
2. **Unseen Lighting**: 训练时固定 lighting，eval 用 20 个 held-out lighting configs
3. **Unseen Pose**: 训练在 small square (half-edge 7.5cm)，eval 在 outer border (half-edge 10.5cm)
4. **Unseen Clutter**: 16 train objects, 9 eval objects；distractor 数量 2-8，一半 held-out

### 6.3 Metrics

- Success rate: 128 episodes per run, 3 seeds, report mean
- 这是严格的 evaluation protocol

---

## 7. 重要的 Design Choices 与 Intuition

### 7.1 为什么 KL 而不是 MSE on actions?

Action 是 distribution（Gaussian for continuous control），不是 point estimate。KL 衡量 distribution-level 的差异，更 robust。MSE 只 compare mean，ignore variance。

### 7.2 为什么 stop-gradient 在 perturbed view?

如果两边都有 gradient，两个 distribution 会互相靠近（invariance）或互相远离（sensitivity），可能导致 oscillation。Stop-gradient 让 perturbed view 作为 fixed anchor，original view 去适应它。类似 BYOL ([ Grill et al., 2020](https://arxiv.org/abs/2006.07733)) 的 stop-gradient 设计。

### 7.3 为什么 sensitivity 用 min(c, ·) clip?

没有 clip，policy 可能为 maximize KL 而 collapse 到 degenerate solution（比如对 task-altering view 输出 deterministic 但无意义的 action）。Clip 提供 soft constraint：达到 $c$ 后停止 push，让 PPO reward 接管。

### 7.4 为什么 lighting 提升这么大，但 lighting 没用于构造 views?

这是 **abstract robustness transfer** 的证据。Invariance objective 让 policy 学到 "task-irrelevant variation → ignore" 的 general rule，不是 specific 到 distractor/texture。Lighting 是另一种 task-irrelevant variation，policy 自动 apply 了学到的 invariance。

这让我想到 **meta-learning** 的视角：invariance objective 是在学 "如何 ignore"，而不是 "ignore 什么"。

### 7.5 为什么 sensitivity 单独效果小，联合效果好?

Sensitivity 单独只说 "应该变"，但不说 "应该往哪变"。没有 invariance 的 anchor，policy 可能往 random 方向变。Invariance 提供 "不变区" 的定义，sensitivity 在这个定义基础上 push "应变区"。两个一起 shape 了 action space 的 **有意义几何结构**。

---

## 8. 我的联想与延伸思考

### 8.1 Information Bottleneck 视角

Invariance objective 是在 **compress task-irrelevant information**（关于 distractor/texture 的信息不应该影响 action）。Sensitivity objective 是在 **preserve task-relevant information**（关于 target pose 的信息应该影响 action）。这本质是一个 **information bottleneck** on action distribution，类似 IB 但作用在 action 而非 representation。

### 8.2 Equivariance 的联系

Sensitivity 其实是在鼓励对 task-relevant transformation 的 **equivariance**：如果 target pose 变了，action 应该相应地变。这是 group equivariance 在 RL 中的软实现。

### 8.3 Causal Inference 视角

Task-preserving view = do(distractor = removed, background = changed)
Task-altering view = do(target_pose = perturbed)

KL divergence 在 measure 这些 intervention 的 causal effect on action。Invariance 要求 task-preserving intervention 的 causal effect ≈ 0，sensitivity 要求 task-altering intervention 的 causal effect > c。

这和 **counterfactual reasoning** 有关：policy 在 learn "如果 distractor 不在，action 应该一样"。

### 8.4 扩展到 Language Variations

同样的 framework 可以扩展到 **language invariance**：
- Paraphrase invariance: "pick up the cup" vs "grasp the mug" → same action
- Negation sensitivity: "pick up the cup" vs "do not pick up the cup" → different action

这会让 VLA 对 language variations 也 robust。

### 8.5 自动发现 Task-relevant Factors

当前 task-preserving 和 task-altering views 是 **人工设计** 的。如果能自动 discover 哪些 visual factors 是 task-relevant 的，就更 general。可能用 causal discovery 或 attention analysis。

### 8.6 Curriculum Learning

先学 invariance（建立 anchor），再加 sensitivity（在 anchor 基础上 push）。可能比同时学更 stable。当前实验是同时加，没探索 curriculum。

### 8.7 与 Contrastive Learning 的关系

Invariance 像正样本对的 contrastive learning（拉近），sensitivity 像负样本对的 contrastive learning（推远）。但这里 contrastive 的是 **action distribution**，不是 representation。这是 **action-level contrastive learning**。

### 8.8 Multi-task Extension

不同 task 的 task-relevant factors 不同。Pick-and-place 的 task-relevant 是 target pose，pouring 的 task-relevant 可能是 liquid level。Framework 需要 per-task 的 view construction，能不能 learn 一个 general view constructor?

### 8.9 Sim-to-Real

当前依赖 simulator 的 ground-truth mask。Real-world 用 SAM 3 估计 mask，但 mask 误差会 propagate 到 view construction，影响 robustness gain。这是个 open question。

### 8.10 与 DPO/GRPO 的兼容性

本文用 PPO。但 GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) 在 LLM RL 中流行，SimpleVLA-RL ([Li et al., 2025](https://arxiv.org/abs/2504.05118)) 把 GRPO 用到 VLA。PAIR-VLA 的 auxiliary objectives 原则上可以加到 GRPO，但 GRPO 没有 explicit action log-likelihood（用 group baseline），KL 计算需要 adapt。

---

## 9. 局限性与未来方向

### 9.1 当前局限
- 只在 simulation 评估，没验证 sim-to-real
- 依赖 ground-truth segmentation mask
- Single task (pick-and-place)，multi-task 有效性未知
- Task-altering view 只用 target pose perturbation，其他 task-relevant factors（如 object shape change）未探索

### 9.2 未来方向
- Real-world deployment with SAM-based mask
- Multi-task extension with per-task view constructors
- Automatic discovery of task-relevant factors
- Language variation invariance
- Curriculum: invariance first, then sensitivity
- Compatibility with GRPO and other RL algorithms

---

## 10. 总结

这篇 paper 的核心贡献是把 visual robustness 从 **observation-level** 提升到 **behavior-level**。关键 insight 是：visual generalization 不只是见过更多 variation，而是知道哪些 variation 应该 ignore、哪些应该 react。

技术实现很 elegant：两个 KL objectives（一个 minimize，一个 maximize with clip）配合 stop-gradient，加到 PPO 上，零 inference cost。

实验结果 strong：π0.5 上 +16.62%，OpenVLA 上 +9.10%，且 transfer 到 unseen lighting（未用于构造 views）。

最让我 excited 的是 **abstract robustness transfer** 的现象：invariance 学到的是 "如何 ignore" 的 meta-skill，不是 specific 到某个 augmentation。这暗示了 VLA model 可以从 behavior-level supervision 中学到更 general 的 robustness，超越传统 domain randomization 的 scope。

---

## References

- [PPO - Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [GAE - Schulman et al., 2015](https://arxiv.org/abs/1506.02438)
- [OpenVLA - Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- [π0 - Black et al., 2024](https://arxiv.org/abs/2410.24164)
- [π0.5 - Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054)
- [ManiSkill3 - Tao et al., 2024](https://arxiv.org/abs/2410.00425)
- [LoRA - Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- [CURL - Laskin et al., 2020](https://arxiv.org/abs/2004.04136)
- [DRQ - Kostrikov et al., 2020](https://arxiv.org/abs/2004.13649)
- [Domain Randomization - Tobin et al., 2017](https://arxiv.org/abs/1703.06907)
- [RobustVLA - Zhang et al., 2025](https://arxiv.org/abs/2511.01331)
- [BiPS - Zhang et al., 2025](https://arxiv.org/abs/2512.22120)
- [RLinf - Yu et al., 2025](https://arxiv.org/abs/2509.15965)
- [SAM 3 - Carion et al., 2025](https://arxiv.org/abs/2511.16719)
- [SimpleVLA-RL - Li et al., 2025](https://arxiv.org/abs/2504.05118)
- [GRPO - Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- [BYOL - Grill et al., 2020](https://arxiv.org/abs/2006.07733)
- [Invariance RL - Zhang et al., 2020](https://arxiv.org/abs/2006.10742)
- [Open X-Embodiment - O'Neill et al., 2024](https://arxiv.org/abs/2310.08864)
- [DROID - Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945)
- [RT-2 - Brohan et al., 2023](https://arxiv.org/abs/2307.15818)
- [Octo - Octo Team, 2024](https://octo-models.github.io/)

希望这个解析帮你 build 起对这篇 paper 的 intuition。如果你想深挖某个方面（比如 flow-based VLA 的 SDE formulation，或者 abstract robustness transfer 的机制），告诉我。
