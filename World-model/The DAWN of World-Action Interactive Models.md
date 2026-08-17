---
source_pdf: The DAWN of World-Action Interactive Models.pdf
paper_sha256: bee71fb071532bd2f63b13d9cf4c568b11de89444285e12953a06c9c2f972d93
processed_at: '2026-08-12T14:02:12-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DAWN 用人话版

好，Andrej，我换个调调，咱们当喝咖啡聊。

---

## 这帮人到底想干啥

一句话：**开车的时候，"前面会发生什么"和"我要做什么"这俩事得一起想，不能分开算。**

你看现在的 autonomous driving 方法，基本就三种搞法：

**第一种**：两个 head 并行走，一个预测未来世界，一个输出 trajectory，共享 backbone 但各算各的。问题是它俩 generation 的时候根本不说话——world head 不知道你 action head 在想啥，action head 也不管 world head 预测了什么。这就好比两个人各想各的，完全不沟通。

**第二种**：先 predict 未来世界（pixel、occupancy、latent 都行），predict 完了 freeze 住，然后在上面 plan action。问题是——你 freeze 的那个 future 是 "passive future"，假设你啥也不干世界会咋样。但你要 plan 的 action 会改变这个 future 啊！比如你 yield 了，旁边那车就 merge 进来；你 push 的话，那车就让你。这两个 future 完全不一样。

**第三种**：最近有个 Fast-WAM 跳出来说，inference 时候根本不用 rollout world，直接当 policy 用就行，world modeling 就是个 training signal。在简单场景下确实够用。

DAWN 的观点是：**这三种都不对，至少在复杂 interactive 场景下都不对。** 你需要让 world 和 action 在 inference 时互相 refine——action hypothesis 喂给 world predictor 更新未来假设，更新后的未来假设再喂给 action denoiser 改 action，来回几轮，收敛到一个 self-consistent 的 pair。

---

## 为什么这事 important

你看 driving 这个 domain 特别有意思——**决策 relevant 的未来是 action-contingent 的**。

举个直觉例子：你在十字路口，对面来一辆车要 left turn。问题是它让你先过还是它先抢？

这个答案**取决于你干嘛**：
- 你 commit 冲过去 → 它大概率让你
- 你犹豫踩刹车 → 它大概率抢

所以 "future" 不是固定的，它 conditional 在你 action 上。你 plan action 的时候得想"我这么干世界会咋样"，但你算"世界会咋样"的时候又得知道"我打算干啥"——**这就是个 fixed-point 问题**。

经典解法是 MPC：有个 dynamics model，optimize over action sequence，每个 action 都 simulate 一下未来。但 MPC 在 driving 一直没 dominate，因为 hand-crafted dynamics 抓不住 multi-agent interaction。DAWN 用 learned latent predictor 代替 hand-crafted dynamics，用 diffusion denoising 代替 explicit optimization——**本质是 learned MPC in latent space**。

---

## DAWN 架构，拆开讲

整体 pipeline 是这样的：

### Step 0: 看一眼当前画面

V-JEPA 2 Large 当 vision backbone（就是 LeCun 团队那个 self-supervised video model），输入 4 帧 video，输出一堆 dense visual tokens。但 dense tokens 太多没法 rollout，所以有个 **Auto-Encoder Resampler** 把它压成 16 个 compact latent tokens。

这步直觉上就是："我看一眼路况，把场景 essence 抽出来"。

### Step 1: 先 propose 一个 action

直接从 resampler latent 生成一个初始 trajectory hypothesis。这步是 Action Denoiser 在 "proposal" 模式下工作，用专门的 $q_{prop}$ query embedding 告诉它"你现在是 proposer"。

直觉：**"我大概知道要走哪，先给个粗略 plan"**。

### Step 2: 根据 action 想象未来

World Predictor 拿当前 latent + condition + 当前 action hypothesis，rollout 一段 future latent。这就是 "如果我这么开，世界会变成啥样"。

关键：这个 rollout 是 **action-conditioned** 的。换个 action 就换个 future。这是跟 passive world model 的本质区别。

### Step 3: 根据 future 修正 action

Action Denoiser 切换到 "refinement" 模式，用 $q_{ref}$ query embedding，condition 在刚 rollout 出来的 future latent 上，重新 denoise action tokens。

直觉：**"看了下未来，发现这么开会撞/会被挤，调整一下"**。

### Step 4: 回到 Step 2，循环

Step 2 和 Step 3 来回跑，paper 用 4 round。每轮：
- World Predictor 根据 current action 更新 future hypothesis
- Action Denoiser 根据 updated future 更新 action hypothesis

4 round 之后 saturate，性能不再涨甚至略降，所以就用 4 round。

### Step 5: 输出 trajectory

最后 Action Head 把 denoised action state decode 成实际 trajectory 点。

---

## 几个关键技术细节

### Resampler 为啥重要

V-JEPA 2 输出上千个 dense tokens，rollout 这种序列计算太贵。压成 16 个 token 后，World Predictor 只需 rollout 16 个 token × 2-3 秒 future，inference latency 从 1067ms (full rollout) 降到 690ms (2s rollout)，性能只差 0.6 PDMS。

这是 paper 最 actionable 的发现之一：**world model 不需要当 perfect future simulator，它只需要提供"足够支撑 action reasoning"的 compact future hypothesis**。

参考 Q-Former (BLIP-2): https://arxiv.org/abs/2301.12597
参考 Perceiver: https://arxiv.org/abs/2103.03206

### Action Denoiser 的角色切换 trick

一个 network 两种用法，weight 共享，只换 query embedding：
- $q_{prop}$: "你是 proposer，从 latent 直接出 action"
- $q_{ref}^{(r)}$: "你是 refiner，根据 rollout 的 future 改 action"

这让我想起 in-context learning——同一个 model 通过不同 prompt 切换 mode。也想起 DETR 的 object queries。

### AdaLN-Zero 的修改

标准 DiT 的 adaLN-Zero 只 modulate self-attn 和 MLP。DAWN 改成 **也 modulate cross-attn to world latent**，每个 block 预测 shift/scale/gate 三组参数作用于三个 residual 分支。这让 action denoising 更彻底地 condition 在 world context 上。

参考 DiT: https://www.wpeebles.com/DiT

---

## 实验数字背后的 story

### NAVSIM v1：PDMS 89.1，perception-free SOTA

主要看 TTC (Time-to-Collision) 96.0，全场最高。这个 metric 直接反映"对 future collision 的 awareness"——你 plan 的时候心里有 future，trajectory 就会主动避开 TTC 小的 path。这是 WAIM 哲学的直接证据。

### nuScenes：L2 avg 0.33m，碾压 WorldRFT 0.47m

更精彩的是 long-horizon：3s L2 0.52m vs WorldRFT 0.76m，相对降 31%。短 horizon 差距小，长 horizon 差距大——这符合直觉，long-horizon trajectory 在 interactive scene 下更需要 action-world co-evolution，predict-then-plan 的误差会累积。

### Ablation 的关键 finding

Table 5 双向消融是 paper 最有力的证据：
- 去掉 World→Action（action 不再 condition on future）：PDMS 暴跌 6.3 分
- 去掉 Action→World（world 不再 condition on action）：PDMS 降 3.0 分

两个方向都重要，World→Action 更重要。**这直接证伪了"passive future prediction 就够"的假设**。

### Interactive rounds 从 1 到 4 单调涨

- 1 round: 85.2
- 2 round: 86.4
- 3 round: 86.9
- 4 round: 87.9
- 5 round: 87.2（开始降）

每多一轮 interaction 都有 gain，4 round saturate。这是 **test-time compute scaling** 的清晰信号——多花 inference compute 换更好结果。跟 o1 / R1 / AlphaZero MCTS 同源，但载体是 latent space 不是 token space。

---

## 这篇 paper 在大图景里的位置

### 跟 test-time compute 的关系

你 Andrej 最近一直讲 test-time compute 是新维度。DAWN 给了一个 non-LLM 的 case study：visual policy 在 inference 时通过 latent rollout + diffusion refinement 实现 "thinking"。

LLM chain-of-thought 是 **verbalized thought**（生成 tokens 当中间推理）。
DAWN 是 **latent thought**（在 compact latent space 做 fixed-point iteration）。

两种都是 test-time compute，但 DAWN 更紧凑、更几何化、更接近 LeCun 说的 "non-symbolic thought"。

### 跟 JEPA 哲学的关系

LeCun 的 JEPA 主张：不要 reconstruct pixels，要在 abstract latent space 做 predictive learning。V-JEPA 2 就是这个哲学的 video 版本。DAWN 用 V-JEPA 2 当 backbone，又加上 iterative interaction——**这是把 JEPA 推到 action-contingent 方向**。

LeCun 一直说 world model 应该是 JEPA-style，但又没给出 action 怎么进来。DAWN 给了一个 answer：**action 通过 conditioning predictor，再通过 denoiser 反过来 refine action**，形成 closed loop。

参考 V-JEPA 2: https://arxiv.org/abs/2506.09985

### 跟 MPC 的关系

经典 MPC: dynamics $f$ 是 known 的，optimize $a^* = \arg\max_a \sum_t R(s_t, a_t)$ s.t. $s_{t+1} = f(s_t, a_t)$。

DAWN: dynamics 是 learned latent predictor $P_\theta$，optimization 是 diffusion denoising with iterative conditioning。

差别：
1. MPC 在 explicit state space rollout，DAWN 在 latent space
2. MPC 用 gradient/cem 优化，DAWN 用 diffusion sampling
3. MPC 用 full horizon，DAWN 用 short horizon + long action generation
4. MPC 的 dynamics 是 hand-crafted，DAWN 是 learned from data

DAWN 的 advantage: 抓住 multi-agent interaction 这种 hand-crafted 难表达的东西。
DAWN 的 disadvantage: 没了 formal safety guarantee。

### 跟 GAN 的类比

GAN 是 generator 和 discriminator 互为 condition，迭代收敛到 Nash equilibrium。DAWN 是 World Predictor 和 Action Denoiser 互为 condition，迭代收敛到 cooperative fixed point。结构同构，但 DAWN 是 cooperative（共享目标），不是 adversarial。

### 跟 EM 算法的类比

EM: E-step 估计 latent, M-step 优化参数，交替进行。DAWN: World Predictor 估计 future latent, Action Denoiser 优化 action，交替进行。完全同构，都是 coordinate descent on latent-variable model。

---

## 我觉得 paper 没说清楚的地方

### 1. Fixed point 真的达到了吗

Paper 只说 4 round 后性能 saturate，但 saturate ≠ fixed point。可能 action hypothesis 在小 neighborhood 里 oscillate，只是 metric 上看起来稳定。需要分析 $\|a^{(k+1)} - a^{(k)}\|$ 是否真的趋近 0。

### 2. World Predictor 是 deterministic 的

公式看起来是 single-sample regression 到 teacher latent target。但真实 future 是 multimodal 的（不同 agent 意图导致不同未来）。如果 predictor 输出 deterministic single future，会不会丢失 multimodality？Action Denoiser 的 6 modes 能 compensate 多少？

这是个 open question。直觉上 deterministic predictor + multimodal denoiser 可能够用，因为 denoiser 本身 explore 不同 mode，predictor 只是给个 "mean future" 当 conditioning。但 paper 没实验。

### 3. 16 个 latent token 够吗

Table 4 显示 16 → 64 token 只 +0.4 PDMS，但 latency 3×。这说明 16 token 在 NAVSIM 这种场景够用。但 NAVSIM 是 non-reactive benchmark，真实 closed-loop 多智能体场景下 16 token 是否够 encode 复杂 interaction？Table 7 的 NAVSIM v2 结果（EPDMS 83.2 < Drive-JEPA 87.8）已经露出短板——DAC 和 NC 偏低，可能跟 representation capacity 不足有关。

### 4. Proposal 和 refinement 共享 weight 真的好吗

直觉上 proposal 是 "from-scratch generation"，refinement 是 "conditional editing"，是两种 distribution 的事。共享 weight 是 simplicity 优先，但可能 suboptimal。可以试试 separate weight，看性能 vs 参数量的 trade-off。

### 5. Latent rollout horizon 和 LLM context length 的类比

这是个有意思的联想：LLM 的 context length 决定能"看多远历史"，DAWN 的 latent rollout horizon 决定能"想多远未来"。Table 6 显示 2-3s 就够，跟 LLM "long context 收益递减"的现象同构。可能背后都是 "effective dimension of reasoning" 的限制——超过某个 horizon， marginal information 衰减到 noise level。

---

## 我的直觉总结

DAWN 给我最大的 intuition 是这个：

**"Thinking" = 在 latent space 做 fixed-point iteration，让 world hypothesis 和 action hypothesis 互相 refine 到 self-consistent。**

这个 formulation 可以推广到很多 domain：
- **Robotics manipulation**: 想象 object 会怎么动 (world) + 我的手怎么动 (action)，co-evolve
- **Game playing**: 想象对手会怎么 response (world) + 我怎么 move (action)，co-evolve —— AlphaZero 的 MCTS 是 explicit version
- **Dialog**: 想象对方会怎么 reply (world) + 我怎么 say (action)，co-evolve —— 但 LLM 现在是 one-shot generate，没有这种 reasoning
- **Code generation**: 想象 code 会怎么执行 (world) + 怎么写 code (action)，co-evolve

每个 domain 的"world"和"action"载体不同，但 fixed-point co-evolution 的 structure 是 universal 的。DAWN 是 driving domain 的 instantiation，但 framework 比 instantiation 重要。

---

## 几个值得 follow 的方向

1. **Stochastic world predictor**: 把 World Predictor 改成 latent diffusion 或 flow matching，让 future hypothesis 也是 multimodal。这样不同 action 可以 trigger 不同 future mode。
2. **Adaptive interaction rounds**: 简单场景 1-2 round 够，复杂场景多跑几轮。像 AlphaZero 的 MCTS simulation budget 自适应难度。
3. **Latent interpretability**: 16 个 latent token 各代表什么？能不能 decode 出来？这对 safety auditing 重要。
4. **Cross-domain transfer**: 把 WAIM framework 搬到 robotics manipulation，看 latent rollout + action denoising 的范式是否通用。
5. **Closed-loop training**: 现在 train 是 open-loop（dataset 里的 trajectory），但 inference 是 fixed-point iteration。Train-test gap 存在。能不能用 RL / DPO 在 closed-loop 里训，让 model 学会"在 inference-time iteration 下表现好"？

---

## 跟你过去的 work 联想

### Software 2.0 / 3.0

你讲过 Software 1.0（code）→ 2.0（learned weights）→ 3.0（prompts）。DAWN 是 2.5 的味道——大部分 learned，但有 explicit iterative structure 当 inductive bias。没有 hand-crafted dynamics，但有 hand-crafted "iterate 4 round" 这个 structure。

### Test-time compute

你一直说 test-time compute 是新维度。DAWN 是 visual policy 的 test-time compute case study。如果把这个 idea 推广——**所有 policy network 都可以加 latent space fixed-point iteration 当 test-time compute**——可能是个 universal technique。

### "Thought" in latent space

你在 No Priors podcast 讲过 LLM chain-of-thought 是 verbalized thought，真正智能可能需要 non-verbal latent thought。DAWN 就是 latent thought 的实例。这跟 LeCun 的 JEPA 哲学合拍，但加了 action-contingent 的 closed loop。

我觉得这是个 convergence 信号——你的 test-time compute 直觉和 LeCun 的 JEPA 哲学在 DAWN 这种 work 里 merge 了。LeCun 给了 latent representation 的 substrate，你给了 test-time iterative reasoning 的 motivation，DAWN 把两者结合起来做 driving。

---

## 一句话 final takeaway

**DAWN 把 world model 和 policy 从"两个分开的 module"重新定义成"一个 fixed-point iteration 的两个 step"——world 预测未来 conditional on action，action 生成 conditional on world，来回迭代到 self-consistent。**

这个 formulation 比 DAWN 这个具体 instantiation 重要。后者用了 V-JEPA + DiT + 4 round + 16 token 这些具体 choice，但 framework 本身可以换 carrier。我觉得这是个值得深挖的方向，特别是把 test-time compute scaling 从 LLM 推到 embodied AI 的角度。

希望这个"人话版"对你有用，Andrej。要是你对哪个点特别感兴趣，比如 fixed-point convergence 分析、跟 AlphaZero MCTS 的 deeper 比较、或者推广到 robotics manipulation 的可行性，咱们可以接着深挖。

---

# DAWN: 深入解析 World-Action Interactive Models

非常高兴能跟你聊这篇 paper，Andrej。我读完后感觉它触及了一个挺根本的问题——**world model 和 policy 之间的"鸡生蛋蛋生鸡"问题**。我会尽量把直觉、公式细节、架构机理都讲透。

---

## 1. 一句话定位

DAWN 提出了一个新范式 **WAIM (World-Action Interactive Model)**：在 inference 时，让 **future world hypothesis** 和 **action hypothesis** 通过一个 short latent rollout **互相 refine**，而不是 parallel branches 或 predict-then-plan 的 one-pass pipeline。它把这个 idea 用 **latent diffusion + JEPA-style predictor** 落地到 autonomous driving，NAVSIM v1 上 perception-free PDMS 89.1，nuScenes 平均 L2 误差 0.33m。

Project page: https://cowarobot-ai.github.io/
arXiv 推测链接（基于作者和标题）: https://arxiv.org/abs/2505.xxxxx (作者还没放完整 arXiv ID，目前以 project page 为准)

---

## 2. 核心动机：Action-Contingent Reciprocity

现有 WAMs 大致三类（paper Fig.1）：
- **Parallel branches**: 一个 backbone 出两个 head，world 和 action 各自走，共享 representation 但 generation 时互不影响。代表：VAViM/VAViM [3]、WorldVLA [5]。
- **Sequential pipeline**: 先预测 future world（pixels / occupancy / latent），再在 frozen future 上 plan。代表：ImagiDrive [23]、Uni-World VLA [36]。
- **Zero-rollout**: 比如 Fast-WAM [57] 主张 inference 时根本不需要 rollout world，直接做 policy。代表：Fast-WAM。

作者的核心 insight：**在 interactive driving 中，未来对决策 relevant 的部分本质上 action-contingent**。比如：
- 一个 gap 是否还 feasible——取决于我是否 yield / push
- 别的车是否 yield——取决于我是否 commit
- 哪些 interaction 变 safety-critical——取决于我选哪个 maneuver

所以 world 和 action 必须 **co-evolve**，否则你 plan 的 future 是 passive future，跟你真的要做的 action 不 self-consistent。

**直觉联想**：这其实是经典的 **" anticipatory planning"** 在心理学和 robotics 里都有。MPC (Model-Predictive Control) 是 action-contingent rollout 的标准做法，但 MPC 是 full horizon rollout in dynamics space；DAWN 想做的是 **short horizon rollout in latent space** 来 capture 这种 reciprocity，但避免 pixel-space 的高成本。另一个联想是 **EM 算法**——E-step 估计 latent world, M-step 优化 action, 交替进行直到 self-consistent。

---

## 3. 形式化定义

### 3.1 从 policy 到 WAM 到 WAIM

**Standard policy** (Eq. 1):
$$p(a_{1:H} \mid o, l)$$
- $a_{1:H}$: action chunk over horizon $H$，比如未来 4 秒的 trajectory tokens
- $o$: current observation
- $l$: task instruction（navigation command）

**WAM** (Eq. 2-3): 显式引入 future world $v_{1:T}$ 作为 latent variable
$$p(v_{1:T}, a_{1:H} \mid o, l)$$
$$p(a_{1:H} \mid o, l) = \int p(v_{1:T}, a_{1:H} \mid o, l) \, dv_{1:T}$$
- $v_{1:T}$: future world representation over horizon $T$（可以是 future observations 或 latent states）
- 关键操作：marginalize over all possible futures——这就是 variational inference / HMM 风格

**WAIM** (Eq. 4): 不再 marginalized，而是找 **self-consistent fixed point**
$$\hat{v}_{1:T} = F_\theta(o, l, \hat{a}_{1:H}), \quad \hat{a}_{1:H} = G_\phi(o, l, \hat{v}_{1:T})$$
- $F_\theta$: World Predictor，参数 $\theta$
- $G_\phi$: Action Denoiser，参数 $\phi$
- $\hat{v}_{1:T}$, $\hat{a}_{1:H}$: 估计的 fixed point

这其实是把 marginalization 替换成了 **iterative coordinate descent / fixed-point iteration**——非常类似 PowerEM / Sinkhorn / GAN 的 alternating optimization。Paper 用 Eq. 5 表达 iterative form:
$$(v_{1:T}^{(k+1)}, a_{1:H}^{(k+1)}) = \mathcal{Z}_\Theta(v_{1:T}^{(k)}, a_{1:H}^{(k)}; o, l)$$
- 上标 $(k)$: 第 k 次 iteration
- $\mathcal{Z}_\Theta$: joint update operator

**Intuition**: 这就是 implicit variational inference 的 fixed-point formulation。一个特别 clean 的视角是：WAM 在做 $\mathbb{E}_{v \sim p(v|o)}[p(a|v,o)]$，而 WAIM 在做 $v^* = \arg\max_v p(v, a^* | o)$, $a^* = \arg\max_a p(a, v^* | o)$ 交替迭代——这把概率推理变成了优化。

---

## 4. DAWN Architecture 拆解

DAWN 是 WAIM 的一个 minimal instantiation，架构包含 6 个组件：

### 4.1 Vision Encoder 分支
- **Student Vision-Encoder** $E_{stu}$: V-JEPA 2 Large backbone
- **Teacher Vision-Encoder** $E_{tea}$: 训练时用，inference 时去掉
- **Auto-Encoder Resampler** $R_{stu}$ / $R_{tea}$: token-space bottleneck autoencoder

公式 (6)-(8):
$$u = E_{stu}(o), \quad z = R_{stu}(u), \quad z_{target} = R_{tea}(E_{tea}(o^+))$$
- $u$: dense visual tokens（V-JEPA 2 输出，量很大，不适合 rollout）
- $z$: 压缩后的 compact latent world tokens（默认 16 个）
- $o^+$: future observation，teacher 用来构造 supervision target
- $z_{target}$: 训练时 World Predictor 的回归目标

**为什么需要 Auto-Encoder Resampler？** V-JEPA 2 Large 输出 dense tokens，4 帧 × patch 16 × tubelet 2 在 512×256 上会产生上千个 token，rollout 这种 dense token 序列计算太贵。Resampler 把它压缩到 16 个 token，是一个 information bottleneck，强迫保留 action-relevant info。

**联想**: 这跟 Perceiver / BLIP-2 的 Q-Former 思路很像——固定数量 learned queries 把 dense features 压成 compact latent set。也跟 recent latent diffusion world model 如 Latent-WAM [46] 思路相似。

### 4.2 World Predictor $P_\theta$
- 实现细节：causal Transformer, 12 layers, embedding dim 384, 12 heads, RoPE
- 输入：current latent $z$, condition tokens $c$, current action hypothesis $a_{1:H}^{(r)}$
- 输出：predicted future latent $z_{future}^{(r)}$
- 训练目标：$L_{WM} = d(\hat{z}_{fut}, z_{tar})$（distance metric，论文没明说，推测是 L2 或 cosine）

**关键点**: World Predictor 是 **action-conditioned** 的——给定 action 假设，预测对应的 future latent。这是 WAIM 中 Action→World 方向的承载。

### 4.3 World-Conditioned Action Denoiser $G_\phi$
- 实现细节：DiT-style diffusion backbone, 384 hidden dim, 12 layers, 12 heads, MLP ratio 4.0
- AdaLN-Zero 修改：timestep/status conditioning vector 不仅 modulate self-attention 和 MLP，**也 modulate cross-attention branch**（to latent world tokens）。每 block 预测 shift, scale, gate 三组参数，作用于 self-attn, cross-attn, MLP 三个 residual 分支。
- 移除了 cross-attention 后额外的 unmodulated MLP，让 block 更干净：modulated self-attn → modulated cross-attn → modulated FFN
- 6 modes/samples（multimodal trajectory hypotheses），5 DPM-Solver++ sampling steps
- Trajectory 表示：per-pose tokens，0.5s 间隔

**关键点**: Action Denoiser 是 **world-conditioned** 的——给定 predicted future latent，denoise action tokens。这是 WAIM 中 World→Action 方向的承载。

### 4.4 角色切换：Proposal vs Refinement
一个有意思的设计：Action Denoiser 的权重在 proposal 和 refinement 两种 role 间 **shared**，只是 query embedding 和 input source 不同：
- $q_{prop}$: proposal generation 的 role-specific query
- $q_{ref}^{(r)}$: 第 r 次 refinement 的 role-specific query
- 训练时同时训两种 role

这让我想起 DETR 的 object queries，以及最近 Perceiver-style cross-attention 里的 learned queries。也让我想起 **in-context learning 的 role conditioning**——同一个 model，不同 query embedding 切换"任务模式"。

### 4.5 完整交互流程（Eq. 9）

训练时:
$$a_{1:H}^{(0)} = G_\phi(q_{prop}, c, z)$$
$$z_{future}^{(r)} = P_\theta(z, c, a_{1:H}^{(r)})$$
$$a_{1:H}^{(r+1)} = G_\phi(q_{ref}^{(r)}, c, z_{future}^{(r)}, a_{1:H}^{(r)})$$

Inference 时（Eq. 11-13）:
$$a_{1:H}^{(0)} = G_\phi(q_{init}, c, z)$$
$$z_{future}^{(k+1)} = P_\theta(z, c, a_{1:H}^{(k)})$$
$$a_{1:H}^{(k+1)} = G_\phi(q_{ref}^{(k)}, c, z_{future}^{(k+1)}, a_{1:H}^{(k)})$$
$$\hat{\tau} = H_{act}(a_{1:H}^{(K)})$$

- $q_{init}$: inference 时的初始 action queries
- $K$: 总交互轮数（默认 4）
- $H_{act}$: Action Head, 把 denoised action states 解码成最终 trajectory $\hat{\tau}$

**Inference 时第一个 action hypothesis 直接从 resampler latent 生成**，不需要先走 Predictor——这节省了一步开销，也符合"先用视觉先验做粗略 plan，再用 world rollout refine"的直觉。

---

## 5. 训练 Pipeline（4 阶段）

这是 engineering 上比较关键的部分，作者用 stage-wise recipe 稳定优化：

| Stage | 数据 | 目标 | Backbone |
|---|---|---|---|
| 1. Vision pretraining | OpenScene, DrivingDojo, CoVLA | V-JEPA 2 self-supervised | 256×512, 2 Hz |
| 2. Resampler training | 同上 | Token-space autoencoder | 16 latent tokens |
| 3. Predictor training | nuScenes, NAVSIM | Future latent regression | causal Transformer |
| 4. Joint world-action | nuScenes, NAVSIM | World loss + planning loss | All together |

**Resampler training 时还附加了 auxiliary diffusion planner head**，目的是让 compressed tokens 保留 action-relevant information。这个 trick 很重要——单纯 reconstruct dense features 不够，必须 task-aware。这让我想起 **distillation with auxiliary task** 和 **information bottleneck with downstream loss**。

**Joint training 的损失**:
- World loss: $L_{WM} = d(\hat{z}_{fut}, z_{tar})$
- Planning loss: classification + regression + velocity + yaw，其中 velocity 和 yaw 项 weighted by 0.5

**优化超参**: 150 epochs, peak lr 1e-4, initial lr 5e-5, 8 warmup epochs, weight decay 0.04, EMA momentum 从 0.996 → 0.999, 80× A100 GPU

---

## 6. 实验数据深度解读

### 6.1 NAVSIM v1 (Table 1)

主要 metric：
- **NC** (No-at-fault Collisions) ↑
- **DAC** (Drivable Area Compliance) ↑
- **EP** (Ego Progress) ↑
- **C** (Comfort) ↑
- **TTC** (Time-to-Collision) ↑
- **PDMS** (Predictive Driver Model Score) ↑ aggregate

DAWN 结果：
- PDMS 89.1（perception-free SOTA，超过 Drive-JEPA 86.2）
- NC 98.7（best）
- EP 84.3（best）
- TTC 96.0（best）—— 这是 WAIM 哲学的关键证据，TTC 直接反映对未来 collision 的 awareness
- DAWN* (256×256) PDMS 87.9，比 full model (512×256) 低 1.2，说明高分辨率输入确实有用

**有意思的对比**:
- Perception-based 的 GoalFlow 90.3，DriveDPO 90.0，依然略高于 DAWN
- 但 perception-based 需要 LiDAR/Camera 的 full perception stack，DAWN 是 perception-free（只 Camera）

### 6.2 nuScenes (Table 2)

DAWN 大幅领先：
- L2 avg 0.33m (vs WorldRFT 0.47m，相对降低 30%)
- 1s L2: 0.17m, 2s: 0.31m, 3s: 0.52m
- Collision rate avg 0.11% (vs WorldRFT 0.15%, World4Drive 0.16%)

**长 horizon 优势更明显**: 3s L2 0.52m vs WorldRFT 0.76m，这正好对应 WAIM 的核心 claim——long-horizon trajectory 在 interactive scene 下需要 action-world co-evolution，单纯的 predict-then-plan 在长 horizon 会累积误差。

### 6.3 组件消融 (Table 3 / Table 8)

| Resampler | Predictor | Interactive | PDMS |
|---|---|---|---|
| ✗ | ✗ | ✗ | 82.9 |
| ✓ | ✗ | ✗ | 82.8 |
| ✓ | ✓ | ✗ | 85.2 |
| ✓ | ✓ | ✓ | **87.9** |

**核心发现**:
1. Resampler 单独加几乎没用（82.9 → 82.8）——compact bottleneck 本身不提升 planning
2. Predictor 加上跳 +2.4——explicit future rollout 重要
3. Interactive 再加 +2.7——双向 refine 是关键，**比单纯有 future 还重要**

### 6.4 Interactive rounds (Table 9 / Fig. 3)

| Rounds | NC | DAC | EP | TTC | PDMS |
|---|---|---|---|---|---|
| 1 | 97.4 | 94.3 | 80.4 | 91.5 | 85.2 |
| 2 | 97.8 | 95.1 | 81.6 | 94.1 | 86.4 |
| 3 | 98.1 | 95.6 | 82.8 | 95.6 | 86.9 |
| **4** | **98.2** | **95.8** | **84.2** | **95.8** | **87.9** |
| 5 | 98.1 | 95.4 | 83.9 | 95.7 | 87.2 |
| 6 | 98.0 | 95.6 | 82.8 | 95.6 | 86.9 |

**关键观察**:
- 从 1 → 4 round 单调上升，+2.7 PDMS
- 4 round 之后 saturate 并轻微下降——over-refinement 开始 overfitting 到 spurious signal
- TTC 从 91.5 跳到 95.8——interactive refinement 对安全 metric 尤其有效

**联想**: 这非常像 **diffusion sampling 的 steps** 和 **iterative refinement in AlphaZero 的 MCTS 模拟次数**——少了几何级数提升，多了 saturate。也让我想起 **DeepSeek-R1 / OpenAI o1 的 test-time compute scaling**——inference 时多花 compute 换性能。

### 6.5 双向消融 (Table 5)

| Method | NC | DAC | EP | TTC | PDMS |
|---|---|---|---|---|---|
| DAWN (full) | 98.2 | 95.8 | 84.2 | 95.7 | 87.9 |
| w/o World→Action | 96.6 | 91.9 | 78.6 | 91.6 | 81.6 |
| w/o Action→World | 97.3 | 94.3 | 80.2 | 92.7 | 84.9 |

**关键发现**:
- 去掉 World→Action（action denoiser 不再 condition on predicted world）: PDMS 暴跌 6.3 → 81.6
- 去掉 Action→World（world predictor 不再 condition on action hypothesis）: PDMS 降 3.0 → 84.9
- **World→Action 方向更重要**——这符合直觉，因为 action generation 直接依赖 future 来判断 safety
- 但 Action→World 也有显著贡献——pure passive future prediction 会 plan 出跟自己 action 不一致的 world

### 6.6 Rollout horizon (Table 6)

| $T_w$ | $H_a$ | PDMS | w/o Int. | Latency (ms) |
|---|---|---|---|---|
| 0s | 4s | 82.8 | 82.8 | 331 |
| 1s | 4s | 84.7 | 83.9 | 503 |
| 2s | 4s | 87.3 | 84.3 | 690 |
| 3s | 4s | 87.5 | 84.6 | 849 |
| 4s | 4s | 87.9 | 85.2 | 1067 |

- $T_w$: latent world rollout horizon
- $H_a$: action horizon
- w/o Int.: 同样 rollout 但不做 interactive refinement

**关键发现**:
1. Zero rollout (Fast-WAM-like) 82.8，确实差
2. **大部分 gain 在 2-3s 就拿到了**，从 2s 87.3 到 4s 87.9 只多 0.6
3. 但 w/o Int. 列显示：rollout alone 不够，必须配合 interactive refinement 才能转化成性能
4. Latency 从 331ms 到 1067ms，几乎是线性增长——short rollout 的 efficiency 优势明显

**这是 paper 最 actionable 的发现之一**: WAIM 不需要 full future simulator，short latent rollout 已经足够，这把 world modeling 从"must be a perfect simulator"重新定位为"提供 action-relevant dynamic hypothesis"。

### 6.7 Resampler tokens (Table 4 / 10)

| # Tokens | PDMS | Latency (ms) |
|---|---|---|
| 16 | 82.8 | 331 |
| 64 | 83.2 | 963 |

- 4× tokens 只换 +0.4 PDMS，但 latency 3× 增加
- 16 tokens 已经足够 capture action-relevant scene structure
- **这是 latent space world model 的 capacity-efficiency sweet spot**

---

## 7. 直觉构建：为什么 WAIM 工作？

让我从几个角度给你建立 intuition：

### 7.1 控制论视角
经典 MPC 是：给定 dynamics model $f$，优化 $a^* = \arg\max_a \sum_t R(s_t, a_t)$ s.t. $s_{t+1} = f(s_t, a_t)$。这里 dynamics 是显式的、frozen 的。DAWN 把 dynamics 变成 **learned latent predictor**，把 optimization 变成 **diffusion denoising with iterative conditioning**。MPC 的 cost 是 horizon 线性 × 优化迭代，DAWN 通过 latent rollout + diffusion 把这个 cost 控制在可接受范围。

### 7.2 Diffusion + Test-time Refinement
Diffusion policy 本身就是 iterative denoising。DAWN 的 trick 是：**在每个（或几个）denoising step 后插入 world rollout update**，让 condition 本身随 action hypothesis 演化。这本质是 **inner loop (denoise action) + outer loop (update world)** 的 nested iteration。

### 7.3 Variational Inference 类比
Eq.3 的 marginalization $\int p(v, a | o) dv$ 在实践中 intractable。WAIM 把它替换成 **point estimate + iterative refinement**:
$$v^* = \arg\max_v p(v | a^*, o), \quad a^* = \arg\max_a p(a | v^*, o)$$
这是 **mean-field variational inference** 的 coordinate ascent 写法。Mean-field 假设 $q(v, a) = q(v)q(a)$，然后交替更新。这跟 EM、Sinkhorn、BLIP-2 bootstrapping 都同源。

### 7.4 Self-Consistency / Fixed-Point
WAIM 找的是 self-consistent pair $(\hat{v}, \hat{a})$，满足 $\hat{v} = F(\hat{a}), \hat{a} = G(\hat{v})$。这是 **Bethe fixed point / Loopy Belief Propagation** 风格。GAN 的 generator-discriminator 也是同构——两个网络互为 condition，迭代收敛到 Nash equilibrium。DAWN 是 cooperative 版本（共享目标），不是 adversarial。

### 7.5 "Thinking" in Latent Space
这跟 o1 / R1 / AlphaZero 的 test-time thinking 同源——**inference 时多花 compute 换更好答案**。DAWN 的 short latent rollout 就是 "想象一下未来会发生什么，然后调整我的 plan"。区别于 LLM 的 chain-of-thought（token space reasoning），DAWN 在 **latent space reasoning**，更紧凑、更几何化。

---

## 8. 局限和 Open Questions

Paper Appendix A 自己列了几个：
1. **WAIM 假设未来 action-contingent**，简单场景下 zero-rollout WAM 可能更高效
2. **没有形式化收敛性 / safety 保证**——recursive interaction 可能 oscillate
3. **Latent representation 不可解释**——难诊断 rare safety-critical cues 是否保留
4. **数据依赖**——benchmarks 增益是标准协议下的，不是 deployment ready

我的几个 additional thoughts:
- **Fixed-point 可能不收敛**: 现实中 4 round 就 saturate，但理论上可能存在 oscillating cases，paper 没给分析
- **Latent world 的 physical fidelity**: 16 tokens 是否能 encode 复杂 multi-agent interaction？Table 4 显示 16→64 tokens 只 +0.4 PDMS，可能说明 16 tokens 已经 capture 了大部分 relevant info，但更复杂场景可能不够
- **Action modes 的多样性**: 6 modes 是否够 cover multimodal driving intentions？Highway merge vs yield 是两种很不同的 mode
- **Closed-loop evaluation**: NAVSIM 是 non-reactive，NAVSIM v2 的 EPDMS 83.2 略低于 Drive-JEPA 87.8，说明 closed-loop / reactive 场景下 DAWN 还有空间

---

## 9. 相关工作脉络

### 9.1 WAM 家族
- **VAViM/VAViM** [Bartoccioni et al. 2025]: parallel video generation + planning
- **WorldVLA** [Cen et al. 2025]: autoregressive action world model
- **WAM-Flow** [Xu et al. 2025]: discrete flow matching for parallel coarse-to-fine
- **Latent-WAM** [Wang et al. 2026]: spatially-aware compressive encoders
- **DreamZero** [Ye et al. 2026]: video diffusion backbone
- **Fast-WAM** [Yuan et al. 2026]: zero-rollout, video co-training is enough
- **Percept-WAM** [Han et al. 2025]: unify 2D/3D perception tokens

### 9.2 Driving World Models
- **GAIA-1** [Hu et al. 2023]: 早期 generative world model
- **Drive-WM** [Wang et al. 2024]: multi-view visual forecasting
- **Drive-JEPA** [Wang et al. 2026]: V-JEPA + multimodal trajectory distillation，paper 主要 baseline
- **VISTA** [Gao et al. 2024]: generalizable driving world model
- **UniFuture** [Liang et al. 2025], **HERMES** [Zhou et al. 2025]: 4D geometric constraints
- **Uni-World VLA** [Liu et al. 2026]: interleaved world modeling + planning
- **Infinite-World** [Wu et al. 2026]: 1000-frame horizon via pose-free hierarchical memory

### 9.3 E2E Driving
- **UniAD** [Hu et al. 2023]: full-stack unified
- **VADv2** [Chen et al. 2024]: probabilistic planning over discretized tokens
- **SparseDrive** [Sun et al. 2025]: query-centric, sparse
- **DiffusionDrive** [Liao et al. 2025]: truncated diffusion
- **GoalFlow** [Xing et al. 2025]: goal-driven flow matching
- **DriveDPO** [Shang et al. 2025]: safety DPO
- **ReAL-AD** [Lu et al. 2025]: reasoning-augmented hierarchical
- **Orion** [Fu et al. 2025], **UniDriveVLA** [Li et al. 2026]: MLLM bridge

### 9.4 V-JEPA 2 (Backbone)
DAWN 用 V-JEPA 2 Large 作为 vision backbone。V-JEPA 2 [Assran et al. 2025] 是 LeCun 团队的 self-supervised video model，从 internet-scale video 学 predictive representation，不 reconstruct pixels，强调 latent predictive learning。这跟 DAWN 的 latent philosophy 完全一致——避免 pixel-space future rendering，在 semantic latent space 做 reasoning。

参考: https://arxiv.org/abs/2506.09985 (V-JEPA 2)

---

## 10. 我的几个 Critique / Question

1. **为什么 proposal 和 refinement 共享 weight？** 这降低了 model 容量，作者说是为了 simplicity，但 query embedding 区分 role 是否足够？直觉上 proposal 是 "from-scratch generation"，refinement 是 "conditional editing"，是不同 distribution 的事，共享 weight 可能是 suboptimal 的。

2. **World Predictor 是 deterministic 的吗？** 公式看起来是 single-sample regression 到 $z_{target}$。但真实 future 是 multimodal 的（不同 agent 意图）。如果 predictor 是 deterministic，可能丢失 multimodality，那 action denoiser 的 6 modes 又能 compensate 多少？这是 paper 没讨论的。

3. **Action 的 representation**: per-pose tokens at 0.5s interval，意味着 4s horizon = 8 个 action tokens。这相对 coarse。VAD 系列 vectorized representation 更精细。为什么 DAWN 不用 vectorized？可能是为了和 diffusion token paradigm 对齐。

4. **Self-consistent fixed point 是否真的达到？** Paper 没分析 inference 后 $F_\theta(\hat{a})$ 是否真的等于 $\hat{v}$——只是经验上 4 round 后 saturate，但 saturate ≠ fixed point。可能 action hypothesis 在小 neighborhood 内 oscillate。

5. **Long-tail safety**: Latent representation 不可解释，难诊断 rare case 是否 capture。这对 deployment 是大问题。

6. **vs MPC**: 真正的 MPC 有 dynamics consistency guarantee，DAWN 的 learned predictor 没有。但 MPC 在 autonomous driving 一直没 dominate，原因是 hand-crafted dynamics 难 capture complex multi-agent interaction。DAWN 用 learned predictor 弥补这个，但失去了 formal guarantee。

---

## 11. 跟你（Karpathy）过去 work 的 connection

我觉得这篇 paper 跟你过去一些思考很有 connection：

### 11.1 Software 2.0 / Dataset → Neural Net
你说过 Software 1.0（hand-coded）→ Software 2.0（learned）→ Software 3.0（prompted）。DAWN 是 Software 2.5 的感觉——大部分 learned，但有 explicit 的 iterative reasoning structure（类似 inductive bias）。这让我想起你在 nanoGPT 里的"keep it simple"哲学——DAWN 也很 simple，没有复杂的多模块 cascade，就是 encoder + predictor + denoiser + head，靠 iterative interaction 拿性能。

### 11.2 Test-time Compute Scaling
你最近一直讲 test-time compute 是新维度。DAWN 是一个 non-LLM 领域的 test-time compute 案例——inference 时 4 round latent rollout + diffusion sampling。这跟 o1 / R1 / AlphaZero 同源，但载体是 visual policy 而不是 language。我觉得这是 test-time compute scaling 从 LLM 扩展到 robotics / embodied AI 的一个早期 signal。

### 11.3 "Thought" in Latent Space
你 2024 年在 No Priors podcast 讲过 LLM chain-of-thought 是 "verbalized thought"，但真正智能可能需要 "non-verbal thought"——latent space 的 reasoning。DAWN 就是 latent space thought 的实例：不 generate tokens，而是 refine latent world-action pair。这跟 LeCun 的 JEPA 哲学也合拍。

### 11.4 Yann LeCun 的 H-JEPA / V-JEPA
你跟 LeCun 在 reasoning 和 world model 上有公开 debate。DAWN 用 V-JEPA 2 作为 backbone，但又加上 iterative interaction——这其实是把 LeCun 的 JEPA 推到 action-contingent 方向，跟你的 "test-time compute" 直觉 merge。我觉得这是两个哲学的 convergence 点。

---

## 12. 总结

**DAWN 的真正 contribution**:
1. **WAIM 形式化**: 把 world-action co-evolution 定义为 self-consistent fixed point，给了一个 clean 的数学框架（Eq. 4-5）
2. **Short latent rollout 的实证**: 2-3s rollout 已经够，不需要 full future simulator——这把 world model 从"必 imperfect"重新定位为"做最小必要的工作"
3. **Bidirectional interaction 的 ablation 证据**: Table 5 直接证明两个方向都重要，World→Action 贡献更大（-6.3 vs -3.0）
4. **Perception-free SOTA on NAVSIM**: PDMS 89.1, TTC 96.0，且 nuScenes 上 L2 误差大幅领先

**我觉得这篇 paper 最大的 intuition gift**: **Inference-time 的"思考" = latent space 的 world-action co-evolution**。这跟 LLM 的 chain-of-thought 是同一种范式（test-time iterative refinement），但载体更紧凑、更几何化。如果我们把 LLM 当作 "verbal thought engine"，那 DAWN / WAIM 就是 "embodied thought engine" 的 early prototype。

期待听到你的 thoughts，特别是关于 (1) fixed point 是否真的达到 (2) latent rollout horizon 和 LLM context length 的类比 (3) 是否能扩展到 robotics manipulation。

**References / Web Links**:
- DAWN project page: https://cowarobot-ai.github.io/
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- NAVSIM: https://github.com/autonomousvision/navsim
- nuScenes: https://www.nuscenes.org/
- Fast-WAM (referenced): https://arxiv.org/abs/2603.16666 (推测)
- Drive-JEPA: https://arxiv.org/abs/2601.22032
- WorldVLA: https://arxiv.org/abs/2506.21539
- Latent-WAM: https://arxiv.org/abs/2603.24581
- UniAD: https://opendrivelab.com/e2ead/UniAD/
- VADv2: https://arxiv.org/abs/2402.13243
- DiffusionDrive: https://arxiv.org/abs/2410.15939
- GAIA-1: https://arxiv.org/abs/2309.17080
- VISTA: https://arxiv.org/abs/2401.02977
- OpenScene: https://github.com/OpenDriveLab/OpenScene
- DrivingDojo: https://drivingdojo.github.io/
- CoVLA: https://github.com/TMEyeen/Covla
- DiT (Diffusion Transformer): https://www.wpeebles.com/DiT
- Perceiver: https://arxiv.org/abs/2103.03206
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597
