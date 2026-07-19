---
source_pdf: A Path Towards Autonomous Machine Intelligence.pdf
paper_sha256: 5dda279009beae358f08b18b2a4374bfbe65f8c9ed773ea0b9b49f9068566022
processed_at: '2026-07-17T20:49:32-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# LeCun 的 "A Path Towards Autonomous Machine Intelligence" 深度解读

Hey Karpathy! 这篇 paper 我读过好几遍，每次读都能挖出一些新的 insight。LeCun 在这里 essentially 是在 propose 一整个 cognitive architecture blueprint，目的是替代当前以 LLM + RL 为主导的 paradigm。我尽量按 architecture → JEPA → H-JEPA → planning → 与其他 paradigm 对比 这条线展开，并把公式里每个 symbol 的物理含义都讲清楚。

---

## 1. Big Picture: LeCun 在 attack 什么 problem

LeCun 开篇就提了三个 core challenges，这三个 challenges 基本定义了整篇 paper 的 motivation：

1. **如何让机器通过 observation 学习 world model**（而不是必须通过昂贵的 interaction）
2. **如何让 reasoning & planning 与 gradient-based learning 兼容**（对抗 symbolic logic 与 deep learning 的分裂）
3. **如何学习 hierarchical representations** 以支持 multi-time-scale 的 prediction & planning

一个核心 claim：动物和人类之所以 sample-efficient（adolescent 学开车只要 20 小时），是因为我们有一个 internal **world model** 可以做 mental simulation，让我们在真正 action 之前先 "imagine" 后果。当前的 RL system 缺少这种能力，所以必须 trial-and-error 几百万次才能学好一个 Atari game。LeCun 把这个 deficit 归结为缺少一个好的、可训练的、能 handle uncertainty 的 world model。

这里有 LeCun 后续在很多 talk 中反复强调的 "house cat > self-driving car" argument：一只猫从来不需要几百万个 episode 就能学会在房间里 navigate、jump、hunt。这背后是 common sense，而 common sense = a collection of world models。

参考链接：
- LeCun 在 NYU 的 talk: https://www.youtube.com/watch?v=ODM5Mh3J1tQ
- 原文 paper PDF: https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 2. The Overall Cognitive Architecture (Section 3, Figure 2)

LeCun 提出的 architecture 由 6 个 modules 组成，全部 differentiable（这很关键，让 backprop 可以贯穿整个 system）：

### 2.1 六个 Modules

| Module | Function | Brain Analogy |
|--------|----------|---------------|
| **Configurator** | Executive control，modulate 其他 modules 的 parameters & attention | Prefrontal cortex |
| **Perception** | $s[0] = Enc(x)$，从 sensor 输入估计当前 world state | Sensory cortex |
| **World Model** | $s[t+1] = Pred(s[t], a[t])$，预测 future states | Prefrontal cortex |
| **Cost** | $C(s) = IC(s) + TC(s)$，输出 scalar "energy" | Amygdala + reward circuit |
| **Short-term Memory** | Key-value memory，存过去 states & costs | Hippocampus |
| **Actor** | Propose action sequences，optimize via MPC | Premotor cortex |

### 2.2 Cost Module 的细节 (Section 3.2)

这是整个 architecture 的 "driver"。Cost 由两部分组成：

$$C(s) = IC(s) + TC(s) \tag{1}$$

$$IC(s) = \sum_{i=1}^{k} u_i \, IC_i(s) \tag{2}$$

$$TC(s) = \sum_{j=1}^{l} v_j \, TC_j(s) \tag{3}$$

**Symbol 解释**：
- $s$：world state representation（由 perception module 输出）
- $IC_i(s)$：第 $i$ 个 intrinsic cost submodule，e.g. pain signal, hunger, battery level, proximity to danger
- $u_i \in \mathbb{R}$：configurator 设定的 weight，控制第 $i$ 个 drive 在当前 task 中的权重。例如 curiosity-driven exploration 时把 curiosity term 调大
- $TC_j(s)$：第 $j$ 个 trainable critic submodule，预测 future intrinsic cost
- $v_j$：critic 的权重，同样由 configurator 调制

LeCun 强调 $IC$ 必须 **immutable**（不可训练）—— 否则 agent 可以学会 hack 自己的 reward（wireheading problem），把 pain signal 永远设成 0。这与 RL 里的 reward shaping 不一样，因为 RL 里 reward 来自 environment 是黑盒，但这里 $IC$ 是 designer 直接 specified 的 differentiable function。

这一点和 **Anthropic 的 Constitutional AI**、**OpenAI 的 reward modeling** 都形成对比。LeCun 的 view 是：basic drives 应该 hard-wired，但 long-term value 应该是 learned。

### 2.3 Mode-1 vs Mode-2 (Kahneman's System 1/2 analogy)

LeCun 借用 Kahneman 的 "Thinking Fast and Slow" 框架：

**Mode-1 (reactive, System 1)**：
$$a[0] = A(s[0])$$
直接 policy mapping，no planning，one forward pass。对应 LLM 的 next-token prediction，或者 reflex behavior。

**Mode-2 (deliberative, System 2)**：
完整 MPC loop：
1. Perception: $s[0] = P(x)$
2. Actor propose: $(a[0], \dots, a[T])$
3. World model roll out: $s[t+1] = Pred(s[t], a[t])$，得 sequence $(s[1], \dots, s[T])$
4. Cost evaluate: $F(x) = \sum_{t=1}^{T} C(s[t])$
5. Optimize action sequence via gradient descent through unrolled compute graph
6. Execute first action $a[0]$, repeat

$$F(x) = \sum_{t=1}^{T} C(s[t]) \tag{Section 3.1.2}$$

**Intuition**：Mode-2 = differentiable MPC。整个 roll-out 是 differentiable 的，所以 $\nabla_a F$ 可以通过 backprop through time 得到，然后 actor 用 gradient descent 找最优 action sequence。这是 **classical optimal control** (Bryson & Ho 1969) 的 modern reincarnation，只是 world model 和 cost 都是 learned。

LeCun 还提到 Mode-2 → Mode-1 的 "compilation"：当某个 Mode-2 task 跑过很多次后，可以把优化结果 distill 进 policy module $A(s[t])$，让 agent 之后用 Mode-1 reflexively 完成。这其实就是 **amortized inference** 的思想，类似 Dayan & Hinton 1993 的 "wake-sleep"。

---

## 3. JEPA: The Heart of the Paper (Section 4.4)

### 3.1 Why not Generative Models?

LeCun 在 Section 4.1-4.3 花了大量篇幅论证 generative architectures 有问题。他的 argument 是：

**Generative model 直接预测 $y$（e.g. 下一个 video frame 的所有 pixels）会有 multi-modality problem**：一辆车到 fork，可能左转也可能右转。如果 model 直接生成 pixel，要么 average 两个 outcome（blurry prediction），要么必须用 GAN/VAE/VQ-VAE 这些 trick。但 GAN 有 mode collapse 和 training instability，VAE 有 posterior collapse 和 blurry reconstruction。

更严重的是 **curse of dimensionality**：高维 $y$ space 里，contrastive methods 需要的 negative samples数量指数增长。

### 3.2 JEPA Architecture (Figure 12)

JEPA 的核心 idea 是 **在 representation space 而不是 pixel space 做预测**。结构如下：

$$s_x = Enc_x(x), \quad s_y = Enc_y(y)$$
$$E_w(x, y, z) = D(s_y, \text{Pred}(s_x, z)) \tag{10}$$
$$\check{z} = \arg\min_{z \in \mathcal{Z}} E_w(x, y, z) \tag{11}$$
$$F_w(x, y) = \min_{z \in \mathcal{Z}} E_w(x, y, z) = D(s_y, \text{Pred}(s_x, \check{z})) \tag{12}$$

**Symbol 解释**：
- $x$：observed input（e.g. 视频过去 frames）
- $y$：target input（e.g. 视频未来帧），训练时 observed，inference 时可能 unobserved
- $Enc_x, Enc_y$：两个 encoder（可以 share parameters 也可以不 share，可以处理不同 modality 比如 video + audio）
- $s_x \in \mathbb{R}^{d_x}$：$x$ 的 abstract representation
- $s_y \in \mathbb{R}^{d_y}$：$y$ 的 abstract representation
- $z \in \mathcal{Z}$：latent variable，capture 不可从 $x$ 预测的 information（e.g. 车左转还是右转）
- $\text{Pred}$：predictor module，输入 $s_x$ 和 $z$，输出预测的 $\tilde{s}_y$
- $D(\cdot, \cdot)$：distance / divergence，通常是 L2 norm $\|s_y - \tilde{s}_y\|^2$
- $\check{z}$：inference 后最优 latent
- $F_w(x, y)$：marginalized energy，已 minimization over $z$

**关键 insight**：JEPA 是 **non-generative** 的。它不能直接生成 $y$，它只告诉你 $s_y$ 应该长什么样。这看起来像缺点，其实是优点——因为 encoder $Enc_y$ 可以选择性地 ignore 不可预测的 details（e.g. carpet texture、树叶随风摆动），只保留可预测的 abstract content（e.g. 车的位置、速度）。

这就解决了 multi-modality：encoder 的 invariance 自然把多个 plausible $y$ 映射到同一个 $s_y$，而 latent $z$ 捕获离散的多模态。

### 3.3 Multi-modality via Two Mechanisms

LeCun 指出 JEPA 有 **两种 multi-modality mechanism**：

1. **Encoder invariance**：$Enc_y$ 把多个 $y$ map 到同一个 $s_y$，自然 collapse 掉不可预测的 details
2. **Latent variable**：$z$ 变化 over $\mathcal{Z}$ 时，$\text{Pred}(s_x, z)$ 产生多个 plausible $\tilde{s}_y$

**Example**：$x$ = 车接近 fork 的 video clip。$s_x, s_y$ 表示车的位置和速度。$z \in \{0, 1\}$ 表示左转还是右转。$z=0$ 时 $\tilde{s}_y$ 是左转后的位置，$z=1$ 时是右转后的位置。

---

## 4. Training JEPA: Why Non-Contrastive (Section 4.5)

### 4.1 The Four Criteria

LeCun 列出 JEPA 训练的 4 个 criteria：

1. **Maximize $I(s_x; x)$**：$s_x$ 要 retain 关于 $x$ 的 information
2. **Maximize $I(s_y; y)$**：$s_y$ 要 retain 关于 $y$ 的 information
3. **Make $s_y$ predictable from $s_x$**：minimize $D(s_y, \tilde{s}_y)$
4. **Minimize $I(z; \text{everything else})$**：latent $z$ 的 information content 要最小

Criteria 1, 2, 4 共同防止 **informational collapse**：如果不约束 $s_x$ 的 info content，encoder 可以输出常数 $s_x = c$，那 energy 永远是 0（完美的 trivial collapse）。

Criterion 4 防止另一种 collapse：如果 $z$ 的维度等于 $s_y$ 的维度，predictor 可以 ignore $s_x$，直接 copy $z$ 到 output，那 $\check{z} = s_y$ 永远让 energy = 0。这也是为什么 VAE 的 latent 必须有 KL regularization。

### 4.2 VICReg (Section 4.5.1)

VICReg = **V**ariance-**I**nvariance-**C**ovariance **Reg**ularization，是 LeCun 组 (Bardes et al. 2021) 提出的 non-contrastive SSL method。结构如下：

$$s_x \xrightarrow{\text{expander}} v_x \in \mathbb{R}^{d_v}$$
$$s_y \xrightarrow{\text{expander}} v_y \in \mathbb{R}^{d_v}$$

Loss = Invariance + Variance + Covariance:

$$\mathcal{L} = \underbrace{\|s_x - s_y\|^2}_{\text{Invariance}} + \lambda \underbrace{\sum_i \max(0, \gamma - \sqrt{\text{Var}(v_{\cdot,i}) + \epsilon})}_{\text{Variance (hinge)}} + \mu \underbrace{\sum_{i \neq j} \frac{C_{ij}^2}{N}}_{\text{Covariance}}$$

**Symbol 解释**：
- $v_{\cdot,i}$：batch 上第 $i$ 个 component 的取值向量
- $\text{Var}(v_{\cdot,i})$：第 $i$ 个 component 在 batch 上的 variance
- $\gamma$：variance 的 threshold（通常 $\gamma=1$）
- $C_{ij}$：covariance matrix $C = \frac{1}{N}\sum_n (v_{n} - \bar{v})(v_n - \bar{v})^\top$ 的第 $(i,j)$ entry
- $\lambda, \mu$：loss weights
- $\epsilon$：small constant for numerical stability

**Intuition**：
- **Invariance term**：拉近 $s_x$ 和 $s_y$（criterion 3）
- **Variance term**：强制每个 component 的 std ≥ $\sqrt{\gamma}$，防止 collapse 到常数（criterion 1, 2）
- **Covariance term**：让不同 components 之间 decorrelated，相当于 maximize info content per dimension（criterion 1, 2）

**为什么 VICReg 比 contrastive methods 好**？Contrastive methods (SimCLR, MoCo, InfoNCE) 需要 negative samples 来 push apart。在高维 $s_y$ space 里，需要 exponentially多的 negatives 才能 "cover" 所有方向。VICReg 是 **dimension-contrastive** 而非 sample-contrastive：它 push apart 的是 representation 的不同 dimensions，只需要 batch statistics 就够了。

参考：
- VICReg paper: https://arxiv.org/abs/2105.04906
- Barlow Twins (类似 idea): https://arxiv.org/abs/2103.03230
- LeCun 关于 SSL 的 lecture: https://www.youtube.com/watch?v=7SGtBRMEgC8

### 4.3 Latent Variable Regularization

Critic 4 的实现方式有四类：

1. **Discretization**: $z$ 是 discrete categorical，e.g. VQ-VAE 风格。如果 $|\mathcal{Z}| = K$，则最多 $K$ 个 zero-energy points
2. **Dimensionality/rank minimization**: $z \in \mathbb{R}^d$ 但 $d \ll d_y$，e.g. Implicit Rank-Minimizing AE (Jing et al. 2020)
3. **Sparsification**: $R(z) = \alpha \|z\|_1$，类似 sparse coding (Olshausen & Field 1996), LISTA (Gregor & LeCun 2010)
4. **Fuzzification / stochastic**: $z \sim \mathcal{N}(\mu, \sigma^2)$ with KL regularization, 这是 VAE (Kingma & Welling 2013) 的核心 idea

LeCun 在这里其实是在做一个 unified view：很多看似不同的 method（VQ-VAE, VAE, sparse coding, denoising AE）都是 regularized latent variable EBM 的特例，区别只在于 regularizer $R(z)$ 的选择。

---

## 5. H-JEPA: Hierarchical Prediction (Section 4.6)

### 5.1 Why Hierarchical?

单一的 JEPA 只能在一个 abstraction level 做 prediction。但人类 plan 是分层的：

- **High-level**：我明天 9 点要去机场（abstract, long-horizon）
- **Mid-level**：坐 8 点的地铁，转机场快线
- **Low-level**：抬腿走路，开门，刷卡
- **Motor-level**：每毫秒的肌肉 contraction

低层 representation 含 detail 多，可以做 short-term prediction；高层 representation 抽象，可以做 long-term prediction。这正好对应一个 driving example：精确预测未来 10 秒的 trajectory 很难（其他车、红绿灯都 unpredictable），但预测 "1 小时后到家" 这种 abstract outcome 容易。

### 5.2 H-JEPA Architecture (Figure 15)

Stack 多层 JEPA：

$$s_1[0] = Enc_1(x) \tag{level 1 encoding}$$
$$s_2[0] = Enc_2(s_1[0]) \tag{level 2 encoding}$$
$$\vdots$$
$$s_L[0] = Enc_L(s_{L-1}[0]) \tag{level L encoding}$$

每一层独立做 prediction：
- Level 1: short-term, low-abstraction prediction
- Level 2: longer-term, higher-abstraction prediction
- ...

**关键 inductive bias**：高层 encoder 自然会 ignore 难以 long-term predict 的 details，所以高层 representation 更 abstract，可以做 longer-horizon prediction。

### 5.3 Hierarchical Planning (Section 4.7, Figure 16)

这是 LeCun 觉得最难但也最有意思的部分。规划流程：

1. **Top-level objective**: $C(s_2[4])$ — 高层 cost 定义 final goal
2. **Top-level optimization**: 推断 abstract action sequence $(a_2[2], a_2[4])$ 使得 $C(s_2[4])$ 最小
3. **Subgoal propagation**: $(a_2[2], a_2[4])$ 作为 condition 喂给 lower-level cost modules $C(s[2]), C(s[4])$
4. **Lower-level planning**: lower-level 推断 action sequence 满足 subgoal costs
5. **Action execution**: 最底层输出 actual motor commands

LeCun 在 paper 中特别点出：**"actions" at high level 不是 real motor actions，是 condition on lower-level states**。这其实呼应了 classical control theory 中 setpoint servo 的 idea —— 你给 lower level 一个 target，lower level 自己 figure out 怎么达到。

但是 LeCun 也承认这里有一个 unsolved question：**how does the configurator learn to decompose a complex task into a sequence of subgoals**? 这是整个 architecture 中最 under-specified 的部分。

参考相关工作：
- Director (Hafner et al. 2022): https://arxiv.org/abs/2206.04114 — hierarchical world model with RL
- Trajectory Transformer (Janner et al. 2021): https://arxiv.org/abs/2106.01338 — 把整个 trajectory 喂给 transformer

---

## 6. Handling Uncertainty (Section 4.8, Figure 17)

### 6.1 Sources of Uncertainty

LeCun 列了 7 种 uncertainty 来源，分成 aleatoric 和 epistemic：

| Type | Source |
|------|--------|
| Aleatoric 1 | World intrinsically stochastic |
| Aleatoric 2 | World deterministic but chaotic |
| Aleatoric 3 | World deterministic but partially observable |
| Epistemic 1 | Sensors give partial info |
| Epistemic 2 | Perception module loses info |
| Epistemic 3 | World model has limited capacity |
| Epistemic 4 | World model under-trained |

### 6.2 Latent Variable Sampling

每个 prediction step都 sample 一个 latent $z$，从 regularizer-defined distribution：

$$P(z) = \frac{\exp(-R(z))}{\int_{z'} \exp(-R(z'))} \tag{Gibbs distribution}$$

每个 sample 给出一条不同的 prediction trajectory。如果 $z$ 是 discrete with $k$ values，total trajectories = $k^t$，exponential blow-up。需要 MCTS-style 的 pruning & directed search。

最终 action selection 不只 minimize expected cost，也 minimize **variance** of cost（risk-averse planning）：

$$a^* = \arg\min_a \left[ \mathbb{E}_{z}[C(s_T)] + \beta \cdot \text{Var}_{z}[C(s_T)] \right]$$

这是 **risk-sensitive optimal control** 的 standard formulation，与 Tamar et al. 2016 等工作呼应。

### 6.3 Ego Model vs World Model

LeCun 提出一个有意思的 architectural choice：**separate ego model from world model**。理由是 agent 自己的 action 效果相对 deterministic（我抬手，手就抬起来），不需要 latent variable。但 external world 是 chaotic 的，需要 latent。这对应 Sobal et al. 2022 的工作：https://arxiv.org/abs/2204.07184

---

## 7. Short-term Memory as Key-Value Store (Section 4.9)

LeCun 提出 world state 应该存在 key-value memory 里，因为大多数 action 只改变 state 的一小部分。Memory operations：

**Read**:
$$\text{Mem}(q) = \sum_j c_j v_j \tag{15}$$
$$\tilde{c}_j = \text{Match}(k_j, q) \tag{16}$$
$$c = \text{Normalize}(\tilde{c}) \tag{17}$$

**Write**:
$$v_j = \text{Update}(r, v_j, c_j) \tag{20}$$

**Symbol**：
- $k_j, v_j$：第 $j$ 个 memory slot 的 key 和 value
- $q$：query vector
- $c_j$：attention weight（softmax-style）
- $r$：要写入的新 value
- $\text{Update}(r, v, c) = cr + (1-c)v$：interpolation

**Intuition**：每个 memory slot 对应一个 entity（bottle, kitchen, dining room）。当事件发生（bottle 从 kitchen 移到 dining room），只 update 相关 slots，其他不变。这远比把整个 state 通过 RNN hidden state 传递 efficient。

参考：
- Memory Networks (Miller et al. 2016): https://arxiv.org/abs/1606.03163
- Entity Networks (Henaff et al. 2017): https://arxiv.org/abs/1612.03969
- Neural Turing Machine: https://arxiv.org/abs/1410.5401

---

## 8. Amortized Inference for Latent Variables (Appendix)

直接做 $\check{z} = \arg\min_z E_w(x, y, z)$ 需要迭代 optimization，expensive。Amortized inference 训一个 encoder $\tilde{z} = Enc(s_x, y)$ 直接 predict $\check{z}$。

$$\min_{Enc} H(\check{z}, \tilde{z})$$

但这里有 **cheating risk**：encoder 看到 $y$，可以直接把 $y$ 的全部 info 通过 $z$ 传到 predictor，导致 collapse。所以 **regularizer $R(z)$ 在 amortized inference 里更 critical**，必须严格 limit $z$ 的 info content。

VAE 的 inference network $q_\phi(z|x)$ 就是 amortized inference 的特例，KL term 就是 $R(z)$。LISTA (Gregor & LeCun 2010) 的 encoder 也是 amortized sparse coding。

---

## 9. 与其他 Paradigm 的对比 (Section 8.3)

### 9.1 "Scaling is Not Enough"

LeCun 直接 attack LLM-scaling school。两个 main arguments：

**Argument 1**: LLM 是 tokenized generative model，token 化对 text 友好，但对 continuous high-dim signal（video）不友好。在 video prediction 里，无法用 categorical distribution over pixels 来 represent uncertainty。需要 JEPA 这种在 representation space 做 prediction 的 architecture。

**Argument 2**: LLM 缺少 abstract latent variables，所以无法做 multi-interpretation reasoning 或 goal-directed planning。LLM 的 "reasoning" 是 next-token autoregression，没有真正的 search & planning loop。

这与 LeCun 后来在 Twitter/X 上对 GPT-4, Sora 等的批评一脉相承。例如他批评 Sora 缺少 world model 时，本质上就是基于这篇 paper 的 argument：Sora 是 generative pixel prediction，cannot reliably predict long-horizon physical consistency。

参考：
- LeCun's debate with Yann on Twitter: https://twitter.com/ylecun/status/1761439909608247568
- LeCun's talk "Path Towards Autonomous Machine Intelligence" at Meta AI: https://www.youtube.com/watch?v=8NYNt69XFE8

### 9.2 "Reward is Not Enough"

LeCun 反驳 Silver et al. 2021 的 "Reward is Enough" (https://arxiv.org/abs/2106.04593)。他的 argument：

- RL 的 scalar reward 是 low-bandwidth feedback
- Model-free RL 的 sample inefficiency 源于此
- 在 LeCun 的 architecture 里，**绝大部分 parameters 在 world model 里，通过 self-supervised prediction 训练**，reward 只在 cost module 里占很小一部分

这与 Karl Friston 的 **Active Inference** / Free Energy Principle 有 strong resonance：agent 通过 minimize prediction error / surprise 学习，而不是通过 reward。Friston 的工作：https://arxiv.org/abs/2205.11523

### 9.3 "Do We Need Symbols?"

LeCun 反对 Marcus & Davis 2019 (https://mitpress.mit.edu/9780262043045/rebooting-ai/) 的 view that deep learning needs symbolic reasoning bolted on。他的 argument：reasoning = energy minimization / constraint satisfaction，可以是 fully differentiable 的。Symbolic reasoning 只是一种特殊 case，当 action space discrete 时可以用 MCTS / dynamic programming。

---

## 10. 后续工作 & Empirical Evidence

Meta 在 paper 之后陆续推出了一系列实证工作试图 validate JEPA idea：

1. **I-JEPA** (Assran et al. 2023, CVPR 2024): https://arxiv.org/abs/2301.08243 — 图像 SSL with JEPA，show that non-contrastive JEPA 可以 match or beat Masked Autoencoder on ImageNet linear probe and semi-supervised
2. **V-JEPA** (Bardes et al. 2024): https://arxiv.org/abs/2301.08243 — video version，predict representations of future frames
3. **V-JEPA 2** (2025): 最近的工作，扩展到 action-conditioned video prediction 和 robot learning

不过 LeCun 也承认 (Section 8.1)，整个 architecture 的 full instantiation 还没人做出来。**Hierarchical planning with H-JEPA 仍然是一个 open problem**。

相关工作 link：
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-video-model/
- Meta's V-JEPA 2: https://ai.meta.com/blog/v-jepa-2/

---

## 11. 我自己的 commentary & open questions

读这篇 paper 时，我有几个 thoughts：

**1. JEPA 的 "non-generative" 立场很激进**

整个 ML community 几乎都在押注 generative model（diffusion, autoregressive, VAE），LeCun 这一篇是 rare voice arguing for non-generative predictive architecture。如果 JEPA 的 representation 真的能 capture physical common sense，它应该能 beat MAE / SimMIM / DINO 在 downstream physics reasoning tasks 上。I-JEPA 在 linear probe 上确实 competitive，但还看不到 clear win on physics-grounded tasks。

**2. Hierarchical planning 部分严重 under-specified**

Section 4.7 的 Figure 16 看起来很 elegant，但 configurator 怎么 decompose task into subgoals？这其实是个 chicken-and-egg problem：你需要一个 planner 来 decompose，但 decompose 本身需要 planning。LLM 在这里可能反而是 missing piece —— LLM 作为 high-level subgoal proposer，H-JEPA 作为 low-level world model。这是 SayCan 等工作 (https://saycan.github.io/) 的思路，但 LeCun 在 paper 里没有讨论这个 hybrid。

**3. 与 Active Inference 的关系值得 explore**

Friston 的 Free Energy Principle 其实是 JEPA 的 theoretical physics 版本。Active inference agent 也是通过 minimize predictive energy 来 act，也有 hierarchical model (HM-Active Inference)。LeCun paper 里 cite 的 prediction-related cognitive science work 没怎么 cite Friston，但这两个 framework 应该可以 unify。Smith et al. 2022 的 "A step towards understanding the neural substrates of mental illness..." 等工作在尝试 build bridge。

参考：https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(09)00031-0

**4. Latent variable 的 regularization 仍然是 art**

Discrete / sparse / low-dim / stochastic 四种方法各有 tradeoff，没有 clear winner。在实际 robot learning 里，discrete latent 在 multi-modal action distribution 上 work，但 loses fine-grained info。Continuous + KL 容易 collapse。这部分需要更多 empirical work。

---

## 12. Final Summary

这篇 paper 的 contribution 可以总结为四点：

1. **A complete cognitive architecture**，全 differentiable，6 modules + Mode-1/Mode-2 distinction
2. **JEPA / H-JEPA**，non-generative predictive world model that operates in representation space, hierarchical for multi-time-scale planning
3. **Non-contrastive SSL paradigm** (VICReg-style) that avoids curse of dimensionality
4. **Hierarchical MPC** under uncertainty via latent variable sampling

它是一个 **position paper** 而非 technical paper。LeCun 自己也明说 "many of these ideas have been formulated by many authors in various contexts" — 他的 contribution 是 assembly into a consistent whole。

对于 build intuition，我建议这样 mental model：**JEPA 是一个不生成 y 但能预测 y 的 representation 的 energy-based model**。它通过 encoder invariance 和 latent variable 两条路径 handle multi-modality，通过 VICReg-style regularization 避免 collapse。H-JEPA 是把单层 JEPA stack 起来，高层 abstract 长 horizon，低层 detailed short horizon。Planning 用 differentiable MPC，gradient 从 cost 通过 world model backprop 到 action。

最值得 follow 的工作：
- Meta FAIR 后续的 I-JEPA, V-JEPA 系列
- Karl Friston 的 Active Inference 文献
- Hafner 的 Dreamer V3 和 Director
- Sobal et al. 的 ego/world model 分离工作

Hope this helps build intuition! 这篇 paper 的野心很大，但 instantiate 起来每一步都是 PhD-level problem。如果真要 bet，我觉得 JEPA-as-SSL 会先 work（已经 work），H-JEPA + hierarchical planning 是更 long-term 的 bet。
