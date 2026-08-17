---
source_pdf: VLA-JEPA Enhancing Vision-Language-Action Model with Latent World Model.pdf
paper_sha256: 5deec841d5a7da63e5bab3593798e1741d44d354507586bec8a2f9b576f12677
processed_at: '2026-08-13T02:46:18-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 VLA-JEPA

## 一句话总结

现在大家都在用互联网视频去预训练机器人策略，但主流方法学偏了 —— 它们以为自己在学"动作"，其实学的是"画面怎么变"。VLA-JEPA 说：你们别去预测像素了，去预测未来画面的抽象表示，而且绝对不能让模型偷看未来帧。就这么简单一个改动，效果炸了。

---

## 到底哪里出了问题？

想象你在看一个人切菜的视频。你想从里面学"动作"。但视频里变化的东西太多了：

- 切菜的手在动（这个跟动作有关）
- 相机角度在晃（跟动作无关，但变化很大）
- 背景里有人在走动（跟动作无关，但变化很大）
- 光线忽明忽暗（跟动作无关，但变化很大）

主流方法（LAPA、UniVLA 这些）的做法是：把"下一帧"和"当前帧"做差，把这个差值压缩成一个 latent action。听起来挺合理对吧？但你想想，模型在优化这个目标的时候，它会去编码什么？

它会去编码那些**变化最显著的东西**。而在真实视频里，变化最显著的往往是相机抖动、背景乱动、光照变化 —— 恰恰是那些跟你真正想控制的动作完全无关的东西。结果你的 latent action 变成了一个"画面差异编码器"，而不是"动作语义编码器"。

这就像你想学开车，但老师只告诉你"下一秒画面会变成什么样"。你确实能预测画面，但你学到的可能是"路边的树会往后退"，而不是"方向盘转了多少度"。

---

## 更要命的问题：信息泄露

这个更 subtle。很多方法在训练时，把当前帧和未来帧**同时**喂给模型。模型要预测"给定当前帧和某个 latent action，未来会怎样"。听起来没问题对吧？

但你想，模型如果聪明的话，它会发现一个 shortcut：与其去理解"latent action 怎么影响未来"，不如直接把未来帧的信息偷偷塞进 latent action 里。这样 latent action 就直接编码了"未来长什么样"，预测起来零误差，loss 降得飞快。

但这个 latent action 在语义上是空的。它对训练有用，对控制没用。因为推理的时候你根本不知道未来 —— 你没有未来帧可以偷。

---

## VLA-JEPA 怎么解决的？

核心 idea 两点：

**第一，别在像素空间预测，在抽象空间预测。**

用一个 frozen 的 encoder 把视频帧编码成抽象表示，然后在这个抽象空间里预测未来。为什么这个有用？因为 encoder 已经帮你把"相机抖动、光照变化、背景杂乱"这些 low-level 噪声滤掉了。剩下的抽象表示里，"动"的东西基本就是跟任务相关的 state 变化。模型预测这个，就自然学到 dynamics。

这就像让模型预测"菜有没有被切开"这个抽象状态，而不是预测"每个像素变成什么颜色"。前者逼模型理解任务语义，后者让模型去纠结光照纹理。

**第二，绝对不让模型偷看未来帧。**

训练时，VLM 只看到当前帧。未来帧只用来构造监督目标 —— 通过 frozen encoder 编码成一个 target representation，然后让 world model 的预测去对齐这个 target。未来信息从来不是输入，永远是监督信号。

这样模型就没法作弊了。它必须真正理解"当前状态 + 我假设的动作 → 未来状态"这个因果结构，而不是把未来直接拷贝过来。

---

## 整个 pipeline 是什么样？

很简单，就两步：

**预训练阶段**：拿大量人类视频和少量机器人数据，训一个 latent world model。模型看当前帧，输出一堆 latent action tokens，这些 tokens 去驱动 world model 预测未来的抽象状态。监督信号是 frozen encoder 编码的真实未来状态。同时机器人数据上还加一个 flow matching 的 action prediction loss。

**微调阶段**：在具体任务数据上 fine-tune，让 action head 学会输出精确的 end-effector 轨迹。

对比之前 LAPA 那套三阶段 pipeline（预训练 representation → 学 latent action 并对齐 → 学 policy），VLA-JEPA 直接一步到位，简单很多。

---

## 为什么用 JEPA？

JEPA 是 Yann LeCun 一直推的架构。核心哲学：**智能的本质是 world model —— 你得能在脑子里想象"如果我做 X，世界会变成什么样"**。

但 LeCun 认为，你不该在像素层面做这个想象。像素层面太 noisy、太细节、太多不可控因素。你应该在 abstract representation 层面做想象。

VLA-JEPA 就是把这个哲学塞进 VLA。latent action tokens 本质上是模型对"会发生什么"的假设，world model 学的是"给定当前状态和我对动作的假设，下一个抽象状态是什么"。这跟人脑想象未来的方式更像 —— 你不会在脑子里渲染每个像素，你想象的是"门会被推开"这种抽象概念。

---

## 实验告诉我们什么？

几个关键 takeaways：

**LIBERO（标准 benchmark）**：VLA-JEPA 平均 97.2%，跟 OpenVLA-OFT、π₀.5 差不多，但用的训练数据少很多。说明把 latent action 学对，比堆数据更有效。

**LIBERO-Plus（压力测试）**：这个最重要。7 种扰动维度（相机、光照、背景、语言等等），VLA-JEPA 在 5 种上 SOTA，平均 79.5%。第二名 OpenVLA-OFT 才 69.6%。这说明 JEPA 学到的 representation 真的更 robust，因为它在抽象空间学，天然对 appearance 变化免疫。

**人类视频到底有没有用？** 这是最有意思的发现。在标准任务上，去掉人类视频影响不大甚至更好（SimplerEnv Google Robot：78.4% vs 65.2%）。但在扰动任务上，人类视频贡献巨大（79.5% vs 62.9%，差了 16.6%）。

为什么？因为人类视频里没有机器人的动作轨迹信息（人手和机械爪的 kinematics 完全不同），所以没法直接学"怎么执行动作"。但人类视频里充满了"什么时候该重试""失败后怎么恢复""物体的 affordance"这类 meta-skill。这些 skill 不依赖具体 embodiment，能跨形态迁移。

**真实世界实验**：VLA-JEPA 会"重复抓取" —— 抓失败后自动张开爪子重试。π₀ 和 π₀.5 都不会，因为它们的训练数据里没有这种 demo。VLA-JEPA 从人类视频里学到了"抓失败要重试"这个 temporal decision，然后内部 map 到自己的物理动力学去执行。这是 cross-embodiment transfer 的一个很漂亮例子。

---

## 一句话直觉

**你选什么样的预测目标，就学到什么样的表示。** 预测像素，学到像素模式；预测抽象状态（且不让偷看），学到动力学语义。这是 representation learning 的普适原则，VLA-JEPA 只是把它正确地应用到机器人学习上。

整个工作最打动我的是它的 simplicity —— 没有花哨的多阶段对齐，没有复杂的 codebook 设计，就是"在抽象空间预测 + 不让偷看"。但这个简单设计背后是对 latent action 退化问题的深刻诊断。在 VLA 越来越复杂的当下，这种"把核心目标做对"的工作很 refreshing。

---

# VLA-JEPA 深度讲解

这篇paper 是2026年2月来自USTC、SJTU、Tsinghua等团队的工作，核心thesis非常清楚：**现有 latent-action pretraining 方法学错了东西**，它们仍锚定在 pixel variation 上，而不是 action-relevant state transition semantics。VLA-JEPA 用 JEPA-style 的 leakage-free state prediction 来 sidestep 这些问题。

---

## 1. 为什么 latent action pretraining 常常学错东西 —— 四个 failure modes

这是 paper 最有价值的部分，作者花了大量篇幅做诊断。我逐个拆解：

### Failure Mode 1: Pixel-level objectives bias toward appearance

很多 latent-action 方法（LAPA、UniVLA、IGOR、MotoGPT等）本质上是用 "future" 作监督，要么直接预测 future pixels，要么把 frame-to-frame 变化压缩成一个 latent variable 当作 action。即便用 VQ-VAE 做离散化压缩，supervision signal 仍然被 **what changes visually** 主导 —— texture、illumination、background clutter、viewpoint。这些 factors 的特征是 **high-variance but low-control**：容易预测，但和 policy 真正要 master 的 controllable DoF 关系很弱。

直觉上：你在 minimize 一个 pixel reconstruction / prediction loss，模型找到的最 easy path 是去 encode 那些变化最大、最可预测的东西 —— 而这些往往是 nuisance，不是 action。

### Failure Mode 2: Real-world videos amplify noisy motion

在 human videos 和 in-the-wild footage 上，camera motion 和 non-causal background changes 可能比 interaction-induced state changes 还要强。基于 frame-difference 的 latent-action objectives 因此被 incentive 去 encode 这些 dominant signals，结果 latent action 变成了 **delta-frame encoder of nuisance motion**。

这点非常关键：你在 internet video 上 pretrain，video 里 "动" 的东西大部分不是 "agent 控制" 的东西。一个手持摄像机的 vlogger 走路，画面里 90% 的 motion 是相机自身运动。

### Failure Mode 3: Information leakage → latent action collapses into shortcut

这是最 subtle 也最致命的问题。很多 pipeline（LAPA、UniVLA）在 modeling transitions 时，把 current observation 和 future observation 都 feed 进同一个 module，或者允许 future context 影响 learned action variable。这创造了一个 **easy shortcut**：latent action 可以直接 encode future 本身，而不是去 capture "state transition 应该被如何解释"。

结果：latent action 在语义上变成 empty —— 对 training loss 有用，但对 control 没有意义。这个观察和 Zhang et al. 2025 的 "What do latent action models actually learn?" [82] 一致，他们系统性地揭示了 latent action 的退化问题。

参考：[What do latent action models actually learn?](https://arxiv.org/abs/2506.15691)

### Failure Mode 4: Multi-stage pipelines are fragile

为了 stabilize training，LAPA、Villa-X、XR-1 这些方法用了三阶段甚至更多：representation pretraining → latent action learning/alignment → policy learning。每阶段都有自己的 objective、hyperparameters，stage 之间容易 inconsistency，evaluation 也变难。

---

## 2. JEPA 的核心 insight —— 为什么这是正确的解法

Yann LeCun 的 JEPA (Joint-Embedding Predictive Architecture) 的核心 idea 是：**不在 pixel space 预测，而在 latent space 对齐**。

直觉是这样的：如果让模型 predict future pixels，模型会被 low-level details（光照、纹理、相机抖动）劫持；如果让模型 predict future 的 *representation*，并且这个 representation 来自一个 frozen target encoder，模型就被 forced 去 learn "什么样的 high-level state 会发生"，而 appearance nuisance 在 representation 层面已经被 encoder 滤掉了。

JEPA 的另一个关键是 **asymmetric design**：target encoder 是 frozen 的（stop-gradient），predictor 只能看到 context，不能看到 target 的 input。这样 predictor 没法 cheat —— 它必须真正学 dynamics。

参考：
- [V-JEPA: Latent video prediction for visual representation learning](https://arxiv.org/abs/2304.08471)
- [V-JEPA 2: Self-supervised video models enable understanding, prediction and planning](https://arxiv.org/abs/2506.09985)
- [I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture](https://arxiv.org/abs/2301.08243)

---

## 3. VLA-JEPA 的架构（Figure 1, Figure 2）

让我把架构拆解成几个组件：

### 3.1 VLM Backbone

采用 **Qwen3-VL [3]** 作为核心 VLM，基于 Qwen3 [77] + SigLIP-2 [70] vision encoder。VLM 在大规模 pretraining 时已经 acquire 了 world knowledge（image understanding、object detection），这些可以被 transfer 到 robot control。

关键设计：在 vocabulary 里加入两个 special learnable tokens：
- `⟨latent_i⟩` —— 表示第 $i$ 个时间步的 latent action，encode $s_i$ 到 $s_{i+1}$ 之间的 state transition
- `⟨action⟩` —— embodied action token，用于生成 robot action

### 3.2 World State Encoder

用 **V-JEPA2** [2] 作为单视角 video encoder $F(\cdot)$。多视角通过 concatenation 聚合：

$$s_{t_i} = \Vert_v F(I_{v, t_i})$$

变量解释：
- $s_{t_i}$ —— 时间步 $t_i$ 的 unified world state representation
- $I_{v, t_i}$ —— 视角 $v$ 在 $t_i$ 时刻的 image frame
- $F(\cdot)$ —— V-JEPA2 encoder（frozen）
- $\Vert_v$ —— 跨视角的 concatenation operator

这个设计让 VLA-JEPA 能利用 multi-view data，但又不依赖 single-view 的全部信息。

### 3.3 Latent World Model（time-causal attention）

这是 paper 最漂亮的部分。Latent action tokens 从 VLM 出来后，condition 一个 auto-regressive Transformer world model。

Latent action 通过 VLM 产生：

$$z_{t_i} = p_\theta^{VLM}(\langle \text{latent}_i \rangle \mid \{I_{j, t_0}\}_{j=0}^v, \ell)$$

变量解释：
- $z_{t_i}$ —— 第 $i$ 个 latent action token 在 $t_i$ 时刻的 latent representation
- $p_\theta^{VLM}$ —— VLM 模型（参数 $\theta$）
- $\langle \text{latent}_i \rangle$ —— learnable special token，在 input sequence 里被 replicate $K$ 次
- $\{I_{j, t_0}\}_{j=0}^v$ —— 初始时刻 $t_0$ 的多视角 image observations
- $\ell$ —— language instruction

注意：**VLM 只看到 $t_0$ 的 observation**，绝不看到 future frames。这就是 "leakage-free" 的关键。

然后 world model 预测 future states：

$$\hat{s}_{t_{1:i+1}} = p_\theta^{WM}(s_{t_{0:i}}, z_{t_{0:i}})$$

变量解释：
- $\hat{s}_{t_{1:i+1}}$ —— 预测的 $[t_1, t_{i+1}]$ 时间窗内的 world state chunk
- $p_\theta^{WM}$ —— world model（参数 $\theta$）
- $s_{t_{0:i}}$ —— ground-truth 的历史 world states（来自 frozen target encoder）
- $z_{t_{0:i}}$ —— VLM 产生的 latent action representations

**Attention 机制**是 time-causal：
- **Within 同一时间步**：所有 latent action tokens 和 world state tokens 之间 bidirectional full attention
- **Across 时间步**：严格 causal —— 时间步 $t$ 的 tokens 只能 attend 到 $\leq t$ 的 tokens，future 被 mask 掉

这个设计非常关键：它既保证了 single-step 内的 bidirectional context（state 和 action 互相影响），又保证了 temporal causality（防止 future 信息泄露到 past）。

### 3.4 训练目标 —— ELBO 视角下的 JEPA loss

Paper 用一个 ELBO 推导来 frame 这个 objective。给定 frozen target encoder $F(\cdot)$ 产生 target world states $s_{t_i}$，world model $p_\theta^{WM}$ conditioned on $z_{t_i}$ 预测 $\hat{s}_{t_i}$，objective 是：

$$\log p(s_{t_{1:T}} \mid z_{t_{0:T-1}}) \geq \sum_{k=1}^T \mathbb{E}_{s_{t_k} \sim F(\cdot)} [\log p_\theta(\hat{s}_{t_k} \mid s_{t_k})] - D_{KL}[F(\cdot) \| p_\theta^{WM}]$$

变量解释：
- 左边：given latent actions $z_{t_{0:T-1}}$，future states $s_{t_{1:T}}$ 的 log-likelihood
- 右边：ELBO 分两项
  - 第一项：reconstruction term，对每个 $t_k$，在 $F(\cdot)$ 产生的 $s_{t_k}$ 分布上取期望，预测 $\hat{s}_{t_k}$ 的 log-likelihood
  - 第二项：KL divergence，between frozen target encoder $F(\cdot)$ 和 online world model $p_\theta^{WM}$

由于 $F(\cdot)$ 是 deterministic 的（stop-gradient），KL term 退化，ELBO 简化为 latent space 的 reconstruction loss：

$$\mathcal{L}_{WM} = \sum_{k=1}^T \mathbb{E}_{s_{t_k} \sim F(\cdot)} (\hat{s}_{t_k} - s_{t_k})$$

变量解释：
- $s_{t_k}$ —— ground-truth world state at time $t_k$（from frozen encoder）
- $\hat{s}_{t_k}$ —— predicted world state
- $T$ —— video prediction horizon

注意这个 loss 形式简单，但内涵深刻：模型在 **latent space** 做 L2 预测，不是 pixel space。

---

## 4. Action Prediction with Flow Matching

### 4.1 Action Token Conditioning

对于 robot data，latent action tokens 学好后，作为 conditioning signal 指导 embodied action 生成。在 latent action tokens 之后 append `⟨action⟩` tokens，利用 VLM 的 causal attention 捕捉 `⟨action⟩`、latent action tokens、initial image、language instruction 之间的依赖。

$$z_a = p_\theta^{VLM}(\langle \text{action} \rangle \mid \{I_{i, t_0}\}_{i=0}^v, \ell, \langle \text{latent}_i \rangle)$$

变量解释：
- $z_a$ —— global action-conditioning representation
- $p_\theta^{VLM}$ —— VLM 模型
- $\langle \text{action} \rangle$ —— learnable embodied action token
- $\{I_{i, t_0}\}_{i=0}^v$ —— 多视角初始 observations
- $\ell$ —— language instruction
- $\langle \text{latent}_i \rangle$ —— 之前产生的 latent action tokens

### 4.2 Conditional Flow-Matching Action Head

这部分借鉴了 π₀ [9] 的设计。Flow matching 用于 model 一个 distribution over continuous action trajectories。

定义 time-dependent interpolation：

$$a_t = (1-t)\epsilon + t a_{0:H}, \quad t \sim \mathcal{U}(0,1)$$

变量解释：
- $a_t$ —— interpolation point 在 time $t$
- $t$ —— flow time，uniform 分布在 $[0,1]$
- $\epsilon \sim \mathcal{N}(0, I)$ —— Gaussian noise（起点）
- $a_{0:H}$ —— ground-truth action sequence over horizon $H$（终点）

Action head 参数化一个 vector field $v_\theta(a_t, t \mid z_a)$，conditioned on $z_a$，训练去 match ground-truth conditional flow。Flow matching objective：

$$\mathcal{L}_{FM} = \mathbb{E}_{a_{0:H}, \epsilon, t} \left[ \| v_\theta(a_t, t \mid z_a) - (a_{0:H} - \epsilon) \|_2^2 \right]$$

变量解释：
- $v_\theta(\cdot)$ —— 模型预测的 velocity field
- $(a_{0:H} - \epsilon)$ —— linear interpolation 诱导的 target velocity
- $\|\cdot\|_2$ —— L2 norm
- 期望 over $a_{0:H}$（ground truth actions）、$\epsilon$（noise）、$t$（flow time）

直觉：flow matching 是 diffusion 的连续版本。它学一个 vector field，把 noise distribution "flow" 到 data distribution。Inference 时从 $\epsilon$ 出发，沿 $v_\theta$ 积分得到 $\hat{a}_{0:H}$。

参考：
- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

### 4.3 整体 objective for robot data

$$\mathcal{L} = \mathcal{L}_{FM} + \beta \mathcal{L}_{WM}$$

变量解释：
- $\mathcal{L}_{FM}$ —— flow matching action prediction loss
- $\mathcal{L}_{WM}$ —— world model alignment loss
- $\beta$ —— tunable hyperparameter，控制 WM loss 的权重

注意：robot data 上同时优化两个 objective，让 state-transition dynamics 能被 leverage 用于 control。

---

## 5. 实验数据深度解读

### 5.1 LIBERO (Table 1)

VLA-JEPA 在 4 个 task suite 中 2 个 SOTA，平均 97.2% 成功率，超过 OpenVLA-OFT (97.1%) 和 π₀.5 (96.9%)。

关键观察：
- Top methods (OpenVLA-OFT、π₀.5) 都依赖 extensive robot datasets 做 pretraining
- VLA-JEPA 用 less training data 达到更好 performance
- 之前的 latent-action methods (UniVLA、Villa-X、LAPA、CoT-VLA) 都不如 VLA-JEPA

消融 "w/o human videos"：96.1% vs 97.2%，说明 human video pretraining 在 LIBERO 上贡献 marginal。这其实是一个重要的 negative result —— 在 ID (in-distribution) scenario，高质量 expert demos 比 human video 更关键。

### 5.2 SimplerEnv (Table 2)

Google Robot 上 VLA-JEPA avg 65.2%，WidowX 上 57.3%。

特别值得分析：
- LAPA 在 WidowX 上 57.3%，但用了 successful rollouts only（100 rollouts）—— 因为 train on successful rollouts 有效 mitigate real-to-sim gap
- Villa-X 用了大规模 robot data + human videos，在 WidowX 上达到 SOTA 40.8%... 但等等，paper 里 Villa-X 的 WidowX avg 是 40.8%，VLA-JEPA 是 57.3%，所以 VLA-JEPA 实际超过 Villa-X
- VLA-JEPA、UniVLA、RoboVLMs、Moto 都用了 < 1% 的 Villa-X training data，但 VLA-JEPA 最 competitive

消融 "w/o human videos" 在 Google Robot 上 78.4% > 65.2%（with human videos）。这看起来 counterintuitive，但印证了 Q1 的分析：human videos 对 ID / real-to-sim 场景不直接帮助，因为它们 lack action trajectory 信息。

### 5.3 LIBERO-Plus (Table 3) —— 最重要的 benchmark

LIBERO-Plus 是 stress-test VLA robustness 的 benchmark，7 个 perturbation dimension。VLA-JEPA avg 79.5%，在 5/7 perturbation 上 SOTA。

特别的优势在：
- Language: 85.4% (vs OpenVLA-OFT 79.5%)
- Light: 95.6% (vs OpenVLA-OFT 88.7%)
- Background: 93.6% (vs OpenVLA-OFT 93.3%)
- Layout: 85.1% (vs OpenVLA-OFT 74.2%)
- Robot: 67.1% (vs OpenVLA-OFT 31.9%)

Human video pretraining 这里贡献显著：79.5% (with) vs 62.9% (without)，差了 16.6%。这是 paper 的核心 positive result。

为什么 human video 在 perturbation scenario 帮助大？作者的 hypothesis：human videos lack action trajectory info，但能 enhance robustness 和 stability of pre-existing skills（比如 repeated grasping）。

### 5.4 Real-World (Figure 4)

Franka Research 3 arm + Robotiq 2F-85 gripper，3 个 camera（2 third-person + 1 wrist），100 个 demo（3 picking/placing tasks）。

VLA-JEPA 在 ID 和 object-layout OOD 上 SOTA，task OOD 上 second-best。

最有趣的 qualitative observation：**VLA-JEPA 学会了 repeated grasping** —— 失败后 reopen gripper 重试。π₀ 和 π₀.5 都不会，因为 training data 里没有 repeated grasping demos。VLA-JEPA 从 human videos 里学到了这个 skill。作者 argue：repeated grasping 不需要额外 physical dynamics，只需要学 "when to regrasp" —— 这个 temporal decision 学到后，policy 能内部 map 到自己的 physical dynamics 执行。

---

## 6. Ablation 关键分析

### 6.1 Human video 比例 effect (Figure 5)

LIBERO-Plus 上，随着 human video 比例增加，robustness 在所有 perturbation dimension 上 monotonically 提升。这印证 "human video 增强 robustness 而非引入 new action execution capability"。

### 6.2 Attention 可视化 (Figure 6)

比较 LAPA、UniVLA、VLA-JEPA 的 latent action tokens 对 image tokens 的 attention：
- **LAPA**：attention 过度 dense，包含 operation-irrelevant details（unrelated objects on desktop）。原因：pretraining 时 information leakage，latent action 退化成 target image 的 compressed representation
- **UniVLA**：通过 task-relevant textual guidance 缓解，但 overemphasize semantics，导致 attention 到 operation-irrelevant background（stationary pen in human video、tablecloth texture in real-world wrist view）
- **VLA-JEPA**：更精确地 focus 在 operation 上（robotic arm、hand、要 manipulate 的 objects）

### 6.3 Future video horizon T (Table 4)

$T \in \{4, 8, 16\}$：
- $T=8$ 最好 (avg 96.1%)
- $T=4$：信息不足，long-horizon tasks 表现差
- $T=16$：信息冗余，goal-oriented 任务上最好（简单目标），但 spatial 任务（需要 fine-grained manipulation）上最差

直觉：T 应该和 action horizon 接近。T 太小，dynamics 信息不够；T 太大，引入 redundant 信息干扰 fine-grained control。

---

## 7. 架构细节（Appendix A）

### Latent World Model 配置 (Table 5)

- Transformer layers: 12
- Attention heads: 8
- Image token dim: 2048
- Image tokens per time step: 256
- Action token dim: 2048
- Action tokens per time step: 3
- Number of view: 2
- Future video horizon: 8

Latent action token $\langle \text{latent}_i \rangle$ 被 replicate $K$ 次，$K = 24/T$，$T$ 是 future video horizon。当 $T=8$ 时 $K=3$。

### Action Head 配置 (Table 6)

- DiT-B 架构（Diffusion Transformer）
- Transformer layers: 16
- Attention heads: 12
- Token dim: 1024
- State dim: 8
- Action dim: 7
- Future action horizon: 7
- Denoising timesteps: 4（很少！这是 π₀-Fast 风格的 fast inference）

Action token 被 repeat 32 次作为 conditioning。

### Training details

- Pretraining: SSv2 (220K human videos) + Droid (76K robot demos)
- Batch size: 256 (8 A100 GPUs, batch 32/GPU)
- Peak LR: 1e-5 (VLM + world model), 1e-4 (action head)
- Pretraining: 50K steps
- Simulation fine-tuning: 30K steps
- Real-world fine-tuning: 20K steps

---

## 8. 我的直觉和 broader context

### 8.1 这篇 paper 在 VLA 演化中的位置

VLA 发展脉络：
1. **RT-1/RT-2** [10, 92]：直接在 robot data 上 fine-tune multimodal LLM
2. **OpenVLA, π₀** [39, 9]：大规模 robot data + VLM backbone
3. **Latent action pretraining** (LAPA, UniVLA, Moto, Villa-X) [79, 13, 20, 19]：从 video 学 latent action 再 align 到 real action
4. **VLA-JEPA**：用 JEPA 思想 fix latent action 的退化问题，single-stage training

VLA-JEPA 在第 3 代和第 4 代之间架桥，核心 contribution 是 "把 latent action pretraining 做对"。

参考：
- [RT-1: Robotics Transformer](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [LAPA: Latent Action Pretraining from Videos](https://arxiv.org/abs/2410.11758)

### 8.2 和 LeCun's world model 哲学的一致性

LeCun 长期主张：intelligence 的核心是 world model —— 一个能 predict "如果采取某个 action，world 会怎样变化" 的 internal model。JEPA 是这个哲学的 instantiation：predict abstract representations of future，not pixels。

VLA-JEPA 把这个哲学带入 VLA：latent action tokens 本质上是 "agent 内部对 'what will happen' 的 hypothesis"，world model 学的是 "given current state + latent action hypothesis, predict next abstract state"。这和 LeCun 的 H-JEPA、V-JEPA2 [2] 思路完全一致。

参考：
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun's position paper)

### 8.3 Leakage-free 设计的深度意义

Information leakage 是 latent variable model 的通病。在 VAE 里，posterior collapse 是类似问题。在 contrastive learning 里，negative sampling 不当会 collapse。

VLA-JEPA 的 leakage-free 设计本质上是说：**target encoder 只产生 supervision signal，永远不是 input**。这阻止了 predictor 学一个 identity mapping 的 shortcut。

这个 insight 可以 generalize 到很多场景：任何时候你用 future 信息做监督，要小心 future 是否泄露到了 model input。这个 principle 在 video prediction、world model learning、reward shaping 都适用。

### 8.4 为什么 human video 不直接帮 action execution，但帮 robustness

Paper Q1 的分析很有启发性。Human videos 缺乏 action trajectory（人手和 robot gripper 的 kinematics 完全不同），所以 VLA 没法直接从中学到 "how to execute"。但 human videos 充满了 **temporal structure of behavior**：
- When to regrasp
- How to recover from failure
- Object affordance
- Task-relevant visual features

这些 "meta-skills" 能被 transfer，并在 fine-tuning 时和 robot's own dynamics 结合成 stable policy。

这让我想到 imitation learning 里的 "learning to learn" 思路：先学 high-level structure，再 adapt 到具体 embodiment。

### 8.5 和 Dreamer / world model literature 的关系

VLA-JEPA 的 latent world model 和 Dreamer [Hafner et al.] 思路相似，但有重要区别：
- Dreamer 在 RL 里学 world model，从 experience replay 学
- VLA-JEPA 在 supervised pretraining 阶段学 world model，从 video + robot data 学
- Dreamer 用 world model 做 planning（imagine rollouts）
- VLA-JEPA 用 world model 做 representation learning 的 auxiliary objective

这其实是 "world model as representation learner" vs "world model as planner" 的不同哲学。

参考：
- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)

### 8.6 Flow matching vs Diffusion

VLA-JEPA 用 flow matching 而非 discrete diffusion。Flow matching 的优势：
- 训练更 stable
- Inference 可以用很少的 denoising steps（这里只用 4 steps！）
- 对 continuous action space 自然

π₀ [9] 和 π₀-Fast [61] 都用 flow matching。这是 VLA 社区正在 converge 的设计 choice。

参考：
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [π₀-Fast: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)

### 8.7 Limitations 我能看出的

1. **V-JEPA2 依赖**：target encoder 用 frozen V-JEPA2，可能不是 optimal。如果 V-JEPA2 representation 偏向某些 visual features，会 bottleneck VLA-JEPA。
2. **Multi-view handling 简陋**：只做 concatenation，如果有 >2 views 就选 2 个。Attention-based fusion 可能更好。
3. **Real-world demos 只 100 个**：scale up 时是否 still hold？
4. **Comparison with π₀.5**：在 task OOD 上 π₀.5 更好，说明 text-based reasoning 有价值。VLA-JEPA 没有显式的 language reasoning chain。
5. **Horizon T 的 selection**：T=8 是 empirical optimal，但不同 task 可能需要不同 T。Adaptive horizon 可能更好。

### 8.8 未来方向联想

- **Joint V-JEPA2 training**：现在 V-JEPA2 是 frozen。如果 joint train V-JEPA2 + world model + VLM，可能学出更 action-relevant representations
- **Hierarchical latent actions**：当前 latent action 是 flat sequence。Hierarchical（high-level goal + low-level motor）可能 better for long-horizon tasks
- **Active world model**：world model 当前是 passive predictor。如果让它在 inference 时做 planning（像 MuZero），可能 enable online adaptation
- **Cross-embodiment transfer**：human video 到 Franka arm 是 cross-embodiment。如果加更多 embodiments（human、dog、various robots），能 scale 到 general agent 吗？
- **Language-conditioned world model**：当前 language 只是 VLM input。如果让 world model 显式 attend to language，可能学到 "language → dynamics" 的 causal structure

---

## 9. 关键 reference links

- [VLA-JEPA Project Page](https://ginwind.github.io/VLA-JEPA/)
- [VLA-JEPA Code](https://github.com/ginwind/VLA-JEPA/)
- [VLA-JEPA HuggingFace](https://huggingface.co/ginwind/VLA-JEPA/)
- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [π₀ paper](https://arxiv.org/abs/2410.24164)
- [LAPA paper](https://arxiv.org/abs/2410.11758)
- [UniVLA paper](https://arxiv.org/abs/2505.21672)
- [Villa-X paper](https://arxiv.org/abs/2501.12001)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [SimplerEnv](https://simpler-env.github.io/)
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626)
- [What do latent action models actually learn?](https://arxiv.org/abs/2506.15691)
- [Droid dataset](https://arxiv.org/abs/2403.12945)
- [Something-Something v2](https://developer.qualcomm.com/software/qualcomm-ai-hub/datasets/something-something-v2)

---

## 10. 总结直觉

VLA-JEPA 的核心 message：**要让 latent action 真正有意义，必须 prevent information leakage，且 predict 在 abstract space 而非 pixel space**。

这背后的更深层直觉：**supervision signal 的 nature 决定了 representation 的 nature**。如果你的 supervision 是 pixel difference，你学到 pixel difference encoder；如果你的 supervision 是 latent state prediction（from frozen encoder），你学到 dynamics-aware abstract representation。

这是 representation learning 的普适原则，超越 VLA：**选择什么样的 prediction target，决定了你学到什么样的 representation**。JEPA 的精髓就是用 "abstract representation prediction" 替代 "pixel prediction"，从而逼 model 学 semantic abstraction 而非 surface pattern。

VLA-JEPA 把这个原则应用到 VLA，并且通过 leakage-free 设计避免了 latent action 退化成 future-frame shortcut。结果是一个 simpler（2-stage vs multi-stage）、more robust（在 perturbation benchmark 上 SOTA）、more data-efficient（< 1% of Villa-X data）的 VLA framework。

这是一篇 method design 非常 clean 的 paper，核心 idea 简单（leakage-free JEPA for VLA），执行 well，实验充分。在 VLA 越来越 complex 的当下，这种 "把核心 objective 做对" 的工作值得重视。
