---
source_pdf: Improving Robotic Generalist Policies via.pdf
paper_sha256: 96acb4801b8dbe0d4ceb412172e541dbbf47e05a28cc51cd3934e0e7b7a88049
processed_at: '2026-08-19T16:05:16-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FRS

Andrej，我换个讲法，抛开公式，讲点直觉。

---

## 一句话版本

你有一个很厉害的 robot generalist policy（比如 $\pi_{0.5}$），它见过很多东西，脑子里有很多 "reasonable behavior"。但你给它一个新 task，它可能就是做不对——**not because 它不会，而是因为它不知道在这个新场景下该调出哪段记忆**。

FRS 就是个 trick：你给它一个非常粗糙的 hint（"往左走"），它把这个 hint 反向塞进 policy 的 noise space，再正向跑一遍，出来的 action 就自动变成 **policy 自己会做的那种 fine-grained 动作**，而且方向还跟你的 hint 大致一致。

---

## 一个直觉类比

想象一个超厉害的厨师，他脑子里存了几千道菜的做法。你跟他说 "做点酸的"，他可能愣住——因为有太多可能性，他不知道你想吃哪种酸的。但如果你跟他说 "做个番茄味的汤"，他立马就能调出番茄汤的 recipe，做出一道非常 fine-grained 的好汤。

FRS 做的就是这件事：把一个粗糙的语义信号（"番茄味" + "汤"）当作 key，去 query 厨师脑子里的 recipe bank，找到最近的 matching recipe，然后让厨师按那个 recipe 做。

厨师不是被 fine-tune 了，厨师还是那个厨师，只是你帮它做了一次 **retrieval**。

---

## 为什么 Flow Matching 可以这么干

这是整个 paper 最 cool 的 insight。

Flow matching policy 的本质是一套 ODE：从 noise $a_0$ 积分到 action $a_1$。因为它是 deterministic 的（跟 DDPM 的 stochastic chain 不同），所以这个 mapping 是可逆的——给定一个 action，你可以倒着积分回去，找到一个 noise，使得正向积分能产生这个 action。

但是——这里有个 subtle 的点——倒着积分会有 integration error。你用 10 个 Euler steps 倒着跑，再正着跑回来，得到的 action 跟原始 reference 不完全一样。它会被"吸"到 policy 学过的 nearby action mode 上。

这个"吸"是 feature，不是 bug。因为你给 VLM 的粗糙 direction 本身就不是好的 robot action——它是 OOD 的。你希望它被拉回 policy 的 in-distribution action mode 上，同时保留"方向大致对"这个信息。

这就像：你给一张糊掉的草图让 GAN 生成图片，GAN 不会还原你的草图，它会生成一张跟草图方向一致但细节真实的图。FRS 是 robot action 版本的这件事。

---

## 为什么直接让 VLM 输出 action 不行

因为 VLM 擅长 semantic reasoning（"sponge 用来擦桌子"），但它不擅长输出 dexterous 的 7-DOF joint trajectory。你让它直接输出 action，精度太差。

VLA 擅长输出 fine-grained action，但它不知道在这个新场景下该做啥——它会被它的 prior 拉到某个 "reasonable" mode 上，但不一定是 task 对的那个 mode。

所以 division of labor 是：**VLM 负责 high-level semantic reasoning，VLA 负责 low-level action generation，FRS 是中间的 interface**。VLM 说 "go left and pick up the sponge"，FRS 把这个粗糙指令翻译成 VLA 会说的一种方言（noise space），VLA 听懂了，输出它最擅长的那种精细动作。

---

## 三个用法，从简单到复杂

### 用法一：Zero-shot 在线 steering

最直接的方式。每一步，VLM 看一眼画面，告诉你 "往左"，你把这个 direction 转成粗糙 action chunk，通过 FRS 投影成 fine-grained action，执行。

**问题**：每步都要 query VLM，慢、贵。

### 用法二：DSBC（用 BC 把 FRS 的成功 distill 成小 policy）

FRS 执行过程中，你不仅得到了 action $\hat{a}_1$，还得到了对应的 noise $\hat{a}_0$。这个 noise 是"如果给 VLA 这个 noise，它会 denoise 出一个好的 action"。

所以你可以 collect FRS 成功的 $(o, \hat{a}_0)$ pair，训一个小的 noise policy：输入 observation，输出应该给 VLA 什么 noise。

这个 noise policy 极小（MLP 3 层），训练 1 分钟，1 GB memory，只需要 10 条成功 trajectory。

**为什么这比 standard BC 好**：standard BC 直接学 action，10 条 trajectory 学不出来，因为 action space 太高维，10 个数据点根本 cover 不住。但 noise policy 只需要 predict 一个 7D 向量，而且有 VLA 做 "safety net"——noise policy 进 OOD state 时输出可能烂，但 VLA 会把烂 noise 当作普通 noise prior 处理，denoise 出一个 "reasonable" action。这是 standard BC 没有的 implicit fallback 机制。

### 用法三：DSRL + FRS（用 FRS 跳过 RL 的冷启动）

如果 zero-shot FRS 还不够好，继续跑 RL 改进。但 RL 冷启动很慢，因为开始时 noise policy 是随机的，几乎碰不到好 action mode。

FRS 的成功 trajectory 提供了 prior data。你把它塞进 replay buffer，再加个 BC auxiliary loss 把 noise policy 拴在 FRS noise 附近。这样 RL 从一个有 semantic meaning 的起点开始 explore，而不是从纯随机开始。

---

## 整个 pipeline 的 elegant 之处

1. **VLM 不会的事**：输出 fine-grained action → 不让它干
2. **VLA 不会的事**：在新场景下自动选对 behavioral mode → 不让它单独干
3. **FRS 干的事**：把 VLM 的粗糙信号翻译成 VLA noise space 的好 noise，让 VLA 输出它最擅长的 action

这是一种典型的 **divide and conquer** 思路。让每个 component 干自己擅长的，不要 push 它干不擅长的。VLM 不输出 action，VLA 不做 long-horizon reasoning。

---

## 这个 trick 在图像领域其实老早有了

图像 editing 领域，DDIM inversion、null-text inversion、Prompt-to-Prompt 这些工作早就把 "把 image 反向积分成 noise，再正向生成" 这套玩透了，用来做 image editing。

FRS paper 的 contribution 是：**把这套 idea 搬到 robot policy 上，而且发现 integration error 的 "拉回 in-distribution mode" 效果反而对 steering 有用**。图像领域追求的是精确重构再编辑，robot 领域追求的是把 OOD signal 拉到 in-distribution mode，目标不同，但 mechanism 同源。

reference: https://arxiv.org/abs/2301.02204 (null-text inversion)

---

## 我觉得最 elegant 的 design decision

**用 10 个 integration steps 而不是更多**。

这看起来是个 hyperparameter 选择，但背后是 deep insight。如果你用 100 steps，reconstruction 会很准——$\hat{a}_1 \approx a_1$。但这意味着 $\hat{a}_0$ magnitude 大，是 OOD noise。你不想让 noise policy 学 OOD noise，你想让它学 in-distribution noise。

10 steps 刚好让 reconstruction "过得去"（方向大致对），但 noise magnitude 还算小（接近 $\mathcal{N}(0, I)$）。这是个 tradeoff sweet spot：保留 semantic direction 的同时，noise 本身可被 noise policy 容易学出来。

---

## 还有个 surprising 的发现

paper appendix 做了个实验：在 FRS 输出的 noise $\hat{a}_0$ 上再加 $\mathcal{N}(0, I)$ 的 noise（scale $\sigma = 1$ 甚至 2），再 denoise，performance 不降反升。

这意味着什么？FRS 找到的不是单个好 noise，而是 **noise space 里的好 region**。附近的所有 noise 都 denoise 出不错的 action。

这给 noise policy 训练极大的 robustness——你不需要精确预测那个 noise，只需要预测到附近就行。VLA 的 flow 会把附近的 noise 都拉到同一个好 mode 上。这是为什么 DSBC 只用 10 条 trajectory 就能 work 的根本原因。

---

## 什么情况下 FRS 会 fail

如果场景里压根没有 reasonable action mode（比如 task 完全 OOD，VLA 从来没见过类似的东西），FRS 没东西可以拉，再怎么 reverse flow 也 project 不到好的 action 上。

如果 VLM 给的方向完全错（应该往左却给往右），FRS 会执行往右的合理动作，但 task 还是 fail。

所以 FRS 的 power 上界是 base VLA 的 prior coverage。它只做 retrieval，不做 extrapolation。要 extrapolation 还是要 RL。

---

## 对未来的 hint

FRS 暗示了一种新的 foundation model 使用范式：**不要 fine-tune foundation model，而是 train 一个小的 adapter policy 在它的 latent space 上操作**。这个 adapter policy 可以是 noise policy（DSBC/DSRL），可以是 value head（用于 ranking），可以是任何东西。foundation model 保持 frozen，所有 task-specific 知识都 encode 在小 policy 里。

这跟 LoRA 有点像，但 LoRA 是 weight space adapter，FRS 是 noise space adapter。因为 flow matching 的 noise space 有 structure（nearby noise → nearby action mode），这个 adapter 不需要很大就能 work。

future direction 我觉得最 promising 的是：把这个 idea 用到 **video generation model 当 world model** 的场景上。Mimic-VWM、GR00t 这类工作都在把 video generation 当 policy 用，video model 也是 flow matching 的，FRS 应该直接适用——VLM 给粗略 visual trajectory，FRS 投影到 video model 的 mode 上得到 fine-grained video，再 decode 成 robot action。这会是一个很自然的 extension。

---

希望这个版本更 intuitive 一点。核心就一句话：**用 flow reversibility 把粗糙信号 free 地投影到 foundation policy 的 in-distribution mode 上，省掉 trial-and-error**。

---

# Flow Reversal Steering (FRS) 技术深度解析

Andrej 你好！这篇paper非常有意思，让我来详细拆解。这是 Physical Intelligence 团队后续工作 $\pi_{0.5}$ 的延展，核心 idea 是把 flow matching policy 的 ODE 反着积分一遍，就能把 "粗略的 reference action" 投影到 generalist policy 的 in-distribution action mode 上。

paper 链接：https://arxiv.org/abs/2506.15799（DSRL，FRS 的 RL 基础）
project page：https://flow-reversal-steering.github.io/
$\pi_{0.5}$ base policy：https://arxiv.org/abs/2410.24164
$\pi_0$ flow matching VLA：https://arxiv.org/abs/2410.24164

---

## 1. 核心问题与动机

**Generalist policy 的困境**：像 $\pi_{0.5}$、OpenVLA、GR00t N1 这类 VLA 在大规模数据上训练后，确实有 "reasonable behavior" 的 prior——比如见过 sponge、bowl、towel 就隐含了 wiping/counting/cleaning 的潜能。但是这个 prior 是**隐式**的，policy 自己采样的时候未必会落在正确的 mode 上，尤其在 OOD scene 下，base policy 可能几乎全失败（LIBERO-90 里有 62 个 task success rate ≤ 40%）。

**已有 steering 方法的局限**：
- **Classifier-free guidance / diffusion guidance**（https://arxiv.org/abs/2207.12570）：要求有 discriminator，对新 task 不灵活。
- **Partial noising**（Yoneda et al. 2023, Wang et al. 2024）：把 reference action 部分加噪再 denoise，但对 noise scale 很敏感，t 太小没效果，t 太大信号全丢。
- **Sample-and-rank**：parallel sample 多个 action，post-hoc scoring 选最好的。问题是当 base policy 对正确 action 的概率密度极低时，怎么 sample 都 sample 不到。
- **DSRL**（https://arxiv.org/abs/2506.15799）：把 noise $a_0$ 当作 action，跑 RL 找好 noise。但是冷启动时随机 sample noise 几乎碰不到好 action mode，RL explore 极慢。

FRS 想解决的核心问题是：**如何把 VLM/human 提供的 semantic coarse guidance（比如 "往左移"），不经过 expensive trial-and-error，直接 ground 到 generalist policy 的 fine-grained action mode 上**。

---

## 2. Flow Matching VLA 的数学背景

先把 flow matching policy 的数学建好，因为后续 FRS 完全是基于这套 ODE 的可逆性。

### 2.1 Forward Flow (Denoising)

给定 BC dataset $\{(o, a_1)\}$，sample 时间 $t \in [0, 1]$，构造 partially noised action：

$$a_t = t \cdot a_1 + (1-t) \cdot a_0, \quad a_0 \sim \mathcal{N}(0, I)$$

- $a_1 \in \mathbb{R}^d$：clean action（专家示范）
- $a_0 \in \mathbb{R}^d$：纯高斯噪声
- $t \in [0, 1]$：flow time，t=0 时纯噪声，t=1 时纯 action
- 这个 interpolation是 linear 的，跟 DDPM 的 forward diffusion 不一样，**没有 stochastic noise injection**，这点很关键

学习 velocity field $v_\theta(a_t, t | o)$ 满足 ODE：

$$\mathrm{d}a_t = v_\theta(a_t, t | o) \, \mathrm{d}t$$

训练 loss 就是 conditional flow matching 的 regression：

$$\mathcal{L}_\theta = \mathbb{E}_{o, a_1, a_0, t} \left[ \| v_\theta(a_t, t | o) - (a_1 - a_0) \|^2 \right]$$

- target $(a_1 - a_0)$ 是从 $a_0$ 到 $a_1$ 的常速度向量
- 这个形式比 DDPM 的 $\epsilon$-prediction 简洁很多

### 2.2 Euler Integration for Sampling

部署时从 $a_0 \sim \mathcal{N}(0, I)$ 出发，用 Euler integration 离散积分得到 action：

$$a_{t+h} \leftarrow a_t + v_\theta(a_t, t | o) \cdot h, \quad t \in \{0, h, \dots, 1-h\}$$

记作 $a_1 \leftarrow \mu_\theta(a_0, o)$。$\pi_{0.5}$ 默认用 $1/h = 10$ steps（$h = 0.1$）。

### 2.3 关键性质：Deterministic Map

Flow matching 的核心特性是 $\mu_\theta: a_0 \mapsto a_1$ 是**确定性映射**（给定 observation $o$）。这跟 DDPM 的 stochastic Markov chain 不同，DDPM 是不可逆的。Flow 是 ODE，原则上可逆。FRS 就是 exploit 这个 reversibility。

---

## 3. Flow Reversal：核心方法

### 3.1 反向 Euler Integration

把上面的 forward Euler integration 反着写：

$$a_{t-h} \leftarrow a_t - v_\theta(a_t, t | o) \cdot h, \quad t \in \{1, 1-h, \dots, h\}$$

- 起点：$a_1$ = reference action（可能是粗糙的，比如 VLM 给的方向向量）
- 终点：$\hat{a}_0$ = 计算出来的"对应 noise"
- 把这个反向过程记作 $\hat{a}_0 \leftarrow \mu_\theta^{-1}(a_1, o)$，hat 表示这是 computed noise，不是 sampled noise

然后再 forward 一次：

$$\hat{a}_1 \leftarrow \mu_\theta(\hat{a}_0, o)$$

当 $h \to 0$ 时，理论上 $\hat{a}_1 = a_1$（精确重构）。但是有限 steps 下有 integration error，所以 $\hat{a}_1 \approx a_1$ 但不完全相等。

### 3.2 为什么这能 work？Intuition

这是 paper 最微妙的点。paper 里 Appendix C 做了详细分析，核心发现：

**Observation 1: OOD action 的 noise magnitude 大，in-distribution action 的 noise magnitude 小**

LIBERO-90 数据上的实验（Fig. 14）：
- Ground truth action（OOD for $\pi_{0.5}$-LIBERO，因为没在 90 上训）→ $\|\hat{a}_0\|_2$ 较大
- $\pi_{0.5}$ sampled action（in-distribution）→ $\|\hat{a}_0\|_2$ 较小
- 与 $\mathcal{N}(0, I)$ 的 $\chi$ 分布对比，in-dist noise 更接近 0

直觉：flow 学到的是把 noise 映射到 data manifold 的 mapping。Data manifold 上的点对应小 noise，远离 manifold 的点对应大 noise。这跟 DDIM inversion 在图像编辑中的应用（https://arxiv.org/abs/2301.02204 null-text inversion）是同源的 idea。

**Observation 2: Integration steps 的 tradeoff**

| Integration steps | Reconstruction MSE | Noise magnitude |
|---|---|---|
| 5 steps | 高 | 低 |
| 10 steps | 中 | 中 |
| 50 steps | 低 | 高 |
| 100 steps | 最低 | 最高 |

steps 越多，integration error 越小，reconstruction 越准。但 paper 发现 reconstruction 越准反而不好——因为这意味着 $\hat{a}_0$ 越来越 OOD（magnitude 大），学到的是 "off-manifold noise that happens to map to reference"。

paper 选 $h = 0.1$（10 steps）作为 sweet spot：reconstruction 还过得去，但 noise magnitude 仍然比较 in-distribution。

**Observation 3: FRS 把 OOD action 投影到 nearby in-distribution mode**

理想化实验（Appendix B.1，4 个 mode 的 mixture of Gaussians）：
- 取原点附近的点（OOD）
- 通过 FRS 反向再正向
- 得到的 $\hat{a}_1$ 被推向最近的 mode，log-likelihood 提升
- 但是 displacement（移动距离）受 steps 数控制：steps 少 → 移动多、log-likelihood 提升多；steps 多 → 移动少、log-likelihood 提升少

这跟 SDEdit（https://arxiv.org/abs/2108.01050）的 spirit 有点像，但机制不同。SDEdit 是 forward 加噪破坏信息再 denoise；FRS 是 reverse 积分找出 latent noise 再 denoise。Fig. 4 对比了两者：forward diffusion 用 interpolation $a_t = t a_1 + (1-t) a_0$，t→0 时信号全没；reverse flow integration 是 deterministic 的 data-to-noise map，保留信息。

### 3.3 关键 ablation：Adding Noise to FRS Noise

Appendix C.3 里有个 surprising 的实验：在 $\hat{a}_0$ 上加 Gaussian noise $\sigma \epsilon$ 再 denoise：

$$\hat{a}_1 \leftarrow \mu_\theta(\hat{a}_0 + \sigma \epsilon, o), \quad \epsilon \sim \mathcal{N}(0, I)$$

发现 $\sigma = 1$ 甚至 $\sigma = 2$ 反而**提升** performance。这说明 FRS 找到的不是单个好 noise，而是**好 noise 的 region**。附近的 noise 都 map 到好的 action mode。这给 DSBC 和 DSRL 训练提供了 robustness——不需要精确预测 noise，只要落在好 region 就行。

---

## 4. FRS 的三个应用范式

### 4.1 Zero-shot Online Steering

**Pipeline**：
1. VLM (Gemini-ER-1.6) 或 human 看 observation，输出 coarse direction（"move +x", "move -z"）
2. Program 把 direction 转成 reference action chunk $a_1$（用 IK 或直接 Cartesian delta）
3. Flow reversal: $\hat{a}_0 \leftarrow \mu_\theta^{-1}(a_1, o)$
4. Denoise: $\hat{a}_1 \leftarrow \mu_\theta(\hat{a}_0, o)$
5. Execute $\hat{a}_1$

每一步都 query reasoner，所以叫 "online"。

**Reference action 的设计**：
- VLM 输出 3D Cartesian direction vector + "more"/"less" 表示 magnitude
- 把 direction normalize 到 [-1, 1]（用 LIBERO 的 normalization stats）
- Rotation 设 0，gripper 设中间值（不强制，让 policy 决定）
- Padding 维度（$\pi_{0.5}$ 输出 32 维，LIBERO 只用 7 维）设成 ground truth padding value，然后对应的 noise 维度设成 $\mathcal{N}(0, I)$，相当于 inpainting

**为什么 VLM 的 coarse action 直接执行不行**：因为 VLM 缺乏 fine-grained spatial reasoning，给的 direction 精度太低。直接执行这些 directional action 在 LIBERO 上 overall 降低 performance。但作为 reference 给 FRS 就够了——FRS 把它"投影"到 VLA 的 fine-grained action mode 上，既保留了 semantic direction，又获得了 dexterity。

### 4.2 DSBC: Diffusion Steering via Behavioral Cloning

Online steering 需要每步都 query reasoner，开销大。能不能把 FRS 的成功轨迹 distill 成一个小 policy？

**Insight**：FRS 不仅输出 action $\hat{a}_1$，还输出对应的 noise $\hat{a}_0$。这个 $\hat{a}_0$ 是 "expert noise"——把它 denoise 就能得到 good action。所以可以训练一个**noise policy** $\pi_\phi^{noise}(\hat{a}_0 | o)$，输入 observation，输出应该用哪个 noise 给 VLA。

$$\pi_\phi^{noise} \leftarrow \arg\max_\pi \mathbb{E}_{o, a_0} [\log \pi(a_0 | o)]$$

部署时：sample $\hat{a}_0 \sim \pi_\phi^{noise}(\cdot | o)$，再 $\hat{a}_1 \leftarrow \mu_\theta(\hat{a}_0, o)$。

**两种 data 来源**：
1. **Online DSBC**：用 zero-shot FRS 的成功 rollout，collect $(o, \hat{a}_0)$ pair
2. **Offline DSBC**：用现有 demonstration dataset $(o, a_1)$，flow reversal 得到 $\hat{a}_0$，再训练。不需要 online execution 验证

**为什么 DSBC 比标准 BC 好**：这是 paper 里最美的 insight。标准 BC 在 small data regime 下会 compounding error——policy 进入 OOD state 后不知道怎么办。DSBC 不同：

- noise policy 在 OOD state 输出的 noise 可能也烂
- 但是 VLA $\mu_\theta$ 把这个 noise 当作普通的 noise prior 处理
- VLA 的 flow 会把这个 noise denoise 到 "reasonable" action mode
- 相当于 DSBC 隐式地 "fall back" 到 VLA 的行为先验

这就是为什么在 real-world DROID 上，standard BC 用 10 trajectories 完全失败，DSBC 用 10 trajectories 涨 60%。

**实现细节**：
- Noise policy 是个小 MLP $(128, 128, 128)$
- 输出 averaged noise（across chunk axis），因为 Appendix C.4 发现 noise 在 chunk 维度上 variance 低
- Tanh clipping 范围 $[-5, 5]$（比 DSRL 原版 $[-1, 1]$ 宽，更 expressive）
- Loss 用 NLL 或 MSE 都行
- 训练 < 1 minute，1 GB GPU memory（不需要加载 VLA）
- 数据：LIBERO 平均 18 rollouts/task，real-world 10 rollouts/task

### 4.3 DSRL + FRS: Bootstrapping RL

当 zero-shot FRS 还不够好（比如 LIBERO 一些 task base policy 几乎全失败），用 RL 进一步改进。

**核心 loss**：

$$\pi_\phi^{noise} \leftarrow \arg\max_\pi \underbrace{\mathbb{E}_{o \sim \mathfrak{B}, a_0 \sim \pi(\cdot|o)} [Q^{noise}(o, a_0)]}_{\text{RL objective}} + \lambda \underbrace{\mathbb{E}_{(o, a_0) \sim \mathfrak{D}^+} [\log \pi(a_0 | o)]}_{\text{BC auxiliary objective}}$$

- $\mathfrak{B}$：replay buffer，包含 FRS prefill + online rollout
- $\mathfrak{D}^+$：FRS 成功轨迹的 $(o, \hat{a}_0)$ pair
- $\lambda$：BC weight（实验中 = 1）
- $Q^{noise}(o, a_0)$：noise-space Q function，由 SAC 训练

**两个关键改动**：
1. **Prefill replay buffer with FRS trajectories**：解决 RL 冷启动问题。没有 FRS 时，DSRL 在 base policy 几乎全失败的 task 上 explore 极慢，因为 sample 的 noise 随机，几乎碰不到 good action mode。FRS 提供了"专家级"noise 作为 prior data。
2. **Auxiliary BC loss**：soft behavior constraint，让 noise policy 不要离 FRS 的好 noise 太远。这是 offline-to-online RL 的标准技巧（https://arxiv.org/abs/2106.06863 AWAC 类似思想）。

**两个实验 setting**：
- **15 tasks where FRS works well**：用 20 个 FRS rollout（含 success 和 failure）prefill，跑 200k steps SAC
- **10 tasks where base policy ≈ 0% and FRS ≈ 8%**：只用 1 个 FRS success（要跑 50 次才得到一次 success！）warmstart，证明 even minimal FRS signal 能 jumpstart RL

**对比 baseline**：
- Standard DSRL（无 FRS prefill）
- Residual RL（akin to PLD，https://arxiv.org/abs/2511.00091）
- RoboMeter as VLM reward model（https://arxiv.org/abs/2603.02115）

DSRL + FRS 在两个 setting 都最好。在 hard 10-task setting，标准 DSRL 只到 30%，DSRL + FRS 显著更高。

---

## 5. 实验数据深度解读

### 5.1 LIBERO Simulation

| Setting | Tasks | Base Policy | Key Result |
|---|---|---|---|
| Zero-shot VLM FRS | Spatial/Object/Goal + 62 hard | $\pi_{0.5}$-LIBERO (not trained on 90) | 11 个 base ≤ 2% 的 task 提升 ≥ 10% |
| DSBC | 15 tasks (FRS ≥ 10% boost) | $\pi_{0.5}$-LIBERO | matches zero-shot FRS，超 standard BC |
| DSRL + FRS (easy) | 15 tasks | $\pi_{0.5}$-LIBERO | 超 DSRL 和 residual RL |
| DSRL + FRS (hard) | 10 tasks (base ≈ 0%) | $\pi_{0.5}$ fine-tuned only on 90 by Jain et al. | 超 DSRL，超 RoboMeter reward |

特别值得注意：在 62 个 hard task 上，base policy 在 11 个 task 上 success rate ≤ 2%（50 trial 里只有 0 或 1 次成功）。这种 task 之前 RL 根本无从下手——没有 reward signal。FRS 通过 VLM steering 能拿到一些 success，给 RL 提供了最早的 reward signal。

### 5.2 Real-World DROID

6 个 task，base policy 是 $\pi_{0.5}$-DROID：

| Method | Avg Success Rate |
|---|---|
| Base $\pi_{0.5}$ VLA | 低（多个 task 接近 0） |
| Standard BC (flow matching) w/ 10 FRS rollouts | 完全失败 |
| Zero-shot FRS w/ human | 中等 |
| **DSBC w/ 10 FRS rollouts** | **+60% absolute boost** |

Towel hanging task：base 5% → DSBC 50% → DSRL+FRS 80%。完整的 zero-shot → BC → RL pipeline 在一个 real task 上跑通。

### 5.3 Training Cost

- DSBC: < 1 minute training, 1 GB GPU memory, 10-18 trajectories
- 不需要 fine-tune full VLA（需要 hundreds of GBs）
- Small noise policy: 3-layer MLP, 7D output

---

## 6. 设计决策的 Intuition 总结

我把 paper 里散落的设计 decision 整理一下，每一个都有明确的实验支撑：

1. **10 integration steps ($h = 0.1$)**：tradeoff between reconstruction fidelity and noise in-distributionness。少 steps → noise 更靠近 $\mathcal{N}(0,I)$ 但 reconstruction 差；多 steps → reconstruction 好但 noise OOD。10 是 OpenPi 默认，也是 sweet spot。

2. **Noise averaging across chunk axis**：Appendix C.4 发现 FRS 产生的 noise 在 chunk 维度上 variance 低（temporal correlation），所以直接 average & repeat 就行。这让 noise policy 只需要 predict 一个 7D vector，不用 predict 完整 chunk。

3. **Padding noise 设为 $\mathcal{N}(0, I)$**：$\pi_{0.5}$ 输出 32 维但 LIBERO 只用 7 维，剩下是 padding。Flow reversal 算完 noise 后，把 padding 维度的 noise 替换成 fresh Gaussian sample，不影响 denoise 结果。这也缩小了 noise policy 的输出空间。

4. **Tanh clipping $[-5, 5]$**：比 DSRL 原版的 $[-1, 1]$ 宽，让 noise policy 能 express 更 OOD 的 reference action（比如 VLM 给的纯方向向量）。

5. **Reference action 用简单 cardinal direction**：Appendix C.3 显示用 oracle policy action 比 VLM cardinal action 好（50% vs 33.7% on hard split），但 VLM cardinal 已经足够 boost base policy，且更 scalable。

6. **VLM 只看 third-person camera + plumb line annotation**：LIBERO 没 wrist camera depth 信息，VLM 不擅长 fuse 多视角。加一条从 gripper 到桌面的 plumb line 帮 VLM 判断 spatial relationship。

---

## 7. 与相关工作的 positioning

### 7.1 Flow/Diffusion Inversion 家族

图像领域已经成熟：
- **DDIM Inversion**（https://arxiv.org/abs/2010.02502 Song et al.）：把 DDIM sampler 反着跑，找到 latent noise 对应 image
- **Null-text Inversion**（https://arxiv.org/abs/2301.02204 Mokady et al.）：精确 inversion 用于 image editing
- **Prompt-to-Prompt**（https://arxiv.org/abs/2208.01626 Hertz et al.）：cross-attention control
- **Rectified Flow inversion**（https://arxiv.org/abs/2410.10792 Rout et al.）：flow model 版本
- **FireFlow**（https://arxiv.org/abs/2412.07517 Deng et al.）：fast rectified flow inversion

FRS 是把这套图像 inversion 的 idea 搬到 robot action space，但用法不同：图像编辑是精确重构再局部 edit，FRS 是故意用 integration error 把 OOD action 投影到 nearby in-distribution mode。

### 7.2 Robotic Policy Steering

- **Diffusion Policy**（https://arxiv.org/abs/2303.04137 Chi et al.）：第一个把 diffusion 用在 robot policy
- **Diffusion For Shared Autonomy**（https://arxiv.org/abs/2302.12244 Yoneda et al.）：partial noising 做 human steering，FRS paper 显示这 baseline 不如 FRS
- **Inference-time Policy Steering**（https://arxiv.org/abs/2402.19104 Wang et al.）：human interaction steering
- **DSRL**（https://arxiv.org/abs/2506.15799 Wagenmaker et al.）：noise-space RL，FRS 的 RL 基础
- **UniSteer**（https://arxiv.org/abs/2605.10821 concurrent work）：concurrent 工作也用 flow reversal invert human action，做 noise-space RL。FRS 区别在于 (1) refine coarse action admit VLM guidance, (2) BC without RL, (3) bootstrap RL with non-human guidance

### 7.3 VLA + VLM 协同

- **PIVOT**（https://arxiv.org/abs/2402.07872 Nasiriany et al.）：visual prompting 让 VLM 直接输出 action
- **MOKA**（https://arxiv.org/abs/2406.09238 Liu et al.）：mark-based visual prompting
- **Code as Policies**（https://arxiv.org/abs/2209.07753 Liang et al.）：LLM 生成 code 调用 primitive
- **VoxPos**（https://arxiv.org/abs/2307.05973 Huang et al.）：3D value map

这些方法的局限：VLM 直接输出 action 太 coarse。FRS 把 action generation 的 onus 完全交给 VLA，VLM 只做 high-level semantic reasoning，分工明确。

### 7.4 VLA RL 改进

- **$\pi^*_{0.6}$**（https://arxiv.org/abs/2511.14759 Physical Intelligence）：VLA 通过 experience learning
- **RLDG**（https://arxiv.org/abs/2412.09858 Xu et al.）：robot generalist policy distillation via RL
- **PLD / Residual RL for VLA**（https://arxiv.org/abs/2511.00091 Xiao et al.）：residual policy 改 VLA
- **Policy Agnostic RL**（https://arxiv.org/abs/2412.06685 Mark et al.）：offline+online RL fine-tune

这些方法都没有用 semantic knowledge bootstrapping，纯靠 reward signal。FRS 的差异点是引入 VLM/human 的 semantic prior 来 jumpstart。

---

## 8. 我对 FRS intuition 的总结

如果让我用一句话概括 FRS 的 essence：

> **Flow matching policy 的 ODE 是可逆的。把任意 reference action 反着积分得到的 noise，再用 policy 正向 denoise，由于 integration error 的存在，结果会被"吸"到最近的 in-distribution action mode 上。这给了一个把 coarse guidance "投影" 到 VLA behavioral prior 的免费机制。**

这跟以下经典 idea 有 resonance：

1. **SDEdit**（https://arxiv.org/abs/2108.01050）：forward 加噪破坏信号再 denoise，让输出 in-distribution。但 SDEdit 的 forward 是 stochastic 的，破坏程度由 $t$ 控制；FRS 的 reverse 是 deterministic 的，"破坏"程度由 integration steps 控制，机制不同。

2. **Classifier-free Guidance**（https://arxiv.org/abs/2207.12570）：用 conditional 和 unconditional score 的差做 guidance。FRS 不需要单独训 unconditional model，直接用 flow reversal 实现 "guided sampling"。

3. **Retrieval-augmented generation**：把 query 投影到 latent space 再 decode。FRS 有点像 retrieval——把 coarse action "retrieve" 到 VLA 学过的 nearest behavior mode。

4. **Inpainting**：把已知的 action chunk 部分 "inpainted" 到 noise space，未知部分由 VLA fill in。Appendix D 提到 padding noise 设 $\mathcal{N}(0,I)$ 就是 inpainting 思路。

5. **Manifold projection**：FRS 本质是把 OOD point 投影到 learned data manifold 上。这跟 VAE 的 latent regularization、autoencoder 的 denoising 机制都有关联。

---

## 9. Limitations 和 Future Directions

Paper 自己承认的局限：

1. **Reference action 还是要相对 reasonable**：如果 VLM 给的 direction 完全 wrong direction，FRS 也救不了。FRS 只是把 coarse action 投影到 nearby mode，如果附近没好 mode，没用。

2. **VLM 系统 engineering 粗糙**：用的是 cardinal direction + 简单 prompt，更复杂的 VLM 系统（chain-of-thought 更深、更好的 spatial reasoning）应该能进一步提升 zero-shot performance。

3. **Integration steps 是 hyperparameter**：理想上应该 adaptive，但 paper 用固定 10 steps。

4. **只在 manipulation 上验证**：locomotion、navigation、bimanual 等其他场景还没试。

5. **Real-world RL 只在一个 task (towel hanging) 上 demo**：完整的 real-world DSRL+FRS pipeline 还需要更多验证。

我自己的额外联想：

- **Action chunk 之间的 temporal coherence**：Appendix C.4 发现 noise 在 chunk 维度 variance 低，这暗示 VLA 学到的 noise manifold 是 low-rank 的。这跟 Pi-zero 论文里提到的 action chunk 的 temporal smoothness 是一回事。能否设计更高效的 noise parameterization？比如用一个 low-rank 加 sinusoidal basis 来 represent noise？

- **Multi-modal steering**：当前 FRS 是 single-modal——给定 reference，输出一个 mode。如果 scene 有 multiple reasonable mode（比如桌上有 3 个 bowl 都可以拿），FRS 会随机选一个附近的。能否让 FRS explicit branch 出 multiple hypothesis，由 VLM post-hoc 选？

- **Hierarchical FRS**：高层次 VLM 给 long-horizon plan（"先拿 sponge，再擦桌子，再放回"），每个 sub-goal 用 FRS ground 到 action。这跟 ReAct、Tree of Thoughts 思路类似。

- **Active learning**：FRS 失败的 rollout 也有价值——可以告诉 VLM "你刚才那个 direction 不对"，让 VLM 在线学习。这跟 Human-in-the-loop RL（https://arxiv.org/abs/2410.21845 HIL-SERL）结合应该很强。

- **Noise space 的 geometry**：Appendix C 的发现（OOD action 对应大 noise magnitude，in-dist 对应小 noise magnitude）暗示 noise space 有内在几何结构。能否 explicit learn 一个 noise manifold，用 Riemannian geometry 工具分析？

- **与 World Model 结合**：近期 World Action Model（https://arxiv.org/abs/2602.15922 Mimic-VWM）把 video generation 当 policy。Video model 也是 flow matching/diffusion，FRS 应该直接适用——用 VLM 给粗略 visual trajectory，FRS 投影到 video model 的 mode 上得到 fine-grained video，再 decode 成 action。

- **与 Test-time Scaling 结合**：RoboMonkey（https://arxiv.org/abs/2506.17811）做 VLA 的 test-time sampling scaling。FRS 提供了 cheap way to generate candidate actions（不同 reference → 不同 steered action），可以 sample 多个 reference 方向，每个 FRS 出一组 action，再 verify。这比纯 random sample 更 efficient。

---

## 10. 公式变量含义速查表

为方便 build intuition，我把所有公式里出现的 symbol 整理一下：

| Symbol | 含义 | 维度 |
|---|---|---|
| $o$ | observation（image + state） | 复杂 |
| $a_1$ | clean action（专家示范或 reference action） | $\mathbb{R}^d$（LIBERO $d=7$, DROID $d=8$, $\pi_{0.5}$ full $d=32$） |
| $a_0$ | 纯高斯噪声 | $\mathbb{R}^d$ |
| $a_t$ | partially noised action，$a_t = t a_1 + (1-t) a_0$ | $\mathbb{R}^d$ |
| $t$ | flow time，$t \in [0, 1]$，t=0 纯噪声，t=1 纯 action | scalar |
| $h$ | Euler integration step size，$h = 0.1$ default | scalar |
| $v_\theta(a_t, t \| o)$ | learned velocity field，预测 $\mathrm{d}a_t / \mathrm{d}t$ | $\mathbb{R}^d$ |
| $\mu_\theta(a_0, o)$ | 完整 denoise 过程，从 $a_0$ 到 $a_1$ | $\mathbb{R}^d \to \mathbb{R}^d$ |
| $\mu_\theta^{-1}(a_1, o)$ | flow reversal，从 $a_1$ 到 $\hat{a}_0$ | $\mathbb{R}^d \to \mathbb{R}^d$ |
| $\hat{a}_0$ | computed noise（hat 表示非 sampled） | $\mathbb{R}^d$ |
| $\hat{a}_1$ | reconstructed action after FRS | $\mathbb{R}^d$ |
| $\pi_\phi^{noise}(a_0 \| o)$ | noise policy，DSBC/DSRL 训练的小 policy | parameter $\phi$ |
| $Q^{noise}(o, a_0)$ | noise-space Q function | $\mathbb{R}$ |
| $\mathfrak{B}$ | DSRL replay buffer | dataset |
| $\mathfrak{D}^+$ | FRS 成功轨迹的 $(o, \hat{a}_0)$ pair | dataset |
| $\lambda$ | BC auxiliary loss weight | scalar，= 1 in experiments |
| $\beta$ | SAC actor KL penalty 或 BC loss weight | scalar |
| $\sigma$ | additional noise added to $\hat{a}_0$ for ablation | scalar |

---

## 11. 架构图解析（Fig. 2）

Paper 的 Fig. 2 展示了完整 pipeline：

1. **Left block - Reasoner**：Human 或 VLM 看 task，输出 coarse reference action（比如 "+x direction"）。Reasoner 不需要输出精确 action，只输出大致方向。

2. **Middle block - FRS**：
   - Input: coarse reference action $a_1$
   - Flow reversal: 反向 Euler 积分 $v_\theta$，得到 $\hat{a}_0$
   - Flow denoising: 正向 Euler 积分，得到 $\hat{a}_1$
   - $\hat{a}_1$ 是 in-distribution for VLA，但保留 reference 的 semantic direction

3. **Right block - Three usage modes**：
   - **Zero-shot FRS**（Sec 5.2）：直接执行 $\hat{a}_1$
   - **DSBC**（Sec 5.3）：collect $(o, \hat{a}_0)$ 做 supervised learning 训练 noise policy
   - **DSRL + FRS**（Sec 5.4）：$(o, \hat{a}_0)$ 作为 prior data + auxiliary BC loss，加上 SAC 训 noise policy

整个系统的 beauty 在于：**reasoner 只需要做 high-level semantic reasoning，VLA 只需要做 low-level dexterous action generation，FRS 是两者之间的 bridge**。这种 division of labor 比让 VLM 直接输出 action（PIVOT, Code as Policies）或让 VLA 自己长程 reason（embodied CoT）都更 scalable。

---

希望这个深度解析能 build your intuition，Andrej！如果想更深入讨论某个 aspect（比如 noise space geometry、flow reversal 的 integration error 分析、或 VLA + RL 的未来方向），可以继续聊。
