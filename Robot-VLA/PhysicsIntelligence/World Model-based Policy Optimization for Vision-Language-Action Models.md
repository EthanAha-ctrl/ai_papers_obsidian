---
source_pdf: World Model-based Policy Optimization for Vision-Language-Action Models.pdf
paper_sha256: 1f3e88874bcb716ad0417347f9c6a73f1dd70c873e7872d3ceed6c57af2b7598
processed_at: '2026-08-13T05:05:45-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，如果用最直观的大白话来拆解这篇 paper，它讲的其实是一个**“让机器人在‘梦里’闭门造车、自我进化”**的故事。

你知道，现在的 Vision-Language-Action (VLA) 模型基本都在靠 Imitation Learning (IL) 起家，也就是模仿人类示教。这就像是学生只会死记硬背标准答案，一旦考试遇到没见过的题，稍微走偏一点就彻底崩溃，完全没有纠错能力。

为了解决这个问题，Reinforcement Learning (RL) 是个自然的出路，让机器人自己去试错。但在真机上跑 RL，你要收集几百万次物理交互，这既费钱又危险，而且 on-policy RL 需要实时收集数据再更新，真机根本转不起来。

WMPO 的核心 bet 就是：**我们干脆构建一个能以假乱真的 Video World Model，把 VLA policy 关在这个“梦境”里，让它自己在里面靠 imagination 跑 GRPO，完全不用碰真机，等它在梦里练成了，再放出来。**

下面我用大白话结合底层的数学和架构细节，把这个“造梦”和“练功”的过程拆开讲。

---

### 1. 造梦机器：Pixel-space World Model

以前 Dreamer 那一套 model-based RL 喜欢在 abstract latent space 里 rollout，因为快。但 VLA 是在 web-scale 的 image 数据上预训练出来的，它的视觉理解都在 pixel space 里。你如果把它塞进一个 latent dynamics model 里，相当于让一个视觉专家闭着眼睛摸黑干活，prior 全废了。

所以 WMPO 必须在 pixel space 造梦。它的架构基于 OpenSora 的 video diffusion backbone，但做了几个极其关键的手术：

**手术一：换掉 3D VAE，换成 SDXL 的 2D VAE**
3D VAE 为了压缩视频，会在 temporal dimension 上狠压，导致 fine-grained motion details (比如机械臂微小的偏移) 直接糊掉。对于机器人这种毫米级操作，这是致命的。换成 2D VAE 后，只在单帧空间压缩，不压缩时间维，保住了动作细节。预测完之后，再 decode 回 pixel space 喂给 VLA。

**手术二：Frame-level Action Control (AdaLN)**
怎么把 action 信号注入到 video generation 里面？如果把 action token 拼到 image token 里，很容易 attention 混乱。WMPO 用了 AdaLN 的变体，给每一帧配上一个 action，通过 MLP 生成 scale $\gamma_1^i$、shift $\beta_1^i$ 和 residual scale $\alpha_1^i$：
$$
\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_1^i) \cdot \text{Block}\left(\gamma_1^i \cdot \text{LayerNorm}(\mathbf{x}^i) + \beta_1^i\right)
$$
这就像是给每一帧的 transformer block 施加一个由 action 决定的“滤镜偏置”，让模型 frame-wise 听从指挥。

**手术三：Noisy-frame Conditioning (防 drift)**
长视频自回归生成最怕 compounding error，前面一帧稍微有点噪，后面就 snowball 成乱码。WMPO 的 trick 是在训练时，给 conditioning frame 故意加 50/1000 steps 的 diffusion noise。这相当于让 world model 在训练时就习惯“不完美的前置条件”，这样推理时哪怕前面的 frame 有点 drift，它也能 robust 地往后续接，能稳定生成几百帧的长 trajectory。

### 2. 造梦前的心智调整：Policy Behavior Alignment

光有 OpenSora 架构还不行。你的 world model 如果只在 Open X-Embodiment (OXE) 这种全都是成功操作的 expert dataset 上预训练，它学到的只是“完美世界”。但 RL 训练时 policy 会瞎探索，经常走歪路。一旦 policy 走到歪路，world model 没见过这种 state distribution，就会 hallucinate，预测出不真实的 failure dynamics，RL 训练直接崩盘。

为了解决这个 distribution shift，WMPO 用了极其重要的一步：**用 policy 自己在真机上跑出来的少量轨迹 (比如 128 条) 去 fine-tune world model**。
这一步让 world model 适应了 policy 真实的“笨拙行为分布”，学会了怎么 faithful 地模拟 failure mode。没有这一步，RL 根本在 imagination 里站不住脚。

### 3. 梦里的裁判：Lightweight Reward Model

在梦里练功，怎么知道自己练对了？需要个裁判。WMPO 训了一个极简的 Reward Model。
它拿 VideoMAE 加个 linear head，只输出 0 或 1。怎么构造数据？
- Positive: 成功 trajectory 的最后几帧。
- Negative: 成功 trajectory 的中间几帧（防止裁判只学会看开头结尾）+ 失败 trajectory 的任意片段。

它不用复杂的 reward shaping，就是一个 sparse 的 binary outcome。推理时用 sliding window 扫描 trajectory，只要有一段 clip 超过 threshold 就算成功。实验说这玩意儿 F1 score > 0.95，极大压制了 reward hacking。

### 4. 梦里练功：On-Policy GRPO

万事俱备，开始练功。WMPO 选择了 LLM 后时代很火的 GRPO 作为 optimizer。

**为什么选 GRPO？**
GRPO 靠 group-relative advantage 去掉 critic，这对 VLA 这种大模型极其友好。而且它极其适合 world model 的场景！在真机 RL 里，你让同一个 initial state 重复 rollout 8 次几乎不可能，但在 world model 里，同一个 $s_0$ 想跑几次跑几次。

**怎么跑 GRPO？**
从真机采个 initial state $s_0$，在 world model 里 rollout G=8 条 imagined trajectories。
如果这 8 条全成功或全失败，全丢掉不要（Dynamic Sampling）。因为 advantage 全 0，没梯度。
只要里面有成功有失败，就算 advantage：
$$
\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}
$$
这里 $R_i$ 就是第 $i$ 条 trajectory 的 0/1 reward。然后计算新旧 policy的 ratio，套上 clip：
$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^T \min\left(r_{i,t}(\theta) \hat{A}_i, \, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_i\right) \right]
$$
这里它用了 DAPO 的 asymmetric clip，$\epsilon_{\text{low}} = 0.20, \epsilon_{\text{high}} = 0.28$。正向 advantage 允许更大的 ratio 变化，鼓励探索新行为。而且它干脆去掉了 KL regularization，省显存，完全靠 clip 控制偏移。

### 5. 练功成果与涌现能力

这么一套循环下来，效果有多好？

**实验数据 (Table 1):**
在 Mimicgen simulation 的 Square 任务上，base policy 成功率是 24.2%。
只用 128 条真机 rollout budget，WMPO 就干到了 32.8%，DPO 是 28.1%，GRPO 甚至只有 25.0%。
把 budget 加到 1280，WMPO 飙升到 45.3%，几乎翻倍。它极其 data efficient，且 scaling behavior 极好。

**涌现能力:**
这是最 cool 的地方。Base policy 遇到 collision 就死磕，直到 timeout。WMPO policy 在梦里见过千万次失败，竟然自己学会了 self-correction！它撞到之后会自己抬起手，重新对齐，再插进去。这种行为 expert data 里根本没有，完全是从 imagination 的 failure distribution 里学出来的。而且它的成功轨迹更短，因为不再 stuck 了。

**Real-world:**
在 5mm 间隙的Square插入真机任务上，Base 53%，DPO 60%，WMPO 达到了 70%，仅仅用了 200 条 expert demo + 128 条 policy rollout。

### 6. 这套范式的 broader implications

如果我们往深了想，WMPO 其实证明了 Robotics 里一个极其性感的范式：
**A high-fidelity video generator + a reward classifier = A universal RL environment.**

以前我们搞 sim-to-real，需要物理引擎，需要标定，需要调摩擦系数，太脆弱了。WMPO 把 environment 整个 in-context 化进了一个 generative model 里。只要你的 world model 足够好，任何 task 都能在这个“梦境”里 self-play 进化。

而且它的 lifelong learning loop 极具想象力：
采 128 条真机数据 -> fine-tune world model -> 在 imagination 里跑 GRPO 更新 policy -> 用新 policy 再采 128 条...
这就像 AlphaGo 的 self-play 一样，只要这个 loop 转起来，policy 就能无限逼近物理极限。它完全绕开了人类示教的天花板。

当然，目前它的 limitation 也很明显：action 必须离散化成 256 bins（这阻碍了它用 $\pi_0$ 那种 flow matching policy），world model 在极度精细的 collision 上偶尔会 hallucinate (Appendix C 的 Fig 9)，而且 video diffusion rollout 的算力开销巨大。但这些都是工程问题，方向是对的。

**一句话总结 WMPO：** 它用 OpenSora 造了一个机器人能听懂的物理梦境，通过让 policy 在梦境里闭门造车试错，生生试出了现实世界里无法学到的 self-correction 能力。

**References:**
- WMPO Project: [https://wm-po.github.io/](https://wm-po.github.io/)
- Deep Visual Foresight (鼻祖): [https://arxiv.org/abs/1610.00668](https://arxiv.org/abs/1610.00668)
- GRPO (DeepSeekMath): [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- DAPO (Asymmetric clip): [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

---

# WMPO: World Model-based Policy Optimization for VLA — 一次对 "imagination-based RL" 的 pixel-space 重构

Andrej 你好。这篇 WMPO (World Model-based Policy Optimization) 我读了之后第一感受是：它本质上是把 model-based RL 在 LLM 后时代重新做了一遍，但把 world model 从 latent space (Dreamer 那一系) 搬到了 pixel-space video diffusion 上，然后用 GRPO 做 on-policy 优化。这个组合乍看简单，但里面有大量 "把 video generation 当 RL environment 用" 的工程细节是关键。下面我从 first principles 把它拆开讲。

---

## 1. 这篇 paper 在整个 landscape 里的位置

先快速过一下背景坐标轴，这样能 build intuition。

**Vision-Language-Action (VLA) 训练范式的三条路：**
- (a) **Imitation Learning (IL)**：从 expert demonstration 学。问题是 compounding error (DAgger 那篇 Ross et al. 2011 的核心观察 [ref: https://arxiv.org/abs/1011.0686])，policy 没见过 failure state 就不会 recovery。
- (b) **Real-world RL**：直接和物理世界交互。问题是 sample complexity，机器人上百万次 rollout 不现实，且 on-policy 几乎不可能 (每次 update 都要新数据)。
- (c) **WMPO**：在 video-generative world model 里做 imagination-based RL，完全在 pixel space，绕开真实交互。

它的"对手"主要是两条线：
- **Latent world model 派**：Dreamer 系列 ([ref: https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104))，在 RSSM latent state 上 rollout。问题：和 VLA 预训练的 image feature space 对不上，VLA 的 web-scale image prior 用不上。
- **Sim-based / real-world RL 派**：VLA-RL ([ref: https://arxiv.org/abs/2505.18719](https://arxiv.org/abs/2505.18719)), SimpleVLA-RL ([ref: https://arxiv.org/abs/2509.09674](https://arxiv.org/abs/2509.09674))。这些方法 sample efficiency 都很差，且 sim-to-real 工程负担重。

WMPO 的核心 bet：**一个高保真 pixel-space world model 既可以是 "数据增广器" 也可以是 "完整 RL environment"**。这其实回到了 Finn & Levine 2017 的 Deep Visual Foresight ([ref: https://arxiv.org/abs/1610.00668](https://arxiv.org/abs/1610.00668)) 的思路，但那时候 generative model 还不够强，现在 video diffusion 到了能 reproduce robot-object interaction 的水平 (IRASim [ref: https://arxiv.org/abs/2406.14540](https://arxiv.org/abs/2406.14540), Cosmos [ref: https://arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575))，这条路才重新走得通。

---

## 2. 核心数学框架：把 MDP "外包" 给 generative model

### 2.1 Problem formulation (Section 3.1)

把 manipulation 写成 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$：

- $\mathcal{S} = \mathcal{I} \times \mathcal{G}$：state 是 image observation $I_{0:K}$ 和 language instruction $g$ 的乘积。这里有个很强的假设：**robot state 完全由 image 定义**，不带 proprioceptive state (paper 后面 implementation 里明确 "we omit the robot proprioceptive state")。这是个 simplifying choice，paper 自己也提到 POMDP 留给未来。
- $\mathcal{A}$：action chunk $a_t \in \mathbb{R}^{K \times D}$。$K$ 是 chunk length (实验里 $K=8$)，$D$ 是 DoF。每个 dimension 离散化成 256 bins —— 这一步是为了让 VLA 能用 next-token prediction 的形式输出 action (OpenVLA 的路子 [ref: https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246))。
- $\mathcal{P}$：transition function 由 world model $p_\phi(s_{t+1} | s_t, a_t)$ 实现。
- $\mathcal{R}$：learned reward model $R_\psi(\tau) \in \{0, 1\}$，是 trajectory-level 的 sparse binary signal。

**优化目标 (Eq. 1):**
$$
\max_\theta \, \mathbb{E}_{\tau \sim \pi_\theta, p_\phi}\left[ R_\psi(\tau) \right]
$$

- $\theta$: policy $\pi_\theta$ 的参数
- $\tau \sim \pi_\theta, p_\phi$: trajectory 通过 policy 采样 action、world model 采样 next state 联合生成
- $R_\psi(\tau)$: 整条 trajectory 的二值 reward

这个公式表面平常，实质上是个深刻的 decoupling：**RL 训练的 transition 和 reward 都被替换成 learned models**，real environment 只提供 initial state $s_0 \sim \mathcal{D}$。这等价于把 environment 整个 in-context 化进神经网络里。

---

## 3. World Model 的架构细节 (Section 3.2) — 这里的工程细节最值得挖

### 3.1 Backbone 选择

Base 是 OpenSora ([ref: https://arxiv.org/abs/2412.20404](https://arxiv.org/abs/2412.20404)) 的 video diffusion transformer。关键改动：
- 把 OpenSora 的 **3D VAE 换成 SDXL 的 2D VAE** ([ref: https://openreview.net/forum?id=di52zR8xgf](https://openreview.net/forum?id=di52zR8xgf))。理由是 3D VAE 时序压缩太狠，fine-grained motion details 会丢，对 manipulation 这种毫米级精度的任务灾难性。SDXL 的 2D VAE 不做 temporal compression，单帧保真度高。
- Diffusion 在 VAE latent space 做，但 **decode 回 pixel space 再喂给 VLA**，不重训 VLA 的 image encoder 在新 latent space 上。这是和 RSSM ([ref: https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)) 那一脉的关键区别 —— 保持 representation 一致性。

### 3.2 Frame-level Action Control via AdaLN (这段最 tricky)

要让 world model 接受 action conditioning，paper 借鉴 IRASim ([ref: https://arxiv.org/abs/2406.14540](https://arxiv.org/abs/2406.14540))，扩展 AdaLN ([ref: https://arxiv.org/abs/1911.07013](https://arxiv.org/abs/1911.07013)) 把 action 信号 inject 到每一帧。

公式 (Section 3.2 内嵌)：
$$
\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_1^i) \cdot \text{Block}\left(\gamma_1^i \cdot \text{LayerNorm}(\mathbf{x}^i) + \beta_1^i\right)
$$

变量解释：
- $\mathbf{x}^i$：第 $i$ 帧的 feature representation (在 transformer block 内部)
- $a_i$：第 $i$ 帧对应的 action
- $\gamma_1^i, \beta_1^i, \alpha_1^i$：由 MLP 从 $a_i$ 生成的 modulation 系数
  - $\gamma_1^i$：LayerNorm 输出的 scale
  - $\beta_1^i$：LayerNorm 输出的 shift
  - $\alpha_1^i$：residual connection 的 scale (gating)
- Block：either MHA or FFN

直觉：每帧独立调制，让 action 信号 frame-wise 注入到 transformer 的每个 block。比把 action 拼到 token sequence 里更结构化，也避免了 attention 把 action token 和 image token 混在一起可能引入的 misalignment。

### 3.3 Noisy-frame Conditioning (长 horizon 生成的关键 trick)

纯 autoregressive video generation 在长 horizon 下会 compounding error —— 早期一帧略歪，后面就 snowball 成 garbage。paper 的 trick：

**训练时把 conditioning frame $I_{i-m:i}$ 加 50/1000 steps 的 diffusion noise**，不保持 clean。

直觉：训练时人为制造 "imperfect conditioning"，inference 时即使前面帧有点 drift，模型也能 robust 处理。这其实是 data augmentation 思想在 video diffusion 上的应用，类似 classifier-free guidance 训练时随机 drop condition。

效果：能稳定生成 hundreds of frames 的 trajectory，对 RL 这种长 horizon rollout 至关重要。

### 3.4 Trajectory Generation (Eq. 2)

$$
I_{i:i+K} \sim p_\phi(I_{i-c:i}, a_{i:i+K})
$$

- $c$：conditioning frames 数 (实验 $c=4$)
- $K$：一次生成 $K$ 帧 (实验 $K=8$，和 action chunk length 对齐)
- $I_{i-c:i}$：前面 $c$ 帧作为 context
- $a_{i:i+K}$：对应的 action chunk

循环到 maximum length $N$ 得到 $\tau = \{I_{0:N}, a_{0:N}\}$。这是 **clip-level autoregressive** 而非 frame-level，paper 强调这点对 reward assignment 重要：短 horizon prediction 很难定义准确 reward，且 reward hacking 严重 (policy 学会骗 short predictor)。

### 3.5 Policy Behavior Alignment — 这是把 IL-trained world model 变成 RL environment 的核心

World model 在 Open X-Embodiment (OXE) ([ref: https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)) 上预训练。OXE 几乎全是成功 demo，所以 world model 学到的 dynamics 偏向 "成功轨迹的 manifold"。

但 RL 训练时 policy 会 explore 失败 region，world model 没见过这种 state distribution 就会 hallucinate 不真实的 failure dynamics，RL 训练崩盘。

解决：**用 policy 自己 rollout 收集的少量真实 trajectory (实验里 P=128 或 1280) 微调 world model**。这个 step 让 world model 适应 policy 的 (state, action) 分布，并学会 faithful 模拟 failure mode。

这点和 "RL + model" 的经典陷阱 distribution shift 直接对应：world model 在 expert distribution 下 ok，policy 一 explore 就 OOD。Policy Behavior Alignment 是 WMPO 的 in-distribution adaptation 机制。

---

## 4. Reward Model (Section 3.3) — lightweight 但有几个细节

### 4.1 数据构造

- Positive sample：成功 trajectory 的 terminal clip $c_N$ (长度 $L=8$)
- Negative sample：成功 trajectory 的中间 clip (避免 trivial 区分) + 失败 trajectory 的任意 clip
- 每 batch 内 balance positive/negative，解决 class imbalance

### 4.2 架构

VideoMAE ([ref: https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)) encoder + linear head，BCE loss。

### 4.3 Inference

Sliding window (stride $s=1$) 遍历 trajectory，每 clip 算 success probability。任一 clip 超过 threshold $\tau_{\text{thr}}$ 即判 success。

直觉：terminal clip 对成功判定最有信息量 (任务完成通常有明确 visual cue)，中间 clip 提供辅助判别。这个设计同时避免了 frame-level reward shaping 的复杂性和 trajectory-level single decision 的脆性。

效果：F1 score > 0.95 across all tasks，reward hacking 被有效压制。

---

## 5. GRPO 作为 Policy Optimizer (Section 3.4)

### 5.1 为什么 GRPO 而非 PPO

GRPO ([ref: https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)) 在 DeepSeek-R1 ([ref: https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)) 上证明了在 sparse reward 下的稳定性。对 VLA 这种 binary task success 的 reward landscape 特别合适。Group-relative advantage 也省去了 critic 网络 (PPO 需要训练 value function)，对 VLA 这种大模型场景算力友好。

### 5.2 Dynamic Sampling

从 DAPO ([ref: https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)) 借来：如果一个 group 里所有 trajectory 都成功或都失败，丢弃重采样。理由：全 success 或全 failure 时 advantage 全 0，gradient vanishing，浪费算力。这点在 physical world RL 几乎做不到 (同一 state 多次 rollout 工程上极难)，但在 world model 里 trivial —— 同一个 $s_0$ 想跑多少次跑多少次。这是 WMPO 的一个隐性 superpower。

### 5.3 公式详解

**Log-prob (Eq. 3):**
$$
\log \pi_{\theta_{\text{old}}}(a_t | s_t) = \sum_{i=1}^{K} \sum_{j=1}^{D} \log \pi_{\theta_{\text{old}}}(a_t^{i,j} | s_t)
$$

- $a_t$：time $t$ 的 action chunk
- $a_t^{i,j}$：chunk 内第 $i$ 个 action 的第 $j$ 个 DoF
- $K$：chunk length
- $D$：DoF 数

把 chunk 的 log-prob 拆成每个 dimension 独立 log-prob 之和，是因为 action 离散成 256 bins 后每个 dimension 是独立的 categorical distribution。

**Loss (Eq. 4):**
$$
\mathcal{J}(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D}, \{\tau_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^T \min\left(r_{i,t}(\theta) \hat{A}_i, \, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_i\right) \right]
$$

- $G$：group size (实验 8)
- $T$：trajectory length
- $r_{i,t}(\theta)$：probability ratio (Eq. 5)
- $\hat{A}_i$：trajectory $i$ 的 normalized advantage
- $\epsilon_{\text{low}} = 0.20, \epsilon_{\text{high}} = 0.28$：asymmetric clip (DAPO 的 trick，positive advantage 允许更大 ratio 变化鼓励探索)

**Ratio & Advantage (Eq. 5):**
$$
r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} | s_{i,t})}, \quad \hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}
$$

- $r_{i,t}$：新旧 policy 在 trajectory $i$ 时间 $t$ 的 action 上的概率比，类似 PPO 的 importance ratio
- $R_i$：trajectory $i$ 的 binary reward
- $\hat{A}_i$：group-relative normalized advantage —— 同一 $s_0$ 的 G 个 trajectory 互相比较，去掉 baseline (mean)，归一化 (std)

关键：**没有 KL divergence regularization** (跟 DAPO 一致)，去掉 reference model，省显存。代价是失去 explicit constraint，依赖 clip 控制偏移。

---

## 6. Algorithm 1 的 pipeline (Appendix A)

伪代码值得细看，几个关键点：

```
while not converged:
    B = []
    while |B| < B:  # 收集 batch
        sample s_0 from D
        for i in 1..G:
            imagine τ_i with π_θold + p_φ
            R_i = R(τ_i)
            if all(R_1..R_G) or none(R_1..R_G): continue  # dynamic sampling
            compute group mean μ, std σ
            Â_i = (R_i - μ) / σ
            precompute log π_θold(a_t^i | s_t^i) for all t
            append (τ_i, Â_i) to B
    
    for epoch in 1..E:
        for mini-batch M from B:
            update θ with Eq. 4
    θ_old ← θ
```

注意几个细节：
- `precompute log π_θold`：reference log-prob 算一次缓存，省算力
- 多 epoch E 复用 batch：on-policy 的标准做法，但每个 mini-batch update 都会 push policy 偏离 θ_old，clip 控制这个偏离
- Iterative: `θ_old ← θ` 后下一轮用新 policy 重新 imagine

---

## 7. 实验结果深度解析

### 7.1 Main comparison (Table 1)

| Rollout budget P | Method | Coffee | StackThree | ThreePieceAssembly | Square | Mean |
|---|---|---|---|---|---|---|
| - | Base | 43.8 | 46.9 | 19.5 | 24.2 | 33.6 |
| 128 | GRPO | 38.3 | 52.3 | 17.2 | 25.0 | 33.2 |
| 128 | DPO | 43.8 | 53.9 | 23.4 | 28.1 | 37.3 |
| 128 | **WMPO** | **61.7** | **56.3** | **37.5** | **32.8** | **47.1** |
| 1280 | GRPO | 47.7 | 54.7 | 20.3 | 25.8 | 37.1 |
| 1280 | DPO | 52.3 | 57.0 | 26.7 | 33.6 | 42.4 |
| 1280 | **WMPO** | **75.0** | **64.1** | **46.1** | **45.3** | **57.6** |

关键 observations：
- P=128 时 WMPO 比最强 baseline 高 +9.8 mean
- P=1280 时 gap 扩大到 +15.2
- GRPO 在 P=128 时经常 underperform base policy (38.3 vs 43.8 on Coffee) —— sample 太少，update 不充分
- DPO 平台化 (static data reuse 限制)
- WMPO 随 budget scaling 持续改进 —— 这正是 model-based RL 应该有的 scaling behavior

注意 Square task 从 24.2 → 45.3 (P=1280)，几乎翻倍。这是 RL 真正学会 "self-correction" 行为的体现。

### 7.2 Generalization (Table 2)

三种 disruption：position shift、background change、texture change。

WMPO 在所有 disruption 上都最好，特别 texture disruption (10.9 → 16.4)。DPO 在 background 和 texture 上反而比 base 还差 —— 印证 DPO 容易学 spurious visual cue。WMPO 在 world model 里 rollout 大量 "imagined" 数据，相当于 implicit data augmentation，generalization 更鲁棒。

### 7.3 Emergent Behavior (Section 4.3)

最有意思的部分：
- **Self-correction**：Square 任务中 base policy 遇 collision 就死磕到底直到 timeout，WMPO policy 学会 lift → realign → insert。这种行为 expert demo 里没有，是从 world model 大量失败轨迹的 imagination 里学出来的。
- **Shorter successful trajectories** (Fig. 5)：WMPO policy 减少了 stuck 行为，平均 trajectory 长度显著短于 base。Reward 是 sparse binary，但 paper 推测这是因为 stuck 通常导致 timeout 失败，policy 学会避开 stuck state。

### 7.4 Real-world (Section 4.6)

"Insert square into stick" (5mm clearance)：
- Base: 53%
- DPO: 60%
- WMPO: 70%

只用了 200 expert demo + 128 policy rollout。这个数字在 real robot fine-grained manipulation 上很可观。

### 7.5 Lifelong Learning (Fig. 6)

迭代式：collect 128 rollout → WMPO update → 用新 policy collect 128 → ...

StackThree 任务上 WMPO 持续改进，DPO 训练不稳定。和 300/428/556 expert demo 的 base 对比，WMPO 仅靠 policy 自己 rollout (无需 human) 就能逼近甚至超越更多 expert demo 的效果。这是 scalable self-improvement 的 evidence。

---

## 8. 限制和未来方向

Paper 自己提的 (Appendix D)：
- 只支持 discrete action (256 bins)，不支持 flow-based policy (π0 [ref: https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164) 那种 flow matching)。未来用 FlowGRPO ([ref: https://arxiv.org/abs/2505.05470](https://arxiv.org/abs/2505.05470)) 扩展。
- 假设 fully observable (omit proprioceptive state)，POMDP 留给未来。

我能想到的额外限制：
- **World model 质量 ceiling**：imagination-based RL 的上限是 world model 的 fidelity。fine-grained manipulation (5mm clearance) 上 world model 能不能精确捕捉微小 collision？Fig. 9 已经显示有 rare failure。如果任务精度要求再上一个数量级，这套方法的可靠性存疑。
- **Reward model bottleneck**：reward model F1 > 0.95 看起来高，但在 long-horizon task 上累积 error。如果 trajectory 长 N=100，即使 per-clip accuracy 99%，整体 trajectory-level 准确率会下降。
- **Initial state distribution $\mathcal{D}$**：还是需要少量真实 trajectory 提供 initial states 和做 behavior alignment。完全 zero-real-interaction 还做不到。
- **Computational cost**：world model 在 32 H100 上训练 12M + 3M steps，policy optimization 也在 32 H100。每个 batch 需要在 world model 里 rollout G=8 trajectories × N frames 的 video diffusion forward，相比直接 sim-based RL (e.g. Mimicgen) 算力开销巨大。
- **No off-policy correction**：去 KL regularization 鼓励探索，但完全没约束下 policy 可能 drift 到 world model 的 OOD 区域，长期可能 collapse。paper 里的 dynamic sampling 部分缓解但不根治。

---

## 9. 联想和延伸思考

### 9.1 和 Dreamer 系列的对照

Dreamer V1/V2/V3 ([ref: https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193), [ref: https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)) 的核心 insight 是：在 latent space rollout 比 pixel space 快几个数量级。WMPO 反其道而行回到 pixel space，理由是 VLA 的预训练 representation 在 pixel space。这是个 representation-first vs speed-first 的 tradeoff。如果未来 VLA 的 image encoder 能 fine-tune 到 latent space representation，Dreamer 路线可能重新 competitive。

### 9.2 和 LLM RLHF 的对照

GRPO + Dynamic Sampling + asymmetric clip 这套配方直接从 LLM RL (DAPO, DeepSeekMath, DeepSeek-R1) 借过来。区别：
- LLM 里 reward 可来自 rule-based verifier (math, code)，WMPO 必须 learn reward model
- LLM rollout 是纯文本生成 (autoregressive, cheap)，WMPO rollout 是 video diffusion (expensive)
- LLM 同一 prompt 多次 sampling trivial，WMPO 在 physical world 难，在 world model 里 trivial —— 这点反而比 LLM 还方便

### 9.3 World Model 作为 "Universal Simulator"

WMPO 实际上证明了一个更深的 claim：**一个足够好的 video generation model + 一个 reward model = 一个 RL environment**。这把 sim-to-real 问题转化成 generative-model-quality 问题。如果 Cosmos / Genie 3 ([ref: https://genie.deepmind.com/](https://genie.deepmind.com/)) / Sora 类模型继续进步，未来 robotics RL 的 bottleneck 可能完全转移到 reward specification 上。

### 9.4 和 Self-Play / AlphaGo 的对应

AlphaGo 的 self-play 是 policy 互相对弈生成数据。WMPO 是 policy 在 world model 里自己 rollout，相当于 "policy 和 environment 对弈"。AlphaGo 的 environment (Go rules) 是 deterministic perfect information，WMPO 的 environment (world model) 是 stochastic imperfect model。这种 "在 learned environment 里 self-improve" 的范式可能比 AlphaGo 更 general —— 任何 task 都能套这个框架，只要有足够数据训 world model。

### 9.5 Lifelong Learning 的真正含义

Section 4.5 的 lifelong learning demo 我觉得是 paper 最低估的贡献。它意味着：
```
loop:
    collect trajectories with current policy
    fine-tune world model on these trajectories
    use world model to imagine & optimize policy with GRPO
```
这等价于一个 self-contained iterative self-improvement loop。如果 world model fine-tuning 增量足够 cheap (paper 里没明确说 incremental fine-tune 的成本)，这个 loop 可以一直转下去，无限逼近 task 的物理上限。这非常像 AlphaGo 的 self-play 训练 loop，但对应 robotics。

### 9.6 关于 Reward Hacking 的 Open Question

Paper 说 reward model F1 > 0.95 所以没 reward hacking。但严格说，reward hacking 可能在 two-level 都发生：
1. **Policy → reward model**：policy 生成 trajectory 让 reward model 误判 success (e.g. 找到 reward model 的 blind spot 视觉特征)
2. **World model → reality**：world model 生成不真实但 reward model 认为成功的 trajectory，policy 学会 exploit world model 的 hallucination

第 1 种 paper 部分解决 (high F1)。第 2 种更微妙 —— world model 和 reward model 都可能在同一类 OOD state 上失败，policy 一旦找到这个 "failure mode 的交集" 就能 exploit。Dynamic sampling 部分缓解 (uniform failure 被丢弃)，但不根治。这点是 imagination-based RL 的 fundamental risk。

### 9.7 为什么不用 Diffusion Policy 做世界模型 + Policy 联合训练

Paper 把 world model 和 policy 分开训，policy 是 OpenVLA-OFT ([ref: https://arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645))。一个 alt design：让 world model 和 policy share backbone，类似 Dreamer 的 latent dynamics + policy head 共享 representation。这能省算力且 representation 一致性更好。但 paper 不走这条路，可能因为 OpenVLA-OFT 是 frozen pretrained backbone，重训成本高。

---

## 10. Take-aways for Build Intuition

如果让我提炼这篇 paper 最值得带走的几个 idea：

1. **Pixel-space > Latent-space for VLA RL**：因为 VLA 的 prior 在 pixel space，world model 必须在同一个 representation 上才能利用 prior。
2. **Policy Behavior Alignment 是 distribution shift 的工程解药**：world model 在 expert data 上学，policy 在自己 distribution 上 rollout，必须用 policy 自己的 data 微调 world model 才能 cover failure mode。
3. **Clip-level autoregressive + Noisy-frame conditioning** 是长 horizon video generation 的两个关键 trick，缺一不可。
4. **Dynamic Sampling 让同一 initial state 反复 rollout 成为 superpower**：physical world 做不到，world model 里 trivial，这让 GRPO 这种 group-relative 方法充分发挥。
5. **Reward = trajectory-level binary + lightweight classifier**：sparse reward 比 dense shaped reward 更鲁棒，前提是有 reliable classifier。
6. **Lifelong learning loop**：policy rollout → world model fine-tune → imagine & optimize → repeat，这是 robotics self-improvement 的范式雏形。

---

## References

- WMPO Project Page: [https://wm-po.github.io/](https://wm-po.github.io/)
- OpenVLA: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- OpenVLA-OFT: [https://arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645)
- GRPO (DeepSeekMath): [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- DAPO: [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
- DeepSeek-R1: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- DPO: [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- Dreamer V3: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- Dreamer V2: [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193)
- OpenSora: [https://arxiv.org/abs/2412.20404](https://arxiv.org/abs/2412.20404)
- SDXL: [https://openreview.net/forum?id=di52zR8xgf](https://openreview.net/forum?id=di52zR8xgf)
- IRASim: [https://arxiv.org/abs/2406.14540](https://arxiv.org/abs/2406.14540)
- Open X-Embodiment: [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)
- Mimicgen: [https://arxiv.org/abs/2310.17596](https://arxiv.org/abs/2310.17596)
- VideoMAE: [https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)
- Deep Visual Foresight: [https://arxiv.org/abs/1610.00668](https://arxiv.org/abs/1610.00668)
- DAgger (Ross et al. 2011): [https://arxiv.org/abs/1011.0686](https://arxiv.org/abs/1011.0686)
- Cosmos World Foundation Model: [https://arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575)
- VLA-RL: [https://arxiv.org/abs/2505.18719](https://arxiv.org/abs/2505.18719)
- SimpleVLA-RL: [https://arxiv.org/abs/2509.09674](https://arxiv.org/abs/2509.09674)
- π0 (flow-based VLA): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- FlowGRPO: [https://arxiv.org/abs/2505.05470](https://arxiv.org/abs/2505.05470)
- AdaLN paper: [https://arxiv.org/abs/1911.07013](https://arxiv.org/abs/1911.07013)

---

如果你要我深挖某一块 —— 比如 AdaLN 的 frame-level modulation 实际怎么实现、GRPO 的 asymmetric clip 数值怎么选、world model 的 OOD failure 怎么 detect，或者 lifelong learning loop 的 stopping criterion 怎么定 —— 直接告诉我，我接着展开。
