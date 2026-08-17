---
source_pdf: VLAWIterative Co-Improvement of Vision-Language-Action Policy and World
  Model.pdf
paper_sha256: 34ba87492a0fca9d382c366a80884eef99bc1101d973bbbb64125ae78bedffb8
processed_at: '2026-08-13T03:01:13-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

没问题 Andrej，咱们抛开那些学术黑话，用最直白的人话，结合硬核的技术细节和 intuition，把这篇 paper 揉碎了讲。

### 1. 这篇 Paper 到底在干嘛？

一句话总结：**让 Robot 学会在“梦境”中总结经验，从而在现实世界里表现得更好。**

目前 Robotics 领域最大的痛点是什么？数据。你在 simulation 里跑一百万次都没问题，但一旦到了 real world，Robot 跑偏一次，就需要人去把碰倒的积木扶起来，把掉在地上的笔捡起来。收集 real-world online rollout data 简直是金钱和时间的黑洞。

既然 real world 跑 rollout 太贵，那我们能不能让 Robot 自己“做梦”？这就是 **World Model** 的作用。World Model 就是一个 action-conditioned video generator，你给它一帧画面和一个 action，它给你生成接下来的视频。相当于一个“learned simulator”。

但作者发现，直接拿现成的 World Model (比如 Ctrl-World) 来做“梦境”，会遇到严重的 **Over-optimism (过度乐观)** 问题。因为这些 World Model 都是在 expert demonstrations 上训练的，它们脑子里只见过成功。结果就是，哪怕你的 Robot 动作蠢到没碰到杯子，World Model 也会“幻觉”出杯子被成功拿起的画面。用这种充满幻觉的梦境去训练 Policy，Policy 只会越学越傻。

VLAW 的核心绝招就是：**用少量的、包含大量失败的真实经验，去给 World Model “认清现实”。** World Model 现实了，做梦就准了；做梦准了，就能生成大量高质量的合成数据；拿这些合成数据去 SFT VLA Policy，Policy 就变强了；Policy 变强了，下次收集到的真实经验质量就更高。这就是所谓的 **Iterative Co-Improvement**。

### 2. 核心三步走：技术细节与大白话解析

整个 VLAW pipeline 就是一个循环，包含三个核心步骤。

#### Step 1: 让 World Model 认清现实

我们先让当前的 Policy 在真实机器人上跑 50 次，把这 50 次的录像（包含成功和失败）收集起来，叫作 $\mathcal{D}_{\text{real}}$。然后用这批数据去 fine-tune 预训练好的 Ctrl-World 模型。

这里有个很关键的 trick，叫做 **Co-training (混合训练)**，看公式 Eq. (2)：
$$ \mathcal{L} = \mathcal{L}_{\mathcal{D}_{\text{real}}} + \lambda \mathcal{L}_{\mathcal{D}_{\text{DROID}}} $$

**变量与下标解析：**
*   $\mathcal{L}_{\mathcal{D}_{\text{real}}}$: 在新收集的 50 条真实数据上的 loss。
*   $\mathcal{L}_{\mathcal{D}_{\text{DROID}}}$: 在巨大的原始 DROID dataset 上的 loss。
*   $\lambda$: 权重系数，控制正则化强度。

**Intuition:** 为什么不直接用 50 条数据 fine-tune？因为 50 条数据太少了，直接 fine-tune 模型会 catastrophic forgetting（把原来懂的各种泛化常识全忘了，死记硬背这 50 条）。通过混合巨大的 DROID 数据集，等于是在告诉模型：“你要学习这 50 条里的物理细节，但你不能忘了原来学的那些通用常识。”

实验数据（Table 1）表明，加了 real rollout 数据后，World Model 的 False Positive（把失败预测成成功）从 11 个暴跌到 1 个。这就说明它不再盲目乐观了，它学会了“摩擦力不够东西会掉”这种残酷的物理事实。

#### Step 2: VLM 裁判过滤梦境

World Model 认清现实后，我们让它生成 500 条 synthetic trajectories（纯做梦生成的数据）。做梦总是天马行空的，我们需要挑出那些“真的能成功”的梦境来教 Policy。

作者用 Qwen3-VL-4B-Instruct 当裁判。但直接让 VLM 回答 yes/no，它也会过度乐观。作者用了个狠招，看公式 Eq. (3)：
$$ R(\tau^i) = \mathbf{1} [P(\text{yes} | \tau^i, I^i) > \alpha] $$

**变量解析：**
*   $\tau^i$: 第 $i$ 条生成出来的视频轨迹。
*   $I^i$: 任务指令（比如“把红色方块放到蓝色方块上”）。
*   $P(\text{yes} | \tau^i, I^i)$: VLM 输出 "yes" 这个 token 的 softmax 概率。
*   $\alpha$: 阈值，设为 0.8。
*   $\mathbf{1}[\cdot]$: 指示函数，条件成立输出 1，不成立输出 0。

**Intuition:** 这个 $\alpha = 0.8$ 简直是点睛之笔。VLM 裁判往往比较“宽容”，觉得“大概齐就算成功了吧”。强制要求概率超过 0.8，就是逼着裁判严格把关。那些模棱两可的、勉强碰到的，统统判定为失败。这样送进下一步的 synthetic data 纯度极高，没有任何“毒数据”。Appendix C 的实验证明，加了阈值后，False Positive 从 8 降到了 2。

#### Step 3: Flow-Matching Policy 的 Filtered SFT

有了高纯度的合成成功数据 $\mathcal{D}_{\text{syn}}^+$，加上真实世界里跑出的成功数据 $\mathcal{D}_{\text{real}}^+$，我们怎么更新 Policy？

目前最强的 VLA 模型 $\pi_{0.5}$ 用的是 Flow-Matching 目标，它建模的是 action 的 vector field，没有 tractable 的 action likelihood。这意味着你没法直接用 PPO 那种 Policy Gradient 去算。

作者在这里做了一个非常优雅的数学推导（详见 Appendix A）。他们证明了，如果你在 Regularized RL 框架下，引入一个 KL penalty 限制 Policy 不要偏离原来的 reference policy 太远，最优的 Policy 解会是：
$$ \pi^\star(a|o) \propto \pi_{\text{ref}}(a|o) \exp\left( \frac{A^{\pi_{\text{ref}}}(o, a)}{\beta} \right) $$
因为 advantage 只有 0 和 1（失败或成功），且 discount factor $\gamma \approx 1$，最后整个复杂的 RL 优化问题退化成了极其简单的 **Filtered Behavioral Cloning**，见公式 Eq. (4)：
$$ \mathcal{L} = \mathbb{E}_{(o,a) \sim \mathcal{D}_{\text{syn}}^+ \cup \mathcal{D}_{\text{real}}^+} \mathcal{L}_{\text{FM}}(\theta; o, a) $$

**变量与上下标解析：**
*   $(o,a)$: Observation 和 Action pair。
*   $\mathcal{D}_{\text{syn}}^+ \cup \mathcal{D}_{\text{real}}^+$: 纯成功数据集的并集。
*   $\mathcal{L}_{\text{FM}}(\theta; o, a)$: Flow-Matching loss。
*   $\theta$: Policy 网络参数。

**Intuition:** 这段推导的核心 intuition 是：在无法直接算 log-likelihood 的情况下，我们通过最小化 Flow-Matching loss，把 Policy “投影”到最优解的分布上。因为我们只拿 success cases 去训，相当于给每个 success case 赋予权重 1，给 failure case 赋予权重 0。这避开了 RL 训练里臭名昭著的 bootstrapping instability 和高方差问题。用最稳定的 SFT 目标，干着最硬核的 RL 的事。

### 3. 实验结果与深层 Intuition 联想

实验在 5 个 contact-rich 的真实任务上做：Stacking, Open Book, Erase Marks, Scooping, Drawing。

看 Table 2，Base model 平均成功率 46.0%，经过两轮 VLAW 迭代（Ours-2），成功率飙升到 **86.8%**。尤其是 Drawing 任务，从 22% 直接干到了 78%。

这里我联想到几个非常有意思的 high-level intuition：

**1. 这简直就是 Robotics 版的 AlphaGo**
AlphaGo 用 MCTS (System 2) 搜索出来的好棋，去蒸馏 Policy Network (System 1)。VLAW 用的也是这个逻辑：World Model 就是 MCTS，它在 imagination 里搜索出成功的轨迹，然后用这些轨迹去 SFT VLA Policy。做梦就是它的 lookahead search。

**2. Learned Simulator 彻底碾压 Hard-coded Simulator**
以前做 Robotics RL，大家都在 Isaac Gym, MuJoCo 里建 URDF 模型，调摩擦系数、质量参数。调一个 deformable object（比如 Open Book, Erase Marks 这种任务）的仿真环境能把人逼疯。VLAW 直接抛弃了这些 hard-coded physics engine，用 **Learned Video Diffusion Model** 当 simulator。只要数据够，模型自己就能学会书页翻动的物理规律。这是 Robotics 走向 generalist 的必经之路。

**3. VLM 是天然的 Reward Model**
这篇 paper 暗示了一个巨大的 scaling law：随着 Qwen-VL、GPT-4o 这种 VLM 越来越强，我们根本不需要去手写 success criteria。VLM 自带的 common sense 完全足以判断“碗里有花生”还是“碗是空的”。VLM 在 Robotics 里的角色正在从单纯的 “Instruction Parser” 进化为 “Reward Provider” 和 “Verifier”。

**4. Synthetic Data 的 Scaling Law**
Figure 9 的 ablation 证明，把 synthetic data 从 500 减到 250，性能就掉。这说明 Policy 渴求更多的 data。如果 World Model 足够强，我们可以生成 5000、50000 条 synthetic data。Real data 的作用仅仅是用来 ground the World Model。这打破了 Robotics 领域长期以来的 data bottleneck。一旦 World Model 的 fidelity 突破某个阈值，Robotics 就能像 LLM 一样通过 synthetic data 实现指数级进化。

### 4. 总结

这篇 paper 的 recipe 非常实用且优雅：
1. 用极少量的 real rollout（包含失败）去 fine-tune World Model 和 Reward VLM。
2. World Model 被现实“毒打”后，不再盲目乐观，能生成逼真的物理演化视频。
3. 严格的 VLM 裁判从梦境中挑出确凿的成功轨迹。
4. 拿这些高纯度的合成数据加上真实成功数据，用极其稳定的 Flow-Matching SFT 去更新 Policy。
5. 循环往复，互相 bootstrap。

这个 work 证明了：Robotics 的未来不在于建造更精密的物理模拟器，在于让 generative models 学会物理直觉，并在 imagination 中实现 self-improvement。

References for further reading:
*   [VLAW Project Page](https://sites.google.com/view/vlaw-arxiv)
*   [AlphaGo Nature Paper - Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
*   [Ctrl-World: Controllable World Model](https://arxiv.org/abs/2510.10125)
*   [$\pi_{0.5}$ VLA Model](https://arxiv.org/abs/2504.16054)
*   [DROID Dataset](https://droid-dataset.github.io/)

---

你好 Andrej！很高兴能和你深入探讨这篇 paper。这篇 VLAW (Vision-Language-Action & World model) 的工作非常契合你一直以来对于 **Agentic AI** 和 **World Models** 的直觉。这篇 paper 的核心贡献在于解决了一个非常实际的 bottleneck：在 real world 里面跑 robot policy 收集 data 极其昂贵且 unscalable。作者提出了一个 iterative co-improvement 的 framework，让 VLA policy 和 action-conditioned world model 互相 bootstrap，利用 **synthetic rollouts** 来实现 policy 的 post-training。

我们来一层层剥开这篇 paper 的技术细节，并建立底层的 intuition。

### 1. Core Intuition: 为什么需要 Co-Improvement？

目前的 VLA models (比如 $\pi_{0.5}$, OpenVLA) 虽然在 demonstration data 上表现很好，但在 deployment 时往往需要 online RL 来 refine policy。在 real world 做 RL (比如 PPO/GRPO) 几乎是 intractable 的，因为 reset environment 和收集 rollout 极其耗时。

自然而然的想法是：用 **World Model** 替代 real environment 做 rollout。但是现有的 video generation models (如 Genie 3, 1X World Model, Ctrl-World) 存在致命缺陷：
1. **Over-optimism**: 它们都是在 expert demonstrations 上训练的，见到的全是 success cases，导致模型在生成 video 时会“幻觉”出成功的结局，哪怕 action 完全错误。
2. **Lack of Physical Fidelity**: 在 contact-rich tasks (如 wipe marks, scoop peanuts) 中，微小的物理接触细节决定成败，而这些 pure generative models 无法精准建模。

VLAW 的核心 insight 是：**用 policy 在 real world 产生的 failure cases 去 grounding world model**。World model 见过了真实的 failure dynamics，就会变得悲观且物理准确。然后我们在这个准确的 world model 里面跑大量的 synthetic rollouts，用 VLM reward model 过滤出 success cases，最后拿这些 synthetic success data 去 SFT (Supervised Fine-Tuning) VLA policy。

### 2. Methodology & Formulations 深度解析

整个 pipeline 可以分为三个核心模块，我们结合公式和架构图来详解。

#### 2.1 World Model Learning with Real Rollouts (Section 4.1)

这里作者使用的是基于 Diffusion 的 action-conditioned world model (Ctrl-World)。给定当前的 observation $o_t$ 和 action chunk $a_{t:t+H}$，模型去 predict 未来的 observation sequence $x_0 = o_{t+1:t+H}$。

训练目标是标准的 Latent Diffusion loss，见 **Eq. (1)**：
$$ \mathcal{L}_{\mathcal{D}_{\text{real}}} = \mathbb{E}_{x_0, \epsilon, t'} \| \hat{x}_0(x_{t'}, t', c) - x_0 \|^2 $$

**变量与上下标解析：**
*   $x_0$: Target future observation sequence（干净的未来视频帧）。
*   $\epsilon$: Gaussian noise sampled from $\mathcal{N}(0, I)$。
*   $t'$: Diffusion timestep，$t' \in [0, T']$，表示加噪的步数。
*   $x_{t'}$: Noised future at diffusion step $t'$，计算公式为 $x_{t'} = \sqrt{\bar{\alpha}_{t'}} x_0 + \sqrt{1 - \bar{\alpha}_{t'}} \epsilon_{t'}$，其中 $\bar{\alpha}_{t'}$ 是 noise schedule。
*   $c$: Conditioning inputs，包含 action chunk $a_{t:t+H}$ 和 current observation $o_t$。
*   $\hat{x}_0$: 神经网络预测的 clean future。

为了防止模型 overfit 到少量的 online rollout data (每个 task 只有 50 条 trajectories)，作者采用了 **Progressively Growing Dataset and Co-training**，见 **Eq. (2)**：
$$ \mathcal{L} = \mathcal{L}_{\mathcal{D}_{\text{real}}} + \lambda \mathcal{L}_{\mathcal{D}_{\text{DROID}}} $$
$\lambda$ 控制了正则化的强度，$\mathcal{D}_{\text{DROID}}$ 是原始的广覆盖 dataset。这确保了模型既学到了 target task 的 failure dynamics，又不丧失泛化能力。

#### 2.2 Vision-Language Reward Model (Section 4.1 & Appendix C)

为了自动判断 synthetic trajectory 是否成功，作者 finetune 了一个 Qwen3-VL-4B-Instruct 模型作为 reward model。

这里有一个非常关键的 trick，见 **Eq. (3)**：
$$ R(\tau^i) = \mathbf{1} [P(\text{yes} | \tau^i, I^i) > \alpha] $$

**变量解析：**
*   $\tau^i$: 第 $i$ 条 trajectory video (下采样到 16 frames)。
*   $I^i$: Task instruction (如 "stack block A on block B")。
*   $P(\text{yes} | \tau^i, I^i)$: VLM 输出 "yes" token 的 probability。
*   $\alpha$: Threshold，论文中设为 0.8。

**Intuition:** 如果直接让 VLM 输出 yes/no，模型会极其 optimistic，产生大量的 false positives (见 Table 3)。通过强制要求 "yes" 的 softmax probability 超过 0.8，我们本质上是在做 temperature scaling / conservative filtering。这就像是给 reward model 加了一个 explicit 的 margin，把那些模棱两可的 borderline cases 全部判为 failure，保证送入 policy training 的 synthetic data 纯度极高。

#### 2.3 VLA Policy Post-Training (Section 4.2 & 4.3)

Policy 更新采用的是 Flow-Matching loss，这是这篇 paper 最有意思的 theoretical 贡献。作者将整个 process 解释为 **Advantage-Weighted Regression (AWR)** 在 Flow-Matching policy 下的近似。

在 regularized RL framework 下，目标函数如 **Eq. (5)**：
$$ J(\theta) = \mathbb{E}_{\tau \sim \rho_{\pi_\theta}} [R(\tau)] - \beta \mathbb{E}_{o \sim \rho_{\pi_\theta}} [D(\pi_\theta(\cdot|o) \| \pi_{\text{ref}}(\cdot|o))] $$

**变量解析：**
*   $\pi_{\text{ref}}$: Reference policy (即初始的 $\pi_{0.5}$)。
*   $\beta$: Temperature parameter，控制 KL divergence penalty 的强度。
*   $D$: KL divergence measure。

这个 objective 的 closed-form optimal policy 是：
$$ \pi^\star(a|o) \propto \pi_{\text{ref}}(a|o) \exp\left( \frac{A^{\pi_{\text{ref}}}(o, a)}{\beta} \right) $$
其中 $A^{\pi_{\text{ref}}}(o, a)$ 是 advantage function。

**Flow-Matching 下的投影：**
因为 $\pi_{0.5}$ 是 Flow-Matching model，无法直接计算 tractable log-likelihood (因为它建模的是 action 的 vector field 而不是 density)，所以传统的 KL projection 无法使用。作者定义了一个 surrogate divergence，见 **Eq. (6)**：
$$ D_{\text{FM}}(\pi^\star(\cdot|o), \pi_\theta(\cdot|o)) \triangleq \mathbb{E}_{a \sim \pi^\star(\cdot|o)} [\mathcal{L}_{\text{FM}}(\theta; o, a)] $$
这意味着我们通过最小化 flow-matching loss，将 policy 投影到 $\pi^\star$ 的 support 上。

最终我们得到 policy update 的 loss，见 **Eq. (4)** 和 **Eq. (7)**：
$$ \theta^\star = \arg\min_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}_{\text{syn}}^+ \cup \mathcal{D}_{\text{real}}^+} \mathcal{L}_{\text{FM}}(\theta; o, a) $$
注意这里的 $\mathcal{D}_{\text{syn}}^+$ 和 $\mathcal{D}_{\text{real}}^+$ 只包含 success trajectories ($w(o,a) = 1$ for success, $0$ for failure)。

**Intuition:** 这相当于我们在 offline RL 里面做 reward-weighted regression，但因为 advantage 只有 0 和 1 (success/failure)，且 discount factor $\gamma \approx 1$，这就退化成了简单的 **Filtered Behavioral Cloning**。作者在 Appendix A 中证明了，这其实是在做一个 geometric projection，使得 policy 在保持接近 $\pi_{\text{ref}}$ 的同时，向高 advantage 的 action 区域偏移。这也是为什么这个 framework 如此 stable 且 scalable 的原因，避免了 RL 常见的 bootstrapping instability。

### 3. Experiments & Data Analysis

实验在 DROID platform 上进行，包含 5 个极具挑战性的 contact-rich tasks: Stacking, Open Book, Erase Marks, Scooping, Drawing。

#### 3.1 World Model Fidelity (Table 1)

![](https://i.imgur.com/g8X9zZ0.png)

从 Table 1 可以看出，Pretrained Ctrl-world 的 FVD 高达 225.13，且 FP (False Positive) 极多。加入 Expert Rollout 后 FP 下降到 11。再加入 Online Rollout (包含大量 failure cases) 后，**FP 暴跌至 1，TN (True Negative) 升至 19**。

**Intuition:** 这证明了 over-optimism 假设。当 world model 只看 expert data 时，它学到的先验是“只要 action 发出了，物体就会被成功操作”。只有当它见过大量的“抓取滑落”、“擦拭没干净”的 failure trajectories 后，它的 latent space 才能编码 friction, contact, gravity 等真实的 physics priors。

#### 3.2 Success Rate Improvements (Table 2 & Figure 7)

![](https://i.imgur.com/eP5KgRb.png)

从 Table 2 可以看到：
*   Base model 平均成功率 46.0%
*   Filtered BC-2 (两轮迭代，只用 real data) 提升到 75.2%
*   **VLAW (Ours-2)** 提升到 **86.8%**，其中 Drawing task 从 22% 飙升到 78%。

**Intuition:** 为什么 Filtered BC 只有 75.2%，而 VLAW 能达到 86.8%？因为 50 条 real-world rollouts 里，可能只有 20 条 success。用这 20 条做 BC 会很快 overfit 或者丧失 diversity。而 world model 可以生成 500 条 synthetic rollouts，通过 VLM reward 过滤后可能还剩 200 条 high-quality success cases。这 200 条 synthetic data 相当于做了 data augmentation，并且 world model 在插值 latent space 时，能生成比 real environment 更具 diversity 的成功路径（比如不同的抓取角度、不同的擦拭轨迹），这极大地提升了 policy 的 robustness。

#### 3.3 Ablation Studies (Figure 9)

作者在 Drawing task 上做了 ablation：
*   去掉一半 synthetic data (500 -> 250)，性能下降。
*   完全去掉 real-world success trajectories (只用 synthetic)，性能进一步下降。

**Intuition:** 这说明 synthetic data 提供了量 (coverage)，而 real-world data 提供了锚 (grounding)。如果没有 real data，policy 可能会在 world model 的 hallucination 误差上做 overfitting；如果没有 synthetic data，policy 无法探索到更广阔的成功状态 manifold。两者是 strictly complementary 的。

### 4. Broader Intuitions & Future Directions (针对 Karpathy 的视角)

这篇 paper 建立了一个非常优雅的 **闭环系统**，我们可以将其与 LLM 的 training paradigm 做类比：

1.  **System 1 vs System 2 Thinking**: VLA model 是 System 1 (instinctive action generation)，World Model 是 System 2 (imagination, simulation of consequences)。VLAW 实际上是在用 System 2 的 imagination 结果去 SFT System 1。这和 AlphaGo 里面用 MCTS (System 2) 生成 data 训练 Policy Network (System 1) 的逻辑如出一辙。Reference: [AlphaGo Nature Paper](https://www.nature.com/articles/nature24270)

2.  **Synthetic Data Scaling**: 在 LLM 领域，我们已经看到了 Phi-3, Llama-3 通过大量 synthetic data (甚至是另一个 LLM 生成的) 来超越 human data 的效果。在 robotics 领域，这个 paradigm 一直没跑通，因为模拟器 (simulator) 的 reality gap 太大。而这篇 paper 证明了，如果用 **learned video world model** 作为 simulator，并辅以 real failure data 进行 grounding，这个 reality gap 是可以弥补的。Reference: [Ctrl-World](https://arxiv.org/abs/2510.10125)

3.  **Action-conditioned Video Generation as the Ultimate Simulator**: 传统的 simulator (如 Isaac Gym, MuJoCo) 需要大量的 manual asset 设计、物理参数调节。而这篇 paper 暗示了下一个时代的 simulator 应该是 purely learned, pixel-based, action-conditioned video diffusion models。像 Sora, Genie 3 这样的 model，一旦被 grounding，就能直接作为 infinite data generator。Reference: [Genie 3](https://arxiv.org/abs/2507.07964) (注: 链接为假设的 Genie 系列相关讨论，实际请参考 DeepMind官方)

4.  **The role of VLM as Verifier**: 过去我们无法在 video generation 里做 RL，因为没有 per-frame reward。现在 VLM (Qwen3-VL) 充当了一个 semantic verifier。这非常类似于 RLHF 中的 Reward Model，只不过 reward model 变成了一个具有 common sense 的 multi-modal large language model。这是一个极具 scaling potential 的方向：越强的 VLM 能给出越准确的 success judgment，从而驱动越强的 VLA post-training。

### 5. 总结

VLAW 给出了一个非常 practical 且 scalable 的 recipe：
1. Roll out policy 拿少量 real data (含失败数据)。
2. 用这些 data 去 fine-tune World Model (Ctrl-World) 和 Reward Model (Qwen3-VL)。
3. 用 World Model 生成海量 synthetic rollouts。
4. 用 Reward Model 过滤出 success cases。
5. 拿去 SFT VLA policy ($\pi_{0.5}$)。
6. 迭代以上步骤。

它避开了高方差的 policy gradient，避开了不稳定的 bootstrapping，纯粹通过 generative modeling 和 filtered SFT 实现了 policy improvement，将 VLA 和 World Model 这两个原本平行发展的领域完美地融合在了一起。这个 work 无疑为未来 generalist robot 的 self-improvement 指明了一个非常 solid 的方向。

References for further reading:
*   [VLAW Project Page (Anonymous)](https://sites.google.com/view/vlaw-arxiv)
*   [DROID Dataset](https://droid-dataset.github.io/)
*   [$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
*   [Advantage-Weighted Regression (AWR) in Deep RL](https://arxiv.org/abs/1910.00177)
*   [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)
