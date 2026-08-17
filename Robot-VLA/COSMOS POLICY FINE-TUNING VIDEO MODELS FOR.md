---
source_pdf: COSMOS POLICY FINE-TUNING VIDEO MODELS FOR.pdf
paper_sha256: 05696475c9b5481afbf88673a65f80a46cf3fb87a5755c1410e83d0f468cc6e0
processed_at: '2026-08-03T17:34:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，我们用“人话”来拆解这篇 Cosmos Policy，我会尽量用直白的类比把这里的 intuition 给你 build 起来，同时保留技术细节的深度。

### 1. 核心直觉：把 Robot 当成“会画画的导盲犬”

想象你有一个非常擅长“脑补”未来画面的 video diffusion model（Cosmos-Predict2）。以前大家觉得它只会做视频，要想让它控制 robot，得在它屁股后面接一个专门的“动作翻译器”。但 NVIDIA 这帮人说：**搞那么复杂干嘛？直接把机械臂的动作当成一种“特殊的颜色”或者“特殊的画面”画进视频里不就行了。**

这就是 **Latent Frame Injection** 的核心。你的 video 序列本来是一帧帧图像，现在我们在图像序列里硬塞进去几个“假的帧”：
*   一帧代表机械臂当前姿态
*   一帧代表未来要执行的动作
*   一帧代表未来的“得分”

塞进去的时候有个工程 trick：机械臂动作只有 14 维，但图像的 latent volume 是 $H' \times W' \times C'$（比如 $20 \times 20 \times 16$）。维度对不上怎么办？直接把 14 维的 action 向量 **copy paste 铺满** 整个 tensor。Diffusion model 根本不在乎，它只是看到一个 volume 然后去 denoise。对于 network 来说，这就相当于在所有 spatial 位置上预测同一个常量，这是一个 perfectly well-posed 的数学问题。

### 2. 那个救命的 Noise Trick：为什么生成动作不能像生成图片那样？

这是 paper 里极具 insight 的一个点。Diffusion model 生成东西，是从一个纯噪声 $\sigma_{max}$ 开始，一步步把噪声去掉，最后得到干净的数据 $\sigma_{min} \approx 0$。

Base Cosmos video model 训练时，噪声分布是 log-normal，偏爱低噪声。这很好理解：生成视频时，高噪声阶段只需要画个大轮廓，低噪声阶段才需要抠细节（比如毛发、纹理），所以网络要多练低噪声。

**但在 Robot Action 里，这会死人的。**
如果你动作差了 1 厘米，机械臂直接把桌子砸了，或者没抓住东西，后面的轨迹全废了。这就是 **Cascading Errors**。

**Cosmos Policy 的解法：** 
强行改训练时的噪声分布，塞进 30% 的高噪声 uniform distribution，让网络把生成动作“大方向”的本事练好。
更绝的是在 inference 时，它把截止的 $\sigma_{min}$ 从 $0.002$ 提高到了 $4$。

**Intuition:** 当 $\sigma$ 极小快接近 0 时，信噪比极低，网络其实是在对一个微小的随机扰动做过度反应，产生高频抖动。对于图片这是磨皮细节，对于机械臂控制这就是帕金森震颤。在 $\sigma=4$ 处截断，相当于取了一个 risk-averse 的平滑动作，机械臂瞬间变稳了。

### 3. 数学公式拆解：EDM 去噪与 Value Function

来看它的 base loss function：
$$ \mathcal{L}(D_{\theta}, \boldsymbol{\sigma}) = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \mathbf{n}} \left[ \| D_{\theta}(\mathbf{x}_0 + \mathbf{n}; \boldsymbol{\sigma}, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right] $$

**变量拆解：**
*   $D_{\theta}$: 就是你那个巨大的 Diffusion Transformer，参数是 $\theta$。
*   $\boldsymbol{\sigma}$: 当前的噪声强度。上标无，下标无，就是一个标量。
*   $\mathbf{x}_0$: 干净的目标数据。在这里它包含了图像的 latent、action 的 latent、value 的 latent。
*   $\mathbf{c}$: Text condition（任务指令），用 T5-XXL 编码的。
*   $\mathbf{n}$: 从 $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ 采样的 Gaussian noise。

它预测的 Value function 用了 Monte Carlo return：
$$ V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi} \left[ \gamma^{H-t} R(s_H, a_H) \mid s_t = s \right] $$

**变量拆解：**
*   $V^{\pi}(s)$: Policy $\pi$ 在状态 $s$ 下的预期得分。
*   $\tau \sim \pi$: 顺着 policy $\pi$ 滚出去的轨迹。
*   $\gamma$: Discount factor，衰减系数。
*   $H$: 总的时间步 horizon。
*   $t$: 当前时间步。
*   $R(s_H, a_H)$: 最后一步的 terminal reward（任务成功得 1，失败得 0）。通过 $\gamma^{H-t}$ 把这个终局奖励往回折算。

### 4. Joint Training：逼着网络“带着脑子做动作”

普通的 Imitation Learning 就是输入图像，输出动作 $p(a|s)$。Cosmos Policy 觉得这太 lazy 了。

它搞了个 **Joint Training**，每一步训练 batch 切成三份：
1.  **50% 练 Policy:** 输入 $s$，输出 $a, s', V(s')$。强迫网络在给动作的同时，还要“脑补”做完动作后的画面 $s'$ 和得分 $V(s')$。
2.  **25% 练 World Model:** 输入 $s, a$，输出 $s', V(s')$。
3.  **25% 练 Value Function:** 输入 $s, a, s'$，输出 $V(s')$。

**Intuition 构建：** 
这就是 Auxiliary Supervision。如果网络只需输出 $a$，它可能会学到一种 short-cut mapping（比如看到杯子在左边，手就往左移）。但如果强迫它同时预测 $s'$（未来的画面），它就必须在内部 representation 里 encode 物理规律，比如“手碰到杯子会把它推走”。这让它变成了一个自带 World Model 的 agent。Ablation study 里，去掉这个 auxiliary future state prediction，LIBERO 成绩掉了一大截。

### 5. 真实世界 Planning：在脑子里“多想几步”

这篇 paper 还搞了个 Test-time Planning。
先用人类 demonstration 数据训出 base Cosmos Policy。
然后把这个 policy 放到真实环境里跑，收集一堆 rollout data（里面有很多失败的操作）。
用这些失败+成功的数据，再 fine-tune 出一个 **Planning Model**。这个 Planning Model 特别擅长评估：“如果在当前状态下做 action $a$，未来的画面 $s'$ 会是什么样？得分 $V(s')$ 是多少？”

执行任务时（Best-of-N Sampling）：
1.  Base policy 一次性生成 8 个候选动作。
2.  Planning Model 在脑海里分别模拟这 8 个动作的未来画面。
3.  给这 8 个未来画面打分。
4.  执行得分最高的那个动作。

为了稳，它还对未来画面预测 3 次，对得分预测 5 次，取个 majority mean 的 ensemble。这就类似于你下棋时在脑子里推演了好几个分支，挑胜率最高的那步走。在难搞的 ziploc 袋任务上，这一招硬生生把成功率拔高了 12.5%。

### 6. 疯狂联想：Software 2.0 的终极形态

顺着 Karpathy 的 Software 2.0 逻辑往下推，这篇 paper 给了我极大的震撼。

**1. Modality 边界彻底消失：** 
我们以前总觉得 image 是 image，action 是 action，proprioception 是 proprioception。我们要设计不同的 encoder 和 decoder。Cosmos Policy 告诉你，不用。只要你能把它们塞进同一个 tensor shape，Transformer 就能把它们当成同一种东西去 model。Action 本质上就是“环境状态随时间变化的导数”，它和 video 里的 optical flow 没有本质区别。这暗示着未来一切 modality 都会统一在 token/latent 的 sequence modeling 里。

**2. Robotics = Next-Token Prediction：**
大家都在找 Robotics 的 GPT moment。这篇 paper 指了一条路：不要去 train 一个 action-specific model。去 fine-tune 一个已经见过几千万小时物理世界视频的 World Foundation Model。这个 base model 已经懂了重力、碰撞、物体持久性。你只需要用极少量的 robot demonstration data（比如 50 个 demos），让它把这种“物理直觉”映射到具体的 motor command 上。RoboCasa 上只用 50 个 demo 打败用 300 个 demo 的 GR00T-N1.5，这就是 pretraining 力量的证明。

**3. 延展思考：计算的浪费与进化：**
Latent Frame Injection 里把一个 14 维向量 copy 成一个 $20 \times 20 \times 16$ 的 volume，这在计算上是极度奢侈的。Attention 机制在那边算一堆一模一样的 copy。但这恰好是 deep learning 的美妙之处：我们用 compute 换取了 architectural simplicity。未来可能我们会看到一种 hybrid 架构：图像走 VAE tokenizer，action 走 lightweight adapter 直接进 attention 的 K/V 空间。但现阶段，为了复用万亿参数的 video pretraining 权重，这种计算浪费是完全值得的。

**4. Test-time Scaling 在 Robotics 的降临：**
LLM 里我们用 Best-of-N, Beam Search, Tree of Thoughts 来提升推理能力。Cosmos Policy 把 Best-of-N 搬到了 Robotics。以前 robot 控制是 reactive 的，现在它变成了 deliberative。你给它 8 个 GPU，它能在 5 秒内想清楚 2 秒后该干啥。虽然现在还很慢（5 秒延迟），但这绝对是指明了未来的方向：Robot 的智力也会随着 inference compute 的增加而 scale up。

**Reference Links:**
*   Cosmos Policy Project Page: https://research.nvidia.com/labs/dir/cosmos-policy/
*   Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575
*   EDM Diffusion Formulation: https://arxiv.org/abs/2206.00364
*   Wan2.1 Video Model: https://arxiv.org/abs/2503.20314

---

这篇 NVIDIA 与 Stanford 合作的 paper 《COSMOS POLICY: FINE-TUNING VIDEO MODELS FOR VISUOMOTOR CONTROL AND PLANNING》非常精彩，它展示了一种极度简洁且极具启发性的方法，将预训练的 video foundation model 直接转化为 state-of-the-art 的 robot policy。作为 Andrej，我深知将 high-dimensional video generation 与 low-level control 结合的痛点。这篇 paper 的核心哲学是：**不要为 robot action 去设计新的网络结构，直接把 action 当作 video latent frame 的一部分，让 diffusion model 一起生成。**

下面我为你做深度的技术拆解，试图 build your intuition 关于它为什么 work。

---

### 1. 核心架构哲学：Latent Frame Injection

传统的 robot learning 通常会遇到 modality mismatch 问题：video model 处理的是 pixel/latent space，而 robot control 需要处理 low-dimensional 的 action space 和 proprioception。以前的 work（如 UVA 或 Video Policy）往往需要设计 inverse dynamics model 或者额外的 action diffuser。

Cosmos Policy 提出了 **Latent Frame Injection**。它的 base model 是 Cosmos-Predict2-2B，一个基于 Wan2.1 VAE 的 latent video diffusion model。Wan2.1 tokenizer 会将 raw video 压缩成 latent sequence。Cosmos Policy 的做法是：在一个 video 的 latent frame sequence 中，直接插入额外的 latent frame 来表示 robot proprioception、action chunk 和 state value。

**Intuition 构建：** 
你可以把 video diffusion model 想象成一个在 high-dimensional latent space 中进行 score matching 的通用概率推演引擎。只要你能把你的数据变成它认识的 shape（即 $H' \times W' \times C'$ 的 latent volume），它就能用同样的 denoising 机制去建模这个数据的分布。对于 1D 的 action vector（比如 14维的 joint angle）或者 scalar 的 value，paper 的做法是将其 normalize 到 $[-1, 1]$，然后进行 **duplication**，把这个低维向量复制铺满整个 latent volume。网络在 denoising 时，相当于在所有的 spatial positions 上预测同一个值，这在数学上完全是一个 well-posed 的 diffusion 任务。

**序列排列方式：**
对于双臂 ALOHA（3个相机），它的 11 个 latent frame 序列排列为：
$(blank, s_{proprio}, img_{wrist}, img_{3rd\_person\_1}, img_{3rd\_person\_2}, a, s'_{proprio}, img'_{wrist}, img'_{3rd\_person\_1}, img'_{3rd\_person\_2}, V(s'))$
这种排列方式完美契合了 causal/temporal 的 left-to-right autoregressive 逻辑，即 $(s, a, s', V(s'))$。

---

### 2. 数学公式与底层原理

Cosmos Policy 的 base model 采用 EDM (Elucidating the design space of Diffusion Models) 的去噪得分匹配框架。

**基础 Diffusion Loss：**
$$ \mathcal{L}(D_{\theta}, \boldsymbol{\sigma}) = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \mathbf{n}} \left[ \| D_{\theta}(\mathbf{x}_0 + \mathbf{n}; \boldsymbol{\sigma}, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right] $$

**变量与上下标详解：**
*   $D_{\theta}$: Denoiser network (Diffusion Transformer, 参数为 $\theta$)。
*   $\boldsymbol{\sigma}$: Noise level (标量，表示当前添加的噪声标准差)。
*   $\mathbf{x}_0$: Clean VAE-encoded image/modalities sequence（原始干净的 latent frame）。这里的 shape 对于 video 是 $(1+T') \times H' \times W' \times 16$。$T'=T/4$ 表示时间维度压缩了4倍，$H', W'$ 是空间压缩（通常为 $1/8$）。
*   $\mathbf{c}$: Text conditioning (由 T5-XXL 编码的 task description embedding，通过 cross-attention 注入)。
*   $\mathbf{n}$: i.i.d. Gaussian noise，从 $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ 中采样，用于 corrupt $\mathbf{x}_0$。

**Value Function 公式：**
由于使用 sparse reward（只有 terminal reward $R(s_H, a_H) \in [0, 1]$），Value function 退化为 Monte Carlo return：
$$ V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi} \left[ \gamma^{H-t} R(s_H, a_H) \mid s_t = s \right] $$

**变量详解：**
*   $V^{\pi}(s)$: Policy $\pi$ 在状态 $s$ 的 value function。
*   $\tau \sim \pi$: 从 policy $\pi$ 采样得到的 trajectory。
*   $\gamma$: Discount factor（折扣因子），将 terminal reward backpropagate 到前面的 time step。
*   $H$: Time horizon（总时间步数）。
*   $t$: 当前 time step。
*   $R(s_H, a_H)$: Terminal reward。

---

### 3. 关键 Trick：Noise Distribution Shift

这是这篇 paper 中极具 engineering 洞察力的一个点。Base Cosmos video model 的噪声采样分布是 log-normal：
$$ \ln(\sigma) \sim \mathcal{N}(P_{mean}, P_{std}^2) $$
其中 $P_{mean} = 1.39, P_{std} = 1.2$。这个分布将大部分训练权重放在了 low noise level（$\sigma$ 较小）上。

**Intuition 构建：** 
为什么 video generation 喜欢低噪声？因为在高噪声（$\sigma_{max}=80$）时，图像只是一片模糊的色块，网络只需要预测大概的 layout；而在低噪声时，网络需要预测精细的 texture 和 coherent 的细节。对于 video，高频细节的 loss 是巨大的，所以需要更多的 low-$\sigma$ 训练。

但对于 **Robot Action** 生成，这灾难性的。Diffusion 生成是从 $\sigma_{max}$ 逐步降到 $\sigma_{min}$ 的。如果网络在 high-$\sigma$ 阶段（生成初期）训练不足，它给出的 initial denoising 方向就是错的。Video 错一点只是稍微模糊，Action 错一点机械臂就会撞桌子，导致 cascading errors。

**Cosmos Policy 的解决方案：**
将分布改为 hybrid log-normal-uniform：0.7 概率采样原 log-normal，0.3 概率采样 uniform distribution $\mathcal{U}(1.0, 85.0)$。强行拉高 high-$\sigma$ 尾部的权重。

同时，在 inference 时，将 $\sigma_{min}$ 从原来的 $0.002$ 提高到 $4$。
**Intuition:** 在 $\sigma \approx 0$ 时，信噪比极低，网络本质上是在瞎猜。对于 action，最后那几步去噪带来的不是精度，而是噪声。在 $\sigma=4$ 处截断，相当于取一个 expected risk-averse 的平滑解，这极大提高了 action 的 robustness。

---

### 4. Joint Training 架构与 MDP 融合

Cosmos Policy 将 Policy、World Model 和 Value Function 统一在一个 network 里，通过 conditioning mask（在 latent sequence 上决定哪部分加噪，哪部分作为 condition）来实现不同任务的训练。

每一步训练，batch 被切分为：
*   **50% Policy Training:** 目标是 $p(a, s', V(s') | s)$。Condition 是 $s$。
*   **25% World Model Training:** 目标是 $p(s', V(s') | s, a)$。Condition 是 $s, a$。
*   **25% Value Function Training:** 目标是 $p(V(s') | s, a, s')$。Condition 是 $s, a, s'$。

**Intuition 构建：** 
这里最精妙的是 **Auxiliary Targets** 的作用。Policy 不仅仅预测 action，还要预测未来的 state $s'$ 和 value $V(s')$。这强迫 representation 在生成 action 时，必须 internalize action 的 causal effect（即做了这个 action，世界会变成什么样）。这相当于把 model-based RL 的 forward dynamics 和 policy learning 绑定在同一个计算图里，共享了所有的 backbone weights。从实验结果看（Table 4），去掉 auxiliary losses 导致 LIBERO 成功率下降 1.5%，去掉 pretrained model 下降 3.9%。

---

### 5. Test-time Planning 与 Best-of-N Sampling

这是 paper 的另一个亮点：将 rollout data 加入训练，实现 model-based planning。

**数据流转：**
1. 用 demonstration 训练 base policy。
2. 部署 base policy 收集 rollout data（包含成功和失败的轨迹）。
3. 用 rollout data fine-tune 出一个 "Planning Model"（强化了 World Model 和 Value Function 的能力，90% batch 用于 World Model/Value，10% 用于 Policy）。
4. Test-time: 用 base Policy 生成 N 个 candidate action chunks，用 Planning Model 预测每个 action chunk 导致的 $s'$ 和 $V(s')$，选择 $V(s')$ 最高的 action 执行。

为了增强 robustness，它还使用了 ensemble 机制：对每个 action 预测 3 次 future state，对每个 future state 预测 5 次 value，总共 15 个 value prediction。聚合方法是 "majority mean"（先判断 majority 预测 success 还是 fail，然后在该 group 内取平均）。

**Intuition 构建：** 
Demonstration data 只有 positive samples，这导致 Value function 无法识别 failure。Rollout data 引入了 negative samples（由于 robot 控制误差导致的失败），使得 World Model 能够预测 "如果抓不住 ziploc bag 的边缘，未来图像会是什么样"，Value function 能够给这种未来打低分。通过 Best-of-N search，Policy 实际上是在进行一种 Test-time MCTS（Monte Carlo Tree Search）的浅层近似，利用计算量换取 success rate。实验表明，在困难的 ziploc bag 任务上，这种 planning 带来了 12.5% 的显著提升。

---

### 6. 实验数据解析

**Table 1: LIBERO Simulation Results**
*   Cosmos Policy 达到 98.5% 平均成功率。
*   值得注意的是 LIBERO-Object 达到了 100.0%。
*   超越了强大的 VLA models 如 $\pi_{0.5}$ (96.9%) 和 CogVLA (97.4%)。这证明了 video pretraining 带来的 spatiotemporal priors 在空间理解和物理互动上优于纯 image-text pretrained VLA。

**Table 2: RoboCasa Simulation Results**
*   Cosmos Policy 只用了 **50 个** human demos，达到了 67.1% 的 SOTA。
*   其他方法如 GR00T-N1.5 用了 300 个 demos 才达到 64.1%，Video Policy 用了 300 个达到 66.0%。
*   这显示了 Video foundation model 极高的 data efficiency。

**Table 3: ALOHA Real-World Results**
*   在四个高难度双臂任务上，Cosmos Policy 综合得分 93.6%。
*   在 "put candies in bowl" 这种高 multimodality 任务上（糖果散落，抓取顺序任意），Cosmos Policy 得分 89.6%，而 $\pi_{0.5}$ 为 98.7% (但在 OOD 上只有 90.0%)。
*   在 "put candy in ziploc bag" 这种高精度任务上，Cosmos Policy 得分 85.4%，完胜 $\pi_{0.5}$ 的 61.5%。

---

### 7. 联想与延伸

*   **Software 2.0 in Robotics:** 这篇 paper 是 Karpathy 提出的 Software 2.0 理念在 Robotics 领域的极致体现。不再有人工设计的 PID 控制、state estimator 或独立的 inverse kinematics 求解器。所有的东西——perception, planning, control, value estimation——都变成了同一个 Neural Network 在 latent space 里的 denoising 过程。
*   **Duplication 机制的计算浪费与未来演进：** Latent Frame Injection 中的 duplication（把 14维 action 复制成 $H' \times W' \times C'$ 的 volume）在计算上是非常浪费的。这相当于让 attention 机制在处理大量的 redundant tokens。未来的 work 可能会采用混合 token 的方式：image 走 VAE tokenization，action 走一个 lightweight MLP projection 映射到 1-2 个专用 tokens，然后在 Transformer 内部进行 cross-attention。但 paper 证明了，为了不修改 base model 架构，这种 waste 是值得付出且可以 scale 的。
*   **从 $\pi_0$ 到 Cosmos Policy：** Physical Intelligence 的 $\pi_0$ 使用 Flow Matching 生成 action，本质上是 continuous token 的 generation。Cosmos Policy 则更进一步，将 action 完全视为 video 的一种 modality。这暗示着未来 Robotics 的 Foundation Model 可能不再区分 VLA (Vision Language Action) 和 VVM (Vision Video Model)，一切皆是 Sequence Modeling。
*   **Model-based RL 的复兴：** 通过 latent space 的 world model 进行 planning，让人联想到 Dreamer 系列。区别在于 Dreamer 是在 RL loop 中 online 训练，而 Cosmos Policy 是通过 offline imitation + rollout data 的 semi-supervised 形式逼近类似的效果。这种 best-of-N search 的 test-time scaling 很可能会成为后续 robotics 的标配，类似于 LLM 中的 best-of-N sampling 或 beam search。
*   **与 NVIDIA Cosmos Ecosystem 的协同：** NVIDIA 正在构建一个 Physical AI 的 ecosystem。Cosmos-Predict2 是其 World Foundation Model。Cosmos Policy 展示了如何在这个 World Model 上“生长”出 Policy。这类似于 LLM 中先 pretrain 一个 base model，然后通过 RLHF/instruction tuning 让它遵循指令。Rollout data 的 fine-tuning 机制在概念上非常接近 RLHF。

**Reference Links:**
*   NVIDIA Cosmos Research Page: https://research.nvidia.com/labs/dir/cosmos-policy/
*   Cosmos World Foundation Model Paper: https://arxiv.org/abs/2501.03575
*   Wan2.1 Video Generative Models: https://arxiv.org/abs/2503.20314
*   EDM Paper: https://arxiv.org/abs/2206.00364
*   $\pi_0$ Paper: https://arxiv.org/abs/2410.24164

这篇 paper 极度推荐精读，它不仅是一个 SOTA 的工程实现，更是一种将 Robotics 视为纯粹的 Generative Modeling 哲学宣言。
