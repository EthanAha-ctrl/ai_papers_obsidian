---
source_pdf: EXPO.pdf
paper_sha256: 7d6328a359b62c084000e5a2fd582abfbb47c483c008788dfff602de1c1f9a14
processed_at: '2026-08-04T06:20:49-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EXPO 用人话讲

## 这篇论文在解决什么问题？

想象 diffusion policy 是一个**会画画的机器人**，它从一团noise开始，一步步擦除noise，最后画出一个action（动作）。这就像一个画家从模糊的草稿开始，10笔之内画出清晰的作品。

现在你希望这个画家**画得更值钱**——也就是让动作的reward更高。传统RL的做法是告诉画家："你最后那幅画卖得不好，回去把每一笔都改一改。"

问题是画家画了10笔，gradient要从最后那幅画传回到第1笔，传到中间要么消失了，要么爆炸了。画家就崩溃了，越改越差。

这就是diffusion policy做online RL fine-tuning的核心痛点：**chain太长，gradient不稳定**。

## EXPO 的三招

### 第一招：Base Policy 只管模仿，不管reward

让diffusion policy继续跟着demonstration学模仿学习。它照着demo画，画多风格的画（multi-modal），根本不去碰Q-value。

这就像画家只管跟着菜谱做菜，做川菜、粤菜、日料都会，不去想哪道菜最赚钱。**稳定，不会崩。**

### 第二招：小助手做微调

雇一个**小厨师**（edit policy），是个简单的高斯分布。大厨师做完菜，小厨师在菜上做小修改：盐多一点、辣少一点。修改幅度限制在 $\beta$ 范围内，不能改太多——否则就变成另一道菜了。

小厨师的目标：让菜更好吃（Q-value更高），同时保持一定的随机性（entropy）去尝试新口味。

公式就是：
$$\max_{\hat{a}} \left[ Q(s, a + \hat{a}) - \alpha \log \pi_{\text{edit}}(\hat{a}|s, a) \right]$$

翻译成人话：找一个微调 $\hat{a}$，让edited action的Q值高，但又不要太确定，留点探索空间。

小厨师只有3层MLP，轻量级，gradient不需要穿过10步denoising chain，所以**训练稳如老狗**。

### 第三招：老板盲品选最好的

实际出菜时，大厨师做8道菜（sample N=8），小厨师对每道菜做微调，得到16个candidates。老板（on-the-fly policy）每道尝一口，选Q值最高的那道端给顾客。

$$\pi_{\text{OTF}} = \arg\max_{a \in \{a_1, \tilde{a}_1, a_2, \tilde{a}_2, \cdots, a_8, \tilde{a}_8\}} Q_\phi(s, a)$$

这一步**完全不需要backprop**，就是暴力采样+选最大，所以Q function一旦变了，behavior立即跟着变。不像传统policy gradient要慢慢更新参数才能align。

## 关键：TD backup也要用这个on-the-fly policy

这是EXPO和IDQL最大的区别。IDQL只在采样action时用"选最大Q"的策略，TD target还是用IQL那套implicit方法。EXPO在TD target也算一遍on-the-fly action，用这个action的Q值做backup。

为什么重要？如果只在sampling用，TD backup变成SARSA-like——跟着当前policy的action走，learning慢。用on-the-fly做TD backup才是真正的Q-learning，learning快。Ablation实验直接证明了这点（Figure 6）。

## 为什么这套分工能work？

| 角色 | 干什么 | 优化方式 | 稳不稳 |
|------|--------|----------|--------|
| Base policy | 学multi-modal behavior | Imitation learning | 稳 |
| Edit policy | 局部微调到高Q值 | 1-step gradient | 稳 |
| On-the-fly | 全局选最好的 | 非参数 argmax | 稳 |

三个稳定的组件拼起来，还是稳定。原来的做法是把"长denoising chain + value gradient"硬凑在一起，两个不稳定因素叠加，就崩了。

## 实验说了什么

12个任务，4个domain：
- **Antmaze**：四足机器人走迷宫
- **Adroit**：28自由度机械手转笔、开门、移球
- **Robomimic**：7自由度Franka抓物体
- **MimicGen**：7自由度Franka穿针、叠方块

所有任务都是**sparse reward**（只有成功不成功，没有中间reward）。

结果：
1. **Online从零学**：EXPO几乎全部超过RLPD（已知sample efficiency很强的方法），除了relocate-binary（dataset太窄IL学不出东西）
2. **Offline-to-online**：EXPO最大的优势是**从offline到online不掉performance**。其他方法像DAC会直接collapse，Cal-QL在简单任务好但在难任务慢，IDQL pretrain后online提升不了
3. **Ablation**：去掉edit policy、去掉TD backup的on-the-fly，性能都会明显掉

## $\beta$ 怎么选

- 任务有好的demo数据，$\beta$ 取小（0.05或0.1）——只需微调
- 任务需要探索找最优策略，$\beta$ 取大（0.5或0.7）——允许大改动

Adroit的pen-binary用0.7因为要探索，Robomimic用0.05因为是精细操作。

## 局限性

1. **计算贵**：每个batch example都要sample 8个action过Q network算Q值，还要过diffusion采样，UTD=20意味着每步环境交互要训练20次。N=8 × batch_size=256 × ensemble=10 critic = 一堆forward
2. **依赖先验**：offline dataset或pre-trained policy至少要有一个能用的，如果prior完全垃圾，EXPO也救不了
3. **没探索完全uninformed prior**的情况

## 一句话总结

EXPO把"让diffusion policy更值钱"这个不稳定问题，拆成了三个稳定子问题：**模仿学习保稳定，小网络做微调追Q值，暴力采样选最好的**。不硬碰denoising chain的gradient问题，而是绕过去。

## 相关阅读

- IDQL（最像的baseline）: https://arxiv.org/abs/2304.10573
- RLPD（sample efficiency对标）: https://proceedings.mlr.press/v202/ball23a.html
- Cal-QL（offline-to-online对标）: https://arxiv.org/abs/2303.05479
- DAC（diffusion + action gradient）: https://arxiv.org/abs/2405.20555
- Diffusion Policy原始paper: https://arxiv.org/abs/2307.01849
- QSM: https://arxiv.org/abs/2312.11752
- DPPO（另一条路，PPO on diffusion）: https://arxiv.org/abs/2409.00588

个人觉得这套思路很优雅——**遇到不稳定问题不要硬刚，绕过去用非参数方法bridge**。这种"divide and conquer"的设计哲学在RL method里很值得借鉴。想象把它推广到LLM RLHF场景：base model = pretrained LLM（保持stable generation），edit policy = small RL head（在token层面做small perturbation toward reward），on-the-fly = best-of-N selection。说不定能避免RLHF把base model的generation distribution搞坏的问题。

---

# EXPO: Stable Reinforcement Learning with Expressive Policies 深度解析

## 1. 论文核心动机与问题的本质

**Andrej** 这篇paper要解决一个非常具体且关键的问题：当我们已经有一个pre-trained的expressive policy（比如diffusion policy或flow-matching policy），想要用online RL进一步fine-tune时，会遇到gradient propagation的稳定性问题。

让我先build一下intuition。考虑一个diffusion policy $\pi_\theta(a|s)$，它通过 $T$ 步denoising chain生成action：
$$a_0 \sim \mathcal{N}(0, I), \quad a_{t-1} = f_\theta(a_t, s, t) + \sigma_t \epsilon, \quad a = a_T$$

如果要做类似TD3/SAC的policy improvement，需要计算：
$$\nabla_\theta Q(s, a_\theta(s)) = \nabla_\theta Q(s, f_\theta \circ f_\theta \circ \cdots \circ f_\theta(a_0, s))$$

这个gradient需要通过 $T$ 次denoising step反向传播，当 $T=10$ 甚至更大时，gradient要么vanishing要么exploding，导致policy training不稳定。这就是为什么直接对expressive policy做value maximization困难。

**EXPO的核心insight**：不直接优化expressive policy的value，而是构造一个"on-the-fly"的implicit policy来完成value maximization。

## 2. 方法架构详解

EXPO由三个核心组件构成：

### 2.1 Base Expressive Policy $\pi_{\text{base}}$

用稳定的imitation learning objective训练，论文用diffusion policy (DDPM)作为canonical example：

$$\min_\psi \mathbb{E}_{t \sim \mathcal{U}(\{1,\cdots,T\}), \epsilon \sim \mathcal{N}(0,I), (s,a) \sim \mathcal{D}} \left[ \| \epsilon - \epsilon_\psi(\sqrt{\bar{\alpha}_t} a + \sqrt{1-\bar{\alpha}_t}\epsilon, s, t) \| \right]$$

变量解析：
- $\psi$：diffusion model参数
- $t$：从uniform distribution $\{1, \ldots, T\}$ 采样的diffusion timestep
- $\epsilon$：标准Gaussian noise $\mathcal{N}(0, I)$，作为target
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$：累积noise schedule，$\alpha_i$是variance preserving scheduler中的coefficient
- $\sqrt{\bar{\alpha}_t} a + \sqrt{1-\bar{\alpha}_t} \epsilon$：前向noisy action
- $\epsilon_\psi(\cdot, s, t)$：noise prediction network，conditioned on state $s$ 和 timestep $t$

关键点：base policy **从不**直接被Q-function的gradient训练，这保证了它的稳定性。

### 2.2 Edit Policy $\pi_{\text{edit}}$

这是一个**轻量级Gaussian policy**，输入 $(s, a)$，输出action edit $\hat{a}$：

$$\tilde{a} \gets a + \hat{a} \tag{1}$$

其中：
- $a \sim \pi_{\text{base}}(\cdot|s)$：base policy采样的action
- $\hat{a} \sim \pi_{\text{edit}}(\cdot|s, a)$：edit policy输出的Gaussian noise
- $\tilde{a}$：refined action

Edit policy的training objective是标准SAC-style的entropy-regularized policy loss：

$$L(\pi_{\text{edit}}) = -\mathbb{E}_{(s,a) \sim \mathcal{D}, \hat{a} \sim \pi_{\text{edit}}(\cdot|s,a)} \left[ Q_\phi(s, a + \hat{a}) - \alpha \log \pi_{\text{edit}}(\hat{a}|s,a) \right] \tag{2}$$

变量解析：
- $Q_\phi(s, a + \hat{a})$：critic network $Q_\phi$ 对edited action的value估计
- $\alpha$：entropy temperature coefficient，控制exploration-exploitation balance
- $\log \pi_{\text{edit}}(\hat{a}|s,a)$：edit policy的log probability，用于entropy maximization
- 负号：因为我们要**maximize** $Q_\phi - \alpha \log \pi$，等价于**minimize** $-Q_\phi + \alpha \log \pi$

**Edit distance constraint**：为了防止edited action偏离base action太远，将 $\hat{a}$ clip到 $[-\beta, \beta]$ 范围。$\beta$ 是关键hyperparameter：
- $\beta$ 小（如0.05）：适合fine-grained manipulation任务（Robomimic、MimicGen）
- $\beta$ 大（如0.7）：适合需要exploration的任务（Adroit中的pen-binary）

### 2.3 On-the-Fly Policy $\pi_{\text{OTF}}$

这是EXPO最巧妙的design：一个non-parametric的implicit policy，用于action selection：

$$\pi_{\text{OTF}}(a|s, \pi_{\text{base}}, \pi_{\text{edit}}, \phi) = \arg\max_{a \in \bigcup_{i=1}^N \{a_i, \tilde{a}_i\}} Q_\phi(s, a)$$

其中：
- $a_i \sim \pi_{\text{base}}(\cdot|s)$：从base policy采样 $N$ 个action candidates（论文用 $N=8$）
- $\tilde{a}_i = a_i + \hat{a}_i$：每个 $a_i$ 经过edit policy后的refined action
- $Q_\phi(s, a)$：critic对action的评估
- $\arg\max$：选Q-value最高的action

**关键insight**：on-the-fly policy有两个用途：
1. **Environment interaction**：用 $\tilde{a}_t^* \sim \pi_{\text{OTF}}$ 采样action用于rollout
2. **TD backup**：用 $\tilde{a}_{t+1}^*$ 计算TD target

这两点都很重要——ablation study证明只用于sampling而不用于TD backup会导致性能大幅下降（类似SARSA而非Q-learning）。

### 2.4 Q-Function Update

Critic的TD loss：

$$\min_\phi \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ (r_t + \gamma Q_{\phi'}(s_{t+1}, \tilde{a}_{t+1}^*) - Q_\phi(s_t, a_t))^2 \right]$$

变量解析：
- $\phi$：critic network参数
- $\phi'$：target critic参数，通过EMA更新 $\phi' \leftarrow \rho \phi' + (1-\rho)\phi$，$\rho=0.005$
- $\gamma = 0.99$：discount factor
- $r_t$：immediate reward
- $\tilde{a}_{t+1}^* \sim \pi_{\text{OTF}}(\cdot|s_{t+1})$：on-the-fly policy选出的最优action
- $Q_{\phi'}(s_{t+1}, \tilde{a}_{t+1}^*)$：target Q-value
- $Q_\phi(s_t, a_t)$：current Q-value prediction

注意：这里TD target用的是 $a_t$（replay buffer中的action），而不是 $\tilde{a}_t^*$，这是标准的off-policy Q-learning formulation。

## 3. 完整Algorithm 1 流程解析

```
Require: Prior dataset D_data = {(s_i, a_i)}; optional π_base
Initialize: π_edit, Q_φ, Q_φ', UTD ratio G

while training:
    for each env step t:
        # Rollout phase
        Sample ã_t* from π_OTF(·|s, π_base, π_edit, φ')
        Take action ã_t*, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}) in replay buffer
        
        # Training phase (UTD=G)
        for g=1,...,G:
            Sample mini-batch (s, a, r, s') from buffer
            Sample ã*' from π_OTF(·|s', π_base, π_edit, φ')
            Compute target: y = r + γ Q_φ'(s', ã*')
            Update φ: minimize (y - Q_φ(s, a))^2
            Update target: θ' ← ρθ' + (1-ρ)θ
            Update π_base with supervised IL loss L_IL
            Update π_edit: maximize Q_φ(s, a+â) - α log π_edit(â|s)
```

**关键implementation details**（Table 1）：
- Q-Ensemble size: 10（用min over 2）
- N Action Samples: 8
- UTD Ratio: 20（high update-to-data ratio for sample efficiency）
- T (diffusion steps): 10
- Base policy: 3 residual blocks, hidden dim 256
- Edit policy: 3-layer MLP, hidden dim 256
- Dropout 0.1 only on Adroit（narrow dataset需要regularization）

## 4. 与Prior Work的关键区别

### vs. IDQL (Hansen-Estruch et al., 2023)
- IDQL也用diffusion policy + IL training + sample N actions选max Q
- **关键区别**：IDQL只在online exploration用implicit policy，TD backup用implicit Q-learning (IQL) objective，**没有action edit**
- EXPO在TD backup也用on-the-fly policy，且加入了edit policy做local value maximization
- Ablation证明这两个差异都很关键

### vs. RLPD (Ball et al., 2023)
- RLPD用simple Gaussian policy，UTD=20，oversample offline data
- EXPO用expressive policy + edit，能capture multi-modal behavior

### vs. DAC (Fang et al., 2024)
- DAC用action gradient指导diffusion denoising process
- 在offline pretraining表现好，但online fine-tuning会collapse
- EXPO避免直接gradient propagation，更稳定

### vs. Cal-QL (Nakamoto et al., 2023)
- Cal-QL用calibrated Q-function避免offline-to-online的performance drop
- 但用simple Gaussian policy，无法利用expressive policy的multi-modality

### vs. QSM (Psenka et al., 2023)
- QSM用Q-score matching训练diffusion policy
- 在challenging continuous control上unstable

### vs. Residual RL (Ankile et al., 2024; Yuan et al., 2024)
- 这些方法在expressive policy外加residual policy
- EXPO是on-the-fly action selection + edit，且用off-policy TD而非on-policy PPO

## 5. 实验结果深度分析

### 5.1 Benchmarks（12 tasks across 4 domains）

| Domain | Tasks | 特点 |
|--------|-------|------|
| Antmaze | medium-diverse, medium-play, large-diverse, large-play | Navigation, 4个任务 |
| Adroit | pen-binary, door-binary, relocate-binary | 28-DoF hand, high-dim, narrow dataset |
| Robomimic | Lift, Can, Square | 7-DoF Franka, manipulation |
| MimicGen | Threading, Stack | 7-DoF Franka, fine-grained |

所有任务都是**sparse reward**（binary success signal）。

### 5.2 Online RL Results（Q1）

Key findings：
- EXPO在几乎所有任务上超过RLPD，除了relocate-binary-v0（dataset太narrow，IL无法extract useful behavior）
- 相比IDQL和QSM，EXPO能从scratch学习，而这两个expressive policy baseline往往学不动
- sample efficiency提升2-3x

### 5.3 Offline-to-Online RL Results（Q2）

Key findings：
- **EXPO没有offline-to-online performance drop**——这是critical advantage
- 原因：base policy stays close to behavior distribution via IL，edit policy只做local refinement
- DAC虽然pretrain好但online collapse
- Cal-QL在easy tasks好，hard tasks差
- IDQL pretrain后无法online improvement（policy constrained too much）

### 5.4 Ablation Studies（Q3）

**Ablation 1: On-the-fly policy in TD backup**（Figure 6）
- 只在sampling用OTF vs. 同时在TD backup用OTF
- 结果：TD backup用OTF至关重要——否则变成SARSA-like，learning慢

**Ablation 2: Action edits**（Figure 7）
- With edit vs. without edit（只sample from base policy选max Q）
- pen-binary: 无edit收敛到suboptimal
- Square: 即使有good IL policy，edit仍然needed for continuous refinement

**Ablation 3: Offline dataset quality**（Figure 8）
- Varying number of demonstrations for Square
- Clear correlation: IL policy performance ↔ EXPO fine-tuning performance
- 这说明EXPO依赖于offline data的quality

**Ablation 4: No offline dataset after pretraining**（Figure 9）
- 用pre-trained policy rollout warm-start，不保留offline data
- EXPO仍然work，Cal-QL + SAC不行
- 说明pre-trained policy本身可以作为strong prior

## 6. Intuition Building: 为什么EXPO work？

### 6.1 Base Policy的角色：Stable Behavior Prior

Base policy通过IL训练，提供了**stable, multi-modal behavior distribution**。这个distribution：
- Captures multi-modal nature of demonstrations（diffusion的优势）
- Provides good initialization for exploration
- 不被Q-function的noise污染

### 6.2 Edit Policy的角色：Local Value Maximization

Edit policy解决了一个**local optimization problem**：
- 只在 $[-\beta, \beta]$ 范围内edit
- 相当于在每个mode内部做refinement
- Gaussian + entropy regularization提供within-mode exploration

### 6.3 On-the-fly Policy的角色：Global Mode Selection

On-the-fly policy解决**global optimization**：
- 从N个candidates（包括base和edited）中选max Q
- 跨mode selection
- 非参数化，立即反映Q-function变化

### 6.4 分工的美学

| Component | Optimization Scope | Parameterization | Stability |
|-----------|-------------------|------------------|-----------|
| Base policy | Behavior cloning (no value) | Expressive (diffusion) | Stable (IL) |
| Edit policy | Local value max | Light Gaussian | Stable (1-step) |
| OTF policy | Global mode selection | Non-parametric | Stable (no grad) |

这个分工避免了"长denoising chain + value gradient"的不稳定组合。

## 7. 局限性与Future Work

论文提到：
1. **Computational cost**：sampling N actions for TD backup expensive（每个batch example都要sample N actions）
2. **Prior assumption**：需要reasonable prior（offline data或pre-trained policy）
3. 完全uninformed prior的情况未探索

## 8. 相关References与深入阅读

**核心方法paper**：
- IDQL: https://arxiv.org/abs/2304.10573
- RLPD: https://proceedings.mlr.press/v202/ball23a.html
- Cal-QL: https://arxiv.org/abs/2303.05479
- DAC: https://arxiv.org/abs/2405.20555
- QSM: https://arxiv.org/abs/2312.11752

**Diffusion Policy基础**：
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2307.01849
- Flow Matching: https://arxiv.org/abs/2405.16173

**Offline-to-Online RL**：
- AWAC: https://arxiv.org/abs/2006.09359
- Cal-QL: https://arxiv.org/abs/2303.05479

**Expressive Policy RL**：
- Diffusion policies as expressive policy class: https://arxiv.org/abs/2208.06193
- Diffusion Policy Policy Optimization (DPPO): https://arxiv.org/abs/2409.00588
- Consistency models as policy class: https://openreview.net/forum?id=v8jdwkUNXb

**π0 (Black et al., 2024)** - 大规模robot foundation model，论文中提到expressive policy的重要性：
- https://arxiv.org/abs/2410.24164

## 9. 个人视角的Critique与延伸思考

**Strengths**：
1. 方法非常practical，不引入新的instability source
2. 对policy parameterization agnostic，可以apply到任何expressive policy
3. Ablation study做得很thorough，每个design choice都有验证

**Potential concerns**：
1. $\beta$ 的tuning可能task-dependent，论文虽然给了guidelines但仍有sensitivity
2. N=8 samples × ensemble=10 critic × UTD=20 = 大量computation
3. Edit policy的Gaussian可能limit了表达力——如果需要更复杂的edit怎么办？

**延伸联想**：
- 这个framework是否能扩展到LLM RLHF？base policy = pretrained LLM, edit policy = small RL head?
- On-the-fly policy selection类似于best-of-N sampling，这与recent work on process reward models有联系
- $\beta$ constraint类似于trust region / natural policy gradient的思想

这篇paper的contribution非常clean：通过decomposition把stable IL和unstable value maximization分离，用non-parametric selection bridge它们。这种"divide and conquer"的思路在RL method design中很优雅。
