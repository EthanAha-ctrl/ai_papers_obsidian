---
source_pdf: Learning Smooth Humanoid Locomotion.pdf
paper_sha256: 793807693ee23fffb76ba41299a36514c6d8fcec1ba88a31916fb02c5f84679a
processed_at: '2026-08-05T13:45:53-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LCP

好的Karpathy，我换个更接地气的讲法，从intuition出发，不堆公式。

## 1. 一句话版

**RL policy在sim里训练出来会"抽搐"（jittery），传统解法是加一堆罚款（smoothness reward）或者post-hoc装个"减速带"（low-pass filter），这篇paper说：直接告诉policy"你的输出对输入的敏感度不许超过某个上限"，一行公式搞定。**

## 2. 为什么RL policy会抽搐？——一个物理直觉

想象你在simulator里训练policy，simulator的motor是一个"理想电机"：你给它什么command，它瞬时输出什么torque，没有延迟、没有带宽限制、没有热饱和。

policy是一个optimization机器，它会**发现一个捷径**：在相邻两个timestep，给出截然相反的action（比如t时刻输出+5，t+1时刻输出-5），能更激进地驱动robot到目标状态，reward更高。这就是bang-bang control（LaSalle 1960 [https://www.sciencedirect.com/science/article/pii/S147466701770095X]）——最优控制理论里早就知道，unconstrained LQR的解就是bang-bang的。

**sim里policy觉得自己在开F1，到了real robot发现自己开的是拖拉机**。真实电机的带宽可能就几十Hz，你让它500Hz频繁切换方向，它要么tracking不上（PD controller跟不上），要么过热，要么直接跳闸。sim-to-real就崩了。

## 3. 传统解法为什么笨——用个比方

### 3.1 Smoothness Reward = 罚款制度

你在reward里加很多penalty term：
- "action变化太快，罚！"
- "joint速度太高，罚！"  
- "energy消耗太大，罚！"
- "action的三阶导数太大，罚！"

这就像**给policy贴罚单**。问题：
1. 每个robot的罚款额度都要重新调——Fourier GR1和Unitree H1的joint inertia差几倍，同一个"罚5块"的含义完全不同
2. 调权重是黑艺术，调不好就reward hacking——policy把joint锁死不动来avoid velocity penalty
3. 这些penalty是environment的一部分，**不可微**，policy gradient只能通过sampling估计，信号噪声比低

### 3.2 Low-pass Filter = 减速带

在policy输出后面挂一个filter $a_t = \alpha a_{t-1} + (1-\alpha)\pi(o_t)$。

这就像给policy装了个"慢半拍装置"。问题更糟：**policy发现自己的action被filter吃掉了，credit assignment错位**——它说"我要往左"，filter让它"慢慢往左"，policy看不到自己说话的效果，干脆瞎说，结果exploration被damped，policy学不好。

TABLE I(a)的数据验证了这点：low-pass filter的task return只有24.98，比LCP和smoothness reward都低 [https://lipschitz-constrained-policy.github.io]。

## 4. LCP的Elegant之处

### 4.1 核心insight——一句话点破

作者做了个motivating experiment（Fig. 3）：**有smoothness reward训练的policy，它的gradient norm $\|\nabla_s \log\pi(a|s)\|$天然就比没有的小很多**。

换句话说：smoothness reward本质上是在**implicit地约束policy的Lipschitz常数**。既然如此，为什么我们不直接explicit地约束它？

这是一个abstraction layer的提升——从"惩罚一堆symptom"上升到"约束root cause"。

### 4.2 Lipschitz是什么——用开车打比方

Lipschitz continuity的数学定义：
$$d_Y(f(x_1), f(x_2)) \leq K \cdot d_X(x_1, x_2)$$

变量解释：
- $f$：我们要约束的函数（policy）
- $x_1, x_2$：两个不同的input observation
- $d_X$：input之间的distance
- $d_Y$：output之间的distance
- $K$：Lipschitz常数——"敏感度上限"

**人话翻译**：policy这个"翻译官"对observation的敏感度有上限。observation变化1单位，action最多变化$K$单位。$K$越小，policy越"迟钝"，越smooth。

开车比方：Lipschitz常数就像是**限速牌**。它不限制你开去哪儿（task完成），只限制你瞬间的sensitivity（action变化率）。smoothness reward是每超速一次贴一张罚单，你还得等警察告诉你罚了多少才能调整。LCP是直接告诉你"限速60"，你训练时就主动学会不超速。

### 4.3 怎么实现——为什么这个form特别elegant

数学推导链：
1. **原始目标**（公式4）：约束所有state-action pair的gradient norm $\leq K$
2. **不可计算**：max over所有state不可能枚举
3. **近似1**（公式5）：用expectation替代max（借鉴TRPO [http://arxiv.org/abs/1502.05477]的heuristic）
4. **近似2**（公式6）：Lagrangian relaxation，引入multiplier $\lambda$
5. **工程简化**（公式7）：直接固定$\lambda_{gp}$作为hyperparameter

最终loss长这样：
$$\max_\pi J(\pi) - \lambda_{gp} \mathbb{E}_{s,a \sim \mathcal{D}}[\|\nabla_s \log\pi(a|s)\|^2]$$

变量解释：
- $J(\pi)$：RL原始目标，cumulative reward
- $\lambda_{gp}$：用户指定的penalty系数，paper里用0.002
- $\nabla_s \log\pi(a|s)$：log-likelihood对observation的gradient，衡量"observation微小扰动如何影响action分布"
- $\mathcal{D}$：从policy rollout收集的state-action pair

**为什么用$\log\pi$不用$\pi$**：因为log-likelihood的gradient是score function，在policy gradient estimator $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log\pi \cdot Q]$里天然出现，有information几何意义。

**为什么这个form elegant**：
1. **Differentiable**：gradient penalty本身可以backprop，不像smoothness reward要走sampling-based policy gradient，信号干净
2. **一行代码**：`loss += lambda_gp * (grad_norm ** 2).mean()`，torch.autograd直接搞定
3. **Morphology-invariant**：Lipschitz常数不依赖于具体的joint配置，是函数层面的抽象

## 5. 和WGAN-GP的神似——一个deep connection

这个form和WGAN-GP（Gulrajani et al. 2017 [https://arxiv.org/abs/1704.00028]）几乎一模一样。

WGAN-GP做什么？给discriminator加gradient penalty，让它满足1-Lipschitz条件，这样Wasserstein distance才有意义。

LCP做什么？给policy加gradient penalty，让它满足$K$-Lipschitz条件，这样output才smooth。

**底层数学是同一个**：Lipschitz continuity。不同domain，同一数学。WGAN-GP对GAN做的事，LCP对RL正在做。

这让我联想到一个更深的connection：adversarial robustness（Madry et al. [https://arxiv.org/abs/1706.06083]）。一个classifier的adversarial robustness等价于它的Lipschitz常数有界——如果input变化$\epsilon$时output最多变化$K\epsilon$，那对抗扰动就无法造成大改变。

所以**smoothness、Lipschitz、adversarial robustness其实是同一个数学对象的三个面相**：
- Locomotion视角：action对observation的sensitivity
- 函数视角：gradient norm有界
- Security视角：对抗扰动鲁棒

LCP本质是在policy里inject了"input robustness regularization"，恰好等价于output smoothness。这点paper没明说但很powerful。

## 6. 实验里几个值得品味的细节

### 6.1 $\lambda_{gp}$的sweet spot（TABLE I(b)）

| $\lambda_{gp}$ | Action Jitter ↓ | Task Return ↑ |
|---|---|---|
| 0.0 | 42.19 | 28.87 |
| 0.001 | 3.69 | 26.32 |
| **0.002** | **3.21** | **26.03** |
| 0.005 | 2.10 | 23.92 |
| 0.01 | 0.17 | 16.11 |

这个曲线太经典了——**正则化的U型**：
- 太小没用，policy仍然jittery
- 太大瘫痪，policy变sluggish
- sweet spot在0.002

最key的发现是：**同一个0.002在4个不同humanoid上都work**（Fourier GR1T1/T2、Unitree H1、Berkeley Humanoid）。这是smoothness reward做不到的——smoothness reward的每个weight都要per-robot retune。

**为什么？** 我的intuition：Lipschitz常数是函数层面的抽象，不依赖于具体的joint configuration。smoothness reward的"penalize joint velocity"在inertia差10倍的robot上含义完全不同；但"penalize gradient norm"是对policy mapping本身的约束，与morphology解耦。

### 6.2 GP应用位置的ablation（TABLE I(c)）

| GP applied to | Action Jitter ↓ |
|---|---|
| Whole obs (含history) | **3.21** |
| Current obs only | 7.16 |

差2倍。为什么？

paper用了ROA（Regularized Online Adaptation [https://arxiv.org/abs/2210.04537]），policy的input包含observation history。如果只约束current obs的gradient，policy可以通过history channel间接产生jitter——它把"抽搐信号"塞进没被约束的input维度。

intuition：**Lipschitz约束必须施加在完整的input-output mapping上**，不然policy会找到没被约束的输入维度exploit。这就像治水——你堵住一个口子，水会从别处冒出来。必须全局约束。

### 6.3 Real world的jitter反而比sim小（TABLE III）

Real world deployment的Action Jitter基本在1.1-1.7之间，比sim的3.21还低。

乍看反直觉，仔细想想合理：**real actuator本身有带宽限制**，天然是个low-pass filter，所以policy的aggressive action被hardware过滤掉了。但这也说明sim的jitter高不代表real会jittery——关键看policy是否在尝试aggressive action。

LCP训练的policy在sim里jitter低，说明它**从policy层面就放弃了aggressive action**，而不是依赖hardware filter。这就是为什么LCP transfer好——它约束的是policy的"意图"，不只是"行为"。

### 6.4 No Smoothness的Task Return最高（28.87）

这个数字很重要。它证明了一件事：**simulator里reward最高的是"作弊"的policy**——在idealized dynamics下，bang-bang control是最优解。这正是sim-to-real gap的本质：sim里最优的policy在real里最差。

LCP的价值在于：牺牲一点sim里的task return（28.87 → 26.03），换取real deployability。这是**正则化的经典trade-off**——train loss换generalization。

## 7. 为什么这个paper重要——我的判断

### 7.1 Abstraction layer的提升

好研究的标志是**把一个看似复杂的工程问题归结为一个干净的数学概念**。

- 工程视角："action变化太快怎么罚？joint速度怎么罚？energy怎么罚？每个robot怎么调？"
- 数学视角："policy应该是Lipschitz continuous的"

LCP把前者抽象成后者。一旦抽象到位，解法自然浮现——gradient penalty。

### 7.2 一个$\lambda$替代一堆weight

从工程角度，最大的win是：**一个hyperparameter $\lambda_{gp}$替代了smoothness reward的十几个weight**。这在跨robot transfer时省了大量调参时间。TABLE I(b)显示$\lambda_{gp}$在0.001到0.005之间都work，robustness高。

### 7.3 Differentiable的power

传统smoothness reward不可微，只能通过policy gradient间接优化，信号噪声比低。LCP的gradient penalty可以直接backprop through（用double backward，PyTorch原生支持），优化信号干净。这就像从REINFORCE进化到actor-critic——信号质量的提升带来训练stability。

### 7.4 潜在的跨domain应用

LCP的form不局限于locomotion：
- **Manipulation**：diffusion policy [https://arxiv.org/abs/2303.04137]的score function也可以加Lipschitz penalty
- **Autonomous driving**：driving policy对sensor noise的鲁棒性
- **Finance/Trading**：trading strategy的smoothness
- **Language model**：output对prompt的sensitivity（adversarial robustness的LLM版）

我预感LCP会become standard practice，就像WGAN-GP对GAN做的事一样。

## 8. 几个我自己延伸的思考方向

### 8.1 Adaptive Lipschitz Constant

paper用固定的$\lambda_{gp}=0.002$。但不同gait phase需要的smoothness不同——stance phase需要stiff control（响应快），flight phase需要soft control（避免landing冲击）。

可能的改进：让$\lambda_{gp}$随gait phase $\phi_t$变化，$\lambda_{gp}(\phi_t)$由一个小网络learned。或者用dual gradient descent让$\lambda_{gp}$自动调整。

### 8.2 Per-dimension Lipschitz

当前LCP约束的是整个gradient norm。但不同observation维度的"重要性"不同——gait phase的sensitivity应该高，joint position的sensitivity应该低。

可能的改进：weighted gradient penalty，$\sum_i w_i \|\nabla_{s_i} \log\pi\|^2$，权重$w_i$可以是learned或者per-robot specified。这会保留morphology-invariance的同时allow更精细的控制。

### 8.3 Lipschitz Network Architecture

当前LCP只penalize input-output gradient，不动网络内部。理论上可以用1-Lipschitz网络层（orthogonal weight [https://arxiv.org/abs/1802.05957]、spectral normalization、Frobenius normalization [https://arxiv.org/abs/1804.04368]）让整个网络inherently Lipschitz。这会给出更紧的bound，但优化更难。hybrid approach可能更好——内部用spectral norm，外部加gradient penalty。

### 8.4 Connection to Diffusion Policy

Diffusion policy [https://arxiv.org/abs/2303.04137]在manipulation领域大火。它的smoothness来自采样过程的逐步refine，没有显式Lipschitz constraint。LCP的form可以直接apply——penalize score function $\epsilon_\theta(x_t, t)$的gradient。这是一个promising direction，目前没人做。

### 8.5 Connection to Actuator Modeling

LCP本质是在policy里inject一个"actuator bandwidth prior"。更根本的解法是让simulator直接建模actuator dynamics（first-order low-pass + torque saturation）。这样训练出的policy**自然就是smooth的**，因为env本身penalize jitter。

问题是这种model-based sim速度慢、参数难get。LCP是policy-side surrogate，用简单的gradient penalty替代复杂的env-side modeling。这是经典model-free vs model-based的trade-off，LCP选了前者。

## 9. 一个最终的类比

如果把policy比作一个厨师：
- **No smoothing**：厨师疯狂撒盐撒糖，菜在ideal厨房里尝着好，端到real restaurant客人咸得喝不下水
- **Smoothness reward**：每撒一撮盐罚款5块、每撒一撮糖罚款3块、每翻一次锅罚款2块——厨师得自己算账，每种菜罚款额度不同，新厨师上岗要重新学罚款单
- **Low-pass filter**：在厨师手后面装个阻尼，他撒盐动作变慢——但厨师觉得自己使不上劲，菜也炒不香
- **LCP**：直接规定"厨师手的sensitivity有上限"——你切菜幅度不许超过某个范围，至于切什么、怎么切，随你

LCP的elegant在于它约束的是**抽象层面的行为属性**，而不是具体动作的symptom。这是好研究的标志——abstraction对了，一切自然简洁。

## Web Links Reference

核心paper与project:
- LCP Project Page: https://lipschitz-constrained-policy.github.io
- Berkeley Humanoid platform: https://arxiv.org/abs/2407.21781

理论基础:
- Bang-Bang Principle (LaSalle 1960): https://www.sciencedirect.com/science/article/pii/S147466701770095X
- WGAN: https://arxiv.org/abs/1701.07875
- WGAN-GP: https://arxiv.org/abs/1704.00028
- TRPO (expectation heuristic): http://arxiv.org/abs/1502.05477

RL与sim-to-real:
- PPO: https://arxiv.org/abs/1707.06347
- IsaacGym: https://arxiv.org/abs/2108.10470
- Deep WBC (ROA): https://arxiv.org/abs/2210.04537
- AMP (Adversarial Motion Priors): https://doi.org/10.1145/3450626.3459670

Lipschitz网络与adversarial robustness:
- Spectral Normalization: https://arxiv.org/abs/1802.05957
- Frobenius Norm Regularization: https://arxiv.org/abs/1804.04368
- Madry Adversarial Training: https://arxiv.org/abs/1706.06083

延伸方向:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Humanoid Locomotion RL (Radosavovic): https://arxiv.org/abs/2402.19469
- ExBody (Expressive Whole-Body Control): https://arxiv.org/abs/2402.16796

如果你想我继续深挖哪个方面——比如gradient penalty在PyTorch里怎么实现double backward、ROA的stop-gradient为什么work、或者LCP用到diffusion policy的具体form——告诉我就行。

---

# Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies 深度解析

Karpathy你好，这篇paper我读了之后觉得非常clean，思路简洁但insight很深。让我从intuition出发给你掰开揉碎讲讲。

## 1. 核心问题的intuition

这篇paper要解决一个非常实际的问题：**RL policy在simulator里训练时，会产生jittery的bang-bang行为**。为什么会这样？因为simulator的动力学过于理想化——motor可以在任意state下瞬间输出任意torque，没有带宽限制、没有摩擦死区、没有热饱和。policy在这个"天堂"里发现，高频切换action能更激进地优化reward，于是倾向于bang-bang control（LaSalle, 1960 [https://www.sciencedirect.com/science/article/pii/S147466701770095X]）。

但现实世界的actuator是有带宽限制的，PD controller的command tracking跟不上policy的剧烈切换，于是torque saturate、电机过热、机械结构冲击——sim-to-real直接失败。

## 2. 传统方法的瓶颈

### 2.1 Smoothness Rewards
经典做法是在reward里加一堆penalty term：
- Joint velocity penalty
- Joint acceleration penalty  
- Energy consumption penalty
- Action rate penalty
- Action jitter (三阶导数) penalty

这种approach有几个根本问题：
- **不可微**：reward是environment的一部分，policy gradient只能通过sampling-based estimator（REINFORCE/PPO）间接优化，signal-to-noise ratio低
- **超参数地狱**：每个robot都要重新调权重，walking vs running weight不一样，新机器人要试错几周
- **Reward hacking**：policy会找到reward的loophole，比如把joint锁在某个位置避免velocity penalty

### 2.2 Low-pass Filter
在policy输出后面挂一个filter $a_t = \alpha \cdot a_{t-1} + (1-\alpha) \cdot \pi(o_t)$。问题是：
- Filter本身就**damping exploration**，policy拿不到真实action的反馈，credit assignment错位
- Filter的cutoff frequency是固定的，但不同gait phase需要的bandwidth不一样
- 不是end-to-end differentiable

## 3. LCP的核心数学思想

### 3.1 关键insight
作者做了一个非常漂亮的motivating experiment（Fig. 3）：对比有无smoothness reward训练的policy，发现**有smoothness reward的policy，其gradient norm $\|\nabla_s \log\pi(a|s)\|$显著更小**。

这就点出了一个关键connection：**smoothness reward本质上是在implicit地约束policy的Lipschitz常数**。既然如此，为什么不直接explicit约束Lipschitz常数？这就是paper的名字来源。

### 3.2 Lipschitz Continuity
Lipschitz continuity的formal definition：
$$d_Y(f(x_1), f(x_2)) \leq K \cdot d_X(x_1, x_2)$$

变量解释：
- $(X, d_X)$ 和 $(Y, d_Y)$：两个metric space，$d_X, d_Y$ 分别是输入和输出的metric（通常是Euclidean distance）
- $f: X \to Y$：我们要约束的函数（这里是policy $\pi$）
- $K$：Lipschitz常数，限制函数变化率的上界

intuition：函数图像不能太陡，任意两点之间的output distance被input distance的$K$倍限制。一个$K$-Lipschitz的函数，slope永远不超过$K$。

corollary（公式2）：
$$\|\nabla_x f(x)\| \leq K \implies f \text{ is Lipschitz continuous}$$

注意**逆命题不成立**——Lipschitz连续的函数不一定处处可微，但可微点的gradient norm一定被$K$ bound。这给了我们一个**充分条件**：只要能bound住gradient norm，就一定Lipschitz连续。

### 3.3 从constraint到gradient penalty

原始的constrained optimization（公式4）：
$$\max_\pi J(\pi) \quad \text{s.t.} \quad \max_{s,a} \|\nabla_s \log\pi(a|s)\|^2 \leq K^2$$

变量解释：
- $J(\pi)$：RL objective，即discounted cumulative reward $\mathbb{E}[\sum_t \gamma^t r_t]$
- $\pi(a|s)$：stochastic policy
- $\nabla_s \log\pi(a|s)$：log-likelihood对state的gradient，衡量state微小扰动如何影响action分布
- $K$：用户指定的Lipschitz常数上界

这里用$\log\pi$而不是$\pi$，是为了和score function gradient estimator对齐：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log\pi(a|s) \cdot Q]$，log-likelihood的gradient本身就有信息几何意义。

**问题**：$\max_{s,a}$在整个state-action space上不可计算（intractable）。

**近似1**（公式5）：用expectation替代max（Schulman et al., TRPO [http://arxiv.org/abs/1502.05477] 的heuristic）：
$$\mathbb{E}_{s,a \sim \mathcal{D}}[\|\nabla_s \log\pi(a|s)\|^2] \leq K^2$$

这里$\mathcal{D}$是从policy rollout收集的state-action pair dataset。这是**pointwise constraint的mean field近似**——比max弱，但优化上tractable。

**近似2**（公式6）：Lagrangian relaxation，引入multiplier $\lambda$：
$$\min_{\lambda \geq 0} \max_\pi J(\pi) - \lambda\left(\mathbb{E}[\|\nabla_s \log\pi\|^2] - K^2\right)$$

理论上应该min-max alternating优化，但实践中**直接固定$\lambda_{gp}$作为hyperparameter**更简单。这就退化成了公式7：

$$\max_\pi J(\pi) - \lambda_{gp} \mathbb{E}_{s,a \sim \mathcal{D}}[\|\nabla_s \log\pi(a|s)\|^2]$$

### 3.4 与WGAN-GP的connection

这个form和WGAN-GP（Gulrajani et al. 2017 [https://arxiv.org/abs/1704.00028]）几乎一模一样。WGAN-GP是给discriminator加gradient penalty，让discriminator满足1-Lipschitz条件，从而让Wasserstein distance有估计。

LCP的insight是：**policy本身也应该是Lipschitz的**——因为现实中observation sensor有noise、有latency，policy不应该对observation的微小扰动产生剧烈反应。这其实是一种**input robustness regularization**，恰好等价于output smoothness。

这点很深刻。我联想到adversarial robustness领域（Sinho et al. [https://arxiv.org/abs/1805.12514]），Lipschitz constant小 $\iff$ adversarially robust $\iff$ smooth，这是同一个数学对象的三个面相。

## 4. 实现细节

### 4.1 训练pipeline
- Simulator: IsaacGym [https://arxiv.org/abs/2108.10470]，4096 parallel environments
- Algorithm: PPO [https://arxiv.org/abs/1707.06347]
- Sim-to-real: Regularized Online Adaptation (ROA)（Fu et al. [https://arxiv.org/abs/2210.04537]）

ROA的核心是有一个privileged encoder $\mu(e) \to z^\mu$（用mass、CoM、motor strength等真实参数编码环境），和一个adaptation module $\phi$（只用proprioceptive history估计$z^\phi$），二者用L2 loss + stop-gradient对齐（公式8）。

### 4.2 Observation design
$$\mathbf{o}_t = [\phi_t, \mathbf{c}_t, \mathbf{s}_t^{robot}, \mathbf{a}_{t-1}]$$

- $\phi_t \in \mathbb{R}^2$：gait phase，sin/cos编码的周期clock signal（这是bipedal locomotion的core trick，借鉴了Peng et al. AMP/DeepMimic的phase variable）
- $\mathbf{c}_t = [v_x^{cmd}, v_y^{cmd}, v_{yaw}^{cmd}]$：command，线速度$x \in [0, 0.8]$ m/s, $y \in [-0.4, 0.4]$ m/s, yaw $\in [-0.6, 0.6]$ rad/s
- $\mathbf{s}_t^{robot}$：joint positions + velocities
- $\mathbf{a}_{t-1}$：上一时刻action（让policy自己学temporal smoothness）

privileged info $e_t$：base mass, CoM offset, motor strength, root linear velocity——这些online测不到的。

### 4.3 关键的GP应用位置（TABLE I(c)）
这是paper里一个很关键但容易被忽略的ablation：
- GP on whole obs（含history）：Action Jitter **3.21**
- GP on current obs only：Action Jitter **7.16**

差2倍以上！为什么？因为ROA的adaptation module $\phi$吃的是observation history，如果只约束current obs的gradient，policy可以通过history channel间接产生jitter——gradient norm bound不住整个mapping。

这个发现给我一个intuition：**Lipschitz约束必须施加在完整的input-output mapping上**，否则policy会找到没被约束的input维度exploit。这和input smoothing的regularization theory一致。

### 4.4 $\lambda_{gp}$的sweep（TABLE I(b)）
- $\lambda_{gp} = 0$：Action Jitter **42.19**，Task Return **28.87**（jittery但reward最高，典型的reward hacking）
- $\lambda_{gp} = 0.001$：Action Jitter **3.69**，Task Return 26.32（边缘可deploy）
- $\lambda_{gp} = 0.002$（authors' choice）：Action Jitter **3.21**，Task Return 26.03
- $\lambda_{gp} = 0.005$：Action Jitter 2.10，Task Return **23.92**（开始下降）
- $\lambda_{gp} = 0.01$：Action Jitter 0.17，Task Return **16.11**（over-smooth，policy变得sluggish）

这个曲线很经典——**正则化的sweet spot**。太小没用，太大瘫痪。$\lambda_{gp}=0.002$是经验调出来的。我注意到这个值在所有4个robot上都用同一个，说明LCP对$\lambda$的sensitivity比较低，这是相比smoothness reward的优势。

### 4.5 Reward curriculum（Appendix B）
一个trick：负reward加scaling factor $s_{t,i}$，初始0.8，episode length < 50时×0.9999，> 400时×1.0001，上限2.0。

intuition：早期episode短，少加penalty让policy explore基本gait pattern；后期稳定了，penalty加重，逼迫policy优化细节。这是**self-paced curriculum**的一种实例化。

## 5. 实验结果深度解读

### 5.1 vs Baselines（TABLE I(a)）
| Method | Action Jitter | DoF Vel | Energy | Task Return |
|---|---|---|---|---|
| LCP | **3.21** | **10.65** | **24.57** | 26.03 |
| Smoothness Reward | 5.74 | 11.35 | 25.92 | 26.56 |
| Low-pass Filter | 7.86 | 11.72 | 32.83 | 24.98 |
| No Smoothness | 42.19 | 12.92 | 42.68 | **28.87** |

几个观察：
1. LCP在所有smoothness metric上都优于或持平smoothness reward——这很impressive，因为LCP**没有任何直接优化这些metric的reward term**，只是约束了Lipschitz常数
2. No smoothing的Task Return最高（28.87），但Action Jitter是42.19——典型的"reward hacking导致high return but unrealistic behavior"。这印证了open-loop simulator的idealization问题
3. Low-pass filter最差，验证了它damping exploration的hypothesis

### 5.2 Sim-to-Sim Transfer（TABLE II）
IsaacGym → Mujoco transfer，task return略有下降但robust。Fourier GR1从26.03降到24.33，Unitree H1降到21.74，Berkeley Humanoid保持26.50。说明full-size humanoid的domain gap更大（更重的link、更高的CoM、更复杂的contact dynamics）。

### 5.3 Real World Deployment（TABLE III）
在smooth/soft/rough三种terrain上deploy，Action Jitter在real world基本保持1.1-1.7（比sim的3.21还低！）。这有点意外，可能的解释：real actuator本身有bandwidth limit，相当于天然的low-pass filter，所以真实jitter比sim小。但**sim的jitter高意味着policy仍然在尝试aggressive action**，只是被real hardware filter了——这也是为什么sim的smoothness metric不一定直接predict real world performance，但LCP训练的policy确实更可deploy。

### 5.4 外力扰动恢复
paper提到可以recover from external forces，但没给quantitative metric。supplementary video里应该有demo。

## 6. 我的延伸思考与潜在方向

### 6.1 联系到adversarial robustness
LCP让我想起Madry et al.的adversarial training [https://arxiv.org/abs/1706.06083]——adversarial example的$\ell_\infty$扰动等价于一个worst-case Lipschitz analysis。如果policy是$K$-Lipschitz的，那对observation noise的鲁棒性有一个quantitative bound：$\|\Delta a\| \leq K \|\Delta o\|$。这意味着LCP-trained policy天然有bounded sensitivity to sensor noise。这点paper没明说但很powerful。

### 6.2 联系到input smoothing in deep RL
最近的工作如 Sobolev Training [https://arxiv.org/abs/1806.06206]、Lipschitz regularization for value function [https://arxiv.org/abs/2007.02935] 都在用类似的idea。LCP把这些idea第一次系统化地应用到humanoid locomotion。

### 6.3 联系到1-Lipschitz neural network
理论上可以让整个网络是1-Lipschitz的——用orthogonal weight、spectral normalization [https://arxiv.org/abs/1802.05957]、gradient penalty。但LCP只penalize input-output gradient，不动网络内部。这是trade-off：implementation简单，但理论上不够tight。Gouk et al. [https://arxiv.org/abs/1804.04368] 探索过把Lipschitz constraint施加在每一层，可能给更紧的bound，但优化更难。

### 6.4 联系到bang-bang control theory
LaSalle的bang-bang principle说LQR/LQP最优control经常是bang-bang的。RL之所以产生jittery behavior，是因为它在sim里找到了类似的最优解——把actuator当ideal source用。LCP本质是在objective里加了一个**actuator bandwidth prior**，惩罚policy试图超越这个bandwidth的部分。这是一个implicitly model-based regularizer。

### 6.5 联系到diffusion policy
最近diffusion policy [https://arxiv.org/abs/2303.04137] 在manipulation领域大火。diffusion policy天然smooth（DDPM的denoising process是逐步refine，不会突然跳），但它的smoothness来自采样过程，没有显式Lipschitz constraint。LCP的form可以直接apply到diffusion policy上——penalize score function的gradient。这是一个promising direction。

### 6.6 Hyperparameter敏感性 vs generalization
paper最有说服力的结果是**同一个$\lambda_{gp}=0.002$在4个不同形态的humanoid上都work**——Fourier GR1T1/T2（21 joints）、Unitree H1（19 joints）、Berkeley Humanoid（12 joints）。这暗示LCP的hyperparameter确实有cross-morphology generalization，这是smoothness reward做不到的。可能的原因：Lipschitz常数是一个**morphology-invariant的smoothness measure**，不依赖于具体的joint configuration。

### 6.7 局限性思考
paper自己提到只验证了walking，没验证running/jumping等dynamic gait。我怀疑LCP对highly dynamic behavior会有issue——跳跃需要瞬间的大action change，Lipschitz constraint可能over-restrict。可能的fix：**adaptive Lipschitz constant**，让$K$随gait phase变化（stance phase K小，flight phase K大）。或者把$\lambda_{gp}$做成learnable（dual gradient descent）。

### 6.8 Connection to implicit dynamics model
LCP假设policy应该是Lipschitz的，但更根本的问题是actuator dynamics不是Lipschitz的。如果simulator能建模actuator bandwidth（first-order low-pass + torque saturation），那训练出的policy**自然就是smooth的**——因为env本身就penalize jitter。问题是这种model-based sim speed慢、参数难get。LCP是用一个**policy-side surrogate**替代env-side modeling，简化实现但trade-off了一些物理fidelity。

## 7. 总结

这篇paper的beauty在于：**把一个看似复杂的工程问题（怎么调smoothness reward权重）归结为一个干净的数学概念（Lipschitz continuity），然后用一个简单的differentiable penalty实现**。整个method只有几行代码，但works on 4 different humanoids zero-shot。这种**abstraction level的提升**是好研究的标志。

我的intuition总结：
1. Smoothness reward是implicit Lipschitz constraint，LCP是explicit Lipschitz constraint——把implicit变explicit，省了调一堆reward weight的功夫
2. Differentiability是关键：能直接backprop through gradient penalty，比sampling-based policy gradient信号强得多
3. Lipschitz constant是morphology-invariant的smoothness metric，这解释了为什么同一个$\lambda_{gp}$跨robot work
4. LCP本质是在policy里inject一个"actuator bandwidth prior"——告诉你policy不要去sim的idealized region去explore

我个人觉得这篇paper会become standard practice。WGAN-GP对GAN做的事，LCP对RL正在做。期待看到LCP扩展到manipulation、loco-manipulation、甚至humanoid的whole-body control。

## Web Links Reference
- Paper project page: https://lipschitz-constrained-policy.github.io
- WGAN: https://arxiv.org/abs/1701.07875
- WGAN-GP: https://arxiv.org/abs/1704.00028
- PPO: https://arxiv.org/abs/1707.06347
- TRPO: http://arxiv.org/abs/1502.05477
- IsaacGym: https://arxiv.org/abs/2108.10470
- Deep WBC (ROA): https://arxiv.org/abs/2210.04537
- Spectral Normalization: https://arxiv.org/abs/1802.05957
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- AMP (Adversarial Motion Priors): https://doi.org/10.1145/3450626.3459670
- Madry Adversarial Training: https://arxiv.org/abs/1706.06083
- Bang-Bang Principle (LaSalle): https://www.sciencedirect.com/science/article/pii/S147466701770095X
- Humanoid Locomotion RL (Radosavovic): https://arxiv.org/abs/2402.19469
- Expressive Whole-Body Control (ExBody): https://arxiv.org/abs/2402.16796
- Berkeley Humanoid platform: https://arxiv.org/abs/2407.21781

如果你想我深挖某个aspect——比如ROA的stop-gradient细节、reward curriculum的收敛行为、或者LCP在diffusion policy上的extension——告诉我，我可以再展开。
