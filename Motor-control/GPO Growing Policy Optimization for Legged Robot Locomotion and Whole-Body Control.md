---
source_pdf: GPO Growing Policy Optimization for Legged Robot Locomotion and Whole-Body
  Control.pdf
paper_sha256: db96f8357fd8c291e93cce95adfe31b732d2fbddcc7cc14fef198846590df825
processed_at: '2026-08-04T22:11:19-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GPO 的人话版

Andrej, 前面讲得太学术了，让我换个画风，像在白板上跟你边画边聊。

---

## 一句话概括

**训练机器人走路的时候，别一开始就给它全身肌肉随便用，先让它学会用小力气站稳，再慢慢给它放开力气，它学得又快又好。**

---

## 故事版

想象你教一个婴儿学走路。

**做法A (PPO)**：第一天就把婴儿扔到操场上，说"随便跑吧，跑得好给糖吃"。婴儿连站都不会，就在地上乱蹬，腿到处飞，每次都摔。偶尔一次歪打正着往前挪了一步，婴儿也搞不明白是哪条腿做对了。这种random exploration效率极低——婴儿在"全肌肉空间"里乱采样，gradient signal被淹没在noise里。

**做法B (GPO)**：第一天把婴儿的腿用橡皮筋绑住，只能动一点点。婴儿在这个小范围内反复试，很快发现"稍微往前倾一下身体就能往前挪一步"——因为action space小，每次尝试的结果可预测性强，reward signal清晰。等婴儿在小范围内学会稳定的微步后，第二天橡皮筋松一点，第三天再松一点……到最后完全松开，婴儿已经掌握了精细的balance control，大动作只是小动作的scale-up。

GPO就是在代码里实现了这个"橡皮筋逐渐松开"的过程。

---

## 核心机制（用代码说）

Standard PPO的做法：

```python
a = policy_network(obs)           # 网络输出任意大小的action
ã = clip(a, -32, 32)             # 硬截断到硬件允许范围
env.step(ã)                      # 执行
```

问题：`clip` 在边界处gradient直接为零。早期训练policy乱输出大值，大量samples被clip掉，gradient信息全丢了。

GPO的做法：

```python
a = policy_network(obs)           # 网络输出latent action
β_t = β_max * gompertz(t)         # 时间t的"橡皮筋松紧度"
ã = β_t * tanh(a / β_t)           # soft squash，处处可导
env.step(ã)                       # 执行
```

- 训练早期 `t` 小，`β_t` 很小（比如0.5），`tanh` 把所有action压到 [-0.5, 0.5] 范围内
- 训练中期 `β_t` 增长到10，action范围扩展到 [-10, 10]
- 训练晚期 `β_t` 接近32（硬件上限），action范围几乎不受限

`tanh` 的好处是smooth的——即使action很大也只是soft saturation，gradient不会突然消失，永远有微弱的梯度信号告诉policy"你在边界附近了"。

---

## 为什么这个work？三个intuition

### Intuition 1: Gradient noise 随action space 平方增长

Policy gradient本质是 `score function × advantage`。Score function的magnitude和action range成正比——action range越大，`∇log π(a|s)` 越大，gradient estimator的variance越大。

数学上：`Var[g_t] ∝ β_t²`

所以action space小一个数量级，gradient variance小两个数量级。早期训练时advantage估计本身就很noisy，如果action space再放大gradient noise，SNR就低到update方向几乎是random walk。GPO通过限制action space把noise压下去，让learning signal浮出来。

类比：你想听清一个人说话（learning signal），但周围有施工噪声（gradient noise）。GPO的做法是早期把施工关掉（small β_t），你能听清楚；等你会听人话了，再慢慢把施工开起来，你已经能filter out噪声了。

### Intuition 2: PPO的update rule没有被破坏

你可能会担心：transform action会不会改变PPO的trust region properties？

答案是不会。因为 `tanh` transformation的Jacobian term和policy参数 θ 无关（前提是σ不学，只学μ）。在importance ratio `π_θ(ã)/π_θold(ã)` 里，Jacobian term分子分母都有，直接约掉，剩下的是latent action space里的standard PPO ratio。

所以GPO本质上是：**在latent action space里做标准PPO，在environment interface加一个time-varying squashing layer**。PPO的所有theoretical guarantees（clipped surrogate、trust region）都保留。

### Intuition 3: Gradient distortion自动消失

你可能会担心：transformation改变了gradient的统计性质，会不会train不稳定？

paper证明gradient distortion的上界正比于 `|β_max - β_t|`。

- 早期 `β_t` 小，distortion大——但这时候我们 *故意* 想要distortion，因为small action space就是我们的目标
- 晚期 `β_t → β_max`，distortion → 0，GPO自动退化成standard PPO

这是个self-correcting mechanism：distortion在最该出现的时候出现（早期），在最不该出现的时候消失（晚期）。

---

## 为什么用Gompertz而不是Linear？

这是paper最有洞察力的实验发现之一。

试了四种"松橡皮筋"的schedule：

| Schedule | 形状 | 效果 |
|---|---|---|
| No growth (PPO) | 一直松 | 最差 |
| Linear | 匀速松 | 差，early stage松太快，late stage又松得不够 |
| Sigmoid | 对称S形 | 中等，early stage还行但late stage unlock不够 |
| **Gompertz** | **不对称——慢启动、快中段、慢收尾** | **最好** |

Gompertz为什么最好？因为learning的natural dynamics也是不对称的：

- **Early stage**：policy在random init附近，需要强restriction来stabilize。Gompertz的slow start正好给足时间。
- **Middle stage**：policy已经学到基本control，需要rapid unlock来escape local optimum去explore更aggressive的行为。Gompertz的fast middle正好。
- **Late stage**：policy接近收敛，需要充分unlock来refine asymptotic behavior。Gompertz的slow end（但已接近1）正好。

Linear growth在early stage松太快（policy还没站稳就给大action space，又乱了）；Sigmoid在late stage松太慢（action space还没完全unlock训练就结束了）。只有Gompertz的asymmetric shape匹配了learning dynamics的asymmetric phases。

这个发现说明：**growth schedule的shape比"是否grow"本身更重要**。任何grow都比不grow好，但grow得好和grow得差可以差很多。

Gompertz function其实是生物生长曲线（tumor growth、population dynamics都用它），可能不是巧合——动物学运动技能的过程可能也follow类似的asymmetric schedule：婴儿运动技能 acquisition 先慢（爬）、中快（走→跑）、后慢（精细调优）。

---

## 一个被paper藏起来的关键细节

Reward里的command也乘了 `β_t`：

```python
# Standard PPO
reward = -|v_x_actual - v_x_cmd|  # v_x_cmd = 1.0 m/s

# GPO
reward = -|v_x_actual - v_x_cmd * β_t|  # 早期 β_t=0.05, 实际command = 0.05 m/s
```

为什么必须这么做？如果early stage机器人只能输出小torque（β_t=0.5），你却要求它跑1 m/s，物理上做不到。Robot拼了命也达不到command，reward永远是负的，policy学不到任何useful gradient。

所以command也必须scale down：早期只要求机器人爬0.05 m/s，等它的actuation capacity上来了再逐步提高要求。这是task demand curriculum，和action capacity curriculum同步。

Paper的理论分析只cover了action transformation，没cover command scaling。但在实际实现里这两者是coupled的——光restrict action不restrict command，early stage的reward会被"要求过高、做不到"的negative signal淹没。

我觉得这个细节其实可以单独写一篇paper。

---

## 实验结果一句话总结

- **训练曲线**：GPO早期reward上升明显比PPO快（high SNR → fast early convergence），asymptotic reward更高（no-worse basin selection + 充分unlock）
- **Tracking精度**：PPO在quadruped上commanded 0.5 m/s实际只爬到0.2 m/s（error -0.3），GPO error只有0.015——一个数量级的差距
- **Gait质量**：GPO产生清晰的periodic stepping pattern，PPO的joint trajectories erratic，hexapod还会出现"只用前腿撑着、后腿闲着"的degenerate gait
- **硬件robustness**：被5kg哑铃侧砸、走泥地下坡、被人往下踩，GPO全部100%成功recover，PPO在side push直接0%成功率直接倒地

---

## 对你的connection

你做nanoGPT时肯定也观察到类似的phenomenon——用sequence length curriculum（先训短序列，再grow到长序列）比直接训长序列更efficient。背后的principle可能是一样的：

**Progressive capacity unlocking matches the natural curvature of the learning landscape.**

早期learning landscape在"小capacity"区域更smooth、更容易navigate，等policy/network在小capacity区域converge到reasonable basin后，再unlock更大capacity让它explore。如果一开始就给full capacity，learning在high-dimensional space里random walk，SNR太低，convergence极慢。

GPO把这个principle在legged robot的action space上做得很clean。我觉得这个principle是universal的——在sequence length、model size、reward complexity、domain randomization范围、action space等维度都应该有类似的"progressive unlocking"效应，只是大家的formalization程度不同。

GPO可能只是这个更大principle的一个具体instance。期待未来有人把它generalize成一个unified framework。

---

## 最后的TL;DR

| | PPO | GPO |
|---|---|---|
| Action space | 固定，一直full range | 从小到大smooth grow |
| 早期gradient noise | 大（β²大） | 小（β_t²小） |
| 早期SNR | 低 | 高 |
| 早期convergence | 慢 | 快 |
| Asymptotic return | baseline | ≥ baseline |
| 实现复杂度 | 基线 | 加一个tanh和一行schedule代码 |
| 理论保证 | standard PPO | update equivalence + bounded distortion |

GPO就是个"几乎免费"的改进——一行代码加tanh，一个schedule函数，理论上不破坏PPO，实验上全面提升。这种orthogonal、plug-and-play、principled的工作我最喜欢。

Reference:
- [GPO paper](https://arxiv.org/abs/2502.12674) (原文)
- [PPO](https://arxiv.org/abs/1707.06347)
- [SAC (tanh squashing的出处)](https://arxiv.org/abs/1801.01290)
- [DREAMWAQ (actor-critic架构)](https://arxiv.org/abs/2305.06563)
- [Gompertz function](https://en.wikipedia.org/wiki/Gompertz_function)
- [Curriculum Learning (Bengio 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

# GPO: 从PPO的Action Space视角重新审视Legged Robot Control

Andrej, 这篇paper抓住了一个我之前在PPO实现里反复纠结但没想透的点：**action space的"形状"在训练过程中应该是不变的吗？** GPO的核心proposal非常简洁——用一个time-varying的tanh-based squashing transformation `ã = β_t · tanh(a/β_t)`，让effective action space从窄到宽平滑grow。听起来像个engineering trick，但paper把它升级成了有理论保证的优化框架。下面是我读完后的deep dive。

---

## 1. Big Picture Intuition: 为什么这是一个真问题

在torque-based legged robot control里，你直接把policy输出当joint torque喂给电机。这意味着：

- contact-rich dynamics完全暴露给policy（没有low-level PD servo做buffer）
- action的每一维都直接对应物理量，small torque perturbation可能造成base orientation的大幅destabilization
- 12维（quadruped）或18维（hexapod）的连续action space，加上intermittent foot contacts，sample efficiency很差

PPO在这种setting下的typical failure mode是：**早期训练时policy在大action space里乱踩，gradient variance巨大，optimization landscape很rugged**。工程师们empirically早就知道怎么做：早期把torque limit收紧，等policy学会stable stance后再放松。但这是个heuristic，没有principled framework，更没有理论分析它对优化几何的影响。

GPO做的就是把这个engineering trick正儿八经写进algorithm里，并分析清楚为什么它work。

直觉上：你想让agent先在restricted action space里学会"微步稳走"（low-amplitude but reliable control），然后逐步解锁"大步跑跳"的能力。这就像婴儿学走路，先crawl，再stand，再walk，再run——每个阶段的motor space都比上一阶段大。

---

## 2. 核心数学：Action Transformation的细节

### 2.1 The Transformation

GPO用的transformation是：

$$\tilde{a} = \beta_t \cdot \tanh\!\left(\frac{a}{\beta_t}\right)$$

变量含义：
- $a$ — policy network输出的latent action，假设 $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$，$\sigma$ 与 $\theta$ 无关（标准Gaussian policy设置）
- $\tilde{a}$ — 实际执行到环境的action（joint torque）
- $\beta_t = a_{\text{limit}} \cdot f(t)$ — 时变的range parameter，$f(t)$ 单调递增且 $\lim_{t\to\infty} f(t) = 1$，所以 $\beta_t$ 从小到大增长，最终达到 $a_{\text{limit}}$
- $a_{\text{limit}}$ — 硬件允许的最大action magnitude（quadruped实验里=32，hexapod=40）

性质：
- **Near-linear regime** ($|a| \ll \beta_t$)：$\tanh(x) \approx x$，所以 $\tilde{a} \approx a$。在小action下transformation几乎是identity。
- **Saturation regime** ($|a| \gg \beta_t$)：$\tanh$ 饱和到 $\pm 1$，$\tilde{a} \approx \pm \beta_t$。Soft action limit。
- **Smooth everywhere**：相比hard clip $\text{clip}(a, -a_{\text{limit}}, a_{\text{limit}})$，tanh处处可导，没有gradient断点。

为什么这个重要？hard clip在边界处 $|a| = a_{\text{limit}}$ 不可导，policy gradient在那里直接vanish。在legged robot control里，早期训练policy经常会hit action limit（因为还没学会fine control），大量samples的gradient就这么丢了。tanh-based squashing保留了所有gradient information。

paper中关键assumption：$\frac{a}{\beta_t} \in [-0.5, 0.5]$，即policy的latent action大部分时间在near-linear regime。这个assumption在practice里成立（实验Fig. 4验证了joint torque确实远小于limit），但在理论分析里它给了我们一个clean bound。

### 2.2 为什么选Gompertz Growth Function

paper比较了四种growth function（Tab. I）：
- **No growth** (PPO baseline): $\beta_t$ 始终是 $a_{\text{limit}}$
- **Linear**: $\beta_t = k \cdot t$ with $k = 1/3000$
- **Sigmoid**: $\beta_t = (1 + \exp(-k(t-t_0)))^{-1}$, $k = -2.3\times10^{-3}$, $t_0 = 3000$
- **Gompertz**: $\beta_t = e^{-e^{-k(t-t_0)}}$, $k = 3\times10^{-5}$, $t_0 = 2.4\times10^4$

Gompertz的形状是**asymmetric**的：
- 早期 ($t \ll t_0$)：$\beta_t \approx 0$，增长极慢
- 中期 ($t \approx t_0$)：加速增长
- 晚期 ($t \gg t_0$)：饱和到1，增长极慢

Gompertz function的derivative满足一个有趣的性质（写成 $\beta$ 的ODE）：$\frac{d\beta}{dt} = k \beta \cdot (-\ln \beta)$。当 $\beta \to 0$ 时 $-\ln\beta \to \infty$ 但 $\beta \to 0$，所以 $d\beta/dt \to 0$，这是slow start的来源。

实验结果（Fig. 3）：
- **Gompertz最好**：早期训练reward上升快且stable，asymptotic reward最高
- **Linear最差**：unstable或premature saturation——constant growth rate太aggressive，early stage还没建立stable control就unlock大action
- **Sigmoid中间**：比linear好但不如Gompertz，因为symmetric S-curve在early stage还是grow太快

这个结果的intuition：early training需要**强restriction**来stabilize optimization；late training需要**充分unlock**来escape suboptima。Gompertz的asymmetric shape正好匹配这两个需求——slow start给early stage足够stability，rapid middle给late stage足够exploration budget。

我觉得这个empirical observation是paper最强的selling point之一。它说明action space growth的**schedule shape**比"是否growth"本身更重要。Linear growth和Gompertz growth都是"逐渐grow"，但效果差很多，因为growth rate的时间分布不一样。

---

## 3. Update Equivalence: 为什么GPO不破坏PPO

这是paper的理论core。要证明的是：尽管我们用了time-varying transformation，PPO的clipped surrogate objective保持structure不变。

### 3.1 Importance Ratio在GPO下的形式

Standard PPO的importance ratio：
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

GPO下，环境看到的是 $\tilde{a}$，所以importance ratio变成：
$$r_t^{\text{GPO}}(\theta) = \frac{\pi_\theta(\tilde{a}_t | s_t)}{\pi_{\theta_{\text{old}}}(\tilde{a}_t | s_t)}$$

关键：$\pi_\theta(\tilde{a}|s)$ 是induced density on $\tilde{a}$ space，需要change of variables。原密度 $\pi_\theta(a|s)$ 定义在latent action $a$ 上，transformation $a \mapsto \tilde{a} = h_\beta(a)$ 是bijective（在 $\beta$ 固定时）。

由change of variables formula：
$$\pi_\theta(\tilde{a}|s) = \pi_\theta(a|s) \cdot \left|\frac{da}{d\tilde{a}}\right|, \quad \text{where } a = h_\beta^{-1}(\tilde{a}) = \beta \cdot \text{arctanh}(\tilde{a}/\beta)$$

Jacobian determinant：
$$\frac{d\tilde{a}}{da} = \text{sech}^2\!\left(\frac{a}{\beta}\right) = 1 - \tanh^2\!\left(\frac{a}{\beta}\right) = 1 - \left(\frac{\tilde{a}}{\beta}\right)^2$$

所以：
$$\left|\frac{da}{d\tilde{a}}\right| = \left(1 - \left(\frac{\tilde{a}}{\beta}\right)^2\right)^{-1}$$

对d-dimensional action：
$$\pi_\theta(\tilde{\mathbf{a}}|s) = \pi_\theta(\mathbf{a}|s) \cdot \prod_{i=1}^{d} \left(1 - \left(\frac{\tilde{a}_i}{\beta}\right)^2\right)^{-1}$$

定义Jacobian term $J(\theta) := \prod_{i=1}^{d} \left(1 - (\tilde{a}_i/\beta)^2\right)^{-1}$。关键观察：**$J$ 不依赖于 $\theta$**（因为 $\sigma$ 与 $\theta$ 无关，$\mu_\theta$ 不出现在Jacobian中，$\tilde{a}_i$ 是sampled fixed value，$\beta$ 在一个PPO epoch内固定）。

所以在importance ratio里 $J$ 完全抵消：
$$r_t^{\text{GPO}}(\theta) = \frac{\pi_\theta(a_t|s_t) \cdot J}{\pi_{\theta_{\text{old}}}(a_t|s_t) \cdot J} = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = r_t^{\text{PPO}}(\theta)$$

**Insight**：tanh squashing的Jacobian term与 $\theta$ 无关（前提是 $\sigma$ 不学），所以importance ratio不变，PPO的clipped surrogate objective结构完全保留。这是个clean result，说明GPO在optimization层面没有破坏PPO的trust region properties。

这一点其实和SAC里的tanh squashing很像——SAC也是用 $\tanh$ 把Gaussian squashed到bounded action space。区别在于SAC用fixed $\beta$，GPO用time-varying $\beta_t$。SAC还需要在Jacobian term上做处理（因为SAC学 $\sigma$），但GPO假设 $\sigma$ fixed，所以Jacobian term干净地抵消了。

Reference: [SAC paper](https://arxiv.org/abs/1801.01290), [PPO paper](https://arxiv.org/abs/1707.06347)

### 3.2 Gradient Distortion Bound

虽然update rule不变，但gradient estimator的统计性质会变。paper证明了一个bound来quantify这个distortion。

设 $\phi(\beta) := h_\beta^{-1}(\tilde{a}) = \beta \cdot \text{arctanh}(\tilde{a}/\beta)$。则：

$$\nabla_\theta \log \pi_{\theta,\beta}(\tilde{a}|s) = \frac{h_\beta^{-1}(\tilde{a}) - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s) = \frac{\phi(\beta) - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)$$

GPO ($\beta_t$) 与 PPO ($\beta_{\max}$) 的gradient差：
$$\left\|\nabla_\theta \log \pi_{\theta,\beta_t}(\tilde{a}|s) - \nabla_\theta \log \pi_{\theta,\beta_{\max}}(\tilde{a}|s)\right\| = \frac{1}{\sigma^2} \left|\phi(\beta_t) - \phi(\beta_{\max})\right| \cdot \left\|\nabla_\theta \mu_\theta(s)\right\|$$

用mean value theorem：
$$\left|\phi(\beta_t) - \phi(\beta_{\max})\right| \leq \sup_{\beta \in [\beta_t, \beta_{\max}]} |\phi'(\beta)| \cdot |\beta_{\max} - \beta_t|$$

计算 $\phi'(\beta)$，令 $u := \tilde{a}/\beta$：
$$\phi'(\beta) = \text{arctanh}(u) - \frac{u}{1 - u^2}$$

在 $|u| \leq \tanh(0.5)$（对应 $|a/\beta| \leq 0.5$）时，sup在边界取得：
$$\sup_{|u| \leq \tanh(0.5)} \left|\text{arctanh}(u) - \frac{u}{1-u^2}\right| = \left|0.5 - \frac{\tanh(0.5)}{1 - \tanh^2(0.5)}\right| = \frac{\sinh(1) - 1}{2}$$

最终bound：
$$\left\|\nabla_\theta \log \pi_{\theta,\beta_t}(\tilde{a}) - \nabla_\theta \log \pi_{\theta,\beta_{\max}}(\tilde{a})\right\| \leq \underbrace{\frac{\sinh(1) - 1}{2\sigma^2} \cdot |\beta_{\max} - \beta_t|}_{=: C} \cdot \left\|\nabla_\theta \mu_\theta\right\|$$

**Intuition**：gradient distortion正比于 $|\beta_{\max} - \beta_t|$。在early training $\beta_t$ 小，distortion大（但有正向作用，因为我们想要smaller action space）；在late training $\beta_t \to \beta_{\max}$，distortion $\to 0$，GPO收敛到standard PPO的gradient。这是个**self-correcting** property——distortion在最需要的时候（late stage）自动消失。

这个 $\sinh(1) - 1)/2 \approx (1.175 - 1)/2 \approx 0.087$ 是个相当小的常数，所以即使 $\beta_t$ 离 $\beta_{\max}$ 还有距离，gradient distortion的absolute magnitude也是bounded的。

---

## 4. 早期Stage分析：为什么Smaller Action Space更Efficient

这是paper最有意思的部分，用stochastic optimization的标准tools（strong convexity, L-smoothness, SGD convergence）来quantify "smaller action space → faster convergence"。

### 4.1 Gradient Variance

single-sample policy gradient estimator：
$$g_t = \nabla_\theta \log \pi_\theta(\tilde{a}_t | s_t) \cdot A_t$$

其中 $A_t$ 是advantage，假设 $\mathbb{E}[A_t] = 0$, $\mathbb{E}[A_t^2] = \sigma_A^2 < \infty$。

paper证明（Appendix C）：
$$\text{Var}[g_t] \leq c \cdot \beta_t^2, \quad c := \sigma_A^2 \cdot K^2, \quad K = \frac{2}{\sigma^2} \|\nabla_\theta \mu_\theta(s)\|$$

**关键insight**：gradient variance与 $\beta_t$ **平方**成正比。早期 $\beta_t$ 小，variance小；asymptotically $\beta_t \to \beta_{\max}$，variance恢复到PPO水平。

证明sketch：因为 $h^{-1}(\tilde{a}) = \beta \cdot \text{arctanh}(\tilde{a}/\beta)$，且 $|\text{arctanh}(x)| \leq |x|/(1-|x|)$ for $|x|<1$。设 $|\tilde{a}| \leq r\beta$ with $r = 1/2$，则 $|h^{-1}(\tilde{a})| \leq \beta \cdot r/(1-r) = \beta$。再加 $|\mu_\theta| \leq M\beta$，得 $|h^{-1}(\tilde{a}) - \mu_\theta| \leq 2\beta$，所以 $\|\nabla_\theta \log \pi_\theta(\tilde{a}|s)\| \leq (2\beta/\sigma^2) \|\nabla_\theta \mu_\theta\| = K\beta$。乘以 $A_t$ 取variance即得。

### 4.2 Signal-to-Noise Ratio (SNR)

SNR的定义：
$$\text{SNR}(\beta_t) \approx \frac{S_0}{\sqrt{c}} \cdot \frac{1}{\beta_t}, \quad S_0 := \|\mathbb{E}[g_t]\|_{\beta_t = \beta_{\min}}$$

**Intuition**：SNR与 $\beta_t$ 成反比。Action space小→SNR大→learning signal cleaner→每一步update更reliable。这解释了为什么GPO early training能更快学到stable control。

SNR在stochastic optimization里是convergence的关键。SNR太低意味着gradient estimator的noise dominate signal，参数update方向几乎是random walk。在legged robot这种high-dim continuous control里，early training的SNR本来就低（advantage估计noise大、policy output方向乱），如果action space还大，SNR会被进一步压低。

### 4.3 Convergence Error Bound (SGD-style Analysis)

设 $J(\theta)$ 是expected return，$\theta^*$ 是local optimum（$\nabla J(\theta^*) = 0$）。假设 $J$ 满足 $\mu$-strong convexity和 $L$-smoothness，固定step size $\eta \leq \mu/L^2$。

由standard SGD analysis（Appendix D给出完整推导）：
$$\mathbb{E}\|\theta_t - \theta^*\|^2 \leq \underbrace{(1 - \eta\mu)^t \|\theta_0 - \theta^*\|^2}_{\text{transient, exponential decay}} + \underbrace{\frac{\eta}{\mu} c \beta_t^2}_{\text{steady-state noise floor}}$$

**Intuition**：第一项是transient error，指数衰减（与step size和strong convexity有关）；第二项是steady-state error，与gradient variance（即 $\beta_t^2$）成正比。

对fixed action space ($\beta_t = \beta_{\max}$)：steady-state error是 $\frac{\eta c}{\mu} \beta_{\max}^2$。
对GPO早期 ($\beta_t \ll \beta_{\max}$)：steady-state error是 $\frac{\eta c}{\mu} \beta_t^2 \ll \frac{\eta c}{\mu} \beta_{\max}^2$。

所以在相同step数 $T_1$ 后，GPO的参数距离local optimum更近。

### 4.4 Early-Stage Return Advantage

设两个training protocol，相同step size $\eta$：
- **GPO**: 早期用小 $\beta_t$，逐步增长
- **Fixed baseline**: 始终用 $\beta_{\max}$

paper证明（Appendix E）在 $\mu$-strong concavity的basin内：
$$\mathbb{E}[J(\theta_{T_1})] \geq J(\theta^*) - \frac{\mu}{2}\left(\varepsilon^2 + \frac{\eta c}{\mu} \beta_{T_1}^2\right)$$
$$\mathbb{E}[J(\bar{\theta}_{T_1})] \geq J(\theta^*) - \frac{\mu}{2}\left(\varepsilon^2 + \frac{\eta c}{\mu} \beta_{\max}^2\right)$$

其中 $\varepsilon$ 是decayed transient term，$\beta_{T_1} < \beta_{\max}$。

**因为 $\beta_{T_1} < \beta_{\max}$，GPO的lower bound更紧**，即expected return严格更高。

这个结果的物理直觉：smaller action space等于在parameter space加了implicit regularization——policy只能在"低能量"的action manifold上探索，这天然地prefer stable low-amplitude control policies。这些policies在early training的contact-rich dynamics下更容易稳定执行，产生的trajectories更informative（不crash），advantage估计更准确，整个learning loop更efficient。

---

## 5. 晚期Stage分析：为什么Asymptotic不会变差

这是paper另一半的理论保证——gradually grow action space不能sacrifice asymptotic performance。

### 5.1 Local Quadratic Approximation

在 $\theta^*$ 附近（stationary point），return局部quadratic：
$$J(\theta) = J(\theta^*) - \frac{1}{2}(\theta - \theta^*)^\top H (\theta - \theta^*), \quad \mu I \preceq H \preceq L I$$

其中 $H$ 是local curvature (Hessian)，positive definite。

stochastic update：
$$\theta_{t+1} = \theta_t + \eta(\nabla J(\theta_t) + \xi_t), \quad \mathbb{E}[\xi_t] = 0, \quad \mathbb{E}[\|\xi_t\|^2] \leq c\beta_t^2$$

在quadratic近似下 $\nabla J(\theta) = -H(\theta - \theta^*)$，所以error dynamics是线性随机递归：
$$e_{t+1} = (I - \eta H) e_t + \eta \xi_t, \quad e_t := \theta_t - \theta^*$$

如果 $\eta \leq \mu/L^2$，则 $(I - \eta H)$ 是contraction，存在 $\rho \in (0, 1)$ 使得 $\|I - \eta H\|_2^2 \leq \rho$。

### 5.2 Steady-State Error Bound

Unrolling recursion并取 $t \to \infty$：
$$\limsup_{t\to\infty} \mathbb{E}\|e_t\|^2 \leq \frac{\eta^2 c}{1 - \rho} \beta_\infty^2$$

其中 $\beta_\infty := \limsup_{t\to\infty} \beta_t$。

对fixed baseline（$\beta_t \equiv \beta_{\max}$）：
$$\limsup_{t\to\infty} \mathbb{E}\|\bar{e}_t\|^2 \leq \frac{\eta^2 c}{1 - \rho} \beta_{\max}^2$$

因为GPO设计上 $\beta_\infty \leq \beta_{\max}$（如果 $f(t) \to 1$ 严格，则 $\beta_\infty = \beta_{\max}$；如果 $f(t)$ 在某个值 $<1$ 处saturate，则 $\beta_\infty < \beta_{\max}$），所以：
$$\limsup_{t\to\infty} \mathbb{E}\|e_t\|^2 \leq \limsup_{t\to\infty} \mathbb{E}\|\bar{e}_t\|^2$$

由quadratic form $J(\theta) = J(\theta^*) - \frac{1}{2} e^\top H e$ 和 $H \preceq LI$：
$$\mathbb{E}[J(\theta_t)] \geq J(\theta^*) - \frac{L}{2} \mathbb{E}\|e_t\|^2$$

所以：
$$\liminf_{t\to\infty} \mathbb{E}[J_{\text{GPO}}] \geq \liminf_{t\to\infty} \mathbb{E}[J_{\text{fixed}}]$$

**Intuition**：GPO在asymptotic regime下不会比fixed baseline差，可能更好（如果 $\beta_\infty < \beta_{\max}$ 严格，或者GPO早期converge到更好的basin）。

paper说的"potentially improved"在我看来有点subtle。如果 $f(t)$ 严格趋近1，那 $\beta_\infty = \beta_{\max}$，steady-state error相同，asymptotic return相同。但实际实验里GPO比PPO asymptotic更好——这部分可能来自**basin selection effect**：GPO在early stage converge到的local basin本身就比PPO wide-explore收敛到的basin更好，所以asymptotically在更好的basin内优化。

这部分的theoretical analysis其实没有完全capture实验现象。我觉得这是paper的一个weak spot——late-stage advantage的empirical evidence远强于theoretical guarantee。可能需要更精细的分析（比如basin切换的概率、non-local convergence）才能fully explain。

---

## 6. Reward Design的细节：Command也Scale

这是个我差点miss的细节。看Appendix F.4 (Tab. IV)：

```
r_tracking,x = φ(v_x - v_x^cmd * β_t)   weight 10dt
r_tracking,yaw = φ(ω_yaw - ω_yaw^cmd * β_t)   weight 5dt
r_tracking,pitch = φ(θ - θ^cmd * β_t)   weight 5dt
r_tracking,height = φ(h_b - h_b^cmd * β_t)   weight 7dt
```

**注意command也乘了 $\beta_t$**！

这个设计的intuition：如果early stage action space小（机器人只能输出小torque），那要求机器人track $v_x^{\text{cmd}} = 1$ m/s是不合理的（机器人物理上做不到）。所以command也scale down：早期只要求track $v_x^{\text{cmd}} \cdot \beta_t = 0.05$ m/s这样的small velocity，随着 $\beta_t$ 增长commanded velocity也增长。

这是task difficulty curriculum和action capacity curriculum的**同步**。光restrict action space是不够的，要同时restrict task demand，否则reward signal会被infeasible command dominate，policy学不到有用的gradient。

这个细节paper没有在main text里highlight，但它其实是GPO能work的关键之一。理论上paper只分析了action transformation的影响，没分析command scaling的影响——后者是pure engineering。如果只用GPO transformation但command不scale，early stage的reward可能被"commanded velocity但实际爬不动"的negative signal淹没。

Reference: [SATA paper](https://arxiv.org/abs/2502.12674)（疲劳建模的来源，也用了类似的reward scaling idea）

---

## 7. 架构与实验设置

### 7.1 Actor-Critic Architecture (DREAMWAQ-style)

paper采用DREAMWAQ的asynchronous actor-critic架构。关键components：

**Observation** (actor input + critic input):
$$o_t = [w_t, g_t, q, \dot{q}, v_{\text{cmd}}, \tau, \zeta_t]^\top$$

变量含义：
- $w_t$ — base angular velocity (3-dim, IMU)
- $g_t$ — projected gravity (3-dim, IMU)
- $q, \dot{q}$ — joint positions and velocities (12-dim for quadruped, 18-dim for hexapod)
- $v_{\text{cmd}}$ — command velocities (task-dependent)
- $\tau$ — torque feedback (从motor读回来)
- $\zeta_t$ — fatigue state (SATA-style)

**Estimator** (DREAMWAQ的核心):
- 输入：observation history $o_{t-10:t}$（10步window）
- 输出：$e_t = [\hat{v}_t, z_t]$
  - $\hat{v}_t$ — base linear velocity的估计（privileged信息，训练时用真值监督，部署时只靠proprioceptive）
  - $z_t$ — latent dynamics representation（next observation prediction的副产品）

**Actor**: condition on $(o_t, e_t)$，输出joint torques。
**Critic**: 只condition on $o_{t-10:t}$（不condition on estimator output），估计state value。

这个架构的核心是teacher-student distillation：训练时critic可以"看到"privileged info，actor只看proprioceptive + estimator output；部署时estimator的输出仍然是proprioceptive-only的，但已经"内化"了训练时的privileged info。

Reference: [DREAMWAQ paper](https://arxiv.org/abs/2305.06563)

### 7.2 Fatigue Modeling (SATA)

$$\zeta_t = (\zeta_{t-1} + |\tau| \cdot dt) \cdot \gamma, \quad \gamma = 0.95$$

变量：
- $\zeta_t$ — fatigue accumulator (每个joint独立)
- $|\tau|$ — 当前torque magnitude
- $dt$ — control time step (= 0.005s in experiments)
- $\gamma$ — decay factor

这是一个exponentially-decaying integral of torque magnitude。Intuition：长时间high-torque输出会让joint进入"fatigued"状态，policy需要学会distribute load到多个joint避免任何单一joint过载。Reward里有 $r_{\text{fatigue}} = -\zeta \cdot |\tau_d \cdot \kappa_{\text{scale}}|$ 项penalize fatigued state下的高torque输出。

### 7.3 No Gait Priors

paper强调：**no phase synchronization, no predefined footfall schedules, no leg-usage constraints**。Gait完全emergent。

这点挺有意思。在legged robot RL里，gait priors（如CPG-based regularization、predefined duty cycle、phase rewards）非常常见，因为它们能极大加速convergence到reasonable locomotion pattern。但paper deliberately去掉了这些，让GPO自己emerge gait——这是为了isolate GPO本身的effect，证明GPO不需要inductive bias也能学到好policy。

这也让gait emergence成为evaluation metric：GPO产生的gait是不是reasonable？Fig. 5和Fig. 4显示GPO有清晰的periodic stepping pattern，PPO则degenerate（部分legs dominate support，其他legs under-utilized）。

Reference: [CPG-RL](https://arxiv.org/abs/2208.05035), [Energy→Gait emergence](https://arxiv.org/abs/2111.01674)

---

## 8. 实验数据深度解读

### 8.1 Growth Function Ablation (Fig. 3)

四种growth function的训练曲线对比（quadruped和hexapod）。我的观察：

1. **PPO (no growth)** 在quadruped上convergence最慢，在hexapod上甚至不converge到GPO水平。这印证了"wide exploration early = high gradient variance = slow learning"的论点。

2. **Linear growth** 在early stage有一定improvement but late stage plateau很低。Intuition：linear growth在early stage grow太快，policy还没建立stable low-amplitude control就被unlock到high-amplitude space，导致return collapse或suboptimal convergence。

3. **Sigmoid** 比linear好，因为S-curve的slow start给了early stage更多时间在restricted space里学。但symmetric S-curve的slow end意味着action space在late stage还没fully unlock，限制了exploration。

4. **Gompertz** 最好。asymmetric shape完美匹配learning dynamics：early stage需要强restriction (slow start) → stable convergence to local basin；middle stage需要rapid unlock (fast middle) → escape early basin去explore；late stage需要充分unlock (slow end但已接近1) → refine asymptotic policy。

这个ablation在我看来是paper最强的empirical evidence。它把"是否要grow action space"这个问题细化为"how to schedule the growth"。任何grow都比不grow好，但grow schedule的shape决定asymptotic performance。

### 8.2 Tracking Error (Tab. II)

| Metric | GPO | PPO |
|---|---|---|
| $v_x, v_y$ (m/s, quadruped) | 0.015 ± 0.035 | -0.30 ± 0.10 |
| $v_x, v_y$ (m/s, hexapod) | 0.00 ± 0.05 | -0.10 ± 0.05 |
| $H_b$ (m, quadruped) | 0.005 ± 0.0025 | -0.03 ± 0.02 |
| $H_b$ (m, hexapod) | 0.02 ± 0.005 | -0.12 ± 0.03 |

PPO在quadruped上 $v_x$ tracking error是 -0.30 m/s——这意味着commanded 0.5 m/s，机器人实际只能爬到0.2 m/s。这是严重的underactuation symptom，说明PPO policy没有学会effective propulsion。GPO的error只有0.015 m/s，准确度高一个数量级。

$H_b$ 也有显著差距：PPO在hexapod上commanded 0.25m但只能达到0.13m（error -0.12m），说明base一直在下沉，没有维持upright posture的能力。GPO则准确维持target height。

这些quantitative gap背后的机制：PPO在大action space早期学到的是"软"的low-authority policy（因为大action variance下high-torque action的disadvantage被average掉），而GPO在restricted action space里被forced学会如何用limited torque budget维持upright posture和accurate tracking。当 $\beta_t$ grow后，GPO policy把已经学到的"低authority精细control"能力扩展到"高authority大action"场景，resulting in更efficient和accurate的control。

### 8.3 Joint-Level Periodicity (Fig. 4)

paper展示了front-left leg的joint torque和velocity时间序列。GPO的曲线有clear periodicity（gait cycle可辨识），PPO的曲线erratic。

**Intuition**：periodic joint trajectories是well-coordinated gait的signature。生物力学上，efficient locomotion就是周期性的（muscle activation pattern repeat each cycle）。GPO通过early-stage的restricted action space强制policy学到"低能量、可重复"的control pattern，这种pattern在action space grow后保持，形成emergent periodic gait。PPO在wide action space里随机探索，可能学到的是non-periodic、energy-wasting的compensatory pattern。

### 8.4 Hardware Validation (Tab. III)

| Test | GPO | PPO | DeCAP (Torque) | DeCAP (Position) |
|---|---|---|---|---|
| Side push | 100% | 0% | 0% | 80% |
| Challenging terrain | 100% | 20% | 0% | 40% |
| Vertical stomp | 100% | 40% | 60% | 70% |

GPO在所有test都是100% success rate。PPO在side push直接0%——机器人被5kg哑bell砸一下直接倒。

注意DeCAP (Position) 表现也不错，因为position control有low-level PD servo做stabilization，本身就更robust。但DeCAP (Torque) 表现差，说明torque-based control本质难，单纯用DeCAP的decaying action prior不够。GPO在torque-based control下能达到100% robustness，是strong result。

Reference: [DeCAP paper](https://arxiv.org/abs/2410.08458)

---

## 9. 我的思考与可能的Limitations

### 9.1 Strengths

1. **Theoretical cleanliness**: update equivalence证明很漂亮，gradient distortion bound给了一个self-correcting property。整个analysis在standard stochastic optimization framework下展开，assumptions合理。

2. **Empirical consistency**: 实验结果和理论预测高度consistent——early stage reward上升快（high SNR），asymptotic return高（no-worse basin selection），joint trajectories有periodicity（emergent gait）。

3. **Generality**: GPO不改policy architecture、不改reward design（除了command scaling这个minor detail）、不改训练pipeline。它只在action transformation层面介入，是orthogonal to其他improvements（curriculum、teacher-student、domain randomization）。

### 9.2 Open Questions

1. **$\beta_t$ schedule的hyperparameter sensitivity**: Gompertz的 $k = 3\times 10^{-5}$ 和 $t_0 = 2.4\times 10^4$ 是怎么选的？是否task-dependent？如果换一个robot（比如humanoid 30+ DOF）这些参数还work吗？paper没做这个sensitivity analysis。

2. **Command scaling的theoretical justification**: 我前面highlight过，reward里的 $v^{\text{cmd}} \cdot \beta_t$ 是个隐藏的curriculum mechanism。paper的理论分析只覆盖action transformation，没cover command scaling。如果只有action transformation但command不scale，GPO还能work吗？这是个值得ablate的细节。

3. **Asymptotic "potentially improved"的机制**: paper证明了no-worse，但实验里GPO比PPO asymptotic好很多。这个gap的理论解释缺失。我猜可能的原因：
   - GPO在early stage converge到更好的basin（basin selection effect）
   - GPO的training trajectory更stable，避免了PPO的catastrophic policy collapse
   - 实际 $\beta_\infty < \beta_{\max}$（即使 $f(t) \to 1$ 严格，但finite training steps下没真正reach 1）

4. **Action distribution assumption $|a/\beta_t| \leq 0.5$**: 这个假设在experiments里被验证（Fig. 4显示torque远小于limit），但是否在更aggressive task（如jumping、high-speed running）下还成立？这些任务需要burst of high-torque，policy可能经常hit boundary。

5. **Continuous vs Discrete Growth**: GPO用continuous $\beta_t$ growth，但实际训练是minibatch SGD。每个minibatch的 $\beta_t$ 不同，可能导致intra-epoch的distribution shift。是否考虑epoch-level step growth（每个PPO epoch固定 $\beta_t$，epoch之间grow）会更stable？

6. **Beyond Locomotion**: paper只在legged locomotion上测试。GPO的核心idea——restrict early action space然后gradually expand——对其他high-dim continuous control task（如manipulation、autonomous driving）也适用。期待看到更多domain的validation。

### 9.3 联想到的相关工作

- **Curriculum Learning** (Bengio 2009): GPO可以看作action-space-level curriculum。区别是curriculum通常指task difficulty，GPO指action capacity。
  Reference: [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380)

- **Annealing in RL**: $\epsilon$-greedy的 $\epsilon$ annealing、softmax temperature annealing、SAC的entropy coefficient tuning都是逐渐放宽exploration的mechanism。GPO是action range annealing，orthogonal to这些。
  Reference: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

- **Action Space Shaping in Continual Learning**: 我记得有一些work在manipulation里用progressive action space（先学7-DOF end-effector，再学finger DOF），但不是stochastic policy下的smooth transformation。GPO的smooth transformation更principled。

- **Domain Randomization Progressive**: 从narrow domain randomization到wide，和GPO从narrow action space到wide是parallel ideas。
  Reference: [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978)

- **Trust Region Methods**: TRPO/PPO的trust region是parameter space的，GPO的"trust region"是action space的——限制early stage的action magnitude来stabilize exploration。两个orthogonal的trust region concepts。

- **Action Clipping in Legged Robot RL**: 这是legged robot community的"folk wisdom"。ETH的RSL、Berkeley的Hybrid Robotics、MIT的Biomimetic Robotics都在code里用tight torque limit early然后relax。GPO把这个folk wisdom正儿八经形式化了。
  Reference: [Learning quadrupedal locomotion over challenging terrain](https://www.science.org/doi/10.1126/scirobotics.abc5986)

- **Gompertz Function in Biology**: Gompertz最初是model tumor growth和population dynamics的。Asymmetric S-shape是很多biological growth process的characteristic。GPO借用它来model action space growth，可能不是coincidence——motor learning in animals可能也follow类似的asymmetric schedule。
  Reference: [Gompertz function](https://en.wikipedia.org/wiki/Gompertz_function)

---

## 10. Implementation Pseudo-code

为了build concrete intuition，我把GPO的训练loop写出来：

```python
# Pseudocode
β_max = a_limit  # hardware limit
f = lambda t: exp(-exp(-k * (t - t0)))  # Gompertz
β = lambda t: β_max * f(t)

for t in range(total_steps):
    # Collect rollouts
    for step in rollout:
        # Sample latent action from Gaussian policy
        μ = actor_θ(obs)
        a = sample(N(μ, σ²))  # σ fixed, not learned
        
        # GPO transformation
        β_t = β(t)
        ã = β_t * tanh(a / β_t)
        
        # Execute ã in environment (joint torques)
        next_obs, reward, done = env.step(ã)
        
        # Store (obs, a, ã, reward, ...) in buffer
        # Note: importance ratio uses a (latent), not ã
    
    # Compute advantages (GAE)
    advantages = GAE(...)
    
    # PPO update (uses importance ratio on a, not ã)
    for epoch in range(K_epochs):
        for batch in buffer:
            # Importance ratio on latent action a
            ratio = exp(log_prob_θ(a) - log_prob_θold(a))
            
            # Standard PPO clipped objective
            clipped_ratio = clip(ratio, 1-ε, 1+ε)
            loss = -min(ratio * advantage, clipped_ratio * advantage).mean()
            
            # Backprop and update θ
            θ.grad = autograd(loss)
            optimizer.step(θ)
```

关键implementation点：
1. **importance ratio用latent action $a$ 算**，不用 $\tilde{a}$。因为Jacobian term抵消，直接用latent action更简单。
2. **环境执行用 $\tilde{a}$**，但log_prob存在buffer里的是latent $a$ 的log_prob。
3. **$\beta_t$ 每个rollout step更新**，但每个PPO epoch内是固定的（rollout阶段continuous growth，update阶段可以treat成近似fixed）。
4. **reward里的command也scale by $\beta_t$**：`reward = φ(v_x - v_x_cmd * β_t)`。

---

## 11. Final Thoughts

这篇paper给我的最大启发是：**很多RL的"folk wisdom"工程技巧，背后其实有深刻的optimization theory**。Action clipping、torque limit relaxation这些都是legged robot community做了好几年的事情，但从来没人把它写成带理论保证的algorithm。GPO做到了，而且做得很干净——既没破坏PPO的structure，又通过standard stochastic optimization analysis给了quantitative convergence guarantees。

从更宏的视角看，GPO其实是"**Action Space Curriculum**"的instantiation。Curriculum learning传统上指task difficulty curriculum（terrain难度、disturbance强度等），GPO把它扩展到action capacity curriculum——agent先在小action space里学会precise control，再扩展到大action space里学agile maneuver。

这个idea我觉得还可以进一步推广：
- **State space curriculum**: 早期restrict observation space（只用IMU，后期加vision）
- **Reward shaping curriculum**: 早期用sparser reward，后期用dense reward
- **Discount factor curriculum**: 早期short horizon (small γ)，后期long horizon (large γ)

这些都是progressive unlocking的instances，背后可能都有类似的optimization theory（variance reduction、SNR improvement、basin selection）。

我之前在nanoGPT里训练GPT-2时也观察到类似的phenomenon：用curriculum（先训短sequence，再grow到长sequence）比直接训长sequence更efficient。这背后可能是同一个principle——**progressive capacity unlocking matches the natural curvature of the learning landscape**。

Reference: [nanoGPT](https://github.com/karpathy/nanoGPT), [Karpathy's PPO from scratch](https://github.com/karpathy/cosi/blob/master/cosi.py)

总之GPO是个clean、principled、empirically strong的工作。它不会替代PPO，但它给PPO加了一个orthogonal的improvement维度，可以和其他techniques（curriculum、teacher-student、domain randomization）叠加。我预期这个idea会被legged robot community广泛adopt，因为它truly plug-and-play。
