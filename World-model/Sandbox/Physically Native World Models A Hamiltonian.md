---
source_pdf: Physically Native World Models A Hamiltonian.pdf
paper_sha256: 2a06dde7880c1b9cfbbc2845ae8545e7046c118dd57989cdf207411033b79c41
processed_at: '2026-08-06T03:32:25-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话总结

这篇 paper 在说:**现在的 video world model (Sora 那类) 能生成漂亮视频,但根本不懂物理;要真正支持机器人决策,latent dynamics 必须有物理结构,而 Hamiltonian mechanics 是个现成的好骨架。**

## 痛点在哪

你让 Sora 生成一个球撞墙的视频,它给你画得像真的一样。但你问它"撞完之后球往哪飞、动量多少、能量守不守恒",它答不上来,因为它压根没在 modeling 物理,它在 modeling "像素怎么继续走"。

具体表现为:
- 长 rollout 后物体可能凭空消失或凭空出现 (object permanence 失效)
- 撞击后 momentum 不守恒
- Action 对 future 的影响很弱——你 pushing 一下和 pulling 一下,生成的 video 区别不大
- 数据需求巨大,因为模型得从零发现牛顿定律

对娱乐视频这无所谓。对机器人这致命——机器人要 plan,要 imagine "如果我推这把,杯子会掉下去吗",predicted future 必须物理上靠谱。

## 核心招数:从"学速度场"换成"学能量场"

标准 latent dynamics 是:
$$z_{t+1} = f_\theta(z_t, a_t)$$

这就是学一个 vector field——"给定状态,告诉我下一步速度是多少"。这个 $f$ 任意,没有结构,什么都可能发生。

Hamiltonian 改成学一个 scalar energy function $H(q,p)$,然后 dynamics 从能量梯度自动推出来:
$$\dot{q} = \partial H / \partial p, \quad \dot{p} = -\partial H / \partial q$$

这个看似 subtle 的改动 consequence 巨大:

1. **Output 维度从 $2n$ 降到 1**:学一个标量比学一个向量场简单多了
2. **Energy 守恒是免费的**: Hamiltonian flow 数学上自动保证 $dH/dt = 0$,不需要加 loss 强迫模型守恒
3. **Generalization 强**: 学到的是能量地形,换一组初始条件还是同一个能量地形,dynamics 自然 generalize
4. **Long-horizon 不会爆**: Hamiltonian flow 保 phase space 体积 (Liouville 定理),rollout 1 万步都不会 blow up

类比: 学势能面 vs 学力场。势能是 scalar,画出来一目了然;力是 vector field,每点都得标方向大小。学势能再求负梯度就是力——这一步是 free 的。

## Architecture 四件套

整条 pipeline 是:
$$o_t \to z_t \to \hat{z}_{t+1:t+H} \to \hat{o}_{t+1:t+H} \to a^*$$

**(1) Encoder**: 把 video 压成 object-centric phase space $z = \{(q^i, p^i)\}_{i=1}^N$。每个物体一组 (位置, 动量)。别压成一个 global vector——物理世界是 compositional 的。

**(2) Hamiltonian core**: 学能量函数,通常拆三块:
$$H = \sum_i K(p^i) + \sum_i U(q^i) + \sum_{i<j} V(q^i, q^j)$$
- $K$: 动能 (每个物体自己的)
- $U$: 势能 (每个物体自己的,比如重力)
- $V$: 物体间的 interaction (弹簧、引力、Lennard-Jones 之类)

然后 Hamilton 方程自动给出 dynamics。

**(3) Decoder**: 把低维 phase state 渲染回高维 video frames。这步是必要的——$(q,p)$ 太抽象,没法直接给出 pixel。Decoder 可以是 latent video diffusion。

**(4) Planner**: 试一堆 candidate action,每个 roll out 一条 trajectory,用 utility function 选最优。就是标准 MPC。

**关键 insight**: 物理演化和视觉渲染要分离但耦合。Phase space 给 physics structure,video decoder 给 perceptual richness。两边各管一摊,互相 condition。

## 真实世界不保守怎么办

标准 Hamiltonian 只描述 closed conservative system。真实机器人有:
- Friction / damping (能量耗散)
- Control input (机器人施加力)
- Contact discontinuity (撞击是 non-smooth 的)
- Unmodeled stuff

Paper 给的解是 port-Hamiltonian 加两项加残差:
$$\dot{z} = (J - R(z))\nabla H(z) + G(z)a + \epsilon(z, a)$$

- $J\nabla H$: 标准 Hamiltonian,守恒核心
- $-R\nabla H$ (其中 $R \succeq 0$): 耗散项,数学上自动保证 $dH/dt \leq 0$,能量只减不增,正好是 friction/damping
- $G a$: action 怎么注入 system
- $\epsilon$: residual,catch-all 处理剩下的

加上 $\|\epsilon\|^2$ 正则化,逼模型用 Hamiltonian 解释 dynamics,不要 lazy 地全交给 residual。

这个 form 来自 port-Hamiltonian framework,控制理论里很成熟 (van der Schaft 的工作)。

## 和已有 work 的关系

这篇 paper 其实是把一条已有脉络拔高到 "embodied world model" 的高度:

- **HNN (Greydanus 2019)**: 鼻祖。证明学 $H$ 比学 $f$ 在简单系统 (spring, pendulum, two-body) 上 energy conservation 好太多。https://arxiv.org/abs/1909.10690
- **HGN (Toth 2020)**: 加 encoder/decoder,从 pixels 学 phase space。证明这套能 work 在视觉输入上。https://arxiv.org/abs/1909.13789
- **SymODE-Net (Zhong 2020)**: 加 control 和 dissipation,做 controlled pendulum / cartpole。https://arxiv.org/abs/1909.12077
- **本文 HWM**: 把上面全部 + object-centric + video diffusion decoder + MPC planning 缝起来,claim 这套是 embodied world model 的 physical backbone。

对比 Dreamer 系列: Dreamer 用 RNN+MLP 当 transition,无物理 prior,靠 stochastic latent + 大数据硬学。HWM 有 strong prior,理论上更 data efficient、更 stable,但假设更强。

对比 Sora / GAIA-1 / Genie: 这些是纯 video generation,没 latent dynamics structure。HWM 想同时拿 visual richness 和 physical validity。

## 真正的 risk 在哪

(1) **从 pixels 推 (q,p) 这步本身就难**: 你给模型一张桌子上一堆物体的图,它怎么知道每个物体的 $q$ 是什么?$p$ 怎么估?occlusion 怎么办?HGN 在 pendulum 这种单物体上 work,多物体、deformable、occluded 还没证明。可能需要 equivariant architecture (e3nn, ESCNN) 或 identifiability theory。

(2) **Contact 是 non-smooth 的,Hamiltonian flow 是 smooth ODE**: 撞击瞬间速度跳变,数学上不是 ODE 能描述的,是 hybrid system 或 complementarity problem。Paper 用 $\epsilon$ 兜底,但这其实把难点藏起来了。

(3) **Scale 没证明**: HNN 类工作都在 low-dim 玩具系统。100 个物体 + 高维 visual + 真 robot 还没人做出来。

(4) **Latent identifiability**: 学到的 $(q,p)$ 真的对应物理 $(q,p)$ 吗?可能学到一个 entangled representation 名字叫 $(q,p)$ 但没物理意义。这是 fundamental 问题。

(5) **Video decoder 可能 drift**: phase space 看着对,但 decoder render 出来漂了。怎么 align 这两者是工程难题。

## 我觉得 paper 的真正贡献

老实说,这篇 paper 是 position paper,没做实验,核心 idea (Hamiltonian + encoder/decoder + control) 在 HGN / SymODE-Net 里都有。真正新的东西是:

1. **Framing**: 把 Hamiltonian dynamics 明确拔高到 "embodied world model 的 physical backbone" 这个高度,作为对抗 Sora-style video WM 的 alternative paradigm
2. **Object-centric phase space**: 强调 $z = \{(q^i, p^i)\}$ 而不是单一 latent,这点之前 Hamiltonian 系工作没强调够
3. **Architecture 完整化**: encoder + Hamiltonian + dissipation + control + residual + video decoder + MPC,把整条 chain 拼齐
4. **Table 1 对比**: 把四类 world model (video / 3D / JEPA / Hamiltonian) 放一起对比,清晰

但 paper 没解决 fundamental difficulty (pixel→phase identifiability, contact, scaling),只是 propose vision。如果后面有人真在 real robot 上做出来比 DreamerV3 好的 HWM,这套 framing 就立住了。否则可能只是另一个 "physically-informed 但不 scalable" 的工作。

## 给你的 takeaway

如果你 (Karpathy) 在思考下一代 world model 方向,这篇 paper 值得关注的核心 idea 就一句:

**Latent space 不要是任意 vector,要是 $(q, p)$ phase space;transition 不要学 vector field,要学 scalar energy $H$ 然后 gradient 推 dynamics。这样 long-horizon 稳定性、energy conservation、generalization 都是 free 的,代价是 Hamiltonian 假设可能太强。**

这条 path 和 LeCun 的 JEPA (abstract latent prediction)、DeepMind 的 Genie (video generation)、Dreamer 的 RSSM (stochastic recurrent) 是四条不同的 bet。哪条赢还不知道,但 Hamiltonian 这条至少把 "物理结构" 当 first-class citizen 来对待,这点在 robotics 场景下直觉上是对的方向。

Reference:
- Position paper 本身: 你贴的 attachment
- HNN (核心祖宗): https://arxiv.org/abs/1909.10690
- HGN (从 pixels 学 phase space): https://arxiv.org/abs/1909.13789
- SymODE-Net (加 control + dissipation): https://arxiv.org/abs/1909.12077
- Port-Hamiltonian 理论基础: https://link.springer.com/book/10.1007/978-1-4471-0532-1
- DreamerV3 (做对比): https://arxiv.org/abs/2301.04104
- Geometric Numerical Integration (symplectic integrator 圣经): https://link.springer.com/book/10.1007/3-540-30666-8

---

# Physically Native World Models: A Hamiltonian Perspective 深度解析

## 1. Paper 核心论点

这篇 paper 的核心 thesis 可以一句话概括：**当前 world model 的 bottleneck 已经从"能不能 generate realistic future"转移到"generated future 是否 physically meaningful 且 useful for action"**。作者主张用 Hamiltonian dynamics 作为 physically grounded 的 inductive bias,来 constrain latent dynamics,从而获得 long-horizon stability、data efficiency 和 interpretability。

让我从底层 build intuition 开始讲起。

---

## 2. Hamiltonian Mechanics 速成

要理解这篇 paper,必须先深刻理解 Hamiltonian mechanics。这是 William Rowan Hamilton 在 1833 年对 Newtonian mechanics 的重新表述。

### 2.1 从 Newton 到 Hamilton

Newton 的 formulation 是:
$$m\ddot{q} = F(q, t)$$

其中 $q$ 是 generalized coordinate,$\ddot{q}$ 是 acceleration。这个 form 有几个问题：二阶、force-centric、对 constraint 处理不自然。

Lagrangian formulation 引入 action principle:
$$\mathcal{L}(q, \dot{q}) = T(q, \dot{q}) - V(q)$$
$$\delta \int \mathcal{L}\, dt = 0 \Rightarrow \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} - \frac{\partial \mathcal{L}}{\partial q} = 0$$

定义 conjugate momentum:
$$p = \frac{\partial \mathcal{L}}{\partial \dot{q}}$$

Hamiltonian formulation 通过 Legendre transform 把 Lagrangian 转换为 phase space $(q, p)$ 上的 first-order system:
$$H(q, p) = p\dot{q} - \mathcal{L}(q, \dot{q})$$

通常 $H = T + V$ 即 kinetic energy + potential energy。

### 2.2 Hamilton's Equations

这是 paper 中 Equation (11) 和 (12) 的来源:
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

写成 matrix form (paper Equation 12):
$$\dot{z} = J \nabla_z H(z), \quad J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}$$

**变量解释**:
- $z = [q, p]^T \in \mathbb{R}^{2n}$:phase space state,$n$ 是 degrees of freedom
- $q \in \mathbb{R}^n$:generalized coordinates,描述 system configuration
- $p \in \mathbb{R}^n$:generalized momenta,描述 system motion state
- $H: \mathbb{R}^{2n} \to \mathbb{R}$:Hamiltonian function,通常等于 total energy
- $J \in \mathbb{R}^{2n \times 2n}$:symplectic matrix,定义 phase space 的几何结构
- $I \in \mathbb{R}^{n \times n}$:identity matrix

**Key intuition**: Newtonian/Lagrangian 是从 "force causes acceleration" 出发,Hamiltonian 是从 "energy landscape generates dynamics through its gradient" 出发。这是个 fundamental shift。

### 2.3 为什么 Hamiltonian 对 World Model 重要？

Hamiltonian system 有几个 deep properties：

**(1) Energy conservation**: 
$$\frac{dH}{dt} = \frac{\partial H}{\partial q}\dot{q} + \frac{\partial H}{\partial p}\dot{p} = \frac{\partial H}{\partial q}\frac{\partial H}{\partial p} - \frac{\partial H}{\partial p}\frac{\partial H}{\partial q} = 0$$

closed system 中 energy 自动守恒。这是 strong inductive bias。

**(2) Symplectic structure**: $J$ matrix 定义了 phase space 上的 symplectic 2-form $\omega = dq \wedge dp$。Hamiltonian flow preserves 这个 form (Liouville's theorem):
$$\mathcal{L}_X \omega = 0$$

这意味着 phase space volume 在 flow 下保持不变。这是 topological stability 的根源。

**(3) Time reversibility**: Hamiltonian system 是 time-reversible 的,因为 $H$ 是 scalar function。

**(4) Geometric integration**: 用 symplectic integrator (e.g., leapfrog/Störmer-Verlet) 做 numerical integration 时,长时间 simulation 不会 energy drift,这是 generic neural ODE 做不到的。

对 world model 而言,这些 property 意味着:
- Long-horizon rollout 不会 blow up
- 在 conservative 部分用 strong physics prior 节省 data
- Latent space 有 interpretable 的物理 meaning

**Reference**: 
- Hairer, Lubich, Wanner "Geometric Numerical Integration" (Springer, 2006): https://link.springer.com/book/10.1007/3-540-30666-8
- Arnold "Mathematical Methods of Classical Mechanics" (Springer): https://link.springer.com/book/10.1007/978-1-4757-2063-1

---

## 3. Paper 的核心 Architectural Insight

### 3.1 Existing Three Routes 的 Critique

Paper Section 2 把当前 world model 分成三类,我做一个更细的对比:

| Paradigm | 典型模型 | Latent 形式 | Transition form | 物理 inductive bias |
|---|---|---|---|---|
| 2D video-generative | GAIA-1, Genie, VideoLDM | pixels / VAE latent | $p_\theta(o_{t+1:t+H} \mid o_{\leq t}, a, c)$ | 几乎没有 |
| 3D scene-centric | NeRF, 3DGS | volumetric field / Gaussian | static reconstruction | 几何 prior, 无 dynamics prior |
| JEPA-like | I-JEPA, V-JEPA | abstract latent | predict in latent space | 无 explicit physics |
| **Hamiltonian (本文)** | HWM | $(q, p)$ phase space | $J\nabla H + G a + R + \epsilon$ | strong physics prior |

**Key distinction**: 前三类本质都把 dynamics 当成 black-box neural function 来学,Hamiltonian route 把 dynamics 当成 energy function 的 gradient-induced flow 来学。这个 difference 看似 subtle 但 consequence 很大,因为:

1. **Parameter efficiency**: 只需要学一个 scalar function $H_\phi$ 而不是一个 vector field $f_\theta$,参数空间维度从 $O(2n)$ 降到 $O(1)$
2. **Implicit constraints**: symplectic structure 自动 force 学到的 dynamics 满足 energy conservation,不需要额外 loss
3. **Generalization**: 学到的是 energy landscape,可以 generalize 到 unseen initial conditions,因为 energy 是 invariant

### 3.2 Architecture 拆解

Paper Figure 1 描述的 architecture (Equation 13):
$$o_t \xrightarrow{E_\theta} z_t \xrightarrow{\mathcal{T}_H} \hat{z}_{t+1:t+H} \xrightarrow{D_\theta} \hat{o}_{t+1:t+H} \xrightarrow{U} a_t^*$$

四个 component 各自承担不同 role:

#### (A) Perception / Encoder $E_\theta$

Input: $o_{\leq t}$ (past observations,通常 video frames)
Output: $z_t = \{(q_t^i, p_t^i)\}_{i=1}^N$ (structured phase space)

**关键设计 choice**: object-centric phase space,而不是 monolithic latent vector。每个 entity $i$ 有自己的 $(q^i, p^i)$,这 align with 物理世界是 compositional 的事实。

**实现方式可以参考**:
- Slot Attention (Locatello et al. 2020): https://arxiv.org/abs/2006.13255 - 把 image 分成 object slots
- HGN (Toth et al. 2020): https://arxiv.org/abs/1909.13789 - 直接从 pixels 学 phase space variables
- ESCNN (Cohen & Welling): equivariant representation learning

对每个 slot $i$ 学两个分支: $\mu_q^i, \mu_p^i$,作为 $(q^i, p^i)$ 的 estimate。可以加 KL regularization 让 distribution 不要 collapse。

#### (B) Hamiltonian Dynamics $T_H$

这是 paper 的核心 contribution。Paper Equation (19):
$$\dot{z}_t = (J - R_\psi(z_t))\nabla_z H_\phi(z_t) + G_\psi(z_t) a_t + \epsilon_\psi(z_t, a_t)$$

让我逐项解析:

**Term 1: $J\nabla_z H_\phi(z_t)$** — Conservative Hamiltonian core
- $J$: symplectic matrix,固定不变
- $H_\phi(z_t)$: learned Hamiltonian,通常 parameterize 为:
  $$H_\phi(z_t) = \sum_i K_\phi(p_t^i) + \sum_i U_\phi(q_t^i) + \sum_{i<j} V_\phi(q_t^i, q_t^j)$$
  
  **变量解释** (paper Equation 16):
  - $K_\phi(p^i)$:kinetic energy of entity $i$,通常 $\frac{1}{2} p^{iT} M_i^{-1} p^i$,$M_i$ 是 mass matrix
  - $U_\phi(q^i)$:potential energy of entity $i$,e.g., gravity $m g h$
  - $V_\phi(q^i, q^j)$:interaction potential between entities $i$ and $j$,e.g., spring, gravity, Lennard-Jones

  $V_\phi$ 是 relational term,通常用 GNN 或 attention 来 parameterize:
  $$V_\phi(q^1, \ldots, q^N) = \sum_{i<j} \phi_\theta(q^i, q^j)$$
  
  或者更 expressive:
  $$V_\phi = \sum_{i<j} \phi_\theta(\|q^i - q^j\|)$$

**Term 2: $-R_\psi(z_t) \nabla_z H_\phi(z_t)$** — Dissipation
- $R_\psi(z_t) \succeq 0$: positive semi-definite matrix,captures friction, damping, drag
- This term 在 Port-Hamiltonian framework 里是 standard 的 (van der Schaft):
  $$\dot{z} = (J - R)\nabla H + g u$$

  从 energy 角度看:
  $$\dot{H} = \nabla H^T \dot{z} = \nabla H^T (J - R) \nabla H = -\nabla H^T R \nabla H \leq 0$$
  
  所以 $R$ 项让 energy 单调 decrease,符合 friction/dissipation 的物理。Beautiful。

**Term 3: $G_\psi(z_t) a_t$** — Control input
- $G_\psi(z_t)$:control matrix,描述 action 如何注入 system
- $a_t$:action vector,e.g., joint torques, end-effector force
- 在 robotics 中,$G$ 通常取决于 robot configuration,所以是 state-dependent

**Term 4: $\epsilon_\psi(z_t, a_t)$** — Residual
- Captures contact discontinuities, unmodeled dynamics, perception noise
- Paper 用 $\mathcal{L}_{res} = \|\epsilon_\psi\|_2^2$ (Equation 20) regularize,保证 Hamiltonian 是 dominant explanation 而不是 degenerate 到 generic transition network

**Numerical integration**: 得到 $\dot{z}$ 之后要积分成 $z_{t+1}$。Paper 提到 symplectic integration,推荐 leapfrog:
$$\begin{aligned}
q_{t+\Delta/2} &= q_t + \frac{\Delta}{2} \frac{\partial H}{\partial p}(q_t, p_t) \\
p_{t+1} &= p_t - \Delta \frac{\partial H}{\partial q}(q_{t+\Delta/2}, p_t) \\
q_{t+1} &= q_{t+\Delta/2} + \frac{\Delta}{2} \frac{\partial H}{\partial p}(q_{t+\Delta/2}, p_{t+1})
\end{aligned}$$

Symplectic Euler 是更简单 variant:
$$q_{t+1} = q_t + \Delta \frac{\partial H}{\partial p}(q_t, p_{t+1}), \quad p_{t+1} = p_t - \Delta \frac{\partial H}{\partial q}(q_t, p_t)$$

**References**:
- Port-Hamiltonian systems (van der Schaft): https://link.springer.com/book/10.1007/978-1-4471-0532-1
- Symplectic ODE-Net (Zhong et al.): https://arxiv.org/abs/1909.12077
- Hamiltonian Neural Networks (Greydanus): https://arxiv.org/abs/1909.10690

#### (C) Video Generation / Decoder $D_\theta$

Paper Equation (21):
$$p_\theta(o_{t+1:t+H} \mid o_{\leq t}, a_{t:t+H-1}, \hat{z}_{t+1:t+H})$$

Decoder 可以是:
- **Deterministic**: 一个 CNN decoder 把 $\hat{z}_{t+1:t+H}$ render 成 frames
- **Diffusion-based**: 以 $\hat{z}$ 为 condition 的 latent video diffusion (类似 Sora, Genie 的 architecture)
- **Hybrid**: use $\hat{z}$ 来 modulate spatial features in a pretrained video diffusion model

为什么 paper 要保留 video generation 这一步？因为 phase space $(q, p)$ 是 low-dimensional abstraction,无法直接 recover 高维 visual detail。Phase space 给 physics structure,video decoder 给 perceptual richness。这正好对应 paper Section 6 结尾说的 "physical evolution and visual rendering should be separated but coupled"。

**实现方式参考**:
- Latent Diffusion (Rombach et al.): https://arxiv.org/abs/2112.10752
- VideoLDM (Blattmann et al.): https://arxiv.org/abs/2304.08818
- 可以把 $\hat{z}$ 作为 cross-attention condition 注入到 diffusion U-Net 的某些 layer

#### (D) Planning / Decision Interface $U$

Paper Equation (24):
$$m^* = \arg\max_{m \in \{1,\ldots,M\}} U(\hat{z}^{(m)}_{t+1:t+H}, \hat{o}^{(m)}_{t+1:t+H})$$

具体而言,从 candidate actions $\{a^{(m)}_{t:t+H-1}\}_{m=1}^M$ 出发,每个 candidate roll out 一条 trajectory,然后用 utility function $U$ 评估并选最优。

$U$ 可以是:
- Reward function: $U = \sum_t r(z_t, a_t)$
- Goal-reaching: $U = -\|z_T - z_{goal}\|^2$
- Constraint satisfaction: $U = -\sum_t \text{violations}_t$

这是 standard MPC framework (Camacho & Bordons): https://link.springer.com/book/10.1007/978-0-85729-398-5

### 3.3 Loss Function 应该长什么样

Paper 没有明确给出 total loss,但可以推测是:

$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{phase} + \lambda_2 \mathcal{L}_{energy} + \lambda_3 \mathcal{L}_{res}$$

其中:
- $\mathcal{L}_{recon} = -\log p_\theta(o_{t+1:t+H} \mid \hat{z}_{t+1:t+H})$:reconstruction
- $\mathcal{L}_{phase} = \|z_t - \hat{z}_t\|^2$ 或 contrastive:对比预测的 phase 与 encoder 重新 encoding 的 phase
- $\mathcal{L}_{energy} = |H(\phi(\text{rollout})) - H(\text{encoder}(o_{t:t+H}))|$:energy consistency
- $\mathcal{L}_{res} = \|\epsilon_\psi\|_2^2$:paper Equation 20,regularize residual

---

## 4. 与相关工作的 deep connection

### 4.1 HNN → HGN → HWM 的 evolution

**HNN (Greydanus et al. 2019)** https://arxiv.org/abs/1909.10690:
- 学 $H_\theta(q, p)$,然后通过 autodiff 得到 $\dot{q}, \dot{p}$
- Loss: $\|\hat{\dot{z}} - \dot{z}\|^2$
- 在 simple systems (spring, pendulum, two-body) 上 demonstrate energy conservation 比 generic MLP 好得多

**HGN (Toth et al. 2020)** https://arxiv.org/abs/1909.13789:
- 加入 encoder/decoder,从 pixels 学 phase space
- 用 VAE-like encoder,$\mu_q, \mu_p$ 输出
- Demonstrate 能从 pixels reconstruct pendulum 等系统

**SymODE-Net (Zhong et al. 2020)** https://arxiv.org/abs/1909.12077:
- 加入 control input:$\dot{z} = J\nabla H + G a$
- 加入 dissipation: 用 positive semi-definite parameterization
- 在 controlled pendulum, cartpole 上 demonstrate

**HWM (本文)**:
- 把以上 + object-centric + video decoder + planning 全部整合
- Lift 到 embodied intelligence 的 level

### 4.2 与 Dreamer 系列对比

Dreamer (Hafner et al.) https://arxiv.org/abs/2301.04104 (DreamerV3):
- Latent:RSSM (Recurrent State-Space Model),stochastic latent
- Transition:RNN + MLP,无 explicit physical structure
- Actor-critic in imagination

HWM 与 Dreamer 的核心差异:
1. **Deterministic vs Stochastic**: HWM 是 deterministic Hamiltonian flow,Dreamer 是 stochastic recurrent。但 HWM 可以加 $\epsilon_\psi$ noise,或者用 stochastic Hamiltonian (e.g., Langevin dynamics)
2. **Structure vs Black-box**: HWM 有 strong inductive bias,Dreamer 没有
3. **Interpretability**: HWM 的 phase space 可以直接 visualize,Dreamer 的 latent 不可解释
4. **Composability**: HWM 的 object-centric structure 让 entity-level reasoning 变得可能

### 4.3 与 Sora 类 video model 对比

Sora (OpenAI, 2024): https://openai.com/sora/

Sora 是 video diffusion,没有 explicit action conditioning,没有 latent dynamics structure。Paper 在 Section 1 批评的 "generating future video is not equivalent to understanding the physical world" 正是 Sora 类模型的 fundamental limit。Sora 可以 generate visual plausible 5 秒视频但无法 predict 物理结果:

- 撞击后的物体 momentum 不守恒
- Long roll out 后 object permanence 失效
- 没法 condition on action

HWM 的 ambition 是给 video generation 加上一个物理 backbone,让 generative quality 和 physical validity 同时满足。

### 4.4 与 Neuro-Symbolic / Differentiable Physics 对比

- Differentiable physics (Brakel et al., de Avila Belbute-Peres et al.): https://arxiv.org/abs/2007.02198
  把 explicit physics simulator 嵌入到 NN 训练里。
- Hybrid: NN 学 residual,physics simulator 提供 backbone

HWM 用 *learned* Hamiltonian $H_\phi$,所以是更 soft 的 approach,不需要 hard-code physical laws。这是 trade-off: 更 general 但更难 constrain。

---

## 5. 深入讨论 Paper 提到的 Challenges

### 5.1 Real-world robotics 的 hard parts

**Friction & contact discontinuities**: Coulomb friction 是 non-smooth 的,Hamiltonian flow 是 smooth ODE。Paper 用 $R_\psi$ + $\epsilon_\psi$ 处理,但 contact event (e.g., 一物体刚撞上另一物体) 是 hybrid dynamics,需要 event detection 或者 complementarity formulation。

可能的 extension:
- Hybrid Hamiltonian system with mode switching
- Contact-implicit time stepping (Anitescu, Stewart)
- Learned contact predictor 作为 $R$ 的 state-dependent parameterization

**Deformable objects**: 刚体 Hamiltonian 是 well-developed (Celtic / Euler angles / quaternions),但 soft body 需要 infinite-dimensional phase space。可能需要:
- Modal reduction: 把 soft body 投影到低维 mode space (e.g., first N eigenmodes of elasticity operator)
- Hamiltonian reduced-order models (Lall, Marsden)

**Partial observability**: Pixels 是 high-dim 但 information-poor。Phase space $(q,p)$ 从 pixels 推出来是 inverse problem。HGN 在 simple systems 上 work,但在 real robotics (e.g., occluded objects, multi-object scenes) 上仍未被证明。

### 5.2 Learning phase space from pixels 的 difficulty

即使 latent 命名为 $(q, p)$,network 实际学到的可能是 entangled representation。可能需要:
- **Equivariant architecture**: 把 SE(3) 等价性 hard-coded 到 encoder 里,这样 latent 自动有 geometric meaning
  - Reference: Tensor Field Networks https://arxiv.org/abs/1802.08219
  - e3nn library: https://e3nn.org/
- **Identifiability**: 用 identifiability theory (Khemakhem et al.) 确保 latent 与 true $(q, p)$ 有 identifiable mapping
- **Symmetry constraint**: 用 canonicalization 或者 invariance loss

### 5.3 Compositional generalization

Paper 提到的 future direction "open-ended physical generalization" 需要:
- Object-centric 是 step 1,但还要 module compositionality
- Graph-based Hamiltonian: 每个 entity 是 node,interaction 是 edge,这样 unseen composition (新物体组合) 可以 generalize
- 类似 NRI (Neural Relational Inference) https://arxiv.org/abs/1802.04687

### 5.4 Evaluation 怎么做

Paper Section 7 提到 "current evaluation protocols still provide limited evidence"。这确实是 open problem。可能的方向:

- **Physics consistency metric**: rollout 后 measure $\Delta H / H$ (energy drift), linear momentum conservation, angular momentum conservation
- **Counterfactual accuracy**: 给定 action $a$ vs $a'$,measure model 是否给出 differentiating prediction
- **Transfer benchmark**: train on pendulum,test on double pendulum
- **Long-horizon stability**: 1000-step rollout 的 phase space volume preservation

---

## 6. 实验 / 架构推演 (paper 没有 experiments,我加 speculation)

虽然 paper 是 position paper 没有 experiments,让我推测可能的 benchmark:

### 6.1 可能的 toy experiments

**Benchmark 1: Pendulum / Cartpole**
- Baseline: Dreamer, HNN, generic MLP
- Metric: rollout MSE, energy drift $\|H(z_T) - H(z_0)\|/H(z_0)$, data efficiency (samples to reach threshold reward)
- 预期: HWM 在 energy drift 和 long-horizon 上明显胜出

**Benchmark 2: Multi-body gravitational system**
- N=3, 4, 5 bodies
- Train on N=3,test on N=4,5 (compositional generalization)
- 预期: object-centric HWM generalize,monolithic fails

**Benchmark 3: Pushing task (robotics)**
- 7-DoF arm pushing various objects
- Baseline: visual foresight, Dreamer
- 预期: HWM 在 contact-rich 部分需要 $R + \epsilon$,但仍是 best

### 6.2 可能的 ablation table (hypothetical)

| Model | Phase Space | Dissipation | Control | Residual | MSE (10-step) | Energy Drift |
|---|---|---|---|---|---|---|
| Pure MLP | ✗ | - | ✗ | - | 0.142 | 78% |
| HNN | ✓ | ✗ | ✗ | ✗ | 0.089 | 1.2% |
| SymODE | ✓ | ✓ | ✓ | ✗ | 0.073 | 4.5% (controlled) |
| HWM (full) | ✓ | ✓ | ✓ | ✓ | 0.058 | 5.1% |
| HWM no-res | ✓ | ✓ | ✓ | ✗ | 0.067 | 3.2% |

(这些数字是推测,但根据 HNN paper 和 SymODE paper 的 trend)

---

## 7. 与 Broader Trend 的 connection

### 7.1 Foundation Model + Physics 的 trending direction

- **Differentiable simulation + NN**: taichi-jax, DiffTaichi https://github.com/yuanming-hu/difftaichi
- **Physics-informed NN**: Raissi et al. https://arxiv.org/abs/1711.10561
- **Symmetry-aware representation**: ESCNN, e3nn
- **World model foundation**: 这是 LeCun一直在 push 的 JEPA 路线 vs DeepMind 的 Genie 路线 vs 本文 Hamiltonian 路线

### 7.2 Universal Simulator 的 vision

最终愿景是 "physically native foundation model": 一个能 simulate 任意 physical system 的 model。要求:
- Cross-domain generalization
- Compositional reasoning
- Long-horizon stability
- Interpretable latent

Hamiltonian + object-centric + video generation 是一条 promising 路径,但绝不是 only path。其他 possibilities:
- **Diffusion-based physics**: 把 physics rollout 做成 diffusion (e.g., BPD - Brownian Dynamics Diffusion)
- **Quantum-inspired**: 用 Hilbert space formulation
- **Neural operator**: PDE solver as NN (Li et al. Fourier Neural Operator) https://arxiv.org/abs/2010.08895

---

## 8. 我的 critical thoughts

### 8.1 优点

1. **Strong physics prior**: 对 conservative 部分确实有 strong inductive bias,理论上 long-horizon 稳定性比 generic NN 强
2. **Interpretability**: phase space 有物理 meaning,可以 debug
3. **Composition**: object-centric 是 right direction
4. **Unifying view**: 把 perception / dynamics / generation / planning 分离 是 clean abstraction

### 8.2 潜在问题

1. **Generality concern**: 真实世界很多 dynamics 不是 Port-Hamiltonian 可以 capture 的,e.g., thermal, chemical, biological。但 paper 也承认这点,只 claim 是 inductive bias 不是 literal description

2. **Scale concern**: HNN/HGN 在 simple systems 上 work,但 scale 到 100+ objects, deformable bodies, vision-rich scenes 是否还 tractable? Paper 没有给出 scaling evidence

3. **Control + dissipation 的 expressiveness**: $R \succeq 0$ + $G a$ 的形式对 robotics 够吗?Lagrangian with non-conservative forces 是更 general 的 formulation

4. **Latent identifiability**: 学到的 $(q, p)$ 是否真的对应物理 $(q, p)$?这是 fundamental 问题,paper 没解决。需要 identifiability theory 或 contrastive supervision

5. **Video generation 与 Hamiltonian 的 mismatch**: Video decoder 给 visual detail,但 phase space 是抽象的。如果 decoder 在很多 step 之后失去 grounding,会发生 "rendering drift" 而 phase space 看起来还正常,导致 visual inconsistency

### 8.3 想看的 future work

1. **Large-scale 实证**: 在 CALVIN, LIBERO, Real Robot 上 benchmark against DreamerV3, GR-1
2. **Identifiability analysis**: 给出 identifiability 条件
3. **Hybrid formulation**: 把 contact / event-driven 整合进 Hamiltonian framework
4. **Foundation model scaling**: object-centric phase space 是否能 scale to 1000+ objects
5. **Cross-embodiment**: 一个 HWM 能否 transfer from robot A 到 robot B,只通过改 $G_\psi$
6. **Multi-modal Hamiltonian**: vision + language + action 都 grounded 在同一 phase space

---

## 9. Key Takeaways

1. **Paper 的 core claim**: world model 应该有 physical structure,Hamiltonian 是一个 principled choice
2. **Core mechanism**: $\dot{z} = (J - R)\nabla H + G a + \epsilon$,把 conservative / dissipative / control / residual 分离
3. **Key insight**: 学 energy function $H$ 而不是 vector field $f$,得到 implicit constraints 和 generalization
4. **Architecture**: object-centric phase space encoder + Hamiltonian transition + video decoder + MPC planning
5. **主要 reference 工作**: HNN (Greydanus), HGN (Toth), SymODE-Net (Zhong), Hamiltonian Generative Networks

---

## 10. Useful Links

**Hamiltonian Neural Networks family**:
- HNN: https://arxiv.org/abs/1909.10690
- HGN: https://arxiv.org/abs/1909.13789
- SymODE-Net: https://arxiv.org/abs/1909.12077
- Lagrangian NN: https://arxiv.org/abs/2003.04630

**World Model baselines**:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- DreamerV3: https://arxiv.org/abs/2301.04104
- GAIA-1: https://arxiv.org/abs/2309.17080
- Genie: https://arxiv.org/abs/2402.15391
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08058

**Object-centric & relational**:
- Interaction Networks: https://arxiv.org/abs/1612.00222
- Graph Network-based Simulators (GNS): https://arxiv.org/abs/2002.09405
- Slot Attention: https://arxiv.org/abs/2006.13255

**Physics-informed ML**:
- PINN: https://arxiv.org/abs/1711.10561
- Neural ODE: https://arxiv.org/abs/1806.07366
- Differentiable Physics (Brakel): https://arxiv.org/abs/2007.02198

**Mathematical background**:
- Arnold, Mathematical Methods of Classical Mechanics: https://link.springer.com/book/10.1007/978-1-4757-2063-1
- Hairer et al. Geometric Numerical Integration: https://link.springer.com/book/10.1007/3-540-30666-8
- Port-Hamiltonian (van der Schaft): https://link.springer.com/book/10.1007/978-1-4471-0532-1

---

总结一句: 这篇 paper 是一个 **position paper**,提出 physically grounded world model 的 vision,把 Hamiltonian mechanics 作为 latent dynamics 的 backbone。它没有 experiments,但给出了清晰的 architectural framework,把 perception (encoder) → structured latent $(q,p)$ → Hamiltonian-inspired transition → video decoder → planning 这条 pipeline 串起来。如果后续能在 robotics benchmark 上 demonstrate 优势,这可能成为 next-gen world model 的一个重要 axis。但 proof 还在 wait,从 HNN 到 embodied robotics 的 scale gap 巨大,需要在 representation learning、scaling、contact handling 等多个 frontier 都有突破才能 realize paper 描述的 vision。
