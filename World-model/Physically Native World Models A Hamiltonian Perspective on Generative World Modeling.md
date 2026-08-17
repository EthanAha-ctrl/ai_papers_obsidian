---
source_pdf: Physically Native World Models A Hamiltonian Perspective on Generative
  World Modeling.pdf
paper_sha256: c6f9236b2709a7a214834adb128560bc155a1eace3084698ee611038588b179a
processed_at: '2026-08-06T03:29:49-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 一句话总结

现在所有world model都在学"视频怎么连续",但这篇paper说应该学"能量怎么流动"。

---

## 现在的world model到底什么问题

想象你训练一个video model,输入"机器人推杯子"的视频,让它预测未来。model学到的是什么?

它学到的是**像素层面的统计规律**:这个位置的像素下一帧会移到那个位置,这个颜色会渐变成那个颜色。

这就像你看了无数个台球比赛视频,学会了"白球撞红球,红球会滚走",但你完全不知道**为什么**会滚走。你不知道动量守恒,不知道能量转换,不知道摩擦力让球减速。

后果是什么?

**短期预测很准,长期预测崩盘**。因为你是靠记忆和模式匹配预测,不是靠理解物理规律。推10帧没问题,推100帧杯子可能飞到天上,或者穿透桌子,或者凭空消失。

**换个场景就废**。你见过铁杯子,换成木杯子就不知道怎么推了。因为你不理解mass影响dynamics,你只是记住了铁杯子的pixel pattern。

**action控制力弱**。你生成视频看起来很流畅,但你说"轻轻推"和"用力推"生成的future差不多,因为model对action的conditioning只是加了个condition vector,没有真正的因果机制。

---

## Hamiltonian mechanics到底在说什么

先把Hamiltonian mechanics讲明白,这是理解整篇paper的关键。

### 物理世界的state其实有两个部分

你描述一个物体的状态,需要两样东西:
- **它在哪**:position $q$(generalized coordinate)
- **它怎么动**:momentum $p$(generalized momentum,可以粗略理解为mass × velocity)

这两个拼在一起叫**phase space** $[q, p]$。

为什么需要两个?你只知道物体在哪,不知道它往哪走,没法预测未来。你知道杯子在桌上,但不知道它静止还是滑动,未来完全不同。

现有world model的latent vector把所有信息揉成一坨,你不知道哪些维度是"位置",哪些是"动量",哪些是"颜色",哪些是"语义"。这就叫**entangled representation**,对物理推理是灾难。

### 能量决定一切

Hamiltonian的核心insight:整个系统的dynamics由一个**标量energy function** $H(q, p)$ 决定。

不是直接学"state怎么变",而是学"能量长什么样",然后通过公式自动算出state怎么变:
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

翻译成人话:
- **位置的变化率** = 能量对动量的梯度(动量越大,跑得越快)
- **动量的变化率** = 能量对位置的负梯度(势能梯度产生力,力改变动量)

这就是Newton第二定律的另一种写法,但更优雅,更general。

### 为什么用energy function比直接学vector field好

这是整篇paper最核心的insuition,必须讲透。

**维度压缩**:假设系统有$n$个自由度,phase space是$2n$维。直接学dynamics要学一个$2n \to 2n$的mapping,即vector field,有$2n$个output。学energy function只需学$2n \to 1$的scalar field,只有1个output。hypothesis space指数级缩小。

**强约束**:energy function不是任意的,它必须满足物理规律。比如kinetic energy总是正的,potential energy有特定structure。这些constraint让model不会学到non-physical的dynamics。

**泛化**:学到一个energy landscape,它对**所有initial condition**都有效。你学到了spring的energy $H = p^2/2m + kq^2/2$,不管spring从哪开始,energy function都准确。但如果你学的是trajectory $q(t)$,换个initial condition就得重学。

**长horizon稳定**:Hamiltonian system有个数学性质叫**symplectic**,phase space volume在演化中守恒,能量也守恒。这意味着数值积分不会让能量drift,rollout 1000步energy还在那。普通NN transition rollout 100步就可能blow up。

---

## 这篇paper的核心架构

整个pipeline四步:

### Step 1: Encode — 从像素到phase space

$$o_t \xrightarrow{E_\theta} z_t = \{(q_t^i, p_t^i)\}_{i=1}^N$$

observation(图像/视频)通过encoder,输出N个entity的phase variable。

每个entity有自己的一对$(q^i, p^i)$,代表第$i$个物体(杯子、桌子、机器人手臂、工具等)的位置和动量。

这叫**object-centric representation**,把scene拆成object而非压成一个global vector。因为物理世界是compositional的,object之间通过contact, force相互作用,压成一坨就丢失了这种structure。

### Step 2: Evolve — Hamiltonian rollout

核心:学一个energy function $H_\phi(q^1, \dots, q^N, p^1, \dots, p^N)$,分解成三部分:

$$H_\phi = \sum_i K_\phi(p^i) + \sum_i U_\phi(q^i) + \sum_{i<j} V_\phi(q^i, q^j)$$

- $K_\phi$: 每个object自己的kinetic energy(和motion有关)
- $U_\phi$: 每个object自己的potential energy(和position有关,如gravity)
- $V_\phi$: object之间的interaction energy(如contact, spring, gravitational attraction)

然后dynamics自动从energy function推导:
$$\dot{z} = J \nabla_z H_\phi(z), \quad J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}$$

这个$J$矩阵叫**symplectic matrix**,它的作用是把gradient $\nabla H$旋转90度,变成dynamics。$J$是固定的,不学习。model只学$H_\phi$。

### Step 3: 但真实世界有摩擦和控制

纯Hamiltonian只描述conservative system(无摩擦、无外力),但robotics世界有:
- **Friction/damping**:energy会耗散
- **Control input**:robot主动施加力
- **Contact discontinuity**:碰撞是瞬间事件,非smooth
- **Deformable object**:soft body无法用rigid body Hamiltonian描述

所以扩展成controlled-dissipative form:

$$\dot{z} = (J - R_\psi(z)) \nabla_z H_\phi(z) + G_\psi(z) a_t + \epsilon_\psi(z, a_t)$$

逐项解释:

**(J - R)∇H**: Hamiltonian核心 + 耗散
- $J \nabla H$: 保守部分,energy守恒
- $-R \nabla H$: 耗散部分,$R \succeq 0$保证energy只减不增
- 为什么$R$要半正定?因为耗散的物理意义就是energy单调下降。数学上:$dH/dt = \nabla H \cdot (J-R)\nabla H = -\nabla H \cdot R \nabla H \leq 0$

**$G a_t$**: 控制输入
- $a_t$是action(robot施加的力/力矩)
- $G$是control matrix,描述action如何映射到各entity的phase variable
- 典型case:$a_t$直接加到某些object的momentum上

**$\epsilon$**: Residual
- 处理无法被smooth Hamiltonian描述的现象:碰撞冲击、deformable变形、视觉因素
- 通过L2 regularization约束:$\mathcal{L}_{res} = \|\epsilon\|^2$
- 为什么约束?不约束的话model会把所有dynamics塞进residual,Hamiltonian part就废了,退化成普通NN transition

这个form叫**Port-Hamiltonian system**,是robotics里处理非保守系统的标准framework,有严格数学基础。

### Step 4: Decode — 从phase space到视频

$$\hat{o}_{t+1:t+H} \sim D_\theta(\hat{z}_{t+1:t+H})$$

predicted phase trajectory送进video generator(diffusion, latent video model等),render成未来视频。

**关键**:video generator的input是物理structured的phase trajectory,不是free-form latent。它只需要rendering,不需要学dynamics。

### Step 5: Plan — 用rollout选action

生成$M$个candidate action sequence,各自rollout,看哪个future的utility最高:
$$m^* = \arg\max_m U(\hat{z}^{(m)}, \hat{o}^{(m)}), \quad a_t^* = a_t^{(m^*}$$

这就是MPC的思路,但rollout是物理structured的而非black-box。

---

## 为什么这个方向重要

### 对比1: vs. 纯video diffusion

Video diffusion学到的是pixel distribution。它不知道"杯子掉到地上会碎"是因为impact force超过material strength,它只是见过类似的pixel pattern。

Hamiltonian World Model的latent variable直接是$(q, p)$,energy function有physical meaning,rollout有数学保证。video generator只负责rendering,不负责dynamics。**物理部分和视觉部分解耦**。

### 对比2: vs. Dreamer系列

Dreamer的RSSM用recurrent network做latent transition,$z_{t+1} = f_\theta(z_t, a_t)$,没有任何physical structure。latent dimension可以是500维,你不知道哪些是position,哪些是velocity,哪些是task-specific feature。

Hamiltonian World Model强制latent是phase variable,transition由energy function诱导。hypothesis space从"任意vector field"压缩到"symplectic vector field induced by scalar energy"。

### 对比3: vs. JEPA

JEPA predict in abstract latent space,避免pixel reconstruction。philosophy对,但latent没有physical grounding。你predict出来的latent representation,不知道对应position还是energy还是其他什么。

Hamiltonian World Model的latent有明确physical meaning:$(q, p)$就是phase variable,$H$就是energy。可解释,可诊断,可用于planning。

---

## 现实挑战

### 从pixel学$(q, p)$很难

你给一张机器人推杯子的图,怎么知道$q$在哪、$p$是多少?pixel到phase variable的inverse problem是ill-posed。

HGN(Toth 2020)用VAE从pixel学phase variable,在simple system(spring, pendulum)上work,但scale到multi-object scene还是open problem。可能需要slot attention, object discovery, 3D perception辅助。

### Contact和deformable

Hamiltonian flow是smooth ODE,但contact是discontinuous event(瞬间velocity反转),deformable是infinite-dimensional system(场而非particle)。

可能需要:
- **Hybrid system**:Hamiltonian flow + event function + reset map,处理collision
- **Field Port-Hamiltonian**:infinite-dimensional extension处理deformable
- **Graph neural network**:object间interaction用GNN建模

### Evaluation难

怎么证明model真的学到physical regularity而非task-specific correlation?

不能只用reconstruction error或task reward,需要:
- **Energy conservation check**:long-horizon rollout的energy drift
- **OOD generalization**:换mass, friction, geometry后performance
- **Counterfactual reasoning**:不同action下的future是否符合物理直觉

---

## 我的intuition总结

这篇paper的核心thesis可以用一个类比:

**现有world model像背菜谱**:你背了1000道菜的做法,能复现,但换个食材就懵了,因为你不知道为什么这个步骤那个步骤。

**Hamiltonian World Model像学化学**:你学了能量守恒、热力学定律,虽然复杂,但任何食材任何锅你都能推理出大概结果。

inductive bias从"memorize pattern"变成"learn physical law"。代价是architecture更复杂,从pixel学phase variable更难。收益是data efficiency, long-horizon stability, interpretability, OOD generalization。

### 为什么现在做这个

几个trend converge:
1. **Video generation成熟了**:diffusion model能生成realistic video,但暴露了"visual ≠ physical"的问题
2. **HNN/HGN/Symplectic ODE-Net发展**:physics-informed neural network在simple system上证明可行
3. **Embodied AI爆发**:robotics需要action-controllable, physically reliable的world model,不只好看
4. **Foundation model范式**:大家想找"physical world foundation model"的architecture,Hamiltonian是natural candidate

### 可能的演进路径

短期(1-2年):
- 在simple robotic scene(推物体、抓取)上validate,对比DreamerV3和video diffusion
- object discovery + phase variable learning的end-to-end pipeline

中期(3-5年):
- Scale到multi-contact, articulated object, simple deformable
- 结合3D perception(neural radiance field / 3DGS)做geometry-aware phase variable

长期(5-10年):
- Physical world foundation model:compose physical mechanism across unseen object, material, embodiment
- 可能与differentiable physics engine(MuJoCo, Brax)融合,neural + symbolic hybrid

---

## 关键references深入

- **HNN**(Greydanus 2019):Hamiltonian World Model的直接ancestor,首次用NN parameterize $H$并通过autodiff算dynamics。https://arxiv.org/abs/1906.01563
- **HGN**(Toth 2020):HNN + VAE,从pixel学phase variable,证明feasible。https://arxiv.org/abs/1909.12077
- **Symplectic ODE-Net**(Zhong 2020):引入control input到Hamiltonian framework,本paper的eq.(19)直接extend这个line。https://arxiv.org/abs/1909.12077
- **Port-Hamiltonian**(van der Schaft):eq.(19)的$(J-R)$form的数学基础,robotics经典framework。
- **DreamerV3**(Hafner 2023):latent dynamics model SOTA,本paper的main critique target之一。https://arxiv.org/abs/2301.04104
- **GAIA-1**(Hu 2023):autonomous driving video world model,visual强physical弱。https://arxiv.org/abs/2309.17080
- **V-JEPA**(Bardes 2024):video的JEPA,latent prediction但无physical grounding。https://arxiv.org/abs/2404.08471

---

## 最后的联想

这篇paper让我想到几个broader的方向:

**Symplectic diffusion**:最近有些工作把score-based diffusion的backward SDE写成dissipative Hamiltonian form,可能unify generative model和physical simulator。本paper的framework天然fit这个方向。

**Differentiable physics + neural**:Brax, MuJoCo MJX这些differentiable physics engine和neural world model的融合。Hamiltonian World Model可以看作"learned differentiable physics engine with neural energy function"。

**World model as foundation model**:如果Hamiltonian World Model scale up成功,可能成为robotics的"physics foundation model",下游task只需fine-tune energy function的某些component,而非从头学dynamics。

**Connection to LeCun's JEPA**:LeCun强调predict in abstract space,Hamiltonian World Model给了一个physical grounding的"abstract space"——phase space。可能是JEPA philosophy的physical realization。

**Energy-based model(EBM)的revival**:Hamiltonian World Model本质是energy-based model的special case,phase space + symplectic structure。EBM在generative modeling里被diffusion overshadow,但在physical modeling可能revive。

---

简单总结:这篇paper不是engineering breakthrough,是**perspective shift**。它说stop treating world model as video predictor with action conditioning,start treating it as physical simulator with video rendering。Hamiltonian mechanics是已知的best mathematical framework for physical evolution,把它作为inductive bias嵌入generative world modeling是natural且principled的选择。long-term可能比scale up video diffusion更fundamental。

---

# Physically Native World Models: Hamiltonian视角下的生成式World Modeling

## 1. Paper的核心论点与motivation

这篇paper由清华的Sen Cui等人撰写,试图回答一个根本问题:**当前world model的瓶颈已经从"能否生成真实未来"转向"生成的未来是否physical meaningful且对action有用"**。

作者认为现有world model研究沿三条部分分离的路线发展:
- **2D video-generative models**(GAIA-1, Genie, Video Diffusion)强调视觉未来合成
- **3D scene-centric models**(NeRF, 3D Gaussian Splatting)强调空间重建
- **JEPA-like latent models**(I-JEPA, V-JEPA)强调抽象predictive representation

每条路线都有进展,但对embodied decision making而言,它们无法提供物理可靠、action可控、长horizon稳定的预测。作者提出用**Hamiltonian mechanics**作为physically grounded backbone来重新思考world modeling。

---

## 2. 现有World Model的根本limitation

### 2.1 Visual fidelity ≠ Physical validity

现代video generative model能合成视觉上plausible的sequence,但visual plausibility无法保证physical validity。一个生成的trajectory可能短期看起来realistic,但逐渐违反:
- **Object permanence**(物体永久性)
- **Contact consistency**(接触一致性)
- **Momentum conservation**(动量守恒)
- **Action-effect causality**(action-effect因果性)

对entertainment video synthesis,perceptual realism够用;对robotics/autonomous systems,预测future必须dynamically feasible, controllable, reliable。

### 2.2 弱结构的latent transition

典型latent formulation:
$$z_{t+1} = f_\theta(z_t, a_t) \tag{1}$$

其中$z_t$是learned latent state,$a_t$是action。这个formulation兼容RNN, diffusion video predictor, Transformer, structured SSM,但**没有specify latent state代表什么physical quantity,也没有specify transition应该preserve什么structure**。结果:模型学到pixel/latent space的short-term correlation,没有acquire稳定的physical dynamics,导致compounding errors, long-horizon drift, OOD generalization差, data requirements高。

### 2.3 Long-horizon rollout的compounding errors

World model的价值在于multi-step rollout用于planning。但learned dynamics model的compounding prediction errors很严重:小的one-step inaccuracy累积产生不可靠的long-horizon trajectories[Talvitie 2017; Janner 2019]。planning时query OOD state/action会使问题更严重。

### 2.4 Action conditioning弱

许多模型生成的future对action的conditioning弱。对contact-rich manipulation,force, momentum, contact geometry的small difference决定task success。如果model生成visually coherent future但对action conditioning弱,或者无法区分feasible/infeasible action outcome,对planning不可靠。

---

## 3. Hamiltonian Mechanics基础 — build intuition

### 3.1 Phase space作为natural latent state

Hamiltonian mechanics从structured state representation开始:
$$z_t = [q_t, p_t] \tag{10}$$

变量解释:
- $q_t \in \mathbb{R}^n$: **generalized coordinates**,描述系统configuration(如joint angles, object positions)
- $p_t \in \mathbb{R}^n$: **generalized momenta**,描述系统motion state($p = m\dot{q}$,或更一般的conjugate momentum)
- 二者构成**phase space** $\mathbb{R}^{2n}$

为什么phase space适合world model?因为agent不仅要reason about物体在哪,还要reason about物体如何移动、如何响应future action。unstructured latent vector无法区分"位置"和"动量",但phase space天然分离这conjugate variables。

### 3.2 Energy function诱导structured dynamics

给定Hamiltonian $H(q, p)$(标量energy function),evolution遵循**Hamilton's equations**:
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q} \tag{11}$$

含义:
- $\dot{q} = \partial H / \partial p$: 位置的变化率由能量对动量的梯度驱动(经典力学中$H = p^2/2m + U(q)$,则$\dot{q} = p/m$,即velocity = momentum/mass)
- $\dot{p} = -\partial H / \partial q$: 动量的变化率由能量的负梯度驱动(Newton第二定律:force = $-\nabla U$,$\dot{p} = F$)

紧凑form:
$$\dot{z} = J \nabla_z H(z), \quad J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix} \tag{12}$$

其中$J$是**symplectic matrix**,$I$是$n \times n$单位阵。$J$的核心作用:
- $J^T = -J$(反对称)
- $J^2 = -I$(旋转90度的几何意义)
- $\nabla H \cdot J \nabla H = 0$(保证能量守恒:$dH/dt = \nabla H \cdot \dot{z} = \nabla H \cdot J \nabla H = 0$)

**Key insight**: 模型不再直接fit $z_{t+1} = f_\theta(z_t, a_t)$这个black-box vector field,而是**学习一个energy landscape,通过其gradient生成dynamics**。这是HNN[Greydanus 2019]的核心创新。

### 3.3 为什么Hamiltonian structure支持long-horizon stability

Hamiltonian system有两个关键性质:

**1. Phase-space volume preservation**(Liouville定理):
$$\nabla_z \cdot \dot{z} = \nabla_z \cdot (J \nabla_z H) = \text{tr}(J \cdot \text{Hess}(H)) = 0$$

因为$J$反对称,Hessian对称,trace为零。phase space中的volume element在evolution下不变,这prevent了energy concentration或dissipation的artificial artifact。

**2. Energy conservation**(对conservative system):
$$\frac{dH}{dt} = \nabla H \cdot \dot{z} = \nabla H \cdot J \nabla H = 0$$

因为$J$反对称,$\nabla H \cdot J \nabla H = 0$。这让long-horizon rollout不会drift到non-physical高能量状态。

配合**symplectic integrator**(如Stormer-Verlet, implicit midpoint),数值积分也能preserve这些结构[Hairer 2006]。Neural approach如HNN, HGN[Toth 2020], Symplectic ODE-Net[Zhong 2020]证明可以incorporate这些structure-preserving principles到learned dynamics。

### 3.4 Data efficiency via physical priors

Hamiltonian dynamics把learning burden从"fit每个trajectory独立"转到"learn一个global energy landscape generalize across initial conditions"。Raissi的PINN[Battaglia 2016; Sanchez-Gonzalez 2020; Raissi 2019; Cranmer 2020]证明physical priors能显著提升data efficiency。

---

## 4. Hamiltonian World Model架构详解

### 4.1 总体pipeline

$$o_t \xrightarrow{E_\theta} z_t \xrightarrow{\mathcal{T}_H} \hat{z}_{t+1:t+H} \xrightarrow{D_\theta} \hat{o}_{t+1:t+H} \xrightarrow{U} a_t^* \tag{13}$$

四个组件:
- $E_\theta$: **Encoder**,从observation $o_t$映射到latent state $z_t$
- $\mathcal{T}_H$: **Hamiltonian transition**,在phase space演化
- $D_\theta$: **Decoder/generator**,渲染future observation
- $U$: **Utility function**,评估并选action

这个separation很重要:modern video generative model能synthesize realistic future[Ho 2022; Blattmann 2023; Hu 2023; Bruce 2024],但physical reliability主要依赖latent dynamics的structure。

### 4.2 Object-centric structured latent representation

scene表示为entity-level phase variables:
$$z_t = \{(q_t^i, p_t^i)\}_{i=1}^N \tag{14}$$

变量:
- $N$: entity数量
- $q_t^i \in \mathbb{R}^{d_i}$: 第$i$个entity的generalized coordinate
- $p_t^i \in \mathbb{R}^{d_i}$: 第$i$个entity的generalized momentum
- entity对应object, robot link, tool, surface, 或dynamically relevant region

这种object-centric representation由physical scene的compositional nature驱动:object通过contact, geometry, force, constraint相互作用。Interaction Network[Battaglia 2016], graph-based simulator[Sanchez-Gonzalez 2020], structured world model[Kipf 2020]证明relational representation提升physical domain的generalization。

### 4.3 Hamiltonian decomposition与Hamilton's equations

learned Hamiltonian function:
$$H_\phi(z_t) = H_\phi(q_t^1, \dots, q_t^N, p_t^1, \dots, p_t^N) \tag{15}$$

自然parameterization分解为kinetic, potential, interaction:
$$H_\phi(z_t) = \sum_i K_\phi(p_t^i) + \sum_i U_\phi(q_t^i) + \sum_{i<j} V_\phi(q_t^i, q_t^j) \tag{16}$$

各项含义:
- $K_\phi(p_t^i)$: **Kinetic energy**,第$i$个entity的motion energy(经典力学中$K = \|p\|^2/2m$,neural版可学习mass matrix)
- $U_\phi(q_t^i)$: **Potential energy**,第$i$个entity的self potential(gravity, spring, internal configuration energy)
- $V_\phi(q_t^i, q_t^j)$: **Interaction potential**,第$i$和第$j$个entity之间的相互作用能(gravitational, electrostatic, contact potential, spring-like等)

Dynamics由Hamilton's equations诱导:
$$\dot{q}_t^i = \frac{\partial H_\phi}{\partial p_t^i}, \quad \dot{p}_t^i = -\frac{\partial H_\phi}{\partial q_t^i} \tag{17}$$

紧凑form:
$$\dot{z}_t = J \nabla_z H_\phi(z_t), \quad J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix} \tag{18}$$

这里$\nabla_z H_\phi$通过autodiff计算,$J$是固定的symplectic structure matrix。模型学的是标量$H_\phi: \mathbb{R}^{2N d} \to \mathbb{R}$,通过$J$与gradient组合成$2Nd$维vector field。这把learning problem从"学一个$2Nd \to 2Nd$的vector field"降到"学一个$2Nd \to \mathbb{R}$的scalar field",大幅压缩hypothesis space。

### 4.4 Controlled-dissipative extension

真实世界不是closed conservative system。Robotics涉及control input, friction, damping, impact, contact switch, perception uncertainty。作者提出controlled-dissipative Hamiltonian form:

$$\dot{z}_t = \underbrace{(J - R_\psi(z_t))}_{\text{structured}} \nabla_z H_\phi(z_t) + \underbrace{G_\psi(z_t) a_t}_{\text{control}} + \underbrace{\epsilon_\psi(z_t, a_t)}_{\text{residual}} \tag{19}$$

各项详解:

**(1) Hamiltonian core: $J \nabla_z H_\phi(z_t)$**
- 保留conservative dynamics的物理backbone
- $J$是symplectic matrix,encode phase space的几何structure

**(2) Dissipation: $-R_\psi(z_t) \nabla_z H_\phi(z_t)$**
- $R_\psi(z_t) \succeq 0$: 半正定dissipation matrix,由网络$R_\psi$学习
- 物理含义:沿energy gradient方向"耗散"energy,对应friction, damping, drag等nonconservative force
- 为什么$R \succeq 0$?因为dissipation让energy单调不增:
$$\frac{dH}{dt} = \nabla H \cdot \dot{z} = \nabla H \cdot (J - R) \nabla H = \underbrace{\nabla H \cdot J \nabla H}_{=0} - \nabla H \cdot R \nabla H \leq 0$$
- 这是**Port-Hamiltonian system**的经典form(van der Schaft & Maschke),robotics里控制dissipative system的标准framework

**(3) Control: $G_\psi(z_t) a_t$**
- $a_t \in \mathbb{R}^m$: action vector(m维control input)
- $G_\psi(z_t) \in \mathbb{R}^{2Nd \times m}$: control matrix,描述action如何注入到各entity的phase variables
- 经典例子:$\dot{p} = F_{ext}$对应$G = [0; I]^T$,即action直接影响动量

**(4) Residual: $\epsilon_\psi(z_t, a_t)$**
- 捕获contact discontinuity, non-rigid deformation, unmodeled visual dynamics等难以smooth Hamiltonian flow描述的现象
- 通过regularization约束:
$$\mathcal{L}_{res} = \|\epsilon_\psi(z_t, a_t)\|_2^2 \tag{20}$$
- 确保Hamiltonian structure是dominant explanation,而非degenerate成generic transition network

这种设计与controlled Hamiltonian learning[Zhong 2020]和symplectic neural dynamics[Hairer 2006]一致。residual term应该保持constrained,使模型用Hamiltonian structure作为dominant explanation。

### 4.5 Video generative decoder

预测latent trajectory送给video generative decoder:
$$p_\theta(o_{t+1:t+H} \mid o_{\le t}, a_{t:t+H-1}, \hat{z}_{t+1:t+H}) \tag{21}$$

这把物理latent evolution与perceptual rendering解耦。decoder可以是diffusion model, latent video model,或任何conditional video generator,但**它的输入是物理structured的phase trajectory而非free-form latent**。

### 4.6 Planning interface

给定candidate action sequences $\{a_{t:t+H-1}^{(m)}\}_{m=1}^M$:

**Rollout latent trajectories:**
$$\hat{z}_{t+1:t+H}^{(m)} = \mathcal{T}_H(z_t, a_{t:t+H-1}^{(m)}) \tag{22}$$

**Render visual futures:**
$$\hat{o}_{t+1:t+H}^{(m)} \sim D_\theta(\hat{z}_{t+1:t+H}^{(m)}) \tag{23}$$

**Select optimal action:**
$$m^* = \arg\max_{m \in \{1, 2, \dots, M\}} U(\hat{z}_{t+1:t+H}^{(m)}, \hat{o}_{t+1:t+H}^{(m)}), \quad a_t^* = a_t^{m^*} \tag{24}$$

变量:
- $M$: candidate action sequences数量
- $U(\cdot)$: utility function,可包含task reward, safety constraints, energy cost等
- $m^*$: 最优action sequence的index
- $a_t^*$: 第一个时间步选定的action(MPC-style receding horizon)

这与visual foresight[Finn & Levine 2017; Ebert 2018], MPC[Camacho & Bordons 2013], latent imagination[Hafner 2020, 2023]对齐,区别在于imagined future基于structured phase-space rollout而非unconstrained video predictor。

---

## 5. 范式对比Table分析

| Paradigm | Representation | Strength | Limitation |
|----------|---------------|----------|------------|
| **2D video-generative**(Video Diffusion, VideoLDM, GAIA-1, Genie) | Image/video sequences | 强perceptual realism;灵活conditional future synthesis;直观observation-level prediction | 弱explicit physical structure;long-horizon drift;action consistency脆弱 |
| **3D scene-centric**(NeRF, 3DGS) | Geometry-aware 3D scene | 强spatial grounding, view consistency, explicit scene structure | 重建>动力学;action-conditioned physical evolution支持有限 |
| **JEPA-like**(I-JEPA, V-JEPA) | Predictive latent representation | 紧凑抽象预测;避免expensive pixel reconstruction;潜在scalable | latent variable与explicit physics弱tie;control/dynamics可解释性有限 |
| **Hamiltonian / physically native** | Structured latent phase space $(q, p)$ | 结构化physical inductive bias;improved interpretability;潜在better long-horizon stability和data efficiency | 难从pixel学习;真实world dynamics非fully conservative;contact-rich/deformable仍困难 |

Hamiltonian World Models试图combine structured latent prediction的advantage和physical inductive bias,通过phase space建模+energy-based transition提升stability, interpretability, data efficiency[Greydanus 2019; Toth 2020; Zhong 2020]。

---

## 6. 关键Related Work的延伸思考

### 6.1 Hamiltonian Neural Networks (HNN)[Greydanus 2019]

HNN是这paper的直接intellectual ancestor。核心idea:用网络parameterize $H_\theta(q, p)$,通过autodiff计算$\dot{q}, \dot{p}$,loss是predicted trajectory与ground truth的MSE。实验显示在spring, pendulum, two-body, Kepler等系统上比baseline NN有显著better generalization和long-horizon stability,尤其在energy conservation上。

但HNN局限:需要ground truth $(q, p)$,不能直接从pixel学习,且只处理conservative system。
Reference: https://arxiv.org/abs/1906.01563

### 6.2 Hamiltonian Generative Networks (HGN)[Toth 2020]

HNN升级版,从pixel学习phase variable。用VAE-style encoder把image encode成$(q, p)$,然后Hamiltonian decoder rollout,再reconstruct image。关键:用Variational Autoencoder发现canonical coordinate system。
Reference: https://arxiv.org/abs/1909.12077

HGN的limitation:latent dimension固定,只能处理simple system;multi-objective interaction处理弱;无法处理control input。

### 6.3 Symplectic ODE-Net[Zhong 2020]

把symplectic integrator与neural ODE结合,并引入control input。形式上最接近本paper的eq.(19)的controlled Hamiltonian部分。
Reference: https://arxiv.org/abs/1909.12077

### 6.4 Lagrangian Neural Networks[Cranmer 2020]

与Hamiltonian parallel的Lagrangian perspective:$L = T - V$,$\dot{q} = \partial L / \partial \dot{q}$。优势:不需要conjugate momentum,直接在$(q, \dot{q})$空间。劣势:loss of symplectic structure。
Reference: https://arxiv.org/abs/2003.04630

### 6.5 Port-Hamiltonian Systems

van der Schaft & Maschke的Port-Hamiltonian framework正是eq.(19)的理论基础:
$$\dot{x} = (J(x) - R(x)) \nabla H(x) + g(x)u + d(x)$$

其中$J$是interconnection matrix,$R \succeq 0$是dissipation,$g$是control port,$d$是external disturbance。这是robotics和非conservative physical system的标准modeling framework。本paper把Port-Hamiltonian idea引入generative world modeling是合理且elegant的。

### 6.6 Object-centric world models

Kipf的Contrastive Learning of Structured World Models[Kipf 2020]用object slot + relational inference。本paper的eq.(14)直接对应这种object-centric decomposition。
Reference: https://arxiv.org/abs/1910.01741

### 6.7 Dreamer系列[Hafner 2019, 2020, 2023]

PlaNet, Dreamer, DreamerV2, DreamerV3是latent dynamics model的代表。它们用RSSM(Recurrent State-Space Model)做latent transition。本paper的critique:Dreamer的latent transition是unconstrained neural network,没有phase space structure,没有energy conservation inductive bias。
Reference: https://arxiv.org/abs/2301.04104

### 6.8 GAIA-1[Hu 2023]与Genie[Bruce 2024]

GAIA-1: autonomous driving的generative world model,输入video+text+action,输出未来video。Transformer-based,大规模训练。
Reference: https://arxiv.org/abs/2309.17080

Genie:从large-scale Internet video学习action-controllable interactive environment,无需access到ground truth action。
Reference: https://arxiv.org/abs/2402.15391

二者都属于2D video-generative route,本paper认为它们visual fidelity强但physical structure弱。

### 6.9 I-JEPA与V-JEPA

I-JEPA[Assran 2023]:image self-supervised learning,predict latent representation of masked target blocks given context。
Reference: https://arxiv.org/abs/2301.08243

V-JEPA[Bardes 2024]:video extension,predict masked spatio-temporal patches的representation。
Reference: https://arxiv.org/abs/2404.08471

LeCun的JEPA philosophy是"predict in abstract representation space, not pixel space",这与本paper对pixel reconstruction的critique一致。但JEPA的latent没有physical grounding,这是Hamiltonian World Model要解决的。

---

## 7. 实验思考(本paper是position paper,无实验)

本paper是position/vision paper,没有实验数据。但如果要validate Hamiltonian World Model,合理的实验设计:

**Benchmark**: 
- **MuJoCo physical scenes**:multi-body dynamics, contact, friction
- **robomimic / robosuite**:manipulation tasks
- **ShapeNet physics**:rigid body + deformable
- **Atari**:低physical但高compositional

**Metrics**:
- **Long-horizon rollout error**:50-step, 100-step, 200-step的state prediction error
- **Energy drift**:predicted energy vs ground truth energy随horizon变化
- **Data efficiency**:fixed task performance vs training trajectory数量
- **Action controllability**:不同action下的predicted future divergence,与ground truth divergence的correlation
- **OOD generalization**:不同mass, friction, geometry下的performance
- **Planning success rate**:用world model做MPC的task success

**Baselines**:
- DreamerV3(latent dynamics)
- SORA-style video diffusion
- I-JEPA/V-JEPA
- Symplectic ODE-Net(物理baseline)
- HGN(pixel-conditioned物理baseline)

---

## 8. Challenges与Limitations

### 8.1 真实环境非ideal Hamiltonian

real embodied environment有friction, control input, impact, dissipation, partial observability, non-rigid interaction。Hamiltonian dynamics应作为structural prior而非literal description[Zhong 2020; Raissi 2019]。

### 8.2 Phase variable学习困难

即使latent state写成$(q, p)$,从pixel infer meaningful coordinate和momenta是nontrivial representation learning problem。HGN[Toth 2020]显示可行,但scale到复杂scene仍开放。可能需要Slot Attention, Object-centric encoder等辅助。

### 8.3 Contact-rich与deformable

contact-rich manipulation, articulated system, deformable object可能需要hybrid, graph-based, event-driven extension beyond smooth Hamiltonian flow。Hamiltonian + complementarity formulation可能是一个方向。

### 8.4 评估protocol不成熟

当前evaluation protocol对"是否学到reusable physical regularity"提供limited evidence。仅用reconstruction error或task reward难以区分"exploit task-specific correlation"与"learned transferable physical dynamics"。

---

## 9. Future Directions的延伸联想

### 9.1 Physical world foundation model

作者vision是"physical world foundation model",combine:
- 2D video generation的perceptual richness
- 3D scene representation的spatial grounding
- JEPA-like latent prediction的compact abstraction
- Hamiltonian phase-space dynamics的stability

### 9.2 Open-ended physical generalization

从scene-level physical rollout到open-ended physical generalization:学习reusable physical mechanism可compose across unseen object, material, embodiment, task。这类似"physics engine as foundation model"。

### 9.3 可能的延伸方向

**Symplectic Neural ODE with contact**:处理discontinuous contact event,用event function + reset map hybrid system。

**Neural Port-Hamiltonian for deformable**:用infinite-dimensional Port-Hamiltonian(field theory)处理deformable body, fluid。

**Energy-aware exploration**:用learned Hamiltonian做curiosity-driven exploration,鼓励visit high-energy或high-curvature region。

**Compositional Hamiltonian**:不同object/entity的Hamiltonian componentize compose,类似differentiable physics engine。

**Diffusion + Hamiltonian**:score-based diffusion model的backward SDE可以写成dissipative Hamiltonian form,可能unify generative model和physical simulator。

---

## 10. Intuition building总结

**核心intuition**:当前world model的limitation是它们学习"surface statistics"——pixel或latent的temporal correlation,而非"deep structure"——energy, momentum, constraint等物理量。Hamiltonian perspective强制model learning一个energy landscape,通过其gradient生成dynamics,这把hypothesis space从"任意vector field"压缩到"由scalar energy function诱导的symplectic vector field"。

**为什么这能改善long-horizon stability**:Hamiltonian flow preserve phase-space volume和能量,配合symplectic integrator,numerical rollout不accumulating energy drift。unconstrained NN transition在multi-step rollout后容易blow up或collapse到低能量manifold。

**为什么这能改善data efficiency**:学习scalar energy function $H: \mathbb{R}^{2n} \to \mathbb{R}$比学习vector field $f: \mathbb{R}^{2n} \to \mathbb{R}^{2n}$的parameter效率高得多。加上energy conservation是强constraint,把hypothesis space大幅压缩。

**为什么controlled-dissipative form重要**:真实world有control和dissipation,纯Hamiltonian不sufficient。Port-Hamiltonian的$(J - R)$structure保证既保留symplectic structure(conservative part),又allow能量耗散(dissipative part),还allow control port input。这是mathematically principled且physically faithful的extension。

**为什么residual term需要constrained**:如果$\epsilon_\psi$不受约束,model会退化成"learn generic transition, Hamiltonian part只是一个unused regularizer"。$\mathcal{L}_{res}$强制residual small,确保Hamiltonian structure是dominant explanation。但contact discontinuity等genuinely non-smooth现象需要residual有足够capacity,这是trade-off。

---

## Reference Links

- HNN: https://arxiv.org/abs/1906.01563
- HGN: https://arxiv.org/abs/1909.12077
- Symplectic ODE-Net: https://arxiv.org/abs/1909.12077
- Lagrangian NN: https://arxiv.org/abs/2003.04630
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- PlaNet: https://arxiv.org/abs/1810.02576
- Dreamer: https://arxiv.org/abs/1912.01603
- DreamerV2: https://arxiv.org/abs/2010.02193
- DreamerV3: https://arxiv.org/abs/2301.04104
- GAIA-1: https://arxiv.org/abs/2309.17080
- Genie: https://arxiv.org/abs/2402.15391
- Video Diffusion: https://arxiv.org/abs/2204.03458
- VideoLDM: https://arxiv.org/abs/2304.08818
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08471
- NeRF: https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- Interaction Network: https://arxiv.org/abs/1806.01877
- Learning to Simulate (GNS): https://arxiv.org/abs/2002.09405
- Contrastive Structured World Models (C-SWM): https://arxiv.org/abs/1910.01741
- PINN: https://www.sciencedirect.com/science/article/pii/S0021999118305527
- Port-Hamiltonian (van der Schaft): https://www.springer.com/gp/book/9781846287892
- Geometric Numerical Integration (Hairer): https://link.springer.com/book/10.1007/3-540-30666-8
- Visual Foresight: https://arxiv.org/abs/1812.00568
- Deep Visual Foresight: https://arxiv.org/abs/1610.06953
- MBPO: https://arxiv.org/abs/1906.08253
- Self-correcting models: https://arxiv.org/abs/1708.07496
- RT-2: https://arxiv.org/abs/2307.15818
- PaLM-E: https://arxiv.org/abs/2303.03378

---

这篇paper虽然position性质,但它精准地把world model研究推向"物理native"的方向。Hamiltonian perspective不只是loss term的添加,而是对latent state representation, transition mechanism, 和generative pipeline的系统性重新设计。从HNN到HGN到Symplectic ODE-Net再到本paper,可以看到physics-informed deep learning从simple system到complex scene的逐步演进。对于build embodied intelligence foundation model,这可能是比pure scale-up video diffusion更fundamental的path。
