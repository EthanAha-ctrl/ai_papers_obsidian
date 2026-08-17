---
source_pdf: Humanoid-Gym.pdf
paper_sha256: f5a95d5797dc0b788087fc454fef0923f4d5055a14d69645902c0fcbc96e097e
processed_at: '2026-08-05T08:03:23-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Humanoid-Gym

Andrej，我换一种讲法，像咱们coffee chat那样聊聊这篇paper。

## 一句话总结

这帮人搞了个开源工具包，让humanoid robot在simulation里学会走路，然后直接扔到real world里就能走，zero-shot，不用再fine-tune。听着简单，做起来要命。

---

## 为什么humanoid这么难搞

四条腿的狗子robot，RL已经玩得很溜了，ETH那帮人（Legged Gym）几分钟就能train出来一个能跑的policy。但humanoid两条腿，问题完全不在一个量级。

你可以这样想：四条腿的狗子像一个桌子，哪怕三条腿着地也很稳。两条腿的humanoid像一个倒立的pendulum，重心稍微偏一点，啪，摔了。而且humanoid有12个joint，dynamics耦合得要命，hip动一下 ankle就得跟着补偿，所有东西都是coupled的。

更恶心的是sim2real gap。Simulation里的robot是数学公式，perfect sensor、perfect motor、perfect contact。Real world里什么都有noise——IMU漂移、motor磨损、地面friction时变、通信delay。对四条腿来说，这些noise是"小风小雨"，robot robust一点就扛过去了。对两条腿来说，一个dust bunny就能让你摔倒。

所以Humanoid-Gym的核心mission就是：**在sim里把policy训练得足够robust，让它在real world的混乱中也能稳住。**

---

## 他们的整体思路

很简单三步：

**第一步**：在Isaac Gym里快速训练。Isaac Gym是Nvidia搞的GPU physics simulator，可以同时跑8192个robot environments，training快到飞起。问题是它的物理引擎（PhysX）用了很多approximation，contact model不够准，仿真和现实有gap。

**第二步**：把train好的policy扔到MuJoCo里验证。MuJoCo是另一个physics simulator，CPU跑的，慢，但contact model很准，业界公认的"较真"simulator。

**第三步**：如果MuJoCo里能跑通，再扔到real robot上。如果MuJoCo都跑不通，那别浪费real robot的时间了，回sim继续改。

整个pipeline的intuition是：**Isaac Gym负责"快"，MuJoCo负责"准"，real robot负责"验"**。这是一个非常工程化的workflow，不是science breakthrough，是craftsmanship。

---

## 训练里的几个关键设计

### 1. Asymmetric Actor Critic - 给policy开外挂

这是整个framework里最聪明的trick之一。

传统的Actor Critic，actor和critic看一样的信息。但训练humanoid的时候有个矛盾：
- 你想让policy学会从real sensor能拿到的信息里做决策（因为deploy时只有这些）
- 但你又想critic能给出准确的value估计，来指导actor学习
- Real sensor拿不到的东西（比如ground friction是多少、有没有external push force、robot的base linear velocity），sim里其实是已知的

**Asymmetric Actor Critic的解法**：训练时给critic开外挂，让它看到所有privileged info；actor只看real deploy时能拿到的info。Deploy的时候，critic扔掉，只留actor。

你可以这样想：critic像老师，actor像学生。老师有答案书（privileged info），能准确告诉学生"你这个action好不好"。学生只有自己的眼睛耳朵（observation），但通过老师的反馈，学生慢慢学会了从有限信息里infer出正确的action。最后考试（deploy）时老师不在，但学生已经训练出来了。

公式上就是这样：

Policy loss（actor学的）:
$$\mathcal{L}_\pi = \min\left[\frac{\pi(a_t|o_{\le t})}{\pi_b(a_t|o_{\le t})} A^{\pi_b}(o_{\le t}, a_t), \mathrm{clip}\left(\frac{\pi(a_t|o_{\le t})}{\pi_b(a_t|o_{\le t})}, 0.8, 1.2\right) A^{\pi_b}(o_{\le t}, a_t)\right]$$

- $\pi(a_t|o_{\le t})$: 当前policy在partial observation history下选action $a_t$ 的概率
- $\pi_b(a_t|o_{\le t})$: behavior policy（旧版）的概率，用来算ratio
- $A^{\pi_b}(o_{\le t}, a_t)$: advantage，由critic（看full state $s_t$）算出来
- clip到 $[0.8, 1.2]$ 是防止policy update步子迈太大

Value loss（critic学的）:
$$\mathcal{L}_v = \|R_t - V(s_t)\|_2$$

注意critic输入是 $s_t$（full state with privileged info），不是 $o_{\le t}$。这就是"asymmetric"的关键——actor和critic输入不一样。

Reference: Pinto et al. 2017, https://arxiv.org/abs/1710.06542

### 2. Clock Signal - 告诉policy现在是gait的哪一phase

这个设计看着简单，但很关键。

Humanoid walking本质是periodic motion——左脚抬、右脚撑、右脚抬、左脚撑，循环往复。RL policy如果完全从零学这个periodic pattern，非常难，因为exploration空间太大，大部分random action直接摔。

作者的trick是给policy inject一个clock signal：
$$[\sin(2\pi t / C_T), \cos(2\pi t / C_T)]$$

- $t$: 当前时间
- $C_T$: gait cycle time，一个完整周期的时间长度
- $\sin, \cos$: 让signal周期性循环，且smooth无discontinuity

Policy看到这个2D signal，就知道"哦，现在gait cycle走到这里了，该抬左脚了"。这相当于给policy一个"节奏感"，不用自己从头学periodicity。

Intuition：就像跳舞放节拍器。你不用自己数拍子，跟着节拍器走就行。Policy不用自己探索periodic pattern，跟着clock走就行。

为什么用sin和cos两个？因为单个sin在波峰波谷时gradient接近0，policy不知道时间往哪个方向走。加个cos，两个phase-shift 90度的signal组成一个2D rotating vector，任何时刻都有明确方向，就像clock的指针在2D平面上转。

### 3. Frame Stack 15 - 给policy记忆

Policy输入是单帧observation的15帧stack。这意味着policy每个决策时刻都能看到过去15帧的observation，相当于有150ms的历史记忆。

为什么需要历史？因为real world是POMDP（Partially Observable MDP），很多关键信息从单帧observation看不出来，只能从history推断。比如：
- Base linear velocity：IMU直接测不准，但可以从orientation变化history推算
- Ground friction：从foot slip的pattern推断
- External disturbance：从unexpected body motion推断

Single observation维度47，stack 15帧，total input维度705。给MLP够了。

Critic的privileged observation只stack 3帧，因为privileged info本来就是current state的"上帝视角"，不需要太多历史。

### 4. Stance Mask - 强制contact pattern

这个是reference motion prior的另一个injection。

作者定义了一个stance mask $I_p(t)$，告诉你每个时刻哪只脚应该在地面：
- $[1, 0]$: 左脚stance，右脚swing
- $[0, 1]$: 右脚stance，左脚swing  
- $[1, 1]$: 双脚stance（double support phase）

Reward里有一项：
$$r_{contact} = \phi(I_p(t) - I_d(t), \infty)$$

- $I_p(t)$: reference的stance mask（planned）
- $I_d(t)$: actual detected contact（从simulator拿）
- $w = \infty$ 意味着 $\phi(e, \infty) = \exp(-\infty \cdot \|e\|^2)$，error非0时reward直接是0

这是一个hard constraint：policy必须让contact match reference。如果不match，这一项reward直接清零。

Intuition：与其让policy完全自由探索contact pattern（很大概率探索出hopping、shuffling、单脚跳各种weird gait），不如直接告诉它"你必须左脚右脚交替着地"。剩下的自由度就是how to swing leg、how to balance，这才是control的真正难点。

---

## Action设计 - 为什么用PD target而不是torque

这是RL locomotion的一个经典设计选择。

**选项A**：Policy输出joint torque $\tau$，直接控制motor
**选项B**：Policy输出joint position target $\theta_{target}$，给PD controller，PD controller算torque

Humanoid-Gym用选项B。Policy输出12D的joint position target，PD controller在1000Hz频率下执行：
$$\tau_i = K_p^i (\theta_{target,i} - \theta_i) - K_d^i \dot{\theta}_i$$

- $\theta_{target,i}$: policy输出的target（action）
- $\theta_i, \dot{\theta}_i$: 当前joint的position和velocity
- $K_p^i, K_d^i$: PD gains（固定，不learned）

为什么这样设计？几个好处：

1. **Lower bound of stability**：PD controller本身是个model-based feedback controller，即使policy输出garbage，PD也会尽量让joint去target，不至于完全失控
2. **Policy只需要做high-level planning**：100Hz frequency下，policy思考"下一步把脚放哪里"，PD在1000Hz下保证"脚确实去到那里"
3. **Sample efficiency**：action space是position而不是torque，dimension一样但semantics更structured，RL学起来快很多

Intuition：就像开车，你不用直接控制引擎的fuel injection（torque），你控制方向盘和油门（target），车的ECU帮你处理底层细节。

---

## Reward Function - 全是craftsmanship

Reward是RL里最art的部分。作者把reward分成三大类：

### Tracking rewards（正reward，鼓励做对的事）

核心kernel：
$$\phi(e, w) = \exp(-w \cdot \|e\|^2)$$

- $e$: error（比如actual velocity - command velocity）
- $w$: sensitivity，越大越strict
- $\|e\|^2$: squared error
- 整个function是exponential decay，error小时reward接近1，error大时reward快速归0

为什么用这个而不是linear或quadratic？因为exponential decay的gradient在error小时小（避免overshoot），error大时也小（避免exploding gradient）。这是个numerically stable的shaping reward。

具体tracking项：
- Linear velocity tracking: $\phi(\dot{P}_{xyz}^b - \text{CMD}_{xyz}, 5)$，scale 1.2
- Angular velocity tracking: $\phi(\dot{P}_{\alpha\beta\gamma}^b - \text{CMD}_{\alpha\beta\gamma}, 5)$，scale 1.0
- Orientation tracking: $\phi(P_{\alpha\beta}^b, 5)$，scale 1.0，target是roll/pitch = 0（直立）
- Base height tracking: $\phi(P_z^b - 0.7, 100)$，scale 0.5，target height 0.7m，$w=100$ 非常strict

注意height的weight $w=100$比velocity的 $w=5$ 大20倍。Intuition：height变化对stability影响极大，robot一旦蹲下或踮脚就很难恢复，必须严格保持。

### Gait rewards（正reward，鼓励正确的gait pattern）

- Contact pattern: $\phi(I_p(t) - I_d(t), \infty)$，scale 1.0，前面说过，hard constraint
- Joint position tracking: $\phi(\theta - \theta_{target}, 2)$，scale 1.5，跟着reference motion走

### Regularization（负reward，penalize坏行为）

- Default joint: $\phi(\theta_t - \theta_0, 2)$，scale 0.2，不要离default pose太远
- Energy cost: $\|\tau\|\|\dot{\theta}\|$，scale -0.0001，torque × velocity = power，省电
- Action smoothness: $\|a_t - 2a_{t-1} + a_{t-2}\|_2$，scale -0.01，二阶差分penalty
- Large contact force: $\max(F_{L,R} - 400, 0, 100)$，scale -0.01，不要stomp ground

Action smoothness用二阶差分 $a_t - 2a_{t-1} + a_{t-2}$ 而不是一阶 $a_t - a_{t-1}$，这个细节很有意思。一阶差分penalty会让action变化慢，但可能allow "steady but wrong" action。二阶差分相当于jerk penalty，允许action变化但惩罚"变化的change"，输出会更smooth natural。

**正负reward scale的不对称**：正reward总和约6.9，负reward总和约-0.02。这个设计很重要——如果负reward太大，policy会学会"什么都不做"来avoid penalty，反而学不到locomotion。正reward必须dominate，让policy有动力去track command。

---

## Domain Randomization - Sim2Real的magic trick

这是整个paper的secret sauce。

Sim2real gap的本质：sim里的physics参数和real world不一样。如果你在sim里用固定参数训练，policy会overfit到那些参数，一到real就崩。

Domain randomization的思路：训练时把physics参数随机化，让policy见识过各种parameter combination，deploy时不管real是哪个参数，policy都在训练分布里见过类似的。

具体randomization项（Table III）：

**Sensor noise（additive Gaussian）**：
- Joint position: ±0.05 rad
- Joint velocity: ±0.5 rad/s
- Angular velocity: ±0.1 rad/s
- Euler angle: ±0.03 rad

这些range对应real sensor的actual noise level。训练时给observation加这些noise，policy学会对sensor noise robust。

**Physics参数randomization**：
- Friction: [0.1, 2.0] uniform — 地面可以从ice到rubber
- Motor strength: [95%, 105%] Gaussian — motor扭矩有±5% variance
- Payload: [-5, +5] kg Gaussian — robot可能多背5kg或少背5kg
- System delay: [0, 10] ms uniform — 通信延迟

注意payload ±5kg对1.2m humanoid（大概20-30kg）是巨大变化，20%的mass variation。这逼着policy必须真正robust to mass distribution change，不能overfit到某个特定mass。

Intuition：就像training一个运动员，让他穿不同重量的鞋、在不同摩擦力的地板、背着不同重量的背包都练过，比赛时不管什么条件都能发挥。

---

## Sim2Sim Calibration - 为什么MuJoCo能predict real

这是paper里最undervalued的部分。

作者做了一件事：calibrate MuJoCo的参数让它的dynamics尽量接近real world。验证方法是：

1. **Sine wave tracking**：让robot的leg swing joint去track一个sine wave reference，对比MuJoCo和real的joint position trajectory。Fig. 3显示两者几乎重合。
2. **Phase portrait**：在steady state walking下，画knee和ankle joint的 $(\theta, \dot{\theta})$ 平面trajectory。Fig. 4显示MuJoCo和real的phase portrait几乎重合，Isaac Gym偏离更大。

为什么phase portrait重要？因为它刻画的是steady state下的limit cycle。两个dynamical system如果phase portrait一致，说明它们的energy balance、damping characteristic、natural frequency都接近。这是dynamics matching的strong indicator。

整个sim2sim calibration的intuition：**如果MuJoCo能predict real，那policy在MuJoCo里能跑通就在real里大概率也能跑通**。这对没有real robot的研究者意义重大——你可以用MuJoCo验证，confidence比直接从Isaac Gym transfer到real高得多。

---

## Engineering细节里藏的魔鬼

几个让我觉得"这帮人是真的搞过real robot"的细节：

1. **100Hz policy + 1000Hz PD**：这个双层frequency是humanoid control的standard practice。Policy太慢react不过来，PD太快policy来不及思考。
2. **System delay randomization 0-10ms**：10ms对100Hz是1个control step，这个range刚好cover typical communication delay。
3. **Frame stack 15 vs 3**：actor需要更多历史来infer latent state，critic有privileged info所以不需要太多历史。
4. **Entropy coef = 0.001**：比标准PPO低很多。Humanoid一explore就摔，不能让policy太random。
5. **Learning rate 1e-5**：比一般PPO小10倍，因为batch size 8192×24巨大，gradient estimate很准，小step size够了。

---

## 这篇paper的limitation

说句实话，paper有几个明显短板：

1. **没有quantitative evaluation**：没给"能走多远"、"成功率多少"、"跌倒几次"这些hard number。只有qualitative的"能走"。
2. **没有baseline comparison**：没和其他humanoid RL方法比。
3. **Reward weight是hand-tuned**：没说怎么tune的，这些weight对性能影响巨大但不可reproduce。
4. **PD gains没披露**：$K_p, K_d$是多少？这对sim2real很重要。
5. **Network architecture没说**：MLP几层？hidden dim多少？activation function？
6. **Gait cycle time $C_T$ 没说**：step frequency是多少Hz？

这些missing detail让paper更像一个engineering report而不是research paper。但作为开源framework的announcement，它完成了使命——让别人能build upon。

---

## 我觉得最有价值的takeaway

如果让我提炼这篇paper最值得学习的几个点：

1. **Asymmetric Actor Critic + Privileged Info**：这个pattern不只适用于robot，任何有"训练时知道但deploy时不知道"的信息的场景都适用。
2. **Phase signal injection**：给policy inject structure prior（periodicity, contact pattern）比让policy从零学要efficient得多。
3. **Sim2Sim作为sim2real的stepping stone**：与其直接从fast-but-inaccurate sim跳到real，不如用slow-but-accurate sim做中间验证。
4. **Domain randomization的range要physically reasonable**：randomize太狠policy学不到东西，randomize太轻不够robust。这些range是trial-and-error的结晶。
5. **Reward design是craftsmanship**：exponential kernel、正负reward不对称、jerk penalty而不是velocity penalty，这些细节是工程经验的体现。

Reference:
- Humanoid-Gym project: https://sites.google.com/view/humanoid-gym
- Legged Gym (ETH base): https://github.com/leggedrobotics/legged_gym
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Gym: https://arxiv.org/abs/2108.10470
- MuJoCo: https://mujoco.readthedocs.io/
- Asymmetric Actor Critic: https://arxiv.org/abs/1710.06542
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- Domain Randomization (Tobin et al.): https://arxiv.org/abs/1703.06907
- Real RobotEra: https://www.robotera.com/

希望这个version更对intuitive understanding有帮助，Andrej。本质上这篇paper是engineering craft rather than scientific breakthrough，但很多valuable的细节就藏在这些craft里。

---

# Humanoid-Gym 深度解析

Andrej，这篇paper是RobotEra团队（Xinyang Gu, Yen-Jen Wang, Jianyu Chen）的工作，核心贡献是开源了一个基于Nvidia Isaac Gym的humanoid robot RL framework，重点解决zero-shot sim2real transfer这个老大难问题。下面我从工程直觉的角度逐层拆解。

## 1. 核心问题与设计哲学

Humanoid locomotion的sim2real gap比quadruped大得多，根本原因在于：
- **欠驱动系统的高动态性**：humanoid只有两只脚支撑，contact sequence更复杂，small mass perturbation就能导致fall
- **高DOF coupled dynamics**：XBot-S和XBot-L都是12个actuated joint (每条腿6个：hip yaw/roll/pitch, knee pitch, ankle pitch/roll)，加上floating base 6 DoF，total 18 DoF
- **POMDP setting**：real-world只有partial observation，sim是full observability

他们的设计哲学可以总结为：**"在Isaac Gym里fast train，在MuJoCo里carefully validate，最后在real world里deploy"**。这个三段式pipeline（Fig. 2）的关键insight是——Isaac Gym（GPU-based, PhysX）快但不准，MuJoCo（CPU-based, convex optimization contact）慢但准。通过calibration让MuJoCo逼近real，那么policy在MuJoCo里能跑通就在real里大概率也能跑通。

Reference: 
- Isaac Gym paper: https://arxiv.org/abs/2108.10470
- MuJoCo: https://mujoco.readthedocs.io/
- Legged Gym (ETH基础): https://github.com/leggedrobotics/legged_gym
- RSL_RL: https://github.com/leggedrobotics/rsl_rl

## 2. RL Formulation: POMDP + Asymmetric Actor Critic

### 2.1 MDP定义
$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, T, \mathcal{O}, R, \gamma \rangle$

变量含义：
- $\mathcal{S}$: state space (sim里full observable)
- $\mathcal{A}$: action space (12D joint position targets)
- $T(\mathbf{s}'|\mathbf{s}, \mathbf{a})$: transition dynamics
- $\mathcal{O}$: observation space (real里partial)
- $R(\mathbf{s}, \mathbf{a})$: reward function
- $\gamma \in [0,1]$: discount factor (论文用0.994，相当长horizon，effective horizon ≈ 1/(1-0.994) ≈ 167 steps)

Policy $\pi(\mathbf{a}|\mathbf{o}_{\leq t})$ 在deploy时只看observation history，不看privileged info。Expected return:
$$J = \mathbb{E}[R_t] = \mathbb{E}\left[\sum_t \gamma^t r_t\right]$$

### 2.2 PPO Loss
$$\mathcal{L}_\pi = \min\left[\frac{\pi(a_t|o_{\le t})}{\pi_b(a_t|o_{\le t})} A^{\pi_b}(o_{\le t}, a_t), \mathrm{clip}\left(\frac{\pi(a_t|o_{\le t})}{\pi_b(a_t|o_{\le t})}, c_1, c_2\right) A^{\pi_b}(o_{\le t}, a_t)\right]$$

变量解释：
- $\pi$: 当前policy
- $\pi_b$: behavior policy (旧版用于sample的policy)
- $A^{\pi_b}$: advantage estimator，基于GAE计算
- $c_1 = 0.8, c_2 = 1.2$: clip范围（比标准PPO的[0.8, 1.2]略宽，作者可能想增加exploration）
- ratio $\frac{\pi}{\pi_b}$ 衡量新旧policy偏离程度

注意这里actor只看 $o_{\leq t}$（partial observation history），但critic看 $s_t$（full state包括privileged info），这就是**Asymmetric Actor Critic**（Pinto et al. 2017, https://arxiv.org/abs/1710.06542）的核心思想。Intuition是：训练时让critic用god mode view给出更准确的value估计，但deploy时actor不需要privileged info，依然能在real world里work。这相当于把teacher forcing的思想引入RL。

### 2.3 Value Function Loss
$$\mathcal{L}_v = \|R_t - V(s_t)\|_2$$

变量：
- $R_t$: 实际return (GAE-bootstrap后的target)
- $V(s_t)$: critic对state $s_t$ 的value估计
- L2 norm，标准regression

### 2.4 GAE (Generalized Advantage Estimation)
参考Schulman et al. 2015: https://arxiv.org/abs/1506.02438

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k}(\gamma\lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是TD error。
论文里GAE discount $\lambda = 0.95$，balance bias和variance。

## 3. Observation Space 设计 - 工程直觉

Table I 是这篇paper最informative的部分之一，逐项拆解：

### 3.1 Actor可见的Observation (47D × 15 frames stack = 705D input)
- **Clock Input (2D)**: $[\sin(2\pi t/C_T), \cos(2\pi t/C_T)]$
  - 这就是periodic phase signal，告诉policy当前在gait cycle的哪个phase
  - $C_T$ = cycle time，1完整gait周期
  - 用sin/cos而不是linear ramp的好处：周期性smooth且无discontinuity
  - Intuition：这是inject prior knowledge about periodic locomotion的最elegant方式
- **Commands (3D)**: $\dot{P}_{x,y,\gamma}$ — 想要的forward/lateral velocity和yaw rate
- **Joint Position (12D)**: $\theta$ — 当前6个joint per leg × 2 legs
- **Joint Velocity (12D)**: $\dot{\theta}$
- **Angular Velocity (3D)**: $\dot{P}_{\alpha\beta\gamma}^b$ — base的roll/pitch/yaw rate
- **Euler Angle (3D)**: $P_{\alpha\beta\gamma}^b$ — base orientation
- **Last Actions (12D)**: $a_{t-1}$ — 帮助policy意识到自己刚才做了什么，smooth control

加起来：2+3+12+12+3+3+12 = 47 ✓

**Frame stack 15**：意味着policy看到过去15帧的observation，total input维度 = 47 × 15 = 705。这个temporal context让policy能从history推断一些latent state（比如base linear velocity，real world里IMU直接测不准）。

### 3.2 Critic可见的Privileged Observation (73D × 3 frames stack = 219D)
额外包括：
- **Friction (1D)**: 地面摩擦系数
- **Body Mass (1D)**: robot质量
- **Base Linear Velocity (3D)**: 在real里没有sensor直接测这个
- **Push Force (2D)**: 外部扰动力
- **Push Torques (3D)**: 外部扰动torque
- **Tracking Difference (12D)**: 当前joint position vs target
- **Feet Contact detection (2D)**: 左右脚是否着地

73 = 47 + 1 + 1 + 3 + 2 + 3 + 12 + 2 + 2 = 73 ✓

Frame stack 3（privileged短一些，因为privileged info本身就是当前状态描述）。

**关键insight**：训练时critic知道地面friction是多少、payload多重，能学到"这种情况下value应该这样估计"。但actor不知道，必须从history间接infer。这就是典型的symmetric information asymmetry trick。

## 4. Action Space与Control Pipeline

### 4.1 Action语义
Action是**joint position targets**给PD controller：
$$\tau_i = K_p^i (\theta_{target,i} - \theta_i) - K_d^i \dot{\theta}_i$$

- $\theta_{target,i}$: policy输出（action）
- $\theta_i, \dot{\theta}_i$: 当前joint state
- $K_p^i, K_d^i$: PD gains

PD controller在1000Hz跑，policy在100Hz跑。这意味着每10个PD tick才有1个新target。这种hierarchical frequency结构的好处：
1. policy不需要handle micro-level control，留给PD
2. 100Hz对MLP够快，对real-time system也够慢能process
3. 给policy的action一个"natural smoothing"

### 4.2 Gait Phase与Reference Motion
Reference motion用sinusoidal wave生成，involving pitch, knee, ankle。具体形式类似：
$$\theta_{target}(t) = \theta_{offset} + A \sin(2\pi t / C_T + \phi)$$

每个joint有自己的amplitude $A$ 和phase offset $\phi$。

### 4.3 Stance Mask $I_p(t)$
Fig. 6显示stance mask是contact planning：
- [0, 1]: 左脚swing，右脚stance
- [1, 0]: 左脚stance，右脚swing
- [1, 1]: 双脚stance (DS phase)
- [0, 0]: 双脚swing (理论上不应该出现)

这个mask和sinusoidal reference phase同步，inject prior knowledge about which foot should be on ground at which phase。

## 5. Reward Design - 工程上的art

总reward: $r_t = \sum_i r_i \cdot \mu_i$
- $r_i$: 各reward component
- $\mu_i$: weight (scale)

**关键kernel function**:
$$\phi(e, w) := \exp(-w \cdot \|e\|^2)$$

- $e$: tracking error
- $w$: sensitivity weight
- $\|\cdot\|^2$: squared L2 norm
- Intuition: 这是一个exponential decay的reward，error越小reward越接近1，error大时快速趋于0。比起linear/quadratic penalty，它的好处是gradient在error小时小（avoid overshooting），error大时也小（avoid exploding gradient）。

### 5.1 各Reward拆解 (Table IV)

| Reward | Equation | Scale $\mu_i$ | Intuition |
|--------|----------|---------------|-----------|
| Lin. vel tracking | $\phi(\dot{P}_{xyz}^b - \text{CMD}_{xyz}, 5)$ | 1.2 | 让base linear velocity跟command，$w=5$ sensitivity |
| Ang. vel tracking | $\phi(\dot{P}_{\alpha\beta\gamma}^b - \text{CMD}_{\alpha\beta\gamma}, 5)$ | 1.0 | yaw/roll/pitch rate跟command |
| Orientation tracking | $\phi(P_{\alpha\beta}^b, 5)$ | 1.0 | roll/pitch保持0（直立）|
| Base height tracking | $\phi(P_z^b - 0.7, 100)$ | 0.5 | $w=100$ 极strict，target height 0.7m |
| Velocity mismatch | $\phi(\dot{P}_{z,\gamma,\beta}^b - \text{CMD}_{z,\gamma,\beta}, 5)$ | 0.5 | 这些command设为0，penalize垂直bounce |
| Contact Pattern | $\phi(I_p(t) - I_d(t), \infty)$ | 1.0 | 让contact match reference，$w=\infty$ 是hard constraint |
| Joint Position Tracking | $\phi(\theta - \theta_{target}, 2)$ | 1.5 | 跟踪reference motion |
| Default Joint | $\phi(\theta_t - \theta_0, 2)$ | 0.2 | 偏离default pose的小penalty，避免weird pose |
| Energy Cost | $\|\tau\|\|\dot{\theta}\|$ | -0.0001 | torque × velocity = power，penalize energy |
| Action Smoothness | $\|a_t - 2a_{t-1} + a_{t-2}\|_2$ | -0.01 | 二阶差分penalty，避免jitter |
| Large contact | $\max(F_{L,R} - 400, 0, 100)$ | -0.01 | clamp foot contact force under 400N |

注意：
- $\text{CMD}_{z, \gamma, \beta}$ 在Velocity mismatch里**故意设为0**，因为我们不想控制vertical velocity、yaw rate（不是command driven的那个）、pitch rate，想保持它们为0确保stable walking。注意$\gamma$出现在Ang vel tracking（command driven）和mismatch（penalize）中，看似矛盾——其实mismatch里把command减掉后是额外的，应该减command之后的residual仍然要小，但作者这里可能是处理不同subset的命令，需要看code。
- $w=\infty$ 在Contact Pattern里相当于 $\phi \to \mathbb{1}[\text{perfect match}]$，这是hard reward shaping
- $w=100$ for height 比 $w=5$ for velocity 更strict，因为height变化对stability影响极大
- Action smoothness用二阶差分 $a_t - 2a_{t-1} + a_{t-2}$ 而不是一阶 $a_t - a_{t-1}$，intuition是二阶差分对应"jerk"的概念，让action trajectory smooth

### 5.2 Reward scale总和
正reward: 1.2 + 1.0 + 1.0 + 0.5 + 0.5 + 1.0 + 1.5 + 0.2 = 6.9
负reward: -0.0001 - 0.01 - 0.01 ≈ -0.02
正负差距大，避免reward hacking by inaction。

## 6. Domain Randomization - Sim2Real的magic

Table III的domain randomization是zero-shot transfer的关键：

| Parameter | Range | Operator | Distribution | Intuition |
|-----------|-------|----------|--------------|-----------|
| Joint Position | [-0.05, 0.05] rad | additive | Gaussian 1σ | sensor noise on encoder |
| Joint Velocity | [-0.5, 0.5] rad/s | additive | Gaussian | velocity estimation noise |
| Angular Velocity | [-0.1, 0.1] rad/s | additive | Gaussian | IMU gyro noise |
| Euler Angle | [-0.03, 0.03] rad | additive | Gaussian | IMU attitude estimation noise |
| System Delay | [0, 10] ms | - | Uniform | communication/compute delay |
| Friction | [0.1, 2.0] | - | Uniform | terrain friction unknown |
| Motor Strength | [95, 105]% | scaling | Gaussian | motor torque constant variance |
| Payload | [-5, 5] kg | additive | Gaussian | added mass uncertainty |

**关键insight**：
- 这些range都是物理上reasonable的noise amplitude
- Friction用Uniform因为完全不知道terrain性质
- Motor Strength用scaling因为相对torque error
- Payload ±5kg对1.2m humanoid是大变化，强制robustness
- System delay 0-10ms对100Hz control是1个step内，刚好能cover

## 7. Hyperparameters分析 (Table II)

- **8192 environments**: massively parallel training，Isaac Gym GPU优势
- **2 training epochs per iter**: PPO多epoch复用data
- **Batch size 8192 × 24**: 大batch增加gradient estimate质量
- **Episode 2400 steps**: 在100Hz下24秒episode
- **Discount γ = 0.994**: effective horizon ≈ 167 steps ≈ 1.67秒，对应gait cycle几倍
- **GAE λ = 0.95**: 标准value
- **Entropy coef = 0.001**: 比较低，作者不希望太多exploration noise（humanoid一explore就fall）
- **c1=0.8, c2=1.2**: PPO clip range
- **Learning rate 1e-5**: 比一般PPO小，因为batch大
- **Frame stack 15 / 3**: 前面分析过

## 8. Calibration - Sim2Sim的核心

### 8.1 Sine Wave Tracking
Fig. 3显示leg swing sine wave在MuJoCo和real world里几乎重合。这是end-to-end dynamics calibration的间接证据——如果joint position tracking trajectory一致，说明PD controller + motor + dynamics整体response匹配。

### 8.2 Phase Portrait分析
Fig. 4是phase portrait of left knee和left ankle pitch joint在5秒、0.5 m/s行走时的状态。Phase portrait是 $(\theta, \dot{\theta})$ 平面上的trajectory：
- 如果是perfect limit cycle，phase portrait是closed curve
- MuJoCo和real world的phase portrait几乎重合
- Isaac Gym和real world偏离更大

**Intuition**：phase portrait重合意味着系统在steady state下的能量平衡一致。这是dynamics matching的strong indicator。

## 9. 实验设计与Limitations

### 9.1 成功的部分
- XBot-S (1.2m) 和 XBot-L (1.65m) 都实现zero-shot transfer
- Flat和uneven terrain都能traverse
- 两台不同size robot验证了framework generalization

### 9.2 未提及的limitation
1. 没有quantitative evaluation metric（没有distance、speed、stability rate等具体数字）
2. 没有failure case analysis
3. Uneven terrain的具体roughness没量化
4. 没有comparison with其他humanoid RL方法
5. Reward weight $\mu_i$ 是手动tuned，没说怎么tune的
6. PD gains $K_p, K_d$ 没披露
7. Gait cycle time $C_T$ 没披露
8. 神经网络架构（MLP层数、hidden dim、activation）没披露

## 10. 与同期工作的context

- **Radosavovic et al. (2023) "Learning humanoid locomotion with transformers"**: https://arxiv.org/abs/2303.03381 — 用transformer替代MLP做humanoid locomotion
- **Radosavovic et al. (2024) "Humanoid locomotion as next token prediction"**: https://arxiv.org/abs/2402.19469 — 把locomotion当成autoregressive prediction问题
- **Cheng et al. (2024) "Expressive whole-body control for humanoid robots"**: https://arxiv.org/abs/2402.16796 — upper body + locomotion联合control
- **He et al. (2024) "Human-to-humanoid real-time whole-body teleoperation"**: https://arxiv.org/abs/2403.04436 — 用人control humanoid采集数据

Humanoid-Gym定位是**infrastructure work**，提供一个让其他人能复现和build upon的baseline framework。在这个意义上它和Legged Gym（ETH）的角色类似，但focus在humanoid上。

## 11. 给Karpathy的几个深入思考点

1. **为什么用PD target作为action而不是torque**？这是RL locomotion的经典debate。PD target相当于把控制分成两层：low-level PD（model-based）+ high-level policy（learned）。这其实是implicit 的hierarchical RL，且lower level提供stability prior。

2. **Periodic phase signal [sin, cos]的必要性**？理论上policy可以从joint history里推断phase。但explicit inject phase让policy不需要学这个映射，sample efficiency大大提高。这是informed exploration的一种形式。

3. **Asymmetric Actor Critic的sim2real解释**：训练时critic看privileged info相当于"上帝视角指导"，但actor必须自己infer。Deploy时critic消失，actor依然能work，这意味着actor其实学会了从observable history里近似推断那些privileged info。这是一种知识distillation。

4. **Contact Pattern $w=\infty$的隐含意义**：这相当于一个hard constraint，policy必须match stance mask。这是把motion planning的contact schedule和RL control结合，比让policy完全free explore contact pattern更sample efficient。

5. **Sim2Sim calibration的更深层意义**：如果MuJoCo和real匹配，且Isaac Gym和MuJoCo不匹配，那么Isaac Gym训练的policy能transfer到MuJoCo就说明policy本身对dynamics robust。这是一个"dynamics generalization"的间接测试。

## 12. 代码与资源

- Project page: https://sites.google.com/view/humanoid-gym
- GitHub repo: https://github.com/robotera/Humanoid-Gym (推测)
- RobotEra官网: https://www.robotera.com/
- 基础设施: legged_gym + rsl_rl

希望这个deep dive对你的intuition building有帮助，Andrej。这篇paper虽然contribution看起来modest（engineering framework），但里面有很多craftsmanship detail值得学习。Reward weights、domain randomization ranges、observation design这些都是经过trial-and-error得到的工程知识，对想build humanoid RL system的人极有价值。
