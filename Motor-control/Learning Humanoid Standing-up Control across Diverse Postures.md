---
source_pdf: Learning Humanoid Standing-up Control across Diverse Postures.pdf
paper_sha256: 92961cf4b5eb503e9fcd5d8a7bfcd2bb7269025a49dfb832e34e52ebb1b74ec0
processed_at: '2026-08-05T13:21:03-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HoST 用人话讲

## 一句话说清楚这paper在干嘛

让一个人形机器人（Unitree G1），从各种乱七八糟的姿势（躺在地上、靠着墙、趴在台阶上、斜坡上）自己站起来，全程用RL from scratch学，不用任何reference motion，最后能直接部署到真实机器人上。

就这么个事。听起来简单，做起来极难。

---

## 为什么这事儿难

你想想，humanoid robot站起来这个动作，对人来说就是个翻身-撑地-跪起-站立的过程，三岁小孩都会。但对机器人来说，这事儿简直是地狱难度：

**第一，contact sequence是变的**。走路的时候脚永远在底下，contact pattern是固定的。但站起来的时候，先是背着地，然后手着地，然后膝盖着地，然后脚着地，这个sequence每次都不一样，取决于你初始姿势是什么样。RL policy得自己figure out什么时候该把哪儿放下来。

**第二，这是个multi-stage task**。翻身、跪起、站立是三个完全不同的sub-skill，每个的dynamics都不一样。你要是只用一个reward function让policy自己学，它根本不知道先干啥后干啥。

**第三，high DoF + wide joint limits = violent motion**。G1有23个关节，每个关节都能转很大角度，torque也大。你用RL去explore，policy最喜欢干的事就是把所有关节瞬间打到极限，然后机器人就开始像鱼一样在地上狂跳。在sim里这能站起来，到real robot上电机直接烧了。

**第四，bipedal balance本身就难**。四足机器人摔倒了自己正过来很容易，因为four legs = stable support。但两足机器人重心高、支撑面就两个脚掌大，起身过程中稍微angular momentum不对，摔回去了。

---

## 之前的人都怎么搞的，为啥不行

**Classical方法（model-based trajectory optimization）**：
日本AIST的Kanehiro他们2003年就搞过，HPR-2 humanoid能躺下站起来。方法是人工设计motion trajectory，然后用motion planning去track。问题是：
- 每个场景都要重新设计trajectory，换个姿势就得重来
- 对disturbance极敏感，推一下就崩
- 计算量大，实时性差

**RL + reference motion方法（DeepMimic, ASE）**：
NVIDIA的Peng et al.用motion capture data做reference，让RL去imitate。问题是：
- 你得有motion clip，但humanoid从各种姿势起身的clip没有现成的
- 只能做clip里有的posture，换个没见过的posture就fail
- 大部分都在sim里跑，没deploy到real robot

**RL from scratch方法（Tao et al. Learning to Get Up）**：
这个最接近HoST的思路，但只在sim里跑，没上real robot，而且动作特别暴力，真机跑不了。

所以HoST的定位很清楚：**第一个做到real robot + from scratch + 多种posture + high DoF的stand-up control**。Table I那个对比表就是这个意思。

---

## HoST怎么解决的 - 四个核心trick

### Trick 1: Multi-Critic - 这是最关键的

直觉上讲，single critic PPO在这任务上完全fail（Table IIIa显示0% success rate）。原因是reward太复杂了，task reward是sparse的（只有达到高度才给），style reward是dense的（每步都算），regularization是tiny weight但every step都有，post-task reward只有最后几步有。一个critic要去predict这么一个multi-modal的return，gradient之间互相打架，policy update就发散了。

HoST的做法：把reward分成4个group，每个group一个critic。

- $r^{task}$: 高度、朝向 - "你站起来了吗？"
- $r^{style}$: 关节别超限、脚别打滑、腰别乱扭 - "站起来的姿势像不像样？"
- $r^{regu}$: torque、action rate、smoothness - "动作别太暴力"
- $r^{post}$: 站起来后要稳住 - "别站起来又坐下"

每个critic $V_{\phi_i}$ 学自己的return：

$$\mathcal{L}(\phi_i) = \mathbb{E}[\| r_t^i + \gamma V_{\phi_i}(s_t) - \bar{V}_{\phi_i}(s_{t+1}) \|^2]$$

- $r_t^i$: group $i$ 在 $t$ 时刻的reward
- $\gamma = 0.99$: discount factor
- $\bar{V}_{\phi_i}$: target value network

然后每个critic独立算GAE advantage，再做batch-level normalization后加权求和：

$$A = \sum_i w^i \cdot \frac{A_{\phi_i} - \mu_{A_{\phi_i}}}{\sigma_{A_{\phi_i}}}$$

- $A_{\phi_i}$: critic $i$ 的advantage
- $\mu_{A_{\phi_i}}, \sigma_{A_{\phi_i}}$: 当前batch的均值和标准差
- $w^i$: group weight

**这个normalization是必须的**，因为task advantage可能是100的量级，regularization advantage可能是0.001的量级，直接加起来regularization就消失了。

**人话总结**：与其让一个critic去学一个四不像的return，不如让四个critic各学各的，每个都学的简单干净，最后再democratic vote一下。

### Trick 2: Force Curriculum - 给机器人"撑腰"

从地上爬起来最难的是最开始那一下翻身。你想，机器人平躺在地上，random action下去根本翻不了身，因为翻身需要angular momentum，但平躺状态下没有势能可以借。

HoST的做法很萌：**在robot base上吊一个向上的力**，一开始200N（G1大概35kg，所以200N差不多托住一半体重），然后随着训练progress，如果policy能在episode结束时保持head height达标，就减20N，直到减到0。

**这个trick的灵感来源很妙**： developmental psychology的研究说人类婴儿学站立时，父母会托一把，然后慢慢放手。这个force curriculum就是这个"托一把"。

**为什么RND不行**：Table IIIb显示，用RND curiosity reward的版本在ground/wall/slope上全fail，只有platform上有点success。因为RND鼓励的是"探索novel state"，在23-DoF humanoid上，随便乱动就是novel state，policy根本不会朝"起身"这个目标去explore。而force curriculum是**goal-directed**的，直接把policy拉到能起身的trajectory附近。

**人话总结**：与其让机器人自己瞎试，不如先"帮"它一把让它学会大致trajectory，再慢慢撤掉帮助让它自己fine-tune。

### Trick 3: Action Bound Curriculum - 限速

这个trick简单粗暴但有效。Policy的action $a_t$ 是joint position delta，公式是：

$$p_t^d = p_t + \beta \cdot a_t$$

- $p_t^d$: PD target
- $p_t$: 当前joint position
- $a_t \in [-1, 1]^{23}$: normalized action
- $\beta$: rescaler, 控制每步最大joint position变化

训练初期 $\beta = 1$，意味着每步joint能变化 $\pm 1$ rad，这是大range，方便explore。然后curriculum往下调，每次降0.02，下界0.25。

**为什么必须限速**：如果不限速，policy学会的最优解是"用最大力瞬间把关节打到目标位置"，这在sim里work，到real robot上就是疯狂抖动，因为real电机有torque limit、backlash、friction，高频大动作直接激发structural resonance，机器人就废了。

**Table IIIc的ablation很说明问题**：
- w/o bound: success 98.8%，但smoothness从2.90飙到9.52，energy从1.35飙到3.59
- 固定bound=0.25: success还行但energy高，因为初始就限太死，explore不到efficient motion
- curriculum: smoothness 2.90, energy 1.35 - sweet spot

**人话总结**：先让机器人手脚大开地学，等它会了再让它"优雅"地动。直接一上来就让它优雅，它根本学不会；一直让它暴力，real robot扛不住。

### Trick 4: L2C2 Smoothness Regularization - 防抖

L2C2来自Kobayashi IROS 2022，核心思想是**不仅actor要平滑，critic也要平滑**。因为critic不平滑会导致policy gradient在不同state上有spurious jump，间接让actor输出也spike。

Loss是：

$$\mathcal{L}_{L2C2} = \lambda_\pi D(\pi_\theta(s_t), \pi_\theta(\bar{s}_t)) + \lambda_V \sum_i D(V_{\phi_i}(s_t), V_{\phi_i}(\bar{s}_t))$$

- $\bar{s}_t = s_t + (s_{t+1} - s_t) \cdot u$, $u \sim \mathcal{U}(\cdot)$: 在 $s_t$ 和 $s_{t+1}$ 之间插值的state
- $D$: L2距离
- $\lambda_\pi = 1$, $\lambda_V = 0.1$: actor和critic的权重

**这个trick在sim里看不出明显差异**（Table IIIc的w/o-L2C2在sim里success还行），但**一到real robot就崩**（Table IV显示slope上0/5 success）。因为real hardware对oscillation更敏感，sim里小的抖动到real上被friction和torque limit放大成大抖动。

**人话总结**：L2C2在sim里是个"nice to have"，在real robot上是"must have"。这种sim里看不出来但real里critical的trick，是sim-to-real最难的部分。

---

## Observation Space的设计也有讲究

$$s_t = [\omega_t, r_t, q_t, p_t, \dot{p}_t, a_{t-1}, \beta]$$

- $\omega_t$: base angular velocity (IMU)
- $r_t, q_t$: base roll, pitch
- $p_t, \dot{p}_t$: 23个joint的position和velocity
- $a_{t-1}$: 上一步action
- $\beta$: action rescaler

**两个亮点**：

**第一，只用proprioception，没有vision**。因为起身过程中，robot自己姿态变化主导dynamics，terrain几何信息在4种terrain设计下，proprioception已经足够disambiguate。这和locomotion with perception（HIM, Humanoid Parkour）的思路不同，但在这任务上work。

**第二，把过去5个state拼起来feed给policy**。Table IIId显示，history=1在Wall terrain上success rate从94.2%掉到64.5%。Intuition是：同样的joint configuration，刚刚是被手撑着还是被膝盖撑着，后续动作完全不同。5步历史是隐式的contact estimation - 类似Boston Dynamics的proprioceptive contact detection思路。

---

## Terrain设计 - 4种场景

- **Ground**: 平地，最简单
- **Platform**: 台阶上（20cm-92cm高度），模拟坐在椅子上
- **Wall**: 墙边（14°-84°倾斜），模拟靠着东西
- **Slope**: 斜坡（1°-14°），模拟斜面

这4种terrain对应real world的常见场景。训练时用4种terrain，deploy时还测了草地、木台、石路、树靠 - generalization还不错。

---

## Domain Randomization - CoM offset最critical

Table II的randomization包括mass、friction、PD gain、control delay等一堆。但Fig 9a的ablation显示，**CoM offset是最critical的** - 关掉它success rate从100%掉到30%。

**Intuition**：起身是重心control的任务，CoM不准=task definition本身错了。其他randomization（friction、delay等）是robustness的事，CoM是correctness的事。

---

## Sim-to-Real的Phase Plot分析

Fig 9b是我觉得这篇paper最有价值的诊断。作者把hip和knee的phase plot（position vs velocity）在sim和real上对比：

- joint position: sim和real一致
- joint velocity: **有明显discrepancy**

这意味着real电机的torque输出和sim命令有差距。position level被PD controller的integrator平滑了，velocity level直接暴露。

**这就是为什么real部署时要把hip stiffness从sim的150调到200，knee从200调到275**。因为real的effective stiffness比sim低，得补偿。这个发现和ANYmal的Hwangbo et al. (Science Robotics 2019)一致 - actuator model是legged robot sim-to-real的关键瓶颈。

---

## Real Robot实验结果

**Main results (Table IV)**:
- HoST: 20/20 success, smoothness 4.01
- w/o-L2C2: 11/20 success, smoothness 6.54

w/o-L2C2在slope上完全fail，platform上only 2/5。L2C2的criticality实锤了。

**Emergent properties**（没训练过但涌现出来的能力）：
- 3kg球撞击不倒（Fig 10a）
- 12kg payload（2倍trunk质量）能站起来（Table V）
- 15° slippery slope能平衡（Fig 10c）
- 大力推搡后能recover不摔倒（Fig 10b）

这些emergent property说明force curriculum学到的不仅是"从地上站起来"，而是更general的"return to upright pose"。

**Outdoor泛化**（Fig 8）：
草地、木台、石路、石台、树靠都能work。这些terrain训练时没见过，policy generalize了。

---

## Reward Design的intuition

Table VI的reward很复杂，但intuition清楚：

**Task reward**：高度、朝向达标就给reward，sparse的，"你站起来了吗？"

**Style reward**：用一堆indicator function $\mathbb{1}(\cdot)$做hard penalty，"关节别超限、脚别打滑、腰别乱扭"。这部分的目的是**禁止sim里能做但real里不行的motion**，是sim-to-real的关键。

**Regularization**：action rate、smoothness、torque，tiny weight但every step都加，"动作别太暴力"。

**Post-task**：站起来后要稳住，"别站起来又坐下"。

每个group一个critic，各学各的return，避免gradient conflict。

---

## 我觉得这篇paper的亮点和不足

**亮点**：

1. **Multi-critic for multi-stage task这个framing很清楚**，解决了single critic在复杂reward下diverge的问题
2. **Force curriculum的婴儿学习metaphor很motivating**，而且实验上证明比RND强
3. **Action bound curriculum**简单trick解决大问题，Fig 6的trade-off analysis佐证了必要性
4. **L2C2在multi-critic上的integrate**是工程亮点
5. **Real-world deployment非常extensive**：4种terrain + outdoor + payload + disturbance
6. **Phase plot分析**揭示了actuator model的sim-to-real gap，是honest diagnostic

**不足**：

1. **Motion自然度有限**：没有reference motion，动作还是机器式
2. **Perception缺失**限制了outdoor generalization，sitting on sofa这种场景会fail
3. **Prone和Supine同时训练interfere**没解决，说明reward shaping还不够general
4. **Energy efficiency**和human比还有差距
5. **没有fall detection和handoff to locomotion**，整个recovery system不完整

---

## 对field的影响预测

- Multi-critic RL会在humanoid control里被广泛adopt
- Action bound curriculum会成为sim-to-real transfer的standard trick
- Stand-up control会变成humanoid robot的标准capability
- 后续工作会在perception-augmented stand-up和full recovery pipeline上展开

---

## 参考链接

- Project page: https://humanoid-standingup.github.io
- Code: https://github.com/OpenRobotLab/HoST
- Multi-Critic原始paper: https://openreview.net/forum?id=WcCfITY2bQS
- L2C2: https://ieeexplore.ieee.org/document/9982242
- Hwangbo et al. actuator modeling: https://www.science.org/doi/10.1126/scirobotics.aau5872
- Learning to Get Up (Tao et al.): https://tianxintao.github.io/projects/learning-to-get-up/
- FRASA (Gaspard et al.): https://arxiv.org/abs/2410.08655
- RND: https://arxiv.org/abs/1810.12894
- Domain Randomization: https://arxiv.org/abs/1703.06907

---

## 最后给你（Karpathy）的intuition

1. **Multi-critic本质是gradient conflict mitigation**，不是reward decomposition trick。在multi-stage task里，single critic学的return是multi-modal的，optimization难度爆炸。Multi-critic让每个critic学homogeneous return，大大降低难度。

2. **Curriculum不只是调reward，也能调env force和action bound**。force curriculum给policy"撑腰"让它先学rough trajectory，action bound curriculum后期收紧让它fine-tune到smooth motion。

3. **Proprioception在contact-rich task里能encode contact history**。5步历史是隐式的contact estimation，替代explicit contact sensor。

4. **Smoothness regularization在sim里看不出差异，real上必死**。L2C2这种Lipschitz regularization是real deployment的必要条件。

5. **CoM offset的randomization最critical**，因为起身是重心control任务，CoM不准=task definition错了。

6. **Phase plot是诊断sim-to-real gap的最好工具**，position level看着没问题，velocity level暴露actuator model缺陷。

7. **Force curriculum学到的不只是stand-up，是general "return to upright" capability**，所以有那些emergent robustness property。

希望这些intuition对你有用。HoST的核心贡献是把stand-up从"用model-based方法解决的特殊问题"变成了"用RL from scratch可以scale的general capability"，这对humanoid robot的fall recovery和broader deployment意义重大。

---

# HoST: Humanoid Standing-up Control 深度解析

## 1. Paper 的 Core Thesis

这篇paper来自Shanghai AI Lab的OpenRobotLab（也是PALM团队，何凯明最近加入）， tackling的核心问题是humanoid robot从任意fallen/diverse posture站起来的control。 在humanoid robot领域, locomotion和loco-manipulation已经被RL攻陷得差不多了（参见AnyTeleop, OmniH2O, HumanPlus等），但**fall recovery / stand-up from arbitrary pose**这块儿长期被model-based trajectory optimization把持着，原因是这个task有几个unique的难点：

- **Time-varying contact points**: 从地上爬起来, 手、膝盖、脚跟、躯干都会轮番着地, contact sequence不是固定的
- **Multi-stage motor skills**: 先翻身(righting) → 跪起(kneeling) → 站立(rising)，三个阶段是emergent的，需要连续transition
- **Precise angular momentum control**: bipedal system重心高、支撑面窄，fall recovery过程中稍微angular momentum发散就摔回去
- **High DoF wide joint limits → violent motions**: Unitree G1有23 DoF, 每个关节都有wide position range和大torque，纯RL from scratch会学出"地鼠打洞"式的爆力起身

Table I是这篇paper的positioning statement。注意它claim同时满足：real robot + no prior trajectory + beyond ground + high DoF + 1-stage training。 前面几个工作（Peng et al. ASE, Haarnoja et al. soccer robot等）都是缺一个或者两个的。

参考链接：
- Project page: https://humanoid-standingup.github.io
- Code: https://github.com/OpenRobotLab/HoST
- 类似的quadruped recovery工作: https://arxiv.org/abs/1901.07517
- ASE (Peng et al.): https://research.nvidia.com/labs/toronto-ai/ASE/
- Multi-critic原始paper (Mysore et al. ICLR 2022): https://openreview.net/forum?id=WcCfITY2bQS

---

## 2. MDP Formulation 的细节

### 2.1 Observation Space

$s_t = [\omega_t, r_t, q_t, p_t, \dot{p}_t, a_{t-1}, \beta]$

- $\omega_t \in \mathbb{R}^3$: robot base的angular velocity (IMU读数)
- $r_t, q_t \in \mathbb{R}$: base的roll和pitch角(注意没有yaw, 因为yaw对起身不重要, 而且yaw在IMU中是unobservable的, 长时间drift)
- $p_t, \dot{p}_t \in \mathbb{R}^{23}$: G1的23个joint的位置和速度
- $a_{t-1} \in \mathbb{R}^{23}$: 上一步的action
- $\beta \in (0, 1]$: action rescaler, 标量, 用作implicit speed bound

**关键insight**: HoST只用了**proprioceptive state**, 没有任何exteroceptive perception(无depth camera, 无IMU-based height map)。 这点和很多locomotion with perception的工作不同(如HIM[28]、Humanoid Parkour[55])。 这是因为standing-up过程中, robot本身姿态变化主导了dynamics, 而terrain的几何信息虽然重要, 但在4种terrain的设计下, proprioception已经足够disambiguate。

**Historical state的augmentation**: 论文把**过去5个state**拼起来feed给policy, 这是个非常关键的设计。 Table III (d)的ablation显示, history=1 (即只用当前state)在Wall terrain上掉到64.5% success rate, 而history=5则达到94.2%。 这个intuition是: contact-rich任务中, current state对接触历史的"memory"是必要的, 因为同样的joint configuration, 刚刚是被手撑着还是被膝盖撑着, 后续动作完全不同。 这是隐式的contact estimation - 类似于后来Boston Dynamics的"proprioceptive contact detection"思路。

### 2.2 Action Space

PD controller作为底层actuator, action是joint position的**delta**:

$p_t^d = p_t + \beta \cdot a_t$

$\tau_t = K_p \cdot (p_t^d - p_t) - K_d \cdot \dot{p}_t$  (Eq. 1)

- $p_t^d$: PD target joint position
- $K_p, K_d$: PD的stiffness和damping
- $a_t \in [-1, 1]^{23}$: 被bound到[-1,1]的normalized action
- $\beta$: 把normalized action rescale到实际joint position delta, **关键** - 这是隐式的speed bound

这个设计的intuition是: 一个raw action值0.5在β=1时意味着0.5 rad的joint position step, 但在β=0.25时只意味着0.125 rad的step。 通过curriculum让β从1逐渐降到0.25, policy的训练过程是"先做大动作explore, 然后精细微调"。 这个trick在quadruped RL里也常见, 但HoST是第一个把它formalize成"motion speed bound"的。

Policy运行频率50Hz, PD controller 200Hz(sim)/500Hz(real), 这意味着每个policy step对应4-10个PD step, 这种asymmetric rate是standard practice。

---

## 3. Multi-Critic Architecture - 这篇paper最critical的innovation

### 3.1 为什么Single Critic Fail

Table III(a)最dramatic的ablation: 单critic的版本在所有terrain上success rate都是0%。 这看起来很反直觉, 因为single critic是PPO的标准实现, 在locomotion上都work。

**intuition**: standing-up的reward非常diverse:
- task reward: 高度达到target, base orientation正确
- style reward: 关节不要超限, 脚不要打滑, 腰不要乱转
- regularization: action rate, smoothness, torque
- post-task: 站起来后稳定

这些reward的**scale和time-scale完全不同**:
- task reward是sparse的: 在h_base > H_target才有大量reward
- style reward是dense的: 每步都计算
- regularization是非常small weight但每步都加
- post-task只有最后几步有

single critic要approximate一个巨大的multi-modal value function, gradient conflict很严重。 这就导致policy update被各种reward的pull-tug搞得发散。 **Multi-critic的核心insight**: 把reward按"temporal profile + functional role"分组, 每组单独学一个value function, 这样每个critic学的都是更homogeneous的return, approximation更容易。

### 3.2 Multi-Critic Formulation

每个reward group $i$ 有自己的critic $V_{\phi_i}$, loss:

$\mathcal{L}(\phi_i) = \mathbb{E}[\| r_t^i + \gamma V_{\phi_i}(s_t) - \bar{V}_{\phi_i}(s_{t+1}) \|^2]$  (Eq. 2)

- $r_t^i$: reward group $i$ 在时间$t$的total reward
- $V_{\phi_i}$: critic network
- $\bar{V}_{\phi_i}$: target value function (用EMA或者periodic update)

每个critic独立用GAE[41]计算advantage:

$A_{\phi_i} = \sum_l (\gamma\lambda)^l \delta_{t+l}^i$

where $\delta_t^i = r_t^i + \gamma V_{\phi_i}(s_{t+1}) - V_{\phi_i}(s_t)$ is TD error of critic $i$.

**Aggregation的关键技巧**: 多个advantage不能直接相加, 因为不同critic的advantage scale完全不同(task advantage可能是100, regularization advantage可能是0.001)。 HoST用**batch-level normalization**:

$A = \sum_i w^i \cdot \frac{A_{\phi_i} - \mu_{A_{\phi_i}}}{\sigma_{A_{\phi_i}}}$

- $\mu_{A_{\phi_i}}, \sigma_{A_{\phi_i}}$: 当前batch里advantage $i$ 的均值和标准差
- $w^i$: reward group $i$ 的weight, 手工设置

这个归一化是借用了Multi-Critic原始paper (Mysore et al. ICLR 2022, [33])的思路, 但HoST应用到了非常different的setting - 之前的multi-critic主要在motion style transfer里用, HoST是用在了一个**contact-rich的multi-stage task**上。

**PPO Policy Gradient**:

$\mathcal{L}(\theta) = \mathbb{E}[\min(\alpha_t(\theta) A_t, \text{clip}(\alpha_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$  (Eq. 3)

- $\alpha_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: importance ratio
- $\epsilon = 0.2$: clip ratio, 防止policy update过大

Architecture: actor是3层MLP [512, 256, 128], 每个critic是2层MLP [512, 256]。 **注意**: critic是smaller的, 因为每个critic学的return更简单, 不需要那么多capacity。

---

## 4. Force Curriculum - Exploration Strategy

### 4.1 为什么 Exploration Hard

从flat random init开始, G1的23 DoF在ground上几乎不可能探索到"先翻身、再跪起"的sequence, 因为：
- 翻身需要angular momentum, 但初始pose是stable supine, 没有"势能"驱动翻身
- 翻身完成后需要立即把momentum消掉, 否则就over-rotate
- 翻身→跪起的transition需要恰好在base接近vertical时开始engage膝盖和hip

random noise action在这种情况下基本只能产生"鱼跳"式无效运动。

### 4.2 Force Curriculum 的设计

灵感来源: 人类婴儿学站立时, 父母会托一把([6, 49] - Claxton 2012 developmental psychology)。 HoST的模拟方式是: **在robot base上施加垂直向上的force $\mathcal{F}$**, 初始200N, 然后随着训练progress, 如果当前episode结束时head height $h_{head}$ 达到target $H_{head}$, 就把force减小20N。 下界是0N。

**Key design decisions**:
- Force只在base接近vertical时才生效(避免在fall阶段强行吊起来)
- 减小条件: episode结束时的head height达到target, 这意味着policy必须自己学会maintain高度, 而不是依赖force
- 减小步长: 20N/step, 在4096 parallel env下, 大约几百个iteration就完全fade out

### 4.3 为什么RND Fails

Table III(b)显示, 用RND(random network distillation, [1]) curiosity reward的版本在ground/wall/slope上都fail, 只有platform上有点success (19.8%)。 

**intuition**: RND的intrinsic reward是"state novelty", 在high-DoF humanoid上, 任何random action都会产生novel state, 所以RND会鼓励policy到处探索, 但不会导向"起身"这个具体target。 而force curriculum是**goal-directed**的: 它直接把policy拉到能起身的trajectory附近, 让policy学会"剩下的事自己做"。

参考:
- RND paper: https://arxiv.org/abs/1810.12894
- Curriculum learning经典: https://arxiv.org/abs/2107.07815

---

## 5. Motion Constraints - Real-World Deployability的关键

### 5.1 Action Bound (Rescaler) β

这部分我前面在MDP里讲过β的作用, 这里讲**为什么需要curriculum式decrease**:

- 一开始β=1: policy可以输出[-1,1]的全范围action, 在β=1时, joint position step是rad级别, 这是必要的, 否则policy无法学会大range的motion
- 但如果保持β=1, 最终policy会学到用最大action输出快速切换关节位置, 这在real hardware上表现为"机器人疯狂抖动", 因为:
  1. 真实电机的torque limit比sim里更紧
  2. 真实关节有backlash、friction
  3. 高频大动作会激发structural resonance

- β下降的curriculum: 训练初期β=1, 当policy能在episode结束时维持head height达target, β减小0.02, 下界0.25

**Table III(c) ablation**:
- w/o-Bound: success rate还行(98.8% on ground), 但motion smoothness $E_{smth}$从2.90升到9.52, energy $E_{engy}$从1.35升到3.59。 这意味着robot确实能站起来, 但站起来的过程像在做CrossFit
- Bound0.25(固定): ground上99.8%但energy很高(2.75), 因为初始就限制action range, 探索不到efficient motion pattern
- Curriculum β: 在中间找到了balance, 2.90的smoothness + 1.35的energy

### 5.2 L2C2 Smoothness Regularization

L2C2来自Taisuke Kobayashi的IROS 2022 paper [20], 全称"Locally Lipschitz Continuous Constraint"。 HoST把它integrate到multi-critic里。

**核心思想**: 不仅actor要平滑, **critic也要平滑**。 因为critic不平滑会导致policy gradient在不同state上有spurious的jump, 间接造成actor输出也spike。

Loss:

$\mathcal{L}_{L2C2} = \lambda_\pi D(\pi_\theta(s_t), \pi_\theta(\bar{s}_t)) + \lambda_V \sum_i D(V_{\phi_i}(s_t), V_{\phi_i}(\bar{s}_t))$

- $\bar{s}_t = s_t + (s_{t+1} - s_t) \cdot u$, $u \sim \mathcal{U}(\cdot)$: 在$s_t$和$s_{t+1}$之间的interpolated state
- $D$: 距离metric, 通常是L2
- $\lambda_\pi = 1$, $\lambda_V = 0.1$: actor的smoothness比critic更重要

**Intuition**: 如果在$s_t$和$s_{t+1}$之间任意一点$\bar{s}_t$, policy和value function的输出应该和两端close, 这就是Lipschitz continuity的local定义。 这种regularization和Wasserstein GAN的gradient penalty思路是同源的(都强制平滑)。

**Real-world implication**: Table IV显示, w/o-L2C2在real ground上还能work (5/5), 但在slope上直接fail (slope 0/5%), 在wall上only 4/5。 也就是说sim里看不出巨大差异, 但sim-to-real一上来, 没有L2C2的版本就崩了。 这是因为real hardware对oscillation更敏感, 任何sim里small的oscillation都会被real friction和torque limit放大。

参考:
- L2C2原始paper: https://ieeexplore.ieee.org/document/9982242
- 类似的Lipschitz regularization: https://arxiv.org/abs/1801.07198

---

## 6. Reward Design - 完整的Table VI解析

这是这篇paper最复杂的部分。 Table VI把所有reward分成4个group, 每个group有独立critic:

### 6.1 Task Reward $r^{task}$ ($w^{task}=2.5$)

- **Head height**: $f_{tol}(h_{head}, [1, \inf], 1, 0.1)$ - 头部高度达到1m的目标, 用tolerance function (Gaussian fall-off)
- **Base orientation**: $f_{tol}(-\theta_{base}^z, [0.99, \inf], 1, 0.05)$ - projected gravity vector接近-1(垂直)

注意: $f_{tol}(i, b, m, v)$ 是OpenAI Spinning Up / DMC [47]里的标准tolerance reward, 在bound $b$内reward=1, 超出bound后用Gaussian衰减, 在margin $m$距离处衰减到$v$。

### 6.2 Style Reward $r^{style}$ ($w^{style}=1$)

这部分最有意思, 大量用了indicator function $\mathbb{1}(\cdot)$做hard penalty:
- **Waist yaw deviation**: $-10 \cdot \mathbb{1}(|q_{waist}| > 1.4)$ - 腰部yaw超过1.4 rad直接罚10
- **Hip roll/yaw deviation**: 类似的indicator, 防止hip joint超限
- **Knee deviation**: 地面是-0.25权重, PSW是0, 因为PSW场景knee需要更大range
- **Shoulder roll**: -10(ground), -2.5(PSW), 防止肩部外展过度
- **Foot displacement**: $\exp(-2\|q_{base}^{xy} - q_{foot}^{xy}\|^2 \cdot \text{clip}(\cdot, 0.3, \inf)) \times \mathbb{1}(h_{base} > H_{stage2})$ - 鼓励CoM在support polygon内, 受[11]启发
- **Ankle parallel**: 用keypoints高度方差, 鼓励ankle保持水平
- **Foot distance**: $\|q_{feet}^l - q_{feet}^r\|^2 > 0.9$ 罚-10, 防止两脚并在一起
- **Feet stumble**: $\mathbb{1}(\exists i, |F_i^{xy}| > 3|F_i^z|)$ 罚-25/PSW, -10/G, 防止脚打滑(horizontal force过大)
- **Shank orientation**: $f_{tol}(\text{mean}(\theta_{shank}^{l,r}[2]), [0.8, \inf], 1, 0.1) \times \mathbb{1}(h_{base} > H_{stage1})$ - 鼓励小腿垂直地面
- **Base angular velocity**: $\exp(-2\|\omega_{base}^{xy}\|^2) \times \mathbb{1}(h_{base} > H_{stage1})$ - 鼓励base角速度低

**Key insight**: style reward的核心目的是**约束起身过程中的姿态质量**, 而不是"教policy如何起身"。 这是HoST能直接sim-to-real的关键 - 这些constraints把"在sim里能做但在real里不行"的motion都禁止了。

### 6.3 Regularization Reward $r^{regu}$ ($w^{regu}=0.1$)

标准regularization, 不多讲:
- Joint acceleration, action rate, smoothness(jerk-like项), torques, joint power, joint velocity, tracking error, position limits, velocity limits

### 6.4 Post-task Reward $r^{post}$ ($w^{post}=1$)

只在$h_{base} > H_{stage2}$时激活, 即机器人已经基本站起来后:
- Base angular velocity低
- Base linear velocity低
- Base orientation正确
- Base height达到target (0.7m / 0.6m slope)
- Upper-body posture: $\exp(-0.1\|p_{upper} - p_{upper}^{target}\|^2)$ - 跟踪target姿态
- Feet parallel: $\exp(-20|h_{feet}^l - h_{feet}^r| \cdot \text{clip}(\cdot, 0.02, \inf))$ - 两脚水平

**这个group的intuition**: 起身动作完成后, 机器人需要进入"stable standing"状态, 这样后续可以被locomotion controller接管。 没有post-task reward, policy会学到"达到高度然后立刻又坐下"这种degenerate solution。

---

## 7. Domain Randomization - Table II深度解析

| 项 | 数值 | Intuition |
|---|---|---|
| Trunk Mass | U(-2, 5) kg | 真实机器人payload变化, 比如背个backpack |
| Base CoM offset | U(-0.12, 0.12)m (XY), U(-0.08, 0.08)m (Z) | sim里CoM不准, 真实机器人组装误差 |
| Link mass | U(-0.8, 1.2) × default | 整体mass scaling |
| Friction | U(0.1, 1) | 不同地面材质 |
| Restitution | U(0, 1) | 撞地反弹 |
| P Gain | U(0.85, 1.15) × nominal | PD gain不准 |
| D Gain | 同上 | |
| Torque RFI | U(-0.05, 0.05) × limit | Random Force Integral, [2]的技术 |
| Motor Strength | U(0.9, 1.1) | 电机扭矩常数偏差 |
| Control delay | U(0, 100)ms | 控制器延迟, 真实系统从50-100ms不等 |
| Initial joint angle offset | U(-0.1, 0.1) rad | 起始姿态扰动 |
| Initial joint angle scale | U(0.9, 1.1) × default | |

**Fig 9(a)的ablation特别有信息量**: CoM position的randomization是最critical的, 关掉它success rate从100%掉到~30%。 这是因为CoM offset直接决定重心位置, 而起身是重心control的任务。

参考:
- Domain Randomization原始paper: https://arxiv.org/abs/1703.06907
- RFI (Random Force Integral): https://arxiv.org/abs/2404.07093

---

## 8. Sim-to-Real Gap Analysis - Phase Plot的发现

Fig 9(b)是这篇paper最有价值的诊断工具之一。 作者把hip和knee两个关键joint的phase plot (position vs velocity)在sim和real上对比, 发现:
- joint position trajectory: sim和real相当一致
- joint velocity trajectory: **有显著discrepancy**

这个discrepancy的物理含义是: 真实电机的torque输出和sim的命令torque有差距, 这个差距在position level被PD controller的integrator平滑掉了, 但在velocity level直接体现。

**这是为什么real robot部署时需要把hip stiffness从150调到200, knee从200调到275**: sim里的PD gain已经被训练时固定, 但real的effective stiffness比sim低, 所以需要补偿。 这个insight和Hwangbo et al. (Science Robotics 2019, [15])在ANYmal上的发现是一致的: actuator model在legged robot sim-to-real中是关键瓶颈。

参考:
- Hwangbo et al.: https://www.science.org/doi/10.1126/scirobotics.aau5872

---

## 9. Experimental Results - 关键数据点

### 9.1 Simulation Main Results (Table III)

完整的HoST在4个terrain上:
- Ground: 99.5% success, smoothness 2.90, energy 1.35
- Platform: 99.8%, 3.39, 0.58
- Wall: 94.2%, 4.66, 1.08
- Slope: 98.5%, 5.31, 0.83

注意slope和wall的smoothness明显高于ground, 这是因为这些terrain需要额外的balancing动作。

### 9.2 Real Robot Experiments (Table IV)

HoST vs HoST-w/o-L2C2 in real world:
- HoST: 20/20 success, smoothness 4.01
- w/o-L2C2: 11/20 success, smoothness 6.54

w/o-L2C2在slope上完全fail (0/5), 在platform上only 2/5, 这印证了L2C2在real hardware上的criticality。

### 9.3 Robustness Tests (Fig 5)

四种disturbance:
1. CoM position offset in sagittal direction
2. Consistent sagittal force
3. Initial joint angle offset
4. Random torque dropout ratio

History=1的版本在所有disturbance下都degrade严重, 说明historical state对robustness的贡献。

### 9.4 Payload and Torque Dropout (Table V)

- Payload: 4-12kg都能3/3 success, 但smoothness从1.75升到1.85
- 12kg payload = 2x trunk mass (G1 trunk ~6kg), 这个数字相当impressive
- Torque dropout: 0.05-0.2 ratio都能work, 0.25时1/3 (这里是typo应该是2/3或者33%)

---

## 10. Emergent Properties - 让我最兴奋的部分

### 10.1 3kg球撞击恢复 (Fig 10a)

作者用3kg球撞击机器人, 机器人能在不摔倒情况下保持稳定。 这没有被显式训练过, 是policy在force curriculum下学到的"general balancing"能力的emergent property。

### 10.2 大力扰动恢复 (Fig 10b)

施加能让机器人倾斜很多但没完全倒下的force, policy能recover回来。 这意味着policy学到的不仅是"从地上站起来", 而是更general的"return to upright pose"。

### 10.3 15° slippery slope动态平衡 (Fig 10c)

15° slippery slope (低friction)上能保持平衡, 即使slip也能adjust CoM。 这种adaptivity是domain randomization带来的。

### 10.4 Prone和Side-lying Posture (Fig 12)

虽然训练时主要用supine posture, 但学到policy能handle prone和side-lying。 **但**论文也指出prone和supine一起训练会interfere, 这是个open problem。

### 10.5 Extension to H1/H1-2 (Fig 13)

HoST被扩展到Unitree H1 (19 DoF)和H1-2 (27 DoF)上。 H1-2比G1更重更大, 需要的torque更高。 作者观察到两个sim-to-real gap:
1. 需要更高的joint stiffness来compensate insufficient torque
2. upper-body posture有noticeable deviation

这部分还在exploratory stage。

---

## 11. Trade-off Analysis (Fig 6) - 重要的Engineering Insight

Fig 6展示了motion speed vs smoothness vs energy的triple trade-off:

- **Speed vs Smoothness**: inverse relationship, 起身慢的更smooth, 起身快的更jerky。 这就是为什么HoST用action bound β来"刻意限速"
- **Speed vs Energy**: 通常energy随speed增加, 但在Wall terrain上反常 - 慢动作反而更耗能, 因为push against vertical surface需要持续force。 这是个非常physically intuitive的发现

这个trade-off的practical implication是: **real-world deployable humanoid policy不能只追求speed**, 必须把smoothness作为一等公民考虑。 这就是为什么HoST的设计里β curriculum和L2C2是必须的。

---

## 12. Limitations 和 Future Work - 我的批判性思考

论文Section VIII列出的limitation:

1. **Perception缺失**: 只用proprioception, 在某些场景(如sitting on sofa)会fail, 因为需要visual来识别障碍物。 这点和HIM[28]、Humanoid Parkour[55]的perception-augmented policy比起来是弱点。 解决方向: 加depth vision + early fusion或late fusion

2. **Prone和Supine的interference**: 论文观察到同时训练prone和supine时, sampled rollouts之间interfere, 这可能是因为初始姿态太不同, 导致critic学的return bimodal。 解决方向: 用conditional policy (posture-conditioned)或MoE-style multi-head policy

3. **Integration with existing systems**: HoST目前是standalone的, 还没和locomotion controller整合。 Fall recovery完整pipeline需要: detect fall → trigger stand-up policy → handoff to locomotion。 这个handoff是个open problem, 类似work在quadruped上有[23]

我觉得还有几个未提及的limitation:

4. **No explicit contact schedule**: HoST的contact sequence是emergent的, 但在一些critical场景, 比如narrow space里起身, 可能需要explicit reasoning about contact。 解决方向: contact planner + RL controller的hybrid架构

5. **Energy efficiency不算突出**: ground上energy 1.35虽然不高, 但human起身energy要低很多。 这是因为human利用了momentum transfer和joint compliance, 而HoST主要靠brute force的PD control

6. **Upper body motion不自然**: 看videos能看出, HoST的起身动作不像human那样流畅, 有点"机械"。 这是因为style reward主要是constraint-based的, 没有reference motion。 解决方向: 用少量human motion作为soft guidance

---

## 13. 和Related Work的Positioning

### 13.1 vs Model-based Trajectory Optimization

经典方法如Kanehiro et al. [17, 18] (Hirobi robot)用预定义trajectory + Mahalanobis distance planning。 优势: 保证safety, 可解释。 劣势: 不能generalize到unseen posture, 需要fine-tune per scenario, 计算expensive。

HoST的优势: 一次训练, 多posture通用。 HoST的劣势: 没有safety guarantee, 训练不稳定。

### 13.2 vs Peng et al. ASE [36], DeepMimic [35]

ASE和DeepMimic都用predefined motion clip作为reference, 通过reward shaping让policy imitate。 优势: motion自然, 像human。 劣势: 依赖motion clip, 不能generalize到没clip的posture。

HoST的positioning: 不要clip, 但牺牲了motion naturality。 这是个fundamental trade-off。

### 13.3 vs Haarnoja et al. Soccer Robot [12]

Soccer robot用了predefined motion clip + RL fine-tune, 在bipedal上能work但限定在足球场景。 HoST是more general的stand-up。

### 13.4 vs Tao et al. Learning to Get Up [46]

Tao et al. 是sim only, 没有real robot部署。 HoST的main contribution之一是sim-to-real transfer。

### 13.5 vs Gaspard et al. FRASA [10]

FRASA是real robot部署, 但用low DoF机器人(Robotis OP3, 20 DoF但small size), 没有high DoF humanoid。 HoST在G1(23 DoF, 35kg)上validate。

### 13.6 vs Quadruped Stand-up [23, 30, 52]

Quadruped stand-up比较容易, 因为有4 leg stable support。 HoST的bipedal stand-up是更难的问题。 但quadruped里的一些技术(如curriculum、exploration strategy)被HoST借鉴了。

---

## 14. 我对这篇paper的整体评价

**Strengths**:
1. **Multi-critic for multi-stage task**这个framing很clear, 解决了single critic在complex reward下diverge的问题
2. **Force curriculum**的婴儿学习metaphor很motivating, 而且在实验上证明比RND强
3. **Action bound curriculum**这个简单trick解决了real-world deployment的大问题, 而且被Fig 6的trade-off analysis佐证
4. **L2C2在multi-critic上的integrate**是个工程亮点
5. **Real-world deployment非常extensive**: 4种terrain, outdoor, payload, disturbance, 真的很solid
6. **Phase plot分析**揭示了actuator model的sim-to-real gap, 是个honest的diagnostic

**Weaknesses**:
1. **Motion自然度有限**: 没有reference motion, motion看起来还是机器式的
2. **Perception的缺失**限制了outdoor generalization
3. **Prone和Supine同时训练的interference**没解决, 说明reward shaping还不够general
4. **Energy efficiency**和human比还有差距
5. **没有explicit的fall detection和handoff**, 整个recovery system不完整

**对field的影响预测**:
- 这篇paper会推动multi-critic RL在humanoid control里的adoption
- Action bound curriculum会成为sim-to-real transfer的standard trick
- Stand-up control会变成humanoid robot的标准capability之一
- 后续工作会在perception-augmented stand-up和full recovery pipeline上展开

**和Karpathy自己的工作的connection**:
- Karpathy在Tesla的Optimus humanoid project肯定关心stand-up recovery
- HoST的multi-critic思路可能可以apply到Tesla的multi-objective driving场景
- 但Karpathy更可能关心的是: 这种proprioception-only policy能否scale到更复杂的manipulation + locomotion组合, 这是个open question

---

## 15. 推荐的Further Reading

如果你(Karpathy)对这块儿感兴趣, 我推荐这些:

**Humanoid Locomotion RL**:
- Real-world humanoid locomotion with RL (Radosavovic et al.): https://www.science.org/doi/10.1126/scirobotics.adi8956
- HIM: https://arxiv.org/abs/2411.14386
- Humanoid Parkour: https://arxiv.org/abs/2410.03886

**Whole-body Control**:
- Expressive Whole-Body Control (Cheng et al. RSS 2024): https://expressive-humanoid.github.io
- OmniH2O: https://omni-humanoid.github.io
- HOMIE (最近的工作, 类似OmniH2O思路): https://homie-ohw2v2.github.io

**Multi-Critic RL**:
- Mysore et al. ICLR 2022: https://openreview.net/forum?id=WcCfITY2bQS
- Robot Keyframing (Zargarbashi et al. CoRL 2024): https://robotkeyframing.github.io

**Sim-to-Real**:
- Domain Randomization: https://arxiv.org/abs/1703.06907
- RMA (Rapid Motor Adaptation): https://arxiv.org/abs/2107.04034

**Actuator Modeling**:
- Hwangbo et al. Science Robotics 2019: https://www.science.org/doi/10.1126/scirobotics.aau5872

**Stand-up Related**:
- Learning to Get Up (Tao et al. SIGGRAPH 2022): https://tianxintao.github.io/projects/learning-to-get-up/
- FRASA (Gaspard et al. 2024): https://arxiv.org/abs/2410.08655

---

## 16. 最后的Intuition Building

我希望你从这篇paper带走以下几个key intuitions:

1. **Multi-critic不是decomposition trick, 而是gradient conflict mitigation**: 在multi-stage task里, single critic学的return是multi-modal的, optimization difficulty爆炸。 Multi-critic让每个critic学一个homogeneous return, 大大降低optimization难度

2. **Curriculum不是只调reward, 也可以调env force和action bound**: force curriculum给policy"撑腰"让它先学会rough trajectory, action bound curriculum后期收紧让它fine-tune到smooth motion

3. **Proprioception在contact-rich task里能encode contact history**: 通过5步历史, policy隐式infer contact sequence。 这是替代explicit contact estimation的elegant方案

4. **Smoothness regularization在sim里看不出差异, 在real上必死**: L2C2这种Lipschitz regularization是real-world deployment的必要条件, 不是可选优化

5. **Domain randomization的CoM offset是最critical的**: 因为起身是重心control task, CoM不准=task definition本身错了

6. **Phase plot是诊断sim-to-real gap的最好工具**: position level看着没问题, velocity level暴露了actuator model的缺陷

希望这些能build你的intuition。 HoST这个工作的核心贡献是把stand-up从一个"用model-based方法解决的特殊问题"变成了"用RL from scratch可以scale的general capability", 这对humanoid robot的fall recovery和broader deployment意义重大。
