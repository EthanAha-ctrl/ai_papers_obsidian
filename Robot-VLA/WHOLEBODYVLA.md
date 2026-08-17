---
source_pdf: WHOLEBODYVLA.pdf
paper_sha256: 2f8935e81f31fcfc9486907ee61ab749810884c64dfded89791e952e28603d44
processed_at: '2026-08-13T04:24:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说WholeBodyVLA

## 一句话总结

让人形机器人能"走过去、蹲下、双手抓东西、转身放到车上"这种连续动作串起来跑通，靠两招：**从廉价的人类视频里偷学动作知识**，和**把"走多快"改成"走/停/转"的简单指令**。

---

## 问题在哪

先说为什么这事难。你现在看那些robot demo——RT-2、Pi0、OpenVLA——都是在table前固定不动，arm动来动去抓东西。这相对简单，因为base不动，problem space小。

但真正有用的robot得能移动。你让它去仓库拿个箱子放到推车上，它得：走到箱子前→蹲下→双手抓→站起来→转身→放到车上。这一串动作里，**走和抓不是分开的两件事**——你走的时候就得想"我走到哪个位置抓起来最方便"，蹲的时候得想"蹲多高才能稳稳抓住"。这叫manipulation-aware locomotion，就是"为了抓东西而走"，而不是"瞎走一段再开始抓"。

现有方案两条路都走不通：

**第一条：模块化拼接**。一个module负责navigate，一个module负责grasp，中间用VLM planner切换。听起来合理，但实际中robot走完一步到了个尴尬位置——离桌子太远抓不到，或者朝向不对——下面manipulation module就傻眼了。Error accumulates，越走越歪。

**第二条：end-to-end学**。直接从vision输出whole-body joint commands。理论最优雅，但需要海量whole-body teleop data。这种数据怎么采？你得让人穿exoskeleton或者用MoCap驱动humanoid做动作，一小时数据可能要几千美元。Boston Dynamics那个Atlas demo就是靠MoCap堆出来的，贵得离谱。

所以两个bottleneck：**数据太贵**，**底层控制太烂**。

---

## 招数一：从视频里偷学

核心insight特别简单：**人类学新技能是靠看别人做的**，不是靠自己做一遍teleop。

你想想，YouTube上随便一个第一人称视角的cooking视频，里面就包含了大量信息——手怎么伸、身体怎么转、什么时候蹲下。这些视频action-free（没有action label），但visual change pattern全在里面。

Latent action learning就是把这个idea工程化。具体做法：

拿相邻两帧 $(o_t, o_{t+k})$，用一个encoder压缩成一个discrete code $c_t$，这个code代表"从frame t到frame t+k发生了什么action"。怎么训？用这个code去reconstruct第t+k帧，reconstruction loss逼codebook学到predictive的representation。这是VQ-VAE的标准套路，codebook就是一个lookup table，每个entry对应一种visual change pattern。

关键trick是**分开训两个LAM**。为什么？

Manipulation视频里相机不动，画面变化全来自arm运动，所以LAM会学会attend到arm region。

Locomotion视频里相机一直在动，画面变化来自环境相对相机的运动，所以LAM会attend到整个scene。

如果混在一起训，LAM就confused了——同样看到arm相对环境的位移，到底是arm动了还是相机动了？分不清楚，latent code就ambiguous。所以分开训：一个manipulation LAM专管arm运动，一个locomotion LAM专管camera运动。

训完LAM之后，VLA（用Prismatic-7B做backbone）去预测这两个latent code。注意是**联合预测**——同时输出 $c_t^{\text{mani}}$ 和 $c_t^{\text{loco}}$，在一个unified space里学"走和抓怎么配合"。

最后finetune阶段加一个轻量decoder，把latent code映射成robot实际动作：上半身joint angles + 下半身locomotion command。用LoRA finetune，数据是少量teleop trajectories（每task 50条）。

最impressive的result：**用100%人类视频pretrain + 25条teleop finetune ≈ 0% pretrain + 200条teleop**。8倍数据效率。这跟LLM里"pretrain on internet text, finetune on instruction data"完全一个套路——pretrain on action-free video（便宜），finetune on teleop（贵但少量）。

数据采集也极便宜：一个人戴个GoPro在头上去做各种动作——走、转、蹲、approach桌子上的东西但不用真的抓。8种motion primitive，300小时数据就来了。对比MoCap动辄几十万美元的setup，这是数量级的成本下降。

---

## 招数二：把"走多快"改成"走/停/转"

现有humanoid RL controller几乎全用velocity tracking——给一个连续velocity command $v^{\text{cmd}} \in \mathbb{R}^3$，让robot跟踪。问题：

1. **Start-stop语义是implicit的**。$v=0$ 不等于"刹车"，robot可能还在drift。你让它走到桌子前停下抓东西，它停不准，差个20cm就抓不到。
2. **不同speed下gait fragmented**。0.1m/s和0.5m/s的步态pattern完全不一样，controller学起来困难。
3. **Episode-level controllability没supervision**。刹车精度、heading fidelity这些loco-manipulation真正关心的指标，velocity tracking reward里根本没有。

LMO（Loco-Manipulation-Oriented）的核心改动：**把连续velocity换成discrete intent**。

Command变成 $u_t = [s_x, s_y, s_\psi, h^*]$，其中 $s_x, s_y, s_\psi \in \{-1, 0, 1\}$ 就是"前进/后退/不动"、"左/右/不动"、"左转/右转/不动"三个flag，$h^*$ 是stance height（用于squat）。

这有什么好处？Start-stop语义explicit了——flag从0变1就是"开始走"，从1变0就是"停"。Controller不用再猜"v=0.01是不是该停了"。

但discrete flag直接用会突变——0突然变1，加速度无穷大。所以用一个soft gating：
$$v_k^{\text{ref}}(t) = v_k^{\text{goal}} \tanh[\alpha(s_k - \bar{s}_k(t))]$$

$\bar{s}_k$ 是exponential smoothed flag，$\tanh$ 是saturating function。这样intent变化被smooth成可预测的velocity ramp，加速度有界。

Training分两stage：

**Stage I**：先学个基本gait别摔倒。Command intent非0时随机sample一个goal speed，upper body用简单pose target，joint limit逐步放宽让腿适应disturbance。

**Stage II**：精修loco-manipulation需要的precision和stability。
- Cruising speed固定，不再random sample，避免gait fragmented
- Directional accuracy reward：episode结束时penalize yaw drift，逼controller精确执行turn
- Stand-still penalty：stationary时惩罚leg action，防止站着乱动
- **Structured perturbation**：从AgiBot-World sample arm motion clip，time-warp后replay到upper body，让腿去compensate真实的inertial coupling。这比random push disturbance realistic多了——真正抓东西时手臂运动会扰动重心，腿得学会动态平衡。

Reward function一大堆项（Table 4里列了几十项），但核心就是：intent execution + posture + locomotion structure + smoothness + stability。

结果：LMO在turning上position error 0.05m，velocity-based controller是0.26m。5倍精度提升。对loco-manipulation来说，转得准才能抓得到。

---

## 为什么两个招数加起来work

单独看每个招数都不算全新——latent action learning有UniVLA、LAPA做过，discrete command interface在classical robotics里也有类似idea。但这篇paper的关键是把它们**组合成一个coherent system**解决一个具体问题。

**Latent learning解决"VLA不知道怎么规划"的问题**。从视频里学到locomotion和manipulation的prior，VLA在unified latent space里学两者怎么配合。没有这个pretrain，VLA只看到50条teleop trajectory，根本学不到"走到哪里适合抓"这种high-level prior。

**LMO解决"规划好但执行不了"的问题**。VLA说"前进0.5米然后停"，velocity-based controller可能停0.3米或0.7米，误差累积后面就崩了。LMO的discrete command让execution更faithful，VLA的规划才能真正落地。

两者接口：VLA输出discrete locomotion command $[s_x, s_y, s_\psi, h^*]$，LMO执行。这个接口设计很关键——既保留了VLA的high-level planning能力，又给LMO留出了precise execution的空间。如果VLA直接输出joint angles，就既不好planning又不好execution。

---

## 结果快报

三个task：bag packing（走+抓+蹲+放）、box loading（蹲+抓+转身+放）、cart pushing（抓+推50kg车）。

WholeBodyVLA平均78%成功率，比modular pipeline高14%，比OpenVLA-OFT高21%，比GR00T高36%。

Generalization：换初始位置、换桌子高度、换object、换scene appearance、甚至跨terrain（台阶、泡沫、碎石、人造草），都能hold住。

数据效率：pretrain让teleop data需求降8倍。

Failure analysis很诚实——主要failure来自locomotion precision不够（走偏了导致抓不到），不是catastrophic fail。说明limitation很clear，未来improving approach precision就能提升。

---

## 我觉得真正有意思的地方

**Cross-embodiment latent space**。Figure 8那个retrieval实验特别漂亮——同一个latent action code在human video和robot demo里都对应"前进"语义。说明VQ-VAE学到的是visual dynamics，不是motor commands。理论上这个latent space可以transfer到不同humanoid（Unitree H1、Figure、Optimus），虽然paper没实测，但implication很exciting。

**Pretrain-finetune paradigm确认在robotics成立**。LLM靠pretrain on internet text实现data efficiency，VLA靠pretrain on action-free video实现同样的事。这意味着robotics的data scaling可能不像想象那么难——YouTube就是最大的robot dataset。

**Discrete command interface的哲学**。其实人控制自己身体也不是连续velocity——你想"走过去"是一个discrete intent，具体步频步幅是subconscious调的。LMO的discrete interface某种程度上更接近人类motor control的abstraction level。

---

## 几个我没想明白的地方

**Codebook size没明说**。VQ-VAE的codebook collapse问题（只有少数code被激活）怎么处理？EMA trick？Restart？这个对reproduction很重要。

**Sim-to-real on heavy load**。50kg cart pushing在MuJoCo里训，domain randomization再强，真实接触 dynamics和摩擦还是有gap。Paper里success rate 22/25不错，但不知道这个gap有多critical。

**Long-horizon scaling**。现在task最多也就几个subgoal。真正long-horizon（几十步的task）时，latent action的累积误差怎么办？有没有可能引入hierarchical latent（high-level subgoal latent + low-level action latent）？

**Safety没讨论**。50kg cart推出去如果controller failure可能伤人。Real deployment需要safety layer，paper没提。

整体来说，这篇paper的contribution是**证明了一个recipe**：action-free video pretrain + discrete command interface = cheap data + precise execution for humanoid loco-manipulation。如果这个recipe能scale up到更多skill、更多embodiment，humanoid robot的data和control瓶颈可能被根本性突破。

---

# WholeBodyVLA：统一Latent空间的人形机器人全身Loco-Manipulation控制

## 1. 核心问题与Motivation

这篇paper瞄准一个特别实际的问题：**humanoid robot在大空间里的loco-manipulation**。你想想，现有的humanoid工作要么是in-place manipulation（像RT-2、OpenVLA、Pi0这些 tabletop 场景），要么是纯navigation/locomotion（像VLN那类工作），真正把"走过去 + 蹲下 + 双手抓取 + 转身放置"这种连续动作串起来在真实世界跑通的工作几乎没有。

作者识别出两个核心瓶颈：

**第一个是数据稀缺**。Teleoperation data 极贵——HOMIE 需要isomorphic exoskeleton cockpit，FALCON需要MoCap input，R2S2要MoCap input。Boston Dynamics的LBM demo也是依赖昂贵的MoCap。而humanoid的whole-body teleop数据规模远比tabletop manip小一个数量级。Open X-Embodiment、AgiBot World这些数据集主要还是固定基座的arm manip。

**第二个是low-level controller的精度问题**。现有humanoid RL controller（HOMIE、AMO、FALCON、R2S2、ULC）几乎全部用velocity-tracking objective。你让它跟踪一个连续velocity command $v^{\text{cmd}} \in \mathbb{R}^3$，但loco-manipulation需要的是precise start-stop、precise turning、stable squatting。Velocity tracking有几个问题：(1) start-stop语义是implicit的，机器人不知道什么时候该刹车；(2) 不同speed range下gait会fragment；(3) episode-level的controllability（比如刹车精度、heading fidelity）没有supervision。

这两个问题串起来——即使你high-level VLA规划得再好，low-level execution一塌糊涂，整个系统就崩了。作者在Appendix C.3里专门做了failure mode analysis，发现stumble、path deviation、turn with advance这些错误大多来自low-level controller，不是VLA本身的决策问题。

参考链接：
- HOMIE: https://humanoid-iitd.github.io/homie.github.io/
- AMO: https://amo-robot.github.io/
- FALCON: https://falcon-humanoid.github.io/
- R2S2: https://r2s2humanoid.github.io/
- Boston Dynamics Atlas LBM: https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/

---

## 2. 方法分解：三个核心模块

### 2.1 Unified Latent Action Model (LAM)

这是这篇paper最有趣的地方。核心inspiration来自Genie (Bruce et al., 2024) 和 UniVLA (Bu et al., 2025b) 的latent action learning——把action-free video通过inverse dynamics压缩成discrete latent code，然后用作pseudo-action label来监督VLA训练。

公式形式：给定连续两帧 $(o_t, o_{t+k})$，LAM encoder $\mathcal{E}_i$ 输出continuous latent vector：
$$z_t = \mathcal{E}_i(o_t, o_{t+k})$$

其中 $i \in \{\text{mani}, \text{loco}\}$ 表示manipulation或locomotion分支。然后quantize到codebook最近entry：
$$c_t^i = \arg\min_{c \in \mathcal{C}_i} \|z_t - c\|_2$$

这里 $\mathcal{C}_i$ 是第 $i$ 个LAM的codebook。Decoder $\mathcal{D}_i$ 从 $o_t$ 和 quantized $c_t$ 重建 $o_{t+k}$：
$$\hat{o}_{t+k} = \mathcal{D}_i(o_t, c_t)$$

训练loss是标准VQ-VAE objective（来自Van Den Oord et al., 2017）：
$$\mathcal{L}_{\text{LAM}} = \mathcal{L}_{\text{mse}} + \|\text{sg}[c_t] - z_t\|_2^2 + \beta \|c_t - \text{sg}[z_t]\|_2^2$$

变量解释：
- $\mathcal{L}_{\text{mse}} = \|o_{t+k} - \hat{o}_{t+k}\|_2^2$：reconstruction MSE
- $\text{sg}[\cdot]$：stop-gradient operator，防止梯度流过
- $\beta$：commitment cost，让encoder输出commit到codebook entry上
- 第一项codebook loss让codebook追上encoder，第二项commitment loss让encoder commit到codebook

**关键设计决策——为什么分两个LAM**：

作者发现如果在mixed data上train一个shared LAM，效果会suboptimal。原因是modality冲突：
- 在manipulation video里，camera是static的，image变化主要来自arm运动，所以LAM会attend到arm region
- 在locomotion video里，camera持续移动，image变化来自environment相对相机的运动，所以LAM会attend到整个scene
- 更糟糕的：在loco-manipulation video里（arm也在FOV里），arm-environment相对位置变化既可能来自arm motion也可能来自camera motion，shared LAM会产生ambiguous encoding

Table 9里的RRG（Relative Reconstruction Gain）metric直接验证了这点：
$$\text{RRG} = \frac{\text{MSE}_{\text{base}} - \text{MSE}_{\text{recon}}}{\text{MSE}_{\text{base}}}$$

其中 $\text{MSE}_{\text{base}} = \text{MSE}(o_t, o_{t+k})$ 是copy前帧作为baseline的error，$\text{MSE}_{\text{recon}} = \text{MSE}(\hat{o}_{t+k}, o_{t+k})$ 是LAM重建的error。RRG越高说明latent code越有predictive power。Separate LAM在所有task的primitives上都比shared LAM好。

### 2.2 VLA Training

LAM pretrain好之后，train VLA policy $\pi_\theta$ 联合预测manipulation和locomotion两个latent action：
$$\pi_\theta(c_t^{\text{mani}}, c_t^{\text{loco}} \mid o_t, \ell)$$

这是最大似然估计：
$$\min_\theta [-\log \pi_\theta(c_t^{\text{mani}}, c_t^{\text{loco}} \mid o_t, \ell)]$$

其中 $\ell$ 是language instruction。这个joint prediction是关键——它强迫model在一个cohesive action space里学locomotion和manipulation的interaction。

Stage III finetune时，加一个lightweight decoder $f$：
$$a_t = f(\hat{c}_t^{\text{mani}}, \hat{c}_t^{\text{loco}}, s_t)$$

其中 $s_t$ 是robot state，输出 $a_t$ 包含：(1) upper-body joint positions，(2) locomotion command。这个decoder用LoRA (Hu et al., 2022) finetune。

VLA backbone用Prismatic-7B（和OpenVLA-OFT一样），8×H100训练20,000 steps，batch size 1024。

参考链接：
- VQ-VAE: https://arxiv.org/abs/1711.00937
- DINOv2: https://arxiv.org/abs/2304.07193
- UniVLA: https://univla.github.io/
- Genie: https://arxiv.org/abs/2402.19459
- LAPA: https://arxiv.org/abs/2410.11758
- Prismatic VLM: https://arxiv.org/abs/2402.07865

### 2.3 Loco-Manipulation-Oriented (LMO) RL Policy

这是第二个核心贡献。Observation space非常精简——只用proprioceptive信息：
$$O_t = [u_t, \omega_t, \mathbf{g}_t, \mathbf{q}_t, \dot{\mathbf{q}}_t, \mathbf{a}_{t-1}]$$

变量解释：
- $u_t$：当前command（discrete flags + stance height）
- $\omega_t \in \mathbb{R}^3$：base angular velocity
- $\mathbf{g}_t \in \mathbb{R}^3$：gravity vector（IMU测的）
- $\mathbf{q}_t, \dot{\mathbf{q}}_t$：joint positions和velocities
- $\mathbf{a}_{t-1}$：previous action

**Discrete command interface**：
$$u_t = [s_x, s_y, s_\psi, h^*] \in \{-1, 0, 1\}^3 \times \mathbb{R}$$

变量：
- $s_x \in \{-1, 0, 1\}$：forward/backward intent
- $s_y \in \{-1, 0, 1\}$：lateral intent
- $s_\psi \in \{-1, 0, 1\}$：yaw intent
- $h^*$：stance height（连续值，用于squat）

这跟传统velocity tracking $v^{\text{cmd}} \in \mathbb{R}^3$ 对比起来，最大好处是explicit start-stop semantics——flag从0变到±1就是"开始"，从±1变回0就是"停止"。

**Reference shaping**——把discrete flag平滑成velocity reference：
$$v_k^{\text{ref}}(t) = v_k^{\text{goal}} \tanh[\alpha(s_k - \bar{s}_k(t))]$$
$$\bar{s}_k(t) \gets (1-\lambda)\bar{s}_k(t-1) + \lambda s_k$$

变量：
- $k \in \{x, y, \psi\}$：三个axis
- $v_k^{\text{goal}}$：goal speed magnitude（fixed）
- $\alpha$：tanh的steepness
- $\bar{s}_k(t)$：exponentially smoothed flag，$\lambda$ 是smoothing coefficient
- $\tanh$ 起到saturating nonlinearity作用，bounding acceleration

这个设计很像MPC里soft gating的思路——避免impulsive acceleration。

**Two-stage curriculum**：

**Stage I (Basic Gait Acquisition)**：
- $s_k \neq 0$ 时 sample $v_k^{\text{goal}} \sim \mathcal{U}([0, v_k^{\text{max}}])$
- Upper body用简单pose target，固定interval resample，smooth interpolation
- Joint limit curriculum factor 逐步放松，让legs适应越来越强的disturbance

**Stage II (Precision and Stability)**：
- 固定 $v_k^{\text{goal}} = \bar{v}_k$（cruising speed标准化）
- Directional accuracy reward：
$$\mathcal{T}_{\text{dir}} = |\text{wrap}(\psi_{\text{end}} - \psi_{\text{start}})|$$

  其中 $\text{wrap}(\cdot)$ 把角度wrap到 $[-\pi, \pi]$，episode从flag flip $0 \to \pm 1$ 开始，回到0且stabilize时结束。

- Stand-still penalty（stationary时惩罚leg action）：
$$\mathcal{T}_{\text{stand}} = \|a_i^{\text{leg}}\|_2^2$$

- Structured perturbation：从AgiBot-World sample arm motion clip，time-warp + noise injection
$$\omega_{i+1} = \min(L, \omega_i + (\gamma + \delta_i)\Delta t), \quad \omega_0 = 0$$
$$q_i^{\text{tar}} = q^{\text{arm}}(\omega_i) + \varepsilon_i$$

  变量：
  - $L \sim \text{Unif}[0.8, 2.5]$：clip length
  - $\gamma \sim \text{Unif}[0.8, 1.5]$：playback speed
  - $\delta_i \sim \text{Unif}[-0.25, 0.25]$：per-step speed jitter
  - $\varepsilon_i \sim \mathcal{N}(0, 0.05^2)$：joint noise

  这个structured perturbation很关键——它逼legs去compensate realistic inertial coupling，而不是random force。

**Reward function summary**（Table 4）：

总reward大致结构：
- **Intent Execution**：forward/lateral/yaw intent execution（exp kernel形式 $\exp\{-4(v - s \cdot v^{\text{goal}})^2\}$），height tracking，vertical velocity suppression
- **Posture & Joints**：roll/pitch stabilization（$\|\mathbf{g}_x\|^2 + \|\mathbf{g}_y\|^2$），hip/ankle deviation，knee squat penalty，DoF acceleration/velocity/torque limits
- **Locomotion Structure**：feet air time，foot clearance，lateral spacing，feet parallelism，no-fly penalty，foot slip，foot stumble
- **Energy & Smoothness**：action rate penalty（$\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$），2nd-order smoothness（$\|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|^2$），torque usage，stand-still penalty
- **Stability**：joint tracking error

特别提一下foot stumble penalty：$\mathbf{1}\{|F^x| > 3|F^z|\}$——当水平接触力大于3倍垂直接触力时惩罚，这是为了防止脚拖地。

Domain randomization（Table 5）覆盖：
- Joint torque injection $[-0.05, 0.05]$
- Friction $[0.1, 3.0]$
- Payload mass torso $[-5, 10]$ kg, hands $[-0.1, 0.3]$ kg
- PD gain scaling $[0.9, 1.1]$
- Push disturbances up to 0.5 m/s，每4秒一次
- DOF/IMU lag $[0, 10]$ timesteps

参考链接：
- HOMIE: https://humanoid-iitd.github.io/homie.github.io/
- ExBody2: https://arxiv.org/abs/2412.13196
- ASAP: https://asap-rl.github.io/
- OmniH2O: https://arxiv.org/abs/2406.08858

---

## 3. 系统架构与数据流

Figure 2 里的pipeline大致是：

```
Egocentric Manipulation Videos  →  Manipulation LAM (VQ-VAE + DINOv2)
Egocentric Locomotion Videos    →  Locomotion LAM (VQ-VAE + DINOv2)
                                          ↓
                          VLA (Prismatic-7B) pretrained
                          on latent action prediction
                                          ↓
                          LoRA finetune on teleoperation
                          → Action decoder f
                                          ↓
                          ┌──────────────┴──────────────┐
                          Upper-body joints          Locomotion command
                          (7-DoF × 2 arms)           $u_t = [s_x, s_y, s_\psi, h^*]$
                          @ ~10 Hz                   @ ~10 Hz
                                                          ↓
                                              LMO RL Policy @ 50 Hz
                                              (MuJoCo trained, NanoPi部署)
                                                          ↓
                                              Lower-body joint torques
```

部署时VLA在RTX 4090上跑，LMO在NanoPi上跑，用ZeroMQ over Ethernet通信。

**数据采集pipeline**（Figure 4）：单人戴head-mounted camera（RealSense D435i 或 GoPro）做8种motion primitives——advance、turn、squat等，approach potential manipulation goal但不需要真的grasp。收集了~300小时。

这个设计很巧妙——它把"teleop data is expensive"这个问题绕过去了。人类egocentric video极便宜（一个GoPro就行），又包含丰富locomotion + manipulation affordance信息。

---

## 4. 实验结果深度解析

### 4.1 Main Results (Table 2)

三个task：
1. **Bag Packing**：grasp paper bag → sidestep → squat → place in carton
2. **Box Loading**：squat → grasp box → turn → place on cart
3. **Cart Pushing**：grasp handle → push 50kg cart forward

每个task分两个subgoal，每个subgoal 25 trials。

| Method | Bag P. | Box L. | Cart P. | Avg |
|---|---|---|---|---|
| Modular Design | 22+12 / 50 | 9+9 / 50 | 22+22 / 50 | 64.0% |
| GR00T w/ LMO | 20+10 / 50 | 6+4 / 50 | 12+11 / 50 | 42.0% |
| OpenVLA-OFT w/ LMO | 19+6 / 50 | 12+12 / 50 | 22+14 / 50 | 56.7% |
| WholeBodyVLA | 23+13 / 50 | 19+17 / 50 | 23+22 / 50 | **78.0%** |

比Modular Design高14%，比OpenVLA-OFT高21.3%，比GR00T高36%。

有意思的ablation：
- **w/o LAM**：39.3%（掉了38.7%）
- **w/ manip. LAM only**：63.3%（缺了locomotion pretrain，掉了14.7%）
- **w/ shared LAM**：66.0%（mixed LAM，掉了12%）
- **w/ vel-based RL**：54.0%（掉了24%，91.7%的gap来自第二个subgoal，即locomotion部分）
- **w/o RL**：基本全fail（lower body joint直接预测太hard）

这组数据说明：
1. Unified latent learning贡献了38.7%的提升
2. 分开两个LAM比shared LAM好（但差距不大，12%）
3. Locomotion pretrain对需要移动的task非常重要
4. LMO比velocity-based RL提升24%，主要在locomotion-heavy subgoal

### 4.2 Generalization (Figure 3)

12个generalization实验分三组：

**Start-pose generalization**：
1. Distance X-axis (1.0/1.25/1.5m)
2. Distance Y-axis (±25/50/75cm)
3. Orientation (±30/45/60°)
4. Height (60/45/25cm训练；55/40/20cm unseen)

**Scene generalization**：
5. Unseen object
6. Unseen table
7. Unseen object position

**Extended tasks**：
8. Terrain traversal（步、泡沫、木板、碎石、人造草）
9. Long-horizon manipulation
10. Visual navigation（跟随地面箭头）
11. Vacuum cleaning
12. Wiping stains

Figure 3 (a) 显示locomotion generalization——pretrain用0%/25%/50%/100% human video（AgiBot World固定100%），纵轴success rate，横轴teleop finetune data量。**100% human video pretrain + 25 teleop trajectories ≈ 0% human video + 200 teleop trajectories**。这非常impressive——说明latent pretrain能把teleop data需求降低8倍。

Figure 3 (b) 显示manipulation generalization——类似的trend，AgiBot World scaling也显著减少teleop需求。

### 4.3 LMO Ablation (Table 3)

在MuJoCo里测：
- **Locomotion accuracy**：position/orientation error（forward/backward, left/right, turning）
- **Manipulation stability**：CoM Sway (CoMS)
$$\text{CoM Sway} = \sqrt{\frac{1}{T}\int_0^T \|\mathbf{c}(t) - \bar{\mathbf{c}}\|^2 dt}$$

其中 $\mathbf{c}(t) \in \mathbb{R}^2$ 是horizontal CoM trajectory，$\bar{\mathbf{c}}$ 是temporal mean。

| Method | Fwd/Bwd Pos/Quat | L/R Pos/Quat | Turn Pos/Quat | Standing CoMS | Squatting CoMS |
|---|---|---|---|---|---|
| LMO (ours) | 0.21/0.05 | 0.55/0.06 | 0.05/0.19 | 0.03 | 0.03 |
| LMO w/o Eq. 3 | 0.24/0.07 | 0.61/0.09 | 0.05/0.28 | 0.04 | 0.03 |
| LMO w/o stage 2 | 0.27/0.09 | 0.72/0.11 | 0.20/0.32 | 0.05 | 0.07 |
| LMO w/o stage 1 | 0.30/0.11 | 0.66/0.13 | 0.46/0.34 | 0.05 | 0.04 |
| Vel-based | 0.24/0.12 | 0.60/0.17 | 0.26/0.20 | 0.06 | 0.05 |

关键发现：
- 去掉Eq. 3 directional accuracy reward → turning orientation error从0.19涨到0.28（47% increase）
- 去掉Stage II → trajectory error和squatting sway都涨，turning orientation error 0.32
- 去掉Stage I → 没有stable gait foundation，turning position error 0.46（最大）
- Velocity-based policy整体差，特别是turning（0.26 vs 0.05）和lateral（0.60 vs 0.55）

### 4.4 Cross-embodiment Latent Space (Figure 8, C.5)

Figure 8 展示了一个特别漂亮的结果——用同一个latent action code retrieve human和robot的clip，发现它们语义对齐。比如latent code "Go Forward"在human video和robot demo里都对应前进片段。这说明VQ-VAE LAM学到的是embodiment-agnostic的visual change representation，而不是specific joint configuration。

这是unified latent learning能work的根本原因——latent code绑的是visual dynamics，不是motor commands。

### 4.5 State Ablation (Table 6, 7)

一个有趣的ablation：去掉proprioceptive state $s_t$ 输入到action decoder，发现性能只稍微降一点（78% → 76.7%在原始setting，64%在visual variation）。说明WholeBodyVLA主要靠visual observation，state只是辅助。

### 4.6 Execution Time (Table 8)

WholeBodyVLA在三个task上的平均时间：
- Bag Packing: 18.4s + 29.7s
- Box Loading: 16.8s + 7.6s
- Cart Pushing: 11.3s + 12.7s

比Modular Design慢一点（modular有human operator优化路径），但比GR00T (26.3s + 38.6s)快很多。

---

## 5. Failure Mode Analysis (Figure 7, C.3)

对4种primitive（advance、sidestep、squat、turn）各收集50个failure case做annotation。

关键发现：
- **Advance/Sidestep/Turn**：failure majority来自locomotion——"object/basket unreachable"（因为stance或orientation偏差），catastrophic collision或stumble较少
- **Squat**：failure更均匀分布——locomotion（incorrect final height或descent时contact）和pick/place error（arm trajectory不准或grasp alignment差）

这个分析指向未来方向——improving approach precision（尤其turning、lateral、squat）能直接减少下游manipulation failure。

---

## 6. 与相关工作的对比

### 6.1 Modular pipelines (Being-0, R2S2, HEAD, FALCON)

这些方法把navigation/manipulation分成discrete skill，用VLM planner切换。问题：
- Skill边界brittle——locomotion结束后robot可能在task-infeasible configuration
- 依赖cloud-based perception，latency高
- R2S2、FALCON、HITTER还需要MoCap input

WholeBodyVLA把locomotion和manipulation放在一个unified latent space里joint optimize，避免了skill切换的error accumulation。

### 6.2 End-to-end VLA (GR00T, Humanoid-VLA, LBM)

- GR00T N1.5: NVIDIA的humanoid foundation model，主要focus manip，locomotion弱
- Humanoid-VLA: 主要focus locomotion，没有manip
- Boston Dynamics LBM: end-to-end但需要expensive MoCap data，workspace limited

WholeBodyVLA通过unified latent learning从cheap action-free video学，scale up更容易。

### 6.3 Latent action learning (Genie, LAPA, UniVLA, IGOR)

- Genie (DeepMind): generative interactive environment，latent action是核心
- LAPA: Latent Action Pretraining from videos
- UniVLA: task-centric latent actions
- IGOR: image-goal representations

这些都是tabletop manip或navigation，没有把latent action应用到humanoid whole-body loco-manipulation。WholeBodyVLA的贡献是把这个idea扩展到locomotion和manipulation两个modality，并发现需要分开train。

参考链接：
- GR00T N1: https://arxiv.org/abs/2503.14734
- Being-0: https://arxiv.org/abs/2503.12533
- Humanoid-VLA: https://arxiv.org/abs/2502.14795
- IGOR: https://arxiv.org/abs/2411.00785
- LAPA: https://arxiv.org/abs/2410.11758
- Pi0: https://arxiv.org/abs/2410.24164
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/

---

## 7. 我的几点思考

### 7.1 Unified latent learning为什么work

核心insight：**latent code绑的是visual dynamics，不是motor commands**。这意味着同一个"前进"latent code在human video和robot demo里都对应前进visual pattern，尽管human和robot的joint configuration完全不同。

这种cross-embodiment transfer在tabletop manip里已经被UniVLA、LAPA验证，但这篇paper第一次把它扩展到whole-body loco-manipulation。关键问题是camera motion和arm motion的confounding——manipulation LAM里camera static，locomotion LAM里camera moving，分开train避免这个ambiguity。

### 7.2 LMO的discrete command interface为什么better

Velocity tracking有几个fundamental问题：
1. **Implicit start-stop**：$v=0$ 不等于"刹车"，robot可能drift
2. **Gait fragmentation**：不同speed regime下gait pattern不一致
3. **No episode-level supervision**：刹车精度、heading fidelity没有reward

Discrete command $\{-1, 0, 1\}^3$ 直接encode "start/stop/方向"，配合Stage II的directional accuracy reward $\mathcal{T}_{\text{dir}}$ 和stand-still penalty $\mathcal{T}_{\text{stand}}$，让controller学到precise regulation而不是continuous tracking。

这个思路跟classical control里的"hybrid systems"或"mode-switching controllers"很像——discrete mode切换 + continuous regulation。

### 7.3 数据efficiency的implication

Figure 3 (a) 的数据scaling曲线特别有启示意义：100% human video pretrain + 25 teleop trajectories ≈ 0% pretrain + 200 teleop trajectories。8倍数据efficiency。

这暗示未来humanoid robot的data collection可能主要靠action-free video（极其便宜），teleop data只是finetune用少量。这跟LLM里"pretrain on internet text, finetune on instruction data"的recipe完全parallel——pretrain on action-free video, finetune on teleop trajectories。

### 7.4 局限和未来方向

作者承认的限制：
- Long-horizon dexterous tasks还handle不好
- 没有lightweight mapping和memory for extended planning
- 没有active perception for cluttered/dynamic environments

我觉得还有几个潜在问题：
1. **Sim-to-real gap**：LMO在MuJoCo里train，虽然domain randomization很heavy，但50kg cart pushing这种heavy load场景sim-to-real可能still有gap
2. **Latent action的interpretability**：discrete codebook大小没明确说，codebook collapse问题（VQ-VAE经典问题）怎么处理？
3. **Multi-embodiment**：现在只在AgiBot X2上验证，能不能transfer到其他humanoid（Unitree H1、Tesla Optimus、Figure 02）？Figure 8的cross-embodiment retrieval暗示了可能性但没实测
4. **Safety**：50kg cart pushing如果control failure可能伤人，没有safety layer讨论
5. **Closed-loop latency**：VLA 10Hz + LMO 50Hz，在dynamic environment可能不够fast

参考链接：
- Unitree H1: https://www.unitree.com/h1
- Figure 02: https://www.figure.ai/
- Tesla Optimus: https://www.tesla.com/we-robot

---

## 8. 总结

WholeBodyVLA这篇paper的核心贡献是把**unified latent learning**（从action-free video学）和**LMO RL policy**（discrete command interface）结合起来，解决humanoid loco-manipulation的data scarcity和execution precision两个问题。

技术细节上：
- 两个separate LAM（manipulation + locomotion）避免modality conflict
- VLA联合预测 $c_t^{\text{mani}}, c_t^{\text{loco}}$ 在unified latent space
- LMO用 $\{-1, 0, 1\}^3 \times \mathbb{R}$ discrete command + 2-stage curriculum + structured perturbation
- 8倍teleop data efficiency提升
- 真实世界AgiBot X2上78% avg success rate

我觉得这篇paper最大的价值是验证了**"action-free video pretrain + teleop finetune"**这个recipe在humanoid whole-body control上也能work，跟LLM/VLA的pretrain-finetune paradigm完全一致。如果这个方向scale up成功，未来humanoid robot的data acquisition瓶颈可能被彻底打破。

项目主页：https://opendrivelab.com/WholeBodyVLA
AgiBot World: https://www.agibot.com/en/agibot-world
OpenDriveLab: https://opendrivelab.com/

---

如果让我预测下一步——我猜2026年会有几个group把这个framework扩展到：
1. **更diverse的loco-manipulation skill**（爬楼梯、跨越障碍、双手coordination的whole-body task）
2. **Multi-embodiment latent space**（一个pretrain model transfer到多种humanoid）
3. **Active perception**（VLA主动控制camera视角，类似RT-2的active perception）
4. **Hierarchical latent**（high-level latent规划subgoal，low-level latent执行，类似Diffusion Policy的hierarchical结构）
5. **Sim-to-real的latent action alignment**（用latent action作为sim和real的common interface，bridge sim-to-real gap）

整体来说，这篇paper在humanoid loco-manipulation领域是一个milestone——它把latent action learning这个tabletop manip的成熟技术扩展到whole-body，并用discrete command interface解决了low-level controller的precision问题。期待看到后续scale up的工作。
