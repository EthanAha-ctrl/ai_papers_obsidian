---
source_pdf: Human-Level Whole-Body Reactions for Intense Contact-Rich Environments.pdf
paper_sha256: ee1972cbae34b77818814ac2fbe8a72ca92853fb51059b19d6b016a0d36f5358
processed_at: '2026-08-05T07:46:47-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Thor 用人话讲

Andrej, 好, 我换个讲法, 就当咱俩 coffee chat, 我把这个 paper 从头到尾用大白话给你捋一遍, 关键地方还是把 math 摆出来, 但重点说 "为什么这么做"。

---

## 这篇 paper 到底在解决啥问题

想象你让一个 humanoid robot 去开一扇很紧的 fire door, 或者拉一个装了 70 kg 负载的小推车。Robot 得一边保持不摔, 一边使出很大的劲。这事难在哪?

**难在 robot 一使劲就摔。**

传统 control 方法 (model-based) 的思路是: 我精确知道 robot 每个关节的 mass / inertia / friction, 我用 QP solver 算出每个 joint 该输出多少 torque, 同时保证 Zero Moment Point (ZMP) 在脚底下那个 support polygon 里。这听着很美, 问题是:
- 真实世界 friction 你不知道, 门把手阻力你不知道
- 你得装 force sensor 去测 external force, 但很多场景装不了
- 一旦环境稍微 unstructured, model 就崩

RL 方法的思路是: 让 robot 在仿真里自己试, 试出个好 policy。问题是 humanoid DoF 太多 (G1 有 29 个), 单个 neural network 学 29 维输出 + 高频观测, critic 的 variance 爆炸, 训不出来, 或者训出来也很弱。

所以之前的 humanoid RL (Falcon, HOMIE, Unitree default policy) 都偷偷加了一个假设: **robot 的 center of mass (CoM) 投影必须在脚的 support polygon 里面**。这等于给自己上了一个 ceiling, 因为:

> 你自己搬过重物吧? 你搬冰箱的时候你是站直了搬, 还是 lean back 用身体 weight 抗? 肯定 lean back。Lean back 的时候你的 CoM 是在两脚之外的。

所有之前的 humanoid RL 都在 "站直了拉", 所以拉力上不去。Thor 说: 凭啥必须站直? 我要让 robot 学会 lean back, 把 CoM 弄到 support polygon 外面去, 用 body weight 当 leverage, 这样才能拉出大劲。

这就是 paper 的核心 thesis, 一句话总结:

> **让 humanoid 学会像人一样, 拉重物时主动倾斜身体, 让 CoM 跑到脚外面去, 用体重当 counterweight。**

---

## 他们的两个 core idea

### Idea 1: 把 robot 拆成三段训练 (Decoupled Policy)

之前的工作要么用一个 policy 控全身 (维度爆炸), 要么拆成 upper body + lower body 两段。Thor 说不够, 得拆成三段: **upper body / waist / lower body**, 各自有独立的 actor-critic network。

为什么 waist 单独拎出来?

你想想 tug-of-war (拔河), 力是怎么传导的: 脚踩地产生 friction → 膝盖髋关节 → **腰** → 肩膀 → 手。腰是 force chain 的 bottleneck。如果你的 policy 把腰归到 lower body 里, locomotion 的 reward 会把腰压得死死的, 腰不敢动; 如果把腰归到 upper body 里, manipulation 的 reward 又会把腰带歪。

腰得独立, 有自己的 reward, 自己的 critic, 自己的 "想法", 才能在中间当好 transmission hub。

具体数学:

整个 policy 拆成
$$\pi = [\pi_l, \pi_w, \pi_u]$$

- $\pi_l$: lower body policy (12 DoF, 两条腿)
- $\pi_w$: waist policy (3 DoF, 腰)
- $\pi_u$: upper body policy (14 DoF, 两条胳膊 + 头颈)

每个 sub-agent 有自己的参数 $\theta^i, \phi^i$, 其中 $i \in \{l, w, u\}$。三个 agent 共享同一个 observation (全身状态), 但各自算自己的 advantage, 各自更新自己的参数。

用 PPO, 每个 agent 独立的 clipped objective:

$$L_i^{CLIP}(\theta^i) = \mathbb{E}_t\left[\min\left(r_t^i \hat{A}_t^i, \text{clip}(r_t^i, 1 \pm \epsilon^i)\hat{A}_t^i\right)\right]$$

变量解释:
- $r_t^i = \frac{\pi_{\theta^i}(a_t^i|s_t)}{\pi_{\theta^i_{old}}(a_t^i|s_t)}$: importance sampling ratio, 新 policy 和旧 policy 在同一 state-action 下的概率比
- $\hat{A}_t^i$: 这个 agent 自己的 advantage estimate, 只用这个 agent 自己的 reward 算
- $\epsilon^i = 0.15$: clip ratio, 比 OpenAI 默认的 0.2 紧一点, 说明他们想更保守地更新 (三个 agent 互相耦合, 太激进的更新容易 destabilize)

三个 agent 的总 loss:
$$\mathcal{L}_{total} = \sum_{i \in \{l,w,u\}} \mathcal{L}_i + \lambda_c \mathcal{C}(a_t^i)$$

$$\mathcal{C}(a_t^i) = \frac{1}{T}\sum_{t=1}^{T}\left(\|a_t^l\|_2^2 + \|a_t^w\|_2^2 + \|a_t^u\|__2^2\right)$$

这个 $\mathcal{C}$ 是 torque regularization, 防止三个 agent 各自拼命输出 torque 互相打架。$\lambda_c$ 是唯一把它们 "soft 耦合" 在一起的 term。没有它, 三个 agent 会各自为政, robot 就抽风了。

**这里有个很重要的但 paper 没强调的 design choice**: critic 多吃了一堆 privileged 信息 $\mathcal{P}_t = (v_t, o_t, F_t)$ —— linear velocity, orientation, external force。Actor 不吃这些 (deploy 时测不到 force)。这是 asymmetric actor-critic, 参考 [ANYmal 论文 Hwangbo 2019](https://arxiv.org/abs/1901.08652) 的套路。但 Thor 没做 explicit distillation, 直接让 actor 从 proprioception (torso 角度 + 角速度) 隐式推断 external force。这有个 risk: 静态 load (没角速度信号) 时 actor 可能瞎。Paper 没测试这种 edge case。

[Multi-Agent DRL Survey (Gronauer 2022)](https://link.springer.com/article/10.1007/s10462-020-09927-w) 讲了为什么 MA-RL 比 single-agent 在 high-dim 下更 sample efficient。

---

### Idea 2: FAT2 Reward —— 教 robot 学会 "lean back"

这是 paper 真正的核心创新。FAT2 = Force-Adaptive Torso-Tilt。

**Intuition**: 你拉重物时, 拉力越大, 你应该 lean 得越多。这个 reward 就是告诉 policy: "当前 external force 是 $F_h$, 你应该 lean $\beta$ 度, 你实际 lean $\beta'$ 度, 差越远 reward 越低。"

那 $\beta$ 怎么算? 用 ZMP 推。

#### ZMP 推导 (简化版)

把 robot 当 rigid body, quasi-static 假设 (没大加速度)。Robot 受四个力:
- $\vec{F_s}$: 地面支持力 (vertical, 向上)
- $\vec{F_f}$: 脚底摩擦 (horizontal)
- $\vec{F_h}$: 手上的拉力 (与地面夹角 $\alpha$)
- $\vec{F_g}$: 重力 (在 CoM 上, 向下)

Force balance:
$$\vec{F_s} + \vec{F_f} + \vec{F_h} + \vec{F_g} = \vec{0}$$

Torque balance about support polygon centroid:
$$\vec{r}_{CoM} \times \vec{F_g} + \vec{r_h} \times \vec{F_h} = \vec{0}$$

($\vec{F_s}, \vec{F_f}$ 通过 support center, moment arm = 0, 不贡献 torque)

化简后 (公式 16):
$$|F_h| |r_h| \cos\varphi \cos\alpha = |F_g| |r_{CoM}| \cos\beta$$

变量解释:
- $|F_h|$: 拉力大小
- $|r_h|$: hand 到 support center 的距离
- $\varphi$: $\vec{r_h}$ 与 vertical 的夹角
- $\alpha$: 拉力与地面夹角
- $|F_g|$: 重力大小 = $mg$
- $|r_{CoM}|$: CoM 到 support center 距离
- $\beta$: **torso 倾斜角** (我们要 solve 的)

求解 $\beta$ (公式 17):
$$\beta = \cos^{-1}\frac{|F_h||r_h|\cos\varphi\cos\alpha}{|F_g||r_{CoM}|} \leq \beta^{max}$$

$\beta^{max} = 0.9$ rad ≈ 51.6°, 这是 hand-tuned 上界。没有它, policy 会过度倾斜把自己摔了。

**为什么 backward pull > forward pull**: 看 $\cos\beta$ 这项。Backward pull 时 robot lean back, $\beta$ 增大, $\cos\beta$ 减小, 公式右边能容纳更大的 $|F_h|$。所以 backward 允许更大的 pull force。

最大可输出拉力 (公式 19):
$$|F_h^{max}| = \frac{|F_g||r_{CoM}|\cos\beta^{max}}{|r_h|\cos\varphi\cos\alpha}$$

这个公式说明: 拉 || 的 ceiling 是 body weight × CoM-to-foot 距离 ÷ hand-to-foot 距离。Body weight 越大, 越能拉 (这就是为什么拔河比赛体重大的人占便宜)。

#### Reward 形式 (公式 18)

$$R_{FAT2} = \exp\left(-\frac{\|\beta - \beta'\|^2}{\sigma_t}\right)$$

- $\beta$: 公式 (17) 算出来的 target tilt
- $\beta'$: robot 实际的 tilt (从 proprioception 读)
- $\sigma_t$: temperature, 控制 reward 的 sharpness (越小越严格)

这是高斯形 reward, smooth gradient, 训练友好。

**对比之前 humanoid RL 的 reward**: Falcon / HOMIE 都在 reward 里加 "stay upright" (penalize torso tilt), Thor 反过来: "tilt 这个 angle 才给 reward"。这是把 human intuition 显式编进 reward 里。

参考: [ZMP 原始论文 Vukobratović 1972](https://link.springer.com/article/10.1007/BF02557668), [3D LIPM Kajita 2001](https://ieeexplore.ieee.org/document/976094) — 这是 humanoid control 的奠基 model, 假设 CoM 永远在 support polygon 上方。Thor 实际上是 **learn to violate LIPM assumption**, 这是 paradigm shift。

---

## 训练流程

### Two-Stage Curriculum

- **Stage 1** (~3.4 小时): 低 force disturbance, 让 robot 学会稳定 walking
- **Stage 2** (~3.4 小时): 极端 force disturbance, 让 robot 学会 force adaptation

Isaac Gym 仿真, RTX 4090, 总共 ~7 小时训完。

### Domain Randomization

External force 的方向和 magnitude 加 Gaussian noise, 模拟真实世界的不确定性 (门把手 friction 不一样, 地面粗糙度不一样)。这是 sim-to-real 必须。

### Upper-body Imitation from AMASS

Training 时, upper body 的 target motion 从 [AMASS dataset](https://amass.is.tue.mpg.de/) 随机采样 human mocap 数据。这让 $\pi_u$ 学会 track arbitrary human upper-body motion, deploy 时才能配合 VR teleoperation (operator 戴 VR, IK 重定向后输出 upper body target) 做 manipulation。

注意: 这是 **diversity-driven imitation**, 不是过拟合到单一 task motion。这跟 [DeepMimic (Peng 2018)](https://arxiv.org/abs/1804.02717) 一脉相承, 但 Thor 不要求 upper body "match reference motion exactly", 而是 "track distribution of human motions", 这样 policy 学到的是 "human-like upper body behavior" 而非 "perform specific motion"。

### Deployment 频率分层

- RL policy inference: **50 Hz** (三个 actor network 并行)
- PD controller: **500 Hz**

Actor 输出 desired joint positions $q_{des}$, PD controller 转 torque:
$$\tau = K_p(q_{des} - q) + K_d(\dot{q}_{des} - \dot{q})$$

- $\tau$: 输出 torque
- $K_p, K_d$: PD gains (positive)
- $q, \dot{q}$: 实际 joint position 和 velocity

高频 PD 提供 low-level stability (motor level), RL policy 做 high-level adaptation (whole-body coordination)。这是 humanoid 标准 setup, 参考 [Walk These Ways (Margolis 2023)](https://arxiv.org/abs/2212.03252)。

---

## 实验结果

### Peak Pulling Force (Table II)

G1 体重 35 kg, 重力 ~343 N。

| Method | $F_{db}^{180°}$ (N) | $F_{df}^{0°}$ (N) | vs Falcon |
|--------|---------------------|-------------------|-----------|
| **Thor** | **167.7 ± 2.4** | **145.5 ± 2.0** | +68.9% / +74.7% |
| Falcon | 99.3 ± 1.3 | 83.3 ± 2.7 | baseline |
| HOMIE | 62.3 ± 3.7 | 48.1 ± 2.9 | — |
| Unitree Default | 59.2 ± 1.7 | 68.9 ± 4.0 | — |

Thor 双手 backward pull 达 **167.7 N ≈ 48% body weight**, 比 SOTA (Falcon) 高 69%。这个数字放在 humanoid RL 里是碾压级。

命名解读:
- $F_{db}^{180°}$: dual-hand, backward locomotion, 180° 方向 (正后方)
- $F_{df}^{0°}$: dual-hand, forward locomotion, 0° 方向 (正前方)
- $F_{sp}^{180°}$: single-hand, stand in place, 180° 方向

**关键观察**:
- Backward > Forward: 因为 lean back 可以让 body weight 直接 leverage
- Dual-hand >> Single-hand: 单手拉时 robot 还要 counter torsional moment about vertical axis, force budget 被分散 ($F_{sp}^{180°}$ = 58.5 N vs $F_{dp}^{180°}$ = 127.4 N, 差 2.2x)

### Ablation (回答 Q3)

- **Thor¹** (only FAT2, no decouple): ~80-90% of Thor, 部分 task 持平
- **Thor²** (only decouple, no FAT2): 显著低于 Thor¹

结论: **FAT2 是主力, decouple 是 stabilizer**。

但有个 hidden detail: Thor¹ 在大 force 时 waist 出现 anomalous roll-angle deviation, decouple 解决这个问题。这说明 **decouple 的主要价值不在 raw force 提升, 而在 coordination 的 cleanness**。这跟 [Multi-Agent RL known issue](https://arxiv.org/abs/2011.09993) 一致 — high-dim 单 policy 在某些维度上会 "outsource" 给无关 joint, 导致 weird behavior。

### Real-World Tasks

- **Fire door**: 单手 pull 60 N, backward。Falcon / HOMIE / Default 都失败 (lateral deviation)
- **Loaded rack (70 kg)**: 130 N pull, 成功
- **Wheelchair (60 kg robot inside)**: push + turn
- **Whiteboard wiping**: 低 force 但需要 smooth whole-body coordination

为了 friction 一致性, 他们设计了 custom shoe cover 增加脚底 friction coefficient。这是个 real-world trick, paper 一笔带过但很重要 — 没有 friction guarantee, FAT2 的 lean-back 策略可能直接打滑。

---

## 跟其他工作的关系

### Falcon (前 SOTA)

[Falcon (Zhang 2025)](https://arxiv.org/abs/2505.06776) 是第一个 explicitly 做 force-adaptive humanoid loco-manip 的 RL work。它的做法是 training 时 gradually 增加 external force on EE, 让 policy 学会 resist force。但 Falcon 假设 CoM 留在 support polygon 内。Thor 打破这个假设, 允许 CoM 出去。这导致 Thor 在所有 task 上都碾压 Falcon。

### HOMIE

[HOMIE (Ben 2025)](https://arxiv.org/abs/2502.13013) 用 exoskeleton cockpit 做 teleoperation, lower body 用 RL, upper body 用 exoskeleton 直接驱动。Thor 跟它的区别是: Thor 全身都是 RL, upper body 只在 training 时用 AMASS motion, deploy 时用 VR IK 重定向。

### HOVER / OmniH2O / Exbody / ASAP

这一系列 work 都是 humanoid WBC, 但 focus 不同:
- [HOVER (He 2025)](https://arxiv.org/abs/2410.21229): multimodal policy distillation, 把多种 control mode 融合
- [OmniH2O (He 2024)](https://arxiv.org/abs/2406.08858): universal teleoperation
- [Exbody2 (Ji 2025)](https://arxiv.org/abs/2412.13196): expressive WBC
- [ASAP (He 2025)](https://arxiv.org/abs/2504.10343): sim-real physics alignment

这些 work 都没 explicit 处理 high-intensity force interaction。Thor 填了这个空白。

### Facet (Impedance + RL)

[Facet (Xu 2025)](https://arxiv.org/abs/2505.06883) 用 RL 学一个 virtual mass-spring-damper system, 提供 controllable compliance。Thor 走的是不同路线: 不学 impedance parameter, 直接学 whole-body posture response。两条路线未来可能 merge — 用 FAT2 学 nominal posture, 然后叠 impedance layer 做 fine force control。

### Biomechanics Prior

[Tug-of-war biomechanics (Cayero 2022)](https://www.mdpi.com/1660-4601/19/1/3) 是 paper 引用的 reference, 研究 human tug-of-war 里 lean angle 与 team pulling force 的 correlation。Thor 把这个 expert knowledge 编进 reward, 等于 **RL 用 biomechanics prior 做 inductive bias**。这跟 [DeepMimic](https://arxiv.org/abs/1804.02717) 用 human mocap 做 reward 一脉相承, 只是 reward signal 从 "match this motion" 变成 "match this posture given this force"。

---

## Limitations & 我的想法

1. **Hyperparameter sensitivity**: paper 最后说 entropy / LR / reward scale 都要 manual tune, particularly 因为三个 agent 互相 depend。MA-RL 训练 instability 是 known issue。Future work 他们说想从 human demo video 学 expert knowledge auto-tune, 这其实是想走 [VPT (Video PreTraining, Baker 2022)](https://arxiv.org/abs/2210.02071) 路线给 humanoid RL 提供 reward prior。

2. **No explicit force sensing at deploy**: 完全靠 torso proprioception (角度 + 角速度) 推断 external force。**Static load (没角速度信号) 时 actor 可能瞎**。Paper 没测这种 edge case。理论上应该补一个 force sensor 或者做 teacher-student distillation, 把 privileged critic 的 force sensing 蒸馏到 actor 里。

3. **$\beta^{max} = 0.9$ rad 是 hand-tuned**: 这个 magic number 决定了 robot 敢 lean 多远。理想情况下应该是个 function of foot geometry + friction coefficient + ankle torque limit。Paper 里没讨论这个 sensitivity。

4. **Quasi-static 假设**: FAT2 公式假设 no significant acceleration。Dynamic scenario (eg. 突然 force jerk, 或者 robot 在 running 时拉) 公式 (17) 可能不准。RL 可以 compensates by exploration, 但 dynamic 版 FAT2 应该基于 angular momentum 而非 ZMP。

5. **Single-EE focus**: 整个 FAT2 推导是 single pulling force。Bimanual pushing / carrying symmetric load, 或者非对称 load (一边重一边轻), 公式需要扩展成 vector torque balance。

6. **Bipedal-only**: 这个 framework 对 quadruped 不直接适用, 因为 quadruped support polygon 大很多, lean 的几何完全不同。但 idea (CoM-outside-support + body-weight-leverage) 可以 transfer。

---

## 给你的 Intuition

我看完这个 paper 之后有几个 mental model:

**(1) Reward design > network architecture**。Ablation 显示 FAT2 贡献 80-90%, decoupling 是 secondary。这说明在 humanoid RL 里, **"告诉 policy 什么是 good posture"** 比 "给 policy 更好的 network" 重要得多。这跟 [Intrinsic Reward in RL](https://arxiv.org/abs/1906.07965) 的发现一致 — RL 的 sample efficiency bottleneck 通常是 credit assignment, 不是 capacity。

**(2) Physics prior in reward is huge**。FAT2 用 ZMP 几何推导出一个 target posture, 然后让 RL 朝这个 target 收敛。这比 "let RL figure it out" sample efficient 多了, 又比 "hard constrain with QP" 灵活。这是个 sweet spot — **soft physics prior**。比 [Differentiable Physics](https://arxiv.org/abs/2106.07340) 优雅, 只 inform reward, 不 constrain policy space。

**(3) ZMP-as-target 而非 ZMP-as-constraint**。这是 paper 最深的 insight。Model-based control 用 ZMP 当 hard constraint (QP solver 里), RL 用 ZMP 当 reward target —— 后者允许 occasional violation 以换取 higher force output。这种 "soft physics prior" 是 learn-to-violate-LIPM 的关键, 也是走向 truly human-level athleticism 的必经之路。

**(4) Decoupling 的真实价值在 coordination cleanliness, 不在 raw performance**。单 decoupling (Thor²) 性能反而不如单 FAT2 (Thor¹)。但 Thor¹ 在 high force 时 waist 会 misbehave。所以 decouple 是 "safety / cleanliness layer", FAT2 是 "performance layer"。这种 division of labor 是个 design pattern, 在其他 high-dim control 问题里应该也适用。

**(5) Implicit force sensing via proprioception works, but with caveats**。Robot 没 force sensor, 靠 torso 角度 + 角速度推 external force。这 validate 了 [Berniker & Kuo](https://arxiv.org/abs/1902.02446) 一类 work 里 "proprioception sufficient for adaptive control" 的论点。但 zero-velocity static load 时可能盲, 是个 risk。

**(6) "Learn to violate assumptions" 是 humanoid RL 的下一个 frontier**。LIPM 假设 CoM 在 support polygon 上方, 这是 20 年 humanoid control 的 cornerstone。Thor 第一个 systemically break 这个假设并 outperform。这暗示 humanoid RL 的下一波进展会来自 "learn to violate simplifying assumptions we made for tractability" — 比如学习 impact (碰撞时瞬间 CoP 不连续), 学习 sliding (脚打滑时 ZMP 完全失效), 学习 flight phase (跳跃时单脚 / 双脚离地)。

---

## Reference Links

- [Thor Project Page](https://baai-aether.github.io/baai-thor/)
- [AMASS Dataset](https://amass.is.tue.mpg.de/)
- [PPO Paper (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- [Falcon (Zhang 2025)](https://arxiv.org/abs/2505.06776)
- [HOMIE (Ben 2025)](https://arxiv.org/abs/2502.13013)
- [HOVER (He 2025)](https://arxiv.org/abs/2410.21229)
- [OmniH2O (He 2024)](https://arxiv.org/abs/2406.08858)
- [Exbody2 (Ji 2025)](https://arxiv.org/abs/2412.13196)
- [ASAP (He 2025)](https://arxiv.org/abs/2504.10343)
- [Facet — Impedance + RL (Xu 2025)](https://arxiv.org/abs/2505.06883)
- [Unitree G1](https://www.unitree.com/g1/)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [3D LIPM (Kajita 2001)](https://ieeexplore.ieee.org/document/976094)
- [ZMP original (Vukobratović 1972)](https://link.springer.com/article/10.1007/BF02557668)
- [ANYmal — Learning Agile Motor Skills (Hwangbo 2019)](https://arxiv.org/abs/1901.08652)
- [RMA (Kumar 2021)](https://arxiv.org/abs/2107.04034)
- [DeepMimic (Peng 2018)](https://arxiv.org/abs/1804.02717)
- [Walk These Ways (Margolis 2023)](https://arxiv.org/abs/2212.03252)
- [Multi-Agent DRL Survey (Gronauer 2022)](https://link.springer.com/article/10.1007/s10462-020-09927-w)
- [Tug-of-war biomechanics (Cayero 2022)](https://www.mdpi.com/1660-4601/19/1/3)
- [Low back biomechanics during deadlift](https://www.tandfonline.com/doi/abs/10.1080/24725832.2021.2017924)
- [Mobile-TeleVision (Lu 2025)](https://arxiv.org/abs/2502.06457)
- [AMO (Li 2025)](https://arxiv.org/abs/2505.03738)
- [HumanPlus (Fu 2024)](https://arxiv.org/abs/2406.10454)
- [Visual Imitation for Humanoid Control (Allshire 2025)](https://arxiv.org/abs/2505.03729)
- [Clone (Li 2025)](https://arxiv.org/abs/2506.08931)
- [Twist (Ze 2025)](https://arxiv.org/abs/2505.02833)
- [BeyondMimic (Liao 2025)](https://arxiv.org/abs/2508.08241)
- [VPT — Video PreTraining (Baker 2022)](https://arxiv.org/abs/2210.02071)
- [Differentiable Physics Survey](https://arxiv.org/abs/2106.07340)
- [Intrinsic Reward in RL](https://arxiv.org/abs/1906.07965)

---

## 一句话总结

> Thor 把 humanoid RL 从 "stay upright and resist force" 的 paradigm 推进到 "adaptively lean to maximize force", 关键 trick 是用 ZMP 几何推导出一个 target tilt angle, 把它当 reward target 而非 hard constraint。这让 robot 第一次在 RL framework 里 systemically violate LIPM 假设, 拉力提升 69-75%, 真正接近 human-level 的 force-interaction 能力。

希望这个 "人话版" 帮你 build intuition 了。

---

# Thor: Humanoid Whole-Body Reactions for Contact-Rich Environments — 深度解析

Andrej, 这篇 paper 我读了之后觉得有几个非常 elegant 的设计点。我从 intuition 角度把它拆解给你看,顺便把相关的 lineage 都串起来。

---

## 1. The Core Problem They're Attacking

Unitree G1 是一个 29 DoF、35 kg、1.32 m 的 humanoid。它的 base challenge 是: **做"开门、拉车、推轮椅"这种 high-intensity force-interaction 任务时,怎么不摔。**

传统方法 (model-based) 卡在需要精确 model + 显式 external force measurement,部署受限。RL 方法卡在:
- **High-dimensionality**: 29 DoF + 高频观测 → 单 policy 的 critic variance 爆炸
- **CoM-in-support-polygon 假设**: 几乎所有 baseline (Falcon, HOMIE, Unitree default) 都假设 CoM 投影在 support polygon 内。这等于 self-imposed ceiling——human 拉重物时恰恰会 **lean back 让 CoM 出 support polygon**,用 body weight 当 counter-leverage。

Thor 的两个核心 contribution 直接对应这两个 bottleneck:
1. **Decoupled three-agent actor-critic** (upper / waist / lower)
2. **FAT2 reward**: explicitly 鼓励 CoM 出 support polygon 来 generate 更大的 interaction force

---

## 2. Decoupled Architecture — 为什么是 三段 而不是 两段

这是这个 paper 最有意思的设计选择。前面的 work 基本都是 upper/lower 两段 split (Falcon, HOMIE, AMO, Mobile-TeleVision)。Thor 加了 **waist as independent agent**。

### 2.1 Waist 单独拎出来的 physical intuition

Human biomechanics 里,tug-of-war、deadlift、carrying 这些任务的关键不在于胳膊多有劲,而在于 **waist 是 force chain 的 bottleneck**: ground friction → ankle/knee/hip → **waist** → shoulder → hand EE。Waist 的姿态直接决定了:
- 力的 transmission efficiency
- Upper body 相对 lower body 的 moment arm
- 能否把 body weight 投到 EE 反方向

如果 waist 没独立 policy,它要么 stuck 在 lower body 里 (locmotion priority 把它压死),要么 stuck 在 upper body 里 (manipulation priority 把它带歪)。Thor 把它拎出来给独立 reward 和独立 critic。

### 2.2 数学表述

整体 policy 分解:
$$\pi = [\pi_l, \pi_w, \pi_u]$$
其中 $i \in \mathcal{I} = \{l, w, u\}$ 各自是 lower / waist / upper。每个 sub-agent 独立的 Actor $\pi_{\theta^i}(a^i | s)$ 和 Critic $V_{\phi^i}(s)$。

**Shared observation** 但 **independent parameters**, 这是经典的 **CTDE (Centralized Training, Decentralized Execution)** paradigm [Multi-agent DRL survey](https://link.springer.com/article/10.1007/s10462-020-09927-w)。

TD-residual 和 GAE 每个 agent 独立计算:

$$\delta_t^i = r_t^i + \gamma V_{\phi^i}(s_{t+1}) - V_{\phi^i}(s_t)$$

$$\hat{A}_t^i = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^i$$

- $\gamma$: discount factor = 0.98 (相对高,说明他们关心 long-horizon stability)
- $\lambda$: GAE bias-variance knob = 0.95 (high variance, low bias — 因为仿真里可以 afford)
- $\delta_t^i$: 每个 agent 自己的 TD error,只看自己的 reward $r_t^i$ 和自己的 value function
- $\hat{A}_t^i$: 每个 agent 自己的 advantage estimate

PPO clipped objective 每个 agent 独立:
$$L_i^{CLIP}(\theta^i) = \mathbb{E}_t\left[\min\left(r_t^i \hat{A}_t^i, \text{clip}(r_t^i, 1 \pm \epsilon^i)\hat{A}_t^i\right)\right]$$

其中 $r_t^i(\theta^i) = \frac{\pi_{\theta^i}(a_t^i|s_t)}{\pi_{\theta^i_{old}}(a_t^i|s_t)}$ 是 importance sampling ratio, $\epsilon^i = 0.15$ 是 clip ratio (比 OpenAI 默认的 0.2 略紧)。

**Total objective** 加了一个 cross-agent torque regularization:
$$\mathcal{L}_{total} = \sum_{i \in \mathcal{I}} \mathcal{L}_i(\theta^i, \phi^i) + \lambda_c \mathcal{C}(a_t^i)$$

$$\mathcal{C}(a_t^i) = \frac{1}{T}\sum_{t=1}^{T}\left(\|a_t^l\|_2^2 + \|a_t^w\|_2^2 + \|a_t^u\|_2^2\right)$$

这个 $\mathcal{C}$ 很关键 — 没有 it,三个 agent 会各管各、拼命输出 torque 互相打架。$\lambda_c$ 是唯一耦合它们的 term,起到 **soft coordination** 作用。

### 2.3 Privileged Critic

Critic 多吃了 $\mathcal{P}_t = (v_t, o_t, F_t)$ —— linear velocity, orientation quaternion, external 6D force。Actor 不吃这些 (deploy 时测不到)。这是 [Learning Agile and Dynamic Motor Skills for Legged Robots (Hwangbo 2019)](https://arxiv.org/abs/1901.08652) 的 teacher-student privileged learning 套路,但 Thor 没做显式 distillation,而是直接 asymmetric actor-critic。这有个风险: actor 没有 explicit force sensing 时,能不能 robust 推断出 external force?他们用 **torso angular displacement + angular velocity on y-axis** 作为 implicit proxy,这个是 paper 里一个 understated 但很重要的 design choice。

---

## 3. FAT2 — Force-Adaptive Torso-Tilt Reward

这是 paper 的真正的核心创新。**让 robot 学会"在拉重物时 lean back"**,这听起来 obvious,但 reward design 极不 trivial。

### 3.1 ZMP 推导

模型简化成 rigid body,quasi-static 假设 (no significant acceleration)。设 robot 在拉一个力 $\vec{F_h}$,$\alpha$ 是这个力与地面的夹角。

**Force equilibrium** (公式 11):
$$\vec{F_s} + \vec{F_f} + \vec{F_h} + \vec{F_g} = \vec{0}$$

- $\vec{F_s}$: vertical ground reaction force (支持力)
- $\vec{F_f}$: horizontal friction at feet
- $\vec{F_h}$: hand 上的交互力
- $\vec{F_g}$: gravity at CoM

水平分量 $\vec{F_h^x}$ 被 friction 平衡 (条件: $|F_h^x| < \mu_s |F_s|$), 垂直分量 $\vec{F_h^y}$ 被 support force 吸收。

**Torque equilibrium** about support polygon centroid (公式 14):
$$\vec{r}_{CoM} \times \vec{F_g} + \vec{r_h} \times \vec{F_h^x} + \vec{r_h} \times \vec{F_h^y} = \vec{0}$$

化成标量形式 (公式 15):
$$|F_h| d_1 \cos\alpha + |F_h| d_3 \sin\alpha = |F_g| |r_{CoM}| \cos\beta$$

变量:
- $d_1 = |r_h| \cos\varphi$: EE 到地面的垂直距离
- $d_3 = |r_h| \sin\varphi$: EE 到 support center 的水平距离
- $\varphi$: $\vec{r_h}$ 与垂直方向夹角
- $\alpha$: pulling force 与地面夹角
- $\beta$: **torso tilt angle** (我们要 solve 的)
- $|r_{CoM}| \cos\beta$: CoM 到 feet 的水平距离 (β 越大,CoM 越远离 feet 投影中心)

因为 $\sin\varphi$ 很小 (手离 vertical axis 不远),$d_3$ 项可忽略,简化成 (公式 16):

$$|F_h| |r_h| \cos\varphi \cos\alpha = |F_g| |r_{CoM}| \cos\beta$$

**Target tilt angle** (公式 17):
$$\beta = \cos^{-1}\frac{|F_h||r_h|\cos\varphi\cos\alpha}{|F_g||r_{CoM}|} \leq \beta^{max}$$

$\beta^{max} = 0.9$ rad ≈ 51.6°,这个上界很关键 — 没有它,policy 会过度倾斜把自己摔了。

**Max pull force** (公式 19):
$$|F_h^{max}| = \frac{|F_g||r_{CoM}|\cos\beta^{max}}{|r_h|\cos\varphi\cos\alpha}$$

这就是为什么 backward pulling > forward pulling 的原因 — backward 时 robot 可以更大程度 lean back 让 CoM 出 support polygon,因此 $|F_h^{max}|$ 更大。

### 3.2 FAT2 Reward (公式 18)

$$R_{FAT2} = \exp\left(-\frac{\|\beta - \beta'\|^2}{\sigma_t}\right)$$

- $\beta$: 由当前 external force $F_h$ 通过公式 (17) 算出的 target tilt
- $\beta'$: robot 实际的 tilt angle
- $\sigma_t$: temperature,控制 reward sharpness

这是 Gaussian-shaped reward,经典 RL reward shaping 形式。好处是 smooth gradient,坏处是 $\sigma_t$ 要 tune。

### 3.3 Intuition: 为什么这是 big deal

之前的 humanoid RL (Falcon, HOMIE) 都在 reward 里加 "stay upright" term —— 拉力越大越要站直。Thor 反过来: 拉力越大,**应该越倾斜**。这等于 **reward shaping 里 inverted 了 human intuition**。

更深层的: 这里的 ZMP 不是用来当 hard constraint (像 model-based QP controller 那样),而是用来 **推导一个 target posture**,然后 reward 朝这个 target 收敛。这让 RL 有 "physical prior" 但不 loss flexibility。

[ZMP original paper — Vukobratović & Stepanenko 1972](https://link.springer.com/article/10.1007/BF02557668)

---

## 4. Training Pipeline

### 4.1 Two-stage Curriculum

- Stage 1: low force disturbance → 学 robust locomotion
- Stage 2: extreme force disturbance → 学 force adaptation

每个 stage ~3.4 小时 on RTX 4090,Isaac Gym。

### 4.2 Domain Randomization

External force 的方向和 magnitude 加 Gaussian noise,这个对 sim-to-real 极其重要 — 真实场景里 door handle、rack 的 friction / inertia 都有 variation,没有 DR,policy 会 overfit 仿真。

### 4.3 Upper-body Imitation from AMASS

Training 时 upper body target motion 从 [AMASS](https://amass.is.tue.mpg.de/) 随机采样 — 这是关键,让 $\pi_u$ 学会 track arbitrary human upper-body motion,deploy 时才能配合 VR teleoperation (IK 重定向后) 做 manipulation。这其实是 **diversity-driven imitation**,而不是过拟合到单一 task motion。

### 4.4 Deployment

- Policy inference: 50 Hz (三个 actor network 并行)
- PD controller: 500 Hz

Actor 输出 desired joint positions,PD controller 转 torque:
$$\tau = K_p (q_{des} - q) + K_d (\dot{q}_{des} - \dot{q})$$

这个 50/500 Hz 分层是 humanoid 标准 setup (参考 [ANYmal](https://arxiv.org/abs/1901.08652) 和 [Exbody](https://arxiv.org/abs/2402.16796))。高频 PD 提供 low-level stability,RL policy 做 higher-level adaptation。

---

## 5. Results 详解

### 5.1 Peak Pulling Force

Table II 关键数据:

| Method | $F_{db}^{180°}$ (N) | $F_{df}^{0°}$ (N) | Improvement vs Falcon |
|--------|---------------------|-------------------|----------------------|
| **Thor** | **167.7 ± 2.4** | **145.5 ± 2.0** | — |
| Falcon | 99.3 ± 1.3 | 83.3 ± 2.7 | +68.9% / +74.7% |
| HOMIE | 62.3 ± 3.7 | 48.1 ± 2.9 | — |
| Unitree Default | 59.2 ± 1.7 | 68.9 ± 4.0 | — |

G1 体重 35 kg ≈ 343 N。Thor 单 peak 167.7 N ≈ **48% body weight**。这个数字 human 来拉也很可观了。

**为什么 backward > forward**: 因为 lean back 时 body weight 直接 leverage 成 counter-force;lean forward 时,friction 必须提供反作用力,但摩擦系数有限 (即使他们加了 custom shoe cover)。

**为什么 dual-hand >> single-hand**: 单手拉时,robot 还要 counter torsional moment about vertical axis,所以 force budget 被分散。Table II 里 $F_{sp}^{180°}$ (single-hand pull, stand still) 只有 58.5 N,而 $F_{dp}^{180°}$ 是 127.4 N,差距 2.2x。

### 5.2 Ablation

- **Thor¹**: only FAT2 (no decouple) → 达 Thor 80-90%,某些 task 持平
- **Thor²**: only decouple (no FAT2) → 显著低于 Thor¹

结论: **FAT2 是主力,decouple 是 stabilizer**。但 ablation 里有 detail —— Thor¹ 在大 force 时 waist 出现 anomalous roll-angle deviation,decouple 解决这个问题。这其实暗示了 **decouple 的主要价值不在 raw force 提升,而在 coordination 的 cleanness**。

### 5.3 Real-World Tasks

- Fire door: ~60 N single-hand pull,backward
- Loaded rack (70 kg): 130 N
- Wheelchair (60 kg robot inside): push + turn
- Whiteboard wiping: 低 force 但需要 smooth whole-body coordination

---

## 6. Connections to Broader Literature

### 6.1 Multi-Agent RL for Robot Control
[Learning advanced locomotion for quadrupedal robots (Wang 2024)](https://www.mdpi.com/2218-6581/13/6/86) 把四足机器人每条腿当 agent。Thor 把这个 idea 推到 humanoid,但用了 **anatomically meaningful decomposition** (upper/waist/lower) 而不是 spatial decomposition (per-limb),这更符合 force transmission physics。

### 6.2 Humanoid Loco-Manipulation Lineage

- [OmniH2O](https://arxiv.org/abs/2406.08858) — universal teleoperation
- [Exbody / Exbody2](https://arxiv.org/abs/2412.13196) — expressive WBC
- [HOVER](https://arxiv.org/abs/2410.21229) — multimodal policy distillation
- [HOMIE](https://arxiv.org/abs/2502.13013) — exoskeleton cockpit + lower RL
- [Falcon](https://arxiv.org/abs/2505.06776) — first 显式 force-adaptive humanoid loco-manip
- [ASAP](https://arxiv.org/abs/2504.10343) — sim-real physics alignment
- [HOMIE / Mobile-TeleVision / Clone / Twist](https://arxiv.org/abs/2502.13013) — teleoperation-driven

Thor 的位置: **第一个 explicitly 把 ZMP-violating posture (CoM outside support polygon) 作为 reward signal 的 humanoid RL framework**。Falcon 也在做 force adaptation,但 stays within support polygon。Thor breaks 这个 implicit constraint。

### 6.3 LIPM vs. Learnable Posture

[3D LIPM (Kajita 2001)](https://ieeexplore.ieee.org/document/976094) 是 humanoid control 的奠基 model — CoM 永远在 support polygon 上方。Thor 实际上在 **learn to violate LIPM assumption**,这是一个 paradigm shift。Human 真实运动里大量违反 LIPM (跳跃、leaning、quick direction change),所以这是走向 truly human-level 的必经之路。

### 6.4 Privileged Critic & Teacher-Student
[CtF (Hwangbo 2019)](https://arxiv.org/abs/1901.08652), [RMA (Kumar 2021)](https://arxiv.org/abs/2107.04034), [ANYmal deployment](https://arxiv.org/abs/1909.06686)。Thor 没做显式 distillation,直接 asymmetric。这可能是个 limitation — actor 要从 implicit signal 推断 external force,可能不如 distilled student 稳。

### 6.5 Force Control 的另一条线: Impedance Control + RL
[Facet (Xu 2025)](https://arxiv.org/abs/2505.06883) — RL 学一个 virtual mass-spring-damper 的 reference tracking,explicitly controllable compliance。Thor 走的是不同路线: 不学 impedance parameter,直接学 whole-body posture response。两条路线未来可能 merge —— 用 FAT2 学 nominal posture,然后叠 impedance layer 做 fine force control。

### 6.6 Tug-of-war Biomechanics
[Analysis of tug of war competition (Cayero 2022)](https://www.mdpi.com/1660-4601/19/1/3) — paper 里引用的 biomechanics reference。Human tug-of-war 里 lean angle 与 team pulling force 有强 correlation。Thor 把这个 expert knowledge 编进 reward,等于 **RL 用 biomechanics prior 做 inductive bias**。这跟 [DeepMimic](https://arxiv.org/abs/1804.02717) 用 human mocap 做 reward 的思路是一脉相承的,只是 reward signal 不是 "match this motion" 而是 "match this posture given this force"。

---

## 7. Limitations & Open Questions

1. **Hyperparameter sensitivity**: paper 最后说 entropy / LR / reward scale 都要 manual tune,particularly 因为三个 agent 互相 depend。MA-RL 训练 instability 是 known issue ([QPLEX, QMIX limitations](https://arxiv.org/abs/2011.09993))。

2. **No explicit force sensing at deploy**: 完全靠 torso proprioception 推断。这意味着 **slow / static force 可能 detect 不到** (没有 angular velocity 信号)。Paper 里 Figure 5 显示 force 与 tilt 角有 monotonic 关系,但单点静持重物时,policy 能否 adapt 不清楚。

3. **Privileged critic 没 distill**: 可能 limit sample efficiency 的 upper bound。如果做 student-teacher distill (像 [Walk these ways](https://arxiv.org/abs/2212.03252)),deploy-time actor 可能更 robust。

4. **Quasi-static 假设**: FAT2 公式假设 no significant acceleration。Dynamic scenario (eg. 突然 force jerk,或者 robot 在 walking 时拉) 公式 (17) 可能不准。但 RL 可以 compensates by exploration,所以实际效果可能 OK。

5. **$\beta^{max} = 0.9$ rad 是 hand-tuned**: 这个 magic number 决定了 robot 敢 lean 多远。理想情况下应该是个 function of foot geometry + friction coefficient + ankle torque limit。Paper 里没讨论这个。

6. **Single-EE focus**: 整个 FAT2 推导是 single pulling force。Bimanual pushing / carrying symmetric load 时,paper 的几何模型需要扩展。

7. **Future work 提到 learn from human demo video**: 这暗示他们想走 [Visual imitation enables contextual humanoid control](https://arxiv.org/abs/2505.03729) 路线 — 从 YouTube 视频 extract posture-force relationship,自动 tune $\sigma_t$ 和 $\beta^{max}$。

---

## 8. Take-aways for Building Intuition

我看完这个 paper 之后有几个 mental model:

**(1) Reward design > network architecture**。Ablation 显示 FAT2 贡献 80-90%,decoupling 是 secondary。这说明在 humanoid RL 里,**"告诉 policy 什么是 good posture"** 比 "给 policy 更好的 network" 重要得多。这跟 [Intrinsic Reward in RL](https://arxiv.org/abs/1906.07965) 的发现一致 — RL 的 sample efficiency bottleneck 通常是 credit assignment,不是 capacity。

**(2) Physics prior in reward is huge**。FAT2 不是 "let RL figure it out",而是 "give RL a physics-derived target"。这跟 model-based RL 里的 [Differentiable Physics](https://arxiv.org/abs/2106.07340) 思路相通 — 用 physics prior 做 inductive bias。但比 model-based 优雅 — 只 informed reward,不 constrain policy space。

**(3) Decoupling 的真实价值在 coordination cleanliness,不在 raw performance**。Paper 的 ablation 有点 hidden message: 单 decoupling (Thor²) 性能反而不如单 FAT2 (Thor¹)。但 Thor¹ 在 high force 时 waist 会 misbehave。所以 decouple 是 "safety / cleanliness layer",FAT2 是 "performance layer"。这种 division of labor 是 design pattern。

**(4) ZMP-as-target 而非 ZMP-as-constraint**。这是 paper 最深的 insight 之一。Model-based control 用 ZMP 当 hard constraint (QP 里),RL 用 ZMP 当 reward target —— 后者允许 occasional violation 以换取 higher force output。这种 "soft physics prior" 是 learn-to-violate-LIPM 的关键。

**(5) Implicit force sensing via proprioception**。Robot 没 force sensor,靠 torso 角度 + 角速度推。这 validate 了 [Berniker & Kuo](https://arxiv.org/abs/1902.02446) 一类 work 里 "proprioception sufficient for adaptive control" 的论点。但也设了 limit — 真正 zero-velocity static load 可能盲。

---

## Reference Links

- [Thor Project Page](https://baai-aether.github.io/baai-thor/)
- [AMASS Dataset](https://amass.is.tue.mpg.de/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Falcon](https://arxiv.org/abs/2505.06776)
- [HOMIE](https://arxiv.org/abs/2502.13013)
- [HOVER](https://arxiv.org/abs/2410.21229)
- [OmniH2O](https://arxiv.org/abs/2406.08858)
- [Exbody2](https://arxiv.org/abs/2412.13196)
- [ASAP](https://arxiv.org/abs/2504.10343)
- [Facet (Impedance + RL)](https://arxiv.org/abs/2505.06883)
- [Unitree G1](https://www.unitree.com/g1/)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [3D LIPM (Kajita 2001)](https://ieeexplore.ieee.org/document/976094)
- [ZMP original (Vukobratović 1972)](https://link.springer.com/article/10.1007/BF02557668)
- [Learning Agile Motor Skills (Hwangbo 2019, ANYmal)](https://arxiv.org/abs/1901.08652)
- [RMA (Kumar 2021)](https://arxiv.org/abs/2107.04034)
- [DeepMimic](https://arxiv.org/abs/1804.02717)
- [Walk These Ways (Margolis 2023)](https://arxiv.org/abs/2212.03252)
- [Multi-Agent DRL Survey](https://link.springer.com/article/10.1007/s10462-020-09927-w)
- [Tug-of-war biomechanics](https://www.mdpi.com/1660-4601/19/1/3)
- [Low back biomechanics during deadlift](https://www.tandfonline.com/doi/abs/10.1080/24725832.2021.2017924)
- [Mobile-TeleVision](https://arxiv.org/abs/2502.06457)
- [AMO](https://arxiv.org/abs/2505.03738)
- [HumanPlus](https://arxiv.org/abs/2406.10454)
- [Visual Imitation for Humanoid Control](https://arxiv.org/abs/2505.03729)
- [Clone](https://arxiv.org/abs/2506.08931)
- [Twist](https://arxiv.org/abs/2505.02833)
- [BeyondMimic](https://arxiv.org/abs/2508.08241)
- [Visual Imitation Contextual Control](https://arxiv.org/abs/2505.03729)

---

## Final Thought

Andrej, 我觉得这个 paper 真正的 contribution 不是 167.7 N 这个数字,而是它 **demonstrate 了 humanoid RL 可以 systematically violate LIPM 假设并 outperform**。前面 Falcon / HOMIE 一代 humanoid RL 还在 "stay upright + locomote" 的 paradigm 里,Thor 把 paradigm 推到 "adaptively violate upright to maximize task force"。这跟 human athleticism 的本质更接近 — elite 运动员都在 boundary of balance 之外 operate。

下一步我想看的是: 把 FAT2 的 idea 推到 **dynamic** scenario (running pull, jumping push),那里 quasi-static ZMP 推导 break down,需要 capture-body 或者 angular-momentum-based reward。还有 bimanual non-symmetric force (比如提箱子一边重一边轻),那需要 generalized FAT2 with vector torque balance。

Paper 自己说未来要从 human demo video 学 expert knowledge 来 auto-tune hyperparameter —— 这其实是想用 [VPT / Video PreTraining](https://arxiv.org/abs/2210.02071) 那种思路给 humanoid RL 提供 reward prior。如果这条路走通,可能 humanoid RL 就摆脱了 reward engineering 的 curse。

希望这个分析对你 build intuition 有帮助!
