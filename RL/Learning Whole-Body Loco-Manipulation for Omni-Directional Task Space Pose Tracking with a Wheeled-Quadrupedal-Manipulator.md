---
source_pdf: Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space
  Pose Tracking with a Wheeled-Quadrupedal-Manipulator.pdf
paper_sha256: 8d950160a4fdaa7d1cd474f3e6590800ed3d9adf0575a23f8c5301b179f7f24d
processed_at: '2026-08-05T14:18:35-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话总结

他们造了个 **wheeled-quadruped 加机械臂** 的机器人 (WQM), 用一个 unified RL policy 直接追踪 6D end-effector pose, 不需要人告诉机器人 "先走再抓" 或者 "base 往哪走"。机器人自己决定什么时候用轮子挪、什么时候伸 arm、什么时候两者同时动。

核心创新是一个叫 **RFM (Reward Fusion Module)** 的东西 — 换掉了传统 RL 里 "把所有 reward 加权求和" 的做法。

---

## 为什么 weighted sum 不行

传统做法: 你有一堆 reward term, position tracking、orientation tracking、energy、smoothness 等等, 每个 term 乘个 weight 加起来:

$$r = \omega_1 r_1 + \omega_2 r_2 + \dots + \omega_n r_n$$

问题在哪? 想象机器人离 target 还有 5 米。这时候 orientation tracking reward $r_{eo}$ 根本没意义 — 你还在走路呢, arm 朝哪指完全无所谓。但 weighted sum 里 $\omega_{eo} r_{eo}$ 始终在产生 gradient, policy 会被拉着 "边走边扭 arm", 浪费 energy 还容易摔。

**本质问题**: linear combination 表达不了 "**当 A 满足后 B 才重要**" 这种 conditional 关系。loco-manipulation 天生是 hierarchical、multi-stage 的任务, weighted sum 把它压成扁平结构, 必然妥协。

参考 model-based control 领域早就知道这个道理: HWBC (Hierarchical WBC) 用 null-space projection 保证 priority, 效果远好于 WWBC (Weighted WBC) 的 soft compromise。
- HWBC: https://ieeexplore.ieee.org/document/7803338
- WWBC NMPC: https://ieeexplore.ieee.org/document/10160599

RFM 干的事: **把 HWBC 的 hierarchy 思想搬进 RL reward design**。

---

## RFM 三个组件, 各解决一个问题

### 1. Reward Prioritization (RP): 用乘法做 priority

核心 idea 极简:

$$r_t = r_{ep} + r_{ep} \cdot r_{eo}$$

- $r_{ep}$: position tracking reward, 范围 $(0, 1]$
- $r_{eo}$: orientation tracking reward, 范围 $(0, 1]$

看偏导数就懂了:

$$\frac{\partial r_t}{\partial r_{eo}} = r_{ep}$$

当 $r_{ep}$ 接近 0 (position 错得离谱), orientation 的 gradient 自动趋零 — **orientation reward 被 gate 掉了**, policy 不会浪费精力管朝向。当 $r_{ep}$ 接近 1 (position 到位了), orientation gradient 满血复活。

**零 weight 需要调**, 这是关键。传统做法你要调 $\omega_{ep}$ 和 $\omega_{eo}$ 的比例, 调不好就崩。这里乘法天然产生 gating, 没有自由度。

再嵌一层: 把 regularization 放最外层

$$r_t = r_{reg} + r_{reg} \cdot (r_{ep} + r_{ep} r_{eo})$$

意思是: **只有当 robot 处于安全稳定状态 ($r_{reg}$ 高), tracking reward 才生效**。如果 robot 快摔了 ($r_{reg} \to 0$), 整个 tracking 部分被 squash, policy 不会为了 tracking 分数去冒险。

这就是 **可微的 null-space projection**, 用 reward gradient 实现了 model-based control 的 task hierarchy。

### 2. Enhancement: 把精度拉上来

两个 trick:

**Micro Enhancement**:

$$r_{ep}^* = r_{ep} + (r_{ep})^M, \quad M > 1$$

$(r_{ep})^M$ 当 $r_{ep} \to 1$ 时贡献大, 当 $r_{ep} \to 0$ 时贡献几乎为零。效果: 只在 "已经接近 target" 时加大 reward gradient, 远距离不动。

为什么不直接增大 weight? 增大 weight 会等比例放大所有 error range 的 gradient, 远距离时 position reward 过度主导, 压制 locomotion。Micro enhancement 是 **selective sharpening** — 只在 goal 附近把 reward 曲面 "lift up"。

**Cumulative Penalty**:

$$e_t^{cb} = e_{t-1}^{cb} + \kappa \cdot \epsilon_t$$

累积误差, 类似 PID 的 integral term。如果 robot 卡在某个稳态误差 (比如 arm stretch 到极限还差 5cm), $\epsilon_t$ 不为零, $e^{cb}$ 持续增长, 该 state 的 value 持续下降, policy eventually 被推出 local minima 去探索。

巧妙之处: $\kappa = (1 - \mathcal{D}(\epsilon_t^{ref}))$, 在 locomotion 阶段 $\kappa \to 0$ 不累积, 在 manipulation 阶段 $\kappa \to 1$ 全力累积。**只在该精调的时候才积分误差**。

参考 potential-based reward shaping 理论: https://papers.nips.cc/paper/1999/hash/464d828b85b8bed9e422e5f7722878d2-Abstract.html

### 3. Loco-Mani Fusion: smooth 切换两个 mode

定义一个 phase variable:

$$\epsilon_t^{ref} = \max\{\epsilon_0 - v \cdot t, 0\}$$

- $\epsilon_0$: 初始 SE(3) 距离
- $v$: 用户指定的接近速度
- $\epsilon_t^{ref}$: 理想进度条, 随时间线性降到 0

注意 **$\epsilon_t^{ref}$ 只依赖初始状态和时间**, 跟 robot 实际状态无关。它是个 "你应该到哪了" 的 reference。

然后过 sigmoid:

$$\mathcal{D}(x) = \frac{1}{1 + e^{-5(x-\mu)/l}}$$

当 $\epsilon_t^{ref}$ 大 (距离远), $\mathcal{D} \to 1$, locomotion reward 主导; 当 $\epsilon_t^{ref}$ 小 (距离近), $\mathcal{D} \to 0$, manipulation reward 主导。中间 sigmoid 平滑过渡。

最终 reward:

$$r_t = (1-\mathcal{D}) \cdot r_{mani} + \mathcal{D} \cdot r_{loco} + r_{basic}$$

**为什么这比硬切换好**: 之前的工作如 Deep WBC [1] 用两个 mode 手动切换, 切换时有明显停顿。这里 sigmoid 是连续可微的, reward 在两个 mode 间 soft blend, policy 学到的是 smooth transition。

paper 推荐参数 $\mu = 2 l_a$, $l = 2 l_a$ ($l_a$ 是 arm span), 几何意义: 当距离约两个 arm span 时开始过渡, 过渡宽度也是两个 arm span。

参考 Deep WBC: https://arxiv.org/abs/2301.08357

---

## 训练 pipeline

标准 RMA-style teacher-student:

- **Teacher**: 能看到 privilege info (contact force, friction, EE twist), 编码成 latent $z \in \mathbb{R}^{32}$, 跟 policy 一起 PPO 训练
- **Student**: 只看 proprioceptive, 用 10 帧 history 估 $\hat{z}$, loss 是 MSE, warm start from teacher, 再 PPO fine-tune

这样 real world 部署时 policy 能从 history 推断隐含动力学参数, 不需要直接测 friction / payload。

参考 RMA: https://arxiv.org/abs/2107.04034

硬件: LimX W1 (wheeled quadruped, 16 actuators) + AIRBOT 6-DOF arm。4096 parallel envs 在 RTX 3090 上, teacher 训 8 小时, student 4 小时。Isaac Gym 训练, Mujoco 做 sim-to-sim 验证, 再部署 real。

---

## 实验数据说话

Ablation study (每个 policy 跑 1000 次):

| Policy | 成功率 | 功耗 | 关节加速度 | pos error | ori error |
|--------|--------|------|-----------|-----------|-----------|
| **RFM 完整版** | **99.0%** | 71W | 144 | **0.022m** | **0.041rad** |
| w/o Loco-Mani Fusion | 82.1% | 123W | 333 | 0.023m | 0.066rad |
| w/o Reward Prioritization | 93.0% | 65W | 171 | 0.045m | 0.029rad |
| w/o Enhancement | 98.7% | 62W | 121 | 0.036m | 0.117rad |
| w/o RFM (纯 weighted sum) | 92.9% | 87W | 187 | 0.168m | 0.030rad |

几个关键 takeaway:

1. **w/o Loco-Mani Fusion**: 功耗翻倍, 加速度翻倍, 成功率暴跌。两个 mode 的 reward 同时争夺 gradient, policy 在 mode 间 jittery oscillation, 能耗与失败齐升。

2. **w/o Reward Prioritization**: nominal deviation 暴涨到 0.393m (vs 0.115)。机器人用危险姿态换 tracking 分, 因为没有 regularization gating, agent 敢冒险。有趣的是 ori error 反而最低 (0.029), 因为 weighted sum 让 ori 全程生效, 但代价是 posture 乱。paper 故意调了 48 小时 weight 仍救不回来。

3. **w/o Enhancement**: 功耗最低看似健康, 但 ori error 灾难性 (0.117rad)。机器人找到 "大致到位精度差" 的稳态就停了, low power 是因为不努力 — **懒惰但稳定的次优解**。

4. **w/o RFM**: 纯 weighted sum, 8 个 weight 调两天, pos error 0.168m (RFM 的 8 倍), nominal deviation 0.467m (4 倍)。

Hardware 实测: pos error < 5cm, ori error < 0.1rad, SOTA。

---

## 这篇 paper 的意义

核心 insight: **RL reward design 不该是 flat 的 weighted sum**。复杂任务有 hierarchy、有 phase、有 conditional relevance, reward fusion 也该有结构。RFM 用乘法做 gating、用 sigmoid 做 phase switching、用 cumulative penalty 做积分控制, 把 model-based control 几十年积累的 hierarchy 思想用可微的方式搬进 RL。

更深层: 这套思路不限于 loco-manipulation。任何 multi-stage whole-body control 问题 — humanoid 双手操作、multi-agent coordination、甚至 humanoid locomotion + manipulation — 都可以用 RFM 的 nesting 乘法 + phase variable 来结构化 reward。

传统 RL 社区长期把 reward design 当 "黑艺术", 调 weight 靠玄学。RFM 提供了一种 **有数学依据的 reward 结构化方法**, hyperparameter 极少 (mani 内部零 weight, loco 内部一个), 可解释性强。这是 reward engineering 从 ad-hoc 走向 principled 的一步。

Limitation paper 自己也说了: 没有 perceptive input, confined space 场景受限。下一步必然加 vision, 类似 Visual WBC [19] 或 DTC [5] 的思路。
- Visual WBC: https://arxiv.org/abs/2403.16967
- DTC: https://www.science.org/doi/10.1126/scirobotics.adh5401

---

# Whole-Body Loco-Manipulation with RFM: 深度解析

这篇 paper 解决的核心问题: **让 wheeled-quadrupedal manipulator (WQM) 直接追踪 task space 中固定的 6D end-effector pose**, 用单一 unified policy 协调轮子、腿、机械臂, 而无需 user 给 base velocity command。真正的难点在于 locomotion 和 manipulation 这两个 modalities 之间存在天然的 hierarchical 与 multi-stage 结构, 传统的 weighted-sum reward 无法表达这种 conditional relevance。RFM (Reward Fusion Module) 用非线性乘法 + sigmoid phase variable 来解决。

---

## I. 核心动机: 为什么 weighted-sum reward 失败

传统 RL reward 设计采用 $r = \sum_i \omega_i r_i$, 本质是 linear combination。对于 loco-manipulation 这种 long-horizon multi-stage 任务, linear combination 存在根本缺陷:

**Intuition**: 当 EE target 距离 robot 很远时, orientation tracking reward $r_{eo}$ 应该 **完全 irrelevant** — robot 还在 locomotion 阶段, 追 orientation 既浪费 energy 又增加 failure rate。但 weighted sum 中 $\omega_{eo} r_{eo}$ 始终贡献 reward gradient, 即使 $r_{eo}$ 很小, policy 仍会被拉向"边走边扭 arm"的次优解。Linear combination 无法表达 "**只有当 position 满足后, orientation 才被关注**" 这种 conditional gating。

这与 model-based control 中 **HWBC (Hierarchical Whole-Body Control)** vs **WWBC (Weighted WBC)** 的对比完全一致 — HWBC 用 null-space projection 保证 priority, WWBC 用 soft weight 只能得到 compromise。RFM 把 HWBC 的 priority 思想移植到 RL reward 设计里。

参考文献:
- HWBC: https://ieeexplore.ieee.org/document/7803338
- WWBC (Perceptive locomotion NMPC): https://ieeexplore.ieee.org/document/10160599

---

## II. RFM 三大组件深度拆解

### A. Reward Prioritization (RP): 乘法门控实现 hierarchy

核心公式 (3):

$$r_t = r_{ep} + r_{ep} \cdot r_{eo}$$

变量说明:
- $r_{ep} \in (0, 1]$: EE position tracking reward
- $r_{eo} \in (0, 1]$: EE orientation tracking reward
- $r_t$: 融合后的 manipulation tracking reward

**为什么这个形式实现 priority**: 看偏导数
$$\frac{\partial r_t}{\partial r_{ep}} = 1 + r_{eo} \geq 1 \quad (\text{始终非零})$$
$$\frac{\partial r_t}{\partial r_{eo}} = r_{ep} \quad (\text{只有 } r_{ep} \text{ 大时才显著})$$

当 $r_{ep} \to 0$ (position 错得离谱), $\partial r_t / \partial r_{eo} \to 0$, orientation 的 gradient 被自动屏蔽; 当 $r_{ep} \to 1$, $\partial r_t / \partial r_{eo} \to 1$, orientation 全力生效。**没有 weight 需要调**, 这是关键优势。

**几何直觉**: $r_t$ 在 $(r_{ep}, r_{eo}) \in [0,1]^2$ 平面上是一张曲面, 在 $r_{ep}=0$ 这条边上沿 $r_{eo}$ 方向斜率为 0, 在 $r_{ep}=1$ 这条边上斜率为 1。这种 "wedge" 形状天然编码了 priority。

**第二层 priority** (5): 把 regularization 提到最外层
$$r_t = r_{reg}^{mani} + r_{reg}^{mani} \cdot (r_{ep} + r_{ep} r_{eo})$$

$r_{reg}^{mani}$ 表示 "维持 smooth feasible motion" 的 reward (如避免极端 joint 角度)。当 $r_{reg}^{mani} \to 0$ (即将失败), 整个 tracking term 被 squash 到 0, agent 不会为了 tracking 的高分去冒险。

**推广到 base velocity command** (6):
$$r_t = r_{reg}^{mani} + r_{reg}^{mani}(r_{ep} + r_{ep} r_{eo}) + r_{reg}^{loco} \cdot r_{bv}$$

这种 nesting 结构让 reward design 从 "调 8 个 weight" 变成 "嵌套乘法", 大幅减少 hyperparameter。

### B. Enhancement: 精度提升的两板斧

#### 1. Micro Enhancement

公式 (7):
$$r_{ep}^* = r_{ep} + (r_{ep})^M, \quad M > 1$$

其中 $M$ 是 enhancement 参数 (paper 中没明说具体值, 但 Fig.3 用 $r_{ep} = e^{-d_p/0.25}$ 演示)。

**关键 insight**: $r_{ep}^M$ 当 $r_{ep} \to 1$ (误差小) 时趋近 1, 加上去相当于在近目标区域把 reward 曲面 "lift up", 增大 local gradient; 当 $r_{ep} \to 0$ (误差大), $r_{ep}^M \to 0$ (因为 M>1 衰减更快), 对远距离 gradient 几乎无影响。

**为什么不直接增大 weight $\omega_{ep}$**: 增大 weight 会等比例放大所有 error range 的 gradient, 可能导致远距离时 position reward 过度主导, 压制 locomotion 和 regularization。Micro enhancement 是 **selective curvature sharpening** — 只在 "已经接近" 时加力, 类似 **shaped reward near goal**。

这与 optimization 中 trust-region 方法的精神相通: 在不同 error scale 用不同的 curvature。

#### 2. Cumulative Penalty Mechanism

公式 (8):
$$r_{cb,t} = e_t^{cb}, \quad e_t^{cb} = e_{t-1}^{cb} + \kappa \cdot \epsilon_t$$

变量:
- $e_t^{cb} \in \mathbb{R}$: 累积误差状态 (paper 用 clip 限制上界为 20)
- $\kappa$: 累积权重, 关键是 $\kappa = (1 - \mathcal{D}(\epsilon_t^{ref}))$ — 在 locomotion 阶段 $\kappa \to 0$ 不累积, 在 manipulation 阶段 $\kappa \to 1$ 全力累积
- $\epsilon_t$: 当前 SE(3) tracking error (9)

公式 (9) 定义 SE(3) 距离:
$$\epsilon_t = a_1 \|\log(R_{ee,t}^* R_{ee}^T)\|_F + a_2 \|P_{ee,t}^* - P_{ee}\|_2$$

变量与几何意义:
- $R_{ee,t}^* \in SO(3)$: 目标 orientation
- $R_{ee}^T$: 当前 EE orientation 的转置
- $R_{ee,t}^* R_{ee}^T \in SO(3)$: relative rotation
- $\log(\cdot): SO(3) \to \mathfrak{so}(3)$: matrix logarithm, 把 rotation 映射到 axis-angle 向量 $\omega \in \mathbb{R}^3$
- $\|\cdot\|_F$: Frobenius norm, $\|\omega\|_F = \sqrt{2(1-\cos\theta)}$ 比例于旋转角 $\theta$
- $a_1, a_2$: 位置与朝向的权重

这是 Park (1995) 提出的 SE(3) bi-invariant metric, 参考: https://www.osti.gov/biblio/70500

**Cumulative penalty 的双重作用**:
1. **克服 local minima**: 如果 agent 卡在某个稳态误差 (如 arm stretch 到极限但还差 5cm), $\epsilon_t$ 不为零, $e^{cb}$ 持续增长, 该 state 的 value $V(s)$ 持续下降, eventually policy 会被推出去探索。
2. **模仿 PID 的 I 项**: 经典控制中 integral term 消除稳态误差, 这里 RL 没有显式 integrator, 但 cumulative penalty 提供了类似的 "记忆"。

paper 还提到 $e^{cb}$ 必须作为 **privilege 信息输入 Critic** (但不需要输入 Actor)。这是因为 $e^{cb}$ 包含历史, Critic 用它估 value 更准, 而 Actor 用 policy parameter 隐式处理历史。

参考 potential-based reward shaping 理论 (Ng et al. 1999 保证 optimal policy invariance): https://papers.nips.cc/paper/1999/hash/464d828b85b8bed9e422e5f7722878d2-Abstract.html

### C. Loco-Mani Fusion: phase variable 驱动的 smooth switching

这是 RFM 最精妙的部分。先定义 phase reference:

公式 (11):
$$\epsilon_t^{ref} = \max\{\epsilon_0 - v \cdot t, 0\}$$

变量:
- $\epsilon_0$: 初始时刻的 SE(3) 距离 (9)
- $v \in (v_{min}, v_{max})$: 用户指定的 "期望接近速度", 决定 robot 多快从远处挪到目标
- $t$: 时间
- $\epsilon_t^{ref}$: 一个随时间线性下降到 0 的 reference trajectory

**关键**: $\epsilon_t^{ref}$ **只依赖初始状态和时间**, 与 robot 实际状态无关。它是一个 "理想进度条"。

然后引入 sigmoid gate (12):
$$\mathcal{D}(x; \mu, l) = \frac{1}{1 + e^{-5(x-\mu)/l}}$$

变量:
- $x = \epsilon_t^{ref}$
- $\mu$: sigmoid 中心点, paper 推荐 $\mu = 2 l_a$ ($l_a$ 为 arm span)
- $l$: sigmoid 宽度, 推荐 $l = 2 l_a$

当 $\epsilon_t^{ref} \gg \mu$ (距离远), $\mathcal{D} \to 1$, locomotion mode; 当 $\epsilon_t^{ref} \ll \mu$ (距离近), $\mathcal{D} \to 0$, manipulation mode; 中间过渡。

最终 reward fusion (10):
$$r_t = (1 - \mathcal{D}(\epsilon_t^{ref})) \cdot r_{mani} + \mathcal{D}(\epsilon_t^{ref}) \cdot r_{loco} + r_{basic}$$

**为什么这种设计是 smooth 的**: $\mathcal{D}$ 是 sigmoid, 导数连续, reward 在两个 mode 之间 soft blend。不像之前工作 [1] 那样硬切换 mode 导致明显停顿。而且 $\epsilon_t^{ref}$ 是 time-based, robot 的 policy 学习到 "用 locomotion 让实际 SE(3) 距离跟上 reference 下降速度", 这给 locomotion 一个明确的 coarse target。

**displacing WQM reward** (13):
$$\mathbf{r}_{dw,t} = \exp(-\tilde{e}_t^{ref}/\sigma_s), \quad \tilde{e}_t^{ref} = \max\{|\epsilon_t^{ref} - \epsilon_t| - \gamma, 0\}$$

变量:
- $\epsilon_t$: 实际 SE(3) 误差
- $\epsilon_t^{ref}$: reference 误差
- $\gamma$: release parameter, 中途放宽精度要求
- $\sigma_s$: 温度参数

这个 reward 鼓励 robot 的实际进度 ($\epsilon_t$) 跟上 reference 进度 ($\epsilon_t^{ref}$), 但允许 $\gamma$ 的 slack。

**总结 RFM 全貌** (14):
$$
\begin{aligned}
\mathbf{r}_{mani} &= r_{reg}^{mani} + r_{reg}^{mani}(r_{ep}^* + r_{ep} r_{eo}^*) + r_{pb} - r_{cb} + r_{ac} \\
\mathbf{r}_{loco} &= r_{reg}^{loco} + r_{reg}^{loco} r_{dw} - \omega_{sa} r_{sa} \\
\mathbf{r}_t &= (1-\mathcal{D}) r_{mani} + \mathcal{D} r_{loco} + r_{basic}
\end{aligned}
$$

其中:
- $r_{pb}$: potential-based reward (Ng 风格)
- $r_{ac}$: all-contact reward (四轮着地 = 1)
- $r_{sa}$: static arm reward (locomotion 时 arm 收拢)
- $r_{basic}$: action rate, smoothness, collision, alive, power, torque 等

paper 强调: **mani 内部零 weight**, **loco 内部仅一个 weight** ($\omega_{sa}$)。整个 RFM 的 hyperparameter 数量极少, 远少于传统 weighted-sum。

---

## III. Training Pipeline: Teacher-Student with RMA-style privilege

### A. Policy input (Table I)

| 类别 | 项 | 维度 |
|------|-----|------|
| Proprioceptive $\mathbf{o}_t$ | non-wheel joint pos $q^{nw}$ | 18 |
| | joint vel $\dot{q}$ | 22 |
| | EE position ${}^B P_{ee}$ | 3 |
| | EE orientation $\text{Vec}({}^B R_{ee})$ | 9 |
| | last action $\mathbf{a}_{t-1}$ | 22 |
| | base angular vel ${}^B \omega_b$ | 3 |
| | projected gravity ${}^B g$ | 3 |
| | SE(3) distance ref $\epsilon_t^{ref}$ | 1 |
| Command $\mathbf{c}_t$ | 6D EE target ${}^B T_{ee}^*$ | 12 |
| Privilege latent $\hat{z}_t$ | estimated latent | 32 |

总输入维度 ~125。注意 command 是 body frame 表示的, 因为 agent 应该 agnostic to inertial frame (参考 Pedipulate [16] 思路)。

### B. Teacher-Student 结构

**Teacher**: 
- privilege encoder $E_\phi$: 输入 contact forces, friction, EE twist 等 → $z \in \mathbb{R}^{32}$
- policy $\pi_\theta(\mathbf{o}_t, \mathbf{c}_t, z)$ 输出 action
- 用 PPO 联合训练 encoder + policy

**Student**:
- privilege estimator $\hat{E}_{\hat{\phi}}$: 输入 10 帧 proprioceptive history → $\hat{z} \in \mathbb{R}^{32}$
- loss: $\mathcal{L} = \text{Mean}(\|\hat{z} - \text{sg}[z]\|^2)$, sg = stop gradient
- 初始化时从 teacher policy warm start
- 用 PPO fine-tune

这种两阶段架构参考 RMA (Rapid Motor Adaptation): https://arxiv.org/abs/2107.04034
通过 latent variable 让 policy 推断隐含动力学参数 (payload mass, friction, terrain), 实现 sim-to-real。

### C. Action space

WQM 共 22 个 joint:
- 18 个 non-wheel (4 leg × 3 + arm × 6): 位置控制 $a_t^{nw}$ → PD → $\tau_t^{nw}$
- 4 个 wheel: 速度控制 $a_t^w$ → PD (只用 D 项) → $\tau_t^w$

公式 (2):
$$\tau_t^{nw} = K_p^{nw}(a_t^{nw} - q_n + q_t) - K_d^{nw} \dot{q}_t^{nw}$$
$$\tau_t^w = K_d^w(a_t^w - \dot{q}_t^w)$$

注意 wheel 没有 $K_p$ 项 (因为是 velocity control, 不需要 position 误差), 而且 $a_t^{nw}$ 是相对于 nominal position $q_n$ 的偏移, 这是 legged robot 标准做法。

---

## IV. 实验数据深度分析

### A. Ablation Study (Table III, 1000 trials each)

| Policy | SR ↑ | Power(W)↓ | Acc(rad/s²)↓ | Pos err(m)↓ | Ori err(rad)↓ | Devi(m)↓ |
|--------|------|-----------|--------------|-------------|---------------|----------|
| **Ours (RFM)** | **99.0%** | 70.98 | 143.75 | **0.022** | **0.041** | 0.115 |
| w/o LMF | 82.1% | 122.92 | 332.88 | 0.023 | 0.066 | 0.228 |
| w/o RP | 93.0% | 65.32 | 171.34 | 0.045 | 0.029 | **0.393** |
| w/o En. | 98.7% | 62.09 | 121.32 | 0.036 | 0.117 | 0.117 |
| w/o RFM | 92.9% | 87.22 | 186.94 | 0.168 | 0.030 | 0.467 |

**逐行 insight**:

1. **w/o LMF (Loco-Mani Fusion)**: 把 (1-D)*mani + D*loco 换成 $\omega_1 r_{mani} + \omega_2 r_{loco} + \omega_3 r_{basic}$, 调了 2 天找到 $\{1.2, 0.4, 1.0\}$。
   - SR 暴跌到 82.1%, power 翻倍 (122W vs 70W), joint acceleration 翻倍 (333 vs 144)。
   - **解释**: 没有 phase variable, locomotion 和 manipulation reward 同时争夺 gradient, policy 在两个 mode 间 jittery oscillation, 能耗与失败率齐升。

2. **w/o RP**: 把 (5) 的乘法 hierarchy 换成 weighted sum, 调了 48 小时 (paper 故意花长时间调以表诚意)。
   - nominal deviation 暴涨到 0.393m (vs 0.115) — wheel 位置偏离 nominal, 意味着 robot 用危险姿态换 tracking 分。
   - 有趣: ori err 反而最低 (0.029), 因为 weighted sum 让 ori 全程生效, 但代价是 posture 乱。
   - **解释**: 没有 regularization 的 priority gating, agent 敢于冒险。

3. **w/o Enhancement**: power/acc 最低, 看似健康, 但 ori err 灾难性 (0.117)。
   - **解释**: 没有 micro enhancement 和 cumulative penalty, agent 找到一个 "大致到位但精度差" 的稳态就停下来。low power 是因为不努力 — "懒惰但稳定" 的次优解。

4. **w/o RFM (Standard PPO)**: 8 个 weight 调两天。
   - pos err 0.168m (8 倍于 ours!), nominal devi 0.467m (4 倍)。
   - 完全失败的设计空间探索。

### B. Hardware (Table IV)

| 任务 | Pos err(m)↓ | Ori err(rad)↓ | SE(3)↓ |
|------|-------------|---------------|--------|
| fixed points | 0.028±0.019 | 0.089±0.073 | 0.145 |
| spatial circle | 0.048±0.022 | 0.085±0.025 | 0.181 |

对比 [1] Deep Whole-Body Control 和 [16] Pedipulate, pos err < 5cm, ori err < 0.1rad, 是 state-of-the-art。

hardware validation 视频可在 paper website 查询 (paper 提到 "refer to this website"), GitHub 关联项目:
- Legged Gym 框架: https://github.com/leggedrobotics/legged_gym
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- Mujoco: https://mujoco.org/

### C. Training setup

- 4096 parallel envs on RTX 3090
- Teacher: 7000 iterations × 24 steps
- Student: 4000 iterations
- 8h teacher + 4h student
- Isaac Gym → Mujoco sim-to-sim → real WQM (LimX W1 + AIRBOT 6-DOF arm)
- LIO (Lidar Inertial Odometry) 估 base world position, FK 算 EE world pose

---

## V. 跨工作对比与定位

| 工作 | 平台 | 6D? | unified? | base cmd? |
|------|------|-----|----------|-----------|
| Fu et al [1] Deep WBC | quadruped+arm | 仅 3D | 2 mode 切换 | 需要 |
| Wang et al [2] P3O | wheel-leg+arm | 仅 3D | unified | 需要 |
| Portela [3] Force ctrl | legged+arm | 3D + force | unified | 需要 |
| Pedipulate [16] | quadruped | 6D? | leg as arm | - |
| Visual WBC [19] | quadruped+arm | 6D via IK | decoupled (IK + RL leg) | vision planner |
| **This paper** | **WQM** | **6D direct** | **unified, smooth** | **不需要** |

[19] 用 IK 控制 arm 是因为 RL 无法 track EE orientation — 这正是 RFM Enhancement 解决的痛点。IK 的 pseudoinverse Jacobian 不考虑 joint limit 和 dynamics, 限制了 whole-body 协同。

参考:
- Deep WBC: https://arxiv.org/abs/2301.08357
- Visual WBC: https://arxiv.org/abs/2403.16967
- Pedipulate: https://arxiv.org/abs/2403.18920
- DTC (Deep Tracking Control): https://www.science.org/doi/10.1126/scirobotics.adh5401
- ANYmal parkour: https://www.science.org/doi/10.1126/scirobotics.adi7566
- ASC (Adaptive Skill Coordination): https://arxiv.org/abs/2301.13088

---

## VI. 深层直觉与扩展思考

### 1. RFM 与 WBC null-space projection 的同构

HWBC 中, 高优先级任务的 Jacobian $J_1$ 定义 constrained space, 低优先级任务通过 $N(J_1) = I - J_1^+ J_1$ 投影到 null-space。RP 中 $r_{ep}$ 起到 "soft null-space projector" 的作用 — 当 $r_{ep}$ 小, orientation gradient 被屏蔽 ($\partial r_t / \partial r_{eo} = r_{ep} \to 0$); 当 $r_{ep}$ 大, orientation 被释放。这是 **可微的 null-space gating**, 用 reward gradient 实现了 model-based control 的 hierarchy。

参考: Sentis & Khatib 2005 Task-Space WBC: https://journals.sagepub.com/doi/10.1177/0278364905058061

### 2. Cumulative penalty 与 potential-based shaping 的理论保证

Potential-based reward shaping (Ng 1999) 保证不改变 optimal policy, 形式 $F(s,s') = \gamma \Phi(s') - \Phi(s)$。Cumulative penalty $r_{cb}$ 严格说不是 potential-based (它是积分, 不是差分), 但精神类似 — 它给 "stuck state" 持续惩罚, 等价于降低该 state 的 value。这种 "value decay over stuck time" 等价于给 exploration bonus, 是一种 intrinsic motivation 的简化版。

参考 intrinsic motivation / count-based exploration: https://arxiv.org/abs/1606.01847

### 3. Phase variable 与 curriculum learning 的关系

$\epsilon_t^{ref} = \max\{\epsilon_0 - vt, 0\}$ 本质是一个 **time-indexed curriculum**。训练初期 robot 学远距离 locomotion (大 $\epsilon_0$), 后期学近距离 manipulation。$v$ 控制课程进度。这与 Nakashima curriculum 思想一致, 但 RFM 把 curriculum 直接 embed 到 reward function 而非 reward shaping, 更稳定。

### 4. Sim-to-real 的关键 enabler

Student 通过 10 帧 proprioceptive history 估计 latent $\hat{z}$, 这让 policy 在 real world 推断 friction / payload / terrain 而无需传感器直接测量。LIO 提供 base world pose (因为 proprioceptive 无法观测 global position), 但 arm EE pose 用 FK 从 joint 算, 完全 proprioceptive。

### 5. Limitation 与未来方向

paper 自己承认: 没有 perceptive input, 在 confined space / obstacle 场景受限。未来必然要加 vision (depth / RGB), 类似 Visual WBC [19] 或 DTC [5] 的 perceptive locomotion 思路。可能的扩展:
- Vision latent 也通过 privilege estimator 注入, 替代或补充 $z$
- RFM 可扩展到更多 phase (如 grasp, lift, place), 用 multi-stage $\mathcal{D}$ 函数链
- 与 diffusion policy 结合, 把 RFM 作为 reward shaping for diffusion-based loco-manipulation

---

## VII. 总结

这篇 paper 的核心贡献是把 model-based WBC 的 hierarchy/null-space/priority 思想, 用 **非线性乘法 reward fusion** 移植到 RL。三个机制各司其职:
- **RP**: 任务间 hierarchy (position 先, orientation 后; regularization 最高)
- **Enhancement**: 精度 (micro sharpening near goal + integral-like penalty)
- **Loco-Mani Fusion**: phase-driven smooth mode transition

整套系统的优雅之处在于: 用极少 hyperparameter 表达复杂 multi-stage objective, 同时达到 SOTA 精度 (5cm / 0.1rad)。RFM 的思想可推广到任何 multi-stage whole-body control 问题 (humanoid, bimanual manipulation, multi-agent coordination), 是 RL reward design 的重要方法论进步。

如果你想 build deeper intuition, 建议复现 ablation 中的 w/o LMF 和 w/o RP — 这两个 case 的 failure mode (jittery motion 和 dangerous posture) 最能说明 RFM 各组件的不可替代性。
