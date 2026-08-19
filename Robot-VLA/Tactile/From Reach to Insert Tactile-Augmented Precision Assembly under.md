---
source_pdf: From Reach to Insert Tactile-Augmented Precision Assembly under.pdf
paper_sha256: 3ff08430fb826521b802d82bb5f6b64577388cd5386775867421ce57421abfd5
processed_at: '2026-08-19T08:20:06-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Paper

## 一句话总结

教 robot 干一件特别精细的活儿——把一个金属柱子插进一个比它大不到 0.05 mm 的洞里，而且还得先自己把柱子捡起来。

## 这个事儿到底有多难

0.05 mm 是什么概念？你拔一根头发下来，头发直径大概 0.07 mm。这个洞比头发还细。

你让 robot 抓个柱子往洞里塞，clearance 这么小，只要 position 偏了一丁点——比如 1 mm——柱子就卡在洞口，然后 robot 还继续往下压，force 瞬间飙到 50 N，工件可能就废了。

更烦的是，这篇 paper 的 setup 是 **grasp-to-insert**：peg 不是预先固定好在 gripper 里的，是随机扔在桌面上 5×5 cm 范围内，robot 得先自己抓起来。你抓 peg 的时候稍微偏一点，peg 在手指里就是歪的，这个歪斜会一路累积到 insertion 阶段被放大。

之前大部分 peg-in-hole paper 都作弊——peg 已经固定好、对齐好了，robot 只要往下插就行。这篇 paper 做的是更接近真实工业场景的版本。

## 人会怎么做这件事

你闭上眼睛想一下自己怎么把钥匙插进锁孔：
1. 先用眼睛看着，手拿着钥匙往锁孔附近凑——这阶段靠 vision
2. 钥匙碰到锁孔边沿了，你看不见里面什么情况，靠手感调整——这阶段靠 tactile

人天然就是两段式的。这篇 paper 的核心 insight 就是：**让 robot 也分两段干活**。

## 两段式的设计

### Reach 阶段：用眼睛

用 Diffusion Policy（就是现在很火的那个把 action 当 image 生成的 IL 方法）学一个"抓起来凑过去"的策略。输入是 camera image + robot 自己的 pose，输出是未来 16 步的 action sequence。100 条 human demonstration 就能训出来，能 generalize 到 5×5 cm 的随机初始位置。

这一段没 contact，vision 够用，IL 够用。

### Insert 阶段：靠手感

一旦 peg 到了洞口附近，切换到 RL（SAC）。这一段有 contact，vision 基本废了——peg 和 hole 的 misalignment 是 sub-mm 级别的，camera 根本看不见，但 tactile sensor 能感觉到。

为什么 insert 不用 IL？因为 IL 没有 recovery 能力。peg 卡住了，IL policy 就懵了——它没见过这种状态，out of distribution。RL 能从失败中学习，能探索 recovery 策略。

## 这篇 paper 真正的 contribution

SAC、Diffusion Policy 都是现成的，不是这篇 paper 发明的。这篇 paper 做的是几个针对这个具体问题的 engineering trick，每个都挺巧妙：

### Trick 1: Tactile baseline subtraction

你抓 peg 的时候，抓偏一点，tactile sensor 就有个 static offset——比如 peg 歪了，torque sensor 一直有个读数。但这个读数跟 insertion 成不成功没关系，只是 grasp 的 artifact。

所以作者算了一个 **tactile baseline**：抓完之后静止那几帧的 tactile 平均值 $\bar{\mathbf{f}}$。然后训练时用的不是 raw tactile，而是 $\Delta \mathbf{f} = \mathbf{f} - \bar{\mathbf{f}}$——减掉 grasp offset，只保留 insertion 过程中真正因 contact 产生的变化。

这就像你称体重之前先去皮——tare。简单但关键。

### Trick 2: Tactile 只给 critic，不给 actor

这是我觉得最聪明的设计。**Actor 只看 vision，critic 额外看 tactile**。

为什么？tactile signal 的方差太大了——不同 grasp pose 下 raw tactile 差异巨大。如果把这个高方差 signal 直接喂给 actor，actor 的 action distribution 会被扰乱，训练不稳定。

但 critic 的任务只是估计 value，不需要输出 action，可以容忍更高维的 input。Tactile 给 critic 提供了 contact-aware 的信息，让 value estimate 更准。Actor 通过 policy gradient 从 critic 那里间接学到正确的行为，但自己不用直接处理 noisy 的 tactile。

这跟 privileged information 只给 critic 的思路一样，参考 Pinto et al. 的 Asymmetric Actor Critic (https://arxiv.org/abs/1802.09477)。

### Trick 3: Tactile Group Sampling

RL 训练时有个问题：replay buffer 会被某些 grasp configuration dominate。比如 policy 早期总是用某个固定 pose 抓 peg，buffer 里全是这个 pose 的 transition，其他 pose 的数据太少。

解决方案：把 transition 按 grasp 后的 torque $(\bar{M}_x, \bar{M}_y)$ 分成 2×4=8 个 group。每次采样时先 uniform 选 group，再在 group 里 uniform 选 transition。这样稀有 grasp configuration 被过采样，dominant configuration 被欠采样，replay 分布更均衡。

这个 idea 跟 Prioritized Experience Replay (https://arxiv.org/abs/1511.05952) 思路类似，但 PER 用 TD error 做 priority，这里用 grasp configuration 做 group，更 targeted。

### Trick 4: Critic Warmup

Sparse reward（只有成功才给 reward=1）下，早期 critic 的 TD bootstrap 会有 huge variance——critic 自己估值不准，用不准的估值做 target 训自己，garbage in garbage out。

所以先用 demonstration data 做 supervised pretraining：直接让 critic regression 到 Monte Carlo return $G$。bypass bootstrap error，给 critic 一个 grounded 的 value surface。

Loss 从 0.503 降到 0.002，predicted Q value 对齐到 demo 成功率 0.94。这个 initialization 让后续 online RL 稳很多。

### Trick 5: Location-based Dynamic Switching

Reach 和 insert 之间什么时候切换？用一个 AABB（axis-aligned bounding box）来定义 switch region $\mathcal{X}_{\mathrm{init}}$。这个 box 是所有 demo 终点和 failure 点的并集。

随着训练进行，failure 点会扩展 box，让 policy 在更大区域里学 recovery。但同时做 outlier rejection 防止异常 termination 把 box 撑爆。

## 结果怎么样

最关键的 0.05 mm clearance 下：

| Method | 成功率 |
|--------|--------|
| 纯 IL | 0% |
| IL+RL (无 tactile) | 52% |
| IL+RL (full, with tactile critic + group sampling) | 67% |

而且 full method 把 peak force 从 50 N 压到 20 N（-60%），torque 从 160 N·mm 压到 90 N·mm（-44%）。

训练时间：30 分钟 human-in-the-loop，大约 1000-2000 transitions。这在 real-robot RL 里是惊人的少。

## 为什么 tactile 在 tight tolerance 下才显现价值

看 1.5 mm clearance 下，IL+RL 和 full method 成功率几乎一样（都是 ~97%）。说明大 clearance 下 vision + position 足够，不需要精细 tactile。

到 0.25 mm，full 比 IL+RL 高 15%。到 0.05 mm，又高 15%。

**Tactile 的价值在 contact-rich + tight tolerance 下才显现。** 这给了个重要 intuition：如果你做的任务 clearance 不小，别费劲搞 tactile 了，vision 够用。

## 一个特别 elegant 的副作用：Implicit Force Minimization

Full method 没有 explicit 的 force penalty term，但 force 自动降了 60%。为什么？

因为 critic 通过 tactile 信号"感知到"即将发生的高 force 状态，给这些 state 低 value。Policy gradient 推动 actor 远离高 force action。Actor 本身不直接看 tactile，但通过 critic 的 value signal 学到了 force-aware behavior。

Implicit force minimization，very elegant。

## 我的 overall take

这篇 paper 没发明任何新算法。SAC 2018 年的，Diffusion Policy 2023 年的，RLPD 2023 年的，HiL-RL 2024 年的。它做的是把这些 building blocks 合理组合，加了几个 targeted trick，解决了一个具体问题。

但这恰恰是当前 robotics 最需要的——**system-level innovation 比 algorithm-level innovation 更有价值**。RL 在 robotics 里的瓶颈从来不是算法本身，而是怎么把它们 glue 赚来处理真实世界的 messiness。

几个 trick 的 pattern 应该可以 transfer 到其他 contact-rich 任务：connector insertion、bolt threading、snap-fit assembly。Tactile as critic-side auxiliary 这个 idea 尤其有意思——未来 VLA 要做 precision assembly 的话，tactile 也许应该走 critic 路线而不是 actor 路线。

## 相关链接

- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- SAC: https://arxiv.org/abs/1801.01290
- RLPD: https://arxiv.org/abs/2306.00909
- HiL-RL (Luo et al. Science Robotics): https://www.science.org/doi/10.1126/scirobotics.adh5678
- Asymmetric Actor Critic: https://arxiv.org/abs/1802.09477
- IndustReal: https://arxiv.org/abs/2305.17110
- AutoMate: https://arxiv.org/abs/2404.24430
- PER: https://arxiv.org/abs/1511.05952
- VLA-Touch: https://arxiv.org/abs/2507.17294

---

# From Reach to Insert: Tactile-Augmented Precision Assembly 深度讲解

Andrej 你好，这篇paper我觉得在robotic assembly这个细分领域里挺有代表性的——它把"工业高精度装配"这个老问题，用现代IL+RL stack重新做了一遍，并且做了一个非常合理的engineering trade-off。下面我从intuition出发，逐层拆解。

## 1. 这篇paper想解决的真正问题

工业assembly里有个经典的sub-millimeter peg-in-hole问题：clearance c < 0.1 mm，公差甚至到 0.05 mm。这个数字意味着什么？人头发丝直径大约 70 μm = 0.07 mm，所以 0.05 mm 比一根头发还细。在这种尺度下，position error 几个 mm 就会导致peg卡死在hole入口，interaction force 几十牛顿瞬间产生，可能损坏工件或robot。

更关键的设置是 **grasp-to-insert**：peg 是随机放在 5×5 cm 区域里的，robot 必须先 grasp，再做 insertion。这跟传统"peg已经预对齐固定在gripper里"的setup完全不同。grasp 引入的 in-hand pose error 会累积，然后在 contact 阶段被急剧放大。Fig. 2 把 prior works 按 task setting 和 tolerance 两个维度做了划分，本文落在最难的那一格。

参考一下相关工作做对比：
- IndustReal (RSS 2023): https://arxiv.org/abs/2305.17110 — sim-to-real，但不一定 grasp-to-insert
- FORGE (RA-L 2025): https://arxiv.org/abs/2501.00786 — force-guided exploration
- Human-in-the-loop RL (Science Robotics 2024): https://www.science.org/doi/10.1126/scirobotics.adh5678 — Luo et al. 的 HiL-RL 框架，本文明显受其影响
- AutoMate (RSS 2024): https://arxiv.org/abs/2404.24430

## 2. 核心intuition：为什么要 decompose 成 Reach + Insert？

人做这种装配动作时的认知过程其实可以拆成两段：
- **Reach**: 用 vision 做 open-loop 抓取和接近——这个阶段没有 contact，vision 提供的 global geometry 已经够用
- **Insert**: 一旦 contact 发生，vision 基本失效（occlusion、sub-mm misalignment 看不见），必须切到 tactile feedback 做高频 reactive correction

作者很聪明地把这两段对应到两种 learning paradigm：
- Reach 用 **Diffusion Policy (IL)** —— sample efficient，能 generalize 到 5×5 cm 的初始位置分布
- Insert 用 **SAC (RL)** —— 能学习 contact-rich recovery 行为，从失败中学习

这是 **divide and conquer** 的思路：把 exploration space 切小。如果整个 grasp-to-insert 都用 RL 做，10 Hz 控制频率下从 5 cm 外探索到 sub-mm 对齐，sample complexity 会爆炸。Reach policy 先把 peg 送到 hole 附近的 "init region" $\mathcal{X}_{\mathrm{init}}$，RL 只在这个小区域里学 fine-grained 的策略。

这个思想跟 Residual RL (Johannink et al. 2019, https://arxiv.org/abs/1812.08254) 有相似之处，但本文不是 residual，而是 hard switch。

## 3. Reach Stage: Diffusion Policy 详解

公式 (1):

$$a_{t:t+H-1} \sim \pi_{\mathrm{reach}}(\cdot \mid o_t^{\mathrm{reach}})$$

变量解释：
- $a_{t:t+H-1}$: 从时刻 $t$ 开始，长度为 $H$ 的 action sequence（不是 single action）
- $o_t^{\mathrm{reach}}$: reach stage 的 observation，包含 end-effector pose + multi-view images（wrist camera + global camera）
- $H = 16$: action horizon，即一次推理出未来 16 步动作
- $T = 2$: observation horizon，用过去 2 帧观测

这里用的是 **receding-horizon control**（也叫 model predictive control 思想）：每次推理出 16 步，但只执行前几步，然后重新观测、重新推理。这样可以平滑 trajectory，同时保持 reactive 能力。

Diffusion Policy 本质上是把 action sequence 当成 image 来生成——用 DDPM 在 action 空间做去噪。原paper: https://diffusion-policy.cs.columbia.edu/，Chi et al. RSS 2023。

每个 geometry 训 100 demos、200 epochs、batch size 64，encoder 是 ResNet-18 处理 128×128 图像。这个数据量在工业 assembly 里算少的，但 reach 阶段任务相对简单所以够用。

## 4. Insert Stage: 这是本文真正的核心贡献

Insert stage 被建模为 POMDP $\mathcal{M} = \{\mathcal{O}, \mathcal{A}, \rho, \mathcal{P}, r, \gamma\}$。POMDP 而非 MDP 是因为 tactile 和 vision 都有 partial observability——你不能直接观测到 peg 和 hole 之间的精确相对位姿。

### 4.1 Reward 设计

公式 (2):

$$r_t = \begin{cases} 1, & \text{if insertion succeeds} \\ 0, & \text{otherwise} \end{cases}$$

注意这是 **sparse binary reward**，没有 dense shaping。这是设计选择：dense reward 容易让 policy 学到 reward hacking 行为（比如只往下压不真的 align）。但 sparse reward 带来的问题就是 sample efficiency 差，所以后面要 critic warmup + tactile group sampling 来补救。

### 4.2 SAC backbone

公式 (3) critic loss:

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(o,a,r,o',d)}\left[(Q_\phi(o,a) - y)^2\right]$$

公式 (4) target value:

$$y = r + \gamma(1-d)\mathbb{E}_{a' \sim \pi_\theta(\cdot|o')}\left[Q_{\bar{\phi}}(o',a') - \alpha \log \pi_\theta(a'|o')\right]$$

变量解释：
- $\phi$: critic network 参数
- $\bar{\phi}$: target critic 参数，用 Polyak average 慢更新
- $\theta$: actor 参数
- $\alpha$: temperature coefficient，控制 entropy 权重（SAC 的核心：max entropy RL）
- $d \in \{0,1\}$: terminal flag，episode 是否结束（成功 insertion 或者 abort）
- $\gamma$: discount factor
- $(1-d)$: mask，终止状态不 bootstrap 未来 value

公式 (5) actor loss:

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_o\left[\mathbb{E}_{a \sim \pi_\theta(\cdot|o)}[\alpha \log \pi_\theta(a|o) - Q_\phi(o,a)]\right]$$

这其实就是 standard SAC。SAC 原paper: https://arxiv.org/abs/1801.01290，Haarnoja et al. ICML 2018。

### 4.3 RLPD training pipeline

作者用了 **RLPD (Reinforcement Learning with Prior Data)**，Ball et al. ICML 2023: https://arxiv.org/abs/2306.00909。两个 replay buffer：
- $\mathcal{D}_{\mathrm{demo}}$: demonstration/prior buffer，包含初始 demos 和 human intervention 数据
- $\mathcal{D}_{\mathrm{on}}$: online buffer，policy 自己 roll 出来的数据

每次 update 从两个 buffer 里大约等比例采样组成 mini-batch。这个 mixing ratio 很重要——prior data 防止 policy 漂移到 bad region，online data 让 policy 能 explore 新策略。

### 4.4 Asymmetric Actor-Critic with Tactile Critic

这是本文我觉得最有意思的设计。**Actor 只用 vision，critic 额外用 tactile**。

为什么这样设计？intuition 是这样的：
- Tactile signal 高度依赖于 grasp pose——你 grab peg 的位置稍微偏一点，初始 torque $M_x, M_y$ 就完全不一样
- 如果把这个高方差 signal 直接喂给 actor，actor 的 policy 分布会被 grasp-dependent offset 扰乱，训练不稳定
- 但 critic 的任务是估计 value，不需要输出 action，可以容忍更高的 input dimensionality。Tactile 给 critic 提供了 contact-aware 的额外信息，让 value estimate 更准确
- Actor 通过 policy gradient 从 critic 那里"学到"正确的视觉行为，但不需要直接处理 tactile

这跟 asymmetric actor-critic 在 navigation 里的用法类似——Pinto et al. 的 Asymmetric Actor Critic (https://arxiv.org/abs/1802.09477)：privileged information 只给 critic 用。

### 4.5 Tactile Baseline 和 Tactile Change

公式 (8):

$$\bar{\mathbf{f}} = \frac{1}{K}\sum_{k=0}^{K-1} \mathbf{f}_{t_b+k}$$

变量解释：
- $\bar{\mathbf{f}} = [F_x, F_y, F_z, M_x, M_y, M_z]^\top$: 6-DoF force/torque 向量的均值
- $t_b$: gripper 闭合后稳定窗口的起始 index
- $K$: 窗口长度
- $\mathbf{f}_t$: 时刻 $t$ 的 raw tactile reading

然后定义 tactile change:

$$\Delta \mathbf{f}_t = \mathbf{f}_t - \bar{\mathbf{f}}$$

这个 subtraction 非常关键。它的物理意义是：**减掉 grasp 引入的 static offset，只保留 insertion 过程中动态产生的 contact variation**。

想象你握住一个 peg——如果握偏了，peg 在 gripper 里是倾斜的，静态下 torque sensor 就有读数。但这个读数跟 insertion 是否成功无关，只是 grasp artifact。如果你不 subtract，critic 会把这个 grasp offset 也学进去，导致 value estimate 高度依赖于 grasp configuration。

Fig. 4 直观展示了这一点：不同 grasp pose 下 raw tactile 差异巨大，但 baseline-subtracted signal 都能 converge 到一致的 contact pattern。

### 4.6 Critic Warmup with Monte Carlo target

公式 (9):

$$\mathcal{L}_{\mathrm{MC}}(\phi) = \mathbb{E}_{(s,a,G) \sim \mathcal{D}_{\mathrm{demo}}}\left[(Q_\phi(s,a) - G)^2\right]$$

变量解释：
- $G$: 从 demonstration trajectory 预计算的 Monte Carlo return，即 $\sum_t \gamma^t r_t$
- $(s, a, G)$: 从 demo buffer 采样的 transition 配上 MC return

这是 **supervised pretraining**，不是 TD learning。为什么要这样做？因为 sparse reward + 早期 critic 的 TD bootstrap 会有 huge variance——critic 自己估值不准，用不准的估值做 target 再训自己，就是 garbage in garbage out。

直接 regression 到 MC return 绕过了 bootstrap error，给 critic 一个 grounded 的 value surface。Paper 里提到 critic loss 从 0.503 降到 0.002，predicted $Q_{\mathrm{mean}}$ align 到 $G \approx 0.94$（demo 成功率 ~94%）。

这个 trick 在 offline RL 里有类似思想，比如 CQL (https://arxiv.org/abs/2006.04779)、AWAC (https://arxiv.org/abs/2006.09359)，但本文用的是最直接的 MC regression。

### 4.7 Tactile Group Sampling

这是另一个我觉得很精巧的设计。问题：off-policy training 时 replay buffer 会被某些 grasp configuration dominate——比如 policy 早期总是用某个固定 pose 抓 peg，所以 buffer 里这个 pose 的 transition 占绝大多数。

解决方案：把 transition 按 grasp 后的 baseline torque $(\bar{M}_x, \bar{M}_y)$ 分组。

公式 (10):

$$g(\bar{\mathbf{f}}) = \big(b_x(\bar{M}_x), b_y(\bar{M}_y)\big)$$

变量解释：
- $b_x(\cdot), b_y(\cdot)$: 把 $\bar{M}_x, \bar{M}_y$ 离散化的 binning function
- $N, M$: 两个轴上的 bin 数（本文 $2 \times 4 = 8$ 个 group）

训练时先 **uniformly sample group**，然后在选中的 group 里 uniform sample transition。这等于在 replay 上做了 **importance reweighting**——稀有 grasp configuration 被过采样，dominant configuration 被欠采样。

为什么选 $(\bar{M}_x, \bar{M}_y)$ 而不是 6 维全部？因为这两个 torque component 对 grasp pose 最敏感（peg 在 gripper 里倾斜主要反映在 $M_x, M_y$ 上，不是 $F_z$）。

这个 idea 跟 Prioritized Experience Replay (https://arxiv.org/abs/1511.05952) 思路类似，但 PER 用 TD error 做 priority，这里用 grasp configuration 做 group，针对 tactile imbalance 这个具体问题。

效果在 Fig. 8 里：Q-value disagreement 比baseline 低很多——说明 value estimation 在不同 grasp mode 下更一致。

### 4.8 Location-based Two-stage Switching

公式 (6):

$$I_{\mathrm{ins}}(\mathbf{p}) = \begin{cases} 1, & \mathbf{p} \in \mathcal{X}_{\mathrm{init}} \\ 0, & \text{otherwise} \end{cases}$$

公式 (7):

$$\pi(a|o) = (1 - I_{\mathrm{ins}}(\mathbf{p}))\pi_{\mathrm{reach}}(a|o) + I_{\mathrm{ins}}(\mathbf{p})\pi_{\mathrm{insert}}(a|o)$$

变量解释：
- $\mathbf{p} \in \mathbb{R}^3$: end-effector position
- $\mathcal{X}_{\mathrm{init}}$: 一个 AABB (axis-aligned bounding box)
- $\mathcal{P}_{\mathrm{demo}}$: demo 轨迹的 end points 集合
- $\mathcal{P}_{\mathrm{fail}}$: 失败 episode 的 end points 集合
- $\mathcal{X}_{\mathrm{init}} = \mathrm{AABB}(\mathcal{P}_{\mathrm{demo}} \cup \mathcal{P}_{\mathrm{fail}})$: 包含所有 demo 终点和失败点的 bounding box

这是个 **dynamic 的 switch region**——随着训练进行，failure point 会扩展 bounding box，让 policy 在更大的区域里学习 recovery。但同时用 statistical outlier rejection 防止异常 termination 把 box 撑爆。

很 elegant 的 engineering：既给了 policy recovery 的余地，又防止了 exploration 空间爆炸。

## 5. Algorithm 1 流程解析

让我把 Algorithm 1 的逻辑过一遍：

1. 输入 reach demos 和 insertion demos
2. 初始化两个 policy 和三个 buffer（reach demo, insert demo, online）
3. 从 demo 终点构造初始 switch region $\mathcal{X}_{\mathrm{init}}$
4. **Critic Warmup**: 用 MC loss 在 demo buffer 上预训 critic
5. 用 diffusion policy 训 reach policy
6. 主循环：
   - 执行 reach policy，记录 tactile baseline $\bar{\mathbf{f}}$，更新 $\mathcal{X}_{\mathrm{init}}$
   - 当 end-effector 进入 $\mathcal{X}_{\mathrm{init}}$ 时，切换到 insert policy
   - 每个 insertion attempt：
     - 如果有 failure point，从 $\mathcal{P}_{\mathrm{fail}}$ 重置（curriculum-like）
     - 执行 RL action（或 human intervention action），存到 buffer
     - 每 U 步用 tactile group sampling 训 insert policy
   - 失败时把 end point 加到 $\mathcal{P}_{\mathrm{fail}}$，更新 $\mathcal{X}_{\mathrm{init}}$

这里的 $\mathcal{P}_{\mathrm{fail}}$ reset 是个 implicit curriculum——policy 反复在失败的 configuration 上练习，类似 self-imitation 或者 adversarial curriculum。

## 6. 实验数据表深度解读

Table I 是核心结果，我把它重新整理一下方便看intuition：

| Clearance | Method | Square | Round | Hex | L-shape | Tri | Avg |
|-----------|--------|--------|-------|-----|---------|-----|-----|
| 1.5 mm | IL | 10/15 | 13/15 | 10/15 | 11/15 | 13/15 | 76% |
| 1.5 mm | IL+RL | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 | ~97% |
| 1.5 mm | Full | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 | ~97% |
| 0.25 mm | IL | 4/15 | 2/15 | 4/15 | 2/15 | 6/15 | 24% |
| 0.25 mm | IL+RL | 12/15 | 8/15 | 12/15 | 13/15 | 14/15 | 79% |
| 0.25 mm | Full | 14/15 | 14/15 | 14/15 | 14/15 | 15/15 | ~94% |
| 0.05 mm | IL | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0% |
| 0.05 mm | IL+RL | 8/15 | 6/15 | 8/15 | 5/15 | 12/15 | 52% |
| 0.05 mm | Full | 11/15 | 7/15 | 10/15 | 8/15 | 14/15 | 67% |

几个值得注意的pattern：

**(a) IL 在 0.05 mm 完全崩溃**。0% 成功率。这印证了 IL 缺乏 recovery 能力的根本缺陷——一旦 contact 后 peg 在 gripper 里 drift，policy 就 OOD 了。

**(b) Tactile augmentation 在 tight tolerance 下增益最大**：
- 1.5 mm: IL+RL vs Full 几乎没区别（97% vs 97%）——大 clearance 下不需要精细 tactile
- 0.25 mm: Full 把 IL+RL 从 79% 拉到 94%（+15%）
- 0.05 mm: Full 把 IL+RL 从 52% 拉到 67%（+15%）

这给了我们一个重要intuition：**触觉的价值在 contact-rich + tight tolerance 下才显现**。大 clearance 下 vision + position 足够。

**(c) Geometry 有显著差异**。Triangular 在所有方法下都最容易（IL 在 0.25 mm 都有 6/15，Full 在 0.05 mm 有 14/15）；Square、Round、L-shape 在 0.05 mm 比较难。这可能跟 geometry 的 chamfer 设计、self-aligning 特性有关——triangular hole 有斜边，peg 滑进去更自然；square hole 没有这种 funnel effect。

**(d) 全部方法在 1.5 mm 下都接近 ceiling**。说明 1.5 mm 这个 regime 已经"被解决"了，研究价值有限。这也呼应了 Fig. 2 里 prior work 的分布——大部分在 > 1 mm。

## 7. Safety / Interaction Force 分析

Fig. 9 的数据很有意思：
- Vanilla IL+RL: peak normal force > 50 N, torque variation > 160 N·mm
- Full method: peak normal force < 20 N, torque variation < 90 N·mm
- 减少：force 60%, torque 44%

为什么 tactile critic 能降低 force？intuition：critic 通过 tactile 信号"感知到"即将发生的高 force 状态，给这些 state 低 value。Policy gradient 推动 actor 远离高 force action。Actor 本身不直接看 tactile，但通过 critic 的 value signal 学到了 force-aware behavior。

这是个很漂亮的 **implicit force minimization**——没有显式的 force penalty term，但通过 critic 的 value estimation 自然涌现了 low-force policy。

参考 Reactive Diffusion Policy (Xue et al. RSS 2025, https://arxiv.org/abs/2504.20865) 用 slow-fast visual-tactile policy，思路有相通之处。

## 8. 训练效率

- 30 分钟 human-in-the-loop training 就达到 zero-intervention success
- ~1000-2000 environment transitions
- 4000 transitions 后 vanilla IL+RL 也收敛
- Tactile group sampling 把首次 zero-intervention success 提前到 ~1000 transitions

这个数据量在 real-robot RL 里是惊人的少。对比一些 reference：
- SAC 原paper 在 gym 里要 millions of steps
- Luo et al. Science Robotics 的 HiL-RL 也要几千 transitions

少的原因有几方面：
1. RLPD prior data 加持
2. Critic warmup 给了好的初始化
3. Tactile group sampling 加速收敛
4. Switch region 限制了 exploration space
5. Human intervention 提供高质量 demonstration

## 9. 跟相关工作的联系和我的联想

### 9.1 跟 Diffusion Policy 的关系
Reach stage 直接用了 Diffusion Policy。但作者没有在 insert stage 用 Diffusion Policy——因为 insert 需要 contact-rich 的 reactive behavior，diffusion 的 multi-step denoising 比较慢，而且 IL 缺乏 recovery。这是合理的 paradigm 选择。

最近有一些工作尝试把 diffusion 用到 contact-rich：TacDiffusion (Wu et al. ICRA 2025, https://arxiv.org/abs/2412.01978)，Reactive Diffusion Policy (https://arxiv.org/abs/2504.20865)。但本文选择了 SAC，更成熟、更快。

### 9.2 跟 VLA / Tactile-LM 的关系
最近一波 VLA 模型（OpenVLA, RT-2, Octo）开始处理 manipulation，但都是 coarse manipulation。Precision assembly 这种 sub-mm tolerance 任务，VLA 的 vision-only 输入不够。有几个工作开始把 tactile 加到 VLA：
- VLA-Touch (https://arxiv.org/abs/2507.17294)
- TLA: Tactile-Language-Action (https://arxiv.org/abs/2503.08548)
- ManipLLM (https://arxiv.org/abs/2312.09358)

本文的 asymmetric actor-critic 设计（vision 给 actor，tactile 给 critic）其实给了一个有趣的方向：**tactile 不一定要进 actor**。如果未来 VLA 要做 precision assembly，也许 tactile 应该作为 critic-side 的 auxiliary signal，而不是 actor input。

### 9.3 跟 Sim-to-Real 的关系
本文完全在 real robot 上训练，30 分钟就收敛。这是非常 impressive 的工程成就。但对比 IndustReal (https://arxiv.org/abs/2305.17110)、VT-Refine (https://arxiv.org/abs/2506.10950) 这些 sim-to-real 路线，本文放弃了大规模 sim training 的优势，换来的是：
- 不需要建精确的 tactile sim（这是出了名的难，tactile sensor 的 contact model 很复杂）
- 不需要 domain adaptation
- 直接在真实 sensor 上学

但代价是：generalization 受限。每种 geometry 都要单独训 reach policy（100 demos each），整个 pipeline 要重跑。如果要做 100 种 geometry，scalability 是问题。

### 9.4 跟 In-Hand Manipulation 的联想
In-hand manipulation 也有类似的"grasp introduces variance"问题。Surprisingly Robust In-Hand Manipulation (Bhatt et al. RSS 2021, https://robot-learning.cs.utexas.edu/primitive-robustness/) 用 funnels 来吸收 uncertainty。OpenAI 的 Rubik's cube (https://arxiv.org/abs/1910.07113) 用 sim-to-real + domain randomization。

本文的 tactile baseline subtraction 思路——**减掉 grasp-dependent offset**——其实在 in-hand manipulation 里也适用。比如 Rubik's cube 解魔方时，gripper 握住 cube 的姿态会引入 sensor offset，这个 offset 跟 cube 解算状态无关，应该 subtract 掉。

### 9.5 跟 Active Inference 的关系
Tactile-based Active Inference (Kamijo et al. https://arxiv.org/abs/2309.15681) 用 free energy principle 做 peg-in-hole。Active inference 把 control 和 perception 统一在 variational free energy 最小化下。本文的 critic 估计 value 其实跟 active inference 里的 value estimation 有形式上的相似——都是在估计 "未来状态有多好"。但 active inference 还有 generative model 部分，本文是 model-free。

### 9.6 跟 Force/Torque Control 经典方法的对比
经典 Whitney 1982 peg-in-hole（https://dspace.mit.edu/handle/1721.1/5794）、 impedance control (Hogan 1985)、hybrid position/force control——这些方法在 structured environment 下很有效，但需要精确的 contact model 和几何先验。本文的 RL 方法 essentially 学到了一个 implicit 的 impedance policy——从 Fig. 9 看，policy 自动 regulate 了 force，但这个 impedance 是 learned 不是 designed。

这其实跟 Learning Variable Impedance Control (Zhang et al. RA-L 2021, https://arxiv.org/abs/2010.04157)、Reinforcement Learning on Variable Impedance (Luo et al. ICRA 2019, https://arxiv.org/abs/1903.08965) 思路相通——把 impedance parameter 作为 policy 输出，让 RL 学习怎么调。

### 9.7 Tactile Sensor Hardware
用的是 Paxini PX-6AX GEN3，6-axis force/torque tactile sensor。这是个比较新的高精度 sensor。其他常用 tactile sensor：
- GelSight (MIT, https://people.csail.mit.edu/mkusper/gelsight.shtml) — optical tactile，高 resolution
- DIGIT (Meta, https://digit.ml/) — small optical tactile
- BioTac (Syntouch) — fluid-filled, multi-modal
- TacTip (Bristol) — optical

Paxini 提供 6-axis F/T，跟 GelSight 这种 high-res image-based sensor 不同。本文用的是低维 signal (6D)，这也解释了为什么 critic 可以直接 raw 输入。如果是 GelSight image，可能需要先做 representation learning（参考 Dexterity from Touch, https://arxiv.org/abs/2303.12076）。

### 9.8 Critic Warmup 跟 Offline RL 的联系
Critic warmup 用 MC regression，跟 BCQ (https://arxiv.org/abs/1812.02900)、CQL (https://arxiv.org/abs/2006.04779)、IQL (https://arxiv.org/abs/2110.06169) 这些 offline RL 方法有相似之处——都用 prior data 给 value function 一个 grounded 初始化。但本文不是 offline RL，是 online RL with warmup。

最近的 AWAC (https://arxiv.org/abs/2006.09359) 思路更接近：用 prior data 通过 advantage-weighted regression 引导 policy，然后 online fine-tune。

### 9.9 跟 HiL-RL 的关系
Human-in-the-loop RL 的开山之作是 Luo et al. Science Robotics 2024 (https://www.science.org/doi/10.1126/scirobotics.adh5678)。本文明显借鉴了这个思路——operator 用 SpaceMouse 在 policy 进入 unrecoverable state 时 takeover，这些 corrective action 存到 demo buffer。

HiL-RL 的核心 insight 是：human intervention 不是 "training wheels"，是 high-quality demonstration。本文把这个 idea 用到了 precision assembly 这个具体场景。

## 10. 我会问作者的问题 / 局限性

**1. Per-geometry training 的 scalability**。每种 geometry 要 100 demos + 30 min RL。如果工业场景有 50 种 peg shape，成本累积。

**2. Tactile sensor calibration**。Paxini 这种 sensor 的 reading 会随温度、湿度、老化漂移。Tactile baseline subtraction 能补偿 session 内的 grasp offset，但不能补偿跨 session 的 sensor drift。

**3. Insertion depth 限制**。Paper 没说 insertion depth 是多少。如果是浅 insertion（几 mm），比较容易；深 insertion（cm 级）需要持续 tactile feedback，policy 可能不同。

**4. Reset mechanism**。Real-robot RL 的 reset 成本很高。Paper 没详细说怎么 reset——是 human 手动 reset 还是 automated？这影响 30 min training 的实际 wall-clock 时间。

**5. Asymmetric actor-critic 在推理时的优势**。Actor 不需要 tactile input，意味着 deployment 时不需要 tactile sensor 吗？看起来不行，因为 actor 还是用 critic 训出来的，tactile 通过 value signal 影响了 actor 的学习。但 deploy 时如果 tactile 失效，actor 行为是否会退化？这点 paper 没明确讨论。

**6. Comparison with model-based hybrid control**。没跟经典的 spiral search、impedance control 做直接 baseline 比较。可能因为 0.05 mm 下这些方法都失败，但 0.25 mm 应该可以比。

**7. Generalization 到 novel geometry**。每种 geometry 训一个 reach policy。Insert policy 是否能 zero-shot transfer 到没训过的 shape？这是工业应用的关键问题。AutoMate (https://arxiv.org/abs/2404.24430) 在这点上做得更好。

## 11. 整体评价

这篇 paper 是个很扎实的 systems paper。它没发明新的 RL 算法，没发明新的 sensor，没发明新的 IL method——它把已有的 building blocks（Diffusion Policy, SAC, RLPD, HiL-RL, asymmetric actor-critic）组合得非常合理，针对 precision assembly 这个具体问题做了几个 targeted 改进：
- Tactile baseline subtraction（处理 grasp offset）
- Tactile critic（asymmetric design）
- Tactile group sampling（处理 replay imbalance）
- Critic warmup with MC（处理 sparse reward bootstrap error）
- Location-based dynamic switch（implicit curriculum）

每个 trick 单独看都不novel，但组合在一起让 0.05 mm 的 grasp-to-insert 在 30 分钟 real-robot training 下达到 67% 成功率，并且把 peak force 从 50 N 压到 20 N——这个结果在两年前是不可想象的。

我觉得这种 **"system-level innovation beats algorithm-level innovation"** 的风格正是当前 robotics 需要的。RL 在 robotics 里的瓶颈从来不是算法本身——SAC 2018 年就有了，Diffusion Policy 2023 年就有了——而是怎么把它们 glue 起来处理真实世界的 messiness。这篇 paper 给了一个很好的 case study。

对未来工作的 hint：tactile as critic-side auxiliary、dynamic task decomposition、MC warmup for sparse reward RL——这些 patterns 应该可以 transfer 到其他 contact-rich 任务（connector insertion、bolt threading、snap-fit assembly）。

## 参考链接汇总

- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- SAC: https://arxiv.org/abs/1801.01290
- RLPD: https://arxiv.org/abs/2306.00909
- HiL-RL (Luo et al.): https://www.science.org/doi/10.1126/scirobotics.adh5678
- IndustReal: https://arxiv.org/abs/2305.17110
- FORGE: https://arxiv.org/abs/2501.00786
- AutoMate: https://arxiv.org/abs/2404.24430
- Residual RL: https://arxiv.org/abs/1812.08254
- Asymmetric Actor Critic: https://arxiv.org/abs/1802.09477
- CQL: https://arxiv.org/abs/2006.04779
- AWAC: https://arxiv.org/abs/2006.09359
- IQL: https://arxiv.org/abs/2110.06169
- PER: https://arxiv.org/abs/1511.05952
- VT-Refine: https://arxiv.org/abs/2506.10950
- Reactive Diffusion Policy: https://arxiv.org/abs/2504.20865
- TacDiffusion: https://arxiv.org/abs/2412.01978
- VLA-Touch: https://arxiv.org/abs/2507.17294
- TLA: https://arxiv.org/abs/2503.08548
- ManipLLM: https://arxiv.org/abs/2312.09358
- Dexterity from Touch: https://arxiv.org/abs/2303.12076
- OpenAI Rubik's cube: https://arxiv.org/abs/1910.07113
- Surprisingly Robust In-Hand Manipulation: https://robot-learning.cs.utexas.edu/primitive-robustness/
- Variable Impedance RL: https://arxiv.org/abs/1903.08965
- Active Inference Peg-in-Hole: https://arxiv.org/abs/2309.15681
- Whitney 1982: https://dspace.mit.edu/handle/1721.1/5794
- GelSight: https://people.csail.mit.edu/mkusper/gelsight.shtml
- DIGIT: https://digit.ml/
- BCQ: https://arxiv.org/abs/1812.02900
