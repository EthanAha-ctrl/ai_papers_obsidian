---
source_pdf: Sumo- Dynamic and Generalizable.pdf
paper_sha256: c31c69efdf1f68e8ead4b88b3a2d15cce150e0eefa3bef1234f3a91c28efa534
processed_at: '2026-08-12T11:29:16-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Sumo 用人话版

## 一句话总结

**先训练一个"会走路不会摔"的通用 robot policy，然后用 MPC 在 test 时实时指挥它去干各种 manipulation 活儿。**

## 为什么这件事以前做不好

Spot 这种四足机器人，你让它走路，现在 RL 已经训得很好了 — 翻墙、跳沟、跑酷都行。但你让它去扶起一个 15 kg 的轮胎（比它手臂最大负载 11 kg 还重），就抓瞎了。

传统两种做法都有问题：

**End-to-end RL**：训一个 policy 直接从 state 输出 joint torque。问题是你得给一堆 reward（"保持 gait"、"不要摔"、"往前走"、"gripper 靠近物体"、"别把物体撞飞"...15 个 term），训完只能干这一个活儿。换个 object（从 box 换到 tire）就挂了，因为 tire 是空心橡胶、摩擦不一样、几何不一样。重新训又得 2 小时 GPU + 调 reward。

**End-to-end MPC**：在线规划 joint torque。问题是 Spot 有 19 个 joint，sample-based MPC 在 19 维空间里搜索，又慢又找不到好解；更致命的是 robot 是 open-loop unstable，你随便 sample 一串 torque，0.3 秒 robot 就摔了，rollout 全是 noise，啥信息都提取不出来。

## Sumo 的核心 idea

把问题拆成两层：

**下层**：用一个已经训好的通用 policy（比如 Relic），它吃"指令"（base velocity、arm joint target、torso pose 这些），输出 joint command，保证 robot 不摔、走得稳。这个 policy 是 frozen 的，不重训。

**上层**：用 sample-based MPC（CEM）在"指令空间"里搜索 — 不是搜 joint torque，是搜"让 base 往左 0.3 m/s、arm 第三个 joint 转 30 度"这种 high-level 命令。

这样做有三个立刻能感受到的好处：

1. **搜索空间小了**：从 19 维 joint 降到 9 维命令（base 3 + arm 6），sample efficiency 指数级提升
2. **dynamics 稳了**：因为下层 policy 已经内化了"不要摔"的反馈控制，MPC 在 rollout 时 robot 不会随便 diverge，每个 sample 都是有意义的信号
3. **cost function 简单了**：不用管 gait、foot schedule、balance 这些 — policy 自己管。MPC 的 cost 只需要说"把 tire 立起来"这种 manipulation 目标，3 个 term 就够

## 最关键的工程 trick：policy-in-the-loop rollout

普通 sample-based MPC 的 rollout 是：
```
state x → dynamics f(x, u) → next state
```
其中 u 是 joint torque，直接 sample。

Sumo 改成：
```
state x → policy π(x, c) → joint command u → dynamics f(x, u) → next state
```
其中 c 是 high-level command，MPC 在 c 空间 sample，policy 把 c 翻译成 u。

实现上：MuJoCo 里每一步先 forward policy 拿到 u，再用 u 跑一步 physics。32 个 parallel rollout 在 CPU 上跑 1.5 秒 horizon，43 ms 搞定，刚好 fit 在 50 ms 的 MPC 周期里。

这个 trick 让"在 unstable robot 上跑 sample-based MPC"从 impossible 变成 realtime。

## 为什么能 generalize

这是 paper 最 make sense 的地方。

RL 的 generalization 是"训练 distribution 覆盖测试 distribution"。但 open world 里你不可能预先训所有 object。

Sumo 的 generalization 是另一种：**test-time 换 model / 换 cost**。

- 换 object：MPC rollout 里用的 object model 换一下就行，cost 不变。Box 换 tire，直接换 MuJoCo 里的 mesh + mass + friction
- 换 task：cost 从"move to goal"换成"upright orientation"，立刻能做 upright

不需要 retrain。因为 high-level 决策是 online search，不是 offline memorize。

## 实验 striking 的点

1. **Tire upright 10/10 成功**，tire 15 kg > Spot arm 11 kg capacity。Robot 自己 emergent 出"蹲下、arm 勾 tire 边、前腿撑地、torso 后仰翻 tire"的策略 — 没人教它这个，是 MPC 在 cost 上 sample 出来的

2. **E2E RL 在 tire/tire rack 上崩**：同样 reward 训练，box 能 100%，tire 只有 30%。因为 tire 几何和摩擦 OOD

3. **调参速度**：Bayesian optimization 调 cost weight，Sumo 10 CPU 小时达到 90% success；hierarchical RL 100 GPU 小时达到同样水平。**一个数量级的 wall-clock 加速**

4. **8 个 real-world Spot task + 4 个 G1 sim task**，涵盖 upright / stack / drag / push / open door 五种 manipulation mode

## 一个我觉得很妙的小设计

Leg selection variable $s_{\text{leg}} \in [-1, 1]$：
- $s < -0.5$：用 front-left leg 当 manipulator
- $s > 0.5$：用 front-right leg
- 中间：不用腿

CEM 把它当 continuous 来 sample，但 threshold 后是 discrete 决策。Sample-based MPC 天然 handle mixed discrete + continuous，gradient-based 方法处理这个超痛苦。这种"用连续参数 + threshold 做 discrete 选择"的 pattern 其实挺 general 的。

## 架构思想的更深层 intuition

Sumo 本质上是在说：**RL 和 MPC 不要二选一，要分工**。

- RL 擅长的：offline 大量 trial-and-error，学 general skill（走路、保持平衡），domain randomization 让它 robust
- MPC 擅长的：online reasoning，test-time 适应新 model / 新 cost，closed-loop feedback 修正 sim-to-real gap

让 RL 做"学 general substrate"，让 MPC 做"adapt to specific task"。两个 subproblem 各自 tractable，组合起来 1+1 > 2。

这个思想和 LLM foundation model 时代是同构的：
- Pre-trained WBC policy ≈ pre-trained LLM
- CEM 的 32 rollouts × 20Hz ≈ inference-time tree-of-thought
- Cost function ≈ prompt（告诉模型"做什么"）
- Test-time swap cost/model ≈ in-context learning

未来如果出现真正 generalist 的 WBC policy（不只是 Spot 专属的 Relic），这个框架会更强 — Sumo 已经给了"怎么 steering"的 template。

## Limitations

- 依赖 MoCap 做 state estimation，lab-only
- Object model 假设已知（geometry + pose），real world 需要 perception pipeline
- Sim-to-real gap 依然存在（虽然 MPC closed-loop 缓解）
- CEM 在 action dim > 20 时会吃力，需要更聪明的 sampler

## 我读完的 takeaway

对 robotics practitioner：
1. 不要一上来 end-to-end RL，先想能不能 decompose
2. Pre-train general WBC policy 是值得的投资，下游能被多种 task steering
3. Cost function 设计是 manipulation 核心 — 把 task knowledge encode 进 cost 比 encode 进 reward 更 flexible
4. Sample-based MPC 现在已经能 realtime，policy-in-the-loop 是让它在 legged robot 上 work 的关键 trick

对研究者：
1. "High-MPC + low-RL" 这个 paradigm 因为 generalist tracking policy 的成熟变得 viable 了，值得探索
2. Generalization 的新定义 — test-time compositional generalization via swap model/cost — 比 distribution robustness 更实际
3. Policy-in-the-loop rollouts 可以推广到其他 sample-based 方法（diffusion planning、CMA-ES 等）

Paper code 在 [sumo.rai-inst.com](https://sumo.rai-inst.com/) 开源了。

---

# Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation — 技术深度解析

## 1. Paper 的 Big Picture 与 Motivation

这篇paper (RAI Institute + MIT, [项目主页](https://sumo.rai-inst.com/)) 想解决一个 legged robot 领域长期没被攻克的问题：**让四足 / 人形机器人用整个 body 动态操作 large / heavy / geometrically complex 的物体**。比如 Spot 把一个比自身 lifting capacity 还重的 tire 立起来、把比自己还高的 crowd barrier 拖动、把 tire 叠到另一个 tire 上，或者 G1 humanoid 开门、推 table。

这类 loco-manipulation 的难点有 threefold：
- **Whole-body coordination**：单靠 arm 不够，需要 torso pitch/roll、legs（甚至作为 manipulator 的 leg）、gripper 协同；
- **Dynamic contact-rich**：不是 quasi-static pick-and-place，需要利用 object 的 passive dynamics；
- **Generalization**：每个新 object 的 geometry / mass / friction 都不同，train-then-deploy 的 RL paradigm 容易 OOD 失败。

作者的关键 insight：**pre-trained generalist WBC (whole-body control) policy + test-time sample-based MPC** 的 hierarchy，能同时拿到 RL 的 robust locomotion 和 MPC 的 test-time flexibility。

注意它探索的是 **high-level MPC + low-level RL** 的 paradigm，反过来常见于很多 hybrid RL-MPC 工作（high-level RL + low-level MPC，例如 [DTC](https://www.science.org/doi/10.1126/scirobotics.adh5401)、[RAMBO](https://ieeexplore.ieee.org/document/10905808)）。这个反转之所以现在可行，是因为最近出现了 powerful 的 generalist tracking policies（如 [Relic](https://arxiv.org/abs/2506.07876)、[OmniTrack](https://arxiv.org/abs/2602.23832)、[BeyondMimic](https://arxiv.org/abs/2508.08241)、[SONIC](https://arxiv.org/abs/2511.07820)），它们本来是为 teleoperation / motion tracking 设计的，但可以作为被 steering 的 substrate。

## 2. 系统架构

参考 paper Fig. 2：

```
                        High-level Sample-based MPC (20 Hz)
                                  (CEM, 32 rollouts, 1.5s horizon)
                                  input: state estimate (robot + object)
                                  output: a = [a_base, a_arm, a_torso, a_leg, a_gripper]
                                              │
                                              ▼
                       ┌──────────────────────────────────────┐
                       │  Low-level WBC policy (50 Hz, RL)     │
                       │  Relic (Spot) / MJLab vel (G1)        │
                       │  input: state + c = [c_base, c_arm,   │
                       │         c_gripper, c_leg, c_torso]   │
                       │  output: u (joint-level commands)     │
                       └──────────────────────────────────────┘
                                              │
                                              ▼
                                    Robot (Spot / G1) + Object
                                              │
                                              │ state @ 120 Hz (MoCap) + 333 Hz (encoders)
                                              ▼
                                   (asynchronous CEM update)
```

关键频率：low-level policy 50 Hz（每 20 ms 出一次 joint command），high-level MPC 20 Hz（每 50 ms 出一次 command $\mathbf{a}$）。MPC 内部需要 rollout 1.5 s horizon，所以并行 32 个 rollout 单次 43.45 ms（见 Table I），刚好能卡在 50 ms 内。这是非常 tight 的实时预算。

异步执行（paper Sec. V-A）通过 AMD Threadripper 5995WX 64 核机器跑 dynamics rollout，WiFi 发送给 Spot。

## 3. Policy-in-the-Loop Parallel Rollouts — 工程核心

这是整篇 paper 最关键的工程 idea。Sample-based MPC 的 rollout 一般是：

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t) \quad \text{(Eq. of motion)}$$

其中 $\mathbf{u}_t$ 是 joint-level torque / PD target。对应 paper Fig. 3(a)。

Sumo 改成 **network-policy-augmented dynamics**：

$$\mathbf{x}_{t+1} = f\Big(\mathbf{x}_t,\ \underbrace{\mathbf{u}_t = \pi_{\theta}(\mathbf{x}_t, \mathbf{c}_t)}_{\text{low-level policy}}\Big),\quad \mathbf{c}_t = g(\mathbf{a}_t)$$

对应 paper Fig. 3(b)。这里 $\pi_\theta$ 是 frozen pre-trained WBC policy（神经网），$g(\cdot)$ 是从 high-level action $\mathbf{a}$ 到 policy command $\mathbf{c}$ 的简单映射（补 default 值、mask 等）。

为什么这个 trick 这么 powerful？三个原因：

### 3.1 Reduced action space
Spot 全身有 19 DoF，加上 gripper 大约 20 个 joint command。但 high-level 只需要 sample
$$\mathbf{a}_{\text{default}} = [\mathbf{a}_{\text{base}} \in \mathbb{R}^3, \mathbf{a}_{\text{arm}} \in \mathbb{R}^6] \in \mathbb{R}^9$$
就能 cover 大部分 loco-manipulation 任务。**Action dim 从 ~20 降到 9**。

对 CEM 这种 zeroth-order 方法，sample 复杂度大致是 $O(N^d)$（$d$ 是 action dim，$N$ 是 per-dim samples）。降一倍维度是指数级 speedup。

### 3.2 Stabilized dynamics
Legged robot 是 open-loop unstable system：你给它一串 random joint torques，robot 0.3 秒就摔了。Sample-based MPC 用 single-shooting rollout，意味着每个 sample 都要 forward simulate 整个 horizon，rollout 一旦 diverge 就完全是 noise，gradient/information 全失。这是 sample-based MPC 一直没法直接用在 humanoid 上的根本原因。

Low-level policy 充当 stabilizing feedback controller：从 $\mathbf{a}$ 到 $\mathbf{u}$ 的映射里 $\pi_\theta$ 已经内化了"不要摔"的先验。这样 high-level planner 看到的是一个 **closed-loop stable** 的 dynamics，rollout 不会爆。

这一点直觉上类似 MPC 文献里的 "shooting vs collocation" — collocation 通过约束保持 feasibility，shooting 容易 diverge。Sumo 用 policy 当作 implicit feedback collocation。

### 3.3 Simpler cost function
因为 locomotion 部分（保持 gait、不摔、track base velocity）由 low-level policy 负责，high-level cost 只需要表达 **manipulation objective**。看 Eq. 3：

$$J_{\text{Move}} = w_{\text{goal}} \|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{goal}}\|_2 + w_{\text{gripper}} \|\mathbf{p}_{\text{gripper}} - \mathbf{p}_{\text{obj}}\|_2 + w_{\text{vel}} \|\mathbf{v}_{\text{obj}}\|_2$$

- $\mathbf{p}_{\text{obj}}, \mathbf{p}_{\text{goal}}, \mathbf{p}_{\text{gripper}}$：世界坐标系下 object / goal / gripper 的 3D position
- $\mathbf{v}_{\text{obj}}$：object 的线速度（penalize 撞飞物体）
- $w_{\text{goal}}, w_{\text{gripper}}, w_{\text{vel}}$：权重，跨任务通常 fixed

对比 E2E RL 用了 15 个 reward terms（gait phase、foot-height schedule、progress、gripper approach、stay-behind-object、reach bonus 等），Sumo 用 3 个 reward term 就 cover 五个 object 的 Move 任务。这是巨大的工程简化。

### 3.4 Timing 数据 (Table I)

| Method | 32 rollouts × 1.5s |
|---|---|
| MuJoCo with low-level policy | 43.45 ± 1.88 ms |
| MuJoCo only | 21.73 ± 1.86 ms |

Policy inference 大约吃掉 20 ms（neural network forward + MuJoCo 的 state 拼接），但依然 fit 在 50 ms 的 MPC budget 内。

实现细节：作者用 C++ 改 MuJoCo（CPU），thread pool 做 32 路并行。没用 GPU。这点其实值得思考 — 现在 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 已经 GPU 化，未来 Sumo 跑 1000+ rollouts 在 GPU 上应该可行，sample efficiency 会再上一个台阶。

## 4. Action Space 设计 — 隐藏的巧妙

Paper Sec. III-D 给了 action space 的 modular 设计。对 Spot + Relic policy，full command 是 $\mathbf{c}_{\text{policy}} \in \mathbb{R}^{25}$：
- $\mathbf{c}_{\text{base}} \in \mathbb{R}^3$：base 的 SE(2) velocity（$v_x, v_y, \omega_z$）
- $\mathbf{c}_{\text{arm}} \in \mathbb{R}^6$：arm 6 个 joint 的 angle target
- $\mathbf{c}_{\text{gripper}} \in \mathbb{R}^1$：gripper 位置
- $\mathbf{c}_{\text{leg}} \in \mathbb{R}^3 \times 4 = \mathbb{R}^{12}$：四条腿的 joint angle target（Relic 允许 leg 当 manipulator）
- $\mathbf{c}_{\text{torso}} \in \mathbb{R}^3$：torso pitch / roll / height

Sumo 默认只在 $\mathbf{a}_{\text{base}} \in \mathbb{R}^3, \mathbf{a}_{\text{arm}} \in \mathbb{R}^6$ 上 sample，其余补 default。但需要 multi-limb coordination 时（如 upright 重 tire），可以开 $\mathbf{a}_{\text{torso}} \in \mathbb{R}^3$ 和 $\mathbf{a}_{\text{leg}} \in \mathbb{R}^7$。

### 4.1 Leg selection variable (Eq. 1)

$$\mathbf{c}_{\text{leg}}^{\text{masked}} = \begin{cases} 
[\mathbf{c}_{\text{leg}}[0{:}3], 0, 0, 0] & \text{if } s_{\text{leg}} < -0.5\ (\text{FL}) \\
[0, 0, 0, \mathbf{c}_{\text{leg}}[3{:}6]] & \text{if } s_{\text{leg}} > 0.5\ (\text{FR}) \\
[0, 0, 0, 0, 0, 0] & \text{else (no legs)}
\end{cases}$$

变量解释：
- $\mathbf{c}_{\text{leg}} \in \mathbb{R}^6$：high-level sample 出的 leg joint angle target，前 3 维给 FL（front-left），后 3 维给 FR（front-right）
- $s_{\text{leg}} \in [-1, 1]$：selection variable，CEM 当成 continuous 来 sample，但实际是 discrete 决定"用 FL / 用 FR / 都不用"
- $\mathbf{c}_{\text{leg}}[0{:}3]$：FL leg 的 3 个 joint（hip yaw、hip pitch、knee）
- $\mathbf{c}_{\text{leg}}^{\text{masked}}$：最终传给 policy 的 6 维 leg command，没被选中的 leg 部分置 0

这个 trick 让 CEM 在同一个 continuous action vector 里 mixed discrete + continuous decision。Sample-based 方法天然支持这个（不像 iLQR 直接死掉），作者顺便把这个优势暴露出来。

### 4.2 Gripper binary action (Eq. 2)

$$\mathbf{c}_{\text{gripper}} = \begin{cases}
\mathbf{c}_{\text{gripper\_close}} & \text{if } \mathbf{a}_{\text{gripper}} > 0 \\
\mathbf{c}_{\text{gripper\_open}} & \text{if } \mathbf{a}_{\text{gripper}} \leq 0
\end{cases}$$

$\mathbf{a}_{\text{gripper}} \in [-1, 1]$ 是 high-level scalar，threshold 后变成 binary open/close。Simple 但有效。

### 4.3 Full modular action

$$\mathbf{a} = [\mathbf{a}_{\text{base}}, \mathbf{a}_{\text{arm}}, \mathbf{a}_{\text{torso}}, \mathbf{a}_{\text{leg}}, \mathbf{a}_{\text{gripper}}] \in \mathbb{R}^{3+6+3+7+1} = \mathbb{R}^{20}$$

任务决定开几个 module。这给了 test-time 调 action space 的能力：tire upright 用 base + arm + torso + leg（$3+6+3+7=19$），barrier drag 只用 base + arm + gripper（$3+6+1=10$）。

## 5. Cost Function 深度分析

Appendix 里给了 8 个 Spot task + 4 个 G1 task 的完整 cost。挑几个讲：

### 5.1 Tire Upright (Eq. 6)

$$\begin{aligned}
J_{\text{Tire Upright}} &= w_{\text{orient}} \cdot \exp(|\mathbf{y}_{\text{tire}_z}| / \sigma) \\
&+ w_{\text{gripper}} \cdot \|\mathbf{p}_{\text{gripper}} - \mathbf{p}_{\text{gripper}}^{\text{des}}\|_2 \\
&+ w_{\text{foot}} \cdot \min(\|\mathbf{p}_{\text{fr}} - \mathbf{p}_{\text{fr}}^{\text{des}}\|_2, \|\mathbf{p}_{\text{fl}} - \mathbf{p}_{\text{fl}}^{\text{des}}\|_2) \\
&+ w_{\text{torso}} \cdot \|\mathbf{p}_{\text{torso}} - \mathbf{p}_{\text{torso}}^{\text{des}}\|_2 \\
&+ w_{\text{ctrl}} \cdot \|\mathbf{u}\|_2 + J_{\text{safety}}
\end{aligned}$$

变量：
- $\mathbf{y}_{\text{tire}_z}$：tire 的 y-axis 在世界 z 方向的分量。Tire 平躺时 y-axis 大致水平（值 ≈ 1），upright 时 y-axis 朝上（值 ≈ 0）。$\exp(\cdot/\sigma)$ 把它变成 smooth penalty，$\sigma$ 控制 sharpness
- $\mathbf{p}_{\text{gripper}}, \mathbf{p}_{\text{fr}}, \mathbf{p}_{\text{fl}}, \mathbf{p}_{\text{torso}}$：四个 end-effector 的 world position
- $\mathbf{p}^{\text{des}}$：根据 tire 当前位置 dynamically 算出的"应该在哪儿"。这是关键设计 — 不是固定目标，是 **relative target**
- $w_{\text{foot}}$ 用 $\min$ 而非 sum：意思是"前后脚任一只靠近 tire 即可"，鼓励 exploration 哪只脚去 push
- $J_{\text{safety}}$：fall penalty

这里能看出 cost function 设计的层次：**orientation 是主目标，end-effector proximity 是 sub-goal（怎么实现 orientation），ctrl + safety 是 regularizer**。MPC 的 sample 在这个 cost 上做 CEM 就能 emergently 找到"用脚踩 tire 边缘 + 用 arm 推 + torso 前倾"的协调策略。

### 5.2 Crowd Barrier Upright (Eq. 7)

$$\begin{aligned}
J_{\text{Barrier}} &= w_{\text{orient}} \cdot (1 - \exp(\alpha \cdot (\mathbf{z}_{\text{barrier}} \cdot \mathbf{z}_{\text{world}} - 1))) \\
&+ w_{\text{grasp}} \cdot \min(\|\mathbf{p}_{\text{grip}} - \mathbf{p}_{\text{grasp}}^L\|, \|\mathbf{p}_{\text{grip}} - \mathbf{p}_{\text{grasp}}^R\|) \\
&+ w_{\text{grip}} \cdot (1 - |\mathbf{x}_{\text{grip}} \cdot \mathbf{x}_{\text{barrier}}| + 1 - |\mathbf{y}_{\text{grip}} \cdot \mathbf{z}_{\text{barrier}}|) \\
&+ w_{\text{approach}} \cdot \min(\|\mathbf{p}_{\text{torso}} - \mathbf{p}_{\text{appr}}^L\|, \|\mathbf{p}_{\text{torso}} - \mathbf{p}_{\text{appr}}^R\|) \\
&+ w_{\text{vel}} \cdot \|\mathbf{v}_{\text{obj}}\|_2^2 + w_{\text{ctrl}} \cdot \|\mathbf{u}\|_2 + J_{\text{grasp}} + J_{\text{safety}}
\end{aligned}$$

变量：
- $\mathbf{z}_{\text{barrier}}, \mathbf{z}_{\text{world}}$：barrier 和世界的 z 轴。Barrier 平躺时 $\mathbf{z}_{\text{barrier}} \perp \mathbf{z}_{\text{world}}$，点积 0；upright 时 $\mathbf{z}_{\text{barrier}} \parallel \mathbf{z}_{\text{world}}$，点积 1。$1 - \exp(\alpha \cdot (\cdot - 1))$ 是 smooth saturation
- $\mathbf{p}_{\text{grasp}}^{L/R}$：barrier 上 left/right 两个 grasp point
- $\mathbf{x}_{\text{grip}}, \mathbf{y}_{\text{grip}}$：gripper 自己的 x / y axis。Orientation cost 要求 gripper x 对齐 barrier 长轴，gripper y 对齐 barrier z（即 gripper 横握 barrier）
- $\mathbf{p}_{\text{appr}}^{L/R}$：barrier left/right 两个 approach 站位
- $J_{\text{grasp}}$：通过 gripper closing 时的 position error 检测 grasp success，penalize 空 closing

这个 cost 已经把 manipulation 的"affordance"encode 进去了 — 告诉 robot"barrier 有两个 grasp point，你从哪边都行，gripper 要这样握"。这相当于把 task knowledge 通过 cost 注入，而不是 reward engineering 训进 policy。

### 5.3 Tire Stack (Eq. 10)

$$\begin{aligned}
J_{\text{Stack}} &= w_{xy} \|\mathbf{p}_{\text{top}}^{xy} - \mathbf{p}_{\text{bottom}}^{xy}\|_2 + w_z |p_{\text{top}}^z - p_{\text{desired}}^z| \\
&+ w_{\text{orient}} (1 - \mathbf{y}_{\text{top}} \cdot \hat{\mathbf{u}}_{\text{stack}}) \\
&+ w_{\text{bottom}} (\|\mathbf{v}_{\text{bottom}}\|_2 + \|\omega_{\text{bottom}}\|_2) \\
&+ w_{\text{gripper}} \|\mathbf{p}_{\text{gripper}} - \mathbf{p}_{\text{gripper}}^{\text{des}}\|_2 \\
&+ w_{\text{torso}} \|\mathbf{p}_{\text{torso}} - \mathbf{p}_{\text{torso}}^{\text{des}}\|_2 + w_{\text{ctrl}} \|\mathbf{u}\|_2 + J_{\text{safety}}
\end{aligned}$$

变量：
- $\mathbf{p}_{\text{top}}^{xy}, \mathbf{p}_{\text{bottom}}^{xy}$：top tire 和 bottom tire 的 XY 平面位置（top 要在 bottom 正上方）
- $p_{\text{top}}^z, p_{\text{desired}}^z$：top tire 的 z 坐标和目标 z（bottom tire 高度 + tire 半径 + tire 半径）
- $\mathbf{y}_{\text{top}}$：top tire 的 y 轴（直立时朝上）
- $\hat{\mathbf{u}}_{\text{stack}}$：从 top tire 指向 bottom tire 的 unit vector。Stacked 时 $\mathbf{y}_{\text{top}} \parallel -\hat{\mathbf{u}}_{\text{stack}}$
- $\|\mathbf{v}_{\text{bottom}}\|_2, \|\omega_{\text{bottom}}\|_2$：bottom tire 的线速度 + 角速度，penalize bottom tire 被撞歪

这 cost 分两层：先把 top tire "lift + upright"（$w_z, w_{\text{orient}}$），再 align 到 bottom tire 正上方（$w_{xy}$），并 protect bottom tire 不被推乱（$w_{\text{bottom}}$）。一个非常 elegant 的 multi-stage objective。

## 6. 实验核心结论

### 6.1 Hierarchical Simplifies Loco-Manipulation (Fig. 4)

5 个 Move task（box, chair, cone, tire, tire rack），20 trials：

| Method | box | chair | cone | tire | tire rack |
|---|---|---|---|---|---|
| Sumo | ~100% | ~90% | ~90% | ~80% | ~85% |
| E2E RL | ~100% | ~95% | ~90% | ~30% | ~20% |
| E2E MPC | ~50% | ~40% | ~40% | ~30% | ~30% |

两个观察：
- **E2E MPC 全线挣扎**。即使给它 50Hz update + locomotion cost，joint-space sampling 在 19-DoF Spot 上还是 curse-of-dim 太重，加上 unstable dynamics 让 single-shooting rollout 大部分 diverge。
- **E2E RL 在 box/chair/cone OK，tire/tire-rack 崩**。即便用同样 reward 训练，box → tire 的 geometry transfer 失败。Tire 是 hollow torus + deformable rubber，sim-to-real gap 大。

Sumo 两边都好：相比 E2E MPC 享受 stabilized + reduced action space；相比 E2E RL 不需要 retrain，cost function 在新 object 上往往直接 work。

### 6.2 Test-Time Search Enables Generalization (Fig. 5)

更 striking 的实验：把 hierarchical RL（low-level policy 固定 + high-level RL）训练在 box Move 任务上，test 到 tire / cone 时 success rate 砍半。Sumo 只换 object model（cost 不变），immediate work。

更 striking：把 cost 从 Move (Eq. 3) 换成 Upright (Eq. 4)，Sumo 直接做 upright tire/cone/barrier。Hierarchical RL 必须重训。

这是 paper 的核心论点：**generalization 不是 "policy 跨 distribution 泛化"，而是 "planner 在 test-time 适应新 model + 新 cost"**。这个 generalization 的定义比 RL literature 里说的 generalization 更实际。

### 6.3 Sample Efficiency (Fig. 6)

Bayesian optimization 调 cost weights，对比 Sumo vs Hierarchical RL 在 Move Box 上的 success rate vs wall-clock compute：

- Sumo：~10 CPU hours 达到 ~90% success
- HRL：~100 GPU hours 达到 ~90% success

**Order-of-magnitude compute reduction**。原因：Sumo 调 weights 只需重跑 MPC，每次 ~30 s；HRL 调 reward weights 需要 retrain policy，每次 ~2 hours。

这点对 practitioner 极其重要 — RL 的"reward engineering loop"是最耗时的部分，Sumo 把这个 loop 缩短一个数量级。

## 7. Real-World Spot Demos (Table II)

8 个 task × 10 trials：

| Task | Time | Success | Limit | Weight |
|---|---|---|---|---|
| Tire Upright | 9.2 ± 4.7 s | 10/10 | 30 s | 15 kg (>arm capacity 11 kg) |
| Barrier Upright | 10.5 ± 7.1 s | 9/10 | 30 s | 16 kg |
| Cone Upright | 10.2 ± 7.9 s | 9/10 | 30 s | 3.5 kg |
| Chair Upright | 27.3 ± 19.1 s | 8/10 | 60 s | 16.5 kg |
| Tire Stack | 16.5 ± 8.4 s | 8/10 | 30 s | 15 kg on 15 kg |
| Barrier Drag | 20.2 ± 6.7 s | 9/10 | 30 s | 15 kg |
| Tire Rack Drag | 19.1 ± 6.2 s | 9/10 | 30 s | 10 kg |
| Rugged Box Push | 38.3 ± 16.9 s | 10/10 | 90 s | 20 kg |

注意 **tire 15 kg > Spot arm nominal 11 kg lifting capacity**。这种 task用 pick-and-place 是 impossible 的 — 必须 whole-body 用 legs + torso 帮忙。Sumo 通过 sampling 自动 emergent 出"先蹲下，arm 勾住 tire 边缘，前腿撑地推 tire，torso 后仰把 tire 翻起来"的策略。

Chair Upright 时间长且方差大（27 ± 19 s），是因为 chair 重心高 + 几何复杂，每次推的角度不同需要 replan 多次。MPC 在这种场景特别合适 — 它 online 重新规划，不会被 training distribution 的"canonical 路径"卡死。

## 8. G1 Humanoid Sim Demos (Table III)

4 个 sim task：box push (10/10, 4.7s), chair push (10/10, 6.9s), door open (10/10, 4.7s), table push (8/10, 4.9s)。

G1 用的是 MJLab 标准 velocity-tracking policy，**没有专门 train loco-manipulation**。Sumo 直接 override arm command，policy 只管 locomotion 部分。这种 setup 比 Spot 简单（没 multi-limb coordination），但 work 说明 framework 是 robot-agnostic 的。

Door Open 的 cost (Eq. 16)：
$$\begin{aligned}
J_{\text{G1 Door}} &= w_{\text{goal}} \|\mathbf{p}_{\text{pelvis}} - \mathbf{p}_{\text{goal}}\|_2 + w_{\text{hand}} \|\mathbf{p}_{\text{right}} - \mathbf{p}_{\text{handle}}\|_2 \\
&- w_{\text{pelvis}} \|\mathbf{p}_{\text{pelvis}} - \mathbf{p}_{\text{door}}\|_2 - w_{\text{facing}} \mathbf{x}_{\text{robot}} \cdot \mathbf{x}_{\text{world}} \\
&+ w_{\text{ctrl}} (\|\mathbf{v}_{\text{base}}\|_2 + \|\mathbf{q}_{\text{arm}} - \mathbf{q}_{\text{arm}}^{\text{default}}\|_2) + J_{\text{safety}}
\end{aligned}$$

- $\mathbf{p}_{\text{goal}}$ 在门对面（必须开门才能到）
- $\mathbf{p}_{\text{handle}}$ 是门把手位置
- $-w_{\text{pelvis}} \|\mathbf{p}_{\text{pelvis}} - \mathbf{p}_{\text{door}}\|_2$：**negative cost**，鼓励靠近门
- $-w_{\text{facing}} \mathbf{x}_{\text{robot}} \cdot \mathbf{x}_{\text{world}}$：鼓励 robot 朝前
- $\mathbf{q}_{\text{arm}}^{\text{default}}$：arm default pose，penalize 乱甩 arm

这里 negative cost 的用法很有意思 — 用 $-w$ 的 negative weight 让"靠近"成为 reward 而不是 penalty。这在 MPC 里是 OK 的（cost 可负），但在 RL 里 reward shaping 更 tricky。

## 9. CEM 超参细节

Paper Sec. IV-A 给了 CEM 配置：
- Optimizer: Cross-Entropy Method ([CEM tutorial](https://arxiv.org/abs/1604.00772))
- Horizon: 1.5 s
- Rollouts: 32 parallel
- Elite samples: 3（top-3 用来更新 distribution）
- Update frequency: 20 Hz
- Action parameterization: 4 spline control points over horizon，中间 linear interpolation
- Noise variance schedule: linear ramp 0.02 → 0.6 over horizon

Spline control points 这个设计很关键 — 直接 sample 1.5s × 20Hz = 30 个 timestep 的 action 是 30 × 9 = 270 dim，CEM 死掉。用 4 个 spline knot point 把 dim 降到 4 × 9 = 36，并且 action naturally smooth。这个 trick 来自 [Predictive Sampling (Howell et al.)](https://arxiv.org/abs/2212.00541) 和 [Judo library](https://github.com/bdaiinstitute/judo)。

Noise ramp 从 0.02（near-term 精确）到 0.6（far-term 探索）也是经典 MPC horizon 设计 — 近端要精确执行，远端允许 exploration。

## 10. 与相关工作的对比 — 把 Sumo 放进 landscape

### 10.1 RL for WBC
- [Hwangbo et al. Science Robotics 2019](https://www.science.org/doi/10.1126/scirobotics.aau5872)：ANYmal RL locomotion，证明 sim-to-real RL locomotion 可行
- [Extreme Parkour (Cheng et al.)](https://ieeexplore.ieee.org/document/10610200)：agile locomotion，但 RL 难以跨 task
- [Relic (Zhu et al.)](https://arxiv.org/abs/2506.07876)：Sumo 直接用的 low-level policy
- [SONIC (Luo et al.)](https://arxiv.org/abs/2511.07820)：humanoid whole-body motion tracking，最近很强

### 10.2 Sample-based MPC
- [MPPI (Williams et al.)](https://ieeexplore.ieee.org/document/7487277)：information-theoretic derivation
- [Predictive Sampling (Howell et al.)](https://arxiv.org/abs/2212.00541)：MuJoCo 实时 MPPI，Sumo 直接借鉴 spline trick
- [Judo (Li et al.)](https://github.com/bdaiinstitute/judo)：开源 sample-based MPC library
- [CMA-ES (Hansen)](https://arxiv.org/abs/1604.00772)：CEM 的进化策略 cousin
- [Trajectory Bundle Method (Tracy et al.)](https://arxiv.org/abs/2509.26575)：unify SCP + sampling
- [Model-based Diffusion (Pan et al.)](https://arxiv.org/abs/2407.01573)：diffusion 视角看 sample-based MPC

### 10.3 Loco-Manipulation
- [Pedipulate (Arm et al.)](https://ieeexplore.ieee.org/document/10611307)：用 quadruped leg 当 manipulator
- [Legs as Manipulator (Cheng et al.)](https://ieeexplore.ieee.org/document/10610200)：pushing tasks
- [Versatile Multicontact Planning (Sleiman et al.)](https://www.science.org/doi/10.1126/scirobotics.adg5014)：model-based，需要 fixed contact mode
- [RAMBO (Cheng et al.)](https://ieeexplore.ieee.org/document/10905808)：RL-augmented MPC for loco-manipulation，high-level RL + low-level MPC，正好和 Sumo 反过来
- [DTC (Jenelten et al.)](https://www.science.org/doi/10.1126/scirobotics.adh5401)：deep tracking control，同样 high-RL low-MPC

### 10.4 Hybrid RL-MPC
- [DTC](https://www.science.org/doi/10.1126/scirobotics.adh5401)、[RAMBO](https://ieeexplore.ieee.org/document/10905808)、[CAJUN (Yang et al.)](https://arxiv.org/abs/2306.09557)、[Residual MPC (Jeon et al.)](https://arxiv.org/abs/2510.12717)：都是 high-RL low-MPC pattern
- Sumo 是 **high-MPC low-RL** 的 representative，motivated by 强大 generalist tracking policies 的出现

### 10.5 Steering Pre-trained Policies
- [Steering Your Generalists (Nakamoto et al.)](https://arxiv.org/abs/2410.13816)：tabletop manipulation 用 value guidance steering foundation model
- [Learning to Search (Jain et al.)](https://arxiv.org/abs/2506.05294)：robust imitation via search
- [Planning with Diffusion (Janner et al.)](https://arxiv.org/abs/2206.09278)：diffusion + guidance 思想类似 Sumo 的 test-time cost steering

## 11. Intuition Building — 为什么这个架构 work

让我从更高维度思考为什么 Sumo work：

### 11.1 Decomposition 的力量
Loco-manipulation 是个 **compositional hard problem**：locomotion（continuously unstable，needs feedback）+ manipulation (contact-rich, geometric reasoning)。End-to-end RL/MPC 都要同时解决两边。

Sumo 的 decomposition：
- Low-level policy 负责"continuous stabilization + locomotion primitives"，这是 RL 擅长的（domain randomization 大量重复 trial-and-error）
- High-level MPC 负责"contact-rich reasoning over novel object"，这是 MPC 擅长的（test-time model + cost swap）

两个 subproblem 各自 tractable，组合就 work。这是个好的 divide-and-conquer 例子。

### 11.2 Generalization 的重新定义
RL literature 的 generalization 通常是"distribution shift robustness"。Sumo 提出另一种：**test-time compositional generalization**，通过 swap model 和 cost 实现。

这个 view 更接近 foundation model 时代的思路：pretrain 一个 general substrate（WBC policy），test-time 通过 guidance / planning / prompting 适应新任务。差别是 LLM 用 prompt，Sumo 用 cost function + dynamics model。

### 11.3 Sample-based MPC 适合 contact-rich 的根本原因
Contact 是 non-smooth、discontinuous dynamics。Gradient-based 方法（iLQR、DDP、collocation）在 contact 处 Jacobian 不存在或不可靠，需要 contact-implicit 处理（如 [Le Cleac'h et al.](https://arxiv.org/abs/2107.05616) 的 COCO）。Sample-based 完全 bypass derivative，自然 handle contact。

加上 low-level policy 稳定 dynamics 后，sample 在 command space 而非 joint space，efficiency 大幅提升。

### 11.4 为什么 RL 在 tire / tire rack 上失败
Tire 是 hollow + deformable + 高摩擦。Sim-to-real 里 rubber 摩擦极难建模（静摩擦、动摩擦、stiction 都不同）。RL policy 在训练 distribution 里学到的"如何 push tire"在 real-world 不 work。

Sumo 的 MPC rollout 也用 sim tire，理论上也有 sim-to-real gap。但因为 MPC 是 closed-loop online replanning，每 50 ms 重新观察真实 state，重新 plan。Real tire 摩擦比 sim 小？MPC 看到没推动就加大 force。Real tire 滑出去？MPC 立刻重新 sample 路径。

这是 MPC 相对 open-loop RL 的本质优势 — **feedback + receding horizon**。

### 11.5 Selection variable 的玄机
Eq. 1 的 $s_{\text{leg}} \in [-1, 1]$ 是个 continuous 参数，但通过 threshold 变成 discrete 选择。CEM 在 continuous space sample，自然 explore 三种模式（FL/FR/none）。这是 sample-based 方法对 mixed-integer programming 的天然友好性，gradient-based 方法处理这个非常痛苦。

这个设计也提示一个更 general 的 pattern：**用 continuous relaxation + threshold 让 sample-based planner 自动做 discrete reasoning**。在 LLM agent / tool use 那边也能看到类似 idea（连续 router + sparse activation）。

## 12. Limitations 和 Future Work

Paper 自己提了三个：
1. **State estimation 依赖 MoCap**：lab-only。需要 onboard perception（depth camera + SLAM）。
2. **Sim-to-real gap 仍存在**：虽然 MPC closed-loop 缓解，但 model 系统识别 + online parameter update 能进一步改善。
3. **没有 human priors**：未来可以加 foundation model 自动生成 cost function（类似 [Eureka](https://arxiv.org/abs/2310.12931)、[L2R](https://arxiv.org/abs/2306.08647)）。

我补充几个观察：
- **Scalability of CEM**：32 rollouts 在 9-dim action 还行，action dim 升到 20+ 时 CEM sample complexity 会爆炸。可能需要 smarter sampling（CMA-ES with covariance learning，或 diffusion-based sampling）。
- **Horizon 1.5s 太短**：复杂的 multi-stage task（如 tire stack 包含 lift + place）需要 longer planning。可能需要 hierarchical planner（high-level task planner + mid-level MPC）。
- **GPU 化潜力**：现在用 CPU thread pool，限制了 rollout 数。用 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) GPU 后可以跑 1000+ rollouts，sample efficiency 大幅提升。
- **Object model 假设 known**：cost 用 $\mathbf{p}_{\text{obj}}, \mathbf{q}_{\text{obj}}$ 需要 object pose 已知。Real world 需要 perception pipeline 提供 object pose + geometry。这是 deployment 的 big barrier。
- **与 diffusion policy 的关系**：[Diffusion Policy (Chi et al.)](https://arxiv.org/abs/2303.04137) 也是 sample-based，但是 learn conditional generation。Sumo 的 MPC 是 model-based sample。未来可能 fusion — diffusion policy 当 prior，MPC 当 test-time refinement。

## 13. 与 LLM / Foundation Model 时代的呼应

Sumo 的架构思想其实和 foundation model 时代的 inference-time scaling 思想是同构的：
- **Pre-train general substrate**：WBC policy 类似 pre-trained LLM
- **Test-time compute**：CEM 的 32 rollouts × 20Hz = 640 rollouts/sec，类似 chain-of-thought / tree-of-thought 的 inference-time search
- **Cost / reward as prompt**：cost function 告诉 planner "做什么"，类似 prompt 告诉 LLM "做什么"
- **Generalization via test-time adaptation**：换 object model = 换 context，换 cost = 换 instruction

这个同构提示一个 future direction：如果 WBC policy 越来越 general（类似 GPT-4 之于 GPT-2），test-time planning 的负担会减少，因为 policy 本身就能 handle 更多变异。Sumo 当前依赖 Relic（特定于 Spot），未来用真正 generalist WBC policy，框架会更强大。

## 14. 总结

Sumo 的贡献：
1. **架构**：high-level sample-based MPC + low-level pre-trained WBC policy 的 hierarchy
2. **Key trick**：policy-in-the-loop rollouts，让 MPC 在 stabilized + reduced action space 上 sample
3. **Generalization**：test-time swap object model / cost，无需 retrain
4. **Practical**：cost 简单（3-5 terms）、CEM 高效（order-of-magnitude less compute than HRL tuning）
5. **Real-world**：8 个 Spot task + 4 个 G1 sim task，包括超越 robot payload 的 tire upright

它最大的 insight 在我看来是：**RL 和 MPC 不是二选一，而是分工**。RL 做 offline 的"学 general skill"，MPC 做 online 的"adapt to specific task"。这种 decomposition 把两种方法的力量放大、弱点互补，是个值得 deploy 的 pattern。

对 robotics practitioner 的启示：
- 不要一上来就 end-to-end RL，先看能不能 decompose
- Pre-train general WBC policy 是值得的 investment，未来能被多种 task steering
- Cost function 设计是 manipulation 的核心 — 把 task knowledge encode 进 cost 比 encode 进 reward 更 flexible
- Sample-based MPC 现在已经能 realtime，加上 policy-in-the-loop 后实用性大幅提升

## References

- [Sumo 项目主页](https://sumo.rai-inst.com/)
- [Relic paper](https://arxiv.org/abs/2506.07876)
- [MJLab](https://github.com/mujocolab/mjlab)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [Predictive Sampling](https://arxiv.org/abs/2212.00541)
- [Judo library](https://github.com/bdaiinstitute/judo)
- [MPPI original](https://ieeexplore.ieee.org/document/7487277)
- [CMA-ES tutorial](https://arxiv.org/abs/1604.00772)
- [Fast Contact-Implicit MPC (Le Cleac'h)](https://arxiv.org/abs/2107.05616)
- [Eureka (LLM reward design)](https://arxiv.org/abs/2310.12931)
- [L2R (Language to Rewards)](https://arxiv.org/abs/2306.08647)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Planning with Diffusion](https://arxiv.org/abs/2206.09278)
- [Steering Your Generalists](https://arxiv.org/abs/2410.13816)
- [SONIC humanoid WBC](https://arxiv.org/abs/2511.07820)
- [BeyondMimic](https://arxiv.org/abs/2508.08241)
- [OmniTrack](https://arxiv.org/abs/2602.23832)
- [Extreme Parkour](https://ieeexplore.ieee.org/document/10610200)
- [RAMBO](https://ieeexplore.ieee.org/document/10905808)
- [DTC](https://www.science.org/doi/10.1126/scirobotics.adh5401)
- [Pedipulate](https://ieeexplore.ieee.org/document/10611307)
- [Versatile Multicontact Planning](https://www.science.org/doi/10.1126/scirobotics.adg5014)
- [Trajectory Bundle Method](https://arxiv.org/abs/2509.26575)
- [Whole-body MPC with MuJoCo (Zhang et al.)](https://arxiv.org/abs/2503.04613)
- [HumanoidBench](https://arxiv.org/abs/2410.10840)
- [Slomo (Zhang et al.)](https://ieeexplore.ieee.org/abstract/document/10246373)
- [Learning to Search (Jain et al.)](https://arxiv.org/abs/2506.05294)
- [Residual MPC (Jeon et al.)](https://arxiv.org/abs/2510.12717)
