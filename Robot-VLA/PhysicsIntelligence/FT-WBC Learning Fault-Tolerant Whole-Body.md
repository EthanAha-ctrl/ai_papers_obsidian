---
source_pdf: FT-WBC Learning Fault-Tolerant Whole-Body.pdf
paper_sha256: c41a1b19d19434b6f8aaeb6bf5aaeb191cdf376cdd34fb67123c59733ee904f2
processed_at: '2026-08-19T08:27:05-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用最直觉的方式再讲一遍。

---

## 这篇 paper 到底在干嘛

想象一个四足机器人，背上驮着一条机械臂。它走路的时候，机械臂要去够东西。够不到怎么办？身体倾斜一下，pitch forward 或者 roll 一下，相当于"弯腰去捡地上的东西"。

问题来了：如果这时候某条腿的电机突然坏了（过热、卡死、扭矩衰减），你还傻乎乎地往那条腿的方向倾斜，重心就压到坏腿上了，坏腿撑不住，直接摔。

这篇 paper 就是解决这个问题的。**两个 module 搞定**：

1.  **Fault Estimator (FE)**：从腿的历史 proprioception 数据里猜，哪个关节坏了
2.  **Posture Adaptation Module (PAM)**：知道哪个关节坏了之后，把机械臂要的身体倾斜计划改一下，别往坏腿那边倒

就这样。核心 idea 就这么简单。但实现细节很精巧，我们展开说。

---

## 为什么光在 training 里注入 fault 不够用

你可能会想：直接在 RL training 里 randomize fault，让 policy 自己学不就行了？

Paper 试了，不行。原因很 intuitive：actuator fault 是 **hidden state**。你只看当前那一帧的 joint position 和 velocity，根本无法区分"这个关节是正常工作但遇到了大负载"还是"这个关节电机坏了在衰减"。这就像你开车，突然加速没力了，你不知道是上坡了还是发动机坏了，得看一段时间的历史才能判断。

Actor network 直接从 raw observation 里提取这种 long-term dynamics 信息很困难，因为 RL policy 天然偏向 short-term reactive behavior。所以 paper 把 fault inference 显式地拆出来，单独训一个 module 专门做这件事。

这和 RMA (Rapid Motor Adaptation) 的思路一脉相承：RMA 把 environment extrinsics 的推断从 actor 里拆出来，FT-WBC 把 actuator health status 的推断从 actor 里拆出来。

> 参考：[RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)

---

## FE 的细节

FE 的 input 是最近 $K=5$ 步的 lower-body observation history $o_{t-K:t}^{\text{leg}}$。每个 timestep 的 observation 包含 joint positions, joint velocities, previous actions 等等，共 56 维。5 步就是 $5 \times 56 = 280$ 维的 input。

通过一个 MLP 把它映射到一个 12 维的 fault probability vector：

$$\hat{\mathbf{f}}_t = \sigma(W_f \cdot \text{MLP}_{\text{enc}}(o_{t-K:t}^{\text{leg}}) + b_f)$$

变量解释：
- $\hat{\mathbf{f}}_t \in \mathbb{R}^{12}$：12 个 leg joint 各自的 predicted fault probability，每个值 $\hat{f}_j \in [0, 1]$
- $\sigma$：Sigmoid，输出概率
- $W_f, b_f$：projection layer 的 weight 和 bias

训练用 MSE loss，监督信号是 ground-truth binary fault vector $\mathbf{f}_t \in \mathbb{R}^{12}$，$f_j \in \{0, 1\}$，1 表示该 joint faulty。

$$\mathcal{L}_{\text{fault}} = \frac{1}{12}\sum_{j=1}^{12}(\hat{f}_j - f_j)^2$$

**Intuition**：这本质上是一个 implicit system identification。5 步 history 能捕捉到 actuator 对 PD 控制指令的响应延迟或衰减模式。healthy joint 给了 action 指令，joint 会快速响应；weakened joint 响应会迟滞或幅度不够；locked joint 完全不动。MLP 从这些 dynamics residual 里推断 fault 状态。

**Warm-up trick**：训练初期 FE 是随机初始化的，输出的是 garbage。如果直接把 garbage 喂给 Actor，会严重干扰 RL 的 early-stage learning。所以前 3000 iterations，Actor 直接拿 ground-truth fault label 用，之后才切换到 FE 的 output。这个 trick 在 teacher-student distillation 里很常见，确保 student 在 early phase 有一个稳定的 supervisor。

> 这和 [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/) 里用 VAE 编码 latent state 的思路很像，只不过 FT-WBC 的 FE 是一个极其轻量化的特化版本，专门推断 actuator health latent。

---

## PAM 的细节

PAM 的工作是：拿到 arm policy 要的 body posture plan（pitch 和 roll），结合 FE 给的 fault 信息，输出一个修改后的 safe posture command。

$$\tilde{\mathbf{u}}_t = \tanh(W_o h_t + b_o)$$

- $h_t$：融合了 posture plan $\Delta\mathbf{u}_t \in \mathbb{R}^2$ 和 fault vector $\hat{\mathbf{f}}_t \in \mathbb{R}^{12}$ 的 hidden feature，input dimension 是 $2 + 12 = 14$
- $W_o, b_o$：output layer 参数
- $\tanh$：把输出限制在 $[-1, 1]$

然后做物理限幅：

$$\tilde{c}_t^p = \text{clip}(0.4\tilde{u}_t^p, -0.3, 0.3)$$
$$\tilde{c}_t^r = \text{clip}(0.4\tilde{u}_t^r, -0.2, 0.2)$$

- $\tilde{u}_t^p, \tilde{u}_t^r$：tanh 输出的 pitch 和 roll 分量
- 0.4 是 scale factor
- Pitch 被限制在 $[-0.3, 0.3]$ rad（约 $\pm 17°$），Roll 在 $[-0.2, 0.2]$ rad（约 $\pm 11°$）

**Intuition**：PAM 相当于在 policy 输出端加了一个 **learned safety filter**。传统控制里你会用 Control Barrier Function (CBF) 或者 explicit constraint 来做 safety guarantee，PAM 是用 data-driven 的方式学到了这个 mapping。当 FE 预测到某条腿高概率 faulty 时，PAM 会主动 suppress 那些 risk-inducing 的 pitch/roll 指令，把 body posture 拉回 safe region。

> 参考：[A General Safety Framework for Learning-Based Control](https://arxiv.org/abs/2105.08730) — CBF + RL 的结合方向

---

## Reward 设计里的生物启发

Paper 的 reward function 里有两个特别有意思的 fault-oriented term：

### 1. Hold faulty-joint motion penalty

$$r_{\text{motion}}^{\text{fault}} = \sum_{j=1}^{12} m_j^{\text{fault}} \dot{q}_j^2$$

- $m_j^{\text{fault}} \in \{0, 1\}$：fault mask，joint $j$ faulty 时为 1
- $\dot{q}_j$：joint velocity
- weight = -0.2

这个 penalty 让 policy 尽量别动坏掉的关节。就像人崴脚之后，你会本能地减少那只脚的活动幅度，避免给受伤部位施加更多 load。

### 2. Contralateral compensatory support reward

$$r_{\text{axis}} = \exp\left(-\frac{y_{\text{healthy}}^2}{\sigma^2}\right), \quad \sigma = 0.2\text{m}$$

- $y_{\text{healthy}}$：healthy contralateral foot 在 yaw-aligned frame 下的 lateral coordinate
- $\sigma = 0.2$：控制 reward 的 sharpness

当左前腿坏了，policy 被 encouraged 让右前腿（对侧健康腿）向机身中矢面 靠拢。$y_{\text{healthy}}$ 越接近 0（越靠近中矢面），reward 越高。

**Intuition**：这完全是人类走路的直觉。你左腿受伤了，你会本能地把重心右移，同时右腿往中间收，让支撑面更稳定，重心投影落在安全区域内。Paper 让 RL 自己学出了这种 compensatory gait，但 reward shaping 给了一个正确的 inductive bias。

---

## Curriculum Learning：怎么注入 fault

Paper 定义了三种 fault state，用 $F_j \in \{0, 1, 2\}$ 表示：

- $F_j = 0$：healthy
- $F_j = 1$：locked joint（卡死，只允许在当前位置 $\pm 0.05$ rad 内微动）
- $F_j = 2$：weakened motor（扭矩乘以系数 $k_\tau \in [0, 1]$）

**Locked joint** 的定义：
$$q_j^{\text{cmd}'} = \text{clip}(q_j^{\text{cmd}}, c_j - q_{\text{thr}}, c_j + q_{\text{thr}}), \quad q_{\text{thr}} = 0.05\text{rad}$$

- $q_j^{\text{cmd}}$：policy 输出的 joint position command
- $c_j$：locked 时的 joint position
- $q_{\text{thr}}$：允许的微小抖动范围

**Weakened motor** 的定义：
$$\tau_j' = k_{\tau,j} \cdot \tau_j, \quad k_{\tau,j} \in [0, 1]$$

- $\tau_j$：PD controller 计算出的 desired torque
- $k_{\tau,j}$：weakening coefficient，1 = 完全健康，0 = 完全失效

**Weakening-severity curriculum**：

$$\rho(t) = \rho_0 + \text{clip}\left(\frac{t - t_0}{T}, 0, 1\right)(\rho_1 - \rho_0)$$

- $t$：当前 training iteration
- $\rho_0 = 0, \rho_1 = 0.3$：起始和终点的 severe-fault sample 占比
- $t_0 = 0, T = 5000$：curriculum 的时间跨度

随着训练进行，$\rho(t)$ 从 0 增到 0.3。这意味着训练后期有 30% 的 sample 其 $k_\tau$ 集中在 $[0, 0.25]$ 的 severe weakening 区间。如果用 uniform sampling，severe failure 的 sample 太稀疏，policy 学不到极端情况下的 recovery。

**Intuition**：如果一上来就大量注入 $k_\tau = 0$ 的极端 fault，网络容易 collapse 或者陷入 trivial solution（比如直接趴下不动）。Curriculum 让 network 先学会在 mild fault 下 locomotion，再逐步适应 severe fault，loss landscape 更平滑。这和你之前关于 training dynamics 和 plasticity loss 的讨论方向一致。

---

## 实验数据里的关键 insight

### Simulation 数据（Table 2）

| Fault Type | RD Survival | Ours Survival | RD Workspace | Ours Workspace |
|:---|:---|:---|:---|:---|
| Partial Weakening ($k_\tau=0.1$) Avg | 37.5% | 91.8% | 0.25 m³ | 0.73 m³ |
| Complete Weakening ($k_\tau=0.0$) Avg | 29.4% | 80.8% | 0.19 m³ | 0.66 m³ |

几个关键 insight：

1.  **Calf 和 Thigh 故障最难处理**：因为这两个 joint 直接承担支撑和推进力，且 base pitch 调整极度依赖它们。Calf 故障时，robot 几乎无法生成有效 ground reaction force。

2.  **Ablation study 的逻辑**：
    - `w/o FE`：性能暴降。Actor 完全 blind to fault，无法合成 compensatory force，会 divergent oscillation 然后摔。
    - `w/o PAM`：在 low-risk 情况下 workspace 甚至略高（因为没有 safety 限制），但 severe fault 下 survival rate 暴跌。
    - **结论**：FE 提供感知，PAM 提供极端情况下的 survival guarantee。两者缺一不可。

### Real-World Pick-and-Place（Table 3）

最震撼的是 **Locked Fault (LF) 在 90cm 高度** 的数据：

| Height | Condition | Survival | Success |
|:---|:---|:---|:---|
| 90 cm | Locked Fault | 14/20 (70%) | 9/20 (45%) |

Locked joint 是 **training distribution 之外** 的 fault type（训练只用 torque weakening），但 FT-WBC 依然 zero-shot 实现了 70% survival。Success 降到 45%，因为 PAM 保守地限制了 extreme upward pitch，宁可放弃任务也要保命。

**Intuition**：这体现了 PAM 的 "survival-first" 设计哲学。在真实世界部署时，你宁可机器人站着不动也不要它摔坏了。这个 trade-off 是通过 reward shaping 和 PAM 的 clip 机制自然涌现的，而非 hard-coded。

> 这种 "保守优先" 的策略和 [Safe RL](https://safety-gymmy.readthedocs.io/) 领域的 constraint enforcement 思路相通。

---

## PAM 的力学验证（Table 1）

Paper 用两个 metric 量化 PAM 的效果：

**Fault-leg load ratio**：
$$R_j = \frac{F_{z,j}}{\sum_i F_{z,i}}$$

- $F_{z,i}$：foot $i$ 的 vertical contact force
- $j$：faulty leg
- 衡量故障腿承受了多少比例的重力

**Fault-side tilt score**：
$$S_j = \max(0, \mathbf{d}_j^\top \bar{\mathbf{g}}_{xy})$$

- $\mathbf{d}_j = \mathbf{r}_j^{xy} / \|\mathbf{r}_j^{xy}\|_2$：从 base 到 faulty foot 的水平方向单位向量
- $\bar{\mathbf{g}}_{xy} = [g_x, g_y]^\top$：重力在水平面的投影
- 如果点积 > 0，说明重力方向偏向 faulty leg，有倾覆风险

结果：

| Method | $R_{\text{FL}}$ ↓ | $S_{\text{FL}}$ ↓ |
|:---|:---|:---|
| Robo-Duet | 41.7% | 0.0706 |
| w/o PAM | 41.4% | 0.0680 |
| **Ours** | **39.2%** | **0.0000** |

PAM 把 $S_{\text{FL}}$ 压到了精确的 0.0000。这意味着它通过调整 posture，把重力投影硬拉回了 safe region。这是一个非常直观的力学验证：PAM 确实在做 safety filtering，不是随便改改就交差的。

---

## 更远的联想

1.  **Multi-joint concurrent failure**：Paper 在 Limitations 里承认只处理了 single joint failure。如果多个 joint 同时坏，fault space 变成 $\binom{12}{k}$ 的高维组合，FE 的 12 维 output 可能不够表达。一个可能的改进方向是用 set-based representation（如 DeepSets 或 attention）来处理 variable-size fault combinations。

2.  **VLA (Vision-Language-Action) 的结合**：目前的高层指令 $c_t^{\text{arm}}$ 是预设的。如果接入 LLM/VLM，让它根据场景生成 "pick up the object on the front-right" 的指令，再由 arm policy 生成 posture plan，PAM 做底层 safety arbitration，就构成了一个完整的 VLA 机器人系统。这和 [Google RT-2](https://robotics-transformer2.github.io/) 或者 [Stanford Mobile ALOHA](https://mobile-aloha.github.io/) 的方向可以结合。

3.  **Neuro-symbolic safety**：PAM 是 learned safety filter。如果换成 symbolic CBF + neural policy，可能能得到更强的 hard safety guarantee。这和 [differentiable CBF](https://arxiv.org/abs/2210.04375) 的方向一致。

4.  **Domain randomization vs. fault randomization**：传统的 domain randomization 是在 environment extrinsics 上做 randomization（friction, mass, terrain）。FT-WBC 加了一层 **intrinsic randomization**（actuator health status），这是 sim-to-real 的一个新的 dimension。未来如果 motor wear-and-tear 能被建模成一个 stochastic process，training 时可以更精确地模拟真实硬件退化。

5.  **Plasticity 和 continual learning**：Curriculum learning 在这里用得很好，但如果 fault 是在 deployment 阶段动态变化的（比如电机逐渐老化），policy 能不能在线 adapt？这涉及到 continual learning 和 plasticity loss 的问题，你之前讨论过的那些方向在这里也适用。

---

## 一句话总结

**FT-WBC 把 fault-tolerant control 从纯 locomotion 扩展到了 loco-manipulation，核心是用 FE 做隐式的 system identification（猜哪个关节坏了），用 PAM 做 learned safety filtering（别往坏腿那边倒），两者配合让机器人在 actuator failure 下还能继续干活。**

---

Andrej，非常荣幸能为你解读这篇 paper。这篇 FT-WBC (Fault-Tolerant Whole-Body Control) 探讨了腿式 loco-manipulation (带机械臂的四足机器人) 在执行器发生故障时的 whole-body control 问题。核心挑战在于，机械臂会引起 Center-of-Mass (CoM) 的偏移以及动态扰动，当底层的 leg actuator 发生故障 (如电机过热、卡死、扭矩衰减) 时，系统极易失稳。传统的 fault-tolerant control 往往只关注 locomotion，而这篇 paper 将其延伸到了 loco-manipulation 的耦合问题中。

以下我将从 architecture、formula、training paradigms 以及 experimental data 几个维度为你深度解析，试图 build 你的 intuition。

### 1. Architecture Overview 与 Core Intuition

FT-WBC 采用了 **decoupled upper- and lower-body policy architecture**。直觉上讲，这种解耦极大地降低了 exploration space 的复杂度。
*   **Upper body (Arm policy $\pi_{\text{arm}}$)**: 负责跟踪 end-effector 的 target pose (记为 $c_t^{\text{arm}}$)，同时输出一个 desired base posture plan (包含 pitch 和 roll，记为 $\mathbf{u}_t$)。
*   **Lower body (Leg policy $\pi_{\text{leg}}$)**: 负责维持 base 的 stability 并合成 compensatory gait，接收 adapted posture command 和 fault information。

在 nominal 情况下，机械臂为了扩大 workspace，会要求 base 产生较大的 pitch 或 roll 倾角。如果此时某个支撑腿的 actuator 突然发生 weakening 或 locked 故障，盲目执行这个 posture plan 会让 CoM 直接偏移出 degraded support polygon，导致灾难性的摔倒。为了解决这个 conflict，paper 引入了两个关键 module：

#### 1.1 Fault Estimator (FE, $E_\theta$)
在真实物理世界中，actuator 的 fault (如电机内部磁极退磁或齿轮卡死) 属于 **hidden state**，无法直接测量。Actor network 很难仅凭当前帧的 shallow features (如 joint position 和 velocity) 做出有效反应。FE 的作用就是从历史的 proprioceptive 数据中推断当前的 actuator health status。

#### 1.2 Posture Adaptation Module (PAM, $G_\phi$)
PAM 充当了一个 **high-level intent arbitrator** (高层意图仲裁者)。它接收 arm policy 发来的 desired posture plan 以及 FE 预测的 fault vector，将其映射到一个 safe posture command space。如果 PAM 判断当前的 posture request 会将 CoM 推向已经失效的腿，它会主动 clip 或修改这个 command，执行 "survival-first" 的策略。

---

### 2. Mathematical Formulation 与技术细节

为了让你更直观地理解，我们来拆解其数学表达。

#### 2.1 Fault Estimator (FE) 的前向传播
FE 使用长度为 $K=5$ 的 lower-body observation history $o_{t-K:t}^{\text{leg}}$ 作为输入。通过一个 Multi-Layer Perceptron (MLP) 提取 latent feature，再通过 Sigmoid 输出概率：

$$
z_t = \text{MLP}_{\text{enc}}(\mathbf{o}_{t-K:t}^{\text{leg}})
$$
$$
\hat{\mathbf{f}}_t = \sigma(W_f z_t + b_f)
$$

*   $z_t$: 提取出的 fault-sensitive latent representation。
*   $\sigma$: Sigmoid activation function，将输出压缩到 $(0, 1)$ 区间。
*   $W_f, b_f$: 可学习的权重矩阵和偏置向量。
*   $\hat{\mathbf{f}}_t \in \mathbb{R}^{12}$: 预测的故障向量。四足机器人有 4 条腿，每条腿 3 个关节 (Hip, Thigh, Calf)，共 12 个 actuator。每个元素 $\hat{f}_j \in [0,1]$ 表示第 $j$ 个 leg joint 发生故障的 predicted probability。

FE 的训练采用了 Mean Squared Error (MSE) loss，监督信号是 binary ground-truth fault vector $\mathbf{f}_t \in \mathbb{R}^{12}$：

$$
\mathcal{L}_{\text{fault}} = \frac{1}{12} \sum_{j=1}^{12} (\hat{f}_j - f_j)^2
$$

**Intuition**: 这里本质上是一个 implicit system identification。类似于 Luenberger observer 或 Kalman Filter 的思想，只不过这里用 high-capacity 的 MLP 去拟合复杂的 non-linear dynamics 残差。前 5 步的 history 能够捕捉到 actuator 对 PD 控制指令响应的迟滞或衰减，从而推断出 weakening 程度。为了避免 RL 训练初期 FE 的 random guess 干扰 Actor 收敛，paper 采用了 Warm-up Mechanism，前 3000 iterations 直接给 Actor 喂 ground-truth label，这类似于 Teacher-Student distillation 中的早期阶段。

#### 2.2 Posture Adaptation Module (PAM) 的映射
PAM 接收 upper-body 的 plan $\Delta \mathbf{u}_t$ 和 FE 输出的 $\hat{\mathbf{f}}_t$，融合后输出 adapted posture command $\tilde{\mathbf{u}}_t$：

$$
\tilde{\mathbf{u}}_t = \tanh(\mathbf{W}_o h_t + \mathbf{b}_o)
$$

*   $h_t$: 融合了 posture plan 和 fault vector 的 hidden feature。
*   $\mathbf{W}_o, \mathbf{b}_o$: 输出层的 weight 和 bias。
*   $\tanh$: 激活函数，限制输出在 $[-1, 1]$。

随后进行物理限幅，映射到真实的 base pitch 和 roll 范围：

$$
\tilde{c}_t^p = \text{clip}(0.4 \tilde{u}_t^p, -0.3, 0.3)
$$
$$
\tilde{c}_t^r = \text{clip}(0.4 \tilde{u}_t^r, -0.2, 0.2)
$$

*   $\tilde{u}_t^p, \tilde{u}_t^r$: 分别是 pitch 和 roll 分量。
*   系数 0.4 和 clip 边界确保了 base 倾角在物理可行的范围内 (Pitch: $[-0.3, 0.3]$ rad, Roll: $[-0.2, 0.2]$ rad)。

**Intuition**: 这相当于在 RL policy 的输出端加上了一个 learned safety layer。传统控制理论中常用 Control Barrier Functions (CBFs) 来保证 safety，PAM 则是通过 data-driven 的方式学到了如何在故障状态下 tighten constraint。当 FE 预测到高概率的 fault 时，PAM 会主动 suppress 那些 risk-inducing 的 pitch/roll 指令。

#### 2.3 Fault-Oriented Reward Design
Reward function 中加入了一些极具生物学启发的 term，引导 policy 学会 compensatory behavior：

**Hold faulty-joint motion penalty**:
$$
r_{\text{motion}}^{\text{fault}} = \sum_{j=1}^{12} m_j^{\text{fault}} \dot{q}_j^2
$$
*   $m_j^{\text{fault}} \in \{0, 1\}$: binary fault mask。如果 joint $j$ faulty (即 $F_j > 0$)，则为 1。
*   $\dot{q}_j$: joint velocity。
这个 penalty 鼓励 policy 减少对故障关节的使用，避免给损坏的 motor 施加不稳定的指令。

**Contralateral compensatory support reward**:
$$
r_{\text{axis}} = \exp\left(-\frac{y_{\text{healthy}}^2}{\sigma^2}\right), \quad \sigma = 0.2 \text{ m}
$$
*   $y_{\text{healthy}}$: 健康的对侧腿在 yaw-aligned frame 下的 lateral coordinate。
当一侧腿发生故障，policy 会被鼓励让对侧的健康腿向机身中矢状面 靠拢。这在生物学上非常 intuitive，就好比人类崴脚后，会本能地把重心移向健康的一侧，并让健康腿内收以提供更稳定的支撑。

#### 2.4 Weakening Severity Curriculum
为了增加对 severe actuator degradation 的 robustness，paper 对 fault injection 采用了 curriculum learning：

$$
\rho(t) = \rho_0 + \text{clip}\left(\frac{t - t_0}{T}, 0, 1\right) (\rho_1 - \rho_0)
$$
*   $t$: 当前训练 iteration。
*   $\rho_0 = 0, \rho_1 = 0.3, t_0 = 0, T = 5000$。
随着训练进行，$\rho(t)$ 从 0 线性增加到 0.3。这意味着在训练后期，有高达 30% 的样本其 weakening coefficient $k_\tau$ 会集中在 $[0, 0.25]$ 的近完全失效区间。这有效缓解了 uniform sampling 导致的 severe failure 样本稀疏问题。

---

### 3. Experimental Data 深度解析

我们来看实验数据，从中挖掘出更深的物理意义。

#### 3.1 Simulation Experiments (Table 2)
在 Partial Weakening ($k_\tau = 0.1$) 和 Complete Weakening ($k_\tau = 0.0$) 下，FT-WBC 相比 baseline (Robo-Duet) 有质的飞跃。

| Faulty Joint | Partial Survival Rate | Complete Survival Rate | Partial Workspace | Complete Workspace |
| :--- | :--- | :--- | :--- | :--- |
| Average | 91.8% (Ours) vs 37.5% (RD) | 80.8% (Ours) vs 29.4% (RD) | 0.73 m³ (Ours) vs 0.25 m³ (RD) | 0.66 m³ (Ours) vs 0.19 m³ (RD) |

*   **Calf 和 Thigh 故障最难处理**: 从表中可以看出，Calf 和 Thigh 故障导致的性能下降远大于 Hip 故障。这是因为 Calf 和 Thigh 直接承担了主要的支撑和推进力，并且 base pitch 的调整极度依赖这两个关节。如果它们 weakening，robot 几乎无法生成有效的 ground reaction force。
*   **Ablation Study 的启示**: `w/o FE` 的性能极差，因为 leg policy 完全 blind to fault，无法合成 compensatory force，会迅速 divergent 振荡并摔倒。`w/o PAM` 在低风险情况下 workspace 甚至略高，这是因为没有 safety 限制，arm 可以肆无忌惮地伸展；但在 severe fault 下，其 survival rate 暴跌。这说明 **FE 提供了感知能力，而 PAM 提供了在极端情况下的 survival guarantee**。

#### 3.2 Real-World Pick-and-Place (Table 3)
真实世界实验设计了极具挑战的 Ground-to-Table 任务，目标高度包括 50cm, 75cm 和 90cm。90cm 目标迫使 quadruped 进入 extreme upward pitch，这在支撑腿受损时极其危险。

| Height | Cond. | Survival | Success |
| :--- | :--- | :--- | :--- |
| 90 cm | Locked Fault (LF) | 14/20 (70%) | 9/20 (45%) |

值得注意的是，这里测试了 **Locked Fault (LF)**，即关节完全卡死，这属于 training distribution 之外的 unmodeled fault。FT-WBC 依然实现了 70% 的 survival rate。虽然 success rate 降到 45%，但这正是 PAM "survival-first" 设计哲学的体现：它保守地限制了 extreme upward pitch，宁可牺牲任务成功率，也要防止 CoM 逃逸出 degraded support polygon 导致机器人损坏。

#### 3.3 Table 1: PAM 的力学验证
Table 1 定义了两个极佳的 metric 来量化 PAM 的作用：
1.  **$R_j = F_{z,j} / \sum_i F_{z,i}$**: 故障腿承受的垂直接触力占总接触力的比例。如果 PAM 不工作，robot 盲目倾斜会导致 $R_j$ 极高，压垮故障腿。实验显示 PAM 将 $R_{\text{FL}}$ 从 41.7% 降到了 39.2%。
2.  **$S_j = \max(0, d_j^\top \bar{\mathbf{g}}_{xy})$**: Fault-side tilt score。$d_j$ 是从 base 指向 faulty foot 的水平向量，$\bar{\mathbf{g}}_{xy}$ 是重力在水平面的投影。如果两者点积大于 0，说明重力方向偏向 faulty leg，有倾覆风险。PAM 成功将 $S_{\text{FL}}$ 压到了 0.0000，这意味着它通过调整 posture，强行把重力投影拉回了安全区域。这非常直观地展示了 PAM 作为一个 safety filter 的力学效果。

---

### 4. 延伸联想与 Web Links

阅读这篇 paper 时，我脑海中浮现出许多相关的技术脉络，与你之前的 work 和整个 Deep Learning 生态有很多交汇点：

1.  **Hidden State Inference 与 World Models**: FE 本质上是在做 model-based RL 中的 latent state inference。这让你可能联想到你在 Liang-Kong Wang 和 Jayagen 开发的 World Models 里的 VAE，或者 Danijar Hafner 的 Dreamer 系列通过 recurrent state space model (RSSM) 推断 unobserved 的情况。在这里，FE 是一个极其轻量化的特化版本，专门推断 actuator 的 health latent。
    *   *Reference*: [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/)
    *   *Reference*: [Dreamer (Hafner et al.)](https://dream-rl.github.io/)

2.  **RMA (Rapid Motor Adaptation) 的延伸**: Appendix A.2 提到他们保留了 RMA-style 的 adaptation module 用来预测 friction 和 restitution。FT-WBC 的 FE 可以看作是 RMA 思想在 fault-tolerance 领域的扩展。RMA 推断环境动力学，FE 推断自身本体动力学 的退化。
    *   *Reference*: [RMA: Rapid Motor Adaptation (Kumar et al.)](https://arxiv.org/abs/2107.04034)

3.  **Constraint Enforcement in RL**: PAM 的设计让我想到 Constrained Policy Optimization (CPO) 或者 Safe RL。传统的 safe RL 是在 reward 里加 penalty，而 PAM 是直接在 action space 层面做了一个 learned projection。如果将来的工作能将 Control Barrier Function (CBF) 与这个 PAM 结合，可能会得到更强的 hard safety guarantee。
    *   *Reference*: [A General Safety Framework for Learning-Based Control (Achermann et al.)](https://arxiv.org/abs/2105.08730)

4.  **Loco-Manipulation 的 Sim-to-Real Gap**: 他们在 Jetson Orin Nano 上以 50Hz 运行，直接 zero-shot sim-to-real。这种成功极大地依赖于 Isaac Gym 的物理仿真精度以及 domain randomization。这和 Stanford 的 Mobile ALOHA 或者 Berkeley 的 Robo-Duet (本文 baseline) 走的技术路线一脉相承。
    *   *Reference*: [Robo-Duet: Learning a Cooperative Policy (Pan et al.)](https://arxiv.org/abs/2502.20585) (由于 paper 未发表全链接，附上 ArXiv 搜索) https://ft-wbc.github.io/ (项目主页)

5.  **Fault Curriculum 与 Plasticity Loss**: 论文中的 curriculum 策略，随着 iteration 增加，逐渐注入更严重的 fault。这让我想到你之前关于 training dynamics 和 plasticity loss 的讨论。如果一开始就注入 $k_\tau = 0$ 的极端故障，网络极易陷入 local minima 或者发生 mode collapse。Curriculum 学习在这里起到了平滑 loss landscape 的作用。

6.  **多模态/大模型融合的未来**: 目前的高层指令 $c_t^{\text{arm}}$ 还是预设的或由传统的 motion planning 给定的。如果将大语言模型 (LLM) 接入，作为 high-level intent generator，LLM 可以根据环境视觉信息生成 "pick up the object on the front-right" 的指令，然后由 arm policy 生成 posture plan，PAM 再根据底层 hardware fault 进行 safety arbitration。这就构成了一个完整的 neuro-symbolic 或 VLA (Vision-Language-Action) 机器人系统。

总而言之，这篇 paper 通过将 fault estimation 和 posture adaptation 显式地解耦到 RL 框架中，极大地提升了腿式机器人在极端硬件退化下的生存能力和任务连续性。FE 和 PAM 这两个模块的设计虽然简洁，但在物理逻辑上非常自洽，是其能够实现 zero-shot generalization 到 unmodeled locked fault 的关键所在。
