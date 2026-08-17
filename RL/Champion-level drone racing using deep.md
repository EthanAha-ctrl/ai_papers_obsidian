---
source_pdf: Champion-level drone racing using deep.pdf
paper_sha256: c9bcbd98ddbf2b0a18b4cbdbc4e62e429c6a659e1b1054b5d1c2e0219b8da40f
processed_at: '2026-08-03T15:31:13-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Swift — 用人话讲

## 一、这事儿到底牛在哪

你想想 AlphaGo 2016年 beat Lee Sedol，当时全世界炸了。但 AlphaGo 下棋，棋盘是 19×19，state 完全 observable，rules deterministic，没有任何 physical noise。

后来 AlphaStar beat Dota 2 大师，GT Sophy beat Gran Turismo 冠军。这些都很猛，但都是在 **video game** 里。Video game 的 physics engine 是 deterministic 的，state 是 perfect observable 的，sim 就是 real，sim-to-real gap 是 zero。

Drone racing 不一样。Drone 在真实空气里飞，撞风、撞气流、电池掉电压、motor 有 friction、camera 有 motion blur。State estimation 永远 noisy。Sim 永远不是 real。

Swift 是 first autonomous physical robot 在 real physical sport 里达到 world champion 水平。这是 mobile robotics 的 AlphaGo moment。 

参考: https://www.nature.com/articles/s41586-023-06419-4

---

## 二、Drone Racing 这个 sport 到底是啥

FPV (First-Person View) drone racing 是这么个 sport：

- Pilot 戴一个 VR headset
- Drone 上有 camera，把视频流实时传到 headset
- Pilot 从 drone 的视角"看"环境，飞过一连串 gate
- Speed > 100 km/h，加速度几倍 gravity，thrust 5倍自身重量
- Track 通常 30×30×8 米的体积里布 7 个 gate，一圈 75 米

人类 pilot 训练多年，反应延迟约 220ms，camera 120Hz。

Swift 反应延迟 40ms，camera 30Hz。Swift 用 IMU（人类 pilot 没这待遇，因为他们不在 drone 上感受加速度）。Swift compute 在 NVIDIA Jetson TX2 上。

---

## 三、Swift 系统长啥样 — 两段式架构

想象你在开车。你需要 (1) 知道自己在哪、速度多少、朝向哪 (2) 决定怎么打方向盘和油门。Swift 一样分两段：

### 第一段：Observation Policy（感知）

这段又分三层：

**Layer 1: VIO (Visual-Inertial Odometry)**

硬件是 Intel RealSense T265（已经停产，但是个 commodity VIO sensor）。Camera + IMU 融合，100Hz 输出 drone 的 metric pose estimate。

VIO 的痛点是 **drift**：高速飞行时 motion blur 导致 visual feature tracking 失败，linear odometry 累积误差。一圈 17 秒下来，drift 可能到米级。

参考: https://www.intelrealsense.com/tracking-camera-t265/

**Layer 2: Gate Detector (CNN)**

一个 U-Net，输入 384×384 grayscale image，输出每个 gate 的 4 个 corner 的 segmentation。U-Net 是 encoder-decoder + skip connection，特别适合 pixel-level prediction。每层 filters 数 (8,16,16,16,16,16)，kernel size (3,3,3,5,7,7)。

部署到 Jetson 上用 TensorRT + FP16，40ms 一次 inference（25Hz）。

参考: https://arxiv.org/abs/1505.04597

**Layer 3: Kalman Filter**

VIO 高频（100Hz）但 drift。Gate detection 低频（25Hz）但准（gate 是 known landmark，位置在 track layout 里事先知道）。

Kalman filter 状态 x = [p_d, v_d] ∈ ℝ^6，p_d 是 VIO 的 position drift，v_d 是 drift velocity。

公式(12)-(14) 是经典 Kalman：

$$x_{k+1} = F x_k, \quad P_{k+1} = F P_k F^T + Q$$

F 是 constant velocity transition matrix：
$$F = \begin{bmatrix} I^3 & dt \cdot I^3 \\ 0 & I^3 \end{bmatrix}$$

意思是 drift 假设按 constant velocity 演化。

Update step:
$$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R)^{-1}$$
$$x_k^+ = x_k^- + K_k(z_k - H x_k^-)$$

z_k 是 gate detection 给的 pose measurement。R 是 measurement covariance，用 IPPE (Infinitesimal Plane-based Pose Estimation) sampling 估计 — 对每个 gate 跑 20 次 perturbed corner detection，看 pose 分布的 covariance。

关键：**只用 gate detection 纠正 translation，不动 orientation**。因为 VIO 的 orientation 估计质量好（gravity 给绝对参考），gate detection 频率低不值得为它改 orientation。

参考 IPPE: https://link.springer.com/article/10.1007/s11263-014-0725-4

### 第二段：Control Policy

一个 **2-layer MLP**，128 hidden units/layer，LeakyReLU(0.2)。

输入 o_t ∈ ℝ^31：
- 15D robot state：position(3) + velocity(3) + rotation matrix(9)
- 12D next gate relative pose：4 corners × 3 coordinates
- 4D previous action

输出 a_t ∈ ℝ^4：mass-normalized collective thrust + 3-axis body rates。

为什么输出 thrust + body rates 而不是 individual motor commands？因为 low-level Betaflight PID 来处理 motor dynamics，policy 给的命令更抽象、更 transferable。这和 OpenAI Rubik's Cube 用 fingertip torques 而不是 joint angles 类似。

CPU 上 inference 8ms（125Hz control frequency）。

训练算法 PPO。Critic 训练时用 privileged info（exact pose），policy 部署时用不着 critic。

参考 PPO: https://arxiv.org/abs/1707.06347
参考 asymmetric actor-critic: https://arxiv.org/abs/1810.06494

---

## 四、Simulation — 想要 sim-to-real，先得 sim 够真

公式(1) 是 quadrotor dynamics：

$$\dot{x} = \begin{bmatrix} \dot{p}_{WB} \\ \dot{q}_{WB} \\ \dot{v}_W \\ \dot{\omega}_B \\ \dot{\Omega} \end{bmatrix}$$

变量：
- p_{WB}: body frame B 相对 world frame W 的 position
- q_{WB}: attitude quaternion
- v_W: inertial velocity (world frame)
- ω_B: body rates
- Ω: 4 个 motor 实际转速

受力：f_prop (propeller lift) + f_aero (drag, dynamic lift, induced drag) + gravity。

Torque：τ_prop (propeller torque) + τ_mot (motor yaw) + τ_aero (blade flapping) + τ_iner (gyroscopic = -ω_B × Jω_B)。

Propeller model 是 quadratic：f_i = c·Ω_i²，τ_i = c_d·Ω_i²。

### Aerodynamics — Grey-Box Polynomial

First-principles 建模 aerodynamics 很难（blade flapping, induced drag, wake interaction...）。他们用 data-driven grey-box model：

$$f_x \sim v_x + v_x|v_x| + \bar{\Omega}^2 + v_x \bar{\Omega}^2$$

每项物理意义：
- v_x: linear drag
- v_x|v_x|: quadratic drag (保留符号)
- Ω²: propeller wake 效应
- v_x·Ω²: forward velocity × propeller-induced flow 的 cross effect

6 个输出 (f_x, f_y, f_z, τ_x, τ_y, τ_z)，每个用 linear + quadratic combination。系数从 real flight data 辨识，motion capture 提供 ground truth force。

这是 track-specific aerodynamics — 类比人类 pilot 在 track 上训练一周熟悉气流。

参考: https://rpg.ifi.uzh.ch/docs/RAL21_Kaufmann.pdf

### Battery + ESC Model

公式(5) motor power:
$$P_{mot} = \frac{c_d \Omega^3}{\eta}$$

Ω³ 关系：功率随转速立方增长（这就是为啥高速飞行功耗爆炸）。

公式(6) steady-state motor speed mapping:
$$\Omega_{i,ss} \sim 1 + U_{bat} + \sqrt{u_{cmd,i}} + u_{cmd,i} + U_{bat}\sqrt{u_{cmd,i}}$$

U_bat 是当前电池电压（随放电和功率 draw 变化），u_cmd 是 PWM command。建模这玩意儿重要，因为电池电压掉下来同样 PWM 给出的 motor speed 变了。

参考: https://rpg.ifi.uzh.ch/docs/RAL22_Bauersfeld.pdf

### Betaflight PID quirks

仿真精确建模 Betaflight 的几个怪癖：
- D-term reference 恒为 0（pure damping）
- I-term 在 throttle cut 时 reset
- Motor saturation 时 body rate control 优先（按比例 downscale 所有 motor signals）

这些细节让 sim 预测 individual motor command 误差 < 1%。

---

## 五、Reward — Perception-Aware Racing 是关键创新

总 reward:
$$r_t = r_t^{prog} + r_t^{perc} + r_t^{cmd} - r_t^{crash}$$

### Progress Reward

$$r_t^{prog} = \lambda_1 [d_{t-1}^{Gate} - d_t^{Gate}]$$

d_t 是 drone 到 next gate 的距离。每步 reward = 距离缩小量。Dense shaped reward，比 sparse "pass gate = 1" 好训得多。

### Perception-Aware Reward

$$r_t^{perc} = \lambda_2 \exp[\lambda_3 \cdot \delta_{cam}^4]$$

- δ_cam: camera optical axis 和 next gate center 的夹角
- λ_3 = -10.0（负的）
- δ_cam^4 让 reward 对角度偏差很敏感

这是 paper 的核心 insight 之一：**让 policy 主动把 camera 指向 next gate**，从而 improve perception quality。

为啥这重要？RL agent 默认是 passive perceiver — 只优化 task reward。但 drone racing 里 perception quality 直接影响 control 性能。如果 policy 学会飞过 gate 但 camera 指向别处，VIO drift 会爆。

这就是 **Active Perception** in RL — agent 学会主动 orient sensor 去 maximize information gain。

参考: https://rpg.ifi.uzh.ch/docs/IROS19_Kaufmann.pdf

### Action Smoothness

$$r_t^{cmd} = \lambda_4 a_t^\omega + \lambda_5 \|a_t - a_{t-1}\|^2$$

a_t^ω 是 body rates，惩罚大角速度。第二项是 jerk 惩罚。λ_4 = -2e-4, λ_5 = -1e-4，都很小。

### Crash Penalty

$$r_t^{crash} = 5.0 \text{ if } p_z < 0 \text{ or gate collision}$$

Terminal cost，触发则 episode 结束。

### 训练超参

- γ = 0.99, ε = 0.2 (PPO clipping)
- 100 parallel agents
- episode 1500 steps
- Adam, lr = 3e-4
- 1×10⁸ interactions，50 分钟训练完 (i9-12900K + RTX 3090)
- Fine-tuning: 2×10⁷ interactions

注意：**不做 domain randomization on dynamics**，而是 fine-tune with real data。这是和 OpenAI Rubik's Cube 思路的关键区别。

---

## 六、Sim-to-Real — Residual Modeling 是 magic

仿真再真也不是 real。两类 discrepancy：

### Residual Observation Model — Gaussian Process

VIO drift 是 stochastic 的（同一 state 两次 VIO 输出不同）。用 GP 建模 residual position/velocity/attitude（9 个 1D GP）。

公式(10) RBF kernel + noise:
$$\kappa(z_i, z_j) = \sigma_f^2 \exp\left(-\frac{1}{2}(z_i - z_j)^T L^{-2}(z_i - z_j)\right) + \sigma_n^2$$

- z_i, z_j: data features (robot state)
- L: diagonal length scale matrix
- σ_f: signal variance
- σ_n: noise variance

GP 给 posterior distribution over residual functions。从 posterior sample 100 个 realizations，每个是 temporally consistent drift 轨迹。Fine-tuning 时 rollout 不同 realization，policy 学到 robust to perception noise distribution。

为啥用 GP 不用 NN？小数据集（3 次 rollout ~50 秒）下 GP sample efficient，uncertainty quantification 天然支持，overfitting 风险低。

### Residual Dynamics Model — k-NN

$$a_{res} = KNN(s, c)$$

s 是 state，c 是 commanded thrust，a_res 是 residual acceleration。k=5，dataset 800-1000 samples。

为啥 dynamics 用 k-NN，perception 用 GP？Empirical 发现：
- Perception residuals **stochastic** → 需要 distribution model (GP)
- Dynamics residuals **largely deterministic** → non-parametric regression 够

选 model 要 match data nature。

参考 ANYmal: https://www.science.org/doi/10.1126/scirobotics.aau5872

### 数据效率惊人

只用 **3 次 real-world rollout (~50 秒飞行)** 来识别 residual models。一圈 lap 约 17 秒，3 次 rollout ≈ 9 laps，覆盖 track 所有 segments。

对比 AlphaGo 用 millions of self-play games — physical robot 数据贵，必须 sample efficient。

---

## 七、Results — 真刀真枪 race 人类世界冠军

### Head-to-Head Races

| Matchup | Races | Best Time | Wins | Losses | Win Ratio |
|---|---|---|---|---|---|
| A. Vanover (2019 DRL World Champion) vs Swift | 9 | 17.956s | 4 | 5 | 0.44 |
| T. Bitmatta (2x MultiGP World Cup Champion) vs Swift | 7 | 18.746s | 3 | 4 | 0.43 |
| M. Schaepper (3x Swiss Champion) vs Swift | 9 | 21.160s | 3 | 6 | 0.33 |
| **Total** | **25** | **17.465s** | **15** | **10** | **0.60** |

Swift 最快 lap 17.465s，领先人类最快 0.5 秒。

10 次失败：
- 40% 撞对手（race 时 drone 互相干扰气流）
- 40% 撞 gate
- 20% 慢

### Track 分段分析（Extended Data Table 1d）

Track 分 4 段：

**Segment 1 (start, 起飞段):**
- Swift: 6.52 m/s, 1.27s
- Vanover: 5.62 m/s, 1.42s
- Bitmatta: 5.93 m/s, 1.36s
- Swift 起飞快 120ms，加速更猛

**Segment 2 (中段):**
- Swift 13.72 m/s, 1.82s
- Vanover 13.83 m/s, 1.83s
- 持平

**Segment 3 (Split-S — 最难的一段):**
- Swift: 13.70 m/s, **1.96s**, path **26.89m**
- Vanover: 13.86 m/s (更快), 2.09s, path 28.91m
- Swift 走 tighter line，path 更短，时间更快
- Vanover 速度高但 path 长

**Segment 4:**
- 持平

**Full Race:**
- Swift: avg 13.11 m/s, 866W, 29.16N thrust, **17.46s**, path **228.85m**
- Vanover: 12.96 m/s, 843W, 28.65N, 17.96s, path **232.79m**
- Swift 走了更短 path

### 关键 Insight: Swift 怎么赢的

1. **起飞快**（120ms 反应优势）
2. **Split-S 走 tighter line**（RL value function 让 long-horizon optimization，human pilot 只 plan 1 gate ahead）
3. **平均 path 更短**（228.85m vs 232.79m）
4. **更高 thrust 用得更满**（一直贴 actuation limit）

人类 pilot 的优势：**strategic thinking**。如果 human pilot 有 lead，会 slow down 降 crash 风险。Swift 不知道对手在哪，永远 push 最快 — 这意味着 Swift 领先时 over-risk，落后时 under-risk。如果加 opponent-aware RL，Swift 可能更猛。

参考 human pilot horizon study: https://ieeexplore.ieee.org/document/9372805

### Lap Time 分布

Swift: 低 mean, 低 variance — 一致 push
Human: 高 mean, 高 variance — 每 lap 决定要不要 push

---

## 八、Simulation Ablation — 证明 residual model 是关键

Extended Data Table 1c，四种 setting × 四种 approach：

**Setting 1 (idealized dynamics + ground-truth obs):**
- Zero-shot RL: 4.88s, 100% completion
- Domain Rand: 5.06s, 100%
- **Time-Opt + MPC: 4.60s, 100%** ← MPC 在理想条件下最快
- Ours: 4.88s, 100%

**Setting 2-4 (有任何 domain shift):**
- Zero-shot, Domain Rand, Time-Opt+MPC: 全部 collapse (0-19% completion)
- **Ours: 100% completion, 5.20-5.42s**

关键 claim：**传统 MPC 在 perfect conditions 下能 beat RL，但 robustness 崩盘。RL + residual modeling 在 domain shift 下保持性能**。

即使 baselines 也 access GP noise model (Extended Data Table 1b)：
- Zero-shot + GP: 42% completion
- Domain Rand + GP: 19%
- Time-Opt + GP: 19%
- **Ours + GP: 100%**

即使在 fair comparison 下，Ours 还是唯一 100%。原因：我们 fine-tuning 时 sample GP uncertainty distribution，policy 真正 robust to perception noise distribution，不只是 mean。

参考 Time-Opt MPC: https://www.science.org/doi/10.1126/scirobotics.abh1221

---

## 九、Hardware

- Drone 重量：870g
- 最大 static thrust：35N
- **Thrust-to-weight ratio: 4.1** (人类极限竞速 drone 4-5)
- Frame: Armattan Chameleon 6"
- Motors: T-Motor Velox 2306
- Props: 5" 3-bladed
- Compute: NVIDIA Jetson TX2 + Connect Tech Quasar carrier
  - 6-core CPU @ 2GHz
  - 256 CUDA cores GPU @ 1.3GHz
- Sensors: Intel RealSense T265 (VIO @ 100Hz, grayscale camera @ 30Hz)
- Low-level: STM32 @ 216MHz, Betaflight firmware

**公平对比**: 人类 pilot 用相同硬件，但 Jetson + T265 换成 ballast weight，确保重量、shape、propulsion 一样。

**Latency 对比:**
- Swift sensorimotor latency: **40ms**
- Human expert latency: **220ms**

但 Swift camera 30Hz vs human 120Hz — refresh rate 是 Swift 劣势。

参考: https://github.com/betaflight/betaflight

---

## 十、几个容易错过的技术细节

### 1. Quaternion vs Rotation Matrix

Sim 内部用 quaternion，但 policy input 用 rotation matrix (9D)。因为 quaternion 有 double covering（q 和 -q 同一 rotation），网络会 confused。Rotation matrix 唯一，没歧义。

### 2. Privileged Critic

Value network 训练时 access exact pose/orientation/velocity（privileged），policy network 只 access observation。这是 asymmetric actor-critic。Critic 部署时不用，所以可以"作弊"用 privileged info。这让 critic 给更准 value estimate 来 guide policy learning。

### 3. Mass-Normalized Thrust

Policy 输出 mass-normalized collective thrust (单位 m/s²)，不是 raw thrust (N)。这 normalize 掉质量，让 policy transferable 到不同 drone。

### 4. Random Gate Initialization

训练 reset 到 random gate + bounded perturbation around previously observed state。比从同一起点训练好 — policy 学会从 track 任何位置恢复，有 curriculum + generalization 效果。

### 5. GP Sampling 不是 Mean

Fine-tuning 不是用 GP predict mean，是从 posterior **sample 100 个 realizations**，每个是 temporally consistent drift 轨迹。每次 rollout 用不同 realization，policy 学到 robust to distribution，不只是 mean shift。

### 6. IPPE for Gate Pose

Gate 是 planar object，4 个 corner 定义 plane。IPPE (Infinitesimal Plane-based Pose Estimation) 给 closed-form pose 解。比 general PnP 在 planar case 下更准、更稳。

### 7. Motor Saturation Handling

仿真精确 model Betaflight 在 motor saturation 时 body rate control 优先 (proportional downscale 所有 motor signals)。如果 sim 不 model 这，real flight 遇 saturation 时 policy 会 behave unexpectedly。

---

## 十一、为啥 Swift 赢 — 用人话总结

1. **Hybrid architecture** — Classical VIO + Learning-based gate detection + Learning-based control。各模块单独 optimize，比 end-to-end 好 engineer。

2. **Residual modeling > Domain randomization** — 收集少量 real data (50秒) 建 residual model，fine-tune policy 到 specific domain shift。比 broad robustness 在 narrow task 上更 effective。

3. **Perception-aware reward = Active perception** — Reward 让 policy 主动把 camera 指向 next gate，improve VIO accuracy，positive feedback loop。

4. **Value function = Long-horizon planning** — RL policy 的 value function 隐含 encode long-horizon optimization，比 MPC (horizon limited by compute) 和 human (1 gate ahead) 看得远。

5. **Hardware fairness** — Swift 和 human pilot 用同 drone 同重量同 propulsion。Swift 唯一额外优势是 IMU 和低 latency (40ms vs 220ms)，但 camera 30Hz vs 120Hz 是劣势。

6. **Data efficiency** — 50秒 real flight data 足够 identify residual models，因为 residual model 是 low-dim，GP/k-NN 是 sample efficient non-parametric methods。

---

## 十二、Limitations

1. **No crash recovery** — Human pilot crash 后能继续飞（如果 hardware OK），Swift 没 train recovery
2. **Appearance sensitivity** — Gate detector 依赖 training appearance，光照变化可能失效
3. **No opponent awareness** — Swift 不知道对手在哪，导致 strategic suboptimality
4. **Single track specialization** — Fine-tune 到一个 track，跨 track generalization 未测试
5. **30Hz camera** — 比 human 120Hz 慢

未来方向：diverse condition training、multi-agent RL for opponent awareness、crash recovery、cross-track meta-learning。

---

## 十三、几个能让你 build intuition 的对比

### Swift vs AlphaGo

| | AlphaGo | Swift |
|---|---|---|
| Environment | Board game | Real physical |
| State observability | Perfect | Noisy (VIO drift) |
| Physics | Deterministic | Aerodynamics, battery, motor |
| Sim-to-real gap | None | Large |
| Opponent | Self-play | Real human champions |
| Compute | TPU cluster | Jetson TX2 |

### Swift vs OpenAI Rubik's Cube

| | Rubik's Cube | Swift |
|---|---|---|
| Sim-to-real | Domain Randomization (ADR) | Residual models |
| Data efficiency | Massive DR training | 50s real flight |
| Task | Manipulation | Agile locomotion |
| Speed | Slow, careful | 100+ km/h |
| Performance | Beat human | Beat world champion |

### Swift vs GT Sophy

| | GT Sophy | Swift |
|---|---|---|
| Environment | Video game physics | Real physics |
| Sim-to-real gap | Zero (game engine) | Large |
| State observability | Perfect | Noisy |
| Speed | 300 km/h virtual | 100 km/h real |

GT Sophy 容易些，因为 game engine 是 deterministic，state perfect。Swift 要解决真实 physical world 的 noise。

### Swift vs Time-Optimal MPC

| | MPC | Swift |
|---|---|---|
| Ideal conditions | **4.60s** (faster) | 4.88s |
| Real conditions | 9% completion | **100%** |
| Robustness | Brittle | Robust |

MPC 在 perfect conditions 下能 beat RL，但 robustness 不行。RL + residual model 在 real conditions 下保持性能。

---

## 十四、一句话总结

**Swift 证明 RL 不止能玩 video game，能在 real physical competitive sport beat human world champion。关键是 hybrid system + residual sim-to-real modeling + perception-aware reward，用 50 秒 real data 就能让 RL policy 适应真实 world。**

这是 mobile robotics 的 AlphaGo moment。下一步扩展到 autonomous driving、personal robotics、aerial delivery。

---

## 参考链接汇总

- Paper: https://doi.org/10.1038/s41586-023-06419-4
- UZH RPG group: https://rpg.ifi.uzh.ch/
- Pseudocode & data: https://doi.org/10.5281/zenodo.7955278
- Agilicious framework: https://agilicious.dev/
- PPO: https://arxiv.org/abs/1707.06347
- U-Net: https://arxiv.org/abs/1505.04597
- Asymmetric actor-critic: https://arxiv.org/abs/1810.06494
- IPPE: https://link.springer.com/article/10.1007/s11263-014-0725-4
- GP for ML: http://www.gaussianprocess.org/gpml/
- ANYmal (residual model inspiration): https://www.science.org/doi/10.1126/scirobotics.aau5872
- Prior Loquercio 2021: https://www.science.org/doi/10.1126/scirobotics.abg5810
- Time-Optimal MPC: https://www.science.org/doi/10.1126/scirobotics.abh1221
- Aerodynamics model: https://rpg.ifi.uzh.ch/docs/RAL21_Kaufmann.pdf
- Battery model: https://rpg.ifi.uzh.ch/docs/RAL22_Bauersfeld.pdf
- Human pilot horizon study: https://ieeexplore.ieee.org/document/9372805
- Betaflight: https://github.com/betaflight/betaflight
- Intel RealSense T265: https://www.intelrealsense.com/tracking-camera-t265/
- GT Sophy (similar milestone in video game): https://www.nature.com/articles/s41586-021-04357-7
- OpenAI Five: https://arxiv.org/abs/1912.06680
- AlphaPilot 2019 competition: https://www.nytimes.com/2019/03/26/technology/alphapilot-ai-drone-racing.html
- DARPA AlphaPilot: https://www.darpa.mil/news-events/2019-03-19
- Sim-to-real survey: https://thegradient.pub/sim2real/

希望这些人话讲解帮你 build intuition。如果想深挖某个细节（比如 GP sampling procedure、PPO implementation、Betaflight PID quirks），继续问。

---

# Champion-level Drone Racing using Deep RL — 深度解读

## 一、Why this paper matters — 里程碑意义

这篇来自UZH RPG (Robotics & Perception Group, Davide Scaramuzza) 的工作于2023年8月发表于Nature，是mobile robotics和machine intelligence领域一个真实里程碑：首次在physical sport中，autonomous mobile robot达到world-champion-level performance。

之前的deep RL里程碑 — AlphaGo (2016)、AlphaZero (2017)、StarCraft II (2019)、Dota 2 (OpenAI Five, 2018)、Gran Turismo (Sony GT Sophy, 2022) — 全部在simulation或video game环境。Gran Turismo虽然跑在物理引擎上，但游戏physics是deterministic且state是perfect observable的。而这篇paper面对的是real world，需要sim-to-real transfer、noisy perception、unmodeled aerodynamics，并且对手是flesh-and-blood世界冠军。

Drone racing本身特性决定这是agile robotics最难benchmark之一：quadrotor施加超过自身重量5倍的力，速度>100 km/h，加速度几倍于gravity，在confined space里maneuver。对state estimation要求极高，因为任何estimation drift都会被高速放大。

参考：https://rpg.ifi.uzh.ch/docs/science23/Kaufmann_Nature_2023.pdf
论文官网：https://doi.org/10.1038/s41586-023-06419-4

---

## 二、System Architecture — 两段式 hybrid 设计

Swift是hybrid system（learning-based + classical perception），这是关键设计选择：

### 2.1 Observation Policy（感知）

包含三个子模块：

**1) Visual-Inertial Odometry (VIO)**
- 硬件：Intel RealSense Tracking Camera T265，提供100Hz的VIO estimate
- VIO是相对metric estimate，会有drift（特别是高速下motion blur导致feature丢失）
- 参考实现：https://www.intelrealsense.com/tracking-camera-t265/

**2) Gate Detector (CNN)**
- Architecture：6-level U-Net
- 每层卷积filters数量：(8, 16, 16, 16, 16, 16)
- 卷积kernel size：(3, 3, 3, 5, 7, 7) — 越深层kernel越大，捕捉更大receptive field
- 最后额外一层：12 filters（对应4个gate corners × 3 coordinates，或者12个corner heatmaps）
- Activation：LeakyReLU(α=0.01)
- 输入：384×384 grayscale image（T265提供）
- 部署：NVIDIA Jetson TX2上port到TensorRT，FP16，40ms forward pass
- 任务：semantic segmentation of gate corners
- U-Net结构对sparse keypoint detection非常合适 — encoder-decoder skip connection保留spatial precision

**3) Kalman Filter (融合VIO + Gate Detection)**
- gate detector输出gate corners的image coordinates
- 用IPPE (Infinitesimal Plane-based Pose Estimation) 算法把corner pixels解算成drone相对gate的6DoF pose
- 用track layout先验，assign每个detected gate到track中最近的那个
- Kalman filter状态：x = [p_dᵀ, v_dᵀ]ᵀ ∈ ℝ^6
  - p_d: VIO的translational drift (3D)
  - v_d: drift velocity (3D)
- 注意：只correct translation，不动orientation（因为VIO orientation估计质量好，gate detection频率低）
- 关键：IPPE sampling-based estimation of measurement covariance R — 对每个gate，跑20次perturbed corner detection，用pose distribution近似R

### 2.2 Control Policy

- 网络：2-layer MLP，128 hidden units/layer，LeakyReLU(negative_slope=0.2)
- 输入o_t ∈ ℝ^31：
  - 15维robot state：position(3) + velocity(3) + attitude rotation matrix(9)
  - 12维next gate relative pose：4个corner × 3 coordinates
  - 4维previous action
- 输出a_t ∈ ℝ^4：mass-normalized collective thrust + 3-axis body rates
  - 选择thrust+body-rate而不是individual motor commands是有讲究的：这种control modality对sim-to-real transfer更鲁棒，因为low-level Betaflight PID去处理motor dynamics
- 推理：CPU上8ms一次（control frequency ~100Hz+）
- 训练算法：PPO (Proximal Policy Optimization)
- Value network（critic）训练时使用privileged information（exact pose/orientation/velocity），部署时不用

关键insight：robot attitude用rotation matrix而不是quaternion输入policy — quaternion有double covering ambiguity（q和-q代表同一rotation），会让network confused。

参考PPO：https://arxiv.org/abs/1707.06347

---

## 三、Quadrotor Dynamics — 仿真环境细节

公式(1)是状态空间形式：

$$\dot{\mathbf{x}} = \begin{bmatrix} \dot{\mathbf{p}}_{\mathcal{WB}} \\ \dot{\mathbf{q}}_{\mathcal{WB}} \\ \dot{\mathbf{v}}_{\mathcal{W}} \\ \dot{\boldsymbol{\omega}}_{\mathcal{B}} \\ \dot{\boldsymbol{\Omega}} \end{bmatrix}$$

变量含义：
- **p_{WB}** ∈ ℝ³：body frame B 相对 world frame W 的position
- **q_{WB}** ∈ S³：body相对world的attitude quaternion
- **v_W** ∈ ℝ³：inertial velocity (world frame)
- **ω_B** ∈ ℝ³：body rates (roll/pitch/yaw rate in body frame)
- **Ω** ∈ ℝ⁴：4个motor的实际转速
- m：quadrotor总质量
- **q_{WB} ⊙**：quaternion rotation operator
- **f_prop**：propeller lift force合力
- **f_aero**：aerodynamic force (drag, dynamic lift, induced drag)
- **g_W**：gravity vector in world frame
- **J**：quadrotor的inertia matrix (3×3)
- **τ_prop**：propeller thrust产生的torque
- **τ_mot**：motor speed变化产生的yaw torque
- **τ_aero**：aerodynamic torque (blade flapping等)
- **τ_iner**：inertial counter-torque = -ω_B × (J·ω_B) (gyroscopic effect)
- **k_mot**：motor time constant
- **Ω_ss**：steady-state motor speed（控制信号对应的目标转速）

公式(2)-(4) — Propeller model：

$$\mathbf{f}_i(\Omega_i) = [0, 0, c \cdot \Omega_i^2]^T$$

$$\boldsymbol{\tau}_i(\Omega_i) = [0, 0, c_d \cdot \Omega_i^2]^T$$

- **Ω_i**：第i个motor转速
- **c**：lift coefficient（推力系数）
- **c_d**：drag coefficient
- 注意是Ω²关系，这是propeller在低Mach数下的standard quadratic model

### 3.1 Aerodynamic Model — Grey-Box Polynomial

这块是关键，因为传统first-principles难以建模aerodynamics。他们用data-driven grey-box model：

- 6个输出：f_x, f_y, f_z (body frame forces), τ_x, τ_y, τ_z (body frame torques)
- 自变量：v_x, v_y, v_z (body frame velocity), v_xy (水平速度模), Ω² (mean squared motor speed)
- 形式：linear + quadratic combinations，例如

$$f_x \sim v_x + v_x|v_x| + \bar{\Omega}^2 + v_x \bar{\Omega}^2$$

物理直觉：
- v_x项：linear drag
- v_x|v_x|项：quadratic drag（保留符号）
- Ω²项：propeller对aerodynamic的影响（wake effect等）
- v_x·Ω²项：cross-coupling（propeller-induced airflow和forward velocity的交互）

系数从real flight data辨识，使用motion capture作为ground truth force/torque测量。这是用data fit track-specific aerodynamics — 类比于人类pilot训练一周熟悉track。

参考agile flight aerodynamics paper：https://rpg.ifi.uzh.ch/docs/RAL21_Kaufmann.pdf

### 3.2 Battery + ESC Model

公式(5)：电机功耗
$$P_{\text{mot}} = \frac{c_d \Omega^3}{\eta}$$

- **η**：电机+ESC efficiency
- **Ω³**关系：功耗 ∝ 转速³（这是为什么高速飞行功耗急剧上升）

公式(6)：steady-state motor speed mapping
$$\Omega_{i,ss} \sim 1 + U_{bat} + \sqrt{u_{cmd,i}} + u_{cmd,i} + U_{bat}\sqrt{u_{cmd,i}}$$

- **U_bat**：当前电池电压
- **u_cmd,i**：第i个motor的PWM command
- **U_bat · √u_cmd**项：电压随功率draw下降的cross effect

这个细节很重要：电池电压随飞行时间下降，导致同样PWM command产生不同motor speed。仿真要捕捉这个，否则长期flight policy会失效。

参考battery model：https://rpg.ifi.uzh.ch/docs/RAL22_Bauersfeld.pdf

---

## 四、Reward Design — Perception-Aware Racing

公式(7)：总reward

$$r_t = r_t^{\text{prog}} + r_t^{\text{perc}} + r_t^{\text{cmd}} - r_t^{\text{crash}}$$

### 4.1 Progress Reward

$$r_t^{\text{prog}} = \lambda_1 [d_{t-1}^{\text{Gate}} - d_t^{\text{Gate}}]$$

- **d_t^Gate**：t时刻drone center of mass到next gate center的距离
- 直觉：靠近gate就reward，dense shaped reward（vs sparse "pass gate = 1" reward）
- 这种dense progress reward来自Song et al.的工作 (https://arxiv.org/abs/2106.08705)

### 4.2 Perception-Aware Reward — 关键创新

$$r_t^{\text{perc}} = \lambda_2 \exp[\lambda_3 \cdot \delta_{\text{cam}}^4]$$

- **δ_cam**：camera optical axis与next gate center的夹角
- **λ_3 = -10.0**：负的，所以δ_cam越大reward越小
- **λ_2 = 0.02**：scaling
- **δ_cam^4**：四次方让reward对角度偏差更敏感（vs linear或quadratic）

这个reward是paper的核心insight之一：**让policy学会主动把摄像头对准next gate**，从而improve perception quality。这是active perception在RL中的体现。

人类pilot自然这样飞（FPV headset强迫他们看向前面），但autonomous agent如果只优化速度，可能学会"飞过gate但camera指向别处"，导致VIO drift爆炸。

### 4.3 Action Smoothness

$$r_t^{\text{cmd}} = \lambda_4 a_t^{\omega} + \lambda_5 \|a_t - a_{t-1}\|^2$$

- **a_t^ω**：commanded body rates（惩罚大的角速度）
- **||a_t - a_{t-1}||²**：连续action的L2差（jerk惩罚）
- **λ_4 = -2e-4, λ_5 = -1e-4**：很小负值

### 4.4 Crash Penalty

$$r_t^{\text{crash}} = \begin{cases} 5.0, & \text{if } p_z < 0 \text{ or collision with gate} \\ 0, & \text{otherwise} \end{cases}$$

- 触发crash则episode结束，penalty = 5.0
- **p_z < 0**：地面之下
- 这是terminal cost

### 4.5 训练超参

- γ (discount factor) = 0.99
- ε (PPO clipping) = 0.2
- 100 parallel agents
- episode length: 1500 steps
- 每次reset：random gate initialization + bounded perturbation around previously observed state
- Adam optimizer, lr = 3e-4
- 总训练量：1×10^8 environment interactions (50 min on i9-12900K + RTX 3090 + 32GB DDR5)
- Fine-tuning: 2×10^7 interactions

注意：**不进行domain randomization on dynamics** — 他们做fine-tuning with real data instead，这是和OpenAI Rubik's Cube等工作的关键区别。

---

## 五、Sim-to-Real Transfer — Residual Modeling

这是paper最elegant的部分。Discrepancies有两类：

### 5.1 Residual Observation Model — Gaussian Process

VIO在高速下drift很大（motion blur导致feature tracking失败 → linear odometry drift）。他们用GP建模drift。

输入：ground-truth robot state
输出：residual position (3), velocity (3), attitude (3) — 共9个独立1D GP

公式(10) — RBF kernel + noise：

$$\kappa(\mathbf{z}_i, \mathbf{z}_j) = \sigma_f^2 \exp\left(-\frac{1}{2}(\mathbf{z}_i - \mathbf{z}_j)^T L^{-2}(\mathbf{z}_i - \mathbf{z}_j)\right) + \sigma_n^2$$

变量：
- **z_i, z_j**：两个data points的特征向量（这里就是robot state）
- **L**：diagonal length scale matrix，控制每个input dimension的"相关性带宽"
- **σ_f**：data signal variance（输出amplitude）
- **σ_n**：noise variance（observation noise）

直觉：GP给的不是point estimate，是posterior distribution over residual functions。从posterior采样100个realizations，每个都是temporally consistent的drift轨迹。fine-tuning时rollout不同realization，policy学到robust to perception noise distribution。

为什么用GP而不用NN？GP好处：
1. Uncertainty quantification天然支持
2. 小数据集（3次rollout ~50秒）下overfitting风险低
3. 可以sample新realization，模拟stochastic perception failure

### 5.2 Residual Dynamics Model — k-NN Regression

公式(11)：
$$\mathbf{a}_{\text{res}} = \text{KNN}(\mathbf{s}, c)$$

- **s**：platform state
- **c**：commanded mass-normalized collective thrust
- **a_res**：residual acceleration（实际 - 仿真预测）
- k = 5, dataset 800-1000 samples

为什么dynamics用k-NN，perception用GP？作者说empirically发现：
- **Perception residuals是stochastic的**（同一state两次VIO输出不同）→ 需要distribution model (GP)
- **Dynamics residuals是largely deterministic的**（同一state+command产生的力基本一样）→ 用non-parametric regression足够

这个insight很重要：选model要match data nature。

参考Empirical Actuator Models (ANYmal)：https://www.science.org/doi/10.1126/scirobotics.aau5872

---

## 六、Gate Detection + VIO Drift Correction 细节

公式(12)-(14) Kalman Filter for drift estimation：

状态：x = [p_dᵀ, v_dᵀ]ᵀ ∈ ℝ^6

公式(12) — Prediction：
$$\mathbf{x}_{k+1} = F \mathbf{x}_k, \quad P_{k+1} = F P_k F^T + Q$$

公式(13) — F (transition) 和 Q (process noise)：
$$F = \begin{bmatrix} I^{3×3} & dt \cdot I^{3×3} \\ 0^{3×3} & I^{3×3} \end{bmatrix}, \quad Q = \begin{bmatrix} \sigma_{pos} I^{3×3} & 0 \\ 0 & \sigma_{vel} I^{3×3} \end{bmatrix}$$

- σ_pos = 0.05, σ_vel = 0.1
- 这是constant velocity model for drift

公式(14) — Update：
$$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R)^{-1}$$
$$\mathbf{x}_k^+ = \mathbf{x}_k^- + K_k(\mathbf{z}_k - H(\mathbf{x}_k^-))$$
$$P_k^+ = (I - K_k H_k) P_k^-$$

- **K_k**：Kalman gain
- **z_k**：measurement (gate detection pose)
- **H_k**：measurement matrix (这里 H = [I³ 0³] 即只观察position drift)
- **R**：measurement covariance，通过IPPE sampling估计（每gate跑20次perturbed corner detection）

直觉：VIO是高频(100Hz)但drift；gate detection是低频(25Hz)但准确。Kalman filter用低频accurate measurement纠正高频drifting estimate。这是sensor fusion经典操作。

如果同帧检测到多个gate，所有relative pose estimates stacked一起进同一update step（multi-measurement update）。

---

## 七、Results — Head-to-Head Races

### 7.1 Race Results

| Matchup | Races | Best Time | Wins | Losses | Win Ratio |
|---|---|---|---|---|---|
| A. Vanover vs Swift | 9 | 17.956s | 4 | 5 | 0.44 |
| T. Bitmatta vs Swift | 7 | 18.746s | 3 | 4 | 0.43 |
| M. Schaepper vs Swift | 9 | 21.160s | 3 | 6 | 0.33 |
| **Total** | **25** | **17.465s** | **15** | **10** | **0.60** |

Swift fastest lap: **17.465s** (半秒领先人类最快)

10次失败原因：
- 40% 碰撞对手 (collision with opponent)
- 40% 碰撞gate
- 20% 速度慢

### 7.2 Segment-by-Segment Analysis (Extended Data Table 1d)

Extended Data Table 1d给出最关键数据 — track分4段：

**Segment 1 (start):**
- Swift: speed=6.52m/s, time=1.27s
- Vanover: speed=5.62m/s, time=1.42s
- Bitmatta: speed=5.93m/s, time=1.36s
- **Swift起飞快120ms，加速度更高**

**Segment 2:**
- Swift: 13.72m/s, 1.82s, 25.01m
- Vanover: 13.83m/s, 1.83s, 25.31m
- 几乎持平

**Segment 3 (Split-S):**
- Swift: 13.70m/s, 1.96s, 26.89m
- Vanover: 13.86m/s, 2.09s, 28.91m
- **Swift path更短(26.89 vs 28.91m)，时间更快(1.96 vs 2.09s)**
- **但Vanover speed更高(13.86 vs 13.70)** — 说明Swift走tighter line

**Segment 4:**
- Swift: 13.43m/s, 1.61s, 21.62m
- 几乎持平

**Full Race:**
- Swift: 13.11m/s avg, 866W avg, 29.16N avg thrust, 17.46s, 228.85m
- Vanover: 12.96m/s, 843W, 28.65N, 17.96s, 232.79m
- **Swift走了更短的path (228.85 vs 232.79m)**

### 7.3 Lap Time Distribution

Figure 3a显示：
- Swift lap time分布：低mean，低variance — 一致push最快
- Human pilots lap time：高mean，高variance — 每lap决定要不要push

关键insight：**Human pilots是strategic的**。如果他们有lead，会slow down降crash风险。Swift没意识到对手位置，永远push最快 — 这意味着Swift领先时over-risk，落后时under-risk。

### 7.4 Perception-Aware Behavior 比较

Figure 4c-d分析maneuver after gate 2和Split-S：

**关键发现：**
1. Swift在tight turns找tighter line (less overshoot)
2. **Human pilots把camera对准next gate更早** — perception-driven
3. Swift可以执行maneuver时camera不指向gate，依赖inertial + visual odometry against environment features

**为什么Swift能赢？** RL的value function让Swift在longer timescale上optimize — 不是greedy on next gate，而是optimize整个lap的time。Human pilots planning horizon大约1 gate ahead (Pfeiffer & Scaramuzza 2021, https://ieeexplore.ieee.org/document/9372805)。

这是model-free RL在sequential decision-making上的优势：value function propagate long-term reward back to当前action。

---

## 八、Simulation Ablation (Extended Data Table 1c)

四种部署setting：
1. Idealized dynamics + ground-truth obs
2. Idealized dynamics + noisy obs
3. Realistic dynamics + ground-truth obs
4. Realistic dynamics + noisy obs

对比baselines：
- Zero-shot transfer (from Loquercio 2021, learning-based)
- Domain randomization
- Time-optimal trajectory + MPC (from Foehn 2021)

**关键结果：**

Setting 1 (idealized):
- Zero-shot: 4.88s, 100% completion
- Domain Rand: 5.06s, 100%
- Time-Opt+MPC: **4.60s**, 100% — **MPC在理想条件下最快！**
- Ours: 4.88s, 100%

Setting 2-4 (any domain shift):
- Zero-shot, Domain Rand, Time-Opt+MPC — 全部collapse (0-19% completion)
- Ours: 100% completion, 5.20-5.42s

**Insight：** 传统MPC在perfect conditions下能beat RL，但robustness崩盘。RL+residual modeling在domain shift下保持性能。这是这篇paper的核心claim：**sim-to-real的关键是data-driven residual models，不是domain randomization**。

Extended Data Table 1b — 即使baselines也access GP noise model：
- Zero-shot + GP noise: 42% completion
- Domain Rand + GP noise: 19%
- Time-Opt + GP noise: 19%
- **Ours + GP noise: 100%**

即使在fair comparison（都给noise model）下，Ours还是唯一100% completion。原因：我们policy在fine-tuning时不仅用noise model的mean，也sample from uncertainty distribution，让policy真正robust to perception noise distribution。

---

## 九、Hardware Configuration

- 总重量：870g
- 最大static thrust：35N
- **Thrust-to-weight ratio：4.1** (人类极限竞速drone通常4-5)
- Frame: Armattan Chameleon 6"
- Motors: T-Motor Velox 2306
- Propellers: 5" 3-bladed
- Compute: NVIDIA Jetson TX2 + Connect Tech Quasar carrier
  - 6-core CPU @ 2GHz
  - GPU: 256 CUDA cores @ 1.3GHz
  - Gate detection: GPU, FP16, 40ms
  - Policy inference: CPU, 8ms
- Sensing: Intel RealSense T265 (VIO @ 100Hz, 30Hz grayscale camera)
- Low-level: STM32 @ 216MHz, Betaflight firmware
- Human pilots用相同硬件，但Jetson+T265换成ballast weight (fair comparison)

**Latency对比：**
- Swift sensorimotor latency: **40ms**
- Human expert latency: **220ms** (Pfeiffer & Scaramuzza 2021)

但Swift的**camera 30Hz vs human 120Hz** — refresh rate是Swift劣势。

Agilicious framework reference: https://agilicious.dev/
Betaflight reference: https://github.com/betaflight/betaflight

---

## 十、Build Intuition — 几个关键insight

### Insight 1: Hybrid architecture > End-to-End

Swift不是端到端pixels→commands。它hybrid：
- Classical VIO (well-understood, robust, decades of robotics)
- Learning-based gate detection (handles texture/illumination variation)
- Learning-based control policy (handles nonlinear dynamics & perception coupling)

为什么不全end-to-end？因为：
1. **Perception module可以单独train with大量labeled gates**，比reward shaping容易
2. **VIO是commodity** — 用现成T265，不重新发明轮子
3. **Control policy输入是low-dim state (31D)**，比raw images (384×384=147k dims)好train得多
4. **Compositionality** — 各模块可以独立improve

这和Tesla FSD、Waymo的架构思路类似：hybrid systems beat pure end-to-end in safety-critical real-world。

### Insight 2: Residual Models > Domain Randomization

OpenAI Rubik's Cube (2019)用了massive domain randomization (ADR — Automatic Domain Randomization)。Swift反其道而行：**收集少量real data，build residual model，fine-tune policy**。

为什么？
- Domain randomization让policy robust to a *range* of conditions，但可能suboptimal在每个specific condition
- Residual model用少量data识别实际domain shift，policy fine-tune到这个specific shift
- Drone racing是narrow task (一个track, 一个drone)，narrow adaptation > broad robustness

这是 **narrow AI done right** — 不追求generality，追求在specific task上达到expert+ performance。

### Insight 3: Perception-Aware Reward = Active Perception

RL agent默认是passive perceiver — 只优化task reward，不管sensor质量。但drone racing中perception quality直接影响control performance。

加入perception-aware reward让policy学会：
- 主动orient camera指向next gate
- 在maneuver中preserve gate visibility
- 这间接improve VIO accuracy → improve control performance

这是**Active Perception**或**Perception-Aware Planning**在RL中的体现。参考Scaramuzza组早期工作：https://rpg.ifi.uzh.ch/docs/IROS19_Kaufmann.pdf

### Insight 4: Value Function > MPC for Long-Horizon

Extended Data Table 1c显示：
- Time-Optimal MPC: 4.60s (ideal conditions)
- RL: 4.88s (ideal conditions)
- 但MPC在real conditions崩盘，RL不崩

为什么RL能在Split-S上beat人类？因为RL policy的value function隐含encode了long-horizon planning。MPC受horizon限制 (computation cost exponential in horizon)，人类受限于cognitive horizon (~1 gate ahead)。

RL的value function是approximate dynamic programming的结果，把future reward "压缩"进当前state value — 等价于无限horizon optimal control的approximation。

### Insight 5: Data Efficiency — 50秒飞行数据

Swift只用了**3次real-world rollout (~50秒飞行)** 来识别residual models。这个数据效率惊人。

为什么50秒够？
1. Residual model是low-dim (state → small correction)
2. GP/k-NN是sample efficient non-parametric methods
3. 一圈lap约17秒，3次rollout = 约9 laps，足够覆盖track所有segments

对比AlphaGo用millions of self-play games — physical robot数据贵，必须sample efficient。

### Insight 6: Sim-to-Real是Engineering Problem

Swift的成功不是单一算法突破，是**systems engineering**：
- High-fidelity simulator (aerodynamics, battery, ESC, Betaflight PID)
- Accurate residual models (GP for stochastic, k-NN for deterministic)
- Stable perception pipeline (VIO + gate detection + Kalman)
- Robust control modality (thrust+body rate vs raw motor commands)
- Hardware-software co-design (same drone for human and AI)

这呼应了Karpathy你自己说过的"Software 2.0"思想：现代AI系统是data + compute + architecture + engineering的组合，不是单一算法magic。

---

## 十一、Limitations & Future Work

Paper诚实地承认：

1. **No crash recovery** — 人类pilot可以crash后继续飞（如果hardware OK），Swift没训练recovery
2. **Appearance sensitivity** — gate detector依赖training时appearance，光照变化可能失效
3. **No opponent awareness** — Swift unaware of opponent，导致strategic suboptimality (前面说的over/under-risk)
4. **30Hz camera limit** — vs human 120Hz
5. **Single track specialization** — 一周训练，fine-tune到一个track

未来方向：
- Train gate detector + residual model in diverse conditions (illumination, weather)
- Multi-agent RL for opponent-aware racing
- Crash recovery policy
- Cross-track generalization (meta-learning?)
- 真正参加DRL联赛：https://www.drlracing.com/

---

## 十二、Related Work联想

### 12.1 Gran Turismo Sophy (Sony, 2022)

类似milestone — RL beat人类GT Sport champion。但GT是simulated physics，state是perfect。Swift面对真实physics。

GT Sophy paper: https://www.nature.com/articles/s41586-021-04357-7

### 12.2 OpenAI Five (Dota 2, 2018)

Dota 2是部分observable multi-agent，但仍是video game。Swift是physical robot。

OpenAI Five: https://arxiv.org/abs/1912.06680

### 12.3 ANYmal (ETH Zurich, 2019)

ANYmal用RL+actuator network做sim-to-real for legged locomotion。Swift借鉴了actuator network思想，扩展到perception residuals。

ANYmal paper: https://www.science.org/doi/10.1126/scirobotics.aau5872

### 12.4 Loquercio et al. "Deep Drone Racing" (2019, 2021)

UZH组前一作工作，用domain randomization做sim-to-real drone racing。Swift是这些工作的延续和超越。

Deep Drone Racing: https://ieeexplore.ieee.org/document/8793913
Learning high-speed flight: https://www.science.org/doi/10.1126/scirobotics.abg5810

### 12.5 Foehn et al. "Time-optimal Planning for Quadrotor Waypoint Flight" (2021)

传统MPC方法做drone racing，理想条件下最快。Swift在real conditions下beat它。

Foehn paper: https://www.science.org/doi/10.1126/scirobotics.abh1221

### 12.6 AlphaPilot Competition (2019)

DARPA-funded autonomous drone racing competition。最好队伍比professional human pilot慢2倍。Swift在4年内从2x slower到faster than human。

AlphaPilot: https://www.darpa.mil/news-events/2019-03-19

### 12.7 Sim-to-Real综述

Andrey Kurenkov的sim-to-real综述: https://thegradient.pub/sim2real/

---

## 十三、Technical Nuggets — 易错过细节

### 13.1 Quaternion vs Rotation Matrix

仿真内部用quaternion，但policy input用rotation matrix (9D)。原因：quaternion的double covering (q和-q同一rotation)会让network confused，rotation matrix没有这问题。

### 13.2 Privileged Information for Critic

Value network训练时access exact position/orientation/velocity (privileged info)，policy network只access observation。这是asymmetric actor-critic (Pinto et al. 2018, https://arxiv.org/abs/1810.06494)。

直觉：critic需要accurate value estimate来guide policy learning，但部署时不需要deploy critic，所以可以"作弊"。

### 13.3 Mass-Normalized Thrust

Policy输出mass-normalized collective thrust (单位 m/s²)，不是raw thrust (N)。这normalize掉质量变化，让policy transferable到不同drone。

### 13.4 Betaflight Peculiarities

仿真准确建模了Betaflight PID的quirks：
- D-term reference恒为0 (pure damping)
- I-term在throttle cut时reset
- Motor saturation时body rate control优先 (proportional downscaling所有motor signals)

这些细节让sim预测individual motor commands误差 < 1%。

### 13.5 GP Sampling for Fine-Tuning

GP不只是predict mean residual，是从posterior sample 100个realizations。每个realization是一条temporally consistent的drift trajectory。Fine-tuning时每次rollout用不同realization，policy学到robust to perception noise distribution，不只是mean。

### 13.6 IPPE for Gate Pose

IPPE (Infinitesimal Plane-based Pose Estimation)是planar object pose estimation的经典算法。Gate是平面，4个corner定义一个plane。IPPE给出closed-form解的pose。

IPPE paper: https://link.springer.com/article/10.1007/s11263-014-0725-4

### 13.7 Random Gate Initialization

训练时reset到random gate with bounded perturbation around previously observed state at that gate。这比从同一起点训练更好，因为policy学会从track任何位置恢复 (generalization + curriculum effect)。

### 13.8 Motor Saturation Handling

仿真里精确建模了motor saturation下的control allocation — body rate control优先。如果simulation不model这个，real flight遇到saturation时policy会behave unexpectedly。

---

## 十四、Open Questions / 思考点

1. **Why not end-to-end pixels→controls?** Swift的hybrid设计可能出于engineering efficiency，但end-to-end理论上能学到更tight perception-control coupling。是否可能end-to-end在更大compute + data下beat Swift？

2. **GP vs Bayesian NN for residual modeling?** GP sample efficient但scalability有限。Bayesian NN或deep ensembles可能在大data下更好。

3. **Multi-track meta-learning?** Swift fine-tune到single track。如果训练meta-policy能快速adapt到新track，更接近human pilot能力。

4. **Real RL in real world?** Swift还是sim-trained + real fine-tune。完全real-world RL（with safety constraints）是否可能？参考offline RL + safety literature。

5. **Symmetry exploitation?** Quadrotor有rotational symmetry (yaw invariant)，policy是否exploit这结构？paper未讨论。

6. **Why PPO not SAC or TD3?** PPO是on-policy，sample efficiency不如off-policy。Swift用100 parallel agents抵消这劣势。但SAC可能更sample efficient for fine-tuning phase。

7. **Long-horizon credit assignment in Split-S?** Split-S需要coordinated roll+pitch+descent。reward如何backpropagate到早期action？PPO的GAE (Generalized Advantage Estimation)如何handle？

8. **Gate detector failure modes?** 30Hz camera + motion blur在gate附近（高速过gate时）detection quality如何？paper未详细讨论failure rate。

---

## 十五、Conclusion

Swift代表mobile robotics和machine intelligence的intersection的一个成熟范式：

**Hybrid system + RL control + sim-to-real via residual models + perception-aware reward + privileged critic training**

在narrow but physically demanding task上达到world champion水平。这是AlphaGo moment for physical robotics — 证明RL不止能玩video game，能在real physical competitive sport beat human。

下一步：扩展到更多domain (autonomous driving, personal robotics, aerial delivery)，handle broader conditions，多agent竞争。

参考链接汇总：
- Paper: https://doi.org/10.1038/s41586-023-06419-4
- RPG group: https://rpg.ifi.uzh.ch/
- Code (pseudocode): https://doi.org/10.5281/zenodo.7955278
- Discussion by Scaramuzza: https://www.youtube.com/watch?v=j0P8OuZS01Y (建议搜索"Swift drone racing UZH"找相关talks)
- Agilicious: https://agilicious.dev/
- Prior work Loquercio 2021: https://www.science.org/doi/10.1126/scirobotics.abg5810
- Time-optimal MPC: https://www.science.org/doi/10.1126/scirobotics.abh1221
- PPO: https://arxiv.org/abs/1707.06347
- U-Net: https://arxiv.org/abs/1505.04597
- GP for ML: http://www.gaussianprocess.org/gpml/
- IPPE: https://link.springer.com/article/10.1007/s11263-014-0725-4
- Asymmetric actor-critic: https://arxiv.org/abs/1810.06494
- ANYmal: https://www.science.org/doi/10.1126/scirobotics.aau5872
- GT Sophy: https://www.nature.com/articles/s41586-021-04357-7
- OpenAI Five: https://arxiv.org/abs/1912.06680
- Human pilot study: https://ieeexplore.ieee.org/document/9372805
- Battery model: https://rpg.ifi.uzh.ch/docs/RAL22_Bauersfeld.pdf
- Aerodynamics model: https://rpg.ifi.uzh.ch/docs/RAL21_Kaufmann.pdf

希望这些technical details和intuition能帮你build mental model of this work。如果你对某个细节感兴趣（比如GP sampling procedure、PPO implementation specifics、Betaflight PID modeling），可以继续深挖。
