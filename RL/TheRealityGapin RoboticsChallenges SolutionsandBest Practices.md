---
source_pdf: TheRealityGapin RoboticsChallenges SolutionsandBest Practices.pdf
paper_sha256: 66b65a329f27178a815b1a2dac43acf7126dd0f14b001bd255b449b536648101
processed_at: '2026-08-12T15:18:34-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用工程师视角讲讲这篇 paper

## 这篇 paper 到底在讲啥

想象你在 sim 里把一个 quadruped 训得能跑能跳，搬到真实机器上一通电，它直接趴窝。你的第一反应通常是"simulator 不够逼真"。这篇 paper 说：别急，先弄清楚是 *哪一段* 不逼真。

它把 sim 和 real 之间的差距拆成四块：**dynamics、perception、actuation、system design**。每一块下又有一堆子问题，加起来大概 14 个 source。然后所有"怎么解决"的方法被归到两大类——*reducing the gap*（让 sim 更像 real）和 *overcoming the gap*（让 policy 对 gap 鲁棒）。

就这么简单。剩下的几百页都是在给这个骨架填细节、给公式、给 reference。

---

## 四个 gap 用人话讲

### 1. Dynamics gap：物理没对上

sim 把 robot 当成"完全刚体 + 理想关节 + 点接触 + Gaussian noise"。real 里：

- link 会弯，joint 有 backlash，motor 有内阻
- ground friction 不是常数，而是速度、压力、湿度的函数
- 接触面积不是点，是一小块变形区
- 噪声不是高斯，是状态相关、时间相关

这里有个特别容易被忽略的：**battery**。motor torque 直接正比于电压，而真实电池在大电流下电压会 sag：

$$V_{\text{batt}}(I) = V_{\text{oc}} - I \cdot R_{\text{internal}}$$

$V_{\text{oc}}$ 是开路电压，$I$ 是瞬时电流，$R_{\text{internal}}$ 是内阻。结果就是 quadruped 想猛跳的瞬间，电池供不上电，torque 跟不上，policy 就崩了。sim 里根本没建模这个。

### 2. Perception gap：传感器没对上

sim 里用 OpenGL 加 Z-buffer 渲染，pinhole camera model，depth 是干净的几何投影。real 里：

- 镜头有 lens flare、chromatic aberration、rolling shutter
- depth sensor 在边缘有 flying pixels、depth shadow、量化噪声
- LiDAR 有 beam divergence、material-dependent reflectivity
- IMU 有 drift，GPS 有 multipath

最简单的缓解：**别用 RGB**，用 depth 或 point cloud。你省掉一整类 gap。Dex-Net https://berkeleyautomation.github.io/dex-net/ 就是这么干的。

### 3. Actuation gap：最被忽视的一块

sim 里你发一个 torque command，下一帧就到位。real 里这个 command 要经过：

- 厂商的 low-level controller（黑盒，有 anti-windup、resonance filter）
- power electronics（PWM 量化、dead-time、current cap）
- motor 本身是二阶系统（electrical time constant $\tau_e \sim 1-5$ ms + mechanical $\tau_m \sim 10-50$ ms）

**这还没算 gearbox 的 backlash 和 friction。**

结果：sim 里 policy 发了个"瞬间反向 5 Nm"的命令，real driver 要么饱和要么延迟 20 ms，闭环就发散。

工程师层面我见过最实在的做法：**sim 里直接复制 real 的控制栈**，包括延迟、量化、饱和。Isaac Lab 的 `ActuatorNet` 就是把真实 motor 的输入输出对学一个网络，嵌到 sim 里。

### 4. System design gap：软件栈没对齐

sim 里通信完美，reset 用魔法把物体瞬移到指定位置。real 里：

- ROS 之间有 packet loss、jitter
- 有 virtual wall、e-stop 这类 safety 机制改变行为
- reward 在 sim 里用 ground-truth collision，real 里拿不到

这些"小细节"经常被忽略，但能让 sim 95% 成功率的策略在 real 上跌到 30%。

---

## 两条解决路径

### A. Reduce the gap：把 sim 拉近 real

最朴素也最稳。三种做法：

**System Identification**：测真实参数，塞回 sim。mass、friction、latency、control frequency 都能标定。Chebotar 2019 的"online DR adjustment" https://arxiv.org/abs/1810.10356 就是边训边用 real data 调 randomization 范围。

**Residual Model**：sim 物理对了一半，剩下一半用神经网络补：

$$s_{t+1}^{\text{real}} \approx T_{\text{sim}}(s_t, a_t) + f_\theta(s_t, a_t)$$

$f_\theta$ 是个小 MLP，用 real trajectory 数据训。Golemo 2018 https://arxiv.org/abs/1810.02125 是早期工作。

**Real-to-Sim**：直接用 real 图像和几何重建 sim 场景。NeRF / 3D Gaussian Splatting 现在能做得相当好 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 。但 dynamics 还得另外标定。

### B. Overcome the gap：让 policy 鲁棒

**Domain Randomization (DR)**：sim 里把 friction 从 0.3 随机到 1.2，mass 从 0.8 kg 到 1.2 kg，光照、纹理全随机。policy 学一个"对任何参数都 work"的策略。

代价：参数范围太大，policy 学到一个 mediocrity——什么环境都能凑合，什么都不强。Automatic DR https://arxiv.org/abs/1910.07113 的解法是边训边扩范围。

**RMA (Rapid Motor Adaptation)** https://roboticsproceedings.github.io/rss17/p022.pdf ：sim 训练时用 privileged info（真实摩擦、mass）训一个 latent encoder，部署时从 proprioception 历史反推这个 latent。等于让 policy 在线做 sysID。

公式上分两阶段：

$$\text{Stage 1: } z_t = e_\psi(e_t), \quad a_t = \pi_\omega(o_t, z_t)$$

$e_t$ 是 environment 参数，sim 里能拿到。$z_t$ 是 latent encoding。

$$\text{Stage 2: } \hat{z}_t = \hat{e}_\chi(o_{t-k:t}, a_{t-k:t})$$

$\hat{e}_\chi$ 用 behavior cloning 训，让它从可观测的历史推 $z_t$。部署只用 $\hat{e}_\chi$ 和 $\pi_\omega$。

**Teacher-Student**：teacher 在 sim 用 privileged obs 训，student 用 distillation 学只用 real 可观测 obs。Radosavovic 2024 https://www.science.org/doi/10.1126/scirobotics.adi9579 把这招推到 humanoid 上，loss 用 RL + distillation 混合：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{RL}} + (1-\alpha) \mathcal{L}_{\text{distill}}$$

$\alpha$ 用 schedule，前期 RL 主导，后期 distill 主导。

---

## 几个我觉得最值得记住的 insight

### 1. Reality gap 大 ≠ Performance gap 大

paper 用公式明确区分：

$$G_{\text{perf}}(\mathcal{M}_s, \mathcal{M}_r, \pi) = |J_{\mathcal{M}_s}(\pi) - J_{\mathcal{M}_r}(\pi)|$$

sim 和 real 的 POMDP 是 $\mathcal{M}_s, \mathcal{M}_r$，$J$ 是 discounted return。**这个值才是我们关心的**，simulator fidelity 只是手段。

这意味着你可以接受一个粗糙 simulator，只要 policy 对它的"粗糙点"不敏感。Lambert 2018 https://arxiv.org/abs/2002.07609 证明：model-based RL 不需要全局准确，只要在 high-return 区域准确就够了。

### 2. SRCC 是真有价值的 metric

Sim-to-Real Correlation Coefficient：

$$\text{SRCC} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}$$

$x_i, y_i$ 是第 $i$ 个 policy 在 sim / real 上的成功率。

**SRCC 高意味着 sim 性能是 real 性能的好 predictor**。这样你才能放心在 sim 里做 architecture search、hyperparameter tuning。如果 SRCC 低，每次调参都得跑 real robot，simulator 的价值就只剩"提供 prior"。

Kadian 2020 https://ieeexplore.ieee.org/document/9017453 提出这个指标，我觉得所有 sim-to-real paper 都应该报。

### 3. Offline Replay Error 是最便宜的 diagnostic

给定一条 real trajectory $\{s_t^{\text{real}}, a_t\}$，在 sim 里 open-loop 重放 $a_t$：

$$\mathcal{E}_{\text{replay}} = \frac{1}{T}\sum_{t=1}^T \|s_t^{\text{sim}} - s_t^{\text{real}}\|^2$$

不用闭环部署，只用 log 数据。**open-loop 误差大不一定 closed-loop 失败**（policy 能纠错），**但 open-loop 误差小基本保证 closed-loop OK**。是个保守但便宜的 sanity check。

### 4. Contact-rich 是真的难

free-space 飞行或 walking，dynamics smooth，误差可以 Lipschitz bound：

$$\|T_{\text{real}}(s,a) - T_{\text{sim}}(s,a)\| \leq L \cdot \|s - s^*\|$$

policy 用 smoothness regularization 就能 robust。

contact-rich manipulation，$T$ 在 contact 切换点不可微，$L$ 局部爆炸，DR 范围也跟着爆炸。这就是为什么 dexterous manipulation 的 sim-to-real 比 locomotion 晚好几年才做出来 https://arxiv.org/abs/1910.07113 。

---

## 最让我兴奋的方向

paper 6.3 提到 **world model** 替代或补足 physics simulator。Genie https://arxiv.org/abs/2402.15391 、Cosmos https://arxiv.org/abs/2501.03575 、V-JEPA 2 https://ai.meta.com/blog/v-jepa-2-world-model-physical-understanding/ 都在朝这走。

直觉：world model 用 real 数据训，reality gap 天然小。physics simulator 是 white-box，world model 是 black-box。**长期这俩会融合**——physics 提供 prior，world model 补 residual。

另一个是 **simulation-based inference**（paper 6.4）。把 DR 的 randomization distribution 从均匀分布换成 posterior：

$$p(\theta | s^r) \propto p(s^r | \theta) p(\theta)$$

$\theta$ 是 sim 参数，$s^r$ 是 real state。这个 likelihood 不可解析，用 BayesFlow https://arxiv.org/abs/2101.10762 这种 neural posterior estimation 解。比"无脑均匀 DR"聪明太多。

---

## 给从业者的实操建议

paper 给的 6 步 recipe，我加一条自己的：

1. 设计 sim 时把所有相关变量都放进去
2. 尽量 reduce gap（sysID + software stack 对齐）
3. 设计训练方法 overcome 剩余 gap
4. 大规模并行训练
5. Real 评估
6. 根据 real 结果调 sim 参数，迭代
7. **从第一天就把 real 评估集成进 dev loop**

第 7 条是我加的。等 sim 跑通才上 real，gap 会累积到无法诊断。Iterative Residual Tuning https://arxiv.org/abs/2003.03075 是这个思路的代表。

---

## 一句话总结

这篇 paper 的价值在于：**它给你一张地图**。看到任何 sim-to-real 方法，你能立刻判断它在 reduce 哪个 sub-gap，或者 overcome 哪个 sub-gap，用什么 metric 验证，代价多少。

记住一个核心区分：**reality gap 是 simulator 的属性，performance gap 是 policy + simulator 的属性**。我们最终关心的是后者。前者只是手段。

想 build intuition 的话，挑一个 sub-gap（比如 contact dynamics），把 paper 引用的 3-4 篇代表作读一遍，你会看到同一种物理直觉在不同方法里的不同数学表达。

参考链接合集：

- RPG UZH https://rpg.ifi.uzh.ch/research_groups.html
- Isaac Sim https://docs.omniverse.nvidia.com/isaacsim/latest/
- MuJoCo https://mujoco.readthedocs.io/
- Isaac Lab https://isaac-sim.github.io/IsaacLab/main/
- RMA https://roboticsproceedings.github.io/rss17/p022.pdf
- Drone Racing Nature paper https://www.nature.com/articles/s41586-023-06461-5
- Real-World Humanoid Locomotion https://www.science.org/doi/10.1126/scirobotics.adi9579
- Open X-Embodiment https://robotics-transformer-x.github.io/
- DROID https://droid-dataset.github.io/
- GR00T N1 https://arxiv.org/abs/2503.14734
- BayesFlow https://arxiv.org/abs/2101.10762
- Genie https://arxiv.org/abs/2402.15391
- Cosmos https://arxiv.org/abs/2501.03575
- V-JEPA 2 https://ai.meta.com/blog/v-jepa-2-world-model-physical-understanding/

---

# The Reality Gap in Robotics — 综述深度讲解

## 0. 一句话定位

这篇 survey 把 sim-to-real 这个长期被"玄学化"的问题, 用 POMDP 的语言形式化, 把 reality gap 拆解为 **dynamics gap**、**perception gap**、**actuation gap**、**system-design gap** 四个子类, 并把所有解决方案归到两个正交范畴: *reducing the gap* (提高 simulator fidelity) 与 *overcoming the gap* (让 policy 对 gap 鲁棒)。论文链接与作者主页: Elie Aljalbout (UZH RPG) https://rpg.ifi.uzh.ch/people.html ; Davide Scaramuzza https://rpg.ifi.uzh.ch/research_groups.html ; NVIDIA robotics http://rmp.nvidia.com 。

---

## 1. 为什么这份 paper 值得反复读

我认为它最值得的一点, 是它把 "reality gap" 这个口语化的词, 用两条 divergence 公式 (Eq. 2) 和一条 absolute difference (Eq. 4) 钉死, 这样后续所有的算法都能映射回这两条 gap 上, 避免"我用 domain randomization 就行了"这种笼统表述。

更关键的是, paper 在 Section 2.3 区分了 **reality gap** 与 **performance gap**:

$$G_{\text{perf}}(\mathcal{M}_s, \mathcal{M}_r, \pi) = \left| J_{\mathcal{M}_s}(\pi) - J_{\mathcal{M}_r}(\pi) \right|$$

其中 $\mathcal{M}_s, \mathcal{M}_r$ 分别是 simulated 和 real POMDP, $\pi$ 是策略, $J_{\mathcal{M}}(\pi) = \mathbb{E}_{\tau \sim p(\tau|\pi,\mathcal{M})}[\sum_t \gamma^t R(s_t, \pi(b_t))]$ 是该环境下的 discounted return。

这个区分非常关键: **reality gap 可以很大, performance gap 可以很小** — 只要 policy 对 gap 分布鲁棒就行。这直接颠覆了 "我要把 simulator 做到 photorealistic" 的执念。这一点对思考 foundation model 在 robotics 上的应用至关重要, 因为神经 world model (Genie, Cosmos, GAIA-1) 即便不 photorealistic, 也可能满足 performance gap 小。

---

## 2. 形式化框架: POMDP 与两类 gap

### 2.1 POMDP 定义

POMDP tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{Z}, \mathcal{O}, \gamma)$:

- $\mathcal{S} \subseteq \mathbb{R}^n$: state space (隐变量, 包含 robot + environment 的全部配置)
- $\mathcal{A} \subseteq \mathbb{R}^m$: action space (control command)
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: transition dynamics, $s_{t+1} \sim \mathcal{T}(\cdot | s_t, a_t)$
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: reward
- $\mathcal{Z} \subseteq \mathbb{R}^h$: observation space
- $\mathcal{O}: \mathcal{S} \to \mathcal{Z}$: observation model, $z_t \sim \mathcal{O}(\cdot | s_t)$
- $\gamma \in [0, 1)$: discount factor

belief $b_t \in \mathcal{B}(\mathcal{S})$ 是 state 上的分布, policy $\pi: \mathcal{B}(\mathcal{S}) \to \mathcal{A}$。

### 2.2 Reality gap = dynamics gap + perception gap

$$G_{\text{dyn}} = \mathbb{E}_{(s,a) \sim \mathcal{M}_r}\left[ D\left( T_{\text{sim}}(\cdot|s,a) \,\|\, T_{\text{real}}(\cdot|s,a) \right) \right]$$

$$G_{\text{perc}} = \mathbb{E}_{s \sim \mathcal{S}_r}\left[ D\left( O_{\text{sim}}(\cdot|s) \,\|\, O_{\text{real}}(\cdot|s) \right) \right]$$

这里 $D$ 是任意 divergence (KL, Wasserstein, MMD 等都可以)。注意: 这两个期望都是在 **real environment 的 state-action 分布**上取的。这是非常 subtle 但重要的点: gap 是用 real 上的 visitation 衡量, 而不是 sim 上的, 否则 policy 可以"绕开" sim 里没采到的不真实区域, 让 sim 算出来的 gap 偏低。

### 2.3 Performance gap

$$G_{\text{perf}} = |J_{\mathcal{M}_s}(\pi) - J_{\mathcal{M}_r}(\pi)|$$

sim-to-real 的最终目标 (Eq. 5) 是 $\pi^* = \arg\min_\pi G_{\text{perf}}$, 而不是直接 $\min G_{\text{dyn}} + G_{\text{perc}}$。这两条优化路径有时方向一致, 有时冲突。

---

## 3. Reality Gap 的来源: 一张完整 taxonomy

paper 把 gap 拆为 4 大类, 14 个子类。我用一个伪数据表来总结每个 sub-gap 的"典型量级"与"最常见缓解手段":

| Gap 来源 | 典型 root cause | 量级举例 | 主流缓解 |
|---|---|---|---|
| Rigid body 假设 | 链节实际可弯曲、关节有 backlash | link compliance ~0.5–2 mm | Residual dynamics learning (Golemo 2018) |
| Chaotic dynamics | 流体、湍流 | 不可重复 | Domain randomization on initial cond. |
| Stochasticity | 地面摩擦随机 | $\mu \sim \mathcal{U}(0.3, 1.2)$ | DR + RMA |
| Battery / power | voltage sag under load | 电压下降 10–20% | System identification of $V(I)$ curve |
| Contact dynamics | point contact + linear friction cone | 真实接触面积 cm² 量级 | Soft contact models, learned residual |
| Parameterization | mass, inertia 误差 | mass ±10%, CoM ±2 cm | SysID (Chebotar 2019) |
| Numerical integrator | Euler vs RK4 vs implicit | $10^{-3}$ 量级能量漂移 | 更高阶 + 更小 dt |
| Human-robot | 不可预测行为 | N/A | 简单 reactive model + DR |
| Wear/thermal | 磨损、温度漂移 | 寿命周期 drift | Online sysID |
| Asset fidelity | 低分辨率 mesh | 1k–10k poly | 高精度 USD asset, NeRF/3DGS |
| Sensor model | pinhole + Z-buffer | 缺 rolling shutter / lens flare | 物理渲染, 光线追踪 |
| Sensor noise | Gaussian 假设 | 真实 depth 噪声非高斯 | 学习 noise model |
| Actuator model | 一阶系统假设 | 时间常数 5–20 ms | 二阶 + 死区 + 齿隙建模 |
| Low-level control | 厂商固件黑盒 | 隐藏滤波 | 软件 stack 对齐, latency 标定 |
| Power electronics | PWM 量化, 电流 cap | 100 µs 延迟 | 加到 simulator 中 |
| Communication | 包丢失, 抖动 | 1–10 ms jitter | 网络模拟 + DR |
| Safety mechanisms | virtual walls | 真实有 sim 没 | 显式建模 |
| POMDP formulation | privileged reward / reset | privileged 信息泄漏 | Asymmetric actor-critic, teacher-student |

这张表是我自己根据 paper Section 3 整理的, 没在原 paper 中出现, 但我觉得用它构建 intuition 会更清晰。

### 3.1 一个特别值得讲清楚的点: 为什么 contact 这么难

paper 3.1.1 的 Contact Dynamics 段落是核心。MuJoCo 用 soft contact + convex approximation, Bullet / PhysX 用 implicit complementarity。真实接触: 接触面会变形, friction 随相对速度变化 (Stribeck curve), sticking/slipping 切换有 hysteresis。这些都会让 policy 在 sim 里学到"通过贴着接触面打边缘 case" 的 exploit。这是为什么 dexterous manipulation 比纯 locomotion 难得多 — locomotion 接触面大且周期性强, 误差被平均掉; manipulation 接触点少且接触面小, 单点误差占主导。

参考: MuJoCo soft contact paper https://mujoco.readthedocs.io/en/latest/computation/index.html ; PhysX articulated toolbar https://nvidia-omniverse.github.io/PhysX/physx/index.html 。

### 3.2 Battery gap — 一个被严重忽视的来源

paper 3.1.1 的 Battery 段落是整篇 paper 最被低估的洞察之一。motor torque $\tau \propto V_{\text{batt}}$, 而真实 $V_{\text{batt}}$ 在大电流下有 sag:

$$V_{\text{batt}}(I) = V_{\text{oc}} - I \cdot R_{\text{internal}}$$

其中 $V_{\text{oc}}$ 是开路电压, $R_{\text{internal}}$ 是内阻 (温度与 SOC 的函数)。这导致快速加速时 transient torque deficit, 这在 quadruped 跳跃或 drone racing 中尤其致命 — drone racing 的 champion-level policy https://www.nature.com/articles/s41586-023-06461-5 就专门把 motor + battery + propeller aerodynamics 一起做了 system identification。

### 3.3 Actuation gap — 几乎没人建模的最深层 gap

paper 3.3 的三个子类 (Actuator Models, Low-level Control, Power Electronics) 是我觉得 sim-to-real 文献中讨论最不充分的部分:

- **Actuator 是高阶非线性的**: 真实 motor 有 electrical time constant $\tau_e \approx L/R$ (通常 1–5 ms) 和 mechanical time constant $\tau_m \approx J/b$ (10–50 ms), 两个一阶串联构成二阶系统。sim 通常建模成一阶 $\dot{\tau} = (u - \tau)/\tau_m$。
- **Low-level 控制器是黑盒**: 比如 Franka 的 joint torque 控制器内部有 anti-windup, resonance suppression 滤波, 这些都不开放。
- **Power electronics**: PWM 分辨率有限 (8–12 bit), 死区时间 (dead time) 在零点附近会有 dead-zone, driver 有 hard current/voltage cap。

直觉: 一个在 sim 里训练的 policy 如果不知道这些, 它会生成"在 sim 看似合理但物理上不可达"的 torque command, 真实驱动器直接饱和或延迟, 闭环就崩。

---

## 4. 解决方案: reducing vs. overcoming

paper Figure 3 的 taxonomy 是整篇 survey 的精华, 我把它扩成下面这个流程图:

```
Reality Gap
├── Reduce the gap (improve sim/sim↔real alignment)
│   ├── Improve simulation fidelity
│   │   ├── System identification (offline / online / iterative)
│   │   ├── Learned residual models (state-side, action-side)
│   │   └── Real-to-sim environment reconstruction
│   ├── Choice of modalities & representations
│   │   ├── Depth/point cloud > RGB
│   │   ├── Keypoint / foundation model embedding
│   │   └── Joint velocity > joint torque
│   └── Design choices
│       ├── High-bandwidth low-level control
│       ├── Software stack alignment
│       ├── Hardware co-design (compliant actuators, passive stability)
│       └── Constrain dynamics (quasi-static)
│
└── Overcome the gap (make policy robust to gap)
    ├── Domain generalization & adaptation
    │   ├── Domain randomization (DR) + automatic DR + offline DR
    │   ├── Adversarial training (RARL)
    │   ├── Meta-learning + RMA (Rapid Motor Adaptation)
    │   └── Domain adaptation (GAN, feature alignment)
    ├── Data selection & exploration
    │   ├── Co-training with real data
    │   ├── Sim-to-real-driven exploration
    │   └── Active system identification (ASID)
    └── Policy architecture & regularization
        ├── Modularity (perception ↔ control 分离)
        ├── Privileged information (teacher-student)
        ├── Representation learning (contrastive, alignment loss)
        └── Policy regularization (smoothness Lipschitz, action penalty)
```

### 4.1 System Identification — 用 Bayes 视角看

paper 6.4 引入 simulation-based inference (SBI):

$$p(\theta | \{s_i^r\}, \{s_j^s\}) \propto p(\{s_i^r\} | \theta) p(\theta)$$

其中 $\theta$ 是 simulator 参数。likelihood $p(\{s_i^r\}|\theta)$ 一般 intractable, 因为 $s^s = g(P_s, \theta)$ 是一个复杂 generative process。解决方法:

- **Approximate Bayesian Computation (ABC)**: 接受 $\theta$ 当 $\|s^s - s^r\| < \epsilon$
- **Neural Posterior Estimation (NPE)**: 用 neural network 直接回归 $p(\theta|s^r)$, 见 BayesFlow https://arxiv.org/abs/2101.10762

这个思路特别 elegant: 把 DR 的 randomization distribution 从 $\mathcal{U}[\theta_{\min}, \theta_{\max}]$ 换成 $p(\theta|s^r)$, 让 randomization 范围与真实数据吻合。BayesSim https://roboticsproceedings.github.io/rss05/papers/2019_X29.pdf 是这条线的开创工作。

### 4.2 Residual Simulation — 公式细节

paper 提了两类 residual:

**State-side residual**: $s_{t+1} = T_{\text{sim}}(s_t, a_t) + f_\theta(s_t, a_t)$, $f_\theta$ 是学出来的网络, 训练目标 $\min_\theta \sum \|s_{t+1}^{\text{real}} - T_{\text{sim}}(s_t, a_t) - f_\theta(s_t, a_t)\|^2$。

**Action-side residual** (inverse dynamics, Christiano 2016 https://arxiv.org/abs/1610.03518): 学 $g_\phi: a^{\text{sim}} \to a^{\text{real}}$ 使 $T_{\text{real}}(s_t, g_\phi(a^{\text{sim}})) \approx T_{\text{sim}}(s_t, a^{\text{sim}})$。等价于在 sim 里用 $a^{\text{sim}}$, 部署时把它通过 $g_\phi$ 转成 real 上能产生相同状态演化的 action。

直觉: state-side 适合 simulator 物理模型整体偏差小, 局部 nonlinear correction 能 fix 的情况; action-side 适合 actuator 建模偏差主导, 因为 action 是 driver 之前的命令, 通过 inverse 把 driver 非线性"反向补偿"掉。

### 4.3 RMA — 隐式 system identification 的代表作

RMA (Kumar 2021, https://roboticsproceedings.github.io/rss17/p022.pdf ) 在 sim 训练时用 privileged info (摩擦、mass、torque limit) 训一个 encoder $e_\psi: \tau_{1:t} \to z_t$, 然后 policy $\pi_\omega(o_t, z_t) \to a_t$。部署时 $z_t$ 用历史 proprioception 在线推断 (再 fine-tune 几步), 完成隐式 sysID。

更细节地, RMA 分两阶段:

1. **Stage 1**: $z_t = e_\psi(e_t)$, $e_t$ 是 environment 参数 (privileged), 在 sim 用 RL 训 $\pi, e$ 联合。
2. **Stage 2**: 训 $\hat{e}_\chi: [o_{t-k:t}, a_{t-k:t}] \to z_t$ 用 behavior cloning, 让它从可观测历史预测 $z_t$。

这个 trick 后来被 Radosavovic 2024 (Real-World Humanoid Locomotion RL, https://www.science.org/doi/10.1126/scirobotics.adi9579 ) 用 teacher-student distillation 的形式推到了 humanoid 上。

### 4.4 Teacher-Student with Privileged Info

paper Section 4.2.3 强调 privileged information 是 sim-to-real 的关键 trick。模式:

- Teacher 在 sim 中训练, observation = (真实 obs, privileged obs: contact, exact object pose, friction, mass...)
- Student 通过 distillation 学只用真实 obs, 模仿 teacher 的 action distribution

实现细节 (Radosavovic 2024): 用 mixed loss $\mathcal{L} = \alpha \mathcal{L}_{\text{RL}} + (1-\alpha) \mathcal{L}_{\text{distill}}$, $\alpha$ 用 schedule, 早期 RL 权重大, 后期 distill 大。

### 4.5 Domain Randomization 的极限

DR 的核心假设: 在 $\theta \sim p(\theta)$ 上训练, 测试时 real $\theta^* \in \text{supp}(p)$。问题:

1. 若 $p$ 范围太大, policy 学到一个"any-environment but mediocre"的策略 (这个 trade-off 在 DeepXplanner 里有量化分析)
2. 若 $p$ 范围太小, real $\theta^*$ 落在分布外
3. Automatic DR (Akkaya 2019 https://arxiv.org/abs/1910.07113 ) 让 curriculum 自动扩张 $p$ 范围, 当当前 performance 超阈值再扩张

### 4.6 Modularity 与 Representation Learning

paper 4.2.3 提到 modularity: 把 perception 和 control 解耦, perception 用预训练 encoder (R3M, VC-1, Voltron), control 用 RL。好处:

- perception 部分可以单独用 real 数据 fine-tune
- control 不受 perception gap 直接影响, 只受 latent 表征 gap 影响

Xing et al. 2024 (Contrastive Learning for Robust Scene Transfer https://arxiv.org/abs/2405.18757 ) 用 contrastive loss 学一个 background-agnostic feature, 让 drone 在不同 visual scene 间 transfer。

---

## 5. 评估指标 — 这部分是 sim-to-real 文献最薄弱环节

### 5.1 SRCC (Sim-to-Real Correlation Coefficient)

$$\text{SRCC} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}$$

其中 $x_i$ 是第 $i$ 个 policy 在 sim 中的成功率, $y_i$ 在 real 中的成功率。SRCC 接近 1 说明 sim 性能是 real 性能的好 predictor, 这对开发迭代至关重要 — 我们希望 sim 性能涨了 real 也涨。Kadian 2020 (https://ieeexplore.ieee.org/document/9017453 ) 是这个 metric 的提出。

**直觉构建**: SRCC 高意味着 simulator 的"相对 ranking" 与 real 一致, 即使绝对成功率差很多。这意味着在 sim 里做 ablation、调超参、选架构是可信赖的。SRCC 低意味着 sim 性能完全无法预测 real, 你必须每次都跑 real experiment, 那 simulator 的价值就只剩"提供 prior"。

### 5.2 Offline Replay Error

$$\mathcal{E}_{\text{replay}} = \frac{1}{T} \sum_{t=1}^T \| s_t^{\text{sim}} - s_t^{\text{real}} \|^2$$

给定一条 real trajectory $\{s_t^{\text{real}}, a_t\}$, 在 sim 里 open-loop 重放 $a_t$, 看每个 $s_t$ 偏多少。

这是诊断 dynamics gap 的最便宜办法: 不需要 closed-loop deploy, 只需要 log 数据。直觉: 如果 open-loop 误差已经爆炸, closed-loop 一定爆炸 (因为 policy 看到的 state 已经偏离了它训练时的 distribution)。

### 5.3 视觉保真度

- 分布级: FID (Fréchet Inception Distance), KID (Kernel Inception Distance), Inception Score, t-SNE
- 单图级: SSIM, PSNR, IPD (Instance Performance Difference, Chen 2024 https://arxiv.org/abs/2411.07375 )

### 5.4 Success Rate / Cumulative Reward / Task-Specific

最直接也最局限的指标。Task-specific 例如: object-distance-to-goal for pushing, lap time for racing, time-to-goal for navigation。

---

## 6. 开放问题与我的延伸

### 6.1 Wrong Models, Better Controllers (Section 6.1)

paper 6.1 提了一个反直觉但极重要的观点: model-based RL 不需要模型准确, 只需要模型在 "high-return area" 准确。Lambert 2018 (Objective Mismatch in MBRL https://arxiv.org/abs/2002.07609 ) 证明: 用 control performance 直接优化模型, 比用 likelihood 优化模型性能更好。

直觉: 模型在 state-action 空间的某些角落不准确没关系, 只要 policy 不会去那里就行。这其实是 model-based RL 的"分布外"问题: 模型在 $p_{\text{rollout}}(s, a)$ 之外不准没影响, 在其内必须准。

### 6.2 Differentiable Simulators (Section 6.2)

paper 6.2 提到 Warp https://developer.nvidia.com/warp-python , Taichi https://github.com/taichi-dev/taichi , JAX https://github.com/google/jax 。这些工具能解析地算梯度, 让 sysID 与 policy learning 都可以 gradient-based。

但 differentiable sim 的痛点: contact 这种 non-smooth 现象梯度不定义。解决方案:

- Smooth contact approximation (MuJoCo 的 soft contact)
- Implicit differentiation (Brakel et al. https://arxiv.org/abs/2110.00571 )
- Stochastic gradient via score function (likelihood-ratio)

### 6.3 World Models (Section 6.3) — 我最兴奋的方向

paper 6.3 把 video model 和 world model 放在一起讨论:

- **Video model**: 给定过去帧 + 条件 (text, action), 生成未来帧。代表: Sora, Genie https://arxiv.org/abs/2402.15391 , GAIA-1。
- **World model**: 学环境的 internal representation, 可以 imagine 未来 state 用于 planning。代表: DreamerV3, Genie, Cosmos https://arxiv.org/abs/2501.03575 , GEM https://arxiv.org/abs/2504.08007 。

paper 提出关键 insight: world model 用 **real 数据** 训练, 天然 reality gap 小。这跟 simulator 互补 — simulator 是 white-box 物理模型, world model 是 black-box 数据驱动模型。未来这两条线会融合: simulator 提供 prior + 大规模 data bootstrapping, world model 提供真实 dynamics 适配。

延伸思考: 如果 world model 能做到足够好, sim-to-real 就变成 "world-model-to-real", 而 world model 在 real 上训, gap 天然小。问题就变成: 怎么用 sim 数据 bootstrap world model, 然后在 real 上 fine-tune。这个 paradigm 类似 vision-language model 在 web 数据上 pretrain, 在下游 fine-tune。

### 6.4 Simulation-Based Inference (Section 6.4)

已经在 4.1 讲过。补充: NPE (Neural Posterior Estimation) 在 robotics 上的代表:

- BayesSim (Ramos 2019): DR distribution 用 posterior
- Neural Posterior Domain Randomization (Muratore 2022 https://arxiv.org/abs/2205.10576 )
- STReSSD (Matl 2021): 从声音做 sysID

### 6.5 Simulation for Large Robotics Models (Section 6.5)

这一节对 2024-2026 的 robotics foundation model (RT-2, Open X-Embodiment https://robotics-transformer-x.github.io/ , GR00T N1 https://arxiv.org/abs/2503.14734 , π0) 极其相关:

- real 数据有限 (DROID https://droid-dataset.github.io/ , Open X-Embodiment), sim 可以 augment
- MimicGen https://arxiv.org/abs/2310.17569 : 从少量人类 demo 自动生成大量 sim demo
- DexScale (Liu 2025 https://arxiv.org/abs/2503.20570 ): 自动化 sim 数据 scaling

关键开放问题: 给 sim 数据加多少 reality gap reduction 才能让 VLA / foundation policy transfer? 如果 reality gap 大, 大模型可能反而"过拟合 sim artifact", 比 small policy 还差 (Barreiros 2025 https://arxiv.org/abs/2507.05331 的观察)。

---

## 7. 我的几点延伸思考 (build your intuition)

### 7.1 Sim-to-Real 的本质是 distribution mismatch

把 sim 看成 $p_{\text{sim}}(s, a)$, real 看成 $p_{\text{real}}(s, a)$。policy $\pi$ 学的是 $p_{\text{sim}}$ 上的最优, deploy 在 $p_{\text{real}}$。所有 sim-to-real 方法都是某种形式的 distribution shift 缓解:

- DR: 把 $p_{\text{sim}}$ 扩张到包含 $p_{\text{real}}$
- SysID: 把 $p_{\text{sim}}$ 拉到 $p_{\text{real}}$
- Residual: 把 $p_{\text{sim}}$ 修正到接近 $p_{\text{real}}$
- Domain adaptation: 学 invariant feature 使 $p_{\text{sim}}(z) \approx p_{\text{real}}(z)$
- Co-training: 在 $p_{\text{sim}} \cup p_{\text{real}}$ 上混合训练

这跟 supervised learning 的 domain adaptation 完全同构, 只是这里 distribution 在 trajectory 空间, 且 closed-loop 让 mismatch 复合放大。

### 7.2 Closed-loop 误差累积: stability 视角

考虑一个 LQR 问题, real dynamics $A_r, B_r$, sim $A_s, B_s$。Closed-loop policy $u = -K x$。Real closed-loop: $x_{t+1} = (A_r - B_r K) x_t$, 期望 stable 当 $\rho(A_r - B_r K) < 1$。Sim 训练得到的 $K$ 在 sim 上 stable: $\rho(A_s - B_s K) < 1$, 但 real 上可能 $\rho > 1$。

直觉: closed-loop 让小误差指数放大。这就是为什么 offline replay error 是个非常 conservative 的指标 — open-loop 误差大不一定 closed-loop 失败 (policy 能纠正), 但 open-loop 误差小则 closed-loop 几乎一定 OK。

### 7.3 Contact-rich 比 Free-space 难的本质

Free-space 飞行/游泳/走路, dynamics smooth, $G_{\text{dyn}}$ 可以用 Lipschitz constant bound:

$$\|T_{\text{real}}(s,a) - T_{\text{sim}}(s,a)\| \leq L \cdot \|s - s^*\|$$

policy 通过 robust 控制 (Lipschitz-bounded policy) 就能克服。Contact-rich manipulation, $T$ 在 contact 切换点不可微, 局部 Lipschitz constant 爆炸, 必须依赖精确 contact 建模或大量 DR。

### 7.4 Hardware-Software Co-design

paper 4.1.3 提到硬件 co-design 是减少 gap 的 fundamental 路径, 但没展开。我的直觉: 把驱动器做成"线性 + 低延迟 + 高带宽"比在 sim 里建精确的复杂模型更划算。例子:

- 直接驱动 (DD) motor 比带 gear 的 motor 好建模
- Quasi-direct drive (QDD) 在 legged robot 上的成功 (MIT Mini Cheetah)
- Proprioception-only (无 exteroception) 的 locomotion 能 transfer 就是因为 gap 小

这跟 ML 里的"模型-数据" co-design 类似: 与其花大力气做精确模型, 不如让硬件对模型不敏感。

### 7.5 VLA / Foundation Model 在 sim-to-real 上的新挑战

把 sim-to-real 放到 2025-2026 的 robotics foundation model 语境下:

- RT-2, OpenVLA, GR00T N1 这些 VLA 大部分用 real demo (DROID, Open X), sim 数据起 augmentation 作用
- VLA 的 perception gap 通过大规模预训练 + DR 大幅缩小
- VLA 的 dynamics gap 反而更突出 — 因为 VLA 通常用 end-effector pose 或 joint position 作 action, 这些抽象的 action 在 real 上会被底层 controller "过滤", sim 里没建模这个 filtering 就出 gap
- Universal Manipulation Interface (UMI https://arxiv.org/abs/2402.10329 ) 提出一个聪明做法: 让 sim 和 real 用同一个 handheld gripper interface, 强制 action 语义对齐

直觉: foundation model 时代, sim-to-real 的 bottleneck 从 perception (被预训练解决) 转到 dynamics + low-level control。这部分 paper 3.3 已经预测到了, 是 paper 最有前瞻性的章节之一。

### 7.6 Neural Simulator / World Model 路线 vs. Physics Simulator 路线

paper 6.3 已经在讨论 world model 替代或补足 physics simulator 的可能。我倾向于: 短期 (2-3 年) hybrid 主导 — physics simulator 提供骨架 (kinematics, basic dynamics, collision detection), world model 提供"软"部分 (friction 细节, contact 不确定性, perceptual rendering) 的 residual。长期 (5-10 年) 看 world model 能不能 scale 到 universal embodiment, 如果能, 物理 simulator 会退化成"训练 world model 的 prior"。

Cosmos https://arxiv.org/abs/2501.03575 , Genie 2 https://arxiv.org/abs/2402.15391 , V-JEPA 2 https://ai.meta.com/blog/v-jepa-2-world-model-physical-understanding/ 都在朝这个方向走。

---

## 8. 一份给研究者的"sim-to-real recipe"

paper Section 4 给了 6 步 recipe, 我加上每步对应的"代价/收益"判断:

1. **设计 sim 包含所有相关变量** — 低代价高收益, 但容易遗漏关键变量
2. **尽量 reduce gap** (sysID, fidelity) — 中代价高收益, 边际效益递减
3. **设计 training method 克服剩余 gap** — 中代价高收益, 需要 task-specific 知识
4. **massively parallel training** — 高 GPU 代价, 但单位数据成本低
5. **Real 评估** — 高代价不可绕过
6. **根据 real performance 调 sim 参数再训** — 迭代闭环, 唯一能系统性收敛的路径

我额外加一条 personal recipe: **从第一天就把 real 评估集成进 dev loop**, 别等到 sim 跑通再上 real, 否则 sim gap 累积到无法诊断。这条参考 Zhao et al. Iterative Residual Tuning https://arxiv.org/abs/2003.03075 。

---

## 9. 参考链接清单

我整理了一份关键的 web 链接方便深挖:

- Survey 原文 (arXiv 应该会有): 暂未上线, 可关注 Elie Aljalbout 主页 https://rpg.ifi.uzh.ch/people.html
- RPG (UZH Scaramuzza lab) https://rpg.ifi.uzh.ch/research_groups.html
- NVIDIA robotics research http://rmp.nvidia.com
- Isaac Sim https://docs.omniverse.nvidia.com/isaacsim/latest/
- MuJoCo https://mujoco.readthedocs.io/
- Orbit (Isaac Lab) https://isaac-sim.github.io/IsaacLab/main/
- Genie (DeepMind world model) https://arxiv.org/abs/2402.15391
- Cosmos (NVIDIA world foundation model) https://arxiv.org/abs/2501.03575
- RMA (Rapid Motor Adaptation) https://roboticsproceedings.github.io/rss17/p022.pdf
- Champion-level Drone Racing https://www.nature.com/articles/s41586-023-06461-5
- Real-World Humanoid Locomotion RL https://www.science.org/doi/10.1126/scirobotics.adi9579
- Open X-Embodiment https://robotics-transformer-x.github.io/
- DROID dataset https://droid-dataset.github.io/
- GR00T N1 https://arxiv.org/abs/2503.14734
- MimicGen https://arxiv.org/abs/2310.17569
- UMI https://arxiv.org/abs/2402.10329
- BayesFlow (NPE) https://arxiv.org/abs/2101.10762
- BayesSim https://roboticsproceedings.github.io/rss05/papers/2019_X29.pdf
- Christiano Inverse Dynamics https://arxiv.org/abs/1610.03518
- Domain Randomization review https://www.frontiersin.org/articles/10.3389/frobt.2022.799893/full
- NVIDIA Warp https://developer.nvidia.com/warp-python
- Taichi https://github.com/taichi-dev/taichi
- JAX https://github.com/google/jax
- V-JEPA 2 https://ai.meta.com/blog/v-jepa-2-world-model-physical-understanding/

---

## 10. 一句话总结

这篇 paper 的真正贡献是给 sim-to-real 提供了一个 **POMDP-based, gap-decomposed, reducing-vs-overcoming dichotomy** 的统一语言, 让所有方法都能在同一个 taxonomy 下定位和比较。读完之后, 你应该能在看到任何一篇新 sim-to-real paper 时, 第一时间判断: 它 reduce 哪个 sub-gap, 或者 overcome 哪个 sub-gap, 用什么 metric 验证, 代价多少。这比记住"DR 好"或"RMA 好"有用得多 — 它给你的是地图, 不是罗盘。

如果你想 build deeper intuition, 我建议挑一条 sub-gap (比如 contact dynamics), 把 paper 引用的 3-4 篇代表作 (Dex-Net, NeuroBEM, IndustReal, RMA) 都过一遍, 看每种方法在同一个 gap 上的不同处理方式, 你会发现 reducing 和 overcoming 是同一种物理直觉的两种数学表达。
