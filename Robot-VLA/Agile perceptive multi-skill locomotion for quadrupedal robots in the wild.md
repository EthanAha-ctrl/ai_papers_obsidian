---
source_pdf: Agile perceptive multi-skill locomotion for quadrupedal robots in the
  wild.pdf
paper_sha256: 32f1dfe40c0a6b2d8caad5e700f6bedccfb197759421f17059f446e8df7c0341
processed_at: '2026-08-18T00:23:29-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在干啥

Reference: https://www.science.org/doi/10.1126/scirobotics.adz7397

---

## 一句话总结

他们让一只机器狗在野外森林、城市台阶、室内障碍物上跑，最高瞬间速度 6 m/s，过程中自己根据地形和速度切换 trot / bound 两种步态，全程只用 onboard 传感器，零样本 sim-to-real。

---

## 难点在哪

机器人 locomotion 听起来像控制问题，实际上是个"数据来源"问题。你想训一个 RL policy 让机器狗跑，最大的瓶颈不是 PPO 调参，是**"什么是好的动作"这个信号从哪来**。

几个传统路子都有问题：

**1. 手写 reward → RL 从零学**
所有东西都得 reward engineer。你得想清楚怎么罚 foot dragging、怎么鼓励 foot clearance、怎么避免 body shake。Reward 一多就开始互相打架（foot clearance 要抬腿，但抬太高又消耗 energy）。而且 RL 从零探索，接触动力学是高度非凸的，policy 经常陷在 local optimum 里走不出 trot，更别说跳 60 cm 台阶了。

**2. 用动物 mocap 数据当 reference（AMP / ASE）**
AMP [Peng 2021](https://arxiv.org/abs/2104.02180) 训一个 discriminator，policy 动作越像 mocap 越给奖励。问题是：
- 动物 mocap 数据采集贵、规模小
- Discriminator 会把 policy 锁死在 mocap 分布附近。mocap 里没有"跳台阶"的样本，policy 想跳，discriminator 就给低分
- 而且 mocap 只有 kinematics（关节角度），没有 dynamics（torque）。policy 还得自己学怎么 produce 那个运动

**3. HRL 分两个 expert + 高层 selector**
先训个 trot expert，再训个 bound expert，上层 selector 选哪个。问题：两个 expert 各自训的时候 latent space 不共享，切换时 distribution mismatch，4 m/s 跑着突然从 bound 切 trot，相位不对就直接摔。[ANYmal Parkour](https://www.science.org/doi/10.1126/scirobotics.adi7566) 就是这个路子，但只在训练时见过的障碍上能 deploy。

**4. Trajectory optimization 给 state reference，再训个 tracking policy**
TO 算出来最优轨迹（关节角、CoM 轨迹），然后用 RL 学一个 policy 去 track 这个轨迹。[DTC, Jenelten 2024](https://www.science.org/doi/10.1126/scirobotics.adh5401)、[Wu 2023](https://arxiv.org/abs/2303.14524) 都是这个思路。缺点：多了一个 RL 阶段，sample efficiency 差，而且 tracking 的 reward design 也很烦。

---

## 这篇 paper 的 trick

核心一句话：**用 trajectory optimization 不只生成"该怎么动"的 state 轨迹，还同时生成"该出多少力"的 torque，然后 pretrain 一个 Transformer VAE encoder + 两个 gait-specific 的 torque decoder，RL policy 只需要在 16 维 latent space 里挑动作 + 12 维 auxiliary 修正项。**

类比一下你熟悉的 LLM 范式：

- TO 数据 = synthetic pretraining corpus（不是 scrape internet，是 scrape 物理优化器）
- TVAE = pretrain 一个 representation
- Gait-specific decoder = 两个 task head
- RL policy = downstream fine-tune
- Auxiliary action = LoRA / adapter，冻结主模型只调小头
- Gait selection logit = 1 维 MoE router

### 为什么 TO 给 torque 这件事关键

之前的工作用 TO 都是只拿 state trajectory（关节角、速度、CoM 轨迹），然后再训 RL policy 去 track。这篇直接让 TO 同时输出 torque，然后训一个 decoder 学 "state → torque" 映射。

好处：
- Decoder 直接输出 12 维关节 torque，**不用再训 tracking policy**
- TO 给的 torque 是物理一致的（满足动力学），不像 mocap 只有 kinematics
- Decoder freeze 之后就是个"动作字典"，policy 只负责查字典 + 微调

### 2D SRBD TO 怎么这么快

他们用 Single Rigid Body Dynamics（SRBD），把机器狗简化成 sagittal plane（侧视图）的一个刚体，自由度只有 $(x, z, \theta)$ 三个：水平位置、垂直高度、pitch 角。前后两对腿各自用一条合成的 ground reaction force（GRF）表示。

GRF 用 Bézier curve 参数化，约束一个 gait cycle 内净冲量为零（动量守恒）：

$$
m(\dot{x}_T - \dot{x}_0) = \int_0^T (F_F^x + F_H^x)\, dt
$$

- $m$: 机器狗质量 45 kg
- $\dot{x}_T, \dot{x}_0$: 一个 gait cycle 末端 / 起始的水平速度
- $F_F^x, F_H^x$: 前腿 / 后腿的水平 GRF
- $T$: 一个步态周期，约 0.3 秒

意思就是：跑一步前后水平动量不变，所以 GRF 在一个周期内积分必须为零。Bézier curve 的 control points 满足这个约束后，$\dot{x}, \dot{z}$ 自动周期化，剩下只优化 $z, \theta, \dot{\theta}$ 周期性。

Newton 法找周期轨道：

$$
f(\mathbf{q}_0) = (z_T - z_0)^2 + (\theta_T - \theta_0)^2 + \lambda_1(\dot{\theta}_T - \dot{\theta}_0)^2 + \lambda_2 \theta_{\max}^2
$$

- $\mathbf{q}_0 = (z_0, \theta_0, \dot{\theta}_0)$: 初始条件
- 前三项: 周期性约束（周期末状态 = 周期初状态）
- $\lambda_2 \theta_{\max}^2$: 罚过大的 pitch 振荡，bound 用 $\lambda_2 = 1$，trot 用 $\lambda_2 = 0$（trot 本来 pitch 就小）

Bézier 曲线的好处：Jacobian 和 Hessian 都能解析求，Newton 一步迭代 21 µs。结果：

**15.5 小时的 motion-torque paired data，8 分钟生成完。**180,000 条轨迹，500 万 time steps，速度 -2 ~ 7 m/s 全覆盖。

这是整个 paper 的 throughput 核心 —— 没有 TO 这么快，pretraining 数据规模根本撑不起来。

---

## 三个 stage

### Stage 1: Pretrain TVAE + Decoder（离线一次）

输入：3 帧 proprioception（关节角、速度、身体姿态、接触状态等，69 维 × 3 帧）

```
proprioception (69×3) → Transformer encoder (8 heads, 2 layers, hidden 64)
                    → 16 维 latent z（μ, σ）
                    → reparameterization 采样
                    → 两个 decoder head:
                       - trot decoder → 12 维 torque
                       - bound decoder → 12 维 torque
```

Loss：
$$
\mathcal{L}_{\text{TVAE}} = \underbrace{\sum_{t=1}^{3} \|s_t - \hat{s}_t\|_2^2}_{\text{reconstruction}} + 0.1 \cdot \underbrace{D_{KL}(q(z|s) \| \mathcal{N}(0, I))}_{\text{latent 正则}}
$$

Decoder 单独训，loss 就 torque MSE：
$$
\mathcal{L}_{\text{dec}} = \|\tau_t - \hat{\tau}_t\|_2^2
$$

关键设计：**encoder 共享，decoder 分 gait**。这样 latent space 是 unified 的（trot 和 bound 在同一个 16 维空间里），但 torque mapping 是 gait-conditioned。这跟 [Petrovich ActCond TVAE 2021](https://arxiv.org/abs/2104.05370) 同构。

### Stage 2: RL Fine-tune（Isaac Gym PPO）

Policy 输入 203 维观测（proprio 117 + extero latent 32 + gait info 54），输出 29 维 action：

- $a_{\text{latent}} \in \mathbb{R}^{16}$: 输入 pretrained decoder，出 12 维 feedforward torque $\tau_{\text{dec}}$
- $a_{\text{aux}} \in \mathbb{R}^{12}$: auxiliary offset，走 PD controller
- $\text{logit}_{\text{gait}} \in \mathbb{R}^1$: 过 sigmoid，> 0.5 选 bound，< 0.5 选 trot

最终 torque：

$$
\tau_{\text{input}} = \tau_{\text{dec}} + k_p(q_{\text{default}} - q_t + a_{\text{scale}} \cdot a_{\text{aux}}) - k_d \dot{q}_t
$$

- $\tau_{\text{dec}}$: decoder 出来的 feedforward torque，**carry 主信号**（gait 周期、CoM 动力学、contact timing）
- $k_p = 80, k_d = 2$: PD 增益
- $q_{\text{default}}$: 站立默认关节角
- $q_t, \dot{q}_t$: 当前关节角 / 速度
- $a_{\text{scale}} = 0.2$: auxiliary 缩放
- $a_{\text{aux}}$: policy 学的修正项

直觉：**$\tau_{\text{dec}}$ 像 pretrain 模型的 frozen 主干，$a_{\text{aux}}$ 像 LoRA adapter**。decoder freeze，policy 只调 latent + aux。

频率分配也像生物：
- Gait selection: 2 Hz（像动物的 central pattern generator）
- Latent + aux action: 100 Hz（像 spinal reflex）

### Stage 3: Perception Distillation（DAgger）

Teacher 用 privileged heightmap（255 维 raw），student 用 depth camera + 2D LiDAR。

为什么必须 LiDAR：4 m/s × 2 Hz decision rate = 至少 4 m lookahead。RealSense D435 depth 有效距离只有 0.3–3 m，不够。Hokuyo UST-30LX 2D LiDAR 给 0.6–5 m 范围，40 Hz 更新。

Student 架构：
```
depth image (87×58×10 history) → CNN → 128 维
                                    ↓
LiDAR (45 维) + proprio + action history → MLP
                                    ↓
                              concat → GRU → 32 维 latent
```

DAgger + BPTT 训练，MSE 蒸馏 teacher 的 32 维 latent。

---

## 为什么这套设计有效

### Latent space 天然 smooth transition

trot 和 bound 共享一个 encoder 训出来的 16 维 latent space。PCA 可视化（Fig. 5A）显示：
- Pretraining 数据：trot 和 bound 两个 cluster，部分 overlap
- RL 训练早期：policy latent 集中在 pretrained cluster 子集
- RL 后期：扩展到 cluster 外围
- Real-world deploy：familiar gait 时 overlap pretrained，unseen terrain（log jump）时 explore novel region

因为 latent space 是 shared 的，trot ↔ bound 之间天然 smooth interpolation。HRL 的两个独立 expert 切换时 distribution mismatch，APT-RL 没这问题。

### Auxiliary action 处理 OOD

Fig. 5C-E 三个 case study 特别直观：

**Log jump（TO 数据里没有的动作）**：
跑平地时 $\tau_{\text{dec}}$ 主导，起跳瞬间 $\tau_{\text{aux}}$ 显著增大，生成 TO dataset 没有的 jumping torque。

**Leg breakage（OOD 故障）**：
断腿后 policy 主动减小 HFE motor 的 $\tau_{\text{dec}}$，增大 $\tau_{\text{aux}}$，改变腿轨迹维持平衡。

**In-place rotation（TO 2D 数据里没有 yaw）**：
HAA（髋关节外展内收）motor 在 2D 数据里没有 torque → $\tau_{\text{dec}} = 0$，完全靠 $\tau_{\text{aux}}$ 实现 yaw 旋转。KFE（膝关节）仍由 $\tau_{\text{dec}}$ 主导。

直觉：**$\tau_{\text{dec}}$ 是 in-distribution feedforward，$\tau_{\text{aux}}$ 是 OOD adaptation**。跟 LoRA 在 LLM 里的角色完全一致。

### Feedforward torque 不可替代

Fig. S8 把 $\tau_{\text{dec}}$ scale 从 0% → 100%：

- 0%: success rate **2.9%**
- 100%: success rate **94.6%**

去掉 $\tau_{\text{dec}}$，只剩 PD + aux 完全不行。这证明 TO-pretrained torque prior 是 dominant component，aux 单独撑不起来。

### PD-target 形式等价

他们还证明 $\tau_{\text{input}} = \tau_{\text{dec}} + k_p(q_{\text{default}} - q_t + a_{\text{scale}} a_{\text{aux}}) - k_d \dot{q}_t$ 可以代数重写成标准 PD：

$$
\tau_{\text{input}} = k_p(q_{\text{default}} + q_{\text{ref}} - q_t) - k_d \dot{q}_t, \quad q_{\text{ref}} = a_{\text{scale}} a_{\text{aux}} + \frac{\tau_{\text{dec}}}{k_p}
$$

两种形式 success 94.3% vs 94.3%，vel tracking 1.383 vs 1.382 —— 几乎完全一样。所以 torque formulation 和 PD-target formulation 是 representational choice，不是 capability 差异。

---

## 实际跑得怎么样

### Froude number 对比动物

$$
Fr = \frac{v^2}{g \cdot L}
$$

- $v$: locomotion speed
- $g$: 9.81 m/s²
- $L$: 腿长 ~0.5 m

| 场景 | 速度 | Fr | 动物对应 |
|------|------|-----|--------|
| 60 cm 台阶跳过 | 4.25 m/s | 3.85 | slow gallop |
| 三级楼梯跳下（落地前峰值）| 6.0 m/s | 7.69 | fast gallop |

动物经验值 [Hoyt & Taylor 1981 Nature](https://www.nature.com/articles/292239a0):
- walk: Fr < 0.5
- trot: 0.5 < Fr < 2.5  
- gallop: Fr > 2.5

这台机器狗达到了动物 fast gallop 的 Froude 量级，是 perceptive quadruped 新 benchmark。

### Gait 自动选择

Fig. 6 几个真实实验：
- 同样 2 m/s 命令：低障碍（0.175 m）→ trot；高障碍（0.44 m）→ bound
- 同样地形：1 m/s → trot；>4 m/s → bound

Simulation 上测 gait fraction（Fig. 6B）：bounding fraction 随 speed 和 difficulty 单调上升，但不同地形斜率不同。High step / hurdle / gap 低速就开始 bound；rough / discrete 高速才转。

Aggregate metrics 对比 Auto / Trot-only / Bound-only（Fig. 6C-ii）：

| 策略 | Best perf rate | Avg regret | Worst case |
|------|----------------|------------|------------|
| Auto | **44.44%** | **4.99%** | **0.711** |
| Trot | 23.81% | 24.63% | 0.028 |
| Bound | 31.75% | 13.28% | 0.137 |

Auto 三项全面胜出，worst case 0.711 远高于 trot 的 0.028。说明 multi-gait 不只是平均好，是 **robustness 显著好**。

### vs AMP / Vanilla RL（Fig. 7B）

- AMP: success 接近，但 COT 更高（discriminator 把 policy 锁在 reference 附近，obstacle adaptation 差）
- Vanilla RL: COT 接近 APT-RL，但 success rate variance 大，每个 terrain 都要 reward tuning
- APT-RL: 一致地 low COT + high success，虽然 motion prior 只来自 flat ground

### vs HRL + Residual（Fig. 7C, D）

HRL 早期得益于 pretrained expert，unseen terrain 上卡住，residual policy 也救不回来。APT-RL 全程探索 latent space（exploration bonus 逐渐 decay 到零），最终更高 terrain difficulty + 更高 velocity tracking reward，sample 还更少。

### Random gait transition robustness（Fig. 7E）

2 Hz 随机切换 trot↔bound，50% 概率：
- APT-RL transition success 高
- HRL + residual 在高速 + 高 difficulty 大幅下降
- Base pitch 轨迹：APT-RL 平滑过渡；HRL 失稳 stumbling

### Sensor ablation（Fig. 8）

- Depth only: 强在 hurdles、stepping stones（precise local geometry）
- LiDAR only: 强在 low stairs、rough、gaps、high steps（long range）
- Both: 全面最优

---

## 工程上容易忽略的细节

### 机械振动吸收器（Fig. S1）

机器狗高速 bound 时冲击 >10g，机械旋转式 LiDAR 经常 shutdown。

解决：3D printed PLA damping 结构，装在 LiDAR 和 head 之间。Spiral springs 抑制 rotational moment，10 mm 位移内 low translational stiffness + high rotational stiffness。FEA 验证。

这跟 [Burden 2024 "Why animals can outrun robots"](https://www.science.org/doi/10.1126/scirobotics.adi9754) 指出的 sensor-body coupling 问题对应。动物有 muscle + tendon 做 passive damping，robot 必须专门 mechanical design。Software-only 解不了。

### 控制频率分层

- Low-level torque control: 2 kHz
- Action policy（GRU + MLP actor）: 100 Hz
- Perception module（CNN inference）: 25 Hz（GPU TensorRT）
- Estimator: 1 kHz
- Camera: 25 Hz
- LiDAR: 40 Hz
- Data logging: 2 kHz

Policy 在 CPU 用 ONNXRUNTIME，perception 在 GPU 用 TensorRT。CPU 跑 CNN 实时性不够。

---

## Limitations

作者自己承认：

1. **只 sagittal plane motion**：没 rapid turning、lateral walking。需要 3D TO dataset 扩展
2. **只 trot + bound**：pace、gallop、crawl 没集成
3. **Robot-specific TO**：换机器人要重做 TO + retrain decoder（preliminary 在 ANYmal、Go1、HOUND bipedal mode 上 demo 了，Movie S4）
4. **没 high-level navigation + semantic understanding**：现在是 velocity command 驱动，没 autonomous goal-directed exploration

---

## 我觉得你可能会感兴趣的几个 parallel

### 1. 这就是 LLM pretrain + fine-tune 范式的 robotics 版本

LLM: scrape internet → pretrain LM → fine-tune on task
APT-RL: TO 生成 synthetic motion-torque data → pretrain TVAE + decoder → RL fine-tune on terrain

TO data 在这里扮演 "synthetic pretraining corpus"，跟用 code / math synthetic data pretrain LLM 思路一致 [DeepSeekMath, Liu 2024](https://arxiv.org/abs/2402.03300)。

### 2. Latent action codebook ≈ VQ-VAE codebook

TVAE 的 16 维 continuous latent 跟 [VQ-VAE, van den Oord 2017](https://arxiv.org/abs/1711.00937) 的 discrete codebook 类似，只是 continuous + reparameterization。RL policy 在 codebook 里"predict next token"式选 action。

### 3. Auxiliary action ≈ LoRA / Adapter

- Base model（decoder）: frozen，carry 主信号
- Adapter（aux action）: small trainable，task-specific 修正
- 跟 [LoRA, Hu 2021](https://arxiv.org/abs/2106.09685) 哲学一致

### 4. Gait selection logit = 最简 MoE

$a_{\text{gait}}$ 是 1 维 router，选 trot expert 或 bound expert。这就是最简化的 [MoE, Shazeer 2017](https://arxiv.org/abs/1701.06538)，只是 expert 是 decoder 而不是 full network。

### 5. Distillation = Knowledge Distillation

Teacher（heightmap + privileged）→ Student（depth + LiDAR），MSE loss on latent。跟 [Hinton 2015 KD](https://arxiv.org/abs/1503.02531) 完全同构，modality 是 perception 而不是 logits。

### 6. Curriculum + exploration bonus decay

Terrain 10 级 curriculum + latent exploration bonus 逐渐 decay 到零。这跟 LLM RLHF 里 KL penalty decay、curriculum on task difficulty 一致 [InstructGPT, Ouyang 2022](https://arxiv.org/abs/2203.02155)。

---

## 最后的 intuition

这篇 paper 真正的 contribution 不是某个单一 trick，是把五个 piece 组合起来才跑通：

1. **2D SRBD TO 大规模生成 paired (state, torque) data** —— 让 pretraining 数据来源 physically grounded 且便宜
2. **TVAE + gait-specific decoder pretrain** —— latent space unified，torque mapping 分 gait
3. **Latent action + auxiliary action joint training** —— frozen prior 主导 + small adapter 修正
4. **Depth + LiDAR dual modality perception distillation** —— 解决高速 lookahead 不够的问题
5. **Mechanical vibration isolation** —— software 解不了的 sensor coupling 问题

缺一个都跑不到 6 m/s。这就是 robotics 跟 ML 最大的不同 —— ML paper 一个 trick 改个 loss 就能涨点，robotics paper 必须 hardware + algorithm + perception + mechanical 一起对齐才 work。

Engineering takeaway：**TO 数据 + VAE latent codebook + hybrid torque/PD control + dual-modality perception + mechanical vibration isolation**，这五件套是 KAIST HOUND 跑到 6 m/s 的 recipe。

Paper link: https://www.science.org/doi/10.1126/scirobotics.adz7397
Code: https://zenodo.org/records/20645964
Supplementary movies: Movie S1-S4（urban + forest deploy, aux action analysis, gait selection, multi-robot generalization）

---

# APT-RL: 用 Trajectory Optimization 数据 Pretrain 一个 Torque Decoder

这篇 paper 是 KAIST Hae-Won Park 组发在 Science Robotics 2026 的工作,核心思路其实非常 Karpathy-style ——把 robotics 控制问题重新 cast 成一个 "pretrain representations + downstream RL fine-tune" 的范式,只不过 pretrain data 来自 trajectory optimization 而不是 internet scrape。

Reference link: https://www.science.org/doi/10.1126/scirobotics.adz7397
Code/Data: https://zenodo.org/records/20645964

---

## 1. 问题本质:为什么 prior 工作都做不到 6 m/s + 多 gait + 真实野外

先 build 一下 intuition。Quadruped 高速 locomotion 的难点在于:

- **多 gait 共存**: trot (低速稳定、COT 低)、bound (高速、可飞越障碍)、pronk、pace、gallop...动物根据 speed 和 terrain 自动切换 [Hoyt & Taylor 1981, Nature](https://www.nature.com/articles/292239a0)
- **Gait transition 要平滑**: 在 4 m/s 跑动时突然从 bound 切到 trot,落地相位、contact timing 都要对齐,不然直接 face plant
- **Perception 要 lookahead**: 4 m/s + 2 Hz 决策 = 至少 4 m lookahead,depth camera (RealSense D435) 有效距离只有 0.3–3 m,不够

Prior 工作的痛点:
- ANYmal Parkour [Hoeller et al. 2024, Sci Rob](https://www.science.org/doi/10.1126/scirobotics.adi7566): HRL 选 skill,但只能在 training 时见过的 obstacle 上 deploy
- Robot Parkour Learning [Zhuang et al. 2023, CoRL](https://arxiv.org/abs/2309.05665): 速度 ~1 m/s
- AMP [Peng et al. 2021, ToG](https://arxiv.org/abs/2104.02180): 需要 animal mocap data,且 discriminator 把 policy 锁在 reference distribution 附近,unseen terrain 难泛化
- Rapid locomotion [Margolis et al. 2024, IJRR](https://arxiv.org/abs/2208.07860): 单 gait 高速,没 multi-skill

这篇 paper 的 key insight:**TO 不只给 state trajectory,还给 torque**。这就避免了 AMP / ASE / DreamWaQ 那种 "先有 reference motion,再训一个 tracking policy" 的两阶段设计。直接 pretrain 一个 torque decoder,RL policy 只需要在 latent space 里 pick action。

---

## 2. Pipeline 三阶段总览

```
Stage 1: Representation Learning (offline, 数小时一次)
   2D SRBD TO → 180k trajectories (state + torque pairs)
        ↓
   TVAE encoder (Transformer, 16-dim latent z)
        ↓
   Trot decoder + Bound decoder (各自 12-dim torque output)

Stage 2: Reinforcement Learning (Isaac Gym, PPO)
   Policy π(s) → [a_latent(16), a_aux(12), logit_gait(1)]  = 29-dim action
        ↓
   Gait select (2 Hz) → trot/bound decoder → τ_dec
   τ_input = τ_dec + k_p·(q_default - q_t + a_scale·a_aux) - k_d·q̇_t

Stage 3: Perceptual Distillation (DAgger + BPTT)
   Teacher: 255-dim raw heightmap + LiDAR
        ↓ distill
   Student: CNN(depth 87×58×10) + MLP(LiDAR 45-dim) + GRU → 32-dim latent
```

整体直觉:**TO 给你一套 "torque prior dictionary",TVAE 把它压缩成 16-dim latent codebook,RL policy 在 codebook 里挑动作,auxiliary action 负责 codebook 外的修正**。这跟 LLM 里 "pretrain embedding + task head + adapter" 的结构同构。

---

## 3. Stage 1 详解:Impulse-scale Trajectory Optimization

### 3.1 2D SRBD 模型

机器人简化为 sagittal plane 的 single rigid body,广义坐标:

$$
\mathbf{q} := (x, z, \theta)
$$

- $x$: body 水平位置 (fore-aft)
- $z$: body 垂直高度
- $\theta$: body pitch angle

Front / Hind 两对腿,每对用一条合成的 ground reaction force (GRF) 表示。注意这是 2D 模型,所以没有 roll、yaw、lateral。

### 3.2 周期性约束 — 动量守恒

为了让 gait cycle 周期化,需要 net impulse = 0(一个 cycle 后动量回到初值):

$$
m(\dot{x}_T - \dot{x}_0) = \int_0^T (F_F^x + F_H^x)\, dt
$$

$$
m(\dot{z}_T - \dot{z}_0) = -\int_0^T mg\, dt + \int_0^T (F_F^z + F_H^z)\, dt
$$

变量含义:
- $m$: robot mass (KAIST HOUND 45 kg)
- $g$: 9.81 m/s²
- $F_{F,H}^{x,z}$: front/hind leg 的水平/垂直 GRF
- $[0, T]$: 一个 gait cycle,典型 ~0.3 s
- $\dot{x}_T, \dot{z}_T$: cycle 末端速度; $\dot{x}_0, \dot{z}_0$: cycle 起始速度

把 GRF 参数化为 Bézier curve,只要 Bézier control points 满足上面两个 integral 约束,$\dot{x}, \dot{z}$ 自动周期化。剩下只需要优化 $z, \theta, \dot{\theta}$ 的周期性。

### 3.3 Newton 法找 periodic orbit

cost function:

$$
f(\mathbf{q}_{\text{opt},0}) := (z_T - z_0)^2 + (\theta_T - \theta_0)^2 + \lambda_1 (\dot{\theta}_T - \dot{\theta}_0)^2 + \lambda_2 \theta_{\max}^2
$$

- $\mathbf{q}_{\text{opt},0} := (z_0, \theta_0, \dot{\theta}_0)$: 待优化的初始条件 (3 维)
- 前三项: 周期性约束
- 第四项 $\lambda_2 \theta_{\max}^2$: 正则化过大的 pitch 振荡 (只对 bound 启用,$\lambda_2 = 1.0$;trot 用 $\lambda_2 = 0$,因为 trot pitch 本来就小)
- $\lambda_1 = 0.5$ for both gaits

Newton update:

$$
\mathbf{q}_{\text{opt},0}^{\text{new}} := \mathbf{q}_{\text{opt},0} - \alpha H^{-1} J^\top
$$

- $J = \nabla f$: 3 维 gradient
- $H = \nabla^2 f$: 3×3 Hessian
- $\alpha = 1$ (full step)

Bézier 曲线的好处:**Jacobian 和 Hessian 都能解析求出**,不用数值差分。平均一次 solve 21.582 µs。

### 3.4 Dataset 规模

- 每 gait 90,000 trajectories,共 180,000
- 5,560,286 time steps @ 100 Hz = 15.5 小时 motion
- 速度范围 -2 ~ 7 m/s,body height 0.45–0.5 m,stance 0.13–0.2 s,swing 0.15–0.21 s
- **总生成时间 8 分钟**(包括 optimization + preprocessing + saving)

这个 throughput 非常关键 —— TO 生成比 animal mocap 便宜几个数量级,而且 torque 是 physically grounded 的(不像 mocap 只有 kinematics)。

---

## 4. Stage 1 详解:TVAE Representation Learning

### 4.1 架构

```
Input: 3 frames proprioception s_t, s_{t+1}, s_{t+2} ∈ R^{69×3}
       (body lin/ang vel, height, foot height, contact, gravity,
        joint pos/vel, history)
       ↓
   Tokenize + Transformer encoder
   (hidden 64, 8 heads, 2 layers, GeLU)
       ↓
   Token 0 → μ ∈ R^16
   Token 1 → log σ² ∈ R^16
       ↓
   Reparameterization: z = μ + σ·ε,  ε ~ N(0, I)
       ↓
   ┌─────────────────┬────────────────────┐
   Trot decoder      Bound decoder         (separate heads)
   ↓                 ↓
   τ_t ∈ R^12        τ_t ∈ R^12
```

latent dim 16 是有意选小的 —— RL 阶段 policy 要实时 sample + decode,16 维足够覆盖 trot/bound 在 -2~7 m/s 的 motion manifold。

### 4.2 Encoder loss

$$
\mathcal{L}_{\text{TVAE ENC}} := \mathcal{L}_{\text{recon}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}
$$

- $\mathcal{L}_{\text{recon}} := \sum_{t=1}^{3} \| s_t - \hat{s}_t \|_2^2$: 3 帧 state reconstruction MSE
- $\mathcal{L}_{\text{KL}} := D_{\text{KL}}(q(z|s) \| \mathcal{N}(0, I))$: posterior 跟 unit Gaussian 的 KL
- $\lambda_{\text{KL}} = 0.1$

### 4.3 Decoder loss

$$
\mathcal{L}_{\text{TVAE DEC}} := \| \tau_t - \hat{\tau}_t \|_2^2
$$

- $\tau_t$: TO 给的 ground-truth torque
- $\hat{\tau}_t$: decoder 输出

注意 encoder 是 shared (trot 和 bound 共用),decoder 是 gait-specific。这样 latent space 是 unified 的,但 torque mapping 是 gait-conditioned。这跟 [Petrovich et al. 2021, ActCond TVAE](https://arxiv.org/abs/2104.05370) 的 human motion synthesis 思路一致,只是这里 action 是 torque 而非 pose。

### 4.4 Dataset size ablation (Fig. S6)

| Dataset % | Final loss |
|-----------|------------|
| 1% | 高 |
| 10% | 中 |
| 50% | 低 |
| 100% | 最低 |

说明 15.5h 数据不是浪费,encoder 确实需要 dense coverage 才能 capture speed × gait 的完整 manifold。

---

## 5. Stage 2 详解:APT-RL Policy

### 5.1 Action 分解 (这是全文最 clever 的设计)

Actor network (MLP 512→256→128, ReLU) 输出 29 维:

| Component | Dim | 含义 |
|-----------|-----|------|
| $a_{\text{latent}}$ | 16 | 输入 pretrained decoder 的 latent code |
| $a_{\text{aux}}$ | 12 | auxiliary joint-space offset (走 PD controller) |
| $\text{logit}_{\text{gait}}$ | 1 | gait selection logit,过 sigmoid → $a_{\text{gait}} \in [0,1]$ |

Gait selection 规则:
$$
a_{\text{gait}} < 0.5 \Rightarrow \text{trot decoder}, \quad a_{\text{gait}} \geq 0.5 \Rightarrow \text{bound decoder}
$$

频率分配:
- Gait selection: **2 Hz** (每 0.5 s 决策一次,锁定 0.5 s)
- Latent + auxiliary action: **100 Hz**

这种低频高层 + 高频低层的混合,跟动物 CNS (gait pattern generator ~Hz) + spinal reflex (~100 Hz) 的分层同构,也跟 [CPG-RL, Bellegarda 2022](https://arxiv.org/abs/2207.10181) 的 philosophy 类似。

### 5.2 Torque 计算公式 (Fig. 2B-ii-a)

$$
\tau_{\text{input}} = \tau_{\text{dec}} + k_p \cdot (q_{\text{default}} - q_t + a_{\text{scale}} \cdot a_{\text{aux}}) - k_d \cdot \dot{q}_t
$$

$$
\tau_{\text{dec}} := \mathbb{1}[a_{\text{gait}} < 0.5] \cdot \text{Decoder}_{\text{trot}}(a_{\text{latent}}) + \mathbb{1}[a_{\text{gait}} \geq 0.5] \cdot \text{Decoder}_{\text{bound}}(a_{\text{latent}})
$$

变量:
- $\tau_{\text{dec}}$: pretrained decoder 输出的 feedforward torque (12 维)
- $k_p = 80$: PD 比例增益
- $k_d = 2$: PD 微分增益
- $q_{\text{default}}$: nominal joint angle (standing pose)
- $q_t, \dot{q}_t$: 当前 joint position / velocity
- $a_{\text{scale}} = 0.2$: auxiliary action 的 scale

直觉解读:
- $\tau_{\text{dec}}$ 是 **feedforward 主力**,carry 了 gait-cyclic dynamics (contact timing、CoM trajectory、angular momentum),从 TO 学来
- $a_{\text{aux}}$ 是 **修正项**,相当于在 PD target 上加 offset,让 policy 能处理 TO dataset 没覆盖的 3D motion (lateral、yaw、unseen obstacle)
- 这个结构跟 [Residual Policy Learning, Silver 2018](https://arxiv.org/abs/1812.06298) 不同 —— residual 是 fixed base + additive correction,这里 latent 和 aux 是 **jointly trained** 的

### 5.3 Latent action KL regularization

$$
\mathcal{L}_{\text{latent KL}} := D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_{i=1}^{d} \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
$$

- $d = 16$: latent 维度
- 系数 $2.5 \times 10^{-6}$ (非常小)

注意:**没有 encoder regularization** (prior work [CoMic, Hasenclever 2020](https://arxiv.org/abs/2010.11443) 用了)。原因是 encoder 只在 2D flat terrain 训,如果强制 policy latent 分布跟 encoder posterior 对齐,policy 就被锁在 flat terrain 上了。

### 5.4 Reward 结构 (Table S3)

四类 reward:

**Task**:
$$
r_{\text{lv}} = \exp\left( -\left\| \frac{v_{b,xy}^{\text{cmd}} - \bar{v}_{b,xy}}{v_{b,x}^{\text{cmd}}} \right\|_2^2 / 0.25 \right), \quad v_{b,x}^{\text{cmd}} > 0.1
$$
速度归一化 tracking,weight 1.5

**Regularization**: latent smoothness, aux smoothness, aux scale, joint torque, joint accel, termination penalty

**Gait related**: 
- $r_{\text{gd}} = c_{\text{gd}}(\bar{\psi}_{1:T_\psi^l} - 0.5)^2$, $c_{\text{gd}} = 1 - \text{epoch}/50000$: 鼓励 gait diversity (不要太频繁切换)
- $r_{\text{gc}} = c_{\text{gc}} \sum (\psi_{t+1} - \psi_t)^2$: gait change penalty

**Style**: foot clearance, foot brushing, foot airtime, nominal config, foot collision, foot contact velocity, roll limit, body shaking, simultaneous contact

PPO 超参: horizon 50, lr 3e-4, KL threshold 0.008, discount 0.99, entropy 0.001, clip 0.2, batch 204800, mini-batch 40960

---

## 6. Stage 3 详解:Perception Distillation

### 6.1 为什么需要 LiDAR + Depth 双模态

| Modality | Range | Update | Vertical FoV | 用途 |
|----------|-------|--------|--------------|------|
| RealSense D435 depth | 0.3–3 m | 25 Hz | ~50° | dense local geometry |
| Hokuyo UST-30LX 2D LiDAR | 0.6–5 m | 40 Hz | 平面扫描但装在头上,有 3× 垂直视野 | sparse long-range |

4 m/s × 2 Hz decision = 4 m lookahead,depth 不够。LiDAR 给 0.6–5 m 范围、45 bins × 0.1 m resolution。

### 6.2 Student architecture (Table S5)

```
Depth image (87×58×10 history) → CNN [32, 64 channels, kernel 5,3]
                                    → MaxPool → 128-dim embedding
                                    ↓
2D LiDAR (45-dim) + proprio + action history → MLP [256, 128, 64]
                                    ↓
                              Concatenate
                                    ↓
                              GRU (1 layer, 512 hidden, tanh)
                                    ↓
                              32-dim exteroceptive latent
```

Training: DAgger + BPTT, MSE loss to match teacher's 32-dim output。Teacher 直接吃 255-dim raw heightmap + LiDAR。

### 6.3 硬件 vibration absorber (Fig. S1)

这是工程上很容易被低估的点:**机械旋转式 LiDAR 在高速 bounding 时承受 >10g impact,经常 shutdown**。

解决方案:
- 3D-printed PLA damping structure,装在 LiDAR 跟 head 之间
- Spiral springs 抑制 rotational moment
- 设计目标: 10 mm 位移内 low translational stiffness + high rotational stiffness
- FEA 验证

这跟 [Burden et al. 2024, Sci Rob "Why animals can outrun robots"](https://www.science.org/doi/10.1126/scirobotics.adi9754) 指出的 sensor-body coupling 问题直接对应 —— 动物有 muscle+ tendon 做 passive damping,robot 需要专门 mechanical design。

---

## 7. 实验结果深度解析

### 7.1 Gait effectiveness across terrains (Fig. 4)

5 个 gait decoder (trot/bound/pace/gallop/pronk) 在 4 类地形上对比:
- **Success Rate, Velocity Tracking, 1/COT**

| Terrain | Best gait | 解释 |
|---------|-----------|------|
| Rough & discrete | trot | 稳定、COT 低 |
| Stairs (low cmd) | trot | step-by-step 精确 |
| Stairs (high cmd) | bound | 飞越式 |
| High steps | bound | 大高差需要纵向 impulse |
| Hurdles | bound/trot 混合 | 看高度 |
| Stepping stones | trot | 精确 foot placement |

结论: trot + bound 是最 complementary 的组合,其他 gait 在特定 case 偶尔好但 variance 大。

### 7.2 Latent space 结构 (Fig. 5A, B)

PCA 可视化:
- Pretraining 数据: trot 和 bound 形成两个 cluster,部分 overlap
- RL 早期: policy latent 集中在 pretrained cluster 子集
- RL 后期: 扩展到 cluster 外围
- Real-world deployment: 在 familiar gait 时 overlap pretrained,在 unseen terrain (log jump) 时 explore novel latent region

t-SNE 显示 latent 按 gait type clustering,每个 gait 内还有 sub-cluster 对应不同 speed/obstacle。这说明 **2D flat TO 训出来的 latent space,3D 复杂环境 deploy 时仍然有结构**。

### 7.3 Auxiliary torque 的角色 (Fig. 5C-E)

三个 case study:

**Log jump (unseen scenario)**:
- 跑平地时: $\tau_{\text{dec}}$ 主导
- 起跳瞬间: $\tau_{\text{aux}}$ 显著增大,生成 TO dataset 里没有的 jumping torque
- 总 torque = dec + aux

**Leg breakage (OOD)**:
- 断腿后 policy 主动减小 HFE motor 的 $\tau_{\text{dec}}$,增大 $\tau_{\text{aux}}$
- 改变 leg trajectory 维持平衡

**In-place rotation (TO 数据没有的 motion)**:
- HAA (hip abduction-adduction) motor 在 TO 2D 数据里没有 torque → $\tau_{\text{dec}} = 0$
- 完全靠 $\tau_{\text{aux}}$ 实现 yaw rotation
- KFE motor 仍由 $\tau_{\text{dec}}$ 主导

直觉:**$\tau_{\text{dec}}$ 是 in-distribution feedforward,$\tau_{\text{aux}}$ 是 OOD adaptation**。这跟 LoRA / adapter 在 LLM 里的角色同构 —— frozen base model + small trainable adapter。

### 7.4 Gait selection behavior (Fig. 6)

Real-world:
- 2 m/s cmd: 低障碍 (0.175 m) → trot; 高障碍 (0.44 m) → bound
- 同一地形: 1 m/s → trot; >4 m/s → bound

Simulation gait fraction plot (Fig. 6B) 显示 bounding fraction 随 speed 和 difficulty 单调上升,但不同地形斜率不同:
- High step / hurdle / gap: 低速就开始 bound
- Discrete / rough: 高速才转 bound

Aggregate metrics (Fig. 6C-ii):

| Strategy | Best perf rate ↑ | Avg regret ↓ | Worst case ↑ |
|----------|------------------|--------------|--------------|
| **Auto** | **44.44%** | **4.99%** | **0.711** |
| Trot | 23.81% | 24.63% | 0.028 |
| Bound | 31.75% | 13.28% | 0.137 |

Auto 在三个指标上全面胜出,且 worst case 显著高 —— 说明 multi-gait 不只是 average 好,而是 robustness 好。

### 7.5 vs AMP / Vanilla RL (Fig. 7B)

- AMP: success rate 接近,但 **COT 更高**(因为 discriminator 把 policy 拉向 reference,限制了对 obstacle 的 adaptation)
- Vanilla RL: COT 接近 APT-RL,但 **success rate variance 大**,需要 terrain-specific reward tuning
- APT-RL: 一致地 low COT + high success,尽管 motion prior 只来自 flat ground

### 7.6 vs HRL + Residual (Fig. 7C, D)

- HRL 早期得益于 pretrained expert,但 unseen terrain 上卡住,residual policy 也救不回来
- APT-RL 全程探索 latent space (有 exploration bonus,逐渐 decay),最终达到更高 terrain difficulty + 更高 velocity tracking reward
- Sample efficiency: APT-RL 用更少 sample 达到更好 performance (Fig. 7D)

### 7.7 Random gait transition robustness (Fig. 7E)

2 Hz 随机切换 trot↔bound,probability 50%:
- APT-RL transition success rate 高
- HRL + residual 在高速 + 高 difficulty 时 success 大幅下降
- Base pitch trajectory (Fig. 7E-iii): APT-RL 平滑过渡到新 gait;HRL 失稳、stumbling

直觉:APT-RL 的 latent space 是 shared 的,trot/bound 之间天然 smooth interpolation;HRL 是两个独立 expert,切换时 distribution mismatch。

### 7.8 Sensor ablation (Fig. 8)

| Modality | 强项地形 | 弱项 |
|----------|----------|------|
| Depth only | hurdles, stepping stones (precise local geometry) | long-range 不足 |
| LiDAR only | low stairs, rough, gaps, high steps | 局部细节差 |
| **Both** | **全面最优** | — |

### 7.9 Feedforward torque ablation (Fig. S8)

把 $\tau_{\text{dec}}$ scale 从 0% → 100%:

| $\tau_{\text{dec}}$ scale | Success rate | Vel tracking |
|---------------------------|--------------|--------------|
| 0% | **2.9%** | 0.751 |
| 100% | **94.6%** | 1.384 |

**结论: $\tau_{\text{dec}}$ 是 dominant component,aux PD 单独完全不行**。这证明 TO-pretrained torque prior 不可替代。

### 7.10 PD-target 等价性 (Fig. S9)

把 $\tau_{\text{input}} = \tau_{\text{dec}} + k_p(q_{\text{default}} - q_t + a_{\text{scale}} a_{\text{aux}}) - k_d \dot{q}_t$ 重写为标准 PD:
$$
\tau_{\text{input}} = k_p(q_{\text{default}} + q_{\text{ref}} - q_t) - k_d \dot{q}_t, \quad q_{\text{ref}} = a_{\text{scale}} a_{\text{aux}} + \tau_{\text{dec}} / k_p
$$

两种形式 success 94.3% vs 94.3%,vel tracking 1.383 vs 1.382,1/CoT 2.394 vs 2.384 —— 几乎完全一样。所以 torque formulation vs PD-target formulation 是 representational choice,不是 capability 差异。

---

## 8. Froude Number 分析 — 跟动物对比

$$
Fr = \frac{v^2}{g \cdot L}
$$

- $v$: locomotion speed
- $g$: 9.81 m/s²
- $L$: leg length (KAIST HOUND ~0.5 m)

| 实验 | v (m/s) | Fr | 动物对应 |
|------|---------|-----|----------|
| 60 cm 高台阶跳过 | 4.25 | 3.85 | slow gallop |
| 三级楼梯跳下 (peak before impact) | 6.0 | 7.69 | fast gallop |

动物 gait transition 经验 [Hoyt & Taylor 1981](https://www.nature.com/articles/292239a0):
- walk: Fr < 0.5
- trot: 0.5 < Fr < 2.5
- gallop: Fr > 2.5

这台机器人达到了 **动物 fast gallop 的 Froude number 量级**,这是 perceptive quadruped 的新 benchmark。Reference: [Wimberly et al. 2021, Proc Royal Society B](https://royalsocietypublishing.org/doi/10.1098/rspb.2021.0937) 关于哺乳动物 gait evolution 的释放约束。

---

## 9. 跟 Karpathy 视角的 connection

几个我觉得你会感兴趣的 parallel:

### 9.1 "Pretrain + Fine-tune" 范式迁移

- LLM: scrape internet → pretrain LM → fine-tune on task
- APT-RL: TO 生成 synthetic motion-torque data → pretrain TVAE + decoder → RL fine-tune on downstream terrain

TO data 在这里扮演 "synthetic pretraining corpus" 的角色,跟用 code / math synthetic data pretrain LLM 思路一致 [Liu et al. 2024 DeepSeekMath](https://arxiv.org/abs/2402.03300).

### 9.2 Latent action codebook ≈ VQ-VAE codebook

TVAE 的 16-dim continuous latent 跟 [VQ-VAE, van den Oord 2017](https://arxiv.org/abs/1711.00937) 的 discrete codebook 类似,只不过这里是 continuous + reparameterization。RL policy 在 codebook 里 "predict next token" 式地选 action。

### 9.3 Auxiliary action ≈ Adapter / LoRA

- Base model (decoder): frozen,carry 主信号
- Adapter (aux action): small trainable,task-specific 修正
- 跟 [LoRA, Hu 2021](https://arxiv.org/abs/2106.09685) 的 low-rank adaptation 哲学一致

### 9.4 Mixture of Experts via gait selection logit

$a_{\text{gait}}$ 是一个 1-d router,选 trot expert 或 bound expert。这就是最简化的 [MoE, Shazeer 2017](https://arxiv.org/abs/1701.06538),只不过 expert 是 decoder 而非 full network。

### 9.5 Distillation = Knowledge Distillation

Teacher (heightmap + privileged) → Student (depth + LiDAR),MSE loss on latent。跟 [Hinton 2015 KD](https://arxiv.org/abs/1503.02531) 完全同构,只是 modality 是 perception 而非 logits。

### 9.6 Curriculum learning + exploration bonus

Terrain difficulty 10 级 curriculum + latent exploration bonus decay。这跟 LLM RLHF 里的 KL penalty decay、curriculum on task difficulty 思路一致 [OpenAI InstructGPT, Ouyang 2022](https://arxiv.org/abs/2203.02155).

---

## 10. Limitations & Future Work (作者自述)

1. **只 sagittal plane motion**: 没有 rapid turning、lateral walking。需要 3D TO dataset 扩展 decoder
2. **只 trot + bound 两个 gait**: pace, gallop, crawl 没集成
3. **Robot-agnostic 但需要 robot-specific TO**: 换机器人要重做 TO + retrain decoder (preliminary 在 ANYmal, Go1, HOUND bipedal mode 上 demo 了,Movie S4)
4. **没 high-level navigation + semantic understanding**: 现在是 velocity command 驱动,没 autonomous goal-directed exploration

---

## 11. 总结 intuition

APT-RL 的核心 contribution 可以浓缩成一句话:

**用 2D SRBD TO 大规模生成 (state, torque) paired data,pretrain 一个 TVAE encoder + gait-specific torque decoder,RL policy 在 16-dim latent space 选 action + 12-dim auxiliary PD offset + 1-dim gait router,通过 DAgger distillation 把 heightmap teacher 蒸馏成 depth+LiDAR student,最终在 KAIST HOUND 上零样本 sim-to-real 实现 6 m/s peak speed 的多 gait perceptive locomotion。**

跟 prior work 的核心区别:
- AMP 用 discriminator,这里用 **explicit decoder** (避免 mode collapse + 可解释)
- HRL 用 separate expert,这里用 **shared latent space** (天然 smooth transition)
- Residual policy 是 additive correction to fixed base,这里 **latent + aux jointly trained** (latent 主导,aux 修正)
- ASE / DreamWaQ 还要再训 tracking policy,这里 **decoder 直接出 torque,跳过 tracking**

Engineering takeaway: **TO 数据 + VAE latent codebook + hybrid torque/PD control + dual-modality perception + mechanical vibration isolation** —— 这五个 piece 加一起才能在 real wild 跑到 6 m/s,缺一个都跑不动。

---

## Key References

- [Paper (Sci Rob 2026)](https://www.science.org/doi/10.1126/scirobotics.adz7397)
- [Zenodo code/data](https://zenodo.org/records/20645964)
- [AMP, Peng 2021](https://arxiv.org/abs/2104.02180)
- [ANYmal Parkour, Hoeller 2024](https://www.science.org/doi/10.1126/scirobotics.adi7566)
- [Robot Parkour Learning, Zhuang 2023](https://arxiv.org/abs/2309.05665)
- [Learning perceptive locomotion, Miki 2022](https://www.science.org/doi/10.1126/scirobotics.abk2822)
- [Rapid locomotion, Margolis 2024](https://arxiv.org/abs/2208.07860)
- [DTC, Jenelten 2024](https://www.science.org/doi/10.1126/scirobotics.adh5401)
- [Why animals can outrun robots, Burden 2024](https://www.science.org/doi/10.1126/scirobotics.adi9754)
- [ActCond TVAE, Petrovich 2021](https://arxiv.org/abs/2104.05370)
- [VQ-VAE, van den Oord 2017](https://arxiv.org/abs/1711.00937)
- [LoRA, Hu 2021](https://arxiv.org/abs/2106.09685)
- [MoE, Shazeer 2017](https://arxiv.org/abs/1701.06538)
- [CPG-RL, Bellegarda 2022](https://arxiv.org/abs/2207.10181)
- [Residual Policy Learning, Silver 2018](https://arxiv.org/abs/1812.06298)
- [CoMic, Hasenclever 2020](https://arxiv.org/abs/2010.11443)
- [ASE, Peng 2022](https://arxiv.org/abs/2205.01906)
- [KAIST HOUND design, Shin 2022](https://arxiv.org/abs/2202.02814)
- [Hoyt & Taylor 1981 Nature](https://www.nature.com/articles/292239a0)
- [Wimberly 2021 Royal Society B](https://royalsocietypublishing.org/doi/10.1098/rspb.2021.0937)
- [Isaac Gym, Makoviychuk 2021](https://arxiv.org/abs/2108.10470)
- [PPO, Schulman 2017](https://arxiv.org/abs/1707.06347)
- [DAgger, Ross 2011](https://arxiv.org/abs/1011.0686)
