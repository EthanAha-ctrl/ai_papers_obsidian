---
source_pdf: Demonstrating Mujoco Playground.pdf
paper_sha256: d3b9ea24adefed948ac7280964e83bd3f66f2eda8c892e859213be4edc56e740
processed_at: '2026-08-18T05:09:04-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：MuJoCo Playground 在干什么

## 一句话

一群人把 MuJoCo 搬到 GPU 上（叫 MJX），又把一个 batch renderer（Madrona）缝进去，让你在单张显卡上 10 分钟训完一个 robot policy，然后直接 zero-shot 部署到真实 robot 上。`pip install playground` 就能跑。

项目页：<https://playground.mujoco.org>，代码 <https://github.com/google-deepmind/mujoco_playground>。

---

## 为什么这事重要——time-to-robot 才是 KPI

RL training 真正的瓶颈 **不在算法**，在 reward design 这个人机交互循环。你写个 reward，跑一遍，看 robot 抽风，改 reward，再跑一遍。如果每跑一次要 8 小时，你一天能试 2 次；如果 10 分钟，你一天能试 60 次。productivity 差 30 倍。这就是为什么把整个 physics + rendering + training 都塞进单 GPU 这件事有意义——它把 reward engineering 从"过夜跑实验"变成"喝杯咖啡回来就出结果"。

这也是为什么 Karpathy 你自己一直在推 single-file transformer 训练那种 vibe——快速 iteration loop 才是 research 的真正杠杆。

---

## 它的技术 stack 长什么样

```
   ┌────────── single GPU ──────────┐
   │  MJX physics (JAX, XLA)        │
   │       ↓ vmap 8192 envs        │
   │  Madrona batch renderer       │
   │       ↓ 64x64 RGB             │
   │  Brax PPO  (or RSL-RL)        │
   │       ↓                       │
   │  ONNX export                  │
   └──────────┼────────────────────┘
              ↓
         real robot (ros2_control, C++)
```

关键点：**全程不出 GPU**。物理 step、render、policy forward、policy backward 全在 device memory 里。传统 pipeline 是 physics 在 CPU、render 在 GPU、training 又跨 host，光是 D2H/H2D copy 就吃掉一大半时间。MJX 用 JAX 重写 MuJoCo，`vmap` 一行就把 step function 复制成 8192 个并行 env，这是 throughput 的核心来源。

---

## 三个最值得注意的工程 trick

### 1. Asymmetric actor-critic——sim-to-real 的免费午餐

idea 极简：训练时 critic 拿 privileged information（真实 contact force、真实 perturbation、unfiltered velocity），actor 只拿 deploy 时也能拿到的 proprioception。critic 用 oracle 视角算 advantage 更准，actor 学一个 deploy-time 可执行的 policy。

形式上 PPO loss 没变：

$$\mathcal{L}^{\text{PPO}} = \mathbb{E}_t \big[ \min(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \big]$$

- $r_t(\theta) = \pi_\theta(a_t|s_t^\pi) / \pi_{\theta_{\text{old}}}(a_t|s_t^\pi)$：importance ratio，新旧 policy 在同一 (state, action) 上的概率比
- $\hat{A}_t$：GAE advantage estimate
- $\epsilon$：clip range，典型 0.2
- 关键：$s_t^\pi$ 是 actor 看的 state，$s_t^V$ 是 critic 看的 state，**两者不同**

这相当于给训练时的 critic 一个作弊视角。直觉上，critic 知道"现在 robot 被推了 5N"，所以它能准确判断当前 action 好/坏；actor 看不到这个推力，但要学着从 proprioception 里推断出鲁棒的 action。部署时 critic 扔掉就行，actor 留下。

reference: Pinto et al. 2018 <https://arxiv.org/abs/1711.01246>

### 2. Torque control 而非 position control——缩小 sim-to-real gap 的物理直觉

Franka 那个 non-prehensile 任务，policy 直接输出 7D motor torque，bypass 掉 Franka FCI 的低层 PID。

为什么这能缩小 gap？position control 的链路是：policy 给 $q_{\text{des}}$ → Franka 内部 PD 算 torque → 电机执行。sim 和 real 的 PD 行为不可能完全一致，position tracking 误差会传到 torque 上，contact-rich 任务里这误差被 contact stiffness 放大成巨大 mismatch。

torque control 链路是：policy 给 $\tau$ → 电机执行。sim 与 real 之间只差一个 motor 模型，这个 motor model 相对好辨识（论文里就拆了 DYNAMIXEL 算了 reflected inertia）。

具体 control law（公式 1）：

$$\tau = k_p(q_{\text{des}} - q) - k_d \dot{q}$$

- $\tau$：motor torque，单位 N·m
- $k_p$：position gain，N·m/rad
- $k_d$：velocity gain，N·m·s/rad
- $q_{\text{des}}$：desired joint angle，rad
- $q$：actual joint angle，rad
- $\dot{q}$：joint angular velocity，rad/s

policy 在 locomotion 任务里输出 $q_{\text{des}}$（通过 PD 转成 torque），在 Franka non-prehensile 任务里直接输出 $\tau$。后者的 sim-to-real 效果好得多，paper 里 35 次试验 85.7% 成功率。

LEAP hand 那段更有意思，作者拆了 DYNAMIXEL 算 reflected inertia：

$$I_a = k_g^2 \cdot I_r, \quad I_r = \frac{1}{2} m_r r_r^2$$

- $I_a$：折算到输出轴的 armature inertia，kg·m²
- $k_g = 288.35$：gear ratio，减速比
- $I_r$：rotor 自身转动惯量
- $m_r = 2.0 \times 10^{-3}$ kg：rotor 质量
- $r_r = 4.12 \times 10^{-3}$ m：rotor 半径

gear ratio 平方后是 83000 倍。意思是 rotor 的微小转动惯量被放大 83000 倍到关节端，high-speed 运动时这个 reflected inertia 主导了关节动力学，sim 里忽略它必死。这就是为什么 OpenAI 当年 Dactyl 要做精细 actuator modeling，也是为什么这篇 paper 花一节讲 system identification。

### 3. Phase variable 显式 gait clock——阻止 policy 学出 shuffling

humanoid locomotion 训练里有个常见 failure mode：policy 发现"两只脚都贴地、身体往前 shuffle"能拿到 tracking reward，就陷在 shuffling gait 里出不来。

解法是给 observation 喂一个 phase 变量 $\phi \in [-\pi, \pi]$，对应每只脚的 gait cycle。observation 里加 $\cos\phi, \sin\phi$（连续平滑编码），reward 里加一项：

$$r_{\text{phase}} = k_{\text{phase}} \exp\left(-\frac{\|p_{f,z} - r_z(\phi)\|^2}{\sigma_{\text{phase}}}\right)$$

- $p_{f,z}$：actual foot height
- $r_z(\phi)$：desired foot height as function of phase，典型是 sinusoid（脚在 phase=0 时抬起，phase=π 时落地）
- $\sigma_{\text{phase}}$：temperature，控制 tolerance

直觉：给 policy 一个 explicit clock，比让它从 proprioception 隐式学 clock 容易得多。这相当于注入一个 motor prior，把 gait search space 从"所有可能步态"收窄到"按这个 phase 节奏抬腿"的子空间。H1、G1、Berkeley Humanoid 三个平台都用这个 trick。

---

## Madrona batch renderer 的 throughput 拆解

这是 paper 里最 insightful 的数据（Table X, XI，RTX 4090）：

Cartpole pixel-based training 每个 step 的耗时分解：
- physics: 2%
- rendering: 6%
- inference: 1%
- **policy update: 91%**

Franka pick-cube pixel-based：
- physics: 24%
- rendering: 18%
- inference: 1%
- **policy update: 57%**

直觉：**bottleneck 已经从 simulator 转到 network update**。这是 GPU RL infra 的一个 phase transition。5 年前 RL 慢是因为 CPU physics；现在 MJX + Madrona 把 physics/render 压到 <10%，CNN-based PPO update 反而成了大头。

implication：接下来 sample-efficient RL 算法（DreamerV3 <https://arxiv.org/abs/2301.04104>、TD-MPC2 <https://arxiv.org/abs/2403.15711>、EfficientZeroV2 <https://arxiv.org/abs/2403.00564>）在这套 stack 上应该 shine，因为它们用 latent imagination rollout 替代真实 env step，policy update 对 wallclock 的占比会进一步上升。

---

## Reward 设计的工程配方

Locomotion 的 reward 是十几项加权和（Table VI）。挑几个最有意思的讲：

**Tracking reward 用 exp kernel 而非 L2**：

$$r_v = k_v \exp\left(-\frac{\|\text{cmd}_{v,xy} - v_{xy}\|^2}{\sigma_v}\right)$$

- $\text{cmd}_{v,xy} \in \mathbb{R}^2$：joystick command
- $v_{xy} \in \mathbb{R}^2$：actual base velocity
- $\sigma_v$：temperature

为什么用 exp？exp 在 close-to-target 处 gradient 非零，policy 持续被 push 往更准走；远端 reward 不为零但衰减，policy 不会"放弃"。L2 penalty 在 target 处 gradient 已经零，policy 停下不动。这是 reward shaping 的经典 know-how。

**Stand-still penalty 的 dual-use trick**：

$$r_{\text{standstill}} = k_{\text{standstill}} \|\text{cmd}_{v,xy}\|$$

- 当 command 为零（原地站），这个项为零，不干扰 stand-still behavior
- 当 command 大（要走），这个 penalty 也大，把"啥也不干"的 reward 拉低，让 tracking reward 主导

一个 reward term 同时 cover "站住别动"和"跟 command 走"两种 mode，这是 reward engineering 里很 elegant 的设计。

**Energy 用 mechanical power**：

$$r_{\text{energy}} = k_{\text{energy}} \|\dot{q} \cdot \tau\|$$

- $\dot{q} \cdot \tau$：joint velocity 点乘 torque = mechanical power
- 累积就是 mechanical energy

这是物理量直接进 reward，sim-to-real 之间语义一致。

---

## 三个 sim-to-real 实验的 takeaway

| Task | Robot | 训练时间 | Success | Key trick |
|------|-------|---------|---------|-----------|
| In-hand cube reorient | LEAP hand | 30 min × 2×4090 | mean 7.1 rotations | system ID on DYNAMIXEL |
| Non-prehensile block push | Franka + Robotiq | 10 min × 16×A100 | 85.7% | 200Hz torque control |
| Pick cube from pixels | Franka | 10 min × 1×4090 | 12/12 | Madrona batch render |

三个任务一个比一个 trick 不同，但都遵循"domain randomization + curriculum + asymmetric actor-critic"三件套。这套配方现在基本是 sim-to-real 的标准操作。

---

## 局限性

paper 自己列了三条，我加注：

1. **JIT compile 1–3 min**：调 reward 反复 recompile 很烦。JAX 这套 static shape 的代价。
2. **Contact 计算 O(possible contacts) 而非 O(active contacts)**：JAX 要编译时确定 shape，所以 contact buffer 按"最大可能接触数"分配。场景里 8 只脚可能接触就按 8 分配，即使当前 step 只有 4 个真接触。Genesis (<https://arxiv.org/abs/2501.00612>) 用 Taichi dynamic shape 想绕开这个，但 sim-to-real demo 还少。
3. **Vision-based 还很初期**：只 2 个 demo，复杂任务没做。

我加一条 paper 没说的：**reward 仍手写**。Eureka (<https://arxiv.org/abs/2310.12931>) 那套 LLM 自动写 reward 是一个方向，但 Playground 当前所有 reward 还是人写的。

---

## 我的延伸直觉

**infrastructure-as-product 阶段**。这篇 paper 标志着 robot RL infra 进入这个阶段：发 SDK 配 paper，而不是发 paper 配 repo。这与 Hugging Face Transformers 路径相似——先做 infrastructure 占位，再让 community 往里填 env。Menagerie 做 asset 层，Playground 做 task 层。1-2 年内大概率出现 robot env 的 Hub 和 leaderboard。

**MJX vs Genesis vs Warp 的 next round**。MJX 成熟度最高、sim-to-real demo 最多；Genesis 刚出，Taichi 路线，dynamic shape 更灵活但生态薄；Warp (<https://github.com/NVIDIA/warp>) 是 NVIDIA 的 Python GPU sim framework，与 Isaac Lab 协同。三方目前都在抢"开源 GPU physics for robot RL"这个位。MJX 的护城河是 MuJoCo 的 legacy 兼容性和 DeepMind 的 backing。

**torque-level RL 与 force-aware policy 的趋势**。Franka 那个实验暗示一个趋势：直接学 motor torque 比 学 joint position 更适合 contact-rich 任务。surgical robot (Haiderbhai et al. <https://arxiv.org/abs/2405.19223>)、peg insertion、in-hand manipulation 都在往这个方向走。深层原因是 contact-rich 任务的 sim-to-real gap 主要在 force 量级，不在 position 量级。

**phase variable = motor primitive 的最简版本**。cos/sin phase encoding 这种 explicit gait clock 可以看作最朴素的 motor prior。下一步可能 是 learnable phase（如 trajectory generator + DMP），或 attention over learned gait library。HumanoidBench (<https://arxiv.org/abs/2406.10200>)、HumanPlus (<https://arxiv.org/abs/2406.10454>)、iDP3 (<https://arxiv.org/abs/2409.12941>) 都在这条线上。

**playground 与 Isaac Lab 的分野**。Isaac Lab 走 PhysX + Omniverse path-traced rendering，视觉质量强但 PhysX GPU 闭源。Playground 走 MJX + Madrona CUDA ray tracer，全开源，rendering 更"工程化"。对研究者可改性差别巨大——想加个新 contact model 在 MJX 改 JAX 一行，在 Isaac Lab 需要改 PhysX 源码（拿不到）。但 Isaac Lab 在 industrial deployment 上更成熟，因为 NVIDIA 全 stack 支持。两者定位不同，短期不会互相替代。

---

## 一句话总结

这是 robot RL 的 "single-file training" 时刻——10 分钟 train 完一个能上真机的 policy，全靠把 physics + render + RL 全塞进单 GPU 闭环。算法没创新，工程整合是主菜。time-to-robot 从小时级降到分钟级，reward engineering 终于能像写代码一样 iterate。接下来真正的 frontier 在 sample-efficient RL 算法（world model 路线）和自动 reward design（LLM 路线），simulator 本身已经不是瓶颈。

---

# MuJoCo Playground：一篇系统级 sim-to-real infrastructure 论文

你好 Karpathy。我先一句话总结这篇 paper 的位置，然后展开技术细节、公式、实验表、架构图与延伸联想。

这篇 paper 的本质，是把 GPU-accelerated physics (MJX)、GPU batch rendering (Madrona)、on-policy RL (Brax PPO/SAC) 这三个原本分散的子系统，**整合为一个单 GPU 上的 closed-loop pipeline**，并配上端到端 sim-to-real 的 demo 集。卖点并非单一新算法，而是 **time-to-robot**：在单 GPU 上 5–30 分钟内 train 完，零样本部署到 Go1、G1、Berkeley Humanoid、Booster T1、LEAP hand、Franka 等 6 个平台。

项目页面：<https://playground.mujoco.org>；代码：<https://github.com/google-deepmind/mujoco_playground>。

---

## 1. 这篇论文处在什么位置（research genealogy）

把它放进时间线理解直觉：

- **Isaac Gym / Isaac Lab** (Makoviychuk et al., 2021; Mittal et al., 2023, <https://arxiv.org/abs/2108.10470>, <https://arxiv.org/abs/2305.14487>): 最早将 PhysX GPU + RL 训练放在同一 GPU 上完成大规模 locomotion 学习；但 PhysX GPU 是 closed-source。
- **Brax** (Freeman et al., 2021, <https://github.com/google/brax>): JAX-based differentiable physics，开源，但与 MuJoCo 不完全兼容。
- **MJX** (MuJoCo XLA, 2024, <https://mujoco.readthedocs.io/en/latest/mjx.html>): DeepMind 把 MuJoCo 的物理 reimplementation 为 JAX，原生 XLA、原生 on-device、原生 vmap。
- **Madrona** (Shacklett et al., 2023, <https://arxiv.org/abs/2305.13731>): Stanford/FATAHalian group 的 GPU ECS + batch renderer，专门为 many-world simulation 设计。
- **MuJoCo Menagerie** (Zakka et al., 2022, <https://github.com/google-deepmind/mujoco_menagerie>): 一堆 robot XML assets。
- **DM Control Suite** (Tassa et al., 2018, <https://arxiv.org/abs/1801.00690>): 经典 RL benchmark，被 MJX port 进来当 entry-level 任务。
- **ManiSkill3** (Tao et al., 2024, <https://arxiv.org/abs/2410.00425>) 与 **Genesis** (2024, <https://arxiv.org/abs/2501.00612>): 平行 work，方向类似但实现路线不同。

Playground 把这一串东西 splice 到一起，构成「**asset (Menagerie) → physics (MJX) → rendering (Madrona) → RL (Brax/RSL-RL) → deployment (ONNX + ROS2 control)**」这条完整链路。

---

## 2. 整体架构图（mental model）

```
┌─────────────────────────────────────────────────────────────┐
│                   Single GPU (A100 / 4090 / H100)            │
│                                                              │
│   ┌────────────────┐    ┌──────────────────┐                 │
│   │   MJX Physics  │←──│  Madrona Batch   │                 │
│   │  (JAX/XLA)     │    │   Renderer       │                 │
│   │  vmap(N envs)  │    │  (CUDA raytrc or │                 │
│   │  step()        │    │   Vulkan raster) │                 │
│   └──────┬─────────┘    └────────┬─────────┘                 │
│          │                       │                          │
│          ▼                       ▼                          │
│      s_t  (proprio)        o_t (RGB 64x64x3)                 │
│          │                       │                          │
│          └───────────┬───────────┘                          │
│                      ▼                                      │
│            ┌──────────────────┐                              │
│            │  Brax PPO / SAC  │ ← asymmetric actor-critic    │
│            │  (JAX) or        │   policy: state^π            │
│            │  RSL-RL (torch)  │   value: privileged_state^V  │
│            └────────┬─────────┘                              │
│                     │                                       │
│                     ▼                                       │
│              π_θ  (policy net)                              │
│                     │                                       │
│                     ▼                                       │
│              export ONNX ───┐                               │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Real robot             │
                  │  (ros2_control C++      │
                  │   on FCI / EtherCAT /   │
                  │   Unitree SDK)          │
                  │  policy @ 15–50 Hz      │
                  │  estimator @ 500–2000 Hz│
                  └─────────────────────────┘
```

直觉上这是 RL 版的 "single-device closed loop"：每一层都不需要跨 host 通信，state/observation/policy 全程留在 GPU 显存里，只通过 ONNX 在最后一步物理离开 GPU。

---

## 3. MJX 物理引擎深度

### 3.1 JAX + XLA 的代价与收益

MJX 把 MuJoCo 的 C 代码逐函数重写为 JAX，编译为 XLA。关键 implications：

1. **Static shape requirement**: JAX 必须在编译时知道所有 tensor shape。这导致 contact 计算的成本不是 O(active contacts) 而是 O(possible contacts in scene)。例如，Go1 的 8 只脚可能产生接触，编译时需要预留对应 buffer，即使大部分 step 实际只有 4 个接触点。这就是 paper 中反复提到的 limitation。
2. **JIT compile 时间**: 1–3 min。在 reward tuning 的反复迭代场景下，这是真实痛点。典型曲线见 Fig. 12 reward curve 前面的 flat 段。
3. **vmap 天然友好**: 同一个 step function 加 `jax.vmap` 就能跑 8192–32768 个并行环境，这就是 throughput 的来源。

### 3.2 PD controller 公式

Locomotion 与 manipulation 的核心底层 control law：

$$\tau = k_p (q_{\text{des}} - q) - k_d \dot{q} \tag{1}$$

变量解释：
- $\tau$：motor torque（N·m）
- $k_p$：proportional gain（位置增益，单位 N·m/rad）
- $k_d$：derivative gain（速度增益，单位 N·m·s/rad）
- $q_{\text{des}}$：desired joint position（目标关节角，rad）
- $q$：current joint position（实际关节角，rad）
- $\dot{q}$：current joint velocity（关节速度，rad/s）

$q_{\text{des}}$ 从 policy action 推出，分两种 mapping：

**Joystick 任务**（绝对位置）：

$$q_{\text{des},t} = q_{\text{default}} + k_a a_t \tag{2}$$

- $q_{\text{default}}$：nominal standing pose 的关节角
- $k_a$：action scale，把 $a_t \in [-1, 1]^{\text{dim}}$ 映射到关节偏移量
- $a_t$：policy 在时刻 $t$ 输出的 action vector

**其他任务**（相对位置）：

$$q_{\text{des},t} = q_{\text{des},t-1} + k_a a_t \tag{3}$$

这种 relative 形式对 handstand / fall recovery 等需要在大范围关节空间漫游的任务更友好。

### 3.3 LEAP hand 的 DYNAMIXEL 系统辨识

这是 paper 里很有意思的一段。原 LEAP hand sim-to-real 没做精细 system identification，作者拆解 DYNAMIXEL 转子计算 armature inertia：

$$I_a = k_g^2 \cdot I_r \tag{4}$$

- $I_a$：reflected armature inertia（折算到输出轴后的等效转动惯量，kg·m²）
- $k_g = 288.35$：gear ratio（减速比，来自 datasheet）
- $I_r$：rotor 自身转动惯量，按 uniform disk 假设：

$$I_r = \frac{1}{2} m_r r_r^2 \tag{5}$$

- $m_r = 2.0 \times 10^{-3}$ kg：rotor 质量
- $r_r = 4.12 \times 10^{-3}$ m：rotor 半径
- 代入得 $I_r = 1.7 \times 10^{-8}$ kg·m²

直觉：gear ratio 是 288 倍，平方后接近 83000，armature inertia 把 rotor 的微转动惯量放大到关节级别可观的值。这就是为什么 sim 里忽略 armature 会让 sim-to-real 失败——高速运动时 reflected inertia 的 gyroscopic 效应显著。

DYNAMIXEL 实际 control law：

$$i = k_p^m (\theta_{\text{des}}^m - \theta^m) - k_d^m \dot{\theta}^m \tag{6}$$

- $i$：motor current command（电流指令，A）
- $k_p^m, k_d^m$：电机端 PD 增益（与关节端 $k_p, k_d$ 单位不同）
- $\theta_{\text{des}}^m, \theta^m$：motor 端目标角与实际角（rad）

通过 $\tau = k_t i$ 把 current 换算为 torque。但 DYNAMIXEL 不做真正的 current feedback，所以训练时需要 randomize $k_p, k_d$ 来 cover 这个 mismatch。

---

## 4. Madrona Batch Renderer 深度

### 4.1 ECS 架构

Madrona 是 GPU ECS（Entity-Component-System）：<https://arxiv.org/abs/2305.13731>。每个 entity 是一组 component（位置、mesh、material），system 是处理 component 的 kernel。这让 many-world simulation 天然并行：每个 world 是一个 ECS context，所有 context 共享同一份 GPU kernel 但读写各自 memory。

两个 backend：
- **CUDA software ray tracer**（本文实验使用）：支持 lighting、shadows、textures、geometry materials。在 NVIDIA GPU 上跑。
- **Vulkan rasterizer**：fallback，更广适用。

### 4.2 与 MJX 集成

通过 JAX 的 low-level primitive (`jax.core.Primitive`, `jax.prim` 体系) 把 Madrona 的 init/render C++ 函数暴露给 JAX。这样 renderer 调用可以与 `jit`、`vmap` 串联，物理 step 和 render step 共享同一个 GPU memory pool，避免 H2D / D2H copy。

**关键 throughput 数据**（Table X，RTX 4090）：

| 环境              | Env step FPS | with Pixels FPS | with Inference FPS | with Training FPS |
|-------------------|--------------|-----------------|--------------------|-------------------|
| CartpoleBalance   | 1.37×10⁶     | 4.03×10⁵        | 3.41×10⁵           | 3.13×10⁴          |
| PandaPickCubeCart | 6.40×10⁴     | 3.69×10⁴        | 3.60×10⁴           | 1.56×10⁴          |

### 4.3 Bottleneck 转移（Table XI）

这是论文最有 insight 的一个表，按每 step 耗时拆分：

| 环境              | Physics | Rendering | Inference | Policy Update |
|-------------------|---------|-----------|-----------|----------------|
| CartpoleBalance   | 0.02    | 0.06      | 0.01      | **0.91**       |
| PandaPickCubeCart | 0.24    | 0.18      | 0.01      | **0.57**       |

直觉：传统 on-policy RL 的瓶颈在 data collection（physics+render），现在 Madrona + MJX 把这部分压到 <10%（Cartpole）或 ~43%（Franka），主要时间花在 **PPO update 的 CNN forward/backward**。这个趋势在很多 GPU RL paper 里出现，比如 Isaac Lab 的 vision pipeline 也是这样。意味着接下来优化方向应该转向 RL algorithm 的 batch efficiency（如 GRPO 路线、importance sampling variance reduction），不再是 simulator 速度。

---

## 5. Asymmetric Actor-Critic：sim-to-real 的关键 trick

paper 用 Pinto et al. 2018 (<https://arxiv.org/abs/1711.01246>) 的 asymmetric actor-critic：

- **Policy network** $π_\theta(s_t^\pi)$：只能看到 deployment 时也能拿到的 observation（proprioception + noisy exteroception）。
- **Value network** $V_\phi(s_t^V)$：训练时拿到 privileged information——真实 robot pose、contact forces、external perturbation、unfiltered velocity 等。

直觉：这相当于给 critic 一个 oracle 视角来 bootstrap return，让 critic 估计更准；actor 仍然学一个 deploy-time 可执行的 policy。形式上，PPO loss 不变：

$$\mathcal{L}^{\text{PPO}} = \mathbb{E}_t \Big[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \Big] + c_v \| V_\phi(s_t^V) - \hat{G}_t \|^2 + c_H \mathcal{H}(\pi_\theta)$$

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t^\pi)}{\pi_{\theta_{\text{old}}}(a_t|s_t^\pi)}$：importance ratio
- $\hat{A}_t$：GAE advantage estimate
- $\hat{G}_t$：return estimate
- $\mathcal{H}$：entropy bonus

关键差异在 $s_t^\pi \neq s_t^V$：value 用更富 observation。Table XV 中看到 `policy_obs_key: "state"`, `value_obs_key: "privileged_state"`。

架构上（Table XVI 等）：policy 与 value 都是 3 层 MLP (512, 256, 128)，Swish 激活。vision 任务换成 CNN + 2 个 256 dense 层。

---

## 6. Locomotion 的 reward 设计详解（Table VI）

Locomotion 的 reward 是十几项加权和：

$$r_{\text{total}} = \sum_i w_i r_i \tag{7}$$

然后 clip 到非负。逐项解释：

### 6.1 Tracking 类（要 maximize）

$$r_v = k_v \exp\left(-\frac{\| \text{cmd}_{v,xy} - v_{xy} \|^2}{\sigma_v}\right) \tag{8}$$

- $\text{cmd}_{v,xy} \in \mathbb{R}^2$：joystick 给的 forward/lateral velocity command
- $v_{xy} \in \mathbb{R}^2$：实际 base 在 xy 平面的线速度
- $\sigma_v$：temperature，决定 tracking tolerance
- $k_v$：weight

exp-kernel 而非 L2 是关键：exp 形式在 close-to-target 处梯度不为零，远端 gradient 衰减但 reward 始终为正，配合下面 penalty 项才能让 policy 平衡 tracking 与 smoothness。

$$r_\omega = k_\omega \exp\left(-\frac{\| \text{cmd}_{\omega,z} - \omega_z \|^2}{\sigma_\omega}\right) \tag{9}$$

- $\text{cmd}_{\omega,z}$：yaw rate command
- $\omega_z$：实际 base yaw rate

### 6.2 Gait shaping

$$r_{\text{air}} = \text{clip}\big((T_{\text{air}} - T_{\min}) \cdot C_{\text{contact}}, 0, T_{\max} - T_{\min}\big) \tag{10}$$

- $T_{\text{air}}$：foot airtime（脚在空中持续时间）
- $T_{\min}, T_{\max}$：min/max desired airtime
- $C_{\text{contact}} \in \{0, 1\}$：foot 刚触地时为 1，否则 0
- 直觉：鼓励"每只脚有足够 air time 但有节奏触地"。

$$r_{\text{phase}} = k_{\text{phase}} \exp\left(-\frac{\| p_{f,z} - r_z(\phi) \|^2}{\sigma_{\text{phase}}}\right) \tag{11}$$

- $\phi \in [-\pi, \pi]$：phase variable per foot
- $r_z(\phi)$：desired foot height as function of phase（典型为 sinusoid）
- $p_{f,z}$：actual foot height
- 直觉：强制每只脚按 phase 走出"抬腿-落下"曲线，引入显式 gait clock，避免 policy 学出 shuffling 步态。

### 6.3 Stability 与 smoothness penalty

$$r_{\text{ori}} = k_{\text{ori}} \| \phi_{\text{body},xy} \|^2 \tag{12}$$

base 在 roll/pitch 方向的倾角平方，惩罚躯干晃动。

$$r_\tau = k_\tau \| \tau \|^2, \quad r_q = k_q \| q - q_{\text{nominal}} \|^2, \quad r_{\text{rate}} = k_{\text{rate}} \| a_t - a_{t-1} \|^2$$

torque、关节角偏离 nominal、action 平滑度，三项标准 penalty。

$$r_{\text{energy}} = k_{\text{energy}} \lVert \dot{q} \cdot \tau \rVert \tag{13}$$

- $\dot{q} \cdot \tau$：mechanical power（关节速度点乘力矩）
- 直觉：energy = power × time，按 step 累积即总机械能。在 quadruped 上是关键，续航靠它。

### 6.4 Termination 与 stand-still

$$r_{\text{term}} = k_{\text{term}} \cdot \text{done}, \quad r_{\text{standstill}} = k_{\text{standstill}} \| \text{cmd}_{v,xy} \|$$

- $\text{done}$：episode 是否 terminate（机器人翻倒等）
- 直觉：stand-still penalty 在 command 为零时把 reward 拉到 0，鼓励原地不动；command 大时这个 penalty 自动小，让 tracking 项 dominate。这是让一个 reward function 同时 cover "站着别动" 与 "跟 command 走" 的经典手法。

### 6.5 Finetuning curriculum

三阶段：
1. Flat ground, narrow command range {1.5, 0.8, 1.2} m/s 与 rad/s，100M steps
2. Flat ground, wider yaw range {1.5, 0.8, 2π}，50M steps
3. Rough terrain（Perlin noise heightfield），100M steps

直觉：先学基本 gait，再学大转角，最后学 rough terrain 鲁棒性。rough terrain 用 Perlin noise 生成，因为 Perlin 是 smooth random field，能模拟草地/碎石地连续起伏。

---

## 7. Manipulation 三个 sim-to-real 实验深度

### 7.1 LEAP Hand In-Hand Reorientation

任务：把 7cm cube 在 palm 中 reorient 到随机目标 SE(3) pose，每个目标距上一次 ≥ 90°，达到 0.4 rad tolerance 切换新目标。

Hardware setup（参考 DROP <https://arxiv.org/abs/2409.14562> 与 DexPoint）：
- LEAP hand palm 朝下倾斜 20°，避免 cube 卡在 finger-palm gap
- 单 RealSense D415 在 workspace 上方
- pose estimator from DexTreeM <https://arxiv.org/abs/2211.10864>，15 Hz
- policy @ 20 Hz（USB-Dynamixel jitter 限制）

Domain randomization：friction、cube mass、joint offsets、motor friction、reflected inertia、PD gains、link mass、sensor noise；2 cm positional noise + 0.1 rad rotational noise on cube pose。

两阶段训练：
- Phase 1 (200M steps)：no random pose injection, no torque limit
- Phase 2 (100M steps fine-tune)：以 0.1 概率 inject 随机 pose（模拟 occlusion "freak-out"），加 torque limit 匹配硬件

**结果（Table I，10 trials）**：mean = 7.1 consecutive rotations，median = 3.5。失败模式：cube 卡在 finger-palm gap（最频繁），index/thumb interlocking（hardware flex 导致）。

### 7.2 Franka-Robotiq Non-Prehensile Block Reorientation

任务：把 yoga block 从随机 initial pose 推到 fixed goal pose（位置误差 < 3 cm，旋转误差 < 10°）。

关键 trick：**200 Hz 直接 torque control**（不是 position control）。policy 输出 7D motor torque，bypass Franka FCI 的低层 PID。

公式上，policy 输出 $\tau_t \in \mathbb{R}^7$ 直接进 FCI，重力补偿由 Franka 内置完成：

$$\tau_{\text{cmd}} = \tau_{\text{gravity}} + \tau_{\text{policy}} \tag{14}$$

Domain randomization 与 curriculum：
- 随机化 initial joint position、block pose、action delay（1–3 step）、observation delay（6–12 step）
- curriculum 渐增 block displacement 与 rotation range

**结果（Table II，35 trials）**：success 85.7% ± 12.2%，median position error 1.95 cm，rotation error 1.72°。

训练时间：16×A100，10 min。

直觉：torque control 比 position control 的 sim-to-real gap 更小，因为 contact-rich 任务里 position control 会"硬刚"——sim 里能精确跟踪但 real 里执行器达不到，导致大 mismatch；torque control 让 robot "compliant"，policy 学到的是 force-level interaction，sim-to-real 之间不会突然 stiff。

### 7.3 Franka PickCube from Pixels

任务：从 64×64×3 RGB 图像 grasping 2×2×3 cm cube 并抬升 10 cm。

简化：end-effector 限制在 Y-Z plane（X 固定），action 是 (Δy, Δz, jaw_open/close)，3 维。policy @ 15 Hz，RTX 3090 inference。

Domain randomization for vision gap：
- 光照、阴影、相机 pose、物体颜色随机化
- brightness post-processing
- gripper action delay 0–5 step（模拟机械延迟，最大 250 ms）

reward 用 Dexpbt 风格 <https://arxiv.org/abs/2306.11847>：sparse + dense，dense 项取 running max：

$$r_t = \text{clip}\Big(\sum_i r_{t,i} - \max(r_1, r_2, \dots, r_{t-1}), 0\Big) \tag{15}$$

- $r_{t,i}$：第 $i$ 个 dense reward term 在时刻 $t$ 的值
- $\max(r_1, \dots, r_{t-1})$：到目前为止 reward 历史最大值
- 直觉：reward 只能"上升"，每步的 marginal progress 才计入。这让 dense reward 不再 dominate，sparse term（成功 lift）更显著。

**结果**：12/12 trials 100% success。

训练时间：single RTX 4090，10 min。

---

## 8. 训练 throughput 数据汇总

### 8.1 DM Control Suite（Table IV，PPO，A100）

代表环境 throughput：

| Env               | PPO steps/s    | SAC steps/s |
|-------------------|----------------|-------------|
| CartpoleBalance   | 718,626        | 30,891      |
| CheetahRun        | 435,162        | 18,819      |
| HumanoidRun       | 91,617         | 5,886       |
| WalkerRun         | 141,581        | 6,069       |

直觉：PPO 比 SAC 快约 20×，因为 PPO 是 on-policy、batch-friendly；SAC 维护 replay buffer，每步需要采样 + 多次 gradient step，访存 pattern 不友好。

### 8.2 Locomotion（Table VII）

| Env                              | PPO steps/s |
|----------------------------------|-------------|
| BarkourJoystick                  | 385,920     |
| BerkeleyHumanoidJoystickFlat     | 120,145     |
| G1Joystick                       | 106,093     |
| Go1JoystickFlatTerrain           | 417,451     |
| Go1JoystickRoughTerrain         | 291,060     |
| Go1Getup                         | 204,416     |
| SpotFlatTerrainJoystick          | 404,931     |

直觉：biped 比 quadruped 慢约 3×，因为 DoF 多、contact 复杂（G1 有 29 DoF）。

### 8.3 Manipulation（Table IX）

| Env                  | PPO steps/s |
|----------------------|-------------|
| AlohaSinglePegInsert | 121,119     |
| LeapCubeReorient     | 76,354      |
| PandaOpenCabinet     | 136,007     |
| PandaPickCube        | 140,386     |
| PandaPickCubeCart    | 38,015      |
| PandaRobotiqPushCube | 487,341     |

PandaPickCubeCartesian 慢是因为 200 Hz torque control 下 sub-step 多。

### 8.4 GPU topology scaling（Fig. 6, LeapCubeReorient）

| Topology   | Wallclock |
|------------|-----------|
| 1×4090     | ~2080 s   |
| 8×H100     | ~670 s    |

3× 加速。但作者说 "we leave tuning hyperparams per topology as future"——更大的 GPU 配置理论上应该用更多 envs，但当前 hyperparam 没调。

---

## 9. Vision-based 训练的瓶颈再分析

公式化每 step 总耗时：

$$t_4 = t_{\text{train}} + t_{\text{infer}} + t_{\text{render}} + t_{\text{env\_step}} \tag{16}$$

对应 Table X 最后一列。从 Table XI：

**Cartpole**：$t_{\text{train}}/t_4 = 0.91$，policy update 几乎独占。
**Franka**：$t_{\text{train}}/t_4 = 0.57$，physics 与 render 占 24% + 18%。

为什么 Cartpole 比 Franka 更"training bound"？因为 Cartpole physics 极轻，但 CNN policy 一样大（64×64 输入决定 conv 层规模与任务复杂度无关）。这给了 insight：**vision-based RL 的瓶颈从 simulator 转到 network**，接下来该优化的是 sample-efficient RL 算法（如 world model DreamerV3 <https://arxiv.org/abs/2301.04104>、TD-MPC2 <https://arxiv.org/abs/2403.15711>），而不是再榨 simulator 速度。

延伸：这个 finding 与 ManiSkill3 论文 (<https://arxiv.org/abs/2410.00425>) Fig. 5 的 bottleneck 分析一致——当 FPS 上到一定水平后，PPO update 反而成了 wallclock 大头。

---

## 10. Limitations 评注

paper 列了三条，我加注：

1. **JIT compile 1–3 min**：对反复调 reward 的 user 很烦。Workaround 是用 `jax.jit` 的 warmup + persistent cache。
2. **Contact 不随 active contact scale**：JAX 静态 shape。这是 MJX 当前最大架构债。Genesis (<https://arxiv.org/abs/2501.00612>) 用 Taichi 的 dynamic shape 想绕过这个问题，但目前 sim-to-real demo 还少。Warp (<https://github.com/NVIDIA/warp>) 也是同类思路。
3. **Vision-based training 还在早期**：只有 2 个 demo 环境，复杂任务（cabinet opening, bi-arm peg insertion）的 vision 版还没做。

我加一条 paper 没提：**task-reward engineering 仍手工**。论文用 Eureka <https://arxiv.org/abs/2310.12931> 半自动化是方向，但目前 Playground 所有 reward 仍是人写的。

---

## 11. 我的延伸联想与直觉

### 11.1 与 Isaac Lab 的定位差异

Isaac Lab 走 PhysX + Omniverse 渲染 + RTX 路线，rendering quality 强（path-traced），但 PhysX GPU 闭源。Playground 走 MJX + Madrona 路线，全开源，rendering 更"工程化"（CUDA ray tracer + 简化 shading）。对研究者的可改性差别巨大：你想加一个新 contact model 在 Playground 改 JAX 一行；在 Isaac Lab 需要改 PhysX 源码（拿不到）。

### 11.2 与 DreamerV3 / TD-MPC2 / EfficientZeroV2 的契合

paper 提到 vision-based policy 训练 sample efficiency 差，常用 teacher-student distillation <https://arxiv.org/abs/2107.01246>。但 Playground 把 bottleneck 转到 policy update，**这恰恰是 world model 类算法（DreamerV3、TD-MPC2）应该 shine 的场景**——它们用 imagination rollout 在 latent space 训练，policy update 对 wallclock 的依赖更小。Playground + DreamerV3 是一个明显的 next step。

### 11.3 与 LEAP hand / Berkeley Humanoid 的硬件 democratization

LEAP hand <$5000，Berkeley Humanoid 也是 low-cost。Playground 让 single GPU + 5 min training + cheap hardware 形成 democratization 三角。这让我联想到当年 ImageNet + GPU 让 deep learning 普及。下一个里程碑可能是：**社区贡献 robot env 的"app store"**。Menagerie 已经在做 asset 层面，Playground 在做 task 层面。

### 11.4 ONNX export 的部署抽象

paper 用 ONNX Runtime <https://onnxruntime.ai/> 在 robot 上跑 policy，这样训练用 JAX/PyTorch，部署与框架解耦。典型 inference 50 Hz 在边缘 CPU 上也能跑（小 MLP）。这条 pipeline 与 NVIDIA Isaac ROS 的 vision pipeline 不同——后者绑 TensorRT，前者更通用。

### 11.5 Asymmetric actor-critic 与 RLHF 的隐含联系

注意到 asymmetric actor-critic 的结构与 RLHF 中 actor 用 GAE（critic 提供 baseline）相似。事实上 privileged information for value 的思想在 model-based RL 中也常见——value 用真实 dynamics 计算 advantage，policy 用 learned approximation。这与 DreamerV3 的 critic-with-true-state 思路高度相似。

### 11.6 Phase variable 的 gait clock 与 H1 H1 gait tracking

paper 对 humanoid 用 phase variable $\phi \in [-\pi, \pi]$，obs 用 $\cos\phi, \sin\phi$，reward 用 $r_z(\phi)$ 作为 foot height target。这与 H1 InplaceGaitTracking 任务里 explicit phase signal 一致。直觉上，**给 policy 一个 explicit clock 比让它从 proprioception 隐式学 clock 容易得多**——后者 RL 经常 collapse 到 shuffling gait。这个 phase 变量也是 "intrinsic motor primitive" 的最简版本。

延伸到 humanoid locomotion 的近期工作：HumanoidBench <https://arxiv.org/abs/2406.10200>、HumanPlus <https://arxiv.org/abs/2406.10454>、iDP3 <https://arxiv.org/abs/2409.12941>，phase variable 都还在用。

### 11.7 Torque control 与 compliance 的物理直觉

Franka 那个 non-prehensile 任务，作者明确说 "learning to control torques rather than joint positions, the agent develops smooth, compliant behavior that transfers effectively to hardware"。这背后的物理：position control 的输入是 $q_{\text{des}}$，PD 误差大时输出 torque 爆炸；torque control 直接塑形 contact force，sim-to-real 之间 force 范围是物理量（N），可以匹配得很好。

这与 surgical robot RL <https://arxiv.org/abs/2404.16001> 的 force-aware policy 思路同源。

---

## 12. 复现 checklist（如果你想跑）

1. `pip install playground`（需要 CUDA 12.x）
2. 选环境：`from playground import envs; env = envs.locomotion.get_env_handler("Go1JoystickFlatTerrain")`
3. Train：`from brax.training.agents.ppo import train`，跑 PPO
4. Export ONNX：`playground.utils.to_onnx`
5. Deploy：在 robot 上跑 `ros2_control` + ONNX runtime node（paper 仓库提供 C++ wrapper）

Reference repo：
- Playground: <https://github.com/google-deepmind/mujoco_playground>
- MJX docs: <https://mujoco.readthedocs.io/en/latest/mjx.html>
- Brax: <https://github.com/google/brax>
- RSL-RL: <https://github.com/leggedrobotics/rsl_rl>
- Menagerie: <https://github.com/google-deepmind/mujoco_menagerie>
- Madrona: <https://github.com/madrona-entity-system/madrona>
- ONNX Runtime: <https://onnxruntime.ai/>
- Berkeley Humanoid paper: <https://arxiv.org/abs/2407.21781>
- Barkour: <https://arxiv.org/abs/2305.14654>
- DM Control Suite: <https://arxiv.org/abs/1801.00690>
- Isaac Gym: <https://arxiv.org/abs/2108.10470>
- Isaac Lab / Orbit: <https://arxiv.org/abs/2305.14487>
- ManiSkill3: <https://arxiv.org/abs/2410.00425>
- Genesis: <https://arxiv.org/abs/2501.00612>
- Eureka: <https://arxiv.org/abs/2310.12931>
- DreamerV3: <https://arxiv.org/abs/2301.04104>
- TD-MPC2: <https://arxiv.org/abs/2403.15711>
- EfficientZero V2: <https://arxiv.org/abs/2403.00564>
- Dexpbt: <https://arxiv.org/abs/2306.11847>
- DROP (LEAP hand): <https://arxiv.org/abs/2409.14562>
- Robot Parkour: <https://arxiv.org/abs/2309.05665>
- Extreme Parkour: <https://arxiv.org/abs/2304.13653>

---

## 13. 关键 takeaway

1. **Time-to-robot** 是真正的 KPI。RL 的 reward design 是 iterative 的人机过程，训练 5 min vs 5 hr 在 productivity 上差 100×，比 GPU 利用率高 10% 重要得多。
2. **Bottleneck 转移到 policy update**。vision-based RL 的下一步优化在 algorithm 端，不在 simulator。
3. **Asymmetric actor-critic + torque control + domain randomization + curriculum** 是当前 sim-to-real 的工程配方四件套，paper 把它们打包成 single-pipeline。
4. **MJX 的 static shape 限制** 是当前最大的工程债，Genesis/Warp/Taichi 路线可能在下一轮胜出，但 MJX 当前成熟度最高。
5. **开源 + pip install + Colab-friendly** 是 democratization 的关键，比任何算法 novelty 都更重要——让 1000 个 lab 能跑比让 1 个 lab 跑得快重要。

直觉上，这篇 paper 标志着 robot RL infra 进入"infrastructure-as-product"阶段：不再是发一篇 algorithm paper 配一个 repo，而是发一个 SDK 配一个 paper。这与 Hugging Face Transformers 的轨迹相似。期待 1-2 年内出现 robot env 的 Hub、benchmark board、leaderboard。
