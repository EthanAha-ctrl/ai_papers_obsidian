---
source_pdf: MuJoCo Playground.pdf
paper_sha256: 1227d94174639184a24c907fa1dda4b0639edc6fabdcc7ec4cfe17198c30afe3
processed_at: '2026-08-05T20:57:32-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 MuJoCo Playground

Andrej, 咱们坐下来聊, 把这篇 paper 拆成"为什么他们要这么干"和"到底干了啥"两块说。

---

## 这篇 paper 到底在干嘛

一句话: **以前你想用 RL 训一个机器人在真实世界走路, 要么花一堆钱买 closed-source 的 NVIDIA Isaac Gym, 要么自己搭几个月环境; 现在他们做了一个 open-source 的, 一行 `pip install`, 一张 RTX 4090, 几分钟训完, 直接 zero-shot 部署到真机。**

就这么个事。听起来简单, 但里面每一个环节都很难, 他们把整条 pipeline 串通了。

项目地址: https://mujocoplayground.github.io

---

## 为什么以前做 sim-to-real 很痛苦

想象你 2022 年想训一个 quadruped 走路。流程是这样的:

1. 你得装 Isaac Gym (closed-source, 要申请), 或者用 MuJoCo CPU 版 (慢)
2. 物理仿真在 CPU 上跑, 几千个并行 envs 要开多进程, IPC 烦死人
3. 训一个 Go1 走路, 多台机器跑好几天
4. 你想改一个 reward term, 又要好几天
5. 想做 vision-based? 渲染慢到你想哭, 只能先 state 训再 distill 给 vision, pipeline 翻倍复杂
6. 训完了部署到真机, 发现 sim 跟 real 差太多, 又得好几天调 domain randomization

核心痛点: **reward design 是个手艺活, 你得反复试错——"我让它 track velocity, 它就给我趴下不动; 我加个 stand penalty, 它给我疯狂蹦"。每次试错要好几天, 你的迭代速度就被卡死了。**

Playground 的核心 insight 就一句: **训练速度从"几天"压到"几分钟"之后, 整个研发范式变了, 你可以像调 hyperparameter 一样调 reward, 像跑 notebook 一样跑 sim-to-real。**

---

## 他们怎么做到这么快的

三件事:

### 1. MJX: 把 MuJoCo 用 JAX 重写

MuJoCo 本来是 C 写的, 跑在 CPU 上。DeepMind 的人用 JAX 重写了一遍, 叫 MJX (MuJoCo XLA), 能跑在 GPU 上。

直觉: JAX 让你写一个 `step` 函数, 然后 `vmap` 一下, 8192 个并行 envs 就在一行代码里搞定了, 而且整个 step 跟 policy forward 可以 fuse 成一个 GPU kernel, 没有 host-device 来回。

代价: JAX 要求 compile time 知道所有 tensor shape, 但物理仿真里 contact 数量是 runtime 动态变的。MJX 没办法, 只能按"可能 contact 数量"分配固定 buffer。对于 LEAP hand 这种 contact-rich 任务, 大量 compute 是浪费的。

类比: 就像你写 CUDA kernel, 不能动态分配 shared memory, 只能按 worst case 分配, 简单 task 浪费, 复杂 task 撑爆。

文档: https://mujoco.readthedocs.io/en/stable/mjx.html

### 2. Madrona: GPU 上批量渲染

如果要做 vision-based RL, 你要给每个 env 渲染一张图。传统做法是调 OpenGL, 一个 env 一个 thread, 慢且数据回传 GPU 有 PCIe bottleneck。

Madrona 是 Stanford 那帮人 (Shacklett, Fatahalian 等) 写的 GPU ECS 框架, 里面有个 CUDA batch ray tracer, **一次给 N 个 env 渲染 N 张 64x64 图, 全程在 GPU 上不出去**。

Paper: https://dl.acm.org/doi/10.1145/3592450
Code: https://github.com/shacklettbp/madrona

结果 (Figure 7): Cartpole with 64x64 pixels 能跑到 **~403k env steps/sec** on RTX 4090。这意味着你可以直接用 pixel 做 RL, 不需要 teacher-student distillation 那套繁琐 pipeline。

### 3. 把这三层 fuse 在一起

关键: physics (MJX) + rendering (Madrona) + RL (Brax PPO) 全在 JAX 里, 全在 GPU 上, 全能 `jit` 成一个 graph。这就是为什么 Cartpole state-based 能跑到 1.37M steps/sec, pixel-based 403k steps/sec。

---

## 训练快了之后, 真正改变了什么

Paper Section I 有一句话很关键, 我翻译一下:

> RL 最善于找到获取 reward 的策略, 但 resulting behavior 经常以你想不到的方式 irregular。所以 reward design 和 training 必须反复 iterate, time-to-robot 变得 critical。

这句话翻译成人话: **你给机器人一个 reward, 它就会找各种 exploit 方式最大化这个 reward, 你看着它各种诡异行为, 改 reward, 再训, 再看, 再改...这个 loop 的速度决定了你研发的节奏。**

如果一次 loop 要一天, 你一天能试 5 个 reward 变体; 如果一次 loop 要 5 分钟, 你一上午能试 50 个。这是质的差别。

实测数据 (Section IV):
- Go1 joystick: 5 分钟训完 (2x RTX 4090)
- Berkeley Humanoid: 15 分钟
- G1 / T1: 30 分钟
- LEAP hand in-hand reorientation: 30 分钟
- Franka non-prehensile: 10 分钟 (16x A100)
- Franka pick-cube from pixels: 10 分钟 (单 4090)

他们 8 周内部署了 6 个不同机器人平台的 sim-to-real policy。这个速度在以前是不可想象的。

---

## sim-to-real 是怎么搞成的

核心招数就一招: **domain randomization**。直觉是——

别试图精确建模真实世界。你在仿真里把所有可能的"真实"都 enumerate 一遍: friction 从 0.3 到 1.5 随机, mass ±20%, sensor noise 各种加, lighting 各种变... 然后让 policy 在所有这些 variations 上都能 work。部署到真机时, 真机只是这些 variations 里的一个 sample, policy 自然 robust。

具体清单 (Section B.22, C.42, C.6):

**Locomotion**:
- Sensor noise: gravity, base velocity, joint pos/vel 全加噪声
- Dynamic: link COM, reflected inertia, joint friction, calibration offset
- Task: ground friction, payload mass

**LEAP hand**:
- Friction (cube-finger), cube mass, joint offsets, motor friction
- PD gains 也随机化 (因为 DYNAMIXEL servo 实际是 current control 不是 torque control, 跟仿真不完全一致)
- 2cm position noise + 0.1 rad rotation noise on cube pose
- 0.1 概率 "pose injection" 模拟 occlusion 导致的 pose estimator 突然跳变

**Vision**:
- Lighting, shadows, camera pose, object color, brightness 全随机
- 0-250ms gripper action delay

这招是 Tobin et al. (IROS 2017) 提出的, paper 在这把它推到极致。Ref: https://arxiv.org/abs/1703.06907

---

## PD 控制层这个细节挺有意思

Policy 不直接输出 torque, 而是输出 desired joint position, 然后用一个 PD 控制器转换:

$$\tau = k_p(q_{des} - q) - k_d \dot{q}$$

变量讲清楚:
- $\tau$: 最终施加到 joint 上的 torque
- $k_p$: proportional gain, 位置误差的放大系数
- $k_d$: derivative gain, 速度阻尼
- $q_{des}$: policy 输出的 desired joint position
- $q$: actual joint position (传感器读的)
- $\dot{q}$: actual joint velocity

$q_{des}$ 怎么从 policy action 来的? 两种方式:

**Joystick 任务** (绝对式):
$$q_{des,t} = q_{default} + k_a \cdot a_t$$

$a_t$ 是 policy 输出 ∈ [-1, 1], $k_a$ 是 action scale (比如 0.25 rad), $q_{default}$ 是站立默认姿势。直觉: policy 决定"偏离默认姿势多少"。

**其他任务** (相对式):
$$q_{des,t} = q_{des,t-1} + k_a \cdot a_t$$

直觉: policy 决定"相对上一时刻变化多少"。

LEAP hand 那里有个坑值得注意 (Section C.42): DYNAMIXEL servo 实际控制的是 motor current, 不是 torque, 控制律是:

$$i = k_p^m(\theta_{des}^m - \theta^m) - k_d^m \dot{\theta}^m$$

而且单位、量纲跟仿真里不一样。Paper 假设 $\tau = k_t \cdot i$ (torque constant), 通过 motor spec sheet 推映射。同时把 $k_p, k_d$ 也加 domain randomization 来弥补这个 sim-real 不一致。

教训: **仿真里的 low-level controller 跟真机的 low-level controller 永远不可能完全一致, 与其精调不如 randomize。**

---

## Reward 设计: locomotion 的全套 term

Section B.23 的 Table VI 列了 12 个 reward term, 我挑几个关键的讲直觉:

### Tracking reward (胡萝卜)

$$r_v = k_v \cdot \exp\left(-\frac{\|cmd_{v,xy} - v_{xy}\|^2}{\sigma_v}\right)$$

变量:
- $cmd_{v,xy}$: joystick 给的期望 base 线速度 (二维, body frame)
- $v_{xy}$: actual base 线速度
- $\sigma_v$: 温度参数, 控制衰减速度
- $k_v$: scaling

直觉: 误差为 0 时 reward = $k_v$ (最大), 误差大时指数衰减。比线性 `1 - error` 平滑, gradient 也好。Gaussian shape 是 RL reward shaping 的常用套路。

同理 $r_\omega$ 跟踪角速度。

### Gait shaping (步态塑造)

**Feet Airtime**:
$$r_{air} = \text{clip}((T_{air} - T_{min}) \cdot C_{contact}, 0, T_{max} - T_{min})$$

$T_{air}$ 是脚在空中时长, $T_{min}/T_{max}$ 是期望 airtime 区间, $C_{contact}$ 是 contact indicator。鼓励脚抬起来一段时间, 防止拖步。

**Feet Phase** (humanoid 关键):
$$r_{phase} = k_{phase} \cdot \exp\left(-\frac{\|p_{f,z} - r_z(\phi)\|^2}{\sigma_{phase}}\right)$$

$\phi$ 是 phase variable ∈ $[-\pi, \pi]$ 循环, $r_z(\phi)$ 是 desired foot height 作为 phase 函数。把"两脚交替"这个 prior 直接 inject 进 reward, 显著降低 humanoid 学步难度。

Observation 里用 $\cos(\phi), \sin(\phi)$ 编码 phase, 这个 trick 来自 Shao et al. RA-L 2022, 现在 humanoid RL 几乎标配。Ref: https://arxiv.org/abs/2109.13362

### Penalties (大棒)

- $r_\tau = k_\tau \|\tau\|^2$: torque penalty, 防止暴力踩
- $r_{rate} = k_{rate} \|a_t - a_{t-1}\|^2$: action smoothness, 防止抖
- $r_{energy} = k_{energy} \|\dot{q} \cdot \tau\|$: mechanical power, 物理能耗
- $r_{lin\_z} = k_{lin\_z} \|v_z\|^2$: 别上下蹦
- $r_{ang\_xy}$: 别乱 roll/pitch

### Asymmetric actor-critic

Policy 网络只看 proprioceptive + noisy exteroceptive (跟真机传感器一致的观测); Value 网络看 privileged info (真实 contact force, 真实 perturbation force)。

直觉: 部署时机器人没法直接读 contact force, 但训练时 critic 知道这些能让 value 估计更准, PPO variance 更低。Pinto et al. RSS 2018 的经典 trick。Ref: https://arxiv.org/abs/1610.05696

---

## 实验数据: 哪些漂亮, 哪些还有坑

### LEAP Hand In-Hand Reorientation (Table I)

10 trials, 连续旋转次数: 3, 27, 8, 2, 15, 3, 4, 1, 3, 5
Median 3.5, Mean 7.1, std 极大。

这个分布很 telling: **policy 在 edge case 上 catastrophic fail**。失败模式:
1. Cube 卡在手指与手掌缝隙 → stall
2. Index 跟 thumb 物理互锁 (低成本硬件的 flex)

对比 OpenAI 2019 的 rubik's cube, 那个用了 384 台机器 + 64k envs + 数月训练 + 复杂的 Pose CNN + teacher-student。这里 30 分钟 2x 4090, 离那个水平还差不少, 但 open 和 reproducible 强太多了。

### Franka Non-Prehensile (Table II)

35 hardware trials:
- Success: **85.7% ± 12.2%** mean, 100% median
- Position error: 1.95 cm median
- Rotation error: 1.72° median

关键训练设置: **200 Hz direct torque control**。直接学 torque 比 学 position 难得多 (reward shaping 复杂、action space 高频), 但得到的 policy 自然 compliant, sim-to-real 不需要额外做 impedance 调节。

训练: 16x A100 上 10 分钟, 2B timesteps。

### Franka PickCube from Pixels

12 trials, 100% success。10 分钟单 RTX 4090 训完。这个 task 简化了: Y-Z plane only, 64x64 单 RGB, 3D action (Y, Z, jaw open/close)。

reward 用了 DexpBT (Handa et al. RSS 2023, https://arxiv.org/abs/2210.13702) 的 dense + sparse 组合:

$$r_t = \text{clip}\left(\sum_i r_{t,i} - \max(r_1, r_2, ..., r_{t-1}), 0\right)$$

直觉: dense reward 只奖励"超过历史最好"的进度, 防止 policy 刷 dense reward 而不真正接近 sparse goal。挺巧的 shaping。

---

## Bottleneck 转移: 这个发现挺重要的

Table X / XI 把 training loop 分解:

| 组件 | Cartpole 时间占比 | Franka 时间占比 |
|------|------|------|
| Physics step | 2% | 24% |
| Rendering | 6% | 18% |
| Policy inference | 1% | 1% |
| **Policy update (PPO backprop)** | **91%** | **57%** |

直觉: **一旦你把 physics + rendering 都跑到几十万 steps/sec, 数据生成不再是瓶颈, PPO 的 backprop (CNN forward+backward over 64x64x3) 成了新瓶颈。**

这跟 LLM pretraining 反过来了——LLM 里 forward + kernel launch 占主导, backward 已经被各种 fusion 优化得很狠; vision-RL 这里 CNN 还没被重视到那种程度。

实务含义: 想再加速 vision-RL, 应该花精力在 small efficient CNN (MobileNetV3-tiny 之类), 在 batched conv kernel 上做文章, 而不是再去榨 physics 的油。

---

## 跟其他 simulator 比 (Figure 23)

跟 IsaacLab 和 ManiSkill3 比 (数据来自 ManiSkill3 paper, https://arxiv.org/abs/2410.00425):

- State-based Cartpole: 三家差不多, 1-2M steps/sec
- 128x128 pixels: Playground 显著领先 (Madrona batch render 的优势)
- 256x256: 优势继续

Figure 24 把每 step 时间分解成 physics + rendering:
- MJX physics 比 ManiSkill3 快 (MJX 优化好)
- Madrona rendering 在所有分辨率下都比对手快
- 综合 2-3x throughput

当然 cross-simulator benchmark 永远有 "你 task setup 不一样" 的争议, 但量级感是对的。

---

## RL 库可插拔 (Figure 8)

Playground 主推 Brax (JAX PPO/SAC), 也支持 RSL-RL (PyTorch, ETH leggedrobotics, https://github.com/leggedrobotics/rsl_rl)。

两个 PPO 实现在 Go1 上 wallclock-to-reward 曲线几乎重合, 都 5 min 内收敛。好消息: 不用绑死 JAX 生态, 想用 PyTorch 的也能玩。

Brax 优势是 end-to-end XLA 编译, GPU 上没 host overhead; RSL-RL 优势是 PyTorch 调试方便、跟 Isaac Lab 兼容。

---

## 哪些地方还有坑

Paper Section VI 自报三条:
1. **JIT compile 慢** (1-3 min), 你想 hyperparam sweep 的话每次重 compile 1 分钟很烦
2. **Contact 不能按 active 缩放**, contact-rich 任务浪费 compute
3. **Vision-based 还 early**, 只 demo 了 toy task

我再加几条 paper 没明说的:

4. **MJX 跟 MuJoCo C 不完全 numerical identical**。MJX 是重写, solver iterations, line search 行为有微妙差异。Table III 里那些 "iterations=2, ls_iterations=4" 修改就是为了 match。

5. **No deformable objects, no fluids, no soft contacts**。Madrona terrain heightfield 还没做。surgical、cloth、fluid manipulation 完全不够。

6. **Vision sim-to-real 只 demo 了 PickCube 这种 toy task**。LEAP hand 用 pose estimator 不是 pixel, Franka non-prehensile 也是。真正 end-to-end pixel policy 在 contact-rich 任务上 sim-to-real 还没解。

7. **Reward design 仍然 hand-engineered**。Playground 没集成 Eureka 那种 LLM-assisted reward 设计。Reward shaping 12 个 term 都是手调的, 这才是 time-to-robot 的真正 bottleneck。

8. **Real-world deployment 栈不开放**。Paper 提到用 ros2-control + ONNX Runtime + 自定义 EtherCAT master, 但这些代码不在 paper 范围。Sim-to-real 真正难的一半是 deploy infra, 这块还得自己搭。

---

## 我会怎么用这个 framework

如果我现在要做一个 quadruped 项目:

1. **第一周**: `pip install playground`, 跑通 Go1 joystick, 在办公室地板上 reproduce sim-to-real
2. **第二周**: 改 reward 成我 task 的, 用 Eureka 或手写, 快速 iterate
3. **第三周**: 切到自己的 robot, 改 XML (从 Menagerie 复制模板), domain randomization 调到 hardware 实测 noise level
4. **第四周**: 集成 Madrona vision, 如果 task 需要

整个 stack 学习曲线比 Isaac Lab 陡峭程度低, 关键 JAX/PyTorch 之间能切换, 不被锁死。

---

## 跟你视角的几个联想

### "Software 2.0" 在 robotics 上的具体落地

这 paper 训出来的 policy 就是一堆 weights, 通过 ONNX 在 robot 上跑。整个 pipeline 里人类写的 "Software 1.0" 代码 (reward shaping, domain randomization, PD 控制律) 是 scaffolding, 最终运行时只有那 5MB 的 network。但 scaffolding 仍然 critical, 不能省。

### 跟 nanoGPT 的类比

nanoGPT 把 GPT2 训练压到 1 个 file 8 个 GPU 几小时; Playground 把 quadruped sim-to-real 压到 1 个 pip install 1 个 GPU 几分钟。这种 "reproducible, hackable baseline" 对一个领域的研究速度影响巨大, 远比那些 SOTA-but-closed 系统长期价值大。

### Scaling laws for robot learning

Paper 没讨论, 但下一个 natural step: 给定 fixed GPU 小时预算, 多 envs + 简单 task vs 少 envs + 复杂 task, 哪个 sample efficient? Vision-based 的 scaling law 跟 LLM 一样吗?

### World models vs model-free RL

Paper 全用 PPO/SAC model-free。但 DreamerV3 (https://arxiv.org/abs/2301.04104) / TD-MPC2 (https://arxiv.org/abs/2401.12987) 在 DM Control Suite 上已经更强, 跟 Playground 集成后会不会让 locomotion sim-to-real 更 sample efficient? 这是个开放问题。

---

## 一句话总结

**MuJoCo Playground 是 robotics sim-to-real 领域的 "nanoGPT moment": 开源、单 GPU、几分钟训练、能 deploy, 把过去需要整个团队几个月的工作压成一个 notebook。它没解决 reward design 这个真正的 bottleneck, 没解决 vision sim-to-real 在复杂任务上的 gap, 也没解决 MJX 在 contact-rich 场景下的 scalability, 但作为一个 community baseline, 它把整个领域的入门门槛和迭代速度拉到了一个新的水位。**

---

# MuJoCo Playground 深度讲解

Andrej, 这是一篇典型的"systems-as-research"论文, 跟你之前在 Tesla 做的很多工作有精神上的共鸣——把整个 stack 从 "研究原型" 拉到 "工程师一晚上能跑通" 的可用性门槛。下面我从底层一路刨到 sim-to-real, 顺手把直觉和坑都点出来。

---

## 1. 这篇 paper 到底在做什么定位

把它放在当下的 robotics sim 生态里看, 它本质是 Isaac Lab / Isaac Gym 阵营的一个**开源替代品 + 工程化整合**:

- **Physics**: 用 MJX (MuJoCo XLA, JAX 重写版) 替代 closed-source 的 PhysX GPU backend
- **Rendering**: 用 Madrona batch renderer 替代 Isaac Gym 的 hardware rasterizer
- **RL**: 同时支持 Brax (JAX) 和 RSL-RL (PyTorch) 两个 PPO 实现
- **Assets**: 直接复用 MuJoCo Menagerie

这里关键 trade-off: 用 JAX 换来了"全在 GPU 上、可 jit/vmap、可与 renderer 端到端 fused"的好处, 代价是 static shape 约束导致 contact 计算不能按 active contacts 缩放 (后面细讲)。

Paper 的 project page: https://mujocoplayground.github.io  
MJX 文档: https://mujoco.readthedocs.io/en/stable/mjx.html  
Menagerie: https://github.com/google-deepmind/mujoco_menagerie

---

## 2. 整体架构图 (mental model)

```
                ┌──────────────────────────────────────────────┐
                │            Single GPU Device                 │
                │                                              │
   random key ─►│  ┌─────────┐    ┌──────────┐   ┌─────────┐  │
                │  │  MJX    │───►│ Madrona  │──►│  Brax   │  │
                │  │ Physics│    │ Batch    │   │  PPO    │  │
                │  │ (JAX)  │    │ Renderer │   │  (JAX)  │  │
                │  └─────────┘    └──────────┘   └─────────┘  │
                │        ▲              ▲             │       │
                │        │              │             ▼       │
                │   domain rand    visual DR      policy update│
                │                                              │
                └──────────────────────────────────────────────┘
                                │
                                ▼  ONNX export
                        ┌───────────────┐
                        │  Real Robot   │
                        │  (ros2-control)│
                        └───────────────┘
```

整条 pipeline 一行 `pip install playground` 就装好, Colab notebook 可以一气呵成跑完。这个"装得上、跑得通、能 deploy"是它的核心 deliverable, 跟当年 fast.ai 风格一脉相承。

---

## 3. MJX 物理: 为什么用 JAX, 代价是什么

### 3.1 JAX 带来的好处

MJX 把 MuJoCo 的 C 引擎用 JAX 重写, 这样物理 step 变成一个 pure function:

$$s_{t+1} = \text{step}(s_t, a_t; \theta)$$

其中 $s_t$ 是 state (qpos, qvel, time, active contacts...), $a_t$ 是 action (joint torque 或 position target), $\theta$ 是 model params (masses, inertias, friction...)。一旦是 pure function, 你就可以:

- `vmap` over 8192 个并行 envs, 一行代码
- `jit` 整个 env step, 跟 policy forward 融成一个 kernel
- `grad` 通过 physics (可微仿真), 给 trajectory optimization 或 system ID 用

这就是 Table IV 里 CartpoleBalance 能跑到 **~718k PPO steps/sec** on A100 的根本原因——所有 envs 在同一 kernel launch 里 step, 没有 host↔device 来回。

### 3.2 代价: Static Shape 痛点

JAX 要求 compile time 知道所有 tensor shape。但物理里 contact 数量是 runtime dynamic 的——同一帧 8192 个 envs 里, 有的 env foot 在地上 (有 contact), 有的腾空 (无 contact)。

MuJoCo C 版本用稀疏结构处理, MJX 没法这么做, 于是采用最朴素的办法: **按"可能 contact 数量"分配固定大小 buffer**, 然后用 mask。

```
cost ∝ max_possible_contacts  (not active_contacts)
```

这就是 Section VI 里那句"computation time related to contacts does not scale like the number of active contacts in the scene, but like the number of possible contacts in the scene"的来源。对于 LEAP hand 这种 contact-rich 任务, 这意味着大量 wasted compute, 也是为什么 Section D.2 显示 Franka PickCube 物理一步 ~15μs, 比 Cartpole 的 0.73μs 慢 20x 的根本原因。

Warp (NVIDIA) 和 Taichi 用 dynamic shape/稀疏结构解决这个问题, 但失去 JAX 的 functional 可组合性。这是当下 GPU physics 的一个根本张力。

### 3.3 JIT 编译时间

Section VI 提到 1-3 分钟 compile。这其实是 JAX XLA 编译整条 env step + policy forward 的代价。Figure 12 的 reward vs wallclock 曲线开头那段平的 (前 100-200 秒) 就是 compilation, 不是在训。对长训任务无所谓, 但如果你想 hyperparam sweep, 每次重 compile 1 分钟是相当烦的——这是为什他们要强调"minutes to train"前面其实有个"minutes to compile"。

参考: https://github.com/google/brax

---

## 4. Madrona Batch Renderer: vision-based RL 的关键

### 4.1 为什么传统 vision-based RL 很慢

传统做法 (Isaac Gym 之前) 是: 每个并行 env 调 OpenGL 渲染一个 64x64 图, 渲染本身是单 thread 的, 多 envs 用多进程, 数据回传 GPU 又有 PCIe 来回。vision-RL 的 sample efficiency 又极差 (PPO 跑 pixel Cartpole 至少 10M+ steps), 综合下来训一天起步。

Teacher-student distillation 是 workaround: 先用 state 训一个 expert, 再让 vision student 模仿, 这样 student 不用 RL explore, 速度快。代价是 pipeline 复杂, student 上限受 expert 限制。

### 4.2 Madrona 的路子

Madrona (Shacklett et al., ACM TOG 2023, https://dl.acm.org/doi/10.1145/3592450) 是一个 GPU ECS (Entity-Component-System) 框架, 里面实现了两个 backend:

- **CUDA batch ray tracer** (这篇 paper 用的): 用 ray tracing 一次给 N 个 env 渲染 N 张图, 完全在 GPU 上, 不出 device
- **Vulkan rasterizer**: 更快但功能略弱

支持 lighting, shadows, textures, materials。Figure 4 给的几个渲染样张质量还行, 离 photoreal 远, 但对 sim-to-real 够用, 因为主要靠 domain randomization 抹平 gap。

### 4.3 与 JAX 的桥接

关键工程: Madrona 是 C++/CUDA 写的, 怎么跟 JAX 互操作? Paper 提到用 low-level JAX primitives (大概是通过 `jax.ffi` 或 custom primitive) 把 Madrona 的 init/render 函数包成 JAX 可调用的 op, 这样 `jit` 和 `vmap` 仍然 work。

这块 paper 写得很省, 细节得看 Madrona repo: https://github.com/shacklettbp/madrona

### 4.4 Throughput 数据 (Figure 7)

- CartpoleBalance with 64x64 pixels: **~403k env steps/sec** (单 RTX 4090)
- PandaPickCubeCartesian with 64x64 pixels: **~37k env steps/sec**

Cartpole 物理便宜, 渲染占主导; Franka 物理贵 (关节多、contact 多), 物理占主导, 所以图像分辨率变化对 Franka FPS 影响不大——这点 paper 写得清楚。

### 4.5 Bottleneck 转移 (Table X / XI)

这是 paper 最 important 的一个 finding。把 training loop 分解:

| 组件 | Cartpole 时间占比 | Franka 时间占比 |
|------|------|------|
| Physics step | 2% | 24% |
| Rendering | 6% | 18% |
| Policy inference | 1% | 1% |
| **Policy update (PPO backprop)** | **91%** | **57%** |

**直觉**: 一旦你把 physics + rendering 都跑到 40 万 steps/sec, 数据生成不再是瓶颈, PPO 的 backprop (CNN forward+backward over 64x64x3) 成了新瓶颈。

这跟 LLM pretraining 反过来了——LLM 是 forward+kernel launch 占主导, backward 已经被各种 fusion 优化得很狠; vision-RL 这里 CNN 还没被重视到那种程度。

实务含义: 想再加速 vision-RL, 应该花精力在 small efficient CNN (MobileNetV3-tiny 之类), 在 batched conv kernel 上做文章, 而不是再去榨 physics 的油。

---

## 5. Reward 设计: locomotion 全套 term 拆解

Section B.23 的 Table VI 是 locomotion RL 的"标准配方", 值得一个个看:

### 5.1 Tracking rewards (the "carrot")

$$r_v = k_v \exp\left(-\|cmd_{v,xy} - v_{xy}\|^2 / \sigma_v\right)$$

变量: $cmd_{v,xy}$ 是 joystick 给的期望 base 线速度 (二维, body frame), $v_{xy}$ 是 actual base 线速度, $\sigma_v$ 是温度参数控制 reward 衰减快慢, $k_v$ 是 scaling。

直觉: 用 Gaussian-shaped reward, 误差为 0 时 reward = $k_v$, 误差大时指数衰减。比线性 `1 - error` 平滑, gradient 也好。

同理 $r_\omega$ 跟踪角速度。

### 5.2 Gait shaping

**Feet Airtime**:
$$r_{air} = \text{clip}((T_{air} - T_{min}) \cdot C_{contact}, 0, T_{max} - T_{min})$$

$T_{air}$ 是脚在空中时长, $T_{min}/T_{max}$ 是期望 airtime 区间, $C_{contact}$ 是 contact indicator。鼓励脚抬起来一段时间, 防止拖步。

**Feet Clearance**:
$$r_{clear} = k_{clear} \cdot \|p_{f,z} - p_{f,z}^{des}\|^2 \cdot \|v_{f,xy}\|^{0.5}$$

防止脚抬太高 (浪费能量), 也防止脚抬起时水平速度过大 (乱踢)。

**Feet Phase** (humanoid 用):
$$r_{phase} = k_{phase} \cdot \exp\left(-\|p_{f,z} - r_z(\phi)\|^2 / \sigma_{phase}\right)$$

$\phi$ 是 phase variable, 在 $[-\pi, \pi]$ 循环, $r_z(\phi)$ 是 desired foot height 作为 phase 函数。这把"两脚交替"这个 prior 直接 inject 进 reward, 显著降低 humanoid 学步难度。Observation 里也用 $\cos(\phi), \sin(\phi)$ 编码 phase。

这是 Shao et al. (RA-L 2022) 的老 trick, 但现在 humanoid RL 几乎标配。

### 5.3 Penalities (the "stick")

- $r_\tau = k_\tau \|\tau\|^2$: torque penalty, 防止暴力踩
- $r_q = k_q \|q - q_{nominal}\|^2$: joint position 别离 nominal 太远
- $r_{rate} = k_{rate} \|a_t - a_{t-1}\|^2$: action smoothness, 防止抖
- $r_{energy} = k_{energy} \|\dot{q} \cdot \tau\|$: mechanical power, 物理意义上的能耗
- $r_{lin\_z}, r_{ang\_xy}$: 别上下蹦, 别乱 roll/pitch
- $r_{termination}, r_{standstill}$: 失败和原地发呆都要罚

注意 $r_{standstill} = k_{standstill} \|cmd_{v,xy}\|$ 是个**只在 cmd=0 时生效的 anti-reward**——意思是"如果你命令是 0 而你还在动, 应该罚", 但这个式子写反了吧? 它乘的是 cmd 的范数, 那意味着 cmd 越大罚越多... 

我重读一下: 我猜应该是 $\|cmd_{v,xy}\| \cdot \|v_{xy}\|$ 之类, 即 "命令 0 时还在动" 的乘积项。Paper 里这个式子看上去像 typo 或者写残了, 反正工程上肯定是个 anti-reward, 不深究。

### 5.4 Asymmetric actor-critic

Policy 网络只看 proprioceptive + noisy exteroceptive (跟机器人 onboard 传感器一致的观测); Value 网络看 privileged info (真实 contact force, 真实 perturbation force, 真实 joint torque)。

直觉: 部署时机器人没法直接读 contact force, 但训练时 critic 知道这些能让 value 估计更准, 降低 variance, PPO 更稳。这是 Pinto et al. (RSS 2018) 的经典 trick。

---

## 6. PD 控制层: action 怎么变成 torque

$$\tau = k_p(q_{des} - q) - k_d \dot{q}$$

变量: $q_{des}$ 是 desired joint position, $q$ 是 actual, $\dot{q}$ 是 joint velocity, $k_p, k_d$ 是 PD gains。

$q_{des}$ 怎么来的? 两种:
1. Joystick task: $q_{des,t} = q_{default} + k_a a_t$ (绝对, 偏离默认姿势)
2. 其他: $q_{des,t} = q_{des,t-1} + k_a a_t$ (相对, incremental)

这里 $k_a$ 是 action scale, 一般设很小 (e.g. 0.25 rad), 保证 policy 输出 [-1, 1] 映射到合理 joint 范围。

**LEAP hand 的坑** (Section C.42): DYNAMIXEL servo 实际控制的是 motor current, 不是 torque, 控制律是:
$$i = k_p^m(\theta_{des}^m - \theta^m) - k_d^m \dot{\theta}^m$$

且单位、量纲跟训练时不一样。Paper 假设 $\tau = k_t i$ (torque constant), 通过 motor spec sheet 推导映射关系。同时为弥补"假 current control 跟真 current control 的偏差", 在训练里把 $k_p, k_d$ 也做 domain randomization。

这是个非常实操的细节, 揭示了一个普遍教训: **仿真里 PD 控制律跟真实 low-level controller 不可能完全一致, 与其精调不如 randomize**。

---

## 7. Domain Randomization 的完整清单

整套 sim-to-real 的核心, 把它当 checklist 看:

**Locomotion**:
- Sensor noise (gravity, velocity, joint pos/vel)
- Link center-of-mass
- Reflected inertia
- Joint friction, calibration offset
- Ground friction
- Payload mass

**Manipulation (LEAP)**:
- Friction (cube-finger)
- Cube mass
- Joint offsets
- Motor friction
- Reflected inertia
- PD gains
- Link masses
- 2cm position noise, 0.1 rad rotation noise on cube pose
- "Pose injection" (0.1 概率注入 freak-out, 模拟 occlusion)

**Vision**:
- Lighting
- Shadows
- Camera pose
- Object color
- Brightness post-processing
- 0-250ms gripper action delay

**Non-prehensile (Franka)**:
- 1-3 step action delay, 6-12 step observation delay (随机)
- Initial joint positions
- Block pose
- Curriculum 逐渐扩大 displacement 和 rotation 范围

这套配方的精髓在于: **不要试图精确建模真实世界, 而是让 policy 在所有"可能的真实"上都 work**。Tobin et al. (IROS 2017) 最早提出, 这篇 paper 把它推到极致。

---

## 8. 实验数据深度解读

### 8.1 LEAP Hand In-Hand Reorientation (Table I)

10 trials, 连续旋转次数: 3, 27, 8, 2, 15, 3, 4, 1, 3, 5

- Median: 3.5
- Mean: 7.1
- Std 极大: trial 2 拿到 27 次, trial 8 只 1 次

这个分布很 telling: 不是 policy 不行, 是 policy 在 edge case 上 catastrophic fail。失败模式:
1. Cube 卡在手指与手掌缝隙 → stall
2. Index 跟 thumb 物理互锁 (低本硬件的 flex)

对比 OpenAI 2019 的 rubik's cube 工作, 那个用了 384 台机器 + 64k envs + 数月训练。这里 30 分钟 2x 4090, 离 OpenAI 那个水平还差不少, 但开放性和 reproducibility 强太多了。

### 8.2 Franka Non-Prehensile (Table II)

35 hardware trials:
- Success: 100% median, **85.7% ± 12.2%** mean
- Position error: 1.95 cm median, 5.28 ± 3.26 cm mean  
- Rotation error: 1.72° median, 3.32 ± 1.59° mean

值得注意的训练设置: **200 Hz direct torque control**。这是一个反直觉的选择——直接学 torque 比 学 position 难得多 (reward shaping 复杂、action space 高频), 但得到的 policy 自然 compliant, sim-to-real 不需要额外做 impedance 调节。Holt et al. (CoRL workshop 2024, https://openreview.net/forum?id=gzADUWLD9X) 是相关 prior work。

训练: 16x A100 上 10 分钟。2B timesteps (Table XXVIII)。算下来 ~200M steps/sec, 跟 Table IX 的 PandaRobotiqPushCube ~487k steps/sec × 16 GPU ≈ 7.8M steps/sec 大致对得上 (action_repeat=4 之类)。

### 8.3 PickCube from Pixels

12 trials, 100% success。这个 task 简化了: Y-Z plane only, 64x64 单 RGB, 3D action (Y, Z, jaw open/close)。

10 分钟单 RTX 4090 训完, 这是 paper 最"炫"的 result, 也是 Madrona 价值的最佳 showcase。reward 用了 DexpBT (Handa et al. RSS 2023) 的 dense + sparse 组合:

$$r_t = \text{clip}\left(\sum_i r_{t,i} - \max(r_1, r_2, ..., r_{t-1}), 0\right)$$

意思是 dense reward 只奖励"超过历史最好"的进度, 防止 policy 刷 dense reward 而不真正接近 sparse goal。这是个挺巧的 reward shaping, 值得记一笔。

### 8.4 Locomotion sim-to-real

Go1: joystick, fall recovery, handstand, footstand 全部 zero-shot 部署成功。
Berkeley Humanoid (6 DoF/leg, 低成本): joystick 在硬地、软地、滑地上都稳。
G1 (29 DoF): 标准室内地板稳定走。
T1 (23 DoF): 同上。

训练时间:
- Go1 flat: 5 min (2x 4090) → rough terrain finetune
- Berkeley Humanoid: <15 min
- G1/T1: <30 min

这个速度让 reward iteration 变得现实可行, 这是 paper 反复强调的核心价值主张。

### 8.5 GPU Topology Scaling (Figure 6)

LeapCubeReorient:
- 1x 4090: 2080s (~35 min)
- 8x H100: 670s (~11 min)
- 大概 3x speedup

但 paper 明说 hyperparam 没按 topology tune (envs 数固定 8192), 这意味着 scaling 远没榨干。直觉上 8x H100 应该能轻松 8x 8192 envs = 65k envs, 但 PPO 的 batch size 也会跟着变, learning rate/minibatch 都得重调。这是 future work。

---

## 9. 跟其他 simulator 的对比 (Figure 23/24)

Figure 23 把 Cartpole stepping rate 跟 IsaacLab 和 ManiSkill3 比 (数据来自 ManiSkill3 paper, https://arxiv.org/abs/2410.00425):

- State-based: Playground ≈ IsaacLab ≈ ManiSkill3, 都在 1-2M steps/sec
- 128x128 pixels: Playground 显著领先 (Madrona batch render 的优势)
- 256x256: 优势继续

Figure 24 把每 step 时间分解成 physics (深色) + rendering (浅色) 两块:
- 在低分辨率下, MJX physics 比 ManiSkill3 快, 这是 MJX 优化好
- Madrona rendering 在所有分辨率下都比对手快
- 综合下来 Playground 是 2-3x throughput

当然这种 cross-simulator benchmark 永远有 "你 task setup 不一样" 的争议, 但量级感是对的。

---

## 10. RL 库可插拔 (Figure 8)

Playground 主推 Brax (JAX PPO/SAC), 但也支持 RSL-RL (PyTorch, ETH leggedrobotics 维护, https://github.com/leggedrobotics/rsl_rl)。

Figure 8 显示两个 PPO 实现在 Go1 上 wallclock-to-reward 曲线几乎重合, 都在 5 min 内收敛到成功 gait。这对社区是好消息: 不用绑死在 JAX 生态, 想用 PyTorch 的也能玩。

实务上 Brax 的优势是 end-to-end XLA 编译, GPU 上没 host overhead; RSL-RL 的优势是 PyTorch 调试方便、跟 Isaac Lab 兼容。

---

## 11. Limitations 我自己的解读

Paper Section VI 自报的三条:
1. JIT 慢 (1-3 min)
2. Contact 不能按 active 缩放
3. Vision-based 还 early

我再加几条 paper 没明说的:

4. **MJX 跟 MuJoCo C 不完全 numerical identical**。MJX 是重写, solver iterations, line search 行为可能有微妙差异, Table III 里那些 "iterations=2, ls_iterations=4" 修改就是为了 match。如果你已经用 MuJoCo C 训了大量 baseline, 切到 MJX 要小心 subtle 不同的 dynamics。

5. **No deformable objects, no fluids, no soft contacts**。Madrona 也提到 terrain heightfield 还没做。这对 surgical、cloth、fluid manipulation 完全不够。

6. **Vision sim-to-real 只 demo 了 PickCube 这种 toy task**。LEAP hand 仍然用 pose estimator 不是 pixel, Franka non-prehensile 也是。真正 end-to-end pixel policy 在 contact-rich 任务上 sim-to-real 仍然没解。

7. **Reward design 仍然 hand-engineered**。虽然 paper 说 Eureka 这种 LLM-assisted reward 设计是 future, 但 Playground 自己没集成。Reward shaping 的 12 个 term 都是手调的, 这才是真正 time-to-robot 的 bottleneck, 不是训练速度。

8. **Real-world deployment 栈不开放**。Paper 提到用 ros2-control + ONNX Runtime + 自定义 EtherCAT master, 但这些代码不在 paper 范围。Sim-to-real 真正难的一半是 deploy infra, 这块还得自己搭。

---

## 12. 我会怎么用这个 framework

如果让我现在就要做一个 quadruped 项目, 我会:

1. **第一周**: pip install playground, 跑通 Go1 joystick, 在我自己办公室地板上 reproduce sim-to-real
2. **第二周**: 把 reward 改成我 task 的, 用 Eureka 或自己手写, 快速 iterate (这正是 paper 强调的 reward design loop)
3. **第三周**: 切到我们自己的 robot, 改 XML (从 Menagerie 复制模板), domain randomization 调到我们 hardware 实测的 noise level
4. **第四周**: 集成 Madrona vision, 如果 task 需要

整个 stack 的学习曲线比 Isaac Lab 陡峭程度低, 关键是 JAX/PyTorch 之间能切换, 不被锁死。

---

## 13. 跟你 (Karpathy) 视角的几个联想

1. **"Software 2.0" 在 robotics 上的具体落地**: 这 paper 训出来的 policy 就是一堆 weights, 通过 ONNX 在 robot 上跑。整个 pipeline 里人类写的"Software 1.0"代码 (reward shaping, domain randomization, PD 控制律) 是"scaffolding", 最终运行时只有那 5MB 的 network。但 scaffolding 仍然 critical, 不能省。

2. **跟 nanoGPT 的类比**: nanoGPT 把 GPT2 训练压到 1 个 file 8 个 GPU 几小时; Playground 把 quadruped sim-to-real 压到 1 个 pip install 1 个 GPU 几分钟。这种"reproducible, hackable baseline"对一个领域的研究速度影响巨大, 远比那些 SOTA-but-closed 系统更有长期价值。

3. **Scaling laws for robot learning**: 这 paper 没讨论, 但下一个 natural step 是: 给定 fixed GPU 小时预算, 多 envs + 简单 task vs 少 envs + 复杂 task, 哪个 sample efficient? Vision-based 的 scaling law 跟 LLM 一样吗? 我觉得这块会出大 paper。

4. **World models vs model-free RL**: Paper 全用 PPO/SAC model-free。但 DreamerV3/TD-MPC2 这些 world model 方法在 DM Control Suite 上已经更强, 跟 Playground 集成后会不会让 locomotion sim-to-real 更 sample efficient? 这是个开放问题。Ref: https://arxiv.org/abs/2301.04104 (DreamerV3), https://arxiv.org/abs/2401.12987 (TD-MPC2)

---

## 14. 关键 Reference 一览

| 组件 | Paper/Repo | Link |
|------|-----------|------|
| MJX physics | MuJoCo XLA docs | https://mujoco.readthedocs.io/en/stable/mjx.html |
| Madrona renderer | Shacklett et al. ACM TOG 2023 | https://dl.acm.org/doi/10.1145/3592450 |
| Madrona code | GitHub | https://github.com/shacklettbp/madrona |
| Brax RL | Freeman et al. 2021 | https://github.com/google/brax |
| RSL-RL | leggedrobotics | https://github.com/leggedrobotics/rsl_rl |
| Menagerie assets | DeepMind | https://github.com/google-deepmind/mujoco_menagerie |
| Isaac Lab (对比) | Mittal et al. RA-L 2023 | https://github.com/isaac-sim/IsaacLab |
| ManiSkill3 (对比) | Tao et al. 2024 | https://arxiv.org/abs/2410.00425 |
| Berkeley Humanoid | Liao et al. 2024 | https://arxiv.org/abs/2407.21781 |
| LEAP Hand | Shaw et al. RSS 2023 | https://leap-hand.github.io/ |
| Dextreme (LEAP prior) | Handa et al. ICRA 2023 | https://arxiv.org/abs/2210.13702 |
| Domain randomization | Tobin et al. IROS 2017 | https://arxiv.org/abs/1703.06907 |
| Asymmetric AC | Pinto et al. RSS 2018 | https://arxiv.org/abs/1610.05696 |
| Phase gait | Shao et al. RA-L 2022 | https://arxiv.org/abs/2109.13362 |
| EfficientZero V2 | Wang et al. 2024 | https://arxiv.org/abs/2403.00564 |
| DreamerV3 | Hafner et al. 2023 | https://arxiv.org/abs/2301.04104 |
| TD-MPC2 | Hansen et al. 2024 | https://arxiv.org/abs/2401.12987 |
| Project page | Playground | https://mujocoplayground.github.io |

---

## 一句话总结

MuJoCo Playground 是 robotics sim-to-real 领域的 "nanoGPT moment": **开源、单 GPU、几分钟训练、能 deploy, 把过去需要整个团队几个月的工作压成一个 notebook**。它没解决 reward design 这个真正的瓶颈, 没解决 vision sim-to-real 的复杂任务, 也没解决 MJX 在 contact-rich 场景下的 scalability, 但作为一个 community baseline, 它把整个领域的入门门槛和迭代速度拉到了一个新的水位。这种工作的价值往往被 SOTA paper 低估, 但长期看影响大得多。
