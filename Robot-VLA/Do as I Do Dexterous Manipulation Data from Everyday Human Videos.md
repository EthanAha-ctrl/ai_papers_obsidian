---
source_pdf: Do as I Do Dexterous Manipulation Data from Everyday Human Videos.pdf
paper_sha256: fc774e7c1815615652ddfdc843cc9a8204eb8c8cffe1a5bde7f55e837801c09e
processed_at: '2026-08-18T06:23:51-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DO AS I DO

## 1. 这篇 paper 到底想干嘛

想象一个场景：你在 YouTube 上看到一个人 **whisking eggs**（打鸡蛋）的视频，3 秒钟的 clip。你想让机器人也学会这个动作。

传统做法是什么？你得：
- 找个 teleop operator（teleoperation rig 操作员）去手动操作机器人
- 或者在 simulation 里设计 reward function，让 robot 自己探索

这两种方式都贵、都慢、都不可 scale。

但这世界上有数百万小时的 **human doing stuff** 的视频。如果能把一个 YouTube 视频直接变成机器人能执行的 trajectory，data 问题就解决了。

**DO AS I DO 就是干这个的**——给一段 in-the-wild RGB 视频，输出 robot 可以在真实世界执行的 dexterous manipulation action。

核心问题在于：人手和机器手长得完全不一样，你怎么把"人打鸡蛋"映射到"机器手打鸡蛋"？

---

## 2. 为什么现在能做这件事

两个领域的成熟让这件事变得可行：

### Vision foundation models 成熟了
- 5 年前你想从单张 RGB 图重建 3D hand，那是 PhD thesis 级别的问题
- 现在有 **HaWoR** [45] (https://arxiv.org/abs/2404.06507) 直接给你 hand mesh
- 有 **SAM 3D** [11] (https://arxiv.org/abs/2511.16624) 从单张图生成 object mesh
- 有 **MoGe** [10] (https://arxiv.org/abs/2507.02546) 给你 depth + camera intrinsics

这些 model 都在大量数据上 pretrain 过，robust to motion blur, occlusion, low resolution。

### GPU physics simulator 成熟了
- **MuJoCo Warp** [13] (https://mujoco.readthedocs.io/en/latest/mjwarp/) 是 NVIDIA GPU 加速版本
- 一秒能跑几千次 simulation
- 这让你可以用 **sampling-based optimization**——就是随机生成一堆 candidate action，simulate 它们，挑 reward 最高的

这两个东西一组合，pipeline 就成立了：vision model 给你 "人在干什么" 的 4D reconstruction，simulator 帮你找到 "robot 怎么干同样的事" 的 action sequence。

---

## 3. Pipeline 的两个 Stage

### Stage 1: Reconstruction（从 video 到 4D trajectory）

输入：一段 RGB video clip
输出：hand + object 的 4D trajectory（3D position + orientation 随时间变化）

**关键 insight**：SAM 3D 是个 generative model，它学到了 object shape 和 pose 的联合分布。但 generative model 每次生成都是独立的，没有时间连贯性。

作者的 trick 是：**把 generative model 改造成 tracker**。

具体怎么做？

1. 选一帧作为 **anchor frame**，用 SAM 3D 生成 object 的 shape（一个 mesh，固定不变）
2. 后续每一帧，**只更新 pose，不更新 shape**
3. 更新 pose 时，用一个 guided diffusion——本质上是"在 SAM 3D 的 flow matching ODE 积分过程中，每个 Euler step 都把 trajectory 往上一帧的 pose 方向拉一点"

这个"拉一点"的强度叫 $\alpha_p$（guidance strength）。

公式长这样（Eq. 1）：

$$x_t^p = (1 - \alpha_p)(x_{t-\Delta}^p + \Delta v_\theta^p) + \alpha_p z_{ref}^p(t)$$

人话翻译：
- $x_t^p$：当前 Euler step 的 pose latent
- $x_{t-\Delta}^p$：上一个 Euler step 的 pose
- $\Delta v_\theta^p$：model 本来想 denoise 出去的方向（free generation）
- $z_{ref}^p(t)$：往上一帧 pose 插值的 target
- $\alpha_p$：你有多相信上一帧（0=完全不信，1=完全信上一帧）

如果 $\alpha_p$ 太大，tracker 太 rigid，物体转得快就跟不上；如果太小，tracker 漂移。作者用一个 **adaptive guidance**——用 2D point track 估计物体转得多快，转得快就降 $\alpha_p$，转得慢就升 $\alpha_p$。

公式是 $\alpha_p(k) = \max(0.1, 0.7 - 0.09|\Delta\theta_k|)$，其中 $\Delta\theta_k$ 是物体在 frame $k$ 的 in-plane 旋转角度。

**Stage 1 还有个关键步骤**：hand-object alignment。HaWoR 给的 hand 在 near-metric space，SAM 3D 给的 object 在 pointmap space，scale 不一样。作者用一个 elegant 的 1D least-squares 把物体沿 camera viewing ray 滑动到正确深度。

### Stage 2: Retargeting（从人手 trajectory 到机器手 action）

输入：reconstructed 4D trajectory（hand + object）
输出：robot joint commands

这里的核心 challenge 是：reconstructed trajectory 是 noisy 的。Hand 和 object 可能没对齐，第一帧可能 object 没被 grasp 但 reference 说 grasp 了，等等。

作者用 **MPPI**（Model Predictive Path Integral）sampling-based optimization，基于 SPIDER [15] (https://arxiv.org/abs/2511.09484) 的 framework。每 0.5s 规划一次，horizon 3s，1024 samples × 32 iterations。

但 baseline SPIDER 在 noisy data 上 success rate 只有 25%。作者加了三个 trick：

#### Trick 1: Warmup Steps

**直觉**：reference 的第一帧可能是 garbage——比如 object 在地上但 hand mesh 和 object mesh 重叠（reconstruction error）。如果你直接从这帧开始 track，optimizer 永远 recover 不了。

**解法**：在 reference 前面加 $H$ 个 warmup steps。这些 steps 里 object 被 "weld" 在空中不动，hand 自由移动去 grasp 它。Warmup 结束后 weld 释放，正常 simulate。

**为什么不直接 sample 一个 good grasp？** 因为这需要 task-specific heuristic。Warmup 是 task-agnostic 的——它只是给 optimizer 更多探索空间。

效果：success rate 25% → 66%。

#### Trick 2: Random Force Perturbation

**直觉**：sampling 优化容易陷入 local minima。比如 hand "balancing on fingertips"——briefly 能 track 但轻微一碰就崩。

**解法**：sample rollout 时随机施加 force/torque。这迫使 optimizer 找 robust solution，而不是 fragile equilibrium。

灵感来自 sim-to-real domain randomization——OpenAI 的 Rubik's cube hand [69] (https://arxiv.org/abs/1910.07113) 和 Rudin et al. 的 legged locomotion [70] (https://arxiv.org/abs/2109.11978)。

参数：每 1024 samples 中 4 个加扰动，force_scale=0.5, torque_scale=0.5, perturb_prob=0.05。

效果：OakInk2 上 position error 0.06 → 0.03，rotation error 0.25 → 0.14。

#### Trick 3: Transition Reward

**直觉**：物体在 "rest"（在地上）和 "in-hand"（被抓）之间转换是 trajectory 的关键时刻。但 tracking reward 太 soft——它只是 "position 差多少"，不区分 "应该 pick up 但没 pick up" 和 "pick up 了但稍微偏一点"。

**解法**：加 constant penalty。Reference 说应该 rest 时如果 object 没接触 floor → penalty。Reference 说应该 in-hand 时如果 hand-object 没 contact → penalty。

效果：reconstruction data 上 67% → 71%。

---

## 4. 实验结果讲人话

### Reconstruction 效果

在标准 benchmark DexYCB 和 HOI4D 上，DO AS I DO 都拿到 SOTA。更重要的是在 150 个 in-the-wild internet videos 上做 human eval：

- Ours 赢 67%
- FoundationPose [17] (https://arxiv.org/abs/2312.08344) 赢 18%
- Tie 15%

Fleiss' κ = 0.65，说明 rater 之间 substantial agreement，不是 random noise。

### Retargeting 效果

在 noisy reconstructed data 上，**从 25% baseline 拉到 71% success rate**。在 clean MoCap data (OakInk2 [73], https://arxiv.org/abs/2403.19417) 上从 72% 拉到 81%。

这说明 tricks 不只是 fix noisy data 的问题，它们对 clean data 也有提升——说明 baseline SPIDER 本身有改进空间。

### Real-world deployment

500 个 human-verified trajectories，其中 53% 来自 internet videos, 31% egocentric, 16% generated videos (Sora-style)。

10 个 task 在 dual UR3e + Sharpa Wave hand (22-DoF) 上部署成功：whisking, pouring, dusting, squeezing, tamping, erasing, stirring, hammering, spreading, picking。

---

## 5. 最 actionable 的发现：Data Filtering Playbook

这部分是最 "honest" 的 section，也是对想做 data scaling 的人最有用的。

作者从 **100DOH** [76] (http://fouheylab.github.io/100DOH/) dataset 抽样 2000 个 10-second clips。注意：100DOH **本身已经过滤过 hand-object interaction**，不是 raw internet video。

过滤后剩多少？

| 过滤原因 | 剩余 clips |
|---------|-----------|
| 初始 | 2000 |
| 有意义 hand-object interaction | 187 (9.35%) |
| 去掉边界问题 | 146 |
| 去掉无 activity / shot 切换 | 117 |
| 去掉 camera motion 问题 | 103 |
| 去掉 SAM 3D 失败 | 93 |
| 其他 | 83 (4.15%) |

**结论**：即使从已经过滤过的 dataset 开始，**yield 只有 4-5%**。

**人话**：你想用 internet video 做 robot data，预期 20x 的 loss rate。要拿到 1M 个 useful trajectory，你得爬 20M 个原始 video clips。这是 data scaling 的现实 tax，不是免费的 lunch。

---

## 6. 这篇 paper 的哲学

DO AS I DO 代表一种 **modular decomposition** 哲学：

- 用 vision foundation model 做 perception
- 用 physics simulator 做 dynamics
- 用 sampling optimization 在中间 bridge

这和 end-to-end learning（直接 video → action）形成对比。End-to-end 听起来更优雅，但需要海量 labeled data，而 labeled data 恰恰是我们没有的。

Modular pipeline 的好处是每个 component 可以独立改进——SAM 3D 更好了，reconstruction 自动变好；MuJoCo Warp 更快了，retargeting 自动变快。坏处是 error 累积——每个 module 的 noise 会传到下游。

**对 Karpathy 直觉的 build**：这篇 paper 本质是在问 "我们能不能用现有 foundation model 的组合来 bridge observational data 和 experiential data 的 gap"。答案是 yes，但 yield 4-5%。这意味着 short term 内 modular pipeline 是 realistic path，long term 如果 VLA model (π0, RT-2 类) 能直接 consume video，这种 modular pipeline 可能被替代。

但 **data filtering playbook 那个 20x 的数字** 是 fundamental 的——不管你用 modular pipeline 还是 end-to-end model，internet video 的信噪比就在那里。这是 scaling law 的 hidden cost。

---

## 7. 我觉得有意思的细节

1. **Guided diffusion 的数学很 elegant**——本质上是用 flow matching 的 probability path 做 tracking，把 generative model 的 prior 用作 regularizer。这个 idea 可以推广到其他 generative model 改 tracker 的场景（比如 video generation model 改 video tracker）。

2. **Warmup steps 的设计哲学很干净**——不假设任何 grasp heuristic，只是给 optimizer 更多探索空间。这是 general-purpose 的设计原则。

3. **Transition reward 的 insight**——很多 manipulation task 的关键不是 continuous tracking，而是 discrete state transition（rest ↔ in-hand）。这个 insight 可以推广到其他 RL/imitation learning 场景。

4. **Adaptive guidance 用 2D point track**——这是一个 cheap 的信号（point tracking 比 3D pose tracking 简单得多），但能显著改善 3D pose tracking 质量。Good engineering is about finding the right cheap signal。

5. **Object base plate trick**——给 upright object 加 flat base 让它稳定。这种小 trick 在 deployment 里很重要，paper 里就一句话带过，但实际工程价值很大。

---

## 8. Limitations 和未来方向

作者自己承认的：
- 只支持 rigid objects（衣服、食物不行）
- Monocular depth 有 ambiguity（无法严格区分 visual contact 和 physical contact）
- 只重建 hand + object，不重建 scene（不知道桌子在哪、有没有障碍物）
- Sim-to-real gap 限制了 upper bound

我觉得还可以延伸的方向：
- **Scene reconstruction**：结合 Habitat 或 ConceptGraphs 重建全场景，让 robot 知道 environment constraints
- **Deformable objects**：Gaussian Splatting-based deformable tracking
- **Multi-camera**：ego + exo 多视角减少 depth ambiguity
- **Tactile sensing**：加 Digit/GelSight 传感器解决 contact ambiguity
- **World model integration**：结合 DreamDojo [35] (https://arxiv.org/abs/2602.06949) 做 longer horizon planning
- **VLA pretraining**：Being-H0 [32] (https://arxiv.org/abs/2507.15597) 和 EgoVLA [31] (https://arxiv.org/abs/2507.12440) 已经在做 human video pretrain VLA，DO AS I DO 可以提供更精确的 action label

---

## 9. 一句话总结

**DO AS I DO 把 vision foundation model 当 perception prior，physics simulator 当 dynamics prior，用 sampling-based optimization 在中间填 gap，把 internet video 变成 robot trajectory——但 yield 只有 4-5%，这是 data scaling 的现实 tax。**

Reference links:
- Project page: https://do-as-i-do.com
- SAM 3D: https://arxiv.org/abs/2511.16624
- HaWoR: https://arxiv.org/abs/2404.06507
- MoGe-2: https://arxiv.org/abs/2507.02546
- SPIDER: https://arxiv.org/abs/2511.09484
- FoundationPose: https://arxiv.org/abs/2312.08344
- Flow Matching: https://arxiv.org/abs/2210.02747
- MuJoCo Warp: https://mujoco.readthedocs.io/en/latest/mjwarp/
- 100DOH: http://fouheylab.github.io/100DOH/
- OakInk2: https://arxiv.org/abs/2403.19417
- BootsTAPIR: https://arxiv.org/abs/2402.00847
- OpenAI Rubik's cube hand: https://arxiv.org/abs/1910.07113
- Rudin legged locomotion: https://arxiv.org/abs/2109.11978
- mink IK library: https://github.com/kevinzakka/mink
- CoACD: https://arxiv.org/abs/2105.01738

---

# Do as I Do 深度解析

## 1. Big Picture Intuition

这篇 paper 想解决一个根本性问题：**人类孩子看一遍大人做动作就能模仿，机器人为什么做不到？** 核心矛盾在于 robots 几乎只能从 experiential data（自己亲身尝试、teleoperation、simulation exploration）中学习，而 humans 拥有海量 observational data（看别人做）。如果能将 Internet 上数百万小时的 RGB 视频转换为 robot 可执行 trajectory，data scaling 问题就彻底破解了。

作者提出的 pipeline 叫 DO AS I DO，灵感来自 1970 年 MIT 的 "copy demo" [3] 和 Efros 等人 2003 年的 "Recognizing action at a distance" [5]（Jitendra Malik 是共同作者）。Pipeline 分两步：(1) **Reconstruction**：从 monocular RGB 视频重建 4D hand-object trajectory；(2) **Retargeting**：将重建的人类轨迹映射到 dexterous robot hand 上，生成 dynamically-feasible action sequence。

关键 technical bet 在于两个相邻领域的成熟：
- 3D vision foundation models（SAM 3D [11], HaWoR [45], MoGe [10]）能从单张图重建 depth/object/hand
- GPU-parallel physics simulators（MuJoCo Warp [13], Isaac [14]）使 sampling-based optimization 在分钟级就能跑完

Paper 的网站：https://do-as-i-do.com

---

## 2. Pipeline 架构详解

参考 Figure 2 (Method Overview) 和 Figure 7 (Reconstruction Architecture)，整体 pipeline 如下：

```
Input: RGB video clip (ego or exo, in-the-wild)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ Reconstruction Stage                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ SAM 3     │   │ MoGe     │   │ HaWoR    │    │
│  │ seg mask  │   │ depth +  │   │ hand 3D  │    │
│  │           │   │ intrins  │   │ tracking │    │
│  └──────────┘   └──────────┘   └──────────┘    │
│        │              │              │           │
│        ▼              ▼              ▼           │
│  ┌───────────────────────────┐                  │
│  │ SAM 3D: anchor frame mesh  │                  │
│  │ (shape + pose joint dist)  │                  │
│  └───────────────────────────┘                  │
│        │                                         │
│        ▼                                         │
│  ┌───────────────────────────┐                  │
│  │ Guided Diffusion Tracking │                  │
│  │ (fix shape, evolve pose)  │                  │
│  └───────────────────────────┘                  │
│        │                                         │
│        ▼                                         │
│  ┌───────────────────────────┐                  │
│  │ Adaptive α_p via BootsTAP │                  │
│  │ + Clustering pose select  │                  │
│  └───────────────────────────┘                  │
│        │                                         │
│        ▼                                         │
│  Hand-Object Alignment (scale + gravity)        │
└──────────────────────────────────────────────────┘
        │
        ▼ 4D metric trajectory (hand+object)
┌──────────────────────────────────────────────────┐
│ Retargeting Stage                                 │
│  ┌───────────────┐  ┌──────────────┐             │
│  │ Kinematic     │→ │ MPPI sampling│             │
│  │ retarget mink │  │ in MuJoCoWarp│             │
│  └───────────────┘  └──────────────┘             │
│        + Warmup Steps                            │
│        + Random Force Perturbation               │
│        + Transition Reward                       │
│        + Reference Blending                      │
└──────────────────────────────────────────────────┘
        │
        ▼
Robot actions (UR3e + Sharpa Wave, 50 Hz)
```

设计哲学是 **modular decomposition**——并非 end-to-end 训练一个 monster model，而是把 vision foundation model（SAM3D, HaWoR）和 physics simulator（MuJoCo Warp）作为可组合模块，只在中间用 guided diffusion 这一个关键 trick 桥接起来。这与 SPIDER [15] (https://arxiv.org/abs/2511.09484) 的思路一致，但 SPIDER 假设输入是 clean MoCap reference，这里要处理 noisy reconstructed reference。

---

## 3. Reconstruction 技术细节

### 3.1 关键 insight：将 generative model 改造成 tracker

SAM 3D [11] (https://arxiv.org/abs/2511.16624) 是 image-to-3D generative foundation model，学到 shape 和 pose 的联合分布：

$$p_\theta(x^s, x^p \mid c)$$

其中：
- $x^s$：object shape latent（在 TRELLIS [23] 的 structured latent space 中）
- $x^p$：6-DoF pose（13 维：3 translation + 4 unit quaternion + 6 老式 6D rotation repr 中的某些，具体见 codebase）
- $c$：conditioning（RGB image + object mask）
- $\theta$：SAM 3D 模型参数

如果对每帧独立跑 SAM 3D，会得到每帧不同 mesh + 没有时间连贯性的 pose 序列。**核心观察是 shape 和 pose 在 latent space 中是 decoupled 的，可以固定 shape latent、只更新 pose latent。**

具体做法：选定 anchor frame $\bar{x}^s$ 作为 shape reference，然后在 frame $k$ 给定上一帧 pose $x_{k-1}^p$，去采样 $x_k^p \sim p_\theta(x_k^p \mid x^s = \bar{x}^s, c_k)$，并 biased toward $x_{k-1}^p$。这就把 tracking 变成了 conditional sampling。

### 3.2 Guided Diffusion 公式详解

SAM 3D 用 flow matching [77] (https://arxiv.org/abs/2210.02747) 训练。Flow matching 的核心 ODE：

$$\dot{x} = v_\theta(x_t, t, c)$$

其中 $v_\theta$ 是 velocity field（neural network 输出）。Sample 通过从 $x_0 \sim \mathcal{N}(0, I)$ 沿线性路径 $x_t = (1-t)x_0 + t x_1$ 积分得到。

作者的 guided diffusion 修改每个 Euler step：

$$x_t^s = \underbrace{(1 - \alpha_s)(x_{t-\Delta}^s + \Delta v_\theta^s)}_{\text{denoising}} + \underbrace{\alpha_s z_{ref}^s(t)}_{\text{blending}}$$

$$x_t^p = \underbrace{(1 - \alpha_p)(x_{t-\Delta}^p + \Delta v_\theta^p)}_{\text{denoising}} + \underbrace{\alpha_p z_{ref}^p(t)}_{\text{blending}} \quad \text{(Eq. 1)}$$

变量含义：
- $t$：flow matching 的时间参数，从 0（纯噪声）到 1（target sample）
- $\Delta$：Euler step size
- $v_\theta^s, v_\theta^p$：velocity $v_\theta(x_{t-\Delta}, t-\Delta, c)$ 的 shape/pose 分量
- $\alpha_s, \alpha_p \in [0,1]$：guidance strength，控制 blending 强度
- $z_{ref}^s(t) = (1-t)\epsilon^s + t\bar{x}^s$：shape target interpolant，从 initial noise $\epsilon^s$ 插值到 anchor shape
- $z_{ref}^p(t) = (1-t)\epsilon^p + t x_{k-1}^p$：pose target interpolant，从 initial noise $\epsilon^p$ 插值到上一帧 pose

**直觉解释**：每个 Euler step，模型本来要 denoise 出一个 free sample，作者强行把 trajectory 拉向 reference interpolant。$\alpha$ 越大，越像 previous frame（tracking 越稳但越 rigid）；$\alpha$ 越小，越像 free generation（容易漂移但能跟踪 fast motion）。这等价于在 flow matching probability path 上加 repainting-style guidance [65] (https://arxiv.org/abs/2201.09865)。

### 3.3 Adaptive Guidance：$\alpha_p$ 的数据驱动设定

固定 $\alpha_p$ 会过 rigid 或产生 spurious flips。作者用 2D point tracks (BootsTAPIR [67], https://arxiv.org/abs/2402.00847) 估计物体 in-plane 旋转 $\Delta\theta_k$，然后：

$$\alpha_p(k) = \max(0.1, 0.7 - 0.09|\Delta\theta_k|)$$

变量：
- $\Delta\theta_k$：frame $k$ 相对 frame $k-1$ 的物体 in-plane 旋转角度（radians，由 SVD fit 2D rigid transform 得到）
- 系数 0.7, 0.09, 0.1 是手调的——0.7 是默认静止时的 guidance，0.09 是 rotation 的 penalty 系数，0.1 是下界防止完全失去 tracking

**直觉**：物体静止时强引导（保持稳定），物体旋转时弱引导（允许 pose 自由探索）。Table 5 ablation 显示这个 trick 在 HOI4D 上把 F-5 从 0.62 提到 0.72，提升 16%。

### 3.4 Per-frame Pose Selection

由于 guided sampling 是 stochastic 的，每帧采样 $N=25$ 个 candidates $\{x_{k,i}^p\}_{i=1}^N$。Principled 选择是用模型自身的 conditional log-density 排序：

$$\log p_\theta(x_{k,i}^p \mid \bar{x}^s, c_k) = \log p_0(x_0^p) + \int_0^1 \text{tr}\left(\frac{\partial v_\theta^p(x_t, t, c_k)}{\partial x_t^p}\right) dt \quad \text{(Eq. 2)}$$

变量：
- $x_0^p$：pose candidate $x_{k,i}^p$ 的 noise pre-image（从 $t=1$ 反向积分 ODE 到 $t=0$ 得到）
- $\text{tr}(\cdot)$：Jacobian 的 trace（instantaneous change-of-variables，来自 flow matching theory [77]）
- $D_p = 13$：pose 维度
- $T = 25$：ODE steps
- $N = 25$：candidates per frame

计算成本：$N \cdot T \cdot (1 + D_p) \approx 8700$ 次 forward+backward 通过 diffusion backbone per frame，比 generation 本身贵两个数量级，video scale 不可行。

**替代方案**：用 weighted SE(3) distance 聚类：

$$d(x_i^p, x_j^p) = w_t \|t_i - t_j\|_2 + w_r \cdot 2\arccos|\langle q_i, q_j\rangle| \quad \text{(Eq. 3)}$$

变量：
- $t_i, t_j$：translations（3D vectors）
- $q_i, q_j$：unit quaternions（4D）
- $\langle q_i, q_j\rangle$：quaternion inner product
- $2\arccos|\langle q_i, q_j\rangle|$：SO(3) 上的 geodesic angle（取绝对值是因为 $q$ 和 $-q$ 表示同一旋转）
- $w_t, w_r$：translation 和 rotation 的权重

丢弃小 cluster 当 outlier，剩余 cluster 按 2D silhouette IoU 排序选最佳。Empirically 与 log-likelihood ranking 效果相当，但快 30 倍。这是非常 Karpathy-style 的工程取舍——ablation table 5 显示在 DexYCB 上 clustering (0.71 F-5) ≈ log-likelihood (0.72 F-5)，差距忽略不计。

### 3.5 Hand-Object Alignment 的几何

HaWoR 输出 hand mesh 在 near-metric space；SAM 3D + MoGe 输出 object 在 pointmap space，scale 不一致。对齐方法：

1. 对 hand mask 的每个像素 cast ray 到 HaWoR mesh，第一 hit 平均得到 $\mathbf{c}_{hand}^H$
2. 同一组像素在 MoGe pointmap 上的 3D 值平均得到 $\mathbf{c}_{hand}^M$
3. Object mask 像素在 MoGe pointmap 上平均得到 $\mathbf{c}_{obj}^M$
4. 计算深度 scale ratio：$k = z_{hand}^H / z_{hand}^M$
5. 目标 object 位置：$\mathbf{obj}_{target} = \mathbf{c}_{hand}^H + k(\mathbf{c}_{obj}^M - \mathbf{c}_{hand}^M)$
6. 固定 mesh orientation，优化 scalar scale $s$（沿 camera ray 滑动）：

$$s^* = \arg\min_s \|\mathbf{obj}_{pos}(s) - \mathbf{obj}_{target}\|^2 = \frac{\mathbf{t}^\top(\mathbf{obj}_{target} - \mathbf{c}_{mesh})}{\mathbf{t}^\top\mathbf{t}}$$

变量：
- $\mathbf{c}_{mesh}$：visible mesh vertices 的 centroid（不含 translation）
- $\mathbf{t}$：mesh 在 camera frame 的 translation 方向
- $\mathbf{obj}_{pos}(s) = \mathbf{c}_{mesh} + s\mathbf{t}$：物体可见表面 centroid 作为 $s$ 函数

这是一个 closed-form 1D least squares（投影到 viewing ray 上找最佳深度），非常 elegant。最后用 GeoCalib [68] 做 gravity alignment。

---

## 4. Retargeting 技术细节

### 4.1 MPPI Framework

Retargeting 基于 Pan et al. SPIDER [15] 的 sampling-based optimization，本质是 Model Predictive Path Integral (MPPI) 控制。每 0.5s planning 一次，horizon 3s，每步 1024 samples，32 iterations，sim_dt=0.005s (200 Hz)。

reward 由多部分组成（Table 4）：
- `pos_rew_scale = 1.0`：object position tracking
- `rot_rew_scale = 0.3`：object rotation tracking
- `base_pos_rew_scale = 0.1`：robot base position tracking
- `base_rot_rew_scale = 0.03`：robot base rotation
- `joint_rew_scale = 0.01`：finger joints tracking
- `terminal_rew_scale = 10.0`：horizon 终端奖励
- `penetration_penalty_scale = 3000.0`：防止 exploiting simulator
- `transition_penalty_scale = 0.5`：本文新增

Kernel annealing：across iterations 和 prediction horizon 双重 anneal，从 broad exploration 到 local refinement。

### 4.2 三个核心创新

#### (1) Warmup Steps

**问题**：reconstructed reference 第一帧可能 hand 和 object 在 impossible state（比如 object 没被 grasp 但 reference 说在 hand 中），后续 horizon $H$ 由于 annealed sampling 早期不充分探索，无法 recover。

**解法**：在 reference 前面 prepend $H$ warmup steps。Warmup 期间 object 被 "weld" 在 mid-air 不动，hand 自由移动去 grasp。Warmup 结束后 weld 释放，simulation 正常进行。

**关键**：这不假设任何 grasp heuristic，只是给 optimizer 探索空间去找 stable initial state。Figure 4 top 显示这避免了 "object 没被抓起就掉" 的失败模式。

#### (2) Random Force Perturbation

**问题**：sampling 容易陷入 local minima，比如 hand "balancing on fingertips"——briefly 能 track 但轻微扰动就崩。

**解法**：在 sample rollout 时随机施加 force/torque：
- `perturb_force_scale = 0.5`
- `perturb_torque_scale = 0.5`
- `perturb_prob = 0.05`（每步扰动概率）
- `perturb_continue_prob = 0.95`（扰动后继续的概率）
- `num_perturb_samples = 4`（每 1024 samples 中 4 个加扰动）

灵感来自 sim-to-real domain randomization [69, 70]（OpenAI Rubik's cube hand, Rudin et al. legged locomotion）。Table 3 显示 OakInk2 上 pos error 从 0.06 降到 0.03（50% 提升！），rot error 从 0.25 降到 0.14（44% 提升）。

#### (3) Transition Reward

**问题**：物体在 "rest"（在地面/桌面）和 "in-hand"（被抓握）之间转换是 trajectory 的关键 inflection point。但单纯 tracking reward 太 "soft"，无法鼓励这种 step-function transition。结果就是 robot 在该 pick up 的时候没 pick up，该 place 的时候没 place（Figure 4 bottom）。

**解法**：加 constant penalty term：
- Reference timestep 处于 "rest" 状态（hand-object distance > $\epsilon$）时，如果 object 没接触 floor → penalty
- Reference timestep 处于 "in-hand" 状态（distance < $\epsilon$）时，如果 hand-object 没 contact → penalty

`transition_penalty_scale = 0.5` 看起来不大，但配合 `penetration_penalty_scale = 3000.0` 的高对比，足以驱动 optimizer 做出 transition。

### 4.3 工程细节（Appendix B）

1. **Reference Blending**：每 planning step 的初始 samples 中心是 previous plan 的 controls + 下一 chunk reference。直接 append 会产生 sharp transition 和 jerky motion，需要 interpolation blending。
2. **Robust Kinematic Retargeting**：kinematic retargeting（用 mink [55], https://github.com/kevinzakka/mink）是 cheap 的，所以跑 multiple random initial poses 避免局部最优。
3. **Object Base**：有些视频物体初始不在地面（比如容器里立着的 spoon），作者给 mesh 底部加 flat "plate" base，只与 floor contact。这是 task-agnostic 的 stabilization trick。

Object mesh 用 CoACD [78] (https://arxiv.org/abs/2105.01738) 做 convex decomposition，再 dilate 2mm 稳定 contact-rich interaction。

---

## 5. 实验数据表分析

### 5.1 Reconstruction (Table 2)

| Method | DexYCB F-5↑ | DexYCB F-10↑ | DexYCB CD↓ | HOI4D F-5↑ | HOI4D F-10↑ | HOI4D CD↓ |
|--------|-------------|--------------|------------|------------|--------------|-----------|
| HO [48] | 0.24 | 0.48 | 4.76 | 0.28 | 0.51 | 3.86 |
| MCC-HO [51] | 0.36 | 0.60 | 3.74 | 0.52 | 0.78 | 1.36 |
| G-HOP [54] | 0.31 | 0.49 | 8.11 | 0.69 | 0.91 | 0.63 |
| FoundationPose [17] | 0.69 | 0.89 | 0.89 | 0.71 | 0.91 | 0.49 |
| Any6D [47] | 0.69 | 0.88 | 0.97 | 0.71 | 0.91 | 0.50 |
| **Ours** | **0.71** | **0.93** | **0.66** | **0.72** | 0.91 | 0.49 |

关键观察：
- Joint hand-object 方法（HO, IHOI, HORSE, MCC-HO, G-HOP）整体远差于 object tracker 类（FPose, Any6D, Ours）。原因：joint methods 在 in-the-wild 视频上 hand 和 object 信号相互干扰。
- FoundationPose 和 Any6D 是 SOTA 6-DoF object trackers，但它们设计时假设 clean lab video，对 occlusion/motion blur robustness 不足（Figure 5）。
- Ours 在 DexYCB 上提升明显（F-10: 0.93 vs 0.89，CD: 0.66 vs 0.89），在 HOI4D 上和 FPose 持平。
- 在 150 个 in-the-wild videos 的 human eval 中，Ours 赢 67%，FPose 赢 18%，tie 15%。Fleiss' κ = 0.65（substantial agreement）。

### 5.2 Retargeting Ablation (Table 3)

| Method | Recon Success↑ | Recon Pos↓ | Recon Rot↓ | OakInk2 Success↑ | OakInk2 Pos↓ | OakInk2 Rot↓ |
|--------|---------------|------------|------------|------------------|--------------|--------------|
| Annealed Sampling (baseline) | 0.25 | 0.08 | 0.40 | 0.72 | 0.08 | 0.32 |
| + Warmup | 0.66 | 0.06 | 0.28 | 0.77 | 0.06 | 0.25 |
| + Perturbation | 0.67 | 0.06 | 0.30 | 0.79 | 0.03 | 0.14 |
| + Transition Reward | **0.71** | **0.05** | **0.28** | **0.81** | 0.03 | 0.15 |

观察：
- **Warmup 是 game changer**：reconstruction data 上 success rate 从 25% → 66%（+164%）。这印证了作者直觉——noisy first frame 是主要失败模式。
- **Perturbation 对 pos/rot error 提升大**（OakInk2 pos: 0.06 → 0.03），但 success rate 提升小（0.77 → 0.79）。说明 perturbation 主要让已经成功的 trajectory 更精确，而非救活失败的。
- **Transition Reward 主要救 specific failure modes**（pick/place 失败），所以 reconstruction data 上提升明显（0.67 → 0.71）但 OakInk2 上提升小。
- OakInk2 (clean MoCap) baseline 0.72 已经高，说明 SPIDER 设计的 annealed sampling 在 clean data 上工作良好；Ours 在 noisy data 上 0.71 接近 clean data baseline，证明 robustification 成功。

### 5.3 Human Data Filtering Playbook (Section 4.5)

这是 paper 最 actionable 的 section。从 100DOH [76] (http://fouheylab.github.io/100DOH/) 抽样 2000 个 10s clips（已经预过滤过 hand-object interaction）：

| 过滤阶段 | 剩余 clips | 累计 yield |
|---------|-----------|-----------|
| 初始 | 2000 | 100% |
| 有意义 hand-object interaction | 187 | 9.35% |
| 去掉 hand/object 超出边界 (41) | 146 | 7.30% |
| 去掉无 activity / 跨 shot (29) | 117 | 5.85% |
| 去掉 camera motion 失败 (14) | 103 | 5.15% |
| 去掉 SAM 3D 失败 (10) | 93 | 4.65% |
| 其他失败 (10) | 83 | 4.15% |

**结论**：即使从已经过滤过的 100DOH 开始，最终 yield 只有 4-5%。意味着如果想得到 1M 个 useful trajectory，需要爬 20M 个原始 video clips。这是 data scaling 的现实 tax。

---

## 6. 真实世界部署

500 个 human-verified trajectories：
- Internet videos: 53%
- Egocentric datasets: 31%
- Generated videos (Sora 类): 16%

10 个 task 部署在 dual UR3e + Sharpa Wave hand 上（22-DoF hand），50 Hz command：
- Whisking, pouring, dusting, squeezing, tamping, erasing, stirring, hammering, spreading, picking

涵盖 grasp types (Feix taxonomy [74])：writing tripod, power, ventral, parallel extension。

部署流程：
1. Retargeting 得到 hand actions
2. 在 MuJoCo digital twin (Figure 11) 中验证 self-collision、table contact
3. Manual align initial pose (x, y, z, yaw) 与 robot workspace
4. mink IK 把 hand trajectory 映射到 UR3e arm joint commands
5. Real-world rollout 在 half speed 执行

---

## 7. Limitations & 未来方向

作者自承：
1. **Rigid object assumption**：无法处理可变形物体（衣服、食物、纸张）。这是 SAM 3D 上游限制。
2. **Monocular depth ambiguity**：单目 RGB 无法严格区分 visual contact 和 physical contact。这意味着 hand-object distance 估计有歧义。
3. **场景重建缺失**：只重建 hand + object，没有 obstacles、articulations、其他 scene context。Intention 表达需要 hand-scene interaction（比如"杯子放在桌上" vs "杯子悬浮"）。
4. **Sim-to-real gap**：MuJoCo 物理只是近似，upper bound 了真实世界性能。

未来联想方向：
- **Scene reconstruction**：结合 Habitat-style scene reconstruction 或 3D scene graph (ConceptGraphs) 重建全场景
- **Deformable objects**：用 Gaussian Splatting-based deformable tracking 或 PhysGaussian
- **Multi-camera**：ego+exo 多视角减少 depth ambiguity
- **World model integration**：结合 DreamDojo [35] (https://arxiv.org/abs/2602.06949) 或 EgoScale [30] (https://arxiv.org/abs/2602.16710) 的 world model 做更长 horizon planning
- **VLA pretraining**：Being-H0 [32] (https://arxiv.org/abs/2507.15597) 和 EgoVLA [31] (https://arxiv.org/abs/2507.12440) 已经在做从 human video pretrain VLA，DO AS I DO 可以为这类工作提供更精确的 action label

---

## 8. 相关 Work 联想（Table 1 之外的脉络）

**Dexterous manipulation from human video 的演化**：

- **早期 kinematic retargeting** (DexMV [6], https://arxiv.org/abs/2204.12490)：纯几何映射，不考虑 forces
- **MoCap-based dynamics retargeting** (ManipTrans [59], DexMachina [60])：clean reference + RL/sampling，但需要 MoCap
- **Single human demo + RL sim-to-real** (H2Sim2Robot [16], https://arxiv.org/abs/2504.12609)：LiDAR scan + FPose，仍需要 lab setup
- **Video → policy** (VideoDex [29], VideoManip [18], DexImit [7], DexMan [22])：越来越自动化，但通常需要 multi-view 或 generated video
- **Ego-only approaches** (EgoZero [9], https://arxiv.org/abs/2505.20290; EgoDex [75], https://arxiv.org/abs/2505.11709; EgoVerse [33], https://arxiv.org/abs/2604.07607)：smart glasses 数据源
- **Pretrain-based** (R3M [28], VIP [26], LIV [27])：visual representation，不直接给 action
- **Generative video models as data source** (DreamDojo [35])：用 Sora-style video generation model 作为 data source

DO AS I DO 的独特定位：**唯一同时支持 self + generated + egocentric + internet 数据的 modular pipeline**，且无 grasping prior 或 object class 限制。

**相邻领域联想**：
- **In-context robot learning** (RT-2, OpenVLA)：VLA 模型，未来可能直接 consume DO AS I DO 生成的 trajectory data
- **Foundation model-based grasping** (AnyGrasp, Dex-Net)：传统 grasping 侧重 stable grasp，DO AS I DO 侧重 trajectory imitation
- **Tactile sensing** (Digit, GelSight)：未来可以加 tactile modality 解决 contact ambiguity 问题
- **Diffusion policy** (Diffusion Policy [21], 3D Diffusion Policy)：DO AS I DO 生成的 data 可以直接 train diffusion policy，闭环整个 learning loop

---

## 9. 整体评价

**优点**：
- Pipeline modular，每个 component 可独立改进
- Engineering 细节扎实（warmup, perturbation, transition reward 都是工程直觉驱动）
- 在 noisy reconstruction 上工作良好，这是真实 deployment 的实际场景
- Human data playbook 是 rare 的 honest data analysis

**潜在 concerns**：
- Modular pipeline 误差累积：HaWoR + SAM3D + MoGe 每个都有 noise，cumulative error 没有显式 modeling
- 22-DoF Sharpa Wave 是特定 embodiment，跨 embodiment transfer 没有验证
- Real-world deployment 10 个 task 不算大规模，long-horizon task、bimanual coordination 都没充分 stress test
- Computational cost：paper 没明确说 reconstruction + retargeting 全 pipeline 跑一个 video 要多久。Retargeting 用 1024 samples × 32 iterations × 多个 planning steps，应该是分钟到小时级别

**对 Karpathy 直觉的 build**：这篇 paper 本质是 "把 vision foundation model 当作 perception prior，physics simulator 当作 dynamics prior，用 sampling-based optimization 在两者之间填 gap"。这种 modular 组合哲学与 end-to-end learning 形成对比。如果未来 VLA 模型（π0, RT-2 类）能直接 consume video → action 的 end-to-end mapping，这种 modular pipeline 可能被替代；但在当前 data scaling 阶段，它是 realistic 的 data generation 方案。

Reference links:
- Project page: https://do-as-i-do.com
- SAM 3D: https://arxiv.org/abs/2511.16624
- SPIDER (retargeting baseline): https://arxiv.org/abs/2511.09484
- FoundationPose: https://arxiv.org/abs/2312.08344
- Flow Matching: https://arxiv.org/abs/2210.02747
- MuJoCo Warp: https://mujoco.readthedocs.io/en/latest/mjwarp/
- 100DOH dataset: http://fouheylab.github.io/100DOH/
- OakInk2: https://arxiv.org/abs/2403.19417
- BootsTAPIR: https://arxiv.org/abs/2402.00847
- mink (IK library): https://github.com/kevinzakka/mink
- CoACD (convex decomposition): https://arxiv.org/abs/2105.01738
- DexMV (early dexterous retargeting): https://arxiv.org/abs/2204.12490
- H2Sim2Robot: https://arxiv.org/abs/2504.12609
- EgoZero: https://arxiv.org/abs/2505.20290
- EgoDex: https://arxiv.org/abs/2505.11709
