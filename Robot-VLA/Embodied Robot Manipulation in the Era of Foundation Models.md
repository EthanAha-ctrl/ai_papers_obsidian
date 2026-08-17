---
source_pdf: Embodied Robot Manipulation in the Era of Foundation Models.pdf
paper_sha256: f2e503e949b7d0f1a8ae8ab7774afab2e8e145e0b799fb78f1613beda52a14ea
processed_at: '2026-08-04T03:36:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍这篇 survey

## 这篇 paper 到底在干嘛

它想回答一个问题：**foundation model 进来之后，让 robot 抓东西、挪东西、用东西这件事，整个 field 变成啥样了**。

以前你让 robot 干活，得给它写一个专门程序："看到这个杯子，往左 30 度伸手，闭合 gripper 到 2cm..."。换一个杯子、换一张桌子，程序就得重写。foundation model（GPT、CLIP 这种）进来之后，你希望直接说"把杯子给我"，robot 自己想办法。

这中间的 gap 怎么填，整个 field 这两年爆炸式发展，paper 太多太乱。这篇 survey 的价值在于它给了一个**清爽的 mental model**：

```
┌─────────────────────────────────────────┐
│  High-Level Planner  ← 想做什么          │
│  (大脑 / 前额叶皮层)                      │
├─────────────────────────────────────────┤
│  接口：language / code / latent / 3D     │
├─────────────────────────────────────────┤
│  Low-Level Controller ← 具体怎么动       │
│  (小脑 / 脊髓)                            │
└─────────────────────────────────────────┘
```

上面那层决定"意图"，下面那层决定"动作"。两层中间有个接口，接口的形态决定了整个 system 的形状。这是这篇 survey 最 key 的 framing。

参考：[Awesome-Robotics-Manipulation github](https://github.com/RayBai/awesome-robotics-manipulation)、[arXiv 2510.10903](https://arxiv.org/abs/2510.10903)

---

## High-Level Planner：robot 的"前额叶"

这层让 robot **想**。它可以输出五类东西：language、code、motion trajectory、affordance map、3D scene structure。每类对应一条研究 line。

### SayCan：语义合理性 × 物理可行性

[SayCan (Brohan et al., CoRL 2022)](https://say-can.github.io/) 是 build intuition 最好的起点。你跟 LLM 说"我渴了"，它会建议"开柜子 → 拿可乐 → 打开 → 喝"。它知道**语义上**哪一步合理。但 robot 不能只看语义——它还得知道**物理上**可乐在不在够得到、柜子能不能开。SayCan 把这两件事相乘：

$$
\text{score}(a_i) = \underbrace{p_{\text{LLM}}(a_i \mid \text{task}, a_{<i})}_{\text{语言上合理吗}} \cdot \underbrace{p_{\text{aff}}(a_i \mid s)}_{\text{物理上做得到吗}}
$$

- $a_i$：第 $i$ 步候选 skill（"开柜子"、"拿可乐"）
- $p_{\text{LLM}}$：LLM 给的条件概率，task description + history $a_{<i}$ 作为 context
- $p_{\text{aff}}$：robot 自己 interaction 数据上学到的 affordance prior
- $s$：当前 state

这就像你问 ChatGPT"晚上吃啥"它说"日料"，但还得乘以"我家附近有没有日料店、钱包够不够"。两个相乘才是真正可执行的 choice。

### Inner Monologue：闭环才有救

SayCan 是**开环**的——一次 plan 完，robot 就闷头执行，错一步就废。[Inner Monologue (Huang et al.)](https://arxiv.org/abs/2207.05608) 让 LLM "边做边想"：robot 每走一步，把当前 scene description、上次 action 成不成功、人有没有插话都 feed 回 LLM，让它 re-plan。从开环到闭环是关键转折。

### Code as Policies：把"想"绑到代码上

[Code as Policies (Liang et al., ICRA 2023)](https://code-as-policies.github.io/) 解决纯 language plan 的精度问题。"把杯子放桌上"这种话，robot 不知道放哪、什么 pose。trick 是给 LLM 一堆 API：

```python
cup_pos = get_position("cup")
table_pos = get_position("table")
grasp(cup_pos)
move_to([table_pos[0], table_pos[1], table_pos[2]+0.1])
release()
```

LLM 的 compositionality（for loop、if else、function call）直接变成可组合的执行结构，还能 unit test。Statler 维护一个 explicit world state（一个 LLM-readable 的 JSON），解决 context length 问题。

### VoxPoser：LLM 产 3D value map，optimizer 求轨迹

[VoxPoser (Huang et al., CoRL 2023)](https://voxposer.github.io/) 更激进——LLM 直接输出一个 3D 空间里的"吸引/排斥场" $V(x) \in \mathbb{R}^3$，每个 voxel 一个值。然后 classical motion optimizer 在这个 field 上求一条平滑、无碰撞的轨迹：

$$
\tau^* = \arg\min_\tau \int_0^T \big( \underbrace{\|\nabla V(\tau(t))\|^2}_{\text{跟着 value map 走}} + \underbrace{\lambda \|\ddot\tau(t)\|^2}_{\text{轨迹要 smooth}} \big) \, dt
$$

- $\tau: [0,T] \to \mathbb{R}^3$：end-effector 在 3D 空间的轨迹
- $\nabla V$：value map 的梯度，把 robot 拉向目标 voxel、推离 obstacle voxel
- $\lambda$：smoothness 权重
- $\ddot\tau$：轨迹二阶导（加速度）

LLM 给**语义**（哪是目标、哪是障碍），optimizer 给**几何保证**（collision-free、smooth）。这是 foundation model + classical control 的 hybrid，两边的强项都用上了。

[ReKep](https://rekep.github.io/) 进一步引入 relational keypoint constraint，让 LLM 选 keypoint（杯把、桌角），再在 keypoint 之间加距离/角度约束喂给 nonlinear optimizer。

### Affordance：物体"能让你做什么"

Gibson 的 affordance 概念：物体本身 affordance 决定你能对它做什么 action。把手 afford "pull"，杯口 afford "drink from"。

[Transporter Networks (Zeng et al., CoRL 2021)](https://transporternets.github.io/) 在 2D image 上直接 predict pick-and-place heatmap：

$$
\text{pick}(u,v) = \arg\max_{u', v'} \langle f_\theta(I)(u,v), f_\theta(I)(u', v') \rangle
$$

- $f_\theta(I)$：image 的 feature map
- $(u,v)$：pick 位置（pixel 坐标）
- $(u', v')$：feature 最匹配的 place 位置

整个过程全卷积、spatially equivariant。简单粗暴但 work。

[CLIPort (Shridhar et al.)](https://cliport.github.io/) 把 CLIP 加进来——CLIP 负责"理解语言说的是啥"（semantic branch），Transporter 负责"在哪抓哪放"（spatial branch）。两路解耦，是一个非常 successful 的 design pattern。

### 3D Representation 当 Planner：NDF 的魔法

[NDF (Simeonov et al., ICRA 2022)](https://yilundu.github.io/ndf/) 学一个 SE(3)-equivariant 的 field：

$$
\mathcal{F}_\theta: \mathbb{R}^3 \to \mathbb{R}^d, \quad \mathcal{F}_\theta(Tx) = T\mathcal{F}_\theta(x) \text{ for any } T \in \text{SE}(3)
$$

- $\mathcal{F}_\theta$：把 3D 点映射到 d 维 feature 的网络
- $T$：任意 SE(3) 变换（旋转 + 平移）
- $x$：3D 点

意思是物体 rotate/move，feature 也跟着 rotate/move。这样 few-shot grasp transfer 变成 feature matching：你 demo 一次怎么抓这个杯子，下次新杯子只要 feature 对得上，grasp pose 就 transfer 过去。[F3RM](https://f3rm.github.io/) 把 CLIP feature distill 进 3D field，能 language-conditioned 抓东西。[D³Fields](https://arxiv.org/abs/2409.17065) 推广到 dynamic scene。

**为什么这层当 planner 用？** 因为它输出的不是 motor command，是 mid-level 的 grasp candidate 或 SE(3) constraint，介于 perception 与 control 之间。这是 paper 一个很妙的 insight。

---

## Low-Level Controller：robot 的"脊髓"

Low-level 把 perception 转成 executable action。paper 把它拆成四个 axis：**怎么学**、**看什么**、**中间想什么**、**怎么输出动作**。

### Axis 1：Learning Strategy — 怎么学

| 学法 | 直觉 | 代表 |
|---|---|---|
| RL | 试错 + reward | QT-Opt, Dreamer, TD-MPC, VLA-RL |
| IL | 看专家怎么做就抄 | BC, ACT, pose-level IL, GAIL |
| Auxiliary task | 加点额外 task 让 representation 更好 | World model, goal extraction |

**RL** 里 Dreamer 的核心是 latent world model：

$$
\hat{s}_t \sim q_\phi(\hat{s}_t \mid h_t, o_t) \quad \text{posterior (看到观察后)}
$$
$$
\tilde{s}_t \sim p_\phi(\tilde{s}_t \mid h_t) \quad \text{prior (rollout 用)}
$$

- $h_t$：deterministic recurrent state
- $\hat{s}_t$：stochastic latent（posterior 用于 learning，prior 用于 imagination rollout）
- $o_t$：observation

Policy 在 latent space 想象未来，actor-critic 在 imagined trajectory 上学。这就像 robot 在"做梦"里练习，醒来用学好的 policy。

**IL** 最朴素的是 BC：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o, a) \sim \mathcal{D}_{\text{expert}}} \|a - \pi_\theta(o)\|^2
$$

- $o$：observation
- $a$：专家动作
- $\pi_\theta(o)$：policy 输出

问题：**compounding error**。每步错一点，长程任务崩盘。[ACT (Zhao et al.)](https://tonyzhaozh.github.io/aloha/) 用 transformer 一次 predict k 步 action chunk，把每步误差"摊薄"。Pose-level IL 直接预测 SE(3) end-effector pose，让底层 controller 处理 joint execution，对 insertion 类任务特别好。

### Axis 2：Input Modeling — 看什么

**(a) 2D Vision-Action**：[Diffusion Policy (Chi et al.)](https://diffusion-policy.cs.columbia.edu/) 是 milestone。把 action 生成表达成 conditional denoising：

$$
a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(a_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(a_t, t, o)\right) + \sigma_t z
$$

- $a_t$：noise level $t$ 下的 noisy action chunk（一般 16 步 7-DoF）
- $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$：noise schedule
- $\epsilon_\theta$：预测 noise 的网络（UNet 或 transformer）
- $o$：conditioning observation
- $z \sim \mathcal{N}(0, I)$：随机噪声

**关键直觉**：输出是 multimodal distribution。同一张图，杯子可以从左抓也可以从右抓。Gaussian BC 只能给一个 mean，表达不了这种 multimodality；diffusion 可以。

**(b) 3D Vision-Action**：[DP3](https://3d-diffusion-policy.github.io/) 把 point cloud 加进 conditioning；[RVT-2](https://rvt-2.github.io/) 用 multi-view transformer 推 3D action target。

**(c) VLA — 重头戏**。paper 给了一个 2D/3D × model-oriented/model-agnostic 的四象限。我用 mental map 看更清楚：

```
                  Model-Oriented                    Model-Agnostic
2D    Non-LLM VLA: RT-1, VIMA          Inference: RoboMonkey, CronusVLA
      LLM/VLM VLA: RT-2, OpenVLA, π0   RL post-train: SimpleVLA-RL, ConRFT
      Latent VLA: UniVLA, AgiBot-GO1   Auxiliary: CoT-VLA, ReconVLA
      Dual-system: LCB, HiRobot, G0    Efficiency: TinyVLA, VLA-Cache
      
3D    Embedding: SpatialVLA, GeoVLA    (mostly unexplored)
      Alignment: BridgeVLA
      World model: 3D-VLA, Evo-0
```

几个里程碑：
- [RT-1](https://robotics-transformer.github.io/)：transformer + discretized action token
- [RT-2](https://robotics-transformer2.github.io/)：把 action 当 text token，与 VLM co-train，zero-shot 跨 embodiment
- [OpenVLA](https://openvla.github.io/)：开源 7B，frozen VLM + lightweight action head
- [π0](https://arxiv.org/abs/2410.24164)：PaliGemma + flow-matching action expert，目前 strongest generalist VLA 之一
- [UniVLA](https://univla.github.io/)：在 visual feature space 学 latent action，再 decode 到 embodiment-specific control

Dual-system VLA 受 Kahneman System 1/System 2 启发：

$$
\underbrace{a_{\text{fast}}^{(t)} = \pi_{\text{fast}}(o_t, \ell)}_{\text{System 1, ~50Hz, reactive}}, \quad \underbrace{\ell = \pi_{\text{slow}}(o_{t-\Delta}, L)}_{\text{System 2, ~5Hz, deliberative}}
$$

- $a_{\text{fast}}^{(t)}$：第 $t$ 步 fast action
- $\ell$：latent intent（System 2 输出）
- $\Delta$：System 2 触发周期
- $L$：language instruction
- $o_t$：当前 observation

System 2 在 slow timescale 给 abstract plan/latent，System 1 在 fast timescale 执行。难点是 arbitration（何时 handoff）、credit assignment（失败算谁的）、consistency（两 system 是否对齐）。

**(d) Tactile-based**：vision-VLA 在 contact-rich 任务上基本 saturated，下一个突破点是 tactile。[Sparsh](https://sparsh-website.github.io/) 在大规模 self-supervised visuotactile data 上学 generalizable embedding，是这一类 backbone 候选。[Tactile-VLA](https://arxiv.org/abs/2507.09160) 把 tactile 加进 VLA pipeline。[Touch Begins](https://arxiv.org/abs/2506.13762) 提出"vision 做不到时用 tactile 收尾"的两阶段策略。

**(e) Extra modalities**：Force（[ForceVLA](https://arxiv.org/abs/2505.08150)）、Audio（[ManiWAV](https://maniwav.github.io/)）在 occlusion 下做 imitation。

### Axis 3：Latent Learning — 中间想什么

这层是 perception 与 policy 之间的"中间表示"。paper 分两类。

**(a) Pretrained latent**：按数据源分三类。
- General image (ImageNet)：[Theia](https://arxiv.org/abs/2407.00390) distill 多个 vision foundation model 进 compact representation
- Human egocentric (Ego4D)：[VC-1](https://embodied-agent.org/)、[R3M](https://jyy222.github.io/r3m/)、[Voltron](https://voltron-robot.github.io/)
- Robot 自己的 data (BridgeV2)：[RPT](https://robotic-mlrpt.github.io/)、[Premier-TACO](https://arxiv.org/abs/2405.15711)

Premier-TACO 的 temporal contrastive loss：

$$
\mathcal{L}_{\text{TACO}} = -\mathbb{E} \log \frac{\exp(\text{sim}(z_t, z_{t+k}^+))}{\sum_{z^-} \exp(\text{sim}(z_t, z^-))}
$$

- $z_t = g_\theta(o_t, a_t)$：action-conditioned representation
- $z_{t+k}^+$：同 trajectory 未来 frame（正样本）
- $z^-$：negative samples

让 representation encode **action-conditional dynamics**——不光"看到啥"，还"看完之后会怎么样"。

**(b) Latent action**：这层我觉得是论文最有价值的 contribution 之一。把 action abstraction 本身当作学习目标。

Discrete 路线：[LAPA](https://latentactionpretraining.github.io/) 先从 visual transition 学 latent action codebook，再让 VLM 直接 predict latent action token。这就像让 VLM 学会"看视频就能猜出 robot 在做什么动作"，然后再让它"看着 scene 自己说出该做什么动作"。

Continuous 路线：[MimicPlay](https://mimicplay.github.io/) 从 human play 学 latent plan $\ell_t$，再 decode 到 robot control：

$$
\ell_t = g_\phi(o_t, o_{\text{goal}}), \quad a_t = \pi_\theta(o_t, \ell_t)
$$

- $\ell_t$：latent action plan
- $g_\phi$：plan generator
- $o_{\text{goal}}$：goal image
- $\pi_\theta$：low-level policy

让 long-horizon imitation 不需要 long-horizon demos——人类随便玩，robot 学着玩。

Koopman-based：[KOROL](https://arxiv.org/abs/2406.13821) 把 nonlinear dynamics lift 到 linear latent space：

$$
g(f(x)) = A \, g(x)
$$

- $f$：nonlinear dynamics
- $g$：lifting function（encoder）
- $A \in \mathbb{R}^{d \times d}$：linear operator

long-horizon rollout 可以用矩阵幂 $A^k$ 闭环算，planning 大大简化。

### Axis 4：Policy Learning — 怎么输出动作

| Policy | 形式 | 代表 |
|---|---|---|
| MLP | $a = \text{MLP}(z)$ | R3M, RPT |
| Transformer chunking | 一次 predict k 步 | ACT, BAKU |
| Transformer autoregressive | next-token | ICRT, CARP |
| Diffusion | iterative denoising | Diffusion Policy, DP3 |
| Flow Matching | deterministic ODE | π0, RTC |
| State-space (Mamba) | linear recurrence | MAIL |
| Frequency | spectral token | FreqPolicy |

**Flow Matching vs Diffusion** 是 build intuition 的关键。Diffusion 是 stochastic SDE，要迭代 50-1000 步。FM 是 deterministic ODE：

$$
\frac{da_t}{dt} = v_\theta(a_t, t, o), \quad a_0 \sim p_0, \quad a_1 \sim p_{\text{data}}
$$

- $v_\theta$：vector field（学到把 noise distribution $p_0$ transport 到 data distribution $p_{\text{data}}$）
- $a_t$：时间 $t$ 上的 action
- $o$：conditioning

FM 推理只需几步 Euler ODE 求解，trajectory 更 smooth。这对 robot control 关键——latency 和 smoothness 都重要。π0 选 FM 不是偶然。[RTC](https://arxiv.org/abs/2506.07339) 通过 overlap action 生成与 ongoing control 实现 real-time execution。

---

## 四个 Core Challenge

1. **Robot Brain**：通用架构（支持多 modality、多 embodiment）、终生学习（不遗忘、能 transfer）、long-horizon "funnel of success"（planner 给 wide funnel，controller 收敛进 success region）、smooth motion generation（dynamics-consistent + impedance-like）
2. **Data Bottleneck & Sim-to-Real**：data flywheel（robot 自主 collect → filter → relabel → retrain）、differentiable simulator（[DiffTORI](https://arxiv.org/abs/2410.05067)）
3. **Multimodal Physical Interaction**：tactile、audio、proprioception fusion；deformable material（cloth、cable、fluid、granular）需要 graph/field-based object representation
4. **Safety**：intrinsic safety（kinematic/dynamic limit、force/energy regulation）、multi-robot coordination、human-robot collaboration（intent inference、shared autonomy）、fault detection & recovery

---

## 我自己的几个直觉

1. **High-level 与 low-level 的接口是 latent**。SayCan 接口是 discrete skill，VoxPoser 接口是 3D value map，Code as Policies 接口是 API call，UniVLA 接口是 latent action token。Latent action 最有潜力——它把抽象与执行解耦，让 high-level 用 semantic-level reasoning、low-level 用 embodiment-specific execution。

2. **Diffusion / Flow Matching 是 manipulation 的 de-facto policy**。因为 multimodality 与 trajectory-level 生成，在 contact-rich 任务上比 Gaussian BC 健壮太多。FM 比 Diffusion 在 latency 与 smoothness 上更优。

3. **VLA 三大流派**：单 system 大 VLM（OpenVLA, π0）、dual-system（LCB, RationalVLA, HiRobot）、latent-action bridge（UniVLA, AgiBot-GO1）。第三流派最有潜力，因为它显式建模 embodiment gap。

4. **3D representation 与 foundation model 还没真正 merge**。大多数 VLM 在 2D image-text 上 pretrain，没 intrinsic 3D 理解。SpatialVLA、GeoVLA、F3RM 是 early attempt，standardization 缺失。

5. **RL post-training 是 VLA 下一个 scaling dimension**。Pre-LLM 时代 RL 解决 sample efficiency；VLA 时代 RL 解决"IL 学不出 success-critical behavior"。[SimpleVLA-RL](https://arxiv.org/abs/2412.03481)、[VLA-RL](https://arxiv.org/abs/2505.18719)、[ConRFT](https://arxiv.org/abs/2502.05450) 是开端。

6. **Tactile 与 audio 是 unexploited frontier**。vision-VLA 在 contact-rich 任务上 saturated，下一个突破点是把 tactile representation 像 CLIP 那样预训练好。

7. **Funnel of success** 这个比喻我特别喜欢。high-level planner 给 wide funnel（允许 deviation），low-level controller 逐步把 state 收敛进 success region。现代体系需要 explicit coupling：planner output 要包含 feedback-suitable sub-goal，controller 要包含 recovery mode。

---

## 一句话总结

这篇 survey 给出的 mental model 就是：**foundation model 时代的 robot manipulation = high-level planner（用 LLM/VLM/3D field 想）+ low-level controller（用 diffusion/FM/latent action 动）+ 一个 latent 接口把它们粘起来**。整个 field 的所有 paper 都可以塞进这个框架的某个格子里。这种 unification 比追某个具体 SOTA 更有价值，因为它告诉你下一个 SOTA 大概率出在哪个格子的空白处。

想深入聊哪一格我都可以再展开。比如 latent action learning 的 VQ collapse 问题、dual-system VLA 的 credit assignment、diffusion policy 的 action chunk vs autoregressive trade-off——这些都是有 open problem 的格子。

---

# Embodied Robot Manipulation in the Era of Foundation Models — 深度讲解

## 1. Paper 的整体定位与 contribution

这篇 survey 来自 Xi'an Jiaotong University、HKUST(GZ)、Westlake、PKU、Sydney 等多个 group 的合作 (github: [Awesome-Robotics-Manipulation](https://github.com/RayBai/awesome-robotics-manipulation))， arXiv 编号 2510.10903 (2025)。 它的定位有几个特点：

1. **不按 model class 组织， 而按 abstraction 层次组织**。 市面上大多数 VLA survey ([Ma et al. 2405.14093](https://arxiv.org/abs/2405.14093), [Zhong et al. 2507.01925](https://arxiv.org/abs/2507.01925), [Wolf et al. Frontiers 2025](https://www.frontiersin.org/articles/10.3389/frobt.2025.1606247)) 会把 Diffusion Policy、 VLA、 Generative 方法分别当作独立 chapter。 本文把它们重新嵌进一个统一的 two-level abstraction： **high-level planner** 负责 task reasoning / decomposition / geometric intent； **low-level controller** 负责把 perception 转成 executable action。
2. **引入 training-paradigm-oriented taxonomy**， 把 low-level 拆成 Learning Strategy (RL/IL/Auxiliary)、 Input Modeling、 Latent Learning、 Policy Learning 四个 axis。 这样做的好处是， 当你看一篇新 paper， 可以从这四个 axis 直接判断它"在做什么改动"。
3. **Affordance 与 3D representation 被提升为 planner**。 这是与传统 perception→control 二分法的关键差别——例如 NDF / F3RM / Gaussian Splatting 输出的不是 control signal， 是 mid-level 的 grasp candidate 或 SE(3) constraint， 因此更接近 planner。

整体结构 (Fig. 1)： High-level (Sec II) 六个 component： LLM-based planning、 MLLM-based planning、 Code generation、 Motion planning、 Affordance、 3D representation。 Low-level (Sec III) 四个 component： Learning Strategy、 Input Modeling、 Latent Learning、 Policy Learning。 最后 (Sec IV) 四个 challenge： Robot brain、 Data/sim-to-real、 Multimodal physical interaction、 Safety。

---

## 2. High-Level Planner — 为什么需要再拆 6 个 sub-direction

Planner 在 foundation model 时代被重新定义： 它不是 symbol planner (PDDL/STRIPS)， 而是任何"把 task spec 与 observation 映射成结构化 action intent"的模块。 论文把它的输出空间分成 language、 code、 motion、 affordance、 3D structure 五类， 这五类彼此**正交但可叠加**。 下面我一个个拆。

### 2.1 LLM-based Task Planning

最早期的 SayCan ([Brohan et al. CoRL 2022, arXiv:2204.01691](https://arxiv.org/abs/2204.01691)) 把 LLM 当作 token-level skill selector：

$$
\text{score}(a_i) = \underbrace{p_{\text{LLM}}(a_i \mid \text{task}, a_{<i})}_{\text{language relevance}} \cdot \underbrace{p_{\text{aff}}(a_i \mid s)}_{\text{learned affordance}}
$$

变量含义： $a_i$ 是第 $i$ 步候选 skill (来自固定 skill library)， $p_{\text{LLM}}$ 是 LLM 给出的 conditional likelihood (task description $+$ history $a_{<i}$ 为 context)， $p_{\text{aff}}$ 是从 robot interaction 数据上学到的 affordance prior (通常是 BC 或 RL 学出的 success probability)， $s$ 是当前 state。 这个乘积形式把"语义上的合理" 与"物理上可行"做了 soft AND。

Grounded Decoding ([Huang et al. NeurIPS 2023, arXiv:2303.00869](https://arxiv.org/abs/2303.00869)) 把这种 coupling 推到 token level： 在每个 decoding step 同时用 language model logits 与 grounding model log-prob 给 logits 加权， 实现了 open-vocabulary planning， 不再依赖固定 skill set。 Inner Monologue ([Huang et al. arXiv:2207.05608](https://arxiv.org/abs/2207.05608)) 引入 closed-loop feedback， 让 LLM 能"看到" scene description、 success/failure、 human prompt， 从而 re-plan。 这是从**开环 plan**到**闭环 plan**的关键转折， 否则 LLM hallucination 会导致不可恢复的执行错误。

后续 LLM+P ([Liu et al. arXiv:2304.11477](https://arxiv.org/abs/2304.11477)) 把 LLM 与 classical PDDL planner 组合： LLM 负责 natural language→PDDL problem 翻译， 真正 planning 交给 Fast Downward， 解决 long-horizon reasoning 的不可靠问题。 REFLECT ([Liu et al. CoRL 2023, arXiv:2306.10396](https://arxiv.org/abs/2306.10396)) 让 robot 把失败 episode summarize 成 textual summary 再 feed 回 LLM。 多 agent 工作 ([MALMM, arXiv:2411.17636](https://arxiv.org/abs/2411.17636)) 用多个 LLM 各自负责 perception / planning / verification。

**直觉**： LLM planner 真正有价值的部分是 (i) 语义 commonsense、 (ii) 长序列 decompose、 (iii) failure explanation。 它最弱的部分是 spatial/physical precision， 所以一定要和 affordance 或 motion 模块耦合。

### 2.2 MLLM-based Task Planning

从 unimodal LLM 到 MLLM， 主要解锁的是 "vision 与 language 联合 reasoning"。 论文分两条线：

**(a) 通用 MLLM 适配**。 PaLM-E ([Driess et al. ICML 2023, arXiv:2303.06869](https://arxiv.org/abs/2303.06869)) 把 robot state (pose、 image、 force) 当作 token sequence 注入 PaLM 一起训练， 是真正的 embodied multimodal model， 但代价是需要在 robot 数据上 cotrain， 计算量极大。 VILA ([Du et al. ICLR 2024, arXiv:2310.02599](https://arxiv.org/abs/2310.02599)) 直接 zero-shot 用 GPT-4V 做 planning， 依赖 video prediction 做 lookahead。 PG-InstructBLIP ([Gao et al. ICRA 2024, arXiv:2403.08461](https://arxiv.org/abs/2403.08461)) 通过 object-centric finetune 注入物理 prior。

**(b) Robotics-specific MLLM**。 RoboBrain ([Ji et al. CVPR 2025, arXiv:2507.01432](https://arxiv.org/abs/2507.01432))、 Gemini Robotics ([arXiv:2503.20020](https://arxiv.org/abs/2503.20020))、 RynnEC ([arXiv:2508.14160](https://arxiv.org/abs/2508.14160)) 是新一代 robotics-specific MLLM， 它们 explicit 地把 manipulation-relevant 数据 (grasp pose、 contact point、 trajectory) 作为 supervision。 这些模型不再只是"看到 scene 描述一下"， 而是"看到 scene 输出 actionable sub-goal"。

EmbodiedGPT ([Mu et al. NeurIPS 2023, arXiv:2305.09996](https://arxiv.org/abs/2305.09996)) 把 chain-of-thought 推广到 embodied setting： $\text{CoT}: (I, L) \rightarrow \text{reasoning trace} \rightarrow \text{action}$， 这样 long-horizon task 可以靠 reasoning trace 分摊误差。 AHA ([Duan et al. arXiv:2410.00371](https://arxiv.org/abs/2410.00371)) 教 MLLM 识别并解释 execution failure， 这是 failure-aware planning 的 representative work。

### 2.3 Code Generation

纯 language plan 的根本问题是 lack of precision： "把杯子移到桌子上"无法表达 pick angle、 trajectory constraint。 Code as Policies ([Liang et al. ICRA 2023, arXiv:2209.07753](https://arxiv.org/abs/2209.07753)) 给 LLM 暴露 perception API (`get_position("cup")`, `get_size(...)`) 与 control API (`move_to`, `grasp`)， 让它生成可执行 Python：

```python
# generated by LLM
cup_pos = get_position("cup")
table_pos = get_position("table")
grasp(cup_pos)
move_to([table_pos[0], table_pos[1], table_pos[2]+0.1])
release()
```

这种 abstraction 把 LLM 的 compositionality (for-loop、 if-else、 recursive call) 直接转化成 executable structure， 而不需要把每一步都 token-level decode。 ProgPrompt ([Singh et al. ICRA 2023, arXiv:2211.11583](https://arxiv.org/abs/2211.11583)) 类似但环境 grounding 不同。 Demo2Code ([Wang et al. NeurIPS 2023, arXiv:2310.07950](https://arxiv.org/abs/2310.07950)) 把长 horizon demonstration 总结成 code skeleton， 是从 demos 提取 reusable skill 的方向。 SHOWTELL ([Murray et al. CoRL 2024](https://openreview.net/forum?id=...)) 干脆去掉 language 中介， 直接从 visual demo 生成 policy code。 Statler ([Yoneda et al. ICRA 2024, arXiv:2306.17840](https://arxiv.org/abs/2306.17840)) 维护一个 explicit world state (一个 LLM-readable 的 JSON-style state)， 解决 context length 与 state tracking 问题。 HyCodePolicy ([arXiv:2508.02629](https://arxiv.org/abs/2508.02629)) 把 symbolic execution trace 与 perceptual feedback 交织。

**直觉**： Code planning 的本质是把 LLM 的 reasoning 能力绑定到一个**外部的、 确定性的、 可组合的**执行 substrate 上， 而不是直接靠 next-token prediction 做 action。 这避免了 LLM 输出 jitter、 让 verification 变得可做 (你可以 unit-test generated code)。

### 2.4 Motion Planning via LLM/VLM

更进一步， 让 foundation model 直接产出**连续 motion objective**， 而不是 discrete skill。

VoxPoser ([Huang et al. CoRL 2023, arXiv:2207.00895](https://arxiv.org/abs/2207.00895)) 构造一个 3D value map $V(x) \in \mathbb{R}^3$ (每个 voxel 一个 attraction/repulsion 值)， LLM 给出 affordance、 constraint、 target， 然后用 classical motion optimizer (RRT / CHOMP / trajopt) 在 $V$ 上求轨迹：

$$
\tau^* = \arg\min_\tau \int_0^T \big( \|\nabla V(\tau(t))\|^2 + \lambda \|\ddot\tau(t)\|^2 \big) \, dt
$$

变量： $\tau: [0,T]\to \mathbb{R}^3$ 是 end-effector trajectory， $\nabla V$ 是 value map 的梯度 (吸引到目标 voxel、 远离 obstacle voxel)， $\lambda \|\ddot\tau\|^2$ 是 smoothness regularizer。 LLM 提供 $V$ 的语义， motion optimizer 提供 smoothness 与 collision-free guarantee。

CoPa ([Huang et al. IROS 2024, arXiv:2403.07488](https://arxiv.org/abs/2403.07488)) 引入 visual prior (具体 part、 grasp surface)， ManipLLM ([Li et al. CVPR 2024, arXiv:2402.14729](https://arxiv.org/abs/2402.14729)) 学习 object-centric contact-aware representation。 ReKep ([Huang et al. CoRL 2025, arXiv:2310.09966](https://arxiv.org/abs/2310.09966)) 引入 relational keypoint constraint：

$$
\mathcal{C}_{ij}(\mathbf{k}_i, \mathbf{k}_j) = 0, \quad \mathbf{k}_i \in \text{scene}, \mathbf{k}_j \in \text{scene}
$$

每个 $\mathbf{k}$ 是 LLM 选出的 keypoint (e.g. cup handle、 table corner)， $\mathcal{C}_{ij}$ 可以是 distance、 angle、 alignment constraint。 这些 constraint 喂给 nonlinear optimizer (e.g. IPOPT) 求 trajectory， 实现"用 LLM 提供约束、 用 optimizer 求 motion"。

GeoManip ([Tang et al. arXiv:2501.09783](https://arxiv.org/abs/2501.09783)) 把 geometric constraint 显式作为 LLM 与 controller 之间的 interface。 DiffusionSeeder ([Huang et al. CoRL 2025, arXiv:2403.11875](https://arxiv.org/abs/2403.11875)) 用 diffusion model 生成 motion seed、 再用 trajectory optimizer polish， 体现了 diffusion + classical optimization 的混合范式。

### 2.5 Affordance as Planner

Affordance 概念来自 Gibson ([1979](https://en.wikipedia.org/wiki/Affordance))， 指物体"能提供何种 action"。 论文从四个视角分类， 这部分我觉得是 high-level planner 中最直觉的：

**(a) Geometric affordance**。 物体功能由 3D shape 与 kinematics 决定。 Ditto ([Jiang et al. CVPR 2022, arXiv:2204.02988](https://arxiv.org/abs/2204.02988)) 通过 physical interaction 恢复 articulation model (joint axis、 range)。 GAPartNet ([Geng et al. CVPR 2023, arXiv:2211.01020](https://arxiv.org/abs/2211.01020)) 提出 cross-category part taxonomy： 抽屉的 handle 与门的 handle 共享 "pull handle" affordance， 即使物体类别不同。 CPM ([Liu et al. CoRL 2023, arXiv:2309.15768](https://arxiv.org/abs/2309.15768)) 把 manipulation skill 表达为 geometric constraint 的组合， 而不是 monolithic action。

**(b) Visual affordance**。 从 2D image 直接预测 pixel-wise affordance map。 Transporter Networks ([Zeng et al. CoRL 2021, arXiv:2010.14407](https://arxiv.org/abs/2010.14407)) 用 spatially equivariant feature matching：

$$
\text{pick}(u,v) = \arg\max_{u', v'} \langle f_\theta(I)(u,v), f_\theta(I)(u', v') \rangle
$$

变量： $f_\theta(I)$ 是 image feature map， 第一步 pick 位置 $(u,v)$ 是 $\arg\max$， 第二步 place 位置由 feature matching 给出， 整个过程是全卷积、 可 spatial-equivariant。 VAPO ([Borja-Diaz et al. ICRA 2022, arXiv:2205.10709](https://arxiv.org/abs/2205.10709)) 从 play data 自监督学 affordance map， 大大降低 expert demonstration 依赖。

**(c) Semantic affordance**。 早期 affordance-based IL ([Lopes et al. IROS 2007](https://ieeexplore.ieee.org/document/4399013)) 把 object part name (handle、 lid) 与 trajectory 关联， 跨物体泛化。 这种范式被 foundation model 推广为 CLIPort ([Shridhar et al. CoRL 2022, arXiv:2109.12098](https://arxiv.org/abs/2109.12098))：

$$
\text{action} = \text{Transporter}(I, \text{CLIP}(L))
$$

把 semantic reasoning (CLIP language branch) 与 spatial localization (Transporter spatial branch) 解耦， 这是非常 successful 的设计 pattern。 后续 part-level CLIPort ([PartInstruct, RSS 2025, arXiv:2406.01281](https://arxiv.org/abs/2406.01281); [SAGE, RSS 2024, arXiv:2404.07976](https://arxiv.org/abs/2404.07976)) 把它细化到 part level。

**(d) Multimodal affordance**。 融合 vision、 language、 3D、 tactile。 RoboPoint ([Yuan et al. CoRL 2025, arXiv:2406.10721](https://arxiv.org/abs/2406.10721)) 是 VLM 输出空间点用于 spatial affordance prediction， UAD ([Tang et al. ICRA 2025](https://arxiv.org/abs/2410.08168))、 RAM ([Kuang et al. CoRL 2025](https://arxiv.org/abs/2404.01489)) 探索跨 object/task transferable affordance representation。

### 2.6 3D Representation as Planner

这一节我特别想强调， 因为它直接 build intuition for "mid-level abstraction" 这个 concept。

**Gaussian Splatting for manipulation**。 Splat-MOVER ([Shorinwa et al. CoRL 2025, arXiv:2405.18028](https://arxiv.org/abs/2405.18028)) 把 open-vocabulary semantic distilled 进 3D Gaussian Splatting scene， 每个高斯 $G_i = (\mu_i, \Sigma_i, \alpha_i, c_i, \ell_i)$， $\mu_i$ 是位置、 $\Sigma_i$ 协方差、 $\alpha_i$ opacity、 $c_i$ color、 $\ell_i$ 是 semantic label。 它直接在 3D 上 propose grasp candidate， 而不需要在 2D 做 grounding。 RoboSplat ([Yang et al. RSS 2025](https://arxiv.org/abs/2410.18914)) 通过直接 manipulate 重建 scene 合成多样化 demo， 解决 one-shot generalization 问题。

**Implicit Descriptor Field**。 NDF ([Simeonov et al. ICRA 2022, arXiv:2112.05124](https://arxiv.org/abs/2112.05124))：

$$
\mathcal{F}_\theta: \mathbb{R}^3 \to \mathbb{R}^d, \quad \text{SE(3)-equivariant}
$$

给定物体 $O$， 对任意 pose $T \in \text{SE}(3)$ 与 3D point $x$， 都有 $\mathcal{F}_\theta(T x) = T \mathcal{F}_\theta(x)$ (特征域随 pose 变换)。 这让 few-shot pose transfer 直接通过 descriptor matching 实现： $\hat{T} = \arg\min_T \|\mathcal{F}_\theta(T x_{\text{obs}}) - \mathcal{F}_\theta(x_{\text{demo}})\|$。

F3RM ([Shen et al. CoRL 2023, arXiv:2308.07931](https://arxiv.org/abs/2308.07931)) 把 CLIP feature distill 进 3D field， 实现 language-conditioned grasping。 D³Fields ([Wang et al. CoRL 2025, arXiv:2409.17065](https://arxiv.org/abs/2409.17065)) 推广到 dynamic scene， 支持 image-specified goal 的 zero-shot rearrangement。 Imagination Policy ([Huang et al. CoRL 2025, arXiv:2410.10773](https://arxiv.org/abs/2410.10773)) 把 action inference 当作"先 imagine 目标 point cloud、 再对齐 observed geometry"的 local generative process。

**直觉**： 3D representation 当 planner 的关键， 是它**几何 fidelity 与 semantic grounding 兼具**。 2D VLA 在 spatial reasoning 上的 failure， 很大程度源于 2D 输入无法直接表达 6DoF action； 3D field 直接把 action 落在 SE(3) 空间里。

---

## 3. Low-Level Learning-Based Control

论文把 low-level 拆成 Learning Strategy / Input Modeling / Latent Learning / Policy Learning， 这是它的核心 contribution。 我重点讲前两个， 然后深入 Latent Action 与 Policy Learning。

### 3.1 Learning Strategy

**RL 路径**。 详见表格 I。

| Category | Subcategory | Representative |
|---|---|---|
| Model-Free RL | Pre-training | QT-Opt, PTR, V-PTR |
| Model-Free RL | Fine-Tuning | Residual RL, RLDG, V-GPS, PA-RL |
| Model-Free RL | VLA-RL | iRe-VLA, RIPT, VLA-RL, ConRFT |
| Model-Based RL | Imagination | Dreamer, MWM |
| Model-Based RL | Planning | GPS, TD-MPC |
| Model-Based RL | Differentiable | SAPO, SAM-RL, DiffTORI |

QT-Opt ([Kalashnikov et al. arXiv:1806.10293](https://arxiv.org/abs/1806.10293)) 用 distributed Q-learning $Q(s,a) \leftarrow r + \gamma Q(s', a')$ 在 580k grasp 试做 vision-based grasping。 PTR ([Kumar et al. arXiv:2210.05178](https://arxiv.org/abs/2210.05178)) 在大规模 offline 数据上预训练 value function 与 visual representation， 然后 few-shot adapt 到新 task。

Model-based RL 的典型 Dreamer ([Hafner et al. ICLR 2020, arXiv:1912.01603](https://arxiv.org/abs/1912.01603)) 用 RSSM (Recurrent State-Space Model)：

$$
\begin{aligned}
h_t &= f_\phi(h_{t-1}, s_{t-1}, a_{t-1}) \\
\hat{s}_t &\sim q_\phi(\hat{s}_t \mid h_t, o_t) \quad \text{posterior} \\
\tilde{s}_t &\sim p_\phi(\tilde{s}_t \mid h_t) \quad \text{prior} \\
\hat{o}_t &\sim p_\phi(\hat{o}_t \mid \hat{s}_t), \quad \hat{r}_t \sim p_\phi(\hat{r}_t \mid \hat{s}_t)
\end{aligned}
$$

变量： $h_t$ 是 deterministic recurrent state， $\hat{s}_t$ 是 stochastic latent (posterior 在观察后、 prior 在 rollout 用)， $\hat{o}_t, \hat{r}_t$ 是 reconstructed observation/reward。 Policy 在 latent space 做 imagination rollout： $\pi_\theta(a \mid \hat{s})$ 通过 actor-critic 在 imagined trajectory 上学。 TD-MPC ([Hansen et al. ICML 2022, arXiv:2204.07105](https://arxiv.org/abs/2204.07105)) 联合学 dynamics、 value、 policy， 在 inference time 用 MPC：

$$
\tau^* = \arg\max_{a_{0:H}} \sum_{t=0}^{H} \gamma^t Q_\theta(s_t, a_t), \quad s_{t+1} = T_\theta(s_t, a_t)
$$

DayDreamer ([Wu et al. CoRL 2023, arXiv:2206.14176](https://arxiv.org/abs/2206.14176)) 把 Dreamer 落地到真机。

VLA-RL (iRe-VLA [arXiv:2501.16664](https://arxiv.org/abs/2501.16664), RIPT [arXiv:2507.19364](https://arxiv.org/abs/2507.19364), VLA-RL [arXiv:2505.18719](https://arxiv.org/abs/2505.18719), ConRFT [arXiv:2502.05450](https://arxiv.org/abs/2502.05450)) 是把 RL 当 VLA post-training 信号。 SimpleVLA-RL ([arXiv:2412.03481](https://arxiv.org/abs/2412.03481)) 用 verifiable outcome-based reward 与 group-level policy gradient (类 GRPO 思路)：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{g \sim \mathcal{G}, \{a_i\}_{i=1}^G \sim \pi_\theta(\cdot|g)} \frac{1}{G}\sum_i \tilde{r}_i \log \pi_\theta(a_i | g)
$$

变量： $g$ 是 instruction group, $G$ 是 group size, $\tilde{r}_i$ 是 advantage-normalized reward, $\pi_\theta(a_i|g)$ 是 VLA 的 token-level action probability。 这种 RLHF-style post-training 让 VLA 从 IL 的"copy expert" 升级到 "optimize outcome"。

**IL 路径**。 BC 仍然是最广泛的形式：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o, a) \sim \mathcal{D}_{\text{expert}}} \| a - \pi_\theta(o) \|^2
$$

问题： compounding error， 长程任务误差累积。 解法 hierarchical / action chunking / trajectory-level supervision。 ACT ([Zhao et al. RSS 2023, arXiv:2304.13705](https://arxiv.org/abs/2304.13705)) 用 transformer 一次预测 $k$ 步 action chunk， 减轻每步误差。 Pose-level IL ([Sun et al. arXiv:2505.09424](https://arxiv.org/abs/2505.09424)) 直接预测 SE(3) end-effector pose， 让低层 controller 处理 joint-level execution， 对 insertion 类任务特别有效。

Reward-based IL 包括 Inverse RL (从 demo 推 cost function via KKT 条件 [Englert et al. IJRR 2017](https://journals.sagepub.com/doi/10.1177/0278364917720293))、 GAIL ([Ho & Ermon NeurIPS 2016, arXiv:1606.03476](https://arxiv.org/abs/1606.03476)) 用 discriminator $D_\phi$ 区分 expert 与 policy sample：

$$
\min_\pi \max_\phi \mathbb{E}_{\text{expert}}[\log D_\phi(o, a)] + \mathbb{E}_{\pi}[\log(1 - D_\phi(o, a))]
$$

LfO (Imitation from Observation) 去掉 action label， 只用 visual trajectory： VIP ([Ma et al. ICLR 2023, arXiv:2210.03011](https://arxiv.org/abs/2210.03011)) 用 time-contrastive objective 学跨 embodiment transferable representation。 Differentiable physics-based LfO ([Chen et al. CVPR 2023](https://arxiv.org/abs/2305.04330)) 用 differentiable simulator 把 video trajectory 投影到 contact-consistent trajectory。

### 3.2 Input Modeling

这是把 perception 与 policy 耦合的最核心 axis。 论文按 modality 分四类： V-A、 VLA、 Tactile-based、 Extra modalities。

#### 3.2.1 Vision-Action Models

**2D vision**： Diffusion Policy ([Chi et al. IJRR 2023, arXiv:2303.04137](https://arxiv.org/abs/2303.04137)) 是这一类的 milestone。 它把 visuomotor control 表达为 conditional denoising：

$$
a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( a_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(a_t, t, o) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

变量详解： $a_t$ 是 noise level $t$ 下的 noisy action chunk (通常是 $T_a = 16$ 步的 7-DoF end-effector action)， $\alpha_t = 1 - \beta_t$、 $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ 是 noise schedule， $\epsilon_\theta$ 是 UNet / transformer 参数化的 noise predictor， $o$ 是 conditioning observation (image feature、 robot proprioception)， $\sigma_t$ 是方差项 (DDPM 用 $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$， DDIM 用 0)， $z$ 是标准高斯噪声。 整个过程从 $a_T \sim \mathcal{N}(0, I)$ 出发， $T$ 步迭代得到 $a_0$。 关键 property： 输出 distribution 是 multimodal， 因为同一 observation 可能有多个合理 action (e.g. 杯子可以从左、 右两侧 grasp)， 这是 Gaussian BC 无法表达的。

HDP ([Wang et al. TRO 2025](https://arxiv.org/abs/2410.01219)) 把 Diffusion Policy 分层： 高层生成 keypoint、 低层跟踪。 HPT ([Wang et al. NeurIPS 2024, arXiv:2409.20537](https://arxiv.org/abs/2409.20537)) 在不同 embodiment 间对齐 vision 与 proprioception， 关键是建立 heterogeneous input → shared latent → heterogeneous output 的接口。

**3D vision**： RVT ([Goyal et al. CoRL 2023, arXiv:2310.05748](https://arxiv.org/abs/2310.05748))、 RVT-2 ([Goyal et al. RSS 2024, arXiv:2406.09158](https://arxiv.org/abs/2406.09158)) 用 multi-view transformer 推断 3D action target。 DP3 ([Ze et al. RSS 2024, arXiv:2403.03954](https://arxiv.org/abs/2403.03954)) 把 point cloud 编码进 diffusion process：

$$
\epsilon_\theta(a_t, t, \text{PointNet}(P))
$$

其中 $P \in \mathbb{R}^{N \times 3}$ 是 scene point cloud， PointNet 输出全局 feature。 GenDP ([Wang et al. CoRL 2024, arXiv:2409.20508](https://arxiv.org/abs/2409.20508)) 重建 3D semantic field 作为 conditioning， 实现 category-level generalization。

#### 3.2.2 VLA Models — 论文核心

Figure 5 的 taxonomy 按 (2D vs 3D) × (model-oriented vs model-agnostic) 四象限组织。 我把主要方法画成一张 mental map：

```
                    Model-Oriented                 Model-Agnostic
              +-------------------------------------------+-------------------------+
   2D         | Non-LLM VLA: RT-1, VIMA, HULC, RoboBERT   | Inference-time:        |
              | LLM/VLM VLA: RT-2, OpenVLA, π0, π0.5       |   RoboMonkey, CronusVLA|
              | Latent VLA: UniVLA, villa-X, AgiBot-GO1   | RL post-training:      |
              | Quantized: VQ-VLA, MiniVLA, FAST          |   SimpleVLA-RL, VLA-RL, |
              | Dual/Multi-system: LCB, HiRT, RationalVLA,|   RIPT, ConRFT          |
              |   HiRobot, OpenHelix, TriVLA, G0          | Auxiliary task: CoT-VLA,|
              |                                           |   TraceVLA, ReconVLA   |
              |                                           | Efficiency: TinyVLA,   |
              |                                           |   VLA-Cache, CEED-VLA  |
              +-------------------------------------------+-------------------------+
   3D         | 3D Embedding: SpatialVLA, GeoVLA, FP3,    | View selection,         |
              |   RoboMM                                  | feasibility filter      |
              | Spatial Alignment: BridgeVLA, LtS&A       | (mostly unexplored)    |
              | 3D World Model: 3D-VLA, Evo-0,            |                         |
              |   ChainedDiffuser, SGR                    |                         |
              +-------------------------------------------+-------------------------+
```

**2D model-oriented** 详解：

Non-LLM VLA 起点 RT-1 ([Brohan et al. RSS 2023, arXiv:2212.06817](https://arxiv.org/abs/2212.06817))： transformer 把 image+language token encode， 输出 discretized action token (每个 DoF 离散化成 256 bin)。 VIMA ([Jiang et al. ICML 2023, arXiv:2210.03094](https://arxiv.org/abs/2210.03094)) 用 multimodal prompt 支持组合指令。 HULC ([Mees et al. RAL 2022, arXiv:2207.07850](https://arxiv.org/abs/2207.07850)) 在 unstructured data 上做 language-conditioned IL。

LLM/VLM-based VLA： RT-2 ([Zitkovich et al. CoRL 2023, arXiv:2307.15818](https://arxiv.org/abs/2307.15818)) 把 robot action 当作 text token， 与 web-scale VLM co-train， 实现 zero-shot 与 cross-embodiment transfer。 OpenVLA ([Kim et al. CoRL 2025, arXiv:2406.09246](https://arxiv.org/abs/2406.09246)) 用 frozen VLM (Llama-2 + CLIP) + lightweight action head， 开源 7B 模型。 π0 ([Black et al. RSS 2025, arXiv:2410.24164](https://arxiv.org/abs/2410.24164)) 用 PaliGemma backbone + flow-matching action expert， 是目前 strongest 的 generalist VLA 之一。 π0.5 ([arXiv:2504.16054](https://arxiv.org/abs/2504.16054)) 进一步做 open-world generalization。

Latent VLA： UniVLA ([Bu et al. RSS 2025, arXiv:2503.21678](https://arxiv.org/abs/2503.21678)) 在 visual feature space 学 language-conditioned latent action， 再 decode 成 embodiment-specific control。 AgiBot-GO1 ([arXiv:2503.12445](https://arxiv.org/abs/2503.12445)) 在 vision-language backbone 与 low-level controller 间插入 latent action planner， 让 heterogeneous data (human+robot) 共训。

Action Quantization： VQ-VLA ([Wang et al. ICCV 2025, arXiv:2412.19258](https://arxiv.org/abs/2412.19258)) 把连续 action 用 VQ-VAE-style codebook 量化：

$$
z = \arg\min_{k} \|x - e_k\|_2, \quad e_k \in \mathcal{C} \subset \mathbb{R}^d
$$

变量： $x$ 是 continuous action vector， $e_k$ 是 codebook entry， $z$ 是 codebook index (token)。 这样 continuous control → discrete token， 与 LLM 接口对齐， 但需要警惕 codebook collapse (大部分 code 不被使用)。 FAST ([Pertsch et al. arXiv:2501.09747](https://arxiv.org/abs/2501.09747)) 把 action 映到 frequency domain 的 token (DCT-based compression)， 提高长程稳定性。

Dual/Multi-system VLA： 受 Kahneman System 1/System 2 启发。 LCB ([Shentu et al. IROS 2024, arXiv:2407.20611](https://arxiv.org/abs/2407.20611)) 用 latent code 把 LLM (System 2) 与 fast policy (System 1) 桥接。 HiRT ([Zhang et al. CoRL 2025, arXiv:2410.05255](https://arxiv.org/abs/2410.05255))、 RationalVLA ([Song et al. arXiv:2506.10826](https://arxiv.org/abs/2506.10826))、 HiRobot ([Shi et al. arXiv:2502.01138](https://arxiv.org/abs/2502.01138))、 OpenHelix ([Cui et al. arXiv:2505.03912](https://arxiv.org/abs/2505.03912))、 TriVLA ([Liu et al. ICCV 2025, arXiv:2410.07017](https://arxiv.org/abs/2410.07017))、 G0 ([Jiang et al. arXiv:2509.00576](https://arxiv.org/abs/2509.00576)) 是这条线。 关键 design： System 2 在 slow timescale 给出 abstract plan/latent， System 1 在 fast timescale 执行。 用符号表示：

$$
\underbrace{a_{\text{fast}}^{(t)} = \pi_{\text{fast}}(o_t, \ell)}_{\text{System 1, ~50 Hz}}, \quad \underbrace{\ell = \pi_{\text{slow}}(o_{t-\Delta}, L)}_{\text{System 2, ~5 Hz}}
$$

变量： $\ell$ 是 latent intent， $\Delta$ 是 System 2 触发周期， $L$ 是 language instruction。 这套结构的好处： System 2 处理 high-bandwidth reasoning、 System 1 处理 high-bandwidth control， 各司其职； 难点是 arbitration (何时 handoff)、 credit assignment (失败算谁的)、 consistency (两 system 是否一致)。

**2D model-agnostic**：

- Inference-time： RoboMonkey ([Kwok et al. CoRL 2025, arXiv:2502.11830](https://arxiv.org/abs/2502.11830)) 多次采样 + voting； CronusVLA ([arXiv:2502.15835](https://arxiv.org/abs/2502.15835)) 做 calibration。
- RL post-training： 见 §3.1 表格。
- Auxiliary： CoT-VLA ([Zhao et al. CVPR 2025, arXiv:2412.09669](https://arxiv.org/abs/2412.09669)) 在 action 前输出 reasoning chain； TraceVLA ([Zheng et al. ICLR 2025, arXiv:2412.02416](https://arxiv.org/abs/2412.02416)) 用 visual trace prompt 增强 spatial-temporal awareness； ReconVLA ([Song et al. AAAI 2026, arXiv:2506.06436](https://arxiv.org/abs/2506.06436)) 加 reconstruction objective 强制 perception-action consistency。
- Efficiency： TinyVLA ([Wen et al. RAL 2025, arXiv:2409.15814](https://arxiv.org/abs/2409.15814))、 VLA-Cache ([arXiv:2505.12781](https://arxiv.org/abs/2505.12781)) adaptive token caching、 CEED-VLA ([arXiv:2506.13725](https://arxiv.org/abs/2506.13725)) consistency + early-exit decoding。

**3D VLA**： SpatialVLA ([Qu et al. RSS 2025, arXiv:2501.09651](https://arxiv.org/abs/2501.09651)) 注入 3D positional encoding 与 adaptive action grid。 GeoVLA ([Sun et al. arXiv:2508.09071](https://arxiv.org/abs/2508.09071)) 用 point-based geometric embedding。 FP3 ([Yang et al. arXiv:2503.08950](https://arxiv.org/abs/2503.08950)) 在 large-scale 3D 数据上 train foundation policy。 BridgeVLA ([Li et al. NeurIPS 2025, arXiv:2412.15235](https://arxiv.org/abs/2412.15235)) 用 cross-view heatmap prediction 对齐 point cloud。 3D-VLA ([Zhen et al. ICML 2024, arXiv:2405.15874](https://arxiv.org/abs/2405.15874)) 用 language-guided point cloud diffusion 预测未来 scene state。

#### 3.2.3 Tactile-based Action Models

Figure 6 把 tactile 路径分为： tactile latent (CLTP [arXiv:2505.08194](https://arxiv.org/abs/2505.08194), Sparsh [arXiv:2411.02479](https://arxiv.org/abs/2411.02479])、 tactile-action (Feel the Force, RoboPack)、 tactile-vision-action (T-DEX, RotateIt, VTTB, VITaL, Reactive Diffusion Policy)、 tactile-language-action (TLA, Octopi)、 tactile-vision-language-action (Tactile-VLA, VTLA, Touch Begins)。

Sparsh 用 large-scale self-supervised visuotactile pretraining 学 generalizable tactile embedding， 是这一类的 backbone 候选。 Tactile-VLA ([arXiv:2507.09160](https://arxiv.org/abs/2507.09160)) 把 tactile 加进 VLA pipeline， 改善 contact-rich 任务的 generalization。 VTLA ([arXiv:2505.09577](https://arxiv.org/abs/2505.09577)) 用 preference learning 做 language-conditioned insertion。 Touch Begins ([arXiv:2506.13762](https://arxiv.org/abs/2506.13762)) 提出"vision 做不到时用 tactile 收尾"的两阶段策略： vision language model 做 coarse localization， tactile 做 fine contact execution。

#### 3.2.4 Extra Modalities

Force： Force-aware reactive policy ([arXiv:2501.06419](https://arxiv.org/abs/2501.06419))、 ForceVLA ([arXiv:2505.08150](https://arxiv.org/abs/2505.08150)) 用 force 作为额外 modality 训练 VLA。 Audio： Play it by Ear ([Du et al. RSS 2022](https://arxiv.org/abs/2206.09225))、 Smart Sensory Fusion ([Li et al. CoRL 2023, arXiv:2310.05453](https://arxiv.org/abs/2310.05453))、 ManiWAV ([Liu et al. CoRL 2025, arXiv:2406.19464](https://arxiv.org/abs/2406.19464)) 在 occlusion 下用 audio 做 imitation。

### 3.3 Latent Learning — 论文另一核心 contribution

这一节其实是 VLA 的"中间层"设计空间。 论文把它分 Pretrained Latent 与 Latent Action 两类。

#### Pretrained Latent Learning

按数据源分三类：

**(i) General image dataset**： Parisi et al. ([ICML 2022](https://arxiv.org/abs/2203.01473))、 Theia ([Shang et al. CoRL 2025, arXiv:2407.00390](https://arxiv.org/abs/2407.00390)) distill 多个 vision foundation model 进 compact representation。 **直觉**： 把 ImageNet-trained encoder 直接拿来， 即使不是 robot-specific， 也比 from-scratch 训练强， 因为 image feature 已经 encode 了 natural image 的几何与 semantic prior。

**(ii) Human egocentric dataset**： VC-1 ([Majumdar et al. NeurIPS 2023, arXiv:2310.18231](https://arxiv.org/abs/2310.18231)) 在 Ego4D 上 masked pretrain。 R3M ([Nair et al. CoRL 2023, arXiv:2203.12601](https://arxiv.org/abs/2203.12601)) 用 time-contrastive + video-language objective。 Voltron ([Karamcheti et al. RSS 2023, arXiv:2302.12766](https://arxiv.org/abs/2302.12766)) 加 language grounding。 HRP ([Srirama et al. RSS 2024, arXiv:2404.12218](https://arxiv.org/abs/2404.12218)) 注入 human interaction affordance prior。

**(iii) Robotic dataset**： RPT ([Radosavovic et al. CoRL 2023, arXiv:2212.06574](https://arxiv.org/abs/2212.06574)) mask sensorimotor pretrain， Premier-TACO ([Zheng et al. ICML 2024, arXiv:2405.15711](https://arxiv.org/abs/2405.15711)) 用 temporal action-driven contrastive loss：

$$
\mathcal{L}_{\text{TACO}} = -\mathbb{E}_{(o_t, a_t, o_{t+k})} \log \frac{\exp(\text{sim}(z_t, z_{t+k}^+))}{\sum_{z^-} \exp(\text{sim}(z_t, z^-))}
$$

变量： $z_t = g_\theta(o_t, a_t)$ 是 action-conditioned representation， $z_{t+k}^+$ 是同 trajectory 未来 frame 的正样本， $z^-$ 是 negative samples。 这让 representation encode action-conditional dynamics。

Manipulation-centric representation ([Jiang et al. ICLR 2025, arXiv:2410.18213](https://arxiv.org/abs/2410.18213)) 显式优化 representation 与 manipulation performance 的对齐， 是 robot-centric pretrain 的最新方向。

#### Latent Action Learning

把"action abstraction"本身作为学习目标。 分 discrete 与 continuous 两类。

**(a) Discrete / VQ latent action**。 ILPO ([Edwards et al. ICML 2019](https://arxiv.org/abs/1905.09058))、 LAPO ([Schmidt & Jiang ICLR 2024, arXiv:2310.08748](https://arxiv.org/abs/2310.08748)) 从 observation-only demo 用 inverse dynamics 推 latent action structure。 LATENT Action (LAPO) 核心： 给 video $o_{1:T}$， 学 $z_t$ 使 $o_{t+1} \approx \text{dec}(z_t, o_t)$， 用 information bottleneck 让 $z$ 紧凑。 LAPA ([Ye et al. ICLR 2024, arXiv:2410.11758](https://arxiv.org/abs/2410.11758)) 把 latent action 与 VLM 接通： 先从 visual transition 学 latent action codebook， 再让 VLM 直接 predict latent action token。 DreamGen ([Jang et al. arXiv:2505.12705](https://arxiv.org/abs/2505.12705)) 用 video world model + inverse dynamics 合成大规模 robotic dataset。

**(b) Continuous latent action**。 MimicPlay ([Wang et al. CoRL 2023, arXiv:2302.12238](https://arxiv.org/abs/2302.12238))：

$$
\ell_t = g_\phi(o_t, o_{\text{goal}}), \quad a_t = \pi_\theta(o_t, \ell_t)
$$

从 human play data 学 latent action plan $\ell_t$， 再用 robot proprioception data 学 low-level decoder， 实现 long-horizon imitation without long-horizon demos。 CLAM ([Liang et al. arXiv:2505.04999](https://arxiv.org/abs/2505.04999)) 从 unlabeled demo 推 continuous latent action。 CoMo ([arXiv:2406.19040](https://arxiv.org/abs/2406.19040)) 从 internet video 学 latent motion embedding。

**Implicit world model**： VPP ([Hu et al. ICML 2025, arXiv:2411.14802](https://arxiv.org/abs/2411.14802)) 把 video foundation model adapt 到 manipulation， 把 predicted visual latents 聚合 conditioning diffusion policy：

$$
\hat{o}_{t+k} = V_\phi(o_t, a_t), \quad a_t \sim \pi_\theta(a_t | o_t, \{\hat{o}_{t+k}\}_{k=1}^K)
$$

变量： $V_\phi$ 是 video predictor， $\hat{o}_{t+k}$ 是预测未来 visual latent， $\pi_\theta$ 是条件 diffusion policy。 FLARE ([arXiv:2505.15659](https://arxiv.org/abs/2505.15659))、 Genie Envisioner ([arXiv:2508.05635](https://arxiv.org/abs/2508.05635)) 是同类工作。

**Latent diffusion policy**： LAD ([arXiv:2506.14608](https://arxiv.org/abs/2506.14608)) 在 shared latent action space 训 diffusion policy 实现 cross-embodiment transfer。 KOAP ([Bi et al. ICRA 2025](https://arxiv.org/abs/2411.12828)) 用 diffusion planner + Koopman-based controller 做 long-horizon execution。 LaDi-WM ([arXiv:2505.11528](https://arxiv.org/abs/2505.11528)) 用 latent diffusion world model 预测 semantic 与 geometry conditioning action generation。

**Koopman-based latent dynamics**： KoDex ([Han et al. CoRL 2023](https://proceedings.mlr.press/v229/han23a.html))、 KOROL ([Chen et al. CoRL 2024](https://arxiv.org/abs/2406.13821)) 学 Koopman representation， 把非线性 manipulation dynamics 抬到 linear latent space：

$$
g(f(x)) = A g(x), \quad g: \mathcal{X} \to \mathbb{R}^d
$$

变量： $f$ 是 nonlinear dynamics， $g$ 是 lifting function (encoder)， $A \in \mathbb{R}^{d \times d}$ 是 linear operator。 这让 long-horizon rollout 可以用 closed-form 矩阵幂 $A^k$ 计算， 大大简化 planning。

**Goal/instruction-conditioned**： Procedure Cloning ([Yang et al. NeurIPS 2022, arXiv:2210.00715](https://arxiv.org/abs/2210.00715)) 用 chain-of-thought style latent procedural abstraction。 UniVLA 在 cross-embodiment 数据上无监督学 task-centric latent action token。

### 3.4 Policy Learning — action 生成形式

| Policy | 数学形式 | Key 方法 |
|---|---|---|
| MLP | $a = \text{MLP}_\theta(z)$ | R3M, RPT |
| Transformer (autoregressive) | $a_{<t} \to a_t$ | ICRT, CARP |
| Transformer (action chunking) | $a_{1:k} = \text{Transformer}(z)$ | ACT, BAKU |
| Diffusion | iterative denoising | Diffusion Policy, DP3, 3D Diffuser Actor |
| Flow Matching | ODE transport | π0, RTC |
| State-space (Mamba) | linear recurrence | MAIL |
| Frequency | spectral token | FreqPolicy |

**Transformer policy**： ACT 的核心是 CVAE + transformer：

$$
\mathcal{L}_{\text{ACT}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}, \quad \mathcal{L}_{\text{recon}} = \|\hat a_{1:k} - a_{1:k}^*\|^2
$$

变量： $\hat a_{1:k}$ 是预测的 $k$-step action chunk， $\beta$ 是 KL weight。 Action chunking 让每步 inference 只 forward 一次 transformer， 大幅降低 jitter。

ICRT ([Fu et al. ICRA 2025, arXiv:2502.20532](https://arxiv.org/abs/2502.20532)) 与 CARP ([Gong et al. ICCV 2025, arXiv:2503.12581](https://arxiv.org/abs/2503.12581)) 把 manipulation 当作 next-token prediction over action sequence， 类 LLM in-context imitation， few-shot demo 时直接 context-conditioned。

**Diffusion Policy** 已在 §3.2.1 详解。 主要 extension： 3D-aware (DP3)、 equivariant (Equibot [arXiv:2501.09956](https://arxiv.org/abs/2501.09956))、 mixture-of-experts (Sparse Diffusion Policy [arXiv:2410.20747](https://arxiv.org/abs/2410.20747))、 world model coupling (Unified World Models [arXiv:2501.00370](https://arxiv.org/abs/2501.00370))。

**Flow Matching Policy**： 与 diffusion 用 stochastic SDE 不同， FM 用 deterministic ODE：

$$
\frac{d a_t}{d t} = v_\theta(a_t, t, o), \quad a_0 \sim p_0, \quad a_1 \sim p_{\text{data}}
$$

变量： $v_\theta$ 是 vector field， $a_t$ 是时间 $t$ 上的 action (从 noise distribution $p_0$ transport 到 data distribution $p_{\text{data}}$)， $o$ 是 conditioning。 FM 推理只需 ODE 求解 (Euler 步)， 比 DDPM 的 50-1000 步迭代快， 而且 trajectory 更 smooth， 这对 robot control 很关键。 π0 用 FM 作为 action expert， RTC ([Black et al. arXiv:2506.07339](https://arxiv.org/abs/2506.07339)) 通过 overlap action 生成与 ongoing control 实现 real-time execution。

**直觉**： MLP / Transformer 是"mapping" framework， diffusion / FM 是"generation" framework。 Generation framework 优势是能表达 multimodal action distribution， 这对 manipulation 是 key——同一 observation 可能有多个合理 action。 Generation framework 的代价是 inference latency， FM 主要解决这个问题。

---

## 4. 四个 Core Challenge — 我的解读

### 4.1 Robot Brain

论文列出 (i) 通用架构， (ii) 终生学习， (iii) long-horizon execution "funnel of success"， (iv) smooth motion generation。

**Funnel of success** 这个比喻来自 classical manipulation ([Mason](http://www.cs.cmu.edu/~cmason/papers/papers.html))： high-level planner 给出 wide funnel (允许 deviation)， low-level controller 逐步把 state 收敛进 success region。 现代体系需要 explicit coupling： planner output 要 contain feedback-suitable sub-goal， controller 要 contain recovery mode。

### 4.2 Data Bottleneck & Sim-to-Real

Data flywheel： robot 自主 collect → filter → relabel → retrain。 挑战： 在 noisy heterogeneous trajectory 上做 reliable filtering、 automated labeling、 data valuation。 Sim-to-real： contact-rich / deformable 需要 higher-fidelity simulation、 differentiable simulator ([DiffTORI](https://arxiv.org/abs/2410.05067), SAM-RL) 提供 gradient-informed policy optimization。

### 4.3 Multimodal Physical Interaction

Vision-centric 不够。 需要 tactile、 audio、 proprioception fusion 进 unified temporal-coherent representation， despite 异步 rate 与 noise。 Deformable material (cloth、 cable、 fluid、 granular) 的 state space 是高维、 接触 dynamics 主导。 需要 graph-based 或 field-based object representation、 physics-informed inference、 partial observability 稳定的 learning。

### 4.4 Safety & Collaboration

Intrinsic safety (kinematic/dynamic limit、 force/energy regulation)、 inter-robot safety (predictive coordination)、 human-robot collaboration (intent inference、 shared autonomy)、 fault detection & recovery (monitoring、 safe fallback)。 Hybrid paradigm： learning-based 适应性 + classical control (MPC、 impedance) 在 safety-critical 时 guarantee。

---

## 5. 综合直觉与几个观察

1. **High-level planner 与 low-level controller 的接口是 latent**。 SayCan 接口是 discrete skill， VoxPoser 接口是 3D value map， Code as Policies 接口是 API call， UniVLA 接口是 latent action token。 不同 interface 决定了"什么 information 流过 boundary"。 Latent action 是最有潜力的——它把抽象与执行解耦， 让 high-level 用 semantic-level reasoning、 low-level 用 embodiment-specific execution。

2. **Diffusion / Flow Matching 是 manipulation 的 de-facto policy**。 因为其 multimodality 与 trajectory-level 生成， 在接触丰富任务 (insertion、 deformable) 上比 Gaussian BC 健壮得多。 FM 比 Diffusion 在 latency 与 smoothness 上更优， π0 选择 FM 不是偶然。

3. **VLA 现在分三大流派**： (i) 单 system 大 VLM (OpenVLA, π0)； (ii) Dual-system (LCB, RationalVLA, HiRobot)； (iii) Latent-action bridge (UniVLA, AgiBot-GO1)。 第三流派我认为最有潜力， 因为它显式建模 embodiment gap。

4. **3D representation 与 foundation model 仍未真正 merge**。 大多数 VLM 在 2D image-text 上 pretrain， 没有 intrinsic 3D 理解。 SpatialVLA、 GeoVLA、 F3RM 是 early attempt， 但 standardization 缺失。

5. **Post-training (RL) 是 VLA 下一个 scaling dimension**。 Pre-LLM 时代 RL 主要解决 sample efficiency； 在 VLA 时代 RL 解决"IL 学不出 success-critical behavior"。 SimpleVLA-RL、 VLA-RL、 ConRFT 是开端， 未来 group-relative、 verifiable reward 的设计 pattern 会越来越多。

6. **Tactile 与 audio 是 unexploited frontier**。 现在 vision-VLA 的性能在 contact-rich 任务上 saturated； 下一个突破点是把 tactile representation 像 CLIP 一样预训练好， 让 VLA 能"感觉到"。

参考资料补充阅读：
- [Awesome-Robotics-Manipulation (论文 github)](https://github.com/RayBai/awesome-robotics-manipulation)
- [RT-2 project page](https://robotics-transformer2.github.io/)
- [OpenVLA project page](https://openvla.github.io/)
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [Diffusion Policy project](https://diffusion-policy.cs.columbia.edu/)
- [SayCan project](https://say-can.github.io/)
- [VoxPoser project](https://voxposer.github.io/)
- [ReKep project](https://rekep.github.io/)
- [NDF project](https://yilundu.github.io/ndf/)
- [F3RM project](https://f3rm.github.io/)
- [CLIPort project](https://cliport.github.io/)
- [MimicPlay project](https://mimicplay.github.io/)
- [UniVLA project](https://univla.github.io/)
- [Dreamer project](https://dreamerv3.com/)
- [3D-VLA project](https://blue-taon.github.io/3d-vla/)
- [DayDreamer project](https://daydreamer.github.io/)
- [RoboBrain paper](https://arxiv.org/abs/2507.01432)
- [Gemini Robotics paper](https://arxiv.org/abs/2503.20020)

如果你想就某一条 line (e.g. latent action learning 的 VQ collapse 问题、 dual-system VLA 的 credit assignment、 diffusion policy 的 action chunk vs autoregressive 设计 trade-off) 深入聊， 我可以再展开技术细节与实验对比。
