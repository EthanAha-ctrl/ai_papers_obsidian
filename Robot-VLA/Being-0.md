---
source_pdf: Being-0.pdf
paper_sha256: 0b2fdd057052dc92006861e060d20327aa85b6bf2e6fa776da5d09ff0ae4fee7
processed_at: '2026-07-18T16:02:30-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Being-0 论文深度解析

Andrej, 这篇 paper 我读了几遍,觉得它在 humanoid robot + FM agent 这条线上确实做出了一个非常 concrete 的工程贡献。下面我把整个 system 拆开来给你讲清楚,重点 build intuition。

---

## 1. 核心动机: 为什么 FM + skills 直接拼起来不行?

这篇 paper 一开篇就点出一个很真实的问题: 现在大家都在说 "FM 做 planning + low-level skill library 做 execution = agent", 搁 robot arm 上 (SayCan, Code as Policies) 和 quadruped 上 (Chen et al. 2024) 看起来都 work, 但直接搬到 humanoid 上却挂了。Table 1 里 baseline (w/o Connector) 在三个 long-horizon task 上是 **0.00 / 0.00 / 0.00**, 这个数字其实非常扎眼。

为什么 humanoid 特别难? 作者点出三个 reason, 我觉得第三个最关键:

1. **Bipedal locomotion 天然不稳定**。Wheeled robot 你给它一个 waypoint 它能精确停在那儿, humanoid 走两步 position 就 drift 了, 需要高频闭环修正 joystick command。GPT-4o 一秒几 token 的频率根本跟不上。

2. **GPT-4o 对 3D scene 的 embodied understanding 很差**。Paper 里 Figure 3 给了一个很直观的例子: GPT-4o 看着图说 "turn_left(LARGE)", 结果直接转过 target 了。它判断 depth / direction 经常出错, 导致 navigation plan 本身就错。

3. **Navigation 结束时 pose 不对, 后续 manipulation 直接挂**。这是最 subtle 的点, 也是 paper 核心贡献之一。Figure 5 演示得很清楚: 机器人走到 table 旁边, 但朝向偏了, "grasp cup" 的 manipulation policy (ACT) 是 fixed horizon 的, 它需要从一个 reasonable 的 initial pose 开始, 你给它一个奇怪视角的 cup, 它抓不到。**Long-horizon task 失败的原因不是单个 skill 挂了, 是 skill 之间的 stitching 挂了**。

这三点综合起来, 就引出了 Connector 的设计。

---

## 2. 系统架构: 三层 hierarchy

整体框架:

```
┌─────────────────────────────────────────┐
│  Foundation Model (GPT-4o, cloud)         │  ← High-level planning, reasoning, success detection
│  Input: instruction l, image o^l         │
│  Output: language-based subtask plan      │
└─────────────────────────────────────────┘
                  ↓ (subtask description)
┌─────────────────────────────────────────┐
│  Connector (VLM, onboard Jetson AGX)     │  ← Grounded skill planning, visual navigation, pose adjustment
│  Backbone: VideoLLaMA2 / VideoOrion+     │
│  Input: subtask, binocular images        │
│  Output: skill command (loco/manip)      │
└─────────────────────────────────────────┘
                  ↓ (skill command)
┌─────────────────────────────────────────┐
│  Modular Skill Library (onboard)         │
│  - Locomotion: RL policy π^L, 50 Hz      │
│  - Manipulation: ACT policy π^M, 10 Hz   │
│  - Active neck control (2-DoF)            │
└─────────────────────────────────────────┘
                  ↓ (joint target a = (a^l, a^u, a^h))
            PD controller → 41-DoF Unitree H1-2
```

关键设计理念是 **计算分层 deployment**: cloud 上跑 heavy FM (GPT-4o), onboard Jetson 上跑 lightweight VLM (VideoLLaMA2 量级) + 所有 skills。这样 high-level decision 可以慢 (秒级), 但 reactive control 可以快 (1 秒以内 VLM inference, 10-50 Hz skill control)。这个 latency profile 是 humanoid 能用起来的前提。

---

## 3. Modular Skill Library: decoupling locomotion 和 manipulation

作者做了一个很 pragmatic 的选择: **不强行做 whole-body unified policy** (像 ExBody2、OmniH2O、HumanPlus 那种), 而是把 lower body (locomotion) 和 upper body (manipulation) 拆开训练。理由是现在 RL-based whole-body policy 在 sim-to-real 上做 manipulation 还很脆, 数据也不够; 而 decouple 之后可以各用最合适的训练方法。

### 3.1 Locomotion policy

形式化定义:

$$\pi^L(a^l \mid q^l, q^u, \dot{q}, \omega; v^g)$$

变量解释:
- $a^l$: lower-body joint target action (PD controller 设定值), 维度对应 13-DoF lower body
- $q^l$: lower body joint positions (13 维)
- $q^u$: upper body joint positions (7+7+6+2 = 22 维), 这里作为 context 是为了让 locomotion policy 知道上半身在干什么 (比如提着重物会影响 balance)
- $\dot{q}$: joint velocities
- $\omega$: IMU 给的 root linear + angular velocity, 6 维
- $v^g$: **goal-conditioned joystick command**, 比如 "forward 0.3 m/s, turn left 0.2 rad/s", 这是 skill 的接口

训练在 Isaac Gym (Makoviychuk et al. 2021) 里做 RL, 用 domain randomization (friction, mass, external force) 来 sim-to-real, 50 Hz 控制频率。Locomotion skill library 就是 9 个离散化版本: {no action, go straight, walk backwards, turn left, turn right, sidestep left, sidestep right, tilt head, turn head}。

### 3.2 Manipulation policy

形式化:

$$\pi^{M_i}\bigl([a_j^u, a_j^h]_{j=t}^{t+K} \mid o_t^l, o_t^r, q_t^u, q_t^h\bigr)$$

变量解释:
- $M_i$: 第 $i$ 个 manipulation skill (例如 "grasp_bottle"), 每个对应一个独立训练的 policy
- $a_j^u, a_j^h$: 时刻 $j$ 的 upper body + neck joint target
- $[a_j^u, a_j^h]_{j=t}^{t+K}$: **action chunk**, 预测未来 $K$ 步的 action 序列, 这是 ACT (Zhao et al. 2023) 的核心 trick
- $K$: chunk size, training 时 $K=30$, deployment 时 $K=10$ (inference 时只执行前 10 步, 然后重新 query, 减小 compounding error)
- $o_t^l, o_t^r$: binocular 左右相机 RGB image (ZED-mini)
- $q_t^u, q_t^h$: upper body + neck 当前 joint position

训练数据采集用 Apple Vision Pro teleoperation (Cheng et al. 2024b 的 OpenTeleVision 范式): human 戴 Vision Pro, 看到的就是 robot binocular 投回来的画面, 手部动作通过 retargeting 映射到 robot 7-DoF arm + 6-DoF dexterous hand, 10 Hz 记录 trajectory:

$$\tau = \{(o_t^l, o_t^r, q_t^u, q_t^h, a_t^u, a_t^h)\}_{t=1}^{T}$$

数据规模很轻量: 一个 skill 50~150 条 trajectory, teleoperation 不到 1 小时。Table 8 里 "grasp cup" 用了 200 条, "carry basket" 只用 25 条。这是这个 work scalability 的关键: **加一个新 skill 的成本是 1 小时数据 + 一次 ACT training**, 不需要重训整个系统。

Backbone 是 ResNet-50 (ImageNet pretrained) 处理 binocular images, 然后 ACT Transformer 做 action chunking。Hyperparameters (Table 9): 500k steps, batch 90, lr 1e-5, gradient clip 10。

---

## 4. Foundation Model: Cradle 框架的改造

作者用了 Tan et al. 2024 的 **Cradle** framework (原本是给 open-world game / software control 设计的 GPT-4o agent) 改造成 humanoid agent。FM 每次循环做三件事:

1. **Reasoning**: 描述当前 image + instruction, 判断现在在哪个 stage
2. **Detection**: 评估上一个 skill 是否成功
3. **Planning**: 从 skill library 里选下一个 skill

prompt 设计很讲究, Appendix B.3 里 Table 14/15/16/17/18 给了完整的 prompt。比如 Table 14 的 information gathering prompt 里有一句很经典: "If the target is not in the current image frame, but it has been found previously, use the recent actions and the previous frames to reason about its position, location, and orientation." —— 这是让 FM 做一个 "object permanence" 的 reasoning, 用文字模拟 spatial memory。

但作者发现 FM 直接调 skill library 有三个问题 (上一节讲过), 这就引出了 Connector。

---

## 5. Connector: 这篇 paper 的真正贡献

Connector 的本质是一个 **VLM-based embodied translator**, 把 FM 抽象的语言 plan 翻译成具体 skill command, 同时做高频闭环。这一节是 paper 的灵魂。

### 5.1 VLM 训练

Backbone 用 **VideoLLaMA2** (Cheng et al. 2024c), 但作者实际是从 **VideoOrion+** (Feng et al. 2024, 同一作者团队的 object-centric VLM) checkpoint 起步的, 只去掉了 object-centric branch (为了效率)。Vision encoder 用 **SigLIP** (Zhai et al. 2023, sigmoid loss 的 CLIP 变体), projection 是 MLP。

训练分两阶段:
- **Stage 1**: 先在 300K 通用 visual grounding data (来自 Visual Genome (Krishna et al. 2016), ChatterBox (Tian et al. 2024), Visual-CoT (Shao et al. 2024)) 上 finetune, 让 model 学会 general visual grounding
- **Stage 2**: 在自采的 indoor navigation data 上 finetune, 多任务学习

自采数据 (Table 11):
- Bounding box detection: 14,784 samples
- Yes/no question: 20,536 samples
- Image description: 1,530 samples
- Ground transition detection: 1,530 samples
- Action planning: 771 samples

总共 3,177 张第一人称 indoor image, 但因为多任务标注每张图能衍生多个 sample。Image description 部分是 GPT-4o 先标, 然后 human refine。

训练细节: global batch 128, local batch 2 + gradient accumulation, lr $2 \times 10^{-5}$ cosine schedule, warmup 0.03, AdamW zero weight decay, BF16 + TF32 mixed precision, gradient checkpointing, max seq 4096 tokens, 每个视频 sample 16 frames, 3 epochs。

**关键直觉**: VLM 不是用来取代 FM, 而是把 FM 抽象的 plan (比如 "find a table") 落地成可执行 skill ("move_towards(table)" 或者 "search_for(table)")。FM 还是负责 high-level task decomposition 和 subtask reasoning, Connector 负责 grounded skill selection 和 reactive control。

### 5.2 Grounded skill planning

VLM 能在两个方向修正 FM 的 plan:

- **Downgrade**: FM 说 "grasp a cup", 但 VLM 看到机器人离 table 还远, 它就把 "grasp a cup" 理解成 long-term goal, 输出 "move_towards(table)" 当 immediate skill。这避免了 FM 直接 call 一个不可执行的 manipulation skill。
- **Upgrade**: FM 说 "find a table", 但 VLM 检测到 table 已经在 view 里且距离合适, 它直接 signal "navigation success", 让 FM 跳到下一步 "grasp cup"。这避免了无意义的 search 行为。

这个机制很像 **SayCan (Ahn et al. 2022) 的 affordance grounding**, 但 SayCan 是用 LLM 概率 + affordance value 排序, 这里是直接用 VLM 看 image 做 grounding, 更接近 visual grounding 的范式。

### 5.3 Visual navigation

当 goal object 在 view 内时, VLM 检测出 bounding box, 然后结合 binocular synthetic depth 估算相对位置, 选合适的 locomotion skill:

Algorithm 1 (move_towards) 的核心 loop:

```
for iteration in 1..max_iter:
    (img, depth) ← get_camera()
    bbox ← VLM.detect(target)
    if bbox is None: stop; break
    if depth < threshold: stop; break    # 到达
    if obstacle: sidestep_avoid()
    else:
        if angle to target small: go_straight
        else: turn_towards_target
```

如果 goal 不在 view 里, VLM 触发 **exploration routine** (Algorithm 2, search_for): 持续往一个方向 turn head + body 直到看到 target, 这就是 active vision 的好处 —— 2-DoF neck 能主动扫视。

### 5.4 Pose adjustment (最 subtle 的贡献)

这是 paper 我觉得最 clever 的地方。问题: navigation 到 target 附近后, 机器人朝向可能歪了, 离 manipulation target 也可能不是最佳距离, ACT policy 从这个 weird initial state 开始就抓不到。

VLM 额外预测一个 **optimal alignment direction** —— 机器人应该从哪个角度接近物体。如果当前朝向偏离这个 alignment, Connector 触发一个 **composite skill**: 同时转 head + 向前走, 让机器人沿 **arc-shaped path** 接近 target, 最终落在一个对 manipulation 友好的 pose。

Algorithm 3 (adjustment) 伪代码:

```
for iteration in 1..max_iter:
    set_head_pose(direction)            # 头先看向 alignment 方向
    img ← get_camera()
    bbox, angle ← VLM.detect(target)
    if within_threshold:
        stop, face target; break
    if angle large: turn(direction)
    elif angle small: turn(small)
    elif angle == 0:
        if obstacle: avoid
        else: move_forward
```

这个设计直觉上是: 不要直线冲到 target 然后原地 turn (原地 turn 在 humanoid 上代价大, 不稳), 而是边走边调整, 画一段弧线靠近。这跟人走近桌子拿东西的动作很像。

Table 2 的 ablation 显示这个 adjustment 对 grasp 类 task 帮助巨大:
- Grasp-bottle: 2/5 → 4/5
- Grasp-coffee: 1/5 → 4/5
- Place-coffee (m, 放在咖啡机小面积上): 0/5 → 3/5

而 place-basket 之类的粗放 place 反而对 initial pose 不敏感 (4/5 → 3/5, 略降可能是噪声)。

---

## 6. Active vision: 这个 ablation 我觉得最 dramatic

Table 3 给了一个非常清晰的对比。固定 pitch angle 相机:

| Camera config | Navigation (table) | Navigation (coffee) | Grasp coffee | Place coffee |
|---|---|---|---|---|
| Fixed Cam (0.3) | 5/5 | 5/5 | 0/5 | 0/5 |
| Fixed Cam (0.6) | 0/5 | 0/5 | 2/5 | 1/5 |
| Fixed Cam (0.9) | 0/5 | 0/5 | 4/5 | 5/5 |
| **Being-0 (active)** | **5/5** | **5/5** | **5/5** | **5/5** |

直觉非常清楚: navigation 要看远 (small pitch 0.3), manipulation 要看近 (large pitch 0.9, 看 table 上面的物体)。**任何 fixed pitch 都是一个 trade-off**, 而 active camera 让 robot 在两个任务阶段动态切换视角, 直接两全其美。

这其实给了一个 generalizable 的 design principle: 在 humanoid 上, perception 不应该 passive, head 和 camera 应该作为 action space 的一部分被 agent 主动控制。这点和人类视觉系统一致 —— 我们也是通过 eye movement + head movement 来主动采样视觉信息。

---

## 7. 实验数据

### 7.1 Long-horizon task (Table 1)

平均 completion rate **84.4%**:
- Fetch-bottle: 0.90 (baseline 0.00, 提升 +0.90)
- Deliver-basket: 0.80 (baseline 0.00, +0.80)
- Prepare-coffee: 0.75 (baseline 0.00, +0.75)
- Make-coffee: 0.90 (baseline 0.90, +0.00) ← 这个 task 不需要 navigation, 所以 baseline 也 work
- Deliver-coffee: 0.87 (baseline 0.33, +0.54)

Table 7 给了 sub-process breakdown, 非常 informative。比如 Prepare-coffee 分四步: navigate-to-table (5/5) → grasp-cup (4/5) → navigate-to-coffee-machine (3/5) → place-cup (3/5)。可以看到失败主要发生在后续步骤, 这是 long-horizon compounding error 的典型 pattern。

### 7.2 Efficiency (Table 4)

- w/o Connector: 2.3 cm/s, 0/5 success
- Fixed Cam (0.3): 8.5 cm/s, 5/5
- Being-0: 9.6 cm/s, 5/5, **4.2× speedup**

baseline 慢的原因是 GPT-4o 每次都来一遍 reasoning + planning + detection, 而且经常 plan 错导致原地打转。Connector 的 VLM 在 onboard 上 ~1 秒 inference, 而且 grounding 更准, 不浪费步数。

### 7.3 Navigation robustness (Table 5)

- In-room: 1.00
- In-room with obstacles: 0.80
- Cross-room: 0.83

Cross-room 要求 FM 多步 reasoning (先找出口, 再走到目标 room), 这个 0.83 已经很可观。

### 7.4 Manipulation generalization (Table 6)

| Skill | Seen | Unseen | Perturb |
|---|---|---|---|
| Grasp-bottle | 0.86 | 0.63 | 0.77 |
| Handout-snack | 0.90 | 1.00 | 0.80 |
| Place-pole | 0.90 | - | 0.80 |
| Play-chess* (tactile) | 0.90 | - | 0.90 |

Unseen object 上 grasp-bottle 从 0.86 跌到 0.63, 说明 ACT policy 对 visual distribution 还是比较敏感。但 play-chess (带 tactile sensor 的 dexterous hand) 0.90 → 0.90 under perturbation, 说明 tactile 反馈极大地提升了 closed-loop robustness, 这跟 CHAMP、Open-TeleVision 系列的发现一致。

---

## 8. 相关工作联想

这篇 paper 处于几个 research line 的交汇点, 我帮你 map 一下:

### 8.1 Embodied agent with FM

- **SayCan** (Ahn et al. 2022, https://arxiv.org/abs/2204.01691): 最早把 LLM 当 planner + affordance value 选 skill, 但是 quadruped, 没 VLM 中间层
- **Inner Monologue** (Huang et al. 2022, https://arxiv.org/abs/2207.05608): LLM 做 embodied reasoning + success detection, 但还是 robot arm
- **Code as Policies** (Liang et al. 2023, https://arxiv.org/abs/2209.07753): LLM 直接生成 robot control code, 灵活但缺视觉 grounding
- **Cradle** (Tan et al. 2024, https://arxiv.org/abs/2412.06469): Being-0 直接基于 Cradle 改造, Cradle 原本做 game / software agent

### 8.2 VLA (vision-language-action) models

这条线是 end-to-end 训练 FM 直接输出 action, 跟 Being-0 的 modular 思路截然不同:

- **RT-1** (Brohan et al. 2022, https://arxiv.org/abs/2212.06817), **RT-2** (https://arxiv.org/abs/2307.15818): Google 的 robot transformer 系列
- **OpenVLA** (Kim et al. 2024, https://arxiv.org/abs/2406.09246): 开源 VLA
- **π0** (Black et al. 2024, https://arxiv.org/abs/2410.24164): Physical Intelligence 的 VLA flow model, 用 diffusion + flow matching
- **GR-2** (Cheang et al. 2024, https://arxiv.org/abs/2410.06158): video-language-action, 用 web-scale video 预训练
- **RDT-1B** (Liu et al. 2024, https://arxiv.org/abs/2410.07864): bimanual diffusion foundation model

VLA 路线的好处是 end-to-end, 坏处是 humanoid + dexterous hand + active camera 这个组合没有大规模数据集, 训不出来。Being-0 用 modular skill 路线绕开了这个问题。

### 8.3 Humanoid teleoperation + imitation

- **HumanPlus** (Fu et al. 2024a, https://arxiv.org/abs/2406.10454): Stanford humanoid shadowing
- **OmniH2O** (He et al. 2024a, https://arxiv.org/abs/2406.08858): universal teleoperation
- **OpenTeleVision** (Cheng et al. 2024b, https://arxiv.org/abs/2407.01512): Being-0 直接用这个 VR teleop 方案
- **Mobile ALOHA** (Fu et al. 2024b, https://arxiv.org/abs/2401.02117): bimanual mobile manipulator
- **ACE** (Yang et al. 2024, https://arxiv.org/abs/2408.11805): cross-platform visual-exoskeleton teleop
- **ACT** (Zhao et al. 2023, https://arxiv.org/abs/2304.13705): action chunking transformer, Being-0 manipulation policy 的基础
- **Diffusion Policy** (Chi et al. 2023, https://arxiv.org/abs/2303.04137): 另一个主流 manipulation policy

### 8.4 Humanoid locomotion

- **Real-world humanoid RL** (Radosavovic et al. 2024, https://arxiv.org/abs/2402.19769): Berkeley 的 humanoid walking RL
- **Humanoid parkour** (Zhuang et al. 2024, https://arxiv.org/abs/2406.10759): CMU 的 parkour
- **H1-2 / H1 / H1-2 series**: Unitree 的硬件, Being-0 用的就是 H1-2

### 8.5 Whole-body unified policy (Being-0 的反面)

- **ExBody / ExBody2** (Ji et al. 2024, https://arxiv.org/abs/2412.13196): expressive whole-body control
- **Hover** (He et al. 2024b, https://arxiv.org/abs/2410.21229): versatile neural whole-body controller
- **HumanPlus** 上面提过

这条线想用单一 policy 控制 whole body, Being-0 明确说这条路目前做不出 diverse manipulation skill, 所以 decouple。

### 8.6 VLM backbone 相关

- **VideoLLaMA2** (Cheng et al. 2024c, https://arxiv.org/abs/2406.07476): Being-0 Connector 的 backbone
- **VideoOrion** (Feng et al. 2024, https://arxiv.org/abs/2411.16156): 同组工作, object-centric video understanding
- **SigLIP** (Zhai et al. 2023, https://arxiv.org/abs/2303.15343): sigmoid loss 的 CLIP 变体, vision encoder
- **MiniGPT-v2** (Chen et al. 2023, https://arxiv.org/abs/2310.09478): multi-task VLM 接口
- **BLIP-2** (Li et al. 2023, https://arxiv.org/abs/2301.12597): Q-Former + frozen LLM
- **Qwen-VL** (Bai et al. 2023, https://arxiv.org/abs/2308.12966): 阿里的 VLM
- **Flamingo** (Alayrac et al. 2022, https://arxiv.org/abs/2004.09975): few-shot VLM 鼻祖

---

## 9. 我觉得 paper 的几个 limitation 和开放问题

### 9.1 Connector 训练数据规模偏小

3,177 张 indoor image + 14K bounding box annotation + 20K yes/no, 这在 VLM 时代是非常小的数据。Bbox 检测 19 个 category, yes/no 18 个 category, 都是 indoor scene 特化的。这意味着 Connector 实际上是个 **narrow specialist**, 不是 generalist VLM。换一个 office 场景可能就要重新采数据 finetune。

作者用 300K 通用 visual grounding data 做 stage-1 finetune 是为了缓解这个问题, 但还是远小于 LLaVA、Qwen-VL 这类 VLM 的训练数据量。**怎么把 Connector 做成 zero-shot generalizable** 是个开放问题。

### 9.2 FM 还是瓶颈

Table 4 显示 baseline 速度只有 2.3 cm/s, 主要因为 GPT-4o 每次 reasoning + planning + detection 一轮要好几秒。即使有 Connector 把 navigation 部分加速到 9.6 cm/s, 但 high-level subtask 切换还是要等 GPT-4o。作者在 Conclusion 里也提到这个 limitation, 说未来可以用 lightweight robotic FM 替代。我觉得这是个机会点, 比如 fine-tune 一个 7B 级别的 VLM 替代 GPT-4o 整个流程, 完全 onboard 化。

### 9.3 Manipulation skill 还是个 narrow library

Table 8 列了 10 个 manipulation skill: carry basket, handout snack, grasp bottle, grasp cup, open beer, place basket, place cup, place pole, play chess, play toy bricks。这是一个很 toy 的 set, 跟人类 daily task 的多样性还差几个数量级。而且每个 skill 都是独立 ACT policy, 没法 compose (比如 "拿起篮子放到桌上再从篮子里取杯子" 这种)。ACT 本身的 chunk size K=10 也限制了 temporal horizon, 大概 1 秒量级的 manipulation。

未来方向: (a) 用 diffusion policy 或者 VLA 替代 ACT, 让单 policy 能 multi-task; (b) 把 skill library 扩展到几百个, 用 retrieval + composition; (c) tactile + force feedback integration。

### 9.4 没有 complex locomotion

Paper 明确说目前只做 flat ground walking, 没 crouch / sit / jump / stair climbing。这意味着机器人不能坐在椅子上操作低矮物体, 也不能爬高取东西。这个 limitation 实际上限制了 task horizon 的多样性 —— 比如 "make coffee" 如果咖啡机在矮台上, 现在的 humanoid 只能站着弯腰去够, 而人类会蹲下来。

### 9.5 No dynamic environment

所有 task 都是 static environment (除了 visual perturbation 测试)。如果 task 期间有人走动、物体被别人移动, Connector 的 VLM grounding 可能会失效。作者提到 future work 要做 robust error handling 和 fail-safe, 但目前是个 open problem。

---

## 10. 我的整体评价

这篇 paper 是一个**非常 solid 的 engineering paper**, 不是 algorithm 突破。它的贡献在于:

1. **System decomposition 干净**: FM / Connector / Skill Library 三层职责清晰, deployment 上合理分工 cloud vs onboard
2. **Connector 这个中间层是个非常对的设计**: 既不是 SayCan 那种纯 LLM affordance, 也不是 VLA 那种 end-to-end, 而是 grounded translator + reactive control
3. **Pose adjustment 这个 insight 值得记住**: navigation 和 manipulation 之间不是简单 stop-and-go, 而是要用 arc-shaped approach 落在好 pose 上
4. **Active vision 的 ablation 数据说服力极强**: Table 3 那张表, 任何 fixed pitch 都有 trade-off, active 完美解决, 这是 hardware-aware design 的好例子

但也是 paper 的局限所在 —— 它更像是把现有 components (GPT-4o, VideoLLaMA2, ACT, RL locomotion, Vision Pro teleop) 拼起来做成了一个可演示的系统, 离 general-purpose humanoid agent 还很远。如果未来 1-2 年内 humanoid 数据集 (像 Open X-Embodiment 但 for humanoid + dexterous hand) 起来, VLA end-to-end 路线很可能反超这种 modular 方案。但在数据稀缺的当下, Being-0 这种 modular + 轻量 VLM connector 的设计是非常 pragmatic 的选择。

---

## Reference links

- Being-0 paper (本篇): 项目页通常会有 video, 但 paper 本身 arxiv 链接需要查 (作者 Haoqi Yuan, Zongqing Lu 团队, 应该是 Peking University 的工作, 推测挂在 https://arxiv.org/abs/2507.1250 附近, 但需要核实)
- SayCan: https://arxiv.org/abs/2204.01691
- Inner Monologue: https://arxiv.org/abs/2207.05608
- Code as Policies: https://arxiv.org/abs/2209.07753
- Cradle: https://arxiv.org/abs/2412.06469
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- GR-2: https://arxiv.org/abs/2410.06158
- RDT-1B: https://arxiv.org/abs/2410.07864
- HumanPlus: https://arxiv.org/abs/2406.10454
- OmniH2O: https://arxiv.org/abs/2406.08858
- OpenTeleVision: https://arxiv.org/abs/2407.01512
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- ACE: https://arxiv.org/abs/2408.11805
- ACT: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Real-world humanoid RL: https://arxiv.org/abs/2402.19769
- Humanoid parkour: https://arxiv.org/abs/2406.10759
- ExBody2: https://arxiv.org/abs/2412.13196
- Hover: https://arxiv.org/abs/2410.21229
- VideoLLaMA2: https://arxiv.org/abs/2406.07476
- VideoOrion: https://arxiv.org/abs/2411.16156
- SigLIP: https://arxiv.org/abs/2303.15343
- MiniGPT-v2: https://arxiv.org/abs/2310.09478
- BLIP-2: https://arxiv.org/abs/2301.12597
- Qwen-VL: https://arxiv.org/abs/2308.12966
- Flamingo: https://arxiv.org/abs/2004.09975
- Visual Genome: https://arxiv.org/abs/1602.07332
- ChatterBox: https://arxiv.org/abs/2401.13307
- Visual-CoT: https://arxiv.org/abs/2403.16999
- 3D Diffusion Policy: https://arxiv.org/abs/2403.03954
- OKAMI: https://arxiv.org/abs/2410.11792 (Li et al. 2024a 的 single-video imitation)
- Open X-Embodiment dataset: https://arxiv.org/abs/2310.08864 (相关 context, 虽然不是 humanoid)
- GPT-4 technical report: https://arxiv.org/abs/2303.08774
- Isaac Gym: https://arxiv.org/abs/2108.10470

希望这些分析对你 build intuition 有帮助, Andrej。如果你想深挖某一块 (比如 ACT 的 action chunking 为什么 work, 或者 VLA 路线和 modular 路线的数据 scaling law 比较), 我可以再展开。
