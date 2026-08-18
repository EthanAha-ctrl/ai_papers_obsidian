---
source_pdf: Embodied large language models enable robots to complete complex tasks
  in unpredictable environments.pdf
paper_sha256: c16ad3e724392e278d8256f3956ad48dd963dfefb97000b342921713311612f3
processed_at: '2026-08-18T10:36:04-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 ELLMER

好, 让我换个口吻, 把这篇 paper 嚼碎了讲。

## 一句话概括

这篇 paper 做的事就一句话: **让机器人能用 GPT-4 听人话, 然后真的在物理世界里执行**。听起来好像没那么难, 但你仔细想想, GPT-4 自己连杯子在哪都看不见, 也感受不到水的重量。所以核心问题就是——怎么把 GPT-4 的"脑子"接到一个有眼睛、有手、能感受力的机器人身上。

## 这个问题为什么 hard

你给 GPT-4 一句话: "我累了, 帮我冲杯咖啡"。GPT-4 会 decomposition: 找杯子 → 舀咖啡粉 → 倒热水。这个 plan 本身不难。但接下来呢? "找杯子"这个 action 怎么真的执行? 杯子在哪? 抽屉是推的还是拉的? 倒水倒到多少才停? 如果人突然把杯子挪了怎么办?

这就是 long-horizon task 的 pain。short-horizon 就是单步动作, 比如抓起一个东西; long-horizon 是一连串动作, 前一步影响后一步, 中间任何一个环节出问题整条 chain 就崩了。

RL 和 imitation learning 在 long-horizon 上都不太行——RL 需要大量 episode, imitation learning 需要大量 demo, 而且一旦环境变了, 训练时没见过的 variation 就 GG。这就是 covariate shift problem, 参考这篇 [Chang et al. 2021](https://arxiv.org/abs/2106.15761)。

## ELLMER 的思路

"我有个机器人手臂, 我有个 GPT-4, 我有 camera, 我有 force sensor——我把它们缝起来。"

听起来很朴素, 但 paper 的 contribution 就在这种 system-level 的缝合上。具体怎么缝?

### 第一层缝: GPT-4 output 不是 action, 而是 code

这是最聪明的一个设计决策。早期 LLM-for-robotics 工作让 LLM 直接 output "抓起杯子"这种 natural language action description, 表达力很弱——你没法表达 loop、没法表达 conditional、没法 compose 多个 primitives。

ELLMER 沿着 [Code as Policies](https://arxiv.org/abs/2209.07753) 的思路, 让 GPT-4 output Python code。比如:

```python
mug_pose = vision_system.get_pose("white_mug")
robot.move_to(mug_pose)
robot.grasp(force_limit=5)
robot.move_to(kettle_pose)
robot.pour(target_volume=100ml)
```

code 的好处是什么? 你可以写函数调用, 可以有 control flow, 可以 compose。这就像你给 GPT-4 一堆 LEGO 积木 (knowledge base 里的 motion primitives), 它自己拼一个程序出来。这是 software 2.0 的经典思路——与其让人写规则, 不如让 LLM 写。

### 第二层缝: RAG 给 GPT-4 一个"动作百科"

GPT-4 不知道你这个特定机器人的 API 长什么样、不知道你有哪些 motion primitives 可用、不知道倒水应该用多大的 force。怎么办?

传统做法: 把所有 API 文档塞进 prompt context window。问题: knowledge base 一大, context window 就爆, 而且 GPT-4 在长 context 里 retrieve 信息的能力会下降 (lost in the middle 问题)。

ELLMER 用 RAG 解决。knowledge base 是一个 markdown 文件, 里面写好了各种 motion primitive 的文档和 example code。每次 GPT-4 处理 user query, 系统先从 knowledge base 里 retrieve top-k relevant chunks, 塞进 context, GPT-4 基于这些 examples 写 code。

直觉上, RAG 就是给 LLM 一个"随时可查的说明书"。你问它怎么倒水, 它先去说明书里查"倒水"那一节, 然后基于查到的内容写代码。这比让 LLM 把整本说明书背下来靠谱得多。

paper 给的 faithfulness 数据很有意思:

| Model | 不用 RAG | 用 RAG | 提升 |
|-------|---------|--------|------|
| GPT-4 | 0.74 | 0.88 | +0.14 |
| GPT-3.5 | 0.78 | 0.86 | +0.08 |
| Zephyr-7B | 0.37 | 0.44 | +0.07 |

注意: **RAG 对强 model 的相对提升更大**。这说明弱 model 即使把正确文档喂到嘴边也用不好。所以做 LLM-robot 不要为了省钱用弱 model + RAG, 性价比反而差。

### 第三层缝: Vision + Force dual feedback

这是 ELLMER 真正跟 VoxPoser、RT-2 拉开差距的地方。前面那两个 work 主要用 vision, force 基本没用。

为什么 force 重要? 举个具体例子: 倒水。你用 vision 看水面高度, 但一旦水倒下去, 杯子口被你的 gripper 挡住了, camera 看不见水位。这时候你怎么知道倒了多少? 靠 force sensor——水倒进杯子, 杯子变重, force sensor 测到 upward force 增加, 你就知道倒了多少。

paper 用了一个 quasi-static approximation: $F_{\text{up}} \approx mg$。说白了就是把 force sensor 当体重秤用——杯子+水的总重量 = 向上的支持力 (因为静态平衡)。这个 approximation 在慢速倒水时挺准: pitch velocity 4 m/s 时误差才 5.4%。但快了就不行, 30 m/s 时误差能到 20 g/s, 因为 quasi-static 假设崩了——水在加速运动, 动力学不再简单。

这就是为什么 ELLMER 用 Kinova 七自由度机械臂配 ATI 六轴 force sensor——不是为了炫技, 是因为 vision 单独搞不定需要 fine-grained physical interaction 的任务。

## 系统里那些频率的事

这是 paper 里最容易被忽略但最关键的工程细节:

| 模块 | 频率 |
|------|------|
| Kinova 关节控制 | 40 Hz |
| Robotiq gripper | 50 Hz |
| Force sensor | 100 Hz |
| Vision (Grounding DINO + SAM) | 1/3 Hz |
| Workspace safety check | 10 Hz |

看到没, vision 是 1/3 Hz——**三秒钟才更新一次 object pose**。为什么? Grounding DINO 推理慢。这就是 embodied AI 的 fundamental 挑战: 不同 modality 的 timescale 差好几个数量级。

直觉: 你想想人类, 视觉系统更新率大概 10 Hz (你眼睛 saccade 之间的整合 window), 但 spinal reflex 可以 100 Hz。人类 brain 通过 temporal hierarchy 把这些 timescale 缝起来——spinal cord 管 fast reflex, cortex 管 slow planning。ELLMER 用 ROS 的多节点架构实际上做了一个类似的 temporal abstraction: force node 在 100 Hz 做 fast reflex (比如碰到东西就停), LLM 在 0.1 Hz 做 slow planning (比如"换一种咖啡")。

## 那些 safety clamp

paper 里有一段轻描淡写但极其重要的内容:

- Linear velocity: $\pm 0.05$ m/s
- Angular velocity: $\pm 60°$/s
- Force limit: 20 N
- Workspace bounds: x=[0.0, 1.1], y=[-0.3, 0.3], z=[0, 1.0]

关键句: "This is coded into the fundamental motion primitives; therefore, error in the language model will not override this."

翻译成人话: **不管 GPT-4 hallucinate 出什么鬼代码, 机械臂都不可能超过这些物理 limit**。这是 LLM-driven robotics 的铁律——你必须假设 LLM 会犯错, 然后用 deterministic layer 兜底。这个思路跟 [Bengio et al. 2024 的 AI safety 框架](https://www.science.org/doi/10.1126/science.adn0117) 一致。

## RAG 在 robotics 里的隐喻

paper 里有一句我觉得特别漂亮: "RAG can be seen as providing a cultural milieu of knowledge from which a robot can draw."

这个类比点到了 embodied cognition 的精髓。人类不是所有东西都从头学, 我们用语言和文化把前人积累的 knowledge 传递下来。你不需要自己发明"用勺子舀粉"这个动作——你看别人做一次就会了。RAG 在 ELLMER 里扮演的就是这个 "cultural knowledge" 的载体——knowledge base 里存的是别人写好的 motion primitive, robot 通过 RAG 检索调用, 而不是每次从零学。

这个 framing 把 robotics 从"reinforcement learning 慢慢试错"范式拉到了"通过语言 retrieval 复用文化知识"范式。某种意义上, ELLMER 是在 emulate 人类文明的 knowledge transfer 机制。

## 那个画画的 demo

paper 里提到用 DALL-E 生成 image, 然后从 image silhouette 提取 trajectory, 让机械臂在盘子上画动物。这个 demo 我觉得是 paper 里最 underwhelming 但最 suggestive 的部分。

为什么 underwhelming? 因为 DALL-E 出 silhouette, 然后做 contour extraction, 这个 pipeline 没什么 novel 的。RT-2 那种 VLA model 端到端生成 action trajectory 比这个 elegant 得多。

为什么 suggestive? 因为它指向了一个 future direction: **从 generative model 直接 derive action trajectory**。DALL-E 本来是用来生成 image 的, 但你把 image 的 contour 当成 trajectory, 就完成了一个从 "image generation" 到 "motion generation" 的 semantic shift。未来如果 diffusion policy 或 video generation model 直接 output spatiotemporal trajectory, 就能绕过这个 contour extraction 的 hack。这个方向可以看 [Hierarchical Diffusion Policy](https://arxiv.org/abs/2410.18944)。

## paper 自己承认的 limitation

1. **Affordance map 是 prior knowledge**: kettle/spoon/door handle 的 affordance 是 hand-crafted, 不是 learned。但作者引用了 [one-shot affordance learning](https://arxiv.org/abs/2312.09953) 说这块可以 plug in。
2. **No proactive adaptation**: robot 能 react, 不能 mid-task switch。比如倒水倒一半, 人说"不要了换茶", 系统做不到。这需要更频繁地 poll LLM 做 plan revision。
3. **Complex force dynamics 没建模**: liquid viscosity、container geometry 对 force profile 的影响都 ignore 了, 所以高 pitch velocity 下 accuracy 崩。
4. **Vision latency**: 1/3 Hz 太慢, 跟不上 fast motion。这是 Grounding DINO 推理慢导致的, 不是 fundamental limit, 但短期没法解决。

## 跟我 (Karpathy) 的 software 2.0 框架的关系

你 (我) 一直在讲 software 2.0——用 neural network 替代手写规则。ELLMER 是这个思路的 embodied 延伸: GPT-4 (software 2.0) 生成的 code 调用 force/vision feedback primitives (software 1.5)。这个 1.5 layer 很关键——你不能直接让 LLM output torque command, 那样太低层, LLM 没这种 resolution; 你也不能让 LLM output "倒水"这种 abstract command, 那样太高层, 没法执行。中间这层 motion primitives 就是 sweet spot, 类似 human motor cortex 的 muscle synergies。

所以 ELLMER 的 contribution 不在发明新算法, 而在找到这个 layering 的 sweet spot: **GPT-4 做 planner + code generator, motion primitives 做 executor, force/vision feedback 做 ground truth**。这个架构很可能就是 LLM-driven robotics 的 canonical stack, 后面的人会在细节上优化, 但整体 shape 大概就这样了。

## 你应该带走的核心 intuition

1. **LLM output code, 不是 action**。Code 比 natural language 表达力强几个数量级, 是 LLM-robot 的关键 abstraction。
2. **RAG = robot 的 cultural knowledge**。让 LLM 每次都从 curated knowledge base 里 retrieve, 而不是把所有东西塞进 context window。
3. **Force feedback 不是 optional**。Vision 搞不定 occlusion 和 fine-grained physical interaction, force 是必须的。
4. **Frequency heterogeneity 是 feature 不是 bug**。不同 modality 在不同 timescale 上跑, 用 ROS 的 multi-node 架构天然处理这件事。
5. **Safety 必须 hard-coded**。LLM hallucination 不可避免, deterministic safety layer 是底线。
6. **GPT-4 + RAG > 弱 model + RAG**。RAG 对强 model 的相对提升更大, 别图省钱。

如果想真的 build intuition, 看完 paper 后去看 [Supplementary Video 1](https://github.com/ruaridhmon/ELLMER), 把 video 里 robot 的动作和 Fig. 4 的 force/velocity/position plot 对上。看一次你就明白 force signal 在不同 task 里长得多不一样——drawer 拉开是 xy plane 的 force/torque 大, 杯子放下是 upward force spike, 倒水是 upward force 线性增长。把这些 pattern 看明白了, 你就知道为什么 force feedback 是 ground truth layer。

---

# ELLMER Paper 深度解析

Andrej, 这篇 paper 让我想到你之前在 Eureka Labs 讲的 "software 2.0" 的延伸——但这次是从 cognition 层往下延伸到 sensorimotor 层。让我从 architecture、关键技术细节、以及 intuition 三个层面来拆解。

## 1. 核心问题定位

这篇 paper 的 motivation 其实很尖锐: 传统的 robotic manipulation 和 LLM 是两条 parallel streams:

- **Stream 1 (Sensorimotor)**: RL、imitation learning 擅长 specific motions, 但遇到 novel context 或 distribution shift 就歇菜 (covariate shift problem, reference: [Chang et al. 2021](https://arxiv.org/abs/2106.15761))
- **Stream 2 (LLM cognition)**: VoxPoser、RT-2、PaLM-E 这些 work 把 LLM 接到 robot 上, 但有两个关键缺陷: (1) 缺 force feedback; (2) 缺 RAG, 知识库静态塞进 context window, 不可扩展

ELLMER 的 thesis 是: 把这两个 stream 用一个 unified framework 缝起来, 关键的"胶水"是 **RAG + force/vision dual feedback loop**。这其实是一个系统设计 paper, 而不是新算法 paper——但系统设计本身就值得仔细看。

## 2. 整体架构解析

看 Fig. 1, 系统被一条蓝色虚线分成 high-level 和 low-level:

### High-level (Cognitive Layer)

```
User voice query
    ↓
GPT-4 (with image C from Azure Kinect)
    ↓
[Task decomposition] L → {L1, L2, ..., LN}
    ↓
[RAG retrieval] from curated knowledge base (markdown)
    ↓
Python code generation → sent via JSON to EC2 server
```

关键点: GPT-4 不是直接 output action, 而是 output **Python code**。这个思路明显延续自 [Code as Policies (Liang et al. 2023)](https://arxiv.org/abs/2209.07753) 的 "code for dynamic policies"。Code 比 natural language action description 表达力强太多——可以有 loops、conditionals、可以 compose 多个 primitives。

### Low-level (Sensorimotor Layer)

ROS 节点跑在不同频率上, 这个 frequency heterogeneity 很关键:

| Node | Frequency | Purpose |
|------|-----------|---------|
| Kinova joint update | 40 Hz | Robot control loop |
| Robotiq gripper | 50 Hz | Gripper position control |
| Force sensor (ATI) | 100 Hz | F/τ readings |
| Vision module | ~1/3 Hz | Object pose (慢因为 Grounding DINO 推理慢) |
| Workspace bounds check | 10 Hz | Safety clamp |

直觉: **频率分层本质上是一个 temporal abstraction hierarchy**。Vision 慢得离谱 (1/3 Hz) 是因为 Grounding DINO 推理 latency 大; force 快 (100 Hz) 因为是 raw signal。这个 frequency mismatch 是真实世界 embodied AI 的 fundamental 挑战——不同 modality 的 timescale 不一致, 类似人类 retina 更新率 ~10Hz 但 spinal reflex 可以 ~100Hz。

## 3. 关键技术细节

### 3.1 RAG 部分

paper 用的是 vector RAG, 核心公式 (虽然 paper 没显式写, 但可以从描述 reconstruct):

$$\text{sim}(q, s_i) = \frac{E(q) \cdot E(s_i)}{\|E(q)\| \|E(s_i)\|}$$

其中:
- $q$: user query 的 embedding
- $s_i \in \{s_1, s_2, ..., s_m\}$: knowledge base 中第 $i$ 个 chunk 的 embedding
- $E(\cdot)$: encoder (这里用的是 OpenAI 内置 embedder)
- top-$k$ chunks 被选为 context

paper 测了三种 RAG 实现: Haystack、Verba、Azure Cloud AI, 都 work。最终选了最简单的: markdown 文件 + OpenAI Custom GPT 的 Knowledge feature。这个决策很务实——说明在 robot application 这个 scale 下, RAG 的 sophistication 还不是 bottleneck, knowledge base 的 curation quality 才是。

Faithfulness 评估结果很能说明问题:

| Model | w/o RAG | w/ RAG | Δ |
|-------|---------|--------|---|
| GPT-4 (0613) | 0.74 | 0.88 | +0.14 |
| GPT-3.5-turbo | 0.78 | 0.86 | +0.08 |
| Zephyr-7B-beta | 0.37 | 0.44 | +0.07 |

直觉: RAG 对强 model 的相对提升更大, 因为强 model 能更好地 utilize retrieved context。弱 model 即使 retrieve 到了正确 context 也用不好。这个发现对 productionize LLM-driven robots 有指导意义——不要为了省钱用弱 model + RAG, 性价比反而差。

### 3.2 Task Dependency 形式化

paper 给了一个 conditional probability 表达:

$$P(L_{2A}, L_{2B} | L_1)$$

- $L_1$: 前置任务 (例如"找到 mug")
- $L_{2A}, L_{2B}$: 后续可选分支 (例如"打开 cupboard" vs "直接 grasp mug")

这个 formulation 其实是 POMDP 的简化版, 但没有显式的 reward 或 transition model。LLM 在这里扮演的角色是 **heuristic planner** —— 它基于常识 (world knowledge in weights) 决定 branch probability, 而不是基于 learned dynamics。这是当前 LLM-for-robotics 的一个 implicit assumption: LLM 的 common sense 足以替代 learned world model。

### 3.3 Vision System

用的是 **Grounded-SAM** (即 Grounding DINO + Segment Anything 的组合):

1. Grounding DINO 做 open-set detection (输入 text prompt + image, 输出 bounding boxes)
2. MobileSAM 在 boxes 内做 segmentation
3. 从 depth (Azure Kinect) 重建 3D voxel
4. 从 voxel mesh 提取 grasp pose

Calibration 用 14cm AprilTag, alignment 精度 $<10^{-6}$ (这个数字看起来太好了, 估计是 numeric precision 而不是真实物理精度, paper 没说清楚)。

关键的数据点:
- COCO zero-shot transfer: AP = 52.5 (Grounding DINO 本身的 benchmark)
- White cup: 100% detection rate (ideal conditions)
- Occlusion 20-30%: ~90% success
- Occlusion 80-90%: ~20% success (塌方了)

直觉: 这个 occlusion curve 是 exponential decay shape, 说明 vision system 在 partial observation 下没有 robust 的 amodal completion 能力。这其实是当前 foundation vision model 的通病——SAM 系列在 amodal segmentation 上依然弱。如果有 [DevNet](https://arxiv.org/abs/2312.06605) 或类似的 amodal 方法接进来, 应该能 push 这个 curve。

### 3.4 Force Feedback

这是这篇 paper 真正区别于 VoxPoser / RT-2 的地方。Force module 的核心是:

**Gravity compensation calibration**:

$$F_{\text{global}} = T_{\text{end\_effector\_to\_robot\_base}} \times F_{\text{local}}$$

- $F_{\text{local}}$: ATI sensor 读到的 force vector (在 end-effector frame)
- $T_{\text{end\_effector\_to\_robot\_base}}$: 4×4 transformation matrix (从 forward kinematics 算)
- $F_{\text{global}}$: 转换到 robot base frame 的 force

这个 transform 看起来是 3×3 rotation 而不是 full 4×4 (因为只 transform force vector, 不 transform position), paper 写得有点 loose。

**Pouring estimation** (quasi-static assumption):

$$F_{\text{up}} \approx mg$$
$$\Delta F_{\text{up}} \approx \Delta mg$$

- $F_{\text{up}}$: global frame 下的 upward force
- $m$: 当前容器内 liquid mass
- $g$: gravitational acceleration
- $\Delta$ 表示时间差分

直觉: 这是把 force sensor 当 scale 用。但 quasi-static 假设在 pitch velocity 大时就崩了——paper 报告:

| Pitch velocity | Pouring accuracy |
|----------------|------------------|
| 4 m/s | ~5.4 g per 100 g (5.4% error) |
| 30 m/s | ~20 g/s error (massive) |

这里 paper 没说 30 m/s 是不是 angular velocity 还是 linear, 但如果是 angular 那这个数字太离谱了 (Kinova Gen3 max joint speed 才 ~60°/s ≈ 1 rad/s)。我怀疑是 typo 或者单位错误, 但不影响 qualitative 结论: dynamic pouring 是 open problem。

### 3.5 Camera-Robot Transform

$$P^R = T_{AR} \times (T_{CA} \times P^C)$$

- $P^C$: point in camera frame
- $T_{CA}$: camera → AprilTag transform
- $T_{AR}$: AprilTag → robot base transform
- $P^R$: point in robot base frame

这是一个 chain of rigid transforms。AprilTag 在这里充当一个 known anchor, 让 camera frame 可以 register 到 robot frame。

## 4. Safety Constraints

paper 里有一段被低估的内容是 hard-coded safety:

- Linear velocity: $\pm 0.05$ m/s
- Angular velocity: $\pm 60°$/s
- End-effector force: 20 N limit
- Workspace bounds: x=[0.0, 1.1], y=[-0.3, 0.3], z=[0, 1.0]
- Check frequency: 10 Hz

关键句: "This is coded into the fundamental motion primitives; therefore, error in the language model will not override this."

直觉: 这是一个非常重要的设计原则—— **LLM 的 output 不能 bypass safety layer**。在 LLM-driven robot 里, hallucination 是不可避免的, 你必须假设 LLM 会 output 荒谬的 command, 然后用 deterministic layer 兜底。这跟 [Bengio et al. 2024](https://www.science.org/doi/10.1126/science.adn0117) 提的 AI safety 框架是吻合的。

## 5. ROS 工作流细节

paper 描述的 ROS architecture:

```
[Kinova Kortex driver] --40Hz--> [joint states]
       |
[Robotiq gripper node] --50Hz, USB--> [gripper position]
       |
[Vision node] --1/3Hz--> [object poses] (via Grounding DINO + SAM)
       |
[Force node] --100Hz--> [6-axis F/τ] (with 5-step averaging)
       |
[LLM-generated code] --> [twist commands + gripper commands]
```

Twist command 是 6-DOF velocity command (3 linear + 3 angular), 这个 abstraction 在 ROS 里叫 `geometry_msgs/TwistStamped`。Force node 用 quaternion-based 3×3 rotation matrix 来 transform local force 到 global, 并保留 raw 和 last-5-timestep average 两种 reading。5-step averaging 是 simple low-pass filter, 在 100Hz 下 effective cutoff ~10Hz, 适合滤掉高频振动噪声。

## 6. Energy / Carbon Footprint

paper 给了一个有意思的数字:

- NVIDIA RTX 2080: ~225 W typical load
- Kinova arm: ~36 W
- Task duration: ≤4 min
- Carbon: ~7 g CO₂ per task

直觉: 这个数字其实挺低的。相比之下, 一个 GPT-4 API call 的 training-equivalent carbon 要高得多。但 paper 没算 GPT-4 inference 的 carbon——因为用的是 API, 算在 OpenAI 头上了。如果按 [Patterson et al. 2024](https://arxiv.org/abs/2309.09733) 的估算, GPT-4 一次 chat completion 大概 ~0.5-2 g CO₂, 一个 task 大概几十次 LLM call, 加起来可能比 robot 本身还多。完整 LCA 应该把这个算进去。

## 7. Limitations (paper 自己承认的)

1. **Affordance map 是 prior knowledge**: kettle/spoon/door handle 的 affordance 是 hand-crafted 的, 不是 learned。但作者引用 [Li et al. 2024](https://arxiv.org/abs/2312.09953) 的 one-shot affordance learning 说这块可以 plug in。
2. **No proactive adaptation**: robot 可以 react to perturbation, 但不能 mid-task switch (例如: 倒水倒一半 user 说"不要了换茶")。这需要更频繁的 LLM polling 和 plan revision。
3. **Complex force dynamics 未建模**: pouring 的 viscosity、container geometry 对 force profile 的影响没建模, 所以高 pitch velocity 下 accuracy 崩。
4. **Vision latency**: 1/3 Hz 太慢, 跟不上 fast motion。

## 8. 我的 critical commentary

**真正 novel 的部分**: RAG + force feedback 的组合。RAG 在 NLP 里已经成熟, 但在 robotics 里基本没人用。paper 把 motion primitives 当成 "cultural knowledge" 的 retrieval unit, 这是一个很 elegant 的 analogy。

**被 over-claimed 的部分**: "ELLMER allows the robot to complete the tasks" 这个 statement 在受控环境下成立, 但 occlusion > 50% 时 vision 就崩了, high pitch velocity 时 pouring accuracy 也崩了。离 "scalable intelligent robot" 还有距离。

**与 SOTA 对比缺位**: paper 只跟 VoxPoser 比 faithfulness, 没跟 RT-2、PaLM-E、Code as Policies 在相同 task 上做 head-to-head。这是 reviewer 应该 push 的地方。

**Intuition for Andrej**: 这篇 paper 本质上是把 "Software 2.0" (LLM-generated code) 接到 "Hardware 1.5" (force-feedback ROS primitives) 上。它的 contribution 不在任何单一模块的突破, 而在于 system integration 的工程智慧: 不同 frequency 的 modality 怎么 sync、LLM hallucination 怎么 sandbox、knowledge base 怎么 curate 而不是 memorize。这种 system paper 在 ML 圈经常被 underrate, 但真实世界 deployment 的 bottleneck 通常就在这。

## 9. 关键 references 值得追

- [Code as Policies](https://arxiv.org/abs/2209.07753) - ELLMER 的 direct ancestor
- [VoxPoser](https://arxiv.org/abs/2304.10677) - 主要 baseline
- [RT-2](https://arxiv.org/abs/2307.15818) - 另一种 LLM+robot 范式 (VLA model)
- [PaLM-E](https://arxiv.org/abs/2303.12789) - embodied multimodal LLM
- [Grounding DINO](https://arxiv.org/abs/2303.05499) - vision backbone
- [Segment Anything](https://arxiv.org/abs/2304.02643) - segmentation backbone
- [Original RAG paper](https://arxiv.org/abs/2005.11401) - Lewis et al. 2020
- [ELLMER GitHub repo](https://github.com/ruaridhmon/ELLMER)
- [Zenodo archive](https://doi.org/10.5281/zenodo.14483539)

如果你想真正 build intuition, 我建议看 Supplementary Video 1 (paper 里提到的那个 robot 做咖啡 + 画 plate 的 demo), 然后再回过头看 Fig. 4 的 force/velocity/position plots——把"robot 在干什么"和"force signal 长什么样"对上, intuition 就建立起来了。Force feedback 在机器人里就是 proprioception 的 analog, 理解了 force signal 的 shape, 就理解了为什么 LLM 单独不够、必须有 sensorimotor layer 兜底。
