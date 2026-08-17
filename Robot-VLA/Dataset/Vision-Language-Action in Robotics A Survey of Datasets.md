---
source_pdf: Vision-Language-Action in Robotics A Survey of Datasets.pdf
paper_sha256: fb7a23ade074c143157cda34fe8c5b2f1ec4848066507f64e50d2d3f1a94e7db
processed_at: '2026-08-13T01:40:58-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 VLA Survey

Andrej, 咱们抛开那些 academic 的框架, 用大白话聊一下这篇 paper 到底在说什么。

---

## 一句话总结

**做 robot 的 VLA 模型, 大家都在卷 model architecture, 但真正卡脖子的是 data**。

这就好比说, 你 GPT 训得好, 不是因为你 transformer 架构多牛, 而是因为你有 Common Crawl + Wikipedia + RLHF 的高质量 data pipeline。VLA 现在就缺这个。

---

## 为什么要写这篇 paper

现在 VLA 这个 field 有点像几年前的 LLM, 大家都在比谁的 model 大、谁的 architecture 新。但作者团队观察到一个现象:

> **你 model 再 fancy, 没有好的 data 喂进去, 出来的 policy 在 real world 还是 garbage**。

具体表现就是:
- **Real-world data 贵到爆**: 雇人 teleoperation 收 data, 一个 episode 几分钟, 收 10 万 episodes 要花几十万美元 + 几个月时间
- **Synthetic data 便宜但假**: simulator 里生成的 trajectory, physics 不准, 渲染不像, 拿到 real robot 上就崩
- **Benchmark 一团乱**: 每个 paper 用不同的 task definition、不同的 success criterion, 根本没法比较
- **Data engine 各自为战**: 有人从 YouTube 视频抠 data, 有人用 LLM 生成 task, 有人用 world model, 但没有一个 unified 的 framework

所以作者说: 咱们别再只盯着 model 了, **data infrastructure 才是 first-class research problem**。

---

## VLA 到底是什么

用最朴素的话说:

$$a_t = \pi(o_t, l)$$

就是你给 robot 两样东西:
- **$o_t$**: 眼睛看到的 (camera image)
- **$l$**: 耳朵听到的 (language instruction, 比如 "把那个红色杯子拿起来")

然后 robot 输出一个 **$a_t$**: 手该怎么动。

听起来简单, 但魔鬼在细节里。

---

## Action Space 这件事有多麻烦

你看 Table 1 里那个 Action 列, 乱得一塌糊涂:

- **Open X-Embodiment**: Mixed EEF (22 种 robot, 每种 control frequency 不一样, action dimension 不一样, 坐标系不一样)
- **RT-1**: Delta EEF (只记录 "相对上一帧怎么动")
- **DROID**: Absolute EEF (记录 "目标绝对位置")
- **RH20T**: EEF + DoF (既有末端执行器, 也有关节角度)

**这就像你收集了一个 corpus, 里面有英文、中文、阿拉伯文、还有乱码, 全混在一起。**

然后你想 train 一个 language model, 但你连 tokenization 都没标准化, 怎么 train?

这就是 Open X-Embodiment 的根本问题: scale 上去了, 但 **interface consistency 下来了**。

反过来看 RT-1、BridgeData V2 这种 single-embodiment dataset, interface 很 clean, 但 embodiment scope 太窄, 换个 robot 就得重新 collect。

这就是论文说的 **fidelity-cost trade-off**:

```
高 fidelity (real, single robot, clean interface) 
    ↔ 
低 cost (synthetic, multi-robot, messy interface)
```

**你没法两个都要。**

---

## Benchmark 现状有多混乱

作者画了那个 2D landscape (Figure 3), 横轴是 task complexity, 纵轴是 environment structure。

**最 striking 的是右上角几乎空的**。

现有 benchmark 的分布:

- **左下角 (简单 task + 简单环境)**: Meta-World, LIBERO, SimplerEnv
  - 都是 tabletop, 10 步以内完成, 单个 object
  - 像是 "把 block 从 A 点移到 B 点"
  - **问题**: 太简单了, 做得好不代表 real world 能用

- **右下角 (复杂 task + 简单环境)**: CALVIN, GemBench, COLOSSEUM
  - 5 个 sequential instruction
  - **CALVIN 的数据触目惊心**: 5 个连续 instruction, **success rate 0.08%**
  - 你没看错, **0.08%**, 几乎等于随机
  - 这说明什么? **long-horizon composition 是 catastrophic failure**

- **右上角 (复杂 task + 复杂环境)**: BEHAVIOR-1K, VLABench
  - 1000 个 everyday activity, full-room 环境
  - 问题是太复杂了, 你都不知道 model 为什么 fail
  - 是 perception 挂了? planning 挂了? 还是 execution 挂了?

**中间和左上角几乎没人做**。

---

## Data Engine 三大流派

这是 paper 最有意思的部分。作者把 data engine 分成三类:

### 流派 1: Video-to-Data (从 YouTube 偷师)

**核心 idea**: 互联网上有海量 human video, 能不能用来 train robot?

代表工作:
- **H2R**: 检测 human hand 3D pose → retarget 到 robot kinematics → 用 inpainting 把 hand 换成 robot arm
- **RoboWheel**: 加了 physics-aware SDF penalty, 让 contact timing 和 grasp semantics 保持一致
- **UniSim**: 最激进, 直接 learn 一个 conditional video diffusion world model, 在里面做 closed-loop training

**为什么这条路有前途**: internet video 是无限的, cost 几乎为零
**为什么这条路难走**: human hand 有 21+ DOF, robot gripper 通常只有 1 DOF, mapping 本质上是 information loss

类比: 这就像你想从人类驾驶视频学自动驾驶, 但人类用方向盘 + 油门 + 刹车, 你的 car 用 steer angle + acceleration, mapping 不 trivial。

### 流派 2: Hardware-Assisted (用硬件帮忙)

**核心 idea**: 用特殊设计的 hardware 让 human teleoperation 更便宜、更 scalable。

代表工作:
- **ALOHA**: 双臂 teleoperation, $20k, kinematic isomorphism (leader arm 和 follower arm 结构一样, 直接 1:1 mapping)
- **GELLO**: 3D 打印 exoskeleton, $300, 比 VR baseline reliability 高 30%
- **UMI**: GoPro + SLAM gripper, 完全 portable, 12 person-hours 收 30 个 location 的 data, 比 standard teleoperation 快 3×
- **Lucid-XR**: VR headset 上跑 physics simulation, <12ms latency, 再用 diffusion 把渲染变 photorealistic

**为什么这条路有前途**: physical grounding 保证, 不存在 embodiment gap
**为什么这条路难走**: scalability 受限于物理世界, 再快也快不过 simulator

类比: 这就像你可以用 iPhone 拍 video 收 data, 但你没法用 iPhone 在 1 小时内生成 100 万条 trajectory。

### 流派 3: Generative Data Engine (用 AI 生成 data)

**核心 idea**: 用 LLM / diffusion model / world model 自动生成 robot training data。

四个子方向:

**3a. Trajectory Reuse (轨迹复用)**
- **MimicGen**: 200 个 human seed → 50k demonstrations, 方法是把 task 分成 subtask, 然后对 object-centric frame 做 spatial transform
- **DynaMimicGen**: 加了 DMP (Dynamic Movement Primitives) 支持 moving objects
- **DemoGen**: 全合成, 用 3D point cloud editing, 1 个 demo → 8 个 real-world task, 74.6% success

**3b. LLM-driven Generation (用 LLM 写 task)**
- **GenSim**: LLM 生成 simulation task code
- **RoboGen**: LLM + RL + motion planning, 69 个 task, 77.4% avg success
- **RoboTwin 2.0**: LLM 生成 + VLM 观察 + feedback loop + domain randomization, 100k+ trajectories

**RoboTwin 2.0 的 pipeline 很优雅**:
```
LLM 写 task code
    → Simulator 执行
    → VLM 看执行结果
    → VLM 说 "这里失败了, 因为..."
    → LLM 根据反馈修改 code
    → 循环直到 success
```

这就像你让 GPT-4 写 code, 然后 Claude 帮你 review, 两个 AI 互相迭代。

**3c. Visual Augmentation (视觉增强)**
- **ROSIE**: text-to-image diffusion 做 inpainting, 把 "pick up chip bag" 变成 "pick up towel", +115% performance
- **RoboEngine**: plug-and-play, 有专门的 Robo-SAM segmentation model
- **EMMA**: 多视角一致性, 用 DreamTransfer

**3d. Predictive World Models (预测性世界模型)**
- **PointWorld**: 3D point flow, zero-shot MPC
- **IRASim**: trajectory-to-video diffusion, Push-T IoU 从 0.637 → 0.961, 和 simulator correlation 0.99
- **3D-VLA**: multimodal goal state generation
- **Genie**: 200k hours internet video, unsupervised latent action discovery, 但只有 1 fps

**为什么这条路有前途**: 理论上可以无限 scale
**为什么这条路难走**: physical grounding 不可靠, sim-to-real gap 存在

---

## 四个 Open Challenges, 用大白话讲

### Challenge 1: Representation Alignment

**问题**: 22 种 robot, 怎么让一个 model 都能控制?

**类比**: 你有一个 multi-lingual corpus, 英文用 Latin alphabet, 中文用汉字, 阿拉伯文从右往左写。你怎么 tokenize?

**可能方向**: Action tokenization + embodiment embedding, 类似 LLM 的 BPE, 为 robot action 设计 universal vocabulary。

### Challenge 2: Multimodal Supervision

**问题**: vision 不够, 接触丰富的 task 需要 tactile feedback, 但 tactile data 太少且无标准。

**类比**: 你训 LLM 只有 text, 没有 image, 那你永远做不了 GPT-4V。VLA 现在就是只有 vision, 没有 touch。

### Challenge 3: Reasoning Assessment

**问题**: 当前 benchmark 只看 success rate (binary), 无法区分是哪一步挂了。

**类比**: 你考学生, 只看期末考试 pass/fail, 不知道是上课没听还是考试紧张。你需要 unit test + midterm + final 的 hierarchical evaluation。

### Challenge 4: Scalable Data Generation with Physical Grounding

**问题**: 你可以生成海量 data, 但你怎么知道这些 data 在 real world 管用?

**论文的 thesis**: 
```
Real scene 3D scanning → High-fidelity simulation → 
Automated trajectory generation → Real-robot validation
```

把真实世界 digitize 到 simulation 里, 在 sim 里大规模 generate data, 再用 real robot 做最后校准。

---

## 和 LLM 的平行对比 (build intuition)

| LLM 历程 | VLA 当前状态 |
|----------|--------------|
| 早期: GPT-2, 架构创新 | 早期: RT-1, OpenVLA, 架构创新 |
| 中期: 发现 data quality 重要 | 当前: 发现 data infrastructure 重要 |
| Common Crawl = 粗 data | Open X-Embodiment = 粗 data |
| Wikipedia = 高质量 corpus | RT-1 / DROID = 高质量 corpus |
| Instruction tuning | Task-specific fine-tuning |
| RLHF | (还没有, 但是方向) |
| Constitutional AI (synthetic) | Generative data engines |
| MMLU benchmark | (还没有, 需要 hierarchical benchmark) |

**VLA 正在经历 LLM 3-5 年前的阶段**, data infrastructure 在分化, model architecture 在收敛。

---

## 我的几个直觉判断

### 1. World Model 会成为 dominant paradigm

**UniSim + Genie + IRASim** 这条线, 本质上是在说: **与其在 real world 收 data, 不如 learn 一个 world model, 在 model 里无限 roll out**。

这就像 AlphaGo 先 learn 一个 game model, 然后自我对弈百万局。VLA 也可以: 先 learn 一个 manipulation world model, 然后在里面 generate 无限 trajectory。

### 2. Action Tokenization 是 low-hanging fruit

现在 Open X-Embodiment 的 mixed action space 简直是灾难。如果有人能 design 一个 universal action tokenizer, 自动 handle:
- 不同 DOF
- 不同 control frequency
- EEF vs Joint space
- Absolute vs Delta

这就是 VLA 的 BPE moment。

### 3. Tactile 会成为下一个 modality war

就像 LLM 从 text-only 到 multimodal, VLA 会从 vision-only 到 vision + tactile + proprioception。RH20T 是先驱, 但 scale 不够。谁先收集到 million-scale tactile dataset, 谁就赢。

### 4. Real-to-Sim-to-Real 是 practical path

论文提的 "high-fidelity scene reconstruction" 方向, 具体来说:
- 用 iPhone LiDAR 扫厨房
- 重建 mesh + texture
- 在 MuJoCo / Isaac Sim 里标定 physics
- 用 motion planner 生成海量 trajectory
- 在 real robot 上 validate

这可能是未来 2-3 年最 practical 的 data engine 方案。

### 5. Evaluation 需要革命

当前 success rate 的 binary evaluation 太粗糙。我们需要:

```
Level 1: Perception accuracy (你看到 object 了吗?)
Level 2: Instruction understanding (你知道要做什么吗?)
Level 3: Plan validity (你的 action sequence 合理吗?)
Level 4: Execution precision (你的 action 准吗?)
Level 5: Recovery capability (出错了能恢复吗?)
```

每层独立 score, 才能做 failure attribution。

---

## 最后的一句话

**VLA 的 future 不在于更大的 transformer, 而在于更好的 data engine + 更精细的 evaluation protocol**。

这就像 2018 年的 NLP, 大家还在比 BERT 和 GPT 谁强, 但真正的 breakthrough 来自 GPT-3 的 data scale + RLHF 的 data quality。

VLA 的 "GPT-3 moment" 会来自某个 data engine 的 breakthrough, 不是某个 model architecture 的创新。

---

## Reference Links

- Paper GitHub: https://github.com/ziyaow1010/vla-datasets-benchmarks
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://roboticstransformer2.github.io/
- DROID: https://droid-dataset.github.io/
- ALOHA: https://tonyzhaozh.github.io/aloha/
- UMI: https://universal-manipulation-interface.github.io/
- MimicGen: https://mimicgen.github.io/
- RoboGen: https://robogen-website.github.io/
- Genie: https://arxiv.org/abs/2402.15391
- UniSim: https://universal-simulator.github.io/unisim/
- CALVIN: https://calvinrobot.github.io/
- LIBERO: https://libero-project.github.io/
- BEHAVIOR-1K: https://behavior.stanford.edu/
- GraspVLA: https://arxiv.org/abs/2505.03233

---

# VLA in Robotics: A Data-Centric Survey 深度解析

Andrej,这篇 survey 的核心 thesis 非常清晰: **VLA 的未来瓶颈在 data infrastructure,而非 model architecture**。这与你一直在强调的 "data is the new code" 思想高度共鸣。让我从 data-centric 的视角,逐层拆解这篇论文。

---

## 1. 核心问题形式化 (Formulation)

### 1.1 VLA Policy 的数学定义

$$a_t = \pi(o_t, l)$$

**变量解析:**
- $a_t \in \mathcal{A}$: time step $t$ 的 action vector,维度取决于 robot 的 control space
- $\pi: \mathcal{O} \times \mathcal{L} \rightarrow \mathcal{A}$: policy function (由神经网络参数化)
- $o_t \in \mathcal{O}$: time step $t$ 的 observation,可以是:
  - 单帧 RGB image: $o_t \in \mathbb{R}^{H \times W \times 3}$
  - RGB-D: $o_t \in \mathbb{R}^{H \times W \times 4}$
  - Point cloud: $o_t \in \mathbb{R}^{N \times 3}$
  - Video sequence: $o_t \in \mathbb{R}^{T \times H \times W \times 3}$
- $l \in \mathcal{L}$: language instruction,通常用 tokenized sequence 表示 $l = (l_1, l_2, ..., l_K)$

**Intuition:** 这里 $l$ 在整个 episode 内是固定的,这暗示 VLA 的 temporal credit assignment 问题极其困难——如果 episode 长 100 步,你不知道哪一步的 action 对最终 task completion 贡献最大。这正是后面 CALVIN benchmark 暴露的核心问题。

### 1.2 Action Space 的两个维度

**Dimension 1: Control Target**
- **EEF (End-Effector) space**: $a_t \in \mathbb{R}^{7}$ (3 position + 4 quaternion orientation) 或 $\mathbb{R}^{6}$ (3 position + 3 Euler angles)
- **DoF (Joint) space**: $a_t \in \mathbb{R}^{n}$,其中 $n$ 是 robot 的自由度 (Franka Panda: $n=7$, dexterous hand: $n=20+$)

**Dimension 2: Parameterization**
- **Absolute**: $a_t^{\text{abs}}$ = target state in chosen space
- **Delta**: $a_t^{\Delta} = a_t^{\text{abs}} - a_{t-1}^{\text{abs}}$

**Intuition:** Delta action 的优势在于它 decouples 了 absolute position 的学习——policy 只需要预测 "下一步往哪移动",不需要记住 "我现在在哪"。这类似于 residual learning 在 ResNet 中的作用。Open X-Embodiment 的 mixed EEF 格式则是最大的 alignment 挑战,因为不同 robot 的 EEF 定义、坐标系、控制频率都不一样。

### 1.3 Success Rate 评估

$$\mathrm{SR} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \mathbb{I}[\text{task completed in } e]$$

**变量解析:**
- $\mathcal{E}$: evaluation episode 集合
- $|\mathcal{E}|$: episode 总数
- $\mathbb{I}[\cdot]$: indicator function,条件满足返回 1,否则返回 0
- 求和: 对所有 episodes 的 completion 结果取平均

**Intuition:** 这个 metric 有个根本问题——它是 binary 的,完全忽略了 partial progress。如果一个 robot 执行了 90% 的 task 但最后一步失败,SR=0;另一个 robot 执行了 10% 但恰好满足 success criterion,SR=1。论文提到 progress-based scores 有时被采用,但没有成为主流。这是一个巨大的 evaluation gap。

---

## 2. Datasets: Fidelity-Cost Trade-off 的根本矛盾

### 2.1 Real-World Datasets 层级

论文呈现了一个清晰的层级结构:

| Dataset | Embodiment | Scale | Action | 核心定位 |
|---------|------------|-------|--------|----------|
| Open X-Embodiment | 22 robots | ~1M+ episodes | Mixed EEF | Cross-embodiment pretraining |
| RT-1 | Everyday Robots | 130k episodes | Delta EEF | Fleet-scale, interface consistency |
| DROID | Franka Panda | 76k episodes | Absolute EEF | In-the-wild visual diversity |
| BridgeData V2 | WidowX 250 | 24k episodes | Delta EEF | Low-cost standardized |
| RH20T | 4 robots | 110k episodes | EEF+DoF | Multimodal (tactile/audio) |
| Ego4D | Human hands | 3,000 hours | N/A | Semantic prior injection |

**Intuition:** 这里的核心 trade-off 可以用一个公式表达:

$$\text{Fidelity} \times \text{Scale} = \text{Constant} \approx \text{Budget}$$

- Open X-Embodiment 追求 Scale,牺牲了 interface consistency
- RT-1 追求 interface consistency,限制了 embodiment diversity
- RH20T 追求 modality fidelity,牺牲了 scale (相对小)

### 2.2 Synthetic Datasets 的策略

| Dataset | Strategy | 优势 | 劣势 |
|---------|----------|------|------|
| SynGrasp-1B | Procedural randomization | 1B grasp samples | Quasi-static assumption |
| RoboCasa | Kitchen simulation suite | Diverse scenes | Rendering artifacts |
| RoboGen | LLM-generated tasks | 100+ tasks auto-generated | Physical implausibility |
| MimicGen | Trajectory reuse from seeds | 50k from 200 seeds | Subtask structure assumption |

**Intuition:** Synthetic data 的核心问题是 sim-to-real gap。论文引用 GraspVLA 的工作,说明即使有 billion-scale synthetic data,仍需要 real-world data 来做 final calibration。这暗示了一个重要的 research direction: **sim 和 real 的比例应该是多少?** 目前没有定论。

参考链接:
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- BridgeData V2: https://github.com/rail-berkeley/bridge_data_v2
- RH20T: https://github.com/RH20T-Dataset/RH20T
- Ego4D: https://ego4d-data.org/

---

## 3. Benchmarks: Two-Dimensional Landscape

### 3.1 架构图解析 (Figure 3)

论文提出了一个 2D landscape:

```
                    Environment Structure
                    (Diversity ↑)
                         │
           BEHAVIOR-1K ● │     ● Open X-Embodiment
                         │
                         │     ● VLABench
                         │
    ─────────────────────┼──────────────────── Task Complexity
                         │                (Compositional ↑)
           COLOSSEUM ●   │
                         │
           GemBench ●    │     ● CALVIN
                         │
           Meta-World ●  │     ● LIBERO
           SimplerEnv ●  │
                         │
                         ▼
```

**Intuition:** 这个 landscape 的关键 insight 是: **现有 benchmarks 大多集中在左下角 (simple tabletop, short-horizon)**,右上角 (complex multi-scene, long-horizon) 几乎是空白。这正是 VLA generalization 的 frontier。

### 3.2 Table-top Benchmarks 详解

**Simple Short-horizon:**

- **Meta-World**: 50 tasks, low-dim state observations
  - 问题: vision perception 被 simplified,无法测试 visual grounding
  - 优势: 可控,可复现

- **LIBERO**: 4 suites (spatial, object, goal, long)
  - 大多数 task 在 10 steps 内完成
  - Procedural variation 存在但 interaction 仍然 tabletop-bound

- **SimplerEnv**: 
  - 核心设计哲学: "sufficiently realistic to preserve sim-to-real ranking consistency"
  - 这是一个非常聪明的 insight: 不追求 photorealism,追求 evaluation 的 ranking 一致性

**Complex Long-horizon (Tabletop):**

- **CALVIN**: 
  - Long-horizon: 最长 5 sequential instructions
  - Zero-shot generalization to unseen environment
  - **关键数据**: 5 sequential instructions → **0.08% success rate**
  - Intuition: 这个数字揭示了 long-horizon composition 的灾难性退化。不是线性 degradation,是 exponential。这暗示当前的 VLA 缺乏 temporal abstraction 机制。

- **GemBench**: 
  - Hierarchical generalization: object placement → instance → compositional
  - 暴露 higher complexity levels 的 bottleneck

- **COLOSSEUM**: 
  - 14 axes of perturbation
  - 关键 insight: **single-axis robustness does NOT extrapolate to multi-axis**
  - 这意味着 "robust to lighting" + "robust to object color" ≠ "robust to both"

### 3.3 Multi-scene Benchmarks

- **BEHAVIOR-1K**:
  - 1000 everyday activities
  - Predicate-based language (explicit multi-stage objectives)
  - Rigid + deformable + fluid objects
  - Full-room + multi-room environments

- **VLABench**:
  - Composite language-conditioned tasks
  - Long-horizon multi-step reasoning
  - Intermediate reasoning grounded in scene semantics
  - Systematic failures in multi-step logical tasks

**Intuition:** BEHAVIOR-1K 和 VLABench 代表了 benchmark 设计的另一个极端:极度复杂,但 evaluation 变得 almost impossible——如果 task 涉及 50 步和 20 个 object,你如何 disentangle 是 perception 失败、planning 失败、还是 control 失败?

参考链接:
- CALVIN: https://calvinrobot.github.io/
- LIBERO: https://libero-project.github.io/
- Meta-World: https://github.com/rlworkgroup/metaworld
- BEHAVIOR-1K: https://behavior.stanford.edu/
- VLABench: https://github.com/OPEN-IAI/VLABench
- COLOSSEUM: https://github.com/UT-AI-Robotics/TheColosseum

---

## 4. Data Engines: 三大范式深度解析

### 4.1 Video-to-Data Engine

**核心挑战:** Embodiment gap (human hand vs robot manipulator)

| Engine | Input | Method | Key Metric |
|--------|-------|--------|------------|
| H2R | Egocentric video | 3D hand pose → retarget → inpainting | +1.3-10.2% sim, +3-23% real |
| RoboWheel | Egocentric video | SDF + residual RL | Cross-embodiment (6/7-DOF) |
| Video2Policy | Internet video | Mesh + 6D pose + GPT-4o code | 88% sim success, 100+ videos |
| X-Humanoid | Ego-Exo4D | Video diffusion (Wan 2.2) | 60+ hours conversion |
| GenMimic | Video gen output | 4D lifting + keypoint tracking | Zero-shot transfer |
| UniSim | Internet + robot | Conditional video diffusion | 3-4× better than baselines |

**技术细节:**

H2R 的 pipeline:
```
Egocentric Video 
    → 3D Hand Pose Detection (MANO model)
    → Kinematic Retargeting (optimization-based)
    → Robot Arm Rendering
    → Segmentation + Inpainting (replace hands)
    → VLA Training Data
```

**Intuition:** Video-to-data 的核心 insight 是: **internet video 是 virtually unlimited 的,但 embodiment gap 是 fundamental 的**。H2R 用 inpainting 来弥合 visual gap,但 kinematic retargeting 的 loss 是 open problem——human hand 有 21+ DOF,robot gripper 通常只有 1 DOF,这个 mapping 本质上是 information loss。

UniSim 代表了更激进的 direction: 直接学习一个 conditional video diffusion world model,让 VLA 在 learned simulator 里做 closed-loop training。这接近 "world model as data engine" 的 ultimate vision。

参考链接:
- H2R: https://arxiv.org/abs/2505.11920
- RoboWheel: https://arxiv.org/abs/2512.02729
- Video2Policy: https://arxiv.org/abs/2502.09886
- X-Humanoid: https://arxiv.org/abs/2512.04537
- UniSim: https://arxiv.org/abs/2310.06114

### 4.2 Hardware-Assisted Engine

**核心 trade-off:** Cost vs Precision vs Scalability

| Engine | Cost | Key Innovation | Performance |
|--------|------|----------------|-------------|
| ALOHA | ~$20k | Bimanual kinematic isomorphism | 80-90% with ACT |
| GELLO | <$300 | 3D-printed exoskeleton | +30% reliability vs VR |
| UMI | ~$500 | GoPro + SLAM gripper | 71.7% zero-shot, 3× faster |
| DexCap | Moderate | EMF gloves + RGB-D | 72% multi-finger |
| Lucid-XR | VR headset | Physics sim on VR + diffusion | 5× effective data |

**关键技术细节:**

ALOHA 的 kinematic isomorphism:
```
Leader arm (human control) 
    → Joint angles θ_leader
    → Direct mapping θ_follower = θ_leader
    → Follower arm (robot) executes
```
这种 1:1 mapping 消除了 IK 求解的延迟,实现 50Hz+ 的 bimanual control。

UMI 的 portable design:
```
GoPro (240 fps) + Gripper mechanism + SLAM tracking
    → 6DoF gripper pose estimation
    → IK retargeting to any robot
    → In-the-wild data collection
```

**Intuition:** Hardware-assisted engines 的核心价值在于 **physical grounding 的保证**。Video-to-data 有 embodiment gap,generative engine 有 sim-to-real gap,但 hardware 直接在 real physics 中操作。代价是 scalability 受限于物理世界。UMI 的 insight 是: 把 "teleoperation setup" 从 robot 上解耦到 human hand 上,用 SLAM 来 bridge gap,这是 data collection democratization 的重要一步。

参考链接:
- ALOHA: https://arxiv.org/abs/2304.13705
- GELLO: https://arxiv.org/abs/2309.13037
- UMI: https://arxiv.org/abs/2402.10329
- DexCap: https://arxiv.org/abs/2403.07788

### 4.3 Generative Data Engine

这是论文中内容最丰富的部分,分为四个子类:

**Sub-category 1: Trajectory Reuse**

| Engine | Method | Key Metric |
|--------|--------|------------|
| MimicGen | Object-centric subtask segmentation + transform | 50k from 200 seeds |
| DynaMimicGen | DMP adaptation for moving objects | Dynamic task support |
| DemoGen | 3D point cloud editing | 74.6% from single demo |

MimicGen 的核心算法:
```
Given: Seed demonstration D = {τ_1, τ_2, ..., τ_T}
       New object configuration C_new

1. Segment D into subtasks: {s_1, s_2, ..., s_K}
2. For each subtask s_k:
   - Identify object-centric frame F_k
   - Transform: s_k' = T(F_k → F_k_new) × s_k
3. Stitch subtasks: D_new = s_1' ∘ s_2' ∘ ... ∘ s_K'
4. Validate: IK feasibility + collision check
```

**Sub-category 2: LLM-driven Generation**

| Engine | Method | Scale | Performance |
|--------|--------|-------|------------|
| GenSim | LLM → task code | 100+ tasks | - |
| RoboGen | LLM + RL + motion planning | 69 tasks | 77.4% avg |
| RoboTwin 2.0 | LLM + VLM feedback + domain randomization | 100k+ trajectories | - |

RoboTwin 2.0 的 feedback loop:
```
LLM generates task code 
    → Simulation executes
    → VLM observer detects failure
    → VLM provides corrections
    → LLM refines code
    → Iterate until success
```

**Sub-category 3: Visual Augmentation**

| Engine | Method | Performance Gain |
|--------|--------|------------------|
| ROSIE | Text-to-image diffusion inpainting | +115% |
| RoboEngine | Robo-SAM + background generation | Similar to ROSIE |
| EMMA | DreamTransfer multi-view consistency | - |

ROSIE 的 insight: 用 text-to-image diffusion 做 semantic inpainting,可以把 "pick up chip bag" 变成 "pick up towel",从而 create unseen tasks from existing demonstrations。

**Sub-category 4: Predictive World Models**

| Engine | Method | Key Capability |
|--------|--------|----------------|
| PointWorld | 3D point flows | Zero-shot MPC |
| IRASim | Trajectory-to-video diffusion | 0.99 correlation with sim |
| 3D-VLA | Multimodal goal state generation | 3D-LLM aligned |
| Genie | Unsupervised latent action discovery | 200k hours video |

IRASim 的 evaluation:
```
Trajectory → Video diffusion → Generated video
    → Evaluate: IoU with ground truth
    → Push-T: 0.637 → 0.961 IoU (with planning)
    → Correlation: 0.99 with sim
```

**Intuition:** Predictive world models 是 generative engine 的 ultimate direction。如果我们可以 learning 一个足够 accurate 的 world model,VLA 就可以在 model 内做 unlimited closed-loop training,完全 bypass 物理世界的 cost。但问题在于: **temporal consistency** 和 **physical accuracy** 仍然不足。Genie 用 200k hours internet video 学 latent action,但只有 1 fps 生成速度和 16 frame memory,离 real-time deployment 还很远。

参考链接:
- MimicGen: https://mimicgen.github.io/
- GenSim: https://arxiv.org/abs/2310.01361
- RoboGen: https://robogen-website.github.io/
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- ROSIE: https://arxiv.org/abs/2302.11550
- Genie: https://arxiv.org/abs/2402.15391

---

## 5. 四个 Open Challenges 的深度分析

### 5.1 Representation Alignment across Embodiments

**问题:** Open X-Embodiment 有 22 种 robot,每种有不同的:
- Control frequency (3Hz - 20Hz)
- Action dimension (4D - 20D)
- EEF vs Joint space
- Gripper type (parallel jaw, suction, dexterous)

**可能的 solution direction:**

Action tokenization + embodiment embedding:
$$a_t = \text{Tokenizer}_e(a_t^{\text{raw}}; e)$$

其中 $e$ 是 embodiment identifier。这类似于 LLM 中不同语言共享 vocabulary 的思路。

### 5.2 Multimodal Supervision

**问题:** Contact-rich manipulation (插入、紧握、精细操作) 需要 tactile feedback,但:
- Tactile sensor 昂贵且 fragile
- RH20T 只有 110k episodes (vs Open X 的 1M+)
- Tactile signal 的 representation 缺乏标准化

**Intuition:** 这类似于 LLM 中 text-only vs multimodal 的差距。VLA 目前是 "vision-dominant",但 real manipulation 是 "vision + touch + proprioception" 的 fusion。Missing modality 可能是 fundamental limit。

### 5.3 Reasoning Assessment

**问题:** 当前 benchmark 无法 disentangle:
- Perception failure (看不见 object)
- Planning failure (不知道下一步做什么)
- Memory failure (忘了 instruction)
- Control failure (执行不准)
- Recovery failure (出错后无法恢复)

**可能的 solution:** Hierarchical evaluation protocol:
```
Level 1: Perception accuracy (object detection, pose estimation)
Level 2: Instruction understanding (task classification, goal prediction)
Level 3: Plan generation (action sequence validity)
Level 4: Execution accuracy (action precision)
Level 5: Recovery capability (disturbance handling)
```

### 5.4 Scalable Data Generation with Physical Grounding

**核心 tension:**
$$\text{Generation Scale} \gg \text{Grounding Reliability}$$

Video-to-data 有 pose estimation noise,LLM-driven 有 physical implausibility,world model 有 sim-to-real gap。

**Future direction (论文的 thesis):**
High-fidelity scene reconstruction + physically accurate simulator:
```
Real-world scene 
    → 3D scanning (LiDAR + RGB-D)
    → Geometric reconstruction (mesh + texture)
    → Physics calibration (friction, mass, contact)
    → High-fidelity simulation
    → Automated trajectory generation
    → VLA training data
```

---

## 6. 我的额外联想与思考

### 6.1 与 LLM Pretraining 的平行对比

VLA data 的发展路径与 LLM 高度相似:

| LLM | VLA |
|-----|-----|
| Web text scraping | Internet video (Ego4D, YouTube) |
| Wikipedia quality corpus | Open X-Embodiment |
| Instruction tuning (SFT) | Task-specific fine-tuning (RT-1, DROID) |
| RLHF | Human preference on trajectories |
| Synthetic data (Constitutional AI) | Generative data engines |

**Key difference:** LLM 的 data 是 discrete token,VLA 的 data 是 continuous + multimodal + embodiment-specific。这使得 VLA 的 data curation 更难标准化。

### 6.2 Scaling Laws 在 VLA 中的挑战

LLM 的 scaling law: $L(N, D) \propto N^{-\alpha} D^{-\beta}$

VLA 的 challenge:
- Data heterogeneity (embodiment, action space)
- Evaluation noise (success rate vs progress)
- Sim-to-real gap (no clear proxy metric)

**Intuition:** VLA 可能需要 **embodiment-conditional scaling law**:
$$L(N, D, e) \propto N^{-\alpha} D_e^{-\beta_e}$$

其中 $D_e$ 是 embodiment $e$ 的 data,$\beta_e$ 是 embodiment-specific exponent。

### 6.3 与 Open-World Learning 的关系

论文没有深入讨论,但 VLA 的 ultimate goal 是 open-world deployment。这要求:
- **Out-of-distribution generalization** (unseen objects, scenes)
- **Compositional generalization** (new task from known skills)
- **Continual learning** (adapt without catastrophic forgetting)

当前 benchmark 几乎都不测试这些 capability。LIBERO-PRO 和 LIBERO-X 开始关注 robustness,但仍然局限在 tabletop。

### 6.4 Data Quality vs Data Quantity 的根本问题

论文的核心 message 可以总结为:

**当前 VLA 的瓶颈: 不是 data 量不够,而是 data 的 physical grounding 和 evaluation 的 disentanglement 不够。**

这与 LLM 的 "quality > quantity" trend 一致。但 VLA 更难,因为:
- "Quality" 的定义 embodiment-specific
- "Grounding" 需要 physical validation
- "Evaluation" 需要 real-robot deployment

### 6.5 未来 Research Direction 的大胆推测

基于论文的 analysis,我推测以下几个 direction 会 become dominant:

1. **World Model as Universal Data Engine**: UniSim 方向的 extension,用 large-scale video pretraining + robot data fine-tuning 来 learn 一个 universal world model,所有 VLA 在这个 model 内训练。

2. **Embodiment-Aware Action Tokenization**: 类似 LLM 的 BPE,为 robot action 设计一个 universal action vocabulary,自动 handle 不同 embodiment 的 mapping。

3. **Hierarchical Benchmark with Disentangled Metrics**: 像 MMLU 之于 LLM,VLA 需要一个 hierarchical benchmark,每个 level 测试不同 capability。

4. **Real-to-Sim-to-Real Pipeline**: 用 3D scanning 重建 real scene → 在高保真 sim 中生成 data → 用 real-robot 做 final validation。这可能是 paper 提到的 "high-fidelity scene reconstruction" 的 concrete implementation。

5. **Cross-Embodiment Foundation Models**: 类似 CLIP 之于 vision-language,VLA 需要一个 cross-embodiment 的 foundation model,通过 contrastive learning 对齐不同 robot 的 representation。

---

## 7. 总结

这篇 survey 的核心贡献是:

1. **提出 data-centric 的 VLA taxonomy**: Datasets + Benchmarks + Data Engines 的三 pillar 框架
2. **揭示 fidelity-cost trade-off**: Real-world (high fidelity, low scale) vs Synthetic (low fidelity, high scale) 的根本矛盾
3. **2D benchmark landscape**: Task complexity × Environment structure,暴露 evaluation 的 structural gaps
4. **Data engine 的三大范式**: Video-to-data, Hardware-assisted, Generative
5. **四大 open challenges**: Representation alignment, Multimodal supervision, Reasoning assessment, Scalable grounding

**Final intuition:** VLA 正在经历 LLM 5 年前的阶段——model architecture 在收敛,data infrastructure 在分化。未来的 breakthrough 更可能来自 data engine 的创新 (类似 Chinchilla 对 LLM scaling law 的 insight),而非 model architecture 的 tweak。论文提到的 "high-fidelity scene reconstruction + physically accurate simulator" 方向,可能就是 VLA 的 "Common Crawl + Wikipedia" moment。

参考资源:
- 论文的 GitHub repo: https://github.com/ziyaow1010/vla-datasets-benchmarks
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://roboticstransformer2.github.io/
- ALOHA: https://tonyzhaozh.github.io/aloha/
- MimicGen: https://mimicgen.github.io/
- Genie: https://arxiv.org/abs/2402.15391
