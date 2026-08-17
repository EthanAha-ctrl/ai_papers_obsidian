---
source_pdf: RoboMatrix.pdf
paper_sha256: 601a8f86a4aa5fe4c616c99fee1765ebd7ebae7c1c8359874ef93fbf5e54191a
processed_at: '2026-08-12T01:04:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 RoboMatrix

Andrej，咱们抛开学术黑话，聊聊这篇paper到底在干啥。

---

## 一句话总结

**别让机器人学"整个任务"，让它学"基本动作"，然后像搭积木一样拼起来。**

就这么简单。整个paper的core idea就这一句。

---

## 为啥要这么搞？

想象你教小孩做家务。

### Task-centric 的教法（现有方法）

你跟小孩说："今天教你'把可乐罐放进白盒子里'这个任务"，然后让他从头到尾练100遍。

明天你说："现在教你'把紫色方块放进抽屉里'"，又得从头练100遍。

后天你说："教你'爬坡然后把绿色罐子放进抽屉'"——又是一个全新任务，又得从头练。

问题很明显：
- 每个任务都得重新collect data，累死人
- 任务千千万，永远collect不完
- 哪一步做错了你都不知道，因为整个流程是个black box

### Skill-centric 的教法（RoboMatrix）

你换个思路：先教小孩几个**基本功**——
- "走到某个东西旁边"
- "抓住某个东西"  
- "走到某个容器旁边"
- "松手"
- "爬坡"
- ......

一共就8个基本功。然后任何新任务，你只需要告诉他"先做A，再做B，再做C"。

比如"把可乐罐放进白盒子"就拆成：
1. 走到可乐罐旁边
2. 抓住可乐罐
3. 走到白盒子旁边
4. 把可乐罐定位到白盒子上方
5. 松手

你看，**基本功是有限的，任务的组合是无限的**。这就是compositional generalization的核心直觉。

---

## 具体怎么实现的？

RoboMatrix分三层，我用人话挨个说：

### 第一层：大脑（Scheduling Layer）

就是一个GPT-based agent，干两件事：

**第一件事——拆任务**：你跟它说"把红罐子放进白盒子"，它就帮你拆成上面那5个subtask。但它不能瞎拆，只能从predefined的8个skill里选。就像你给小孩一个"动作菜单"，他只能从菜单里点菜。

**第二件事——执行前check一下**：比如要执行"抓住红罐子"之前，先看一眼——红罐子在不在视野里？如果不在，就别执行了，省得白忙活。这用的是Grounding DINO做object detection。

这个"check before act"的设计特别实用。你想啊，如果罐子根本不在桌上，你还让机器人去抓，那就是纯纯浪费时间，还可能撞到别的东西。

### 第二层：肌肉（Skill Layer）

这一层真正干活，有两种model：

**VLA Model（管"灵活活儿"）**

就是Vicuna 1.5（一个LLM）+ CLIP（一个vision encoder）。输入一张图和一个skill prompt，输出一个动作。

但这里有个很聪明的工程细节——**动作怎么表示？**

他们把连续动作discretize成7个维度，每个维度256个bin：

$$\epsilon, \Delta X, \Delta Y, \Delta\theta_{yaw}, \Delta\mu_{pos}, \Delta\nu_{pos}, \phi$$

用大白话说：
- $\epsilon$ = "我做完了吗？"（stop signal，0或1）
- $\Delta X, \Delta Y$ = 机器人在地上往哪挪多少（前后左右）
- $\Delta\theta_{yaw}$ = 机器人转多少度
- $\Delta\mu_{pos}, \Delta\nu_{pos}$ = 机械臂的gripper往哪移
- $\phi$ = 夹爪开还是关

每个维度切成256份，就像把连续的temperature分成"很冷/冷/凉/温/热/很热"这种离散档位。这样动作就变成了7个token，LLM可以直接output。

**为啥不用absolute position？** 他们试过，发现robot会overfit到training时的具体坐标，换个位置就傻了。用relative position（"往前挪0.3米"而不是"移到坐标(2.5, 3.1)"），model学到的是view-invariant的skill，generalization好得多。

**另一个很妙的trick——interval prediction**

训练时，input是第$t$帧的image，但supervision是第$t+10$帧的action，不是第$t$帧的。

为啥？因为如果用当前帧的action做supervision，model会学到一种"磨磨唧唧"的行为——每步只挪一点点，robot移动超级慢。用10帧之后的action做supervision，model学到的是"往前看"的planning，动作更果断、更流畅。

10帧是tune出来的——太小robot太慢，太大robot太猛容易不精准。

**Hybrid Model（管"精确活儿"）**

有些活儿VLA真不擅长，比如射击。这时候用传统方法反而更好：

- **Shooting**：用YOLO-World检测目标位置，然后PD control把gimbal对准，再考虑gravity compensation调整pitch，最后开火。这是classic visual servoing，精度远超VLA。
- **Searching**：机器人原地转圈，每隔一段拍张照，用YOLO-World看目标在不在，在就停。
- **Climbing**：用IMU读pitch angle判断斜坡角度，调speed，到顶了（pitch归零）就停。

这里的intuition是：**VLA擅长处理不确定性（object在哪、长啥样），PD control擅长精确控制（gimbal对准中心）**。right tool for right job。

### 第三层：手脚（Hardware Layer）

用的是DJI RoboMaster系列的两个robot：
- **EP robot**：有机械臂+夹爪，干manipulation
- **S1 robot**：有云台+发射器，干shooting

通信用DDS协议，decentralized的，没有master node bottleneck。VLA model在cloud server上跑，robot通过LAN把image传过去，server返回action。latency大概100ms。

---

## 8个Meta-skill是哪些？

```
1. Move to <object>          → 走到物体旁边
2. Grasp <object>            → 抓住物体
3. Move to <container>       → 走到容器旁边
4. Position <obj> over <container> → 把物体举到容器上方
5. Release <object>          → 松手
6. Place <object>            → 放置物体
7. Open/Close drawer          → 开/关抽屉
8. (其他auxiliary skills)
```

就这8个，能拼出paper里测的所有long-horizon task。

---

## Data Collection的smart做法

这是我觉得最practical的部分：

```
第一轮：
  给8个skill各collect一些data（均匀分配）
  → 训练
  → 上真机测试
  → 记录哪些skill表现差

第二轮：
  只给差的skill补data
  → 合并数据
  → 重新训练
  → 再测
  → ...

循环到所有skill都够好
```

对比task-centric：如果某个task失败，你得重新collect整个task的data（可能1000 frames）。skill-centric只需要补那个差的skill（200 frames）。**data efficiency提升5倍左右**。

---

## 实验结果说了啥？

### 1. Skill-centric完胜Task-centric

| | 简单任务 | 中等任务 | 难任务 |
|---|---|---|---|
| Task-centric | 100% | 80% | 40% |
| Skill-centric | 100% | 100% | 80% |

简单任务两者差不多，但**任务越长越复杂，skill-centric的优势越大**。这很intuitive——task-centric是死记硬背，任务一变就崩；skill-centric是学基本功再组合，换个组合方式照样能干。

### 2. 5级generalization测试

从"见过物体+见过场景"到"没见过物体+没见过场景"，难度逐级递增：

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Task-Centric (mini) | 80% | 30% | 20% | 70% | **0%** |
| Skill-Centric (mini) | 90% | 80% | 60% | 80% | 50% |
| Skill-Centric (full) | **100%** | **100%** | **90%** | **100%** | **80%** |

Task-centric在Level 5（完全unseen场景+unseen物体）直接0%——彻底崩了。Skill-centric还有80%。这就是compositional generalization的力量。

### 3. Pretrain很重要

| 方式 | 整体成功率 |
|---|---|
| 不pretrain | 30% |
| 用web data pretrain | 80% |
| 用web data + robot data pretrain | **100%** |

先在大规模web data上align视觉和语言，再用robot data做domain-specific alignment，最后SFT。三步走，缺一不可。

### 4. Model size matters

7B model在unseen场景上70-80%，13B model直接拉到100%/90%。**VLA也follow scaling law**——bigger is better。

### 5. Cross-embodiment初步可行

在EP robot上训练的model，直接deploy到S1 robot上，成功率20%。虽然不高，但至少说明skill representation有一定的embodiment-invariance。这比task-centric的0%强多了。

---

## 我的直觉和质疑

### 啥是对的？

1. **Skill-centric的paradigm根本就是对的**。这和人类学东西一模一样——先学走、学抓、学放，再组合成复杂任务。end-to-end learn整个task space是走不通的，task空间是infinite的。

2. **Hybrid model很务实**。承认VLA不是万能的，精确控制该用传统方法就用传统方法。这种"right tool for right job"的engineering judgment很mature。

3. **Execution Checker防崩**。执行前先看物体在不在，这个设计避免了大量catastrophic failure，实际部署中非常有用。

4. **Iterative data collection很smart**。只给差的skill补data，而不是无脑全collect一遍。这种"哪里不会补哪里"的策略效率极高。

### 啥是有问题的？

1. **Skill granularity是个art**。8个skill够不够？谁定的？如果新任务需要"倒水"，这不在现有skill里，得手动加。**automatic skill discovery**是个大open problem，paper没解决。

2. **Long-horizon success rate理论上应该更低**。每个skill ~90%成功率，5步chain下来理论上是 $0.9^5 \approx 59\%$，但paper报了80%。这说明要么有某种implicit error recovery，要么实验设计上有点optimistic。paper没详细讨论这个gap。

3. **Cross-embodiment只有20%**。说明skill representation还远远没到embodiment-invariant的程度。两个robot其实差别不大（都是DJI RoboMaster），换成完全不同的platform（比如Franka arm或Boston Dynamics Spot）大概率会更差。

4. **Stop signal的可靠性**。$\epsilon$ 如果误触发，task直接premature termination。paper没说这个failure mode的频率，但我怀疑这是主要error source之一。

5. **100ms latency**。cloud inference的100ms delay对reaction-heavy的task是个bottleneck。如果物体在动、或者需要快速调整，这个latency可能致命。

6. **Skill list是manual的**。新skill需要human refine后加入list，这是个human-in-the-loop bottleneck。真正要scale到open-world，需要automatic skill discovery + automatic skill learning。

---

## 和其他工作的关系

| 方法 | 思路 | 问题 |
|---|---|---|
| RT-1/RT-2 | End-to-end learn task | Task空间无限，generalize不了 |
| SayCan | LLM规划 + predefined skills | 有点像，但skill是hardcoded的 |
| OpenVLA | Open-source VLA | 还是task-centric |
| VoxPoser | LLM生成3D value map | 适合manipulation但不好scale |
| **RoboMatrix** | **Skill-centric + hierarchical** | **最接近"学基本功再组合"的人类认知模式** |

RoboMatrix和SayCan其实思路类似，但SayCan的skill是hardcoded的API，RoboMatrix的skill是learned的VLA model，而且有iterative data collection的data pipeline。这是进步。

---

## 脑洞一下未来

1. **Automatic skill discovery**：让LLM分析failure case，自动propose"我觉得需要一个新skill叫XX"，然后自动collect data + train。这才能真scale。

2. **Skill composition learning**：现在是sequential的（A做完做B），未来能不能学parallel的（两支arm同时做不同skill）？

3. **Skill之间的transition learning**：A到B的smooth transition本身就是个skill，现在被忽略了。

4. **World model + Skill**：执行skill前先在world model里"imag"一遍，预测会不会成功。就像人类action前会在脑子里simulate。

5. **Skill的"难度自适应"**：简单场景用简单skill，难场景自动切换到更精细的skill版本。

6. **Language-guided skill refinement**：用natural language给skill feedback（"你抓太用力了"），skill model自动调整。

---

## 最核心的takeaway

如果你想build一个general robot agent，**先想清楚你的skill vocabulary是什么**，再想怎么compose。

别想着end-to-end learn整个task space——那就像试图背下所有可能出现的句子，永远背不完。学语法+词汇，才能generate infinite sentences。

RoboMatrix就是在robot domain做这件事：学有限个meta-skills（词汇），用LLM来compose（语法），完成无限个tasks（句子）。

这个intuition我觉得是对的，虽然execution上还有很多粗糙的地方。

---

### 参考链接

- [RoboMatrix GitHub](https://github.com/WayneMao/RoboMatrix)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [SayCan Project](https://say-can.github.io/)
- [VIMA Benchmark](https://vimalabs.github.io/)
- [VoxPoser Project](https://voxposer.github.io/)
- [Grounding DINO](https://arxiv.org/abs/2405.10300)
- [YOLO-World](https://arxiv.org/abs/2401.17770)
- [LLaVA](https://arxiv.org/abs/2310.03744)
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
- [Mobile ALOHA](https://mobile-aloha.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
- [PaLM-E](https://palm-e.github.io/)

Andrej，hope this helps build your intuition！有啥想dive deeper的，随便问。

---

# RoboMatrix 深度解析

 Andrej，这篇paper的核心直觉其实非常elegant——把robot learning从"task-level memorization"拉回到"skill-level composition"。让我从底层逻辑到工程实现给你拆解。

---

## 1. Core Intuition: 为什么Skill-centric?

### 1.1 Task-centric的根本问题

现有VLA model（如RT-1、RT-2、OpenVLA）基本上都在做end-to-end的task imitation。问题在于：

- **Data efficiency极差**: 一个long-horizon task（比如"open drawer → put cube in → close drawer"）需要collect完整episode。假设一个task平均1000 frames，而每个skill只占200 frames，那80%的data其实是冗余的——因为同一个"Grasp"动作在不同task里几乎identical。
- **Combinatorial explosion**: open-world的task空间是infinite的，你永远无法collect所有task的数据。但meta-skill的空间是finite且enumerable的——这就是paper的key insight。
- **Error localization impossible**: end-to-end black-box模型失败时，你完全不知道是perception错了、planning错了、还是low-level control错了。

### 1.2 Skill-centric的数学直觉

可以formalize一下：假设task space $\mathcal{T}$ 和skill space $\mathcal{S}$，task-centric方法是学习mapping $\pi_\theta: \mathcal{T} \rightarrow \mathcal{A}$（task到action sequence），而skill-centric是学习 $\pi_\theta: \mathcal{S} \rightarrow \mathcal{A}$ 加上一个decomposer $D: \mathcal{T} \rightarrow \mathcal{S}^*$。

关键observation是：$|\mathcal{S}| \ll |\mathcal{T}|$。paper中只用8个meta-skills就能组合出paper里测试的所有long-horizon tasks。这类似于NLP里"few phonemes → infinite sentences"的compositional generalization。

参考: 
- [VIMA: General Robot Manipulation with Multimodal Prompts](https://arxiv.org/abs/2210.03094) - 提出了类似的multimodal prompt分解思路
- [RT-2: Vision-Language-Action Models](https://arxiv.org/abs/2307.15818) - task-centric的SOTA，但完全end-to-end

---

## 2. 三层Hierarchical Architecture

### 2.1 整体架构图解析

```
┌─────────────────────────────────────────────────┐
│  Modular Scheduling Layer (High-level)           │
│  ┌──────────────┐    ┌────────────────────┐    │
│  │ Task-Planning│    │ Execution Checker   │    │
│  │ Agent (GPT)  │───▶│ (Grounding DINO v1.5)│   │
│  └──────────────┘    └────────────────────┘    │
│         │                       │               │
└─────────┼───────────────────────┼───────────────┘
          ▼                       ▼
┌─────────────────────────────────────────────────┐
│  Skill Layer (Middle-level)                      │
│  ┌─────────────────┐    ┌────────────────────┐  │
│  │ VLA Model        │    │ Hybrid Model       │  │
│  │ (Vicuna + CLIP)  │    │ (PD + YOLO-World)  │  │
│  │ - Move/Grasp/... │    │ - Search/Shoot/    │  │
│  └─────────────────┘    │   Climb           │  │
│                          └────────────────────┘  │
└─────────────────────────────────────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────────────────────────────────────┐
│  Hardware Layer (Low-level)                      │
│  DDS Communication / Controller / Stage Observer │
│  [RoboMaster EP] [RoboMaster S1]                 │
└─────────────────────────────────────────────────┘
```

### 2.2 Modular Scheduling Layer详解

这层是"大脑"，核心是两个模块：

**Task-Planning Agent**: 基于GPT + LangChain，输入是task description（text或audio-to-text），输出是ordered subtask sequence。关键设计是**skill list prompt**——agent只能从predefined的8个meta-skills里选，这避免了LLM hallucinate出不存在的skill。

比如输入"put red can into white box"，agent会输出：
```
1. Move to red can
2. Grasp red can  
3. Move to white box
4. Position red can over white box
5. Release red can
```

如果agent分解出了新skill，会被manual refine后加入skill list——这是一种**human-in-the-loop的skill database扩展机制**。

**Execution Checker**: 这个设计非常关键。在执行每个subtask前，先用Grounding DINO v1.5检测目标object是否在scene中。比如执行"Grasp red can"前，先确认red can确实在camera视野里。这避免了skill model在object不存在时强行执行导致的catastrophic failure。

参考:
- [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300) - open-vocabulary detection
- [LangChain](https://arxiv.org/abs/2310.05421) - LLM agent framework

---

## 3. VLA Model技术细节

### 3.1 架构

```
Image (336×336) ──▶ CLIP-Large ──▶ 2 Linear Layers ──▶ Visual Embedding
                                                            │
Skill Prompt ──────────────────────────────────────────▶  Concat
                                                            │
                                                    ┌───────▼───────┐
                                                    │  Vicuna 1.5   │
                                                    │  (7B or 13B)  │
                                                    └───────┬───────┘
                                                            │
                                                    Discrete Action Tokens
                                                    (7 dims × 256 bins)
```

### 3.2 Action Discretization详解

paper的action representation公式是核心：

$$\epsilon, \Delta X, \Delta Y, \Delta\theta_{yaw}, \Delta\mu_{pos}, \Delta\nu_{pos}, \phi$$

变量解释：
- $\epsilon$: **stop signal**，决定当前skill是否完成。这个设计非常聪明——VLA model自己学会何时stop，而不是外部hardcode固定步数
- $\Delta X, \Delta Y$: ground plane上的相对位移（relative position，paper证明relative比absolute好得多）
- $\Delta\theta_{yaw}$: yaw轴rotation angle增量
- $\Delta\mu_{pos}, \Delta\nu_{pos}$: end-effector的pose参数（gripper position的二维增量）
- $\phi$: gripper binary status (open/close)

每个dimension被discretize成256 bins，这样action就变成了7个discrete tokens，可以直接用LLM的vocabulary机制处理。

**关键工程细节**: RT-2是overwrite 256个low-frequency tokens，而RoboMatrix是**新增256个special tokens**。这避免了disrupt原始vocabulary的semantic structure——理论上更clean，虽然会增加embedding table大小。

### 3.3 两阶段训练

**Stage 1: Alignment Training**
- Freeze CLIP vision encoder
- Unfreeze projection + LLM
- Co-fine-tune on LLaVA-665K (web multimodal data) + rough robot image-action pairs
- 目的：让visual embedding和robot domain对齐
- 耗时180小时，8×A100

**Stage 2: Supervised Fine-tuning (SFT)**
- Unfreeze所有参数（包括vision encoder）
- 用60K visual-action instruction data
- 1 epoch，lr=2e-5，warmup=0.01
- 耗时30小时

### 3.4 两个关键engineering tricks

**Trick 1: Relative Position Encoding**
paper发现absolute position会让model overfit、失去generalization。这很intuitive——绝对坐标依赖于camera calibration和world frame，而relative position是view-invariant的。

**Trick 2: Interval Prediction (future frame supervision)**
```
Input: frame_t
Supervision: action_{t+10}  (而不是 action_t)
```

这个trick非常elegant。如果用 $action_t$ 作为supervision，model会学到"average behavior"——预测的动作变化很小，robot移动很慢。用future frame action，model学到的是**forward-looking planning**，动作更smooth且decisive。10 frames是tuned出来的sweet spot——太小动作太慢，太大动作太aggressive导致不精确。

参考:
- [LLaVA 1.5](https://arxiv.org/abs/2310.03744) - multimodal alignment baseline
- [RT-1](https://arxiv.org/abs/2212.06817) - action discretization的先驱
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) - LLaMA2-based chatbot

---

## 4. Hybrid Model: VLA不是万能的

paper的一个重要insight：**不是所有skill都适合VLA**。对于high-determinism的skill，传统control theory + detector反而更好。

### 4.1 三种Hybrid Skills

**Searching Skill**:
```
Algorithm: Search(target)
1. Set angular velocity ω for chassis/gimbal rotation
2. While not found:
   a. Rotate by ω·dt
   b. Capture image
   c. YOLO-World.detect(target) in image
   d. If detected: stop rotation, return success
3. If 360° scanned without detection: return failure
```

**Shooting Skill** (Visual Servoing with PD control):
```
Algorithm: Shoot(target)
1. While |bbox_center - image_center| > tolerance:
   a. bbox = YOLO-World.detect(target)
   b. error = bbox_center - image_center
   c. u = K_p·error + K_d·d(error)/dt  # PD control
   d. gimbal.command(u)
2. distance = IR_sensor.read()
3. pitch_adjust = gravity_compensation(distance)
4. blaster.fire(aim=pitch_adjust)
```

**Climbing Skill**:
```
Algorithm: Climb(ramp)
1. pitch = IMU.read_pitch_angle()
2. While pitch > threshold:  # 还在斜面上
   a. pitch = IMU.read_pitch_angle()
   b. v = adaptive_speed(pitch)  # 斜度越大速度越快
   c. chassis.command(v)
3. chassis.stop()  # pitch ≈ 0, 已到平台
```

### 4.2 Hybrid Model的intuition

VLA擅长处理**unstructured environments**（object placement, orientation, category的不确定性），但对**精确control**不行。PD control对deterministic目标（如gimbal centering）的精度远超VLA。这种"right tool for right job"的hybrid design是工程上的合理选择。

参考:
- [YOLO-World](https://arxiv.org/abs/2401.17770) - real-time open-vocabulary detection
- [Visual Servoing](https://en.wikipedia.org/wiki/Visual_servoing) - classic robotics control

---

## 5. Meta-skill构建与Skill Database

### 5.1 8个VLA Meta-skills

paper提取了8个meta-skills（Figure 5）：

| Skill | Description | Input | Output Action |
|-------|-------------|-------|---------------|
| Move to `<obj>` | 移动到目标物体附近 | image + skill prompt | ΔX, ΔY, Δθ_yaw |
| Grasp `<obj>` | 抓取物体 | image + skill prompt | Δμ, Δν, φ(close) |
| Move to `<container>` | 移动到容器 | image + skill prompt | ΔX, ΔY, Δθ_yaw |
| Position `<obj>` over `<container>` | 把物体定位到容器上方 | image + skill prompt | ΔX, ΔY, Δμ, Δν |
| Release `<obj>` | 释放物体 | image + skill prompt | φ(open), ε(stop) |
| Place `<obj>` | 放置物体 | image + skill prompt | combined |
| ... | ... | ... | ... |

### 5.2 Iterative Data Collection

这是一个非常practical的data pipeline：

```
Cycle 1: 
  Collect N demos for all 8 skills (均匀分布)
  → Train VLA model
  → Test on real robot
  → Record underperforming skills

Cycle 2:
  Collect more data ONLY for underperforming skills
  → Merge into database
  → Retrain
  → Test again
  → ...

→ Converge to high-quality skill database
```

对比task-centric：如果某个task失败，你需要重新collect整个task的data（可能1000 frames）。skill-centric只需要collect那个skill的data（200 frames）。效率提升5×左右。

---

## 6. 实验结果深度分析

### 6.1 5-Level Generalization Protocol

paper基于VIMA设计了5级评估（Figure 7）：

| Level | 评估维度 | 难度描述 |
|-------|---------|---------|
| Level I | Object Gen. (easy) | Seen object, seen scene |
| Level II | Object Gen. (hard) | Unseen object, seen scene |
| Level III | Transition | Unseen object + slight scene change |
| Level IV | Scene Gen. (easy) | Seen object, unseen scene |
| Level V | Scene Gen. (hard) | Unseen object, unseen scene |

### 6.2 Task-centric vs Skill-centric (Table 1)

| Method | Dataset | L1 | L2 | L3 | L4 | L5 |
|--------|---------|----|----|----|----|----| 
| Task-Centric | Mini | 80% | 30% | 20% | 70% | 0% |
| Skill-Centric | Mini | 90% | 80% | 60% | 80% | 50% |
| Skill-Centric | Full | **100%** | **100%** | **90%** | **100%** | **80%** |

**关键观察**：
1. 在easy levels (I)，两种方法差距小（80% vs 90%）——简单任务不需要compositional reasoning
2. 在Level II（unseen object），task-centric暴跌到30%，skill-centric保持80%——这说明skill-centric学到了object-agnostic的skill representation
3. Level V（unseen scene）task-centric完全失败（0%），skill-centric有50%/80%——这是compositional generalization的真正体现
4. Mini vs Full dataset：full dataset在所有level都有显著提升，说明skill data的scaling law依然成立

### 6.3 Pretraining Ablation (Table 2)

| Method | Overall | Move cola | Grasp can | Move box | Position | Release |
|--------|---------|-----------|-----------|----------|----------|---------|
| w/o Pretrain | 30% | 50% | 80% | 40% | 30% | 90% |
| w/ Web Pretrain | 80% | 90% | 100% | 100% | 80% | 100% |
| w/ Robotics Pretrain | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |

**Interpretation**：
- "Release" skill即使在no pretrain时也有90%——因为这是最简单的skill（只是open gripper + stop）
- "Move to cola"在no pretrain时只有50%——movement需要更好的visual grounding
- Web pretrain（LLaVA-665K）大幅提升到80%——general visual-language alignment有帮助
- Robotics pretrain进一步到100%——domain-specific alignment crucial

### 6.4 Model Size Ablation (Table 4)

| Size | Move Seen | Move Unseen | Grasp Seen | Grasp Unseen | Long-Horizon |
|------|-----------|-------------|------------|--------------|--------------|
| 7B | 90% | 70% | 100% | 80% | 70% |
| 13B | **100%** | **100%** | **100%** | **90%** | **100%** |

13B在unseen scenarios和long-horizon上优势明显。这验证了VLA model也follow LLM的scaling law——bigger model, better generalization。

### 6.5 Long-Horizon Difficulty (Table 5)

| Method | Avg | Easy (3 steps) | Medium (5 steps) | Hard (>5 steps) |
|--------|-----|----------------|-------------------|------------------|
| Task-Centric | 73% | 100% | 80% | 40% |
| Skill-Centric | **93%** | 100% | **100%** | **80%** |

**这是paper最重要的结果**。Easy task两者持平，但Hard task差距是40% vs 80%——task-centric在long-horizon上指数衰减（每步成功率0.9^5 ≈ 59%），skill-centric因为可以reuse skills而保持高成功率。

### 6.6 Cross-Embodiment (Table 3)

| Method | Dataset | Task1 | Task2 | Cross-Embod |
|--------|---------|-------|-------|-------------|
| Task-Centric | Mini | 0% | 0% | 0% |
| Skill-Centric | Mini | 0% | 0% | 0% |
| Skill-Centric | Full | **40%** | **50%** | **20%** |

Mini dataset完全失败在long-horizon——data不够。Full dataset达到40-50%在复杂task上，20%在cross-embodiment（从EP robot迁移到S1 robot）。这个cross-embodiment结果虽然只有20%，但已经是significant的——说明skill representation有一定embodiment-invariance。

参考:
- [VIMA Benchmark](https://vimalabs.github.io/) - generalization evaluation protocol
- [OpenVLA](https://arxiv.org/abs/2406.09246) - open-source VLA baseline

---

## 7. 硬件与通信架构

### 7.1 RoboMaster平台

| Robot | 特色组件 | 应用场景 |
|-------|---------|---------|
| EP (Engineering) | 2-DOF robotic arm + gripper | manipulation tasks |
| S1 (Warrior) | 2-DOF gimbal + blaster | shooting tasks |

共同组件：
- Mecanum wheel chassis（omnidirectional movement）
- Monocular RGB camera (1280×720 @ 30fps)
- IMU (50Hz update)
- Audio module (2m range)

### 7.2 DDS通信架构

```
┌─────────────────────────────────────────┐
│  Cloud Server (A100 GPU)                 │
│  ┌─────────────────────────────────────┐ │
│  │ VLA Inference Server                │ │
│  └─────────────────────────────────────┘ │
└─────────────┬───────────────────────────┘
              │ LAN (DDS protocol)
              ▼
┌─────────────────────────────────────────┐
│  Robot Side (Client)                     │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Publisher     │  │ Subscriber       │ │
│  │ (sensor data) │  │ (control signals)│ │
│  └──────────────┘  └──────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │ Controller (action → motor signal) │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

DDS的decentralized特性允许**multi-robot parallel execution**——没有master node bottleneck。这和ROS2的design philosophy一致。

### 7.3 ROS2 Node Graph (Figure 11)

```
┌──────────────────┐     ┌──────────────────┐
│ robomaster_ros   │     │ robomatrix_client│
│ (basic control)  │     │ (task planning)  │
└──────────────────┘     └──────────────────┘
         │                       │
         │ ROS topics            │ ROS service
         ▼                       ▼
┌──────────────────┐     ┌──────────────────┐
│ teleoperation    │     │ robomatrix_server│
│ (joystick)       │     │ (VLA skills)     │
└──────────────────┘     └──────────────────┘
```

参考:
- [ROS2](https://docs.ros.org/en/rolling/) - Robot Operating System
- [DDS](https://www.omg.org/spec/DDS/) - Data Distribution Service

---

## 8. 我的思考与Critical Analysis

### 8.1 Strengths

1. **Skill-centric paradigm的intuition非常对**：这和human learning很像——我们学"walk"、"grasp"、"reach"这些primitive skills，然后compose它们做complex tasks。
2. **Hybrid model设计pragmatic**：承认VLA不是万能的，对deterministic task用传统control，这是工程上的mature choice。
3. **Iterative data collection**：只对underperforming skill collect data，data efficiency高。
4. **Execution Checker**：这个"check before act"的设计避免了catastrophic failure，非常practical。

### 8.2 Limitations & Open Questions

1. **Skill granularity**: 8个meta-skills是否optimal？如果task需要"pour water from cup"，这不在现有skill set里，需要manual add。automatic skill discovery是个open problem。
2. **Hierarchical planning的rigidity**: Task-Planning Agent只能从predefined skill list里选。如果需要"new emergent skill"，需要human-in-the-loop refine。
3. **Long-horizon success rate**: Hard task只有80%，考虑到每个skill ~90%成功率，5步理论上是0.9^5 ≈ 59%。他们的80%说明有某种error recovery机制？paper没详细讨论这个。
4. **Cross-embodiment只有20%**: 说明skill representation还远没有embodiment-invariant。需要更多robot platform验证。
5. **Stop signal的可靠性**: ε stop signal如果误触发会导致task premature termination。paper没讨论这个failure mode的频率。
6. **Inference latency**: VLA model在cloud server上跑，network latency ~100ms。对于需要fast reaction的task（如catching falling object）可能不够。

### 8.3 与相关work的对比

| Method | Paradigm | Generalization | Hierarchy | Skill Discovery |
|--------|----------|---------------|-----------|-----------------|
| RT-1/RT-2 | Task-centric | Low | No | No |
| OpenVLA | Task-centric | Medium | No | No |
| SayCan | LLM + skills | Medium | Yes | Manual |
| VoxPoser | LLM + 3D maps | Medium | Yes | No |
| **RoboMatrix** | **Skill-centric** | **High** | **Yes (3-layer)** | **Manual + iterative** |

RoboMatrix的独特之处是**skill database的iterative expansion机制**——这不是一次性定义skills，而是根据real-world failure feedback持续refine。

参考:
- [SayCan](https://say-can.github.io/) - LLM as planner
- [VoxPoser](https://voxposer.github.io/) - 3D value maps for manipulation
- [Code as Policies](https://code-as-policies.github.io/) - LLM generates executable code

---

## 9. Future Directions (我的speculation)

1. **Automatic skill discovery**: 用LLM分析failure modes，自动propose new skills，而不是manual refine。
2. **Skill composition learning**: 现在是sequential composition，未来可以学parallel composition（两只arm同时做不同skill）。
3. **Skill transfer across embodiments**: 需要更好的embodiment-invariant representation，可能需要learning forward/inverse dynamics in a shared latent space。
4. **Self-supervised skill refinement**: 用RL在real-world fine-tune每个skill，而不是纯imitation learning。
5. **World model integration**: 在skill之间加入world model预测，做"imagine before act"。

---

## 10. 总结

RoboMatrix的核心贡献是把robot learning从**"learn tasks"** 转向 **"learn skills + compose"**。这是一个paradigm shift，类似于从"memorize sentences"到"learn grammar + vocabulary"。三层hierarchy（scheduling / skill / hardware）让系统modular、interpretable、且可扩展。

虽然还有很多open problems（automatic skill discovery、cross-embodiment transfer、long-horizon error recovery），但这个framework的intuition是对的——compositional generalization是通向open-world robot autonomy的必经之路。

**关键takeaway**：如果你要build一个general robot agent，先想清楚你的"vocabulary of skills"是什么，再想怎么compose它们。不要试图end-to-end learn整个task space——那是dead end。

---

### Additional References

- [Mobile ALOHA](https://mobile-aloha.github.io/) - bimanual mobile manipulation
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - visuomotor policy via diffusion
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/) - 3D representations for policy
- [PaLM-E](https://palm-e.github.io/) - embodied multimodal language model
- [RT-H](https://rt-hierarchy.github.io/) - action hierarchies using language

希望这个解析帮你build起了对skill-centric robot learning的intuition！如果想dive deeper into某个具体模块（比如VLA的action discretization、DDS通信细节、或者skill database的iterative pipeline），我可以再展开。
