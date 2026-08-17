---
source_pdf: PrimitiveVLA Learning Reusable Motion Primitives for Efficient and Generalizable
  Robotic Manipulation.pdf
paper_sha256: e0dd9eac616991b2822a820352d9cad045d68260d2975d4ef51f11ac32fa3e35
processed_at: '2026-08-06T06:10:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用人话来说，这篇 Paper 处理的是机器人学习的一个核心痛点：**现在的 VLA 模型太像“死记硬背的学渣”了。**

想象一下你教一个小孩做菜。如果你让他把“番茄炒蛋”的每一个微小肌肉动作从头到尾死记硬背下来，下次你让他做“番茄汤”，他就完全懵了，因为他觉得这是两个完全不同的宇宙级任务。哪怕切番茄、倒水这些基础动作他其实早就学过了，但他无法把长任务拆解成基础动作来复用。

现在的 VLA（Vision-Language-Action）模型比如 OpenVLA 或者 π₀.5，就是这种状态。它们采用一种 Direct Instruction-to-Control Mapping 的范式：输入“把碗放进抽屉”，模型直接去预测一整条长长的 action trajectory。这导致模型把所有的视觉场景、语言指令和底层动作全部 entangle（纠缠）在一起。你教它开了微波炉，它依然学不会开抽屉，哪怕这两个动作在物理层面都是“把手往外 Pull”。

PrimitiveVLA 的核心 intuition 就一句话：**别让机器人背整篇课文，逼它先学会“偏旁部首”和“拼音”，然后通过拼装来写新文章。** 

作者提出了一套 Disassemble & Assemble（拆解与拼装）的范式，把所有复杂的机器人操作任务，拆解成 11 个最基础的物理动作原语，比如 Grasp（抓）、Move（平移）、Place（放）、Pull（拉）、Twist（拧）。你只要把这 11 个基础动作练熟，理论上就能组合出无穷无尽的新任务。

为了实现这个直觉，这套框架干了三件非常巧妙的事，我给你扒一扒里面的技术细节：

### 1. 怎么把长视频切碎？

公开数据集里只有任务级别的标签（比如“把红色的杯子放到右上角”），没有告诉你哪一秒在做 Grasp，哪一秒在做 Place。人工去标太贵了。

作者的解法是搞了一个自动化的“智能剪刀”流水线：
**第一步：用 VLM 当导演，看剧本。** 把任务指令和采样下来的 RGB 图片序列喂给 Qwen3-VL，让它推理出这个任务大体上经历了哪些动作。比如看到“开抽屉放碗”的视频，VLM 会输出一个纯文本序列：$[Pull, Grasp, Lift, Place, Push]$。这就给了一个宏观的骨架，防止把人在操作时手抖产生的微小杂音误切分成多余的动作。

**第二步：用 LLM 当剪辑师，精确找切口。** 知道了动作顺序，怎么找到精确的时间帧 $t$ 来切分？作者让 DeepSeek-V3 写 Python 代码来找。切分的数学公式是：
$$t_{\mathrm{end}} = \min\{t \mid t > t_{\mathrm{start}} + \delta, \phi_i(s_{t-k:t+k}) = \mathrm{True}\}$$
这里的变量意思很直白：
- $t_{\mathrm{start}}$ 是当前动作开始的时间点。
- $\delta$ 是一个时间缓冲期（设为 10 步），防止动作刚一开始就误判结束。
- $\phi_i$ 是 LLM 生成的 Python 切分函数。
- $s_{t-k:t+k}$ 是一个大小为 $k$ 的滑动窗口里的机器人本体感觉数据，包含夹爪开合和 6-DoF 坐标。

比如要切分 Grasp 动作，LLM 生成的 $\phi_i$ 逻辑就是：“如果在这个滑动窗口里，夹爪宽度保持不变（说明夹住了），而且 z 轴高度开始显著上升（说明提起来了），这就满足切分条件 True”。这种基于物理状态的代码极其精确，避开了 VLM 黑盒预测时间边界的不稳定性。

### 2. 怎么解决“杯子”和“碗”的视觉干扰？

如果你直接用拆出来的数据去微调模型，模型会过拟合于具体的物体。怎么让模型学到纯粹的“抓取”动作，而不在乎抓的是杯子还是碗？

作者搞了一个叫 **MCR (Multimodal Canonical Representation)** 的东西。直觉就是把视觉和语言全部标准化。
**语言层面：** 别再说“抓起黑色的碗”了，统一改成说黑话：“Grasp the object with color_1”。
**视觉层面：** 利用 SAM (Segment Anything Model) 和 Cutie 视频追踪模型，把目标物体在画面里抠出来，打上 color_1 的纯色块遮罩。原始图像 $o_i$ 和遮罩 $M_i$ 做逐元素相乘 $\tilde{o}_i = o_i \otimes M_i$。

经过 MCR 处理后，模型看到的画面就是几个色块，听到的指令是“抓那个色块”。Task-specific 的环境信息全被抹掉了，模型只能去学“靠近色块 -> 闭合夹爪 -> 抬起”这个纯粹的物理运动规律。这就是 MCR 能让 OOD（分布外）泛化能力暴涨的核心原因。

### 3. 推理时怎么把这些动作拼起来执行？

到了真机运行或者测试新任务时，模型没见过这个场景，怎么把学到的偏旁部首写成一篇文章？

作者设计了一个双线程的 Inference 机制。
**规划线程：** 遇到新指令，先用 Retrieval-Augmented Planning 从训练时的 Disassembly Library 里检索 top-3 最相似的任务给 VLM 当例子，防止 VLM 瞎编。VLM 规划出新任务的动作序列 $\mathcal{S}_{\mathrm{plan}}$。

**执行与监控线程：** 机器人的 VLA 大脑在一个线程里源源不断地输出 action，另一个线程在后台跑 LLM 生成的轻量级 Python 监控脚本。由于推理时没有未来状态，监控脚本只能看过去 10 帧的 History Window $\mathcal{H}_t$ 算变化率。公式 $\mathrm{SwitchTrigger} \iff \forall s \in s_{\mathrm{stat}}, f_{\mathrm{switch}}(s, p_i) = \mathrm{True}$ 意思是：当滑动窗口里所有统计指标（比如 z 轴速度）都满足当前动作的结束条件时，就瞬间把指令切换到下一个动作。这就实现了一个鲁棒的闭环控制。

---

### 数据验证的 Intuition

你去看实验数据，这套 Intuition 是完全 Work 的，甚至有点惊人：

**Data Efficiency（数据效率）爆表：** 
看 Table 4，OpenVLA 用 100% 的数据训练，平均成功率是 78.70%。但是加上 PrimitiveVLA 框架，**只用 50% 的数据**，成功率就达到了 80.30%，反而超过了 baseline。这就好比你只给小孩看一半的做菜视频，但他因为掌握了切菜、翻炒的基础动作，做出来的菜比死记硬背整个视频的小孩还要好。因为 Libero-90 里有 86 个任务都要用到 Grasp，拆解后 Grasp 这个 primitive 被疯狂重复训练，形成了密实的高质量监督信号。

**Long-Horizon Generalization（长程泛化）起飞：**
看 Table 3，在长序列任务 Libero-Long 上，当前的 SOTA 模型 $\pi_{0.5}$ 成功率只有 30.50%。加上 PrimitiveVLA 之后，直接干到了 **80.25%**。因为对于直接映射的模型，多走几步误差就雪崩了；而 PrimitiveVLA 每次只专注于执行好当前的一个小 primitive，执行完再切下一个，大大降低了长程累积误差。

**Error Mode 的降维打击：**
我最喜欢的是 Figure 8 里的 failure mode 分析。OpenVLA 这个 baseline 在遇到没见过的任务时，65% 的失败是因为 Imitative Interference（模仿干扰：让它拿右边的碗，它非去拿左边训练时见过的碗），23% 是 Frozen Behaviors（吓僵了不知所措一动不动）。
而 PrimitiveVLA 这两类“低级认知错误”直接降到了 12% 和 10.1%。它的主要错误变成了 Action Failures（动作执行变形，占 60.4%）。这说明什么？说明模型在心智层面上已经完全理解了新任务要干嘛，它真真切切地在尝试组合新动作，仅仅是因为底层物理执行不够完美而失败。这是从“智障”到“笨手笨脚”的质变。

总之，这篇 paper 在向整个 Embodied AI 社区传递一个 Karpathy 式的直觉：**别再迷信把模型做大、把数据堆多了。如果你对 action space 本身缺乏结构和先验，模型就只能做个大海绵，毫无组合泛化能力。把轨迹拆解成 motion primitives，就等于给机器人的 action 建立了 sub-word tokenization，这才是通向 generalist robot 的正确路线。**

Paper link: [PrimitiveVLA arXiv](https://arxiv.org/abs/2505.09689) (注: 链接基于arXiv常见格式推断)
Related Codebase (SAM): [Segment Anything](https://github.com/facebookresearch/segment-anything)

---

# PrimitiveVLA 深度解析:从 Monolithic Trajectory 到 Reusable Motion Primitives

## 1. 核心问题的 Intuition

当前 VLA (Vision-Language-Action) model 比如 RT-2, OpenVLA, π₀.5 在 fine-tuning 时存在一个 fundamental flaw,作者称之为 **Direct Instruction-to-Control Mapping**。这个 paradigm 把 high-level instruction 直接映射到 low-level action sequence,导致 model 学到的是 scene-specific 的 monolithic trajectory,而不是 reusable 的 motion pattern。

设想一个简单场景:"open the cabinet" 和 "open the microwave" 这两个 task,物理层面共享 "pull" 这个 primitive,但在 Direct Mapping paradigm 下,model 会把它们当作两个独立 task 来 memorize。当新 task "open the drawer" 出现时,model 没有结构化的 mechanism 去 reuse 之前学到的 "pull" pattern,只能从头学起。这就是为什么 data efficiency 差,generalization 弱。

PrimitiveVLA 的核心 insight:把学习对象从 "task-specific trajectory" 转移到 "task-agnostic primitive",通过 **Disassemble & Assemble** paradigm,让 VLA master 一组 reusable 的 motion primitives,再在 inference 时 assemble 它们来解决 novel task。

参考链接:
- OpenVLA paper: https://arxiv.org/abs/2406.09246
- RT-2 paper: https://robotics-transformer2.github.io/
- π₀ paper: https://arxiv.org/abs/2410.24164

---

## 2. Primitive Library 设计

Table 1 定义了 11 个 reusable primitives,分为三大类:

**Spatial Transport (空间搬运类)**
- **Grasp**: approach + seize + preliminary lift,关键 kinematic feature 是 gripper 闭合后伴随 z 轴小幅上升
- **Place**: descend + release + vertical retreat,gripper 打开后 z 轴抬升脱离
- **Lift**: 持续 z 轴正方向位移,gripper 保持闭合
- **Move**: xy 平面大位移,gripper 闭合状态保持

**Contact & Interaction (接触交互类)**
- **Push**: 物体在 surface 上滑动,gripper 未闭合,z 轴基本不变
- **Pull**: 物体朝 robot 方向被拖拽
- **Insert**: 物体或 gripper 对齐并插入 constrained slot
- **Press**: 向下施加 force

**Orientation (姿态调整类)**
- **Twist**: gripper 绕自身中心轴旋转 (roll),用于 knob 类机构
- **Tilt**: 绕 wrist 之外关节旋转 (pitch/yaw)
- **Rotate**: 沿物体固定轴旋转,比如开 laptop lid

这个 taxonomy 的设计 intuition 是:这些 primitives 覆盖了 standard gripper-based manipulation 的主要 kinematic mode,而且每个 primitive 都能用 proprioception state 的统计特征来 deterministic 地检测边界。比如 Grasp 的检测条件就是"gripper width 保持恒定 ($w < \epsilon$) 同时 end-effector height 开始增加 ($\Delta z > \delta$)"。

---

## 3. Problem Formulation 数学解析

### 3.1 Base Formulation

在 step $t$,policy 观测到 state tuple $(o_t, s_t)$,其中:
- $o_t = \{I_t^{(global)}, I_t^{(wrist)}\}$,分别是第三人称视角 RGB image 和 wrist-mounted camera 的 RGB image
- $s_t \in \mathbb{R}^7$ 是 proprioceptive state,包含 6-DoF end-effector pose 和 gripper state
- $a_t \in \mathbb{R}^7$ 是 action,包含 6-DoF delta pose 和 gripper action

一个完整 demonstration 是 trajectory:
$$\tau = \{(o_t, s_t, a_t)\}_{t=1}^{T}$$

### 3.2 两种 Paradigm 的 Loss 对比

**Direct Instruction-to-Control Mapping (传统方法)**:
$$\mathcal{L}_{task} = -\sum_{t=1}^{T} \log \pi_\theta(a_t \mid o_t, s_t, l)$$

这里 $l$ 是 high-level task instruction,$T$ 是 trajectory 总长度。这个 loss 把整个 trajectory 作为一个 indivisible unit 来优化,instruction $l$ 和 action $a_t$ 之间是 tight coupling。model 倾向于 memorize "看到这个 scene + 这个 instruction → 输出这个 action" 的 mapping,而不是理解 underlying motion pattern。

**Primitive-Centric Disassemble & Assemble (PrimitiveVLA)**:
$$\mathcal{L}_{prim} = -\sum_{i=1}^{N} \log \pi_\theta(a_i \mid \tilde{o}_i, s_i, c_i)$$

变量含义:
- $N$ 是 trajectory 被切分后的 primitive 数量
- $\tilde{o}_i$ 是经过 mask 处理后的 observation (MCR 的一部分)
- $c_i$ 是 canonical primitive instruction,比如 "grasp the object with green mask"
- $i$ 索引的是 primitive 而非 raw timestep

关键区别在于 condition 变成了 canonical instruction $c_i$ 而非 task-specific instruction $l$,observation 经过 mask 处理保留了 task context 但去除了 scene-specific 视觉干扰。这个 reformulation 让 model 学的是 "给定这个 primitive type + 这个 object mask → 执行这个 motion pattern"。

---

## 4. Framework 架构详解 (Figure 2 解析)

PrimitiveVLA 框架分为两个 phase,通过 shared **Multimodal Canonical Representation (MCR)** 桥接:

### 4.1 Fine-tuning Phase: Primitive Disassembly

这个 phase 的目标是把 monolithic trajectory $\tau$ 自动切分成 primitive-aligned samples $\dot{\tau} = \{(\tilde{o}_i, s_i, a_i, c_i)\}_{i=1}^{N}$。分两步走:

**Step 1: Primitive Sequence Reasoning (Section 4.1.1)**

用 VLM (具体是 Qwen3-VL) 作为 reasoning engine,输入:
- Task instruction $l$
- Example RGB sequence $\mathcal{V}_\tau$ (从 demonstration 中采样)
- Primitive library $\mathcal{C}$

输出 primitive sequence:
$$\mathcal{S} = f_{\mathrm{VLM}}(l, \mathcal{V}_\tau, \mathcal{C})$$

$\mathcal{S} = [p_1, p_2, \dots, p_k]$ 是 ordered primitive list,但**没有 boundary 信息**,只给出 temporal order。比如 task "open the top drawer and put the bowl in it" 输出 $[pull, grasp, lift, place, push]$。

这些 $(l, \mathcal{S})$ pairs 被存入 **Disassembly Library** $\mathcal{D}$,作为 inference phase 的 retrieval database。

为什么需要 VLM 做这一步?因为纯 state-based segmentation 对 unstructured trajectory 中的 jitter 非常敏感。人在做 "move" 时手会轻微抖动,在 proprioception data 里可能表现为微小 rotation 或 height change。没有 semantic prior 的话,这些抖动会被错误地切分成独立的 "Rotate" 或 "Adjust" primitive。VLM 从 visual + semantic 角度先确定 macroscopic task flow,filter 掉这些 noise。

**Step 2: State-Based Boundary Segmentation (Section 4.1.2)**

有了 primitive sequence $\mathcal{S}$,下一步是 localize 每个 primitive 的 start 和 end point。作者用 LLM (DeepSeek-V3) 自动生成 Python code $\phi_i$ 来定义每个 primitive 的 termination criteria:

$$t_{\mathrm{end}} = \min\{t \mid t > t_{\mathrm{start}} + \delta, \phi_i(s_{t-k:t+k}) = \mathrm{True}\}$$

变量含义:
- $t_{\mathrm{start}}$ 是当前 primitive 的起始 timestep
- $\delta$ 是 temporal offset (paper 中设为 10 steps),避免在 primitive 刚开始时就误触发
- $\phi_i$ 是 primitive $p_i$ 对应的 Python function,输入是 local window 的 state $s_{t-k:t+k}$ (前后各 $k$ 帧的滑动窗口)
- $k$ 是 window size,用于 capture motion dynamics 而非瞬时 noise

为什么用 LLM 生成 code 而非手动写?因为手动为每个 task 写 segmentation code 等于为每个 task 设计 reward function,不可 scale。LLM 接收 primitive 的 physical definition 和 generic segmentation criteria,生成 task-agnostic 的 segmentation code。这样 code 关注的是 "grasp 这个 primitive 的物理特征" 而非 "pick up the cup 这个 task 的特征"。

Figure 3 (Left) 的 Algorithm 展示了完整的 segmentation 逻辑:从 $t_0 + \delta$ 开始遍历,当 $\Phi(T, t)$ 返回 True 时返回 $t$ 作为 boundary。

### 4.2 Inference Phase: Primitive Assembly

这个 phase 的目标是在 closed-loop 环境中,把学到的 primitives assemble 成 coherent task execution。

**Step 1: Primitive Planner (Section 4.2.1)**

给定 test instruction $l_{\mathrm{test}}$,需要生成 primitive sequence $\mathcal{S}_{\mathrm{plan}}$。直接用 VLM 生成会有 hallucination 风险,比如生成训练时没见过的 primitive 组合。解决方案是 **Retrieval-Augmented Planning**:

$$\mathcal{S}_{\mathrm{plan}} = f_{\mathrm{VLM}}(l_{\mathrm{test}}, o_0, \mathcal{C}, \mathrm{Retrieve}(l_{\mathrm{test}}, \mathcal{D}))$$

$\mathrm{Retrieve}(l_{\mathrm{test}}, \mathcal{D})$ 从 Disassembly Library $\mathcal{D}$ 中通过 semantic cosine similarity 检索 top-3 最相似的 $(l, \mathcal{S})$ pairs。这些 retrieved exemplars 作为 in-context learning 的 prior,constrain VLM 生成的 sequence 不偏离 fine-tuning distribution。

**Step 2: Primitive Switch (Section 4.2.2)**

这是最 tricky 的部分:在线执行时如何决定何时从 primitive $p_i$ 切换到 $p_{i+1}$?与 fine-tuning 时不同,inference 时**没有 future states**,不能用对称的 temporal window。

解决方案是用 LLM (DeepSeek V3) 基于之前生成的 segmentation code $\phi_i$ 生成 real-time switch code $f_{\mathrm{switch}}$。这个 code 只用 history sliding window $\mathcal{H}_t$ (size $W$,paper 中是 10 帧):

$$\mathrm{SwitchTrigger} \iff \forall s \in s_{\mathrm{stat}}, f_{\mathrm{switch}}(s, p_i) = \mathrm{True}$$

变量含义:
- $s_{\mathrm{stat}}$ 是从 history window $\mathcal{H}_t$ 计算的 statistical trends (比如 rate of change)
- $f_{\mathrm{switch}}$ 是针对 primitive $p_i$ 的 switch condition function
- $\forall$ 表示所有统计指标都满足条件才触发 (AND logic)

Figure 3 (Right) 的 `CHECKSWITCH` function 展示了具体逻辑:如果 history window 不足 WindowSize 则返回 False (防止冷启动),否则计算 statistics $s_{\mathrm{stat}} \leftarrow \mathrm{CALCSTATISTICS}(\mathcal{H}_t)$,然后检查 $\Phi(s_{\mathrm{stat}})$ 是否满足。

执行时采用 **dual-threaded architecture**:
- Execution thread: VLA model 持续输出 action
- Monitoring thread: 并发评估 switch trigger

这样 task progression 不阻塞 action 输出,实现 reactive closed-loop execution。

### 4.3 Multimodal Canonical Representation (MCR)

MCR 是连接 fine-tuning 和 inference 的核心桥梁,解决 **Contextual Interference** 问题:primitives 需要在不同 task 间保持 consistent 且 reusable,但 task-specific details 会干扰 primitive learning。

**Semantic Unification (语义统一)**

把所有属于同一 primitive $p_i$ 的 sample 映射到单一 canonical instruction $c_i$。Table 8 给出完整 mapping:

| Primitive | Canonical Instruction |
|-----------|------------------------|
| grasp | "Grasp the masked object with color_1" |
| lift | "Lift the object with color_1" |
| move | "Move to above the object with color_2" |
| place | "Place in the object with color_2" |
| push | "Push the object with color_1" |
| ... | ... |

这里 $color_1$ 和 $color_2$ 是 semantic mask 的 color index,分别代表被操作物体和目标位置。

Intuition:"grasp the black bowl" 和 "grasp the red mug" 都映射成 "grasp the object with color_1"。这样 model 学到的是 "grasp" 这个 motion pattern 本身,而非特定物体与 action 的 association。

**Visual Compatibility (视觉兼容)**

用 mask 处理 observation:
$$\tilde{o}_i = o_i \otimes M_i$$

变量含义:
- $o_i$ 是 raw RGB observation
- $M_i$ 是 object-centric mask,通过 SAM (Segment Anything Model) 生成初始 mask,然后用 Cutie (video object segmentation) 在 trajectory 中持续 tracking
- $\otimes$ 是 element-wise multiplication (mask 应用)

这个 mechanism 的 intuition 是:mask 成为 task-specific context 的 primary carrier,而 VLA input 保持 uniform representation。model 通过 mask 知道 "操作哪里",通过 canonical instruction 知道 "做什么 motion",两者解耦。

参考链接:
- SAM: https://arxiv.org/abs/2304.02643
- Cutie: https://arxiv.org/abs/2310.11482

---

## 5. 实验数据深度分析

### 5.1 Data Efficiency (RQ1) - Table 4 解析

Table 4 是理解 PrimitiveVLA 价值的关键。看 OpenVLA backbone 的数据:

| Setting | Libero-Object | Libero-Spatial | Libero-Goal | Libero-90 | Mean |
|---------|----------------|-----------------|-------------|-----------|------|
| OpenVLA 50% | 83.40% | 73.40% | 73.60% | 65.89% | 74.07% |
| OpenVLA 100% | 87.40% | 82.80% | 74.00% | 70.60% | 78.70% |
| OpenVLA + Ours 50% | 87.60% | 87.00% | 73.20% | 73.40% | 80.30% |
| OpenVLA + Ours 100% | 90.60% | 91.20% | 82.20% | 79.80% | 85.95% |

关键观察:**PrimitiveVLA 用 50% data (80.30%) 超过了 baseline 用 100% data (78.70%)**。这是 data efficiency 的强证据。

为什么能这样?Section D.2 给出三个 mechanism:

1. **Motion pattern 集中 supervision**:Libero-90 有 90 个 task,但 Table 12 显示 "grasp" 这个 primitive 在 86 个 task 中都出现。传统 VLA 要学 90 个 "instruction → trajectory" mapping,每个 mapping 数据稀疏。PrimitiveVLA 把 sparse task distribution 重塑成 dense action distribution,同一个 grasp motion pattern 被反复 reinforce。

2. **Structural alignment**:Libero-Spatial 和 Libero-Object 的 challenge 主要是环境变化 (物体位置/类别) 而非逻辑复杂度。PrimitiveVLA 把这些 task 统一成 $[grasp, move, place]$ sequence,把学习负担从 "理解多阶段语言逻辑" 转移到 "master 短程 primitive"。

3. **Equal training weight for short primitives**:传统 VLA 用 uniform temporal down-sampling,但 $lift$, $press$ 这类 short-duration action 会在长 trajectory 中被 "稀释"。PrimitiveVLA 把每个 primitive 切成独立 training unit,short primitive 得到 equal weight。

### 5.2 Cross-Task Generalization (RQ2) - Table 3 解析

**Libero-90-Novel (unseen tasks)**:
- OpenVLA: 7.38% → PrimitiveVLA: 45.50% (6× improvement)
- OpenVLA-OFT: 13.50% → 71.00%
- π₀.5: 56.00% → 75.50%

**Libero-Long (long-horizon)**:
- OpenVLA: 4.50% → PrimitiveVLA: 38.50%
- OpenVLA-OFT: 3.75% → 66.50%
- π₀.5: 30.50% → 80.25% (SOTA 从 30.50% 提升到 80.25%)

π₀.5 在 long-horizon 上的提升尤其惊人。Section D.1 的 failure mode 分析 (Figure 8) 给出 intuition:

**Baseline (OpenVLA)** 的 failure 分布:
- Imitative Interference: 65.0% (看到 OOD task 错误地执行 ID 中相似 task)
- Frozen Behaviors: 23.1% (任务太复杂,model "冻结" 不动作)
- 这两类 high-level cognitive failure 占 88%+

**PrimitiveVLA** 的 failure 分布:
- Imitative Interference: 12.0% (大幅下降)
- Frozen Behaviors: 10.1% (大幅下降)
- Action Failures: 60.4% (成为主要 error mode)
- Primitive Switching: 6.5%
- Motion Connection: 6.9%

这个 error distribution 的 migration 非常有启发性:PrimitiveVLA 把 error 从 "high-level 语义混淆" 迁移到 "low-level 执行挑战"。Model 不再 "不知道做什么",而是 "知道做什么但执行不够完美"。这是 generalization 能力提升的 direct evidence —— model 真的在尝试 novel task,只是 motion execution 还有改进空间。

### 5.3 Ablation Study (RQ3) - Table 6 解析

Table 6 解耦 Primitive Disassembly 和 MCR 的贡献:

**OpenVLA-OFT backbone**:
| Setting | Libero-90 (ID) | Libero-90-Novel (OOD) | Libero-Long |
|---------|-----------------|------------------------|-------------|
| Baseline | 89.70% | 13.50% | 3.75% |
| w/o MCR (Disass. Only) | 89.60% | 15.00% | 52.30% |
| w/o Disass. (MCR Only) | 94.30% | 60.00% | 39.75% |
| Ours (PrimitiveVLA) | 94.70% | 71.00% | 66.50% |

关键 insight:
- **MCR 是 OOD transfer 的 primary driver**:w/o Disass. (MCR Only) 把 OOD 从 13.50% 提到 60.00%。MCR 通过 canonical instruction 和 mask 把 task context 标准化,让 model 能 transfer 学到的 motion pattern 到新 scene。
- **Primitive Disassembly 是 long-horizon stability 的关键**:w/o MCR (Disass. Only) 把 Long 从 3.75% 提到 52.30%。Disassembly 让 multi-stage task 变成 primitive sequence,每个 primitive 是 short-range,容易学习,switching mechanism 管理 transition。
- **两者结合产生 synergistic effect**:PrimitiveVLA (94.70% / 71.00% / 66.50%) 在所有维度都超过 single-component ablation,说明 Disassembly 和 MCR 是 complementary 的。

### 5.4 RLBench 鲁棒性 - Table 5

RLBench 数据 diversity 更高,10 个 task (T1-T10):
- OpenVLA: 49.5% mean
- OpenVLA + Ours: 56.5% mean (+7.0%)

某些 task 提升明显:T7 (Take Off Scales) 15% → 40%,T8 (Take Lid Off Saucepan) 95% → 100%,T10 (Take Umbrella) 55% → 65%。

### 5.5 Real-World Evaluation - Table 7

UR5e + Robotiq 2F-85 gripper,11 个 task 分三类:

**In-Distribution (T1-T6)**:
- π₀.5 baseline: 70% mean
- π₀.5 + Ours: 90% mean

**Task Generalization (T7-T9, OOD)**:
- π₀.5: 20% mean
- π₀.5 + Ours: 57% mean

T9 (Pick blue cup, novel object) 提升惊人:10% → 80%。

**Compositional Generalization (T10-T11)**:
- π₀.5: 10% mean (T10 完全失败 0%)
- π₀.5 + Ours: 65% mean (T10: 60%, T11: 70%)

这证明 real-world 中 primitive 组合也 work,即使 baseline 在 compositional task 上完全无法启动。

---

## 6. Latency 和 Computational Overhead (Table 13)

| Model | Chunk Size | Baseline Latency | PrimitiveVLA Latency |
|-------|------------|------------------|----------------------|
| OpenVLA | 1 | ~500 ms | ~540 ms |
| OpenVLA-OFT | 5 | ~88 ms | ~96 ms |
| π₀.5 | 10 | ~67 ms | ~72 ms |

额外 overhead 来自:
- **Cutie mask tracking**: 256×256 分辨率 9ms/frame,768×768 分辨率 30ms/frame
- **Switch logic**: Python based,极轻量
- **VLM/LLM planning**: 在 pre-execution 阶段完成,real-time 无开销

对于 chunk size 较大的 VLA (如 π₀.5 用 chunk=10),每个 chunk 只更新一次 mask,overhead 可忽略 (67ms → 72ms,仅 +5ms)。

---

## 7. 与 Related Work 的关键对比

### 7.1 vs 传统 Hierarchical RL (SPiRL, OPAL, BeT, VQ-BeT)

这些方法用 continuous latent 或 vector quantization 做 unsupervised skill discovery。问题是没有 explicit semantic supervision,学到的 latent $z$ 或 code 缺乏 interpretable, reusable 的 physical definition。比如 VQ-BeT 的 codebook 可能学到 "motion cluster 1, 2, 3",但没有 "这是 grasp,那是 place" 的 semantic label,无法用于 VLA fine-tuning 中的 instruction grounding。

PrimitiveVLA 用 explicit primitive taxonomy (11 个) + VLM semantic reasoning,每个 primitive 都有清晰 physical definition,可以直接对应 canonical instruction。

### 7.2 vs LLM-based Planner (SayCan, Code as Policies)

SayCan 和 Code as Policies 用 LLM 做 high-level planning,但只停留在 symbolic level,没有 low-level action execution capability。它们需要 separate trained low-level policy。

PrimitiveVLA 把 high-level reasoning (VLM planner) 和 low-level control (VLA model) 整合在统一框架内,通过 MCR 保持两端 representation consistency。

### 7.3 vs π₀.5

π₀.5 (reference [6],实际是 π₀.7) 用 world model 引入 predictive information 做 hierarchical control,但 fine-tuning 仍 maintain Direct Instruction-to-Control Mapping,instruction 和 action 紧耦合。

PrimitiveVLA 在 π₀.5 上应用后,Libero-Long 从 30.50% 飞跃到 80.25%,证明即使 SOTA foundation model 也能从 Disassemble & Assemble paradigm 获益。

### 7.4 vs Pivot-R, Manipulate Anything

这些方法只用 instruction 和 image 预测 segment,无法生成 fine-tuning 所需的 action-aligned segmentation。PrimitiveVLA 用 state-based code 做 boundary detection,产生 action-aligned 的精确 boundary,可以直接作为 fine-tuning sample。

参考链接:
- SayCan: https://proceedings.mlr.press/v205/ichter23a.html
- Code as Policies: https://arxiv.org/abs/2209.07753
- π₀.5: https://arxiv.org/abs/2504.16054

---

## 8. Limitations 和 Future Direction

**当前 limitation**:
1. Primitive set 依赖 pre-defined kinematic taxonomy,11 个 primitive 对 standard gripper 够用,但无法覆盖 dexterous manipulation (比如 in-hand rotation, multi-finger coordination)
2. Switch logic 是 heuristic based,在某些 OOD scenario 会出现 Primitive Switching error (6.5%) 和 Motion Connection error (6.9%)
3. VLM/LLM 偶尔产生 noise (<5% in LIBERO, <10% in RLBench),主要是 primitive merging 和 mask drift

**Future work 方向**:
1. Unsupervised primitive disassembly:动态扩展 action space,自动发现新 primitive
2. End-to-end differentiable planning:让 high-level reasoning 和 low-level control 联合优化,消除 heuristic switch 的 gap
3. 更大的 primitive library:覆盖 dexterous manipulation

---

## 9. 个人 Intuition 总结

从 Karpathy 的视角看,这篇 paper 的核心贡献是把 "learning to act" 从 "memorize trajectory" 重塑为 "master reusable motion vocabulary + compose them"。这让我联想到 NLP 中 word embedding 的 intuition:与其让 model memorize 每个 sentence,不如让它先 master word,再通过 composition 处理任意 sentence。

PrimitiveVLA 的 11 个 primitive 就像 robot action 的 "word",MCR 是标准化的 "tokenize" process,Disassembly Library 是 "corpus",VLM planner 是 "language model" 生成 "sentence" (primitive sequence),switch controller 是 "parser" 管理 "word boundary"。

这种 paradigm shift 的威力在数据上得到验证:50% data 超过 baseline 100% data,OOD task 6× improvement,long-horizon 从 30.50% 到 80.25%。这说明 motion pattern 的 reuse 是 robotic manipulation 的关键 inductive bias,而传统 Direct Mapping 浪费了大量 capacity 在 scene-specific memorization 上。

值得注意的 design choice:作者没有完全抛弃 VLM/LLM 的 semantic reasoning 能力,而是用 VLM 做 high-level planning (capturing semantic structure) + 用 rule-based code 做 low-level boundary detection (ensuring precision)。这种 hybrid approach 避开了 VLM 作为 black-box segmentation tool 的 stochasticity 问题 (Appendix B 详细对比 Qwen3.5-Omni-Plus 直接做 segmentation 的失败案例)。

对 embodied AI 社区的启示:future VLA model design 应该更多考虑 "action vocabulary" 的结构化,而非单纯 scale up model size 或 dataset size。PrimitiveVLA 证明,正确的 problem formulation (disassemble & assemble) 比 brute-force data scaling 更高效。

Paper link: https://arxiv.org/abs/2505.09689 (注:实际 link 可能需验证)
Code (如开源): 建议查看作者 GitHub https://github.com/PengShaohui 或 ICT/CAS 相关 repo
