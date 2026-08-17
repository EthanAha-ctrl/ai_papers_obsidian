---
source_pdf: VP-VLA Visual Prompting as an Interface for.pdf
paper_sha256: 3571447dfce7138acb74f7c38e5225a39e24752bb968aacfa5785fa01e89b3ae
processed_at: '2026-08-13T03:26:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VP-VLA

好，Andrej，咱们抛开那些公式和表格，用大白话聊。

---

## 一句话讲清楚这 paper 在干嘛

现在所有 VLA 模型都犯一个毛病——**让一个人同时干三件事**：听懂你说啥、找到要操作的东西在哪、再控制机械臂精确地动。这三件事谁都难，塞到一个 forward pass 里，结果就是哪件都干不好。

VP-VLA 的核心 idea 特别朴素：**把它拆成两个人，一个负责想，一个负责动手，中间用"画图"来沟通**。

---

## 用个你熟悉的类比

想象你教一个学徒开车去某个陌生地方。

**现在的 VLA 做法**：你跟学徒说"去机场"，然后指望他一把方向盘打到底，同时脑子里还得认路、避让、换挡。这肯定乱套。

**VP-VLA 的做法**：你旁边坐一个"导航员"（System 2），他看一眼地图，然后在车窗玻璃上画个十字标和框——"看，往这个方向开，停在这个区域里"。学徒（System 1）只需要盯着玻璃上的标记，把它当目标追着开就行。

导航员不每秒都说话，只在**关键节点**（比如拐弯、到达路口）才开口。学徒一直专心开车。

这个"画在玻璃上的标记"，就是 paper 里的 **visual prompt**——一个 crosshair（十字标）表示"抓这里"，一个 bounding box 表示"放这儿"。

---

## 为什么要这么拆？

Paper 开头有个特别扎心的发现：**你把 instruction 换成乱码，模型性能居然没怎么变**。

这说明啥？说明模型压根没在"听懂"语言，它在 overfit——记住训练分布长啥样了。换个新物体、换个没见过的位置，立刻翻车。

根本原因是 **language 和 action 这俩东西形状对不上**。language 是符号的、抽象的，action 是连续的、几何的。你让一个 network 直接从"把杯子放到盘子里"映射到7个关节角度，中间那个 gap 太大了，model 只能瞎猜。

VP-VLA 的 insight 是：**先把 language 变成 visual 空间里的东西**（一个十字、一个框），然后再从 visual 到 action。后一步其实 model 本来就擅长（visuomotor tracking，跟 mice 抓老鼠似的本能）。前一步交给一个专门负责"想"的 VLM。

这就把一个巨难的问题，拆成俩中等难的问题。

---

## 两个人怎么配合？

### System 2（"想"的人）

用的是 Qwen3-VL-4B，一个 4B 参数的 VLM。它干两件事：

**第一件：把复杂任务拆成小步骤。**

你说"拿起红酒放进柜子然后关柜门"，它拆成：["拿红酒", "放红酒进柜子", "关柜门"]。这一步叫 task decomposition。

**第二件：告诉 System 1 该看哪儿。**

它在当前画面里找到"红酒"在哪、"柜子"在哪，然后用 SAM3 这个 segmentation 模型把红酒抠出来，在它中心画个十字；把柜子放酒的那个区域框个框。

这两步加起来，把一句抽象的 instruction 变成了 image 上两个具体的几何标记。

### 关键设计：什么时候叫醒 System 2？

如果每帧都叫醒它，太慢（4B 模型 inference 比 action MLP 慢几个数量级）。所以 paper 用了个很巧的 trigger——**gripper 状态变化**。

抓东西：gripper 从 open 变 closed，说明抓住了，该进入"移动到目标"阶段了。
放东西：gripper 从 closed 变 open，说明放了，该进入"下一步"或"撤退"了。

这俩瞬间就是"语义阶段切换"的物理信号，贼便宜也贼准。System 2 只在这些时刻被叫醒一次，其他时间 System 1 闷头执行就行。

这就像导航员不会每秒都喊"前面左转"，只在路口才说话。

---

## System 1 怎么用这些标记？

System 1 还是那个 VLA（QwenOFT 架构），输入比标准 VLA 多了一张图——**画了 visual prompt 的那张 overhead 图**。

它干的事变了：从"理解意图"变成"追着标记走"。这简单多了。

但这里有个**隐患**：万一 model 把画上去的十字和框当成图像噪声给忽略了呢？毕竟 overlay 长得也不像自然物体。

---

## Grounding Loss：逼 model "说出来"

这是 paper 最聪明的地方之一。

训练的时候，除了让它预测 action，还逼它**显式预测 visual prompt 的坐标**。

具体说，把 image 切成 1000 个 bin（Qwen-3-VL 的设计），然后问 VLM："那个十字在哪个 bin？"、"那个框的四个角分别在哪个 bin？"

用 Cross-Entropy loss，逼它精确到 bin。**只 backprop 通过 VLM，不通过 action decoder**。

intuition 是：你不逼它说出来，它可能"看着"了但没真"看进去"。你逼它把位置报出来，它的 internal representation 就必须真的 attend 到那个十字上，不能糊弄。

这招效果非常明显：ablation 里去掉这个 loss，性能从 53.8 掉到 49.4，掉了 4 个点。

---

## 为什么只在 key frame 做 grounding？

一开始我以为每帧都做会更准，结果 ablation 打脸了：**每帧都做反而更差**（49.5 vs 53.8）。

后来想明白了：action 大部分时候是连续的、平滑的，不需要每帧都"报告位置"。你逼它每帧都报，它就得每帧都"分心"去算坐标，训练信号反而 noisy，训练不稳定。

只在**关键帧**（任务开始 + gripper 状态变化那帧）报一次，相当于只在大决策点让 model "看清"，其他时候让它专心 execute。这跟人类做事挺像——你抓东西那一瞬间会盯紧，移动过程中其实注意力是放松的。

---

## 为什么 visual prompt 要单独一张图？

Ablation 里有个 variant：直接把十字和框画在原始 RGB 上。结果 50.8，比单独一张图（53.8）差 3 个点。

为啥？因为**混在一起，原始 visual feature 被污染了**。那个十字几像素大，但 model 要同时理解"这是啥物体"（需要 raw pixel）和"标记在哪"（需要 overlay），两个任务对 attention 的需求冲突。分开两张图，各管各的，干净。

---

## 实验里最打动我的几个数字

**Robocasa（GR1 humanoid）**：整体 +5 个点，但在"拿起东西放进柜子并关门"这种多步任务上 +10.6 个点。说明拆分任务这个设计在 long-horizon 上特别值钱。

**SimplerEnv（WidowX）**：+8.3 个点，尤其"把茄子放黄篮子"从 70.8 蹿到 95.8。这任务需要精确识别物体+精确放位置，正好是 visual prompt 的甜点。

**Real-world 垃圾分类**：
- 训练集内：87.5% vs baseline 80%
- 训练集外：85% vs baseline 63.3%
- **退化幅度只有 2.5 个点，baseline 退了 16.7 个点**

最后这条特别重要。它说明 visual prompt 让 model 学到的是"类别级 grounding"——知道瓶子该放绿箱、香蕉该放红箱——而不是死记训练时哪个物体长啥样。换個没见过的蓝色杯子、红鞋子、打乱的魔方，照样分对。

Baseline 的失败模式很有意思：红鞋子它放不对（依赖颜色 heuristic），打乱的魔方它抓不住（pattern overfitting），海绵它不知道该放哪箱（没真懂类别）。这些都是"没 grounding"的经典症状。

---

## 这让我联想到什么？

### 1. 跟你在 Tesla 推的 explicit spatial reasoning 很像

visual prompt 本质就是**让 model 显式地"看见"要操作哪里**，而不是隐式从语言推。这思路我觉得是 VLA 往前走的关键之一——geometry 不能全交给 implicit representation，得给它一个显式的锚点。

### 2. 跟人类大脑的 dual-stream theory 对应

人类视觉有两条通路：
- **Ventral stream（"what"通路）**：识别物体是啥，走颞叶，类似 System 2 的语义理解
- **Dorsal stream（"where/how"通路）**：定位物体在哪、怎么抓，走顶叶，类似 System 1 的 visuomotor

VP-VLA 的拆分某种程度 mirror 这个结构：VLM 负责语义（what），controller 负责空间执行（where/how），visual prompt 是连接两 stream 的 shared representation。

### 3. 跟 Chain-of-Thought 的关系

LLM 里 CoT 让 model 显式输出中间推理 token。VLA 里没法输出"推理"，但可以输出"视觉标记"作为中间状态。visual prompt 就是 VLA 版的 CoT——把隐式推理显式化、externalize 到一个 model 自己能看见的 format。

### 4. 跟 RL 里的 hierarchical decomposition

Options framework、MAXQ 这些 hierarchical RL 老思路，本质也是"高层规划 + 低层执行"。但 RL 那套需要学 option boundary，很 pain。VP-VLA 用 gripper 状态当 boundary，**绕过了学习**，直接用一个物理信号定义 phase 边界。Cheap 但 effective。

### 5. 跟自动驾驶的 trajectory prediction 类比

你在 Tesla 那会儿，planning 模块也是分层的：high-level intent（变道、转弯）+ low-level trajectory。中间用"路径点"或"目标车道"作为 interface。VP-VLA 是把这个 idea 搬到 manipulation，用 crosshair/box 替代 path point。

---

## 这 paper 的局限我也能想到

1. **SAM3 是 single point of failure**。SAM3 没分割对，整个 pipeline 就崩了。Paper 没量化 segmentation failure 的影响，这是个隐患。

2. **Gripper trigger 在非抓取任务上失效**。push、poke、open drawer 这种非 prehensile 操作，gripper 状态不变，trigger 就瞎了。$\phi$ 得重新设计。

3. **只 overlay overhead camera**。wrist camera 的 prompt 怎么传？多视角一致性？Paper 没讲。

4. **第一次推理延迟**。Qwen3-VL-4B + SAM3 一起跑，延迟可能上百 ms。对高频控制（比如 50Hz）是个问题。

5. **Visual prompt 是 static 的**。抓起来之后物体跟着 gripper 走，但 crosshair 还停在原图位置。要么用 tracking 更新，要么让 System 1 学会"标记只是初始参考"。

---

## 我的整体判断

这篇 paper 做的事不复杂，但**做对了**。它的价值不在 novelty 爆棚，在于**把一个对的 insight 执行得很扎实**：

- Dual-system 这个 idea Kahneman 提了几十年了
- Visual prompting 在 VLM 里早就有
- Hierarchical VLA 也一堆人做过

但它把这三者**用 grounding loss 这个关键 glue** 黏在一起，然后在 strong baseline（π₀.₅、GR00T-N1.6）上证明能涨点，real-world OOD 泛化也真的 work。这种"insight 正确 + execution 扎实 + 实验有说服力"的工作，是这个领域现在最需要的。

我个人觉得这个方向有潜力——**intermediate representation design** 可能是 VLA 下一波突破的关键。纯靠 scale up model 和 data，边际收益在递减；但加一个对的结构化 interface，能撬动很大的 gain。

参考链接：
- 项目主页: https://visualprompt-vla.github.io/
- starVLA codebase: https://github.com/starVLA/starVLA
- Qwen3-VL: https://arxiv.org/abs/2505.09388
- SAM3: https://arxiv.org/abs/2511.16719
- Kahneman《Thinking, Fast and Slow》: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

你觉得这个 dual-system 在 humanoid 全身操作上能 scale 吗？我挺好奇你对 visual prompt 这种 interface 在更复杂 embodiment 上的直觉。

---

# VP-VLA: Visual Prompting 作为 VLA 模型的高层推理与底层控制的接口

非常棒的一篇 paper，Andrej。这篇工作触及了当前 VLA (Vision-Language-Action) 领域一个非常核心的痛点——**monolithic bottleneck**，也就是让一个单一 network 同时承担 instruction parsing、spatial reasoning 和 motor execution 三件事。让我深入讲讲，试图 build 起这个方法的 intuition。

---

## 1. 核心动机:为什么需要 decouple?

### 1.1 现有 VLA 的问题

现有的 VLA 模型比如 OpenVLA、RT-2、π₀ 基本都是 **monolithic architecture**:

$$a_t = \pi_\theta(l, o_t)$$

其中 $l$ 是 language instruction，$o_t = \{o_t^1, o_t^2, \ldots, o_t^m\}$ 是来自多个 camera (overhead 或 wrist-mounted) 的 visual observation，$a_t = \{a_t^1, \ldots, a_t^n\}$ 是 action chunk (n 个连续动作以补偿 inference delay)。

这种端到端的方式有几个 observed 问题:

1. **Gibberish language 不影响性能** ([10, 48] 的发现): 把有意义的 instruction 替换成乱码，性能几乎不变——这说明 model 其实没有真正 grounding 语言，而是在 overfit 训练分布。

2. **OOD 退化严重**: 遇到 novel object category 或 unseen spatial position 就 fail (Fig. 2 的红框失败案例)。

3. **Spatial precision 差**: 单一 forward pass 需要同时做 instruction interpretation、spatial grounding 和 low-level control，每一项都是 hard problem，couple 在一起必然 trade-off。

### 1.2 Dual-System 的灵感

这个想法直接借鉴自 Kahneman 的 *Thinking, Fast and Slow* [14]:

- **System 2 (慢思考)**: deliberative reasoning，slow，computationally expensive，负责 high-level planning
- **System 1 (快直觉)**: fast, automatic，负责 sensorimotor execution

在 VLA 里这对应:
- System 2 = VLM planner (Qwen3-VL-4B)
- System 1 = 高频 controller (VLA policy)

关键 insight 是: **visual prompt 作为两者之间的 structured interface**，把抽象的语言变成具体的 spatial anchors (crosshair + bounding box)，这样 System 1 不用再 "interpret intent"，只需要做 "visuomotor tracking"。

---

## 2. 整体架构解析

参考 Fig. 3 (项目主页: https://visualprompt-vla.github.io/)，整体 pipeline:

```
Language Instruction l
        ↓
   [System 2 Planner] ← Event-triggered (gripper state change)
        ↓
   Subtask decomposition + target object/location names
        ↓
   [SAM3 Segmentation] → masks
        ↓
   Visual Prompt ψ_t = {Crosshair C, Bounding Box B}
        ↓
   Overlay on overhead camera → I_vp^t
        ↓
   [System 1 Controller π_θ] ← (o_t, l, I_vp^t)
        ↓
   Action chunk a_t
```

让我重点拆解每一个 module。

---

## 3. System 2 Planner $P_{S2}$ 细节

### 3.1 Event-Driven Task Decomposition

一个非常 elegant 的设计是 **event-driven execution loop**。之前的 hierarchical VLA 比如 Hi-Robot [33] 或 DexGraspVLA [46] 往往在每个 timestep 都调用 high-level planner，computationally wasteful。

VP-VLA 的核心假设: **manipulation tasks 由离散 semantic phases 组成 (grasp、place 等)，phase 之间的 transition 由 physical event 标记**。

形式化定义 transition event:

$$E_t = \mathbb{1}(|\phi(S_t) - \phi(S_{t-1})| > \epsilon)$$

变量解释:
- $E_t \in \{0, 1\}$: 是否在时刻 $t$ 触发了 transition event
- $S_t$: 时刻 $t$ 的 robot physical interaction state
- $\phi(\cdot)$: state-mapping function (这里 instantiate 为 gripper status)
- $\epsilon$: 变化阈值
- $\mathbb{1}(\cdot)$: indicator function

在 tabletop manipulation 中，$\phi$ 就取 gripper status (open/closed)。Gripper 从 open → closed (抓取成功) 或 closed → open (放置完成) 触发 re-evaluation。这非常物理直觉: **抓取完成的瞬间就是从 "approach object" 进入 "move to destination" 的时刻**。

### 3.2 Visual Prompt Generation

触发后，VLM planner 处理 $(l, o_t)$ 输出当前 subtask 和 entities:

$$s_k, e_{obj}, e_{loc} = \text{VLM planner}(l, o_t, S_t)$$

- $s_k$: 当前 subtask (第 k 个)
- $e_{obj}$: target object name (e.g., "wine")
- $e_{loc}$: target location name (e.g., "cabinet")，可能为 null

然后 segmentation model $\mathcal{G}$ (这里是 SAM3 [7]) 把 names 映射到 visual prompts:

$$\psi_t = \mathcal{G}(o_t, e_{obj}, e_{loc})$$

其中:
- $\psi_t = \{C, B\}$
- $C \in \mathbb{R}^2$: crosshair 中心 (target object 的 centroid)
- $B \in \mathbb{R}^4$: bounding box $[x_1, y_1, x_2, y_2]$ (placement region)

这里的 design choice 很关键:
- **对于 manipulation primitive (e.g., "pick")**: 生成 crosshair $C$ 在 object centroid，作为 interaction anchor。这把 policy 的 search space 从整张图缩小到一个 localized region。
- **对于 placement primitive**: 生成 bounding box $B$ 定义 spatial constraint。

最后 overlay 在 overhead camera 上得到 $I_{vp}^t$。注意一个 ablation 发现 (Table 6): **separate visual prompt 而不是 direct overlay 在 RGB 上** 效果更好 (50.8% vs 53.8%)，避免和 raw visual features 干扰。

---

## 4. System 1 Controller $\pi_\theta$ 细节

### 4.1 Architecture

$$a_t = \pi_\theta(l, o_t, I_{vp}^t) = h_\psi(f_\omega(l, o_t, I_{vp}^t))$$

- $f_\omega$: VLM backbone (Qwen3-VL-4B-Instruct) 处理 multi-modal inputs 到 high-level embeddings
- $h_\psi$: action decoder (MLP 或 diffusion)
- $\theta = \{\omega, \psi\}$

这里的 codebase 基于 starVLA [9] (https://github.com/starVLA/starVLA)，架构采用 QwenOFT，即把 OpenVLA-OFT [17] 里的 Prismatic VLM [15] 替换成 Qwen3-VL-4B。

### 4.2 Training Objective (这是 paper 的核心 contribution 之一)

**关键挑战**: visual prompting 的风险是 model 把 overlay 当成 image noise 而忽略掉。Paper 引入了 auxiliary visual grounding objective 来 force model 内化这些 spatial coordinates。

形式化:
$$\mathcal{L}_{total} = \mathcal{L}_{action}(\theta) + \lambda \cdot \mathbb{1}_{event} \cdot \mathcal{L}_{grounding}(\omega)$$

变量解释:
- $\mathcal{L}_{action}$: action prediction loss (用 L1 loss)
- $\mathcal{L}_{grounding}$: visual grounding loss (用 Cross-Entropy)
- $\lambda = 0.1$: 平衡系数
- $\mathbb{1}_{event}$: 只在 key frames (first frame 和 $E_t = 1$ 的 frame) 上应用 grounding
- $\omega$: VLM 参数 (注意 grounding loss **只 backprop 通过 VLM**)

**Grounding task 形式化**:
- 遵循 Qwen-3-VL 设计，把 image 维度分成 $N = 1000$ 个 uniform bins
- 对于 crosshair 中心 $(x, y)$: query VLM 预测 discretized 2D location
- 对于 bounding box: query VLM 预测 $[x_1, y_1, x_2, y_2]$ 4 个 discretized values
- Output 用 structured JSON 格式
- Loss 用 CE 而非 MSE，paper 说 CE 提供 "sharper and more structured training signal"

**Intuition**: 这个 grounding loss 等于让 VLM 明确 "说出" crosshair/box 的位置，强迫它的 internal representation 真的 attend 到这些 overlay，而不是 spurious correlation 到别处。

### 4.3 为什么只在 key frames?

Ablation (Table 6) 显示:
- w/o grounding: 49.4%
- w/ all frame grounding: 49.5% (比 key-frame only 差)
- Full (key-frame only): 53.8%

Dense grounding 引入 redundant/noisy constraints导致 training unstable。Selective grounding 在 supervision strength 和 training stability 之间取得平衡。

---

## 5. 实验数据深度分析

### 5.1 Robocasa-GR1-Tabletop Benchmark

数据集: 24 个 tabletop kitchen tasks，24,000 videos，来自 PhysicalAIRobotics-GR00T-X-Embodiment-Sim [2]。Humanoid robot GR1。每个 task 50 trials。

| Method | Average |
|--------|---------|
| Isaac-GR00T N1.5 | 48.2 |
| Isaac-GR00T N1.6 | 47.6 |
| QwenGR00T | 47.8 |
| QwenPI | 43.9 |
| QwenOFT | 48.8 |
| QwenFAST | 39.0 |
| **VP-VLA (Ours)** | **53.8 (+5.0)** |

特别值得注意的 gains:
- **"PnP * to * Close" (multi-step)**: 54.3 vs QwenOFT 43.7 (+10.6)。这验证了 System 2 的 task decomposition 在 long-horizon 任务上的价值。
- **"PnP Novel From Placemat To Plate"**: 70.0 vs 52.0 (+18.0)。OOD generalization 大幅提升。
- **"PnP Novel From Tray To Plate"**: 66.0 vs 56.0 (+10.0)。

### 5.2 SimplerEnv Benchmark

WidowX robot，4 个 tasks，BridgeDataV2 + Fractal 数据集 fine-tune 70k steps。

| Method | Average |
|--------|---------|
| OpenVLA-OFT | 41.8 |
| CogACT [21] | 51.3 |
| VideoVLA | 53.1 |
| π₀ | 53.1 |
| π₀.₅ [12] | 57.1 |
| Isaac-GR00T-N1.6-Bridge | 57.1 |
| QwenOFT + Qwen3VL | 50.0 |
| **Ours + Qwen3VL** | **58.3 (+8.3)** |

最显著的 gain: **"Put Eggplant in Yellow Basket": 95.8 vs 70.8 (+25.0)**。这个任务需要 precise object identification + target location grounding，正是 visual prompt 最能发挥的场景。

### 5.3 Real-World 实验 (非常 informative)

三个 task suites:

**Task 1: Robotic Waste-Sorting Categorization** (Table 3)

10k training steps，50 trajectories per training object。结果:

| Setting | VP-VLA | QwenOFT | Gap |
|---------|--------|---------|-----|
| In-Domain | 87.5% | 80.0% | +7.5 |
| OOD | 85.0% | 63.3% | +21.7 |
| **Generalization gap** | **2.5%** | **16.7%** | - |

这里非常关键: VP-VLA 的 ID-OOD gap 只有 2.5%，而 baseline 有 16.7% 的退化。这说明 visual prompt 让 model 学到的是 **category-level grounding** 而非 instance memorization。

特别看几个失败案例:
- Red shoe (OOD): QwenOFT 7/10 (依赖 color heuristic)，VP-VLA 9/10
- Scrambled Rubik's cube (OOD，同 semantic 不同 visual): QwenOFT 3/10 (pattern overfitting)，VP-VLA 9/10
- Sponge (OOD): QwenOFT 5/10 (不知道放哪个 box)，VP-VLA 8/10

**Task 2: Object Reference by Attribute** (Table 4)

"Pick up the <color> egg"，4×4 grid，200 demos (50 per color)。

| Setting | VP-VLA | QwenOFT |
|---------|--------|---------|
| In-Domain | 77.1% | 58.3% |
| OOD Color (purple/green) | 75.0% | 29.2% |
| OOD Position | 75.0% | 54.2% |

OOD Color 的 gap 巨大 (75 vs 29.2)。Baseline 经常 confuse novel color 与 visually similar training color，或默认抓离 gripper 最近的 egg (spatial proximity bias)。VP-VLA 把 linguistic attribute 和 visual instance 真正 disentangle 了。

**Task 3: Egg Carton Placement** (Table 5)

"Pick up egg and place at line X, column Y"。Partial credit: 1.0 for target, 0.5 for adjacent, 0.25 for diagonal。

| Setting | VP-VLA | QwenOFT |
|---------|--------|---------|
| In-Domain | 91.3% | 70.6% |
| OOD (novel row-col combos) | 68.8% | 55.0% |

Baseline 在 L3C3 只有 1/5 (vs VP-VLA 4.5/5)，因为 vertical + horizontal 联合 reasoning 困难。VP-VLA 因为有 bounding box 作为 spatial constraint，几何 grounding 更可靠。

---

## 6. Ablation Studies 详解

Table 6 的完整 ablation (Robocasa):

| Variant | Avg | 分析 |
|---------|-----|------|
| w/o grounding | 49.4 (-4.4) | 无 grounding loss，overlay 可能被当 noise |
| w/ all frame grounding | 49.5 (-4.3) | Dense supervision 反而 noisy |
| w/ point (instead of crosshair) | 47.3 (-6.5) | Point 提供弱 spatial extent info |
| w/ direct overlay | 50.8 (-3.0) | 与 raw RGB 干扰 |
| **Full** | **53.8** | 最佳 |

还有 Table 8 的 decomposition ablation (SimplerEnv):

| Method | Average |
|--------|---------|
| w/o decomposition (同时渲染 crosshair + box) | 57.3 |
| + decomposition | 58.3 |

特别在 "Put Eggplant in Yellow Basket" 上: w/o decomp 79.2 vs w/ decomp 95.8。Concurrent prompts 引入 visual noise，confuse policy 的 attention。

---

## 7. Related Work 的位置

在 VLA landscape 里，VP-VLA 处于一个非常有趣的 position:

### 7.1 与 end-to-end VLA 的对比

OpenVLA [18], RT-2 [4], π₀ [3], GR00T-N1.6 [2] 都是 monolithic。它们的优势是 simple，劣势是 spatial precision 和 OOD 弱。

### 7.2 与 reasoning-decomposed VLA 的对比

两条 prior path:
1. **Training-free pipelines** (SayCan [1], AffordGrasp [34]): GPT + traditional control。Problem: grounding 精度低。
2. **End-to-end affordance prediction** (COA-VLA [20], Hamster [23]): 让 VLA 直接 predict bounding box/trajectory。Problem: 难训练，predicted affordance 不一定 executable。

VP-VLA 的差异化: **separate subtask reasoning from action execution**，用 pretrained VLM 做 instruction decomposition + SAM3 生成 visual overlay。保留了 VLA 原生的 visual understanding，同时提供 precise spatial guidance。

### 7.3 与其他 visual prompting 方法的对比

- **TraceVLA [45]**: 用 visual trace (轨迹标记) 增强 spatial-temporal awareness
- **CoT-VLA [44]**: visual chain-of-thought，用 goal images 作为 interface
- **FlowVLA [47]**: visual chain of thought + motion reasoning
- **DreamVLA [43]**: dense geometric supervision

VP-VLA 的独特之处在于 **structured visual prompts** (crosshair + box) 而非 dense affordance 或 goal image，并且有 grounding objective 强制对齐。

### 7.4 与 cognitive science 的连接

Dual-system 思想在 robotics 中的应用越来越广泛。Related ideas:
- ECoT (Embodied Chain-of-Thought) [42]: 显式 reasoning steps
- Hi-Robot [33]: hierarchical VLA with open-ended instruction following
- BayesianVLA [24]: latent action queries 做 Bayesian decomposition

---

## 8. 一些更深的 intuition 和思考

### 8.1 为什么 visual prompt 比 text prompt 好?

一个深层的思考: **language 是 high-dimensional symbolic space，而 action 是 low-dimensional continuous space**。让一个 network 直接 map language → action，中间的 "shape mismatch" 巨大。

Visual prompt 把 language **先 project 回 visual space** (crosshair 在 image 哪个位置)，然后再 map visual → action。这条路径 shape 更 aligned，因为 action 本质上就是 visual motor mapping (类似人类大脑的 dorsal stream "where/how" pathway)。

这有点像 Chain-of-Thought 在 LLM 里让 model 显式输出中间推理，VLA 里则让 model 显式输出中间 visual grounding。

### 8.2 Event-driven 的精妙

$\phi$ 取 gripper status 是一个非常 cheap 但 semantic-rich 的 choice。Gripper 状态变化是 manipulation primitive 边界的 near-perfect physical proxy:
- Open → Closed: 抓住了，进入 transport phase
- Closed → Open: 放下了，进入 retreat phase

这意味着 System 2 不需要经常被调用，只在 semantic phase 转换时。这极大节省了 compute (Qwen3-VL-4B 的 inference 比小 action MLP 慢得多)。

未来 work 可能扩展 $\phi$ 到更丰富的 state，比如 force-torque sensor、contact detection、visual change detection 等。

### 8.3 Grounding loss 的本质

$\mathcal{L}_{grounding}$ 用 CE 而非 MSE 是个 interesting choice。Paper 说 CE 提供 "sharper" signal。我的理解是:
- MSE: continuous regression，model 可以预测 "大致在那个区域"
- CE: classification over 1000 bins，逼 model 精确到 bin

这相当于一种 **discrete spatial attention**——让 VLM 在 token space 里"说出"位置，本质上和 visual question answering 的 grounding 任务一致。

而且 grounding loss 只 backprop 通过 $\omega$ (VLM) 而非 $\psi$ (action decoder)，这是为了让 VLM 的 representation 对齐 visual prompt，而 action decoder 自由学习如何利用这些 aligned representations。

### 8.4 与人类 cognitive 的类比

人类做 manipulation 时:
- **Parietal cortex** (顶叶): 处理 spatial attention、grasp planning (类似 System 2)
- **Motor cortex** + **Cerebellum**: 执行 smooth motor commands (类似 System 1)
- **Visual cortex**: 提供 shared representation

VP-VLA 的架构某种程度上 mirror 这个: VLM 在 "thinking" 阶段确定 where，action model 在 "acting" 阶段执行 how，visual prompt 是 connecting interface。

### 8.5 Limitations 和 potential issues

虽然 paper 没明确说，我能想到几个潜在问题:

1. **SAM3 依赖**: 如果 SAM3 segmentation 失败 (e.g., occluded object, ambiguous text reference)，整个 pipeline 崩。Paper 没量化 segmentation failure rate 的影响。

2. **Gripper status assumption**: 在 non-prehensile manipulation (push、poke) 或 bimanual 任务里，gripper status 不是好的 phase indicator。$\phi$ 需要重新设计。

3. **Single camera overlay**: 目前只在 overhead camera overlay，wrist camera 的 visual prompt 怎么处理? 多视角一致性?

4. **Latency**: 虽然是 event-driven，但 SAM3 + Qwen3-VL 一起跑，首次规划延迟可能比较大。Real-time control 是否受影响?

5. **Language 仍然需要**: 即使有 visual prompt，$l$ 仍然作为 input。这说明 model 还没完全摆脱 language 依赖。未来能否完全去掉 $l$，只用 visual prompt?

---

## 9. 一些值得探索的方向

基于 VP-VLA 的启发，几个可能的 future direction:

### 9.1 动态 visual prompt

当前 visual prompt 是 static 的 (一个 crosshair 或一个 box)，但 task 执行中 target 可能在移动 (e.g., 抓住后 object 跟着 gripper 移动)。能否让 visual prompt **随着 execution 动态更新**? 比如用 tracking 而非 segmentation。

### 9.2 Multi-modal visual prompts

除了 crosshair 和 box，还可以:
- **Arrow**: 指示 push/pull 方向
- **Heatmap**: dense affordance
- **Trajectory trace**: 类似 TraceVLA
- **3D points**: 在 depth image 上

### 9.3 Self-supervised grounding

当前 grounding label 来自 SAM3 的 segmentation。能否让 model **自己 generate visual prompt**，再用 consistency loss 自监督? 类似 iterative refinement。

### 9.4 与 Diffusion Policy 的结合

Action decoder $h_\psi$ 当前是 MLP。如果换成 diffusion policy [27 RDT-1B]，能否处理更复杂 multi-modal action distribution?

### 9.5 Hierarchical events

当前 $\phi$ 只看 gripper。能否学习一个 hierarchical event detector:
- Low-level: gripper state
- Mid-level: contact change
- High-level: scene state change (object moved to target)

---

## 10. 实现细节的参考价值

### 10.1 Training hyper-parameters

- 8 GPUs
- AdamW optimizer
- VLM learning rate: 1e-5
- Action model learning rate: 1e-4
- $\lambda = 0.1$ (grounding loss weight)
- SAM3 detection threshold: 0.5, mask threshold: 0.5
- 只保留最高 score 的 visual prompt (target object & location 各一个)

### 10.2 Data preparation pipeline

这是非常实用的部分:

1. Rule-based decomposition original task → subtask list
2. At key frames, VLM predicts current subtask + target object/location names
3. 用 text-conditioned segmentation (SAM3) 在所有 frames 上得到 masks 和 boxes (直到下一个 key frame)
4. Convert to visual prompts: crosshair at mask centroid + bounding box for placement region
5. 每个 processed episode 存: per-frame masks, boxes, VLM subtask records
6. **Failures 直接丢弃** (避免 noisy supervision)

这个 pipeline 的好处是 **可 scale**: 一旦 rule-based decomposition 和 VLM subtask prediction 确定，SAM3 的 text-conditioned segmentation 可以 batch 处理大量 frames。

---

## 11. 总结

VP-VLA 是一个 **结构清晰、动机明确、实验充分** 的工作。它的核心贡献可以总结为:

1. **Concept**: 把 VLA 的 monolithic bottleneck 通过 dual-system + visual prompt interface 解耦
2. **Method**: Event-driven System 2 planner + structured visual prompts (crosshair + box) + key-frame grounding objective
3. **Empirical**: 在 Robocasa (+5.0)、SimplerEnv (+8.3)、real-world (ID 87.5%, OOD 85%) 都显著优于 strong baselines，特别是 OOD 场景

这篇 paper 给我最大的 intuition 是: **intermediate representation matters**。在 VLA 里，visual prompt 作为一个 explicit 的 spatial bridge，把 abstract language 转化为 actionable geometry，比直接 language → action 的 "黑箱" mapping 更 sample efficient、更 robust、更 interpretable。

某种程度上，这呼应了你在 Tesla 时推动的 "explicit spatial reasoning" 思路——让 model 显式地"看"到要操作哪里，而不是隐式地从语言推断。

---

## References

- VP-VLA 项目主页: https://visualprompt-vla.github.io/
- starVLA codebase: https://github.com/starVLA/starVLA
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- SAM 3: https://arxiv.org/abs/2511.16719
- GR00T N1: https://arxiv.org/abs/2503.14734
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691
- Hi-Robot: https://arxiv.org/abs/2502.19417
- ECoT: https://arxiv.org/abs/2407.08693
- TraceVLA: https://arxiv.org/abs/2412.10345
- CoT-VLA: https://arxiv.org/abs/2501.04693 (实际是另一篇，需核实)
- COA-VLA: https://arxiv.org/abs/2412.20451
- CogACT: https://arxiv.org/abs/2411.19650
- RoboCasa: https://arxiv.org/abs/2406.02523
- SimplerEnv: https://arxiv.org/abs/2405.05941
- Kahneman, *Thinking, Fast and Slow*: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

如果你对某个具体 module (比如 grounding loss 的实现细节、SAM3 的 text-conditioned segmentation、event-driven 触发逻辑) 想深入探讨，我们可以继续聊。我很想听听你对这个 dual-system 设计在自动驾驶或 humanoid 上 potential 的看法。
