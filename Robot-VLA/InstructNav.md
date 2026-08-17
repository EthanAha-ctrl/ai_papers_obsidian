---
source_pdf: InstructNav.pdf
paper_sha256: 491e7fef88ae44cd15a8c5561aaa33476c2b95a678603941af71f976f188e356
processed_at: '2026-08-05T10:00:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# InstructNav 用人话讲

## 这篇 paper 在搞啥事

说白了就是想造一个 robot，你说一句话它就能在新环境里走到你想要的地方，关键它 **完全不用训练，也不需要预先建好地图**。

听起来好像没啥稀奇，但你看看现状就知道为啥这值得发 paper 了：

- 你说 "找把椅子" → 这是 ObjNav 任务，得有 efficient exploration 能力
- 你说 "出门左转走过床去厨房" → 这是 VLN 任务，得 step-by-step follow 指令
- 你说 "我渴了" → 这是 DDN 任务，得 commonsense reasoning 知道"渴了要找水喝"

以前做 navigation 的 paper 基本都是钉死在某一类上：做 ObjNav 的人用 ObjNav 数据训练，做 VLN 的人用 R2R 数据训练，做 DDN 的人用 DDN 数据训练。每个领域自己玩自己的，因为 navigation 数据本身就很稀缺，联合训练效果不好。所以你会看到一堆 specialized model，但没人能做一个 generic system 吃下所有指令类型。

InstructNav 的 claim 就是：**老子一个 system 三个任务全干，而且 zero-shot**，在 R2R-CE 上第一次实现 zero-shot navigation，HM3D ObjNav 上比 SOTA 高 10.48%，DDN 上比 SOTA 高 86.34%。听起来就很猛。

---

## 它怎么做到的——核心 idea

你就把它想成一个 robot 的大脑分成三层：

**第一层：前额叶规划 (DCoN)**

每一步都用 GPT-4 重新规划。它把任何指令 decompose 成一个 chain：

```
Action_1 - Landmark_1 → Action_2 - Landmark_2 → ...
```

比如 "我渴了" 会被 GPT-4 翻译成 `Approach - bottle of water`，因为 GPT-4 的 commonsense 知道渴了要找水。比如 "走到 arched wooden doors" 会自动 align 成 `Approach - Doorway`，因为 segmentation 模型吐出来的 label 是 "Doorway"。

关键点：**每走一步都重新规划**，因为环境是 partial observation 的，你刚看到的 object 会影响下一步该干啥。这个 dynamic 是重点，static plan 在 unknown 环境下肯定会挂。

参考 [Chain-of-Thought](https://arxiv.org/abs/2201.11903) 的思路，让 LLM 输出 JSON 格式：
```json
{"Reason": "...", "Action": "Approach", "Landmark": "bottle of water", "Flag": "False"}
```

`Flag=True` 就 stop。这跟 [ReAct](https://arxiv.org/abs/2210.03629) 的 reasoning-action interleaving 思路一致。

**第二层：基底核 + 视觉皮层 (Multi-sourced Value Maps)**

光有规划不行，GPT-4 给你一句 `Approach the bottle of water` 你怎么把它变成 robot 的具体动作？这中间缺一个"语言到空间"的桥梁。这就是 value maps 要干的活。

它造了四张 value map，每张都是 3D 空间里每个 navigable point 的 value（0 到 1 之间）：

1. **Semantic Value Map $m_s$**: 距离 target landmark 越近 value 越高。靠 GLEE segmentation 模型识别 RGB-D 里的物体，投射到 3D 点云，然后算 navigable area 上每个点到 landmark 的距离，反向 normalize。公式核心是：

$$d_{sem} = \min_{q \in PCD_{obj}} \| p - q \|, \quad \forall p \in PCD_{nav}$$

变量含义：$p$ 是 navigable 区域的一个 candidate 位置，$q$ 是 landmark 物体的 3D 点云中的某点，$\| p - q \|$ 是 3D 欧氏距离。$\min$ 就是对所有 landmark 点取最小。

然后：
$$C_{sem} = 1 - \frac{d_{sem} - \min(d_{sem})}{\max(d_{sem}) - \min(d_{sem})}$$

$\min(d_{sem})$ 和 $\max(d_{sem})$ 是在所有 navigable $p$ 上的极值，归一化到 [0,1] 再 reverse 一下，离 landmark 近的点 value 高。

2. **Action Value Map $m_a$**: 根据 DCoN 给的 action 类型在空间特定区域赋值 1。比如 `Move Forward` 就给当前 robot 朝前那个 quadrant 赋 1，`Explore` 就给已探索区域的 frontier boundary 赋 1。这是为了让 robot 知道"该往哪个大方向走"。

3. **Trajectory Value Map $m_t$**: 离历史轨迹越远 value 越高。这是 "反 visited bonus"，避免 robot 在屋里来回打转。

$$d_{traj} = \min_{h \in PCD_{traj}} \| p - h \|$$

注意原 paper 公式 3 写的是 $\min_{q \in PCD_{traj}}$ 但变量解释用的是 $h$，这是笔误，应该是 $h$ 代表 history trajectory 里的一个点。

正向 normalize：
$$C_{traj} = \frac{d_{traj} - \min(d_{traj})}{\max(d_{traj}) - \min(d_{traj})}$$

距离历史轨迹远 value 高，鼓励探索新区域。

4. **Intuition Value Map $m_i$**: 这是最有意思的一张。用 GPT-4V 看 6 张 panorama 图（从 12 个方向等间隔采样），让它输出 chain-of-thought reasoning + 下一步该往哪个方向走。然后那个方向的 FOV 区域赋 value 1。

$$\langle CoT_i, Dir_i \rangle = MLM(P_i; I, A_i, L_i)$$

变量含义：$MLM$ 是 multimodal large model 函数，$P_i$ 是当前 panorama 图，$I$ 是原始 instruction，$A_i, L_i$ 是 DCoN 当前 step 给的 action 和 landmark。输出 $\langle CoT_i, Dir_i \rangle$ 分别是推理过程和方向 ID。

为啥要这张 map？因为前面三张 map 有处理不了的情况：
- Semantic map 没法区分"前面那张桌子"和"后面那张桌子"（relational reasoning）
- Action map 没法表达"在桌子和沙发之间走"（between 这种关系）

GPT-4V 直接看图就能 handle 这种 multimodal spatial reasoning，把它的判断投影到空间上就是 Intuition Value Map。

**第三层：执行**

四张 map 直接相加：

$$m = m_s + m_a + m_t + m_i$$

障碍物区域置零，然后 argmax 找 value 最高的点作为 waypoint：

$$(x_i, y_i, z_i) = \arg\max_{(x,y,z) \in m} P(x,y,z)$$

最后用 A* 在这个 value map 上规划路径。Simulator 里用 rotate-then-forward 这种离散动作执行，真实 robot 上直接控速度。

**Stop 信号**: DCoN 输出 `Flag=True` 或者 GPT-4V 说 "Stop" 就停。

---

## 整个 pipeline 的 intuition

你就这么理解：**GPT-4 用 commonsense 把"我渴了"翻译成"找水"，然后 GPT-4V 看图判断"水大概在左边"，然后 value maps 把"找水"和"在左边"这两个信息一起 spatial 投影成一张 3D 空间上的 heatmap，最后 A* 在 heatmap 上找峰值当 waypoint 走过去。每走一步重新来一遍这个循环。**

很优雅的地方在于 modular 设计——每个模块职责清晰，可以 ablation 检查每个模块的贡献。Linear combination 的 value map 也是 soft voting，简单但 work。

---

## 实验结果——到底有多 work

### HM3D ObjNav (Table 1)

| Method | Training Free | SR | SPL |
|---|---|---|---|
| SemExp | ✗ | 37.9 | 18.8 |
| OVRL (trained) | ✗ | 62.0 | 26.8 |
| L3MVN | ✓ | 50.4 | 23.1 |
| VLMF | ✓ | 52.5 | 30.4 |
| **InstructNav** | ✓ | **58.0** | 20.9 |

InstructNav SR 58，超过所有 zero-shot baseline。比 SOTA VLMF 高 $\frac{58-52.5}{52.5} = 10.48\%$（相对提升，paper abstract 写的就是这个数）。

但注意 **SPL 反而比 VLMF 低**（20.9 vs 30.4），说明 InstructNav 走的路径更长。这是 DCoN 每步重规划的代价——路径不优。这是这个方法的一个内在 trade-off：动态灵活但路径 jitter。

### R2R-CE VLN (Table 2)

| Method | Training Free | SR | SPL |
|---|---|---|---|
| Seq2Seq | ✗ | 25 | 22 |
| CMA | ✗ | 32 | 30 |
| NaVid | ✗ | 37 | 36 |
| **InstructNav** | ✓ | 31 | 24 |

**这是 paper 最硬核的 claim**：首次在 R2R-CE 上实现 zero-shot，SR 31 超过一堆 task-trained 模型（Seq2Seq、CWP-CMA、Ego2Map-NaViT 等）。

但要注意一个 caveat：**论文排除了使用 MP3D 训练的 waypoint predictor 的方法**（ETPNav、BEVBert 等）。这些方法本身在 R2R-CE 上更强，排除了它们相当于在一个相对弱的 baseline 集合上比较。论文虽然声明了 "for fairness"，但这个 fairness 选择对 InstructNav 的 claim 是有利的。

比 NaVid 低 6 个点，但 NaVid 是用 video VLM 训练过的，InstructNav 完全 training-free，这个对比已经够 striking 了。

### DDN Demand-Driven (Table 3)

| Method | SR | SPL |
|---|---|---|
| DDN (trained) | 16.1 | 8.4 |
| ChatGPT-Prompt | 0.3 | 0.01 |
| MiniGPT-4 | 2.9 | 2.0 |
| **InstructNav** | **30.0** | **14.2** |

DDN 是 InstructNav 最猛的 benchmark。相对 DDN trained SOTA 提升 $\frac{30-16.1}{16.1} = 86.34\%$，几乎翻倍。

为啥 DDN 提升这么大？因为 DDN 任务的本质就是 commonsense reasoning——"我渴了" → "要找水喝" → "找 bottle"。这恰好是 LLM 最擅长的事。DCoN 直接把 LLM 的 commonsense 能力释放出来了。

---

## Ablation 说了啥

### N 的选择 (Figure 4)

GPT-4V 的 panorama 输入用几张图？N=4 (前后左右)、N=6 (12 方向间隔采样)、N=12 (全 12 方向)。

N=6 最优。**Intuition**: 太少丢信息，太多 MLM 理解负担重。这跟 [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) 关于 image token 数与 reasoning 性能 trade-off 的发现一致。

### DCoN 和四张 map 的 ablation (Table 4)

| Ablation | ObjNav SR | VLN SR | DDN SR |
|---|---|---|---|
| Full | 56 | 30 | 33 |
| w/o DCoN | **44** (-12) | **23** (-7) | **22** (-11) |
| w/o Semantic VM | 44 (-12) | 21 (-9) | 25 (-8) |
| w/o Action VM | 51 | 28 | 28 |
| w/o Trajectory VM | 52 | **19** (-11) | 21 |
| w/o Intuition VM | 54 | **17** (-13) | 20 |

几个关键 takeaway：

1. **DCoN 最 critical**，去掉后三个 task 都崩。这说明 LLM commonsense 是这个 system 的灵魂
2. **Intuition VM 对 VLN 影响最大**（-13）。VLN 指令大量包含 "walk between", "the front table" 这种 relational spatial reasoning，前 3 张 map 处理不了，必须靠 GPT-4V 看图
3. **Trajectory VM 对 ObjNav 影响小但对 VLN/DDN 大**。ObjNav 找到就停，不容易打转；VLN/DDN 是 long-horizon 任务，没有 trajectory penalty 就会在屋里反复跑
4. **Semantic VM 对所有任务都 critical**，它直接决定目标定位精度

### 开源 LLM 替代 (Table 5)

| LLM + MLM | ObjNav SR | VLN SR | DDN SR |
|---|---|---|---|
| GPT-4 + GPT-4V | 56 | 30 | 33 |
| Llama3 70B + GPT-4V | 50 | 23 | 21 |
| GPT-4 + LLaVA1.6 34B | 50 | 17 | 28 |
| Llama3 70B + LLaVA1.6 34B | 50 | 12 | 18 |

**Intuition**: ObjNav 上开源模型跟 GPT 差距不大（6 个点），因为 ObjNav 任务简单（找 object category），LLM 只要做简单的 "explore + approach" planning。VLN 上差距巨大（12 vs 30），因为 VLN 需要 fine-grained spatial reasoning，LLaVA1.6 在 panorama 多图输入下表现远不如 GPT-4V。

这暗示：**framework 本身 robust，能吸收 weaker LLM 的输出，但 LLM reasoning 能力是 VLN bottleneck**。所以 future work 里换更强的 open-source MLM（比如 [LLaVA-NeXT 72B](https://llava-vl.github.io/blog/2024-01-30-llava-next/) 或者 InternVL 这种）可能会缩小这个 gap。

---

## Real robot 实验

硬件很朴素：Turtlebot 4 + ORBBEC RGB-D + RPLIDAR + ThinkPad + Raspberry Pi。推理放远程 RTX 4090 工作站，WiFi 通信。

软件栈用 [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) 做自定位，[Navigation2](https://github.com/ros-planning/navigation2) 做 point-to-point 导航。InstructNav 只负责 high-level waypoint 决策，low-level 执行交给 ROS2 标准导航栈。这种 modular 设计对 deployment 友好。

测试场景覆盖 apartment、office、library、gallery、teaching building 五种室内环境。

**Intuition**: 这个部署架构是教科书式的 modular robotics——感知 + 规划 + 执行解耦，每层可独立替换。InstructNav 真正的贡献在 "规划" 这一层，其它都用现成组件。

---

## 我看完之后的思考

**Conceptual advance**:
- DCoN 的 dynamic re-infer 是真贡献。以前 [NavGPT](https://arxiv.org/abs/2305.86916)、[SayNav](https://arxiv.org/abs/2309.07294) 这类纯 LLM planner 都是 static plan 一次，partial observation 下容易挂。DCoN 让 LLM 变成 online reasoner，跟 [ReAct](https://arxiv.org/abs/2210.03629) 思路一致
- Multi-sourced Value Maps 的 modular 设计很 clean。每个 map 职责清晰，可以独立 ablation，可以替换模型（GPT-4V → LLaVA），可以扩展（加新的 map 类型）
- 第一次把 ObjNav + VLN + DDN 统一到一个 zero-shot framework，这个 unification 本身就是 contribution

**可以质疑的点**:

1. **Value map 简单相加 $m = m_s + m_a + m_t + m_i$**，权重都是 1。为啥不是 $m = \alpha_s m_s + \alpha_a m_a + ...$？learnable weights 或者 contextual weights 可能解决 SPL 偏低的问题（trajectory 长）
2. **GPT-4 调用频率太高**: 每个 decision step 调 DCoN + GPT-4V 两次 API，500 step episode 可能上千次 call，latency 和 cost 都是问题
3. **R2R-CE 上排除 waypoint predictor 方法**: 这个 comparison 不完全公平。ETPNav、BEVBert 这些方法本身在 R2R-CE 上 SR 能到 50+，论文排除了它们，claim "超过一堆 task-trained 方法" 的强度被削弱
4. **Semantic Value Map 单帧 segmentation**: 没 temporal aggregation，occlusion 问题没解决。作者在 limitation 里承认了，提到未来用 [amodal segmentation](https://arxiv.org/abs/2312.12484)
5. **Stop judgment 不一致**: 有时是 DCoN Flag=True，有时是 GPT-4V 说 Stop，两个 stop signal 怎么协调没讲清楚
6. **没有 trajectory smoothness 学习**: SPL 低说明路径不优，DCoN 每步重规划可能导致 jitter

**与相关工作脉络**:

InstructNav 站在一个 interesting 的位置上：
- 继承了 [ESC](https://arxiv.org/abs/2304.04607) (commonsense exploration)、[L3MVN](https://arxiv.org/abs/2304.05501) (LLM target prior)、[VLMF](https://arxiv.org/abs/2310.07868) (VLM frontier) 这些"用 LLM/VLM 做 navigation prior"的思路
- 融合了 [VLMaps](https://arxiv.org/abs/2211.04298) (semantic map) 和 [NavGPT](https://arxiv.org/abs/2305.86916) (LLM planner) 两条线
- 跟 [NaVid](https://arxiv.org/abs/2402.15852) (end-to-end video VLM) 形成 modular vs end-to-end 路线对比。NaVid 走 end-to-end video VLM 训练，InstructNav 走 modular zero-shot。两条路都值得 follow，最后可能 merge

---

## 适合 follow-up 的方向

1. **Value map 权重学习**: 把 4 张 map 的权重做成 learnable 或者 contextual（根据 instruction type 动态调权重）
2. **LLM cache**: DCoN 的 reasoning 可以 cache，相邻 step 观察变化不大时复用上次 plan，减少 API call
3. **Temporal semantic map**: 多帧 segmentation aggregation + amodal completion 解决 occlusion
4. **End-to-end distillation**: 把 InstructNav 的 GPT-4 + GPT-4V 蒸馏成一个小 model，解决 latency 和 cost 问题
5. **更细粒度 action**: 现在 action space 是 "Explore / Approach / Move Forward / Turn" 这种 coarse 的，可以扩展成 "Go upstairs / Open door / Push object" 这种 interaction action
6. **加入 3D scene graph**: 像 [SayPlan](https://openreview.net/forum?id=wMpOMO0Ss7a) 那样建 3D scene graph，让 LLM 在 scene graph 上 reason 而不是只在 language 上 reason
7. **External memory**: DCoN 每步重规划但没显式 memory module。可以加 episodic memory（之前看到啥）+ semantic memory（room layout 常识）
8. **Benchmark on harder instructions**: 现在测试的 ObjNav/VLN/DDN 都是单一类型，可以造一个 mixed instruction benchmark，比如 "我渴了，去厨房找点喝的，回来时把灯关了"

---

## Reference 链接汇总

核心:
- [项目主页](https://sites.google.com/view/instructnav)
- [R2R-CE Benchmark](https://arxiv.org/abs/2007.00114)
- [HM3D Dataset](https://arxiv.org/abs/2109.08238)
- [DDN Paper](https://arxiv.org/abs/2305.01886)
- [Habitat Challenge 2023](https://aihabitat.org/challenge/2023/)

技术组件:
- [GLEE Segmentation](https://arxiv.org/abs/2312.09158)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [ReAct (reasoning-action interleaving)](https://arxiv.org/abs/2210.03629)
- [GPT-4V System Card](https://arxiv.org/abs/2309.17421)
- [Llama3](https://github.com/meta-llama/llama3)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

相关工作:
- [ESC (commonsense exploration)](https://arxiv.org/abs/2304.04607)
- [L3MVN (LLM target prior)](https://arxiv.org/abs/2304.05501)
- [VLMF (VLM frontier)](https://arxiv.org/abs/2310.07868)
- [VLFM (vision-language frontier maps)](https://arxiv.org/abs/2310.03242)
- [VLMaps (semantic map)](https://arxiv.org/abs/2211.04298)
- [NavGPT (LLM planner)](https://arxiv.org/abs/2305.86916)
- [SayNav (LLM dynamic planning)](https://arxiv.org/abs/2309.07294)
- [SayPlan (3D scene graph)](https://openreview.net/forum?id=wMpOMO0Ss7a)
- [NaVid (video VLM VLN)](https://arxiv.org/abs/2402.15852)
- [ETPNav (waypoint predictor)](https://arxiv.org/abs/2304.03047)
- [Amodal Completion in the Wild](https://arxiv.org/abs/2312.12484)

Deployment:
- [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox)
- [Navigation2](https://github.com/ros-planning/navigation2)
- [AI2-THOR](https://arxiv.org/abs/1712.05474)
- [ProcTHOR](https://arxiv.org/abs/2206.06994)

---

## TL;DR

InstructNav 干的事就是：**用 GPT-4 的 commonsense 当 online planner 把指令拆成 "action-landmark" 链，用 GPT-4V 的 visual reasoning 当 spatial projector 把"该往哪走"翻译成空间 heatmap，再用四张 value map 做 soft voting 决定 waypoint，最后 A* 执行。每走一步重新来一遍。**

它牛在第一次用一个 zero-shot framework 同时吃下 ObjNav + VLN + DDN 三类任务，而且结果都 competitive。弱点是路径不优（SPL 低）、API 调用多、依赖 GPT-4 这种闭源大模型。但作为一个 modular zero-shot system，这个 conceptual framework 是值得 follow 的，未来加 learnable weights、LLM cache、temporal aggregation、end-to-end distillation 应该能继续 push。

---

# InstructNav 深度解析：Generic Instruction Navigation 的 Zero-shot 系统

## 1. 论文核心问题与Motivation

这篇 paper 来自 Peking University + Southeast University + Oxford，由 Yuxing Long 和 Wenzhe Cai 等人完成。项目主页：https://sites.google.com/view/instructnav

它要解决的核心矛盾非常清晰：**instruction-guided navigation 领域被人为分割成三类子任务，每类用各自的训练数据训练专门模型，但真实世界中机器人应该能 follow 任何形式的自然语言指令。**

- **Object Goal Navigation (ObjNav)**: 给一个 object category（如 "chair"），在 unseen environment 中找它。代表工作 [SemExp](https://arxiv.org/abs/2006.13171), [L3MVN](https://arxiv.org/abs/2304.05501), [ESC](https://arxiv.org/abs/2304.04607)。
- **Vision-Language Navigation (VLN)**: 给 step-by-step 指令（如 "Walk out of the bathroom, turn left, walk past the bed..."）。代表 benchmark [R2R-CE](https://arxiv.org/abs/2007.00114), models [CMA](https://arxiv.org/abs/2007.00114), [ETPNav](https://arxiv.org/abs/2304.03047), [NaVid](https://arxiv.org/abs/2402.15852)。
- **Demand-Driven Navigation (DDN)**: 给抽象人类需求（如 "I am thirsty"），通过 commonsense reasoning 找到能满足 demand 的物体。代表工作 [DDN](https://arxiv.org/abs/2305.01886)。

三类任务的 navigation strategy 差异巨大：ObjNav 重 efficient exploration，VLN 重 step-by-step following，DDN 重 commonsense reasoning。数据稀缺使得联合训练困难，以往工作都被钉死在某一类上。InstructNav 提出**一个统一 zero-shot 框架同时处理三类**，这在领域内是首次。

---

## 2. 整体架构 Build Intuition

把整个 pipeline 想象成大脑 + 小脑 + 感官的分工：

```
┌─────────────────────────────────────────────────────────────┐
│   Instruction I (e.g., "I am thirsty, find something to drink")│
└──────────────────────────────┬──────────────────────────────┘
                               ▼
            ┌───────────────────────────────────────────┐
            │   Dynamic Chain-of-Navigation (DCoN)       │  ← "前额叶规划"
            │   LLM (GPT-4) re-infers every step         │
            │   Output: {Action_i, Landmark_i, Flag}    │
            └──────────────────┬────────────────────────┘
                               ▼
   RGB-D, Pose, Semantic Seg (GLEE) → 3D Point Cloud
                               ▼
   ┌──────────────── Multi-sourced Value Maps ────────────────┐
   │                                                            │
   │  m_s Semantic VM   m_a Action VM                           │
   │  m_t Trajectory VM  m_i Intuition VM (GPT-4V)              │  ← "基底核 + 视觉皮层"
   │                                                            │
   └─────────────────────┬─────────────────────────────────────┘
                         ▼
          m = m_s + m_a + m_t + m_i  (合成)
                         ▼
          argmax over m → waypoint (x_i, y_i, z_i)
                         ▼
          A* path planning → low-level action
                         ▼
          Stop if DCoN Flag=True or GPT-4V says "Stop"
```

核心 insight：**用 LLM 的 commonsense 当"规划器"，用 value map 当"几何投影器"，把语言指令连续地翻译成 spatial 上的 high-value 区域**。这正是 [SayNav](https://arxiv.org/abs/2309.07294)、[NavGPT](https://arxiv.org/abs/2305.86916)、[VLMF](https://arxiv.org/abs/2310.07868) 等工作的延伸，但加入了"动态更新 + 多 value map 协同"两件事。

---

## 3. Dynamic Chain-of-Navigation (DCoN) 细节

### 3.1 Schema 设计

定义一个 navigation schema：
$$\text{CoN} = \text{Action}_1 - \text{Landmark}_1 \to \text{Action}_2 - \text{Landmark}_2 \to \cdots$$

每个 node 由 (Action, Landmark) 对组成，整个 chain 模拟人类 plan-then-execute 的过程。这与 LLM 的 [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903) 天然 align，所以可以直接让 GPT-4 输出。

### 3.2 Prompt 工程

Prompt 包含四部分：
- **Robot Definition**: 定义 candidate navigation actions，如 `Explore`, `Approach`, `Move Forward`, `Turn Left/Right`, `Enter`, `Exit`
- **Navigation Strategy**: 不同任务类型的 strategy description，让 LLM 结合 house layout commonsense 推理
- **Prediction Format**: `{'Reason':..., 'Action':..., 'Landmark':..., 'Flag':...}` JSON
- **Episode Information**: 当前 observation 已观测的 object 列表

### 3.3 三个核心问题的解决

**问题1: Semantic Label Misalignment**
- 指令说 "arched wooden doors"，但 segmentation 模型输出 "Doorway"
- DCoN 在 prompt 中通过 `<Requirements for Landmarks>` 优先选择 observed objects，并在 reasoning 中显式对齐语义

**问题2: Unseen Target → Exploration**
- 目标 landmark 没观测到时，DCoN 输出 `Action: Explore`, `Landmark: TV`，意思是"去有 TV 的区域找 sofa"（sofa 通常和 TV 共现）

**问题3: Abstract Demand → Commonsense Reasoning**
- "I am thirsty" → DCoN 输出 `Action: Approach`, `Landmark: bottle of water`，因为 LLM 内化的 commonsense 知道 thirsty → drink → bottle

### 3.4 "Dynamic" 的关键含义

每个 decision step 重新 infer DCoN（注意这里 cost 较高，每个 episode 调 GPT-4 可能几十次），根据新观测的 objects 更新下一步 action 和 landmark。这避免了 static plan 在 partial observation 下的失败，类似 [ReAct](https://arxiv.org/abs/2210.03629) 的 reasoning-action interleaving。

---

## 4. Multi-sourced Value Maps 公式级解析

### 4.1 Semantic Value Map m_s

输入：DCoN landmark $L_i$，从 RGB-D + segmentation 得到 scene semantic point cloud $PCD_{obj}$（landmark 对应的 3D 点），navigable area $PCD_{nav}$（无 obstacle 的地面）。

对每个 navigable position $p \in PCD_{nav}$，计算到 landmark 点云 $q \in PCD_{obj}$ 的最小欧氏距离：

$$d_{sem} = \min_{q \in PCD_{obj}} \| p - q \|, \quad \forall p \in PCD_{nav} \tag{1}$$

变量含义：
- $p = (p_x, p_y, p_z)$: navigable 区域中的 query point
- $q = (q_x, q_y, q_z)$: landmark 物体点云中的某点
- $\| p - q \|$: 3D 欧氏距离

然后做 min-max normalization（注意是 reversed，距离近 → value 高）：

$$C_{sem} = 1 - \frac{d_{sem} - \min(d_{sem})}{\max(d_{sem}) - \min(d_{sem})} \tag{2}$$

变量含义：
- $d_{sem}$: 当前点 p 到最近 landmark 的距离
- $\min(d_{sem}), \max(d_{sem})$: 在所有 $p \in PCD_{nav}$ 上的最小/最大 $d_{sem}$
- $C_{sem} \in [0, 1]$: 越接近 landmark，value 越高

**Intuition**: 把空间中"靠近目标物体"的程度编码成一张连续 heatmap，把 LLM 的 "Approach the sofa" 翻译成 spatial gradient。

### 4.2 Action Value Map m_a

不同的 action 类型对空间赋予不同的 value pattern：

| Action | Value Pattern |
|---|---|
| Move Forward / Turn around / Turn right / Turn left | 1.0 赋给当前 robot 位置的全景 FOV 四个 quadrant 中对应的方向 sector |
| Explore | 1.0 赋给 explored environment 的 frontier boundary（类似 [frontier exploration](https://arxiv.org/abs/1906.05705)） |
| Enter / Exit | 替换成 "Approach" + "Doorway" landmark（decompose 成更 primitive 的操作） |
| Approach | 不在 m_a 上赋值，完全由 m_s 实现 |

这个设计是为了在 simulator 离散动作空间和连续几何规划之间搭桥。

### 4.3 Trajectory Value Map m_t

记录 robot 历史轨迹 $PCD_{traj}$，对 navigable area 中每个 $p$ 计算到历史点 $h \in PCD_{traj}$ 的最小距离：

$$d_{traj} = \min_{h \in PCD_{traj}} \| p - h \|, \quad \forall p \in PCD_{nav} \tag{3}$$

注意这里下标不一致（公式3用了 q 但下面写 h，应为笔误，原意是 $h$）。然后正向 normalize（距离远 → value 高）：

$$C_{traj} = \frac{d_{traj} - \min(d_{traj})}{\max(d_{traj}) - \min(d_{traj})} \tag{4}$$

变量含义：
- $h = (h_x, h_y, h_z)$: 历史轨迹中某个采样点
- $d_{traj}$: p 到最近历史点距离
- $C_{traj} \in [0,1]$: 远离已走过的区域 value 高

**Intuition**: 等价于一个"反 visited bonus"，避免 robot 在 partial observation 下反复来回探索同一区域。与 [Active Neural SLAM](https://arxiv.org/abs/2001.07606) 的 exploration bonus 类似但更软。

### 4.4 Intuition Value Map m_i

这是 InstructNav 的另一关键创新，用 Multimodal Large Model (MLM, 这里用 GPT-4V) 做 visual reasoning。

输入：
- Visual: N=6 张从 12 个方向中等间隔采样的 RGB 视图，拼成 panorama $P_i$，每张标注 Direction ID（1~12）
- Textual: 完整 instruction $I$, 当前 DCoN 的 $A_i$ 和 $L_i$

输出：MLM 先输出 chain-of-thought $CoT_i$，然后输出下一步方向 $Dir_i$。

$$\langle CoT_i, Dir_i \rangle = MLM(P_i; \, I, A_i, L_i) \tag{5}$$

变量含义：
- $MLM(\cdot)$: multimodal large model 推理函数
- $P_i$: 当前位置 panorama 图
- $I$: 原始 navigation instruction
- $A_i, L_i$: DCoN 给的当前 action 和 landmark
- $CoT_i$: MLM 的 chain-of-thought 文本
- $Dir_i$: MLM 推荐的下一步方向 ID

$Dir_i$ 对应方向的 FOV 区域在 Intuition Value Map 上赋值 1.0。如果该区域无 navigable position，则把 failure feedback 回传给 MLM 让它 re-predict。

**Intuition**: $m_s$ 处理不了 "the front dining table"（多张桌子中选前面那张）这种 relational spatial reasoning；$m_a$ 处理不了 "walk between"。$m_i$ 把 MLM 的 multimodal reasoning 能力直接 spatial 投影，弥补这两类 failure mode。

### 4.5 决策合成

最终决策 value map 是四个 value map 直接相加：

$$m = m_s + m_a + m_t + m_i \tag{6}$$

然后 obstacle 区域置零，取 argmax 得到目标 waypoint：

$$(x_i, y_i, z_i) = \arg\max_{(x,y,z) \in m} P(x,y,z) \tag{7}$$

用 A* 算法在 m 上规划路径，simulator 下用 rotate-then-forward 离散执行；真实世界直接控制速度。Stop 条件：DCoN Flag=True 或 GPT-4V 输出 "Stop"。

**Intuition**: 这是一种 "ensemble of value maps" 设计，每个 map 编码一种 navigation prior，相加等价于 soft AND/OR 组合。这种设计可解释性强，方便 ablation。

---

## 5. 实验结果深入分析

### 5.1 HM3D ObjNav (Table 1)

| Method | Training Free | SR | SPL |
|---|---|---|---|
| SemExp | ✗ | 37.9 | 18.8 |
| OVRL (trained) | ✗ | 62.0 | 26.8 |
| L3MVN (zero-shot) | ✓ | 50.4 | 23.1 |
| VLMF | ✓ | 52.5 | 30.4 |
| **InstructNav** | ✓ | **58.0** | 20.9 |

观察：SR 上超过所有 zero-shot baselines，比 SOTA (VLMF) 高 5.5 个点（论文 abstract 写 10.48% 是相对改进比例 $\frac{58.0-52.5}{52.5} \approx 10.48\%$）。但 SPL 反而比 VLMF 低，说明 InstructNav 的 trajectory 更长（可能因为 DCoN 每步重规划导致路径不优）。Navigation Error 2.58 表明即使失败也接近目标。

### 5.2 R2R-CE VLN (Table 2)

| Method | Training Free | SR | SPL |
|---|---|---|---|
| Seq2Seq | ✗ | 25 | 22 |
| CMA | ✗ | 32 | 30 |
| NaVid | ✗ | 37 | 36 |
| **InstructNav** | ✓ | 31 | 24 |

这是 paper 最 striking 的 claim：**首次在 R2R-CE 上实现 zero-shot navigation**，SR 31% 超过 Seq2Seq、CWP-CMA、Ego2Map-NaViT 等一众 task-trained 模型。但要注意：
- 论文特意排除了使用 MP3D 训练的 waypoint predictor 的方法（如 ETPNav, BEVBert），否则这些方法会更高
- 比 SOTA NaVid 低 6 个点，但 NaVid 是 video-based VLM 训练过的
- NE 6.89 是所有方法中最低的，说明 InstructNav 倾向于"走得很近但没到 stop point"

### 5.3 DDN Demand-Driven (Table 3)

| Method | SR | SPL |
|---|---|---|
| DDN (trained) | 16.1 | 8.4 |
| ChatGPT-Prompt | 0.3 | 0.01 |
| MiniGPT-4 | 2.9 | 2.0 |
| **InstructNav** | **30.0** | **14.2** |

DDN 是 InstructNav 提升最大的 benchmark：相对 DDN trained baseline 提升 $\frac{30.0-16.1}{16.1} = 86.34\%$。这说明 commonsense reasoning 在 DCoN 框架下被很好地释放了——abstract demand 通过 LLM 的 commonsense 自动 decompose 成具体 landmark search。

---

## 6. Ablation Study 关键发现

### 6.1 N 的选择 (Figure 4)

N=4, 6, 12 三种 panorama 配置。N=6 (12 方向间隔采样 6 张) 在三个 task 上都最好。**Intuition**: N 太小丢关键视觉信息；N 太大 MLM visual understanding burden 过重。这呼应 [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) 关于 image token 数与性能的 trade-off。

### 6.2 DCoN 和四个 Value Map 的 ablation (Table 4)

最 striking 的发现：

| Ablation | ObjNav SR | VLN SR | DDN SR |
|---|---|---|---|
| Full InstructNav | 56 | 30 | 33 |
| w/o DCoN | **44** (-12) | **23** (-7) | **22** (-11) |
| w/o Semantic VM | 44 (-12) | 21 (-9) | 25 (-8) |
| w/o Trajectory VM | 52 | **19** (-11) | 21 |
| w/o Intuition VM | 54 | **17** (-13) | 20 |

关键发现：
1. **DCoN 是最 critical 组件**，去掉后三个 task 都大幅降级
2. **Intuition VM 对 VLN 影响最大**（-13），因为 VLN 指令需要 multimodal relational reasoning（"walk between the table and sofa" 这种 m_s/m_a 处理不了的）
3. **Trajectory VM 对 ObjNav 影响小但对 DDN/VLN 大**，因为 ObjNav 是 find-once 任务，VLN/DDN 是 long-horizon 任务容易重复
4. **Semantic VM 对所有任务都 critical**，因为它直接决定目标定位

### 6.3 开源 LLM 替代 (Table 5)

| LLM + MLM | ObjNav SR | VLN SR | DDN SR |
|---|---|---|---|
| GPT-4 + GPT-4V | 56 | 30 | 33 |
| Llama3 70B + GPT-4V | 50 | 23 | 21 |
| GPT-4 + LLaVA1.6 34B | 50 | 17 | 28 |
| Llama3 70B + LLaVA1.6 34B | 50 | 12 | 18 |

**Intuition**: 在 ObjNav 上开源模型接近 GPT（差距 6 个点），但在需要复杂 reasoning 的 VLN 上差距很大（12 vs 30）。这说明 DCoN + Multi-sourced Value Maps 这个 framework 本身设计足够 robust，可以吸收 weaker LLM 的输出；但 LLM 的 reasoning 能力是 VLN 的瓶颈。开源 LLM 在 VLN 上的退化主要来自 LLaVA1.6 在 panorama 多图输入下空间理解不够好。

---

## 7. Real Robot 实验

硬件：Turtlebot 4 + ORBBEC Astra Pro Plus RGB-D camera + RPLIDAR-A1 + ThinkPad E14 + Raspberry Pi 4B，远程 RTX 4090 加速。

软件栈：[SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) 自定位，[Navigation2](https://github.com/ros-planning/navigation2) 做 point-to-point 导航与 dynamic obstacle avoidance。

测试场景：apartment、office、library、gallery、teaching building 五种代表性室内环境，每种都用 diverse instruction types。

**关键 takeaway**: simulator（discrete action space）到 real-world（continuous control）的迁移通过 Navigation2 框架实现，InstructNav 只负责 high-level waypoint decision。这种 modular 设计是 deployment 友好的。

---

## 8. Limitations 与未来方向

作者自己点出三个问题：

1. **依赖闭源 large model**: GPT-4/GPT-4V API 限制 real deployment，open-source 替代品性能 gap 大
2. **Semantic Value Map 受 occlusion 影响**: 当目标物体被遮挡时定位不准
3. **未来方向**: 设计 data generation pipeline 解决数据稀缺 + end-to-end generic navigation model + 用 amodal segmentation ([Amodal Completion in the Wild](https://arxiv.org/abs/2312.12484), [Tri-layer plugin](https://arxiv.org/abs/2210.13961)) 改善 occlusion 鲁棒性

---

## 9. 我的 Intuition 总结与 Critical Reflection

**亮点**:
1. DCoN 的"每步动态 re-infer" 是真正的 conceptual advance，把 LLM 从 static planner 升级为 online reasoner
2. Multi-sourced Value Maps 设计非常 modular，每个 map 有明确职责，方便 ablation 和扩展
3. 第一次让一个 system 同时吃下 ObjNav + VLN + DDN 三类任务，unification 价值大

**可质疑的点**:
1. **Value map 简单相加 m = m_s + m_a + m_t + m_i**：四个 map 权重都是 1，没有学习权重。为什么不是 $m = \alpha_s m_s + \alpha_a m_a + ...$？这可能解释 SPL 偏低（轨迹长）的问题
2. **GPT-4 调用频率高**: 每个 decision step 调一次 DCoN + 一次 GPT-4V（Intuition Map），500 step episode 可能上千次 API call，cost 和 latency 是 deployment 阻碍
3. **R2R-CE 上排除了 waypoint predictor 训练的方法**: 这个 comparison 不完全公平，论文虽然声明了但还是削弱 claim 强度
4. **Semantic Value Map 依赖 GLEE 单帧 segmentation**: 没 temporal aggregation，occlusion 问题只靠 future explore 缓解
5. **没有 trajectory smoothness 学习**: SPL 低暗示轨迹不优，DCoN 每步重规划可能产生 jitter

**与相关工作的脉络**:
- 继承自 [ESC](https://arxiv.org/abs/2304.04607) (commonsense exploration)、[VLMF](https://arxiv.org/abs/2310.07868) (VLM frontier)、[VLFM](https://arxiv.org/abs/2310.03242) (vision-language frontier maps)、[L3MVN](https://arxiv.org/abs/2304.05501) (LLM 目标 prior)
- 把 [NavGPT](https://arxiv.org/abs/2305.86916)、[SayNav](https://arxiv.org/abs/2309.07294) 这类纯 LLM planner 与 [VLMaps](https://arxiv.org/abs/2211.04298) 这类 semantic map 系系统融合
- 与最近 [NaVid](https://arxiv.org/abs/2402.15852) (video VLM VLN) 形成对比：NaVid 走 end-to-end video VLM 路线，InstructNav 走 modular zero-shot 路线

---

## 10. Reference Web Links

- **项目主页**: https://sites.google.com/view/instructnav
- **R2R-CE Benchmark**: https://arxiv.org/abs/2007.00114
- **HM3D Dataset**: https://arxiv.org/abs/2109.08238
- **Habitat Challenge**: https://aihabitat.org/challenge/2023/
- **DDN Paper**: https://arxiv.org/abs/2305.01886
- **GLEE Segmentation**: https://arxiv.org/abs/2312.09158
- **Chain-of-Thought**: https://arxiv.org/abs/2201.11903
- **GPT-4V System Card**: https://arxiv.org/abs/2309.17421
- **Llama3**: https://github.com/meta-llama/llama3
- **LLaVA-NeXT**: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- **VLMaps (related work)**: https://arxiv.org/abs/2211.04298
- **VLFM (related work)**: https://arxiv.org/abs/2310.03242
- **ESC (related work)**: https://arxiv.org/abs/2304.04607
- **L3MVN (related work)**: https://arxiv.org/abs/2304.05501
- **NaVid (related work)**: https://arxiv.org/abs/2402.15852
- **NavGPT (related work)**: https://arxiv.org/abs/2305.86916
- **ETPNav (related work)**: https://arxiv.org/abs/2304.03047
- **SLAM Toolbox**: https://github.com/SteveMacenski/slam_toolbox
- **Navigation2**: https://github.com/ros-planning/navigation2
- **AI2-THOR**: https://arxiv.org/abs/1712.05474
- **ProcTHOR**: https://arxiv.org/abs/2206.06994

---

## 11. 给你的 Intuition 速记

如果用一句话概括 InstructNav 的本质：**用 LLM commonsense 当 online planner + MLM visual reasoning 当 spatial projector + 多 value map 做 soft voting，把"任意自然语言指令"翻译成"空间中可微的 navigation heatmap"，再交给 A* 执行。** 它的核心 contribution 不是某一个模块，而是这套 **dynamic LLM planning → multi-value-map spatial projection → A* execution** 的 modular 设计在 zero-shot 设定下首次跨三个 navigation 子任务 work。

值得 follow-up 的方向是：value map 加权学习（替代简单相加）、LLM cache 减少 API call、temporal aggregation 的 semantic map 解决 occlusion、把整个 pipeline 蒸馏成 end-to-end model 解决 latency。
