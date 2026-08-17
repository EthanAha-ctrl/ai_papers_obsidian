---
source_pdf: Fast-SmartWay.pdf
paper_sha256: 6e8af6105c60eee8eb52993b210417301b68b2921dabf21ad41d8b7d88a84ad5
processed_at: '2026-08-04T07:12:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Fast-SmartWay

## 一句话概括

**让机器人别再像个傻子一样每走一步就转圈看四周了，直接看前方三个方向，让 GPT-4o 告诉它怎么走就行。**

---

## 这玩意儿到底在解决什么问题

先说背景。你给机器人一句指令，比如"走过客厅，进厨房，停在冰箱旁边"，它得自己导航过去。这就是 VLN (Vision-and-Language Navigation)。

以前的玩法有个大毛病：机器人每走一步，得原地转一圈拍 12 张照片（每 30 度一张），凑成一个 panorama，然后喂给模型让它决定下一步往哪走。

这有什么问题呢？

- **慢得要死**。Hello Robot 真机测试，光转圈拍照就花 22 秒一步
- **机器人很尴尬**。很多便宜机器人只有前面一个摄像头，你让它转 360 度它做不到
- **中间那个 waypoint predictor 纯属多余**。它先从 panorama 里挑几个"看起来能走"的点，再让 navigator 选。问题是 predictor 根本不懂你的指令，它只看图，挑出来的点可能跟你的目标八竿子打不着

所以这篇论文的核心 idea 很简单粗暴：**砍掉 panorama，砍掉 waypoint predictor，直接拿前面三张图 + 文字指令，让 GPT-4o 输出"转多少度、走多远"。**

---

## 但直接这么干会出什么问题

Frontal view 只能看到前方 60 度范围（左 30 度到右 30 度），机器人的视野一下子从 360 度缩到 60 度。这就像你戴着马眼罩在房子里找东西。

主要会出两类问题：

**问题一：走进死胡同不知道回头**

你跟它说"去厨房"，它前面看到一条走廊就钻进去了，结果是个储物间。它没有全景视野，不知道背后其实有另一条路通厨房。

**问题二：一步走错，步步走错**

单步决策容易 greedy，哪个方向看起来最像厨房就走哪个。但走了三步才发现走错了，这时候已经偏了。

所以论文搞了两个补救机制，合起来叫 **Uncertainty-Aware Reasoning**。

---

## 补救机制一：Disambiguation（"我迷路了让我重新看看"）

这个特别直觉。就是让 GPT-4o 每一步输出的时候，额外判断一个 boolean：`Confuse: true/false`。

什么时候标 true 呢？三种情况：
- 指令本身模糊
- 不知道目标在哪
- 眼前看到的东西跟指令预期对不上

一旦 confuse 了，机器人就触发"紧急回头"模式：原地转一圈，重新拍 12 张全景图，重新理解环境，重新规划方向。

这就像你迷路的时候会停下来，四周张望一下，重新定位。

关键细节：重新转圈的时候，不只看新的全景图，还会把之前的 **Trajectory Summary**（走过的路）和 **Instruction Progress**（指令完成到哪一步了）一起喂给 GPT-4o。这样它不是从零开始重新规划，是在已有上下文基础上修正。

---

## 补救机制二：FPBR（"前想想后想想"）

Future-Past Bidirectional Reasoning，名字听着玄乎，其实就是让 GPT-4o 在 prompt 里多做两件事。

**Future Prediction（往前想）**：选下一步动作之前，先在脑子里模拟一下。比如"如果我左转，我应该会看到一条通往厨房的走廊"。然后再跟实际看到的东西对比，如果对不上就重新考虑。

**Past Recall（往回想）**：回忆上一步选了哪个方向、当时是怎么推理的。如果上一步说"往前走应该能看到冰箱"，结果现在看到的是床，那就说明走错了，要改主意。

这个机制不需要训练，不需要额外的 memory module，纯粹是 prompt engineering。就是让 MLLM 自己在每一步的 reasoning 里做 self-consistency check。

---

## 一个很巧妙的技术细节：depth 怎么喂给 GPT-4o

GPT-4o 能看图，但 depth map 它读不懂。直接给一张灰度图它不知道那个像素值代表几米。

论文的做法是把 depth 转成文字：

1. 三张 depth 图分别投影成 partial point cloud
2. 只取每张图的下半部分（因为只关心地面附近的障碍物）
3. 对每个 angular column，找最近的障碍物距离
4. 把这 120 度视野切成 5 个方向 bin（左 60 度、左 30 度、正前、右 30 度、右 60 度）
5. 每个 bin 算个平均距离，然后跟两个阈值比：
   - 小于 0.5 米 → "very close obstacle"（别撞了）
   - 0.5 到 4 米 → "obstacle at X meters"（注意前方 X 米有障碍）
   - 大于 4 米 → "path is clear for X meters"（畅通）

最后 GPT-4o 拿到的输入长这样：

```
- If you turn left 60°, obstacle appears at 2.3 meters
- If you turn left 30°, path is clear for 5.0 meters
- If you go forward, path is clear for 3.8 meters
- If you turn right 30°, there is a very close obstacle
- If you turn right 60°, obstacle appears at 1.5 meters
```

加上 RGB 图像里用 RAM 模型识别出的物体标签：

```
- Left view: chair, table
- Front view: doorway, hallway
- Right view: wall
```

这样 GPT-4o 就有了"前方几何结构 + 语义内容"的完整文字描述，配合三张 RGB 图，做决策就靠谱多了。

**这个 geometric-to-linguistic 的转换是整篇论文最实用的 trick**。你不需要训练一个 sensor tokenizer，不需要让 MLLM 学读 depth，就是把数字翻译成人话，让 LLM 用它已经擅长的语言推理能力干活。

参考 RAM 模型: https://recognize-anything.github.io/

---

## 实验数据说了啥

### 真机（Hello Robot）

| 方法 | 视角 | 拍照时间 | 推理时间 | 总时间 | 成功率 | 导航误差 |
|---|---|---|---|---|---|---|
| RecBERT | 全景 | 22.4s | 0.02s | 22.4s | 20% | 3.92m |
| SmartWay | 全景 | 22.4s | 6.85s | 29.3s | 32% | 3.01m |
| 本文 | 前方三视角 | 5.13s | 7.26s | 12.4s | 36% | 2.78m |

重点：**总时间从 29 秒砍到 12 秒，成功率反而从 32% 涨到 36%**。

这说明全景那额外的 9 张图带来的信息，根本不值得花 22 秒去采集。前方三张图加上好的 reasoning 就够了。

### 仿真（R2R-CE）

跟同样用前方视角的 SmartWay 对比：
- SmartWay 前方视角：SR 7.25（基本废了）
- 本文：SR 27.75（起死回生）

这个对比最能说明问题。同样的输入（前方三视角），没有 Disambiguation 和 FPBR 的 SmartWay 直接崩了，加上这两个模块性能翻了快 4 倍。

跟全景视角的 SOTA 比，SR 几乎持平（27.75 vs 29），但 SPL 和 nDTW 更好，说明路径走得更高效更自然。

### Ablation

- 两个模块都去掉：SR 19.75
- 只加 Disambiguation：SR 24.25（+4.5）
- 两个都加：SR 27.75（再 +3.5）

Disambiguation 贡献更大，因为它直接救活了那些会卡死的 episode。FPBR 是锦上添花，让正常情况下的路径更连贯。

---

## 我的几个直觉判断

**第一，geometric-to-linguistic 是 MLLM 时代 embodied AI 的通用模式**。你不需要训练专门的 sensor encoder，把传感器数据翻译成文字让 LLM 推理就行。这个思路在 robotics manipulation、autonomous driving 里都在用，Fast-SmartWay 在 navigation 上验证了它 work。

**第二，panoramic view 可能被高估了**。VLN 领域一直觉得 360 度视野很重要，但这篇论文的数据说明，前方视野 + 好的 reasoning 完全能 compensate。真实机器人很少有全景相机，这个 finding 对 deployment 很有价值。

**第三，让模型自己判断"我 confused 了"比 hard-code recovery rule 好**。传统方法会设个阈值，比如连续 N 步没进展就触发 recovery。Fast-SmartWay 让 MLLM 自己判断，考虑了 instruction progress、visual cues、trajectory history 的综合信号，更灵活。

**第四，FPBR 这种 prompt-level temporal reasoning 是个 cheap 但有效的设计**。不需要训练 memory network，不需要 transformer 的 cross-attention，就是在 prompt 里加两句"想想未来、回忆过去"，就能显著提升 temporal consistency。这个 trick 可以迁移到很多 LLM-based agent 的场景。

**第五，最大的 limitation 还是依赖 GPT-4o API**。真要部署在机器人上，网络延迟、API 成本、privacy 都是问题。这篇论文证明了 MLLM-based navigation 的 upper bound，但真正落地还需要 open-source MLLM (Qwen-VL, LLaVA) 能达到类似性能。目前 Qwen2.5-VL-72B 在这类任务上已经接近 GPT-4o，这个方向很有希望。

参考 Qwen-VL: https://github.com/QwenLM/Qwen2.5-VL

---

## 最后吐槽一句

这篇论文的 Figure 1 里画的对比图特别清楚：上面是传统方法，一坨 waypoint candidate 气泡，下面是本文，直接三张图进去一个箭头出来动作。这个图比所有文字描述都直观。

但是论文标题说 "Panoramic-Free"，其实第一步还是要做一次 panoramic scan 来确定初始方向，以及 disambiguation 时也要转圈。所以严格说是 "Step-wise Panoramic-Free"，不是完全不用全景。这个标题稍微有点 overclaim，不过 step-wise 省掉全景已经是很大的工程价值了。

---

# Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot VLN-CE 深度解析

## 1. 论文核心 motivation 与领域坐标

VLN-CE (Vision-and-Language Navigation in Continuous Environments) 的演化路径很清晰:

- **VLN (离散)**: Anderson et al. 2018 引入, agent 在预定义 navigation graph 的节点间跳跃
- **VLN-CE (连续)**: Krantz et al. ECCV 2020 提出, 去掉 graph, agent 在 3D 空间连续运动, 但通常需要 waypoint predictor 生成 candidate
- **Zero-shot VLN-CE**: 用 LLM/MLLM 替代训练好的 navigator, 利用 foundation model 的 generalization

Fast-SmartWay 处于 zero-shot VLN-CE 这一支, 解决两个痛点:
1. **Panoramic observation 的硬件/时间成本**: 传统设置需要 12 个 RGB-D views (每 30° 一个), 真实机器人要么转 360° 慢, 要么需要昂贵的 panoramic camera
2. **Waypoint predictor 的语义脱节**: predictor 只用 RGB-D 训练, 不理解 instruction, 生成的 candidate 可能在视觉上 traversable 但语义上无关

参考链接:
- VLN-CE benchmark: https://jacobkrantz.github.io/vlnce/
- Habitat simulator: https://habitat.ai/
- R2R 原始数据集: https://bring2asrd.github.io/

## 2. Problem Formulation 的关键转变

### Classical setting
$$I_t = \{(I_i^{rgb}, I_i^{depth}) | i = 1, ..., 12\}$$

每个 step 都需要 12 个 views (0°, 30°, ..., 330°), 其中:
- $I_i^{rgb} \in \mathbb{R}^{H \times W \times 3}$: RGB 图像
- $I_i^{depth} \in \mathbb{R}^{H \times W}$: depth map

### Fast-SmartWay setting
- $t = 0$: 一次 panoramic scan $\rightarrow I_0$ (12 views), 用于全局 orientation
- $t > 0$: 仅 3 个 frontal views $\rightarrow I_t = \{(I_i^{rgb}, I_i^{depth}) | i = 1, 2, 3\}$, 对应 $\{330°, 0°, 30°\}$ 即 left/front/right

这个 hybrid 设计的 intuition: 初始时刻机器人不知道朝向, 需要一次全景理解; 一旦方向确定, 后续每一步只需"看前方"做局部决策, 大幅降低 perception latency。

## 3. End-to-End Pipeline 的技术细节

### 3.1 Spatial-Semantic Textual Description Generation

这是替代 waypoint predictor 的核心模块。整体流程: depth images $\rightarrow$ partial point cloud $\rightarrow$ 1D distance vectors $\rightarrow$ textual prompts。

**Step 1: Depth $\rightarrow$ 3D point cloud**
- 对 3 个 depth images $\{I_i^{depth} | i \in \{L, F, R\}\}$ 做 center crop (减少 peripheral distortion)
- 只保留 bottom half (关注地面附近障碍物, 因为机器人是 ground robot)
- 用 pinhole camera model 投影到 3D space
- 再用 robot pose + 每个视图的相对 yaw 转换到 world frame

**Step 2: Ground-plane distance 计算**

公式 (1):
$$D_j^{(i)} = \sqrt{x^2 + z^2}$$

变量含义:
- $i \in \{L, F, R\}$: view direction 索引
- $j \in \{1, ..., w\}$: image column 索引
- $(x, z)$: 3D 点在 robot local frame 的 2D 坐标 (忽略 $y$ 高度, 因为只关心 ground-plane 距离)

**Intuition**: $x$ 是横向, $z$ 是深度, 投影到地面就是欧氏距离, 这样无论障碍物多高都统一编码为"水平距离"。

公式 (2): 每列取最小值
$$D_j^{(i)} = \min_{k \in \{1, ..., h/2\}} \sqrt{x_{k,j}^2 + z_{k,j}^2}$$

变量:
- $k$: image row 索引 (只在 bottom half, 即 $h/2$ 行)
- $x_{k,j}, z_{k,j}$: 第 $k$ 行第 $j$ 列像素对应的 3D 点

**Intuition**: 对每列 (即每个 angular slice), 取最近的障碍物距离。这等于把 depth map 压成"前方各角度最近的障碍距离"曲线, 类似一个简化的 1D laser scan 表征。

**Step 3: Discretization 到 5 bins**

$\{D^{(L)}, D^{(F)}, D^{(R)}\}$ 拼接覆盖 -60° 到 +60°, 分成 5 个 bin: $\{-60°, -30°, 0°, +30°, +60°\}$。每个 bin 计算平均距离 $\bar{d}_b$。

**Step 4: Textual description 生成 (公式 3)**

$$\ell_b = \begin{cases} 
\text{"If you [direction], there is a very close obstacle"} & \text{if } \bar{d}_b < d_{close} \\
\text{"If you [direction], obstacle appears at } \bar{d}_b \text{ meters"} & \text{if } d_{close} \leq \bar{d}_b < d_{mid} \\
\text{"If you [direction], path is clear for moving forward in } \bar{d}_b \text{ meters"} & \text{otherwise}
\end{cases}$$

阈值: $d_{close} = 0.5m$, $d_{mid} = 4m$。

最终输出 $\mathcal{L}_{spatial}^{(5)} = \{\ell_1, ..., \ell_5\}$, 每个 $\ell_i$ 对应一个方向。

**Intuition**: 这本质上是把 depth 几何信息翻译成 MLLM 能"理解"的自然语言。MLLM 不擅长直接读 depth 数值, 但能很好地理解"前方 2 米有障碍"这种描述。这种"geometric-to-linguistic" 的转换是 zero-shot 方法的常见技巧, 让 foundation model 不需要额外的 sensor tokenizer。

### 3.2 Semantic 描述

用 RAM (Recognize Anything) 模型对 RGB 图像做 tagging:
- Frontal view: $\mathcal{O}^{(3)} = \{\mathcal{O}_L, \mathcal{O}_F, \mathcal{O}_R\}$
- Panoramic view: $\mathcal{O}^{(12)} = \{\mathcal{O}_1, ..., \mathcal{O}_{12}\}$

参考: https://recognize-anything.github.io/

### 3.3 Initial step prompt (panoramic)

输入:
1. Instruction $\mathcal{I}$
2. $\mathcal{O}^{(12)}$ + $\mathcal{L}_{spatial}^{(12)}$
3. Task description $\mathcal{T}$
4. 12 个 panoramic RGB images $I_{1...12}^{rgb}$

输出 (structured):
- Thought: 自然语言推理
- Selected Image (1-12): 初始朝向
- Safe Distance: 安全前进距离
- Trajectory Summary: 环境概览
- Instruction Progress: 把 instruction 拆 subgoals, 第一标 In Progress, 其余 Not Started

### 3.4 Step-wise prompt (frontal)

输入:
1. 3 个 RGB images (left/center/right)
2. Instruction $\mathcal{I}$, $\mathcal{O}^{(3)}$, $\mathcal{L}_{spatial}^{(5)}$, valid actions list
3. 上一 step 的 context: Observed objects $\mathcal{O}$, Previous Selected Image, Instruction Progress, Trajectory Summary, Previous Thought

输出 JSON:
- Thought, Selected Image (1/2/3), Action Options, Degree, Safe Distance, Confuse (bool), Updated History

避免卡死的技巧: 比较 $\mathcal{L}_{spatial}$ 与上一 step, 若 unchanged 则轻微右移。

## 4. Uncertainty-Aware Reasoning (核心创新)

### 4.1 Disambiguation Module

触发条件 (Confuse = true):
1. Instruction 模糊
2. Goal 不清晰
3. Visual cues 与 instruction progress 冲突

触发后行为:
- 360° 旋转, 收集 12 RGB + $\mathcal{O}^{(12)}$ + $\mathcal{L}_{spatial}^{(12)}$
- 同时注入 Trajectory Summary 和 Instruction Progress (历史 context)
- Prompt 指导 MLLM: (a) 识别已完成步骤, (b) 检测当前朝向与 instruction 意图错配, (c) 推荐 re-orientation 方向

**Intuition**: 这是一种"self-correction via re-perception"机制。Frontal view 容易陷入局部最优 (例如选了错走廊后无法回头), Disambiguation 让机器人主动"重新看一下四周", 类似人类迷路时的行为。

### 4.2 Future-Past Bidirectional Reasoning (FPBR)

灵感来自 NavBench (arXiv:2506.01031) 的 Local Observation ability。

**Future Prediction**: 让 MLLM 心理模拟候选 action 的视觉后果, 如 "If I turn left, I expect to see a hallway leading to a kitchen"。基于 current RGB + semantic + spatial + instruction landmarks 评估 plausibility。

**Past Recall**: 用 Previous Selected Image 和 Previous Thought 回忆上一步决策, 比较当前 observation 与 $t-1$ 时的预测。若 mismatch (如期望 "kitchen" 但看到 "bed"), 修正 policy。

**Intuition**: 这相当于给 MLLM 一个"内部 consistency check"。单步 reasoning 容易 greedy, FPBR 引入 temporal bidirectional context — 前向模拟防止 myopic 决策, 后向回忆防止 cascading errors。这与 chain-of-thought 在时间维度上的扩展类似。

## 5. 实验结果深度解读

### 5.1 Real-world (Hello Robot, Table I)

| Method | View | Perception | Inference | Total | SR↑ | NE↓ |
|---|---|---|---|---|---|---|
| RecBERT [7] | Panoramic | 22.40 | 0.02 | 22.42 | 20 | 3.92 |
| SmartWay [16] | Panoramic | 22.40 | 6.85 | 29.25 | 32 | 3.01 |
| Ours | Frontal | 5.13 | 7.26 | 12.39 | 36 | 2.78 |

关键观察:
- Perception time 从 22.40s 暴跌到 5.13s (约 23%), 这是 frontal view 直接节省的旋转时间
- Inference time 略升 (7.26 vs 6.85), 因为 MLLM 需要在有限 view 下做更多 reasoning
- 总 latency 是 SmartWay 的 42.4%, SR 反而更高 (36 vs 32)

**Intuition**: 这说明 panoramic 的额外信息并没有换来相应的性能提升, 反而 latency 成本巨大。Frontal view + 更好的 reasoning 能弥补信息缺失, 这是 efficient embodied AI 的重要 insight。

Hello Robot 参考: https://hello-robot.com/

### 5.2 Simulator (R2R-CE, Table II)

最关键的对比 (zero-shot frontal):
- SmartWay [16] frontal: SR $7.25 \pm 3.77$, SPL $6.32 \pm 3.16$
- Ours: SR $27.75 \pm 2.22$, SPL $24.95 \pm 2.70$, nDTW $51.83 \pm 1.54$

SR 从 7.25 跳到 27.75 (约 3.8 倍), 这是 ablation 的核心 evidence: 简单把 SmartWay 改成 frontal view 会崩, 但 Fast-SmartWay 的 reasoning module 让 frontal view 也能 work。

与 panoramic SOTA 比较:
- SmartWay panoramic: SR 29, SPL 22.08, nDTW 41.77
- Ours frontal: SR 27.75, SPL 24.95, nDTW 51.83

SR 几乎持平 (差 1.25), SPL 和 nDTW 反而更好, 说明 frontal view 方法虽然成功率略低, 但路径更高效、更"自然"。

### 5.3 Ablation (Table III)

| 配置 | SR | SPL | nDTW |
|---|---|---|---|
| w/o Disambiguation & FPBR | 19.75 | 17.42 | 50.83 |
| + Disambiguation only | 24.25 | 21.14 | 52.06 |
| Full (Disambiguation + FPBR) | 27.75 | 24.95 | 51.83 |

Disambiguation 单独贡献: SR +4.5, SPL +3.72 (主要救死锁/局部最优)
FPBR 叠加贡献: SR +3.5, SPL +3.81 (主要保证全局一致)

**Intuition**: 两个 module 解决不同问题 — Disambiguation 是"出问题时自救", FPBR 是"平时防止出错"。两者互补, 单独任一都不到位。

## 6. 与相关工作的关系网

| 方法 | View | Pipeline | 备注 |
|---|---|---|---|
| NavGPT [14] | Panoramic | Visual $\rightarrow$ text $\rightarrow$ GPT-4 | 早期 zero-shot, 文本桥接 |
| Open-Nav [15] | Panoramic | Waypoint predictor + LLM | 两阶段, 候选评 分 |
| SmartWay [16] | Panoramic | Waypoint + GPT-4o + backtrack | 多模态直接推理 |
| Navid [24] | Frontal (sequential) | Video-based VLM | ego-centric 早期工作 |
| Fast-SmartWay (本文) | Frontal (3 views) | End-to-end MLLM | 无 waypoint predictor |

关键方法学差异:
- SmartWay [16] 仍依赖 waypoint predictor, Fast-SmartWay 用 textual spatial description 替代
- Navid [24] 用 video 序列, Fast-SmartWay 用固定 3 views (更简单, 更适合实时)
- Disambiguation 类似 SmartWay 的 backtrack, 但触发逻辑不同: backtrack 是基于动作历史, Disambiguation 是基于 MLLM 自评 uncertainty

参考:
- NavGPT: https://arxiv.org/abs/2305.16986
- Open-Nav: ICRA 2025
- Navid: https://arxiv.org/abs/2402.15852

## 7. 局限与潜在改进方向

1. **依赖 GPT-4o API**: zero-shot 但 cost 高, 真实部署受网络/API 限制。可用 open-source MLLM (如 Qwen-VL, LLaVA) 替代测试
2. **MLLM 输出随机性**: 论文跑 4 次取平均, 说明 output variance 大。可引入 structured decoding 或 self-consistency
3. **初始 panoramic scan 仍需要**: 完全 panoramic-free 未实现, 未来可探索 SLAM-based pose estimation 或 incremental map building
4. **5 bins 离散化粗糙**: 30° 分辨率可能错过精细方向。可以 learnable bin 或 finer discretization
5. **Distance thresholds ($d_{close}=0.5, d_{mid}=4$) 硬编码**: 可根据环境自适应
6. **FPBR 只做 1-step future simulation**: 长程规划仍弱, 可扩展到 multi-step rollout 或 tree search
7. **没有显式 mapping**: 与 BEVBert [8] / ETPNav [9] 的 topometric map 思路结合可能进一步提升

参考:
- BEVBert: https://arxiv.org/abs/2212.04385
- ETPNav: https://arxiv.org/abs/2310.06737

## 8. 给我的核心 takeaways

1. **Geometric-to-linguistic 是 zero-shot embodied AI 的通用 pattern**: 把 sensor 数据翻译成语言, 让 MLLM 用语言推理能力做决策, 避开 sensor tokenizer 的训练成本
2. **Frontal view 不一定是性能瓶颈**: panoramic 的额外信息边际效用递减, 更好的 reasoning 比更多 sensor 数据更值钱
3. **Uncertainty detection 比 hard-coded recovery 更灵活**: 让 MLLM 自己判断"我confused吗", 比预设 trigger 更 generalizable
4. **Bidirectional reasoning (未来+过去) 是 temporal consistency 的 cheap 实现**: 不需要训练 memory module, 用 prompt engineering 让 MLLM 自检
5. **End-to-end 不一定最好, 但一定更实用**: 去掉 waypoint predictor 牺牲 some structure, 换来 deployment 简化

这篇文章的 essence 是: 在 MLLM 时代, VLN-CE 不再需要精心设计的中间表征 (waypoint predictor), 把感知转成语言、让 MLLM 直接 reasoning action 就行, 关键是 design 好 uncertainty handling 机制保 robustness。
