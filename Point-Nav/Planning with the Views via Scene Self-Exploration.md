---
source_pdf: Planning with the Views via Scene Self-Exploration.pdf
paper_sha256: 41135f32d6baea06f73b79dc16bb9018304ddb1d66255fa5f3bb1230e5d65a8a
processed_at: '2026-08-06T04:44:43-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
现在的 AI 能看懂"转一下头会看到啥"，但不会"连着转好几下找到目标视角". "自己摸索 + 把摸索经验总结成地图"。一个房间里，有人给你看了一张照片说："这是从房间另一个角落拍的，你过去找到那个位置。", 脑子里推理："往左转两步应该能看到那个窗户"; "再往前走一点，视角应该就对了"; 一边走一边看，不断调整方向.
这是 view planning —— 一系列 camera 移动指令，最终找到目标拍照位置。跟 navigation 不一样是：不需要"人走过去"，你需要的是"camera 的 6-DoF pose 对上"。本质上这是一个 localization 问题，不是 navigation 问题。

建了个 benchmark 叫 VIEWSUITE，在 300 个真实 ScanNet 室内场景上测试了 13 个 frontier VLM。
planning gap：

**单步能力还行**：给 VLM 一个 action，问它"执行后看到啥"（P2V task）或者"从 A 到 B 需要什么 action"（V2P task），最好的模型能到 50-70%。说明 VLM 确实有基本的"view-action 对应"知识。

**多步就崩了**：让它连续发 10 步 action 去找目标（IVP task），最好的 Gemini 3.1 Pro 也只有 21.4%，大多数模型低于 10%，open-weight 模型全低于 5%。

类比一下：这就像一个人知道"左转是什么效果"、"前进是什么效果"，但让他连续做 10 个动作去指定位置，他就懵了。**单步的 knowledge 没法 compose 成多步的 plan**。

---

## 三个诊断发现

作者还做了三个 ablation 来找瓶颈：

**1. 给更多 turn 有用吗？** 把 10 turn 加到 20、30。10→20 有提升，20→30 几乎没变化。说明瓶颈不是"时间不够"，是"不会 plan"。

**2. 渲染质量更好有用吗？** 用 3D Gaussian Splatting 替换 point cloud rendering。IVP 只涨了 0.2-1.9 points。说明瓶颈不是"看不清楚"，还是"不会 plan"。

**3. 旋转还是平移更难？** 这有个有意思的 asymmetry：
- 单步 task（P2V/V2P）主要是 **rotation 难**——因为累积旋转在脑子里不好模拟，旋转是非交换的
- 多步 task（IVP）主要是 **position 难**——因为 3D 平移需要空间布局理解和路径规划

这说明单步能力跟多步能力的 bottleneck 是不一样的，单步好不等于多步好。

---

## 直接 RL 为什么不行

作者先试了直接用 RL（PPO）来训。base model 成功率只有 2.5%，结果 PPO 训完也才 3.2%，GRPO 加 filtering 也才 5.2%，就算是"PPO + 成功 trajectory 上做 SFT"的 bootstrapping 也只有 6.2%。

原因很简单：**reward 太稀疏了**。要么全对（+1），要么全错（0）。100 次尝试里只有 2-3 次成功，PPO 根本估不出有意义的 gradient。就像让一个不会游泳的人靠"游到对岸给 1 分，没到给 0 分"来学游泳，他永远学不会，因为他连哪只手该怎么划都不知道。

---

## 核心 insight：失败轨迹也有信号

这是 paper 最关键的想法。

假设 agent 从 A 走到 B，没到 target C，reward = 0。从 RL 角度看，这条 trajectory 是废的。

但从 **view transition** 角度看：A 到 B 这条 transition 是真实的、valid 的！它告诉你"在 A 位置发这个 action 会到 B 位置，看到这样的画面"。这个信息跟原目标 C 完全无关，但它本身是有价值的 knowledge。

类比人类：你走错路了，但你学到了"这个走廊通向那个房间"。这个知识以后去任何地方都有用。

**View graph** 就是把所有 trajectory 的 transition 都攒起来：
- Node = 一个 viewpoint（带渲染画面）
- Edge = 从一个 viewpoint 到另一个的 action sequence
- 不管 trajectory 成功失败，每条都往 graph 里加 edge

---

## View Graph Distillation 怎么工作

光有 graph 还不够，得把它变成 training data。作者提出一个 **task reformulation** 的 trick：

从 graph 里随便 sample 一条 path，比如 $A \to B \to C \to D$。现在我把 $A$ 当"初始视角"，把 $D$ 当"目标视角"，中间的 action chain 就是"正确答案"。这就变成了一条 valid IVP demonstration！

关键在于：**不管原来那条 trajectory 是不是去 D 的，只要 A 到 D 的 path 在 graph 里存在，这就是一条合法的训练数据。**

这比 Hindsight Experience Replay 更强。HER 是 per-episode 的 relabel——一条 trajectory 的 endpoint 当目标。但 view graph 是 cross-episode 的：不同 trajectory 可能经过同一个 viewpoint，A→B（来自 trajectory 1）和 B→C（来自 trajectory 2）可以拼成 A→B→C，即使没有任何一条 trajectory 实际走过 A→B→C。

这样 supervision signal 密度爆炸。从数据看：iter 0 有 4K nodes / 2.8K edges，iter 1 暴涨到 62K nodes / 62K edges。每条 edge 都是 valid transition，组合成 path 后能产生海量训练数据。

---

## 整个 training loop

```
重复 4 轮：
  1. 让 agent 用 PPO 在环境里探索（self-exploration）
     - 所有 trajectory（成功失败都有）被攒进 view graph
  2. 从 graph 里 sample path，reformulate 成 IVP training data
  3. 用这些数据做 SFT（supervised fine-tuning）
  4. SFT 后的 model 作为下一轮 PPO 的起点
```

这个 loop 的精髓是 **两种 learning 互补**：
- RL（PPO）负责 **sharpen** policy distribution——让 high-reward trajectory 概率更高
- SFT（graph distillation）负责 **reshape** policy distribution——注入大量 graph-derived demonstration，让 model 见识过更多 valid transition pattern

单靠 RL：sparse reward 学不动。单靠 SFT：没有足够的成功 demonstration。两者交替才能 bootstrap 起来。

---

## 结果多炸裂

| 方法 | IVP 成功率 |
|---|---|
| Base Qwen2.5-VL-7B | 2.5% |
| GPT-5.4 Pro | 18.5% |
| Gemini 3.1 Pro | 21.4% |
| Direct PPO | 3.2% |
| Success-Only Bootstrapping | 6.2% |
| Random graph（用随机 action 建 graph） | 13.0% |
| 1 iteration | 12.0% |
| 2 iterations | 27.9% |
| **3 iterations (Ours)** | **47.8%** |

几个值得注意的对比：

**47.8% vs 6.2%**：这俩方法结构完全一样（PPO + SFT 交替），唯一区别是 SFT 数据来源——成功 trajectory vs graph-reformulated path。这个对比直接证明 view graph + task reformulation 是核心，不是简单交替训练的功劳。

**47.8% vs 13.0%**：random graph 只有 13%。因为 random action 探索的区域跟 model 在 evaluation 时访问的区域不重合，supervision transfer 不好。说明 **on-policy exploration** 很关键。

**1→2→3 iter 的 12%→28%→48%**：典型的 bootstrap 效应——每轮 SFT 提升起点，下一轮 RL 探索更有效区域，graph 质量更高，SFT 信号更密，良性循环。

---

## Model 学到了什么策略

作者分析 trained model 的行为，发现它学到了一个 **"先探索后逼近"** 的两阶段策略：

- 前几 turn：scene coverage 快速上升（广泛环顾四周，搞清楚自己在哪）
- 中间 turn：target intersection 加速（锁定方向，向目标移动）
- 最后 turn：提交精确 estimate

这个策略在 base model 和 frontier model 上**都看不到**——它们要么"第一 turn 就提交"（base），要么"乱转"（frontier）。

Action distribution 也变了：base model 最爱 `move_forward`（埋头直走），trained model 变成 `turn_left` + `turn_right` 占 33%（先环顾再动）。这就像新手司机只懂往前开，老司机知道先观察再行动。

Attention pattern 也有变化：trained model 在 early layer 分配更多 attention 给 image（先把画面看清楚），deep layer attention 下降（过渡到符号推理）。Turn 之间 image attention 单调下降（信息累积够了，不需要反复看图）。Base model 没这个 pattern。

---

## 迁移性：学到的不是 narrow skill

这个我觉得是最 cool 的 finding。作者问：IVP 学到的 spatial prior 能不能迁移到别的 view-related task？

实验：对 base model 和 trained model 都做一样的 GRPO post-training，比谁终点高。

- VIEWSUITE 内部的 P2V/V2P：trained model 反超 8-12 points
- 外部 benchmark MindCube（不同 scene、不同 action、不同 rendering）：trained model 涨 ~10 points

注意一开始 trained model 的 P2V 还比 base 低（25.7 vs 32.1），因为 IVP 训练让它 overfit 了。但 post-training 之后反超，说明它学到的不是 narrow 的 IVP skill，而是一种 **general 的 3D space understanding**，可以 strengthen 其他 view-dependent reasoning。

---

## 为什么这篇 paper 重要

我觉得有三层意义：

**1. 诊断层面**：清晰揭示了 VLM 的 planning gap——local knowledge ≠ compositional planning。这跟 LLM 上"能做单步 reasoning 但多步 reasoning 就崩"的现象遥相呼应，说明这是个 fundamental 的 compositionality 问题。

**2. 方法层面**：view graph distillation 本质上是一种 **把 sparse RL 问题转化成 dense SFT 问题** 的技术。核心 trick 是：失败 trajectory 也包含 valid transition knowledge，把这些 knowledge 结构化成 graph，就能产生海量 supervision。这个 pattern 我觉得可以推广到很多 sparse-reward agentic setting。

**3. 实证层面**：7B 小模型超过所有 frontier 大模型，而且 priors 可迁移。这说明对于特定能力，**structured training pipeline 比 brute-force scale 更有效**。

---

## 我的几点直觉

**View graph 本质上是个 discrete world model。** 它编码了 "在 state A 发 action b 会到 state C" 的 transition knowledge，只是用 graph 而不是 neural network 来表示。Distillation 阶段把 graph 蒸馏进 weights，相当于让 model 内化这个 world model。这跟 LeCun 说的 JEPA 之类的 world model 思路有异曲同工之处，只是更 explicit、更结构化。

**为什么 7B 能超 GPT-5.4 Pro？** 因为 GPT-5.4 Pro 是 zero-shot 推理，没有针对 view planning 的训练。而 7B 是经过 4 轮 iterative RL+SFT 专门训练的。这说明对于需要 spatial reasoning 的 composition 任务，**针对性的 training pipeline 比 general reasoning ability 更重要**。这也呼应了 R1 的发现：RLVR 能 unlock 模型的特定能力，远超 prompt engineering。

**这套方法能推广吗？** 我觉得可以。任何"action sequence 很长 + reward 只在终端 + 单步 transition 是 valid knowledge"的场景都适用。比如 tool use（调用 API 的 sequence）、code generation（编辑操作的 sequence）、theorem proving（tactic 的 sequence）。核心都是：把失败的 exploration 也变成 supervision，攒成 graph，再蒸馏回 model。

Andrej，这篇 paper 的 intuition 你 buy 吗？我觉得 view graph 这个 idea 其实挺 natural 的——人类学空间导航可能也是这么干的，走错路也在积累 spatial map。
