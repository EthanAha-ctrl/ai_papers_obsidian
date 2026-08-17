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

# 用人话讲讲这篇 paper

好，我把公式那些先放一边，用大白话把这篇 paper 的故事讲一遍。

---

## 一句话概括

现在的 AI 能看懂"转一下头会看到啥"，但不会"连着转好几下找到目标视角"——作者用一套"自己摸索 + 把摸索经验总结成地图"的方法，让一个 7B 的小模型在这件事上碾压了 GPT-5.4 Pro 和 Gemini 3.1 Pro。

---

## 背景是什么问题

想象你站在一个房间里，有人给你看了一张照片说："这是从房间另一个角落拍的，你过去找到那个位置。"

你需要做什么？你得在脑子里推理：
- "往左转两步应该能看到那个窗户"
- "再往前走一点，视角应该就对了"
- 一边走一边看，不断调整方向

这就是 **view planning**——你要发出一系列 camera 移动指令，最终找到目标拍照位置。

跟普通 navigation 不一样的地方在于：你不需要"人走过去"，你需要的是"camera 的 6-DoF pose 对上"。本质上这是一个 **localization 问题**，不是 navigation 问题。

---

## 作者发现了什么

作者建了个 benchmark 叫 VIEWSUITE，在 300 个真实 ScanNet 室内场景上测试了 13 个 frontier VLM。

核心发现是一个 **planning gap**：

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

---

# Planning with the Views via Scene Self-Exploration 深度讲解

Andrej，这篇 paper 我读了很久，里面有几个 design choice 其实非常微妙，值得展开讲。我先 build intuition，再 dive into 技术。

参考链接：
- Project: https://viewsuite.github.io
- Code: https://github.com/mll-lab-nu/ViewSuite
- Models: https://huggingface.co/collections/MLL-Lab/viewsuite-models
- Data: https://huggingface.co/collections/MLL-Lab/viewsuite-datasets

---

## 1. 核心问题与 motivation 的直觉

这篇 paper 想回答一个看似简单、实则深刻的问题：**VLM 能不能"在脑子里"操纵 camera？** 

所谓 view planning，就是给定一个 target view，agent 要发出一系列 6-DoF camera action，让 viewpoint 一步步逼近 target，最后提交一个 6-DoF pose estimate。

这里有个直觉需要先建立：view planning 和 embodied navigation 在表面动作上很像（都是发离散 action），但本质不同。

- Embodied navigation：agent 有 body，要 navigate 到某个 physical location，reward 是"有没有到达"
- View planning：agent 只操纵 camera，没有 body、没有 affordance，本质上是一个 **localization 问题**——action 是 evidence-gathering operator，最终要提交一个 6-DoF estimate

这个 distinction 很重要，因为它决定了为什么不能直接套用 navigation 的方法。

### 两个 decoupled 的能力

paper 把 view planning 拆成两个耦合能力：

1. **View-action understanding（single-turn）**：知道一个 action 如何 transform view
2. **Multi-turn composition**：把多个 transform 串起来，累积的 observation 让 agent localize target

这种 decomposition 的好处是诊断性强：如果 IVP 失败，可以追问是哪一层挂了。

---

## 2. VIEWSUITE Benchmark 设计

### 2.1 环境构建

基于 ScanNet（https://doi.org/10.1109/CVPR.2017.261）的 ~300 个真实室内场景，用 Open3D（http://arxiv.org/abs/1801.09847）渲染 point cloud。

关键设计：
- **12 个离散 action**（6 translation + 6 rotation）
  - Translation step size $s_t = 0.5$ m，沿 camera local axes
  - Rotation step size $s_r = 30°$，分别绕 yaw/pitch/roll
  - Discrete snapping：每次 rotation 后，把 Euler 角 snap 到 $s_r$ 的整数倍，保证 action sequence 严格可逆

为什么 snapping 重要？因为如果 rotation 不 snap，累计误差会让 viewpoint 漂离 grid，而 target 是从 grid 上采样的，agent 永远到不了。

### 2.2 三个 diagnostic task

paper 设计三个 task 形成诊断梯度：

**Path-to-View (P2V)**：给定 init view + action sequence → 从 4 个选项中选 resulting view
- 测 forward simulation：你能不能在脑子里执行 action

**View-to-Path (V2P)**：给定 init view + target view → 从 4 个选项中选 action sequence  
- 测 inverse inference：你能不能反推 action

**Interactive View Planning (IVP)**：给定 init view + target view + top-down → 多 turn 发 action，最后提交 6-DoF estimate
- 测 multi-turn planning

P2V 和 V2P 是 single-turn、multiple-choice；IVP 是 multi-turn、open-ended（要输出 6 个数）。

### 2.3 Evaluation metric 与 success threshold 校准

这是我觉得 paper 里最严谨的地方之一。IVP 的 success 不是随便定的，而是做了 human alignment study。

**Viewpoint distance 公式**：

$$d_{\mathrm{pos}} = \|\mathbf{t}_1 - \mathbf{t}_2\|_2, \quad d_{\mathrm{rot}} = \arccos\left(\frac{\mathrm{tr}(R_1^\top R_2) - 1}{2}\right)$$

变量解释：
- $\mathbf{t}_1, \mathbf{t}_2 \in \mathbb{R}^3$：两个 camera 的 translation（position）
- $R_1, R_2 \in SO(3)$：两个 camera 的 rotation matrix（3×3 正交矩阵）
- $R_1^\top$：$R_1$ 的转置
- $\mathrm{tr}(\cdot)$：trace，矩阵对角元素之和
- $d_{\mathrm{pos}}$：position 的欧氏距离（米）
- $d_{\mathrm{rot}}$：rotation 的 geodesic angle（弧度/角度），衡量两个 orientation 的"夹角"

这个 rotation distance 公式来自 $SO(3)$ 上的标准 geodesic metric：$\mathrm{tr}(R_1^\top R_2) = 1 + 2\cos\theta$，其中 $\theta$ 是两个 rotation 之间的夹角。

**Unified view distance**：

$$d = \sqrt{(d_{\mathrm{pos}}/s_t)^2 + (d_{\mathrm{rot}}/s_r)^2}$$

这里除以 step size 是为了归一化：让 $d$ 的 1 个 unit 大致对应 1 个 atomic action。比如 $d=3$ 意味着 init 和 target 大概差 3 步。

**Success criterion**：

agent 成功当且仅当 $d_{\mathrm{pos}} \leq \beta_t s_t$ 且 $d_{\mathrm{rot}} \leq \beta_r s_r$，其中 $\beta_t = \beta_r = 1$。

paper 做了 human alignment study（Table 8）：让人类判断 rendered view pair 是否 depict same place，然后 sweep $(\beta_t, \beta_r)$ 找最大化 F1 的组合。$(1, 1)$ 给出 F1=0.915, accuracy=0.920，是最优的。如果放宽到 $\beta_t = 2$（1m），precision 掉到 0.72，说明人类认为这种距离已经 visibly different。

这种 calibration 让 benchmark 的 success 定义有 human grounding，避免 arbitrary threshold。

### 2.4 Reward 函数

IVP 被建模成 finite-horizon MDP：

- State：rendered view $o_t$ + current 6-DoF viewpoint $p_t \in SE(3)$
- Action：$a_t \in \mathcal{A}$（12 个离散 action）
- Transition：$p_{t+1} = T(p_t, a_t)$（deterministic）
- Terminal：agent 提交 $\hat{p}^* \in SE(3)$

**Reward**：

$$r(\hat{p}^*, p^*) = \mathbf{1}[d_{\mathrm{pos}}(\hat{p}^*, p^*) \leq \beta_t s_t \wedge d_{\mathrm{rot}}(\hat{p}^*, p^*) \leq \beta_r s_r] + 0.1 \mathbf{1}_{\mathrm{format}}$$

变量：
- $\hat{p}^*$：agent 提交的 estimate
- $p^*$：ground-truth target
- $\mathbf{1}[\cdot]$：indicator function
- $\wedge$：logical and
- $\mathbf{1}_{\mathrm{format}}$：输出格式正确（必需有 `<action>answer(x,y,z,...)</action>`）

这是极度 sparse 的 reward：要么全对（+1），要么全错（0），中间没有 shaping。这是后面 RL 失败的 root cause。

---

## 3. Planning Gap 的发现

paper 评了 13 个 frontier VLM（7 个 proprietary + 6 个 open-weight），核心发现是 **planning gap**：

### 3.1 数字

从 Table 2：

| 类别 | P2V All | V2P All | IVP All |
|---|---|---|---|
| GPT-5.4 Pro | 53.1 | 50.7 | 18.5 |
| Gemini 3.1 Pro | 48.8 | 49.5 | 21.4 |
| GPT-5.4 | 47.8 | 45.6 | 16.6 |
| Qwen2.5-VL-7B | 29.5 | 24.2 | 2.5 |

几个观察：
1. P2V/V2P 上最好的模型能到 50-70%（远超 25% 随机），说明 VLM 有 non-trivial 的 view-action 知识
2. IVP 上最好只有 21.4%，多数低于 10%，open-weight 全部低于 5%
3. Short-horizon 上 GPT-5.4 Pro 能到 70.7% (P2V)，但 Long-horizon 掉到 43.8%

这说明 VLM 能做"局部 mental simulation"，但 cumulative transformation 一长就崩。

### 3.2 三个 bottleneck 分析

paper 做了三个 ablation 来找 bottleneck：

**Turn budget 是否是瓶颈？**（Table 3）
- 把 budget 从 10 加到 20、30
- 10→20 有提升（Claude Opus 4.6 几乎翻倍），但 20→30 边际收益接近 0
- 说明瓶颈是 planning ability，不是 exploration horizon

**Rendering quality 是否是瓶颈？**
- 用 3D Gaussian Splatting（https://arxiv.org/abs/2308.04079）重新渲染
- IVP 只 +0.2 到 +1.9 points（marginal）
- P2V/V2P 反而 mixed：Gemini 3.1 Pro +6.5 on P2V，但 GPT-5.4 -14.5 on V2P
- 说明 visual fidelity 不是 IVP 的瓶颈

**Rotation vs. translation 哪个更难？**（Figure 2）
- P2V/V2P：随 **rotation distance** 退化（GPT-5.4 Pro 在 P2V rotation bins 上掉 ~25 points）
- IVP：随 **position distance** 崩溃（GPT-5.4 Pro ~7× drop）

这个 asymmetry 很有意思。Single-turn 的难处是"在脑子里累积旋转"——旋转是 non-commutative 的，cumulative rotation 在 mental simulation 里很 tricky。Multi-turn IVP 的难处是 position——因为 3D translation 需要 spatial layout understanding 和 path planning，远超简单的 orientation control。

Sample-level factor analysis（Spearman $\rho$）证实：IVP 上 position distance 的 $\rho$ 到 -0.42（GPT-5.4 Pro），是最强负相关因子。

---

## 4. 方法的核心 Insight

这是 paper 最精彩的部分。我先讲 intuition，再讲 mechanism。

### 4.1 为什么直接 RL 不行

paper 试了三种 RL 方法，全失败：

| 方法 | IVP All |
|---|---|
| Direct PPO | 3.2% |
| Direct GRPO (filter) | 5.2% |
| Success-Only Bootstrapping | 6.2% |
| **Ours** | **47.8%** |

为什么？因为 **base success rate 只有 2.5%**。在如此 sparse 的 reward 下，PPO 几乎学不到 gradient——绝大多数 trajectory 的 reward 都是 0，advantage 估计噪声极大。

GRPO + reward-variance filtering（参考 RAGEN: https://arxiv.org/abs/2504.20073）也只到 5.2%。filtering 把高 variance 的 prompt 滤掉，但本质上还是依赖 success trajectory。

Success-Only Bootstrapping（PPO + 在成功 trajectory 上做 SFT）也只到 6.2%，因为它只能利用 ~2.5% 的成功 trajectory，signal 极度稀疏。

### 4.2 关键 insight：失败 trajectory 也有信号

paper 的核心 observation：

> **Every trajectory, whether or not it reaches its goal, traces valid view transitions through the scene.**

这句话听起来平凡，但实际很深刻。考虑一个 agent 从 A 走到 B 但没到达 target C。从 RL 角度看，这条 trajectory 是失败的，reward = 0。但从 view transition 角度看，A→B 这条 transition 是 valid 的——它是 scene 中真实的、可执行的 viewpoint 变化。

这个观察的类比是人类 spatial mapping：即使走错路，你也学到了"哪个房间连到哪个走廊"。

**View graph** 就是这个 insight 的形式化：
- Node = viewpoint（含 rendered view）
- Edge = action sequence connecting two viewpoints
- 一条 trajectory $\tau = (v_0, a_1, v_1, a_2, v_2, \dots, a_K, v_K)$ 自动贡献 K 条 edge 给 graph

无论 $\tau$ 成功失败，graph 都在增长。

### 4.3 View Graph Distillation 的 mechanism

paper 提出一个 task reformulation operator：

$$\mathcal{R}(P) = (o_{\mathrm{init}} = \nu_0, \; o_{\mathrm{target}} = \nu_K, \; (a_1, \dots, a_K), \; \hat{p}^* = p_{\nu_K})$$

变量：
- $P = (\nu_0, a_1, \nu_1, \dots, a_K, \nu_K)$：graph 中任意一条 path
- $\nu_i$：第 i 个 node（viewpoint + rendered view）
- $a_i$：第 i 条 edge 上的 action sequence
- $K$：path 长度
- $o_{\mathrm{init}}$：start node 的 rendered view，作为 initial view
- $o_{\mathrm{target}}$：end node 的 rendered view，作为 target view
- $(a_1, \dots, a_K)$：完整的 action chain，作为 ground-truth action sequence
- $\hat{p}^* = p_{\nu_K}$：end node 的 6-DoF viewpoint，作为 ground-truth target estimate

**这个 operator 的关键性质**：无论原 trajectory 成功还是失败，reformulation 后都是 valid IVP demonstration。

直觉上，这相当于 hindsight relabeling（类比 Hindsight Experience Replay, https://arxiv.org/abs/1707.01495）：把 trajectory 实际到达的 endpoint 当作"目标"。但 paper 做得更深——它不是 episode-by-episode 地 relabel，而是 **aggregate 所有 trajectory 成 graph，再从 graph 中 sample path**。

### 4.4 为什么 graph 比 per-episode relabel 更强

这是 paper 的真正贡献。Hindsight Experience Replay 是 per-episode 的：每条 trajectory relabel 成"以 endpoint 为目标"的 demonstration。但 paper 的 view graph 是 cross-episode 的：

- 不同的 trajectory 可能经过相同 viewpoint（deduplication 后是同一个 node）
- 这意味着一条 trajectory 发现的 transition $A \to B$ 可以被另一条 trajectory 的 $B \to C$ 复用
- 从 graph 中 sample path 时，可以组合出 agent 从未实际走过的 path

这种结构化 knowledge 让 supervision signal 密度爆炸：

从 Table 12：
- Iter 0：4,067 nodes, 2,875 edges
- Iter 1：61,862 nodes, 62,445 edges
- Iter 2：66,492 nodes, 65,577 edges

每条 edge 都是 valid transition，组合成 path 后产生的 supervision 远超 raw trajectory 数量。

### 4.5 三种 SFT task

paper 从 graph 生成三种 supervision：

1. **Multi-turn view planning**（primary）：path length 3-5，把 path reformulate 成 IVP demonstration，每 path oversample 10 次
2. **View difference estimation**：给定两个 view，predict unified view distance $d$（regression）
3. **View difference MCQ**：同样的 setup，但 multiple choice

后两个 auxiliary task 防止 model 过拟合到单一 format，并且强制 model 学会 spatial distance 的 sense。

### 4.6 整体算法

Algorithm 3 的核心 loop：

```
Initialize policy π_θ0, empty graph G_0
For k = 0 to K-1:
    # Self-exploration stage
    Run PPO updates of π_θk on environments with reward (Eq. 3)
    Append trajectories: G_{k+1} = G_k ∪ traj(π_θk)
    
    # View graph distillation stage
    Sample paths {P_i} ⊂ G_{k+1}
    Reformulate: D_{k+1} = {R(P_i)} via Eq. 4
    Fine-tune via SFT: θ_{k+1} = argmin L_SFT(θ; D_{k+1})
```

注意几个细节：
- Graph 是 persistent 的，跨 iteration 累积（iter 2 还能用 iter 0 发现的 transition）
- 前 3 个 iteration 各 60 步 PPO（快速 bootstrap），最后 1 个 iteration 跑到 convergence
- 每 iteration 后做 3 epoch SFT

这种 alternation 的本质是 **distribution reshaping**：RL sharpen policy distribution（让 high-reward trajectory 概率更高），SFT reshape policy distribution（注入大量 graph-derived demonstration）。两者协同克服 sparse reward。

---

## 5. 实验结果深度解析

### 5.1 Main result（Table 4）

| Method | Short | Long | All |
|---|---|---|---|
| Qwen2.5-VL-7B (base) | 7.1 | 0.0 | 2.5 |
| GPT-5.4 Pro | 32.6 | 11.0 | 18.5 |
| Gemini 3.1 Pro | 28.8 | 17.4 | 21.4 |
| Direct PPO | 7.0 | 1.2 | 3.2 |
| Direct GRPO (filter) | 10.8 | 2.2 | 5.2 |
| Success-Only Bootstrapping | 14.0 | 2.0 | 6.2 |
| Random-graph | 25.4 | 6.4 | 13.0 |
| 1 iter + RL | 24.3 | 5.4 | 12.0 |
| 2 iter + RL | 49.7 | 16.2 | 27.9 |
| **Ours (Qwen2.5-VL-7B)** | **67.2** | **36.9** | **47.8** |
| Ours (Qwen3-VL-8B) | 56.8 | 19.4 | 32.5 |

几个关键 ablation：

**Random-graph**（用 random action generator 建 graph，而非 model trajectory）：只有 13.0%。这说明 on-policy graph 很关键——random trajectory 探索的区域 model 在 evaluation 时很少访问，reformulation 的 supervision transfer 不好。

**Iteration 数量**：1 iter (12.0) → 2 iter (27.9) → 3 iter (47.8)。这是 bootstrap 效应：每一轮 SFT 让下一轮 RL 的起点更高，RL 又能 explore 更有效区域，graph 质量更高，SFT 信号更密。

**Success-Only Bootstrapping vs Ours**：差距是 6.2% vs 47.8%。这是 paper 最强的 ablation，因为两者结构相同（PPO + SFT 交替），唯一区别是 SFT data 来源：成功 trajectory vs graph-reformulated path。这个对比直接证明 view graph + task reformulation 是关键，不是 alternation 本身。

### 5.2 评估 protocol 的 robustness（Table 9）

paper 还检查了两个 protocol 细节：

**No-Snap**：不把 rotation snap 到 grid，直接执行 raw rotation magnitude
- Ours: 47.8 → 19.6（掉很多，因为 agent drift off grid）
- 但仍超 Gemini 3.1 Pro (15.7) 和 GPT-5.4 (13.0)

**No-Submit**：不要求 explicit submit，pose 进入 threshold 就算成功
- Ours: 47.8 → 60.2（涨很多，因为不要求"主动 commit"）
- Gemini 3.1 Pro: 21.4 → 31.5

排序在三个 protocol 下都保持，说明 gain 不是 protocol artifact。

---

## 6. Model 学到了什么？

### 6.1 探索策略（Figure 4）

paper 跟踪两个 metric：
- **Scene coverage ratio**：$\frac{|\bigcup_{t=0}^T V_t|}{|V_{\mathrm{total}}|}$（累积看到多少 scene）
- **Target intersection ratio**：$\frac{|\bigcup_{t=0}^T V_t \cap V_{\mathrm{target}}|}{|V_{\mathrm{target}}|}$（累积看到多少 target view 的内容）

变量：
- $V_t$：turn $t$ 可见的 vertex 集合
- $V_{\mathrm{total}}$：整个 scene 的 point cloud
- $V_{\mathrm{target}}$：target viewpoint 可见的 vertex 集合

**学到的策略**是两阶段：
1. **Early turns**：scene coverage 快速上升（broadly explore）
2. **Middle turns**：target intersection 加速（move toward target），最终到 ~55%

这种 "explore then approach" 模式在 base model 和 frontier model 上都**没有**——他们要么 flat 要么 erratic。

这个 finding 很有意思，因为它说明 model 不是单纯模仿 demonstration，而是学到了 **adaptive exploration strategy**。

### 6.2 Action distribution 的演化（Figure 10）

- Iter 0：`move_forward` 占 18%（base policy 倾向直走）
- Iter 2：`turn_left` + `turn_right` 占 ~33%（旋转 dominate），translation 在 6 个方向上更平衡

这说明 model 学会了"先环顾四周再移动"，而不是 base 的"埋头直走"。

### 6.3 Attention pattern 的变化（Figure 13）

paper 分析 image attention fraction：$\frac{\sum_{k \in \mathcal{I}} \alpha_{q,k}}{\sum_k \alpha_{q,k}}$

变量：
- $\alpha_{q,k}$：query token $q$ 对 key token $k$ 的 attention weight
- $\mathcal{I}$：image token 的 index 集合

发现两层 pattern：
- **Layer-wise**：trained model 在 early layer (L0-L4) 分配更高 image attention（更强 visual grounding），deep layer (L8+) attention 下降（transition 到 text-space reasoning）
- **Turn-wise**：trained model 的 image attention 随 turn 单调下降（信息累积，不再需要反复看图）

base model 没有这个 pattern——它在 deep layer 反而 image attention 更高，且 turn-wise flat。

这个 finding 提示：training 让 model 学会了"early visual grounding → late symbolic reasoning"的分工，类似于人类"先看清楚再算"。

### 6.4 Spatial prior 的迁移（Table 5）

这是我最喜欢的 finding。paper 问：IVP 学到的 spatial prior 能否迁移到其他 view-related task？

实验设计：对 base model 和 trained model 都做 identical GRPO post-training，看谁起点高、终点高。

| Task | Model | Init | Post |
|---|---|---|---|
| P2V | Base | 32.1 | 45.1 |
| P2V | Ours | 25.7 | 57.3 |
| V2P | Base | 29.2 | 44.8 |
| V2P | Ours | 31.6 | 52.8 |
| MindCube | Base | 33.0 | 56.3 |
| MindCube | Ours | 33.1 | 66.2 |

变量：
- Init：post-training 前的 accuracy
- Post：post-training 后的 accuracy

注意 base 的 Init 反而更高（32.1 vs 25.7 on P2V），因为 trained model 在 IVP 上 overfit 了。但 Post 之后，trained model 反超 8-12 points。

外部 benchmark MindCube（https://arxiv.org/abs/2506.21458，no shared scenes/actions/rendering）上 trained model 也涨 ~10 points。

这说明 IVP 学到的不是 narrow skill，而是 **transferable spatial priors**——一种对 3D space 的 general understanding，可以 strengthen 其他 view-dependent reasoning。

---

## 7. 与相关工作的 positioning

### 7.1 与 visual search benchmarks 的对比（Table 1）

VIEWSUITE 与之前 visual search 工作的关键区别：
- V*（https://arxiv.org/abs/2312.14135）：2D image 内 LLM-guided search
- ActiView（https://arxiv.org/abs/2410.04659）：2D image zoom/shift
- H*Bench（https://arxiv.org/abs/2511.20351）：360° panorama head rotation

VIEWSUITE 是第一个在 **真实 3D scene** 上做 **full 6-DoF** **multi-turn** view planning 的 benchmark。

### 7.2 与 Hindsight Experience Replay 的关系

paper 在 related work 里明确说了，HER 是 per-episode relabel，而 view graph 是 **cross-episode aggregation**。具体区别：

- HER：trajectory $\tau = (s_0, a_1, s_1, \dots, s_K)$ → relabel 成 "以 $s_K$ 为目标"的 demonstration
- View graph：把所有 $\tau$ 的 transition 累积成 graph $G$，再从 $G$ 中 sample path $P$（可能跨多条 $\tau$），reformulate $P$ 成 demonstration

后者 supervision signal 更 dense，因为 graph 上的 path 数量远多于 trajectory 数量。

### 7.3 与 agentic-RL 的关系

R1（https://arxiv.org/abs/2501.12948）证明 outcome-supervised RL 能大幅提升 LLM reasoning。但 R1 的 reward 是数学正确性，相对 dense（每步都有 verifiable answer）。IVP 的 reward 极度 sparse（只有最终 6-DoF estimate 对/错），导致 RLVR 直接应用失败。

paper 的 framework 本质上是 **把 sparse RLVR 问题转化成 dense SFT 问题**，通过 view graph 这个中间表示。

---

## 8. 我自己的几点思考

### 8.1 View graph 是不是一种 "implicit world model"？

View graph 编码了 "viewpoint A 经过 action $a$ 到 viewpoint B" 的 transition knowledge。这非常像 world model，只是离散化的、graph-structured 的。

更有趣的是，distillation 阶段把 graph 蒸馏成 SFT data，相当于让 model **internalize** 这个 graph-based world model 到 weights 里。Attention analysis 显示 trained model 在 deep layer 减少 image attention，可能就是这个 internalization 的体现——它不再需要频繁看图，因为 graph 已经变成了 parametric knowledge。

### 8.2 为什么 Qwen2.5-VL-7B 比 Qwen3-VL-8B 表现更好？

47.8 vs 32.5 是不小差距。paper 没深究，但我猜测 Qwen2.5-VL-7B 的 visual encoder 或 coordinate-aware representation 可能更适合 6-DoF 任务。这提示我们：base model 的 visual grounding 能力直接决定 RL 上限。

### 8.3 为什么 direct PPO 这么差？

2.5% → 3.2% 几乎没动。这其实呼应了 RAGEN（https://arxiv.org/abs/2510.16907）的发现：multi-turn agentic RL 在 sparse reward 下极易 collapse。IVP 比一般 agentic task 更难，因为 final action 是 open-ended 6-DoF vector，不是 discrete choice。

### 8.4 Limitation 和未来方向

paper 自己说了：static indoor scene + discrete 12-action + 7B/8B scale。我觉得更 fundamental 的 limit 是：

- **没有 dynamic object**：真实世界有动的物体，view graph 假设 viewpoint transition 是 deterministic 的
- **没有 occlusion reasoning**：如果中间有物体挡住，view transition 还成立吗？
- **Scaling**：把这套搬到 70B 或者 video-augmented model 上会怎样？

另外，paper 用的是 point cloud rendering，虽然有 ablation 显示 GS rendering 帮助有限，但我怀疑这是因为 IVP bottleneck 在 planning 不在 perception。如果 base model 的 perception 已经够用，更高 fidelity 当然帮助不大。但换一个 perception-bottleneck 的 task（比如需要看小文字），GS 可能就关键了。

---

## 9. 总结

这篇 paper 的核心 contribution 我总结为三层：

1. **Diagnostic contribution**：揭示 VLM 的 planning gap——local view transition 理解 (~50-70%) 不能 compose 成 multi-turn planning (<21%)
2. **Methodological contribution**：view graph distillation——把失败 trajectory 也变成 supervision 的 mechanism，本质上是 cross-episode hindsight + structured knowledge aggregation
3. **Empirical contribution**：7B model 从 2.5% → 47.8%，超过所有 frontier model，且 priors 可迁移

技术上最值得借鉴的是 **view graph 这个中间表示**。它不是单纯的 hindsight replay，而是把 exploration 产物结构化成一个 reusable knowledge base，再通过 task reformulation 转成 diverse supervision。这种 "exploration → structured knowledge → distillation" 的 pattern 我觉得可以推广到其他 sparse-reward agentic setting，比如 tool use、web navigation 等等。

Andrej，你觉得这个 framework 能不能套到代码生成或者 theorem proving 上？我直觉上觉得可以——把"失败的 proof attempt"也当成 valid reasoning chain 蒸馏回去，应该能 break sparse reward 的瓶颈。
