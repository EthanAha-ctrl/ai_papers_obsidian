---
source_pdf: RoboCerebra.pdf
paper_sha256: cca6d9a78c2a4768497fba712c96a1434f91e4beb6891fb02c12eed5b53ef90d
processed_at: '2026-08-12T00:33:26-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版本

好,我把那些公式和表格先放一边,讲讲这篇paper到底在搞什么。

---

## 这paper在解决一个什么烦恼

现在搞机器人,大家都在做一件事:训练一个模型,看图、听指令、直接输出动作。这种叫 VLA(Vision-Language-Action),比如 OpenVLA、RT-2 这些。想法很美——一个模型端到端搞定一切。

但有个老问题:**任务一长就崩**。

你让机器人"把牛奶拿去微波炉加热"——可能要开门、拿牛奶、关门、放微波炉、设时间、启动、等好、拿出来。这一串下来,几千个细小的手臂动作。一个 reactive 模型,走到第5步忘了第2步见过什么,或者中间牛奶滑掉了,它就懵了。

RoboCerebra 的作者说:这事儿得用分层思路解决。**上面一个"大脑"(VLM)负责想,下面一个"小脑"(VLA)负责做**。大脑不操心怎么抓,小脑不操心为什么抓。这跟人干活一样——你不会每动一下手都重新思考"我要干嘛",你脑子里有个计划,手自动执行。

但问题是:**之前没人有合适的benchmark来测这个"大脑"到底好不好用**。现有数据集要么任务太短(几十步),要么没有动态变化,要么没标注。于是他们造了一个。

---

## 他们造了个什么benchmark

简单说,造了一千个**超长的家务任务**,平均每个任务2972步——是现有数据集的6倍长。任务类型还分了六种:

- **Ideal**:理想环境,测试基本能力
- **Memory Exploration**:让你去翻柜子找东西
- **Memory Execution**:东西藏起来了,你得记得之前看过啥
- **Random Disturbance**:中途有人把东西撞歪了
- **Observation Mismatching**:你看到的跟你以为的不一样
- **Mix**:以上各种麻烦组合在一起

这六类任务,每类测一个认知能力——记忆、反思、纠错、planning。

---

## 数据怎么造出来的

纯靠人手动录太贵,纯靠LLM生成又不靠谱。他们搞了个混合pipeline:

1. **LLM出题**:给GPT一些物体的描述,让它编任务("热牛奶"、"摆餐具"),再分解成步骤
2. **规则转代码**:把"把牛奶从冰箱放到微波炉"这种话转成simulator能跑的指令
3. **双重验证**:先用simulator检查物理上能不能做,再用GPT-4o看一眼渲染图判断"这场景合不合常理"
4. **真人执行**:最后还是雇人花400小时在仿真器里一条条录的,另外200小时检查质量

花了650人小时。这就是为什么之前没人做——**贵**。

---

## 关键发现:三个让人意外的结果

### 发现一:视觉输入根本没用

最striking的结果:GPT-4o 不看图,只看文字任务描述,成功率15.10%。让它看图,16.04%。差了1个百分点。

这意味着什么?**现在的"VLM"在机器人planning里基本上就是个LM**。视觉信息对高层规划贡献几乎为零。模型根本没学会怎么从画面里提取有用信息来指导下一步。

你可能会问:那视觉有啥用?答:目前主要用在**反思**——看着自己刚做完的活判断"我完成了吗"。但即便这块,Qwen2.5-VL经过专门fine-tune,从47%提到66%——但代价是planning能力从44%掉到30%。**按下了葫芦浮起了瓢**。

### 发现二:即使给完美的plan,也只有25%成功率

他们用人类标注的ground-truth plan去跑,成功率25.16%。这说明**底层VLA本身是个大瓶颈**。

数学上很好理解:假设单步成功率85%,10个subtask串起来是 $0.85^{10} \approx 20\%$。跟实验数据吻合。

要打破这个ceiling,要么把单步VLA做得更准,要么让"大脑"能发现"小脑"干砸了然后重新规划。后者现在还没人做好。

### 发现三:专门模型打不过通用模型

他们测了一些专门为机器人训练的VLM——RoboBrain-2.0、Cosmos-Reason1、VeBrain。这些模型在具体任务(比如Memory Exploration)上确实比通用Qwen2.5-VL强,但**整体planning还是落后GPT-4o一截**。

教训:**广泛的推理能力 > 任务特定的specialization**。至少在long-horizon planning这个domain,GPT-4o这种"什么都懂一点"的模型,反而比"机器人专项训练"的模型表现好。

---

## 为什么你该关心这paper

如果你在搞robotics learning,这paper给你三个信号:

1. **纯端到端VLA走不远**。几千步的trajectory,reactive policy必然掉链子。分层是必须的,不是可选的。
2. **当前VLM的视觉能力被高估了**。大家以为多模态就是"看懂图",其实现在主要还是靠语言推理。视觉grounding需要fundamental breakthrough。
3. **Benchmark本身是最有价值的贡献**。HPE framework很朴素,但benchmark给了社区一个明确的measuring stick,以后做long-horizon planning都得来这测。

---

## 一句话总结

> 当前机器人领域的"大脑"模型,本质上还是个"会用文字推理的语言模型",离真正的"看懂世界并长期规划"还有距离。RoboCerebra把这个差距量化出来了——GPT-4o在复杂长程任务上16%,理论上限25%,实用还远得很。

---

# RoboCerebra: 深度技术解析

非常exciting的一篇paper, 让我详细展开。这篇工作的核心立意从 Kahneman 的 "Thinking, Fast and Slow" 范式出发, 把 robotic imitation learning 的 focus 从 System 1 (fast, reactive VLA policies) 推到 System 2 (slow, deliberative VLM planning)。接下来我会一层层剥开来看。

---

## 1. Motivation: 为什么 System 2 在 robotics 里是个问题

现有 VLA models 比如 RT-2、OpenVLA 本质上是把 VLM 当成 reactive policy 用——visual + language input,直接吐出discretized action tokens。这种部署方式实际上 underutilize 了 VLMs 最强的能力: semantic abstraction, relational reasoning, contextual planning。

现有 long-horizon benchmarks 的问题:
- **RLBench**: 100 train / 100 test tasks, 单步为主
- **ALFRED**: 7 train / 7 test scenes, long-horizon 但无 dynamic variation
- **CALVIN**: 34 tasks, long-horizon 但缺 fine-grained decomposition
- **LIBERO-Long**: 10 long tasks, 500 steps
- **RoboCasa**: 100 tasks, LLM-generated 但无 human trajectories
- **VLABench**: 100 tasks, long-horizon 但无 human demo

RoboCerebra 的 action sequence 平均 2972.4 steps, 是现有 benchmarks 的 **~6×**, 而且加了 dynamic scene variations 和 time-segment annotations。

一个关键直觉: 如果一个 trajectory 跨越 20+ atomic subtasks, 中间还要处理 object displacement、memory recall、plan-perception misalignment, 纯 reactive policy 根本没办法 maintain instruction fidelity across 这种 temporal span。Table 3 里 OpenVLA-Libero100 在 Ideal setting 下只有 4.05% SR, fine-tuned OpenVLA 也只有 7.84%, 而 HPE 能到 21.10%。这个 gap 就是 System 2 reasoning 的价值。

---

## 2. Task Suite 设计: 六类 subtask 的认知维度

定义了六种 subtask types, 每个针对一个 cognitive function:

| Subtask Type | Cognitive Demand | Example |
|---|---|---|
| **Ideal** | Baseline, static, fully observable | 简单的 pick-and-place 序列 |
| **Memory Exploration** | 主动探索, 形成内部表征 | 检查 cabinet 各个 compartment 找 butter |
| **Memory Execution** | 记忆检索, 完成 goal | 在 closed container 里 retrieve 之前看到的东西 |
| **Random Disturbance** | 处理 unexpected 环境变化 | 物体被碰撞移位后恢复 plan |
| **Observation Mismatching** | plan-perception 不一致 | 视觉和预期不符时 re-plan |
| **Mix** | memory + dynamic 组合 | 长程 re-planning under uncertainty |

这里的 key insight: memory tasks 需要 model 不仅"看到什么"还要"记住什么没看到"。Observation Mismatching 测的是 model 能不能 detect 自己 plan 的 perceptual violation。这些 capability 在 short-horizon benchmark 里根本测不出来。

---

## 3. Dataset Construction Pipeline: Top-down 生成 + 人类执行

### 3.1 Cascaded Task Generation

Pipeline 三阶段, 我详细展开:

**Stage 1: Structured Object Representation**
从 Libero item library 随机采样 objects, 每个 object 转成结构化表示:
```python
{
  "category": "container",
  "affordances": ["open", "close", "place_in"],
  "spatial_context": "countertop"
}
```

**Stage 2: GPT-o3-mini Task Generation + Decomposition**
把 structured representations 喂给 GPT, 让它:
1. 生成 high-level task description (e.g., "Heat milk in the microwave")
2. 分解成 step-by-step subtask instructions
3. 验证 preconditions, postconditions, object interactions across steps

关键 prompt strategy: **affordance-aware + spatially grounded reasoning**。这避免了 LLM 生成 "magically teleport object" 这种 physically infeasible 的 plan。

**Stage 3: Rule-based Code Generation**
每个 action step 解析成 spatial/relational constraints, 比如:
```
place(milk, from=short_fridge_upper_region, to=coffee_table_top)
```
然后通过 rule-based mapping 转成 simulator-executable code 构造 scene。

### 3.2 Dual-loop Verification

这是 paper 里很聪明的设计:

**Loop 1: Symbolic Simulator Loop**
- Validate object states consistency
- Check relational constraints across steps
- 物理 feasibility 检查

**Loop 2: Vision-Language Verification (GPT-4o)**
- Multi-view RGB-D rendering
- Spatial plausibility evaluation
- Common-sense alignment (比如 "milk 不应该放在 fork 上面")

这种 dual-loop 保证了 physical realisability + semantic plausibility。

### 3.3 Human Demonstration 成本

Table 2 给了 time breakdown:
- Gen. (GPT prompt engineering): 20 hours
- Program (rule definitions): 30 hours  
- Anno. (human trajectories + time annotation): **400 hours**
- Check (human verification): **200 hours**

总共 650 人小时。这反映 long-horizon 任务的高质量数据获取 cost 极高, 也解释了为什么之前 benchmark 不愿意做这件事。

---

## 4. Dataset 统计分析

让我详细读 Fig. 3 的统计:

**Action distribution (Fig. 3b)**:
- 主导 primitives: place, pick, pour
- 稀有 fine-grained actions: turn, return, store
- 共 12 种 action categories

**Compositional diversity (Fig. 3c)**:
- 平均每个 task 涉及 3.5 个 action categories
- **>10% tasks 涉及 ≥5 个 action types**
- 高 compositional complexity

**Trajectory length**:
- 平均 2972.4 simulation steps
- 对比: LIBERO ~500, ALFRED ~几百
- 6× longer

这个 length scale 是关键——意味着 model 必须跨过 thousands of low-level actions 维持一个 high-level plan。

---

## 5. Evaluation Protocol: 四维度 metrics

这是 paper 最 mathematical 的部分, 我详细拆公式。

### 5.1 Task Success Rate (SR)

$$\mathrm{SR}_i = \frac{1}{K_i} \sum_{k=1}^{K_i} \mathbf{1}\left[\psi\left(s_i^{(k)}\right)\right]$$

变量解释:
- $i$: task index, $i \in \{1, 2, ..., N\}$
- $K_i$: task $i$ 的 key object state transitions 数量 (minimal sufficient conditions)
- $s_i^{(k)}$: task $i$ 的第 $k$ 个 key state transition
- $\psi(s)$: simulator-internal predicate function, 返回 True 如果 target condition 在 state $s$ 下成立
- $\mathbf{1}[\cdot]$: indicator function, 条件成立为 1, 否则为 0

直觉: 这不是 binary success/fail, 而是"完成了多少比例的关键 state transitions"。在 long-horizon 任务里, 一个 model 可能完成了 8/10 个 subtasks 但最后一步失败——binary metric 会判 0, 而 SR 给 0.8。这反映了 partial credit 的设计哲学。

### 5.2 Average Plan Match Accuracy (Acc_P)

$$\operatorname{Acc}_P = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\pi_i^{\mathrm{pred}} = \pi_i^{\mathrm{GT}}\right]$$

变量:
- $N$: 总 task 数 (这里是 60 test tasks)
- $\pi_i^{\mathrm{pred}}$: LLM/VLM 预测的高层 plan sequence
- $\pi_i^{\mathrm{GT}}$: 人类标注的 ground-truth plan sequence
- 比较方式: exact sequence matching

注意这里是 **exact match**, 不允许 partial order。这其实挺严格——在 long-horizon 任务里, 一个 model 可能 generate 一个 semantically equivalent 但 ordering 不同的 plan, 会被判为错误。

### 5.3 Plan Efficiency (η)

$$\eta = \frac{\mathrm{SR}}{\mathrm{Len}} = \frac{\mathrm{SR}}{\frac{1}{N} \sum_{i=1}^{N} \left|\mathcal{A}_i\right|}$$

变量:
- $\mathcal{A}_i = [a_1, a_2, ..., a_T]$: task $i$ 的 actual symbolic execution trace
- $a_t$: step $t$ 的 discrete symbolic action
- $|\mathcal{A}_i|$: trace 长度
- $\mathrm{Len}$: 平均 plan length across N tasks

Intuition: 同样的 SR, plan 越短越好。这惩罚 model "绕远路"——比如明明可以直接 pick 却要先把所有 cabinet 都 open 一遍。在 Memory Exploration 里, 这种 metric 特别重要。

### 5.4 Action Completion Accuracy (Acc_C)

$$\operatorname{Acc}_C = \frac{1}{M} \sum_{j=1}^{M} \mathbf{1}\left[\delta(q_j)\right]$$

变量:
- $M$: VideoQA benchmark 里 human-written binary questions 数量
- $q_j$: 第 $j$ 个 binary question
- $\delta(q_j)$: verification function, 返回 1 如果 model 能从 execution 推断出正确答案

这个 metric 主要测 **reflection ability**——model 看着自己的 execution video 能不能回答 "你刚才完成 subtask X 了吗" 这类问题。Table 5 显示 Qwen2.5-VL-7B-SFT 在 Acc_C 上从 47.74 跳到 66.83, 这是 fine-tuning 带来的 reflection 能力提升。

---

## 6. Hierarchical Planning and Execution (HPE) Framework

这是 paper 的 baseline framework, 我画一下 mental model:

```
High-Level Task Instruction
         │
         ▼
   ┌─────────────┐         ┌──────────────┐
   │  VLM (S2)   │ ──────► │  Memory Bank │
   │ Low-freq    │ ◄────── │ (subgoal seq)│
   │ observations│         └──────────────┘
   └─────────────┘                │
         │                        │ active subgoal
         ▼                        ▼
   step-level subgoals     ┌─────────────┐
                          │  VLA (S1)    │
                          │ High-freq    │
                          │ observations │
                          └─────────────┘
                                 │
                                 ▼
                          fine-grained actions
```

### 6.1 Training: Two-stage SFT

**Stage 1: VLA training**
- 数据来源: long-horizon demonstrations 分解成 (image, instruction, action) tuples
- 方法: OpenVLA-style, continuous actions discretize 成 tokens
- Loss: next-token prediction
- 目标: acquire reusable visuomotor primitives

**Stage 2: VLM training**
- 数据来源: video + step-level instructions, 包含 success/failure labels
- 方法: contrastive supervision
- 目标: 学会 associate visual sequences with instruction completion status

### 6.2 Inference: Closed-loop Coordination

执行时的 dynamics:
1. VLM 解析 high-level task → step-level subgoal sequence → 存入 memory bank
2. VLA 持续 query 当前 active subgoal, 基于 high-freq vision 执行 actions
3. VLM 周期性 monitor execution progress
4. 检测到 subgoal 完成 / deviation → update memory 或 refine instruction

这里有个重要的 temporal abstraction: VLM 在低频 (slow) 更新, VLA 在高频 (fast) 执行。这 mimics 了 human 的 "think slowly, act quickly" 模式。

---

## 7. 实验结果: 关键 Insights

### 7.1 Main Results (Table 3)

| Method | Avg | Ran. | Obs. | Exp. | Exe. | Mix | Ideal |
|---|---|---|---|---|---|---|---|
| OpenVLA-Libero100 | 2.00 | 4.59 | 1.35 | 0.18 | 1.86 | 0.00 | 4.05 |
| OpenVLA* | 4.57 | 7.84 | 8.65 | 1.06 | 2.06 | 0.00 | 7.84 |
| Planner+OpenVLA* | 16.04 | 18.63 | 19.45 | 8.04 | 16.69 | 11.48 | 21.92 |
| HPE | 16.55 | 18.63 | 19.18 | 9.06 | 17.83 | 13.21 | 21.10 |
| GT-plan (Table 4) | 25.16 | 26.85 | 30.68 | 19.47 | 23.48 | 19.26 | 31.23 |

关键观察:

1. **System 1 alone 在 long-horizon 彻底失败**: OpenVLA 在 Mix 是 0.00%。即使 fine-tune (OpenVLA*) 也只到 4.57% avg。说明 reactive policy 在长程任务里 fundamentally 撑不住。

2. **System 2 带来 ~3-4x 提升**: Planner+OpenVLA* 跳到 16.04%, HPE 到 16.55%。这是 planning 带来的 pure gain。

3. **GT-plan upper bound 25.16%**: 即使完美 planning, 还是只有 25%。这意味着 **System 1 (VLA) 本身就是 bottleneck**。如果 VLA 自己执行单步 subtask 都有 ~30% 失败率, 那么跨 10 个 subtasks 的 compound success 就只有 $0.7^{10} \approx 2.8\%$。这与实验数据吻合。

4. **HPE 在 Mix 上明显优于 Planner+OpenVLA** (13.21 vs 11.48): 这证明 closed-loop re-planning 在 memory+dynamic 组合场景下是必要的。但在 Ideal 上 HPE 反而略低于 Planner (21.10 vs 21.92), 说明在简单场景下 extra reasoning overhead 反而引入 noise。

### 7.2 Planner Ablation (Table 4)

最 striking 的发现:

| Planner | Avg |
|---|---|
| GT-plan | 25.16 |
| GPT-4o | 16.04 |
| **GPT-4o-Blind** | **15.10** |
| Qwen2.5-VL | 11.19 |
| Qwen2.5-VL-Blind | 11.87 |
| LLaVA-Next-Video | 11.37 |
| LLaVA-Next-Blind | 8.00 |

**GPT-4o-Blind (无视觉) 只比 GPT-4o (full visual) 低 1%**! 这是 paper 里最 surprising 的发现之一。

这意味着:
- 当前 VLM 的 visual grounding 对 long-horizon planning 的 contribution 几乎可以忽略
- Language reasoning 本身已经能解决大部分 planning 问题
- Vision 的真正价值应该在 reflection (Acc_C), 但当前 model 没充分用上

这也解释了为什么 Qwen2.5-VL-Blind (11.87) 反而比 Qwen2.5-VL (11.19) 略高——visual input 在没 fine-tune 的情况下可能是 noise source。

### 7.3 System 2 Multi-dimensional Evaluation (Table 5)

| Model | Acc_P ↑ | Acc_C ↑ | SR ↑ | Len ↓ | η ↑ |
|---|---|---|---|---|---|
| GPT-4o | 68.33 | 32.66 | 16.04 | 10.67 | 1.50 |
| GPT-4o-Blind | 61.37 | 0.00 | 15.10 | 10.73 | 1.41 |
| LLaVA-Next-Video-7B | 40.00 | 37.19 | 11.37 | 8.33 | 1.36 |
| Qwen2.5-VL-7B | 44.67 | 47.74 | 11.19 | 8.30 | 1.34 |
| Qwen2.5-VL-7B-SFT | 30.00 | 66.83 | 9.33 | 6.95 | 1.32 |

非常 interesting 的 trade-off:

1. **Qwen2.5-VL-SFT 的 Acc_C 从 47.74 → 66.83** (reflection 大幅提升)
2. **但同时 Acc_P 从 44.67 → 30.00** (planning 下降!)
3. **SR 反而从 11.19 → 9.33**

这是一个非常重要的 negative result: **fine-tuning on reflection 数据会损害 planning 能力**。可能原因:
- SFT 让 model 过度关注 visual judgment
- 损害了原始 LLM 的 chain-of-thought planning 能力
- 这暗示 System 2 的 multiple cognitive functions 之间有 interference

GPT-4o 的均衡表现 (Acc_P=68.33, Acc_C=32.66, SR=16.04) 说明 raw foundation model 的 reasoning capability 是 most robust 的。

### 7.4 VLA Architecture Independence (Table 6)

对比 OpenVLA 和 π0-fast 作为 System 1:

| Configuration | OpenVLA | π0-fast |
|---|---|---|
| GPT-4o + VLA | 16.04 | 15.15 |
| GT-plan + VLA | 25.16 | 23.04 |
| Qwen2.5-VL + VLA | 11.19 | 13.19 |

VLA 架构变化对 System 2 性能影响很小 (~1-2%)。这证明:
- **System 2 reasoning capability 独立于 System 1 implementation**
- VLM planner 的 bottleneck 不在 VLA 选择
- 改进空间主要在 VLM reasoning 本身

### 7.5 Specialized Embodied VLMs (Table 7)

| Model | Para | Avg | Exp. | Mix |
|---|---|---|---|---|
| LLaVA-N-Blind | 7B | 8.00 | 3.54 | 0.37 |
| Cosmos-Reason1 | 7B | 8.41 | 5.55 | 8.73 |
| VeBrain | 8B | 9.41 | 7.06 | 4.21 |
| Qwen2.5-VL | 7B | 11.19 | 2.63 | 6.67 |
| LLaVA-N-Video | 7B | 11.37 | 1.07 | 3.70 |
| RoboBrain-2.0 | 7B | 11.40 | 9.92 | 7.27 |
| GPT-4o | - | 16.04 | 8.04 | 11.48 |

Insight: Specialized embodied VLMs (RoboBrain-2.0, Cosmos-Reason1, VeBrain) 在 Memory Exploration 和 Mix 上表现更好 (RoboBrain-2.0 Exp.=9.92 vs Qwen2.5-VL 2.63), 但整体 planning 还是落后 GPT-4o。这说明:
- Specialization 提升了 perception-grounded reasoning
- 但 high-level planning generalization 还没解决
- Foundation model 的 broad reasoning capability 依然 most valuable

### 7.6 Memory Tasks Deep Dive (Table 8)

| VLM | SR_Exp. | SR_Exp.-only | η_Exp. | SR_Exe. | Acc_Dec. |
|---|---|---|---|---|---|
| Qwen2.5-VL | 3.54 | 50.0 | 0.17 | 12.39 | 10.0 |
| GPT-4o | 9.06 | 80.0 | 0.32 | 17.83 | 30.0 |

公式 (5) 和 (6) 用来计算 exploration efficiency:

$$\mathrm{Comp}_{\mathrm{Exp.}} = \frac{|\pi_G \cap \pi_{GT}|}{|\pi_{GT}|}$$

$$\eta_{\mathrm{Exp.}} = \frac{1}{N} \sum_{i=1}^{N} \frac{\mathrm{Comp}_{\mathrm{Exp.}}}{|\pi_G|}$$

变量:
- $\pi_G$: predicted exploration plan
- $\pi_{GT}$: ground truth exploration plan
- $\mathrm{Comp}_{\mathrm{Exp.}}$: completeness (predicted plan 覆盖了多少 GT 必要 steps)
- $|\pi_G|$: predicted plan 长度

Intuition: 既要求覆盖度高 (completeness), 又要求 plan 简洁 (短 length)。一个 model 如果盲目 open 所有 cabinet, completeness 可能高但 length 也高, efficiency 反而低。

GPT-4o vs Qwen2.5-VL 在 memory tasks 上差距巨大:
- **Exploration-only**: 80% vs 50% (找 object 的能力)
- **Decision Accuracy**: 30% vs 10% (3x gap!)
- **SR_Exp.**: 9.06 vs 3.54 (找到 + 完成整体)

这揭示了 memory task 的两个 bottleneck: (1) exploration 的 systematic 性, (2) decision 的 grounded reasoning。

---

## 8. Qualitative Case Studies Insights

### 8.1 Planning Quality (Fig. 14, 15)

GPT 的 plan 不仅 step 正确, 还包括 contextual spatial cues 和 object restoration (比如 "place wine bottle back to far-left side")。Qwen 倾向 omit spatial descriptors, LLaVA 直接 generate wrong actions (把 mug 从错误位置移到错误 tray)。

### 8.2 Sub-plan Decomposition (Fig. 16)

GPT 在找 butter 任务里 **systematic top-down 探索所有 compartments**, 包含 open + close actions。Qwen 直接 assume butter 在第一个 compartment, 缺 fallback。这反映了 hierarchical reasoning 的差距。

### 8.3 Memory Task Completion Awareness (Fig. 17, 18)

GPT 能在 visual evidence 出现时正确 update task completion status。Qwen 即使看到 butter 也 "denial", 在 empty cabinet 场景下 fail to infer "absence as valid evidence"。

这暴露了当前 VLMs 的一个 fundamental limitation: **negative inference** (从 "没看到 X" 推断 "X 不存在")。这本质上是 model 缺乏 epistemic state tracking。

---

## 9. Intuition Building: 这篇 paper 的 meta-takeaways

让我站在 Karpathy 视角总结几个深层 insight:

### 9.1 System 1 / System 2 在 Robotics 的真实 separation

这篇 paper 实证证明了: 在 long-horizon robotic manipulation 里, System 1 和 System 2 的 separation 是 **necessary, not optional**。纯 VLA 在 >2000 步 trajectory 上根本维持不了 instruction fidelity。Hierarchical decomposition 不是工程方便, 是 cognitive necessity。

### 9.2 VLM 的 visual input 价值被高估 (现阶段)

GPT-4o-Blind ≈ GPT-4o 这个结果非常 striking。它说明:
- 当前 VLM 的 visual grounding 能力还没准备好 support long-horizon planning
- Language reasoning 是 main driver
- Vision 的真正价值在 reflection (Acc_C), 但需要 task-specific training

这暗示未来 VLM 的 R&D 应该 focus on **更好的 visual-temporal grounding**, 而不是简单堆 visual encoder。

### 9.3 GT-plan 25% 上限暴露 VLA bottleneck

即使完美 planning, SR 也只有 25%。这意味着 **System 1 本身就是 bottleneck**。Long-horizon SR ≈ $\prod_{k=1}^{K} p_k$, 如果 single subtask success 是 0.85, 10 个 subtasks 就是 $0.85^{10} \approx 20\%$。这解释了为什么 HPE 的 SR 在 Ideal 是 21%, 跟理论 compound success 吻合。

要突破这个 ceiling, 必须:
- 改进 single-step VLA 精度
- 引入 error recovery mechanism
- 让 System 2 能 detect System 1 failure 并 re-plan

### 9.4 Fine-tuning 的 trade-off

Qwen2.5-VL-SFT 提升了 Acc_C 但降低了 Acc_P 和 SR。这揭示了一个深刻问题: **当前 SFT 是 zero-sum 的**。Reflection 和 Planning 在 model capacity 上竞争。

Future direction 应该探索:
- Multi-task SFT with balanced objectives
- LoRA-based module separation
- Curriculum learning that preserves base capability

### 9.5 Specialization vs Generalization

RoboBrain-2.0、Cosmos-Reason1 这些 specialized embodied VLMs 在 specific tasks 上更强, 但 overall planning 落后 GPT-4o。这印证了: **broad reasoning capability > task-specific specialization**, 至少在 long-horizon planning 这个 domain。

---

## 10. Limitations 和 Future Directions

Paper 自己承认的 limitations:
1. System 1 ↔ System 2 的 bidirectional communication 有限
2. Evaluation protocol 可以扩展到 subtask ordering 和 failure recovery
3. 真实世界 deployment 未验证

我补充几个 critical 的:
1. **Simulation-only**: 虽然 long-horizon focus 减轻了 sim-to-real concern, 但 real-world visual complexity 是另一回事
2. **GPT-4o as System 2**: 用 closed-source model 作为 benchmark baseline 有 reproducibility 问题
3. **Anchor points 设计**: Sec 5.1 提到 "anchor points" decouple step-switching from model。这可能 hide 了 model 自己 decide when to switch 的能力评估
4. **Memory mechanism 简化**: 当前 memory bank 只是 subgoal sequence。真正的 episodic memory、spatial memory、semantic memory 没区分

---

## 11. 相关 work 的延伸阅读

如果你想 build deeper intuition, 推荐这些 related works:

**Hierarchical Robotics Control**:
- [Hi Robot](https://arxiv.org/abs/2502.19417) - hierarchical VLA
- [Bumble](https://arxiv.org/abs/2410.06237) - building-wide mobile manipulation
- [SayCan](https://arxiv.org/abs/2204.01691) - early LLM planning for robotics

**VLA Models**:
- [OpenVLA](https://openvla.github.io/) - open-source VLA
- [RT-2](https://robotics-transformer2.github.io/) - Google's VLA
- [π0](https://www.physicalintelligence.company/blog/pi0) - Physical Intelligence's flow matching VLA
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - diffusion-based policies

**Long-horizon Benchmarks**:
- [LIBERO](https://liberoproject.github.io/)
- [CALVIN](https://calvinrobot.github.io/)
- [RoboCasa](https://robocasa.ai/)
- [ALFRED](https://askforalfred.com/)
- [VLABench](https://github.com/VLABench/VLABench)

**System 1 / System 2 in AI**:
- [Kahneman's "Thinking, Fast and Slow"](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- [Sakana AI's paper on System 1/2 in LLMs](https://arxiv.org/abs/2507.01917)
- [OpenAI o1 system card](https://openai.com/index/learning-to-reason-with-llms/) - System 2 in language models

**Embodied VLMs**:
- [RoboBrain](https://github.com/RoboBrain-Foundation/RoboBrain) - specialized embodied VLM
- [Cosmos-Reason1](https://build.nvidia.com/nvidia/cosmos-reason1-7b) - NVIDIA's embodied reasoning model
- [VeBrain](https://github.com/Tianxing-Chen/VeBrain) - vision-language model for robotics

---

## 12. 我的 final thoughts

这篇 paper 最 valuable 的贡献不是 HPE framework (那个比较 simple), 而是 **benchmark 本身**。它把 long-horizon + dynamic + memory 三个维度 systematic 地 codify 进了 evaluation protocol。

最 striking 的发现是: **当前 SOTA VLMs 在 long-horizon robotic planning 上还很差**。GPT-4o 只有 16% SR, 距离 GT-plan 的 25% 还有 9% gap, 距离 practical deployment 远得不能再远。

更深刻的是 GPT-4o-Blind ≈ GPT-4o 的发现——这说明我们当前的 "VLM" 在 robotics 场景下其实主要 work as "LM"。Vision 的 grounding 能力还需要 fundamental breakthrough。

如果让我预测 next steps:
1. **Visual-temporal grounding 的 breakthrough**: 当前 VLM 看视频还是 frame-by-frame, 真正的 temporal reasoning 模型还没出现
2. **Episodic memory architectures**: 单纯 subgoal queue 不够, 需要 vector-DB-style 的 episodic memory
3. **Self-correction via System 1 feedback**: System 2 应该能从 System 1 的 execution failure 学到东西
4. **Reasoning about hidden state**: 这是真正的 partial observability reasoning, 当前 model 基本做不到

这篇 paper 给了 community 一个清晰的 measuring stick。下一步就是 race to improve。
