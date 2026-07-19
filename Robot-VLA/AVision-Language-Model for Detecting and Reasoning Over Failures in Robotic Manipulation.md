---
source_pdf: AVision-Language-Model for Detecting and Reasoning Over Failures in Robotic
  Manipulation.pdf
paper_sha256: d647c7bff1ccb2d0ae805bab761177aba8cd50d81d0418ca174e9ae97054791b
processed_at: '2026-07-18T12:52:10-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AHA: 一个用于检测和推理 Robotic Manipulation Failures 的 Vision-Language Model

非常好的 paper，让我从多个 angle 深入讲解一下。这篇 paper 是 NVIDIA + UW + MIT + AI2 团队的工作，核心 contribution 集中在一个被忽视的关键问题上：**如何让 VLMs 真正"看懂"机器人执行过程中的 failure，并且用自然语言解释出来**。下面我按照 build intuition 的顺序展开。

---

## 1. Core Motivation: 为什么需要 Failure Reasoning？

当前 robotics 社区用 LLM/VLM 做 manipulation 大致分两个流派：

1. **Promptable 方式**：VoxPoser (https://arxiv.org/abs/2307.05973)、CoPa (https://arxiv.org/abs/2403.08248)、ReKep (https://arxiv.org/abs/2409.01652) — 给 visual prompt，让模型直接生成 low-level action
2. **Instruction-tuning 方式**：RoboPoint (https://arxiv.org/abs/2406.10721)、Octopi (https://arxiv.org/abs/2405.02794)、RT-2 (https://arxiv.org/abs/2307.15818) — fine-tune VLM 服务于特定任务

这两个流派都擅长 task execution，但都有一个盲点：**failure 时怎么办？** 比如机器人把 cube 抓到一半掉地上，GPT-4o 大概率只会说 "the robot is performing the task" 这种正确的废话。Human 会立刻识别"啊，掉下来了"，并给出原因"gripper 没有闭合到位"。

这篇 paper 把这个 gap 抓得很准：**failure detection 不应该是个 binary classification 问题，而是 free-form reasoning 问题**。这个 reformulation 是整篇 paper 的灵魂。

类比一下：就像 RLHF (https://arxiv.org/abs/2203.02155) 之于 LLM alignment，没有 human feedback，模型只会 next-token-predict；有了 failure reasoning，robotics 的 closed-loop 才真正完整。

---

## 2. Failure Taxonomy: 7 个 failure modes 的设计

这是 paper 里我认为最值得深入思考的部分。作者从 DROID dataset (https://arxiv.org/abs/2403.12945) 和 Open-X Embodiment (https://arxiv.org/abs/2310.08864) 中分析 teleop 和 autonomous policy rollout，结合 REFLECT (https://arxiv.org/abs/2306.15724) 的 prior work，归纳出 7 类失败：

### Object-centric failures（3类）：

| Failure Mode | 触发条件 | Sim 实现方式 |
|---|---|---|
| **No_Grasp** (Incomplete Grasp) | Gripper 到达 grasp pose 但没 close | 在相关 keyframe 省略 gripper open/close command |
| **Slip** (Inadequate Grip Retention) | 抓到 object 后，移动过程中 grip 松掉 | 在 grasp 之后 timed release gripper |
| **Wrong_object** (Wrong Target Object) | 操作了错误物体（"pick red cup" 但抓了 green cup） | 把 keyframe 重新 assign 到另一个 object，保持相对 pose |

### Action-centric failures（4类）：

| Failure Mode | 触发条件 | Sim 实现 |
|---|---|---|
| **Translation** (Misaligned keyframe) | X/Y/Z 轴有 translation offset 导致失败 | perturb keyframe 位置 |
| **Rotation** (Incorrect Rotation) | roll/yaw/pitch 有 offset | perturb keyframe 朝向 |
| **No_Rotation** (Missing Rotation) | 平移到位但没旋转 | constrain keyframe 旋转轴 |
| **Wrong_action** (Wrong Action Sequence) | 动作顺序错误（先 push cube 再开 drawer） | reorder keyframe activations |

**Intuition 这里很关键**：这个 taxonomy 不是随意划分的，而是基于 RLBench (https://arxiv.org/abs/1909.12271) 的 **keyframe-based formulation** 自然诞生的。RLBench 原生把 task 拆成 keyframes，每个 keyframe 是一个 waypoint。这给了 FailGen 一个"调控点" — 你可以精准地 perturb 某一个 keyframe 来诱导特定 failure。

这让我想到一个更深层的问题：**这个 taxonomy 是 exhaustive 的吗？** 显然不完全是 — paper 在 limitations 里也承认了。比如 dynamic failures（物体被外力撞击飞出）、perceptual failures（occlusion 导致看不到 object）、grasp force 不够导致 crush object 等。但作为一个 v1.0 的 taxonomy，它已经覆盖了 manipulation policy rollout 中 80%+ 的常见 case。

---

## 3. FailGen Pipeline: Simulated Data 的可扩展性

### 3.1 为什么用 Sim 而不是 Real？

最直觉的想法是 — 我们为什么不直接从 DROID、Open-X 这些 real dataset 里捞 failure 标注？答案有三层：

1. **Real failure 数据极其稀疏**：teleop 数据大部分是成功的，autonomous policy 失败样本虽然多，但分布长尾，难以系统化覆盖
2. **标注成本**：每条 failure trajectory 需要 human 写自然语言解释，cost 太高
3. **Coverage 不够**：real 数据的 failure mode 是 emergent 的，你无法控制分布

FailGen 的核心 trick：**用 simulation 的 success condition checker 做自动 labeling**。你在 RLBench 里 perturb 一个 keyframe，sim 会自动告诉你 task 是否 success。如果 fail，FailGen 同时记录了是哪个 keyframe、哪种 perturbation 导致的，自然语言 explanation 可以从 template 自动生成。

### 3.2 数据生成细节

- **49K+ image-query pairs across 79 RLBench tasks**
- 每个 task 系统地 sweep 所有 keyframes × 所有 7 种 failure modes 的所有 valid configuration
- 自动生成 YAML config，包含 failure mode、parameters (distance、sequence、gripper retention strength)、对应 keyframe
- Language template 描述 robot 在两个连续 keyframe 之间的动作

**关键 insight**：FailGen 是一个 environment wrapper，可以套到任意 simulator 上。Paper 里展示了：
- RLBench → AHA training data
- ManiSkill (https://arxiv.org/abs/2107.14483) → ManiSkill-Fail eval set
- RoboFail (来自 REFLECT 的 real-world UR5 数据) → cross-embodiment eval

这种"simulator-agnostic"的设计非常有工程价值，让我想到 MimicGen (https://arxiv.org/abs/2310.17596) 的 trajectory adaptation 思路，但 MimicGen 是生成 success demo，FailGen 是生成 failure demo — 一个问题的两端。

---

## 4. Input Formulation: 关键frame 轨迹图像的构造

这部分是 paper 里技术上最容易被忽视但很重要的细节。让我详细解释公式。

### 4.1 输入图像矩阵 I

Paper 把输入图像构造为一个矩阵 **I**，其中：

- **行索引** 对应不同的 camera viewpoint：$\{V_0, V_1, \ldots, V_n\}$
  - $V_i$ 表示第 $i$ 个视角的 camera，比如 front camera、wrist camera、side camera
  - $n$ 是总 camera 数减 1

- **列索引** 对应 temporal sequence of keyframes：$\{S_0, S_1, S_2, \ldots, S_t\}$
  - $S_j$ 表示第 $j$ 个 sub-task（由两个连续 keyframe 定义）
  - $t$ 是当前 sub-task index

所以矩阵 **I** 的元素：
$$I_{i,j} = \text{image captured by camera } V_i \text{ at keyframe } S_j$$

**细节**：
- 时间序列从左到右排列
- 如果当前 sub-task 是 $S_t$，那么 $S_{t+1}, S_{t+2}, \ldots$ 位置用 **white image patches** 填充
- 所有 camera viewpoint 沿着行方向 concatenate

这样构造的好处：
1. **Temporal reasoning**：模型可以 "看" 整个 rollout 历史，理解 trajectory dynamics
2. **Multi-view fusion**：解决 occlusion 问题，front camera 看不到的细节 wrist camera 可能看到
3. **固定 shape**：无论 task 长短、camera 多少，可以统一 pad 成固定 size 输入 VLM

### 4.2 Prompt 模板

```
For the given sub-tasks, first determine it has succeed by choosing from ["yes", "no"] 
and then explain the reason why the current sub-tasks has failed.
```

注意：这里的 "succeed" 是 paper 原文 typo（应该是 "succeeded"），但 prompt 设计很精妙：
- 强制 binary decision 先做（避免模型含糊其辞）
- 然后必须给 explanation（这才是 paper 的核心 contribution）

---

## 5. Model Architecture & Training

### 5.1 架构解析

基于 LLaVA-v1.5 (https://arxiv.org/abs/2304.08485) 的标准架构：

```
Image Input
   ↓
[Image Encoder: CLIP ViT-L/14] ← FROZEN (pre-trained weights kept)
   ↓
[2-Layer Linear Projector] ← TRAINABLE (project image tokens → text token space)
   ↓
Language Tokenizer ← FROZEN
   ↓
[Multimodal Tokens Concatenation]
   ↓
[LLM: LLaMA-2-13B] ← TRAINABLE
   ↓
Autoregressive Output (Yes/No + Explanation)
```

**关键设计 decision**：
- **Image encoder frozen**：保留 CLIP 的 visual representation
- **Projector + LLM trainable**：让 LLM 学会 "interpret" image tokens 在 robotics failure context 下的含义
- 类似 RoboPoint 的做法 — co-finetune 保留 general knowledge

### 5.2 Co-finetuning 策略

Table 1 显示了 training data 的组成：

| Data Source | Quantity | Purpose |
|---|---|---|
| AHA dataset (Train) | 49K | Domain-specific failure reasoning |
| VQA dataset (from LLaVA) | 665K | 保留 general visual QA 能力 |
| LVIS (https://arxiv.org/abs/1908.03195) | 100K | 保留 object detection / grounding 能力 |

为什么 co-finetune 重要？如果不加 VQA + LVIS，模型会 **catastrophically forget** 通用视觉理解能力，变成只能描述 robot failure 的 narrow model。Table 3 显示 AHA-13B 在 MMBench、ScienceQA、TextVQA、POPE、VizWiz 上和 LLaVA-13B 几乎持平（平均 1.5% margin）— 说明 co-finetune 起作用了。

---

## 6. Experimental Results: 关键数字解读

### 6.1 主实验 Table 2 分析

Evaluation 设置：
- **3 datasets**：
  1. AHA dataset (Test) — 11K pairs，10 个 unseen RLBench tasks
  2. ManiSkill-Fail — 130 pairs，4 tasks（不同 simulator、不同 viewpoint）
  3. RoboFail — real-world UR5 robot failures，7 tasks

- **4 metrics**：
  1. **ROUGE-L**：基于 longest common subsequence 测文本质量
  2. **Cosine Similarity**：embedding 空间的语义相似度
  3. **Binary Success (%)**：纯二分类 accuracy
  4. **LLM Fuzzy Match**：用 Claude-3-Sonnet 做 teacher-student prompting 评判语义相似

关键数据点：

| Model | AHA Test (avg) | ManiSkill-Fail (avg) | RoboFail (avg) |
|---|---|---|---|
| LLaVA-v1.5-13B | 0.249 | 0.125 | 0.152 |
| GPT-4o (zero-shot) | 0.491 | 0.278 | 0.273 |
| GPT-4o-ICL (5-shot) | 0.498 | 0.599 | 0.547 |
| AHA-7B | 0.673 | 0.653 | 0.490 |
| **AHA-13B** | **0.693** | **0.704** | **0.526** |

**几个值得注意的细节**：
1. GPT-4o 在 AHA Test 上表现还行（因为 RLBench 风格训练分布），但 ManiSkill 和 RoboFail 上急剧退化 — 说明 general VLM 对 cross-embodiment、cross-simulator 的泛化很弱
2. AHA-7B 已经超过 GPT-4o-ICL，AHA-13B 进一步拉开
3. RoboFail 上 AHA 优势最小（仅 +4.9% vs GPT-4o-ICL），说明 sim → real 仍然是最大挑战

### 6.2 Scaling Law (Figure 3 Left)

Paper 测试了 [3K, 6K, 12K, 34K, 48K, 60K] 不同数据量级的影响，在 ManiSkill-Fail 上测得 **quadratic fit gradient ≈ 0.0022**。

这个 quadratic 拟合的含义：性能提升随数据量呈**幂律缓慢增长**（concave up 但增长放缓），暗示继续 scale up AHA dataset 边际收益递减但仍然 positive。这跟 LLM 的 Chinchilla scaling law (https://arxiv.org/abs/2203.15556) 类似 — 但 robotics 的 scaling 比 NLP 慢，可能因为 visual diversity 的 bottleneck。

---

## 7. Downstream Applications: AHA 作为 "Failure Critic"

这是 paper 最有应用价值的部分。AHA 被嵌入到三个不同 paradigm 的 robotics framework 中：

### 7.1 Eureka Reward Generation（+22.34%）

Eureka (https://arxiv.org/abs/2310.12931, https://eureka-research.github.io/) 是 NVIDIA 自己的工作 — 用 LLM 自动生成 dense reward function code for RL。原始 Eureka 的 reflection 只看 policy training statistics（成功率、episode length 等）。

AHA 的集成方式：
- 在 Eureka 的 reflection loop 里，加上 **policy rollout 时的 visual failure explanation**
- 比如 AHA 输出："The robot gripper rotated with an angle offset"，Eureka 反馈给 LLM，让它调整 reward function 中 rotation penalty 项

5 个 ManiSkill task 上，AHA 比 GPT-4o 反馈提升 22.34% success rate。这个数字很可观，因为 dense reward 的微小调整在 RL 中会放大。

### 7.2 PRoC3S TAMP（+36.7%）

PRoC3S (https://arxiv.org/abs/2406.05572) 是 MIT 的 task and motion planning work — 用 LLM 生成 Language-Model Program (LMP)，然后在 sim 中测试大量 plan 找 valid plan。

AHA 集成方式（双向）：
1. **Failure explanation feedback**：sim 中失败的 plan，AHA 看可视化后给解释，feed back 给 PRoC3S 的 LLM
2. **Plan verification**：PRoC3S 找到 "valid" plan 后，AHA 再 verify 是否真正 achieve goal

10 trials × 3 tasks，100 sampling steps 上限，AHA 比 GPT-4o 提升 36.7%。这是三个 downstream 中提升最大的 — 因为 TAMP 的瓶颈就在于"什么叫真正失败"，传统 checker 只能检测 IK、collision 等 geometric failure，semantic failure（比如 drawer 开了但物体没放进去）检测不到。

### 7.3 Manipulate-Anything Sub-task Verification（+5%）

Manipulate-Anything (https://arxiv.org/abs/2406.18915) 是同一团队之前的 zero-shot manipulation framework，原本用 GPT-4V 做 sub-task verification。AHA 替换该模块，4 个 RLBench task，每个 25 episodes，平均 +5%。

提升相对小，因为 Manipulate-Anything 已经用了 GPT-4V，AHA 优势主要在 fine-tune 过的 robot-specific reasoning。

**平均 +21.4%** 跨三个任务，paper 强调比 GPT-4 系列提升显著。

---

## 8. 关键 Insights & 联想

### 8.1 为什么 sim-trained 能 generalize 到 real？

这是 paper 最 surprising 的结果。AHA 完全在 RLBench 训练，但 RoboFail 上还能 +4.9% over GPT-4o-ICL。我的理解：

1. **Failure mode 是 abstract 的**：一个 "Slip" failure 在 RLBench 和 real-world 视觉上差异巨大，但语义上都是 "object 被抓起后掉落"。AHA 学的是这个**抽象语义**，不是低层 pixel pattern。
2. **Multi-view + temporal input format**：让模型关注的是 trajectory dynamic 而非单帧 photorealism
3. **LLaVA base 的 strong prior**：CLIP 的 visual representation 本身已经对 real-world image 有不错的 robustness

这暗示一个更深的结论：**robotics foundation model 的 generalization 主要瓶颈不在 visual realism，而在 semantic grounding 的 quality**。

### 8.2 与 LLaVA-style VLM 的关系

AHA 选择了 fine-tune LLaVA-v1.5-13B，而不是用更强的 base（比如 LLaVA-NeXT-34B）。这有个 trade-off：
- **Pro**：13B 可以单卡 fine-tune，open-source 友好
- **Con**：base model 的 visual reasoning 上限有限

Table 2 显示 LLaVA-NeXT-34B 作为 base 比 LLaVA-1.5-13B 强很多（RoboFail 上 0.188 vs 0.203 cosine sim），但 fine-tune 后 AHA-13B 反超。这暗示 **fine-tune 的 data quality > base model size**，至少在这个 task 上。

### 8.3 与 RT-2、OpenVLA 等 VLA 的关系

一个自然的问题：AHA 和 Vision-Language-Action (VLA) model 比如 RT-2、OpenVLA (https://arxiv.org/abs/2406.09246) 是什么关系？

我的理解：
- VLA 输出 **action**，AHA 输出 **failure reasoning**
- 两者是**互补**的：VLA 是 actor，AHA 是 critic
- 完全可以想象一个 closed-loop 系统：VLA 执行 → AHA 检测失败 → 把 failure explanation feed back 给 VLA 重新规划

这跟 RLHF 中 actor + reward model 的关系类似 — 但 reward model 是 scalar，AHA 是 language feedback，更接近 RLHF with detailed critique (类似 Self-Refine, https://arxiv.org/abs/2303.17651)。

### 8.4 与 REFLECT 的关系

REFLECT (https://arxiv.org/abs/2306.15724) 是 Stanford 的工作，用 LLM summarize robot failure。AHA 和 REFLECT 的关键区别：

| Aspect | REFLECT | AHA |
|---|---|---|
| 输入 | Multi-modal summary | Raw image (multi-view + temporal) |
| 模型 | Off-the-shelf LLM | Fine-tuned VLM |
| 输入形式 | 文字描述状态 | 直接看图 |
| Failure taxonomy | 7 类（AHA 借鉴） | 7 类（相同） |

AHA 实际上把 REFLECT 的 failure taxonomy 拿过来用了，但改了输入模态 — 从"LLM 看 summary"变成 "VLM 直接看图"。这是从 symbolic perception 到 visual perception 的演进。

### 8.5 Failure Taxonomy 的局限与扩展

Paper limitations 里提到了，但我觉得可以更展开。**这 7 类 failure 都假设 single-arm manipulation + rigid object + tabletop scene**。如果扩展到：

- **Bimanual manipulation**：两只手协调失败（一只手抓歪了导致另一只手接不到）
- **Deformable object**：cloth folding 失败（皱褶位置不对）
- **Articulated object**：drawer 没拉到位、door 没关严
- **Tool use**：hammer 没砸到 nail、screwdriver 滑了
- **Contact-rich**：assembly 任务中 parts 没对齐

这些都会引入**全新的 failure mode**。未来 work 可以：
1. 扩展 taxonomy（可能 12-15 类）
2. 在更多 simulator（Isaac Lab、SAPIEN）上跑 FailGen
3. 加入 physics-based failure（force 不足导致 crush）

---

## 9. 我的批判性思考

### 9.1 优点

1. **Problem formulation 非常准确** — 从 binary 到 free-form reasoning 这个 reformulation 抓得很准
2. **Data pipeline 可复现** — FailGen 是 open-source 的，其他人可以在自己 simulator 上复用
3. **Downstream integration 多样** — 三个不同 paradigm 都验证了，说明 AHA 不是 one-trick pony
4. **Co-finetune 保留通用能力** — 这个细节容易做错，paper 做对了

### 9.2 缺点 / 改进空间

1. **RoboFail 上提升只有 4.9%** — sim-to-real 仍然是大头。可以引入 domain randomization 或 real failure 数据的 co-finetune
2. **7 类 failure taxonomy 不够 exhaustive** — paper 自己承认了，但 v2 应该扩展
3. **Image matrix I 的设计有冗余** — 把所有 keyframe 拼起来，如果 task 有 20 个 keyframe，图像会非常 wide。可以换成 attention-based temporal aggregation
4. **没有对比 VLA-based failure detection** — 比如 RT-2 内部已经有 failure 表示，AHA 没和它对比
5. **Evaluation 中 LLM Fuzzy Match 用 Claude-3-Sonnet** — 这个 metric 引入了 Claude 的 bias，比较 open-source model 时可能不公平
6. **缺少 long-horizon task 测试** — RoboFail 是 short task，AHA 在长 horizon（>10 sub-tasks）任务上表现未知

### 9.3 未来方向联想

- **Active learning with AHA**：让 AHA 自己识别"我不确定"的 failure case，主动 query human annotation
- **AHA + Diffusion Policy**：AHA 检测到 failure 后，可以条件化 diffusion policy 重新生成修正 trajectory
- **AHA as reward signal**：把 AHA 的 binary success 当 dense reward，直接 train RL policy（比 Eureka code generation 更 end-to-end）
- **Multi-agent AHA**：多个 AHA 实例互相辩论 failure 原因，类似 Self-Consistency
- **AHA for Human-Robot Interaction**：robot 能向 human 解释"我为什么失败了"，提升 trust（HRI 经典问题，https://arxiv.org/abs/2303.16010）
- **Video-based AHA**：现在用 keyframe image，未来用 video input（比如 Video-LLaVA, https://arxiv.org/abs/2311.10122）可能捕捉更细的 dynamics

---

## 10. 总结

AHA 这篇 paper 在一个被忽视的问题上做了扎实的工作。三个核心 contribution：

1. **FailGen** — 一个 simulator-agnostic 的 failure data generation pipeline，这个 tool 单独看就已经很有价值
2. **AHA** — 第一个针对 robotic failure reasoning 的 open-source fine-tuned VLM，在多个 OOD dataset 上 beat GPT-4o
3. **Downstream integration** — 三个不同 paradigm 都验证了 failure reasoning feedback 的价值

更深层的 takeaway：**robotics foundation model 的下一个 frontier 是"知道什么时候错了"**。当前 community 大量精力在 "execution"，但 closed-loop 系统需要 "execution + failure detection + correction"。AHA 补齐了中间这一环。

Reference link：
- Project page: https://aha-vlm.github.io
- LLaVA: https://arxiv.org/abs/2304.08485
- Eureka: https://arxiv.org/abs/2310.12931
- RoboPoint: https://arxiv.org/abs/2406.10721
- REFLECT: https://arxiv.org/abs/2306.15724
- PRoC3S: https://arxiv.org/abs/2406.05572
- Manipulate-Anything: https://arxiv.org/abs/2406.18915
- RLBench: https://arxiv.org/abs/1909.12271
- ManiSkill: https://arxiv.org/abs/2107.14483
- MimicGen: https://arxiv.org/abs/2310.17596
- DROID: https://arxiv.org/abs/2403.12945
- Open-X Embodiment: https://arxiv.org/abs/2310.08864
- Octopi: https://arxiv.org/abs/2405.02794

如果你想深入某个部分（比如 FailGen 的具体 YAML schema、Eureka 反馈机制的具体 prompt、AHA 在 PRoC3S 里的 replan 逻辑），我可以再展开。
