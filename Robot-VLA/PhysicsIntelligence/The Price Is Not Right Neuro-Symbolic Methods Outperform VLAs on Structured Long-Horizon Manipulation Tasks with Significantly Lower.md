---
source_pdf: The Price Is Not Right Neuro-Symbolic Methods Outperform VLAs on Structured
  Long-Horizon Manipulation Tasks with Significantly Lower.pdf
paper_sha256: 05d73acb3ec3619cda8608fd86b23febdd8aeed8a9f8642c34055b9d72dfbd92
processed_at: '2026-08-12T14:48:40-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

两个团队比赛搭积木，一个用大模型硬学，一个用"小脑+大脑"分层做——结果分层那个不仅赢得多，用电量还少了80倍。

---

## 这篇paper到底在吵什么

最近robotics圈有个很火的narrative：**VLA（Vision-Language-Action）大模型就是robotics的未来**。π₀、OpenVLA、GR00T这些model，参数量大、数据量大、pretraining猛，号称能做generalist robot policy。整个community有种"scale is all you need"的乐观——再大一点、再多数据一点、就什么都能干了。

但这篇paper跳出来说：**等一下，你们有没有算过price？**

作者选了一个非常structured的任务——Towers of Hanoi（汉诺塔），让最hot的VLA（π₀）和一个neuro-symbolic model（NSM）head-to-head比，同时测两个东西：
1. Task success rate
2. Energy consumption

结果很striking：NSM在task上碾压VLA，在energy上更是碾压到几乎两个数量级。

---

## 为什么选汉诺塔

汉诺塔这个任务选得非常聪明。它有几个特殊性质：

**它是perfectly structured的**。规则就一句话：大块不能压小块。State是discrete的（哪个block在哪个位置），transitions是deterministic的。3个block要7步最优解，4个block要15步——步数随block数指数增长。

**它有long-horizon compositional structure**。你学会"pick A place on B"这个primitive，理论上就能组装出任意n-block的solution。如果model真的学到了rule，给它4个block它也能解；如果只是memorize trajectory，给4个block它就崩。

**它刚好是symbolic methods的sweet spot**。classical planning几十年来就在解这类问题，PDDL planner几毫秒就能算出optimal plan。而end-to-end neural model要从pixel + language直接学到"大不压小"这个rule，相当于让model从零rediscover symbolic reasoning。

所以这个任务的设计本身就是个**biased benchmark**——它倾向于structured methods。paper作者也承认这点。但他们的argument是：**很多real-world manipulation task（工业装配、rule-based operation）其实就是structured的**，所以这个comparison有practical relevance。

---

## 两个选手

### 选手A：π₀ VLA

π₀是Physical Intelligence 2024年底放出来的VLA model [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)。架构上：
- PaliGemma 2B做vision-language backbone
- Gemma 300M做action header
- 用flow matching替代diffusion做action generation
- 一次output一段action trajectory（action chunking），不是step-by-step

作者fine-tune了两个版本：
- **E2E-VLA**：command永远是"Play Towers of Hanoi"，model自己infer下一步干什么
- **PG-VLA**：external planner给subtask commands（"Pick the blue block", "Place on red block"...），VLA只负责执行

训练数据：300个完整Hanoi trajectories（包括3-block和随机配置）。

### 选手B：NSM（Neuro-Symbolic Model）

来自Lorang et al.的prior work [arXiv:2508.21501](http://arxiv.org/abs/2508.21501)。架构上是**两层**：

**高层（大脑）**：classical symbolic planner
- 从demonstrations自动learn PDDL domain（不用手写）
- 用Metric-FF planner [JAIR 2003](https://www.jair.org/index.php/jair/article/view/10249)算optimal plan
- 输出operator序列

**低层（小脑）**：diffusion policies
- 每个operator对应一个diffusion policy
- 用relative pose observation（object pose w.r.t. end-effector）
- 训练loss是标准diffusion BC：$$\mathcal{L}(\pi) = \frac{1}{T} \sum_{t=0}^{T} \| \pi(\tilde{s}_t) - a_t \|^2$$
  - $T$是trajectory长度
  - $\tilde{s}_t$是continuous state（这里是relative pose）
  - $a_t$是expert action

**关键细节**：NSM训练时**只见过50个single-step stacking demos**（pick-and-place pairs），**从未见过完整Hanoi solution**。它从这些primitives里infer出PDDL domain，然后用classical planner自己solve Hanoi。

这就像：VLA看了300场完整象棋对局想学下棋，NSM只学了"车怎么走、马怎么走"，然后自己用规则引擎下整盘棋。

---

## 结果：一边倒

### Task performance

| Task | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| 单步pick-place | 87% | 59.6% | **99%** |
| 3-block Hanoi | 34% | **0%** | **95%** |
| 4-block Hanoi（没见过） | 0% | 0% | **78%** |

**单步任务**：NSM几乎完美，VLA还行。这个差距主要是low-level control的精度——NSM用relative pose + diffusion policy，VLA从pixel直接学。

**3-block Hanoi**：这是key result。E2E-VLA只有34%——它训练时见过这个exact task配置，仍然只能做对1/3。PG-VLA是**0%**——即使给了正确subtask commands，仍然完全失败。

PG-VLA的0%特别值得思考。它单步成功率59.6%，看起来不低，但14个subtask累积下来 $0.596^{14} \approx 0.0004$——几乎必然失败。这是long-horizon task的**exponential decay problem**：每步必须接近完美，否则指数崩溃。

NSM的95%说明它单步成功率约 $0.95^{1/14} \approx 0.996$。差别只有几个percentage point，但乘14次方后差距巨大。

**4-block Hanoi**：NSM 78%，两个VLA都0%。这是generalization test。NSM训练时没见过4-block，但它的PDDL planner只要看到更大的instance就能solve。VLA则只能重复训练时见过的3-block trajectory。

### Energy consumption

**Training**：

| | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| Time | 1天16小时 | 1天15小时 | **34分钟** |
| Total Energy | 68.5 MJ | 64.9 MJ | **0.85 MJ** |

**80倍的energy gap**。VLA要跑1.5天，NSM跑半小时。

**Inference**（每个episode）：

| | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| Total Power | 115.2 W | 114.0 W | **19.4 W** |
| 3-Block Energy | 7.96 kJ | 6.94 kJ | **0.83 kJ** |

NSM推理**完全不用GPU**——symbolic planner在CPU上跑，diffusion policy小到CPU足够。VLA必须GPU-backed，每次inference都吃power。

### VLM Planner结果

作者还测了三个VLM做planning的能力（给initial和goal image，让VLM输出pick-place command sequence）：

| VLM | Optimal | Invalid |
|---|---|---|
| GPT-5 | 84% | 16% |
| Qwen 7B | 0% | **100%** |
| PaliGemma 3B | 0% | **100%** |

GPT-5还行但84%远非perfect，latency 63秒太慢。Qwen和PaliGemma完全无法plan——所有plan都invalid。这强烈support Kambhampati的"LLMs can't plan"论点 [arXiv:2402.01817](https://arxiv.org/abs/2402.01817)。

---

## 为什么VLA会输得这么惨

我自己的intuition：

### 1. Long-horizon是end-to-end model的天然杀手

n步任务，单步成功率$p$，整体成功率约$p^n$。当$n=14$时：
- $p=0.99$ → 整体0.87
- $p=0.95$ → 整体0.49
- $p=0.90$ → 整体0.23
- $p=0.85$ → 整体0.10

VLA单步看起来不差，但累积起来崩了。NSM的structured decomposition给了它**automatic error recovery**——每个skill有termination condition，没完成不进下一步，相当于automatic retry。

### 2. VLA学的是correlation，不是rule

E2E-VLA看了300场Hanoi，它学到的是"看到这种画面之后action大概是这样"的statistical pattern。4-block Hanoi上它的behavior是直接执行3-block trajectory——说明它memorize了特定长度的sequence，没learn到rule。

NSM从primitives学PDDL domain，domain本身capture了"大不压小"的rule。给它4个block，classical planner立刻能用相同rule solve。这是**compositional generalization**——symbolic methods的天然优势。

### 3. PG-VLA的0%揭示了execution fidelity问题

PG-VLA有正确plan仍然0% success，说明问题不在planning而在execution。可能原因：
- Language command到action的mapping fuzziness：同一"Pick blue block"在不同context下trajectory差异大，model容易confuse
- No symbolic state feedback：grasp失败后没有"state不匹配"的signal，model继续执行
- Action chunking减少per-step visual feedback frequency

### 4. "Generalist"在structured task上是bug不是feature

VLA的卖点是generalist——一个model干所有事。但在structured task上，generalization across tasks和performance on specific structured task有trade-off。你用同一个大model干各种task，每个task都只能做到"还行"，而structured task要求"接近完美"。NSM的specialized architecture在specific domain上能做到near-perfect。

---

## 重要的critique

这篇paper结果striking，但有几个值得push back的点：

**1. Task太structured**。汉诺塔是perfectly symbolic的任务——discrete states, deterministic, clear rules。如果换成contact-rich manipulation（开瓶子、折衣服）或visual diversity大的任务，VLA的advantage可能就显现了。作者承认这点，但argue说很多real-world task（工业装配）确实是structured的。

**2. LoRA fine-tuning可能不optimal**。30k steps + default hyperparams，没tune。Full fine-tuning可能改善VLA表现。但counter-argument：即使performance改善，energy gap仍在。

**3. NSM的YOLOv8 detector是隐藏的prior**。YOLOv8 pretrained on COCO，本身带了massive visual prior。VLA是从raw pixel学，某种意义上是"公平"比较的disadvantage。不过paper说detector也是从demonstrations训的，没额外annotation。

**4. Generalization只测了4-block**。如果能测5-block、6-block，看NSM的generalization曲线，结论会更solid。78%已经比VLA的0%好太多，但趋势如何？

---

## 这篇paper的bigger picture

### 1. Energy应该成为first-class metric

VLA社区narrative一直围绕capability，没人systematic measure energy。但real robot deployment的bottleneck往往是power budget——你不可能给robot背个GPU server。这篇paper把energy放在和task performance同等位置，这是个**framing contribution**。

**希望VLA community以后report energy as standard metric**，不只report success rate。

### 2. 对"scale is all you need"的calibration

不是说VLA没价值，而是说VLA不是universal solution。在structured long-horizon manipulation这种特定domain，neuro-symbolic methods在两个维度都碾压VLA。这是对当前VLA hype的重要calibration。

### 3. Hybrid是未来

纯symbolic在perception-rich场景fail（你得handcraft perception module），纯neural在long-horizon structured场景fail。未来的方向可能是**principled hybrid**：
- VLM做perception + language understanding
- Symbolic planner做formal reasoning
- Diffusion policy做low-level control
- 三层之间有principled interface

π₀和GR00T的dual system design是某种hybrid，但缺少formal symbolic layer。如何把classical planning的correctness guarantee和neural network的flexibility结合，是个open question。

### 4. Compositional generalization vs distribution shift generalization

paper里NSM的"generalize to 4-block"是compositional generalization——同一rules下更大instance。VLA的"generalization"通常是distribution shift——新object、新scene。两种generalization本质不同：

- Compositional generalization是symbolic methods的强项
- Distribution shift generalization是end-to-end methods的强项（前提是scale足够）

对比architecture时必须分清在讨论哪种generalization，否则就是apples to oranges。

---

## 一句话总结

**在structured long-horizon manipulation任务上，neuro-symbolic methods在performance和efficiency两个维度都碾压VLA。VLA的"scale is all you need" narrative在energy-constrained robotics setting下站不住脚。**

paper链接：https://price-is-not-right.github.io

这篇paper对VLA community是个healthy wake-up call——别只盯capability，也得算算price。

---

# The Price Is Not Right: 一篇关于VLA vs. Neuro-Symbolic的硬核对比

## 1. Paper的Core Thesis

这篇paper的核心论点非常直接：在structured long-horizon manipulation任务上，end-to-end的VLA (Vision-Language-Action) foundation models不仅**task performance远不如**neuro-symbolic架构，而且在**energy consumption**上差了近两个数量级。作者选了Physical Intelligence的π₀作为VLA代表，选了Lorang et al.的few-shot neuro-symbolic imitation learning [arXiv:2508.21501](http://arxiv.org/abs/2508.21501) 作为NSM代表，在Robosuite的Towers of Hanoi变种上做了head-to-head比较。

paper标题"The Price Is Not Right"是个双关——"price"既指energy/compute cost，也暗讽当前VLA社区对"foundation model能解决一切"的乐观预期在structured任务上price-performance ratio严重失衡。

---

## 2. 实验任务设计：为什么是Towers of Hanoi？

Towers of Hanoi这个任务的选择非常有讲究，它具备几个关键性质：

1. **Long-horizon**: 3-block需要7步最优move，4-block需要15步，每个move本身又是pick+place两个sub-action
2. **Procedural constraints**: 大块不能放小块上——这是hard rule，不是soft preference
3. **Compositional structure**: 同一个pick-and-place primitive可以复用
4. **Generalization可测**: 3-block训练，4-block测试，立刻能看出是否学到了rule vs. memorized trajectory
5. **Discrete state space小，连续control space复杂**: 这正是TAMP (Task and Motion Planning)经典设置

他们用blocks替代discs，rectangular areas替代rods，主要是为了减少manipulation本身的难度，让比较聚焦在planning和long-horizon execution上。

三种任务难度梯度：
- **Individual Move**: 单次pick-and-place，纯low-level execution
- **3-Block Hanoi**: 训练时见过的long-horizon任务
- **4-Block Hanoi**: 训练时**没见过**的，测generalization

---

## 3. NSM架构详解（这是paper的技术精华）

NSM来自Lorang et al.的prior work，核心idea是**从少量demonstrations同时学symbolic domain和low-level policies**，不需要manually specify PDDL。

### 3.1 Symbolic Abstraction学习

输入：raw demonstration trajectories D
提取：node transitions τ^node = (n, l, n')

其中：
- **n, n'**: 高层states (abstract states)
- **l**: human-assigned label（transition的语义标签）

这些transitions形成graph G = ⟨V, E, L⟩：
- **V**: nodes，代表abstract states
- **E**: edges，代表skills
- **L**: edge labels

**Minimal bisimulation** Ḡ：用bisimulation minimization压缩graph，移除redundant states但保持equivalence。这一步是为了避免symbolic domain里出现过多冗余predicates导致planner搜索空间爆炸。Bisimulation的经典定义是两个states行为等价当且仅当对所有action，next-state分布等价；这里用的是deterministic version。

然后用**ASP-based solver**（Answer Set Programming，来自Bonet & Geffner, 以及Rodriguez et al.的工作 [KR 2021](https://doi.org/10.24963/kr.2021/51)）从compressed graph里infer PDDL domain：

σ = ⟨E, F, S, O⟩

各变量含义：
- **E** (Entities): 实体集合，比如blocks, areas
- **F** (Predicates): boolean或numerical predicates，比如 `on(block_1, block_2)`, `clear(block_1)`, `at(block_1, area_left)`
- **S** (States): grounded predicates组成的symbolic state集合
- **O** (Operators): 算子集合，每个operator o ∈ O有preconditions ψ和effects ω

### 3.2 Planning Task形式化

T = ⟨E, F, O, s₀, s_g⟩

- **s₀**: initial symbolic state
- **s_g**: goal symbolic state

Planner (这里用**Metric-FF** [Hoffmann 2003](https://www.jair.org/index.php/jair/article/view/10249))输出plan:

P = [o₁, ..., o_|P|]

即一个operator序列，能从s₀通过逐次应用operator的effects到达s_g。

### 3.3 Low-level Control: Diffusion Policy

每个operator o_i对应一个neural skill π_i ∈ Π，进一步分解为action-step sub-policies π_{i,j}，带termination conditions（这借鉴了Sutton的**options framework** [Sutton, Precup, Singh 1999]）。

Diffusion policy的训练目标：

**IL loss**:
$$\mathcal{L}(\pi) = \frac{1}{T} \sum_{t=0}^{T} \| \pi(\tilde{s}_t) - a_t \|^2$$

变量含义：
- **T**: trajectory length
- **s̃_t**: continuous state at time t（这里指relative pose observation）
- **a_t**: expert action
- **π(s̃_t)**: policy输出的predicted action

但diffusion policy实际不是直接regress，而是用**denoising score matching**。训练时expert action被加Gaussian noise，denoising network ε_θ学习从noisy action恢复clean action，conditioned on state。Inference时reverse diffusion process迭代refine noise sample生成action：

p_θ(a_t | s_t)

Diffusion policy相对普通BC的好处：
1. **Multi-modal action distribution**: 同一state下可能多个合理action，MSE loss会回归到mean造成mode collapse，diffusion能capture多模态
2. **Stochastic policy**: 自然支持exploration和robustness

### 3.4 关键设计：Relative Pose Observations

paper里特别强调所有diffusion policies operate on **relative pose observations**——object poses expressed w.r.t. end-effector。这点非常重要：
- Absolute pose会让policy需要学camera extrinsics + robot kinematics
- Relative pose把几何推理"hardcode"进observation space，policy只需学residual control
- 这其实是inductive bias，但合理的inductive bias

### 3.5 Object Detection Module

为了和VLA保持input modality一致（只用images + proprioception，不用object-space ground truth），NSM训练了：
- **YOLOv8** [Ultralytics](https://github.com/ultralytics/ultralytics): bounding-box detection
- **Lightweight gradient boosting regressor**: 从2个camera views (static + wrist)估计3D object pose

这里有个细节值得注意——YOLOv8 + gradient boosting这种组合比较"old school"，但恰好体现了paper的thesis：**不需要巨大的neural network，结构化方法+轻量学习就够了**。

### 3.6 Feature Selector φ

为了sample efficiency，observation经过task-relevant feature selector φ，只保留operator-relevant objects E_{o_i}，并express在end-effector relative coordinates里。这相当于attention mechanism的symbolic版本——不是soft attention，而是hard symbolic masking。

### 3.7 Execution Loop

```
1. User specifies (s₀, s_g) → mapped to PDDL instance T
2. Metric-FF solves T → plan P = [o₁, ..., o_n]
3. For each o_i:
   a. Invoke policy π_i
   b. π_i内部sequences π_{i,j} until termination
   c. 环境state更新
4. 检测是否到达goal
```

---

## 4. VLA Fine-tuning细节

### 4.1 π₀架构

π₀来自Physical Intelligence [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)，核心特点：
- **PaliGemma 2B**作为vision-language backbone
- **Gemma 300M**作为action header
- **Flow matching** for high-frequency control（替代diffusion，更高效）
- **Action chunking**: 一次输出一段action trajectory而非单步

### 4.2 LoRA Fine-tuning配置

两种配置，都用LoRA：
- **E2E-VLA**: command始终是"Play Towers of Hanoi"，monolithic学习
- **PG-VLA**: command是structured subtask commands ("Pick the blue block", "Place the blue block on the red block"等)，配合external planner

LoRA fine-tuning 30k steps，用OpenPi官方脚本默认hyperparameters。

### 4.3 训练数据对比（关键！）

| | VLA | NSM |
|---|---|---|
| Episodes | 300 | 50 |
| 内容 | Full Towers of Hanoi runs + random valid configs | 只有stacking (pick-and-place pairs) |
| 是否见过Hanoi完整解 | 是 | **否** |

这是paper最有意思的setup之一：**NSM从未在训练时见过Towers of Hanoi的完整解**，只见过单步stacking demonstrations，然后从这些stacking demos里infer出整个PDDL domain，再用classical planner来solve Hanoi。VLA则看了300个完整的Hanoi trajectories。

这种对比是公平的吗？某种意义上是"apples to oranges"——VLA是end-to-end从轨迹学，NSM是structured从primitives学。但paper的论点正是：**对于structured任务，后者更data-efficient**。

---

## 5. 实验结果深度分析

### 5.1 能耗对比（Table I, II）

**Training energy**:

| Metric | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| Time | 1d 16h 26m | 1d 15h 42m | **34m** |
| GPU Mean Power (W) | 423.6 | 409.1 | 316.5 |
| GPU Energy (MJ) | 61.7 | 58.5 | **0.65** |
| Total Energy (MJ) | 68.5 | 64.9 | **0.85** |

差距：**~80×**。这个数字很惊人。LoRA fine-tuning 1.5天 vs. NSM训练34分钟。

值得注意：NSM的CPU utilization (10.5%)远高于VLA (3.12%)，说明NSM更多依赖symbolic reasoning (CPU-bound)，VLA更多是matrix multiplication (GPU-bound)。

**Inference energy**:

| | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| GPU Power (W) | 72.4 | 70.8 | **0** |
| Total Power (W) | 115.2 | 114.0 | **19.4** |
| 3-Block Episode Energy (kJ) | 7.96 | 6.94 | **0.83** |

NSM推理完全不用GPU——symbolic planner在CPU上跑，diffusion policy虽然用neural network但很小，CPU足够。VLA推理必须GPU-backed，每次episode能耗~10× NSM。

### 5.2 Task Performance（Table II）

| Task | E2E-VLA | PG-VLA | NSM |
|---|---|---|---|
| Individual Move | 87.0% | 59.6% | **99.0%** |
| 3-Block Hanoi | 34.0% | **0.0%** | **95.0%** |
| 4-Block Hanoi (unseen) | 0.0% | 0.0% | **78.0%** |

几个值得深挖的点：

**1. PG-VLA在Individual Move上反而比E2E-VLA差**（59.6% vs 87.0%）

这反直觉。paper的解释是："diversity within each command category reduced execution fidelity"。意思是PG-VLA训练时同一command对应多种实际场景（不同block颜色、位置），模型难以学到精确control。E2E-VLA虽然command单一，但它"memorized"了完整trajectory template，反而执行更准。

这其实是end-to-end model的**一个隐蔽优势被structured decomposition抵消了**——decomposition引入了variance。

**2. PG-VLA在3-Block上0% success**

即使给了optimal plan（subtask commands），PG-VLA完全无法完成。这个结果非常重要——它说明**VLA的execution fidelity不足以支撑多步long-horizon任务**。即使每步都给正确instruction，单步成功率若<1/14（14个subtask），整体成功率会指数衰减到接近0。

paper观察到PG-VLA大多在第3或第5个subtask失败，最远到第9个subtask。这暗示error accumulation的exponential decay。

**3. E2E-VLA的bimodal distribution**

E2E-VLA的task advancement rate是49.6%，但分布是bimodal——要么很早失败（前4步内），要么基本完成。这暗示它学到了某种"trajectory template"，一旦进入template轨道就能跑完，一旦偏离就完全崩溃。

**4. NSM的4-block generalization 78%**

这是paper最亮眼的结果。NSM训练时**只见过stacking pairs**，从未见过完整Hanoi，但能在4-block Hanoi（15步最优解）上达到78% success。这证明了：
- Learned PDDL domain真的capture了Towers of Hanoi的rules
- Classical planner能generalize到更大instance（这是symbolic planning的天然优势）
- Low-level diffusion policies能泛化到新的block configurations

### 5.3 VLM Planner对比（Table III, IV）

| VLM | Optimal (%) | Invalid (%) | Latency (s) |
|---|---|---|---|
| GPT-5 | **84** | 16 | 63.1 |
| Qwen 7B | 0 | **100** | 1.83 |
| PaliGemma 3B | 0 | **100** | 0.22 |

GPT-5大幅领先但84%仍非perfect，且latency 63秒太长。Qwen和PaliGemma完全无法plan——100% invalid。这印证了Kambhampati的"LLMs can't plan"论点 [arXiv:2402.01817](https://arxiv.org/abs/2402.01817)。

---

## 6. 为什么VLA在structured任务上fail？我的intuition

paper的discussion里提到了几个原因，但我想更深入地build intuition：

### 6.1 Long-horizon的exponential decay

如果单步成功率p，n步任务的整体成功率约p^n。3-block Hanoi有14个subtask：
- p=0.95 → 0.95^14 ≈ 0.49
- p=0.90 → 0.90^14 ≈ 0.23
- p=0.85 → 0.85^14 ≈ 0.10

E2E-VLA 34% success暗示单步成功率约0.91——这其实不低！但14步累积下来就崩了。NSM 95% success暗示单步成功率约0.996，几乎完美。

**这就是structured任务对end-to-end model的残酷之处**：每步必须接近完美。

### 6.2 VLA学的是trajectory correlation，不是rule

E2E-VLA训练时看300个完整Hanoi trajectory，它学到的更像是"第3步之后通常第4步是这样"的statistical correlation，而不是"大块不能放小块上"的rule。

证据：4-block Hanoi上E2E-VLA的behavior是执行3-block的trajectory——它没学rule，只memorize了特定长度的sequence。

### 6.3 PG-VLA fail的更深层原因

PG-VLA有正确plan仍然失败，说明问题不在planning而在execution。可能原因：
1. **Language command到action的mapping fuzziness**: "Pick the blue block"在不同context下需要的trajectory差异大，但语言command相同，model容易confuse
2. **No error recovery**: 一旦grasp失败，没有symbolic state feedback告诉它"重新try"，只能继续执行
3. **视觉feedback integration不足**: π₀的action chunking可能减少了per-step visual feedback的frequency

NSM在这点上天然有优势：每个skill有termination condition，没完成不会进next operator，相当于有automatic retry机制。

---

## 7. 关于paper的critique

虽然paper结果striking，但有几个值得push back的点：

### 7.1 Task太structured

Towers of Hanoi是**完美symbolic**的任务——discrete states, deterministic transitions, clear rules。这种任务恰好是symbolic methods的"主场"。如果换成：
- Contact-rich manipulation（开瓶子、折衣服）
- Visual diversity大（不同光照、texture）
- Dynamic environment（移动物体）

VLA的优势可能就显现了。paper在Conclusion里也承认了"this comparison focuses on a structured benchmark"。

### 7.2 LoRA fine-tuning可能不optimal

30k steps LoRA with default hyperparams是否是π₀的最佳配置？Full fine-tuning或更长训练可能改善VLA表现。但paper的counter-argument是：即使改善performance，energy gap仍在。

### 7.3 NSM的Object Detection是"cheating"吗？

NSM用YOLOv8 + gradient boosting做object detection，这其实是把perception module单独训练了。VLA是从raw pixel end-to-end学。这某种意义上NSM用了**额外标注**或**额外训练数据**来训detector。

不过paper里说detector也是从demonstrations训的，没有额外标注。但YOLOv8是pretrained on COCO的，这本身就是massive prior。

### 7.4 Generalization test只有4-block

如果能测5-block、6-block，NSM的generalization曲线会更clear。78%在4-block上已经比VLAs的0%好得多，但趋势如何？

---

## 8. 与相关工作的联系

### 8.1 Kambhampati的LLM planning critique

paper引用了Kambhampati et al.的"LLMs can't plan, but can help planning in LLM-modulo frameworks" [arXiv:2402.01817](https://arxiv.org/abs/2402.01817)。这篇paper的VLM planner结果（Qwen 100% invalid）强烈支持Kambhampati的论点。

Kambhampati的core argument是：LLM是approximate retriever不是reasoner，planning需要compositional generalization，而LLM的next-token prediction本质上不support这种composition。他的LLM-modulo框架建议LLM做generator，external formal verifier做critic，iterative refine。

NSM的PDDL planner正是这种"formal reasoner"——它guarantee plan的correctness（w.r.t. domain model）。

### 8.2 Diffusion Policy的起源

Chi et al.的Diffusion Policy [RSS 2023](https://arxiv.org/abs/2303.04137) 是NSM low-level control的基础。Diffusion policy的multi-modal capture能力对manipulation很关键——同一observation下"从左边抓"和"从右边抓"都合理，MSE regression会averaging出无效action。

### 8.3 Options Framework

Sutton, Precup, Singh的options framework [AI 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521) 是NSM sub-policy decomposition的理论基础。Options = (initiation set, policy, termination condition)。NSM的每个skill π_i对应一个option，termination condition learned from data。

### 8.4 Bisimulation minimization

这是formal methods里的经典概念，Milner提出。在MDP里两个states是bisimilar当且仅当它们对所有action产生相同的reward distribution和bisimilar successor states。Bisimilar states可以merge而不损失optimal policy。

NSM用bisimulation来compress symbolic state graph，避免PDDL domain里出现redundant predicates。这其实是把RL里的state abstraction theory用到了symbolic learning里。

### 8.5 PDDL和Classical Planning

PDDL (Planning Domain Definition Language) [McDermott et al. 1998](https://www.informatik.uni-freiburg.de/~ki/teaching/ss05/planung/pddl.pdf) 是classical planning的标准语言。Metric-FF是Hoffmann的planner，用relaxed planning graph heuristic做forward search，支持numeric fluents。

### 8.6 VLA evolution

VLA lineage:
- **RT-1/RT-2** (Google) → early VLA
- **OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) → open-source VLA
- **GR00T N1** (NVIDIA) [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) → dual system design
- **π₀** (Physical Intelligence) [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) → flow matching + action chunking
- **UniVLA** [arXiv:2505.06111](https://arxiv.org/abs/2505.06111) → latent action unified architecture

paper选π₀是因为它的open-weight + fine-tuning infrastructure公开，结果可复现。

### 8.7 LIBERO benchmark

π₀在LIBERO上pretrained，paper用Franka Panda（LIBERO同平台）是为了align pretraining distribution。LIBERO是[Long-horizon Language-conditioned Benchmark for Robot manipulation](https://lifelong-robot-learning.github.io/libero/)，近期VLA标准benchmark。

---

## 9. 我的Takeaways和Intuition Building

### 9.1 关于architecture choice

这篇paper最核心的intuition：**architecture应该match任务的structure**。

- 任务structured（rules, compositionality, discrete states）→ symbolic structure pays off
- 任务unstructured（visual diversity, contact-rich, creative）→ end-to-end learning pays off

这其实是No Free Lunch在robotics的体现。VLA的卖点"generalist"在structured任务上是bug不是feature——generalization across tasks和performance on specific structured task有trade-off。

### 9.2 Energy as first-class metric

paper最contribution的事情是**把energy放在和task performance同等的位置**。这很重要：
- 部署robot不可能背个GPU server
- Edge robotics的bottleneck往往是power budget
- 训练成本（$$ + carbon）也应该被report

VLA社区的"scale is all you need"narrative在energy-constrained setting下站不住脚。

### 9.3 Hybrid是未来

paper结尾说"hybrid neuro-symbolic architectures remains well motivated"。我agree。纯symbolic在perception-rich场景fail，纯neural在long-horizon structured场景fail。Hybrid的open question是**how to interface**——什么时候用symbolic，什么时候用neural，如何转换。

π₀的dual system design（VLM + diffusion transformer）其实是某种hybrid，但缺少formal symbolic layer。GR00T N1也类似。未来的方向可能是：
- LLM/VLM做perception + language understanding
- Symbolic planner做formal reasoning
- Diffusion policy做low-level control
- 三层之间有principled interface

### 9.4 关于"generalization"的定义

paper里NSM的"generalization to 4-block"是**compositional generalization**——同一rules下更大instance。VLA的"generalization"通常是**distribution shift generalization**——新object、新scene。这两种generalization本质不同：
- Compositional: symbolic methods的强项
- Distribution shift: end-to-end methods的强项（前提是scale足够）

对比时必须分清在讨论哪种generalization。

### 9.5 对VLA community的启示

1. **Report energy**: 不只report success rate，也report kWh
2. **Long-horizon benchmark**: 不能只测short-horizon pick-and-place
3. **Compositional generalization**: 测试同rule不同instance的能力
4. **Structured decomposition可能不是free lunch**: PG-VLA的失败显示naive decomposition反而hurt performance
5. **Error recovery很重要**: 14步任务每步必须接近完美

---

## 10. 代码和资源

- **Paper website**: https://price-is-not-right.github.io
- **π₀ (OpenPi)**: https://github.com/Physical-Intelligence/openpi
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **Robosuite**: https://github.com/ARISE-Initiative/robosuite
- **Metric-FF**: https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **Lorang et al. NSM**: [arXiv:2508.21501](http://arxiv.org/abs/2508.21501)
- **Kambhampati LLM planning**: [arXiv:2402.01817](https://arxiv.org/abs/2402.01817)
- **Weights & Biases (能耗监测)**: https://www.wandb.com/

---

## 11. 开放问题

读完这篇paper我会想：

1. **如果用更大的VLA（如GR00T N1的full size而非LoRA）会不会改变结果？** paper没有测，但energy gap大概率更大。
2. **如果给VLA加上symbolic planner作为guidance（类似PG-VLA但planner更强）能不能救回来？** PG-VLA的失败说明不能简单加，需要更deep integration。
3. **NSM的symbolic abstraction learning在更complex domain（如cooking）能work吗？** cooking的symbolic structure不那么clear，bisimulation可能不直接apply。
4. **能否用一个neural network学习NSM的整个pipeline，达到类似performance和energy efficiency？** 这其实是meta question——structured architecture能否被large enough neural net完全absorb？
5. **VLA的action chunking和NSM的options framework本质相似？** 都是hierarchical temporal abstraction，但一个implicit一个explicit。能否设计architecture让VLA的chunking变成explicit options？

---

## 12. 最后的meta comment

这篇paper在我看是**对VLA hype的重要calibration**。不是说VLA没价值，而是说VLA不是universal solution。在structured long-horizon manipulation这种特定domain，neuro-symbolic methods在performance和efficiency两个维度都碾压VLA。

paper title "The Price Is Not Right"恰好点出当前VLA社区的盲点——大家盯着capability，没人systematic measure price。而price在real-world deployment上是hard constraint。

希望这篇paper能push VLA community：
1. Report energy as standard metric
2. Test long-horizon compositional generalization
3. Compare against structured baselines而非只ablation against own variants
4. Reconsider whether "scale is all you need"在energy-constrained robotics setting还成立

Reference paper: [The Price Is Not Right (Duggan, Lorang, Lu, Scheutz)](https://price-is-not-right.github.io)
