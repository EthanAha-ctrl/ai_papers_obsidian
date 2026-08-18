---
source_pdf: FailSafe Reasoning and Recovery from Failures in Vision-Language-Action
  Models.pdf
paper_sha256: 57a5b9ca67d2124e8e04cf4021a1c14dc458943246f6ae2208ca4ed1ce231bd4
processed_at: '2026-08-18T12:11:56-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FailSafe

## 一句话版本

VLA models 训练数据全是 "完美演示"，deployment 一旦出错就傻眼。FailSafe 在 simulator 里**故意制造各种翻车场景**，然后记录 "怎么翻的 + 怎么救回来的"，拿这些数据 fine-tune 一个 VLM 当 "外挂医生"，每 10 步给 VLA 体检一次，发现要翻车就开药方输出 corrective action。

---

## 为什么这事有意思

现在 VLA field 的画风大概是这样：大家拼命堆 data、堆 parameters，从 OpenVLA 到 π₀，success rate 漂亮得很。但有个尴尬的事实——**所有 training data 都是 demo 成功的 trajectory**。这就好比教小孩骑自行车，只给他看别人骑得帅的视频，从来不让他看摔跤的视频，也不教他摔了怎么爬起来。结果就是真骑车一旦要摔，他完全不会反应。

这 paper 的 insight 很朴素：**失败和恢复本身就是 policy 的一部分，不能只学 success**。人学技能也是这样——你学会开车不是因为驾校教练只演示 perfect 路线，而是因为你在各种差点撞墙、差点追尾的惊险中积累了 recovery instinct。

---

## FailSafe 在干嘛，一张图说清楚

想象一个 robotic arm 在 ManiSkill 里执行 Pick Cube 任务，正常 trajectory 是：

$$A \to B \to C \to D$$

翻译成人话就是：approach → grasp → lift → retreat。

FailSafe 干的事：在某个 stage 偷偷把 target pose $P_B$ 扰成 $P_{B'}$，于是机械臂跑出一条**注定失败的 trajectory** $A \to B' \to C \to D$。然后它去找一个 "correction pose" $P_c$，让 replan 后的轨迹变成：

$$A \to P_d \to P_c \to B \to C \to D$$

**如果这条 corrected trajectory 最终 task 成功了**，就把这个 $(P_d, P_c, \Delta A)$ 三元组存进 dataset。如果没用就扔掉。

就这么简单。整个 pipeline 的精髓就是那个 **systematic verification**——你 claim 的 corrective action 必须在 simulator 里真跑一遍证明能救活任务，不能只是"看起来有道理"。

---

## 三个 failure mode，人话版

作者只定义了三种 basic failure：

**Translation failure**：机械臂往某个方向偏了一点，比如本该 grasp cube 正上方，结果偏了 10cm，抓了个空气。

**Rotation failure**：gripper 的 roll/pitch/yaw 偏了，比如本该水平夹，结果歪了 30 度，cube 滑掉了。

**No-op failure**：机械臂卡住了不动，类似于 VLA 输出了一串 no-op action，机器人就僵在那。

作者说得很直白：看起来这三种太简单了对吧？但**复杂的失败基本都能追溯到这三个 root cause**。比如 transport 阶段 cube 掉了，往回追溯往往是 grasp 那一步就有 rotation 偏差。这跟 root cause analysis 的思路一样——表面问题在下游，根因在上游。

---

## Action collection 这块为啥不 trivial

naive 想法：既然 $P_{B'} = P_B + \epsilon$，那 corrective action $\Delta A = -\epsilon$ 不就完事了？

作者说不行，因为：

1. **直接逆向扰动可能撞上物体**。比如 gripper 已经在 cube 里面了，你直接反推可能撞 cube 侧面。
2. **Correction 要在 failure 即将显现的任何 timestep 都能 work**，不局限于出错的那个 stage。
3. **7-DoF pose 不是简单 Euclidean 空间**，rotation 的叠加在 SO(3) 上是非线性的，不能简单减。

所以他们的做法是：在 failure trajectory 上从第 10 步开始遍历 deviated pose $P_d$，每个 $P_d$ 去 correct trajectory 上随机采一个 $P_c$，算 7-DoF difference 作为 $\Delta A$。然后用 motion planner replay 验证。

这里有个**反直觉的 design choice**：虽然 failure type 是单标签（比如 "trans_x failure"），但生成的 $\Delta A$ **不是 1-sparse 的**，它在所有 7 个维度上都有值。作者解释说，failure type 只是标记 dominant error source，不限制 $\Delta A$ 只改一个维度。这其实挺合理——真实世界失败往往是 multi-modal 的，translation 错了通常伴随 rotation 也错一点，强行 1-sparse 反而不自然。

---

## Systematic verification 是这套系统的灵魂

我特别想强调这个 step，因为这是区别 FailSafe 和 AHA、RoboFAC 的关键。

AHA 和 RoboFAC 也会在 simulator 里扰动 poses 生成 failure，但它们只输出 textual reasoning："the gripper is misaligned, should move left"。这种自然语言反馈看起来很 expressive，但**没法直接喂给 VLA 的 action head**——VLA 要的是 7 维 continuous action，不是一句 "move left a bit"。

FailSafe 多走了一步：每个 candidate $\Delta A$ 都要在 simulator 里**真跑一遍**，看 task 最终能不能完成。能完成才进 dataset。这就保证了 dataset 里每一条 failure-action pair 都是 **闭环验证过的 executable recovery**。

你 Karpathy 应该会喜欢这个设计——这跟你在 nanoGPT 里强调的 "data quality is everything" 一脉相承。别让模型学一堆 "看起来对的垃圾 correction"，从源头就把 data quality 控制住。

---

## FailSafe-VLM 怎么当外挂

Deployment 时的流程很直观：

```
VLA model 跑 10 步 → FailSafe-VLM 接管 → 看图判断有没有要翻车的苗头 
→ 如果有就输出 ΔA 救一下 → 接着让 VLA 继续
```

这里有两个 design choice 值得说：

**为什么是每 10 步而不是每步？** 因为 VLM 的 forward pass 比 VLA 慢一个数量级，每步 invoke 的话延迟爆炸。而且单步 action 太 noisy，看不出 failure trend。10 步差不多对应 failure 显现的时间尺度。

**为什么用 LLaVA-OneVision-7B 当 base？** 因为它已经有 spatial reasoning 能力（SigLIP vision tower + Qwen2-7B language），fine-tune 时只需要把 robot domain 的知识注入。而且 co-training 了一个 RoboPoint VQA mixture 来强化 spatial affordance reasoning——判断 gripper 是否对齐 cube 本质就是 spatial affordance 问题。

---

## 实验结果，人话解读

### VLA 性能提升

| VLA | Baseline | + FailSafe | 提升 |
|---|---|---|---|
| π₀-FAST | 78.7% | 82.7% | +4% |
| OpenVLA | 14.7% | 37.3% | **+22.6%** |
| OpenVLA-OFT | 90.7% | 98.7% | +8% |

直觉读法：**baseline 越弱，FailSafe 救得越多**。OpenVLA 的 baseline 才 14.7%，因为它用 discrete action token 在 continuous control 上天然吃亏，FailSafe 正好补这块短板。π₀-FAST 和 OFT 的 baseline 已经很强了，headroom 小，提升就小。

这其实告诉我们一个有意思的事：**FailSafe-VLM 不是在 "锦上添花"，而是在 "雪中送炭"**。VLA 越弱，failure 越多，FailSafe 的价值越大。

### 跟 GPT-4o、Gemini 比

这块结果最 striking：

- GPT-4o detect failure 还行（70%），但 reasoning 具体失败类型只有 19.6% accuracy
- Gemini detect failure 62%，reasoning 14%
- Qwen2.5-VL 直接摆烂，全输出 "no failure"

而 FailSafe-VLM：binary success 90.9%，failure type accuracy 83.7%，recovery action cosine similarity 0.65。

**人话翻译**：GPT-4o 能看出来 "好像出问题了"，但说不清是啥问题、怎么修。FailSafe-VLM 不仅看出来了，还能精确指出 "是 trans_x 方向偏了" 并且给出具体 7 维 corrective action。

这说明 general VLM 的 spatial reasoning 能力在 fine-grained robot control 这种 task 上**根本不够用**。这块需要 task-specific training data。这也呼应了为什么 FailSafe pipeline 有存在价值——不是 "VLM 已经能做这个"，而是 "需要专门 data 才能做"。

### Cross-embodiment 和 cross-object

FailSafe-VLM 在 Franka Panda 上训练，直接拿去帮 xArm 6 救场，stack cube 从 56% 拉到 76%。换成没见过的 Sphere 和 Charger，平均也提升 17.4%。

这说明学到的 failure reasoning **不绑定具体 robot 或 object**，是 task-space 层面的抽象能力。translation 偏了就是偏了，不管你是 7-DoF Panda 还是 6-DoF xArm，不管你抓的是 cube 还是 sphere，偏了就要修。

---

## 你 Karpathy 视角下的 intuition

从你 Software 2.0 的框架看，FailSafe 做的事其实很自然：

传统机器人有 explicit exception handler，触发条件由工程师写死——这是 Software 1.0。FailSafe 把 exception handler 也变成 learned 的，并且用大量 failure data 训练——这是 Software 2.0 的延伸。

更妙的是，它用的不是 RL 而是 **supervised learning with simulator-verified data**。这避免了 RL 的 sample efficiency 灾难，又保留了 "环境反馈" 这个信号。某种意义上，systematic verification 就是一个 "discriminative reward model"——只不过它输出的是 binary success/fail 而不是 scalar reward。

还有一层 analogy：FailSafe-VLM 每 10 步接管一次，输出 correction，这很像 **inference-time compute** 的思路。OpenAI o1 在 inference 时花更多 compute 做 reasoning 来提升 accuracy，FailSafe-VLM 在 inference 时花更多 compute 做 failure detection + recovery 来提升 success rate。两者都是 "用 inference 时算力换 performance" 的 trade-off。

---

## 这 paper 的 limitation，我不客气的说

1. **只测了 3 个 task**：pick/push/stack cube，这在 ManiSkill 里属于最简单的 manipulation。真要证明 pipeline 通用性，得上 peg-in-hole、articulated object manipulation 这种有 insight 的 task。

2. **Failure mode 太简单**：只有 translation/rotation/no-op。真实世界 failure 远比这复杂——物体 deform、friction 突变、partial occlusion、sensor dropout，这些都没覆盖。作者说 "复杂 failure 是 basic mode 的复合"，这个 claim 需要更多 empirical support，不是 self-evident 的。

3. **No real-world robot eval in main paper**：只有 supplementary video。sim-to-real 的 gap 在 VLA 领域是 open problem，sim 里 +22.6% 不代表 real world 也能 +22.6%。

4. **Closed-loop on ΔA 缺失**：FailSafe-VLM 输出 ΔA 后，VLA 执行了，但没有 verify ΔA 是否真的修对了。如果 ΔA 本身错了，没有 fallback。理想情况应该有个 closed-loop correction，或者 learned 一个 "confidence threshold" 决定是否 trust FailSafe-VLM 的 output。

5. **Camera view generalization 打折扣**：虽然测试了 novel view，但仍然是 ManiSkill 渲染风格。Real-world camera 有 blur、lighting 变化、occlusion，distribution shift 大得多。

---

## 我的 takeaway

如果让我一句话总结 FailSafe 的价值：**它把 "失败恢复" 从 human-in-the-loop 变成 VLM-in-the-loop，并且用 simulator-verified data 保证了 recovery action 的可执行性**。

这块 field 之前有两个 gap：
1. **AHA/RoboFAC 只输出 language feedback**，不能直接 feed VLA
2. **OLAF/YAY 依赖 human intervention**，不能 scale

FailSafe 同时填了这两个 gap：automated generation + executable action。这是真正的 contribution，虽然 architecture 上没有 fancy novelty（就是 fine-tune 个 LLaVA-OV-7B），但 **pipeline design 的 insight 值得 follow**。

你 Karpathy 一直说 "pipeline > architecture"，这 paper 恰好是个好例子。没有新 architecture，没有新 loss function，就是把 failure generation + verification + action collection 这条 pipeline 设计对了，结果就很 work。

未来我觉得最值得 follow 的方向：**把 FailSafe-VLM 内化进 VLA 自己**，让 VLA 在 forward pass 里就能 self-detect failure + self-correct。这样就不需要 10 步外挂的模式，而是每步都有 implicit failure awareness。这跟 ECoT (Embodied Chain-of-Thought) 的思路会 merge，最终走向 "self-correcting VLA"。

这种模型在 deployment 时不需要 external assistant，自己就是一个 complete system。我觉得这才是终局形态。FailSafe 是这个方向的重要 early step。

---

Web links 留几个核心的：

- FailSafe project page: https://jimntu.github.io/FailSafe/
- ManiSkill 3 (simulator): https://github.com/haosulab/ManiSkill
- OpenVLA: https://github.com/openvla/openvla
- π₀ blog: https://www.physicalintelligence.company/blog/pi0
- AHA (prior work): https://aha-vla.github.io/
- RoboPoint (co-training data source): https://robopoint.github.io/
- LLaVA-OneVision: https://github.com/LLaVA-VL/LLaVA-NeXT

---

# FailSafe: 给 VLA 装上"失败免疫系统"

## 1. Paper 的核心定位

Karpathy 你应该对 VLA (Vision-Language-Action) 的演进很熟悉——从 RT-2、OpenVLA 到 π₀, 整个 field 在做的事情本质上就是把你 Software 2.0 的理念延伸到 motor control 层面: 用神经网络直接把 pixels + language 映射成 7-DoF end-effector action. 这篇 FailSafe paper 关注的是 VLA 的一个**被严重忽视的盲区**: training data 全是 clean ground-truth trajectories, deployment 时一旦出现 small perturbation 机器人就会 frozen, 因为它从未见过 "如何从 failure 状态回到正轨".

FailSafe 的核心 contribution 可以浓缩成一个公式化的循环:

$$\mathcal{D}_{\text{FailSafe}} = \{(\mathbf{I}_{t:t+10}, \ell, f, \Delta\mathbf{A}) \mid \text{verify}(\text{replay}(P_d, P_c) = \text{success})\}$$

其中 $\mathbf{I}_{t:t+10}$ 是 10 帧多视角图像观测, $\ell$ 是 task instruction, $f \in \{\text{trans}_x, \text{trans}_y, \text{trans}_z, \text{rot}_x, \text{rot}_y, \text{rot}_z, \text{no-op}\}$ 是 failure type, $\Delta\mathbf{A} \in \mathbb{R}^7$ 是 7-DoF recovery action. 这个 verify 的闭环是关键——拒绝任何"看起来合理但实际无法救活任务"的 recovery action.

Project page: https://jimntu.github.io/FailSafe/

---

## 2. 为什么这个问题重要: 与 prior work 的对照

 robotics failure reasoning 的 prior art 主要分两派:

**派系 A: Human-in-the-loop correction**
- OLAF (Liu et al. 2023): 让 human verbalize correction, LLM 选 candidate action. https://arxiv.org/abs/2310.05045
- YAY (Shi et al. 2024): 把 human 的高层语言反馈插入 policy. https://arxiv.org/abs/2403.12910

**派系 B: 自动化 failure generation, 但只输出 language feedback**
- AHA (Duan et al. 2024): 在 simulator 中扰动 key poses, VLM 输出 textual failure analysis. https://aha-vla.github.io/
- RoboFAC (Lu et al. 2025): 类似 pipeline, 提供 textual correction. https://arxiv.org/abs/2505.12224
- REFLECT (Liu et al. 2023): LLM 基于 hierarchical summary 判断 failure. https://arxiv.org/abs/2310.15044

FailSafe 与这些工作的本质区别: **直接生成 robot-executable 的 7-DoF delta action, 而不是 "move left a bit" 这种语言指令**. 这非常关键——VLA models 是 action-conditioned 的, 自然语言 "the gripper should move left to align with the center of the cube" 这种 instruction 既没有 magnitude, 也没有 endpoint, 根本无法直接 feed 给 VLA 的 action head. 这就是为什么 FailSafe 不依赖 language grounding 这层 indirection.

---

## 3. Failure Generation: 把"错误"做成 first-class data

### 3.1 三种 failure mode 的设计动机

FailSafe 定义三种基本 failure mode:

| Failure mode | 数学定义 | 噪声范围 |
|---|---|---|
| Translation | $\Delta\mathbf{t} \in \{x, y, z\}$ 方向偏移 | $\pm 0.1$ (meters, ManiSkill scale) |
| Rotation | $\Delta\mathbf{r} \in \{\text{roll}, \text{pitch}, \text{yaw}\}$ 偏移 | $\pm 1$ radian (~57°) |
| No-ops | gripper 在某时间段 frozen | 时间步随机 |

作者特别强调: **这三种 mode 看起来 trivial, 但它们是 multi-step failure 的 root cause**. 比如物体在 transport 阶段 slip, 通常追溯回去是 initial grasp 时 translation 或 rotation 偏差. 这与因果推理中的 "root cause analysis" 思想一致.

### 3.2 Motion planning 的 stage 分解

ManiSkill 把每个 task 分解为多个 stage, 例如 Pick Cube 典型是:

$$A \to B \to C \to D$$

- $A$: approach (gripper 接近 object 上方)
- $B$: grasp (descend + close fingers)
- $C$: lift (沿 z 轴上升)
- $D$: retreat (移动到 target pose)

FailSafe 通过 YAML config + custom env wrapper, 把某一 stage 的 target pose $P_B$ 扰动到 $P_{B'}$, 跑出 trajectory $A \to B' \to C \to D$, 如果最终 task 失败就保留. 这种 perturbation 是 **stage-level** 而不是 **timestep-level**, 这是关键——它对应了真实部署中 VLA 在 sub-goal 层面出错的情况.

### 3.3 Delayed failure 的处理

这是 paper 里很微妙的一个点: **failure 可能在 root error 之后好几步才显现**. 比如 grasp 阶段的 rotation error, 可能要等到 lift 阶段 object 才掉下来. FailSafe pipeline 显式覆盖这种 delayed failure case, 这对训练模型的 "future-aware failure detection" 很重要——模型不能只看当下, 还要看 trajectory 后续会不会出问题.

---

## 4. Action Collection: 最 nontrivial 的算法部分

这一节是 paper 中最技术性的部分, 我详细解析.

### 4.1 为什么不能直接用 perturbation 作为 delta action

naive 想法: 既然 $P_{B'} = P_B + \epsilon$, 那 $\Delta\mathbf{A} = -\epsilon$ 不就行了吗? 答案是不行, 因为:
- 直接 inverse perturbation 可能导致 gripper-object collision
- 失败发生在某 stage, 但 correction 应该在 failure 即将显现前的任意 timestep 都能用 (robustness 要求)
- 7-DoF pose 之间不是简单 Euclidean 空间, rotation 的叠加是非线性 SO(3)

### 4.2 Pose pair mapping 算法

设 correct trajectory 为 $\tau_c = \{P_c^{(1)}, P_c^{(2)}, \ldots, P_c^{(N)}\}$, failure trajectory 为 $\tau_d = \{P_d^{(1)}, \ldots, P_d^{(M)}\}$, 每个 $P \in \text{SE}(3) \times \mathbb{R}$ (6-DoF pose + 1 gripper).

算法:

```
for i in range(10, M):                  # 从第 10 步开始, 早期 detection 不可靠
    P_d = P_d^{(i)}
    j = random_sample(range(10, N-3))   # correct trajectory 的有效窗口
    P_c = P_c^{(j)}
    ΔA = P_c ⊖ P_d                      # 7-DoF difference (注意 ⊖ 是 SE(3) 减法)
    candidate_pairs.append((P_d, P_c, ΔA))
```

**窗口限制的设计动机**:
- 起始 10 步不用: 早期 trajectory 还在 approach 阶段, deviation 还没成形, detection 不靠谱
- 倒数 3 步不用: 给后续 motion planning 留出 collision avoidance 的余量
- no-op 特殊处理: $P_c$ 从 $P_d$ 之后 3-10 步采样, 因为 no-op 的 fix 是 "advance along the correct path"

### 4.3 ΔA 的 sparse/non-sparse 问题

这是 paper 里最反直觉的设计. 虽然 failure type 是单一 (比如 trans_x), 但 **ΔA 在所有 7 维上都有值**, 不是 1-sparse. 作者解释: failure type 用来标记 **dominant error source**, 而不是限制 ΔA 的维度. 这点很关键, 因为:

1. 真实失败往往是 multi-modal (translation + rotation 同时偏差)
2. ΔA 的 non-sparse 形式更自然地 fit VLA 的 continuous action space
3. 这避免了 "must classify failure first, then act" 的硬决策, 给 model 留出 soft reasoning 空间

数学上, 给定 $f \in \{1, \ldots, 7\}$ 是 dominant failure axis, 实际 $\Delta\mathbf{A}$ 满足:

$$\|\Delta\mathbf{A}\|_0 > 1, \quad |\Delta\mathbf{A}_f| \approx \max_k |\Delta\mathbf{A}_k|$$

---

## 5. Systematic Verification: 数据质量的守门员

这是我最喜欢的设计. 每个 $(P_d, P_c, \Delta\mathbf{A})$ 候选都必须通过 replay 测试:

$$\text{verify}(P_d, P_c) = \mathbb{1}\left[\text{MotionPlanner}(A \to P_d \to P_c \to B \to C \to D) = \text{success}\right]$$

也就是说, simulator 实际跑一遍 $A \to P_d \to P_c \to B \to C \to D$, 看最终 task 是否完成. 只有成功才进 dataset.

**为什么这一步是 critical**:
- 防止学到 "对抗性" recovery action (看起来修对了 pose 但 object 已经掉了)
- 保证了 **distribution over effective corrections** 而非 "plausible-looking corrections"
- 给 training signal 加了 ground-truth 闭环, 类似 RL 中的 reward verification

这让我想到你 Karpathy 在 nanoGPT 训练中强调的 "data quality is everything"——FailSafe 在 data 生成阶段就闭环验证, 而不是依赖 downstream model 自己学.

---

## 6. FailSafe-VLM 架构和训练

### 6.1 模型选择

作者选 LLaVA-OneVision-7B (LLaVA-OV-7B) 作为 base. 检视其 architecture:

- **Language backbone**: Qwen2-7B-Instruct (https://github.com/QwenLM/Qwen2)
- **Vision tower**: SigLIP (https://github.com/google-research/big_vision)
- **Projector**: 2-layer GELU MLP, 2× hidden expansion
- **Feature 来源**: vision encoder 的 penultimate layer (倒数第二层, 这个 trick 值得注意——penultimate layer 通常比 final CLS token 保留更多 spatial info)

LLaVA-OV repo: https://github.com/LLaVA-VL/LLaVA-NeXT

### 6.2 训练 hyperparameters

| 超参 | 值 | 备注 |
|---|---|---|
| GPUs | 32 × H100 | DeepSpeed ZeRO 3 |
| Epochs | 1 | 防止 overfit failure distribution |
| Base LR | $1 \times 10^{-5}$ | |
| Vision tower LR | $2 \times 10^{-6}$ | 比 base 小 5×, 保护 pretrain features |
| LR schedule | cosine decay, 3% warmup | |
| Weight decay | 0 | |
| Precision | bfloat16 / TF32 | |

**Vision tower LR 更小** 这一点很经典——vision encoder 已经在 100M+ image-text pairs 上 pretrain, 不能用大 LR 抹掉, 但又必须 fine-tune 适配 robot observation domain. 这是混合预训练 + 任务适配的标准技巧.

### 6.3 Co-training with RoboPoint VQA mixture

这点是 paper 里被一笔带过但实际很重要的细节. FailSafe-VLM 训练时混入了 RoboPoint VQA 数据 (https://robopoint.github.io/). RoboPoint 是一个 spatial affordance prediction 的 VLM, 训练任务是 "图像中哪里可以放物体 / 哪里可以抓".

**为什么 co-training 有意义**:
- FailSafe 任务需要 spatial reasoning (判断 gripper 是否对齐 cube)
- RoboPoint 提供大量 spatial grounding 监督
- 类似 multi-task learning 中 "auxiliary task 提升 main task" 的思想

---

## 7. 数据集统计深度解读

Table I 的 distribution 透露了 task 结构的 information:

| Task | No-ops | Trans_x | Trans_y | Trans_z | GT |
|---|---|---|---|---|---|
| Pick Cube | 7,485 | 10,575 | 5,295 | **0** | 24,351 |
| Push Cube | 12,057 | 2,394 | 13,947 | 2,385 | 16,893 |
| Stack Cube | 6,693 | 11,511 | 9,792 | **0** | 14,717 |

**直觉解读**:
- **Pick Cube 和 Stack Cube 没有 Trans_z failure**: 因为这两个 task 的关键 motion 在 xy 平面对齐, z 方向偏差通常导致 approach 阶段直接 fail, 不构成 "interesting failure"
- **Push Cube 的 Trans_y 远多于 Trans_x**: pushing 任务对 y 方向偏差敏感 (cube 会被推歪)
- **Push Cube No-ops 最多**: pushing 中 gripper 卡住的情况容易触发
- **Rot 类 failure 在 Pick Cube 中极少** (60-69): 立方体 grasp 对 rotation 不敏感, rotation failure 不会让 task fail

这种 distribution 不是手动设定的, 而是 simulator 自动跑出来后根据 "是否最终失败" 筛选的. 这反映了**任务几何结构与 failure mode 的内在耦合**.

Failure-to-success ratio = 2.3:1, 这个比例比传统 robot dataset 高很多——FailSafe 故意把 negative sample 拉高, 让模型学到 "failures are common, recovery is essential".

---

## 8. 实验结果: 三层泛化能力

### 8.1 VLA 性能提升 (Table II)

| VLA | Baseline | + FailSafe | Δ |
|---|---|---|---|
| π₀-FAST | 78.7% | 82.7% | **+4.0%** |
| OpenVLA | 14.7% | 37.3% | **+22.6%** |
| OpenVLA-OFT | 90.7% | 98.7% | **+8.0%** |

**直觉解读**:
- **OpenVLA 提升最大 (+22.6%)**: 它的 baseline 才 14.7%, headroom 巨大, FailSafe-VLM 救活的 cases 多. OpenVLA 用 discrete action token, 在需要精细 continuous correction 时表现差, FailSafe-VLM 正好补这块.
- **π₀-FAST 提升小 (+4%)**: baseline 78.7% 已经很高, 接近 ceiling. 而且 π₀-FAST 用 flow matching 生成 continuous action, 自身已经比 OpenVLA robust. https://www.physicalintelligence.company/blog/pi0
- **OpenVLA-OFT 中间 (+8%)**: OFT 用 action chunking + regression loss 改进了 OpenVLA, baseline 90.7%, FailSafe 又把它推到 98.7%.

注意 Pick Cube 和 Stack Cube 在 baseline 强时 (π₀-FAST 88%, 96%) 提升为 0, 说明这些 task 已经 saturated. 提升 0 不代表 FailSafe-VLM 失效, 而是没有空间提升.

### 8.2 Cross-object generalization (Table III)

Sphere + Charger 是 training 时从未见过的物体. OpenVLA-OFT 平均提升 +17.4%. 这说明 **failure reasoning 不依赖 object identity, 而是依赖 spatial/geometric patterns**. 这个 generalization 是 VLM 的 spatial inductive bias 带来的福利.

### 8.3 Cross-embodiment generalization (Table IV)

xArm 6 robot (vs training 时 Franka Panda) 上测试, stack cube 从 56% → 76% (+20%), 其他 task 不降. 作者解释: failure scenarios 与 embodiment 解耦, 因为 motion-level failure (translation, rotation, no-op) 是 task-level 抽象, 不绑定具体关节.

这个结论其实挺激进——说明 FailSafe-VLM 学到的是 **task-space failure reasoning**, 而非 joint-space reasoning. 这与 "operational space control" 的思想一致.

### 8.4 与其他 VLM 对比 (Table V)

| Model | Binary Success ↑ | Accuracy ↑ | Cosine Sim ↑ |
|---|---|---|---|
| Qwen2.5-VL | 0.2401 | 0.2401 | 0.0000 |
| Gemini-2.5-flash | 0.6229 | 0.1412 | -0.0121 |
| GPT-4o | 0.7007 | 0.1960 | 0.0117 |
| **FailSafe-VLM** | **0.9094** | **0.8368** | **0.6522** |

**最 striking 的对比**:
- Qwen2.5-VL 完全 broken on this task (cosine = 0, 总是输出 no failure + zero action)
- GPT-4o 能 detect failure (0.70) 但 reasoning 失败 (accuracy 0.196, cosine 0.01)
- FailSafe-VLM accuracy 比 GPT-4o 高 4.3×

**关键 insight**: 大模型 (GPT-4o, Gemini) 的 general VQA 能力无法直接 transfer 到 fine-grained robot failure reasoning. 这块需要 task-specific training data. 这与 "generalist model ≠ specialist model" 的传统 wisdom 一致, 但在 VLA 这个 specific sub-domain 上差距比想象中大.

**Cosine similarity 0.65 够用**: 作者强调, 不需要 cosine = 1.0 因为 action space 是 multi-modal 的, 多个不同的 ΔA 都能 recover 同一 failure. 这与 diffusion policy 的 multi-modality 思想一致.

### 8.5 Inference overhead (Table VI)

| VLA | Speed Δ |
|---|---|
| π₀-FAST | +3.9s |
| OpenVLA | +9.1s |
| OpenVLA-OFT | +3.8s |

延迟主要来自 simulator replanning. 真实部署可以用 real-time action chunking (Black et al. 2025, https://arxiv.org/abs/2506.07339) 减小.

---

## 9. Qualitative Analysis (Figure 4)

Figure 4 显示 OpenVLA 控制 gripper 时, x 轴和 z 轴 trajectory 的 evolution. 关键观察:
- 早期 gripper **近乎 frozen** (clean trajectory training 中没见过)
- FailSafe-VLM 检测到 potential failure, 输出 ΔA 把 gripper 推向 ground-truth pose (绿色段)
- gripper 到达 cube 的 x 位置 (~0.02) 后, 后续 x 偏差不再 critical (因为已经 align 了)
- OpenVLA 接回控制, 完成 lift

这里有个**很重要的 design choice**: FailSafe-VLM 每 10 步接管一次, 而不是每步都 invoke. 这是因为:
1. 推理延迟: VLM forward pass 比 VLA 慢一个数量级
2. 失败 detection 需要看 trajectory 上下文, 单步动作太 noisy
3. 10 步窗口对应大约 0.5-1 秒 (取决于 control frequency), 这是 failure 显现的时间尺度

---

## 10. 与相关工作的联系和我的思考

### 10.1 与 π₀ / Diffusion-VLA 的关系

π₀ 用 flow matching 生成 continuous action, 比 OpenVLA 的 discrete token 强很多. FailSafe 的设计哲学其实和 π₀ 是互补的:
- π₀ 解决 "如何生成 robust action"
- FailSafe 解决 "action 错了怎么 recover"

未来可能的工作: 把 FailSafe-VLM 的 failure reasoning 能力直接嵌入 π₀ 的 flow matching head, 而不是作为 external assistant. 这就接近 "self-correcting VLA" 的 ultimate form.

### 10.2 与 AHA 的关系

AHA (https://aha-vla.github.io/) 和 FailSafe 都用 simulator perturbation 生成 failure data, 但 AHA 只输出 textual reasoning, FailSafe 输出 executable action. 两者的 data generation pipeline 几乎是平行的, 但 output modality 不同.

可以这样理解: AHA 教 VLM "说" 什么是错的, FailSafe 教 VLM "做" 什么是对的. 后者更直接 fit VLA control loop.

### 10.3 与 RoboPoint 的关系

Co-training with RoboPoint 不是偶然. RoboPoint 教 VLM 在图像中 predict spatial affordance (哪里可以 grasp, 哪里可以 place). FailSafe 需要 VLM 判断 gripper 是否对齐 object——这本质是 spatial affordance reasoning. 两者在 spatial reasoning 层面 overlap, 所以 co-training 有正向 transfer.

### 10.4 Karpathy 你的 Software 2.0 视角下的 FailSafe

从你的 Software 2.0 角度看, FailSafe 做的是: **给 Software 2.0 的 robot policy 加上一个 "免疫系统"**. 传统机器人有 explicit exception handler (Software 1.0), 触发条件由工程师写死. FailSafe 把这个 exception handler 也用神经网络表达, 并且用大量失败数据训练. 这是 Software 2.0 的逻辑延伸: 不仅 forward policy 是 learned 的, failure detection + recovery policy 也是 learned 的.

类比到 LLM: 这就像 RLHF 中, SFT 模型 (VLA) + reward model (FailSafe-VLM) 的关系. FailSafe-VLM 给 VLA 提供 "corrective gradient signal", 但是在 inference time 而非 training time. 这有点像 inference-time scaling / test-time compute 的思想, 类似 OpenAI o1 系列.

### 10.5 类比: AlphaGo 的 self-play

FailSafe 的 systematic verification 让我想到 AlphaGo 的 self-play. AlphaGo 用 MCTS 验证 move value, FailSafe 用 motion planner replay 验证 action value. 两者都是:
1. 生成 candidate action
2. 用 environment/simulator 闭环验证
3. 保留有效 candidate, 抛弃无效的

差别: AlphaGo 是 RL self-play, FailSafe 是 supervised + motion planner. 但 "use simulator to verify and filter" 这步是相通的.

---

## 11. Limitations 和未来方向

### 11.1 Paper 承认的 limitation

1. **只支持 motion-level recovery, 没有 object-level recovery**: 比如 "抓错了物体" 这种 failure 无法处理
2. **推理延迟高**: OpenVLA 一次 task 要 121s, 用了 FailSafe 后再加 9s, 占比 ~7.5%
3. **限于 ManiSkill 3 tasks**: pick/push/stack cube, 还是相对简单的 manipulation

### 11.2 我看到的潜在问题

1. **Failure mode 分布偏 simple**: 只用 3 种 basic failure, 真实世界 failure 的 long-tail 远比这复杂 (e.g., 物体 deform, friction 变化, sensor noise). 但作者辩护说复杂 failure 可分解为 basic failure 的组合, 这个 claim 需要更多 empirical support.

2. **Stage-level perturbation vs timestep-level**: FailSafe 在 stage level 扰动, 但真实 VLA 错误是 timestep level 累积的. 两者分布可能有 gap.

3. **Camera view generalization 不完全**: 虽然测试了 novel view, 但只是 "VLA training view", 仍然在 ManiSkill 渲染风格内. Real-world camera view (blur, lighting, occlusion) 会有更大 distribution shift.

4. **No closed-loop on ΔA after deployment**: FailSafe-VLM 输出 ΔA 后, VLA 执行 ΔA, 但没 verify ΔA 是否真的修对了. 如果 ΔA 错了, 没有 fallback. 这可以加一个 closed-loop correction.

### 11.3 我觉得的未来方向

1. **Object-level failure recovery**: 扩展 failure mode 包含 "wrong object grasped", "object dropped", "wrong target location". 这需要 semantic-level reasoning 而不仅是 geometric.

2. **Self-correcting VLA**: 把 FailSafe-VLM 内化为 VLA 的一部分, 用 chain-of-thought reasoning 在 VLA forward pass 内做 self-correction. 这就和 ECoT (Embodied Chain-of-Thought, https://arxiv.org/abs/2407.08693) 殊途同归了.

3. **Real-world transfer**: 现在 sim-only. 需要 sim-to-real 的 domain adaptation. 可能结合 domain randomization 或 differentiable simulation.

4. **Failure data as RL reward**: FailSafe dataset 可以转化为 RL reward signal, 训练 VLA 在 failure-prone 区域自动 generate 鲁棒 action. 这是 RLHF 的 robotics 版本.

5. **Active failure detection**: 当前是固定每 10 步检测, 未来可以 learned 一个 "when to check" policy, 类似 active inference.

---

## 12. 给 Karpathy 你的 build-intuition 总结

| 维度 | FailSafe 的设计 choice | 直觉 |
|---|---|---|
| Failure representation | 3 basic modes (trans/rot/no-op) | 复杂 failure 是 basic mode 的复合, 类似 CNN 的简单 filter 组合表达复杂 pattern |
| Action representation | 7-DoF delta, non-sparse | 给 model soft reasoning 空间, 避免硬决策 |
| Verification | Simulator replay | 用 environment 闭环验证, 拒绝 "看似合理但实际无效" 的 correction |
| Architecture | LLaVA-OV-7B + RoboPoint co-train | VLM 已有 spatial reasoning, fine-tune 适配 robot domain |
| Deployment | 每 10 步接管 | 时机选择对应 failure 显现的时间尺度 |
| Failure:Success ratio | 2.3:1 | 故意 oversample failure, 让模型重视 failure case |

整体上, FailSafe 是 VLA 领域一个相当务实的 paper——它不追求 SOTA on task success (那是 VLA 的事), 而是解决 deployment robustness 这个被忽视的痛点. 它的 contribution 更像 "infrastructure + methodology" 而不是 "new architecture". 这点和你 Karpathy 一直强调的 "data + pipeline > architecture" 思想一致.

唯一让我有点遗憾的是 evaluation 还限于 ManiSkill simulation, 没有真实 robot demo (除了 supplementary video 里提到的). 但作为 conference paper, 已经是相当完整的 story.

### 关键 web links 总结

- FailSafe 项目主页: https://jimntu.github.io/FailSafe/
- LLaVA-OneVision: https://github.com/LLaVA-VL/LLaVA-NeXT
- ManiSkill 3: https://github.com/haosulab/ManiSkill
- OpenVLA: https://github.com/openvla/openvla
- OpenVLA-OFT: https://github.com/openvla/openvla-oft
- π₀ (Physical Intelligence): https://www.physicalintelligence.company/blog/pi0
- AHA: https://aha-vla.github.io/
- RoboFAC: https://arxiv.org/abs/2505.12224
- RoboPoint: https://robopoint.github.io/
- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL
- Real-time action chunking: https://arxiv.org/abs/2506.07339
- Karpathy 你的 nanoGPT (作为相关参考): https://github.com/karpathy/nanoGPT

希望这个解读对你 build intuition 有帮助. 如果你想深入某一节 (比如 systematic verification 的算法伪代码, 或 co-training 的具体 mixture ratio, 或 trajectory 的具体 visualization), 我可以进一步展开.
