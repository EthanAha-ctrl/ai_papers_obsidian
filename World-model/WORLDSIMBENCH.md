---
source_pdf: WORLDSIMBENCH.pdf
paper_sha256: f0bf9f2dd99b7636c6c9994b28925bbfeafd9bcb06b56bff77b3c1e7c60e091d
processed_at: '2026-08-13T05:56:23-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WorldSimBench

## 一句话版本

现在大家都在喊 video generation model 是 "world simulator"，但到底什么算 world simulator？Sora 生成的视频看着很炫，但真的能用来驱动机器人干活吗？这帮人就搞了个 benchmark，发现：**不能，差得远**。

---

## 背景：为什么要搞这个

先说 motivation。现在 predictive models 这个概念太宽泛了，从 GPT 生成 text plan，到 DALL-E 生成 goal image，到 Sora 生成 video，全都叫 "predictive model"。这就有个问题：你连自己在哪个 level 都没搞清楚，怎么评估？

所以他们先搞了个 hierarchy：

- **$S_0$**: 输出 text，比如 LLM 给你写个 plan："第一步去厨房，第二步拿杯子"
- **$S_1$**: 输出 image，比如生成一张 "目标状态" 的图片
- **$S_2$**: 输出 video，但只是 "好看的 video"，没有物理意义
- **$S_3$**: 输出 **actionable video** —— video 里的物理是正确的，3D 是对的，可以翻译成控制信号驱动 agent

$S_3$ 就是你理想中的 world simulator。Ha & Schmidhuber 2018 那篇 "World Models" (https://worldmodels.github.io/) 是这个概念的祖宗，Yang et al. 2023 重新炒热了这个词。

问题是：**现有的 benchmark 全部停在 $S_2$**。VBench (https://vchere.github.io/vbench-web/) 看的是 aesthetic，EvalCrafter (https://evalcrafter.github.io/) 看的是 feature similarity。这些 metric 对 $S_3$ 完全没用——你生成的 video 再好看，如果物理不对，机器人执行就会撞墙。

所以 WorldSimBench 就是为了填这个 gap。

---

## 核心设计：两条腿走路

### 第一条腿：Explicit Perceptual Evaluation（人眼评估）

说白了就是：**让人类看视频打分，然后训练一个 model 模仿人类的打分**。

但不是随便打分。他们设计了一套 hierarchical 的 evaluation dimensions，分三个层面：

**Visual Quality**（视频本身好不好看）：
- Background/Foreground Consistency
- Aesthetics

**Condition Consistency**（视频跟输入 instruction 对不对得上）：
- Instruction Alignment
- Scenario Alignment

**Embodiment**（这是最关键的，传统 benchmark 没有的）：
- **Perspectivity**: 有没有 3D 深度感，光影逻辑对不对
- **Trajectory**: 物体运动轨迹合不合理
- **Embodied Interaction**: 碰撞、抓取、形变这些物理交互对不对
- **Velocity**: 速度变化对不对（比如水里应该游得慢）
- **Safety**（AD 专用）: 有没有闯红灯、逆行
- **Key Element**（AD 专用）: 行人、车辆、红绿灯渲染得好不好

三个 scenario 各自定义不同的 dimension 组合：

| Scenario | Visual Quality | Condition Consistency | Embodiment |
|----------|---------------|----------------------|------------|
| OE (Minecraft) | BC, FC | IA, SA | VC, TJ, EI |
| AD (自动驾驶) | AE | IA | PV, TJ, KE, SF |
| RM (机械臂) | AE, BC, FC | IA | PV, TJ, EI |

然后他们用一堆 video generation model 生成视频，让人类标注员按这些 dimension 打分，同时还要求写 reason（"为什么给这个分"）。

最终搞出了 **HF-Embodied Dataset**，35,701 个 tuples，每个包含 video + instruction + 多维度分数 + human feedback reason。

Dataset 统计：

| Scenario | #instructions | #videos | #dims | #positive | #negative |
|----------|--------------|---------|-------|-----------|-----------|
| OE | 270 | 8,401 | 7 | 121,249 | 79,965 |
| AD | 5 | 15,870 | 6 | 56,768 | 35,044 |
| RM | 2,556 | 11,430 | 7 | 70,672 | 9,338 |

注意 AD 只有 5 个 instruction（前进、后退、左转、右转、停车），因为驾驶的 action space 本身就这么简单。

还有个有趣的细节：**OE 的打分是 binary (1-2)，AD 和 RM 是 1-5**。因为人类看 Minecraft 视频时倾向于 "行/不行" 的二分判断，而驾驶和机械臂视频有更连续的质量梯度。这个 observation 本身就挺有 intuition 的。

---

### 训练 Human Preference Evaluator (HPE)

有了 dataset，下一步就是训练一个自动打分器。他们用的是 **Flash-VStream** (https://github.com/IVGSZ/Flash-VStream)，一个 VideoLLM，然后用 **LoRA** fine-tune。

LoRA 的原理简单说：冻结原 model 的权重 $W_0$，加一个 low-rank 的增量：

$$W = W_0 + BA$$

- $W_0 \in \mathbb{R}^{d \times k}$：原始权重，$d$ 是 input dimension，$k$ 是 output dimension，**冻结不训练**
- $B \in \mathbb{R}^{d \times r}$：可训练矩阵
- $A \in \mathbb{R}^{r \times k}$：可训练矩阵
- $r \ll \min(d, k)$：rank，远小于原始维度

这样只需要训练 $d \times r + r \times k$ 个参数，而不是 $d \times k$ 个。比如 $d=4096, k=4096, r=16$，参数从 16M 降到 130K。

训练配置：
- Video sampling: 每 4 帧取 1 帧
- Optimizer: AdamW
- LR: 2e-5, cosine decay
- Warmup: 0.03
- 4 epochs
- 4× A100 80GB

一个关键工程决策：**训练时不用 human 写的 reason 做 Chain-of-Thought**。因为不同标注员写的 reason 差异太大（"inconsistent trajectory" vs "movement looks weird" vs "not smooth"），model 学不到稳定的 pattern。这个决策很有 intuition——当你 label noise 大的时候，强行让 model 学习 noisy reasoning 反而会 hurt。

HPE 的效果（vs GPT-4o）：

| Scenario | Metric | GPT-4o | HPE | GPT-4o (zero-shot) | HPE (zero-shot) |
|----------|--------|--------|-----|--------------------|-----------------| 
| OE | Acc | 72.8 | **89.4** | 66.5 / 78.5 | **71.6 / 87.9** |
| AD | PLCC | 0.28 | **0.60** | 0.03 / -0.04 | **0.34 / 0.49** |
| RM | PLCC | 0.07 | **0.43** | -0.06 / 0.17 | **0.47 / 0.44** |

PLCC 就是 Pearson 相关系数：

$$\text{PLCC} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

- $x_i$：第 $i$ 个样本的 model 预测分数
- $y_i$：第 $i$ 个样本的人类标注分数
- $\bar{x}, \bar{y}$：各自均值
- $n$：样本总数

GPT-4o 在 zero-shot 评估时居然出现 **negative correlation**（-0.04 在 AD@Lavie），说明 GPT-4o 对 embodied dimension 的判断跟人类是**系统性相反**的。而 HPE 即使 zero-shot 也能保持正相关，说明 fine-tune 学到的是 generalizable 的 embodied understanding，不是 overfitting。

---

### 第二条腿：Implicit Manipulative Evaluation（闭环任务评估）

这是论文最 novel 的部分。思路极其简单粗暴：

**别问视频好不好看，让机器人试着用这个视频干活，看能不能干成。**

Pipeline：
1. 当前 observation + text instruction → World Simulator（video generation model）→ 生成 future video
2. Future video → video-to-action model → action sequence
3. 在 simulation 里执行 action
4. 走 $k$ 步后，重新采样 video（因为预测会 drift）
5. 循环直到任务结束
6. 看 task success rate

三个 scenario：

**OE (Minecraft)**：
- 用 MineRL simulator
- Video-to-action: Steve-1 (https://github.com/Shalev-Lifshitz/Steve-1) 的 goal-based policy
- Action space: 10 个 binary action（W/A/S/D/space/shift/ctrl/E/鼠标左键等）
- 任务：collect wood, collect dirt, collect seeds, travel, dig
- Metric: travel distance, dig depth, item count
- 每个任务 10 trials，每 trial 3000 frames（2.5 分钟）

**AD (自动驾驶)**：
- 用 CARLA simulator (https://carla.org/)
- Benchmark: LangAuto-Tiny（路线 < 150m）
- Video-to-action: LMDrive (https://github.com/hao-shao/LMDrive)
- Metric: Driving Score = Route Completion × Infraction Score
  - RC: 完成路线百分比
  - IS: 违规扣分后的分数
  - DS = RC × IS（主指标）
  - 还有 VC（撞车）, PC（撞人）, LC（撞墙）, RV（闯红灯）, OI（偏离道路）

**RM (机械臂)**：
- 用 CALVIN benchmark (https://calvin.cs.uni-freiburg.de/)
- Robot: Franka Emika Panda, 7-DOF
- Protocol: train on env A/B/C → test on env D (zero-shot)
- Video-to-action: Susie (https://github.com/blacksmithsmith/susie) 的 goal-based policy
- 1000 个 instruction chain，每个 5 个 sequential task
- Metric: 各 task 的 success rate, average task length

---

## 实验结果：video model 离 world simulator 还很远

### Explicit Evaluation 的关键发现

**OE (Table 7)**：几乎所有 model 在 Velocity 上得满分，但这是假象——model 根本不生成动态物体，所以 velocity 没 violation。Embodied Interaction 最难（DynamicCrafter 只有 1.45/2.0），因为 block 破碎、物体形变这种物理太难了。

**AD (Table 8)**：Instruction Alignment 普遍满分（5.0/5.0），因为就 5 个简单命令。但 Perspectivity 和 Key Element 很低——model 生成的视频没有 3D 深度感，行人车辆渲染质量差。OpenSora 最好（Overall 4.40），Perspectivity 4.4, Trajectory 4.8。

**RM (Table 9)**：最 striking 的发现——**Instruction Alignment 极低（1.0-2.6），但 Trajectory 接近满分（5.0）**。这看着矛盾，实际说明了 model 的 failure mode：机械臂动作是平滑的（TJ 高），但完全是 aimless 的乱动（IA 低）。因为没有 targeted movement，也就没有 collision/penetration error，所以 EI 和 TJ 被 "artificially inflated"。这是一个很 subtle 的 metric gaming 现象。

### Implicit Evaluation 的关键发现

**OE (Table 11)**：

最 shocking 的发现：**加 image conditioning 反而降低 performance**。

| Model | Condition | AVG | Travel Dist. |
|-------|-----------|-----|-------------|
| Open-Sora-Plan | Text only | 26.38 | 342.91 |
| Open-Sora-Plan | Text + Image | **10.28** | **195.14** |
| DynamicCrafter | Text + Image | 4.06 | 130.04 |
| EasyAnimate | Text + Image | 4.84 | 157.12 |

Text-only 的 OpenSora 最好（AVG 27.80），但加 first frame 后 DynamicCrafter 和 EasyAnimate 几乎完不成任何任务。这说明 **当前 model 无法有效 fuse 多个 conditional inputs**，image condition 干扰了物理规则和 3D scene 的生成。

**AD (Table 12)**：

| Model | DS(↑) | RC(↑) | VC(↓) | LC(↓) |
|-------|-------|-------|-------|-------|
| Open-Sora-Plan | 31.05 | 38.25 | 2.40 | 4.40 |
| DynamicCrafter | 24.49 | 37.19 | 5.03 | 4.90 |
| EasyAnimate | 17.41 | 28.48 | **0.00** | **29.34** |

EasyAnimate 的 VC=0（没撞车），但 LC=29.34（疯狂撞墙），DS 最低。这说明 "不撞车" 不等于开得好——model 生成的视频让 agent 一直在墙上蹭。

**RM (Table 13)**：

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len |
|-------|--------|--------|--------|--------|--------|---------|
| Open-Sora-Plan | 0.85 | 0.70 | 0.60 | 0.40 | 0.40 | 2.95 |
| DynamicCrafter | **0.95** | 0.75 | 0.55 | 0.25 | 0.25 | 2.75 |
| EasyAnimate | 0.90 | 0.60 | 0.35 | 0.10 | 0.10 | 2.05 |

Success rate 随 task 复杂度近乎 linearly 下降。DynamicCrafter 在 task 1 最好（0.95），但 task 4-5 掉到 0.25。Open-Sora-Plan 虽然单 task 略低，但 long-horizon 更稳定（task 4-5 保持 0.40）。

---

## Explicit vs Implicit 的一致性

大部分结论一致：trajectory 生成好的 model（DynamicCrafter），在 trajectory-focused scenario（AD, RM）的闭环任务中也好。

但有有趣的分歧：DynamicCrafter 在 Explicit evaluation 中 Overall 更高，但在 OE（需要频繁交互）和 RM 的 long-sequence task（4, 5）中，反而不如 Open-Sora-Plan。

原因：**Explicit evaluation 衡量 "单次生成质量"，Implicit evaluation 衡量 "持续生成稳定性"**。一个 model 可能单次生成很漂亮，但 closed-loop 反复采样时 inconsistency 会 accumulate，导致 long-horizon 任务失败。

这个 insight 对未来 world simulator 的设计很重要：**temporal consistency 比 single-frame quality 更关键**。

---

## 所以结论是什么

1. **当前的 video generation model 本质上是 2D pixel predictor，不是 3D world model**。它们学的是 pixel-level statistical pattern，不是 physical law。

2. **主要缺陷**：
   - 物理交互（碰撞、形变、抓取）几乎完全不行
   - 3D 深度感和 perspective 很差
   - Instruction following 差（尤其机械臂，生成 aimless 动作）
   - Long-horizon 不稳定，error 会 accumulate
   - Multi-condition fusion 有问题（加 image 反而变差）

3. **要成为真正的 world simulator，可能需要**：
   - Explicit physics integration（在架构里加 physics engine）
   - 3D native representation（NeRF, 3D Gaussian, voxel）
   - Action-conditioned generation（让 model 接受 action input）
   - Hierarchical planning + memory（处理 long sequence）

4. **评估方法论的意义**：WorldSimBench 提出了 "actionability-as-evaluation" 范式——不用 ground truth video，而是用 closed-loop task success 来 implicitly 评估。这可能是未来 world model 评估的标准范式。

---

## 相关链接

**这篇 paper**：
- Project page: https://iranqin.github.io/WorldSimBench.github.io/

**World Simulator 概念**：
- World Models (Ha & Schmidhuber 2018): https://worldmodels.github.io/
- Sora technical report: https://openai.com/sora/
- Vista (自动驾驶 world model): https://github.com/hologerance/Vista

**被评估的 Video Generation Models**：
- OpenSora: https://github.com/hpcaitech/Open-Sora
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Lavie: https://github.com/YaohuiW/LaVie
- DynamicCrafter: https://github.com/Doubiiu/DynamiCrafter
- AnimateDiff: https://github.com/guoyww/animatediff
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate

**Simulation Environments**：
- MineRL (Minecraft): https://minerl.io/
- CARLA (自动驾驶): https://carla.org/
- CALVIN (机械臂): https://calvin.cs.uni-freiburg.de/

**Video-to-Action Models**：
- Steve-1: https://github.com/Shalev-Lifshitz/Steve-1
- LMDrive: https://github.com/hao-shao/LMDrive
- Susie: https://github.com/blacksmithsmith/susie

**VideoLLM (HPE base model)**：
- Flash-VStream: https://github.com/IVGSZ/Flash-VStream

**其他相关 benchmark**：
- VBench: https://vchere.github.io/vbench-web/
- EvalCrafter: https://evalcrafter.github.io/
- AgentBench: https://thudm.github.io/AgentBench/
- EgoPlan-Bench: https://gary3410.github.io/EgoPlan-Bench/

---

# WorldSimBench 深度解析：视频生成模型作为 World Simulator 的评估范式

## 1. 核心思想与 Motivation

这篇 paper 的核心贡献在于提出了一个 **predictive models 的 hierarchy 分类体系**，并针对其中最高阶段 $S_3$（World Simulators）设计了第一个系统性的评估 benchmark。理解这篇论文的 intuition 在于：当前的 video generation models 虽然在 aesthetics 上已经相当成熟，但当我们试图把它们当作 "world simulator" 来用——即用生成的 video 来驱动 embodied agent 的 action——就会发现这些 video 在物理规则、3D 场景理解、actionability 上存在根本性缺陷。传统的 benchmark（如 VBench、EvalCrafter）只关注 feature similarity 或 aesthetic quality，完全无法捕捉这种 "actionability" 的 gap。

Paper 的 hierarchy 定义如下：

| Stage | Output Modality | Embodiment Level | 代表性 Benchmark |
|-------|----------------|-----------------|------------------|
| $S_0$ | Text | 最低 | AgentBench, EgoPlan-Bench, MMWorld, VAB |
| $S_1$ | Image | 中低 | LEGO |
| $S_2$ | Video | 中 | VBench, EvalCrafter |
| $S_3$ | Actionable Video | 最高 | **WorldSimBench** (本文) |

$S_3$ 阶段的关键特征是：生成的 video 必须 integrate robust 3D scene understanding 和 physical rule priors，从而能被翻译成 executable actions。这正是 Sora 这类 model 试图达到但尚未完全实现的目标。

Project page: https://iranqin.github.io/WorldSimBench.github.io

## 2. 双重评估框架的架构解析

WorldSimBench 的核心设计哲学是 **dual evaluation**：既从 human perception 的角度显式评估 visual fidelity，又从 closed-loop embodied task 的角度隐式评估 actionability。这两个维度是互补的，单独任何一个都无法全面评估 World Simulator。

### 2.1 Explicit Perceptual Evaluation

这部分的核心是构建一个 **Hierarchical Evaluation Dimension** 体系，分为三个层面：

**Visual Quality**:
- OE: Background Consistency (BC), Foreground Consistency (FC)
- AD: Aesthetics (AE)  
- RM: Aesthetics (AE), Background Consistency (BC), Foreground Consistency (FC)

**Condition Consistency**:
- OE: Instruction Alignment (IA), Scenario Alignment (SA)
- AD: Instruction Alignment (IA)
- RM: Instruction Alignment (IA)

**Embodiment**（这是区别于传统 video benchmark 的关键）:
- OE: Velocity (VC), Trajectory (TJ), Embodied Interaction (EI)
- AD: Perspectivity (PV), Trajectory (TJ), Key Element (KE), Safety (SF)
- RM: Perspectivity (PV), Trajectory (TJ), Embodied Interaction (EI)

这些 embodiment 维度的设计很有 intuition：**Perspectivity** 评估 3D 深度感和光影逻辑，这直接关系到 video 是否能提供有效的 depth cue 给下游 controller；**Trajectory** 评估物体运动的逻辑性，这关系到生成的 waypoint 是否 physically feasible；**Embodied Interaction** 评估碰撞和交互时的形变是否符合物理规则，这关系到 grasp、push 等动作的仿真保真度。

### 2.2 HF-Embodied Dataset 构建

Dataset 的统计信息（Table 4）：

| Scenario | #instructions | #videos | #dims | #actions | #positive | #negative |
|----------|--------------|---------|-------|----------|-----------|-----------|
| OE | 270 | 8,401 | 7 | 11 | 121,249 | 79,965 |
| AD | 5 | 15,870 | 6 | 5 | 56,768 | 35,044 |
| RM | 2,556 | 11,430 | 7 | 26 | 70,672 | 9,338 |

总共 35,701 个 tuples，每个 tuple 包含 video、text instruction、multi-dimensional scores、以及 fine-grained human feedback（reason）。

AD 的 instruction 只有 5 个（move forward, backward, turn left, turn right, stop），这是因为 autonomous driving 的 action space 本身就是高度离散化和结构化的。而 OE 和 RM 的 instruction 更多样，因为 task space 更丰富。

值得注意的是 **OE 的 scoring range 是 1-2**（binary perception），而 AD 和 RM 是 1-5。这反映了 human 在评估不同场景时的认知模式差异：Minecraft 的 video 质量更像是 "对/错" 的 binary 判断，而 driving 和 manipulation 的质量有更连续的梯度。

### 2.3 Human Preference Evaluator (HPE)

HPE 基于 **Flash-VStream**（一个 VideoLLM），通过 **LoRA** fine-tuning 训练。这里的关键技术细节：

LoRA 的核心思想是冻结预训练权重 $W_0$，引入 low-rank decomposition：

$$W = W_0 + \Delta W = W_0 + B A$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 是冻结的预训练权重，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$ 是 rank。训练时只更新 $B$ 和 $A$。上标 $d$ 表示 hidden dimension，$k$ 表示 output dimension，$r$ 是 LoRA rank（通常远小于 $d$ 和 $k$）。

训练设置：
- Video sampling frequency: 4（即每 4 帧采样 1 帧）
- Optimizer: AdamW
- Learning rate: 2e-5
- Scheduler: cosine decay
- Warmup ratio: 0.03
- Epochs: 4
- Hardware: 4× A100 80GB

Prompt template 示例（Figure 6）：
```
<Video>
The given autonomous driving video is generated by a generative model based on the input instruction: {instruction}. Please rate the video based on the following criteria:
{Dimension}: {Dimension Explanation}
```

一个重要的设计决策：**训练时不使用 annotated reason 做 CoT (Chain-of-Thought)**，因为 different annotator 的 reason varies a lot，model 难以 learn。这是一个很有 intuition 的工程决策——当 label noise 很大时，强行让 model 学习 noisy 的 reasoning 反而会 hurt performance。

### 2.4 Implicit Manipulative Evaluation

这是论文最 novel 的部分。核心思路：**把 World Simulator 当作 low-level decision maker**，通过 video-to-action model 把生成的 video 转换成 control signal，然后在 closed-loop simulation 中执行，用 task success rate 来 implicitly 评估 video quality。

Pipeline 如下（Figure 3）：
1. Current observation + text instruction → World Simulator → predicted future video
2. Predicted video → pre-trained video-to-action model (IDM or goal-based policy) → action sequence
3. Execute action for k timesteps in simulation
4. After k timesteps, refresh prediction（重新采样 video）
5. Repeat until task termination

三个 scenario 的 simulation platform：

**OE (Open-Ended Embodied Environment)**:
- Simulator: MineRL (Minecraft)
- Observation: RGB images
- Action space: keyboard + mouse controls（Table 10 列出了 10 个 binary actions：forward, back, left, right, jump, inventory, sneak, sprint, attack）
- Video-to-action model: Steve-1 (goal-based policy)
- Metrics: travel distance (X-Z plane max displacement), dig depth (Y axis max displacement), log/seed/dirt item count
- Testing: 10 trials per task, 3000 frames (2.5 min) per trial

**AD (Autonomous Driving)**:
- Simulator: CARLA
- Benchmark: LangAuto-Tiny (route < 150m)
- Video-to-action model: LMDrive
- Metrics: 8 个指标
  - Route Completion (RC): 完成路线的百分比
  - Infraction Score (IS): 违规惩罚后的分数
  - **Driving Score (DS) = RC × IS**（这是 primary ranking metric）
  - Vehicle Collisions (VC), Pedestrian Collisions (PC), Layout Collisions (LC), Red Light Violations (RV), Offroad Infractions (OI)

**RM (Robot Manipulation)**:
- Simulator: CALVIN (4 environments A, B, C, D)
- Protocol: train on A, B, C → test on D (zero-shot)
- Robot: Franka Emika Panda, 7-DOF
- Video-to-action model: Susie (goal-based policy)
- Testing: 1000 instruction chains, 每个 chain 包含 5 个 sequential tasks
- Metrics: success rate at task 1-5, average task length completed

## 3. 实验结果深度分析

### 3.1 HPE vs GPT-4o (Table 3, 6)

Table 3 的核心发现：

| Scenario | Metric | GPT-4o | HPE | GPT-4o@OpenSora | HPE@OpenSora | GPT-4o@Lavie | HPE@Lavie |
|----------|--------|--------|-----|-----------------|--------------|--------------|-----------|
| OE | Acc(↑) | 72.8 | 89.4 | 66.5 | 71.6 | 78.5 | 87.9 |
| AD | PLCC(↑) | 0.28 | 0.60 | 0.03 | 0.34 | -0.04 | 0.49 |
| RM | PLCC(↑) | 0.07 | 0.43 | -0.06 | 0.47 | 0.17 | 0.44 |

**PLCC (Pearson Linear Correlation Coefficient)** 的公式：

$$\text{PLCC} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

其中 $x_i$ 是 model $i$ 预测的 score，$y_i$ 是 human 标注的 score，$\bar{x}$ 和 $\bar{y}$ 是各自的均值，$n$ 是样本数。PLCC 范围是 $[-1, 1]$，越接近 1 表示正相关越强。

关键发现：**GPT-4o 在 zero-shot 评估 OpenSora (AD) 和 Lavie (RM) 时出现 negative correlation**（-0.04 和 0.17，其中 -0.04 接近 0 但实际是负相关）。这说明 GPT-4o 对 video quality 的 judgment 和 human perception 存在系统性偏差，尤其是在 embodied dimensions 上。而 HPE 即使在 zero-shot setting 下也保持正相关性，证明了 fine-tuning 带来的 generalization。

Table 6 的 per-dimension 分析更细致：在 RM 的 Trajectory 和 Embodied Interaction 维度，GPT-4o 显示 negative correlation（-0.01 和 -0.14），而 HPE 显示强正相关（0.56 和 0.43）。这说明 GPT-4o 完全无法理解物理交互的合理性，而这正是 World Simulator 最关键的 capability。

### 3.2 Explicit Perceptual Evaluation 结果 (Tables 7-9, Figure 4)

**OE (Table 7)**，scoring range 1-2：

| Model | BC | FC | IA | SA | VC | TJ | EI | Overall |
|-------|-----|-----|-----|-----|-----|-----|-----|---------|
| ModelScope | 1.9 | 2.0 | 2.0 | 1.7 | 2.0 | 2.0 | 1.75 | **1.91** |
| DynamicCrafter | 1.9 | 2.0 | 1.5 | 2.0 | 2.0 | 2.0 | 1.45 | 1.84 |
| Lavie | 1.3 | 2.0 | 1.7 | 1.7 | 2.0 | 2.0 | 1.8 | 1.79 |
| OpenSora | 1.6 | 1.9 | 1.6 | 1.8 | 2.0 | 2.0 | 1.6 | 1.79 |
| Open-Sora-Plan | 1.4 | 1.9 | 1.7 | 1.7 | 2.0 | 1.5 | 1.6 | 1.69 |
| EasyAnimate | 1.4 | 1.8 | 1.5 | 2.0 | 2.0 | 1.22 | 1.45 | 1.62 |
| AnimateDiff | 1.3 | 1.3 | 1.2 | 1.7 | 1.4 | 1.38 | 1.55 | **1.40** |

关键 insight：**几乎所有 model 在 Velocity (VC) 上都得满分 2.0**，但这不是因为 model 真的生成了正确的 velocity，而是因为 "limited occurrences of object movement"——model 根本不生成动态内容，所以 velocity维度没有 violation。这是一个典型的 "trivially satisfying the metric" 问题。

**Embodied Interaction (EI) 是最难的维度**：DynamicCrafter 只有 1.45，EasyAnimate 1.45。这是因为 block shattering、object deformation 等物理交互极其复杂，需要 model 理解材质属性、碰撞力学等。

**AD (Table 8)**，scoring range 1-5：

| Model | AE | IA | PV | TJ | KE | SF | Overall |
|-------|-----|-----|-----|-----|-----|-----|---------|
| OpenSora | 3.55 | 5.0 | 4.4 | 4.8 | 3.65 | 5.0 | **4.40** |
| ModelScope | 2.8 | 5.0 | 3.35 | 4.0 | 3.0 | 5.0 | 3.86 |
| DynamicCrafter | 2.6 | 4.0 | 3.4 | 3.8 | 2.65 | 5.0 | 3.57 |
| Lavie | 2.15 | 5.0 | 2.2 | 2.8 | 2.1 | 5.0 | 3.21 |
| Open-Sora-Plan | 1.6 | 5.0 | 1.55 | 1.4 | 1.45 | 3.2 | 2.37 |
| AnimateDiff | 1.55 | 5.0 | 1.55 | 1.0 | 1.3 | 3.8 | 2.37 |
| EasyAnimate | 1.5 | 3.4 | 1.4 | 1.4 | 1.3 | 2.6 | **1.93** |

Insight：**IA (Instruction Alignment) 普遍得满分 5.0**（除了 EasyAnimate 和 DynamicCrafter），这是因为 AD 的 instruction 只有 5 种简单命令（forward, backward, left, right, stop），model 很容易 align。但这掩盖了一个事实：虽然 instruction align 了，但 **Perspectivity (PV) 和 Key Element (KE) 得分很低**——model 生成的 video 缺乏 3D 深度感，行人和车辆等关键元素渲染质量差。

OpenSora 在 AD 上表现最好（4.40），尤其在 Perspectivity (4.4) 和 Trajectory (4.8) 上领先，这可能与 OpenSora 的 Diffusion Transformer 架构和大规模 training data 有关。

**RM (Table 9)**，scoring range 1-5：

| Model | AE | BC | FC | IA | PV | TJ | EI | Overall |
|-------|-----|-----|-----|-----|-----|-----|-----|---------|
| DynamicCrafter | 3.97 | 4.08 | 4.0 | 2.6 | 5.0 | 5.0 | 4.31 | **4.14** |
| Lavie | 3.8 | 3.9 | 4.0 | 1.8 | 4.95 | 5.0 | 4.1 | 3.94 |
| OpenSora | 3.85 | 4.0 | 3.95 | 1.3 | 4.75 | 5.0 | 4.1 | 3.85 |
| Open-Sora-Plan | 4.0 | 4.0 | 4.0 | 1.0 | 4.9 | 5.0 | 4.0 | 3.84 |
| ModelScope | 3.63 | 4.1 | 4.0 | 1.18 | 4.9 | 5.0 | 4.0 | 3.83 |
| AnimateDiff | 3.8 | 3.9 | 4.0 | 1.0 | 4.95 | 5.0 | 4.1 | 3.82 |
| EasyAnimate | 3.55 | 3.45 | 3.65 | 1.2 | 4.8 | 4.3 | 3.45 | **3.49** |

最 striking 的发现：**IA (Instruction Alignment) 得分极低**（1.0-2.6），而 TJ (Trajectory) 得分接近满分 5.0。这看似矛盾，实则揭示了 model 的 failure mode：**model 生成的 robotic arm 动作是 "aimless" 的**——轨迹本身是平滑合理的（所以 TJ 高），但完全不遵循 instruction（所以 IA 低）。由于没有 targeted movement，也就没有 object interaction 或 penetration errors，导致 EI 和 TJ 被 "artificially inflated"。

### 3.3 Implicit Manipulative Evaluation 结果 (Tables 11-13, Figure 5)

**OE (Table 11)**：

Text-only condition：
| Model | AVG | Collect Wood | Collect Dirt | Collect Seed | Travel Dis. | Dig Depth |
|-------|-----|-------------|-------------|-------------|-------------|-----------|
| OpenSora | 27.80 | 21.20 | 70.20 | 10.40 | 339.87 | 3.20 |
| Open-Sora-Plan | 26.38 | 19.90 | 50.20 | 7.30 | 342.91 | 20.20 |
| Lavie | 26.06 | 23.50 | 56.00 | 11.60 | 270.20 | 12.20 |
| ModelScope | 21.05 | 14.00 | 52.20 | 6.30 | 240.72 | 8.70 |
| AnimateDiff | 13.10 | 7.40 | 22.90 | 3.30 | 274.19 | 4.50 |

Text & Image condition：
| Model | AVG | Collect Wood | Collect Dirt | Collect Seed | Travel Dis. | Dig Depth |
|-------|-----|-------------|-------------|-------------|-------------|-----------|
| Open-Sora-Plan | 10.28 | 11.10 | 12.50 | 2.60 | 195.14 | 5.70 |
| EasyAnimate | 4.84 | 0.20 | 0.70 | 1.70 | 157.12 | 5.90 |
| DynamicCrafter | 4.06 | 0.40 | 0.30 | 1.30 | 130.04 | 5.30 |

**Critical finding：image conditioning 反而降低 performance**。Open-Sora-Plan 从 text-only 的 26.38 降到 text & image 的 10.28。这看似反直觉——加入 first frame conditioning 应该提供更多 information——但实际上说明 **model 在处理 multiple conditional inputs 时存在冲突**，image condition 干扰了 model 对物理规则和 3D scene 的生成能力。

**AD (Table 12)**：

| Model | DS(↑) | RC(↑) | IS(↑) | VC(↓) | PC(↓) | LC(↓) | RV(↓) | OI(↓) |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Open-Sora-Plan | 31.054 | 38.249 | 0.767 | 2.400 | 0.000 | 4.401 | 1.133 | 3.514 |
| DynamicCrafter | 24.491 | 37.189 | 0.599 | 5.030 | 0.000 | 4.896 | 0.937 | 3.221 |
| EasyAnimate | 17.414 | 28.475 | 0.607 | 0.000 | 0.000 | 29.344 | 0.000 | 1.690 |

**Driving Score (DS) = Route Completion (RC) × Infraction Score (IS)**

Open-Sora-Plan 以 DS=31.054 领先。有趣的是 EasyAnimate 的 VC=0（无车辆碰撞）但 DS 最低，因为它的 LC=29.344（layout collisions 极高）——model 生成的 video 导致 agent 不断撞墙。这说明 **单纯的 "安全"（不撞车）并不等于好的 driving**，还需要 route completion 和整体驾驶质量。

**RM (Table 13)**：

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len. |
|--------|--------|--------|--------|--------|--------|-----------|
| Open-Sora-Plan | 0.85 | 0.70 | 0.60 | 0.40 | 0.40 | 2.95 |
| DynamicCrafter | 0.95 | 0.75 | 0.55 | 0.25 | 0.25 | 2.75 |
| EasyAnimate | 0.90 | 0.60 | 0.35 | 0.10 | 0.10 | 2.05 |

Success rate 随 task complexity 递减的模式很有 intuition：从 task 1 到 task 5，success rate 近乎 linearly 下降。这反映了 **long-horizon planning 的困难**——每个 task 的 error 会 accumulate，导致后续 task 的 success rate 指数级下降。

DynamicCrafter 在 task 1 上最高（0.95），但在 task 4-5 上降到 0.25。Open-Sora-Plan 虽然在 task 1 上略低（0.85），但在 task 4-5 上保持 0.40，说明它的 **robustness 更好**，生成的 video 质量更稳定。

## 4. Explicit vs Implicit Evaluation 的一致性分析

论文的一个关键发现是 **两个 evaluation 的结论大部分一致，但存在有趣的分歧**：

**一致的部分**：
- DynamicCrafter 在 Explicit Perceptual Evaluation 中 Trajectory 表现好，在 Implicit Manipulative Evaluation 的 AD 和 RM（trajectory-focused scenarios）中也表现出色。
- OpenSora 在 AD 的 Explicit evaluation 中 Overall 最高，在 Implicit evaluation 中也表现优异。

**分歧的部分**：
- 在 OE（需要 frequent interactions）和 RM 的 long-sequence tasks（4, 5）上，DynamicCrafter 的 Implicit performance 不如 Open-Sora-Plan，尽管 Explicit evaluation 中 DynamicCrafter 更好。
- 原因：这些 tasks demand stable, high-quality video generation for guidance，Open-Sora-Plan 在 robustness 上更优。

这个分歧说明：**Explicit evaluation 衡量的是 "单次生成质量"，而 Implicit evaluation 衡量的是 "持续生成稳定性"**。一个 model 可能在单次生成上表现优异，但在 closed-loop 的反复采样中由于 inconsistency 而表现不佳。这是 World Simulator 评估的一个关键 insight。

## 5. 技术细节与 Implementation

### 5.1 Video Generation Model Finetuning (Table 5)

| Model | Short Videos (frames) | Long Videos (frames) |
|-------|----------------------|---------------------|
| Open-Sora-Plan | 16 | 64 |
| Lavie | 16 | 48 |
| ModelScope | 16 | 60 |
| OpenSora | 16 | 48 |
| AnimateDiff | 16 | 64 |
| DynamicCrafter | 16 | 60 |
| EasyAnimate | 16 | 64 |

两种 video length 用于增强 evaluation set 的多样性。Short videos ~20 frames，long videos ~60 frames。

**Open-Sora-Plan (TI2V) 的架构修改**：基于 DynamicCrafter 的设计，将 first frame 作为 condition 并扩展 channel dimensions，使 model 能接受 first frame 作为输入。这是一个 text+image-to-video 的架构，用于支持 OE 和 RM 中需要 first frame conditioning 的场景。

### 5.2 Training Datasets

- **OE**: OpenAI Contractor Gameplay Dataset (VPT)，human contractor 玩 Minecraft 的录制数据，包含 keypresses 和 mouse movements。还创建了 supplementary "Explore" dataset，通过 multiple pre-trained Steve-1 agents 生成 trajectories，并随机切换 model、重置 memory、调整 orientation 来增强 distribution diversity。
- **AD**: nuScenes training set，按 Vista 的方法采样 25 frames @ 10 Hz 的 video clips。Ego-vehicle commands 分为 turn right, turn left, go straight, stop。
- **RM**: RH20T-P，基于 RH20T 的 primitive-level robotic manipulation dataset，包含 meticulously defined primitive skills 和 spatial knowledge。排除了包含 explicit coordinate information 的 instructions 以增强 generalization。

## 6. 关键 Insights 与未来方向

### 6.1 Current World Simulators 的主要缺陷

1. **物理规则理解不足**：尤其是 Embodied Interaction（block shattering, object deformation）和 Velocity（动态物体运动）。
2. **3D 场景表示弱**：Perspectivity 得分普遍低，model 生成的 video 缺乏 depth cue。
3. **Instruction following 差**：尤其在 RM 中，model 生成 aimless actions 而非 targeted manipulation。
4. **Long-horizon 不稳定**：success rate 随 task 复杂度快速下降，说明 model 缺乏 temporal consistency。
5. **Multi-condition 处理弱**：image conditioning 反而降低 performance，说明 model 无法有效 fuse 多个 conditional inputs。

### 6.2 评估方法论的意义

WorldSimBench 的核心贡献不仅是 benchmark 本身，更是提出了一种 **"actionability-as-evaluation"** 的范式：通过 closed-loop task performance 来 implicitly 评估 video generation quality。这种方法有几个 advantage：

1. **无需 ground truth video**：embodied task 没有 definite ground truth for actionable video，传统 feature similarity 方法失效。
2. **捕捉 physical plausibility**：如果生成的 video 物理不合理，agent 执行时会碰撞、失败，直接反映在 task success rate 上。
3. **捕捉 temporal consistency**：closed-loop 需要反复采样 video，inconsistency 会 accumulate 并导致 failure。

### 6.3 与 Sora 等 Frontier Model 的关系

这篇 paper 发表时（2024年），Sora 已经展示了 impressive 的 video generation 能力，但 OpenAI 自己也承认 Sora 在 "complex physics" 和 "long-term consistency" 上存在局限。WorldSimBench 的评估结果从 academic 角度验证了这一点：即使是最好的 model（OpenSora、Open-Sora-Plan）在 Embodied Interaction 和 long-horizon tasks 上也表现不佳。

这指向一个根本性问题：**当前的 video generation models 本质上是 2D pixel predictors，而非 3D world models**。它们学习的是 pixel-level 的 statistical patterns，而非 physical laws。要成为真正的 World Simulator，可能需要：

1. **Explicit physics integration**：在架构中加入 physics engine 或 neural physics simulator。
2. **3D representation**：从 2D pixel space 转向 3D scene representation（如 NeRF, 3D Gaussians, voxel grids）。
3. **Action-conditioned generation**：让 model 接受 action input 并生成 action-conditioned video，而非仅仅 text-conditioned。
4. **Long-horizon planning**：引入 hierarchical planning 或 memory mechanism 来处理 long sequences。

## 7. Limitations 与 Critique

Paper 自己指出的 limitation：World Simulator 的应用场景远不止 robots，不同场景有不同的 physical representations，如何有效评估其他场景需要更多探索。

我认为还有几个值得讨论的点：

1. **Video-to-action model 的 bottleneck**：Implicit evaluation 依赖 pre-trained video-to-action model（Steve-1, LMDrive, Susie）。如果这些 model 本身有 limitation，评估结果可能被 confound。Paper 没有详细分析 video-to-action model 的 error 如何 propagate 到最终 metric。

2. **Human annotation 的 bias**：HF-Embodied Dataset 的 human annotation 可能存在 annotator bias，尤其是对于 "Embodied Interaction" 这种主观性较强的维度。Paper 提到 reason varies a lot，但没报告 inter-annotator agreement。

3. **Benchmark coverage**：三个 scenario（OE, AD, RM）虽然 representative，但 missing 了一些重要场景，如 humanoid locomotion、dexterous manipulation、multi-agent interaction。

4. **Computational cost**：Implicit Manipulative Evaluation 需要 closed-loop simulation，每轮评估要跑 10 trials × 3000 frames（OE），或 1000 instruction chains × 5 tasks（RM），computational cost 相当高。这可能限制了 benchmark 的 scalability。

## 8. 相关工作与 Reference Links

### Video Generation Models 评估
- VBench: https://vchere.github.io/vbench-web/
- EvalCrafter: https://evalcrafter.github.io/

### Embodied Agent 评估
- AgentBench: https://thudm.github.io/AgentBench/
- EgoPlan-Bench: https://gary3410.github.io/EgoPlan-Bench/
- MMWorld: https://github.com/acleoai/MMWorld
- VisualAgentBench (VAB): https://vab-um.github.io/

### Video Generation Models
- OpenSora: https://github.com/hpcaitech/Open-Sora
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Lavie: https://github.com/YaohuiW/LaVie
- ModelScope: https://github.com/modelscope/modelscope
- AnimateDiff: https://github.com/guoyww/animatediff
- DynamicCrafter: https://github.com/Doubiiu/DynamiCrafter
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate

### Simulation Environments
- MineRL: https://minerl.io/
- CARLA: https://carla.org/
- CALVIN: https://calvin.cs.uni-freiburg.de/

### Video-to-Action Models
- Steve-1: https://github.com/Shalev-Lifshitz/Steve-1
- LMDrive: https://github.com/hao-shao/LMDrive
- Susie: https://github.com/blacksmithsmith/susie

### VideoLLM
- Flash-VStream: https://github.com/IVGSZ/Flash-VStream

### World Simulator 概念
- World Models (Ha & Schmidhuber): https://worldmodels.github.io/
- Sora (OpenAI): https://openai.com/sora/
- Vista (Autonomous Driving World Model): https://github.com/hologerance/Vista

## 9. 总结

WorldSimBench 是第一个系统性地评估 World Simulator 的 benchmark，其核心创新在于：

1. **Predictive model hierarchy** ($S_0$-$S_3$)：为 predictive models 的 embodiment 程度提供了清晰的分类框架。
2. **Dual evaluation framework**：Explicit Perceptual Evaluation（human-aligned video scoring）+ Implicit Manipulative Evaluation（closed-loop task performance），两者互补。
3. **HF-Embodied Dataset**：35,701 个 fine-grained human feedback tuples，覆盖 20 个 embodied dimensions，可用于训练更 aligned 的 video evaluator。
4. **Actionability-as-evaluation paradigm**：通过 video-to-action 的 closed-loop 评估，捕捉传统 metric 无法衡量的 physical plausibility 和 temporal consistency。

实验结果表明，当前的 video generation models 距离真正的 World Simulator 还有显著差距，尤其是在 physical interaction、3D understanding、instruction following 和 long-horizon stability 上。这篇 paper 为未来的 World Simulator 研究提供了一个 rigorous 的 evaluation foundation，也指明了需要改进的关键方向。
