---
source_pdf: Nav-R1 Reasoning and Navigation in Embodied Scenes.pdf
paper_sha256: 8c3ef9965ad51e174ef238c24a7b2a96f34fc2b0bd198d27fe7ec6fae712f981
processed_at: '2026-08-05T21:55:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Nav-R1 用人话讲一遍

Andrej, 没问题, 我换个更接地气的讲法, 但核心技术细节我都保留, 帮你 build intuition。

---

## 这篇 paper 在解决什么现实问题?

想象你让一个 robot 去厨房拿可乐。这个 robot 要做的事情其实特别复杂:

1. 它得先 "看懂" 房间长什么样 (perception)
2. 它得 "理解" "去厨房拿可乐" 是什么意思 (reasoning)
3. 它得 "规划" 怎么走 (planning)
4. 它得实时 "迈步" 别撞墙 (control)

这四件事以前是分开做的, 各有各的 model。但分开做有个大问题: perception 的结果 reasoning 接不住, reasoning 的结论 planning 用不上, planning 的路径 control 执行不了。就像四个部门各干各的, 没人对接。

Nav-R1 想做的事情就是: **把这四件事塞进一个 model 里**, 让它们自然协同。

参考下 Fig. 1, 它把 dialogue, reasoning, planning, navigation 全部 unify 了: [paper GitHub](https://github.com/AIGeeksGroup/Nav-R1)

---

## 为什么这件事很难? 两个核心矛盾

### 矛盾 1: Reasoning 要深, 但 Action 要快

你要让 robot 真正 "想清楚" 房间布局、指令含义、历史路径, 这个计算量很大, 慢。你要让 robot 实时避障、转弯、停步, 这个要求毫秒级响应, 快。

你用一个 model 同时干这两件事, 要么想得太浅 (reasoning 不够), 要么走得太慢 (control 来不及)。

这其实就是 Kahneman ([Thinking Fast and Slow](https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555)) 说的 System 1 和 System 2 的分工问题。人也一样: 你走路不需要思考 (System 1, fast), 但你在陌生商场找厕所得停下来想想 (System 2, slow)。

### 矛盾 2: 直接 RL 训不动, CoT 会乱飞

如果你直接拿一个 3D-VLM 做 RL (像 DeepSeek-R1 ([arxiv 2501.12948](https://arxiv.org/abs/2501.12948)) 那样从 scratch 开始), model 产生的 reasoning chain 会 incoherent——它可能说 "我看到一把椅子所以应该前进", 但下一秒又说 "椅子在左边所以应该右转", 自己跟自己打架, 训练根本 converge 不了。

---

## Nav-R1 的三个核心招数

### 招数 1: 先蒸馏 110K 条 "正确推理过程" 做 cold-start

既然直接 RL 训不动, 那就先让一个 "聪明老师" (Gemini 2.5 Pro) 写好 110K 条 step-by-step 的 reasoning trace, 让 Nav-R1 先 supervised 学一遍。

数据构造流程 (Fig. 3):

```
给你看: 第一人称 RGB-D 图 + 指令 "走到左边木头椅子" + 可选动作 [前进/左转/右转/停]
         ↓
Gemini 2.5 Pro 写: 
  <action>...</action>} \text{ 格式} \\ 0, & \text{否则} \end{cases}$$

这个就是看 model 有没有按规定格式输出。你可能觉得这个 reward 很 trivial, 但它其实 super important。它强制 model 把 "思考" 和 "决策" 分开, 这样下游才能分别评估 reasoning quality 和 action quality。Table VIII 的 ablation 证明: 去掉这个 reward, SR 从 42.2% 掉到 39.9%。

**Reward 2: Understanding Reward (理解分)**

$$R_{\mathrm{understanding}} = R_{\mathrm{ans}} + R_{\mathrm{sem}}$$

这个 reward 由两部分组成:

- $R_{\mathrm{ans}}$: 答案是否完全正确 (1 或 0)
- $R_{\mathrm{sem}} = \mathrm{CLIPScore}(I, \hat{a})$: 答案和图像的 CLIP 语义相似度

变量解释:
- $I$: 输入的 RGB-D 图像
- $\hat{a}$: model 生成的答案文本
- $\mathrm{CLIPScore}$: CLIP ([arxiv 2103.00020](https://arxiv.org/abs/2103.00020)) 的 image-text alignment score, 本质是 $\cos(E_{\text{image}}, E_{\text{text}})$, 范围 $[0, 1]$

intuition 是这样的: $R_{\mathrm{ans}}$ 是 sparse 的, 大多数时候是 0, 提供不了 gradient; $R_{\mathrm{sem}}$ 是 dense 的, 即使答案不精确, 只要语义相关就有分, 提供了 learning signal。两者互补, model 既不会犯 fact error, 也不会说废话。

**Reward 3: Navigation Reward (导航分)**

$$R_{\mathrm{navigation}} = \underbrace{\exp(-k \cdot D_F(T, \hat{T}))}_{R_{\text{path}}} + \underbrace{\exp(-k \|\hat{p} - p\|^2)}_{R_{\text{end}}}$$

变量含义:
- $T$: model 预测的轨迹, $\hat{T}$: ground-truth 轨迹
- $D_F$: 轨迹距离, 通常是 Fréchet Distance 或 DTW, 衡量两条 path 形状像不像
- $p$: 预测终点, $\hat{p}$: 真实终点
- $k$: decay 系数, 控制 reward 衰减速度
- $\exp(-\cdot)$: 指数衰减, 距离越近 reward 越接近 1, 越远越接近 0

这个 reward design 的精妙之处在于 path 和 endpoint 分开打分。如果只看 endpoint, model 可能会 "走捷径" 穿墙; 如果只看 path, model 可能会死板 follow 参考路径但最终没到目标。两个 reward 组合, 既要 "走得好" 又要 "到得了"。

Table VIII 的 ablation 证明三者缺一不可:
- 只有 Format + Understanding: SR 39.4%
- 只有 Format + Navigation: SR 38.7%
- 只有 Understanding + Navigation: SR 39.9%
- 三者全有: SR 42.2%

### 招数 3: Fast-in-Slow 双系统架构

这是最 intuitive 但也最 elegant 的设计。参考 Fig. 2。

**Slow System (System 2)**:
- 低频运行 (每 n 步跑一次)
- 输入: egocentric RGB-D + 语言指令 + historical context
- 输出: latent feature $h_t$, 编码 "这个房间的整体语义 + 我要去哪 + 之前怎么走的"
- 作用: 像 "战略规划", 决定大方向

**Fast System (System 1)**:
- 高频运行 (每步都跑)
- 输入: 当前 observation $(o_{t+1}, \ldots, o_{t+H})$ + slow system 传来的 $h_t$
- 输出: 短期动作序列 $\{a_{t+1}, \ldots, a_{t+H}\}$
- 作用: 像 "战术执行", 决定每一步怎么迈

公式 (10):

$$\{a_{t+1}, \dots, a_{t+H}\} = \pi_{\mathrm{fast}}(o_{t+1:t+H}, h_t)$$

- $a_{t+i}$: 第 $t+i$ 步的 action (如 forward, turn-left, stop)
- $o_{t+1:t+H}$: 从 $t+1$ 到 $t+H$ 步的 observation sequence
- $h_t$: slow system 在第 $t$ 步输出的 "战略 latent"
- $H$: short-horizon 长度, paper 推荐 $n \approx 3$, 即 slow 跑 1 次指导 fast 跑 3 次
- $\pi_{\mathrm{fast}}$: fast policy network

**关键 trick**: Fast system 复用 Nav-R1 的 final transformer blocks, 这样它继承了 System 2 的 pretrained knowledge, 但计算量很轻。你可以理解为 "fast system 是 slow system 的一个 lightweight head"。

异步协调的 ratio 是 1:n, $n \approx 3$。intuition 是: 每 3 步重新 "思考" 一次战略, 中间 3 步用 fast system 快速执行。Table VII 的 ablation 证明这个设计是必要的:
- Slow-only: SR 61.2% (想得多但执行慢)
- Fast-only: SR 58.7% (执行快但没大局观)
- Dual-system: SR 72.5% (两者协同)

---

## 实验结果讲人话

### Navigation 结果 (Table II, III)

在 R2R-CE val-unseen 上, Nav-R1 的 SR 是 72.5%, 而 VLN-R1 ([arxiv 2506.17221](https://arxiv.org/abs/2506.17221)) 只有 30.2%, NaVILA ([arxiv 2412.04453](https://arxiv.org/abs/2412.04453)) 是 54.0%, CorrectNav ([arxiv 2508.10416](https://arxiv.org/abs/2508.10416)) 是 65.1%。

在 HM3D-OVON val-unseen 上, Nav-R1 SR 42.2%, 而 MTU3D ([ICCV 2025](https://arxiv.org/abs/2508.08465)) 是 40.8%, Uni-NaVid ([RSS 2025](https://arxiv.org/abs/2412.04453)) 是 39.5%。

这些数字说明: Nav-R1 在 navigation 任务上确实 SOTA, 平均提升 8% 左右。

### Dialogue/Reasoning/Planning 结果 (Table IV)

这里有个有意思的发现: Nav-R1 在 dialogue, reasoning, planning 上的表现和 3D-R1 基本持平, 没有明显提升。

这其实符合预期。Nav-R1 的 RL training 主要优化 navigation reward, 并没有训练额外的 understanding module。它的 strategy 是 "保住 reasoning/planning 能力的前提下, 尽量提升 navigation"。这说明 RL 的 reward 设计没有 catastrophic forgetting, 没有为了 navigation 牺牲 dialogue 能力。

### Real-world 结果 (Table V)

在真实 robot (WHEELTEC R550 + Jetson Orin Nano) 上测试三个场景: meeting room, lounge, corridor。

Nav-R1 在 meeting room 上 NE 1.23m, SR 1.03; MTU3D NE 1.64m, SR 0.73。在 lounge 上 Nav-R1 NE 0.98m, SR 1.12。

注意 SR > 1 这个数字看起来奇怪, 其实是因为他们的 SR 定义可能 normalized 了, 或者多次测试取平均。真实世界测试中 Nav-R1 明显比所有 baseline 好, 证明 simulation 到 real 的 sim2real transfer 成功了。

### Test-time Efficiency (Table VI)

Jetson Orin Nano 上跑 NaVid 要 320ms/帧, Uni-NaVid 410ms/帧, 这种延迟在 real-time navigation 里基本不可用。Nav-R1 用 cloud inference, 95ms/帧, 基本可以 real-time。

---

## 为什么这个工作 important? 我的 take

1. **Dual-system 设计是 embodied AI 的正确抽象**。纯粹的 end-to-end model 无法同时做好 reasoning 和 control, Fast-in-Slow 提供了一个 principled 的解耦方案。

2. **Cold-start + RL 的两阶段训练范式在 embodied 领域 work**。这和 DeepSeek-R1 在 LLM 领域验证的范式一致, 但 Nav-R1 把它迁移到 3D vision-language-action 场景, 证明这个范式有 generalizability。

3. **三个互补 reward 的设计很 elegant**。Format reward 保证结构, Understanding reward 保证语义, Navigation reward 保证路径, 各司其职, ablation 也证明缺一不可。

4. **Cloud-assisted real-world deployment 是务实选择**。承认 Jetson Orin Nano 跑不动大 model, 用 WiFi 6E + cloud inference 绕开, 这种工程务实值得借鉴。虽然 limitation 章节也承认了这不是终极方案, 但短期 practical。

---

## 参考链接汇总

- [Nav-R1 GitHub](https://github.com/AIGeeksGroup/Nav-R1)
- [Nav-R1 Project Page](https://aigeeksgroup.github.io/Nav-R1)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - RL reasoning 的 LLM 范式
- [GRPO 原论文](https://arxiv.org/abs/2402.03300) - DeepSeekMath
- [Kahneman Thinking Fast and Slow](https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555) - Dual-system 理论
- [3D-R1](https://arxiv.org/abs/2507.23478) - Nav-R1 的 base model
- [VLN-R1](https://arxiv.org/abs/2506.17221) - 同期 VLN RL 工作
- [Uni-NaVid](https://arxiv.org/abs/2506.04453) - Video-based VLA baseline
- [NaVILA](https://arxiv.org/abs/2412.04453) - Legged robot VLA
- [MTU3D](https://arxiv.org/abs/2508.08465) - Visual grounding + exploration
- [StreamVLN](https://arxiv.org/abs/2507.05240) - Slow-fast context modeling
- [CorrectNav](https://arxiv.org/abs/2508.10416) - Self-correction VLN
- [R2R-CE](https://arxiv.org/abs/2004.08906) - Continuous environment VLN benchmark
- [HM3D-OVON](https://arxiv.org/abs/2310.08737) - Open-vocabulary ObjNav
- [SOON](https://arxiv.org/abs/2103.05152) - Scenario-oriented ObjNav
- [SQA3D](https://arxiv.org/abs/2304.07008) - 3D reasoning benchmark
- [3D-LLM](https://arxiv.org/abs/2307.12981) - Embodied dialogue benchmark
- [CLIP](https://arxiv.org/abs/2103.00020) - Image-text alignment
- [LoRA](https://arxiv.org/abs/2106.09685) - Parameter-efficient tuning
- [PPO](https://arxiv.org/abs/1707.06347) - RL baseline

---

**TL;DR**: Nav-R1 = Cold-start SFT (110K CoT 数据) + GRPO RL (三个互补 reward) + Fast-in-Slow dual-system (异步 1:3 协调)。在 embodied navigation, dialogue, reasoning, planning 上全面提升, 真实 robot 部署成功。核心 insight: reasoning 和 control 解耦, 各自做擅长的事, 像人一样 "想一步, 走三步"。

---

# Nav-R1: Embodied Foundation Model 的深度技术解读

Andrej, 这篇 paper 是一个非常有意思的工作,它把 RL reasoning (DeepSeek-R1 风格) 和 embodied navigation 结合起来,同时引入了 Kahneman 的 dual-system 认知理论来解决 long-horizon reasoning 和 low-latency control 之间的 tension。让我从 intuition 层面一层一层拆解给你看。

---

## 1. 高层直觉:为什么需要 Nav-R1?

embodied navigation 领域长期存在两个 fundamental tension:

**Tension 1: Reasoning trace 的 incoherence**
现有的 LVLM-based navigation agent 往往直接从 observation 映射到 action,缺少中间的 reasoning chain。这导致 generalization 很 brittle,尤其在 unseen environment 下,reasoning trace 会变得 semantically inconsistent。

**Tension 2: Long-horizon vs. Low-latency 的矛盾**
semantic reasoning 需要处理 historical context、scene understanding、instruction alignment,这个计算量很大,天然是 slow 的;但 real-time navigation 要求每一步都要快速响应,reactive control 需要 fast。如果用同一个 model 同时做两件事,要么 reasoning 太浅,要么 control 太慢。

Nav-R1 的核心 insight 就是借鉴 Kahneman 的 *Thinking, Fast and Slow* ([book link](https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555)):
- **System 2 (slow)**: 处理 long-horizon semantic reasoning,低频运行,输出 latent guidance $h_t$
- **System 1 (fast)**: 高频运行,复用 System 2 的 transformer blocks,输出 short-horizon action sequence

两者以 **1:n (n≈3)** 的频率比异步协调。这是一个很 elegant 的设计——它和 StreamVLN ([arxiv 2507.05240](https://arxiv.org/abs/2507.05240)) 的 slow-fast context modeling 思路有异曲同工之妙,但 Nav-R1 更进一步把它做成了异步的 dual-system。

---

## 2. Nav-CoT-110K: Cold-Start 的数据引擎

这是整个工作的基石。直接对 large 3D-VLM 做 RL (像 DeepSeek-R1 那样从 scratch 开始) 会导致 policy 产生 semantically incoherent 的 CoT,无法 converge。所以需要先 SFT bootstrapping。

### 2.1 Data Engine Pipeline

```
Input: (egocentric RGB-D, instruction, candidate actions, format spec)
        ↓
    Gemini 2.5 Pro  (作为 reasoner)
        ↓
Output:  <action> decision </action>
        ↓
Two-stage filtering:
  (i) Rule-based: 丢弃 incomplete / logically inconsistent
  (ii) Trajectory verification: action feasibility vs ground-truth paths
        ↓
Nav-CoT-110K (从 115K raw 蒸馏到 110K)
```

### 2.2 数据规模对比 (Table I 分析)

| Dataset | Scenes | Modality | Tasks | Env |
|---------|--------|----------|-------|-----|
| R2R | 90 MP3D | L | 22K | DE |
| R2R-CE | 90 MP3D | L | 4.5K | CE |
| RxR-CE | 90 MP3D | L | - | CE |
| SOON | 90 MP3D | L | 30K | DE |
| OVON | 181 HM3D | L | 53K | CE |
| **Nav-CoT-110K** | **342 (MP3D+HM3D)** | **V, L, P** | **110K** | **CE** |

这里有一个值得注意的点:Nav-CoT-110K 是唯一同时覆盖 V(ision), L(anguage), P(point cloud) 三种 modality,并且同时覆盖 ObjNav 和 VLN 两个 task type 的 dataset。这种 multi-task, multi-modal 的覆盖是 RL stage 能学到 generalizable reward 的前提。

参考: [HM3D-OVON](https://arxiv.org/abs/2310.08737), [R2R-CE](https://arxiv.org/abs/2004.08906), [SOON](https://arxiv.org/abs/2103.05152)

---

## 3. Architecture 深度解析 (Fast-in-Slow)

这是这篇 paper 最核心的 architectural innovation。参考 Fig. 2。

### 3.1 Slow System (System 2)

- **输入**: egocentric RGB-D frames + language instructions + historical context
- **频率**: 低频 (每 n 步更新一次)
- **输出**: latent feature $h_t$,编码 scene semantics + temporal dependencies + global goal
- **功能**: aggregate visual history into compact memory states,保证 scene-level semantic consistency

### 3.2 Fast System (System 1)

- **输入**: high-frequency multimodal inputs $(o_{t+1}, \ldots, o_{t+H})$ + slow system 的 $h_t$
- **频率**: 高频 (每步都跑)
- **输出**: short-horizon action sequence $\{a_{t+1}, \dots, a_{t+H}\}$
- **关键**: **复用** Nav-R1 的 final transformer blocks,继承 System 2 的 pretrained knowledge,但保持 lightweight

公式 (10) 如下:

$$\{a_{t+1}, \dots, a_{t+H}\} = \pi_{\mathrm{fast}}(o_{t+1:t+H}, h_t)$$

**变量解释**:
- $a_{t+i}$: 第 $t+i$ 步的 action (通常是 discretized navigation command,如 forward, turn-left, stop)
- $o_{t+1:t+H}$: 从 $t+1$ 到 $t+H$ 的 observation sequence,包含 RGB, depth, point cloud tokens
- $h_t$: slow system 在第 $t$ 步输出的 latent guidance
- $H$: short-horizon 的长度 (paper 没明说,但推测是 3-5 步,和 $n$ 一致)
- $\pi_{\mathrm{fast}}$: fast policy network

### 3.3 为什么这个设计 work?

intuition 是这样的:slow system 像人在陌生环境里"环顾四周,理解布局",fast system 像"基于理解执行具体动作"。如果只 fast-only (Table VII ablation),SR 只有 58.7%,因为缺少 global semantics;只 slow-only,SR 61.2%,因为 real-time execution 跟不上。Dual-system 拉到 72.5%,说明这两个 component 是真正 complementary 的。

这让我想起 ([Kahneman 理论](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)) 中 System 1 的"effortless, automatic" 和 System 2 的"effortful, deliberate" 的分工。

---

## 4. GRPO-based RL Framework:三个互补的 Reward

这是技术含量最高的部分。Group Relative Policy Optimization (GRPO) 来自 DeepSeekMath ([arxiv 2402.03300](https://arxiv.org/abs/2402.03300)),核心思想是 sample N 个 candidate responses,用 group-relative advantage 替代 absolute value function。

### 4.1 三个 Reward 的设计

#### (a) Format Reward $R_{\mathrm{Format}}$ (Eq. 1)

$$R_{\mathrm{Format}} = \begin{cases} 1, & \text{if output adheres to format} \\ 0, & \text{otherwise} \end{cases}$$

这里 format 是指 ` <action>...</action>` 或 ` <answer>...</answer>` 的结构。这个 reward 看起来简单,但非常 critical——它强制 model 把 reasoning 和 decision 解耦,使得 RL 训练时可以分别 evaluate 两者。Table VIII 的 ablation 显示,去掉这个 reward 后 SR 从 42.2% 掉到 39.9%,说明结构化输出对 reasoning quality 有显著影响。

#### (b) Understanding Reward $R_{\mathrm{understanding}}$ (Eq. 4)

$$R_{\mathrm{understanding}} = R_{\mathrm{ans}} + R_{\mathrm{sem}}$$

其中:

**Answer Reward** (Eq. 2):
$$R_{\mathrm{ans}} = \begin{cases} 1, & \text{predicted answer equals ground truth} \\ 0, & \text{otherwise} \end{cases}$$

**Semantic Reward** (Eq. 3):
$$R_{\mathrm{sem}} = \mathrm{CLIPScore}(I, \hat{a})$$

- $I$: paired RGB-D image
- $\hat{a}$: generated answer text
- $\mathrm{CLIPScore}$: CLIP-style image-text alignment score,通常定义为 $w \cdot \max(\cos(E_I, E_T), 0)$,其中 $E_I, E_T$ 是 CLIP image/text encoder 的 L2-normalized embedding

这个 design 很巧妙——$R_{\mathrm{ans}}$ 是 sparse 的 (0/1),确保 factual correctness;$R_{\mathrm{sem}}$ 是 dense 的 (continuous),提供 gradient signal 即使 answer 不完全匹配 ground truth。两者结合避免了 factual error 和 semantically irrelevant output 两种 failure mode。

#### (c) Navigation Reward $R_{\mathrm{navigation}}$ (Eq. 7)

$$R_{\mathrm{navigation}} = R_{\mathrm{path}} + R_{\mathrm{end}}$$

**Path Reward** (Eq. 5):
$$R_{\mathrm{path}} = \exp\big(-k D_F(T, \hat{T})\big)$$

**Endpoint Reward** (Eq. 6):
$$R_{\mathrm{end}} = \exp\big(-k \|\hat{p} - p\|^2\big)$$

**变量解释**:
- $T$: predicted trajectory,即 agent 实际走过的 path $\{p_1, p_2, \ldots, p_T\}$,每个 $p_i \in \mathbb{R}^3$ (or $\mathbb{R}^2$ for floor plan)
- $\hat{T}$: ground-truth reference trajectory
- $D_F(\cdot)$: trajectory distance metric,通常是 **Fréchet distance** 或 **Dynamic Time Warping (DTW)**,衡量两条 trajectory 的 shape similarity,对 time-warping robust
- $k$: decay coefficient,控制 reward 的 sharpness。$k$ 越大,对 deviation 惩罚越重
- $p, \hat{p}$: predicted / ground-truth endpoint (3D position)
- $\|\hat{p} - p\|^2$: squared Euclidean distance

**intuition**: $\exp(-k \cdot \text{distance})$ 这种 form 在 robotics RL 里很经典,叫 **exponential kernel reward**。它的好处是:
1. Always positive (no negative reward spiking)
2. Smooth gradient (避免 reward cliff)
3. 当 distance=0 时 reward=1 (perfect),distance 越大 reward 越接近 0

把 path 和 endpoint 拆开是必要的:只优化 endpoint 会导致 agent 走 "shortcut" (比如穿墙),只优化 path 会导致 agent 死板 follow reference 但最终没到目标。

### 4.2 GRPO Objective (Eq. 8, 9)

**Advantage normalization** (Eq. 8):
$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$

- $r_i$: 第 $i$ 个 candidate response 的 total reward ($R_{\mathrm{Format}} + R_{\mathrm{understanding}} + R_{\mathrm{navigation}}$ 之和)
- $\mathbf{r} = \{r_1, r_2, \ldots, r_N\}$: group of N samples 的 reward vector
- $\hat{A}_i$: normalized advantage,表示第 $i$ 个 response 相对于 group 平均的好坏程度

这个 normalization 的精妙之处在于:它消除了 absolute reward scale 的影响,只保留 relative ordering。这意味着即使 reward function 的 scale 不完美 calibrated,GRPO 仍然能 work。

**GRPO Loss** (Eq. 9):

$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_c \Bigg[ \frac{1}{G} \sum_{i=1}^{G} \Big( \min\big(\rho_i \hat{A}_i, \mathrm{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\big) - \beta \cdot \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \Big) \Bigg]$$

其中:
- $\rho_i = \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\mathrm{old}}}(o_i | q)}$: importance sampling ratio,新旧 policy 的 probability ratio
- $\varepsilon$: clipping range (通常 0.1-0.2),防止 policy update 步长过大
- $\beta$: KL penalty coefficient (paper 里 best value 是 0.02,见 Table IX)
- $\pi_{\mathrm{ref}}$: frozen reference policy (通常是 SFT 后的 model)
- $\mathbb{D}_{\mathrm{KL}}$: KL divergence,$\sum_x \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\mathrm{ref}}(x)}$
- $G$: group size

**KL penalty 的作用**:防止 RL 训练"跑偏"——如果 $\beta$ 太小 (0.005,Table IX),policy 偏离 reference 太远,SR 反而掉到 64.3%;如果 $\beta$ 太大 (0.05),exploration 被压制,SR 67.8%。Sweet spot 在 0.02,SR 71.3%。这是一个典型的 bias-variance tradeoff。

参考: [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [GRPO 原论文](https://arxiv.org/abs/2402.03300), [PPO 经典论文](https://arxiv.org/abs/1707.06347)

---

## 5. 训练 Pipeline 的两个阶段

### Stage 1: Cold-Start SFT
- **Base model**: 3D-R1 ([arxiv 2507.23478](https://arxiv.org/abs/2507.23478)),本身已有 3D reasoning + vision-language alignment
- **Data**: Nav-CoT-110K
- **Epoch**: 2
- **Batch size**: 8
- **Optimizer**: AdamW, weight decay 0.01
- **LR schedule**: cosine annealing, $10^{-4} \to 10^{-5}$
- **目的**: 让 model 学会 `
