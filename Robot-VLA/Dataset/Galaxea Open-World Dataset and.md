---
source_pdf: Galaxea Open-World Dataset and.pdf
paper_sha256: 151aa3493bcf6c52a62a30e0f84a1ba686dd22a84fc7de602a92abd5e49639c1
processed_at: '2026-08-04T11:54:14-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Galaxea G0

Andrej，说白了这篇paper讲的就是一件事：**别迷信大数据，数据得跟你的robot对得上才行。**

---

## 核心story一句话版

Galaxea团队搞了个robot，用了23个自由度，能满屋子跑。他们收集了500小时真实人类生活场景的数据——厨房、办公室、零售店、家里。然后用这些数据训练了一个dual-system的VLA模型。结果发现：**在别人的robot数据上pre-train，反而比从零开始训练还差。**

就这么简单的一个发现，但很重要。

---

## 为什么这个发现让人意外

整个robotics community过去两年的mantra是：**越多data越好，越diverse越好**。Open-X-Embodiment搞了22种robot的混合数据，大家觉得这就是通往generalist robot的金光大道。

Galaxea说：hold on，等一下。

你看看我们的实验——

```
排名（从好到差）:
G0 (Full)              ← Stage-1 + Stage-2 都做
G0 (Stage-2 400h)      ← 只在自己的robot数据上pre-train  
G0 (Stage-2 200h)      ← 同上但数据少
π0                     ← Physical Intelligence的baseline
G0 (Scratch)           ← 啥pre-train都不做
G0 (Stage-1)           ← 只在cross-embodiment数据上pre-train
```

看到没？**Stage-1排在最后，比啥都不做还差。**

---

## Intuition——为什么会这样

想象你在学开车。

Stage-1 pre-training就像是：你先去骑了1000小时自行车、摩托车、拖拉机、挖掘机。然后有人给你一辆F1赛车说"来，fine-tune一下"。

问题是什么？你骑自行车学的"转向"动作，跟F1赛车的方向盘完全是两码事。你不仅没用上那些经验，反而要花精力**忘掉**那些坏习惯。

具体到robot：

- OXE里大部分是7-DoF fixed-base single-arm robot（比如Franka、WidowX）
- Galaxea R1 Lite是23-DoF，有mobile base、有torso、有双臂
- 当model在Franka数据上学到"伸手抓东西"的action representation
- 这个representation对mobile bimanual robot来说完全是**misaligned**的

最明显的是Bed Making任务。这个任务需要chassis移动+torso俯仰+双臂协调——whole-body control。OXE里根本没有这种数据。结果就是：

```
Bed Making per-skill scores:
                    chassis控制   torso控制
G0 (Stage-2)         很好          很好
G0 (Stage-1)         很差          很差
π0                   很差          很差
G0 (Scratch)         一般          一般
```

Stage-1和π0在whole-body skills上**比从零训练还差**。这就是negative transfer。

---

## Dataset长什么样

### Galaxea R1 Lite robot

- 两个6-DoF arm（带spherical wrist + parallel gripper）
- 3-DoF torso（能上下+前后倾）
- 6-DoF omnidirectional base（最高1.5 m/s）
- 总共23-DoF，payload 5kg
- 1个head camera + 2个wrist camera (Intel RealSense D405)

Data collection用的**isomorphic teleoperation**——操作员拿着跟robot形状一样的master arm来操控。好处是不用做human→robot的retargeting，arm始终在reachable范围内。

### 数据规模

| 项目 | 数量 |
|------|------|
| Trajectories | 100K |
| Hours | 500 |
| Tasks | 150类 |
| Scenes | 50个真实场景 |
| Objects | 1600+ |
| Skills | 58种 |
| Sites | 11个物理地点 |

Scenes覆盖residential、catering、retail、office四大类。每个episode按subtask切分，annotation用**fixed schema**（不是自由文本），这样labeling快且一致。

跟其他dataset对比：

| Dataset | 特点 | 问题 |
|---------|------|------|
| BridgeData V2 | 单robot，量大 | 场景limited |
| DROID | 单robot，量大 | Lab环境 |
| OXE | 22种robot | 数据质量参差不齐 |
| AgiBot World | 规模大 | Lab环境 |
| **Galaxea** | **单robot，真实场景** | **规模中等** |

Galaxea的bet是：**real-world scene diversity > embodiment diversity**。

---

## 模型架构——Dual System

### 灵感来源

Kahneman的《Thinking, Fast and Slow》——人脑有两个system：
- System 1：快速、reactive、下意识
- System 2：慢速、deliberative、有意识

映射到robot：
- **G0-VLA** = System 1：看到画面+指令→输出action，高频跑
- **G0-VLM** = System 2：理解人的高级指令→拆成subtask序列给VLA，低频跑

两个model**异步运行**。VLM不需要每帧都跑，只在需要replan的时候跑。

### G0-VLA结构

```
Input: o_t (3 cameras), l_t (subtask instruction), s_t (proprioception)
        ↓
   [SigLIP vision encoder] → [MLP projector] → vision embeddings
        ↓
   [PaLiGemma Transformer] ← language tokens + proprio tokens
        ↓ (输出KV cache)
   [Action Expert with flow matching] → A_t (action chunk)
```

Base model是PaLiGemma (3B)，vision encoder是SigLIP，然后接一个flow matching的action expert。

### G0-VLM结构

Base model是Qwen2.5-VL。特别的地方是input不only当前帧，而是**k帧历史观测+action**，间隔1秒。这样VLM能理解task progression。

训练数据怎么来的？从Galaxea dataset采样episode，用DeepSeek-R1生成human-style的高层指令。比如：

- 原始subtask annotation: "pull chair" → "push chair" → "done"
- DeepSeek-R1生成的human instruction: "I'm going to sit down, could you pull the chair out for me?"
- Robot response: "I'm working on it!"

---

## 三阶段训练——核心方法

### Stage-1: Cross-embodiment pre-training（只训练VLM）

数据：1000h OXE + 500h Galaxea（high-level description only）+ 200h in-house

方法：FAST tokenizer把action变成离散token，然后用标准next-token prediction训练VLM：

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)$$

- $\mathbf{A}_t^d$: N个discrete action tokens
- $a_i^d$: 第i个action token
- $a_{<i}^d$: 前面所有已生成的token
- $o_t, l_t, s_t$: 观测、指令、proprioception

**关键决策：Stage-1不训练action expert。** 原因：
1. Cross-embodiment数据annotation质量不一致，action expert会学到noise
2. Flow matching loss在model未converge时会harm learning

### Stage-2: Single-embodiment pre-training（训练完整VLA）

数据：Galaxea Open-World Dataset，带subtask-level annotation

方法：Flow matching loss

$$\mathcal{L}_{\mathrm{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)} \left[ \| \nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t) \|^2 \right]$$

- $A_t$: 真实action chunk，horizon H
- $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$: noisy action（$\tau$从0到1插值，$\varepsilon$是Gaussian noise）
- $\nu_\theta(\cdot)$: model预测的flow vector field
- $u(\cdot)$: target flow

**Flow matching vs Diffusion的intuition**：
- Diffusion学的是reverse noise process，路径弯弯绕绕
- Flow matching学的是一个vector field，定义从noise到data的直线路径
- Flow matching更稳定，路径更短

### Post-training: Task-specific fine-tune

每个任务最多100条trajectory，用跟Stage-2一样的flow matching objective。

---

## 实验结果的关键takeaway

### Finding 1: Single-embodiment pre-training是王道

```
Average progress score across benchmarks:
G0 (Full)       ████████████████████  最高
G0 (Stage-2)    ██████████████████    很好
π0              ███████████████       还行
G0 (Scratch)    █████████████         一般
G0 (Stage-1)    ███████████           最差（！）
```

### Finding 2: Few-shot transfer也靠Stage-2

只用20条trajectory fine-tune时，Stage-2 pre-trained model**显著领先**。Stage-1 alone没有明显优势。

这说明single-embodiment pre-training给了model一个**embodiment-specific的inductive bias**，让新任务的学习更快。

### Finding 3: VLM planning需要domain adaptation

Table 1的数字很striking：

| Model | Table Bussing | Microwave | Bed Making | Blocks |
|-------|---------------|-----------|------------|--------|
| Gemini-2.5-pro | 32% | 16% | 54% | 55% |
| Qwen2.5-VL-72B | 26% | 17% | 48% | 22% |
| **G0-VLM** | **83%** | **74%** | **78%** | **76%** |

Gemini-2.5-pro这么强的model，在没有robotic domain adaptation的情况下，instruction accuracy只有16-55%。**通用VLM不足以做robotic planning。**

---

## 这篇paper的真正价值

不在于architecture novelty——dual system、flow matching、FAST tokenizer都是existing ideas。

真正价值在于**一个clean的empirical finding**：

> **Cross-embodiment pre-training在embodiment gap大的时候会有negative transfer。Single-embodiment pre-training才是key。**

这对整个community是一个wake-up call：
- OXE的multi-embodiment narrative需要重新审视
- Data的**alignment**比volume更重要
- Future work可能需要**embodiment-aware pre-training**——根据embodiment similarity来选择性pre-train

---

## 我的批判性思考

### 没解决的问题

1. **Stage-1真的完全没用吗？** Paper说它在"universal action patterns"（pick-place, push-pull）上有帮助，但在embodiment-specific skills上有害。那能不能做**selective Stage-1**——只用embodiment-similar的数据？

2. **Scaling law缺失**：500h single-embodiment data够不够？如果到5000h会怎样？Single-embodiment的generalization ceiling在哪？

3. **Dual-system的latency没讲清楚**：VLM多久replan一次？VLA的action chunk horizon k是多少？VLM改变subtask时如何平滑过渡？

4. **Stage-1 + Stage-2的顺序是否optimal**？能不能联合训练？能不能做curriculum learning式的interleaving？

### 一个大胆的猜测

我觉得未来的方向可能是**embodiment-conditioned pre-training**：

给每个embodiment一个embedding，让model在pre-train时就知道"这是Franka的数据"还是"这是Galaxea的数据"。这样model可以学习"哪些skill可以cross-embodiment transfer，哪些是embodiment-specific的"。

这跟NLP里的language-conditional training类似——一个model学多种语言，但每种语言有自己的language token。

---

## 最后

Andrej，这篇paper的essence就是一句话：

**Data alignment beats data volume. 在robot learning里，跟你的robot morphology对不上的数据，pre-train反而会拖后腿。**

这是对当前VLA community "data scaling at all costs" narrative的一个重要counterpoint。Galaxea承诺open-source dataset和models（https://opengalaxea.github.io/G0/），这对community验证这些findings非常关键。

**References:**
- G0 Project: https://opengalaxea.github.io/G0/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/
- π0 Paper: https://arxiv.org/abs/2410.24164
- π0.5 Paper: https://arxiv.org/abs/2504.16054
- FAST Tokenizer: https://arxiv.org/abs/2501.09747
- PaLiGemma: https://arxiv.org/abs/2407.07726
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Flow Matching Original Paper: https://arxiv.org/abs/2208.14535
- SayCan: https://say-can.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- OpenHelix: https://arxiv.org/abs/2505.03912
- AgiBot World: https://arxiv.org/abs/2503.06669
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2309.00918
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B: https://arxiv.org/abs/2410.07864
- HybridVLA: https://arxiv.org/abs/2503.10631
- CogACT: https://arxiv.org/abs/2411.19650
- Kahneman's Book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- Negative Transfer Discussion: https://arxiv.org/abs/2507.06219

---

# Galaxea Open-World Dataset & G0 Dual-System VLA — 深度解析

Andrej，这篇paper的核心 contribution 非常 clean，本质上是回答了一个在 VLA community 里争议已久的问题：**cross-embodiment pre-training 到底有没有用？** Galaxea 团队的回答是 nuanced 的 — cross-embodiment pre-training 在 embodiment gap 大的时候会 **degrade** 性能，而 single-embodiment pre-training 才是关键。这个发现直接挑战了 Open-X-Embodiment 社区的主流叙事。

---

## 1. Dataset — Galaxea Open-World Dataset

### 1.1 规模与多样性

| Metric | Value |
|--------|-------|
| Trajectories | 100K |
| Hours | 500 |
| Tasks | 150 categories |
| Scenes | 50 (residential, catering, retail, office) |
| Objects | 1,600+ unique |
| Skills | 58 operational |
| Embodiment | Single (Galaxea R1 Lite, 23-DoF) |

关键 design choice：**single embodiment across all scenes**。这与 Open-X-Embodiment [1] 的 multi-embodiment aggregation 形成鲜明对比。OXE 的 heterogeneity 带来了 annotation inconsistency 和 environmental noise，而 Galaxea 选择了 consistency + scene diversity 的 trade-off。

### 1.2 Hardware — Galaxea R1 Lite

23-DoF breakdown：
- **2× 6-DoF arms** (spherical wrists + parallel grippers)
- **3-DoF torso** (vertical + pitch，扩展 workspace)
- **6-DoF vector-drive omnidirectional base** (up to 1.5 m/s)

Payload 5 kg，reach 60 cm。Perception：1 stereo RGB head camera + 2 Intel RealSense D405 RGB-D wrist cameras。

**Isomorphic teleoperation** — 这是值得注意的 engineering choice。相比 VR teleoperation（如 Aloha [1.5]），isomorphic mapping 直接将 operator 的运动映射到 robot kinematics，保证 arms 始终在 reachable postures 内，避免了 IK failures 和 human→robot retargeting 问题。参考 Aloha 的 retargeting 挑战：https://tonyzhaozh.github.io/aloha/

### 1.3 Annotation — Subtask-level

每个 episode 被 segmented 成 atomic subtasks，annotation 采用 **fixed schema**（非 free-form text）。这加速了 labeling 并保证 consistency。关键 frame sampling weights 在 G0-VLM 训练时被 boost：subtask termination moments 和 gripper state changes。

---

## 2. G0 Dual-System Architecture

### 2.1 Kahneman System 1 / System 2 框架

灵感来源：Daniel Kahneman《Thinking, Fast and Slow》[4]。
- **System 1 (G0-VLA)**：fast, reactive，高频运行，translate sensory inputs → low-level actions
- **System 2 (G0-VLM)**：deliberative, planning，低频运行，decompose high-level command → subtask sequence

**异步运行**是关键 — VLM 不需要每帧都跑，避免推理延迟拖累实时控制。这与 Hi Robot [19] 的 hierarchical VLA 思路类似：https://arxiv.org/abs/2502.19417

### 2.2 G0-VLA 架构详解

```
Input: o_t (3 cameras), l_t (subtask instruction), s_t (proprioception)
   ↓
[SigLIP vision encoder] → [MLP projector] → vision embeddings
   ↓
[PaLiGemma Transformer] ← language tokens + proprio tokens
   ↓ (KV cache)
[Action Expert (flow matching)] → A_t = a_{t:t+k} (action chunk, horizon k)
```

VLM backbone：**PaLiGemma** [29] (3B params)，SigLIP vision encoder + single-layer MLP projector + standard Transformer。Reference: https://arxiv.org/abs/2407.07726

### 2.3 G0-VLM 架构

Base model：**Qwen2.5-VL** [30] (https://arxiv.org/abs/2502.13923)。Instruction tuning 数据来自 Galaxea Open-World Dataset，配合 DeepSeek-R1 合成 human-style high-level instructions。

Input format 很特别：k-frame historic observations + actions at 1-second intervals，让 VLM 能处理 long-horizon temporal context。这解决了单帧 VLM 无法理解 task progression 的问题。

---

## 3. Three-Stage Training Curriculum — 核心创新

### 3.1 Stage-1: Cross-embodiment Pre-training (VLM only)

**关键 insight**：Stage-1 **只训练 VLM**，不训练 action expert。原因有二：

1. Cross-embodiment 数据的 annotation quality 和 action accuracy 参差不齐，action expert 会学到 noisy 信号
2. Flow matching loss 在 model 未收敛时可能 harm learning process

**Action tokenization**：采用 **FAST tokenizer** [11] (https://arxiv.org/abs/2501.09747)，将连续 action chunks 转为离散 token sequence。这让 VLM 可以用标准 next-token prediction 训练：

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)$$

变量解释：
- $\mathbf{A}_t^d$: N 个 discrete action tokens (由 FAST tokenizer 产生)
- $a_i^d$: 第 i 个 discrete action token
- $a_{<i}^d$: 前面所有已生成的 action tokens
- $o_t$: visual observation at time t
- $l_t$: language instruction at time t
- $s_t$: proprioceptive state at time t
- $N$: action token sequence length

Training data mixture：
- 1,000 hours OXE trajectories
- 500 hours Galaxea Open-World (high-level descriptions only, 排除 low-level annotations)
- 200 hours in-house data (high-level only)

### 3.2 Stage-2: Single-embodiment Pre-training (Full VLA)

此时 action expert 被 newly initialized，与 pre-trained VLM 一起训练。Objective 是 maximum likelihood：

$$\max_\theta \mathbb{E}_{p(A_t, o_t, l_t, s_t)} \left[ \log \pi_\theta(A_t \mid o_t, l_t, s_t) \right]$$

实现方式：**Flow matching loss**：

$$\mathcal{L}_{\mathrm{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)} \left[ \| \nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t) \|^2 \right]$$

变量详解：
- $A_t$: 真实 action chunk，horizon $H$，即 $A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]$
- $o_t, l_t, s_t$: 同上
- $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$: **interpolated noisy action**
  - $\tau \in [0, 1]$: flow time parameter (插值参数)
  - $\varepsilon$: noise sample，通常 from standard Gaussian $\mathcal{N}(0, I)$
  - 当 $\tau=0$: pure noise; 当 $\tau=1$: clean action
- $\nu_\theta(\cdot)$: VLA 预测的 **flow vector field** (neural network parameterized by $\theta$)
- $u(A_t^\tau \mid A_t) = A_t - A_t^\tau / (1-\tau)$ 或者更准确说是 conditional flow 的 target
- $\theta$: 所有可训练参数

**Flow matching vs Diffusion** 的 intuition：
- Diffusion (DDPM) 学习 reverse noise process，需要迭代 denoising
- Flow matching 学习一个 vector field $\nu_\theta$，定义从 noise distribution 到 data distribution 的 continuous flow
- Flow matching 可以学习 **更直的概率路径**（optimal transport），训练更稳定
- 理论 reference: Lipman et al. "Flow Matching for Generative Modeling" https://arxiv.org/abs/2208.14535

这与 π0 [12] 的方法一脉相承 (https://arxiv.org/abs/2410.24164)，但 G0 在此基础上加了 3-stage curriculum。

### 3.3 Post-training: Task-specific Fine-tuning

Same objective as Stage-2，但限制在 ≤100 trajectories per task。测试 generalization ability。

---

## 4. 关键实验发现 — Build Intuition

### 4.1 Pre-trained Weights Comparison (Figure 9)

Ranking (average progress score):
```
G0 (Full) > G0 (Stage-2 400h) > G0 (Stage-2 200h) > π0 > G0 (Scratch) > G0 (Stage-1)
```

**最 striking 的发现**：G0 (Stage-1) **比 from scratch 还差**！

这 contradicts 了 "pre-training always helps" 的 common belief。Intuition：
- Stage-1 学到的是 cross-embodiment 通用 action patterns（pick-and-place, push-pull）
- 但 Galaxea R1 Lite 的 morphology（23-DoF, omnidirectional base, torso pitch）与 OXE 里的 robots 差异巨大
- Stage-1 学到的 action representations **misaligned** with target embodiment
- 当 fine-tuning 时，model 需要 **unlearn** 这些 misaligned representations，比 from scratch 更难

### 4.2 Embodiment-specific Actions (Figure 11) — Bed Making

Bed Making 需要 **whole-body coordination**：chassis + torso + arms。这是 OXE 里完全没有的 skill。

Per-skill breakdown：
- Stage-2 pre-training 在 chassis control 和 torso control 上 **substantially better**
- Stage-1 和 π0 在这些 skills 上 **worse than scratch**

Intuition：OXE 里的 robots 主要是 fixed-base single-arm (e.g., Franka, WidowX)，没有 mobile base 和 torso。当 model 在这些数据上 pre-train 后，它学到的 "action prior" 是 fixed-base 的，反而干扰了 mobile whole-body control 的学习。

### 4.3 Few-shot Transfer (Figure 10)

20 trajectories fine-tuning on Table Bussing 和 Microwave Operation：
- Stage-2 pre-trained models **significantly outperform** others
- Stage-1 alone **no clear advantage** over scratch

这说明 single-embodiment pre-training 提供了 **embodiment-specific inductive bias**，让 few-shot learning 更 efficient。Cross-embodiment pre-training 没有提供这种 bias。

### 4.4 G0-VLM Evaluation (Table 1)

| Model | Table Bussing | Microwave | Bed Making | Blocks |
|-------|---------------|-----------|------------|--------|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-32B | 21.3 | 14.8 | 54.2 | 21.0 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM 超越 baselines **50%+**。关键 insight：**general-purpose VLMs 不足以做 robotic planning**。即使 Gemini-2.5-pro 这样强大的模型，在没有 domain adaptation 的情况下，instruction accuracy 只有 15-55%。原因：
1. General VLMs 不理解 action primitives（什么是 "pickable"，什么是 "graspable"）
2. 缺乏 robot's affordance knowledge
3. 没有学习过 command-observation alignment in robotic context

---

## 5. 与相关工作的深度联系

### 5.1 VLA 模型谱系

| Model | Action Generation | Base VLM | Key Innovation |
|-------|-------------------|-----------|----------------|
| RT-1 [25] | Transformer (discrete) | - | First scalable VLA |
| RT-2 [25] | Autoregressive | PaLM-E | Web knowledge transfer |
| OpenVLA [10] | Autoregressive | Llama-2 | Open-source, 7B |
| π0 [12] | Flow matching | PaLiGemma | Flow matching for VLA |
| RDT-1B [13] | Diffusion | - | Bimanual, 1B |
| CogACT [14] | Hybrid | - | Cognition + action synergy |
| HybridVLA [17] | Hybrid | - | Diffusion + autoregression |
| **G0-VLA** | **Flow matching** | **PaLiGemma** | **3-stage curriculum, dual-system** |

References:
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B: https://arxiv.org/abs/2410.07864
- CogACT: https://arxiv.org/abs/2411.19650
- HybridVLA: https://arxiv.org/abs/2503.10631
- TinyVLA: https://arxiv.org/abs/2409.12514

### 5.2 Dual-System Robotics 历史

- **TAMP** [2]: 早期 task planning + motion control 解耦
- **SayCan** [3]: LLM as zero-shot planner (https://say-can.github.io/)
- **Code as Policies**: LLM generates executable code
- **VoxPoser**: VLM composes 3D value maps
- **Hi Robot** [19]: Hierarchical VLA with open-ended instructions
- **OpenHelix** [18]: Open-source dual-system VLA (https://arxiv.org/abs/2505.03912)
- **G0**: This paper

### 5.3 大规模 Manipulation Datasets 对比

| Dataset | Hours | Embodiments | Scenes | Annotation |
|---------|-------|-------------|--------|------------|
| BridgeData V2 [21] | ~13K | Single (WidowX) | Limited | Task-level |
| DROID [22] | ~76K | Single (Franka) | Lab | Task-level |
| Open-X-Embodiment [1] | ~1M+ | 22+ robots | Diverse | Mixed |
| AgiBot World [24] | Large | AgiBot | Lab | Subtask |
| RoboMIND [23] | Large | Multi | Lab | Subtask |
| **Galaxea** | **500** | **Single** | **50 real-world** | **Subtask** |

Galaxea 的 uniqueness：**real-world scenes** (residential, retail, catering, office) 而非 controlled lab environments。

References:
- BridgeData V2: https://arxiv.org/abs/2309.00918
- DROID: https://arxiv.org/abs/2403.12945
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877

---

## 6. Critical Analysis & Open Questions

### 6.1 Cross-embodiment Pre-training 的 Paradox

这篇 paper 最 provocative 的发现：**cross-embodiment pre-training 可能有害**。这与社区主流 narrative (OXE, RT-X) 矛盾。

可能的解释：
1. **Embodiment gap metric**：需要量化 embodiment similarity。Galaxea R1 Lite (23-DoF, mobile, bimanual) vs OXE robots (mostly 7-DoF fixed-base single-arm) 的 gap 太大
2. **Action space alignment**：不同 embodiments 的 action space dimension 和 semantics 不同，即使 tokenize 后也 misaligned
3. **Negative transfer**：当 source 和 target domain 差异过大时，会发生 negative transfer (reference: https://arxiv.org/abs/2507.06219)

但这个结论需要 caveat：
- Stage-1 仍然对 "universal action patterns" (pick-and-place, push-pull) 有帮助
- 问题出在 **embodiment-specific skills** (whole-body, mobile base)
- 可能的 solution：**selective cross-embodiment pre-training** — 只 pre-train 在 embodiment-similar 的数据上

### 6.2 Dual-System 的 Latency Analysis (missing)

Paper 没有详细讨论 VLM 和 VLA 的 **异步运行机制**：
- VLM 的运行频率是多少？
- VLA 的 action chunk horizon k 是多少？
- VLM 生成 subtask 的延迟如何处理？
- 如果 VLM 在 VLA 执行中途改变 subtask，如何平滑过渡？

这些是 real-world deployment 的关键问题。Hi Robot [19] 对此有更详细讨论。

### 6.3 Flow Matching vs Autoregression 的 Trade-off

G0 在 Stage-1 用 autoregressive (FAST tokenizer)，Stage-2 用 flow matching。这个混合策略很有意思：

- **Autoregressive**：leverage VLM 的 pretrained knowledge，但 slow
- **Flow matching**：higher throughput，但 risk degrading VLM capabilities

G0 的 solution：Stage-1 只训练 VLM（autoregressive），Stage-2 才引入 flow matching action expert。这样 VLM 的 representations 在 Stage-1 已经 stabilized，Stage-2 的 flow matching 不会 harm 它。

### 6.4 Scalability Concerns

500 hours single-embodiment data — 这个规模相对于 OXE 的 1M+ hours 还是小。问题：
- Single-embodiment pre-training 的 scaling law 是什么？
- 当 Galaxea dataset 扩展到 5000 hours 时，性能会如何变化？
- Single-embodiment 的 generalization ceiling 在哪里？

### 6.5 G0-VLM 的 Reasoning Capability

Paper 用 DeepSeek-R1 合成 human-style instructions，但 **没有 eval VLM 的 reasoning depth**。未来工作可能需要：
- Chain-of-thought planning evaluation
- Multi-step lookahead planning
- Error recovery planning

---

## 7. 对 Future Research 的 Implications

### 7.1 Embodiment-Aware Pre-training

未来可能需要 **embodiment embedding** — 让 model 知道自己在 pre-train 什么 embodiment，从而决定是否 transfer 某些 skills。

### 7.2 Real-World Data > Lab Data

Galaxea 证明了 real-world scene diversity 的重要性。这可能推动更多 in-the-wild data collection efforts。

### 7.3 Hybrid Action Generation

G0 用了 autoregressive (Stage-1) + flow matching (Stage-2) 的 hybrid。HybridVLA [17] 探索了更复杂的 hybrid architectures。这可能是一个 promising direction。

### 7.4 VLM-VLA Co-training

G0 的 VLM 和 VLA 是 separately trained 的。未来可能需要 **joint training** — 让 VLM 的 planning 和 VLA 的 execution 互相 calibrate。

---

## 8. 总结

这篇 paper 的核心 contribution 不是 architecture novelty（dual-system 和 flow matching 都是 existing ideas），而是 **rigorous empirical study**：

1. **Dataset contribution**：500 hours real-world, single-embodiment, subtask-annotated
2. **Training curriculum**：3-stage，解决 cross-embodiment pre-training 的 pitfalls
3. **Key finding**：Cross-embodiment pre-training 可以 **harm** performance when embodiment gap is large

对于 VLA community，这是一个 **cautionary tale** — 不要盲目相信 "more data is always better"。Data 的 **alignment** with target embodiment 和 task distribution 比 sheer volume 更重要。

最后，Galaxea 团队承诺 open-source dataset 和 models (https://opengalaxea.github.io/G0/)，这对 community 是巨大的 gift，尤其是 real-world scene diversity 这一维度。

---

**Key References:**
- G0 Project: https://opengalaxea.github.io/G0/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- PaLiGemma: https://arxiv.org/abs/2407.07726
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Flow Matching: https://arxiv.org/abs/2208.14535
- SayCan: https://say-can.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- OpenHelix: https://arxiv.org/abs/2505.03912
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2309.00918
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B: https://arxiv.org/abs/2410.07864
- HybridVLA: https://arxiv.org/abs/2503.10631
- CogACT: https://arxiv.org/abs/2411.19650
- Kahneman's Theory: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
