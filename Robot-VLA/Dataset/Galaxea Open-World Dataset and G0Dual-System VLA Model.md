---
source_pdf: Galaxea Open-World Dataset and G0Dual-System VLA Model.pdf
paper_sha256: 151aa3493bcf6c52a62a30e0f84a1ba686dd22a84fc7de602a92abd5e49639c1
processed_at: '2026-08-04T11:51:54-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 G0 这篇 Paper

好，我换个画风，咱们像在咖啡馆白板上画图一样聊这篇 paper。

---

## 这帮人到底干了啥

Galaxea 这个团队造了一款 mobile bimanual robot，叫 R1 Lite，23 个自由度，能满地跑，有两只手臂，能干家务活。然后他们让这机器人去 50 个真实场景里（厨房、客厅、办公室、零售店）采了 500 小时的操作数据，100K 条 trajectory，150 种 task。

基于这些数据，他们训了一个 dual system：
- 上面一个 "大脑"（VLM），听人说话，看场景，想下一步该干啥
- 下面一个 "小脑"（VLA），拿到指令，直接输出 action chunk 控制关节

就这么简单。

---

## 为什么不直接用一个 end-to-end VLA？

直觉上你觉得一个 model 端到端多清爽。但实际 deployment 时有个很 ugly 的问题：**VLM 太慢了**。

你想想，Qwen2.5-VL 这种 model，72B 参数，跑一次 forward 要几百毫秒。但机器人控制 loop 要 30-50Hz，每 20-30ms 就得出一个 action。你把大 VLM 塞进 control loop 里，机器人就卡成 PPT 了。

所以 dual system 的核心 motivation 就是 **decouple frequency**：
- VLM 以 1-2Hz 低频跑，输出 "下一个 subtask 是啥"（比如 "把盘子放进微波炉"）
- VLA 以 30-50Hz 高频跑，拿到 subtask instruction 后自己连续输出 action

VLM 慢没关系，它 plan 完一次，VLA 可以 execute 好几秒。这就像你开车：你的 conscious mind（System 2）每隔几秒决定 "该变道了" 或 "该减速了"，但你的肌肉控制（System 1）是连续高频在微调方向盘和油门的。

这个 pattern 在 SayCan [3]、Hi Robot [19]、OpenHelix [18] 里都有，是 robotics 领域越来越 standard 的设计。

参考：
- SayCan: https://say-can.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417

---

## Dataset 的赌注：Single Embodiment, Open World

现在 robotics data 领域有两条路线在打架：

**路线 A（OXE 派）**：能搞多少种 robot 就搞多少种，能搞多少 scene 就搞多少 scene，追求 diversity。Open X-Embodiment [1] 就是这个思路，聚合了 20 多种 robot 的数据。

**路线 B（Galaxea 派）**：用一种 robot，但去真实世界采数据，追求 scene richness × annotation quality。

Galaxea 押的是路线 B。

为什么？因为 action space 的 consistency 太重要了。你把 Franka 的 7-DoF arm action 和 mobile bimanual 23-DoF whole-body action 混在一起训，model 要花大量 capacity 去 figure out "这个 action 是给哪种 body 用的"。这就像把中文、日文、韩文混在一起训一个 model，但每个 sample 不告诉你语言标签 — model 得自己猜，猜错了就 garbage。

Galaxea 的 bet 是：**lock 死 embodiment，让 action expert 把全部 capacity 花在 vision-language grounding 上**。

这个 bet 在实验里被 validate 了 — Stage-2 single-embodiment pre-training 在所有 benchmark 上都吊打 Stage-1 cross-embodiment pre-training。

---

## 三阶段训练：用人话讲

### Stage-1：让 VLM "见世面"

数据：1,700 小时混合数据（OXE + Galaxea high-level + in-house）

做法：只训 VLM，不训 action expert。用 FAST tokenizer 把 action chunk 转成 discrete token，然后用标准 next-token prediction 训练。

这阶段的目的：让 VLM 理解 "抓"、"放"、"推" 这些 semantic concept，理解物体怎么运动，理解 instruction 和 visual scene 的对应关系。**不追求学到具体 motor skill**。

类比：就像你看了一万小时烹饪视频，你知道 "翻炒" 是什么意思，知道鸡蛋煮熟会变白，但你手没动过，不会真炒。

### Stage-2：让 VLA "练肌肉"

数据：Galaxea Open-World Dataset 全量（500h，subtask-level annotation）

做法：VLM weights 从 Stage-1 继承，新加一个 action expert，用 flow matching loss 训练。

这阶段目的：让 action expert 学会在 R1 Lite 这个 specific body 上生成 precise action。因为所有数据都是同一个 robot 采的，action space 完全 consistent，model 可以纯粹 focus on "看到这个画面 + 听到这个指令 → 输出这组关节角度"。

类比：你终于进了厨房，手握锅铲，开始实际练习翻炒。你的 "烹饪知识"（Stage-1）帮你理解该做什么，你的 "肌肉记忆"（Stage-2）帮你做到。

### Post-training：考前冲刺

每 task 用 ≤100 条 trajectory 精调，测 generalization。

---

## Flow Matching 到底在干嘛

这个我觉得值得用最朴素的方式讲一下。

你有一段 ground-truth action chunk $A_t$（比如未来 50 步的关节角度序列）。你想训一个 network 去生成这种 action chunk。

Flow matching 的思路：

1. 在 noise $\varepsilon$ 和 ground truth $A_t$ 之间画一条直线
2. 在这条线上随机取一个点 $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$，$\tau \in [0,1]$
3. 让 network 预测这条线的 "方向"（velocity），即 $u = A_t - \varepsilon$
4. Loss 就是预测 velocity 和真实 velocity 的 MSE

$$\mathcal{L} = \|\nu_\theta(A_t^\tau, \tau, \text{context}) - (A_t - \varepsilon)\|^2$$

Inference 时从纯 noise 出发，沿 network 预测的方向走 10-20 步 Euler integration，就生成了一段 action chunk。

这比 autoregressive 快（不需要逐 token decode），比经典 diffusion 简单（ODE 不用 SDE），是 π0 [12] 拓展出来的范式。

参考:
- π0: https://arxiv.org/abs/2410.24164
- Flow Matching for generative modeling: https://arxiv.org/abs/2210.02747
- Conditional Flow Matching 原始 paper: https://arxiv.org/abs/2305.10410

---

## 最反直觉的发现：Cross-embodiment Pre-training 可能有害

这是整篇 paper 最 punchy 的结论。

他们比了 6 个 configuration：

| Config | Stage-1 (cross-emb) | Stage-2 (single-emb) | 结果 |
|---|---|---|---|
| Scratch | ✗ | ✗ | baseline |
| Stage-1 only | ✓ | ✗ | **比 scratch 还差** |
| Stage-2 200h | ✗ | ✓ | 好 |
| Stage-2 400h | ✗ | ✓ | 更好 |
| Full | ✓ | ✓ | 最好 |
| π0 baseline | π0 official | - | 中等偏下 |

Stage-1 only 比 scratch 还差！这意味着在 OXE 上 pre-train 了 1,700 小时，结果不仅没帮助，还拖了后腿。

为什么？我的 intuition：

OXE 里的 robot 几乎都是 fixed-base single-arm（Franka、WidowX、XArm），没有 mobile chassis，没有 pitch-able torso。Galaxea R1 Lite 是 23-DoF whole-body mobile robot。这两者的 action space 差异巨大 — 不是一个 distribution 的问题，是两个几乎 disjoint distribution。

Stage-1 让 VLM 学到的 action prior 是 "fixed-base arm 怎么动"，到了 Stage-2 要 unlearn 这个 wrong prior 再重新学 mobile whole-body 怎么动。unlearn 比 learn from scratch 更难。

特别是 Bed Making 这个 task — 它需要 chassis 移动 + torso 前倾后仰 + arm 协调。OXE 里压根没有这种 data。Stage-1 pre-training 在这个 task 上不仅没 transfer，还 actively hurt。

这就像你学了 10 年开手动挡轿车，然后让你开自动挡叉车 — 你的 muscle memory（踩离合、换挡）不仅没用，还会干扰你。

这个发现对 OXE 社区的 "generalist robot foundation model" 叙事是个 challenge。可能的 synthesis 是：

**Cross-embodiment pre-training 适合学 world model / visual grounding（Stage-1 只训 VLM 就 OK），但不适合直接学 action policy（需要 single-embodiment data）。**

G0 的三阶段正好印证了这个 decomposition — Stage-1 只训 VLM（学 world knowledge），Stage-2 才训 action expert（学 motor skill）。

---

## G0-VLM 训练里的一个小 trick

他们用 DeepSeek-R1 [DeepSeek-R1: https://arxiv.org/abs/2501.12948] 来 synthesize training data。

具体做法：
- 从 dataset 里取一段 trajectory 的 text annotation（subtask sequence）
- 喂给 DeepSeek-R1，让它 imagine 场景，生成 human-style instruction
- 比如 subtask sequence 是 "approach chair → grasp chair → pull back"，R1 生成 "I'm going to sit down, can you pull the chair out for me?"

**关键：不喂 image 给 R1，只喂 text**。Paper 的 argument 是 annotation 质量够高，text-only reasoning 足够 infer 场景。

这个 trick 的价值在于：把 reasoning LLM 的能力用在 **offline data curation** 而不是 online inference。Reasoning model 太慢不能放进 robot control loop，但用来生成 training data 完美 — 反正只跑一次。

我觉得这个 pattern 未来会被更多工作采用。

---

## Benchmark 设计的 intuition

四个 task 各有针对性：

| Task | 设计意图 |
|---|---|
| Table Bussing | 测 precise pick-and-place + dual-arm coordination |
| Microwave Operation | 测 appliance interaction + multi-step sequencing |
| Bed Making | 测 **whole-body control**（chassis + torso + arms 协调） |
| Blocks Stacking | 测 language following + precise placement |

Bed Making 是最 interesting 的 — 它是专门设计来 expose cross-embodiment pre-training 的 weakness 的。因为 whole-body mobile manipulation 是 OXE 里完全没有的能力维度。

这个 benchmark design 的 lesson 是：**你的 benchmark 必须覆盖你的 target embodiment 的 unique capability，否则 cross-embodiment pre-training 的 failure mode 会被 hidden**。

---

## G0-VLM 的 SFT 效果

Table 1 里 G0-VLM（fine-tuned Qwen2.5-VL）在 instruction accuracy 上吊打所有 baseline：

- Gemini-2.5-pro: 32% on Table Bussing
- Qwen2.5-VL-72B: 26.3%
- G0-VLM: **83.3%**

这说明了什么？**general-purpose VLM 不懂 robot action primitive**。你给 Gemini 看一张桌面照片问 "下一步该干啥"，它会输出一堆 natural language，但不是 VLA 能 execute 的 atomic instruction。

SFT 的价值是 **把 VLM 的 output space 从 free-form text 收敛到 executable action vocabulary**。这是 domain adaptation，不是 capability 提升。

---

## 我的 Take-away

1. **Embodiment consistency > embodiment diversity**，至少在 action policy learning 这个层面。Cross-embodiment data 可以用来训 VLM 的 world knowledge，但不要直接用来训 action expert。

2. **Dual system 是 deployment necessity**，不是 architecture preference。VLM 太慢，必须和 VLA 异步。

3. **Flow matching 是当前 VLA action generation 的 best practice**，比 autoregressive 快，比经典 diffusion 稳。

4. **Subtask-level annotation 是 key**。Task-level annotation 太粗，model 学不到 instruction-action 对应。Galaxea 的 subtask segmentation 是它能 work 的重要 enabler。

5. **Reasoning LLM 做 offline data synthesis** 是一个被低估的 trick，未来会 scale。

6. **Benchmark 必须 probe target embodiment 的 unique capability**，否则 pre-training failure 不可见。

---

**一些值得继续追的 thread**:

- π0.5 [16] 的 cross-embodiment transfer 是怎么 work 的？它和 G0 的矛盾怎么 reconcile？可能 π0.5 的 embodiment gap 更小。https://arxiv.org/abs/2504.16054
- "Is diversity all you need?" [27] 也在 question embodiment diversity，值得对照读。https://arxiv.org/abs/2507.06219
- HybridVLA [17] 的 autoregressive + diffusion 混合设计和 G0 的 Stage-1 autoregressive + Stage-2 flow matching 有没有 deeper connection？https://arxiv.org/abs/2503.10631
- FAST tokenizer [11] 的 DCT 压缩对 action chunk 的 frequency component 有什么影响？高频 action（精细操作）会不会被 DCT 滤掉？https://arxiv.org/abs/2501.09747
- AgiBot World [24] 也是 single-embodiment + real scene，和 Galaxea 的方法论很像，两者对比能 reveal 什么？https://arxiv.org/abs/2503.06669

总之这篇 paper 最值得记住的一句话：**在 physical AI 里，data 的 "一致性" 可能比 "多样性" 更重要 — 至少在 action 层面是这样。**

---

# Galaxea Open-World Dataset and G0 Dual-System VLA Model — 深度解读

Andrej, 这篇paper 在我看来是 physical AI 领域一个相当有意思的 data-centric + curriculum-driven 的工作。它和 π0、OpenVLA、AgiBot World 这一波工作形成了一个明确的对照。我把它拆成几层来讲，目的 是帮你 build intuition about **why single-embodiment pre-training dominates cross-embodiment pre-training in this regime**，这是这篇paper 真正的反直觉发现。

---

## 1. Paper 的整体定位

这篇工作的核心 claim 有三件事：

1. **Data layer**: 发布 Galaxea Open-World Dataset — 500 小时、100K trajectories、150 tasks、50 scenes、1,600 objects、58 skills，全部用同一个 Galaxea R1 Lite 平台采集，subtask-level language annotation。
2. **Model layer**: G0 dual-system — System 2 (G0-VLM, Qwen2.5-VL based) 做 high-level planning，System 1 (G0-VLA, PaLiGemma + flow-matching action expert) 做 low-level execution，两者异步运行。
3. **Training curriculum**: 三阶段 — cross-embodiment pre-training (Stage-1) → single-embodiment pre-training (Stage-2) → task-specific post-training。

真正 punchy 的实验结论是：**当 target embodiment 与 cross-embodiment pre-training 数据之间存在 large morphology gap 时，Stage-1 不仅帮助有限，甚至可能 degrade 性能**。这和 Open X-Embodiment 社区主流叙事 "more diverse data = better generalist" 产生了 tension。

参考链接：
- Galaxea project page: https://opengalaxea.github.io/G0/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 (open-world generalization): https://arxiv.org/abs/2504.16054

---

## 2. Dataset 设计哲学：Why "Single Embodiment + Open World"?

### 2.1 与现有 dataset 的对照

让我把它和目前主流 datasets 放在一张 mental table 里：

| Dataset | Embodiment 多样性 | Scene 真实度 | Annotation 粒度 | Scale |
|---|---|---|---|---|
| BridgeData V2 [21] | single ( WidowX ) | lab / kitchen | task-level | 60K demos |
| DROID [22] | single (Franka variants) | diverse labs | task-level | 76K demos |
| Open-X-Embodiment [1] | multi (20+ robots) | mixed | mixed | 1M+ demos |
| AgiBot World [24] | single (AgiBot) | semi-controlled | subtask | 100K demos |
| RoboMIND [23] | multi | controlled | subtask | 55K demos |
| **Galaxea Open-World** | **single (R1 Lite)** | **real human env** | **subtask** | **100K / 500h** |

Galaxea 的赌注是：**与其追求 embodiment diversity，不如追求 scene diversity × embodiment consistency**。这是一个非常 specific 的 bet，背后的 intuition 是 VLA model 学 action 时最大的 variance 来源 是 action space 本身，把 embodiment 锁死可以让 action expert 把 capacity 全部花在 vision-language grounding 上。

### 2.2 Hardware: Galaxea R1 Lite

23-DoF：
- 2 × 6-DoF arms (spherical wrists, parallel grippers, 5kg payload, 60cm reach)
- 3-DoF torso (vertical + pitch)
- 6-DoF vector-drive omnidirectional base (max 1.5 m/s)

这是一个 mobile bimanual platform，和 π0 用的 Franka + mobile base 或者 ALOHA 的 tabletop setup 都不同。chassis + torso + dual-arm 的 whole-body coordination 是它独有的能力维度，paper 后面 Bed Making task 就是专门 test 这个维度。

### 2.3 Teleoperation: Isomorphic Mapping

这里有一个我觉得很聪明的设计选择：**isomorphic teleoperation**，直接把 human operator 的关节映射到 robot kinematics，不走 VR + retargeting pipeline。

好处：
- 避免 IK failure
- 保证 arm 在 reachable posture 内
- 不需要 human-to-robot morphology retargeting

代价：operator 必须 physically co-locate with robot。这个 trade-off 在 data scale 不是 ultimate bottleneck 的时候是值得的，因为 retargeting 引入的 noise 会污染 action grounding。

参考：
- DROID teleop design: https://droid-dataset.github.io/
- ALOHA / Mobile ALOHA: https://tonyzhaozh.github.io/aloha/

---

## 3. G0 Dual-System Architecture

### 3.1 为什么是 Dual System？

Paper 借鉴 Kahneman 的 System 1 / System 2 framework [4]。这个 paradigm 在 robotics 上的最近复兴来自 SayCan [3]、Code as Policies、VoxPoser 这一脉，以及最近的 Hi Robot [19]、OpenHelix [18]、VLA-OS [20]。

G0 的设计：
- **G0-VLM (System 2)**: Qwen2.5-VL base，low frequency，做 high-level task decomposition，输出 atomic subtask instructions
- **G0-VLA (System 1)**: PaLiGemma + flow-matching action expert，high frequency，做 reactive control

两者 **asynchronous** 运行，这是关键 — System 2 慢思考不需要等 System 1 的每一个 action step，可以并行 re-plan。

### 3.2 G0-VLA 内部架构

在 time step $t$，输入：
- $o_t$: 三路 camera observation (1 stereo RGB head + 2 wrist RGB-D)
- $l_t$: language instruction (subtask level)
- $s_t$: robot proprioceptive state (23-DoF joint positions/velocities)

输出：
- $\mathbf{A}_t = a_{t:t+k}$: action chunk with horizon $k$ (receding horizon control)

VLM 部分结构：
1. **SigLIP vision encoder** 处理三张图像 → image embeddings
2. **Single-layer MLP projector** → projected visual tokens
3. 这些 visual tokens 进 Transformer，与 tokenized language + proprioception + action tokens 做 cross-attention

Action expert 部分：
- 接 VLM 的 KV cache
- 用 **flow matching** 生成 continuous action

这是 π0 的 design pattern 的直接继承 [12]，paper Section 2 也明确说 "employs two training methods similar to that of π"。

---

## 4. Three-Stage Training Curriculum — 最核心的部分

这是 paper 的灵魂。我要把每一阶段的 loss、数据、motivation 都讲透。

### 4.1 Stage-1: Cross-Embodiment Pre-training (Autoregressive)

**数据**: ~1,000h OXE + 500h Galaxea (high-level task description only, no subtask annotation) + 200h in-house (high-level only)

**关键设计**: 只 train VLM component，不 train action expert。

**Action tokenizer**: FAST tokenizer [11] — 把 continuous action chunk 转成 discrete token sequence。FAST 的核心是 1D DCT (Discrete Cosine Transform) 压缩 + byte-pair encoding，把一段 action chunk 压成 256-token 词汇表里的 ~30-40 tokens。

**Loss**: 标准 next-token prediction cross-entropy：

$$
p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)
$$

变量解释：
- $\mathbf{A}_t^d$: 在 time $t$ 的 discrete action token sequence，长度 $N$（$N \approx 30-40$，由 FAST 决定）
- $a_i^d$: 第 $i$ 个 discrete action token
- $a_{<i}^d$: 已经生成的前 $i-1$ 个 action tokens（autoregressive conditioning）
- $o_t, l_t, s_t$: 视觉、语言、本体感觉观测

**为什么 Stage-1 只 train VLM 不 train action expert**：
1. Cross-embodiment 数据的 action quality 不一致，不同 robot 的 action space semantics 不一样，flow matching 学不到稳定 signal
2. Diffusion / flow loss 在 representation 还没 converge 之前可能 hurt learning

我的 intuition：Stage-1 本质上是在做 **"VLM 学习世界知识"**，让 vision encoder + LLM 理解 "什么是抓"、"什么是放置"、"物体如何运动"，但不强求它学到具体 motor priors。

### 4.2 Stage-2: Single-Embodiment Pre-training (Flow Matching)

**数据**: Galaxea Open-World Dataset 全量（500h，subtask-level annotation）

**架构**: VLM (from Stage-1) + newly initialized action expert

**Loss**: Flow matching，最大似然等价：

$$
\max_\theta \mathbb{E}_{p(A_t, o_t, l_t, s_t)} \left[ \log \pi_\theta(A_t \mid o_t, l_t, s_t) \right]
$$

具体 flow matching loss：

$$
\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)} \left[ \| \nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t) \|^2 \right]
$$

变量解释：
- $A_t$: ground-truth action chunk，horizon $H$
- $\tau \in [0, 1]$: flow time parameter，表示从 noise 到 data 的插值进度
- $\varepsilon$: Gaussian noise source
- $A_t^\tau = \tau A_t + (1-\tau) \varepsilon$: linearly interpolated noisy action — 当 $\tau=0$ 是 pure noise，当 $\tau=1$ 是 ground truth
- $\nu_\theta(\cdot)$: VLA 预测的 flow velocity field (neural network output)
- $u(A_t^\tau \mid A_t) = A_t - \varepsilon$: target flow velocity (linear interpolation 的 constant velocity)

Inference 时从 $\tau=0$ 的 noise 出发，用 Euler method (或更高阶 ODE solver) 沿 $\nu_\theta$ 预测的 flow field 走到 $\tau=1$，得到 action chunk。

这是 Conditional Flow Matching (CFM) 的标准 form，和 π0、RDT-1B [13] 是一个 family。

**Stage-2 的两个关键 enabler**：
1. **Single embodiment**: action space 完全一致，action expert 不需要跨 embodiment adapt
2. **Language-action alignment**: subtask-level segmentation 让 instruction 和 trajectory 有 fine-grained 对应

### 4.3 Post-training: Task-specific Fine-tuning

每 task 用 ≤100 trajectories，和 Stage-2 同样的 flow matching loss。这部分是 test generalization 的 probe。

---

## 5. G0-VLM 训练 — System 2 怎么 build

### 5.1 Base model: Qwen2.5-VL [30]

参考: https://arxiv.org/abs/2502.13923

### 5.2 数据构造 pipeline

1. 从 Galaxea dataset 采样 episode
2. **Key frame sampling**: 在 subtask 即将终止 / gripper state 变化时给 higher sampling weight — 这个设计是为了让 VLM 学到 task transition
3. 提取 head camera image + subtask annotation
4. 用 **k-frame historic context**: 过去 $k$ 秒的 image + action，让 VLM 有 temporal memory

得到 $D_{\text{labeled}} = \{ \text{task name}, o_{t-k:t}, l_{t-k:t} \}$

### 5.3 用 DeepSeek-R1 做 instruction synthesis

这一步很 clever：用 reasoning LLM 在 $D_{\text{labeled}}$ 上生成 human-style high-level instruction 和 robot verbal response。

**Prompt 给 LLM 的内容**:
- Task name (e.g. "pull and push chairs")
- Historic + current subtasks
- Next subtask

**LLM output**:
- Human-style verbal instruction ("I am going to be seated, could you help pull the chair out?")
- Robot's response ("I am working on it!")

**关键 design choice**: **不 feed image 给 reasoning LLM**，只用 text annotation。Paper 的 argument 是 atomic action annotation 质量足够高，LLM 的 reasoning 能力足够 infer 场景。这避免了 multimodal reasoning LLM 的 cost，同时把 image grounding 留给 fine-tuned VLM 自己学。

参考:
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Hi Robot (类似 hierarchical 设计): https://arxiv.org/abs/2502.19417

---

## 6. 实验设计与发现 — Build Intuition

### 6.1 Benchmarks (4 个 task，每个有 progress score)

| Task | Points | 测试维度 |
|---|---|---|
| Table Bussing | 6 (3 pick + 3 place) | precise pick-and-place, dual-arm coordination |
| Microwave Operation | 5 | appliance interaction, multi-step |
| Bed Making | 4 | **whole-body control (chassis + torso + arms)** |
| Blocks Stacking | 6 | language following, precise placement |

每个 task 跑 10 次，取平均 progress score。100 training trajectories per task，4 epochs。

### 6.2 关键实验 1: Pre-trained Weights Comparison

比较 6 个 configuration：

| Config | Stage-1 | Stage-2 | Total pretraining |
|---|---|---|---|
| G0 (Scratch) | ✗ | ✗ | 0 |
| G0 (Stage-1) | ✓ | ✗ | cross-embodiment only |
| G0 (Stage-2 200h) | ✗ | ✓ (200h) | single-emb only |
| G0 (Stage-2 400h) | ✗ | ✓ (400h) | single-emb only |
| G0 (Full) | ✓ | ✓ (400h) | both stages |
| π0 (baseline) | π0 official weights | - | Physical Intelligence's recipe |

**Findings (Figure 9)**:
- **G0 (Full)** 最高 average progress score — 在 Table Bussing / Microwave / Bed Making 上 object-picking 最强
- **G0 (Stage-2 400h)** 和 **G0 (Stage-2 200h)** 在 language following、action consistency、whole-body control 上最好
- **G0 (Stage-1) 最差**，比 from scratch 还差 — 这是 paper 的关键反直觉发现

**Intuition**: Stage-1 学到的是 "universal action patterns" (pick, place, push, pull)，这些 high-level pattern 可能在 cross-embodiment 数据上学到的是 "average" action distribution，反而把 Stage-2 学 specific morphology 的 prior 给 wash out 了。

### 6.3 关键实验 2: Few-shot Transfer (20 trajectories, 10 epochs)

测试 Table Bussing + Microwave Operation。

**Findings (Figure 10)**:
- Stage-2 pre-training 显著提升 few-shot 成功率和 execution smoothness
- **Stage-1 alone 没有明显优势 over scratch** — 这是又一个 cross-embodiment pre-training 失效的证据

### 6.4 关键实验 3: Embodiment-specific Actions (Bed Making per-skill)

Bed Making 拆成 4 个 skill：
1. Move toward bed (chassis)
2. Lift torso + grasp quilt (torso + arms)
3. Lean torso back (torso)
4. Move to flatten quilt (chassis + arms)

**Findings (Figure 11)**:
- Stage-2 single-embodiment pre-training 大幅提升 chassis + torso 控制
- Stage-1 cross-embodiment + π0 在 chassis actions 上 instruction following 弱，torso 控制 less accurate
- **某些 skill 上 cross-embodiment pre-training 比 scratch 还差**

**Hypothesis**: OXE dataset 里的 robot (Franka, WidowX, XArm 等) 几乎都是 tabletop fixed-base，没有 mobile chassis 和 pitch-able torso 这种 DoF。pre-training 让模型学到的 action prior 在这些维度上是 wrong，反而要 unlearn。

### 6.5 G0-VLM 评估 (Table 1)

比较 Gemini-2.5-pro, Qwen2.5-VL-7B/32B/72B, G0-VLM (fine-tuned)：

| Model | Table bussing | Microwave | Make bed | Build blocks |
|---|---|---|---|---|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-32B | 21.3 | 14.8 | 54.2 | 21.0 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM 在所有 task 上 accuracy **超过 baseline 50%+**。这 validate 了 paper 的 hypothesis：robotic application 需要的 不只是 general VLM understanding，而是 action-grounded instruction generation，必须通过 domain-specific SFT 获得。

---

## 7. 我的 Intuition 与联想

### 7.1 关于 Cross-embodiment Pre-training 的失效

这个发现让我想到几个 related work：

1. **"Is diversity all you need?" [27]** — Shi et al. 2025: 这篇 paper 也在 question embodiment diversity 的 value，发现 task diversity > embodiment diversity。链接: https://arxiv.org/abs/2507.06219

2. **RT-2 [25]** 的 cross-embodiment transfer 是有效的，但 RT-2 主要 test 的是 fixed-base single-arm robot 之间 transfer，morphology gap 小。

3. **OpenVLA [10]** 的 cross-embodiment fine-tuning 也是在同一类 robot (Franka-like) 内有效。

Galaxea R1 Lite 的 23-DoF whole-body mobile morphology 是 OXE 数据里几乎完全不存在的 distribution。这就像在 language model 上用中文 pre-train 然后去 fine-tune Python code generation — 不是不能 transfer，但 prior 是错的。

### 7.2 关于 Action Representation: Autoregressive vs Diffusion

Stage-1 用 autoregressive (FAST tokenizer)，Stage-2 用 flow matching。这种 **混合范式** 越来越流行：
- π0: VLM autoregressive + flow matching action expert
- HybridVLA [17]: explicit collaborative design
- CogAct [14]: synergize cognition + action

我的 intuition 是 autoregressive 适合 "semantic" action (high-level what to do)，diffusion/flow 适合 "motor" action (low-level how to move)。Stage-1 学 semantic，Stage-2 学 motor，这个 division 是合理的。

参考:
- HybridVLA: https://arxiv.org/abs/2503.10631
- CogAct: https://arxiv.org/abs/2411.19650
- FAST: https://arxiv.org/abs/2501.09747

### 7.3 关于 Dual-System 的 Asynchronous 设计

G0 的 VLM 和 VLA 异步运行 — VLM 低频 (e.g. 1-2 Hz) plan subtask，VLA 高频 (e.g. 10-50 Hz) execute action。这个设计避开了一个大坑：如果 VLM 和 VLA 同步运行，VLM 的 inference latency 会成为 action control loop 的 bottleneck。

这个 pattern 在 Hi Robot [19] 和 OpenHelix [18] 里也有。我预测 future work 会更多往这个方向走，因为 VLM 越来越大 (GPT-5, Gemini 3)，把大 model 放进 control loop 是死路。

### 7.4 关于 Dataset 的 "Real World" Claim

Galaxea 强调 "open-world" "real human environment"。但要小心一个 selection bias：虽然 scene 是 real 的，但 task 是 pre-defined 的 150 个 category，object 是 1,600 个 curated items。这和 truly open-ended instruction following 还有 gap。

AgiBot World [24] 也是类似 design — real scene 但 controlled task。真正 open-ended 的 dataset 可能需要 self-supervised exploration data，那是另一个 frontier。

### 7.5 关于 Reasoning LLM 做 Instruction Synthesis

用 DeepSeek-R1 在 text annotation 上 synthesize human-style instruction — 这个 pattern 我觉得很有 future。它把 LLM 的 reasoning 用在了 "data curation" 而不是 "online inference"，避开了 reasoning model 在 robot control loop 里太慢的问题。同时不让 reasoning LLM 看 image，把 perception 留给 fine-tuned VLM，分工清晰。

类似的思路在 HELM-VLA、ECoE 这类工作里也有出现。

### 7.6 与 RT-2 / OpenVLA 的根本不同

RT-2 [25] 和 OpenVLA [10] 都是 end-to-end VLA，没有显式的 dual-system。G0 把 planning 和 execution 显式 decouple，好处是：
- VLA 不需要学 long-horizon planning，capacity 集中在 motor control
- VLM 可以 leverage 通用 LLM 的 reasoning capability
- 两者可以独立迭代

代价是：
- 需要定义 subtask vocabulary，限制了 open-endedness
- 两个 model 的 training / deployment 复杂度更高

### 7.7 What's Missing / Future Direction

我觉得这篇 paper 有几个没有 fully address 的点：

1. **No comparison to truly end-to-end VLA with long-horizon capability** — 比如 π0.5 [16] 是单 system 但能做 long-horizon，缺少 head-to-head
2. **Subtask vocabulary 的 coverage** — VLM 只能输出 pre-defined atomic actions，这限制了 truly novel task
3. **Failure mode analysis 不够** — paper 没有详细分析 VLA 在哪些 corner case 上 fail
4. **Cross-embodiment pre-training 失效的 boundary** — 多大的 morphology gap 会触发？是否可以用 embodiment encoding 修复？
5. **Real-world generalization to unseen scene** — benchmarks 都是在 dataset 覆盖的 scene type 内，unseen scene 的 zero-shot 没测

---

## 8. 总结

这篇 paper 真正的贡献在我看来 是：

**在 physical AI 这个 regime，"更多样化的 data" 不一定是 "更好的 data"。Embodiment consistency × scene diversity 是比 embodiment diversity × scene consistency 更 productive 的 axis，至少在 single-embodiment deployment 场景下。**

这个 finding 如果被后续工作 replicate，会对 Open X-Embodiment 这一脉的 "通用 robot foundation model" 叙事产生挑战。可能的 synthesis 是：cross-embodiment pre-training 适合学 **world model / visual-language understanding**，single-embodiment pre-training 适合学 **action policy**。G0 的三阶段 curriculum 正好对应这个 decomposition — Stage-1 学 world prior (VLM only)，Stage-2 学 action policy (VLA)。

Action representation 的混合范式 (autoregressive for semantic, flow matching for motor) 也会是 future VLA 的 default pattern。

最后，G0-VLM 用 reasoning LLM offline synthesize instruction 这个 trick 我觉得是一个被低估的 idea，未来会有更多工作在这个方向上做 scaling。

---

**Key References**:
- Galaxea G0: https://opengalaxea.github.io/G0/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- PaLiGemma: https://arxiv.org/abs/2407.07726
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- SayCan: https://say-can.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- OpenHelix: https://arxiv.org/abs/2505.03912
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877
- DROID: https://droid-dataset.github.io/
- BridgeData V2: https://robotlearning.github.io/bridgedata/
- HybridVLA: https://arxiv.org/abs/2503.10631
- CogAct: https://arxiv.org/abs/2411.19650
- RDT-1B: https://arxiv.org/abs/2410.07864
- "Is diversity all you need?": https://arxiv.org/abs/2507.06219
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kahneman "Thinking, Fast and Slow": https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
