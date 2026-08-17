---
source_pdf: THEORY OF SPACE.pdf
paper_sha256: 79bf433cd80d17b97f971eaffec019517c1b12bd1775268f4e1d887ef3abfc01
processed_at: '2026-08-12T15:14:19-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Theory of Space 的人话版

## 这 paper 在干啥

一句话：**测 foundation model 能不能当个有空间感的 embodied agent**。

以前的 spatial benchmark 都是给 model 一张图或一段文字，问 "A 在 B 的哪边"。model 答对了就算 spatial reasoning OK。这 paper 说这不够——真实 agent 在 partial observability 下，得**自己决定往哪走、看什么、什么时候停**，脑子里慢慢拼出一张地图。这跟被动答题完全是两回事。

paper 把这个能力叫 **Theory of Space (ToS)**，跟 Theory of Mind (ToM) 对着来。ToM 测你能不能猜别人脑子里在想啥，ToS 测你能不能猜世界长啥样。

---

## 核心实验长啥样

一个 grid 世界，几个房间，每个房间里几个 object（比如 chair、lamp、vase）。Agent 从一个随机位置出发，视野只有 90° 前方。它能做四件事：
- `Goto` 走到当前可见的某个 object 旁边
- `Rotate` 原地转 90/180/270°
- `Observe` 看一眼 90° FOV 内的东西
- `Query` 拿某个 visible object 的绝对坐标（cost 高，少用）

目标是：**探索完之后，脑子里有一张准确的整个 scene 的地图**。然后 paper 用一堆 spatial task 测这张地图质量——比如 "从 A 看 B 在哪"、"给你一个动作序列，预测最后看到啥"、"给你一个视角，反推你在哪"。

**最聪明的设计**：text world 和 vision world 用同一个 scene seed 跑两遍。Text world 里 observation 是离散符号（"chair 在 front-left、near"），vision world 里是 384×384 的 ego-centric RGB。这样可以把 **perception 失败** 和 **reasoning 失败** 分开。

---

## 三个核心操作

paper 把 ToS 拆成三个能力：

**Construct（建图）**：探索过程中把零碎的 local observation 拼成 globally consistent 的地图。形式上是估计 posterior $B_t(S) \approx P(S \mid h_t)$，$h_t$ 是观察+动作的 history。

**Revise（改图）**：环境变了之后能不能更新 belief。paper 借用 developmental psychology 的 false belief paradigm——Sally-Anne test 的空间版。Agent 探索完之后，偷偷把 4 个 object 搬位置或转方向，让 agent 重新探索并报告变化。

**Exploit（用图）**：用建好的地图解下游 spatial task。分两类：
- **Route knowledge**：egocentric、path-based 的推理（"从这走到那会看到啥"）
- **Survey knowledge**：allocentric、map-like 的推理（"A 在 B 的哪边"、"这个视角对应地图上哪个位置"）

这个 route/survey 的区分来自 spatial cognition 的经典文献（Siegel & White 1975, Montello 1998）。

---

## 不把 model 当黑盒：Belief Probing

这是 paper 最有意思的地方。大多数 benchmark 只看 final answer 对不对。ToS 要求 model **每一步都吐出当前 belief**，形式是 JSON 格式的 cognitive map——所有观察过的 object 的全局坐标和朝向。

这让 paper 能 diagnose 失败模式：
- **Correctness**：最终地图跟 ground truth 对多少
- **Perception**：当前 FOV 里的东西看对了没
- **Stability**：之前看对的东西，后面有没有被 corrupt
- **Local↔Global consistency**：新观察和全局地图有没有矛盾
- **Self-tracking**：agent 知不知道自己在哪
- **Uncertainty**：给一张 top-down 图，能不能认出哪些点还没看过

还有一个 clever 的 validation：给 model oracle ground-truth map，performance 飙到 95%。说明 **map format 本身信息充分，bottleneck 是 model 自己建不出好 map**。

---

## 四个关键发现

### 1. Active-Passive Gap

| | Active（自己探索） | Passive（喂干净轨迹） |
|---|---|---|
| GPT-5.2 text | 72.0 | 90.4 |
| Gemini-3 Pro text | 81.5 | 86.5 |
| GPT-5.2 vision | 46.0 | 57.1 |

被动 setting 给 model 一个 scripted proxy agent 走出来的干净轨迹，model 只需要 reason。主动 setting 让 model 自己规划。差距巨大。

**直感**：model 不是不会 reason，是**不知道下一步该看哪**。最 striking 的是 text world 下 GPT-5.2 掉 18 分——text world 的 perception 已经是 symbol 了，没有 ambiguity，这个 gap **纯粹是 exploration policy 的失败**。

更扎心的对照：让 text model 跟着 SCOUT proxy 的 9 步轨迹走，GPT-5.2 拿 83.9，比自己 active exploration 的 72.0 还高。**model 被牵着走比自己乱逛表现好**。

### 2. Inefficiency

- SCOUT proxy：~9 步覆盖完
- Human：9.8 步
- GPT-5.2 vision：17.2 步
- Claude-4.5 Sonnet vision：19.6 步

Model 走得更多，结果还更差。GPT-5.2 的策略是 "find the door"——看到门就冲过去，经常当前房间没扫完就跳走。Gemini-3 Pro 更系统化，rotate-and-scan，类似 SCOUT。Figure 4 的 information gain 曲线很直观：GPT-5.2 一开始涨快（因为找门策略覆盖广），后面 plateau；Gemini 涨得慢但持续涨。

### 3. Modality Gap：Vision 远差于 Text，尤其是 Orientation

Table 5 拆开看 vision model 的 belief 质量：

| Model | Correctness (ori.) | Correctness (pos.) | Perception | Stability |
|---|---|---|---|---|
| GPT-5.2 | **20.2** | 42.0 | 33.5 | 53.7 |
| Gemini-3 Pro | 32.2 | 62.5 | 52.1 | 61.8 |

**Orientation correctness 只有 20-32%**——接近 chance level。Foundation model 从单视角 image 几乎**不会推断 object 朝哪**。这直接搞死了 perspective-taking task（Table 2: ~36% accuracy）。

更反直觉的是 **Stability 只有 53-62%**。意思是：agent 之前明明看对的东西，后面 turn 会把它 overwrite 成错的。这不是 perception 的问题，是 **belief maintenance 的问题**——long horizon 下 memory 会 corrupt。

### 4. Belief Inertia：旧 prior 拽着不放

False Belief task 里，4 个 object 被偷偷搬走或转方向。Agent 重新探索后报告。

| Model | World | Positional Inertia ↓ | Orientation Inertia ↓ |
|---|---|---|---|
| GPT-5.2 | Text | 5.5 | 12.5 |
| Gemini-3 Pro | Text | 7.9 | 5.7 |
| GPT-5.2 | Vision | **68.9** | **34.7** |
| Gemini-3 Pro | Vision | **51.1** | 14.4 |

Vision agent 的 positional inertia 高达 68.9%。意思：**agent 明明已经看到 object 在新位置了，还是 report 旧坐标**。这是 false belief 的 spatial 版——agent "知道" 变了，但旧 prior 牢牢拽着 belief。

paper 用一个很精细的公式量化这个 inertia，核心是看 revision 后的 residual error 是否还指向旧 prior 方向，并用 proximity weight 防止 "agent 已经 revise 走了但 error 碰巧同向" 的误判。

---

## 这些发现为啥重要

对做 model 的人来说，这 paper 把几个 bottleneck 摆得很清楚：

**1. Active exploration 是独立能力，不能被 reasoning 能力掩盖。** 现在 MLLM benchmark 大多是 passive 的，model 看起来 spatial reasoning 还行。一到 active setting 就崩。这说明我们需要专门训练或设计 exploration policy——可能要引入 information gain reward、curiosity-driven exploration objective，甚至 active inference 框架。

**2. Belief drift 指向 memory architecture 缺陷。** 当前 MLLM 把 history 当 token sequence 喂 transformer，attention 随 token 数稀释。Stability 53% 说明这个架构在 long horizon 下根本撑不住 spatial belief。需要 persistent structured memory——把 episodic observation 压缩成 semantic state（cognitive map），而不是全靠 attention over raw history。

**3. Belief inertia 很可能是 RLHF 副作用。** RLHF 训出来的 model 倾向 "stick to a confident answer"。在需要 belief revision 的场景下，这是 anti-pattern。可能需要新的 training objective——比如 belief tracking loss、或专门针对 false belief scenario 的 DPO。

**4. Vision orientation perception 是硬瓶颈。** Foundation model 从单视角 image 推断 object facing 几乎是 chance level。这是 training data 问题——facing label 稀缺。可能需要 synthetic data（Objaverse + random facing annotation）或 multi-view supervision。

---

## 我的几个联想

**跟 world model 的关系**：ToS 是 foundation model world modeling 的 explicit 版本。Dreamer / JEPA 走 implicit latent representation 路线，ToS 走 explicit structured representation 路线。两个流派各有道理——explicit 更 interpretable、更可 diagnose，但 lossy；implicit 更 expressive 但黑盒。未来可能 converges 到 hybrid：latent world model + structured probing head。

**跟 SLAM 的关系**：ToS 本质是 "SLAM for foundation model"。Classical SLAM 有 perfect sensor 和 motion model，ToS 让 MLLM 用 raw perception 做 SLAM-like 任务。一个有趣的 research direction：能不能把 classical SLAM 的 probabilistic machinery（EKF、particle filter）嫁接到 MLLM 上，作为 external memory module？

**跟 cognitive map neuroscience 的对应**：paper 直接用 Tolman 1948 的 cognitive map 概念，这很 grounding。但 neuroscience 里 place cells / grid cells 的 representation 是 continuous、metric、自动 learned 的；ToS 让 model 输出 discrete JSON，这是人为 imposed 的 representation。一个可能的 follow-up：让 model 自由 form representation，然后用 linear probe 读出它的 latent map——这更接近 neuroscience 的做法。

**Multi-agent 是显然的 next step**：paper 自己提到了。空间 belief 的 multi-agent 版本会引入新挑战—— belief sharing、coordinate frame alignment、谁去看哪的分工问题。这跟 multi-agent RL 里的 decentralied POMDP 很像。

---

## 最终 take

这 paper 的真正贡献是 **把 evaluation paradigm 从 "答对没" 转向 "脑子里地图建得怎么样"**。这个 paradigm shift 让我们能 diagnose 具体 failure mode，而不是只看一个 aggregate score。

最有 diagnostic value 的两个指标是 **Stability**（暴露 belief drift）和 **Belief Inertia**（暴露 prior 的 grip）。这两个 finding 都指向同一个深层问题：**foundation model 缺乏真正的 persistent, revisable structured memory**。当前架构（transformer + token history）在 long-horizon spatial belief 维护上根本不够用。

对 Karpathy 这种做 world model 的人来说，这 paper 提供了一个很清晰的 diagnostic framework——你拿你的 world model 来跑 ToS，看它的 probed cognitive map 质量、stability、inertia 怎么样，就能知道你的 world model 在 spatial belief 上到底行不行。

---

# Theory of Space: 一篇关于 Foundation Model 空间认知的 Benchmark Paper

## 1. 一句话理解这篇 paper 的核心动机

作者把 ToM (Theory of Mind) 的范式搬到 spatial embodied agent 上。ToM 测的是 agent 能不能建模**别人隐藏的心理状态**；ToS (Theory of Space) 测的是 agent 能不能在 partial observability 下，通过 active exploration 主动构造、修正、利用一个**隐藏的物理空间结构 belief**。这个 framing 很重要，因为它把 evaluation 的对象从 "答对多少题" 转向 "你脑子里那张地图形成得怎么样、稳不稳定、能不能改"。

论文网站: https://theory-of-space.github.io/
代码: https://github.com/mll-lab-nu/Theory-of-Space
数据: https://huggingface.co/datasets/MLL-Lab/tos-data

---

## 2. ToS 的形式化定义：三个核心操作

paper 把 ToS 定义为对一个 spatial structure $S \in \mathcal{S}$ 的 posterior 估计能力。Agent 跟环境交互产生 history $h_t = (o_{0:t}, a_{0:t})$，然后有三种 operation：

**Construct (构造)**:
$$B_t(S) \approx P(S \mid h_t)$$

把 partial observation $h_t$ 整合成一个 globally consistent 的 posterior $B_t(S)$。这里 $B_t$ 是 belief，$S$ 是真实的空间结构（包括所有 object 的 (x,y) 坐标和 orientation）。

**Revise (修正)**:

当环境从 $S$ 变到 $S'$ 时，agent 要通过新的 exploration $\Delta h$ 把 belief 拉向新的 ground truth:
$$B_{t+\Delta t} \to P(S' \mid h_{t+\Delta t})$$

这个 formulation 直接借用了 developmental psychology 里的 false belief paradigm (Wimmer & Perner 1983, https://doi.org/10.1016/0010-0277(83)90004-5)。原版 ToM false belief test 测的是儿童能否理解 Sally 把球放篮子 A，离开后 Anne 移到篮子 B，Sally 回来还会去 A 找；这里 paper 把 "Sally 的旧 belief" 换成 "agent 的旧 cognitive map"。

**Exploit (利用)**:

把 belief 喂给 policy $\pi(a_t \mid B_t)$ 来解下游任务 $\tau$，性能用 $\mathcal{I}(\pi(\cdot \mid B_t), \mathcal{T})$ 测。

---

## 3. Benchmark 设计：Text-Vision Parallel World

这个设计是 paper 最聪明的地方之一。同一个 scene seed 在 text world 和 vision world 各跑一遍，可以直接把 perception failure 和 reasoning failure 分离开。

**环境**：$N \times M$ grid 的多房间 indoor layout，房间图是树形拓扑（connected + acyclic，无 loop，避免循环路径带来的歧义）。每个 scene 有 $n$ 个 object，每个 object 有 2D integer coordinate + cardinal orientation (N/E/S/W)。

**Action space**:
- `Goto`: 直接到 currently visible object
- `Rotate`: 90°/180°/270° in-place
- `Observe`: 90° FOV 内可见物体 (cost=1)
- `Query`: 拿一个 visible object 的绝对 2D 坐标 (cost=2，鼓励只在 ambiguity 时用)

**Spatial relation 离散化** (非常关键的设计决策):
- **Allocentric direction**: 8 个 45° bins = {N, NE, E, SE, S, SW, W, NW}
  - 例如 N bin = $[-22.5°, 22.5°)$
- **Egocentric direction**: 5 个 label 在 90° FOV 内
  - front-left $[-45°, -22.5°)$
  - front-slight-left $[-22.5°, 0)$
  - front = $0°$
  - front-slight-right $(0, 22.5°]$
  - front-right $(22.5°, 45°]$
- **Distance**: 6 bins
  - same = 0
  - near = (0, 2]
  - mid = (2, 4]
  - slightlyfar = (4, 8]
  - far = (8, 16]
  - veryfar = (16, 32]

这种 discretization 看似简单，但它把 foundation model 的 input/output 都钉死在一个可控的 symbolic 空间里。任何 ambiguity 都来自模型本身，不来自 sensor noise。

---

## 4. Information Gain 公式逐符号拆解

paper 定义了一个 normalized information gain 来量化 exploration efficiency:

$$\mathcal{E} = 1 - \frac{\sum_{i=1}^{N} \log_2 \max(1, C_i)}{N \log_2 M}$$

逐个变量讲清楚:

- $N$ = scene 中 object 总数 (paper 主实验用 12)
- $M$ = 任意 object 在 exploration 开始时的可能 position 数（uniform prior over 20×20 grid，所以 $M$ ~ 网格点数）
- $C_i$ = object $i$ 经过 AC-3 arc-consistency 约束传播之后还 consistent 的位置数
- $\log_2 \max(1, C_i)$ = object $i$ 还剩下多少 bits 的 uncertainty（用 Shannon entropy 的 discrete 版本近似）
- $\max(1, \cdot)$ 是防止 $\log_2(0)$ 的安全垫，因为 $C_i$ 至少是 1
- 分子 $\sum_{i=1}^{N} \log_2 \max(1, C_i)$ = 整个 scene 还剩多少 bits uncertainty
- 分母 $N \log_2 M$ = 初始总 uncertainty
- $\mathcal{E} \in [0, 1]$
  - $\mathcal{E} = 0$ 当 $C_i = M$ for all $i$：啥也没学到
  - $\mathcal{E} = 1$ 当 $C_i = 1$ for all $i$：所有 object 完美 localized

**AC-3 的角色**：paper 用经典的 constraint satisfaction (AC-3 arc consistency) 来维护每个 object 的 feasible domain。新观察被编译成 unary/binary constraints（egocentric direction bin, distance bin, room visibility, ALLDIFFERENT 防 collision）。AC-3 iteratively prune 掉 unsupported cell，沿 arc 传播到 fixed point。

AC-3 reference: Mackworth 1977, https://www.cs.toronto.edu/~mack/CS1506/AC-3-mackworth.pdf

---

## 5. Cognitive Map Probing 的指标

paper 不把 agent 当黑盒，而是要求 agent 在每一步 externalize 它的 belief 成 JSON 格式的 cognitive map。这个 idea 借鉴自 neuroscience 里 Tolman 1948 (https://doi.org/10.1037/h0061626) 提出、O'Keefe & Dostrovsky 1971 (https://doi.org/10.1152/jn.1971.31.4.625) 神经层面证实、Moser 夫妇 2005 (Hafting et al., https://doi.org/10.1038/nature03721) 发现 grid cells 的 cognitive map 概念。

**Positional accuracy**:
$$\text{pos.acc} = (K/N) \cdot e^{-\text{RMSE}/L}$$

- $K$ = agent 预测的 object 数
- $N$ = ground truth object 数
- $K/N$ = coverage ratio（agent 漏报多少 object）
- $\text{RMSE}$ = 预测坐标 vs ground truth 坐标的 root mean squared error
- $L$ = scene 中所有 ground truth object 位置的 RMS ℓ2-norm（用来 normalize scale）
- $e^{-\text{RMSE}/L}$ = exponential decay：RMSE 越大，这一项越小
- 整体: coverage 乘以 accuracy，两者都要好才高分

**Directional accuracy**: 任意 object pair 之间方向关系对的比率。

**Facing accuracy**: object 的 facing direction (N/E/S/W) 预测对的比率。

**Stability**: 之前观察过的 object 在后续 turn 中 belief 是否 degrade。形式化是 per-turn check：if 当前预测不比上一 turn 差就给 1。这个指标揭示了 belief drift——前面看到对的，后面被 corrupt。

**Local ↔ Global consistency**: 同一 turn 内 local snapshot 和 global map 没有矛盾。诊断新证据是否 coherent 地整合进 global belief。

**Self-tracking**: agent 能不能稳定估计自己的 pose。从预测的 global map 反推 agent 状态 vs ground truth agent state。

---

## 6. Belief Inertia 公式逐符号拆解

这个公式是 paper 最 technical 的部分，用来量化 agent 在 environment shift 之后是不是还 stuck 在旧 prior 上。

对每个 shifted object $i$：

$$s_i^{pos} = \underbrace{\frac{\mathbf{e}_i^\top \mathbf{v}_i}{\|\mathbf{e}_i\|\|\mathbf{v}_i\| + \epsilon}}_{\text{Directional alignment }(\cos\theta_i)} \cdot \underbrace{\exp\left(-\frac{\|\mathbf{b}_i^{new} - \mathbf{b}_i^{old}\|^2}{2\sigma^2}\right)}_{\text{Proximity weight } (w_i)}$$

变量定义:
- $\mathbf{b}_i^{old}$ = shift 之前 agent 对 object $i$ 的 belief 坐标
- $\mathbf{b}_i^{new}$ = revision 之后 agent 对 object $i$ 的 belief 坐标  
- $\mathbf{g}_i^{new}$ = shift 之后 object $i$ 的 ground truth 坐标
- $\mathbf{v}_i = \mathbf{b}_i^{old} - \mathbf{g}_i^{new}$ = **prior-offset 向量**：从新 ground truth 指回旧 belief
- $\mathbf{e}_i = \mathbf{b}_i^{new} - \mathbf{g}_i^{new}$ = **post-revision error 向量**：从新 ground truth 指向新 belief
- $\cos\theta_i$ = 这两个向量的余弦相似度
- $\sigma$ = dynamic noise scale = re-exploration 中首次 re-observe unchanged object 时的 RMS localization error
- $\epsilon$ = 防 zero-division 的数值稳定项

**直感解读**：
- 如果 revision 完美，$\mathbf{e}_i = 0$，分子为零 → $s_i^{pos} = 0$，没有 inertia。
- 如果 revision 后 agent 仍然把 object 放在旧位置附近（即 $\mathbf{b}_i^{new} \approx \mathbf{b}_i^{old}$），那么 $\mathbf{e}_i \approx \mathbf{v}_i$，cosine = 1，且 $\exp$ 项也接近 1（因为 $\mathbf{b}_i^{new} - \mathbf{b}_i^{old} \approx 0$），所以 $s_i^{pos} \to 1$。
- 中间情况：agent 部分修正了，但残差仍然偏向旧 prior 方向 → $s_i^{pos} > 0$。

**Proximity weight $w_i$ 的作用**：如果 agent 的 belief 已经离旧 prior 远了（说明它确实 revise 了），即使剩余 error 还和 $\mathbf{v}_i$ 同方向，也降权。这个 design 防止 "agent 已经 revise 走了，但仍然被算 inertia"。

**Orientation inertia**：
$$s_i^{ori} = \mathbb{1}(\phi_i^{new} = \phi_i^{old})$$

直接 indicator：如果 agent revision 后预测的 orientation $\phi_i^{new}$ 等于 shift 前的 $\phi_i^{old}$（说明根本没改），就算 1。简单粗暴但有效。

---

## 7. 实验结果：四大关键发现

### Finding 1: Active-Passive Gap

Table 2 (active) vs Table 3 (passive) 对比：

| Model | Text Active | Text Passive | Vision Active | Vision Passive |
|-------|------------|-------------|---------------|----------------|
| GPT-5.2 | 72.0 | 90.4 | 46.0 | 57.1 |
| Gemini-3 Pro | 81.5 | 86.5 | 57.3 | 60.5 |
| Claude-4.5 Sonnet | 65.9 | 73.6 | 29.6 | 43.1 |

**直觉**: 在被动 setting 下，agent 拿到 scripted proxy (SCOUT 或 STRATEGIST) 生成的干净 trajectory，只需要 reasoning；active setting 下要自己规划 trajectory。性能差距说明 active exploration 本身是 bottleneck——即使模型 reasoning 能力 OK，它**不知道下一步看哪里**。

最 striking 的是 text world 下 GPT-5.2 从 90.4 掉到 72.0，掉了 18 个点。注意 text world 里 perception 是被完全 discretize 好的 symbol，没有视觉 ambiguity，所以这个 gap 完全是 planning/exploration policy 的失败。

### Finding 2: Inefficiency

Table 2 的 Avg.step 列:
- GPT-5.2 vision: 17.2 步
- Gemini-3 Pro vision: 13.6 步
- Claude-4.5 Sonnet vision: 19.6 步
- SCOUT proxy: ~9 步
- Human: 9.8 步

paper 还做了一个很巧的对照：让 text world 的 agent 跟着 SCOUT 的 ~9 步 trajectory 走，GPT-5.2 拿到 83.9，Gemini-3 Pro 拿到 86.7，都**超过**他们自己 active exploration 的 72.0/81.5。说明模型不是 reasoning 不行，是探索策略不行。

Figure 4 的 information gain 曲线显示 GPT-5.2 一开始涨得快（因为 find-door 策略），后面 plateau；Gemini-3 Pro 涨得慢但持续涨，因为 rotate-and-scan 系统化。

### Finding 3: Modality Gap

Text 远远好于 vision。这看似 trivial（vision 更难），但 paper 通过 cognitive map probing 把 modality gap 分解到具体维度。

Table 5（vision world）:
| Model | Correctness (ori.) | Correctness (pos.) | Perception | Stability | Self-tracking | Uncertainty |
|-------|----|----|----|----|----|----|
| GPT-5.2 | 20.2 | 42.0 | 33.5 | 53.7 | 93.3 | 57.0 |
| Gemini-3 Pro | 32.2 | 62.5 | 52.1 | 61.8 | 98.8 | 64.5 |

**Vision 模型的 orientation perception 几乎是 chance level**（GPT-5.2 只有 20.2%）。这解释了为什么 perspective taking 任务（Table 2 vision: persp.take ~36%）这么差。Self-tracking 高（93-99%）说明 agent 知道自己在哪，但不知道 object 朝哪。

### Finding 4: Belief Drift + Belief Inertia

**Belief Drift** (Table 5)：vision 下 final map correctness 远低于 per-turn perception。例如 GPT-5.2 vision: perception = 33.5，但 final correctness (position) = 42.0（这个数居然比 perception 高？应该是 perception 比较低的时候还没积累完整 observation）。更关键的是 Stability 只有 53.7——agent 一开始看到对的，后面 turn 把它 overwrite 成错的。

**Belief Inertia** (Table 7):

| Model | World | Belief Inertia (pos.) ↓ | Belief Inertia (ori.) ↓ |
|-------|-------|---|---|
| GPT-5.2 | Text | 5.5 | 12.5 |
| Gemini-3 Pro | Text | 7.9 | 5.7 |
| GPT-5.2 | Vision | **68.9** | **34.7** |
| Gemini-3 Pro | Vision | **51.1** | 14.4 |

Vision agent 的 positional inertia 高达 68.9%（GPT-5.2）和 51.1%（Gemini-3 Pro）。这意味着即使 agent 直接看到了 object 被移动的新位置，它仍然 report 旧坐标。这是 false belief 的 spatial 版本——agent "知道" object 移了，但旧 prior 牢牢拽着它的 belief。

---

## 8. 与其他工作的对照

### 8.1 与 ToM 的精确对应

paper 的整个 framing 是和 ToM 平行：

| ToM | ToS |
|-----|-----|
| 隐藏的心理状态 | 隐藏的空间结构 |
| Sally-Anne test | False Belief task (object shift) |
| Perspective taking | Perspective taking (egocentric ↔ allocentric) |
| Belief revision under new info | Belief revision under environment shift |
| Cognitive penetration of perception | Perception-driven mapping errors |

ToM 经典文献: Baron-Cohen et al. 1985, https://doi.org/10.1016/0010-0277(85)90022-8

### 8.2 与 Active Learning 的关系

经典 active learning (Settles 2009, https://morganclaypool.com/doi/10.2200/S00252ED1V01Y200910AIM005) 是 agent 选 query 来减少 model uncertainty，通常在 classification/regression setting。ToS 是 active learning 的 embodied 版本——agent 通过物理动作（move/rotate）选 observation，目标是减少 spatial belief entropy。Information gain 公式直接对应 Bayesian active learning 的 entropy reduction。

### 8.3 与 Active Inference / FEP 的关系

Friston 的 Free Energy Principle (Friston 2010, https://doi.org/10.1038/nrn2787) 强调 organism 通过 active inference 减少 surprise / expected free energy。ToS 的 framework 在精神上很 FEP：agent 通过 exploration 减少 spatial uncertainty。但 paper 没有显式 call out FEP，可能是怕 frame 太重。

### 8.4 与经典 cognitive map 文献

- Tolman 1948 "Cognitive maps in rats and men": https://doi.org/10.1037/h0061626
- O'Keefe & Dostrovsky 1971 (place cells): https://doi.org/10.1152/jn.1971.31.4.625
- Hafting et al. 2005 (grid cells): https://doi.org/10.1038/nature03721
- Moser & Moser 2017 Nobel lecture: https://www.nobelprize.org/prizes/medicine/2014/moser/lecture/

paper 直接用 cognitive map 作为 belief 的 canonical representation，这是非常 elegant 的选择——既有 neuroscience grounding，又能 JSON serialize。

### 8.5 与 Spatial reasoning benchmark 的对照

| Benchmark | Setting | 评估对象 |
|----------|---------|---------|
| bAbI (Weston 2015) | Text | Passive reasoning |
| SpartQA (Mirzaee 2021) | Text | Spatial QA |
| StepGame (Shi 2022) | Text | Multi-hop reasoning |
| 3DSR-Bench (Ma 2024) | Single image | Spatial relation |
| SpatialRGPT (Cheng 2024) | Single image | Grounded spatial reasoning |
| SpatialVLM (Chen 2024) | Single image | VLM spatial capability |
| MMSI-Bench (Yang 2025c) | Multi-image | Multi-frame spatial intelligence |
| VSI-Bench (Yang 2025a) | Video | Spatial memory from video |
| MindCube (Yin 2025) | Multi-view | Layout prediction |
| **ToS (this paper)** | **Active embodied** | **Belief construction + revision + exploitation** |

ToS 的独特之处是 task-agnostic active exploration——没有指定 "find the red chair"，而是测 agent 能不能自发地构造完整 spatial belief。

VSI-Bench reference: https://arxiv.org/abs/2503.11044 (Yang et al. "Thinking in Space")
MindCube reference: https://arxiv.org/abs/2506.21458

### 8.6 与 Embodied QA 的对照

| Benchmark | Active? | Task-driven? |
|----------|---------|-------------|
| EQA (Das 2018) | Yes | Yes (answer specific Q) |
| IQA (Gordon 2018) | Yes | Yes |
| OpenEQA (Majumdar 2024) | Yes | Yes |
| REVERIE (Qi 2019) | Yes | Yes |
| ALFRED (Shridhar 2020b) | Yes | Yes (instruction following) |
| TEACh (Padmakumar 2022) | Yes | Yes |
| Mind Palace (Ginting 2025) | Yes | Yes (long-horizon EQA) |
| EXCALIBUR (Zhu 2023) | Yes | No (task-agnostic, but RL-trained) |
| **ToS** | **Yes** | **No (curiosity-driven)** |

EXCALIBUR reference: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_EXCALIBUR_Encouraging_and_Evaluating_Embedded_Exploration_CVPR_2023_paper

### 8.7 与 SLAM 的对照

classical SLAM (Simultaneous Localization And Mapping) 解决的是机器人 pose + map 的 joint estimation。ToS 可以看作 "SLAM for foundation model"：
- SLAM 有 perfect perception (laser/depth) 和强 prior (motion model)
- ToS 让 foundation model 用 raw image / symbol 做 SLAM-like 任务
- SLAM 的 map 是 occupancy grid；ToS 的 map 是带 semantic label 的 object map
- SLAM 解决 metric uncertainty；ToS 关心 epistemic uncertainty（哪里还没看过）

ORB-SLAM reference: https://doi.org/10.1109/TRO.2015.2473675

### 8.8 与 World Model 的对照

- Dreamer 系列 (Hafner 2019-2023): latent world model + RL
- JEPA (LeCun 2022, https://openreview.net/pdf?id=BZ5a1r-kVsf): predictive latent space
- NaViT/Genie (DeepMind 2024)

这些 world model 大多在 latent representation 里学。ToS 反过来——它要求 agent 显式 externalize 一个 symbolic/structured belief (JSON cognitive map)。这是 foundation model world modeling 的两个不同流派：implicit latent vs explicit structured。

---

## 9. 一些被 paper 忽略但我觉得值得讨论的点

### 9.1 Externalization Gap 的本质

paper 自己发现 "explicit reasoning" 比 "direct answering" 略差（alignment test），称之为 "externalization gap"。这其实是 Chain-of-Thought reasoning 的反例——经典 CoT 文献说 explicit reasoning 提升 performance，这里反而 degrade。

可能解释：
1. ToS 的 cognitive map 是 structured JSON，对 LLM 来说是 "lossy compression"，模型 internal representation 比 JSON 更丰富。
2. JSON 输出引入了 format constraint，消耗 attention budget。
3. Foundation model 的 spatial knowledge散布在 attention weights 里，强迫它 serialize 成 Cartesian (x,y) 反而损失信息。

这跟 Binz 2023 "Language models can solve computer tasks" 类似——structured planning 不一定帮 foundation model。

### 9.2 Long-horizon Memory Architecture

belief drift 这个 finding 在我看来指向一个明确的方向：foundation model 缺乏真正的 **persistent working memory architecture**。当前 MLLM 都是把 history 当 token sequence 喂进 transformer，attention 会随 token 数稀释。需要的是：
- Episodic memory (event-locked)
- Semantic memory (compressed facts)
- Working memory (current belief state)

这和 neuroscience 里的 memory taxonomy 直接对应 (Tulving 1972, https://doi.org/10.1007/978-3-642-46928-4_4)。Cognitive map 本质上应该活在 semantic memory 里，但 LLM 没有 explicit 机制把 episodic observation 压缩进 semantic state。

### 9.3 Probing 的 Validity

paper 用 Pearson correlation 验证 probed map correctness 和 downstream performance 的关联 (Table 6)，相关性 0.4-0.6，p<0.001。这是一个 conservative 但合理的 proxy。但要注意：
1. Correlation ≠ causation
2. Map quality 可能和 reasoning 能力同时被 foundation model 的某个 latent factor 决定（比如 spatial intelligence）
3. Sufficiency test 用 oracle map 时 performance 飙到 95%，说明 map format 信息充分，但不代表 agent 自己生成的 map 是它的 latent state 的 faithful readout

### 9.4 Active Exploration 的 Reinforcement Learning 视角

ToS 的 active exploration 其实可以 frame 成一个 POMDP：
- State $S$ = 真实空间结构
- Observation $o_t$ = 90° FOV snapshot
- Action $a_t$ = Goto/Rotate/Observe/Query/Terminate
- Reward $r_t$ = information gain $\Delta\mathcal{E}$
- Belief state $B_t(S) = P(S \mid h_t)$

如果用 PPO 训一个 exploration policy 会怎么样？EXCALIBUR (Zhu 2023) 试过 RL，但他们的 map 是 implicit 在 policy weight 里的。ToS 显式 probe 的 design 可以用来 evaluate RL policy 的 belief quality，这是 future work。

### 9.5 视觉 perception bottleneck

Table 5 显示 vision 模型 orientation perception 几乎 chance。这指向一个具体技术 bottleneck：foundation model 不会从单视角 image 推断 object facing。原因是 training data 里 facing label 稀缺。

可能 fix:
1. Synthetic data training (Objaverse + random facing)
2. Multi-view supervision
3. Implicit 3D representation (NeRF / 3D Gaussian Splatting)

3D Gaussian Splatting reference: https://doi.org/10.1145/3592114

---

## 10. 我的 take

这篇 paper 在我看来最重要的 contribution 不是 benchmark 本身，而是它**把 evaluation paradigm 从 "answer correctness" 转向 "belief quality"**。这是一个 paradigm shift。当前大多数 spatial benchmark (3DSR-Bench, MMSI-Bench, VSI-Bench) 都是 multiple choice 或 open-ended QA，测的是 model 的 surface 表现。ToS 直接 probe model 的 latent state，这让我们能 diagnose 失败模式。

最有 diagnostic value 的两个指标:
1. **Stability**：暴露了 belief drift，揭示了 foundation model 在 long horizon 下的 memory corruption 问题
2. **Belief Inertia**：揭示了一个非常 specific 的 failure mode——模型即使看到新证据也无法 overwrite 旧 prior

这两个 finding 在我看来都和 RLHF 有关。RLHF 训出来的 model 倾向 "stick to a confident answer"，这恰好在需要 belief revision 时是 anti-pattern。可能需要新的 training 方法比如 belief tracking objective 或 active inference style 的 exploration reward。

paper 的 limitation:
1. 只测 indoor multi-room，没测 outdoor large-scale
2. Action space 太离散 (Goto to visible object)——真实 robot 不可能直接 teleport
3. 没测 multi-agent，paper 自己提到这是 future work
4. Probing 是 lossy compression（paper 自己承认）
5. 没测 spatial scale generalization（房间数 2/3/4 已经测过，但没测房间大小变化）

相关值得 follow 的工作:
- EmbodiedGPT (https://arxiv.org/abs/2305.01832)
- PIVOT (https://arxiv.org/abs/2402.07772)
- SpatialBot (https://arxiv.org/abs/2406.23436)
- SpatialRGPT-Bench (https://arxiv.org/abs/2406.23436)

---

## 11. 总结

Theory of Space 是一个把 ToM、cognitive map neuroscience、active learning 三个传统结合起来的 benchmark。它最强的 finding 是:

1. **Active-Passive Gap**: foundation model 在被动 reasoning 时还行，主动 exploration 就崩。
2. **Belief Drift**: 即使 perception OK，spatial belief 在 long horizon 下会 degrade。
3. **Belief Inertia**: 即使看到新证据，旧 prior 拽着不放，vision 模型尤其严重。
4. **Orientation Gap**: vision 模型几乎不会从 image 推断 object facing。

这些 finding 加起来指向一个清晰的 research agenda：foundation model 需要 persistent structured memory architecture、需要 belief tracking training objective、需要更好的 3D perception pretraining。这个 paper 的价值在于它把这些 bottleneck 用一个 unified benchmark 显式化了。
