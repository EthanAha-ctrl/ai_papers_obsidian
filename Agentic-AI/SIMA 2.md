---
source_pdf: SIMA 2.pdf
paper_sha256: a12ecfd60003a02d1039d2f137cb43b5d829a491c8c451fd2651b3fad3226cd2
processed_at: '2026-08-12T06:22:32-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，既然你要 "用人话说说"，但同时又要求技术细节、公式和直觉，那我就用 Karpathy 式的 "storytelling + math" 来拆解。我们抛开学术黑话，直接看这个系统到底在干嘛，为什么这么干，以及它通向哪里。

这里的核心其实是四个直觉：**把操作当文字写下来**、**给录像配旁白**、**AI 当自己的教练**、**大脑指挥小脑**。

---

### 1. 核心直觉一：把键盘鼠标操作 "写" 出来

要训一个能打游戏的 Gemini，最大的障碍是什么？大模型本来是个话痨，只会吐文字。打游戏需要精确的 200ms 内按下键盘 `W` 并滑动鼠标。

以前的做法（比如 RT-2 https://robotics-transformer2.github.io/）是在 tokenizer 里硬塞 256 个特殊 token 代表动作。SIMA 2 彻底放弃这个思路。它直接让模型生成一段结构化文本，比如 `<action> press W, move mouse (dx=12, dy=-5) </action>`，然后外层写个死板的解析器，把这段文本翻译成真实的键盘鼠标事件。

这招极其巧妙。因为 action 变成了 text，模型输出 action 就像写一句代码一样自然。所有的 reasoning、dialogue、action 全部在同一个 token stream 里流淌。

形式化看这个条件概率分布：

$$
p_\theta(y_t \mid o_{\leq t}, x_{\leq t}, y_{<t})
$$

变量解释：
- $\theta$: Gemini Flash-Lite 模型的参数。
- $y_t$: 当前时间步 $t$ 模型输出的 text token（可能包含 reasoning 文字，也可能包含 action 指令）。
- $o_{\leq t}$: 观察到的历史图像帧，下标 $\leq t$ 表示从开始到当前时间步的所有视觉输入。
- $x_{\leq t}$: 人类用户的历史指令输入。
- $y_{<t}$: 模型自己之前生成过的所有 token（包括它自己之前的内心独白和已执行的操作）。

直觉：模型像写日记一样，边写想法，边写下动作指令。外部的游戏环境读取动作指令那一行去执行，然后模型在下一轮继续写。没有独立的 policy head，没有特殊的 action token，一切皆是 language generation。

---

### 2. 核心直觉二：给人类游戏录像 "配旁白"

数据是 VLA 模型的命门。DeepMind 雇人玩了一堆游戏（No Man's Sky, Valheim 等），收集了海量的 `(画面, 键盘操作)` 数据。

问题来了：人类打游戏时是不说话的，也不做逻辑推理。模型如果只学 `(画面, 键盘操作)`，它只会变成一个无脑的模仿机器，完全不会思考 "我要先砍树因为我要做木板"。

为了让模型学会边想边打，他们发明了 **Bridge Data**。做法非常暴力且优雅：把高质量的游戏录像片段拿出来，喂给聪明的 Gemini Pro，让它给录像写 "旁白" 和 "内心独白"。

比如 Gemini Pro 看到一段录像，会生成这样的标注：
> *Internal Reasoning: I see a tree. The user asked for wood. I need to equip the axe first.*
> *Action: Press 'E' to open inventory.*
> *Dialogue: I'm going to chop down that tree now.*

这就构成了包含多模态交织的高质量训练数据。小模型 SIMA 2 (Flash-Lite) 在这上面做 SFT，本质上是一种跨模态的行为蒸馏。它学到的仅仅是如何按键，更学到了 "在什么情境下应该产生什么推理，并导向什么动作"。

这里的 loss function 就是标准的自回归交叉熵：

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(o, x, y) \sim \mathcal{D}} \left[ \sum_{i=1}^{|y|} \log p_\theta(y_i \mid o, x, y_{<i}) \right]
$$

变量解释：
- $\mathcal{L}_{SFT}(\theta)$: 模型参数 $\theta$ 在 Supervised Fine-Tuning 阶段的损失函数。
- $\mathbb{E}$: 期望符号，表示在数据集 $\mathcal{D}$ 上求平均。
- $o, x, y$: 从混合数据集 $\mathcal{D}$ 采样的一个 episode，其中 $o$ 是图像观察，$x$ 是指令，$y$ 是目标输出序列（旁白+动作）。
- $i$: 输出序列 $y$ 中第 $i$ 个 token 的位置。
- $|y|$: 序列 $y$ 的总长度。

为了防止把基础模型学傻了（Catastrophic forgetting），训练数据里必须要混入大量原始的 Gemini 预训练数据。

**Mixture 权重公式（直觉构建）：**

$$
\mathcal{D}_{\text{train}} = \alpha \mathcal{D}_{\text{human}} + \beta \mathcal{D}_{\text{bridge}} + \gamma \mathcal{D}_{\text{Gemini-pretrain}}
$$

变量解释：
- $\mathcal{D}_{\text{human}}$: 纯人类操作数据，提供 motor control 信号。权重 $\alpha$ 最大。
- $\mathcal{D}_{\text{bridge}}$: Gemini Pro 生成的推理+操作交织数据，提供 reasoning 信号。权重 $\beta$ 较小但质量极高。
- $\mathcal{D}_{\text{Gemini-pretrain}}$: 原始文本/图片数据，防止模型忘记怎么说话和推理。权重 $\gamma$ 用于维持 Pareto frontier。

论文 Table 1 的实验数据证实了这个 mixture 策略的威力：

| Benchmark (Baseline Gemini) | SFT Only | SFT + RL | Intuition |
| :--- | :--- | :--- | :--- |
| LCB (Code) | -4.0% | -8.4% | 代码能力基本保住了 |
| AIME (Math) | -25.5% | -15.4% | RL 阶段数学能力居然反弹了 |
| GPQA Diamond (STEM) | -16.3% | -19.5% | 科学推理有微小下降 |

如果没有 $\gamma$ 这一项混合数据，这表格里的数字大概会掉到 -50% 甚至更多。VLA 模型最大的坑就是训完动作后模型变成哑巴，混合数据是把模型按在 Pareto frontier 上不掉下去的关键。

参考链接：Hancock et al. 2025 关于 VLA 灾难性遗忘的研究 https://arxiv.org/abs/2509.22195

---

### 3. 核心直觉三：AI 当自己的教练和裁判

最让我兴奋的是 Section 4.5 的 Self-Improvement 机制。把 SIMA 2 扔进一个完全没见过的新游戏（比如 ASKA），它一开始玩得很烂。怎么办？

传统做法：再雇人打游戏补数据。SIMA 2 做法：启动三个 Gemini 实例，玩一个无限循环的 "自我进化" 游戏。

**架构图解析：**

1. **Task Setter (出题官)**: 看着当前游戏画面，生成一个在这个场景下可以完成的指令，比如 "去熄灭那个篝火"。
2. **SIMA 2 Agent (考生)**: 尝试执行这个指令，生成一段轨迹 $\xi$。
3. **Reward Model (裁判)**: 看着考生的录像，根据 rubric 打分，范围是 $r \in [0, 100]$。50分及格。

**强化学习更新公式：**

$$
\pi_{t+1} = \text{Train}\left(\pi_t, \{(\tau_i, \xi_i, r_i)\}_{i=1}^{N}\right)
$$

变量解释：
- $\pi_t$: 第 $t$ 次迭代时的 Agent 策略。
- $\tau_i$: Task Setter 提出的第 $i$ 个任务指令。
- $\xi_i$: Agent 执行任务产生的轨迹 (状态-动作序列)。
- $r_i$: Reward Model 给该轨迹打的分数。

这就是一个纯靠 Foundation Model 驱动的闭环 RL。不需要人写 reward function，不需要人标 preference。Gemini 的世界知识足够它判断 "去熄灭篝火" 这个任务到底做没做完。

论文里 ASKA 的实验结果非常震撼：经过几轮自我迭代，Agent 的平均得分超过了有数小时游戏经验的人类参考录像。并且它真的学会了从零开始识别新物体（雨水收集器）并执行新动作（扑灭篝火）。

更夸张的是在 Genie 3 (https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 生成的逼真环境里做自我进化。在 "城市" 环境里刷题，居然 transfer 到了 "大自然" 环境里也能用。这隐隐约约摸到了 Clune 2019 提出的 AI-GAs (https://arxiv.org/abs/1905.10985) 的门槛：一个算法在无限生成的世界里永远学习下去。

---

### 4. 核心直觉四：大脑指挥小脑

SIMA 2 用的是 Gemini Flash-Lite，为了低延迟，牺牲了深度推理能力。论文 Section 4.4 提供了一个极其优雅的系统级解决方案：Hierarchical Composition。

让反应慢但极聪明的 Gemini Pro 当 "大脑"，让反应快但稍笨的 SIMA 2 当 "小脑"。

**层级控制循环：**

$$
(i_t, s_t) = \text{GeminiPro}(o_{t-k:t}, s_{t-k}, i_{t-k:t-1}) \quad \text{every } k \text{ steps}
$$

$$
a_t = \pi_{\text{SIMA2}}(a_t \mid o_t, i_t, h_t) \quad \text{every step}
$$

变量解释：
- $k$: 大脑思考的间隔步数（比如每 10 帧思考一次）。
- $s_t$: Gemini Pro 在时间 $t$ 写下的 text summary，充当长期记忆，解决 context window 有限的问题。
- $i_t$: Gemini Pro 下达给 SIMA 2 的自然语言微观指令。
- $o_{t-k:t}$: 过去 $k$ 帧的视觉观察。
- $a_t$: SIMA 2 每一帧输出的低级键盘鼠标操作。
- $h_t$: SIMA 2 自己的短期上下文。

直觉：你给 Gemini Pro 一张复杂的搭篝火图纸。Gemini Pro 每过几秒看一眼画面，在脑子里记下 "已经砍完树了，下一步该找石头了"，然后对 SIMA 2 说 "去捡两块石头"。SIMA 2 操纵鼠标键盘去捡石头。捡完汇报 "捡完了"，Gemini Pro 再下达下一步指令。

这种设计直接把 reasoning 和 acting 解耦了。底层 SIMA 2 负责通用的 "视觉-运动" 映射能力，上层接不同的 Gemini 版本就能获得不同的智商。未来 Gemini 3 出来了，直接换上层，SIMA 2 的底子还能继续用。

---

### 5. 跨维度的泛化：游戏到现实

把 SIMA 2 扔进 Genie 3 生成的 photorealistic 环境里，它居然能完成导航任务。这个结果对我们做机器人或具身智能的人来说意义极其重大。

它证明了：在游戏中通过键盘鼠标学到的 "3D 导航、空间推理" 能力，底层是 modality-agnostic 的。游戏画面的风格只是一种 surface variation。模型学到的不是 "在 No Man's Sky 里怎么走"，它学到的是 "看到一个 3D 场景，提取可行走路径，并通过键盘鼠标接口输出移动向量" 这个抽象映射。

这就给物理世界的 robotics 指了一条明路。键盘鼠标本质上是一种 universal API。如果 Genie 3 能生成足够逼真的物理世界，且机器人控制器接受键盘鼠标输入，SIMA 2 这套技术栈几乎可以原封不动地迁移过去。

参考链接：Gemini Robotics 1.5 (https://arxiv.org/abs/2510.03342) 正在探索把这套 VLA 架构搬到真实机器人上。

---

### 总结：SIMA 2 的技术哲学

如果要我用一句话 build your intuition: **SIMA 2 把 embodied agent 从一个 specialized policy 训练问题，转化成了 generalist foundation model 的 capability extension 问题。**

一切皆是 language。Reasoning 是 language，Action 是 structured language。只要你的 action space 能被 text 描述，你的 reward 能被 language model 评估，你的 task 能被 language model 生成，你就可以构建一个没有任何人工干预的 self-improving loop。

这篇论文不仅仅是把 Gemini 塞进游戏里打游戏那么简单，它是对未来 AGI 体系架构的一次预演：用世界模型生成环境，用大模型做出题和裁判，用 VLA 做执行体，在无限的经验流中持续进化。这正是 Silver 和 Sutton 在 "Era of Experience" (https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) 里描绘的图景的技术实现雏形。

---

# SIMA 2 深度技术解读

 Andrej，这篇 SIMA 2 论文信息密度很高，我会从架构直觉、数据流、训练动力学、self-improvement 闭环这几个角度拆解，尽量让你能 build intuition about why each design choice matters。

---

## 1. 论文核心定位：从 instruction-follower 到 interactive partner

SIMA 1 (DeepMind, 2024, https://arxiv.org/abs/2404.10179) 本质上是一个 **multimodal policy** 把 映射到 keyboard/mouse action，language encoder 是 from-scratch 训练的，所以 vocabulary 被 annotation 数据集锁死。它解决的是 "can a single agent follow short instructions across many games" 这个问题。

SIMA 2 把问题 lift 到一个新 dimension：**the agent itself is a Gemini**。这就把 embodied action 从一个独立的 policy learning 问题，变成 foundation model 的 capability extension 问题。核心 shift 在于：vision, language, reasoning, action 全部 flow through 同一个 token stream，这意味着 action 不再是一个 isolated output head，而是 model 的 language behavior 的一个 subtype。

这个 shift 直接解锁了 SIMA 1 不可能有的能力：
- **Embodied dialogue**：用户问 "go check those egg-shaped objects and tell me what material they are made of"，agent 真的会 navigate 过去、用 on-screen OCR 信息回复 "they appear to be plants containing Carbon"。这是 information-seeking behavior，需要 action 和 language 在同一 inference pass 内交替产生。
- **Internal reasoning**：用户说 "go to the house colored like a ripe tomato"，agent 内部生成 "ripe tomato → red house → target is the red house on the right"，然后 act。Reasoning 是 causal 的中间变量，不是 post-hoc rationalization。
- **Multi-modal prompting**：可以用 sketch（手画的 tree）来指示 "find object of this kind and interact appropriately"。

这种设计的关键直觉是：**action 是一种特殊的 language generation**，而不是 model 的一个 isolated head。这个想法在 RT-2 (https://robotics-transformer2.github.io/) 里已经出现过，但 RT-2 把 action 离散化成 256 个 token 拼到 vocabulary 里；SIMA 2 走得更彻底——直接用 structured text 描述 keyboard/mouse 事件，让 parser 去转。

---

## 2. Agent-Environment Interface：structured text 作为 action medium

这是我最感兴趣的设计点。看 Figure 3 的 interface：

**Input stream**:
- 720p RGB frames（periodic sampling，没明确给 Hz，估计 10–15 Hz）
- 历史 context：previous natural language inputs + agent 自己产生的 internal reasoning 和 responses

**Output**: structured text，deterministically parse 为：
- 96 个 keyboard keys 的事件
- mouse clicks
- discretized relative mouse movement 

关键句："Instead of predicting discrete action tokens from a predefined set, the agent is trained via SFT to generate a structured text output."

这个选择有几个 deep 的 implications：

1. **No new vocabulary**：不需要像 RT-2 那样把 action 嵌入 tokenizer 的 token space，避免 tokenizer retraining 和 action token 占用 vocabulary budget。
2. **Output modalities composable**：同一个 text stream 可以同时携带 action、reasoning、dialogue，agent 自己决定什么时候产生哪种。这就是为什么 bridge data 里包含 "no-ops"——agent 需要学到 "task 完成后停止 act" 这个状态。
3. **Parser is deterministic**：低级 control 仍然 reliable，因为 text → keyboard/mouse 的映射没有 ambiguity。
4. **Inherits Gemini's instruction-following**：因为 output format 本身是 text，所有 Gemini 学到的 format adherence 能力直接 transfer。

这种设计的 cost：output sequence 变长（一个 mouse move 要写 "move mouse (dx=12, dy=-5)" 这种），inference latency 上升。论文选 Gemini Flash-Lite 而不是 Pro，就是为了这个 latency constraint。这也解释了为什么他们后续做 hierarchical composition with Gemini Pro——把慢的 deep reasoning 和快的 low-level control 解耦。

形式化一下 agent 的 output 分布：

$$
p_\theta(y_t \mid o_{\leq t}, x_{\leq t}, y_{<t})
$$

其中 $o_{\leq t}$ 是 frame history，$x_{\leq t}$ 是 user instruction history，$y_{<t}$ 是 agent 自己的历史输出（含 reasoning, dialogue, action）。每个 $y_t$ 是 text token sequence，parse 后得到 $(a^{\text{kb}}_t, a^{\text{mouse}}_t, \text{text}_t)$，其中 action 部分被 emit 到 environment，text 部分被记入 history 供下一步 conditioning。

---

## 3. Data Pipeline：三种数据混合

这是 SIMA 2 训练的真正 secret sauce。论文 Section 3.3 描述了一个三组分 mixture：

### 3.1 Human Data（volume 主要来源）

四种采集模式：

**(a) Single-person post-hoc annotation**: 玩家自由玩，事后给 frame-aligned 语言标注。优点：naturalistic behavior diversity；缺点：language 和 action 没有 causal link（标注是 hindsight 的）。

**(b) Setter-Solver 双人 annotation**: Setter 看 gameplay 实时下指令，Solver 执行。Language 总是 precede action，建立 causal link。这是更 "correct" 的 imitation source。

**(c) Game-Tasks episodic**: 预设 + 起始 state + 时间 limit，玩家 declare success 后结束。用于 specific skill acquisition 和 evaluation。

**(d) Human ratings**: binary success + side-by-side comparisons。用于 calibrate reward model。

**关键预处理**：trajectories 被切成 "spans"——每个 span 是一个 task instruction 关联一段 frames+actions。这个 span 结构对训练很重要，因为 agent 学的是 conditional policy $p_\theta(\text{action} \mid \text{instruction}, \text{context})$，span 把 instruction 和 action segment 对齐。

### 3.2 Bridge Data（关键的 multimodal glue）

这是 paper 里最 subtle 的设计。Human gameplay 不包含 reasoning 和 dialogue，所以直接用 human data 训练，agent 学不到 SIMA 2 需要的 "reason + act + converse" 交织行为。

Bridge data 的构造：
1. 从 human data 里选少量高质量 trajectory
2. 用 **Gemini Pro** 给每条 trajectory 标注 causally consistent 的 internal reasoning 和 dialogue
3. 标注必须从 agent 的 ego-centric perspective 出发，和 observable scene + embodied behavior 一致
4. Prompt 内做 variation，诱导 robustness

最终 bridge examples 包含的 capability 类型：
- Error correction（"我走错方向了，转回去"）
- Explicit instruction following
- Instruction chaining（多步序列）
- Visual question answering
- Memory reliance
- Long-horizon behavior
- **No-ops**（任务完成后保持静止）——这个特别重要，因为它教 agent 识别 task completion 的 "stop signal"

Bridge data 用 Gemini Pro 而不是 Flash-Lite 生成，因为标注质量直接决定 agent 的 reasoning 上限。这是 "teacher → student" 的 distillation pattern：strong model 生成 reasoning trace，weak model 学习生成同类 trace。

### 3.3 Non-gameplay Gemini Pretraining Data

论文明确说："mixture crucial to maintain the original capabilities of the base model, such as vision understanding, dialogue, reasoning, and promptability."

这是 catastrophic forgetting 的 mitigation。VLA 文献（Hancock et al. 2025, https://arxiv.org/abs/2509.22195; Zhou et al. 2025 ChatVLA）都观察到：纯 action data finetune 会 "erode conversational ability entirely"。SIMA 2 通过混合 pretraining distribution 数据避免这个 trap。

直觉上，训练 distribution 应该是：

$$
\mathcal{D}_{\text{train}} = \alpha \mathcal{D}_{\text{human}} + \beta \mathcal{D}_{\text{bridge}} + \gamma \mathcal{D}_{\text{Gemini-pretrain}}
$$

其中 $\alpha, \beta, \gamma$ 是 mixture weights，论文没给具体数字，但提到 human data 是 volume 主要来源，bridge data 是 "relatively small number of high-quality examples"。$\gamma$ 应该够大以 maintain reasoning，但够小不淹没 embodied signal。

---

## 4. 训练流程：SFT → RL with verifiable rewards

### Stage 1: SFT

起点：pretrained Gemini Flash-Lite checkpoint
目标：在 mixed dataset 上做 next-token prediction，但 target 序列是 structured text（含 action、reasoning、dialogue）。

Loss 形式上是 standard LM loss：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(o, x, y) \sim \mathcal{D}} \left[ \sum_t \log p_\theta(y_t \mid o, x, y_{<t}) \right]
$$

但因为 $y$ 含 action tokens，gradient 会通过 action prediction 流回 vision encoder 和 LM trunk，让 model 学会 vision → action mapping。

### Stage 2: RL with verifiable rewards

Verifiable task 定义为 tuple $(s_0, \tau, V)$：initial state, text instruction, verification function。

Reward 来源：
- Embodied task 完成：$r = V(\xi)$ where $\xi$ 是 trajectory
- Dialogue task 正确：grounded QA 的 correctness
- Shaped rewards 用于 instruction-following 和 controllability

Task 来源：
- Human contractor 在 random game state 提多个 achievable tasks
- 用 verifier function 扫所有 human trajectory 找 goal completion points，配 nearby state
- Filter 掉 human 也完不成的 hard tasks
- 对话 task：从 human data 随机 screenshot + human-suggested QA pair

**重要 scope 限制**：RL 阶段只在 training environments，ASKA 和 MineDojo 完全 held out。这是为了保持 generalization 评估的 cleanliness。

RL 算法论文没明说，但 "RL from verifiable rewards" 这个 phrase 和 Mankowitz et al. 2023 (https://www.nature.com/articles/s41586-023-06004-9), Wen et al. 2025 (https://arxiv.org/abs/2506.14245) 的引用暗示类似 GRPO / RLHF 的 verifiable-reward 变体。形式上：

$$
\nabla_\theta J = \mathbb{E}_{\xi \sim \pi_\theta, \tau \sim \mathcal{T}} \left[ R(\xi, \tau) \nabla_\theta \log \pi_\theta(\xi \mid \tau, s_0) \right]
$$

但具体 advantage estimation、KL regularization、PPO clip 等细节都没披露。

---

## 5. Pareto Frontier：embodied competence vs general intelligence

这是 Section 4.3 最 interesting 的分析。论文定义了两个 competing objective：
- Embodied competence（gameplay task performance）
- General reasoning（math, code, STEM）

数据点（Table 1）：

| Benchmark | SFT | SFT + RL |
|-----------|-----|----------|
| LCB (Code) | -4.0% | -8.4% |
| AIME (Math) | -25.5% | -15.4% |
| GPQA Diamond (STEM) | -16.3% | -19.5% |

关键 observations：

1. **Baseline Gemini Flash-Lite 在 embodied task 上只有 3.2% success，Pro 只有 7.0%**。这证明 embodied competence **不是** emergent property of language+vision pretraining。这呼应 Majumdar et al. 2024 OpenEQA (https://arxiv.org/abs/2403.20531) 和 Yang et al. 2025 EmbodiedBench (https://arxiv.org/abs/2502.09560) 的发现：foundation model 在 embodied reasoning 上有 fundamental deficit。

2. **SFT + RL 在 AIME 上比纯 SFT 好 10 个百分点（-25.5% → -15.4%）**。这是 counterintuitive 的——RL 通常被认为会 further 引发 forgetting。可能的 explanation：
   - RL 阶段的 verifiable reward 信号 reinforce 了 step-by-step reasoning
   - RL 数据 mixture 含 non-gameplay component
   - RL 让 model 从 "passive pattern matcher" 变成 "active reasoner"，对 math 有 spillover

3. **Code 和 STEM 在 RL 阶段 further degrade**（-4.0 → -8.4, -16.3 → -19.5）。Code 可能因为 RL 数据里没有 code generation 信号；STEM 可能因为 verifiable reward 的 task 太 game-specific。

4. **Overall degradation 在所有 benchmark 上 < 20%**。比起文献里 VLA finetune 完全 destroy conversational ability 的 case，SIMA 2 的 mixture strategy 算非常成功。

Pareto intuition：mixture training让 model 沿 frontier 移动而不是掉下 frontier。Human baseline（同一套 evaluation suite，无 timeout）几乎 100%，SIMA 2 在 training environment average 已经 close the gap（Figure 6 显示 SIMA 2 ≈ 2x SIMA 1，接近 human with timeout）。

---

## 6. Generalization：Held-out environment 的 quantitative evidence

### 6.1 Training environment 内的 held-out states

Figure 7 按 8 个 skill category 拆分（Table 2）：
- Interaction
- Navigation
- Menu Use
- Tool Use
- Construction
- Object Management
- Resource Gathering
- Combat

SIMA 2 在 Interaction 和 Object Management 上接近 human；Combat 上明显落后，因为 Combat 需要精细 motor control + split-second decision（例如 Valheim hunting deer 需要 downwind crouch + quick attack + 追击）。

### 6.2 完全 held-out environments

两个 quantitative test：

**ASKA** (Viking survival game, 2024 Early Access):
- SIMA 2 比 SIMA 1 高 >10%
- SIMA 1 只能做最 basic task（开 map、捡身边 object）
- SIMA 2 能 generalize 到 non-trivial menu use 和 multi-step instruction
- Example: "Activate the Eye of Odin → Select the right villager → Click confirm" 这种 sequential menu task SIMA 2 能完成

**MineDojo** (https://arxiv.org/abs/2206.01394, Fan et al.):
- 50 个 programmatic task，15 个 random seeds each
- SIMA 1 只完成 2 类（harvest dirt, combat spider）
- SIMA 2 完成 26/50 类
- 论文归因于 Gemini 的 Minecraft world knowledge（互联网上 Minecraft content 极多）

**Human baseline**（naive player 第一次玩）：
- MineDojo: ~19% (16 tasks)
- ASKA: ~32% (25 tasks)

这数据有 methodological 注意点：human "naive" 状态极难 maintain，因为 2-3 个 task 后 human 就 learn 了 game mechanics。所以他们用 first-attempt 数据。这暗示 SIMA 2 的 zero-shot generalization 已经接近 "human 首次接触新游戏" 的水平。

### 6.3 极端 generalization：The Gunk 和 Genie 3

**The Gunk** (story-driven platformer):
- 视觉 dark、mechanic 独特（suction tool）、非 open-world
- 通过手动指令 SIMA 2 推进前 15-20 分钟到 "Campsite" checkpoint
- 学会 scan、climb、jump、absorb gunk（用 on-screen cue "ABSORB" + "HOLD" 推理 left mouse button hold）

**Genie 3** (https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/, Ball et al. 2025):
- Photorealistic environments，text 或 initial frame conditioning
- SIMA 2 在 urban 和 naturalistic settings 都能 navigate to points of interest
- 这是 paper 里最 ambitious 的 qualitative result：**trained only on games, generalizes to photorealistic worlds**

直觉上，这暗示 embodied capability 的 "skeleton"（视觉导航、keyboard/mouse motor mapping）是 modality-agnostic 的，game 的 visual style 只是 surface variation。

---

## 7. Self-Improvement Loop：最 ambitious 的部分

Section 4.5 是 paper 的灵魂。整个 setup 是 Clune 2019 AI-GAs (https://arxiv.org/abs/1905.10985) 的 first concrete instantiation。

### 7.1 三 foundation model 协同

```
┌─────────────────┐
│  Gemini Task    │ ──── generates ────> task τ
│  Setter         │
└─────────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │  SIMA 2 Agent  │ ─── executes ───> trajectory ξ
                  │  π_θ           │
                  └────────────────┘
                                              │
                                              ▼
┌─────────────────┐
│  Gemini Reward  │ <── scores ─── r ∈ [0, 100]
│  Model          │
└─────────────────┘
                          │
                          ▼
                  Update π_θ with (τ, ξ, r)
```

形式化 self-improvement iteration：

$$
\pi_{t+1} = \text{Train}(\pi_t, \mathcal{D}_t)
$$

$$
\mathcal{D}_t = \{(\tau_i, \xi_i, r_i)\}_{i=1}^{N_t}
$$

$$
\tau_i \sim p_{\text{setter}}(\tau \mid s_i, \pi_t)
$$

$$
\xi_i \sim \pi_t(\cdot \mid \tau_i, s_i)
$$

$$
r_i = R_{\text{Gemini}}(\xi_i, \tau_i) \in [0, 100]
$$

Success threshold: $r_i \geq 50$ 视为 success。

### 7.2 Task Setter 设计

Task setter 接收：
- 当前 environment state
- Agent 当前 policy state
- (optional) downstream evaluation feedback → 可以 steer task distribution

Task setter 可以做 **curriculum learning**：根据 agent 当前 capability 提 just-out-of-reach tasks，induce learning progress（这个 idea 来自 Colas et al. 2022, https://arxiv.org/abs/2012.09830; Zhang et al. 2023 OMNI, https://arxiv.org/abs/2306.01711）。

### 7.3 Reward Model 设计

Gemini 评分用 rubric：
- Task completion
- Directedness（不做 unnecessary action）

Rubric 通过 **human preference pairs 校准**——和 RLHF 的 reward modeling 类似，但 reward signal 直接来自 Gemini 而不是 trained reward model。

这相当于把 Gemini 当 **universal reward function**（Faldor et al. 2025 OMNI-EPIC, https://arxiv.org/abs/2405.15768 的 idea）：任何 task 都能被 score，不需要 hand-craft reward。

### 7.4 ASKA 上的 self-improvement 实验

**Fixed task set 实验**（Figure 15）：
- Task 类型：Resource Gathering, Environment Interaction, Navigation, Menu Use
- 例子："Gather the berries", "Extinguish the campfire", "Go near the raw food silo", "Open the workshop hut menu"
- Human reference trajectories 来自 multiple-hour ASKA 经验玩家
- **迭代后 average score 超过 human reference**
- Initial SIMA 2 在 <25% task 上 success（score > 50），最终 agent 在所有 task 上 success
- Behavioral emergence：学会 navigate to novel object (rain collector)、acquire new skill (extinguish campfire)

**Game progression 实验**（Figure 16b）：
- Full self-improvement setup with task setter
- Task setter focus on game progression-relevant skills: resource gathering, crafting, menu use, building
- Downstream eval feedback 让 task setter focus on weak skill (e.g., ASKA 独特的 crafting menu)
- 最终 agent 能在 1 hour 内 progress 到 build shelter——比 SIMA 2 显著远

### 7.5 Genie 3 上的 cross-environment transfer

这个实验是 paper 最 forward-looking 的：

- Train environments: Genie 3 urban (e.g., "find a lollipop in a candy store")
- Test environments: Genie 3 natural (e.g., "navigate to a red mushroom")
- Self-improve on train → 几乎所有 train task 提升 25+ 分
- **Test task 也 improve**——在完全不同 visual style 的 environment 上提升

这是 open-ended learning 的 holy grail 的早期证据：**在 environment A 上 self-improve，capability transfer 到 environment B**。如果这个 scaling 继续下去，理论上可以构建一个永远 learn 的 agent，在无限 procedurally-generated environments 上 acquire diverse skills，越来越 general。

这正好 instantiate Clune 2019 的三大 AI-GA pillar：
1. Meta-learning architecture → SIMA 2 本身（VLA architecture）
2. Meta-learning objective → Gemini 作为 task setter + reward model
3. Effective environment generation → Genie 3 作为 universal world model

---

## 8. Hierarchical Composition with Gemini Pro

Section 4.4 描述的 setup：

```
   User instruction
        │
        ▼
   ┌─────────────────┐
   │  Gemini Pro     │ ─── every k steps ───> natural language
   │  (slow, deep)   │                         instruction i_t
   │  maintains      │                         + summary s_t
   │  text memory    │
   └─────────────────┘
                          │
                          ▼
                  ┌──────────────────┐
                  │  SIMA 2          │ ──── fast loop ───> low-level actions
                  │  (Flash-Lite)    │
                  └──────────────────┘
```

形式化：

$$
i_t, s_t = \text{GeminiPro}(o_{t-k:t}, s_{t-k}, i_{t-k:t-1}) \quad \text{every } k \text{ steps}
$$

$$
a_t = \pi_{\text{SIMA2}}(a_t \mid o_t, i_t, h_t) \quad \text{every step}
$$

其中 $s_t$ 是 Gemini Pro 自己生成的 text summary，下一轮调用作为 input，相当于 recurrent memory。这让系统能 maintain long-horizon context 超过 SIMA 2 自己的 context window。

**Figure 14 example**：用 campfire building diagram 指导 multi-step task（chop wood → gather stone → craft campfire）。Gemini Pro 把 diagram 拆成 step sequence，每步发 instruction 给 SIMA 2，SIMA 2 执行并回报 progress，Gemini Pro 更新 summary 和下一步 instruction。

**Appendix B 的更高级例子**：
- "Do the opposite of what I tell you"：agent 必须保持初始 instruction 的 memory，对每个新 instruction 推理 "opposite"。这 test abstract reasoning + memory。
- 21 Questions game：agent 反转 role，自己 drive exploration 和提问，user 答 yes/no。Test 主动 exploration + 推理。

这个 hierarchical setup 的 deep implication：SIMA 2 是个 **embodied substrate**，可以被任意更强的 Gemini version 指挥。当 Gemini 3 / 4 / 5 出来，直接 swap 上层就获得更强 reasoning，不需要重训底层 VLA。这是 modularity 设计哲学。

---

## 9. 关键 intuition 总结

让我给你 build 几个 mental model：

### Intuition 1: Action as language generation

SIMA 2 的核心 insight 是 "action 是 text 的一种 structured form"。一旦你接受这个 framing：
- 不需要 separate policy head
- 不需要 new tokenizer
- 不需要 new training algorithm——standard LM SFT + RL 都 work
- Multimodal interleaving（reasoning + action + dialogue）natural 出现

代价是 inference latency：text 比 discrete action token 长。但 Flash-Lite + structured format 的 speed 够 real-time。

### Intuition 2: Bridge data 是 "reasoning distillation"

Human gameplay 只教 motor control，不教 reasoning。Bridge data 用 Gemini Pro 生成 reasoning trace，让 Flash-Lite 学到 reasoning pattern。这是 **strong teacher 到 fast student 的 capability transfer**，发生在 input-output level（不是 weight level）。

类比：你给我看大量 "Go to the red house" 的 → keyboard action 数据，我学不到 reasoning。你给我看 Gemini Pro 写的 "based on 'ripe tomato', I identify the red house as target" 这种 reasoning + action pair，我学到 "color metaphor → object identification" 这个 reasoning pattern。

### Intuition 3: Self-improvement loop 是 bootstrapped RLHF

经典 RLHF 需要 human preference data。SIMA 2 用 Gemini 作为 universal reward function 替代 human preference。这 scaling 极好——Gemini 评分任何 trajectory 都行，不需要 task-specific reward engineering。

Task setter 也用 Gemini，意味着 task distribution 也是 model-driven 的，不需要 human curriculum design。**整个 learning loop 是 foundation-model-bootstrapped**，human 只在 initial calibration（preference pair 校准 rubric）时参与。

### Intuition 4: Generalization from game to photorealistic 是 "embodied skeleton" 的 transfer

SIMA 2 在 Genie 3 photorealistic environment 上 work，暗示视觉导航和 keyboard/mouse mapping 这些底层 capability 是 **scene-agnostic** 的。Game 的具体 visual style 是 surface-level variation；agent 学的是 "see 3D scene → decide movement → emit mouse keypress" 这个 abstract pipeline。

这给 robotics transfer 带来希望：如果 SIMA 2 能从 game 转移到 Genie 3 photorealistic world，进一步 transfer 到 real-world robot（通过 keyboard/mouse 接口控制 robot）就有 plausible path。Gemini Robotics 1.5 (https://arxiv.org/abs/2510.03342) 已经在做这个 direction。

### Intuition 5: Mixture training = Pareto frontier maintenance

如果纯 action data finetune，model 沿 embodied competence axis 走，但 reasoning axis 掉下 frontier。如果纯 pretraining data，反过来。SIMA 2 的 mixture 让 model 沿 frontier 移动，两个 axis 都 improve / maintain。

数学上，gradient 来自三个 source：
- $\nabla \mathcal{L}_{\text{human}}$ → push toward motor control
- $\nabla \mathcal{L}_{\text{bridge}}$ → push toward reasoning + action integration
- $\nabla \mathcal{L}_{\text{pretrain}}$ → maintain base capabilities

Mixture weight 决定 trajectory 在 frontier 上的 position。

---

## 10. Limitations 和 open questions

论文 Section 5 承认的：
- Very long-horizon multi-step reasoning 不足
- Context window 受限（low-latency 约束）
- Low-level precision action（精细 motor control）仍是 open challenge

我看到的几个 methodological gap：

1. **Latency 数字没披露**：Flash-Lite + structured text output 的实际 inference 速度？frame rate 多少？这决定 real-time 性能不能。

2. **RL 算法没明说**：是 PPO、GRPO (https://arxiv.org/abs/2402.03300)、DPO、还是某种 verifiable-reward 变体？AIME 在 RL 阶段反而 improve 的现象，没 RL 算法细节很难解释。

3. **Self-improvement 的 training signal 怎么用**：reward $r \in [0, 100]$ 是 continuous，是做 regression、preference learning、还是 threshold 成 binary success？Trajectory-level 还是 step-level credit assignment？

4. **Genie 3 上的 generalization 量化不够**：只在 navigation task 上做，而且 train/test split 是 urban/natural 这种 coarse partition。真正的 cross-modal generalization（例如 train on urban transfer to underwater）没测。

5. **Combat 为什么差**：paper 解释是 motor difficulty，但没量化 motor skill vs tactical decision 的 attribution。Valheim hunting deer 需要 downwind approach + crouch + quick attack，这到底是 perception 问题、planning 问题、还是 motor precision 问题？

6. **Bridge data 的 "causally consistent" 怎么验证**：Gemini Pro 生成的 reasoning 是否真的 causally drove action，还是 post-hoc rationalization？这个 distinction 对 training signal 质量 critical。

---

## 11. 与相关工作的 positioning

| 工作 | 关系 |
|------|------|
| SIMA 1 (https://arxiv.org/abs/2404.10179) | Direct predecessor，相同 evaluation framework |
| RT-2 (https://robotics-transformer2.github.io/) | VLA 先驱，但用 action token；SIMA 2 用 structured text |
| OpenVLA (https://arxiv.org/abs/2406.09246) | Open source VLA，robot focus |
| π0, π0.5 (https://arxiv.org/abs/2410.24164) | Physical robot VLA，flow matching |
| Gemini Robotics 1.5 (https://arxiv.org/abs/2510.03342) | 同公司 robot VLA，可能 share underlying tech |
| Genie 2 / 3 (https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | World model，提供训练 environment |
| Voyager (https://arxiv.org/abs/2305.16291) | Minecraft agent，用 LLM 做 skill library |
| VPT (https://arxiv.org/abs/2206.11695) | Minecraft，video pretraining for behavioral cloning |
| OMNI / OMNI-EPIC (https://arxiv.org/abs/2306.01711, https://arxiv.org/abs/2405.15768) | Open-endedness，model 作为 task setter 和 reward |
| Clune AI-GAs (https://arxiv.org/abs/1905.10985) | 理论框架，SIMA 2 + Genie 3 是 instance |
| Silver & Sutton "Era of Experience" (https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) | 哲学定位，experience-stream learning |
| BALROG (https://arxiv.org/abs/2411.04405) | LLM/VLM game reasoning benchmark |
| Lumine (https://arxiv.org/abs/2511.08892) | ByteDance 的 3D open-world generalist |
| Game-TARS (https://arxiv.org/abs/2510.23691) | Pretrained foundation model for game agent |

SIMA 2 的独特组合：**Gemini backbone + structured text action + game training + Genie 3 generalization + Gemini self-improvement loop**。每个 component 单独看都不新，但组合起来是第一次完整 demo of open-ended embodied self-improvement in diverse 3D worlds。

---

## 12. 我的整体 take

这篇 paper 的重要性不在任何 single technique，而在 **完整 stack 的 integration**。Foundation model 提供 reasoning，game environment 提供 embodied training ground，Genie 提供 infinite environment generation，Gemini 提供 universal task + reward，整个系统形成一个 **closed-loop self-improvement engine**。

这是 DeepMind 2024-2025 工作的集大成：SIMA 1 验证 generalist embodied agent 可行，Genie 1/2/3 验证 universal world model 可行，Gemini Robotics 验证 VLA 在 robot 上可行。SIMA 2 把这些 thread 拉到一起，提供一个 concrete recipe。

如果你 build intuition about 一个 thing，build 关于 **"action as language generation"** 的 intuition。这个 framing 解决了 VLA 的根本 tension——如何让 model 既能 act 又能 reason。SIMA 2 的答案：让 action 本身就是 language 的一种形式，所有 reasoning capability 自动 flow through。

进一步 reading 我推荐：
- SIMA 1 paper: https://arxiv.org/abs/2404.10179
- RT-2 paper: https://robotics-transformer2.github.io/
- Clune AI-GAs: https://arxiv.org/abs/1905.10985
- Genie 2 blog: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Silver & Sutton Era of Experience: https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
- Hancock et al. Actions as Language: https://arxiv.org/abs/2509.22195
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- Wen et al. RLVR: https://arxiv.org/abs/2506.14245
- MineDojo: https://arxiv.org/abs/2206.01394
- OMNI: https://arxiv.org/abs/2306.01711
- OMNI-EPIC: https://arxiv.org/abs/2405.15768
- Voyager: https://arxiv.org/abs/2305.16291
- BALROG: https://arxiv.org/abs/2411.04405
- EmbodiedBench: https://arxiv.org/abs/2502.09560
- OpenEQA: https://arxiv.org/abs/2403.20531
- Lumine: https://arxiv.org/abs/2511.08892
- Game-TARS: https://arxiv.org/abs/2510.23691
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- VPT: https://arxiv.org/abs/2206.11695

需要我深入展开哪一块？比如 self-improvement 的 reward model 具体怎么 calibrate、bridge data 生成的 prompt 怎么设计、或者 hierarchical composition 的 memory mechanism？
