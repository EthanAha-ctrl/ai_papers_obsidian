---
source_pdf: Vega Learning to Drive with Natural Language Instructions.pdf
paper_sha256: a77ef435eb5210fe35f9cff1737c09d41f223ed734212b6aaae6f36b3dade7a5
processed_at: '2026-08-13T00:13:11-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Vega 用人话讲讲

## 这篇paper到底在干嘛

一句话：**让自动驾驶车听人话**。

你现在上车的导航只能听"左转"、"直走"这种死板指令。Vega想让你说"前面那车太慢了，超过去赶下个绿灯"，车真的能听懂照做。

就这么个事。听起来简单，但实际上触及了autonomous driving里一个很根本的问题。

---

## 为什么这事难

传统自动驾驶走两条路：

**路线一：抄作业**
给模型看一万个人怎么开车，让它学个"average driver"。问题在于，不同司机开车风格不同——有人激进有人保守，你一平均，得到的是个四不像的"中庸策略"。而且模型只会模仿，不知道"为什么"这么开。

**路线二：指令集**
把驾驶行为压缩成closed set：左转、右转、直走、停车。这跟第一代语音助手似的，你必须说exact phrase才能识别，稍微换个说法就懵了。

Vega想做的第三条路：**open-ended instruction following**。用户用自然语言说任何驾驶意图，模型理解并执行。

---

## 三个核心idea

Vega的技术贡献我归结成三件事，一件比一件聪明。

### Idea 1: 造数据的trick

你手头有10万段expert driving video，每段都是"车在什么场景下做了什么action"。但你没有"这action对应的语言指令"。

怎么办？**反过来问VLM**。

拿Qwen2.5-VL-72B，给它看一段视频的前4帧+后10帧，跟它说："你描述一下这10帧里车在干嘛，然后给我一句instruction能指导agent做出同样行为"。

VLM就生成类似"Follow the car ahead and go straight through the intersection"这种指令。

这里有个细节trick很关键：VLM对ego-vehicle自身的运动感知不准。它能看到"前面有车在动"，但不太能准确说"我自己在加速还是减速"。所以作者加了一层rule-based补充——用speed、acceleration、turn rate阈值把场景分类，转成closed-set instruction，作为auxiliary prompt喂给VLM，让它生成更准确的指令。

最终拿到85,109个train scene + 12,144个test scene，每个scene都有instruction annotation。

**Intuition**：这步本质上是在做"反向翻译"——从expert action反推intent。传统imitation learning是"看action学action"，instructional driving是"看action+intent的对应关系学intent→action的mapping"。数据结构变了，模型能学的东西也变了。

项目主页：https://zuosc19.github.io/Vega

---

### Idea 2: 为什么加world model——sparse vs dense supervision的问题

这是paper里最深刻的insight。

你想训一个model：输入是图像+指令，输出是8个2D waypoint（24个数字）。

问题来了：**输入是百万像素的图像+几百token的语言指令，输出只有24个数字**。从高维到低维的mapping，gradient信号极其稀疏。模型学不到rich representation，容易overfit到surface pattern。

Vega的解法：**让模型同时预测未来画面**。

不只预测action，还预测"如果我执行这个action，下一帧画面会变成什么样"。

这一下supervision density就爆炸了：
- Action loss：24个数字的MSE
- Image loss：整个未来帧的pixel-level MSE

从"24维监督"变成"百万维监督"。

**为什么这有用？** 因为要生成合理的未来画面，模型必须真正理解：
- 这条instruction意味着什么语义行为
- 执行这个action后世界会如何演化
- Instruction、action、visual outcome三者之间的causal chain

action loss只告诉模型"对不对"，image loss告诉模型"懂不懂"。

Ablation数据特别有说服力：

| Setting | PDMS |
|---------|------|
| 有future frame prediction | 77.9 |
| 去掉future frame prediction（只剩action） | **51.8** |

去掉world model直接暴跌26个点。这数字说明world model不是"锦上添花"，是"雪中送炭"。

**Karpathy式的intuition**：这跟你教小孩开车一个道理。光跟小孩说"你这样转方向盘不对"（sparse reward），他学不会。但如果你让他开，然后指着窗外说"你看，你这么转，车就往那边偏了"（dense visual feedback），他就理解了action和consequence的关系。World model就是给model提供这种"dense visual feedback"。

参考Dreamer和world model的祖师爷工作：
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer V3: https://arxiv.org/abs/2307.15328

---

### Idea 3: Architecture——怎么把理解、生成、规划塞进一个model

这是工程上最难的部分。

Vega需要三种capability：
1. **Language + image understanding**（读懂场景和指令）→ autoregressive擅长
2. **Image generation**（生成未来画面）→ diffusion擅长
3. **Action planning**（输出trajectory）→ 低维但需精确

历史上大家怎么mix这两种paradigm？

**方案A: 全部AR**。把image quantize成discrete token，用next-token prediction统一处理。简单，但生成质量差。代表：Emu3、Janus、Chameleon。

**方案B: VLM+外挂diffusion**。VLM输出几个latent token，传给外部diffusion model生成image。问题是信息瓶颈——VLM理解了半天，只能通过几个token传递，损失大。

**方案C: Integrated transformer**。AR和diffusion在同一个transformer里，attention全打通，无信息损失。代表：Bagel、Transfusion、JanusFlow。

Vega选C，base是Bagel-7B。

但Vega在Bagel基础上做了关键扩展：**Mixture-of-Transformers (MoT)**。

#### MoT vs MoE——区别很关键

MoE（Mixture of Experts）：所有token共享attention，只在FFN层路由到不同expert。Sparsely activated。

MoT：**attention和FFN都duplicate**，每个modality有自己独立的transformer weights。每个token走自己modality的完整path，但cross-modality attention是dense的。

Vega的三个module：
- **Understanding transformer**：基于Qwen2.5，3584 hidden size，28层，处理text+image understanding
- **Generation transformer**：同尺寸，处理image generation
- **Action expert**：**256 hidden size**！极小

为什么action expert能这么小？因为action是低维的（24个数），不需要3584维的representation来编码。这是个很聪明的parameter efficiency设计。

Ablation验证了这个选择：

| Setting | PDMS |
|---------|------|
| 用Action Expert (256 dim) | 77.9 |
| 用VLM module处理action | 77.6 |
| 用Diffusion module处理action | **19.7**（灾难） |

用diffusion module处理action直接崩了。为什么？因为diffusion module的weights是为了image VAE latent优化的，跟action的分布完全不是一个domain。

**Intuition**：这就像让人去做完全domain mismatch的任务。你让一个画师去写代码，他会用画画的思路去理解代码，结果灾难。每个modality需要自己的"inductive bias"，MoT就是给每个modality专门的weights学专门的bias。

MoT paper: https://arxiv.org/abs/2411.04996

---

## 那个最妙的training trick

讲个细节，paper里一笔带过但我觉得特别精妙。

**问题**：训练时，action和future image都是diffusion process，要在一个autoregressive sequence里同时denoise。但sequence是causal的——后面的token attend到前面的token。如果前面是noisy latent，后面attend到的就是noise，跟inference时（前面已fully denoised）的分布不一致。

**解法**：每个target latent复制两份。
- 第一份加noise，作为denoising target
- 第二份保持clean，作为后续token的condition
- Mask掉noisy那份，不让后面token看到

这样后面token attend到的永远是clean state，跟inference一致。

**Intuition**：这就像考试时，后面的题不需要看前面题的草稿，只需要看前面题的最终答案。Vega在训练时让后面token看前面token的"最终答案"（clean latent），而不是"草稿"（noisy latent）。这种design避免了information leakage和distribution shift的双重问题。

---

## Attention Mask设计反映modality本质

这个细节也很有意思。

Vega的attention mask是分层的：
- **Block级别**：lower triangular，每个block只能看之前的block
- **Text block内部**：causal mask + RoPE（语言顺序敏感）
- **Image/Action block内部**：full attention + sinusoidal positional embedding（不区分顺序）

为什么？因为text是sequential的——"overtake the car"这3个token有顺序意义。image是set-like的——一张图的所有patch共同描述同一帧，没有inherent顺序。

**Intuition**：different modality有不同的topology。强行用同一种mask会mismatch。这种"modality-aware mask design"反映了对data structure的深度理解。

---

## 实验里值得注意的点

### Best-of-N的本质

Vega†用N=6的best-of-N，从86.9提到89.4。

这本质是**inference-time scaling**。diffusion sampling是stochastic的，多次sample得到不同trajectory，用scoring function选最好的。

跟LLM里beam search、majority voting一个道理。但这里有个隐忧：scoring function用的是什么？如果用NAVSIM metric，部署时没有ground truth，怎么办？

这是个open problem。实际部署需要realistic scoring function，可能是另一个learned model。

### NAVSIM v1 vs v2的差异

Vega在v1表现稍弱（87.9 vs DriveVLA-W0的93.0），v2表现强（86.9，SOTA级别）。

作者解释：v1的metric偏向risk-averse policy，Vega学的是"alternative valid strategies"，不一定是最保守的。v2加了更多metric（lane keeping、history comfort等），Vega的instruction following能力在这些metric上更突出。

而且Vega只用1x camera，对手都用3x+LiDAR。单目接近SOTA本身就很impressive。

### Extended Comfort偏低

Vega在EC上76.3，比DiffusionDrive的87.7差不少。EC衡量jerk、lateral acceleration等舒适度。

**为什么？** 因为instruction-following模型在执行指令时会更"果断"地调整速度——用户说"加速追上前面那车"，它就真的加速，jerk自然大。而imitation learning学的是average behavior，天然更平滑。

这是个fundamental trade-off：**个性化vs舒适性**。听用户话就意味着有时要"激进"。这个trade-off未来可能通过instruction里加"smoothly"这种qualifier来缓解。

---

## 更大的图景

Vega坐落的几个trend交汇点：

1. **End-to-end VLA**：放弃modular pipeline，unified model
2. **World model as supervision**：dense visual feedback替代sparse action reward
3. **Instruction following**：从imitation到personalized driving
4. **Unified understanding+generation**：AR+diffusion融合
5. **Mixture architectures**：modality-specific参数，cross-modality interaction

Vega把五个trend整合到一个working system，这个integration本身就是contribution。

---

## Karpathy角度的思考

你一直强调的"learn by doing + dense feedback"理念，跟Vega的world model supervision本质相同：

- Sparse action supervision = 只看期末成绩
- Dense image supervision = 每一步都有visual feedback

后者学得快、学得深。这个principle在robotics、driving、甚至LLM agent训练里都越来越主流。

Vega还暗示一个方向：**world model可以作为implicit reward model**。如果action和instruction不一致，生成的future image会偏离ground truth——这个偏离本身就是reward signal。未来可以用world model做model-based RL或MPC，sample多个action，用world model评估consequence，选最优。这就是best-of-N的延伸——N=∞就是MPC。

这种"generative world model + planning"的范式，跟Dreamer、MuZero的思路殊途同归：用learned world model做lookahead planning。区别在Dreamer用latent state，Vega用pixel-level image，supervision更dense但computation更贵。

MuZero: https://arxiv.org/abs/1911.08265

---

## Limitations我看到的

1. **Latency**：AR+diffusion+best-of-N，real-time能不能跑？paper没给数字，潜在bottleneck
2. **Instruction distribution bias**：instruction是Qwen2.5-VL生成的，有VLM自己的bias，跟真实用户指令可能有gap
3. **Closed-loop没验证**：NAVSIM v2是pseudo-simulation，真实路测或CARLA closed-loop才是终极test
4. **Scoring function for best-of-N**：部署时用什么选best？open problem
5. **Long-horizon memory**：只看4帧history，长程任务（multi-turn instruction、episodic memory）做不了

未来方向我认为是：
- Vega + RLHF：用world model做reward shaping + 人类preference fine-tune
- Vega + explicit reasoning chain：把AlphaDrive的CoT reasoning加进来
- Vega + closed-loop training：在CARLA里真正close the loop

---

## 一句话总结

Vega证明了一件事：**给driving model加上"想象未来"的能力，它就能更好地"听懂指令"**。

这个insight听起来反直觉——为什么预测画面能帮助规划轨迹？因为dense visual supervision逼着model真正理解instruction→action→consequence的causal chain，而不是surface pattern matching。

这跟人类学开车的process很像：新手盯着前方，脑子里不断"预测"下一秒会发生什么，根据预测调整方向盘。Vega把这个process显式建模了。

Project: https://zuosc19.github.io/Vega  
Code: https://github.com/zuosc19/Vega

---

# Vega: Learning to Drive with Natural Language Instructions 深度解析

## 1. Paper 核心动机与定位

这篇 paper 由 Tsinghua University 的 Jiwen Lu 团队（Wenzhao Zheng, Sicheng Zuo 等）和 GigaAI 合作完成，项目主页：https://zuosc19.github.io/Vega，代码：https://github.com/zuosc19/Vega。它处在 autonomous driving 里一个正在快速演进的 sub-field：**instructional driving**，即让 driving agent 不仅模仿 expert policy，还能听懂 open-ended 自然语言指令执行个性化驾驶行为。

核心问题表述得非常清晰。现有 VLA 模型（如 EMMa、Orion、DiffusionDrive、AutoVLA、ReCogDrive 等）大都走两条路：
- 模仿 averaged expert policy，行为保守；
- 仅处理 closed-set 的简单 navigation commands（"turn left"、"go straight"），无法 generalize 到 "overtake the front car to catch the next green light" 这类 open-ended 指令。

Vega 想做的本质上是个性化驾驶 agent：用户在着急赶路时给一个 high-level intent，agent 据此规划 trajectory 并预测 resulting future scene。

---

## 2. InstructScene Dataset 的构建（关键细节）

这是 paper 里很关键的一块。基于 NAVSIM（85k train scenes）扩展出 instruction annotation。

### 2.1 标注 Pipeline（两阶段）

**Stage One: Scene Understanding**
- 用 Qwen2.5-VL-72B-Instruct 作为 annotation model
- 输入：14 帧前视 camera image @ 2Hz，分辨率 1920×1080
- 前 4 帧 = past + current observation（agent 推理时能看到）
- 后 10 帧 = future observation（agent 推理时看不到，仅用于 annotation）
- 第一次 prompt：描述前 4 帧的场景，识别 traffic participants + static objects
- 第二次 prompt：描述后 10 帧的 vehicle driving behavior 及与已识别 participants 的 interaction

**Stage Two: Instruction Formulation**
- 把视觉输入 + Stage One 的 scene description 一起 prompt 给 VLM
- 让 VLM 生成 concise driving instructions，使得 agent 据此能预测后 10 帧的 action

### 2.2 Rule-Based 补充（关键 trick）

VLM 对 ego-vehicle 自身 motion 感知不准，所以加了一层 rule-based instructions：
- 用 speed, acceleration, turn rate 三个阈值切分场景
- 转成自然语言（closed-set，但 motion cues 精确）
- 作为 auxiliary prompt 喂给 VLM，让 VLM 综合生成 diverse + accurate instructions

最终得到 85,109 train + 12,144 test annotated scenes，构成 InstructScene。

**Intuition**：这一步其实在做 reverse engineering of instructions——把 expert trajectory 反过来翻译成"指导这个 trajectory 的语言指令"。这正是从 imitation driving 转向 instructional driving 的关键：训练数据的 supervision 信号从 action 变成，这样 agent 必须理解 L 才能预测 A。

参考链接：
- Qwen2.5-VL: https://qwenlm.github.io/blog/qwen2.5-vl/
- NAVSIM: https://github.com/autonomousvision/navsim

---

## 3. 从 Imitation Driving 到 Instructional Driving 的形式化

Paper 用四个公式清晰地区分了不同范式：

### Eq 1: 传统 end-to-end 模仿驾驶
$$A_t = \mathcal{M}([I_{t-T}, \ldots, I_t], [A_{t-T}, \ldots, A_{t-1}])$$

- $A_t$: 当前时刻 ego car 的 action
- $[I_{t-T}, \ldots, I_t]$: 过去 $T+1$ 帧的图像观测（$T$ 是历史窗口长度）
- $[A_{t-T}, \ldots, A_{t-1}]$: 过去 $T$ 个 action
- $\mathcal{M}$: 端到端 driving model

### Eq 2-4: Modular pipeline (Perception-Prediction-Planning)
$$\mathbf{z} = \mathcal{P}_{er}(I_{t-T}, \ldots, I_t)$$
$$\mathbf{v} = \mathcal{P}_{re}(\mathbf{z})$$
$$A_t = \mathcal{P}_{lan}(\mathbf{z}, \mathbf{v}, [A_{t-T}, \ldots, A_{t-1}])$$

- $\mathbf{z}$: scene representation（perception 输出）
- $\mathbf{v}$: agents 的 future motion（prediction 输出）
- $\mathcal{P}_{er}, \mathcal{P}_{re}, \mathcal{P}_{lan}$: 三个模块

这种 modular pipeline 依赖昂贵的 3D annotation（box, map, trajectory labels）。

### Eq 5: 现有 VLA 模型
$$A_t, D_t = \mathcal{W}([I_{t-T}, \ldots, I_t], [A_{t-T}, \ldots, A_{t-1}])$$

- $D_t$: textual description of the scene（VLM 输出）
- $\mathcal{W}$: VLA model

VLA 没有 instruction input，只能输出 averaged policy + 描述。

### Eq 6: Vega 的 instructional driving
$$A_t = \mathcal{V}([I_{t-T}, \ldots, I_t], [A_{t-T}, \ldots, A_{t-1}], L_t)$$

- $L_t$: 当前时刻的自然语言指令（新增的关键输入）
- $\mathcal{V}$: instruction-based driving model

**Intuition**：这四个公式的演进逻辑是 input 信息量的逐步扩充——从只有 observation+history action，到 modular 中间表示，到 VLA 加 description，到 instructional driving 加 language command。Language 在最后一种范式里从"输出"变成了"输入"，这是质的转变。

---

## 4. Unified Generation + Planning 的核心 Insight

这是 paper 最关键的思想。Vega 不只预测 action $A_t$，还同时预测 future image $I_{t+1}$（或 $I_{t+K}$）。这是 vision-language-**world**-action model，多了一个 world modeling dimension。

### 4.1 为什么要加 world modeling？

Paper 给出核心论证：**Information disparity problem**。
- Visual + language input 是高维 dense signal
- Action output 是低维 sparse signal（8 个 2D waypoints）
- 高维→低维的 mapping 学起来很困难，sparse supervision 不够

加 future image generation 后：
- 提供 dense, pixel-level supervision
- 强制模型学习 instruction → action → visual outcome 的 causal chain
- World modeling 让模型理解 "如果我做了 A_t，世界会变成 $I_{t+1}$"
- Generation 和 planning 互为监督，mutual refinement

这跟 Doe-1 [82]、DriveVLA-W0 [33] 思路一脉相承，但 Vega 是第一个把 instruction-following 加进去的 vision-language-world-action model。

参考：
- Doe-1: https://arxiv.org/abs/2412.09627
- DriveVLA-W0: https://arxiv.org/abs/2510.12796
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122

### 4.2 训练目标的因果结构

Paper 强调"causal chain"：agent perceives $I_t$ → receives $L_t$ → decides $A_t$ → observes $I_{t+1}$。模型架构用 causal attention 强制这个 reasoning pathway。

---

## 5. Joint Autoregressive-Diffusion Architecture（架构详解）

这是 Vega 的技术核心。它融合了三种范式：autoregressive (understanding) + diffusion (generation) + MoT (parameter decoupling)。

### 5.1 三种 unified understanding+generation 范式对比

Paper Section 2.3 总结了三种 pipeline：

1. **Quantized AR** (Emu3, Chameleon, Unified-IO 2, JanusPro)：把图像 quantize 成 discrete tokens，全部用 next-token prediction。简单但图像质量不如 diffusion。
2. **External Diffusion** (DreamLLM, Seed-X)：VLM 输出几个 latent token 作为 condition 喂给外部 diffusion model。信息瓶颈（窄接口）。
3. **Integrated Transformer** (Transfusion, Bagel, JanusFlow, LMFusion)：AR 和 diffusion 在同一个 transformer 里，深度整合，无信息损失。

Vega 选择了第三种，base model 是 Bagel-7B [6]。

参考：
- Bagel: https://arxiv.org/abs/2505.14683
- Transfusion: https://arxiv.org/abs/2408.11039
- JanusFlow: https://arxiv.org/abs/2410.13848
- Emu3: https://arxiv.org/abs/2409.18869

### 5.2 Input Encoding 细节

**Text**
- 用 Qwen2.5 tokenizer
- 自然语言指令 $L_t$ → text tokens

**Image (understanding + generation)**
- Forward-view camera only（单目，参数效率高）
- 用 VAE encoder 把图像编码成 latents $F_t^V$（用于 generation）
- 用 SigLIP2 ViT encoder 再编码一份 image features，append 到 VAE latents（用于 understanding）
- 这种 dual encoding 借鉴 Bagel 的设计：VAE latents 用于 diffusion 生成，SigLIP features 用于语义理解

**Action**
- 把 2D absolute trajectory $\text{traj} = [(x, y, \theta), \ldots]$ 转成 relative movements $A = (\Delta x, \Delta y, \Delta \theta)$
- 关键 trick：relative representation 让不同时刻的 action 共享同一分布，便于 normalize
- 用 linear head 投影到 model 的 latent dimension

### 5.3 Input Sequence 构造（非常关键）

序列结构：
```
[I_{t-T}, ..., I_t, A_{t-T}, ..., A_{t-1}, L_t, (A_t^{noisy} 或 A_t^{clean} + I_{t+K}^{noisy})]
```

历史 images 和 actions 放最前，然后是 instruction，最后是 prediction target。

**Attention Mask 设计**（这是 paper 里很精妙的一处）：
- 整体是 block lower triangular matrix
- 每个 block（一张 image、一个 action、或 instruction）只能 attend 到之前的 block
- Text block 内部：strict lower triangular（causal self-attention）+ consecutive RoPE indices
- Image/Action block 内部：full attention mask + shared RoPE index + sinusoidal positional embedding（用 relative position 而非 RoPE）

**Intuition**：text 是 sequential、causal 的（语言 token 顺序敏感），所以用 causal + RoPE；image/action 内部是 set-like（所有 token 共同描述同一帧），所以用 bidirectional + sinusoidal。这种混合 mask 设计反映了不同 modality 的内在结构差异。

### 5.4 Noisy/Clean Latent Trick（解决 train/inference mismatch）

这是 paper 里我最欣赏的技术细节之一。

**问题**：训练时要同时优化多个 diffusion process（action 和 image）在一个 autoregressive sequence 里。如果直接 concat noisy input，后面的 tokens 会 attend 到前面的 noisy latents，但 inference 时是 fully denoised，造成 mismatch。

**解决方案**：
- 对每个 latent 做 duplicate
- 第一份 $F_t^{\text{noisy}}$：加 noise，做 denoising supervision target
- 第二份 $F_t^{\text{clean}}$：保持 clean，作为后续 token 的 condition input
- Mask 掉 $F_t^{\text{noisy}}$ 不让后续 token attend，只让它们 attend clean 版本

**Intuition**：这个 trick 让训练时每个 diffusion step 的 condition 都对应 inference 时的 clean state，解决 information leakage + mismatch 双重问题。本质上是个训练时的 "look-ahead" 机制：让后面的 prediction 看到的不是当前正在 denoise 的中间状态，而是最终 clean 状态。

### 5.5 Mixture-of-Transformers (MoT) 设计

来自 Meta 的 MoT paper [38]，Vega 在此基础上做了 modality-specific 的细化。

**MoT vs MoE 的关键区别**：
- MoE：只有 FFN 用 separate weights，attention 是 shared
- MoT：attention + FFN 都 duplicate，整个 transformer 块都是 separate

Vega 的具体配置：
- **Understanding transformer**：基于 Qwen2.5 LLM，hidden size 3584，depth 28 layers，处理 text + visual understanding tokens
- **Generation transformer**：同样 design，处理 image generation tokens
- **Action expert**：hidden size **256**（注意这个 256，是 Vega 的 parameter efficiency 核心），处理 action planning tokens

两个大 transformer 都从 Bagel-7B 初始化，action expert 是新建的小模块。

**Forward 过程**：
1. Interleaving multimodal sequence 切分成 segments
2. 每个 segment 路由到对应 module（understanding / generation / action）
3. 在 attention 层时，segments re-assemble 计算全局 causal attention（cross-modality interaction）
4. FFN 层时，再 split 到各自 module

**Intuition**：MoT 的核心 insight 是不同 modality 需要不同的 "inductive bias"。Text understanding 需要强大的 language reasoning capability（Qwen2.5），image generation 需要强大的 visual synthesis capability（Bagel diffusion），action planning 是低维但需要精确的 motor control（small hidden size 够用）。Cross-modality interaction 通过 shared attention 实现，而 modality-specific capability 通过 separate FFN + attention weights 实现。

参考：
- MoT: https://arxiv.org/abs/2411.04996
- LMFusion: https://arxiv.org/abs/2412.15188
- Qwen2.5: https://arxiv.org/abs/2412.15115

---

## 6. 训练目标与 Inference

### 6.1 Loss Function

**Action Loss** (Eq 7):
$$\mathcal{L}_A = \mathbb{E}_{A_t^{(N)}, \epsilon, m}\left[\|\epsilon - \hat{\epsilon}(A_t^{(N)}, \epsilon, m, I_t^{(-T)}, L_t)\|^2\right]$$

- $A_t^{(N)} = [A_t, \ldots, A_{t+N-1}]$：未来 $N$ 步的 action plan（$N=8$）
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：sampled Gaussian noise（diffusion 的标准 noise）
- $m$：random diffusion timestep（noise level）
- $I_t^{(-T)} = [I_{t-T}, \ldots, I_t]$：过去 $T+1$ 帧的 observation
- $L_t$：current instruction
- $\hat{\epsilon}$：模型预测的 noise
- 这是 standard diffusion loss（MSE on noise prediction），对应 normalized relative action

**Image Loss** (Eq 8):
$$\mathcal{L}_V = \mathbb{E}_{F_{t+K}^V, \epsilon, n}\left[\|\epsilon - \hat{\epsilon}(F_{t+K}^V, \epsilon, n, I_t^{(-T)}, L_t, A_t^{(N)})\|^2\right]$$

- $F_{t+K}^V$：future image 的 VAE latents（$K$ 是未来时刻 index）
- $n$：random diffusion timestep
- 关键：image generation conditioned on action plan $A_t^{(N)}$
- 这正反映了 causal chain：instruction $L_t$ → action $A_t^{(N)}$ → future image $I_{t+K}$

**Joint Objective**:
$$\mathcal{L}_{\text{pretrain}} = \lambda_A \cdot \mathcal{L}_A + \lambda_V \cdot \mathcal{L}_V$$

$\lambda_A = \lambda_V = 1.0$（equal weighting）。

### 6.2 Classifier-Free Guidance (CFG) 训练 trick

为 enable CFG in inference，训练时随机 drop：
- Text tokens
- ViT features
- Clean VAE latents
- Clean action tokens

同一 modality 属于不同 image/action 的 tokens 一起 drop 或 keep（不是独立 drop）。这是为了避免训练分布与 inference 分布的差异。

### 6.3 Inference

- 用 Classifier-Free Guidance Diffusion 生成 action
- 同时启用 image guidance + text guidance
- Inference 时 future image prediction conditioned on fully denoised action
- 主要 inference 任务是 action planning，但保留 image generation capability

### 6.4 Training Setup

- 200k steps on 8 H20 GPUs
- 4 historical images，predict 8 future actions + 1 future image（actions 末端的 image）
- Learning rate 2e-5 with 2500 warmup steps
- Per-device batch size 1（gradient accumulation 推测用了，但 paper 没明说）
- EMA decay 0.9999（保存 EMA model 作为 checkpoint）

---

## 7. 实验结果深度解读

### 7.1 NAVSIM v2 主结果（Table 1）

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|--------|-----|------|------|------|-----|------|-----|-----|-----|--------|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **Vega** | **98.9** | 95.3 | 99.4 | 99.9 | 87.0 | 98.4 | 96.5 | 98.3 | 76.3 | **86.9** |
| **Vega†** | **99.2** | 96.6 | 99.5 | 99.9 | 87.5 | **98.7** | **97.4** | **98.4** | 84.5 | **89.4** |

(Vega† = best-of-N with N=6)

观察：
- Vega 在 No at-fault Collision (NC) 上很强（99.2），说明 instruction-following 没牺牲安全性
- 在 Driving Direction Compliance、Traffic Light Compliance、Lane Keeping、History Comfort 上达到 SOTA
- Extended Comfort (EC) 偏低（76.3 / 84.5），跟 DriveVLA-W0 的 58.9 比已经好，但比 DiffusionDrive 的 87.7 差。EC 衡量 jerk, lateral acceleration 等，instruction-following 模型在遵循指令时可能更"激进"地调整速度，导致 comfort 下降
- Best-of-N 策略带来显著提升：86.9 → 89.4

### 7.2 NAVSIM v1 主结果（Table 2）

| Method | Sensors | NC↑ | DAC↑ | TTC↑ | C.↑ | EP↑ | PDMS↑ |
|--------|---------|-----|------|------|-----|-----|-------|
| UniAD | 6x Cam | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DiffusionDrive | 3x Cam + L | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| AutoVLA† | 3x Cam | 99.1 | 97.1 | 97.1 | 100 | 87.6 | 92.1 |
| DriveVLA-W0† | 1x Cam | 99.3 | 97.4 | 97.0 | 99.9 | 88.3 | 93.0 |
| **Vega** | **1x Cam** | 98.9 | 95.3 | 96.1 | 100 | 81.6 | 87.9 |
| **Vega†** | **1x Cam** | 99.2 | 96.6 | 96.9 | 100 | 83.4 | 89.8 |

观察：
- Vega 只用 **单目前视相机**，而 SOTA 方法大多用 3x 或 6x camera + LiDAR
- PDMS 87.9 / 89.8 接近 SOTA 但没超过（DriveVLA-W0† 93.0）
- Paper 解释：NAVSIM v1 metrics 不平衡，偏向 risk-averse policy，而 Vega 学到了 alternative valid strategies
- EP（Ego Progress）较低，说明 Vega 倾向更保守 / 更遵循 instruction 而非"激进前进"

### 7.3 Ablation: Future Frame Prediction（Table 3）

| Setting | PDMS↑ | EPDMS↑ |
|---------|-------|--------|
| Random Frame | 77.3 | 75.2 |
| Action Only | 51.8 | 48.9 |
| Next Frame | 77.9 | 76.0 |

关键发现：
- **去掉 future frame prediction (Action Only)**：PDMS 暴跌到 51.8！这印证了 world modeling 的 dense supervision 至关重要
- **Random Frame**（随机选 8 个 future frame 之一，并在 text prompt 中指定 index）：77.3，跟 Next Frame 接近
- **Next Frame**（predict $I_{t+1}$）：77.9
- 结论：future prediction 任务的存在 >> 具体哪个 future frame。World modeling 本身的作用是 supervision signal，不是"精确预测下一帧"那么严格

**Intuition**：这个 ablation 说明 world modeling 在 Vega 里更像一个 regularizer / auxiliary task，强制 representation 学习 world dynamics。具体预测哪个 frame 不重要，重要的是 prediction task 给了 dense pixel-level gradient。

### 7.4 Ablation: Action Expert（Table 4）

| Setting | PDMS↑ | EPDMS↑ |
|---------|-------|--------|
| Use Diffusion | 19.7 | 19.6 |
| Use VLM | 77.6 | 75.7 |
| Action Expert | 77.9 | 76.0 |

关键发现：
- 用 generation transformer (Diffusion) 处理 action：灾难性失败（19.7）
- 用 understanding transformer (VLM) 处理 action：77.6，接近 Action Expert
- 独立 Action Expert（hidden size 256）：77.9，最好

**Intuition**：
- Diffusion transformer 处理 action 失败的原因：它的 weights 优化目标是 image generation（VAE latent 的 noise prediction），跟 action distribution 完全不同
- VLM 处理 action 接近 Action Expert，说明 VLM 的 representation 已经包含 driving-relevant 信息
- 独立 Action Expert 的优势：parameter efficient（256 vs 3584 hidden size）+ slightly better performance
- 这个 ablation 印证了 MoT 的核心假设：不同 modality 需要不同 inductive bias

### 7.5 Interleaving Observation-Action Ablation（Figure 4）

预训练时 image-action sequence 长度对 finetuning loss 的影响：
- Non-interleaving (original)：收敛慢，最终 loss 高
- Interleaving length 2/4/6：初始 loss 高（pretraining-finetuning mismatch），但收敛快，最终 loss 低
- 预训练 interleaving length 越长，finetuning 最终 loss 越低

**Intuition**：Interleaving 让模型学到 image-action 的 dynamics（看到 $I_t$ + $A_t$ 后能更好预测 $I_{t+1}$）。这种 temporal dynamics 学习在 pretraining 时积累，finetuning 时虽然 sequence 结构变了，但学到的 dynamics knowledge 迁移过来加速收敛。这跟 LLM pretraining 的一般规律一致：pretraining 学到的是 universal structure，finetuning 是 adaptation。

参考：
- NAVSIM v1: https://github.com/autonomousvision/navsim  
- NAVSIM v2 (pseudo-simulation): https://arxiv.org/abs/2506.04218
- DiffusionDrive: https://arxiv.org/abs/2410.15349
- AutoVLA: https://arxiv.org/abs/2506.13757

---

## 8. 关键 Visualization 分析（Figure 1, 5, 6）

### 8.1 Figure 1: 同一 scene + 不同 instruction → 不同 trajectory

展示了 Vega 的核心能力：
- Instruction A "Stop immediately and remain still" vs Instruction B "Pull up to the side" → 同一 scene 下生成完全不同的 trajectory
- "Follow the car, go straight through the intersection" → 复合指令，agent 同时执行 follow + go straight
- "Stop at the crosswalk, wait for the light to turn green" → 时间维度上的复合行为

### 8.2 Figure 5: Instruction-based speed control

两个 scene，每个 scene 测试两条 instruction：
- Scene 1: "Accelerate immediately to catch up with the car in front" vs "Proceed along the lane at constant speed"
- Scene 2: "Remain steady and follow the car in front" vs "Gradually slow to a stop and remain stationary"

Visualization 显示在 front-view image + BEV map 上画 trajectory，模型确实能根据 instruction 调整 speed profile。

### 8.3 Figure 6: Joint action + image generation

最 impressive 的 visualization：同一 scene + 不同 instruction → 不同 action sequence + 不同 future image。
- 关键场景：approaching intersection、encountering another vehicle
- Action 和 image 都 instruction-consistent
- 这印证了 world modeling 的成功：模型真的学到了 "如果我做 A，世界会变成 I"

---

## 9. 与相关工作的深度对比

### 9.1 vs DiffusionDrive [39]

DiffusionDrive 是 CVPR 2025 的工作，也是 diffusion-based trajectory planning。
- 共同点：都用 diffusion 做 action planning
- 不同点：DiffusionDrive 没有 instruction input，没有 world modeling，纯 planning
- Vega 的优势：instruction following + world model supervision

### 9.2 vs DriveVLA-W0 [33]

DriveVLA-W0 是 arXiv 2025 的工作，把 world modeling 整合进 VLA。
- 共同点：都用 world modeling 提供 dense supervision
- 不同点：DriveVLA-W0 没有 instruction following，是 imitation driving
- DriveVLA-W0 在 NAVSIM v1 PDMS 93.0 比 Vega 89.8 高，但它无法做 instructional driving

### 9.3 vs Doe-1 [82]

Doe-1 是 closed-loop driving 的 world model。
- 共同点：unified understanding + prediction + planning
- 不同点：Doe-1 没有 language instruction，没有 image generation（更多 occupancy / scene representation）
- Vega 更接近 generative driving agent

### 9.4 vs AutoVLA [85], ReCogDrive [35]

这两个是 SOTA VLA 方法。
- AutoVLA 用 adaptive reasoning + reinforcement fine-tuning
- ReCogDrive 用 reinforced cognitive framework
- Vega 没有 RL，但用 world modeling 替代了 RL 的角色——通过 dense visual supervision 来 align instruction understanding 和 action execution
- Vega 在 NAVSIM v1 上稍弱（这些方法用 3x camera + RL），但 Vega 用 1x camera 也接近，且多了 instruction following 能力

### 9.5 vs Bagel-7B [6]

Vega 的 base model 是 Bagel-7B，这是 unified understanding+generation 的 integrated transformer。
- Bagel 处理 text + image understanding + image generation
- Vega 扩展到 text + image + **action** 三模态，并加入 causal driving structure
- Action expert 是 Vega 在 Bagel 上的核心架构创新

### 9.6 vs LLM + RL Approaches (AlphaDrive [29])

AlphaDrive 用 RL 训练 VLM 在 driving 上 reasoning。
- 互补关系：Vega 用 world modeling 提供 dense supervision，AlphaDrive 用 RL 优化 reasoning chain
- 未来方向：把 RL reasoning + world modeling supervision 结合，类似 AlphaDrive + Vega 的 union

参考：
- Bagel: https://github.com/ByteDance-Seed/Bagel
- Doe-1: https://arxiv.org/abs/2412.09627
- AlphaDrive: https://arxiv.org/abs/2503.07608

---

## 10. 我的 Intuition 与深度思考

### 10.1 Information Disparity 的本质

Paper 反复强调的 information disparity 问题，我觉得可以更精确地表述为：**low-dimensional action space is a bottleneck for learning rich visual-language representations**。

想象一个 extreme case：如果 action 只是 1 维（"前进 vs 停止"），从数百万像素 + 几百 token 的 instruction 学到这个 1 维 signal，gradient 几乎全是"被压缩"过的，模型学不到 rich intermediate representation。

加 future image generation 后，loss surface 是 high-dimensional 的，每个 pixel 都贡献 gradient，representation 被强制 rich。这种 rich representation 反过来 help action planning——即便 action loss 是 sparse 的，模型已经从 image loss 学到了好 representation。

这跟 self-supervised learning 的核心 insight 一致：dense auxiliary task 学到的 representation 是 downstream task 的好 initialization。但 Vega 是 multi-task joint training，不是两阶段。

### 10.2 World Model 作为 "Implicit Reward Model"

RL 视角看，world model $\hat{\epsilon}(F_{t+K}^V | \ldots)$ 实际上是个 "implicit reward / constraint"：
- 如果 action $A_t^{(N)}$ 跟 instruction $L_t$ 不一致，生成的 future image 会跟 ground truth 偏离
- 如果 action 在物理上不合理，future image 会出现不合理的 scene evolution
- Image loss 提供了 action 一致性的 implicit signal

这跟 model-based RL（Dreamer, MuZero）的 world model 用法类似：world model 用于 planning 时的 lookahead。Vega 是用 world model 做 training 时的 dense supervision，而不是 inference 时的 planning。一个自然 extension：在 inference 时用 world model 做 model predictive control (MPC) — sample 多个 action，用 world model 评估 resulting image，选 best。这正是 best-of-N 的 spirit。

### 10.3 Instruction Following vs Imitation Learning 的本质区别

Imitation learning 学的是 $p(A | I)$，即 observation 到 action 的 mapping。
Instructional driving 学的是 $p(A | I, L)$，即加 condition 的 mapping。

关键区别：$p(A | I)$ 会 collapse 到 average behavior（multiple valid actions under same observation，average 后是 conservative）；$p(A | I, L)$ 能 keep multi-modality（不同 $L$ 给不同 $A$）。

Vega 用 diffusion 做 action generation 是有道理的——diffusion naturally 处理 multi-modal distribution，而 AR 会有 mode collapse 倾向。这跟 Diffusion Policy (Chi et al., RSS 2023) 的 motivation 一致。

参考：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Dreamer: https://danijar.com/dreamer/
- MuZero: https://arxiv.org/abs/1911.08265

### 10.4 MoT 的 Parameter Efficiency 分析

Vega 的 architecture parameter 大致估算：
- Understanding transformer (Qwen2.5 base, 3584 hidden, 28 layers): ~3-4B
- Generation transformer (Bagel diffusion): ~3-4B
- Action expert (256 hidden): ~10-50M
- Vision encoders (VAE + SigLIP2): ~1B

Total ~7-9B，比单 7B 模型大一些，但 Action expert 几乎 negligible。这是 Vega 的一个聪明设计：把"贵"的 capability（language reasoning, image generation）放 transformer 里，"便宜"的 capability（action planning）用小 module。

跟 MoE 对比，MoT 没有 router，所有 modality 都参与每次 forward，但 FFN 是 modality-specific 的。MoE 是 sparse activation（每次只 activate k experts），MoT 是 dense activation（每个 token 走自己 modality 的 FFN，但 cross-modality attention 是 dense）。所以 MoT 的 FLOPs 比 MoE 高，但比"完全独立的三个 transformer"低。

### 10.5 单目 vs 多目：Vega 的"经济性"主张

Vega 只用 1x forward-view camera，这是有意为之：
- NAVSIM SOTA 大多用 3x 或 6x camera + LiDAR
- Vega 用 1x 也能达到接近 SOTA 的 planning 性能
- 这呼应 paper 开头说的 "vision-centric autonomous driving is promising due to economic advantages"
- Instruction following 能力不依赖 surround view，单目足够理解 scene + 跟随 instruction

但 paper 也暗示 surround view 能进一步提升 performance。Vega 的 1x 配置是为了证明 concept，不是为了 beat 多 sensor 方法的绝对数字。

### 10.6 Best-of-N 的本质

Vega† 用 best-of-N (N=6) 提升 86.9 → 89.4 (+2.5 EPDMS)。这个 trick 的本质：
- Diffusion sampling 是 stochastic 的，多次 sample 得到 different trajectories
- 用某个 scoring function（这里推测是 NAVSIM metric proxy）选最好的
- 在 inference 时实现类似 "test-time scaling"

这跟 LLM inference 的 best-of-N / beam search / majority voting 思路一致。在 driving 上，scoring function 是关键——paper 没详细说，但推测用 NAVSIM metrics 或它们的代理。这暗示一个 limitation：实际部署时需要 realistic scoring function，不能依赖 ground truth metric。

### 10.7 跟 RLHF 的对比思考

LLM 的 instruction following 是通过 RLHF 学的——reward model + PPO。Vega 没用 RL，但通过 world modeling 提供了类似 reward signal 的 dense supervision。

这种 supervised approach 的优势：
- 没有 reward hacking 风险
- 训练稳定
- 不需要 reward model

劣势：
- 仍然 bounded by expert data（即便加了 instruction annotation）
- 没有真正的 "preference learning"
- 无法 discover 新的 driving strategy

未来方向：Vega + RLHF 的结合——用 world model 做 reward shaping + 人类 preference 数据 fine-tune。这跟 LLM 的 RLHF 路径相似。

### 10.8 Future Image Generation 的实际意义

Vega 生成 future image 不只是 auxiliary task，还有 deployment 价值：
- 可解释性：driver 看 future image 就知道 agent 准备做什么
- Safety verification：future image 里如果有 unexpected object，可以 trigger safety fallback
- Simulation：可以 generate 大量 training data（合成新 scene）

这跟 GAIA-1, Vista, DriveDreamer 的 video world model 一脉相承，但 Vega 把 generation 和 action planning 紧密耦合起来。

参考：
- GAIA-1: https://arxiv.org/abs/2309.17080
- Vista: https://vista-worldmodel.github.io/
- DriveDreamer: https://arxiv.org/abs/2309.09777

---

## 11. Limitations 与潜在扩展方向

Paper 没显式列 limitations，但从我的分析里能看出几个：

### 11.1 延迟问题
Integrated transformer 里 AR + diffusion 是 sequential 的，加上 best-of-N (N=6)，inference latency 可能很高。Real-time driving 要求 ~10Hz decision making，Vega 是否能实时？Paper 没给 latency 数字，是个潜在 bottleneck。

### 11.2 Instruction Annotation 的 OOD 问题
Instructions 是用 Qwen2.5-VL-72B 生成的，会有 VLM 自己的 bias。用户真实指令的 distribution 可能跟 InstructScene 的 distribution 不同，造成 OOD。可以用 human annotation 验证 / 补充。

### 11.3 Multi-Modal Future 的 Handling
Diffusion 能 sample multi-modal future，但 best-of-N 选 best 的时候用什么 criterion？如果用 NAVSIM metric，会 bias toward "safe" trajectory，丢失 instruction following 的多样性。Need a metric that balances instruction compliance + safety。

### 11.4 Closed-Loop Evaluation
NAVSIM v2 是 "pseudo-simulation"（reactive 但还是 simulation）。Real closed-loop evaluation（如 CARLA, real road test）才能验证 Vega 的实际 deployment capability。Doe-1 强调 closed-loop，Vega 在这块没验证。

### 11.5 Action Expert 的 Capacity
Action expert hidden size 256 是基于 "action 是低维" 的假设。如果 action 扩展到 high-dim（如 full vehicle control: steering, throttle, brake, gear），256 可能不够。需要 sensitivity analysis。

### 11.6 Memory Mechanism
Vega 只看 4 帧 history。Long-horizon driving（如 multi-turn instruction, episodic memory）需要更长的 memory。RAG-style memory 或 recurrent state 可能是 extension。

### 11.7 VLM Reasoning Integration
Vega 是 direct mapping，没有显式 chain-of-thought reasoning。AutoVLA 用了 reasoning，AlphaDrive 用了 RL reasoning。把 Vega 的 world modeling + explicit reasoning chain 结合是个 promising direction。

参考：
- CARLA: https://carla.org/
- AutoVLA (RL fine-tuning): https://arxiv.org/abs/2506.13757

---

## 12. 总结：Vega 在 autonomous driving 大图景里的位置

Vega 在 autonomous driving 的几个 trend 交汇点上：
1. **End-to-end VLA**：从 modular pipeline 到 unified model
2. **World Models**：用 generative future prediction 做 dense supervision
3. **Instruction Following**：从 imitation 到 instructional driving
4. **Unified Understanding+Generation**：AR + diffusion 的 integrated transformer
5. **Mixture Architectures**：MoT 比 MoE 更彻底的 modality decoupling

Vega 的 contribution 是把这五个 trend 整合到一个 working system，在 NAVSIM 上达到 competitive performance，同时开启 instructional driving 这个 new capability。

对 Karpathy 而言，Vega 的核心 insight 跟你在 Eureka Labs / 教育领域强调的 "learn by doing + dense feedback" 思路相通：world model 提供的 dense pixel-level supervision 类似 "做项目时每个 step 都有反馈"，比 sparse action supervision（类似"只看最终成绩"）学得更好。这个 paradigm 在 robotics、driving、甚至 LLM agent 训练里都越来越主流——用 environment feedback 代替 sparse task reward。

未来如果能结合 RL fine-tuning（AutoVLA 路线）+ explicit reasoning chain（AlphaDrive 路线）+ real closed-loop evaluation，instructional driving 可能真的能 deploy 到 personal car 上，让 driver 说 "今天赶时间，开快点" 而车真的懂。

---

## Reference Links 汇总

**Vega & Related Tsinghua Works**
- Vega Project: https://zuosc19.github.io/Vega
- Vega Code: https://github.com/zuosc19/Vega
- LDM (Large Driving Models): https://github.com/wzzheng/LVM
- Doe-1: https://arxiv.org/abs/2412.09627
- OccWorld: https://arxiv.org/abs/2401.08509 (Wenzhao Zheng)
- GaussianAD: https://arxiv.org/abs/2412.10371

**VLA Methods**
- EMMa: https://arxiv.org/abs/2410.23262
- AutoVLA: https://arxiv.org/abs/2506.13757
- ReCogDrive: https://arxiv.org/abs/2506.08052
- OpenDriveVLA: https://arxiv.org/abs/2503.23463
- DriveVLA-W0: https://arxiv.org/abs/2510.12796
- Orion: https://arxiv.org/abs/2503.19755

**World Models**
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- GAIA-1: https://arxiv.org/abs/2309.17080
- GAIA-2: https://arxiv.org/abs/2503.20523
- Vista: https://vista-worldmodel.github.io/
- DriveDreamer: https://arxiv.org/abs/2309.09777
- DriveDreamer-2: https://arxiv.org/abs/2401.14379

**Unified Understanding + Generation**
- Bagel: https://github.com/ByteDance-Seed/Bagel
- Transfusion: https://arxiv.org/abs/2408.11039
- Janus: https://arxiv.org/abs/2410.13848
- JanusPro: https://arxiv.org/abs/2501.17811
- Emu3: https://arxiv.org/abs/2409.18869
- MoT: https://arxiv.org/abs/2411.04996
- LMFusion: https://arxiv.org/abs/2412.15188

**Benchmarks**
- NAVSIM: https://github.com/autonomousvision/navsim
- NAVSIM v2 paper: https://arxiv.org/abs/2506.04218
- CARLA: https://carla.org/

**Base Models**
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Qwen2.5-VL: https://qwenlm.github.io/blog/qwen2.5-vl/
- SigLIP: https://arxiv.org/abs/2303.15343

**RL & Reasoning Extensions**
- AlphaDrive: https://arxiv.org/abs/2503.07608
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Dreamer V3: https://arxiv.org/abs/2307.15328
- MuZero: https://arxiv.org/abs/1911.08265
