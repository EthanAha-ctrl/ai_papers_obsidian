---
source_pdf: PIVOT-R Primitive-Driven Waypoint-Aware World.pdf
paper_sha256: 5741ecf52b1eefc6069ac402fd1ee4ccd0ccae1d3638c5b304ab2d2c1054e6a1
processed_at: '2026-08-06T04:16:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PIVOT-R 人话版

## 一句话版本

让机器人做任务,别直接从 instruction 硬 predict 每一步 action,先让 VLM 帮你把任务拆成 "approach → grab → lift" 这种 primitive chunk,然后只预测每个 chunk 结束时的关键画面(waypoint),最后用一个小 model 去把当前画面往 waypoint 画面靠。三个模块各跑各的频率,VLM 慢悠悠想,action head 高频出信号,互不阻塞。

---

## 为什么要这么搞

想象你教小孩搭积木。你不会跟他说 "手指往左移 3cm,再下移 2cm,夹紧 0.5N" 这种鬼话。你会说 "先靠近那块红的,抓住,举起来"。小孩脑子里有个 **目标画面**——"手抓着红积木悬在半空",然后自己想怎么动。

之前的 robot learning 基本是两种路子:

**路子 A**:end-to-end transformer,vision + language token in,action token out([RT-2](https://robotics-transformer.github.io/rt2/), [Octo](https://octo-models.github.io/))。问题在于每个 timestep 都得让模型吐一个 action,模型为了拟合 dense 数据,就 memorize 表面 pattern,泛化就差。而且 VLM 这么大的模型每步都推理一次,延迟爆炸。

**路子 B**:world model,[Daydreamer](https://daydreamer.github.io/) / [3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA) / [Surfer](https://arxiv.org/abs/2310.14672) 这类。想法挺好——预测未来嘛。但它们在 **每个 timestep 都建模** 一帧未来,这是浪费。真正影响任务成败的就那几个关键瞬间:抓到的瞬间、抬起的瞬间、放下的瞬间。中间那些 "手在空中缓慢移动" 的帧,predict 它们干嘛?它们对任务结果没 semantic 贡献,反而引入 noise,让 long-horizon 误差累积。

PIVOT-R 的核心 insight 就一句话:**别预测每一帧,只预测 task-relevant 的 waypoint 帧**。这跟你之前在 [Tweet](https://twitter.com/karpathy) 上讲的 "predict structured latent 而不是 raw next token" 是一个道理——chunking + abstraction 比 dense prediction 更 sample-efficient。

---

## Waypoint 是什么

作者在 SeaWave 数据集上手工标了 waypoint。满足下面任一条件的帧就算 waypoint:

1. **Primitive action 完成的瞬间**——比如 "grasp" 这个动作抓到了东西
2. **运动学转折点**——机械臂速度归零,或者 gripper 状态翻转(open↔close)

这两个定义互补——第一个是 **语义层面** 的 chunk 边界,第二个是 **运动学层面** 的 chunk 边界。

消融实验很说明问题(Table 4):
- 两个都用:74.19%
- 只用 primitive completion:69.10%(掉 5%)
- 只用 robot state change:**43.65%(掉 30%!)**
- 用 next frame 代替 waypoint:44.45%(掉 30%)
- 用 final frame:61.36%(掉 13%)

**关键 takeaway**:光看运动学信号没用,必须有语义。光看终点也不行,要中间的关键 chunk。这个 finding 跟 NLP 里 BPE tokenization 的哲学很像——chunk 要有语义意义,纯字符级或纯词级都不好,sub-word chunk 才 sweet。

---

## 架构三件套

### 模块一:VLM 做 primitive parsing

用 [LLaVA](https://github.com/haotian-liu/LLaVA) 做,输入 "Give me a container of drinking water" + 当前图像。三轮对话 prompt:

1. 描述场景:"桌子上有红瓶、蓝瓶、杯子"
2. 想象 action sequence:"move to cup → grasp cup → lift"
3. 决定当前该做啥:"move to cup"

输出一个文本 primitive action,比如 `move to the cup`。10 个 primitive 类别:`close to / grasp / move up / move down / release / rotate / push / pull / open / close`。

**频率 3Hz**。VLM 慢,所以低频跑。意图层面变化本来也慢——"我要去拿杯子" 这种决策几秒内不会变。

### 模块二:Scene prediction(12 层 transformer)

输入:primitive action 的 CLIP text feature + 历史 4 帧图像 CLIP feature
输出:**waypoint 帧的 CLIP feature**(不是图像!)

$$F_{M'_t} = \Phi_{sp}(\text{CLIP\_text}(P_t), \text{CLIP\_image}(O_{t-h:t}))$$

这里有个关键设计——**不 decode 像素,只 predict CLIP feature**。借鉴 [I-JEPA](https://github.com/facebookresearch/ijepa) 的思路:pixel-level prediction 让模型纠结于 texture / lighting 这种没用的细节,feature-level prediction 让模型聚焦 semantic content。消融显示换成 MAE-style pixel prediction 掉 7.8%。

**频率 10Hz**。场景语义变化中速。

### 模块三:Action prediction(3 层 transformer,极小)

输入:predicted waypoint feature + 当前 visual feature + robot state MLP
输出:7 维 action(6 DoF delta + binary gripper)

$$A'_t = \Phi_{ap}(F_{M'_t}, \text{CLIP\_image}(O_{t-h:t}), \text{MLP}(S_{t-h:t}))$$

**频率 30Hz**。控制信号要高频,Franka panda 标配控制频率。

action head 只 3 层 transformer 就够,消融显示加大反而过拟合(-2.9%)。直觉是:给定 waypoint feature 这么强的 guidance,policy 头只需做 local navigation——"把当前状态往 waypoint 拉就行",任务被 simplify 了。

---

## AHE:三个模块各跑各的

三个模块用 multithreading,每个线程从上一个模块的 buffer 拉最新数据。如果 buffer 没新数据,就用旧的。彼此不阻塞。

效果:
- 同步执行(所有模块每步都跑):77.11% 成功率,756ms 一步 → 控制频率 1.3Hz,机器人动不了
- 异步执行(各跑各的):74.19%,27ms 一步 → 控制频率 37Hz,能 closed-loop
- **28× speedup,只掉 2.9%**

这个 trade-off 太值了。同步版的 2.9% 提升来自每步都拿最新 VLM 输出,但实际上 "我要去拿杯子" 这种高层意图 300ms 内不会变,用旧 result 完全够用。这跟 control theory 里 multi-rate control 是一个思想——外环慢做 trajectory planning,内环快做 MPC。

之前没人这么做是因为大家觉得 "VLM 必须每步都看新观测才能正确决策",但 PIVOT-R 发现 **高层语义决策的 stale tolerance 比想象的高得多**。

---

## 结果有多炸

[SeaWave benchmark](https://github.com/SeaWave-Benchmark),四个难度等级:

| Model | L1 | L2 | L3 | L4 | Mean |
|-------|-----|-----|-----|-----|------|
| RT-1 | 67 | 49 | 39 | 35 | 47.6 |
| Octo | 70 | 48 | 35 | 34 | 46.6 |
| GR-1 | 77 | 56 | 37 | 34 | 51.1 |
| Surfer | 75 | 61 | 45 | 38 | 54.7 |
| **PIVOT-R** | **88** | **78** | **73** | **58** | **74.2** |

平均比 SOTA 高 19.45%。而且 **越难的任务提升越大**:
- Level 1(verb+noun,"pick the milk"):+13%
- Level 4(描述 + 空间推理,"retrieve the object behind the one on the right"):+20%

直觉解释:简单任务大家都做得好,waypoint 的优势不明显。复杂任务需要 reasoning,VLM 把它拆成 primitive + waypoint guidance,直接砍掉了 policy 的 search space——policy 不用从抽象指令推到 action,只需从 waypoint 画面推到 action,中间的 "语义 gap" 被 waypoint 填上了。

real-world 实验在 Franka 上也验证了(Table 2),mean 40.28% vs Surfer 34.26%。但 pushing 任务上 PIVOT-R 略输 Surfer(25 vs 32)——pushing 是 contact-rich 任务,连续 force interaction,waypoint 的 "discrete key frame" 假设不适用。这是方法的明确 limitation。

---

## 有意思的 emergent capability

### OOD instruction 泛化

输入:"Hello, I want to use the coffee machine, but something is blocking it."

VLM 推理:咖啡机前面有瓶子挡着,应该把瓶子拿走 → mapping 到学过的 `pick the bottle` → PIVOT-R 执行。

这种 reasoning 能力来自 VLM,但 grounding 在 PIVOT-R 学过的 primitive action set 上,所以不会乱来。这跟 [VILA](https://github.com/openvla/vila) 用 GPT-4 做 planning 一个味道,但 PIVOT-R 把 planning 和 waypoint grounding 显式结合,比纯 LLM planning 更 actionable。

### 新任务分解

没学过的任务 "move the middle object to the left" → VLM 拆成 `close to → clamp → move up → push left → put down → unclamp` → 每个 primitive 都学过 → 串起来执行。

这本质上是 **primitive composition**,类似 [Voyager](https://voyager.minedojo.org/) 的 skill library,但 manipulation 版。

### Ego4D pre-training

用 [Ego4D](https://ego4d-data.org/) 的 first-person human video(3500 小时)pre-train scene prediction module。因为这些视频里人手 manipulates 物体,跟机器人任务有结构相似性。

- Co-training 跟 SeaWave 混训:**效果降**(数据分布差太多)
- Pre-training 先 Ego4D 再 fine-tune SeaWave:**unseen background +4.16%, distractors +3.00%**

这验证了 scene prediction module 能从 human video 学到 transferable dynamics prior。跟 [SWIM](https://worldmodel.github.io/) 思路类似,但 PIVOT-R 用 feature prediction 不用 image generation,更轻量。

---

## 关键 design choices 背后的 intuition

### 为什么用 CLIP feature 做 waypoint target,不用 image

CLIP feature 是 semantic-level,对 "桌子上有杯子" 这种 content 敏感,对 "灯光从左打还是右打" 这种 nuisance 不敏感。waypoint 想表达的是 "下一个 sub-goal 的语义状态",不是像素级精确。这种 abstraction 让 model 天然 robust 到 distractor / lighting / background——验证了 Table 3 的 generalization 结果。

### 为什么 VLM 换大换小都没差(Table 4)

LLaVA → GPT-4:74.19 → 74.92(+0.7%)
LLaVA → Qwen-VL:74.19 → 73.28(-0.9%)

说明 PIVOT-R **不依赖 VLM 的 capacity**。VLM 只做 coarse primitive parsing,真正学 dynamics 的是 scene prediction module(12 层 transformer)。这跟你之前讲过的观点对得上——**架构和数据结构比模型大小重要**。RT-2 把 7B VLM 硬上机器人,效果反而不如 PIVOT-R 这种 "VLM 做高层 + 小 model 做 low-level" 的 hierarchical 设计。

### 为什么 action head 这么小就够

给定 waypoint feature 这么强的 sub-conditioning,action prediction 退化成 "从当前观测 navigate 到 waypoint" 的 local 问题。不需要 long-horizon planning,3 层 transformer 够了。这跟 [Octo](https://octo-models.github.io/) 的 observation 一致——policy head 不需要大,关键是 representation。

---

## 我觉得哪里还不够

### Waypoint 离散假设对 contact-rich 任务失效

Pushing、peg-in-hole 这种连续 force interaction 任务,没有清晰的 "primitive 完成" 瞬间。Push to 实验掉点就是这个 reason。要解决得引入 force/torque 的 continuous representation,不能只靠视觉 waypoint。

### Primitive set 是固定的 10 类

要 scale 到 general manipulation,primitive 集得 learnable。可以参考 [Genie](https://sites.google.com/view/genie-2024) 的 latent action model——从数据里自动 discover action primitives,而不是人工定义。

### CLIP feature 的精度限制

CLIP 是 semantic-level,对 1cm 级精度不够。要 fine-grained manipulation 可能需要 hierarchical feature——CLIP 高层 semantic + DINO low-level spatial。当前 PIVOT-R 在 "Put Spoon on Towel" 这种需要精确放置的任务上 SIMPLER 实验(Table 10)成功率只有 0.417,比 "Put Eggplant in Basket" 的 0.875 低很多,印证了这个 limitation。

### VLM 推理延迟的 tail latency

AHE 假设 VLM 推理稳定 3Hz,但 LLM 偶尔 cache miss / preemption 会 spike。如果 VLM thread 卡住 1 秒,scene prediction 就 stale 100ms,虽然能 fallback 到上一帧,但 waypoint 质量下降。real-time 系统需要 deadline-aware scheduling + watchdog,论文没讨论这个。

---

## 对你(Karpathy)的角度看

PIVOT-R 本质上是把你在 [Jetson nano talk](https://www.youtube.com/watch?v=VMj6DX5h7tQ) 里讲的 "world model 应该 predict latent structure not raw token" 的 idea,在 manipulation 上做了具体 instantiation。而且它做了一个 I-JEPA 没做的事——**把 latent prediction 跟 action execution 耦合**,让 predicted latent 成为 action head 的 sub-goal。这其实是 "predictive coding for control" 的一个 elegant 实现。

更深一层,PIVOT-R 验证了一个 thesis:**robot learning 不一定要 end-to-end from pixels to actions**。把任务 structurally chunk 成 primitive → waypoint → action 三层,每层用合适 capacity 的 model,用 asynchronous execution 把它们粘起来,反而比 billion-scale VLA model 又快又好。这跟你一直倡导的 "small model + good structure > big model + brute force" 的 thesis 一致。

Web links:
- 项目主页:[https://abliao.github.io/PIVOT-R](https://abliao.github.io/PIVOT-R)
- [I-JEPA](https://github.com/facebookresearch/ijepa)
- [V-JEPA](https://github.com/facebookresearch/jepa)
- [RT-2](https://robotics-transformer.github.io/rt2/)
- [Octo](https://octo-models.github.io/)
- [Surfer / SeaWave](https://arxiv.org/abs/2310.14672)
- [Daydreamer](https://daydreamer.github.io/)
- [3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA)
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Genie](https://sites.google.com/view/genie-2024)
- [SIMPLER benchmark](https://simpler-env.github.io/)
- [Ego4D](https://ego4d-data.org/)
- [Voyager](https://voyager.minedojo.org/)
- [VILA](https://github.com/openvla/vila)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MResT multi-resolution transformer](https://arxiv.org/abs/2401.14502)

---

# PIVOT-R 深度技术解析：Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation

这篇 paper 来自 Sun Yat-sen University + Peng Cheng Laboratory + Dataa Robotics + Huawei Noah's Ark Lab，核心解决的是 language-guided robotic manipulation 中两个被长期忽视的问题：**(1) 关键 waypoint 的世界建模能力弱**，**(2) 模块执行的计算冗余高、效率低**。我会从动机、架构、公式、实验、消融、以及它跟你熟悉的工作（V-JEPA、Dreamer、RT-2 等）的关系层层 build up intuition。

---

## 1. 核心动机：为什么之前的方法不行

### 1.1 Waypoint 缺失的问题

之前的 end-to-end manipulation model（如 [RT-2](https://robotics-repository.github.io/rt2/), [RT-X](https://robotics-transformer-x.github.io/), [Octo](https://octo-models.github.io/), [RoboFlamingo](https://robotflamingo.github.io/)）直接学习一个 mapping：

$$
(\text{instruction } l, \text{ observation } O_t) \rightarrow \text{action } A_t
$$

这种 dense token-level 的拟合让模型容易 **memorize surficial pattern**，而非学到 transferable dynamics。人类做 "move a cup" 时，内部 world model 自然地 chunk 成：

> close to the cup → grab → move up → put down

这种关键 frame 被作者定义为 **waypoint** M。如果模型在每个 timestep 都做预测，误差会在 long-horizon 中被 amplification（因为 local spatiotemporal 下 low-level actions 非常相似，randomness 累积）。这跟你在 [Tweet 关于 world models 的讨论](https://twitter.com/karpathy) 里讲的 "predict next token vs predict latent structure" 是一个味道。

### 1.2 同步执行的冗余问题

之前的方法（如 [Surfer](https://arxiv.org/abs/2310.14672), [Daydreamer](https://daydreamer.github.io/), [3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA)）把 VLM、world model、action head 全部以相同频率串行执行。但 VLM 推理一次要几百 ms，而 action head 需要每 30Hz 输出一次 control signal，强行同步会让 control loop 完全被 VLM bottleneck 卡死。

这跟 [MResT](https://arxiv.org/abs/2401.14502) 的 multi-resolution sensing 思想呼应，但 MResT 只做了 spatial-temporal resolution，没做 world modeling。

---

## 2. 整体架构

PIVOT-R = **WAWM (Waypoint-Aware World Model)** + **Action Prediction Module** + **AHE (Asynchronous Hierarchical Executor)**

![Architecture overview](https://abliao.github.io/PIVOT-R)

三个模块以不同频率运行：
- **Primitive Action Parsing (VLM)**: 频率 $v_1 = 3$ Hz（慢）
- **Scene Prediction (12-layer Transformer)**: 频率 $v_2 = 10$ Hz（中）
- **Action Prediction (3-layer Transformer)**: 频率 $v_3 = 30$ Hz（快）

满足 $v_1 < v_2 < v_3$。多线程运行，每个线程从 buffer 取最新数据；如果某模块未完成处理，返回上一帧结果。这种设计哲学类似 [ACT (Action Chunking with Transformers)](https://tonyzhaozh.github.io/aloha/) 的 chunking，但 PIVOT-R 把它推到了跨模块 asynchronous execution。

---

## 3. 数学公式逐个拆解

### 3.1 整体 formulation (Eq. 1)

$$
\pi(\text{VLM}(l, O_t), O_{t-h:t}, S_{t-h:t}) \rightarrow M'_t, A'_t
$$

变量含义：
- $\pi$: 整个 trainable manipulation policy（30M 参数，VLM 和 encoder 都 frozen）
- $l$: user language instruction，例如 "Give me a container of drinking water"
- $O_{t-h:t}$: 历史 $h+1$ 帧观测图像，$h=3$（4 帧 context window）
- $S_{t-h:t}$: 历史 robot state，6 维 $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$
- $M'_t$: 预测的 waypoint（feature 表示）
- $A'_t$: 预测的 7 维 action（6 DoF delta + 1 维 binary gripper）

### 3.2 Trajectory 数据结构 (Eq. 2)

$$
Tra = \{l, [O_1, S_1, A_1, M_1], \dots, [O_T, S_T, A_T, M_T]\}
$$

$T$ 是轨迹长度。**关键设计**：作者在 SeaWave 13K 数据集上额外标注了 waypoint $M$。Waypoint 定义为满足下面两条之一：
1. **Primitive Action Completion (PAC) frame**：某个 primitive action 完成的瞬间
2. **Robot State Change (RSC) frame**：机械臂速度 ≈ 0 或 gripper state 翻转

消融实验（Table 4）显示 **PAC 是主要贡献者**（去掉 PAC 只掉 5.1%），只用 RSC 掉 30.54%。这个对比极其重要——它说明 waypoint 的语义意义（"动作完成")比纯运动学意义（"速度归零"）更重要。

### 3.3 Primitive Action Parsing (Eq. 3)

$$
P_t = \big(l, \text{VLM}(\text{Prompt}(l), O_t)\big)
$$

- $P_t$: 给 scene prediction module 的 waypoint indicator（文本形式）
- $\text{Prompt}(l)$: 三轮对话 prompt（见 Appendix F.1）
  1. 第一轮：让 VLM 描述场景
  2. 第二轮：让 VLM 想象完成任务的 actions list
  3. 第三轮：让 VLM 输出当前应该做的单个 action

Primitive actions 共 10 类，object-centered 原则：`close to`, `grasp`, `move up`, `move down`, `release`, `rotate+(dir)`, `push+(dir)`, `pull+(dir)`, `open`, `close`。这个 action vocab 比 [CLIPort](https://cliport.github.io/) (pick/place/push)、[Transporter](https://transporternets.github.io/) 的简单 vocab 复杂得多，能覆盖更多 manipulation skill。

### 3.4 Scene Prediction (Eq. 4)

$$
F_{M'_t} = \Phi_{sp}\big(E_{\text{text}}(P_t), E_{\text{image}}(O_{t-h:t})\big)
$$

- $\Phi_{sp}$: 12-layer Transformer scene prediction module
- $E_{\text{text}}, E_{\text{image}}$: 都是 [CLIP-ViT-B/32](https://github.com/openai/CLIP) encoder
- $F_{M'_t} \in \mathbb{R}^{b \times n \times d}$，其中 $n=49$（ViT-B/32 patch tokens 数量 = $7\times 7$），$d=512$（CLIP 特征维度）

每层 Transformer 包含 self-attention + cross-attention + FFN。Scene prediction **预测的是 CLIP feature**，不是 pixel。这点是关键设计——参照 [I-JEPA](https://github.com/facebookresearch/ijepa) 的思想：predict in latent space 比 predict in pixel space 更 sample-efficient，也更聚焦 semantic content。消融 Table 4 显示换成 MAE-style pixel prediction 掉 7.8%，作者的解释是 "pixel-level prediction focuses too much on detail, causing key information to be ignored"。

### 3.5 Action Prediction (Eq. 5)

$$
A'_t = \Phi_{ap}\big(F_{M'_t}, E_{\text{image}}(O_{t-h:t}), \text{MLP}(S_{t-h:t})\big)
$$

- $\Phi_{ap}$: 3-layer Transformer action prediction module（极轻量）
- 输入三路：waypoint feature $F_{M'_t}$ + 历史 visual feature + 历史 robot state MLP encoding
- 输出 $A_t = (S, G) \in \mathbb{R}^{1\times 7}$，$G \in \{0, 1\}$ 是 binary gripper command

Action discretization 跟 [RT-1](https://robotics-transformer.github.io/) 一样：每维 256 bins，均匀分桶。Loss 用 Cross Entropy。

### 3.6 Loss

$$
\mathcal{L} = \mathcal{L}_{\text{scene}} + \mathcal{L}_{\text{act}}
$$

- $\mathcal{L}_{\text{scene}}$：predicted waypoint feature 与 CLIP-encoded ground-truth waypoint feature 的 $L_2$ distance
- $\mathcal{L}_{\text{act}}$：Cross Entropy on discretized action bins

注意 $\mathcal{L}_{\text{scene}}$ 不 decode 图像，直接在 feature space 监督。这个 trick 让训练快且稳定，跟 [V-JEPA](https://github.com/facebookresearch/ijepa) 的 predictor 在 latent space 操作一脉相承。

---

## 4. Asynchronous Hierarchical Executor (AHE) 设计哲学

AHE 是这篇 paper 最 practical 的贡献。核心 insight：**不同模块的语义信息变化速率不同**，强行同步是浪费。

| 模块 | 频率 | 信息变化速率 | 计算量 |
|------|------|-------------|--------|
| VLM (LLaVA) | 3 Hz | 慢（高层意图） | 高 |
| Scene Prediction | 10 Hz | 中（场景语义变化） | 中 |
| Action Prediction | 30 Hz | 快（控制信号） | 低 |

实现上用 multithreading，每个模块有自己的 thread + buffer。例如 VLM thread 把 output 推入 buffer，scene prediction thread 每隔 100ms 从 buffer 拉最新结果。如果 buffer 没更新，就用上一帧结果。

这套设计跟 control theory 里的 **multi-rate control** 是一个味——外环做 trajectory planning，内环做 MPC。在 robot learning 领域，[ACT](https://tonyzhaozh.github.io/aloha/) 的 action chunking、[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 的 temporal diffusion 也类似——但 PIVOT-R 把它显式做成了跨模块异步调度。

实测数据（Table 1 + Table 4）：
- 同步执行（PIVOT-R w/o AHE）：77.11% mean success rate，**756 ms** 每步
- 异步执行（PIVOT-R w/ AHE）：74.19% mean success rate，**27 ms** 每步
- **28× speedup**，只掉 2.9% 性能

这个 trade-off 对真实机器人 control 非常关键——30Hz 是 Franka panda 控制循环的常见频率，如果模型 756ms 一次输出，机器人根本没法跑 closed-loop。

---

## 5. 实验结果深度分析

### 5.1 SeaWave Benchmark 主表 (Table 1)

| Model | Level 1 | Level 2 | Level 3 | Level 4 | Mean | Time (ms) |
|-------|---------|---------|---------|---------|------|-----------|
| Gato | 34.74 | 30.53 | 23.16 | 20.00 | 27.11 | 139 |
| BC-Z | 41.05 | 32.63 | 23.16 | 25.26 | 30.53 | 12 |
| Octo | 69.79 | 48.48 | 34.69 | 33.58 | 46.64 | 18 |
| SUSIE | 78.89 | 48.48 | 32.50 | 29.17 | 47.26 | 434 |
| RT-1 | 67.38 | 49.47 | 38.95 | 34.74 | 47.64 | 21 |
| GR-1 | 77.08 | 55.56 | 37.31 | 34.33 | 51.07 | 35 |
| Surfer | 74.74 | 61.05 | 45.26 | 37.89 | 54.74 | 24 |
| **PIVOT-R** | **88.06** | **77.55** | **73.33** | **57.82** | **74.19** | 27 |

观察要点：
1. **Level 越难，相对提升越大**。Level 1 提升 13.32%，Level 4 提升 19.93%。这说明 waypoint guidance 在需要 reasoning 的复杂任务上收益更大——Level 4 任务如 "retrieve the object located behind the one to the right"，需要 visual perception + decision making，PIVOT-R 把它分解成 primitive actions，每个 action 又有 waypoint target，大幅降低了 policy 的 search 空间。
2. **PIVOT-R 在 Level 3 上提升 28.07%，是最大 jump**。Level 3 测的是 intention inference（"could you please go grab a refreshing beverage for me"——不直接说目标物名）。这说明 VLM 解析 primitive action + waypoint prediction 学到了 **functional reasoning**，超越了纯 object name matching。
3. **速度 27ms 跟 Surfer (24ms)、RT-1 (21ms) 同量级**，比 SUSIE (434ms) 快 16 倍。SUSIE 慢是因为用 video diffusion predictor 生成 sub-goal，PIVOT-R 用 feature prediction 绕开了生成开销。

### 5.2 Real-World 实验 (Table 2)

| Model | Pick up | Put on | Push to | Mean |
|-------|---------|--------|---------|------|
| Octo | 34.72 | 27.78 | 4.17 | 22.22 |
| RT-1 | 40.28 | 22.22 | 19.44 | 27.31 |
| GR-1 | 26.39 | 29.17 | 8.33 | 21.30 |
| Surfer | 41.67 | 29.17 | 31.94 | 34.26 |
| **PIVOT-R** | **54.17** | **41.67** | 25.00 | **40.28** |

注意 PIVOT-R 在 "Push to" 上比 Surfer 还低（25 vs 31.94）。作者解释：push 过程中向下的力增加阻力，模型难预测适应。这反映了 waypoint model 在 contact-rich 任务上的局限——waypoint 假设是 "discrete key frame"，但 pushing 是 continuous force interaction，更适合 force-aware model（参考 [ForceSight](https://arxiv.org/abs/2401.00465)）。

### 5.3 Generalization (Table 3)

PIVOT-R 在 unseen backgrounds / changing lights / distractors 上仍保持 59.17% / 61.67% / 55.83%，相比 Surfer 提升 12.5/15.84/15.0 个点。泛化能力来自 waypoint 的 **abstraction**——visual distractor 不影响 primitive action 的解析，scene prediction 在 CLIP feature space 操作也天然 robust 到 texture / lighting 变化。

### 5.4 SIMPLER Benchmark (Table 10, Appendix D)

[SIMPLER](https://simpler-env.github.io/) 是 real-world proxy benchmark（基于 BridgeData），PIVOT-R 在 Put Eggplant in Yellow Basket 上达到 0.875，整体 mean 0.393，超过 Octo-Small (0.295)。这进一步验证了 cross-embodiment 泛化能力。

---

## 6. 消融研究：每个设计为什么必要

### 6.1 Waypoint 选择策略 (Table 4 关键数据)

| Waypoint 选择策略 | Mean Success | Delta |
|--------------------|--------------|-------|
| **PAC + RSC (full)** | 74.19 | baseline |
| 仅 PAC | 69.10 | -5.09 |
| 仅 RSC | 43.65 | **-30.54** |
| next frame | 44.45 | -29.70 |
| interval (5-frame) | 48.52 | -25.70 |
| final frame | 61.36 | -12.80 |

build intuition：
- **next frame / interval frame 掉 30%**：太 trivial，模型学到的是 short-term dynamics，跟 next-frame prediction 退化。
- **final frame 掉 12.8%**：只看终点，丢失了中间关键 chunk，对 long-horizon task 不够。
- **RSC 单独用很差**：纯运动学信号 noise 大（机械臂减速可能 transient），缺乏语义 grounding。
- **PAC + RSC 互补**：PAC 给语义 chunking，RSC 捕获 PAC 漏掉的运动学转折（比如 grasp 成功但无 primitive 切换）。

### 6.2 VLM 选择 (Table 4)

| VLM | Mean | Delta |
|-----|------|-------|
| LLaVA (default) | 74.19 | baseline |
| Qwen-VL | 73.28 | -0.9 |
| GPT-4 | 74.92 | +0.7 |

**关键 finding**：换更大 VLM（GPT-4）几乎无提升，换小 VLM（Qwen-VL）也只掉 0.9%。这说明 PIVOT-R **不依赖 VLM 的 capacity**，VLM 只做 coarse primitive parsing，重活儿在 scene prediction module。这点对你（Karpathy）应该很 relevant——这呼应了你 "LLM 大不一定更好，关键是 architecture" 的观点。

### 6.3 AHE vs Synchronous

| 配置 | Mean | Time (ms) |
|------|------|-----------|
| PIVOT-R (AHE, async) | 74.19 | 27 |
| PIVOT-R w/o AHE (sync) | 77.11 | **756** |

同步版本反而高 2.9%，因为每步都拿到最新 VLM 推理结果。但 756ms 控制频率 ~1.3Hz，对 closed-loop robot 不可行。AHE 的设计哲学是 **"good enough" sub-goal** 比 "perfect but stale" sub-goal 更重要。

### 6.4 Scene Prediction 设计

| 设计 | Mean | Delta |
|------|------|-------|
| Feature prediction (default) | 74.19 | baseline |
| Pixel-level prediction (MAE-style) | 66.32 | -7.8 |

Pixel prediction 在 manipulation 上反而差，因为 pixel 重建会让 model 浪费 capacity 在 texture / lighting 细节上，loss 噪声大。Feature prediction（CLIP feature L2）更聚焦 "what's in the scene" 而非 "how it looks exactly"。这跟 [I-JEPA](https://github.com/facebookresearch/ijepa) 关于 non-generative SSL 的论证完全一致。

### 6.5 Action Module 容量

| Action module | Mean | Delta |
|---------------|------|-------|
| 3-layer Transformer (default) | 74.19 | baseline |
| Larger Transformer | 71.23 | -2.9 |

更大 action module **过拟合**。原因是 action prediction 是相对简单任务（给定 waypoint feature + 当前 observation，输出 7 维），3 层足够。这跟 [Octo](https://octo-models.github.io/) 的 finding 一致——policy head 不需要大。

---

## 7. Feature Analysis 深入 (Appendix C.1)

作者在 Fig 7 画了两条距离曲线：

- $D_1 = \|F_{O_t} - F_{M_t}\|_2$：当前观测 feature 到 ground-truth waypoint feature 的距离（蓝）
- $D_2 = \|F_{M'_t} - F_{M_t}\|_2$：predicted waypoint feature 到 ground-truth 的距离（红）

随着 task 推进：
- $D_1$ **单调下降**——观测逐步逼近 waypoint，符合 action prediction module "pull $O_t$ to $M_t$" 的目标
- $D_2$ **保持小且稳定**——预测 waypoint 误差方差小，提供 consistent guidance

这个分析很 elegant——它把 waypoint prediction 的 "guidance quality" 量化成了 feature space 的距离稳定性。直觉上，$F_{M'_t}$ 是 "下一个关键 sub-goal 在哪" 的稳定 beacon，让 action module 不必做 long-horizon planning，只需 local navigation to waypoint。

---

## 8. Emergent Capabilities

### 8.1 OOD Instruction Generalization (Appendix C.2.1)

例子：input "Hello, I want to use the coffee machine, but something is blocking it."
- VLM 解析为 "remove the bottle in front of the coffee machine"
- 进一步 mapping 到学过的 "pick the bottle"
- PIVOT-R 执行

这种 capability 跟 [VILA](https://github.com/openvla/vila)、[CoPa](https://github.com/huang-hx/CoPa) 用 GPT-4 做 planning 的思路类似，但 PIVOT-R 把 VLM reasoning 跟 waypoint grounding 耦合，比纯 LLM planning 更 actionable。

### 8.2 New Task Decomposition (Appendix C.2.2)

例子："Can you move the middle object to the left?"（没学过的任务）
- VLM 分解为 primitive sequence: `close to → clamp → move up → push left → put down → unclamp`
- 每一步用学过的 action 执行

这本质上实现了 **primitive composition**，类似 [RoboGen](https://arxiv.org/abs/2311.02455) 或 [Voyager](https://voyager.minedojo.org/) 的 skill library 思想，但 PIVOT-R 的 primitive 是在 manipulation trajectory 上 chunking 出来的，更 grounded。

### 8.3 跨数据集 Pre-training (Appendix C.2.3)

用 [Ego4D](https://ego4d-data.org/) 的 "Short Term Object Interaction Anticipation" 数据（3,500 小时 first-person video）：
- **Co-training**：Ego4D + SeaWave 混训 → 略降（数据分布差异大）
- **Pre-training**：Ego4D 预训 + SeaWave 微调 → unseen backgrounds +4.16%, distractors +3.00%

这验证了 scene prediction module 可以从 human video 学到 transferable dynamics，类似 [SWIM](https://worldmodel.github.io/) 的 idea，但 PIVOT-R 用 feature prediction 而非 image generation，更 lightweight。

---

## 9. 跟其他 World Model 工作的关系

| 工作 | World modeling 方式 | 与 PIVOT-R 关系 |
|------|---------------------|-----------------|
| [Daydreamer](https://daydreamer.github.io/) | Dreamer RL，latent imagination | PIVOT-R 不做 RL，做 imitation；用 waypoint chunking 而非 dense rollout |
| [3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA) | 3D occupancy + VLM | PIVOT-R 用 2D feature，但加 asynchronous execution |
| [Surfer](https://arxiv.org/abs/2310.14672) | Progressive reasoning on every step | PIVOT-R 直接 baseline；PIVOT-R 主要改进是 waypoint focus + AHE |
| [UniPi](https://uni-pi.github.io/) | Video generation as policy | PIVOT-R 不 generate video，只 predict feature；避免 video-action inconsistency |
| [SUSIE](https://surr.github.io/) | Video sub-goal predictor | PIVOT-R 比 SUSIE 快 16×，因为不 decode video |
| [GR-1](https://gr1-manipulation.github.io/) | Video pre-training | 互补：GR-1 用 video 学 dynamics prior，PIVOT-R 用 waypoint 做 trajectory structure |
| [V-JEPA / I-JEPA](https://github.com/facebookresearch/ijepa) | Latent predictive SSL | PIVOT-R 的 scene prediction loss 直接借鉴 I-JEPA 的 feature L2 |
| [Genie](https://sites.google.com/view/genie-2024) | Spatiotemporal video tokenizer + dynamics | Genie 是开放域游戏 world model，PIVOT-R 聚焦 manipulation waypoint |

---

## 10. Prompt 设计 (Appendix F)

作者精心设计了三轮 VLM 对话：
1. **Scene description**：让 VLM 先理解环境
2. **Action imagination**：让 VLM 列出完成任务的所有 action sequence
3. **Current action**：让 VLM 输出当前时刻应执行的单个 action

这种 chain-of-thought 风格 prompt 跟 [ReAct](https://react-lm.github.io/) / [Tree of Thoughts](https://tree-of-thought.github.io/) 是一个家族。关键 trick 是把 task 分解成 manageable primitive set，避免 VLM 自由发挥导致 OOD action。

---

## 11. 局限性与个人思考

作者自承 limitation：action 和 instruction 有时不一致（"push left" 被执行成 "push front"）。这暴露了 **VLM primitive parsing 和 low-level action 之间的 alignment gap**——primitive 是离散语义标签，action 是连续控制信号，没有显式 binding。

更深层 limitation 我推测：
1. **Waypoint 假设离散任务结构**：对 contact-rich 任务（pushing、peg-in-hole）失效。Push to 任务实验掉点正是这个 reason。
2. **CLIP feature 作为 waypoint target** 限制了 fine-grained 控制精度。CLIP feature 是 semantic-level，对 1cm 级精度不够。可能需要 hierarchical feature（CLIP 高层 + DINO low-level）。
3. **AHE 的 buffer 设计假设模块独立**：实际 VLM 推理延迟如果偶发 spike（比如 LLM cache miss），可能让下游模块 stale 期变长。需要 deadline-aware scheduling。
4. **Primitive action 集 10 类**：仍然相对受限。要让 PIVOT-R scale 到 general manipulation，需要让 primitive 集本身 learnable（类似 [Genie 的 latent action model](https://sites.google.com/view/genie-2024)）。

对你（Karpathy）的角度看，PIVOT-R 验证了一个你在 [Jetson nano 系列演讲里提过的观点](https://www.youtube.com/watch?v=VMj6DX5h7tQ)：**predict structured latent（waypoint）比 predict raw next token 更 sample-efficient，更适合 robotic control**。它把 I-JEPA 的 latent prediction idea 落到了 manipulation，并通过 asynchronous execution 解决了 VLM bottleneck——这两个 contribution 合在一起，让 30M 参数的小模型击败了 billion-scale VLA model（[RT-2](https://robotics-transformer.github.io/) 等）。

---

## 12. 关键 Web Links

- **项目页**：[https://abliao.github.io/PIVOT-R](https://abliao.github.io/PIVOT-R)
- **SeaWave Benchmark**：[https://github.com/SeaWave-Benchmark](https://github.com/SeaWave-Benchmark) （Surfer 论文：[https://arxiv.org/abs/2310.14672](https://arxiv.org/abs/2310.14672)）
- **I-JEPA**：[https://github.com/facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)
- **RT-1**：[https://robotics-transformer.github.io/](https://robotics-transformer.github.io/)
- **RT-2**：[https://robotics-transformer.github.io/rt2/](https://robotics-transformer.github.io/rt2/)
- **RT-H**：[https://rt-hierarchical.github.io/](https://rt-hierarchical.github.io/)
- **Octo**：[https://octo-models.github.io/](https://octo-models.github.io/)
- **GR-1**：[https://gr1-manipulation.github.io/](https://gr1-manipulation.github.io/)
- **Surfer**：[https://arxiv.org/abs/2310.14672](https://arxiv.org/abs/2310.14672)
- **Daydreamer**：[https://daydreamer.github.io/](https://daydreamer.github.io/)
- **3D-VLA**：[https://github.com/UMass-Foundation-Model/3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA)
- **UniPi**：[https://uni-pi.github.io/](https://uni-pi.github.io/)
- **SUSIE**：[https://surr.github.io/](https://surr.github.io/)
- **VILA**：[https://github.com/openvla/vila](https://github.com/openvla/vila)
- **CoPa**：[https://github.com/huang-hx/CoPa](https://github.com/huang-hx/CoPa)
- **CLIPort**：[https://cliport.github.io/](https://cliport.github.io/)
- **PerAct**：[https://peract.github.io/](https://peract.github.io/)
- **CLIP**：[https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- **LLaVA**：[https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **MAE**：[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)
- **Ego4D**：[https://ego4d-data.org/](https://ego4d-data.org/)
- **SIMPLER**：[https://simpler-env.github.io/](https://simpler-env.github.io/)
- **MResT**：[https://arxiv.org/abs/2401.14502](https://arxiv.org/abs/2401.14502)
- **DriveDreamer**：[https://drivedreamer.github.io/](https://drivedreamer.github.io/)
- **Genie**：[https://sites.google.com/view/genie-2024](https://sites.google.com/view/genie-2024)
- **ACT (ALOHA)**：[https://tonyzhaozh.github.io/aloha/](https://tonyzhaozh.github.io/aloha/)
- **Diffusion Policy**：[https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)
- **Karpathy - Jetson nano + world models talk**：[https://www.youtube.com/watch?v=VMj6DX5h7tQ](https://www.youtube.com/watch?v=VMj6DX5h7tQ)（你最近关于 world model 是 "predict next state in latent space" 的讨论，PIVOT-R 是这个 idea 在 manipulation 的一个 concrete instantiation）

---

## 13. 一句话总结 Intuition

PIVOT-R 把 manipulation trajectory 看作 **"primitive action → waypoint" 的 hierarchical chunking**，在 CLIP feature space 做 latent prediction（避开 pixel 重建开销），再用 asynchronous multi-rate execution 让小模型绕开 VLM 的速度 bottleneck——本质上是把 I-JEPA 的 latent SSL 思想 + multi-rate control theory + VLM-as-planner 三个 idea 在 robot manipulation 上做了一个 elegant 的 engineering integration。
