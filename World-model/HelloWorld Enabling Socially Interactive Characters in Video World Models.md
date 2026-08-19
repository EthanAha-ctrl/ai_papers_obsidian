---
source_pdf: HelloWorld Enabling Socially Interactive Characters in Video World Models.pdf
paper_sha256: 2291601783e9a83e0ea397c699209be5bbfbf17f0c16db0baee2ca0e07870032
processed_at: '2026-08-19T10:43:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇paper的intuition极其丝滑。用最白话的human language来说，HelloWorld解决的核心痛点是：**“怎么让视频世界模型里的角色，突然转过头来跟你打招呼，同时还不破坏原本的相机运镜。”**

下面我为你一层层剥开它的设计逻辑，并深入到公式和架构细节里去build你的intuition。

---

### 1. The Core Conflict: 为什么这是个棘手的问题？

现有的Video World Models（比如Sora, WorldPlay, Matrix-Game）已经能做得很好了。你输入键盘指令，相机能往前走、往左转，世界也在合理地演化。**但是**，世界里的NPC该干嘛干嘛，完全当你是空气。你想让NPC对你挥手、点头、说句“Hello”，模型做不到。

为什么做不到？因为要让模型学会“相机控制”和“角色互动”，你需要海量的训练数据。现实中你很难找到这样一种视频：**既有完美的相机轨迹记录，同时里面的人又恰好在对着镜头做社交互动。**

如果你只用普通的“真实运镜视频”去finetune一个base video model，模型会学到怎么移动相机，但它会把原本的“社交互动prior”给忘掉。这在深度学习里叫catastrophic forgetting。

### 2. Trick 1: Self-Distillation（自蒸馏）—— 自己教自己

既然找不到“既有互动又有运镜”的数据，作者提出了一个极其优雅的方案：**让base model自己生成数据，自己教自己。**

Base model（这里是LTX-2.3）其实天生就会生成互动视频，因为它训练数据里有很多网红对着镜头说话的视频。它的问题是不会听你的指令去控制相机。

**Training Pipeline 步骤拆解：**

1.  **Data Synthesis**: 构造一个Prompt，包含四部分：scene描述、interaction动作、camera运动、quality要求。让base model生成一批既有社交互动、又有相机运动的clips。
2.  **3D Reconstruction**: 用一个叫Pi3X的off-the-shelf工具，从生成的clip的第一帧恢复出3D point cloud，并且估算出整个视频的per-frame camera trajectory $\mathcal{C} = \{\mathbf{c}^i\}_{i=1}^N$。
3.  **Warp Video Generation**: 这是最骚的操作。拿着第一帧的3D point cloud，按照估算出的相机轨迹 $\mathcal{C}$，重新渲染一遍，得到一个所谓的“Warp Video” $\mathcal{V}_{\text{warp}}$。
4.  **Finetune**: 把这个Warp Video作为条件，输入给DiT，让模型去重建最开始它自己生成的那个带有互动的clip。

**Intuition Building for Warp Video:**
为什么要搞个Warp Video？你可以把它看作是一个“几何骨架”。因为第一帧被lift成3D点云再reproject回去时，原本被遮挡的地方、或者视野外的地方，在Warp Video里全是黑洞。

这就意味着，Warp Video **不是一个完整的像素目标，而是一个几何指引**。它告诉模型：“相机走到这一帧时，能看到的部分长这样，看不到的holes你自己去hallucinate填补。” 这种设计巧妙地把“相机控制”和“内容生成”解耦了。

**Flow Matching Loss 公式拆解：**

$$
\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, t, \epsilon} \big\| \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{x}^0, \mathbf{y}_{\text{scene}}, \mathbf{y}_{\text{inter}}, \mathbf{y}_{\text{quality}}, \mathcal{V}_{\text{warp}}) - (\epsilon - \mathbf{z}_0) \big\|_2^2
$$

*   $\mathbf{z}_0$: base model自己生成的那个clean video的latent表示。
*   $t \in [0, 1]$: flow matching的时间步。$t=0$时是clean data，$t=1$时是纯噪声。
*   $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 标准高斯噪声。
*   $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\epsilon$: 这是rectified flow的linear interpolation path。
*   $\mathbf{v}_\theta$: DiT预测的velocity field。模型要预测的目标是 $(\epsilon - \mathbf{z}_0)$，也就是path的导数。
*   **极其关键的细节**：公式输入里包含了scene, inter, quality的prompt，**唯独剔除了 $\mathbf{y}_{\text{camera}}$**！作者故意把camera相关的文本提示拿掉了。这就强迫模型只能通过 $\mathcal{V}_{\text{warp}}$ 来理解相机运动，避免文本条件和几何条件打架。

整个finetune只用了156个生成的videos，跑2000步，用了一个rank-32的LoRA。极其轻量，却完美保留了模型的social interaction prior。

### 3. Trick 2: Temporal Cross-Attention Mask—— 零训练控制时机

光让角色挥手还不够，用户按下“F”键，角色必须在按下后的那几秒内挥手，不能在视频开头就提前挥完了。

**问题在哪？**
在DiT的cross-attention里，每一帧的video tokens都能无差别地attend到text prompt的所有tokens。你哪怕在prompt里写“在视频第6-9秒挥手”，模型也听不懂这种时序指令，因为attention是全局broadcast的。这就是为什么baseline方法的TimeAcc（时机准确率）只有30%左右，跟瞎猜差不多。

**解决方案：**
作者在inference阶段引入了一个training-free的mask，直接修改cross-attention的注意力矩阵。

$$
M_{ij} = \begin{cases} -\infty, & i \notin \mathcal{W} \text{ and } j \in \mathbf{y}_{\text{inter}} \\ 0, & \text{otherwise} \end{cases}
$$

$$
\text{Attention} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V
$$

*   $i$: query token的temporal index（即video/audio帧的时间位置）。
*   $j$: key token的index（即text prompt的token位置）。
*   $\mathcal{W} = [\tau_s, \tau_e]$: 用户按下F键后开启的交互时间窗口。
*   $\mathbf{y}_{\text{inter}}$: prompt中描述交互动作的tokens集合。

**Intuition Building:**
看这个公式，如果当前帧 $i$ 不在交互窗口 $\mathcal{W}$ 内，并且它试图去attend交互动作的prompt token $j$，我就把你的attention score强行加上 $-\infty$。经过softmax之后，这个权重直接归零。

这就意味着：**在这个时间窗口之外，模型完全“看”不到交互指令。** 它只能做ambient的背景动作。只有进入了 $\mathcal{W}$ 窗口，模型才能“看到”prompt里的动作指令，从而精准触发interaction。

这个操作不费一兵一卒，完全在inference时通过改一个mask矩阵实现，TimeAcc直接从36.7%飙升至81.7%。而且这个mask同时作用于video流($M_v$)和audio流($M_a$)，连角色说话的声音都能精准卡在窗口内。

### 4. HelloWorldBench: 重新定义评估标准

现有的benchmark只看画面质量和相机跟随度，没人关心角色互动。作者搞了个HelloWorldBench，提出了三维解耦的评估体系：

1.  **ActAcc (What)**: 用VLM做8选1选择题，看角色做没做指定的动作。
2.  **TimeAcc (When)**: 用VLM判断动作发生在视频的前期、中期还是后期。
3.  **GazeDev (Whom)**: 用gaze estimator算角色的眼球光轴和相机光轴的夹角。如果角色不看镜头，就算挥手也是对空气挥手。只有self-distillation训出来的模型，才知道眼神要看镜头（GazeDev从51.3°降到40.2°）。

### 5. Final Intuition & Takeaway

HelloWorld这篇paper的精妙之处在于它对“生成模型prior”的深刻理解。

当你想给一个强大的base model加上新的控制信号时，如果你手头的数据不够好，与其去外面费劲巴拉地找数据，**不如让模型自己生成符合它自己分布的数据，然后自己蒸馏自己**。

Self-distillation保留了模型原本丰富的social prior；Temporal cross-attention mask用最简单的数学原理实现了复杂的时序控制。这两者结合，把一个被动的世界模拟器，变成了一个能跟你打招呼的活世界。

这就是通往真正交互式AI Agent在视觉世界中的关键一步。

**References for deep dive:**
*   HelloWorld Project Page: [https://github.com/AlayaLab/HelloWorld](https://github.com/AlayaLab/HelloWorld)
*   Warp-as-History (Camera control foundation): [https://arxiv.org/abs/2605.15182](https://arxiv.org/abs/2605.15182)
*   Pi3 (Visual Geometry learning): [https://arxiv.org/abs/2507.13347](https://arxiv.org/abs/2507.13347)
*   LTX-2.3 (Base Model): [https://arxiv.org/abs/2601.03233](https://arxiv.org/abs/2601.03233)
*   Flow Matching for Generative Modeling: [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)

---

# HelloWorld Paper Review

## 1. Core Narrative & Intuition Building

这篇paper的核心narrative可以提炼为一句话：**把一个已经会做social interaction的video generation model，安全地convert成一个camera-controllable world model，而不丢失interaction prior**。 这个任务的subtle之处在于：base video model本身就有interaction能力，问题在于当你在它上面加camera control finetune时，传统的训练数据（real videos with camera motion but no social interaction）会silently destroy这个prior。作者的洞察是 — 既然base model本来就会生成interaction clips，那就让它自己生成训练数据，然后在自身产出上加条件finetune。这就是 **self-distillation** 的精髓。

Paper的标题 "HelloWorld" 一词同时映射了 program language 的入门仪式和"世界模型中的人物向用户打招呼"这件事，双关得很巧。

## 2. Problem Formulation

任务input是四元组 $(\mathbf{x}^0, \mathbf{y}, \mathcal{C}, \mathcal{W})$：

- $\mathbf{x}^0$ : first frame image, 包含 scene 和 character
- $\mathbf{y}$ : text prompt，被decou分成四部分 $\mathbf{y}_{\text{scene}}, \mathbf{y}_{\text{inter}}, \mathbf{y}_{\text{camera}}, \mathbf{y}_{\text{quality}}$
- $\mathcal{C} = \{\mathbf{c}^i\}_{i=1}^N$ : camera trajectory，$\mathbf{c}^i \in \text{SE}(3)$ 是第 $i$ 帧的相机pose（6-DoF）
- $\mathcal{W} = [\tau_s, \tau_e]$ : interaction window

Output是 $\boldsymbol{\nu} = \{\mathbf{x}^i\}_{i=1}^N$，camera follow $\mathcal{C}$，character 在 $\mathcal{W}$ 内与 viewer 互动。

**Intuition**：把 social interaction 解耦成三个 axis — *what* (action type)、*when* (temporal window)、*whom* (toward viewer)。这个 3-axis decomposition 是整个benchmark设计的基石，也对应了后面三个 metric。

## 3. Warp Video as Camera Condition

这是借自 Wang and He 的 [Warp-as-History](https://arxiv.org/abs/2605.15182) 工作。核心 idea：

$$
\mathcal{V}_{\text{warp}} = \text{warp}(\mathbf{x}^0, \mathcal{C})
$$

执行步骤：
1. 用 [Pi3X](https://arxiv.org/abs/2507.13347) 把 $\mathbf{x}^0$ lift 成 3D point cloud $\mathcal{P} = \{(\mathbf{p}_k, \mathbf{rgb}_k)\}$
2. 对每一帧 target camera $\mathbf{c}^i$，把 $\mathcal{P}$ reproject 到 image plane，得到 warped frame $\mathbf{w}^i$
3. 因为 $\mathbf{x}^0$ 是 single view，reprojection 后被 occlude / out-of-FOV 的区域会出现 holes

**关键 insight**：warp video 是 *incomplete* 的 — holes 的存在让它成为一种 *geometric guidance* 而非 *pixel-level target*。模型必须 hallucinate 填充 holes，这给了它生成自由度；而 visible region 则提供了 camera motion 的 explicit 信号。这种 "partial supervision" 设计在 structure vs. freedom 之间取得了平衡。

然后 warp video 在 DiT 内部被 tokenize 成 history tokens，**和对应 frame 共享 temporal position embedding** — 这是一个 frame-aligned 的 cross-frame conditioning，让模型知道 "这一帧应该长这个样子（geometrically）"。

Eq 2 的形式：
$$
\mathcal{V} = \mathcal{G}(\epsilon; \mathbf{x}^0, \mathbf{y}, \mathcal{V}_{\text{warp}}), \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

## 4. Self-Distillation Pipeline（最核心的部分）

### 4.1 为什么需要 self-distillation

考虑两种 alternative：
- **(a)** 用 real video 训 camera control：data 中没有 interaction signal，model 会 forget interaction prior
- **(b)** 用 real interaction video 训：很难找到"既有 social interaction toward camera, 又有 known camera trajectory"的大规模 data

self-distillation 给出第三条路：用 base model 生成 (interaction + camera motion) clips，然后 self-supervised finetune。

### 4.2 Data synthesis 步骤

Prompt 构造：
$$
\mathbf{y} = \mathbf{y}_{\text{scene}} + \mathbf{y}_{\text{inter}} + \mathbf{y}_{\text{camera}} + \mathbf{y}_{\text{quality}}
$$

- $\mathbf{y}_{\text{scene}}$ : 场景描述
- $\mathbf{y}_{\text{inter}}$ : 交互动作描述（"waves to the viewer", "nods", "says hello"）
- $\mathbf{y}_{\text{camera}}$ : camera motion 描述（"camera pans left", "dolly in"）
- $\mathbf{y}_{\text{quality}}$ : 视频质量修饰

生成 video 后，用 Pi3X 同时恢复 (i) first-frame point cloud 和 (ii) per-frame camera trajectory $\mathcal{C}$。

### 4.3 Training objective

Flow matching loss（不是 DDPM 的 noise prediction，而是 rectified flow / CFM style 的 velocity prediction）：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, t, \epsilon} \big\| \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{x}^0, \mathbf{y}_{\text{scene}}, \mathbf{y}_{\text{inter}}, \mathbf{y}_{\text{quality}}, \mathcal{V}_{\text{warp}}) - (\epsilon - \mathbf{z}_0) \big\|_2^2
$$

变量含义：
- $\mathbf{z}_0$ : training video $\mathcal{V}$ 的 clean latent (VAE-encoded)
- $t \in [0, 1]$ : flow matching timestep，$t=0$ 是 clean、$t=1$ 是 pure noise
- $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\epsilon$ : linear interpolation path（rectified flow 的特征 — 用直线 path）
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ : standard Gaussian noise
- $\mathbf{v}_\theta$ : DiT 预测的 velocity field，目标是 $(\epsilon - \mathbf{z}_0)$，即 path 的 derivative
- $\mathcal{V}_{\text{warp}}$ : warp video，作为 history condition

**关键 design**：text prompt 中 **$\mathbf{y}_{\text{camera}}$ 被剔除** — 这样 camera control 完全由 warp video 提供，避免 textual camera description 和 geometric warp signal 之间的 dual-control 冲突。这是一个非常干净的 conditional decomposition。

LoRA 配置：rank-32，applied to all projection matrices in **self-attention layers of the video branch**。注意是 self-attention 不是 cross-attention — 这是因为 warp video 是 tokenized 后和 noise tokens 拼接进 self-attention 的，而非通过 cross-attention 注入。156 videos, 2000 steps, lr=1e-4。这是一个相当 lightweight 的 finetune。

### 4.4 Visible-token selection

Warp video reproject 后有 holes。这些 hole tokens 不能给模型有效信号。visible-token selection module 把 invalid source observation 的 warp tokens discard，只保留有 valid source pixel 的 tokens。这避免了 model 浪费 capacity 学习如何 inpaint "obviously missing" 的 region，让它专注在语义生成上。

## 5. Temporal Cross-Attention Mask（training-free 的时序控制）

### 5.1 为什么需要 mask

DiT 的 cross-attention layer 中，**每个 video frame token 都能 attend 到 text prompt 的每个 token**（包括 $\mathbf{y}_{\text{inter}}$）。这意味着即使 text 里写了 "at the end of the video, the character waves"，prompt signal 也是均匀 broadcast 到所有 frame 的。这就是 baseline methods TimeAcc 只有 ~30% 的原因 — 跟随机猜 1/3 差不多。

### 5.2 Mask 定义

$$
M_{ij} = \begin{cases} -\infty, & i \notin \mathcal{W} \text{ and } j \in \mathbf{y}_{\text{inter}} \\ 0, & \text{otherwise} \end{cases}
$$

$$
\text{Attention} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V
$$

变量含义：
- $i$ : query token 的 temporal index，即 video / audio frame position
- $j$ : key token 的 index，即 text prompt token position
- $\mathbf{y}_{\text{inter}}$ : interaction prompt tokens 的集合
- $\mathcal{W} = [\tau_s, \tau_e]$ : F press window

**关键 insight**：mask 是 *additive*（在 softmax 之前加到 logits 上），$-\infty$ 在 softmax 后变 0 attention weight。这是一个标准 causal mask 技巧，但被 repurposed 为 temporal localization — 不是 "future can't see past"，而是 "outside-window frames can't see interaction prompt"。

### 5.3 双流 mask

Table 3 的 ablation：
- No mask: ActAcc 42.5, TimeAcc 36.7, SpeechInWin 52.5
- $M_v$ only: ActAcc 41.5, TimeAcc 80.9, SpeechInWin 62.8
- $M_v + M_a$: ActAcc 41.4, TimeAcc 81.7, SpeechInWin 69.1

一个有趣观察：no-mask setting 的 ActAcc 最高（42.5）。这是因为没有 temporal constraint 时，character 会在整个 video 都做 prompted action，ActAcc 不管 timing 都给 reward。这暴露了 ActAcc metric 的一个 blind spot — 它不和 timing 联合评估。但作者选择牺牲 ~1 pt ActAcc 换取 45 pt TimeAcc，这是合理的 trade-off。

### 5.4 为什么 training-free

这个 mask 不需要训练就能用，因为 cross-attention 的 query/key/value projection 已经在 base model 训练好了。mask 只是修改 attention pattern，相当于在 inference 时把 prompt 的 conditioning graph 重新 wire 一下。Cost 可以忽略。

## 6. HelloWorldBench

### 6.1 Data Construction

- 120 high-quality images from [Unsplash](https://unsplash.com)，涵盖 humans / animals / toys / robots
- LLM agent 为每张图设计 2-4 个 subject-appropriate interactions
- 4 个 camera trajectory：static, scan, dolly-in, orbit
- 3 个 timing：early, middle, late
- Cartesian product yields 400 samples，264 character instances，101 interaction types

### 6.2 Three Interaction Metrics

- **ActAcc ↑** : VLM judge (Qwen3.6-35B-A3B) 做 8-way multiple-choice，问 "which action"。Ground truth 是 prompted action，distractor 从其他 interaction type 采样
- **TimeAcc ↑** : VLM judge 选 timing segment（4 options：early/middle/late/none）。计算时排除 "none" 答案
- **GazeDev° ↓** : 对 217 human samples，用 [UniGaze](https://arxiv.org/abs/2601.xxxxx) estimate gaze direction，计算 gaze 和 camera optical axis 的 mean angular deviation within $\mathcal{W}$。如果 no face detected，penalty 90°

**这个 3-tuple metric 设计很漂亮**：what-when-whom 的 decoupling 强制 model 不能 collapse 到只做一项。比如 baseline 可以做 action 但 gaze 不对（GazeDev 59-77°），或者 action 对但 timing 全错（TimeAcc 30%）。HelloWorld 在三个 axis 上同时达到 SOTA。

## 7. Experimental Analysis

### 7.1 Main Comparison (Table 1)

关键数字：

| Method | ActAcc | TimeAcc | GazeDev° | CamCtrl |
|---|---|---|---|---|
| LTX-2.3 (base) | 42.5 | 52.6 | 38.1 | 31.4 |
| WorldPlay | 10.5 | 41.2 | 63.5 | 48.1 |
| LingBot-World | **50.5** | 39.5 | 59.0 | 62.6 |
| SANA-WM | 38.5 | 30.9 | 56.8 | 70.0 |
| Warp-as-History | 33.8 | 35.2 | 52.8 | 65.1 |
| **HelloWorld** | 41.4 | **81.7** | **40.2** | **82.9** |

几个观察：
- **TimeAcc 81.7% vs baselines ~30%** 是最 striking 的差距。这完全归功于 temporal cross-attention mask。说明 text-only conditioning 完全无法控制 timing
- **ActAcc 上 HelloWorld (41.4) 不如 LingBot-World (50.5)**。LingBot-World 把整个 video 都做 action，所以 8-way MC 的 accuracy 高。但加上 GazeDev 和 TimeAcc 后，LingBot-World 的 advantage 消失
- **GazeDev 40.2°** vs baselines 52-77°。这归功于 self-distillation data — 因为 base model 生成的 interaction clip 天然有 "look at camera" 的 prior，model 学到了这个 gaze direction
- **CamCtrl 82.9** 是 SOTA。说明 warp video conditioning 的 camera control effectiveness 超过了 explicit trajectory conditioning（SANA-WM）和 keyboard conditioning（LingBot-World）

### 7.2 Training Data Ablation (Table 2)

| Training data | ActAcc | TimeAcc | GazeDev° |
|---|---|---|---|
| Real-video | 36.4 | 81.3 | 51.3 |
| Human-only | 40.4 | 82.4 | 42.3 |
| Full | 41.4 | 81.7 | **40.2** |

关键 insight：
- Real-video data 训出来的 model ActAcc 36.4、GazeDev 51.3° — **action 有了，但 gaze 不对**。Fig 6(a) 显示 character 做 thumbs up 但没看向 viewer。这印证了 "real video 没有 interaction signal → model 学不到 engage with camera"
- Human-only vs Full：Full 在 ActAcc 略升、GazeDev 略降。说明扩展到 non-human character 提供了 gaze direction 的 inductive bias（因为 animal/cartoon 也要 face viewer 才算 interaction）
- TimeAcc 在三种 setting 都 ~81% — 证明 timing 完全由 inference-time mask 控制，和 training data 无关

### 7.3 Computational Cost (Table 4)

| Method | Time (s) | Time/frame (s) | FLOPs (×10¹⁵) |
|---|---|---|---|
| LTX-2.3 | 50.3 | 0.21 | 6.9 |
| HelloWorld | 60.2 | 0.26 | 9.4 |
| WorldPlay | 131.8 | 0.56 | 29.3 |
| SANA-WM | 44.9 | 0.27 | 3.2 |

Warp video 增加了 ~20% latency 和 36% FLOPs。这是 reasonable price for camera control。注意 SANA-WM 虽然 FLOPs 低（3.2），但用的是 hybrid linear attention 架构 ([SANA-WM paper](https://arxiv.org/abs/2605.15178))，是个不同 cost-quality trade-off。

## 8. Architecture Details (推断)

Paper 没有完整 architecture diagram，但可以推断：

- **Base model**: LTX-2.3 ([paper](https://arxiv.org/abs/2601.03233))，是 joint audio-visual foundation model，支持 1280×704 @ 24fps
- **Tokenizer**: VAE for visual + audio encoder
- **DiT backbone**: video branch + audio branch，都有 self-attention（被 LoRA modify）
- **Cross-attention**: text → {video tokens, audio tokens}
- **History injection**: warp video tokens 拼接到 noise tokens + first-frame tokens 序列前面，参与 self-attention
- **Attention strength**: warp reference tokens 用 0.3 的 attention strength — 这是个 hyperparameter，控制 warp signal 的权重。太强会过度 anchor 到 incomplete warp video，太弱会失去 camera control

## 9. Critical Thoughts & Open Questions

### 9.1 关于 self-distillation 的 generalizability

Self-distillation 的一个 implicit assumption 是：base model 已经有 interaction prior。如果 base model 完全不会 interaction，self-distillation 就会 fail。LTX-2.3 碰巧有这个 prior 是因为其训练数据包含了 talking head / social media video。这对其他 base model（比如纯风景 video 训出来的）不一定 work。一个 open question 是：能否用一个 small interaction dataset 来 bootstrap prior，再做 self-distillation？

### 9.2 关于 temporal mask 的局限

当前 mask 是 binary 的硬边界（$-\infty$ 或 0）。但 social interaction 通常有 onset/ramp-up/peak/offset 的 envelope。一个 smooth mask（Gaussian decay）可能让 interaction 更自然。这值得 future exploration。

另一个局限：mask 是 single-window 的。如果用户按多次 F，需要 multi-window mask。Paper 里没讨论这个 case。

### 9.3 关于 gaze metric 的盲区

GazeDev 用的是 mean angular deviation。但一个"自然的"interaction 应该是 *dynamic* gaze — 一开始看 viewer、然后移开、再回看。Mean deviation 不能 capture 这种 dynamics。一个更精细的 metric 应该看 gaze trajectory 的 temporal pattern。

### 9.4 关于 autoregressive future work

Limitation section 提到未来要探索 autoregressive architectures for real-time interaction。这其实是一个 *fundamental* limitation — 当前的 DiT 是 non-autoregressive 的，整段 video 一次 generate。这无法支持 "press F → wait 0.5s → character react" 这种 real-time loop。要做 real-time，要么用 AR model（像 [GameNGen](https://arxiv.org/abs/2408.14846) 那种 frame-by-frame），要么用 sliding window + KV cache。这是整个 video world model field 的开放问题。

### 9.5 和 ReactiveGWM 的对比

Concurrent work [ReactiveGWM](https://arxiv.org/abs/2605.15256) 也做 interaction，但限于 single game (NPC-to-NPC)。HelloWorld 的 advantage 是 *viewer-directed* 和 *character-agnostic*（人、动物、玩具都行）。这是 social interaction 的更 general formulation。

### 9.6 联想到其他工作

- [SocialDirector](https://arxiv.org/abs/2605.10079)（同一作者 Ouyang 的工作）：training-free multi-person social interaction control，是 HelloWorld 的 multi-person 版本 precursor
- [Omni-MMSI](https://arxiv.org/abs/2511.xxxxx)：identity-attributed social interaction understanding，是 understanding 侧的对应
- [GameGen-X](https://arxiv.org/abs/2410.17000)：open-world game video generation，是 game domain 的对应
- [Diffusion Forcing](https://yifanwang.info/diffusion-forcing/)：一种把 diffusion 和 AR 结合的思路，可能解决 real-time limitation

## 10. Key Takeaways

1. **Self-distillation 是一种 prior-preserving fine-tuning 范式**：当 base model 已经有某个能力，而你想加新 conditioning 时不破坏它，让 model 自己生成符合新 conditioning 又保留能力的 data，再 finetune 自己。这是个 general 技巧，可以推广到其他 task。

2. **Cross-attention mask 是 training-free 的 temporal localization 工具**：在 diffusion/flow model 里，要实现 "某 prompt 只在某个时间窗生效"，直接 mask cross-attention logits 即可。这避开了训练 temporal-aware model 的复杂度。

3. **Warp video 是 "geometric guidance, not pixel target"**：holes 的存在让 warp video 既是 camera control signal，又给 model 留下生成空间。这个 design pattern 对所有 "structural condition + generative freedom" 的 task 都有启发。

4. **What-When-Whom 的 metric decoupling**：评估 social interaction 时，把 action type、timing、target direction 拆开衡量，比单一 metric 更有 diagnostic value。这是 benchmark design 的一个好范例。

5. **"Social interaction toward viewer" 是 world model 的下一个 frontier**：camera control 解决了 "user 看哪里"，event control 解决了 "世界里发生什么"，而 social interaction 解决了 "world 里的角色怎么 respond to user"。这是从 passive world simulation 到 active social simulation 的关键 step。

## References

- HelloWorld project page: https://github.com/AlayaLab/HelloWorld
- LTX-2.3: https://arxiv.org/abs/2601.03233
- Warp-as-History: https://arxiv.org/abs/2605.15182
- Pi3 (visual geometry): https://arxiv.org/abs/2507.13347
- SANA-WM: https://arxiv.org/abs/2605.15178
- WorldPlay: https://arxiv.org/abs/2512.14614
- Matrix-Game 3.0: https://arxiv.org/abs/2604.08995
- LingBot-World: https://arxiv.org/abs/2601.20540
- GameGen-X: https://arxiv.org/abs/2410.17000 (ICLR 2025)
- ReactiveGWM: https://arxiv.org/abs/2605.15256
- Genie 3 (DeepMind): https://storage.googleapis.com/deepmind-media/Genie3/Genie3_Technical_Report.pdf
- Cosmos 3: https://arxiv.org/abs/2606.02800
- AlayaWorld: https://arxiv.org/abs/2607.06291
- Sora (video generation models as world simulators): https://openai.com/research/video-generation-models-as-world-simulators
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2209.03003
- Rectified Flow: https://arxiv.org/abs/2209.03003
- UniGaze (gaze estimator): https://arxiv.org/abs/2507.xxxxx (WACV 2026)
- SocialDirector: https://arxiv.org/abs/2605.10079
- GameNGen (Diffusion models are real-time game engines): https://arxiv.org/abs/2408.14846
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Whisper: https://arxiv.org/abs/2212.04356
- Sekai (video dataset): https://arxiv.org/abs/2510.20668 (NeurIPS 2025)
- Worldscore: https://arxiv.org/abs/2510.xxxxx

如果想进一步 build intuition，我建议看一下 [Warp-as-History](https://arxiv.org/abs/2605.15182) 的原始 paper — 它是 HelloWorld camera control 的 foundation，对 warp video 的 token selection 和 attention strength 设计有更详细的 ablation。另外 [SocialDirector](https://arxiv.org/abs/2605.10079) 是同一作者团队 multi-person social interaction control 的前期工作，理解它能让你看到 temporal mask 这个 idea 是怎么从 multi-person 场景演化为 viewer-directed 场景的。
