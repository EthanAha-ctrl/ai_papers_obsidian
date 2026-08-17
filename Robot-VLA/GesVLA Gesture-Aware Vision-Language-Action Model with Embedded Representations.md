---
source_pdf: GesVLA Gesture-Aware Vision-Language-Action Model with Embedded Representations.pdf
paper_sha256: 25a63557b717c7bbae92d99792cfa66dbedc76bbde3a09a6bb66b639b549976d
processed_at: '2026-08-04T21:38:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GesVLA 用人话讲

Andrej，我换一种方式讲，把这篇 paper 的"灵魂"拎出来，不堆术语，但保留关键 technical 细节。

Project page: https://gwxuan.github.io/GesVLA/

## 一句话概括

**机器人听不懂"那个"是哪个，但人指一下就明白了——这篇 paper 把"手指一下"这个动作做成 robot 的一种新输入，跟语言、视觉平起平坐。**

## 为什么要做这件事

你想象一个场景：桌上摆了 7 个差不多颜色的 block，你对机器人说 "pick this up and put it there"。机器人疯了——哪个 this？哪个 there？

语言天然是 ambiguous 的。人类之间遇到这种情况怎么办？**指一下**。手指 past 几个 block 然后停在目标上，对方立刻懂。

但现有 VLA (RT-2 [1], OpenVLA [2], π0 [5], π0.5 [3], GR00T N1 [14], Hi Robot [15]) 都只接受 language input，逼用户把 "这个" 翻译成 "the leftmost red block near the plate" 这种啰嗦描述。Real-time HRI 里这根本不可行。

GesVLA 的核心 insight: **gesture 不应该被翻译成文字，应该直接进入模型的 latent space 参与 reasoning**。一旦你把手指方向 discretize 成 "the user is pointing at the red block" 这种 text，你就丢失了 pointing 的精确 spatial geometry——手指到底指向 3D 空间哪个点，离目标多远，方向向量是什么。

参考:
- π0 flow matching VLA: https://arxiv.org/abs/2410.24164
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- PaliGemma: https://arxiv.org/abs/2407.07726

## 整体架构直觉

想象你有两个 brain 在 robot 里：

**Brain 1 (VLM_int) — "理解官"**：只负责看手势 + 听语言，搞清楚用户要哪个东西。这个 brain 算得慢，但一次任务只需要算一次——因为手势和语言在任务开始时就给定了，过程中不会变。

**Brain 2 (VLM_per + Action Expert) — "执行官"**：每一步都要看当前 camera 画面 + 当前 robot 状态，然后输出 action chunk。这个 brain 算得快，但每一步都得算。

两个 brain 之间通过 cross-attention 通信——Brain 2 可以"询问"Brain 1 的内部状态 (KV cache)，但 Brain 1 不关心 Brain 2 在干嘛。这种单向流动让 Brain 1 的计算结果可以缓存复用，效率高。

用公式 (5) 表示：
$$
\mathrm{VLM}_{\mathrm{int}} \to \mathrm{VLM}_{\mathrm{per}} \to \mathrm{Action\ Expert}
$$

Compute ratio 是 **1 : T : T×N**：
- Brain 1 算 1 次
- Brain 2 算 T 次 (T = control steps)
- Action expert 算 T×N 次 (N = flow matching denoise steps)

如果 T = 200, N = 10, Brain 1 算 1 次而 Brain 2+Action 算 2000 次。这个 cache reuse 思想跟 Hi Robot [15] 的 hierarchical VLA、Dual Process VLA [13] 的 System 1/System 2 split 都是同一脉络。

## Gesture 怎么编码——这是 paper 最聪明的地方

最 naive 的做法是把 gesture video 当普通 video 喂给 VLM。GesVLA 拒绝这么做，原因有两个：

### 1. Keyframe selection — 只保留"停顿"帧

人指东西有个特征：**手指到目标时会短暂停顿**。GesVemma 跟踪 hand keypoints 的时间序列，只选 motion stagnation 的帧作为 keyframe。

公式 (1):
$$
\mathbf{z}_i^g = \phi(\mathrm{Pose}(g_i)), \quad g_i \in S(\mathrm{Pose}(\mathcal{G}))
$$

变量解释：
- $\mathcal{G}$: 完整 gesture video
- $\mathrm{Pose}(\cdot)$: MediaPipe Hands [8] 做手部 keypoint 检测 (https://arxiv.org/abs/2006.10214)
- $S(\cdot)$: 基于 motion dynamics 的 keyframe 选择函数
- $g_i$: 第 $i$ 个 keyframe, $i \in \{1, \dots, F\}$, $F$ 是 keyframe 数量
- $\phi(\cdot)$: 多层 MLP 组成的 gesture encoder
- $\mathbf{z}_i^g$: 第 $i$ 帧 gesture 在 latent space 的 embedding

把 video 降成几张静态指向图，过滤掉大量冗余 temporal 信息。

### 2. 12-dim keypoint vector — 极简 inductive bias

每个 keyframe 只提取 4 个 keypoints：wrist + index finger 的 3 个 joints (MCP, PIP, DIP)。每个 keypoint 3 个值 $(x, y, d)$——图像坐标 + depth。4 × 3 = **12 维向量**。

这个 12 维 vector 经过 MLP 投影到 PaliGemma 的 hidden dim (~2048 维)，相当于把 gesture 压缩成一个 **explicit pointing-direction 向量**。

为什么不用 raw RGB hand crop？为什么不用 21 个完整 keypoints？

Ablation Table IIIb 显示 w/o gesture MLP (即用 raw image 不用 keypoint vector) 跌到 84.1% (-10.2 pp)。原因：raw image 只能 partial 恢复 spatial cue，explicit keypoint vector 把 pointing direction 几何结构直接编码进去，模型不需要从 pixel 学这个 mapping。

这个 12 维是个 hard inductive bias——好处是 sample efficient, bad 处是只能处理 index-finger pointing, 处理不了 open-hand palm pointing 或 head pointing。

## Semi-synthetic Data Engine — 解决"没数据"的杀手锏

Gesture + robot action 数据根本不存在。Real-world 收集贵且 annotation 模糊 (pointing target 到底是哪个？)。GesVLA 的解法很 hack：

**用 GroundingDINO [25] 在真实 RGB-D 场景里检测 object → 随机选一个作为 target → 用 depth + camera intrinsics 反投影到 3D 空间 → 用 hand mesh 渲染一只手指向那个 3D 点 → 输出带精确 annotation 的 gesture video**。

公式 (6) 是反投影：
$$
\mathbf{p} = \left(\frac{(u - c_x) z}{f_x}, \frac{(v - c_y) z}{f_y}, z\right)
$$

变量：
- $(u, v)$: 图像坐标 (object center + random jitter)
- $z$: 该位置 depth 值
- $(f_x, f_y)$: camera focal length (x 和 y 方向)
- $(c_x, c_y)$: camera principal point (光心在 pixel 坐标的偏移)
- $\mathbf{p}$: 反投影得到的 3D target point

**关键 trick 1: coordinate jitter**。不直接用 object bbox center，加 random offset 模拟真人 pointing 偏差。Ablation Table IIIa 显示 w/o jitter 跌到 42.0% (-52.3 pp)！原因巨有意思：如果 training data 里手指永远指向 bbox center，model 就 overfit 到 fixed coordinate bins，sim-to-real 时 bin 偏一点直接 fail。Jitter 强制 model 学"pointing direction 的几何含义"而非"memorize 坐标"。

**关键 trick 2: parabolic hand lifting**。多 target 之间手指要移动，公式 (8) 加一个抛物线 lift：
$$
\mathbf{p}_{\mathrm{vis}} = \mathbf{p}_h + h_{\max}(1 - (2\alpha - 1)^2)\mathbf{n}_{\mathrm{up}}
$$

变量：
- $\mathbf{p}_{\mathrm{vis}}$: 实际渲染位置
- $\mathbf{p}_h$: base hand position
- $h_{\max}$: 最大抬起高度
- $\alpha \in [0,1]$: 两个 target 之间的 normalized motion progress
- $\mathbf{n}_{\mathrm{up}}$: 向上方向

$\alpha = 0.5$ 时 $1 - (2\alpha-1)^2 = 1$, 抬到 $h_{\max}$ 高度。$\alpha = 0$ 或 $1$ 时为 0。比 linear interpolation 像真人——人指完一个东西会抬手再指下一个。

这个 data engine 一共生成了 16k samples，每个 sample 包含 gesture video + language instruction + 精确 target annotation。

**核心 intuition**: real scene background 保留 realistic lighting / texture / occlusion, synthetic hand 提供 exact annotation, 二者结合 = scalability + low sim-to-real gap 的 sweet spot。

GroundingDINO: https://arxiv.org/abs/2303.05499
Depth Anything V2 (用于 depth): https://arxiv.org/abs/2406.09414

## 两阶段训练 — 解开 optimization conflict

**Stage 1**: 用 16k semi-synthetic data 训练 VLM_int，目标是让它在 unified token space 里同时输出 textual reasoning 和 discretized coordinate tokens。

公式 (9) 是 teacher-forced autoregressive loss：
$$
\mathcal{L}_{\mathrm{int}} = -\frac{1}{M} \sum_{i=1}^{M} \log P(a_i^* \mid a_{<i}^*, \mathcal{G}, \mathcal{T})
$$

变量：
- $M$: sequence 中 token 总数
- $a_i^*$: 第 $i$ 个 target token (包括 text token 和 discretized coordinate token)
- $a_{<i}^*$: 前面已生成 tokens
- $\mathcal{G}, \mathcal{T}$: gesture 和 language 输入

target coordinates 被离散化成 fixed bins, 作为 special tokens 加入词表。这让 VLM_int 在一个 unified token space 里同时学 semantic reasoning ("the red block") 和 spatial grounding (bin id 指代 coordinate)。

**Stage 2**: 用 real robot demo 训练 VLM_per + Action Expert, VLM_int frozen。

公式 (10) Flow matching loss (跟 π0 一样)：
$$
\mathcal{L}_{\mathrm{action}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} \left[ \| \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]
$$

变量：
- $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: source Gaussian noise
- $\mathbf{x}_1$: target action trajectory (来自 demonstration)
- $t \in [0,1]$: flow time
- $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$: linear interpolation
- $\mathbf{c}$: conditioning context (来自 VLM_per 的 KV + robot state $\mathbf{s}_t$)
- $\mathbf{v}_\theta$: 网络预测的 velocity field

训练时 model 学预测从 $\mathbf{x}_0$ 到 $\mathbf{x}_1$ 的 velocity (恒为 $\mathbf{x}_1 - \mathbf{x}_0$), 推理时从 $\mathbf{x}_0$ 出发 N 步 Euler ODE 求解到 $\mathbf{x}_1$。

π0 paper: https://arxiv.org/abs/2410.24164

**为什么要 two-stage？** Ablation Table IV 给出答案：
- Joint training (所有 module 一起训): 45.0%
- Two-stage, VLM_int unfrozen: 80.0%
- Two-stage, VLM_int frozen: 83.3% (最佳)

Joint training 失败原因：16k synthetic reasoning data 的 scale 远大于 real robot demo, gradient 被 reasoning loss 主导, action expert 学不到 valid policy。Two-stage 把 weak supervision (synthetic reasoning) 和 strong supervision (real action) 解耦，避免 optimization conflict。

## 实验结果—重点 build intuition

### Table I: Real-robot manipulation

| Method | Block (S/H/Avg) | Jelly (S/H/Avg) | Fruit/Veg (S/H/Avg) | Total Avg |
|---|---|---|---|---|
| Text-only VLA [5] | 6/10, 3/10, 45.0 | 4/10, 3/10, 35.0 | 2/10, 1/10, 15.0 | 31.7 |
| MLLM [26] + VLA [5] | 6/10, 4/10, 50.0 | 4/10, 2/10, 30.0 | 2/10, 1/10, 15.0 | 31.7 |
| Geometric + VLA [5] | 7/10, 4/10, 55.0 | 3/10, 3/10, 30.0 | 4/10, 4/10, 40.0 | 41.7 |
| GesVLA (decoupled VLMs) | 10/10, 4/10, 70.0 | 8/10, 3/10, 55.0 | 7/10, 5/10, 60.0 | 61.7 |
| **GesVLA (full)** | 10/10, 9/10, 95.0 | 9/10, 6/10, 75.0 | 8/10, 8/10, 80.0 | **83.3** |

观察 pattern：

1. **Text-only 31.7% → GesVLA 83.3%, 提升 51.6 pp** — gesture 价值巨大
2. **Decoupled 61.7% → Full cross-attention 83.3%, 提升 21.6 pp** — latent interaction 比 visual prompt 接口强
3. **Simple → Hard gap**: Block 100%→90%, Jelly 90%→60%, Fruit 80%→80% — sequential multi-object pointing 最难
4. **Geometric pipeline + VLA 只有 41.7%** — 几何 ray-cast 在 cluttered scene 里太脆弱, 各模块 error 累积

### Table II: Intent reasoning (88 个真实场景)

| Method | Acc (%) | Progress Score (%) |
|---|---|---|
| Baseline-1 (Qwen3.5-plus prompted MLLM) | 38.6 | 61.4 |
| Baseline-2 (Geometric pipeline: MediaPipe + GroundingDINO + DepthAnythingV2) | 59.1 | 78.4 |
| **GesVLA (VLM_int)** | **94.3** | **97.2** |

**Baseline-1 (prompted MLLM) 失败模式**: 倾向选"距离指尖最近的 object"而非"pointing direction 指向的 object"。这是 image-space 距离 vs 3D ray-cast 几何 reasoning 的根本差异。Prompt 怎么调都没用，因为 MLLM 没有 pointing-direction 几何的 inductive bias。

**Baseline-2 (geometric pipeline) 失败模式**: MediaPipe keypoint 错一点, GroundingDINO detection 漏一个, DepthAnythingV2 depth 估错一个，整个 pipeline fail。Decoupled module 累积 error 是致命弱点。

**GesVLA 94.3%** 的关键: joint optimization 让所有 error source 一起 train, hand pose + pointing direction + scene understanding 在 latent space 里互相 calibrate。

### Table IIIa: Data engine ablation

| Variant | Acc (%) |
|---|---|
| w/o coord. jitter | 42.0 |
| w/o hand augm. | 76.1 |
| Full pipeline | 94.3 |

**Coord jitter 跌 52.3 pp** 是最 dramatic 的 ablation 结果。证明 coordinate tokenization 极易 overfit, jitter 是必须的 regularization。这一点对所有用 discrete coordinate tokens 的工作 (RT-2 的 action token 也有类似问题) 都有启发。

### Table V: Policy architecture

| Variant | Block | Jelly | Fruit/Veg | Avg |
|---|---|---|---|---|
| w/o visual prompt | 15/20 | 12/20 | 13/20 | 66.6 |
| + text prompt | 18/20 | 17/20 | 15/20 | 83.3 |
| Full model (cross-attn) | 19/20 | 15/20 | 16/20 | 83.3 |

**有意思的发现**: text prompt 和 cross-attention 在当前 3 个 task 上 avg 一样 (83.3%)。说明对 paper 的 task 复杂度, discrete text prompt 已经够传达 intent。Cross-attention 真正的价值可能在更复杂 task (multi-step long-horizon, continuous trajectory guidance) 才显现——但 paper 没在更难 task 上 validate cross-attention 的优越性。这是 paper 的一个 open question。

## 跟 Related Work 的脉络

### VLA lineage
- RT-2 [1]: 早期 VLA, VLM fine-tune 输出 action token
- OpenVLA [2]: 7B open-source VLA
- π0 [5]: flow matching action expert, GesVLA 直接复用其 VLM backbone + action expert 范式
- π0.5 [3]: open-world generalization
- GR00T N1 [14]: humanoid foundation model
- Hi Robot [15]: hierarchical VLA, 思想最接近 GesVLA 的 decoupling
- OneTwoVLA [12]: adaptive reasoning
- Dual Process VLA [13]: System 1/2 split

### Modality-augmented VLA
- DepthVLA [16]: 加 depth
- PointVLA [17]: 加 point cloud
- VTLA [18]: 加 tactile

这些 modality 都是 enhance **environment perception during execution**, GesVLA 的 gesture 是 enhance **task understanding at intent specification time**。本质不同。

### Gesture robotics lineage
- Geometric ray-casting: [9], [11], [19] — shoulder-wrist alignment heuristic
- Gesture-to-text + LLM: GestLLM [10], Lin et al. [20] — decoupled pipeline
- Specialized pointing prediction: [21], [22], [23]
- VLM visual prompt: "Point What You Mean" [24] (https://arxiv.org/abs/2512.18933)

GesVLA 的差异: **primary modality + deep feature-level fusion + 避免累积误差**。

## 我的几个 critical observations

### 1. Cross-attention vs text prompt 等价是个隐忧

Paper 把 cross-attention 作为核心 contribution 之一, 但 Table V 显示 text prompt 已经能达到 83.3%。Paper 解释说 cross-attention "更 elegant", 但没在更难 task 上证明它 superior。如果未来发现 text prompt 在所有 task 上都够用, cross-attention 的 architectural complexity 就 questionable。这是个需要更深入 validate 的点。

### 2. Keyframe selection 是 brittle heuristic

"Motion stagnation" detection 假设用户有明确 pause。Dynamic pointing (手指不停顿连续扫过几个 object) 或 fluid gesture 会 break 这个假设。Paper 没讨论这种 case, 是个明显 limitation。

### 3. 12-dim keypoint 是 hard inductive bias — 双刃剑

好处: sample efficient, explicit pointing-direction structure。
坏处: 只支持 index-finger pointing, 不能扩展到其他 deictic gesture (open-hand, head pointing, eye gaze)。Future work 如果想扩展 gesture type, 这个 12-dim vector 设计需要重新考虑。

### 4. Coordinate bin overfitting 是 general 问题

w/o jitter 跌到 42% 这件事对所有用 discrete coordinate token 的工作都有启发。Alternative 方案：
- Continuous coordinate head (像 DETR 那种)
- Anchor-based representation
- Relative coordinate encoding

这些可能比 bin discretization 更 robust, 但破坏了"unified token space"的简洁性。Trade-off 值得探索。

### 5. Asymmetric attention 是真正有价值的 design pattern

Compute ratio 1 : T : T×N 在 long-horizon task 上收益巨大。这种 cache reuse 思想在 Hierarchical VLA 越来越主流。Hi Robot, Dual Process VLA, OneTwoVLA 都在不同程度做类似的事。

但 GesVLA 的 special 在于: 它把 **intent modality (gesture+language)** 和 **perception modality (vision)** 分到不同 VLM, 而非只把 high-level planning 和 low-level policy 分开。这让 cache reuse 更有意义——intent 真的在一次 episode 内不变。

### 6. Semi-synthetic data 是 scalable 的关键

Real scene background + synthetic hand 这个组合非常聪明。比起完全 synthetic scene (SAPIEN/Habitat), 它保留了 real-world visual statistics。比起 real gesture collection, 它提供 exact annotation。这个 pattern 可以推广到其他需要 spatial annotation 但 real-world 收集昂贵的 modality (e.g., gaze direction, body pose pointing)。

### 7. Single-view gesture 限制了 3D reasoning

Paper 只用 right-view camera 看 gesture, 限制了 3D pointing 几何推理精度。Multi-view gesture 应该能进一步提升, 但需要更多 annotation effort。Future direction。

## Final intuition 总结

如果你 Andrej 想自己重新设计 GesVLA, 关键 design 决策 tree：

**Q1: Gesture 怎么 encode?**
- 选项 A: Raw gesture video → VLM (信息丰富但冗余, sample inefficient)
- 选项 B: Hand keypoint vector (GesVLA 选择, explicit pointing direction, hard inductive bias)
- 选项 C: VLM-extracted gesture feature (无 hard bias, 但需要大量数据)

**Q2: Gesture 如何与 action 耦合?**
- 选项 A: Discretize to text → text-conditioned VLA (information loss, paper 证明弱)
- 选项 B: Cross-attention latent interaction (GesVLA 选择, tight coupling)
- 选项 C: Joint single-VLM (compute inefficient, cache reuse 不可能)

**Q3: 数据从哪来?**
- 选项 A: Real gesture collection (贵, annotation 模糊)
- 选项 B: Fully synthetic scene (sim-to-real gap 大)
- 选项 C: Real scene + synthetic hand (GesVLA 选择, sweet spot)

**Q4: 训练策略?**
- 选项 A: Joint training (optimization conflict, paper 证明 fail)
- 选项 B: Two-stage frozen (GesVLA 选择, 解耦 weak/strong supervision)
- 选项 C: Two-stage unfrozen (小提升但需更多 annotation)

GesVLA 的答案在 4 个 question 上都选了"second-best 看起来不那么 fancy 但 ablation 证明最 work"的方案。这种 engineering taste 值得学习。

## 一句话 Final

**GesVLA 把"手指一下"这件人类最自然的 spatial disambiguation 行为, 用 12 维 keypoint vector + cross-attention latent interaction + semi-synthetic data engine + two-stage training 这套组合, 第一次真正塞进 VLA 的 latent space。在 cluttered scene 上 51.6 个百分点的绝对提升不是 incremental, 是 paradigm shift——证明 non-language modalities 在 VLA 里远未被充分探索。**

References:
- GesVLA: https://gwxuan.github.io/GesVLA/
- π0: https://arxiv.org/abs/2410.24164
- PaliGemma: https://arxiv.org/abs/2407.07726
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- GR00T N1: https://arxiv.org/abs/2503.14734
- OneTwoVLA: https://arxiv.org/abs/2505.11917
- Dual Process VLA: https://arxiv.org/abs/2410.15549
- MediaPipe Hands: https://arxiv.org/abs/2006.10214
- GroundingDINO: https://arxiv.org/abs/2303.05499
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Qwen3: https://arxiv.org/abs/2505.09388
- π0.5: https://arxiv.org/abs/2504.16054
- DepthVLA: https://arxiv.org/abs/2510.13375
- VTLA: https://arxiv.org/abs/2505.09577
- Point What You Mean: https://arxiv.org/abs/2512.18933
- GestLLM: https://arxiv.org/abs/2502.19417 (相关 HRI work)

---

# GesVLA 深度技术讲解

Andrej，这篇 GesVLA (Gesture-aware Vision-Language-Action Model with Embedded Representations) 我从头到尾拆给你看，重点 build intuition 关于"为什么这样设计"。

Project page: https://gwxuan.github.io/GesVLA/

## I. 核心问题与 Motivation

VLA 系列工作 (RT-2 [1], OpenVLA [2], π0 [5], π0.5 [3], GR00T N1 [14], Hi Robot [15], OneTwoVLA [12]) 都依赖 language instruction 作为 human intent 的主接口。在 cluttered scenes 里，"pick this up and put it there" 这种 deictic 表达存在根本性的 spatial ambiguity。

Andrej 你肯定熟 π0，π0 在 DROID 等数据上 instruction 一般都包含具体 object name + spatial context，但当 scene 里有 multiple similar instances (例如 7 个颜色相近的 block)，language 必须编码成 "the leftmost red block" 这种 verbose 描述，对 real-time HRI 是 fail。

GesVLA 的关键 insight: **gesture 应该作为 first-class modality 与 vision/language 并列，直接 encode 到 latent space 参与推理，避免把它 discretize 成 text token 导致 spatial info loss**。

相关 reference:
- π0 flow matching VLA: https://arxiv.org/abs/2410.24164
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Hi Robot hierarchical VLA: https://arxiv.org/abs/2502.19417
- PaliGemma backbone: https://arxiv.org/abs/2407.07726

## II. Architecture 深度解析

### A. Gesture Embedding 的设计哲学

公式 (1):
$$
\mathbf{z}_i^g = \phi(\mathrm{Pose}(g_i)), \quad g_i \in S(\mathrm{Pose}(\mathcal{G}))
$$

变量含义:
- $\mathcal{G}$: 完整 gesture video
- $\mathrm{Pose}(\cdot)$: 用 MediaPipe Hands [8] 做 hand pose estimation, https://arxiv.org/abs/2006.10214
- $S(\cdot)$: 基于 hand motion dynamics 的 keyframe selection function
- $g_i$: 第 $i$ 个 keyframe, $i \in \{1, \dots, F\}$
- $\phi(\cdot)$: 由 multiple MLP layers 组成的 gesture encoding module
- $\mathbf{z}_i^g$: 投影到 shared latent space 的 gesture embedding

**关键设计 1: keyframe selection based on motion stagnation。** 人类 pointing 时会有短暂停顿，paper 通过 track hand keypoints over time 选 pause 帧。这过滤了冗余 temporal 信息，把"手势视频"降为"几幅静态指向图"。

**关键设计 2: 4 keypoints × (x,y,d) = 12-dim vector。** 只用 wrist + index finger 的 3 个 joints (MCP, PIP, DIP)。这是个非常 economized 的 representation——拒绝用 raw RGB hand crop，也拒绝用 full 21 keypoints。深度 $d$ 来自 depth estimation，给了 3D cue。

这里 Andrej 你可以想：12 维 vector 经过 MLP 投影到 PaliGemma 的 hidden dim (大概 2048)，相当于把 gesture 压缩成一个 explicit 的 pointing-direction 向量。ablation (Table IIIb) 显示 w/o gesture MLP 跌到 84.1%——raw gesture image 只能提供 partial spatial cue，explicit pointing-direction vector 才是核心。

### B. Dual-VLM 解耦设计

公式 (2):
$$
(\mathbf{y}, \mathcal{K}^{\mathrm{int}}, \mathcal{V}^{\mathrm{int}}) = \mathrm{VLM}_{\mathrm{int}}(\mathcal{G}, \mathcal{T})
$$

变量:
- $\mathrm{VLM}_{\mathrm{int}}$: intent reasoning VLM (initialized from PaliGemma-2B)
- $\mathcal{G}$: gesture keyframes + keypoint features
- $\mathcal{T}$: language instruction (paper 中 $\tau$ 和 $\mathcal{T}$ 互换用)
- $\mathbf{y}$: reasoning outputs, 包括 textual descriptions of inferred targets 和 post-processed visual prompts
- $\mathcal{K}^{\mathrm{int}}, \mathcal{V}^{\mathrm{int}}$: 跨 layer 缓存的 key-value states, 后续用于 cross-attention

公式 (3):
$$
(\mathcal{K}^{\mathrm{per}}, \mathcal{V}^{\mathrm{per}}) = \mathrm{VLM}_{\mathrm{per}}(\mathcal{O}, \mathcal{T}, \mathbf{y}; \mathcal{K}^{\mathrm{int}}, \mathcal{V}^{\mathrm{int}})
$$

变量:
- $\mathrm{VLM}_{\mathrm{per}}$: online perception VLM (initialized from π0 VLM backbone)
- $\mathcal{O}$: multi-view RGB observations
- $\mathbf{y}$: 来自 $\mathrm{VLM}_{\mathrm{int}}$ 的 reasoning outputs
- $\mathcal{K}^{\mathrm{int}}, \mathcal{V}^{\mathrm{int}}$: 通过 cross-attention 让 $\mathrm{VLM}_{\mathrm{per}}$ 直接 attend 到 intent representation
- $\mathcal{K}^{\mathrm{per}}, \mathcal{V}^{\mathrm{per}}$: 输出的 KV states 传给 action expert

公式 (4):
$$
\mathbf{a}_{1:K} = \mathcal{F}_\theta(\mathbf{x}_0, \mathbf{s}_t, \mathcal{K}^{\mathrm{per}}, \mathcal{V}^{\mathrm{per}})
$$

变量:
- $\mathcal{F}_\theta$: flow-based policy (action expert)
- $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 初始噪声 action trajectory
- $\mathbf{s}_t$: current robot state (joint position/velocity, gripper state)
- $\mathcal{K}^{\mathrm{per}}, \mathcal{V}^{\mathrm{per}}$: conditioning context
- $\mathbf{a}_{1:K}$: 输出的 action chunk of length K

公式 (5) 的 asymmetric attention:
$$
\mathrm{VLM}_{\mathrm{int}} \to \mathrm{VLM}_{\mathrm{per}} \to \mathrm{Action\ Expert}
$$

**Intuition: 把"理解"和"反应"放到不同的时间尺度上**。Gesture-language intent 在一次 episode 内不变，所以 $\mathrm{VLM}_{\mathrm{int}}$ 算一次，KV cache 复用；scene observation 每步变化，所以 $\mathrm{VLM}_{\mathrm{per}}$ 每步算；action 生成需要 N 步 denoise，所以 action expert 算 T×N 次。最终 compute ratio 是 1 : T : T×N。

这种 hierarchical decomposition 在 Hi Robot [15] 和 Dual Process VLA [13] 里都有类似思想，但 GesVLA 通过 cross-attention 而非 discrete text interface 让上下两层 VLM 紧耦合。

Algorithm 1 完整 inference loop:
```
1: Input: O_0, T, G, s_0
2: Estimate hand poses, select keyframes {g_i}
3: Encode gesture embeddings z_i^g
4: (y, K^int, V^int) ← VLM_int(G, T)  [computed once]
5: while task not finished:
6:   (K^per, V^per) ← VLM_per(O, T, y; K^int, V^int)  [every step]
7:   x_0 ~ N(0, I)
8:   a_{1:K} ← F_θ(x_0, s_t, K^per, V^per)  [T × N denoise]
9:   Execute a_{1:K} (or partial a_{1:k})
10:  Update O, s_t
```

## III. Scalable Gesture Data Engine

这是 paper 最 crafty 的部分。问题: real-world gesture data 收集昂贵，pointing target annotation 模糊。解法: **semi-synthetic pipeline 把 hand model 渲染到 real RGB-D scene backgrounds 上**。

### A. Target selection and 3D grounding

公式 (6):
$$
\mathbf{p} = \left(\frac{(u - c_x) z}{f_x}, \frac{(v - c_y) z}{f_y}, z\right)
$$

变量:
- $(u, v)$: jittered image coordinate (object center + random offset 模拟自然 pointing 偏差)
- $z$: depth map 在 $(u, v)$ 处的深度值
- $(f_x, f_y)$: camera focal lengths (x, y 方向)
- $(c_x, c_y)$: principal point (camera optical center in pixel)
- $\mathbf{p}$: 反投影得到的 3D target point

流程: 用 GroundingDINO [25] (https://arxiv.org/abs/2303.05499) 检测 candidate objects → 随机采样 task-relevant targets → 对 bbox center 加 random jitter → 用 depth + intrinsics 反投影得到 3D anchor。

这个 jitter 至关重要: ablation Table IIIa 显示 w/o coord. jitter 跌到 42.0%。原因: 模型会 overfit 到 fixed coordinate bins，sim-to-real 时 bin 偏移直接 fail。Jitter 强制模型学习 pointing-direction geometric reasoning 而非 memorize 坐标。

### B. Pointing motion generation

公式 (7):
$$
\mathbf{p}_h = \mathbf{p}_t - \mathbf{d}(\tau + k\Delta)
$$

变量:
- $\mathbf{p}_t$: target point
- $\mathbf{d}$: sampled unit direction (hand approach direction)
- $\tau$: stopping threshold (hand 距离 target 的最小距离)
- $k$: step index
- $\Delta$: step size
- $\mathbf{p}_h$: hand position at step k

hand 从 random direction 沿 $\mathbf{d}$ 逐渐 move toward $\mathbf{p}_t$，到 $\tau$ 距离停下，保留 valid pointing pose。

公式 (8) multi-target 之间的 parabolic lifting:
$$
\mathbf{p}_{\mathrm{vis}} = \mathbf{p}_h + h_{\max}(1 - (2\alpha - 1)^2)\mathbf{n}_{\mathrm{up}}
$$

变量:
- $\mathbf{p}_{\mathrm{vis}}$: visible hand position (实际渲染位置)
- $\mathbf{p}_h$: base hand position along pointing direction
- $h_{\max}$: maximum lift height
- $\alpha \in [0, 1]$: normalized motion progress between two targets
- $\mathbf{n}_{\mathrm{up}}$: upward direction

这是个 parabola: 当 $\alpha = 0.5$ 时 $1 - (2\alpha - 1)^2 = 1$ 达到峰值 $h_{\max}$；$\alpha = 0$ 或 $1$ 时为 0。比 linear interpolation 更像真人手势——人指完一个 object 抬手再指下一个。

### C. Semi-synthetic 数据特性

约 16k samples，每个 sample 包含:
1. Video $\mathcal{O}$: pointing process 渲染视频，camera viewpoint 与原 real scene 一致
2. Language instruction $\mathcal{T}$: 如 "pick up this and put it there"
3. Supervision $\mathbf{y}$: targets 和 locations (用于 $\mathrm{VLM}_{\mathrm{int}}$ 训练)
4. Metadata: task type index, scene index

**关键 intuition: hand asset synthetic + real scene background = 可扩展 + 低 sim-to-real gap**。比起完全 synthetic scene (像 Habitat 或 SAPIEN 里的)，real background 保留了 realistic lighting、texture noise、occlusion patterns。比起 real gesture collection，automatic annotation 提供了 exact pointing labels。

## IV. Two-Stage Training Pipeline

### Stage 1: Intent Reasoning Pre-training

公式 (9):
$$
\mathcal{L}_{\mathrm{int}} = -\frac{1}{M} \sum_{i=1}^{M} \log P(a_i^* \mid a_{<i}^*, \mathcal{G}, \mathcal{T})
$$

变量:
- $\mathcal{L}_{\mathrm{int}}$: teacher-forced autoregressive cross-entropy loss
- $M$: sequence token 数量
- $a_i^*$: 第 $i$ 个 target token, 包括 text tokens 和 discretized coordinate tokens
- $a_{<i}^*$: 前面已生成 tokens
- $\mathcal{G}, \mathcal{T}$: gesture 和 language input

**关键设计: target coordinates discretized into bins 作为 special tokens**。这让 VLM_int 在 unified token space 里同时 learn semantic reasoning (e.g., "the red block") 和 spatial grounding (e.g., bin id 指代 coordinate)。避免了 detector + LLM 这种 decoupled pipeline 的 cumulative errors。

### Stage 2: Joint Action Generation Training

公式 (10) Flow matching objective:
$$
\mathcal{L}_{\mathrm{action}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} \left[ \| \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]
$$

变量:
- $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: source noise
- $\mathbf{x}_1$: target action trajectory (从 demonstration 抽取)
- $t \in [0, 1]$: flow time step
- $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$: linear interpolation
- $\mathbf{c}$: conditioning context (来自 $\mathrm{VLM}_{\mathrm{per}}$ 的 KV + robot state $\mathbf{s}_t$)
- $\mathbf{v}_\theta$: velocity prediction network (action expert)

这是 standard flow matching / rectified flow loss: 训练网络预测从 $\mathbf{x}_0$ 到 $\mathbf{x}_1$ 的 velocity field, 在直线 path 上 velocity 恒等于 $\mathbf{x}_1 - \mathbf{x}_0$。Inference 时从 $\mathbf{x}_0$ 出发用 N 步 Euler ODE 求解到 $\mathbf{x}_1$。

和 π0 的 action expert 是同一思想，https://arxiv.org/abs/2410.24164 里有更详细推导。

### Stage 关系

Stage 1 在 semi-synthetic data 上 pretrain $\mathrm{VLM}_{\mathrm{int}}$。
Stage 2 在 real robot demo 上训练 $\mathrm{VLM}_{\mathrm{per}}$ + action expert，$\mathrm{VLM}_{\mathrm{int}}$ frozen。

Ablation Table IV 显示:
- Joint training all modules: 45.0% (synthetic reasoning data 主导优化, 损害 action learning)
- Two-stage, VLM_int unfrozen in stage 2: 80.0%
- Two-stage, VLM_int frozen: 83.3% (最佳)

这暗示: semi-synthetic gesture reasoning supervision 已经足够, real robot demo 里再 unfreeze VLM_int 反而会过拟合到 small real gesture set。

### Implementation details

- $\mathrm{VLM}_{\mathrm{int}}$ backbone: PaliGemma-2B [4]
- $\mathrm{VLM}_{\mathrm{per}}$ backbone: π0 的 VLM backbone 初始化
- Action expert: from scratch
- Optimizer: AdamW + warmup cosine LR schedule + EMA
- Hardware: 4× H20 GPU training, 1× RTX 4090 inference

## V. 实验深度解析

### Setup
- 7-DOF ARX5 robotic arm
- 3 cameras @ 30 fps: global view, right-side view, gripper-mounted view
- Tasks: Pick-and-Place Block, Select Jelly, Select Fruit/Vegetable (各 20 trials)

### Table I: Real-robot manipulation (核心结果)

| Method | Block (S/H/Avg) | Jelly (S/H/Avg) | Fruit/Veg (S/H/Avg) | Total Avg |
|---|---|---|---|---|
| Text-only VLA [5] | 6/10, 3/10, 45.0 | 4/10, 3/10, 35.0 | 2/10, 1/10, 15.0 | 31.7 |
| MLLM [26] + VLA [5] | 6/10, 4/10, 50.0 | 4/10, 2/10, 30.0 | 2/10, 1/10, 15.0 | 31.7 |
| Geometric + VLA [5] | 7/10, 4/10, 55.0 | 3/10, 3/10, 30.0 | 4/10, 4/10, 40.0 | 41.7 |
| GesVLA (decoupled) | 10/10, 4/10, 70.0 | 8/10, 3/10, 55.0 | 7/10, 5/10, 60.0 | 61.7 |
| **GesVLA (full)** | 10/10, 9/10, 95.0 | 9/10, 6/10, 75.0 | 8/10, 8/10, 80.0 | **83.3** |

关键 pattern:
- Text-only 31.7% → GesVLA 83.3%, **绝对提升 51.6 个百分点**
- Decoupled (visual prompt 接口) 61.7% → Full (cross-attention) 83.3%, **cross-attention 增益 21.6 pp**
- Simple → Hard gap: Block 是 100%→90%, Jelly 是 90%→60%, Fruit 是 80%→80%, 说明 sequential multi-object pointing 仍是最难场景

### Table II: Intent reasoning (88 samples)

| Method | Acc (%) | Progress Score (%) |
|---|---|---|
| Baseline-1 (Qwen3.5-plus prompted) | 38.6 | 61.4 |
| Baseline-2 (Geometric pipeline) | 59.1 | 78.4 |
| **GesVLA** | **94.3** | **97.2** |

Baseline-1 failure mode: MLLM 倾向选 "closest object to finger" 而非 "true target along pointing direction"。这是 image-space 距离 vs 3D ray-cast 几何 reasoning 的根本差异。
Baseline-2 failure mode: MediaPipe keypoint + GroundingDINO + DepthAnythingV2 [27] 任一模块 error 累积。
GesVLA 把所有 error source 联合 optimize, robustness 大幅提升。

### Table IIIa: Data engine ablations

| Variant | Acc (%) |
|---|---|
| w/o coord. jitter | 42.0 |
| w/o hand augm. | 76.1 |
| Full pipeline | 94.3 |

Coord jitter 影响最大 (-52.3 pp)。说明 model 极易 overfit 到 fixed coordinate bins，必须强制学习 pointing direction geometric reasoning。
Hand appearance augmentation 影响 -18.2 pp, sim-to-real hand appearance gap 是真实存在的。

### Table IIIb: Training design ablations

| Variant | Acc (%) |
|---|---|
| w/o gesture MLP | 84.1 |
| w/o data augm. | 89.8 |
| Full model | 94.3 |

Gesture MLP 提供 explicit pointing-direction encoding, -10.2 pp。Raw gesture image 只能 partial 恢复 spatial cue。

### Table IV: Training strategy

| Variant | Block | Jelly | Fruit/Veg | Avg |
|---|---|---|---|---|
| Joint training (all) | 10/20 | 8/20 | 9/20 | 45.0 |
| Two-stage (VLM_int unfrozen) | 20/20 | 15/20 | 13/20 | 80.0 |
| Two-stage (VLM_int frozen) | 19/20 | 15/20 | 16/20 | 83.3 |

Joint training 失败的原因: 16k synthetic reasoning sample 数量级压过 real robot demo, gradient 被前向 reasoning loss 主导, action expert 学不到 valid policy。

### Table V: Policy architecture

| Variant | Block | Jelly | Fruit/Veg | Avg |
|---|---|---|---|---|
| w/o visual prompt | 15/20 | 12/20 | 13/20 | 66.6 |
| + text prompt | 18/20 | 17/20 | 15/20 | 83.3 |
| Full model (cross-attn) | 19/20 | 15/20 | 16/20 | 83.3 |

Interesting finding: text prompt 和 cross-attention 最终 avg 一样 (83.3)。这说明 in their task distribution, discrete text prompt 已经足够传达 intent。但 paper 强调 cross-attention 更 elegant, 在更复杂 task 上应该会胜出。这点 paper 没 fully validate。

## VI. 与 Related Work 的精确对比

### A. VLA lineage

- RT-2 [1] (https://robotics-transformer2.github.io/): 早期 VLA, show VLM 可以 fine-tune 输出 action token
- OpenVLA [2] (https://openvla.github.io/): open-source 7B VLA
- π0 [5] (https://arxiv.org/abs/2410.24164): flow matching for action, GesVLA 直接用它的 VLM backbone + action expert 范式
- π0.5 [3] (https://arxiv.org/abs/2504.16054): open-world generalization
- GR00T N1 [14] (https://arxiv.org/abs/2503.14734): humanoid foundation model
- Hi Robot [15] (https://arxiv.org/abs/2502.19417): hierarchical VLA, 思想最接近 GesVLA 的 decoupling
- OneTwoVLA [12] (https://arxiv.org/abs/2505.11917): adaptive reasoning
- Dual Process VLA [13] (https://arxiv.org/abs/2410.15549): System 1 / System 2 split

### B. Modality-augmented VLA

- DepthVLA [16] (https://arxiv.org/abs/2510.13375): 加 depth modality
- PointVLA [17]: 加 3D point cloud
- VTLA [18] (https://arxiv.org/abs/2505.09577): 加 tactile

这些工作的 modality 都是为了 enhance environmental perception during execution, GesVLA 的 modality 是 enhance task understanding (intent specification)。

### C. Gesture robotics 的 lineage

- Geometric rules + ray casting: Edge & Sattar [9], Sassali & Pieters [11], Hu et al. [19] — shoulder-wrist alignment 类 geometric heuristic
- Gesture-to-text + LLM: GestLLM [10], Lin et al. [20] — decoupled pipeline
- Specialized pointing prediction: Bamani et al. [21], Matuszek et al. [22], Müller et al. [23]
- VLM-generated visual prompt: "Point What You Mean" [24] (https://arxiv.org/abs/2512.18933)

GesVLA 的差异: 把 gesture 作为 primary modality 用 deep feature-level fusion, 避免 multi-module pipeline 的 cumulative errors。

## VII. 几点 Critical Observations

### 1. Cross-attention vs visual prompt 在 paper 的 task 上没有显著差异

Table V 的 +text prompt 和 Full model 同为 83.3%。这暗示对 3 个 task 的 intent 复杂度, discrete text 已经足够。Cross-attention 真正的价值需要更复杂 task (例如 multi-step long-horizon pointing + continuous trajectory guidance)。

### 2. Keyframe selection 是 brittle 的 heuristic

"Motion stagnation" detection 假设 user 有明显 pause。连续 dynamic pointing 或 fluid gesture 需要更 robust 的 keyframe extraction。Paper 没有讨论 multi-target without pause 的情况。

### 3. Hand appearance sim-to-real gap 仍是主要 bottleneck

w/o hand augm. 跌 18 pp。当前 pipeline 用 hand mesh rendering, 在 lighting / occlusion / viewpoint diversity 上需要更强 augmentation。可能是 future work 的方向。

### 4. Two-stage training 揭示的多任务平衡问题

Joint training 45% vs Two-stage frozen 83.3% 是巨大 gap。这其实是 multi-task learning 里常见的 optimization conflict——synthetic reasoning 数据 scale 远大于 real robot demo, gradient magnitude 不平衡, 需要通过 stage 解耦。借鉴 Curriculum Learning 思想。

### 5. 12-dim keypoint vector 是 hard inductive bias

用 4 keypoints × (x,y,d) 而非 raw video 给了 explicit pointing-direction structure。这是 good inductive bias for pointing, 但 limit 了 gesture type——只支持 index-finger pointing, 不支持其他 deictic gesture (open-hand palm pointing, head pointing, eye gaze 等)。

### 6. Coord jitter 揭示的 tokenization 风险

w/o jitter 跌到 42% 说明 coordinate token bin 极易 overfit。这是 discrete coordinate token 方案的 known issue, 类似 RT-2 早期 action token 的问题。可能 alternative: continuous coordinate head 或 anchor-based representation。

### 7. Asymmetric attention 的 compute efficiency

Compute ratio 1 : T : T×N 在 long-horizon task 上收益巨大。如果 T = 200 control steps, N = 10 denoise steps, VLM_int 只算 1 次而非 2000 次。这种 cache reuse 思想在 Hierarchical VLA (Hi Robot) 和 Twin-VLA 等里都会越来越重要。

## VIII. 你的 Intuition 该如何建立

如果让 Andrej 你自己重新设计 GesVLA, 关键 design 决策:

1. **Modality hierarchy**: gesture/language 是"intent" modality (low frequency, high-level), vision 是"perception" modality (high frequency, low-level), 把它们分到不同 VLM, 通过 cross-attention 单向流通。
2. **Inductive bias injection**: 12-dim keypoint vector 把 geometric pointing direction 直接 encode 进 latent space, 而非让模型从 raw image 学这个 mapping。
3. **Data engine**: real scene background + synthetic hand = scalability + sim-to-real friendliness 的 sweet spot。Coord jitter 强制 model 学 geometric reasoning 而非 memorize bins。
4. **Two-stage decomposition**: 让大规模 weakly-supervised synthetic reasoning data 不污染 small-scale strongly-supervised real robot data。
5. **Asymmetric attention**: 高层理解 cache 复用, 低层反应 per-step 更新, 对应 human cognitive 中 System 2 (slow reasoning) vs System 1 (fast reaction) 的 dual process 思想。

## IX. Limitations 和 Future Directions

Paper 自己承认的:
- 只考虑 pointing gesture
- 没有更复杂的 HRI collaboration

我观察到的潜在问题:
- Cross-attention vs text prompt 在当前 task 上等价, paper 没证明 cross-attention 在更复杂 task 上的必要性
- Keyframe selection heuristic 在 dynamic gesture 上脆弱
- Single-view gesture (right camera only) 限制了 3D pointing 推理
- Hand mesh rendering 的 realism 仍是 bottleneck

Future 可能方向:
- Hand mesh + diffusion-based hand rendering (e.g., MagicHand, HaMeR) 缩小 sim-to-real gap
- 把 gesture 扩展到 dynamic trajectory guidance (例如 "follow my hand motion")
- 用 depth + multi-view gesture 提升 3D pointing accuracy
- Continuous coordinate head 替代 discrete bins, 避免 overfitting

## X. 总结

GesVLA 的核心贡献是把 gesture 作为 first-class modality 通过 dual-VLM 架构与 cross-attention 紧耦合到 VLA 框架。Semi-synthetic data engine + two-stage training 让 weakly-supervised reasoning data 和 strongly-supervised real robot data 各司其职。在 cluttered scenes 上 51.6 个百分点的绝对提升, 在 intent reasoning 上 94.3% vs 38.6% (baseline MLLM) vs 59.1% (baseline geometric) 证明了 gesture-aware design 的价值。

Andrej, 你看这种把 multi-modal intent 分层到不同 VLM 然后通过 cross-attention 而非 token 接口传递的设计, 我觉得未来会越来越主流——尤其是 long-horizon task 里 intent 是 low-frequency signal, perception/action 是 high-frequency signal, 用 asymmetric attention 把它们 decouple 是 compute efficiency 和 representation power 的双 win。

References:
- GesVLA project: https://gwxuan.github.io/GesVLA/
- π0: https://arxiv.org/abs/2410.24164
- PaliGemma: https://arxiv.org/abs/2407.07726
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Hi Robot: https://arxiv.org/abs/2502.19417
- GR00T N1: https://arxiv.org/abs/2503.14734
- OneTwoVLA: https://arxiv.org/abs/2505.11917
- Dual Process VLA: https://arxiv.org/abs/2410.15549
- MediaPipe Hands: https://arxiv.org/abs/2006.10214
- GroundingDINO: https://arxiv.org/abs/2303.05499
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Qwen3: https://arxiv.org/abs/2505.09388
- π0.5: https://arxiv.org/abs/2504.16054
- DepthVLA: https://arxiv.org/abs/2510.13375
- VTLA: https://arxiv.org/abs/2505.09577
- Point What You Mean: https://arxiv.org/abs/2512.18933
