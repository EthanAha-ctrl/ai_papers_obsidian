---
source_pdf: EMMA End-to-End Multimodal Model for.pdf
paper_sha256: c207fd303d7fd230c622bc84309c2dcf835fbaaa12d834935d1fb4b728dc8972
processed_at: '2026-08-18T11:00:30-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 EMMA

## 一句话版本

把开车这件事变成"看图说话"——给模型看摄像头画面，问它"接下来怎么开"，它用文字把答案写出来。

---

## 为什么这件事有意思

传统自动驾驶是个大拼盘：一个模块认车，一个模块认路，一个模块猜别人怎么走，一个模块决定自己怎么走。模块之间用人为设计的"格式"传话（比如"前方3米有辆白色轿车"这种结构化数据）。

问题在于：这些人为设计的"传话格式"是提前定死的，遇到没见过的情况就卡壳。比如路上突然出现个梯子——你的物体检测模块根本没"梯子"这个类别，它传给下游的就是"无物体"，下游 planner 就直接撞上去了。

EMMA 的思路是：**别传话了，一个人全干**。

它用 Gemini 当大脑。Gemini 在网上看过几亿张图、几万亿字文本，它知道梯子是什么、松鼠是什么、施工锥是什么，即使你训练数据里没标这些。这就是 pre-trained world knowledge 的威力。

---

## 它具体怎么干

### 输入

三样东西：
1. **摄像头视频**（车周围一圈的画面）
2. **导航指令**（一句话："直行" / "左转" / "右转"）
3. **过去几秒车在哪**（一串坐标，比如"刚才在(-4.09, 0.01)，再之前在(-3.94, 0.01)..."）

### 输出

**未来几秒车该去哪**，也是一串坐标，比如"(0.83, 0.01) (1.72, 0.01) (2.67, 0.02)..."

就这样。输入画面+指令+历史，输出未来轨迹。中间没有任何"先检测物体再预测再规划"这种步骤，一个模型从头到尾。

### 坐标怎么变成文字

这是最 hacky 但也最聪明的地方：直接把数字写成文字。

坐标 (9.01, 3.22) 就写成文本 "9.01, 3.22"。

你可能会想：这不是很浪费 token 吗？每个数字要拆成好几个字符 token。用专门的"坐标 token"不是更紧凑？

他们试过对比，结论是：**直接写数字更好**。原因是 Gemini 预训练时学的是自然语言，你对齐到自然语言格式，它才能调动预训练学到的知识。你搞一套新 token，它就得从头学这套编码，pre-training 的优势就浪费了。

这跟 RT-2 在机器人里干的事一模一样：把连续动作用文字表示，让 VLM 直接输出动作。

---

## Chain-of-Thought：让它先想再开

光输出坐标还不够。他们还让模型先"说一段话"解释为什么要这么开，再输出轨迹。

这段话分四层：

1. **场景描述**："晴天，白天，四车道，中间有人行横道"
2. **关键物体**："行人位于(9.01, 3.22)，前方有车位于(11.58, 0.35)"
3. **物体行为**："行人在 sidewalk 上站着，看向马路，可能要过马路"
4. **驾驶决策**："保持低速"

然后才输出轨迹。

效果：比直接输出轨迹好 **6.7%**。

为什么有用？直觉上，让模型先 verbalize 它看到了什么、在想什么，等于强制它做一遍 scene understanding 和 reasoning，这些中间表示会 help 后面的 trajectory 生成。这和人开车类似——你不是直接打方向盘，你先扫一眼场景、判断关键物体、决定动作、再执行。

最妙的是这些 rationale 不需要人标。用现成的 perception model 找关键物体，用 Gemini 自己生成场景描述，用规则算 meta decision。整个 pipeline 是自动的。

---

## 不只是 planning，还能干别的

既然所有输入输出都是文字，那换个 prompt 就能换任务：

- **"检测图中所有物体"** → 输出 3D bounding box（坐标+尺寸+朝向+类别，全写成文字）
- **"前方可行驶车道在哪"** → 输出 road graph（车道线用一串点表示）
- **"前方道路是否临时 blocked"** → 输出 yes/no

同一个模型，同一套权重，靠 prompt 切换任务。这就是 generalist。

---

## 一起训练反而更好

最反直觉的发现：把 planning、detection、road graph 三个任务一起训练，每个任务都比单独训练好。

- Detection 提升了 **5.5%**
- Road graph 提升了 **2.4%**
- Planning 提升了 **1.4%**

为什么？因为这些任务本来就相互关联。你要规划好轨迹，得知道车在哪（detection）；要知道车道走向（road graph）。反过来，planning 的信号告诉模型"哪些物体是关键的需要关注"，等于给 detection 免费的 attention supervision。

Planning 是"hub task"——它需要全局理解，所以和它一起 train 的任务都受益最多。

---

## 结果怎么样

**Planning**：
- nuScenes 上比之前最好的方法（BEV-Planner）好 17%，而且 BEV-Planner 也是 self-supervised 的，公平对比
- WOMD 上 5 秒预测 horizon 比 MotionLM 好 22%，而 MotionLM 用了 LiDAR、人工标注的 road graph、traffic light 状态，EMMA 只用摄像头

**3D Detection**：
- Waymo Open Dataset 上比 BEVFormer precision 高 16%，recall 高 5.5%
- 近距离大幅领先，远距离优势消失（camera 分辨率限制）

**Road Graph**：
- 几个 representation choice 影响巨大：动态采样 vs 固定采样差 70-90%，ego 坐标系对齐差 25-60%

---

## 几个我觉得最 elegant 的细节

1. **标点符号有用**：把车道线写成 "(x1,y1 and x2,y2);..." 比 "x1 y1 x2 y2;..." 好 10%。Gemini 预训练时学的语言结构（括号、and、分号）真的在 work，它更适应这种有结构的表达。

2. **按深度排序检测框**：Pix2Seq 原文说检测框顺序无所谓，但 EMMA 发现按距离远近排序更好。因为 autoregressive 模型先输出近的物体，给远的物体提供了 context。

3. **Data 还没 saturate**：24M 场景的训练数据，loss 还在降。说明如果给更多数据，质量会继续提升。这是 scaling law 在自动驾驶上的直接证据。

4. **没见过的东西也能避**：训练数据没标松鼠，但模型能减速避松鼠。因为 Gemini 见过松鼠的图片，它的 world knowledge 迁移过来了。

---

## 还有什么问题

1. **只用摄像头，没 LiDAR**：远距离 depth 估计有物理限制，远物体检测不如 LiDAR 方案
2. **只看 4 帧**：没有 long-term memory，无法记住几秒前看到的路牌
3. **Open-loop 评测**：所有实验都是"放录像让你预测"，不是真正闭环跑。闭环需要 sensor simulation，计算成本极高
4. **实时性**：Gemini Nano 在云端跑 OK，上车还要 distillation + 量化。他们提到一个优化版能 3 FPS，比 UniAD 的 1.8 FPS 快，但离 10+ Hz 实时还有距离
5. **一致性没保证**：模型可以同时输出轨迹+检测+road graph，但不保证它们互相一致。轨迹说变道但检测没看到旁边的车——这种 inconsistency 在 safety-critical 场景是隐患

---

## 我的核心 takeaway

这篇 paper 证明了一件事：**当你有一个足够强的 pre-trained MLLM，很多你以为需要专门设计的 task，其实可以 cast 成 language generation**。

自动驾驶看起来是高度 geometric、physical 的任务，但本质上是 scene understanding + reasoning + decision。这三件事 MLLM 都擅长。

更广义地说，这预示一种 paradigm：**generalist foundation model + task-specific prompt** 可能在很多"看似需要 specialization"的领域都 competitive。关键是你能不能找到正确的 input/output representation，让 task 对齐到 pre-trained model 的 language space。

EMMA 的 representation choice 看着 ad-hoc（坐标直接写数字、车道线用括号和 and 连接），但 ablation 证明这些 choice 重要——因为它们 leverage 了 pre-trained language structure。这不是"随便选个格式"，这是"精心对齐到预训练分布"。

真正的瓶颈接下来不是 model architecture，而是 **closed-loop evaluation infrastructure** 和 **safety verification**。模型够强了，问题是怎么证明它够安全。

---

# EMMA: 把 Autonomous Driving 重新 cast 成 MLLM 的 VQA

Andrej 你好，这篇 paper 我读完后直觉上感觉它做的事情非常 elegant：把整个 autonomous driving stack 压扁成一个 sequence-to-sequence 问题，所有非 sensor 的 input/output 全部用 natural language text 表示，然后直接 fine-tune 一个 pre-trained MLLM（Gemini 1.0 Nano-1）来做 planning、detection、road graph、scene understanding 这些事情。这种做法和 RT-2 在 robotics 里把 action token 化成 text 的思路是高度同源的，本质都是借用 LLM 预训练得到的 language representation space 作为 universal interface。

paper 链接：https://openreview.net/forum?id=kH3t5lmOU8
arXiv 版本：https://arxiv.org/abs/2410.23262
Waymo blog：https://waymo.com/blog/2024/10/introducing-emma/

---

## 1. 核心思想的 intuition

传统 modular autonomous driving stack（perception → prediction → planning → control）的痛点是 module 之间的 interface 是 human-engineered 的 symbolic representation（比如 agent boxes、lane polylines、traffic light states），这种 interface 在 long-tail scenario 下会 bottleneck，因为 pre-defined schema 不一定能 cover novel scene。End-to-end（UniAD, VAD, PARA-Drive）尝试用 differentiable module 解决这个问题，但仍然是 task-specific 的，且 dataset 规模有限。

EMMA 的核心 insight 是：**Gemini 这种 MLLM 已经在 internet-scale data 上学到了非常 rich 的 world knowledge 和 reasoning 能力，autonomous driving task 本质上可以 cast 成一个 visual question answering 问题**。它把 "navigate through this intersection" 当成 "describe this image and answer this question" 一样处理。

这个 framing 有几个直接的好处：
1. **Pre-trained world knowledge 直接迁移**：比如 model 见过 construction cone、squirrel、dog、ladder 的图片，即使 fine-tuning dataset 里没标这些 class，model 也能在 planning 时避开它们（Figure 8 的 visualization 直接展示了这点）。
2. **Chain-of-thought reasoning 直接可用**：可以要求 model 先 explain rationale 再 output trajectory，这给了 explainability 和 performance boost 双重收益。
3. **Multi-task natural**：所有 task 都 share 同一个 language space，co-training 不需要 architecture 改动，只需要 task-specific prompt。

---

## 2. 方法细节

### 2.1 整体 formulation

最核心的 equation 是 Eq 1：

$$\mathbf{O} = \mathcal{G}(\mathbf{T}, \mathbf{V})$$

变量含义：
- $\mathcal{G}$：Gemini model（auto-regressive，处理 interleaved text 和 visual input）
- $\mathbf{T}$：natural language prompts（包括 routing command、ego history、task instruction）
- $\mathbf{V}$：images 或 videos（这里是 surround-view camera，stitched 成一张图或一段 video）
- $\mathbf{O} = (o_1, o_2, ..., o_n)$：output 是一串 token，通过 next-token prediction 生成：

$$P(\mathbf{O} | \mathbf{T}, \mathbf{V}) = \prod_{i=1}^{n} P(o_i | o_{<i}, \mathbf{T}, \mathbf{V})$$

这里 $o_{<i}$ 表示前面已经生成的 tokens，$n$ 是 output 长度。这是 standard autoregressive factorization。

**关键 design choice：坐标怎么表示？** 这是整篇 paper 最微妙的地方。他们考虑了两种方案：

- **方案 A（EMMA 采用）**：直接把 floating point number 写成 text，比如 BEV 坐标 $(9.01, 3.22)$ 就写成 text "9.01, 3.22"。这和 RT-2（Brohan et al., 2023, https://robotics-transformer2.github.io/）在 robotics control 里做的事情一样。优点是和 pre-trained language space 完全 compatible，缺点是 token 数量多（每个数字要分 digit）。
- **方案 B（不采用）**：用 special tokens 表示 discretized 坐标，类似 MotionLM（Seff et al., 2023, https://arxiv.org/abs/2309.16534）。优点是 token 紧凑，缺点是失去了和 pre-trained language representation 的 alignment。

EMMA 选 A 是为了 "maximally reuse the knowledge from pre-trained weights"。这个选择后面在 road graph ablation 里得到了一个有趣的验证：用 "(", ",", "and", ")" 这种 punctuation 比 raw "xy xy" 序列好 10%，说明 pre-trained language structure 真的在 work。

### 2.2 End-to-End Motion Planning（Eq 2）

$$\mathbf{O}_{\text{trajectory}} = \mathcal{G}(\mathbf{T}_{\text{intent}}, \mathbf{T}_{\text{ego}}, \mathbf{V})$$

三个 input：
1. $\mathbf{V}$：surround-view camera video
2. $\mathbf{T}_{\text{intent}}$：high-level routing command，比如 "go straight", "turn left", "turn right"。这相当于 Google Maps 给的 navigation instruction。
3. $\mathbf{T}_{\text{ego}} = \{(x_t, y_t)\}_{t=-1}^{-T_h}$：过去 $T_h$ 个 timestamp 的 ego vehicle waypoints（BEV 坐标，plain text 表示）。下标 $t$ 从 $-1$ 到 $-T_h$ 表示从最近一帧往回数。可以扩展加上 velocity、acceleration。

output：
$$\mathbf{O}_{\text{trajectory}} = \{(x_t, y_t)\}_{t=1}^{T_f}$$

未来 $T_f$ 个 timestamp 的 ego waypoints（同样 BEV plain text）。在 WOMD 上 $T_f$ 对应 8s，nuScenes 上对应 3s，internal dataset 上对应 5s。

这个 formulation 的三个特性 paper 强调了：
- **Self-supervised**：只要 future ego position 作为 label，不需要 human annotation
- **Camera-only**：不用 LiDAR、radar
- **HD map free**：不需要 HD map，只需要 routing intent

这和 BEV-Planner（Li et al., 2024, https://arxiv.org/abs/2310.20574）的 setup 几乎一样，区别是 EMMA 把 backbone 换成 Gemini。BEV-Planner 那篇 paper 之前的一个重要发现是：很多 open-loop end-to-end planning method（比如 UniAD）其实在 overfit ego status（历史 trajectory），即使 perception 模块崩了，planning metric 还是好的。EMMA 的 ablation 里 random init（Table 3 中 EMMA (random init) = 0.37m avg L2）vs Gemini init（0.32m）vs Gemini init + internal pretrain（0.29m），可以看出 pre-trained representation 确实在 work，不只是 ego status extrapolation。

### 2.3 Chain-of-Thought Reasoning（Eq 3）

这是 EMMA 最有意思的部分。他们把 driving rationale 拆成 4 个 coarse-to-fine 层次：

$$({O}_{\text{rationale}}, \mathbf{O}_{\text{trajectory}}) = \mathcal{G}(\mathbf{T}_{\text{intent}}, \mathbf{T}_{\text{ego}}, \mathbf{V})$$

其中 $\mathbf{O}_{\text{rationale}}$ 是 ordered text output $(R1, R2, R3, R4)$：

- **R1 - Scene description**：天气、time of day、road type、traffic situation。Example: "The weather is clear and sunny, and it is daytime. The road is four-lane undivided street with a crosswalk in the middle."
- **R2 - Critical objects**：识别影响 ego driving 的 agent，并要求精确的 3D/BEV 坐标。Example: "pedestrian at [9.01, 3.22], vehicle at [11.58, 0.35]"。这里把 visual grounding 嵌入 rationale，这是很关键的设计。
- **R3 - Behavior description**：critical objects 的 current status 和 intent。Example: "The pedestrian is currently standing on the sidewalk, looking toward the road, and maybe preparing to cross the street."
- **R4 - Meta driving decision**：12 个 category 的高层决策（Table 6 有完整 list，基于 0s/1s/3s 三个时间点的 speed 状态组合），比如 "Keep speed, then brake"。

这 4 个 rationale 组件是 **automated 生成的**，不需要额外 human label：
- Critical objects 来自 off-the-shelf perception/prediction expert models
- Scene 和 behavior description 来自 Gemini 配合 visual/text prompt
- Meta decision 用 heuristic 算法分析 ground-truth trajectory

inference 时 model 先 generate R1→R2→R3→R4 再 generate trajectory。但 paper 提到一个细节：训练后 prediction order (rationale, trajectory) vs (trajectory, rationale) 对 quality 影响不大，意味着 reasoning 和 trajectory 在 latent space 里是相互 informed 的，不一定需要 reasoning 在前。这对 real-time deployment 很重要——可以先输出 trajectory 再 early stop，rationale 留作 explainability 输出。

**CoT ablation（Table 4）的数据**：

| Scene desc | Critical obj | Meta decision | Behavior desc | 相对 baseline 提升 |
|---|---|---|---|---|
| ✓ | ✗ | ✗ | ✗ | +0.0% |
| ✗ | ✓ | ✗ | ✗ | +1.5% |
| ✗ | ✗ | ✓ | ✗ | +3.0% |
| ✗ | ✓ | ✓ | ✗ | +5.7% |
| ✗ | ✓ | ✓ | ✓ | +6.7% |

直觉解释：
- Scene description 单独没收益（+0%），因为 weather/time 这种全局 context 对 short-horizon planning 影响小，但它对 explainability 有价值
- Critical object 单独 +1.5%，因为 grounding 关键 agent 位置确实有用
- Meta decision 单独 +3.0%，收益最大，因为它直接 encode 了 speed profile 的 intent
- 组合起来 +6.7%，有 synergy 但不是线性叠加

这个 ablation 让我想到 DriveCoT（Wang et al., 2024b, https://arxiv.org/abs/2403.16996）和 DriveVLM（Tian et al., 2024, https://arxiv.org/abs/2402.12289）的 CoT 设计，但 EMMA 的 critical object grounding 是用 explicit 3D 坐标，而不是 image bounding box，这对 driving 更直接。

### 2.4 EMMA Generalist（Eq 4, 5, 6）

generalist 的关键在于用 task-specific prompt 来 switch behavior。

**3D Object Detection（Eq 4）**：

$$\mathbf{O}_{\text{boxes}} = \mathcal{G}(\mathbf{T}_{\text{detect\_3D}}, \mathbf{V})$$

每个 box 表示成 7D tuple：$(x, y, z, l, w, h, \theta, cls)$
- $(x, y, z)$：center location in vehicle frame
- $l, w, h$：length, width, height
- $\theta$：heading angle
- $cls$：class label in text

转 text 时每个 floating point 写到 2 decimal places，空格分隔。这借鉴了 Pix2Seq（Chen et al., 2022, https://arxiv.org/abs/2109.10852）的思路。

一个反直觉的发现：Pix2Seq 原文说 box order 不重要，但 EMMA 发现 **按 depth 排序 box 能显著提升 detection quality**。这可能是因为 Gemini 的 autoregressive nature，先输出近的 box 给后续远 box 提供了 context。

**Road Graph Estimation（Eq 5）**：

$$\mathbf{O}_{\text{roadgraph}} = \mathcal{G}(\mathbf{T}_{\text{estimate\_roadgraph}}, \mathbf{V})$$

road graph 表示成 polyline 的集合，每个 polyline 是一串 ordered waypoints。text 编码 example：`"(x1,y1 and x2,y2 and ... and xn,yn); ..."`，其中 `;` 分隔不同 polyline。

这个任务的 ablation（Figure 6）有几个非常有意思的发现，值得 build intuition：

1. **Dynamic sampling > Fixed sampling**（差 40-90%）：每条 lane 的 waypoint 数量根据 curvature 和 length 动态调整，而不是固定数量。直觉是：直道 2 个点就够了，弯道需要更多点才能 capture 曲率。保持 waypoint density 而不是 waypoint count。

2. **Ego-origin aligned sampling > Naive global sampling**（差 25-60%）：lane 在 global coordinate frame 里存的时候 origin 是任意的，直接 transform 到 ego frame 会导致 sample point 落在奇怪的位置。应该先在 ego frame 里 sample，这样 waypoint 相对 ego 的距离是 consistent 的。

3. **Shuffled ordering within distance bin**：把 lane 按离 ego 远近分 bin，bin 内随机 shuffle。这增强了 model 对 unordered output 的 robustness。

4. **Padding with "invalid" token**：类似 Pix2Seq，把 polyline 数量和每条 polyline 的点数都 pad 到 fixed length，用 "invalid" token 标记 padding，最后加 "valid"/"invalid" 标签。这避免了 training 时 premature truncation。

5. **Punctuation and semantic redundancy**：用 "(x,y and x,y);..." 比 "xy xy;..." 好 ~10%。这是 paper 里我最喜欢的一个 finding——它直接证明了 Gemini pre-trained 的 language structure 在 work，model 更适应 natural language 的 syntactic pattern。

**Scene Understanding（Eq 6）**：以 temporary blockage detection 为例：

$$\mathbf{O}_{\text{temporary\_blockage}} = \mathcal{G}(\mathbf{T}_{\text{temporary\_blockage}}, \mathbf{T}_{\text{road\_user}}, \mathbf{V})$$

prompt: "is the road ahead temporarily blocked?"

这个 task 用来测试 model 对 scene 的 holistic understanding，因为判断是否 block 需要综合多个 cue（construction、emergency vehicle、debris 等）。

### 2.5 Generalist Training

training 时按 dataset size 比例 sampling：

$$P(\text{task}) = \frac{|\mathbf{D}_{\text{task}}|}{\sum_t |\mathbf{D}_t|}$$

总 iteration 数 = $e \times \sum_t |\mathbf{D}_t|$，确保每个 task 被 $e$ 个 epoch 覆盖。

这个 simple mixture 策略 surprisingly 有效（Table 5）：

| e2e planning | 3D detection | road graph | planning 提升 | detection 提升 | road graph 提升 |
|---|---|---|---|---|---|
| ✗ | ✓ | ✓ | - | +1.6% | +2.4% |
| ✓ | ✗ | ✓ | +1.4% | - | +3.5% |
| ✓ | ✓ | ✗ | -1.4% | +5.6% | - |
| ✓ | ✓ | ✓ | +1.4% | +5.5% | +2.4% |

直觉解释：
- **Planning task 是 "hub" task**：detection 和 road graph 都从 co-training with planning 受益最多（detection +5.6%, roadgraph +3.5%），因为 planning 需要 holistic scene understanding，这个 signal 反过来 sharpen 了 perception。
- **Detection + Planning 的 synergy 最强**：detection 单独 +5.6%，因为 planning trajectory 直接 informs model 哪些 object 是 critical 的，相当于 free attention supervision。
- **Detection 和 road graph 一起 train（不带 planning）反而让 planning 略降**（-1.4%）：这暗示 planning 不仅是 perception 的简单组合，它需要自己的 supervision signal 才能学好。
- **三任务 co-train 最佳**：detection +5.5%, roadgraph +2.4%, planning +1.4%，全部 positive。

这个 finding 和 UniAD multi-task training 的 spirit 一致，但 EMMA 的 unification 更深——所有 task share 同一个 decoder，而 UniAD 还有 separate task head。

---

## 3. 实验数据深度解析

### 3.1 WOMD Planning（Table 2）

| Method | L2(m) 1s | L2(m) 3s | L2(m) 5s |
|---|---|---|---|
| MotionLM* | 0.045 | 0.266 | 0.696 |
| Wayformer* | 0.046 | 0.252 | 0.628 |
| EMMA | 0.032 | 0.248 | 0.681 |
| EMMA (w/ CoT) | 0.030 | 0.241 | 0.664 |
| EMMA+ | 0.030 | 0.225 | 0.610 |
| **EMMA+ (w/ CoT)** | **0.027** | **0.203** | **0.543** |
| EMMA† (PaLI) | 0.034 | 0.274 | 0.797 |
| EMMA+† (PaLI) | 0.031 | 0.239 | 0.680 |

几个观察：
1. EMMA+ (w/ CoT) 在 5s horizon 上比 MotionLM 好 **22%**（0.543 vs 0.696），这是个非常显著的 gap
2. CoT 在 EMMA+ 上的增益（0.610 → 0.543 = 11%）比在 EMMA 上（0.681 → 0.664 = 2.5%）大得多，说明 CoT 需要足够 data 才能发挥
3. MotionLM 和 Wayformer 用了 LiDAR-based offboard perception model 生成的 agent boxes、人工 road graph、traffic light states，而 EMMA 只用 camera + ego history，pure self-supervised。这个对比很有说服力。
4. PaLI 版本（EMMA†）比 Gemini 版本差一些，但依然超过 MotionLM，说明 method 不 strongly tied to specific MLLM

**Multi-trajectory sampling（Figure 3）**：Top-K sampling 最多 K=24 个 trajectory，然后取 pairwise L2 距离最小的 "median" trajectory。这个 trick 让 ADE@5s 持续提升但 12+ 之后 diminishing return。MotionLM/Wayformer 是 sample 192 个 trajectory 然后 k-means 聚成 6 类。EMMA 用更少 sample 但更聪明的 selection（median vs cluster center）。

### 3.2 nuScenes Planning（Table 3）

| Method | Self-sup? | L2 1s | L2 2s | L2 3s | Avg L2 |
|---|---|---|---|---|---|
| UniAD | ✗ | 0.42 | 0.64 | 0.91 | 0.66 |
| DriveVLM | ✗ | 0.18 | 0.34 | 0.68 | 0.40 |
| VAD | ✗ | 0.17 | 0.34 | 0.60 | 0.37 |
| OmniDrive | ✗ | 0.14 | 0.29 | 0.55 | 0.33 |
| Ego-MLP* | ✓ | 0.15 | 0.32 | 0.59 | 0.35 |
| BEV-Planner | ✓ | 0.16 | 0.32 | 0.57 | 0.35 |
| EMMA (random init) | ✓ | 0.15 | 0.33 | 0.63 | 0.37 |
| EMMA | ✓ | 0.14 | 0.29 | 0.54 | 0.32 |
| **EMMA+** | ✓ | **0.13** | **0.27** | **0.48** | **0.29** |

关键对比：
- EMMA vs BEV-Planner（同样 self-supervised）：**17.1% 提升**（0.35 → 0.29）
- EMMA vs OmniDrive（用了大量 perception human label）：**12.1% 提升**（0.33 → 0.29）
- EMMA (random init) vs EMMA (Gemini init)：0.37 vs 0.32，说明 pre-trained weight 贡献 ~13.5%
- EMMA vs EMMA+：0.32 vs 0.29，internal 24M scene pretrain 贡献 ~9.4%

注意 nuScenes 上 multi-trajectory sampling 没收益，paper 推测是因为 3s horizon 太短、scenario 太简单，top-1 已经足够。这和 WOMD 上 8s 需要 sampling 形成对比。

### 3.3 Internal Dataset Scaling（Figure 4）

data scaling curve 显示 4 个 data 比例（3%, 10%, 30%, 100% of 24M examples）的 eval perplexity vs training FLOPs。结论：
1. 更大 dataset 持续降低 best achievable perplexity
2. 小 dataset overfit 很快
3. **24M dataset 上 quality 还没 saturate**

这一点很重要：意味着如果 Waymo 把 dataset 扩到 100M+，quality 可能继续提升。这是 scaling law 在 autonomous driving 上的直接 evidence。

### 3.4 WOD 3D Detection（Figure 5, 11）

EMMA+ vs BEVFormer：
- Vehicle precision @ same recall: **+16.3%**
- Vehicle recall @ same precision: **+5.5%**
- F1 score: EMMA+ 更高

按距离 breakdown（Figure 11）：近距离（<30m）EMMA+ 大幅领先，远距离（>50m）优势消失。Paper 把这归因于 camera input resolution 限制——Gemini 的 vision encoder 是为 general image 设计的，不是为高分辨率 driving 场景优化的。这暗示未来用 higher resolution camera input 或 specialized vision encoder 可能进一步提升远距离 performance。

### 3.5 Road Graph Ablation（Figure 6）

最 dramatic 的是 dynamic vs fixed sampling（70-90% 改变）和 ego-origin alignment（25-60% 改变）。这两个 finding 其实是 **representation engineering** 的胜利——如何把 continuous 几何信息编码成 text token 序列，对 quality 影响巨大。

### 3.6 Scene Understanding（Figure 7）

temporary blockage detection：
- 直接 fine-tune：超过 naive human baseline，但不如 human + filtering（过滤掉 'unsure' 答案）
- Naive co-train with road graph：没提升
- Pre-train on road graph then co-train（short）：有提升
- Pre-train on road graph then co-train（long）：显著提升

这个 finding 很有意思：naive mixture 不行，需要 long pre-training 才能 transfer。这暗示 task 之间的 transfer 不是 symmetric 的，需要先 build 一个 strong representation（road graph 提供 spatial structure）再 add scene understanding head。

---

## 4. Limitations 和我自己的思考

paper Section A.5 诚实讨论了几个 limitation，我从 Karpathy 视角补充一些直觉：

1. **Long-term memory**：目前只 process 4 帧。Driving 需要 reasoning over 多秒甚至分钟级别的 context（比如 remembering 远处的 construction sign）。这需要 memory module（类似 RAG 或 recurrent state）或者更长的 video understanding 能力。Gemini 1.5 Pro 已经支持 million token context，未来版本可能直接解决。

2. **LiDAR/Radar fusion**：MLLM pre-training 主要在 image/text 上，3D sensor encoder 还没达到 camera encoder 的 scale。这是一个 chicken-and-egg 问题：要 pre-train 好 3D encoder 需要 large 3D dataset，但 large 3D dataset 又依赖 sensor deployment。可能的解法是 camera-LiDAR contrastive pre-training（类似 CLIP）来 align 两个 modality。

3. **Closed-loop evaluation**：open-loop metric（L2, collision rate）被 BEV-Planner 等 paper 证明可以 gamed。NAVSIM（Dauner et al., 2024, https://arxiv.org/abs/2406.15300）是更好的方向，但真正的 closed-loop 需要 sensor simulation，computation cost 极高。EMMA 的 closed-loop evaluation 还没做，这是一个重要的 future work。

4. **Real-time deployment**：Gemini 1.0 Nano-1 latency 在 cloud 上 OK，但 onboard deployment 需要 distillation 或 SARA-RT（Leal et al., 2024, https://arxiv.org/abs/2312.07563）这种 self-adaptive attention 加速。paper 提到一个 latency-optimized variant 能做到 3 FPS，已经比 UniAD 的 1.8 FPS 快 67%，但离 real-time 10+ Hz 还有距离。

5. **Verification & consistency**：generalist model 可以同时输出 trajectory、detection、road graph，但没有机制保证它们 mutually consistent。比如 trajectory 可能 indicate 一条 lane change，但 detection 没看到 adjacent lane 的车。CoT 部分 mitigate 但没根本解决。这是一个 safety-critical 的问题。

6. **Distribution shift & OOD**：paper 展示了 squirrel、dog、ladder、garbage bag 等 OOD object 的成功 case，但没系统性地 evaluate OOD robustness。Pre-trained MLLM 的 world knowledge 是双刃剑：它 generalize 好，但也可能 confidently hallucinate（比如把 shadow 当成 obstacle）。

---

## 5. 连接到 broader research context

### 5.1 和其他 MLLM-for-driving 工作的关系

- **DriveGPT4**（Xu et al., 2024, https://arxiv.org/abs/2402.12281）：用 LLM explain action，但是 modular 的
- **LMDrive**（Shao et al., 2024, https://arxiv.org/abs/2404.05466）：closed-loop LLM driving，但不是 end-to-end fine-tune
- **DriveVLM**（Tian et al., 2024, https://arxiv.org/abs/2402.12289）：VLM + driving，但是用 VLM 做 scene understanding 然后 feed 给 traditional planner
- **OmniDrive**（Wang et al., 2024a, https://arxiv.org/abs/2405.01533）：3D vision-language model，更接近 EMMA 但 architecture 不同
- **DriveCoT**（Wang et al., 2024b, https://arxiv.org/abs/2403.16996）：CoT for driving，但 reasoning chain 设计不同
- **DriveLM**（Sima et al., 2024, https://arxiv.org/abs/2312.14115）：graph VQA for driving

EMMA 的独特之处在于：**Gemini 作为 first-class citizen**，而不是 auxiliary reasoner。所有 task 都在同一个 model 内 solve，没有 specialized perception head 或 planner module。

### 5.2 和 Robotics 的 connection

EMMA 和 RT-2（https://robotics-transformer2.github.io/）思路几乎一样：把 continuous action/coordinate 表示成 text token，然后 fine-tune pre-trained VLM。这种 "language as universal interface" 范式正在 robotics、driving、UI navigation（like WebVoyager）等多个领域 converge。

更广义地，这反映了 **foundation model 时代的 paradigm shift**：从 task-specific architecture 转向 generalist model + task-specific prompt。Gato（Reed et al., 2022）、PaLM-E（Driess et al., 2023, https://palm-e.github.io/）、Gemini Robotics 都是这条线上的工作。

### 5.3 Open questions 我觉得值得探索

1. **Tokenization of continuous output**：text 表示 floating point 真的是最优吗？每 digit 一个 token 太浪费。VQ-VAE 风格的 learned codebook 可能更 compact，但要和 language pre-training align 是 tricky 的。
2. **3D reasoning 的根本限制**：纯 camera input 在远距离 depth estimation 上有物理限制。如何 elegantly fuse LiDAR 进 MLLM 框架？
3. **Closed-loop training**：目前所有实验都是 open-loop imitation learning。RL fine-tuning in closed-loop simulator 可能 unlock 下一级 performance，但 sensor simulation cost 是 bottleneck。
4. **Active perception**：人类 driver 会主动转头看盲区。MLLM 能否 learn 主动 query 不同 camera 或请求更高 resolution region？这和 attention mechanism 设计有关。
5. **Long-horizon planning**：5s prediction horizon 不够做复杂 maneuver（比如 lane change 需要 10s+）。是否需要 hierarchical planning（route-level → maneuver-level → trajectory-level）？
6. **Safety verification**：如何 formally verify MLLM-based planner 的 safety property？这是 deployment blocker。

---

## 6. 总结直觉

EMMA 这篇 paper 给我的最大启发是：**当你有一个足够 strong 的 pre-trained MLLM，很多原本认为需要 specialized architecture 的 task，其实可以 cast 成 language generation**。Autonomous driving 看起来是高度 geometric 和 physical 的 task，但本质上它需要的是 scene understanding + reasoning + decision，这三件事 MLLM 都很擅长。

这个发现的意义超出 driving 本身——它预示着 **generalist foundation model 可能在很多 "看似需要 specialization" 的 domain 都能 competitive**，只要你能找到正确的 input/output representation。EMMA 的 representation choice（BEV 坐标 as plain text、polyline as "((x,y and x,y));..."）看似 ad-hoc，但 ablation 证明这些 choice 是重要的，因为它们 leverage 了 pre-trained language structure。

Paper 也诚实承认这不是终点：LiDAR fusion、closed-loop eval、real-time deployment、long-horizon reasoning 都是 open problem。但作为一个 "MLLM as first-class citizen for driving" 的 proof-of-concept，它非常 convincing。

我个人的预测：未来 2-3 年内，我们会看到 EMMA-style architecture 在 closed-loop benchmark 上 surpass modular stack，并且 onboard deployment 会通过 distillation + quantization + specialized attention 实现。真正的 bottleneck 不是 model architecture，而是 **closed-loop evaluation infrastructure** 和 **safety verification**。

---

Reference links:
- EMMA OpenReview: https://openreview.net/forum?id=kH3t5lmOU8
- EMMA arXiv: https://arxiv.org/abs/2410.23262
- Waymo blog: https://waymo.com/blog/2024/10/introducing-emma/
- Gemini paper: https://arxiv.org/abs/2312.11805
- RT-2: https://robotics-transformer2.github.io/
- Pix2Seq: https://arxiv.org/abs/2109.10852
- MotionLM: https://arxiv.org/abs/2309.16534
- Wayformer: https://arxiv.org/abs/2207.05814
- BEV-Planner: https://arxiv.org/abs/2401.01986
- UniAD: https://arxiv.org/abs/2212.10156
- VAD: https://arxiv.org/abs/2303.12077
- OmniDrive: https://arxiv.org/abs/2405.01533
- DriveVLM: https://arxiv.org/abs/2402.12289
- DriveCoT: https://arxiv.org/abs/2403.16996
- DriveLM: https://arxiv.org/abs/2312.14115
- LMDrive: https://arxiv.org/abs/2404.05466
- DriveGPT4: https://arxiv.org/abs/2402.12281
- PaLM-E: https://palm-e.github.io/
- SARA-RT: https://arxiv.org/abs/2312.07563
- NAVSIM: https://arxiv.org/abs/2406.15300
- BEVFormer: https://arxiv.org/abs/2203.17270
- WOD: https://arxiv.org/abs/1912.04838
- WOMD: https://arxiv.org/abs/2403.20326（WOMD-Lidar）
- Chain-of-Thought paper: https://arxiv.org/abs/2201.11903
- PaLI-X: https://arxiv.org/abs/2305.18565
- ChauffeurNet: https://arxiv.org/abs/1812.03079
- DAVE-2: https://arxiv.org/abs/1604.07316
- ALVINN: https://papers.nips.cc/paper/1988

如果你想 drill deeper 进任何一个 section（比如 road graph 的 polyline encoding 细节、CoT 的具体 prompt template、或 generalist co-training 的 task sampling 策略），我可以再展开。
