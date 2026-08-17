---
source_pdf: QUANTIPHY.pdf
paper_sha256: bf80d110c59702360c656cdb49eb1ab5e6993125e9a9842d7c1c11de33c58786
processed_at: '2026-08-06T07:32:46-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 QUANTIPHY

## 一句话版本

**给 VLM 看一段视频,告诉它"这车长 5 米",问它"车速多少"——结果发现最强的模型也就跟普通人差不多,而且它根本没在看视频,是在瞎猜。**

---

## 背景故事

你有没有过这种体验:看一个人走路,你能大致猜出他 1.7 米左右,走速大概 1.2 m/s?你不需要拿尺子量,脑子里有个"常识库"。

现在的 VLM(GPT-4o、Gemini 这些)其实也是这么干活的。你问它"视频里这车多长",它不是真的去数像素,而是想:"嗯,车,一般 4-5 米吧"——然后给你一个看起来合理的数字。

问题是:**它到底是在"看"视频,还是在"背"常识?**

之前没人能回答这个问题,因为所有 benchmark 都是选择题。选择题有个致命缺陷:答案 3.1 米和 31 米都算错,但 31 米比 3.1 米离谱十倍。选择题区分不了这个差距。

QUANTIPHY 的核心 idea 就是:**别出选择题了,让模型直接吐数字,然后看它吐的数字跟 ground truth 差多少。**

---

## 任务到底长啥样

给你一段 2-3 秒的视频,比如一辆黄车从左开到右。然后我告诉你:

> "这车长 5.67 米"

问你:

> "这车在 2 秒时速度多少?车宽多少?"

就这么简单。理论上,你只需要:
1. 数一下车在视频里占多少像素(比如 135 像素长)
2. 算出比例:5.67m / 135px ≈ 0.042 m/px
3. 数车 2 秒时挪了多少像素,乘以比例,得到速度
4. 量车宽多少像素,乘以比例,得到实际宽度

**这是初中物理加小学算术。** 一个有像素级 access 的理想 agent 应该能几乎完美完成。

---

## 结果:所有人都翻车了

| 谁在答 | 得分 |
|--------|------|
| 普通人 | 55.6 |
| ChatGPT-5.1(最强模型) | 53.1 |
| Gemini-2.5 Pro | 49.6 |
| Claude Sonnet 4.5 | 22.8(惨) |

**最强的模型跟普通人差不多,没有一个超过人类平均。**

但关键是——人类不是理论上限。人类只能粗略估("那车大概 4 米吧"),而 VLM 有像素级精确 access,理论上应该秒杀人类。结果没有。

**VLM 根本没在用它"看"到的像素。**

---

## 三个杀手实验

### 实验一:把视频拿掉

如果模型真的在看视频,那把视频拿掉后应该暴跌。结果呢?

ChatGPT-5.1: 有视频 56.1 → 没视频 39.0

**掉了一点,但没掉到地板。** 意思是:就算不给它看视频,只告诉它"有辆车,长 5.67 米",它也能猜出个差不多对的答案。

为什么?因为它训练时见过几百万张车的图,它知道车大概长啥样、大概跑多快。**它是在"背答案",不是在"解题"。**

### 实验二:给它一个离谱的 prior(最精彩)

把 prior 改一下:不说"车长 5.67 米",说"车长 5670 米"(放大 1000 倍)。

如果模型真在 reasoning,它应该想:"哦,prior 变了 1000 倍,那我算出来的速度和宽度也该变 1000 倍。"

结果呢?**几乎所有模型都直接崩了,MRA 掉 70-80%。**

ChatGPT-5.1: 56.1 → 15.4

模型看到"车长 5670 米"后的 thinking trace 大概是:
> "5670 米?这不对吧,车哪有这么长?不管了,车一般就 4-5 米,我就按 4.5 米算吧..."

**它直接无视了你给的 prior,转头用自己记忆里的"常识"去猜。**

这就坐实了:VLM 不是 input-faithful reasoner,它是被 parametric prior 绑架的 guesser。你给它什么数字,它根本不在乎,它只在乎"车应该多长"。

### 实验三:给它 CoT 让它一步步算

论文作者心想:也许模型是跳步了?那我把步骤拆开,一步一步引导:

> Step 1: 先告诉我车在视频里多少像素长?
> Step 2: 算一下像素和米的比例?
> Step 3: 再告诉我目标的像素值?
> Step 4: 最后算出实际值?

结果:**21 个模型里只有 3 个变好,其他 18 个反而变差了。**

为什么?因为 **VLM 根本不会数像素**。让它说"车在视频里多少像素长"它就懵了,然后这个错误一路传到后面,越错越离谱。

CoT 不是万能药。当中间步骤本身就超出模型能力时,CoT 只会放大错误。

---

## 一个特别扎心的例子

论文里有个 case:Blender 模拟了一个篮球场景,但故意把重力设成 1 m/s²(不是地球的 9.8)。问模型球的加速度。

ChatGPT-5.1 直接输出:**9.8 m/s²**

它看都没看视频里的运动轨迹,直接背了标准重力常数。**你设的物理规则在它眼里不存在,它只知道"地球上东西往下掉是 9.8"。**

---

## 数据怎么来的

三个来源:

1. **Blender 模拟(300 个视频)**:完全可控,ground truth 精确。有个聪明的 trick——先画一条好看的轨迹,然后用 Python 重新参数化,让它符合 $s(t) = vt$ 或 $s(t) = \frac{1}{2}at^2$。这样既好看又物理正确。

2. **实验室拍摄(112 个视频)**:4 台 Orbbec ToF 深度相机,多视角重建 3D 轨迹。放弃了 FoundationPose 这类 AI 模型(在遮挡下不稳),用纯几何方法。

3. **网上爬的(72 个视频)**:只有 2D,因为没有 calibrated depth。用信用卡、车道线这些已知尺寸的参照物来标定。

还有 85 个视频是拿 SAM 2 把背景抠掉的,用来测"背景复杂度对推理有没有影响"。

---

## 评分怎么算的(MRA)

不用简单的"对/错",而是一组 tolerance threshold:

> 在 10% 误差内算对一次,20% 算对一次......95% 算对一次,取平均。

$$\text{MRA} = \frac{1}{10} \sum_{\theta \in \{0.1, 0.2, ..., 0.95\}} \mathbb{1}\left(\frac{|\hat{y} - y|}{|y|} < 1 - \theta\right)$$

$\hat{y}$ 是模型预测,$y$ 是 ground truth,$\theta$ 是置信度阈值。$\theta = 0.95$ 意味着只允许 5% 误差,很严;$\theta = 0.1$ 意味着允许 90% 误差,很松。十个档平均下来就是最终分。

比 MSE 好的地方:不怕 outlier 把分数拉爆;比 accuracy 好的地方:不会因为一个超大的 error 就判零分。

---

## 为什么背景复杂反而得分高?

有个反直觉的发现:**复杂背景 > 简单背景 > 纯色背景。**

按理说背景越乱越干扰。但论文的解释是:复杂背景里有参照物——地砖、窗户、车道线,模型拿这些当"隐式尺子"用。

这恰好进一步印证了核心发现:模型靠的是 contextual cue,不是 pixel-level measurement。

---

## 对我们的启示

1. **VLM 的"reasoning"大部分是 pattern matching。** 你给的 input 它不一定真用,它更信自己的记忆。

2. **Scale 有帮助但有上限。** Qwen3-VL 从 2B 到 32B 涨了不少,但离人类还有差距,更别说理论极限。

3. **CoT 不是万能解。** 当中间步骤超出能力时,分解只会让错误传播。

4. **要做真正的 physical reasoning,可能需要 hybrid architecture**——neural + symbolic,或者让 VLM 能调用外部工具(计算器、OpenCV)来干精确活。

5. **Counterfactual probe 是个绝佳 diagnostic 工具。** 以后所有 benchmark 都该加这个——不测你猜得对不对,测你到底有没有在用 input。

---

## 一句话总结这篇 paper 到底干了什么

**它造了一个让 VLM 算物理量的考场,然后发现学霸们全在作弊——不是在看题解题,是在背答案。而且你把题改了它们都懒得看,继续背原来的答案。**

参考链接:
- 项目主页: https://quantiphy.stanford.edu/
- 代码: https://github.com/Paulineli/QuantiPhy
- 数据集: https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation

---

# QUANTIPHY 深度解析：量化 VLM 物理推理的 First Principles 视角

## 1. Motivation：为什么这是个真正重要的问题

物理世界的定量理解是 generalist AI agent 的核心瓶颈。Karpathy 你应该最有感触——我们训练的 VLM 本质上是在拟合大规模数据分布，但这些数据隐式包含的 Newtonian dynamics、kinematic relationships 究竟被模型学到了什么程度？现有 benchmark（PhysBench、STAR、CLEVRER、Physion++）几乎全部采用 VQA 范式，ground truth 被选项约束，无法区分 "3.1m" 和 "31m" 这种 10x 量级的错误。QUANTIPHY 的核心 insight 在于：**物理推理的真正测试必须落在 numerical accuracy 上，而非 qualitative plausibility 上**。

这让我想到一个更深层的问题：当前 VLM 的 "physical reasoning" 到底是 reasoning 还是 pattern matching？这篇 paper 用 counterfactual probe 给出了相当有说服力的答案——**VLMs 是 powerful guessers conditioned on textual hints，远非 faithful visual measurers**。

参考链接：
- Project page: https://quantiphy.stanford.edu/
- Dataset: https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation
- Code: https://github.com/Paulineli/QuantiPhy
- PhysBench: https://physbench.github.io/
- VSI-Bench (启发 MRA metric 的工作): https://arxiv.org/abs/2412.14171

## 2. 任务形式化定义：从 pixel space 到 world space 的 rescaling

这是整篇 paper 数学上最精彩的部分，值得仔细拆解。

### 2.1 Pixel space kinematics

给定固定相机拍摄的视频，物体在时刻 $t$ 的 pixel-space 位置记为 $\mathbf{X}_t^{\mathrm{pixel}} \in \mathbb{R}^2$。通过 finite differences 计算 velocity 和 acceleration：

$$\mathbf{V}_t^{\mathrm{pixel}} \approx \frac{\mathbf{X}_{t+\mathrm{d}t}^{\mathrm{pixel}} - \mathbf{X}_t^{\mathrm{pixel}}}{\mathrm{d}t}$$

$$\mathbf{A}_t^{\mathrm{pixel}} \approx \frac{\mathbf{X}_{t+2\mathrm{d}t}^{\mathrm{pixel}} - 2\mathbf{X}_{t+\mathrm{d}t}^{\mathrm{pixel}} + \mathbf{X}_t^{\mathrm{pixel}}}{\mathrm{d}t^2}$$

变量解释：
- $\mathbf{X}_t^{\mathrm{pixel}}$：物体在时刻 $t$ 的 pixel-space 位置向量，上标 `pixel` 表示单位是 [pixel]
- $\mathrm{d}t$：相邻帧之间的时间间隔，等于 $1/\text{fps}$
- $\mathbf{V}_t^{\mathrm{pixel}}$：pixel-space velocity，单位 [pixel/s]
- $\mathbf{A}_t^{\mathrm{pixel}}$：pixel-space acceleration，单位 [pixel/s²]
- 第二个公式是经典的二阶中心差分（second-order central difference），相比前向差分能减少 noise 放大

这个公式本身没什么神秘——它就是离散版本的导数定义。关键 insight 在于：**video 本身只定义了 pixel-space kinematics，world-space 信息完全丢失**。

### 2.2 Scale factor $\gamma$：连接两个空间

论文引入 scalar scale factor $\gamma > 0$，单位是 [world length / pixel]：

$$\mathbf{S}^{\mathrm{world}} = \gamma \mathbf{S}^{\mathrm{pixel}}$$
$$\mathbf{V}_t^{\mathrm{world}} = \gamma \mathbf{V}_t^{\mathrm{pixel}}$$
$$\mathbf{A}_t^{\mathrm{world}} = \gamma \mathbf{A}_t^{\mathrm{pixel}}$$

这里的关键假设是 **沿 motion direction 的 uniform scaling**——这是 2D planar motion 的自然结果，但对于 3D motion（涉及 depth 变化）需要 depth info 作为额外 prior。

**为什么这个公式如此重要**：它把整个 quantitative kinematic inference 归约成了一个简单的 algebraic operation。给定任何一个 world-space prior（比如 object size $S^{\mathrm{world}}$），加上对应 pixel-space 测量（$S^{\mathrm{pixel}}$），就能解出 $\gamma = S^{\mathrm{world}} / S^{\mathrm{pixel}}$，然后任何其他 world-space 量都能通过 rescaling 得到。

**理论上 VLM 应该能完美完成这个任务**——只需要： identification of relevant pixels, (2) finite difference, (3) 一次 division 得到 $\gamma$, (4) 一次 multiplication 得到 target。但实验显示，最好的模型（ChatGPT-5.1）也只有 53.1% MRA，**远低于理论 ceiling**。

这个 gap 是整篇 paper 的核心发现。

### 2.3 三轴 benchmark 设计

QUANTIPHY 沿三个 axis 组织数据：

| Axis | 取值 | 含义 |
|------|------|------|
| Dimensionality | 2D / 3D | 2D: 物体在 image plane 平行平面运动，depth 不变；3D: 涉及 z-axis，depth 变化 |
| Physical prior | Static / Dynamic | Static: 提供 object size $S^{\mathrm{world}}$（常数）；Dynamic: 提供 velocity $\mathbf{V}_t^{\mathrm{world}}$ 或 acceleration $\mathbf{A}_t^{\mathrm{world}}$ at timestamp $t$ |
| Scene difficulty | SX/MX/SS/MS/SC/MC | 第一个字母：S (single object) / M (multiple objects)；第二个字母：X (plain) / S (simple texture) / C (complex) |

四个核心 task category：**2D-Static, 2D-Dynamic, 3D-Static, 3D-Dynamic**。

总共 569 unique videos，3355 questions，平均每个 video 配 ~6 个 question。Video 时长 2-3 秒，磁盘占用 ~115MB——对硬件友好。

## 3. Benchmark 构建的三源数据 pipeline

这部分 paper 写得相当详细，我来拆解每条 pipeline 的技术细节。

### 3.1 Blender Simulation (300 videos)

这是最可控的 source。论文用了两种 motion paradigm：

**Keyframed Motion**：通过 rigged skeleton 或手画 animation curve 定义运动，visually plausible 但 **physics 不保证**。比如 lunar-walking 场景，astronaut 的 push-off/airtime/landing 是 artist 编辑的 curve，没用 Moon 的重力常数算。这点非常重要——意味着如果 VLM 用 pre-trained gravitational prior，会在这类场景上 hallucinate。

**Physics-Driven Motion**：通过 Blender 物理引擎（rigid body dynamics）直接模拟，比如 bowling ball 撞 pins。这类 video 的 ground truth 是 exact 的。

论文用 Python script 自动化 motion 生成。一个核心 trick 是 **arc-length reparameterization**：

```python
# 采样原始动画曲线
positions = [obj.matrix_world.translation for f in frames]
# 累积弧长
distances = cumsum([||p_i - p_{i-1}||])
# 重新参数化：用 s(t) = v*t 或 s(t) = v_0*t + 0.5*a*t^2 沿曲线移动
target_dist = speed * t  # 或 0.5 * a * t^2
# 找到对应 segment，线性插值
ratio = (target_dist - distances[i-1]) / (distances[i] - distances[i-1])
new_loc = positions[i-1].lerp(positions[i], ratio)
```

这个 trick 让作者可以 **保留 authored trajectory 的视觉丰富性，同时强加物理上 well-defined 的 kinematic profile**。非常聪明的设计。

另一个有意思的细节是 **uniform scaling 处理**：当 real-world scale 物体在视频里几乎不可见（比如 ice cube 落入 cup 只占 1 frame），作者对整个 scene uniform scale 10x，**保留物理关系但增加 perceptual visibility**。这其实是在 measure "VLM 能否从 visual kinematics 推回 physical quantities"，而不是 "VLM 是否 memorize 了 standard object sizes"。

### 3.2 Lab Capturing (112 videos, 仅 3D)

使用 **4 台 Orbbec Femto Mega camera** 构造 multi-view stereo 系统。相机规格：
- Depth technology: Time of Flight (ToF)
- Wavelength: 850nm
- Depth range: 0.25-5.46m
- Depth resolution: up to 1024×1024 @ 15fps (WFOV) 或 640×576 @ 30fps (NFOV)
- RGB: up to 3840×2160 @ 25fps
- 处理器: NVIDIA Jetson Nano

annotation pipeline：**纯几何方法**，放弃 FoundationPose 等 6D pose estimation 模型（在 occlusion 下不稳定）。具体流程：
1. Multi-view calibration with checkerboard
2. 主相机 RGB + aligned metric depth
3. UI tool 让 annotator 在 target object center 点击，读取对应 metric depth
4. Back-project 到 world coordinates using calibrated intrinsics
5. Finite difference 计算 velocity / acceleration

公式（lab data）：
$$v_k^{\mathrm{world}} \approx \frac{x_{k+1}^{\mathrm{world}} - x_k^{\mathrm{world}}}{\Delta t}$$
$$a_k^{\mathrm{world}} \approx \frac{x_{k+1}^{\mathrm{world}} - 2x_k^{\mathrm{world}} + x_{k-1}^{\mathrm{world}}}{\Delta t^2}$$

变量解释：
- $x_k^{\mathrm{world}}$：第 $k$ 帧物体在 world coordinate 的位置
- $\Delta t$：frame interval = 1/fps
- 至少在 5 个相邻 frame 上 annotate 用于 smoothing

### 3.3 Internet Scraping (72 videos, 仅 2D)

最 challenging 的 source，因为缺少 calibrated depth / camera intrinsics。论文限制 internet video 只能做 2D inference。Annotation 用 **pixel ruler + reference object**：

$$\gamma = \begin{cases} \frac{\mathbf{S}^{\mathrm{world}}}{\mathbf{S}^{\mathrm{pixel}}} & \text{if size prior} \\ \frac{|\mathbf{V}_{t_0}^{\mathrm{world}}|}{|\mathbf{V}_{t_0}^{\mathrm{pixel}}|} & \text{if velocity prior} \\ \frac{|\mathbf{A}_{t_0}^{\mathrm{world}}|}{|\mathbf{A}_{t_0}^{\mathrm{pixel}}|} & \text{if acceleration prior} \end{cases}$$

然后任何 target 量通过 $\gamma \cdot (\cdot)^{\mathrm{pixel}}$ rescaling 得到 world-space value。

参考 object 通常是：credit card (85.6mm × 53.98mm), lane width, conveyor belt speed, gravity $g = 9.8 m/s^2$。

### 3.4 Segmented Data (85 videos)

用 **SAM 2 + Grounding DINO 1.5** 自动 segment 前景 object，替换 background。这创造了 controlled experiment 测试 background complexity 影响。

## 4. Evaluation Metric: Mean Relative Accuracy (MRA)

### 4.1 公式

$$\mathrm{MRA} = \frac{1}{10} \sum_{\theta \in \mathcal{C}} \mathbb{1}\left(\frac{|\hat{y} - y|}{|y|} < 1 - \theta\right)$$

变量解释：
- $\hat{y}$：model prediction
- $y$：ground truth
- $\theta$：confidence threshold，取自 $\mathcal{C} = \{0.1, 0.2, \ldots, 0.9, 0.95\}$（10 个值）
- $\mathbb{1}(\cdot)$：indicator function，条件满足返回 1，否则 0
- $1 - \theta$：相对误差 tolerance，$\theta$ 越大 tolerance 越严

**Intuition**：MRA 是一组 binary accuracy 的平均，每个 accuracy 对应一个 tolerance threshold。比如 $\theta = 0.5$ 意味着 tolerance 是 50% 相对误差；$\theta = 0.95$ 意味着 5% 相对误差。最终 MRA ∈ [0, 1]，越高越好。

### 4.2 为什么用 MRA 而不是 MAPE / MSE？

论文给出了几条理由，我来扩展一下：

1. **Calibrated discretization**：类似 object detection 中 mAP@IoU 的思想。不直接用 continuous relative error（容易被 outlier 主导），而是分级评分。
2. **Robustness to ambiguity**：物体边界定义有歧义（比如 hair 算不算 person height）。MRA 在合理 tolerance 内给 full credit。
3. **Measurement noise tolerance**：temporal aliasing, motion blur, imprecise priors 都会引入 noise。MRA 比回归 loss 更宽容。
4. **Precedent**：VSI-Bench [Yang et al. 2025] 已经验证 MRA 是 stable and discriminative 的 metric，后续工作 [Holistic Eval, Video Reasoning without Training] 也采用了。

**值得吐槽**：MRA 仍然有局限。比如 $\hat{y} = 0$（model 输出 0）在 $y \neq 0$ 时 relative error = 1，所有 threshold 都 fail，MRA = 0。这其实惩罚过重——一个 model 输出 0（比如 API error）和一个输出完全无关数字的 model 同样得 0 分。论文 supplementary A.3 提到了这个问题，把 API failure 当 MRA=0 处理。这可能会拉低 model 的真实 ranking，特别是 API 不稳定的 model。

### 4.3 Aggregation

- Question-level MRA → category-level MRA（对 4 个 category 分别平均）
- Overall score = 4 个 category MRA 的 unweighted mean
- 如果 model 在 5 次 retry 后都没输出 parseable number，记为 fail（MRA=0）

## 5. 实验结果深度分析

### 5.1 Main results (Table 1)

让我把关键数字梳理一下：

| Model | Size | 2S | 2D | 3S | 3D | Avg |
|-------|------|----|----|----|----|-----|
| **Human baseline** | - | 50.0 | 59.1 | 55.2 | 57.9 | **55.6** |
| ChatGPT-5.1 | - | 46.3 | 56.2 | 51.5 | 58.3 | **53.1** |
| Gemini-2.5 Pro | - | 44.8 | 57.5 | 42.4 | 53.7 | 49.6 |
| Gemini-2.5 Flash | - | 40.3 | 53.2 | 43.6 | 57.4 | 48.6 |
| Grok-4.1 Fast | - | 39.4 | 49.5 | 42.4 | 48.6 | 45.0 |
| ChatGPT-5 | - | 36.6 | 35.0 | 25.9 | 33.1 | 32.6 |
| Claude Sonnet 4.5 | - | 19.6 | 23.0 | 19.6 | 29.1 | 22.8 |
| Qwen3-VL-Instruct-32B | 32B | 35.8 | 51.6 | 43.2 | 53.4 | 46.0 |
| InternVL-3.5-30B | 30B | 36.7 | 45.4 | 38.6 | 42.0 | 40.7 |
| Qwen3-VL-Instruct-8B | 8B | 26.0 | 47.8 | 35.1 | 46.3 | 38.8 |
| ... | ... | ... | ... | ... | ... | ... |
| Fuyu-8B | 8B | 9.5 | 14.7 | 9.5 | 16.2 | 12.5 |

**关键观察**：

1. **No VLM 超越 human baseline**。ChatGPT-5.1 在 2D-Dynamic 上 56.2 略低于 human 59.1，但 overall 53.1 < 55.6。
2. **Scaling effect 显著但有 diminishing return**。Qwen3-VL: 2B (29.0) → 8B (38.8) → 32B (46.0)。InternVL-3.5: 2B (25.0) → 8B (35.4) → 30B (40.7)。
3. **Dynamic tasks 受益于 scale 更多**。Qwen3-VL 在 2D-Dynamic 上从 2B 到 32B 几乎翻倍（39.0 → 51.6）。**这暗示 temporal integration 是 emergent ability，需要足够 capacity**。
4. **Claude Sonnet 4.5 表现意外差**（22.8），远低于 Gemini-2.5 Pro (49.6)。这可能反映 Anthropic 在 visual grounding 上的训练 emphasis 不同，或者 output format 偏好 verbose reasoning 不利于 numerical precision（见 G 部分 parsing 处理）。
5. **MiniCPM-V 4.5 在 3D 上完全崩溃**（3S=0.4, 3D=0.0）。这暗示 depth integration 对小模型是 fundamental 难题。

### 5.2 Theoretical ceiling vs. actual performance

论文有个关键 insight：**human baseline 不是理论 ceiling**。一个 ideal agent with precise frame-level pixel access 应该能 almost perfectly recover $\gamma$ 和 target quantities，因为这就是简单的 algebra。但 VLM 实际只达到 ~50% MRA，**说明它们 fundamentally under-utilize visual precision**。

这让我想到一个 analogy：把 VLM 给一个 calculator，让它算 $3.14159 \times 2.71828$，它可能输出 8.5（接近正确值 8.5397...），但你期望它能输出精确值。**VLM 的"reasoning"在数值层面是 approximate pattern matching，而非 symbolic computation**。

### 5.3 Scene context effect (Figure 5)

| Scene Type | Observation |
|-----------|-------------|
| SX/SS/SC (single object) | MRA 较低 |
| MX/MS/MC (multiple objects) | MRA 较高 |
| X (plain background) | 与 S (simple texture) 相近 |
| C (complex background) | **意外地比 X/S 略高** |

**Multiple objects 帮助推理**——这很 intuitive，因为其他 objects 提供了 implicit reference standards（比如另一个 ball, 一个 ruler-like structure）。

**Complex background 帮助推理**——这点反直觉。论文的 explanation 是：realistic backgrounds 提供额外 reference cues（tiles, windows, road markings）。**但这其实暗示 VLM 在利用 scene context 作为 scale anchor，而非纯粹从 video motion 推理**。这与后面 counterfactual 实验的发现高度一致。

## 6. 关键诊断实验：VLMs 是 guessers 不是 measurers

这部分是 paper 的灵魂。

### 6.1 Video ablation: prior only vs. video + prior

在 161 个 2D video-pair subset 上，比较两种 setting：
- **Video + Prior**：标准 setting
- **Prior only**：移除 video，只保留 prompt（含 physical prior, object description, question）

**Hypothesis**：移除 video 后应该有显著 performance drop，因为 VLM 失去了 visual evidence。

**Actual finding**：对大多数 model，**drop 远小于预期**。ChatGPT-5.1: 56.1 → 39.0；Gemini-2.5 Pro: 60.9 → 46.1；Qwen3-VL-32B: 50.1 → 37.2。一些 model 在 prior-only setting 上甚至和 video+prior setting 相当。

**Implication**：**VLM 已经能从 textual hints（"a car", "5.67m"）"猜" 出合理答案，video 增益有限**。这解释了为什么 ChatGPT-5.1 在 prior-only 上还有 39.0 MRA——它 internalize 了 typical car dimensions 和 speeds。

### 6.2 Counterfactual prior：真正揭露 memorization

这是最 sharp 的实验。对每个 instance，把原始 prior 乘以 scalar factor $\alpha \in \{0.001, 0.01, 0.1, 0.2, 5, 50, 100, 200, 500, 700\}$。

**Hypothesis**：如果 model 真的 conditional on provided prior 推理，prediction 应该按 $\alpha$ 缩放：$y^{\mathrm{cf}} = \alpha \cdot y$。MRA 应该几乎不 drop。

**Actual finding**：**几乎所有 model 的 MRA 都 drop 70-80%**。即使给 numerically precise 但 altered prior，output 仍然接近 real-world experience implied 的 magnitude。

**这是 paper 最重磅的结论**：

> VLMs are not yet input-faithful quantitative reasoners. They only weakly exploit pixel-level information in videos, and they do not reliably condition on the exact numerical priors provided in the prompts.

具体数字（Table 2 部分）：

| Model | Video+Prior | Prior only | Counterfactual | CoT |
|-------|-------------|------------|----------------|-----|
| ChatGPT-5.1 | 56.1 | 39.0 | 15.4 | 27.7 |
| Gemini-2.5 Pro | 60.9 | 46.1 | 29.9 | 49.8 |
| Gemini-2.5 Flash | 49.8 | 36.1 | 14.4 | 22.4 |
| Qwen3-VL-32B | 50.1 | 37.2 | 34.0 | 23.1 |

Gemini-2.5 Pro 在 counterfactual 上 29.9（相对原始 60.9 drop ~50%），已经是表现最好的之一。其他 model drop 80%+。

**让我用 Karpathy 你熟悉的语言解释这个现象**：当前 VLM 训练目标是 $p(\text{text} | \text{video, text context})$，最大化训练数据 likelihood。训练数据中 "car length" 几乎总是 ~4-5m，所以 model 学到的是 $p(\text{length} | \text{"car"}) \approx \delta(4.5)$。当你 input "car length = 5670m"，model 的 posterior 几乎不变，因为 textual context 已经把 output 锚定在 ~4.5m。

这其实是 **deeply ingrained 的训练动力学问题**——LLM training 让 prior knowledge strongly override explicit numerical input。要修复需要 architecture-level 或 objective-level 的 intervention，而非 prompt engineering。

### 6.3 Chain-of-Thought: 反而有害

论文设计了一个 4-step CoT prompt：
1. Pixel-level source property（"What is [prior object's property] in pixels?"）
2. Scale estimation（"What is the proportional relationship between pixels and [kinematic scale]?"）
3. Pixel-level target property
4. World-level target property

**Hypothesis**：分解成 step-by-step 应该帮助 VLM 执行 faithful reasoning。

**Actual finding**：**只有 3 个 model 有提升**（ChatGPT-5, Fuyu-8B 等少数）。其他 18 个 model **performance 下降**，有的下降很多。

**为什么 CoT 有害**：

1. **Error propagation**：step 1 如果 estimate pixel property 错了，后续 step 都基于错误 input。论文写道："Many models appear unable to reliably solve the intermediate numeric subproblems, so decomposing the task mainly amplifies and propagates early errors."

2. **VLM 不擅长 pixel-level measurement**：要求 model 输出 "car length in pixels" 这种 precise 数值，远超出当前 VLM 能力。它们能 coarse 描述（"the car is about 100-200 pixels long"），但无法精确测量。

3. **CoT 打破了 model 的 implicit shortcut**：当直接问 "car speed" 时，model 可能走 prior-based shortcut。当强制走 pixel → scale → target pipeline 时，model 必须显式执行每步，但这些步对 VLM 都是 hard problem。

**这个发现对 LLM reasoning 研究有重要 implication**：CoT 不是 universally helpful，特别是当 intermediate steps 本身超出 model 能力时。这呼应了你（Karpathy）在 "nanoGPT" 系列中提到的——autoregressive model 的 error compound over tokens，CoT 把这个 problem 放大。

## 7. 案例研究：ChatGPT-5.1 的 Thinking trace 分析

Paper supplementary A.1 给出 4 个 case study，非常 informative。

### Case 1: Faithful pixel-prior reasoning

2D 场景：yellow car 横向移动。Prior: car length = 5.67m。问 (1) speed at 2.0s, (2) width in meters。

ChatGPT-5.1 的 thinking trace：
1. Identify relevant frames around t=2.0s
2. 用 OpenCV-style tool 获取 bounding box
3. 取 longer side (135px) 作为 car length in pixel space
4. Calibrate $\gamma = 5.67 / 135 \approx 0.042$ m/pixel
5. 计算 width: $58/135 \times 5.67 \approx 2.44$m

**这是 input-faithful 的 ideal behavior**，预测接近 ground truth。

### Case 2: Counterfactual prior (×1000)

同样的 video，prior 改为 "car length = 5670m"。

ChatGPT-5.1 thinking trace：
- 显式注意到 "5670m is implausible"
- **abandon video 和 prior**
- 用 "typical car width-to-length ratio" 启发式
- 输出 plausible-looking width（relative accuracy ~0.9，但完全基于 pre-trained knowledge）

**这是 "right answer for wrong reasons"**。如果只看 outcome metric，会 judge 这个答案为 "good"，但 underlying reasoning 完全 ignore provided evidence。**这暴露了 outcome-based metric 的根本缺陷**。

### Case 3: Video ablation

只给 text prompt（含 "car length = 5.67m"），无 video。
- Speed at 2.0s: 12 m/s（远错于 ground truth）—— motion estimation 需要 visual evidence
- Width: 合理值（relative accuracy ~0.7）—— **基于 pre-trained prior over car dimensions**

**Implication**：即使 video available，size inference 可能主要 driven by pre-trained knowledge，而非 explicit pixel measurement。

### Case 4: Counterfactual physics（basketball with $a = 1 m/s^2$ 而非 $g$）

Blender 模拟篮球场景，acceleration = 1 m/s²（非标准 gravity）。Prior: ball diameter。

ChatGPT-5.1 **完全 ignore video 和 non-standard trajectory**：
- 输出 $a = 9.8 m/s^2$（标准 gravity）
- 输出 $v = 9.8 \times 1.5 = 14.7 m/s$
- 两个 query relative accuracy 都 = 0

**这显示 pre-trained gravitational prior 极强**，即使 scene violate 它也 dominate 推理。

## 8. 与相关工作的对比

### 8.1 vs. PhysBench [Chow et al. 2025]

PhysBench 是 qualitative VQA-based benchmark，state-of-the-art VLM 在 PhysBench 上 ~60% accuracy，远低于 human 95%。QUANTIPHY 把这个 gap 进一步暴露——**即使在 numerical accuracy 上，VLM 也只能达到 human baseline 水平，远低于理论 ceiling**。

PhysBench 链接: https://physbench.github.io/

### 8.2 vs. VSI-Bench [Yang et al. 2025]

VSI-Bench 是 spatial understanding benchmark，用 numerical metric 但只针对 static objects。QUANTIPHY 把这个 paradigm 扩展到 dynamic kinematic inference。论文 reference VSI-Bench 的 MRA metric。

VSI-Bench 链接: https://arxiv.org/abs/2412.14171

### 8.3 vs. CLEVRER, Physion++, PHYRE

这些是更早期的 qualitative benchmark，多在 synthetic 环境下评估 collision/falling/rebounding。QUANTIPHY 大幅扩展到 real-world 和 simulated 混合数据，并引入 quantitative dimension。

### 8.4 vs. FoundationPose [Wen et al. 2024]

FoundationPose 是 6D pose estimation 的 SOTA，但需要 color + depth + object mesh + camera parameters。QUANTIPHY 故意 probe **"in-the-wild" setting**：只有 monocular RGB video + textual prior。这是两种 complementary 的方法——FoundationPose 是 precise 但 narrow，VLM 是 broad 但 imprecise。

FoundationPose 链接: https://nvlabs.github.io/FoundationPose/

### 8.5 vs. Super-VSI [Yang et al. 2025]

Super-VSI 是 VSI-Bench 的 follow-up，显示 numerical spatial understanding 能力可以 empower embodied AI。QUANTIPHY 可以看作 Super-VSI 的 kinematic 版本。

Super-VSI (Cambrian-S): https://arxiv.org/abs/2511.04670

## 9. 数据集细节与潜在问题

### 9.1 Video type 分布（Table 4）

569 videos 中：
- 2D: 328 (58%)
- 3D: 241 (42%)
- Blender: 300, Internet: 72, Captured: 112, Segmented: 85

**最大的 single category** 是 V2MC (velocity prior, 2D, multiple objects, complex background) with 51 videos。最小的几个 category 只有 4-5 videos（如 V3SS, V3SC, V3SX）。这种 long-tail 分布可能让某些 category 的 evaluation noise 较大。

### 9.2 Motion type 的多样性

涵盖 motion pattern：
- Uniform motion
- Accelerated / decelerated linear motion
- Projectile motion
- Pendulum-like oscillation
- Centripetal motion
- Microscopic (red blood cells)
- Astronomical (planetary motion)
- Extraterrestrial (lunar walking)

但 **paper 明确 exclude**：
- Rotational dynamics
- Non-rigid / deformable objects
- Dynamic camera viewpoint
- Multi-body interaction

这些是 future work 的方向。

### 9.3 Annotation quality 的不均匀

- Blender simulation: 完全 precise（从 script 读出）
- Lab capture: metric depth + multi-view stereo，中等 precision
- Internet: manual pixel ruler + reference object，**least precise**

Internet data 的 annotation noise 可能影响 evaluation fairness。不过论文限制了 internet data 比例。

### 9.4 Frame rate 处理

论文 normalize 所有 video 到 480p，但 **保留所有 frame（不 subsample）**。理由：spatial resolution 对物理推理影响小，但 temporal fidelity 对 velocity/acceleration tracking 影响大。这是个 reasonable choice，但意味着 input token count 在 long video 上可能 explode。

## 10. 启发与未来方向

### 10.1 VLM architecture 的根本问题

QUANTIPHY 暴露的核心问题是：**当前 VLM architecture 没有显式的 "numerical reasoning module"**。Vision encoder 输出 embedding，LLM backbone 做 autoregressive decoding，整个 pipeline 没有 slot for precise arithmetic。

可能的 architecture-level intervention：
1. **External tool use**：让 VLM 显式调用 calculator / OpenCV（Case 1 中 ChatGPT-5.1 已经 implicitly 这样做）
2. **Differentiable physics layer**：在 VLM 末端加上 physics-informed layer，强制遵守 kinematic equations
3. **Token-level numerical supervision**：训练时 explicit supervise on numerical output，而非 just next-token prediction

### 10.2 Training data 的 intervention

VLM 训练数据中物理定量标注稀少。可能需要：
- Synthetic physics-rich data pretraining（类似 physics simulator-augmented training）
- Counterfactual augmentation：在 training 时 explicit 包含 counterfactual priors，force model to condition on input
- Self-supervised kinematic consistency loss

### 10.3 Evaluation 的下一步

QUANTIPHY 是 first step，但 future benchmark 应该扩展到：
- Rotational dynamics（angular velocity, torque）
- Multi-body interaction（collision, friction）
- Deformable objects
- Dynamic viewpoint（ego-motion compensation）
- Real-time physical reasoning（streaming setting）

### 10.4 与你（Karpathy）的研究兴趣的交叉

你最近在 nanoGPT 和 education 方面做了很多工作。QUANTIPHY 暴露的现象——LLM 的 parametric prior 强烈 override explicit input——其实是 **autoregressive LM 的 fundamental property**。

考虑一个 toy experiment：训练一个 small transformer 在 synthetic arithmetic task 上，input 是 "$x = 5670$, compute $f(x)$"。如果 training distribution 大部分是 $x \in [1, 10]$，model 会 strongly bias toward $f(x) \approx f(\text{typical } x)$，即使 input 是 5670。这就是 in-context learning 和 parametric prior 的 trade-off，QUANTIPHY 在 VLM 上 empirical 验证了这点。

要 fix 可能需要：
- Attention mechanism 的 modification（让 input token 有更高 weight）
- Training distribution 的 rebalance
- Explicit compositional reasoning module

## 11. 我的几个 criticisms 和 open questions

1. **MRA 的 zero-penalty 问题**：API failure 等同 MRA=0，可能 distort ranking。建议 paper 未来 report "valid response rate" 作为辅助 metric。

2. **Counterfactual experiment 的 scope**：只在 161 个 2D pairs 上做，相对整个 3355-question dataset 比例较小。建议扩展到 3D 和更多 scene type。

3. **Human study 的 sample size**：paper 没明确说参与者数量，Figure 39 的 boxplot 暗示 participants 数量有限。Human baseline 55.6 MRA 可能 underestimate true human ceiling。论文也承认 top human participant 在 2D 上 0.721，3D 上 0.724——**远高于 average**。

4. **CoT 的 negative result 可能是 prompt design 问题**：4-step decomposition 也许不是最优。可以尝试 tree-of-thought, program-aided reasoning (让 VLM 输出 Python code 然后 execute)。

5. **Cross-model pattern 的 explanation**：为什么 Gemini-2.5 Pro 在 counterfactual 上比 ChatGPT-5.1 好（29.9 vs 15.4）？是否反映 training data 或 RLHF 的差异？这个值得深入 ablation。

6. **Video representation 的影响**：当前所有 model 用 frame-based input。CogVLM2-Video 和 MiniCPM-V 4.5 接受 native video file，但表现一般。是否专门的 video architecture（比如 with temporal attention）会有帮助？

## 12. 总结：QUANTIPHY 的真正贡献

QUANTIPHY 不只是一个新 benchmark，它是一个 **diagnostic tool** 揭示了当前 VLM 的 fundamental limitation：

> VLMs act more as approximate guessers based on semantic context rather than precise visual measurers.

这个 finding 对 embodied AI、autonomous driving、AR/VR 等领域有深远 implication。如果 VLM 不能 faithful 使用 visual input 和 numerical prior，那么 deploying 它们到 safety-critical physical reasoning task 是危险的。

Paper 最有价值的是 **counterfactual probe**——这是一个 sharp diagnostic，可以直接 distinguish "reasoning" vs "memorization"。建议未来所有 VLM physical reasoning benchmark 都包含类似 probe。

**对 Karpathy 你来说**，QUANTIPHY 验证了你可能早就直觉感到的东西：当前 LLM/VLM 的 "reasoning" 大部分是 sophisticated pattern matching，而非 symbolic computation。要真正实现 physical reasoning，可能需要回到 first principles——symbolic system, hybrid neural-symbolic architecture, 或者 fundamentally new training paradigm。

---

**主要 reference links 汇总**：
- QUANTIPHY Project: https://quantiphy.stanford.edu/
- QUANTIPHY Code: https://github.com/Paulineli/QuantiPhy
- QUANTIPHY Dataset: https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation
- VSI-Bench (MRA metric origin): https://arxiv.org/abs/2412.14171
- Super-VSI / Cambrian-S: https://arxiv.org/abs/2511.04670
- PhysBench: https://physbench.github.io/
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounding DINO 1.5: https://arxiv.org/abs/2405.10300
- CLEVRER: https://arxiv.org/abs/2003.11664
- Physion++: https://arxiv.org/abs/2106.05694
- PHYRE: https://arxiv.org/abs/1908.00736
- VideoPHY-2: https://arxiv.org/abs/2503.06800
- Thinking in Space (VSI-Bench 论文): https://arxiv.org/abs/2412.14171
- Orbbec Femto Mega (lab 用的 depth camera): https://shop.orbbec.com/products/femto-mega
- BlenderKit: https://blenderkit.com/
- Sketchfab: https://sketchfab.com/
