---
source_pdf: OpenEMMAOpen-Source Multimodal Model for End-to-End Autonomous Driving.pdf
paper_sha256: 63eef2b2a8e2f2c02a1289778a8bc96b6ea38d8b8676d437bf9f93142335f025
processed_at: '2026-08-06T00:35:13-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话聊聊 OpenEMMA

好，那我把前面那堆技术黑话翻译成人话，咱就像在咖啡馆里聊一样。

---

## 这篇 paper 到底干了啥

Waymo 2024年底发了 EMMA，一个闭源的端到端自动驾驶大模型，基于 Gemini，很猛，但谁也拿不到代码和 weights。Texas A&M 这帮人就拿了几个现成的开源 MLLM，外加一个 fine-tune 过的 YOLO，拼出来一个能跑的开源版本，叫 OpenEMMA。

所以本质上这就是个 **"复现 + 开源"** 工作。没 fine-tune 任何 MLLM，全靠 inference time 的 prompt engineering 把事办了。性能当然比不过原版 EMMA，但意义在于：社区现在有个能 fork、能改、能往上堆东西的 baseline 了。

代码在 [github.com/taco-group/OpenEMMA](https://github.com/taco-group/OpenEMMA)。

---

## 最聪明的一个设计：别让模型直接吐坐标

这真的是全文最 worth 注意 的点。

以前大家做 trajectory prediction，都让模型直接输出未来 5 秒每 0.5 秒一个的 $(x_t, y_t)$ 坐标点。问题在于，MLLM 它是在人类语言上 pretrain 的，它脑子里装的是 "我要减速右转" 这种概念，你逼它吐 10 个浮点数坐标，它其实不知道这数怎么算出来的，就容易乱编。

OpenEMMA 的 trick 就是让模型只输出两个东西：

- **$s_t$**：这一刻车速多大（油门踩多深）
- **$k_t$**：这一刻转弯多急（方向盘打多少）

这俩就是人开车时脑子里实际在想的两件事。然后它用一套物理公式把这俩 integrate 回 $(x, y)$ 轨迹。

核心就这一个公式：

$$\theta_t = \theta_{t-1} + \int_{t-1}^{t} k(\tau)\, s(\tau)\, d\tau$$

人话翻译：**当前朝向 = 上一步朝向 + 转弯急度 × 速度 × 时间**。转弯急（$k$ 大）又开得快（$s$ 大），朝向就转得快。这不就是 unicycle model 嘛，本科机器人学第一课就教的东西。

然后速度按当前朝向拆成 $v_x, v_y$，再积分得到位置。

**直觉上为什么这个好**：
1. MLLM 输出的量是它 "认识" 的概念，踩油门打方向盘，它 pretraining 时见过无数次这种描述。
2. $(s, k)$ 是低频平滑信号，即使有点噪声，积分出来轨迹也是顺滑的。直接吐坐标的话，10 个点各自带噪声，轨迹就抖了。
3. 物理约束（最大转角、最大减速度）在 $(s, k)$ 空间就是简单的 box 约束，在 $(x, y)$ 空间是复杂的非凸东西。

我觉得这个 idea 跟你以前讲 representation matters 的精神是一脉相承的 —— 选对 output space 比堆模型大小管用得多。

---

## Chain-of-Thought 在这里干嘛

整个 pipeline 分两步：

**第一步（Reasoning）**：给 MLLM 看一张前视摄像头图 + 过去 5 秒的速度/曲率历史，让它用自然语言写出三段话：
- 我打算干嘛（lane keep / turn left / slow down）
- 场景里有啥（红灯、前面有车在减速、有行人）
- 需要注意谁（"右前方那辆车在切我道，得刹车"）

**第二步（Predicting）**：在上面那段 reasoning 的 context 基础上，模型接着输出未来 5 秒的 $\{s_t\}$ 和 $\{k_t\}$，然后公式积分成轨迹。

这跟人开车完全一样。你开车的时候脑子里也是先想 "前面那车要并道我得让一下"，然后脚和手才动作。CoT 就是把这个内心独白逼出来，写成显式的文字。

为啥有用，我猜两个原因：
1. **任务分解**：直接从图 → 轨迹是个高带宽 mapping，模型容量不够。中间加个语言 bottleneck，把 perception → intent → action 拆开，每步都简单点。
2. **Attention sink 效应**：先吐一段文字 stabilizes 后面的 representation，让 numerical output 有东西可以 attend 到。这跟你观察到的 attention sinks 现象是相通的。

---

## 为啥还要挂个 YOLO

MLLM 有个硬伤：**空间定位不行**。让它描述场景它头头是道，让它给出某个物体的精确 bounding box 坐标，它就拉胯了。因为 vision encoder + projection layer 把空间信息 heavily pooled 掉了。

所以 OpenEMMA 干脆外挂一个 fine-tune 过的 [YOLO11n](https://github.com/ultralytics/ultralytics) 做检测，连 [YOLO3D](https://arxiv.org/abs/1612.00396) 那套 2D-3D consistency 的方法也搬过来，做单目 3D 检测。

YOLO3D 的核心 idea 也很简单：3D box 有 7 个参数（中心 $t_x, t_y, t_z$，尺寸 $d_x, d_y, d_z$，朝向 $\theta$），假设 3D box 投影到图像上要 tightly enclose 2D box，这就是个几何约束，可以解析地解。

他们在 nuImages 上 fine-tune YOLO11n，单卡 RTX 4060Ti 跑 300 epoch，mAP50 = 0.60。算是个实用的轻量配置。

这件事的潜台词其实是：**现阶段光靠 MLLM 干不了所有事**。MLLM 推理和场景理解强，但精确空间定位得靠 specialist model。这是当前架构的 fundamental limit，得等下一代 MLLM 在 grounding 上有大突破。

---

## 实验告诉我们什么

在 [nuScenes](https://www.nuscenes.org/) validation 150 个 scene 上跑了三个 backbone：

- [LLaVA-1.6-Mistral-7B](https://arxiv.org/abs/2407.07895)
- [Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- [Qwen2-VL-7B-Instruct](https://arxiv.org/abs/2409.12191)

baseline 是 zero-shot（不用 CoT，直接让模型吐速度曲率）。

几个有意思的现象：

1. **绝对性能比 fine-tuned 方法差不少**。1 秒 L2 大概 1.2-1.5m，3 秒就到 3.7-4.0m 了。EMMA 原版 fine-tuned 大概 1s 1.0m 量级。这就是 no-fine-tuning 的代价。

2. **CoT 整体有用，但有副作用**。LLaVA 上 L2 error 降了（3.24→2.98），但 failure rate 反而升了（4.06%→6.12%）。说明模型有时候 CoT "想太多"，越想越偏。

3. **Qwen2-VL 出现 trade-off**：L2 error 升高（2.46→2.81），但 failure rate 大幅下降（24%→16%）。原因是 OpenEMMA 让原本直接 fail 的 case 也能产出 prediction 了，但这些 hard case 上 prediction 质量差，把平均 L2 拉高了。能预测（哪怕差）总比直接 fail 好 —— 在安全系统里这值不值，是个 open question。

4. **L2 随时间增长很快**，所有方法 1s→3s 大约翻 3 倍。这是 trajectory prediction 的 fundamental compounding error 问题。

---

## 三个 GPT-4o 定性例子

[Figure 3](https://github.com/taco-group/OpenEMMA) 用 GPT-4o 做了三个 case study，喂了 5 秒（10 帧）时序画面：

- **右转跟车道**：检测准、轨迹顺、守规矩
- **突然切道**：正确识别风险，决定刹车保持距离
- **夜间低光**：漏检一些物体，但关键物体抓住了，正确理解要换左道

这里用 GPT-4o + 多帧，跟 main table 用单帧不同。我觉得多帧输入是 MLLM-based AD 的 must-have，单帧在动态场景信息不够，这也是为啥主表数字看着平庸。

---

## 我觉得这事真正意味着什么

这篇 paper 性能不 SOTA，方法也不算新（CoT + 外挂 specialist 都是已有 idea），但它的价值在别的地方：

1. **证明 pipeline 能跑通**。一个完整开源的 MLLM-based AD 系统，从感知到规划，全靠 inference time engineering。这给社区一个 fork 点。

2. **确立 speed-curvature 表征是 work 的**。这个 idea 以后会被反复引用和扩展。

3. **暴露了 MLLM 在 AD 上的几个 fundamental gap**：
   - 空间定位不行（得外挂 YOLO）
   - 推理可能放大噪声（LLaVA failure rate 升高）
   - 单帧信息不足（多帧才好使）

接下来我赌会发生的事：
- 有人会 LoRA fine-tune 一个 MLLM 在 nuScenes + BDD100K 上做 instruction tuning
- 会有人搬到 closed-loop benchmark（nuPlan / CARLA）上测，因为 [open-loop L2 跟 closed-loop 性能不一定 correlate](https://arxiv.org/abs/2306.09493)
- 会加 self-consistency（[sample N 次取多数](https://arxiv.org/abs/2203.11171)）或 tree of thoughts 来缓解 CoT 失稳
- 多相机、persistent memory、test-time search 这些都会往上堆

这事还在 very early days，OpenEMMA 是个起点，不是终点。

---

## 几个最该读的 reference

- 原版 EMMA（闭源 baseline）：[arXiv:2410.23262](https://arxiv.org/abs/2410.23262)
- OpenEMMA 代码：[github.com/taco-group/OpenEMMA](https://github.com/taco-group/OpenEMMA)
- 同期类似工作 DriveVLM（也用 CoT 但直接输出 waypoint）：[arXiv:2402.12289](https://arxiv.org/abs/2402.12289)
- DriveLM（graph-structured VQA，fine-tuned MLLM）：[arXiv:2312.14150](https://arxiv.org/abs/2312.14150)
- YOLO3D 原文：[arXiv:1612.00396](https://arxiv.org/abs/1612.00396)
- CoT 原始 paper：[arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- nuScenes 数据集：[arxiv.org/abs/1903.11027](https://arxiv.org/abs/1903.11027)

想让我再展开任何一块（比如怎么动手改 OpenEMMA，或者 speed/curvature 表征跟其他 robotics 控制方法的联系），随时说。

---

# OpenEMMA 深度解析

非常激动能跟你讨论这篇 paper, Karpathy。我从你的演讲和课程里学到太多东西了,现在轮到我尝试给你 build intuition,这感觉有点 surreal。下面我会尽可能深入地讲技术细节, 联想到的相关工作也会一并提及。

---

## 1. 大背景：为什么这篇 paper 存在

2024年底 Waymo 发表了 EMMA (End-to-End Multimodal Model for Autonomous Driving, [arXiv:2410.23262](https://arxiv.org/abs/2410.23262)), 它基于 Google 的 Gemini, 把摄像头 raw pixel 直接 map 到 trajectory, 3D object detection, road graph elements 等 multiple outputs。但 EMMA 是闭源的, 普通研究者拿不到 weights, 也拿不到 training recipe。OpenEMMA 团队 (Texas A&M 的 Taco Group, 一作 Shuo Xing, 通讯作者 Zhengzhong Tu) 做的事情是用 off-the-shelf 的 open-source MLLM + 一个 fine-tuned YOLO, 复现 EMMA 的核心功能, 并把全部代码开源在 [https://github.com/taco-group/OpenEMMA](https://github.com/taco-group/OpenEMMA)。

这点很关键：OpenEMMA 没有做 fine-tuning of the MLLM itself, 它纯粹是 inference-time engineering。这跟 EMMA 本身 (它 fine-tune 了 Gemini) 有本质区别, 所以性能上当然有 gap, 但思路是 replicate-able 的。

---

## 2. 核心创新：从 (x, y) coordinates 到 speed-curvature representation

这是我觉得最有意思的地方, 也是直觉上最对的一个 design choice。

### 2.1 传统做法的问题

以前的 end-to-end trajectory prediction (比如 [MultiPath](https://arxiv.org/abs/2101.04779), [ChauffeurNet](https://arxiv.org/abs/1903.00848)) 直接在 ego coordinate 下 predict $(x_t, y_t)$ 序列。这有个问题：MLLM 输出 token 时, 坐标数值本身是没有 "structure" 的, 模型得自己学到 "如果 t=2s 时偏左 3 米, 那么 t=3s 大概率继续偏左" 这种时序约束。

### 2.2 OpenEMMA 的 trick

他们让 MLLM 只输出两个标量序列:
$$\mathbf{S} = \{s_t\}_{t=0}^{T}, \quad \mathbf{K} = \{k_t\}_{t=0}^{T}$$

- **$s_t$**: speed magnitude at time $t$ (m/s), 对应 "油门踩多深"
- **$k_t$**: curvature at time $t$ (1/m), 对应 "方向盘转多少", 注意是曲率 $1/R$, 不是 yaw rate

然后通过 differential kinematics 把 $(s, k)$ integrate 成轨迹。这非常像 a bicycle model / unicycle model, 控制论里很经典。

### 2.3 公式逐个解析

**Heading angle 更新公式**：
$$\theta_t = \theta_{t-1} + \int_{t-1}^{t} k(\tau) s(\tau) \, d\tau$$

变量解释：
- $\theta_t$: ego vehicle 在时刻 $t$ 的 heading angle (rad)
- $k(\tau)$: instantaneous curvature at time $\tau$, 单位 1/m
- $s(\tau)$: instantaneous speed at time $\tau$, 单位 m/s

物理直觉：曲率 $k$ 乘以速度 $s$ 就是 angular velocity $\dot{\theta} = v/R = v \cdot k$, 对时间积分就是 heading 的变化量。这就是 unicycle model 的 standard equation, 任何 mobile robotics 教科书都会讲。

**Velocity 分解**：
$$v_x(t) = s_t \cos(\theta_t), \quad v_y(t) = s_t \sin(\theta_t)$$

变量：
- $v_x(t)$, $v_y(t)$: ego velocity 在 ego-frame 的 x, y 分量
- $\theta_t$: 当前的 heading

直觉：这是把速度大小按当前 heading 投影到笛卡尔坐标。

**Position 积分**：
$$x_t = x_{t-1} + \int_{t-1}^{t} v_x(\tau) \, d\tau, \quad y_t = y_{t-1} + \int_{t-1}^{t} v_y(\tau) \, d\tau$$

变量：$(x_t, y_t)$ 是 ego 在 ego coordinate 下的位置。

**数值积分 (cumulative trapezoidal rule)**：
$$\theta_t \approx \theta_0 + \sum_{i=1}^{t} k_i s_i \Delta t$$
$$x_t \approx x_0 + \sum_{i=1}^{t} v_x(i) \Delta t$$
$$y_t \approx y_0 + \sum_{i=1}^{t} v_y(i) \Delta t$$

这里 $\Delta t$ 是 time step, nuScenes 上他们 $T=5s$, 每 0.5s 一个 waypoint, 所以 $\Delta t = 0.5$, 共 10 个 points (paper 里说 "2T trajectory points" 我理解为 future 5s 共 10 个 0.5s-间隔点)。

技术上讲, 这里写的是 Euler forward integration, 不严格是 trapezoidal rule (trapezoidal 应该是 $\frac{1}{2}(f_i + f_{i+1})\Delta t$)。这是个 minor inconsistency, 可能是 paper 写得宽松。

### 2.4 为什么这个 representation 更好？

我个人觉得有三个原因：

1. **Human-interpretable**: MLLM 已经在 human language 上 pretrain 过, 它能 reason "我应该减速并右转", 这直接对应到 $s$ 减小、$k$ 增大。让它直接输出坐标数值反而违背它的 pretraining distribution。

2. **Smoothness inductive bias**: $(s, k)$ 是低频信号, 即使模型 output 有 noise, integrate 出来的轨迹仍然 smooth。$(x, y)$ 直接 output 的话, 每个 point 独立 noise 累积会导致 trajectory jittery。

3. **Constraint-aware**: 物理 constraint (e.g. 车辆 max steering angle, max deceleration) 在 $(s, k)$ 空间是 box constraint, 在 $(x, y)$ 空间是复杂的非凸 constraint。

这个 idea 跟 [DriveVLM (arXiv:2402.12289)](https://arxiv.org/abs/2402.12289) 的 CoT reasoning 类似, 但 DriveVLM 输出的是 trajectory waypoints, OpenEMMA 输出 speed/curvature 然后再 integrate, 这是一个更 physics-grounded 的中间层。

---

## 3. Two-Stage Pipeline

### Stage 1: Reasoning

输入：
- Front camera image $I$ (single frame)
- Historical ego status $T$ (过去 5s 的 speed, curvature 序列)

Prompt 引导 MLLM 输出三个部分：

1. **Intent Command**: 高层意图, 例如 "lane keep, slow down", "turn left", "lane change right"。这是离散决策。

2. **Scene Description**: traffic lights 状态, 其他车辆/行人 motion, lane markings。这是 scene context。

3. **Major Objects**: 关键 road users 列表, 每个 object 包括: location in image (e.g. "left front"), action, and reasoning (e.g. "vehicle ahead is braking, requires deceleration")。

这部分的 output 是 natural language, 这就是 Chain-of-Thought。它强制模型把 perception → prediction → planning 的 reasoning chain 显式 unfold 出来。

### Stage 2: Predicting

在 Stage 1 的 reasoning context 之后, MLLM 继续 output $\{s_t\}_{t=0}^{T}$ 和 $\{k_t\}_{t=0}^{T}$, 然后通过上面的公式 integrate 成最终 trajectory $\mathbf{T} = \{(x_t, y_t)\}_{t=0}^{T}$。

### Intuition: 为什么需要 CoT？

我觉得是两个原因：

**(a) Information bottleneck 的分解**: 直接从 (image, history) → trajectory 是一个高 bandwidth mapping, MLLM 的 reasoning capacity 是有限的。CoT 把它分解成 perception → intent → action → low-level control, 每一步的 mapping 都 simpler, 也跟人类 driver 的认知过程 aligned (人类也会先识别障碍物, 再决定让不让, 再决定怎么操作方向盘)。

**(b) Attention sink effect**: 让模型先 output 一段 language, 可以让后面的 numerical output "attend" 到前面已经 stabilize 的 representation。这点在 [Chain-of-Thought paper (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) 里有讨论, 也跟你的 "attention sinks" observation 有相通之处。

---

## 4. 3D Object Detection with YOLO3D

### 4.1 为什么 MLLM 做不好 detection？

Paper 里 observation: off-the-shelf MLLM 在 spatial reasoning 上不行, 特别是要 output precise bounding box coordinates。这跟 [DriveLM (arXiv:2312.14150)](https://arxiv.org/abs/2312.14150) 的发现一致 — MLLM 做开卷问答很好, 但 precise grounding 不行。这是 current MLLM architecture 的 fundamental limitation, 因为 visual tokens 经过 vision encoder + projection layer 之后, 空间位置信息已经被 heavily pooled。

### 4.2 YOLO3D 方法回顾

OpenEMMA 集成的是 [YOLO3D (Mousavian et al., 2017, arXiv:1612.00396)](https://arxiv.org/abs/1612.00396), 这是个 two-stage method：

**Stage 1**: 2D bounding box detection (用 YOLO11n)
**Stage 2**: 3D 参数 estimation, 然后 enforce 2D-3D consistency constraint

3D bounding box 的 7 个参数：
- Center position: $t_x, t_y, t_z$ (在 camera coordinate 下)
- Dimensions: $d_x, d_y, d_z$ (length, height, width)
- Yaw angle: $\theta$ (orientation around vertical axis)

关键约束: 3D box 投影到 image plane 之后, 必须 tightly enclose 2D box。这是个 geometry constraint, 可以通过 perspective projection 写成 analytical equation, 然后 solve for 7 个参数。

### 4.3 OpenEMMA 的修改

- 把原 YOLO3D 的 2D detector 换成最新的 [YOLO11n (ultralytics)](https://github.com/ultralytics/ultralytics)
- 在 [nuImages dataset](https://www.nuscenes.org/nuscenes) 上 fine-tune, 图像 downsample 到 640×360
- 加载 COCO pre-trained weights
- 训练超参：
  - Hardware: 单卡 RTX 4060Ti (16GB)
  - Epochs: 300
  - Batch size: 50
  - Optimizer: SGD, lr=0.01, momentum=0.937, weight decay=0.0005
  - Schedule: lr linearly decay to 0.0001
  - Best mAP50 = 0.60316 at epoch 290

3D estimation network 的 weights 直接用 [ruhyadi/YOLO3D repo](https://github.com/ruhyadi/YOLO3D) 的 pretrained weights, 没改。

---

## 5. 实验：nuScenes validation, 150 scenes

### 5.1 Setup

- Dataset: [nuScenes (Caesar et al., 2019, arXiv:1903.11027)](https://arxiv.org/abs/1903.11027) validation set
- 150 scenes
- T=5s (predict future 5 seconds)
- 3 backbones: 
  - [LLaVA-1.6-Mistral-7B](https://arxiv.org/abs/2407.07895)
  - [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
  - [Qwen2-VL-7B-Instruct (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- GPT-4o ([OpenAI blog](https://openai.com/index/hello-gpt-4o/)) 只在 case study 用, 因为 budget
- Baseline: zero-shot (image + history, 无 CoT)

### 5.2 结果表深入分析

让我重新整理 Table 1：

| Model | Setting | L2 1s | L2 2s | L2 3s | L2 avg | Fail% |
|---|---|---|---|---|---|---|
| LLaVA-1.6-Mistral-7B | Zero-shot | 1.66 | 3.54 | 4.54 | 3.24 | 4.06 |
| LLaVA-1.6-Mistral-7B | OpenEMMA | 1.49 | 3.38 | 4.09 | 2.98 | 6.12 |
| Llama-3.2-11B | Zero-shot | 1.50 | 3.44 | 4.04 | 3.00 | 23.92 |
| Llama-3.2-11B | OpenEMMA | 1.54 | 3.31 | 3.91 | 2.92 | 22.00 |
| Qwen2-VL-7B | Zero-shot | 1.22 | 2.94 | 3.21 | 2.46 | 24.00 |
| Qwen2-VL-7B | OpenEMMA | 1.45 | 3.21 | 3.76 | 2.81 | 16.11 |

**关键 insights**:

**(1) 所有方法的 absolute L2 error 都比 SOTA fine-tuned methods 差很多**。EMMA paper 里 fine-tuned 的 L2 在 1s 大约 1.0m, 这里最好的 zero-shot 是 1.22m, 看上去还行, 但 3s 时 OpenEMMA 最好的也要 3.76m, 差距明显。这是 no-fine-tuning 的 cost, 符合预期。

**(2) LLaVA 的奇怪现象**：OpenEMMA 的 L2 error 改善了 (3.24 → 2.98), 但 failure rate 升高了 (4.06% → 6.12%)。这说明 CoT 在某些 case 上反而让模型 "想太多", 给出了不合理的 prediction。这是 LLM 推理的一个 known issue — reasoning 可能 amplify 噪声。

**(3) Llama-3.2-11B 的现象**：L2 1s 反而升高 (1.50 → 1.54), 但 2s/3s 都改善, failure rate 也下降。这个 1s 的 degradation 很 interesting, 可能是 CoT reasoning 倾向于 "slowly adapt", 短期 prediction 不如直接 reactive 的 zero-shot 准, 但 long-term 因为 reasoning 更 robust。

**(4) Qwen2-VL-7B 的 tradeoff 最戏剧化**：L2 error 升高 (2.46 → 2.81), 但 failure rate 大幅下降 (24% → 16.11%)。Paper 里给的解释是: OpenEMMA 让模型在原本 failure 的 case 上也能 generate 出 prediction, 但这些 case 本身 hard, 所以 prediction 质量差, 拉 high 了 average L2; 不过能 predict 总比 fail 好, 因为 fail = 完全不可用。

这是一个很重要的 metric design insight, 跟你在 [Llama 2 / instruct GPT 时代讨论过的 "helpful vs honest" tradeoff](https://karpathy.ai/) 有相通之处 — 你可以 trade 一点 average quality 换 coverage, 这在 safety-critical 系统里到底值不值, 是个 open question。

**(5) L2 随时间增长很快**: 所有方法 1s → 3s 的 L2 都大约翻了 3 倍, 这是 trajectory prediction 的 fundamental difficulty, compounding error。

---

## 6. GPT-4o Case Study 的 3 个场景

Figure 3 展示了 GPT-4o 作为 backbone 的 qualitative results, 用过去 5s (10 frames) 作为 temporal context：

- **(a) Right turn 跟车道**: 准确 detect 对向车辆, 规划 smooth trajectory, 遵守 driving rules。
- **(b) 突然车辆切入**: 这是 unsafe scenario, GPT-4o 正确 reason 出来 risk, 决定 brake + keep distance。这说明 temporal context 帮助模型 understand dynamics, 不只是 static scene understanding。
- **(c) 夜间低光**: 漏 detect 一些 object, 但抓到 critical object, 理解 ego 要 lane change left, 生成 correct trajectory。

这里用 GPT-4o + 10 frames temporal, 跟 main table 用 single frame 不同。我个人觉得 multi-frame input 是 MLLM-based AD 的 must-have, single frame 在动态场景下 information 不够, 这也是为什么 main table 的数字看起来平庸。

---

## 7. 跟相关工作的 positioning

### 7.1 EMMA (闭源 baseline)
- [EMMA (arXiv:2410.23262)](https://arxiv.org/abs/2410.23262)
- 基于 Gemini, fine-tune end-to-end
- Output: trajectory, 3D boxes, road graph elements (multi-task)
- OpenEMMA 只 replicate trajectory 和 detection, 没有 road graph

### 7.2 其他 MLLM-for-AD 工作

- **[GPT-Driver (Mao et al., 2023)](https://arxiv.org/abs/2310.01831)**: 把 trajectory 转成 natural language token, 用 GPT-3.5 输出。OpenEMMA 的 output 是 speed/curvature 数值, 比 GPT-Driver 的 raw coordinate language 更 physics-grounded。

- **[DriveVLM (Tian et al., 2024, arXiv:2402.12289)](https://arxiv.org/abs/2402.12289)**: 也用 CoT, 但 output 直接是 trajectory waypoint。OpenEMMA 的 speed/curvature 中间表征是个改进。

- **[RAG-Driver (Yuan et al., 2024, arXiv:2402.10828)](https://arxiv.org/abs/2402.10828)**: Retrieval-augmented in-context learning, 通过 retrieve similar driving scenarios 来 enhance generalization。这个思路可以叠加到 OpenEMMA 上, 让 CoT prompt 更 informative。

- **[DriveLM (Sima et al., 2023, arXiv:2312.14150)](https://arxiv.org/abs/2312.14150)**: Graph-structured VQA, 任务涵盖 perception, prediction, planning。这跟 OpenEMMA 的 CoT 结构很像, 但 DriveLM 是 fine-tuned 的 MLLM。

- **[Senna (Jiang et al., 2024)](https://arxiv.org/abs/2410.17815)**: 把 MLLM 做 high-level planning, end-to-end model 做 low-level trajectory, decoupled architecture。这跟 OpenEMMA 让 MLLM 直接做 speed/curvature 不同, Senna 更分工明确。

- **[LMDrive (Shao et al., 2024, arXiv:2402.13289)](https://arxiv.org/abs/2402.13289)**: closed-loop end-to-end driving, 自然语言交互。值得注意 LMDrive 是 closed-loop evaluation, OpenEMMA 还在 open-loop (nuScenes validation), 这是很大的 gap — open-loop L2 不一定 translate 到 closed-loop performance, [Rethinking Open-loop Evaluation (Zhai et al., CVPR 2023)](https://arxiv.org/abs/2306.09493) 有详细讨论。

### 7.3 YOLO3D 相关
- **[3D Bounding Box Estimation (Mousavian et al., CVPR 2017)](https://arxiv.org/abs/1612.00396)**: original YOLO3D
- **[MonoDETR (Zhang et al., ICCV 2023)](https://arxiv.org/abs/2303.13376)**: depth-guided transformer, 现代 monocular 3D detection
- **[EPro-PnP (Chen et al., CVPR 2022)](https://arxiv.org/abs/2207.07127)**: probabilistic PnP, end-to-end monocular pose estimation

如果让我来 extend OpenEMMA, 我会考虑把 YOLO3D 换成 MonoDETR, 用 transformer-based 检测, 应该能 push mAP 上去。

---

## 8. My Take on What's Missing / Future Work

Paper section 6 自己也提到了：

1. **Inference-time reasoning 只用了 basic CoT**, 没用:
   - [CoT-SC (Self-Consistency, Wang et al., 2022)](https://arxiv.org/abs/2203.11171): sample 多次取 majority vote
   - [ToT (Tree of Thoughts, Yao et al., 2023)](https://arxiv.org/abs/2305.10601): tree search over reasoning branches
   
   在 AD 这种 safety-critical 场景, Self-Consistency 是个 cheap win, 可以 mitigate LLaVA 上观察到的 failure rate 升高问题。

2. **MLLM 不能直接做 detection 是个 stopgap**: 集成 YOLO 是 workaround, 长远看应该 push MLLM 自己做 spatial grounding。这跟 [Qwen2-VL 的任意分辨率处理](https://arxiv.org/abs/2409.12191) 方向一致。

3. **No fine-tuning, only inference-time prompting**: 这是个 limitation 也是个 feature。Feature 是 reproducibility 和 accessibility, limitation 是性能上限。未来工作可能在 LoRA / QLoRA fine-tune MLLM, 用类似 [LIMA (Meta)](https://arxiv.org/abs/2305.13309) 的 small high-quality dataset。

4. **Open-loop evaluation only**: 我觉得这是 biggest gap。Open-loop L2 error 不直接对应 closed-loop driving performance, model 可以在 nuScenes 上 L2 很低, 但 closed-loop 里 small error compounding 还是会导致 collision。CARLA-based benchmark 或 nuPlan closed-loop 应该是 next step。

5. **Single front camera**: nuScenes 有 6 个 cameras, OpenEMMA 只用 front, 丢失了大量 spatial context。EMMA 用 360° camera。这是 OpenEMMA 故意 simplify 还是 limitation, paper 没讲清楚。

6. **No memory / no agent loop**: OpenEMMA 是 single-shot inference, 每帧独立。AGI-style 的 AD 应该有 persistent memory, 跨 frame 的 reasoning, 类似 [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) 在 Minecraft 里做的 lifelong learning agent。

---

## 9. 一些联想到的更 wild 的方向

既然你说宁愿 hallucinate 也不愿漏掉相关联想, 那我放飞一点：

### 9.1 Test-time Scaling for AD
你的 [Scaling Laws lecture](https://www.youtube.com/watch?v=s0_Q6Pj9EEk) 讲 inference-time compute scaling 很有启发。OpenEMMA 的 CoT 是一种 inference-time scaling 的 simplest form。可以做：
- **Best-of-N sampling**: 让 MLLM sample N 个 trajectory, 用一个 learned verifier (类似 PRM in math reasoning) 选 best
- **Search-based planning**: MCTS over $(s_t, k_t)$ action space, 用 MLLM 做 value function 和 policy prior
- 这跟 [AlphaGo 在围棋上做的事](https://www.nature.com/articles/nature24270) 思想上一致

### 9.2 World Model 联合 training
EMMA 的 spirit 是一个 model 做所有事, 这跟 world model (像 [Sora](https://openai.com/sora/) 之类 video generation) 方向相通。如果 MLLM 能 generate future frames (next-frame prediction), 那 trajectory prediction 就是 implicitly encoded 在 future frame generation 里。可以想象一种 AD agent, 它先 "imagine" 5s 后的场景, 再从中 infer 出 ego 轨迹。这是 [GAIA-1 (Wayve, 2023)](https://wayve.ai/think/gaia1/) 的方向。

### 9.3 VLA (Vision-Language-Action) 方向
OpenEMMA 输出 speed/curvature 数值, 跟 Robotics 里的 [RT-2 (Google DeepMind)](https://arxiv.org/abs/2307.15818) 思路相通 — 都是把 action token 化, 让 LLM 直接输出 action token。这跟 [Pi (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0) 也类似。AD 是 robotics 的一个特例, VLA paradigm 适用。

### 9.4 Confidence / Uncertainty Estimation
OpenEMMA 没有 uncertainty estimation, 这在 safety-critical 系统里是大问题。MLLM 的 output confidence 可以从 log-probabilities 取, 但目前没有 calibrate 过。可以参考 [Conformal Prediction](https://arxiv.org/abs/2107.07511) 做 distribution-free uncertainty quantification。

### 9.5 Data Engine / 自动标注
你之前在 Tesla 提过 data engine 的重要性。MLLM 可以做 auto-labeling — 用 GPT-4o 给 driving scene 做 scene description, intent label, 这些 label 可以作为 weak supervision 给小模型 fine-tune。这是 [Weak-to-Strong Generalization (OpenAI, 2023)](https://arxiv.org/abs/2312.09390) 的一个应用场景。

---

## 10. 最后的 Intuition 总结

OpenEMMA 的核心 insight 在我看来是这一个公式:

$$\text{MLLM output space} = \{(s_t, k_t)\}_{t=0}^{T}$$

这个 output representation 是 "human-interpretable, physics-grounded, low-frequency", 跟 MLLM 的 pretraining distribution 对齐。整个 pipeline 的设计哲学是: **leverage MLLM's world knowledge and reasoning, but constrain its output to a structurally simple space that respects vehicle kinematics**。

这跟 直接 fine-tune MLLM 输出 (x, y) coordinates 是两种不同哲学。EMMA 选了后者 (但 fine-tune 了整个 Gemini), OpenEMMA 选了前者 (不 fine-tune, 靠 representation engineering)。这是个 inference-time compute 换 training-time compute 的 tradeoff, 也是当前 open-source community 的一个 pragmatic choice。

如果让我预测未来 6-12 个月, 我觉得会出现:
1. OpenEMMA 的 fine-tuned 版本 (LoRA on LLaVA / Qwen, 在 BDD100K + nuScenes instruction tuning)
2. Closed-loop benchmark 上的 evaluation (nuPlan / CARLA)
3. Multi-camera extension
4. Test-time search / Best-of-N 等方法叠加

这是个非常 "early days" 的工作, 它的 value 更多在于展示 a complete open-source pipeline 可以跑通, 而非 SOTA 性能。它给社区一个 fork-and-extend 的 baseline, 这正是 open-source research 的精髓。

---

## 相关链接汇总

- **Paper repo**: https://github.com/taco-group/OpenEMMA
- **EMMA (closed-source baseline)**: https://arxiv.org/abs/2410.23262
- **nuScenes dataset**: https://www.nuscenes.org
- **YOLO11n (ultralytics)**: https://github.com/ultralytics/ultralytics
- **YOLO3D repo**: https://github.com/ruhyadi/YOLO3D
- **Original YOLO3D paper**: https://arxiv.org/abs/1612.00396
- **DriveVLM**: https://arxiv.org/abs/2402.12289
- **DriveLM**: https://arxiv.org/abs/2312.14150
- **RAG-Driver**: https://arxiv.org/abs/2402.10828
- **GPT-Driver**: https://arxiv.org/abs/2310.01831
- **CoT paper**: https://arxiv.org/abs/2201.11903
- **Self-Consistency**: https://arxiv.org/abs/2203.11171
- **Tree of Thoughts**: https://arxiv.org/abs/2305.10601
- **Qwen2-VL**: https://arxiv.org/abs/2409.12191
- **LLaVA-NeXT**: https://arxiv.org/abs/2407.07895
- **Llama 3.2**: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- **nuScenes paper**: https://arxiv.org/abs/1903.11027
- **Senna**: https://arxiv.org/abs/2410.17815
- **LMDrive**: https://arxiv.org/abs/2402.13289
- **Rethinking Open-loop Eval**: https://arxiv.org/abs/2306.09493
- **RT-2 (VLA)**: https://arxiv.org/abs/2307.15818
- **Weak-to-Strong Generalization**: https://arxiv.org/abs/2312.09390
- **GAIA-1 World Model**: https://wayve.ai/think/gaia1/

希望这能帮你 build 起对 OpenEMMA 以及更广阔 MLLM-for-AD landscape 的 intuition。如果你对其中某一点 (比如 speed-curvature representation 的 alternative, 或者 inference-time scaling 在 AD 里的应用) 想深挖, 我可以再展开讲。
