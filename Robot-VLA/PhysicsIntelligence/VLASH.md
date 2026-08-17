---
source_pdf: VLASH.pdf
paper_sha256: df0a64a49dd257c348f1053311248450095c187e0268dc97f8a18d2d28b27cec
processed_at: '2026-08-13T02:59:43-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VLASH

好，我换种讲法。把这个 paper 当一个故事讲，技术细节嵌在里面。

---

## 这个 paper 在解决一个什么真实问题

你去刷 X 或者 YouTube 上的 VLA demo 视频，仔细看会发现一个事儿：几乎所有 demo 都被加速了 5× 到 10×。你 pause 一下看原始帧，会发现 robot 动作是**一顿一顿的**——往前挪一点、停顿、再挪一点、再停顿。

这个 stuttering 不是 robot 本身的限制，是**inference 机制**造成的。现在 VLA 的部署范式是 synchronous：robot 站着不动，让模型 inference 吐出一个 action chunk（比如未来 50 步的动作），robot 执行前 K 个，执行完停下来，再 inference 下一批。inference 大概 100-500ms，执行期间 robot 是 idle 的——这就是 stall。

更糟的是**反应延迟**。你扔个球过去，robot 要先把当前画面送进 VLA，等 inference 出 chunk，才能开始动。整个链路下来 500ms+，球早飞过去了。所以 VLA 现在能 fold clothes、能 pick and place，但**打不了乒乓球**。

这个问题的本质是：VLA inference 太慢了，慢到没法做 real-time 闭环控制。

---

## 为什么这事之前没被好好解决

async inference 这事其实早就有人想到——你边执行上一个 chunk 边 inference 下一个 chunk，inference 时间就 hide 在执行后面了。听起来很美好，但实际部署会崩。

崩的原因非常 fundamental。你 inference 启动那一刻看到的 robot state 是 $s_t$，但 inference 跑完大概 $\Delta$ 个 control step 之后，新 chunk 才真正开始执行。这时 robot 已经被上一个 chunk 带到了 $s_{t+\Delta}$ 这个新位置。你基于 $s_t$ 生成的 action 被施加在 $s_{t+\Delta}$ 上，state 跟 action **对不上**。

这就好比你开车看后视镜——你看到的是 200ms 前的路况，但你现在要根据它打方向盘。naive async 就是这种"开车看后视镜"，动作会晃、会乱、会失败。

现有 fix 方案各有各的别扭：
- **RTC** ([arXiv:2506.07339](https://arxiv.org/abs/2506.07339)) 在 inference 时做 iterative inpainting，runtime 里有额外开销
- **A2C2** ([arXiv:2509.23224](https://arxiv.org/abs/2509.23224)) 给模型加 correction head，改架构
- **SmolVLA naive async** ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) 直接切 chunk，承认会 unstable

这些方案都在 **inference time 或 architecture level 去打补丁**。但 VLASH 发现一个更深的事儿：模型从来没被训练过去理解"我现在看到的 state 是过去的，我要为未来的 state 生成 action"这个概念。你给它 future state 它不会用，甚至会 ignore——因为标准 fine-tuning 数据里 state 和 action 永远是对齐的，模型学到的就是"看到什么 state 就生成什么 action"，没有时间错位这个概念。

---

## VLASH 的核心 insight，用人类来类比

人打乒乓球也有反应延迟，从视网膜看到球到肌肉开始响应大概 200ms。但人能打。为什么？因为人**用 body 的 internal model 预测**：当我的挥拍动作真正到达击球点时，球大概在哪里、我的手臂大概在哪个位置。我们的大脑一直在 forward simulate 自己的身体状态。

VLASH 给 VLA 装的就是这个能力。但 implementation 非常干净，没学什么 forward model，因为 robot 的 proprioceptive dynamics 是**已知**的——action 就是 delta，state 就是累积：

$$s_{t+\Delta} = s_t + \sum_{i=0}^{\Delta-1} a_{t+i}$$

- $s_t$: inference 启动时的 robot state
- $a_{t+i}$: 上一个 chunk 里第 $i$ 个还没执行的 delta action
- $\Delta$: inference latency（control steps 数）
- $s_{t+\Delta}$: 新 chunk 真正开始执行时的 robot state

这些 $a_{t+i}$ 是上一个 inference 早就吐出来的，**已知**。所以 $s_{t+\Delta}$ 是个**纯算术** ——前向积分，没有任何学习成分。这就是 Fig. 3(c) 那个 $s_3 = s_1 + a_1 + a_2$ 的来历。

Inference 启动时，VLASH 把当前的 visual $o_t$（这是 stale 的）和 rollforward 出来的 future robot state $s_{t+\Delta}$ 一起塞给 VLA。模型为"未来那个时刻"生成 action。这就把 prediction interval 和 execution interval 在 robot state 维度上对齐了。

注意 environment observation 没法 rollforward——球飞到哪儿是 unknown。但实验证明，只 rollforward robot state 已经够稳。这暗示 manipulation task 的短时 dynamics 里，**robot 自身 state 变化比 environment 变化对 action choice 影响大得多**。在几百 ms 的时间尺度上，球还没飞多远，但 robot 的关节已经动了好几度。

---

## 但你直接在 test time 喂 future state，模型不会用

这是 paper 里最有意思的一个发现。作者在 $\pi_{0.5}$ 上做对照实验：fine-tuning 时**完全删掉 state 输入**（只用 visual），在 LIBERO 上的 success rate 居然**比带 state 输入的版本还高**。

这说明 VLA 在标准训练里严重 **overfit visual, underfit state**。Visual 已经把 state 信息间接编码了（你看图就能反推 robot 在哪），额外加 state 反而是噪声。

这就解释了为什么不能只在 test time 喂 future state——模型根本没学会 state 这个 modality 的语义。你给它一个 $s_{t+\Delta}$，它不知道这是什么意思。

所以 VLASH 必须从训练端动手。

---

## Training trick:制造"视觉等价类"逼模型学 state

VLASH 的 fine-tuning augmentation 非常简单。给定轨迹 $\{(o_t, s_t, a_t)\}$，标准训练是：

$$(o_t, s_t) \mapsto a_{t:t+H-1}$$

VLASH 改成随机采样 offset $\delta \in \{0, 1, \ldots, \Delta_{\max}\}$，构造：

$$(o_t, s_{t+\delta}) \mapsto a_{(t+\delta):(t+\delta+H-1)}$$

- $o_t$: 当前 frame 的 visual，**不变**
- $s_{t+\delta}$: 真实 trajectory 里 $\delta$ 步之后的 robot state
- $a_{(t+\delta):(\cdot)}$: 对应的 future action chunk
- $\delta$: 随机 offset，覆盖部署时可能的最大 inference delay $\Delta_{\max}$

关键 trick 是 visual $o_t$ 固定，state 变。同一张图对应不同 state $s_{t+\delta}$，对应**不同**的 ground truth action。模型没法再光看图猜 action 了——同一张图对应多个合法 action，只有 state 能区分。

这是在**主动制造 ambiguity** 来强制模型 attend 到 state。这个思路其实在 contrastive learning 里很常见——你让任务在某个 modality 上 underdetermined，模型就被迫去用另一个 modality。VLASH 把这个 idea 用在 supervised fine-tuning 上。

随机 $\delta$ 而不是固定 $\delta$ 是因为部署硬件可能不同，inference delay $\Delta$ 不确定。训练时见过 $\delta \in [0, \Delta_{\max}]$ 任意值，部署时无论 $\Delta$ 是几都能 work。而且 $\delta = 0$ 退化成标准训练，所以 sync inference 的性能自动保留——这点在 Table 3 验证了：VLASH fine-tuned 模型在 sync 下 96.6 vs standard 96.8，几乎无损。

---

## Shared observation packing:把训练效率拉回来

这个 augmentation 的 naive 实现有个浪费：每个 $\delta$ 都要独立 forward 一次 VLA，但 visual $o_t$ 是共享的，vision encoder 是 VLA 最贵的一部分。$\pi_{0.5}$ 用 SigLIP + 2 张图 + language prompt，observation 大概 700 个 token；state + action chunk 只有 50 个 token。重复 encode observation 是巨大浪费。

VLASH 的解法是 **block-sparse attention packing**（Fig. 4）。把序列打包成：

$$[o_t, \ (s_t, A_t), \ (s_{t+1}, A_{t+1}), \ \ldots, \ (s_{t+\Delta_{\max}}, A_{t+\Delta_{\max}})]$$

Attention mask：
- Observation tokens 之间互相 attend（标准 VLA 行为）
- 每个 offset branch $(s_{t+\delta}, A_{t+\delta})$ 可以 attend 所有 observation 和 branch 内部
- 不同 offset branch 之间互相 mask

Position encoding 每个 branch 都从 observation token 长度开始算——从模型视角，这等价于 $N_\delta$ 个独立训练样本，但 observation 只 encode 一次。

效果在 Table 3：每个 training step 从 420.99ms 降到 129.29ms，**3.26× speedup**。effective batch size 5× 增长，token 数量只增 ~20%。这是非常干净的工程优化。

这个 pattern 在 LLM 里也有——[FlashAttention](https://arxiv.org/abs/2307.08691) 的 KV cache reuse、[FlexAttention](https://arxiv.org/abs/2405.13753) 编译任意 attention pattern。本质都是"共享 expensive prefix + 多个 cheap branches"。VLASH 把这个 trick 自然地用到了 VLA fine-tuning 上。

---

## Action quantization:把执行速度极限再往前推

async inference 把 inference latency 隐藏掉之后，瓶颈变成 robot 物理执行速度。VLA 在 50Hz teleoperation 数据上训练，每个 action 是 fine-grained micro-step。但任务其实不需要这么细——很多中间 waypoint 是冗余的。

VLASH 把 LLM weight quantization 的思想搬到 action 上。对 quantization factor $q$：

$$\hat{a}_i = a_{iq} + a_{iq+1} + \cdots + a_{(i+1)q-1}$$

- $a_j$: 原始 fine-grained delta action
- $q$: 连续合并的 action 数
- $\hat{a}_i$: 第 $i$ 个 macro-action
- 因为是 delta action，累加 = 净位移

$q=3$ 意味着每 3 个 micro-action 合并成 1 个 macro-action，robot 从第 0 个的起点直接走到第 2 个的终点，跳过中间 2 步。控制频率从 50Hz 降到 50/3 ≈ 17Hz，但每个 step 移动距离 3×。

这个类比的 deep 在哪：[GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438) 都在说同一件事——**当 numeric representation 的精度超过任务需求时，quantize 是 free lunch**。VLA 的 action sequence 也有同样的 over-specification：50Hz 训练数据细到 robot 真不需要的精度。Quantize 它就免费提速。

实验 trade-off（Fig. 7 + Table 1）：
- $q=1$（不 quantize）：1.12× speedup, 94% score
- $q=2$：2.03× speedup, 还是 94%——**free lunch 区**
- $q=3$：2.67× speedup, 掉 4.7% score——开始付费

跟 LLM INT8/INT4 完全一个 trade-off curve，都有 sweet spot。

---

## 几个实验数据点让我印象深刻

**Kinetix (Fig. 6)**：高动态 benchmark，包含 throwing / catching / balancing。delay = 4 steps 时：
- Naive async: 51.2%
- VLASH: 81.7%
- Sync upper bound: ~85%

**+30.5% 绝对提升**，几乎追平 sync 的 upper bound。这种 gap 在 dynamic task 上是巨大的，说明 future-state-awareness 在高 dynamic regime 下几乎是 must-have。

**Reaction latency (Table 2)**：50Hz 控制，$K=25$，execution 500ms。

| GPU | Sync | Async | Speedup |
|---|---|---|---|
| RTX 5090 | 530ms | 30ms | 17.4× |
| RTX 4090 | 536ms | 36ms | 14.9× |
| RTX 5070 | 564ms | 64ms | 8.8× |

Sync 反应要 530ms，乒乓早飞过去了。Async 30ms，在 5090 上接近人类反应阈值。这是 ping-pong demo 能 work 的物理基础。

**LIBERO $\pi_{0.5}$ (Table 1)**：delay=1, 2 时 SR 居然**比 sync 略高**（97.2% vs 96.8%）。这暗示 future state 在某些 task 上不只是 compensation，而是提供了更好的 planning context——模型"预见到自己未来的位置"反而规划得更好。这点很有意思，可能跟 MPC 里 receding horizon 比 single-step 优是同一个道理。

---

## ping-pong demo 为什么重要

paper 的 headline demo 是用 $\pi_{0.5}$ + VLASH 跟人对打乒乓。看 Figure 1 的 frame sequence，第三帧 robot 已经开始反应了——这是亚秒级 perception-to-action loop。

这事在 VLA 领域是**第一次**。之前的乒乓球 robot 都是 classical hierarchical planning + trajectory optimization——[Hitter humanoid ping-pong](https://arxiv.org/abs/2508.21043)、[Galaxea G0](https://arxiv.org/abs/2509.00576) 都走这条路。VLASH 证明纯 VLA + async inference 也能做到，而且是 generalist policy——同一个 $\pi_{0.5}$ 既会打乒乓，又会 fold clothes、会 sort cubes。

Symbolic value 在于：VLA 从 "slow manipulation" regime 跨进了 "dynamic physical interaction" regime。这是 robot RL 几十年想做的事，VLA 用一个 training trick 就跨过去了。

---

## 我个人觉得这 paper 真正的洞察

**最深的洞察不是 future-state-awareness 本身**，而是 Sec. 4.2 那个观察：**VLA 在标准 fine-tuning 里 under-use state**。删掉 state 输入性能反而更高，这暴露了当前 VLA 训练的一个根本问题——proprioceptive signal 在 visual-rich 数据里被 marginalize 了。

VLASH 的 offset augmentation 是一个**间接 fix**——通过制造 visual ambiguity 逼模型 attend state。但这只是治标。更激进的解法可能是：
- 在 data augmentation 上加 state perturbation
- 用 state 做 conditioning（FiLM / AdaRMSNorm），而不是简单 concat 进 sequence
- 在 architecture 层面给 state 一个 privileged channel

Sec. 7.4 提到把 state embedding inject 到 AdaRMSNorm 作为 conditioning signal——这其实就是个起步。我赌后续工作会沿这个方向走，因为 state under-utilization 是比 async inference 更根本的问题。

---

## 这 paper 在 VLA 演化里站在哪

我把 VLA 发展分三阶段看：

1. **Capability phase**: RT-2, OpenVLA, $\pi_0$, $\pi_{0.5}$, [Gr00t N1](https://arxiv.org/abs/2503.14734), [Gemini Robotics](https://arxiv.org/abs/2503.20020) ——证明 VLA 能解复杂 task
2. **Efficiency phase**: TinyVLA, SmolVLA, token pruning——让 VLA 跑得动
3. **Real-time phase**: RTC, A2C2, **VLASH**——让 VLA 跑得**实时**，能做 dynamic interaction

VLASH 是 real-time phase 里目前最干净的方案，因为：
- 不改架构（对 $\pi_0$, SmolVLA）
- 不加 inference overhead
- Fine-tuning 时只用一个 data augmentation trick
- 对 $\pi_{0.5}$ 这种把 state 转 text token 的设计，可选加一个 zero-init state projection（Sec. 7.4）

"零开销 + 不改架构 + 一个 training trick" 这种 elegant 的方案通常是 field 成熟的标志。这告诉我一个重要 signal：**VLA 已经强到需要研究 real-time deployment 了**。几年前大家在争论 VLA 能不能 fold clothes，现在大家在争论 VLA 能不能打 ping-pong。这个 shift 本身比 VLASH 的具体 trick 更重要。

---

## 几个我会关注的延伸方向

**1. Environment rollforward**：VLASH 只 rollforward robot state。要打真正的 dynamic interaction（抓飞行物、对抗），environment 也得 rollforward。自然会想到学一个 latent dynamics model 来推 environment representation，[DreamerV3](https://arxiv.org/abs/2301.04104) / [TD-MPC](https://arxiv.org/abs/2304.01601) 那套思路搬到 VLA inference 上。VLASH 就变成 model-based RL 的 inference-time 版本。

**2. Adaptive quantization**：$q=2$ 是 free lunch 区，但这是 task-dependent。Precision assembly 可能 $q=1$ 最优，throwing 可能 $q=5$ 都没问题。让模型或 controller 自适应选 $q$——类似 LLM 的 mixed-precision——可以让 single policy 跑多种速度档。

**3. State under-utilization 的根本 fix**：如前述，给 state 一个 privileged conditioning channel 而非 concat 进 sequence。这个方向可能催生新架构。

**4. Diffusion policy 的 async**：$\pi_{0.5}$ 是 flow matching，multi-step denoising 本身就引入"内部"delay。VLASH 现在把 denoising delay + network latency 一起处理，但更细的方案可能是 async denoising 内部的不同 step——inference-time 的 pipeline parallelism。

**5. VLA + world model 融合**：如果 environment rollforward 用 learned dynamics model，那个 model 本身就是个 world model。VLASH 的未来版本可能是 VLA + world model 联合训练，inference 时用 world model 做 rollforward——这就跟 [Dreamer](https://arxiv.org/abs/2301.04104) 在 VLA 上的实现等价了。

---

## 一句话总结

VLASH 说的是：**别在 inference time 修 async misalignment，让模型在 training time 学会处理它**。一个 state rollforward 加一个 temporal offset augmentation，干净利落地把 VLA 从"slow manipulation"推进到"real-time dynamic interaction"。方法本身 elegant，更深远的意义是它揭示了 VLA field 正在从"能不能做"转向"能不能实时做"——这是 field 成熟的标志。

Code 在 [github.com/mit-han-lab/vlash](https://github.com/mit-han-lab/vlash)，作者是 Song Han 组 + NVIDIA + Tsinghua，这种"deep insight + clean method + killer demo"的组合，我很久没在 robotics paper 里见到了。

---

# VLASH: 让 VLA 跑实时反应控制的核心思路

Andrej, 这篇 paper 我觉得思路非常干净,它抓住的是 VLA 部署里一个被人长期忽视但本质很 fundamental 的问题——**prediction interval 与 execution interval 的 temporal misalignment**。它没有改架构、没有加 head、没有 inpainting,只用一个 training trick + 一个推理时的 state rollforward 就解决了。下面我把整条 pipeline 拆开讲,并尽量把直觉和公式对齐起来。

---

## 1. 问题本质:为什么 sync inference 慢,async inference 又不稳

### 1.1 Synchronous inference 的 stall 问题

Action chunking policy 的形式:

$$\pi_\theta(A_t \mid o_t, s_t), \quad A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$$

- $\pi_\theta$: 参数为 $\theta$ 的 policy
- $o_t$: 当前 environment observation (图像 / multi-view)
- $s_t$: robot proprioceptive state (joint positions, gripper state)
- $A_t$: 长度为 $H$ 的 action chunk (prediction horizon)
- $t$: controller timestep

标准部署只执行前 $K \leq H$ 个 action($K$ 叫 execution horizon),然后重新 inference。问题在于 inference 期间 robot 是 idle 的——它在等网络吐 chunk。这导致 stop-and-go, demo 视频经常 5–10× 加速来掩盖这种 stuttering。

### 1.2 Asynchronous inference 与 misalignment

Async inference 让 robot 一边执行上一个 chunk 一边做下一个 chunk 的 inference,inference 时间被 hide 在 execution 后面。但出现一个根本性 misalignment,见 Fig. 2:

$$I_t^{\text{pred}} = [t, t+K) \quad \text{vs.} \quad I_t^{\text{exec}} = [t+\Delta, t+\Delta+K)$$

- $I_t^{\text{pred}}$: 模型在 inference 启动时假设的"目标时间窗口"
- $I_t^{\text{exec}}$: actions 真正被执行的时间窗口
- $\Delta > 0$: inference latency,以 control step 为单位

也就是说模型是基于 $s_t$ 这个**已经在执行时刻过期**的 state 生成 actions 的,但这些 actions 实际要作用于 $s_{t+\Delta}$ 那个未来 state。这就是 Fig. 3(b) 画的情况——naive async 看到的 state $s_1$ 是 stale 的,真正执行时 robot 已经到了 $s_3$,所以 actions 跟不上。

### 1.3 现有方案的痛点

- **RTC** [Black et al. 2025, arXiv:2506.07339](https://arxiv.org/abs/2506.07339): freeze guaranteed-to-execute actions, inpaint 剩余的。引入 runtime overhead,部署复杂。
- **A2C2** [Sendai et al. 2025, arXiv:2509.23224](https://arxiv.org/abs/2509.23224): 加 correction head,改架构,有 overhead。
- **Naive async (SmolVLA)** [Shukor et al. 2025, arXiv:2506.01844](https://arxiv.org/abs/2506.01844): 直接切 chunk,严重 misalignment。

VLASH 的关键洞察:**所有这些方法都在"修"模型,但模型从来没被训练去理解"我现在看到的 state 是过去的, 我要为未来的 state 生成 action"。**

---

## 2. VLASH 的核心 idea:Future-State-Awareness

### 2.1 State rollforward 的几何直觉

看 Fig. 3(c) 那张图,这是整篇 paper 的灵魂。当 inference 在 $s_1$ 启动时,robot 还会执行剩余的 $a_1, a_2$(previous chunk 的尾巴),然后新 chunk 才接管。这两个 action 是**已知**的——上一个 inference 早就生成完了。

所以执行时刻的 robot state 是可计算的:

$$s_{t+\Delta} = s_t + \sum_{i=0}^{\Delta-1} a_{t+i}$$

- $s_t$: 当前 inference 启动时刻的 robot state
- $a_{t+i}$: 上一个 chunk 中第 $i$ 个待执行的 delta action
- $\Delta$: inference 延迟(以 control step 计)
- $s_{t+\Delta}$: 新 chunk 真正开始执行时刻的 robot state

这是**纯前向模拟**,完全 free——没有 forward model,没有学习,就是把已知的 delta action 累加到当前 state 上。在 Fig. 3(c) 里就是 $s_3 = s_1 + a_1 + a_2$。

**直觉类比**: 人类也有 ~200ms 的视觉-运动 reaction delay。我们打乒乓球时不会等看到球的最新位置再挥拍——我们会用 internal model 预测"当我的挥拍动作真正到达接触点时,球大概在哪里"。VLASH 让 VLA 具备同样的能力:用 body state 的 rollforward 来补偿 visual + inference delay。

### 2.2 但 environment 是未知的——为什么这样还能 work?

关键区分:**robot state 可前向 rollforward,environment observation 不可**。但实验表明,只 rollforward robot state 就足够稳。这暗示 VLA policy 在大多数 manipulation 任务里,environment 的短时间内变化(几百 ms 内)对 action choice 的影响远小于 robot 自身 state 的影响。

这其实呼应了 [OpenVLA / π₀ / SmolVLA 这一系] 的一个隐含事实:很多 manipulation task 是 quasi-static 的,environment 的 visual 在 200ms 内基本不变,真正变化的是 robot 自身姿态。所以补偿 robot state 已经捕获了大部分 misalignment。

### 2.3 一个关键的负面发现:VLA under-use state

作者发现一个反直觉的现象,在 Table 1 的注释里:

> "fine-tuning without state input (visual only) consistently outperforms fine-tuning with state input on LIBERO"

也就是说 $\pi_{0.5}$ 在 LIBERO 上,把 robot state 完全删掉,性能反而更高。这说明 VLA 在训练中**过度依赖 visual** 而**忽略 state**——既然 visual 已经隐含 state,加 state 反而是噪声。

这就解释了为什么"单纯在 test time 喂 future state"行不通——模型根本没学会怎么用 state 这个输入。所以 VLASH 必须从训练端动手。

---

## 3. Training-side fix:Temporal Offset Augmentation

### 3.1 数据构造方式

给定轨迹 $\{(o_t, s_t, a_t)\}$,标准 fine-tuning 训练:

$$(o_t, s_t) \mapsto a_{t:t+H-1}$$

VLASH 改成 temporal offset augmentation:

$$\delta \sim \text{Uniform}\{0, 1, \ldots, \Delta_{\max}\}$$
$$(o_t, s_{t+\delta}) \mapsto a_{(t+\delta):(t+\delta+H-1)}$$

- $\delta$: 随机采样的时间偏移量
- $\Delta_{\max}$: 最大偏移(覆盖部署时可能的最大 inference delay)
- $o_t$: 视觉 observation **不变**
- $s_{t+\delta}$: 用真实 trajectory 里的 future robot state
- $a_{(t+\delta):(\cdot)}$: 对应的 future action chunk

**关键**: 同一张 image $o_t$ 配上不同 $\delta$ 的 state $s_{t+\delta}$,对应**不同**的 ground-truth action chunk。模型没法只靠 visual 来预测 action 了——它必须 attend 到 state 上,因为同一 visual 对应多个合法 action,只有 state 能区分。

这本质上是在做 **conditional ambiguity injection**: 制造视觉上的等价类,强制模型学习 state-conditioned policy。这个思路让我想到 contrastive learning 里强制模型关注 hard negative 的策略——你让 task 在某个 modality 上变得 underdetermined,模型被迫用另一个 modality。

### 3.2 为什么 $\delta$ 要随机采样

部署时 inference delay $\Delta$ 是不确定的——不同 GPU、不同 batch size、不同控制频率,都对应不同 $\Delta$。固定 $\delta$ 训练会让模型 overfit 到一个特定 delay。随机 $\delta \in [0, \Delta_{\max}]$ 让模型学会"任意 future state 都可以是 input",这样 deployment 时无论 $\Delta$ 是几都能工作。

更妙的是:当 $\delta = 0$ 时退化成标准 training,所以 sync inference 的性能**自动保留**。Table 3 的对比就证明了这点——VLASH fine-tuned 模型在 sync inference 下 $\Delta=0$ 时,跟 standard fine-tuned 表现一致(96.6 vs 96.8)。

---

## 4. Efficient Fine-tuning:Shared Observation Packing

### 4.1 Naive 实现的低效

如果对每个 $\delta$ 独立跑一次 VLA forward,$o_t$ 会被编码 $N_\delta$ 次,而 vision encoder 是 VLA 最贵的部分。$\pi_{0.5}$ 用 SigLIP + 2 张图 + language prompt,光 observation token 就约 700 个;state+action chunk 只有约 50 个 token。重复编码 observation 是巨大浪费。

### 4.2 Block-sparse packing

VLASH 把 sequence 打包成:

$$[o_t, \ (s_t, A_t), \ (s_{t+1}, A_{t+1}), \ \ldots, \ (s_{t+\Delta_{\max}}, A_{t+\Delta_{\max}})]$$

Attention mask 结构(Fig. 4):
- **Observation tokens 之间**:全互相 attend(标准 VLA behavior)
- **每个 offset branch 内部** $(s_{t+\delta}, A_{t+\delta})$:可以 attend 所有 observation tokens,加上自己 branch 内的 tokens
- **不同 offset branches 之间**:互相 mask 掉,不 attend

**Positional encoding 技巧**: 每个 branch 的 $(s_{t+\delta}, A_{t+\delta})$ 的 position 都从 observation token 长度开始算。从模型视角,这等价于 $N_\delta$ 个独立的 $(o_t, s_{t+\delta}, A_{t+\delta})$ 训练样本——但 observation 只编码一次。

### 4.3 数量级收益

$N_\delta = 5$ 个 offsets:
- Token 数量增长: $\sim 50 \times 5 = 250$ 个额外 token,相对 700 个 observation 是 ~20% 增长
- 有效 batch size: 5× 增长
- Per-step 时间:Table 3 显示 3.26× speedup(从 420.99ms 降到 129.29ms per step)

这意味着在**相同 effective batch size** 下,VLASH 的 wall-clock training time 远短于 baseline。这种"共享 expensive prefix + 多个 cheap branches"的 pattern 在 LLM 里也常见——比如 [Flash Attention](https://arxiv.org/abs/2205.14135) 里的 KV cache reuse,或者 [FlexAttention](https://arxiv.org/abs/2405.13753) 编译任意 attention pattern。VLASH 这里的 block-sparse mask 是同一个家族的思路。

---

## 5. Action Quantization:把推理速度极限再往前推

### 5.1 动机

Async inference 已经把 inference latency 隐藏掉了。瓶颈变成 robot 物理执行速度。VLA 通常在 ~50Hz teleoperation 数据上训练,每个 action 是 fine-grained micro-step。但任务其实不需要这么细——很多中间 waypoint 是冗余的。

### 5.2 公式

对 quantization factor $q$:

$$\hat{a}_i = a_{iq} + a_{iq+1} + \cdots + a_{(i+1)q-1}$$

- $a_j$: 原始 fine-grained delta action
- $q$: 量化因子(连续合并的 action 数)
- $\hat{a}_i$: 第 $i$ 个 macro-action,等价于把 $q$ 个连续 micro-action 累加
- 因为是 delta action,累加 = 净位移

Fig. 5 的例子 $q=3$: $\hat{a}_0 = a_0 + a_1 + a_2$,robot 从 $a_0$ 起点直接走到 $a_2$ 终点,跳过中间 2 步。

### 5.3 与 LLM weight quantization 的类比

这是 paper 里我最喜欢的一个类比:
- LLM: 16-bit weights → 8-bit/4-bit,精度下降小,推理大幅加速([GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438))
- VLA: 50Hz fine-grained actions → q× coarser macro-actions,精度下降小,执行大幅加速

这个类比非常 deep,因为它本质都在说:**numeric representation 的精度超过任务需求时,quantize 它是 free lunch**。

### 5.4 实验 trade-off

Table 1 和 Fig. 7:
- $q=1$ (no quantization): 1.12× speedup, 94% score
- $q=2$: 2.03× speedup, 仍 94% 左右 score —— 这就是 free lunch 区
- $q=3$: 2.67× speedup, 掉 4.7% score —— 开始付费

这种 trade-off curve 跟 LLM 的 INT8 / INT4 quant 非常像,存在一个 sweet spot。

---

## 6. 实验数据深入解读

### 6.1 Kinetix (Fig. 6) —— 高动态基准

Kinetix 是 [Matthews et al. 2025](https://arxiv.org/abs/2410.23256) 的动态任务 benchmark (throwing, catching, balancing)。每个点 1024 rollouts,12 tasks。

最关键的数据点 (delay = 4):
| Method | Success Rate |
|---|---|
| Naive Async | 51.2% |
| RTC | (degrades rapidly) |
| **VLASH** | **81.7%** |
| Sync (upper bound) | ~85% |

**+30.5% 绝对提升**,几乎追平 sync 的 upper bound。这数字非常惊人,说明 future-state-awareness 不只是一个 trick,它在 high-dynamics regime 下几乎是必须的。

### 6.2 LIBERO with $\pi_{0.5}$ (Table 1)

| Delay | SR | Time | Speedup | ΔSR |
|---|---|---|---|---|
| Sync | 96.8% | 8.4s | — | — |
| VLASH Δ=1 | 97.2% | 7.2s | 1.17× | +0.4 |
| VLASH Δ=2 | 97.1% | 6.4s | 1.31× | +0.3 |
| VLASH Δ=3 | 94.6% | 5.7s | 1.47× | -2.2 |
| VLASH Δ=4 | 93.1% | 5.8s | 1.45× | -3.7 |

注意 Δ=1, 2 时**性能反而略升**——这说明 async + future state awareness 在某些 task 上甚至比 sync 更好,可能因为 future state 提供了更好的 planning context。Δ≥3 后开始衰减,因为 future state 估计误差累积。

### 6.3 LIBERO with SmolVLA-450M (Table 4)

SmolVLA 是 [HuggingFace LeRobot 团队](https://arxiv.org/abs/2506.01844)的 450M 模型,这里证明 VLASH 对不同架构 generalize:

| Delay | SR | Speedup |
|---|---|---|
| Sync | 78.96% | — |
| VLASH Δ=3 | 79.06% | 1.35× |

Δ=3 时 SR 比 sync 还略高 +0.10。这个 generalization 很重要——说明方法不依赖 $\pi_{0.5}$ 特殊架构。

### 6.4 反应延迟 (Table 2)

控制频率 50Hz, $K=25$, execution duration ~500ms。最大 reaction latency = "看到环境变化到做出反应" 的延迟:

| GPU | Sync (ms) | Async (ms) | Speedup |
|---|---|---|---|
| RTX 5090 | 530.4 | 30.4 | **17.4×** |
| RTX 4090 | 536.1 | 36.1 | 14.9× |
| RTX 5070 | 564.1 | 64.1 | 8.8× |

Sync 反应要 530ms——乒乓球早就飞过去了。Async 只要 30ms,在 RTX 5090 上已经接近人类的 reaction threshold。这就是为什么能打 ping-pong。

### 6.5 Fine-tuning efficiency (Table 3)

| Method | Time/Step (ms) | 10K | 20K | 30K |
|---|---|---|---|---|
| Original | 420.99 | 94.1 | 97.1 | 96.8 |
| VLASH | 129.29 | 87.1 | 94.4 | 96.6 |
| Speedup | 3.26× | | | |

早期收敛慢,因为 effective batch 变小了(per-GPU physical batch 4 vs 16)。但 30K step 后追平,且每步 3.26× 快——总 wall-clock training time 大幅缩短。这个观察也很重要:VLASH 在 sync inference 下评估也基本无损 (96.6 vs 96.8),证明 offset augmentation 不破坏原始能力。

---

## 7. 与相关工作的 positioning

### 7.1 vs RTC [arXiv:2506.07339](https://arxiv.org/abs/2506.07339)

RTC 的思路是 inference-time fix:freeze 必然要执行的前几个 action,然后 inpaint 剩余的。它在 inference 时跑一个额外的 iterative refinement,有 overhead,且不能彻底消除 misalignment——因为 inpainting 仍基于过期 visual。

VLASH 把 fix 推到 training time,让模型自己学会处理 misalignment。Inference 时 zero overhead。这是范式上的区别:RTC 是 test-time patch,VLASH 是 train-time capability。

### 7.2 vs A2C2 [arXiv:2509.23224](https://arxiv.org/abs/2509.23224)

A2C2 加 correction head,改架构。VLASH 不改架构(对 $\pi_0$, SmolVLA),只在 $\pi_{0.5}$ 上可选加一个 zero-init 的 state projection (见 Sec. 7.4)——zero-init 保证初始不破坏 pretrained behavior。

### 7.3 vs SmolVLA naive async [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)

SmolVLA 自己有 naive async 实现,作者承认 misalignment 严重。VLASH 实际上可以看作 SmolVLA async 想要实现但没做到的"正确版本"。

### 7.4 与 RL 里的 model-based rollforward 的关系

VLASH 的 state rollforward 在思想上非常类似 [TD-MPC](https://arxiv.org/abs/2304.01601) 或 [Dreamer](https://arxiv.org/abs/1912.01603) 里的 latent rollout——用 learned/known dynamics 把 state 往前推。差别是 VLASH 只推 robot proprioception,不推 environment,且 dynamics 是已知的(delta action 累加),不需要学。这是 manipulation task 的特殊性带来的简化。

### 7.5 与 LLM quantization 的类比 (作者明确点出)

[AWQ (Lin et al.)](https://arxiv.org/abs/2306.00978), [GPTQ (Frantar et al.)](https://arxiv.org/abs/2210.17323), [SmoothQuant (Xiao et al.)](https://arxiv.org/abs/2211.10438) 都是同一思想:精度过剩时,quantize 是 free lunch。Action quantization 是这个思想在 control 频域上的迁移。这个类比非常 deep,我猜后续会有人把这个 idea 推到更激进的 action representation 上(比如学习离散 codebook, 类似 VQ-VAE)。

### 7.6 与 ping-pong robot 的相关 work

[Galaxea G0](https://arxiv.org/abs/2509.00576), [Hitter (humanoid table tennis)](https://arxiv.org/abs/2508.21043) 这些工作主要靠 classical hierarchical planning +专门的轨迹优化来做乒乓球。VLASH 证明纯 VLA + async inference 也能做到,且是 generalist policy——这个 demo 的 symbolic value 很大。

---

## 8. 我对方法的几点评议

### 8.1 真正的洞察:state under-utilization

最深的洞察是 Sec. 4.2 那个观察:VLA overfit visual,underfit state。这其实暴露了当前 VLA 训练的一个根本问题——proprioceptive signal 在 visual-rich 训练数据里被 marginalize 了。VLASH 的 offset augmentation 是一个间接 fix,但我认为更激进的解法可能是:
- 在 data augmentation 上加 state perturbation
- 用 state 做 conditioning(像 FiLM/AdaRMS),而不是简单 concat
- Sec. 7.4 提到的把 state inject 到 AdaRMSNorm 就是一个起步,我猜后续工作会沿这个方向走

### 8.2 Environment drift 的未解决问题

VLASH 只补偿 robot state,environment 不补偿。在 Kinetix 这种高动态环境里这个简化已经够 work,但真正的 dynamic interaction (e.g., 抓飞行物、对抗) 里 environment drift 不可忽略。下一步自然是学一个 latent dynamics model 来 rollforward environment representation,类似 [DreamerV3](https://arxiv.org/abs/2301.04104)。这样 VLASH 就变成 model-based RL 的 inference-time 版本。

### 8.3 Action quantization 的极限

$q=2$ 是 free lunch,$q=3$ 开始付费。但这是 task-dependent 的——precision assembly 任务的 sweet spot 可能 $q=1$,而 throwing 任务可能 $q=5$ 都没问题。一个开放问题是让模型**自适应** quantization factor,类似 LLM 里的 mixed-precision。这可以让 single policy 跑多种速度档。

### 8.4 与 diffusion-based VLA 的关系

$\pi_{0.5}$ 是 flow matching 模型,inference 是 iterative denoising。VLASH 的 async 框架对 diffusion policy 也 work,因为 flow matching 也是 chunk generation。但对 single-step policy (e.g., 纯 transformer) 更直接。这里有个微妙点:diffusion 的 multi-step denoising 本身就引入了"内部" delay,VLASH 是把这个 delay 加上 network latency 一起处理。

---

## 9. 一些 implementation 细节

### 9.1 Real-world deployment 配置

- Robot: Galaxea R1 Lite (dual 7-DOF arm), LeRobot SO-101 (6-DOF)
- GPU: 笔记本 RTX 4090 / 5090 / 5070
- Inference: torch.compile + CUDAGraph + kernel fusion ([PyTorch 2](https://arxiv.org/abs/2305.11685))
- $\pi_{0.5}$ 的特殊处理: state projection 而非 tokenizer (Sec. 5.2),因为 $\pi_{0.5}$ 原本把 state 转 text token,会破坏 numerical structure

### 9.2 Training 配置 (Table 5)

- Batch size: 32
- Steps: 30K
- Optimizer: AdamW, lr=5e-5, betas=[0.9, 0.95], wd=1e-10
- Scheduler: cosine decay with 1K warmup, peak 5e-5, decay 到 2.5e-6

这个配置非常保守,说明 VLASH 不需要 hyperparameter tuning——直接 plug 进现有 fine-tuning pipeline 就 work。

### 9.3 GPU 选择的意义

Table 2 在 5090 / 4090 / 5070 上对比非常关键——它证明 async 的收益是 hardware-agnostic 的。5070 上 inference latency 64ms,async 仍能 8.8× speedup。这暗示未来边缘部署(VLA on robot 本体)也能受益,不需要 H100。

---

## 10. 这篇 paper 在 VLA 演化中的位置

我把 VLASH 放在 VLA 演进的三阶段里:

1. **Capability phase**: RT-2, OpenVLA, $\pi_0$, $\pi_{0.5}$, Gr00t, Gemini Robotics —— 证明 VLA 能解复杂 task
2. **Efficiency phase**: TinyVLA, SmolVLA, token pruning —— 让 VLA 跑得动
3. **Real-time phase**: RTC, A2C2, VLASH —— 让 VLA 跑得**实时**,能做 dynamic interaction

VLASH 是 real-time phase 里目前最干净的方案。它告诉我一个 important signal:**VLA 已经强到需要研究 real-time deployment 了**,这是 field 成熟的标志。几年前大家在争论 VLA 能不能 fold clothes,现在大家在争论 VLA 能不能打 ping-pong。这个 shift 本身就比 VLASH 的具体 trick 更重要。

而 ping-pong demo 的 symbolic 意义在于:它第一次把 VLA 从"slow manipulation" 拉到 "dynamic physical interaction" 的 regime。这是 robotic RL 几十年想做的事,VLA 用一个 training trick 就跨过去了。后续大概率会有一波工作沿着 dynamic VLA 这个方向涌现。

---

## References

- VLASH paper (本篇)
- [π₀.5 — Physical Intelligence](https://arxiv.org/abs/2504.16054)
- [SmolVLA — HuggingFace LeRobot](https://arxiv.org/abs/2506.01844)
- [RTC: Real-Time Chunking](https://arxiv.org/abs/2506.07339)
- [A2C2: Leave No Observation Behind](https://arxiv.org/abs/2509.23224)
- [Kinetix benchmark](https://arxiv.org/abs/2410.23256)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [Gr00t N1 — NVIDIA](https://arxiv.org/abs/2503.14734)
- [Gemini Robotics](https://arxiv.org/abs/2503.20020)
- [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) / [FlexAttention](https://arxiv.org/abs/2405.13753)
- [PyTorch 2 (torch.compile)](https://arxiv.org/abs/2305.11685)
- [DreamerV3](https://arxiv.org/abs/2301.04104) / [TD-MPC](https://arxiv.org/abs/2304.01601)
- [Galaxea G0](https://arxiv.org/abs/2509.00576) / [Hitter humanoid ping-pong](https://arxiv.org/abs/2508.21043)
- [VLASH GitHub repo](https://github.com/mit-han-lab/vlash)
