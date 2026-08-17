---
source_pdf: Training-Time Action Conditioning for Efficient Real-Time Chunking.pdf
paper_sha256: 02963d5ae4e060e2be3cc8299f5e63046cdeafcb72127242c79fcb3331fa0108
processed_at: '2026-08-12T18:09:14-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，Karpathy，我换个画风，用大白话把这篇 paper 的故事讲一遍。

---

## 一句话说清楚

机器人控制有个老大难问题：模型太大，跑一次要 100 多毫秒，但机器人要 50Hz 的控制频率，等不起。之前的解法 RTC 是在推理时用 inpainting 来"补齐"chunk 之间的缝隙，但这个 inpainting 本身又要额外算 backprop，等于拆东墙补西墙。这篇 paper 的 idea 特别简单：**既然推理时反正要面对 delay，不如训练时就模拟这个 delay，让模型提前学会"接到上一步的接力棒"怎么往下跑**。推理时就变成普通 forward pass，啥额外开销都没有。

---

## 问题到底是啥

想象你在打游戏，比如打 FPS。你的操作和画面之间如果有 100ms 延迟，你会觉得很难受——你开枪了，但画面上敌人已经移走了。机器人控制也是一样的。

VLA 模型现在都是 billion 参数级别的，比如 $\pi_{0.6}$。你给它一张图、一句话，它输出未来 H 个 action。但问题是从你给它输入到它吐出结果，已经过了 100ms。这 100ms 内机器人在干啥？它得继续执行上一次的指令，但环境已经变了，上一次的指令可能已经过时了。

这就好比开车看后视镜——你看到的永远是过去的画面，等你反应过来，车已经偏了。

### Action Chunking 怎么帮忙但也怎么添乱

Action Chunking 的 idea 是：别一个 action 一个 action 地预测，一次性预测一整段（比如 8 步），然后慢慢执行。好处是 trajectory 平滑——一段连续的曲线，不会一顿一顿的。

但坏处是：执行这 8 步的时候，模型是"闭眼"的，它不看环境。如果第 3 步的时候环境变了（比如杯子被碰了一下），模型还是傻乎乎地执行原来那 8 步。这就是 open-loop 的问题。

### RTC 的 trick：异步执行

RTC 的解法很聪明：**你别等当前 chunk 执行完才开始推理下一个 chunk，提前就开始推理**。这样推理的 100ms 就"藏"在当前 chunk 的执行时间里了。

听起来很完美，但有个 catch。假设推理花了 $d$ 个 timestep（比如 $d=5$）。当你推理完的时候，当前 chunk 已经执行了 5 步。你新预测的 chunk 的前 5 步必须和当前 chunk 的那 5 步**完全对上**，否则机器人会"抽搐"——前一个 action 说往左，下一个 action 突然说往右。

这前 5 步就叫 **action prefix**。问题变成：怎么让模型生成一个 chunk，它的开头和给定的 prefix 无缝衔接？

### RTC 的解法：Inference-time Inpainting

RTC 用了一个叫 pseudoinverse guidance 的技术。简单说就是：模型正常生成 chunk，但在每个 denoising step，用 backprop 算一下"当前生成的 chunk 的前 5 步和给定的 prefix 差多少"，然后用这个梯度把生成方向往 prefix 那边"拉"。

这能 work，但有两个问题：

1. **每个 denoising step 都要 backprop 一次**。5 步 denoising 就是 5 次 backprop。这些 backprop 是纯额外开销，直接增加 latency。实测多了大概 27ms。

2. **当 prefix 很长时，这个 guidance 会失准**。因为 pseudoinverse guidance 本质是 Jacobian 线性化——在当前点附近做一阶 Taylor 展开。prefix 越长，约束越复杂，线性化假设越不成立。就像你用一个直线去近似一个曲线，曲线弯得越厉害，直线越不准。

---

## 这篇 paper 的 idea

核心 idea 就一句话：**把推理时干的事搬到训练时**。

具体来说：训练时，随机采样一个 delay $d$，把 ground truth chunk 的前 $d$ 个 action 当作 prefix"喂"给模型，让模型学会在前 $d$ 个 action 已知的情况下预测剩下的 action。这样推理时模型已经"习惯"了接 prefix，直接 forward pass 就行，不需要任何 backprop。

这就像训练一个运动员：你不是比赛时才告诉他"你会遇到落后 5 秒的情况"，而是训练时就模拟各种落后场景，让他提前学会应对。比赛时他自然就会了。

---

## 怎么实现的

这个实现特别优雅，只需要改 3 个地方，代码量大概几行。核心 hack 是 **per-token flow matching timestep**。

### 背景：Flow Matching 怎么工作

Flow matching 的训练过程是这样的。你有一个 ground truth action chunk $\mathbf{A}_t$（长度 H），你采样一个噪声 $\boldsymbol{\epsilon}$，然后做一个线性插值：

$$\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1 - \tau) \boldsymbol{\epsilon}$$

- $\tau \in [0, 1]$：flow matching timestep，控制"干净程度"
- $\tau = 0$：纯噪声
- $\tau = 1$：纯 ground truth
- $\mathbf{A}_t^\tau$：混合后的 noisy action

模型 $\mathbf{v}_\theta$ 的任务是预测 velocity，即"从当前混合点走向 ground truth 的方向"：

$$\mathcal{L} = \mathbb{E} \| \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) - (\boldsymbol{\epsilon} - \mathbf{A}_t) \|^2$$

- $\mathbf{v}_\theta$：神经网络
- $\boldsymbol{\epsilon} - \mathbf{A}_t$：target velocity（从 noise 指向 data 的方向）
- $\mathbf{o}_t$：observation，作为 condition

推理时从 $\tau = 0$（纯噪声）出发，一步步积分到 $\tau = 1$（data），就像顺着水流漂到目的地。

### 核心 Hack：每个 token 有自己的 $\tau$

标准做法里，整个 chunk 的所有 action token 共享一个 $\tau$。这篇 paper 的 hack 是：**让每个 action token 有自己的 $\tau$**。

具体来说，对于 prefix 的那 $d$ 个 token：
- 把它们的 $\tau$ **固定设为 1**
- 把它们的输入**固定设为 ground truth action**（不加噪声）

对于 postfix 的那 $H - d$ 个 token：
- $\tau$ 随机采样（和原来一样）
- 输入是 $\tau \mathbf{A} + (1-\tau)\boldsymbol{\epsilon}$（和原来一样）

然后 loss 只在 postfix 上计算。

### 为什么这样能 work

$\tau = 1$ 意味着 "已经是 fully denoised 的 final data"。当模型通过 adaLN-zero 看到 prefix token 的 $\tau = 1$，它知道这些 token 已经"定型"了，不需要再去 denoise 它们。同时，transformer 的 self-attention 让 postfix token 能"看到" prefix token 的内容，从而学会"在这个 prefix 的基础上往下续写"。

这就像写作文续写：给你前半段（prefix），你写后半段（postfix）。训练时给你各种前半段，让你学会续写。考试时给你真实的前半段，你自然就会续写了。

### 代码就几行

看 Algorithm 1 的核心：

```python
# 采样 delay
delay = jax.random.randint(delay_rng, (b,), 0, max_delay)

# prefix mask：前 delay 个 token 为 True
prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]

# prefix 的 τ = 1，postfix 的 τ 随机
time = jnp.where(prefix_mask, 1.0, time[:, None])

# 构造 noisy input：prefix 用 ground truth，postfix 用 noisy
x_t = time[:, :, None] * action_chunk + (1 - time[:, :, None]) * noise

# 跑模型
pred_v_t = model(observation, x_t, time)

# loss 只在 postfix 上算
postfix_mask = jnp.logical_not(prefix_mask)[:, :, None]
loss = jnp.sum(loss * postfix_mask) / (jnp.sum(postfix_mask) + 1e-8)
```

就这么简单。不改 architecture（参数量不变），不改 runtime，训练代码加几行，推理代码也加几行（把 prefix 强制塞进去就行）。

### 推理时怎么做

推理时，prefix 来自上一个 chunk 已经执行的那些 action。每个 denoising step：

1. 把 prefix token 的输入设为给定的 prefix action
2. 把 prefix token 的 $\tau$ 设为 1
3. 把 postfix token 的输入按正常 flow matching 更新
4. forward pass 一次

没有任何 backprop。就是普通的 flow matching 采样，只不过有些 token 被"钉住"了。

---

## 架构图怎么看

看 Fig 2，这是 $\pi_{0.6}$ 的 action expert（一个 DiT-style transformer）：

- **左边**：observation tokens（来自 VLM backbone 的 visual + language features）
- **中间**：action tokens，共 H 个。前 $d$ 个是 prefix（红色，$\tau = 1$，输入是 ground truth），后 $H - d$ 个是 postfix（其他颜色，$\tau$ 随机，输入是 noisy）
- **右边**：输出，只在 postfix 上算 loss

Transformer 的 self-attention 让所有 token 互相看到。prefix token 带着 $\tau = 1$ 和真实 action 值，postfix token 带着 noisy action 和随机 $\tau$。模型学会"看到 prefix 后续写 postfix"。

### adaLN-zero 的细节

DiT 用 adaLN-zero 来注入 $\tau$。公式是：

$$h' = (1 + \text{scale}(\tau)) \odot h + \text{shift}(\tau)$$
$$\text{output} = \text{gate}(\tau) \odot \text{MLP}(h')$$

- $h$：hidden state
- $\text{scale}, \text{shift}, \text{gate}$：都是 $\tau$ 的函数（通过一个小 MLP 从 $\tau$ 算出来）
- $\odot$：element-wise multiplication

标准 DiT 里，所有 token 共享一个 $\tau$，所以 scale/shift/gate 是全局的。
修改后，每个 token 有自己的 $\tau_i$，所以 scale/shift/gate 是 per-token 的。

**关键**：这不需要增加参数！因为 scale/shift/gate 是 $\tau$ 的函数，函数本身不变，只是输入 $\tau$ 从一个标量变成一个向量。小 MLP 的参数完全一样。

---

## 实验结果

### 模拟实验：Kinetix

Kinetix 是一个 2D 物理仿真环境，任务是把各种形状推到目标位置。

设置：
- $H = 8$（预测 8 步）
- 4-layer MLP-Mixer 架构
- 32 epochs
- 评估 2048 rollouts per data point

结果看 Fig 3：

- $d = 0, 1$：training-time RTC 略差（marginally worse）
- $d = 2$：training-time RTC 开始赢
- $d = 3, 4$：training-time RTC 显著赢，gap 越来越大

**为什么 $d = 0, 1$ 时略差**？因为 training-time RTC 的 loss 只在 postfix 上算。当 $d = 1$ 时，第一个 action 永远是 prefix，永远不参与 loss，所以第一个 action 的训练 supervision 比标准方法少一点。这是一个 trade-off：用"少训练部分 action"换"推理时无 backprop"。

**为什么 $d$ 大时 training-time 显著赢**？因为 inference-time RTC 的 pseudoinverse guidance 在 $d$ 大时失效。prefix 越长，Jacobian 线性化越不准，guidance 方向越歪。Training-time 直接学习条件分布，没有线性化假设，所以更 robust。

### 真实世界实验

两个任务：
1. **Box building**：折纸盒，需要精确的 bimanual manipulation
2. **Espresso making**：磨豆、压粉、萃取、倒咖啡，一整套流程

用 $\pi_{0.6}$ base model，fine-tune 8000 steps，batch size 512。

推理在远程 H100 上，5 步 denoising：
- Training-time RTC：108ms end-to-end latency（$d \approx 5$）
- Inference-time RTC：135ms end-to-end latency（$d \approx 7$）

**省了 27ms，大概 20%**。这 27ms 就是 5 次 backprop 的开销。

结果看 Fig 5：training-time RTC 和 inference-time RTC 在 success rate 和 duration 上都持平。两者都比 synchronous baseline 快（synchronous 有明显的 chunk 间停顿）。

---

## 和其他方法的对比

### SmolVLA

SmolVLA 也有异步执行，但不解决 chunk 之间的 discontinuity。结果就是 chunk 之间会有 "jerks"——机器人突然抽一下。Training-time RTC 通过 prefix conditioning 显式解决这个。

### A2C2

A2C2 加了一个轻量级的 correction head 来修正 discontinuity。这是"打补丁"的思路。Training-time RTC 是"从源头解决"的思路——训练时就让模型学会连续。

### VLASH

VLASH 只 condition on **一个** future action。Training-time RTC condition on **一整段** prefix。作者强调这个区别：full prefix 提供更强的连续性约束。直觉上，知道未来 5 步比知道未来 1 步更能保证平滑。

### Classifier-Free Guidance 的类比

这个类比我觉得最能 build intuition：

- **Classifier guidance**（类似 inference-time RTC）：训练一个 unconditional model，推理时用额外 classifier 的梯度来 guide。需要 backprop，容易在 OOD 区域失效。
- **Classifier-free guidance**（类似 training-time RTC）：训练时随机 drop condition，直接学习 conditional + unconditional 两种模式。推理时无需额外网络，直接 forward。

Classifier-free guidance 最终取代了 classifier guidance 成为主流。Training-time RTC 可能也会。

参考：Classifier-Free Guidance (Ho & Salimans): https://arxiv.org/abs/2207.12598

### ControlNet 的类比

ControlNet 用一个 trainable copy 来注入 condition（如 edge map, depth map）。Training-time RTC 用 per-token $\tau$ 来注入 prefix condition。

区别：Training-time RTC 不需要额外的 network copy，参数量完全不变。更优雅。

参考：ControlNet: https://arxiv.org/abs/2302.05543

---

## 限制

### 灵活性

Training-time RTC 只支持 "hard" prefix——前 $d$ 个 action 必须完全匹配。Inference-time RTC 还能做 "soft" masking——对 prefix 之外的 overlapping actions 用 exponentially decreasing weights 来做更柔和的过渡。

这个 soft masking 在 training-time 不好实现（因为你没法在训练时模拟"soft" condition）。这是 training-time 方案的根本限制。

### Delay 分布

训练时要选一个 delay 分布（比如 Unif[0, 10]）。如果真实推理延迟超出这个范围（比如突然网络抖动，delay = 15），模型可能 OOD。

这就像你在训练时只见过 100ms 以内的延迟，突然遇到 300ms 延迟，模型不知道怎么办。

### 部分训练 supervision

如前所述，当 $d > 0$ 时，prefix 的那些 action 不参与 loss。这意味着模型对 prefix 部分的"生成能力"略弱。在 $d = 0, 1$ 时这个影响可见（虽然很小）。

---

## 我的延伸思考

### "编译" inference 复杂性到 training

这篇 paper 体现了一个很通用的设计原则：**如果推理时有某种固定模式的复杂性，考虑把它"编译"到训练时**。

这个原则的实例：
- Scheduled sampling：把 inference 时的 exposure bias 编译到训练时
- Professor forcing：把 inference 时的 sequence-level dynamics 编译到训练时
- Classifier-free guidance：把 inference 时的 guidance 编译到训练时
- 这篇 paper：把 inference 时的 inpainting 编译到训练时

参考：
- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- Professor Forcing: https://arxiv.org/abs/1610.09038

### Per-token timestep 的通用性

Per-token flow matching timestep 是一个被低估的 idea。它不只能用于 prefix conditioning，还能用于：

- **Inpainting 任意子序列**：不只是 prefix，任何位置的 action 都可以"钉住"
- **Multi-resolution generation**：粗粒度 token 先 denoise（$\tau$ 先到 1），细粒度 token 后 denoise
- **Hierarchical planning**：high-level action token 先确定，low-level token 后填充

这可能是一个值得深入探索的方向。

### 训练-推理一致性原则

这篇 paper 是 "训练时模拟推理条件" 原则的又一个实例。这个原则的核心：**训练分布和推理分布越一致，模型越 robust**。

在 RTC 的场景下，推理时模型会看到"prefix 已确定"的情况。如果训练时不模拟这个，模型在推理时就处于 OOD 状态。Training-time RTC 通过训练时模拟，消除了这个 OOD。

### 为什么这个 idea 现在才出现

这个 idea 看起来很 obvious（训练时模拟推理条件），但有几个因素让它现在才被 explicit 地提出：

1. **RTC 本身就是最近的工作**（2025 年 6 月）。Inference-time inpainting 的开销问题只有在 RTC 被实际部署后才会暴露。
2. **Per-token adaLN-zero 的灵活性**只有在 DiT-style 架构（如 $\pi_0$ 系列）普及后才容易实现。如果用 U-Net style diffusion，per-token timestep 不好做。
3. **VLA 的实时控制需求**最近才变得迫切（模型大到 100ms+ latency）。

### 公式符号的一个细节

Paper 正文公式 (2) 的 target 是 $(\boldsymbol{\epsilon} - \mathbf{A}_t)$，但 Algorithm 1 代码里是 `(action_chunk - noise)`，即 $(\mathbf{A}_t - \boldsymbol{\epsilon})$。

这看起来矛盾，其实是 **积分方向**的问题：
- 如果 $\tau: 0 \to 1$（noise → data），velocity target 是 $\mathbf{A}_t - \boldsymbol{\epsilon}$
- 如果 $\tau: 1 \to 0$（data → noise），velocity target 是 $\boldsymbol{\epsilon} - \mathbf{A}_t$

Algorithm 1 的 sample 函数里 `time` 从 0 递增到 1，所以是 $\tau: 0 \to 1$ 方向，target 应该是 $\mathbf{A}_t - \boldsymbol{\epsilon}$。代码是对的。正文公式 (2) 的符号可能有笔误，或者用了不同约定。

### $\pi$ 系列的演进脉络

- $\pi_0$：基础 VLA，flow matching 架构
- $\pi_{0.5}$：open-world generalization
- $\pi_{0.6}$：当前版本
- $\pi_{0.6}^*$：RL fine-tuning，learns from experience
- 这篇 paper：training-time RTC，实时控制增强

可以看出 PI 团队在实时控制上的持续迭代。每一步都在解决上一步暴露的问题：基础模型 → latency 问题（RTC）→ RTC 的开销问题（training-time RTC）。

### 工程价值

这篇 paper 的工程价值很高：

1. **Drop-in replacement**：不改 architecture，不改 runtime，几行代码
2. **20% latency reduction**：135ms → 108ms
3. **Single checkpoint for all delays**：一个模型支持 0-10 的所有 delay
4. **Fine-tune 即可**：不需要从头训练，从 base model fine-tune 8000 steps 就行

这种 "simple, effective, drop-in" 的风格是 PI 团队的特点。$\pi_0$ paper 也是这种风格——没有花哨的 architecture innovation，就是把 flow matching + VLM 组合好，execution 干净利落。

### 可能的 failure mode

1. **Delay 超出训练范围**：训练 delay ≤ 10，真实 delay = 15，OOD
2. **Prefix 误差累积**：如果上一个 chunk 的 action 本身有误差，prefix 就是"错"的，postfix 会继承误差
3. **Non-stationary delay**：delay 在 episode 内剧烈变化时可能不稳定

### 与 RL 的潜在 connection

这篇是 imitation learning 框架。但 "conditioning on committed actions" 在 RL 中有对应：Q-learning 的 target $Q(s', a')$ conditioning on $a'$。如果把这个 idea 扩展到 RL fine-tuning（如 $\pi_{0.6}^*$），可能有有趣的交叉。

---

## 总结

这篇 paper 的故事用一句话讲：**推理时的 inpainting 是一种"打补丁"，训练时的 conditioning 是"从源头解决"**。后者更简单、更便宜、更 robust，代价是少一点灵活性（不能 soft masking）和需要选 delay 分布。

核心 hack 是 per-token flow matching timestep：prefix token 的 $\tau = 1$（fully denoised），postfix token 的 $\tau$ 随机。Transformer 的 self-attention 自然地让 postfix "看到" prefix，学会续写。

结果：20% latency reduction，performance 持平，几行代码，drop-in replacement。

这种 "把推理复杂性编译到训练" 的思路非常通用，值得收藏到 design pattern toolbox 里。

---

**参考链接**：
- 本 paper: Physical Intelligence, 2025
- RTC: https://arxiv.org/abs/2506.07339
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_{0.6}^*$: https://arxiv.org/abs/2511.14759
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Policy: https://arxiv.org/abs/2304.13705
- DiT: https://arxiv.org/abs/2212.09748
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- ControlNet: https://arxiv.org/abs/2302.05543
- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- Professor Forcing: https://arxiv.org/abs/1610.09038
- SmolVLA: https://arxiv.org/abs/2506.01844
- VLASH: https://arxiv.org/abs/2512.01031
- A2C2: https://arxiv.org/abs/2509.23224
- Kinetix: https://arxiv.org/abs/2410.23208

---

# Training-Time Action Conditioning for Efficient Real-Time Chunking 详解

Karpathy 你好，这篇 paper 来自 Physical Intelligence 的 Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine，是 Real-Time Chunking (RTC) 系列的延续工作。核心 contribution 极其简洁：**把 inference-time inpainting 的 action prefix conditioning 搬到 training time，通过模拟 inference delay 让模型直接学会 conditioning on prefix，从而消除任何 inference-time computational overhead**。下面我从直觉、方法、实验三个层面尽量展开。

---

## 1. 背景：VLA 的实时性困境

### 1.1 问题本质

VLA 模型（如 π₀, π₀.₅, π₀.₆, OpenVLA, RT-2, GR00T, Gemini Robotics）参数量已经到 billions 级别，但机器人控制需要 50Hz 甚至更高的频率。一个 H100 跑 5 步 denoising 的 end-to-end latency 大约 100-135ms，对应 50Hz 控制器下 5-7 个 timestep 的 delay。

这就形成了一个根本张力：**模型越大越聪明，但越难实时**。

### 1.2 Action Chunking 的两难

Action Chunking（ACT, Diffusion Policy, ALOHA 系列的 de facto 标准）的核心 idea 是：模型一次性预测 H 个 future actions，然后执行 s ≤ H 步。好处是 trajectory 平滑（一次预测一段），坏处是 s 步内模型是 "open-loop" 的，无法反应环境变化。

如果 s = 1（完全 closed-loop），trajectory 会因为 inference latency 而出现 "jerks"（chunk 之间的不连续）。
如果 s = H（完全 open-loop），latency 消除了，但失去 reactivity。

### 1.3 Real-Time Chunking (RTC) 的解法

RTC [5] 的核心 idea：**异步执行**。当前 chunk 还在执行时，下一个 chunk 已经在并行推理。这样 inference latency 被 "藏" 在了 execution 里面。

但有一个问题：当下一个 chunk 推理完成时，已经过去了 d 个 timestep（d = inference delay in controller timesteps）。这 d 个 timestep 对应的 action 已经从上一个 chunk 执行了，称为 **action prefix**。新 chunk 必须和这个 prefix 连续，否则出现 "jerk"。

RTC 的解法：用 inference-time inpainting（基于 pseudoinverse guidance [18, 21]）把 prefix "inpaint" 进新 chunk 的生成过程。这需要在每个 denoising step 计算 vector-Jacobian product (VJP)，也就是 backprop。

**这就是这篇 paper 要解决的问题**：inpainting 的 backprop 带来了额外 latency，部分抵消了 RTC 的初衷。

参考：
- RTC paper: https://arxiv.org/abs/2506.07339
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- π₀.₆ model card: Physical Intelligence 官方
- Diffusion Policy: https://arxiv.org/abs/2304.13705
- ACT/ALOHA: https://arxiv.org/abs/2304.13705

---

## 2. 问题形式化

### 2.1 符号定义

- $\mathbf{o}_t$: observation at controller timestep $t$
- $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}]$: action chunk of length $H$ (prediction horizon)
- $s$: execution horizon (每 chunk 执行 $s$ 步, $s \leq H$)
- $d$: inference delay (in controller timesteps)

### 2.2 Action Prefix 的几何关系

参见图 1。如果在 timestep $t$ 开始推理，结果在 $t + d$ 才可用。那么 $[a_t, a_{t+1}, \ldots, a_{t+d-1}]$ 这 $d$ 个 action 必须从**上一个 chunk** 取（已经 committed），称为 action prefix。

有效性约束：$d \leq H - s$。这是因为上一个 chunk 在 $t - s$ 时刻开始，覆盖到 $t - s + H - 1$，所以从 $t$ 到 $t + d - 1$ 必须落在上一个 chunk 的覆盖范围内，即 $t + d - 1 \leq t - s + H - 1$，化简得 $d \leq H - s$。

### 2.3 Flow Matching 基础

策略 $p(\mathbf{A}_t | \mathbf{o}_t)$ 用 conditional flow matching [13] 训练：

$$\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1 - \tau) \boldsymbol{\epsilon}$$

- $\tau \in [0, 1]$: flow matching timestep（插值参数）
- $\mathbf{A}_t$: ground truth action chunk
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise
- 当 $\tau = 0$: $\mathbf{A}_t^\tau = \boldsymbol{\epsilon}$（纯噪声）
- 当 $\tau = 1$: $\mathbf{A}_t^\tau = \mathbf{A}_t$（纯数据）

Loss:

$$\mathcal{L}(\theta) = \mathbb{E} \, \| \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) - (\boldsymbol{\epsilon} - \mathbf{A}_t) \|^2$$

- $\mathbf{v}_\theta$: 神经网络（velocity field predictor）
- $(\boldsymbol{\epsilon} - \mathbf{A}_t)$: target velocity（注意方向！从 noise $\to$ data，所以是 $\boldsymbol{\epsilon} - \mathbf{A}_t$，对应 $\tau$ 从 0 增到 1）
- 推理时：从 $\tau = 0$（纯噪声）积分到 $\tau = 1$（数据），用 Euler 法：$\mathbf{A}_t^{\tau + \Delta\tau} = \mathbf{A}_t^\tau + \Delta\tau \cdot \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)$

参考：
- Flow Matching 原始 paper: https://arxiv.org/abs/2210.02747
- π₀ 用 flow matching 而不是 DDPM/DDIM，原因在 π₀ paper 里有讨论

---

## 3. Inference-Time RTC 的 inpainting 机制（对比基准）

### 3.1 Pseudoinverse Guidance

RTC [5] 用的是 [18, 21] 的 pseudoinverse guidance 方法。核心 idea：把 prefix action 当作 "观测约束"，用 guidance 把生成轨迹拉向满足约束的方向。

形式上，设我们要约束 $\mathbf{A}_t$ 的前 $d$ 个 action 等于给定的 prefix $\mathbf{A}_{prefix}$。在 flow matching 的每一步，计算：

$$\mathbf{v}_{guided} = \mathbf{v}_\theta + \lambda \cdot \nabla_{\mathbf{A}_t^\tau} \mathcal{R}(\mathbf{A}_t^\tau)$$

其中 $\mathcal{R}$ 是约束 residual（比如 $\|\mathbf{A}_t^\tau[:d] - \mathbf{A}_{prefix}\|^2$），$\nabla$ 通过 vector-Jacobian product 计算。

### 3.2 Soft Masking

RTC 还做了一个增强：除了 hard prefix（red in Fig 1），还对剩余的 overlapping actions（yellow in Fig 1）做 "soft masking"，用 exponentially decreasing weights。这需要 pseudoinverse guidance 的灵活性（weight 可以任意）。

### 3.3 问题

1. **计算开销**：每个 denoising step 都要 backprop 一次 VJP。如果 5 步 denoising，就是 5 次 backprop。
2. **线性化假设失效**：pseudoinverse guidance 依赖 Jacobian linearization。当 prefix 很长（d 大）时，linearization 假设破坏，guidance 方向不准。这篇 paper 的实验验证了这一点。
3. **延迟**：实测 inference-time RTC 的 end-to-end latency 是 135ms（d ≈ 7），比 training-time RTC 的 108ms（d ≈ 5）多了 27ms。

参考：
- Pseudoinverse-guided diffusion: https://arxiv.org/abs/2201.12473 (ICLR 2023)
- Training-free linear image inverses via flows: https://arxiv.org/abs/2310.04432

---

## 4. Training-Time RTC 的核心方法

### 4.1 核心 Insight

**既然 inference delay 是已知的（训练时可以模拟），为什么不直接训练模型学习条件分布 $p(\mathbf{A}_{t+d:H} | \mathbf{o}_t, \mathbf{A}_{t:d})$？**

这样推理时只需要一次 forward pass（标准 flow matching 采样），没有任何 backprop。

### 4.2 形式化

学习目标变为：

$$p(\mathbf{A}_{t+d:H} | \mathbf{o}_t, \mathbf{A}_{t:t+d})$$

- $\mathbf{A}_{t:t+d}$: action prefix（前 $d$ 个 ground truth action，来自同一 chunk）
- $\mathbf{A}_{t+d:H}$: action postfix（剩余 $H - d$ 个 action）

### 4.3 三个最小修改

实现上只需要 3 个改动（见 Algorithm 1 和 Fig 2）：

#### 修改 1: Per-token flow matching timestep

在标准 DiT [16] 架构中，flow matching timestep $\tau$ 通过 adaLN-zero 全局 conditioning。这里改为 **每个 action token 有自己的 $\tau$**。

对于 DiT 的 adaLN-zero：
$$\text{adaLN}(h) = (1 + \text{scale}(\tau)) \cdot h + \text{shift}(\tau), \quad \text{output} = \text{gate}(\tau) \cdot \text{MLP}(\text{adaLN}(h))$$

标准做法：scale, shift, gate 是 $\tau$ 的函数，所有 token 共享。
修改后：scale, shift, gate 是 $\tau_i$ 的函数，token $i$ 有自己的 $\tau_i$。**参数量不变**，只是 conditioning 变成 per-token。

#### 修改 2: Prefix 用 ground truth, $\tau = 1$

```python
prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]  # shape: (b, ah)
time = jnp.where(prefix_mask, 1.0, time[:, None])  # prefix 的 τ = 1
x_t = time[:, :, None] * action_chunk + (1 - time[:, :, None]) * noise
```

- `prefix_mask`: 前 `delay` 个 token 为 True
- `time`: prefix tokens 设为 1.0, postfix tokens 保持随机采样的 $\tau$
- `x_t`: prefix tokens = ground truth action (因为 $\tau = 1 \Rightarrow x_t = A_t$), postfix tokens = $\tau A_t + (1-\tau)\epsilon$

**直觉**：$\tau = 1$ 意味着这些 token 已经是 "fully denoised" 的 final data。模型通过 adaLN-zero 看到 $\tau = 1$，就知道这些 prefix 是确定的，只需要专注于 denoise postfix。

#### 修改 3: Loss 只在 postfix 计算

```python
postfix_mask = jnp.logical_not(prefix_mask)[:, :, None]
loss = jnp.sum(loss * postfix_mask) / (jnp.sum(postfix_mask) + 1e-8)
```

prefix tokens 不参与 loss，因为它们是 "given" 的，不是模型需要预测的。

### 4.4 推理时的采样

```python
def sample_actions(rng, model, observation, action_prefix, delay, num_steps):
    x_t = jax.random.normal(rng, (b, ah, ad))  # 初始噪声
    time = 0.0
    for _ in range(num_steps):
        x_t = jnp.where(prefix_mask[:, :, None], action_prefix, x_t)  # 强制 prefix
        time_masked = jnp.where(prefix_mask, 1.0, time)  # prefix τ=1
        v_t = model(observation, x_t, time_masked)
        x_t = x_t + dt * v_t  # Euler 积分
        time = time + dt
    return x_t
```

关键：每步都把 prefix 强制设为 ground truth（来自上一个 chunk 的 committed action），$\tau$ 设为 1。模型只更新 postfix。这就是标准 flow matching 采样，没有任何 backprop。

### 4.5 训练时的 delay 采样

由于真实世界 inference delay 不固定（网络波动、GPU 负载等），训练时随机采样 $d$：

- 模拟实验：从 $\{0, 1, 2, 3, 4\}$ 采样，exponentially decreasing weights（高 delay 少见，所以少训练）
- 真实实验：从 $\text{Unif}[0, 10]$ 采样，支持最大 200ms latency（50Hz robot）

**这是一个 single checkpoint 支持所有 delay 的设计**，不需要为每个 delay 训练单独的模型。

参考：
- DiT (Scalable Diffusion Models with Transformers): https://arxiv.org/abs/2212.09748
- adaLN-zero 在 DiT paper 里有详细描述

---

## 5. 架构图解析（Fig 2）

Fig 2 展示了 conditioning 架构在 $\pi_{0.6}$ action expert 上的应用。$\pi_0$ 系列的 action expert 是一个 flow matching 的 transformer，输入包括：

1. **Observation tokens**: 来自 VLM backbone 的 visual + language features
2. **Action tokens**: $H$ 个 action token，每个对应一个 future timestep
3. **Flow matching timestep $\tau$**: 通过 adaLN-zero 注入

修改后的架构：
- Prefix action tokens（前 $d$ 个）：输入是 ground truth action（non-noisy），$\tau = 1$
- Postfix action tokens（后 $H - d$ 个）：输入是 noisy action $\tau A + (1-\tau)\epsilon$，$\tau$ 随机

Transformer 的 self-attention 让 postfix tokens "看到" prefix tokens，从而学会 conditioning on prefix。这是 key：**conditioning 不是通过额外的 cross-attention 或 concatenation，而是通过共享 self-attention + per-token $\tau$ 实现的**。

---

## 6. 实验

### 6.1 模拟实验：Kinetix Benchmark

#### 设置
- Benchmark: dynamic Kinetix [15]
- Architecture: 4-layer MLP-Mixer [25]
- Prediction horizon: $H = 8$
- Training: 32 epochs
- Training-time RTC: 从第 24 epoch resume，fine-tune 8 epochs with action prefix conditioning（保证 total compute 一致）
- Delay 采样: $\{0, 1, 2, 3, 4\}$, exponentially decreasing weights
- 评估: 2048 rollouts per data point, 95% Wilson score intervals
- Execution horizon: $s = \max(d, 1)$

#### 结果（Fig 3）

| Inference Delay $d$ | Naive Async | Inference-Time RTC | Training-Time RTC |
|---|---|---|---|
| 0 | - | baseline | marginally worse |
| 1 | - | baseline | marginally worse |
| 2 | - | baseline | **better** |
| 3 | - | baseline | **significantly better** |
| 4 | - | baseline | **much better** |

**关键发现**：delay ≥ 2 时，training-time RTC 开始超越 inference-time RTC，gap 随 delay 增大而扩大。

#### 为什么 delay 大时 training-time 更好？

作者的解释：inference-time RTC 依赖 Jacobian linearization。当 prefix 长（d 大）时，linearization 假设破坏，guidance 方向不准。Training-time 直接学习条件分布，没有 linearization 假设。

**我的延伸 intuition**：这类似于 classifier-free guidance vs classifier guidance 的对比。Classifier guidance（类似 inference-time inpainting）需要单独的 classifier，依赖梯度，容易在 OOD 区域失效。Classifier-free guidance（类似 training-time conditioning）直接训练条件生成模型，更 robust。

#### 为什么 delay = 0, 1 时 training-time 略差？

因为 training-time RTC 的 loss 只在 postfix 上计算。当 delay = 0 时，所有 action 都是 postfix，正常训练。当 delay = 1 时，第一个 action 永远是 prefix（不参与 loss），所以第一个 action 的训练 supervision 略少。这导致 delay = 0, 1 时 marginally worse。

**这是一个有趣的 trade-off**：training-time RTC 用 "部分 action 不训练" 换取 "inference 时无 backprop"。

### 6.2 真实世界实验：Box Building & Espresso Making

#### 设置
- Base model: $\pi_{0.6}$ [24]
- Tasks: box building, espresso making（来自 $\pi_{0.6}^*$ [1]）
- Fine-tuning: 8000 gradient steps, batch size 512
- Delay 采样: $\text{Unif}[0, 10]$
- Inference: remote H100, 5 denoising steps
- Latency: training-time RTC = 108ms (d ≈ 5), inference-time RTC = 135ms (d ≈ 7)
- Robot: 50Hz

#### 结果（Fig 5）

| Method | Espresso Success | Espresso Duration | Box Success | Box Duration |
|---|---|---|---|---|
| Synchronous | lower | longer (pauses) | lower | longer |
| Inference-Time RTC | high | fast | high | fast |
| Training-Time RTC | high (parity) | fast (parity) | high (parity) | fast (parity) |

**关键发现**：training-time RTC 在 performance 和 speed 上都与 inference-time RTC 持平，但 computational cost 更低（108ms vs 135ms，节省 27ms = 20% latency）。

#### Latency 差异的来源

- Inference-time RTC: 每个 denoising step 需要 VJP (backprop)，5 步 = 5 次 backprop
- Training-time RTC: 纯 forward pass，5 步 = 5 次 forward

27ms 的差异主要来自这 5 次 backprop 的开销。

参考：
- $\pi_{0.6}^*$: https://arxiv.org/abs/2511.14759
- Kinetix: https://arxiv.org/abs/2410.23208
- MLP-Mixer: https://arxiv.org/abs/2105.01601

---

## 7. 与相关工作的对比

### 7.1 SmolVLA [20]

SmolVLA 也有异步执行，但**不解决 inter-chunk discontinuity 问题**，导致 chunk 之间有 OOD "jerks"。Training-time RTC 通过 conditioning on prefix 显式解决 discontinuity。

### 7.2 A2C2 [19] (Leave No Observation Behind)

A2C2 用一个 lightweight correction head 来修正 chunk 之间的 discontinuity。这是 "add-on" 方案，而 training-time RTC 是 "end-to-end" 方案。

### 7.3 VLASH [22]

VLASH conditioning on **a single future action**。Training-time RTC conditioning on **a full prefix of future actions**。作者强调这个区别：full prefix 提供更强的连续性约束。

### 7.4 Hierarchical VLA (Gemini Robotics, GR00T)

这些方法用 System 2 (heavyweight planner) + System 1 (lightweight action generator) 的分层设计。Training-time RTC 是 single-model 方案，orthogonal to hierarchical 设计，可以组合使用。

### 7.5 MiniVLA, SmolVLA (efficient architectures)

这些通过修改 architecture 来加速。Training-time RTC 不修改 architecture，是 training recipe 的改变。

参考：
- SmolVLA: https://arxiv.org/abs/2506.01844
- A2C2: https://arxiv.org/abs/2509.23224
- VLASH: https://arxiv.org/abs/2512.01031
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- MiniVLA: https://github.com/Stanford-ILIAD/openvla-mini
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818

---

## 8. 限制和未来方向

### 8.1 限制

1. **灵活性**：只支持 "hard" prefix conditioning，不能像 inference-time RTC 那样做 "soft" masking（对 overlapping actions 用 exponentially decreasing weights）。
2. **Delay 分布选择**：需要根据预期 inference latency 选择训练时的 delay 分布。如果真实 latency 超出训练范围，可能 OOD。
3. **训练 compute**：略微增加 training compute（需要 fine-tune with prefix conditioning）。

### 8.2 未来方向

作者提到 "best of both worlds"：结合 training-time 和 inference-time 的优点。可能的 direction：
- Training-time conditioning on hard prefix + inference-time soft masking on overlapping actions
- 自适应 delay 采样（根据真实 latency 动态调整训练分布）
- 多 delay checkpoint 的 mixture

---

## 9. 延伸思考

### 9.1 与 Classifier-Free Guidance 的类比

这个方法的本质是 **把 inference-time guidance 转化为 training-time conditioning**，类似于 classifier-free guidance 取代 classifier guidance 的思路：

- Classifier guidance: 训练一个 unconditional model + 一个 classifier，inference 时用 classifier 梯度 guidance
- Classifier-free guidance: 训练时随机 drop condition，inference 时直接用 conditional generation

同样：
- Inference-time RTC: 训练 unconditional policy，inference 时用 inpainting guidance
- Training-time RTC: 训练 conditional policy（condition on prefix），inference 时直接采样

Classifier-free guidance 最终成为主流，因为更简单、更 robust。Training-time RTC 可能也会。

参考：Classifier-Free Guidance (Ho & Salimans): https://arxiv.org/abs/2207.12598

### 9.2 与 ControlNet 的类比

另一种视角：prefix conditioning 类似于 ControlNet 的 conditional generation。ControlNet 用一个 trainable copy 来注入 spatial condition（如 edge, depth）。Training-time RTC 用 per-token $\tau$ 来注入 temporal condition (prefix action)。

区别：Training-time RTC 不需要额外的 network copy，参数量不变。这是更优雅的设计。

参考：ControlNet: https://arxiv.org/abs/2302.05543

### 9.3 Per-token timestep 的更广泛应用

Per-token flow matching timestep 是一个很通用的 idea。可以用于：
- **Inpainting 任意子序列**（不只是 prefix）
- **Multi-resolution generation**（粗粒度 token 先生成，细粒度 token 后生成）
- **Hierarchical planning**（high-level action 先确定，low-level action 后生成）

这可能是一个被低估的 contribution。

### 9.4 与 Reactive Diffusion 的关系

这篇 paper 的核心张力是 "reactivity vs smoothness"。Training-time RTC 的解法是 "learn to be reactive to prefix"。这让人联想到 reactive diffusion 的 line of work，但更具体：不是 reaction to environment，而是 reaction to **committed actions**。

### 9.5 训练-推理一致性

这篇 paper 体现了一个重要原则：**训练时模拟推理时的条件**。这和 scheduled sampling、professor forcing、student forcing 等 idea 一脉相承。在 diffusion/flow matching 框架下，这个 idea 的 instantiation 就是 "训练时模拟 inference delay"。

参考：
- Scheduled sampling: https://arxiv.org/abs/1506.03099
- Professor forcing: https://arxiv.org/abs/1610.09038

### 9.6 π 系列的演进

- π₀: flow matching VLA，基础架构
- π₀.₅: open-world generalization
- π₀.₆: 当前版本（model card 未完全公开）
- π₀.₆*: learns from experience（RL fine-tuning）
- 这篇 paper: training-time RTC 作为 π₀.₆ 的实时控制增强

可以看出 π 团队在实时控制上的迭代：从基础 flow matching → RTC → training-time RTC。每一步都在降低 latency 或提升 robustness。

### 9.7 工程价值

这篇 paper 的工程价值很高：
1. **Drop-in replacement**: 不改 architecture, 不改 runtime, 只改几行训练代码
2. **20% latency reduction**: 135ms → 108ms，对实时控制很显著
3. **Single checkpoint for all delays**: 不需要为不同 latency 训练不同模型

这种 "simple, effective, drop-in" 的风格是 Physical Intelligence 团队的一贯特点（π₀ paper 也是这种风格）。

### 9.8 可能的 failure mode

虽然 paper 没有详细讨论，但可以推测一些 failure mode：
1. **Delay 超出训练范围**：如果训练时 delay ≤ 10，但真实 delay = 15，模型可能 OOD
2. **Non-stationary delay**：如果 delay 在 episode 内剧烈变化，可能影响稳定性
3. **Prefix action 的累积误差**：如果上一个 chunk 的 action 有误差，prefix 本身就是 "wrong" 的，postfix 会继承这个误差。这是所有 chunking 方法的通病。

### 9.9 与 RL 的关系

这篇 paper 是 imitation learning 框架下的工作。但 "training-time conditioning on committed actions" 的 idea 在 RL 中也有对应：Q-learning 的 target $Q(s', a')$ conditioning on $a'$。可以想象，如果把这个 idea 扩展到 RL fine-tuning（如 $\pi_{0.6}^*$ 的 RL），可能有有趣的 connection。

### 9.10 公式中的细节

回看 Algorithm 1 的 loss 计算：

```python
loss = (pred_v_t - (action_chunk - noise)) ** 2
```

注意 target 是 `(action_chunk - noise)`，对应 $(\mathbf{A}_t - \boldsymbol{\epsilon})$。但 paper 正文公式 (2) 写的是 $(\boldsymbol{\epsilon} - \mathbf{A}_t)$。

这看起来矛盾，但实际上是因为 **积分方向**：
- 如果从 $\tau = 0 \to 1$（noise → data），target velocity 是 $\mathbf{A}_t - \boldsymbol{\epsilon}$
- 如果从 $\tau = 1 \to 0$（data → noise），target velocity 是 $\boldsymbol{\epsilon} - \mathbf{A}_t$

Algorithm 1 用的是 $\tau = 0 \to 1$ 方向（`time = 0.0` 初始，`time += dt` 递增），所以 target 是 `action_chunk - noise`。Paper 正文公式 (2) 的 $(\boldsymbol{\epsilon} - \mathbf{A}_t)$ 可能是笔误，或者用了不同的约定。读者需要注意这个细节。

实际上，看 Algorithm 1 的 sample 函数：
```python
x_t = x_t + dt * v_t  # 从 τ=0 开始，递增
```
这是 $\tau: 0 \to 1$ 方向，velocity 应该指向 data，即 $\mathbf{A}_t - \boldsymbol{\epsilon}$。所以 Algorithm 1 是 self-consistent 的，正文公式 (2) 的符号可能有误。

---

## 10. 总结

这篇 paper 的核心贡献：

1. **Idea**：把 inference-time inpainting 的 action prefix conditioning 转化为 training-time conditioning，通过模拟 inference delay 实现。
2. **方法**：3 个最小修改（per-token $\tau$, prefix 用 ground truth + $\tau=1$, loss 只在 postfix），不改 architecture, 不改 runtime。
3. **结果**：模拟实验中 delay ≥ 2 时优于 inference-time RTC；真实实验中保持 parity 同时减少 20% latency。
4. **意义**：作为 RTC 的 drop-in replacement，实用价值高。体现了 "训练-推理一致性" 的设计原则。

这篇 paper 的风格我很喜欢：**simple idea, clean execution, strong results**。没有花哨的 architecture innovation，只是把一个 inference-time 的复杂性 "compile" 到 training time。这是好的系统设计的典范。

---

**参考链接汇总**：
- 本 paper (Training-Time RTC): Physical Intelligence, 2025
- RTC (predecessor): https://arxiv.org/abs/2506.07339
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- π₀.₆*: https://arxiv.org/abs/2511.14759
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Policy: https://arxiv.org/abs/2304.13705
- ACT/ALOHA: https://arxiv.org/abs/2304.13705
- ALOHA Unleashed: https://arxiv.org/abs/2410.13126
- DiT: https://arxiv.org/abs/2212.09748
- Pseudoinverse Guidance: https://arxiv.org/abs/2201.12473
- Training-free flow inverses: https://arxiv.org/abs/2310.04432
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- ControlNet: https://arxiv.org/abs/2302.05543
- SmolVLA: https://arxiv.org/abs/2506.01844
- VLASH: https://arxiv.org/abs/2512.01031
- A2C2: https://arxiv.org/abs/2509.23224
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- MiniVLA: https://github.com/Stanford-ILIAD/openvla-mini
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- Kinetix: https://arxiv.org/abs/2410.23208
- MLP-Mixer: https://arxiv.org/abs/2105.01601
- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- Professor Forcing: https://arxiv.org/abs/1610.09038
