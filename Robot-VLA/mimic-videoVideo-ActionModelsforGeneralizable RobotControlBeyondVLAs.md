---
source_pdf: mimic-videoVideo-ActionModelsforGeneralizable RobotControlBeyondVLAs.pdf
paper_sha256: a7d2d3fa9367921bde900889e862025afa1e28e93437f287aa018b3b17e39658
processed_at: '2026-08-05T18:18:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 mimic-video

## 一句话总结

**与其让机器人从头学物理常识，不如直接用会"想象"的视频模型当大脑，机器人只负责照着想象出来的画面去执行。**

## 问题出在哪

现在的 VLA 模型 (比如 π0.5、OpenVLA) 走的是这条路:

> 先拿一个在互联网图片+文字上训练的 VLM 当脑子 → 再喂机器人动作数据微调

问题在哪? **图片是静态的**。一张猫的图片不会告诉你猫怎么跳、杯子怎么摔、布怎么折。VLM 看了几亿张图，学到的是"这是杯子"、"那是桌子"这种**语义知识**，完全不懂**物理因果**。

结果就是: 机器人要搞清楚"我推这个杯子会怎样"这种 basic 物理常识，只能靠昂贵的人工遥操作数据**从头学**。数据需求量大得离谱，根本 scale 不动。

## mimic-video 的核心 insight

互联网上**视频**数据太多了 — 人做饭、装配零件、折叠衣服。视频天然包含:
- 物体怎么动
- 怎么变形  
- 碰撞会怎样
- 抓取的顺序

如果你拿一个在互联网海量视频上预训练的视频生成模型 (NVIDIA 的 Cosmos-Predict2)，它已经**隐式学会了**这些物理常识。你让它"想象"一下任务怎么完成，它脑中已经有谱了。

那 action decoder 要做什么? 极其简单的事 — **看图说话**:
- Video model 想象: "先伸手 → 抓住杯子 → 提起来 → 放到左边"
- Action decoder 翻译: "对应关节角度是 [0.1, 0.2, -0.3, ...]"

这个翻译任务在机器人学习里叫做 **Inverse Dynamics Model (IDM)** — 给你起点和终点，求中间的动作。IDM 是出了名的**简单、单峰、非因果**，几十分钟数据就能学会。

## 为什么这个分解很聪明

机器人学习里有两个子问题:

| 子问题 | 难度 | 特性 |
|--------|------|------|
| Forward dynamics: 已知动作，预测未来会发生什么 | **难** | 多峰 (杯子可能往左倒也可能往右倒)、因果 |
| Inverse dynamics: 已知起点和未来，求动作 | **简单** | 单峰、非因果 |

传统 VLA 把两个问题捆在一起，全靠稀缺 robot data 学。

mimic-video 的策略:
- **难的部分** (forward dynamics + planning) → 丢给互联网视频预训练 (海量免费数据)
- **简单部分** (inverse dynamics) → 留给 action decoder (只需要少量 robot data)

这就是为什么 paper 里 Figure 2 那个 "oracle 实验" 那么震撼: 如果你把**真实未来视频**的内部特征直接喂给 action decoder，成功率接近 100%。意思就是 — **控制问题本质上就等于视觉预测问题**，剩下的翻译是 trivial 的。

## 架构长什么样

两个 flow matching 模型拼在一起:

```
[当前画面 + 语言指令]
        ↓
   视频大脑 (Cosmos-Predict2, 2B参数, 冻结)
        ↓
   想象出未来画面的"草稿" (latent表示)
        ↓
   小巧的 action decoder (DiT, 从头训练)
        ↓
   机器人动作序列
```

### 关键细节 1: 不需要真的生成视频

视频模型是 diffusion 模型，要一步步去噪才能生成完整画面。mimic-video 发现一个反直觉的事:

**不需要完全去噪。停在中间的 "模糊草稿" 阶段反而效果更好。**

极端情况下，**完全不去噪 (τ_v=1, 纯噪声输入)** 直接一次 forward pass 就够了 — 因为视频模型的**中间层特征**已经编码了"要做什么"的高层计划，最后那些层只是在补像素细节。

这就 win-win: 性能最好 + 速度最快 (不需要 ODE 积分)。

### 关键细节 2: 训练时让 decoder 见识各种噪声级别

训练时，video latent 的噪声水平 τ_v 和 action 的噪声水平 τ_a **独立采样**。所以 decoder 见过从纯噪声到清晰画面的所有情况。Inference 时用 τ_v=1 (纯噪声) 也在训练分布内，不会 OOD。

### 关键细节 3: 两阶段训练

1. **Stage 1**: 用 LoRA 微调视频大脑，让它熟悉机器人领域的视觉风格 (200小时 bimanual 视频)
2. **Stage 2**: **冻结**视频大脑，只训练小 action decoder (1-2 小时的 teleop 数据)

昂贵的人工遥操作数据**只用在轻量 decoder 上**。

## 实验结果有多炸

### 仿真 (LIBERO)
mimic-video **从零训练** → 93.9% 成功率，接近 OpenVLA-OFT **微调后**的 96.9%。π0.5-style VLA **从零训练**只有 85.9%。

### 真机双臂灵巧手 (最 striking)
mimic-video **只用单个工作区摄像头** → 93% 成功率
强 baseline (DiT-Block Policy) **用 5 个摄像头** (workspace + 4个腕部摄像头) → 74.1%

这意味着什么? 视频模型能**脑补**被遮挡的部分，因为它见过太多类似视频，知道"抓东西时手会挡住物体，但物体应该还在那"。

而且训练数据极少:
- Package sorting: 1 小时 33 分钟 (512 episodes)
- Tape stowing: 2 小时 14 分钟 (480 episodes)

这种数据量在传统 VLA 范式下想都别想。

### Sample Efficiency
**10 倍**数据效率提升。mimic-video 用 10% 的数据就能达到 VLA baseline 用 100% 数据的最好成绩。用 2% 数据 (每个任务 1 个 episode) 还能拿到 77% 成功率。

## 为什么停在中间噪声反而好

paper 给两个解释 (Appendix E):

**解释 1: 噪声当数据增强**
训练时 decoder 看的是 ground-truth 视频，inference 时视频模型生成的画面可能有点不真实或 subtly 偏差。完全去噪等于直接用可能有误差的生成结果，会 OOD。**留点噪声相当于 data augmentation**，让 decoder 更鲁棒，不依赖虚假的细节线索。

**解释 2: 视频模型中间层更有信息**
diffusion 模型在早期去噪阶段 (高噪声)，内部 hidden states 必须**编码完整计划** — "要从这走到那，需要做哪些 transformation"。
到了晚期去噪阶段 (低噪声)，输入已经接近目标，模型学到的只是"做点小修正让像素对齐"，hidden states 信息量反而下降。

所以 decoder 直接 cross-attention 到中间层特征就够了，不需要等视频完全生成。

## 我的几点直觉

### 1. 这篇 paper 戳中了 VLA 范式的真正痛点

VLA 的根本问题是 **modality mismatch**: VLM 学的是语义，robotics 需要的是物理。这不是靠更多 robot data 能解决的 — 是 pretraining 数据选择错了。

### 2. "让模型干它擅长的事" 是 ML 的永恒智慧

视频生成模型花了几个亿 GPU 小时在互联网视频上，已经会"想象物理过程"了。你非要让它学动作是浪费 — 让它继续干它擅长的 (想象)，把"翻译"这种简单活交给小模型。这就是分工。

### 3. "中间表征 > 最终输出" 这个 finding 很深刻

不只是这篇 paper。LLM 的 probing 研究早就发现中间层比最后输出层更 informative。这里又印证一次 — 生成模型的**过程表征**比**最终产物**更有价值，因为最终产物被 task-specific format 压缩了。

### 4. Inference-time τ_v tuning 是免费午餐

一个训练好的模型，调一个 inference 超参就能再涨 10 个点。这种 free lunch 在 robotics 里太少了。意味着同一个模型可以 per-task 调优而不用重训。

## 可能的发展方向

1. **V-JEPA 2 替代 Cosmos**: Meta 的非生成式自监督视频模型，不需要 pixel generation，可能更快
2. **Action-conditioned video model**: 让视频模型接受动作输入，可以做想象+搜索的 planning
3. **Multi-view video foundation model**: 解决当前单视角限制
4. **Cross-embodiment VAM**: 一个 frozen 视频大脑 + N 个轻量 decoder 对应 N 种机器人
5. **Joint training**: 不冻结视频大脑，让 representation 更适配 action decoding (但要小心 catastrophic forgetting)

## 最后

这篇 paper 我觉得是 robotics foundation model 方向的**范式转换**之作。它把问题从 "怎么让 VLM 懂物理" 重新定义为 "怎么让 video model 当 robot 大脑"。10 倍数据效率 + 单视角超越多视角这两个结果，足以说明 video prior 比 vision-language prior 对 control 任务更本质。

接下来 1-2 年，我会 watch:
- NVIDIA 是否推出 action-conditioned Cosmos 变体
- Meta 是否用 V-JEPA 2 做 robotics
- 是否有人 scale 到 cross-embodiment unified VAM
- 真机 inference 速度能否到 30Hz 以上

如果这些都发生，VAM 范式很可能取代 VLA 成为 robotics foundation model 的主流路线。

相关链接:
- 项目主页: https://mimic-video.github.io
- Cosmos: https://arxiv.org/abs/2501.03575  
- Cosmos 2: https://arxiv.org/abs/2511.00062
- π0.5: https://arxiv.org/abs/2504.16054
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Flow Matching: https://arxiv.org/abs/2210.02747

---

# mimic-video: Video-Action Models 深度解析

## 1. 核心论点与 Intuition

这篇 paper 的核心 claim 非常 sharp: **当前 VLA 范式存在根本性的 modality mismatch**。VLM backbone (如 PaliGemma, LLaMA-Vision) 在 internet-scale image-text pairs 上 pretrain，这些数据是 **static** 的，完全没有 temporal dynamics。结果就是 policy 要从稀缺昂贵的 robot demonstrations 中从头学习 physical physics (物体如何 move、deform、react to forces)。

mimic-video 的 thesis 是: **video 是更好的 pretraining modality**，因为 video inherently captures "how things are done" — 物体如何 move、deform、react to forces。视频模型已经 implicit 编码了 physical dynamics + visual action plans，所以 action decoder 只需做一件简单的事 — 把 visual plan translate 成 low-dimensional motor commands，这本质上是 **Inverse Dynamics Model (IDM)**。

这里有一个非常重要的 decomposition insight (paper Sec III):
- Long-horizon, multi-modal planning → offload 给 video backbone (这是 hard 部分)
- Low-level, unimodal, non-causal inverse dynamics → 给 action decoder (这是 easy 部分)

IDM 是 non-causal 是因为 inverse dynamics 是 a Markovian translation: 给当前 state + future visual plan, 输出 action chunk，不需要建模 future distribution。这正好对应 Lynch et al. [34] 和 Mees et al. [35] 的工作。

paper Sec III 的 oracle case study 是最 informative 的部分 (Fig. 2):
- 用 ground-truth future video latents conditioning → 接近 100% success rate
- 说明 control effectively reduces to visual prediction
- Policy performance scales directly with video model quality
- Action decoder 不需要 expensive teleoperation data — 它只是一个 lightweight translator

这个 finding 本身就是 VAM 范式可行的 existence proof。

## 2. Architecture 详解 (Fig. 3 解析)

### 整体架构

mimic-video 是 **dual flow matching architecture**，两个独立 flow schedule:

```
Video Model: v_φ(z_past^0, z_future^τv, l, τv) → p_φ(z_future^0 | z_past^0, l)
Action Policy: π_θ(A_t^τa, q_t, h^τv, τa, τv) → p_θ(A_t^0 | q_t, h^τv, τv)
```

其中 **h^τv = v_φ^(k)(z_past^0, z_future^τv, l, τv)** 是 video model 第 k-th layer 的 hidden states (paper 用 k=19，见 Appendix C)，作为 cross-attention 的 conditioning signal 给 action decoder。

### Video Backbone: Cosmos-Predict2

具体选择 NVIDIA 的 **Cosmos-Predict2** [38, 37]:
- 2B 参数 latent Diffusion Transformer (DiT) [41]
- 输入: 3D-tokenizer 编码的 video frame sequence
- Context: 5 frames clean latent patch embeddings (past)
- Target: "noisy" latent patches for future frames
- Each transformer layer 三部分:
  1. **Self-attention** over full video sequence (past + future)
  2. **Cross-attention** to T5 [44] language embeddings
  3. **Two-layer MLP**
- 与 π0.5 一样，使用 **partial denoising** — flow 只跑到中间 τ_v 就停止

Cosmos-Predict2 来自 NVIDIA Cosmos platform: https://arxiv.org/abs/2501.03575
新版 Cosmos 2: https://arxiv.org/abs/2511.00062

### Action Decoder: Lightweight DiT

架构设计有几个关键 details:
- **Input**: proprioceptive state q_t 和 action chunk A_t 经过 two separate MLPs 编码，然后 **concatenate** 形成 sequence dimension
- **Learned absolute positional encodings** 加入每个 token
- **Mask token** 随机替换 proprioceptive state encoding 防止 overfitting
- Each decoder layer 三部分:
  1. **Cross-attention** to h^τv (video model 的 intermediate representations)
  2. **Self-attention** over action sequence
  3. **Two-layer MLP**
- Residual paths + AdaLN [41] modulation
- AdaLN input 是 **low-rank bilinear-affine encoding** of (τ_v, τ_a) — 这很关键，让 decoder 知道当前 video 和 action 处于什么 noise level

这个 (τ_v, τ_a) 的 joint conditioning 是 elegant design — decoder 必须知道 video latent 处于哪个 denoising stage 才能正确 interpret。

## 3. Flow Matching 数学推导

Flow Matching [31] 是这篇 paper 的数学基础。让我详解：

### Conditional Optimal Transport Path

```
x^τ = (1-τ)x^0 + τε,  τ ∈ [0, 1]      ... Eq (1)
```

变量含义:
- **x^0**: clean data sample (来自真实 data distribution p_0)
- **ε ~ N(0, I)**: 标准 Gaussian noise
- **τ**: flow time, 0 = clean, 1 = noise
- **x^τ**: 在 clean data 和 noise 之间的 linear interpolation

这是 **optimal transport** 路径 — 直线插值，比 DDPM 的 curved path 更 efficient。

### Conditional vs Marginal Vector Field

```
u_τ(x^τ|x^0) := d/dτ x^τ = ε - x^0           (conditional vector field)
u_τ(x^τ) = E_{p(x^0|x^τ)} u_τ(x^τ|x^0)        (marginal, intractable)
```

- **u_τ(x^τ|x^0)**: 给定 endpoint x^0 的 conditional flow field — trivially computable
- **u_τ(x^τ)**: marginal flow field — 需要 marginalize over posterior p(x^0|x^τ)，**intractable**

### Flow Matching Loss

```
L_CFM = E_{T(τ), p_0(x^0), p_τ(x^τ|x^0)} ||v_θ(x^τ, τ) - u_τ(x^τ|x^0)||^2    ... Eq (2)
```

变量:
- **T(τ)**: flow time 的 sampling distribution (in [31] 是 Uniform[0,1])
- **v_θ**: neural network 估计的 vector field
- **u_τ(x^τ|x^0)**: supervision target — 注意是 **conditional** field，tractable

关键 trick: 用 tractable conditional field 作为 supervision signal，但学到的 v_θ 会逼近 intractable marginal field (因为 E[v_θ] = u_τ marginal via regression to conditional mean)。

### Inference: ODE Integration

```
x̂^0 = ε + ∫_1^0 v_θ(x̂^τ, τ) dτ                ... Eq (3)
```

从 τ=1 (pure noise) 积分到 τ=0 (clean sample)。这是 backward in flow time。

### Mimic-video 的关键 modification: Partial Denoising

paper 不让 flow 跑到 τ=0，而是停在中间 **τ_v > 0**:
```
z_future^τv = z_future^1 + ∫_1^τv v_φ(z_past^0, z_future^τv', l, τv') dτv'
```

然后 h^τv = v_φ^(k)(z_past^0, z_future^τv, l, τv) — 取第 k 层 hidden states，传给 action decoder。

**特殊 case: τ_v = 1** — 此时 z_future^1 = ε (pure noise)，video backbone 只需要 **single forward pass** — 没有任何 ODE integration！这是 real-time inference 的关键。

## 4. Action Sampling Algorithm (Algorithm 1)

```
Algorithm 1: Action Sampling(k, τ_v)
Input: z_past^0, q_t, l
1: z_future^1, A_t^1 ~ N(0, I)
2: z_future^τv ← z_future^1 + ∫_1^τv v_φ(z_past^0, z_future^τv', l, τv') dτv'
3: h^τv ← v_φ^(k)(z_past^0, z_future^τv, l, τv)
4: A_t^0 ← A_t^1 + ∫_1^0 π_θ(A_t^τa, q_t, h^τv, τa, τv) dτa
5: return A_t^0
```

Step 2-3 是 video flow (partial denoising to τ_v)
Step 4 是 action flow (full denoising from noise to clean actions)

τ_v = 1 special case 下，step 2 退化 (积分上限=下限)，整个流程只需要:
1. Single video forward pass (from pure noise)
2. Full action denoising

这就是 paper 强调的 efficient marginal action sampling。

## 5. Training Algorithm (Algorithm 2)

```
Algorithm 2: Action Decoder Training(k, T_v, T_a)
1: repeat
2:   z_0^past, z_0^future, a_0, s_0, l ~ p_0(...)
3:   τ_v ~ T_v(τ_v); τ_a ~ T_a(τ_a)        ← independent sampling!
4:   ε_v, ε_a ~ N(0, I)
5:   z_τv^future ← (1-τ_v)z_0^future + τ_v ε_v
6:   a_τa ← (1-τ_a)a_0 + τ_a ε_a
7:   h_τv ← v_φ^(k)(z_0^past, z_τv^future, l, τ_v)    ← video backbone FROZEN
8:   Gradient step on ||π_θ(a_τa, s_0, h_τv, τ_a, τ_v) - u_τa(a_τa|a_0)||^2
9: until converged
```

关键设计点:
- **τ_v 和 τ_a 独立采样** — 这是核心创新，让 action decoder 在所有 noise level 上都 robust
- T_v 用 logit-normal distribution (匹配 video pretraining)
- T_a(τ_a) ∝ √(τ_a - 0.001) — following π0 [3]，bias toward higher noise (more challenging regime)

这个 independent sampling 设计让 inference 时 τ_v=1 (pure noise) 也在 training distribution 内 — 所以 single forward pass 才能工作。

### 两阶段 Training

**Stage 1: Video backbone finetuning**
- 用 LoRA [23] 在 robotics video datasets 上 finetune Cosmos-Predict2
- 只调 LoRA 参数，保留 internet-scale pretrain 知识
- 目的: align generalist backbone 到 robot domain (visual style + dynamics)

**Stage 2: Action decoder training**
- Video backbone **frozen**
- Action decoder from scratch
- 只需要 scarce task-specific action data

这个 decomposition 让 expensive robot teleoperation 只用在 lightweight decoder 上。

## 6. 实验结果详解

### Table I: SIMPLER-Bridge Results

| Model | Carrot | Spoon | Blocks | Eggplant | Avg SR (%) |
|-------|--------|-------|--------|----------|-----------|
| OpenVLA (finetuned) [26] | 4.2 | 8.3 | 0.0 | 45.8 | 14.6 |
| Octo (finetuned) [49] | 8.3 | 12.5 | 0.0 | 43.1 | 16.0 |
| ThinkAct (pretrained) [24] | 37.5 | 58.3 | 8.7 | 70.8 | 43.8 |
| FLOWER (finetuned) [46] | 13.0 | 71.0 | 8.0 | 88.0 | 45.0 |
| π0.5-style VLA (scratch) | 25.0 | 29.2 | 20.8 | 66.7 | 35.4 |
| **mimic-video (scratch)** | 37.5 | 37.5 | 12.5 | 100.0 | **46.9** |
| mimic-video (+ τv-tuning) | 54.2 | 41.7 | 29.2 | 100.0 | **56.3** |

观察:
- mimic-video 从 scratch 训练 (只用 BridgeDataV2 action data) 超越所有 finetuned baselines
- 在 Eggplant 任务上达到 100% success rate
- **per-task τv-tuning** 提升 +9.4% — 这是 inference-time policy optimization 的 free lunch

### Table II: LIBERO Results

| Model | Spatial (%) | Object (%) | Goal (%) | Avg (%) |
|-------|-------------|------------|----------|---------|
| Diffusion Policy (scratch) | 78.3 | 92.5 | 68.3 | 79.7 |
| Octo (finetuned) | 78.9 | 85.7 | 84.6 | 83.1 |
| DiT Policy (finetuned) | 84.2 | 96.3 | 85.4 | 88.6 |
| OpenVLA (finetuned) | 84.7 | 88.4 | 79.2 | 84.1 |
| OpenVLA-OFT (finetuned) | 96.2 | 98.3 | 96.2 | 96.9 |
| π0.5-style VLA (scratch) | 79.2 | 94.0 | 84.4 | 85.9 |
| **mimic-video (scratch)** | 94.2 | 96.8 | 90.6 | **93.9** |

mimic-video 从 scratch 训练在 LIBERO 上接近 OpenVLA-OFT (finetuned)，且显著超过 π0.5-style VLA。

### Table III: Real-world Bimanual Dexterous

| Model | Packing | Package handover |
|-------|---------|------------------|
| DiT-Block Policy [10] | 11.0 | 30.0 |
| DiT-Block Policy (+ wrist cams) | 42.6 | 74.1 |
| **mimic-video** | **72.0** | **93.0** |

最 striking 的发现: **mimic-video 只用 single workspace camera** 就超过 multi-view DiT-Block Policy。说明 video prior 能 bridge occlusion-induced visual uncertainty — 这正是 video model 的 strong suit。

训练数据量:
- Video backbone finetune: 200 hours bimanual video
- Action decoder for sorting: 仅 1h 33m (512 episodes)
- Action decoder for stowing: 仅 2h 14m (480 episodes)

### Fig. 5 & 6: Sample Efficiency & Convergence

Fig. 5 显示 **10x sample efficiency**:
- mimic-video 用 10% data 达到 VLA baseline 的 max success rate
- 用 2% data (1 episode/task) 仍有 77% success rate on LIBERO
- 用 2% data 已 competitive with Diffusion Policy baseline

Fig. 6 显示 **2x convergence speed**:
- mimic-video action decoder 收敛更快
- Asymptotic success rate 更高
- 即便 VLA baseline 经历 FAST-pretraining on task data，仍然被超越

## 7. τ_v 反直觉现象 (Sec V-C, Appendix E)

这是 paper 最 intellectually interesting 的发现。

### Hypothesis: 更高 fidelity video → 更好 policy? **FALSE**

Fig. 7 显示在 SIMPLER-Bridge 上，**τ_v = 1 (pure noise) 反而 performance 最好**。

直觉上应该 τ_v → 0 (fully denoised) 时 mutual information I(z_future^τv; A^0) 最大，policy 应该更好。但实际相反。

### 两个解释机制 (Appendix E)

**Mechanism 1: Distribution Mismatch + Noise as Augmentation**

Training 时 decoder 看到的是 **ground truth** future video 的 latents。Inference 时 video model 生成可能 imperfect 或 subtly OOD。完全 denoise → 强 distribution shift → 性能下降。

留 noise 在 visual plan 中 = **train/test-time augmentation**:
- 防止 decoder 依赖 spurious ground-truth visual cues
- 类似 goal-conditioned policy [21] 用 image augmentation 提升 robustness

Oracle case study (Fig. 2) 印证: ground-truth latents → perfect performance，所以 regular inference 的不足来自 generation errors。

**Mechanism 2: Information Content of Intermediate Representations**

这个 explanation 更深刻。Flow matching 模型在 denoising 过程中:
- **中间 τ_v**: hidden states 必须 encode rich dynamics info + 必要 transformations 到达 clean video
- **τ_v → 0**: input 已经接近 target，video model 学到的是 **close-to-identity mapping**，hidden states 变得不 informative

Fig. 8 直接验证: 当 decoder conditioned on "noisy" ground-truth latents, **最优 τ_v ≈ 0.4** — 中间值，不是 0 (clean) 也不是 1 (pure noise)。

这暗示 video model 在 early layers (high noise regime) 编码 high-level plan，在 late layers (low noise) 只做 refinement。

### Practical implication

τ_v = 1 时:
- Single forward pass of video backbone (no ODE integration)
- Best average performance
- Fastest inference

这是 **win-win**: 性能最好 + 速度最快。这个 finding 让整个架构 practical。

## 8. 与相关工作对比

### VLA 系列
- **RT-2** [60] (Google, 2023): VLA 范式开创者，用 PaLI-X 5B VLM
  - https://arxiv.org/abs/2307.15818
- **OpenVLA** [26] (Stanford/Berkeley, 2024): 开源 VLA, 7B Llama-2 backbone
  - https://arxiv.org/abs/2406.09246
- **π0** [3] (Physical Intelligence, 2024): Flow matching action decoder + VLM backbone
  - https://arxiv.org/abs/2410.24164
- **π0.5** [25] (Physical Intelligence, 2025): Open-world generalization, FAST tokenization
  - https://arxiv.org/abs/2504.16054
- **Knowledge Insulation** [13]: 两阶段训练 (autoregressive + flow matching decoder)
  - https://arxiv.org/abs/2505.23705

### Video-based Policy Learning
- **Video Policy** [30] (CMU, 2025): 显式建模 joint video-action distribution，但不支持 efficient marginal sampling
  - https://arxiv.org/abs/2508.00795
- **CoT-VLA** [56] (NVIDIA, 2025): VLM 生成 subgoal image + actions 自回归
  - https://arxiv.org/abs/2503.22020
- **LAPA** [54] (Microsoft, 2025): 预训练 VLM predict "latent actions" (image diff encoding)
  - https://arxiv.org/abs/2410.11758
- **FLARE** [58] (NVIDIA, 2025): Align VLA representations 与 future VLM embeddings (implicit world model)
  - https://arxiv.org/abs/2505.15659
- **Unified World Models** [59] (Berkeley, 2025): 从 scratch 学 unified model
  - https://arxiv.org/abs/2504.02792
- **Dreamitate** [29] (Columbia, 2024): Generate pixel-space video + tool tracking
  - https://arxiv.org/abs/2406.16862
- **Video Language Planning** [15] (Google, 2023): Pixel-based IDM after video generation
  - https://arxiv.org/abs/2310.10625

### Video Foundation Models
- **Cosmos** [37]: NVIDIA 物理 AI world foundation model platform
  - https://arxiv.org/abs/2501.03575
- **Cosmos 2** [38]: World simulation with video foundation models
  - https://arxiv.org/abs/2511.00062
- **V-JEPA 2** [1]: Meta 自监督 video model，理解+预测+planning
  - https://arxiv.org/abs/2506.09985
- **Video models as zero-shot learners** [53]: Compositional reasoning emergent in video models
  - https://arxiv.org/abs/2509.20328

### World Models for Control
- **Strengthening Generative Robot Policies** [43]: Predictive world modeling
  - https://arxiv.org/abs/2502.00622
- **CTRL-World** [20]: Controllable generative world model
  - https://arxiv.org/abs/2510.10125
- **Gemini Robotics in Veo World** [48]: 用 world simulator 评估 policies
  - https://arxiv.org/abs/2512.10675

## 9. 关键 Insight 总结与 Build Intuition

### Insight 1: Video Pretraining > Vision-Language Pretraining for Control

为什么? VLM 数据是 static (image-text)，缺乏 temporal physics。Video 数据 inherent 包含:
- Object permanence across frames
- Physics (gravity, collision, deformation)
- Causal dynamics (action → effect)
- Procedure knowledge (how to pour, fold, sort)

所以 video model 已经 implicit 学了 **forward model** (隐式 world model)，剩下只需 inverse dynamics。

### Insight 2: Inverse Dynamics is Easy, Forward Dynamics is Hard

这是 paper 的 conceptual core:
- Forward dynamics (predict future from current state + action): **multi-modal, causal, hard**
- Inverse dynamics (predict action from current state + future): **unimodal, non-causal, easy**

把 hard 部分 offload 给 video backbone (用海量 internet video 学)，easy 部分给 lightweight decoder (用 scarce robot data 学)。这对应 Lynch et al. [34] "Learning Latent Plans from Play" 的思想。

### Insight 3: Partial Denoising = Free Inference-Time Optimization

τ_v 是 free hyperparameter，inference 时可调:
- τ_v = 1: 最 fast，平均最好 (default)
- Per-task tuning: +9.4% on SIMPLER

这相当于一种 **inference-time policy specialization**，无需 retraining。

### Insight 4: 中间 Representations > 最终 Output

Fig. 8 显示最优 τ_v ≈ 0.4 (不是 0)。这暗示 video DiT 的中间 layers 编码 high-level plan，final layers 做 low-level refinement。

这跟 LLM 中间 layer probing 发现 (Tenney et al.)，BERT 中间层更 informative 类似 — representations 在最终 output 阶段被 "compressed" 成 task-specific format。

## 10. 局限性与 Future Directions

paper 自己指出:
1. **Single-view only** — video backbone 只支持 single workspace view，限制了 spatial reasoning 和 occlusion robustness。Multi-view video model (类似 4D reconstruction) 可能解决。
2. **Cross-embodiment unification** — 还没 scale 到 large-scale cross-embodiment pretraining。
3. **Real-world task diversity** — 当前只在两个 bimanual task 测试。

我额外想到的 limitations:
1. **Action data quality** — Action decoder 还是需要 high-quality teleop data，无法完全 zero-shot
2. **Latency** — 即使 τ_v=1，2B Cosmos backbone forward pass 仍很重，real-time 控制频率受限 (paper 没报告 Hz)
3. **Video model hallucination** — 如果 video backbone 生成 physically impossible plan，action decoder 没有纠正机制
4. **Long horizon planning** — Action chunk 长度 H_a 可能限制长时序任务

## 11. 联想到的扩展方向

### A. Self-Supervised Video Pretraining 替代 Generative Video

Cosmos-Predict2 是 generative model。V-JEPA 2 [1] (Meta) 是 **non-generative** self-supervised video model，可能更适合:
- 不需要 pixel-space generation (本身就更 efficient)
- Latent predictive learning 直接 align with action decoder use case
- 速度可能更快

V-JEPA 2: https://arxiv.org/abs/2506.09985

### B. Joint Video-Action Diffusion

paper 用 sequential denoising (先 video partial, 再 action full)。是否可以 **joint diffusion** over (z, A) pair with shared flow time? 类似 Unified World Models [59] 的思路。

可能 advantage: 让 video 和 action 互相 influence during denoising。

### C. Action-Conditioned Video Model

paper 用 **non action-conditioned** video model (类似 Pi0.5 也是)。但 video model 如果 action-conditioned (像 world model):
- 可以想象多个 candidate action 的 future
- 用世界模型评估挑选最优 action
- 类似 search-based planning

这是 **Generative World Models** 的方向 — paper 提到了 [1, 43]。

### D. Hierarchical VAM

Action chunk H_a 限制长时序。可以 hierarchical:
- High-level VAM: 长 horizon visual planning
- Low-level VAM: 短 horizon precise control

类似 hierarchical RL 但 with video latents。

### E. Multi-View Video Foundation Models

当前 Cosmos 是 single-view。如果 video model 天生 multi-view (像 4D reconstruction 或 NeRF-based):
- 直接解决 occlusion 问题
- Wrist camera 不再 needed (Fig. 4 真实任务中很 critical)
- 类似 DreamGaussian + video

### F. Cross-Embodiment VAM Foundation Model

paper 没做 unified cross-embodiment model。如果用 large video model + multi-embodiment action decoders:
- One frozen Cosmos backbone
- Multiple lightweight decoders per embodiment
- 类似 RT-X [9] 但 with video prior

### G. Video Model In-the-Loop with Real Robot

当前 video model 是 open-loop plan generation。可以闭环:
- Real obs → video plan → action → real obs (next step)
- 用 video model 验证 action 执行效果
- 类似 MPC with learned world model

### H. Joint Training of Video Backbone + Action Decoder

paper 用 frozen video backbone (训练效率考虑)。但 joint training 可能:
- 让 video representation 更适合 action decoding
- Risk: catastrophic forgetting of physics priors
- 解决方案: LoRA on video backbone + decoder training

### I. Video Model Finetuning with Action Data

Stage 1 finetune video backbone with video only。能否用 action labels 进一步 finetune? 
- Action-conditioned video generation
- 类似 HPT (Heterogeneous Pretrained Transformer) 的 latent space

### J. Text-to-Video Pretraining 的 Implicit Reward

Internet video 中 human action 是 implicit optimal demonstration。能否用 RLHF-style 方法从 video preference 学 reward model? 类似 RLHF for LLMs。

## 12. 总结

mimic-video 的核心贡献是 conceptual shift:
**从 "VLM + action data 学 physics" 到 "Video model 已懂 physics + action data 学 IDM"**

这个 shift 让:
- Robot data 需求降低 10x
- 收敛速度提升 2x
- Single-view 能超越 multi-view baseline (occlusion robustness)
- Inference 时 single forward pass 可行 (τ_v=1)

Technical contributions:
- Dual flow matching architecture with independent (τ_v, τ_a) schedules
- Partial denoising strategy for efficient marginal action sampling
- Inference-time τ_v tuning as free policy optimization
- Empirical finding: intermediate representations > final output for action decoding

我个人认为这是 robotics foundation model 方向 **paradigm-shifting** 的工作。VLA 的根本 modality mismatch 问题被指出，video 作为 pretraining modality 的优势被实证。

接下来几年我会 watch:
1. V-JEPA 2 + action decoder 类似架构 (Metas 方向)
2. NVIDIA 推出 action-conditioned Cosmos (Veo Robotics 方向)
3. Cross-embodiment VAM foundation model (取代 RT-X 范式)
4. Real-time inference optimization (τ_v=1 + distillation)

相关 Links 汇总:
- Paper: https://mimic-video.github.io
- Cosmos: https://arxiv.org/abs/2501.03575
- Cosmos 2: https://arxiv.org/abs/2511.00062
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- LAPA: https://arxiv.org/abs/2410.11758
- Video Policy: https://arxiv.org/abs/2508.00795
- Unified World Models: https://arxiv.org/abs/2504.02792
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748
- LIBERO: https://arxiv.org/abs/2306.03310
- SIMPLER: https://arxiv.org/abs/2405.05941
- BridgeDataV2: https://arxiv.org/abs/2308.12952
- FAST: https://arxiv.org/abs/2501.09747 (Pertsch et al.)
- mimic-one: https://arxiv.org/abs/2506.11916
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- CoT-VLA: https://arxiv.org/abs/2503.22020
- Dreamitate: https://arxiv.org/abs/2406.16862
- Video Language Planning: https://arxiv.org/abs/2310.10625
- Robotic Control via Embodied CoT: https://arxiv.org/abs/2407.08693
- Gemini Robotics + Veo: https://arxiv.org/abs/2512.10675
- Strengthening Generative Policies: https://arxiv.org/abs/2502.00622
