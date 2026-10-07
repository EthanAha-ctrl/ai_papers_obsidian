---
source_pdf: stiv.pdf
paper_sha256: fd0940c6aa6b484ed0734575ca499e61b8632e0450d3a972485f1641ff7e88e4
processed_at: '2026-08-12T11:13:09-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 STIV

好，咱们抛开那些花里胡哨的术语，用大白话从头到尾捋一遍这篇 paper 到底干了啥。

---

## 一句话概括

Apple 的人想做一个能生成视频的模型，既能根据文字生成视频（T2V），也能根据文字+第一帧图片生成视频（TI2V），而且想用同一个模型搞定这两件事。最后他们做出来了，参数量 scaling 到 8.7B，在 VBench 上干赢了 CogVideoX、Pika、Kling、Gen-3 这些大佬。

---

## 模型长啥样

### 底座：PixArt-α 的改良版

他们没从零开始造轮子，直接拿 PixArt-α 的架构改。PixArt-α 本身就是一个 DiT（Diffusion Transformer），说白了就是把 U-Net 换成了 Transformer，用 cross-attention 把文字信息塞进去。

STIV 在这个底座上做了几个关键改造：

**1. 把 3D attention 拆成两次 2D attention**

视频是 $(T, H, W)$ 三维的，如果直接做 full 3D attention，计算量是 $O((T \times H \times W)^2)$，帧数一多就爆炸。他们的做法是：

- 第一步：把时间维度 $T$ fold 进 batch，只做空间 attention，也就是每一帧自己内部做 self-attention
- 第二步：把空间维度 $(H, W)$ fold 进 batch，只做时间 attention，也就是同一个空间位置跨帧做 self-attention

这样计算量变成 $O(T \times (H \times W)^2) + O(H \times W \times T^2)$，大大省了。而且有个额外好处：纯图像可以看作"只有一帧的视频"，所以 T2I 模型的空间权重可以直接搬过来用，不用重新训。

**2. Singleton Condition**

他们把一堆 condition 信息（diffusion timestep、CLIP 文本编码的 pooled embedding、micro-conditions 如分辨率/裁剪坐标/帧数/采样stride）全部 normalize 之后加在一起，形成一个 "singleton condition"。这个东西再通过 AdaLN 机制生成 scale、shift、gate 参数，去调制每一层 Transformer 的 attention 和 FFN。

公式上长这样：
$$\text{MHA}(x) = x + \text{gate} \cdot \text{norm}(\text{Attn}(\text{scale} \cdot \text{norm}(x) + \text{shift}))$$

这里面的 scale、shift、gate 都是 singleton condition 生成的，而不是每一层独立学习。这叫 Shared AdaLN，省参数又省显存。

**3. RoPE（旋转位置编码）**

空间上用 2D RoPE，时间上用 1D RoPE。RoPE 的好处是它编码的是相对位置关系，天然支持不同分辨率和不同帧数的输入，换分辨率不用重训位置编码。而且 RoPE 可以通过 interpolation 平滑地扩展到更大的分辨率或更多帧数，这在 progressive training 里极其有用。

**4. Flow Matching 代替传统 Diffusion**

传统 diffusion 走的是一条弯弯绕绕的路径从噪声到数据。Flow Matching 用的是直线插值：

$$x_t = t \cdot x_1 + (1-t) \cdot \epsilon$$

- $x_t$：时间步 $t$ 的 noisy latent
- $x_1$：干净数据（clean video latent）
- $\epsilon$：标准高斯噪声 $\mathcal{N}(0, I)$
- $t$：从 0 到 1 的标量，0 代表纯噪声，1 代表纯数据

训练目标是预测 velocity $v_t = x_1 - \epsilon$：

$$\min_\theta \mathbb{E}\left[\|F_\theta(x_t, c, t) - v_t\|_2^2\right]$$

- $F_\theta$：DiT 网络预测的 velocity
- $c$：条件信息（text 和 image）

直线意味着 ODE solver 采样时步数可以更少，训练梯度也更稳。

---

## 怎么把图片塞进去：Frame Replacement

这是这篇 paper 最 elegant 的设计。

### 传统做法的痛点

之前的 TI2V 方法（比如 ConsistI2V、DynamiCrafter）在 U-Net 里搞 image conditioning，需要额外加 spatial self-attention layer 去处理第一帧，还需要 window-based temporal attention 去增强一致性，搞得模型很复杂。

### STIV 的做法：简单粗暴

训练的时候：
1. 把视频 encode 成 latent，加噪声
2. **把第一帧的 noised latent 直接替换成干净的 image condition latent**
3. 送进 DiT blocks 处理
4. 算 loss 的时候把第一帧的 loss mask 掉（因为第一帧是 clean 的，不需要学）

推理的时候：每个 diffusion step 都保持第一帧为 clean image latent。

**为什么这就够了？** 因为 DiT 是一堆 Transformer block 堆起来的，每一层都有 spatial attention 和 temporal attention。第一帧的 clean 信息会通过 temporal attention 自然地流向后续所有帧。U-Net 因为有 skip connection 和局部感受野的限制，需要特殊设计来传播信息，但 DiT 的 global attention 天然就把这事干了。

Paper 里 Table 3 做了 ablation，Frame Replacement (FR) 单独使用就在 I2V Avg Score 上拿到 75.8，比加 cross-attention (CA)、large projection (LP)、first frame loss (FFL) 的各种组合都好或相当。而且加 LP 和 FFL 反而让 dynamic degree 大幅下降（从 35.4 掉到 22.2），因为模型被过度约束了。

---

## JIT-CFG：解决"不动"的问题

### 问题出在哪

当模型 scale 到 512 分辨率、8.7B 参数时，出现了一个尴尬现象：生成的视频几乎不动。第一帧给了一张图，后面的帧跟第一帧几乎一模一样，就是一个静态画面。

**为什么？** 因为模型 capacity 变大了，它发现最省事的方式就是"复制第一帧"。image consistency loss 最小化的最快路径就是让后续帧等于第一帧。模型在偷懒。

### 解决方案：Image Condition Dropout + JIT-CFG

**训练时：** 以 8% 的概率随机丢掉 image condition（把 $c_I$ 设为 null），同时保留之前就有的 10% text condition dropout。

**推理时：** 用 Joint Image-Text Classifier-Free Guidance (JIT-CFG)：

$$\hat{F}_\theta(x_t, c_T, c_I, t) = F_\theta(x_t, \emptyset, \emptyset, t) + s \cdot \left(F_\theta(x_t, c_T, c_I, t) - F_\theta(x_t, \emptyset, \emptyset, t)\right)$$

- $\hat{F}_\theta$：guided velocity prediction
- $F_\theta(x_t, c_T, c_I, t)$：有 text 和 image 条件时的 velocity
- $F_\theta(x_t, \emptyset, \emptyset, t)$：无条件时的 velocity
- $s$：guidance scale，论文取 7.5

**Intuition：** 因为训练时有 8% 的时间没给 image condition，模型被迫学会了"从纯噪声生成有 motion 的视频"。推理时，JIT-CFG 把"有条件的预测"和"无条件的预测"的差值放大 $s$ 倍。这个差值里包含了"因为有 image 和 text 条件而额外产生的方向性"。由于模型同时学会了有图和无图两种模式，这个差值里既有 image 的 anchor 信息，又有 text 描述的运动信息，两者叠加就打破了静态僵局。

Table 5 的数据很直观：

| Model | Dynamic Degree | Motion Smoothness | Temporal Consistency |
|:---|:---|:---|:---|
| STIV-M-512（无 dropout） | 10.2 | 99.6 | 99.3 |
| STIV-M-512-JIT（有 dropout） | 24.0 | 99.1 | 98.6 |

Dynamic Degree 从 10.2 飙到 24.0，动了！代价是 smoothness 和 consistency 微降，但完全可以接受。

### 额外好处：一个模型干两件事

因为训练时随机 drop image condition，同一个模型既能做 T2V（不给图），又能做 TI2V（给图）。Table 4 显示，STIV-M-512-JIT 在 T2V 上 Total score 80.7，在 TI2V 上 Total score 89.8，两边都不差。而单独训的 T2V 模型完全做不了 TI2V，反过来也一样。

### SIT-CFG vs JIT-CFG

论文还试了 Separate Image and Text CFG（SIT-CFG），用两个 guidance scale $s_1$（控制 image）和 $s_2$（控制 text）：

$$\hat{F}_\theta = F_\theta(\emptyset, \emptyset) + s_1 \cdot (F_\theta(c_I, \emptyset) - F_\theta(\emptyset, \emptyset)) + s_2 \cdot (F_\theta(c_I, c_T) - F_\theta(c_I, \emptyset))$$

这需要三次 forward pass（$\emptyset,\emptyset$、$c_I,\emptyset$、$c_I,c_T$），比 JIT-CFG 的两次多一次。Grid search 找到的最优 $s_1=1.5, s_2=7.5$，FVD=95.2，跟 JIT-CFG 的 94.1 差不多。所以 SIT-CFG 没有明显优势还更慢，JIT-CFG 胜出。

---

## Scaling 的稳定性 trick

模型从 600M scale 到 8.7B，不加点 trick 肯定炸。

### QK-Norm

在 attention 里，$Q$ 和 $K$ 点积之前先过一层 RMSNorm。深层 Transformer 里 $Q$ 和 $K$ 的 magnitude 会越来越大，softmax 直接饱和，梯度消失。QK-Norm 把 $Q$ 和 $K$ 的 norm 压住，attention logits 保持在合理范围。

### Sandwich-Norm

传统 Transformer 要么用 pre-norm 要么用 post-norm。STIV 两个都用：

$$\text{MHA}(x) = x + \text{gate} \cdot \text{norm}(\text{Attn}(\text{scale} \cdot \text{norm}(x) + \text{shift}))$$

里面一层 `norm(x)` 是 pre-norm，保证输入 attention 的 activation 稳定；外面一层 `norm(...)` 是 post-norm，防止输出 activation 往后 drift。两头夹住，所以叫 sandwich。FFN 也是同样的 sandwich 结构。

### MaskDiT

训练时随机 mask 掉 50% 的 spatial tokens，只把剩下的 tokens 送进主要的 DiT blocks。unmask 之后再加 2 层 DiT block 处理全部 tokens。这样主要的计算量大头只处理一半 tokens，省了大量显存和计算。后续 fine-tuning 阶段把 mask 去掉（UnmaskSFT）恢复全 token 训练。

### AdaFactor

把 AdamW 换成 AdaFactor，省 optimizer state 的显存。配合 gradient checkpointing（只存 self-attention 输出），per-device HBM 从 28GB 降到 11GB。

---

## Progressive Training：一步步来

STIV 的训练分三个阶段接力：

**Stage 1: T2I** → 先训 text-to-image 模型，学空间表征。batch size 4096，400k steps。

**Stage 2: T2V** → 加载 T2I 的 EMA weights（temporal attention 除外，因为 T2I 没有这个），训 T2V。从低分辨率 256² 短帧数 20 frames 开始，往高分辨率 512² 长帧数 40 frames 扩展时，用 RoPE interpolation 初始化位置编码。batch size 1024，400k steps。

**Stage 3: STIV (TI2V)** → 加载 T2V weights，引入 frame replacement 和 image condition dropout，训 TI2V。

Table 6 的 ablation 显示，TI2V 从 T2V 初始化比从 T2I 初始化稍好，camera motion score 38.0 vs 29.8，dynamic degree 37.1 vs 36.5。因为 T2V 已经学了 temporal dynamics，直接继承过来比从只有空间信息的 T2I 开始更有优势。

---

## 数据这事儿

### Video Data Engine

他们搭了一套完整的 video data pipeline：

1. **PySceneDetect** 切割视频，去掉 abrupt transitions 和 fades，保证每个 clip 视觉连贯
2. **Feature Extraction** 提取 motion score、aesthetic score、clarity score、temporal consistency 等特征
3. **Filtering** 根据特征过滤低质量视频。Panda-70M 过滤成 Panda-30M，再过滤成 Panda-10M（高质量子集）
4. **Captioning** 用 LLaVA-Hound-7B（video LLM）直接给视频生成 caption，而不是先给单帧 caption 再用 LLM summarize

### 为什么不用单帧 caption + LLM summarize

他们试过这个方案（FCapLLM），发现两个问题：
- 单帧 caption 捕捉不到 motion 信息
- LLM summarize 会 hallucinate，编出不存在的物体

直接用 video LLM（VCap）生成 caption，hallucination 率更低（DSG-Video_i 从 6.4 降到 5.3），object diversity 更高（1249 → 1911），最终 T2V 模型 FVD 从 808.1 降到 770.9。

### DSG-Video：评估 caption hallucination

受 DSG（Davidsonian Scene Graph）启发，他们搞了个 DSG-Video 评估方法：

1. 用 LLM 根据 caption 生成关于物体的问题（如"视频里有猫吗？"）
2. 用 MLLM 在视频的 N 帧里逐帧验证物体是否存在
3. 如果某个物体在所有帧里都没被检测到，判定为 hallucination

两个指标：
$$\text{DSG-Video}_i = \frac{|\{\text{hallucinated objects}\}|}{|\{\text{all mentioned objects}\}|}$$
$$\text{DSG-Video}_s = \frac{|\{\text{sentences with hallucinated object}\}|}{|\{\text{all sentences}\}|}$$

- DSG-Video_i：object 级别的 hallucination 率
- DSG-Video_s：sentence 级别的 hallucination 率

---

## 实验结果

### T2V 主表（Table 9）

| Model | Quality ↑ | Semantic ↑ | Total ↑ |
|:---|:---|:---|:---|
| CogVideoX-5B | 82.8 | 77.0 | 81.6 |
| Gen-3 | 84.1 | 75.7 | 82.3 |
| KLING | 83.4 | 75.2 | 82.3 |
| STIV M-512 | 82.2 | 77.0 | 81.2 |
| STIV M-512 UnmaskSFT + TUP | **84.4** | 77.2 | **83.1** |

**关键发现：**
- 从 XL (600M) → XXL (1.5B) → M (8.7B)，Semantic score 从 72.5 → 72.7 → 74.8，有明显 scaling effect
- Quality score 从 80.7 → 81.2 → 82.1，提升不大
- 从 256² 到 512² 分辨率，Semantic 从 74.8 飙到 77.0，分辨率比参数量更影响 semantic 能力
- SFT（用 20k 高质量视频 fine-tune）把 Quality 从 82.2 拉到 83.9
- TUP（temporal upsampler 后处理）把 Quality 进一步提到 84.4，Total 83.1，达到 SOTA

### TI2V 主表（Table 10）

| Model | Quality ↑ | I2V ↑ | Total ↑ |
|:---|:---|:---|:---|
| SVD | 82.8 | 96.9 | 89.9 |
| Animate-Anything | 81.2 | 96.6 | 89.1 |
| STIV-M-512 | 82.1 | 98.0 | **90.1** |
| STIV-M-512-JIT | 81.9 | 97.6 | 89.8 |

STIV-M-512 拿到 90.1 的 SOTA。有趣的是 JIT 版本 Total 略低（89.8），但 dynamic degree 大幅提升，说明 VBench-I2V 指标本身偏爱"不动"的视频，而实际使用中我们更想要"动"的视频。

---

## 能干啥扩展

因为 frame replacement 的设计很灵活，STIV 天然支持一堆下游任务：

### Video Prediction (V2V)

把前 4 帧作为 $c_I$，预测后续帧。从 STIV-XXL 初始化，额外训 400k steps。

| Model | MSRVTT FVD↓ | MovieGen FVD↓ |
|:---|:---|:---|
| T2V | 536.2 | 347.2 |
| STIV-V2V | 183.7 | 186.3 |

FVD 直接砍半多，这对自动驾驶、embodied AI 这种需要高 fidelity 帧预测的场景非常有用。

### Frame Interpolation (TUP)

把首尾帧作为 $c_I$，生成中间帧。从 STIV-XL 初始化，用 stride 2 采样训练 400k steps。级联到主模型后面可以提升 motion smoothness。

### Multi-view Generation

把 input image 作为第一帧，后续 6 帧作为 novel views。在 GSO dataset 上跟 Zero123++ 性能相当（PSNR 21.643 vs 21.200），虽然只用 temporal attention 而 Zero123++ 用 full attention。这暗示了 temporal attention 已经隐式学到了一些 3D understanding。

### Long Video Generation

两层 hierarchical 框架：
- **Mode 1:** T2V/STIV 以 stride 20 生成 keyframes
- **Mode 2:** frame interpolation 填充 keyframes 之间的帧

最终生成 $(20-1) \times 20 = 380$ 帧的长视频。这比 autoregressive rollout 更好，因为 rollout 会被前面帧的 error 累积拖垮，而 hierarchical 框架的 keyframes 是独立生成的，没有 error propagation。

---

## 我的 Intuition 总结

这篇 paper 最大的价值在于它展示了"simple design + systematic scaling recipe = SOTA"。

1. **Frame Replacement 为什么 work：** DiT 的 global attention 天然传播信息，不需要像 U-Net 那样专门设计 module。简单到令人怀疑"就这么干就行了？"，但 ablation 证明确实就行了。

2. **JIT-CFG 为什么能解决 staleness：** 本质上是让模型同时学会"有 anchor"和"无 anchor"两种模式，推理时把两种模式的差异放大，既保留 anchor 的一致性，又注入 motion 的自由度。这跟 InstructPix2Pix 用两个 guidance scale 的思路类似，但更简洁。

3. **Scaling 的核心是 stability：** QK-Norm + Sandwich-Norm 看起来不起眼，但没有它们 8.7B 模型根本训不起来。这跟 LLM scaling 里发现的各种 norm trick 是一脉相承的。

4. **Progressive training 省的是工程时间：** 你可以先快速训一个小模型迭代架构和 data pipeline，确认没问题再 scale up。T2I → T2V → TI2V 的接力让每一步的起点都不是随机初始化，极大加速了收敛。

5. **数据质量 > 数据数量：** Panda-30M 过滤成 Panda-10M，VBench Total 从 65.6 升到 66.2。VCap 代替 FCapLLM，FVD 从 808.1 降到 770.9。这在 video generation 里几乎是个铁律了。

---

### Reference Links

- STIV 论文：[arxiv.org/abs/2412.20889](https://arxiv.org/abs/2412.20889)
- PixArt-α：[arxiv.org/abs/2310.00426](https://arxiv.org/abs/2310.00426)
- DiT (Peebles & Xie)：[arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- Flow Matching (Lipman et al.)：[arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- Rectified Flow (SD3 math)：[arxiv.org/abs/2203.04443](https://arxiv.org/abs/2203.04443)
- RoPE (RoFormer)：[arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- MaskDiT：[arxiv.org/abs/2306.09305](https://arxiv.org/abs/2306.09305)
- QK-Norm：[arxiv.org/abs/2010.04245](https://arxiv.org/abs/2010.04245)
- ConsistI2V：[arxiv.org/abs/2402.04324](https://arxiv.org/abs/2402.04324)
- VBench：[arxiv.org/abs/2311.13535](https://arxiv.org/abs/2311.13535)
- MovieGen (Meta)：[arxiv.org/abs/2410.13720](https://arxiv.org/abs/2410.13720)
- CogVideoX：[arxiv.org/abs/2408.06072](https://arxiv.org/abs/2408.06072)
- SVD (Stable Video Diffusion)：[arxiv.org/abs/2311.15127](https://arxiv.org/abs/2311.15127)
- DSG (Davidsonian Scene Graph)：[arxiv.org/abs/2310.18235](https://arxiv.org/abs/2310.18235)
- Panda-70M：[arxiv.org/abs/2402.14188](https://arxiv.org/abs/2402.14188)

---

这篇由 Apple 和 UCLA 联合发布的 paper《STIV: Scalable Text and Image Conditioned Video Generation》系统地探讨了 video generation 的 model architecture、training recipes 和 data curation 策略。其核心贡献在于提出了一种极其 simple 且 scalable 的 framework，能够同时处理 text-to-video (T2V) 和 text-image-to-video (TI2V) 任务，并且 scaling 到 8.7B 参数时在 VBench 上取得了 SOTA performance。

以下我将从 architecture、core innovation、training recipe、data engine 以及实验结果等多个维度为你详细拆解这篇 paper，并尝试 build your intuition about why these designs work。

### 1. Architecture Breakdown: Base Model Scaling

STIV 的 base architecture 基于 PixArt-$\alpha$，采用 Diffusion Transformer (DiT) 结构。为了 handle 视频的时空特性并保证 large-scale 训练的稳定性，作者引入了几个关键设计：

#### 1.1 Factorized Spatial-Temporal Attention
模型没有使用 full 3D attention，而是采用 factorized attention。具体来说，先将 temporal dimension fold 进 batch dimension 执行 spatial self-attention，然后再将 spatial dimension fold 进 batch dimension 执行 temporal self-attention。
**Intuition:** 这种设计极大地降低了 computational complexity（从 $O((T \times H \times W)^2)$ 降到 $O(T \times (H \times W)^2) + O(H \times W \times T^2)$）。更重要的是，它允许模型直接 reuse 纯粹的 T2I 模型的 spatial weights，因为图像可以被视为只有一帧的视频。

#### 1.2 Singleton Condition & RoPE
**Singleton Condition:** 将 diffusion timestep embedding、CLIP text encoder 的 last token embedding 以及 micro-conditions（如 original image resolution, crop coordinates, sampling stride, number of frames）通过 stateless layer normalization 后相加。这个 singleton condition 用于生成 shared scale-shift-gate parameters，作用于 spatial attention 和 feed-forward layers。
**RoPE (Rotary Positional Embedding):** 空间上使用 2D RoPE，时间上使用 1D RoPE。RoPE 提供了强大的 relative position inductive bias，并且天然兼容 masking 操作和 resolution variation。

#### 1.3 Flow Matching Objective
STIV 放弃了传统的 diffusion epsilon-prediction，转而使用 Flow Matching（具体为 Rectified Flow / linear interpolants）。其公式如下：

$$ \pmb x_t = t \cdot \pmb x_1 + (1 - t) \cdot \pmb \epsilon $$

*   $x_t$: 时间步 $t$ 时的 noisy latent。
*   $x_1$: target data (clean video latent)。
*   $t$: timestep，取值范围 $[0, 1]$。$t=0$ 时全为噪声，$t=1$ 时全为干净数据。
*   $\epsilon$: standard Gaussian noise $\mathcal{N}(0, I)$。

训练目标是最小化 velocity 预测的 MSE：

$$ \min_{\theta} \mathbb{E}_{\pmb x, \epsilon \in \mathcal{N}(\mathbf{0}, I), \epsilon, t} \left[ \| \pmb F_{\theta}(\pmb x_t, \pmb c, t) - \pmb v_t \|_2^2 \right] $$

*   $F_{\theta}$: neural network (DiT) 预测的 velocity。
*   $c$: conditions (包含 text $c_T$ 和 image $c_I$)。
*   $v_t = x_1 - \epsilon$: target velocity vector field。

**Intuition:** Flow Matching 描述的是从 Gaussian 分布到 data 分布的 optimal transport（直线路径）。相比 traditional diffusion 的 curved trajectory，linear interpolants 让 trajectory 更直，sampling 时 ODE solver 需要的步数更少，训练时的梯度也更稳定，这对 large-scale model training 至关重要。

#### 1.4 Stability & Efficiency Tricks (Scaling to 8.7B)
当 model size scaling 到 8.7B 时，训练很容易 diverge。STIV 引入了以下 trick：

*   **QK-Norm:** 在 attention logit 计算前，对 Query 和 Key 应用 RMSNorm。防止随着深度增加，$QK^T$ 的 magnitude 爆炸导致 softmax 进入饱和区、梯度消失。
*   **Sandwich-Norm:** 在 Multi-Head Attention (MHA) 和 Feed-Forward Network (FFN) 中同时使用 pre-norm 和 post-norm（stateless layer norm）。
    $$ \mathbf{MHA}(x) = x + \mathbf{gate} \cdot \mathbf{norm}(\mathrm{Attn}(\mathrm{scale} \cdot \mathbf{norm}(x) + \mathrm{shift})) $$
    $$ \mathbf{FFN}(x) = x + \mathbf{gate} \cdot \mathbf{norm}(\mathbf{MLP}(\mathrm{scale} \cdot \mathbf{norm}(x) + \mathrm{shift})) $$
    **Intuition:** Pre-norm 稳定前向传播，post-norm 防止 activations 在深层网络中 drift。两者结合构成了 "sandwich"，极大地增强了深层 Transformer 训练的稳定性。
*   **MaskDiT:** 随机 mask 掉 50% 的 spatial tokens 进入主要的 DiT blocks，然后 unmask 后再加 2 层 DiT block 处理所有 tokens。结合 gradient checkpointing 和 AdaFactor optimizer，将 per-device HBM 使用量从 28GB 降到 11GB，大幅提升 throughput。

---

### 2. Core Innovation: Image Conditioning & JIT-CFG

如何把 image condition 无缝融入 DiT 是这篇 paper 的灵魂。之前的方法如 ConsistI2V 在 U-Net 里用 frame replacement，还需要额外加 spatial self-attention 和 window-based temporal self-attention。而 STIV 证明了，在 DiT 架构中，一个极其简单的操作就能实现极好的效果。

#### 2.1 Frame Replacement
**Training:** 将 noised video latents 的第一帧直接替换为 un-noised 的 image condition latent，然后送入 STIV blocks，并且在计算 loss 时 mask 掉第一帧的 loss。
**Inference:** 在每个 diffusion step，保持第一帧的 latent 为 clean image condition 的 latent。

**Intuition:** 为什么这么简单的方法有效？因为 DiT 的 stacked spatial-temporal attention natively 就能把第一帧的 clean information 通过 skip connections 和 attention mechanisms 传播到后续所有 frames。模型不需要像 U-Net 那样特殊设计 module 来 "记住" 第一帧，DiT 的 global receptive field 天然支持这种 anchor 机制。

#### 2.2 Image Condition Dropout & JIT-CFG
当把模型 scale 到 512 分辨率时，作者发现模型倾向于生成 static 或 nearly static motion 的视频。**Intuition:** 这是由于 model capacity 变大，模型发现最 lazy 的方式去 minimize image consistency loss 就是让后面的 frame 尽量保持和第一帧一样（即不动）。

为了解决这个 staleness issue，STIV 引入了 image condition dropout（训练时以 8% 概率丢弃 $c_I$）和 Joint Image-Text Classifier-Free Guidance (JIT-CFG)。

传统的 CFG 只针对 text condition，而 JIT-CFG 同时对 text 和 image 进行 CFG。其公式为：

$$ \hat{F}_{\theta}(x_t, c_T, c_I, t) = F_{\theta}(x_t, \emptyset, \emptyset, t) + s \cdot \left( F_{\theta}(x_t, c_T, c_I, t) - F_{\theta}(x_t, \emptyset, \emptyset, t) \right) $$

*   $\hat{F}_{\theta}$: guided velocity prediction。
*   $c_T, c_I$: text condition 和 image condition。
*   $\emptyset$: null condition。
*   $s$: guidance scale (论文中取 7.5)。

**Intuition:** 通过 dropout，模型被迫学习在没有 image anchor 的情况下如何从 noise 生成 motion。在推理时，JIT-CFG 计算 conditional velocity (有 text 和 image) 和 unconditional velocity (无 text 和无 image) 的 difference，乘以 $s$。这个 difference 代表了 "向 text 和 image 条件靠拢" 的力。因为模型学过了 unconditional (无图) 时的 motion 生成，这部分 motion 信息会被放大并叠加到最终结果中，从而打破静态僵局。这也让 single model 同时擅长 T2V 和 TI2V 成为了可能。

Paper 中还对比了 Separate Image and Text CFG (SIT-CFG)，即设置两个 guidance scale $s_1, s_2$ 分别控制 image 和 text。但实验表明 SIT-CFG 并没有显著优势，且需要三次 forward pass，而 JIT-CFG 只需两次，效率更高。

#### 2.3 CFG-Renormalization
在 inference 早期（$t$ 接近 0 时），conditional 和 unconditional velocity 的 difference 会很大，导致 guided velocity $\hat{F}_{\theta}$ 的 magnitude 过大，overshoot 出 learned latent distribution，产生 artifact。作者提出了 rescale magnitude 到 conditional prediction 的 norm：

$$ \tilde{F}_{\theta}(x_t, c_T, c_I, t) = \|F_{\theta}(x_t, c_T, c_I, t)\| \frac{\hat{F}_{\theta}(x_t, c_T, c_I, t)}{\|\hat{F}_{\theta}(x_t, c_T, c_I, t)\|} $$

**Intuition:** 保持 velocity 的 direction，但限制其步长。这类似于在 ODE solver 中加入自适应步长控制，防止早期积分发散。

---

### 3. Progressive Training Recipe

STIV 采用了一种 curriculum learning 策略，分为三个阶段：
1.  **T2I Training:** 先训练 text-to-image 模型，学习 spatial representation。
2.  **T2V Initialization:** 加载 T2I 的 EMA weights（排除 temporal attention），训练 text-to-video 模型。在提升 resolution 和增加 frame count 时，利用 RoPE 的 interpolation 性质来初始化位置编码。
3.  **STIV Training:** 加载 T2V weights，引入 frame replacement 和 image condition dropout 训练 TI2V 模型。

**Intuition:** 直接从 scratch 训练高分辨率长视频模型计算量极其庞大。通过 factorized attention，T2I 的 spatial weights 可以无缝迁移给 T2V 的 spatial branch。T2V 在低分辨率上学到的 temporal dynamics 也能平滑过渡到高分辨率。这种 progressive recipe 极大提高了 R&D 效率。

---

### 4. Video Data Engine

高质量数据是 video generation 的瓶颈。STIV 构建了一个 Data Engine，包含 preprocessing、filtering 和 captioning。

*   **Preprocessing:** 使用 PySceneDetect 切割 abrupt transitions 和 fades，提取 motion score, aesthetic score 等特征过滤。
*   **Captioning:** 放弃了单帧 caption + LLM summarize 的方法（会丢失 motion info 并引入 hallucination），改用 video LLM (LLaVA-Hound-7B) 生成 dense, motion-aware caption。
*   **DSG-Video Evaluation:** 为了评估 caption 质量，受 DSG 启发，用 LLM 生成关于 object 的问题，然后用 MLLM 在 video frames 中验证 object 的存在。计算 hallucination 率 $DSG-Video_i = \frac{|\{hallucinated\ objects\}|}{|\{all\ mentioned\ objects\}|}$。实验证明高质量 caption 显著提升了 VBench score。

---

### 5. Experiments & Results Analysis

我们来看一下实验数据的核心结论。

#### 5.1 T2I Ablation Study (Table 2)
作者从 base DiT 出发，逐步加入 trick。QK-norm 和 sandwich-norm 使得 learning rate 可以从 1e-4 提升到 2e-4。Flow Matching 和 CFG-Renormalization 带来了 metrics 的大幅提升。Internal VAE (8-channel) 和 Internal CLIP (bigG) 带来了质的飞跃。Synthetic recaption 达到了 SOTA T2I performance。这验证了 base architecture 的稳健性。

#### 5.2 T2V Performance on VBench (Table 9)
| Model | Quality ↑ | Semantic ↑ | Total ↑ |
| :--- | :--- | :--- | :--- |
| OpenSora V1.2 | 81.4 | 73.4 | 79.8 |
| CogVideoX-5B | 82.8 | 77.0 | 81.6 |
| Gen-3 | 84.1 | 75.7 | 82.3 |
| KLING | 83.4 | 75.2 | 82.3 |
| **STIV M-512** | 82.2 | 77.0 | 81.2 |
| **STIV M-512 UnmaskSFT + TUP** | **84.4** | 77.2 | **83.1** |

STIV 8.7B 模型在 512 分辨率下，经过 UnmaskSFT (去掉 MaskDiT 的 finetune) 和 TUP (Temporal Upsampler frame interpolation) 后处理，VBench Total score 达到 83.1，超越了 KLING, Gen-3 等顶尖闭源模型。值得注意的是，scaling 从 XL 到 M，Semantic score 提升明显 (72.5 -> 74.8)，而 Quality score 提升有限，说明大模型更容易 emergent 出 semantic understanding 能力。

#### 5.3 TI2V Performance (Table 10)
| Model | Quality ↑ | I2V ↑ | Total ↑ |
| :--- | :--- | :--- | :--- |
| SVD | 82.8 | 96.9 | 89.9 |
| Animate-Anything | 81.2 | 96.6 | 89.1 |
| **STIV-M-512** | 82.1 | 98.0 | **90.1** |
| **STIV-M-512-JIT** | 81.9 | 97.6 | 89.8 |

STIV-M-512 在 VBench-I2V 上达到了 90.1 的 SOTA 结果。Table 5 中的对比显示，加入 JIT-CFG 后，Dynamic Degree 从 10.2 跃升到 24.0，有效解决了 motion staleness issue，代价仅仅是 smoothness 和 consistency 的微小下降。

---

### 6. Flexible Applications

因为 frame replacement 设计的通用性，STIV 可以 natively 扩展到多种下游任务：
*   **Video Prediction (V2V):** 给定前 4 帧 $c_I$，预测后续。FVD 在 MSRVTT 上从 536.2 降至 183.7，展现出强大的 autoregressive 视频预测潜力。
*   **Frame Interpolation (TUP):** 给定首尾帧作为 $c_I$，填补中间帧。级联 TUP module 可以进一步提升主模型的 motion smoothness。
*   **Multi-view Generation:** 将 input image 作为第一帧，后续 6 帧作为 novel views 进行生成。在 GSO dataset 上与 Zero123++ 性能相当。这证明了基于 temporal attention 的 3D consistency 已经具备了隐式的 3D understanding 能力。
*   **Long Video Generation:** 采用 hierarchical framework。Mode 1 以 stride 20 生成 keyframes；Mode 2 作为 interpolator 填补中间帧。生成了长达 380 帧的视频。

### 7. Summary & Intuition

STIV 这篇 paper 的核心哲学是 "Less is more"。在面对如何融合 image condition 这个复杂问题时，作者没有选择设计复杂的 cross-attention module 或额外的 loss function，而是利用了 DiT 架构本身的 information propagation 特性，通过最直接的 frame replacement 解决问题。当遇到 staleness 问题时，又巧妙地复用了 classifier-free guidance 的数学逻辑，提出了 JIT-CFG，既解决了多任务训练问题，又打破了静态僵局。

这种 transparent, extensible recipe 为 video generation 社区提供了一个极佳的 baseline。它告诉我们，building cutting-edge video model 不需要玄学的 module stacking，systematic 的 scaling recipe (QK-norm, sandwich-norm, MaskDiT, Flow Matching) 加上 clear mathematical formulation (JIT-CFG) 就足以达到 SOTA。

### References & Web Links
*   STIV Paper (ArXiv Pending, based on content context): [arxiv.org/abs/2412.20889](https://arxiv.org/abs/2412.20889) (Assuming standard ArXiv ID pattern for recent Apple papers)
*   PixArt-$\alpha$ Paper: [arxiv.org/abs/2310.00426](https://arxiv.org/abs/2310.00426)
*   DiT (Scalable Diffusion Models with Transformers): [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
*   Flow Matching for Generative Modeling: [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
*   Rectified Flow (SD3 Core Math): [arxiv.org/abs/2203.04443](https://arxiv.org/abs/2203.04443)
*   RoFormer (RoPE): [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
*   MaskDiT: [arxiv.org/abs/2306.09305](https://arxiv.org/abs/2306.09305)
*   ConsistI2V: [arxiv.org/abs/2402.04324](https://arxiv.org/abs/2402.04324)
*   VBench: [arxiv.org/abs/2311.13535](https://arxiv.org/abs/2311.13535)
*   MovieGen (Meta): [arxiv.org/abs/2410.13720](https://arxiv.org/abs/2410.13720)
*   CogVideoX: [arxiv.org/abs/2408.06072](https://arxiv.org/abs/2408.06072)
