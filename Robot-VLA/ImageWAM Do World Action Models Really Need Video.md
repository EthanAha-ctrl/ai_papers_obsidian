---
source_pdf: ImageWAM Do World Action Models Really Need Video.pdf
paper_sha256: 0bf354364a18a2e504b46dcefd895ea923a63f04bc2f4ac12d9e8ea57ea80e2a
processed_at: '2026-08-19T12:14:55-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话重新讲一遍。

---

## 这篇 paper 在说什么

之前做 robot policy 有一派叫 WAM（World Action Model），思路是：让机器人先在脑子里 "放一段未来视频" 想象一下任务怎么完成，再根据这段想象去决定动作。这个思路很直觉——人类做事情好像也是先想清楚再动手。

但放视频很贵，也容易出错。作者就想：机器人真需要放一整段视频吗？还是只要想清楚 "任务做完后画面变成什么样" 就够了？

这就引出了 ImageWAM：用 image editing 模型替代 video generation 模型。

---

## 为什么 editing 比 video 更合适

你想想做 robot manipulation 的本质是什么。给定当前画面和一句指令 "把碗放到炉子上"，机器人需要理解什么？

- 哪个是碗（要抓的 object）
- 哪个是炉子（target receptacle）
- 碗现在在哪、应该挪到哪

这些信息在 "before" 和 "after" 两张图里就够了。中间手怎么移动、相机抖不抖、背景里灯泡闪不闪，对下一步动作没什么用。

Image editing 模型训练目标就是 "给定 source image + instruction，画出 modified image"。它天生就在学：
- 哪里要改
- 改成什么样
- 指令里的每个词对应画面里的什么 region

这跟 robot manipulation 的认知结构完全对得上。

Video generation 多出来一堆麻烦：要保证 16 帧时序连贯、背景不动、物体物理合理……这些 capacity 花在跟 action 无关的事情上。更要命的是，多帧想象一旦错一点，后面帧越错越离谱，action expert 就被带沟里了。

---

## 关键 trick：不 decode 编辑后的图

这是最聪明的地方。

image editing 模型内部是个 diffusion transformer，denoising 过程中每一层都产生 K 和 V 的 cache。作者发现：这些 cache 在 denoising 中途就已经 encode 了 "哪里要改、改成什么样" 的信息，根本不用等到最后 decode 出像素。

所以推理的时候只跑一次 forward 把 KV cache 取出来，直接喂给 action expert。不生成图，不 decode pixel，只偷中间的 "thinking"。

训练的时候故意在不同的 denoising timestep 上 sample，让 action expert 学会从 editing 过程的不同阶段都能读懂信息。推理时固定一个 timestep，跑一次就完。

---

## 为什么不直接用 VLM 当 backbone

VLM（像 OpenVLA、π0 那种）只做 "understanding"——看出来画面里有什么、指令说什么。它没有 "generation" 这一层，所以它没有 "想象未来状态变化" 的能力。

ImageWAM 的论点是：editing model 是 understanding + generation 的 sweet spot。它既有 VLM 的语义理解（instruction 怎么对应到 visual region），又有 generation 的 transformation reasoning（这个 region 应该变成什么）。

但又不能用 unified understanding-and-generation model（像 UniVLA、BagelVLA），因为 understanding 想要 abstract semantic，generation 想要 fine-grained spatial detail，塞进一个 shared transformer 会打架。ImageWAM 的做法：VLM 部分冻结，只训 diffusion editing branch 和 action expert。各管各的。

---

## 实验结果的人话版

**性能**：在 RoboTwin 上 93.38%，比所有 VLA baseline 高，跟 video-WAM 持平。注意 ImageWAM 没用任何 embodied pretraining，baseline 里 π0、π0.5、Motus 都用了大量 robot data 预训练。

**鲁棒性**：LIBERO-Plus 是专门测 perturbation 的 benchmark。ImageWAM 83.1%，FastWAM 只有 51.5%。差距巨大。尤其 camera perturbation 上 ImageWAM 80.8%，FastWAM 16.4%——video model 对视角变化敏感得离谱，editing model 没这个问题。

**效率**：FLOPs 降到 video-WAM 的 1/6，latency 降到 1/4。加 torch.compile 和 static graph 后能压到 69ms，大概 14Hz，接近实时控制。

**真实机器人**：双臂叠碗、折毛巾、开抽屉放笔、挂杯子四个任务，平均 84.5%，比 π0.5 高 12 个点。折毛巾提升最大，因为 deformable object 需要理解 "形状怎么变"，editing prior 特别擅长。

---

## 一句话总结

机器人不需要在脑子里放电影，它只需要知道 "任务做完后画面变成什么样"。Image editing model 天生就在做这件事，把它的中间 thinking 偷出来给 action head 用就够了——更准、更快、更稳。

未来 image editing model 继续变强（FLUX.3、NextStep-2 这种），robot policy 跟着免费升级。这是比 video-WAM 更 scalable 的路线。

---

# ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?

## 核心论点与 Motivation

Andrej 你好，这篇 paper 来自 SJTU + Tencent Robotics X + Tsinghua + EIT + Zhongguancun Academy，project lead 是 Wenyao Zhang。作者抛出一个非常 sharp 的问题：**World Action Models (WAMs) 真的需要 video generation 吗？**他们论证 image editing 才是与 manipulation policy 更匹配的 generative prior。

我先把三个 coupled limitations 列出来，帮 build intuition：

| Video-WAM 的痛点 | 直觉解释 |
|---|---|
| Dense multi-frame future tokens | 推理时必须 decode H+1 帧的 spatio-temporal latent，token 数量爆炸 |
| Action-irrelevant capacity | Video 模型被迫预测 background、camera motion、temporal smoothness，这些和 next action 几乎无关 |
| Long-horizon imagination errors | 多帧 rollout 时小 contact event 的误差会累积，最终误导 action expert |

这三个问题可以归约为一个核心矛盾：**video generation 的 objective 函数和 robot policy 的 objective 函数在频域上不对齐**——video 要求高频时空一致，policy 只关心 low-frequency 的 "scene state change"。Image editing 恰好是 low-frequency 的：它只关心一个 source frame 到一个 target frame 的 transformation，这正好对应 robot manipulation 的 "把当前状态变到任务完成状态"。

作者的三个 editing advantages：
1. **Instruction-to-change alignment**：editing 预训练直接把 language instruction 耦合到 visual modification 上。
2. **Easier goal/change proxy**：editing 只建模 current → target 的 difference，不建模完整 trajectory。
3. **Compact inference path**：editing 的中间 KV cache 已经 encode 了 transformation意图，无需 decode image。

Intuition：image editing model 的 attention pattern 本来就是"找哪里要改、改成什么样"，这和 manipulation 中"找哪个 object、怎么 grasp"的认知结构是 isomorphic 的。Video model 的 attention 则分散在时序一致性上。

Project page: <https://zhangwenyao1.github.io/ImageWAM/>  
Code: <https://github.com/yuyangalin/ImageWAM>

---

## Problem Formulation：从 Video-WAM 到 Image-WAM

设时间步 $t$，robot 接收 observation $o_t$ 和 language instruction $l$，需要预测 action chunk：

$$\mathbf{a}_{t:t+H} = (a_t, a_{t+1}, \ldots, a_{t+H}) \tag{1}$$

- $H$：action horizon（论文中 LIBERO/RoboTwin 都用 $H=16$）
- $\mathbf{a}_{t:t+H}$：从 $t$ 到 $t+H$ 的 action 序列，每个 $a_k$ 通常是 end-effector 的 7-DoF 或 14-DoF（双臂）

Policy objective：

$$\pi_\theta(\mathbf{a}_{t:t+H} \mid o_t, l) \tag{2}$$

Video-WAM 的范式：

$$(o_t, l) \rightarrow \hat{o}_{t+1:t+H+1} \rightarrow \mathbf{a}_{t:t+H} \tag{3}$$

这里 $\hat{o}_{t+1:t+H+1}$ 是 $H+1$ 帧的 future video，作为 intermediate world context。问题是这 $H+1$ 帧 latent token 在 DiT 内部要做 self-attention，复杂度 $O((H+1)^2 \cdot N_{spatial})$，再加上每帧的 VAE decode cost。

ImageWAM 的范式：

$$(o_t, l) \rightarrow \hat{o}_{\text{edit}} \equiv \hat{o}_{t+H+1} \rightarrow \mathbf{a}_{t:t+H} \tag{4}$$

关键转变：**只预测 endpoint frame，并且 inference 时不 decode 这个 frame**，只用它的中间 KV cache 作为 context。$\hat{o}_{\text{edit}}$ 是 single source-conditioned frame，summarize 任务指定的 visual transformation。

---

## Architecture 详解：MoT + Action Expert

ImageWAM 用了三种 image editing backbone：OmniGen2、Ovis-U1、FLUX.2。统一架构是 **Mixture of Transformers (MoT)**——把 self-attention 扩展成 joint self-attention，覆盖四类 token：

```
[ Language context tokens | Visual condition tokens | Visual prediction tokens (noisy) | Action tokens ]
```

Attention mask 设计：
- **Action tokens → 其他 tokens**：单向 attend，让 action 能看到所有 context
- **Noisy visual tokens → context tokens**：只能看 context，保持 context tokens clean
- 这避免了 action 的 noisy gradient 反向污染 visual context（典型的 decoupling trick）

### KV Cache 提取

训练时随机采样 editing denoising timestep $\tau$，在 $\tau$ 处运行 editing branch 一次 forward，对每个 transformer layer $\ell$ 收集 KV cache：

$$\mathcal{C}_{\text{edit}}^\tau = \{(K_\ell^\tau, V_\ell^\tau)\}_{\ell=1}^{L} = f_{\text{edit}}^\tau(o_t, l) \tag{5}$$

- $L$：transformer layer 数量
- $\tau$：image editing 的 denoising timestep（diffusion 的离散 step index）
- $K_\ell^\tau, V_\ell^\tau$：第 $\ell$ 层在 timestep $\tau$ 时的 key 和 value projection
- $f_{\text{edit}}^\tau$：editing branch 在 timestep $\tau$ 的 forward function

**关键 intuition**：这个 cache 是在 visual latent 已经和 task instruction 通过 editing backbone 交互**之后**产生的，所以它 encode 了 "instruction-conditioned visual transformation"。我们不需要等到最后一层 decode 出 image，中间层的 KV 已经携带了 transformation 的 semantic。

这里和 FastWAM 的 prefix-only inference 思路有亲缘关系——FastWAM 在 inference 时也只用 current context 的 KV cache，不 denoise future video tokens。ImageWAM 把这个 idea 推到极致：连 future video latent 都不实例化，只用单次 editing forward 的 cache。

参考 FastWAM: <https://arxiv.org/abs/2603.16666>

### 三个 backbone 的具体配置

| Variant | LLM Backbone | Diffusion Decoder | Action DiT Size | Init Strategy |
|---|---|---|---|---|
| OmniGen2 | Qwen2.5-VL-3B | OmniGen2 DiT | ~760M | Copy + interpolate from image editing model |
| Ovis-U1 | Qwen3-1.7B | ~1.2B MMDiT | ~1.1B | Similar to FLUX init |
| FLUX.2 4B | Qwen3-4B | FLUX.2 double+single stream | ~642M | Lower layers from double-stream, higher from single-stream |
| FLUX.2 9B | Qwen3-8B | FLUX.2 9B | ~952M | Same as 4B |

OmniGen2 variant 用了较小的 DiT hidden dim (1024) 但保持 attention hidden dim (2520)，这是一个 asymmetric 设计——action token 数量少，hidden dim 不用太大，但 attention capacity 要够。

---

## Training Objectives：双 Flow Matching

ImageWAM 同时优化两个 flow matching loss。我详细解释变量。

### 1. Image Editing Flow Matching Loss $\mathcal{L}_{\text{img}}$

设 target future observation $o_{t+H+1}$，它的 VAE latent：

$$z_{t+H+1}^* = E_{\text{vae}}(o_{t+H+1})$$

- $E_{\text{vae}}$：VAE encoder，把 RGB image 压成 latent
- $z_{t+H+1}^*$：target frame 的 clean latent

采样 image noise $\epsilon_z \sim \mathcal{N}(0, I)$ 和 image flow time $r \in (0, 1)$，构造 interpolated latent：

$$z_r = (1-r) z_{t+H+1}^* + r \epsilon_z \tag{6}$$

- $r$：image flow time，$r=0$ 时 $z_r = z^*$（clean），$r=1$ 时 $z_r = \epsilon_z$（pure noise）
- 这是 Rectified Flow / Flow Matching 的标准 interpolation，参考 Lipman et al. 2023

Diffusion image branch 预测 velocity field $u_\phi$：

$$\mathcal{L}_{\text{img}} = \mathbb{E}_{z^*, \epsilon_z, r}\left[\left\| u_\phi(z_r, r \mid o_t, l) - (\epsilon_z - z_{t+H+1}^*) \right\|_2^2\right] \tag{7}$$

- $u_\phi$：image branch 的 velocity predictor，参数 $\phi$
- 目标项 $(\epsilon_z - z_{t+H+1}^*)$：从 noise 指向 clean 的 velocity 向量
- $\|\cdot\|_2^2$：L2 squared norm

**注意**：论文 Eq.(7) 里写了 $z_{t+K}^*$，应该是 $z_{t+H+1}^*$ 的 typo（K 是 H 的 typo，看上下文 K 没有定义）。

直觉：这个 loss 让 editing branch 学会 "从 current observation $o_t$ 和 instruction $l$ 出发，预测 $H+1$ 帧后的 future frame"。但训练完后 inference 时我们不 decode 这个 frame，只取中间 KV cache。

### 2. Action Flow Matching Loss $\mathcal{L}_{\text{act}}$

设 expert action chunk $\mathbf{a}_{t:t+H}^*$（来自 demonstration），action noise $\epsilon_a \sim \mathcal{N}(0, I)$，action flow time $s \in (0, 1)$：

$$\mathbf{a}_s = (1-s) \mathbf{a}_{t:t+H}^* + s \epsilon_a \tag{8}$$

- $s$：action flow time，独立于 image flow time $r$
- $\mathbf{a}_s$：interpolated action sample

Action expert 预测 velocity field $v_\theta$：

$$\mathcal{L}_{\text{act}} = \mathbb{E}_{\mathbf{a}^*, \epsilon_a, s, \tau}\left[\left\| v_\theta(\mathbf{a}_s, s \mid o_t, l, \mathcal{C}_{\text{edit}}^\tau) - (\epsilon_a - \mathbf{a}_{t:t+H}^*) \right\|_2^2\right] \tag{9}$$

- $v_\theta$：action expert 的 velocity predictor，参数 $\theta$
- $s$：action flow-matching time
- $\tau$：image editing denoising timestep（用于提取 cache $\mathcal{C}_{\text{edit}}^\tau$）
- 目标项 $(\epsilon_a - \mathbf{a}_{t:t+H}^*)$：从 noise 指向 expert action 的 velocity

**关键设计**：训练时 sample $\tau$（在多个 denoising step 上均匀采样），让 action expert 暴露于 editing 过程不同阶段的 cache。这相当于 data augmentation——action expert 不能只依赖某一个特定 denoising stage 的 cache，必须 robust 到 editing 的整个 trajectory。这个 trick 在 diffusion distillation 里也常见（比如 Consistency Models 的 sample timestep）。

Joint optimization：

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{img}}$$

两个 loss 一起 backward，但作者特别提到：用 **action-head weight-copy initialization** 防止 visual model 在训练早期被 action 的 noisy gradient 干扰。具体做法是把 image editing model 的 weight copy+interpolate 到 Action DiT，再加 projection layer 支持 action input/output。

### Frozen vs. Trainable 分工

| Component | Status | 原因 |
|---|---|---|
| VLM (Qwen2.5-VL / Qwen3) | Frozen | 提供稳定 language-vision conditioning |
| Multimodal understanding modules | Frozen | 避免 understanding 被 action gradient 污染 |
| Diffusion image generation branch | Trainable | 要学 future frame prediction + 产出有用 cache |
| Action expert (Action DiT) | Trainable | 学 action velocity field |

这呼应了 ablation Q2 的论证：unified understanding-and-generation models 把两件事塞进一个 shared transformer，会互相干扰——understanding 想要 high-level semantic abstraction，generation 想要 fine-grained spatial detail，深层特征需求不同。ImageWAM 用 frozen VLM + trainable diffusion 的 decouple 方式避开了这个问题。

参考 Representation Alignment for Generation (Yu et al. ICLR 2025): <https://arxiv.org/abs/2402.06267>

---

## Efficient Inference：1-step Editing Forward

推理时不跑完整 denoising trajectory，固定一个 editing timestep $\tau^\star$，只做一次 forward：

$$\mathcal{C}_{\text{edit}}^{\tau^\star} = f_{\text{edit}}^{\tau^\star}(o_t, l) \tag{10}$$

然后 action expert 在这个 cache 上 denoise action samples：

$$\hat{\mathbf{a}}_{t:t+H} \sim p_\theta(\mathbf{a}_{t:t+H} \mid o_t, l, \mathcal{C}_{\text{edit}}^{\tau^\star}) \tag{11}$$

对比 video-WAM：
- Video-WAM：每帧都要 denoise（典型 20-50 步）+ decode 多帧 latent → pixel
- ImageWAM：1 次 editing forward + action expert denoise（论文中用 3 步 action denoising）

这里有一个 subtle 的点：训练时 sample $\tau$，inference 时固定 $\tau^\star$。这本质上是把 multi-step editing denoising 蒸馏到 single-step cache extraction。$\tau^\star$ 的选择是个 hyperparameter，论文没明说具体值，但从 FastWAM 的经验看应该是偏中后期的 timestep（信息已经从 noise 中 emerge 出 transformation semantic 的阶段）。

---

## Experiments 详解

### Setup

- **不需要 policy pretraining (P.T.)**：这是和很多 VLA baseline 的关键区别。π0、π0.5、Motus、LingBot-VA 都用了大量 embodied data 预训练。ImageWAM 只在 downstream benchmark demos 上训练。
- **Benchmarks**：LIBERO (4 suites × 10 tasks × 50 demos), LIBERO-Plus (robustness perturbations), RoboTwin 2.0 (50+ bimanual tasks, 27.5k trajectories), Real-world dual-arm (4 tasks × 100 demos)
- **Hardware**：8× NVIDIA H20 GPU, BF16, AdamW, LR 1e-4, warmup cosine scheduler

### RoboTwin 2.0 (Table 1)

| Method | P.T. | Clean | Rand. | Avg. |
|---|---|---|---|---|
| π0 | ✓ | 65.92 | 58.40 | 62.16 |
| π0.5 | ✓ | 82.74 | 76.76 | 79.75 |
| Motus | ✓ | 88.66 | 87.02 | 87.80 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| FastWAM | ✗ | 91.88 | 91.78 | 91.83 |
| **ImageWAM** | ✗ | **93.20** | **93.56** | **93.38** |

注意 ImageWAM 在 randomized setting 上比 clean 还高 0.36%，这是非常 robust 的信号——editing prior 让 policy 关注 task-relevant change 而不是 overfit 到固定 visual configuration。而 π0 在 random setting 上掉了 7.5 个点，典型的 distribution shift 退化。

### LIBERO (Table 2)

ImageWAM 平均 98.4%，competitive with FastWAM (97.6%)、Motus (97.7%)、LingBot-VA (98.5%)。在 Long suite 上 98.4%，明显超过 π0 (85.2%) 和 OpenVLA (53.7%)。

### LIBERO-Plus (Table 3) —— 关键 robustness 实验

| Method | Camera | Robot | Lang. | Light | Bg. | Noise | Layout | Avg. |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| FastWAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| **ImageWAM (FLUX.2 4B)** | **80.8** | 50.3 | **91.4** | **98.1** | **85.5** | **93.8** | **80.5** | **83.1** |

ImageWAM 在 7 个 perturbation 维度上 6 个第一（Robot 输给 Ovis-U1 variant 的 58.4）。最 dramatic 的是 Camera perturbation：FastWAM 只有 16.4%，ImageWAM 80.8%。这说明 video generation 对 camera视角变化非常 sensitive（因为 video model 学的是特定视角的 dynamics），而 editing prior 对 viewpoint 更 robust（因为 editing 本身就要处理 source image 的各种视角）。

LIBERO-Plus paper: <https://arxiv.org/abs/2510.13626>

### Real-World (Table 4)

| Method | T1 (Stack Bowls) | T2 (Fold Towel) | T3 (Open Drawer) | T4 (Hang Cup) | Avg. |
|---|---|---|---|---|---|
| π0 | 57 | 58 | 54 | 54 | 55.8 |
| π0.5 | 83 | 77 | 74 | 55 | 72.3 |
| FastWAM | 88 | 75 | 77 | 76 | 79.0 |
| **ImageWAM** | **94** | **84** | **78** | **82** | **84.5** |

T2 (Fold Towel) 提升最大（+9 over FastWAM）——deformable object manipulation 需要 reasoning about task-relevant visual changes，editing prior 正好擅长这个（编辑任务里很多是 deform 变换）。T3 (Open Drawer) 上两个 WAM-style 方法都大幅超过 π0，说明 world-action reasoning 帮助处理 visual occlusion。

### Efficiency (Table 5)

| Method | Latency | TFLOPs | Intermediate |
|---|---|---|---|
| FastWAM-IDM | 1081 ms | 63.65 | Video |
| FastWAM (1 step) | 302 ms | 13.21 | Cache |
| **ImageWAM** | **263 ms** | **9.72** | Cache |

Latency 降到 video-WAM 的 1/4，FLOPs 降到 1/6。这里 IDM = Inverse Dynamics Model（用 future video decode 后再过 IDM 预测 action），是 video-WAM 的标准做法。

### Efficiency Optimization (Table 11)

加 prefix-only attention + torch.compile + static CUDA graph 后：

| Variant | Latency | Speedup |
|---|---|---|
| FastWAM (1× vid. denoise) | 302 ms | 1.00× |
| ImageWAM (1× vid. denoise) | 263 ms | 1.15× |
| ImageWAM (prefix only) + action loop compile + image prefill compile + action static graph | **69 ms** | **4.38×** |

69ms latency 对应 ~14 Hz control frequency，已经接近实时。这主要得益于 action token 数量少，DiT 的 parallel efficiency 在小 token 数下 suboptimal，compile 后能 squeeze 出 3× speedup。

---

## Ablations

### Q1: 不同 editing backbone

OmniGen2 (71.8%), Ovis-U1 (71.2%), FLUX.2 4B (83.1%) 在 LIBERO-Plus 上都有效。说明 ImageWAM 不依赖特定 editing model，更强的 editing backbone 直接转化为更强的 policy robustness。这是个 very scalable 的 signal——未来 editing model 进步，robot policy 跟着受益。

### Q2: 为什么不用 unified understanding-and-generation models

对比 UniVLA (95.5% LIBERO) 和 BagelVLA (with/without keyframe prediction)。ImageWAM 在 non-keyframe future prediction setting 下超过这些 unified model。论证：understanding 想要 semantic abstraction，generation 想要 spatial detail，shared deep layers 会冲突。

### Q3: Backbone scaling

FLUX.2 4B → 9B，LIBERO-Plus 从 83.1% → 85.21%。改进主要在 Robot (+8.4), Language (+3.8), Background (+2.5), Layout (+2.6)。但 Camera (-1.0), Light (+5.7), Noise (-0.5) 非单调。说明 scaling 帮 instruction-conditioned robustness，但对 pixel-level perturbation 帮助有限——这可能因为 bigger model 更容易 memorize training distribution 的 texture。

---

## Attention Visualization 与 Failure Analysis

Figure 4 的 attention map 显示 ImageWAM 的 attention 集中在 task-relevant change regions（manipulated objects, target receptacles, contact areas），suppressing irrelevant background。这印证了 editing cache 的 "change-centric" 特性。

Figure 5 展示 video-WAM 的 failure case：imagined future frames 在 task-relevant object 附近有 distorted geometry 和 inconsistent spatial layout，这些 artifacts 直接误导 action expert。ImageWAM 因为不 instantiate dense future tokens，避开了 artifact 累积问题。

这让我想到 video generation model 在 fine-grained physical interaction 上的已知 failure mode——它们擅长 "looks like 抓住杯子" 但不擅长 "精确的 contact point + force direction"。Image editing 反而更聚焦于 "before/after 状态差异"，对中间过程不 hallucinate。

---

## 相关联想与 broader context

### 1. 与 Diffusion Policy / Flow Matching 的关系

ImageWAM 的 action expert 本质是 Diffusion Policy (Chi et al. RSS 2023) 的 flow matching 变体。区别在于 conditioning：Diffusion Policy 用 CLIP/DINO visual encoder 出的 feature，ImageWAM 用 editing branch 的 KV cache。后者多了 "instruction-conditioned transformation reasoning" 这一层。

Diffusion Policy: <https://arxiv.org/abs/2303.04137>  
Flow Matching: <https://arxiv.org/abs/2210.02747>

### 2. 与 π0 / π0.5 的关系

π0 用 PaliTema VLM + flow matching action expert，π0.5 加了 open-world generalization。它们的 action expert 也是 flow matching，但 visual context 来自 VLM 的 last hidden state，没有 explicit "future imagination" 步骤。ImageWAM 的 editing cache 可以看作一种 "implicit future imagination"——比 π0 的 pure perception 多了一层 transformation reasoning，比 video-WAM 的 explicit video 少了一层 dense token decoding。

π0: <https://arxiv.org/abs/2410.24164>  
π0.5: <https://arxiv.org/abs/2504.05498>

### 3. 与 World Models (Dreamer 系列) 的关系

传统 world model (Dreamer, Ha & Schmidhuber) 在 latent space rollout，然后从 rollout 的 latent state 预测 action 和 reward。ImageWAM 的 editing cache 类似一个 "single-step latent rollout"——只 rollout 一步到 target state，不做 multi-step rollout。这避开了 multi-step latent rollout 的 error accumulation 问题，是 Dreamer 思路的极简版。

Dreamer: <https://arxiv.org/abs/1910.01312>

### 4. 与 VLA (Vision-Language-Action) 演进的关系

VLA 从 RT-1/RT-2 的 token化 action，到 OpenVLA 的 VLM + action token，再到 π0 的 flow matching action。ImageWAM 可以看作 VLA 的下一代：**VLA + Generative Visual Prior**。区别在于 visual backbone 从 "understanding-only" (CLIP, DINO, SigLIP) 升级到 "understanding + generation" (editing model)。这呼应了最近 "generative models as generalist vision learners" 的趋势（参考 Gabeur et al. 2026, ref [33]）。

### 5. 与 BagelVLA / UniVLA / WorldVLA 的对比

这些是 unified understanding-and-generation VLA。BagelVLA 用 interleaved vision-language-action generation，UniVLA 用 task-centric latent actions，WorldVLA 是 autoregressive action world model。ImageWAM 的关键差异：**decoupled understanding (frozen VLM) + generation (trainable editing branch)**，避免 unified model 的 capacity interference。

### 6. KV Cache 作为 Representation 的思路

把 diffusion model 的中间 KV cache 作为 downstream task 的 representation，这个思路在 NLP 里有先例（GPT 的 KV cache 用于 retrieval），在 vision 里比较新。最近的工作如 REPAST、Representation Alignment for Generation 都在探索 diffusion transformer 的内部 representation 质量。ImageWAM 把这个 idea 应用到 robotics——editing branch 的 KV cache 是 "instruction-conditioned, change-aware" 的 representation，比纯 perception feature 更 action-relevant。

Representation Alignment for Generation: <https://arxiv.org/abs/2402.06267>

### 7. Single-Step Editing Inference 的理论依据

为什么单次 editing forward 就够？直觉：editing model 在 denoising 过程中，前期 timestep 处理 global structure（layout、object identity），后期 timestep 处理 local detail（texture、lighting）。Action prediction 主要需要 global structure level 的 transformation 信息，所以早期-中期 timestep 的 cache 已经足够。这和 Diffusion Models 的 frequency interpretation 一致（Rissanen et al., "Generative Modelling with Inverse Heat Dissipation"）。

### 8. 与 Cosmos / Genie 等 World Foundation Models 的关系

NVIDIA Cosmos 3 (ref [83]) 和 Genie Envisioner (ref [80]) 是 large-scale world foundation models。它们走 video generation 路线。ImageWAM 的论证暗示：对 robot policy 这种 action-centric 任务，可能不需要那么 heavy 的 world model——editing prior 就够。这其实是对 world model 路线的一个 gentle push-back：**world model 的 expressive power 和 policy 的实际需求之间存在 gap**。

Cosmos: <https://arxiv.org/abs/2606.02800>

---

## Training Hyperparameters 速查

| Parameter | Value |
|---|---|
| GPUs | 8× NVIDIA H20 |
| Distributed | DeepSpeed ZeRO-1 (FLUX.2 9B 用 ZeRO-2) |
| Precision | BF16 |
| Optimizer | AdamW (betas 0.9, 0.95) |
| LR | 1e-4 |
| Weight Decay | 1e-2 |
| LR Scheduler | Warmup Cosine |
| Warmup Steps | 0.05 × total |
| Min LR | 0.01 × lr |
| Gradient Clipping | 1.0 |

LIBERO: 2 views, 224×448, 16-frame future, 16-action chunk, 10 epochs, ~18 hours  
RoboTwin: 3 views (wrist-h + vertical concat), 288×256, 16/16, 5 epochs, ~5 days  
Real-world: same as RoboTwin preprocess, 10 epochs, ~18 hours

---

## 总结与 Intuition

ImageWAM 的核心 contribution 是一个 conceptual reframing：**robot manipulation policy 需要的不是 future video simulation，而是 instruction-grounded visual transformation reasoning**。Image editing model 恰好是这个 reasoning 的 native prior。

技术上的三个关键 design choice：
1. **Single endpoint frame prediction** 而非 multi-frame video——避开了 dense token 和 temporal error accumulation
2. **KV cache 作为 action context** 而非 decode image——避开了 pixel-space decoding cost
3. **Frozen VLM + trainable editing branch** 的 decoupling——避开了 understanding/generation interference

结果：FLOPs 1/6、latency 1/4、性能匹配或超过 video-WAM，在 LIBERO-Plus 这种 robustness benchmark 上大幅领先（83.1% vs FastWAM 51.5%）。

对这个方向的 broader implication：robot policy 的 visual backbone 选择，可能不需要追求 "更大的 world model"，而是要找 "objective function 与 action prediction 更对齐的 generative prior"。Image editing 是一个 surprisingly good fit。未来如果 image editing model 继续进步（比如 FLUX.3、NextStep-2），robot policy 可以直接受益，这是一个非常 scalable 的路线。

最后，作者提到的 future direction 是把 editing prior 推广到更广泛的 image model 应用——"language-vision interaction priors in editing models drive our model's effectiveness"。这暗示 editing model 可能是一种被低估的 general-purpose visual reasoning backbone，值得在更多 embodied AI 任务上探索。

参考 paper 列表（精选关键 reference）：
- FastWAM: <https://arxiv.org/abs/2603.16666>
- OmniGen2: <https://arxiv.org/abs/2506.18871>
- Ovis-U1: <https://arxiv.org/abs/2506.23044>
- FLUX.2: <https://bfl.ai/blog/flux-2>
- LIBERO: <https://arxiv.org/abs/2311.12948>
- LIBERO-Plus: <https://arxiv.org/abs/2510.13626>
- RoboTwin 2.0: <https://arxiv.org/abs/2506.18088>
- OpenVLA: <https://arxiv.org/abs/2406.09246>
- π0: <https://arxiv.org/abs/2410.24164>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- Flow Matching: <https://arxiv.org/abs/2210.02747>
