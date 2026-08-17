---
source_pdf: OmniHuman-1 Rethinking the Scaling-Up of One-Stage.pdf
paper_sha256: 44684d1de7fd66e03d3704516818e0370f87c075faa88000f16b1f3dfb0ce8a3
processed_at: '2026-08-05T23:19:12-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OmniHuman 用人话讲

## 一句话总结

ByteDance 这帮人发现：**做 audio-driven 人像动画，数据越 filter 越少，最后剩不到 10%，根本 scale 不动。他们的解法是别只用 audio 一个 condition，把 text、audio、pose 三个 condition 混着训，让每个 condition 吃自己能吃的 data，这样 18.7K hours 的 data 全都能用上。**

就这么简单。其他所有的 architecture trick、training schedule 都是服务于这个核心 idea 的。

---

## 为什么要这么搞？先讲清楚 pain point

### 现有方法的困境

你做 audio-driven talking head，比如 EchoMimic、Loopy、CyberHost、Hallo 这些。audio 跟什么相关？**主要跟嘴形、表情相关**。跟 body gesture、background、camera motion 基本没关系。

那 training data 怎么准备？你得 filter：
- 要 lipsync 准的，否则 model 学坏
- 要 pose 可见的，否则没法监督 body
- 要 front-facing 的，否则 audio 和 motion 对不上
- 要 static background 的，否则 model 把背景 motion 也算到 audio 头上

filter 完了，**90% 的 data 没了**。你说我有 100K hours 视频，最后能用的就 10K hours。SOTA method 的 paper 里自己都说 retention rate 不到 10%。

这就 dead 了 — 你再怎么 scale data，filter 完都是那么多，scaling law 不成立。

### 对比 general video generation

Sora、HunyuanVideo、CogVideoX 为什么能 scale？因为 text-video pairs 几乎不用 filter，text 描述什么 video 就是什么，10M clips 直接灌进去训。**Human animation 享受不到这个红利**。

Reference: 
- Sora: https://openai.com/research/video-generation-models-as-world-simulators  
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- CogVideoX: https://arxiv.org/abs/2408.06072

---

## OmniHuman 的 key insight

**别死磕 audio 一个 condition，混着训**。

具体讲：
- Text condition: 几乎不用 filter，18.7K hours 全能用
- Audio condition: 要 lipsync filter，大概剩 13%（2.4K hours）
- Pose condition: 要 pose visibility filter，比 audio 还少

**Strong condition 需要 stricter filter，weak condition 可以用更 dirty 的 data**。那我就让 weak condition 吃那些 strong condition 吃不了的 data。

这就好像你开个 restaurant，高档菜要用 A5 和牛（filter 严格，量少），但你也可以卖和牛汉堡（filter 宽松，量多），还可以卖牛骨汤（filter 更宽松）。最后所有牛肉部位都用上了，不浪费。

---

## 架构怎么搞？

这部分其实没什么花活，就是 standard MMDiT (Multimodal Diffusion Transformer，就是 SD3、PixArt 那套) 加上几个 condition injection module。

### Audio 怎么进去

wav2vec 提取 multi-scale features → MLP 压缩到 hidden size → 每帧的 audio feature 跟相邻帧拼接 → 通过 cross-attention 注入 MMDiT 每个 block。

公式就是标准 cross-attention:

$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

变量解释：
- $Q$ 是 noisy latent 的 query
- $K, V$ 是 audio tokens 的 key 和 value
- $d_k$ 是 key 的 dimension，用来 scaling 防止 dot product 过大
- 上标 $T$ 是 transpose

这个跟 Loopy、EMO、Hallo 做法基本一样，audio-driven 的标准操作。

### Pose 怎么进去

Pose guider (借用 AnimateAnyone 的设计) 编码 skeleton map → pixel-aligned features → 跟相邻帧拼接 → **直接 channel concatenate 到 noisy latent 上**，不是 cross-attention。

为什么 pose 用 concat 不用 cross-attention？因为 pose 是 spatial-aligned 的，每个 pixel 对应一个 pose 位置，concat 保留了 spatial alignment，cross-attention 会 blur 掉这个 alignment。

### Text 怎么进去

直接用 MMDiT 原本的 text branch，跟 SD3、PixArt-Alpha 一样。Text 跟 image 一起作为 context。

### Reference image 怎么进去 — 这是 paper 里最 elegant 的 trick

传统做法 (AnimateAnyone, EMO, Loopy) 是搞个 **reference network** — 把整个 diffusion backbone 复制一份，trainable，让 reference image 通过这个 copy，然后通过 self-attention 跟主 network 交互。**参数量直接 double**。

OmniHuman 的做法：**复用主 backbone**。

具体操作：
1. Reference image latents 和 noisy video latents 都 flatten 成 token sequence
2. **Pack 在一起**，一起 feed 进 DiT
3. 通过 self-attention 自然交互

关键的 position encoding trick — 用 3D RoPE (Rotary Position Embedding):

$$\text{RoPE}(q_i, r_j) = \text{RoPE}\left(q_i^{(t,h,w)}, r_j^{(0,h',w')}\right)$$

变量解释：
- $q_i$ 是 video token 的 query，position 是 $(t, h, w)$ — $t$ 是 temporal index（第几帧），$h, w$ 是 spatial index
- $r_j$ 是 reference token，position 是 $(0, h', w')$ — **temporal component 被置零**
- 上标括号里的 $t, h, w$ 表示在三个维度上的 position

**为什么 temporal component 置零？** 因为 reference image 是 "timeless" 的，它不属于任何一帧，但又应该能被所有帧 access 到。把 temporal position 置零，相当于把 reference 放在 "time origin" 这个特殊位置，所有 video frame 跟它做 self-attention 时，relative temporal distance 都是 frame index 本身，保持一致。

这个 trick 灵感估计来自 NaViT 的 patch n' pack 思路 — 不同 resolution / aspect ratio 的 token 可以 pack 一起。

Reference: 
- NaViT: https://arxiv.org/abs/2307.06304
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864

### Long video 怎么搞

用 motion frames — 上一 segment 最后 5 帧 feature 拼到下一 segment 的 noise tokens 前面。跟 Open-Sora、Loopy 一样。

---

## Training Strategy — 这才是 paper 的核心

### 两条 Principle

**Principle 1**: Stronger conditioned task 可以 leverage weaker conditioned task 的 data 来 scale up。

逻辑很直白：
- Audio task 要 filter lipsync，剩 13% data
- Text task 几乎不 filter，能吃 100% data
- 那 audio task 之前扔掉的 87% data，在 text task 里全都能用

所以 Stage 1 先用 text + image → video 训，吃满 18.7K hours，让 model 学到 general human motion prior。

**Principle 2**: Condition 越强，training ratio 越低。

这条稍微 subtle 一点。直觉是：**strong condition 会 "suppress" weak condition 的 learning**。

想象 pose 和 audio 同时出现，pose 已经告诉 model 手该怎么动了，model 就懒得学 audio 跟手的关系了。为了让 audio 也能学到 meaningful 的 motion generation，pose 的 training ratio 必须低一些。

具体比例:
$$R_{\text{text}} : R_{\text{audio}} : R_{\text{pose}} \approx 90\% : 50\% : 25\%$$

从弱到强，ratio 大约 halved。

### 三阶段训练

**Stage 1**: Text + Image → Video  
- 用 100% data (18.7K hours)  
- 目标：学 general motion prior  
- 从 pretrained T2V model 开始 fine-tune

**Stage 2**: Text + Image + Audio → Video  
- Drop pose condition  
- 用大约 50% 的 data（audio 部分需要 lipsync filter）  
- 目标：引入 audio-motion correlation

**Stage 3**: Text + Image + Audio + Pose → Video  
- 全 condition  
- Pose ratio 最低（25%）  
- 目标：用 pose 作为 auxiliary guidance，同时保持 audio-driven 能力

### 顺序为什么是 Text → Audio → Pose 而不是 Text → Pose → Audio？

这是 ablation study 的关键发现。

Paper Table 1 下半部分对比了四种配置：

| Method | Conditions (training order) | Sync-C ↑ | FID ↓ | FVD ↓ | HKC ↑ |
|--------|---------------------------|----------|-------|-------|-------|
| IA | Image + Audio (no pose) | 4.987 | 36.01 | 43.74 | 0.882 |
| IPA | Image + Pose + Audio (pose 先) | **2.788** | 38.98 | 44.70 | **0.822** |
| IAP, A<P | Image + Audio + Pose (audio 先, pose ratio 高) | 4.201 | 38.73 | 44.63 | 0.869 |
| IAP, A>P | Image + Audio + Pose (audio 先, pose ratio 低) | 4.934 | 36.66 | 43.36 | 0.886 |

IPA (pose 先训) 灾难性下降，Sync-C 从 4.987 跌到 2.788。**Intuition**: 如果 pose 先训，model 学到 "反正 pose 会告诉我怎么动"，之后再加 audio 时，model 不愿意学 audio-motion mapping，因为 pose 已经够用了。这就像学生习惯了抄答案，再让他自己解题就不愿意了。

IAP A<P (pose ratio 比 audio 高) 也变差，因为 pose 又开始 dominate。IAP A>P (audio ratio 高, pose ratio 低) 才能保持 audio-driven 能力同时享受 pose 的 auxiliary guidance。

### Figure 5 的可视化分析

Paper 里还有个很直观的可视化。用 gradient curves 画 hand motion trajectories：
- IA model (纯 audio): 手部动作夸张、不协调
- IAP model (audio + pose): 手部动作 decouple from audio，更自然

这说明 pose 的作用其实不是直接控制手，而是**让 model 学会 "audio 不一定要 drive 手"**，从而避免 audio 在弱相关区域 over-generate。

---

## Inference 的细节

### Selective CFG

OmniHuman 发现 **CFG 只对 weak condition (audio, text) 有用，对 strong condition (pose) 不用**。

$$\hat{\epsilon} = \epsilon_\theta(z_t, c_{\text{full}}) + s \cdot \left[\epsilon_\theta(z_t, c_{\text{full}}) - \epsilon_\theta(z_t, c_{\text{drop(audio,text)}})\right]$$

变量解释：
- $\epsilon_\theta$ 是 denoising network 的 prediction
- $z_t$ 是 noisy latent at timestep $t$
- $c_{\text{full}}$ 是 full condition (audio + text + pose + image)
- $c_{\text{drop(audio,text)}}$ 是把 audio 和 text 置 null (保留 pose 和 image)
- $s = 6.5$ 是 CFG scale
- 下标 $t$ 表示 diffusion timestep

**Intuition**: CFG 是为了 amplify weak signal。Pose 已经是 pixel-aligned 强信号，再 CFG 反而会 over-sharpen，破坏精确性。Audio 是弱信号，需要 CFG 来 push model 更 "听" audio。

### Condition activation rules

- Audio-driven inference: 激活 audio + text（text 由 captioning model 生成）
- Pose-driven inference: 激活 pose + text，关掉 audio
- Hybrid driving: 全激活

**Subtle rule**: 当激活某个 condition，所有比它更弱的 condition 也激活。因为 training 时 weaker condition 总是出现，inference 时保持 consistency。

---

## 实验数据看效果

### Ablation 1: Text data 的作用 (Principle 1)

在 CelebV-HQ 数据集上:

| Text data ratio | Sync-C ↑ | FID ↓ | FVD ↓ | HKV ↑ | HKC ↑ |
|----------------|----------|-------|-------|-------|-------|
| 0% | 4.299 | 39.80 | 47.86 | 35.82 | 0.871 |
| 25% | 3.311 | 37.95 | 47.04 | 40.39 | 0.877 |
| 50% | 3.696 | 36.26 | 46.22 | 40.69 | 0.872 |
| 100% | **4.987** | **36.01** | **43.74** | **43.54** | **0.882** |

加 text data 不只改善 video quality (FVD 从 47.86 降到 43.74)，**还改善了 lipsync (Sync-C 4.299 → 4.987) 和 hand motion richness (HKV 35.82 → 43.54)**。

这说明 shared backbone 学到 general motion prior 后，能 transfer 到 audio-driven task 上。general video generation 能力是 audio-driven 的 foundation。

**Counterintuitive 发现**: IQA 和 ASE (aesthetic scores) 反而下降。Paper 解释：数据少时 model 倾向生成 training set distribution 的高质量人脸；数据多时 model 学会 follow input image 的 style。**这其实是 generalization 更好的表现** — model 不再 "作弊" 用 training distribution 的 prior，而是真正遵循 input。

### Ablation 2: Training order 和 ratio (Principle 2)

(前面已经 table 贴过了)

**结论**：
1. Pose 不能先训 (IPA 灾难)
2. Audio ratio 要高于 pose ratio (A>P)
3. IAP with A>P 能保持 audio-driven 能力同时支持 hybrid driving

### 与 SOTA 对比

**Portrait animation (Table 2)**:

| Method | Sync-C ↑ | FID ↓ | FVD ↓ |
|--------|----------|-------|-------|
| SadTalker | 3.843 | 36.648 | 171.848 |
| Hallo | 4.130 | 35.961 | 53.992 |
| VExpress | 3.547 | 65.098 | 117.868 |
| EchoMimic | 3.136 | 35.373 | 54.715 |
| Loopy | 4.849 | 33.204 | 49.153 |
| Hallo-3 | 3.933 | 38.481 | 42.125 |
| **OmniHuman** | **5.199** | **31.435** | 46.393 |

OmniHuman 几乎全面领先。注意它是一个 unified model 支持 face/portrait/half-body/full-body，而 baselines 都是 specialized。

**Body animation (Table 3)** — 这是 OmniHuman 最大优势所在:

| Method | Sync-C ↑ | FID ↓ | FVD ↓ | HKV ↑ | HKC ↑ |
|--------|----------|-------|-------|-------|-------|
| DiffTED | 0.926 | 95.455 | 58.871 | - | 0.769 |
| DiffGest + MimicMotion | 0.496 | 58.953 | 66.785 | 23.409 | 0.833 |
| CyberHost | 6.627 | 32.972 | 28.003 | 24.733 | 0.884 |
| **OmniHuman** | **7.443** | **31.641** | **27.031** | **47.561** | **0.898** |

**HKV (hand motion richness) 47.561，是 CyberHost 的 2 倍**。这是 omni-conditions training 最大的 benefit — 通过大量 text-conditioned data 学到丰富的 body/gesture motion prior，transfer 到 audio-driven 上。

**Pose-driven comparison (Table 4)**:

| Method | FID ↓ | FVD ↓ | AKD ↓ |
|--------|-------|-------|-------|
| DisCo | 57.12 | 64.52 | 9.313 |
| AnimateAnyone | 26.87 | 37.67 | 5.747 |
| MimicMotion | 23.43 | 22.97 | 8.536 |
| CyberHost | 20.04 | 7.72 | 3.123 |
| **OmniHuman** | **19.504** | **7.32** | **2.136** |

OmniHuman 在 pose-driven 上也是 SOTA，AKD (action keypoint distance) 最低。**一个 unified model 打败所有 specialized models**。

---

## Build Intuition: 这个 paradigm 为什么 work？

### 1. Data diversity > data purity

传统思路：filter data 到很 pure，让 model 学到 clean audio-motion mapping。  
OmniHuman 思路：用 mixed conditions 让 model 见过更多 motion patterns，即使 audio-motion mapping 不那么 clean。

这跟 LLM 的思路一致 — GPT 训练数据也不是 clean 的，但 diversity 让 model 学到 general language understanding，再 fine-tune 到 specific task。

### 2. Multi-task learning as data augmentation

不同 condition 共享 backbone，互相 regularize。Text task 学到 general motion prior，audio task 学到 audio-motion mapping，pose task 学到 precise pose control。**Shared representation 互相 benefit**。

### 3. Curriculum learning 的 hidden benefit

从 weak condition 到 strong condition 的 training order，本质是 curriculum learning：
- Stage 1 (text only): 学 general video generation，foundation
- Stage 2 (add audio): 学 audio-motion mapping，medium difficulty
- Stage 3 (add pose): 学 precise control，最难

如果反过来先学最 precise 的 pose control，model 会 overfit 到 "照着 pose 抄"，之后学 audio 就学不进去。

### 4. Condition hierarchy 的启示

OmniHuman 揭示了一个 general principle: **在 multi-condition model 里，condition 之间有 hierarchy，strong condition 会 suppress weak condition 的 learning**。要 balance 这个 effect，需要:
- Training order: weak → strong (curriculum)
- Training ratio: weak high, strong low
- Inference CFG: weak condition 用 CFG amplify, strong condition 不用

这个 principle 在其他 multi-modal task 里应该也适用，比如 text-to-image 里 text 和 image prompt 的关系，video generation 里 text 和 reference image 的关系。

### 5. Scaling law 的重新理解

LLM scaling law 说 "data 越多越好"，但 human animation 里直接加 data 不 work (因为 filter 限制)。OmniHuman 重新定义了 "effective data" — 不只看单一 task 的 data 量，看 **所有 condition 联合起来能利用的 data 量**。

$$D_{\text{effective}} = \sum_i D_i$$

其中 $D_i$ 是 condition $i$ 能用的 data，通过 mixing 让 $D_{\text{effective}}$ 最大化。

---

## Limitations 和我 (Karpathy) 的思考

Paper 自己说:
1. Audio-motion 弱相关导致 uncoordinated movements 还是会出现
2. Object interaction 不够 realistic
3. High CFG scale 导致 overfitting

我的延伸思考：

**A. Condition 设计还可以更丰富**  
现在 condition 是 text/audio/pose，但 real human motion 还受 emotion、intention、social context drive。未来可以加:
- Emotion condition (happy/sad/angry)
- Intention condition (explaining/arguing/entertaining)  
- Style condition (formal/casual/theatrical)

**B. Audio-motion weak correlation 是 fundamental**  
Audio 跟 body motion 本来就没多少 correlation，这是物理事实。靠 data scaling 能缓解，但解不了。可能需要:
- Contrastive learning 来强制学 audio-motion fine-grained correspondence
- Implicit motion reasoning (model 先理解 audio 内容语义，再 infer 该怎么动)
- Causal modeling (audio → emotion → motion)

**C. Reference-as-token 的 trick 可以推广**  
RoPE temporal zero 这个 idea 可以推广到任何 "timeless" condition，比如:
- Style image (作为 style reference)
- Identity image (作为 identity anchor)
- Audio embedding (作为 voice identity)

Reference: 
- EMO2 (end-effector guided): https://arxiv.org/abs/2501.10687
- VASA-1: https://arxiv.org/abs/2404.10667

**D. Multi-task hierarchy 的 generalization**  
OmniHuman 的 Principle 2 (strong condition ratio 低) 可能是 multi-task learning 的 general principle。在 LLM multi-task fine-tuning 里也观察到类似现象 — "easy task 会 dominate shared representation"，常见解法是 loss weighting。OmniHuman 用的是 training ratio，本质类似。

---

## 最后总结

OmniHuman 的 contribution 用人话讲就是三条:

1. **Data scaling 问题转化为 data diversity 问题** — 别 filter 到死，mix conditions 让所有 data 都有用
2. **Multi-condition training 需要 hierarchy-aware curriculum** — 先弱后强，强弱配比要调
3. **一个 unified model 打败所有 specialized models** — audio-driven、pose-driven、hybrid-driven 全支持，face 到 full-body 全 cover

最让我觉得 elegant 的还是 Principle 2 — **strong condition 不能先训**。这个 finding 揭示了 multi-condition learning 里一个 subtle 的 dependency，可能在其他领域也适用。

Paper 链接: https://omnihuman-lab.github.io/

总之，ByteDance 这篇 paper 最大的价值不是 architecture innovation (架构很 standard)，而是**从 data scaling 角度重新思考 training paradigm**。在所有人都纠结怎么 filter data 时，他们选择不 filter，混着训。这个思路对整个 generative AI 领域都有启发 — 当 data scarce 时，multi-task + diversity 可能比 single-task + purity 更 effective。

---

# OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models

## 1. Core Problem: 为什么 Human Animation 难以 scale up?

这篇 paper 来自 ByteDance, 核心问题非常清晰. 当前 end-to-end audio-driven human animation 领域 (比如 talking head, talking body) 虽然发展很快, 但是遇到了一个 fundamental bottleneck:

**Data filtering 陷阱**: 为了让 audio-conditioned model 稳定训练, 必须做严格的 data filtering, 包括:
- Lipsync accuracy filtering
- Pose stability filtering  
- Front-facing perspective filtering
- Static background filtering

结果是 SOTA methods (比如 Loopy, CyberHost) 最终 **只保留了 less than 10% 的原始数据**. 这意味着即使你有海量 video data, 经过 filter 之后真正用于训练的很少, scaling 变得 cost-ineffective.

**Audio 本身的 limitation**: audio 主要 correlate facial expressions, 但是对 body pose, background motion, camera movement 几乎没有 correlation. 这就是为什么 audio-driven 方法必须 filter 掉 non-correlated 的部分.

这跟 general video generation (Sora, HunyuanVideo, CogVideoX) 形成鲜明对比, 后者用 O(100M) clips 的 video-text pairs 直接 scale up, 而 human animation 几千小时的数据都难以 scale.

Reference: 
- Sora technical report: https://openai.com/research/video-generation-models-as-world-simulators
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- CogVideoX: https://arxiv.org/abs/2408.06072

---

## 2. Key Insight: Omni-Conditions Training

作者的核心 insight 是: **与其坚持 single-condition (audio) 然后 filter 数据, 不如 mix 多个 motion-related conditions (text, audio, pose), 让不同 condition 利用各自能 cover 的数据**.

这个 insight 背后的逻辑是:

| Condition | Motion correlation strength | Data filtering requirement |
|-----------|----------------------------|---------------------------|
| Text | Weak (最弱) | 几乎不需 filter |
| Audio | Medium | 需要 lipsync filter |
| Pose | Strong (最强) | 需要 pose visibility filter |

Stronger condition 需要 stricter filtering, weaker condition 可以 leverage 更 wide 的 data distribution.

**这个 idea 本质上是把 multi-task learning 当作 data augmentation 的一种形式**: 那些 audio-driven task 不能用的数据 (因为 lipsync 不准), 可以在 text-conditioned task 里用, 这样就 rescue 了大量 "wasted" data.

---

## 3. Architecture: DiT-based Multi-Condition Model

OmniHuman 基于 MMDiT (Multimodal Diffusion Transformer) [Peebles & Xie, 2023; Esser et al., 2024], 从一个 pretrained text-to-video model 出发, 逐步加入 motion-related conditions.

### 3.1 Driving Conditions Injection

**Audio condition**:
- 使用 wav2vec [Baevski et al., 2020] 提取 multi-scale acoustic features
- 通过 MLP 压缩到 MMDiT hidden size, 同时 align framerate (25 fps)
- 每帧 audio features 与相邻 timestamps 的 audio features 拼接, 形成 audio tokens
- 通过 **framewise cross-attention** 注入到 MMDiT 的每个 block

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$ 来自 noisy latent, $K, V$ 来自 audio tokens, $d_k$ 是 key dimension.

**Pose condition**:
- 用 pose guider [AnimateAnyone, Hu et al., 2024] 编码 driving skeleton map sequence
- 得到 pixel-aligned pose features
- 与相邻帧拼接形成 frame-wise pose tokens
- **stack 在 channel dimension** 上, 与 noisy latent 一起 fed into model (这是 spatial-concat 方式, 不是 cross-attention)

**Text condition**:
- 直接用 MMDiT 原本的 text branch (image-text-to-video 的标准做法)

### 3.2 Appearance Conditioning - 优雅的 reference 设计

这是 paper 里一个 subtle 但很重要的 design. 传统方法 (比如 EMO, Loopy, CyberHost) 用 **reference network** — 一个完整的 diffusion backbone 的 trainable copy, 通过 self-attention 与主 network 交互. 这导致参数量翻倍, scalability 差.

OmniHuman 的做法是: **reuse 主 DiT backbone 来 encode reference image**.

具体做法:
- Reference image latents 和 noisy video latents 都 flatten 成 token sequence
- **Packed together** 同时 fed into DiT
- 通过 self-attention 让 reference tokens 与 video tokens 交互

**关键的 position encoding trick**: 修改 3D RoPE [Su et al., 2024], 对于 reference tokens, **temporal component 清零**, 而 video tokens 的 RoPE 保持不变.

这个设计的 intuition: reference image 是 "timeless" 的 (它在时间维度上没有 dynamics), 所以应该把它放在 "temporal position 0" 这个特殊位置, 让所有 video frame tokens 都能通过 self-attention 访问到它, 但它本身不参与 temporal dynamics.

数学上, 假设 3D RoPE 的 position 为 $(t, h, w)$ 对于 video token, 其中 $t$ 是 temporal index, $h, w$ 是 spatial index. 对于 reference token, position 变成 $(0, h_{\text{ref}}, w_{\text{ref}})$.

这样:
$$\text{RoPE}(q_i, r_j) = \text{RoPE}\left(q_i^{(t,h,w)}, r_j^{(0,h',w')}\right)$$

Reference tokens 在 temporal dimension 上被 "anchored" 到 0, 视觉上仍然保持 spatial structure.

**好处**: 不增加任何额外参数, 但充分利用 MMDiT backbone 的 modeling capacity.

### 3.3 Long video generation

为了支持 long video, 用 motion frames [Stypulkowski et al., 2024, Loopy] — 把上一 segment 最后 5 帧的 features 与 noise tokens 拼接, 用于下一 segment 的 generation, 保证 temporal coherence 和 identity consistency.

---

## 4. 两条核心 Training Principles

### Principle 1: Stronger conditioned tasks 可以 leverage weaker conditioned tasks 和对应的数据来 scale up training data

形式化表达: 如果 $D_{\text{audio}}$ 是 audio-conditioned data, $D_{\text{text}}$ 是 text-conditioned data, 则:
$$D_{\text{audio}} \subset D_{\text{text}}$$

因为 audio-conditioned data 的 filtering criteria 包含 text-conditioned 的 criteria. 那 13% 经过 lipsync + pose visibility filter 的数据是 $D_{\text{audio+pose}} \subset D_{\text{full}} = 18.7K$ hours.

这意味着 Stage 1 用 text+image-to-video 训练, 可以利用整个 18.7K hours; Stage 2 加入 audio, 用 subset; Stage 3 加入 pose, 用更小的 subset.

### Principle 2: The stronger the condition, the lower its training ratio should be

这个 principle 的 intuition: 当 audio 和 pose 同时存在时, model 倾向于 rely on pose (更强的 condition), 导致 audio 的 learning 效果被 suppressed. 为了让 audio 也能 learn 到 meaningful motion generation, 需要:
- 给 weaker condition (audio) 更高的 training ratio
- 给 stronger condition (pose) 更低的 training ratio

Paper 里给出具体比例:
- Text: T = 90%
- Audio: A = 50%
- Pose: P = 25%

也就是从弱到强, ratio 大约是 halved.

**为什么这个设计 build intuition?** 想象 gradient descent 的视角: model 的 capacity 是有限的, 如果一个 task 太容易 (strong condition), gradient 很快就收敛到 local optimum, 而且会 dominate shared representation. 通过降低 strong condition 的 ratio, 我们让 model 在 strong condition 上 "slow down", 给 weak condition 留出 learning capacity.

---

## 5. Three-Stage Training Pipeline

| Stage | Conditions active | Data ratio | Goal |
|-------|-------------------|-----------|------|
| Stage 1 | Text + Image | 100% data (18.7K hours) | 学习 general video generation, motion priors |
| Stage 2 | Text + Image + Audio | ~50% data | 引入 audio, drop only pose |
| Stage 3 | Text + Image + Audio + Pose | ~25% for pose | 引入 pose, 但 pose ratio 低 |

**为什么 Stage 1 不引入 audio/pose?** 因为 audio, pose 需要严格 filter, 会浪费数据. 用 text+image 这个 weak condition 可以 leverage 最大数据.

**为什么顺序是 Text → Audio → Pose 而不是 Text → Pose → Audio?**

这是 ablation study 的关键发现. Paper Table 1 下半部分对比:
- **IA** (image + audio, 无 pose): baseline
- **IPA** (image + pose + audio, pose 先于 audio): 各项 metric 全面下降
- **IAP** (image + audio + pose, audio 先于 pose): metrics 与 IA 相当, 还支持 hybrid driving

**Intuition**: 如果先引入 pose (strong condition), model 会 over-rely on pose, 之后再学 audio 时会 struggle, 因为 audio 提供的 motion signal 被 pose "掩盖" 了. 反过来, 先学 audio (相对弱), model 必须自己 generate motion, 学到的 motion prior 更 general, 然后再 fine-tune with pose 就容易.

Figure 5 用 gradient curves 可视化了 hand motion trajectories. IA model (只有 audio) 产生过于夸张的 hand movement, 因为 audio 和 hand motion 相关性弱, model 在 weak signal 下倾向于 "over-generate". IAP (hybrid) 通过 pose 的 guidance 把 hand motion "decouple" from audio, 让 hand movement 更 natural.

---

## 6. Inference Strategy

**CFG (Classifier-Free Guidance) 的 selective 使用**:

CFG scale = 6.5, 但是 **只对 audio 和 text 用 CFG, 不对 pose 用**.

$$\hat{\epsilon}_\theta = \epsilon_\theta(z_t, c_{\text{full}}) + s \cdot (\epsilon_\theta(z_t, c_{\text{full}}) - \epsilon_\theta(z_t, c_{\text{null}}))$$

其中 $c_{\text{full}}$ 包含 audio + text + pose, $c_{\text{null}}$ 把 audio 和 text 置 null (保留 pose 和 image), $s = 6.5$.

**Intuition**: audio 和 text 是 weak motion signal, 需要 CFG 来 amplify 它们的影响; pose 已经是 pixel-aligned 强信号, 不需要 amplify, 反而 CFG 会破坏它的精确性.

**Inference 时 condition 激活规则**:
- Audio-driven: 激活 audio + text (text 由 image captioning model 生成)
- Pose-driven: 激活 pose + text, 关闭 audio
- Hybrid driving: 全部激活

注意一个 subtle rule: **当一个 condition 激活时, 所有比它 motion-related influence 更弱的 condition 也激活**. 这是为了 consistency, 因为 model 训练时 weaker condition 总是出现.

---

## 7. Experimental Results 详解

### 7.1 数据规模

- 总数据: **18.7K hours** in-house human-related data
- 经过 lipsync + pose visibility filter: **13% (~2.4K hours)** 用于 audio 和 pose
- Training: 400 A100 GPUs, 每个 phase ~10 days
- Learning rate: $5 \times 10^{-5}$, AdamW, gradient clip 1.0, batch size 256, weight decay 0.01

### 7.2 Ablation: Principle 1 (Text data scaling)

Table 1 上半部分, 在 CelebV-HQ 上:

| Text data ratio | Sync-C ↑ | FID ↓ | FVD ↓ | HKV | HKC ↑ |
|----------------|---------|-------|-------|-----|-------|
| 0% | 4.299 | 39.80 | 47.86 | 35.82 | 0.871 |
| 25% | 3.311 | 37.95 | 47.04 | 40.39 | 0.877 |
| 50% | 3.696 | 36.26 | 46.22 | 40.69 | 0.872 |
| 100% | 4.987 | 36.01 | 43.74 | 43.54 | 0.882 |

可以看到 100% text data 时, FVD 从 47.86 降到 43.74, Sync-C 从 4.299 升到 4.987, HKV 从 35.82 升到 43.54. 加 text data 不仅改善 video quality, 还改善了 lipsync 和 hand motion richness!

**Counterintuitive finding**: IQA 和 ASE (aesthetic scores) 反而下降. Paper 解释: 这是 model 从 "training set distribution" 转向 "input image distribution" 的结果. 当数据少时, model 倾向于生成 high-quality training distribution 的人脸; 当数据多时, model 学会 follow input image 的 style (可能 input image 质量没那么高), 因此分数下降. 但这其实是 generalization 更好的表现.

### 7.3 Ablation: Principle 2 (Condition order + ratio)

Table 1 下半部分:

| Method | Sync-C ↑ | FID ↓ | FVD ↓ | HKV | HKC ↑ |
|--------|---------|-------|-------|-----|-------|
| IA (baseline) | 4.987 | 36.01 | 43.74 | 43.54 | 0.882 |
| IPA (pose first) | 2.788 | 38.98 | 44.70 | 45.44 | 0.822 |
| IAP, A<P | 4.201 | 38.73 | 44.63 | 40.99 | 0.869 |
| IAP, A>P | 4.934 | 36.66 | 43.36 | 39.39 | 0.886 |

IPA (pose 先训) 是灾难性的: Sync-C 从 4.987 跌到 2.788, HKC 从 0.882 跌到 0.822. 这证实了 Principle 2 — strong condition 不能先训.

IAP A>P (audio ratio > pose ratio) 接近 IA baseline, 而 IAP A<P 反而恶化. 这说明 pose ratio 必须比 audio 低.

### 7.4 与 SOTA 对比

**Portrait animation** (Table 2, CelebV-HQ):

| Method | IQA ↑ | ASE ↑ | Sync-C ↑ | FID ↓ | FVD ↓ |
|--------|-------|-------|----------|-------|-------|
| SadTalker | 2.953 | 1.812 | 3.843 | 36.648 | 171.848 |
| Hallo | 3.505 | 2.262 | 4.130 | 35.961 | 53.992 |
| VExpress | 2.946 | 1.901 | 3.547 | 65.098 | 117.868 |
| EchoMimic | 3.307 | 2.128 | 3.136 | 35.373 | 54.715 |
| Loopy | 3.780 | 2.492 | 4.849 | 33.204 | 49.153 |
| Hallo-3 | 3.451 | 2.257 | 3.933 | 38.481 | 42.125 |
| **OmniHuman** | **3.875** | **2.656** | **5.199** | **31.435** | 46.393 |

OmniHuman 在几乎所有 metric 上领先. 注意 FVD 是 46.393, 比 Hallo-3 (42.125) 略差, 但 Sync-C 大幅领先 (5.199 vs 3.933).

**Body animation** (Table 3):

| Method | IQA ↑ | Sync-C ↑ | FID ↓ | FVD ↓ | HKV | HKC ↑ |
|--------|-------|----------|-------|-------|-----|-------|
| DiffTED | 2.701 | 0.926 | 95.455 | 58.871 | - | 0.769 |
| DiffGest+MimicMotion | 4.041 | 0.496 | 58.953 | 66.785 | 23.409 | 0.833 |
| CyberHost | 3.990 | 6.627 | 32.972 | 28.003 | 24.733 | 0.884 |
| **OmniHuman** | **4.142** | **7.443** | **31.641** | **27.031** | **47.561** | **0.898** |

OmniHuman 的 HKV (hand motion richness) 是 47.561, 几乎是 CyberHost 的 2 倍! 这意味着 OmniHuman 生成的 hand motion 远比其他方法丰富, 这是 Omni-Conditions Training 带来的最大 benefit.

**Pose-driven comparison** (Table 4):

| Method | IQA ↑ | FID ↓ | FVD ↓ | AKD ↓ |
|--------|-------|-------|-------|-------|
| DisCo | 3.707 | 57.12 | 64.52 | 9.313 |
| AnimateAnyone | 3.843 | 26.87 | 37.67 | 5.747 |
| MimicMotion | 3.977 | 23.43 | 22.97 | 8.536 |
| CyberHost | 4.087 | 20.04 | 7.72 | 3.123 |
| **OmniHuman-1** | **4.111** | **19.504** | **7.32** | **2.136** |

OmniHuman 在 pose-driven 任务上也表现很好, AKD (Action Keypoint Distance) 2.136 最低. 注意: OmniHuman 是一个 unified model, 同时擅长 audio-driven 和 pose-driven, 而 baselines 都是专门为某一个 task 设计的.

---

## 8. Build Intuition: 为什么这个 paradigm 重要?

### 8.1 与 LLM scaling law 的类比

OmniHuman 的核心贡献其实是把 **scaling law 的思路引入 human animation**. LLM 能 scale 因为 text data 几乎无限; T2I/T2V 能 scale 因为 image/video-text pairs 容易收集. 但 human animation 一直 scale 不起来, 因为 audio-conditioned data 经过 filter 后只剩 10%.

OmniHuman 的 solution 是: **不要 constrain 在 single condition, 让 multiple conditions 各自利用自己的 data, 共同 feed 同一个 backbone**. 这本质上是把 data scarcity 问题通过 multi-task learning 转化为 data diversity 问题.

### 8.2 与 LoRA / Multi-task Fine-tuning 的类比

Omni-Conditions Training 有点像 multi-task fine-tuning, 但是关键区别是 task 之间有 **hierarchy** (weak → strong condition), 而 LoRA 等方法通常假设 task 是平行的. 这个 hierarchy 使得可以用 curriculum learning 的思路, 从 weak condition (大量 data) 开始, 逐步引入 strong condition (少量 data).

### 8.3 Reference conditioning 的 elegance

Reference network (AnimateAnyone, EMO 等) 是 industry 标准做法, 但参数翻倍. OmniHuman 的 reference-as-token 设计很 elegant, 它本质上把 reference image 当作 "时间维度为 0" 的特殊 video, 让 self-attention 自然处理. 这个 idea 可能来自 NaViT [Dehghani et al., 2024] 的 patch n' pack 思路 — 不同 resolution / aspect ratio 的 token 可以 pack 在一起.

### 8.4 CFG 的选择性使用

传统 CFG 对所有 condition 一起用. OmniHuman 发现 **weak condition 需要 CFG, strong condition 不需要**. 这其实揭示了 diffusion model 里 condition strength 和 CFG 的关系 — CFG 是为了 amplify weak signal, strong signal 已经 saturated 了, 再 amplify 会 overfit.

---

## 9. Limitations 和 Future Directions

Paper 自己提到的 limitations:
1. **Audio-motion 弱相关导致 uncoordinated / overly expressive movements** — audio 和 body motion 本来就没什么 correlation, 即使加了 pose 也只能缓解
2. **Object interaction 不够 realistic** — 这是 training data 不足导致的, 当 input image 偏离 training distribution 时, generation 不自然
3. **High CFG scale 导致 overfitting** — 为了 stability 用 6.5 的 CFG, 但这会让 result overfit to condition

我 (Karpathy) 角度的延伸思考:

- **Future: 加入 intention, style, intensity 等 richer conditions**. 现在 condition 是 audio/pose/text, 但 real human motion 还包含 emotion, intention, social context. 把这些 explicit 加入会让 motion 更 natural.
- **Self-supervised / contrastive learning on audio-motion pairs**. 现在 audio 和 motion 的 weak correlation 是 fundamental limitation. 可以用 contrastive learning 来强制 model 学习 audio-motion 之间的 fine-grained correspondence.
- **Diffusion Transformer 的 scaling 现在已经够强, 瓶颈在 data 而不是 model**. 这篇 paper 印证了这一点 — 通过 smart data strategy (omni-conditions), 一个 standard MMDiT 就能超越 specialized models. 未来 human animation 的 progress 可能更多来自 data curation strategies 而非 architecture innovations.

---

## 10. 总结: OmniHuman 的核心 contribution

1. **Omni-Conditions Training Strategy** — 用 multi-condition 来 scale data, 避免了 single-condition 的 filtering 陷阱
2. **Two training principles** — stronger condition 利用 weaker condition 的 data; stronger condition 用更低 ratio
3. **Reference-as-token design** — 用 RoPE temporal zero trick, 不增加参数实现 appearance conditioning
4. **Unified model** — 一个 model 同时支持 audio-driven, pose-driven, hybrid-driven, 支持 face/portrait/half-body/full-body
5. **18.7K hours 数据, 大幅超过之前 methods**

最让我 (Karpathy) 觉得 elegant 的地方是 Principle 2 — **stronger condition 不能先训**. 这个 finding 揭示了 multi-condition training 里一个 subtle 的 dependency: condition 之间不是独立的, strong condition 会 "suppress" weak condition 的 learning, 必须通过 curriculum (先弱后强) 和 ratio control 来 balance. 这个 insight 在 multi-task learning 领域应该有更广泛的 implication.

---

## References & Further Reading

- OmniHuman project page: https://omnihuman-lab.github.io/
- wav2vec 2.0: https://arxiv.org/abs/2006.11477
- AnimateAnyone: https://arxiv.org/abs/2311.17117
- Loopy: https://arxiv.org/abs/2409.02634
- CyberHost: https://arxiv.org/abs/2409.06680
- EMO: https://arxiv.org/abs/2402.17485
- Hallo3: https://arxiv.org/abs/2412.00733
- VASA-1: https://arxiv.org/abs/2404.10667
- MimicMotion: https://arxiv.org/abs/2406.19680
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- CogVideoX: https://arxiv.org/abs/2408.06072
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- SD3 / MMDiT (Esser et al.): https://arxiv.org/abs/2403.03206
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- NaViT (Patch n' Pack): https://arxiv.org/abs/2307.06304
- EchoMimic: https://arxiv.org/abs/2407.08136
- EMO2: https://arxiv.org/abs/2501.10687
- VLogger: https://arxiv.org/abs/2403.08764
- Panda-70M (data filtering criteria): https://arxiv.org/abs/2406.09310
- DiffTED: https://arxiv.org/abs/2403.18830
- AnimateDiff: https://arxiv.org/abs/2307.04725
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- V-Express: https://arxiv.org/abs/2406.02511
- Q-Align (VLM for IQA/ASE): https://arxiv.org/abs/2312.17090

OmniHuman 这个 paradigm 我觉得非常有启发性. 它不是单纯的 architecture innovation, 而是 **从 data scaling 角度重新思考 human animation 的 training paradigm**. 在 LLM 时代, 大家都意识到 data 是 bottleneck, 但在 human animation 这个 sub-field, 之前没人系统地解决这个 problem. ByteDance 这篇 paper 通过 multi-condition mixing + curriculum + ratio control 三板斧, 把 data efficiency 提升了一个数量级. 这个思路对其他 data-scarce 的 generative task (比如 music-driven dance, sketch-driven animation 等) 应该都有借鉴意义.
