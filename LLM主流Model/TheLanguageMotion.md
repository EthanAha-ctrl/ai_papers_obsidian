---
source_pdf: TheLanguageMotion.pdf
paper_sha256: b912628d29b0f191be9f09a5e254758aa7ec4f46284424f8044201e4f4c870b1
processed_at: '2026-08-12T15:10:45-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
把 3D 人体动作塞进一个 language model 里，跟文字、语音一起当同一种"语言"处理。嘴动、手比划、脸做表情、身体晃 —— 耦合的东西。全塞进一个 LM 里. 

Language model 只认 token，不认连续向量。所以第一步是把 motion "文字化"。

他们的做法是**把人切成四块**：

- **脸**（face）：FLAME 模型那 100 个表情参数 + 1 个 jaw joint
- **手**（hands）：30 个手指关节
- **上半身**（upper body）：13 个关节
- **下半身**（lower body）：9 个关节

每块单独训一个 [VQ-VAE](https://arxiv.org/abs/1711.00937)，把连续的旋转角度量化成离散 codebook index。这样一来，一段 5 秒的 motion 就变成一串 token，跟一句话一样。

**为什么不直接整体 VQ？** 因为脸的微表情和腿的大步走，时间尺度差太远。混在一起 codebook 会被大动作占满，小表情就糊了。分块量化是 [EMAGE](https://arxiv.org/abs/2401.00374) 和 [TalkSHOW](https://arxiv.org/abs/2305.01378) 已经验证过的 recipe，这篇 follow 了。

**为什么不用 HumanML3D 那套表示？** 因为那套是为 "walk forward"、"sit down" 这种 macro 动作设计的，重点在骨骼摆动，丢了扭转旋转。你要建模 expressive gesture——耸肩、扭手腕、撇嘴——H3D format 表达力不够。

语音用 [HuBERT](https://arxiv.org/abs/2106.07447) 量化成 token，50 fps。文字用 [SentencePiece](https://arxiv.org/abs/1808.06226) 切成 WordPiece，32K vocab。三个模态的 token 拼成一个 unified vocabulary $V = \{V_t, V_a, V_f, V_h, V_u, V_l\}$。

**巧妙的 trick**：motion token 在 vocab 里就用 `"<upper 8>"` 这种字符串表示。等于 text embedding table 自动扩展，不需要单独搞 motion embedding layer。这个 trick 来自 [MotionGPT](https://arxiv.org/abs/2306.14767)。

---

## 预训练：最聪明的部分

这是这篇 paper 最值得讲的地方。

常规做法是：你有多少 audio-motion 配对数据，就拿去训。问题是配对数据少得可怜。

他们的做法是：**预训练阶段完全不碰 audio-motion 配对**，只做两件 self-supervised 的事：

### 第一件：让模型学"身体零件之间的语法"

人开心的时候，脸在笑、手在张开、肩往后展——这些是**跨身体部位的 correlation**，universal，不依赖 speech。他们让模型玩一个游戏：给上半身，预测下半身；给脸，预测手；随机组合。

这叫 **spatial alignment**。

同时玩另一个游戏：随机 mask 掉一些帧，让模型预测——[MAE](https://arxiv.org/abs/2111.06377) 在 motion 上的版本。这叫 **temporal alignment**，让模型学时间动力学。

数据来源：BEATv2 + [AMASS](https://amass.is.tue.mpg.de/) 共 60 小时 motion，全是 unpaired 的，便宜。

### 第二件：让模型学"语音和文字是同一回事"

用 [LibriSpeech](https://www.openslr.org/12/) 1000 小时 audio-text 配对，做"听语音预测文字"这种 ASR-like 任务。模型从来没见过 audio-motion 配对，但学完这个，audio embedding 已经对齐到 text embedding space 了。

### 为什么这两件事加起来这么 work

因为你把 downstream 的 audio→motion 任务拆开了：

- audio→text 的 semantic bridge（LibriSpeech 教会了）
- text→motion 的 body language grammar（motion self-supervised 教会了）
- Flan-T5 自带的 text understanding（language model pre-training 教会了）

三件事在预训练阶段分别学到，downstream 只需要 fine-tune 把它们拼起来。所以 **1/32 的配对数据就能打过 EMAGE 用满数据**。这个 data efficiency gain 非常夸张。

Figure 5 那张图是 paper 的核心证据：横轴是训练数据量（1, 1/2, 1/4, ..., 1/32），纵轴是 FGD。full model 在 1/32 数据时 FGD ≈ 6.5，w/o pre-training ≈ 11，EMAGE ≈ 9.5。

---

## Post-training：Instruction Tuning

预训练完，模型已经懂了"motion 的语法"和"audio 的语义"。现在要教它"听指令干活"。

他们手写了几十个 instruction template，组合出一千多个 task variant。比如：

> "Based on [audio], generate a synchronized movement sequence involving both face, hands, upper and lower body." → `[face][hands][upper][lower]`

> "Create leg and foot movements that align with the intensity shifts in [audio]." → `[lower]`

> "What emotion is conveyed by the movements in [face][hands][upper][lower]?" → `[emotion]`

关键设计：**每个 body part 可以单独 prompt**。这 unlock 了一个 cool feature——

---

## Editable Gesture Generation：意外收获

你可以这样玩：

- 给 audio，prompt：生成上半身 gesture
- 给 text "walk forward"，prompt：生成下半身
- 把两段拼起来 = 一个边走边说话的人

[BEAT](https://arxiv.org/abs/2211.14303)、[EMAGE](https://arxiv.org/abs/2401.00374) 这种 speaker-dependent 模型做不到这个——它们的架构是 audio→full body 一条路走死。这里因为 part-level promptable，组合性天然出来。

游戏、VR 里你要 NPC 边走边聊，这就是刚需。

---

## Motion-to-Emotion：真正的新 task

之前的 motion model 都是 motion→caption（"一个人在走路"），没人做过 motion→emotion（"这个人很愤怒"）。

这篇因为做了 instruction tuning + emotion label 训练，能从 motion 反推情绪。Table 3 显示 [MotionGPT](https://arxiv.org/abs/2306.14767) 完全 fail（跟 random 差不多），这个 model 能 work。

为什么这件事重要？**心理健康评估、精神科诊断、HCI**——读懂身体语言是人类的基本能力，机器一直做不到。这是个有真实应用价值的 task。

---

## Limitations：他们说和我说的

### 论文自己承认的

**Discrete tokenization 有信息损失**。VQ codebook 是有限离散集合，fine-grained motion 会被 quantize 到最近 entry，导致 jitter 或 frozen pose。他们 future work 提到 continuous tokenization。

### 我觉得他们没充分讨论的

**[HuBERT](https://arxiv.org/abs/2106.07447) 的 prosody loss**。HuBERT 主要学 phoneme content，对韵律、语调、energy 这些 prosody 信息保留有限。但 co-speech gesture 的 beat alignment 强依赖韵律——重音、停顿、语调。这可能限制了 BC metric 的天花板。

[AudioLM](https://arxiv.org/abs/2209.03143) 的做法是 semantic tokenizer（HuBERT）+ acoustic tokenizer（SoundStream/EnCodec）两层并行，既保 content 又保 prosody。这篇只用了一层，可以扩展。

**50 fps audio vs 30 fps motion**。频率不匹配是 implicit 的，模型自己学。可以显式做 cross-attention 或 resample。

**60 小时 motion 数据是领域天花板**。BEATv2 + AMASS = 60h，对比 [LAION-5B](https://arxiv.org/abs/2210.08414) 的 image-text billion 级数据，motion 数据小了 4-5 个数量级。整个 motion generation 领域都卡在这。

**为什么不试更大的 LM**。220M Flan-T5-Base。上 [Flan-T5-XL](https://arxiv.org/abs/2210.11416) 3B、[UL2](https://arxiv.org/abs/2205.05131) 20B 会不会有 emergent ability？没做 scaling law 是遗憾。但 motion sequence 长度大（T 帧 × 4 parts），$O(L^2)$ attention 扩展到更长序列难。可以试 [Mamba](https://arxiv.org/abs/2312.00752)、[Longformer](https://arxiv.org/abs/2007.14072)。

**Eval metric 的根本问题**。Section 7.5 自己承认：text-to-motion 的标准 metric (FID, R-precision) 都和 H3D-Format 强耦合，不能公平评测 compositional representation。这是 community 级的系统性问题。

---

## 联想：这篇 paper 在大图景里的位置

几个趋势的 intersection：

1. **Multimodal LLM 扩展到时序连续模态**：[LLaVA](https://arxiv.org/abs/2304.08485) (image) → [VideoChat](https://arxiv.org/abs/2305.06355) (video) → [SpeechGPT](https://arxiv.org/abs/2305.13045) (audio) → 这篇 (motion+audio+text)。下一个 frontier 是 robot action tokens（[RT-2](https://arxiv.org/abs/2307.15818)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)）。

2. **Pre-training 在 data-scarce 领域的胜利**：[MAE](https://arxiv.org/abs/2111.06377) 在 vision、[HuBERT](https://arxiv.org/abs/2106.07447) 在 speech、[VQ-VAE](https://arxiv.org/abs/1711.00937) 在 audio 都是同样故事——大模型时代，好的 pre-training 让 downstream 用极少数据。Motion capture 成本高，这个 pattern 尤其重要。

3. **Generative pre-training as translation**：把 "predict modality Y from modality X" 当通用 pre-training pattern。[AudioLM](https://arxiv.org/abs/2209.03143) coarse-to-fine、[MUSE](https://arxiv.org/abs/2209.10752) masked text、[Flamingo](https://arxiv.org/abs/2204.14198) interleaved vision-text 都是同一种思路。

4. **Compositional representation 的胜利**：[EMAGE](https://arxiv.org/abs/2401.00374)、[TalkSHOW](https://arxiv.org/abs/2305.01378)、[NeuralDome](https://arxiv.org/abs/2303.18024) 一路做 part decomposition，这篇升级成 part-level vocabularies + LM-friendly tokens。

---

## 我的核心 takeaway

这篇 paper 最大的 insight 不是技术细节，是 **framing**：

**Human motion 是一种 language，face/hand/body 是它的 part-of-speech，audio 是它的 pronunciation，text 是它的书面形式。用 LM 统一建模，自然就该 work。**

这个 framing 一旦接受，所有 design choice 都很自然：motion 要 tokenize（变成 word）、要 VQ（变成 vocab）、要 part decomposition（变成 syntax）、要 translation pre-training（变成 bilingual alignment）、要 instruction tuning（变成 task following）。

Andrej 你自己常说 "everything is token, everything is transformer"——这篇 paper 是这个 philosophy 在 3D human motion 上的实证。它证明了 LM 范式在这个领域 work，且 data-efficient，且 unlock 新 task。下一代 continuous + prosody-aware + larger backbone 的 motion LM 应该会在这个基础上把质量推到 production-ready。

---

**Project page**: https://languageofmotion.github.io

**主要参考**:
- [Flan-T5](https://arxiv.org/abs/2210.11416) - backbone
- [EMAGE (CVPR 2024)](https://arxiv.org/abs/2401.00374) - compositional representation 来源
- [BEAT (ECCV 2022)](https://arxiv.org/abs/2211.14303) - benchmark dataset
- [MotionGPT (NeurIPS 2023)](https://arxiv.org/abs/2306.14767) - 之前的 motion LM 尝试
- [HuBERT](https://arxiv.org/abs/2106.07447) - audio tokenizer
- [AudioLM](https://arxiv.org/abs/2209.03143) - audio LM 的两层 tokenizer 思路
- [MAE](https://arxiv.org/abs/2111.06377) - masked prediction pre-training 范式
- [Flamingo](https://arxiv.org/abs/2204.14198) - multimodal few-shot pre-training
- [AMASS](https://amass.is.tue.mpg.de/) - large motion dataset
- [SpeechGPT](https://arxiv.org/abs/2305.13045) - audio+text LM 的先驱
- [MoMask (CVPR 2024)](https://arxiv.org/abs/2312.02660) - 下一代 continuous motion generation 方向

---

# The Language of Motion: 深度解析

Andrej，这篇 paper 我看完之后非常兴奋，它做的事情在 conceptual level 上很 elegant：把 3D human motion 的 verbal 和 non-verbal 两个 channel 塞进一个统一的 language model 框架里。这种"everything is token, everything is translation"的视角，本质上是在 follow LLM 时代"token is all you need"的范式。让我把它一层层拆开讲，过程中我会把相关的 prior work、intuition、limitations 都串起来。

---

## 1. Big Picture: 为什么这件事值得做

传统 human motion generation 领域被切成了几个孤岛：

- **Speech-to-motion (co-speech gesture)**：BEAT, EMAGE, TalkSHOW, CaMN, DiffStyleGesture, GestureDiffuClip 这一线工作。它们需要高质量 speech-motion 配对数据，speaker-dependent，迁移到新人成本极高。
- **Text-to-motion**：HumanML3D, T2M-GPT, MDM, MotionGPT, MoMask, MotionDiffuse, ReMoDiffuse 这一线。数据规模大但语义浅。
- **Motion understanding / captioning**：MotionGPT, MotionLLM。

每个孤岛只能用自己的 paired data，跨岛迁移成本极高。论文的核心 insight 是：**human communication 本身是 multimodal 的**，speech (verbal) + gesture/facial/body (non-verbal) 在人类表达里是天然耦合的，把它们拆开建模本身就丢掉了 prior。用 language model 把它们 unify，可以共享 cross-task 的 motion priors，可以借用 LLM 已经学好的 semantic understanding。

这种 motivation 在 [Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP-2](https://arxiv.org/abs/2301.12597)、[LLaVA](https://arxiv.org/abs/2304.08485)、[SpeechGPT](https://arxiv.org/abs/2305.13045)、[AudioLM](https://arxiv.org/abs/2209.03143) 这一脉 work 里都见过，但**第一次被系统性地用到 3D human motion 上**，且第一次把 audio 和 motion 这两种时序模态放进同一个 vocab。

---

## 2. Architecture Walkthrough

整体是 **modality-specific tokenizers + encoder-decoder LM** 的标准 multi-modal LM 形态，但有几个关键的 design choice。

### 2.1 Tokenization：分四个 body parts 这件事是关键

论文没用大家熟悉的 [HumanML3D representation](https://arxiv.org/abs/2205.00589)（H3D-Format），而是 follow [EMAGE](https://arxiv.org/abs/2401.00374) 的 compositional 思路，把 body 拆成四块：

| Body part | Joints | Dim (6D rotation + expr) | 维度 |
|---|---|---|---|
| Lower body $\mathbf{g}_l$ | 9 | $9 \times 6 = 54$ | $\mathbb{R}^{T \times 54}$ |
| Upper body $\mathbf{g}_u$ | 13 | $13 \times 6 = 78$ | $\mathbb{R}^{T \times 78}$ |
| Hands $\mathbf{g}_h$ | 30 | $30 \times 6 = 180$ | $\mathbb{R}^{T \times 180}$ |
| Face $\mathbf{g}_f$ | 1 joint + 100 expr | $6 + 100 = 106$ | $\mathbb{R}^{T \times 106}$ |

整体 motion space $G = \{\mathbf{g}_f, \mathbf{g}_h, \mathbf{g}_u, \mathbf{g}_l\}$。

**Intuition**：为什么拆开？两个原因：
1. **Decoupling statistics**：face 的 jitter 跟 locomotion 的低频轨迹在时间尺度上完全不同，混在一起 VQ 会让 codebook 浪费在大动作上而忽略细节。EMAGE 已经证明这点。
2. **Compositional alignment**：pre-training 阶段要做 "translate upper to lower" 这种 spatial alignment，必须 token 级别可分。

这一点和 [TalkSHOW](https://arxiv.org/abs/2305.01378) 把 body / hand / face 分开建模的 motivation 一致，但这里更进一步，把它做成独立 vocabularies，让 LM 学到的是**"part-level 的 motion grammar"**。

**为什么不选 H3D-Format**：论文给的理由是 H3D 偏 skeletal swing，丢掉了 twisting rotation，对 expressive body language 表达力不够。这是对的——H3D 是为 text-to-motion 设计的，那时大家只关注 "walk forward"、"sit down" 这种 macro motion，对 micro gesture（手腕抖动、肩部扭转）没建模需求。

### 2.2 VQ-VAE 量化

每个 body part 训练一个独立的 VQ-VAE，encoder 是 4-layer TCN：

$$\mathbf{z}^{1:T} = \mathcal{E}(\mathbf{g}^{1:T})$$

然后做 nearest neighbor 量化：

$$\mathbf{q}^t = \mathcal{Q}(\mathbf{z}^t) := \arg\min_{\mathbf{q}^k \in Q} \|\mathbf{z}^t - \mathbf{q}^k\|^2 \tag{1}$$

变量解释：
- $\mathbf{z}^t \in \mathbb{R}^d$：第 $t$ 帧经过 TCN encoder 后的 continuous latent
- $Q = \{\mathbf{q}^k\}_{k=1}^{K}$：codebook，每个 $\mathbf{q}^k$ 是一个 $d$ 维的离散 embedding
- $\mathbf{q}^t$：第 $t$ 帧对应的 codebook index（实际存的是 index，但公式里写的是 embedding 本身）
- $\|\cdot\|^2$：L2 范数平方

直觉上：每一帧 motion 被映射成一个 codebook entry，codebook size 决定表达粒度。这里四个 VQ-VAE 分别有 $K_f, K_h, K_u, K_l$ 个 codebook entries，论文没给具体数值，但参考 EMAGE 应该在 512 左右。

### 2.3 VQ-VAE Loss：multi-level 的 reconstruction

$$\begin{aligned}
\mathcal{L}_{total} = & \mathcal{L}_{rec}(\mathbf{g}, \hat{\mathbf{g}}) + \mathcal{L}_{vel}(\mathbf{g}', \hat{\mathbf{g}}') + \mathcal{L}_{acc}(\mathbf{g}'', \hat{\mathbf{g}}'') \\
& + \mathcal{L}_{mrec}(\mathbf{g}, \hat{\mathbf{g}}) + \mathcal{L}_{mvel}(\mathbf{g}', \hat{\mathbf{g}}') + \mathcal{L}_{macc}(\mathbf{g}'', \hat{\mathbf{g}}'') \\
& + \mathcal{L}_{comm}(\mathbf{g}, \mathbf{q})
\end{aligned} \tag{2}$$

变量解释：
- $\mathbf{g}$：原始 motion，$\hat{\mathbf{g}}$：重建 motion
- $\mathbf{g}', \hat{\mathbf{g}}'$：一阶差分（velocity）
- $\mathbf{g}'', \hat{\mathbf{g}}''$：二阶差分（acceleration）
- $\mathcal{L}_{rec}$：pose-level 重建；lower/upper/hands 用 **Geodesic loss**（旋转空间上的距离，比 L2 更适合 SO(3)），face 用 L2（因为 FLAME 表情参数是欧式空间）
- $\mathcal{L}_{mrec}$：mesh-level 重建，用 SMPL-X 顶点坐标算 L2
- $\mathcal{L}_{comm}$：codebook commitment loss，让 encoder 输出 commit 到 codebook

**Intuition**：为什么需要这么多 loss？因为 motion 重建有个经典问题——pose error 小但 motion jitter 大。加 velocity / acceleration loss 是为了让 tokenized representation 保留**时间一致性**。Mesh loss 是为了保证最终 mesh 渲染质量，因为 body part 坐标误差小不等于 mesh 看起来对（joint 误差会通过 forward kinematics 放大）。

这套 loss 设计可以追溯到 [TEMOS](https://arxiv.org/abs/2202.10479)、[T2M-GPT](https://arxiv.org/abs/2301.06052)、[MotionGPT](https://arxiv.org/abs/2306.14767) 等 motion VQ-VAE 工作，是一个相当标准的 recipe。

### 2.4 Speech tokenization：HuBERT 的选择

用 [HuBERT](https://arxiv.org/abs/2106.07447) 把连续 audio 离散化：
- 采样率 16 kHz
- HuBERT downsampling factor 320
- Token rate 50 fps

motion 是 30 fps，audio 是 50 fps，**频率不一致**。论文用这两个数没问题，但跨模态 alignment 需要模型自己学频率映射。这里我有点担心——一种 alternative 是用 30 fps 的 audio tokenizer，或者做 resample。50 fps 的好处是 audio 信息密度更高，bad case 是 sequence 长度变长。

**Reference**：[AudioLM](https://arxiv.org/abs/2209.03143)、[SoundStream](https://arxiv.org/abs/2107.03312)、[EnCodec](https://arxiv.org/abs/2210.13473) 都是类似的离散化思路。HuBERT 是 self-supervised speech model，它的 codebook 主要捕获 phoneme-level 信息，这对 semantic reasoning 有帮助但对韵律（prosody）可能不足——后面我会讲这个 limitation。

### 2.5 Text tokenization：标准 T5 setup

[SentencePiece](https://arxiv.org/abs/1808.06226) + [WordPiece](https://arxiv.org/abs/1508.07909)，32K vocab，从 T5 继承。这是为了让预训练 LM 的 text embedding 直接能用。

### 2.6 Unified multimodal vocabulary

最终 vocab：
$$V = \{V_t, V_a, V_f, V_h, V_u, V_l\}$$

每个 sub-vocab 都有 special boundary token（`<soa>` / `<eoa>` 等）。Motion token 用 `"<upper 8>"` 这种 string 形式嵌入到 text vocab——这点设计很巧妙，**等于让 LM 的 text embedding table 自动扩展**，不需要单独的 motion embedding layer。这是 [MotionGPT](https://arxiv.org/abs/2306.14767) 率先采用的 trick，这篇 follow 了。

---

## 3. Generative Pre-training：论文最 core 的贡献

这是这篇 paper 最有意思的部分。Pre-training 完全不碰 audio-motion pair，只做两种 self-supervised alignment。

### 3.1 Compositional body motion alignment

#### Spatial alignment

模板：

```
Task Prompts: Translate upper to lower body.
Conditions: Upper Body Tokens V_condition = {v_u^i ∈ V_u | i ∈ {sequence token index}}
Answer: Lower Body Tokens V_answer = {v_l^i ∈ V_l | i ∈ {sequence token index}}
```

**Intuition**：人类动作的 body parts 是**强相关**的——开心时脸上笑，手也会张开；生气时脸皱眉，肩会往前缩。这种跨 part 的 correlation 是 universal prior，不依赖 speech。模型在大量 unpaired motion data（BEAT + AMASS 共 60 小时 motion）上学到这种 correlation，就建立了 **motion distribution 的内部模型**。

#### Temporal alignment

模板：

```
Task Prompts: Translate mask to unmasked motion.
Conditions: Masked Tokens V_condition = {v_m^i ∈ V_m | i ∈ {masked sequence token index}}
Answer: Unmasked Motion Tokens V_answer = {v_m^i ∈ V | i ∈ {unmasked sequence token index}}
```

**Intuition**：这是 [MAE](https://arxiv.org/abs/2111.06377) 在 motion 上的版本。随机 mask 一些 frame，让模型预测，强迫它学 temporal dynamics（什么时候停顿、什么时候爆发、节奏感）。

### 3.2 Audio-text alignment

利用大量 audio-text pair data（这里用 LibriSpeech，~1000 小时）做 ASR-like 任务，把 audio embedding 对齐到 text embedding 空间。**模型从未见过 audio-motion pair**，但学完这个任务，audio 的 semantic content 已经能流到 text embedding space，后续 audio→motion 任务就能复用 LM 的 semantic reasoning。

这一点很关键——它**绕过了 audio-motion 数据稀缺问题**。这和 [SpeechGPT](https://arxiv.org/abs/2305.13045) 的多阶段训练思路类似：先用大量 audio-text 训 audio-text alignment，再用小量 audio-motion 训下游。

### 3.3 为什么这个 pre-training 设计 work

Table 2 ablation 给了答案：

| Setting | FGD↓ | BC↑ | Diversity↑ |
|---|---|---|---|
| W/o pre-training | 5.501 | 7.721 | 14.281 |
| W/o A2T | 5.443 | 7.721 | 14.499 |
| W/o spatial | 6.336 | 7.381 | 14.173 |
| W/o temporal | 6.800 | 7.341 | 13.810 |
| W/o motion | 7.776 | 7.344 | 14.640 |
| Ours | 5.301 | 7.780 | 15.165 |

几个观察：
1. **W/o motion (去掉整个 motion alignment)** 掉得最狠，FGD 从 5.301 → 7.776。说明 motion prior 是 pre-training 的核心。
2. **Temporal 掉得比 spatial 多**（FGD 6.800 vs 6.336）。Temporal dynamics 比 spatial correlation 更难学，pre-training 给的 gain 更大。
3. **A2T 掉得最少**（FGD 5.443 vs 5.301）。说明 audio→motion 的 semantic bridge 主要靠 LM 自己的 text understanding，A2T 是 bonus 不是 foundation。

---

## 4. Post-training：Instruction Following

Pre-training 之后做 instruction tuning，把下游任务格式化成 prompt→answer 形式。论文 Table 4 给了大量 instruction template，我挑几个有代表性的：

- `Audio-to-Full Motion`：Based on [audio], generate a synchronized movement sequence involving both face, hands, upper and lower body. → `[face][hands][upper][lower]`
- `Audio-to-Hands Body Motion`：Generate expressive hand gestures that reflect the cues in [audio]. → `[hand]`
- `Emotion-to-Motion`：Generate a movement sequence that fully embodies the emotion of [emotion] using the face, hands, upper body, and lower body. → `[face][hands][upper][lower]`
- `Motion-to-Emotion`：What emotion is conveyed by the movements in the face, hands, upper body and lower body within [face][hands][upper][lower]? → `[emotion]`

**关键 design**：每个 part 都可以单独 prompt，这 unlock 了 **editable gesture generation**——可以让 upper body 跟着 audio，lower body 跟着 text（比如 "walk forward"），合成出"边走边说话"的 motion。这是 [BEAT](https://arxiv.org/abs/2211.14303)/[EMAGE](https://arxiv.org/abs/2401.00374) 这种 speaker-dependent pipeline 做不到的。

---

## 5. LM Training Objective

$$\mathcal{L}_{LM} = -\sum_{k=0}^{L_t - 1} \log p_\theta(s_t^k | s_t^{<k}, s_i) \tag{3}$$

变量解释：
- $s_t^k$：target sequence 的第 $k$ 个 token
- $s_t^{<k}$：target sequence 中 $k$ 之前所有 token（teacher forcing）
- $s_i$：input token sequence（encoder 输入）
- $\theta$：model 参数
- $L_t$：target sequence 长度

这是标准 encoder-decoder cross-entropy loss。Encoder 处理 input modality tokens，decoder autoregressive 地生成 output modality tokens。最大输入长度 512。

**模型选择**：220M [Flan-T5-Base](https://arxiv.org/abs/2210.11416)。这个 size 选得很合理：
- T5 family 已经 instruction-tuned，post-training 时 instruction following 能力强
- Encoder-decoder 比 decoder-only 更适合 conditional generation
- 220M 足够大以携带 semantic prior，又足够小可以在 8×H100 上 full-parameter finetune（不用 LoRA）

论文明确说**不用 LoRA**，因为目标是最大化 modality alignment，需要全参数 finetune。这和 [LLaVA](https://arxiv.org/abs/2304.08485) 早期的 projector-only tuning 思路不同，更接近 [BLIP-2](https://arxiv.org/abs/2301.12597) 的 full Q-former tuning。

---

## 6. Experiments：SOTA 与数据效率

### 6.1 BEATv2 co-speech gesture generation

| Method | FGD↓ | BC↑ | Diversity↑ | Condition |
|---|---|---|---|---|
| DisCo | 9.417 | 6.439 | 9.912 | audio |
| CaMN | 6.644 | 6.769 | 10.86 | audio, text, facial |
| DiffStyleGesture | 8.811 | 7.241 | 11.49 | audio, style |
| Habibie et al. | 9.040 | 7.716 | 8.213 | audio, text |
| TalkSHOW | 6.209 | 6.947 | 13.47 | audio |
| SynTalker | 6.413 | 7.971 | 12.721 | audio, text |
| EMAGE | 5.512 | 7.724 | 13.06 | audio, text |
| Ours w/o lang pre-train | 7.470 | 6.148 | 14.162 | audio |
| Ours w/o multi-modal pre-train | 5.408 | 7.742 | 14.418 | audio |
| **Ours** | **5.301** | **7.780** | **15.167** | audio |

Metric 解释：
- **FGD (Frechet Gesture Distance)**：生成 motion 分布和真实 motion 分布之间的 Fréchet distance（类似 FID），衡量 realism
- **BC (Beat Correlation)**：motion 节拍和 audio 节拍的相关性，衡量 audio-motion 同步
- **Diversity**：生成 motion 之间的 L1 距离，衡量多样性

几个关键 observation：
1. **只用 audio** 就超过了所有 baseline，包括用 audio+text 的 CaMN、Habibie、SynTalker、EMAGE。这说明 LM 的 semantic understanding 已经内化了 text 的功能，不需要显式喂 transcript。
2. **w/o lang pre-training 掉得很惨**（FGD 5.301 → 7.470）。Flan-T5 自带的 language understanding 是 foundation。
3. **w/o multi-modal pre-training 也掉**（FGD 5.301 → 5.408）。Pre-training 给的 motion prior 是真有贡献的，不只是文本理解。

### 6.2 Data efficiency: 这是 paper 最强的 selling point

Figure 5 显示：在只用 1/32 训练数据时，full model 的 FGD ≈ 6.5，而 w/o pre-training 的 ≈ 11，EMAGE ≈ 9.5。**1/32 数据下比 EMAGE 用满数据还差一点**——这种 generalization gain 非常 dramatic。

**Intuition**：为什么 pre-training 这么有效？因为 motion 的"grammar"是共享的——同一个 speaker 的新数据点只是少量 fine-tuning，把它已有的 motion prior adapt 到这个 speaker 的 idiosyncratic gesture style 上。这和 [Flamingo](https://arxiv.org/abs/2204.14198) 在 few-shot image captioning 上的发现一致：好的 pre-training 让 downstream 任务只需要学"风格微调"。

### 6.3 Editable gesture generation

Section 4.3 展示了一个 emergent 能力：upper body prompt 跟 audio，lower body prompt 跟 text（"walk forward"），合起来就是"边走边说话"。这是 instruction tuning 的 bonus——part-level 的 prompt 设计让模型天然支持组合。

### 6.4 Motion-to-Emotion: 真正的 novel task

| | Bleu@1↑ | Rouge↑ | BertScore↑ |
|---|---|---|---|
| GT | 100 | 100 | 99.9 |
| Random | 2.45 | 4.44 | 0.19 |
| MotionGPT | 1.68 | 10.67 | 2.31 |
| **Ours** | **14.71** | **26.67** | **16.94** |

[MotionGPT](https://arxiv.org/abs/2306.14767) 完全 fail（和 random 差不多），因为它只学过 caption "宏观动作"（walking, sitting），没学过 emotion from subtle gesture。这篇因为做了 motion-text alignment 和 emotion instruction tuning，能反推 emotion。**这个 task 在 mental health / psychiatry / HCI 里有真实应用价值**——读懂身体语言是心理学评估的基础能力。

---

## 7. Limitations 和我的思考

### 7.1 Discrete tokenization 的信息损失

论文 discussion 里提到 "sometimes fails to produce coherent motion potentially due to discrete motion tokenization"。这是 VQ-based motion generation 的通病——codebook 是有限离散集合，fine-grained motion 会被 quantize 到最近的 codebook entry，导致 jitter 或 frozen pose。

**解决方向**：连续 tokenization，例如 [MoMask](https://arxiv.org/abs/2312.02660) 的 residual VQ，或者完全放弃 VQ 用 [Motion Latent Diffusion](https://arxiv.org/abs/2305.12373) 那种 continuous latent。最近 [MotionGPT-2](https://arxiv.org/abs/2406.04419) 在这个方向有探索。

### 7.2 HuBERT 的 prosody loss

HuBERT 主要学 phoneme-level content，对韵律、语调、energy 这些 prosody 信息保留有限。但 co-speech gesture 强依赖韵律（重音、停顿、语调）来对齐 beat。这可能限制了 BC metric 进一步提升的空间。

**解决方向**：可以加一个 prosody-aware tokenizer，比如 [SoundStream](https://arxiv.org/abs/2107.03312) 或 [EnCodec](https://arxiv.org/abs/2210.13473) 的 acoustic tokenizer，和 HuBERT 的 semantic tokenizer 并行使用——这是 [AudioLM](https://arxiv.org/abs/2209.03143) 的 semantic+acoustic 两层方案。

### 7.3 50 fps audio vs 30 fps motion

频率不匹配是 implicit 的，模型需要自己学。可以显式做 cross-attention 或 resample 让两者对齐。

### 7.4 60 hours motion data 的天花板

BEATv2 + AMASS = 60 小时 motion。对比 [LAION-5B](https://arxiv.org/abs/2210.08414) 这种 image-text 的 billion 级数据，motion 数据小了 4-5 个数量级。这是整个 motion generation 领域的天花板。可以用 [NeuralDome](https://arxiv.org/abs/2303.18024)、[EgoGen](https://arxiv.org/abs/2401.08714)、[HOI-M³](https://arxiv.org/abs/2312.06553) 这类 synthetic pipeline 来 augment。

### 7.5 为什么不直接用更大的 LM

220M 是 base T5。如果上 [Flan-T5-XL](https://arxiv.org/abs/2210.11416)（3B）或 [UL2](https://arxiv.org/abs/2205.05131)（20B），是否会有 emergent ability？这个 paper 没做 scaling law 实验，是遗憾。但 motion token sequence 长度很大（T 帧 × 4 parts），attention 是 $O(L^2)$，扩展到更长序列更难。可以用 [Longformer](https://arxiv.org/abs/2007.14072)、[BigBird](https://arxiv.org/abs/2007.14062) 或者 [Mamba](https://arxiv.org/abs/2312.00752) 这类 efficient attention。

### 7.6 Eval metric 的根本问题

论文 Section 7.5 自己指出：text-to-motion 的标准 metric (FID, R-precision) 都和 H3D-Format 强耦合，不能公平评测 compositional representation。这是领域内的系统性问题，需要 community 一起做 representation-agnostic 的 metric。可以参考 [T2M-Coach](https://arxiv.org/abs/2310.19078) 或 human evaluation 的方向。

---

## 8. 把它放进更大的图景

这篇 paper 实际上是几个趋势的 intersection：

1. **Multimodal LLM 扩展到时序连续模态**：从 [LLaVA](https://arxiv.org/abs/2304.08485) (image) → [VideoChat](https://arxiv.org/abs/2305.06355) (video) → [SpeechGPT](https://arxiv.org/abs/2305.13045) (audio) → 这篇 (motion+audio+text)。下一个 frontier 是 robot action tokens（[RT-2](https://arxiv.org/abs/2307.15818)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)）。

2. **Compositional representation 在 motion 上的胜利**：[EMAGE](https://arxiv.org/abs/2401.00374)、[TalkSHOW](https://arxiv.org/abs/2305.01378)、[NeuralDome](https://arxiv.org/abs/2303.18024) 一路都在做 part decomposition，这篇把它升级成 part-level vocabularies + LM-friendly tokens，是自然的下一步。

3. **Pre-training 在 data-scarce 领域的胜利**：和 [MAE](https://arxiv.org/abs/2111.06377) 在 vision、[HuBERT](https://arxiv.org/abs/2106.07447) 在 speech、[VQ-VAE](https://arxiv.org/abs/1711.00937) 在 audio 的故事一样——大模型时代，好的 pre-training 可以让 downstream 用极少数据。Motion 领域因为 motion capture 成本高，这个 pattern 尤其重要。

4. **Generative pre-training as translation**：这篇把 "predict modality Y from modality X" 当成通用 pre-training pattern。这和 [AudioLM](https://arxiv.org/abs/2209.03143) 的 coarse-to-fine、[MUSE](https://arxiv.org/abs/2209.10752) 的 masked text、[Flamingo](https://arxiv.org/abs/2204.14198) 的 interleaved vision-text 一脉相承——都是 "translation as pre-training"。

---

## 9. 一句话总结

**The Language of Motion** 是把 LLM 范式完整移植到 3D human motion 的第一篇 systematic work：用 compositional VQ 把 motion 变成 tokens，用 Flan-T5 当 backbone，用 spatial/temporal/audio-text 三类 translation task 做 generative pre-training，用 instruction tuning 做 downstream。它证明了 LM 范式在 motion 领域 work，且 data-efficient，且能 unlock 新 task (editable generation, motion-to-emotion)。Limitation 在 VQ 的信息瓶颈和 audio tokenizer 的 prosody loss——这些是下一代"continuous + prosody-aware" motion LM 要解决的方向。

---

**Relevant Links**:
- Project page: https://languageofmotion.github.io
- [Flan-T5](https://arxiv.org/abs/2210.11416)
- [EMAGE (CVPR 2024)](https://arxiv.org/abs/2401.00374)
- [BEAT (ECCV 2022)](https://arxiv.org/abs/2211.14303)
- [TalkSHOW (CVPR 2023)](https://arxiv.org/abs/2305.01378)
- [MotionGPT (NeurIPS 2023)](https://arxiv.org/abs/2306.14767)
- [T2M-GPT (CVPR 2023)](https://arxiv.org/abs/2301.06052)
- [HumanML3D (CVPR 2022)](https://arxiv.org/abs/2205.00589)
- [HuBERT](https://arxiv.org/abs/2106.07447)
- [AMASS](https://amass.is.tue.mpg.de/)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- [AudioLM](https://arxiv.org/abs/2209.03143)
- [SpeechGPT](https://arxiv.org/abs/2305.13045)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [MAE](https://arxiv.org/abs/2111.06377)
