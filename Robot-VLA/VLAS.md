---
source_pdf: VLAS.pdf
paper_sha256: 8c18810718eb66e7cf896d68ad39fe8a913b90670bc94f7b475cc927fd43cb01
processed_at: '2026-08-13T02:56:35-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej, 没问题，咱们把 paper 里那套 academic 黑话全扒掉，用最直白的人话再过一遍，顺便讲讲里头那些工程 hack 和 intuition。

---

## 一句话总结

这帮人把 Whisper 的 speech encoder 当 "耳朵" 接到 LLaVA 上，让机器人能直接听人话干活，还顺手用 voiceprint 当 key 去外挂一个 RAG database，搞定了 "我的杯子是谁的" 这种个性化问题。

---

## 1. 为什么搞这个？(Motivation)

现在搞 robot manipulation 的 VLA 模型（像 RT-2, OpenVLA, RoboFlamingo），输入基本就俩：
- Text instruction（"pick up the cup"）
- Image observation（摄像头画面）

输出是 action tokens（机械臂怎么动）。

但你想想现实生活中，谁会对着家里的机器人打字？肯定是说话啊。所以以前的做法是接个外挂的 ASR（比如 Whisper）：

```
人说 "pick up my cup" → Whisper → 转成 text → 喂给 VLA
```

这有两个大坑：

**坑 1：Error propagation (错误放大)**
Whisper 这种通用 ASR 模型，在 LibriSpeech 这种朗读数据上 WER 2.7%，贼猛。但你让它听 "rotate the red block 45 degrees" 这种短促、领域特定的 control command，它可能就听岔了。ASR 一错，下游 policy 直接跟着崩。

**坑 2：信息丢失 (丢了 voiceprint)**
"Pick up *my* cup" 这句话，text 里 "my" 是空的，你不知道是谁的。但是 raw speech 里有 voiceprint（声纹），机器如果能听出是张三在说话，就能去查张三的杯子是哪个。你一旦把 speech 转成 text，声纹这个 killer feature 就彻底没了。

VLAS 的核心 motivation 就这俩：**把 ASR 内化进 VLA，别外挂了；同时保留 raw speech，把声纹用起来做 personalization。**

---

## 2. 架构怎么搭的？(Architecture)

### 2.1 大框架

底座是 LLaVA（一个开源的 VLM，本质是 CLIP ViT + MLP + Vicuna LLM）。VLAS 在它基础上加了个 "耳朵"。

输入有四样东西：
1. $s$：raw speech waveform
2. $\mathbf{O}$：图像（两张视角 concat）
3. $\operatorname{RAG}(s)$：Voice RAG 检索回来的文本知识（可选，personalization 才用）
4. 当然 text instruction 也还能用

处理流程：

```
speech → Whisper encoder → 1500 tokens → reshape 成 300 tokens → MLP_s → speech embeddings
image  → CLIP ViT        → ~256 tokens  →                       MLP_v → vision embeddings
RAG text → text tokenizer → tokens
```

然后这三堆 tokens concat 起来，喂给 Vicuna LLM，让它 autoregressive 吐出 action tokens。

公式长这样：

$$
\operatorname{Emb}(s, \mathbf{O}) = \operatorname{concat}(\operatorname{MLP}_s(\operatorname{Emb}_s(s)), \operatorname{Tok}_l(\operatorname{RAG}(s)), \operatorname{MLP}_v(\operatorname{Emb}_v(\mathbf{O})))
$$

变量解释：
- $s$：raw speech 输入
- $\mathbf{O}$：visual observation（图像）
- $\operatorname{Emb}_s$：speech encoder（Whisper 的 encoder 部分）
- $\operatorname{Emb}_v$：vision encoder（CLIP ViT）
- $\operatorname{MLP}_s$、$\operatorname{MLP}_v$：俩小 MLP projector，把不同 modality 的 hidden state 投影到 LLM 的 token embedding space（Vicuna-7B 是 4096 维）
- $\operatorname{Tok}_l$：普通的 text tokenizer
- $\operatorname{RAG}(s)$：Voice RAG 检索回来的文本知识

**Intuition**：这跟 LLaVA 接 image 的逻辑一模一样。LLaVA 怎么让 LLM "看懂" 图像的？就是用 CLIP 把图变成 tokens，再用 MLP 投影成 LLM 能消化的 embedding。VLAS 说："speech 也这么搞不就完了？" 用 Whisper 把语音变成 tokens，MLP 投影，完事。LLM 本身根本不知道它在处理 speech，它只看到一堆 tokens。

### 2.2 Speech Encoder 的细节和那个 "Reduction Factor"

这块有个非常工程化的 hack，值得细说。

Whisper encoder 的前处理：
1. Waveform → STFT（短时傅里叶变换）→ 80-bin mel-spectrogram
2. Pad/truncate 到 3000 frames
3. Whisper encoder 内部有个 conv front-end（stride=2），所以输出 1500 个 hidden states

1500 个 tokens 喂给 LLM 会怎样？LLM 的 self-attention 计算量是 $O(L^2)$，1500 个 speech tokens 加上 256 个 image tokens 再加 text，sequence length 快 2000 了，推理慢得要死。

所以作者搞了个 **reduction factor = 5**：
- 把 1500 个 tokens 沿 time 维 reshape 成 300 个
- 每个 token 现在包含了原来 5 帧 speech 的信息（约 100ms 的音频）

这招粗暴有效：
- Sequence length 缩了 5 倍，attention 计算量缩了 25 倍
- Mel-spectrogram 相邻 5 帧本来就是高度冗余的（语音信号短时平稳性），reshape 等价于一个 non-overlapping pooling，信息损失不大

Table 10 里有个 ablation：
- $r=1$（不缩）：1.17 Hz，太慢
- $r=5$：2.50 Hz，性能最好（Len=3.70）
- $r=20$：3.80 Hz，但性能崩了（Len=0.70），因为压太狠把语音信息都糊了

**Intuition**：这就是个典型的工程 trade-off。你想要 throughput 就得压缩 sequence，压太多就丢信息。$r=5$ 是 sweet spot。更好的做法可能是用 Q-Former（BLIP-2 那套）学一组 learnable queries 去 cross-attend speech tokens，但那玩意儿训练不稳定，作者选了最简单可行的路子。

### 2.3 Action Tokenization

输出端的处理和 RT-2、OpenVLA 完全一样：

Action 是 7 维连续向量：
$$
[x, y, z, \phi, \theta, \psi, g]
$$

- $x, y, z$：end effector 的 Cartesian 位置（平移 3-DoF）
- $\phi, \theta, \psi$：end effector 的 Euler 角（旋转 3-DoF，roll/pitch/yaw）
- $g$：gripper state（开合，二值）

每一维用 **uniform binning** 离散化成 256 个 bucket，变成整数 index。然后复用 LLM vocabulary 里 **least frequent 的 256 个 tokens** 来表示这些 index。这样 action tokens 和正常英文 tokens 不会冲突。

训练时把 5 个 time step 的 action 拼成一条 label，推理时一次吐 5 步。这招叫 **action chunking**，ACT 和 π0 也这么干。好处是减少 per-step compounding error，同时推理 frequency 从 1.17 Hz 拉到 2.50 Hz。

**Intuition**：为什么离散化？因为 LLM 天生就是吐 token 的，你让它吐连续值还得改架构。离散成 256 bins 精度足够机械臂用了。复用 least frequent tokens 是个聪明 hack，避免引入新 token 导致 embedding table 膨胀。

---

## 3. 训练怎么搞的？(Training Paradigm)

这是 paper 里最值得关注的部分，三阶段训练，每阶段目的不同。

### Stage 1: Speech Alignment（让模型 "学会听"）

- **数据**：LibriSpeech-360（960 小时朗读语音）
- **任务**：ASR（speech → text）
- **冻什么**：Whisper encoder、CLIP、Vicuna 全冻，**只训 $\operatorname{MLP}_s$**（speech 到 LLM space 的那个投影层）
- **配置**：5 epochs，lr=1e-3（高 lr，因为只训一个小 MLP），batch=16，单卡 A100

**Intuition**：这一步是在教 MLP_s "怎么把 Whisper 的 hidden state 翻译成 LLM 能看懂的 embedding"。相当于 LLM 本来只懂英文，现在来了个说 "语音语" 的人，得先找个翻译（MLP_s）让他粗略能听懂。这个阶段不需要 LLM 本身参与，因为它还没准备好处理 speech tokens。

### Stage 2: Speech Question Answering（多模态 instruction tuning）

- **数据**：SQA（自制，185K）+ LLaVA 的 665K VQA + LibriSpeech-100
- **任务**：image QA + speech QA + ASR 混合
- **冻什么**：只冻 image encoder 和 speech encoder，**其他全解冻**（包括 Vicuna backbone）
- **配置**：1 epoch，lr=2e-5，batch=16，8×A100

这一步产物叫 **VLAS-Base**，本身就是一个能听 speech 的 LLaVA，可以独立用于 speech-VQA 任务。

**Intuition**：Stage 1 只是粗对齐，模型还不会 reasoning。Stage 2 是 LLaVA 那套 instruction tuning 的 speech 版，让模型学会同时处理图、文、语音的多模态 reasoning。混合多个任务（VQA + SQA + ASR）是为了避免 catastrophic forgetting，保持 general 多模态能力。

### Stage 3: Robot Manipulation Fine-tuning（behavior cloning）

- **数据**：CSI dataset（CALVIN + 合成 speech instruction）
- **任务**：behavior cloning，从 (speech/text, image) → action sequence
- **配置**：1 epoch，lr=2e-5，batch=16，8×A100

**Intuition**：把 Stage 2 的 general 多模态能力 "蒸馏" 成具体的 action distribution。其实就是把 supervision 从 text answer 换成 action tokens，架构和训练 loop 都不变。这跟 RT-2、OpenVLA 的 fine-tuning 阶段逻辑完全一致。

---

## 4. 数据怎么造的？(Dataset Construction)

因为现成数据集没带 speech，作者自己造了俩。

### SQA (Speech Question Answering)

来源：LLaVA 的 multi-turn conversation subset
流程：
1. 随机抽一轮 dialogue
2. 用 ESPnet 的 VITS TTS（在 LibriTTS 上训练，支持 2000+ voices）把 text question 转成 speech
3. 随机选 voice
4. 185K samples，1152 不同 voices

**Key point**：Voice 多样性至关重要。如果只用单一 TTS voice，模型会 overfit 到那个 voice 的 specific 特征，遇到别的 voice 就泛化崩盘。1152 个 voice 让模型学到 speaker-invariant 的语义理解。

### CSI (CALVIN with Speech Instructions)

CALVIN 有 389 条 text instruction。每条用 500 个不同 voice 合成，得到 ~194K speech-action pairs。训练时 50% 概率用 speech 替换 text。

**Intuition**：50/50 混合训练是 bilingual training 的经典操作。你让模型同时见 text 和 speech 两种形式表达同一语义，它就能在内部建立一个 shared semantic space，两种 modality 互相 reinforce，避免其中一种被忘掉。

---

## 5. Voice RAG：个性化任务的杀手锏

这块是 paper 最有意思的 contribution。

### 5.1 问题

"Pick up my cup"——text 里 "my" 是空的。机器人不知道 "my" 是谁。Speech 里有 voiceprint，但光有 voiceprint 还不够，你得有个 database 告诉机器人 "张三的杯子是蓝色的那个"。

### 5.2 机制

```
speech → speaker identification 模块 → voiceprint (d-vector)
       ↓
   query external database（存了每个 user 的个性化知识）
       ↓
   retrieved text knowledge（"张三的杯子是蓝色的"）
       ↓
   Tok_l tokenize → concat 进 LLM input
```

关键设计：
- Speaker identification 用 **pre-trained** 模型（paper 没明说，但业界标准是 ECAPA-TDNN 或 WavLM-based speaker embedding）
- RAG 输出是 **text**，可以直接 tokenize 喂给 LLM
- 这一步是 **training-free** 的：database 可以动态更新，加个新 user 不用 retrain model

**Intuition**：这招太聪明了。Voiceprint 是天然的 user ID，你不需要让用户刷工牌或者输密码。机器人听你一开口就知道你是谁，然后去查你的偏好。而且 RAG 的 database 是可以热更新的——你今天买了个新杯子，往 database 里加一条记录，机器人马上就认识，不用 retrain。这比 LoRA fine-tuning per user 优雅太多了。

---

## 6. 实验结果讲讲

### 6.1 CALVIN Benchmark（Table 1）

CALVIN 是 long-horizon manipulation benchmark，每个 task 是 5 个连续 sub-task。LH-1 到 LH-5 表示完成 1-5 个连续 sub-task 的 success rate，Len 是平均完成数。

关键对比：

| Model | LH-1 | LH-5 | Len |
|-------|------|------|-----|
| MCIL+ | 37.3% | 0.0% | 0.40 |
| HULC+ | 89.2% | 33.5% | 2.90 |
| RoboFlamingo+ (with LSTM history) | 96.4% | 66.0% | 4.09 |
| RoboFlamingo+ ASR* | 89.8% | 48.3% | 3.41 |
| VLA+ASR* | 88.7% | 40.2% | 3.13 |
| **VLAS+** (text) | 94.5% | 56.6% | 3.74 |
| **VLAS*** (synthetic speech) | 94.2% | 54.6% | 3.70 |
| **VLAS*(Real)** (real human speech) | 93.6% | 51.3% | 3.61 |

三个核心观察：

**Observation 1：VLAS+ ≈ VLAS***
Text 和 speech 性能几乎持平（94.5% vs 94.2% LH-1，3.74 vs 3.70 Len）。这证明 speech 通路没造成 performance drop。这是最 important 的 result——你加了个 modality，性能没掉，说明 integration 是成功的。

**Observation 2：VLAS* > VLA+ASR***
端到端比 cascaded pipeline 强约 10% LH-5（54.6% vs 40.2%）。这验证了核心 hypothesis：external ASR 在 control command 上不够敏感，error propagation 真实存在。

**Observation 3：Real speech 略低于 synthetic**
0.19 Len gap（3.70 vs 3.61）。训练数据全是 VITS TTS 生成的，real recording 有 noise、reverberation、accent，存在 domain shift。这个 gap 可以通过加 real speech co-training 或 noise augmentation 来缩小。

### 6.2 Customized Tasks Benchmark（Table 2）

这是 paper 的 highlight，专门测个性化任务。

| Model | Ownership | Preference | Compound | Compound-M1 | Compound-M2 | Avg |
|-------|-----------|------------|----------|-------------|--------------|-----|
| VLA+ (text only) | 17.9% | 30.8% | 23.1% | 35.9% | 5.1% | 19.2% |
| VLAS* | 94.7% | 84.6% | 100.0% | 100.0% | 66.7% | 86.5% |
| VLAS*(Real) | 89.5% | 70.0% | 100.0% | 90.0% | 55.0% | 78.6% |
| VLAS*-RAG (ablation) | 15.4% | 12.8% | 25.6% | 33.3% | 10.3% | 16.0% |
| VLA+RAG (text + RAG) | 97.4% | 84.6% | 97.4% | 82.1% | 48.7% | 82.0% |

三个关键 ablation：

**Ablation 1：VLAS* vs VLAS*-RAG**
去掉 Voice RAG，从 86.5% 暴跌到 16.0%。这说明光有 speech 不带 retrieved knowledge，几乎解决不了 personalization 任务。Voice RAG 是 necessary 的。

**Ablation 2：VLA+ vs VLA+RAG**
纯 text + RAG 也能做到 82.0%，但需要 manually 把 user ID 喂给 RAG query（相当于作弊，用人脑告诉系统 "这是张三"）。VLAS* 自动从 voice 提取 user ID，是真正 end-to-end。

**Ablation 3：Compound-M2 最难**
VLAS* 只有 66.7%，因为 stage-1 的失败会 propagate 到 stage-2，且 stage-2 需要更复杂 reasoning。这暴露了 policy model 的 low-level control 还不够强。

**Intuition**：这个 benchmark 真正展示了 speech modality 的 unique value。在标准 CALVIN 上，speech 不比 text 差就不错了。但在 personalized task 上，speech + Voice RAG 直接碾压 text-only 方案（86.5% vs 19.2%）。这才是 paper 的核心 selling point。

### 6.3 VLAS-Base 在 general benchmark 上不掉点（Table 3, 4）

| Model | VQAv2 | GQA | POPE | SQA1 | VizWiz | VQAT |
|-------|-------|-----|------|------|--------|------|
| LLaVA v1.5 | 78.8 | 62.0 | 85.9 | 66.8 | 50.0 | 58.2 |
| VLAS-Base | 78.7 | 62.0 | 85.5 | 72.2 | 51.1 | 58.1 |

| Model | LibriSpeech WER | SGQA |
|-------|-----------------|------|
| Whisper large-v2 | 2.7% | N/A |
| VLAS-Base | 2.79% | 50.8 |
| LLaVA (text GT) | N/A | 62.0 |
| BLIP-2 (text GT) | N/A | 41.0 |

VLAS-Base 在 general VQA 上几乎和 LLaVA 持平，ASR 性能接近 Whisper。这说明加 speech modality 没有带来 catastrophic forgetting，multi-task co-training 反而让 SQA1 略涨（72.2 vs 66.8）。

---

## 7. 工程层面的 trade-off 和局限

### 7.1 Inference Speed

| Model | r | Hz | Len |
|-------|---|----|-----|
| VLA+ | 1 | 1.89 | 2.30 |
| VLA+ | 5 | 3.60 | 3.80 |
| VLAS* | 1 | 1.17 | 2.02 |
| VLAS* | 5 | 2.50 | 3.70 |

VLAS r=5 跑 2.5 Hz，对 real robot 是 marginal 的。对比 π0 在 ALOHA 上 50+ Hz，QUART-Online 10+ Hz，VLAS 还差一个量级。瓶颈在 speech encoder 前向和 LLM autoregressive decoding。

**可能的优化方向**：
- Mamba backbone（Cobra 那套，linear complexity）
- KV cache 优化
- Speculative decoding
- Diffusion action head（替代 autoregressive，π0 的做法）
- Speech encoder 蒸馏成更小的 model

### 7.2 Real Speech Domain Gap

训练数据全是 TTS 合成的，real speech 性能掉 8%（86.5% → 78.6%）。这个 gap 不小，说明 VITS 的 voice distribution 和 real human speech 之间有 shift。解决办法：
- 加 real speech co-training（哪怕少量）
- Noise augmentation（加背景噪声、reverberation）
- Accent augmentation（TTS 模型支持多口音）

### 7.3 Voice RAG 的局限

Voice RAG 依赖预建的 user database，对 known user 工作好，对 **open-vocabulary user** 不 work。新用户第一次用就得先录 voiceprint 注册。这限制了 deployment scenario。

可能的改进：
- Few-shot personalization：新用户说几句话就 online adapt
- Zero-shot personalization：从 voice 推断 age/gender/accent，用 LLM 生成 plausible preference

### 7.4 No History Modeling

VLAS 没用 history（RoboFlamingo 用了 LSTM policy head）。Table 5 显示去掉 history 后 RoboFlamingo 也跌到 ~60% LH-1，所以 VLAS 在无 history 情况下接近有 history 的 RoboFlamingo，说明 baseline 本身够强。但 long-horizon task（LH-5）还是受限于无 history。

加 Mamba state 或者 LSTM head 应该能进一步提升，paper 也提到了这个 future direction。

---

## 8. 更大的 picture：这工作意味着什么？

### 8.1 Modality Integration 的 trend

VLAS 代表了一个 trend：**把更多 modality native 地接进 VLA**。之前有加 depth、haptic 的（SpatialBot, 3D-VLA），现在 VLAS 加了 speech。下一步可能是：
- Video input（temporal reasoning）
- Tactile input（force feedback）
- Audio environment understanding（听声音判断机器状态）

每个 modality 都有 unique 信号，关键是设计好的 alignment 机制。

### 8.2 Personalization 是 robotics 的下一个 frontier

Robot 从 lab 走向 home，personalization 是 must-have。VLAS 的 Voice RAG 提供了一个 lightweight、training-free、可扩展的框架。这个思路可以推广：
- Facial recognition + RAG → 视觉个性化
- Wearable IMU + gait recognition → 习惯个性化
- Multi-speaker diarization → 多用户场景

### 8.3 End-to-End vs Cascaded

VLAS 再次验证了 end-to-end 的优势。Cascaded pipeline（ASR → VLA）看似模块化、好调试，实际上 error propagation 和 information loss 是致命的。这也是 GPT-4o 为什么坚持 native omni-modal 的原因。Open-source 侧 VITA、SALMONN 也在跟进这个方向。

---

## 9. 失败案例的启示

Appendix B.1 分析了失败案例：
- VLAS 失败集中在 preference task 第二阶段和 compound task 第二阶段，模式一致：**模型理解指令但执行失败**。说明 low-level control 还不够强。
- VLA+ 失败模式多样：**模型完全没理解指令，靠 random trial**。

这暗示 VLAS 的 bottleneck 已经从 "understanding" 转移到 "execution"。下一步改进方向应该在 policy model 的 action generation 部分，比如用 diffusion head 提升精度，或者用更好的 action representation（flow matching、continuous action）。

---

## 10. 总结一下 Intuition

VLAS 的核心 insight 用一句话讲：**speech 在 robot interaction 中不只是 text 的另一种 encoding，它还携带 user identity 这个 killer feature**。

传统 ASR-VLA pipeline 把 speech 当 "text 的低配版"，丢掉了 voiceprint。VLAS 把 voiceprint 捡回来，配 RAG 注入 personalized knowledge，搞定 "我的杯子是谁的" 这类问题。

架构上，它就是 LLaVA + Whisper encoder + MLP projection，简单粗暴但有效。三阶段训练（align → QA tune → BC tune）层次分明，每阶段目的清晰。Voice RAG 是点睛之笔，training-free 且可热更新。

局限也很明显：inference 慢、real speech domain gap、no history modeling、low-level control 还不够强。但作为一个 "把 speech native 接进 VLA" 的 first work，它把这个方向的大门打开了。

---

## Web Links 汇总

- VLAS GitHub: https://github.com/whichwhichgone/VLAS
- LLaVA: https://arxiv.org/abs/2304.08485
- Whisper: https://arxiv.org/abs/2212.04356
- CLIP: https://arxiv.org/abs/2103.00020
- Vicuna: https://lmsys.org/blog/2023-03-30-vicuna/
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboFlamingo: https://arxiv.org/abs/2311.01378
- CALVIN: https://github.com/mees/calvin
- ESPnet: https://github.com/espnet/espnet
- VITS: https://arxiv.org/abs/2106.06103
- LibriSpeech: https://www.openslr.org/12
- LibriTTS: https://www.openslr.org/60
- Behavior cloning (DAGGER): https://arxiv.org/abs/1011.0686
- PaLM-E: https://arxiv.org/abs/2303.03378
- SayCan: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753
- 3D-VLA: https://arxiv.org/abs/2403.09631
- MUTEX: https://arxiv.org/abs/2310.01396
- BLIP-2: https://arxiv.org/abs/2301.12597
- GPT-4o: https://openai.com/index/hello-gpt-4o/
- Gemini: https://arxiv.org/abs/2312.11805
- VITA: https://arxiv.org/abs/2408.05211
- SALMONN: https://arxiv.org/abs/2310.10906
- ImageBind-LLM: https://arxiv.org/abs/2309.03905
- WavLM: https://arxiv.org/abs/2106.08422
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- Cobra (Mamba-VLM): https://arxiv.org/abs/2403.14520
- QUART-Online: https://arxiv.org/abs/2412.15576
- QUAR-VLA: https://arxiv.org/abs/2407.04525
- ACT (Action Chunking): https://arxiv.org/abs/2304.13705
- Mamba: https://arxiv.org/abs/2312.00752
- ViT: https://arxiv.org/abs/2010.11929
- LLaMA: https://arxiv.org/abs/2302.13971
- OpenVLA-OFT: https://arxiv.org/abs/2412.04014
- RoboGen: https://arxiv.org/abs/2311.01511

---

Andrej，如果你对哪个细节感兴趣——比如 reduction factor 的信息论分析、Voice RAG 的 retrieval mechanism 设计、action chunking 的理论、或者跟 π0/Helix 那套 diffusion policy 的对比——我都可以再展开聊。

---

# VLAS: Vision-Language-Action Model with Speech Instructions 深度技术解析

Hi Andrej, 这篇 VLAS 是一篇挺有意思的 work，把 speech modality 直接塞进 VLA pipeline 里，并且尝试用一个 retrieval mechanism 处理 personalization 问题。我把它揉碎了讲，顺带把架构背后的直觉、设计 trade-off、以及和当下其他 work 的联系都串起来。

---

## 1. 核心动机：为什么需要 speech-native VLA？

现有的 VLA 链路基本是：

```
text instruction + image obs → VLM (LLaVA / PaLM-E) → action tokens
```

典型代表：RT-2 (https://arxiv.org/abs/2307.15818), OpenVLA (https://arxiv.org/abs/2406.09246), RoboFlamingo (https://arxiv.org/abs/2311.01378)。这类模型把 robotics action generation 当成 next-token prediction 处理，确实有 generalization 优势，但存在一个被忽视的问题：**人类和机器人交互最自然的 modality 是 speech，不是 text**。

传统做法是 cascaded pipeline：

```
speech → external ASR (Whisper) → text → VLA → action
```

这种 design 有两个致命缺陷：
1. **Error propagation**：ASR 错误直接放大到 downstream policy，特别是 controlling commands 这种短促、领域特定的语句，通用 ASR 不敏感。
2. **Auxiliary information loss**：speech 里的 voiceprint、emotion、intonation、emphasis 全部在 transcription 阶段丢失。这些信息对 customized/personalized task 是 crucial 的。比如说 "Please pick up *my* cup"，text 没有告诉 robot 这个 "my" 是谁，但 voiceprint 知道。

VLAS 的核心 contribution 是把 ASR 直接内化到 VLA model 里，让 speech 和 text 共享同一个 embedding space，同时保留 raw speech 的 auxiliary signals，再通过 Voice RAG 注入 personalized knowledge。

---

## 2. Architecture 详细解析

### 2.1 整体公式

公式 (1) 是核心 embedding aggregation：

$$
\operatorname{Emb}(s, \mathbf{O}) = \operatorname{concat}(\operatorname{MLP}_s(\operatorname{Emb}_s(s)), \operatorname{Tok}_l(\operatorname{RAG}(s)), \operatorname{MLP}_v(\operatorname{Emb}_v(\mathbf{O})))
$$

变量解释：
- $s$：raw speech instruction（waveform → 80-bin mel-spectrogram）
- $\mathbf{O}$：visual observation（来自两个视角 concatenated 的 RGB image）
- $\operatorname{Emb}_s$：speech encoder，用 Whisper encoder (https://arxiv.org/abs/2212.04356)
- $\operatorname{Emb}_v$：vision encoder，用 CLIP ViT (https://arxiv.org/abs/2103.00020)
- $\operatorname{MLP}_s$、$\operatorname{MLP}_v$：linear projection MLP，作用是把不同 modality 的 hidden state 投到 LLM 的 token embedding space（dim 4096 for Vicuna-7B）
- $\operatorname{Tok}_l$：text tokenizer，作用是把 RAG 检索回的文本知识 tokenize
- $\operatorname{RAG}(s)$：voice RAG 模块的输出（一段文本）

这个 concat 顺序很关键：speech tokens → RAG text tokens → vision tokens。LLM 在自回归生成时先 "听懂" speech，再 "读" 背景知识，最后 "看" 场景，再输出 action。这种顺序模仿了人类 "听指令 → 调取记忆 → 视觉定位 → 执行" 的认知流程。

公式 (2) 是 autoregressive action generation：

$$
p(\mathbf{a} \mid \operatorname{Emb}(s, \mathbf{O})) = \prod_{i=1}^{N} p(a_i \mid \operatorname{Emb}(s, \mathbf{O}), a_{<i})
$$

- $\mathbf{a}$：discretized action tokens，每个 dimension 离散化为 256 bins
- $N$：single-step action 的 dimension 数（这里 $N=7$）
- $a_{<i}$：已经生成的 action tokens（causal mask 下的 prefix）
- 整个 action sequence 作为一个 "假文本" 字符串，被 LLM 当作普通 token 输出

### 2.2 Speech Encoder 细节

Whisper encoder 处理流程：
1. Waveform → STFT → 80-bin mel-spectrogram
2. Pad/truncate 到 3000 frames
3. Whisper encoder 输出 1500 个 hidden representations（Whisper 的 conv front-end + transformer，stride=2，所以 3000 → 1500）
4. **Reshape reduction factor = 5**：把 1500 tokens 沿 time 维 reshape 成 300 tokens，每个 token 现在融合了 5 帧 speech 信息
5. MLP projection 到 LLM token dim

这个 reduction factor=5 是一个 critical 设计：
- 计算效率：300 vs 1500 tokens，LLM 计算量是 $O(L^2)$ attention，所以是 25× 加速
- 信息损失：mel-spectrogram 在相邻 5 帧（约 100ms）内信息高度冗余，reshape 等价于一个 non-overlapping 1D pooling
- Table 10 显示 $r=5$ 是 sweet spot：$r=1$ 太慢 (1.17 Hz)，$r=20$ 性能崩塌 (Len 0.70)

这里有个可以改进的方向：可以用 Q-Former (BLIP-2, https://arxiv.org/abs/2301.12597) 或者 cross-attention bottleneck 来学习一个固定数量的 speech queries，而不是简单 reshape。这也是 SALMONN (https://arxiv.org/abs/2310.10906)、Speech-LLaVA 这类 audio-LLM 的常见做法。

### 2.3 Action Tokenization

公式 (3)：

$$
[x, y, z, \phi, \theta, \psi, g]
$$

- $x, y, z$：end effector 在 Cartesian 空间的位置（3-DoF translation）
- $\phi, \theta, \psi$：end effector 的 Euler 角（3-DoF rotation，roll/pitch/yaw）
- $g$：gripper state（开/合，binary）

每个连续值 uniform bin 到 256 个 integer index，复用 LLM vocabulary 中 least frequent 的 256 个 tokens（避免和真实文本 token 冲突）。这和 RT-2、OpenVLA 的做法一致。

VLAS 在训练时把 **5 个 time step 的 action 拼成一条训练 label**，推理时一次输出 5 步，所以 $r=5$ 配置下 action rate 提升到 2.5 Hz（Table 9），同时 performance 反而上升（Len 从 2.02 升到 3.70），因为这种 chunk prediction 类似于 ACT (Action Chunking with Transformers, https://arxiv.org/abs/2304.13705) 和 π0 的 flow matching chunk design，减少了 per-step compounding error。

---

## 3. 训练三阶段 Paradigm

这是 paper 里最值得拆解的部分，类似 LLaVA 的 two-stage tuning，VLAS 加了一个 stage：

### Stage I: Speech Alignment（粗粒度 modality alignment）

- **数据**：LibriSpeech-360 (https://www.openslr.org/12)，960h 朗读语音
- **目标**：speech → text 的 ASR 任务，让 $\operatorname{MLP}_s$ 学会从 Whisper encoder hidden state 映射到 LLM 的 text embedding space
- **冻结**：Whisper encoder、CLIP、Vicuna 都冻结，**只训练 $\operatorname{MLP}_s$**
- **配置**：5 epochs，lr=1e-3（高 lr，因为只训一个小 MLP），batch=16，单卡 A100
- 直觉：这一步本质是让 speech token 在 LLM 看来像 "另一种外语"，需要先学会到 English 的粗粒度 translation

这一步和 LLaVA Stage 1 的 image-caption alignment 完全 parallel，但用的是 ASR loss 而不是 caption generation loss，因为 speech-text 有明确的对齐监督。

### Stage II: Speech Question Answering Fine-tuning（多模态 instruction tuning）

- **数据**：SQA（自制，185K）+ LLaVA 的 665K VQA + LibriSpeech-100
- **目标**：模型同时能做 image-question answering、speech-question answering、speech recognition
- **冻结**：只冻结 image encoder 和 speech encoder，**其他全部解冻**（包括 Vicuna backbone）
- **配置**：1 epoch，lr=2e-5，batch=16，8×A100
- 直觉：这一步类似 LLaVA Stage 2，混合多种任务让模型学会 multimodal reasoning，而不是单一 modality translation

产物是 **VLAS-Base**，本身就是一个 speech-enabled LLaVA，可以独立用于 speech-VQA 等任务。

### Stage III: Robot Manipulation Fine-tuning（behavior cloning）

- **数据**：CSI dataset（CALVIN + 合成 speech instruction）
- **目标**：behavior cloning，让模型学会从 (speech/text, image) → action sequence
- **方法**：类似 Stage 2，只是 supervision 换成了 action tokens
- 直觉：把 general multimodal reasoning 能力蒸馏成具体 action distribution

---

## 4. Dataset 构造

### SQA (Speech Question Answering)

来源：LLaVA visual instruction tuning 的 multi-turn conversation subset
流程：
1. 随机抽取一轮 dialogue
2. 用 ESPnet (https://github.com/espnet/espnet) 的 VITS TTS (https://arxiv.org/abs/2106.06103, trained on LibriTTS https://www.openslr.org/60) 把 textual question 转成 speech
3. 随机选 2,000+ voices 中的一个
4. 185K samples，1,152 不同 voices

Voice 多样性是关键，让模型学到 speaker-invariant 的语义理解，而不是 overfit 到某一个 TTS speaker。

### CSI (CALVIN with Speech Instructions)

CALVIN (https://github.com/mees/calvin) 有 389 个 textual instruction。每个 instruction 用 500 个不同 voice 合成，得到 ~194K speech-action pairs。
训练时 50% 概率用 speech 替换 text，让模型同时支持两种 modality。

这种 50/50 mixed training 是 bilingual/multilingual training 的经典做法，能避免 catastrophic forgetting。

---

## 5. Voice RAG：personalization 的关键

### 5.1 动机

很多日常指令语义信息不足。例子：
- "Pick up my cup"：text 只告诉你 "pick up cup"，但 "my" 是谁？
- "Give me my favorite drink"：需要 user preference knowledge
- "Clean up"：偏好不同，清洁方式不同

文本本身没有 voiceprint，没法区分 user。Speech 里有，但需要外部 knowledge base 来 retrieved。

### 5.2 机制

```
speech → speaker identification module → voiceprint (d-vector / x-vector)
       ↓
   query external database (user-specific knowledge base)
       ↓
   retrieved text knowledge
       ↓
   Tok_l tokenize → concat 进 LLM input
```

- Speaker identification 用 **pre-trained** 模块（论文没明说，但业界标准是 ECAPA-TDNN (https://arxiv.org/abs/2005.07143) 或基于 WavLM (https://arxiv.org/abs/2106.08422) 的 speaker embedding）
- Voice RAG 的输出是文本，不是 embedding，所以可以被 LLM 直接当 token 处理
- 这一步是 training-free 的：retrieval 数据库可以动态更新，模型不用 re-train

### 5.3 三类 customized task

1. **Object Ownership**：根据 voice 识别用户，pick 属于他的物品
2. **User Preference**：同一条指令，对不同 user 执行不同 action
3. **Compound**：ownership + preference 组合
4. **Compound-Multistage**：两阶段任务，前序结果影响后序

---

## 6. Experiments 深度分析

### 6.1 CALVIN Benchmark（Table 1）

CALVIN 是 long-horizon manipulation benchmark，每个 task 5 个连续 sub-task。
- LH-1 到 LH-5 表示完成 1-5 个连续 sub-task 的 success rate
- Len 表示平均完成 sub-task 数量

关键对比：

| Model | LH-1 | LH-5 | Len |
|-------|------|------|-----|
| MCIL+ | 37.3% | 0.0% | 0.40 |
| HULC+ | 89.2% | 33.5% | 2.90 |
| RT-1+ | 61.7% | - | 2.45 |
| RoboFlamingo+ (with LSTM history) | 96.4% | 66.0% | 4.09 |
| RoboFlamingo+ ASR* | 89.8% | 48.3% | 3.41 |
| VLA+ASR* | 88.7% | 40.2% | 3.13 |
| **VLAS+** (text) | 94.5% | 56.6% | 3.74 |
| **VLAS*** (synthetic speech) | 94.2% | 54.6% | 3.70 |
| **VLAS*(Real)** (real human speech) | 93.6% | 51.3% | 3.61 |

关键观察：
1. **VLAS+ ≈ VLAS***：text 和 speech 性能几乎持平，说明 speech 通路没有造成 performance drop。这是一个 strong result，证明端到端 speech integration 是 viable 的。
2. **VLAS* > VLA+ASR***：端到端比 cascaded pipeline 强 ~10% LH-5。这验证了 paper 的核心 hypothesis：external ASR 在 controlling command 上不够敏感，造成 error propagation。
3. **Real speech 略低于 synthetic**：0.19 Len gap，这个 gap 主要来自 TTS-synthetic 训练和 real speech 的 domain shift。可以理解：训练数据全是 VITS 生成的，real recording 有 noise、reverberation、accent。
4. **VLAS+ vs RoboFlamingo+**：RoboFlamingo 用了 LSTM policy head 来 encode history，VLAS 没用 history。Table 5 显示去掉 LSTM 后 RoboFlamingo 也跌到 ~60% LH-1，所以 VLAS 在无 history 情况下接近有 history 的 RoboFlamingo。

### 6.2 Customized Tasks Benchmark（Table 2）

这是 paper 的 highlight：

| Model | Ownership | Preference | Compound | Compound-M1 | Compound-M2 | Avg |
|-------|-----------|------------|----------|-------------|--------------|-----|
| VLA+ (text only) | 17.9% | 30.8% | 23.1% | 35.9% | 5.1% | 19.2% |
| VLAS* | 94.7% | 84.6% | 100.0% | 100.0% | 66.7% | 86.5% |
| VLAS*(Real) | 89.5% | 70.0% | 100.0% | 90.0% | 55.0% | 78.6% |
| VLAS*-RAG (ablation) | 15.4% | 12.8% | 25.6% | 33.3% | 10.3% | 16.0% |
| VLA+RAG (text + RAG) | 97.4% | 84.6% | 97.4% | 82.1% | 48.7% | 82.0% |

关键 ablation：
- **VLAS* vs VLAS*-RAG**：去掉 Voice RAG，VLAS 从 86.5% 暴跌到 16.0%。这说明 speech 本身（不带 voiceprint-retrieved knowledge）几乎不能解决 personalization 任务。Voice RAG 是 necessary 的。
- **VLA+ vs VLA+RAG**：纯 text + RAG 也能做到 82.0%，但需要 manually 把 user ID 喂给 RAG query（人工 ground truth 标注）。VLAS* 自动从 voice 提取 user ID，所以是真正 end-to-end。
- **Compound-M2 (stage-2)** 是 hardest：VLAS* 66.7%，因为 stage-1 的失败会 propagate 到 stage-2，且 stage-2 需要更复杂的 reasoning。

### 6.3 VLAS-Base 评估（Table 3, Table 4）

VLAS-Base 是 Stage II 产物，是 speech-enabled LLaVA。在 general VQA benchmark 上：

| Model | VQAv2 | GQA | POPE | SQA1 | VizWiz | VQAT |
|-------|-------|-----|------|------|--------|------|
| LLaVA v1.5 | 78.8 | 62.0 | 85.9 | 66.8 | 50.0 | 58.2 |
| VLAS-Base | 78.7 | 62.0 | 85.5 | 72.2 | 51.1 | 58.1 |

VLAS-Base 在 general VQA 上几乎和 LLaVA 持平，SQA1 还略高（72.2 vs 66.8，可能受益于 multi-task co-training 的 regularization effect）。

在 speech benchmark 上：

| Model | LibriSpeech WER | SGQA |
|-------|-----------------|------|
| Whisper large-v2 | 2.7% | N/A |
| VLAS-Base | 2.79% | 50.8 |
| LLaVA (text GT) | N/A | 62.0 |
| BLIP-2 (text GT) | N/A | 41.0 |

VLAS-Base 在 LibriSpeech 上 WER 2.79%，接近 Whisper 2.7%。考虑到 speech spectrogram 被 reduction factor=5 downsample，这个 ASR 性能已经非常接近 SOTA。SGQA 比 LLaVA-with-text 低（50.8 vs 62.0），但仍优于 BLIP-2，说明 speech-to-vision 的 cross-modal reasoning 还有提升空间。

---

## 7. Inference Efficiency（Table 9, 10）

| Model | r | Hz | Len |
|-------|---|----|-----|
| VLA+ | 1 | 1.89 | 2.30 |
| VLA+ | 5 | 3.60 | 3.80 |
| VLAS* | 1 | 1.17 | 2.02 |
| VLAS* | 5 | 2.50 | 3.70 |
| VLAS* | 12 | 2.88 | 3.35 |
| VLAS* | 20 | 3.80 | 0.70 |

VLAS* r=1 是 1.17 Hz，比 VLA+ r=1 (1.89 Hz) 慢，因为多了 speech encoder 前向。但 r=5 时 VLAS* 也达到 2.50 Hz，可以接受。r=20 性能崩塌说明 prediction horizon 太长会引入 distribution shift。

2.5 Hz 对 real robot 是 marginal 的。对比 π0 (Physical Intelligence, https://physical.company/) 在 ALOHA 上能跑到 50+ Hz，QUART-Online (https://arxiv.org/abs/2412.15576) 用 early-exit 优化到 10+ Hz，VLAS 还需要 inference optimization 才能实用化。可能的方向：Mamba backbone (Cobra, https://arxiv.org/abs/2403.14520)、KV cache 优化、speculative decoding、action chunking with diffusion head (类似 π0)。

---

## 8. 与相关 Work 的联系与思考

### 8.1 在 VLA 谱系中的位置

VLA 发展路径：
- **Plan-then-Execute**: PaLM-E (https://arxiv.org/abs/2303.03378), SayCan (https://arxiv.org/abs/2204.01691), Code as Policies (https://arxiv.org/abs/2209.07753)
- **End-to-End VLA**: RT-2, OpenVLA, RoboFlamingo, 3D-VLA (https://arxiv.org/abs/2403.09631), QUAR-VLA (https://arxiv.org/abs/2407.04525), Deer-VLA (NeurIPS 2024)
- **Real-time VLA**: QUART-Online, HRVLA
- **Speech-augmented VLA**: VLAS（this paper）

VLAS 是第一个把 speech 端到端接入 VLA 的工作。之前 MUTEX (https://arxiv.org/abs/2310.01396) 做了 multimodal task specification 但没 leverage VLM 能力。

### 8.2 和 Audio-LLM 的关系

GPT-4o (https://openai.com/index/hello-gpt-4o/) 和 Gemini (https://arxiv.org/abs/2312.11805) 已经做了 native omni-modal，但是 closed。Open-source 侧：
- SALMONN (https://arxiv.org/abs/2310.10906)：speech-text LLM
- VITA (https://arxiv.org/abs/2408.05211)：开源 omni-modal
- Qwen-Audio, Speech-LLaVA, ImageBind-LLM (https://arxiv.org/abs/2309.03905)

VLAS 的 speech integration 思路和 SALMONN 很像（用 Whisper encoder + MLP projection + LLM），区别是 VLAS 还接了 vision 和 action。

### 8.3 与 RAG for Robotics 的联系

Robotics RAG 是个 emerging direction。Code as Policies 实际上就是简易 RAG（API retrieval）。最近 RoboGen (https://arxiv.org/abs/2311.01511) 用 LLM 自动生成 skill library。VLAS 的 Voice RAG 把 retrieval key 换成 voiceprint，这是一个很 clever 的设计——voiceprint 是天然 user ID。

### 8.4 Personalization 的其他路径

VLAS 通过 Voice RAG 做 personalization，但还有其他路径：
- **In-context learning**：把 user profile 写进 prompt（但 speech 没法直接 in-context）
- **LoRA fine-tuning per user**：贵
- **Meta-learning**：复杂
- **Voice RAG**：training-free，可动态更新，但需要预先构建 knowledge base

Voice RAG 的 trade-off：retrieval 质量依赖 database 完整性，对小 user pool 工作好，对 open-vocabulary user 不 work。

### 8.5 Architecture 的可改进方向

1. **Speech encoder**：Whisper 是为 ASR 训练的，hidden state 偏 phonetic。换 WavLM (self-supervised, richer prosody info) 可能更好
2. **Reduction factor**：reshape 是粗暴的 pooling。Q-Former learnable queries 更优雅
3. **No history**：CALVIN LH-5 性能受限于无 history modeling。可以加 Mamba state 或 LSTM policy head（如 RoboFlamingo）
4. **Action head**：discrete tokenization 限制了精度。可以用 diffusion head (π0) 或 flow matching
5. **Real-speech domain gap**：训练全是 TTS，可以加 noise augmentation、real speech co-training

### 8.6 失败案例（Appendix B.1）

VLAS 失败集中在：
- Preference task 第二阶段
- Compound task 第二阶段

模式一致：模型理解指令但执行失败。说明 policy model 的 low-level control 还不够强，和 OpenVLA-OFT (https://arxiv.org/abs/2412.04014) 这类精调 work 的方向是一致的。

VLA+ 失败模式多样：模型完全没理解指令，靠 random trial。

---

## 9. Intuition 总结

VLAS 的核心 insight 是：**speech 在 robot interaction 中不仅是一个 modality 入口，它还携带 user identity**。传统 ASR-VLA pipeline 把 speech 当成 "text 的另一种 encoding"，丢掉了 voiceprint 这个信号。VLAS 把 voiceprint 重新捡回来，并配上 retrieval 注入 personalized knowledge。

这种设计 philosophy 可以推广：
- 表情摄像头 + facial identity → emotion-aware VLA
- 可穿戴 IMU + gait identity → habit-aware VLA
- 多人环境 + speaker diarization → multi-user VLA

更深远的影响：当 robot 进入 home care 场景，personalization 不再是 luxury，而是 necessity。VLAS 提供了一个 lightweight、training-free、可扩展的 RAG 框架，这个思路可以无缝迁移到其他 identity-carrying modality。

---

## Web Links 汇总

- VLAS GitHub: https://github.com/whichwhichgone/VLAS
- LLaVA: https://arxiv.org/abs/2304.08485
- Whisper: https://arxiv.org/abs/2212.04356
- CLIP: https://arxiv.org/abs/2103.00020
- Vicuna: https://lmsys.org/blog/2023-03-30-vicuna/
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboFlamingo: https://arxiv.org/abs/2311.01378
- CALVIN: https://github.com/mees/calvin
- ESPnet: https://github.com/espnet/espnet
- VITS: https://arxiv.org/abs/2106.06103
- LibriSpeech: https://www.openslr.org/12
- LibriTTS: https://www.openslr.org/60
- Behavior cloning (DAGGER): https://arxiv.org/abs/1011.0686
- PaLM-E: https://arxiv.org/abs/2303.03378
- SayCan: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753
- 3D-VLA: https://arxiv.org/abs/2403.09631
- MUTEX: https://arxiv.org/abs/2310.01396
- BLIP-2: https://arxiv.org/abs/2301.12597
- GPT-4o: https://openai.com/index/hello-gpt-4o/
- Gemini: https://arxiv.org/abs/2312.11805
- VITA: https://arxiv.org/abs/2408.05211
- SALMONN: https://arxiv.org/abs/2310.10906
- ImageBind-LLM: https://arxiv.org/abs/2309.03905
- WavLM: https://arxiv.org/abs/2106.08422
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- Cobra (Mamba-VLM): https://arxiv.org/abs/2403.14520
- QUART-Online: https://arxiv.org/abs/2412.15576
- QUAR-VLA: https://arxiv.org/abs/2407.04525
- ACT (Action Chunking): https://arxiv.org/abs/2304.13705
- Mamba: https://arxiv.org/abs/2312.00752
- ViT: https://arxiv.org/abs/2010.11929
- LLaMA: https://arxiv.org/abs/2302.13971
- OpenVLA-OFT: https://arxiv.org/abs/2412.04014
- RoboGen: https://arxiv.org/abs/2311.01511

---

如果你想往哪个方向深挖（比如 speech encoder 的替代设计、Voice RAG 的 retrieval mechanism、action chunking 的理论分析、或者和 π0 / Helix 的对比），我可以继续展开。
