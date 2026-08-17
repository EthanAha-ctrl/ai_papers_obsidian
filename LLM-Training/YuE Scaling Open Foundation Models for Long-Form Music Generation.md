---
source_pdf: YuE Scaling Open Foundation Models for Long-Form Music Generation.pdf
paper_sha256: 1dc99a558b723a3cd2f182e17fd476db0953cde21be78e26f881284eb8ec7391
processed_at: '2026-08-13T06:45:24-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# YuE 人话版：这篇 paper 在干嘛

Andrej，我把 paper 的核心思路用大白话讲一遍，重点讲 **why** 而不只是 **what**。

---

## 一句话总结

**YuE = 拿 LLaMA2 架构，把"生成一首完整的歌"当成 next-token prediction 来做，通过几个关键 trick 让它能撑到 5 分钟且歌词对得上。**

听起来简单，但里面有一堆"坑"需要绕过去。paper 的价值就在于把这些坑都踩过一遍然后告诉你怎么绕。

---

## 1. 为什么这个任务难（直觉版）

想象你要让一个 LM 生成一首 3 分钟的歌。这首歌里有：
- 人声在唱歌词
- 吉他在弹和弦
- 鼓在打节奏
- 贝斯在走 bass line

所有这些声音混在一起变成一个 waveform，然后你要把这个 waveform 压成一串 discrete tokens，让 LM 一个 token 一个 token 地 predict。

问题来了：**一个 token 要同时表示"人声 + 吉他 + 鼓 + 贝斯"在这一瞬间的混合状态**。

这就好比让你用一句话同时描述"厨房里在炒菜 + 客厅里在放电视 + 楼上在洗澡"——你能描述，但信息量太大，容易丢东西。

### 丢什么？丢 lyrics

作者用一个非常 clever 的实验证明这点：他们用 Whisper 跑原始 audio 的 WER，再跑 tokenizer 重建后的 audio 的 WER，差值叫 $\Delta$WER。

结果：**metal > pop > hip-hop**，mixture > vocal-only。

直觉解释：metal 里吉他 distortion + 双踩鼓的 energy 远超人声，tokenizer 的 RVQ 码本会被高 energy 乐器"占满"，vocal 信息被挤掉。这就是为什么之前的 music LM 在 metal 这种 genre 上 lyrics 跟随特别烂。

---

## 2. 第一个核心 idea：Dual-NTP

### 直觉

既然一个 token 装不下两个信号，那就**用两个 token**：一个专门表示 vocal，一个专门表示 accompaniment。

这听起来很 trivial，但关键在于"怎么做"。

之前的工作（SongCreator, MelodyLM）要么改架构，要么先生成完 vocal 再生成 accompaniment（sequential），都有各种问题。

YuE 的做法：每个时间步输出两个 token，按 vocal-accomp-vocal-accomp 交织排列：

$$\big(v_1, a_1, v_2, a_2, \ldots, v_T, a_T\big)$$

然后在 LM 内部用 standard AR factorization，但有个细节：**vocal token 先生成，accompaniment token 生成时已经看到当前帧的 vocal**。

公式 (6)：

$$P(v_t, a_t \mid v_{<t}, a_{<t}) = p(v_t \mid v_{<t}, a_{<t}) \times p(a_t \mid v_{\leq t}, a_{<t})$$

注意第二项是 $v_{\leq t}$ 而不是 $v_{<t}$。这个细节保证 accompaniment 在同一帧内"知道"vocal 在做什么，避免两个 track 打架。

### 为什么这个比 sequential 好

Sequential（先全部 vocal，再全部 accompaniment）的问题：两个 track 在时间上不对齐，模型很难让鼓点对上人声的节奏。

Dual-NTP 让 vocal 和 accompaniment 在每个 frame 都 jointly contextualize，一次 forward pass 就搞定，synchronization 天然对齐。

### 为什么这个比改架构好

保持 standard LM 架构 = 可以直接用 LLaMA2 的所有 infrastructure（Megatron、tensor parallel、flash attention 等）。Scaling 时不用重新造轮子。这个工程层面的考虑在 7B model + 1.75T tokens 的 scale 下很重要。

---

## 3. 第二个核心 idea：CoT（Structural Progressive Conditioning）

### 问题

这个是 paper 里我觉得最有 insight 的部分。

背景：TTS 系统通常 30 秒以内，TTM 也差不多。要扩到 5 分钟，第一反应是"把 context window 拉长"——比如从 4K 拉到 16K。

但作者发现：**即使 context window 是 16K，lyrics conditioning 仍然在 3K tokens 之后开始 degrade，6K 之后完全 fail**。

为什么？**RoPE 的 long-term decay property**。

RoPE 本质上给 attention 加了一个"距离衰减"权重——越远的 token，attention weight 越小。这是设计特性，不是 bug。

问题在于：lyrics 在歌曲开头作为 conditioning 输入，然后模型要生成几千个 audio tokens。当 audio tokens 累积到 3K+ 时，开头的 lyrics 在 attention 里已经"淡出"了，模型基本"忘了"歌词是什么。

### 尝试过的 failed 方案

- 增大 RoPE base (10K → 100K)：这是社区常用的 "ABF" (Adjusted Base Frequency) trick，理论上能扩展有效 attention range。**无效**。
- Curriculum learning（先学短的，再逐渐学长的）：**无效**。
- Vanilla text prepend：**无效**。

作者发现这些"技术性"方案都不行，因为问题本质是 attention decay，不是 position encoding 的数值表达。

### CoT 的 insight

**既然 lyrics 在长距离会衰减，那就别让它衰减——把 lyrics 重新塞回去。**

利用 music 的天然结构：一首歌 = intro + verse + chorus + bridge + outro，每段通常 < 30 秒。作者用 all-in-one 工具自动把歌曲切成 ~14 个 sections。

然后数据组织成这样：

```
[Instruction] [Tags] [Raw Lyrics]
  [START_OF_SEGMENT] [verse] [lyrics for verse 1] [SOA] [audio tokens] [EOA] [END_OF_SEGMENT]
  [START_OF_SEGMENT] [chorus] [lyrics for chorus] [SOA] [audio tokens] [EOA] [END_OF_SEGMENT]
  ...
[EOD]
```

**关键**：每个 section 前面都重新放一遍这段的 lyrics。这样模型在生成每个 section 的 audio 时，lyrics conditioning 都在局部 attention window 内（< 3K tokens），不会衰减。

直觉类比：你在写一篇长论文，如果只在开头写一次大纲，写到第 10 页时你已经忘了大纲是什么。但如果你每写完一章就重新看一遍大纲，就能始终保持对齐。

### 这个 trick 为什么 elegant

- 不需要改架构
- 不需要改 position encoding
- 不需要额外参数
- 利用了 domain prior（music 本来就有结构）
- 解决了长距离 conditioning 的根本问题

### Ablation 数据

Figure 12：0.5B model 上，CoT 在 30s-150s 所有时间区间 WER 都显著优于 Vanilla / Curriculum / ABF。Scaling 到 7B，WER 从 ~70% 降到 ~20%。

---

## 4. 第三个核心 idea：重新设计 ICL

### 传统 speech ICL 的问题

VALL-E / CosyVoice 的 ICL 长这样：

```
[ref text] [input text] [ref audio] [generated audio]
```

即：给一段参考音频 + 参考文本，让模型生成 input text 对应的音频。

用到 music 上有三个问题：

1. **需要 ref text**：music 场景下 lyrics 经常拿不到
2. **单向 continuation**：只能往后生成，不能"给你一段 chorus，帮我写整首歌"
3. **Entanglement**：reference 和 generated 紧耦合，模型容易直接 copy（版权问题）

### YuE 的重新设计

格式：

$$\mathcal{D}_{icl} = A_{ref} \circ \mathcal{D}_{cot}$$

即：随便采样 20-40s 的 reference audio，直接 prepend 到 CoT 数据前面。

支持两种模式：
- **Single-track**：reference 可以是 vocal / accompaniment / mixture
- **Dual-track**：分离的 vocal 和 accompaniment 按 token 交织

### Delayed Activation：这是个重要的教训

作者发现一个反直觉现象：**ICL 数据如果加得太早，模型会"学坏"**。

具体来说：模型发现"直接 copy reference audio"是 shortcut，比"理解 lyrics 然后生成"容易得多。于是模型就学会 copy，lyrics conditioning 完全失效。

更糟糕的是：**这个 shortcut learning 是不可逆的**。即使后续移除 ICL 数据继续训练，模型也回不来了——它只会输出 noise 或 silence，因为"copy"这条路被切断后，它连基本的生成能力都丢了。

这个发现本身就很值得发一篇 paper。它说明：

- **"容易"的数据是危险的**：ICL 让 loss 降得快，但模型走的是捷径
- **Curriculum 很重要**：什么时候给什么数据，比给多少数据更重要
- **Shortcut learning 在 generative model 里是 real issue**：不只是 classification 的现象

### 解决方案

只在 annealing phase（最后 40B tokens，~2% compute）才加 ICL 数据。

结果：模型学会了"用 reference 的 style/timbre，但生成新内容"。比如给一段 Japanese city pop，模型可以转成 English rap 但保留 city pop 的 accompaniment 风格。

---

## 5. Stage-2：为什么需要两阶段

### 问题

Stage-1 用 codebook-0（semantic-rich），能保证 lyrics 跟随和 musical structure，但 audio 质量不够——只有 1 个 codebook，相当于极低 bitrate。

要补全 acoustic details，需要 codebooks 1-7（共 8 个 codebook）。

### 为什么不直接在 Stage-1 就预测 8 个 codebook

- 8 个 codebook 意味着每帧 8 个 token，sequence 变长 8 倍
- 计算量爆炸
- 更重要：Stage-1 的核心任务是 lyrics-to-song 的 semantic alignment，引入 acoustic details 会干扰这个目标

### Stage-2 的设计

Stage-2 是一个 1B 参数的小 LM，接收 Stage-1 的 codebook-0，预测 codebooks 0-7（注意：包括 codebook-0，虽然推理时不用）。

训练时序列组织：

```
[x1_0, x2_0, ..., xT_0,  # 先放所有 codebook-0
 x1_0, x1_1, ..., x1_7,  # 再放 frame 1 的 8 个 codebook
 x2_0, x2_1, ..., x2_7,  # frame 2
 ...
 xT_0, xT_1, ..., xT_7]  # frame T
```

**关键直觉**：把所有 codebook-0 放在开头，让模型先"看到"完整的 semantic outline，再补 residual details。这是 "plan-then-execute" 思路。

推理时 codebook-0 被 clamp（来自 Stage-1），只生成 codebooks 1-7。

### 为什么 training 要 include codebook-0

虽然推理时 codebook-0 来自 Stage-1，但 training 时如果把 codebook-0 也作为预测目标，模型会学到 codebook-0 和 residuals 之间的依赖关系。这样推理时 clamp codebook-0 后，模型能合理推断 residuals。

---

## 6. Tokenizer：为什么 X-Codec

### 失败经验

作者试了 4 个 tokenizer：

| Tokenizer | Type | Reconstruction | LM 能收敛吗 |
|-----------|------|----------------|-------------|
| EnCodec32k | Acoustic | Good | **No** |
| HiFiCodec | Acoustic | Good | **No** |
| SemantiCodec | Semantic+Acoustic | Fair | Yes |
| X-Codec | Semantic+Acoustic | Fair | Yes |

**关键发现**：纯 acoustic tokens 在大规模 in-the-wild 数据上让 LM 完全无法收敛，即使 7B model + 1T tokens 也只有 intermittent 成功，输出基本是 noise。

这反驳了 MusicGen 的假设——MusicGen 用 EnCodec 在相对干净的 MTG-Jamendo 上能 work，但数据规模和复杂度上去之后就崩了。

### 为什么 acoustic tokens 不行

作者的分析：
- Acoustic tokens 优先 compression efficiency，capacity 有限
- Token 表示 lossy，semantic relevance 低
- 模型倾向走 shortcut（直接 copy）
- In-the-wild 数据太复杂，acoustic tokens 的 representation 不够稳定

### 为什么 semantic-acoustic fusion 行

X-Codec 把 HuBERT 的 semantic representation 融合到 codec latent space。codebook-0 捕获 semantic info（melody, vocal content），让 LM 有一个稳定的"语义锚点"可以 converge。

Trade-off：reconstruction 质量从 "Good" 降到 "Fair"，但 LM 能收敛了。这个 trade-off 在 generative setting 下是值得的——你宁可用一个能收敛的 tokenizer 生成稍微糙一点但 semantic 对的 audio，也不想要一个 reconstruction 完美但 LM 学不会的 tokenizer。

---

## 7. Training Strategy：四个 phase 的设计逻辑

### Phase 1: Warm Up (280B tokens)
- 只用 English + Chinese（最高质量）
- 短 context (8K)
- Linear LR warmup
- 目标：建立基本音乐生成能力

### Phase 2: Constant LR (1T tokens)
- 加入 multilingual + lower quality data
- old:new = 2:1 防 distribution shift
- 目标：扩展数据覆盖

### Phase 3: Context Extension (750B tokens)
- Context 8K → 16K
- 移除 single-track unconditional
- 目标：长程依赖

### Phase 4: Annealing (40B tokens, ~2% compute)
- 移除 speech + unconditional music
- 加入 ICL, gender tags, vocal timbre tags
- 用 20K hours 高质量数据
- CoT:ICL = 2:1
- 目标：注入控制能力

### 为什么这样设计

几个直觉：

**1. 简单到复杂**：先学高质量数据建立基础，再加 noise data 扩展泛化。

**2. Annealing 是 leverage 最高的阶段**：只花 2% compute 就 enable 所有 control signals。这和 LLM 里的 "instruction tuning 是 leverage 最高的阶段" 类似。

**3. ICL 必须最后加**：前面讲过，过早加 ICL 导致 shortcut learning 不可逆。

**4. BPM control 被移除**：作者发现 BPM 和 lyrics length 耦合，加 BPM control 反而 degrade lyrics following。这是个反直觉发现——你以为加更多 control 总是好的，但其实 control 之间可能冲突。

---

## 8. Multitask Learning：为什么不能只训 lyrics-to-song

### 问题

Paired lyrics-audio 数据少且 noise 大（web 爬取的 lyrics 经常对不上 audio）。如果只训这个任务，模型会 overfit 到 noise patterns。

### 解法

把 lyrics-to-song 能力分解为四个 sub-capability，每个用对应的任务训练：

1. **TTS** → 学 "text → human vocal" 的 alignment
2. **Music generation** → 学 musicality 和 style
3. **Lyrics-to-song (CoT)** → 整合 1 + 2
4. **ICL** → 学 style/voice transfer

### TTS 的量很重要

太多 TTS → 模型偏向 rap（rap 接近 speech，没有 melodic variation）。
太少 TTS → lyrics 跟随差。

作者找到一个 sweet spot：Music:Speech = 10:1。这个比例可能是经验调出来的，没什么理论依据，但 work。

---

## 9. 实验结果解读

### Human Evaluation (Figure 6)

YuE vs 四个 closed-source：

- **Suno V4**：仍然 SOTA，YuE 落后
- **Udio**：comparable
- **Tiangong**：comparable  
- **Hailuo**：YuE 胜出

作为 open-source model，YuE 能 match Udio 和 Tiangong 已经很 impressive。Suno V4 仍然领先，主要在 acoustic quality 上。

### Vocal Agility (Figure 8)

YuE 的 vocal range 中位数 ~27 semitones，与 Suno V4 持平，远超 Tiangong (~20) 和 Hailuo (~20)。

27 semitones ≈ 2 个多 octave，说明 YuE 的 vocal expressiveness 很强——能唱高音也能唱低音，不是单调的"念歌词"。

### Duration (Figure 9)

YuE 生成最长 audio，duration range 最宽。这直接验证了 CoT 的 long-form 能力。

### Controllability (Figure 7 Right)

YuE 在 genre control, instrument/vocal control, emotion 上表现最强。这验证了 multitask training + CoT + delayed ICL 的组合效果。

### Metric 相关性发现（重要！）

这是 paper 里最 interesting 的 meta-finding：

**Vocal Range 与 Musicality 的 Pearson correlation = 0.857**

这意味着：**vocal range 是 musicality 的强 proxy**。你想快速评估一个 music generation model 好不好，量它的 vocal range 就行，不用做 expensive 的人工评估。

**CLaMP 3 >> CLAP**

CLaMP 3 与所有 controllability 维度都正相关（0.33-0.44），CLAP 基本不相关甚至负相关（-0.25 到 0.14）。

直觉：CLAP 训练时 exposure 不足 singing/music content，在 music 场景下不可靠。CLaMP 3 用 web-scale 训练，更鲁棒。

**KL/FAD 与 acoustic quality 弱相关**

KL 和 FAD 与 VocalQual/AccompQual 的 correlation 只有 -0.15 到 0.23。这意味着**分布匹配类 metric 不能反映 acoustic fidelity**。可能的原因：PaSST backbone 在 generative music 上 OOD，sample size bias。

---

## 10. Emergent Abilities（最 cool 的部分）

Scaling 到 7B 后，YuE 自发涌现一堆能力：

### Vocal Techniques
- Vibrato（颤音）
- Glissando（滑音）
- Bel canto（美声）
- Death growl（死嗓，metal 里那种）
- Mix voice, belting
- Riffs and runs
- Vocal fry
- **京剧唱腔**
- **陕北民歌**

这些都不是 explicitly 训练的，是从 in-the-wild 数据里自发学到的。这说明 Dual-NTP 确实捕获了 vocal performance 的细微差异。

### Spontaneous Performance
- Jazz：lyrics 用完后自动 scat singing
- A cappella：自动生成多声部 harmony
- Folk：vocal 间隙自动插入 harmonica solo

### World Music Fusion
- Chinese gangsta rap + Japanese shamisen
- 融合 Chinese opera + Shanbei folk + traditional Chinese vocals

### Voice Cloning
成功复制 Billie Eilish 和 王菲 的音色，保留 timbre、breathy texture、emotional nuance。

### Style Transfer
Japanese female J-pop → English male rap with city pop accompaniment。不仅换 language 和 vocal timbre，还调整 prosody 和 phrasing 保持 style coherence。

---

## 11. Memorization Check

用 ByteCover2 retrieval model 检测 ICL mode 是否复制训练数据。

结果：YuE 生成的音频与训练集的 cosine similarity 显著低于 Covers80（已知翻唱），与 GTZAN（genre-level 相似度）相当。

结论：**YuE 不会大规模复制训练数据**，即使 ICL 给了 30s reference。模型学的是"recombine patterns"而非"copy"。

这对版权问题很重要——ICL mode 下生成的内容可以认为是 original。

---

## 12. Unsuccessful Attempts（最有价值的部分）

paper 专门有一节讲失败经验，这在 AI paper 里很罕见，值得仔细看。

### 12.1 Acoustic Tokens 完全失败

即使在 7B model + 1T tokens 的 scale 下，EnCodec/HiFiCodec 的 acoustic tokens 仍然无法让 LM 收敛，输出基本是 noise。

教训：**tokenizer 的 representation quality 决定了 LM 能否 scale**。再大的 model 也救不了信息丢失的 tokenizer。

### 12.2 大模型 Unconditional Pre-train 反而有害

小 model（sub-billion）unconditional pre-train 后 finetune 能 work。但 7B model 上 unconditional pre-train 后 finetune 反而建立不了 cross-modal alignment。

作者称为 **"catastrophic inertia"**——大模型内化的 generic prior 太强，finetune 时"扭不过来"。

这个发现和 LLM 里"instruction tuning 在更大 base model 上效果递减"的观察类似。直觉：大模型是"重型卡车"，一旦高速行驶就难拐弯。

### 12.3 Early ICL = 不可逆灾难

前面讲过，再强调一遍：**ICL 数据加太早 → shortcut learning → 移除 ICL 也救不回来 → 模型只会输出 noise/silence**。

这个不可逆性很 scary。说明 generative model 的 training dynamics 里有"point of no return"——一旦学错就回不去。

---

## 13. 我的几个直觉总结

### 13.1 Domain Prior > 通用技术

CoT 解决 long-context 问题，不是靠更聪明的 position encoding 或 attention mechanism，而是靠 music 本身的结构先验。这暗示我们在做 domain-specific 任务时，**先理解 domain structure 比堆技术更有 leverage**。

### 13.2 Tokenizer 是 Bottleneck

YuE 在 acoustic quality 上仍落后 Suno V4，作者自己也承认主要是 tokenizer 限制。Semantic-acoustic fusion 解决了 convergence 问题，但引入了 reconstruction quality 的 trade-off。

下一代 music LM 可能需要：
- 更好的 semantic-acoustic fusion（比如 multi-scale）
- 或者直接用 continuous representation（不走 discrete tokens）
- 或者用 diffusion 做 post-processing（类似 Suno 可能的做法）

### 13.3 Training Data 顺序 = Hyperparameter

Delayed ICL activation, multiphase training, CoT:ICL = 2:1, Music:Speech = 10:1——这些"时机和比例"的选择对最终效果影响巨大，比 LR, batch size 这些传统 hyperparameter 重要得多。

这和 LLM 里的发现一致：**data curriculum 是最重要的 hyperparameter**，但最难调，因为没法 grid search。

### 13.4 Shortcut Learning 是 Generative Model 的隐形杀手

ICL 过早导致 shortcut learning 这个发现很深刻。它说明：

- Loss 下降快 ≠ 学到对的东西
- "容易"的数据可能有害
- 不可逆性让这个问题更严重

在 LLM 里我们见过类似现象：如果 pretrain 阶段就混入太多 instruction data，model 会学"捷径"而不是"理解"。

### 13.5 Evaluation Metric 需要重新审视

- CLAP 在 music 场景下不可靠
- KL/FAD 与 acoustic quality 弱相关
- Vocal Range 是 musicality 的强 proxy（r=0.86）

这说明我们之前对 music generation 的 automatic evaluation 可能很多结论是错的。CLaMP 3 是个改进方向。

---

## 14. 与你工作的联想

Andrej，这篇 paper 让我想起你讲过的几个观点：

**1. "Software 2.0"**：YuE 把整个 music generation pipeline 塞进一个 AR LM，没有 separate modules for lyrics alignment, melody generation, arrangement——全是 next-token prediction。这就是 Software 2.0 的极致体现。

**2. Scaling law 的 domain-specific 表现**：0.5B → 7B 的 scaling 效果显著，emergent abilities（vibrato, scat singing）涌现。但 scaling 不是万能的——acoustic tokens 在 7B scale 仍然失败，catastrophic inertia 在 7B scale 才出现。Scaling 放大一切，包括失败模式。

**3. "Data is all you need"**：YuE 的核心技术贡献（Dual-NTP, CoT, delayed ICL）本质上都是 about "how to organize and present data to the model"。架构是 standard LLaMA2，没改。真正的工作在 data pipeline。

**4. Curriculum matters**：类似你讲 "micrograd" 时强调的 "learning rate schedule 和 data curriculum 是 DL 的 black magic"。YuE 的 four-phase training 和 delayed ICL activation 是这个思想的体现。

---

## 15. Web Links 汇总

### Project
- GitHub: https://github.com/multimodal-art-projection/YuE
- Demo: https://map-yue.github.io/

### Core References
- LLaMA2: https://arxiv.org/abs/2307.09288
- X-Codec: https://arxiv.org/abs/2408.17175
- AudioLM: https://arxiv.org/abs/2209.03143
- VALL-E: https://arxiv.org/abs/2301.02111
- RoPE: https://arxiv.org/abs/2104.09864
- Megatron-LM: https://arxiv.org/abs/1909.08053

### Comparisons
- SongCreator: https://arxiv.org/abs/2409.06029
- MusicGen: https://arxiv.org/abs/2306.05284
- Stable Audio Open: https://arxiv.org/abs/2407.14358
- SemantiCodec: https://arxiv.org/abs/2406.06965

### Evaluation
- MARBLE: https://arxiv.org/abs/2311.09615
- CLaMP 3: https://arxiv.org/abs/2502.10362
- CLAP: https://arxiv.org/abs/2209.03143
- Audiobox-Aesthetic: https://arxiv.org/abs/2502.05139
- ByteCover2: https://arxiv.org/abs/2203.02614

### Other
- Shortcut learning (Geirhos et al.): https://arxiv.org/abs/2007.07106
- CosyVoice: https://arxiv.org/abs/2407.05407
- EnCodec: https://arxiv.org/abs/2210.13438
- Vocos: https://arxiv.org/abs/2306.00814
- Qwen2-Audio: https://arxiv.org/abs/2407.10759

---

希望这版"人话"把核心 intuition 讲清楚了。这篇 paper 最 valuable 的部分不是架构（就是 LLaMA2），而是那几个 "why"——为什么要 dual token，为什么要 CoT，为什么要 delayed ICL，为什么 acoustic tokens 不行。这些 insight 才是能 transfer 到其他 domain 的 knowledge。

---

# YuE: Scaling Open Foundation Models for Long-Form Music Generation 深度讲解

Andrej, 这是一篇关于 lyrics-to-song generation 的重量级 paper。我尽量从架构设计直觉、公式变量含义、实验数据解读三个层面把这篇 paper 拆开来讲。

---

## 1. 总览：为什么这个任务如此困难

YuE 要解决的问题可以形式化为：给定 lyrics text $\ell$、style tags $\mathcal{T}$、可选 reference audio $A_{ref}$，生成完整歌曲 $(\mathbf{v}, \mathbf{a})$，其中 $\mathbf{v}$ 是 vocal track，$\mathbf{a}$ 是 accompaniment track，时长可达 5 分钟。

这个任务的难度可拆解为四个子挑战：

1. **Long-range dependencies**：pop song 通常 3-5 分钟，~15000-25000 audio tokens (50Hz frame rate)
2. **Signal complexity**：polyphonic，vocal + 多乐器同时进行
3. **Linguistic distortion**：singing 改变了 phoneme duration、prosody，与 spoken language 差异大
4. **Data scarcity**：paired lyrics-audio 数据少且噪声大

YuE 的核心思路：基于 LLaMA2 架构，把 lyrics-to-song 当成一个 next-token prediction 任务，通过几个关键创新解决上述挑战。

项目链接：
- GitHub: https://github.com/multimodal-art-projection/YuE
- Demo: https://map-yue.github.io/

---

## 2. 整体架构 (Figure 2 解析)

整个系统包含四个组件：

```
[Audio waveform] ──> [Audio Tokenizer (X-Codec)] ──> [Stage-1 LM (LLaMA2 7B)] 
                                                       ↓ codebook-0 tokens
                                                   [Stage-2 LM (1B)]
                                                       ↓ codebooks 0-7
                                                   [Vocoder (Vocos)]
                                                       ↓ 44.1kHz
                                                   [Audio output]
```

关键设计直觉：**two-stage 解耦**。Stage-1 专注于 semantic 层面的"乐句、歌词、结构"，Stage-2 专注于 acoustic 残差细节。这类似 LLM 中"概念先、细节后"的层级建模哲学，类比 VQ-VAE 的 hierarchical latent 或 AudioLM 的 semantic-acoustic 双阶段。

---

## 3. Stage-1: Music Language Modeling (MuLM)

### 3.1 Track-Decoupled Next-Token Prediction (Dual-NTP)

#### Standard NTP 的问题

经典 AR factorization 是：

$$p(\mathbf{x}_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t}; \theta) \quad \text{(1)}$$

变量说明：
- $\mathbf{x}_{1:T} = (x_1, x_2, \ldots, x_T)$ 是 audio token sequence
- $x_t$ 是第 $t$ 个时间帧的 token
- $T$ 是总帧数
- $\theta$ 是 LM 参数
- $x_{<t} = (x_1, \ldots, x_{t-1})$ 是历史 tokens

推理时：

$$\hat{x}_t = \arg\max_{x_t} p(x_t \mid x_{<t}; \theta) \quad \text{(2)}$$

这个公式对纯 TTS 或纯 TTM 可以收敛，但 lyrics-to-song 任务中，单个 token $x_t$ 必须同时编码 vocal 和 accompaniment 两种信号。当 accompaniment 能量远大于 vocal（例如 metal 音乐）时，token 倾向于"丢失" linguistic 信息。

#### LLAT 量化分析

作者用一个非常聪明的指标量化这个问题：**LLAT (Linguistic information Loss After Tokenization)**

$$\Delta \text{WER} = \text{WER}_{recon} - \text{WER}_{ori}$$

- $\text{WER}_{ori}$：fine-tuned Whisper 在原始 mixture audio 上的 WER
- $\text{WER}_{recon}$：fine-tuned Whisper 在 tokenizer 重建后的 audio 上的 WER
- 差值越大，说明 tokenizer 损失了越多 linguistic 信息

Figure 3 的结果：metal > pop > hip-hop，且 mixture 比 vocal-only 显著高。这个直觉很重要：**当 accompaniment 太响，tokenizer 的 RVQ 码本会优先编码 high-energy 的 accompaniment 而挤压 vocal 信息**。

#### Dual-NTP 解决方案

核心想法：**显式引入 source separation prior**。把每个时间帧拆成两个 token：

$$\big( \underbrace{v_1}_{\text{vocal}}, \underbrace{a_1}_{\text{accomp.}}, \underbrace{v_2}_{\text{vocal}}, \underbrace{a_2}_{\text{accomp.}}, \ldots, \underbrace{v_T}_{\text{vocal}}, \underbrace{a_T}_{\text{accomp.}} \big) \quad \text{(3)}$$

变量：
- $v_t$：第 $t$ 帧的 vocal token
- $a_t$：第 $t$ 帧的 accompaniment token

联合概率：

$$p(\mathbf{v}_{1:T}, \mathbf{a}_{1:T}) = \prod_{t=1}^{T} p(v_t, a_t \mid v_{<t}, a_{<t}; \theta) \quad \text{(4)}$$

推理时：

$$(\hat{v}_t, \hat{a}_t) = \arg\max_{(v_t, a_t)} p(v_t, a_t \mid v_{<t}, a_{<t}; \theta) \quad \text{(5)}$$

关键技巧在于虽然写成 joint，但可 chain 分解为：

$$P(v_t, a_t \mid v_{<t}, a_{<t}; \theta) = p(v_t \mid v_{<t}, a_{<t}; \theta) \times p(a_t \mid v_{\leq t}, a_{<t}; \theta) \quad \text{(6)}$$

注意第二个因子用了 $v_{\leq t}$ 而不是 $v_{<t}$——意思是**vocal token 先生成，accompaniment token 生成时已经看到当前帧的 vocal**。这个细节很重要，它保证 vocal 和 accompaniment 在同帧内有因果顺序，避免同步问题。

#### 与先前方法对比

SongCreator (Lei et al., 2024, NeurIPS) 和 MelodyLM (Li et al., 2024) 都尝试过 dual-track modeling，但通常需要修改 LM 架构或者顺序生成两个 track。YuE 的优势在于完全保留标准 LM 架构，可以直接复用 LLaMA2 的预训练基础设施，scaling 友好。

参考：
- SongCreator: https://arxiv.org/abs/2409.06029
- AudioLM (Borsos et al.): https://arxiv.org/abs/2209.03143

#### VAR 度量与 Dual-NTP 的 ablation

Section 8.2 引入 **Vocal-to-Accompaniment Ratio (VAR)**：

$$\text{VAR} = 10 \log_{10} \left( \frac{\sum_{n=1}^{N} (v(n))^2}{\sum_{n=1}^{N} (a(n))^2} \right) \quad \text{(8)}$$

变量：
- $v(n)$：vocal signal 在第 $n$ 个采样点的振幅
- $a(n)$：accompaniment signal 在第 $n$ 个采样点振幅
- $N$：总采样点数
- 单位 dB，更高 VAR 表示 vocal 更突出

Figure 10 显示：mixture track 在 VAR ~ -8dB 时 ΔWER 超过 20%，而 vocal track 始终保持低 ΔWER（最差 10%）。这说明 source separation prior 让 tokenizer 在低 VAR 场景下仍能保留 linguistic 信息。

Figure 11 显示 Dual-NTP 训练 loss 比 standard NTP 低约 0.4 nats，这在 LM 训练中是显著差异。

### 3.2 Structural Progressive Conditioning (CoT)

#### 长上下文的本质问题

这是 paper 中我觉得最有 insight 的部分之一。作者发现 RoPE（Rotary Position Embedding）的 long-term decay property 是 lyrics-to-song 的根本障碍：

- **3K tokens 开始 degrade**
- **6K tokens 完全失败**

即使预训练上下文扩展到 16K 也无济于事。试图调整：
- 增大 RoPE base：10K → 100K (类似 ABF, Xiong et al. 2023)
- Curriculum learning (逐步增长 audio length)

都无效。

这个发现很有意思，因为通常社区认为长 context 的瓶颈是 position encoding 表达能力，但 YuE 团队发现**即使位置编码理论上支持，attention 的有效作用范围仍然受限**——这是 attention decay 而非 position encoding 的问题。

#### CoT 的设计直觉

YuE 利用 music 的天然结构先验：songs 由 intro/verse/chorus/bridge/outro 组成，每个 section 通常 < 30 秒。

文档组织方式：

$$\mathcal{D}_{cot} = \underbrace{\text{Instruct} \circ \text{Tag} \circ \text{Lyrics} \circ (\bigcirc_{i=1}^{N} s_i) \circ <EOD>}_{\text{Prompt + segmented body}}$$

- $\circ$：sequence concatenation
- Instruct: `"Generate music from the given lyrics segment by segment."`
- Tag: 例如 `"[Genre] jazz male deep vocal romantic big band"`
- Lyrics: 原始未切分的 lyrics 文本
- $\bigcirc_{i=1}^{N} s_i$：N 个 segment 串联

每个 segment $s_i$ 的结构：

$$s_i = [START\_OF\_SEGMENT] \circ \tau_i \circ \ell_i \circ <SOA> \circ \psi_i \circ <EOA> \circ [END\_OF\_SEGMENT]$$

- $\tau_i \in \{[intro], [verse], [chorus], [bridge], [outro]\}$：structure label
- $\ell_i$：segment 对应的 lyric 文本
- $\psi_i$：Dual-NTP audio tokens
- $<SOA>$/$<EOA>$：start/end of audio

**核心直觉**：通过把 lyrics 重复地在每个 segment 前面"重新激活"，绕过了 RoPE 的 long-term decay。这相当于一种 "attention refresh"——每个 30 秒 segment 的 conditioning 在其局部 attention window 内仍然是 strong 的。

这个 idea 在精神上类似 Memformer、Compressive Transformer 等长序列建模思路，但更优雅，因为利用了 domain-specific structural prior。

参考：
- RoPE (Su et al.): https://arxiv.org/abs/2104.09864
- ABF (Xiong et al.): https://arxiv.org/abs/2309.16039
- all-in-one (Kim & Nam, WASPAA 2023): 用于自动 music segmentation

#### CoT Ablation (Figure 12)

设置：0.5B LM 在 500B tokens 上预训练，200B tokens finetune

对比方法：
- **Vanilla**：text prepend conditioning
- **Curriculum**：逐渐增长 conditioning duration
- **ABF**：RoPE base 10K → 100K
- **CoT**：本文方法

结果：CoT 在所有时间区间 (30s-150s) 显著优于其他方法。Vanilla 和 Curriculum 失败的主要原因是模型倾向于先生成 instrumental prelude，导致 singing onset 漂移，与原 prepended lyrics 不对齐。

**Scaling effect**：从 0.5B → 7B，WER 从 ~70% 降到 ~20%。这与 LLM 中 "more params, better long-range" 一致。

### 3.3 Music In-Context Learning

#### 传统 Speech ICL 的问题

TTS ICL 通常这样组织：

$$\underbrace{T_{ref}}_{\text{ref text}} \circ \underbrace{T_{input}}_{\text{input text}} \circ \underbrace{A_{ref}}_{\text{ref audio}} \circ \underbrace{A_{gen}}_{\text{gen audio}}$$

三个问题：
1. **Reference text 必需**：音乐场景下 lyrics 可能不可得
2. **单向 continuation**：限制 bidirectional creativity
3. **Entanglement**：reference 和 gen 紧耦合，可能复制（版权问题）

#### YuE 的重新设计

两种模式：
- **Single-track mode**：reference 可以是 accompaniment / vocal / mixture
- **Dual-track mode**：分离的 vocal 和 accompaniment，token-level interleaved（类似 Dual-NTP）

数据组织：

$$\mathcal{D}_{icl} = A_{ref} \circ \mathcal{D}_{cot}$$

随机采样 20-40s reference segment，prepend 到 CoT 数据前。

#### Delayed Activation 策略

这是个非常有 insight 的发现。ICL 数据是 "easy" data，过早加入会导致 **shortcut learning**（Geirhos et al., 2020）——模型直接 copy reference 而非创作。一旦发生 shortcut learning，**不可逆**：移除 ICL 数据继续训练，模型仍产生 invalid outputs（noise/silence）。

解决方案：只在 annealing phase（最后 40B tokens，~2% 计算预算）加 ICL 数据。

这个观察与 LLM pretraining 中 "数据顺序影响 in-context learning 能力" 的发现一致，也类似 instruction tuning 不能太早进行。

参考：
- Shortcut learning (Geirhos et al.): https://arxiv.org/abs/2007.07106
- VALL-E (Wang et al.): https://arxiv.org/abs/2301.02111
- CosyVoice (Du et al.): https://arxiv.org/abs/2407.05407

---

## 4. Stage-2: Residual Modeling

### 4.1 设计直觉

Stage-1 产生 codebook-0（semantic-rich），Stage-2 补全 codebooks 1-7（acoustic details）。总 codebook 数 $K = 8$（索引 0-7）。

记号：
- $\mathbf{x}_{1:T}^{(0)} = (x_1^{(0)}, \ldots, x_T^{(0)})$：Stage-1 输出的 codebook-0 tokens
- $\mathbf{x}_{1:T}^{(1:7)}$：所有 residual codebooks
- 每个 timestep 是 tuple：$\mathbf{x}_t^{(0:7)} = (x_t^{(0)}, x_t^{(1)}, \ldots, x_t^{(7)})$

### 4.2 Aligned Autoregressive Factorization

$$p(\mathbf{x}_{1:T}^{(0:7)}) = \prod_{t=1}^{T} p(\mathbf{x}_t^{(0:7)} \mid \mathbf{x}_{<t}^{(0:7)}) \quad \text{(7)}$$

这个 factorization 是 **time-aligned**——每帧同时考虑所有 8 个 codebook。

### 4.3 Cross-Conditioning 序列组织

训练时序列组织：

$$[\underbrace{x_1^{(0)}, \ldots, x_T^{(0)}}_{\text{all codebook-0 tokens}}, \underbrace{x_1^{(0)}, x_1^{(1)}, \ldots, x_1^{(7)}}_{\text{frame 1}}, \ldots, \underbrace{x_T^{(0)}, x_T^{(1)}, \ldots, x_T^{(7)}}_{\text{frame T}}]$$

Loss：

$$\mathcal{L}_{Stage2} = -\sum_{t=1}^{T} \log p(\mathbf{x}_t^{(0:7)} \mid \mathbf{x}_{<t}^{(0:7)})$$

**关键设计直觉**：把所有 codebook-0 tokens 放在序列开头，确保模型在预测 residual 之前能 attend 到完整的 semantic 大纲（codebook-0）。这是一种 "plan-then-detail" 的设计。

### 4.4 推理

推理时 codebook-0 tokens 来自 Stage-1，被 clamp（固定）。模型只生成 residual codebooks 1-7。这保证了 stage 之间的 alignment。

参考 SoundStream / EnCodec 的 hierarchical RVQ 思路：
- EnCodec (Défossez et al.): https://arxiv.org/abs/2210.13438
- SoundStream (Zeghidour et al., 2021)

### 4.5 实现

- 1B 参数 Transformer
- 8K context window
- 6 秒 single-track segments
- 共享 acoustic codebook space（speech, vocals, instrumentals, mixtures 都用同一 codebook）

---

## 5. Tokenization

### 5.1 X-Codec (Semantic-Acoustic Fused)

YuE 选择 X-Codec 作为 audio tokenizer：
- 100M 参数 HuBERT-based 语义 encoder，融合到 codec latent space
- 50Hz frame rate
- 12 RVQ layers，每个 codebook size 1024
- 使用前 8 layers
- codebook-0 捕获 semantic info (melody, vocal content)

训练数据：200k 小时 16kHz audio，music:speech:audio effects = 1:1:0.05

参考：
- X-Codec (Ye et al.): https://arxiv.org/abs/2408.17175
- SemantiCodec (Liu et al.): https://arxiv.org/abs/2406.06965
- HiFi-Codec (Yang et al.): https://arxiv.org/abs/2305.02765

### 5.2 Tokenizer 比对 (Table 9)

| Type | Codec | Reconstruction | LM Converge | Invalid Prob. |
|------|-------|----------------|------------|---------------|
| Acoustic | EnCodec32k | Good | No | All |
| Acoustic | HiFiCodec | Good | No | All |
| Semantic+Acoustic | SemantiCodec | Fair | Yes | High |
| Semantic+Acoustic | X-Codec | Fair | Yes | Low |

**关键发现**：纯 acoustic tokens 在 YuE 的 in-the-wild 数据上无法让 LM 收敛，即使 7B model + 1T tokens 也只有 intermittent 成功，输出仍以 noise 为主。这反驳了"acoustic tokens 通用"的假设，在 MusicGen 的相对干净的 MTG-Jamendo 数据上可行，但在大规模 in-the-wild 数据上不行。

SemantiCodec 的问题：AudioMAE 的 patch-based 机制导致 token misalignment 错误传播。X-Codec 用 HuBERT 避免了这个问题。

### 5.3 Special Tokens

```
<EOD> - End of document
<SOA> - Start of audio
<EOA> - End of audio
<stage_1> - Stage 1 marker
<stage_2> - Stage 2 marker
<encodec32k>, <xcodec>, <semanticodec>, <hificodec> - Tokenizer type indicators
```

### 5.4 Vocoder (Vocos-based)

- 16kHz → 44.1kHz upsampling
- Codebook dropout + Gaussian noise 增强鲁棒性

参考 Vocos (Siuzdak): https://arxiv.org/abs/2306.00814

---

## 6. Training Strategy

### 6.1 Multitask Learning

将 lyrics-to-song 能力分解为四个 essential capabilities：

1. Modeling of Human Vocal
2. (paper 表中似乎缺失 #2)
3. Joint Modeling of Vocal and Instrumental
4. Aligning Cross-Modal/Same-Modal Controls

对应三个任务：

**TTS 任务**：
- 数据：WeNetSpeech (zh), LibriHeavy (en), GigaSpeech (en)
- 70k hours
- 顺序拼接多个 TTS pair 形成长 context
- `"Generate speech:"` 指令前缀，50% dropout rate
- 关键 trade-off：太多 TTS → 偏向 rap；太少 → lyrics 跟随差

**Music Generation**：
- 650k hours in-the-wild music
- Qwen2-Audio 自动打 tag (genre, instrument, mood)
- 40% 用 UVR (htdemucs_ft, Kim_Vocal_1, UVR-MDX-NET-Inst_3 ensemble) 分离 vocal/accompaniment
- 双指令：`"Generate music based on the given tags"` 或 `"Generate music in dual-track format based on the given tags"`

**Lyrics-to-Song**：
- 启发式过滤，~10% 保留
- CoT 设计降低对精确 alignment 的依赖
- ~80% match rate

参考 Qwen2-Audio: https://arxiv.org/abs/2407.10759

### 6.2 Multiphase Training

**Phase-1: Warm Up (280B tokens)**
- Linear LR: $0 \to 3 \times 10^{-4}$
- 仅 English + Chinese（质量最高）
- Context 8192 (mix ~163s, dual-track ~81s)
- Global batch 768 (~6.29M tokens)

**Phase-2: Constant LR (1T tokens total)**
- LR $3 \times 10^{-4}$
- 加入 multilingual + lower quality data
- old:new = 2:1 防 distribution shift

**Phase-3: Context Extension (750B tokens)**
- LR $3 \times 10^{-4}$
- Context 8192 → 16384
- 移除 single-track unconditional data

**Phase-4: Annealing with Control Injection (40B tokens)**
- Cosine LR schedule → $3 \times 10^{-5}$
- 移除 speech + unconditional music
- 加入 ICL, gender tags, vocal timbre tags
- BPM control 后被移除（与 lyrics length 耦合，degrade lyrics following）
- 20K hours 高质量数据
- CoT:ICL = 2:1

**总训练 budget**: 1.75T tokens + 40B annealing

**Stage-2**: 2T tokens, 1B params, 8K context, cosine schedule, max LR $3 \times 10^{-4}$

### 6.3 Optimization Details

- Adam optimizer, $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$
- Gradient clipping: 1.0
- Weight decay: 0.1
- Init std: 0.02
- Global batch 768（资源受限时降到 512 或 256）
- LLaMA2 architecture, Megatron-LM codebase

参考 Megatron-LM: https://arxiv.org/abs/1909.08053
参考 LLaMA2: https://arxiv.org/abs/2307.09288

---

## 7. Test-time Strategies

### 7.1 Forced Decoding

- Stage-1：限制 vocab 在 audio 范围内直到 `<EOA>` 出现
- Stage-2：codebook-0 tokens 在每帧被 enforce，residual token 只允许对应 codebook 的 vocab

### 7.2 Sampling & CFG

参数：
- top-k = 50
- top-p = 0.93
- repetition penalty = 1.1
- temperature = 1
- max new tokens = 3000

**Classifier-Free Guidance**：

$$\ell_{cfg}(k) = s[\ell_c(k) - \ell_u(k)] + \ell_u(k)$$

变量：
- $\ell_c(k) = \log p_\theta(k \mid x)$：conditional log-prob
- $\ell_u(k) = \log p_\theta(k \mid \emptyset)$：unconditional log-prob
- $s$：CFG scale，第一段 $s=1.5$，后续段 $s=1.2$

直觉：通过外推 conditional 和 unconditional 的差值，放大 prompt 的影响。

### 7.3 Music ICL at Test Time

- 用歌曲的 chorus section 作为 ICL reference 显著增强 musicality 和 stability
- Dual-track ICL 比 single-track ICL 音质更好
- Dual-track ICL 默认启用

### 7.4 Test-time Tricks Ablation (Figure 14)

| 方法 | Win Rate (Musicality) |
|------|----------------------|
| CoT only | 0.21 |
| ICL only | 0.63 |
| ICL + CFG | 0.79 |

ICL 显著优于 CoT，因为 ICL 把 decoded token space 限制在 musically favorable subspace。CFG 进一步增强这种条件化效果。

---

## 8. Experiments

### 8.1 Data Composition

- TTS: 70k hours → 13B tokens
- Music: 650k hours → 200B+ tokens (mix + demix)
- CoT: 28B tokens
- Annealing ICL: 10B tokens expand 4x = 40B (vocal-ICL, accompaniment-ICL, mix-ICL, dual-ICL)
- Prior to annealing: Conditional:Unconditional = 3:1, Music:Speech = 10:1
- Annealing: CoT:ICL = 2:1

### 8.2 Model Scaling

| Scale | Tokens | GPUs |
|-------|--------|------|
| 0.5B | 500B | 32 H800 |
| 2B | 500B | 96 H800 |
| 7B | 1.75T + 40B annealing | 512 H800 |

### 8.3 Baselines

四个 closed-source：Suno V4, Udio, Hailuo, Tiangong

---

## 9. Main Results

### 9.1 Human Evaluation (Figure 6)

| System | vs YuE (Musicality Win-Tie-Loss) |
|--------|----------------------------------|
| Suno V4 | YuE trails |
| Udio | Comparable |
| Tiangong | Comparable |
| Hailuo | YuE outperforms |

### 9.2 Musicality 细分 (Figure 7, normalized by Suno)

YuE 在以下方面突出：
- **Music structure**：长程结构清晰
- **Music arrangement**：编曲能力

YuE 弱项：
- Vocal acoustic quality（tokenizer 限制）
- Accompaniment acoustic quality

### 9.3 Controllability

YuE 强项：
- Genre adherence
- Instrument/vocal consistency
- Emotion control

中等：
- Emotion（pseudo label 噪声）
- Tempo control（pseudo label 噪声）

### 9.4 Vocal Agility (Figure 8)

| System | Median Vocal Range (semitones) |
|--------|--------------------------------|
| Suno V4 | ~27 (top) |
| YuE | ~27 (top) |
| Tiangong | ~20 |
| Hailuo | ~20 |

Vocal range = song 内 vocal 跨越的 semitone 数。27 semitones ≈ 跨越 2 个多 octave，表明 vocal expressiveness 强。

### 9.5 Duration (Figure 9)

YuE 生成最长 audio，duration range 最宽。Hailuo 最受限。

### 9.6 Model-Based Evaluation (Table 3)

| Metric | Hailuo | SunoV4 | Tiangong | Udio | YuE |
|--------|--------|--------|----------|------|-----|
| KL ↓ | 0.756 | 0.620 | 0.708 | 0.503 | **0.372** |
| FAD ↓ | 2.080 | 1.544 | 2.547 | **1.222** | 1.624 |
| CE ↑ | 7.350 | 7.474 | 7.421 | 7.112 | 7.115 |
| CU ↑ | 7.737 | 7.813 | 7.766 | 7.520 | 7.543 |
| PC ↑ | 6.793 | 6.601 | 6.060 | 6.626 | 6.280 |
| PQ ↑ | 8.132 | 8.120 | 8.220 | 7.803 | 7.894 |
| CLAP ↑ | 0.265 | 0.265 | 0.244 | 0.310 | 0.118 |
| CLaMP3 ↑ | 0.106 | 0.160 | 0.114 | 0.156 | **0.240** |

观察：
- YuE 在 KL 最优，FAD 竞争力
- YuE 在 CLaMP 3 最优（与 human controllability 一致）
- YuE 在 CLAP 显著低（与 human 评估不一致！）

### 9.7 Correlation Analysis (Tables 4-7)

**Table 4: Subjective vs Automatic**

| | Musicality | Average |
|-|-----------|---------|
| Vocal Range | **0.857** | **0.858** |
| CLaMP 3 | 0.333 | 0.264 |
| CE | 0.368 | 0.357 |
| CLAP | -0.072 | 0.086 |

**Vocal Range 与 musicality 相关性 0.85+**——这是个很实用的发现，可作为 musicality 的 proxy metric。

**Table 5: Alignment vs Controllability**

| | LyricFollow | GenCtrl | InstrCtrl | EmoCtrl | Tempo |
|-|-------------|---------|-----------|---------|-------|
| CLAP | -0.25 | 0.01 | -0.07 | 0.14 | 0.09 |
| CLaMP 3 | **0.42** | **0.37** | **0.44** | 0.33 | **0.36** |

CLaMP 3 一致优于 CLAP，特别在 LyricFollow (0.42 vs -0.25) 和 InstrCtrl (0.44 vs -0.07)。

**Intuition**：CLAP 训练时 exposure 不足 singing/music-specific content；CLaMP 3 用 web-scale 训练更鲁棒。

**Table 6: KL/FAD vs Acoustic Quality**

| | AccompQual | VocalQual |
|-|------------|-----------|
| KL | 0.14 | 0.23 |
| FAD | -0.15 | -0.11 |

**意外**：KL 和 FAD 与 acoustic quality 相关性弱。原因：PaSST backbone (AudioSet 预训练) 对 generative music 可能 OOD；样本 size bias。

**Table 7: Content-Based vs Musical Aspects**

| | AccompQual | VocalQual | SongStruct | VAComp | MelAttrac | MusicArr |
|-|------------|-----------|------------|--------|-----------|-----------|
| CE | **0.56** | **0.66** | 0.33 | 0.35 | 0.30 | 0.31 |
| CU | 0.50 | 0.61 | 0.27 | 0.29 | 0.25 | 0.26 |
| PC | -0.09 | 0.00 | -0.24 | -0.20 | 0.00 | -0.16 |
| PQ | 0.27 | 0.36 | 0.05 | 0.06 | -0.03 | 0.02 |

CE 与 VocalQual (0.66) 和 AccompQual (0.56) 相关性最强——CE 可能是 acoustic fidelity 的好 proxy。

参考：
- CLaMP 3: https://arxiv.org/abs/2502.10362
- CLAP: https://arxiv.org/abs/2209.03143
- Audiobox-Aesthetic: https://arxiv.org/abs/2502.05139

---

## 10. Multilingual Evaluation (Table 8)

| Model | Chinese Lyrics | Chinese Music | Korean Lyrics | Korean Music | Japanese Lyrics | Japanese Music |
|-------|----------------|---------------|---------------|--------------|------------------|----------------|
| YuE | 60 | 62 | 55 | 55 | **70** | 52 |
| Suno V4 | **73** | **88** | **75** | 50 | 60 | **80** |
| Udio | 36 | 46 | 62 | **62** | 31 | 51 |
| Hailuo | 30 | 15 | 37 | 60 | 56 | 31 |
| Tiangong | 51 | 39 | 20 | 22 | 32 | 35 |

YuE 在 Japanese lyrics following 上最优（70%），表明 cross-lingual transfer 有效。在 Chinese musicality 上 second (62%)，反映需要更多 culturally-specific training。

---

## 11. Representation Quality (MARBLE, Table 10)

YuE 在 MARBLE benchmark 上的成绩：

| Method | GTZAN Genre Acc↑ | GS Key AccRefined↑ | MTG AP↑ | MTG AUC↑ | EMO R2V↑ | EMO R2A↑ |
|--------|------------------|---------------------|---------|----------|----------|----------|
| MERT (2023) | 78.6 | 65.6 | 29.9 | 83.4 | 61.2 | 74.7 |
| MusicFM (2024) | 83.8 | 63.9 | - | - | 60.3 | 76.3 |
| MuQ (2025) | 85.6 | 65.0 | - | - | 62.8 | 76.1 |
| CLaMP 3 (2025) | 86.6 | 53.8 | 30.2 | 82.4 | 59.1 | 70.0 |
| YuE | 83.4 | **67.0** | 29.2 | 82.7 | 58.9 | 75.0 |

**关键**：YuE 在 GS Key Recognition 上 SOTA (67.0%)，说明模型对 tonality/modality 有良好 sense，这对 in-tune singing 至关重要。

值得注意 YuE 是 generative model 不是专门为 representation learning 设计的，且只使用 codebook-0 tokens，仍在多个任务上 competitive。

参考：
- MARBLE (Yuan et al., NeurIPS 2024): https://arxiv.org/abs/2311.09615
- MERT (Li et al.): https://arxiv.org/abs/2306.00107
- MuQ (Zhu et al.): https://arxiv.org/abs/2501.01108

---

## 12. Emergent Abilities

随 model scaling，YuE 自发涌现多种能力：

**Advanced Vocal Techniques**：
- Vibrato, glissando, bel canto
- Death growl (metal)
- Mix voice, belting
- Riffs and runs
- Vocal fry
- Beijing Opera (京剧)
- Shanbei folk vocals (陕北民歌)

**Spontaneous Performance**：
- Jazz scat singing after lyrics 用完
- A cappella multi-part harmony
- Folk music 中的 harmonica interludes（vocal 间隙自动插入）

**World Music & Pattern Mixing**：
- Chinese gangsta rap + Japanese shamisen
- 融合 Chinese opera + Shanbei folk + traditional Chinese vocals

**Voice Cloning**：成功复制 Billie Eilish 和 Faye Wong (王菲) 的音色，保留 timbral qualities 和 breathy textures。

**Style Transfer**：Japanese female J-pop → English male rap with city pop accompaniment，不仅转换 vocal 特征还调整 prosody 和 phrasing。

**Code Switching**：自然处理多语言/方言在同一 vocal performance 中的切换。

---

## 13. Memorization Analysis (Section 11, Figure 15)

用 ByteCover2 retrieval model 检测 ICL mode 下是否复制训练数据：

- 构造两个 set：$\mathcal{R}$ (training examples) 和 $\mathcal{G}$ (YuE ICL outputs)
- 计算 cosine similarity，分析 top 1% scores
- Baselines: GTZAN (genre-level), Covers80 (known duplicates)

结果：Ref-Gen 相似度分布显著低于 Covers80，与 GTZAN 比也 moderate。短 repetitive motifs (percussive loops) 偶尔出现，但**整体不 extensive copying**。

Reference: ByteCover2 (Du et al., ICASSP 2022): https://arxiv.org/abs/2203.02614

---

## 14. Unsuccessful Attempts (重要！)

这部分非常有价值，记录了失败经验：

### 14.1 Acoustic Tokens

EnCodec/HiFiCodec 在 in-the-wild 数据上让 LM 无法收敛，即使 7B + 1T tokens。原因：
- Acoustic tokens 优先 compression efficiency 而非 representation quality
- 有限 capacity 倾向 shortcut（直接信息复制）
- Lossy nature + limited semantic relevance + excessive focus on reconstruction

### 14.2 Unconditional Pre-train at 7B

小 model (sub-billion) 可以通过 finetune 学到 cross-modal alignment。但 7B model 上 unconditional pre-train 反而 counterproductive：finetune 无法建立有效 alignment。作者称为 **"catastrophic inertia"**——大模型内化过于 generic priors，overshadow conditional mappings。

这个发现与 LLM 中 "scaling 并不总是帮助 finetune" 的观察类似。

### 14.3 Early Activation of ICL

过早加 ICL 数据 → 模型过度依赖 reference audio → shortcut learning → 即使后续移除 ICL 数据，模型仍产生 invalid outputs (noise/silence)。**Shortcut learning 不可逆**，scaling 让问题更严重。

---

## 15. 关键 Insight 总结

1. **Dual-NTP**：显式 source separation prior 是关键，让 single LM 同时建模 vocal 和 accompaniment 而不互相干扰。
2. **CoT**：RoPE 的 long-term decay 是长程 lyrics-following 的根本障碍，利用 music structural prior 重新激活 conditioning 是 elegant 解决方案。
3. **Delayed ICL Activation**：ICL 是 "easy" data，过早加入导致不可逆 shortcut learning。Annealing phase 才加 ICL。
4. **Semantic-Acoustic Fused Tokenizer**：纯 acoustic tokens 在 in-the-wild 数据上无法让 LM 收敛，必须 fuse semantic info。
5. **Vocal Range 作为 Musicality Proxy**：correlation 0.85+，远超 KL/FAD/CLAP 等传统 metric。
6. **CLaMP 3 > CLAP**：CLaMP 3 与 human controllability 评估更一致，CLAP 在 singing/music 场景下不可靠。
7. **Catastrophic Inertia**：大模型 unconditional pre-train 后 finetune 效果反而差——大模型的 generic priors 抑制 conditional mappings。

---

## 16. 可能的联想与未来方向

### 16.1 与 LLM Pre-training 的类比

- CoT 设计让人联想到 LLM 中的 long-context 技术（Ring Attention, YaRN, LongRoPE），但 YuE 通过 domain-specific structural prior 绕过纯技术解决方案。
- Delayed ICL activation 类似 instruction tuning 不能太早进行（mixing pre-training 和 instruction tuning 会有副作用）。
- Catastrophic inertia 类似 "pretrain-finetune gap" 在大模型上更显著。

### 16.2 与 AudioLM/VALL-E 的关系

- YuE 的 two-stage 设计 (semantic → acoustic) 沿袭 AudioLM 思路。
- Dual-NTP 是 VALL-E 的 codec LM 思路的扩展，但用 source separation prior 取代 single-stream。
- X-Codec 的 semantic-acoustic fusion 类似 SpeechTokenizer (Zhang et al., 2023) 的设计哲学。

### 16.3 与 Suno/Udio 的差距

YuE 仍落后 Suno V4。可能的差距来源：
- Acoustic fidelity（tokenizer 限制）
- 更大规模数据
- 更好的 mixing/mastering 后处理
- Suno 可能用 diffusion-based post-processing 而非纯 AR

### 16.4 与 Diffusion Model 的对比

YuE 选择纯 AR 路线（LLaMA2），未用 diffusion。Diffusion model (MusicLDM, MusicLM, Stable Audio Open) 在 audio fidelity 上有优势，但 AR 在 long-form coherence 和 controllability 上更强。YuE 的实验支持这个 trade-off。

参考：
- Stable Audio Open (Evans et al.): https://arxiv.org/abs/2407.14358
- MusicLDM (Chen et al.): https://arxiv.org/abs/2308.01546

### 16.5 Future Work 方向

paper Section 13 提到：
- 改进 acoustic fidelity 和 mixing
- 融入 music knowledge (chord progressions, instrumentation theory)
- 更深 prosodic 和 emotional control
- Multilingual 和 cross-cultural 扩展
- Music education, accessibility, therapy 应用

### 16.6 伦理考量

- 提倡 AI-generated content 标注
- ICL mode 不 extensive copying
- Culturally diverse training data，支持 niche music styles
- 通过 HKUST HREP-2023-0230 伦理审查

---

## 17. 补充材料

### 17.1 Evaluation Dimensions (Appendix A)

13 个评估维度，包括：
- Overall Musicality, Vocal Quality, Accompaniment Quality, Arrangement Complexity, Melodic Memorability, Vocal-Accompaniment Matching, Song Structure Clarity
- Lyrics Following, Multilingual Switching, Genre Control, Instrument/Vocal Control, Emotional Expressiveness, Tempo/Rhythm

### 17.2 Tagging Prompts (Appendix B)

Qwen2-Audio 用 JSON 格式输出：
- Music tagging: `Music_genre`, `Instrument`, `Mood`
- Vocal tagging: `gender`, `age`, `vocal_timbre`

### 17.3 Multilingual Eval (Appendix C)

YuE 在 Japanese lyrics following 上 70% (best)，Chinese 60% (second)，Korean 55% (third)。

---

## 18. 个人 Commentary

Andrej，这篇 paper 给我几个强烈直觉：

1. **AR LM 在多模态生成上仍有 strong scaling law**：YuE 0.5B → 7B 的 scaling effect 显著，类似 LLM 中的 emergent abilities（vibrato, scat singing, death growl 等都是 spontaneous 涌现）。

2. **Domain prior 比通用技术更 elegant**：CoT 利用 music structure 解决 long-context 问题，比单纯调整 RoPE base 或 NTK-aware interpolation 更聪明。这暗示我们解决 domain-specific 问题时应该先理解 domain 的结构。

3. **Tokenizer 的 semantic-acoustic 融合是关键 enabler**：纯 acoustic tokens 在大规模 in-the-wild 数据上不可行，semantic-acoustic fusion 是当前 AR music generation 的必要条件。

4. **Training data 顺序和时机很重要**：Delayed ICL activation 和 multiphase training 都强调"何时学什么"的重要性。这和 LLM 中的 curriculum learning、annealing 思路相通。

5. **Metric 选择需要审慎**：CLAP 与 human 评估不一致，Vocal Range 反而与 musicality 相关性最高。这提醒我们在评估生成模型时要重新审视 metric 的 validity。

6. **Catastrophic Inertia**：大模型 unconditional pre-train 后 finetune 效果反而差，是个反直觉但重要的发现。这与 instruction tuning 在更大 base model 上效果递减的观察类似，值得深入研究。

希望这个讲解帮你 build 直觉。如果你对某个细节（比如 Dual-NTP 的具体实现、CoT 的数据 pipeline、或 Stage-2 的 cross-conditioning）有进一步兴趣，我可以深入展开。
