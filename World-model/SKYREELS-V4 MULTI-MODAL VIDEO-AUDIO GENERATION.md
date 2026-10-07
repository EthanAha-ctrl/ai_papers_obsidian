---
source_pdf: SKYREELS-V4 MULTI-MODAL VIDEO-AUDIO GENERATION.pdf
paper_sha256: 63b06b519ec1cd784611182c7460d8ec6cdb10f6aaa436abc84cea6ee1092b8e
processed_at: '2026-08-12T07:43:42-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 SkyReels-V4

Andrej，咱们抛开术语，用大白话重新聊聊这篇 paper。

---

## 这篇 paper 在干啥

**一句话**：教一个 AI 同时生成画面和声音，而且能改、能补、能参考图、能参考视频、能参考音频——所有这些活儿一个 model 干完。

之前视频生成领域的状态是这样的：
- 你要 T2V，用 Model A
- 你要图生视频，用 Model B  
- 你要改视频，用 Model C
- 你要给视频配音，用 Model D
- 你要参考某张人脸生成视频，用 Model E

每个 model 一套 weight、一套训练数据、一套 inference pipeline。SkyReels-V4 的野心就是：**一套 weight 全包了**。

---

## 为啥这事难

你看 1927 年《爵士歌手》那部电影出来之前，电影都是无声的。观众看火车进站就吓得躲，但没有声音总觉得差点意思。视频生成现在就处在 "silent film" 阶段——Sora、Kling、Wan 出来一堆漂亮的画面，但没声。

给视频加音频这事听着简单，实际坑特别多：

1. **对不齐**：人物张嘴说话，结果声音晚来 200 毫秒，看着像 B 级片配音
2. **逻辑不对**：画面里有人敲鼓，结果生成了钢琴声
3. **质量崩**：如果用两个独立 model（一个生成画面、一个生成声音），中间没有信息交流，就很难做到"敲鼓的画面配鼓声"这种常识

业界之前怎么解决的呢？**shallow fusion**——主 model 还是生成画面，旁边挂一个小 adapter 模块，把音频信号塞进去。这就像你请了一个翻译，但翻译只能听见你说的一半话，翻译质量自然上不去。

SkyReels-V4 的思路是 **deep fusion**：画面和声音两个 stream 从第一层到最后一层全程交换信息，每层都互相 attend。

---

## 三个核心创新，用大白话讲

### 创新 1：两个对称的"大脑"并行处理画面和声音

**架构图（用 ASCII 凑合一下）**：

```
        Text/Image/Video/Audio Reference
                     │
                     ▼
            [Frozen MLLM]
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   Video Branch              Audio Branch
   (从T2V pretrained)         (from scratch)
        │                         │
   ┌────┴────┐               ┌────┴────┐
   │ M layers │               │ M layers │
   │ Dual-Stream│◄──cross-attn──►│Dual-Stream│
   │ (独立参数) │               │ (独立参数) │
   └────┬────┘               └────┬────┘
        │                         │
   ┌────┴────┐               ┌────┴────┐
   │ N layers │               │ N layers │
   │ Single-  │◄──cross-attn──►│ Single-  │
   │ Stream   │               │ Stream   │
   │ (共享参数)│               │ (共享参数)│
   └────┬────┘               └────┬────┘
        │                         │
        ▼                         ▼
     Video                     Audio
```

**人话翻译**：

前 M 层，画面和文本各走各的"专用通道"，但每隔一层就开个会议室让两边碰头——这样画面带着自己的"视觉直觉"、文本带着"语义理解"碰头商量。声音那边也一样。

后 N 层，大家都熟了，干脆合用一个通道，参数共享，省点显存。

**关键 trick**：声音和画面之间有个**双向 cross-attention**。每一层都互相 query 一下对方："你那边进展咋样？我这边嘴唇开始张开了，你那边声音准备好没？"

公式 (5) 看着吓人，其实就是：
- 声音 token 去 query 画面 token，拿到画面信息更新自己
- 画面 token 再去 query 声音 token（已经更新过的），拿到声音信息更新自己

这样两个 modality 在整个网络深度里**全程互相对话**，不是只在最后一层 fusion。这就是为什么 paper 说这叫 deep fusion。

**为啥要对称？** 因为如果声音 branch 比画面 branch 弱（维度小、capacity 低），那 cross-attention 时就得加个 projection layer 把维度对齐。这一加 projection 就破坏了 pretrained weight 的好状态。所以两边用完全一样的架构、一样的 latent dimension，省掉 projection，保留 unimodal pretrain 的"好底子"。

### 创新 2：把所有任务都变成"填空"

这是我觉得最 elegant 的部分。

之前你想做不同的活儿，要用不同的 input 接口：
- T2V：输入 text，输出 video
- I2V：输入 image + text，输出 video
- Edit：输入 video + mask + text，输出 video
- Extend：输入 video 前半段 + text，输出完整 video

每个接口都要写专门的代码、训练专门的数据。

SkyReels-V4 说：**别这么麻烦，所有任务我都看成"填空"**。

具体怎么做？把 input 拼成三个 channel：

```
[Noisy Latent V] [Reference Latent I] [Mask M]
     C通道            C通道              1通道
   要生成的        已知的内容         哪里要填
```

**Mask 怎么用？**

| 任务 | Mask 配置 | 大白话 |
|------|-----------|--------|
| T2V | 全 0 | 全部都要填 |
| I2V | 第0帧=1，其余=0 | 只给第一张图，剩下的填 |
| Extend | 前k帧=1，其余=0 | 给前 k 帧，往后续 |
| Start-End 插帧 | 第0帧和最后帧=1，中间=0 | 给开头和结尾，中间填 |
| Editing | 任意 spatiotemporal 区域=1 | 标记要改的地方 |

**Intuition**: 这就是 BERT 的 masked language modeling 搬到了 spatiotemporal latent space。BERT 是把 token 遮掉让 model 猜，SkyReels-V4 是把 video frame 的某些区域遮掉让 model 补。

一旦接受这个 formulation，**任务的多样性就变成了 mask 形状的多样性**。一个 model、一套 weight、一个 inference pipeline 就能 cover 所有任务。这跟 LLM 用一个 model 做 generation + completion + editing 是一回事。

**音频怎么办？** Inpainting 只 apply 到 video stream。Audio branch 看到 video 被改了之后，从 scratch 重新生成 audio——这样保证音画同步。比如你把视频里的猫换成狗，audio branch 看到新画面会自动生成"汪汪"而不是"喵喵"。

### 创新 3：草稿+精修两步走，省算力

1080p、32fps、15秒——这是 480 帧 1080p 视频。直接让 diffusion model 生成这么多 token，显存爆炸。

SkyReels-V4 的 trick：

**第一步（Base Model）**：生成低分辨率完整序列 + 高分辨率关键帧
- 低分辨率（比如 480p）480 帧都生成
- 高分辨率（1080p）只生成几个 keyframe

**第二步（Refiner）**：把低分辨率插值放大到 1080p，然后在 keyframe 位置用真的高分辨率 keyframe 替换，最后 refine 整个序列。

```
Base Model 输出:
  Low-res: 480p × 480 frames  (cheap, full)
  High-res: 1080p × 8 keyframes (expensive, sparse)
        │
        ▼
   线性插值上采样到1080p
        │
        ▼
   Keyframe 位置替换为真高分辨率
        │
        ▼
   [Refiner with VSA sparse attention]
        │
        ▼
   1080p × 480 frames 最终输出
```

**Intuition**: 这跟视频编码里 I-frame + P-frame/B-frame 的思路一模一样。I-frame 是高质量关键帧（贵），P/B frame 是从 I-frame 衍生出来的（便宜）。SkyReels-V4 让 Base model 只在 keyframe 上"使劲"，剩下交给 Refiner 通过超分+插帧补全。

**Refiner 怎么省算力？** 用 **Video Sparse Attention (VSA)**：先粗略扫一遍找出"重要 token 区域"，只在这些区域做密集 attention。视频的时空冗余度很高（相邻帧的相邻像素大部分时候 attend 到类似区域），full quadratic attention 是浪费。VSA 大概能省 3 倍 attention 计算。

---

## Multi-modal In-Context Learning：最让我兴奋的部分

这部分其实把 LLM 的 in-context learning 思路搬到了 diffusion model 里。

### 场景

用户 prompt：
```
"generate a video of person A from @image_1 
 speaking <dialogue>hello</dialogue> 
 in the style of @video_1"
```

### 两条路径并行 conditioning

**路径 1：通过 MLLM 做 semantic guidance**

把 prompt + @image_1 + @video_1 都喂给 frozen MLLM（类似 Qwen3-Omni 或 Gemini），MLLM 理解 "@image_1 是 person A 的样子，@video_1 是风格参考" 这种复杂指令，输出一个多模态 embedding 给两个 branch 共享。

这就像你跟画师说"用这张照片里的人，按这个视频的风格画一段"，画师脑子里的"理解"就是 MLLM 输出的 embedding。

**路径 2：直接把 reference latent 塞进 self-attention**

MLLM 的 embedding 是"语义级别"的（"这是一个戴眼镜的男性"），但 reference image 里很多 fine-grained 信息（眼角细纹、肤色纹理、发型层次）丢失了。

SkyReels-V4 的做法：把 reference image 经过 VAE encode 之后，**直接 prepend 到 noisy video latent 前面**，一起走 self-attention。

公式 (8): `Z_attn = [Z_cond ; Z_video]`

**Intuition**: 这就是 ChatGPT 里贴一张图让 model 参考的思路——直接把 reference token 当 context，把要生成的 noisy latent 当 query，attention 让 query 去 attend context。Reference image 的所有 pixel-level 信息都能被 generation 直接 access。

### Offset 3D RoPE: 怎么区分"参考"和"目标"

问题来了：如果 reference image 和 video frame 都从 t=0 开始算 RoPE，model 会分不清谁是 context 谁是 target。

SkyReels-V4 的 trick：
- Reference latent 用 **negative temporal index**（-N_cond, -N_cond+1, ..., -1）
- Video latent 用 **positive index**（0, 1, 2, ..., T-1）

这相当于告诉 model："reference 是过去的，video 是未来的，你看着过去生成未来。"

这跟 GPT 的 causal mask 思路类似——用 positional encoding establish "context vs target" 的结构。

---

## 训练策略：从简单到复杂的 progressive schedule

| Stage | 任务 | 分辨率 | 数据 | Epochs |
|-------|------|--------|------|--------|
| 1 | T2I | 256px | 3B image | 3 |
| 2 | T2I + T2V | 256px 16fps 2-10s | 1B image + 400M video | 3 |
| 3 | +Inpaint | 256px 2-15s | 1B + 400M | 2 |
| 4 | Mixed | 256/480px | 100M + 100M | 2 |
| 5 | High-res | 480/720/1080px | 50M + 50M | 2 |
| 6 | +Multi-modal ref | 480-1080px | 20M + 50M | 2 |
| Audio | T2A from scratch | variable length | 几千小时 | 3 |
| Joint | T2V + T2AV + T2A | 720/1080px 5-15s | - | 2 |
| SFT1 | Joint + Multi-modal | 720/1080px | 5M | 3 |
| SFT2 | Curated high-quality | 720/1080px | 1M | 3 |

**Intuition**: 这个 schedule 的思路是"先学简单的，再学复杂的"。

1. **先学图片再学视频**（Stage 1）：图片是视频的子集，先在 3B 图片上把"语义到像素"的 mapping 学扎实，后面加时间维度时 spatial prior 已经稳了，收敛快
2. **先低分辨率学运动，再高分辨率学细节**（Stage 2 vs Stage 5）：256px 时计算便宜，能快速试错学会 motion；上到 1080px 时 motion 已经 OK，只是 refine spatial detail
3. **Inpaint 任务一开始只占 5%**（Stage 3）：避免 inpainting 信号淹掉 T2V 的主信号
4. **Multi-modal reference 最后才加**（Stage 6）：因为这种 conditioning 最复杂，得等 base 能力稳了再上
5. **Joint training 时 T2V 和 T2AV 各占 50%**：防止 model 因为学 joint 而忘了 T2V 单模态能力
6. **最后用 1M 高质量数据 SFT**：类似 LLM 的 SFT 阶段，refine aesthetic

---

## 数据 Pipeline：为啥这么讲究

Audio-visual joint training 最大的坑是 **数据不同步**。如果训练视频本身音画 desync，model 永远学不会 lip-sync。

SkyReels-V4 用 SyncNet 过滤：
- SyncNet 是个 ConvNet，学 sound 和 mouth image 的 joint embedding
- 保留 |offset| ≤ 3 帧 且 confidence > 1.5 的 clip
- 3 帧 ≈ 100ms，比人眼能感知的 desync 阈值（~80ms）略松

**Caption 也讲究**：用 structured caption，`<dialogue>hello</dialogue>` 这种 tag 让 model 知道 "hello" 是要被 spoken 的，不是显示在屏幕上的字。`<sfx>爆炸声</sfx>` 标记音效，`<bgm>悲伤的音乐</bgm>` 标记背景音乐。

用户 free-form 输入时，先经过 prompt enhancer 转换成 structured 格式再喂给 model。

---

## 实验结果

### Artificial Analysis Arena（公开盲测）

| Track | SkyReels-V4 排名 | 跟谁比 |
|-------|-----------------|--------|
| T2V + Audio | **#1** (Elo 1129) | 超 Kling 3.0 (1097)、Veo 3.1 (1080) |
| T2V no Audio | #2 (Elo 1236) | 仅次于 Kling 3.0 (1248) |
| I2V + Audio | #4 (Elo 1067) | Veo 3.1 第一 (1068) |
| I2V no Audio | #7 (Elo 1279) | grok-imagine 第一 (1328) |

**有意思的观察**：SkyReels-V4 在 audio+video 联合 task 上特别强（#1），但纯 visual task 上反而输给 Kling 3.0。这说明 SkyReels-V4 把 capacity 投入到了 audio-visual joint 上，纯 visual 上略有 trade-off。I2V 上更弱（#7）说明 channel-concat inpainting formulation 虽然统一了 task，但在 I2V 这种"第一帧强 condition"场景可能不如专门优化的 I2V model。

### SkyReels-VABench（自己提的 benchmark，2000+ prompt）

5 个维度：
1. Instruction Following（指令遵循）
2. Audio-Visual Sync（音画同步）
3. Visual Quality（画质）
4. Motion Quality（动作质量）
5. Audio Quality（音质）

50 个专业评估员，5 分制绝对打分 + GSB 两两比较。

**结果**：SkyReels-V4 在 overall 平均分最高。最强的是 Prompt Following 和 Motion Quality。Visual Quality 跟 SOTA 持平。Audio-Visual Sync 和 Audio Quality 略优但优势不显著。

---

## 跟其他系统比，SkyReels-V4 强在哪

| 系统 | Multi-modal ref | Joint audio | Inpainting/Edit |
|------|----------------|-------------|-----------------|
| Veo 3.1 | 有限 | ✓ | 有限 |
| Sora 2 | 有限 | ✓ | × |
| Kling 3.0 | 有限 | ✓ | × |
| Kling-Omni | 图+视频 ref | × | 有限 |
| Seedance 2.0 | 有限 | ✓ | × |
| Runway Aleph | ✓ | × | ✓ |
| OmniHuman-1.5 | 音频驱动 | shallow fusion | × |
| **SkyReels-V4** | **图+视频+音频 ref** | **deep fusion** | **unified** |

**唯一一个同时做到三件事的**：multi-modal reference + joint audio + unified inpainting/editing。

---

## Appendix 里的应用案例

paper 的 Appendix A 展示了一堆应用，我挑几个有意思的：

1. **Multi-identity + multi-voice**：3 张人脸 ref + 3 段 audio ref，生成 3 个人对话场景——同时做 identity preserve 和 voice clone
2. **Motion transfer**：image ref 给人脸，video ref 给动作轨迹，生成新人脸做同样动作
3. **Subject inpainting**：mask 指定区域，把视频里的马换成鹿
4. **Attribute editing**：mask 指定领带区域，颜色变蓝
5. **Background editing**：保留前景，背景换成阿马尔菲海岸
6. **Watermark removal**：自动识别并去除水印/字幕/logo
7. **Style transfer**：转成剪纸风、乐高风
8. **Camera control**：把视频重渲染成右摇镜头
9. **Day→Night**：把白天场景改夜晚
10. **Expression transfer**：把参考视频里的人脸表情 transfer 到 image 里的人

这些都是 **一套 weight、一套 inference** 出来的，只需要改 prompt + mask + reference 组合。

---

## 我读完的几个直觉

### 1. Video diffusion 正在重走 LLM 的路

LLM 从 BERT（专门 pretrain）→ GPT-3（统一 generation）→ ChatGPT（instruction tuning + RLHF）走了 5 年。Video diffusion 现在正处在"统一 foundation model"这个节点。SkyReels-V4 把 generation/inpainting/editing 统一到一个 model，就像 GPT-3 把 completion/generation/infilling 统一。

### 2. Channel concat inpainting 会成为标准

这种 formulation 太优雅了——任务的多样性变成 mask 形状的多样性，一个 model 就能 multitask。我预测 2026 年大部分 video diffusion 都会采用这个 interface。

### 3. In-context reference 是 LLM 思路在 diffusion 的胜利

把 reference latent 直接 prepend 到 noisy latent，用 self-attention 做 in-context learning，这个 design pattern 会扩展到更多 modality。未来做 video + music + speech + sound effect joint generation，每个 modality 的 reference 都用类似机制 concat 进来。

### 4. Joint low-res + high-res keyframe 思路会普及

直接 1080p 15s 显存爆炸，这种"先低分辨率全覆盖 + 关键帧高分辨率 + Refiner 补全"的 cascade 思路会变成 1080p+ 长视频生成的标准。和 video codec 的 I/P/B frame 是同一个 idea。

### 5. Audio branch 从 scratch 训练有点可惜

Paper 说 audio backbone 是 from scratch 训的，没用 Stable Audio 之类的 pretrained audio model。我猜是因为要保持和 video branch 的 architectural symmetry（latent dimension、layer 数等都要一致才能做 bidirectional cross-attention）。如果未来能找到方法用 pretrained audio model 初始化 audio branch，audio quality 可能能进一步提升。

### 6. 几个潜在 limitation

- **I2V 上相对弱**：channel-concat inpainting 在"第一帧强 condition"上可能不如专门优化的 I2V
- **Reference 数量受限**：in-context reference 用 full self-attention，参考一多就贵
- **Audio quality 优势不显著**：虽然 deep fusion，但 audio branch 从 scratch 训可能限制 fidelity
- **Inference 还是贵**：1080p 15s + Refiner，real-time 还做不到

### 7. 未来方向猜测

- **AR + diffusion hybrid**：长视频（分钟级）可能需要 autoregressive extension
- **更多 modality**：text overlay、music score、sound effect 各自一个 stream，都用 dual-stream + cross-attention pattern
- **Sparse reference attention**：参考一多就 full attention 太贵，用 VSA 思路做 sparse
- **RLHF for video**：视频美学评估难自动化，可能需要 preference model + RLHF
- **World model 视角**：加 audio 之后 world simulator 多了 sound dimension，未来加 depth/physics/segmentation 可能形成更完整 world model
- **On-device 落地**：consistency model / LCM distillation + TensorRT-like inference framework

---

## 一句话总结

SkyReels-V4 把"视频生成 + 音频生成 + 编辑 + 修复 + 多模态参考"这一堆活儿，用一个 dual-stream MMDiT + channel-concat inpainting + MLLM instruction following 的统一架构干完了，还用 cascade + sparse attention 让 1080p/15s 在工程上可行。这是 video foundation model 走向 "unified multi-modal" 的一个标志性工作。

参考链接：
- Paper 原文（你已经有了）
- Flow Matching: https://arxiv.org/abs/2210.02747
- MMDiT (SD3): https://arxiv.org/abs/2403.03206
- VSA: https://arxiv.org/abs/2505.13389
- SyncNet: https://arxiv.org/abs/2203.14639
- Artificial Analysis Arena: https://artificialanalysis.ai/
- Kling-Omni: https://arxiv.org/abs/2512.16776
- Apollo: https://arxiv.org/abs/2601.04151
- LTX-2: https://arxiv.org/abs/2601.03233
- InstructPix2Pix: https://arxiv.org/abs/2211.09894
- Sora blog (world simulators): https://openai.com/research/video-generation-models-as-world-simulators

---

# SkyReels-V4 深度解析

## 整体 intuition

Andrej 你好！这篇 paper 我读了几遍，先从 high-level 的 intuition 开始构建。

**这篇 paper 的核心 thesis**: video generation 正在从"unimodal T2V + 后接 V2A"这种 decoupled pipeline，走向 **joint audio-visual foundation model**，同时要把 generation/inpainting/editing 三类任务统一到 **一个 architecture + 一套 weight** 里。这和 LLM 里"用一个 model 做 generation + editing + completion"的思路非常类似 —— 都是希望用 **inpainting formulation 作为通用 interface**。

为什么 inpainting 这么 powerful？因为：
- T2V = 全部 frame 都 generate（mask 全 0）
- I2V = 第 0 帧 condition（mask 只在第 0 帧 = 1）
- Video extension = 前 k 帧 condition
- Video editing = 任意 spatiotemporal mask
- Start-end interpolation = 第 0 和第 T-1 帧 condition

这其实就是把 video manipulation 重新 cast 成 **conditional generation problem**，类似 BERT 的 masked language modeling 思路延伸到 spatiotemporal latent space。一旦你接受这个 formulation，**task diversity 变成 mask shape 的 diversity**，单一 model 自然就能 multitask。

类似的，audio-visual joint generation 也是一个 **dual-stream symmetric architecture + bidirectional cross-attention** 的问题，本质上是希望两个 modality 在 depth 上贯穿性地 exchange synchronization cue，而不是只做 shallow fusion（cross-attention adapter 这种 lightweight mechanism）。

paper 的 contribution 可以总结为三句话：
1. **Dual-stream MMDiT + shared MLLM text encoder** —— 让 audio/video 两个 branch 都能理解 multi-modal instruction（text + image + video + audio reference）
2. **Channel-concat inpainting** —— 把 generation/editing 统一
3. **Joint low-res full + high-res keyframe + Refiner** —— 让 1080p/15s 在计算上 feasible

paper 自己也指出：to their knowledge，这是第一个同时 unify (i) multi-modal input (ii) joint video-audio generation (iii) generation/inpainting/editing 三件事的 model。我同意这是一个 meaningful 的里程碑，因为前面 Veo-3.1, Kling-3.0, Sora-2 都只做到部分。

---

## 1. Dual-Stream MMDiT 架构详解

### 1.1 整体结构

架构上是一个 **symmetric twin backbone**：
- Video branch: 从 pretrained T2V model 初始化
- Audio branch: 从 scratch 训练，但 **architectural specification 完全一致**

两个 branch 都共享同一个 frozen MLLM text encoder（处理 text + image + video + audio reference）。这个 shared encoder 是关键 —— 它让 "multi-modal instruction following" 变得自然，因为 audio branch 和 video branch 都能 access 同样的 semantic context，不需要分别 encode 各自的 conditioning。

### 1.2 Hybrid Dual-Stream + Single-Stream block 设计

每个 transformer block 内部是一个 hybrid：

**前 M 层（Dual-Stream）**: video/audio token 和 text token **分别** 走自己的 LayerNorm + QKV + MLP（参数独立），但 **self-attention 是 joint 的**：

公式(1): $\mathbf{Q}_v, \mathbf{K}_v, \mathbf{V}_v = \text{QKV}_v(\text{LayerNorm}_v(\mathbf{x}_v))$

公式(2): $\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t = \text{QKV}_t(\text{LayerNorm}_t(\mathbf{x}_t))$

公式(3): $\mathbf{x}'_v, \mathbf{x}'_t = \text{Attention}([\mathbf{Q}_v; \mathbf{Q}_t], [\mathbf{K}_v; \mathbf{K}_t], [\mathbf{V}_v; \mathbf{V}_t])$

**变量含义**:
- $\mathbf{x}_v$: video（或 audio，取决于 branch）的 token embedding，shape 是 $[N_v, d]$
- $\mathbf{x}_t$: text token embedding，shape $[N_t, d]$
- 下标 $v$ / $t$ 分别表示 video 和 text modality 的 **独立参数** LayerNorm 和 QKV projection
- $[\cdot ; \cdot]$ 是 **token dimension 上的 concatenation**（不是 channel）

**Intuition**: 这个设计的精妙之处在于 —— 早期 layer 让 video/audio 和 text 在同一个 attention pool 里互相 attend（这是 MMDiT 在 Stable Diffusion 3 / FLUX 里已经被验证有效的设计），保证 strong cross-modal alignment。但 parameters 是分开的，所以 video 和 text 各自的 representation space 在早期还能保持自己的 characteristic。

**后 N 层（Single-Stream）**: 直接把 $[\mathbf{x}_v; \mathbf{x}_t]$ 拼起来，用 **共享参数** 的 LayerNorm + QKV + MLP 处理。这是为了 parameter efficiency，避免 dual-stream 一直到末尾导致参数翻倍。

paper 说这个 hybrid 比 pure dual-stream 或 pure single-stream **收敛更快**。这个我直觉上能理解 —— 早期 alignment 需要独立的 normalizer 来稳定两个 modality 的 distribution，后期 alignment 已经做好了，可以 share weight 减少参数量。

### 1.3 Reinforced Text Conditioning via Cross-Attention

公式(4): $\mathbf{x}''_v = \mathbf{x}'_v + \text{Attention}(\mathbf{Q} = \mathbf{x}'_v, \mathbf{K} = \mathbf{x}_t, \mathbf{V} = \mathbf{x}_t)$

**Intuition**: 这个 cross-attention 是为了解决 single-stream 阶段 **text 语义可能被稀释** 的问题。在 single-stream block 里，text 和 video token 共享参数，到深层之后 text 的 signal 可能被 video 的 strong spatial/temporal pattern "absorb" 掉。所以补一个 cross-attention，让 video stream 主动 query text embedding，**强化 text guidance 在生成末端的控制力**。

这个设计让我想起 ControlNet 之于 UNet 的关系 —— 主干做 generation，旁路做 conditioning reinforcement。只不过这里更轻量，就是单层 cross-attention。

### 1.4 Bidirectional Audio-Video Cross-Attention

公式(5):
$$\mathbf{a}'_i = \mathbf{a}_i + \text{CrossAttn}(\mathbf{Q} = \mathbf{a}_i, \mathbf{K} = \mathbf{v}_i, \mathbf{V} = \mathbf{v}_i)$$
$$\mathbf{v}''_i = \mathbf{v}'_i + \text{CrossAttn}(\mathbf{Q} = \mathbf{v}'_i, \mathbf{K} = \mathbf{a}'_i, \mathbf{V} = \mathbf{a}'_i)$$

**变量含义**:
- $\mathbf{a}_i, \mathbf{v}_i$: 第 $i$ 层的 audio 和 video feature
- $\mathbf{v}'_i$ 是 video feature 在 reinforced text cross-attention 之后的版本（即公式4的输出）
- 下标 $i$ 表示 layer index

**Intuition**: 这是 paper 实现 audio-video synchronization 的核心机制。注意是 **bidirectional** —— 不只是 audio condition on video（V2A 方向），还有 video condition on audio（A2V 方向）。这种对称设计让两个 modality 在 **每一层** 都 exchange synchronization cue，而不是只在某个 layer 做 fusion。

paper 强调 architectural symmetry 让两个 modality 共享相同的 latent dimension $d$，**消除了中间 projection layer**，从而保留 unimodal pretraining 时学到的 attention structure。这点其实很关键 —— 如果强行用 projection 把 audio/video 维度对齐，相当于给 pretrained weight 加扰动，可能会破坏 alignment 的"good init"。

### 1.5 Temporal Alignment via RoPE Scaling

这是个细节但很重要的点。Video latent 是 21 frame，audio latent 是 218 token（44.1 kHz × 5s）。

为了让两个 modality 的 **temporal axis 在 attention 中对应起来**，paper 对 audio 的 RoPE frequency 做 scaling:

$$\text{scale} = 21 / 218 \approx 0.09633$$

**Intuition**: RoPE 的频率 $\omega_k = 10000^{-2k/d}$，时间索引 $t$ 经过 RoPE 后是 $e^{i\omega_k t}$。如果 audio 用原始的 $t_a \in [0, 218)$ 而 video 用 $t_v \in [0, 21)$，那么 audio 的"第 100 个 token"对应的 RoPE phase 和 video 的"第 10 个 token"是不同的，attention 时无法对齐。

把 audio 的 frequency 乘以 $21/218$，相当于把 audio 的 effective 时间索引压缩到 $[0, 21)$ 的范围。这样 audio 的 token $i$ 对应的 effective 时间是 $i \cdot 21/218$，video 的 token $j$ 对应时间 $j$，两者在 attention 时就能建立 **temporally consistent correspondence**。

这种 "RoPE scaling" 思路和 LLM 里做 long-context extension 时用的 NTK-aware RoPE scaling 类似 —— 都是通过调整 frequency 来 align 不同 resolution 的时间/位置轴。

### 1.6 Shared Multi-Modal Text Encoder

paper 用一个 **frozen MLLM** 作为 text encoder，处理一个 **concatenated prompt**（visual description + acoustic description 拼起来）。这个 MLLM 能接收 image、video、audio reference 作为输入，输出 multi-modal embedding。

**Intuition**: 这其实是个很聪明的设计。它把 "multi-modal instruction following" 这件事完全 delegate 给 MLLM，让 diffusion model 专注于 generation。MLLM 负责：
- 理解 "用 @image_1 的人做 @video_1 的动作" 这种复杂 instruction
- 输出统一的 multi-modal embedding 给 audio branch 和 video branch 共享
- 提供 in-context learning 的 semantic guidance

这和 Gemini-Nano 之于 Pixel、Qwen3-Omni 之于下游 task 的分工很像 —— LLM 做理解，diffusion 做生成。MLLM 在这里是 **frozen** 的，所以训练成本可控；下游 diffusion 学的是 **如何利用 MLLM 提供的 multi-modal embedding**。

参考：
- Stable Diffusion 3 / FLUX 的 MMDiT: https://stability.ai/news/stable-diffusion-3
- Qwen3-Omni (paper 里用作 audio captioning): https://arxiv.org/abs/2509.17765

### 1.7 Flow Matching Training Objective

公式(6):
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, z_v^0, z_a^0, \epsilon_v, \epsilon_a} \left[ \| \mathbf{v}_\theta^v(t, \mathbf{z}_v^t, \mathbf{z}_a^t, \mathbf{c}) - (\mathbf{z}_v^0 - \epsilon_v) \|^2 + \| \mathbf{v}_\theta^a(t, \mathbf{z}_a^t, \mathbf{z}_v^t, \mathbf{c}) - (\mathbf{z}_a^0 - \epsilon_a) \|^2 \right]$$

**变量含义**:
- $t \sim \mathcal{U}(0,1)$: timestep
- $\mathbf{z}_v^0, \mathbf{z}_a^0$: clean video 和 audio latent (VAE encoded)
- $\epsilon_v, \epsilon_a \sim \mathcal{N}(0, \mathbf{I})$: 两个独立的高斯噪声
- $\mathbf{z}_v^t = t \mathbf{z}_v^0 + (1-t)\epsilon_v$: 线性插值的 noisy latent（注意这里是 flow matching 的 **linear probability path**，不是 DDPM 的 forward process）
- $\mathbf{v}_\theta^v, \mathbf{v}_\theta^a$: 两个 branch 各自预测的 velocity field
- $\mathbf{c}$: conditioning information，包括 MLLM 的 multi-modal embedding + 可选的 spatial-temporal mask

**关键点**: $\mathbf{v}_\theta^v$ 接收 $\mathbf{z}_a^t$ 作为输入，$\mathbf{v}_\theta^a$ 接收 $\mathbf{z}_v^t$ 作为输入 —— 这就是 **joint training**，两个 branch 的 prediction 都依赖于对方的 noisy state，迫使它们学习 synchronized generation。

**Target 的形式**: $\mathbf{z}_v^0 - \epsilon_v$ 是 linear path 的 ground-truth velocity。推导一下：$\mathbf{z}_v^t = t \mathbf{z}_v^0 + (1-t)\epsilon_v$，对 $t$ 求导得 $d\mathbf{z}_v^t/dt = \mathbf{z}_v^0 - \epsilon_v$。所以 flow matching 是回归这个 velocity field。

**Intuition**: 与 DDPM (predict noise $\epsilon$) 不同，flow matching 直接预测从 noise 走向 data 的"速度"。好处是 **trajectory 更线性、更适合用 ODE solver**，inference 时步数可以更少（典型 50 步甚至更少 vs DDPM 1000 步）。这也是为什么 SD3 / FLUX / Sora 都用 flow matching。SkyReels-V4 这个 **joint loss** 把两个 modality 的 velocity prediction 耦合在一起，梯度同时回传给 video branch 和 audio branch，确保它们学到 synchronized。

参考：
- Flow Matching for Generative Modeling (Lipman et al.): https://arxiv.org/abs/2210.02747
- Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206

---

## 2. Unified Video Inpainting via Channel Concatenation

### 2.1 核心公式

公式(7): $\mathbf{Z}_{\text{input}} = \text{Concat}(\mathbf{V}, \mathbf{I}, \mathbf{M})$

**变量含义**:
- $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$: noisy video latent（要 denoise 的目标）
- $\mathbf{I} \in \mathbb{R}^{T \times H \times W \times C}$: VAE-encoded conditional frame（非 condition 的 frame 用 black image latent 填充）
- $\mathbf{M} \in \mathbb{R}^{T \times H \times W \times 1}$: binary mask，1 表示 condition（保留原内容），0 表示 generate
- Concat 是沿 **channel dimension**，所以最终 input channel 是 $2C + 1$

### 2.2 各种 task 的 mask 配置

| Task | Mask 配置 |
|------|----------|
| T2V | $\mathbf{M} = \mathbf{0}$ |
| I2V | $M_{t=0} = 1, M_{t>0} = 0$ |
| Video extension | $M_{t<k} = 1, M_{t\geq k} = 0$ |
| Start-end interpolation | $M_{t=0} = M_{t=T-1} = 1$, others 0 |
| Video editing | 任意 spatiotemporal mask |

**Intuition**: 这个 formulation 的妙处在于 **不需要 task-specific token 或 task-specific adapter**。所有 task 都通过 (noisy latent, conditional latent, mask) 这三元组表达，model 自己学会根据 mask pattern 推断 task 类型。这和 InstructPix2Pix 把 editing 用 (original image, edited image, instruction) 三元组训练的思路是一脉相承的 —— 都是把 task 编码进 input tensor 而不是 input token。

mask 是 $T \times H \times W \times 1$，比 latent 的 $C$ 通道少很多，**信息密度低**，但足够 sparse 地指出 "哪里要 generate"。这是空间上非常 efficient 的 conditioning。

### 2.3 Audio 在 inpainting 时的行为

paper 指出一个重要设计选择：**inpainting 机制只 apply 到 video stream**，audio branch 在 inpainting/editing task 时是 **从 scratch 生成 audio**，condition 在（部分 condition 或 edited 的）video content 上。

**Intuition**: 这是因为 audio 通常和 video 的 visual content 强 coupling（lip-sync、sound effect、environment sound），所以 video 被编辑后，audio 必须重新生成才能保持一致。这个设计避免了 "保留原 audio + 编辑 video" 导致的 desync。Bidirectional cross-attention 让 audio 生成时能 attend 到编辑后的 video，自然产生 acoustic consistency。

参考：
- InstructPix2Pix: https://arxiv.org/abs/2211.09886
- ControlNet video inpainting: https://arxiv.org/abs/2307.07271

---

## 3. Multi-Modal In-Context Learning for Vision-Referenced Generation

这部分是 paper 最有意思的设计之一，因为它把 LLM 的 in-context learning 思路搬到了 video diffusion 里。

### 3.1 Multi-Modal Instruction Following via MLLM

例子 prompt：
```
"generate a video of person A from the reference @image_1 
speaking <dialogue>hello, how are you?</dialogue> 
in the style of person B's @video_1"
```

MLLM 接收这种 prompt + @image_1 + @video_1，输出 multi-modal embedding，audio 和 video branch 都用这个 embedding 作为 conditioning。

**Intuition**: 这种 "用 @reference 指代" 的 prompt 格式和 ChatGPT 的 image input 非常像，让用户用自然语言 compose 多个 reference。MLLM 在这里做的是 **reference binding** —— 把 @image_1 这个符号 binding 到具体 image feature，把 "person A" 这个描述 binding 到 @image_1。这本质上就是 LLM 的 in-context learning，只不过 context 里多了 image 和 video。

### 3.2 In-Context Visual Conditioning via Self-Attention

公式(8): $\mathbf{Z}_{\text{attn}} = [\mathbf{Z}_{\text{cond}}; \mathbf{Z}_{\text{video}}]$

**变量含义**:
- $\mathbf{Z}_{\text{cond}}$: reference image/video frame 经过 VAE encode、pad 到统一 spatial resolution、沿 temporal dimension concat 之后的 latent
- $\mathbf{Z}_{\text{video}}$: noisy video latent（要生成的）
- $[\cdot ; \cdot]$: 沿 **temporal dimension** 的 concatenation

然后这个拼接后的 sequence 走 joint self-attention。

**Intuition**: 这是 paper 的关键 insight —— reference visual signal 不能只通过 MLLM 的 semantic embedding 传递（semantic embedding 太抽象，丢失 fine-grained visual pattern），还需要 **直接把 reference latent 拼到 noisy latent 前面**，让 self-attention 能 access 到 pixel-level 的 reference。

这其实就是把 LLM 的 in-context learning 思路用到 diffusion：reference image 是 "context tokens"，noisy video 是 "query tokens"，attention 让 query 直接 attend 到 context。这种设计让 model 能 "看到" reference 的 texture、identity、pose 等 fine-grained 特征。

### 3.3 Offset 3D RoPE for Temporal Disambiguation

公式(9):
$$\text{RoPE}_{\text{temporal}}(\mathbf{Z}_{\text{cond}, i}) = \text{RoPE}(t = -N_{\text{cond}} + i)$$
$$\text{RoPE}_{\text{temporal}}(\mathbf{Z}_{\text{video}, j}) = \text{RoPE}(t = j)$$

**变量含义**:
- $N_{\text{cond}}$: condition token 总数（reference images/frames 的 token 数）
- $i \in [0, N_{\text{cond}})$: condition token index
- $j \in [0, T)$: video frame index
- $\text{RoPE}(t = -k)$: 把时间索引设为 $-k$，即 **negative temporal index**

**Intuition**: 这是个非常 elegant 的设计。如果 condition 和 video 都从 $t=0$ 开始，RoPE 会把它们当同一时间点的 token，attention 时可能混淆。给 condition 用 **negative index**，相当于把 reference "放到 past"，video "放到 future"，model 能学到 "reference 是 context，video 是 generation target" 的归纳偏置。

paper 还说 spatial index 在所有 token 间保持一致 —— 这意味着 model 知道 reference 的空间结构和 video 的空间结构是同一坐标系，attention 时能建立 spatial correspondence。

这种 "negative temporal index for context" 的思路让我想起 GPT 的 causal mask —— 都是用 positional encoding 来 establish "context vs target" 的结构。

### 3.4 Audio Reference Conditioning

类似的，audio reference (speech sample, music theme, ambient soundscape) 也作为 in-context condition 给 audio branch。这让 model 能 "voice cloning" 或 "music style transfer"。

---

## 4. Training Strategy

paper 用 **progressive multi-stage training**，分三个大阶段：Video Pretrain → Audio Pretrain → Joint Training + SFT。Table 1 详细列了每个 stage 的 task / resolution / data volume / epochs。

### 4.1 Video Pretrain (6 stages)

| Stage | Task | Resolution | Data | Epochs |
|-------|------|-----------|------|--------|
| 1 | T2I | 256px | 3B images | 3 |
| 2 | T2I + T2V | 256px, 16fps, 2-10s | 1B images + 400M videos | 3 |
| 3 | T2I + T2V + Inpaint | 256px, 16fps, 2-15s (Inpaint 5% each) | 1B images + 400M videos | 2 |
| 4 | Mixed Tasks | 256/480px, 16fps, 2-15s | 100M images + 100M videos | 2 |
| 5 | Mixed Tasks | 480/720/1080px, 16fps, 3-15s | 50M images + 50M videos | 2 |
| 6 | Multi-modal Condition (Image/Video Ref 20% each, T2V 60%) | 480/720/1080px, 16fps, 3-15s | 20M images + 50M videos | 2 |

**Intuition**: 这个 schedule 的关键 insight 是 **"先 low-res 学 motion dynamics, 再 high-res 学 detail"**。Stage 2-3 在 256px 学 temporal coherence 是因为这个分辨率下计算便宜、迭代快，model 能快速学会 motion。Stage 5 才上 high-res，这时 motion 已经学好，只需要 refine spatial detail。

Stage 1 的 T2I pretrain 是个 trick —— **先学 spatial composition 再学 temporal**，这样 video model 在 Stage 2 引入时间维度时，spatial prior 已经稳了，能加速 video convergence。这个 insight 在 CogVideoX、HunyuanVideo 里也有类似实践。

Stage 6 引入 multi-modal reference (image/video reference) 是关键 stage —— 在这个 stage 之前 model 还不知道怎么处理 @image_1 这种 reference。20% reference data + 60% T2V 的配比说明 reference-conditioned 是 auxiliary task，不是主导。

### 4.2 Audio Pretrain

- 从 scratch 训练，几百千小时 speech data
- variable-length audio（up to 15s），maximize duration coverage
- 目标：generate consistent audio that respects speaker traits (pitch, emotion)

**Intuition**: 这里说 audio backbone 是 from scratch 的，没有用 pretrained audio model —— 这有点意外，因为 audio diffusion model 已经有不少工作（AudioLDM, Stable Audio 等）。可能是因为 architectural specification 要和 video branch **完全一致** 才能做 bidirectional cross-attention，所以不能用现成的 audio model weight。

variable-length 训练的好处是让 model 学到 **speaker consistency across duration**，这对 lip-sync 和长对话场景很重要。

### 4.3 Joint Training + SFT

- Joint Pretrain: T2V + T2AV + T2A 同时训，50% video data + T2A data，720/1080px, 5-15s
- SFT Stage 1: 5M videos，multi-modal condition 20%
- SFT Stage 2: 1M manually curated high-quality videos

**Intuition**: Joint training 50% video data 用于 T2AV —— 这意味着 T2V 和 T2AV **share 一半 data**，所以 model 不会因为 joint training 而"忘记" T2V 的能力。同时引入 T2A（纯音频生成）是为了让 audio branch 也能独立生成，不只是 "video 的附属"。

SFT Stage 2 的 1M curated 数据是关键 —— 这相当于 RLHF 之前的 SFT 阶段，用 high-quality 数据 refine model 的 aesthetic 和 motion quality。

参考：
- CogVideoX (类似 progressive training): https://arxiv.org/abs/2408.06072
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- WAN-2.1: https://arxiv.org/abs/2503.20314

---

## 5. Refiner: Super-Resolution + Frame Interpolation

这是 paper 的 efficiency strategy 的核心，让 1080p/15s 在计算上 feasible。

### 5.1 高层 idea

**不做直接 1080p generation**，而是：
1. Base model 同时预测：**low-resolution full sequence** + **high-resolution keyframes**
2. Refiner 接收 base model 输出 + high-resolution multi-modal condition + text instruction，做 SR + frame interpolation

### 5.2 Refiner 的 input

Refiner (也是个 DiT) 接收三类 input:
1. Multi-modal visual conditions (image, video, audio reference at high-res)
2. Multi-modal text instructions（和 base model 一致）
3. Base model 输出：low-res predictions for all frames + high-res predictions for keyframes

### 5.3 Latent 组合逻辑

1. **线性插值** low-res latent 到 target high-res
2. 在 keyframe 位置，**用 base model 直接预测的 high-res keyframe latent 替换** interpolated result
3. 把这些 combined latent 和 high-res noisy latent **沿 channel concat** 作为 DiT input

**Intuition**: 这个设计的关键是 **keyframe 提供 high-frequency anchor，interpolation 提供 low-frequency temporal coherence**。Base model 不需要生成全部 1080p frame（计算太贵），只需要在 keyframe 上"用力"，剩下的 frame 由 Refiner 通过 SR + interpolation 补全。这和 video codec 里 I-frame + P-frame/B-frame 的思路很像 —— keyframe 是 I-frame（high quality, expensive），其他 frame 是 P/B（cheap, derived from keyframe）。

### 5.4 Inpainting 在 Refiner 中的支持

Refiner 也支持 inpainting：用 high-res source video 替换 interpolated region 中不需要 inpaint 的部分，用 spatial mask 区分 "需要 refine" vs "保持不变"。这让 Refiner 既能做 unconditional SR，也能做 conditional inpainting。

### 5.5 Video Sparse Attention (VSA)

Refiner 因为要处理 high-res 长序列，attention 计算开销大，paper 用了 **Video Sparse Attention** (VSA, ref [59])：

- **Coarse stage**: aggregate spatial-temporal cube，用 lightweight pooled attention 找出 critical token region
- **Fine stage**: 只在 top-K cube 内做 dense attention

这通过 block-sparse layout 实现硬件友好的稀疏 attention，**减少 ~3× attention 计算成本**。

**Intuition**: video 的 spatiotemporal redundancy 很高（相邻 frame、相邻 pixel 通常 attend 到类似的 region），所以 full quadratic attention 是浪费的。VSA 通过 learnable 的 sparse pattern 把 attention 集中在 information-rich 的 cube 上，既省计算又保留 quality。

参考：
- VSA paper: https://arxiv.org/abs/2505.13389
- Video codec I/P/B-frame: https://en.wikipedia.org/wiki/Video_compression_picture_types

---

## 6. Data Pipeline

paper 用了相当复杂的 data pipeline，三模态（image/video/audio）分别处理。

### 6.1 Data Collection

- **Real-world**: public dataset (LAION, Flickr, WebVid-10M, Koala-36M, OpenHumanVid, Emilia, AudioSet, VGGSound, SoundNet) + licensed in-house (movies, TV series, short videos)
- **Synthetic**: 补 sparse scenario
  - 多语言 text rendering (中、英、日、韩、德、法)
  - 多语言 TTS speech
  - Multi-modal inpainting/editing task (用 segmentation + editing model + controllable generation 合成 paired data)

**Intuition**: synthetic data 主要补三类缺口：(i) 多语言 (ii) rare pronunciation (iii) inpainting/editing 的 paired data。第 (iii) 类尤其重要 —— 因为 inpainting/editing 需要 "input video + edited output" 配对，real-world 数据没有这种 pair，必须用 pipeline 合成。

### 6.2 Audio-Video Synchronization

paper 用 **SyncNet** (ref [58]) 来 filter desynchronized speech video：
- SyncNet 用 ConvNet 学 sound 和 mouth image 的 joint embedding
- 保留满足 $|\text{offset}| \leq 3 \wedge \text{confidence} > 1.5$ 且 minimum mean volume > -60 dB 的 clip

**Intuition**: 这是 audio-visual joint training 的关键 data curation。如果训练数据本身 desync，model 永远学不到 lip-sync。SyncNet filter 保证了 training signal 的 quality。$|\text{offset}| \leq 3$ 是 frame-level tolerance，相当于 ~100ms（at 30fps）—— 这比人眼能感知的 desync threshold (~80ms) 略松，是 reasonable 的 threshold。

### 6.3 Captioning

三种 caption:
- **Short**: 简短描述
- **Long**: 详细描述 (environment, subject, lighting, atmosphere)
- **Structured**: 标准化 order + special token (`<text>`, `<sfx>`, `<dialogue>`, `<singing>`, `<bgm>`)

最终 training stage **只用 structured caption**。用 prompt enhancer 把 free-form user prompt reformat 成 structured representation。

**Intuition**: structured caption 的好处是让 model 能区分 "video 里的字" / "speech content" / "background music" —— 这些都是 audio-visual joint generation 的关键 signal。`<dialogue>hello</dialogue>` 这种 tag 让 model 知道 "hello" 是要被 spoken 出来的，而不是显示在屏幕上的 text。这和 MusicGen 用 Music Description Language 类似 —— 都是把 caption 结构化以提高 control。

参考：
- SyncNet: https://arxiv.org/abs/2203.14639
- MusicGen: https://arxiv.org/abs/2306.05284

---

## 7. 实验 Results 分析

### 7.1 Artificial Analysis Arena

paper 在 4 个 track 上 evaluate (as of 2026-03-18):

| Track | SkyReels-V4 Rank |
|-------|-----------------|
| T2V + Audio | **#1** (Elo 1129) |
| T2V no Audio | #2 (Elo 1236) |
| I2V + Audio | #4 (Elo 1067) |
| I2V no Audio | #7 (Elo 1279) |

**Key observation**: SkyReels-V4 在 audio joint track 上 #1，超过 Kling 3.0 (1097) 和 Veo 3.1 (1080)。但在 no-audio track 上反而不如 Kling 3.0 (#1, 1248)。这暗示 **SkyReels-V4 在 audio+video joint 上特别强**，但纯 visual generation 上 Kling 3.0 略胜一筹。

这其实是一个 reasonable 的 trade-off —— SkyReels-V4 把大量 capacity 用在 audio-visual joint 上，所以纯 T2V 上可能略弱于 "T2V-first" 的 Kling 3.0。但 audio joint 场景下它的优势就体现出来了。

I2V track 上 SkyReels-V4 相对弱（#4 with audio, #7 without audio）—— 这可能是因为 I2V 需要更强的 image conditioning，而 SkyReels-V4 的 channel-concat inpainting formulation 虽然 unify 了 task，但可能在 I2V 这种 "first frame strong condition" 的 case 上不如专门优化的 I2V model。

### 7.2 SkyReels-VABench Human Evaluation

paper 提出 SkyReels-VABench（基于之前的 SkyReels-Bench），2000+ prompt，覆盖 advertising / social / narrative / educational / entertainment。

5 个 evaluate dimension (Table 2):
1. **Instruction Following**: video instruction + audio instruction
2. **Audio-Visual Synchronization**: lip-sync, sound effect alignment, atmospheric matching, spatial audio
3. **Visual Quality**: clarity, color, composition, structure, physical plausibility
4. **Motion Quality**: fluidity, stability, temporal consistency, vividness
5. **Audio Quality**: artifacts absence, spatial soundstage, timbre, signal clarity, dynamic range

Evaluation methodology:
- 50 professional evaluators
- Absolute scoring (5-point Likert)
- GSB pairwise comparison

### 7.3 Results

- **Absolute scoring**: SkyReels-V4 在 5 个 dimension 中 overall 最高
  - 最强: Prompt Following, Motion Quality
  - 持平 SOTA: Visual Quality
  - 略优: Audio-Visual Sync, Audio Quality

- **GSB comparison**: 对比 Veo 3.1, Kling 2.6, Seedance 1.5 Pro, Wan 2.6
  - SkyReels-V4 在 overall quality 上对每个 baseline 都有更高 "Good" 比例

**Intuition**: 这些 result 验证了 dual-stream MMDiT + bidirectional cross-attention 的设计在 audio-visual joint 上是 effective 的。Motion Quality 的强势可能来自 progressive training + multi-stage schedule。Audio-Visual Sync 略优但不显著 —— 可能因为 baseline 也都在这方面投入了 effort。

参考：
- Artificial Analysis: https://artificialanalysis.ai/
- SkyReels-V2 (previous version): https://arxiv.org/abs/2504.13074

---

## 8. 与其他 system 对比

paper 里 mention 了不少 concurrent work，我帮理一下 landscape:

| System | Multi-modal input | Joint audio-video | Inpainting/Edit | Resolution |
|--------|------------------|-------------------|-----------------|-----------|
| Veo 3.1 [1] | Limited | ✓ | Limited | 1080p |
| Sora 2 [2] | Limited | ✓ | × | 1080p |
| Kling 3.0 [17] | Limited | ✓ | × | 1080p |
| Kling-Omni [16] | Image+Video ref | × | Limited | 1080p |
| Seedance 2.0 [18] | Limited | ✓ | × | 1080p |
| Vidu Q3 [19] | Limited | ✓ | × | 1080p |
| Runway Aleph [8] | ✓ | × | ✓ | - |
| OmniHuman-1.5 [10] | Audio-driven | ✓ (shallow fusion) | × | - |
| MultiTalk [15] | Audio-driven | ✓ (shallow fusion) | × | - |
| LTX-2 [36] | Asymmetric streams | ✓ | × | - |
| Apollo [45] | Single-tower | ✓ (Omni-Full Attn) | × | - |
| **SkyReels-V4** | **Text+Image+Video+Mask+Audio** | **✓ (deep fusion)** | **✓ (unified)** | **1080p/32fps/15s** |

**Key differentiation**: 
1. **Deep fusion vs shallow fusion**: OmniHuman/MultiTalk 用 cross-attention adapter 这种 shallow fusion，alignment 不到位。SkyReels-V4 用 bidirectional cross-attention in every layer 这种 deep fusion。
2. **Unified inpainting**: 大部分 audio-visual model 都不做 inpainting/editing。Runway Aleph 做 editing 但不 joint audio。
3. **Multi-modal reference**: Kling-Omni 支持 image+video reference 但不 joint audio。SkyReels-V4 同时支持 image/video/audio reference + joint audio。

参考：
- Kling-Omni: https://arxiv.org/abs/2512.16776
- Apollo: https://arxiv.org/abs/2601.04151
- LTX-2: https://arxiv.org/abs/2601.03233
- Runway Aleph: https://runwayml.com/research/introducing-runway-aleph
- OmniHuman-1: https://arxiv.org/abs/2502.01061
- MultiTalk: https://arxiv.org/abs/2505.22647

---

## 9. Application Examples (Appendix A)

paper 的 Appendix A 展示了大量应用，我把它们分类整理：

### 9.1 Generation
- **Multiple image + audio reference**: 用多张人脸 image + 多个 audio reference 生成多角色对话场景（example 6）—— 这相当于 multi-identity + multi-voice cloning
- **Image + motion reference**: 用 image 定 content/style，用 video reference 定 motion（pose, trajectory）—— 这是 video-to-video motion transfer

### 9.2 Inpainting
- **Subject/Attribute/Background inpainting**: 用 mask 指定 region，replace subject（elk）、change attribute（tie color）、replace background
- **Image reference inpainting**: 用 reference image 指导 inpainting content（"add the man from @image_1 to mask area"）

### 9.3 Editing
- **Local editing**: watermark/subtitle/logo removal（这个用得最多），subject manipulation (add/remove object), local attribute editing (change color/texture/shape), background editing (preserve foreground)
- **Global editing**: style transfer (Paper-Cutting, LEGO), camera control (pan right), scene attributes (day→night)
- **Reference-based editing**: subject ref + motion ref, subject ref + expression ref, background ref + video ref, first-frame ref + effect ref

**Intuition**: 这些 application 体现了 unified inpainting formulation 的真正威力 —— 一套 weight + 一套 inference pipeline 能覆盖几乎所有 video manipulation task。用户不用为不同 task 切换 model，只用调整 prompt + mask + reference 的组合。

特别是 **"subject reference with expression reference"**（Figure 17: transfer facial expression from @video_1 to man in @image_1）—— 这是 deepfake 风格的应用，体现了 multi-modal in-context learning 的强大之处。这种 capability 之前需要专门训的 face reenactment model (比如 SadTalker, Talking Head) 才能做，现在 unified model 直接 cover。

参考：
- SadTalker (face reenactment): https://arxiv.org/abs/2211.12194
- InstructPix2Pix video editing: https://arxiv.org/abs/2307.07271

---

## 10. My Takeaways & Intuition

读完之后我有几个 observation：

### 10.1 关于 architecture 的 intuition

**Dual-stream MMDiT 的对称性是关键**。Audio 和 video 用 **完全一样的 architectural spec** + **相同的 latent dimension**，这样 bidirectional cross-attention 不需要 projection layer，保留 unimodal pretrain 的 alignment。这个 symmetric design pattern 我觉得会成为 future multi-modal diffusion model 的标准 —— 因为非对称设计总是要引入 projection/adapter，破坏 alignment。

**Hybrid dual-stream + single-stream block** 是 parameter efficiency 和 modality alignment 的 sweet spot。早期 dual-stream 让两个 modality 各自保持 representation space，后期 single-stream 让参数 share 减少 cost。这种 "early modality-specific, late modality-shared" 的思路和 MoE / multimodal LLM 里的 design pattern 一致。

### 10.2 关于 inpainting formulation 的 intuition

**Channel concatenation inpainting** 这种统一 formulation 我觉得会成为 video diffusion 的"标准 interface"。它和 LLM 里的 "fill-in-the-middle" 任务非常类似 —— 都是 mask + context → generation。这种 formulation 的好处是 **task diversity 变成 mask pattern 的 diversity**，一个 model 能 multitask。

更进一步思考：mask 是 $T \times H \times W \times 1$，非常 sparse。如果未来要支持更复杂的 instruction（比如 "把这个人变成卡通风格但保留背景"），可能需要更 expressive 的 conditioning，比如 multi-channel mask（每种 edit 类型一个 channel），或者 textual description per-region。

### 10.3 关于 multi-modal in-context learning 的 intuition

**In-context visual conditioning via self-attention** 这部分最让我兴奋。它本质上把 LLM 的 in-context learning 思路搬到了 diffusion：
- Reference latent = context tokens
- Noisy video latent = query tokens
- Self-attention 让 query attend 到 context
- Offset 3D RoPE 用 negative index 表示 "context is past, generation is future"

这种 design pattern 我觉得会延伸到更多 modality。比如未来做 video+music+speech+sound effect joint generation，每个 modality 的 reference 都用类似 mechanism concat 到 noisy latent 前面，统一通过 self-attention 做 in-context conditioning。

### 10.4 关于 efficiency 的 intuition

**Joint low-res + high-res keyframe + Refiner** 这个思路非常工程实用。它本质上把 "1080p 15s generation" 这个 expensive 的问题 decompose 成：
- Base model: 算 cheap 的事（low-res full sequence + sparse high-res keyframe）
- Refiner: 算 expensive 的事但用 sparse attention 加速（high-res SR + interpolation）

这个 decompose 思路其实和 video codec (I/P/B frame)、cascaded diffusion (deepfusion)、progressive rendering 都有思想上的相似 —— **先用 cheap representation 抓住 low-frequency structure，再用 expensive refine high-frequency detail**。

### 10.5 关于 limitation 的思考

paper 没有 explicit discussion limitation，但我可以从 result 推断：
1. **I2V 上相对弱** (#7 without audio) —— channel-concat inpainting 虽然统一 task，但 I2V 这种 "first-frame strong condition" 可能需要更强的 first-frame attention mechanism
2. **Audio Quality 优势不显著** —— bidirectional cross-attention 虽然 deep fusion，但 audio branch 从 scratch 训练可能限制了 audio fidelity。如果能用 pretrained audio model (比如 Stable Audio) init，audio quality 可能更高
3. **Reference 数量限制** —— in-context visual conditioning 通过 temporal concat 实现，reference 越多，attention 的 token 数线性增长，可能限制 multi-reference 场景
4. **Computation cost** —— 虽然 Refiner + VSA 优化了，但 1080p 15s 仍然是 expensive inference，real-time 应用可能 still challenging

### 10.6 对未来方向的联想

顺着 paper 的思路，我觉得几个方向值得 explore：

1. **Autoregressive + diffusion hybrid**: 现在 SkyReels-V4 是 pure diffusion，但 long video generation (分钟级) 可能需要 autoregressive extension。MAGI-1 [37] 已经在探索 autoregressive video。Combine AR (for long context) + diffusion (for frame quality) 可能是 next step。

2. **More modality joint**: 现在是 audio + video，未来可能加 text overlay generation, music score, sound effect, 甚至 haptic signal。每个 modality 都用 dual-stream + bidirectional cross-attention 这个 pattern 可以 scale。

3. **Reference attention 的 sparse 化**: 现在 in-context reference 用 full self-attention，参考一多就 expensive。可能用 VSA 类似思路做 sparse reference attention，让 model 能 handle 几十个 reference。

4. **Mask 作为可学习的 soft conditioning**: 现在 mask 是 binary，未来可能用 soft mask（连续 probability）+ learnable mask prediction，让 model 自己 decide 哪里 edit。

5. **RLHF for video**: 现在 SFT 用 curated data，但 video aesthetic 评估很难自动化。可能需要 RLHF with human preference model，类似 LLM 的 RLHF 流程。

6. **World model 视角**: paper 提到 "video generation models as world simulators" (Sora 的 thesis)。SkyReels-V4 加了 audio，相当于 world simulator 多了 sound dimension。未来继续加 modality（depth, segmentation, physics）可能形成更完整的 world model。

7. **On-device 落地**: 1080p/15s + Refiner + VSA 虽然加速了，但要 real-time on consumer GPU 仍然 hard。可能需要 distillation (consistency model, LCM) 或专门的 inference framework (类似 TensorRT for diffusion)。

---

## 11. 公式汇总与变量速查

| 公式 | 含义 | 关键变量 |
|------|------|---------|
| (1)-(3) | Dual-stream MMDiT block: 独立 QKV + joint attention | $\mathbf{x}_v, \mathbf{x}_t$: video/text token; $v, t$: modality-specific params |
| (4) | Reinforced text cross-attention | $\mathbf{x}'_v$: video after self-attn; $\mathbf{x}_t$: text embedding |
| (5) | Bidirectional audio-video cross-attention | $\mathbf{a}_i, \mathbf{v}_i$: audio/video feature at layer $i$ |
| (6) | Flow matching joint loss | $t \in [0,1]$: timestep; $\mathbf{z}^0, \mathbf{z}^t, \epsilon$: clean/noisy/noise latent; $\mathbf{c}$: conditioning |
| (7) | Channel concat for inpainting | $\mathbf{V}, \mathbf{I}, \mathbf{M}$: noisy/conditional latent + mask |
| (8) | In-context visual conditioning via temporal concat | $\mathbf{Z}_{\text{cond}}, \mathbf{Z}_{\text{video}}$: reference and noisy latent |
| (9) | Offset 3D RoPE | $N_{\text{cond}}$: # of condition tokens; $i, j$: condition/video index |

---

## 12. 最终总结

SkyReels-V4 是一个相当 solid 的工作，三个核心 contribution 都有明确的 technical novelty 和 engineering value：

1. **Dual-stream MMDiT + bidirectional cross-attention**: 把 audio-visual joint generation 从 shallow fusion 推进到 deep fusion
2. **Channel-concat inpainting**: 把 generation/editing 统一到一个 interface，这个 formulation 我觉得会成为 video diffusion 的标准
3. **Joint low-res + high-res keyframe + Refiner**: 让 1080p/15s 在计算上 feasible

实验 result 也验证了设计 —— Artificial Analysis Arena T2V+Audio #1, SkyReels-VABench overall 最高。

我个人的 takeaway 是：**video generation 正在走向 "unified multi-modal foundation model" 这个 paradigm**，类似 LLM 在 2020 年走向 GPT-3 时的范式转变。SkyReels-V4 是这个 paradigm 的 early but very promising instantiation。未来 1-2 年我会 expect 看到：
- 更多 modality 加入 (text overlay, music score, etc.)
- 更长的 duration (minutes+) via autoregressive
- 更好的 controllability via RLHF / preference learning
- 更强的 efficiency via distillation / sparse attention / quantization

希望这个分析对你 build intuition 有帮助，Andrej！如果对某个细节想 deep dive，告诉我具体哪部分，我可以再展开。

参考链接汇总：
- SkyReels-V4 paper (附在这条 message 里)
- Flow Matching: https://arxiv.org/abs/2210.02747
- MMDiT (SD3): https://arxiv.org/abs/2403.03206
- VSA: https://arxiv.org/abs/2505.13389
- SyncNet: https://arxiv.org/abs/2203.14639
- Kling-Omni: https://arxiv.org/abs/2512.16776
- Apollo: https://arxiv.org/abs/2601.04151
- LTX-2: https://arxiv.org/abs/2601.03233
- CogVideoX: https://arxiv.org/abs/2408.06072
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- WAN-2.1: https://arxiv.org/abs/2503.20314
- Artificial Analysis: https://artificialanalysis.ai/
- Qwen3-Omni: https://arxiv.org/abs/2509.17765
- InstructPix2Pix: https://arxiv.org/abs/2211.09894
- Sora world simulators: https://openai.com/research/video-generation-models-as-world-simulators
