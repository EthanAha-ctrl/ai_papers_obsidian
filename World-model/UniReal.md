---
source_pdf: UniReal.pdf
paper_sha256: 08ad9a4ea217ce4a7a0da7966ea5263b978ada9e50c60e85854f4e92af9829f5
processed_at: '2026-08-12T20:06:12-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniReal 的人话版

## 一句话总结

既然 video model 已经很擅长在连续帧之间保持"什么变了、什么没变"，那 image editing 本质上也是同一件事——给几张图，生成几张图，变一些东西、保一些东西。所以干脆把所有 image task 塞进一个 video model 的框架里跑。

## 为什么这个 idea 成立

想想 image editing 到底在干什么：

- Instruct editing：给一张图，改一点，保一点
- Customization：给 reference object，放到新场景
- Object insertion：给 object + background，融合
- Controllable generation：给 condition map，生成图

每个 task 表面不同，但底层都是同一个问题：**有一些 input images 提供约束，要生成 output images，既保持一致性又引入变化**。

而 video generation model（像 Sora）每天在练的就是这件事——第 1 帧到第 2 帧，什么该保持、什么该变。所以这个 paper 的核心 realization 就是：image editing 其实是 "discontinuous video generation"，frames 之间不连续但逻辑上相关。

## 怎么统一

关键 trick 很简单：**把所有 input 和 output images 都当作 video 的 frames**，用 full attention 让它们互相看。

区分不同 image 用两个东西：
- **Index embedding**：告诉模型 "这是 IMG1、IMG2"
- **Image prompt**：告诉模型 "这张图的角色是什么"——是 canvas（要被编辑的底图）、asset（要插入的 reference object）、还是 control（mask/depth 等条件图）

文本里可以用 "put the dog from IMG1 onto the grass in IMG2" 精确指代，因为 T5 tokenizer 里加了 IMG1/IMG2/RES1 这些 special token。

## 为什么要搞 hierarchical prompt

多 task 混训最大的问题是 ambiguity。同样输入 "一张狗的图 + 文本 put it on grass"：

- Editing task：在原图里换背景，保持狗的 layout
- Customization task：生成新场景，只保狗的外观

模型不知道你要哪种。所以加了一层 context prompt 标注 "这是 editing 还是 customization"、"realistic 还是 synthetic style"、"static 还是 dynamic"。

妙的地方在于这些 keyword 跨 task 共享。"realistic style" 这个 tag 在 editing 和 customization 都出现，强迫模型学到通用概念，而不是每个 task 各学各的。

## 数据怎么来

最 genius 的部分在这。不用人工标注，直接从 video 里挖：

随便抽 video 的两帧——前帧当 "before edit"，后帧当 "after edit"，video 的 caption 当 instruction。这 8M 对数据直接就能教模型做 editing。

为什么 work？因为 video 帧之间的自然变化已经涵盖了所有 editing type：
- 物体出现/消失 → add/remove
- 颜色变化 → attribute editing
- 物体移动 → object manipulation
- 光影变化 → lighting editing

而且是 **physics-correct 的**——真实世界的反射、阴影、interaction 天然就在 video 里，不需要人工模拟。

再用 Kosmos-2 拿 bounding box、SAM2 拿 mask、Depth Anything 拿 depth，一条龙把 video 拆成各种 task 的训练数据。

## 为什么能涌现新能力

训练时只见过 single-object insertion，推理时能做 multi-object insertion。能组合 task（insert + pose edit + color change 一起做）。能 zero-shot 迁移发型这种 local feature。

原因：full attention 让不同 task 的 pattern 互相 transfer；video data 让模型学到的是 world dynamics 而非 surface-level editing pattern。模型理解了 "物体放进水里会有涟漪和反射" 这种 physics，而不是记住了某个 editing 的 trick。

## 局限

Input images 超过 5 张就不稳定了，计算也炸。通常 3-4 张够用。要支持更多需要更高效架构和对应训练数据。

## 我的直觉

这个 paper 的本质贡献是两层 realization 叠加：

第一层：**formulation 层面**——image editing 就是 sparse video generation。这个 reformulation 让一个架构 cover 所有 task，不用给每个 task 设计专门的 module。

第二层：**data 层面**——video 是免费的 universal supervision。人工 curate editing data 又贵又有限，video 天然提供 physics-consistent 的 before/after pair，而且是无限量的。

两个 realization 叠起来，你得到一个不需要 task-specific design、不需要 task-specific data 的 universal model，而且因为 ground 在 real-world video 里，它学到的 editing 是 physics-aware 的。这是比 "收集更多 editing data" 或 "设计更复杂 module" 更高维度的解法。

---

# UniReal 深度解析

这篇paper来自 HKU 和 Adobe Research，核心 idea 非常 elegant：**把各种 image generation 和 editing tasks 统一 formulate 成 "discontinuous video generation"**。让我深入拆解技术细节。

## 核心 Insight

观察 video generation model（如 Sora）天然在处理 frame 之间的 **consistency-variation trade-off**，而这正是所有 image editing task 的本质需求。所以 UniReal 把任意数量的 input/output images 当作 video 的 "pseudo-frames"，用一个 video diffusion transformer 统一处理。

## 架构设计详解

### Diffusion Transformer Pipeline

```
Input Images → VAE Encoder → Latent Maps → Patchify → Visual Tokens
                                                        ↓
Text Prompt → T5 Encoder → Text Tokens ─────────────→ Concatenate → Transformer (Full Attention)
                                                        ↑
Noise Latent → + Position Embed + Timestep Embed ──────┘
```

关键点：
- **Full attention** 而非 causal attention，让所有 frames 之间能双向交互
- Visual tokens 上加 **index embeddings**（区分 IMG1, IMG2, RES1, RES2 等）
- Visual tokens 上加 **image prompt**（canvas/asset/control 的 category embedding）
- Position embeddings 对每个 image/noise token，timestep embeddings 只对 noise tokens

### Text-Image Association

特殊 token 设计：`IMG1, IMG2, ...` 指 input images，`RES1, RES2, ...` 指 output images。这些作为 T5 tokenizer 的 special tokens，同时学习对应的 **image index embeddings**，加到对应 image 的 visual tokens 上。

这样文本里可以写 "put the dog from IMG1 on the grassland in IMG2"，模型能建立 visual-text 的精确对应。

### Hierarchical Prompt

三层 prompt 结构：

| 层级 | 作用 | 示例 |
|------|------|------|
| **Base prompt** | 用户意图 | "put this dog on a grassland" |
| **Context prompt** | 任务/数据源属性 tags | "realistic style", "static scenario", "with reference object" |
| **Image prompt** | 输入 image 的角色 | canvas / asset / control |

Input images 分三类 pivot：
- **Canvas image**: 作为 background 的 editing target，layout fixed
- **Asset image**: 提供 reference objects，模型需 implicit segmentation + 模拟 size/position/pose 变化
- **Control image**: mask/edge/depth map，用于 layout/shape regularization

设计哲学：context prompt 的 keywords 可跨任务共享（如 "realistic style" 在 editing 和 customization 都用），**强制学习 common features**；text 天然 compositional，可组合 context prompts 实现 novel function。

## Flow Matching Loss

训练 loss 采用 flow matching（比 standard diffusion 更 general）：

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2$$

其中：
- $\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$：noise sample
- $\mathbf{x}_1$：clean data（target image latent）
- $t \sim \mathcal{U}(0, 1)$：time step
- $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$：linear interpolation
- $v_\theta(\cdot)$：网络预测的 velocity field

相比 DDPM 的 $\epsilon$-prediction，flow matching 的 velocity prediction 在 linear trajectory 下更 stable，且 gradient signal 更均匀。

## 数据构建 Pipeline

这是 paper 的另一核心贡献——从 video 提取 universal supervision：

```
Raw Video
    ↓
Caption Model → Video-level Captions
    ↓
Random 2 Frames + Caption → Video Frame2Frame (8M samples)
    ↓ (subset 200K)
GPT-4o mini → Precise Instructions
    ↓
Kosmos-2 (grounding caption) → Bounding Boxes
    ↓
SAM2 → Mask Tracklets
    ↓
├── Video Multi-object (5M) → Customization
├── Video Object Insertion (1M) → Object Insertion
├── Video ObjectAdd (1M) → Object Insertion with Prompt
├── Video SEG (5M) → Referring Segmentation
└── Video Control (3M) → Perception, Controllable Generation
```

关键 insight：video frames 间天然涵盖 **add / remove / attribute change / structural change** 等所有 editing type，且 physics-consistent（光影、反射、interaction 自然正确）。

## 训练方案

3-stage progressive training：

| Stage | 数据 | Resolution | 目标 |
|-------|------|------------|------|
| 1 | T2I + T2V data | 256×256 | 基础生成能力 |
| 2 | All datasets (Table 1) | 256 | Multi-task learning |
| 3 | - | 512 → 1024 | High-resolution fine-tune |

- 模型：5B parameters
- Learning rate：1e-5，每 stage 带 warm-up
- 支持 arbitrary aspect ratio（position embeddings on patches）

## 实验结果分析

### Instructive Editing (Table 2)

在 EMU Edit 和 MagicBrush test sets 上：

| Metric | UniReal | 最强竞争者 |
|--------|---------|-----------|
| CLIP_dir | **0.127** / **0.151** | EMU Edit 0.109 / 0.135 |
| CLIP_out | **0.285** / **0.308** | EMU Edit 0.231 / 0.261 |
| CLIP_im | 0.851 / **0.903** | ACE 0.895 / - |

CLIP_dir 衡量 text-image change agreement，CLIP_out 衡量 output 与 expected description 相似度——UniReal 在 instruction following 上显著领先。

### Customized Generation (Table 3, DreamBench)

| Model | CLIP-T↑ | CLIP-I↑ | DINO↑ |
|-------|---------|--------|-------|
| DreamBooth | 0.305 | 0.803 | 0.668 |
| SuTI | 0.304 | 0.819 | 0.741 |
| OmniGen | 0.320 | 0.810 | 0.693 |
| **UniReal** | **0.326** | 0.806 | 0.702 |

CLIP-T 最高（instruction following 最强），CLIP-I/DINO competitive——因为部分 test prompt 要求 attribute edit，text fidelity 和 image fidelity 有 trade-off。

### Ablation Study (Table 4)

| Config | MagicBrush CLIP_dir | DreamBench CLIP-T |
|--------|---------------------|---------------------|
| w/o Context Prompt | 0.144 | 0.315 |
| w/o Image Prompt | 0.136 | 0.295 |
| Only Expert Data | 0.139 | 0.309 |
| **UniReal-full** | **0.151** | **0.326** |

关键发现：
1. **Hierarchical prompt** 显著降低 multi-task ambiguity
2. **Multi-task > single-task**——不同 task 互补弥补数据不足
3. 纯 video data 训练的 model 已能 master 多种 editing（Fig. 9），甚至泛化到 multi-input 场景

## Emergent Abilities

训练后涌现的 zero-shot 能力：

1. **Multi-object insertion**：训练只有 single-object，推理支持 multi-object
2. **Task composition**：object insertion + pose editing + color modification 组合
3. **Local feature transfer**：如发型迁移，无需 mask
4. **Object manipulation**：move、resize with background preservation

## 与相关工作对比

| Method | 范式 | 局限 |
|--------|------|------|
| SEED-X / Emu2 | Autoregressive + separate diffusion | 生成质量弱 |
| Transfusion | Discrete text + continuous image | 理解为主 |
| OmniGen | Causal text + bidirectional image | 单向 attention 限制 |
| ACE | Conditioning unit + transformer | 需 task-specific design |
| **UniReal** | **Video framework + hierarchical prompt** | **Input>5 时不稳定** |

UniReal 的独特优势：**video data 提供的 universal supervision** + **full attention 的 bi-directional modeling** + **hierarchical prompt 的 ambiguity reduction**。

## Limitations

- Input images > 5 时稳定性下降、计算 intensive
- 通常 3-4 input images 足够
- 需要大量 input/output 的 task 需特殊训练数据 + 更高效架构

## 我的 Intuition Building

这篇 paper 的核心贡献在于两点 realization：

1. **Task unification via video formulation**：所有 image editing 本质都是 "given some frames, predict other frames with consistency + variation"，这正是 video generation 在做的事。这个 reformulation 让一个 model 架构 cover 所有 task。

2. **Video as universal supervision**：不需要为每个 task 精心 curate data，video frames 间的自然变化已经涵盖了所有 editing type，且物理一致性（光影、反射、interaction）是 free annotation。

Hierarchical prompt 是工程上很聪明的设计——context prompt 的 keywords 跨任务共享强制 common feature learning，这比 per-task embedding 更 generalizable。

Emergent ability 的来源：full attention 让不同 task 的 patterns 能互相 transfer，video data 的 physics grounding 让 model 学到 world dynamics 而非 surface-level editing patterns。

参考链接：
- [Project Page](https://uni-real.github.io/)
- [Sora technical report](https://openai.com/index/sora)
- [Flow Matching paper](https://arxiv.org/abs/2210.02747)
- [SAM2](https://arxiv.org/abs/2408.00714)
- [Kosmos-2](https://arxiv.org/abs/2306.14824)
- [OmniGen](https://arxiv.org/abs/2409.11340)
- [ACE](https://arxiv.org/abs/2410.00086)
