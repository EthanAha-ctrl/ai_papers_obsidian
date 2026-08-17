---
source_pdf: Molmo2.pdf
paper_sha256: d9d0448e5313a12c7a690aed6ed282b9aeec69e1e6e6e27a34c8c54b126b71b9
processed_at: '2026-08-05T20:00:12-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Molmo2 用人话讲

## 这论文到底在干嘛

一句话：**给 VLM 加上"指"的能力，而且是视频里指**。

你问 GPT-5 "视频里机器人抓了几次红方块"，它会给你编个数。你让它"指出来在哪"，它基本懵了——Gemini 3 Pro 指对了 20 次/100，GPT-5 只指对 4 次。这帮闭源模型连这么基本的事都干不好。

Molmo2 就是来填这个坑的。而且**全开源**——数据、权重、代码全给你，不拿闭源模型蒸馏。

---

## 为什么"指"这么难

Image pointing 现在烂大街了，Molmo 原版、Qwen-VL 都能指。但视频指就不一样了：

1. **时间维度**：你问"他什么时候跳起来"，得指到具体那一帧
2. **Re-identification**：同一个人走出画面又回来，得知道还是同一个人
3. **数量爆炸**：128 帧的视频，每帧指几个点，token 数蹭蹭涨

所以不是模型不够大，是**没人有这种数据**。闭源模型也没标过这种数据，所以它们也不会。

---

## 数据怎么搞的——这是论文真正的贡献

### 核心痛点：打字太慢

你想让人标视频 caption，让他打字？一个视频标 900 词，打字得 15 分钟。让他**说**？2 分钟搞定。

Molmo2 的 caption pipeline：
1. 视频切成 10-30 秒小段
2. 标注员**口述**描述 → Whisper 转录 → LLM 顺一下句子
3. 用 Molmo 原版给每帧生成细节 caption 补充
4. LLM 合并成一篇长文

结果：平均 **924 词/视频**，比之前最 dense 的 LLaVA-Video (547) 还多一倍。

### Pointing 数据：8 种问法

不是只问"指一下狗"，他们设计了 8 类 query：
- Objects / Animals / Actions（基本的）
- Referring expressions（"穿红衣服那个男的"）
- Indirect（"她手里拿的那个"）
- Spatial（"左上角的东西"）
- Comparative（"比较高的那栋楼"）
- Visual artifacts（生成视频的瑕疵）← 这个很前瞻

650k 条 query，平均每个视频 6 个点。

### Tracking 数据：聪明地省事

逐帧标 point 不现实。他们的做法：
- 拿现成的 VOS 数据集（有 segmentation mask）
- Bbox 数据集用 SAM-2 转 mask
- 从 mask 里采样一个点（靠近中心但不 flickering）
- 让人写复杂的 text query 描述**多个**物体

---

## 表示格式——小而美的设计

不用 JSON，用 HTML-like 紧凑格式：

```
<points coords="1 1 555 169;2 3 649 154 4 709 162">
描述文字
</points>
```

四个字段：帧号/时间戳、物体ID、x、y。坐标归一化到 0-1000。

**物体 ID 是精髓**：
- 做counting？最后一个 ID 就是总数
- 做 tracking？同一 ID 跨帧就是同一物体

一个格式搞定三个任务。比专门设计 track head 优雅多了。

---

## 模型架构——没什么花活

标准三件套：SigLIP 2 ViT + Connector + Qwen3/OLMo LLM。

两个值得注意的小决定：

**1. Video 用 3×3 pooling，image 用 2×2**
视频 token 太多，压狠一点。Image 要细节，少压点。

**2. Vision token 之间 bi-directional attention**
跨帧的视觉 token 可以互相 attend，不受 causal mask 限制。免费涨点。

---

## 训练三个阶段

**Stage 1 Pre-train**：图片 caption + 图片 pointing + NLP。这里就把 pointing 格式教会了，SFT 阶段不用再学格式。

**Stage 2 SFT**：图片+视频混合。16384 序列长度，30k steps。

**Stage 3 Long-context**：36864 长度，384 帧。只训 2k 步但很贵——用 context parallelism 把一个 example 分到 8 张 GPU。

---

## 三个工程技巧让训练快 15 倍

### 1. Packing：动态规划塞例子

问题：有的例子 200 token，有的 16000 token，padding 浪费太多。

解法：维护 48 个 example 的池子，动态规划求解怎么塞满 16384 长度。一个 packed 序列平均塞 3.8 个例子。**15× 效率提升**。

### 2. Message Tree：一个视频多个标注共享视觉编码

一个视频有 caption + 3 个 QA + pointing，怎么塞进一个序列？

线性化成一棵树，视觉 token 是根，每个标注是一个分支。用 attention mask 让分支互不可见，但都能看到视觉 token。

这样视觉编码只算一次，多个标注共享。

### 3. Token Weighting：长输出别主导 loss

Caption 4000 token，多选题 1 token。不加权的话 caption 主导整个 loss。

公式：普通任务 $4/\sqrt{n}$，caption 0.1，pointing 0.2。$n$ 是 answer token 数。

---

## "先指再数"——Counting 的关键 insight

让模型直接输出数字，高数量完全废。Qwen3-VL 在数 25-60 个物体时 **0% 准确率**。

Molmo2 让模型先 point 所有物体，再数 point 数。把 counting 变成 grounding 的子问题。

结果：25-60 范围 Molmo2-8B 有 7% 准确率，Qwen3-VL 是 0%。虽然也不高，但思路对了。

---

## 实际效果怎么样

**Video Pointing**：Molmo2-4B F1 39.9，Gemini 3 Pro 20.0，GPT-5 4.1，Qwen3-VL-8B 1.5。碾压级别。

**Video Tracking**：在 MeViS 上 J&F 62.3，专门做 tracking 的 Sa2VA-8B 是 46.9。通用模型打专用模型。

**Video Captioning**：F1 43.2，超过 GPT-5 (50.1 是另一个指标，apple-to-apple 比较是 35.8)。在 open 模型里最好。

**Short video QA**：open-weight 里最好，但 long video 落后 Qwen3-VL。作者承认因为缺 10 分钟以上的开源训练数据。

**Human preference Elo**：1057，超过 GPT-5 (1031) 和 Claude Sonnet 4.5 (1008)。

---

## 坑在哪

1. **Vision encoder 还是闭源的**——SigLIP 2 训练数据没全开
2. **生成数据用了闭源 LLM**（Claude 4.5）——但他们坚持不用闭源 VLM，避免循环依赖
3. **长视频 grounding 会退化**——有时候模型在同一帧吐一长串点，或者每帧都吐同一个点
4. **长 caption 会重复**——greedy decoding 生成几千 token 后开始车轱辘话
5. **Point 在物体上位置会漂**——tracking 时点可能在物体不同部位跳

---

## 给你的直觉

**1. Grounding 是 VLM 的下一个 frontier。** 现在 image 指认、video 指认都解决了大半，下一步是 3D 空间指认（机器人场景）。

**2. "Point then Count" 比 "直接 Count" 强。** 把数量问题转化为空间定位问题，cognitive offloading。

**3. Object ID 是最小 tracking 表示。** 不需要单独的 track head，纯文本里带 ID 就够了。简洁就是力量。

**4. Speech-to-text 是 dense caption 唯一可行路径。** 打字速度是人类标注的瓶颈。

**5. Token weighting 是 multi-task 训练必备。** 长输出 task 会悄悄主导你的 loss。

**6. Fully open > open weights。** 没数据的开源是"半开源"，社区没法真正迭代。Molmo2 把数据也开了，这才是真开源。

**7. Packing + Message Tree 是工程杠杆。** 15× speedup 意味着小 lab 也能训大模型。这是让开源能跟闭源掰腕子的基础设施。

---

## 一句话总结

Molmo2 证明了一件事：**不需要蒸馏闭源模型，靠开源数据 + 巧妙的标注 pipeline + 工程优化，能在 video grounding 这个 niche 上打赢所有闭源模型**。数据和工程比架构重要。

---

# Molmo2：开源 Video-Language Model with Grounding 深度解析

## 一、动机与定位

当前 video-language model 领域存在三个核心痛点，Molmo2 正是针对这些问题设计：

1. **Proprietary 模型垄断**：GPT-5、Gemini 3 Pro、Claude Sonnet 4.5 等最强 VLM 全部闭源，weights/data/recipe 都不公开。
2. **开源模型的蒸馏依赖**：LLaVA-Video、ShareGPT4Video 等开源工作依赖 GPT-4o 等闭源 VLM 生成训练数据，导致开源社区无法独立迭代——形成了"先有鸡还是先有蛋"的循环依赖。
3. **Grounding 能力缺失**：即便闭源模型也只能做 high-level video understanding，无法回答 "机器人抓取红色方块几次" 这类需要 spatio-temporal pointing 的问题。Gemini 3 Pro 在 video pointing 上仅 20.0 F1，GPT-5 只有 4.1 F1。

Molmo2 的定位：**fully open**（weights + data + code，no distillation from closed VLMs），同时把 image pointing 范式扩展到 video 的 spatio-temporal domain 与 continuous tracking。

参考：[Molmo2 GitHub](https://github.com/allenai/molmo2) | [Ai2 Playground](https://playground.allenai.org) | [Molmo 原版论文 (CVPR 2025)](https://arxiv.org/abs/2409.17146)

---

## 二、数据集：9 个新数据集的设计哲学

Molmo2 的核心贡献是数据，而非架构创新。下表整理了 9 个新数据集：

| 数据集 | 类型 | 规模 | 关键特性 |
|---|---|---|---|
| Molmo2-Cap | Human | 104k video + 431k clip captions | **平均 924 words/video**，目前最 dense |
| Molmo2-AskModelAnything | Human | 140k QA pairs | 31 cluster 均匀采样，禁止 counting 题 |
| Molmo2-CapQA | Synthetic | 1M QA (200k video × 5) | 基于 Molmo2-Cap 训练的 captioner，scene-level |
| Molmo2-SubtitleQA | Synthetic | 300k QA (100k video × 3) | Whisper-1 转录 + 视觉+字幕联合推理 |
| Molmo2-VideoPoint | Human | 650k queries / 280k videos | **8 categories**，avg 6 points/video，2 fps |
| Molmo2-VideoTrack | Human | 3.6k clips / 15k queries | avg 2.28 objects/query，多物体追踪 |
| AcademicVideoPoint | Curated | 49k pointing/counting | 6 个数据集转换 |
| AcademicVideoTrack | Curated | 11 个 bbox 数据集 | SAM-2 生成 mask，再采样 point |
| Molmo2-MultiImageQA / -MultiImagePoint / -SynMultiImageQA | Human + Synthetic | 72k + 470k + 188k | 2-5 image sets，semantically related |

### 2.1 Molmo2-Cap：dense video caption 的关键 pipeline

为什么之前的 video caption 都很短？因为 typing 速度限制了 annotator 的产出。Molmo2 借鉴 PixMo-Cap 的"speech-to-text"思路：

**Pipeline**：
1. 视频被 adaptive 算法切成 10-30s 不等长 clips（基于 information density）
2. Annotator **口述** clip description → Whisper-1 转录 → LLM 重写为连贯文本
3. 用 **Molmo (原版)** 生成 frame-level caption 补充低层细节
4. LLM 合并 clip caption + frame caption → 最终长 caption

对比数据（平均 words/video）：
- Video Localized Narratives: 75
- RCap / RDCap: 89 / 100
- ShareGPT4Video: 280
- LLaVA-Video-178K: 547
- **Molmo2-Cap: 924**

### 2.2 Molmo2-VideoPoint：8 类 query 覆盖

| Category | 示例 |
|---|---|
| Objects | "Point to all cars" |
| Animals | "Point to birds" |
| Actions/Events | "Point to moments when someone jumps" |
| Referring expressions | "Point to the man in red shirt" |
| Indirect references | "Point to what she is holding" |
| Spatial references | "Point to objects in the upper-left" |
| Comparative references | "Point to the taller building" |
| Visual artifacts (生成视频专属) | "Point to visual defects" |

后两类 (comparative + artifacts) 是为生成视频时代设计的，这是一个前瞻性的设计。

### 2.3 Grounding 表示格式

Molmo2 用一种 **HTML-like 紧凑文本格式** 而非 JSON，大幅减少 token 数：

```html
<points coords="1 1 555 169;2 3 649 154 4 709 162;5 5 758 175 6 808 183 7 852 187">
Inline text describing what's pointed at
</points>
```

格式拆解：
- 第一列：image index (从 1 开始) 或 frame timestamp (秒，1 位小数)
- 第二列：object ID（唯一，用于 counting 和 tracking）
- 第三、四列：x, y 坐标，**归一化到 [0, 1000]**
- 分号分隔不同 frame/image，空格分隔同一 frame 的多个 point

Tracking 格式：
```html
<tracks coords="0.0 1 635 522;0.5 1 606 490 2 511 124;1.0 2 515 164;1.5 2 520 168">
Inline text
</tracks>
```

**Object ID 是关键创新**：它同时支持 counting（最大 ID 即总数）和 tracking（同一 ID 跨 frame 表示同一物体）。

---

## 三、模型架构

### 3.1 总体设计

```
┌─────────────┐     ┌──────────────┐     ┌──────┐
│  ViT (SigLIP│ ──► │  Connector   │ ──► │ LLM  │
│  2 So400m/14│     │  (attention  │     │(Qwen3│
│  384px)     │     │   pool+MLP)  │     │/OLMo)│
└─────────────┘     └──────────────┘     └──────┘
```

### 3.2 Vision Encoder 处理

**Cropping 策略**：
- Image：单 crop 缩放 + 最多 K 个 overlapping crops（train K=8, inference K=24）
- Video：固定 S=2 fps 单 crop，max F=128 frames（long-context 时 F=384）
- 若 video 长度 > F/S，uniform sample F frames，**但最后一帧总是包含**（因为播放器结束后停在那）

**Connector**：
- 取 ViT **倒数第 3 层 + 倒数第 9 层** features（多层特征，类似 FPN 思想）
- Image: **2×2 patch window** → 1 vector，用 multi-head attention pooling (mean patch 作 query)
- Video: **3×3 patch window** → 1 vector（更激进压缩，因为 video token 多）
- 共享 connector 参数（image 和 video 共用）
- 最后 MLP 投影到 LLM embedding 空间

### 3.3 LLM 输入格式

```
[BOS] [VIDEO_START] [frame1_tokens] t=0.0 [frame2_tokens] t=0.5 ... [VIDEO_END]
[subtitle text if available] [user prompt] [assistant response]
```

关键设计：
- **Vision tokens forward-attend to each other**：跨 frame 的 vision token 可以互相 bi-directional attend（不受 causal mask 限制）
- Multi-crop image 加 **column tokens**（标记 aspect ratio，single crop 不加因为是正方形）
- Video frame 间插入 text timestamp（"t=0.0", "t=0.5"...）

### 3.4 三种 model size

| Model | LLM | Params | Dim | Layers | Image/Video Pool |
|---|---|---|---|---|---|
| Molmo2-4B | Qwen3-4B | 4.0B | 2560 | 36 | 2×2 / 3×3 |
| Molmo2-8B | Qwen3-8B | 8.2B | 4096 | 36 | 2×2 / 3×3 |
| Molmo2-O-7B | OLMo3-7B | 7.3B | 4096 | 32 | 2×2 / 3×3 |

ViT 全部用 SigLIP 2 So400m/14 (380M params, 384×384 input, patch 14)，**Connector 80-88M params**。

---

## 四、训练 Recipe：三阶段

### Stage 1: Pre-training (image only)

- **数据混合**：60% captioning (PixMo-Cap) + 30% image pointing (PixMo-Points, PixMo-Count, CoSyn-Point) + 10% NLP (Tulu filtered)
- **32k steps**, batch 128, max len 2560
- 关键：**pointing 放在 pre-training** 让模型先学会输出格式，SFT 阶段更专注语义
- Length conditioning + response-only dropout 0.1（继承 Molmo）

### Stage 2: SFT (joint image+video)

数据混合比例（Table 1）：

| Category | Sampling Rate | Examples |
|---|---|---|
| Image QA | 22.7% | 2.4M |
| Video QA | 18.2% | 2.4M |
| Captions/Long QA | 13.6% | 1.2M |
| Video Pointing | 13.6% | 0.37M |
| Video Tracking | 13.6% | 0.80M |
| NLP | 9.1% | 0.99M |
| Image Pointing | 9.1% | 1.1M |

- **30k steps**, batch 128, max len **16,384**
- Dataset 内按 $\sqrt{n}$ 比例采样（避免大 dataset 主导）
- 30% 的 pointing 图片用 24 crops（确保高分辨率泛化）

### Stage 3: Long-context SFT

- max len **36,864**, F=384 frames
- **2k steps**, batch 128
- 使用 **Ulysses attention** 做 context parallelism (8 GPUs/example)
- Vision encoder 和 attentional pooling 也分布式跨 GPU
- 显著提升 long video QA（67.4 vs 64.4 without），但 caption F1 略降（39.9 → 39.9 持平）

---

## 五、训练关键技术：Packing + Message Tree + Token Weighting

### 5.1 Packing：动态规划求解 bin packing

问题：example 长度从几百到 16k+ 不等，padding 浪费严重。

算法：
1. 维护一个 **M=48 examples** 的 in-memory pool
2. Pool 不满时从训练集采样新 example 加入
3. Pool 满时用 **dynamic programming** 求解：最大化 $\sum_i (T_i + I_i \cdot w_i)$，约束 $\sum T_i \leq 16384$, $\sum I_i \leq 128$
   - $T_i$: example $i$ 的 text token 数
   - $I_i$: example $i$ 的 crop 数
   - $w_i = 30$: 权重超参，平衡 text 和 crop 的"占用"
4. 选中的 examples 拼成一个 packed sequence，从 pool 移除

**关键细节**：
- 在 quantized 版本上求解（round 到 32 的倍数）加速
- $w_i$ 太小会导致 pool 填满 128-crop 的大 example 无法继续 pack
- 集成到 PyTorch DataLoader，每个 worker 独立运行
- 实测 **15× 训练效率提升**（一个 16384 序列平均塞 3.8 个 examples）

### 5.2 Message Tree：多 annotation 的 attention mask

一个 video 可能有多个 annotation（caption, QA1, QA2, pointing...）。如何 pack 进一个序列？

**Message Tree 结构**：
```
[Visual input]  ──── branch 1: Caption
                 ├─── branch 2: QA pair 1
                 ├─── branch 3: QA pair 2
                 └─── branch 4: Pointing query
```

线性化为单一序列，但用 **custom attention mask**：
- 每个 branch 可以 attend 到 **visual input**（forward）
- branch 之间 **互不可见**（mask 掉 cross-branch attention）
- 同一 packed sequence 内不同 examples 也互不可见

Figure 3 展示：lower-left empty block 是不同 example 之间的 mask，upper empty block 是同 example 内不同 branch 的 mask。Frame tokens (dark pink) 用 forward attention（bi-directional on vision tokens）。

### 5.3 Token Weighting：长输出 example 的平衡

问题：video caption 可能 4000+ tokens，multiple choice 只有 1 个 token。如果不加权，长输出 example 会主导 loss（即使采样率低）。

公式：
$$
\text{weight}(e) = \begin{cases}
0.1 & \text{if } e \in \text{video captions} \\
0.2 & \text{if } e \in \text{pointing} \\
\frac{4}{\sqrt{n}} & \text{otherwise}
\end{cases}
$$

其中 $n$ 是 answer token 数。

**Intuition**：
- $4/\sqrt{n}$ 是 sub-linear scaling，对长 answer 衰减但对短 answer 不放大太多
- Caption 和 pointing 单独设低权重，因为它们输出特别密集
- Ablation (Table 8b)：no token weighting → QA avg 64.0 (vs 64.8), Cap F1 40.0 (vs 39.5)
  - 注意：token weighting **提升 QA 但略降 caption**，因为 caption 需要长输出练习

---

## 六、评测结果：跨 benchmark 全景

### 6.1 Video benchmarks (Table 2)

| Model | Short Video Avg | Long Video Avg | Cap F1 | Count Acc | Elo |
|---|---|---|---|---|---|
| GPT-5 | 73.1 | 76.3/70.6 | 50.1/35.8 | - | 1031 |
| Gemini 3 Pro | 71.0 | 78.8/70.0 | 36.0/37.1 | - | 1082 |
| Gemini 2.5 Pro | 71.1 | 80.4/71.2 | 42.1/35.8 | - | 1096 |
| Qwen3-VL-8B | 65.3 | 63.5/59.5 | 26.7/29.6 | 29.6 | 1054 |
| **Molmo2-8B** | **69.9** | **64.1/63.1** | **43.2/35.5** | **35.5** | **1057** |
| **Molmo2-4B** | 69.3 | 64.5/62.8 | 39.9/34.3 | 34.3 | 1041 |

**观察**：
- Molmo2 在 **open-weight 中 short video SOTA**
- Long video 落后 Qwen3-VL-8B（63.1 vs 59.5）——作者归因于缺少 10+ min 开源长视频训练数据
- Caption F1 (43.2) **超越所有 API 模型**包括 GPT-5 (50.1) 和 Gemini 3 Pro (37.1)
- Human preference Elo 1057，**超过 GPT-5 (1031) 和 Claude Sonnet 4.5 (1008)**

### 6.2 Grounding：Molmo2 的杀手锏

**Video Pointing (Table 3)**：

| Model | F1 | Recall | Precision |
|---|---|---|---|
| GPT-5 | 4.1 | 4.4 | 4.2 |
| Gemini 3 Pro | 20.0 | 27.4 | 19.8 |
| Gemini 2.5 Pro | 13.0 | 14.5 | 13.6 |
| Qwen3-VL-8B | 1.5 | 1.5 | 1.5 |
| **Molmo2-8B** | **38.4** | **39.3** | **38.7** |
| **Molmo2-4B** | **39.9** | **42.7** | **39.4** |

Molmo2-4B 的 video pointing F1 **是 Gemini 3 Pro 的 2 倍**，是 Qwen3-VL-8B 的 **26 倍**。这说明闭源 API 模型基本不具备 spatio-temporal pointing 能力。

**Video Tracking (Table 4)**：

| Model | MeViS J&F | Ref-YT-VOS J&F | Ref-DAVIS J&F | ReasonVOS J&F |
|---|---|---|---|---|
| Gemini 3 Pro | 42.5 | 55.0 | 66.6 | 52.6 |
| Sa2VA-8B (specialized) | 46.9 | 70.7 | 75.2 | 55.5 |
| VideoMolmo-7B | 53.9 | 67.3 | 72.5 | 51.1 |
| **Molmo2-8B** | **62.3** | **78.7** | **81.3** | **65.8** |

Molmo2 在所有 tracking benchmark 上**全面超越 specialized 模型**，在 ReasonVOS (需要复杂推理) 上优势尤为明显。

### 6.3 Image benchmarks (Table 6)

Molmo2-8B 在 11 个 image benchmark 平均 **76.3**，其中：
- VQA v2.0: **93.2** (SOTA among open)
- RealWorldQA: 80.1
- Counting (PixMo-Count): 88.5
- DocVQA: 86.0 (略落后 Qwen3-VL-8B 89.6)

Point-Bench (Table 7)：Molmo2-8B **72.7 avg**，超过所有模型包括 dedicated pointing model Poivre (67.5) 和 Gemini Robotics ER-1.5 (67.1)。

### 6.4 Counting 按数量分桶 (Table 17)

| Object Count Range | Qwen3-VL-8B | Molmo2-8B | Gemini 3 Pro |
|---|---|---|---|
| 0-5 | 63.8 | 64.4 | 69.5 |
| 5-10 | 30.6 | 32.9 | 34.1 |
| 10-15 | 15.0 | 26.3 | 24.1 |
| 15-20 | 6.8 | 25.7 | 16.2 |
| 20-25 | 6.3 | 7.9 | 14.3 |
| 25-60 | **0.0** | **7.0** | 12.5 |

**Qwen3-VL 在 25-60 count 范围 0% 准确率**——它只能输出一个数字而非 pointing。Molmo2 通过 "point then count" 策略 (Table 9a) 在 high-count 场景表现接近 Gemini 3 Pro。

---

## 七、关键 Ablation 详解

### 7.1 Bi-directional Vision Attention (Table 8b)

| Setting | QA avg | Cap F1 |
|---|---|---|
| Video-Only (default) | 64.8 | 39.5 |
| No bidir | 64.4 | 38.5 |

Bi-directional attention 让 vision tokens 跨 frame 互相 attend，提升 ~0.4-1.0 point。

### 7.2 Token Weighting (Table 8b)

| Setting | QA avg | Cap F1 |
|---|---|---|
| Default | 64.8 | 39.5 |
| No token weighting | 64.0 | 40.0 |

QA 提升 0.8，Caption 降 0.5——trade-off 但 QA 更重要。

### 7.3 Time Tokens (Table 8b)

| Setting | QA avg | Cap F1 |
|---|---|---|
| Default | 64.8 | 39.5 |
| No time tokens | 64.5 | 37.4 |

移除 frame timestamp 对 caption **打击最大**（-2.1 F1），因为 caption 需要时序叙事。

### 7.4 Pool Size (Table 8b)

| Pool | QA avg | Cap F1 |
|---|---|---|
| 3×3 (default) | 64.8 | 39.5 |
| 4×4 | 64.3 | **37.0** |

4×4 pool 减少 token 数，QA 几乎不变但 **caption 降 2.5**——caption 需要细粒度视觉信息。

### 7.5 Pointing Pre-training (Table 18)

| Setting | Video QA | Image Pointing |
|---|---|---|
| With pointing pretrain | 66.8 | 73.0 |
| No pointing pretrain | 65.9 | 71.8 |

Pointing pre-training 不仅提升 pointing，**还提升 video QA**——因为模型不需要在 SFT 阶段学习输出格式。

### 7.6 SlowFast Test-time Scaling (Table 20)

为支持 long video，论文探索 SlowFast encoding：
- **Slow pathway**: 3×3 pooling（高分辨率）
- **Fast pathway**: 9×9 pooling（低分辨率，覆盖更多 frame）
- Periodic 或 query-based 选 slow frame

| Strategy | Long QA avg | Vision Tokens |
|---|---|---|
| 128 frames (default) | 64.6 | 10.6k |
| 224 frames | 65.6 | 18.6k |
| 128 + SF-query | **65.7** | 10.7k |

**SF-query** 用 SigLIP 2 计算 query-frame 相似度选 slow frame，**用一半 token 达到 224 frame 效果**。

---

## 八、数据标注 Pipeline 细节

### 8.1 Molmo2-Cap 标注流程

1. 视频切分：adaptive 算法基于 information density，10-30s 不等长 clip
2. Annotator 看无声 clip → 口述 → Whisper-1 转录 → 编辑
3. 完成所有 clip 后，描述完整 video
4. Molmo (原版) 生成 frame-level caption
5. LLM 合并 clip + frame caption

### 8.2 Molmo2-VideoPoint 标注流程

1. 2 fps 采样 frame
2. Annotator 看完整无声 video
3. 每个问题：annotator 截图相关 frame → 在截图上 click point → 记录 timestamp + (x, y)
4. 标记 Unanswerable 或 Unsure

### 8.3 Molmo2-VideoTrack 标注流程

挑战：逐 frame 标注 point 不可行，CoTracker/SAM-2 自动生成 track 不稳定。

解决方案：
1. 用现有 VOS dataset 的 segmentation mask 作为 base
2. Bbox dataset → SAM-2 生成 mask tracklet（IoU < 0.5 过滤）
3. 从 mask 采样 point（alpha-weighted score 平衡 centroid 和 boundary）
4. Annotator 看视频 + object tracks → 写复杂 text query 描述**多个**物体
5. 验证轮次过滤低质量 query（保留 ~70%）

---

## 九、Limitations（论文坦白）

1. **Vision encoder 仍闭源**：使用 SigLIP 2（虽然 OpenCLIP 训练，但 data 不完全开放）
2. **数据生成用闭源 LLM**：用 Claude Sonnet 4.5 等，但避免闭源 VLM
3. **Video grounding 不稳定**：高频物体或长视频上会出现 degenerate output（一长串 point 在一帧 或 同一 point 每帧重复）
4. **Caption 重复**：greedy decoding 生成超长 caption 时会重复（Qwen3 已知问题）
5. **Long video grounding 受限**：3 min+ 视频支持有限，因为 grounding 训练数据上限
6. **Point tracking 位置漂移**：track 中 point 在物体上位置会变（因生成 pipeline 不保证一致）

---

## 十、为什么 Molmo2 重要：直觉

### 10.1 Grounding 作为 VLM 的"下一个 frontier"

Image pointing 已经成为标配（Molmo, Qwen2.5-VL, PaliGemma 2 都支持）。但 video grounding 仍是空白——因为它需要：
- **Spatial + temporal 联合定位**
- **Re-identification**（跨 frame 判断是否同一物体）
- **Long-range dependency**（物体消失再出现）

Molmo2 把 image pointing 的简单 2D 范式扩展到 3D (x, y, t)，并用 object ID 串起 tracking——这是一个**最小的、可扩展的 grounding 表示**。

### 10.2 完全开源的意义

之前的"开源" VLM 有三种：
1. **Open weights only**：Qwen-VL, InternVL（weights 开放但 data/recipe 闭源）
2. **Open weights + distilled data**：LLaVA-Video（用 GPT-4o 蒸馏，循环依赖）
3. **Fully open**：Molmo, PLM（data + weights + code 全开）

Molmo2 是第三类的代表。这意味着社区可以：
- 复现结果
- 在数据上做 ablation
- 扩展到新 domain（医疗视频、工业检测等）
- 不被闭源模型"卡脖子"

### 10.3 "Point then Count" 范式

传统 counting 让模型直接输出数字，high-count 完全失效（Qwen3-VL 在 25-60 范围 0%）。Molmo2 让模型先 point 所有物体再数 point 数——把 counting 转化为 grounding 子问题。这是一个**cognitive offloading**：把 counting 的认知负担转移到视觉 grounding。

### 10.4 Packing + Message Tree 的工程价值

这两个技术让训练 throughput 提升 15×，意味着**同样的 GPU 预算可以训更多数据/更大模型**。Message tree 让一个 video 的多个 annotation 共享 visual encoding——这对 video training 尤其重要，因为 video encoding 是 bottleneck。

---

## 十一、后续工作联想与潜在方向

1. **Open vision encoder**：OpenCLIP、DINOv3、EVA-02 等开源 ViT 的 grounding 友好版本——Molmo2 明确呼吁社区做这个
2. **3D grounding**：从 (x, y, t) 扩展到 6-DoF pose 或 3D point cloud
3. **Robotics 集成**：MolmoAct (作者后续工作 [73]) 已经把 Molmo2 用于 action reasoning
4. **Long video grounding**：解决 fps 与 annotation 对齐问题（论文 limitation 提到的）
5. **Audio grounding**：扩展到 audio-event pointing
6. **Multi-modal grounding**：point + bbox + mask + track 统一表示
7. **Test-time scaling**：SF-query 是一个 promising 方向，可扩展到 frame selection for captioning

参考链接：
- [Molmo2 论文全文](https://arxiv.org/abs/2509.04560)
- [Molmo2 GitHub](https://github.com/allenai/molmo2)
- [Molmo 原版 (CVPR 2025)](https://arxiv.org/abs/2409.17146)
- [PixMo 数据集](https://huggingface.co/datasets/allenai/pixmo)
- [VideoMolmo (前序工作)](https://arxiv.org/abs/2506.05336)
- [SAM 2](https://arxiv.org/abs/2408.00714)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [Qwen3-VL](https://arxiv.org/abs/2505.09388)
- [Tulu 3](https://arxiv.org/abs/2411.15124)
- [HOTA metric](https://arxiv.org/abs/2009.07736)

---

## 十二、对 Karpathy 直觉的 build：核心 takeaways

1. **数据是瓶颈，不是架构**：Molmo2 架构是标准 ViT + connector + LLM，创新全在数据收集 pipeline 和训练 recipe。
2. **Speech-to-text 是 dense caption 的关键**：突破 typing 速度限制是 dense caption 的唯一可行路径。
3. **Pointing 是 grounding 的"原子操作"**：counting、tracking、referring 都可以表达为 pointing 的组合。
4. **Object ID 是 tracking 的最小表示**：不需要 separate track head，纯文本格式即可。
5. **Token weighting 是 multi-task 训练的必备**：长输出 task 会 dominate loss，必须显式降权。
6. **Packing + Message tree 是 VLM 训练的工程突破**：15× speedup 让小 lab 也能训大模型。
7. **Bi-directional vision attention 免费提升**：vision token 不需要 causal mask，去掉它即可。
8. **Test-time scaling 在 video 上有效但有限**：SF-query 是 token-efficient 的 promising 方向。
9. **Fully open > open weights**：没有 data 的 open weights 是"半开源"，社区无法真正迭代。
10. **Grounding 是 VLM 的下一个 frontier**：从 image pointing 到 video tracking，grounding 的粒度决定 VLM 的实用边界。
