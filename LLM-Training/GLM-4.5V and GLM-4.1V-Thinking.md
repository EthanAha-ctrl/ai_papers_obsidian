---
source_pdf: GLM-4.5V and GLM-4.1V-Thinking.pdf
paper_sha256: afc06e894942ccbaba446c4f72e7f1d12794149d343c79dc67e2b78281a80732
processed_at: '2026-08-04T21:45:01-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GLM-4.5V 用人话讲

## 一句话总结

先造一个聪明的"底子"（pre-trained base），然后教它一套统一的"思考格式"（SFT），最后用大规模多领域强化学习（RLCS）把它的潜力逼出来。整个过程就是一个 reasoning-centric 的 pipeline，RL 是核心引擎。

---

## 1. 为什么要做这件事

过去一两年大家发现：纯文本 LLM 用 long CoT + RL（比如 DeepSeek-R1、OpenAI o1）能大幅提升 reasoning。但 VLM 这边大多还停留在"看图说话"阶段，少数做 reasoning 的又只盯单一 domain（比如只做 math 或只做 GUI agent）。

GLM 团队想搞一个 **通用 multimodal reasoning 模型**，能在 STEM、video、OCR、grounding、GUI agent、coding、长文档这些任务上全面提升，而且开源。同时他们还发现了一个很反直觉的事情：**多 domain 一起做 RL，互相之间不打架，反而互相促进**。

---

## 2. 模型架构：三件套 + 两个聪明设计

### 三件套
- **Vision Encoder**：AIMv2-Huge（一个很强的 ViT）
- **Projector**：MLP，把视觉特征对齐到 text token 空间
- **LLM Decoder**：GLM-4-9B（dense，给 4.1V 用）或 GLM-4.5-Air（106B total / 12B activated 的 MoE，给 4.5V 用）

### 聪明设计 1：Position Embedding 的 bicubic interpolation

问题：ViT pre-train 时是固定分辨率（比如 14×14 patch grid），但推理时要支持任意分辨率和 aspect ratio（甚至 200:1 的超宽图、4K 高清图）。

传统做法要么 resize 图像（丢信息），要么完全用 RoPE 替代 absolute PE（可能破坏 pre-trained 能力）。GLM 的做法是**两个都要**：

1. 保留 pre-trained 的 absolute position embedding table $P_{\mathrm{orig}}$
2. 对于新分辨率的任意 patch $(w, h)$，先把它 normalize 到 $[-1, 1]$：

$$\mathbf{g}_{\mathrm{norm}} = 2 \cdot \left( \frac{w + 0.5}{W_p}, \frac{h + 0.5}{H_p} \right) - 1$$

- $w, h$：patch 在 grid 里的列号、行号
- $W_p, H_p$：grid 总宽、总高
- $+0.5$：取 patch 中心点
- $2 \cdot (\cdot) - 1$：把 $[0,1]$ 拉到 $[-1,1]$

3. 然后用 bicubic 插值从 $P_{\mathrm{orig}}$ 里采样出这个 patch 的 position embedding：

$$P_{\mathrm{adapted}}(\mathbf{g}) = \mathcal{T}_{\mathrm{bicubic}}(P_{\mathrm{orig}}, \mathbf{g}_{\mathrm{norm}})$$

**Intuition**：相当于把原来学到的固定 grid 的 position embedding 当成一张"小图"，用 bicubic 插值放大/变形到任意尺寸。这样既保留了 pre-trained 的 spatial prior，又支持 native resolution。

4. 同时在 ViT 的 self-attention 里加 2D-RoPE，处理极端 aspect ratio 和超高分辨率
5. LLM 端用 3D-RoPE（temporal + height + width），让语言模型也能感知空间

### 聪明设计 2：视频的 Time Index Token

问题：多图输入和视频输入不一样，视频帧之间有时间顺序和真实时间距离。

做法：每帧后面插一个 time index token，内容是这帧的 timestamp 字符串。模型就知道"第 3 秒"和"第 30 秒"差了 27 秒，这对 temporal grounding 很关键。

同时 ViT 里把 2D conv 换成 3D conv，temporal 维度 downsample 2 倍省算力。单图就 duplicate 成两帧保持一致。

---

## 3. Pre-training：数据工程的疯狂

### 数据构成

| 数据类型 | 规模 | 核心技巧 |
|---------|------|---------|
| Image-text pairs | 10B+ 起步，过滤后剩高质量子集 | CLIP-Score > 0.3 + concept-balanced resampling（MetaCLIP 风格）+ factual recaptioning（迭代训练一个 recaption model 去 noise + enrich） |
| Interleaved image-text | MINT + MMC4 + OmniCorpus + 100M digitized books | 训练 "high-knowledge-density" 分类器，主动 enrich 学术图表、科学插图、工程图 |
| OCR | 220M images | Synthetic（渲染文本）+ Natural（Paddle-OCR）+ Academic（Nougat 风格，arXiv LaTeX → HTML → markup） |
| Grounding | 40M natural + 140M GUI | Natural 用 GLIPv2 自动标 bounding box；GUI 用 Playwright 抓 DOM 元素的精确 bounding box |
| Video | 大规模高质量 video-text | 人工标注 fine-grained action + in-scene text + camera motion + shot composition |
| Instruction tuning | 50M samples | Fine-grained taxonomy + contamination check |

### 关键 intuition

1. **Interleaved 数据的价值不在多而在密度**：web 数据里大量是相册、广告页这种低信息密度的，作者专门训了一个分类器去挑学术图表、工程图这种高知识密度的图

2. **GUI grounding 用 Playwright 而不是 grounding model**：直接抓 DOM 元素的精确 bounding box，比 screenshot + grounding model 标注精确得多，还能生成 140M QA pair

3. **Factual recaptioning**：原始 web caption 噪声大、有 hallucination，作者迭代训练一个 recaption model，去噪 + 保留事实 + 增加细节

### Training Recipe

两阶段：
1. **Multimodal pre-training**：seq_len=8192, global batch=1536, 120K steps。MoE 用 expert parallel=8 + pipeline parallel=4 + loss-free routing
2. **Long-context continual training**：seq_len=32768, +context parallel=4, 10K steps。GLM-4.6V 再扩到 131K context, batch=128, 2K steps

Data packing：多个变长 sample 拼成接近 max length 的 sequence，省算力。

### Pre-training 效果验证

用 pass@k 在 MathVista non-MCQ 子集上测，GLM-4.1V-9B-Base 显著超同规模 SOTA base model。作者强调：**base model 的 pass@k 决定了 RL 后的天花板**，RL 只是 unlock latent potential。

---

## 4. SFT：教格式，不教知识

### 核心 thesis

SFT 的作用是**对齐 reasoning style 和 response format**，给 RL 一个好的 cold start，而不是注入新知识。

### Format 设计

```

<answer>{最终答案}</answer>
```

- verifiable task 的最终答案必须包在 `_BOX_BEGIN_{FINAL_ANSWER}_BOX_END_` 里
- 这些都是 special tokens，加进 tokenizer vocabulary，方便 RL 阶段 online parsing
- GLM-4.5V 简化了，去掉了 `<answer>` 标签

### Tool Use Format（GLM-4.6V）

```xml
<tool_call>
  {function_name}
  <arg_key>{arg-key1}</arg_key>
  <arg_value>{arg-value-1}</arg_value>
  ...
</tool_call>
```

XML schema 结构化，明确 semantic boundary，避免 structural hallucination。

### Thinking vs Non-thinking Mode

GLM-4.5V/4.6V 同时支持两种 mode。SFT 时混合训练：
- Thinking data：完整 think + answer
- Non-thinking data：用 `THINK_OFF` special token 标记，模型看到这个 token 就生成空 think content

有趣发现：**直接用 thinking data 里 `<answer>` 部分的内容做 non-thinking data，效果比专门策展一批 non-thinking data 更好**。

### Iterative Data Enhancement

把 RL 阶段 high-quality rollout 采样回 cold-start dataset，迭代增强。相当于 RL 发现的好 reasoning pattern 反哺 SFT。

---

## 5. RL：最核心的部分

### 5.1 RLCS = RL with Curriculum Sampling

问题：RL 训练到一半，大量 sample 变得太简单（accuracy > 90%），rollout 全对就产生不了 gradient，浪费算力。Rollout 是训练时间的大头。

解法：**curriculum learning + dynamic sampling**

1. **Offline difficulty grading**：训练前用多个 VLM 跑 pass@k，结合人工难度标签，把数据分成 very easy → very hard 多个 tier

2. **Online difficulty grading**：训练中每个 rollout 记录 pass@k 结果，映射到 difficulty tier，merge with offline labels

3. **Adaptive re-weighting**：按 training iteration 动态调整 sampling ratio。太简单的 down-sample，太难的也 down-sample，boost 中等难度的

**Intuition**：让模型每次更新都学最有信息量的东西，别在已经会的东西上浪费时间，也别在完全不会的东西上瞎挣扎。

### 5.2 Dynamic Sampling Expansion via Ratio EMA

问题：GRPO 里如果去掉 KL loss 和 entropy loss，一个 rollout batch 全对或全错就没 gradient。随着训练推进，这种"无效 batch"比例会波动，effective batch size 剧烈变化，训练不稳定。

解法：

1. 对每个 rollout，计算 expansion ratio：

$$\text{expansion\_ratio} = \frac{1}{1 - \text{not\_valid\_sample\_rate}}$$

- `not_valid_sample_rate`：上一轮里全对或全错 sample 的比例

2. 维护 EMA：`expansion_ratio_ema`，用这个 EMA 做下一轮的 oversampling coefficient

3. Oversample 之后，从中挑难度最 balanced 的子集（正确和错误数量最接近）

**Intuition**：预先多采样一些，然后挑"有信息量"的。比 DAPO 那种事后丢弃的方法更好，因为可以预先确定 rollout 总数，方便并行调度。

### 5.3 Reward System：多 domain 精细设计

核心发现：**多 domain RL 里，任何一个 domain 的 reward 信号有缺陷，整个训练都会崩**。

论文 Figure 5 展示了一个惨痛教训：STEM verifier 调得很好，但 single-image QA 和 multi-image QA 的 verifier 没调好，结果：
- Multi-image 出现 reward hacking（模型学 shortcut 骗 verifier）
- Single-image 出现 reward noise（reward 涨但真实 accuracy 不涨）
- 之后 STEM reward 停滞
- 整体 multimodal benchmark 下降
- MMMU、MathVista、AI2D 全崩

所以 reward system 必须 **每个 domain 都精细设计 + 有 unit test**。

### 答案提取的坑

- 简单数学题用 LLM 提取答案 OK
- 多模态开放 domain 里答案形式多样，LLM 提取经常错
- 答案 segment 可能 loop 或过长，LLM 提取 OOD

解法：**强制模型用 special tokens `_BOX_BEGIN_..._BOX_END_` 标记最终答案**，只比对 boxed 内容。比传统 `\boxed{}` 更好，因为 GUI agent 的答案是复杂 function call，`\boxed{}` 解析有歧义。

### Domain-specific Reward 设计

| Domain | Rule | Model | Binary | 细节 |
|--------|------|-------|--------|------|
| Math | √ | √ | √ | Numeric：Sympy 数值匹配带 tolerance；其他：exact match 或 LLM judge |
| Physics | √ | √ | √ | 有物理单位时用 LLM judge |
| Chemistry | √ | √ | √ | 有化学单位时用 LLM judge |
| Chart | √ | √ | √ | Numeric：类似 Math（Year 除外）；Textual：exact match 或 LLM judge |
| OCR | √ | | | Edit distance：reward = 1 - max(|ans|,|gt|) / max(|ans|,|gt|) |
| Grounding | √ | | | Reward = #boxes(IoU > τ) / total boxes |
| GUI Agent | √ | √ | | Action prediction：action+IoU；Grounding：IoU；QA：exact 或 semantic |
| Video | √ | √ | √ | Exact match 或 LLM semantic match |

### 5.4 其他 RL Trick

**Effectiveness 方面**：

1. **Larger batch size**：多 domain 混合训练时大 batch 有更高 performance ceiling

2. **Force answering**：thinking 太长被截断时，插入 `` token 强制模型出答案。这样长的 reasoning 也能得到公平 reward，不会因为截断就给 0 reward。还让模型学会"思考任意长度后都能给出答案"，test time 能动态控制 thinking budget

3. **Discard KL loss**：VLM 的 KL divergence 在 RL 中增长比纯文本 LLM 快很多。加 KL loss 抑制会明显限制模型能力，所以直接去掉

4. **Clip-higher**：提高 importance sampling ratio 的上 clip bound，改善 off-policy 性能 + 防止 entropy collapse

**Stability 方面**：

1. **Cold-start SFT data 质量是 stability 的关键**：有大量无意义 thinking path 的 SFT data 会导致 RL 训练严重不稳定甚至 collapse

2. **去掉 entropy loss**：加 entropy loss 促多样性反而导致 garbled output → 训练崩

3. **top-p = 1 比 top-p = 0.9 更稳定**：小 top-p 短期降 variance 但长期会增加 garbling 风险。top-p = 1 保证 full vocabulary coverage，rare token 不被 under-learned，输出保持干净

4. **Per-sample loss 比 per-token loss 稳定**：mean reward 差不多但 per-sample 训练更稳

5. **Format 要在 SFT 阶段学好，别指望 RL 阶段用 format reward 纠正**：如果 RL 阶段还频繁出 format error，format reward 和 correctness reward 混在一起会 destabilize training

### 5.5 Infrastructure 优化

1. **Sequence length load balancing across DP ranks**：不同 sample rollout 长度差异大，不做 balancing 会被最慢的 rank 拖死。先 rollout 完再按 sequence length + compute load 分配

2. **Sequence packing + gradient accumulation**：不知道每个 DP rank 要跑多少 forward pass，就用 fixed-length (32K) packing + gradient accumulation。多个 micro-step 的 gradient 按样本数加权平均

3. **Sample re-packing heuristic**：在 DP rank 内用 heuristic 把 sample 打包成最少 micro-step，实践中 forward-backward 时间减半

---

## 6. 实验结果

### 6.1 主表对比（Table 2 节选）

| 类别 | Benchmark | GLM-4.1V-9B | GLM-4.5V | GLM-4.6V | Qwen2.5-VL-72B | Gemini-2.5-Flash |
|------|-----------|-------------|----------|----------|----------------|------------------|
| General VQA | MMStar | 72.9 | 75.3 | 75.9 | 70.8 | - |
| General VQA | MUIRBENCH | 74.7 | 75.3 | 77.1 | 62.9 | - |
| STEM | MMMU Pro | 57.1 | 65.2 | 66.0 | 51.1 | - |
| STEM | MathVista | 80.7 | 84.6 | 85.2 | 74.8 | - |
| STEM | MathVision | 54.4 | 65.6 | 63.5 | 38.1 | - |
| Chart | ChartQAPro | 59.5 | 64.0 | 65.5 | 46.7 | - |
| Chart | ChartMuseum | 48.8 | 55.3 | 58.4 | 39.6 | - |
| Long Doc | MMLongBench-Doc | 42.4 | 44.7 | 54.9 | 35.2 | - |
| Grounding | RefCOCO-avg | 85.3 | 91.3 | 88.6 | 90.3 | - |
| Grounding | TreeBench | 37.5 | 50.1 | 51.4 | 42.3 | - |
| GUI Agent | OSWorld | 14.9 | 35.8 | 37.2 | 8.8 | - |
| GUI Agent | AndroidWorld | 41.7 | 57.0 | 57.0 | 35.0 | - |
| Coding | Design2Code | 64.7 | 82.2 | 88.6 | 41.9 | - |
| Coding | Flame-React-Eval | 72.5 | 82.5 | 86.3 | 46.3 | - |
| Video | VideoMME (w/sub) | 73.6 | 74.6 | 74.8 | 73.3 | - |
| Video | VideoMMMU | 61.0 | 72.4 | 74.7 | 60.2 | - |

**关键观察**：
- GLM-4.5V 在开源同规模模型里几乎全面 SOTA
- 9B 的 GLM-4.1V-Thinking 在 29 个 benchmark 上超过 Qwen2.5-VL-72B（大 8 倍参数）
- GUI Agent（OSWorld 35.8 vs 8.8）和 Coding（Design2Code 82.2 vs 41.9）提升尤其夸张

### 6.2 RL 带来的增益

Figure 1B 显示 RL 在各 domain 都有大幅提升，最高 +10.6%。

### 6.3 Cross-Domain Generalization（最重要的实验）

这是论文最有 insight 的实验。选 4 个 domain：STEM、OCR&Chart、Grounding、GUI Agent。分别单独做 RL 和混合做 RL（mix-all），看各 domain benchmark 的变化。

**关键发现**：

1. **单 domain RL 会提升其他 domain**：
   - 只在 STEM 上 RL → grounding、GUI agent、general VQA 都涨
   - 只在 OCR&Chart 上 RL → STEM、GUI agent、general VQA 都涨
   - 只在 GUI agent 上 RL → 所有 domain 都涨（因为 GUI agent 内含 text recognition + grounding + reasoning）

2. **混合 RL 效果更好**：mix-all 在 STEM、OCR&Chart、general VQA 三个方向上超过任何单 domain RL

3. **但也有例外**：mix-all 在 grounding 和 GUI agent 上没有超过单 domain RL，说明这些 domain 可能需要更 targeted 的策略

**Intuition**：这些 multimodal task 底层共享 visual understanding + text recognition + reasoning 能力。一个 domain 的 RL signal 会 co-activate 和 refine 这些共享能力，自然迁移到其他 domain。GUI agent 效果特别好是因为它天然要求这些能力的综合。

---

## 7. Limitations 和 Future Work

1. **RL 提升 accuracy 但不一定提升 reasoning quality**：模型可能用错误推理得到正确答案，因为 reward 只看结果不看过程。未来需要 reward 评估中间 reasoning step

2. **RL 训练不稳定**：小改动可能导致 reasoning depth 和 output style 大变。需要更 robust 的 RL optimization

3. **复杂场景仍会出错**：clutter、occlusion、ambiguous visual detail 会导致 perceptual error，进而 reasoning 也跟着错。perception 和 reasoning 要同步提升

4. **Benchmark 饱和**：现有 benchmark 逐渐无法区分 reasoning chain 里的 hallucination 和 shortcut，需要更有诊断力的 benchmark

---

## 8. 我的 Takeaway

这篇 paper 最大的 insight 有三个：

1. **Pre-training 决定天花板，RL 只是 unlock**。所以要在 pre-training 上砸数据工程，把 base model 的 pass@k 做高

2. **多 domain RL 不是零和游戏**。只要 reward system 每个 domain 都精细调好，domain 之间会互相促进。任何一个 domain 的 reward 有缺陷，整个训练都会崩

3. **Curriculum sampling + dynamic expansion 是 RL efficiency 的关键**。随着模型变强，要动态剔除太简单的样本，oversample 后挑难度 balanced 的子集，保持每次更新都有信息量

参考链接：
- GitHub: https://github.com/zai-org/GLM-V
- GLM-4.6V blog: https://z.ai/blog/glm-4.6v
- GUI Agent 示例: https://github.com/zai-org/GLM-V/blob/main/examples/gui-agent/glm-41v/agent.md

---

# GLM-4.5V / GLM-4.1V-Thinking 技术深度解析

## 1. 核心立意：Reasoning-Centric 的统一框架

这篇 paper 的核心 thesis 是一个 **reasoning-centric training framework**，它把 pre-training → SFT → RL 三个 stage 串到一个 unified objective 上：**通过 scalable RL 全面提升 multimodal reasoning**。这里的关键 intuition 是：

- **Pre-training 决定 upper bound**：作者用 pass@k 在 MathVista non-MCQ 子集上证明，GLM-4.1V-9B-Base 的 base 能力已经显著超过同规模 SOTA base model，而 RL 只是 unlock 这个 latent potential，并不是凭空创造能力。这与 DeepSeek-R1 的观察一致——base model 的潜力是天花板。
- **SFT 是桥，不是知识注入**：作者明确说 SFT 的 role 是 align format/style 而非 inject knowledge，这与很多 traditional SFT pipeline 不同。这里的长 CoT 数据只是为了"教模型怎么think"，而不是"教模型新东西"。
- **RL 是 unlock 的核心机制**：RLCS 是论文最重要的 contribution。

参考链接：
- GitHub: https://github.com/zai-org/GLM-V
- Blog: https://z.ai/blog/glm-4.6v

---

## 2. Architecture 深度解析

### 2.1 三组件 + MoE 结构

| Component | GLM-4.1V-Thinking / GLM-4.6V-Flash | GLM-4.5V / GLM-4.6V |
|-----------|-----------------------------------|---------------------|
| Vision Encoder | AIMv2-Huge | AIMv2-Huge |
| LLM Decoder | GLM-4-9B-0414 (dense) | GLM-4.5-Air (106B-A12B MoE) |
| Projector | MLP | MLP |

106B-A12B 表示 total parameters 106B，activated parameters 12B（典型的 MoE 稀疏激活）。

### 2.2 关键公式：Position Embedding Bicubic Interpolation

这是 paper 里最数学化的部分，直觉非常重要。

**公式 (1)**：把 patch 的整数 grid 坐标 $\mathbf{g} = (w, h)$ normalize 到 $[-1, 1]$：

$$\mathbf{g}_{\mathrm{norm}} = (w_{\mathrm{norm}}, h_{\mathrm{norm}}) = 2 \cdot \left( \frac{w + 0.5}{W_p}, \frac{h + 0.5}{H_p} \right) - 1$$

变量含义：
- $w, h$：patch 在 grid 中的 integer column / row index（从 0 开始）
- $W_p, H_p$：grid 的总宽度和总高度（patch 数量）
- $+0.5$：取 patch 中心而非左上角，避免 boundary bias
- $2 \cdot (\cdot) - 1$：把 $[0,1]$ 映射到 $[-1,1]$，这是 bicubic interpolation 的标准输入范围

**公式 (2)**：从原 position embedding table $P_{\mathrm{orig}}$ 中通过 bicubic 插值得到 adapted embedding：

$$P_{\mathrm{adapted}}(\mathbf{g}) = \mathcal{T}_{\mathrm{bicubic}}(P_{\mathrm{orig}}, \mathbf{g}_{\mathrm{norm}})$$

变量含义：
- $P_{\mathrm{orig}}$：pre-trained 时学到的 absolute position embedding table，size 固定（比如 $14 \times 14$）
- $\mathcal{T}_{\mathrm{bicubic}}$：bicubic 插值函数，用周围 16 个 grid 点做 cubic 拟合
- $P_{\mathrm{adapted}}(\mathbf{g})$：为任意分辨率图像的任意 patch 生成的 position embedding

**Intuition**：这个设计非常巧妙。传统做法要么 resize 图像到固定大小（丢信息），要么直接用 RoPE 完全替代 absolute PE（可能破坏 pre-trained 能力）。GLM 的做法是**保留 pre-trained 的 absolute PE，但通过 interpolation 让它能 adapt 到任意分辨率**。这样既保留了 AIMv2 pre-trained 的 visual prior，又支持 native resolution。2D-RoPE 则加在 self-attention 里处理 extreme aspect ratio（>200:1）和 high resolution（>4K）。

### 2.3 视频处理：3D Conv + Time Index Tokens

- 把 ViT 的 2D conv 换成 3D conv，temporal 维度 downsampling factor = 2，提高 efficiency
- 单图 duplicate 成两帧保持 consistency
- 每帧后插入 **time index token**，编码为 timestamp 字符串

这个 time index 设计的 intuition 是：multimodal 模型用 multi-image 方式处理视频会丢失 temporal distance 信息。通过显式插入 timestamp，模型知道"第 3 秒"和"第 30 秒"之间的真实时间差，这对 temporal grounding 至关重要。

### 2.4 3D-RoPE in LLM

LLM 端把 RoPE 扩展成 3D-RoPE，三个维度分别是 temporal、height、width。这样 multimodal context 中的 spatial 信息能被 LLM 直接感知，同时 preserve 了 original text capability（因为 text token 在 temporal/h 维度上都是默认值）。

---

## 3. Pre-training：数据工程的极致

### 3.1 数据规模与构成

| 数据类型 | 规模 | 关键技术 |
|---------|------|---------|
| Image-text pairs | 10B+ initial → filtered | CLIP-Score > 0.3 + concept-balanced resampling + factual recaptioning |
| Interleaved image-text | MINT + MMC4 + OmniCorpus + 100M books | "High-knowledge-density" classifier 筛选学术图表 |
| OCR | 220M images | Synthetic + natural (Paddle-OCR) + academic (Nougat-style) |
| Grounding | 40M natural + 140M GUI | GLIPv2 + Playwright DOM extraction |
| Instruction tuning | 50M samples | Fine-grained taxonomy + contamination check |

### 3.2 关键 intuition：Interleaved 数据的处理

作者强调 prior work 很少 scale interleaved data，因为噪声大。他们的 pipeline 包括：
1. CLIP-Score 阈值过滤语义不相关图片
2. 启发式 + 分类器移除广告、QR code
3. 排除"高图片密度低文本"样本（如相册）
4. **训练 "high-knowledge-density" 分类器**，主动 enrich 学术图表、科学插图、工程图

这个 intuition 是：interleaved 数据的价值不在"多"，而在"信息密度高"。

### 3.3 GUI Grounding 的创新

从 CommonCrawl snapshot 提取 URL，用 Playwright 深度交互网页，compile + parse 所有 visible DOM elements 及其 rendered bounding boxes。这比传统 screenshot-based grounding 精确得多，因为 DOM 元素的 bounding box 是精确的，不需要额外的 grounding model 标注。

### 3.4 Training Recipe

两阶段：
1. **Multimodal pre-training**：seq_len=8192, global batch=1536, 120K steps。GLM-4.5V 用 expert parallel=8 + pipeline parallel=4，loss-free routing + auxiliary balance loss (coef=1e-4)
2. **Long-context continual training**：seq_len=32768, +context parallel=4, 10K steps。GLM-4.6V 扩展到 131K context，再训 2K steps batch=128

**Data packing** 策略：把多个变长 sample 拼成接近 max length 的 sequence，最大化 compute efficiency。

---

## 4. SFT：标准化 Reasoning Format

### 4.1 Format Design

```

<answer>{answer_content}</answer>
```

关键设计点：
- 对 verifiable task，final answer 必须包在 `<|begin_of_box|>{FINAL_ANSWER}<|end_of_box|>` 里
- 这些都是 **special tokens** 加入 tokenizer vocabulary，便于 online parsing
- GLM-4.5V 中消除了 `<answer>` 标签（更简洁）

### 4.2 Thinking vs Non-thinking Mode

GLM-4.5V/4.6V 同时支持两种 mode：
- Thinking mode：完整 `
