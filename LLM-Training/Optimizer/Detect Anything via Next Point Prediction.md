---
source_pdf: Detect Anything via Next Point Prediction.pdf
paper_sha256: df925724e684b7e9b1a5ece9fcd8ad6527568fe21efcb8ddb8f96ac192b3bb0a
processed_at: '2026-08-18T05:18:36-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Rex-Omni 这篇 paper

---

## 这篇 paper 到底在干啥?

一句话: **教一个 MLLM (多模态大语言模型) 好好做 object detection, 做到能跟 YOLO、DETR 这种专门干检测的模型打一打, 甚至超过它们。**

你可能觉得这事不是早就有人做了吗? 让 GPT-4V 之类输出坐标不就行了? 确实有人这么做, 但都做得不太好。这篇 paper 的贡献就是搞清楚 **为什么做得不好**, 然后对症下药。

---

## 先说背景: 检测这个活儿, MLLM 为什么一直打不过传统 detector?

传统 detector 像 YOLO、DETR、Grounding DINO 这些, 本质上是 **regression**——回归出 box 的坐标 (x, y, w, h)。它们的 loss 是 L1、GIoU 这种 **连续的、对像素偏移敏感的** loss。box 差 2 个像素, loss 就差一点点, 梯度平滑, 模型学得很顺。

MLLM 的做法完全不同。它把坐标当成 **文本 token** 来生成, 比如生成 `<12><42><512><612>` 这样一串 token 表示一个 box。训练用 cross-entropy loss, 就是分类 loss。

这里就出问题了。

**问题 1: CE loss 对几何不敏感**

假设真实坐标是 `<33>`, 模型预测 `<32>`, 像素上可能就差 1 个 bin, 但 CE loss 把这个当成 **完全错误的分类**, 惩罚和预测 `<999>` 一样大。模型学不到 "差一点点其实也还行" 这个概念。

反过来, 一个 box 的 4 个坐标里只错 1 个 token, CE loss 看着不大, 但 box 可能完全错位。CE loss 给不出合理的几何惩罚。

**问题 2: SFT 的 teacher forcing 让模型不会 "自己决定" 预测几个 box**

SFT 训练时, 模型永远看着 ground truth 的前缀来生成下一个 token。Ground truth 里有 5 个 box, 模型就学 "在 GT 的前 4 个 box 之后生成第 5 个 box"。它从来不需要自己决定 "该 stop 了" 或者 "该预测第几个"。

到了 inference, 没有 GT 引导了, 模型就懵了。它会:
- 反复输出同一个 box (duplicate)
- 输出一个超大 box 盖住所有东西 (large-box)
- 或者该预测的没预测 (漏检)

这两个问题, 就是之前所有 MLLM 检测器 (包括 Qwen2.5-VL、SEED1.5-VL 这种很强的模型) 打不过传统 detector 的根本原因。

---

## Rex-Omni 怎么解决的? 三个设计

### 设计 1: 坐标怎么表示? —— 用 special token

他们比较了几种方案:

- Qwen2.5-VL: 绝对坐标, 把 `1921` 拆成 `1` `9` `2` `1` 四个数字 token
- SEED1.5-VL: 相对坐标量化到 1000 bin, 但拆成多个 atomic token
- Rex-Omni: 相对坐标量化到 1000 bin, 每个值用 **一个 special token** `<192>`

为什么用 special token?

第一, 1000-way 分类比 "任意数字组合" 简单多了。学习难度从 unbounded 降到 bounded。

第二, token 数量大幅减少。一个 box 只要 4 个 token `<x0><y0><x1><y1>`。SEED1.5-VL 一个 box 要 148.8 个 token。这在密集场景 (一张图 80 个 box) 下差距巨大——SEED1.5-VL 要生成 5000+ token, Rex-Omni 只要 400 多个。直接关系到能不能跑、跑多快。

---

### 设计 2: 数据从哪来? —— 四个 data engine

要让模型学会 1000 个 coordinate token 和像素位置的映射, 需要海量数据。公开数据集 8.9M 不够, 他们又造了 13M 数据。

四个 engine 各干各的:

**Grounding engine**: 给图生成 caption → 提取名词短语 → **去掉带形容词的** (比如 "green lemon" 丢掉, 只留 "lemon") → 用 DINO-X 检出 box。

去掉形容词这一步很关键。因为 DINO-X 对 "green lemon" 这种描述理解不好, 会把所有 lemon 都框出来, 标注就脏了。宁可少标, 不要标错。

**Referring engine**: 想生成 "穿黄衬衫的男人" 这种描述, 怎么自动关联到 box?

他们用了个巧妙的组合:
- Qwen2.5-VL 生成 referring 表达式
- Molmo 来 **指一个点** (Molmo 理解语言强, 但只输出点)
- SAM 生成 mask
- 看点落在哪个 mask 里, 就把那个 box 关联到这个表达式

这样就能自动生成 "描述 → box" 的监督数据, 不需要人工标。

**Pointing engine**: 从已有 box 转成 point 标注。不是简单取 box 中心 (细长物体会落到外面), 而是 SAM 出 mask, 算最小外接旋转矩形, 取对角线交点。

**OCR engine**: PaddleOCR 标 polygon 和文字, 再转 box。

总共 22M 数据。

---

### 设计 3: 两阶段训练 —— SFT + GRPO

**Stage 1: SFT**

就是标准的 next-token prediction, cross-entropy loss, 22M 数据, 64 张 A100 跑 8 天。这阶段让模型学会 "坐标预测的基本功"。

**Stage 2: GRPO (Group Relative Policy Optimization)**

这是重点。66K 数据 (相比 SFT 的 22M, 只是 0.3%), 8 张 A100 跑 24 小时。

核心思路: 对同一个输入, 让模型 **采样 8 个不同的输出**, 用 reward 函数给每个输出打分, 然后用 RL 优化——分高的多鼓励, 分低的抑制。

reward 函数是 **geometry-aware** 的, 三种:
- Box IoU reward: 预测 box 和 GT 的 IoU 越高, reward 越高, 而且算 F1 同时惩罚漏检和重复
- Point-in-mask reward: 预测点落在 GT mask 里给 reward
- Point-in-box reward: GUI grounding 用, 点落在 box 内给 reward

为什么 GRPO 有效? 这篇 paper 的分析部分讲得很透, 我觉得是全文最精彩的地方:

---

## GRPO 到底干了啥? (这是全文核心)

很多人以为 GRPO 是让模型 "预测得更准"。这篇 paper 证明: **不是。**

他们做了个实验, 只看那些 "SFT 和 GRPO 都预测对了 box 数量、都 match 了 GT" 的样本, 比较两者的 box 精度。结果:

- COCO: SFT 63.0 → GRPO 63.5 (提升 0.5)
- LVIS: SFT 56.6 → GRPO 56.9 (提升 0.3)

**几乎没提升。** SFT 已经把坐标预测的精度学会了, GRPO 在精度上没帮上多少忙。

那 GRPO 提升的是啥? 是 **行为纠正**。

**证据 1: 去重实验**

SFT 模型在 VisDrone 上如果手动去掉重复预测, F1 从 55.6 涨到 62.3 (+15.3%)。说明 SFT 模型 15% 的错误都是重复预测。GRPO 模型去重后只涨 0.1%, 说明它几乎不重复了。

**证据 2: 去大 box 实验**

SFT 模型在 Dense200 上 20.5% 的预测是 "一个超大 box 盖住所有东西" 这种作弊行为。去掉这些后 F1@mIoU 从 44.9 涨到 56.7。GRPO 模型只有 3.5% 是大 box。

**证据 3: 采样实验**

用 SFT 模型高温采样 8 次, 取最好的, COCO 上能达到 72.6 (甚至略超 GRPO 的 72.0)。说明 SFT 模型 **有能力** 预测对, 只是 inference 时随机采样采不到好的。GRPO 教模型更稳定地采到好的。

但在 Dense200 这种复杂任务上, SFT 采样 8 次取最好也只有 38.2, 远低于 GRPO 的 78.4。说明在难任务上, GRPO 不只是提升采样稳定性, 是真的让模型预测得更连贯、更合理。

---

## 整体 intuition: 三个层次

**第一层**: SFT 教模型 "坐标怎么预测" (知识注入)

22M 数据, 8 天, 把 1000 个 token 和像素位置的映射学会。这阶段模型有 latent capability, 但不知道怎么用好。

**第二层**: GRPO 教模型 "行为怎么控制" (能力调用)

66K 数据, 24 小时, 通过 reward 让模型学会:
- 别重复 (Precision 惩罚)
- 别作弊画大 box (IoU reward 让大 box reward 低)
- 该停就停 (没 GT 可匹配的预测 reward 是 0)

这阶段不教新知识, 只教用好已有知识。

**第三层**: Geometry-aware reward 桥接 discrete-continuous gap

CE loss 对像素偏移不敏感, IoU reward 敏感。用 IoU 做 reward 等于在 token 分类的任务上叠加几何先验, 把 Challenge 1 的 mismatch 给 fix 了。

---

## 结果怎么样?

几个亮点数字:

- **COCO zero-shot**: F1@0.5 = 72.0, 超过 DINO-R50 (68.8) 和 Grounding DINO (69.8)。这是 MLLM 首次在这个 setting 下 beat 传统 detector。
- **Dense200**: F1@0.5 = 78.4, SEED1.5-VL 是 76.9, Qwen2.5-VL-3B 只有 0.8 (完全崩)。密集场景 MLLM 一直很弱, 这里 GRPO 提升 +18.2 极其显著。
- **Pointing**: 全面 SOTA, COCO 80.5, Dense200 82.5。
- **OCR**: ICDAR2015 F1@0.5 = 45.2, 超过专门的 PaddleOCR (38.2)。
- **GUI grounding**: ScreenSpot-V2 = 88.4, 在 3B 模型里最强。

---

## 局限

- F1@0.95 (要求极紧的 box) 上还是打不过 regression-based detector。1000-bin 量化是天花板, 极高精度场景受限。
- 推理速度: dense scene 几百个 box 要十几秒, 慢于优化的传统 detector。需要 quantization / distillation。
- Visual prompting 还打不过 T-Rex2 (专门的 expert model)。
- GRPO 的 reward 需要 GT box, RL 阶段只能用 labeled data。

---

## 我觉得这篇 paper 最大的 insight

**MLLM 检测的核心瓶颈不是 "预测坐标的能力不够", 是 "生成行为不受控"。**

SFT 已经把坐标预测的基本功教好了, 但 teacher forcing 的训练方式让模型不知道怎么在自由生成时控制自己的输出结构 (预测几个、何时停、别重复)。

GRPO 用 geometry-aware reward + group-relative advantage, 专门 fix 这个行为问题。66K 数据 (vs SFT 的 22M) 就能把 behavior 拉上来, 说明问题不在 "知识" 而在 "调用"。

这个 insight 其实跟 LLM reasoning 上 RL 的观察一致——RL 的价值更多在 "教模型用好已有能力", 而不是 "灌输新知识"。DeepSeek-R1 那篇也是类似的 story。

对做 MLLM 的人来说, 这篇 paper 的启示是: **如果你用 MLLM 做结构化输出任务 (检测、分割、点云等), 别只盯着 SFT, RL post-training 可能是解决 "行为失控" 的关键。**

---

想让我展开哪个部分都可以, 比如 GRPO 的 reward 匹配细节、data engine 的具体 prompt 设计、或者 coordinate quantization 的 error analysis。

---

# Rex-Omni: Detect Anything via Next Point Prediction — 深度解析

这篇 paper 来自 IDEA Research (Lei Zhang 团队), 核心目标是把 MLLM-based object detection 的性能推到与 regression-based detector (DINO, Grounding DINO) 可比甚至超越的水平。我下面把每个核心设计拆开讲, 重点 build intuition。

---

## 1. 核心问题诊断: 为什么之前的 MLLM 检测器打不过传统 detector?

Paper 在 Section 1 提出两个根本性 challenge, 这两个 challenge 决定了整个 Rex-Omni 的设计:

### Challenge 1: Discrete-to-Continuous Mapping 的 Geometry Mismatch

传统 detector 用 L1 / GIoU 这种 continuous loss, 对小像素 offset 敏感, 梯度信号 smooth。MLLM 把坐标当 token 分类, 用 cross-entropy loss, 这里有个根本 mismatch:

假设 GT token 是 `<33>`, 模型预测 `<32>`, 像素空间可能只差 1 个 bin, 但 CE loss 把它当完全错误分类, 给的惩罚和预测 `<999>` 一样大。反过来, 如果 GT 是 `<0><0><100><100>`, 模型预测 `<0><0><100><1000>`, 只错 1 个 token, CE loss 很小, 但 box 几何上完全错位。这个 asymmetry 是 MLLM 检测器精度上不去的核心原因之一。

### Challenge 2: SFT Teacher Forcing 的 Behavioral Deficiency

SFT 用 teacher forcing, 模型永远 condition 在 GT prefix 上, 训练时 box 数量固定等于 GT 数量。模型从来不需要"决定"预测几个 box。inference 时没有 GT 引导, 模型不知道何时 stop, 导致两种典型错误:
- 重复预测 (duplicate boxes, 坐标几乎相同)
- 漏检 (missed detections)

这个就是 sequence generation 中经典的 exposure bias, 但在检测任务里表现为 box 数量失控。

这两个 challenge 直接对应 Rex-Omni 的三大设计: Task Formulation 缓解 Challenge 1 的 learning difficulty, Data Engine 提供 mapping 所需的大规模监督, GRPO 同时 fix Challenge 1 (geometry-aware reward) 和 Challenge 2 (behavior-aware optimization)。

---

## 2. Task Formulation: 为什么是 Relative Coordinates + Special Tokens?

### 2.1 三种 paradigm 对比 (Figure 3a)

Paper 把 MLLM 坐标预测分三类:
1. **Direct Coordinate Prediction** (Pix2Seq, Shikra, Ferret, Rex-Omni): 坐标作为 vocabulary 中的 discrete token
2. **Retrieval-based** (Groma, ChatRex): LLM 预测 candidate region 的 index
3. **External decoder** (Lisa, SegZero): LLM 输出 special token embedding, 外部 decoder 生成坐标

Rex-Omni 选 Direct, 理由是 simplicity + flexibility, 不依赖额外模块或监督。

### 2.2 Direct prediction 的三种变体 (Figure 3b)

这是非常关键的设计决策:

| 方法 | 坐标表示 | 代表模型 | 一个 box 的 token 数 |
|------|---------|---------|-------------------|
| Relative + Special tokens | `<12><42><512><612>` (每个值 0-999 一个 special token) | Pix2Seq, **Rex-Omni** | 4 |
| Relative w/o special tokens | `0 1 2` `0 4 2` ... (拆成 atomic tokens) | SEED1.5-VL | ~15 |
| Absolute coordinates | `1 9 2 1` (绝对像素值拆数字) | Qwen2.5-VL | variable |

**为什么 Relative + Special tokens 最优?**

**Intuition 1: 学习复杂度降低。** Absolute coordinate 把问题变成 unbounded 分类, 模型要学 0 到 image_width 任意值。Relative + quantization 把它压缩到 1000-way 分类, bounded, 学习难度大幅降低。1000 这个数字是 trade-off: 太小精度不够, 太大分类难度上升。1000 对应 0.1% 像素精度, 对绝大多数检测任务够用。

**Intuition 2: Token efficiency。** 这点对 dense scene 极其关键。Table 17 的数据非常 striking:

| 模型 | COCO tokens/box | Dense200 tokens/box |
|------|----------------|---------------------|
| SEED1.5-VL | 148.8 | 74.5 |
| Rex-Omni | 7.6 | 5.1 |

Dense200 一张图平均 86.7 个 box, SEED1.5-VL 要生成 5446 tokens, Rex-Omni 只要 439 tokens。token 数量直接影响 KV cache 大小、generation 时间、attention 计算量。Figure 18 显示 detect 410-419 个 box 要 16 秒, 这里的瓶颈就是 autoregressive token generation。

**Intuition 3: Special token 的 semantic 独立性。** 用 atomic tokens 拆 `192` 成 `1` `9` `2`, 模型要学数字组合的 compositional 规则, 而且 `1` `9` `2` 和 `9` `2` `1` 在 embedding 空间没有 geometry 关系。Special token `<192>` 是一个 atomic embedding, 模型可以学到这个 embedding 和相邻 token `<191>` `<193>` 在 geometry 上的近邻关系 (虽然 CE loss 不直接鼓励, 但 large-scale data 会让模型隐式学到)。

### 2.3 统一输出格式

所有任务统一成:
```
<|object_ref_start|>PHRASE<|object_ref_end|><|box_start|>COORDS<|box_end|>
```

不同任务 COORDS 格式不同:
- Detection: `[x0, y0, x1, y1]` (4 tokens)
- Pointing: `[x0, y0]` (2 tokens)
- Polygon (OCR): `[x0, y0, x1, y1, x2, y2, ...]`
- Keypoint: JSON 格式 `{"box": <...>, "keypoints": {"left eye": <x><y>, ...}}`

这个统一的 next-point-prediction 范式很优雅: 所有视觉感知任务都是预测一串 coordinate points, 只是 point 数量和语义不同。

参考: [Pix2Seq paper](https://arxiv.org/abs/2109.10852) 最早提出把检测当 language modeling 的思路。

---

## 3. Model Architecture

架构非常 minimal (Figure 4):
- Backbone: Qwen2.5-VL-3B-Instruct
- Vision encoder: native resolution ViT, patch size 28, image tokens 16-2560 (对应 16×28×28 到 2560×28×28 pixels)
- Modification: 复用 vocabulary 最后 1000 个 tokens 作为 coordinate special tokens, **不引入任何新参数**

这里有个 engineering 技巧: 不是 add 新 tokens (会改变 embedding table 大小, 影响 pretrained distribution), 而是 repurpose 末尾 1000 个低频 tokens。这保留了 pretrained model 的所有知识, 只重新赋予这 1000 个 token 新的语义。

Qwen2.5-VL 原本用 absolute coordinates (拆数字), Rex-Omni 在 inference 时把模型输出解释成 relative 0-999 量化值, 然后映射回原图像素。

参考: [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923)

---

## 4. Data Engines: 22M 样本怎么来的?

这是 paper 的重头戏之一。8.9M public data + ~13M generated data = 22M total。

### 4.1 Grounding Data Engine (3M images)

Pipeline (Figure 5 top):
1. **Image Captioning**: Qwen2.5-VL-7B 生成 caption
2. **Phrase Extraction**: SpaCy 提取 noun phrases
3. **Phrase Filtering**: **关键创新**, 去掉带形容词的 phrase (e.g. "green lemon" 丢弃, "lemon" 保留)
4. **Phrase Grounding**: DINO-X 检测对应 box

为什么 Phrase Filtering 重要? Intuition: 当前 grounding model (DINO-X) 对 "green lemon" 这种属性描述理解不行, 会检测所有 lemon, 引入大量 label noise。去掉形容词只保留 category noun, 标注质量大幅提升。这是 data-centric 的思路, 比 model-centric 改进更 cost-effective。

Image source: COYO + SA-1B, 经过 resolution filtering 和 NSFW filtering。

参考: [DINO-X](https://arxiv.org/abs/2411.14347), [SAM / SA-1B](https://arxiv.org/abs/2304.02643)

### 4.2 Referring Data Engine (3M images)

这个 engine 解决一个关键问题: 如何自动生成 "a man in a yellow shirt" 这种 semantically rich 的 referring expression, 并且 grounding 到正确的 box?

Pipeline (Figure 5 bottom):
1. **Expression Generation**: Qwen2.5-VL-7B 根据 image + category 生成 referring expression
2. **Pointing**: Molmo (state-of-the-art referring model) 输出 point
3. **Mask Generation**: SAM 对每个 GT box 生成 mask
4. **Point-to-Box Association**: Molmo 的 point 落在哪个 SAM mask 里, 就把对应的 box 关联到 referring expression

**Intuition**: 这里巧妙利用 Molmo 的 strength (referring understanding 强, 但只输出 point) + SAM (mask generation 强) 组合, 绕过 Molmo 不输出 box 的限制。Point-in-mask 是一个非常 robust 的 association criterion, 比 IoU matching 对 box 边界不敏感。

参考: [Molmo / PixMo](https://arxiv.org/abs/2409.17146), [RexSeek](https://arxiv.org/abs/2503.08507) (这个是同组前作, 强调 one-to-many referring)

### 4.3 Pointing Data Engine (5M samples)

从 box-level annotation 转 point-level:
- SAM 生成 mask
- 算 minimum-area enclosing rotated rectangle (考虑物体旋转)
- 取对角线交点作为 candidate point
- 如果点在 mask 内, 就用, 否则丢弃

**Intuition**: 简单取 box center 在细长物体 (e.g. 笔, 鱼竿) 上会落在 mask 外。minimum rotated rectangle 的对角线交点更接近物体 visual center, 几何上更合理。

### 4.4 OCR Data Engine (2M samples)

PaddleOCR 标注 polygon + transcription, 然后算 minimum axis-aligned rectangle 作为 box representation。

---

## 5. Two-Stage Training Pipeline

### 5.1 Stage 1: SFT

- Loss: standard cross-entropy
- Data: 22M annotated samples
- Hardware: 8 nodes × 8 A100 = 64 GPUs
- Time: 8 days
- Learning rate: vision encoder 2e-6, projection + LLM 2e-5 (vision encoder 用更小 lr 保护 pretrained feature)
- Optimizer: AdamW, warmup 3%, weight decay 0.01
- 全参数更新 (full fine-tuning, 不是 LoRA)

训练数据构造技巧:
- **Conversation Templates**: GPT-4o 生成多个 question template 模拟真实用户
- **Multi-Phrase Queries**: 一张图有 N 个 phrase, 随机 sample 1-N 个组成 query (训练模型处理多类别同时检测)
- **Visual Prompting Training**: 对每个 category 随机 sample 1-N 个 box 作为 visual prompt, 转 coordinate token 嵌入 query

### 5.2 Stage 2: GRPO Reinforcement Post-Training

这是 paper 的核心创新, 我详细讲公式。

#### GRPO Framework

给定 image 和 question $(I, x)$, 从当前 policy $\pi_\theta$ 采样 G 个完整 response $\{o_1, o_2, ..., o_G\}$。每个 response $o_i$ 包含完整 reasoning trace + 最终 coordinate/box 预测。

**公式 (1): Group-relative advantage**

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

变量解释:
- $A_i$: 第 $i$ 个 response 的 advantage (相对值)
- $r_i$: 第 $i$ 个 response 的 scalar reward (由 geometry-aware reward function 计算)
- $G$: group size, 同一 prompt 采样的 response 数量
- $\text{mean}(r_1, ..., r_G)$: G 个 reward 的均值
- $\text{std}(r_1, ..., r_G)$: G 个 reward 的标准差

**Intuition**: GRPO 相对 PPO 的关键创新是不需要 value network。用 group 内的 mean 作 baseline, std 作 normalization。如果一个 response 的 reward 高于 group mean, $A_i > 0$, 鼓励; 低于 mean, $A_i < 0$, 抑制。这避免了训练一个准确的 value function 的难度, 在 MLLM 这种巨大 action space 下特别实用。

**公式 (2): GRPO objective**

$$\mathcal{L}_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\min(\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}) - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]\right]$$

变量解释:
- $\theta$: 当前 policy 模型参数 (要优化的)
- $G$: group size
- $|o_i|$: 第 $i$ 个 response 的 token 长度
- $t$: response 内第 $t$ 个 token 位置
- $\rho_{i,t} = \pi_\theta(o_{i,t} | o_{i,<t}, I, x) / \pi_{old}(o_{i,t} | o_{i,<t}, I, x)$: importance sampling ratio, 当前 policy 和采样时 policy 的概率比
- $\hat{A}_{i,t}$: 在 token $t$ 处的 advantage estimate (通常用 response-level $A_i$ 近似)
- $\epsilon$: PPO clip ratio, 限制单次更新步长, 典型值 0.1-0.2
- $\beta$: KL penalty 系数 (paper 中 0.01)
- $\pi_{ref}$: SFT 后 frozen 的 reference policy
- $\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$: 当前 policy 和 reference policy 的 KL 散度

**Intuition**: 
- $\min(\rho \hat{A}, \text{clip}(\rho) \hat{A})$ 是 PPO 的 clipped surrogate objective, 防止 $\rho$ 过大导致训练不稳定
- $-\beta \mathbb{D}_{KL}$ 是 regularization, 防止 policy 偏离 SFT model 太远, 保留 language 能力
- $1/|o_i|$ 对 response 长度做 normalize, 避免长 response 在 loss 中占比过大

#### Geometry-aware Rewards

三种 reward function 对应不同 task:

**Box IoU Reward** (用于 detection, grounding, referring, OCR)

公式 (3): 对每个 GT box $b_j^*$, 找 IoU 最大的 predicted box:
$$\text{IoU}(b_j^*, \hat{b}_i) = \max_{\hat{b}_i \in \hat{B}} \text{IoU}(b_j^*, \hat{b}_i)$$

变量:
- $b_j^*$: 第 $j$ 个 GT box ($j \in \{1, ..., n\}$, $n$ 是 GT 数量)
- $\hat{b}_i$: 第 $i$ 个 predicted box ($i \in \{1, ..., m\}$, $m$ 是预测数量)
- $\hat{B}$: predicted box 集合

如果 matched predicted box 的 category 和 GT 匹配, $r_j = \text{IoU}$, 否则 $r_j = 0$。

公式 (4): F1-style reward
$$\text{Recall} = \frac{\sum_{j=1}^n r_j}{n}, \quad \text{Precision} = \frac{\sum_{j=1}^n r_j}{m}, \quad r^{IoU} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}$$

变量:
- $n$: GT box 数量
- $m$: predicted box 数量
- $r_j$: 第 $j$ 个 GT box 的 reward
- $\epsilon$: 防除零小常数

**Intuition**: 这个 F1 reward 非常精妙:
- Recall 项鼓励模型找到所有 GT (惩罚漏检)
- Precision 项鼓励模型不要乱预测 (惩罚 false positive)
- 用 IoU 值作 reward 而不是 binary match, 给 continuous gradient signal, 直接 fix Challenge 1 的 discrete-continuous mismatch
- 同时通过 Precision 惩罚 over-prediction, 间接 fix Challenge 2 的 duplicate 问题

**Point-in-Mask Reward** (用于 pointing-based detection, grounding, referring)

公式 (5):
$$\exists \hat{p}_i \in \hat{P}, \quad \text{s.t.} \quad \hat{p}_i \in M_j$$

变量:
- $\hat{P}$: predicted points 集合
- $\hat{p}_i$: 第 $i$ 个 predicted point
- $M_j$: 第 $j$ 个 GT box 对应的 SAM mask

如果存在 point 落在 mask 内且 category 匹配, reward = 1, 否则 0。然后同样算 F1。

**Point-in-Box Reward** (用于 GUI grounding)

Binary: point 落在 GT box 内 reward = 1, 否则 0。简单但有效。

#### GRPO 实现细节

- Data: 66K from SFT dataset (相比 22M SFT 数据, 极小)
- Hardware: 8 A100, 24 hours
- Rollout size: 8 (即 $G=8$)
- KL β: 0.01
- Batch size: 64
- 全参数更新

参考: [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

---

## 6. Benchmark Results 详解

### 6.1 COCO (Table 2)

| Type | Method | Zero-Shot | F1@0.5 | F1@0.95 | F1@mIoU |
|------|--------|-----------|--------|---------|---------|
| Closed-set | DINO-R50 | No | 68.8 | 21.1 | 55.6 |
| Closed-set | DINO-Swin-L | No | 75.6 | 25.4 | 62.1 |
| Open-set | Grounding DINO-SwinT | Yes | 69.8 | 23.0 | 56.6 |
| MLLM | Qwen2.5-VL-3B | Yes | 65.3 | 15.0 | 47.6 |
| MLLM | SEED1.5-VL | Yes | 68.2 | 15.9 | 51.4 |
| MLLM | Rex-Omni-SFT | Yes | 68.1 | 15.8 | 50.4 |
| MLLM | **Rex-Omni** | Yes | **72.0** | **15.9** | **52.9** |

**Intuition**:
- F1@0.5 (宽松 IoU 阈值): Rex-Omni 超过 DINO-R50 和 Grounding DINO, 说明 MLLM 在 "找到物体大致位置" 上已经能 beat 传统 detector
- F1@0.95 (严格 IoU): Rex-Omni 只比 DAB-DETR 略好, 说明在 "极紧 box" 上 MLLM 仍落后, 这是 discrete token 量化精度的天花板 (1000 bins → 0.1% pixel resolution)
- GRPO 提升: SFT → GRPO 在 F1@0.5 上 +3.9, 在 mIoU 上 +2.5, 非常显著

### 6.2 LVIS (Table 3) - Long-tailed

| Method | F1@0.5 | F1@mIoU |
|--------|--------|---------|
| Grounding DINO | 47.7 | 38.8 |
| SEED1.5-VL | 54.7 | 38.5 |
| Rex-Omni-SFT | 52.0 | 39.6 |
| **Rex-Omni** | **54.7** | **44.2** → 46.9 |

LVIS 有 1203 类, 远多于 COCO 的 80 类。MLLM 在这里普遍 beat Grounding DINO, 因为 LLM 的 language understanding 比 CLIP/BERT text encoder 强, 对 rare category 泛化更好。Rex-Omni 在 mIoU 上达到 SOTA, 说明 box 精度跨阈值都很强。

### 6.3 Dense Detection (Table 4) - 最能体现 GRPO 价值

| Method | Dense200 F1@0.5 | VisDrone F1@0.5 |
|--------|-----------------|-----------------|
| Grounding DINO | 36.9 | 55.2 |
| Qwen2.5-VL-3B | 0.8 | 31.5 |
| SEED1.5-VL | 76.9 | 55.9 |
| Rex-Omni-SFT | 60.2 | 55.6 |
| **Rex-Omni** | **78.4** | **61.6** |

Dense200 平均每图 91.2 个 box, VisDrone 平均 box 大小 30.7×32.4 pixels。这里 MLLM 普遍崩盘, Qwen2.5-VL-3B 在 Dense200 只有 0.8 (完全失败)。Rex-Omni SFT 只有 60.2, GRPO 后跳到 78.4, **+18.2 的提升**极其惊人。

这个提升的来源在 Section 6.1 有详细分析。

### 6.4 其他任务

- **Referring (Table 5)**: HumanRef 上 Rex-Omni F1@0.5 = 85.4, 接近 SEED1.5-VL 的 88.2
- **Visual Prompting (Table 6)**: COCO F1@0.5 = 79.1, 接近 T-Rex2 的 72.3 (实际超过, 但 T-Rex2 是 expert model)
- **Pointing (Table 7)**: COCO F1@Point = 80.5, Dense200 = 82.5, 全面 SOTA
- **GUI Grounding (Table 8)**: ScreenSpot-V2 Avg = 88.4, ScreenSpot-Pro = 36.8, 在 3B 模型里最强
- **Layout (Table 9)**: DocLayNet F1@0.5 = 89.5, 接近 DocLayout-YOLO 的 91.2
- **OCR (Table 10)**: ICDAR2015 F1@0.5 = 45.2, beat PaddleOCRv5 的 38.2
- **Spatial Pointing (Table 11)**: RefSpatial location = 54.0, 大幅超过 Gemini-2.5-Pro 的 46.96
- **Keypoint (Table 12)**: COCO F1@OKS0.5 = 44.4, AP10K = 30.1, 跨域泛化强于 X-Pose

---

## 7. In-depth Analysis: Why GRPO Works?

这是 paper 最有价值的部分, 我重点讲。

### 7.1 Training Dynamics (Figure 16)

SFT 阶段 performance steady 但 slow 改善, 最终 plateau。GRPO 阶段只用 66K data (相比 22M), 但 performance 快速 jump。

**Intuition**: SFT 已经把 "如何预测坐标" 的 latent capability 灌进模型了, 但模型不知道如何 "autoregressively 调用" 这个 capability。GRPO 通过 reward signal 教模型何时 stop、如何避免重复, 解锁了 latent capability。这点和 LLM reasoning 上的 RL 观察 (e.g. DeepSeek-R1) 一致: RL 的价值不在于学新知识, 而在于教模型用好已有知识。

### 7.2 Behavioral Correction 1: Duplicate Predictions (Table 13)

定义: 同一 coordinate value 连续出现 ≥10 次, 且总预测数 > 2× GT 数。

| Dataset | Model | F1@0.5 (原始) | F1@0.5 (去重后) | 去重提升 |
|---------|-------|--------------|-----------------|---------|
| COCO | SFT | 68.2 | 70.1 | +1.23% |
| COCO | GRPO | 72.0 | 72.6 | +0.08% |
| VisDrone | SFT | 55.6 | 62.3 | **+15.3%** |
| VisDrone | GRPO | 61.6 | 62.1 | +0.1% |

**Intuition**: SFT 模型在 dense scene 大量重复预测, 去重后 VisDrone 提升 15.3%! GRPO 模型几乎不重复 (去重只提升 0.1%)。这说明 GRPO 的 F1 优势主要来自 "不重复", 而不是 "更准"。GRPO 的 reward 函数中 Precision 项 = $\sum r_j / m$, $m$ 是预测数, 重复预测会稀释 Precision, reward 降低, 模型学到不重复。

### 7.3 Behavioral Correction 2: Large-box Predictions (Table 14)

定义: 只预测 1 个 box 且面积 > 95% 图像面积。

| Model | F1@mIoU (原始) | F1@mIoU (去除大 box) | 大 box 占比 |
|-------|---------------|---------------------|------------|
| SFT | 44.9 | 56.7 | **20.5%** |
| GRPO | 58.3 | 60.0 | 3.5% |

**Intuition**: SFT 模型在 dense scene 会 "作弊", 预测一个超大 box 盖住所有物体, 这样 IoU 和某些 GT 会部分匹配。但这是 failure mode。GRPO 后这种作弊行为几乎消失。机制: F1 reward 中 IoU 值作 reward, 大 box 和单个 GT 的 IoU 通常很低, reward 低, 模型学到不要这么做。

### 7.4 Coordinate Precision: GRPO 提升有限 (Table 15)

只在 SFT 和 GRPO 都预测正确数量 box 且都 match GT 的样本上比较:

| Dataset | SFT F1@mIoU | GRPO F1@mIoU |
|---------|-------------|--------------|
| COCO | 63.0 | 63.5 |
| LVIS | 56.6 | 56.9 |
| HumanRef | 60.0 | 61.2 |

**Intuition**: 在 "行为正确" 的样本上, GRPO 对 coordinate 精度提升只有 ~0.5-1.2。这说明 SFT 已经学会了 coordinate prediction 本身, GRPO 的核心价值是 **行为纠正**, 不是精度提升。这点对理解 GRPO 在 MLLM 检测中的作用非常重要: 不要期待 GRPO 让 box 更紧, 它让 box 数量更对、更不重复。

### 7.5 Sampling Probability 视角 (Table 16)

实验: 用 SFT 模型 high-temperature sampling 8 次, 看 "如果选最好" 能到多少。

| Dataset | SFT | GRPO | SFT-Sampling-Best | SFT-Sampling-Vote |
|---------|-----|------|-------------------|-------------------|
| COCO | 68.2 | 72.0 | 64.6 | 72.6 |
| LVIS | 60.3 | 64.3 | 56.6 | 59.8 |
| Dense200 | 60.2 | 78.4 | 38.2 | 50.6 |

**Intuition**:
- COCO 上 SFT-Sampling-Vote (72.6) 略超 GRPO (72.0), 说明 SFT 在简单任务上 latent capability 足够, GRPO 主要提升 sampling consistency
- LVIS 和 Dense200 上 SFT-Sampling-Vote 远低于 GRPO, 说明复杂任务 GRPO 不只是提升 sampling probability, 而是 **fundamentally 改变 prediction quality**。这是 GRPO 最深刻的贡献。

### 7.6 Token Efficiency (Table 17)

| Model | COCO boxes/img | tokens/img | tokens/box |
|-------|---------------|------------|------------|
| SEED1.5-VL | 4.2 | 631.0 | 148.8 |
| Rex-Omni | 5.9 | 45.3 | 7.6 |

| Model | Dense200 boxes/img | tokens/img | tokens/box |
|-------|-------------------|------------|------------|
| SEED1.5-VL | 73.1 | 5446.3 | 74.5 |
| Rex-Omni | 86.7 | 439.0 | 5.1 |

**Intuition**: Special token 设计在 dense scene 优势巨大, 12-15× 的 token 节省。这直接转化为 inference 速度, Figure 18 显示 0-29 box < 2 秒, 410-419 box > 16 秒, 线性增长。

---

## 8. 整体 Intuition 总结

让我把 Rex-Omni 的核心 insight 总结成几个 build intuition 的点:

### 8.1 MLLM 检测的核心瓶颈不是 "能不能预测坐标", 是 "能不能行为正确"

SFT 已经教会模型预测坐标 (Table 15 证明精度差不多), 但 SFT 的 teacher forcing 让模型不知道何时 stop、如何避免重复。这个 behavioral gap 是 MLLM 检测器打不过传统 detector 的真正原因, 不是精度问题。

### 8.2 GRPO 的价值是教模型 "用" 已有能力, 不是教模型 "新" 能力

22M 数据 SFT 灌入 latent capability, 66K 数据 GRPO 教模型调用。这个 22M : 66K = 333:1 的比例非常说明问题。RL 是 leverage 工具, 不是 knowledge 来源。

### 8.3 Geometry-aware reward 是 discrete-to-continuous 的桥梁

CE loss 对 pixel offset 不敏感, IoU reward 对 pixel offset 敏感。用 IoU 作 reward 等于在 token 分类任务上叠加几何先验, fix 了 Challenge 1。

### 8.4 Special token 设计是 dense scene 的 enabler

不是 luxury, 是 necessity。SEED1.5-VL 在 Dense200 用 5446 tokens/image, KV cache 和 attention 计算爆炸。Rex-Omni 439 tokens, 12× 节省, 让 MLLM 在 dense scene 实际可用。

### 8.5 Data engine 的设计哲学: 用 SOTA model 组合生成监督

Grounding engine = Qwen2.5-VL (caption) + SpaCy (phrase) + DINO-X (grounding)
Referring engine = Qwen2.5-VL (expression) + Molmo (pointing) + SAM (mask)
Pointing engine = SAM (mask) + geometry (rotated rectangle)

每个 engine 把多个 SOTA model 的 strength 组合, 自动生成大规模高质量数据。这是 "用模型训练模型" 的 self-improving loop 雏形。

---

## 9. 局限性和未来方向

Paper 自己提到的:
- Inference speed: dense scene 仍慢, 需要 quantization / distillation
- F1@0.95 上仍落后 regression-based detector, 1000-bin 量化是天花板, 未来可能需要 hierarchical quantization 或 hybrid discrete-continuous head

我认为还有几个值得探讨的点:
- GRPO 的 reward 是 GT-guided matching, 需要 GT box, 这限制了 RL 阶段只能用 labeled data。能否设计 self-reward 机制用 unlabeled data 做 RL?
- 1000 bins 对 4K+ 高分辨率图像可能不够, GUI grounding 在 ScreenSpot-Pro 上 performance 下降可能部分源于此
- Visual prompting 上还打不过 T-Rex2 (expert model), 说明 generative 范式在 feature matching 上仍有差距

---

## 参考链接

- [Rex-Omni Project Page](https://rex-omni.github.io/)
- [IDEA-Research/Rex-Omni Code](https://github.com/IDEA-Research/Rex-Omni)
- [Pix2Seq: A Language Modeling Framework for Object Detection](https://arxiv.org/abs/2109.10852)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [DINO-X](https://arxiv.org/abs/2411.14347)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
- [T-Rex2](https://arxiv.org/abs/2403.14610)
- [Molmo / PixMo](https://arxiv.org/abs/2409.17146)
- [SEED1.5-VL](https://arxiv.org/abs/2505.07062)
- [RexSeek: Referring to Any Person](https://arxiv.org/abs/2503.08507)
- [DeepSeek-VL2](https://arxiv.org/abs/2412.10302)

如果你想深入某个部分 (e.g. GRPO 的 reward 设计细节、data engine 的具体 prompt、coordinate quantization 的 error analysis), 我可以再展开。
