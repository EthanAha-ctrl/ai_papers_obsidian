---
source_pdf: EgoThink Evaluating First-Person Perspective Thinking Capability of Vision-Language
  Models.pdf
paper_sha256: f52b14647c715c4187f0f53bb90bde9e7d6f2f5d10eefc5da2a39b714076e040
processed_at: '2026-08-04T02:42:18-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EgoThink

---

## 一句话总结

**给 VLM 戴上 GoPro, 看它能不能像人一样"从自己视角"理解世界. 结果发现, 哪怕 GPT-4V 也只能拿 65 分.**

---

## 为什么要做这个 benchmark

你想想, 之前所有 VLM benchmark——MME, MMBench, LVLM-eHub——全是 third-person 视角. 就好比一个旁观者站在旁边看照片回答问题: "图里有几只猫?", "桌子什么颜色?".

可 embodied AI 和 robotics 需要的是 first-person 视角. 机器人戴着摄像头, 它看到的世界是"我手里拿着杯子", "我面前有扇门", "我该往左还是往右走". 这跟旁观者看照片是完全两码事.

已有的两个 egocentric benchmark 也很拉胯:
- **EgoVQA**: 只问 object/action, 520 条, 多选题
- **EgoTaskQA**: 只问 spatial/temporal/causal, 40k 条, 众包标注

没有一个 benchmark 系统性地问: VLM 能不能从"我"的角度 think? 这就是 EgoThink 要填的坑.

---

## 怎么造的数据

从 [Ego4D](https://ego4d-data.org/) 里抽帧. Ego4D 是 Meta 搞的大规模第一人称视频库, 3670 小时, 931 个人戴摄像头拍的, 覆盖 9 个国家 74 个地点.

抽帧规则:
- 每隔几十帧取一张, 避免连续帧太像
- 剔除模糊的、不像 egocentric 的
- 同一个视频最多保留 2 张, 保证 diversity

最后筛出来 700 张图, 来自 595 个视频.

然后 6 个标注员手工写 QA. 每人负责 2 个 dimension. 写完之后还有 3 个人交叉 review, 3 个全同意才保留. 严格 triple-review.

700 条数据量不大, 但每条都是手工精修的, 跟 MMBench 那种大规模自动生成的路线不同.

---

## 六个能力, 十二个维度

作者把"第一人称思考"拆成 6 大类, 对应人类自然会问的 6 个问题:

| 人会问什么 | Capability | 细分维度 |
|-----------|-----------|---------|
| 我周围有什么? | Object | Existence / Attribute / Affordance |
| 我在做什么? | Activity | (单维度) |
| 我在哪? | Localization | Location / Spatial Relationship |
| 周围情况怎样? | Reasoning | Counting / Comparison / Situated Reasoning |
| 接下来会发生什么? | Forecasting | (单维度) |
| 我该怎么做? | Planning | Navigation / Assistance |

从 perception 到 cognition 到 action, 一层一层递进.

**Affordance** 这个维度特别有意思. Gibson 1977 年提出 affordance theory——物体"能让你做什么". 杯子能 grasp、能 drink from, 锅能 pour into. 机器人看到物体光知道"这是杯子"没用, 得知道"我能拿它干嘛". 结果 GPT-4V 在 affordance 上只有 58 分, 说明 VLM 学到的还是 object classification, 没学到 object-action 的 coupling.

---

## 评估方法

### 为什么不用 exact match

答案都是 open-ended 的. 比如 "What is around me?" 答案可能是 "a knife and a cutting board", 模型可能回答 "knife, cutting board" 或者 "there is a knife and cutting board on the table". Exact match 直接判错, 太不公平.

### 用 GPT-4 当 judge

给 GPT-4 看 question、reference answer、model output, 让它打分:
- **0 分**: 完全错
- **0.5 分**: 部分对
- **1 分**: 完全对

每个 dimension 50 个 sample, 算平均分乘 100.

### Judge 可靠吗

作者找了 3 个人类标注员给 GPT-4V 的答案打分, 同时用 GPT-4, GPT-3.5-Turbo, Claude-2 也当 judge, 算 Pearson correlation:

- GPT-4 vs Human: **0.68**
- Claude-2 vs Human: **0.68**
- GPT-3.5-Turbo vs Human: 0.43
- 人类之间 Cohen's Kappa: 0.81

GPT-4 跟人类的一致性大约是人类之间 84% 的水平. 够用了, 但不完美. 作者也承认 Planning 维度答案很长, GPT-4 judge 难 capture 所有 detail.

---

## 评测了哪些模型

18 个开源 VLM + GPT-4V. 按 size 分:

### ~7B 梯队 (13 个)
OpenFlamingo, BLIP-2-6.7B, VideoChat, LLaVA-1.5-7B, MiniGPT-4-7B, InstructBLIP-7B, LLaMA-Adapter-7B, Otter-I-7B, PandaGPT-7B, mPLUG-Owl-7B, Video-LLaVA-7B, LLaVA-7B, ShareGPT4V-7B

### ~13B 梯队 (7 个)
InstructBLIP-13B, PandaGPT-13B, LLaVA-13B-Vicuna, BLIP-2-11B, InstructBLIP-11B, LLaVA-13B-Llama2, LLaVA-1.5-13B

这些模型的 architecture 差异很大:
- **Image encoder**: CLIP-ViT-L, EVA-CLIP-ViT-g, ImageBind, LanguageBind
- **Alignment module**: Q-Former (BLIP-2 系), Linear/MLP (LLaVA 系), Cross-attention (Flamingo 系), Early fusion (LLaMA-Adapter)
- **Trainable params**: 从 14M (LLaMA-Adapter) 到 13B (LLaVA-1.5-13B) 跨了 1000 倍
- **是否见过 egocentric data**: 只有 PandaGPT 标了 yes

---

## 核心发现

### 发现一: GPT-4V 最强, 但也只有 65.5 分

GPT-4V 各维度分数:

```
强项 (≥80):
  Attribute 82.0   Situated Reasoning 83.0   Assistance 84.0

中等 (60-80):
  Location 86.0   Navigation 64.0

弱项 (<60):
  Existence 62.0   Affordance 58.0   Activity 59.5
  Spatial 62.0     Counting 42.0     Comparison 48.0
  Forecasting 55.0
```

GPT-4V 强在 commonsense reasoning 和 long-form generation, 弱在 precise spatial/motion understanding. 这很符合直觉——web data 里大量 "这是什么"、"怎么用"的 QA, 但很少有 "我左手拿的东西比右手多几个"这种 egocentric spatial 问题.

### 发现二: Counting 是最难的

所有模型 counting 都差. GPT-4V 才 42 分. 

原因: counting 要同时干两件事——(a) 数数, (b) 理解空间参照 ("in my left hand"). Figure 6 的 case study 显示, VLM 能数出 "3 个东西", 但分不清哪只手有几个.

传统 VQA 里 counting 就难 ([How many?](https://arxiv.org/abs/1807.08590) 早就证明了), egocentric 视角让它难上加难.

### 发现三: 开源模型在某些维度超过 GPT-4V

三个 dimension 开源模型赢了:

| Dimension | 开源冠军 | 分数 | GPT-4V |
|-----------|---------|------|--------|
| Existence | ShareGPT4V-7B | 67.0 | 62.0 |
| Location | BLIP-2-11B | **90.0** | 86.0 |
| Spatial | BLIP-2-11B | **66.0** | 62.0 |

**Existence**: ShareGPT4V-7B 赢. Figure 4 显示 GPT-4V 在 "右手里的筷子"这种小手持物体检测上翻车, InstructBLIP-11B 和 LLaVA-7B 反而对. 可能是 GPT-4V 的 RLHF 让它对小物体 detection 变保守了.

**Location 和 Spatial**: BLIP-2-11B 赢. 这很有意思. BLIP-2 用 FlanT5-XXL 当 LLM backbone, FlanT5 的 instruction tuning 包含大量 QA task, scene classification 能力强. 而 GPT-4V 经过 RLHF 后可能丢失了 precise spatial 信息.

特别是 Spatial relationship——egocentric 的"我的左边/右边/前面/后面"——BLIP-2-11B 达到 66 分超 GPT-4V 4 分. 说明 Q-Former 保留的 spatial tokens 可能比 GPT-4V 经过 RLHF 压缩后的 representation 更好用.

### 发现四: Planning 里 Assistance 远强于 Navigation

GPT-4V: Assistance 84.0, Navigation 64.0. 差 20 分.

Assistance 是 procedural knowledge, "how to do X"——煮鸡蛋步骤、修自行车流程. 这跟 common sense 关系大, 不太需要 precise spatial understanding.

Navigation 要 spatial reasoning + step-by-step planning + scene understanding, 三者都要. Figure 7 bottom case 显示 VLM 的 navigation answer 要么太简略缺关键 detail, 要么忽视图像里的重要信息.

### 发现五: Forecasting 也难

GPT-4V 才 55 分. Figure 7 top case: VLM 把 glove 认成 cloth, 然后 forecasting 直接错了. 经典 error propagation——perception 错了, reasoning 跟着错.

### 发现六: Scaling LLM 只在 trainable 时管用

比较 7B vs 13B:

| Model Family | 7B | 13B | 提升 | LLM 可训? |
|--------------|-----|-----|------|----------|
| LLaVA | 49.6 | 55.1 | +5.5 | ✅ full tune |
| LLaVA-1.5 | 39.0 | 55.3 | **+16.3** | ✅ full tune |
| PandaGPT | 46.2 | 43.1 | -3.1 | ❌ LoRA |
| InstructBLIP | 42.4 | 42.8 | +0.4 | ❌ frozen |

**关键 insight**: Scaling 只在 LLM trainable 时 work. PandaGPT 和 InstructBLIP freeze 了 LLM, 13B 版几乎没提升. LLaVA 系列 full-tune LLM, scaling 帮助大.

这跟 Chinchilla scaling law 一致——compute 要花在 trainable parameters 上, 而不是 frozen parameters. 你把 LLM 从 7B 变 13B, 但 LLM 是 frozen 的, 那多的参数完全没被 update, 自然没 benefit.

### 发现七: Instruction tuning 不一定 help

BLIP-2-11B vs InstructBLIP-11B 唯一区别是 instruction tuning + instruction-aware tokens:
- BLIP-2-11B: 49.6
- InstructBLIP-11B: 51.5
- 提升: +1.9 (marginal)

原因: InstructBLIP 的 instruction tuning data 来自 specific downstream tasks (VQA, GQA 等), 跟 first-person egocentric data distribution 不 match. 通用 instruction tuning 不一定 help specialized domain.

### 发现八: SoM 反而有害

作者试了 [Set-of-Mark](https://arxiv.org/abs/2310.11441)——给 GPT-4V 加 segmentation mask, 测试 visual grounding 信息有没有帮助:

| Method | Existence | Attribute |
|--------|-----------|-----------|
| GPT-4V | 62.0 | 82.0 |
| GPT-4V + SoM | **36.0** (-26) | **62.0** (-20) |

**反直觉**: SoM 实际降低了 performance!

作者解释: mask 遮挡了原图的 color、boundary 信息.

我的 interpretation: GPT-4V 的 visual encoder trained on clean image distribution, 加 mask 改变了 input distribution, 引起 distribution shift. 类似 adversarial perturbation 的效果.

这说明不能简单往图像上加 structured annotation, 要用更巧妙的方式 inject visual grounding 信息.

---

## 为什么 VLM 在 egocentric 上普遍差

根本原因: **VLM 是 web-data trained, 学的是 third-person "observer" 视角**.

Web image 的 caption 多是 "A man is cooking in the kitchen", "There is a cup on the table"——观察者视角的描述. Egocentric 视角需要的是 "I am cooking", "There is a cup in my hand"——actor 视角的描述.

更进一步, egocentric reasoning 需要 self-model:
- "我是哪个 agent"
- "我的手在哪"
- "我面向哪边"
- "这个物体相对我的位置"

当前 VLM 没有 self-model, 只能从图像 superficial cue 推断. 这跟 developmental robotics 的 insight 一致——robot 需要 self-awareness 才能做 egocentric reasoning.

---

## 对 embodied AI 的启示

EgoThink 的结果直接打了 [PaLM-E](https://palm-e.github.io/), [RT-2](https://robotics-transformer2.github.io/), [VoxPoser](https://voxposer.github.io/), [SayCan](https://say-can.github.io/) 这类工作的脸——它们都假设 VLM 有 first-person understanding, 但实际上 VLM 在 Affordance (58)、Spatial (62)、Forecasting (55) 上都很弱.

这意味着 RT-2 类方法的 bottleneck 可能在 VLM 而非 control policy. 如果 VLM 连 "我手里拿的是什么"都认不准, 那它输出的 action token 可靠性存疑.

可能解决方向:
1. **Egocentric pre-training**: 用 Ego4D 大规模 first-person pre-training
2. **Body-aware tokens**: 在 VLM 中 inject "self" token, 显式表示 agent 状态
3. **3D representation**: 用 NeRF 或 3D Gaussian Splatting 提供空间结构
4. **Embodied RL fine-tuning**: 在 simulator 中 fine-tune VLM, 从 action consequence 中学习
5. **Multimodal beyond vision**: 加 proprioception, tactile, audio

---

## 我的评价

### 强项
- 填了 first-person VLM evaluation 的空白
- Taxonomy 设计合理, 从 perception 到 action 全链路
- Triple-review annotation 保证数据质量
- 18 个模型横向对比, ablation 充分
- GPT-4 judge 跟 human correlation 0.68, 可靠性够用

### 弱项
- **700 条太小**. 每 dimension 才 50 sample, standard error 约 7%, 差 10 分以内可能不 significant. 相比 MMBench 2974 条, 这个 size 只够 pilot study.
- **只有 image 没有 video**. 真正 egocentric reasoning 是 sequential, 单 frame 信息有限. Forecasting 和 Planning 没有 temporal context 根本测不准.
- **GPT-4 judge 的 cost 和 reproducibility**. 每次 eval 要调 API, 而且 GPT-4 本身会更新, 不同时间评测结果不可复现.
- **Planning 评估仍粗糙**. Long-form answer 用 GPT-4 judge 0.68 的 correlation 在 Planning dimension 上可能更低, 作者自己都承认这点.
- **Video VLM 没发挥优势**. VideoChat, Video-LLaVA, Otter 被评了, 但 benchmark 是 image-only, 没法 reveal video VLM 的真正强项.

---

## 总结

EgoThink 是 VLM evaluation 从 third-person 走向 first-person 的第一步. 它揭示了一个残酷现实: 即使 SOTA VLM 在第一人称视角下也只拿 65 分, 在 spatial reasoning, counting, forecasting 上更是只有 40-55 分.

当前 VLM 的训练范式——web data + third-person caption——根本不支持真正的 embodied understanding. 要让 VLM 能 think from first-person, 我们需要 fundamentally 不同的 architecture, training data, evaluation methodology.

EgoThink 是个 start, 但远不是 end. 下一代 egocentric benchmark 应该 video-based, interactive, multi-modal, 在 simulator 里 ground truth, 而不是靠人标注 QA.

相关 links:
- [EgoThink arxiv](https://arxiv.org/abs/2312.12973)
- [Ego4D dataset](https://ego4d-data.org/)
- [Set-of-Mark paper](https://arxiv.org/abs/2310.11441)
- [LLM-as-judge (MT-Bench)](https://arxiv.org/abs/2306.05685)
- [PaLM-E](https://palm-e.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)
- [VoxPoser](https://voxposer.github.io/)
- [SayCan](https://say-can.github.io/)
- [MMBench](https://github.com/open-compass/MMBench)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Model)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [InstructBLIP](https://arxiv.org/abs/2307.08504)

---

# EgoThink 深度解析: 从第一人称视角评估 VLM 的"思考"能力

Andrej, 这篇paper触及了一个非常有意思且被严重低估的问题: 当前的 VLM 在 egocentric视角下到底能不能"think"? 整个领域的evaluation suite几乎都是 third-person perspective (MME, MMBench, LVLM-eHub), 而 embodied AI 和 robotics 真正需要的恰好是 first-person 的空间推理、affordance理解、planning. 这篇paper填补了这个空白, 但也暴露了 VLM 在第一人称下的根本性缺陷.

Reference: [arxiv link](https://arxiv.org/abs/2312.12973) | [Ego4D](https://ego4d-data.org/) | [Project page](https://github.com/EGO-Think/EgoThink)

---

## 1. 核心Taxonomy的Intuition

作者把第一人称"思考"分解成6个capability, 12个dimension, 对应人类自然的自我中心问题序列:

| Question | Capability | Dimensions |
|----------|-----------|------------|
| What is around me? | Object | Existence, Attribute, Affordance |
| What am I doing? | Activity | (single) |
| Where am I? | Localization | Location, Spatial Relationship |
| What about the situation? | Reasoning | Counting, Comparison, Situated Reasoning |
| What will happen to me? | Forecasting | (single) |
| How will I do? | Planning | Navigation, Assistance |

这个taxonomy的优雅之处在于它从perception → cognition → action的层级递进. Object和Activity是perception层, Localization和Reasoning是cognition层, Forecasting和Planning是action层. 从机器人学的视角看, 这恰好对应 sensing → state estimation → task planning → motion planning 的pipeline.

特别值得注意的是 **Affordance** 这个dimension. Gibson 在1977年提出 affordance theory ([The Theory of Affordances](https://en.wikipedia.org/wiki/Affordance)), 强调物体"可被提供的action可能性". 这是 robotics 的核心——机器人看到杯子要知道"可grasp、可pour into", 而仅仅识别出"这是杯子"是不够的. paper里GPT-4V affordance 只有58.0, 远低于 attribute 的82.0, 说明VLM学到的还是object classification的视觉表征, 没有真正学到 object-action coupling.

---

## 2. 数据构造 Pipeline 解析

### 2.1 Source Data: Ego4D
[Ego4D](https://ego4d-data.org/) (Grauman et al. CVPR 2022) 是Meta主导的huge-scale egocentric video dataset:
- 3,670 hours video
- 931 unique camera wearers
- 74 global locations across 9 countries

这保证了scene diversity. EgoThink从中抽取images:
1. 提取所有frames → raw image dataset
2. 每隔几十帧sample一次 (避免redundancy)
3. 严格筛选: 排除不清晰、无egocentric特征的image
4. 同一video最多保留2张image (保证diversity)

最终得到 **700 images from 595 videos**, 覆盖kitchen、workshop、outdoor、laboratory等scene类型 (Figure 3).

### 2.2 Annotation Protocol
- 6 annotators, 每人负责2个dimensions
- 一旦image被选, 从候选池中移除 (no repetition)
- 3个额外annotators交叉review
- 全部同意才保留

这个 triple-review 的protocol很严格, 但700的size确实偏小. Table 4显示每个dimension至少50 instances, 这是统计reliability的下限.

### 2.3 Question Type Analysis
Table 4的 TypeQ (question types)很有意思:
- Existence: 2 (yes/no, what)
- Activity: 2 (what, how)
- Spatial Relationship: 5 (where, left/right, near/far, front/behind, etc.)
- Situated Reasoning: 5 (why, what if, etc.)
- Forecasting: 6 (what will, when, etc.)

Question type的diversity反映了任务的cognitive complexity. Forecasting 和 Situated Reasoning 用了最多question type, 对应 highest reasoning demand.

Answer length (LenA) 也有讲究:
- 大多数dimension: 1.6-3 words (short answer)
- Navigation: 18.44 words
- Assistance: 19.12 words

这意味着 planning 类任务需要long-form generation, 评估难度自然上升——short answer可以用exact match, long answer需要semantic comparison. 这也是为什么用GPT-4作judge的motivation.

---

## 3. Evaluation Methodology 深度解析

### 3.1 Single-Answer Grading

作者采用 LLM-as-a-judge 范式 (借鉴 [Zheng et al. 2023](https://arxiv.org/abs/2306.05685) 的MT-Bench). 给定一个question q, model output m, reference answer r, GPT-4 evaluator 输出一个 score:

$$s = \text{GPT-4}(\text{prompt}(q, r, m)) \in \{0, 0.5, 1\}$$

- **s = 0**: 完全错误 (wrong)
- **s = 0.5**: 部分正确 (partially correct)
- **s = 1**: 完全正确 (correct)

每个dimension的final score:

$$\text{Score}_{dim} = \frac{1}{N_{dim}} \sum_{i=1}^{N_{dim}} s_i \times 100$$

其中 $N_{dim}$ 是该dimension的sample数 (通常50或100), $s_i$ 是第i个sample的grading score.

这个 3-level grading 比 binary {0,1} 更nuanced, 能capture "答案方向对但细节错"的情况.

### 3.2 Evaluator Agreement 验证

为了验证 GPT-4 evaluator 的可靠性, 作者做了 human correlation study:

| Evaluator Pair | Pearson r |
|----------------|-----------|
| GPT-4 vs Human | **0.68** |
| GPT-3.5-Turbo vs Human | 0.43 |
| Claude-2 vs Human | 0.68 |
| GPT-4 vs Claude-2 | 0.80 |
| GPT-4 vs GPT-3.5-Turbo | 0.524 |
| Claude-2 vs GPT-3.5-Turbo | 0.536 |

Pearson correlation coefficient 公式:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

其中:
- $x_i, y_i$: 第i个sample上两个evaluator给的score
- $\bar{x}, \bar{y}$: 两个evaluator score的均值
- $n$: sample总数
- $r \in [-1, 1]$, 越接近1表示正相关越强

0.68 表示 strong positive correlation, 但不是perfect. 人类之间的 Cohen's Kappa:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

其中:
- $p_o$: observed agreement proportion (observed)
- $p_e$: chance agreement proportion (expected)
- $\kappa = 0.81$ 表示 almost perfect agreement (>0.81是 Landis & Koch 标准的 "almost perfect")

Insight: GPT-4 evaluator 大约达到 human annotator 之间 84% 的reliability, 这足以支持large-scale evaluation. 但 paper 也承认 Planning dimension的long-form answer评估仍是难题, 因为details难以capture.

### 3.3 与其他 VLM Benchmark 的对比 (Table 1)

| Benchmark | Capability | Perspective | Evaluator | Size |
|-----------|-----------|-------------|-----------|------|
| VL-CheckList | Object/Attr/Rel | Third | Accuracy | 410k |
| MME | General | Third | Accuracy | 2,194 |
| MMBench | General | Third | LLMs | 2,974 |
| EgoTaskQA | Spatial/Temp/Causal | First | Crowdsourcing | 40k |
| EgoVQA | Object/Action/Person | Third/First | Accuracy | 520 |
| **EgoThink** | **First-Person Thinking** | **First** | **LLMs** | **700** |

关键 difference:
- 其他egocentric benchmark 用 multiple-choice (accuracy), EgoThink 用 open-ended (semantic grading)
- 其他只覆盖 narrow capability (EgoVQA: object/action; EgoTaskQA: spatial/temporal/causal), EgoThink 覆盖 perception → cognition → action 全链路
- EgoThink 的 size 700 较小, 但每个 sample 都是 manually crafted 高质量QA, 适合精细评估而非大规模训练

---

## 4. 模型Architecture深度对比

Table 2 列出 18个VLMs 的architecture details. 这里的关键variation是:

### 4.1 Image Encoder维度
- **CLIP-ViT-L** (LLaVA, MiniGPT-4, mPLUG-Owl, OpenFlamingo, Otter): 标准CLIP ViT-Large, 224或336px input
- **EVA-CLIP-ViT-g** (BLIP-2, InstructBLIP): 更强的EVA-CLIP, giant version
- **ImageBind** (PandaGPT): Meta的多模态binding encoder, 可同时handle image/video/audio/text
- **LanguageBind** (Video-LLaVA): 统一image/video到同一feature space
- **BLIP2-VE** (VideoChat, MiniGPT-4): 复用BLIP-2的visual encoder

### 4.2 Alignment Module维度
这是 VLM 设计的 core differentiation:

1. **Q-Former** (BLIP-2, InstructBLIP, VideoChat): 
   - 使用 learnable queries 通过 cross-attention 提取 visual features
   - 把 image 压缩成 fixed number of tokens (通常32或64)
   - 公式: $Q_{out} = \text{softmax}(QK^T/\sqrt{d}) V$, where $Q$ 来自 learnable queries, $K,V$ 来自 image features

2. **Linear/MLP** (LLaVA, LLaVA-1.5, MiniGPT-4, Video-LLaVA):
   - 直接 projection: $Z = W \cdot V$, 其中 $V$ 是 image features, $W$ 是 projection matrix
   - LLaVA用linear, LLaVA-1.5升级成2-layer MLP (GELU activation)
   - 保留全部image tokens (通常256个), information loss更小

3. **Attention** (OpenFlamingo, Otter, mPLUG-Owl):
   - Cross-attention layers inserted between frozen LLM layers
   - 类似Flamingo的 design, 允许 few-shot in-context learning

4. **Early Fusion** (LLaMA-Adapter):
   - Visual tokens 在 LLM early layers 就 fuse 进去
   - 用 adapter (少量参数) 实现efficient tuning

5. **Linear + LLM LoRA** (PandaGPT):
   - Linear projection + LoRA on LLM
   - Trainable params极少 (52M for 13B)

### 4.3 Trainable Parameters (TTP) 关键insight

Table 2 中 TTP (Total Trainable Parameters) 是关键 ablation 维度:

| Model | TTP | Avg Score |
|-------|-----|-----------|
| BLIP-2-6.7B | 108M | 28.1 |
| LLaMA-Adapter-7B | 14M | 42.5 |
| MiniGPT-4-7B | 23M | 40.6 |
| PandaGPT-7B | 38M | 46.2 |
| InstructBLIP-7B | 189M | 42.4 |
| OpenFlamingo-7B | 1.4B | 27.2 |
| Otter-7B | 1.4B | 45.3 |
| LLaVA-1.5-7B | 6.8B | 39.0 |
| LLaVA-7B | 6.7B | 49.6 |
| LLaVA-1.5-13B | **13.0B** | **55.3** |
| LLaVA-13B-Llama2 | 13.0B | 55.1 |

**关键观察**: TTP 与 score 大致正相关, 但不是单调的:
- LLaVA-1.5-13B (13B trainable) 最高 55.3
- LLaVA-7B (6.7B trainable) 49.6
- 但 OpenFlamingo (1.4B trainable) 只有 27.2, 因为它 freeze 了 LLM, 只 train cross-attention

**核心insight**: 仅仅 scaling LLM 不够, 必须 unfreeze LLM 让 instruction tuning 影响 LLM weights. 这就是为什么 LLaVA 系列受益于 scaling, 而 InstructBLIP 和 PandaGPT 不受益——它们 freeze 或只 LoRA-tune LLM.

### 4.4 数据规模维度

- **Pre-training data**: 从 5M (MiniGPT-4) 到 2.8B (Otter, MIMIC-IT) 不等
- **Instruction tuning data**: 从 3.5k (MiniGPT-4) 到 2.8B (Otter) 不等
- **EgoData**: 只有 PandaGPT 用了 egocentric data training
- **Video data**: OpenFlamingo, VideoChat, Otter, PandaGPT, Video-LLaVA 用了

Insight: 大多数 VLM 训练时几乎没用 egocentric data, 这解释了它们在 EgoThink 上普遍表现差的原因. PandaGPT 是唯一标记 EgoData=√ 的, 但它score也不高 (46.2 for 7B), 说明仅仅有egocentric data还不够, 训练objective也要适配.

---

## 5. Experimental Results 深度分析

### 5.1 Overall Results (Table 3)

GPT-4V 的 per-dimension breakdown:
```
Exist: 62.0  Attr: 82.0  Afford: 58.0  Activity: 59.5
Loc: 86.0    Spatial: 62.0  Count: 42.0  Compar: 48.0
Situated: 83.0  Forecast: 55.0  Nav: 64.0  Assist: 84.0
Average: 65.5
```

观察:
1. **强项** (≥80): Attribute, Situated Reasoning, Assistance
2. **中等** (60-80): Location, Navigation
3. **弱项** (<60): Existence, Affordance, Activity, Spatial, Counting, Comparison, Forecasting

GPT-4V 的强项恰好是需要common sense reasoning或long-form generation的能力, 弱项是需要precise spatial/motion understanding的能力. 这反映了 VLM 训练 data distribution 的 bias: web data 多是 third-person 场景描述, 少有 egocentric 的 spatial relationship.

### 5.2 Open-Source VLM 的 best per dimension

| Dimension | Best Open-Source Model | Score | Gap to GPT-4V |
|-----------|----------------------|-------|---------------|
| Existence | ShareGPT4V-7B | 67.0 | +5.0 ✓ |
| Attribute | ShareGPT4V-7B | 75.0 | -7.0 |
| Affordance | LLaVA-1.5-7B | 54.0 | -4.0 |
| Activity | ShareGPT4V-7B | 55.5 | -4.0 |
| Location | BLIP-2-11B | **90.0** | +4.0 ✓ |
| Spatial | BLIP-2-11B | **66.0** | +4.0 ✓ |
| Counting | GPT-4V | 42.0 | (best) |
| Comparison | LLaVA-1.5-13B | 56.0 | +8.0 ✓ |
| Situated | InstructBLIP-11B | 73.0 | -10.0 |
| Forecasting | InstructBLIP-11B | 53.0 | -2.0 |
| Navigation | GPT-4V | 64.0 | (best, but LLaVA-13B-Llama2 close 49) |
| Assistance | GPT-4V | 84.0 | (best) |

**Three dimensions where open-source beats GPT-4V**:
1. **Existence**: ShareGPT4V-7B (67.0) > GPT-4V (62.0) — 这是 surprising, 因为GPT-4V应该object detection更强. 但 Figure 4 case study 显示 GPT-4V 在 handed object detection (chopsticks in right hand) 失败, 而 InstructBLIP-11B 和 LLaVA-7B 成功. 推测 GPT-4V 的 RLHF 可能 cause it to refuse 或 hedge on small object detection.

2. **Location**: BLIP-2-11B (90.0) > GPT-4V (86.0) — FlanT5-XXL 作为 LLM backbone 在scene classification上表现强, 可能因为 FlanT5 的 instruction tuning 包含大量 question answering task.

3. **Spatial**: BLIP-2-11B (66.0) > GPT-4V (62.0) — 这是关键. Spatial relationship 在 egocentric 视角下需要"以自己为中心"的 left/right/front/behind 推理. BLIP-2 的 Q-Former 可能保留了更多 spatial tokens, 而 GPT-4V 经过 RLHF 后可能 loss 了 precise spatial 信息.

### 5.3 Counting 是 hardest dimension

GPT-4V counting 只有 42.0, 所有模型都差. 原因分析:
- Counting 同时需要 (a) object detection, (b) spatial reference understanding ("in my left hand")
- Figure 6 top case: VLMs 能 count 但不能理解 "I holding" 这个 relative position
- 传统 VLM 的 counting 问题 ([How many?](https://arxiv.org/abs/1807.08590)) 已经被证明难, egocentric 让它更难

可能 solution: 引入 explicit object detection + counting head, 而非纯 VLM.

### 5.4 Forecasting 也难

GPT-4V 只有 55.0. Figure 7 top case 显示 VLMs 把 glove 识别成 cloth, 然后 forecasting 就错了. 这是 classic error propagation: perception error → reasoning error. 解决方案可能是 uncertainty-aware forecasting, 让模型 express "I'm not sure about the object" 而非强行 predict.

### 5.5 Planning: Assistance 比 Navigation 强

GPT-4V: Assistance 84.0, Navigation 64.0. 差距 20 points.

原因:
- Assistance 是 procedural knowledge ("how to do X"), 不需要 precise spatial understanding, 更多是 common sense
- Navigation 需要 spatial reasoning + step-by-step planning + scene understanding

Figure 7 bottom case: VLMs 的 navigation answer lack crucial details 或 overlook important image information. 这反映了 VLM 的 hallucination 问题在 planning 任务上更严重.

---

## 6. Ablation Studies 深度

### 6.1 LLM Size Scaling (Figure 8 top)

比较 4 个 model families 的 7B vs 13B:

| Model Family | 7B score | 13B score | Δ | LLM frozen? |
|--------------|----------|-----------|---|-------------|
| LLaVA | 49.6 | 55.1 (Llama2) | +5.5 | ❌ (full tune) |
| LLaVA-1.5 | 39.0 | 55.3 | +16.3 | ❌ (full tune) |
| PandaGPT | 46.2 | 43.1 | -3.1 | ✓ (LoRA) |
| InstructBLIP | 42.4 | 42.8 | +0.4 | ✓ (frozen) |

**Critical insight**: Scaling 只在 LLM trainable 时 work. PandaGPT 和 InstructBLIP freeze LLM, 所以 13B 版本几乎无提升. LLaVA 系列 full-tune LLM, 所以 scaling 帮助大.

这跟 [Chinchilla](https://arxiv.org/abs/2203.15556) 的 scaling law 一致: compute 要花在 trainable parameters 上, 而不是 frozen parameters.

### 6.2 Instruction Tuning (Figure 8 bottom)

对比 BLIP-2-11B vs InstructBLIP-11B (唯一区别是 instruction tuning + instruction-aware tokens):
- BLIP-2-11B: 49.6
- InstructBLIP-11B: 51.5
- Δ: +1.9

提升 marginal. 原因: InstructBLIP 的 instruction tuning data 来自 specific downstream tasks (VQA, GQA, etc.), 与 first-person egocentric data distribution mismatch.

**Insight**: Instruction tuning 的效果取决于 data distribution match. 通用 instruction tuning 不一定 help specialized domain (egocentric).

### 6.3 Set-of-Mark (SoM) 实验

作者用 [Set-of-Mark prompting](https://arxiv.org/abs/2310.11441) 给 GPT-4V 加 segmentation masks, 测试 visual grounding 信息是否有帮助.

| Method | Existence | Attribute |
|--------|-----------|-----------|
| GPT-4V | 62.0 | 82.0 |
| GPT-4V w/ SoM | **36.0** (-26) | **62.0** (-20) |

**反直觉结果**: SoM 实际降低了 performance!

作者的解释: SoM 的 marks 和 masks 遮挡了 original image 的 color、boundary 等信息, 导致 model 误判.

我的 interpretation: 这反映了 GPT-4V 的 visual encoder 已经 trained 了 clean image distribution, 加入 mask 改变了 input distribution, 引起 distribution shift. 类似 adversarial perturbation 的效果.

**未来方向**: 如何在不破坏 original image 信息的前提下 inject structured visual information? 可能的方案:
- 用 text 旁注 而非 visual overlay
- 用 multi-view input (original + annotated)
- 用 hierarchical attention 让 model 自行 decide 何时 use annotation

---

## 7. 与Broader Research的Connection

### 7.1 Embodied AI方向
- [PaLM-E](https://palm-e.github.io/) (Google): 直接把 VLM 嵌入 robot control loop
- [RT-2](https://robotics-transformer2.github.io/): VLM → action token
- [VoxPoser](https://voxposer.github.io/): VLM 生成 3D value map for manipulation
- [SayCan](https://say-can.github.io/): LLM + affordance grounding

这些工作都假设 VLM 有 first-person understanding, 但 EgoThink 显示当前 VLM 在这个 capability 上很弱, 这是 embodied AI 的 bottleneck.

### 7.2 Egocentric Vision
- [Ego4D](https://ego4d-data.org/): massive scale egocentric video
- [EPIC-KITCHENS](https://epic-kitchens.github.io/): kitchen activities
- [EgoObjects](https://research.facebook.com/publications/egoobjects-a-large-scale-egocentric-dataset-for-fine-grained-object-understanding/): egocentric object understanding

这些 dataset 主要用于 perception task (detection, classification), EgoThink 把它们引申到 cognition + planning.

### 7.3 VLM Evaluation
- [MMBench](https://opencompass.org.cn/leaderboard-mmbench): comprehensive 3rd-person VLM eval
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Model): 14 subtask, yes/no format
- [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench): 包括 sequential action understanding
- [MMBench-Video](https://arxiv.org/abs/2310.01810): video VLM evaluation

EgoThink 在这个谱系中填补 first-person gap.

### 7.4 LLM-as-a-Judge
- [MT-Bench](https://arxiv.org/abs/2306.05685): GPT-4 judge for LLM
- [LLM-Eval](https://github.com/LLM-Tuning-Safety/HF_DPO_LLM_Eval): open-source eval framework
- [JudgeBench](https://arxiv.org/abs/2410.12667): evaluating judges themselves

EgoThink 的 GPT-4 judge 与 human correlation 0.68, 在 LLM-as-judge 范式下属于 acceptable but not excellent. 未来可能需要 task-specific judge training.

---

## 8. Critical Analysis & My Take

### 8.1 优势
1. **填补空白**: first first-person comprehensive VLM benchmark
2. **Taxonomy 合理**: 6 capabilities 覆盖 perception → cognition → action
3. **Triple-review annotation**: 高质量data
4. **Comprehensive model coverage**: 18 open-source + 1 API
5. **GPT-4 judge validation**: 与 human correlation 0.68

### 8.2 局限
1. **Size 700 太小**: 相比 MMBench (2974), MME (2194), 这个size的variance大. 每个 dimension 50 sample, standard error 约 √(p(1-p)/50) ≈ 7%, 这意味着 score 差 10 以内可能不 significant.

2. **只有 images, 没有 video**: 真正的 egocentric reasoning 是 sequential, 单 frame 信息有限. Ego4D 本身是 video, 抽 frames 丢失 temporal context. 这对 Forecasting 和 Planning 尤其致命.

3. **GPT-4 judge 的 cost 和 reproducibility**: 每次 evaluation 需要 GPT-4 API call, 不是所有 researcher 都能 afford. 而且 GPT-4 本身会更新, 不同时间评测结果可能不同.

4. **Planning 评估的 difficulty**: 作者承认 Planning 的 long-form answer 评估难. 0.68 的 Pearson 在 Planning dimension 上可能更低.

5. **缺乏 video VLM 的深度评估**: 虽然 VideoChat, Video-LLaVA, Otter 被评估, 但 benchmark 本身是 image-only, 没法 reveal video VLM 的真正优势.

### 8.3 我的 Intuition

EgoThink 揭示了 VLM 的一个 fundamental limitation: 它们是 web-data trained, 本质上理解 third-person "observer" 视角. Egocentric 视角需要的不仅是 visual perception, 而是 embodied understanding — 知道"我是哪个 agent", "我的手在哪", "我面向哪边".

这跟 developmental robotics 的 insight 一致: robot 需要 self-model 才能进行 egocentric reasoning. 当前 VLM 缺少这个 self-model, 只能从图像 superficial cue 推断.

可能 future direction:
1. **Egocentric pre-training**: 用 Ego4D 等做大规模 first-person pre-training
2. **Body-aware tokens**: 在 VLM 中 inject "self" token, 让模型显式表示自己的状态
3. **3D representation**: 用 NeRF 或 3D Gaussian Splatting 提供空间结构信息
4. **Active perception**: VLM 应该能主动 query 不同视角, 而非被动接收 single frame
5. **Embodied RL fine-tuning**: 在 simulator 中 fine-tune VLM, 让它学会从 action consequence 中 learn

---

## 9. 数据点速查表

### 9.1 Top performers per dimension

| Dimension | Top Model | Score | Top Open-Source | Score |
|-----------|-----------|-------|-----------------|-------|
| Existence | InstructBLIP-11B | 74.0 | ShareGPT4V-7B | 67.0 |
| Attribute | GPT-4V | 82.0 | ShareGPT4V-7B | 75.0 |
| Affordance | GPT-4V | 58.0 | LLaVA-1.5-7B | 54.0 |
| Activity | GPT-4V | 59.5 | ShareGPT4V-7B | 55.5 |
| Location | BLIP-2-11B | 90.0 | (same) | 90.0 |
| Spatial | BLIP-2-11B | 66.0 | (same) | 66.0 |
| Counting | GPT-4V | 42.0 | Otter-I-7B | 39.0 |
| Comparison | GPT-4V | 48.0 | LLaVA-1.5-13B | 56.0 |
| Situated | GPT-4V | 83.0 | InstructBLIP-11B | 73.0 |
| Forecasting | GPT-4V | 55.0 | InstructBLIP-11B | 53.0 |
| Navigation | GPT-4V | 64.0 | LLaVA-13B-Llama2 | 49.0 |
| Assistance | GPT-4V | 84.0 | LLaVA-13B-Llama2 | 71.0 |
| **Average** | **GPT-4V** | **65.5** | **LLaVA-13B-Llama2** | **55.1** |

### 9.2 Architecture Comparison Key Models

| Model | Encoder | LLM | Align | TTP | Avg |
|-------|---------|-----|-------|-----|-----|
| GPT-4V | ? | ? | ? | ? | 65.5 |
| BLIP-2-11B | EVA-CLIP-g | FlanT5-XXL | Q-Former | 108M | 49.6 |
| InstructBLIP-11B | EVA-CLIP-g | FlanT5-XXL | Q-Former | 189M | 51.5 |
| LLaVA-13B-Llama2 | CLIP-ViT-L | Llama2-13B | Linear | 13.0B | 55.1 |
| LLaVA-1.5-13B | CLIP-ViT-L-336 | Llama2-13B | MLP | 13.0B | 55.3 |
| ShareGPT4V-7B | CLIP-ViT-L-336 | Vicuna-7B | MLP | 6.7B | 51.9 |
| PandaGPT-13B | ImageBind | Vicuna-13B | Linear+LoRA | 52M | 43.1 |

### 9.3 Scaling Effect Summary

公式化的 scaling effect:

$$\Delta_{\text{score}} = f(\Delta_{\text{TTP}}, \Delta_{\text{LLM size}})$$

当 LLM frozen: $\Delta_{\text{score}} \approx 0$ 即使 $\Delta_{\text{LLM size}} > 0$
当 LLM trainable: $\Delta_{\text{score}} \propto \Delta_{\text{LLM size}}$

这与 [Kaplan scaling law](https://arxiv.org/abs/2001.08361) 的 trainable-parameters-centric 视角一致: meaningful compute 必须花在可更新参数上.

---

## 10. 延伸思考

### 10.1 Egocentric vs Allocentric 在 neuroscience 的对应
[Klatzky 1998](https://link.springer.com/chapter/10.1007/3-540-69342-4_1) 定义:
- **Egocentric**: 以 self 为 center 的 spatial representation (上/下/左/右/front/behind 相对于自己)
- **Allocentric**: 以 world 为 center 的 spatial representation (用 world landmark)

人类 hippocampus 同时 maintain 两种 representation. 当前 VLM 主要学 allocentric (因为 web image 描述是 observer-based), 缺 egocentric. 这与 paper 中 Spatial dimension VLM 表现差的现象吻合.

### 10.2 Affordance 的哲学
[Gibson 1977](https://en.wikipedia.org/wiki/Affordance): "The affordances of the environment are what it offers the animal, what it provides or furnishes, either for good or ill."

Affordance 不是 object 的 intrinsic property, 而是 object-animal relationship. 同一把椅子对人提供"sit" affordance, 对蚂蚁不提供. VLM 学 affordance 必须理解 agent context, 这对纯视觉 VLM 是根本 challenge.

### 10.3 Robot Learning 的 Implication
RT-2 用 VLM 直接输出 action token, 假设 VLM 有 manipulation 所需的 visual understanding. 但 EgoThink 显示 VLM 在 Affordance (58)、Spatial (62) 都弱, 这意味着 RT-2 类方法的 bottleneck 可能在 VLM 而非 control. 解决方案:
- VLM → affordance head fine-tuning
- VLM + 3D representation (e.g., [Point-E](https://arxiv.org/abs/2212.08751), [3D-LLM](https://arxiv.org/abs/2307.12981))
- VLM + tactile/proprioceptive input (multimodal beyond vision)

### 10.4 与 RL 的关系
Reinforcement learning 的 state estimation 本质是 egocentric. EgoThink 可以作为 RL agent 的 evaluation metric. 如果一个 RL policy 用 VLM 作 perception module, 那 VLM 的 EgoThink score 上限决定了 policy 上限.

### 10.5 Future Benchmark 设想
理想 egocentric benchmark 应该:
1. **Video-based**: 真正 sequential reasoning
2. **Interactive**: agent 可以采取 action, 改变 environment
3. **Multi-modal**: 加 proprioception, audio, tactile
4. **Embodied evaluation**: 在 simulator 中 ground truth, 而非 human annotation
5. **Long-horizon planning**: multi-step task, 不只是 single QA

类似 [Habitat](https://aihabitat.org/) + VLM evaluation 的组合, 这可能是 EgoThink 的 next generation.

---

## 11. 总结

EgoThink 是 VLM evaluation 从 third-person 走向 first-person 的 seminal step. 它揭示了:
- 即使 SOTA VLM (GPT-4V) 也只达到 65.5/100
- Counting, Forecasting, Affordance 是 hardest dimensions
- Scaling LLM 只在 trainable 时有效
- Visual grounding (SoM) 不一定 help, 可能 hurt
- Open-source VLM 在某些 dimension (Location, Spatial) 可超 GPT-4V

对 community 的启示: 当前 VLM 还远不能 support 真正的 embodied AI. 我们需要 fundamentally different architecture, training data, evaluation methodology, 才能让 VLM 真正"think" from first-person perspective.

相关links:
- [EgoThink arxiv](https://arxiv.org/abs/2312.12973)
- [Ego4D dataset](https://ego4d-data.org/)
- [Set-of-Mark paper](https://arxiv.org/abs/2310.11441)
- [LLM-as-judge paper](https://arxiv.org/abs/2306.05685)
- [PaLM-E](https://palm-e.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)
- [MMBench](https://github.com/open-compass/MMBench)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Model)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [InstructBLIP](https://arxiv.org/abs/2307.08504)

希望这个 walkthrough 帮你 build intuition, Andrej. 这个 paper 触及的 first-person thinking 是 embodied AI 的真正 frontier, 值得 community 深入探索.
