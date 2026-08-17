---
source_pdf: GRIT Teaching MLLMs to Think with Images.pdf
paper_sha256: 492f0eee430f419408905eac5108f02da3d97007e146a3e83e9ebfaef946a357
processed_at: '2026-08-04T22:22:43-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话说说GRIT

## 这论文到底想解决什么问题

现在那些会做视觉推理的模型（比如R1-V、Vision-R1这些），嘴上说"让我想想..."，其实输出的reasoning chain全是纯文字描述，跟输入图片基本脱节。你根本不知道模型到底在看图的哪个角落得出结论的，它可能压根没认真看图就在那里编。

ChatGPT o3/o4有个很酷的能力叫"thinking with images"——它在思考过程中会真正refer到图片的具体区域。但这是OpenAI闭源的，外面搞不了。

GRIT就是UC Santa Cruz这帮人想用开源方式把这个能力做出来。

## 核心想法其实特别简单

让模型在reasoning chain里**自由混着输出文字和bbox坐标**。比如模型在想"这图里有个杯子在桌子上"的时候，它先吐出一串数字`[42, 73, 180, 250]`指向杯子所在区域，再继续往下推理。

关键点：**bbox生成完不真的把裁剪的图片再喂回模型**。模型得靠自己脑子里对原图的理解来"解释"自己刚画的框。这样既省事（不用多轮image input），又逼模型学会"管理自己的grounding动作"。

输出长这样：
```
<rethink> 一只是大奶牛，一只是小牛犊，都在地上站着 </rethink>
<answer> cow </answer>
```

## 训练方法：GRPO-GR

在GRPO基础上改造的（DeepSeek那套），核心是reward设计特别精妙：

**Format reward**只检查两件事：
1. 特殊token对（``、`<rethink></rethink>`）用没用对
2. 至少吐出一个语法合法的bbox（用regex抓四个逗号分隔的整数）

注意——**它根本不管你bbox指得对不对、reasoning内容是不是nonsense**。只要格式对了就给0.5分。

**Answer accuracy reward**才管语义对不对，用GPT-4o当judge打分，再加个BLEU做微调。

**Counting reward**（可选）：专门给counting任务用，bbox数量得匹配ground truth的数量。

Group sampling算advantage时做normalization（减均值除标准差），不需要value network。这就是GRPO的标准操作。

## 最反直觉的事：20个样本就能训出来

他们从VSR拿10个、TallyQA拿10个，总共20个image-question-answer三元组，训200步，12小时8卡A100，就成了。

为什么这么省？我的理解：

**模型本来就会grounding（能画框），也会reasoning（能写CoT），这两种能力在pre-training阶段都已经存在了**。GRIT干的事是教模型"把两个本事混着用"——就像教一个会跑也会跳的人"跳着跑"，不需要从零教跑步和跳跃。

Format reward极其简单（regex抓数字），所以20个样本上模型很快学会"哦原来要这么输出格式"。然后answer accuracy reward负责筛选——只有当bbox真正帮助推理时模型才能拿高分。RL的group sampling（每个prompt采4个completion）让模型自己探索哪些grounding策略管用。

200步 × 128 batch × 4样本 = 10万多个completion。虽然unique prompt只有20个，但模型见过的completion数量不少。

## 最cool的发现：bbox是"视觉锚点"

Figure 5那个实验特别有意思。他们做了这个对比：

1. 拿GRIT模型正常输出（reasoning里带bbox）
2. 把bbox全部删掉，重新喂给模型让它继续生成后面的reasoning
3. 对比两种情况下，模型在生成rethink段时对input visual tokens的attention

结果：**有bbox的版本，后续reasoning对image的attention明显更高**。

这说明bbox坐标在token sequence里不是摆设——它们像是在推理链里埋的"视觉锚点"，后面的token生成时会自然地query这些锚点对应的图像区域。模型输出`[42,73,433,296]`这些数字token，这些数字在embedding space里跟原图的visual token有某种learned关联（来自pre-training），所以后续reasoning的attention会回流到图像。

**bbox不只是给人类看interpretability的，它是模型自己的"视觉记忆辅助工具"**。

## 跟SFT比为什么RL更强

他们做了Few-shot SFT baseline——用同样的20个样本做监督微调，让模型学着输出text+bbox的pattern。

结果SFT比GRIT差不少。论文解释：SFT学到的是"表面模仿"——看到问题就机械输出某种text+bbox pattern，但bbox和reasoning之间没有真正的逻辑关联。RL训练的模型，bbox必须真的帮它答对才能拿reward，所以bbox是"functional"的，不是装饰。

这跟DeepSeek-R1的哲学一样：**process supervision不是必需的，outcome supervision够用**。GRIT证明这个原则在视觉grounding上也成立。

## 几个我注意到的细节

**Prompt里说用JSON格式输出bbox，但训练完模型不总是遵循**。这是故意的——reward用regex而不是JSON parser检测bbox，给RL更多search space。Prompt只是policy initialization，RL会自己调整实际行为。

**CoT baseline崩了**：直接让Qwen2.5-VL按grounded reasoning format输出，VSR准确率从49.5掉到37.5，MathVista从58.5掉到33.0。这证明off-the-shelf MLLMs根本没法zero-shot同时干grounding和reasoning，两个能力会互相干扰。这正是GRIT要解决的问题——把它们真正打通。

**MathVista和MME上GRIT不如Direct query**。数学推理和综合benchmark不需要spatial grounding，硬塞bbox反而干扰。这说明grounded reasoning format不是万能的，任务自适应是个未解问题。

## 局限挺明显的

1. **Out-of-domain泛化弱**——20样本到7000样本，in-domain持续涨，out-of-domain基本平了。引用了Yue et al.的研究说RL主要在bias现有pattern，不fundamentally改变能力，所以pre-training是天花板。

2. **依赖GPT-4o当judge**——API成本和潜在bias。要open-source化得换judge。

3. **只有bbox这一种grounding modality**——对细粒度任务（比如数重叠物体、子区域内的空间关系）可能不够表达。polygon、mask、point可以试试。

4. **没新的pixel input**——高效但对裁剪区域的细粒度理解可能受限。可以搞optional的高分辨率crop注入。

## 我的核心takeaway

GRIT最大的贡献是证明了一件事：**只要reward设计对（只管format和outcome，不管process），用极少数据RL就能激活MLLM里已有的grounding和reasoning能力，让它们真正协作**。

这给开源社区提供了一个practical path去复现o3/o4的"thinking with images"——不需要海量标注数据，不需要复杂的process supervision，20个样本+精心设计的format reward就够了。

同时也提出了一个深问题：**reasoning chain里的symbolic pointers（比如bbox坐标）到底在模型内部扮演什么角色？** Figure 5的attention分析暗示它们是"视觉记忆的离散锚点"，这可能有更深的应用——不限于bbox，任何symbolic pointer都可能作为跨modality的reasoning bridge。

论文链接：https://grounded-reasoning.github.io
GRPO原文：https://arxiv.org/abs/2402.03300
RL是否真改变reasoning的讨论：https://arxiv.org/abs/2504.13837

---

# GRIT: Teaching MLLMs to Think with Images - 深度技术解析

## 1. 论文核心动机与Intuition

这篇论文来自UC Santa Cruz的Xin Eric Wang组和eBay合作, 核心问题非常清晰: 当前open-source vision reasoning models (比如R1-V, Vision-R1, VLM-R1)生成的reasoning chain全是pure natural language, 缺乏visual grounding。这导致模型的"思考过程"和input image脱节, 我们无法验证模型到底在看图像的哪个region来得出结论。

GRIT的核心insight来自一个观察: **MLLM本身已经具备grounding能力(能输出bbox)和reasoning能力(能生成CoT), 但这两个能力在pre-trained model中是disconnected的**。GRIT通过极其lightweight的RL (仅20个samples!) 让模型学会在reasoning chain中自由interleave bounding box coordinates和natural language, 实现这两种能力的unification。

这让我想到DeepSeek-R1在language domain的做法——RL with verifiable rewards能激发reasoning能力而不需要process supervision。GRIT把这个思路迁移到vision grounding: reward只关心final answer correctness和output format (是否包含valid bbox), 完全不约束reasoning chain的具体内容或bbox的semantic accuracy。

Project page: https://grounded-reasoning.github.io

---

## 2. Grounded Reasoning Paradigm详解

### 2.1 输出结构

给定input image I和question q, 模型生成(c, a):
- c = reasoning chain, 起始于`` token
- c中自由mix natural language tokens T和bounding box coordinates B
- 在第p个generation step, 模型基于(I, q, c_{1:p-1})决定生成text token c_p ∈ T或bbox coordinates c_p ∈ B
- a = final answer, 在`<answer>`token之后

关键设计决策:**生成bbox后不向模型输入cropped image pixels**。模型必须依赖对原始input image的internal understanding来解释自己生成的coordinates。这避免了multi-turn image input的复杂性, 同时强制模型学习"解释自己grounding actions"的能力。

### 2.2 特殊Token结构

```
<reasoning tokens with optional bboxes> 
<rethink> [reflection/analysis] </rethink>
<answer> final_answer </answer>
```

这里有个有趣的设计: `<rethink>` token允许模型在给出初步答案后进行reflection。从Figure 3的例子看, 模型有时会:
- (i) 先grounding再给出答案, 然后在rethink中分析
- (ii) 先给初步答案再rethink修正
- (iii) 对不存在的entity正确处理, 不产生false-positive grounding

这种flexibility来自于reward设计不约束内容, 只约束format。

---

## 3. GRPO-GR算法深度解析

### 3.1 RL Formulation

模型作为policy π_θ, 给定(I, q)生成output sequence (c, a)。训练时对每个(I, q)采样N个候选completions {o_1, ..., o_N}。

**Group-normalized advantage** (公式1):

$$A_i = \frac{r_i - \text{mean}\{r_1, \ldots, r_N\}}{\text{std}\{r_1, \ldots, r_N\} + \delta}$$

变量解释:
- A_i: 第i个completion的normalized advantage
- r_i: 第i个completion的task reward (下面详解)
- mean{r_1, ..., r_N}: group内rewards均值
- std{r_1, ..., r_N}: group内rewards标准差
- δ: 小常数 (10^{-8}), 防止除零

这个normalization是GRPO相比PPO的关键创新——**不需要value network**, 而是用group statistics作为baseline。这意味着每个prompt需要采样多个completions来估计baseline, 计算成本高但实现简单。

### 3.2 Task Reward分解

$$r_i = r_{\text{format}} + r_{\text{count}} + r_{\text{ans}}$$

#### (1) Grounded-reasoning-format reward (公式2):

$$r_{\text{format}} = s_{\text{st}} + s_{\text{bf}}$$

其中:
- s_st = 0.5 × I(correct think token pair) + 0.5 × I(correct rethink token pair)
  - 检查``...``和`<rethink>...` `</rethink>`是否正确出现且顺序正确
  - 每个正确token pair贡献0.5分
- s_bf = 0.5 × I(num_bboxes ≥ 1)
  - 通过regex匹配四个逗号分隔的整数检测bbox
  - 至少一个valid bbox就给0.5分

**关键设计哲学**: format reward只关心**结构和语法**, 不关心**语义正确性**。bbox可以是错误的region, reasoning内容可以是nonsense, 只要format对就给分。这避免了需要bbox annotations的需求。

#### (2) Grounded-target-counting reward (r_count):

专门用于counting任务(VSR+TallyQA混合训练时):
- r_count = 0.5 if 生成bbox数量 == ground truth count
- 鼓励模型在counting reasoning中systematically生成对应数量的bbox

这个reward在ablation study (Table 3)中证明重要——去掉它会导致GIoU从0.387降到0.349 (in-domain), out-of-domain ACC从64.4降到60.0。

#### (3) GPT-aided answer-accuracy reward:

$$r_{\text{ans}} = s_{\text{GPT}} + 0.1 \cdot s_{\text{BLEU}}$$

变量解释:
- s_GPT: GPT-4o作为judge给(q, â, a)三元组打0或1分
- s_BLEU: predicted answer â和ground truth a的sentence-level BLEU-1
- 0.1权重: 因为BLEU对长度mismatch敏感, 降权确保最高reward给精确匹配

用GPT-4o作为judge比rule-based matching更robust, 能处理语义等价但表述不同的情况。Prompt见Figure 8。

### 3.3 GRPO Objective (公式3)

$$\mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left[ \min\left(s_i A_i, \text{clip}(s_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{KL}\left(\pi_\theta(\cdot|q) \| \pi_{\text{ref}}(\cdot|q)\right) \right]$$

变量解释:
- θ: policy parameters (模型weights)
- N: group size (论文中=4)
- s_i: importance ratio = π_θ(o_i|q) / π_{θ_old}(o_i|q)
  - π_θ: current policy
  - π_{θ_old}: policy before update
  - 衡量新旧policy对同一completion的概率比
- A_i: 公式1计算的advantage
- ε: PPO clip range, 定义trust region
- β: KL penalty coefficient
- π_ref: reference policy (通常initial pre-trained model)
- D_KL: KL散度

这是标准PPO-style objective加上KL正则化, 防止policy偏离pre-trained model太远。KL参考DeepSeekMath的GRPO设计: https://arxiv.org/abs/2402.03300

---

## 4. 实验设置详解

### 4.1 Training Setup

| 项目 | 配置 |
|------|------|
| Base models | Qwen2.5-VL-3B, InternVL3-2B |
| Training samples | 20 (10 VSR + 10 TallyQA) |
| Training steps | 200 |
| Total batch size | 128 |
| Group size N | 4 (每个sample采样4个completions) |
| Learning rate | 2×10^{-6} |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Hardware | 8× NVIDIA A100 80GB |
| Training time | ~12 hours |
| Framework | DeepSpeed Zero2 |

**20个samples训练200 steps, batch size 128** —— 这意味着每个unique sample被重复使用多次。RL的sample efficiency来自于group sampling: 每个step生成4×128=512个completions, model从相对ranking中学习。

### 4.2 Testing Data统计 (Table 2)

| Dataset | Count | Avg Q/A length | Multi-choice/Yes-No (%) | Annotated grounding (%) |
|---------|-------|----------------|--------------------------|--------------------------|
| VSR | 288 | 6.7/1.0 | 71.2 | 58.8 |
| TallyQA | 491 | 6.0/1.0 | 0 | 25.6 |
| GQA | 509 | 7.1/1.0 | 58.9 | 25.3 |
| MathVista | 1000 | 38.2/1.2 | 70.8 | - |
| MME | 240 | 13.3/1.0 | 100 | - |
| OVDEval | 2164 | 16.4/4 | 0 | 17.3 |

注意OVDEval的answer length是4 (bbox坐标), 这是open-vocabulary detection任务, grounding是explicit answer而非reasoning的optional component。

---

## 5. 核心实验结果分析

### 5.1 Table 1 主结果深度解读

**Qwen2.5-VL 3B:**

| Method | VSR ACC/GIoU | TallyQA ACC/GIoU | GQA ACC/GIoU | MathVista ACC | MME ACC | OVDEval GIoU |
|--------|--------------|------------------|--------------|---------------|---------|--------------|
| Direct query | 49.5/0.000 | 40.8/0.000 | 55.4/0.000 | 58.5 | 88.9 | 0.389 |
| CoT | 37.5/0.213 | 33.2/0.113 | 39.5/0.269 | 33.0 | 41.3 | 0.388 |
| One-shot ICL | 13.2/0.122 | 36.3/0.268 | 20.4/0.441 | 29.1 | 24.7 | 0.328 |
| Few-shot SFT | 59.7/0.216 | 44.5/0.447 | 64.6/0.475 | 45.0 | 68.3 | 0.391 |
| **GRIT** | **72.9/0.325** | **47.8/0.284** | **62.8/0.485** | 45.0 | 68.3 | **0.391** |

**关键观察:**

1. **Direct query在MathVista和MME上反而最强** (58.5, 88.9), GRIT在这两个dataset上有所下降。这暗示GRIT的grounded reasoning format对某些任务(特别是数学推理和综合评估)可能有干扰, 因为这些任务可能不需要spatial grounding。

2. **CoT和One-shot ICL严重退化** —— CoT在VSR上从49.5降到37.5, MathVista从58.5降到33.0。这验证了论文核心论点: **off-the-shelf MLLMs无法在zero-shot下同时进行grounding和reasoning**。强制它输出bbox会破坏其原有能力。

3. **Few-shot SFT虽好但不及GRIT** —— SFT能学到surface format (interleave text和bbox), 但GRIT的RL训练让grounding真正inform reasoning。这呼应了论文Section 4.2的论断: "supervised fine-tuning primarily learns to mimic the surface form... rather than developing a deeply integrated reasoning process"。

4. **GRIT在OVDEval上GIoU = 0.391, 仅略高于Direct query的0.389**。但GRIT的value在于**统一能力**——同一个model既能VQA又能grounding, 而非每个任务单独最优。

### 5.2 Attention Analysis (Figure 5) —— 最有趣的发现

这个实验设计非常精妙:

1. 取GRIT-trained Qwen2.5-VL在GQA上100个样本的output
2. 用`<rethink>`token切分为pre-rethink (含bbox)和rethink segments
3. 创建alternative pre-rethink: 移除所有bbox
4. 喂回模型生成no-bounding-box rethink content
5. 比较original rethink vs no-bbox rethink生成时对input visual tokens的attention

**结果**: original rethink的visual attention显著高于no-bbox版本。这提供了**mechanistic evidence**: bbox的生成本身会引导后续reasoning更关注image, 即使没有新的pixel input。这是一种"self-attention引导"机制——模型通过输出bbox coordinates在token sequence中植入了visual anchors, 后续token的attention会自然回归到这些anchors对应的visual regions。

这个发现对build intuition非常关键: **reasoning chain中的grounding tokens不仅是给人类看的interpretability aid, 它们是model自己的"视觉记忆辅助"**。

### 5.3 Cross-modal Correlation Metric (Figure 4)

这个evaluation metric设计很巧妙:

1. 提取generated reasoning chain中的bbox集合 {c_i | c_i ∈ B}
2. 从input image随机采样等量bbox作为negative candidates {h_0, ..., h_j}
3. 分别在input image上绘制两组bbox
4. 用GPT-4o判断哪组bbox更corresponds to textual reasoning (bbox coordinates被masked)
5. 重复3次取平均

这个metric本质上是在测**bbox和text的semantic alignment**, 利用GPT-4o的Set-of-Mark能力。GRIT-trained model超过Zero-shot ICL和Few-shot SFT, 但仍低于human baseline——说明RL确实学到了semantic alignment, 但还没达到人类水平。

Set-of-Mark reference: https://arxiv.org/abs/2310.11441

### 5.4 Data Scaling (Figure 6)

训练数据从20 → 500 → 7000:
- **In-domain (VSR, TallyQA)**: 持续提升, 但增长diminishing
- **Out-of-domain (GQA, MathVista)**: 提升subtle

论文引用了Yue et al. (https://arxiv.org/abs/2504.13837)的发现: **RL with verifiable rewards主要bias现有reasoning patterns而非fundamentally改变它们**。这意味着pre-training阶段决定的能力上限, RL只是激活和引导。这对GRIT的implication: 要提升out-of-domain性能, **data variety比data volume更重要**。

---

## 6. 与相关工作的联系与对比

### 6.1 RL for Vision-Language Reasoning谱系

| 方法 | 训练方式 | Grounding方式 | 需要bbox标注 | 关键限制 |
|------|----------|---------------|--------------|----------|
| R1-V (https://github.com/Deep-Agent/R1-V) | GRPO | 无explicit grounding | 否 | 专攻math reasoning |
| Vision-R1 (https://arxiv.org/abs/2503.06749) | RL | symbolic reasoning | 否 | 不集成grounding |
| R1-OneVision (https://arxiv.org/abs/2503.10615) | RL | 无 | 否 | diagram reasoning |
| VLM-R1 (https://arxiv.org/abs/2504.07615) | RL | bbox作为final answer | 是 | 无interleaved reasoning |
| **GRIT** | **GRPO-GR** | **interleaved in reasoning** | **否** | **20 samples即有效** |

VLM-R1是最近的对比: 它用RL训练referring expression comprehension, 但bbox是**output**而非**reasoning中间步骤**。GRIT的核心创新在于把grounding变成reasoning process的一部分。

### 6.2 Visual CoT推理历史

早期工作如Multimodal-CoT (https://arxiv.org/abs/2302.00923)用multi-stage prompting; CCoT用scene graphs作为external tool; UV-CoT用self-generated bbox+auxiliary MLLM supervision。这些都依赖prompting或auxiliary modules, 而非end-to-end学习interleaved reasoning。

VisCoT (https://arxiv.org/abs/2402.05119), CogVLM (https://arxiv.org/abs/2311.03079), CogCoM需要dense annotations——每个reasoning step要link到specific visual evidence。GRIT通过task-level reward alone就实现了, 这是数据效率的根源。

---

## 7. 我的思考与Intuition Building

### 7.1 为什么20个samples够用?

这是最反直觉的发现。我的理解:

1. **Pre-trained MLLMs已具备两种能力**, GRIT只是教它们"如何组合"。这就像教一个会跑步和会跳的人"跳着跑"——不需要重新学跑步和跳跃。

2. **Format reward极度简单**: regex匹配四个逗号分隔的整数 + 特殊token pair。这种syntactic pattern在少量samples上就能稳定学习。

3. **Answer accuracy reward提供semantic signal**: 模型必须让bbox真正帮助reasoning才能得到高分。RL通过group sampling自动探索哪些grounding策略有效。

4. **Group sampling的效率**: 每个step 128 batch × 4 samples = 512 completions。200 steps = 102,400 completions。虽然unique prompts只有20个, 但model见过的completions数量很大。

### 7.2 Bbox作为"thinking anchor"的mechanism

Figure 5的attention analysis让我想到一个更深的问题: **bbox coordinates在token sequence中到底扮演什么角色?**

我的hypothesis: bbox coordinates是**离散化的visual pointers**。当模型生成"[42, 73, 433, 296]"这样的token时, 这些数字tokens在embedding space中与input image的visual tokens有某种learned association (来自pre-training)。后续reasoning tokens的attention会自然query这些pointer tokens, 从而间接attend到对应的visual regions。

这解释了为什么**不需要cropped image作为新input**——coordinates本身已经是"压缩的visual reference"。这也暗示了一种新的reasoning architecture可能性: 用symbolic pointers (不限于bbox)作为reasoning chain中的modality bridges。

### 7.3 RL vs SFT的根本区别

Few-shot SFT vs GRIT的对比是最重要的ablation。SFT学习的是**imitating surface form**——看到question就输出某种text+bbox pattern。GRIT学习的是**functional behavior**——bbox必须帮助得出正确答案。

这呼应了DeepSeek-R1的哲学: **process supervision不是必需的, outcome supervision足够**。GRIT进一步证明这个原则在multimodal grounding上也成立, 只需要设计合适的format reward来约束输出空间。

### 7.4 局限性与未来方向

1. **Out-of-domain generalization弱** —— 7000 samples后仍有瓶颈, 说明pre-training是关键。未来可能需要更diverse的pre-training数据。

2. **MathVista和MME上GRIT不如Direct query** —— 暗示grounded reasoning format不适合所有任务。可能需要task-adaptive的reasoning format selection。

3. **依赖GPT-4o作为judge** —— 这引入了API cost和potential bias。未来需要open-source judge或更强的rule-based reward。

4. **Bbox作为唯一grounding modality** —— 对于fine-grained reasoning (如counting overlapping objects, spatial relation within sub-region), bbox可能不够expressive。可以探索polygon, mask, 或point作为补充。

5. **没有新的pixel input** —— 虽然高效, 但可能限制了对cropped region的细粒度理解。可以考虑optional的high-resolution crop injection。

---

## 8. 与OpenAI o3/o4的联系

论文提到ChatGPT-o3/4有类似的"thinking with images"能力。GRIT是**open-source的首次尝试**实现interleaved visual-textual reasoning via lightweight RL。

OpenAI的可能做法(推测): 可能用更大量data + 更复杂reward + 可能的pixel-level grounding (如attention masks)。GRIT证明**bbox coordinates作为discrete grounding signal已经足够**, 这为open-source社区提供了practical path。

o3/o4 reference: https://openai.com/index/introducing-o3-and-o4-mini/

---

## 9. 实现细节的Practical Considerations

### 9.1 Prompt Suffix设计 (Figure 7)

```
First, think between  and  while output necessary coordinates 
needed to answer the question in JSON with key 'bbox_2d'. 
Then, based on the thinking contents and coordinates, rethink 
between <rethink> </rethink> and then answer the question after <answer>.
```

注意: prompt说用JSON, 但GRIT-trained model不总是遵循。这是intentional的——GRPO-GR用regex而非JSON parser检测bbox, 给RL优化更多search space。这启示: **prompt只是policy initialization, RL会调整实际行为**。

### 9.2 Bbox检测的Regex

论文没给具体regex, 但根据描述应该是类似:
```python
pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
```
匹配[x1, y1, x2, y2]格式。这个宽松的检测避免了JSON parsing的脆弱性。

### 9.3 GPT-4o Judge的Robustness

用GPT-4o作为reward function有风险: judge本身可能有bias, 且API cost不低。论文用固定prompt (Figure 8)确保一致性。未来可考虑:
- Open-source VLM作为judge (如Qwen2.5-VL-72B)
- 训练专门的reward model
- Hybrid rule-based + LLM judge

---

## 10. 总结: GRIT的核心贡献

1. **Conceptual**: 首次提出grounded reasoning paradigm——MLLM在reasoning chain中interleave bbox和text, 不需要新pixel input。

2. **Methodological**: GRPO-GR算法, 用format reward (不约束semantic) + answer accuracy reward, 消除对reasoning chain和bbox annotation的需求。

3. **Empirical**: 仅20 samples训练12小时, 在Qwen2.5-VL-3B和InternVL3-2B上实现grounding和reasoning的unification。

4. **Insight**: Bbox在reasoning chain中起"visual anchor"作用, 增强后续reasoning对image的attention (Figure 5 mechanistic evidence)。

5. **Limitation honesty**: 坦诚承认out-of-domain generalization和数据scaling的挑战, 引用RL主要bias而非change reasoning patterns的近期研究。

GRIT是一个elegant的proof-of-concept, 证明了通过minimal RL训练可以实现"thinking with images"。它为open-source multimodal reasoning提供了新的design space, 也提出了关于grounding symbols在reasoning中role的深刻问题。
