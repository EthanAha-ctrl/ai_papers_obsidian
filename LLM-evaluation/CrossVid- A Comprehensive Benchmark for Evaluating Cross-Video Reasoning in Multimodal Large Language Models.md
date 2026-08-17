---
source_pdf: CrossVid- A Comprehensive Benchmark for Evaluating Cross-Video Reasoning
  in Multimodal Large Language Models.pdf
paper_sha256: a2acf7f5fabb05fe2ae86ffe6f4aba0c1ee0e90ed2fcb13195ead0b1d17af710
processed_at: '2026-08-03T17:54:14-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话聊聊 CrossVid 这篇 paper

## 一、这帮人到底在搞啥

简单说：现在大家都在卷 video MLLM，但所有的 benchmark（Video-MME、MLVU、LongVideoBench、NExT-QA、MVBench）全都只测一个视频。你给模型一段视频，问它点啥。这玩意儿已经卷得差不多了，Gemini-2.5-Pro、Qwen2.5-VL 在 single-video 上都打得有来有回。

但 Xiaohongshu 这帮人发现一个尴尬的事实——**没人在测"多个视频一起喂进去会咋样"**。比如你给模型三段做同一道菜的烹饪视频，问它"哪段视频的做法最容易出水"；或者给两段无人机俯拍的同一路口的视频，问"红车并行时 A 视角里有几辆车在动"——这种任务，模型直接拉胯。

他们就把这种任务叫 **Cross-Video Reasoning (CVR)**，然后搞了一个 benchmark 叫 CrossVid 来专门测它。就这么个事。

## 二、为啥这事值得搞

你可能会想：多视频不就是把单视频的能力复用一下吗？模型 attention 又不挑食，多塞几个视频 token 进去不就完了？

但实际跑出来发现完全不是这么回事。几个根本问题：

**Token 预算被稀释**。假设一个模型能吃 256 帧，单视频你能看 256 帧。现在塞 4 个视频，每个只剩 64 帧。烹饪视频里"裹粉"这一步可能就 2 秒，uniform sampling 一抽，这关键帧直接没了。这就是 paper 里说的 "key frame loss" 错误。

**Cross-video attention 稀疏**。现在的 MLLM 把所有 video token 拼一起做 self-attention，模型根本没学过"哪些 token 属于视频 1、哪些属于视频 2、它们之间该怎么对齐"。它只会傻乎乎地全 attend 一遍。这跟人类做对比的思路不一样——人类是先抓每个视频的关键点，再做 cross-video mapping。

**Temporal grounding 本来就烂**。FSA 任务里要求模型在 video 2 里找到跟 video 1 指定区间功能等价的步骤，还要输出精确时间戳。Gemini-2.5-Pro 只答对 13.4%，人类 85.2%。这玩意儿本质是 cross-video temporal grounding，单视频的 temporal grounding 都没做好，跨视频直接雪崩。

**Multi-view spatial reasoning 烂**。VisDrone 那种俯拍无人机视频，模型在预训练里基本没见过这种视角，加上物体又小，MSR/MOC 任务上模型基本靠蒙。

所以 CVR 真的不是单视频能力的线性外推。这一点是 paper 最有价值的发现。

## 三、Benchmark 长啥样

CrossVid 一共 5331 个视频，9015 个 QA pair，平均每个 query 涉及 770 秒视频，最长的到小时级。任务体系分四层：

**Comparative Analysis**——比较类。比如 4 段同 genre 电影片段，问哪段里 vehicle 对主线最不重要；或者 4 段做同一道菜的视频，问哪段的最后调味步骤与众不同。核心是 inter-video 的语义对比。

**Temporal Understanding**——时序类。三个子任务：
- PI：给电影开头+结尾，猜中间。考验 causal narrative。
- FSA：两段烹饪视频，在 video 2 里找到对应 video 1 某个区间的功能等价步骤。考验 cross-video temporal correspondence。
- PSS：把一段烹饪视频打乱，让模型排回正确顺序。考验 within-video causal ordering。

这里有个很巧的反作弊设计——他们对 PSS 任务做了 temporal realignment，前一个 clip 提前 1-5 秒，后一个 clip 延后 1-5 秒。为啥要这么搞？因为如果不这么搞，模型可能靠"画面颜色连续性""镜头角度平滑"这种 low-level cue 排序，根本不真的理解内容。这种反 shortcut 的 trick 值得偷来用。

**Multi-view Reasoning**——双无人机视角。MSR 问空间关系（A 视角里某物体离开时，B 视角里另一物体在哪），MOC 问计数（某时刻 A 视角里有几辆车在动）。这俩是纯人工标注的，因为 VisDrone 的 bbox 是米级精确，可以拿颜色把 5 个目标物体标出来，让 annotator 看着标记视频出题。

**Free-form QA**——开放式。CCQA 给两段做同一道菜的视频，问"这两段烹饪流程有啥区别"。用 GPT-4.1 做 judge 评分。

## 四、数据怎么造出来的

工业界建 benchmark 的标准套路——semi-automated pipeline：

1. 用 Qwen2.5-VL-72B 给每个视频做 frame-level caption。这里有个 trick：他们按时间分组，相邻 2-8 帧一组，并且如果原数据集有 step interval（YouCook2 有），就按那个分。相当于把 prior structure 注入到 captioner 里。
2. 把同 set 内视频的 caption 喂给 DeepSeek-R1，让它生成 QA pair。Prompt 里强制要求三件事：必须分析视频间关系、必须匹配任务需求、必须输出 rationale。
3. 人工过滤——10 个专家 annotator 把不靠谱的 QA 砍掉。三条标准：跟视频理解无关的砍、只引用单视频的砍、主观的砍。
4. 人工 refinement——改问题去歧义，对单选/多选确保选项唯一正确，对 PSS 做 temporal realignment 反作弊，对 open-ended 检查 scoring points 覆盖度。
5. 第二波专家 quality control 复审。

这套流程里最值得 build intuition 的点是——**他们把 DeepSeek-R1 当成"出题员"**。R1 的 thinking 能力被用来做 video relation reasoning，把 caption 里隐含的对比关系显式化为 QA。这其实暗示了一种 LLM-as-annotator 的新用法——reasoning model 不只用来答 QA，也可以用来出 QA。

## 五、结果说明啥

22 个模型实测，最关键的几个数字：

- Human：89.2%
- Gemini-2.5-Pro：50.4%（最强 MLLM）
- GPT-4.1：45.2%
- Doubao-1.5-VL-pro：44.3%
- GPT-4o：36.8%
- 最强 open-source 是 GLM-4.1V-9B-Thinking：35.1%
- Qwen2.5-VL-72B：34.4%

**39 个点的 human-MLLM gap**，这是相当大的 headroom。

几个有意思的 patterns：

**Closed-source 全面吊打 open-source**。最弱的 closed-source（GPT-4o 36.8%）都比最强的 open-source（GLM-4.1V-9B-Thinking 35.1%）强一点。这个 gap 比 image understanding 上大得多，背后原因大概率是：(a) 闭源有专有视频数据，YouTube-scale；(b) 长上下文+多视频的 inference 工程优化更猛；(c) Gemini 的 native multimodal 架构比 open-source 的 late-fusion 更适合长视频。

**Thinking-enabled 模型在 CVR 上优势特别明显**。Gemini-2.5-Pro 压过 GPT-4.1 / GPT-4o / Doubao。7B 段里 GLM-4.1V-9B-Thinking 和 MiMo-7B 领先。这暗示 CVR 本质是 multi-hop reasoning——先 parse 每个视频、再提取 task-relevant feature、再 align/compare、再 aggregate 回答。这种结构正好是 CoT/thinking 的甜区。

**Temporal 任务上的极端低分**。FSA 上 Gemini-2.5-Pro 也只有 13.4%，PSS 上 Gemini 拿 78.2%。这两个差距说明什么？PSS 是 within-video ordering，模型只要理解单视频的 causal chain 就行；FSA 是 cross-video alignment，要在两个视频间建立 functional correspondence。前者是模型已经有点能力的方向，后者是几乎从零开始。

## 六、几个 Ablation 的技术 insight

**Frame number 的影响不是单调的**。Qwen2.5-VL-72B 在 32→256 frame 下的表现：
- Comparative Analysis：37.0 → 47.5（单调上升，更多 frame 直接帮助对比）
- Temporal Understanding：33.8 → 37.4 → 34.5 → 33.9（非单调！64 frame 时最佳，再加 frame 反而跌）
- Multi-view：基本持平
- CCQA：18.9 → 34.0（暴升 15.1 个点）

这个非单调性非常值得 build intuition。Temporal reasoning 依赖 key events 而非 dense frames。论文举的例子：判断战争片开头结尾中间会演啥，32 frame 时模型能抓到"运兵车队"和"谈判场景"两个关键事件；256 frame 时被"受伤士兵的镜头"这种 atmospheric 信息干扰，反而答错。

这本质上是个 signal-to-noise ratio 问题。更多 frame 既增加 signal 也增加 noise，对 event-centric reasoning 来说 noise 增长可能快于 signal。这暗示 **adaptive keyframe selection 比 uniform sampling 更可能是 CVR 的解**。

**CoT 是放大器，不是修正器**。CoT 三阶段：理解问题 → 逐视频分析 → 跨视频聚合。

- GPT-4.1 加 CoT 几乎无收益，甚至微跌。推理能力已经内化，显式 CoT 反而可能引导走偏。
- Qwen2.5-VL-72B 加 CoT 大幅收益（O.Avg +5.1%, M.Avg +11.4%）。大模型能从 prompt engineering 中受益。
- MiniCPM-o 2.6 加 CoT 反而受损（T.Avg 从 26.4 跌到 18.7）。小模型在长 reasoning chain 上 error accumulate。

这个 finding 跟 math reasoning 上的报告一致——DeepSeek-R1 paper 也提到小模型 RL 训练后 reasoning chain 暴涨但准确率不升反降。**CoT 这种东西对强模型是杠杆，对弱模型是负担**。

## 七、四类错误告诉咱啥

论文手分析了 GPT-4.1 / MiniCPM-o 2.6 / InternVL3-38B / Qwen2.5-VL-72B 在 CoT 下的错误：

**(a) Key frame loss**——uniform sampling 把关键帧抽丢了。例：判断 foie gras 是否裹粉，video 2 实际裹了但 sampled frames 漏掉，Qwen2.5-VL-72B 答错。

**(b) Video understanding error**——抓到关键帧了，但单视频理解就错了。例：判断 hug 在不同电影中的含义，模型定位到了所有 hug 帧，但 video 2 的 hug 含义理解错。

**(c) Cross-video comparison error**——单视频全对，跨视频比较时崩了。例：分析 dim light 在制造悬念氛围中的作用，MiniCPM-o 2.6 对每个视频的 lighting 分析都对，但聚合时只做简单对比，没抓住 cross-video narrative function。

**(d) Format error**——理解对了，输出格式不对。FSA 要求输出 "15,23"，模型输出了自然语言描述。

**核心 insight**：(c) 类错误是 CVR 的本质挑战。单视频能力没法线性外推到多视频。当前 MLLMs 训练数据几乎全是 single-video QA，没有 cross-video comparison 的监督信号。这是一种 capability extrapolation gap。

## 八、对咱做 MLLM 的有啥启示

我自己读完有几个 take-away：

**Token budget 管理是低垂果实**。当前 uniform allocation 太傻了。一个高动作密度 cooking video 和一个静态对话场景分配相同 frame 数，前者信息丢光。Video-specific token compression、hierarchical encoding、cross-video token sharing 都是值得探索的方向。

**Cross-video attention 需要专门设计**。当前 self-attention 让所有 video token 互相 attend，但缺 task-guided cross-video attention。可以借鉴 video retrieval 的思路——先用一个轻量 encoder 把每个视频压成 memory token，比较在 memory token 层面进行，再 selective retrieve fine-grained feature。这跟 RAG 的思路一模一样。

**Multi-video instruction tuning dataset 是缺位的**。类似 LLaVA-NeXT-Interleave 把 multi-image 扩展到 MLLM，multi-video instruction tuning 该提上日程了。可以用 LLM 从同 set 的 caption 合成对比类 QA，做监督训练。

**Temporal grounding 需要重做**。FSA 13.4% 这个分数说明当前 visual token 是 pooled feature，丢失了 precise temporal position 信息。可能的方向：timestamp-aware positional encoding、dense temporal feature preservation、anchor-based temporal span prediction（类似 DETR 的 object query 但用在 temporal 上）。

**Reasoning model 在 video 上的甜区被验证**。thinking-enabled 模型在 CVR 上优势比在 single-video 上更显著，跟 CVR 的 multi-hop 结构吻合。这预示 video reasoning model（o1/R1-style）会是接下来 12-18 个月的一个明确方向。Video-R1 已经开了头，CrossVid 给了它一个清晰的 evaluation framework。

## 九、我自己的几点碎碎念

**CVR 是 single-video 能力的放大镜**。CrossVid 暴露的很多问题在 single-video benchmark 上已经存在，CVR 只是把它们放大了。把 CVR 做好很可能不需要新架构，而是把 single-video 能力做扎实 + 加上 multi-video instruction tuning。

**Long context ≠ multi-video**。LongVA-7B-DPO 这种长上下文 video 模型在 CVR 上只有 18.0%，比 Qwen2.5-VL-72B 的 34.4% 差一大截。long context within a video 跟 multi-video aggregation 是两种不同的能力，训练数据中前者不一定包含后者。

**评估有几个洞**。CCQA 用 GPT-4.1 做 judge，但 GPT-4.1 自己又是被评估对象——self-judge bias。FSA 没说清楚是用 IoU 平均还是 IoU > threshold 的命中率。Multi-video 输入格式（concat? separator token?）也没讲清楚，影响 video boundary 感知。这些都是后续工作可以补的洞。

**Missing oracle baseline**。理想上应该有个实验——给模型 perfect caption 而非 frame，看 pure LLM reasoning 能达到多少。这能 disambiguate "visual perception limit" vs "reasoning limit"。论文没做这个 ablation，让咱没法知道瓶颈到底在哪头。这是一个低垂的实验，下一篇 paper 应该补上。

**Multi-view 的低分部分是 OOD 问题**。VisDrone 的 bird-eye view 在预训练分布外，物体又小。MSR/MOC 的低分不一定全是 multi-view reasoning 能力差，可能也是 view invariance 没学好。如果想干净测 multi-view reasoning，应该用 Ego-Exo4D 这种 ego+exo 配对，分布更接近训练数据。

总的来说，CrossVid 是个 timely 的 benchmark，抓住了 video understanding 演进的下一个 natural step。从 single-image → multi-image → single-video → multi-video 这条路径看，multi-video 是必然方向。50.4% vs 89.2% 的 gap 说明这事还在早期，谁先把 multi-video instruction tuning 这套做出来，谁就能在下一代 video MLLM 上占位。我赌 12 个月内会看到至少三篇 follow-up：multi-video instruction dataset、video reasoning model with RL on CVR、cross-video attention mechanism。

参考资料：
- CrossVid GitHub: https://github.com/chuntianli666/CrossVid  
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Video-R1: https://arxiv.org/abs/2503.21776
- GLM-4.1V-Thinking: https://arxiv.org/abs/2507.01006
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- All-Angles Bench: https://arxiv.org/abs/2504.15280
- LongVideoBench: https://arxiv.org/abs/2407.15754
- Video-MME: https://arxiv.org/abs/2405.21075
- Ego-Exo4D: https://arxiv.org/abs/2406.16776
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- YouCook2: https://arxiv.org/abs/1803.05266
- Assembly101: https://arxiv.org/abs/2204.05859
- MovieChat: https://arxiv.org/abs/2307.16438
- LLaVA-NeXT-Interleave: https://arxiv.org/abs/2407.07895

---

# CrossVid 深度解读：Cross-Video Reasoning 的第一个系统性 Benchmark

## 一、论文定位与核心动机

CrossVid 来自 Xiaohongshu 团队，瞄准的是 Multimodal LLMs (MLLMs) 领域一个长期被忽视的盲点——**Cross-Video Reasoning (CVR)**。现有 video understanding benchmark（如 Video-MME、MLVU、LongVideoBench、NExT-QA）几乎全部聚焦在 single-video 范式，即便有 multi-view 工作（All-Angles Bench、Ego-Exo4D、EgoExoLearn），也局限于"同一场景的多视角"这一很窄的子问题。

CVR 的本质挑战在于：模型需要在 **single query, multi-video input** 的设定下，同时完成 **information aggregation** 和 **cross-video comparison**。这触及了当前 MLLMs 架构层面的几个根本短板——token budget 在多视频间的稀释、cross-video attention 的稀疏化、temporal grounding 在压缩表示下的失效。

GitHub: https://github.com/chuntianli666/CrossVid

## 二、Benchmark 设计哲学

### 2.1 Hierarchical Task 体系

CrossVid 把 CVR 分成 **4 个 high-level dimensions**，下挂 **10 个 specific tasks**：

| Dimension | Tasks | 视频数 / query | 视频源 |
|---|---|---|---|
| Comparative Analysis | BU, NC, CC, PEA | 3-4 | Charades, Animal Kingdom, MovieChat-1K, YouCook2, Assembly101 |
| Temporal Understanding | PI, FSA, PSS | 2-6 | MovieChat-1K, YouCook2 |
| Multi-view Reasoning | MSR, MOC | 2 | VisDrone |
| Free-form QA | CCQA | 2 | YouCook2 |

这个设计有几个值得 build intuition 的点：

**Comparative Analysis** 测试的是 *inter-video 的语义对比能力*——比如 BU 任务要求对比动物行为的目的（cooling method 与 water loss 的关系），需要从行为外观推到 functional intent，跨视频比较。

**Temporal Understanding** 是这篇 paper 最有技术含量的部分。三个子任务对应三种不同的时序推理模式：
- **PI (Plot Inference)**：给定电影开头+结尾，推断中间——属于 *causal narrative reasoning*
- **FSA (Functional Step Alignment)**：在两个 cooking video 间找到 functionally equivalent 的步骤——属于 *cross-video temporal grounding*
- **PSS (Procedural Step Sequencing)**：把打乱的片段排回正确顺序——属于 *within-video causal ordering*，但因为片段来自同一视频，需要推理 segment 间的 causal dependency

**Multi-view Reasoning** 借用 VisDrone 的 44 对同步无人机视频，专门测试 spatial reasoning 和 cross-perspective object tracking。MSR 涉及相对位置推理，MOC 涉及跨视角计数——这要求模型建立 *3D spatial mental model*。

**CCQA** 是 open-ended，要求模型比较两个 cooking video 的流程差异，覆盖所有 scoring points。

### 2.2 视频来源的多样性

数据从 6 个公开 dataset 汇集：Animal Kingdom、MovieChat-1K、YouCook2、VisDrone、Charades、Assembly101。这种 *heterogeneous sourcing* 是有意为之——避免单一 domain 的 spurious correlation，同时覆盖 7 个 primary categories、32 个 genres，视频长度从 1 分钟到 1 小时以上均有分布（Figure 3c）。

平均每个 query 涉及 ~770 秒视频，最长的 case 跨越小时级别。这与 LongVideoBench（avg 473s）、Video-MME（avg 1017s）相比处于中等偏长区间，但因多视频并存，**实际单视频 token 预算被严重稀释**。

## 三、数据构建 Pipeline 的技术细节

CrossVid 用了一条 **semi-automated multi-stage pipeline**（Figure 4），这是工业界建 benchmark 的标准范式，但里面有几个工程上的 trick 值得拆解。

### 3.1 Frame Captioning

用 **Qwen2.5-VL-72B** 对每个视频做密集 frame-level captioning。关键设计：
- **Temporal grouping**：相邻 2-8 帧组成一组，若原数据集有 timestamp segmentation（如 YouCook2 的 step interval），按其分组——这相当于利用了 *prior knowledge* 给 captioner 提供 localization hint
- **Domain-specific prompts**：cooking 视频聚焦 ingredients/utensils/actions；电影聚焦 plot/character
- **Metadata injection**：把原数据集的 plot summary、action label 一并喂入，作为 contextual grounding

这一步产生的 caption 是后续 QA generation 的唯一视觉信息载体，所以 caption 的质量直接决定了 QA 的上限。

### 3.2 QA Generation with DeepSeek-R1

把同 set 内的 frame captions 喂给 **DeepSeek-R1**，让其生成 QA pair。Prompt 设计有三个关键约束：
1. 必须显式分析视频间的关系
2. 必须匹配任务的具体需求（BU 任务聚焦 action pattern 对比）
3. 必须输出 rationale 解释

DeepSeek-R1 的 thinking 能力在这里被用来做 *video relation reasoning*——这相当于让一个 LLM 替代人类做"出题"工作，把 caption 中的对比关系显式化为 QA。

### 3.3 Filtration 的三道门

- 过滤与视频理解无关的问题
- 过滤只引用单视频的问题（"In video three, what color is the car?"）
- 过滤主观或需哲学推理的问题

第三条很有意思——他们刻意排除了 open-ended interpretation 类问题，保留了 *objective, factual* 的问题。这让 CrossVid 的 evaluation 更稳定，但也限制了 creative reasoning 的评估。

### 3.4 Refinement 中的 Anti-Shortcut 设计

**PSS 任务的 temporal realignment** 是一个值得 build intuition 的 trick：

```
preceding clip 提前 1-5 秒，subsequent clip 延后 1-5 秒
```

这相当于在 clip 边界处制造 *visual feature discontinuity*。原因是：模型可能利用 camera angle continuity、color histogram 平滑度等 low-level cue 排序，而非真正理解 causal content。通过制造 discontinuity，迫使模型依赖 *semantic content* 而非 *low-level visual consistency*。

这与 video understanding 中常见的 *shortcut learning* 是同一个问题——CNN-based model 容易学 background cue，MLLM 也可能学 frame-to-frame 的 color/texture continuity。Temporal realignment 是一个简单有效的 counter-measure。

### 3.5 Multi-view 的人工标注

MSR/MOC 用纯人工标注，因为：
- 物体小，coarse caption 无法提供 fine-grained spatial info
- 需要精确的相对位置关系
- VisDrone 提供了 per-frame bounding box，可作为辅助

具体做法：随机抽 5 个物体组成 object combination，每组视频 100 个 combination，用不同颜色标记，annotator 看标记视频后人工出题。**这是 benchmark 中难得的 *ground-truth spatial annotation*，因为 drone 视频的 bbox 是 metric-level precise 的**。

## 四、评估协议与公式

### 4.1 Frame Sampling 策略

每个 query 的 total frame budget 平均分配给所有视频，每视频内 uniform sampling，resize 到长边 360 pixels 保持 aspect ratio。

这个设计有内在问题——**uniform allocation 忽略了视频的信息密度差异**。一个高动作密度的 cooking video 和一个静态对话场景的视频分配到相同 frame 数，前者会信息丢失严重。这是 ablation study 里 "key frame loss" 错误的根源之一。

### 4.2 FSA 任务的 IoU 评估

FSA 是 closed-ended generation，要求模型输出时间区间 $[A_{\text{start}}, A_{\text{end}}]$，用 **IoU** 与 ground truth $[G_{\text{start}}, G_{\text{end}}]$ 比较：

$$
\text{IoU} = \frac{\max\left(0, \min(A_{\text{end}}, G_{\text{end}}) - \max(A_{\text{start}}, G_{\text{start}})\right)}{\max(A_{\text{end}}, G_{\text{end}}) - \min(A_{\text{start}}, G_{\text{start}})}
$$

**变量含义**：
- $A_{\text{start}}, A_{\text{end}}$：模型输出的开始/结束时间戳（秒）
- $G_{\text{start}}, G_{\text{end}}$：ground truth 的开始/结束时间戳
- 分子：overlap 区间长度，若 $\min(A_{\text{end}}, G_{\text{end}}) \leq \max(A_{\text{start}}, G_{\text{start}})$ 则 overlap 为 0
- 分母：union 区间长度

这是标准 temporal IoU，与 video temporal grounding 任务（如 Charades-STA、ActivityNet-Captions）的评估一致。但 FSA 的 twist 在于——**alignment 是跨视频的**，模型要在 video 2 中找到与 video 1 指定区间 functionally equivalent 的片段。这本质上是一种 *cross-video temporal correspondence* 任务。

### 4.3 CCQA 的 GPT-4.1 Judge

Open-ended 评估用 GPT-4.1 做 judge，分两阶段：
1. **Coverage check**：每个 scoring point 覆盖到给 1 分，否则 0 分
2. **Accuracy check**：已覆盖的 scoring point，细节匹配 standard answer 再加 1 分

最终分数 = (coverage + accuracy points) / (2 × scoring points 数)

这种 *two-tier rubric* 比 single holistic score 更稳定，但仍然有 LLM-as-judge 的 inherent bias 问题。理论上应该用 multiple judges 投票或 human cross-validation。

### 4.4 评估指标汇总

- Single-choice：accuracy
- Multi-choice：必须完全匹配 ground truth（"AB" 对 "ABC" 算错）—— 严格但合理
- PSS：必须在每个位置上匹配 ground truth
- FSA：IoU（threshold 未明说，从结果看应该是 IoU > 0.5 算正确？或者用 IoU 值平均？）
- CCQA：GPT-4.1 rubric scoring

## 五、实验结果的核心发现

### 5.1 总体性能（Table 2）

- **Human**：89.2% overall
- **Gemini-2.5-Pro**：50.4% overall（最佳）
- **GPT-4.1**：45.2%
- **Doubao-1.5-VL-pro**：44.3%
- **GPT-4o**：36.8%
- **Open-source 最佳 GLM-4.1V-9B-Thinking**：35.1%
- **Qwen2.5-VL-72B**：34.4%

人类与最佳 MLLM 之间有 ~39% 的 gap，这是相当大的 headroom。尤其在：
- **FSA**：Gemini-2.5-Pro 13.4% vs Human 85.2%（gap 71.8%）
- **PSS**：Gemini-2.5-Pro 78.2% vs Human 89.9%（gap 11.7%，相对最小）
- **MSR/MOC**：均 ~25-32% vs Human 93%+

### 5.2 三个核心观察

**观察 1：CVR 对所有 MLLMs 都难**
FSA 任务的极端低分（13.4%）值得深究。FSA 要求模型：
1. 理解 video 1 指定区间的 functional content
2. 在 video 2 中 scan 所有区间，找到 functionally equivalent 的
3. 输出精确时间戳

这同时考验了 *fine-grained temporal understanding* + *cross-video semantic matching* + *temporal grounding output format*。现有 MLLMs 的视频 token 表示通常是 sparse pooled features，精确到秒级的时间定位能力天生薄弱，叠加跨视频对比后崩溃。

**观察 2：Closed-source 显著优于 Open-source**

最佳 open-source (GLM-4.1V-9B-Thinking 35.1%) < 最差 closed-source (GPT-4o 36.8%)。这个 gap 在 video understanding 任务上比在 image understanding 上更显著，原因可能是：
- Closed-source 模型训练时使用了大量 *proprietary video data*（YouTube-scale）
- 长上下文 + 多视频并存的 inference 工程优化
- Better visual encoder（Gemini 的 native multimodal vs late-fusion open-source）

**观察 3：Thinking-enabled 模型占优**

- Gemini-2.5-Pro (thinking) > GPT-4.1 / GPT-4o / Doubao (non-thinking)
- GLM-4.1V-9B-Thinking > Qwen2.5-VL-7B / InternVL3-8B
- MiMo-7B (reasoning-tuned) 在 7B 段领先

这暗示 **multi-step reasoning 是 CVR 的核心需求**。CVR 任务天然需要先 *parse each video* → *extract task-relevant features* → *align/compare across videos* → *aggregate and answer*，这种 *multi-hop* 结构正好是 CoT/thinking 的甜区。

## 六、Ablation Studies 的技术启示

### 6.1 Frame Number Impact（Table 3）

Qwen2.5-VL-72B 在 32 / 64 / 128 / 256 frames 下的表现：

| Frames | O.Avg | C.Avg | T.Avg | M.Avg | CCQA |
|---|---|---|---|---|---|
| 32 | 33.8 | 37.0 | 33.8 | 35.1 | 18.9 |
| 64 | 36.9 | 39.8 | 37.4 | 35.9 | 25.9 |
| 128 | 39.1 | 45.7 | 34.5 | 36.4 | 32.0 |
| 256 | 39.5 | 47.5 | 33.9 | 34.9 | 34.0 |

几个 non-trivial 的 patterns：

1. **Comparative Analysis (C.Avg) 单调上升** 37.0 → 47.5，提升 10.5%——更多信息直接帮助对比
2. **Temporal Understanding (T.Avg) 非单调**：64 frame 时最佳 37.4，128 frame 时反而下降到 34.5——**信息冗余反而干扰 temporal reasoning**
3. **Multi-view (M.Avg) 基本持平** —— 因为 drone 视频的物体小，更多 frame 不能解决 spatial resolution 不足问题
4. **CCQA 大幅提升** 18.9 → 34.0，提升 15.1% —— open-ended 任务对 detail 覆盖度最敏感

**Build intuition**：Temporal understanding 任务的非单调性揭示了 video understanding 中一个关键 trade-off——**temporal reasoning 依赖 key events，而非 dense frames**。论文举的例子很形象：32 frame 时模型能识别出 "troop convoy" 和 "negotiation scene"，但 256 frame 时被 "generic shots of injured soldiers" 这类 atmospheric 信息干扰，给出基于 "broad military planning associations" 的错误答案。

这本质上是一个 **signal-to-noise ratio** 问题。更多 frame 既增加 signal 也增加 noise，对 *event-centric reasoning* 任务，noise 增长可能快于 signal。这暗示 *adaptive keyframe selection* 比 *uniform sampling* 更可能是 CVR 的解。

### 6.2 CoT Effectiveness（Table 4）

CoT 三阶段：(1) 理解问题 (2) 逐视频分析 (3) 跨视频聚合回答

| Model | Setup | O.Avg | T.Avg | M.Avg |
|---|---|---|---|---|
| GPT-4.1 | w/o CoT | 45.2 | 46.7 | 38.4 |
| GPT-4.1 | w/ CoT | 44.9 | 48.2 | 40.4 |
| Qwen2.5-VL-72B | w/o CoT | 34.4 | 29.2 | 23.5 |
| Qwen2.5-VL-72B | w/ CoT | 39.5 | 33.9 | 34.9 |
| MiniCPM-o 2.6 | w/o CoT | 25.6 | 26.4 | 31.4 |
| MiniCPM-o 2.6 | w/ CoT | 23.7 | 18.7 | 33.3 |

非常有意思的 findings：

1. **GPT-4.1 几乎无收益**（甚至 O.Avg 微降）—— 推理能力已经内化，显式 CoT 反而可能引导模型走偏
2. **Qwen2.5-VL-72B 大幅收益**（O.Avg +5.1%, M.Avg +11.4%）—— 大模型能从 prompt engineering 中受益
3. **MiniCPM-o 2.6 反而受损**（T.Avg 从 26.4 跌到 18.7）—— 小模型在长 CoT 路径上 error accumulate

这暗示一个重要结论：**CoT 是放大器，不是修正器**。基础能力强的模型通过 CoT 结构化推理获益；基础能力弱的模型在长 reasoning chain 中因中间步骤错误导致雪崩。

类似现象在数学推理任务上也有报告：DeepSeek-R1 paper 显示小模型 RL 训练后 reasoning 长度暴涨但准确率不升反降。

## 七、Error Analysis 的四类错误

论文手动分析 GPT-4.1, MiniCPM-o 2.6, InternVL3-38B, Qwen2.5-VL-72B 在 CoT 下的错误：

**(a) Key frame loss**：uniform sampling 导致关键帧缺失。例：判断 foie gras 是否裹粉，video 2 实际裹了粉但 sampled frames 漏掉这一步，Qwen2.5-VL-72B 答错。

**(b) Video understanding error**：模型抓到了关键 frame，但单视频理解就错了。例：判断 hug 在不同电影中的 contextual meaning，模型成功定位了所有 hug 帧，但对 video 2 的 hug 含义理解错误。

**(c) Cross-video comparison error**：单视频理解全对，但跨视频比较时失败。例：分析 dim light 在制造悬念氛围中的作用，MiniCPM-o 2.6 对每个视频的 lighting 分析都对，但聚合时只做了简单对比，未抓住 cross-video narrative function。

**(d) Format error**：模型理解对了，但输出格式不符合规范。例：FSA 要求输出 "begin,end" 形式，模型输出了自然语言描述。

**核心 insight**：(c) 类错误是 CVR 的本质挑战——单视频能力并不能线性外推到多视频能力。当前 MLLMs 的训练数据几乎全是 *single-video QA*，没有 *cross-video comparison* 的监督信号。这是一种 *capability extrapolation gap*。

## 八、对 MLLMs 架构与训练的启示

从 CrossVid 暴露的问题，可以反推 MLLMs 在 CVR 上需要的几项改进：

### 8.1 Token Budget 管理

当前 multi-video 输入下，每视频的 frame 数被稀释。可能的方向：
- **Video-specific token compression**：根据视频复杂度动态分配 token，高信息密度视频多采样
- **Hierarchical encoding**：先 coarse scan 提取 keyframe candidates，再 fine-grained attend
- **Cross-video token sharing**：相似帧的 visual token 共享，节省预算

### 8.2 Cross-Video Attention

当前 MLLMs 的 self-attention 让所有 video token 互相 attend，但缺乏 *task-guided cross-video attention*。可能的架构改进：
- **Cross-video transformer layers**：显式建模 video 间的对齐关系
- **Contrastive video encoding**：让 encoder 学会提取 video 间的 distinguishing features
- **Video-level memory tokens**：把每个视频压缩为一组 memory token，比较在 memory token 层面进行

### 8.3 Training Data 构造

需要 *multi-video instruction tuning data*：
- 给定 N 个相关视频，question 要求 cross-video comparison
- 训练数据可由 LLM 从同 set 的 captions 合成
- 类似 single-image → multi-image 的发展路径（如 LLaVA-NeXT, Idefics2）

### 8.4 Temporal Grounding 增强

FSA 任务 13.4% 的极低分暴露了 *temporal localization* 能力的不足。当前 visual token 是 pooled feature，丢失了 precise temporal position 信息。可能的改进：
- **Timestamp-aware positional encoding**：把时间戳显式编码到 visual token
- **Dense temporal feature preservation**：保留 video clip 的 fine-grained temporal feature
- **Anchor-based temporal reasoning**：类似 DETR 的 object query，为 temporal span 设计 anchor queries

### 8.5 Spatial Reasoning in Multi-view

MSR/MOC 的低分说明 multi-view spatial reasoning 是短板。VisDrone 的 bird-eye view 与一般 web video 分布差距大，预训练数据中此类视角稀少。可能的方向：
- **3D-aware visual encoder**：把多视角 feature 提升到 3D scene representation
- **Ego-Exo 对齐训练**：用 Ego-Exo4D 类数据做 multi-view pretraining
- **Spatial reasoning CoT**：用文字描述逐步构建 spatial mental model

## 九、与相关工作的关联

### 9.1 Single-Video Benchmarks 的局限

Table 1 把 CrossVid 与 11 个先前 benchmark 对比。关键维度：
- **#Videos**：5,331（中等规模，少于 ActivityNet 5800，多于 Video-MME 900）
- **#QA pairs**：9,015（中等，远少于 NExT-QA 52k）
- **#Tasks**：10（较多，仅次于 MMVU 27, Video-MME 12）
- **Multi-video**：✓（除 CrossVid 外只有 All-Angles Bench, Ego-Exo4D, EgoExoLearn）
- **Multi-view**：✓（与 All-Angles Bench, Ego-Exo4D 并列）

CrossVid 在 *multi-video + multi-view + open-ended* 的组合上独此一家。

### 9.2 与 All-Angles Bench 的差异

All-Angles Bench (Yeh et al. 2025) 是最近的多视角 benchmark，90 scenes, 2132 QA pairs, 6 tasks。CrossVid 相比：
- 视频来源更广（6 dataset vs 单一）
- 任务更丰富（10 vs 6）
- 包含 *cross-video* 而非仅 *multi-view*
- 开放式问答

All-Angles Bench 局限于"同场景多视角"，CrossVid 扩展到"多视频语义关联"，包括跨场景的对比。

### 9.3 与 Long-Context Video 工作

LongVideoBench (Wu et al. 2024) avg 473s, Video-MME avg 1017s, MLVU avg 930s——都聚焦于 single long video。CrossVid avg 770s 但分散在多视频上，token 序列长度上与 long-context 工作相当，但 attention pattern 完全不同——long video 是 *within-video temporal dependency*，CVR 是 *cross-video semantic dependency*。

这两者其实是正交的。一个理想的 video MLLM 应该同时擅长 single long video 和 cross-video reasoning，但目前两者都未达到人类水平。

### 9.4 与 Reasoning Models 的兴起

Video-R1, GLM-4.1V-Thinking, MiMo, Kimi-VL-A3B-Thinking 等 reasoning-tuned MLLM 在 CrossVid 上表现突出，呼应了 LLM 领域 o1/R1 的趋势。这暗示 video understanding 的下一步可能不是单纯 scale up visual encoder，而是 *reasoning capability 的迁移*。

参考 Video-R1 paper: https://arxiv.org/abs/2503.21776
参考 GLM-4.1V-Thinking: https://arxiv.org/abs/2507.01006

## 十、Limitations 与潜在问题

虽然 CrossVid 是一个有价值的 benchmark，但有几个值得讨论的局限：

1. **Domain bias**：视频源偏向 cooking（YouCook2 在 5 个任务中出现）、movie（MovieChat-1K）、assembly。科学、医疗、体育等领域覆盖不足。

2. **English-only**：从 prompt 看是英文为主，多语言 CVR 能力未评估。

3. **Closed-source eval 不可复现**：GPT-4.1, Gemini-2.5-Pro, Doubao 的 frame 数（如 GPT-4.1 < 50, Gemini 128）通过 API 调用，实际 API 处理 frame 的方式（如 internal frame compression）不透明，公平性存疑。

4. **Open-ended 评估的 LLM-as-judge bias**：CCQA 用 GPT-4.1 评分，GPT-4.1 本身又是被评估对象——self-judge bias。理想做法是用一个未参与被评估的模型做 judge，或多人 human eval。

5. **FSA 评估阈值未明**：论文未说明 FSA 是用 IoU 平均值还是 IoU > threshold 的命中率，这影响结果可比性。

6. **Multi-video 输入格式的工程细节**：是把多个视频的 frame concat 后一起输入，还是用某种 separator token？这影响模型对 video boundary 的感知。

## 十一、对未来研究的方向建议

基于 CrossVid 暴露的问题，我认为几个有潜力的研究方向：

### 11.1 Multi-Video Instruction Tuning

类似 LLaVA-NeXT-Interleave 把 multi-image 训练扩展到 MLLMs，可以构造 multi-video instruction dataset：
- 跨视频比较
- 跨视频事件对齐
- 跨视频视角转换

### 11.2 Memory-Augmented CVR

人类做 CVR 时会建立 *working memory* 跟踪每个视频的关键信息，然后比较。MLLMs 可以引入：
- Per-video memory bank
- Cross-video attention with memory retrieval
- 类似 RAG 的 video retrieval + reasoning

### 11.3 Test-Time Compute for Video

借鉴 OpenAI o1 的 test-time compute scaling，在 CVR 任务上让模型自适应分配推理 budget。当前 CoT study 显示大模型受益，但小模型受损——可能需要 *adaptive reasoning depth*。

### 11.4 Video-Centric Reward Modeling

用 RL 训练 MLLM 的 video reasoning，类似 Video-R1 但扩展到 cross-video setting。Reward signal 可以来自 CrossVid 类 benchmark。

### 11.5 Fine-Grained Temporal Representation

FSA 13.4% 的极低分指向 temporal representation 的根本问题。值得探索：
- Video clip token with explicit timestamp embedding
- Hierarchical temporal pooling
- Sparse but precise temporal feature preservation

## 十二、我的个人 Take

CrossVid 是一个 timely 的 benchmark，抓住了 video understanding 演进的下一个 natural step。从 single-image → multi-image → single-video → multi-video 的演进路径看，multi-video 是必然方向。

但我也观察到几个值得关注的点：

1. **CVR 是 single-video 能力的延伸而非替代**。CrossVid 暴露的很多问题（temporal grounding 弱、spatial reasoning 弱、long context 下的 attention dilution）在 single-video benchmark 上已存在，CVR 只是放大了这些问题。把 CVR 做好很可能不需要新架构，而是把 single-video 能力做扎实。

2. **Token budget 分配是关键工程问题**。当前 uniform sampling 是 baseline，真正的 win 可能来自 *learned frame selection* 或 *task-adaptive sampling*。这与 retrieval-augmented generation 的思路相似——在推理前先 retrieve 关键 frames。

3. **Multi-view reasoning 的特殊性**：MSR/MOC 的低分部分源于 VisDrone 这种 bird-eye view 在预训练分布外。这暗示 *view invariance* 可能需要专门的 pretrain stage，类似 depth estimation 的 supervised pretrain。

4. **Reasoning model 的 CVR 优势**：thinking-enabled 模型在 CVR 上的优势比在 single-video 上更显著，这与 CVR 任务的多步结构吻合。这预示 *video reasoning models*（类比 math reasoning models）会是接下来的一个研究方向。

5. **评估 LLM-as-judge 的可靠性**：CrossVid 用 GPT-4.1 做 CCQA judge，但 GPT-4.1 是被评估模型。这个 circular dependency 应该在后续工作中被修正。

6. **缺少 retrieval/augmentation 探索**：论文未探索 RAG-style video retrieval、memory-augmented reasoning 等思路。一个合理的实验是：先让模型 select keyframes from all videos，再基于 selected frames 推理——可能能显著缓解 token budget 不足问题。

7. **Missing baseline**：理想上应该有一个 oracle 实验——给模型完美 caption 而非 frame，看模型 pure LLM reasoning 能达到多少。这能 disambiguate "visual perception limit" vs "reasoning limit"。论文未做这个 ablation，让读者无法知道瓶颈到底在哪。

8. **Long context video model 的盲点**：当前 long-context MLLM（如 LongVA, Video-XL）在 CVR 上表现并不突出——LongVA-7B-DPO 仅 18.0%。这暗示 *long context within a video* 与 *multi-video aggregation* 是两种不同的能力，前者训练数据中不一定包含后者。

## 十三、结语

CrossVid 是 2025 年 video understanding 领域一个有标志意义的 benchmark，它把 multi-video reasoning 从 niche 子问题提升为主流评估维度。从 50.4% 的最佳模型 vs 89.2% 的人类基线看，CVR 还处于早期阶段。这篇 paper 的真正价值可能在于：

- 揭示了 *multi-video capability* 与 *single-video capability* 的非线性关系
- 暴露了 *temporal grounding* 和 *cross-video comparison* 的根本短板
- 为 reasoning model 在 video 领域的应用提供了 motivation

接下来 12-18 个月，我预期会看到：
1. 多个 multi-video instruction tuning dataset 出现
2. Video reasoning model（o1-style）在 CVR 上大幅超越 naive MLLM
3. Cross-video attention mechanism 的架构创新
4. CVR 上的 RL training（类似 RLHF 但 reward 来自 benchmark accuracy）

CrossVid 提供了一个清晰的 measurement framework，剩下的就是 engineering 和 modeling 上的推进。

参考资料：
- CrossVid GitHub: https://github.com/chuntianli666/CrossVid
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- All-Angles Bench: https://arxiv.org/abs/2504.15280
- Video-MME: https://arxiv.org/abs/2405.21075
- LongVideoBench: NeurIPS 2024
- Video-R1: https://arxiv.org/abs/2503.21776
- GLM-4.1V-Thinking: https://arxiv.org/abs/2507.01006
- Ego-Exo4D: CVPR 2024
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- YouCook2: https://arxiv.org/abs/1803.05266
- Assembly101: CVPR 2022
- MovieChat: CVPR 2024
