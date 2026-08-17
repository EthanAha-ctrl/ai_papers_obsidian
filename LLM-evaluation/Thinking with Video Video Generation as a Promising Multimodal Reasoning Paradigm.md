---
source_pdf: Thinking with Video Video Generation as a Promising Multimodal Reasoning
  Paradigm.pdf
paper_sha256: 5067dbabb435b0c5c5b9b03ada7aa32629bdef033f488e337041f1d8051983bb
processed_at: '2026-08-12T15:37:43-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇paper说白了，就是复旦的团队想搞清楚一个问题：像 Sora-2 这种能生成视频的模型，它到底有没有"脑子"？它能用来做 reasoning 吗？

为了测这个，他们弄了一个叫 VideoThinkBench 的测试集，去考 Sora-2。结果发现 Sora-2 在几何题上居然能打败 GPT-5 和 Gemini 2.5 Pro，但在数学题上的"聪明"其实是作弊。我把最核心的几个点用大白话拆给你看。

---

### 1. 核心发现：把"时间"当成"置信度"来用

这是这篇paper最天才的idea。

你考 Sora-2 一个几何题，比如"这三条线延长后交于哪个点？"（Ray Intersection）。你让它生成一个10秒的视频，在正确答案上画个红点。

你怎么判断它选了哪个答案？通常的做法是看视频最后一帧。但这篇paper发现，Sora-2 经常在视频结尾"抽风"，突然闪黑屏或者变成测试色条。而且单看一帧，随机性太大。

所以他们发明了 **Major Frame Evaluation**。每5帧抽1帧，看每一帧红点在哪个选项上，最后少数服从多数投票决定。

数学表达就是：
$$ \text{Answer} = \arg\max_{o \in \{A,B,C,D,E\}} \sum_{t \in \text{Frames}} \mathbb{1}[\text{Red\_Pixel\_at}(t) == o] $$

这里的 $o$ 是选项 A 到 E，$t$ 是抽取的具体某一帧，$\mathbb{1}$ 是指示函数（如果红点在第 $t$ 帧落在了选项 $o$ 上，就记1，否则记0）。$\arg\max$ 就是找哪个选项 $o$ 被选中的次数最多。

**Intuition**: 这就好比问一个人一百遍同一个问题，虽然每次回答都有随机性，但如果他心里真知道答案，他答得最多的那个肯定是对的。Sora-2 如果真"懂"这道题，它生成的视频里，红点会在多帧中稳定地停留在正确选项上。这种 temporal consistency（时间一致性）就是它内心"确信度"的 proxy。

看这个数据对比（Arc Connect 谜题）：
- 看最后一帧：56% 准确率
- 投票看多帧：68%
- 投票多帧 + 生成5次视频再投票：**90%**

这个 90% 就是把 test-time scaling（测试时间增加算力）的思路搬到了视频生成上。生成5个视频，每个视频抽多帧投票，直接把准确率拉飞。这说明视频生成模型的 reasoning 信号是散落在时间维度和多次采样里的。

---

### 2. Sora-2 真的会做数学题吗？其实是外部大脑在帮它

Paper 里有个极其诚实的实验，直接把"Sora-2 会做数学"的神话戳破了。

他们发现 Sora-2 在 GSM8K（小学数学）上准确率高达 98.9%，甚至 MMMU（大学多学科题）能到 75.5%。难道视频生成模型自己涌现出了数学能力？

为了查清楚，他们拿开源的 Wan 2.5 视频生成模型做实验。Wan 2.5 有个开关参数叫 `prompt_extend`，其实就是个前置的 prompt rewriter 模型。

实验结果极度打脸：
- **关掉 rewriter**：GSM8K 准确率 **0%**，MMLU 准确率 **0%**。
- **开着 rewriter**：GSM8K 准确率 81.6%，MMLU 准确率 74.1%。

为什么？因为当你输入一句 "小明有3个苹果..."，视频生成模型根本不知道怎么生成解题过程的视频。但 prompt rewriter 会先把这道题解出来，然后把 prompt 改写成视觉指令：
> "一只手写下 '苹果 = 3'。接着写 '吃掉1个'。最后写 '答案是 2'。"

Sora-2 内部大概率也有这么个玩意。所以 Sora-2 生成的数学解题视频，90%以上的情况是：**视频里只有正确答案，解题过程是一坨乱码或者根本不make sense。** 论文里手动检查了115个做对的题，只有13.91%的过程是完全正确且可读的，43.48%的解题过程是"Unreadable or Incorrect Logic"（写得太乱或逻辑全是错的）。

**Intuition**: 视频生成模型本质是个"画笔"，不是"大脑"。你让它做数学题，它其实做不到，必须有另一个"大脑"（大概率是个 VLM）先算好答案，然后把分步脚本喂给它，它负责把这些脚本"演"出来。所以 text-centric reasoning 这条路，对于当前的 video generation model 来说是个 illusion。

---

### 3. 那它到底在哪方面比 LLM 强？画图推理

既然做数学题是靠外挂，那 Sora-2 自己擅长啥？答案是 **dynamic spatial reasoning**（动态空间推理）。

在 Eyeballing Puzzles（几何目测谜题）上，Sora-2 的 Major Frame 投票准确率是 **40.2%**，打败了 Claude 4.5 (35.1%)、GPT-5 (29.7%)、Gemini 2.5 (26.5%)。

最变态的一个任务是 Ray Intersection（延长三条线找交点），Sora-2 拿了 **88%**，而 GPT-5 只有 16%。

为什么？因为像 GPT-5 这种 VLM，它只能盯着静态图片"看"，要在脑子里硬算这三条线延长后交在哪。但 Sora-2 是个视频模型，它直接在视频里"画"出这三条线的延长线，看着它们相交。这就是 paper 提出的 "Thinking with Video" 范式真正的杀手锏：**把 imagination 变成可视化过程**。

人类做几何题也是这样在草稿纸上画辅助线的，视频模型天然具备这种动态模拟的物理直觉，这是纯 text-based LLM 绝对做不到的。

---

### 4. 总结一下 Intuition

这篇 paper 其实讲了一个很清晰的故事：
1. Video generation model 自己做数学题是不行的，那是前置 prompt rewriter 在做题。
2. 但如果任务是"通过画图、模拟物理过程来推理"，Sora-2 这种模型就有天然优势。
3. 评价视频模型的 reasoning，不能看最后一帧，得用 temporal voting。时间维度上的稳定性，等于模型的 confidence。

这篇 paper 最值得你关注的技术点就是把 LLM 的 self-consistency 搬到了视频生成上。把生成5个视频、每个视频抽10帧做 majority vote 当作一种 test-time scaling，这个思路非常有潜力。

**Links for reference:**
- Paper Website: https://thinking-with-video.github.io
- Code: https://github.com/tongjingqi/Thinking-with-Video
- Dataset: https://huggingface.co/datasets/fnlp/VideoThinkBench
- Wan 2.5 模型: https://arxiv.org/abs/2503.20314

---

# Thinking with Video: 深入技术讲解

Andrej，这篇paper很有意思——它本质上是在做一个很aggressive的claim：video generation model（具体是Sora-2）不只是"生成视频的工具"，而是一个潜在的 **unified multimodal reasoner**。让我把它的技术骨架拆给你看。

---

## 1. 论文的核心 positioning 和 motivation

paper的positioning是建在一个递进的 reasoning paradigm 谱系上的：

```
Thinking with Text (CoT)  →  Thinking with Images (o3/o4-mini)  →  Thinking with Video (本文)
```

它指出了前两个paradigm的两个 **structural limitations**：
- **Static constraint**: image 只能capture单个时刻，无法表达dynamic process（比如光线反射的延伸、几何构造的连续过程）
- **Modality separation**: text 和 vision 被当作两个分离的modality处理，unified understanding 和 generation 无法自然融合

这里的intuition其实是：人类在做几何题、空间推理时，脑子里是真的在"画"的——你在心算光线反射路径时，是在mental simulation一个动态过程。Sora-2这种video generation model天然就是temporal的，所以可能更接近这种cognitive process。

但这个claim很危险——因为video generation model的内部机制不透明，到底是真的"在推理"，还是prompt rewriter在前面把活全干了？paper后面会专门分析这个问题（Section 3.2.3），这是这篇paper比较诚实的地方。

---

## 2. VideoThinkBench 的架构设计

Benchmark分两大类，总共 **4,149 samples**：

### 2.1 Vision-Centric Tasks（2,696 samples）
- **Spatial Reasoning**（1,200）: Eyeballing Puzzles（1,050）+ Mazes（150）
- **Inductive Reasoning**（1,496）: ARC-AGI-2（1,000）+ Visual Puzzles（496）

### 2.2 Text-Centric Tasks（1,453 samples）
- Text-Only Math（345）: GSM8K / MATH-500 / AIME24 / AIME25
- Text-Only General Knowledge（739）: BBH / MMLU / MMLU-Pro / GPQA / SuperGPQA
- Multimodal（369）: MathVista / MathVision / MMBench / MMMU

---

## 3. Eyeballing Puzzles 的技术细节——这是paper最有意思的部分

### 3.1 Task设计
21个geometric construction tasks，分三类：
- **Point Tasks**（9个）: Circle Center, Circumcenter, Fermat Point, Incenter, Midpoint, Orthocenter, Point Reflection, Ray Intersection, Triangle Center
- **Line Tasks**（8个）: Angle Bisector, Arc Connect, Circle Tangent Line, Circle Tangent Point, Parallel, Perpendicular, Perpendicular Bisector, Ray Reflection
- **Shape Tasks**（4个）: Isosceles Trapezoid, Parallelogram, Right Triangle, Square Outlier

每个task生成50个multiple-choice samples（A-E），所以是 21 × 50 = 1050 samples。**关键设计是这些任务都是verifiable的**——可以用程序自动判分，不依赖人工judgement。

### 3.2 三种Evaluation Mode的数学细节

这是这篇paper在evaluation methodology上最有insight的部分。Sora-2的输出是video + audio，所以他们设计了三种独立的decoding方式：

**Audio Evaluation**:
prompt要求Sora-2用phonetic alphabet（"Alpha", "Bravo", "Charlie", "Delta", "Echo"）说出答案，然后：
```
Audio → whisper-1 transcription → 提取第一个出现的phonetic word → 与ground truth比较
```

**Last Frame Evaluation**:
prompt要求Sora-2在正确选项上画红点，然后：
```
Last frame → 计算red pixels的average coordination (x̄, ȳ)
→ 选择离(x̄, ȳ)最近的option → 与ground truth比较
```
具体来说，对一个frame $I$，red pixel的average coordination是：
$$\bar{x} = \frac{\sum_{(x,y): R(I_{x,y}) > \tau} x}{|P_R|}, \quad \bar{y} = \frac{\sum_{(x,y): R(I_{x,y}) > \tau} y}{|P_R|}$$
其中 $P_R$ 是red pixel集合，$\tau$ 是红色阈值。

**Major Frame Evaluation**（这个最重要）:
每5帧抽1帧，对每帧独立做Last Frame的判断，然后majority vote（排除"None"option）：
$$\text{Major}(o) = \arg\max_{o \in \{A,...,E\}} \sum_{t \in T} \mathbb{1}[f_t = o]$$
其中 $T$ 是抽取的frame集合，$f_t$ 是第$t$帧的判断结果。

### 3.3 关键数据解读

看Table 2的Overall Average：
- Sora-2 Audio: 28.0%
- Sora-2 Last Frame: 33.4%
- Sora-2 Major Frame: **40.2%** ← 最高
- Gemini 2.5 Pro: 26.5%
- GPT5 High: 29.7%
- Claude Sonnet 4.5: 35.1%

**这里的intuition**：Major Frame（40.2%）显著高于Audio（28.0%）和Last Frame（33.4%）。这说明video generation的temporal consistency是一个signal——model如果真的"理解"了问题，它会在多帧上稳定地指向同一个option。而单帧的noise被temporal aggregation过滤掉了。

特别值得注意的Ray Intersection任务：Sora-2 Major Frame达到 **88%**，而Gemini/GPT5/Claude只有22%/16%/22%。这个task需要model"extend three black lines and find intersection"——这是一个真正的dynamic reasoning task，需要model在video里"画线延伸"，这正是video generation model的天然优势。

### 3.4 Self-Consistency 实验（Table 6）

这是Arc Connect puzzle上的实验，非常漂亮的test-time scaling demonstration：

| Evaluation Method | Single Try | Vote (5 Tries) |
|---|---|---|
| Audio | 12% | 12% |
| Last Frame | 56% | 66% |
| Major Frame | 68% | **90%** |

**Intuition**: Major Frame + 5-try voting从68% → 90%，这个跳跃非常大。这说明video generation model的"reasoning signal"是分布在temporal dimension和multiple samples中的。这和LLM里的self-consistency（Wang et al. 2022）是同一个principle，但在video generation里更显著——因为video本身就是多维的temporal signal。

paper原文一句话很关键："the temporal consistency within a single video is a strong proxy for the model's confidence"——这其实就是把LLM的logit confidence换成了temporal consistency。

---

## 4. Visual Puzzles 的Deviation Metric

paper定义了一个pixel-level的metric来量化生成frame和solution image的差距：

$$\text{Diff} = \sum_{(x,y) \in \text{PuzzleArea}} \delta(\text{Pixel}_{\text{gen}}(x,y), \text{Pixel}_{\text{gt}}(x,y))$$

其中 $\delta$ 根据task type不同：

**Color-filling tasks** 用RGB空间的Euclidean distance:
$$\delta_{\text{color}}(p, q) = \sqrt{(p_r - q_r)^2 + (p_g - q_g)^2 + (p_b - q_b)^2}$$
$p_r, p_g, p_b$ 是generated pixel的RGB分量，$q_r, q_g, q_b$ 是ground truth的RGB分量。

**Shape-drawing tasks** 用binarized coverage difference:
$$\delta_{\text{shape}}(p, q) = \begin{cases} 1, & \text{if } \text{Binarize}(p) \neq \text{Binarize}(q) \\ 0, & \text{otherwise} \end{cases}$$
其中 Binarize 用threshold 245（intensity > 245 → white = 255, else black = 0）。

**Best Frame Selection**：选Diff最小的那帧作为"answer frame"。这个设计很聪明——因为Sora-2可能在task完成后生成无关内容，或者target color/shape是gradually出现的，所以用"best frame"比"last frame"更robust。

看Table 3的数据：Sora-2在Color-Filling上67.0%，在Shape-Drawing上64.9%。Shape-Drawing接近Claude 4.5（68.6%）——而Claude是被给了multiple-choice options的，Sora-2没有。这是inductive reasoning ability的一个indication。

---

## 5. ARC-AGI-2 实验——Few-Shot Learning的evidence

ARC-AGI-2是François Chollet的抽象推理benchmark，需要model从input-output pairs里induce出transformation rule并apply到新input。

### 5.1 Pixel Accuracy的Few-Shot vs 1-Shot实验（Table 7）

| Accuracy Range | Few-Shot | 1-Shot |
|---|---|---|
| 0.00-0.35 | 743 | 788 |
| 0.35-0.65 | 127 | 117 |
| 0.65-1.00 | 130 | 95 |

**Intuition**: Few-Shot比1-Shot在high-accuracy bucket（0.65-1.00）多35个samples，在low-accuracy bucket（0-0.35）少45个samples。这证明Sora-2是一个 **few-shot learner**——它从更多examples里能induce出更好的pattern。

### 5.2 Manual Categorization（Table 10）

100个random samples的分类：
- Fully Correct: 3
- Mostly Correct: 14
- Partially Correct: 28
- Wrong (Did Nothing): 42
- Wrong (Others): 13

**关键insight**: 14个"Mostly Correct"说明model有时候能识别transformation rule，但在execution上有问题。42个"Did Nothing"很有意思——model没有修改output area，而是修改了input area或few-shot example area，这说明它对instruction的理解有困难。

paper还观察到一个self-correction现象（Figure 4）：第一middle frame里3个green pixels错位了1 pixel向上，下一帧自动correct了。这暗示video generation model内部有某种error correction机制——可能是temporal coherence的副产品。

---

## 6. Text-Centric Tasks 的关键设计

### 6.1 Dual-Modality Evaluation

paper用了 $V \cap A$ 和 $V \cup A$ 两个指标：
$$V \cap A = P(\text{video correct} \land \text{audio correct})$$
$$V \cup A = P(\text{video correct} \lor \text{audio correct})$$

**Intuition**: Audio准确率通常高于Video（因为Sora-2的written content generation能力有限）。$V \cup A$ 给出"如果至少一个modality对"的upper bound，$V \cap A$ 给出"两个都稳定对"的lower bound。

### 6.2 关键数据（Table 5）

- GSM8K: Audio 98.9%（V∪A = 98.9%）
- MATH-500: Audio 92.0%（V∪A = 94.0%）
- MMMU: Audio 69.2%（V∪A = 75.5%）
- MathVista: Audio 75.7%（V∪A = 81.1%）—— **超过了Gemini 2.5 Pro的70.0%**

这个MathVista的结果很impressive，说明在multimodal math上Sora-2有竞争力。

---

## 7. Source of Ability 分析——这是paper最诚实的部分

### 7.1 Data Leakage Check（Table 8）

用Qwen3-235B和Gemini 2.5 Pro从原题派生出"同结构但不同数值"的新题，在GSM8K和MATH-500上重测。派生题性能和原题几乎一致（GSM8K派生版100% vs 原版98.9%），排除data leakage。

### 7.2 Reasoning Process Analysis（Figure 6）

对115个"答案对"的case手动分析reasoning process：
- Completely Correct: 13.91%
- Logic Correct with Writing Errors: 29.57%
- Unreadable or Incorrect Logic: **43.48%**
- Missing Solution Process: 11.30%
- Process Unnecessary: 1.74%

**关键insight**: 43.48%的case，虽然最终答案对，但reasoning process是unreadable或logically wrong。这说明Sora-2的"text-centric reasoning"很大程度是"答案直接出现"，而不是真正的step-by-step reasoning。

### 7.3 Prompt Rewriter的smoking gun（Table 9）

这是paper最关键的发现。他们用Wan2.5（开源video generation model）做了一个对照实验，因为Wan2.5有 `prompt_extend` 参数可以开关prompt rewriter：

| Dataset | Prompt Rewrite | Last Frame | Audio | V∩A | V∪A |
|---|---|---|---|---|---|
| GSM8K | ✗ | 0.0 | 0.0 | 0.0 | 0.0 |
| GSM8K | ✓ | 78.4 | 31.9 | 29.2 | 81.6 |
| MMLU | ✗ | 0.0 | 0.0 | 0.0 | 0.0 |
| MMLU | ✓ | 74.1 | 50.0 | 50.0 | 74.1 |
| MMMU | ✗ | 2.0 | 0.0 | 0.0 | 2.0 |
| MMMU | ✓ | 47.0 | 14.0 | 11.0 | 50.0 |

**Smoking gun**: 关闭prompt rewriter后，accuracy几乎归零。这意味着video generation model本身并没有text reasoning能力——是 **prompt rewriter（很可能是一个VLM）在前面把reasoning problem solved了，然后转换成step-by-step visual instructions给video generator执行**。

看Appendix C.3.1的例子，原始prompt：
> "There are 6 girls in the park. If there are twice the number of boys..."

被rewriter重写成：
> "A hand writes 'Girls = 6'. Then 'Boys = 2 × 6 = 12' appears. Next 'Total kids = 6 + 12 = 18' is written..."

**这彻底解构了text-centric reasoning的claim**——Sora-2的"text reasoning"本质是prompt rewriter做了CoT，video generator只是"visualize"了这个CoT。

---

## 8. 我的critical assessment

这篇paper有几个亮点和几个问题：

**亮点**:
1. Benchmark设计technically sound——verifiable tasks, multiple evaluation modes, dual-modality metrics
2. Self-consistency在video generation上的应用是novel的——temporal consistency as confidence proxy这个idea很有价值
3. 对source of ability的honest analysis（特别是prompt rewriter实验）
4. Eyeballing Puzzles的Major Frame evaluation揭示了一个真实的signal

**问题**:
1. **Vision-centric tasks的Major Frame advantage可能被over-interpreted**——temporal consistency高不一定意味着"reasoning"，可能只是generation的temporal smoothing
2. **Text-centric reasoning的claim essentially被自己的实验refute了**——Wan2.5的实验强烈暗示Sora-2的text reasoning也来自prompt rewriter，但paper没有直接验证Sora-2是否也有prompt rewriter
3. **ARC-AGI-2的1.3% accuracy** 在统计上几乎和random noise无法区分，manual categorization的"14个mostly correct"在100个sample里是subjective的
4. 没有open-source model的parallel实验来validate Sora-2的internal mechanism claim

**Build your intuition的takeaway**:
- Video generation model确实有 **structural advantage** 在dynamic spatial reasoning上（Ray Intersection 88%是硬证据）
- 但它的"text reasoning"很可能是 **illusion from prompt rewriting pipeline**，不是video generator本身的能力
- "Thinking with Video"作为paradigm的真正价值在vision-centric dynamic reasoning，不在text reasoning
- Self-consistency across temporal frames是一个genuinely useful的test-time scaling technique for video generation

**Links**:
- Paper website: https://thinking-with-video.github.io
- GitHub: https://github.com/tongjingqi/Thinking-with-Video
- Benchmark: https://huggingface.co/datasets/fnlp/VideoThinkBench
- ARC-AGI-2: https://arxiv.org/abs/2505.11831
- Self-consistency原paper: https://arxiv.org/abs/2203.11171
- Video models as zero-shot reasoners (相关work): https://arxiv.org/abs/2509.20328
- Wan video generation: https://arxiv.org/abs/2503.20314

这个paper最值得关注的不是"text reasoning"那个方向（那个基本被prompt rewriter实验debunk了），而是 **vision-centric dynamic reasoning + temporal self-consistency**这个组合——这可能是video generation model真正能做出unique贡献的niche。
