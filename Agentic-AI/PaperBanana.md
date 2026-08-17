---
source_pdf: PaperBanana.pdf
paper_sha256: 1898ee713401f759f6e239c4287881921085a015eee80fa5c716a5bb83e2a142
processed_at: '2026-08-06T02:04:08-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PaperBanana 人话版：AI 帮你画论文里的图

## 这玩意儿到底在解决什么问题？

你写完一篇 AI paper，method section 写了 3000 字，接下来最痛苦的事来了——你要画 Figure 1，就是那个 "Overview of our framework" 的图。

过去你只有两个选择：
- 用 TikZ 或 Python-PPTX 写代码画，能精确控制但画不出 NeurIPS 2025 那种带 cute robot icon、3D tensor、pastel 配色的精致 diagram
- 用 PowerPoint 或 draw.io 手工拖，能画好看但要花你一整天，而且 draw.io 那种 default 蓝橙配色一看就很 amateur

PaperBanana 想做的事就是：**你把 method section 的文本 + figure caption 丢给它，它直接吐出一张 publication-ready 的图**。

项目主页：https://dwzhu-pku.github.io/PaperBanana/

## 核心思路：把"画图"拆成五份工作

你可以把 PaperBanana 想象成一个设计公司，里面有五个员工，各管一摊：

### 员工 1：Retriever（资料员）

你要画一个 "Agent Framework" 的图，Retriever 就去公司的图库（292 张 NeurIPS 2025 的 reference diagram）里翻，找出 10 张风格最像的给你参考。

有意思的是，Retriever 找参考时，**图的视觉结构比研究主题更重要**。你要画 "Agent Framework"，一张 "Vision Framework" 的图比一张 "Agent Performance Bar Chart" 更有用——因为前者和你要画的图结构一样（都是 framework），后者虽然 topic 相同但完全不是一个画法。

### 员工 2：Planner（策划）

Planner 是大脑。它看你给的 method section 文本 + caption，再参考 Retriever 找来的 10 张样图，写出一大段超详细的"画图说明"。

这段说明长什么样呢？比如要画 Figure 2，Planner 的输出大概是：

"最左边放两个 icon，上面是 document 代表 Source Context，下面是 target 代表 Communicative Intent。中间偏左是一个浅蓝色圆角矩形框，标题叫 Linear Planning Phase，里面从左到右放三个 robot agent..."

从 content 到 layout 到 color 到 line style 全都写清楚。这段文本就是后续画图的蓝图。

### 员工 3：Stylist（美术指导）

Planner 写的说明可能在美学上不够 "NeurIPS 2025"。Stylist 手里有一本自动总结出来的《NeurIPS 2025 Aesthetic Guideline》，里面写了：

- Background 用 cream (#F5F5DC)、pale blue (#E6F3FF)、mint (#E0F2F1) 这种淡雅色，opacity 10-15%
- Trainable module 用暖色（红、橙），Frozen module 用冷色（灰、冰蓝）
- Process node 一律用 rounded rectangle，corner radius 5-10px
- Label 用 sans-serif（Arial、Roboto），math variable 用 serif italic（Times New Roman）
- Agent paper 可以用 cute robot avatar，theoretical paper 要 minimalist

Stylist 拿这本指南把 Planner 的初稿描述"润色"一遍，让它符合 NeurIPS 2025 的现代审美。注意 Stylist 只管美不管内容，不会动你的逻辑。

### 员工 4：Visualizer（画师）

Visualizer 拿着 Stylist 润色后的描述，交给 Nano-Banana-Pro（Google 的 image generation model）画出来。就是公式 (6)：

$$I_t = \mathrm{Image\text{-}Gen}(P_t)$$

变量解释：$I_t$ 是第 $t$ 轮生成的 image，$P_t$ 是第 $t$ 轮的 description，下标 $t$ 表示 iteration 轮数。$P_0 = P^*$ 就是 Stylist 输出的初始描述。

### 员工 5：Critic（质检员）

Visualizer 画完第一版，Critic 上场。它拿着画出来的图，对比你原始的 method section 和 caption，检查：

- 有没有凭空编造 method section 里没提的 module？（Major Hallucination）
- 箭头方向对不对？数据流有没有反？（Logical Contradiction）
- 有没有超出 caption 范围画多余的东西？（Scope Violation）
- 有没有乱码文字、broken LaTeX？（Gibberish Content）

发现问题后，Critic 改进 description，公式 (7)：

$$P_{t+1} = \mathrm{VLM}_{\mathrm{critic}}(I_t, S, C, P_t)$$

变量解释：$P_{t+1}$ 是改进后的 description，$I_t$ 是当前轮生成的 image，$S$ 是 source context（method section），$C$ 是 communicative intent（caption），$P_t$ 是当前轮的 description。下标 $t \to t+1$ 表示一轮迭代。

改完的 $P_{t+1}$ 喂回 Visualizer 重新画。这个 loop 跑 3 轮，最终输出 $I = I_T$（$T=3$）。

## 为什么这个设计能 work？三个关键直觉

### 直觉 1：解耦 "画什么" 和 "怎么画"

传统做法是把 content 和 style 揉在一起让模型一次生成。PaperBanana 的洞察是：**Planner 管 content，Stylist 管 style**，两者解耦。

Style guide 是从 292 张 reference diagram 自动总结的 reusable asset，inference 时不重新生成，每次都用同一本。这样 Planner 可以专注理解 method section 的逻辑，不被美学决策干扰。

### 直觉 2：Retrieval 的价值在于"看到 academic diagram 长什么样"而非精确匹配

Ablation study 里有个反直觉发现：random retriever（随便挑 10 张 reference）和 semantic retriever（精确匹配）性能几乎一样。

| Retriever 类型 | Overall Score |
|---|---|
| Semantic Retriever | 49.2 |
| Random Retriever | 48.3 |
| No Retriever | 44.2 |

这说明 in-context learning 在这里起的作用是"让 Planner 看到 academic diagram 的大致模样"，不需要找到完全匹配的 case。就像你让一个设计师画 logo，给他看 10 个 Apple 的 logo 和给他看 10 个随便什么公司的 logo，效果差不多——关键是让他"进入 logo 设计的思维模式"。

但没有 retriever 性能明显下降（44.2），说明这个"看到样例"的 anchor 是必要的。

### 直觉 3：Stylist 和 Critic 互补，一个加美学一个保内容

Ablation 里最精彩的对比是：

| 配置 | Faithfulness | Conciseness | Aesthetics |
|---|---|---|---|
| 有 Stylist 无 Critic | 30.7 | 79.2 | 72.1 |
| 无 Stylist 无 Critic | 39.2 | 61.7 | 67.4 |
| 有 Stylist 有 Critic (3 轮) | 45.8 | 80.7 | 72.1 |

Stylist 让 Conciseness 涨了 17.5%（61.7 → 79.2），但 Faithfulness 跌了 8.5%（39.2 → 30.7）。为什么？因为 Stylist 倾向于 simplify 来符合美学，而 simplify 会丢技术细节。

Critic 的作用就是把丢掉的细节找回来：Faithfulness 从 30.7 拉回 45.8（+15.1），同时保住 Stylist 带来的 Conciseness 和 Aesthetics 收益。

这就是 agentic design 的精髓：**单个 agent 优化单一目标会破坏其他目标，多 agent 协作才能实现 multi-objective balance**。

## 效果到底有多好？

### Methodology Diagram（Table 1 主结果）

拿 PaperBanana w/ Nano-Banana-Pro 和 Vanilla Nano-Banana-Pro（直接让 image model 画，不加任何 agent）对比：

| 维度 | Vanilla | PaperBanana | 提升 |
|---|---|---|---|
| Faithfulness | 43.0 | 45.8 | +2.8% |
| Conciseness | 43.5 | 80.7 | +37.2% |
| Readability | 38.5 | 51.4 | +12.9% |
| Aesthetics | 65.5 | 72.1 | +6.6% |
| Overall | 43.2 | 60.2 | +17.0% |

Conciseness 涨幅最大（+37.2%），因为 vanilla 模型倾向于把 method section 的整段文字 copy 进图里，PaperBanana 通过 Planner 的 in-context learning 学到了"用 keyword 和 structural shorthand 代替 full sentence"。

**Overall 60.2 > Human reference 50.0**，听起来很夸张，但要理解这是 reference-based scoring——VLM judge 在 model 输出和 human 原图之间选更好的。很多 human diagram 本身就 cluttered 或用了过时配色，PaperBanana 通过 style guide 反而更"modern NeurIPS"。

但 Faithfulness 上 PaperBanana 仍然输给 human（45.8 < 50.0），因为精确还原 method section 里的 fine-grained connectivity（箭头起止点、模块顺序）依然很难。

### Statistical Plot（Figure 4）

Statistical plot 要求数值精确，image generation model 容易出现 numerical hallucination（把 0.4 的 bar 画得超过 0.4 gridline），所以 Visualizer 换成 code generation：

$$I_t = \mathrm{VLM}_{\mathrm{code}}(P_t)$$

把 description 转成 Python Matplotlib 代码再执行。

结果：PaperBanana 在 Conciseness、Readability、Aesthetics 上略微超过 human，Faithfulness 接近 human。说明对结构化数据可视化，code-based agentic framework 已经接近 expert 水平。

## 两个有趣的副作用

### 用 Style Guide 美化人类画的图

既然有一本自动总结的《NeurIPS 2025 Aesthetic Guideline》，能不能拿它来改进人类自己画的图？

实验：让 Gemini-3-Pro 基于 style guide 给人类 diagram 提 10 条改进建议，再用 Nano-Banana-Pro 执行。

结果：56.2% win / 6.8% tie / 37.0% loss。多数情况下能提升美学，但 37% 的 loss 说明并非所有 human diagram 都需要标准化——有些 human 的设计选择比"NeurIPS average"更合适。

### Image Gen vs Code Gen 的 trade-off

对 statistical plot 做了对比实验（Figure 5）：

| 方法 | Faithfulness | Aesthetics |
|---|---|---|
| Image Gen (Nano-Banana-Pro) | 低 | 高 |
| Code Gen (Gemini-3-Pro) | 高 | 中 |

Image gen 画出来更漂亮但数值容易错，code gen 数值准确但视觉稍逊。Appendix Figure 9 给了具体 case：
- Radar chart 里 image model 把 Player A 和 Player B 的 Rebound 数据画反了
- Bar chart 里 image model 重复了 "East 10" category
- 另一个 bar chart 里 "Clinical" 数据是 0.4，image model 画的 bar 明显超过 0.4 gridline

结论：**sparse plot 用 image gen，dense plot 用 code gen**，hybrid 策略最优。

## 当前还做不好的地方

### 1. 输出是 raster 不可编辑

PaperBanana 输出 PNG/JPG，但 academic 更喜欢 vector（SVG、PDF）因为可以无限缩放、精确编辑。三个可能 solution：
- Minor 调整：用 image editing model
- Structural reconstruction：OCR + SAM3 分割 + 重新组装（参考 Edit Banana: https://github.com/BIT-DataLab/Edit-Banana）
- 终极方案：GUI Agent 直接操作 Adobe Illustrator 生成 vector（参考 https://arxiv.org/abs/2510.27452）

### 2. Faithfulness 的 fine-grained gap

Failure case（Appendix Figure 10）显示主要错误是：
- Redundant connections（多余的箭头）
- Mismatched source-target nodes（箭头起点终点对错）

这些 subtle error 连 Critic 都检测不到，根因是 VLM 本身的 visual perception 限制。要解决得靠 foundation model 进步。

### 3. 风格多样性被牺牲

统一 style guide 保证符合 academic standard，但所有图都长得有点像。未来需要 dynamic style adaptation，在保持专业度的前提下允许更多个性化。

## 大图景：为什么这个工作重要

Autonomous AI scientist 的完整 workflow 是：
1. Literature review（已有 AI Scientist: https://arxiv.org/abs/2408.06292）
2. Idea generation
3. Experiment iteration
4. Paper writing
5. **Figure generation（PaperBanana 填补的就是这一环）**
6. Self-review

当这六步都能自动化，scientific discovery 的循环才真正闭环。PaperBanana 虽然是 initial work，但建立了一个可复用的 paradigm：**retrieval + planning + style transfer + iterative refinement**。这个 template 可以迁移到 UI design、patent drafting、industrial schematics 等任何需要 strict community standard 的 domain。

对你我这种研究者来说，最直接的 takeaway 是：下次画 method diagram，可以试试让 PaperBanana 先出一版，再人工微调。作者自己在论文里说，论文里所有标 "[Generated by 🍌]" 的图都是 PaperBanana 画的，他们实际 workflow 是"generate multiple candidates and manually select the best one"。这个 generate-and-select 模式在 generative AI 时代会越来越普遍。

---

# PaperBanana：为 AI 科学家自动化生成 publication-ready 学术插图

## 1. 大图景

这篇论文的核心 motivation 非常直观：autonomous AI scientists 已经能在 literature review、idea generation、experiment iteration 上展现能力，但在 visual communication 这个环节上几乎完全缺失。一篇 paper 的 method section 写完后，研究者还需要花大量时间用 TikZ、Python-PPTX、draw.io 这些工具手工绘制 methodology diagram。这个 bottleneck 在 modern AI paper 里尤其严重——NeurIPS 2025 的 diagram 越来越精致，含 custom shapes、specialized icons、3D tensors 等元素，用 code-based 方式根本无法 express。

PaperBanana 的洞察在于：用 agentic workflow 把 retrieval、planning、styling、rendering、self-critique 解耦成五个 specialized agents，每个 agent 只负责一件事，通过 VLM 和 image generation model 的协作完成端到端的 illustration 生成。

参考链接：
- 项目主页：https://dwzhu-pku.github.io/PaperBanana/
- NeurIPS 2025：https://neurips.cc/

## 2. Task Formulation：从数学上理解这个任务

论文给出了一个很清晰的 formalization。学术 illustration 生成可以写成：

$$I = f(S, C) \tag{1}$$

这里：
- $S$ = source context，即 method section 的文本描述
- $C$ = communicative intent，即 figure caption（"Overview of our framework"）
- $I$ = 生成的 image
- $f$ = 我们要学习的 mapping function

公式 (2) 进一步扩展为 retrieval-augmented 形式：

$$I = f(S, C, \mathcal{E}) \tag{2}$$

其中 $\mathcal{E} = \{E_n\}_{n=1}^N$ 是 retrieved reference examples，每个 $E_n = (S_n, C_n, I_n)$ 是一个三元组。$\mathcal{E} = \emptyset$ 时退化到 zero-shot。

这个 formalization 的精髓在于把"画图"这件事拆成了两个 axis：**what to draw**（由 $S$ 和 $C$ 决定的内容）和 **how to draw**（由 $\mathcal{E}$ 携带的视觉风格规范）。PaperBanana 后续的设计——Retriever 找风格参考、Planner 提内容描述、Stylist 注入风格——就是直接对应这个 factorization。

## 3. PaperBanana 架构详解：五个 Agent 的协奏曲

整个 pipeline 分成两个阶段：
- **Linear Planning Phase**：Retriever → Planner → Stylist，串行处理，输出一个 stylistically optimized description $P^*$
- **Iterative Refinement Loop**：Visualizer ↔ Critic 交替运行 $T=3$ 轮

### 3.1 Retriever Agent

公式 (3) 定义了 generative retrieval：

$$\mathcal{E} = \mathrm{VLM}_{\mathrm{Ret}}\left(S, C, \{(S_i, C_i)\}_{E_i \in \mathcal{R}}\right) \tag{3}$$

- $\mathcal{R}$ = 固定的 reference set（PaperBananaBench 里 292 个 reference cases）
- $\mathrm{VLM}_{\mathrm{Ret}}$ = 用 Gemini-3-Pro 做 generative retrieval

这里有个有意思的设计选择：传统的 dense retrieval（用 embedding 算 cosine similarity）被替换成 VLM-based selection。VLM 直接看 $(S_i, C_i)$ 的 metadata 来排序 candidate。Ranking 的 priority 是：

1. **Best Match**：same topic + same visual intent（"Agent Framework" → "Agent Framework"）
2. **Second Best**：same visual intent，不同 topic（"Agent Framework" → "Vision Framework"）
3. **Avoid**：different visual intent（"Pipeline" → "Bar Chart"）

这个 priority 暗示了一个关键 intuition：**visual structure 比 topic similarity 更重要**。一个 vision paper 的 framework diagram 比一个 agent paper 的 performance chart 更适合作为 "agent framework" 的 reference。这点在 ablation study 里也得到验证——random retriever 和 semantic retriever 性能差不多，说明提供"一般的 structural pattern"就足够了，不需要精确 content matching。

### 3.2 Planner Agent

公式 (4)：

$$P = \mathrm{VLM}_{\mathrm{plan}}(S, C, \{(S_i, C_i, I_i)\}_{E_i \in \mathcal{E}}) \tag{4}$$

- $P$ = 一段详细的 textual description of the target illustration
- $\mathrm{VLM}_{\mathrm{plan}}$ 通过 in-context learning 从 $\mathcal{E}$ 里学到 diagram 的"语法"

Planner 是整个系统的 cognitive core。它的 input 是 unstructured 的 method section，output 是一段结构化的描述，要包括：每个 element 的语义、它们的连接、background style、colors、line thickness、icon styles 等。Appendix E 里给出了 Figure 2 的实际 description，可以看到这种描述极其详尽——从最左边的 input icons，到 middle-left 的 Linear Planning Phase 容器，到 middle-right 的 Iterative Refinement Loop，每个 agent 的 icon 描述、label 位置、arrow 方向、颜色编码（blue for Planning, orange for Refinement）都被显式写出。

### 3.3 Stylist Agent

公式 (5)：

$$P^* = \mathrm{VLM}_{\mathrm{style}}(P, \mathcal{G}) \tag{5}$$

- $\mathcal{G}$ = Aesthetic Guideline，一个从整个 reference collection $\mathcal{R}$ 自动总结出来的风格指南
- $P^*$ = stylistically optimized description

$\mathcal{G}$ 的生成是一个 hierarchical summarization pipeline：
1. 把 reference images 分 batch
2. 每个 batch 用 Gemini-3-Pro 生成 local design report（关注 color palette、shapes、lines、layout、typography）
3. 把所有 batch reports 聚合成 unified style guide

Appendix F 里给出了完整的 $\mathcal{G}$，极其精彩。比如对 NeurIPS 2025 method diagram，它总结出 "Soft Tech & Scientific Pastels" 风格：
- Background：cream/beige (#F5F5DC)、pale blue (#E6F3FF)、mint (#E0F2F1)、pale lavender (#F3E5F5)，opacity 10-15%
- Trainable elements 用 warm tones（red, orange, pink），Frozen elements 用 cool tones（grey, ice blue, cyan）
- Process nodes 用 rounded rectangles（80% 的 case）
- Lines：orthogonal 用于 network architectures，curved 用于 system logic
- Typography：sans-serif for labels，serif italic for math variables
- Domain-specific：agent paper 用 cartoony robots，vision paper 用 frustums 和 ray lines，theoretical paper 用 minimalist graph nodes

### 3.4 Visualizer Agent

公式 (6)：

$$I_t = \mathrm{Image\text{-}Gen}(P_t) \tag{6}$$

- $P_0 = P^*$（初始 description 来自 Stylist）
- Image-Gen 默认用 Nano-Banana-Pro，也试了 GPT-Image-1.5

### 3.5 Critic Agent

公式 (7)：

$$P_{t+1} = \mathrm{VLM}_{\mathrm{critic}}(I_t, S, C, P_t) \tag{7}$$

Critic 看 $I_t$、对比原始 $(S, C)$，找出 factual misalignment、visual glitch，然后输出 refined description $P_{t+1}$ 给 Visualizer 重新生成。

这个 loop 跑 $T=3$ 轮，最终输出 $I = I_T$。Appendix G.1 里 Critic 的 system prompt 很讲究，有 Veto Rules：
- Major Hallucination（编造 method section 没提的 module）
- Logical Contradiction（数据流反向）
- Scope Violation（和 caption 不符）
- Gibberish Content（broken LaTeX、乱码文字）

## 4. PaperBananaBench：基准的构造

### 4.1 数据 pipeline

整个 curation 流程值得仔细看：

1. **Collection**：从 NeurIPS 2025 的 5,275 篇 paper 里随机抽 2,000 篇，下载 PDF
2. **Parsing**：用 MinerU toolkit 解析 PDF，提取 method section 文本 + 所有 diagram 及 caption
3. **Filtering**：
   - 丢弃没有 method diagram 的 paper → 1,359 篇
   - 限制 aspect ratio $w:h \in [1.5, 2.5]$ → 610 篇
     - <1.5 太窄，method diagram 通常需要宽幅 landscape
     - >2.5 现有 image generation model 不支持，且会暴露 human origin（在 side-by-side eval 里产生 bias）
4. **Categorization**：用 Gemini-3-Pro 分成 4 类
   - Agent & Reasoning
   - Vision & Perception  
   - Generative & Learning
   - Science & Applications
5. **Human Curation**：annotator 校验 method description、caption、category；过滤掉太 simplistic、cluttered、abstract 的 diagram → 584 篇
6. **Split**：292 test / 292 reference

测试集的统计：average method section 长度 3,020.1 words，average caption 长度 70.4 words。这个长度比例本身就告诉我们：从 3K 字的密集技术文本里抽出 70 字 caption 范围内的视觉表达，是个非常 ill-posed 的任务。

### 4.2 Evaluation Protocol

核心创新是 **VLM-as-a-Judge with Referenced Scoring**。对每个 dimension，VLM judge 比较 model-generated diagram 和 human-drawn diagram，输出 Model wins / Human wins / Tie，对应分数 100 / 0 / 50。

四个 dimension 分成两组：
- **Content**：Faithfulness & Conciseness
- **Presentation**：Readability & Aesthetics

**Hierarchical Aggregation** 的设计哲学是 "information visualization must primarily show the truth"（来自 Mackinlay 1986 和 Tufte 1983）：
1. Faithfulness 和 Readability 是 primary dimensions
2. Conciseness 和 Aesthetics 是 secondary
3. Primary 维度决定胜负，平局才看 secondary

VLM Judge 的可靠性验证：
- Inter-Model Agreement：Gemini-3-Pro vs Gemini-3-Flash 的 Kendall's tau 在四个维度和 overall 上是 0.51 / 0.60 / 0.45 / 0.56 / 0.55；vs GPT-5 是 0.43 / 0.47 / 0.44 / 0.42 / 0.45
- Human Alignment：Gemini-3-Pro vs human annotators 的 Kendall's tau 是 0.43 / 0.57 / 0.45 / 0.41 / 0.45

这些 correlation 在 0.4-0.6 之间，说明 VLM judge 是 human judgment 的合理 proxy，但远非完美——尤其 Aesthetics 的 0.41 暗示主观维度上 VLM 和人类还有差距。

## 5. 实验结果：主结果与 Ablation

### 5.1 Main Results（Table 1）

| Method | Faithfulness | Conciseness | Readability | Aesthetic | Overall |
|---|---|---|---|---|---|
| GPT-Image-1.5 (Vanilla) | 4.5 | 37.5 | 30.0 | 37.0 | 11.5 |
| Nano-Banana-Pro (Vanilla) | 43.0 | 43.5 | 38.5 | 65.5 | 43.2 |
| Few-shot Nano-Banana-Pro | 41.6 | 49.6 | 37.6 | 60.5 | 41.8 |
| Paper2Any (w/ Nano-Banana-Pro) | 6.5 | 44.0 | 20.5 | 40.0 | 8.5 |
| **PaperBanana (w/ GPT-Image-1.5)** | 16.0 | 65.0 | 33.0 | 56.0 | 19.0 |
| **PaperBanana (w/ Nano-Banana-Pro)** | **45.8** | **80.7** | **51.4** | **72.1** | **60.2** |
| Human (reference) | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |

几个关键观察：

1. **PaperBanana w/ Nano-Banana-Pro 的 Overall 60.2 > Human 50.0**：意味着在 292 个 test case 上，VLM judge 整体上更偏好 PaperBanana 的输出。这听起来惊人，但要注意这是 reference-based scoring——human reference 本身可能就有 cluttered 或过时的 case，PaperBanana 通过 style guide 反而更"modern"。

2. **Faithfulness 上 PaperBanana 仍输给 Human**（45.8 vs 50.0）：这是最 stubborn 的维度，因为要精确还原 method section 里的 fine-grained connectivity（arrow 起止点、模块顺序）非常难。Appendix Figure 10 的 failure case 显示，主要错误是 redundant connections 和 mismatched source-target nodes。

3. **Conciseness 涨幅最大**（+37.2%）：vanilla Nano-Banana-Pro 容易 verbose，把 method section 一字不漏地 copy 进 diagram。Planner 通过 in-context learning 学到了"用 keyword + structural shorthand 而非 full sentence"。

4. **GPT-Image-1.5 在 agentic 框架下也跑不动**（16.0/65.0/33.0/56.0）：说明 text rendering 和 instruction following 能力是瓶颈——GPT-Image-1.5 经常把 method section 里的文字渲染成乱码，这是 academic illustration 的硬伤。

5. **Paper2Any 表现差**（6.5/44.0/20.5/40.0）：它的设计目标是 present high-level ideas 而非 faithful method diagram，objective mismatch 导致 faithfulness 暴跌。

### 5.2 Ablation Study（Table 2）

| # | Retriever | Planner | Stylist | Visualizer | Critic | Faith. | Conc. | Read. | Aesth. | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| ① | √ | √ | √ | √ | 3 iters | 45.8 | 80.7 | 51.4 | 72.1 | 60.2 |
| ② | √ | √ | √ | √ | 1 iter | 38.3 | 75.2 | 50.6 | 68.9 | 51.8 |
| ③ | √ | √ | √ | √ | - | 30.7 | 79.2 | 47.0 | 72.1 | 45.6 |
| ④ | √ | √ | = | √ | - | 39.2 | 61.7 | 47.9 | 67.4 | 49.2 |
| ⑤ | ○ (random) | √ | √ | √ | - | 37.3 | 62.7 | 51.1 | 65.6 | 48.3 |
| ⑥ | - | √ | √ | √ | - | 41.9 | 58.6 | 43.1 | 62.9 | 44.2 |

这里有 4 个关键 insight：

**Insight A：Critic Agent 价值最大**（① vs ③，60.2 vs 45.6）。  
特别是 Faithfulness 从 30.7 → 45.8（+15.1），因为 Stylist 在做 visual polishing 时会丢失技术细节（④ vs ③ 的 faithfulness 39.2 → 30.7 暴跌 -8.5），Critic 通过对比 $(S, C)$ 找回这些细节。多轮 iteration 进一步提升（② 1-iter 51.8 → ① 3-iter 60.2）。

**Insight B：Stylist 是把双刃剑**（③ vs ④）。  
Stylist 让 Conciseness +17.5%（61.7 → 79.2），Aesthetics +4.7%（67.4 → 72.1），但 Faithfulness -8.5%（39.2 → 30.7）。这是个很本质的 trade-off：style standardization 倾向于 simplify，而 simplify 会丢细节。Critic 的存在是为了弥补这个 trade-off。

**Insight C：Retriever 的形式不重要，但有没有 retriever 重要**（④ vs ⑤ vs ⑥）。  
没有 retriever（⑥）的 Overall 44.2，random retriever（⑤）48.3，semantic retriever（④）49.2。Random 和 semantic 差距很小，验证了"general structural pattern 比 precise content matching 更重要"。但没有 retriever 导致 Planner 失去 anchor，输出 verbose 且 visually unrefined。

**Insight D：Conciseness 的提升主要来自 Planner + Stylist**。  
No-retriever 设置（⑥）的 Conciseness 只有 58.6，加上 retriever（④）涨到 61.7，加上 Stylist（③）涨到 79.2。这说明 Conciseness 是 reference + style guide 共同作用的结果——Planner 看到简洁的 reference 后会模仿，Stylist 显式注入 "用 keyword 不用 full sentence" 的 rule。

## 6. 扩展到 Statistical Plots

这个 extension 的设计非常巧妙。Methodology diagram 优先 aesthetics，statistical plot 优先 numerical precision。所以 Visualizer 从 image generation 换成 code generation：

$$I_t = \mathrm{VLM}_{\mathrm{code}}(P_t)$$

这里 $\mathrm{VLM}_{\mathrm{code}}$ 把 description 转成 executable Python Matplotlib 代码。Critic 仍然 $P_{t+1} = \mathrm{VLM}_{\mathrm{critic}}(I_t, S, C, P_t)$，但 $S$ 变成 raw tabular data，$C$ 变成 visual intent（"a bar plot titled X"）。

**Testset Curation**：复用 ChartMimic 的 "direct mimic" subset（2,400 plots from arXiv + matplotlib galleries）。用 Gemini-3-Pro 从代码里抽取 raw data + 生成 visual description + 标注 difficulty。过滤后剩 914 plots，归并成 7 类，最后采样 240 test + 240 reference。

**结果**（Figure 4）：
- Faithfulness: +1.4%
- Conciseness: +5.0%
- Readability: +3.1%
- Aesthetics: +4.0%
- Overall: +4.1%

值得注意的是 PaperBanana 在 Conciseness、Readability、Aesthetics 上**略微超过 human**，Faithfulness 接近 human。这说明对结构化数据可视化，code-based agentic 框架已经接近 expert 水平。

**Coding vs Image Generation** 的对比（Figure 5 和 Section 6.2）很有启发：
- Image generation 在 Readability 和 Aesthetics 上胜出（plot 更 visually appealing）
- 但 Faithfulness 和 Conciseness 上明显落后
- Manual inspection 发现：image model 对 sparse plot 渲染准确，对 dense plot 出现 numerical hallucination 和 element repetition
- 作者建议 hybrid 策略：sparse 用 image gen，dense 用 code

Appendix Figure 9 给出了具体 case：
- Player A vs B 的 radar chart，image model 把 Rebound 轴的数据画反了（Player A 本应低于 Player B，画成相反）
- Sales Distribution bar chart 里，image model 重复了 "East 10" category
- Scientific Article Types 里，"Clinical" 数据值是 0.4，但 image model 画的 bar 明显超过 0.4 gridline

这些 failure case 揭示了 image generation model 在 precise numerical rendering 上的根本 limitation。

## 7. Discussion：两个有趣的扩展

### 7.1 用 Style Guide 美化 Human Diagrams

实验设计：用 $\mathcal{G}$ 让 Gemini-3-Pro 生成 up to 10 actionable suggestions，再用 Nano-Banana-Pro 执行 refinement。在 292 个 test case 上对比 refined vs original human diagram。

结果：win/tie/loss = 56.2% / 6.8% / 37.0%。说明自动总结的 style guide 确实能提升人类 diagram 的美学质量，但 37% 的 loss 也提示——并非所有 human diagram 都需要 stylization，有些 human 的设计选择反而比"NeurIPS average"更合适。

### 7.2 Coding vs Image Generation

参见上文 Section 6.2 的 trade-off 分析。这暗示了一个更广的设计原则：**对"semantic visual"（method diagram）用 image gen，对"quantitative visual"（statistical plot）用 code gen**。

## 8. Limitations 与 Future Directions

作者很诚实地列出了几个核心限制：

### 8.1 Raster Output 不可编辑

PaperBanana 输出 raster image，但 academic context 偏好 vector graphics（无限缩放、精确细节）。三个 potential solution：
1. **Minor adjustment**：用 image editing model（Nano-Banana-Pro 本身）
2. **Structural reconstruction**：OCR + SAM3 + 重新组装（参考 Paper2Any 和 Edit Banana: https://github.com/BIT-DataLab/Edit-Banana）
3. **GUI Agent**：让 agent 直接操作 Adobe Illustrator，生成 fully editable vector graphics（参考 Huang et al. 2026, Sun et al. 2025）

### 8.2 Style Standardization vs Diversity 的 trade-off

统一的 style guide 保证了 academic standard compliance，但牺牲了 stylistic diversity。未来需要 dynamic style adaptation。

### 8.3 Fine-Grained Faithfulness 的 gap

Failure analysis 显示主要错误是 fine-grained connectivity（misaligned arrow、wrong direction）。Critic model 检测不到这些 subtle error，根因是 VLM 本身的 visual perception 限制。这需要 foundation model 的根本进步。

### 8.4 Evaluation Paradigm 的局限

Reference-based VLM-as-a-Judge 有两个问题：
- Faithfulness 量化难：subtle connectivity error 需要 structure-based metric（参考 DiagramEval: https://arxiv.org/abs/2510.25761）
- Aesthetics 主观：textual prompt 不足以 align VLM 和 human preference，需要 trained reward model

### 8.5 Test-Time Scaling for Diverse Preferences

当前只输出一个 result，但 generative model 有 stochasticity，user taste 也是 diverse。自然扩展是 generate-and-select：生成多个 candidate，用 VLM preference model 或 human 选择最合适的。

## 9. Related Work 的关联脉络

### 9.1 Code-based Diagram Generation

这条线从 TikZ 开始：
- **Detikzify** (Belouadi & Eger, 2024): 用 TikZ 合成 scientific figure
- **TikZero** (Belouadi et al., 2025): zero-shot text-guided TikZ synthesis
- **Automatikz** (Hsu & Eger, 2023): text-guided scientific vector graphics
- **PPTAgent** (Zheng et al., 2025): 用 Python-PPT 生成 slides

这些方法的 limitation 在于：TikZ 和 Python-PPT 难以表达 modern AI paper 里的 custom icons、3D tensors、intricate shapes。

### 9.2 Coding-based Data Visualization

- **Data2Vis** (Dibia & Demiralp, 2019): LSTM 把 JSON 转 Vega-Lite
- **LIDA** (Dibia, 2023): LLM 自动生成 visualization
- **MatplotAgent** (Yang et al., 2024): agentic framework for matplotlib
- **CODA** (Chen et al., 2025): collaborative data visualization
- **PlotGen** (Goswami et al., 2025): multi-agent LLM 数据可视化
- **ChartMimic** (Yang et al., 2025b): chart-to-code generation，PaperBanana 复用了它的 dataset
- **Plot2Code** (Wu et al., 2025b): 多模态 LLM 从 plot 生成 code 的 benchmark

### 9.3 Image Generation-based Diagram

- **FigGen** (Rodriguez et al., 2023): text to scientific figure
- **AutoFigure** (Anonymous, 2026, under review): 把 scientific content 转成 symbolic representation 再用 GPT-Image 渲染
- **SciFig** (Huang et al., 2026): 自动 scientific figure generation
- **SridBench** (Chang et al., 2025): diagram generation benchmark，未公开

### 9.4 Foundation Models

PaperBanana 用的核心模型：
- **Gemini-3-Pro**（Comanici et al., 2025）：VLM backbone
- **Nano-Banana-Pro**（DeepMind, 2025）：image generation
- **GPT-Image-1.5**（OpenAI, 2025a）：对比 baseline
- **GPT-5**（OpenAI, 2025b）：VLM judge baseline

## 10. 我的几点 Intuition 总结

1. **Decoupling 是关键**：PaperBanana 把"画什么"（Planner）和"怎么画"（Stylist）解耦。Style guide 是 offline 总结的 reusable asset，不需要每次 inference 重新生成。

2. **Retrieval 的目的不是 content match 而是 structural pattern transfer**：random retriever 接近 semantic retriever 的性能，这是个反直觉但 robust 的发现。它暗示 in-context learning 在这里更多是"看到一些 academic diagram 的样子"而非"看到完全类似的 case"。

3. **Critic Agent 的真正价值在 Faithfulness recovery**：Stylist 会丢技术细节，Critic 通过对比 source 把细节找回来。这个 complementarity 是 agentic design 的精髓——单个 agent 优化单一目标会破坏其他目标，多 agent 协作才能实现 multi-objective trade-off。

4. **Image gen 适合 semantic visual，code gen 适合 quantitative visual**：这是论文最重要的 meta-level insight。未来的 hybrid system 应该 dynamic 选择 backend。

5. **VLM-as-Judge 在 0.4-0.6 correlation 区间**：这意味着自动评估可行但远非 perfect。Future work 必须要 trained reward model 和 structure-based metric。

6. **Raster → Vector 是下一战场**：当前 output 是 4K raster，但 academic 需要 editable vector。GUI Agent 操作 Illustrator 是个很有想象空间的方向。

7. **Test-time scaling 是自然延伸**：generate-and-select 在实际应用里已经被作者用（"we generated multiple candidates and manually selected the best one for presentation"）。把 human selection 替换成 VLM preference model 就能 fully 自动化。

## 11. 进一步阅读推荐

如果你想深入这个方向，建议看：
- Paper2Any: https://github.com/OpenDCAI/Paper2Any
- Edit Banana: https://github.com/BIT-DataLab/Edit-Banana
- ChartMimic: https://chartmimic.github.io/
- MinerU: https://arxiv.org/abs/2509.22186
- DiagramEval: https://arxiv.org/abs/2510.25761
- SridBench: https://arxiv.org/abs/2505.22126
- LIDA: https://github.com/microsoft/lida
- MatplotAgent: https://arxiv.org/abs/2402.10453
- Detikzify: https://arxiv.org/abs/2407.09405

PaperBanana 是 autonomous AI scientist 完整 workflow 的关键一环。当 LLM 能自动写 paper、自动跑 experiment、自动生成 figure、自动 review，整个 scientific discovery 的循环才真正闭环。这个工作虽然 initial，但建立了一个可以 follow 的 paradigm：retrieval + planning + style transfer + iterative refinement，这个 template 可以迁移到 UI design、patent drafting、industrial schematics 等任何需要 strict community standard 的 domain。
