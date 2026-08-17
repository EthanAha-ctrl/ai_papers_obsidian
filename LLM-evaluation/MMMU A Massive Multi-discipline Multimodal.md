---
source_pdf: MMMU A Massive Multi-discipline Multimodal.pdf
paper_sha256: cb474e70a6728110aa39d94480535f482a656c5a65bfc2129b47d66350f1f974
processed_at: '2026-08-05T19:19:27-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MMMU — 用人话讲一遍

Paper: https://arxiv.org/abs/2311.16502
Project: https://mmmu-benchmark.github.io/

## 一句话先说清楚

MMMU 就是给 multimodal model 出的一份**大学考试卷子**, 覆盖 30 个科目, 11.5K 道题, 每道题都带图, 而且不是那种"图里有个猫, 猫是什么颜色"的题, 是那种**你得真懂这个学科才能做出来**的题。GPT-4V 当年只能做对 56%, 人类专家 88%, 这说明 multimodal model 离 "expert AGI" 还远。

## 为什么搞这么个东西

### 起因很简单: 之前的 benchmark 太水了

你看 CogVLM 在 VQA-v2 上 85%, ScienceQA-IMG 上 92%, RefCOCO 上 93% — 听起来 multimodal 已经"解决"了。但你去看看这些 benchmark 上的题:

- VQA-v2: "What color is the car?" — 看一眼就答
- GQA: "Is the ball left of the box?" — 空间关系
- OK-VQA: 需要 external commonsense, 但还是 daily knowledge
- ScienceQA: 大部分是 elementary/middle school level

这些都 saturated 了, 区分不出 model 的真实能力。就像你给高中生做 1+1, 大家都 100%, 你看不出谁更聪明。

MMLU (https://arxiv.org/abs/2009.03300) 解决了 text-only 的 college-level 评估, 但人脑处理问题不是纯文字的 — 医生看 X 光, 化学家看分子结构, 工程师看电路图, 音乐家看乐谱 — **图像在 expert reasoning 里是 first-class citizen**。所以需要一个 college-level multimodal benchmark, 这就是 MMMU。

### AGI 分级里的位置

Google DeepMind 的 Morris et al. (https://arxiv.org/abs/2311.02462) 提了个 AGI 分级框架, Level-3 叫 Expert AGI, 大意是 AI 在 broad range of skilled tasks 上达到 90th percentile 的 skilled adult 水平。 college exam 是衡量 skilled adult 的天然工具, 所以 MMMU 拿 college exam 当 source。

## 数据怎么来的

50 多个 college student annotator, 从 textbook + 网上题库 + 自己出的题里挑。三阶段:

1. **选 subject**: 剔掉 law / linguistics 这种"几乎没有图像"的, 留下 30 个 subject (从 music 到 pharmacology 到 mechanical engineering)
2. **标注**: 学生负责从 source 里挑题, 必要时自己改写
3. **清洗**: 查重 (lexical overlap + URL similarity), 改 typo, 分难度 (very easy / easy / medium / hard), 把 very easy 砍掉约 10%

**Data contamination 防御**: annotators 被明确要求挑**答案不立即可见**的题 — 答案在 separate document 或 textbook 末尾那种。test set labels 不公开, 只有 dev set (5 per subject) 和 validation set (900 题) 可见, test set 10.5K 题 label 保密, 用 EvalAI server 评估。

## 数据长啥样

| 项目 | 数字 |
|---|---|
| 总题数 | 11,550 |
| Dev / Val / Test | 150 / 900 / 10,500 |
| 难度 E : M : H | 28% : 45% : 27% |
| 多选 : 开放 | 94% : 6% |
| 有 explanation 的 | 17.62% |
| 平均 question 长度 | 59.33 tokens |
| 多图题 | 7.39% |
| Image 在 question 中间或末尾的 | 87% |
| Image 在 options 里的 | 3.37% |

**关键**: 平均 question 才 59 tokens, 但一张 image 携带的信息量远超 1000 tokens 的描述。你看那张 BJT 电路图, 几个 resistor + 一个 transistor + 几个 voltage source, 但你要把它翻成文字描述, 至少要 500 字, 而且还会漏掉 topology 细节。这就是为什么 MMMU 是真正的 multimodal, 不是 "image + caption" 的 wrapper。

## 题目长啥样 — 6 个学科各举一例

### Art & Design (Music)
> Among the following harmonic intervals, which one is constructed incorrectly?
> (A) Major third [image 1] (B) Diminished fifth [image 2] ...

每张 image 是一段 sheet music, 你得知道 major third 是 4 semitones, diminished fifth 是 6 semitones, 然后数 sheet music 上的 staff line 间距 — 这种题纯 text LLM 连门都进不去。

### Business (Marketing)
> The graph shown is compiled from data collected by Gallup [image 1]. Find the probability that the selected Emotional Health Index Score is between 80.5 and 82?

图是个 histogram / distribution curve, 要算 normal distribution 的 probability, 用 z-table。需要 (a) 读图认出 distribution 类型, (b) 估计 mean 和 std, (c) 算 z-score, (d) 查表。**4 步链, 任一错全错**。

### Science (Math)
> The region bounded by the graph as shown above. Choose an integral expression that can be used to find the area of R.
> f(x) = x³ − 6x² + 8x, g(x) = −½x² + 2x

4 个 integral 选项, 你得: 看图判断哪个函数在上、哪个在下, 判断 intersection 在哪 (x=0 到 1.5 还是 0 到 2)。**Calculus 知识 + 图形理解**双重需求。

### Health & Medicine (Clinical Radiology)
> You are shown subtraction [image 1], T2 weighted [image 2] and T1 weighted axial [image 3] from a screening breast MRI. What is the etiology of the finding in the left breast?
> (A) Susceptibility artifact (B) Hematoma (C) Fat necrosis (D) Silicone granuloma

**3 张 MRI 图**, 你得懂 MRI 序列 (T1 vs T2 vs subtraction 是什么意思), 懂 breast MRI 的 differential diagnosis, 懂每种 etiology 在不同序列上的 signal characteristic。这是 radiologist 的 day-to-day work, 模型要做到这个水平才能成为 expert assistant。

### Humanities & Social Science (History)
> In the political cartoon, the United States is seen as fulfilling which of the following roles? [image 1]
> (A) Oppressor (B) Imperialist (C) Savior (D) Isolationist

cartoon 上半截和下半截有 contrast — 上半 negative 下半 positive, 整体 narrative 是 US 作为 savior 把贫困国家"救"出来。GPT-4V 因为对 "US imperialism" 有 textual prior, 选了 (B), 没看到 cartoon 本身的 visual irony。这是 **textual prior 压过 visual evidence** 的经典 case。

### Tech & Engineering (Electronics)
> Find VCE for the circuit shown. Neglect VBE.
> IE = VEE / RE = 5V / 4kΩ = 1.25 mA
> VCE = VCC − IE·RL = 10V − 1.25mA × 5kΩ = 3.75V

BJT 电路 DC 分析, 标准的 electronics 101。但模型要:
- 从 schematic 读出 VCC=10V, VEE=5V, RL=5kΩ, RE=4kΩ, 还要看清是 NPN 还是 PNP
- 知道 "neglect VBE" 意味着 VE ≈ VB, 所以 IE ≈ VEE/RE
- 知道 IC ≈ IE (BJT 的 α ≈ 1)
- 算 VCE = VCC − IC·RL

每一步都是 model 平时没专门训练过的 expert reasoning chain。

## 主结果 — 用人话讲

### GPT-4V = 55.7%, 人类 expert = 88.6%

光这个数字就够说明问题: 最强的 model 也才过半多一点。但更 informative 的是**分层结果**。

### 开源 vs 闭源

| 时点 | 模型 | Test overall |
|---|---|---|
| paper submission | BLIP-2 FLAN-T5-XXL | 34.0 |
| paper submission | LLaVA-1.5 | 33.6 |
| paper submission | GPT-4V | 55.7 |
| v4 update | LLaVA-1.6-34B | 44.7 |
| v4 update | InternVL-Chat-V1.2 | 46.2 |
| v4 update | Qwen-VL-MAX | 46.8 |
| v4 update | Claude 3 Opus | 59.4 |
| v4 update | Gemini 1.5 Pro | 62.2 |
| v4 update | GPT-4o | 69.1 |

**Intuition**: 开源在半年里从 34 推到 47, 闭源从 56 推到 69, 双方都在涨, 但闭源涨幅更大。MMMU 是 moving target — 等 model saturation 了就需要 MMMU-Pro (https://arxiv.org/abs/2409.02813) 这种升级版。

### OCR + Caption 对 text-only LLM 没用

这是最 important 的 ablation:

| LLM | 原版 | + OCR | + LLaVA Caption |
|---|---|---|---|
| FLAN-T5-XXL | 31.2 | 31.9 | 31.9 |
| Vicuna-13B | 31.0 | 31.9 | 32.7 |
| GPT-4 text | 33.8 | — | — |

加 OCR 几乎没用, 加 caption 也只涨 1 个点。**为什么?** 因为 MMMU 的图像信息**不是 text-transducable** 的:

- Sheet music: 你 OCR 出 "C E G" 也没用, 你得知道这是 C major triad, 还得知道它的 harmonic function
- Circuit: 你把 resistor 阻值 OCR 出来也没用, 你得知道它在哪个 branch, 跟谁串联谁并联
- MRI: 你 caption "white spot in left breast" 没用, 你得在 3 个序列上 cross-reference 判断 signal characteristic

**这就是 LMM 必须存在的 reason**。pipeline 式 (vision encoder → text → LLM) 必然 fail, 因为信息在第一步就丢了。Vision 必须跟 reasoning 端到端耦合。

### 学科间差异巨大

GPT-4V 在 6 个学科上的成绩:

| Discipline | Test count | GPT-4V |
|---|---|---|
| Art & Design | 1,163 | 65.3 |
| Business | 1,428 | 64.3 |
| Science | 2,426 | 48.4 |
| Health & Medicine | 1,752 | 63.5 |
| Humanities & Social Sci. | 947 | 76.3 |
| Tech & Engineering | 2,784 | 41.7 |

Tech & Engineering 才 41.7%, Humanities 76.3% — **差距 35 个百分点**。Intuition:

- Humanities 的图多为 photo / painting / cartoon, 跟 web pretraining data distribution 接近, 模型见过类似的
- Tech 的图是 schematic / diagram / blueprint, 训练数据里这种 image-text pair 极少
- Tech 的 reasoning chain 长 (circuit 分析要 4-5 步, thermo 要 6-7 步), Humanities 通常 1-2 步就到了

### Image type 维度

GPT-4V 在 30 种 image type 上的表现:

| Image Type | GPT-4V |
|---|---|
| Advertisements | 100.0 |
| Logos and Branding | 85.7 |
| Historical Timelines | 78.6 |
| Portraits | 76.1 |
| Paintings | 75.9 |
| Photographs | 64.2 |
| Medical Images | 59.6 |
| Tables | 61.8 |
| Chemical Structures | 50.6 |
| Mathematical Notations | 50.0 |
| Geometric Shapes | 40.2 |
| Sheet Music | 38.8 |
| Trees and Graphs | 38.9 |

**Intuition**: Web-crawled data 里 photos / paintings / ads 海量, 所以模型见过很多; sheet music / chemical structures / mathematical notations 在 LAION 这种数据集里占比极小, 模型几乎没学到。这是 **pretraining data distribution 直接反映到 benchmark 上** 的最清晰 case。

### 难度差异 — GPT-4V 在 Hard 上也只有 31%

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Fuyu-8B | 28.9 | 27.0 | 26.4 |
| Qwen-VL-7B | 39.4 | 31.9 | 27.6 |
| LLaVA-1.5-13B | 41.3 | 32.7 | 26.7 |
| GPT-4V | 76.1 | 55.6 | 31.2 |

GPT-4V 在 Easy 上 76%, 跟 LLaVA-1.5 在 Easy 上的 41 比起来优势巨大; 但在 Hard 上 GPT-4V 也只有 31, 跟 LLaVA 的 26 差距就 5 个点。

**关键 insight**: 在简单 perception + 1-2 步 reasoning 上, model size 和 data quality 带来巨大优势; 但在长 reasoning chain 上, **所有 model 都 collapse**, scaling 在这里失效了。这跟 reasoning model (o1/o3) 出现的 motivation 完全一致 — long-chain multimodal reasoning 需要 test-time compute, 不是单纯 forward pass 能解的。

## 错误分析 — 最有 learning value 的部分

150 个 GPT-4V 错误样本, 人工标注 root cause:

| Error Type | 占比 |
|---|---|
| Perceptual | 35% |
| Lack of Knowledge | 29% |
| Reasoning | 26% |
| 其他 (text understanding, reject, annotation, extraction) | 10% |

### Perceptual Error 35% — 最常见

分两种:

**Basic perceptual**: 模型连 "from left to right, top to bottom" 这种 spatial ordering 都错。Figure 6 例子, 一个简单的 6 格子 grid, GPT-4V 搞错了顺序。这是 ViT patch embedding → visual token → LLM 这条 pipeline 里 **fine-grained spatial info 丢失** 的直接后果。每张图被压成 576 个 visual token (LLaVA), 一个 grid 9 个格子的位置信息全打散了。

**Domain-specific perceptual**: Figure 84 那个 OS interleaved transaction 图, 两个 transaction A 和 B 在一个 CPU 上交错执行, 画成时间线。GPT-4V 看错了图, 以为 A 在 CPU1, B 在 CPU2, 完全误解了 interleaved 的概念。这种 error 表面上是 perception, 实际上是 **model 没有 domain convention knowledge** — 它不知道这种图怎么读。

还有 Figure 67 那个 US "Savior" cartoon, GPT-4V 选了 (B) Imperialist, 因为它的 textual prior ("US = imperialist") 压过了 visual evidence (cartoon 在画 US 救人)。**Textual bias over visual evidence** 是 LMM 一个系统性问题。

### Lack of Knowledge 29% — 知识盲区

Figure 13: Las Meninas by Velázquez, GPT-4V 正确识别出画作, 但不知道原作放在 King Philip IV 的 study — 选了 Prado Museum。这是 factual knowledge 缺失。

Figure 83 (CS / DFA): 模型看到 double circle 但不知道这在 DFA 里表示 "accept state" — domain convention 知识缺失。

Figure 54 (Clinical Medicine): cardiac catheterization 数据表, 给了 RA / RV / PA / LA / LV / Ao 各自的 O2 saturation 和 pressure。GPT-4V 看到数据但不知道: RV 到 PA 之间 O2 sat "step-up" 提示 VSD (ventricular septal defect), LA 到 LV 之间 O2 sat "step-down" 提示 ASD。这是 medical school 教的核心 pattern recognition, web data 里覆盖稀疏。

### Reasoning Error 26% — 最有 teaching value

**Figure 45 (Calculus)**: 给 f(x) 和 g(x) 的图, 选面积 integral。GPT-4V 没正确判断哪个函数在上, 也搞错了交点 x 范围。

**Figure 81 (Greenshields model)**: 交通流理论, u = u_max − (u_max/k_jam)·k
- u: speed
- k: density  
- u_max: free-flow speed
- k_jam: jam density

给 6 个数据点, linear regression 拟合求 k_jam。GPT-4V **公式写对了, 但 skip 了 regression 计算**, 直接猜最接近 85 的选项 = 110, 正确答案 111。这种 "概念对但 arithmetic skip" 是 LLM 经典 failure mode — 模型在长计算里会 hallucinate shortcut。

**Figure 89 (Thermodynamics / Polytropic)**:
- PV^n = constant (polytropic process)
- First law: ΔU = Q − W, 其中 ΔU = m·Cv·(T2−T1), W = (P2V2−P1V1)/(1−n)
- 需要: 算 m → 解 n → 算 V2 → 算 W → 算 Q

6 步链, GPT-4V 把单位换算错 (T 没用 Kelvin), 把 m 算错, 把 calculation order 弄乱。这是 long-chain reasoning 的典型 collapse。

**Figure 93 (Engineering Dynamics)**: 两小球沿圆弧下滑, energy conservation:

½(2m)v² = mgR → v = √(gR)

GPT-4V 把第二个球的下落高度当成 2R (其实只有 R, 因为两球 rigidly connected, 一起下降 R), 算出 3mgR 的 PE, 然后整个 reasoning 跑偏。这是 conceptual physics 错误, 不是 arithmetic。

## 整体直觉 — 从 MMMU 能学到什么

### 1. Vision-language 接口设计是 bottleneck

现在主流 LMM 架构: CLIP-ViT → projector → LLM

- Visual token 数固定 (LLaVA 576, BLIP-2 32 queries)
- 对一张 natural photo 够用, 对一张 schematic 远远不够
- 没有 "zoom in" 机制 — model 不能在 reasoning 过程中决定看图的哪个局部

下一代方向: dynamic resolution (Qwen2-VL https://qwenlm.github.io/blog/qwen2-vl/), 或 Fuyu 那种 patch-as-token 直接进 LLM, 不做 ViT pooling。

### 2. Expert knowledge 不能只靠 web pretraining

Web data 对 art history / 临床医学 / 电路分析 的覆盖又稀又 noise。三种解法:
- **RAG**: 把 textbook 当 retrieval source
- **Domain SFT**: 用 (image, trace, answer) 三元组 fine-tune
- **Synthetic data**: SPICE 生成电路 + 仿真结果, RDKit 生成分子 + 性质, LilyPond 生成乐谱 + music theory 标注

最后一种是 unbounded 的, 我觉得最有潜力。

### 3. Test-time compute 在 multimodal 上还没充分开发

MMMU hard subset (Calculus, Thermo, Engineering Dynamics) 完美适合 test-time scaling — 让 model 选择下一步 visual action (crop, zoom, OCR sub-region), 加上 chain-of-thought + self-verification。GPT-4o 在 MMMU 上 69.1% 比 GPT-4V 的 55.7% 高这么多, 一部分原因就是 better CoT 和 test-time reasoning。o1 / o3 系列估计还会更高。

### 4. Benchmark 的 lifetime 问题

MMMU v1→v4 半年内: 开源 34→47, 闭源 56→69。这个 saturation 速度比 MMLU 快多了 (MMLU 用了 3 年才让 model 从 70 推到 90)。原因: multimodal field 进展太快, 而且 MMMU 是 4 选 1 多选, 上限较低 (random 25%)。

下一代 benchmark 需要:
- 更大 (避免 saturation)
- Open-ended generation (避免 regex extractable answer)
- Agentic / tool use (类似 GAIA https://arxiv.org/abs/2311.12983)
- Dynamic / hidden test set (避免 contamination)

## 给 Karpathy 的 actionable suggestions

1. **Visual-CoT benchmark**: 在 MMMU hard subset 上, 让 model 选择下一步 visual action (crop region X / zoom / OCR), 把 visual reasoning chain 显式化。这是 test-time compute × multimodal 的天然结合点。

2. **Dynamic visual token budget**: sheet music / chemical structure 应该 allocate 远多于 photo 的 token。可以训一个 image complexity classifier 在线决定 token 数。

3. **Expert synthetic data**: 用 SPICE + RDKit + LilyPond 生成无限 expert data, 配上 chain-of-thought trace。这种 data 让开源 LMM 在 MMMU Tech & Science 子集上快速攀升是可行的。

4. **Modality balance loss**: contrastive loss 与 LM loss 的 weight 固定是 suboptimal 的, expert domain visual token 应该参与更密集的 supervision。

## Reference

- MMMU 主页: https://mmmu-benchmark.github.io/
- Live leaderboard: https://mmmu-benchmark.github.io/#leaderboard
- arXiv: https://arxiv.org/abs/2311.16502
- MMMU-Pro (升级版): https://arxiv.org/abs/2409.02813
- Hugging Face: https://huggingface.co/datasets/MMMU/MMMU
- EvalAI: https://eval.ai/web/challenges/challenge-page/2179
- AGI levels (Morris et al.): https://arxiv.org/abs/2311.02462
- MMLU: https://arxiv.org/abs/2009.03300
- MathVista: https://arxiv.org/abs/2310.02255
- GAIA: https://arxiv.org/abs/2311.12983
- Qwen2-VL (dynamic resolution): https://qwenlm.github.io/blog/qwen2-vl/
- LMM 综述: https://arxiv.org/abs/2309.17421

人话总结一句: MMMU 就是给 multimodal model 出的一份**大学期末考**, 涵盖 30 个科目, 每题都带图, 而且**你得真懂这个学科 + 能读图 + 能多步推理**才能做对。GPT-4V 当年 56%, 现在 GPT-4o 69%, 人类 expert 88%, 还有明显 gap。错误集中在 perception (35%) + knowledge (29%) + reasoning (26%) 三块, 每一块都需要不同的修复路径 — vision encoder 升级 / domain knowledge injection / test-time compute scaling。

---

# MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI — 技术深度讲解

Paper: https://arxiv.org/abs/2311.16502 | Project & Leaderboard: https://mmmu-benchmark.github.io/

## 1. 高层 motivation — 为什么需要一个新 benchmark

### 1.1 AGI 分级框架下的定位
作者们借用 Morris et al. (Google DeepMind) 提出的 AGI 分级 taxonomy (https://arxiv.org/abs/2311.02462), 把 Expert AGI 定义为 Level-3 milestone — AI 在 broad range of tasks 上达到 "至少 90th percentile of skilled adults"。这是一个操作性较强的定义, 关键词是 breadth + depth。

Breadth 通过 college disciplines 来覆盖, depth 通过 college-level subject knowledge + deliberate reasoning 来保证。MMLU (https://arxiv.org/abs/2009.03300) 与 AGIEval (https://arxiv.org/abs/2304.06364) 走的是 text-only 路线; MMMU 把这个思路延伸到 multimodal, 这是关键的设计选择。

### 1.2 与已有 multimodal benchmark 的区别
当时主流 LMM benchmark — VQA-v2 (https://arxiv.org/abs/1612.00837), GQA (https://arxiv.org/abs/1902.09506), TextVQA (https://arxiv.org/abs/1904.08920), OK-VQA (https://arxiv.org/abs/1906.10767), ScienceQA (https://arxiv.org/abs/2209.09113), SEED (https://arxiv.org/abs/2307.16125), MMBench (https://arxiv.org/abs/2307.06281), MM-Vet (https://arxiv.org/abs/2308.02490) — 大量集中在 commonsense / daily knowledge, 图像类型偏 natural image (photo + painting + OCR text)。CogVLM 在 VQA-v2 上能拿到 85%, 在 ScienceQA-IMG 上 92%, 在 RefCOCO 上 93% — 这些数字已经接近饱和, 无法区分 model 的 expert 能力。MMMU 把这个 ceiling 拉低到 GPT-4V 也只有 55.7%, 这才是 informative benchmark 该有的状态。

MathVista (https://arxiv.org/abs/2310.02255) 同期出现, 但 scope 限定在 math; GAIA (https://arxiv.org/abs/2311.12983) 466 道题偏 reasoning + tool use。MMMU 11.5K, 30 subjects, 183 subfields, 30 heterogeneous image types — 这是它真正的 scaling。

## 2. Benchmark 的四个核心 challenge

Figure 1 概括得很清晰:

1. **Comprehensiveness**: 11.5K college-level problems, 6 disciplines × 30 subjects × 183 subfields
2. **Highly heterogeneous image types**: charts, diagrams, tables, chemical structures, music sheets, MRI/CT scans, maps, geometric shapes, DNA sequences, technical blueprints...
3. **Interleaved text-image inputs**: 图像可以出现在 question 的开头/中间/末尾, 也可能出现在 options 里 (3.37% 的题目); 7.39% 的题目有 multiple images
4. **Expert-level perception + reasoning rooted in domain knowledge**

第 4 点是真正的难点。看 Figure 2 里 Tech & Engineering 那道 transistor VCE 题:

> Find VCE for the circuit shown. Neglect VBE.
> Explanation: IE = VEE / RE = 5 V / 4 kΩ = 1.25 mA; VCE = VCC − IE·RL = 10 V − (1.25 mA)(5 kΩ) = 10 V − 6.25 V = 3.75 V

这里模型需要: (a) 从 circuit diagram 中正确读出 VCC, VEE, RL, RE 的拓扑连接与数值; (b) 知道 BJT 的 DC 分析中 IE ≈ VEE/RE (因为 VBE 被忽略); (c) 知道 VCE = VCC − IC·RL 且 IC ≈ IE; (d) 做对单位换算与算术。任一环节失败, 整题崩盘。

## 3. 数据构造 pipeline — 细节决定 benchmark quality

### 3.1 三阶段 collection
- **Stage 1 (subject selection)**: 基于 "visual inputs 在该学科中是否常见" 过滤掉 law、linguistics, 最终锁定 30 subjects
- **Stage 2 (annotation)**: 50+ college students annotators, 从 textbook + 网络资源收集, 必要时基于 expertise 自创问题
- **Stage 3 (cleaning)**: lexicon overlap + source URL similarity 查重; co-author 手动 format/typo check; difficulty 四级分类 (very easy / easy / medium / hard), 把 very easy 的约 10% 删除

### 3.2 Data contamination 防御
annotators 被要求 **优先选择答案不立即可见的题目** — 答案在 separate documents 或 textbook 末尾。这是社区越来越重视的问题 (https://arxiv.org/abs/2308.08193 关于 test set contamination in LMs)。MMMU 没有 test set 公开 labels, 只有 dev (5 per subject) 和 validation (900 题) 暴露, 这和 MMLU 一样的设计。

### 3.3 统计画像 (Table 1)

| 维度 | 数值 |
|---|---|
| Total questions | 11,550 |
| Dev : Val : Test | 150 : 900 : 10,500 |
| Difficulty (E : M : H) | 28% : 45% : 27% |
| Multiple-choice : Open | 94.03% : 5.97% |
| Questions with explanation | 17.62% |
| Avg question length | 59.33 tokens |
| Avg option length | 9.17 tokens |
| Avg explanation length | 107.92 tokens |
| Multiple-image questions | 7.39% |

Question 59.33 tokens 看起来不长, 但每张 image 携带的信息量远超 token count。这是 LMM 评估里 token-count 误导性最强的场景之一。

## 4. 评估协议细节

### 4.1 Answer extraction pipeline
评估的工程难点是 **从长回答中抽取最终答案**。作者设计了 rule-based regex + response-processing workflow, 提取 numbers 与 conclusion phrases。如果模型回答里没有 valid answer:
- multiple-choice → random selection
- open question → 直接判错

这意味着模型如果 "reasoning chain 正确但 final answer 不在指定格式" 会被惩罚, 也意味着 multiple-choice 的 lower bound ≈ random choice 23.9% (4 选 1 平均)。

### 4.2 Baseline 配置

**LMMs (text + image)**:
- Kosmos-2 (1.6B, https://arxiv.org/abs/2306.14824)
- LLaMA-Adapter2-7B (https://arxiv.org/abs/2304.15010)
- BLIP-2 FLAN-T5-XXL (https://arxiv.org/abs/2303.03952)
- InstructBLIP (https://arxiv.org/abs/2305.06500)
- LLaVA-1.5/1.6 (https://arxiv.org/abs/2310.03744, https://arxiv.org/abs/2401.03475)
- OpenFlamingo (https://arxiv.org/abs/2308.01390)
- CogVLM (https://arxiv.org/abs/2311.03079)
- Fuyu-8B (Adept)
- Qwen-VL (https://arxiv.org/abs/2308.12966)
- Otter (https://arxiv.org/abs/2305.03726)
- MiniGPT-4 (https://arxiv.org/abs/2304.10592)
- mPLUG-Owl2 (https://arxiv.org/abs/2311.04257)
- 加上后续加入的 InternVL (https://arxiv.org/abs/2312.14238), Yi-VL, VILA, InternLM-XComposer2-VL, MiniCPM-V, Reka, Claude 3, Gemini 1.0/1.5, GPT-4o

**Text-only LLMs**: GPT-4 text, Llama2-7B, FLAN-T5-XXL, Vicuna-13B; 还跑了 +OCR (MMOCR) 和 +LLaVA Caption 的 ablation — 这是验证 "OCR pipeline 能不能补齐 image 理解" 的关键实验。

**Human experts**: 90 college seniors (3 per subject), 各做 30 道对应 subject 的题, 允许查 textbook, 禁止上网。Best expert = 88.6%, Medium = 82.6%, Worst = 76.2%。这个 human ceiling 是整个 benchmark 的 anchor。

## 5. 主结果 (Table 2 / Table 4) — 关键发现

### 5.1 开源 vs 闭源的 gap
- 文章 submission 时点: BLIP2-FLAN-T5-XXL 与 LLaVA-1.5 ≈ 34%, GPT-4V = 55.7% — gap 约 21 个百分点
- 后续 v4 update: LLaVA-1.6-34B = 44.7%, InternVL-Chat-V1.2 = 46.2%, VILA1.5 = 46.9%, Qwen-VL-MAX = 46.8%, SenseChat-Vision-0423-Preview = 50.3%
- Claude 3 Opus = 59.4%, Gemini 1.5 Pro = 62.2%, GPT-4o = 69.1%

**V4 update 表明开源 model 在 MMMU 上的进展非常快** — 半年时间把 gap 从 21 缩到 ~3 个百分点 (LLaVA-1.6 vs GPT-4V 同期), 但与新闭源 model (GPT-4o) 又拉开新 gap。这是一个 "moving target" 现象, benchmark 的 effective lifetime 取决于 saturation rate。

### 5.2 OCR / Caption 对 text-only LLM 几乎无效
关键 ablation (Table 2 底部):
- FLAN-T5-XXL: 31.2% → +OCR 31.9% → +LLaVA Caption 31.9% — 几乎无提升
- Vicuna-13B: 31.0% → +OCR 31.9% → +LLaVA Caption 32.7% — 微提升
- GPT-4 text: 33.8%

**Intuition**: MMMU 的图像信息不是 "可以 OCR 出来转成文本" 的那种, 而是 circuit topology, 3D molecular configuration, music interval relationship, pathology slide 的 visual pattern。Caption 模型自己也理解不到, 所以传递给 LLM 的也是 garbage-in-garbage-out。这条结论对 LMM 设计有强烈 implication: **visual encoding必须与 reasoning 在同一 model 内 end-to-end**, pipeline-style (vision → text → LLM) 在 expert task 上必然失败。

### 5.3 学科间分布的强烈差异 (Table 2 后半部分)

| Discipline | Test count | GPT-4V |
|---|---|---|
| Art & Design | 1,163 | 65.3 |
| Business | 1,428 | 64.3 |
| Science | 2,426 | 48.4 |
| Health & Medicine | 1,752 | 63.5 |
| Humanities & Social Sci. | 947 | 76.3 |
| Tech & Engineering | 2,784 | 41.7 |

Tech & Engineering 与 Science 最低, Humanities 最高。Intuition: Humanities (history, literature) 的图像多为 photo / painting / cartoon, 与 pretraining 数据 distribution 接近; reasoning chain 较短。Tech 的图像多为 schematic / diagram, 需要 symbolic + spatial + numerical reasoning 的复合能力。

### 5.4 Image type 维度 (Table 13, Figure 4)
GPT-4V 在以下 image type 上特别弱:
- Geometric Shapes: 40.2% (开源 ~25%)
- Sheet Music: 38.8% (开源 ~35%)
- Mathematical Notations: 50.0% (开源 ~22%)
- Chemical Structures: 50.6% (开源 ~26%)
- Trees and Graphs: 38.9%

而在 Photographs 64.2%、Paintings 75.9%、Portraits 76.1%、Advertisements 100%、Logos 85.7% 上很强。

**Intuition**: 训练数据中 web-crawled image-text pair 对自然图像极度偏置; "结构化图像" (sheet music / chemical structures / circuit diagram) 的 supervised signal 不足。这暗示下一代 LMM 训练需要 synthetic expert data — 例如用 RDKit 渲染分子结构 + SMILES 描述, 用 LilyPond 渲染乐谱 + music theory 标注, 用 schematic generator 生成电路 + SPICE 仿真。

### 5.5 Difficulty 分解 (Table 3)
| Model | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Fuyu-8B | 28.9 | 27.0 | 26.4 | 27.4 |
| LLaVA-1.5-13B | 41.3 | 32.7 | 26.7 | 33.6 |
| GPT-4V | 76.1 | 55.6 | 31.2 | 55.7 |

**关键观察**: GPT-4V 在 Easy 76.1%, 在 Hard 31.2% — 几乎和 LLaVA-1.5 在 Easy 上一样 (相对其他模型)。Hard 任务上所有 model 的差异大幅压缩。这与 reasoning model 的 scaling law 不同, 说明 "复杂多模态 reasoning" 是 model class 共同的 bottleneck, 不是仅靠参数 scaling 能解决的。

## 6. Error analysis — 真正 informative 的部分

作者人工标注了 150 个 GPT-4V 错误样本 (Figure 5):
- **Perceptual Errors: 35%**
- **Lack of Knowledge: 29%**
- **Reasoning Errors: 26%**
- 其他: Textual Understanding 6%, Reject to Answer 3%, Annotation Error 2%, Answer Extraction 1%

### 6.1 Perceptual Errors (Figure 6, Figure 67, Figure 84)
两个 sub-type:
- **Basic perceptual**: 例如 Figure 6, "from left to right, top to bottom" 的基本 spatial ordering 错误。这是 "vision encoder + LLM token interface" 的 fundamental limitation — patch embedding 在被压缩成 visual token 后, fine-grained spatial relationship 容易丢失。
- **Domain-specific perceptual**: 模型把 circuit diagram 中的 resistor 看错 (Figure 84 的 interleaved transaction case), 或在 Figure 67 把美国 "Savior" 角色的 cartoon 误解为 "Imperialist" — 后者反映 model 的 textual prior 压过 visual evidence。

### 6.2 Lack of Knowledge (Figure 13, Figure 54, Figure 83)
- Figure 13: GPT-4V 正确识别 Las Meninas, 但不知道原作放在 King Philip IV study — 这是 factual knowledge deficit
- Figure 83: Computer Science 的 DFA, GPT-4V 看到 double circle 但不知道这表示 "accept state" — 这是 domain convention knowledge deficit
- Figure 54: 临床医学, 给了 cardiac catheterization 数据表 (Right atrium / Right ventricle / Pulmonary trunk / Left atrium / Left ventricle / Ascending Aorta 的 O2 saturation 与 pressure), 需要诊断 congenital heart disease。GPT-4V 看到数据但不知道 "step-up" 在 O2 saturation 上提示 VSD / ASD — 这是 medical knowledge recall failure

**Intuition**: 这类 error 暴露 LMM pretraining corpus 的 knowledge coverage gap。Web data 对 art history、音乐理论、电路分析、临床诊断的覆盖是稀疏且 noise 高的。RAG (retrieval-augmented generation) 与 domain-specific fine-tuning 是直接的修复路径。

### 6.3 Reasoning Errors (Figure 45, Figure 81, Figure 89, Figure 93)
最有价值的 case, 因为 perception + knowledge 都对了, 但 reasoning chain 断裂。

**Figure 45 (Calculus)**: 给定 f(x) = x³ − 6x² + 8x 和 g(x) = −½x² + 2x 的图像, 求 region R 的面积 integral expression。GPT-4V 没正确判断上下函数, 选错积分限。

**Figure 81 (Civil Engineering / Greenshields model)**: 给定 speed-density 数据, 用 linear regression 拟合 Greenshields model u = u_max − (u_max/k_jam)·k 求 k_jam。

Greenshields model 公式变量说明:
- u: speed (km/h)
- k: density (veh/km)
- u_max: free-flow speed (k=0 时的渐近值)
- k_jam: jam density (u=0 时的密度)
- linear form: u = m·k + c, 其中 m = −u_max/k_jam, c = u_max

GPT-4V 写出公式但 "skip 了 regression 计算", 直接猜最接近 85 的选项 = 110, 但正确答案 111。这说明 LLM reasoning 在 multi-step arithmetic (mean / variance / slope 估计) 上不稳定, 即使概念正确。

**Figure 89 (Thermodynamics / Polytropic process)**: 给定 helium gas polytropic compression, 求 heat transfer Q_12。

Polytropic process: PV^n = constant
- P: pressure (kPa)
- V: volume (m³)
- n: polytropic exponent (dimensionless)

First law for closed system: ΔU = Q_12 − W_12
- ΔU: internal energy change = m·Cv·(T2 − T1)
- Cv: specific heat at constant volume (helium ≈ 3.12 kJ/kg·K)
- W_12: boundary work = (P2·V2 − P1·V1)/(1−n)
- m: mass = P1·V1/(R·T1), R for helium ≈ 2.0831 kJ/kg·K

GPT-4V 把单位换算错 (T 没用 Kelvin)、把 mass 算错、把 calculation order 弄乱 (需要先解 n 才能算 V2 才能算 W 才能算 Q) — 这是典型 long-chain reasoning failure。

**Figure 93 (Mechanical Engineering / Engineering Dynamics)**: 两小球 mass m 沿圆弧下滑, 求水平位置速度 v 与 bottom 处法向力 N。

Energy conservation: ½·(2m)·v² = mg·R → v = √(gR)
- m: 每个球的质量
- g: 重力加速度
- R: 圆弧半径
- v: 共同速度

Bottom 处向心力方程: N − mg = m·v²/R → N = mg + m·(gR)/R = 2mg

GPT-4V 把 potential energy 算成 3mgR (把第二个球的高度当 2R), 整体 reasoning 错位。这是物理建模上的 conceptual slip, 不是 arithmetic error。

### 6.4 Reject to Answer (Figure 57, Figure 87)
GPT-4V 在 ophthalmic pathology case 拒绝给诊断, 在三相不平衡 Y-connected impedance case 拒绝算 complex phasor 算术。说明模型在被"安全性"过拟合时, 在 expert 域会过度保守; 也在"computation 量大"时主动放弃。

## 7. 从 MMMU 看到的 broader intuition

### 7.1 Vision-Language 接口的设计缺陷
当前主流 LMM 架构 (CLIP-ViT + projector + LLM):
- Visual token count 固定 (LLaVA: 576 tokens per image, BLIP-2: 32 queries)
- 对 expert image (高 information density 的 schematic), 576 tokens 远不够 — 一张 music sheet 上的 note + clef + time signature + accidental 信息量巨大
- 没有任何 mechanism 让 model 在需要时 "zoom in" 到 image 的局部区域

下一代 LMM 可能需要: dynamic-resolution tokenization (类似 Qwen2-VL 的 naive dynamic resolution, https://qwenlm.github.io/blog/qwen2-vl/), 或 hierarchical visual attention (Fuyu-Heavy 思路 — 直接 patch tokenize 而非 ViT pooling)。

### 7.2 Expert knowledge injection 仍是 open problem
MMMU error analysis 证明 RAG / fine-tuning / synthetic data 三个方向必须同时发力:
- **RAG**: 把 textbook knowledge 当作 retrieval source, 在 reasoning 前 retrieve
- **Domain SFT**: 构造大规模 (image, domain description, reasoning trace, answer) 的 instruction tuning set
- **Synthetic data**: 用 simulator (SPICE for circuits, RDKit for chemistry, LilyPond for music, matplotlib for charts) 生成无限 expert data

### 7.3 Test-time compute 在 multimodal 上还没充分开发
MMMU 的 hard reasoning case (Calculus, Thermodynamics, Engineering Dynamics) 完全适合 test-time scaling — 给 model 多轮 visual tool use (crop, zoom, OCR sub-region) + chain-of-thought + self-verification。这正是后续 OpenAI o1 / o3 与 GPT-4o 在 MMMU 上有大幅提升的关键 (GPT-4o = 69.1%)。

### 7.4 Benchmarks as moving targets
MMMU v1→v4 的 evolution 反映 multimodal field 进展速度: 半年内 open-source 从 34% 推到 50%, 闭源从 55.7% 推到 69.1%。下一代 benchmark 需要:
- 更大规模 (避免 saturation)
- 更 dynamic (避免 contamination)
- 包含 open-ended generation (避免 regex-extractable answer)
- 包含 agentic / tool-use (类似 GAIA)

## 8. 与同期 / 后续工作的脉络

- **MMLU-Pro** (https://arxiv.org/abs/2406.01574): text-only, 10 选 1, 提升 MMLU 难度
- **MathVista** (https://arxiv.org/abs/2310.02255): 同期, math-only multimodal
- **MMBench-1.5 / MMBench-CN**: 增加 robustness 维度
- **MMVet / LLaVA-Bench**: free-form evaluation
- **WildVision** (https://arxiv.org/abs/2406.03134): adversarial multimodal VQA
- **MMMU-Pro** (后续工作, https://arxiv.org/abs/2409.02813): 在 MMMU 基础上加难 (10 选 1, 更长 reasoning chain)
- **VisuLogic** (https://arxiv.org/abs/2504.03065): 专门测 visual reasoning, 排除 knowledge confounding
- **CogCom / Visual CoT 系列**: test-time visual reasoning tool use

## 9. 我会推荐 Karpathy 关注的几个具体方向

1. **Test-time compute × visual reasoning**: MMMU hard subset 是 test-time visual tool use (crop / zoom / scroll / query-perception) 最自然的 test bed。可以构建 visual-CoT benchmark: 让 model 选择下一步 "look closer at region X" 或 "extract number from table cell Y"
2. **Dynamic visual token budget**: 类似 Universal Vision-Language Pretraining 的思路, 让 visual token 数随 image complexity 变化。MMMU 上 sheet music / chemical structure 应该 allocate 远多于 photo 的 token
3. **Expert-domain synthetic instruction data**: 用 SPICE + schematic + BJT 的 DC/AC analysis 生成无限 electronics 题, 用 RDKit + IUPAC 生成无限 organic chemistry 题 — 这类数据可以让开源 LMM 在 MMMU Tech & Science 子集上快速攀升
4. **Modality balance loss**: 当前 contrastive 与 next-token prediction 的 loss weight 是固定的, 可以让 expert domain 的 visual token 参与更密集的 supervision (类似 Fuyu 直接 patch-as-token 思路)

## 10. 一句话总结

MMMU 把 multimodal benchmark 从 "commonsense VQA" 推到 "expert AGI" 范畴, 11.5K 题揭示出 LMM 在 perception-knowledge-reasoning 三层都有显著 gap, 且 gap 在 image type × domain × difficulty 三轴上有清晰可解释的分布 — 这使它成为 LMM 评估的 de facto standard。后续 v4 update 引入 Claude 3 / Gemini 1.5 / GPT-4o 显示 benchmark 仍在 discriminate, 但 saturation 速度比预期快, MMMU-Pro / next-gen benchmark 已经在路上。

**参考链接**:
- MMMU 主页: https://mmmu-benchmark.github.io/
- Live leaderboard: https://mmmu-benchmark.github.io/#leaderboard
- MMMU arXiv: https://arxiv.org/abs/2311.16502
- MMMU-Pro: https://arxiv.org/abs/2409.02813
- Hugging Face dataset: https://huggingface.co/datasets/MMMU/MMMU
- EvalAI evaluation server: https://eval.ai/web/challenges/challenge-page/2179
- 关联 LMM 综述 (LMM 的 dawn): https://arxiv.org/abs/2309.17421
