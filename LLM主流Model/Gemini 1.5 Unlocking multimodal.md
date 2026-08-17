---
source_pdf: Gemini 1.5 Unlocking multimodal.pdf
paper_sha256: abcd3b8e4b54721ff2d2323204f7b2ad9212b0ff6c8ed77725657526a5418b13
processed_at: '2026-08-04T13:04:43-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Gemini 1.5

好, 我把刚才那些技术细节掰碎了, 用更直白的方式讲一遍。

---

## 一句话总结

Google 做了个能一口气读 **1000万字** 的 AI, 而且读完之后你问它啥它基本都能答对, 不会被海量信息搞晕。

---

## 1. 为什么要搞这么长的 context?

先回顾下历史:

- **2020年 GPT-3**: 2K tokens (大概 1500 字英文)
- **2023年 GPT-4 Turbo**: 128K tokens (一本书)
- **2023年 Claude 3**: 200K tokens
- **2024年 Gemini 1.5**: **10M tokens** (1000万字, 或者说10本书, 或者5天音频, 或者10小时视频)

这是什么概念? 你可以把整个《战争与和平》(58万字) 扔进去, 外加整个 JAX 代码库 (4万行代码), 再塞进去一段45分钟的视频, 它还能正常工作。

**核心问题**: 之前的模型给你32K context, 你不得不把文档切碎, 用 RAG (retrieval-augmented generation) 去检索相关片段再喂给模型。这中间有信息损失, 而且检索本身可能找不准。Gemini 1.5 的思路是: **别检索了, 全塞进去**。

---

## 2. 怎么做到的? 两个关键

### 2.1 MoE: 稀疏激活

Gemini 1.5 Pro 用的是 **Mixture-of-Experts** 架构。打个比方:

- 传统 dense 模型像一个**全科医生**, 你问什么他都把全部脑细胞调动起来想一遍, 很累
- MoE 模型像一个**大医院**, 里面有几百个专科医生, 你来了一个case, 挂号处 (routing function) 看一眼, 把你分给最相关的2-3个科室看, 其他科室歇着

好处是什么? 你可以建一个超大的医院 (总参数量巨大), 但每次看病的成本很小 (激活参数少)。这就让 Gemini 1.5 Pro 在**训练用的算力**比 Gemini 1.0 Ultra (dense 模型) 少很多的情况下, 性能反而更好。

### 2.2 蒸馏出 Flash 版本

Flash 是 Gemini 1.5 Pro 的"青春版", 用 online distillation 训练 — 就是 Pro 在学的同时, Flash 在旁边模仿 Pro 的输出。好处是又快又便宜, 英文输出 1.5毫秒/字符, 比 GPT-4 Turbo 快好几倍。

---

## 3. 10M tokens 到底能不能用? 验证实验

### 3.1 大海捞针 (Needle-in-a-Haystack)

实验很简单: 拿一堆Paul Graham的essay拼成超长的文本(haystack), 在中间某个位置插一句话: "The special magic Tokyo number is: 12345" (needle), 然后问模型: "Tokyo的magic number是多少?"

结果:
- **文本**: 到530K tokens, 100%答对; 到1M, 99.7%答对; 到**10M**, 还有99.2%答对
- **视频**: 10.5小时的AlphaGo纪录片(7遍拼接), 在某一帧上打上文字, 100%找到
- **音频**: 107小时(将近5天)的多说话人录音, 插入3秒"secret keyword", 100%找到

对比一下: GPT-4 Turbo 只能处理128K, Claude 3最多200K。Gemini直接干到10M, 而且准确率几乎不掉。

### 3.2 更难的版本: 多根针

一根针太简单了。于是搞了个"100根针"版本: 在1M context里藏100个不同的magic number, 要求全找出来。

Gemini 1.5 Pro在128K处保持70%召回率, 到1M还有60%+。GPT-4 Turbo在128K就掉到50%左右。

### 3.3 更更难的版本: MRCR

这个实验设计得很巧妙。想象一段超长对话, 用户让模型写了好几首诗, 关于企鹅的诗、关于火烈鸟的诗、关于北极熊的诗...然后突然问: "把你写的第一首关于企鹅的诗再复现一遍。"

这要求模型不仅要找到那首诗, 还要区分"关于企鹅的诗"和"关于火烈鸟的诗" — 这叫**co-reference disambiguation**, 不仅仅是retrieval, 还有reasoning。

Gemini 1.5 Pro在32K之后就开始碾压GPT-4 Turbo和Claude 3

---

# Gemini 1.5 技术报告深度讲解

这篇 paper 是 Google DeepMind 发布的 Gemini 1.5 系列技术报告, 核心贡献在于将 multimodal LLM 的 context window 从 32K 推到 **10M tokens**, 同时不牺牲 core capabilities。下面从架构、长上下文机制、in-context learning、评估几个维度展开。

---

## 1. 模型架构

### 1.1 Gemini 1.5 Pro: Sparse MoE Transformer

Gemini 1.5 Pro 基于 **sparse Mixture-of-Experts (MoE)** 架构, 继承了 Google 在 MoE 上的长期积累 (Shazeer et al., 2017; GShard, Switch Transformer, GLaM 等)。

MoE 的核心思想是 **conditional computation**: 用一个 learned routing function 将 input 分配到 model 参数的一个子集进行处理。这样可以在保持单次 forward 激活参数量恒定的前提下, 爆炸式增大总参数量。

形式化地, 对于一个 MoE layer, 给定 input token 的 representation $h$:

$$h_{out} = \sum_{i=1}^{N} G(h)_i \cdot E_i(h)$$

其中:
- $N$ 是 expert 总数
- $E_i(\cdot)$ 是第 $i$ 个 expert (通常是一个 FFN)
- $G(h)_i$ 是 gating network 对第 $i$ 个 expert 的分配权重, 通常采用 top-k sparse routing, 即只保留 softmax 后最大的 k 个值, 其余置零

这种设计让 Gemini 1.5 Pro 在 **训练 compute 效率** 上大幅优于 Gemini 1.0 Ultra (一个 dense 模型), 却能达到甚至超越后者的性能。

### 1.2 Gemini 1.5 Flash: Dense Transformer with Distillation

Flash 走的是另一条路线: **dense Transformer decoder**, 通过 **online distillation** 从 Gemini 1.5 Pro 蒸馏而来。Online distillation 指的是 student 和 teacher 同步训练, student 实时模仿 teacher 的输出分布 (区别于 offline distillation 用固定的 teacher logits)。

Flash 的两个关键优化:
1. **Parallel computation of attention and feedforward components**: 传统 Transformer 是 sequential 执行 attention → FFN, Flash 让两者并行, 减少延迟
2. **Higher-order preconditioned optimization methods**: 使用比 Adam 更高阶的 preconditioner (如 Shampoo 或其变体, Becker & LeCun 1989; Heskes 2000) 来改善优化质量

结果是 Flash 在 latency 上极快: English 输出 1.5ms/character, 比 Claude 3 Haiku 快 30%+。

---

## 2. 长上下文: 10M tokens 的突破

### 2.1 Perplexity 服从 Power Law

这是这篇 paper 最让我兴奋的发现之一。作者测量了 cumulative average NLL (negative log-likelihood) 随 token position 的变化, 发现它服从 **power law**:

$$L(x) = \alpha x^{\beta} + \gamma$$

变量解释:
- $x$: token position in the sequence (从 1 开始)
- $L(x)$: cumulative average NLL up to position $x$
- $\alpha$: scale coefficient, 控制曲线整体幅度
- $\beta$: exponent (指数), 必须为负 (因为 NLL 随 context 增长而下降), 控制下降速率
- $\gamma$: asymptote (渐近值), 即 context → ∞ 时的 irreducible loss

关键观察:
- 对于 **long documents**, power law 拟合精确到 **1M tokens** (R² 很高)
- 对于 **code**, power law 精确到 **2M tokens**, 然后在 10M 处出现 deviation (因为 code 中存在重复的 code blocks, 偶尔提供 outsized benefit)
- Gemini 1.0 Pro 只能在 32K 内保持 NLL 下降, 之后就开始恶化

这个 power law 现象与 Kaplan et al. (2020) 发现的 **compute-loss scaling law** 形成对应: 那里是 loss 关于 compute 的 power law, 这里是 loss 关于 context length 的 power law。两者都暗示了某种 underlying 的 statistical structure。

**直觉构建**: 为什么 context 也会遵循 power law? 一个可能的解释是自然语言中存在 hierarchical 的 long-range dependency — 局部依赖 (几个 token) 提供大部分信息增益, 但更远距离的依赖 (段落级、文档级、语料级) 仍持续提供边际收益, 只是收益递减遵循某种重尾分布。

### 2.2 Needle-in-a-Haystack: >99% Recall 到 10M

实验设计:
- **Haystack**: Paul Graham essays 拼接填充到目标 context length
- **Needle**: "The special magic {city} number is: {number}" 插入到不同 depth (0%, 25%, 50%, 75%, 100%)
- **Query**: 询问某个 city 的 magic number

结果:
- **Text**: 100% recall up to 530K tokens, 99.7% at 1M, 99.2% at **10M**
- **Video**: 100% recall up to 10.5 hours (9.9M tokens), 在 AlphaGo 纪录片 7 倍拼接中插入单帧文字
- **Audio**: 100% recall up to 107 hours (9.7M tokens), 在 VoxPopuli 多说话人语料中插入 3 秒 "secret keyword"

对比 GPT-4 Turbo 的 128K context, Claude 3 的 200K context, 这是数量级上的飞跃。

### 2.3 Multi-needle 与 MRCR: 更难的 retrieval

单 needle 太简单, 作者设计了更难的版本:

**Multiple needles**: 在 1M context 中插入 100 个 unique needles, 要求全部 retrieve。Gemini 1.5 Pro 在 128K 处保持 70% recall, 1M 处保持 60%+, 而 GPT-4 Turbo 在 128K 就掉到 ~50%。

**MRCR (Multi-round Co-reference Resolution)**: 这是我认为设计得很巧妙的任务。在一个长对话中, 用户请求 model 写 poems/riddles/essays on different topics, 然后插入两个 adversarially similar 的 requests (比如 "poem about penguins" vs "poem about flamingos"), 之后要求 model 复现其中一个。这测试的不仅是 retrieval, 还有 **disambiguation** 能力。Gemini 1.5 Pro 在 32K 后超越 GPT-4 Turbo 和 Claude 3 Opus, 一直保持到 1M tokens。

---

## 3. In-Context Learning 的惊人能力

### 3.1 Kalamang 翻译: 从一本书学会

这是整篇 paper 最震撼的实验。Kalamang 是巴布亚新几内亚的一种语言, 全球不到 200 个 speaker, 几乎没有网络存在, 因此可以确保 model 在 pre-training 中没见过。

提供的 context:
- ~500 页 reference grammar
- ~2000 entry 双语 wordlist
- ~400 parallel sentences
- 总计 ~250K tokens

评估用 MTOB (Machine Translation from One Book) benchmark, 与一个学习过相同材料的人类对比。

结果 (0-6 scale, 6 best):

| Context | Gemini 1.5 Pro | Human learner |
|---------|----------------|---------------|
| 0-shot | 0.18 | - |
| Half book | 4.14 (kgv→eng) | - |
| Full book | **4.00** (kgv→eng), **5.46** (eng→kgv) | 5.52 (kgv→eng), 5.60 (eng→kgv) |

关键 insight:
- eng→kgv 方向, Gemini 1.5 Pro (5.46) 已经接近 human learner (5.60)
- kgv→eng 方向, 差距稍大 (4.00 vs 5.52), 因为人类在 retrieval 上更强
- 0-shot 所有模型都是随机水平, 证明 pre-training 确实没见过 Kalamang

### 3.2 ASROB: In-Context Speech Recognition

更进一步: 在 context 中加入 45 分钟 Kalamang 语音 + 转写, 测试 model 能否学会 ASR。这是首次在 LLM 中实现 **mixed-modal in-context learning** (text + audio documentation)。

CER (Character Error Rate) 结果:

| Text context | Audio context | Gemini 1.5 Pro CER |
|--------------|---------------|---------------------|
| none | 0-shot | 35.0% |
| both | 800-audioshot | **22.9%** |

从 35% 降到 22.9%, 表明 model 确实从 in-context audio examples 中学到了语音-文字映射。

### 3.3 Many-shot ICL Scaling

在 6 个低资源语言 (Acholi, Abkhaz, Navajo, Bemba, Ewe, Kurdish) 上的 translation 实验, scaling shots 从 0 到 ~1K/4K:

- Gemini 1.5 几乎单调改善, 没有饱和迹象
- GPT-4 Turbo 改善缓慢且很快饱和
- 对 Navajo, 1.5 Pro/Flash 分别获得 +9.5/+15.9 chrF 的 zero-shot 到 many-shot 增益

这暗示了一个新范式: **many-shot ICL** 可以替代 fine-tuning, 尤其适合低资源场景。

---

## 4. 核心能力评估

### 4.1 数学推理

Math-specialized Gemini 1.5 Pro 的结果令人瞩目:

| Benchmark | Gemini 1.5 Pro | Math-Specialized | Claude 3 Opus | GPT-4 Turbo |
|-----------|----------------|------------------|---------------|-------------|
| MATH | 67.7 | **80.6** (91.1 rm@256) | 60.1 | 73.4 |
| AIME 2024 | 2/30 | **7/30** (8/30 rm@256) | 2/30 | 1/30 |
| HiddenMath | 20.1 | **35.2** | 17.3 | 24.6 |

注意 **rm@256**: sample 256 个 solutions, 用 reward model 选最佳。这本质上是 inference-time compute scaling, 类似 OpenAI 的 o1 思路。

一个有趣的 trick: 把整个 SymPy + SciPy 仓库 (~730K tokens) 塞进 context, 让 model 用 Python 解决数学问题, 在 Intermediate Algebra (Levels 4-5) 上从 18.6% 提到 25.8%。

### 4.2 长视频理解

引入新 benchmark **1H-VideoQA**: 125 个 5-way MCQ, 视频 40-105 分钟长。

| Model | 16 frames | 150 frames | Full video (1fps) |
|-------|-----------|------------|---------------------|
| GPT-4V | 36.5% | 52.3% | Not supported |
| Gemini 1.5 Pro | 45.2% | 56.3% | **72.2%** |

关键发现: 1H-VideoQA 比 EgoSchema 更能区分长上下文模型, 因为 EgoSchema 在 16 frames 后就饱和了, 而 1H-VideoQA 持续上升。

### 4.3 Long-document QA: Les Misérables

用 710K tokens 的《悲惨世界》全文做 QA, 用 Bradley-Terry 模型比较不同系统:

$$P(M_A \text{ answers better than } M_B) = \frac{e^{\beta_A}}{e^{\beta_A} + e^{\beta_B}}$$

变量:
- $\beta_A, \beta_B$: model A 和 B 的 "strength" 参数, 通过 maximum likelihood estimation 拟合
- $P$: A 比 B 更好的概率

结果: full-context Gemini 1.5 Pro 比 retrieval-augmented (4K tokens) Gemini 1.5 Pro 在 78% 的情况下给出更好答案, 比 RAG GPT-4 Turbo 在 83% 情况下更好。这直接挑战了 RAG 的必要性 — 当 context 足够长, 直接全塞进去可能更好。

---

## 5. 安全性: Jailbreak 与 Prompt Injection

### 5.1 Jailbreak Robustness

测试三种威胁模型:
- **Blackbox** (Template attack): -51% vs Gemini 1.0 Ultra
- **Greybox** (Template + Mutations): +7% (略差)
- **Whitebox transfer** (GCG from Gemini 1.0 Nano): -6%

有意思的发现: Gemini 1.5 对 **gradient-based** 攻击 (GCG) 抵抗力强, 但对 **human-readable** prompt injection 反而更脆弱。作者假设这是因为 instruction following 能力增强, 反而让 model 更容易被 "自然语言指令" 操纵。

### 5.2 Long-context Safety

设计了 **adversarial needle-in-haystack**: 把 safety-violating prompt 包在 instruction tags 里, 埋在长 context 中。

结果: long-context 的 violation rate **低于** short-context (平均 -28.6%)。原因不是 model 更安全, 而是 model 经常 **找不到 needle**。一旦找到, violation rate 与 short-context 相当。这预示着: 随着 long-context 能力进一步提升, 这类安全风险会上升。

---

## 6. 训练基础设施

- 硬件: 多个 4096-chip TPUv4 pods, 跨数据中心
- 软件: JAX + ML Pathways + GSPMD (自动并行)
- 数据: multimodal, multilingual, 包含 web documents, code, image, audio, video

Pre-training 数据的去污染采用 standard n-gram 方法, 但作者承认这是 imperfect 的, 因此引入了 held-out internal benchmarks (PhysicsFinals, HiddenMath, Natural2Code)。

---

## 7. 关键 Insight 总结

1. **MoE + 长上下文是互补的**: MoE 提供 compute 效率, 长上下文提供信息容量, 两者结合实现了 "用更少 compute 达到更强能力"。

2. **Power law in context length** 暗示了一个 unified scaling picture: 就像 compute-loss 遵循 power law, context-loss 也遵循, 这为预测更长 context 的收益提供了工具。

3. **Many-shot ICL** 可能替代 fine-tuning, 尤其在低资源/长尾场景。Kalamang 实验证明 LLM 可以从 in-context documentation 学到接近人类学习者的能力。

4. **RAG vs Long-context**: 当 context 足够大且模型足够强, 直接全塞可能比 RAG 更好 (Les Misérables 实验)。但 RAG 在 compute 成本上仍有优势。

5. **能力增强带来新风险**: instruction following 增强反而让 prompt injection 更容易成功; long-context 增强会在未来让 buried safety violations 更可能被触发。

6. **评估方法论需要进化**: single needle 已经过饱和, 需要 multi-needle, MRCR, 1H-VideoQA 这类更难的任务来区分模型。

---

## 参考 Web Links

- 论文: https://arxiv.org/abs/2403.05530
- Gemini 1.0 报告: https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf
- MTOB benchmark: https://arxiv.org/abs/2309.16575
- Needle-in-a-Haystack 原始 repo: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- JailbreakBench: https://jailbreakbench.github.io/
- Berkeley Function Calling Leaderboard: https://gorilla.cs.berkeley.edu/blogs/13_bfcl.html
- Many-shot ICL paper: https://arxiv.org/abs/2404.11018
- EgoSchema: https://egoschema.github.io/
- 1H-VideoQA (包含在论文 supplementary)
- Kalamang grammar: https://dictionaria.clld.org/contributions/kalamang
- Google AI Principles: https://ai.google/responsibility/principles/
- Bradley-Terry model 原始论文: Bradley & Terry (1952), Biometrika

---

如果想深入某个方面 (比如 MoE routing 的具体实现, 或 many-shot ICL 的机制解释), 我可以进一步展开。
