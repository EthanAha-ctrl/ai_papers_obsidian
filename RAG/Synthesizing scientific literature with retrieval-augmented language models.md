---
source_pdf: Synthesizing scientific literature with retrieval-augmented language models.pdf
paper_sha256: 87860750103c7d1fe8080680e2ff0f13b79c79134d3cdd4ab86ab19b18c84253
processed_at: '2026-08-12T11:51:13-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OpenScholar

## 一句话说清楚这篇 paper 在干嘛

科学家每天要读海量论文，写综述、回答"这个领域最近有啥进展"这种问题。这事很累，能不能让 AI 帮忙？这篇 paper 做了一个叫 **OpenScholar** 的系统，专门帮科学家做文献综述，而且做得比 GPT-4o 和人类专家还好。

## 为啥这事难

你可能想，这事 GPT-4o 不就能干吗？问它"retrieval-augmented LM 最近有啥进展"，它不就答了吗？

问题在于 **GPT-4o 会瞎编论文**。你让它引用 recent literature，它在 78-90% 的情况下会编造根本不存在的论文标题。听起来很像真的，作者名、期刊名、年份都 plausible，但拿去 Semantic Scholar 一查——不存在。Llama 3.1 8B 更夸张，biomedicine 领域 97.6% 的引用是编的。

为啥会这样？因为 LLM 的知识锁死在 pre-training 截止日期，而且 scientific paper 是 long-tail knowledge——网上讨论少、训练语料里出现频次低，模型记不住。它为了"回答看起来合理"，就 hallucinate 一个 plausible 的标题。

这就像你问一个聪明但不爱读书的人"最近 X 领域有啥新进展"，他会编一些听起来很像的论文标题来应付你。

## OpenScholar 怎么解决的

核心思路其实很朴素：**别让模型凭记忆答题，让它开卷考试**。

### 第一步：建一个大书库（OSDS）

作者们收集了 4500 万篇 open-access 论文，切成 2.36 亿个段落，每个段落都算了 embedding 向量。这个叫 OSDS（OpenScholar DataStore）。

切分方式很暴力——每 256 词切一刀，不管 section 边界。为啥不用 semantic segmentation？因为 2.36 亿段落跑 segmentation 模型太贵，简单粗暴的固定切分在工程上更划算。

### 第二步：找到相关段落（Retrieval）

给一个问题，怎么从 2.36 亿段落里找相关的？两阶段 funnel：

1. **Bi-encoder 初筛**：把 query 和段落都编码成向量，算 cosine similarity，快速召回 top-70。这个 encoder 是在 scientific 语料上继续 pre-train 过的 Contriever，110M 参数，小而专。

2. **Cross-encoder 精排**：把 query + 段落拼一起送进 340M 参数的 reranker，joint encoding，输出一个 relevance score。这个 reranker 是用 Llama 3.1 70B 生成的 synthetic data 训的——让 70B 模型给段落打 1-5 分 relevance，4-5 分当正例，1-2 分当负例，3 分丢弃。

除了 OSDS dense retrieval，还混了 Semantic Scholar API（keyword search）和 You.com web search（限 arXiv/PubMed）。三者结合效果最好，单 web search 效果最差（因为返回的是 HTML 不是 paper 段落，attribution 困难）。

### 第三步：生成回答 + 自我反思（Self-feedback loop）

这是跟标准 RAG 最大的区别。标准 RAG 就是一次生成：

> 给问题 + 检索到的段落 → 生成答案 → 完

OpenScholar 加了个 "草稿 → 自我批评 → 修订" 的循环：

1. 先生成初稿 $y_0$（带 in-line citation markers）
2. 模型自己给初稿写 feedback，最多 3 条自然语言句子，比如"只讲了 QA 任务的结果，该补充其他任务类型的结果"或者"第二段缺引用"
3. 如果 feedback 指出有内容缺失，模型还会生成一个 retrieval query 去补检索
4. 按 feedback 逐条修订，得 $y_1, y_2, ..., y_T$
5. 最后做 citation verification——检查所有该引用的 claim 有没有引用，没有的就补上

这就像一个写作过程：先写草稿，然后自己挑毛病，然后改。好处是 coverage 更全、citation 更准。坏处是慢一点（多次 forward pass）。

### 第四步：用这条 pipeline 蒸馏出一个小模型

这是最聪明的工程决策。作者们用 Llama 3.1 70B 跑上面这套完整 pipeline，生成 10K 个高质量的 question-answer 轨迹，包括初稿、feedback、修订稿。然后做两轮过滤：

- **Pairwise filtering**：比较修订稿 $y_T$ 和初稿 $y_0$ 哪个更好，留好的那个。约 20% 情况下初稿反而更好（因为改多了变啰嗦或跑偏）。
- **Rubric filtering**：让模型评 organization 和 citation accuracy，5 分制，必须两项都 ≥4.5 分才保留。

然后用这批高质量数据 fine-tune Llama 3.1 8B。关键设计是训练数据包含三类：
- $(x \to y)$：从问题生成答案
- $(y_0 \to \mathbf{F})$：从初稿生成 feedback
- $(y_{t-1}, f_t \to y_t)$：给定 feedback 做修订

混合 50% scientific 数据 + 50% 通用 instruction tuning 数据，再加 10 万条 fact verification 数据。

结果是：**fine-tuned 8B 模型在 inference 时不走 feedback loop 也能达到接近完整 loop 的效果**。Ablation 显示去掉 feedback loop，OpenScholar-8B 只掉 1.5 分，而 OpenScholar-GPT-4o 掉 5.3 分。这说明 8B 训练时已经把"反思再修订"的模式内化成了 single-pass 能力。

这就像一个学生，先跟着老师手把手做"先写草稿再修改"的练习，练多了之后，一次就能写出接近草稿+修订水平的终稿。

## 评估：怎么知道这东西真的好用

作者们建了个新 benchmark 叫 **ScholarQABench**，包含 7 个子数据集：

- 3 个 single-paper 任务（SciFact、PubMedQA、QASA）——拿来 reformulated 成 open-retrieval 设置
- 4 个 multi-paper 任务——新标注的，需要综合多篇论文

最核心的是 **Scholar-CS**（100 个 CS 文献综述问题）和 **Scholar-Multi**（108 个跨 CS/物理/生物医学的问题，每个答案由 PhD 专家花约 56 分钟写）。

评估指标三个维度：
- **Correctness**：单论文用 accuracy/ROUGE-L，多论文用 rubric score（专家标注的 must-have/nice-to-have ingredients）
- **Citation F1**：citation precision 和 recall 的调和平均
- **Content quality**：relevance、coverage、organization，用 Prometheus v2 当 LLM judge + 人类专家评估

## 关键结果

### 8B 小模型打败 GPT-4o

在 Scholar-CS rubric score 上：
- GPT-4o（无检索）：45.0
- OpenScholar-8B：51.1（**+6.1**）
- OpenScholar-GPT-4o：57.7（+12.7）

Citation hallucination 比率：
- GPT-4o：CS 78.7%，Bio 94.8%
- OpenScholar-8B：**0% / 0%**

### 人类专家评估

16 位 PhD 级专家在 108 个问题上对比 model 和 human 答案：
- GPT-4o（无检索）：赢 31.9% 的时间
- OpenScholar-8B：赢 50.8%
- OpenScholar-GPT-4o：赢 70%

OpenScholar 的主要优势是 **coverage**——覆盖的论文更多、深度更深。GPT-4o 无检索时 organization 分高（写得更流畅）但 coverage 低（覆盖面窄）。

作者们还做了 length control 实验：把 OpenScholar-GPT-4o 的答案压缩到约 333 词（跟人类答案长度接近），重新让专家评，仍然赢/tie 75%。说明优势不纯粹来自"写得更长"。

### 成本

- OpenScholar-8B：$0.003/question
- OpenScholar-GPT-4o：$0.05/question
- PaperQA2：$0.3-2.3/question
- GPT-4o：$0.006/question

OpenScholar-8B 比 PaperQA2 便宜 100 倍但性能更好。

## 这事为什么成立：我的 intuition

### 1. Knowledge 外置比堆参数有效

8B Llama 没有任何 magical reasoning 能力。它打败 GPT-4o 的根本原因是：**知识存在 OSDS 里而不是参数里**。GPT-4o 再大也记不住 4500 万篇论文的细节，而 OpenScholar 让 8B 模型当"合成器"——它只需要会读、会综合、会引用，不需要记住。

这就像开卷考试 vs 闭卷考试。一个中等学生带一本好字典，能比一个没带字典的优等生答得更好。

### 2. 专门化 retrieval 比通用检索强

Off-the-shelf Contriever 在 scientific domain 上 out-of-domain 表现差。作者在 peS2o 上继续 pre-train，让 bi-encoder 学到 scientific language 分布。Reranker 也是用 synthetic scientific relevance labels fine-tune 过的。

这说明 retrieval system 也需要 domain specialization，不是拿个通用模型就够。

### 3. Inference-time loop 可以 distill 成 parametric 能力

这是最让我感兴趣的部分。作者用 inference pipeline 生成训练轨迹，包含初稿、feedback、修订稿。然后训练 8B 模型。结果是 8B 在 single-pass 时就表现出类似 multi-step refinement 的能力。

这本质上是一种 **inference-time compute → training-time knowledge** 的蒸馏。self-feedback loop 是 inference-time 的"慢思考"，通过训练数据把它内化为 single-pass 的"快思考"。

### 4. Citation accuracy 是 scientific tooling 的门槛

学术界不会接受一个 78% 引用是编造的工具。OpenScholar 把 hallucination 砍到 0% 是关键 enabler。这说明未来 academic tooling 的核心 metric 不是 BLEU/ROUGE，而是 citation F1 + claim support rate。

### 5. Long-context ≠ Long-effective-context

Extended Data Fig 3 显示，Llama 3.1 8B 虽然支持 128K context，但检索段落从 10 增加到 20 时性能就开始降。而 OpenScholar-8B 训练后到 N=20 还 robust。

这说明长上下文利用能力需要专门训练，不是 architecture 给了就免费。这也呼应 "Lost in the middle" 现象——模型对 context 中间的信息利用差。

## 局限性

作者自己坦白了几个问题：

1. **ScholarQABench 标注量小**（CS 110 题，Multi 108 题），annotator 偏差大，静态 benchmark 有 contamination 风险，领域覆盖不全（没 social sciences）
2. **Retrieval 不总能找到最代表性的论文**，缺少 citation network prior
3. **8B 模型 instruction-following 能力弱**，有时输出 factual 不准确
4. **不用 license-protected 论文**，损失部分覆盖
5. **Human eval 没仔细查 citation precision/recall**，annotator 可能更关注写作质量

## 我会关注的 follow-up

1. **Citation graph prior**：在 retrieval 时加入论文间的引用关系，可能找到更 representative 的工作
2. **Multi-modal**：把 figure/table/equation 也纳入 datastore
3. **Multi-turn**：现在的 single-turn 问答无法做深度 follow-up
4. **RL from human citation correction**：用真实人类对 citation 的修正做 RL signal
5. **Long-horizon synthesis**：不是回答一个问题，而是写一个完整的 survey

## 跟你（Karpathy）可能相关的几个点

- 这篇是 "specialized retrieval + small LM" 范式的胜利，跟你在 Tesla/教育领域看到的 "用对工具比堆参数重要" 思路一致
- Self-feedback loop 本质是把 System 2 慢思考显式化然后用训练蒸馏回 System 1，跟你讲过 的 "o1-style reasoning 蒸馏" 思路同源
- Synthetic data pipeline 的 self-improving loop 形态值得关注——inference pipeline 生成训练数据 → 训练 → 更好的 inference pipeline，如果加 RL 可能更强
- Citation accuracy 作为 LLM 进入专业领域的门槛 metric，这个观察可能延伸到 code（function citation?）、法律（case citation）等领域

相关资源链接：
- Paper: https://doi.org/10.1038/s41586-025-10072-4
- Code: https://github.com/AkariAsai/OpenScholar
- Demo: https://openscholar.allen.ai
- Benchmark: https://github.com/AkariAsai/ScholarQABench
- Training data: https://huggingface.co/datasets/OpenSciLM/OS_Train_Data

---

# OpenScholar: 科学文献综述的 retrieval-augmented LM

## 1. 核心问题与动机

这篇 paper 攻击的问题非常具体：**科学文献综合**。当一个 researcher 想问 "What are the recent advances in retrieval-augmented language models for scientific tasks?" 这类问题，需要：
- **Retrieval precision + recall**：从数千万 papers 中找相关 passages
- **Synthesis**：综合多篇 paper 的 findings，而非简单 Q&A
- **Citation accuracy**：每个 claim 都要有可追溯的 reference
- **Up-to-date**：不能依赖 stale 的 pre-training knowledge

作者们在 motivation 里给出一个 striking 的数字：**GPT-4o 在 no-retrieval 设置下 fabricated citations 的比率是 78-90%**（biomedicine 甚至 94.8%）。这个数字的本质是 LLM 的 parametric knowledge 在 long-tail 科学知识上严重不可靠。Llama 3.1 8B 在 biomedicine 上 fabricated 比例 97.6%。这是 Kandpal et al. 2023 发现的 "long-tail knowledge" 问题的极端表现——paper 标题听起来 plausible，但根本不存在。

相关链接：
- PaperQA2 (Skarlinski et al. 2024): https://arxiv.org/abs/2409.13740
- Self-RAG (Asai et al. 2024): https://openreview.net/forum?id=hSyW5go0t8
- 当 not to trust LLMs (Mallen et al. 2023): https://aclanthology.org/2023.acl-long.846/

## 2. OpenScholar 架构解析

OpenScholar 不是一个单一的 model，而是一条 inference pipeline，核心方程：

$$y, \mathbf{C} = G(x, R(x, \mathbf{D}))$$

变量含义：
- $x$：scientific query（输入问题）
- $\mathbf{D}$：OSDS data store（45M papers，236M passages）
- $R$：retriever（包含 bi-encoder + cross-encoder + 外部 API）
- $R(x, \mathbf{D}) = \mathbf{P} = \{p_1, p_2, ..., p_N\}$：top-N retrieved passages
- $G$：generator LM（可以是 fine-tuned Llama 3.1 8B / 70B / GPT-4o）
- $y$：generated response
- $\mathbf{C} = c_1, c_2, ..., c_K$：in-line citations，每个 $c_i$ 对应 $\mathbf{P}$ 中的某段 passage

### 2.1 OSDS（OpenScholar DataStore）

- **Source**：peS2o v3（基于 S2ORC 的 open-access subset）
- **Scale**：45M papers，236M passage embeddings
- **Time cutoff**：peS2o v3 截至 2024 年 10 月，peS2o v2 用于评估（截至 2023 年 1 月）
- **Passage 切分**：每 256 词一个 chunk，加上 paper title 作为 passage prefix
- **Embeddings**：236M passages 都有预计算 dense embedding

切分策略值得注意：作者没有用 semantic segmentation（比如按 section 切），而是固定长度切分。原因是 (1) 不是所有 paper 都保留 semantic structure；(2) 在 236M 量级上跑 segmentation 模型算力太贵。这是一个 scalability vs. precision 的 trade-off。

相关链接：
- peS2o: https://huggingface.co/datasets/allenai/peS2o
- S2ORC (Lo et al. 2020): https://aclanthology.org/2020.acl-main.647/
- Scaling retrieval datastore (Shao et al. 2024): https://arxiv.org/abs/2407.12854

### 2.2 Retriever + Reranker 两阶段

**Bi-encoder retriever $\theta_{\text{bi}}$**：
- 110M params，初始化自 Contriever
- Continual pre-training on peS2o，unsupervised（contrastive + ICT）
- Query 和 passage 独立编码成 dense vector，做 nearest neighbor search
- 检索 top-70 passages 作为 candidates

**Cross-encoder reranker $\theta_{\text{cross}}$**：
- 340M params，初始化自 BGE reranker
- Fine-tune on synthetic data：用 Llama 3.1 70B 对 peS2o abstracts 生成 query，retrieve top-10，让 LLM 打 1-5 分 relevance，4-5 分正例，1-2 分负例，3 分丢弃
- Joint encode query + passage，输出 relevance score
- Meta-filtering：(1) 限制每篇 paper 最多 3 个 passages；(2) 加入 normalized citation count 作为 prior

为什么要两阶段？Bi-encoder 速度快（向量内积）但精度低，cross-encoder 精度高但慢。先 bi-encoder 召回 top-70，再 cross-encoder rerank 选 top-N。这是一个经典的检索 funnel。

### 2.3 多源 retrieval pipeline

OpenScholar 不只用 OSDS dense retrieval，还融合三个来源：
1. **OSDS dense retrieval**（核心，由 $\theta_{\text{bi}}$ 完成）
2. **Semantic Scholar API**（keyword search，按 citation count 排序，取 top-10 papers）
3. **You.com web search**（限制 arXiv/PubMed 学术站点）

Ablation 数据显示三者结合最优：单 OSDS 49.3/44.0（rubric/citation），单 S2 49.1/32.5，单 web 45.9/12.6，组合 49.6/47.6。

注意 web-only citation F1 只有 12.6，因为 web search 返回的多是 HTML 而非 paper passages，attribution 困难。

### 2.4 Self-feedback inference loop（关键创新）

这是 OpenScholar 与 standard RAG 最大区别。Standard RAG 一次性生成：

$$y_0 = G(x, \mathbf{P})$$

OpenScholar 引入三步迭代：

**Step 1: Initial response + feedback generation**
- LM 生成 $y_0$（带 in-line citation markers）
- LM 接着对 $y_0$ 生成 feedback $\mathbf{F} = f_1, f_2, ..., f_T$（最多 3 条）
- 每条 feedback $f_t$ 是自然语言句子，比如 "The answer only includes empirical results on QA tasks. Add results from other task types."
- 如果 feedback 指出 missing content，LM 同时生成 retrieval query $q_t$

**Step 2: Iterative refinement**
- 对每条 $f_k$：
  - 若需要 retrieval，用 $q_k$ 取 extra passages，append 到 $\mathbf{P}$
  - LM 用 $(y_{k-1}, \mathbf{P}, f_k) \to y_k$
- 重复直到所有 feedback 处理完，得 $y_T$

**Step 3: Citation verification**
- 检查 $y_T$ 中所有 citation-worthy statements 是否有 reference 支持
- 若有 unsupported claims，做 post-hoc citation insertion
- 注意：不会删除 sentence，只补 citation

为什么这个 loop 有效？标准 RAG 的问题是：(1) 一次生成易遗漏；(2) citation 容易错位。Self-feedback 把 "草稿 → 自我批评 → 修订" 这个写作 workflow 显式化。这与 Self-RAG (Asai et al. 2024) 思路类似，但 Self-RAG 用 predefined reflection tokens，OpenScholar 用 free-form natural language feedback，更灵活。

相关链接：
- Self-RAG: https://arxiv.org/abs/2310.11511
- Active retrieval augmented generation (Jiang et al. 2023): https://aclanthology.org/2023.emnlp-main.495/
- Reflexion (Shinn et al. 2023): https://arxiv.org/abs/2303.11366

### 2.5 Training data synthesis（关键工程创新）

OpenScholar-8B 之所以能打 GPT-4o，核心在于 **用 inference pipeline 生成 training data** 来 distill。

数据生成 pipeline：
1. 从 peS2o 采样 1M paper abstracts + metadata
2. 筛选 10K 篇 2017 年后高引论文
3. 用 LM 生成 literature review questions（要求多 paper 才能回答）
4. 用 OpenScholar pipeline（基于 Llama 3.1 70B）生成完整 trajectory：$y_0, \mathbf{F}, y_T$

数据过滤（two-step）：
- **Pairwise filtering**：比较 $y_T$ vs $y_0$，保留更好的一个（约 20% 情况下 $y_0$ 更好，因为 over-editing 或 redundancy）
- **Rubric filtering**：用 LM 评分 organization + fact/citation accuracy，5 分制，需 ≥4.5 分才保留

最终训练数据三类：
- Answer generation: $(x \to y)$
- Feedback generation: $(y_0 \to \mathbf{F})$
- Feedback incorporation: $(y_{t-1}, f_t \to y_t)$

数据混合：50% scientific domain + 50% general instruction tuning（含 SciRIFF + Tulu v3），还额外加了 100K 篇高引 paper 的 fact verification + boolean QA 数据。

训练：Llama 3.1 8B Instruct，130K instances，2 epochs。

这是一个 **self-improvement loop** 的典范：用 inference pipeline 生成训练数据 → 训练小模型 → 小模型可以在 inference 时不走完整 feedback loop 也达到类似效果。论文 ablation 显示 OpenScholar-8B 去掉 feedback 只掉 1.5 分 rubric，而 OpenScholar-GPT-4o 去掉 feedback 掉 5.3 分——这说明 8B 训练时已经把 feedback 模式内化了。

相关链接：
- SciRIFF (Wadden et al. 2024): https://aclanthology.org/2024.findings-emnlp.396/
- Tulu 2 (Ivison et al. 2023): https://arxiv.org/abs/2311.10702
- Superfiltering (Li et al. 2024): https://aclanthology.org/2024.acl-long.740/

## 3. ScholarQABench 设计

### 3.1 Benchmark 结构

| Dataset | Task | Domain | Size | Metrics |
|---------|------|--------|------|---------|
| SciFact | Claim → T/F | Biomedicine | 208 | Acc., Cite |
| PubMedQA | Q → Yes/No | Biomedicine | 843 | Acc., Cite |
| QASA | Q → Long-form | CS | 1,375 | Acc. (ROUGE-L), Cite |
| Scholar-CS | Q → Long-form + rubric | CS | 100 | Rub., Cite |
| Scholar-Bio | Q → Long-form | Biomedicine | 1,451 | Cite |
| Scholar-Neuro | Q → Long-form | Neuroscience | 1,308 | Cite |
| Scholar-Multi | Q → Long-form | CS/Physics/Bio | 108 | Cite, LLM, Exp. |

**Single-paper tasks**（前三个）是 reformulated 成 open-retrieval 设置：丢弃原 gold evidence，要求 system 自己 retrieve。

**Multi-paper tasks** 是核心新贡献，由 PhD-level experts 写。每个 Scholar-Multi answer 平均花 56 分钟，付费 $30-45/小时。SCholar-CS 还额外有 rubric annotation（每题平均 4.4 个 must-have/nice-to-have ingredients，每个 ingredient 配 4.4 个 supporting quotes）。

### 3.2 评估指标

**Correctness**：
- Single-paper：accuracy（SciFact/PubMedQA）或 ROUGE-L（QASA）
- Multi-paper Scholar-CS：**Rubric score** = 60% × annotation-driven criteria + 40% × general criteria（length, domain expertise, citation quality, supporting excerpts）。GPT-4o Turbo 当 judge。
- Annotator 间 agreement：human-human 0.62-0.80，human-LLM judge 0.79-0.81

**Citation accuracy**（最重要指标之一）：
- Citation recall：每个 citation-worthy statement 是否有 citation + citation 是否支持
- Citation precision：每个 citation 是否 relevant + 是否 necessary（移除后其他 citation 是否仍 hold）
- 计算 **Citation F1**

**Content quality on Scholar-Multi**：
- Relevance（相关性）
- Coverage（广度+深度）
- Organization（写作流畅度）
- 用 Prometheus v2 当 LLM judge
- Human 评估还加 usefulness 1-5 分

相关链接：
- Prometheus v2 (Kim et al. 2024): https://arxiv.org/abs/2405.01535
- Evaluating verifiability (Liu et al. 2023): https://aclanthology.org/2023.findings-emnlp.468/
- Hurdles to long-form QA (Krishna et al. 2021): https://aclanthology.org/2021.naacl-main.391/

## 4. 关键实验结果

### 4.1 主表（Table 1）核心数字

**Scholar-CS Rubric score**（primary correctness on multi-paper）：
- Llama 3.1 8B (no RAG): 41.9
- Llama 3.1 8B + RAG_OSDS: 46.7
- **OpenScholar-8B: 51.1**
- Llama 3.1 70B: 44.9
- Llama 3.1 70B + RAG_OSDS: 48.5
- OpenScholar-70B: 52.5
- GPT-4o: 45.0
- GPT-4o + RAG_OSDS: 52.4
- **OpenScholar-GPT-4o: 57.7** (+12.7 over GPT-4o)
- PaperQA2: 45.6
- Perplexity Pro: 40.0

**Citation F1 on Scholar-CS**：
- GPT-4o (no RAG): 0.1（基本全 hallucinated）
- Llama 3.1 8B (no RAG): 0
- OpenScholar-8B: 47.9
- OpenScholar-GPT-4o: 39.5
- PaperQA2: 48.0（最高，因为 PaperQA2 反应保守，引用少但准确）

注意 PaperQA2 citation F1 高但 rubric score 低——因为 PaperQA2 倾向引用 1-2 篇 paper 然后逐段总结，coverage 不足。这反映了 **precision-recall trade-off** 在 literature synthesis 中的张力。

### 4.2 Hallucination 统计（Table 2）

CS / Biomedicine fabricated citation ratio：
- Llama 3.1 8B: 92.1% / 97.6%
- Llama 3.1 70B: 78.1% / 96.6%
- GPT-4o: 78.7% / 94.8%
- **OpenScholar-8B: 0% / 0%**

GPT-5（2025 年 8 月发布）复测把 CS hallucination 降到 39%，但仍是问题。

### 4.3 Human evaluation（Table 4）

16 位 PhD-level experts 在 108 个 Scholar-Multi 问题上做 pairwise + fine-grained evaluation：

vs human expert answers：
- GPT-4o (no RAG): win 31.9%, tie 13.8%, lose 54.2% → usefulness 69.7%
- **OpenScholar-8B: win 50.8%, tie 12.3%, lose 36.9% → usefulness 72.1%**
- **OpenScholar-GPT-4o: win 70.0%, tie 6.8%, lose 23.2% → usefulness 80.0%**

**关键 insight**：OpenScholar 优势主要在 **coverage**（+0.7 / +0.9 分），即广度和深度；GPT-4o 没有 retrieval 时 organization 高（4.63）但 coverage 低（4.06），输在覆盖面。

**Length control 实验**：把 OpenScholar-GPT-4o 输出压缩到 ~333 词（接近 human answer 长度），重新评估仍然 win/tie 75%。这证明 OpenScholar 的优势不纯粹来自长度。

专家解释分析：59 个 pairwise 解释里，organization/relevance/coverage/citation 被提及比例是 12%/23%/29%/9%。Coverage 是最主要的决策因素。Annotators 也指出 model citations 有时 outdated 或不够 representative。

### 4.4 Ablation（Extended Data Table 2）

OpenScholar-8B 在 Scholar-CS 上：
- Full: Rubric 51.1, Cite 47.9
- -reranking: 49.4, 42.3（reranker 贡献显著）
- -feedback: 49.6, 28.2（citation 大幅下降）
- -attribution: 50.5, 41.4（citation verification 关键）
- -training（换 vanilla Llama 3.1 8B）: 49.3, 44.0

OpenScholar-GPT-4o：
- Full: 57.7, 39.5
- -reranking: 52.4, 22.9
- -feedback: 55.1, ?

观察：
- Reranking 是单 component 移除后影响最大的（尤其 citation F1）
- Feedback 对 GPT-4o 影响大于对 8B，因为 8B 训练时已经内化了 feedback pattern
- Citation verification 移除后 citation F1 大跌，说明 post-hoc insertion 真的有效

### 4.5 Context length（Top N）效应

Extended Data Fig 3-4 显示 top N 从 5 → 10 → 20 的影响：
- Llama 3.1 8B + RAG：N=5→10 提升，N>10 后开始 degrade（虽然 Llama 3.1 支持 128K context，但有效利用能力差）
- OpenScholar-8B：直到 N=20 仍保持 robust
- Llama 3.1 70B：对长 context 更 robust

这印证了 **Lost in the middle**（Liu et al. 2024）现象，但也表明专门训练可以让小模型更好利用长 context。

相关链接：
- Lost in the middle: https://aclanthology.org/2024.tacl-1.9/
- RECOMP (Xu et al. 2024): https://arxiv.org/abs/2310.04408

## 5. 与相关工作的对比

### 5.1 vs PaperQA2 (Skarlinski et al. 2024)

- PaperQA2 基于 GPT-4o，agent-based，多轮检索
- Citation F1 高（48.0）但 coverage 低，因为策略保守（每次只综合 1-2 篇 paper snippets）
- 成本 $0.3-2.3/question，OpenScholar-8B 仅 $0.003/question
- PaperQA2 data store 不开源，OpenScholar 全开源

### 5.2 vs Self-RAG (Asai et al. 2024)

- Self-RAG 用 reflection tokens（retrieve/no-retrieve, relevant/irrelevant, supported/not-supported）
- OpenScholar 用 free-form natural language feedback，覆盖 organization/completeness/content gap 等多维
- Self-RAG 训练时需要 reference tokens，OpenScholar 用 synthetic trajectory 蒸馏

### 5.3 vs KIWI (Xu et al. 2024)

- KIWI: 200 NLP 问题 + LLM-generated 然后人工编辑的答案
- ScholarQABench: 多领域、纯人工答案 + 自动 eval pipeline + rubric-based correctness
- KIWI 只做 human eval，ScholarQABench 提供 automatic + human

相关链接：
- KIWI: https://aclanthology.org/2024.findings-acl.779/
- ExpertQA (Malaviya et al. 2024): https://aclanthology.org/2024.naacl-long.169/
- AutoSurvey (Wang et al. 2024): https://arxiv.org/abs/2410.23251

## 6. Intuition Building：为什么 8B 能打 GPT-4o？

这是这篇 paper 最 counterintuitive 的结果。我推测几个因素叠加：

**(1) Specialized retrieval vs. parametric knowledge**
GPT-4o 即使有 1.8T params 也无法记住所有 scientific paper 的 details。OpenScholar 把 knowledge 外置到 OSDS，让 LM 只负责 synthesis + attribution。这相当于 GPT-4o 的能力被"锁"在 pre-training 截止日期，OpenScholar 把 knowledge 时空延展。

**(2) Domain-specialized retriever/reranker**
Off-the-shelf Contriever 在 scientific domain out-of-domain 性能差（BEIR 上表现也不均衡）。Continual pre-training on peS2o 让 $\theta_{\text{bi}}$ 学到 scientific language distribution。Reranker 用 synthetic relevance labels 专门 fine-tune，比通用 BGE 更精准。这等于把"找对 papers"这个 sub-problem 单独解决。

**(3) Self-feedback loop 内化到 training**
关键设计：用 inference-time feedback pipeline 生成训练 trajectory，让 8B 模型在 single-pass 时就产生类似 multi-step refinement 的效果。这是 **inference-time compute → training-time knowledge** 的 distillation。Ablation 显示 OpenScholar-8B 去掉 feedback 只掉 1.5 分（51.1→49.6），而 OpenScholar-GPT-4o 去掉 feedback 掉 5.3 分（57.7→52.4），证明 8B 已经把 feedback pattern 内化为 parametric 能力。

**(4) Citation-aware training**
大多数 LM 训练时没有显式的 citation 信号。OpenScholar 的训练数据里 answer 都带 in-line citations，并要求支持性。这让 8B 模型学到"哪些 sentence 需要引用 + 如何引用"。

**(5) Iterative refinement 帮 GPT-4o 更多**
有意思的是 OpenScholar pipeline 让 GPT-4o 提升更多（+12.7）vs Llama 8B 提升（+9.2 over RAG baseline）。这暗示 GPT-4o 在 instruction-following + writing organization 上更强，self-feedback loop 能让它把优势发挥出来。

## 7. Limitations（作者自陈）

1. **ScholarQABench**：human annotation 量小（CS 110 / Multi 108），annotator 偏差；automatic eval 可被 length exploit；静态 benchmark 有 contamination 风险；domain 覆盖不全（无 social sciences）。
2. **OpenScholar**：retrieval 不总能找到 most representative papers；8B 模型 instruction-following 弱；GPT-4o 后端依赖 API 不易复现；不用 license-protected papers。
3. **Human eval**：annotators 可能没仔细查 citation precision/recall；annotator 专长不一定覆盖深 domain knowledge。

## 8. 我对这篇工作的几点 intuition

**(A) Retrieval-augmented systems 是 "weak LM + strong retrieval" 的胜利**
8B Llama 没有任何 magical reasoning 能力，但配上 236M passage 的 datastore + 专门 retriever/reranker，就能打 GPT-4o。这说明对于 knowledge-intensive task，知识存储格式比参数量重要。

**(B) Self-feedback 是 inference-time compute 的一种**
类似 OpenAI o1 的 reasoning，但用自然语言反馈而非 latent thinking。这种"显式反思"的好处是可解释、可调试、可 distill 到单 pass。

**(C) Synthetic data pipeline 形成闭环**
inference pipeline → 生成 trajectory → filter → 训练 → 更好的 inference pipeline。这是 self-improving 的雏形。如果加上 RL 或 online learning，可能形成 stronger loop。

**(D) Citation accuracy 是 LLM 进入 scientific writing 的门槛**
没有一个 academic 会接受 78% fabricated citation 的工具。OpenScholar 把 citation accuracy 提到 0% hallucination 是关键 enabler。这也意味着未来 academic tooling 的核心 metric 不是 BLEU/ROUGE，而是 citation F1 + claim support rate。

**(E) Long-context 不等于 long-effective-context**
Extended Data Fig 3 显示 Llama 3.1 8B 虽然支持 128K，但 N>10 就 degrade。OpenScholar-8B 训练后到 N=20 还 robust。这说明长上下文利用能力可以训练，不是 architecture 给定就免费。也呼应近期 work 说 long-context 的 attention pattern 需要 specific training。

**(F) 与 OpenAI Deep Research / Perplexity 的关系**
OpenScholar-GPT-4o 用 0.05 USD/question，而 Perplexity Pro 是 subscription-based $20/月 capped at 9000 queries = $0.002/query（但 citation 不可提取）。Perplexity 更便宜但封闭。OpenScholar 是开源替代，且 demo 已经 30K users。

## 9. 公共资源

- Code: https://github.com/AkariAsai/OpenScholar
- ScholarQABench code: https://github.com/AkariAsai/ScholarQABench
- Training data: https://huggingface.co/datasets/OpenSciLM/OS_Train_Data
- Demo queries: https://huggingface.co/datasets/allenai/openscilm_queries
- Public demo: https://openscholar.allen.ai
- Expert eval interface: https://github.com/AkariAsai/OpenScholar_ExpertEval
- Paper (Nature): https://doi.org/10.1038/s41586-025-10072-4
- arXiv preprint 应该有，建议搜 "OpenScholar Asai"

## 10. 我会推荐的 follow-up 方向

1. **Citation network prior**：现在 retrieval 主要是 semantic，加上 paper-paper citation graph 可能找到更 representative papers
2. **Multi-modal papers**：figures/tables/equations 没进 datastore，损失信息
3. **Long-horizon synthesis**：现在 single-turn，多轮追问 + 实时更新文献的场景
4. **RL from citation feedback**：当前用 LM judge filter 数据，若引入真实 human citation correction 作 RL signal 可能更强
5. **Domain extension**：social science / humanities 的 paper access 不同，需要新 retrieval 策略
6. **Reasoning over retrieved passages**：当前 LM 直接生成，若加 Chain-of-Thought between passages 可能提升 synthesis 质量

---

整体来说，这篇 paper 是 "specialized LM + specialized retrieval + iterative refinement" 范式在 scientific literature synthesis 上的胜利。最 striking 的是把 hallucination 从 78-98% 砍到 0%，并且 8B 小模型打败 GPT-4o——这给社区一个强信号：对于 knowledge-intensive 任务，把 knowledge 外置到 retrieval system 比堆积参数更有效，且通过 synthetic trajectory distillation 可以把 inference-time loop 内化为 parametric 能力。
