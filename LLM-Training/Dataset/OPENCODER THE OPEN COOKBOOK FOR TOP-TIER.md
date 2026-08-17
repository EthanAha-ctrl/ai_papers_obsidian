---
source_pdf: OPENCODER THE OPEN COOKBOOK FOR TOP-TIER.pdf
paper_sha256: a69c5378cebff4e299ddc9666a9a1199c581cd7477b814b3fbe9946402561ea9
processed_at: '2026-08-06T00:22:34-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenCoder 人话版

好，咱们坐下来聊聊天，把这篇论文掰开了揉碎了说。

---

## 这篇论文到底干了啥

一句话：**训了个还不错的 code LLM，但重点不是模型本身，是把整套配方公开了。**

你看现在 code LLM 这个圈子，Qwen2.5-Coder 性能炸裂但不告诉你数据怎么来的，DeepSeek-Coder 也不说，CodeLlama 更早的事了。你拿不到 training data，拿不到 data pipeline，拿不到中间 checkpoint。你想做研究？对不起，只能拿个 final weights 黑盒玩。

OpenCoder 的态度是：我把锅底都掀开给你看。数据怎么洗的、规则怎么定的、哪些 ablation 试了什么结论、中间 checkpoint 长啥样——全给你。性能不一定是最 top 的，但你想要复现、想要改进，所有材料都齐了。

---

## 数据这块是最核心的

### 他们造了个数据集叫 RefineCode，960B tokens

这个数字跟 The Stack v2 差不多（900B），但内容质量差很多。怎么做到的？靠一套很细致的清洗 pipeline。

### 清洗 pipeline 五步走

**第一步：预处理。** 大于 8MB 的文件扔掉（基本是二进制），按文件后缀过滤出 607 种 programming language。

**第二步：去重。** 这是论文里最有意思的 ablation。

GitHub 上代码重复到什么程度？75% 的文件是完全一模一样的。fork、copy-paste、模板代码，到处都是。

他们做了两层去重：
- Exact dedup：算 SHA256 hash，一样就是一样
- Fuzzy dedup：MinHash + LSH，差不多一样的也去掉

关键来了——**去重粒度选 file-level 还是 repository-level？**

DeepSeek-Coder 选的是 repository-level，意思是同一个 repo 里的文件互相不去重。OpenCoder 试了两种，发现 **file-level 完胜**。

为啥？你想想，repo-level 保留了 7.5% 的数据，file-level 只保留 2.4%。差了三倍。而且对 repo-level 的结果再做 file-level 去重，还能干掉 68% 的数据。说明 repo-level 留下了海量的、跨 repo 的重复文件。

重复数据多了会怎样？模型会 overfit 这些重复内容，training loss 看着挺低，实际上是在死记硬背，泛化能力反而差。

Figure 8 的曲线很直观：file-level 在 HumanEval 和 MBPP 上一路领先 repository-level。

所以结论很简单粗暴：**file-level 去重就完事了，别花心思搞 chunk-level，那个更没用。**

### 第三步：Transformation

有些问题不值得直接删文件，改一下就好。

比如 15% 的代码文件开头有 copyright notice，"Copyright Intel Corporation (C) 2014-2016" 之类的。这种东西重复率极高，跟学编程一点关系没有，模型看了几万遍容易形成 bad habit。直接正则删掉。

还有 PII（个人隐私信息），password、email、IP 之类的，用 `<password>`、`<email>` 这种 placeholder 替换。

### 第四步：Filtering Rules，这是真功夫

他们写了 130 多条规则，分三层：

**第一层：通用文本规则。** 文件多大、多少行，这种所有文本都适用的。

**第二层：通用代码规则。** 比如：
- 长字符串占比超过 20% 的文件，大概率是 base64、hash 之类的垃圾，删
- 十六进制字符超过 40% 的，不是代码，删
- "TODO"、"FIXME" 占比超过 1% 的，模型容易学到光输出 placeholder 不写真代码，删
- "assert" 占比超过 40% 的，基本是 test 文件，代码模式简单重复，删

**第三层：语言特定规则。** 这个是 OpenCoder 的独创。比如 Python：
- function 数量占行数比例超过 20% 的，说明函数都特别短小简单，代码逻辑稀疏，删
- 解析不成 AST 的，语法都是错的，删
- import 语句占比超过 30% 的，没啥实质逻辑，删

C 语言看 "goto" 出现频率，Java 看特定的 boilerplate 模式，每种语言都有自己的 bad pattern。

**规则怎么调阈值？** Appendix A.1 说了个四步法，最后一步挺巧妙的——用强 LLM 算 perplexity（PPL），看 PPL 最高和最低的样本。PPL 太低说明数据太简单没营养，太高说明可能是乱码没规律，两头都该删。

Figure 3 那个 PCA 可视化很直观：The Stack v2 的点散得到处都是，RefineCode 紧凑很多。outliers 就是那些纯注释、纯十六进制、过短代码之类的垃圾。

### 第五步：Data Sampling

Java 449GB 砍到 200GB，HTML 474GB 砍到 64GB。为啥？Java 太多了，HTML 大量是结构标记没什么逻辑。适当 downsample 让分布更平衡。

---

## Code-Related Web Data：从垃圾堆里刨代码知识

这个思路来自 DeepSeekMath。GitHub 上的代码是主力，但互联网上还有大量代码相关的文本——StackOverflow 问答、博客教程、文档——这些对模型理解代码很有价值，但混在 Common Crawl 的海量垃圾里。

怎么捞？四步：

1. 先人工标 50 万条高质量 code-like 数据当种子
2. 用这些种子训个 FastText 分类器
3. 用分类器在 Common Crawl 上召回
4. 按 URL domain 分析，比如 stackoverflow.com/questions 整个 domain 都是代码相关的，手工标注这些 URL，把分类器漏掉的但 URL 匹配的样本也捞回来
5. 迭代三轮，种子越来越多越来越准

最后从 CC、FineWeb、SkyPile 总共捞了 330GB，加上 GitHub text files 里又分类出来 178GB。

Appendix C.1 还专门标了中文的 code-like domain，比如 `%cloud.tencent.com/developer/article%`、`%juejin.cn/post%`、`%www.cnblogs.com%` 这些。这对中文 code LLM 来说是个很有价值的 by-product。

---

## Annealing 阶段：给模型开小灶

Pretraining 跑完了，直接上 SFT 吗？不，中间还有个 annealing 阶段。

这个概念来自 MiniCPM 的 WSD 学习率策略：warmup → stable → decay。decay 阶段学习率指数下降，这时候喂高质量数据，模型会把这些知识"焊"进去。

Annealing 数据配比：
- 84% 是原始分布的 RefineCode（防止 catastrophic forgetting）
- 16% 是高质量数据：Algorithmic Corpus + Synthetic Data

Algorithmic Corpus 是什么？就是从 pretraining data 里挑出包含 "leetcode"、"def solution"、"class solution" 这些关键词的文件。这类代码逻辑性强、外部依赖少、自包含，跟真实交互场景的任务很像。

Synthetic Data 两种：
1. **High Quality Code Snippets**：用强 LLM 生成独立函数 + test cases，跑通过了的留下。跟 phi-1 的 CodeExercises 思路一样。
2. **Code Textbooks**：用 Qwen2-72B 对代码做交互式分析，把抽象的代码知识提取出来讲解。让模型不只学"怎么写"，还学"为什么这么写"。

Ablation 证明这 16% 的高质量数据作用巨大，Figure 9 里去掉之后性能明显掉。

**Intuition：pretraining 阶段是大量泛泛而读，annealing 阶段是考前突击重点。重点不需要多，但得精。**

---

## 模型架构

简单说：
- 1.5B：24 层，hidden 2240，MHA，14 heads，context 4096
- 8B：32 层，hidden 4096，GQA（32 heads / 8 KV heads），context 8192，基本是 Llama-3.1-8B 的架构

RoPE base 在 8B 上用 500000 而不是 10000，这是为了长上下文外推。θ 越大，高频分量衰减越慢，远距离 token 之间的 attention 信号保留得越好。

Vocab 96640，支持中英文 + 607 种编程语言。

---

## SFT：两阶段，顺序很重要

Stage 1：大而全，4M 条
- RealUser-Instruct 0.7M（从 WildChat、Code-290k-ShareGPT 提取真实用户对话）
- Large-scale Diverse-Instruct 2.3M（合成，T=1.0 保证多样性）
- Filtered Infinity-Instruct 1.0M

这个阶段目标是让模型啥都能聊，broad capabilities。

Stage 2：小而精，367K 条
- McEval-Instruct 36K
- Evol-Instruct 111K
- Educational-Instruct 110K（高质量 seed → teacher 生成 → test 验证）
- Package-Instruct 110K（解决过时 package API 的问题，用 PyDoc 拿最新文档合成）

这个阶段专注 code-specific 的精修。

**为什么要两阶段？** Table 10 的 ablation 一目了然：

- 只用 Stage 1：HumanEval 52.4
- Stage 1 + Stage 2：HumanEval 70.1（大幅提升）
- 两个 stage 混一起打乱训：HumanEval 55.5（比只用 Stage 1 好一点点，但远不如两阶段顺序训）

**Intuition：先让模型见世面，再让它专精。顺序很重要，混在一起反而互相干扰。**

这个跟 curriculum learning 的经典结论一致——简单到复杂、宽泛到专精的顺序比一股脑全塞给模型好。

还有个细节值得说：**Package-Instruct** 解决的是个很实际的问题。你 pretraining data 是 2023 年 11 月之前的，但 NumPy、pandas 这些库一直在更新。模型容易学到老版本 API，实际用的时候调不对。所以他们用 PyDoc 拉最新 API 文档，让 teacher model 生成反映当前用法的 QA pair。这对 tool calling 场景特别重要。

---

## 性能怎么样

Base model 上 OpenCoder 很能打：
- 1.5B：HumanEval 54.3，碾压同级（Qwen2.5-Coder-1.5B 是 43.9）
- 8B：HumanEval 66.5，也领先同级

Instruct model 上：
- 8B：HumanEval 83.5，LiveCodeBench 23.2
- 对比 Qwen2.5-Coder-7B-Instruct 的 88.4 / 37.6，确实差一截
- 但 Qwen 用了 23.5T tokens，OpenCoder 只用了 2.5T

**用不到别人九分之一的 training tokens，达到接近的性能，这个效率是很高的。** 这也侧面验证了 RefineCode 的数据质量确实好。

---

## 几个值得琢磨的 ablation

### GitHub Stars 能当过滤信号吗？

SantaCoder 试过，结论是不行。OpenCoder 也试了，确认不行。

用 stars≥5 过滤后，training loss 更低（更容易拟合），但下游性能更差。Figure 11 的 embedding 可视化说得很清楚：过滤后的数据分布窄了很多，diversity 损失严重。

**Intuition：star 多的 repo 确实代码质量高，但都是同一类"好"代码。模型见少了，泛化就差。对 pretraining 来说，diversity 比 quality 更重要（在某个 quality floor 之上）。**

### Chunk-level dedup 有用吗？

Appendix B 说没用。chunk 就是把数据拼起来切成 4096-token 的块再做去重。试了各种组合，chunk-level 单独去重还不如 repo-level，在 file-level 之上加 chunk-level 只多去了 0.04B，基本没用。

**结论：file-level dedup 就是最优解，别折腾别的了。**

---

## 对你的研究有啥用

如果你要做 code LLM 相关研究，这篇论文的价值在于：

1. **一个可靠的 baseline**：拿到 RefineCode 和 OpenCoder checkpoint，你可以做各种实验而不用从头造数据
2. **Data engineering 的 best practice**：130 条 filtering rules、file-level dedup、PPL-based quality check，这些都可以直接拿来用或改进
3. **Ablation 的 template**：dedup level、stars filtering、annealing data、two-stage SFT，每个关键决策都有对照实验
4. **中文 code data 的标注参考**：Appendix C.1 那个 domain 列表对中文 code LLM 开发很实用

---

## 我的 take

这篇论文最让我欣赏的是**工程诚实**。他们不 claim SOTA，而是老老实实说 "Qwen2.5-Coder 比我们强，但他们用了 9 倍的 tokens"。他们把每个决策的 ablation 都做了，包括那些"我们试了但不 work"的实验（比如 chunk-level dedup、stars filtering）也如实报告。这种诚实对研究社区太重要了。

**真正推动一个领域进步的，往往不是那个性能最高的模型，而是那个把过程讲清楚的团队。**

你有了 RefineCode，有了 pipeline，有了 ablation，你就可以站在他们的肩膀上，把资源用在真正新的探索上——更好的 annealing 策略、更聪明的 synthetic data 生成、repository-level understanding、long context code reasoning……

这才是 open science 该有的样子。

---

# OpenCoder 论文深度解读

## 1. 总体定位

OpenCoder 是一个**完全开源**的 code LLM，最大的贡献在于提供了一份"open cookbook"——不仅仅是 model weights，还包括完整的 data processing pipeline、reproducible pretraining dataset（RefineCode, 960B tokens）、大规模 SFT corpus、intermediate checkpoints、以及详尽的 ablation experiments。

核心信息：
- 论文：https://arxiv.org/abs/2411.04905
- 主页：https://opencoder-llm.github.io
- 两个尺寸：OpenCoder-1.5B 和 OpenCoder-8B
- 在 HumanEval 上 1.5B 模型达到 72.5（Instruct 版），8B 模型达到 83.5

与同期 Qwen2.5-Coder-7B（HumanEval 88.4）相比，OpenCoder 性能略低，但其最大的差异化价值在于**完全可复现**。Table 1 中清晰展示了对比：

| Models | Data Pipeline | Reproducible Dataset | SFT Dataset (>1M) | Intermediate Checkpoints | Training Tokens | HumanEval |
|--------|--------------|---------------------|--------------------|--------------------------|-----------------|-----------|
| OpenCoder-8B | ✓ | ✓ | ✓ | ✓ | 2.5T | 83.5 |
| StarCoder2-15B | ✓ | ✓ | ✗ | ✗ | 4.1T | 72.6 |
| Qwen2.5-Coder-7B | ✗ | ✗ | ✗ | ✗ | 23.5T | 88.4 |

---

## 2. RefineCode 数据集：核心技术贡献

### 2.1 数据组成

RefineCode 总共 960B tokens，分为两大部分（Table 2）：

**Raw Code Data（92%）**：
- Github Code: 755B tokens (78.4%)
- Jupyter Notebooks: 11B tokens (1.1%)
- The Stack v2: 120B tokens (12.5%)

**Code-Related Web Data（7.4%）**：
- Processed CC: 13B tokens (1.4%)
- Processed SkyPile: 3B tokens (0.3%)
- Processed FineWeb: 55B tokens (5.7%)
- AutoMathText: 3B tokens (0.3%)

这个组成中，最值得注意的部分是**code-related web data 的召回**，这部分借鉴了 DeepSeekMath 的方法论。

### 2.2 Raw Code 处理 Pipeline

整个 pipeline 包含五个模块（Figure 2）：

#### 2.2.1 Preprocessing

- 排除超过 8MB 的文件（绝大多数是 non-text 文件）
- 通过 linguist 工具识别文件类型，限制到 607 种 programming languages
- 过滤掉低容量或低质量的文件类型

#### 2.2.2 Deduplication

这是论文中最有意思的 ablation 之一。GitHub 上代码重复极其严重，约 75% 的文件是完全重复的。

**Exact Deduplication**：
- 计算 SHA256 hash
- 对相同 hash 的文件，保留 star count 最高且 commit time 最新的版本

**Fuzzy Deduplication**：
- 将 raw text 切成 5-gram pieces
- 计算 2048 个 MinHash functions
- 使用 LSH (Locality-Sensitive Hashing)，bands=16, rows=128

MinHash 的核心思想是：两个集合的 MinHash 值相等的概率等于它们的 Jaccard similarity。公式表述：

$$P(\min(h(S_1)) = \min(h(S_2))) = J(S_1, S_2) = \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$$

其中：
- $S_1, S_2$ 是两个文档的 shingle 集合（这里是 5-gram）
- $h$ 是一个随机排列的 hash function
- $J$ 是 Jaccard similarity

LSH 将 2048 个 MinHash 分成 16 个 bands，每个 band 128 个 hash values。如果两个文档在任何一个 band 上完全匹配，就被认为是候选 duplicate。这样可以大幅降低比较次数。

**关键 ablation 结果（Section 6.1）**：

论文对比了 file-level 和 repository-level 两种 deduplication 策略，在 485M 个 Python 文件上实验：

| Deduplication Level | # Total Rows | # Retained Rows | # Retained Tokens |
|---------------------|--------------|-----------------|-------------------|
| File level | 485,817,123 | 30,488,834 | 32.74B (2.4%) |
| Repository level | 11,037,352 | 7,480,488 | 99.47B (7.5%) |

结果发现：
1. File-level deduplication 保留了 2.4% 的数据，repository-level 保留了 7.5%
2. Figure 8 显示 file-level 在 HumanEval 和 MBPP 上**显著优于** repository-level
3. 对 repository-level 结果再做 file-level dedup，还能去掉 68.4% 的数据
4. 52B tokens 在 repository-level 中存在 character-level 完全等价的文件

这个结论与 DeepSeek-Coder 的做法相反（DeepSeek-Coder 用的是 repository-level）。作者的 intuition 是：file-level 更激进地去重，保持了数据多样性，避免了模型过度记忆重复内容。

#### 2.2.3 Transformation

两类 transformation：

**Copyright Removal**：
- 15% 的文件开头包含 copyright notices（如 "Copyright Intel Corporation (C) 2014-2016"）
- 这些内容高度重复且与 coding tasks 无关
- 通过正则识别并移除

**PII Reduction**：
- 识别 passwords, emails, IP addresses 等敏感信息
- 用 `<name>`, `<password>` 等 placeholder 替换

#### 2.2.4 Filtering Rules

这是论文的技术亮点之一。作者提出了**第一个针对不同 programming language 特性的 heuristic filtering framework**，包含 130+ 条规则，分为三类：

**1. Natural Language Filtering Rules**：
- 对所有 text 文件通用
- 基于文件大小、行数等通用指标

**2. General Code Filtering Rules**（Table 11 给出示例）：
- 长字符串占比 > 0.2 → 过滤（缺乏代码逻辑）
- 十六进制字符占比 > 0.4 → 过滤
- "TODO"/"FIXME" 行占比 > 0.01 → 过滤（避免模型输出 placeholder）
- "assert" 语句占比 > 0.4 → 过滤（通常是 test 文件，代码模式简单重复）

**3. Language-Specific Filtering Rules**（Table 12 给出 Python 示例）：
- Python function 数量占总行数比例 > 0.2 → 过滤（函数过于简单）
- 无法解析成 AST → 过滤（语法错误）
- "import" 语句行占比 > 0.3 → 过滤

论文 Appendix A.1 提供了一个四步规则设计方法论：
1. Quality Signals Designing
2. Coarse Threshold Tuning
3. Fine-grained Threshold Tuning（聚焦于只受单一规则影响的数据）
4. Data Quality Inspection（引入 PPL-based 评估，用强 LLM 计算 perplexity，检查 top-N 和 bottom-N 样本）

PPL-based 评估的 intuition：
- PPL 过低 → 数据过于简单，缺乏可学习知识
- PPL 过高 → 数据可能缺乏可学习模式

**Figure 3 的 PCA 可视化**很有说服力：RefineCode 的 embedding 分布更紧凑，The Stack v2 有更多 outliers。这些 outliers 通常是纯文本注释、纯十六进制数据、过短代码等低质量模式。

#### 2.2.5 Data Sampling

- Java: 449GB → 200GB（downsample，因为体积过大）
- HTML: 474GB → 64GB（downsample，因为非信息性结构内容过多）
- 最终 pretraining 阶段约 730B tokens

### 2.3 Code-Related Web Data 召回

这部分借鉴了 DeepSeekMath 的方法论，但做了重要改进。整个 pipeline 包含四个步骤：

**Step 1: FastText Model Training**
- 先用 BPE tokenizer 分词（为了处理中文）
- 用 FastText 训练分类器

**Step 2: Recall from Common Crawl**
- 用 FastText 模型在 CC 上召回 code-related 数据

**Step 3: Code-related Domain Discovery**
- 按 base URL 分 domain
- domain 中超过 10% 网页是 code-related 的，整个 domain 被标记为 code-related
- 例如 `stackoverflow.com/questions` 被识别为 computer technology questions

**Step 4: URL Annotation**
- 人工标注 code-related URLs
- 将 fastText 漏掉但 URL 匹配的样本加入 seed corpus
- 迭代三次，最终得到约 220GB code-related web data

同样对 FineWeb、SkyPile、AutoMathText 的 web 部分应用相同 pipeline，总共得到 330GB。另外训练一个 classifier 从 GitHub text files 中提取 code-related 数据，得到额外 178GB。

**Appendix C.1** 提供了中文 code-like domains 的详细标注，例如：
- `%cloud.tencent.com/developer/article%` → Code
- `%ask.csdn.net/questions%` → Code
- `%juejin.cn/post%` → Code
- `%www.cnblogs.com%` → Code

---

## 3. Annealing Data

Annealing 阶段是 pretraining 和 SFT 之间的桥梁。采用 MiniCPM 的 WSD (Warmup-Stable-Decay) 学习率策略，在 stable 阶段后进行 decay，用高质量数据进一步提升模型能力。

Annealing data 组成（Table 3）：

| Category | Dataset | # Tokens |
|----------|---------|----------|
| Original Data | RefineCode | 83.94B |
| Original Data | Algorithmic Corpus | 12.44B |
| Synthetic Data | High Quality Code Snippet | 2.71B |
| Synthetic Data | Code Textbooks | 0.91B |

**Original Distribution Data（84%）**：
- 保持与 pretraining 相似的分布
- 防止 catastrophic forgetting

**Algorithmic Corpus**：
- 从 pretraining data 中抽取包含 "leetcode"、"def solution"、"class solution" 关键词的文件
- 这类代码 logic 强、外部依赖少、self-containment 好
- 更接近真实交互场景中的小型独立任务

**Synthetic Data 两种形式**：

1. **High Quality Code Snippets**（受 phi-1 的 CodeExercises 启发）：
   - 用 strong LLM 合成 self-contained 独立函数 + test cases
   - 保留通过 test cases 的数据
   - 扩展到多种 programming languages

2. **Code Textbooks**：
   - 基于 hqcode 数据集
   - 用 Qwen2-72B-Instruct 对代码进行交互式分析
   - 提取并阐述抽象代码知识
   - 目标是让模型从多个视角理解代码

**Section 6.2 的 ablation** 验证了 annealing 阶段高质量数据的重要性：移除 Algorithmic Corpus 和 Synthetic Data 后，性能显著下降（Figure 9）。

---

## 4. Model Architecture & Training

### 4.1 架构

Table 4 详述了两个尺寸的架构：

|  | OpenCoder-1.5B | OpenCoder-8B |
|--|----------------|-------------|
| Layers | 24 | 32 |
| Model Dimension | 2240 | 4096 |
| Attention Heads | 14 | 32 |
| Key/Value Heads | 14 | 8 |
| Activation | SwiGLU | SwiGLU |
| Vocab Size | 96640 | 96640 |
| Positional Embedding | RoPE(θ=10000) | RoPE(θ=500000) |
| Context Window | 4096 | 8192 |

**关键观察**：
- 8B 模型用 GQA (Grouped Query Attention)，KV heads=8（vs. attention heads=32），比值 4:1，这是 Llama-3.1-8B 的标准配置
- 1.5B 模型用 MHA（Multi-Head Attention），KV heads=attention heads=14
- 8B 的 RoPE base θ=500000 比 1.5B 的 θ=10000 大得多，支持更长上下文外推

RoPE 的公式回顾：
$$f(q_m, k_n) = \text{Re}[(q_m e^{im\theta}) \cdot (k_n e^{in\theta})^*]$$

其中 $m, n$ 是 token positions，$\theta_i = 10000^{-2i/d}$（标准设置）。更大的 base $\theta$ 使得高频分量衰减更慢，有利于长序列。

### 4.2 Training Details

**1.5B 模型**：
- 2T tokens，4 epochs（因为 data curation 不完整）
- Annealing: 额外 100B tokens
- WSD schedule，warmup 2000 steps / 8B tokens
- Peak LR: 3e-4，stable 后保持 constant，annealing 阶段指数衰减到 1e-5
- Micro-batch: 4, Global batch: 1024
- 硬件：256×H800，109.5 小时，28034 GPU hours
- 框架：Megatron-LM + DDP gradient overlap

**8B 模型**：
- 2.5T tokens，3.5 epochs
- Annealing: 额外 100B tokens
- 同样的 WSD schedule
- Micro-batch: 1, TP=2, seq len=8192, Global batch: 1024
- 硬件：512×H100，187.5 小时，96000 GPU hours
- 前 130k steps 用 seq len=4096, global batch=2048

---

## 5. Post Training

### 5.1 SFT Data Composition

四个来源：

1. **Open-source Training Data**：
   - Evol-Instruct, Infinity-Instruct, McEval
   - WildChat, Code-290k-ShareGPT（提取真实用户 query）
   - RealUser-Instruct 数据集：高多样性，对齐真实世界复杂度
   - 对低质量 response 用 robust LLM 重新生成

2. **Educational Instruction Synthesis**：
   - 用 scorer model 识别高质量 seed code
   - Teacher model 生成 test cases
   - Python interpreter 执行验证
   - 只保留通过 test 的样本

3. **Package-related Instruction Synthesis**：
   - 解决预训练数据中过时 package 版本的问题
   - 通过 PyDoc 获取 up-to-date API signatures
   - Teacher model 生成反映当前用法的 QA pairs
   - 这对 tool calls 性能很重要

4. **Large-scale Diverse Instruction Synthesis**：
   - LLM 清理 irrelevant context，选 useful sentences 作为 seed
   - Task specification module（配置 language, difficulty, task type）
   - Temperature T=1.0 生成 diverse questions
   - Validation module: 自动 code execution + unit testing
   - LLM refine response（添加 comments 和 explanation）

### 5.2 Two-Stage Instruction Tuning

**Stage 1**：broad capabilities
- RealUser-Instruct: 0.7M
- Large-scale Diverse-Instruct: 2.3M
- Filtered Infinity-Instruct: 1.0M
- Total: ~4M examples

**Stage 2**：code-specific refinement
- McEval-Instruct: 36K
- Evol-Instruct: 111K
- Educational-Instruct: 110K
- Package-Instruct: 110K
- Total: ~367K examples

**Training Details**：
- Stage 1: 1 epoch, batch=4096, LR=2e-5, warmup=100, cosine scheduler
- Stage 2: 3 epochs, batch=512, LR=5e-5, warmup=100, cosine scheduler

**Section 6.4 的 ablation**（Table 10）非常关键：

| | HE | HE+ | MBPP | MBPP+ | BigCodeBench | Code Arena |
|--|----|-----|------|-------|--------------|------------|
| Stage1 only | 52.4 | 48.1 | 68.7 | 57.4 | 22.1 | 5.3 |
| Stage1 + Stage2 | 70.1 | 64.0 | 74.6 | 64.8 | 31.5 | 6.9 |
| Mix Training | 55.5 | 51.2 | 52.0 | 58.7 | 23.9 | 3.8 |

Intuition：
- Stage 1 数据多样但平均质量较低 → 获得 broad capabilities
- Stage 2 数据高质量、code-specific → targeted enhancement
- Mix Training（混合打乱）效果最差，说明两阶段的 curriculum learning 策略至关重要
- Code Arena（真实场景）的提升尤其明显：5.3 → 6.9

### 5.3 Decontamination

- 对所有 SFT data 严格 deduplication
- 移除包含 HumanEval、MBPP 等 test set entry points 的数据
- 10-gram deduplication：移除与 test set 有 10-gram 重叠的数据

---

## 6. Experimental Results

### 6.1 Base Models（Table 6）

OpenCoder-1.5B-Base：
- HumanEval: 54.3, HumanEval+: 49.4
- MBPP: 70.6, MBPP+: 58.7
- BigCodeBench: 51.8 / 24.5 / 5.4

对比同级：
- Qwen2.5-Coder-1.5B: HE=43.9
- Yi-Coder-1.5B: HE=41.5
- StarCoder2-3B: HE=31.7

OpenCoder-8B-Base：
- HumanEval: 66.5, HumanEval+: 63.4
- MBPP: 79.9, MBPP+: 70.4

对比同级：
- Qwen2.5-Coder-7B: HE=61.6
- Yi-Coder-9B: HE=53.7
- DeepSeek-Coder-6.7B: HE=47.6

Base model 上 OpenCoder 表现非常强，尤其 1.5B 在 HumanEval 上 54.3 领先明显。

### 6.2 Instruct Models（Table 7）

OpenCoder-8B-Instruct：
- HumanEval: 83.5, HumanEval+: 78.7
- MBPP: 79.1, MBPP+: 69.0
- BigCodeBench: 40.3 / 16.9
- LiveCodeBench: 23.2

对比：
- Qwen2.5-Coder-7B-Instruct: HE=88.4, LiveCodeBench=37.6（更强）
- Yi-Coder-9B-Chat: HE=82.3, LiveCodeBench=23.4（接近）
- DS-Coder-V2-Lite-Instruct (16B): HE=81.1, LiveCodeBench=24.3

### 6.3 MultiPL-E（Table 8）

OpenCoder-8B-Instruct 平均 71.0，在 8 种语言上：
- Python: 83.5, Java: 72.2, C++: 61.5, C#: 75.9
- TS: 78.0, JS: 79.5, PHP: 73.3, Bash: 44.3

Bash 性能明显偏低是普遍现象（所有模型都这样）。

### 6.4 McEval & MdEval

McEval 覆盖 40 种语言，约 2000 samples。MdEval 覆盖 18 种语言的 debugging。Figure 6 和 Figure 7 显示 OpenCoder-8B-Instruct 在这些 benchmark 上优于同等规模的开源模型。

---

## 7. 关键 Ablation Studies 总结

### 7.1 Deduplication Level（Section 6.1）

核心结论：**file-level > repository-level**，而且 chunk-level deduplication 没有额外收益（Appendix B）。

File-level dedup 保留 2.4% 数据，repo-level 保留 7.5%。File-level 在下游任务上显著更好（Figure 8）。对 repo-level 结果再做 file-level dedup，还能去掉 68.4% 数据——这说明 repo-level dedup 留下了大量重复。

### 7.2 Annealing Data Quality（Section 6.2）

移除高质量 annealing 数据（Algorithmic Corpus + Synthetic Data）后性能明显下降。这验证了 annealing 阶段 "quality > quantity" 的原则。

### 7.3 GitHub Stars（Section 6.3）

用 stars≥5 过滤数据，反而**降低了性能**。

Intuition：star filter 提升了数据质量，但损害了多样性。Figure 11 显示：
- Filtered data 的 training loss 更低（更容易拟合）
- 但 embedding 可视化显示数据分布更窄
- 过滤后的数据仍包含大量结构良好的算法代码，说明 stars 作为过滤信号不够好

### 7.4 Two-Stage Instruction Tuning（Section 6.4）

Stage1 → Stage1+Stage2 → Mix Training 的对比清晰展示了两阶段策略的价值。Mix Training 效果最差，说明 curriculum 很重要。

---

## 8. 与 The Stack 系列对比（Appendix D, Table 15）

|  | # Tokens | # Languages | # Web Data Tokens | # Rules | LS Rules |
|--|----------|-------------|-------------------|---------|----------|
| The Stack v1 | 200B | 88 | 0 | ~15 | ✗ |
| The Stack v2 | 900B | 619 | ~30B | ~15 | ✗ |
| RefineCode | 960B | 607 | ~75B | ~130 | ✓ |

RefineCode 的差异化：
- 更多 web data（75B vs. 30B）
- 更多 filtering rules（130 vs. 15）
- Language-specific rules（The Stack 没有这个特性）

---

## 9. Programming Languages（Appendix E）

RefineCode 包含 607 种 languages，分为三类：
- **Code**：470 种（如 Python, Java, C++, Go, Rust...）
- **Data**：115 种（如 JSON, YAML, XML, TOML...）
- **Text**：22 种（如 Markdown, reStructuredText, TeX...）

Excluded 的一些类型：CSV, SVG, STL, Unity3D Asset, PostScript, Public Key 等。

---

## 10. 对研究社区的价值

OpenCoder 的核心价值在于**可复现性**和**透明性**。它提供了：

1. **完整的 data processing pipeline**：任何人可以复现 RefineCode
2. **Reproducible pretraining dataset**：960B tokens
3. **大规模 SFT dataset**（>1M examples）
4. **Intermediate checkpoints**：可以研究训练动态
5. **详尽的 ablation experiments**：为未来研究提供 baseline

五个关键 takeaways：
1. Code-optimized heuristic rules for data cleaning
2. File-level deduplication 优于 repository-level
3. Recall code-related text corpus 重要
4. High-quality synthetic data 在 annealing 阶段关键
5. Two-stage instruction tuning 策略有效

---

## 11. 我的 Intuition 与思考

从 Karpathy 的视角看，这篇论文最有价值的几点：

**1. Data quality 的工程化**：论文展示了如何系统性地设计 filtering rules，130 条规则分三层（natural language / general code / language-specific）。这种工程化思维与 phi 系列的 "textbooks are all you need" 哲学一脉相承，但更加系统化、可复现。

**2. Deduplication 的 ablation 很有说服力**：file-level vs. repository-level 的对比，加上 chunk-level 无收益的结论，给社区提供了一个清晰的 best practice。DeepSeek-Coder 用 repository-level 是一个值得重新审视的选择。

**3. Annealing 阶段的设计**：84% original distribution + 16% high-quality 的混合比例，以及 synthetic data 的两种形式（code snippets + textbooks），提供了一个好的 template。这与 MiniCPM 的 WSD 策略配合得很好。

**4. Two-stage SFT 的 curriculum**：Stage 1 broad → Stage 2 code-specific，Mix Training 效果差说明顺序很重要。这与深度学习中 curriculum learning 的经典结论一致。

**5. Stars 作为过滤信号的反思**：这个 ablation 很有意思。更高的 quality 但更低的 diversity 导致更差性能。这提醒我们，对于 pretraining data，diversity 可能比 quality 更重要（在某个 quality threshold 之上）。

**6. PPL-based data quality evaluation**：Appendix A.1 提到用 strong LLM 计算 perplexity，检查 top-N 和 bottom-N 样本。这是一个简单但实用的 data quality proxy。

**7. Code-related web data recall**：FastText + URL annotation 的迭代 pipeline，以及中文 code-like domains 的手工标注，展示了如何从 noisy web data 中提取 code knowledge。这部分对中文 code LLM 开发尤其有价值。

**一些可以深入思考的问题**：
- File-level deduplication 是否会损失一些有意义的重复模式（如 design patterns 的多次实现）？
- Annealing 阶段的 84/16 比例是否最优？论文自己也承认 "this mixture ratio might not be ideal"
- Two-stage SFT 的 Stage 1 数据量大（4M），Stage 2 数据量小（367K），这个比例是否可推广？
- RefineCode 的 607 种 languages 中，长尾 languages 的数据量是否足够？

---

## Reference Links

- OpenCoder 主页: https://opencoder-llm.github.io
- OpenCoder 论文: https://arxiv.org/abs/2411.04905
- The Stack v2: https://arxiv.org/abs/2402.19173
- DeepSeekMath (web data recall 方法论): https://arxiv.org/abs/2402.03300
- MiniCPM (WSD schedule): https://arxiv.org/abs/2404.06395
- phi-1 "Textbooks are all you need": https://arxiv.org/abs/2306.11644
- StarCoder: https://arxiv.org/abs/2305.06161
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- EvalPlus (HumanEval+/MBPP+): https://arxiv.org/abs/2305.01210
- BigCodeBench: https://arxiv.org/abs/2406.15877
- McEval: https://arxiv.org/abs/2406.07436
- LiveCodeBench: https://livecodebench.github.io
- MultiPL-E: https://arxiv.org/abs/2108.08299
- MinHash 原始论文: https://doi.org/10.1109/SEQUEN.1997.666900
- Deduplicating training data: https://arxiv.org/abs/2107.06499
- Megatron-LM: https://arxiv.org/abs/1909.08053
- RedPajama: https://github.com/togethercomputer/RedPajama-Data
- FineWeb: https://arxiv.org/abs/2406.17557
- CodeBERT: https://arxiv.org/abs/2002.08155
- hqcode (Code Textbooks 数据集): 见论文中 footnote 2
- FastText: https://arxiv.org/abs/1612.03651
- Linguist (GitHub 语言识别): https://github.com/github/linguist
- OpenCodeEval (评测框架): 见论文中 footnote 7
- LLM360 (透明的 LLM): https://arxiv.org/abs/2312.06550
- OLMo: https://aclanthology.org/2024.acl-long.841/
- MAP-Neo: https://arxiv.org/abs/2405.19327
- WildChat: https://arxiv.org/abs/2405.16109

---

希望这个解读能帮您 build 出对 code LLM 训练 pipeline 的 intuition。这篇论文的价值远超模型本身——它是一份完整的、可复现的 code LLM 工程实践指南。
