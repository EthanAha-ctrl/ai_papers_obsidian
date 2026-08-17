---
source_pdf: Tabular Representation.pdf
paper_sha256: 7579c2ebf6f6060bff05d67fe4530417fe1c058ce7ea8b9b2eb0f7168ec7df16
processed_at: '2026-08-12T12:02:37-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

## 这篇 paper 在干嘛

故事的开头其实很接地气。Microsoft 的人在做 Dataverse Copilot(就是用 LLM 帮人处理 Excel/表格的产品),从遥测数据里发现一个尴尬的事实:**21% 的用户 Excel 文件根本没有 header**。

于是他们想到一个问题:之前学术界说"给 LLM 喂表格用 HTML 格式最好",这个结论是在干净、整齐、有 header 的理想数据上得到的。那现实世界这么脏,结论还成立吗?

答案是:**不成立,而且会反转**。

## 他们怎么测的

思路特别简单粗暴,但很优雅。拿一张表格,做三件事:

第一,把表格用 8 种不同的格式写出来(JSON、HTML、Markdown、CSV、pandas 代码等等)。

第二,给表格"搞破坏",用 8 种方式。比如把行打乱、把列打乱、整个表格转置、把 header 名字换成乱码、把 header 变成 col_0 col_1 这种、把好几列合并成一列等等。这些破坏方式都是模拟现实里"脏数据"的样子。

第三,让 LLM 做 7 种"结构理解"的小测验。比如"第 3 行第 2 列是啥"、"哪个 column 里有值 42"、"这个 column 的数据类型是啥"、"把这张表转置一下"、"把列按这个新顺序重排"。

然后就是看每种格式 × 每种破坏 × 每种任务的成绩。

这里有个很妙的设计:这些任务都是 **self-supervised** 的。意思是,你不需要人去标注答案,因为给定一张表,"第 3 行第 2 列是啥"答案直接从表里读就行。这让评估可以大规模自动化。

## 最重要的发现:之前的结论是错的

Sui et al. 之前那篇 paper 说 HTML 最好。这篇 paper 一加 noise,HTML 直接掉到倒数第二。

真正稳健的是两个格式:**DFLoader**(就是 pandas 的 `pd.DataFrame({...})` 代码片段)和 **JSON**。

为啥?这里有个很深的 intuition。

**JSON 每一行都重复一遍 header 名**。比如 `{"row_0": {"Age": 25, "City": "NYC"}}`,每一行都带着 "Age"、"City" 这些 key。这意味着 LLM 处理任何一行时,局部上下文里就有完整的结构信息,不需要回头去表头找。

**DFLoader 是按列定义的**,每个 column 是一个独立的 list。`Age: [25, 30, 22]`, `City: ["NYC", "LA", "SF"]`。同样,处理任何一列时,信息是局部的、自包含的。

而 CSV、HTML 这些格式呢?header 只在第一行出现一次。LLM 处理到第 50 行的时候,要回忆"第 2 列叫啥来着",就得回头扫第一行。对 LLM 这种 attention-based 架构,这种"长距离回溯"其实是很脆弱的。

**所以核心 intuition 是:信息冗余等于 robustness**。JSON 和 DFLoader 看起来"啰嗦",但正是这种啰嗦让 LLM 在任何局部都能自洽地推理。

## Markdown 是最大的意外

Markdown 在所有任务上几乎都是最差的。Column reorder 只有 50 分(基本就是随机猜),transpose 34 分,reconstruction 24 分。

这对 prompt engineer 是个直接警告:**别用 Markdown 给 LLM 喂表格**。

原因很有意思。Markdown 表格靠 `|` 和空格对齐,但没有显式的 row 标签,也没有显式的 column 标签,对齐还很容易因为 whitespace 丢失。LLM 看一个 Markdown 表格,其实是在看一堆"位置上大致对齐的文本",没有任何 anchor 可以让它追踪"这是第几行第几列"。

人类觉得 Markdown 可读性好,但 LLM 的"可读性"和人类完全不是一回事。LLM 需要的是**显式的、重复的结构标记**,不是视觉对齐。

## 最阴险的发现:Transpose 会让 LLM "精神错乱"

8 种 noise 里最破坏性的是 **TransposeTable**(把表格行列互换)。

拿 JSON 举例,转置之后:
- Navigation 任务(给坐标找值)反而变好了,+20 分
- 但 Column Lookup 和 Row Lookup 直接崩了,-65 和 -76 分

为啥会这样?论文里有个非常生动的观察:LLM 生成答案的时候,会**直接忽略你做的转置**,用原来的 header 当 column 名回答。

这背后的机制特别有意思。LLM 对表格的"内部表征"里,header 和 row index 扮演的是**语义角色**,不只是位置。正常表格里,header 在第一行、row index 在第一列,位置和语义角色是对应的。你一转置,位置变了,但 LLM 还是习惯性地把"第一行的东西"当 header —— 即使现在第一行其实是原来的 row index。

这就像你把一个房间的门牌号和家具标签互换,然后问别人"卧室在哪",他还是会走向原来挂"卧室"牌子的那个房间,即使现在那个房间标着"厨房"。

**直觉结论:不要指望 LLM 自己做 transpose,预处理阶段就转好**。

## 另一个反直觉:Sequential 命名比乱码命名更坑人

有两种给 header 改名的 noise:
- **ArbitraryColumnNames**:改成随机字符串 `x7k2, p9m3`
- **SequentialColumnNames**:改成 `col_0, col_1, col_2`

直觉上你会觉得 random 更糟,因为完全没意义。但实验发现,**Sequential 对 column reorder 任务的杀伤力远大于 Arbitrary**。CSV 在 Sequential 命名下 column reorder 掉了 67 分。

为啥?因为 `col_0, col_1, col_2` 给了 LLM 一个**虚假的秩序感**。模型看到 col_0, col_1, col_2...,会下意识地觉得"按这个顺序输出就对了",即使你让它按别的顺序重排,它也容易被这个 sequential pattern 带跑。

而 arbitrary 的乱码名字呢?模型知道这些名字没规律,反而会老老实实按你给的指令去映射。

这就像考试的时候,完全不懂的学生有时候比"半懂不懂"的学生答得好 —— 因为半懂的人会被自己的错误直觉带跑,完全不懂的人只能严格按题目来。

## 对现实的启示

这篇 paper 其实讲了一个更大的故事:**LLM 在干净 benchmark 上的表现,跟在真实脏数据上的表现,可能差得很远**。

之前那篇 paper 说 HTML 最好,是在理想数据上的结论。很多 prompt engineering 的"最佳实践"其实都是这样 —— 在 demo 数据上看着好,一到生产环境就崩。

这篇 paper 给的实战建议其实很明确:

第一,给 LLM 喂表格,优先用 **pandas 代码** 或 **JSON**。这俩是 LLM 训练数据里大量出现的形式,是它的"母语"。

第二,绝对不要用 Markdown 喂表格,无论它看起来多漂亮。

第三,不要让 LLM 做 transpose 这种破坏语义角色的变换,自己预处理掉。

第四,如果你非要改 column 名,改成无意义的乱码,比改成 col_0 col_1 这种"伪有序"的名字更安全。

第五,HTML 不是万能药。它在 column lookup 这种特定任务上确实强(因为 `<th>` 显式标了 column 名),但代价是 verbose 导致 token 预算吃紧,在 row 相关的任务上反而很差。

## 更深一层

你(Karpathy)可能会想:这些现象背后的机制是什么?

我觉得这篇 paper 其实是在用"行为级 probing"反推 LLM 的内部表征。它没做 attention 分析,没做 logit lens,但结论已经很有暗示性了。

比如 transpose 那个发现,强烈暗示 LLM 对表格的内部模型是**基于语义角色的**,而不是基于位置的。这跟 transformer 本身的机制是吻合的 —— attention 是 content-based 的,不是 position-based 的(positional encoding 只是给了一个 weak inductive bias)。

再比如 JSON/DFLoader 的优势,暗示 LLM 处理长上下文时,**局部自包含**的结构远比全局紧凑的结构友好。这跟 "lost in the middle" 现象是同一类问题 —— LLM 对长距离依赖的可靠性远不如对局部依赖的可靠性。

如果让我接着做,我会拿 Llama-3-8B 这种开源模型,在 JSON vs Markdown 格式下做 attention head 可视化,看 NavigationTest 时哪些 head 在 attend row index,哪些在 attend column name,两种格式下 pattern 应该很不一样。这就把这篇 paper 的行为级结论落到 mechanistic 级别了。

参考的话,Sui et al. 那篇前作 https://arxiv.org/abs/2305.13062 是必读的对照组,Koleva et al. 在 NeurIPS 2022 Table Workshop 那篇 attention 分析 https://arxiv.org/abs/2302.07598 是 mechanistic 方向的起点。

---

# Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs — 深度解读

这篇来自 Microsoft 的 paper,核心命题非常直接:**LLM 处理表格时,输入用什么格式(prompt serialization)其实早就有人研究过,但前人的结论("HTML 最好")是在干净数据上得到的;一旦引入现实世界的 noise,结论会反转**。这个反转在工业部署(如 Dataverse Copilot)里其实很关键 —— 论文开头就提到产品遥测显示 21% 的 Excel 文件根本没有 header。

参考链接:
- 论文 arXiv: https://arxiv.org/abs/2311.10573
- 前人对照工作 Sui et al. 2023: https://arxiv.org/abs/2310.13028
- Microsoft Dataverse Copilot 上下文: https://learn.microsoft.com/en-us/microsoft-copilot-studio/
- NeurIPS 2022 Table Representation Workshop: https://table-representation-workshop.github.io/

---

## 1. 形式化框架(先把这个跑通,intuition 就来了一半)

论文 Section 3 的形式化定义其实是全文最优雅的部分。定义如下符号:

- **T**: 一张带 header row 的 flat table(限定单 datatype per column,排除嵌套)
- **F**: 格式集合,如 F = {JSON, HTML, Markdown, CSV, TSV, DFLoader, DataMatrix, HTMLNoSpace}
- **N**: 噪声操作集合,如 N = {ShuffleRows, ShuffleColumns, TransposeTable, ArbitraryColumnNames, SequentialColumnNames, ShuffleColumnNames, SerializeRow, ColumnMerger}
- **Q**: self-supervised 任务集合,Q = {NavigationTest, ColumnLookupTest, RowLookupTest, DataTypeLookupTest, TableReconstructionTest, TableTransposeTest, TableColumnReorderTest}
- **q ∈ Q**: 一个任务函数,接收表格,生成一组 {(t, a)} 的(问题, 答案)对
- **f ∈ F**: format 函数,把表格 → prompt 字符串
- **n ∈ N**: noise 函数,把 T → T′

评估集合构造公式:

$$
\mathcal{B} = \{\, (f(n(T)),\ t,\ a)\ \mid\ q \in Q,\ f \in F,\ n \in N,\ (t,a) \in q(f(n(T)))\,\}
$$

含义拆解:
- $n(T)$:先对原始表格施加噪声,得到扰动表格 $T'$
- $f(n(T))$:把扰动表格序列化为 prompt 字符串
- $q(f(n(T)))$:基于这个 prompt 字符串,自动派生(self-supervised)若干 (问题, 答案) 对 $(t, a)$
- 三元组 $(f(n(T)), t, a)$ 是一条评估样本,模型在 $t$ 上输出 $\hat{a}$,与 ground truth $a$ 比较

这个公式本质是 **笛卡尔积扫描** over (format, noise, task),让每个交叉点都有数据。对比 Sui et al. [13] 的工作:他们只扫 (format, task),少了 $n$ 这一维 —— 这就是这篇 paper 的"附加维度"。

---

## 2. 八种格式 (Figure 1) —— 直觉图

| Format | 本质 | token 效率 | 结构显式度 | LLM 训练分布匹配 |
|---|---|---|---|---|
| **DFLoader** | pandas `pd.DataFrame({...})` 代码 | 中 | 高(每列独立 list) | 极高(GitHub/Stack Overflow 大量) |
| **JSON** | `{"row_idx": {"col": val}}` | 中 | 高(每行重复 header) | 极高(API 数据) |
| **DataMatrix** | 二维 list,`[[h1,h2],[v1,v2]]` | 高 | 中(依赖位置) | 中 |
| **Markdown** | `| a | b |` 加分隔行 | 中 | 低(对齐靠空格) | 高(README 常见) |
| **CSV** | `a,b\n1,2` | 高 | 中 | 极高 |
| **TSV** | 制表符分隔 | 高 | 中 | 中 |
| **HTML** | `<table><tr>...` | 低(verbose) | 高(显式 tag) | 高(网页爬取) |
| **HTMLNoSpace** | HTML 去空白 | 中 | 高 | 中 |

关键直觉:**HTML 因为 verbose,在 token limit 4097 下能塞的行数大约只有其他格式的一半** —— 这直接限制了 in-context examples 的数量,所以即便 HTML 在某项任务上"概念上"清晰,实际表现也受限。论文 Section 5.1 明确指出这点。

---

## 3. 八种噪声操作 (Figure 2) —— 三类意图

### Spatial Invariance(空间不变性)
- **ShuffleRows**: 随机重排行
- **ShuffleColumns**: 随机重排列
- **TransposeTable**: 行列转置

意图:测试 LLM 是否真正理解"表格的语义结构",还是只是依赖位置先验。一个真正理解表格的模型应该对 shuffle 不敏感。

### Headers(表头信息)
- **ArbitraryColumnNames**: header 替换成随机 alphanumeric 串
- **SequentialColumnNames**: header 替换成 `col_0, col_1, ...`
- **ShuffleColumnNames**: 保留原 header 名,但打乱顺序到错误的列上

意图:模拟"用户 Excel 没 header"或"header 被对手故意篡改"的场景。ShuffleColumnNames 特别阴险 —— 它制造"看似合理的 column 名"对应"错误的数据列",测试 LLM 是否真去看数据还是只看 header。

### Semi-structured Content(半结构化内容)
- **SerializeRow**: 把每行变成 `key1:val1, key2:val2` 单字符串,整个表退化成单列
- **ColumnMerger**: 随机合并 2/3/4 个连续列,中间加 `----` 分隔

意图:模拟现实中的"被塞进一个 cell 的复合信息"(电话+区号、地址)。

---

## 4. 七个 self-supervised 任务 (Figure 3) —— 两大类

### Fact-Finding(查表)
| 任务 | 描述 | 成功条件 |
|---|---|---|
| NavigationTest | 给 (row, col),返回 cell value | 精确匹配 |
| ColumnLookupTest | 给 value,返回含此值的 column name | 精确匹配 |
| RowLookupTest | 给 value,返回含此值的 row index | 精确匹配 |
| DataTypeLookupTest | 给 column name,返回 pandas dtype | dtype 匹配 |

### Transformation(整表变换)
| 任务 | 描述 | 评估 |
|---|---|---|
| TableReconstructionTest | 输入 serialize 形式,输出 8 种格式之一 | cell-wise P/R/F1 |
| TableTransposeTest | 输入 table,输出转置 | cell-wise P/R/F1 |
| TableColumnReorderTest | 给定新列序,重排列 | cell-wise P/R/F1 |

为什么用 **cell-wise F1** 而非 exact-match pass@1?因为 transformation 输出整张表,exact-match 太严苛,partial credit 才能区分"模型完全错" vs "错一行"。这是论文 Section 4.1 的方法学选择。

---

## 5. 实验设置关键参数

- **LLM**: GPT3 text-davinci-003(注意这是 2023 年的实验,不是 GPT-4,所以结论在 GPT-4 / Claude / Gemini 上需要重新验证)
- **Temperature**: 0(确定性)
- **Token limit**: 4097
- **每 (table, format, noise, task) 组合**: fact-finding 100 tests × 15 completions;transformation 25 tests × 5 completions
- **数据集**: 7 个 Kaggle 数据集
- **统计检验**: t-test + **Bonferroni correction**($\alpha = 0.01/8$,因为 8 个格式对比)

$$
\alpha_{\text{Bonferroni}} = \frac{0.01}{8} = 0.00125
$$

变量含义:$\alpha$ 是 family-wise 显著性阈值,除以 8 是因为有 8 个多重比较,严格控制 false positive。

---

## 6. RQ1 结果:Format 主效应(Tables 1 & 2)

### Fact-Finding pass@1

```
              ColumnLookup  DataTypeLookup  Navigation  RowLookup  Overall
CSV              64.43          95.00         65.57       78.14    75.78
DFLoader         72.71          95.29         68.29       82.86    79.79  ★ best
DataMatrix       62.57          84.00         56.57       87.43    72.64
JSON             65.00          96.43         71.43       78.86    77.93
Markdown         61.43          85.86         48.71       73.29    67.32  ✗ worst overall
TSV              67.00          94.00         64.43       78.14    75.80
HTML             79.83 ★ best   94.67         58.83       52.33    71.40
HTMLNoSpace      73.00          93.50         62.00       59.50    72.00
```

**关键观察**:
1. **HTML 在 ColumnLookup 单项最高(79.83)** —— 比 DFLoader 高 6.38 pp(stat. sig. p < 0.01/7)。直觉:HTML 的 `<th>` 显式标记 column name,匹配 value 时模型很容易回溯到对应 `<th>`。
2. **HTML 在 RowLookup 最差(52.33)** —— HTML 不显式标 row index,模型要靠数 `<tr>` 定位,极易错位。
3. **JSON 在 Navigation 最高(71.43)** —— 比 CSV 高 5.86 pp(p < 0.01/7)。直觉:JSON 每行有显式 key(行索引)+ 每行内重复 header 名,模型 scan 时局部上下文丰富。
4. **DFLoader Overall 最高(79.79)** —— 因为它在 4 个任务上"没有短板"。DFLoader 本质是 pandas 代码,LLM 训练时见过海量类似 snippet,这种格式是 LLM 的"母语"。
5. **Markdown 全面差** —— Markdown 缺乏显式结构分隔符,对齐靠空格(很容易丢失),没有显式的 row/column 标签。这给 prompt engineer 一个直接结论:**不要用 Markdown 给 LLM 喂表格**。

### Transformation F1

```
              ColReorder  Reconstruction  Transpose  Overall
CSV              95.33        74.33        99.00     89.55
DFLoader         99.33        98.00        98.33    98.55 ★ best
DataMatrix       92.67        90.67         0.00    61.11 ✗ (Transpose 完全失败!)
JSON             99.67        85.00       100.00    94.89
Markdown         50.00        24.33        34.00    36.11 ✗✗ worst
TSV              93.33        92.33        50.00    78.55
HTML             50.00        86.00        83.33    73.11
HTMLNoSpace      83.33        84.00        83.33    83.55
```

**核心反直觉发现**:
- **DataMatrix 在 TableTranspose 是 0.00** —— 整表全错!直觉:DataMatrix 是 `[[row0], [row1]]`,转置后变成 `[[col0], [col1]]`,但格式本身没标记哪是行哪是列,模型彻底迷失。
- **JSON 在 TableTranspose 是 100.00** —— 完美!直觉:JSON 每行是一个 dict,key 是 header,转置 = 把所有行的同一个 key 收集起来,可以 per-line 独立处理。这就是论文说的"local context + repetition"。
- **Markdown 全面崩盘** —— ColReorder 50.00(基本是随机猜测), Reconstruction 24.33, Transpose 34.00。Markdown 缺乏任何 anchor 让模型追踪列的身份。

**Intuition 总结**:Transformation 任务奖励的是"**信息冗余 + 局部可处理性**"。DFLoader(每列独立)和 JSON(每行重复 header)都满足,CSV/TSV/HTML(全局表头一次)不满足 —— 一旦任务需要跨行/跨列匹配,这些"紧凑格式"就要靠模型维持"表的全局 mental model",而 LLM 在长表格上这种能力很差。

---

## 7. RQ2 结果:Noise × Format 交互(Tables 3 & 4)

### 7.1 TransposeTable 是最具破坏性的噪声

**JSON 在 TransposeTable 噪声下的变化**(Table 3):

| Task | Delta | p-value 显著? |
|---|---|---|
| Navigation | **+20.86** ★ | ** |
| ColumnLookup | **-65.00** | ** |
| RowLookup | **-76.29** | ** |
| DataTypeLookup | -33.86 | ** |

**反直觉**:Navigation 居然上升 20.86 pp!但 column/row lookup 几乎崩溃。

直觉解释:转置后,原 header 变成 row index,原 row index 变成 header。模型在 Navigation(给坐标找值)上反而变好,因为转置后 row index 是 header 名(更有信息量,容易定位);但 ColumnLookup/RowLookup 完全乱掉 —— **模型生成时常常"忽略 transposition",用原 header 当 column 名回答**。论文原文:

> "the LLM's generations for these tasks seem to ignore the transposition and often reply with the former headers (now row indices) as column names and vice versa."

这是这篇 paper 最深刻的发现之一:**LLM 对表格的"内在坐标系"依赖语义角色(header vs row index),而非位置**。一旦转置破坏了语义角色和位置的常规对应,模型陷入混乱。

### 7.2 SequentialColumnNames 严重打击 Column Reorder

CSV 在 SequentialColumnNames 噪声下:

| Task | Delta |
|---|---|
| TableColumnReorder | **-67.33** F1 ★ |

直觉:Column Reorder 任务是给一个新列序,让模型重排。原列名 `Age, Income, ...` 有语义,模型可以理解"新顺序第 3 个是 Age";换成 `col_0, col_1, ...` 后,模型必须完全靠 prompt 里给的新顺序映射,但很可能它没正确解析"新顺序"指令,而是按 col_0, col_1, ... 顺序输出(看起来"对"但错位)。

ShuffleColumnNames 和 ArbitraryColumnNames 也降性能,但降幅小很多 —— 说明真正的杀手是"**赋予列一个看似合理的伪语义**"(sequential),而不是"无语义"(arbitrary)或"乱配语义"(shuffle)。Sequential 触发了模型的某种"虚假 complacency"。

### 7.3 SerializeRow 严重影响 DataTypeLookup

JSON 在 SerializeRow 下:DataTypeLookup **-12.43** pp。

直觉:SerializeRow 把每行变成 `col1:val1, col2:val2` 单字符串,整表退化成单列。模型看不到列的结构,只能解析字符串内容,推断 dtype 难度激增。这直接对应现实场景:**用户从 PDF/邮件复制表格,粘贴到一个 cell 里**,模型要先做"逆 serialization"。

### 7.4 ColumnMerger 对 ColumnLookup 有害,对 RowLookup 中性甚至有益

DataMatrix 在 ColumnMerger 下:
- ColumnLookup: **-8.00** pp
- RowLookup: +4.86 pp(不显著)

直觉:合并列后,column 数减少,但每个 cell 是复合字符串,column lookup 需要解析子字符串,难度上升。而 row lookup 是"找 value 在哪行",合并后 row 内仍包含原 value(只是拼接),且 row 数不变,反而 row 总宽度变小,定位更容易。

### 7.5 Transformation 任务在噪声下普遍脆弱(Table 4)

几个亮点:

- **JSON 在 ColumnMerger 下 Transpose: -75.33** —— 从 100.00 跌到 24.67。合并列破坏了"每个 key 对应一个独立列"的结构先验,模型彻底迷失。
- **JSON 在 SerializeRow 下 Transpose: -100.00** —— 完全归零。整表已经是单列,转置任务失去意义。
- **DFLoader 在 TransposeTable 噪声下 Reconstruction: -98.00** —— 几乎归零。DFLoader 是按列定义的代码,转置后输入是"按行定义",模型要重构回按列定义的代码,推理路径完全错位。

---

## 8. 与 Prior Work 的关键对比

Sui et al. [13] 在干净数据上发现 **HTML 最优**。本文加入噪声后:
- HTML 在 Overall fact-finding 仅 71.40(8 个格式中倒数第二)
- HTML 在 RowLookup 52.33(8 个格式中最差)
- HTML 在 Transformation Overall 73.11(中等)

结论:**HTML 的优势是表面现象**,源于干净数据上 token budget 不是瓶颈,且 column lookup 这种任务占了多数权重。一旦加入现实噪声,HTML 的 verbosity + 隐式 row index 缺陷就暴露。

---

## 9. 方法学局限与 Future Work

论文明确承认的局限:
1. 只用 text-davinci-003,**没有跨 LLM 评估**。这是最大局限 —— GPT-4/Claude/Gemini/Llama-3 训练数据分布不同,JSON/Markdown 训练占比可能更高,结论可能完全反转。
2. 只看 self-supervised structural 任务,**没连到 downstream**(NL2SQL, table QA, data wrangling)。
3. Self-supervised 任务是"toy"性质,真实任务更复杂。

可以做的扩展:
- Cross-LLM:同样 8 格式 × 8 噪声,在 GPT-4o, Claude 3.5, Llama-3.1, Mistral 上重复
- 加入更激进的对抗噪声:Unicode 同形字替换、数字 perturbation、隐式空格注入
- 接到下游:WTQA, TabFact, NL2SQL benchmark 评估
- Tokenizer 层面分析:不同格式在 tiktoken/BPE 下的 token 效率,关联到性能

---

## 10. 直觉提炼(Intuition Takeaways)

1. **结构冗余 ≠ 浪费**:JSON 在每行重复 header,DFLoader 在每个 list 独立列定义。看似冗余,实则给 LLM 局部可处理性。**冗余 = robustness**。
2. **Pandas 代码是 LLM 母语**:DFLoader 几乎全程最强。给 LLM 喂表格,直接给 `pd.DataFrame({...})` snippet。这呼应了 PAL(Program-Aided Language models)、Chain-of-Code 等思路 —— 让 LLM 用它最熟的工具。
3. **Markdown 是表格的 worst enemy**:作为人类可读格式很美,作为 LLM 输入格式灾难。原因是 Markdown 的对齐靠 whitespace,且无显式 row/column 标签。
4. **Transpose 是 LLM 的"格式杀手"**:转置破坏语义角色和位置对应,模型直接 "ignore" 转置,用旧角色回答。提示:**预处理优于让 LLM 自己做 transpose**。
5. **Sequential 命名触发 "false complacency"**:比 arbitrary 命名更害 performance。模型在 `col_0, col_1, ...` 下容易陷入"按编号顺序"的虚假答案。
6. **HTML 在 column lookup 单项强,但全局弱**:verbose 导致 token 预算吃紧,trade-off 不划算。

---

## 11. 对你(Karpathy)视角的延伸思考

这篇 paper 的方法论本质上是 **mechanistic interpretability 的简化版本** —— 不去 probe 内部 representation,而是用"输入扰动 + 任务分解"反推 LLM 对表格的内部模型。这与你在 nanoGPT / "Let's build GPT" 系列里强调的"先看 attention pattern,再谈 emergent ability"思路同源。

可以延伸的研究方向:
- 在 Llama-3-8B 上做 **attention head 分析**:JSON 格式下,哪些 head 在做 "row index → row content" 的 routing?哪些 head 在做 "header name → column content" 的 routing?
- 用 **logit lens** 观察 LLM 在 NavigationTest 上的中间状态,看模型是"先解码出 column 名再解码 cell 值"还是"一步到位"。
- 把 8 种格式抽象为 **"信息密度 vs 结构显式度" 二维平面**,量化每个格式在此平面上的位置,然后看性能等高线。我预测 DFLoader 和 JSON 都在"高结构显式度 + 中等密度"区域,Markdown 在"低显式度 + 中等密度"。

参考可以继续深入的链接:
- Attention analysis on tabular LMs (Koleva et al. NeurIPS 2022): https://arxiv.org/abs/2302.07598
- TAPAS (weakly supervised table parsing): https://arxiv.org/abs/2004.02349
- TUTA (tree-based table transformer): https://arxiv.org/abs/2105.14010
- Chain-of-Thought on tables (Wenhu Chen): https://arxiv.org/abs/2210.06710
- Microsoft "Need for Tabular Representation Learning" (Cahoon et al.): https://arxiv.org/abs/2301.04382
- HumanEval / pass@1 (Chen et al.): https://arxiv.org/abs/2107.03374
- Original chain-of-thought (Wei et al.): https://arxiv.org/abs/2201.11903
- TableGPT: https://arxiv.org/abs/2307.08674

---

## 12. 一句话总结

**When you feed tables to an LLM, give it pandas code (DFLoader) or JSON, never Markdown, never expect it to robustly handle transpositions, and remember that "clean benchmark conclusions" can flip completely the moment you inject realistic noise** —— 这就是这篇 paper 给 prompt engineer 的实战指南,也是给 tabular-LM 研究者的方法论警示。
