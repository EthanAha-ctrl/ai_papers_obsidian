---
source_pdf: ReMEmbR Building and Reasoning Over.pdf
paper_sha256: e2489c60d2fdc47817e6840260cf1026fcc34da7ef9132dff663b63e08c35858
processed_at: '2026-08-11T22:42:08-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个画风，用大白话跟你唠唠这篇paper到底在干啥。

---

## 一句话说清楚

机器人出门溜达了好几个小时，回来你问它"哎你刚才在哪看到咖啡了？"或者"你多久前进的那个楼？"——传统方法根本答不上来，这篇paper就是来解决这个问题的。

---

## 问题到底难在哪

你想啊，机器人跑了一下午，摄像头录了好几个G的视频。你现在想问它点事儿，怎么办？

**笨办法1**：把所有视频帧全塞给GPT-4o看。问题是一条20分钟的视频就有几万帧，GPT-4o的context window直接爆炸，更别说跑一下午的了。而且就算塞得进去，latency也是灾难——paper里测了，5.5分钟的视频GPT-4o要处理90秒才能回答一个问题。

**笨办法2**：存成传统地图。但地图只能告诉你"这里有个桌子"，没法告诉你"20分钟前桌上的杯子掉了"这种动态事件。地图是static的，时间是missing的。

**根本矛盾**：history是unbounded的（机器人一直跑，数据一直涨），但LLM的context window是fixed的。你永远不可能把所有history塞进一个model。

---

## ReMEmbR的核心思路

这篇paper的核心insight特别简单，甚至有点"显然"：

> **你问一个问题，根本不需要全部history，只需要其中一小段就够了。**

比如你问"咖啡在哪"，机器人只需要找到"我看到咖啡"那几个moment就行，其他几十分钟的视频全是废话。

所以问题就变成了：**怎么从海量history里精准捞出来那几条相关的memory？**

答案就是——**retrieval**，也就是RAG (Retrieval-Augmented Generation) 在robotics上的变体。

---

## 系统怎么搭的

整个系统分两步，非常clean的design：

### 第一步：Memory Building（边走边记）

机器人跑的时候，每3秒钟把摄像头画面攒成一小段，送进VILA模型（一个video captioning model）让它生成一句文字描述。

比如3秒的视频可能变成："A robot is moving through a corridor, there is a red fire extinguisher on the wall."

然后把这句话用text embedding model变成一个vector，连同机器人当时的GPS坐标和时间戳，一起塞进vector database。

就这么简单。机器人走到哪记到哪，每3秒一条记录，每条记录有三个field：**caption embedding、position、timestamp**。

paper里用了VILA-1.5-13b做captioning，mxbai-embed-large-v1做embedding。部署在robot上时为了省资源换成了quantized VILA-3b。

### 第二步：Querying（问的时候去捞）

你问一个问题，比如"Where can I get some chips?"

这时候LLM（GPT-4o）作为agent开始工作。它不是直接回答，而是先想：我需要去database里查什么？

它有三种查询手段：
- **按文字查**：搜"chips"或者"snack"相关的memory
- **按位置查**：搜某个(x,y,z)坐标附近的memory
- **按时间查**：搜某个时间段内的memory

关键来了——**它不止查一次**。paper允许最多3轮iterative retrieval。

比如第一轮搜"chips"，啥也没搜到。LLM想了想，可能换个词搜"snack"或者"food"或者"vending machine"。第二轮搜到了"food court"的记录。LLM一看，嗯，有线索了，第三轮可能再搜food court附近的position看看有没有更多细节。

这就是Chain-of-Thought在retrieval上的体现——**一步步逼近答案**。

最后LLM把捞到的memory汇总，输出一个结构化JSON：要么是文字答案，要么是(x,y,z)坐标，要么是时间。

---

## 为什么iterative retrieval这么重要

Table II的ablation特别说明问题：

| 设置 | Long视频正确率 |
|------|---------------|
| 3轮检索（完整版） | 0.61 |
| 只检索1轮（退化为普通RAG） | 0.50 |

掉了一个量级。原因很直觉——很多问题是multi-hop的。比如"你最后一次在室内看到的那个人穿的什么颜色衣服？"这个需要先搜"室内"的memory，再看哪些提到"人"，再找时间最晚的那条，最后看衣服颜色。一轮检索根本搞不定。

---

## 数据集NaVQA怎么来的

作者发现没有合适的数据集能eval这种long-horizon robot QA，就自己造了一个。

基于CODa dataset（UT Austin campus的Husky robot在校园里跑的数据），找了5个robotics专家来标注。

210个问题，分三类：
- **Spatial** (34%)："Where did you see X?" → 输出坐标
- **Temporal** (18%)："When did X happen?" / "How long did you do Y?" → 输出时间
- **Descriptive** (48%)："Was the sidewalk busy?" → 输出文字或yes/no

每个问题按视频长度分三档：Short (<2min), Medium (2-7min), Long (>7min)。

评判标准：空间误差<15m算对，时间误差<2min算对，文字题用LLM-as-judge。

---

## 实验结果的核心takeaway

### Table I 的关键发现：

**GPT-4o碾压一众开源模型，尤其在spatial和temporal reasoning上。**

看Long视频的Positional Error：
- GPT-4o: 46m
- Llama3.1-8b: 165m  
- Codestral: 212m
- Command-R: 189m

开源模型在坐标相关的推理上简直惨不忍睹。原因很简单——LLM本质是next-token predictor，你让它做坐标算术（比如两个GPS点之间谁更近），它其实是靠pattern matching硬猜的，根本不是在做数学。小模型更扛不住。

**Temporal Error也是一样：**
- GPT-4o: 3.6s
- Llama3.1-8b: 18.7s

GPT-4o能比较accurate地算出"15分钟前"对应哪个timestamp，小模型经常搞混relative time和absolute time。

**vs Multi-Frame VLM baseline：**
VLM在Short视频上跟ReMEmbR差不多（0.55 vs 0.62），但Medium和Long直接崩了——context window装不下，标了个X。这就是ReMEmbR的核心优势：**querying latency跟视频总长度无关**。20分钟的视频和2小时的视频，检索时间差不多，因为vector DB搜索是$O(\log N)$的。

### Table II 的ablation还揭示：

**Caption frame rate很关键。** 从2 FPS降到0.5 FPS（6帧/12秒 vs 6帧/3秒），Long正确率从0.61掉到0.38。原因直觉上很清楚——低帧率会错过transient events。比如一个人走过门口只持续2秒，你12秒采样一次很可能完美跳过。

**Captioning model大小影响不大。** 13b vs 3b在最终accuracy上差别很小（0.61 vs 0.50），但3b的throughput高很多倍。这意味着memory building阶段的bottleneck不在caption quality，而在retrieval reasoning quality。这也让on-device deployment变得feasible。

---

## Real Robot Deployment

这部分是最impressive的。他们真的把这套东西跑在了Nova Carter robot上。

硬件配置：
- Jetson Orin 32GB（edge device）
- 3D LiDAR + ROS2 Nav2 AMCL做localization
- Whisper ASR做语音输入（on-device）
- Quantized VILA-3b做captioning（on-device）
- GPT-4o通过cloud API做querying

跑了一个25分钟的memory building phase，然后开始问问题。

成功的例子：
- "Where can I get some chips?" → 机器人导航到cafeteria shelf
- "Take me somewhere with a nice view" → 机器人搜"tall glass windows"、"plants"、"open spaces"，然后导航到一个有大玻璃窗和绿植的lobby

失败的例子：
- "Take me to the soda machine" → 机器人导航到了water fountain。原因是VILA-3b quantized版把water fountain caption成了"silver machine"，LLM agent基于错误caption做了合理但错误的retrieval。

**这个failure mode非常值得深思。** 整个pipeline的error来源不是LLM reasoning能力不够，而是前端VLM captioning的semantic error直接污染了memory database。Garbage in, garbage out。后续不管LLM多聪明，都无法修复input-level的representation error。

---

## 给你build intuition的几个角度

### 1. 这本质上是在做"稀疏注意力"

Transformer的self-attention是dense的——每个token要看所有其他token。ReMEmbR本质上是用LLM agent做了一种**indirection-based sparse attention**。

LLM agent决定query什么、query谁，就是在计算implicit的attention weights。然后只对top-k相关的memory进行"value aggregation"（读取并放进context）。complexity从$O(N^2)$降到了$O(k \cdot \log N)$，其中$k$是retrieval rounds数，$\log N$是vector DB的ANN搜索复杂度。

这就是为什么长视频不会让ReMEmbR变慢——你搜20分钟的视频和2小时的视频，vector DB返回top-m的结果速度差不多。

### 2. 这跟人脑的记忆系统很像

Vector database = Hippocampus（快速记录episode：什么、在哪、什么时候）

LLM agent = Prefrontal cortex（根据问题做retrieval cue，在hippocampus里搜索，然后reconstruct memory）

人回忆过去也是这样——你不会把过去一年的经历全部"播放"一遍，而是根据线索（"上次去那个咖啡馆是跟谁一起？"）先定位到某个episode，再调取细节。

ReMEmbR的iterative retrieval也跟人的memory search过程很像：先想起一个模糊的线索，再基于这个线索refine search，逐步逼近答案。

### 3. Text作为memory medium的trade-off

ReMEmbR选择把video转成text caption再存。这有很大的information loss：
- Spatial关系容易丢（left/right/behind）
- Fine-grained attributes容易丢（red vs blue cup）
- 量化后的VILA-3b还会产生hallucination（water fountain → "silver machine"）

但text作为medium也有巨大优势：
- LLM天然能理解和reasoning over text
- Text embedding的retrieval quality远高于image embedding
- Storage cost极低（一句话vs几帧高分辨率图像）

未来的方向可能是hybrid——存text caption做retrieval，但同时存raw image frame做verification。当LLM retrieved一条memory后，可以调取对应的原始图像帧让VLM做二次确认。但这会增加latency和storage cost。

### 4. Memory Consolidation是下一个大问题

Paper的limitations部分提到了一个真问题：vector DB里会积累大量redundant information。如果机器人在同一个走廊来回走了10趟，数据库里就有10条几乎一样的"corridor with fire extinguisher"记录。这些冗余会dilute检索质量。

人脑的解法是memory consolidation——睡眠时把episodic memory（具体事件）压缩成semantic memory（抽象知识）。比如你去过同一个咖啡馆100次，你不会记得每次的细节，但你"知道"那个咖啡馆在哪里、长什么样。

对于robot，可以设计一个background process：定期对vector DB做clustering，把相似的memory merge成一条"summary memory"，附上statistics（first seen, last seen, frequency）。这既能减少存储，又能提高检索质量。这跟vector DB里的 Hierarchical Navigable Small World (HNSW) graph compression 有异曲同工之妙。

### 5. 跟MobilityVLA的本质区别

MobilityVLA [3] 是concurrent work，也是处理long-horizon robot video。但它的approach是把整个video tour直接塞进Gemini的1M context window。

这有两个fundamental问题：
- **Scalability**：1M context够装1小时的video，那10小时呢？100小时呢？History是unbounded的，fixed context window永远不够。
- **Latency**：处理1M token的inference time是分钟级的，robot不可能等这么久。

ReMEmbR的approach本质上是把"存储"和"推理"decouple了。Memory存储在external vector DB里，inference只处理retrieved subset。这使得系统可以scale到arbitrary length的history。

---

## 我的overall评价

这篇paper的technical contribution其实不算breakthrough——RAG + LLM agent + vector DB都是现成技术。但它的价值在于：

1. **Problem formulation好**：把robot long-horizon memory formalize成了一个retrieval problem，给出了clean的数学定义（公式1和2）
2. **System design pragmatic**：两阶段decoupling，edge-compatible，latency可控
3. **NaVQA dataset填补了空白**：之前没有专门eval robot long-horizon QA的benchmark
4. **Real deployment**：不是只在simulator里跑，真的deploy在robot上并展示了qualitative results

Limitations也很明显：text-only memory representation的information loss、vector DB redundancy问题、对小LLM arithmetic能力的依赖。但作为一个system paper，它定义了problem space并给出了reasonable baseline，后续work有很大的改进空间。

---

**References:**
- Project page: https://nvidia-ai-iot.github.io/remembr
- VILA: https://arxiv.org/abs/2312.07533
- OpenEQA: https://open-eqa.github.io/
- MobilityVLA: https://mobility-vla.github.io/
- CODa Dataset: https://arxiv.org/abs/2303.05552
- mxbai-embed-large-v1: https://www.mixedbread.ai/blog/mxbai-embed-large-v1

---

这篇 paper 《ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation》解决了一个非常核心的 robotics 问题：机器人长时间部署后的 memory 构建与查询。对于像你这样对 neural network architectures 和 embodied AI 有深刻直觉的研究者来说，这篇工作的核心 insight 在于将 unbounded 的 video history 转化为 queryable 的 vector database，并且利用 LLM agent 的 function calling 机制进行 iterative 的 spatio-temporal reasoning，从而绕过了 transformer 中的 quadratic context window 限制。

下面我为你进行极度详细的 technical breakdown。

### 1. Core Problem & Motivation

现有的 robot navigation memory 通常局限于 metric maps 或 semantic maps，这些表示是 static 的，无法捕捉 dynamic events (比如 "10分钟前谁打翻了咖啡")。如果用 VLM 处理长视频，inference latency 和 memory 会随着 context length $N$ 呈 $O(N^2)$ 增长。Concurrent work 比如 MobilityVLA 直接把长视频塞进 Gemini 的 1M context window，这种方法 lacks scalability。ReMEmbR 的设计哲学是：对于特定问题 $Q$，完整的 history $H_{1:K}$ 中只有极小的一个 subset $R^*$ 是必要的。

### 2. Mathematical Formulation 深度解析

Paper 中的公式 1 和 2 定义了 retrieval-augmented memory 的数学基础。

**Equation (1):**
$$p(A | H_{1:K}, Q) = p(A | R^*, Q) \approx p(A | R, Q)$$
s.t. $R \sim F(V)$

变量解释：
*   $A$: 模型预测的答案。
*   $H_{1:K}$: 机器人在 $K$ 分钟内部署积累的完整 history (包含 images, positions, timestamps)。
*   $Q$: 用户提出的自然语言问题。
*   $R^*$: 理论上能够完美回答 $Q$ 所需的最小且最优的 history subset。
*   $R$: 实际通过 sampling function $F$ 采样得到的 subset。
*   $V$: 存储在 vector database 中的 memory representation。
*   $F$: 采样策略，在这里指的是 LLM agent 生成的 query function。

**Equation (2):**
$$R^* = \arg\min_R |R|$$
$$s.t. \arg\max_A p(A | R, Q) = \arg\max_{A'} p(A' | H, Q)$$

变量解释：
*   $|R|$: Subset $R$ 的 cardinality (大小)。
*   目标是寻找最小的 $R$，使得基于 $R$ 推导出的最大似然答案与基于完整 history $H$ 推导出的最大似然答案完全一致。这是一种 Information Bottleneck 的思想，强制模型学会丢弃无关信息。

### 3. Architecture 深度解析

ReMEmbR 将系统解耦为两个阶段，这种解耦对于 edge computing 非常友好。

#### Memory Building Phase
这个阶段在 robot 运行时持续异步执行。
1.  **Video Chunking & Captioning**: 机器人每收集 $t$ 秒的 image frames $H_{I_{i:i+t}}$ (paper 中 $t=3$ 秒，采样 6 frames，即 2 FPS)，送入 VILA-1.5-13b video captioning model [47]。VILA 输出一个自然语言 caption $L_{i:i+t}$，这步是极其关键的 neural compression，将高维 pixel space 压缩到 discrete semantic tokens。
2.  **Multi-Modal Embedding & Storage**: 使用 mxbai-embed-large-v1 [48] 将 caption 转化为 embedding $E(L_{I_{i:i+t}})$。连同机器人的 3D spatial position $H_{P_{i:i+t}}$ 和 timestamp $H_{T_{i:i+t}}$ 一起存入 vector database $V$ 中。
3.  **Indexing**: Vector database 同时建立 text index, spatial index (基于 KNN 或 R-tree 变体), 和 temporal index。

#### Querying Phase (LLM Agent Loop)
当用户提问 $Q$ 时，触发 querying phase。
1.  **State Machine LLM**: LLM (如 GPT-4o) 作为 agent，读取当前的 context $R_{0:i}$ 和 question $Q$，生成 function call $f$ 和 query argument $q$：
    $$R_{i:i+m} = f(q), \text{ where } q = LLM(R_{0:i}, Q)$$
2.  **Function Calls 设计**: Paper 定义了三种 retrieval functions:
    *   **Text retrieval** $f_l(\text{object})$: 基于 semantic embedding 检索包含特定 object 的 memory。
    *   **Position retrieval** $f_p((x,y,z))$: 基于 spatial coordinate 检索某位置附近的 memory。
    *   **Time retrieval** $f_t(\text{HH:MM:SS})$: 基于 timestamp 检索特定时间段的 memory。
3.  **Iterative Retrieval**: 这点极其重要。LLM 最多执行 $k=3$ 次 iterations。每次检索回 $m$ 条 memories，LLM 评估这些 context 是否足够回答问题。如果不够，LLM 会根据当前已知线索生成新的 $q$ 进行下一轮检索 (比如第一轮搜了 "coffee cup"，发现位置在走廊，第二轮可能调用 $f_p$ 搜走廊附近的其他 memory)。这是一种 spatial-temporal 的 Chain-of-Thought (CoT) [31]。

### 4. NaVQA Dataset 解析

为了 evaluate 这个系统，作者构建了 NaVQA dataset，基于 CODa dataset [49] (UT campus 的 outdoor Husky robot 数据)。

*   **规模与分布**: 210 questions，源自 7 个 15-30 分钟的 sequences。
*   **长度分类**:
    *   Short: < 2 minutes
    *   Medium: 2-7 minutes
    *   Long: > 7 minutes
*   **Question Types**:
    *   Descriptive (16% + 32% yes/no): 文本输出，如 "Was the sidewalk busy?"
    *   Spatial (34%): 输出 $(x,y,z)$ 坐标，如 "Where did you see my phone?"
    *   Temporal (14% point-in-time + 4% duration): 输出相对时间 "15 minutes ago" 或绝对时间。
*   **Metrics**:
    *   Spatial: L2 distance < 15m 算 correct。
    *   Temporal: L1 distance < 2 minutes 算 correct。
    *   Descriptive: 使用 LLM-as-a-judge [7] 评估 binary accuracy。

### 5. Experiments & Data Tables 技术讲解

#### Table I: ReMEmbR vs Baselines
对比了 ReMEmbR (GPT-4o, Codestral, Command-R, Llama3.1-8b) 与 LLM with full Captions 和 Multi-Frame VLM (GPT-4o)。

*   **Descriptive Question Accuracy**: 在 Long category 上，ReMEmbR (GPT-4o) 达到 0.65，略低于 LLM with full captions (0.55? 这里 paper 描述可能有误，表格显示 0.55，文字说 outperform，可能是笔误或标准差范围内的差异)。但是 VLM 在 Long 上直接崩溃 (标记为 X，因为 context 超限)。
*   **Positional Error**: 在 Long category 上，ReMEmbR (GPT-4o) 误差 46.25m，相比 Codestral (212.4m) 和 Llama3.1-8b (165.3m) 有巨大优势。说明 function calling 能力和 spatial reasoning 能力高度依赖 base model 的 size 和 alignment。
*   **Temporal Error**: 在 Long 上，ReMEmbR (GPT-4o) 仅 3.6 秒误差，而 Llama3.1-8b 达到 18.7 秒。这暴露了 smaller LLM 在 time arithmetic 上的严重缺陷。
*   **Latency**: 21.5 分钟视频，ReMEmbR 约需 25s/question。VLM 处理 5.5 分钟视频就需 90s。因为 ReMEmbR 的 inference complexity 与 total video length $K$ 无关，只与 retrieval iterations $k$ 和 retrieved subset $R$ 大小相关，呈 $O(1)$ w.r.t $K$ (实际上 vector DB 搜索是 $O(\log K)$)。

#### Table II: Ablations
*   **Iterative Retrieval**: 将 iterations 从 3 降到 1 (退化为传统 RAG)，Long category 的 Overall Correctness 从 0.61 掉到 0.50。这证明单次 retrieval 无法捕捉 long-horizon reasoning 所需的 multi-hop context。
*   **Caption Frame Rate**: 使用 12-sec captions (0.5 FPS) 代替 3-sec captions (2 FPS)，Long correctness 从 0.61 暴跌到 0.38。这非常 intuitive，coarse temporal sampling 会导致 transient events (如物体掉落、人经过) 在 neural compression (VILA) 阶段直接丢失。
*   **Captioning Model Size**: 13b 相比 8b 和 3b 只有微弱优势。但在 robot deployment 中，由于要满足 real-time memory building，paper 最终选了 quantized VILA-3b。

### 6. Real-World Deployment on Nova Carter

部署在 Nova Carter robot [54] 上，计算架构非常值得参考：
*   **Edge Device**: Jetson Orin 32GB。
*   **On-device Models**: Whisper ASR [55], Quantized VILA-3b, ROS2 Nav2 stack with AMCL, 3D LiDAR, Vector DB。全部在本地运行，保证 privacy 和 low latency。
*   **Cloud/API**: GPT-4o 作为 LLM backend (可通过 OpenAI API 或 NVIDIA NIM APIs)。
*   **Failure Mode**: 作者提到一个 qualitative failure case："Take me to the soda machine"，robot 把用户带到了 water fountain，因为 VILA-3b quantized 模型将 water fountain 错误 caption 为 "silver machine"。这是典型的 error propagation：前端 VLM 的 semantic hallucination 直接污染了 vector database，下游 LLM agent 无论 reasoning 多强，都无法纠正这种 input-level 的 representation error。

### 7. Intuition Building & Extended Thoughts

为了 build your intuition，我将其与几个相关领域进行 deep connection：

1.  **Hippocampus vs Neocortex Analogy**: ReMEmbR 的架构与人类大脑记忆系统惊人地相似。Vector database 充当 Hippocampus，负责快速记录 episode (what, where, when)。LLM agent 充当 Neocortex，负责 semantic reasoning 和逻辑推演。人类回忆过去也是先通过线索在 Hippocampus 检索，再送到 Neocortex 重构细节。
2.  **Sparse Attention via Indirection**: Transformer 的 self-attention 是 dense 的 $O(N^2)$。ReMEmbR 实际上是通过 LLM agent 实现了一种 dynamic sparse attention。Agent 决定 query vector DB，本质上是计算 attention weights，然后只对 top-k 高相似度的 keys 进行 value aggregation。由于 vector DB 使用 HNSW 等 ANN 算法，复杂度降为 $O(\log N)$。这比直接用长上下文 LLM 高效几个数量级。
3.  **Error Propagation in Multimodal Pipelines**: 这个 paper 暴露了一个严重的 bottleneck：VILA 的 caption 质量。将 video 转 text 是一种 lossy compression。Spatial 关系 (left/right) 和 fine-grained attributes (red cup vs blue cup) 极易在 captioning 阶段丢失。未来的方向可能是直接将 CLIP image features 存入 vector DB，跳过 text translation 阶段 (类似 Memorybank 或 FAISS-based VLM retrieval)，但这会给 LLM agent 的 query formulation 带来挑战 (因为 query 也要是 image embedding)。
4.  **LLM Arithmetic Weakness**: Table I 中 Llama3.1-8b 和 Codestral 在 Positional Error 上的惨烈表现 (上百米误差) 证明了 LLM 在 implicit arithmetic reasoning 上的无能。当 LLM 检索到 "在坐标 (10, 5) 看到咖啡" 和 "在 (12, 8) 看到杯子"，它需要心算找出两点的中点或最短距离，这种 reasoning 在 autoregressive next-token prediction 中极度不稳定。未来可能需要集成 Python code execution tool 让 LLM 显式计算坐标。
5.  **Memory Consolidation**: Paper 在 limitations 中提到 constantly adding repetitive information 会 dilute useful information。人类大脑有 memory consolidation 过程 (睡眠时将 episodic memory 转化为 semantic memory)。对于 robot，可以引入 background process 定期对 vector DB 进行 clustering 和 merging，将多次看到的同一物体 (不同角度、不同时间) collapse 成一个 semantic node，附上 "first seen time" 和 "last seen time" 的 statistics。

### Reference Web Links

*   **Project Page**: [NVIDIA AI-IOT ReMEmbR](https://nvidia-ai-iot.github.io/remembr) (包含代码、视频、dataset)
*   **VILA Model**: [VILA: On Pre-Training for Visual Language Models](https://arxiv.org/abs/2312.07533)
*   **Mixedbread Embedding**: [mxbai-embed-large-v1 Blog](https://www.mixedbread.ai/blog/mxbai-embed-large-v1)
*   **CODa Dataset**: [UT Campus Object Dataset](https://arxiv.org/abs/2303.05552)
*   **OpenEQA**: [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://open-eqa.github.io/)
*   **MobilityVLA**: [Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs](https://mobility-vla.github.io/)

ReMEmbR 提供了一个非常 pragmatic 的架构范式，展示了如何在 edge device 上组合 multiple small models (VILA, Whisper) 与 cloud LLM API 来实现 long-horizon reasoning。这种 modular design 比追求 monolithic million-context VLM 在工程上更具生命力。
