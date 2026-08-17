---
source_pdf: Thinking with Visual Primitives.pdf
paper_sha256: 1951785873385498b608b7aa66868f9915ecd78f6c578d42ef52d80ad25f7153
processed_at: '2026-08-12T15:39:51-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在搞什么

## 一句话先讲清楚

这篇 paper 说的就一件事：**让模型在思考的时候能用手指着图说话**。

你回想一下自己数一群鸟、走一个复杂迷宫的时候，脑子里在干什么——你肯定不会纯靠语言描述"左边第三只往右飞到第五只旁边"这种绕来绕去的话，你会本能地用手指点，或者用眼神锁定一个位置,然后心里记住"这里我已经数过了""这条路是死胡同"。手指和视线就是在帮你把 working memory 卸载到外部空间。

MLLM 现在的问题是它的 CoT 全是语言 token，语言这个东西天生不适合编码空间信息——你说"the leftmost red car"，当图里有两百辆车的时候,这句话到底指向哪一辆,模型自己都搞不清。这就是 paper 说的 **Reference Gap**:语言作为指向视觉世界的指针，本质上是 ambiguous 的。

作者干的事就是把 bbox 和 point 这两种 visual primitive 当成"思考的原子单位"塞进 CoT 里。模型在 reasoning 的时候可以直接输出 `<|box|>[[x1,y1,x2,y2]]<|/box|>` 这种坐标,等于在思考链里打了个 spatial anchor。后面所有推理都 reference 这个 anchor，不会跑偏。

参考 Kahneman 的 System 1/2 框架: https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow

---

## 为什么纯语言 CoT 在空间推理上一定崩

你想象一个简单场景:图里有 50 只羊,你问模型"有多少只白羊"。

模型如果纯语言 CoT,它会说"我看到左上角有几只,中间有几只,右边有几只……"。问题是:
- "左上角"是哪里？像素坐标 (100,100) 还是 (200,200)？
- "几只"是 3 还是 4？
- 当它说"中间"的时候,图里已经有 20 只被它"数过"了,它怎么记得哪些数过哪些没数过？

这就是 **logical collapse**。语言 token 是 sequential 的、symbolic 的,空间信息是 parallel 的、geometric 的,两者表征 mismatch。一旦 reasoning chain 超过 3-5 步涉及 spatial reference,error 就 cascade。

前沿模型怎么解决？GPT-5.4、Gemini-3-Flash 走的路子是 high-resolution cropping——把图切成几百个 patch,塞进更多 visual token,让模型"看得更清"。这解决的是 **Perception Gap**——看不见的问题。但看清楚了不代表能 reason。50 只羊都看清了,模型还是数不对,因为它没有"指着记数"的机制。

paper 的核心 insight: **看得清 ≠ 想得清**。真正的瓶颈是 Reference Gap。

---

## 架构:把 visual token 压到极致

这模型的 backbone 是 DeepSeek-V4-Flash,一个 284B 总参数 / 13B 激活参数的 MoE LLM。视觉这边用自家 DeepSeek-ViT,支持任意分辨率输入。

压缩 pipeline 我用一个 756×756 的图走一遍,你看这个压缩比有多夸张:

| 阶段 | 操作 | Token 数 | 累计压缩比 |
|---|---|---|---|
| 原始像素 | 756×756×3 bytes | 1,714,608 | 1× |
| ViT patch embedding | 14×14 patch,756/14=54,54²=2916 | 2,916 | ~196× |
| ViT 输出 3×3 空间压缩 | 9 个 patch token 沿 channel 维 concat 后投影成 1 个 | 324 | ~1,764× |
| LLM 内 CSA | 4 个相邻 visual token 的 KV state 合成 1 个 entry | 81 | 7,056× |

最后 756×756 的图在 KV cache 里只占 **81 个 entry**。对比一下,GPT-5.4 处理 800×800 图要 ~3000+ visual token,Gemini-3-Flash ~2500,这模型只要 ~90。差了一个数量级。

这意味着什么？256K context 长度下可以塞进更多 image turns、更长 CoT。对 agentic 多轮场景,比如 GUI agent 连续操作几十步,这个 token efficiency 是生死线。

CSA 的原理可以参考 DeepSeek-V4 paper: https://arxiv.org/abs/2503.24770

---

## 为什么 pretrain 主力是 box 不是 point

这问题看起来 trivial,但作者给了三条理由,第三条最 interesting。

**理由一:Determinism**。box 紧致包住 object,annotation 相对唯一;point 在 object 内任意位置都"合法",没有严格 ground truth。更糟的是 occlusion 场景——你想给背景物体标 point,结果 point 落到前景 occluder 上了,supervision 噪声爆炸。

**理由二:Generalizability**。box = (top-left point, bottom-right point)。训了 box generation 就免费送了 point generation,因为 box 本身就由两个 point 定义。

**理由三,这条最关键:Information richness**。box 还编码 width/height。这意味着 model 在思考的时候,visual primitive 不只是 spatial pointer,还是 information carrier。比如"这个 box 比那个 box 大"这种推理,box primitive 直接支撑,point 就做不到。

这个 framing 其实是把 visual primitive 当作 **thought 的信息载体**,而不仅是 spatial pointer。我觉得这是 paper 最深的 insight 之一。

---

## 数据 pipeline:data-centric 的范本

这部分是工程细节,但值得讲,因为它展示了怎么做大规模 dirty data 的清洗。

作者从 HuggingFace 等网站爬了 97,984 个 box grounding 相关数据源。然后两步过滤:

**Step I — Semantic Review(97,984 → 43,141)**

用 MLLM 自动筛三类 fatal defect:
- **Machine codes**:纯数字 class id 如 "0", "1"。会污染 language modeling
- **Private entities**:如 "MyRoommate", "ID_Card_1"。无法泛化到 universal concept。但 public celebrity 保留——因为 visual features 可以 generalize
- **Ambiguous abbreviations**:工业检测的 "OK"/"NG"。"OK"到底指 intact apple 还是 intact circuit board？没有 visual descriptiveness

**Step II — Visual-Geometric Review(43,141 → 31,701)**

- **Missing annotations**:miss rate > 50% 直接 discard
- **Truncation/offset**:切掉 head/wheels 的 box 不可接受;稍微 loose 可接受。这个 differential tolerance 设计很巧妙,承认现实数据的 noise 是 heteroscedastic 的
- **Mega boxes**:覆盖 > 90% 图像面积的 box,通常是 classification 数据强行转 detection

最后每类最多采 1000 张,全局去重,~40M 高质量样本。

这个 pipeline 对做 data-centric AI 的人是教科书级的参考。

---

## 四个 Task 的设计哲学

这是 paper 最有思想性的部分。每个 task 都对应 visual primitive 的一种"认知功能"。

### Counting — box primitive 的主场

分 coarse-grained 和 fine-grained:

- **Coarse**:数"dogs"这种通用 category。用 **batch grounding**——一次性定位所有 candidate。作者发现 batch grounding 比 sequential enumeration 更高效,因为利用了模型的 holistic localization 能力
- **Fine-grained**:数"white dogs"或"the dog on the left"。强制 **sequential scan**——逐个 verify 每个 candidate against constraint

Thinking template 三步:
1. Intent Analysis(识别 target)
2. Batch Grounding / Sequential Scan(输出 visual primitives)
3. Statistical Summation(基于 primitives 求和)

### Spatial Reasoning & VQA — box primitive + multi-hop

混合 natural scene (GQA) + synthetic scene (CLEVR)。

CLEVR 的优势是 **execution trace 可程序化生成**——每一步 reasoning 都映射到具体 object ID,可以精确 project 3D bbox 到 2D 作为 supervision。这解决了 dense scene 下 GQA relation 结构太简单的问题。

**Negative sample augmentation** 很关键:刻意构造"对象/关系不存在"的样本,训练模型"faithful refusal"而非 hallucinate。这防 RL 阶段的 always-yes reward hacking。

CLEVR: https://cs.stanford.edu/people/jcjohns/clevr/

### Maze Navigation — point primitive 的巅峰

这是 paper 最 ambitious 的 task。要求模型判断 maze 是否 solvable,并给出 path。

**Maze generation**:DFS / Prim / Kruskal 三种算法;三种 topology(rectangular grid, circular concentric + angular sectors, hexagonal honeycomb)。

**Unsolvable maze 构造**特别精彩:先生成 solvable maze + solution path,然后在 path 中段插几面墙。这样 maze 表面看 solvable,但实际需要 full search 才能确认无解。这是 deliberate 的 adversarial design——防止模型基于启发式猜"看起来有路就 solvable"。

**Difficulty 控制**通过 grid size。Easy 几步 local check;nightmare 数百步 primitive operation 的 long-range composition,需要持续 backtrack。Minimum resolution threshold 保证 primitive 在 hardest config 下仍可感知,把 difficulty 限制在 reasoning complexity 而非 visual ambiguity。

**Thinking content synthesis**:把 DFS 探索过程 verbalize,包括 forward exploration 和 backtracking。每一步都 ground 到 image 坐标。这是教模型 think with primitives 而非 perceive primitives。

### Path Tracing — 连续曲线追踪

多条 Bézier curve 互相 tangle,要求 model follow 指定 start point 找到 end point。

**Intersection disambiguation** 是核心 challenge——crossing 处需要 invoke local geometric-continuity primitive 决定哪个 branch 继续。

特别设计了 **uniform-style mode**:所有 line 同色同 stroke width,去掉 color-based shortcut,强制 model 用 curvature continuity。这是诊断 model 是否真的 internalize 了 path-tracing primitive,还是用 color matching 近似。

**Waypoint density 自适应**:直线段稀疏,高曲率/密集交叉区密集——模拟人类在复杂区域"慢下来、仔细看"的行为。

---

## Post-Training:5 阶段专家融合管线

整个管线长这样:

```
Base pretrained model
   ├── Specialized SFT (TwG) → F_TwG      Specialized SFT (TwP) → F_TwP
   ├── Specialized RL  (TwG) → E_TwG      Specialized RL  (TwP) → E_TwP
   └── Unified RFT (init from base) → F
   └── On-Policy Distillation (E_TwG, E_TwP → F)
```

### Specialized SFT

数据配比 70% general multimodal + text,30% specialized visual primitive data。**关键设计**:TwG(thinking with grounding)和 TwP(thinking with pointing)分开训,避免 specialized data 量小导致 mode conflict。

### Specialized RL(GRPO)

用 GRPO 算法,参考 DeepSeekMath: https://arxiv.org/abs/2402.03300

**最聪明的设计**:RL 阶段 **不显式 supervise thinking content 里的 visual primitive**,只监督 final answer。

为什么？因为 cold-start data 的 primitive 已经过严格 verification,模型在 SFT 阶段已经学会语法。RL 阶段放开 primitive supervision 可以大幅扩展训练数据——只需要 (image, question, answer) 三元组,不需要标注 thinking trace。这把 RL data 从几千扩展到无限。

#### 三层 RM 设计

**Format RM**(rule-based,0-1):检查 primitive 语法;对 TwG 额外检查重复 box(防 SFT 模型陷入 box generation infinite loop)。

**Quality RM**(LLM-based GRM,3-tier {0.0, 0.5, 1.0}):检查 redundancy / thinking-response consistency / 自相矛盾 / referred object meaningfulness / **reward hacking**(model 在 response 里伪造 ground truth 骗 RM)。

**Accuracy RM**(task-specific),逐个看:

**(a) Counting — smooth exponential decay**

$$R(\hat{y}, y) = \alpha \cdot \exp\left(-\beta \cdot \frac{|\hat{y} - y|}{|y| + 1}\right)$$

变量解释:
- $\hat{y}$: predicted count
- $y$: ground truth count
- $\alpha = 0.7$: reward scale,控制 max reward
- $\beta = 3$: decay rate,控制 error sensitivity
- $|y| + 1$: normalization term,让大 count 场景对小 deviation 更容忍

这个比 binary exact-match 优越得多。举例:$y=100, \hat{y}=101$ 给 $R \approx 0.7 \cdot e^{-3/101} \approx 0.680$;$\hat{y}=110$ 给 $R \approx 0.7 \cdot e^{-30/101} \approx 0.516$。dense counting 场景下这种 smooth signal 极其重要,模型不会因为差一个就吃零分。

**(b) Maze Navigation — 5-component weighted reward**

- **Causal exploration progress**:从头扫,遇到第一个 wall violation 就 truncate 后续;计算 explored region 到 endpoint 的最短距离 / GT path length。这是 dense reward 的关键——给 model 探索过程的 credit 而非只看 final answer
- **Exploration completeness**(unsolvable 才用):explored region / 全部 reachable region。强制 model 穷尽证明无解
- **Wall violation penalty**:扫描整个 trace 数 wall-violating transition,即使发生在后期也不放过
- **Final path validity**:binary 检查 solvable maze 的 path 连续合法
- **Answer correctness**:solvability 判断是否对

这种 decomposition 让 reward dense 且 informative——每个 correct primitive operation 都得到 credit。模型不会因为最后一步走错就全盘零分。

**(c) Path Tracing — bidirectional trajectory evaluation**

- **Trajectory accuracy**:双向评估
  - Forward:每个 predicted point 到 GT polyline 的最小距离,取平均——penalize 偏离
  - Reverse:每个 GT point 到 predicted polyline 的最小距离——penalize 跳过 curve 部分
  - 两者平均
  
  双向设计是 critical。只 forward 会让 model 输出几个安全的起点附近 point;只 reverse 会让 model 加 hallucinated detour。双向才能强制完整准确 trace。
  
- **Endpoint accuracy**:start/end 坐标到 GT bbox center 的距离,超过 tolerance 归零
- **Trajectory continuity penalty**:若 trajectory 末点到 predicted endpoint 距离超 threshold,固定 penalty——防止 model 输出 partial trajectory 后"jump"到 guessed endpoint
- **Answer correctness**:endpoint label 是否对

#### RL data 难度分桶

用 SFT cold-start model 做 N rollouts,按 correct rollout 数 k 分三档:
- Easy: k = N(全对)
- Normal: 1 ≤ k < N(部分对)
- Hard: k = 0(全错)

**只选 Normal-level**做 RL——这是 GRPO 的经典做法。Easy 没信号,Hard 信号过稀疏。参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948

### Unified RFT

用两个 expert model E_TwG / E_TwP 在 data pool 上 rollout,生成 RFT data。保留所有 Normal-level + 5% Easy(防 catastrophic forgetting)。从 base pretrained model 重新 init 训练,超参与 cold-start SFT 一致。得到 unified model F。

### On-Policy Distillation (OPD)

F 还是不如单个 expert。用 OPD 把 expert 能力合并进 F。**Full-vocabulary logit distillation**:

$$\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\left(\pi_\theta \parallel \pi_{E_i}\right)$$

变量解释:
- $\pi_\theta$: student model(unified F)
- $\pi_{E_i}$: i-th expert teacher(E_TwG 或 E_TwP)
- $w_i$: teacher weight
- $D_{\text{KL}}$: **reverse** KL divergence, $D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$
- N = 2(两个 teacher)

**为什么 reverse KL 不是 forward KL?** Reverse KL 是 mode-seeking——student 倾向在 teacher high-prob 区域匹配,避免在 teacher low-prob 区域 hallucinate high prob。对 visual primitive 这种结构化输出关键。Forward KL 是 mean-seeking,容易在 expert 都不输出的区域产生 spurious mode。

OPD 用 student 自己的 trajectory(on-policy),让 distillation 不会受 off-policy distribution mismatch 影响。

---

## 实验数据怎么读

看 Table 1,几个 striking 观察:

| Benchmark | Gemini-3-Flash | GPT-5.4 | Claude-4.6 | Qwen3-VL-235B | **Ours** |
|---|---|---|---|---|---|
| CountQA EM | 66.1 | 48.3 | 34.8 | 42.7 | **64.9** |
| Pixmo-Count | 88.2 | 76.6 | 68.7 | 77.2 | **89.2** |
| DS_Finegrained_Counting | 79.1 | 84.2 | 82.6 | 87.2 | **88.7** |
| SpatialMQA | 67.0 | 61.9 | 58.2 | 54.5 | **69.4** |
| DS_Spatial_Reasoning | 93.2 | 81.1 | 97.2 | 96.8 | **98.7** |
| **DS_Maze_Navigation** | 49.4 | 50.6 | 48.9 | 49.6 | **66.9** |
| **DS_Path_Tracing** | 41.4 | 46.5 | 30.6 | 24.5 | **56.7** |

**最 striking 的数据点**:topological reasoning(maze/path tracing)。所有 frontier model 都接近 random——maze ~50%,path tracing ~30-46%。GPT-5.4、Gemini-3-Flash、Claude-4.6 **全部 fail**。这印证了核心 thesis——linguistic CoT 处理拓扑结构本质无效。本文模型 66.9% / 56.7%,绝对提升 15-30 个点。

**Counting**:Gemini-3-Flash 在 CountQA 上其实最强(66.1),本文 64.9 略低;但 Pixmo-Count 和 DS_Finegrained_Counting 上本文领先。说明 fine-grained counting 确实受益于 visual primitive。

**公平性细节**:所有 frontier model 都用 low thinking budget,upscale 低分辨率图到 640k pixel threshold。意味着 frontier model 在"满血"思考预算下可能更强,但本文模型用 90 个 KV entry 做到这个水平,token efficiency 数量级领先。

---

## Qualitative 结果的涌现能力

Fig.7-10 展示了几个让我意外的 emergent capability:

**Cross-domain transfer**:Post-training 数据里**没有中文**,但 Fig.7 显示模型能用中文 think + 输出 box primitive。说明 visual primitive 能力是 language-agnostic 的,inherit 自 base model 的 multilingual 能力。

**World knowledge grounding**:Fig.8 用户问"这附近有 NBA 球队吗?"模型识别出金门大桥(输出 `<|box|>[[0,134,882,795]]<|/box|>`),关联到旧金山 → Golden State Warriors → Chase Center。这是 **visual recognition → world knowledge → spatial QA** 的 cross-modal chain。

**Actionable suggestion**:Fig.8 用户问"如何用这台咖啡机做拿铁"。模型依次定位咖啡机、蒸汽棒、牛奶壶、咖啡豆、拿铁按钮、咖啡杯,输出 step-by-step 操作指引。每个 step 都用 box ground。这是 **tool-use reasoning** 的雏形——visual primitive 在这里既是 reference 也是 action target。

**Topological reasoning trace**:Fig.10 maze 例子最震撼。模型用 DFS 风格输出 18 步探索,每步都 ground 到 `<|point|>[[x,y]]<|/point|>`,包括 dead end backtrack。最终输出完整 path 坐标序列和 `\boxed{True}`。这种**长程拓扑推理 + 坐标 trace** 在现有 MLLM 里基本看不到。

---

## 我读出来的几个 critical 观察

paper 自己承认三个 limitation:

1. **Resolution 限制**:fine-grained 场景 primitive 偶有偏差。可以结合 high-resolution cropping 方法
2. **Trigger word 依赖**:当前需要 explicit trigger word 激活。未来希望 model 自主决定
3. **Topological generalization**:复杂拓扑推理仍是 challenge,cross-scenario 泛化有限

我想补充几个 paper 没明说但我觉得重要的:

4. **Reward hacking surface**:Quality RM 已经防 model 伪造 ground truth,但更复杂的 hacking 模式(如生成 syntactically valid 但 semantically empty primitive sequence)可能仍存在

5. **Distillation gap**:F 经过 OPD 仍不如 expert。Multi-expert distillation 的 Pareto frontier 没探索。如果加第三个 expert 比如 polygon primitive,会不会更好?

6. **Evaluation 限制**:In-house benchmark 是 self-constructed,存在 data leakage 风险。虽然迷宫是 procedural 生成,但风格可能被模型 overfit。如果 maze 用不同渲染引擎测,generalization 如何?

7. **Point vs Box 的 trade-off**:paper 主张 box > point,但 path tracing 和 maze 完全用 point,说明 box 在连续/抽象场景反而受限。两种 primitive 的统一表示(如 polygon primitive)可能更 general

8. **为什么 frontier model 在 maze 上这么差?** 我直觉是它们的 CoT training data 里几乎没有真正的 topological reasoning trace。它们会 verbal reasoning,会 math,但"在二维空间里做 DFS 并 maintain backtrack stack"这件事,语言 token 表征不了。这恰好是这篇 paper 的核心 thesis 的最强证据。

---

## Intuition 收尾:这篇 paper 真正的贡献是什么

如果让我提炼一个核心 intuition:

**Visual primitive 是 multimodal System-2 的 working memory substrate**。

就像人类做数学题要写中间步骤、走迷宫要手指着,MLLM 在 spatial reasoning 时也需要"卸载"intermediate state 到外部表示。自然语言 token 不擅长编码 spatial state——它是 symbolic, sequential, ambiguous;visual primitive 是 geometric, parallel, unambiguous。把两者 interleave,等于给 LLM 一个 spatial scratchpad。

更深一层:这其实是 **modality as cognitive prosthetic** 的思想。我们之前认为视觉是 input modality、语言是 reasoning modality、视觉输出是 perception task。这篇 paper 把视觉输出(primitive coordinate)当作 reasoning 的 native component。这个 framing 可能会影响下一代 MLLM 架构设计。

从效率角度看,7056× 压缩 + 13B active MoE + 90 KV entry 做到 frontier-competitive,说明 token efficiency 和 reasoning quality 不是 trade-off——好的 referential mechanism 反而能让小模型做到大模型做不到的事。这对端侧 multimodal agent 是重要信号。

我觉得这篇 paper 最值得记住的一句话是 abstract 里的:"the future of multimodal intelligence lies not just in seeing more pixels, but in developing more precise and less ambiguous referential mechanisms that bridges the gap between language and the visual world." 翻译成人话:别光想着让模型看更清,得让它能指着想。

参考链接汇总:
- DeepSeek-V4(CSA 架构): https://arxiv.org/abs/2503.24770
- DeepSeek-R1(GRPO): https://arxiv.org/abs/2501.12948
- DeepSeekMath(GRPO 原始): https://arxiv.org/abs/2402.03300
- LLaVA: https://arxiv.org/abs/2304.08485
- CLEVR: https://cs.stanford.edu/people/jcjohns/clevr/
- Pixmo/Molmo: https://huggingface.co/allenai/Molmo
- Kosmos-2(早期 grounding): https://arxiv.org/abs/2306.14824
- GRIT(visual CoT): https://arxiv.org/abs/2505.15879
- Kahneman System 1/2: https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow
- HAI-LLM training framework: https://www.high-flyer.cn/en/blog/hai-llm

---

# Thinking with Visual Primitives 深度技术解析

## 1. Core Thesis: 从 Perception Gap 到 Reference Gap

这篇 paper 的核心 insight 在于把 multimodal reasoning 的瓶颈重新定位。主流工作（如 o3/o4-mini、Gemini-3-Flash、Claude 的 visual scaling 路线）一直在 attack **Perception Gap**——通过 high-resolution cropping、dynamic patching 让模型"看得更清楚"。但作者认为真正卡住 System-2 multimodal reasoning 的是 **Reference Gap**：自然语言作为指向连续视觉空间的指针，本质上是 ambiguous 的。

举个直觉例子：模型在 CoT 里说"the leftmost red car"，但当场景里有几十辆车、几条车道、遮挡关系复杂时，"leftmost"这个 linguistic token 没法稳定地 anchor 到像素坐标。一旦 reasoning chain 里有 3-5 步涉及 spatial reference，error 会 cascade，最终 logical collapse。

作者的解法是 cognitive science 里的 **deictic pointer** 思想——人类数东西、走迷宫时会用手指指着，把 working memory 的 load 卸载到环境里。模型也应该 "point while reasoning"：把 bbox 和 point 提升为 thought 的 minimal unit，直接 interleave 到 CoT trajectory 里。

参考 Kahneman *Thinking, Fast and Slow* 中 System 2 的论述：https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow

---

## 2. Architecture: 极致 Visual Token Efficiency

模型 backbone 是 **DeepSeek-V4-Flash**，一个 284B total params / 13B active params 的 MoE LLM，原生支持 **Compressed Sparse Attention (CSA)**。

### 2.1 视觉编码管线（以 756×756 输入为例）

整个压缩链路可以从 raw pixels 追到 KV cache entry：

| Stage | 处理 | Token 数 | 累计压缩比 |
|---|---|---|---|
| Raw pixels | 756×756×3 | 1,714,608 bytes | 1× |
| ViT patch embedding | 14×14 patch size → 756/14 = 54, 54² = 2,916 patch tokens | 2,916 | ~196× |
| ViT output 3×3 spatial compression | 每 9 个相邻 patch token 沿 channel 维 concat 后投影成 1 token | 324 | ~1,764× |
| LLM 内 CSA | 把 visual token 在 KV cache 里再压 4× | 81 KV entries | 7,056× |

**关键变量解析**：
- patch size = 14：标准 ViT 配置，与 CLIP/SigLIP 兼容
- 3×3 spatial compression：类似 Qwen-VL / LLaVA-NeXT 的 pixel shuffle，把空间信息折叠到 channel 维度，保留 fine-grained 信息但减少 sequence length
- CSA factor 4×：这是 DeepSeek-V4-Flash 的 native 能力，把每 4 个相邻 visual token 的 KV state 合并成 1 个 entry，attention 时 sparse 解码

### 2.2 与 frontier 模型的 token efficiency 对比

看 Fig.1a：对于 800×800 图像，GPT-5.4 大概要 ~3000+ visual tokens 进 KV cache，Gemini-3-Flash ~2500，Claude-Sonnet-4.6 ~2000+。本文模型只要 **~90 KV entries**。这意味着在 256K context 长度下，可以塞进更多 image turns 或更长 CoT，对 agentic 多轮场景关键。

这里和 LLaVA 系列的对比可以参考：https://arxiv.org/abs/2304.08485

---

## 3. Pretraining: 大规模 Box Grounding 数据构造

### 3.1 为什么选 box 而不是 point 作为主 pretrain target

作者给了三条理由，我觉得第三条最 interesting：

1. **Determinism**：box 紧致包住 object，annotation 相对唯一；point 在 object 内任意位置都"合法"，没有严格 ground truth。Occlusion 场景下 point 可能落到 occluder 上，supervision 噪声极大。
2. **Generalizability**：box = (top-left point, bottom-right point)，天然包含 point 表示。训了 box generation 就免费送了 point generation。
3. **Information richness**：box 还编码 width/height，可支撑更复杂 reasoning（比如 object scale 比较、relative position 推断）。

这条 reasoning 实际上是把 visual primitive 当作 **thought 的 information carrier**，而不仅仅是 spatial pointer。

### 3.2 数据 pipeline（97,984 → 40M samples）

整个 pipeline 是 data-centric AI 的范本：

**Step I — Semantic-based Review (97,984 → 43,141)**

用 MLLM 自动筛掉三类 fatal semantic defect：
- **Meaningless machine codes**：纯数字 class id 如 "0", "1"，会污染 language modeling 能力
- **Ungeneralizable private entities**：如 "MyRoommate", "ID_Card_1"，无法泛化到 universal concept；但 public celebrity 保留
- **Ambiguous abbreviations**：如工业检测的 "OK" / "NG"，缺乏 visual descriptiveness

**Step II — Visual-Geometric Review (43,141 → 31,701)**

三类 structural defect：
- **Severe missing annotations**：miss rate > 50% 直接 discard
- **Severe truncation/offset**：切掉 head/wheels 的 box 不可接受；稍微 loose 可接受
- **Mega boxes**：覆盖 > 90% 图像面积的 box 通常是 classification 数据强行转 detection

**Category-based sampling**：每类最多采 M=1000 张，全局去重，最终 ~40M 高质量样本。

这种"differential tolerance"设计很巧妙——它承认现实数据的 noise 是 heteroscedastic 的，不该一刀切。

### 3.3 统一的 output format

```
Box:  <|ref|>TARGET<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>
Point: <|point|>[[x1,y1],...]<|/point|>
```

坐标归一化到 **0–999** 的离散整数空间（不是 0-1 浮点，也不是 0-1000）。这是个很 specific 的选择——它给 LLM 的 tokenizer 留了足够 granularity（3 位数字 token），又避免了 token 数过多。

Point 格式刻意 **不输出 object name**，因为 point 要支持 abstract concept（如 trajectory、path waypoint）。

---

## 4. 四个 Task 的设计哲学

这是 paper 最有思想性的部分。每个 task 都对应一种 visual primitive 的"认知功能"。

### 4.1 Counting（~10k cold-start samples）

分 coarse-grained 和 fine-grained 两种：

- **Coarse-grained**：counting 通用 category（"dogs"）。用 **batch grounding**——一次性定位所有 candidate。这里作者发现 batch grounding 比 sequential enumeration 更高效，因为利用了模型的 holistic localization 能力。
- **Fine-grained**：counting 带属性约束（"white dogs" / "the dog on the left"）。强制 **sequential scan**——逐个 verify 每个 candidate against constraint。

Thinking template 三步：
1. Intent Analysis（识别 target category）
2. Batch Grounding / Sequential Scan（输出 visual primitives）
3. Statistical Summation（基于 primitives 求和）

数据源：FSC147, FSC-147, CrowdHuman, Objects365, Open Images V4, OmniCount, DroneVehicle, NucLS 等。

### 4.2 Spatial Reasoning & General VQA（~9k samples）

混合 natural scene (GQA) + synthetic scene (CLEVR)。

CLEVR 的优势是 **execution trace 可程序化生成**——每一步 reasoning 都映射到具体 object ID，可以精确 project 3D bbox 到 2D 作为 supervision。这解决了 dense scene 下 GQA relation 结构太简单的问题。

**Negative sample augmentation**：刻意构造"对象/关系不存在"的样本，训练模型"faithful refusal"而非 hallucinate。这对 RL 阶段的 reward shaping 也很关键——防止 model 通过 always-yes 拿分。

参考 CLEVR: https://cs.stanford.edu/people/jcjohns/clevr/

### 4.3 Maze Navigation（460k samples）—— 最 ambitious 的 task

这是 paper 最惊艳的设计。要求模型判断 maze 是否 solvable，并给出 path。

**Maze generation**：DFS / Prim / Kruskal 三种算法；三种 topology（rectangular grid, circular concentric + angular sectors, hexagonal honeycomb）。

**Unsolvable maze 构造**：先生成 solvable maze + solution path，然后在 path 中段（远离 start/end）插几面墙。这样 maze 表面看 solvable，但实际需要 full search 才能确认无解。这是 deliberate 的 adversarial design——防止模型基于启发式猜。

**Difficulty 控制**：通过 grid size 控制需要 chain 的 connectivity check 数量。Easy = 几步 local check；nightmare = 数百步 primitive operation 的 long-range composition，需要持续 backtrack。**Minimum resolution threshold** 保证 primitive 在 hardest config 下仍可感知，把 difficulty 限制在 reasoning complexity 而非 visual ambiguity。

**Thinking content synthesis**：把 DFS 探索过程 verbalize，包括 forward exploration 和 backtracking。每一步都 ground 到 image 坐标。这是教模型 think with primitives 而非 perceive primitives。

### 4.4 Path Tracing（125k samples）

Maze 是 grid 离散拓扑；Path Tracing 是连续曲线追踪。多条 Bézier curve 互相 tangle，要求 model follow 指定 start point 找到 end point。

**Intersection disambiguation** 是核心 challenge——crossing 处需要 invoke local geometric-continuity primitive 决定哪个 branch 继续。

特别设计了 **uniform-style mode**：所有 line 同色同 stroke width，去掉 color-based shortcut，强制 model 用 curvature continuity。这是诊断 model 是否真的 internalize 了 path-tracing primitive，还是用 color matching 近似。

**Waypoint density 自适应**：直线段稀疏，高曲率/密集交叉区密集——模拟人类在复杂区域"慢下来、仔细看"的行为。

---

## 5. Post-Training Pipeline: 5 阶段专家融合

```
Base pretrained model
   ├── Specialized SFT (TwG) → F_TwG      Specialized SFT (TwP) → F_TwP
   ├── Specialized RL  (TwG) → E_TwG      Specialized RL  (TwP) → E_TwP
   └── Unified RFT (init from base) → F
   └── On-Policy Distillation (E_TwG, E_TwP → F)
```

### 5.1 Specialized SFT

数据配比 70% general multimodal + text, 30% specialized visual primitive data。**关键设计**：TwG 和 TwP 分开训，避免 specialized data 量小导致 mode conflict。训完得到 F_TwG 和 F_TwG。

### 5.2 Specialized RL（GRPO）

用 GRPO 算法（参考 DeepSeekMath: https://arxiv.org/abs/2402.03300）。**最聪明的设计**：**不显式 supervise thinking content 里的 visual primitive**，只监督 final answer。

为什么这样设计？因为 cold-start data 的 primitive 已经过严格 verification，模型在 SFT 阶段已经学会语法。RL 阶段放开 primitive supervision 可以大幅扩展可用的训练数据——只需要。

#### 5.2.1 三层 RM 设计

**Format RM**（rule-based, 0-1 score）：检查 visual primitive 语法；对 TwG 额外检查重复 box（防 SFT 模型陷入 box generation infinite loop）。

**Quality RM**（LLM-based GRM, 3-tier {0.0, 0.5, 1.0}）：检查 redundancy / thinking-response consistency / 自相矛盾 / referred object meaningfulness / reward hacking（如 model 在 response 里伪造 ground truth 骗 RM）。

**Accuracy RM**（task-specific）：

**(a) Counting — smooth exponential decay**

$$R(\hat{y}, y) = \alpha \cdot \exp\left(-\beta \cdot \frac{|\hat{y} - y|}{|y| + 1}\right)$$

变量解析：
- $\hat{y}$: predicted count
- $y$: ground truth count
- $\alpha = 0.7$: reward scale, 控制 max reward
- $\beta = 3$: decay rate, 控制 error sensitivity
- $|y| + 1$: normalization term，让大 count 场景对小 deviation 更容忍

这个设计比 binary exact-match 优越得多。比如 $y=100$, $\hat{y}=101$ 给 $R \approx 0.7 \cdot e^{-3 \cdot 1/101} \approx 0.680$；$\hat{y}=110$ 给 $R \approx 0.7 \cdot e^{-3 \cdot 10/101} \approx 0.516$。dense counting 场景下这种 smooth signal 极其重要。

**(b) Maze Navigation — 5-component weighted reward**

- **Causal exploration progress**：从头扫，遇到第一个 wall violation 就 truncate 后续；计算 explored region 到 endpoint 的最短距离 / GT path length。这是 dense reward 的关键——给 model 探索过程的 credit 而非只看 final answer。
- **Exploration completeness**（unsolvable 才用）：explored region / 全部 reachable region。强制 model 穷尽证明无解。
- **Wall violation penalty**：扫描整个 trace 数 wall-violating transition，即使发生在后期也不放过。
- **Final path validity**：binary 检查 solvable maze 的 path 连续合法。
- **Answer correctness**：solvability 判断是否对。

这种 decomposition 让 reward dense 且 informative——每个 correct primitive operation 都得到 credit。

**(c) Path Tracing — bidirectional trajectory evaluation**

- **Trajectory accuracy**：**双向**评估
  - Forward：每个 predicted point 到 GT polyline 的最小距离，取平均——penalize 偏离
  - Reverse：每个 GT point 到 predicted polyline 的最小距离——penalize 跳过 curve 部分
  - 两者平均
  
  这是极其重要的设计：只 forward 会让 model 输出几个安全的起点附近 point；只 reverse 会让 model 加 hallucinated detour。双向才能强制完整准确 trace。
  
- **Endpoint accuracy**：start/end 坐标到 GT bbox center 的距离，超过 tolerance 归零
- **Trajectory continuity penalty**：若 trajectory 末点到 predicted endpoint 距离超 threshold，固定 penalty——防止 model 输出 partial trajectory 后"jump"到 guessed endpoint
- **Answer correctness**：endpoint label 是否对

#### 5.2.2 RL data 难度分桶

用 SFT cold-start model 做 N rollouts，按 correct rollout 数 k 分三档：
- Easy: k = N（全对）
- Normal: 1 ≤ k < N（部分对）
- Hard: k = 0（全错）

**只选 Normal-level**做 RL——这是 GRPO 的经典做法（参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948）。Easy 没信号，Hard 信号过稀疏。

### 5.3 Unified RFT

用两个 expert model E_TwG / E_TwP 在 data pool 上 rollout，生成 RFT data。保留所有 Normal-level + 5% Easy（防 catastrophic forgetting）。从 base pretrained model 重新 init 训练，超参与 cold-start SFT 一致。得到 unified model F。

### 5.4 On-Policy Distillation (OPD)

F 还是不如单个 expert。用 OPD 把 expert 能力合并进 F。**Full-vocabulary logit distillation**：

$$\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\left(\pi_\theta \parallel \pi_{E_i}\right)$$

变量解析：
- $\pi_\theta$: student model (unified F)
- $\pi_{E_i}$: i-th expert teacher (E_TwG 或 E_TwP)
- $w_i$: teacher weight
- $D_{\text{KL}}$: **reverse** KL divergence, $D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$
- N = 2 (两个 teacher)

**为什么 reverse KL 而不是 forward KL？** Reverse KL 是 mode-seeking——student 倾向在 teacher high-prob 区域匹配，避免在 teacher low-prob 区域 hallucinate high prob。对于 visual primitive 这种结构化输出关键。Forward KL 是 mean-seeking，容易在 expert 都不输出的区域产生 spurious mode。

OPD 用 student 自己的 trajectory（on-policy），让 distillation 不会受 off-policy distribution mismatch 影响。参考原 paper: https://arxiv.org/abs/2503.24770 (DeepSeek-V4)

---

## 6. 实验结果分析

Table 1 关键数据：

| Benchmark | Gemini-3-Flash | GPT-5.4 | Claude-4.6 | Qwen3-VL-235B | **Ours** |
|---|---|---|---|---|---|
| CountQA EM | 66.1 | 48.3 | 34.8 | 42.7 | **64.9** |
| Pixmo-Count | 88.2 | 76.6 | 68.7 | 77.2 | **89.2** |
| DS_Finegrained_Counting | 79.1 | 84.2 | 82.6 | 87.2 | **88.7** |
| SpatialMQA | 67.0 | 61.9 | 58.2 | 54.5 | **69.4** |
| DS_Spatial_Reasoning | 93.2 | 81.1 | 97.2 | 96.8 | **98.7** |
| **DS_Maze_Navigation** | 49.4 | 50.6 | 48.9 | 49.6 | **66.9** |
| **DS_Path_Tracing** | 41.4 | 46.5 | 30.6 | 24.5 | **56.7** |

几个 striking 观察：

1. **Topological reasoning**：所有 frontier model 在 maze/path tracing 都接近 random（~50% / ~40%），**GPT-5.4、Gemini-3-Flash、Claude-4.6 全部 fail**。这印证了 paper 的核心 thesis——linguistic CoT 处理拓扑结构本质无效。本文模型 66.9% / 56.7%，绝对提升 15-30 个点。

2. **Counting**：Gemini-3-Flash 在 CountQA 上其实最强（66.1），本文 64.9 略低；但 Pixmo-Count 和 DS_Finegrained_Counting 上本文领先。说明 fine-grained counting 确实受益于 visual primitive。

3. **Spatial reasoning**：DS_Spatial_Reasoning 上本文 98.7，超过 Claude 的 97.2。这个 benchmark 来自 CLEVR 重构的多选题，验证了 synthetic pretrain + visual primitive 在 multi-hop reasoning 的优势。

4. **公平性细节**：所有 frontier model 都用 low thinking budget，upscale 低分辨率图到 640k pixel threshold。说明 frontier model 在"满血"思考预算下可能更强，但本文模型用 90 个 KV entry 做到这个水平，token efficiency 是数量级领先。

---

## 7. Qualitative 结果的涌现能力

Fig.7-10 展示了几个让我意外的 emergent capability：

### 7.1 Cross-domain transfer

Post-training 数据里**没有中文**，但 Fig.7 显示模型能流利用中文 think + 输出 box primitive。说明 visual primitive 能力是 language-agnostic 的，inherit 自 base model 的 multilingual 能力。

### 7.2 World knowledge grounding

Fig.8 第一个例子：用户问"这附近有 NBA 球队吗？"模型识别出金门大桥（输出 `<|box|>[[0,134,882,795]]<|/box|>`），关联到旧金山 → Golden State Warriors → Chase Center。这是 **visual recognition → world knowledge → spatial QA** 的 cross-modal chain。

### 7.3 Actionable suggestion

Fig.8 第二个例子：用户问"如何用这台咖啡机做拿铁"。模型依次定位咖啡机、蒸汽棒、牛奶壶、咖啡豆、拿铁按钮、咖啡杯，输出 step-by-step 操作指引。每个 step 都用 box ground。这是 **tool-use reasoning** 的雏形——visual primitive 在这里既是 reference 也是 action target。

### 7.4 Counterfactual & humor

Fig.9 展示了 humor understanding、escape room guidance、counter-commonsense VQA。说明 visual primitive 不只是"找东西"的工具，还能 anchor 高级认知任务。

### 7.5 Topological reasoning trace

Fig.10 的 maze 例子最震撼。模型用 DFS 风格输出 18 步探索，每步都 ground 到 `<|point|>[[x,y]]<|/point|>`，包括 dead end backtrack。最终输出完整 path 坐标序列和 `\boxed{True}`。这种**长程拓扑推理 + 坐标 trace** 在现有 MLLM 里基本看不到。

---

## 8. 与相关工作的 differentiation

### 8.1 vs Visual CoT 系列（Shao et al. NeurIPS 2024）

Visual CoT 把 bbox 当作 post-hoc verification mechanism，主要 attack perception-heavy task。本文把 bbox/point 当作 **intrinsic thought medium**，attack reasoning-heavy task（topological）。

参考: https://arxiv.org/abs/2505.15879 (GRIT) 和 https://arxiv.org/abs/2506.11991 (VGR)

### 8.2 vs High-resolution cropping（o3, Gemini-3）

那些方法 attack Perception Gap，scale visual token 数量。本文 attack Reference Gap，**减少** token 数量同时增加 referential precision。两者互补——paper 的 Limitation 也承认可以结合。

### 8.3 vs Pixmo / Molmo

Pixmo-Points 提供高质量 point annotation，本文用其作为预训练数据源之一。但 Pixmo 数据规模和多样性都不够，所以作者从 97k HuggingFace dataset 爬 + 两步过滤构造 40M 样本。

参考: https://huggingface.co/allenai/Molmo

### 8.4 vs Kosmos-2

Kosmos-2 是早期 grounding MLLM，但只在最终答案输出 box，不参与 thinking。本文是 Kosmos-2 思想的 System-2 升级版。

参考: https://arxiv.org/abs/2306.14824

---

## 9. 局限性与未来方向

Paper 自己承认三个 limitation：

1. **Resolution 限制**：fine-grained 场景 primitive 偶有偏差。可以结合 high-resolution cropping 方法。
2. **Trigger word 依赖**：当前需要 explicit trigger word 激活"thinking with visual primitives"。未来希望 model 自主决定。
3. **Topological generalization**：复杂拓扑推理仍是 challenge，cross-scenario 泛化有限。

我还想补充几个 critical 观察：

4. **Reward hacking surface**：Quality RM 已经防 model 伪造 ground truth，但更复杂的 hacking 模式（如生成 syntactically valid 但 semantically empty primitive sequence）可能仍存在。
5. **Distillation gap**：F 经过 OPD 仍不如 expert。Multi-expert distillation 的 Pareto frontier 没探索。
6. **Evaluation 限制**：In-house benchmark 是 self-constructed，存在 data leakage 风险（虽然迷宫是 procedural 生成，但风格可能被模型 overfit）。
7. **Point vs Box 的 trade-off**：paper 主张 box > point，但 path tracing 和 maze 完全用 point，说明 box 在连续/抽象场景反而受限。两种 primitive 的统一表示（如 polygon primitive）可能更 general。

---

## 10. Intuition 总结

如果让我提炼这篇 paper 的核心 intuition：

**Visual primitive 是 multimodal System-2 的 working memory substrate**。

就像人类做数学题要写中间步骤、走迷宫要手指着，MLLM 在 spatial reasoning 时也需要"卸载" intermediate state 到外部表示。自然语言 token 不擅长编码 spatial state——它是 symbolic, sequential, ambiguous；visual primitive 是 geometric, parallel, unambiguous。把两者 interleave，等于给 LLM 一个 spatial scratchpad。

更深一层：这其实是 **modality as cognitive prosthetic** 的思想。我们之前认为视觉是 input modality、语言是 reasoning modality、视觉输出是 perception task。本文把视觉输出（primitive coordinate）当作 reasoning 的 native component。这个 framing 可能会影响下一代 MLLM 架构设计。

从效率角度看，7056× 压缩 + 13B active MoE + 90 KV entry 做到 frontier-competitive，说明 token efficiency 和 reasoning quality 不是 trade-off——好的 referential mechanism 反而能让小模型做到大模型做不到的事。这对端侧 multimodal agent 是重要信号。

参考链接汇总：
- Paper PDF: https://arxiv.org/abs/2506.xxxxx (注：原 paper 来自 DeepSeek-AI，arxiv ID 待补)
- DeepSeek-V4: https://arxiv.org/abs/2503.24770
- DeepSeek-R1 (GRPO): https://arxiv.org/abs/2501.12948
- LLaVA: https://arxiv.org/abs/2304.08485
- CLEVR: https://cs.stanford.edu/people/jcjohns/clevr/
- Pixmo/Molmo: https://huggingface.co/allenai/Molmo
- Kosmos-2: https://arxiv.org/abs/2306.14824
- Visual CoT: https://arxiv.org/abs/2505.15879
- Kahneman book: https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow
