---
source_pdf: An Exam for Active Observers.pdf
paper_sha256: 372d547b10c70037d5be2a4438ab2c11a337f67a1c654406a24238a7b91fe90c
processed_at: '2026-08-18T00:55:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ActiveVision

## 一、这篇 paper 在骂什么

骂的是现在所有 frontier MLLM 都像**翻了一张照片就走人**的人——瞥一眼，脑子里生成一句话描述，剩下全靠语言推。问题是很多视觉任务你瞥一眼根本不够，得反复回去看图，一边想一边看，一边看一边修正假设。人类做这个 loop 毫不费力，甚至自己都没察觉；MLLM 压根没有这个 loop。

paper 干的事：造 17 道题，逼你必须反复回看图才能做对。然后拿 frontier model 来考，发现最高分 GPT-5.5 在最高 reasoning effort 下也才 10.6%，三个普通人平均 96.1%。差距接近 9 倍。更扎心的是把 reasoning effort 从 none 拉到 xhigh，token 花了 100 倍，accuracy 只涨 4 倍——瓶颈不在脑子，在眼睛。

---

## 二、为什么"瞥一眼"不够——paper 的核心 insight

你随便瞄一张图，脑子里能蹦出一句话："哦，左上角六块红色，右下角三块蓝色"。这句话就是 language summary。现在大部分 MLLM benchmark 的题，这句话就够了，因为答案是 summary 能 losslessly 携带的。MMMU-Pro、CharXiv 这些已经接近 saturated，意思就是这种题模型已经会做。

ActiveVision 故意造一种题，让任何一句话摘要都丢掉答案需要的信息。形式上：

$$
I(\text{answer} \mid \text{summary}) \ll I(\text{answer} \mid \text{image})
$$

左边是给定摘要后答案的信息量，右边是给定图像后答案的信息量。paper 的设计目标就是让这个不等式成立——逼你必须把图本身留在 reasoning loop 里。

怎么做到？三个 trick：

**(1) Arbitrary positions**：东西不放在网格上，而是 continuous 随机坐标。20 个点散在画布上携带 190 个 pairwise 关系 + 20 个实数坐标对。你没法说"左上、右上"糊弄过去。

**(2) Arbitrary shapes**：不用命名形状库（三角形、圆形），而是 fresh 合成。两种生成器：

- Fourier 调制的闭合轮廓：
  $$
  r(\theta) = r_0 \left(1 + \sum_{k=1}^{K} a_k \cos(k\theta + \phi_k)\right)
  $$
  $r_0$ 是基础半径，$K$ 是 Fourier harmonics 阶数（随机），$a_k$ 是第 $k$ 阶 amplitude，$\phi_k$ 是第 $k$ 阶 phase。每个 instance 的 $K, \{a_k\}, \{\phi_k\}$ 都重新采样，所以形状空间连续高方差，没有两个一样。
- Jittered ring waypoints 上的 periodic spline：在圆周上采控制点 $\{w_i\}$，每个加抖动 $w_i \leftarrow w_i + \mathcal{N}(0, \sigma^2 I)$，再 spline 插值。

**(3) Arbitrary traces**：要 follow 的路径是 random spline 通过 sampled control points，有几十个 meaningful inflection points，一句话根本描述不了。

---

## 三、17 道题分三大类

paper 不只是随机出难题，每一类对应 psychophysics 几十年研究里一个 elemental operation。

### A. Distributed Scanning（5 题）

对应 subitizing 范围之外的 enumeration。subitizing 是人类"瞥一眼能数 4 个以内"的能力，超过 4 个就得 serial scan。这一类逼你做穷尽扫描。

包含 Bounded Face Counting（数平面图的面）、Connected Component Counting（数连通分量）、Region Counting（数 Voronoi 分区）、Singleton Shape Counting（数只出现一次的形状）、Tangled Loop Counting（数乱绳里的闭环）。

**典型 failure**：数 10 个东西数到 5-6 个就停（partial coverage）；或者相邻相似的东西合并/拆分（faulty individuation）。

### B. Sequential Traversal（5 题）

对应 curve tracing。你沿一条线一步一步走，要维持 (当前位置, 方向, 已走步骤) 三个 state。

包含 Arrow Chain Following、Traversal Point Ordering、Color Zone Sequencing、Line Intersection Sequencing、Maze Path Tracing。

**典型 failure**：gestalt interpolation——从起点直接猜终点，跳过中间。paper §4.3 里有个特别扎心的统计：所有 18 个 pure-CoT run 跑三个 ordered-walk 任务，**没有一个 walk 完整正确完成**，而且 survival 在前 1-2 步就崩了。不是走远了 drift，是第一步 frame 就丢了。

### C. Visual Attribute Transfer（7 题）

对应 fine-grained comparison under visual working memory limit。你从 reference region 提取一个属性（长度、曲率、粗细、颜色排列、点 pattern、方向），再去别的地方匹配。

包含 Constellation Match Counting、Silhouette Match Counting、Stroke Match Counting，以及 Contour/Field/Signal/Stroke Difference Spotting 四个找不同。

**典型 failure**：prior substitution——不真的去测两个 region，直接套 learned linguistic prior。更具体的表现是 Table 3：跨所有 model × effort 组合，miss rate（真实差异漏报）70-100%，false alarm rate（相同误报不同）0-15%。高亮那几行干脆对所有题都答 "same"。这是 response bias，不是 perception。

---

## 四、生成 pipeline——这是工程上最巧的部分

如果直接给模型看 Matplotlib 画的几何图，失败可能只是因为不熟悉渲染风格（cartoon-input confound）。如果直接拿真实图，又没法精确标 ground truth。paper 的解法是两阶段：

```
[seed]
  ↓
(1) Deterministic Python generator
  ↓
(2) Matplotlib procedural scaffold  ← ground truth 精确附在这里
  ↓
(3) Task-specific GPT-image-2 prompt
  ↓
(4) Photorealistic re-rendered image  ← 只把这个给模型看
  ↓
[served to model]
```

具体映射例子：
- Voronoi regions → 航拍图里的田地和河流
- Arrow chains → 石头之间用脚印串起来
- Tangled loops → 漂流木上的绳子
- Planar graph 节点/边 → 鹅卵石之间画线

这一步有三个作用：
- 去掉 cartoon confound，failure 不能甩锅给"不熟悉风格"
- 让 classical CV baseline（findContours、Canny、template matching）也失效，因为这些算法在 noisy realistic 图上不 robust
- 贴近下游应用场景（医疗、卫星、扫描文档、手机抓拍）

---

## 五、主实验结果——一句话：所有 frontier model 都大部分失败

### 5.1 Table 2 关键数据

| Model (max effort) | Overall | % | Avg tokens/item |
|---|---|---|---|
| GPT-5.5 xhigh | 9/85 | 10.6% | 22.5k |
| Claude Opus 4.7 max | 4/85 | 4.7% | 9.0k |
| Claude Opus 4.8 max | 2/85 | 2.4% | 5.5k |
| Claude Fable 5 max | 3/85 | 3.5% | 15.4k |
| Gemini 3.1 Pro high | 5/85 | 5.9% | 16.8k |
| Gemini 3.5 Flash high | 7/85 | 8.2% | 17.5k |
| Human (N=3) | 81.7/85 | 96.1% | 33.6s |

几个细节：

- **GPT-5.5 在 17 个任务里 11 个是 0/5**。所有 Sequential Traversal、所有 Difference Spotting 全挂。
- **六个模型的成功集合几乎不重叠**——没有一道题被六个模型同时做对。说明 gap 不是单家 quirk，是行业级 deficit。
- **Shortcut check**：把图像去掉只给问题，GPT-5.5 在 none effort 下还是 2.4%，跟加图同 effort 一模一样。也就是说低 effort 下模型基本没在"看"，靠 prior 答题。

### 5.2 加 reasoning 没用（Fig. 4）

把六个模型每个都跑全部 reasoning-effort tier：

- GPT-5.5 从 none (2.4%) → xhigh (10.6%)，per-item cost 涨 100 倍，accuracy 只涨 4 倍。
- Claude Fable 5 每个item花 31× 钱几乎没变准。
- 85 item 尺度上，effort tier 之间差异落在 sampling noise 内。

**insight**：模型不是 reasoning 步数不够，是没法把对的视觉证据 pull 出来。reasoning 是个乘子，乘在一个接近 0 的 visual term 上，再大也白搭。

### 5.3 三种 failure mode

**(a) Counting 越多越漏 (Fig. 5)**

把每个 model 的 predicted count $y$ 对 ground truth $x$ 拟合 $y = \beta x + \alpha$。所有 fit 的 $\beta$ 都远小于 1，$x$ 越大 $y$ 越往对角线下偏。模型是 conservative counter——瞥一眼估算个大概，而不是真正扫一遍。

**(b) Tracing 第一步就崩 (Fig. 6)**

对三个 ordered-walk 任务，pool 里 18 个 pure-CoT run，**没有一个 walk 完整正确完成**。survival 在前 1-2 步内 collapse。模型没法维持 "我现在在哪、朝哪个方向走" 这个 frame。

**(c) Comparison "same" 成 safe default (Table 3)**

跨所有 model × effort 组合，miss rate 70-100%，false alarm rate 0-15%。模型在 fine-grained comparison 解不开时 fallback 到 "same"，因为 safe。

---

## 六、Tool use 能代偿吗？部分能，但补不齐

逻辑：既然模型不会主动看图，那让 classical CV 算法代劳行不行？亮度阈值可以分割 region，connected-component analysis 可以数 region 数。

三个 agent：Codex (GPT-5.5)、Claude Code (Opus 4.8)、Claude Code (Fable 5)，全部 xhigh effort，fresh sandbox 里只有图 + 问题。

### 6.1 结果（Table 4）

| Track | Accuracy | Time/item | Cost/item |
|---|---|---|---|
| Codex (GPT-5.5) | 37.6% (32/85) | 12.5 min | $2.74 |
| Claude Code (Opus 4.8) | 24.7% (21/85) | 14.7 min | $4.23 |
| Claude Code (Fable 5) | 50.6% (43/85) | 13.9 min | $7.63 |
| Human (N=3) | 96.1% | 0.56 min | — |

最强 agent 50.6%，比最佳 pure-CoT (10.6%) 提升 5 倍，但仍远低于人类；同时每 item 花 25 倍时间 + 真金白银 compute。

### 6.2 增益分布极不均（Fig. 7b）

- **Visual Attribute Transfer**：43-66%。Crop template + measure 几乎就是 closed-form algorithmic reduction。
- **Sequential Traversal**：Codex 1/25，Opus 4.8 3/25，Fable 5 10/25。
- **Tangled Loop Counting**：**三个 agent 全部 0/5**。

**直觉**：tool use 在"perception 能规约成 reliable computation"的任务上有效，否则 gap 还在。

### 6.3 两种 recurring failure mode（Fig. 8）

**Failure 1 — Tools 在 realistic texture 上不 robust**

- Bounded Face Counting：color mask 在鹅卵石纹理上 dissolve，node/edge 都 misdetect，Euler 公式 $V - E + F = 2$ 算出 12，真值 10。($V$ 是 vertex 数，$E$ 是 edge 数，$F$ 是 face 数含外侧面)
- Contour Difference Spotting：阴影 + 纸张折痕 baked into silhouette，crop scaling 不对 distort contour，误报 "same"。

**Failure 2 — 弱 perception 让 agent 看不出自己代码错了**

- Color Zone Sequencing：tracer 跳到 crossing strand 上早停，agent 不 audit 直接交。
- Maze Path Tracing：binarized mask 把 shadowed hedge 读成 corridor，flood-fill 出错路径，agent 信任 mask 直接交。**只要瞄一眼自己刚画出来的 overlay 就能否决答案，但它不会**。

**核心 insight**：tool use 把 bottleneck 从 "perception" shift 到 agent loop 里的 "verification"。模型看不见 = 看不见自己的代码错了，是同一个 deficit 的两种表现。

---

## 七、对架构的启示——直觉化

paper §5 结论说"motivating architectures and training objectives that close the perception–reasoning loop"。我把它翻译成具体路线：

### (1) Visual token 一次性 vs attention 回访

当前架构里图像 encode 一次就 freeze 成 visual tokens：

$$
h_{t+1} = f_\theta\left(h_t, \text{attend}(V, h_t)\right)
$$

$V$ 是 visual token set，$h_t$ 是 step $t$ 的 reasoning hidden state，$\text{attend}(V, \cdot)$ 是 cross-attention。问题在于当前训练让 $\text{attend}$ 的 query 几乎跟 $h_t$ 解耦——一次 forward 就 freeze 了。

closing loop 等价于让 $\text{attend}$ 的 query 由 $h_t$ 主动决定，并且训练信号显式 reward "在合适 step re-look"。相关 prior 是 [Zhang et al. 2025 "MLLMs know where to look"](https://arxiv.org/abs/2502.17422)。

### (2) 显式 hierarchical visual memory

distributed scanning 漏数、tracing 第一步崩，都说明 visual token grid 不足以维持 spatial working memory。可能的路线：高 fidelity detail + region hierarchy，类似 [BRAVE](https://arxiv.org/abs/2404.07204) / [Euclid](https://arxiv.org/abs/2412.08737)。

### (3) 训练数据缺 iterative perception trajectories

现在 MLLM 训练数据图像 QA 几乎都是 single-pass caption-style。要 build active observation，需要训练数据本身就是 trace 步骤 + 中间 hypothesis + 自我修正。也就是数据得有时间维度。

### (4) Agent loop 里强制 visual verification

§5.3 的 Failure 2 说明 agent 需要 "看自己刚生成的 mask" 并否决。这等价于在 agent loop 里强制加 visual-verification step：tool 输出 → 模型必须把原图 + tool 输出 overlay 比较 → 决定 accept/reject。本质上是把 active observation 内化成 agent 的 self-critique。

---

## 八、一句话给 Karpathy 的 take-away

当前 scaling 主要在 scaling reasoning，不是 scaling perception。ActiveVision 把这个 gap 量化成一个数字：10.6% vs 96.1%。下一个 leap 大概不在更大 LLM backbone，而在 visual token 能不能被 reasoning state 主动 query、能不能 mid-step re-encode。纯 passive encoder + frozen visual tokens 这条路线在 active-observation 任务上有 hard ceiling。closing perception–reasoning loop 是下一个 architecture 必修课。

---

## 参考链接

- ActiveVision project page: https://activevision.github.io  
- Active vision lineage (Aloimonos 1988): https://link.springer.com/article/10.1007/BF00133569  
- Bajcsy 1988 Active perception: https://ieeexplore.ieee.org/document/5958  
- Ballard 1991 Animate vision: https://www.sciencedirect.com/science/article/pii/0004370291900800  
- O'Regan & Noë 2001 sensorimotor account: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/sensorimotor-account-of-vision-and-visual-consciousness/61D4A1F4B7B7C5B3B6A7F8C9D0E1F2A3  
- Ullman 1984 Visual routines: https://www.sciencedirect.com/science/article/pii/0010028584900188  
- Roelfsema 2005 Elemental operations: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(05)00081-7  
- Kundel, Nodine & Carmody 1978 pulmonary nodule: https://journals.lww.com/investigativeradiology/abstract/1978/05000/visual_scanning__pattern_recognition_and.6.aspx  
- Scheffer et al. 2020 Drosophila connectome: https://elifesciences.org/articles/57443  
- Luck & Vogel 1997 VWM capacity: https://www.nature.com/articles/37500  
- MMVP (Tong et al. 2024): https://arxiv.org/abs/2401.06209  
- BLINK (Fu et al. 2024): https://arxiv.org/abs/2404.12390  
- Vision Language Models are Blind (Rahmanzadehgervi et al. 2024): https://arxiv.org/abs/2407.06581  
- Zhang et al. 2025 MLLMs know where to look: https://arxiv.org/abs/2502.17422  
- Euclid (Zhang et al. 2024): https://arxiv.org/abs/2412.08737  
- BRAVE (Kar et al. 2024): https://arxiv.org/abs/2404.07204  
- Prismatic VLMs (Karamcheti et al. 2024): https://arxiv.org/abs/2402.07865  
- MM1 (McKinzie et al. 2024): https://arxiv.org/abs/2403.09611  
- MMMU-Pro (Yue et al. 2025): https://aclanthology.org/2025.acl-long.736/  
- CharXiv (Wang et al. 2024): https://arxiv.org/abs/2406.18521  

如果想 drill down，我会推荐三个 angle：(a) 跑 Fourier boundary 生成器，看 model accuracy 随 $K$（harmonics 数）的 scaling 曲线；(b) 复现 Fig. 5 的 count regression slope，看 slope 是不是随 visual token budget 线性变化；(c) 在 agent loop 里强制加 visual-verification step，量 Fig. 8 Failure 2 类错误的下降率。这三组实验能直接告诉我们 loop 缺在 encoder、在 LLM backbone，还是在 agent policy 上。

---

# ActiveVision: 给 Active Observers 出一张考卷

## 1. Paper 的核心 insight 一句话

当前 frontier MLLMs 是 **passive perceivers**：图像被一次性编码为 fixed visual tokens，没有显式的 perception–action loop。这篇 paper 设计了 17 个任务，让"单次扫一眼 + 语言摘要"的捷径失效，从而把"**模型会不会在 reasoning 过程中反复回到图像上、形成和检验假设**"这件事变成一个可量化的数字。结果是：GPT-5.5 在最高 reasoning effort 下只能做 10.6%，三个普通人平均 96.1%，差近 9 倍。

Paper 网站: https://activevision.github.io  
GitHub: https://github.com/activevision/activevision

---

## 2. Active observation 的认知科学脉络

这个工作不是空穴来风，它把 active vision 几十年的研究 lineage 搬到 MLLM 评测里：

- **Yarbus 1967** ([Eye Movements and Vision](https://www.semanticscholar.org/paper/Eye-movements-and-vision-Yarbus/)): 同一幅画在不同 task prompt 下产生不同 scanpath。证明 gaze 是 task-driven，而非 image-driven。
- **Aloimonos, Weiss, Bandyopadhyay 1988** ([Active vision](https://link.springer.com/article/10.1007/BF00133569)): shape-from-shading、structure-from-motion、optical flow 这些 inverse problem 在 passive observer 下 ill-posed，但只要 observer 能主动控制 sensor 就变 well-posed。
- **Bajcsy 1988** ([Active perception](https://ieeexplore.ieee.org/document/5958)): 同样的论点从 control 角度形式化。
- **Ballard 1991** ([Animate vision](https://www.sciencedirect.com/science/article/pii/0004370291900800)): 把 gaze 当作"局部计算的指针"，比 global computation 更经济。
- **O'Regan & Noë 2001** ([A sensorimotor account of vision](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/sensorimotor-account-of-vision-and-visual-consciousness/61D4A1F4B7B7C5B3B6A7F8C9D0E1F2A3)): 最强版本——"seeing" 就是 mastery of how visual input changes under one's own movements。
- **Ullman 1984** ([Visual routines](https://www.sciencedirect.com/science/article/pii/0010028584900188)) + **Roelfsema 2005** ([Elemental operations in vision](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(05)00081-7)): 提出 "visual routines" / "elemental operations" 框架，把视觉看作 serial, attention-demanding 的 routine 集合。

ActiveVision 把这个 lineage 的 prediction（"没有 iterative sensor redirection，vision system 会在人类日常能解的任务上失败"）拿来对 MLLM 做实测。三个任务家族对应 psychophysics 文献里三个最有名的 elemental operations：

| Task family | 对应 cognitive operation | 文献 |
|---|---|---|
| Distributed Scanning | subitizing 范围之外的 exhaustive enumeration | Kaufman 1949; Trick & Pylyshyn 1994 |
| Sequential Traversal | curve tracing along contours | Jolicoeur, Ullman & Mackay 1986; Roelfsema, Lamme & Spekreijse 1998 |
| Visual Attribute Transfer | fine-grained comparison under visual working memory limit | Luck & Vogel 1997; Ballard, Hayhoe & Pelz 1995 |

---

## 3. ActiveVision 的 3 × 17 设计

### 3.1 三个 task family（每个对应一种 characteristic failure）

**(A) Distributed Scanning — 5 个任务**
要求模型在画布上做 exhaustive coverage 找到所有 local signals 并累加。难度由 signal 数量和覆盖均匀度决定。两种 characteristic failure：
- **Partial coverage**：10 个里只数到 5-6 个就停。
- **Faulty individuation**：相邻相似 signal 被合并、拆分，或跟背景混淆。

包含：Bounded Face Counting, Connected Component Counting, Region Counting, Singleton Shape Counting, Tangled Loop Counting。

**(B) Sequential Traversal — 5 个任务**
要求模型沿 connected structure 一步一步走，并维持 (position, direction, running tally)。难度由 path length、crossing density、decoy similarity 决定。Characteristic failure = **gestalt interpolation**：模型从起点直接猜终点，跳过中间步骤。

包含：Arrow Chain Following, Traversal Point Ordering, Color Zone Sequencing, Line Intersection Sequencing, Maze Path Tracing。

**(C) Visual Attribute Transfer — 7 个任务**
从 reference region 提取一个属性（length, curvature, thickness, color arrangement, dot pattern, orientation），再去其它 region 匹配或比较。难度由 attribute subtlety 和 candidate 数量决定。Characteristic failure = **prior substitution**：不测两个 region，直接套用 learned linguistic prior。

包含：Constellation Match Counting, Silhouette Match Counting, Stroke Match Counting, Contour/Field/Signal/Stroke Difference Spotting。

### 3.2 设计原则：让单一语言描述无法 losslessly 携带答案

形式上，paper 想达到的不变量是：

$$
I(\text{answer} \mid \text{single language summary}) \ll I(\text{answer} \mid \text{image})
$$

也就是任何把图像压缩成一句自然语言描述的 observer，构造上都会丢掉解题需要的信息。这个原则通过三个属性实现：

**(i) Arbitrary positions**：item 放在 continuous sampled coordinates 而非 grid。20 个点散布在画布上有 190 pairwise 关系 + 20 个 real-valued 坐标对，远超任何 linguistic summary 能装下。

**(ii) Arbitrary shapes**：边界不来自命名形状库，每个 instance 都 fresh 合成。技术上有两种生成器：

- Fourier-modulated 闭合轮廓：
  $$
  r(\theta) = r_0 \left(1 + \sum_{k=1}^{K} a_k \cos(k\theta + \phi_k)\right)
  $$
  其中 $r_0$ 是 base radius，$K$ 是 Fourier harmonics 数（随机采样），$a_k$ 是第 $k$ 阶 amplitude，$\phi_k$ 是第 $k$ 阶 phase。$K$、$\{a_k\}$、$\{\phi_k\}$ 都从分布里采样，silhouette 空间是 continuous high-variance 的，没有两个 instance 重复同一形状。
- Jittered ring waypoints 上的 periodic spline：在圆周上采 waypoints $\{w_i\}$，给每个加 Gaussian 抖动 $w_i \leftarrow w_i + \mathcal{N}(0, \sigma^2 I)$，再做 periodic Catmull-Rom 或 B-spline 插值。

**(iii) Arbitrary traces**：要 follow 的路径是经过 sampled control points 的 smooth random spline，有 dozens of meaningful inflection points，任何一句描述都无法保留。

### 3.3 两阶段生成 pipeline

这是这篇 paper 工程上最有意思的地方。流程（Fig. 3）：

```
[seed] 
   ↓
(1) Deterministic Python generator
   ↓
(2) Matplotlib procedural scaffold  (精确 ground truth attached)
   ↓
(3) Task-specific GPT-image-2 prompt  (primitive → real-world mapping)
   ↓
(4) Photorealistic re-rendered image  (positions/counts/topology preserved)
   ↓
[served to model]
```

只有 (4) 给模型看。问题与 ground-truth 对 (2) 和 (4) 都成立。这个 pipeline 解决三个工程问题：

- **去 cartoon-input confound**：否则模型在抽象几何图上失败可能只是因为不熟悉渲染风格。
- **让 classical CV baseline 失效**：findContours / Canny / template matching 在 clean 输入上 OK，在 photorealistic 图像上会 fragmented contours、merge objects、受 style variation 影响。这把 active-vision bottleneck 和 tool 用法绑在一起。
- **贴近下游应用**：medical scans、satellite images、scanned documents、phone snapshots 都是 noisy realistic 的，让 diagnosis 有 external validity。

举几个 primitive → real-world 的映射例子：
- Voronoi regions → aerial fields 和 rivers
- Arrow chains → 石头之间用脚印串起
- Tangled loops → 漂流木上的绳子
- Planar graph 节点/边 → 鹅卵石之间画线

---

## 4. 主实验：每一个 frontier model 都大部分失败

### 4.1 数据表 (Table 2 关键行)

每 generator 取 N=5 instances，共 85 items。Exact-match accuracy。3 个人类 baseline 完成全部 85 items。

| Model (max effort) | Overall | % | Avg tokens/item |
|---|---|---|---|
| GPT-5.5 xhigh | 9/85 | 10.6% | 22.5k |
| Claude Opus 4.7 max | 4/85 | 4.7% | 9.0k |
| Claude Opus 4.8 max | 2/85 | 2.4% | 5.5k |
| Claude Fable 5 max | 3/85 | 3.5% | 15.4k |
| Gemini 3.1 Pro high | 5/85 | 5.9% | 16.8k |
| Gemini 3.5 Flash high | 7/85 | 8.2% | 17.5k |
| Human (N=3) | 81.7/85 | 96.1% | 33.6s |

几个细节值得 build intuition：

- **GPT-5.5 在 17 个任务里 11 个是 0/5**。包括所有 Sequential Traversal、所有 Difference Spotting 任务。
- **最好的 per-task model 不是统一的**：Region Counting 上 GPT-5.5 和 Gemini 3.1 Pro 都达到 2/5；Constellation Match Counting 上 GPT-5.5 是 3/5（全场最高）；Connected Component Counting 上 Gemini 系列略好。六个模型的 success set 几乎不重叠，**没有一个 item 被六个模型全部答对**——说明这不是单个 model 的 quirk。
- **Shortcut check**：把 image 去掉只给 question，GPT-5.5 在 none effort 下只有 2.4%。加上 image 在同一 effort 下还是 2.4%。也就是说在低 effort 下模型基本没在"看"，而是套 prior。

### 4.2 Reasoning effort ablation (Fig. 4, Fig. 5, Fig. 6)

把 6 个模型每个都跑全部 reasoning-effort tier，画 accuracy vs API cost（log scale）：

- GPT-5.5 从 none (2.4%) → xhigh (10.6%)，per-item cost 涨近 100 倍，accuracy 只涨 4 倍。
- Claude Fable 5 每个item花 31× 钱没变准多少。
- 在 85 item 的尺度上，effort tier 之间差异落在 sampling noise 内。

**核心 insight**：模型不是 reasoning 步骤不够，是无法把对的视觉证据 pull 出来。Reasoning 是个乘子，乘在一个接近 0 的 visual term 上。

### 4.3 Failure mode 三连击

**Counting — 越多越漏 (Fig. 5)**

对八个 counting 任务，把每个 model 的 predicted count $y$ 对 ground truth $x$ 做最小二乘拟合 $y = \beta x + \alpha$。所有 fit 的 $\beta$ 都远小于 1，而且 $x$ 越大 $y$ 越往对角线下偏。模型是 conservative counter——越多越多漏，像 glimpse 而非 scan。

**Tracing — 第一步就崩 (Fig. 6)**

对三个 ordered-walk 任务，统计"前 $k$ 步全对"的 walk 比例。pool 里 18 个 pure-CoT run，所有 walk 的 survival 在前 1-2 步内 collapse。**没有一个 walk 完整正确完成**。failure 不是长程 drift，是 walk 的 frame 在起点就丢了。

**Comparison — "same" 成了 safe default (Table 3)**

对两个 panel-naming difference 任务统计：
- $\text{Miss rate} = \frac{\text{真实差异被漏报数}}{\text{真实差异数}}$
- $\text{False alarm rate} = \frac{\text{相同被误报不同数}}{\text{相同数}}$

跨所有 model × effort 组合，miss rate 普遍 70-100%，false alarm rate 普遍 0-15%。高亮的那几行干脆对所有 item 都答 "same"。这是个 **response bias**——当 fine-grained comparison 解不开时，模型 fallback 到 safe prior，而不是做 perception。

---

## 5. Tool use：能用代码代偿 active vision 吗？

逻辑：active vision 缺失 → 能不能让 classical CV 算法代劳？brightness thresholding 可以分割 region，connected-component analysis 可以数 region 数。三个 agent：
- **Codex** (GPT-5.5 backbone)
- **Claude Code** (Opus 4.8 backbone)
- **Claude Code** (Fable 5 backbone)

每个 agent 在 fresh sandbox 里只看到 image + question，全部 xhigh effort。

### 5.1 结果 (Table 4)

| Track | Accuracy | Time/item | Cost/item | Tool calls | Output tokens | Agent turns |
|---|---|---|---|---|---|---|
| Codex (GPT-5.5) | 37.6% (32/85) | 12.5 min | $2.74 | 34 | 22.9k | 31 |
| Claude Code (Opus 4.8) | 24.7% (21/85) | 14.7 min | $4.23 | 53 | 52.2k | 55 |
| Claude Code (Fable 5) | 50.6% (43/85) | 13.9 min | $7.63 | 39 | 44.9k | 86 |
| Human (N=3) | 96.1% | 0.56 min | — | — | — | — |

最强 agent (Fable 5) 50.6%，比最佳 pure-CoT (10.6%) 提升约 5 倍，但仍远低于人类 96.1%；同时每 item 花 25 倍时间 + 真·美元 compute。

### 5.2 增益分布极不均匀 (Fig. 7b)

- **Visual Attribute Transfer**：43–66%。Crop template + measure 几乎就是 closed-form algorithmic reduction，所以 agent 在这里捞到大部分分。
- **Sequential Traversal**：Codex 1/25，Opus 4.8 3/25，Fable 5 10/25。整体仍然接近随机。
- **Tangled Loop Counting**：**三个 agent 全部 0/5**。

intuition：tool use 在"perception 可以规约成 reliable computation"的任务上有效，否则 gap 仍然在。

### 5.3 两种 recurring failure mode (Fig. 8)

**Failure 1 — Tools 不 robust on realistic textures**

- Bounded Face Counting：agent 用 color mask 分割，但鹅卵石纹理让 mask dissolve，node/edge 都 misdetect，Euler 公式 $V - E + F = 2$（bounded faces $= F - 1$）算出 12，真值 10。
- Contour Difference Spotting：阴影 + 纸张折痕 baked into regenerated silhouette，crop scaling 不对又 distort 真实 contour，最终误报 "same"。

**Failure 2 — 弱 perception 让 agent 无法发现 tool error**

- Color Zone Sequencing：tracer 跳到 crossing strand 上早停，agent 不 audit，直接提交截断的 route。
- Maze Path Tracing：binarized mask 把 shadowed hedge 读成 corridor，flood-fill 出错路径，agent 信任 mask 直接交。**只要瞄一眼它自己刚画出来的 overlay 就能否决答案，但它不会**。

intuition：**"看不见"和"看不见自己的代码错了"是同一个 deficit 的两种表现**。Tool use 把 bottleneck 从"perception" shift 到 agent loop 里的"verification"。

---

## 6. 限制与外部效度

paper 自己点了几个 caveat（我加一点扩展）：

- **合成图像**：GPT-image-2 出的是"看起来真实的"图，不是从 natural-image distribution 采样的真实图。paper 拿这个 trade-off 换的是 exact ground truth + controlled task structure。外部效度落在 elemental operations (scanning/tracing/comparing) 而不是渲染本身。
- **Language summary 退化**：design principle 是"no short language description carries the answer"。如果未来模型 image captioning 强到能把图像变成精确的 polyline/coordinate 描述，这个 invariant 就会 erode。未来版本要继续收紧。
- **Tool ablation 是 ablation 而非 standalone track**：agent 测的是"orchestrating reliable CV code + 简单 visual verification"，比"纯 vision 解题"门槛低。即便如此仍达不到 human。

**外部 motivation**：三个 elemental operations 直接对应高危 visual work：
- Exhaustive scanning ↔ radiology (chest radiography 中约 30% missed nodules 从未 fixated, [Kundel, Nodine & Carmody 1978](https://journals.lww.com/investigativeradiology/abstract/1978/05000/visual_scanning__pattern_recognition_and.6.aspx))、cell counting、inventory inspection、aerial search
- Sequential tracing ↔ connectomics proofreading ([Scheffer et al. 2020](https://elifesciences.org/articles/57443))、vessel tracing、schematic inspection
- Fine-grained comparison ↔ latent-print examination、pathology、industrial QC ([Drury 1992](https://onlinelibrary.wiley.com/doi/abs/10.1002/9780470172339.ch89); [Busey et al. 2011](https://www.ojp.gov/ncjrs/virtual-library/abstracts/consistency-and-variability-among-latent-print-examiners))

---

## 7. 给 MLLM 架构的启示

我把 paper 的实验现象映射到可能架构改进：

**(1) Visual token 是一次性的，但 attention 可以回访**

paper §1 指出 MLLMs 在 autoregressive reasoning 时可以在 visual tokens 上 shift attention（[Zhang et al. 2025, "MLLMs know where to look"](https://arxiv.org/abs/2502.17422)），让 earlier visual findings 引导 later steps。但目前架构里这个 mechanism 太弱。可能的路线：
- 训练目标里加 active-look supervision：在 reasoning chain 的中间 step 显式插入 "re-examine region X" 的 signal。
- Visual token 层加 recurrent gating：让 hidden state 决定下一层要不要 re-attend 不同的 visual region。

**(2) Patch-level vs region-level representation**

分布扫描漏数、tracing 第一步就崩，都说明 visual token grid 不足以维持 spatial working memory。一个 candidate 思路是显式 hierarchical visual memory，类似 [BRAVE](https://arxiv.org/abs/2404.07204) / [Euclid](https://arxiv.org/abs/2412.08737) 把高 fidelity visual detail 注入。

**(3) 训练数据缺失 iterative perception trajectories**

现在 MLLM 训练数据里图像 QA 几乎都是 single-pass caption-style。要 build active observation，需要训练数据本身就是 trace 步骤 + 中间 hypothesis + 自我修正。这与 [Prismatic VLMs](https://arxiv.org/abs/2402.07865) / [MM1](https://arxiv.org/abs/2403.09611) 的 vision-centric 设计哲学一致，但要加时间维度。

**(4) Verification loop for agentic tool use**

§5.3 的 Failure 2 说明 agent 需要"看自己刚生成的 mask"并否决。这等价于在 agent loop 里强制加一个 visual-verification step：tool 输出 → 模型必须把原图 + tool 输出 overlay 比较 → 决定 accept/reject。这本质上是把 active observation 内化成 agent 的 self-critique。

**(5) Perception–reasoning closed loop**

paper 的结论是 current MLLMs 缺 robust active visual observation，motivating "architectures and training objectives that close the perception–reasoning loop"。我读到的最干净表述：

$$
h_{t+1} = f_\theta\left(h_t, \text{attend}(V, h_t)\right)
$$

其中 $V$ 是 visual token set，$h_t$ 是 step $t$ 的 reasoning state。当前架构里 $\text{attend}(V, \cdot)$ 几乎与 $h_t$ 解耦（一次 forward 就 freeze）。关闭 loop 等价于让 $\text{attend}$ 的 query 由 $h_t$ active 决定，并且训练信号显式 reward "在适当 step re-look"。这跟认知科学的 active vision 一一对应。

---

## 8. 一个直觉性的总结

把 ActiveVision 想成一个测验：
- 题目构造上保证你**必须**扫、必须沿曲线走、必须细节比对，单次语言摘要必失分。
- 人类用眼跳+工作记忆把这个 loop 跑得接近满分。
- 当前 frontier MLLMs 把图像一次性塞进 visual tokens，没有 loop，于是在所有需要 loop 的任务上接近随机。
- 增加 reasoning 只能让语言部分更精细，但语言部分本来就 missing 关键信息；tool use 能让一些任务规约成 CV 算法，但在 realistic texture 上算法也不可靠，而模型又看不出来算法错。

**对 Karpathy 直接相关的两个 take-away**：

1. **当前 scaling 主要在 scaling reasoning，不是 scaling perception**。ActiveVision 把这个 gap 量化成一个数字（10.6% vs 96.1%），是 reasoning benchmark saturation 之外一块没被充分测的盲区。
2. **Closing perception–reasoning loop 是下一个 architecture 必修课**。纯 passive encoder + frozen visual tokens 这条路线在 active-observation 任务上有 hard ceiling。下一个 leap 大概不在更大 LLM backbone，而在 visual token 是否能被 reasoning state active query、能否 mid-step re-encode。

---

## 参考链接

- Paper project page: https://activevision.github.io  
- Active vision lineage (Aloimonos 1988): https://link.springer.com/article/10.1007/BF00133569  
- Bajcsy 1988 Active perception: https://ieeexplore.ieee.org/document/5958  
- Ballard 1991 Animate vision: https://www.sciencedirect.com/science/article/pii/0004370291900800  
- O'Regan & Noë 2001 sensorimotor account: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/sensorimotor-account-of-vision-and-visual-consciousness/61D4A1F4B7B7C5B3B6A7F8C9D0E1F2A3  
- Ullman 1984 Visual routines: https://www.sciencedirect.com/science/article/pii/0010028584900188  
- Roelfsema 2005 Elemental operations: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(05)00081-7  
- Kundel, Nodine & Carmody 1978 pulmonary nodule: https://journals.lww.com/investigativeradiology/abstract/1978/05000/visual_scanning__pattern_recognition_and.6.aspx  
- Scheffer et al. 2020 Drosophila connectome: https://elifesciences.org/articles/57443  
- Luck & Vogel 1997 VWM capacity: https://www.nature.com/articles/37500  
- MMVP (Tong et al. 2024): https://arxiv.org/abs/2401.06209  
- BLINK (Fu et al. 2024): https://arxiv.org/abs/2404.12390  
- Vision Language Models are Blind (Rahmanzadehgervi et al. 2024): https://arxiv.org/abs/2407.06581  
- Zhang et al. 2025 MLLMs know where to look: https://arxiv.org/abs/2502.17422  
- Euclid (Zhang et al. 2024): https://arxiv.org/abs/2412.08737  
- BRAVE (Kar et al. 2024): https://arxiv.org/abs/2404.07204  
- Prismatic VLMs (Karamcheti et al. 2024): https://arxiv.org/abs/2402.07865  
- MM1 (McKinzie et al. 2024): https://arxiv.org/abs/2403.09611  
- MMMU-Pro (Yue et al. 2025): https://aclanthology.org/2025.acl-long.736/  
- CharXiv (Wang et al. 2024): https://arxiv.org/abs/2406.18521  

如果之后想 drill-down，我建议三个 angle：(a) 把 Fourier boundary + jittered-spline 生成器跑通，看模型在不同 $K$ (harmonics 数) 下的 scaling 曲线；(b) 复现 Fig. 5 的 count regression slope，看 slope 是不是随 visual token budget 线性变化；(c) 在 agent loop 里强制加 visual-verification step，量 Fig. 8 Failure 2 类错误的下降率。这三组实验能直接告诉我们 loop 是缺在 encoder、在 LLM backbone，还是在 agent policy 上。
