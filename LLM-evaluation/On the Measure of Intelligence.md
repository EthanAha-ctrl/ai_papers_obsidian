---
source_pdf: On the Measure of Intelligence.pdf
paper_sha256: 773b9ecec3df33e664ab5a784f9e48818e7fd0d8e211b3243f4f4e0fdecb2fee
processed_at: '2026-08-05T23:40:31-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Chollet 这篇 paper 在讲什么

## 一句话版本

现在的 AI 圈在**作弊** —— 用无限数据和算力砸出"技能"，然后管这叫"智能"。Chollet 说这跟智能没半毛钱关系，智能是**学习新东西的效率**，不是你已经学会了多少东西。

---

## 核心比喻：考试

想象两种考试：

**考试 A**：给学生一亿道题的题库（带答案），然后考试就从里面抽。学生只要背下来就行。这就是现在 AI benchmark 的逻辑 —— ImageNet 1500万张图，AlphaGo 自己跟自己下几百万盘棋，OpenAI Five 打了 45,000 年的 DotA2。

**考试 B**：给学生三道例题，然后出一道从来没见过的题。这才是测**智商**，测的是你从三道例题里**提炼规律**的能力，测的是你**现场学习**的速度。

Chollet 说的就是：整个 AI 领域几十年来一直在做考试 A，然后对外宣称"我们离人类智能越来越近了"。问题是考试 A 考的是**记忆力 + 算力**，跟智能本身**毫无关系**。

---

## 为什么 skill ≠ intelligence

这是全文最关键的 insight，值得反复琢磨。

### "买 skill" 的两种方式

**方式 1：Hard-code**

你是 engineer，看到一个 IQ test 题目，自己想出答案，写成 if/else 程序。这程序能"通过"IQ test，但它有智能吗？没有。智能在你脑子里，在你想出答案的那个**过程**里，不在写下来的代码里。代码只是你思考过程的**结晶产物**。

这就像你证明了 Fermat 大定理，写在纸上。纸上的字没有智能，智能在证明的那个思考过程里。

**方式 2：堆数据**

这个更隐蔽，也更 dangerous。假设你有一个最蠢的学习算法 —— nearest neighbor lookup，本质是个 hash table。给它足够多的训练数据，它也能"解决"任何任务。

比如视频游戏：你把每种可能的 screen state 都存下来，对应一个 action。只要数据够密，覆盖够全，这个 hash table 就能玩得很好。OpenAI Five 就是这个逻辑的高级版本 —— 45,000 年的游戏经验，本质是把 situation space 给 dense-sample 了。

**但这个 hash table 有 generalization 能力吗？零。** 遇到训练数据没覆盖的新情况就崩。OpenAI Five 发布后被非冠军人类玩家几天就找到了 exploit，因为它没有**适应新情况**的能力。

### Chollet 的关键判断

> Deep Learning 模型本质上是 locality-sensitive hash table 的高级版本。它们能被训练到任何 task 上的任意 skill 水平，代价是需要对 input-target space 做 dense sampling。这对 high-value real-world 应用（L5 自动驾驶等）是不现实的。

引用 Bojarski et al.：30 million 个训练场景都不够一个 Deep Learning 模型学会开车。

---

## Generalization 的谱系

Chollet 给了一个很清晰的 hierarchy，这是 build intuition 最好的框架：

| 层级 | 通俗说法 | 例子 |
|------|---------|------|
| **Level 0**：无泛化 | 死记硬背 | 排序算法、井字棋穷举 —— 能证明对所有输入都正确，不存在"没见过"的输入 |
| **Level 1**：Local generalization | **鲁棒性**，同一分布内的新样本 | 猫狗分类器见过 150×150 的猫，能认新的 150×150 猫。这就是现在 Deep Learning 做的事 |
| **Level 2**：Broad generalization | **灵活性**，跨任务跨场景 | L5 自动驾驶遇到从没见过的路况、天气、城市；机器人进陌生厨房煮咖啡。现在 AI 基本做不到 |
| **Level 3**：Extreme generalization | **通用智能**，人类这种 | 只看过几个例子就能解决一个概念上相关但形式上全新的任务 |
| **Level 4**：Universality | 理论概念 | 任何宇宙中的任何任务，No Free Lunch theorem 说不可能 |

**AI 的历史就是沿着这个谱系慢慢往上爬。** Symbolic AI = Level 0，Machine Learning = Level 1，现在想往 Level 2 走。但 AI 圈把 Level 1 的成就（打赢 Go 世界冠军）宣传成"向 Level 3 迈进"，这是混淆了层级。

---

## "通用智能" 到底多通用

这部分很有意思。很多人以为 "Artificial General Intelligence" 意味着能解决**任何**问题的终极 AI。Chollet 说这是误解。

### 人类智能其实很 narrow

No Free Lunch theorem 告诉我们：对所有可能的问题平均而言，任何算法都跟随机猜等价。所以任何"智能"都只能是针对某类问题的特化。

人类智能看起来"通用"，但其实只在**人类经验相关的任务范围**内通用。人类进化来适应东非大草原的奔跑、狩猎、社交，结果也能弹钢琴、解线性代数、游过英吉利海峡。这确实很神奇，但这不等于"通用"。

**人类的局限很明显**：
- **维度偏见**：2D 导航和 shape-packing 极强，3D 还行，4D+ 基本无能。因为海马体的 place cells / grid cells 是为 2D 导航进化的
- **TSP（旅行商问题）**：小规模接近最优，但把目标从"最短路径"改成"最长路径"后，人类表现比最简单的 heuristic 还差（Macgregor & Chu 2011）。因为我们靠的是**感知策略**，不是通用搜索
- **长程规划**：超过几年的规划人类就很不擅长
- **工作记忆**：心算 10 位数乘法基本做不到

### 类比：体能

"体能"这个概念跟"智能"很像。你测体能：100 米跑、马拉松、游泳、引体向上...所有项目成绩正相关，能提取出一个 "physical g factor"。但这不意味着一个很 fit 的人能应对**任何**物理环境 —— 在金星表面或木星大气里，人类身体完全不 fit。

所以 Chollet 的结论：**AI 应该瞄准 "human-like intelligence"，以人类智能为参照系，benchmark 也应该对齐人类**。这不是人类中心主义的偏见，是因为：
1. "通用"永远是相对某个 scope 的
2. 人类经验相关的 scope 是我们唯一能 meaningfully 评估的
3. 我们造 AI 的目的本来就是服务人类

---

## 人类到底有什么 innate priors

这是 ARC 设计的 foundation。Chollet 引用 Spelke & Kinzler (2007) 的 **Core Knowledge theory**：

人类天生自带四套认知系统：

### 1. Objectness & elementary physics
- **Cohesion**：物体是连续的整体，不会散开
- **Persistence**：物体不会突然消失或凭空出现
- **Contact**：物体之间不隔空作用，不能互相穿透

婴儿几个月大就有这些假设。这解释了为什么我们能轻易"看"到一个物体在移动，而不会觉得它每一帧都是不同的东西。

### 2. Agentness & goal-directedness
- 部分"物体"是 agent，有自己意图
- Agent 的行为是 goal-directed 的，且追求效率
- Agent 之间有 contingency 和 reciprocity

你看到一个东西在追另一个东西，会自动理解成"追逃"关系，这是 innate 的。

### 3. Natural numbers & elementary arithmetic
- 对小数量的抽象表征
- 跨感官模态适用（看到 3 个点、听到 3 声响，知道是同一个"3"）
- 能做加减、比较、排序

### 4. Elementary geometry & topology
- 距离、方向、内外关系
- 2D / 3D 导航能力

Chollet 的论点：**测试 human-like general intelligence 的 benchmark 应该只用这些 priors，不用任何后天获得的知识**（语言、chess 规则、箭头符号等）。这样人类和 AI 才能在同一个起跑线上比较。

---

## 形式化定义：用 AIT 量化"泛化难度"

这是 paper 最技术性的部分，但核心直觉很简洁。Chollet 用 **Algorithmic Information Theory**（算法信息论）来量化三个东西：

### 核心工具：Kolmogorov Complexity

$H(s)$ = 输出字符串 $s$ 的最短程序长度。简单说，$H(s)$ 衡量 $s$ 的"信息含量"或"内在复杂度"。

$H(s_1 | s_2)$ = 以 $s_2$ 为输入，产生 $s_1$ 的最短程序长度。衡量"已知 $s_2$ 后，还需要多少信息才能得到 $s_1$"。

### Generalization Difficulty (GD)

$$GD = \frac{H(\text{最简的足够好的 evaluation 解} | \text{最简的 training 最优解})}{H(\text{最简的足够好的 evaluation 解})}$$

**人话翻译**：

- 分子：已知"训练集上表现最优的最简程序"后，还需要修改多少才能得到"测试集上表现足够好的最简程序"
- 分母：这个 evaluation 解本身的复杂度
- 结果归一化到 $[0, 1]$

**GD = 0** 意味着训练最优解直接就能在测试上用，不需要任何泛化（比如排序算法，训练时和测试时逻辑一样）。

**GD 高** 意味着训练时最聪明的做法在测试时不够用，需要**额外学习/适应**才能应对新情况。

### 反直觉但重要的点

Occam's razor 说"最简的假设最可能正确"。在 ML 里这通常被理解为"最简的 model 最能泛化"。

**但 Chollet 说这是错的**。最简的 training-consistent program 往往会**丢弃**那些"对过去没用但对未来可能有用"的信息。泛化要求你**为未来的不确定性做好准备**，这跟压缩过去的数据是**矛盾的**。

**例子**：

训练点：
- $(x = -0.75, label = False)$
- $(x = 0.15, label = True)$

最简 training 解：$\lambda(x): x > 0$ （2 个符号）

但测试点 $(x = -0.1, label = True)$ 会 fail。

如果用 nearest neighbor（存所有训练点），程序更长，但能处理这个测试点。为什么？因为它**保留了更多信息**，为未来不确定性做了准备。

这就是为什么"最简 consistent program" 不一定是最好的泛化器 —— 泛化需要"覆盖更大的未来 situation space"，这有信息成本。

### Priors 的形式化

$$P = \frac{H(Sol) - H(Sol | IS_{initial})}{H(Sol)}$$

**人话**：初始系统（包括 architecture、weights、built-in knowledge）距离"足够好的解"有多近。priors 高 = 你一开始就离答案很近。

注意：这不是"初始系统有多大"（那是 $H(IS_{initial})$），而是"初始系统里**有多少跟任务相关的信息**"。一个巨大的系统但跟任务无关，priors 分还是很低，只被 minimally penalize（因为 indexing overhead 很小）。

### Experience 的形式化

$$E_{total} = \frac{1}{H(Sol)} \sum_t [H(Sol | IS_t) - H(Sol | IS_t, data_t)]$$

**人话**：每一步 data 能减少多少对 solution 的 uncertainty，所有步加起来，归一化。

关键设计：
- **只算 relevant information**：noisy data 不被计入，不惩罚"脏"训练数据
- **只算 novel information**：重复的数据对 fast learner 不计入（它已经学会了），但对 slow learner 继续计入（它还在学）
- **Eager sum**：逐步累加，而非全局池化，这样能区分 fast learner 和 slow learner

### 最终的 intelligence 定义

$$I = \underset{\text{tasks}}{\text{Avg}} \left[ \underbrace{\omega_T \cdot \theta_T}_{\text{任务价值 × 技能阈值}} \cdot \underset{\text{curricula}}{\mathbb{E}} \left[ \frac{\text{泛化难度}}{\text{Priors} + \text{Experience}} \right] \right]$$

**完全人话**：

> 对每个任务，你的 intelligence 贡献 = 这个任务多重要 × 你要达到什么水平 × （这个任务有多需要泛化 / 你用了多少 priors 和 experience 来达到这个水平）。所有任务平均一下就是你的 intelligence。

**高分的人**：
- 能在**高泛化难度**的任务上（很多不确定性、需要适应）
- 用**很少的 priors 和 experience**
- 达到**足够好的 skill**

**低分的人**：
- 只能在低泛化难度任务上（死记硬背就行的）
- 需要海量数据
- 或者靠大量 hard-coded priors

### 核心公式拆解

$$\text{Intelligence} \propto \frac{\text{Generalization Difficulty}}{\text{Priors} + \text{Experience}}$$

这就像**学习效率**：
- GD = 需要学会的新东西有多少（分子，越难越好）
- Priors + Experience = 你花了多少资源来学（分母，越少越好）
- 比值 = 单位资源的"泛化产出"

**一个高智商的人**：给他三道例题，他能解决一个全新的、跟训练例子概念相关但形式不同的任务。

**一个低智商的 AI**：给它一百万张猫图，它能认猫，但换个角度的猫、不同光照的猫、卡通猫就崩了。它"解决"了 cat classification 这个 task，但它的 intelligence ≈ 0，因为 GD 很低（就是分布内泛化），experience 却巨大。

---

## ARC：把这个理念做成 benchmark

### 设计逻辑

ARC 就是 Chollet 试图实现"考试 B"的尝试：

1. **1000 个 task**，每个都是独特的，没有两个 task 概念完全一样
2. 每个 task 只给 **3 个左右**示例（input grid → output grid）
3. 测试时给一个**从没见过的新 task**，要你产出 output grid
4. **只允许用 Core Knowledge priors**：objectness、counting、基本几何、对称性...
5. **不允许**：语言、箭头、真实世界概念（猫、狗、chess 规则）
6. Grid 是 $1 \times 1$ 到 $30 \times 30$，10 种颜色，就是色块网格

### 为什么这个设计

| 设计选择 | 对应的原则 |
|---------|-----------|
| 只给 3 个例子 | 控制 experience |
| 测试任务对 developer 也不知 | 测 developer-aware generalization，不是 skill |
| 只用 Core Knowledge priors | 人类和 AI 在同一起跑线，显式列出假设 |
| 任务手工设计，不是程序生成 | 避免逆向 master program，增加多样性 |
| Binary success（全对才算对）| 避免"近似正确"被算分，强调精确泛化 |
| Private evaluation set | 严格 enforce "developer 不知道测试任务" |

### 人类表现 vs AI 表现

**人类**：高分人类基本能解决大部分 ARC 任务，第一次见就能做。因为人类的 Core Knowledge priors + fluid intelligence 足以从 3 个例子提炼抽象规律。

**AI（包括 SOTA Deep Learning）**：到 2019 年基本无法 approach。到 2024 年，GPT-4o、Claude 3.5 Sonnet 在 ARC 上仍只有 10-30%，而人类 85%+。这**验证了 Chollet 的论点**：Deep Learning 是 local generalization system，在 broad generalization 任务上天然受限。

### ARC solver 长什么样

Chollet 认为应该是 **program synthesis**：

1. **构建 DSL**：Domain-Specific Language，能表达所有 ARC 任务的解。Core Knowledge priors 作为 basis functions。这是最关键的 subproblem。
2. **生成候选**：对每个 task，用 DSL 生成能解释 demo 的 candidate programs
3. **选择 top-3**：基于 simplicity 或 learned likelihood
4. **测试**：用 top candidates 生成 test output

2020-2024 的 ARC Prize 竞赛验证了这个判断：top solutions 基本是 **hand-crafted DSL + search**，而非纯 learning。这某种程度上说明 ARC 确实需要**显式抽象和推理**，不是靠 scale 就能刷过去的。

---

## 对 AI 研究的实际影响

### 应该做什么

1. **把 intelligence 当 objective function**：设计能 approximate Chollet 公式的 metric
2. **开发 broad abilities，别追 skill**：因为 skill 可以被 priors/data 买来
3. **关注 program synthesis**：把"agent"拆成"intelligent system"（生成器）和"skill program"（产物），前者才是智能所在
4. **研究 curriculum**：更好的课程能提升 expressed intelligence
5. **硬编码 Core Knowledge priors**：作为 general AI 系统的基础

### 应该避免什么

1. **别拿"打赢世界冠军"当 AGI 进展的证据**：DeepBlue、AlphaGo、OpenAI Five 都是 local generalization 的极致，跟 broad AI 无关
2. **别迷信"无限数据 + scale"**：这只是在 dense-sample situation space，没有产生泛化能力
3. **别用 task-specific benchmark 衡量 generalization**：必须用**未知任务**测
4. **别混淆 process 和 artifact**：你写的程序没有智能，你的思考过程有

---

## 我自己的 take

这篇 paper 最大的价值是**概念澄清**。它让你看清楚：

**AlphaGo 打败李世石** 那天，AI 领域"离 general AI 更近了"这个感觉是**错的**。我们只是做了一个很窄的、靠无限 self-play 数据训练的 pattern matcher。它跟通用智能的距离，跟 DeepBlue 打败 Kasparov 那天一样远 —— 甚至更远，因为我们产生了"progress is happening"的错觉。

**Chollet 的形式化**虽然不可计算（Kolmogorov complexity 不可计算），但作为**思考框架**极有价值。它告诉你：评价一个 AI 系统，要问的不是"它能做什么"，而是"它学新东西有多快、能泛化多远、用了多少 priors/data"。

**ARC** 是一次诚实的尝试，试图造一个"刷不了题"的 IQ test。它可能不完美，可能被某种巧妙的 DSL + search 攻克，但它至少**问对了问题**：我们到底能不能从 3 个例子学会新东西？

到 2024 年的答案是：人类能，AI（包括最先进的 LLM）基本不能。这个 gap 本身就是最有价值的信号 —— 它说明我们离真正的 general AI 还有很长的路，而且这条路**不是靠 scale 就能走通的**。

参考：
- 原论文：https://arxiv.org/abs/1911.01547
- ARC 数据集：https://github.com/fchollet/ARC
- ARC Prize（2024+）：https://www.arcprize.org/
- Chollet & Knoop 的 ARC-AGI 技术报告：https://arcprize.org/blog
- Spelke Core Knowledge 论文：https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(07)00006-9
- DreamCoder（neuro-symbolic 尝试）：https://arxiv.org/abs/2006.08381
- Legg & Hutter Universal Intelligence：https://www.mmlab.tuwien.ac.at/oldmmlab/aims/UniversalIntelligence.pdf
- Hernández-Orallo 的 AI 评估综述：https://arxiv.org/abs/1611.04206

---

# On the Measure of Intelligence — François Chollet (2019) 详解

## 论文的核心 thesis

这篇论文是 François Chollet (Keras creator) 在 2019 年发表的一篇 long-form position paper,核心论点可以浓缩为一句话:**intelligence 是 skill-acquisition efficiency,而 skill 本身。** 当前 AI community 把 "benchmark skill at specific tasks" (chess, Go, StarCraft, DotA2) 当作迈向 general AI 的 progress metric,这在 Chollet 看来是 conceptual mistake,因为 skill 可以通过 unlimited priors 或 unlimited training data 被 "买来",完全 bypass 掉 generalization power。

论文分三部分:
1. **Part I**: 历史 context,两种 intelligence 观 (skill collection vs general learning ability),generalization spectrum
2. **Part II**: 基于 Algorithmic Information Theory (AIT) 的 formal definition of intelligence
3. **Part III**: ARC (Abstraction and Reasoning Corpus) benchmark 的设计

参考链接:
- 论文 PDF: https://arxiv.org/abs/1911.01547
- ARC dataset: https://github.com/fchollet/ARC
- ARC prize (2020+): https://www.arcprize.org/

---

## Part I: 历史背景与两种 intelligence 观

### I.2 两种 divergent visions

Chollet 指出,所有 intelligence 定义基本都隐含两种取向之一:

**Vision A: Intelligence as collection of task-specific skills** (Minsky, evolutionary psychology)
- 心智是 evolution 产出的 special-purpose mechanisms 的集合
- Minsky 1968: "AI is the science of making machines capable of performing tasks that would require intelligence if done by humans"
- 这种 view 导致 AI field 几十年专注于 narrow task performance

**Vision B: Intelligence as general learning ability** (Turing, McCarthy, Locke's Tabula Rasa)
- 心智是 blank slate,能从 experience 中 acquire arbitrary skills
- McCarthy (paraphrased): "AI is the science and engineering of making machines do tasks they have never seen and have not been prepared for beforehand"
- 这是 connectionism / Deep Learning 的 implicit philosophy

Chollet 认为这两种 view 都是错的 (II.1.3),正确答案是 **innate priors + experience** 共同构成 generalization 的来源 —— 对应 Cattell 的 fluid intelligence (Gf) vs crystallized intelligence (Gc) 二分。

### I.3.2 Generalization Spectrum

Chollet 提出一个 generalization 的层级谱系,这是论文中最有 intuition-building 价值的概念框架之一:

| 层级 | 名称 | 描述 | 例子 |
|------|------|------|------|
| 0 | No generalization | 无 uncertainty,program 可被证明 correct over all inputs | Tic-tac-toe solver, sorting algorithm |
| 1 | Local generalization ("robustness") | 处理同一 task 的 known distribution 中的新 sample | Image classifier 对 unseen cat images |
| 2 | Broad generalization ("flexibility") | 处理 broad category of tasks,包含 creator 未预见的 situations | L5 self-driving, Wozniak's coffee cup test |
| 3 | Extreme generalization ("generality") | open-ended 处理只与过去 experience 有 abstract commonalities 的新 tasks | Human cognition |
| 4 | Universality (theoretical) | 任何 universe 中可实践的 task | No Free Lunch theorem 说不可能 |

这个谱系与 psychometrics 中的 CHC theory 的三层级 hierarchy 对应:
- Top: g factor (general intelligence) ↔ extreme generalization
- Middle: broad cognitive abilities ↔ broad generalization
- Bottom: task-specific skills ↔ local generalization

**关键 insight**: AI history 是一个沿着这个 spectrum 缓慢攀升的过程。Symbolic AI (no generalization) → Machine Learning (local generalization) → 当前尝试 broad generalization。Deep Learning 本质上是 local generalization system,conceptually similar to a locality-sensitive hashtable (II.1.1)。

### I.3.4 Psychometrics 原则

Chollet 从 psychometrics 借用了几个核心原则:
1. **Measure abilities, skills**: abilities 是 broad generalization 的基础,skills 是 abilities 的 crystallized output
2. **Use batteries of tasks, single task**: 必须使用多个 task,且 tasks 必须对 test-taker 和 developer 都 unknown
3. **Reliability**: 可复现
4. **Validity**: 测试的东西必须被清楚理解,且能 predict 其他能力
5. **Standardization**: 共享 benchmark
6. **Freedom from bias**: 不能对某类 test-taker 系统性不利

---

## Part II: Intelligence 的 Formal Definition

### II.1.1 Skill ≠ Intelligence (核心 argument)

这是论文最 critical 的部分。Chollet 指出,可以通过两种方式 "buy" skill without generalization:

**方式 1: Unlimited priors** — engineer 直接 hard-code solution (像 Evans 1960s 的 ANALOGY program 解决 IQ test 的 geometric analogy)

**方式 2: Unlimited training data** — 即使一个 locality-sensitive hashtable (只有 trace generalization power) 也能 solve 任何 task,只要能 dense-sample situation space

具体例子:一个 nearest-neighbor hashtable + 足够多的 training data 可以 "solve" 任何 video game。OpenAI Five 训练了 45,000 years of play,但被非冠军人类玩家几天就找到 exploit 击败;它甚至只能玩 16 个角色而非 100+。AlphaGo/AlphaZero 至今没有在 board games 之外找到应用。

**核心 takeaway**: "Solving" 任何 task with beyond-human performance by leveraging unlimited priors or data 不带来 broad AI 或 general AI 的任何进展。

### II.1.2 The g factor is scoped, universal

Chollet 反对 "Artificial General Intelligence" 作为 universal intelligence 的目标:
- No Free Lunch theorem: 任何两个 algorithms 在所有 possible problems 上 averaged 是 equivalent
- Human g factor 只是在 human-relevant scope 内 "general"
- 类比: human physical fitness 在 human morphology scope 内 "general" (能跑马拉松、攀岩、打篮球),但在 Venus surface 或 Jupiter atmosphere 完全不适用
- Human dimensional bias: 2D navigation / shape-packing 极强,3D 减弱,4D+ 完全无能
- Human 在 TSP (Traveling Salesman Problem) 小规模近最优,但 inverting goal to "longest path" 后表现比最简单 heuristic 还差 (Macgregor & Chu 2011)

**Conclusion**: AI 应该 explicitly target **human-like** intelligence,以 human intelligence 为 reference,benchmark progress against human intelligence。

### II.1.3 Core Knowledge priors

Developmental psychology (Spelke & Kinzler 2007) 提出 **Core Knowledge theory**: 人类 innate priors 分四类:

1. **Objectness & elementary physics**: cohesion (objects move as connected wholes), persistence (objects don't suddenly appear/disappear), contact (no action at a distance, no interpenetration)
2. **Agentness & goal-directedness**: 部分 objects 是 agents,有 intentions,efficiency in goal-directed actions,reciprocity
3. **Natural numbers & elementary arithmetic**: small number abstract representations,跨 modality,可 add/subtract/compare
4. **Elementary geometry & topology**: distance, orientation, in/out relationships, 2D/3D navigation

Chollet 论点: 测试 human-like general intelligence 必须只 rely on 这四类 priors,不依赖任何 acquired knowledge (语言、chess rules、arrows 等符号)。这是 ARC 设计的 foundation。

### II.2 形式化定义

#### II.2.1 Problem Setup

定义一个 task $T$ 由四个 objects 组成:
- `TaskState` (binary string)
- `SituationGen: TaskState → Situation` (可能 stochastic)
- `Scoring: [Situation, Response, TaskState] → [Score, Feedback]` (可能 stochastic)
- `TaskUpdate: [Response, TaskState] → TaskState` (可能 stochastic)

一个 intelligent system $IS$ 由三个 objects 组成:
- `ISState` (binary string)
- `SkillProgramGen: ISState → [SkillProgram, SPState]` (可能 stochastic)
- `ISUpdate: [Situation, Response, Feedback, ISState] → ISState` (可能 stochastic)

`SkillProgram: [Situation, SPState] → [Response, SPState]` 是 stateful function,代表某一时刻 frozen 的 task-specific capability。

交互分两 phase:
- **Training phase**: 反复 generate skill program, response, score, feedback,update IS state
- **Evaluation phase**: 用单一 fixed skill program 处理新 situations (没有 IS 参与)

这里的关键 conceptual device: 把 "agent" 拆成两部分 —— **intelligent system** (program synthesis engine,拥有 intelligence) 和 **skill program** (非智能的 output artifact)。Chollet 认为这个区分被 RL community 长期忽视。

#### 关键 definitions

- **Evaluation result**: 固定 skill program 在特定 evaluation phase instance 上的 score 之和
- **Skill**: evaluation results over all possible evaluation phase instances 的 probabilistic average
- **Optimal skill**: best possible skill program 能达到的最大 skill
- **Sufficient skill threshold** $\theta_T$: 主观的 "解决 task" 的 skill 水平
- **Task value** $\omega_T$: subjective value of achieving sufficient skill at T,用于跨 task 比较 skill (因不同 task 的 scoring function scale 不同)
- **Curriculum**: training phase 中 (situation, response, feedback) 的 sequence
- **Optimal curriculum**: 使 IS 产生最高 skill 的 curriculum
- **Task-specific potential** $\theta_{T,IS}^{max}$: IS 在 task T 上能产生的 best skill program 的 skill
- **Scope**: IS 能产生 sufficient solution 的所有 tasks (with $\omega > 0$) 的子空间
- **Potential**: scope 内所有 tasks 的 task-specific potential 值集合

#### II.2.1 Generalization Difficulty (核心公式)

利用 Algorithmic Information Theory (AIT) 的 **Kolmogorov Complexity** $H(s)$ = 输出 string $s$ 的最短 program 长度。

**Relative Algorithmic Complexity** $H(s_1 | s_2)$ = 以 $s_2$ 为输入产生 $s_1$ 的最短 program 长度 ($s_2$ 本身的长度不计)。

**Generalization Difficulty** (system-centric):

$$GD_{T,C}^{\theta} = \frac{H(Sol_T^{\theta} | TrainSol_{T,C}^{opt})}{H(Sol_T^{\theta})}$$

变量说明:
- $T$: task
- $C$: curriculum
- $\theta$: skill threshold (上标)
- $Sol_T^{\theta}$: task T 上达到 skill ≥ θ 的最短 skill program
- $TrainSol_{T,C}^{opt}$: 给定 curriculum C 下,达到 optimal training-time performance 的最短 program
- $H(Sol_T^{\theta})$: $Sol_T^{\theta}$ 的 Kolmogorov complexity
- $H(Sol_T^{\theta} | TrainSol_{T,C}^{opt})$: 以 $TrainSol_{T,C}^{opt}$ 为输入产生 $Sol_T^{\theta}$ 的最短 program 长度

**Intuition**: GD 衡量 "从最简 training-time solution 修改到 evaluation-time sufficient solution 需要多少额外信息"。如果 $TrainSol$ 直接 generalize 到 evaluation,GD=0 (无 generalization needed)。GD ∈ [0, 1] by construction (因 $H(s|t) \leq H(s)$)。

**反直觉点**: Occam's razor 通常说最简 training solution 也 generalize 最好。但 Chollet 指出 generalization 是 **处理未来 uncertainty 的能力**,与 compression past data 是 antagonistic 的。最简 training solution 可能丢弃 "看似无用但未来有用" 的信息。

例子: 训练点 $(x=-0.75, F), (x=0.15, T)$。最简 training solution: $\lambda(x): x>0$ 或 $\lambda(x): \text{bool}(\text{ceil}(x))$。但 test 点 $(x=-0.1, T)$ 会 fail。Nearest-neighbor (存所有点) 更 "冗长" 但 prepared for future uncertainty。

**Developer-aware Generalization Difficulty**:

$$GD_{IS,T,C}^{\theta} = \frac{H(Sol_T^{\theta} | TrainSol_{T,C}^{opt}, IS_{t=0})}{H(Sol_T^{\theta})}$$

其中 $IS_{t=0} = \{SkillProgramGen, ISUpdate, isState_{t=0}\}$ (IS 的初始状态,包括 built-in priors)。

这 capture 了 "developer 注入的 priors 能多大程度 reduce evaluation-time solution 的 description length"。

#### Priors 的形式化

$$P_{IS,T}^{\theta} = \frac{H(Sol_T^{\theta}) - H(Sol_T^{\theta} | IS_{t=0})}{H(Sol_T^{\theta})}$$

**Intuition**: Priors 是 "初始 IS 距离 sufficient solution 有多近",即嵌入在 IS 中的 relevant information 量。注意不是 IS 总信息量 (那会是 $H(IS_{t=0})$),而是 **与 task 相关的** 部分 —— 大 system with irrelevant knowledge 只被 minimally penalize (仅 indexing/retrieval overhead)。

#### Experience 的形式化

**Step t 的 experience**:

$$E_{IS,T,t}^{\theta} = H(Sol_T^{\theta} | IS_t) - H(Sol_T^{\theta} | IS_t, data_t)$$

其中:
- $IS_t = \{SkillProgramGen, ISUpdate, isState_t\}$
- $data_t = \{Situation_t, response_t, feedback_t\}$

**Intuition**: step t 的 experience = "step t 的 data 能 reduce 多少关于 solution 的 uncertainty" (假设 IS optimally intelligent)。

**Total experience over curriculum C**:

$$E_{IS,T,C}^{\theta} = \frac{1}{H(Sol_T^{\theta})} \sum_t E_{IS,T,t}^{\theta}$$

关键设计点:
- **只计 relevant information**: noisy curriculum 不被惩罚
- **只计 novel information**: repetitive curriculum 不被惩罚 fast learner (但惩罚 slow learner 需要更多 repetition)
- **Eager sum vs global pooling**: 区分 fast/slow learner

#### Intelligence 的最终定义

**Sufficient case**:

$$I_{IS,scope}^{\theta_T} = \underset{T \in scope}{Avg} \left[ \omega_T \cdot \theta_T \underset{C \in Cur_T^{\theta_T}}{\Sigma} \left[ P_C \cdot \frac{GD_{IS,T,C}^{\theta_T}}{P_{IS,T}^{\theta_T} + E_{IS,T,C}^{\theta_T}} \right] \right]$$

**Optimal case**:

$$I_{IS,scope}^{opt} = \underset{T \in scope}{Avg} \left[ \omega_{T,\Theta} \cdot \Theta \underset{C \in Cur_T^{opt}}{\Sigma} \left[ P_C \cdot \frac{GD_{IS,T,C}^{\Theta}}{P_{IS,T}^{\Theta} + E_{IS,T,C}^{\Theta}} \right] \right]$$

变量说明:
- $scope$: IS 的 task scope
- $\omega_T$: task T 的 value (subjective,用于 cross-task 比较)
- $\theta_T$: task T 的 sufficient skill threshold
- $\Theta = \theta_{T,IS}^{max}$: IS 在 task T 上的 potential (max achievable skill)
- $\omega_{T,\Theta}$: achieving potential 的 value
- $Cur_T^{\theta_T}$: 导致 IS 产生 sufficient solution 的 curriculum 空间
- $Cur_T^{opt}$: 导致 IS 产生 highest-skill solution 的 curriculum 空间
- $P_C$: curriculum C 的概率
- $GD_{IS,T,C}^{\theta_T}$: developer-aware generalization difficulty
- $P_{IS,T}^{\theta_T}$: priors
- $E_{IS,T,C}^{\theta_T}$: experience

**Schematic decomposition**:

$$\text{Contribution per task} = \underbrace{\omega_T \cdot \theta_T}_{\text{value-weighted skill}} \times \underbrace{\mathbb{E}_{C \sim Cur}\left[\frac{GD}{P + E}\right]}_{\text{skill-acquisition efficiency}}$$

**核心 intuition**: 
- High intelligence = high-skill solutions for **high-GD tasks** (high uncertainty) using **little priors + experience**
- Intelligence 是 information → future situation space coverage 的 conversion rate
- High skill high intelligence: 完全不同的 concepts
- 必须有 learning/adaptation: 若 IS 初始就能 perform well,GD 低,intelligence 低
- Curve-fitting intelligence: 只产生最简 consistent program 的系统只能在 GD=0 的 task 上得分

#### II.2.2 其他 efficiency dimensions

Chollet 指出 information efficiency 之外还有:
- **Computation efficiency** (skill program 推理 cost / IS training cost)
- **Time efficiency** (latency)
- **Energy efficiency** (生物系统相关)
- **Risk efficiency** (curriculum 过程对 IS 的 risk,生物系统相关)

这些可作为 regularization term 加入 intelligence formula。

#### II.2.3 Practical implications

对研究方向的 implications:
1. **Intelligence 可作为 objective function** (computable approximation)
2. **鼓励 broad abilities 而非 skill**
3. **鼓励 program synthesis** (分离 IS 和 skill program)
4. **鼓励 curriculum development** (optimal curriculum 提升 expressed intelligence)
5. **鼓励 human-like knowledge priors** (Core Knowledge)

对 evaluation 的 implications:
- 量化 generalization difficulty 可 weed out "zero GD" tests
- AI-human comparison 需要 shared scope, fixed skill threshold, comparable priors
- 必须考虑 GD 防止 shortcut (e.g. CV 中 texture vs semantics)

Characterizing 一个 IS 需回答:
1. Scope 是什么?
2. Scope 内 potential 是多少?
3. Priors 是什么?
4. Skill-acquisition efficiency (intelligence) 是多少?
5. 什么 curriculum 最大化 skill / efficiency?

---

## Part III: ARC Benchmark

### III.1.1 ARC 是什么

ARC (Abstraction and Reasoning Corpus) 类似 Raven's Progressive Matrices 的格式:
- **Training set**: 400 tasks
- **Evaluation set**: 600 tasks (400 public + 200 private)
- 每个 task: ~3.3 demonstration examples + 少量 test examples (通常 1,偶尔 2-3)
- 每个 example: input grid + output grid
- Grid: 1×1 到 30×30,10 种 colors/symbols
- **Binary success**: 必须 exact correct on all test examples
- 每个 test example 允许 3 trials,只给 binary feedback
- **Score**: 解决的 evaluation tasks 比例

**关键约束**:
- Evaluation tasks 对 test-taker 和 developer 都 unknown (developer-aware generalization)
- Private evaluation set 在 competition 中严格 enforce 这一点
- Test-taker 必须从 scratch 构造 output grid (决定 height, width, 哪些 symbol 放哪里)

### III.1.2 Core Knowledge Priors in ARC

ARC 显式列出假设的 priors:

**a. Objectness priors**:
- Object cohesion: 通过 color continuity 或 spatial contiguity 解析 objects
- Object persistence: objects 在 noise 或 occlusion 下 persist; input objects 常在 output 中 transformed 出现
- Object influence via contact: 物理 contact (e.g. 一个 object 平移直到 contact 另一个,line "growing" 直到 "rebounds")

**b. Goal-directedness prior**:
- 许多 input/output 可被 modeled 为 intentional process 的 start/end states (虽然 ARC 无 time 概念)

**c. Numbers & counting priors**:
- Counting, sorting (by size), comparing numbers (which appears most/least/equal)
- Addition/subtraction (Core Knowledge number system)
- 量 < ~10

**d. Basic geometry & topology priors**:
- Lines, rectangles (regular shapes 更常见)
- Symmetries, rotations, translations
- Up/downscaling, elastic distortions
- Containing / inside / outside
- Drawing lines, connecting points, orthogonal projections
- Copying, repeating objects

### III.1.3 与 psychometric tests 的区别

1. **fluid intelligence crystallized**: ARC 只测 reasoning/abstraction,不涉及 language/real-world objects
2. **Tasks unknown to developers**: 防止 hard-coding solutions (private set enforce)
3. **Greater task diversity**: 数百 unique tasks,降低 hard-coding 实用性
4. **Manual generation programmatic**: 避免逆向 master program (与 C-Test 区别)

### III.1.4 ARC solver 的可能形态

Chollet 把 ARC 视为 **program synthesis benchmark**。假想 solver:

1. **Develop DSL**: 能表达任何 ARC task 的 solution program。需 hard-code Core Knowledge priors 作为 basis functions,组成 "human-like reasoning DSL"。Chollet 认为这是 general AI progress 的 critical subproblem。
2. **Generate candidates**: 用 DSL 生成把 input grids 映射到 output grids 的 candidate programs,reuse/recombine 之前有用的 subprograms
3. **Select top candidates**: 基于 simplicity 或 likelihood (可在 training set 上训练)。注意:最简 training-consistent program 不一定 generalize (前述 GD argument)
4. **Use top 3 for test**: 生成 test outputs

**Claim**: 若存在 human-level ARC solver,意味着能仅通过少量 demonstrations "program an AI" 处理 wide range of human-relatable tasks。因 ARC solver 与 human intelligence 共享 priors,其 scope of application 应接近 human cognition,既 practically useful 又 easy to interact with。

Chollet 承认这是 speculative,可能像 Newell 1973 期待 chess 带来 broad cognitive progress 一样落空 —— "especially if ARC turns out to feature unforeseen vulnerabilities to unintelligent shortcuts"。

### III.2 Weaknesses

Chollet 自承 ARC 的局限:
1. **GD 未量化**: 没有给出 evaluation set 相对 training set 的 quantitative GD,计划用 human performance 估计
2. **Validity 未建立**: 需 large-sample human studies 验证 predictiveness
3. **Dataset size/diversity 可能有限**: 1000 tasks,可能 conceptual overlap,可能 vulnerable to shortcuts。计划通过 public competition crowd-source 攻击
4. **Closed-ended binary format**: 0/1 scoring 缺 granularity。提出改进: 让 test-taker 动态 request new test inputs,反复 propose solution + receive feedback,直到 reliably correct,score = 所需 feedback 量 (直接对应 II.2 的 intelligence definition,curriculum 由 input generator 控制)
5. **Core Knowledge 可能理解不全**: innate priors 的确切本质仍是 open problem

### III.3 Alternatives

1. **Repurposing skill benchmarks**: 不用新 levels (local generalization),而是 alternative games $X_1, ..., X_n$ with meaningful GD over X。例如 DotA2 AI 测 League of Legends / Heroes of the Storm 的 learning efficiency;或 16 characters 训练后测新 characters 的 first-try performance
2. **Open-ended teacher-student**: ever-learning "teacher" program 生成 tasks,优化 novelty/interestingness for "student" programs。Teacher 从 external source (real world) 汲取 incompressible complexity 保证 open-ended。类似 POET (Wang et al. 2019) 和 anytime intelligence test (Hernández-Orallo & Dowe 2010)

---

## Critique 与后续影响

### 论文的 strengths
1. **Conceptual clarity**: 把 "skill vs intelligence" 的区分讲得非常 precise,通过 AIT 形式化
2. **Historical context**: 两种 intelligence 观的历史溯源 (Minsky vs Turing/McCarthy) 非常 illuminating
3. **Priors/experience/GD 三角形**: 揭示了 "buy skill" 的两种方式,解释了为什么 unlimited data/priors 不带来 generalization
4. **ARC design**: 显式 priors + developer-aware generalization + Core Knowledge 是 concrete 且 actionable 的

### 论文的 weaknesses / 可质疑之处
1. **AIT 公式不可计算**: Kolmogorov complexity $H(s)$ 不可计算,所有公式都是 conceptual device,实际 benchmark (ARC) 完全没用这些公式。Formalism 和 benchmark 之间存在 gap。
2. **Curriculum space $Cur_T^{\theta_T}$ 概念模糊**: 什么是 "所有导致 sufficient solution 的 curriculum"?实践中如何 enumerate / sample?
3. **Core Knowledge 是否足够**: Spelke 的 four systems 远不能 capture human innate cognitive structure。ARC 的 priors 列表实际是 Chollet 的 interpretation,可能过窄或过宽。
4. **ARC 可能被 "narrow" solver 攻克**: 若有人构建足够丰富的 DSL + brute-force search,可能不需 "true intelligence" 就解决 ARC。2020-2024 的 ARC Prize 竞赛结果显示,top solutions 多是 hand-crafted DSL + search,而非 learning-based generalization。这某种程度上验证了 Chollet 的 program synthesis framing,但也暗示 ARC 可能不是 "intelligence" 的充分 measure。
5. **Developer-aware generalization 难以严格 enforce**: 即使有 private set,developer 可通过 public set 间接获取 priors (e.g. 训练 DSL on public tasks)。
6. **Human scope 的 anthropocentrism**: Chollet argue 这是必要且 legitimate,但可能限制了对 non-human-like intelligence 形式的探索。

### 后续发展 (post-2019)
- **ARC Prize**: 2020 起,Kaggle hosted ARC competition。2023-2024 升级为 ARC Prize (https://arcprze.org),奖金池 $1M+
- **ARC-AGI / ARC-2**: 2024 年 Chollet 与 Mike Knoop 推出 ARC-AGI benchmark,半-private evaluation
- **LLM performance**: GPT-4o, Claude 3.5 Sonnet 等前沿 LLM 在 ARC 上表现仍远低于人类 (典型 ~10-30% vs human ~85%+),验证了 Chollet 关于 Deep Learning 缺乏 broad generalization 的论点
- **Neuro-symbolic approaches**: ARC 推动了 neuro-symbolic / program synthesis 方向的研究,如 DreamCoder (Ellis et al. 2021)
- **Critiques**: LeCun 等人批评 ARC 过于侧重 "puzzles",可能不代表 real-world intelligence 的核心

参考:
- ARC Prize: https://www.arcprize.org/
- ARC-AGI paper (Chollet & Knoop 2024): https://arcprize.org/blog
- DreamCoder: https://arxiv.org/abs/2006.08381
- Spelke Core Knowledge: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(07)00006-9

---

## Intuition summary

Chollet 的核心 message 用一段话浓缩:

> Deep Learning 在 ImageNet, Go, StarCraft 上的 success 让人产生 "向 general AI 迈进" 的错觉,但这些本质上是用 unlimited data 在 fixed task 上做 local generalization,等价于 high-capacity hashtable。True intelligence 是 **从有限 experience + priors 高效 acquire 新 skill 的能力**,在 face of uncertainty 时 cover 尽量大 future situation space。衡量 intelligence 必须控制 priors 和 experience,并量化 generalization difficulty —— 否则我们只是在 benchmark 谁有更多 compute/data,而非谁更 intelligent。ARC 是一次 attempt: 显式 Core Knowledge priors, novel evaluation tasks, few-shot format, forcing test-taker 展示 broad generalization 而非 memorization。

这个 perspective shift 的价值在于: 它让 "我们离 general AI 多远" 这个问题变得 **可量化** —— 答案是 "ARC 上 human-level performance with human-level experience",而 "击败 Go world champion" 这种 milestone 实际上 **没有任何 information** about general AI progress。
