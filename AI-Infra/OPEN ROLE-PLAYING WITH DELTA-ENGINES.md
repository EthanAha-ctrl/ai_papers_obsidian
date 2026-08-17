---
source_pdf: OPEN ROLE-PLAYING WITH DELTA-ENGINES.pdf
paper_sha256: d3229fb50d93e97e7d55cd851478ebc194e0145c9f876aba1bc9a3811e175ffc
processed_at: '2026-08-06T00:00:05-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

咱们抛开那些学术套话，用最直白的话来聊聊这篇 paper 到底在搞什么飞机。

### 1. 痛点：游戏设计师的牢笼

你玩传统的 RPG 游戏，比如《Assassin’s Creed》或者《Pokemon》，你以为你在扮演角色，其实你只是在走 designer 画好的迷宫。你想让角色学个新招？只能在系统给你列好的技能树里挑。哪怕你智商 200，想出一个系统根本没预设的招式，游戏就给你报错或者没反应。

这篇 paper 的作者觉得这太没劲了。他的 vision 是：玩家出生时大家都差不多，就像一张白纸，但你通过自然语言告诉游戏“我想干嘛”，游戏就能让你的角色真正长成你想要的样子。比如你说 “Let me learn a talent to burn the enemy”，系统就该听懂，并给你凭空造出一个喷火技能。这就是所谓的 Open Role-Playing Games (ORPGs)。

### 2. 核心魔法：Delta-Engine

这怎么实现呢？传统游戏引擎（像 Unity）是听不懂人话的。作者搞了个叫 **Delta-Engine** 的玩意儿。你可以把它想成一个**随时在 git commit 的活代码库**。

它有两部分：
- **Base Engine**：角色刚出生时的底层代码，可能就几行，只会走和跑。
- **Neural Proxy**：就是一个 LLM（大语言模型），在这篇 paper 里用的是 fine-tuned 的 CodeGemma。

当你输入一句自然语言指令 $x_t$，比如“我要学个大威天龙”，Neural Proxy 这个 LLM 就负责翻译，吐出一段 Python 代码 $\Delta y_t$。这段代码就像个补丁，直接 merge 进角色现有的代码 $y_{t-1}$ 里。角色就这么 evolve 了。

这在思路上跟 Voyager (https://arxiv.org/abs/2305.16291) 在 Minecraft 里写 skill library 很像，把世界状态变成可执行的代码增量。

### 3. 解决失忆：Retrieval

但这里有个大麻烦。角色长到 30 级，技能多了，代码可能 5000 tokens 了。LLM 的 context window 就那么大（CodeGemma 是 8192），塞满了老代码，它就没空间思考新代码怎么写了。

怎么办？**Retrieval**。
论文里给的核心公式其实概念很简单：
$$\Delta y_t = \mathcal{F}(\tilde{y}_{t-1}, x_t)$$
变量意思：
- $\mathcal{F}$ 是 LLM
- $x_t$ 是你的指令
- $\tilde{y}_{t-1}$ 是**只挑出来的相关老代码**，不是全部代码

LLM 先看一眼角色的代码目录（只有方法名，没有具体实现），自己判断“这次学新技能跟哪些老方法有关”，然后把那几个老方法的实现抽出来当参考，再写新代码。这就像程序员改 bug 不会把整个项目代码全看一遍，而是先 grep 一下相关的函数。这招让角色能顺利演化到 30 步不崩。

### 4. 教 AI 写代码：人机结对编程

最大的工程难点其实在于数据。你要教 LLM 听懂各种离谱的指令并写出正确的代码，就需要很多训练数据。但让 AI 自己瞎生成数据（Synthetic Data），会有两个致命问题：
1. **不 Novel**：AI 翻来覆去就是那几招，给个狗它能改成大狗，给个火球它能改成大火球，想不出新机制。
2. **不 Interesting**：生成的东西很无聊。

作者搞了个很聪明的 **Human-AI Co-design** 流程：
- **借原型（Prototypes Enhanced Imagination）**：人类设计师去 Wikipedia 抄一段霸王龙（Tyrannosaurus）的描述，喂给 LLM 说“照着这个设计个宝可梦”。LLM 就能设计出咬合力超强的角色，这比让 AI 凭空想象靠谱多了。
- **打标签（Tags of Interest, ToI）**：怎么判断生成的角色好不好玩？用一套规则去扫代码。如果代码里重载了 `get_power` 方法，说明它能强化攻击，就打个 Interesting 标签。标签太少就扔掉。
- **人机循环**：LLM 生成代码，规则过滤一遍，人类设计师最后把观再检查一遍，合格的数据再丢回库里当下一轮的参考。

### 5. 结果说话

实验结果非常能说明问题。如果只用纯 AI 生成的数据训练模型，遇到玩家自己设计的复杂角色（Hard Test），代码的准确率（Acc%）只有可怜的 58.6%。但是用这种 Human-AI Co-design 生成的数据，准确率能干到 83.9%。加上 Retrieval 机制，能进一步推到 89.7%。

这说明什么？**纯合成数据会让模型越来越蠢（Model Collapse，参考 https://arxiv.org/abs/2305.17493），必须有人类的火花在里面才能跳出分布。**

### 6. 我的延伸联想

这篇 paper 其实是在下一个很大的棋。它把游戏世界变成了一个 **Neurosymbolic** 的结构：神经网络负责理解意图和写代码，符号系统（Python 代码）负责维持状态和执行逻辑。

这跟 GameNGen (https://arxiv.org/abs/2408.14837) 用 Diffusion model 直接生成像素画面走的是完全相反的路。GameNGen 是黑盒，你没法在中间插一个自定义技能进去；Delta-Engine 是白盒，你可以直接 git diff 看看你的角色长了什么新方法。

我甚至觉得这不仅仅是个游戏引擎，这是一种全新的**软件运行时形态**。想象一下，未来的 App 可能不需要开发者写死功能，用户说一句“帮我加个自动回复功能”，LLM 就在后台写好代码 merge 进去了。这其实跟 Toolformer (https://arxiv.org/abs/2302.04761) 甚至 OpenAI 的 Code Interpreter 是一个家族的，只是 Delta-Engine 把生成的代码**持久化**了，成为了系统记忆的一部分。

当然，Safety 是个巨大的大坑。玩家如果说“让我的角色学会删库技能”，系统要是真执行了那就好玩了。这块 paper 只是提了一嘴没展开，但这绝对是未来这种架构能不能落地的生死线。

---

# Delta-Engine: 让游戏世界通过自然语言演化

这篇文章让我想到一个很深的 idea：把 game state 当作一个**可演化的代码库**，而不是一个静态的状态机。每次玩家用自然语言表达意图，neural proxy（LLM）就生成一段**增量代码** patch，merge 进 base engine。这个 metaphor 把游戏的角色演化做成了 software engineering 的一种 runtime 形式 —— 每一个 pokemon 角色都是一份不断 git commit 的代码。

---

## 1. 核心直觉：ORPG 与 Delta-Engine

### 1.1 ORPG 的设计哲学

ORPG（Open Role-Playing Games）瞄准的是一个很 Karpathy-style 的洞见：**人在出生时几乎相同，正是后续分化的选择让我们变得独特**。传统 RPG 把角色锁死在预先写好的分支树里，玩家只能在 designer 预设的"组合空间"里挑。ORPG 想做的事情是把"组合空间"换成"自然语言空间"—— 玩家说 "Let me learn a talent to burn the enemy"，系统就该把这变成一段新的、能跑的代码注入角色。

这背后是两种世界观的对立：

- **closed-world RPG**：designer 穷举可能性，玩家选择 = 选择
- **open-world RPG**（这里说的 ORPG）：designer 提供基础约束（物理、世界观），player **生成**新的可能性

这让我立刻联想到 **Voyager (Wang et al., 2023a)** 在 Minecraft 里做的事情 —— Voyager 把 LLM 写出的 skill 累积进 skill library，未来调用时检索复用。Voyager 的 skill library 本质就是一个 delta-engine，只是 Minecraft 那边 skill 是顶层调用，而这里 delta-engine 是把"代码增量"直接打进角色本身的对象方法。参考：https://arxiv.org/abs/2305.16291

### 1.2 Delta-Engine 的两个组件

Delta-Engine 与 GameNGen（Valevski et al., 2024, https://arxiv.org/abs/2408.14837）走的是两条完全不同的路：

| 维度 | GameNGen | Delta-Engine |
|------|----------|--------------|
| 渲染主体 | diffusion model 直接预测下一帧 pixel | backbone engine 仍然用代码渲染 |
| LLM 角色 | 学习一个 policy→frames 的端到端映射 | 学习 instruction→code patch 的映射 |
| 可解释性 | 黑盒（latent + pixels） | 白盒（可读的 Python） |
| Modularity | 整个游戏被压缩进一个网络 | base engine + neural proxy 解耦 |
| 可编辑性 | 几乎为零 | 可以直接 git diff 看角色长出了什么 |

Delta-Engine 的精彩之处在于它选择了 **neurosymbolic** 的姿态：神经网络负责"翻译意图到代码"，符号引擎（Python 对象）负责"维持状态、执行逻辑"。这与 **Genie (Bruce et al., 2024, https://openreview.net/forum?id=bJbSbJskOS)** 那种 latent world model 也不同 —— Genie 学的是隐式的 transition，Delta-Engine 把 transition 显式地写成代码。

---

## 2. 数学形式化：增量预测与 Retrieval

### 2.1 增量预测公式

**公式 (1)**：

$$\Delta y_t = \mathcal{F}(y_{t-1}, x_t)$$

变量解读：

- $\mathcal{F}$：neural proxy，论文里是 fine-tuned **CodeGemma-7b**（Mesnard et al., 2024, https://arxiv.org/abs/2403.08295）
- $x_t \in \mathcal{X}$：第 $t$ 步的 natural language 指令，例如 "learn a move Rayquazalize that switches types and protects next turn"
- $y_{t-1} \in \mathcal{Y}$：当前 engine state，是一份 Python class 的 source code（在 paper 里是 `GreenBug(PokemonBase)` 这个 subclass）
- $\Delta y_t$：增量代码 patch，通常是几个新方法（用 `@Increment` decorator 标记）

初始条件 $y_0$ 就是 base engine，对 Free Pokemon 来说是 `PokemonBase` 加上初始的两个 move（`Tackle`, `Lundge`）。

**公式 (2)**：

$$y_t = m(\Delta y_t, y_{t-1})$$

- $m$：merge function。实现上就是 Python 的 method binding：通过 `@Increment` decorator 把新方法附加到 subclass，已有的方法可以被新方法 overload（类似 Python 的 method resolution order）。

### 2.2 为什么需要 Retrieval —— 公式 (3)

**公式 (3)**：

$$\Delta y_t = \mathcal{F}(\tilde{y}_{t-1}, x_t)$$

- $\tilde{y}_{t-1} \subset y_{t-1}$：sparse 版本的 engine state，只包含**与本次演化相关的方法实现**

为什么必须 retrieval？想象一个角色演化了 30 次，codebase 已经 5000+ tokens，CodeGemma 的 context window 是 8192 tokens（https://arxiv.org/abs/2403.08295），把整个 engine 塞进去就要 60%+ 的预算，剩下给 instruction 和 generation 的空间被严重压缩，长程 evolution 必然崩。

这个 idea 和 **MemGPT / LongMem**（https://arxiv.org/abs/2310.08560）的思路同源 —— 把长 context 的问题转化为"按需检索 + 局部生成"的问题。但 Delta-Engine 的 retrieval 是 **structural**：它不是嵌入向量相似度检索，而是 **LLM 自己读 skeleton overview 决定要哪几个方法**。

具体流程（参考 Figure 2）：

1. 第一步 prompt：把 engine 的结构骨架（只保留 method name，去掉 implementation）给 LLM，让它输出**需要 retrieve 的 method 名列表**
2. 第二步 prompt：把那些方法的实际 implementation 抽出来，和 instruction 一起塞给 LLM，生成 $\Delta y_t$

这种"LLM 自己决定 retrieve 什么"的设计，和 **Self-Ask / Self-RAG (Yu et al., 2023, https://openreview.net/forum?id=fB0hRu9GZUS)** 是一个 family。它避免了引入第二个 retriever 模型，单模型 end-to-end 完成"想清楚要什么 → 取过来 → 用它生成"。

---

## 3. Free Pokemon：一个具体的 playground

### 3.1 系统架构

Figure 1 展示了两个 engine：

- **Role Engine**（delta-engine）：每个 pokemon 角色一个，随 evolution 步骤扩展
- **Battle Engine**：host 多个角色之间的 battle，相对静态

初始化时，角色通过 JSON spec（species, types, stats）实例化为 `GreenBug(PokemonBase)`，自带 `Tackle` 和 `Lundge`。Figure 1 中的蓝色 stream 显示一次 evolution：玩家说 "learn Rayquazalize"，role engine 触发 scaling，生成两个新方法挂在 subclass 上。

这里有个非常软件工程化的味道 —— Pokemon 角色被建模为**类继承 + 方法重载**的演化树。每一次 evolution 就是一次 OOP 的增量建模。这和 **Program Synthesis as Generative Method (Butler et al., 2017, https://doi.org/10.1145/3102071.3102076)** 的思路遥相呼应 —— 把 game content generation 当作程序合成。

### 3.2 为什么 Pokemon 是好 playground

Pokemon 的几个特性让它特别适合 ORPG 实验：

1. **离散属性空间**：types、stats、moves 都是结构化的，容易 JSON 化
2. **战斗有明确规则**：battle engine 可以独立演化，role engine 只负责角色本身
3. **大众文化**：研究者、志愿者、评测者都对它有 ground truth intuition
4. **既有 baseline 可比**：Pokellmon (Hu et al., 2024, https://arxiv.org/abs/2402.01118) 已经做过 LLM 玩 official Pokemon，与这里形成正交对照 —— Pokellmon 是"用 LLM 玩官方 Pokemon"，Free Pokemon 是"让玩家造自己的 Pokemon"

---

## 4. 数据生成：人机协同的精髓

这一节我觉得是 paper 最有工程价值的部分。它把 LLM 数据合成的两个老问题（novelty、interestingness）用很工程化的方式解决了。

### 4.1 两个核心需求

**Being Novel**：LLM 在合成数据时容易陷入"线性组合"陷阱 —— 给它两个 talent，它就 merge 出一个新 talent；给它一只狗，它就给你一只更大的狗。这是 **Curse of Recursion (Shumailov et al., 2023, https://arxiv.org/abs/2305.17493)** 的另一种表现 —— 合成数据失去 OOD 信号，model collapse。

**Being Interesting**：interestingness 是个高度主观、无精确定义的概念。Nelson & Mateas 2007 就指出过这个难题。

### 4.2 Prototypes Enhanced Imagination

关键 insight：**LLM 不会"凭空想象"，但可以用一个具体的 prototype 作为锚点跳出去**。

流程：
1. Human designer 选一个 prototype 实体，比如从 Wikipedia 抓一段 **Tyrannosaurus** 的描述
2. 把描述 + instruction 给 LLM，让它生成一个基于 Tyrannosaurus 特征（强咬合力）的 pokemon
3. prototype 也可以来自虚拟世界（如 Monster Hunter 里的超自然生物），novelty 更高但语法错误率也更高

这其实就是把 **retrieval-augmented generation** 用在数据生成阶段 —— 用 Wikipedia 当 external memory，给 LLM 注入它训练分布里可能没好好覆盖的概念。可以对比 **Self-Instruct (Wang et al., 2023c, https://aclanthology.org/2023.acl-long.754)**，Self-Instruct 是 seed → expand，Delta-Engine 是 prototype → imagine，前者是数量扩张，后者是分布扩张。

### 4.3 Tags of Interest (ToI)

把"interestingness"近似为 **可被一组离散 tag 命中的事件累积**（Althöfer 2010 的 idea）。

形式化：

- 定义一组 ToI 标签 $\{T_1, T_2, \ldots, T_K\}$，每个标签是一个 binary 维度
- 给定一个 role code，用 **rule-based tagger** 扫描代码，得到一个 $K$-维 0/1 向量 $\mathbf{v} \in \{0,1\}^K$，称为 **interestingness vector**
- 例如：如果 role 重载了 `get_power` 方法，说明它能 boost power，对应 bit 设 1
- 设阈值 $\theta$，如果 $\|\mathbf{v}\|_1 < \theta$（命中 tag 数太少），样本被丢弃

这是一个非常 **mechanistic** 的 interestingness 度量 —— 不去度量"主观感受"，而是度量"代码层面触发了几种有趣机制"。这和 Todd et al. 2024 的 GAVEL（https://arxiv.org/abs/2407.09388）思路接近 —— 用可执行结构作为创意的 proxy。

### 4.4 Human-AI Co-design 的完整 loop

参考 Figure 3：

1. 初始化 sampling pool：**20 个手写 seed**（script-code pair）
2. Human designer 选 prototype，prompt LLM designer（GPT4 或 Claude3）生成 role script
3. LLM designer 再把 script 编程为 role code
   - 关键 trick：**script 生成阶段只用 1 个 in-context example**（避免 in-context bias 抑制 creativity）
   - **code 生成阶段用 5 个 in-context example**（编程要准确度，不要 creativity）
4. Evaluator 串联过滤：
   - Rule-based：编译失败 / 引入未调用方法 → 丢弃
   - ToI 阈值：interestingness vector 不达标 → 丢弃
   - Human：最后人工 check
5. 通过的样本回流到 sampling pool，准备下一轮

这个 loop 的精彩之处是 **Self-Improving** —— sampling pool 越来越大、越来越 diverse，后续 generation 能 in-context learn 到更宽的分布。

另一个我觉得很关键的 trick：**在 co-design 的中段，用训练好的 neural proxy 替换 GPT4/Claude3 作为 designer**。原因是 GPT4/Claude3 在这种窄域、高 precision 的代码任务上反而比 fine-tuned 小模型差。这呼应了 **Instruction-driven Game Engine (Wu et al., 2024b, https://arxiv.org/abs/2404.00276)** 的观察 —— 通用大模型对 instruction-engine 的 nuanced 要求适配性不够。

---

## 5. 实验数据深度解读

### 5.1 数据集统计（Table 1 上半部分）

| Statistic | Roles | Samples | #Evolves | #Length |
|-----------|-------|---------|----------|---------|
| SY.TRAIN | 167 | 500 | 3.0 | 1197.8 |
| CO.TRAIN | 175 | 502 | 2.9 | 1167.3 |
| EASY TEST | 19 | 43 | 2.3 | 997.2 |
| HARD TEST | 16 | 87 | 5.4 | 1841.6 |

几个观察：

- SY.TRAIN 和 CO.TRAIN 在量级上几乎对齐（500 vs 502 samples，平均 evolves 3.0 vs 2.9），控制变量做得好
- HARD TEST 的 #Evolves = 5.4，是训练数据的 ~2 倍；#Length = 1841.6，是训练的 1.58 倍 —— 这说明 hard test 真的在测**长程 scalability**
- EASY TEST 是 19 个 official pokemon 角色，分布与训练集近，预期是 sanity check

### 5.2 主要结果（Table 1 下半部分）

| Performance | Easy Exe% | Easy Acc% | Hard Exe% | Hard Acc% |
|-------------|-----------|-----------|-----------|-----------|
| CodeGemma w. SY. | 95.3 | 86.0 | 86.2 | 58.6 |
| CodeGemma w. CO. | ✓ | 95.3 | 90.8 | 83.9 |
| CodeGemma w. CO. w. RETR. | ✓ | ✓ | 92.0 | 89.7 |

我的解读：

1. **Easy test 上 SY 和 CO 接近**：分布相近时合成数据就够用，co-design 的 marginal 收益不明显（86.0 → 95.3 在 Acc 上）
2. **Hard test 是真正的区分器**：
   - SY 的 Acc 暴跌到 58.6，CO 提升到 83.9（**+25.3** 个百分点），co-design 的 OOD 信号价值在这里爆发
   - 加上 retrieval 再 +5.8 到 89.7
3. **Exe 与 Acc 的 gap**：Hard test 上 Exe 都很高（86~92），说明 model 学会了"生成能跑的代码"的格式先验，但"对的内容"很难。这正是 **format vs semantics gap**，常见于 code LLM 评测
4. **Retrieval 在 Easy 上没提升**：因为 Easy test 单个角色 evolves 少（2.3），context 还远没到上限，retrieval 是冗余的；Hard test 上 retrieval 真的救命

### 5.3 长程 scalability（Figure 4）

实验设计：随机从数据库抽 abilities/moves 反复 prompt，直到 model 给出 non-executable response。重复 100 次，画两个 histogram：以 evolution step 为 x 轴、以 engine size（tokens 数）为 x 轴。

关键现象：

- **无 retrieval**：~20 步、~5000 tokens 处性能衰减到一半
- **有 retrieval**：维持到 ~30 步

注意 5000 tokens 离 CodeGemma 的 8192 limit 还有距离，但性能已经崩 —— 这说明 **effective context** 远小于 nominal context。这是 LLM 长程理解的经典症状，参考 Longformer / LongNet 类工作（https://arxiv.org/abs/2208.10765）的讨论。Retrieval 通过缩窄 effective context 缓解这个问题。

Figure 4 右侧的 case 也很有意思 —— model 只 retrieve `type_change` 这一个方法，足以支撑"切换类型 + 抵挡下一击"的 evolution。这正是 incremental evolution 在自然界里"局部修改"的形态。

### 5.4 数据分布可视化（Figure 5）

用 sentence embedding + t-SNE 在两个空间投影：

- **Semantic space**：co-designed 几乎**包住** synthetic 数据点，synthetic 高度收敛
- **Interestingness space**：co-designed 仍包住 synthetic，且右上角有个红色框区域是 **synthetic 的盲点**，而 hard test（人类 crafted）大多落在这个盲点里

这个图把"为什么 co-design 在 hard test 上吊打 synthetic"讲得非常直观 —— synthetic 的分布根本没覆盖到人类设计师会去探索的区域，模型在那个区域完全没训练信号。这也呼应了 **model collapse on synthetic data** (https://arxiv.org/abs/2305.17493) 的风险：纯合成数据训练的模型会越来越收敛到合成分布，OOD 能力逐渐归零。

---

## 6. 我自己的延伸思考

### 6.1 Delta-Engine 与终身学习

Delta-Engine 本质上是一种 **lifelong / continual learning** 形式 —— 但不是模型权重的持续学习（像 CL literature 那样），而是 **外部 memory 的持续学习**。模型的权重训练完就 frozen，所有的"学习"以代码形式存在 engine state 里。这避开了 catastrophic forgetting，因为根本没有 in-weight learning 在发生。

这种架构让我想到 **REINER / Compositional Memory / 外挂知识库** 的整条 lineage。特别是和 **Voyager 的 skill library** 几乎同构 —— 区别在于 Voyager 的 skill 是"宏观动作序列"，Delta-Engine 的 patch 是"对象方法"，后者更接近 OOP 的语义。

### 6.2 与 Software Engineering LLM 的关系

Delta-Engine 可以被看作 **runtime Copilot** —— GitHub Copilot 是"开发时补代码"，Delta-Engine 是"运行时补代码"。这个隐喻指向一个更激进的方向：未来的应用可能不再有"开发完"这个状态，代码会在部署后由 LLM 持续根据用户意图演化。

参考 **Meta's Toolformer** (https://arxiv.org/abs/2302.04761)、**OpenAI Code Interpreter** 等都是同一 family —— LLM 写代码、执行代码、反馈结果。Delta-Engine 加了一个维度：**代码会被持久化并成为下次生成的 context**。

### 6.3 Safety 的 open problem

Paper 在 conclusion 里点了 safety concern，但没展开。我想补充：

- **Code injection**：玩家可以说 "make my pokemon delete the filesystem"，neural proxy 会编译这个意图为代码。如果 base engine 的 sandbox 不严格，就是漏洞
- **Adversarial prompt**：玩家构造 prompt 让 proxy 生成看起来合法但 battle 时 advantage 极大的方法，破坏游戏平衡
- **Gradient of capability**：在 Free Pokemon 里玩家可以 craft "Thanos pokemon 一回合秒杀任何对手"。这破坏了 multiplayer 的对称性，需要 cap 机制

这部分和 AI agent safety 的整体话题连成一片，参考 **SIMA (Team et al., 2024, https://arxiv.org/abs/2404.10179)** 和 **CRADLE (Tan et al., 2024, https://arxiv.org/abs/2403.03186)** 都面对类似问题。

### 6.4 与 diffusion-based world model 的对比

GameNGen（https://arxiv.org/abs/2408.14837）用 diffusion 学 DOOM 的 frame transition，优点是端到端、能复刻视觉细节；缺点是 latent 不可编辑、不可组合。Delta-Engine 走到反面：完全可编辑、可读、可 git diff，但失去了像素级渲染能力。

我猜下一代方向是 hybrid：**代码做逻辑层 + diffusion 做渲染层**。事实上 Free Pokemon 已经隐含了这个分工 —— battle engine 是手写代码（逻辑层），而 pokemon 的视觉 asset 论文承认是 future work（渲染层）。把 GameNGen 风格的 neural renderer 套到 delta-engine 的代码层之上，可能是 ORPG 的完整形态。

### 6.5 关于"Interestingness"的形式化

ToI 是一个非常工程化的近似 —— 用一组 binary tag 命中数衡量 interestingness。但这是个 **coarse** 度量。更精细的方向：

- 用 **LLM-as-judge** 做 pairwise interestingness 评分，再 rank
- 用 **player engagement** 作为 reward signal（在线 A/B 测试）
- 用 **information-theoretic novelty**：role code 的 algorithmic complexity（Kolmogorov）作为 proxy

这和 **Open-ended learning**（POET, AMIGo 等）的 reward design 是一个问题 family。

---

## 7. 关键 Takeaways

1. **Delta-Engine 把 game state 变成代码库**，每次 evolution 是一次代码 patch + merge，neural proxy（LLM）负责意图→代码翻译
2. **Retrieval 是 scalability 的关键**：让 LLM 自己读 skeleton 决定 retrieve 哪些方法，避免 context window 撑爆
3. **Human-AI Co-design 用 prototype 增强 imagination**，用 ToI 量化 interestingness，生成比纯合成更 OOD 的高质量数据
4. **Hard test 上 co-design + retrieval 把 Acc 从 58.6 推到 89.7**，说明数据多样性和 context 管理同样重要
5. **Pokemon 是个非常好的 testbed**：离散、结构化、大众熟悉、有 baseline 可比

整体来说，这篇 paper 在"LLM + game"这个方向上选了一个非常清楚的切角 —— 不去打端到端渲染（GameNGen 已经占了），也不去打 LLM agent 玩现有游戏（Pokellmon 已经占了），而是开了一个新维度：**让玩家通过自然语言造角色**。这个切角的工程性很强，但留下的开放问题（safety、rendering、规模化的更长 evolution、multi-agent 协同演化）也都非常 juicy。

---

**主要参考链接**：

- Delta-Engine 前作 Instruction-driven Game Engine: https://arxiv.org/abs/2404.00276
- Voyager (Minecraft LLM agent + skill library): https://arxiv.org/abs/2305.16291
- GameNGen (diffusion game engine): https://arxiv.org/abs/2408.14837
- Genie (DeepMind generative interactive env): https://openreview.net/forum?id=bJbSbJskOS
- CodeGemma: https://arxiv.org/abs/2403.08295
- Code Llama: https://arxiv.org/abs/2308.12950
- LoRA: https://arxiv.org/abs/2106.09685
- Pokellmon: https://arxiv.org/abs/2402.01118
- Self-Instruct: https://aclanthology.org/2023.acl-long.754
- Curse of Recursion (model collapse): https://arxiv.org/abs/2305.17493
- CRADLE: https://arxiv.org/abs/2403.03186
- SIMA: https://arxiv.org/abs/2404.10179
- GAVEL (evolution + LLM game gen): https://arxiv.org/abs/2407.09388
- Program Synthesis as Generative Method: https://doi.org/10.1145/3102071.3102076
