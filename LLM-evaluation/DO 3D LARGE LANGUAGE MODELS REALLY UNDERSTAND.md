---
source_pdf: DO 3D LARGE LANGUAGE MODELS REALLY UNDERSTAND.pdf
paper_sha256: 318911d41216a0baa1f2b21651bbf71877b286b8837f114ecb14b03649684f2a
processed_at: '2026-08-03T22:48:45-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用Karpathy的口吻讲讲这篇paper

好，我把这篇paper当故事讲给你听。

---

## 这篇paper到底揭了什么"丑闻"

3D-LLM这个field这两年特别热闹——3D-LLM、LEO、Chat-Scene、GPT4Scene你方唱罢我登场，SQA3D benchmark上的分数从30%一路涨到60%。大家都很high，觉得"3D spatial understanding"在突飞猛进。

但这篇paper上来就甩了一个炸弹：

> **你把所有3D input完全关掉，只给模型看text，它分数几乎不掉。**

什么意思呢？想象你训练了一个号称"能看3D场景回答问题"的模型，它号称理解点云、理解空间关系。然后作者做了个对照实验：把3D encoder那一路全堵死，只让模型看问题描述（"你站在房间中央，左边是什么"之类），结果——**LEO在SQA3D上从52.2掉到50.6，掉1.6个点**。Chat-3D v2更夸张，几乎完全一样。

这就相当于你考试得了90分，结果把你的眼睛蒙上，你还能考88分。说明你那90分里头有88分根本不是靠眼睛看题答的，是靠题目里"提示词"猜的。

这是典型的**shortcut learning**——模型学会的不是"理解3D空间"，而是"看到'black rectangular object on the wall'就回答TV"这种language prior。

---

## 为什么这个"丑闻"之前没人揭穿

答案很简单：**因为benchmark设计得不够狠**。

SQA3D的作者其实意识到过bias问题，他们做过"answer distribution balancing"——就是确保"TV"这个答案不会占太多比例。但这个做法只能解决**最浅层的bias**。

更深的bias是什么？比如：

- **Saliency preference**：人会问的object几乎总是显眼的（沙发、桌子、电视），不会问"墙角的灰尘"
- **Canonical layout**：房间布局有canonical pattern，厨房通常有冰箱，卧室通常有床
- **Guessable commonsense**：ebony and ivory的instrument就是piano，natural light来源就是window

这些bias是**数据生成过程里根深蒂固的**，靠简单的answer balancing根本洗不掉。SQA3D的问题集里，可能有一大半问题用纯text + commonsense就能猜出来。

所以这篇paper的核心洞察是：**"模型能答对"跟"模型真懂3D"完全不是一回事**。

---

## Real-3DQA怎么fix这个问题的

fix的方法特别elegant，核心思想是：

> **如果一个问题，blind模型（只看text）也能答对，那这个问题就是"3D-independent"的，删掉。**

具体流程：

1. 拿3个不同的3D-LLM（LEO、Chat-Scene、3D-LLM）
2. 对每个模型训练两个版本：full版（有3D input）和blind版（没3D input）
3. 对每个问题，如果**full版和blind版都答对了**，这个问题就被判定为3D-independent，删掉
4. 三个模型的"被删问题集"取并集（conservative，宁可错杀也不放过）
5. 再用GPT-4o-mini做一遍text-only filtering，把GPT能猜对的也删掉

结果从3519个问题，删到大概1485个。**一大半问题都是"水货"**。

这个做法的聪明之处在于：它不依赖任何人工定义"什么问题有3D dependency"，而是**用模型行为本身作为判据**。如果一个问题在所有模型上都表现出"3D input无关性"，那它大概率就是真的无关。

---

## 真正的"杀手锏"：Viewpoint Rotation Score (VRS)

光是filtering还不够。作者又加了一个更狠的测试：**viewpoint consistency**。

Intuition特别简单——

> **如果一个人真懂3D空间，你让他转个身，他还是能告诉你"我右边是什么"。**

具体做法：

- 原始场景：agent站在某个位置，朝向某个方向，问"我右边是什么"
- 把agent的viewpoint rotate 90°/180°/270°
- 场景里的物体位置完全不变，只是agent的"前后左右"变了
- 答案根据新viewpoint动态调整（比如原来"右边是whiteboard"，rotate 180°后变成"左边是whiteboard"）
- 对同一个spatial setup生成4个等价的question

然后看模型能不能在4个变体上**一致答对**。

VRS score的计算：

$$\text{VRS} = \frac{1}{4}\sum_{k=1}^{4} P_k$$

其中 $P_k$ = "至少答对k个变体"的问题占比。

这个公式的设计很讲究——**取平均而不只看 $P_4$**，因为只看"全对"的话metric会过于binary，模型答对3个变体和答对1个变体没区别，这丢失了中间档的discrimination。

实验结果（Table 3）非常震撼：

| Model | VRS |
|-------|-----|
| 3D-LLM | 9.6% |
| Chat-3D v2 | 6.6% |
| LEO | 14.3% |
| Chat-Scene | 12.9% |
| GPT4Scene | 18.2% |

最好的GPT4Scene也才18.2%。更狠的是看"4个全对"那一列：

- 3D-LLM: 0.1%
- Chat-3D v2: 0.0%
- LEO: 0.4%
- Chat-Scene: 0.1%
- GPT4Scene: 0.5%

**几乎全部崩溃**。

这说明了什么？**当前的3D-LLM根本没有什么viewpoint-invariant的3D representation**。它们答对的问题，往往是"碰巧"在这个viewpoint下能用其他线索（比如全局scene feature的某种pattern）猜对，而不是真的理解了"我在哪、我朝哪、物体相对我什么位置"这种egocentric spatial frame。

---

## 那怎么"逼"模型真的用3D信息呢？3DR-FT

到这里，paper已经诊断出问题了：模型在偷懒，用text shortcut。下一步自然就是**怎么逼它不偷懒**。

作者提出了3D-aware Reweighted Fine-tuning（3DR-FT），核心idea我用人话讲：

> **对每个training sample，比较一下blind模型和full模型谁更"意外"。如果blind模型也很"意外"（猜不到），说明这个sample必须靠3D信息才能答——那就加大它的training weight。**

具体公式：

$$w_j = \frac{\log p_\phi(y_j \mid y_{<j}, x_{\text{text}})}{\log p_\theta(y_j \mid y_{<j}, x_{\text{text}})}$$

变量解释：
- $\phi$ = blind model（frozen，不更新）
- $\theta$ = full model（正在train的）
- $y_j$ = ground truth的第 $j$ 个token
- $x_{\text{text}}$ = text prompt
- 分子 = blind model对ground truth token的log概率（很负说明blind model很意外）
- 分母 = full model的log概率

**Intuition**：两个都是负数（因为是log概率），如果 $|\log p_\phi| > |\log p_\theta|$，说明blind model比full model更surprised，比值 $> 1$，upweight这个token。如果两个都差不多能猜到，比值 $\approx 1$，基本不变。

然后loss就变成：

$$\mathcal{L}_{\text{3DR-FT}} = \mathbb{E}\left[-\sum_j w_j \log p_\theta(y_j \mid y_{<j}, x_{\text{text}}, x_{\text{3D}})\right]$$

就是给每个token的cross-entropy loss乘了个weight $w_j$。

### Theoretical insight为什么这个loss有效

作者在Appendix E给了理论分析。定义一个"3D dependency gap"：

$$\delta_j = \frac{p_\theta(y_j \mid \text{text}, \text{3D}) - p_\theta(y_j \mid \text{text})}{p_\theta(y_j \mid \text{text})}$$

这个 $\delta_j$ 衡量"3D input让概率提升了多少比例"。$\delta_j = 0$ 意味着3D input完全没用，这正是我们要避免的。

把loss展开后，能分解成两项：

$$\mathcal{L}_{\text{3DR-FT}} = \underbrace{\text{const}}_{\text{跟 }\theta\text{无关}} + \mathbb{E}\left[-\sum_j w_j \log(1 + \delta_j)\right]$$

第二项里，要最小化loss就要让 $\log(1 + \delta_j)$ 尽量大，也就是让 $\delta_j$ 尽量大——**这等价于显式地最大化3D input的marginal contribution**。

这就是3DR-FT的理论核心：它不只在fit answer，而是显式地逼迫模型让3D input"做出贡献"。

---

## 实验结果：3DR-FT的效果

Table 4的结果：

**LEO on Real-3DQA**:
- Supervised FT: 19.1
- Blind FT: 13.6
- **3DR-FT: 29.3** (+10.2)

**Chat-Scene on Real-3DQA**:
- Supervised FT: 22.1
- Blind FT: 14.4
- **3DR-FT: 33.9** (+11.8)

涨幅都很显著。Real-ScanQA（用同样pipeline从ScanQA构建的另一个filtered benchmark）上也涨了——说明3DR-FT不是overfit到Real-3DQA的某个specific quirk。

### 但有个反直觉现象

Table 4里还有个数据：Chat-Scene在**原始SQA3D**上，Supervised FT = 57.2，3DR-FT = 48.9——**反而下降了8个点**。

这乍看很矛盾：你怎么"加强3D依赖"反而让分数下降？

Figure 6的解释特别精彩：

> 3DR-FT后，591个question从correct翻到wrong。其中441个来自filtered set（即被我们判定为"3D-independent"的那批easy questions）。而且这441个里头，**413个(~70%)在Blind FT模型上也是对的**——也就是说，这些question本来就是靠text shortcut答对的，3DR-FT让模型不再用shortcut，反而答不对了。

这个发现validates了3DR-FT的设计——**它在SQA3D上的"退步"恰恰证明SQA3D分数被shortcuts inflate了**。就像你给学生复习时，发现他原来"会"的题都是背答案背的，让他真的理解原理后，那些靠背答案的题反而不会了——但这是好事，因为他开始真的懂了。

---

## 一个意外发现：Situation description根本没用

Table 6的ablation特别有意思，我之前提过但没细讲：

对LEO做inference-time ablation：
- Full input (Situation + Question + 3D): 49.4 EM
- No Situation (只Question + 3D): **49.3 EM**——几乎完全一样！

**"Situation description"——就是"你现在站在房间中央，面朝窗户"那段描述egocentric viewpoint的文字——完全没用**。

这说明现在的3D-LLM对"egocentric viewpoint"的处理基本是装饰性的。模型根本没在用这段描述来建立spatial frame，它可能就是把situation + question当一坨text tokenize了扔进LLM。

这跟VRS的失败是consistent的：如果模型真的用了situation description，那rotation之后应该还能答对；它答不对，说明situation description对它来说就是noise。

---

## 为什么这个field会被"假progress"困扰

我个人的看法——这背后有几个结构性原因：

### 1. Benchmark设计的根本缺陷

3D-QA沿用了2D-QA的"per-item accuracy"protocol。每个question独立打分，answer对了就得分。这种metric完全无法捕捉**"模型是否consistent"**。一个模型可以在4个rotation上各答对一次但完全不一致——因为每次答对可能靠的是不同的shortcut。

Beacon3D (Huang et al. 2025) 从另一个角度也指出类似问题：cross-task consistency（QA vs grounding）。他们发现3D-LLM在QA里说"左边是桌子"，但在grounding task里指向完全不同的位置——两个task之间完全inconsistent。

### 2. Point cloud representation的inherent limitation

看Table 3的per-type breakdown：

- **Distance问题**：所有模型都差（4.3-15.8）
- **Direction问题**：也都差（14.6-25.2）
- **Existence问题**：3D-LLM反而最好（44.2），超过GPT4Scene（36.5）

为什么3D-LLM在existence上反超？作者推测是因为3D-LLM用**point-wise encoding**（保留局部geometric detail），而其他都是object-centric（feature被aggregated到object level，丢失了fine-grained info）。

但distance和direction呢？point-wise也救不了。为什么？因为**distance/direction需要的是global spatial reasoning**，需要模型在某个canonical/egocentric frame下建立物体之间的关系。而现在的3D-LLM都是把3D feature和text拼一起扔进LLM——LLM怎么"算"两个物体距离多远？这个reasoning chain根本没建立起来。

### 3. Training signal的misalignment

Standard SFT的loss是 $\log p(y \mid x_{\text{text}}, x_{\text{3D}})$。模型可以学到两种策略：
- 策略A：真的用 $x_{\text{3D}}$ 来推断答案
- 策略B：忽略 $x_{\text{3D}}$，只用 $x_{\text{text}}$ 猜

由于text shortcut的存在，策略B在某些sample上是更easy path——loss下降更快。模型自然会走策略B。SFT的loss根本没机制来防止这种shortcut。

3DR-FT的本质就是**通过weight manipulation让策略B的"捷径"变难走**，把模型往策略A推。

---

## 这个思路跟相关work的联系

### 2D VQA里的counterfactual思路

这个工作的intellectual lineage可以追溯到Agrawal et al. 2018的VQA-CP（Counterfactual VQA）。VQA-CP的核心思想是：**改变answer distribution**——training set里"网球"问题的答案是"绿色"，test set里改成"黄色"，逼模型真的看图。模型如果靠language prior"球都是绿色"答题，在VQA-CP上会crash。

Real-3DQA跟VQA-CP的精神一致，但做法不同：VQA-CP通过**分布shift**来暴露shortcut，Real-3DQA通过**model-based filtering + viewpoint consistency**来暴露shortcut。VQA-CP需要重新构造数据集，Real-3DQA是post-hoc过滤——更轻量。

### MathVerse (Zhang et al. 2024)

MathVerse也是类似精神，针对multimodal LLM做math reasoning。他们把visual diagram替换成text-only description，看模型分数掉多少。如果掉得少，说明模型没在"看图"，只是在读题。这跟Real-3DQA的blind model对照实验是同一个思路。

### 3DR-FT跟Importance Sampling的联系

3DR-FT的reweighting function：

$$w_j = \frac{S_\phi(y_j)}{S_\theta(y_j)}$$

这个形式在importance sampling里有类似的味道。$S_\phi$ 相当于一个"baseline distribution"下的surprise，$S_\theta$ 是current distribution下的surprise。upweight "baseline分布下罕见但current分布下不罕见"的样本，本质上是在强调"current model能predict的，但baseline model predict不了的"部分——也就是3D input带来的额外信息。

这个idea其实可以推广到任何multimodal setting：
- 2D VLM：用text-only LLM作为baseline，reweight image-dependent samples
- Video understanding：用text-only baseline，reweight video-dependent samples
- Audio-language：用text-only baseline，reweight audio-dependent samples

每一步都是在"让modality真的有用"。

---

## 我对field的展望

这篇paper我觉得是**一个field-resetting work**。它不刷SOTA，但它告诉3D-LLM community：你们这两年reported的progress里，有多少是真的、多少是水。

具体展望：

### 1. Viewpoint equivariance应该是3D-LLM的inductive bias

VRS的全线崩溃（best 0.5%）说明现在的architecture完全没有rotation invariance/equivariance的inductive bias。这是spatial AI的核心property，应该被design into architecture。

可能的方向：
- SE(3)-equivariant point cloud encoders（从group equivariant CNN那line of work借鉴）
- Cross-view contrastive loss（不同viewpoint下同一scene的representation应该相同）
- Explicit egocentric frame encoding（不要让situation description变成decorative text，要真的encode成spatial frame）

### 2. Cross-task consistency应该成为标准metric

Beacon3D提的cross-task consistency（QA vs grounding consistency）和Real-3DQA的cross-question consistency（cross-viewpoint consistency）是互补的。一个测"模型在不同task间是否consistent"，一个测"模型在spatial transform下是否consistent"。

未来3D-LLM的evaluation应该至少包含这两个consistency metric，而**不只是per-item accuracy**。

### 3. "Diagnostic benchmark"应该成为field的standard practice

3D-LLM不应该只在"刷分benchmark"上compare。应该有一组**diagnostic benchmark**专门测：
- Text shortcut的依赖程度（用blind model baseline）
- Viewpoint consistency
- Cross-task consistency  
- Spatial relation的具体subtype（distance, direction, counting, existence）

这样的evaluation才能区分"真懂3D"和"装懂3D"。

### 4. 3D-LLM的training paradigm可能需要根本rethink

现在的3D-LLM training基本都是：拿2D VLM的recipe，加3D encoder，做SFT。这个recipe在2D VLM上work，因为2D image本身就rich、language shortcut相对少；但在3D setting下，3D representation的sparsity + scene layout的canonical nature，让shortcut变得dominant。

3D-LLM可能需要新的training paradigm，比如：
- Contrastive learning：同一scene不同viewpoint应该predict一致
- Counterfactual training：把一个object挪到不同位置，问对应问题，让模型真的learn spatial relation
- Multi-task joint training：QA + grounding + captioning must be consistent

---

## 最终一句话总结

这篇paper告诉3D-LLM community：

> **如果你的模型在blind状态下分数几乎不掉，那你的"3D understanding"是装的；如果你的模型rotate viewpoint之后就crash，那它的"spatial reasoning"是frame-dependent的pattern matching，不是真的spatial cognition。**

Real-3DQA + VRS应该成为3D-LLM evaluation的minimum bar，3DR-FT的思想应该被推广到所有multimodal training。这paper是field-resetting级别的，我预期会引发一波"diagnostic benchmark"的热潮。

---

## References

- Real-3DQA project page: https://real-3dqa.github.io/
- SQA3D original benchmark: https://sqa3d.github.io/
- VQA-CP (Agrawal et al. 2018): https://arxiv.org/abs/1710.06236
- MathVerse (similar diagnostic approach in 2D math): https://arxiv.org/abs/2403.14699
- Beacon3D (cross-task consistency): https://arxiv.org/abs/2503.22420
- LEO (embodied agent in 3D): https://github.com/embodied-agent/LEO
- Chat-Scene: https://github.com/OpenRobotLab/ChatScene
- 3D-LLM: https://github.com/UMass-Foundation-Model/3D-LLM
- "Does your 3D encoder really work?" (Li et al. 2025): https://arxiv.org/abs/2506.05318
- "The point, the vision and the text" (Zhang et al. 2025): https://arxiv.org/abs/2504.04540

---

希望这个讲解帮你build了intuition。这paper的story其实特别干净：诊断问题（blind model对照）→ 揭示问题（benchmark design flaw）→ 解决问题（filtering + VRS + 3DR-FT）。整个work的beauty在于它**用模型自己的行为来诊断模型**——这是最 principled的evaluation philosophy。

---

# 这篇paper的核心质疑:3D-LLMs真的"理解"3D吗?

Andrej,这篇paper我认为是非常重要的"reality check"工作,直击了当前3D-LLM社区的一个核心问题:**community用SQA3D作为评测标杆,但这个benchmark根本无法区分模型是"真的用了3D信息"还是"在用language prior猜答案"**。我详细讲一下,从intuition到公式再到实验细节。

---

## 1. Paper发现的核心问题

作者发现了一个令人震惊的事实:**一个blind model(只看text QA pairs,完全不看3D input)在SQA3D上可以match甚至超越state-of-the-art的3D-LLMs**。Figure 2里展示,LEO在SQA3D上52.2 EM_Refined,但它的blind counterpart能拿到50.6 EM_Refined——意味着3D input只贡献了1.6个点的marginal benefit。

这暴露了一个critical问题:3D-LLM community这两年的"progress"很大部分是illusory的。模型不是学会了spatial reasoning,而是学会了language shortcuts。比如"What is the black rectangular object on the wall?"——答案几乎永远是"TV",这跟3D spatial reasoning毫无关系,完全是commonsense prior。

---

## 2. Real-3DQA Benchmark的构建

### 2.1 Filtering 3D-independent Questions的数学定义

设原始问题集合为 $Q$(来自SQA3D Ma et al. 2023),评估三个3D-LLMs记为 $M_A, M_B, M_C$(对应LEO, Chat-Scene, 3D-LLM)。

对每个模型 $M_X$,定义其3D-independent问题集合:
$$Q_X = \{q \in Q \mid M_X(q) = M_X^{\text{blind}}(q) = \text{correct}\}$$

这里 $M_X(q)$ 是original model对问题 $q$ 的预测,$M_X^{\text{blind}}(q)$ 是该模型的blind版本(去掉3D input,只保留text prompt)的预测。如果一个问题两个版本都答对,说明该问题的答案不依赖3D信息。

然后取三个模型的并集:
$$\bar{Q}_{\text{3D-filtered}} = Q_A \cup Q_B \cup Q_C$$

剩余问题集:
$$Q' = Q \setminus \bar{Q}_{\text{3D-filtered}}$$

注意这里取**union**而不是intersection,这是conservative的做法——只要任何一个模型能在blind状态答对,就认为它是3D-independent的。这避免了某个模型"恰好能猜对"的偶然性。

进一步用GPT-4o-mini做text-only过滤:
$$Q_{\text{GPT}} = \{q \in Q' \mid \text{GPT}(q) = \text{correct}\}$$
$$Q_{\text{final}} = Q' \setminus Q_{\text{GPT}}$$

最终从3519个问题留下大约1485个(Table 7显示LEO的filtering最aggressive,过滤了1197个)。

### 2.2 过滤数据的具体例子(被过滤掉的trivial问题)

Paper在Appendix D.1给了三个经典例子:
- "What is to my left that gives me natural light in the room?" → window
- "What is to my right that I can use washable marker to write with?" → whiteboard  
- "What instrument in front of you is ebony and ivory?" → piano

这些问题的答案完全可以由language + commonsense推断,根本不需要3D context。3D-LLMs之所以"答对"完全是因为prior,而非spatial understanding。

---

## 3. Viewpoint Rotation Score (VRS)的细节

### 3.1 核心intuition

VRS的设计哲学:**如果模型真的理解3D空间,那么rotate observer viewpoint之后,它应该还能给出逻辑一致的答案**。

Figure 4的例子:
- 原始场景:agent面向trash can,table在后方,问"What is on my right?" 答案:whiteboard
- Rotate 90°/180°/270°:agent面向不同对象,但场景中物体之间的spatial关系不变,只是reference frame变了
- 答案要根据新的viewpoint动态调整

### 3.2 VRS公式解析

设四个变体(原始+3个rotation)构成一个batch。定义:
- $N_k$ = 至少答对 $k$ 个问题的instance数,其中 $k \in \{1, 2, 3, 4\}$
- $N_{\text{total}}$ = 总instance数
- $P_k = \frac{N_k}{N_{\text{total}}} \times 100$

最终:
$$\text{VRS} = \frac{1}{4} \sum_{k=1}^{4} P_k$$

**为什么这样设计**:
- $P_1$ = 答对至少1个的占比(宽松)
- $P_4$ = 答对全部4个的占比(严格)
- 取平均而非只看 $P_4$ 是为了避免metric只奖励"全对或全错"的二值化,保留中间档的discrimination power

### 3.3 Table 3的实验结果非常震撼

| 3D-LLM | one | two | three | four | VRS% |
|--------|-----|-----|-------|------|------|
| 3D-LLM | 33.2 | 4.1 | 1.1 | 0.1 | 9.6 |
| Chat-3D v2 | 23.2 | 2.7 | 0.5 | 0.0 | 6.6 |
| LEO | 46.9 | 8.1 | 1.6 | 0.4 | 14.3 |
| Chat-Scene | 43.3 | 7.1 | 1.2 | 0.1 | 12.9 |
| GPT4Scene | 55.5 | 14.3 | 2.5 | 0.5 | 18.2 |

最好的GPT4Scene也只在"四个全对"上达到0.5%——这意味着viewpoint consistency几乎是完全失败的。这跟paper里说的"3D-LLMs几乎从未被评估过rotation robustness"对应——training和evaluation都没有这个signal。

### 3.4 不同问题类型的分析

Table 3的右半部分按问题类型分析:distance, direction, counting, existence。
- **Distance**最差(4.3~15.8):物体间距离判断需要真正的metric 3D understanding
- **Direction**也差(14.6~25.2):需要reference frame comprehension
- **Counting**差异大(16.7~31.7)
- **Existence**最好(21.2~44.2):只需检测物体存在性

有趣的发现:**3D-LLM虽然在VRS上排第二低(9.6%),但在existence类别上拿到44.2%,超过GPT4Scene(36.5%)**。作者推测这是因为3D-LLM用的是**point-wise feature encoding**(而不是object-centric),保留了更细粒度的local geometric cues,对object existence detection有利;但对spatial relation这种需要global reasoning的任务反而吃亏。

---

## 4. 3D-aware Reweighted Fine-tuning (3DR-FT)的数学细节

### 4.1 Standard SFT vs Blind FT的对比

Standard SFT的loss:
$$\mathcal{L}_{\text{SFT}}(\theta) = \mathbb{E}_{\mathcal{D}}\left[-\sum_{j=1}^{T} \log p_\theta(y_j \mid y_{<j}, x_{\text{text}}, x_{\text{3D}})\right]$$

其中 $(x_{\text{text}}^{(i)}, x_{\text{3D}}^{(i)}, y^{(i)}) \sim \mathcal{D}$,$x_{\text{text}}$是situation description+question的text prompt,$x_{\text{3D}}$是point cloud等3D输入,$y$是ground truth answer。

Blind FT:把 $x_{\text{3D}}$ 完全去掉:
$$\mathcal{L}_{\text{BF}}(\theta) = \mathbb{E}_{\mathcal{D}}\left[-\sum_{j=1}^{T} \log p_\theta(y_j \mid y_{<j}, x_{\text{text}})\right]$$

BF模型的存在意义是作为**reference baseline**:它告诉我们"光靠text能猜到什么程度"。

### 4.2 Reweighting function的核心思想

定义surprise function $S_\theta(y, x) = \log p_\theta(y_j \mid y_{<j}, x)$,这是information theory里"surprisal"或"self-information"的标准定义。值越小(越负),model越surprised。

Reweighting function:
$$w_j(y, x_{\text{text}}) := \frac{S_\phi(y, x_{\text{text}})}{S_\theta(y, x_{\text{text}})} = \frac{\log p_\phi(y_j \mid y_{<j}, x_{\text{text}})}{\log p_\theta(y_j \mid y_{<j}, x_{\text{text}})}$$

变量解释:
- $\phi$ = blind model的参数(frozen)
- $\theta$ = 当前training model的参数(被优化)
- $y_j$ = ground truth的第 $j$ 个token
- $y_{<j}$ = ground truth的前 $j-1$ 个token(teacher forcing)
- $x_{\text{text}}$ = text-only prompt

**Intuition**:
- 如果 $w_j$ 大 → blind model比current model更surprised → 这个token是blind model猜不到的,需要3D信息才能预测 → 应该upweight
- 如果 $w_j$ 小 → blind model也能猜到 → 这个token是text-only就能解的 → 应该downweight

注意:由于 $\log p \in (-\infty, 0)$,比值 $w_j$ 在两个值都为负数时为正。如果blind模型比current模型对token $y_j$ 更"surprised"(更小的概率,即更负的log prob),那么 $|S_\phi| > |S_\theta|$,比值 $> 1$,upweight这个token。

### 4.3 3DR-FT Loss

$$\mathcal{L}_{\text{3DR-FT}}(\theta) := \mathbb{E}_{\mathcal{D}}\left[-\sum_{j=1}^{T} w_j(y, x_{\text{text}}) \log p_\theta(y_j \mid y_{<j}, x_{\text{text}}, x_{\text{3D}})\right]$$

跟SFT比,只是多了一个per-token的weight $w_j$。Implementation上是把cross-entropy loss乘以每个token的weight。

### 4.4 Theoretical insight(Appendix E)

定义conditional-independence gap:
$$\delta_j := \frac{p_\theta(y_j \mid y_{<j}, x_{\text{text}}, x_{\text{3D}}) - p_\theta(y_j \mid y_{<j}, x_{\text{text}})}{p_\theta(y_j \mid y_{<j}, x_{\text{text}})}$$

变量解释:
- 分子 = 加入3D input后概率的提升
- 分母 = 不加3D input时的概率
- $\delta_j$ 衡量"3D input让概率提升了多少比例"

当 $\delta_j = 0$ 时,3D input对预测 $y_j$ 没有任何影响——模型在ignore 3D info,这就是我们要避免的。

记 $s_j := p_\theta(y_j \mid y_{<j}, x_{\text{text}})$,则有 $p_\theta(y_j \mid y_{<j}, x_{\text{text}}, x_{\text{3D}}) = s_j(1 + \delta_j)$。

代入3DR-FT loss后展开:

$$\mathcal{L}_{\text{3DR-FT}}(\theta) = \underbrace{\mathbb{E}_P\left[-\sum_j \log p_\phi(y_j \mid y_{<j}, x_{\text{text}})\right]}_{\text{Term 1: BF model perplexity (const w.r.t. θ)}} + \underbrace{\mathbb{E}_P\left[-\sum_j w_j(y, x_{\text{text}}) \log(1 + \delta_j)\right]}_{\text{Term 2: weighted conditional-independence gap}}$$

**关键insight**:
- Term 1跟 $\theta$ 无关(blind model $\phi$ 是frozen的),不影响gradient
- Term 2中的 $-\log(1 + \delta_j)$ 在 $\delta_j > 0$ 时为负,要最小化loss就要让 $\delta_j$ 尽量大
- 所以3DR-FT本质上是在**显式地最大化3D input的marginal contribution** $\delta_j$

这比简单的SFT强很多——SFT的loss是 $-\log p_\theta(y_j \mid \cdots, x_{\text{3D}})$,模型可以选择让 $x_{\text{3D}}$ 不起作用,只要 $s_j$ 足够大就行;而3DR-FT通过weight $w_j$ 显式惩罚了"3D input被忽略"的情况。

---

## 5. Experimental Results的细节解读

### 5.1 Table 2: Performance Drop on Real-3DQA

| 3D-LLM | SQA3D EM | Real-3DQA EM | Drop |
|--------|----------|--------------|------|
| 3D-LLM | 47.8 | 7.5 | -40.3 |
| Chat-3D v2 | 45.0 | 3.4 | -41.6 |
| LEO | 49.4 | 14.3 | -35.1 |
| Chat-Scene | 54.4 | 17.0 | -37.4 |
| GPT4Scene | 60.6 | 33.1 | -27.5 |

**最震撼的是Chat-3D v2跌到3.4% EM**——基本上是chance level。这说明Chat-3D v2在SQA3D上的45% accuracy几乎全靠language shortcuts。

GPT4Scene跌得最少(27.5),因为它用multi-view image作为3D awareness的proxy,比point cloud-based方法的visual grounding更强。但即使如此,从60.6跌到33.1也是巨大的gap。

### 5.2 Table 4: 3DR-FT的效果

LEO on Real-3DQA:
- Supervised FT: 19.1
- Blind FT: 13.6
- **3DR-FT: 29.3** (+10.2 absolute gain)

Chat-Scene on Real-3DQA:
- Supervised FT: 22.1
- Blind FT: 14.4
- **3DR-FT: 33.9** (+11.8 absolute gain)

Real-ScanQA上的结果也一致(LEO从6.1到13.9),说明3DR-FT的gain不局限于SQA3D-derived benchmark。

### 5.3 SQA3D上看似"退步"的puzzle

Table 4里Chat-Scene在SQA3D上:Supervised FT = 57.2,3DR-FT = 48.9。**为什么emphasize 3D反而让SQA3D分数下降?**

Figure 6的解释:591个question从correct翻到wrong,其中441个来自filtered set(即被filter掉的easy questions),且**其中413个(~70%)在Blind FT模型上也是对的**——也就是说这些question原本靠text shortcut就能猜,3DR-FT让model不再用shortcut,反而猜不出来了。

这是paper里非常巧妙的一个发现:**3DR-FT在SQA3D上的"退步"恰恰证明SQA3D的分数被shortcuts inflate了**。这反而validates 3DR-FT的设计——它在push model离开shortcut region。

### 5.4 Figure 5的Attention分析

3DR-FT后,3D tokens(即point cloud tokens)的平均attention score显著提升。这说明模型在做answer generation时,确实更多地"看"3D information了,而不是单纯靠text。这跟Theoretical insight里的 $\delta_j > 0$ 是吻合的——attention增加反映了 $x_{\text{3D}}$ 对预测的marginal contribution增大。

### 5.5 Table 6: LEO Inference Ablation

这是理解3D-LLM内部机制的关键ablation:
- Full input (Situation + Question + 3D): 49.4 EM
- Shuffled 3D: 44.2 EM(损失5.2,说明3D的spatial info有一定作用)
- No 3D: 32.4 EM(损失17,但还是能拿32%——靠text猜)
- No Situation: 49.3 EM(几乎没损失——situation description基本没用!)
- No Question: 0.2 EM(question当然最重要)
- Only Question (no situation, no 3D): 18.6 EM

**最有意思的发现:situation description几乎完全redundant**——加上或去掉几乎没影响(49.4 vs 49.3)。这说明当前3D-LLM对egocentric viewpoint的encoding是失败的,模型并没有真正利用"situation"信息。这跟VRS的失败是consistent的——如果模型真的理解situation,rotation不应该让它崩溃。

---

## 6. 我个人的几个关键takeaways

### 6.1 Benchmark设计的根本问题

SQA3D的"answer distribution balancing"是一种**surface-level debiasing**,但根本不能解决deep biases(saliency preference, canonical layout, guessable answers)。Real-3DQA的model-based filtering(blind model vs full model对比)是一种更principled的方法——直接用"模型行为差异"作为3D dependency的proxy。

### 6.2 Object-centric vs Point-wise的trade-off

Object-centric representations(Chat-3D v2, LEO, Chat-Scene)在spatial reasoning上不如point-wise(3D-LLM)在existence detection上——这个发现挺反直觉的。我推测原因是object-centric representation在aggregation过程中丢失了fine-grained local geometric cues,而这些cues对existence detection重要,但对spatial relation来说,object-centric反而更structured应该更有利——可结果却不support这个直觉,说明现在的object-centric方法也没真正encode spatial relation。

### 6.3 Viewpoint consistency是未来3D-LLM的critical test

VRS揭示的failure非常dramatic——4个rotation全对的只有0.5%。这其实指向一个更深的问题:**现在的3D-LLM没有explicit的viewpoint-invariant representation mechanism**。它们要么是从allocentric scene encoding生成答案,要么是从固定viewpoint的egocentric features生成,完全没有rotation equivariance的inductive bias。

未来工作的方向应该是:
1. 显式design rotation-equivariant features(比如SE(3)-equivariant point cloud encoders)
2. Cross-view consistency loss作为training signal
3. Egocentric situation description的真正encoding(目前是decorative的)

### 6.4 3DR-FT的更广泛意义

3DR-FT的核心思想是"用blind model作为reference来reweight samples",这个idea其实可以推广到任何multimodal scenario:
- 2D VLM:用text-only baseline reweight image-dependent samples
- Audio-language:用text-only baseline reweight audio-dependent samples
- Video-language:用text-only baseline reweight video-dependent samples

这跟Agrawal et al. 2018的VQA-CP(counterfactual)思路有联系,但操作方式不同——VQA-CP是改变answer distribution,3DR-FT是per-sample per-token reweighting,粒度更细。

### 6.5 与最近相关工作的联系

Paper里提到的几个相关工作:
- Beacon3D (Huang et al. 2025):cross-task consistency(QA vs grounding)
- MathVerse (Zhang et al. 2024):text-only vs full input的gap评估(在2D math reasoning)
- VQA-CP (Agrawal et al. 2018):2D VQA的language prior问题

最近还有一些工作也是类似direction:
- "Does your 3D encoder really work?" (Li et al. 2025):质疑pretrain-sft from 2D VLMs在3D VLMs上的效果
- "The point, the vision and the text" (Zhang et al. 2025):质疑point cloud是否真的boost spatial reasoning

这个方向我觉得会越来越重要——community需要从"刷分"转向"诊断"。

---

## 7. References

- Paper arxiv link: https://arxiv.org/abs/2506.05318 (近似,需查实际版本)
- Project page: https://real-3dqa.github.io/
- SQA3D original: https://sqa3d.github.io/
- VQA-CP (counterfactual prior): https://arxiv.org/abs/1710.06236
- MathVerse (类似的2D diagnostic): https://arxiv.org/abs/2403.14699
- Beacon3D (cross-task consistency): https://arxiv.org/abs/2503.22420
- LEO: https://github.com/embodied-agent/LEO
- Chat-Scene: https://github.com/OpenRobotLab/ChatScene
- 3D-LLM: https://github.com/UMass-Foundation-Model/3D-LLM

---

## 8. 总结:这篇paper最重要的contribution

我认为这不是一个"SOTA paper",而是一个**"field-resetting paper"**——它告诉3D-LLM community:**你们这两年reported的progress可能是假的**。Real-3DQA + VRS应该成为future 3D-LLM evaluation的minimum bar,任何claim "spatial understanding"的model都应该在这两个metric上report。

3DR-FT虽然简单,但理论分析(Appendix E)揭示了它本质上是maximizing 3D input的marginal contribution $\delta_j$,这个insight我认为可以推广到很多multimodal场景。

最后,viewpoint consistency的全线崩溃(Table 3的0.5% best)是一个极有impact的finding——它说明3D-LLM的"3D understanding"不仅surface,而且是**frame-dependent的**。这点跟人类cognition完全不同——人类rotate mental model轻而易举,但3D-LLM完全做不到,这是spatial AI的一个核心limitation需要被解决。
