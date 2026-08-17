---
source_pdf: FADE Mitigating Hallucinations by Reducing Language-Prior Dominance.pdf
paper_sha256: 901774e23fce0d65ba04b0dbabe5b929fc48b125a5798bba78d37223a8ff8e1c
processed_at: '2026-08-04T06:27:57-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲FADE

## 一句话版本

LVLM hallucinate的root cause是：FFN在中间靠后那几层偷偷把visual evidence给覆盖了，FADE的做法就是在那几层把FFN的输出"音量调小一点"，让visual evidence能传到最后不被盖掉。

---

## 故事从头讲

### 第一个问题：LVLM为什么会hallucinate？

你给LVLM看一张滑雪的图，问"图里有狗吗"，它说"有"。但图里根本没有狗。

为什么会这样？之前大家的intuition是：model的attention没看对地方，或者visual feature没encode好。

但这篇paper说：**不是attention的问题**。Attention其实一直在好好工作，把图里的visual信息搬到answer该输出的位置了。问题出在FFN身上。

---

### 第二个问题：怎么发现的？

作者用了logit lens这个trick。简单说就是：在transformer的每一层，都把hidden state偷偷投影到vocabulary space，看看"如果在这层就停，model会输出什么"。

结果发现一个很诡异的现象：

对于hallucinated samples（最终答错的那些），在layer 15左右，logit lens显示model其实已经"想"输出正确答案了。P(correct answer)在那个位置已经很高。但是从layer 16开始，这个概率开始往下掉，到layer 22左右就掉到错误的token去了。

这个现象叫 **prediction drift**。中间层是对的，后面被"带偏"了。

就像你考试写到一半答案是对的，改着改着最后写了个错的。

---

### 第三个问题：谁把它带偏的？

Transformer每层有两个component：Attention和FFN。作者把两者的contribution拆开来算。

公式很简单。每层做的事情是：

```
hidden_state = hidden_state + Attention(hidden_state)    # attention写
hidden_state = hidden_state + FFN(hidden_state)           # FFN写
```

作者分别measure这两个"写"操作对correct answer logit的贡献。

结果Table 1很震撼：

| | Attention贡献 | FFN(16-22层)贡献 |
|---|---|---|
| 答对的样本 | +1.2 | +8.4 |
| 答错的样本 | +0.8 | **-3.5** |

看明白了吗？

- Attention在两种情况下都是正的（+1.2和+0.8），都在往correct方向推
- FFN在答对的样本上是+8.4（帮大忙），但在答错的样本上是-3.5（在捣乱）

所以hallucination的mechanism是这样的：

1. Attention把visual evidence搬上来，中间层P(correct)很高
2. FFN在layers 16-22读取residual stream，基于它store的language priors做变换
3. 当visual evidence弱或者ambiguous的时候，FFN的language prior就override了visual evidence
4. Prediction就drift到错误答案去了

---

### 第四个问题：FFN为什么存language priors？

这个要追溯到Geva et al. 2021的发现：FFN本质上是key-value memory。

FFN的第一层是"key lookup"：输入hidden state，跟store的pattern做match。
FFN的第二层是"value retrieval"：把match到的pattern对应的value写回residual stream。

这些key-value pair是在pretraining时从海量文本里学来的。比如"厨房"这个concept出现时，经常跟"冰箱"、"灶台"、"刀"co-occur。FFN就把这些co-occurrence存成了memory。

当LVLM的hidden state里有"厨房"的signal时，FFN自动联想出"冰箱"。即使图里厨房没有冰箱，FFN也会往residual stream里写"冰箱"的signal。

这就是language prior override visual evidence的mechanism。

参考：https://aclanthology.org/2021.emnlp-main.446/

---

### 第五个问题：FADE怎么fix的？

方法简单到令人发指。在critical layers（16-22），把FFN的输出乘以(1-α)：

```
hidden_state = hidden_state + (1 - α) * FFN(hidden_state)
```

α=0.6的意思就是：FFN的输出只保留40%的强度。

就这么一行代码改。

为什么work？因为attention已经把visual evidence写进residual stream了。FFN本来要read这个residual，然后write回去language prior。现在把FFN的write操作attenuate掉，visual evidence就留在residual stream里不被覆盖了。

打个比方：residual stream是一条公路，attention把"正确货物"装上车了。FFN在半路设了个检查站，要把货物换成"language prior的货物"。FADE就是把这个检查站的换货能力削弱60%，让正确货物能运到终点。

---

### 第六个问题：效果如何？

**POPE**（object hallucination binary分类）：
- LLaVA-1.5 GQA Adversarial上，FADE比greedy提升约4个点F1
- 比VCD（contrastive decoding）提升约12个点
- VCD在GQA上crash了，因为noised image破坏了spatial reasoning

**CHAIR**（captioning hallucination）：
- FADE把C_S从49.8降到46.6
- Recall只从80.6降到78.7
- VISTA虽然C_S降到19.2，但Recall暴跌到62.6（over-aggressive）

**效率**：
- FADE只比greedy慢3%（122ms vs 118ms）
- VCD慢2.4倍，VISTA慢3.9倍
- 因为FADE就是element-wise乘法，几乎零开销

---

### 第七个问题：为什么attention在hallucinated samples上也是正的？

这个我觉得是paper最interesting的发现之一。

之前大家的intuition是hallucination = attention没看对图。但这篇paper的数据显示，即使是hallucinated samples，attention也在往correct方向推（+0.8）。

这说明vision encoder + projector其实encode了正确的visual information，attention也正确地把它搬到了answer position。问题纯粹出在FFN的language prior太强，把正确的visual signal给overwritten了。

这个发现对未来的work有implication：与其改attention（像PAI那样amplify visual token权重），不如直接target FFN。

---

### 第八个问题：为什么是Layers 16-22？

这个paper没给theoretical explanation，是empirical发现的。但可以推测：

- Early layers（1-15）：在做low-level feature processing，language prior还没form
- Mid-late layers（16-22）：factual/linguistic knowledge开始dominate，这是Geva et al. 2021发现的FFN knowledge storage的主要区域
- Last layers（23-32）：更多是task-specific的refinement

在13B model（40 layers）上，critical layer是34，proportional position一样（34/40 ≈ 18/32）。说明是relative depth决定的，不是absolute layer index。

---

### 第九个问题：方法的局限

1. **Layer是固定的**：每个model选定一个layer，不能per-sample adaptive。理想情况下应该detect哪一层的FFN在misbehave，动态选layer。

2. **Task-specific α差很大**：POPE用α=0.6，MME用α=0.02，差了30倍。说明不同task对FFN modification的sensitivity差很多。实际deploy需要per-task tuning。

3. **POPE上绝对提升小**：+0.3% F1，statistically significant但practically marginal。不过CHAIR和MME上提升更明显。

4. **Stronger models上提升递减**：Qwen3-VL-8B上FADE几乎和greedy持平。因为stronger models的internal priors更optimized，留给training-free intervention的空间更小。

---

### 第十个问题：这个工作的大图景意义

这篇paper的价值在于它示范了一种paradigm：

**Understand the mechanism → Targeted intervention**

而不是blindly在output level做contrastive decoding或者在attention level做re-weighting。

这跟你Karpathy一直在推的"microscope vs telescope"思路一脉相承。先理解model在干什么，再针对性fix。

后续值得探索的方向：

1. **Neuron-level intervention**：FFN里有几千个neuron，哪些neuron在store有害的language prior？用Sparse Autoencoder (https://transformer-circuits.pub/2023/monosemantic-features/) 可以decompose出interpretable features，然后只suppress那些导致hallucination的feature。

2. **Training-time fix**：与其inference时attenuate，不如training时regularize FFN别学那么多spurious co-occurrence。类似ROME/MEMIT (https://arxiv.org/abs/2210.07229) 的思路但是用于hallucination prevention。

3. **Adaptive layer selection**：检测哪一层的FFN output direction和attention output direction diverge最大，动态选那一层做attenuation。

4. **MoE adaptation**：MoE架构里FFN被拆成多个expert。某些expert可能专门store language priors。可以只suppress那些expert的输出。

---

## 最后总结

这篇paper用一个很clean的mechanistic analysis回答了一个重要问题：LVLM hallucination的root cause是FFN在mid-late layers的language prior override了visual evidence。然后提出了一个极简的fix：在那几层把FFN输出衰减掉。

方法简单，story clean，实验充分。虽然绝对提升不算大，但mechanistic insight对后续工作有启发性。这是"interpretability-driven method design"的一个good example。

---

# FADE: 从Mechanistic Interpretability角度理解LVLM Hallucination

Karpathy你好,这篇paper其实直接接续了你当年在Tesla做interpretability的那条线——把transformer当成circuit来reverse engineer。这里作者做的核心事情是:用residual stream + logit lens的toolkit,把LVLM的hallucination溯源到了FFN这个具体component上。

---

## 1. 核心Intuition:Prediction Drift现象

先build最关键的intuition。看Figure 3那幅图,横轴是layer index,纵轴是P(Correct Answer),用logit lens在每层做projection得到的probability trajectory。

- **Green (correct samples)**:概率轨迹在中间层已经爬到高位,之后稳定保持
- **Red dashed (hallucinated samples)**:中间层 *同样* 爬到高位,但是在layers 16-22这个区间,概率开始drift下来,最终输出错误token

这个drift现象是整个paper的起点。作者提出的问题非常sharp:既然中间层attention已经把visual evidence聚合上来了,P(correct)已经高了,为什么最后会drift到错误答案?一定有某个component在layers 16-22主动把prediction往错误方向推。

这个观察让我想到你在CS231n里讲的"high-dimensional interpolation"的概念——中间层representation其实是接近正确的,但是后续的非线性变换把decision boundary推过去了。

---

## 2. Mechanistic Decomposition:Residual Stream视角

### 2.1 公式分解

Paper用Elhage et al. 2021 (https://transformer-circuits.pub/2021/framework/index.html) 的residual stream framework。每个layer:

$$\tilde{\mathbf{h}}^{(l)} = \mathbf{h}^{(l)} + \text{Attn}^{(l)}(\mathbf{h}^{(l)}) \tag{1}$$

$$\mathbf{h}^{(l+1)} = \tilde{\mathbf{h}}^{(l)} + \text{FFN}^{(l)}(\tilde{\mathbf{h}}^{(l)}) \tag{2}$$

变量解释:
- $\mathbf{h}^{(l)}$: layer $l$ 的输入hidden state(进入attention之前的residual stream)
- $\tilde{\mathbf{h}}^{(l)}$: post-attention hidden state,即attention输出加回residual之后的状态
- 上标$(l)$: 表示第$l$层
- $\text{Attn}^{(l)}$: 第$l$层的attention module输出(已经经过output projection)
- $\text{FFN}^{(l)}$: 第$l$层的FFN module输出

这里的关键insight来自Anthropic的circuit analysis:residual stream是信息的"高速公路",attention和FFN是"read-write"到这条高速公路上的两个操作。

- **Attention**: across positions的信息聚合(token之间互相"看"),它*搬运*信息但不*创造*新概念
- **FFN**: per-position的非线性变换,Geva et al. 2021 (https://aclanthology.org/2021.emnlp-main.446/) 证明FFN本质是key-value memory,存储factual knowledge

### 2.2 为什么这个分解对hallucination分析很重要

LVLM的输入是[vision tokens; text tokens]的拼接。Vision tokens经过vision encoder + projector后被送入LLM的residual stream。所以理论上,visual evidence应该在residual stream里,attention应该负责把它"搬"到answer token的位置上。

如果hallucination发生,要么是:
1. Attention没有正确搬运visual evidence
2. FFN在某个位置注入了language prior,把visual evidence"overwritten"了

Paper通过实验证明是后者。

---

## 3. Differential Logit Lens:量化每个Component的贡献

### 3.1 方法设计

作者用了一个叫"differential logit lens"的方法。核心公式:

$$\Delta_{\text{Attn}}^{(l)}(t) = \text{LM}_{\text{head}}(\tilde{\mathbf{h}}^{(l)})_t - \text{LM}_{\text{head}}(\mathbf{h}^{(l)})_t \tag{3}$$

变量解释:
- $t$: target token(我们关心的那个token,比如correct answer "yes"或某个object name)
- $\text{LM}_{\text{head}}(\cdot)_t$: 把hidden state过unembedding matrix得到token $t$的logit
- $\tilde{\mathbf{h}}^{(l)}$: post-attention state(已经加了attention贡献)
- $\mathbf{h}^{(l)}$: pre-attention state
- $\Delta_{\text{Attn}}^{(l)}(t)$: attention在第$l$层对token $t$的logit贡献

同理定义 $\Delta_{\text{FFN}}^{(l)}(t)$,就是 $\text{LM}_{\text{head}}(\mathbf{h}^{(l+1)})_t - \text{LM}_{\text{head}}(\tilde{\mathbf{h}}^{(l)})_t$。

为什么用differential而不是直接projection?因为LayerNorm是非线性的,如果直接用 $\text{LM}_{\text{head}}(\mathbf{h}^{(l)})_t$ 作为"pre-attention contribution",会把LayerNorm的效应混淆进去。差分掉了公共的LayerNorm影响,更干净地隔离出attention/FFN的纯贡献。

这个技巧参考了Geva et al. 2022 (https://aclanthology.org/2022.emnlp-main.118/) 和Belrose et al. 2023 (https://arxiv.org/abs/2303.08112) 的tuned lens工作。

### 3.2 Correct-Direction Metric:跨样本比较的关键

直接比较 $\Delta$ 值有困难:不同样本的ground truth不同,有的样本correct answer是"yes",有的是"no",有的是具体object name。不能简单平均。

作者定义:

$$C^{(l)} = \Delta^{(l)}(t_{\text{correct}}) - \Delta^{(l)}(t_{\text{incorrect}}) \tag{4}$$

- $t_{\text{correct}}$: ground truth token
- $t_{\text{incorrect}}$: 错误的token(对binary classification就是相反的那个)
- $C^{(l)} > 0$: 这个component push prediction toward correct
- $C^{(l)} < 0$: push toward wrong

这个metric把所有样本"对齐"到同一个方向,可以平均了。本质上就是让"correct方向"成为一个统一的坐标系。

---

## 4. 两个核心发现:Table 1解读

在LLaVA-1.5-7B上,50个POPE-Adversarial样本,40个correct,10个wrong(hallucinated)。

| Prediction | Attn | FFN_total | FFN_{16-22} | FFN_{L18} |
|---|---|---|---|---|
| Correct (n=40) | +1.2 | +1.7 | **+8.4** | +6.0 |
| Wrong (n=10) | +0.8 | -2.0 | **-3.5** | -2.4 |

### OBS-1: Attention Aggregates Visual Evidence (Always Good)

Attention的贡献在correct samples上是+1.2,在wrong samples上是+0.8。**两者都是正的**。这意味着即使hallucinated samples,attention也在把prediction往correct方向推。

这个发现颠覆了一个常见直觉。很多人以为hallucination是attention没"看对"图像。但数据显示attention在hallucinated samples上同样在工作,只是强度稍弱(+0.8 vs +1.2)。

### OBS-2: FFN at Critical Layers is the Culprit

FFN_{16-22}的对比才是震撼的:
- Correct samples: **+8.4** (强力reinforce correct prediction)
- Wrong samples: **-3.5** (强力push toward wrong answer)

这个contrast巨大。在correct case,FFN在layers 16-22是"好人",帮attention巩固correct prediction。但在hallucinated case,FFN变成了"坏人",主动把prediction往错误方向推。

更细看Layer 18单独:+6.0 vs -2.4,差距8.4个logit单位,这一个layer就贡献了主要的divergence。

### 为什么是Layers 16-22?

这个layer range在32-layer LLaMA-7B里属于mid-to-late layers。参考Geva et al. 2021的发现:FFN在mid-late layers存储了大量factual/linguistic priors。在LVLM里,这些priors表现为"看到厨房就联想冰箱"、"看到滑雪就联想狗"这类co-occurrence统计。

当visual evidence比较弱或ambiguous时,这些priors会dominate,overwritten掉attention搬来的visual signal。

---

## 5. FADE:方法极其简单但有效

### 5.1 公式

基于上述分析,作者提出的intervention极其简单:

$$\mathbf{h}^{(l+1)} = \tilde{\mathbf{h}}^{(l)} + (1 - \alpha) \cdot \text{FFN}^{(l)}(\tilde{\mathbf{h}}^{(l)}) \tag{5}$$

- $\tilde{\mathbf{h}}^{(l)}$: post-attention hidden state
- $\text{FFN}^{(l)}(\tilde{\mathbf{h}}^{(l)})$: 原始FFN输出
- $\alpha \in [0, 1]$: attenuation strength
- $(1-\alpha)$: 保留比例

当 $\alpha = 0$: 退化为标准transformer
当 $\alpha = 1$: 完全去掉这一层的FFN(residual-only,只保留attention)

### 5.2 为什么这个方法work

intuition是这样的:attention已经把visual evidence搬到了answer position的residual stream里。FFN在layers 16-22会"读取"这个residual,然后基于它store的language priors做变换。

如果visual evidence强(例如图像里明确有猫),FFN的prior也align,两者合力推correct。
如果visual evidence弱(例如adversarial setting里,问"有没有冰箱"但图像是厨房),FFN的prior就override了attention的weak signal。

FADE的做法:在critical layers把FFN的"write"操作衰减掉,保留attention"write"的visual evidence在residual stream里不被overwrite。

这就像在信息高速公路上设置了一个"language prior减速带",让visual evidence能更顺畅地传到final unembedding。

### 5.3 与Contrastive Decoding的对比

VCD (https://openaccess.thecvf.com/content/CVPR2024/papers/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_Through_Visual_Contrastive_CVPR_2024_paper.pdf) 的做法是:跑两次forward(一次原图,一次noised image),然后contrast两个output distribution:

$$p_{\text{VCD}}(y|x) \propto (1+\beta) p(y|x_{\text{orig}}) - \beta p(y|x_{\text{noise}})$$

问题是:需要2次forward pass,而且contrast在output level,不targeted到具体component。FADE只1次forward,在FFN输出上做element-wise scaling,overhead极小。

---

## 6. 实验结果深度解读

### 6.1 POPE Benchmark (Table 2)

POPE测object hallucination,三种sampling:
- Random: 正负样本1:1随机
- Popular: 偏向常见物体(更难,因为prior强)
- Adversarial: 专门选容易误判的(最难)

关键数据点(LLaVA-1.5-7B, GQA Adversarial):
- Greedy: F1 = 80.4
- VCD: F1 = 72.0 (比greedy还差!因为noised image破坏了spatial reasoning)
- DAMO: F1 = 81.9
- VISTA: F1 = 84.0
- **FADE: F1 = 84.2** (最好)

VCD在GQA上crash是很有意思的发现。GQA的问题依赖scene graph reasoning(空间关系、属性),VCD用Gaussian noise扰动image会破坏这些fine-grained spatial cues,导致contrastive signal本身就很noisy。

### 6.2 CHAIR Benchmark (Table 3)

CHAIR测image captioning里的hallucination:
- CHAIR_S: sentence-level (有多少句包含hallucinated object)
- CHAIR_I: instance-level (平均每句多少hallucinated object)
- Recall: ground truth objects的覆盖率

LLaVA-1.5上:
- Greedy: C_S=49.8, Rec=80.6
- VISTA: C_S=19.2 (很好), 但Rec=62.6 (大幅下降!)
- **FADE: C_S=46.6, Rec=78.7**

VISTA的over-aggressive suppression问题:它用steering vector强力干预,虽然压下了hallucination,但也把很多correct objects压掉了(Rec从80.6→62.6)。

FADE的trade-off更好:hallucination降了3.2个点,Rec只降2个点。

### 6.3 MME Benchmark (Table 4)

MME有10个perception subtask。LLaVA-1.5上:
- Greedy: 1505.7
- PAI: 1508.9
- **FADE: 1519.0** (+13.3)

FADE在Counting上提升最明显:155.0→160.0。Counting需要精确的object grounding,language prior(例如"通常有2-3个")容易导致误数,FFN attenuation正好压制这种prior。

### 6.4 效率对比 (Table 9)

这是FADE的另一个卖点:

| Method | Latency (ms) | vs Greedy |
|---|---|---|
| Greedy | 118 | 1.0× |
| VCD | 285 | 2.4× |
| PAI | 184 | 1.6× |
| DAMO | 150 | 1.3× |
| VISTA | 459 | 3.9× |
| **FADE** | **122** | **1.03×** |

FADE只增加3% latency,因为intervention就是element-wise multiplication。Memory footprint也和greedy一样(14.5GB),没有额外开销。

---

## 7. 消融实验的Insights

### 7.1 Strength α 的sweet spot (Table 12)

LLaVA-1.5-7B POPE,layer固定18:
- α=0.1: 86.05 (太弱)
- α=0.6: **86.31** (最优)
- α=0.8: 86.22 (开始diminishing return)

这个plateau很宽,α∈[0.5, 0.7]都在86.2附近,说明方法对hyperparameter不敏感。

### 7.2 Layer selection (Table 13)

α固定0.5:
- Layer 14: 86.20
- **Layer 18: 86.22** (最优)
- Layer 22: 85.92

Layer 18在32-layer LLaMA里大概是56%深度位置。在13B model (40 layers)里,对应的是Layer 34 (34/40 ≈ 18/32 ≈ 0.56),实验验证确实Layer 34最优。这个proportional transfer很有意思,说明critical layer是 *相对深度* 决定的,不是绝对位置。

### 7.3 Task-dependent hyperparameters (Appendix B)

这个细节值得注意:
- POPE (discriminative): α=0.6, layer 18
- CHAIR (generative): α=1.0, layer 20
- MME (diverse reasoning): α=0.02, layer 17

MME需要的α比POPE小25-35倍!说明diverse reasoning tasks对FFN modification非常sensitive,gentle intervention就够。这提示language prior的"强度"在不同task上分布不同。

---

## 8. 跨架构泛化 (Tables 6-8)

作者还在InternVL3-8B、Qwen2.5-VL-7B、Qwen3-VL-8B上测了。这些是更强的next-gen模型。

InternVL3-8B上:
- MMBench_EN: Greedy 66.15 → FADE **69.24** (+3.1)
- MME Perception: FADE **1734.6** (最高)
- POPE Adv: FADE 88.2 (tied with PAI)
- CHAIR: 所有training-free方法都比greedy差

最后一个观察很insightful:stronger models的internal language priors更optimized,aggressive intervention容易break。FADE的"温和"特性让它在这个regime下仍然competitive。

---

## 9. 为什么这个工作重要:连接到更大的Interpretability图谱

### 9.1 与Anthropic / OpenAI interpretability工作的关系

这篇paper的方法论直接建立在以下工作上:

1. **Elhage et al. 2021** (https://transformer-circuits.pub/2021/framework/index.html): "A Mathematical Framework for Transformer Circuits" — 提出residual stream as information highway,attention as information routing,FFN as per-position memory
2. **Geva et al. 2021** (https://aclanthology.org/2021.emnlp-main.446/): "Transformer Feed-Forward Layers Are Key-Value Memories" — FFN本质是key-value lookup
3. **Geva et al. 2022** (https://aclanthology.org/2022.emnlp-main.118/): "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in Vocabulary Space" — FFN在vocabulary space上直接promote/demote concepts
4. **Belrose et al. 2023** (https://arxiv.org/abs/2303.08112): "Tuned Lens" — logit lens的改进版,per-layer训练affine transform
5. **Meng et al. 2022** (https://arxiv.org/abs/2202.05262): ROME — 定位factual knowledge在FFN的特定layer

### 9.2 与Concurrent Work的对比

Paper提到两个concurrent work:
- **Neo et al. 2025** (ICLR): 分析visual token processing via attention knockouts,但不提hallucination mitigation
- **Re-DeEP** (https://openreview.net/forum?id=...): 针对RAG,target retrieval-augmented setting,需要dual intervention(attention + FFN)

FADE的差异化:在VLM setting下attention仍然reliable,只需要单component intervention(FFN only)。

### 9.3 与你Karpathy的"microscope"视角的连接

你在Tesla时期做的那些work(例如visualizing attention in self-driving,https://arxiv.org/abs/2306.12942等)本质上也是mechanistic interpretability。FADE延续了这条线:不把model当黑盒,而是decompose成attention + FFN,定位问题component,然后做targeted intervention。

---

## 10. Limitations和Future Directions

### 10.1 当前局限

1. **Layer是固定的**:每个架构选定一个critical layer,不能adaptive。实际不同样本可能需要不同layer。
2. **只在7B-8B scale验证**:30B+未测。
3. **Evaluation集中在hallucination benchmarks**:broader reasoning能力未评估。

### 10.2 值得探索的方向

1. **Adaptive layer selection**:用一些activation-based metric动态选layer。例如检测哪一层的FFN output与attention output方向divergence最大。
2. **Training-time regularization**:把FADE的思想集成到training里,让FFN在训练时就少store有害priors。这类似ROME之后的MEMIT (https://arxiv.org/abs/2210.07229) 思路。
3. **更细的component decomposition**:FFN本身可以decompose成两个linear layer + activation,甚至可以定位到具体的neuron。参考Anthropic的Sparse Autoencoder work (https://transformer-circuits.pub/2023/monosemantic-features/)。
4. **MoE architecture**:如果FFN是hallucination source,那MoE的expert选择机制会不会让这个问题更复杂?某些expert可能专门store language priors。
5. **与Chain-of-Thought的交互**:paper提到FADE与CoT compression techniques互补 (Zhang et al. 2026c https://arxiv.org/abs/...). CoT本质是在reasoning层面增加compute,FADE在representation层面修正bias,两者正交。

---

## 11. 我对这篇paper的整体评价

**Strengths**:
1. Mechanistic analysis做得扎实,differential logit lens + correct-direction metric是well-designed
2. 干预方法极简,工程友好,3% overhead
3. 跨6个benchmark、6个architecture验证,generalization evidence充分
4. 与concurrent work对比清晰,positioning准确

**Weaknesses**:
1. Critical layer的identification还是empirical的(why specifically 16-22?没有theoretical explanation)
2. POPE上improvement绝对值小(+0.3% F1),statistically significant但practically small
3. α的task-dependence (POPE 0.6 vs MME 0.02) 暴露了方法的brittleness,需要per-task tuning
4. 没有分析FFN内部的哪些neurons贡献了language prior(可以参考Geva et al. 2023的distributed $d_{kV}$ analysis)

**Overall**: 这是一篇solid的mechanistic interpretability驱动的method paper。它延续了"understand then fix"的范式,比纯engineering的contrastive decoding更有理论insight。后续如果能做到adaptive layer selection + neuron-level intervention,会更有impact。

---

## References

- Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread (2021). https://transformer-circuits.pub/2021/framework/index.html
- Geva, M. et al. "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021. https://aclanthology.org/2021.emnlp-main.446/
- Geva, M. et al. "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in Vocabulary Space." EMNLP 2022. https://aclanthology.org/2022.emnlp-main.118/
- Belrose, N. et al. "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv 2023. https://arxiv.org/abs/2303.08112
- Meng, K. et al. "Locating and Editing Factual Associations in GPT." NeurIPS 2022. https://arxiv.org/abs/2202.05262
- Leng, S. et al. "Mitigating Object Hallucusions in LVLMs Through Visual Contrastive Decoding." CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_Through_Visual_Contrastive_CVPR_2024_paper.pdf
- Anthropic. "Monosemanticity". Transformer Circuits Thread (2023). https://transformer-circuits.pub/2023/monosemantic-features/
- FADE GitHub repo. https://github.com/EasonAI-5589/LLaVA-Hallucination
