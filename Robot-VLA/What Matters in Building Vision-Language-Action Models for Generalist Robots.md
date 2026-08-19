---
source_pdf: What Matters in Building Vision-Language-Action Models for Generalist
  Robots.pdf
paper_sha256: 8c9c4f9d1cd78b5c6f106f5e81f47119bb3de1ddfda31e96dc6ada7d10ea5041
processed_at: '2026-08-13T04:10:27-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

把市面上能买到的8个VLM backbones、4种architecture搭法、3种训练recipe全部买回来
在同一个testbed(CALVIN + SimplerEnv + real robot)上跑了一遍，然后告诉哪种组合最好，哪些popular的做法其实是overrated的。

每个人搭的architecture都不一样：
- RT-2用discrete action token，当成text next-token prediction来做
- OpenVLA跟着RT-2的套路
- π0用continuous action + flow matching + MoE
- GR-1用interleaved history
- Octo也用interleaved
- RoboFlamingo用policy head

这篇文章搞了 RoboVLMs framework。跑了600+实验。

### Q1: 为啥要用VLA，不用别的?

VLA到底比传统model-free policy强在哪?强多少?
强很多。在CALVIN上比之前SOTA GR-1高出1.19 Avg.Len.，real-world 20个task × 5个setting全面碾压Octo和OpenVLA。最惊艳的是出现了self-correction行为——robot第一次抓空了会自己调整位置再试，这个behavior training data里根本没有。

VLM在web-scale图文上学到了"什么是什么"的semantic prior。比如它知道"oven handle长这样"、"drawer是这样开的"。这个prior直接迁移到manipulation，省去了大量exploration。

### Q2: 用哪个VLM backbone?

LLaVA、Qwen-VL、Flamingo、KosMos、还是PaliGemma?

KosMos-2和PaliGemma显著最好

| Backbone    | 参数量 | VL pretrain data | CALVIN Avg.Len. |
| ----------- | --- | ---------------- | --------------- |
| Flamingo 9B | 9B  | 1B+ pairs        | 1.83            |
| Qwen-VL     | 9B  | 350K             | 0.30 (崩了)       |
| KosMos-2    | 2B  | 90M              | 3.59            |
| PaliGemma   | 3B  | 10B tokens       | 3.82            |
|             |     |                  |                 |

不是model size大就好，是VL pre-training的data scale和质量 决定一切。PaliGemma在10B token上pretrain，KosMos在90M高质量pair上pretrain，都拿到了strong VL alignment。
LLaVA和Qwen-VL这种用大量visual token(256+)的backbone，在VLA setup下训练不稳定。加了perceiver resampler把token降到64-256才work。猜测是visual token太多会稀释action gradient信号。

### Q3: VLA的architecture怎么搭?

#### Q3.1: 哪种structure最好?

我该用discrete action还是continuous?要不要history?history怎么塞进去?
- **Continuous action >> Discrete action**: 差距巨大，one-step setup下KosMos continuous 4.04 vs discrete 0.44，快10倍。Discrete action的quantization error会随task horizon累积，long-horizon任务直接崩。
- **Policy head > Interleaved**: 4.49 vs 4.12。Policy head把VLM和history aggregation解耦，VLM保持原始单帧VL fusion能力，history在外部policy head处理。

想象VLM是一个"看图说话"专家。你要让它做的事是"看当前frame + 历史frame，输出action"。
- Interleaved方式相当于逼这个专家看一整本历史相册再给结论，但它训练时从来没这么干过，会confused。
- Policy head方式相当于让专家每页都看一遍并做笔记，最后由另一个decision maker综合笔记做决策。专家做它擅长的事，decision maker做它擅长的事。

#### Q3.2: 哪种structure泛化好、data效率高?
Policy head在zero-shot generalization和data efficiency上都最好。

| Architecture     | 0.1x data | 1x data | 5x data |
| ---------------- | --------- | ------- | ------- |
| Flamingo P.H. 3B | 0.13      | 4.09    | 4.21    |
| KosMos P.H.      | 2.52      | 4.49    | 4.51    |
| KosMos Inter.    | 2.49      | 4.12    | 4.46    |

Policy head在10%数据下就能达到2.52，interleaved 2.49差不多。但1x data时policy head 4.49明显领先。

**Intuition**: Policy head保留了VLM的generalization能力(history不影响VL fusion)，所以zero-shot drop小。同时history aggregation交给专门的policy head(Transformer/RNN)，data efficiency高。

#### Q3.3: 训练loss用哪个?Inference时action怎么执行?

- **Flow Matching ≈ MSE+BCE**: 差距很小。Chunk执行下FM 3.68 vs MSE 3.57(ABC setting)。Diffusion的multi-modal modeling能力在short-horizon deterministic任务上没明显优势，反而增加inference latency。
- **Chunk execution最好**: 比First和Ensemble都好。因为model学的是multi-modal trajectory distribution，只取第一个action相当于mode collapse，丢失planning信息。Chunk执行保留temporal coherence。
- **First execution最差**: ABC下Avg.Len.只有2.45 vs Chunk 3.68。

**Intuition**: 想象你规划了一条10步trajectory从A到B。
- Chunk执行: 按计划走完10步再看下一段。Planning信息保留完整。
- Ensemble: 多个chunk预测的同一时刻action做平均。会平滑掉多模态分布的mode，丢失diversity。

#### Q3.4: 要不要加MoE?
看目标。
- **目标是generalization(open-world)**: 加MoE，ABC setting下MoE 3.84 vs no MoE 3.68。
- **目标是seen场景performance**: 不加，ABCD setting下no MoE 4.10 vs MoE 3.84。

**Intuition**: MoE的设计是VL token走VLM原始FFN，action token走独立expert FFN。这保护了VL representation不被action supervision"污染"。

- Open-world场景下，你需要保留VLM的general semantic understanding，所以保护有用。
- Seen场景下，你需要模型specialize到训练分布，MoE的"保护"反而阻止了specialization。

这也解释了为啥π0用MoE——它的目标是open-world generalization。

### Q4: 大规模cross-embodiment data什么时候用?

三种recipe:
1. **Co-train**: in-domain + OXE一起训。**帮助有限**。
2. **Post-train**: 先Co-train，再in-domain finetune。**对high-frequency task有增益**。
3. **Finetune**: 只用in-domain。**baseline很强**。

关键发现:

- **OXE Co-train without post-training没啥用**。Google Robot上RT Finetune(用同机器人不同task数据)比OXE Co-train还好。
- **Post-train对pick-place类高频task有增益**(Pick Coke Can 0.98 vs Finetune 0.97)，对low-frequency task(move near、open/close drawer)反而下降。因为OXE里pick-place占比极大，post-train把高频skill prior带过来了，但low-frequency skill被"稀释"。
- **In-domain data是王道**。即使是task-agnostic的in-domain data(同机器人不同task)，也比大规模cross-embodiment data有效。
- **Few-shot setting下cross-embodiment pretrain帮助大**。CALVIN few-shot(每task 10条demo)下，pretrain提升17.2% single-task success。

**Intuition**: Cross-embodiment data提供的是"task concept prior"——什么是grasping、什么是pick-place。这种semantic prior在few-shot时有用(你只有10条demo，prior很重要)。但embodiment-specific的dynamics(运动学、控制频率、action space)还是要靠in-domain data学。同机器人的task-agnostic data为啥比cross-embodiment好?因为dynamics一致，只是task不同，模型只需要学新task的semantic，不需要重新calibrate dynamics。

---

最优配置是：

1. **Backbone**: PaliGemma 3B或KosMos-2(看资源)
2. **Architecture**: Policy-Head + Continuous action
3. **History**: Window size 16，policy head用Transformer
4. **Action execution**: Chunk size 10，chunk execution
5. **Loss**: MSE+BCE(简单有效)
6. **MoE**: 目标是open-world就加，是fixed deployment就不加
7. **Data**: Post-train recipe，先OXE co-train，再in-domain finetune
8. **Training**: AdamW，lr 1e-4，warmup 0.25 epoch，5 epoch

---

## 我对这篇paper的critique

**强的地方**:
- 600+实验的scale在robot领域很罕见，control variable做得干净
- RoboVLMs framework本身是重要贡献，统一接口让后续工作可比
- 几个negative result很有价值(cross-embodiment co-train没显著帮助挑战了RT-2以来的common practice)

**弱的地方**:
- CALVIN只有34个short-horizon tabletop task，"generalist"的claim边界有限
- KosMos和PaliGemma为啥好，没完全解耦(data scale vs architecture)
- Discrete action用256 bins可能太粗，FAST那种更精细的tokenizer可能缩小差距
- Diffusion policy"无显著优势"的结论可能premature，大数据scale下可能不一样
- 没探索action chunk size L的影响，这是个重要的missing dimension

参考link:
- Paper: https://arxiv.org/abs/2511.10176
- Code: https://github.com/Robot-VLAs/RoboVLMs
- Project page: https://robovlms.github.io
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- GR-1: https://arxiv.org/abs/2312.13139

如果Karpathy你想深入某个具体experiment的细节、或者某个finding对你自己工作的implications，随时展开。

---

# What Matters in Building Vision-Language-Action Models for Generalist Robots 深度解析

这篇paper来自ByteDance Research联合Tsinghua、BAAI、NUS等机构,第一作者Xinghang Li,通讯作者Huaping Liu、Tao Kong、Hanbo Zhang。论文系统性地回答了构建VLA过程中四个核心问题,跑了600+实验,8个VLM backbones,4种policy架构,建立了一个名叫**RoboVLMs**的统一框架。项目主页:https://robovlms.github.io ,代码:https://github.com/Robot-VLAs/RoboVLMs 。

---

## 一、为什么要做这个study?Big picture的intuition

Robot policy这个领域过去几年分裂成四大流派(paper Fig.1a):

1. **Model-Free Learning**: 把state encode成latent,用一个task-specific policy head预测action。经典做法,泛化性靠dataset coverage堆出来。
2. **Model-Based Learning**: 显式建模robot dynamics + environment affordance,适合door opening这种articulated object,但dynamics model很难学好且很难跨scene泛化。
3. **World Model Based Learning**: 预测future goal image,再用inverse dynamics model回推action。例如UniPi、SuSIE这种思路。
4. **Vision-Language-Action (VLA) Model**: Model-Free的一个special branch,把VLM当作state encoder,直接接收language instruction,输出action。

VLA路线的核心赌注是:**VLM在web-scale图文数据上学到的语义表示,可以直接迁移到robot manipulation这种physical reasoning任务上,从而获得open-world generalization**。这个赌注到底成不成立?paper用实验给了一个肯定的答案。

参考相关工作:
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- GR-1: https://arxiv.org/abs/2312.13139
- Octo: https://arxiv.org/abs/2405.12213

---

## 二、四个Essential Questions以及对应Findings

Paper把整个研究问题拆解得非常清晰(Extended Table 1):

| Essential Question | Research Question | Finding |
|---|---|---|
| Q1: Why VLAs | Q1.1 是否合适?Q1.2 real-world表现? | A1.1: promising path; A1.2: 强effectiveness + robustness |
| Q2: Which backbone | 哪种VLM更合适? | 充分VL pre-training on large-scale data受益 |
| Q3: How to formulate | Q3.1 最佳structure? Q3.2 generalization/data efficiency? Q3.3 training objective & inference? Q3.4 MoE? | continuous action + policy head + history最佳;diffusion ≈ MSE;chunk execution重要;MoE提升generalization |
| Q4: When extra data | 何时引入cross-embodiment? | in-domain data关键;post-training进一步提升 |

---

## 三、VLA的formal formulation,逐变量拆解

### 3.1 VLM的形式

公式(1):
$$\hat{l} = \text{VLM}(I, l_{\text{prompt}})$$

- $I$: 输入image
- $l_{\text{prompt}}$: text prompt(比如question或instruction)
- $\hat{l}$: 输出text token sequence

VLM训练loss公式(2):
$$\ell_{\text{VLM}} = \text{CrossEntropy}(\hat{l}, l_{\text{target}})$$

- $l_{\text{target}}$: ground truth text
- 整个VLM就是在最小化next-token prediction的CE loss

Vision processor把image变成visual tokens公式(3):
$$[\text{OBS}] = (x_1^v, \cdots, x_N^v) = \text{ViT}(I)$$

- $N$: token数量(对应paper里不同backbone的64/256/576)
- $x_i^v$: 第$i$个visual token的embedding
- 上标$v$表示visual modality,区分于language token

### 3.2 VLA的形式

公式(4):
$$a_{t:t+L-1} = \text{VLA}(o_{t-H+1:t}, l_{\text{prompt}})$$

- $t$: 当前time step
- $H$: history observation长度(paper里常用16或8)
- $L$: action chunk长度(常用10)
- $o_{t-H+1:t}$: 从$t-H+1$到$t$时刻的observation序列
- $a_{t:t+L-1}$: 预测的连续action chunk,每个action是7-dim(6-DoF pose + 1 gripper open/close)

observation $o_t = (s_t, I_t)$,$s_t$是proprioceptive state(joint angles等),$I_t$是image(可来自third-view或wrist camera或两者)。

### 3.3 Action Pre-process:两步走的normalization

**Action Clamping** 公式(5):
$$a^{i\prime} = \min(a^{i}_{99\text{th}}, \max(a^{i}_{1\text{st}}, a^i))$$

- $a^i$: 第$i$维原始action
- $a^{i}_{1\text{st}}$、$a^{i}_{99\text{th}}$: 训练数据中第$i$维action的1%和99%分位数
- 这步是为了去掉outlier,避免extreme值拉偏分布

**Action Normalization** 公式(6):
$$\tilde{a}^i = 2 \times \frac{a^{i\prime} - a^{i}_{1\text{st}}}{a^{i}_{99\text{th}} - a^{i}_{1\text{st}}} - 1$$

- $\tilde{a}^i$: normalized action,落在$[-1, 1]$区间
- 最后第7维gripper status $\in \{-1, 1\}$
- inference时需要reverse这个映射恢复成实际action

**Action Discretization**: 每维均匀划分256个bin,把$\tilde{a}$变成7个discrete token $\in [0, 255]$。再加offset(默认10)避开language tokenizer的special token位置。

### 3.4 Action Prediction Losses

**Continuous action loss** 公式(7):
$$l_{\text{VLA}} = \sum_{i=t}^{t+L-1} \text{MSE}(\hat{a}_{i,\text{pose}}, \tilde{a}_{i,\text{pose}}) + \lambda \cdot \text{BCE}(\hat{a}_{i,\text{gripper}}, \tilde{a}_{i,\text{gripper}})$$

- $\hat{a}_{i,\text{pose}}$: 预测的前6维pose
- $\tilde{a}_{i,\text{pose}}$: ground truth前6维
- $\hat{a}_{i,\text{gripper}}$: 预测的gripper state(用sigmoid输出概率)
- $\tilde{a}_{i,\text{gripper}}$: ground truth gripper state(0或1)
- $\lambda$: 平衡weight(默认1)
- MSE处理连续的pose regression,BCE处理binary的gripper开关

**Discrete action loss** 公式(8):
$$l_{\text{VLA}} = \sum_{i=t}^{t+L-1} \sum_{j=1}^{7} \text{CE}([\text{ACT}]_i^j, \tilde{a}_i^j)$$

- $[\text{ACT}]_i^j$: 第$i$时刻第$j$维预测的bin index
- $\tilde{a}_i^j$: ground truth bin index
- 跟VLM next-token prediction的CE loss完全一样

---

## 四、四种VLA Structure的详细架构

Paper最关键的分类是按"history如何聚合 × action space类型"分四类(Extended Fig. 2b):

### 4.1 One-Step Models

只用当前$H=1$的observation。公式(9):
$$\hat{a}_{t:t+L-1} = \text{VLA}(o_t, l_{\text{prompt}})$$

**Continuous variant** 公式(10):
$$\Delta^{[\text{LRN}]} = \text{VLM}(o_t, l_{\text{prompt}}), \quad \hat{a}_{t:t+L-1} = \text{MLP}(\Delta^{[\text{LRN}]})$$

- $[\text{LRN}]$: learnable token,插入到VLM token sequence末尾
- $\Delta^{[\text{LRN}]}$: VLM forward之后该token位置的hidden state
- VLM backbone可以是encoder-decoder(cross-attention融合)也可以是decoder-only(self-attention拼接)
- 代表作: ACT, BC-Z, MVP, R3M, VIMA, 3D Diffuser, RoboMamba, **π0**

**Discrete variant** 公式(11):
$$[\text{ACT}]_{t:t+L-1}^{1:7} = \text{VLM}(o_t, l_{\text{prompt}})$$

- 直接走VLM原本的next-token prediction,action token当成text token生成
- 代表作: RT-1, RT-2, 3D-VLA, LAPA, OpenVLA, Embodied-CoT

### 4.2 Interleaved-Continuous-Action Models

History以interleaved方式喂入,公式(12):
$$O_t = ([\text{OBS}]_{t-H+1}, [\text{LRN}]), \ldots, ([\text{OBS}]_t, [\text{LRN}])$$

- $O_t$: 拼成的token sequence,observation token和learnable action token交替排列
- 每个$[\text{LRN}]$被复制H次插入

公式(13):
$$[\text{LRN}]_{t-H+1:t} = \text{VLM}(O_t), \quad \hat{a}_{t:t+L-1} = \text{MLP}([\text{LRN}]_t)$$

- 只用最后一个$[\text{LRN}]_t$位置的hidden state去predict当前action chunk
- Inference时滑窗推进,每次加上new observation

代表作: GR-1, OCTO, GR-2。

**关键问题**: interleaved方式下,VLM在处理history时被强制进入"非原始预训练格式",可能损害VL fusion能力;而且训练/inference时FLOPs和memory随H线性增长。

### 4.3 Policy-Head-Continuous-Action Models

VLM只处理单步observation,history通过外部policy head聚合。公式(14)、(15):
$$o_t = ([\text{OBS}]_t, [\text{LRN}])$$
$$[\text{LRN}]_t = \text{VLM}(o_t, l_{\text{prompt}})$$
$$a_{t:t+L-1} = h([\overline{\text{LRN}}]_{t-H+1}, \ldots, [\overline{\text{LRN}}]_t)$$

- $h$: policy head,可以是RNN(LSTM/GRU)、Transformer、或Diffusion model
- $[\overline{\text{LRN}}]_{t-H+1}, \ldots, [\overline{\text{LRN}}]_t$: 过去H个时刻VLM输出的learnable token hidden states
- VLM每次只跑单步forward,history modeling完全交给policy head

代表作: RoboFlamingo, RoboUniview, DeerVLA。

**为什么policy head更好?** 核心直觉:VLM的vision-language fusion机制是为单帧image + text设计的,把它强行当成sequence model处理history token会破坏pre-training带来的fusion prior。Policy head把"VL理解"和"temporal aggregation"解耦,各司其职。

---

## 五、核心实验结果逐表精读

### 5.1 Backbone对比 (Extended Table 5)

| Backbone | #Token | Data Scale | Model Size | Avg. Len. |
|---|---|---|---|---|
| Flamingo 3B | 64 | 1B+ | 3B | 1.57 |
| Flamingo 4B | 64 | 1B+ | 4B | 1.71 |
| Flamingo 9B | 64 | 1B+ | 9B | 1.83 |
| Qwen-VL | 256 | 350K | 9B | 0.30 |
| MoonDream | 576 | UNK | 3B | 1.81 |
| UForm | 256 | 10M | 1.3B | 2.28 |
| KosMos | 64 | 90M | 2B | 3.59 |
| PaliGemma | 256 | 10B | 3B | 3.82 |

**关键发现**: KosMos和PaliGemma显著领先。KosMos虽然只有2B但达到了3.59 Avg.Len.,PaliGemma 3B达到3.82。这两个backbone的共同点是**充分的VL pre-training**(KosMos在90M图文对上pretrain,PaliGemma在10B token上pretrain)。

**有意思的失败案例**: Qwen-VL(9B)和LLaVA在原始setup下表现非常差(Qwen-VL只有0.30 Avg.Len.),作者在Appendix里发现加了perceiver resampler做token downsampling之后才能拿到合理性能。这暗示vision token数量过多时,VLA训练会unstable。猜测原因:大量visual token会让policy gradient信号被稀释,而pre-trained VLM的attention pattern本来是为caption/VQA任务优化的,不一定适应action预测所需的spatial precision。

### 5.2 VLA Structure对比 (Table I)

以KosMos backbone为例:

| Structure | Action Space | Avg.Len. (ABCD→D) |
|---|---|---|
| One-Step | Disc. | 0.44 |
| One-Step | Cont. | 4.04 |
| Interleaved | Cont. | 4.12 |
| Policy-Head | Cont. | **4.49** |

三个insight:

1. **Continuous >> Discrete**: 同样one-step,continuous(4.04)几乎是discrete(0.44)的10倍。原因是discrete action的quantization error会随horizon累积,长horizon任务compounding error非常严重。看1-task success rate差距不大(0.316 vs 0.933),但5-task success rate差距巨大(0.001 vs 0.688)。

2. **History > One-step**: 加history的interleaved和policy head都比one-step高(4.12和4.49 vs 4.04)。Partial observability下,历史信息能补全当前state不能反映的dynamics。

3. **Policy head > Interleaved**: 4.49 vs 4.12。Policy head保留了VLM原始的VL fusion能力,history aggregation在VLM之外做,效率更高。

### 5.3 Training Objective和Execution Paradigm (Table IIa)

用PaliGemma backbone(对齐π0 setting):

**关键观察**:

- **Flow Matching ≈ MSE+BCE**: Chunk execution下,FM(ABC: 3.68, ABCD: 4.09) vs MSE+BCE(ABC: 3.57, ABCD: 4.04)。差距很小。Diffusion-based policy在短horizon deterministic任务上不一定有优势,反而增加inference latency。

- **Chunk > Ensemble > First**: Chunk执行最好,因为model学的是多模态trajectory distribution,只取第一个action相当于mode collapse,丢失了planning信息。Ensemble通过加权平均历史chunk对应位置的预测,可以增加temporal consistency,但仍然不如直接chunk执行。

- **First最差**: ABC setting下First execution Avg.Len.只有2.45(vs Chunk 3.68)。缺乏temporal coherence。

### 5.4 MoE结构的效果 (Table IIb)

| Training | MoE | ABC Avg.Len. | ABCD Avg.Len. |
|---|---|---|---|
| MSE+BCE | √ | 3.69 | 3.46 |
| MSE+BCE | × | 3.57 | 4.04 |
| FM | √ | 3.84 | 3.84 |
| FM | × | 3.68 | 4.10 |

**MoE在unseen场景(ABC)帮助大,在seen场景(ABCD)反而有害**。

MoE的设计(Extended Fig. 2a): vision-language token走原始VLM FFN,action token走独立的action expert FFN,两者通过self-attention交互。这保护了VL representation不被action supervision干扰,从而保留generalization。但在seen场景下,模型需要specialize到训练分布,MoE的"保护"反而成了束缚。

这也解释了π0为什么用MoE:它的目标是open-world generalization,不是单一task的SOTA。

### 5.5 Cross-embodiment data的时机 (Extended Fig. 4)

三种recipe:
1. **Co-train**: VLA同时训练in-domain + OXE
2. **Post-train**: 先Co-train,再在in-domain上finetune
3. **Finetune**: 直接在in-domain上训练

**核心发现**:

- Co-train without post-train帮助有限。Bridge上OXE Co-train和Bridge Finetune接近;Google Robot上RT Finetune(用同机器人不同task数据)比OXE Co-train更好。
- Post-train在某些high-frequency task上有增益。Pick Coke Can上Post-train达到0.98,比Finetune(0.97)略高。但Move Near、Open/Close Drawer上反而下降,因为这些task在OXE中frequency较低。
- **In-domain data是王道**: 即使是task-agnostic的in-domain data(比如RT Finetune用了所有RT数据包括非测试task),也比大规模cross-embodiment data有效。

直觉:cross-embodiment data提供了"task concept"的prior(grasping、pick-place这种high-frequency skill),但embodiment-specific的dynamics(运动学、控制频率、动作空间)还是要靠in-domain data学习。

### 5.6 VL Pre-training的effect (Supplementary Table 2)

| Architecture | VL Pretrain | ABCD Avg.Len. |
|---|---|---|
| KosMos Inter. | No | 1.38 |
| KosMos Inter. | Yes | 4.12 |
| KosMos P.H. | No | 2.51 |
| KosMos P.H. | Yes | 4.49 |

**VL pretrain至关重要**: 没有VL pretrain的KosMos Inter.只有1.38,有pretrain的达到4.12,3倍提升。

---

## 六、CALVIN benchmark的SOTA结果 (Extended Table 2)

ABC→D(zero-shot generalization):
| Method | VLA? | Avg.Len. |
|---|---|---|
| MCIL | × | 0.31 |
| RT-1 | × | 0.90 |
| HULC | × | 0.67 |
| GR-1 | √ | 3.06 |
| **KosMos P.H. (RoboVLMs)** | √ | **4.25** |

ABCD→D(seen scene):
| Method | Avg.Len. |
|---|---|
| HULC | 3.06 |
| GR-1 | 4.21 |
| **KosMos P.H. (RoboVLMs)** | **4.49** |

比之前SOTA GR-1在zero-shot setting高出1.19 Avg.Len.,这是个非常显著的提升。

---

## 七、Real-world实验 (Fig. 4d)

20个task × 5个setting(Simple、Unseen Distractor、Unseen Background、Unseen Target Object、Novel Skill Description)。

模型对比:Octo-Base、OpenVLA、KosMos P.H. (RoboVLMs)。

**KosMos P.H.在所有setting下都最好**,尤其在Unseen Background上优势明显。最intriguing的是paper提到KosMos P.H.展现出**self-correction ability**(Extended Fig. 1): robot第一次尝试没抓到oven handle,会自己调整end-effector位置再尝试。这种behavior在training data中不存在,是emergent property。

---

## 八、训练细节 (Appendix B, Supplementary Table 1)

- **Hardware**: 4×8 A100-80G GPUs(32卡)
- **Batch size**: 128( CALVIN ), 1024/128( SimplerEnv Co-train/Post-train )
- **Optimizer**: AdamW
- **Learning rate**: 1e-4(CALVIN), 2e-5(SimplerEnv Co-train), 5e-5(Post-train)
- **Warmup**: 0.25 epoch
- **Window size**: 16(CALVIN), 1(SimplerEnv对齐π0 setting)
- **Chunk size**: 10
- **Train epochs**: 5(CALVIN), 20K iters(SimplerEnv Co-train)

Checkpoint selection是个tricky问题:validation loss和rollout success rate不一定correlate,因为long-horizon rollout的compounding error让offline metric失效。作者采用**fixed training schedule + final checkpoint**策略保证fair comparison。

---

## 九、Limitations和Future Work

作者自己承认:

1. **Action tokenization探索不足**: VQ-VAE、VQGAN、FAST action tokenizer没有充分研究。
2. **VLM backbones有限**: 只覆盖8个,可以扩展更多(VILA、InternVL、CogVLM等)。
3. **Real-time deployment挑战**: 大模型跑30Hz控制仍然困难。

Future direction: long-horizon complex task(e.g. "make breakfast")、step-by-step reasoning、physical interaction generation。

---

## 十、个人对paper的critical reading

**贡献亮点**:
1. **系统性empirical study**: 不追求单点SOTA,而是建立fair comparison framework(RoboVLMs),600+实验覆盖四个维度。
2. **Important negative result**: cross-embodiment co-training without post-training帮助有限,挑战了RT-2、OpenVLA以来的common practice。
3. **Architectural insight**: policy head优于interleaved,验证了"保护VLM原始fusion prior"这个直觉。
4. **MoE conditional benefit**: MoE只对unseen场景有帮助,seen场景反而有害,这对future VLA design有重要指导意义。

**可能的问题**:
1. **CALVIN本身可能太"友好"**: 34个task都是tabletop short-horizon manipulation,limit了"generalist"的claim边界。Real-world的105 task虽然更多,但evaluation只覆盖20 task × 5 setting。
2. **KosMos和PaliGemma的优势来源不清晰**: 是因为pre-training data scale大,还是architecture(decoder-only + native VL fusion)更优?Paper没有做control experiment解耦这两个factor。
3. **Continuous vs Discrete的对比可能不公平**: Discrete action用了256 bins,这是相对粗的量化;如果用更精细的tokenization(如FAST的1D lookup table或VQ-VAE codebook)可能差距会缩小。
4. **Diffusion policy的"无显著优势"结论需要谨慎**: Flow Matching在ABC setting下3.68 vs MSE的3.57,差距不大但确实存在。可能在小数据规模下优势不明显,大数据scale下情况会变。
5. **No investigation of action chunk size**: Chunk size固定10,没研究L的影响。Larger chunk可能planning更long但execution更stale,smaller chunk反之。

参考一些相关工作可能进一步扩展这个study:
- FAST: https://arxiv.org/abs/2510.21228 (action tokenizer)
- CogACT: https://arxiv.org/abs/2411.19650 (discrete action with VQ)
- π0.5: https://arxiv.org/abs/2504.16054 (open-world VLA)
- RDT-1B: https://arxiv.org/abs/2410.07864 (bimanual VLA with diffusion)

---

## 十一、实操take-away:如果我要build一个VLA

基于这篇paper的findings,如果今天我要build一个production-grade VLA,我会这样选:

1. **Backbone**: PaliGemma 3B(如果资源紧)或KosMos-2(轻量优先);LLaVA/Qwen-VL需要加perceiver resampler downsampling到64-256 token。
2. **Structure**: Policy-Head-Continuous,VLM处理单帧,policy head用Transformer做history aggregation。
3. **History**: Window size 16,chunk size 10,Chunk execution。
4. **Training**: MSE+BCE loss(简单且效果好),AdamW,lr 1e-4,warmup 0.25 epoch。
5. **MoE**: 如果目标是open-world generalization,加上MoE;如果是固定deployment场景,不用。
6. **Data strategy**: Post-train recipe,先在OXE上co-train获得skill prior,再在in-domain data上finetune specialization。

---

这篇paper的价值在于它建立了一个systematic empirical baseline,后续VLA工作可以在这个framework下做targeted ablation。RoboVLMs framework本身也是重要贡献:统一接口让任意VLM插上就能变VLA。如果Karpathy你想深入聊某个具体experiment的setup或者某个finding的implications,我可以继续展开。
